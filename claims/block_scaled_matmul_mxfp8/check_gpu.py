"""GPU check of the prepared claim `Claims.BlockScaledMxfp8` (Claim.lean).

Runs the kernel's own host (`source/10-block-scaled-matmul.py`, imported by path):
`initialize_block_scaled(M, N, K, fmt)` builds the tensors and descriptors for `fmt = "mxfp8"`
(`Inst.mixed = false`) or `"mixed"` (`Inst.mixed = true`), and `block_scaled_matmul(…, torch.float16,
…)` launches the kernel, on the launches of SETUP.md and on launches at the edges of `Inst.ok` and
`Pre` (for those, the host's tensors are overwritten in place with other in-domain values before
the launch; the descriptors are the host's). At `M = N = 8192` with `K ≥ 8192` the inputs are built
by `host_like`, the host builder's steps in slices of rows (same distributions, layout and
descriptors), to stay under 4 GB (the host's own builder needs about 4.2 GB at `K = 8192`); the
launch is still the host's `block_scaled_matmul`.

For every element `(i, j)` of `c` it computes, in float64 from the input bytes (not from what the
kernel reads), exactly as `Post` states it:
* the elements of `a` (`e4m3` bytes) and of `b` (`e4m3` bytes, or for `"mixed"` the nibble `k % 2`
  of byte `j·(K / 2) + k / 2`, low nibble first, read as `e2m1`: `xbBits`); the scale byte of
  `a`'s element `(i, k)` at `sIdx K i k` of `a_scale` (and of `b`'s at `sIdx K j k` of `b_scale`),
  `sIdx K r k = (r // 128·(K // 128) + k // 128)·512 + 16·(r % 32) + 4·((r % 128) // 32)
  + (k % 128) // 32`; the scaled values `x·2^(s − 127)`;
* the products' partial sums step by step (`partSum`, `chunkAbs`), the tolerance by the recursion
  `tol(t + 1) = tol(t) + ρ·(|partSum(t)| + tol(t) + chunkAbs(t)) + α`, `ρ = 130·2^-22`,
  `α = 130·2^-126`, and `S = partSum(K / 128)`;
* `Pre`'s value conditions (`InDom` for every element, the room `Σ|p| + tol ≤ 2^127`), which every
  case must meet;
* `Post`: when `|S| + tol ≤ 65504`, `c` finite and `|c − S| ≤ tol + (|S| + tol)·2^-11 + 2^-25`
  (the ratio reported); always, `c` the `float16` rounding of some `v` with `|v − S| ≤ tol` (for a
  finite `c`: `|c − S| ≤ tol + ulp16(c)/2`; for `±∞`: `±(S ± tol) ≥ 65520`).
The frame: the inputs are compared byte for byte (`torch.equal`) with full copies taken before the
launch, and `c` (the host's `torch.empty((M, N))`) is placed between two guard regions filled with
a sentinel, which must keep their bits; `c` itself starts as the sentinel (a NaN), so every element
must have been written.

At a small size the float64 reference is compared with the host's own reference
(`compute_reference=True`, lines 406–418), for both formats. Negative tests: wrong outputs computed
from the same inputs (all zero; one group of 32 products left out; one group's scale doubled; the
scale layout with the indices `(r % 128) // 32` and `(k % 128) // 32` exchanged; for `"mixed"`, the nibbles of each byte
swapped; the output of the neighbouring column) must each break `Post`.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py [--quick]
"""

import importlib.util
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True   # keep source/ free of __pycache__ (-I ignores PYTHONDONTWRITEBYTECODE)
SRC = os.path.join(HERE, "source", "10-block-scaled-matmul.py")
spec = importlib.util.spec_from_file_location("bsm_src", SRC)
bsm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsm)

DEV = "cuda"
RHO = 130.0 / 2 ** 22          # Rules.Num.relDot16 128
ALPHA = 130.0 / 2 ** 126       # Rules.Num.absDot16 128
U16 = 2.0 ** -11               # Format.binary16.u
ETA16 = 2.0 ** -25             # Format.binary16.eta
MAX16 = 65504.0
BF16_MAX = float(torch.finfo(torch.bfloat16).max)
SENT = 0x7E5A                  # a float16 NaN pattern


E2M1 = None                    # the eight e2m1 magnitudes, set in main()


def sidx(K, r, k, swap=False):
    """The claim's sIdx K r k, on integer tensors (`swap`: a wrong layout for the negative tests,
    the indices `(r % 128) // 32` and `(k % 128) // 32` exchanged)."""
    if swap:
        return (r // 128 * (K // 128) + k // 128) * 512 + 16 * (r % 32) + 4 * ((k % 128) // 32) + (r % 128) // 32
    return (r // 128 * (K // 128) + k // 128) * 512 + 16 * (r % 32) + 4 * ((r % 128) // 32) + (k % 128) // 32


def elems(x, mixed, r0, r1, k0, k1, nib_swap=False):
    """float64 values of rows r0..r1-1, columns k0..k1-1 of a matrix: `e4m3` bytes, or (mixed)
    `e2m1` packed two per byte along K, element k the low nibble of byte k // 2 when k is even
    (the claim's `xbBits`; `nib_swap`: the wrong order, for the negative tests)."""
    if not mixed:
        return x[r0:r1, k0:k1].to(torch.float64)
    byt = x[r0:r1, k0 // 2:k1 // 2].to(torch.int64)
    lo, hi = byt & 15, byt >> 4
    if nib_swap:
        lo, hi = hi, lo
    nib = torch.stack((lo, hi), dim=-1).reshape(r1 - r0, k1 - k0)
    mag = E2M1[nib & 7]
    return torch.where(nib >= 8, -mag, mag)


def scaled_rows(x, mixed, sbytes, K, r0, r1, k0, k1, swap=False, nib_swap=False):
    """float64 scaled values of rows r0..r1-1, columns k0..k1-1, with the scale bytes (flat uint8
    tensor) at sIdx; also the scale bytes and the elements themselves."""
    r = torch.arange(r0, r1, device=DEV, dtype=torch.int64)[:, None]
    k = torch.arange(k0, k1, device=DEV, dtype=torch.int64)[None, :]
    s = sbytes[sidx(K, r, k, swap)].to(torch.int64)
    x = elems(x, mixed, r0, r1, k0, k1, nib_swap)
    return x * torch.exp2((s - 127).to(torch.float64)), s, x


def check_domain(x, mixed, sbytes, K, R):
    """InDom for every element of an R × K matrix with its scales."""
    ok = True
    for r0 in range(0, R, 1024):
        r1 = min(R, r0 + 1024)
        for k0 in range(0, K, 8192):
            k1 = min(K, k0 + 8192)
            v, s, xv = scaled_rows(x, mixed, sbytes, K, r0, r1, k0, k1)
            a = v.abs()
            ok &= bool(torch.isfinite(xv).all())
            ok &= bool(((s >= 1) & (s <= 254)).all())
            ok &= bool(((a == 0) | ((a >= 2.0 ** -126) & (a <= BF16_MAX))).all())
    return ok


def ulp16(c):
    """The float16 ulp at |c| (finite c), as float64."""
    a = c.abs().clamp_min(2.0 ** -14)
    _, ex = torch.frexp(a)          # a = m·2^ex, m in [0.5, 1): the exponent of a is ex − 1
    return torch.exp2((ex - 11).to(torch.float64))


def reference(a8, sa, b, sb, mixed, K, i0, i1, N, wrong=None):
    """S, tol and Σ|p| for rows i0..i1-1 (float64), by the claim's recursion; with `wrong`, only S
    for a wrong computation: "drop" (products 0..31 left out), "scale2" (the scale of a's first
    group doubled), "layout" (the scale layout's
    indices `(r % 128) // 32` and `(k % 128) // 32` exchanged), "nibble" (mixed: the two
    nibbles of each byte of b swapped)."""
    T = K // 128
    S = torch.zeros(i1 - i0, N, device=DEV, dtype=torch.float64)
    tol = torch.zeros_like(S)
    absum = torch.zeros_like(S)
    sw = wrong == "layout"
    for t in range(T):
        k0, k1 = 128 * t, 128 * t + 128
        A, _, _ = scaled_rows(a8, False, sa, K, i0, i1, k0, k1, swap=sw)
        B, _, _ = scaled_rows(b, mixed, sb, K, 0, N, k0, k1, swap=sw, nib_swap=(wrong == "nibble"))
        if wrong is None:
            ch = A.abs() @ B.abs().T
            tol = tol + RHO * (S.abs() + tol + ch) + ALPHA
            absum += ch
        elif t == 0 and wrong == "drop":
            A = A.clone(); A[:, :32] = 0
        elif t == 0 and wrong == "scale2":
            A = A.clone(); A[:, :32] *= 2
        S = S + A @ B.T
    return S, tol, absum


def post_ok(c, S, tol):
    """Post's two clauses per element: (ok, finite-clause ratio or 0 where the clause is void)."""
    fin_cl = (S.abs() + tol) <= MAX16
    bound = tol + (S.abs() + tol) * U16 + ETA16
    err = (c - S).abs()
    finite = torch.isfinite(c)
    ratio = torch.where(fin_cl, torch.where(finite, err / bound, torch.full_like(err, math.inf)),
                        torch.zeros_like(err))
    cz = torch.where(finite, c, torch.zeros_like(c))
    half = ulp16(cz) / 2
    mant, _ = torch.frexp(cz.abs())
    below = (mant == 0.5) & (cz.abs() >= 2.0 ** -13) & (S.abs() < cz.abs())
    half = torch.where(below, half / 2, half)   # below a power of two the spacing halves
    c1 = torch.where(finite, err <= tol + half,
                     torch.where(c > 0, S + tol >= 65520.0, S - tol <= -65520.0))
    c1 &= ~torch.isnan(c)
    ok = c1 & (~fin_cl | (finite & (err <= bound)))
    return ok, ratio, fin_cl


def check_output(a8, sa, b, sb, mixed, c16, M, N, K):
    """Post on every element; returns (worst ratio of the finite clause, #finite-clause elements,
    all elements ok, room ok, largest per-block median of tol/|S|)."""
    worst = 0.0
    nfin = 0
    allok = True
    room_ok = True
    rel_tol = []
    RB = 1024
    for i0 in range(0, M, RB):
        i1 = min(M, i0 + RB)
        S, tol, absum = reference(a8, sa, b, sb, mixed, K, i0, i1, N)
        room_ok &= bool((absum + tol <= 2.0 ** 127).all())
        c = c16[i0:i1].to(torch.float64)
        ok, ratio, fin_cl = post_ok(c, S, tol)
        allok &= bool(ok.all())
        worst = max(worst, float(ratio.max()))
        nfin += int(fin_cl.sum())
        nz = S.abs() > 0
        if nz.any():
            rel_tol.append((tol[nz] / S.abs()[nz]).median().item())
    return worst, nfin, allok, room_ok, (max(rel_tol) if rel_tol else 0.0)


class GuardedEmpty:
    """During the host call, torch.empty((M, N), dtype=float16) returns a view between two guard
    regions; everything starts as the sentinel."""

    G = 4096

    def __init__(self, M, N):
        self.M, self.N = M, N
        self.orig = torch.empty
        self.buf = None

    def __enter__(self):
        def fake(*args, **kw):
            shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            if tuple(shape) == (self.M, self.N) and kw.get("dtype") == torch.float16:
                n = self.M * self.N
                self.buf = self.orig(n + 2 * self.G, dtype=torch.int16, device=DEV).fill_(SENT)
                return self.buf[self.G:self.G + n].view(torch.float16).view(self.M, self.N)
            return self.orig(*args, **kw)
        torch.empty = fake
        return self

    def __exit__(self, *exc):
        torch.empty = self.orig
        return False

    def guards_ok(self):
        g = self.G
        return bool((self.buf[:g] == SENT).all()) and bool((self.buf[-g:] == SENT).all())


def snapshot(x):
    """A full copy of a tensor's bytes."""
    return x.contiguous().view(torch.uint8).reshape(-1).clone()


def unchanged(x, copy):
    return torch.equal(x.contiguous().view(torch.uint8).reshape(-1), copy)


def host_like(fmt):
    """The tensors and descriptors `initialize_block_scaled(M, N, K, fmt)` builds (lines 330–449)
    for `fmt` in `"mxfp8"`, `"mixed"`, with the same distributions (`MXFP4Tensor(…).random()` cast to
    `e4m3`, or packed two per byte along K for `"mixed"`'s `b`; scale bytes
    `MXScaleTensor(torch.rand(…) + 1e-8)`), generated in slices of rows so that the host's float32
    temporaries stay small. Returns a builder giving the host function's first eleven entries."""
    def build(M, N, K):
        def fp8_rows(R):
            x = torch.empty(R, K, dtype=torch.float8_e4m3fn, device=DEV)
            for r0 in range(0, R, 512):
                r1 = min(R, r0 + 512)
                x[r0:r1] = bsm.MXFP4Tensor(size=(r1 - r0, K), device=DEV).random().to(torch.float32).to(
                    torch.float8_e4m3fn)
            return x

        def fp4_rows(R):
            x = torch.empty(R, K // 2, dtype=torch.uint8, device=DEV)
            for r0 in range(0, R, 512):
                r1 = min(R, r0 + 512)
                x[r0:r1] = bsm.MXFP4Tensor(size=(r1 - r0, K), device=DEV).random().to_packed_tensor(dim=1)
            return x
        mixed = fmt == "mixed"
        a = fp8_rows(M)
        b = fp4_rows(N) if mixed else fp8_rows(N)
        sa = bsm.MXScaleTensor(torch.rand([M // 128, K // 32 // 4, 32, 16], device=DEV) + 1e-8).data
        sb = bsm.MXScaleTensor(torch.rand([N // 128, K // 32 // 4, 32, 16], device=DEV) + 1e-8).data
        sa = sa.reshape(1, M // 128, sa.shape[1], 2, 256)
        sb = sb.reshape(1, N // 128, sb.shape[1], 2, 256)
        TD = bsm.TensorDescriptor
        a_desc = TD.from_tensor(a, [128, 128])
        b_desc = TD.from_tensor(b, [256, 64 if mixed else 128])
        a_scale_desc = TD.from_tensor(sa, block_shape=[1, 1, 1, 2, 256])
        b_scale_desc = TD.from_tensor(sb, block_shape=[1, 2, 1, 2, 256])
        configs = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "num_stages": 4,
                   "ELEM_PER_BYTE_A": 1, "ELEM_PER_BYTE_B": 2 if mixed else 1, "VEC_SIZE": 32,
                   "disallow_acc_multi_buffer": True}
        return (a_desc, a_scale_desc, b_desc, b_scale_desc, 1, 2, 1, configs, None, a, b)
    return build


def setup(fmt, M, N, K, mutate, seed, builder):
    torch.manual_seed(seed)
    if builder is None:
        res = bsm.initialize_block_scaled(M, N, K, fmt, compute_reference=False)
    else:
        res = builder(M, N, K)
    descs = res[:4]
    rep_m, rep_n, rep_k, configs = res[4:8]
    a8, b = res[9], res[10]
    sa = descs[1].base.reshape(-1).view(torch.uint8)
    sb = descs[3].base.reshape(-1).view(torch.uint8)
    del res
    if mutate is not None:
        mutate(a8, sa, b, sb, M, N, K, fmt == "mixed")
    return descs, (rep_m, rep_n, rep_k, configs), a8, sa, b, sb


def run(name, fmt, M, N, K, mutate=None, seed=0, builder=None):
    mixed = fmt == "mixed"
    descs, (rep_m, rep_n, rep_k, configs), a8, sa, b, sb = setup(fmt, M, N, K, mutate, seed, builder)
    dom = check_domain(a8, False, sa, K, M) and check_domain(b, mixed, sb, K, N)
    before = [snapshot(t) for t in (a8, b, sa, sb)]
    with GuardedEmpty(M, N) as g:
        out = bsm.block_scaled_matmul(*descs, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)
    torch.cuda.synchronize()
    assert out.data_ptr() == g.buf[g.G:].data_ptr(), "the guarded output was not used"
    frame = g.guards_ok() and all(unchanged(t, s) for t, s in zip((a8, b, sa, sb), before))
    del before
    worst, nfin, allok, room, rt = check_output(a8, sa, b, sb, mixed, out, M, N, K)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    status = "OK" if (dom and room and allok and frame and worst < 1) else "FAIL"
    margin = (1 / worst) if worst > 0 else math.inf
    print(f"[{fmt:5s}] {name:34s} M={M:5d} N={N:5d} K={K:6d}  Pre={dom and room}  err/tol={worst:.4f} "
          f"(margin {margin:.2f}x, {nfin} elems in f16 range)  Post={allok}  frame={frame}  "
          f"median tol/|S|={rt:.2e}  peak={peak:.2f}GiB  {status}", flush=True)
    del out, a8, b, sa, sb, descs, g
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return status == "OK"


def host_reference(fmt, M, N, K, seed=1):
    """The float64 reference of the claim against the host's own `reference` (lines 406–418),
    built from the same inputs (`compute_reference=True`)."""
    torch.manual_seed(seed)
    res = bsm.initialize_block_scaled(M, N, K, fmt, compute_reference=True)
    href = res[8].to(torch.float64)
    a8, b = res[9], res[10]
    sa = res[1].base.reshape(-1).view(torch.uint8)
    sb = res[3].base.reshape(-1).view(torch.uint8)
    S, _, _ = reference(a8, sa, b, sb, fmt == "mixed", K, 0, M, N)
    err = float((S - href).abs().max() / href.abs().max())
    ok = err < 1e-6
    print(f"[{fmt:5s}] host reference vs claim's S at {M}x{N}x{K}: max rel diff {err:.2e}  "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    return ok


def negatives(fmt, M, N, K, seed=3):
    """Wrong outputs that `Post` must reject, on the host's inputs: each is rounded to float16
    (`zero`: all zero; `drop`: one group of 32 products left out; `scale2`: one group's scale
    doubled; `layout`: the scale layout with two indices exchanged; `nibble` (mixed): the nibbles
    of each byte swapped; `neighbour`: element (i, j + 1) in place of (i, j)). Prints the share of
    elements each one breaks."""
    mixed = fmt == "mixed"
    _, _, a8, sa, b, sb = setup(fmt, M, N, K, None, seed, None)
    S, tol, _ = reference(a8, sa, b, sb, mixed, K, 0, M, N)
    kinds = ["zero", "drop", "scale2", "layout", "neighbour"] + (["nibble"] if mixed else [])
    ok = True
    for kind in kinds:
        if kind == "zero":
            c = torch.zeros_like(S)
        elif kind == "neighbour":
            c = torch.roll(S, -1, dims=1).to(torch.float16).to(torch.float64)
        else:
            Sw, _, _ = reference(a8, sa, b, sb, mixed, K, 0, M, N, wrong=kind)
            c = Sw.to(torch.float16).to(torch.float64)
        good, _, _ = post_ok(c, S, tol)
        frac = 1 - float(good.double().mean())
        rejected = not bool(good.all())
        ok &= rejected
        print(f"[{fmt:5s}] negative {kind:9s} at {M}x{N}x{K}: Post rejects ({frac:.1%} of elements break it)"
              f"  {'OK' if rejected else 'FAIL'}", flush=True)
    return ok


# ---- in-domain adversarial inputs (written over the host's tensors before the launch) ----

def rand_e4m3(shape, lo_bits=0, hi_bits=0x7E):
    """Random finite e4m3 bit patterns (magnitude field in [lo_bits, hi_bits]), random signs."""
    m = torch.randint(lo_bits, hi_bits + 1, shape, device=DEV, dtype=torch.int32)
    s = torch.randint(0, 2, shape, device=DEV, dtype=torch.int32) << 7
    return (m | s).to(torch.uint8).view(torch.float8_e4m3fn)


def rand_e2m1_bytes(shape, lo=0, hi=7, signed=True):
    """Random bytes of two e2m1 nibbles, magnitudes in [lo, hi] (codes), random signs."""
    def nib():
        m = torch.randint(lo, hi + 1, shape, device=DEV, dtype=torch.int32)
        if signed:
            m |= torch.randint(0, 2, shape, device=DEV, dtype=torch.int32) << 3
        return m
    return (nib() | (nib() << 4)).to(torch.uint8)


def pos_e4m3(shape, lo, hi):
    return rand_e4m3(shape, lo, hi).view(torch.uint8).bitwise_and(0x7F).view(torch.float8_e4m3fn)


def mut_same_sign(a8, sa, b, sb, M, N, K, mixed):
    """All products positive (every truncation pushes the same way), scale 2^-4."""
    a8.copy_(pos_e4m3(a8.shape, 0x30, 0x48))
    b.copy_(rand_e2m1_bytes(b.shape, 1, 7, signed=False) if mixed else pos_e4m3(b.shape, 0x30, 0x48))
    sa.fill_(127 - 4)
    sb.fill_(127 - 4)


def mut_cancel(a8, sa, b, sb, M, N, K, mixed):
    """Partial sums that grow and then cancel: products of one sign on the first half of K and of
    the other on the second half (for mxfp8, b row j is a row of a, negated on the second half)."""
    if mixed:
        a8.copy_(pos_e4m3(a8.shape, 0x38, 0x50))
        bb = rand_e2m1_bytes(b.shape, 1, 7, signed=False)
        bb[:, K // 4:] |= 0x88
        b.copy_(bb)
    else:
        a8.copy_(pos_e4m3(a8.shape, 0x38, 0x50))
        rows = torch.arange(N, device=DEV) % M
        bb = a8[rows].to(torch.float32)
        bb[:, K // 2:] = -bb[:, K // 2:]
        b.copy_(bb.to(torch.float8_e4m3fn))
    sa.fill_(120)
    sb.fill_(120)


def mut_full_range(a8, sa, b, sb, M, N, K, mixed):
    """Every finite e4m3 pattern (subnormals included) and every e2m1 code, with scale bytes in
    116 … 126: some outputs leave float16's range (the rounding clause)."""
    a8.copy_(rand_e4m3(a8.shape))
    b.copy_(rand_e2m1_bytes(b.shape) if mixed else rand_e4m3(b.shape))
    sa.copy_(torch.randint(116, 127, sa.shape, device=DEV, dtype=torch.uint8))
    sb.copy_(torch.randint(116, 127, sb.shape, device=DEV, dtype=torch.uint8))


def mut_extreme_scales(a8, sa, b, sb, M, N, K, mixed):
    """Scale byte 254 (2^127) on `a` with |x| ≤ 1.875, and a small scale on `b` (byte 1, 2^-126,
    with |x| ≥ 1 or 0 for e4m3; byte 2, 2^-125, for e2m1, whose smallest value is 0.5): scaled
    values at both ends of bfloat16's normal range, moderate products."""
    a8.copy_(rand_e4m3(a8.shape, 0x08, 0x3F))          # 2^-6 … 1.875 (normals; 2^-6·2^127 in range)
    if mixed:
        b.copy_(rand_e2m1_bytes(b.shape))
        sb.fill_(2)
    else:
        m = torch.randint(0x38, 0x7E + 1, b.shape, device=DEV, dtype=torch.int32)   # 1 … 448
        m = torch.where(torch.rand(b.shape, device=DEV) < 0.1, torch.zeros_like(m), m)
        s = torch.randint(0, 2, b.shape, device=DEV, dtype=torch.int32) << 7
        b.copy_((m | s).to(torch.uint8).view(torch.float8_e4m3fn))
        sb.fill_(1)
    sa.fill_(254)


def mut_subnormal(a8, sa, b, sb, M, N, K, mixed):
    """Subnormal e4m3 elements (2^-9 … 7·2^-9) on `a` (and `b` for mxfp8; the e2m1 subnormal 0.5
    for mixed), scale byte 140 (2^13)."""
    a8.copy_(rand_e4m3(a8.shape, 0x01, 0x07))
    b.copy_(rand_e2m1_bytes(b.shape, 0, 1) if mixed else rand_e4m3(b.shape, 0x00, 0x07))
    sa.fill_(140)
    sb.fill_(140)


def main():
    global E2M1
    E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV, dtype=torch.float64)
    quick = "--quick" in sys.argv
    ok = True
    for fmt in ("mxfp8", "mixed"):
        # the host draws the same element values for both formats from one seed, so the edges of
        # "mixed" use another seed than those of "mxfp8"
        sd = 0 if fmt == "mxfp8" else 1
        ok &= host_reference(fmt, 256, 512, 512)
        # SETUP.md's launches (the host's data, seed 42 as the script's __main__)
        ok &= run("test validate(8192,8192,8192)", fmt, 8192, 8192, 8192, seed=42, builder=host_like(fmt))
        ok &= run("bench K=512 (default -K)", fmt, 8192, 8192, 512, seed=42)
        if not quick and fmt == "mxfp8":
            ok &= run("bench K=16384 (docstring's largest)", fmt, 8192, 8192, 16384, seed=42,
                      builder=host_like(fmt))
        # edges of Inst.ok
        ok &= run("edge M=N=K=128 (N half a tile)", fmt, 128, 128, 128, seed=sd)
        ok &= run("edge M=128 N=384 K=256", fmt, 128, 384, 256, seed=sd)
        ok &= run("edge K=2^17 (largest), M=128 N=384", fmt, 128, 384, 2 ** 17, seed=sd)
        ok &= run("edge M=1152 N=640 K=1408", fmt, 1152, 640, 1408, seed=sd)
        # edges of Pre (in-domain adversarial values)
        ok &= run("Pre: same-sign products", fmt, 512, 512, 8192, mutate=mut_same_sign, seed=sd)
        ok &= run("Pre: same-sign products, K=2^17", fmt, 128, 256, 2 ** 17, mutate=mut_same_sign, seed=sd)
        ok &= run("Pre: cancelling partial sums", fmt, 512, 512, 8192, mutate=mut_cancel, seed=sd)
        ok &= run("Pre: full range, f16 overflow", fmt, 512, 512, 4096, mutate=mut_full_range, seed=sd)
        ok &= run("Pre: extreme scale bytes", fmt, 512, 512, 4096, mutate=mut_extreme_scales, seed=sd)
        ok &= run("Pre: subnormal elements", fmt, 512, 512, 4096, mutate=mut_subnormal, seed=sd)
        # wrong outputs Post must reject
        ok &= negatives(fmt, 1024, 1024, 512, seed=3 + sd)
        ok &= negatives(fmt, 1024, 1024, 8192, seed=3 + sd)
    print("ALL OK" if ok else "SOME CASE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
