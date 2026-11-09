"""GPU check of the claim `Claims.SsdBmm` (Claim.lean).

Runs the kernel's own host `bmm_chunk_fwd` (source/ssd_bmm.py, imported by path) as Mamba-2's
chunked scan calls it (`bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx,
output_dtype=torch.float32)`) on SETUP.md's launches and on launches at the edges of `Inst.ok`,
and checks `Pre` and `Post` exactly as the claim states them:

* `Pre`: every `seq_idx` entry is >= 0;
* `Post`, for every element `(0, c, g, m, n)` of `out`: when `sameReq` (both tokens
  `c*C + m`, `c*C + n` below `seqlen`, equal `seq_idx`) and the two rows are finite with
  `dotMag <= 2^120`, the value is finite and within `tol` of the exact dot product; when not
  `sameReq`, the value is the float32 `+0.0` bit for bit (whatever the inputs); `tol` is the
  claim's `steps·2^-16·dotMag + 2^-112`, `steps = cdiv(K, 32)` and `dotMag` the sum of the
  products' lifted magnitudes;
* the byte frame: the inputs' storage unchanged, and guard bands of 4096 elements on each side of
  `out` (the host's `torch.empty` allocation is placed inside a sentinel-filled buffer during the
  call) unchanged.

The reference is computed in float64 from the inputs: each bf16 product is exact in float64, and
the float64 sums err by about 1e-16 relative to the magnitudes, far below the tolerance (at least
1.5e-5 relative). Each line reports the largest |error| / tol over the constrained elements; the
end reports tol / |S| (median and worst) over SETUP.md's launches.

Negative tests: wrong outputs that `Post` must reject (the exact result rounded to bfloat16; the
result missing the last `tl.dot` step; no zeroing across requests; the causal variant, which leaves
tiles unwritten).

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "ssd_bmm.py")
spec = importlib.util.spec_from_file_location("ssd_bmm_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
BK = 32
EPS = 2.0 ** -16                            # tol = steps * EPS * dotMag + DELTA
DELTA = 2.0 ** -112
MIN_NORMAL_BF16 = 2.0 ** -126
GUARD = 4096
SENTINEL = 0x7FC0DEAD                       # a NaN pattern; the kernel stores no such value
worst = [0.0]
failures = []
rel_samples = []                            # tol / |S| on SETUP.md's launches (subsampled)
rel_worst = [0.0]


def lift(x):
    """Rules.Num.liftMag .bf16 on float64 values: 0 for 0, else max(|q|, 2^-126)."""
    ax = x.abs()
    return torch.where(ax == 0, ax, ax.clamp(min=MIN_NORMAL_BF16))


def post(out, a, b, seq_idx, C, stats=False):
    """Post's element clauses on `out` (float32 [1, nch, G, C, C]). Returns (holds, worst ratio,
    constrained count). With `stats`, records tol / |S| over the constrained elements."""
    _, T, G, K = a.shape
    nch = (T + C - 1) // C
    assert tuple(out.shape) == (1, nch, G, C, C)
    steps = (K + BK - 1) // BK
    sid = seq_idx[0].long()
    ok = True
    rmax = 0.0
    count = 0
    nb = max(1, (1 << 22) // (G * C * C))         # chunks per batch
    for c0 in range(0, nch, nb):
        c1 = min(nch, c0 + nb)
        t0, t1 = c0 * C, min(T, c1 * C)
        npos = (c1 - c0) * C
        A = torch.zeros((npos, G, steps * BK), dtype=torch.float64, device=DEV)
        Bm = torch.zeros_like(A)
        A[: t1 - t0, :, :K] = a[0, t0:t1].double()
        Bm[: t1 - t0, :, :K] = b[0, t0:t1].double()
        A = A.view(c1 - c0, C, G, steps * BK).permute(0, 2, 1, 3)       # [nb, G, C, K']
        Bm = Bm.view(c1 - c0, C, G, steps * BK).permute(0, 2, 1, 3)
        finA = torch.isfinite(A).all(-1)                                # [nb, G, C]
        finB = torch.isfinite(Bm).all(-1)
        A0 = torch.where(torch.isfinite(A), A, 0.0)
        B0 = torch.where(torch.isfinite(Bm), Bm, 0.0)
        S = A0 @ B0.transpose(-1, -2)
        mag = lift(A0) @ lift(B0).transpose(-1, -2)
        E = steps * EPS * mag + DELTA
        pos = torch.arange(c0 * C, c1 * C, device=DEV).view(c1 - c0, C)
        valid = pos < T
        sp = torch.full((npos,), -1, dtype=torch.long, device=DEV)
        sp[: t1 - t0] = sid[t0:t1]
        sp = sp.view(c1 - c0, C)
        same = valid[:, :, None] & valid[:, None, :] & (sp[:, :, None] == sp[:, None, :])
        same = same[:, None].expand(-1, G, -1, -1)
        got = out[0, c0:c1]
        cons = same & finA[..., :, None] & finB[..., None, :] & (mag <= 2.0 ** 120)
        g = got.double()
        if cons.any():
            fin = torch.isfinite(g[cons]).all().item()
            r = ((g - S).abs() / E)[cons].max().item()
            ok = ok and fin and r <= 1
            rmax = max(rmax, r)
            count += int(cons.sum().item())
            if stats:
                q = (E / S.abs())[cons]
                rel_worst[0] = max(rel_worst[0], q.max().item())
                idx = torch.randint(0, q.numel(), (min(q.numel(), 200000),), device=DEV)
                rel_samples.append(q[idx].float().cpu())
        zero_ok = bool((got.view(torch.int32)[~same] == 0).all().item())
        ok = ok and zero_ok
    return ok, rmax, count


def same_bits(x, y):
    """Bit-for-bit equality of two tensors (NaN payloads included)."""
    v = {torch.bfloat16: torch.int16, torch.float32: torch.int32}.get(x.dtype)
    return torch.equal(x.view(v), y.view(v)) if v is not None else torch.equal(x, y)


def run_host(a, b, seq_idx, C, causal=False):
    """The host's launch, with its `out` allocation placed in a guarded buffer; returns
    (out, frame ok)."""
    keep = [x.clone() for x in (a, b, seq_idx)]
    # the whole storage of each input (the packed xBC tensor for the mixer's views), once
    bases = []
    for x in (a, b, seq_idx):
        base = x._base if x._base is not None else x
        if all(base is not y for y in bases):
            bases.append(base)
    keep_bases = [x.clone() for x in bases]
    orig_empty = torch.empty
    bufs = []

    def empty_guarded(shape, device=None, dtype=None):
        n = math.prod(shape)
        big = orig_empty((n + 2 * GUARD,), device=device, dtype=dtype)
        big.view(torch.int32).fill_(SENTINEL)
        bufs.append(big)
        return big[GUARD:GUARD + n].view(shape)

    torch.empty = empty_guarded
    try:
        out = mod.bmm_chunk_fwd(a, b, C, seq_idx=seq_idx, causal=causal, output_dtype=torch.float32)
    finally:
        torch.empty = orig_empty
    torch.cuda.synchronize()
    frame = len(bufs) == 1 and all(same_bits(x, y) for x, y in zip((a, b, seq_idx), keep))
    frame = frame and all(same_bits(x, y) for x, y in zip(bases, keep_bases))
    for big in bufs:
        gb = big.view(torch.int32)
        frame = frame and bool((gb[:GUARD] == SENTINEL).all()) and bool((gb[-GUARD:] == SENTINEL).all())
    return out, frame


def check(name, a, b, seq_idx, C, stats=False):
    _, T, G, K = a.shape
    pre = bool((seq_idx >= 0).all().item())
    out, frame = run_host(a, b, seq_idx, C)
    ok, r, n = post(out, a, b, seq_idx, C, stats)
    worst[0] = max(worst[0], r)
    good = pre and ok and frame
    nreq = int(seq_idx.max().item()) + 1 if T else 0
    print(f"{'ok ' if good else 'BAD'} {name}: tokens {T} groups {G} dstate {K} chunk {C} "
          f"(strides {tuple(a.stride())}): err/tol {r:.4f} over {n} elements; Pre {pre}, "
          f"zeros+frame {ok and frame}")
    if not good:
        failures.append(name)
    return out


def negative(name, out, a, b, seq_idx, C):
    ok, r, _ = post(out, a, b, seq_idx, C)
    print(f"{'ok ' if not ok else 'BAD'} negative, {name}: Post {'rejects' if not ok else 'ACCEPTS'} "
          f"(largest err/tol {r:.1f})")
    if ok:
        failures.append("negative " + name)


def inputs(seqlens, G, K, gen, conv_dim=None, scale=1.0):
    C_, B_, s = mod.make_inputs(seqlens, G, K, gen, conv_dim=conv_dim, device=DEV)
    if scale != 1.0:
        C_ = (C_.double() * scale).to(torch.bfloat16)
        B_ = (B_.double() * scale).to(torch.bfloat16)
    return C_, B_, s


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # SETUP.md's launches. SGLang's tests (test_mamba_ssm_ssd.py), in bfloat16 with seq_idx:
    cases = [
        ("single, g3 k5", [112], 3, 5, 16, None),
        ("single, g32 k128", [112], 32, 128, 16, None),
        ("single (outside CI), g11 k19 c32", [128], 11, 19, 32, None),
        ("intermediate states, g4 k32 c32", [128], 4, 32, 32, None),
        ("intermediate states, g16 k64 c64", [256], 16, 64, 64, None),
        ("cont c8, (64, 32)", [64, 32, 64], 4, 5, 8, None),
        ("cont c8, (32, 32, 32)", [32, 32, 32, 32, 32, 32], 13, 32, 8, None),
        ("cont c8, (8, 8, 8)", [8, 8, 8, 8, 8, 8], 4, 32, 8, None),
        ("cont c8, (4, 4, 4, 4)", [4] * 8, 13, 5, 8, None),
        ("cont c8, varied", [64, 32, 16, 8, 8, 8, 16, 32, 16, 8], 4, 32, 8, None),
        ("cont c256, (5,) (1,)", [5, 1, 1, 1], 13, 5, 256, None),
        ("cont c256, (5, 30) (1, 2)", [5, 30, 1, 2, 1, 2, 1, 2], 4, 32, 256, None),
        ("cont c128, (138, 225)", [138, 225, 138, 225], 13, 32, 128, None),
        ("chunking c8, (16, 2, 8, 13)", [16, 2, 8, 13], 16, 64, 8, None),
        ("chunking c256, (270, 88, 212, 203)", [270, 88, 212, 203], 16, 64, 256, None),
        ("chunking c256, (16, 20)", [16, 20], 16, 64, 256, None),
        ("chunking c8, (270, 88, 212, 203)", [270, 88, 212, 203], 16, 64, 8, None),
    ]
    # the Mamba-2 mixer: C, B as views of the packed xBC tensor
    cases += [
        ("mixer g8 k128 c256, 32768 tokens", [3000, 1, 700, 4096, 383, 16384, 8204], 8, 128, 256, 10240),
        ("mixer g8 k128 c128, 16384 tokens", [8192, 5, 1, 8186], 8, 128, 128, 10240),
        ("mixer g4 k128 c256 (TP 2)", [5000, 3000], 4, 128, 256, 5120),
        ("mixer g1 k128 c256", [12000, 1, 20767], 1, 128, 256, 4352),
        ("mixer g1 k256 c256 (Falcon-H1)", [8192, 5, 1, 8190], 1, 256, 256, 8704),
    ]
    for name, sl, G, K, C, cd in cases:
        a, b, s = inputs(sl, G, K, gen, conv_dim=cd)
        check(name, a, b, s, C, stats=True)
    # edges of Inst.ok
    edge = [
        ("dstate 1", [7, 30, 2], 2, 1, 16),
        ("dstate 33 (a partial step)", [100, 1, 27], 3, 33, 64),
        ("dstate 1024 (K_le)", [300, 212], 2, 1024, 256),
        ("chunk 2", [3, 1, 5, 2, 9], 2, 16, 2),
        ("chunk 2048", [3000, 1, 1095], 1, 64, 2048),
        ("one token", [1], 2, 128, 256),
        ("200 one-token requests", [1] * 200, 2, 64, 64),
        ("2^20 tokens, g1 k128 c256", [1 << 19, 1, (1 << 19) - 1], 1, 128, 256),
        ("grid z 65535 (c8 g1)", [65535 * 8 - 3], 1, 16, 8),
    ]
    for name, sl, G, K, C in edge:
        a, b, s = inputs(sl, G, K, gen)
        check(name, a, b, s, C)
    # arbitrary non-negative seq_idx (not monotone)
    a, b, _ = inputs([700], 4, 64, gen)
    s = torch.randint(0, 3, (1, 700), generator=gen, dtype=torch.int32).to(DEV)
    check("seq_idx random in {0, 1, 2}", a, b, s, 128)
    # `a`, `b` with stride(1) = 1 (the host does not copy them): any strides are covered
    T, G, K = 300, 2, 48
    a = torch.randn((1, G, K, T), generator=gen).to(torch.bfloat16).to(DEV).permute(0, 3, 1, 2)
    b = torch.randn((1, G, K, T), generator=gen).to(torch.bfloat16).to(DEV).permute(0, 3, 1, 2)
    s = torch.repeat_interleave(torch.arange(2, dtype=torch.int32), torch.tensor([120, 180])).unsqueeze(0).to(DEV)
    check("token-major a, b (stride(1) = 1)", a, b, s, 64)
    # batch strides from 2^31 on (a long prefill's `seqlen·conv_dim`): `int64` arguments
    # (`intArg`); with batch = 1 the kernel multiplies them by `pid_b = 0`, so a view with a large
    # stride(0) over a small storage is the same launch as the long prefill's at these sizes
    a, b, s = inputs([700, 1, 323], 8, 128, gen, conv_dim=10240)
    for big in ((1 << 31) + 7, 262144 * 10240):
        a64 = a.as_strided(a.shape, (big,) + a.stride()[1:], a.storage_offset())
        b64 = b.as_strided(b.shape, (big,) + b.stride()[1:], b.storage_offset())
        check(f"int64 a.stride(0) = b.stride(0) = {big}", a64, b64, s, 256)
    # values: large (dotMag near 2^117), subnormal operands, NaN and infinities
    a, b, s = inputs([500, 12], 2, 128, gen, scale=2.0 ** 54)
    check("values near 2^54", a, b, s, 256)
    a, b, s = inputs([500, 12], 2, 128, gen, scale=2.0 ** -128)
    check("subnormal operands", a, b, s, 256)
    a, b, s = inputs([300, 80, 132], 2, 64, gen)
    a[0, 5, 0, 3] = float("nan")
    a[0, 301, 1, 0] = float("inf")
    b[0, 77, 0, 63] = float("-inf")
    check("NaN and infinities (zeros still exact)", a, b, s, 128)
    # negative tests: Post must reject these outputs
    a, b, s = inputs([1000, 3, 1300], 8, 128, gen, conv_dim=10240)
    ref = mod.reference(a, b, s, 256)
    negative("exact result rounded to bfloat16", ref.to(torch.bfloat16).float(), a, b, s, 256)
    a2 = a.clone()
    a2[..., 96:] = 0
    out_short, _ = run_host(a2, b, s, 256)
    negative("missing the last tl.dot step", out_short, a, b, s, 256)
    out_nomask, _ = run_host(a, b, torch.zeros_like(s), 256)
    negative("no zeroing across requests", out_nomask, a, b, s, 256)
    out_causal, _ = run_host(a, b, s, 256, causal=True)
    negative("causal launch (tiles left unwritten)", out_causal, a, b, s, 256)
    print(f"worst err/tol: {worst[0]:.4f}")
    allq = torch.cat(rel_samples)
    print(f"tol/|S| on SETUP.md's launches: median {allq.median().item():.3e}, "
          f"worst {rel_worst[0]:.3e}")
    if failures:
        print("FAILED:", ", ".join(failures))
        raise SystemExit(1)
    print("all checks hold")
