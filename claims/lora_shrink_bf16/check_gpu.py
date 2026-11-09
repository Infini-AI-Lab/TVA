"""GPU check of the claim for `lora_shrink_bf16`.

Runs the kernel's own host (`source/lora_shrink.py`, imported by path) on the launches of SETUP.md
and on launches at the edges of `Inst.ok`, and checks `Post` as Claim.lean states it, in float64:

* a token with an adapter `l >= 0`, slice `s`: |out - scaling * S| <= tol, `scaling` the host's
  double, S = sum_k x[t,k] * A_s[l,j,k], dotMag = sum_k liftMag(x) liftMag(w), with the host's
  BLOCK_K = bk, SPLIT_K = sk (256, 64 below 128 tokens; 32, 8 from 128):
    tol = bk * cdiv(K, bk * sk) * 2^-21 * |scaling| * dotMag + 2^-100 * (1 + |scaling| + dotMag);
  the room conditions dotMag <= 2^120 and vMag + atomTol <= maxVal are checked too, with
  Claim.lean's
    errChunk p = accErr p (T_p): per step (one tl.dot_add of bk products), from the error e
                after the step before, the larger of the fused form (the step's chunk steps
                continue the chain from e against the part's exact partial sums) and the
                unfused form (the dot's own chain from 0, within L of the step's exact sum D,
                then e + L + 3u (|D| + L + |P| + e) + 4 eta, AddRel.err of the add); one chunk
                step (DotBF16Chunk's first arm):
                e' = (1 + 2^-13 + 2^-20) e + 2^-13 max(|P|, m) + 2^-20 |P + c| + 2^-143,
                P the exact partial sum, c the chunk's exact sum, m its largest lifted product,
    errStep p  = ((1 + rho)^T_p - 1) L_p + T_p (1 + rho)^T_p alpha (DotBF16 per tl.dot step,
                rho = stepRel relDot16 bk, alpha = stepAbs absDot16 bk, T_p the part's steps,
                L_p = sum of liftMag(x) liftMag(w) over its k),
    errPart p  = min(errChunk p, errStep p),
    partMag = sum_p (|partSum p| + errPart p),
    mulTol  = 2^-24 * |s32| * partMag + sk * 2^-150   (s32 the kernel's float32 `scaling`),
    vMag    = |s32| * partMag + mulTol,
    atomTol = ((1 + 2^-23)^sk - 1) * vMag + sk * (1 + 2^-23)^sk * 2^-124;
* a token without an adapter: the element is exactly +0.0 (bit pattern 0), in every slice;
* the byte frame: the host's output is allocated inside a larger buffer whose guard bands (4096
  sentinel elements before and after the [S, M, R] output) must keep their bytes, and every buffer
  the host is given (inputs, each slice's weights, the four metadata tensors) is unchanged.

It reports, per launch, the largest error / tolerance (must be below 1) and its margin, and the
tolerance against the output (tol / |out|, median and largest). Wrong outputs that `Post` must
reject are built from the kernel's own output on a signed launch per configuration and checked
against the same tolerance: all zero, the last block of `BLOCK_K` k dropped, one of the `SPLIT_K`
parts dropped, the scale missing, another adapter's weights, and the slices swapped. The ranks 1,
4 and 8, which the claim excludes (a false `tl.multiple_of`/`tl.max_contiguous` hint, README
"Exclusions"), are run for information only.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os
from fractions import Fraction

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "lora_shrink.py")
spec = importlib.util.spec_from_file_location("lora_shrink_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")

# The library's constants (TVA.Rules.Num.Dot16Chunk, Sm90), exactly, then as float64.
U = Fraction(1, 2 ** 24)          # binary32.u
ETA = Fraction(1, 2 ** 150)       # binary32.eta
REL = lambda K: (K + 2) * Fraction(1, 2 ** 22)    # relDot16
ABS = lambda K: (K + 2) * Fraction(1, 2 ** 126)   # absDot16
RHO = lambda K: REL(K) + 3 * U * (1 + REL(K))     # DotLoop.stepRel relDot16 K
ALPHA = lambda K: ABS(K) * (1 + 3 * U) + 4 * ETA  # DotLoop.stepAbs absDot16 K
RHO_C = 2.0 ** -13                # rhoChunk16
SIG_C = 2.0 ** -20                # sigmaChunk16
ALPHA_C = 2.0 ** -143             # alphaChunk16
AREL, AABS = Fraction(1, 2 ** 23), Fraction(1, 2 ** 124)  # AtomicAddF32
MAXVAL = float((2 ** 24 - 1) * 2 ** 104)          # binary32.maxVal
MIN_NORMAL_BF16 = 2.0 ** -126
BATCH_BYTES = 256 * 2 ** 20       # the [rows, R, chunk] lifted products of one batch of rows


def config(M):
    """The host's (BLOCK_K, SPLIT_K): vLLM's default shrink config."""
    return (256, 64) if M < 128 else (32, 8)


def lift(q):
    """liftMag .bf16: 0 at 0, else max(|q|, 2^-126)."""
    a = q.abs()
    return torch.where(a == 0, a, torch.clamp(a, min=MIN_NORMAL_BF16))


def chunk_bounds(M, K, xr, wl):
    """partMag = sum_p (|partSum p| + errPart p), per element [rows, R] (for the room condition),
    with errPart p = min(errChunk p, errStep p) (Claim.lean): errChunk the accErr
    recursion (tripErr per step: the larger of the fused and the unfused forms of tl.dot_add) over
    the chunks of part p (cw = min(64, bk) consecutive k of each of its blocks b = p, p + sk, ...
    below K, in order), errStep = DotLoop.bound rho alpha T_p L_p (DotBF16 per step; T_p the part's
    steps, L_p its lifted magnitude)."""
    bk, sk = config(M)
    cw = min(64, bk)
    blocks = (K + bk - 1) // bk
    n, R = xr.shape[0], wl.shape[0]
    lx, lw = lift(xr), lift(wl)
    mag = torch.zeros((n, R), dtype=torch.float64, device=DEV)
    rb = max(1, BATCH_BYTES // (R * cw * 8))
    rho, alpha = RHO(bk), ALPHA(bk)
    for p in range(min(sk, blocks)):
        P = torch.zeros_like(mag)
        E = torch.zeros_like(mag)
        L = torch.zeros_like(mag)
        T = len(range(p, blocks, sk))
        for b in range(p, blocks, sk):
            L += lx[:, b * bk:min((b + 1) * bk, K)] @ lw[:, b * bk:min((b + 1) * bk, K)].t()
            P0, E0 = P.clone(), E.clone()
            Pl = torch.zeros_like(mag)   # the step's own partial sums, from 0
            Ll = torch.zeros_like(mag)   # the unfused dot's own chain, from 0
            for c0 in range(b * bk, (b + 1) * bk, cw):
                lo, hi = c0, min(c0 + cw, K)
                if lo >= hi:   # a chunk past K: no product, cSum = cLift = 0
                    m = torch.zeros_like(mag)
                    c = torch.zeros_like(mag)
                else:
                    c = xr[:, lo:hi] @ wl[:, lo:hi].t()
                    m = torch.cat([(lx[i:i + rb, None, lo:hi] * lw[None, :, lo:hi]).amax(-1)
                                   for i in range(0, n, rb)])
                Pn = P + c
                E = (1 + RHO_C + SIG_C) * E + RHO_C * torch.maximum(P.abs(), m) + SIG_C * Pn.abs() + ALPHA_C
                P = Pn
                Pln = Pl + c
                Ll = (1 + RHO_C + SIG_C) * Ll + RHO_C * torch.maximum(Pl.abs(), m) + SIG_C * Pln.abs() + ALPHA_C
                Pl = Pln
            Eu = E0 + Ll + 3 * float(U) * ((P - P0).abs() + Ll + P0.abs() + E0) + 4 * float(ETA)
            E = torch.maximum(E, Eu)
        q = (1 + rho) ** T
        Es = float(q - 1) * L + float(T * q * alpha)
        E = torch.minimum(E, Es)
        mag += P.abs() + E
    return mag


def tolerance(M, K, s, s32, xr, wl, S):
    """Post's tol (Claim.lean `tol`) and whether the room conditions hold, per element."""
    bk, sk = config(M)
    pmag = chunk_bounds(M, K, xr, wl)
    mul = float(U) * abs(s32) * pmag + sk * float(ETA)
    vmag = abs(s32) * pmag + mul
    q = (1 + AREL) ** sk
    atom = float(q - 1) * vmag + float(sk * q * AABS)
    A = lift(xr) @ lift(wl).t()                       # dotMag
    plen = bk * ((K + bk * sk - 1) // (bk * sk))      # partLen = bk * stepsOf 0
    tol = plen * 2.0 ** -21 * abs(s) * A + 2.0 ** -100 * (1 + abs(s) + A)
    room = bool((A <= 2.0 ** 120).all()) and bool((vmag + atom <= MAXVAL).all())
    return tol, room


GUARD = 4096
SENTINEL = 0x7FC0DEAD  # a NaN pattern no store of the kernel produces here


class _TorchWithGuards:
    """`torch` for the host's module during one call, except that `torch.zeros` (the host's output
    allocation, `lora_shrink`'s only call of it) returns a zeroed view inside a sentinel-filled buffer
    with GUARD elements before and after it."""

    def __init__(self):
        self.big = None
        self.n = 0

    def __getattr__(self, name):
        return getattr(torch, name)

    def zeros(self, size, dtype=None, device=None):
        n = 1
        for e in size:
            n *= e
        big = torch.full((GUARD + n + GUARD,), SENTINEL, dtype=torch.int32, device=device).view(dtype)
        view = big[GUARD:GUARD + n].view(size)
        view.zero_()
        self.big, self.n = big, n
        return view


def call_with_guards(fn):
    proxy = _TorchWithGuards()
    saved = mod.torch
    mod.torch = proxy
    try:
        out = fn()
    finally:
        mod.torch = saved
    torch.cuda.synchronize()
    g = proxy.big.view(torch.int32)
    guard_ok = bool((g[:GUARD] == SENTINEL).all()) and bool((g[GUARD + proxy.n:] == SENTINEL).all()) \
        and out.data_ptr() == proxy.big[GUARD:].data_ptr()
    return out, guard_ok


def make(M, K, R, max_loras, S, no_lora_frac, signed, gen):
    rnd = torch.randn if signed else torch.rand
    x = rnd((M, K), generator=gen).to(DEV, torch.bfloat16)
    ws = [rnd((max_loras, 1, R, K), generator=gen).to(DEV, torch.bfloat16) for _ in range(S)]
    mapping = torch.randint(0, max_loras, (M,), generator=gen, dtype=torch.int32)
    mapping[torch.rand(M, generator=gen) < no_lora_frac] = -1
    if no_lora_frac < 1.0 and bool((mapping == -1).all()):
        mapping[0] = max_loras - 1
    return x, ws, mapping.to(DEV)


def per_group(x, ws, mapping, max_loras, R, K):
    """Yields (s, l, rows, x rows, weights, S) for each slice s and adapter l >= 0 with tokens."""
    mapping_cpu = mapping.cpu()
    xd = x.double()
    for s, w in enumerate(ws):
        w3 = w.reshape(max_loras, R, K)
        for l in torch.unique(mapping_cpu).tolist():
            if l < 0:
                continue
            rows = torch.nonzero(mapping_cpu == l, as_tuple=True)[0].to(DEV)
            wl = w3[l].double()
            yield s, l, rows, xd[rows], wl, xd[rows] @ wl.t()


def run_case(name, M, K, R, max_loras, S, no_lora_frac, specialize, scaling, signed, gen, covered=True):
    x, ws, mapping = make(M, K, R, max_loras, S, no_lora_frac, signed, gen)
    meta = mod.lora_kernel_meta(mapping, max_loras, specialize)
    before = [t.clone() for t in [x] + ws + list(meta[:5])]
    out, guard_ok = call_with_guards(lambda: mod.lora_shrink(x, ws, *meta, scaling))
    after = [x] + ws + list(meta[:5])
    frame_ok = guard_ok and all(torch.equal(a, b) for a, b in zip(before, after)) and \
        tuple(out.shape) == (S, M, R) and out.dtype == torch.float32

    s = float(scaling)                                               # the host's double
    s32 = float(torch.tensor(scaling, dtype=torch.float32).item())  # the kernel's float32 scaling
    worst, room_ok, ratios = 0.0, True, []
    for sl, l, rows, xr, wl, Sx in per_group(x, ws, mapping, max_loras, R, K):
        tol, room = tolerance(M, K, s, s32, xr, wl, Sx)
        room_ok &= room
        err = (out[sl][rows].double() - s * Sx).abs()
        worst = max(worst, (err / tol).max().item())
        ratios.append((tol / (s * Sx).abs().clamp(min=1e-300)).flatten())
    neg = (mapping == -1).nonzero(as_tuple=True)[0]
    zero_ok = bool((out[:, neg].contiguous().view(torch.int32) == 0).all()) if neg.numel() else True
    ok = worst < 1 and frame_ok and zero_ok and room_ok
    r = torch.cat(ratios) if ratios else torch.zeros(1, device=DEV, dtype=torch.float64)
    bk, sk = config(M)
    tag = "covered " if covered else "EXCLUDED"
    print(f"[{tag}] {name:36s} M={M:5d} K={K:6d} R={R:3d} maxL={max_loras:3d} S={S} "
          f"BK={bk:3d} SK={sk:2d} -1={no_lora_frac:.2f} spec={int(specialize)} sc={scaling:g} "
          f"{'randn' if signed else 'rand '}: err/tol={worst:.3e} (margin {1 - worst:.3f}) "
          f"tol/|out| med {r.median().item():.2e} max {r.max().item():.2e} "
          f"zeros={'ok' if zero_ok else 'BAD'} frame={'ok' if frame_ok else 'BAD'} "
          f"room={'ok' if room_ok else 'BAD'}  {'PASS' if ok else 'FAIL'}")
    del x, ws, out, meta, before
    torch.cuda.empty_cache()
    return ok, worst


def negatives(name, M, K, R, max_loras, S, scaling, gen):
    """Wrong outputs built from the kernel's output; Post must reject each (some element outside
    tol). Prints, for each, the fraction of elements that would still pass."""
    x, ws, mapping = make(M, K, R, max_loras, S, 0.0, True, gen)
    meta = mod.lora_kernel_meta(mapping, max_loras, False)
    out = mod.lora_shrink(x, ws, *meta, scaling)
    torch.cuda.synchronize()
    s = float(scaling)
    s32 = float(torch.tensor(scaling, dtype=torch.float32).item())
    bk, sk = config(M)
    last = ((K - 1) // bk) * bk                  # the last block of BLOCK_K k
    part = torch.zeros(K, dtype=torch.bool)      # the k of part 1 of SPLIT_K
    for b in range(1, (K + bk - 1) // bk, sk):
        part[b * bk:(b + 1) * bk] = True
    part = part.to(DEV)
    xd = x.double()
    res = {}
    for sl, l, rows, xr, wl, Sx in per_group(x, ws, mapping, max_loras, R, K):
        tol, _ = tolerance(M, K, s, s32, xr, wl, Sx)
        w3 = ws[sl].reshape(max_loras, R, K)
        o = out[sl][rows].double()
        wrong = {
            "all zero": torch.zeros_like(o),
            "last K block dropped": s * (xd[rows, :last] @ wl[:, :last].t()),
            "one split-K part dropped": s * ((xd[rows] * ~part) @ wl.t()),
            "scale missing": o / s if s != 1.0 else o * 2,
            "another adapter": s * (xd[rows] @ w3[(l + 1) % max_loras].double().t()),
        }
        if S > 1:
            wrong["slices swapped"] = out[(sl + 1) % S][rows].double()
        for k, v in wrong.items():
            p = ((v - s * Sx).abs() <= tol)
            a, b = res.get(k, (0, 0))
            res[k] = (a + int(p.sum().item()), b + p.numel())
    all_rej = True
    for k, (a, b) in res.items():
        rejected = a < b
        all_rej &= rejected
        print(f"  negative [{name}] {k:26s}: {a / b:6.1%} of elements pass -> "
              f"{'rejected' if rejected else 'NOT REJECTED'}")
    del x, ws, out, meta
    torch.cuda.empty_cache()
    return all_rej


def tol_stats(M, K, R, max_loras, gen):
    """tol / |out| on signed data, one slice, scaling 1.0."""
    x, ws, mapping = make(M, K, R, max_loras, 1, 0.0, True, gen)
    r = []
    for sl, l, rows, xr, wl, Sx in per_group(x, ws, mapping, max_loras, R, K):
        tol, _ = tolerance(M, K, 1.0, 1.0, xr, wl, Sx)
        r.append((tol / Sx.abs()).flatten())
    r = torch.cat(r)
    bk, sk = config(M)
    print(f"  tol/|out| at M={M} K={K} (BK={bk}, SK={sk}), randn: median {r.median().item():.3e}, "
          f"p90 {r.quantile(0.9).item():.3e}")
    del x, ws
    torch.cuda.empty_cache()


def main():
    gen = torch.Generator().manual_seed(0)
    # (name, M, hidden, rank, max_loras, slices, -1 fraction, specialize, scaling, signed)
    covered = [
        # SETUP.md's host test cases.
        ("setup c1", 1, 4096, 16, 4, 1, 0.0, False, 0.5, False),
        ("setup c4", 128, 4096, 64, 32, 1, 0.1, False, 0.5, False),
        ("setup c5, two slices", 512, 2049, 256, 8, 2, 0.2, False, 0.5, False),
        ("setup c6, three slices (specialize)", 300, 14336, 32, 4, 3, 0.0, True, 0.5, False),
        ("setup c7", 2048, 8192, 16, 128, 1, 0.1, False, 0.5, False),
        ("setup c8", 4096, 5120, 128, 8, 1, 0.0, False, 0.5, False),
        # vLLM's layers (scaling 1.0): decode and prefill, merged and single layers, signed data.
        ("decode, qkv (3 slices)", 8, 4096, 16, 4, 3, 0.0, False, 1.0, True),
        ("decode, down_proj signed", 64, 14336, 64, 8, 1, 0.1, False, 1.0, True),
        ("prefill, gate_up (2 slices)", 2048, 4096, 64, 8, 2, 0.05, False, 1.0, True),
        ("layer, scaling 1.0", 4096, 14336, 256, 32, 1, 0.1, False, 1.0, False),
        ("layer, signed data", 1024, 14336, 64, 8, 1, 0.1, False, 1.0, True),
        # The configuration's boundary.
        ("127 tokens (BK 256, SK 64)", 127, 4096, 32, 4, 2, 0.1, False, 1.0, True),
        ("128 tokens (BK 32, SK 8)", 128, 4096, 32, 4, 2, 0.1, False, 1.0, True),
        # Edges of Inst.ok.
        ("edge: hidden 2", 37, 2, 16, 1, 1, 0.0, False, 0.5, False),
        ("edge: hidden 255 (one masked step)", 64, 255, 16, 2, 1, 0.3, True, 0.5, False),
        ("edge: hidden 2049, odd tail", 33, 2049, 48, 3, 2, 0.5, False, -3.75, True),
        ("edge: hidden 2049, M >= 128", 200, 2049, 48, 3, 3, 0.5, False, -3.75, True),
        ("scaling 0.1 (not a float32)", 256, 4096, 32, 4, 1, 0.1, False, 0.1, True),
        ("scaling 1/3 (not a float32)", 16, 2049, 16, 4, 1, 0.0, False, 1.0 / 3.0, False),
        ("scaling 2^-7 (power of two)", 128, 8192, 64, 4, 1, 0.2, False, 2.0 ** -7, True),
        ("edge: hidden 128256 (largest named)", 40, 128256, 256, 2, 1, 0.2, False, 1.0, False),
        ("edge: hidden 128256, M >= 128", 130, 128256, 64, 2, 1, 0.2, False, 1.0, True),
        ("edge: K = 2^17 (EVEN_K at SK 64)", 16, 131072, 32, 2, 1, 0.0, False, 0.5, False),
        ("edge: K = 2^17 - 1 (masked tail)", 16, 131071, 16, 2, 1, 0.0, False, 0.5, True),
        ("edge: K = 16384 (EVEN_K at SK 64)", 8, 16384, 16, 2, 2, 0.0, False, 1.0, True),
        ("edge: K = 4096, M >= 128 (EVEN_K)", 256, 4096, 16, 2, 1, 0.0, False, 1.0, True),
        ("edge: 128 adapters, G = 129", 4096, 128, 16, 128, 1, 0.05, False, 0.5, False),
        ("edge: one token, rank 256", 1, 4096, 256, 1, 3, 0.0, False, 1.0, True),
        ("edge: tokens 33, rank 16", 33, 512, 16, 1, 1, 0.0, True, 0.5, False),
    ]
    excluded = [
        ("rank 8 (setup c2)", 5, 2049, 8, 8, 1, 0.3, False, 0.5, False),
        ("rank 1 (setup c3)", 37, 2049, 1, 1, 1, 0.5, True, 0.5, False),
        ("rank 4", 64, 4096, 4, 4, 1, 0.0, False, 0.5, False),
    ]
    all_ok = True
    worst = 0.0
    for c in covered:
        ok, w = run_case(*c, gen=gen)
        all_ok &= ok
        worst = max(worst, w)
    for c in excluded:
        run_case(*c, gen=gen, covered=False)
    print("Negative tests (signed data; each must be rejected):")
    neg_ok = True
    neg_ok &= negatives("decode qkv, M=64 K=14336", 64, 14336, 32, 4, 3, 1.0, gen)
    neg_ok &= negatives("prefill gate_up, M=512 K=14336", 512, 14336, 32, 4, 2, 1.0, gen)
    neg_ok &= negatives("prefill, M=256 K=4096, scaling 0.5", 256, 4096, 64, 4, 2, 0.5, gen)
    print("Tolerance against the output:")
    tol_stats(64, 14336, 64, 4, gen)
    tol_stats(512, 14336, 64, 4, gen)
    tol_stats(64, 4096, 64, 4, gen)
    tol_stats(512, 4096, 64, 4, gen)
    tol_stats(130, 128256, 64, 2, gen)
    print(f"covered launches: {'ALL PASS' if all_ok else 'FAILURES'}; worst err/tol {worst:.3e} "
          f"(margin {1 - worst:.3f}); negatives: {'all rejected' if neg_ok else 'NOT ALL REJECTED'}")
    sys.exit(0 if all_ok and neg_ok else 1)


if __name__ == "__main__":
    main()
