"""GPU check of the prepared claim `Claims.TmaPersistentMatmul` (Claim.lean).

Runs the kernel's own host, `matmul_tma_persistent(a, b, False)` of `source/09-persistent-matmul.py`
(imported by path; the file's `__main__` test and benchmark do not run on import, only its
module-level definitions), on the launches SETUP.md lists (`M = N = K = 32`; `M = N = 8192`,
`K = 128, 256, ..., 1024`, with the file's own input draws and layout: `a` contiguous `M x K`, `b` the
contiguous `N x K` `b.T.contiguous()`, the device's `NUM_SMS`) and on launches at the edges of
`Inst.ok` (the launches with `N` or `K` not a multiple of 128, `M = N = K = 32` among them, are
outside `Inst.ok`, `N_dim`/`K_dim`, and checked as extras), and checks `Post` element by element in float64, from the host's inputs, exactly as
`Post` states it:

* for each element `(i, j)` of `c`: `S = sum_k a[i, k] * b[j, k]` (exact products, float64 sum) and
  `E = tol = errAt` over the inner loop's `T = cdiv(K, 64)` steps: `E_0 = 0`,
  `E_{t+1} = (1 + rho) E_t + rho (|S_t| + L_t) + alpha`, with `S_t` the exact partial sum of the
  products `k < 64 t`, `L_t = sum liftMag(a[i, k]) * liftMag(b[j, k])` over step `t`'s `k`
  (`liftMag q = 0` at `0`, else `max(|q|, 2^-14)`), `rho = relDot16 64 = 66 * 2^-22`,
  `alpha = absDot16 64 = 66 * 2^-126`; with `OutSpec`'s first clause, that the reals rounding to
  `c` in `float16` meet `[S - E, S + E]` (some `v` within `E` of `S` rounds to `c`, up to `v`'s
  `float32` grid); and, with room (`|S| + E <= 65504`, which every case here meets), `c` is finite
  and `|c - S| <= E + (|S| + E) * 2^-11 + 2^-25` (`OutSpec`);
* elements whose row of `a` or row of `b` holds a NaN or an infinity need only hold some `float16`
  value; the others are checked as above (one case with non-finite inputs);
* negative tests: wrong results that must each break `Post` on some element: from the same inputs,
  the exact result rounded to `bfloat16`, and the exact result with the last step's 64 products
  left out (a missing K block), rounded to `float16`; from the kernel's own output, two wrong tile
  mappings: two row blocks of 128 swapped, and the two 128-column halves of each 256-column tile
  swapped (the subtiled epilogue's stores exchanged);
* frame: the inputs are unchanged (compared with copies); and direct launches of the source's
  `matmul_kernel_tma_persistent` with the host's descriptors, grid formula and config, for several
  `NUM_SMS` (`1`, `7`, `132`, `1000`), with `a` and `c` inside guarded buffers with padded rows
  (row strides above `K` and `N`), leave every byte outside the matrix `c` unchanged.

Prints, per case, the largest error / tolerance over both clauses (below 1, with its margin) and the
frame results. Usage: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "09-persistent-matmul.py")

spec = importlib.util.spec_from_file_location("tma_persistent_matmul_src", SRC)
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

DEV = "cuda"
RHO = 66 * 2.0 ** -22           # rho = Rules.Num.relDot16 64 (exact in float64)
ALPHA = 66 * 2.0 ** -126         # alpha = Rules.Num.absDot16 64
U16 = 2.0 ** -11                 # Format.binary16.u
ETA16 = 2.0 ** -25               # Format.binary16.eta
MAX16 = 65504.0
CFG = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, EPILOGUE_SUBTILE=True,
           num_stages=4, num_warps=8)   # the config of SETUP.md (source line 501)


def lift(x):
    a = x.abs()
    return torch.where(a == 0, a, a.clamp_min(2.0 ** -14))


def rounding_interval(c):
    """The reals that round to the float16 value c (to nearest even), as [lo, hi], c finite."""
    a = c.abs()
    _, e = torch.frexp(c)
    ulp = torch.where(a >= 2.0 ** -14, torch.ldexp(torch.ones_like(c), e - 11),
                      torch.full_like(c, 2.0 ** -24))
    pow2 = (a > 2.0 ** -14) & (torch.ldexp(torch.ones_like(c), e - 1) == a)
    down = torch.where(pow2, ulp / 2, ulp)          # the spacing toward zero
    up_side = ulp / 2
    down_side = torch.where(a == 0, torch.full_like(c, 2.0 ** -25), down / 2)
    lo = torch.where(c > 0, c - down_side, c - up_side)
    hi = torch.where(c >= 0, c + up_side, c + down_side)
    lo = torch.where(a == 0, -down_side, lo)
    return lo, hi


def tol_err(ad, bd, want=False):
    """S and E = errAt(prodV, liftV, steps K) elementwise for rows `ad` (R x K) of `a` and the
    matrix `bd` (K x N, b transposed); with `want`, also the wrong results computed from the
    inputs."""
    R, K = ad.shape
    N = bd.shape[1]
    la, lb = lift(ad), lift(bd)
    S = torch.zeros((R, N), dtype=torch.float64, device=ad.device)
    E = torch.zeros_like(S)
    last = torch.zeros_like(S) if want else None
    for t in range((K + 63) // 64):
        lo, hi = 64 * t, min(64 * t + 64, K)
        c = ad[:, lo:hi] @ bd[lo:hi, :]
        L = la[:, lo:hi] @ lb[lo:hi, :]
        E = (1 + RHO) * E + RHO * (S.abs() + L) + ALPHA
        S = S + c
        if want:
            last = c
        del c, L
    if want:
        return S, E, {"bf16_output": S.to(torch.bfloat16).double(),
                      "missing_k_block": (S - last).half().double()}
    return S, E


def ratio(S, E, got):
    """The larger of the two clauses' error / tolerance, elementwise (got finite, with room)."""
    lo, hi = rounding_interval(got)
    dist = torch.clamp(torch.maximum(lo - S, S - hi), min=0)
    bound = E + (S.abs() + E) * U16 + ETA16
    r1 = torch.where(E > 0, dist / E.clamp_min(1e-300),
                     torch.where(dist > 0, torch.full_like(E, float("inf")), torch.zeros_like(E)))
    return torch.maximum(r1, (got - S).abs() / bound)


def wrong_maps(c):
    """Two wrong tile mappings of the kernel's output c (M x N): the first two row blocks of 128
    swapped (when M > 128), and the two 128-column halves of every 256-column tile swapped (when
    N > 128)."""
    M, N = c.shape
    out = {}
    if M > 128:
        w = c.clone()
        h = min(256, M) - 128
        w[0:h], w[128:128 + h] = c[128:128 + h].clone(), c[0:h].clone()
        out["tiles_swapped"] = w
    if N > 128:
        w = c.clone()
        for j0 in range(0, N, 256):
            h = min(j0 + 256, N) - (j0 + 128)
            if h > 0:
                w[:, j0:j0 + h], w[:, j0 + 128:j0 + 128 + h] = \
                    c[:, j0 + 128:j0 + 128 + h].clone(), c[:, j0:j0 + h].clone()
        out["halves_swapped"] = w
    return out


def check_out(a, b, c, rows=1024, negatives=False):
    """Largest error / tolerance of c (M x N) against Post for a (M x K) and b (N x K), elements with
    finite rows of a and b, row block by row block; with `negatives`, also the wrong results'
    largest ratios."""
    M, K = a.shape
    bd = b.double().T
    colfin = torch.isfinite(bd).all(dim=0)
    bd = torch.where(torch.isfinite(bd), bd, torch.zeros_like(bd))
    worst = 0.0
    neg = {}
    maps = wrong_maps(c) if negatives else {}
    for r in range(0, M, rows):
        ad = a[r:r + rows].double()
        rowfin = torch.isfinite(ad).all(dim=1)
        ad = torch.where(torch.isfinite(ad), ad, torch.zeros_like(ad))
        if negatives:
            S, E, bad = tol_err(ad, bd, want=True)
            for name, w in maps.items():
                bad[name] = w[r:r + rows].double()
        else:
            S, E = tol_err(ad, bd)
        fin = rowfin[:, None] & colfin[None, :]
        assert ((S.abs() + E)[fin] <= MAX16).all(), "no room in float16"
        got = c[r:r + rows].double()
        assert torch.isfinite(got[fin]).all()
        g = torch.where(fin, got, S)
        rr = torch.where(fin, ratio(S, E, g), torch.zeros_like(S))
        if fin.any():
            worst = max(worst, rr.max().item())
        if negatives:
            for name, x in bad.items():
                neg[name] = max(neg.get(name, 0.0), ratio(S, E, x).max().item())
            del bad
        del ad, S, E, got, g, rr
    return (worst, neg) if negatives else worst


def host_case(name, M, N, K, seed=0, scale=1.0, negatives=False, nonfinite=False):
    """`matmul_tma_persistent(a, b, False)` as the file's `validate` and `bench` call it: `a`, `b`
    drawn with `torch.randn` in `float16`, `b` made the contiguous `N x K` `b.T.contiguous()`."""
    torch.manual_seed(seed)
    a = torch.randn((M, K), device=DEV, dtype=torch.float16)
    b = torch.randn((K, N), device=DEV, dtype=torch.float16)
    if scale != 1.0:
        a = (a.float() * scale).half()
        b = (b.float() * scale).half()
    b = b.T.contiguous()
    if nonfinite:   # a NaN, +inf and -inf in rows of a, an inf in a row of b
        a[3, 5] = float("nan")
        a[7, 0] = float("inf")
        a[M - 1, K - 1] = float("-inf")
        b[11, K // 2] = float("inf")
    a0, b0 = a.clone(), b.clone()
    c = mm.matmul_tma_persistent(a, b, False)
    torch.cuda.synchronize()
    same = torch.equal(a.view(torch.int16), a0.view(torch.int16)) and \
        torch.equal(b.view(torch.int16), b0.view(torch.int16))
    ok = same
    extra = ""
    if negatives:
        w, neg = check_out(a, b, c, negatives=True)
        extra = ", wrong results rejected: " + ", ".join(
            f"{k} {v:.2f} {'yes' if v > 1 else 'NO'}" for k, v in neg.items())
        ok &= all(v > 1 for v in neg.values())
    else:
        w = check_out(a, b, c)
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    print(f"{name}: M={M} N={N} K={K} NUM_SMS={sms} (randn{'' if scale == 1.0 else f' x{scale}'}"
          f"{', non-finite inputs' if nonfinite else ''}): max err/tol {w:.4f} (margin {1 - w:.4f}), "
          f"inputs unchanged {same}{extra}", flush=True)
    del a, b, c, a0, b0
    torch.cuda.empty_cache()
    return w < 1 and ok, w


def cdiv(x, y):
    return (x + y - 1) // y


def direct_case(name, M, N, K, num_sms, pad_a=24, pad_c=40, seed=1):
    """A direct launch of `matmul_kernel_tma_persistent` with the host's grid formula
    `min(NUM_SMS, tiles)`, config and host descriptors (`TensorDescriptor.from_tensor` with the
    pre-hook's block shapes): `a` the first `K` columns of an `M x (K + pad_a)` buffer (row stride
    `K + pad_a`), `b` contiguous `N x K`, and `c` the first `N` columns of rows of `N + pad_c`
    elements inside a guarded buffer."""
    torch.manual_seed(seed)
    G = 4096
    lda, ldc = K + pad_a, N + pad_c
    abuf = torch.randn((M, lda), device=DEV, dtype=torch.float16)
    a = abuf[:, :K]
    b = torch.randn((N, K), device=DEV, dtype=torch.float16)
    store = torch.empty(G + M * ldc + G, device=DEV, dtype=torch.float16)
    c = store[G:G + M * ldc].view(M, ldc)[:, :N]
    PAT = -12345
    store.view(torch.int16).fill_(PAT)
    inside = torch.zeros(store.numel(), dtype=torch.bool, device=DEV)
    idx = (torch.arange(M, device=DEV)[:, None] * ldc + torch.arange(N, device=DEV)[None, :] + G)
    inside[idx.reshape(-1)] = True
    abuf0, b0 = abuf.clone(), b.clone()
    a_desc = mm.TensorDescriptor.from_tensor(a, [128, 64])
    b_desc = mm.TensorDescriptor.from_tensor(b, [256, 64])
    c_desc = mm.TensorDescriptor.from_tensor(c, [128, 128])
    tiles = cdiv(M, 128) * cdiv(N, 256)
    grid = (min(num_sms, tiles),)
    mm.matmul_kernel_tma_persistent[grid](a_desc, b_desc, c_desc, M, N, K, FP8_OUTPUT=False,
                                          NUM_SMS=num_sms, WARP_SPECIALIZE=False, **CFG)
    torch.cuda.synchronize()
    bits = store.view(torch.int16)
    frame = bool((bits[~inside] == PAT).all().item())
    same = torch.equal(abuf.view(torch.int16), abuf0.view(torch.int16)) and torch.equal(b, b0)
    w = check_out(a, b, c)
    print(f"{name}: M={M} N={N} K={K} NUM_SMS={num_sms} (grid {grid[0]}, {tiles} tiles), lda={lda}, "
          f"ldc={ldc}: max err/tol {w:.4f} (margin {1 - w:.4f}), bytes outside the matrix c unchanged "
          f"{frame}, inputs unchanged {same}", flush=True)
    return w < 1 and frame and same, w


def main():
    ok, worst = True, 0.0
    neg = {128, 512, 1024}
    cases = [("v32_outside_Inst.ok", 32, 32, 32, 0, 1.0, True)]  # N, K not multiples of 128
    cases += [(f"setup_k{k}", 8192, 8192, k, 0, 1.0, k in neg) for k in range(128, 1025, 128)]
    cases += [
        # edges of Inst.ok: the largest covered K (2^13), the smallest sizes, partial tiles in every
        # dimension, a second 128-column half wholly past N, larger and subnormal inputs, non-finite
        # inputs; the cases with N or K not a multiple of 128 are outside Inst.ok (N_dim, K_dim),
        # checked as extras
        ("edge_k8192", 256, 512, 8192),
        ("edge_m1_n8_k8", 1, 8, 8),
        ("edge_partial_1000", 1000, 1000, 1000, 2, 1.0, True),
        ("edge_n96_half_outside", 300, 96, 200),
        ("edge_n136", 129, 136, 72),
        ("edge_scale8", 512, 512, 2048, 3, 8.0),
        ("edge_subnormal", 384, 512, 1024, 4, 2.0 ** -12),
        ("edge_nonfinite", 640, 768, 704, 5, 1.0, False, True),
    ]
    for c in cases:
        r, w = host_case(*c)
        ok &= r
        worst = max(worst, w)
    directs = [  # those with K not a multiple of 128 are outside Inst.ok (K_dim), checked as extras
        ("direct_sms1", 384, 768, 776, 1),        # one program, 9 tiles
        ("direct_sms7", 1408, 760, 1000, 7),      # 33 tiles over 7 programs
        ("direct_sms132", 2816, 2560, 304, 132),  # 220 tiles over 132
        ("direct_sms1000", 1280, 1024, 512, 1000),  # 40 tiles, grid 40
    ]
    for c in directs:
        r, w = direct_case(*c)
        ok &= r
        worst = max(worst, w)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"overall: max err/tol {worst:.4f} (margin {1 - worst:.4f}); peak device memory {peak:.0f} MiB; "
          f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
