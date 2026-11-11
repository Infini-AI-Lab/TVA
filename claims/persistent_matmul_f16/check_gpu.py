"""GPU check of the claim `Claims.PersistentMatmul` (Claim.lean).

Runs the kernel's own host, `matmul_persistent(a, b)` of `source/09-persistent-matmul.py` (imported
by path; the file's `__main__` test and benchmark do not run on import, only its module-level
definitions), on the launches SETUP.md lists (`M = N = 8192`, `K = 128, 256, ..., 1024`, with the
file's own input draws and layout: `a` contiguous, `b` the transpose of a contiguous `N x K`, the
device's `NUM_SMS`) and on launches at the edges of `Inst.ok`, and checks `Post` element by element
in float64, from the host's inputs, exactly as `Post` states it:

* for each element `(i, j)` of `c`: `S = sum_k a[i, k] * b[k, j]` (exact products, float64 sum) and
  `E = tol = 2^-15 * accMag + 2^-112`, `accMag = sum_{t < T} (|S_t| + L_t)` over the inner loop's
  `T = cdiv(K, 64)` steps, with `S_t` the exact partial sum of the products `k < 64 t` and
  `L_t = sum liftMag(a[i, k]) * liftMag(b[k, j])` over step `t`'s `k` (`liftMag q = 0` at `0`, else
  `max(|q|, 2^-14)`); with `OutSpec`'s first clause, that the reals rounding to
  `c` in `float16` meet `[S - E, S + E]` (some `v` within `E` of `S` rounds to `c`, up to `v`'s
  `float32` grid); and, with room (`|S| + E <= 65504`, which every case here meets), `c` is finite
  and `|c - S| <= E + (|S| + E) * 2^-11 + 2^-25` (`OutSpec`);
* elements whose row of `a` or column of `b` holds a NaN or an infinity need only hold some
  `float16` value; the others are checked as above (one case with non-finite inputs);
* negative tests: three wrong results computed from the same inputs (the exact result rounded to
  `bfloat16`; a `float16` accumulator rounded after each step of 64 products; the exact result with
  the last step's products left out, rounded to `float16`) must each break `Post` on some element;
* frame: the inputs are unchanged (compared with copies); and direct launches of the source's
  `matmul_kernel_persistent` with the host's grid formula and config, for several `NUM_SMS`
  (`1`, `7`, `132`, `1000`), into a `c` placed inside a guarded buffer (padded rows, and a
  column-major layout) leave every byte outside the matrix `c` unchanged.

Prints, per case, the largest error / tolerance over both clauses (below 1, with its margin) and the
frame results; on the setup's launches also `tol / |c|` (median, worst) and the same for the fine
bound of README Tolerances (`errAt`: `E_0 = 0`, `E_{t+1} = (1 + rho) E_t + rho (|S_t| + L_t) +
alpha`, `rho = 66 * 2^-22`, `alpha = 66 * 2^-126`), which `Post` does not use. Usage: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "09-persistent-matmul.py")

spec = importlib.util.spec_from_file_location("persistent_matmul_src", SRC)
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

DEV = "cuda"
RHO = 66 * 2.0 ** -22           # rho = Device.relDot16 64 (the fine bound only)
ALPHA = 66 * 2.0 ** -126         # alpha = Device.absDot16 64 (the fine bound only)
U16 = 2.0 ** -11                 # Format.binary16.u
ETA16 = 2.0 ** -25               # Format.binary16.eta
MAX16 = 65504.0
CFG = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3,
           num_warps=8)          # the config of SETUP.md (source line 378)


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


def tol_err(ad, bd, want=None):
    """S, E = tol = 2^-15 * accMag + 2^-112 and the fine bound errAt elementwise; with `want`,
    also the wrong results of the negative tests."""
    R, K = ad.shape
    N = bd.shape[1]
    la, lb = lift(ad), lift(bd)
    S = torch.zeros((R, N), dtype=torch.float64, device=ad.device)
    E = torch.zeros_like(S)
    Mag = torch.zeros_like(S)
    acc16 = torch.zeros_like(S) if want else None
    last = torch.zeros_like(S) if want else None
    for t in range((K + 63) // 64):
        lo, hi = 64 * t, min(64 * t + 64, K)
        c = ad[:, lo:hi] @ bd[lo:hi, :]
        L = la[:, lo:hi] @ lb[lo:hi, :]
        E = (1 + RHO) * E + RHO * (S.abs() + L) + ALPHA
        Mag += S.abs() + L
        S = S + c
        if want:
            acc16 = (acc16 + c).half().double()
            last = c
        del c, L
    tol = 2.0 ** -15 * Mag + 2.0 ** -112
    if want:
        return S, tol, E, {"bf16_output": S.to(torch.bfloat16).double(), "fp16_accumulator": acc16,
                      "dropped_last_step": (S - last).half().double()}
    return S, tol, E


def ratio(S, E, got):
    """The larger of the two clauses' error / tolerance, elementwise (got finite, with room)."""
    lo, hi = rounding_interval(got)
    dist = torch.clamp(torch.maximum(lo - S, S - hi), min=0)
    bound = E + (S.abs() + E) * U16 + ETA16
    # E = 0 only at K = 0, where the clause asks for c = 0 exactly (dist = 0)
    r1 = torch.where(E > 0, dist / E.clamp_min(1e-300),
                     torch.where(dist > 0, torch.full_like(E, float("inf")), torch.zeros_like(E)))
    return torch.maximum(r1, (got - S).abs() / bound)


STATS = {"tol": [], "fine": [], "tol_max": 0.0, "fine_max": 0.0}


def check_out(a, b, c, rows=1024, negatives=False):
    """Largest error / tolerance of c against Post (elements with finite rows and columns), row
    block by row block; with `negatives`, also the wrong results' largest ratios."""
    M, K = a.shape
    bd = b.double()
    colfin = torch.isfinite(bd).all(dim=0)
    bd = torch.where(torch.isfinite(bd), bd, torch.zeros_like(bd))
    worst = 0.0
    neg = {}
    for r in range(0, M, rows):
        ad = a[r:r + rows].double()
        rowfin = torch.isfinite(ad).all(dim=1)
        ad = torch.where(torch.isfinite(ad), ad, torch.zeros_like(ad))
        if negatives:
            S, E, F, bad = tol_err(ad, bd, want=True)
        else:
            S, E, F = tol_err(ad, bd)
        fin = rowfin[:, None] & colfin[None, :]
        assert ((S.abs() + E)[fin] <= MAX16).all(), "no room in float16"
        got = c[r:r + rows].double()
        assert torch.isfinite(got[fin]).all()
        g = torch.where(fin, got, S)
        rr = torch.where(fin, ratio(S, E, g), torch.zeros_like(S))
        if fin.any():
            worst = max(worst, rr.max().item())
            nz = fin & (got != 0)
            for key, T in (("tol", E), ("fine", F)) if nz.any() else ():
                q = (T / got.abs())[nz]
                STATS[key].append(q[::97].float().cpu())
                STATS[key + "_max"] = max(STATS[key + "_max"], q.max().item())
        if negatives:
            for name, x in bad.items():
                neg[name] = max(neg.get(name, 0.0), ratio(S, E, x).max().item())
            del bad
        del ad, S, E, F, got, g, rr
    return (worst, neg) if negatives else worst


def host_case(name, M, N, K, seed=0, scale=1.0, negatives=False, nonfinite=False):
    """`matmul_persistent(a, b.T)` as the file's `validate` and `bench` call it: `a`, `b` drawn
    with `torch.randn` in `float16`, `b` made the contiguous `N x K` `b.T.contiguous()`."""
    torch.manual_seed(seed)
    a = torch.randn((M, K), device=DEV, dtype=torch.float16)
    b = torch.randn((K, N), device=DEV, dtype=torch.float16)
    if scale != 1.0:
        a = (a.float() * scale).half()
        b = (b.float() * scale).half()
    b = b.T.contiguous()
    if nonfinite:   # a NaN, +inf and -inf in rows of a, an inf in a column of b.T
        a[3, 5] = float("nan")
        a[7, 0] = float("inf")
        a[M - 1, K - 1] = float("-inf")
        b[11, K // 2] = float("inf")
    a0, b0 = a.clone(), b.clone()
    c = mm.matmul_persistent(a, b.T)
    torch.cuda.synchronize()
    same = torch.equal(a.view(torch.int16), a0.view(torch.int16)) and \
        torch.equal(b.view(torch.int16), b0.view(torch.int16))
    ok = same
    extra = ""
    if M * N == 0:
        w = 0.0
        extra = f", c has {c.numel()} elements"
    elif negatives:
        w, neg = check_out(a, b.T, c, negatives=True)
        extra = ", wrong results rejected: " + ", ".join(
            f"{k} {v:.2f} {'yes' if v > 1 else 'NO'}" for k, v in neg.items())
        ok &= all(v > 1 for v in neg.values())
    else:
        w = check_out(a, b.T, c)
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    st = ""
    if name.startswith("setup"):
        st = "; tol/|c| " + ", ".join(
            f"{k} median {torch.cat(STATS[k]).median().item():.2e} worst {STATS[k + '_max']:.2e}"
            for k in ("tol", "fine"))
    for k in ("tol", "fine"):
        STATS[k], STATS[k + "_max"] = [], 0.0
    print(f"{name}: M={M} N={N} K={K} NUM_SMS={sms} (randn{'' if scale == 1.0 else f' x{scale}'}"
          f"{', non-finite inputs' if nonfinite else ''}): max err/tol {w:.4f} (margin {1 - w:.4f}), "
          f"inputs unchanged {same}{extra}{st}", flush=True)
    del a, b, c, a0, b0
    torch.cuda.empty_cache()
    return w < 1 and ok, w


def cdiv(x, y):
    return (x + y - 1) // y


def direct_case(name, M, N, K, layout, num_sms, seed=1):
    """A direct launch of `matmul_kernel_persistent` with the host's grid formula
    `min(NUM_SMS, tiles)` and config, `c` inside a guarded buffer: `layout` "padded" (rows of
    `N + 13` elements) or "colmajor"; `a` the transpose of a contiguous `K x M` (strides `(1, M)`),
    `b` contiguous `K x N` (strides `(N, 1)`)."""
    torch.manual_seed(seed)
    a = torch.randn((K, M), device=DEV, dtype=torch.float16).t()
    b = torch.randn((K, N), device=DEV, dtype=torch.float16)
    G = 4096
    if layout == "padded":
        ld = N + 13
        store = torch.empty(G + M * ld + G, device=DEV, dtype=torch.float16)
        c = store[G:G + M * ld].view(M, ld)[:, :N]
    else:
        store = torch.empty(G + M * N + G, device=DEV, dtype=torch.float16)
        c = store[G:G + M * N].view(N, M).t()
    PAT = -12345
    store.view(torch.int16).fill_(PAT)
    inside = torch.zeros(store.numel(), dtype=torch.bool, device=DEV)
    idx = (torch.arange(M, device=DEV)[:, None] * c.stride(0)
           + torch.arange(N, device=DEV)[None, :] * c.stride(1) + c.storage_offset())
    inside[idx.reshape(-1)] = True
    a0, b0 = a.clone(), b.clone()
    tiles = cdiv(M, 128) * cdiv(N, 256)
    grid = (min(num_sms, tiles),)
    mm.matmul_kernel_persistent[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                      b.stride(1), c.stride(0), c.stride(1), NUM_SMS=num_sms, **CFG)
    torch.cuda.synchronize()
    bits = store.view(torch.int16)
    frame = bool((bits[~inside] == PAT).all().item())
    same = torch.equal(a, a0) and torch.equal(b, b0)
    w = check_out(a, b, c)
    print(f"{name}: M={M} N={N} K={K} NUM_SMS={num_sms} (grid {grid[0]}, {tiles} tiles) c {layout}, "
          f"a strides {tuple(a.stride())}: max err/tol {w:.4f} (margin {1 - w:.4f}), bytes outside "
          f"the matrix c unchanged {frame}, inputs unchanged {same}", flush=True)
    return w < 1 and frame and same, w


def main():
    ok, worst = True, 0.0
    neg = {128, 512, 1024}
    cases = [(f"setup_k{k}", 8192, 8192, k, 0, 1.0, k in neg) for k in range(128, 1025, 128)]
    cases += [
        # edges of Inst.ok: the largest covered K (2^13), K not a multiple of 64, one tile, K = 1,
        # larger and subnormal inputs, non-finite inputs, K = 0, M = 0
        ("edge_k8192", 256, 512, 8192),
        ("edge_k4095", 1408, 768, 4095),
        ("edge_one_tile_k65", 128, 256, 65),
        ("edge_k1", 384, 512, 1),
        ("edge_scale8", 512, 512, 2048, 3, 8.0),
        ("edge_subnormal", 384, 512, 1024, 4, 2.0 ** -12),
        ("edge_nonfinite", 640, 768, 700, 5, 1.0, False, True),
        ("edge_k0", 256, 512, 0),
        ("edge_m0", 0, 512, 256),
    ]
    for c in cases:
        r, w = host_case(*c)
        ok &= r
        worst = max(worst, w)
    directs = [
        ("direct_padded_c_sms1", 384, 768, 777, "padded", 1),          # one program, 9 tiles
        ("direct_colmajor_c_sms7", 1408, 768, 1000, "colmajor", 7),    # 33 tiles over 7 programs
        ("direct_padded_c_sms132", 2816, 2560, 300, "padded", 132),     # 220 tiles over 132
        ("direct_colmajor_c_sms1000", 1280, 1024, 512, "colmajor", 1000),  # 40 tiles, grid 40
    ]
    for c in directs:
        r, w = direct_case(*c)
        ok &= r
        worst = max(worst, w)
    print(f"overall: max err/tol {worst:.4f} (margin {1 - worst:.4f}); {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
