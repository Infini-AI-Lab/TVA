"""GPU check of the claim `Claims.Matmul` (Claim.lean).

Runs the kernel's own host, `matmul(a, b)` of `source/03-matrix-multiplication.py` (imported by path;
the file's module-level test and benchmark are not executed on import, only its imports, its
`DEVICE` and its functions), on the launches SETUP.md lists (the test, `512 x 512 x 512`, and the
benchmark's `fp16` line, `M = N = K = 256, 384, ..., 4096`, with the file's own input draws), on
launches at the edges of `Inst.ok`, and on launches beyond it (`beyond_*`: `N` or `K` not a positive
multiple of 128, which `Inst.ok` leaves out of scope; they check the kernel past the claim), and
checks `Post` element by element in float64, from the host's inputs, exactly as `Post` states it:

* for each element `(i, j)` of `c`: `S = sum_k a[i, k] * b[k, j]` (exact products, float64 sum),
  `M = pathMag = sum_{t < T} (|S_t| + L_t)` over the loop's `T = cdiv(K, 64)` steps, with `S_t` the
  exact partial sum of the products `k < 64 t` and `L_t = sum liftMag(a[i, k]) * liftMag(b[k, j])`
  over step `t`'s `k` (`liftMag q = 0` at `0`, else `max(|q|, 2^-14)`), and `E = tol =
  2^-15 M + 2^-100`; with `OutSpec`'s first clause, that the reals rounding to `c` in `float16` meet
  `[S - E, S + E]` (some `v` within `E` of `S` rounds to `c`, up to `v`'s `float32` grid); and, with
  room (`|S| + E <= 65504`, which every case here meets), `c` is finite and
  `|c - S| <= 2^-11 |S| + 2^-15 M + 2^-24` (`OutSpec`);
* elements whose row of `a` or column of `b` holds a NaN or an infinity need only hold some
  `float16` value; the others are checked as above (one case with non-finite inputs);
* negative tests: three wrong results computed from the same inputs (the exact result rounded to
  `bfloat16`; a `float16` accumulator rounded after each step of 64 products; the exact result with
  the last step's products left out, rounded to `float16`) must each break `Post` on some element;
* frame: the inputs are unchanged (compared with copies); and direct launches of the source's
  `matmul_kernel` with the host's launch parameters into a `c` placed inside a guarded buffer
  (padded rows, and a column-major layout) leave every byte outside the matrix `c` unchanged.

Prints, per case, the largest error / tolerance over both clauses (below 1, with its margin), the
median of tol / |S| (elements with S != 0) and the frame results, then the largest over the launches
`Inst.ok` covers and over all launches.
Usage: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import ast
import importlib.machinery
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "03-matrix-multiplication.py")


class _DefsOnly(importlib.machinery.SourceFileLoader):
    """Loads the source keeping only its imports, the `DEVICE` assignment and its undecorated or
    `@triton.jit` functions, at their own line numbers (the file runs its test and benchmark at
    module level)."""

    def get_code(self, fullname):
        tree = ast.parse(open(self.path).read(), self.path)
        keep = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                keep.append(node)
            elif isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "DEVICE" for t in node.targets):
                keep.append(node)
            elif isinstance(node, ast.FunctionDef) and all(
                    ast.unparse(dec) == "triton.jit" for dec in node.decorator_list):
                keep.append(node)
        tree.body = keep
        return compile(tree, self.path, "exec", dont_inherit=True)


_loader = _DefsOnly("matmul_src", SRC)
spec = importlib.util.spec_from_file_location("matmul_src", SRC, loader=_loader)
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

DEV = "cuda"
EPS = 2.0 ** -15                 # tol = EPS * pathMag + DELTA
DELTA = 2.0 ** -100
U16 = 2.0 ** -11                 # the second clause: 2^-11 |S| + 2^-15 M + 2^-24
DELTA_C = 2.0 ** -24
MAX16 = 65504.0


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
    """S, M = pathMag and E = tol = 2^-15 M + 2^-100 elementwise; with `want`, also the wrong
    results of the negative tests."""
    R, K = ad.shape
    N = bd.shape[1]
    la, lb = lift(ad), lift(bd)
    S = torch.zeros((R, N), dtype=torch.float64, device=ad.device)
    Mg = torch.zeros_like(S)
    acc16 = torch.zeros_like(S)
    last = torch.zeros_like(S)
    for t in range((K + 63) // 64):
        lo, hi = 64 * t, min(64 * t + 64, K)
        c = ad[:, lo:hi] @ bd[lo:hi, :]
        L = la[:, lo:hi] @ lb[lo:hi, :]
        Mg = Mg + S.abs() + L
        S = S + c
        if want:
            acc16 = (acc16 + c).half().double()
            last = c
    E = EPS * Mg + DELTA
    if want:
        return S, E, Mg, {"bf16_output": S.to(torch.bfloat16).double(), "fp16_accumulator": acc16,
                      "dropped_last_step": (S - last).half().double()}
    return S, E, Mg


def ratio(S, E, Mg, got):
    """The larger of the two clauses' error / tolerance, elementwise (got finite, with room)."""
    lo, hi = rounding_interval(got)
    dist = torch.clamp(torch.maximum(lo - S, S - hi), min=0)
    bound = S.abs() * U16 + EPS * Mg + DELTA_C
    r1 = dist / E
    return torch.maximum(r1, (got - S).abs() / bound)


def check_out(a, b, c, rows=1024, negatives=False):
    """Largest error / tolerance of c against Post (elements with finite rows and columns), row
    block by row block; with `negatives`, also the smallest of the wrong results' largest ratios."""
    M, K = a.shape
    bd = b.double()
    colfin = torch.isfinite(bd).all(dim=0)
    bd = torch.where(torch.isfinite(bd), bd, torch.zeros_like(bd))
    worst = 0.0
    neg = {}
    rel = []
    for r in range(0, M, rows):
        ad = a[r:r + rows].double()
        rowfin = torch.isfinite(ad).all(dim=1)
        ad = torch.where(torch.isfinite(ad), ad, torch.zeros_like(ad))
        if negatives:
            S, E, Mg, bad = tol_err(ad, bd, want=True)
        else:
            S, E, Mg = tol_err(ad, bd)
        fin = rowfin[:, None] & colfin[None, :]
        assert ((S.abs() + E)[fin] <= MAX16).all(), "no room in float16"
        got = c[r:r + rows].double()
        assert torch.isfinite(got[fin]).all()
        g = torch.where(fin, got, S)
        rr = torch.where(fin, ratio(S, E, Mg, g), torch.zeros_like(S))
        if fin.any():
            worst = max(worst, rr.max().item())
        nz = fin & (S != 0)
        rel.append((E[nz] / S[nz].abs()).float().cpu())
        if negatives:
            for name, x in bad.items():
                neg[name] = max(neg.get(name, 0.0), ratio(S, E, Mg, x).max().item())
        del ad, S, E, Mg, got, g, rr
    rel = torch.cat(rel)
    med = rel.median().item() if rel.numel() else float("nan")
    return (worst, med, neg) if negatives else (worst, med)


def host_case(name, M, N, K, draw, seed=0, scale=1.0, negatives=False, nonfinite=False):
    torch.manual_seed(seed)
    if draw == "rand":    # the test: torch.rand(...) - 0.5
        a = torch.rand((M, K), device=DEV, dtype=torch.float16) - 0.5
        b = torch.rand((K, N), device=DEV, dtype=torch.float16) - 0.5
    else:                 # the benchmark: torch.randn
        a = torch.randn((M, K), device=DEV, dtype=torch.float16)
        b = torch.randn((K, N), device=DEV, dtype=torch.float16)
    if scale != 1.0:
        a = (a.float() * scale).half()
        b = (b.float() * scale).half()
    if nonfinite:   # a NaN, +inf and -inf in rows of a, an inf in a column of b
        a[3, 5] = float("nan")
        a[7, 0] = float("inf")
        a[M - 1, K - 1] = float("-inf")
        b[K // 2, 11] = float("inf")
    a0, b0 = a.clone(), b.clone()
    c = mm.matmul(a, b)
    torch.cuda.synchronize()
    same = torch.equal(a.view(torch.int16), a0.view(torch.int16)) and \
        torch.equal(b.view(torch.int16), b0.view(torch.int16))
    ok = same
    extra = ""
    if M * N == 0:
        w, med = 0.0, float("nan")
        extra = f", c has {c.numel()} elements"
    elif negatives:
        w, med, neg = check_out(a, b, c, negatives=True)
        extra = ", wrong results rejected: " + ", ".join(
            f"{k} {v:.2f} {'yes' if v > 1 else 'NO'}" for k, v in neg.items())
        ok &= all(v > 1 for v in neg.values())
    else:
        w, med = check_out(a, b, c)
    print(f"{name}: M={M} N={N} K={K} ({draw}{'' if scale == 1.0 else f' x{scale}'}"
          f"{', non-finite inputs' if nonfinite else ''}): max err/tol {w:.4f} (margin {1 - w:.4f}), "
          f"median tol/|S| {med:.2e}, "
          f"inputs unchanged {same}{extra}")
    del a, b, c, a0, b0
    torch.cuda.empty_cache()
    return w < 1 and ok, w


def direct_case(name, M, N, K, layout, seed=1):
    """A direct launch of `matmul_kernel` with the host's launch parameters, `c` inside a guarded
    buffer: `layout` "padded" (rows of `N + 13` elements) or "colmajor"; `b` non-contiguous
    (the transpose of a contiguous `N x K`)."""
    torch.manual_seed(seed)
    a = torch.randn((M, K), device=DEV, dtype=torch.float16)
    b = torch.randn((N, K), device=DEV, dtype=torch.float16).t()
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
    grid = (triton_cdiv(M, 128) * triton_cdiv(N, 256),)
    mm.matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1), BLOCK_SIZE_M=128, BLOCK_SIZE_N=256,
                           BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3, num_warps=8,
                           ACTIVATION="")
    torch.cuda.synchronize()
    bits = store.view(torch.int16)
    frame = bool((bits[~inside] == PAT).all().item())
    same = torch.equal(a, a0) and torch.equal(b, b0)
    w, _ = check_out(a, b, c)
    print(f"{name}: M={M} N={N} K={K} c {layout}, b strides {tuple(b.stride())}: max err/tol "
          f"{w:.4f} (margin {1 - w:.4f}), bytes outside the matrix c unchanged {frame}, "
          f"inputs unchanged {same}")
    return w < 1 and frame and same, w


def triton_cdiv(x, y):
    return (x + y - 1) // y


def main():
    ok, worst = True, 0.0
    neg = {512, 1024, 2048, 4096}
    cases = [("setup_test_512", 512, 512, 512, "rand", 0, 1.0, True)]
    cases += [(f"setup_bench_{n}", n, n, n, "randn", 0, 1.0, n in neg)
              for n in [128 * i for i in range(2, 33)]]
    cases += [
        # edges of Inst.ok: the largest covered K (2^13), M not a multiple of the block, N half a
        # column tile, larger and subnormal inputs, non-finite inputs, M = 0
        ("edge_k8192", 640, 768, 8192, "randn"),
        ("edge_scale8", 512, 512, 2048, "randn", 3, 8.0),
        ("edge_subnormal", 384, 512, 1024, "randn", 4, 2.0 ** -12),
        ("edge_nonfinite", 300, 640, 768, "randn", 5, 1.0, False, True),
        ("edge_m0", 0, 384, 256, "randn"),
        # beyond Inst.ok (N or K not a positive multiple of 128): K not a multiple of 64, N not a
        # multiple of 128, one row / one column, K = 1, K = 0
        ("beyond_m1000_n777_k4095", 1000, 777, 4095, "randn"),
        ("beyond_m1_n1_k65", 1, 1, 65, "randn"),
        ("beyond_m129_n257_k1", 129, 257, 1, "randn"),
        ("beyond_nonfinite", 300, 500, 700, "randn", 5, 1.0, False, True),
        ("beyond_k0", 200, 300, 0, "randn"),
    ]
    worst_in = 0.0
    for c in cases:
        r, w = host_case(*c)
        ok &= r
        worst = max(worst, w)
        if not c[0].startswith("beyond_"):
            worst_in = max(worst_in, w)
    for c in [("direct_padded_c", 300, 1024, 768, "padded"), ("direct_colmajor_c", 513, 384, 1024, "colmajor"),
              ("beyond_direct_padded_c", 300, 1000, 777, "padded"),
              ("beyond_direct_colmajor_c", 513, 259, 1000, "colmajor")]:
        r, w = direct_case(*c)
        ok &= r
        worst = max(worst, w)
        if not c[0].startswith("beyond_"):
            worst_in = max(worst_in, w)
    print(f"covered launches (Inst.ok): max err/tol {worst_in:.4f} (margin {1 - worst_in:.4f})")
    print(f"overall: max err/tol {worst:.4f} (margin {1 - worst:.4f}); {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
