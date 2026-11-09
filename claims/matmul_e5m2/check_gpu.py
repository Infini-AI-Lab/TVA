"""GPU check of the prepared claim `Claims.MatmulE5m2` (Claim.lean).

Runs the kernel's own host, `matmul(a, b)` of `source/03-matrix-multiplication.py` (imported by path;
the file's module-level test and benchmark are not executed on import, only its imports, its
`DEVICE` and its functions), on the launches SETUP.md lists (the file's `float8e5` run: the test, `512 x 512 x 512`, and the
benchmark's `fp8` line at `M = N = K = 256, 384, ..., 2048`, with the file's own input draws: `randn`
in `float16` cast to `float8_e5m2`, `b` the transpose of a contiguous matrix) and on
edge launches (inside `Inst.ok`, and outside its real-shape conditions `N_dim`, `K_dim`), and checks `Post` element by element in float64, from the
host's inputs, exactly as `Post` states it:

* for each element `(i, j)` of `c`: `S = sum_k a[i, k] * b[k, j]` (exact products, float64 sum) and
  `E = tol a_i b_j K = errAt` over the loop's `n = 2 * cdiv(K, 64)` chunks of 32 products
  (products from `K` on are `0`): `E_0 = 0`, `E_{j+1} = (1 + sigma) E_j + min(rho (X_j + E_j),
  kappa A_j + rho' (X_j + E_j)) + sigma |S_{j+1}| + alpha`, with `S_j` the exact partial sum before
  chunk `j`, `X_j = max(|S_j|, m_j)`, `m_j` the chunk's largest `liftMag(a_k) * liftMag(b_k)`
  (`liftMag q = 0` at `0`, else `max(|q|, 2^-14)`), `A_j` the chunk's `sum |a_k * b_k|`, and
  `rho = 2^-7`, `sigma = 2^-12`, `alpha = 2^-126`, `kappa = 2`, `rho' = 2^-12`; with room
  (`|S| + E <= 65504`, which every case here meets), `OutSpec`'s first clause (the reals rounding to
  `c` in `float16` meet `[S - E, S + E]`, up to `v`'s `float32` grid) and `c` finite with
  `|c - S| <= E + (|S| + E) * 2^-11 + 2^-25` (`OutSpec`);
* frame: the inputs are unchanged (compared with copies); and direct launches of the source's
  `matmul_kernel` with the host's launch parameters into a `c` placed inside a guarded buffer
  (padded rows, and a column-major layout) leave every byte outside the matrix `c` unchanged.

Coherent data (`coherent_case`, `K = 512, 1024, 2048`, and `4096`, outside `Inst.ok`, printed
only): one large product first, then products of one
sign below the accumulator's window; `Post` is checked and the device's own error `|c - S| / |S|` is
printed next to `E / |S|`.

Negative tests: on the file's draws at `K = 512, 1024, 2048`, wrong results (all zero, the first step
left out, the last product left out, `0.9` times the exact result, another column's result, `b`'s rows
shifted by one along `K`), each rounded to `float16`, must each break `Post` on some element.

Prints, per case, the largest error / tolerance over both clauses (below 1, with its margin) and the
frame results, and per negative test the share of elements that break `Post`.
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
U16 = 2.0 ** -11                 # Format.binary16.u
ETA16 = 2.0 ** -25               # Format.binary16.eta
MAX16 = 65504.0
RHO8 = 2.0 ** -7                 # Rules.Num.rhoChunk8
SIG8 = 2.0 ** -12                # Rules.Num.sigmaChunk8
ALP8 = 2.0 ** -126               # Rules.Num.alphaChunk8
KAP8 = 2.0                       # Rules.Num.kappaChunk8
RHOL8 = 2.0 ** -12               # Rules.Num.rhoLossChunk8
F8 = torch.float8_e5m2


def lift(x):
    """Rules.Num.liftMag .f8e5 on float64 values of float8e5 numbers."""
    a = x.abs()
    return torch.where(a == 0, a, a.clamp_min(2.0 ** -14))


def tol_err(ad, bd):
    """errAt (prodV a b K) (liftV a b K) (2 * steps K), elementwise, and the exact sum S."""
    R, K = ad.shape
    N = bd.shape[1]
    n = 2 * ((K + 63) // 64)
    la, lb = lift(ad), lift(bd)
    S = torch.zeros((R, N), dtype=torch.float64, device=ad.device)
    E = torch.zeros_like(S)
    for j in range(n):
        lo, hi = 32 * j, min(32 * j + 32, K)
        if lo < hi:
            c = ad[:, lo:hi] @ bd[lo:hi, :]
            A = ad[:, lo:hi].abs() @ bd[lo:hi, :].abs()
            m = torch.zeros_like(S)
            for k in range(lo, hi):
                m = torch.maximum(m, la[:, k, None] * lb[None, k, :])
        else:
            c = A = m = torch.zeros_like(S)
        X = torch.maximum(S.abs(), m)
        S_next = S + c
        E = ((1 + SIG8) * E + torch.minimum(RHO8 * (X + E), KAP8 * A + RHOL8 * (X + E))
             + SIG8 * S_next.abs() + ALP8)
        S = S_next
    return S, E


def f64(x):
    return x.to(torch.float32).double()


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


def check_out(a, b, c, rows=512):
    """Largest error / tolerance of c against Post, row block by row block; also the median of
    E / |S| (how large the tolerance is next to the outputs)."""
    M, K = a.shape
    bd = f64(b)
    worst = 0.0
    rel = []
    for r in range(0, M, rows):
        ad = f64(a[r:r + rows])
        S, E = tol_err(ad, bd)
        assert ((S.abs() + E) <= MAX16).all(), "no room in float16"
        got = c[r:r + rows].double()
        assert torch.isfinite(got).all()
        lo, hi = rounding_interval(got)
        dist = torch.clamp(torch.maximum(lo - S, S - hi), min=0)
        worst = max(worst, (dist / E).max().item())   # OutSpec's first clause
        bound = E + (S.abs() + E) * U16 + ETA16
        worst = max(worst, ((got - S).abs() / bound).max().item())
        rel.append((E / S.abs().clamp_min(1e-30)).reshape(-1)[::97])
        del ad, S, E, got, bound
    return worst, torch.cat(rel).median().item()


def host_case(name, M, N, K, seed=0, scale=1.0):
    torch.manual_seed(seed)
    a = torch.randn((M, K), device=DEV, dtype=torch.float16)
    if M == N == K:
        # the file's own draw: b = randn((K, N)); b = b.T; then to float8_e5m2
        b = torch.randn((K, N), device=DEV, dtype=torch.float16).T
    else:
        # the same layout for other shapes: the transpose of a contiguous N x K matrix
        b = torch.randn((N, K), device=DEV, dtype=torch.float16).T
    if scale != 1.0:
        a = (a.float() * scale).half()
        b = (b.float() * scale).half()
    a = a.to(F8)
    b = b.to(F8)
    assert b.stride() == (1, K)
    a0, b0 = a.clone(), b.clone()
    c = mm.matmul(a, b)
    torch.cuda.synchronize()
    same = torch.equal(a.view(torch.uint8), a0.view(torch.uint8)) and \
        torch.equal(b.view(torch.uint8), b0.view(torch.uint8))
    w, med = check_out(a, b, c)
    print(f"{name}: M={M} N={N} K={K}{'' if scale == 1.0 else f' (x{scale})'}: "
          f"max err/tol {w:.4f} (margin {1 - w:.4f}), median E/|S| {med:.3f}, inputs unchanged {same}")
    del a, b, c, a0, b0
    torch.cuda.empty_cache()
    return w < 1 and same, w


def direct_case(name, M, N, K, layout, seed=1):
    """A direct launch of `matmul_kernel` with the host's launch parameters, `c` inside a guarded
    buffer: `layout` "padded" (rows of `N + 13` elements) or "colmajor"; `b` row-major (strides `(N, 1)`, a layout
    the host's run does not use)."""
    torch.manual_seed(seed)
    a = torch.randn((M, K), device=DEV, dtype=torch.float16).to(F8)
    b = torch.randn((K, N), device=DEV, dtype=torch.float16).to(F8)
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
    same = torch.equal(a.view(torch.uint8), a0.view(torch.uint8)) and \
        torch.equal(b.view(torch.uint8), b0.view(torch.uint8))
    w, _ = check_out(a, b, c)
    print(f"{name}: M={M} N={N} K={K} c {layout}, b strides {tuple(b.stride())}: max err/tol "
          f"{w:.4f} (margin {1 - w:.4f}), bytes outside the matrix c unchanged {frame}, "
          f"inputs unchanged {same}")
    return w < 1 and frame and same, w


def triton_cdiv(x, y):
    return (x + y - 1) // y


def post_ok(S, E, got):
    """Post's clauses per element (all elements here are in float16's range): c finite, the reals
    rounding to c meet [S - E, S + E], and |c - S| <= E + (|S| + E)·2^-11 + 2^-25."""
    lo, hi = rounding_interval(got)
    dist = torch.clamp(torch.maximum(lo - S, S - hi), min=0)
    bound = E + (S.abs() + E) * U16 + ETA16
    return torch.isfinite(got) & (dist <= E) & ((got - S).abs() <= bound)


def coherent_case(K, M=256, N=256):
    """Coherent data where the device's fp8 accumulation loses most: one large product first
    (`128·128`), then `K − 1` products `1.75·1` of one sign, each below the accumulator's window once
    the sum is large. Checks `Post` and prints the device's own error `|c − S| / |S|` next to
    `E / |S|`."""
    a = torch.full((M, K), 1.75, device=DEV, dtype=torch.float16)
    a[:, 0] = 128
    b = torch.full((K, N), 1.0, device=DEV, dtype=torch.float16)
    b[0, :] = 128
    a, b = a.to(F8), b.to(F8)
    c = mm.matmul(a, b)
    torch.cuda.synchronize()
    S, E = tol_err(f64(a), f64(b))
    got = c.double()
    good = post_ok(S, E, got)
    lo, hi = rounding_interval(got)
    dist = torch.clamp(torch.maximum(lo - S, S - hi), min=0)
    w = max((dist / E).max().item(),
            ((got - S).abs() / (E + (S.abs() + E) * U16 + ETA16)).max().item())
    dev = ((got - S).abs() / S.abs()).max().item()
    print(f"coherent K={K}: M={M} N={N}: device error |c - S|/|S| {dev:.3f}, E/|S| "
          f"{(E / S.abs()).max().item():.3f}, max err/tol {w:.4f}, Post holds {bool(good.all())}")
    return bool(good.all()), w


def negatives(K, rows=512, seed=0):
    """Wrong results, each rounded to float16, on the file's draws (`M = N = K`, the first `rows`
    rows): all zero; the first step's 64 products left out; the last product left out; `0.9` times
    the exact result; another column's result; `b`'s rows shifted by one along `K`. `Post` must
    reject each (some element breaks it); prints the share of elements that do."""
    torch.manual_seed(seed)
    a = torch.randn((K, K), device=DEV, dtype=torch.float16)
    b = torch.randn((K, K), device=DEV, dtype=torch.float16).T
    a, b = a.to(F8)[:rows], b.to(F8)
    ad, bd = f64(a), f64(b)
    S, E = tol_err(ad, bd)
    wrongs = {
        "zero": torch.zeros_like(S),
        "no_first_step": S - ad[:, :64] @ bd[:64, :],
        "no_last_product": S - ad[:, -1:] @ bd[-1:, :],
        "scale_0.9": 0.9 * S,
        "other_column": torch.roll(S, 1, dims=1),
        "k_shifted": ad[:, 1:] @ bd[:-1, :],
    }
    ok = True
    for name, w in wrongs.items():
        good = post_ok(S, E, w.half().double())
        rejected = not bool(good.all())
        ok &= rejected
        print(f"negative K={K} {name}: Post rejects it {rejected} "
              f"({1 - good.double().mean().item():.2%} of elements break it)")
    del a, b, ad, bd, S, E
    torch.cuda.empty_cache()
    return ok


def main():
    ok, worst = True, 0.0
    cases = [("setup_test_512", 512, 512, 512)]
    cases += [(f"setup_bench_{n}", n, n, n) for n in [128 * i for i in range(2, 17)]]
    cases += [
        # edges (K <= 2048): K not a multiple of 64 (a partial last step; K mod 64 <= 32: the last
        # step's second chunk all disabled lanes), M and N not multiples of the blocks, one row /
        # column, K = 1, larger inputs; all but edge_m4096_n384_k2048 and edge_scale16 lie outside
        # Inst.ok's real-shape conditions (N_dim, K_dim: N and K multiples of 128), where Post is
        # checked as well
        ("edge_m1000_n777_k2047", 1000, 777, 2047),
        ("edge_m300_n1000_k2000", 300, 1000, 2000),
        ("edge_m4096_n384_k2048", 4096, 384, 2048),
        ("edge_m1_n1_k65", 1, 1, 65),
        ("edge_m129_n257_k1", 129, 257, 1),
        ("edge_scale16", 512, 512, 1024, 3, 16.0),
    ]
    for c in cases:
        r, w = host_case(*c)
        ok &= r
        worst = max(worst, w)
    for c in [("direct_padded_c", 300, 1000, 777, "padded"), ("direct_colmajor_c", 513, 259, 1000, "colmajor")]:
        r, w = direct_case(*c)
        ok &= r
        worst = max(worst, w)
    for K in (512, 1024, 2048):
        r, w = coherent_case(K)
        ok &= r
        worst = max(worst, w)
    coherent_case(4096)   # outside Inst.ok (K <= 2048): printed only, for README's Exclusions
    for K in (512, 1024, 2048):
        ok &= negatives(K)
    print(f"overall: max err/tol {worst:.4f} (margin {1 - worst:.4f}); {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
