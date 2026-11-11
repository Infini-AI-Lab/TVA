"""GPU check of the claim `Claims.StreamKMatmul` (Claim.lean).

Runs the kernel's own host, `streamk_matmul` of `source/streamk_matmul.py` (imported by path), on
the launches of `SETUP.md` and on launches at the edges of `Inst.ok`, and checks `Post` as it is
stated, for every element `(i, j)` of `c`:

* the element holds a `float32` value (the matrix `c` starts at zero, as `Pre` asks);
* if row `i` of `a` and column `j` of `b` are finite, the value is finite and
  `|c - S| <= tol`, with `S = sum_k a[i,k]*b[k,j]` and `tol` exactly as `Post` defines it, computed
  in `float64` from the host's inputs (`ref_tol`): for the chunks `t < T = cdiv(K, 64)` of 64
  products, `chunkSum t`, `chunkMag t = sum mag16(a)*mag16(b)` (`mag16 q = 0` for `0`, else
  `max(|q|, 2^-14)`), `pre t = sum_{u<t} chunkSum u`, `dev t = max_{s<=t} |pre t - pre s|`,
  `tolMag = sum_{t<=T} dev t + sum_{t<T} chunkMag t`, `tol = 2^-15 * tolMag + 2^-100` (the
  `float16` products are exact in `float64`; the sums' own error is far below `tol`).

It reports, over the elements with finite inputs, the largest error / tolerance, and for the host
launches the median of tol / |S| (elements with S != 0). The frame: `a`
and `b` are unchanged (bitwise), and a direct launch of the module's kernel with `c` a padded
matrix (`stride_cm > N`) inside a sentinel-filled buffer leaves every element outside the matrix
unchanged; direct launches also use other program counts and transposed (strided) `a` and `b`.
That the tolerance is meaningful: on `torch.randn` inputs the host's result rounded to `bfloat16`,
with one 64-chunk left out, or accumulated in `float16` chunk by chunk, fails it at every tested `K`
(64 to 8192); with the mass in the last chunk, the result rounded to `float16` fails it; the result
rounded to `float16` on `randn` inputs is reported (info: it fails up to `K = 512`).

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "streamk_matmul.py")
spec = importlib.util.spec_from_file_location("streamk_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
EPS, DELTA = 2.0 ** -15, 2.0 ** -100


def mag16(q):
    return torch.where(q == 0, torch.zeros_like(q), torch.clamp(q.abs(), min=2.0 ** -14))


def ref_tol(ad, bd):
    """`dotSum` and `tol` for every element, from `float64` operands (`ad` [m, K], `bd` [K, N],
    finite): chunk by chunk of 64, `pre`, `dev = max_{s<=t} |pre t - pre s|` (from the running min
    and max of `pre`) at the chunk boundaries `t = 0 .. T`, and
    `tol = 2^-15 * (sum_{t<=T} dev t + sum_{t<T} chunkMag t) + 2^-100`."""
    K = ad.shape[1]
    T = (K + 63) // 64
    al, bl = mag16(ad), mag16(bd)
    pre = torch.zeros(ad.shape[0], bd.shape[1], dtype=torch.float64, device=ad.device)
    mn, mx = pre.clone(), pre.clone()
    mag = pre.clone()
    for t in range(T):
        sl = slice(64 * t, min(64 * t + 64, K))
        cs = ad[:, sl] @ bd[sl, :]
        cm = al[:, sl] @ bl[sl, :]
        dev = torch.maximum(pre - mn, mx - pre)
        mag += dev + cm
        pre += cs
        mn = torch.minimum(mn, pre)
        mx = torch.maximum(mx, pre)
    mag += torch.maximum(pre - mn, mx - pre)          # dev T
    return pre, EPS * mag + DELTA


def post_check(a, b, c, chunk=512):
    """(worst error / tol over elements with finite inputs, all such outputs finite, count of other
    elements)."""
    worst, finite, other = 0.0, True, 0
    REL.clear()
    bd = b.double()
    bfin = torch.isfinite(bd).all(dim=0)                     # [N]
    bd0 = torch.where(torch.isfinite(bd), bd, torch.zeros_like(bd))
    for i0 in range(0, a.shape[0], chunk):
        ad = a[i0:i0 + chunk].double()
        afin = torch.isfinite(ad).all(dim=1)                 # [m]
        ok = afin[:, None] & bfin[None, :]
        other += int((~ok).sum().item())
        ad0 = torch.where(torch.isfinite(ad), ad, torch.zeros_like(ad))
        S, tl_ = ref_tol(ad0, bd0)
        cc = c[i0:i0 + chunk].double()
        finite &= bool(torch.isfinite(cc)[ok].all().item())
        r = ((cc - S).abs() / tl_)[ok]
        nz = ok & (S != 0)
        REL.append((tl_[nz] / S[nz].abs()).float().cpu())
        if r.numel():
            worst = max(worst, r.max().item())
        del ad, ad0, S, tl_, cc, r
    return worst, finite, other


def direct(a, b, num_programs, pad=5, guard=4096):
    """The host's launch (lines 78-92) with `c` an `M x N` matrix of row stride `N + pad` inside a
    sentinel buffer (the matrix itself zero, as `Pre` asks)."""
    M, K = a.shape
    _, N = b.shape
    SENT = -4321.5
    buf = torch.full((guard + M * (N + pad) + guard,), SENT, device=DEVICE)
    c = buf[guard:guard + M * (N + pad)].view(M, N + pad)[:, :N]
    c.zero_()
    num_tiles = triton.cdiv(M, mod.BLOCK_M) * triton.cdiv(N, mod.BLOCK_N)
    total_iters = num_tiles * triton.cdiv(K, mod.BLOCK_K)
    ipp = triton.cdiv(total_iters, num_programs)
    mod.streamk_matmul_kernel[(num_programs,)](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        ipp, total_iters, BLOCK_M=mod.BLOCK_M, BLOCK_N=mod.BLOCK_N, BLOCK_K=mod.BLOCK_K,
        num_warps=8, num_stages=3)
    torch.cuda.synchronize()
    mask = torch.ones_like(buf, dtype=torch.bool)
    mask[guard:guard + M * (N + pad)].view(M, N + pad)[:, :N] = False
    return c, bool((buf[mask] == SENT).all().item())


results = []
worst_all = 0.0
REL = []


def check(name, a, b, num_programs=None, host=True, progs=()):
    global worst_all
    ab, bb = a.clone(), b.clone()
    ok, parts = True, []
    if host:
        c = mod.streamk_matmul(a, b, num_programs)
        torch.cuda.synchronize()
        w, f, n_other = post_check(a, b, c)
        med = torch.cat(REL).median().item()
        ok &= w <= 1.0 and f
        worst_all = max(worst_all, w)
        parts.append(f"host programs={num_programs}: worst {w:.4f}, median tol/|S| {med:.2e}, finite {f}, "
                     f"non-finite-input elements {n_other}")
        del c
    for p in progs:
        c, frame = direct(a, b, p)
        w, f, _ = post_check(a, b, c)
        ok &= w <= 1.0 and f and frame
        worst_all = max(worst_all, w)
        parts.append(f"direct programs={p}: worst {w:.4f}, frame {frame}")
        del c
    ok &= torch.equal(a.view(torch.int16), ab.view(torch.int16))
    ok &= torch.equal(b.view(torch.int16), bb.view(torch.int16))
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name} M={a.shape[0]} N={b.shape[1]} K={a.shape[1]}: "
          + "; ".join(parts), flush=True)


# The test (lines 96-114): the host's shapes and program counts, torch.randn in float16, seed 0.
torch.manual_seed(0)
for M, N, K, num_programs in [(512, 512, 512, None), (1000, 1000, 1000, None), (3072, 4096, 1024, None),
                              (257, 129, 4097, 7), (64, 64, 64, 3), (8192, 8192, 4096, None)]:
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    small = M * N <= 2 ** 22
    check("test", a, b, num_programs, progs=(1, 132, 1000) if small else ())
    del a, b
torch.cuda.empty_cache()

# Edges of Inst.ok.
g = torch.Generator(device=DEVICE).manual_seed(7)


def rnd(*s, scale=1.0):
    return (torch.randn(*s, device=DEVICE, generator=g) * scale).to(torch.float16)


check("1x1x1", rnd(1, 1), rnd(1, 1), 1, progs=(1, 5))
check("odd shapes", rnd(129, 4100), rnd(4100, 257), 132, progs=(1, 2, 7, 100000 // 1000))
check("K = 2^13", rnd(200, 8192), rnd(8192, 130), 132, progs=(1, 3, 2048))
check("large values", rnd(130, 700, scale=2000.0), rnd(700, 140, scale=2000.0), 13, progs=(4,))
check("all positive", torch.rand(256, 4096, device=DEVICE, generator=g).half(),
      torch.rand(4096, 256, device=DEVICE, generator=g).half(), 132, progs=(5,))
a = rnd(300, 2000, scale=2.0 ** -12)                            # many subnormal operands
b = rnd(2000, 300, scale=2.0 ** -12)
check("subnormal operands", a, b, 132, progs=(9,))
a, b = rnd(256, 640), rnd(640, 256)
a[3, 5], a[7, 0], b[9, 11], b[0, 200] = float("nan"), float("inf"), -float("inf"), float("nan")
check("non-finite inputs", a, b, 132, progs=(3,))
at = rnd(1000, 300).t()                                        # a: 300 x 1000, strides (1, 300)
bt = rnd(200, 1000).t()                                        # b: 1000 x 200, strides (1, 1000)
check("transposed a, b (direct)", at, bt, host=False, progs=(1, 17, 132))

# Meaningfulness: results that are wrong in ways a float32 Stream-K kernel must not be fail Post.
def reject(name, a, b, c):
    w, _, _ = post_check(a, b, c)
    results.append(w > 1.0)
    print(f"{'ok ' if w > 1.0 else 'BAD'} rejected: {name} M={a.shape[0]} N={b.shape[1]} K={a.shape[1]}: "
          f"worst {w:.3f} (must exceed 1)", flush=True)


def report(name, a, b, c):
    w, _, _ = post_check(a, b, c)
    print(f"    (info) {name} K={a.shape[1]}: worst {w:.3f}", flush=True)


for K in (64, 512, 1024, 4096, 4097, 8192):
    a = rnd(256, K)
    b = rnd(K, 256)
    c = mod.streamk_matmul(a, b, 132)
    reject("result rounded to bfloat16 (randn)", a, b, c.to(torch.bfloat16).float())
    report("result rounded to float16 (randn)", a, b, c.half().float())
    cd = c.clone()                                               # one 64-chunk left out
    cd -= (a[:, 64:128].float() @ b[64:128, :].float()) if K > 64 else (a[:, :32].float() @ b[:32, :].float())
    reject("one chunk left out (randn)", a, b, cd)
    # per-chunk float16 accumulation: the accumulator rounded to float16 after every chunk
    acc = torch.zeros(256, 256, device=DEVICE)
    for t in range((K + 63) // 64):
        acc = (acc + a[:, 64 * t:64 * t + 64].float() @ b[64 * t:64 * t + 64, :].float()).half().float()
    reject("float16 accumulation per chunk (randn)", a, b, acc)
    # the mass in the last chunk: rand inputs there, tiny elsewhere
    a2 = (torch.rand(256, K, device=DEVICE, generator=g) * 2.0 ** -10).half()
    b2 = (torch.rand(K, 256, device=DEVICE, generator=g) * 2.0 ** -10).half()
    last = 64 * ((K - 1) // 64)
    a2[:, last:] = torch.rand(256, K - last, device=DEVICE, generator=g).half()
    b2[last:, :] = torch.rand(K - last, 256, device=DEVICE, generator=g).half()
    c2 = mod.streamk_matmul(a2, b2, 132)
    w, _, _ = post_check(a2, b2, c2)
    ok = w <= 1.0
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} mass in the last chunk K={K}: host worst {w:.4f}", flush=True)
    reject("result rounded to float16 (mass in the last chunk)", a2, b2, c2.half().float())

print(f"worst error / tolerance over every element with finite inputs: {worst_all:.4f}")
print(f"{sum(results)}/{len(results)} cases pass")
assert all(results)
