"""GPU check of the prepared claim `Claims.Softmax` (claims/README.md, check 4).

Runs the kernel's own host, `softmax` of `source/02-fused-softmax.py` (imported by path; the
module's top level also runs the tutorial's test and its benchmark, so this script loads it with
`benchmark.run` disabled and makes the benchmark's launches itself), on the launches of `SETUP.md`
and on launches at the edges of `Inst.ok`, and checks `Post` as it is stated, row by row:

* every output element is readable (the kernel wrote a value);
* if the row is `RowOk` (every value finite or `-inf`, one finite), with `sigma_j` the exact softmax
  computed in `float64` from the host's input (`e^(x_j - m) / sum_k e^(x_k - m)`, `m` the row's
  largest value), `sigma_max = 1 / sum_k e^(x_k - m)`, `gap_j = m - x_j` (`0` for `-inf`), and
  `T_j = (elemTol + gapTol*gap_j)*sigma_j + floorTol*sigma_max`: every output is finite and there is
  one factor `sc` with `|sc - 1| <= sumTol(BLOCK_SIZE)` and `|c_j - sc*sigma_j| <= T_j` for every
  `j`. The check computes, per row, the smallest `lam` such that some `sc` in that interval has
  `|c_j - sc*sigma_j| <= lam*T_j` for every `j` (bisection), and reports the largest `lam` (the
  error / tolerance; below 1 means `Post` holds). An element whose `sigma_j` underflows `float64`
  (`gap_j > 745`, or `-inf`) is held to `|c_j| <= floorTol*sigma_max` (stricter than `Post`);
* that the tolerance is meaningful: the host's output rounded to `float16` or `bfloat16`, and the
  exact softmax with a denominator that drops the row's last column, fail it;
* the frame: the input is unchanged (bitwise), and a direct launch with strided input and output
  rows inside a sentinel-filled buffer leaves every element outside the output elements unchanged.

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch
import triton
import triton.testing

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "02-fused-softmax.py")
_orig_run = triton.testing.Mark.run
triton.testing.Mark.run = lambda self, *a, **k: None
spec = importlib.util.spec_from_file_location("softmax_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
triton.testing.Mark.run = _orig_run

DEVICE = mod.DEVICE
ELEM = 2.0 ** -21 + 2.0 ** -30
GAP = 3 * 2.0 ** -24 + 2.0 ** -30
FLOOR = 2.0 ** -133


def sum_tol(B):
    return ((B - 1).bit_length() + 56) * 2.0 ** -24   # (ceil(log2 B) + 56) * 2^-24 (Claim.lean, `sumTol`)


def post_rows(x, out, B, chunk=256):
    """(worst lam over RowOk rows, all RowOk outputs finite, number of rows not RowOk)."""
    s = sum_tol(B)
    worst, finite, notok = 0.0, True, 0
    for r0 in range(0, x.shape[0], chunk):
        xr = x[r0:r0 + chunk].double()
        c = out[r0:r0 + chunk].double()
        isfin = torch.isfinite(xr)
        neginf = xr == -float("inf")
        rowok = (isfin | neginf).all(dim=1) & isfin.any(dim=1)
        notok += int((~rowok).sum().item())
        if not rowok.any():
            continue
        xr, c = xr[rowok], c[rowok]
        finite &= bool(torch.isfinite(c).all().item())
        m = xr.max(dim=1, keepdim=True).values
        e = torch.exp(xr - m)                       # -inf gives 0
        A = e.sum(dim=1, keepdim=True)
        sig = e / A
        smax = 1.0 / A
        gap = torch.where(torch.isfinite(xr), m - xr, torch.zeros_like(xr))
        T = (ELEM + GAP * gap) * sig + FLOOR * smax
        pos = sig > 0
        # elements with sigma_j = 0 in float64: |c_j| <= lam * FLOOR * sigma_max
        z = torch.where(pos, torch.zeros_like(c), c.abs() / (FLOOR * smax))
        lam_zero = z.max(dim=1).values
        rj = torch.where(pos, c / torch.where(pos, sig, torch.ones_like(sig)), torch.zeros_like(c))
        wj = torch.where(pos, T / torch.where(pos, sig, torch.ones_like(sig)), torch.zeros_like(c))
        big = torch.full_like(c, float("inf"))

        def feasible(lam):
            lam_ = lam.unsqueeze(1)
            lo = torch.where(pos, rj - lam_ * wj, -big).max(dim=1).values
            hi = torch.where(pos, rj + lam_ * wj, big).min(dim=1).values
            lo = torch.maximum(lo, torch.full_like(lo, 1 - s))
            hi = torch.minimum(hi, torch.full_like(hi, 1 + s))
            return lo <= hi

        lo_l = torch.zeros(c.shape[0], dtype=torch.float64, device=c.device)
        hi_l = torch.full_like(lo_l, 1e6)
        for _ in range(60):
            mid = (lo_l + hi_l) / 2
            f = feasible(mid)
            hi_l = torch.where(f, mid, hi_l)
            lo_l = torch.where(f, lo_l, mid)
        lam = torch.maximum(hi_l, lam_zero)
        worst = max(worst, lam.max().item())
    return worst, finite, notok


def direct(x, B, ns, npg, pad_in=3, pad_out=5, guard=4096):
    """`softmax_kernel` on `x` copied into rows of stride `n_cols + pad_in`, the output in rows of
    stride `n_cols + pad_out` inside a sentinel buffer, `npg` programs."""
    R, C = x.shape
    SENT = -777.25
    xin = torch.full((R, C + pad_in), 5.0, device=DEVICE)[:, :C]
    xin.copy_(x)
    buf = torch.full((guard + R * (C + pad_out) + guard,), SENT, device=DEVICE)
    y = buf[guard:guard + R * (C + pad_out)].view(R, C + pad_out)[:, :C]
    mod.softmax_kernel[(npg, 1, 1)](y, xin, xin.stride(0), y.stride(0), R, C, BLOCK_SIZE=B,
                                    num_stages=ns, num_warps=8)
    torch.cuda.synchronize()
    mask = torch.ones_like(buf, dtype=torch.bool)
    mask[guard:guard + R * (C + pad_out)].view(R, C + pad_out)[:, :C] = False
    frame = bool((buf[mask] == SENT).all().item())
    return y, frame


results = []
worst_all = 0.0
NS = 4 if mod.SIZE_SMEM > 200000 else 2


def check(name, x, host=True, extra=()):
    """`extra`: (BLOCK_SIZE, num_programs) pairs for direct launches."""
    global worst_all
    before = x.clone()
    parts, ok = [], True
    R, C = x.shape
    if host:
        out = mod.softmax(x)
        torch.cuda.synchronize()
        B = triton.next_power_of_2(C)
        w, f, nok = post_rows(x, out, B)
        ok &= w <= 1.0 and f
        worst_all = max(worst_all, w)
        parts.append(f"host B={B}: worst {w:.4f}, finite {f}, rows not RowOk {nok}")
        del out
    for B, npg in extra:
        y, frame = direct(x, B, NS, npg)
        w, f, nok = post_rows(x, y, B)
        ok &= w <= 1.0 and f and frame
        worst_all = max(worst_all, w)
        parts.append(f"direct B={B} programs={npg}: worst {w:.4f}, frame {frame}")
        del y
    ok &= torch.equal(x.view(torch.int32), before.view(torch.int32))
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name} {R}x{C}: " + "; ".join(parts), flush=True)


# The test (lines 186-190).
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
check("test", x, extra=((1024, 132), (1024, 1), (2048, 1823)))

# The benchmark (lines 203-229): M = 4096, N = 128*i for i = 2..99, torch.randn.
for i in range(2, 100):
    N = 128 * i
    x = torch.randn(4096, N, device=DEVICE, dtype=torch.float32)
    check("bench", x)
    del x
torch.cuda.empty_cache()

# Edges of Inst.ok and of RowOk.
g = torch.Generator(device=DEVICE).manual_seed(5)
ninf, pinf, nan = -float("inf"), float("inf"), float("nan")
x = torch.randn(64, 1000, device=DEVICE, generator=g) * 30
x[0, :] = 0.0
x[0, 0] = 20.0                                   # one large value, tiny outputs
x[1, :] = ninf
x[1, 7] = 1.0                                    # one finite value, the rest -inf
x[2, ::2] = ninf
x[3, :] = ninf                                   # not RowOk: all -inf
x[4, 5] = nan                                    # not RowOk
x[5, 9] = pinf                                   # not RowOk
x[6, :] = torch.linspace(-3.4e38, 3.4e38, 1000)  # near the float32 range: x - max overflows
x[7, :] = 3.4e38
x[8, :] = -3.4e38
x[8, 3] = 3.4e38
x[9, :] = torch.linspace(-120, 0, 1000)          # gaps up to 120
x[10, :] = torch.linspace(-1000, 0, 1000)        # outputs below float32's range
x[11, :] = 2.0 ** -149                           # subnormal inputs
check("edge values", x, extra=((1024, 64), (1024, 7), (4096, 1)))
check("one column", torch.randn(300, 1, device=DEVICE, generator=g), extra=((1, 300), (8, 3)))
check("two columns", torch.randn(300, 2, device=DEVICE, generator=g), extra=((2, 13),))
check("one row", torch.randn(1, 16384, device=DEVICE, generator=g) * 10, extra=((16384, 1),))
check("rows 16384 cols", torch.randn(256, 16384, device=DEVICE, generator=g), extra=((16384, 256),))
check("100 cols", torch.randn(5000, 100, device=DEVICE, generator=g) * 5, extra=((128, 1000), (1024, 5000)))
xs = torch.randn(3, 8, device=DEVICE, generator=g)
xs[:, 0], xs[:, 1] = 0.0, -1.0
check("tiny", xs, extra=((8, 1), (8, 3)))

# Meaningfulness: the host's output rounded to float16 (or bfloat16) must fail Post.
for dt in (torch.float16, torch.bfloat16):
    for shape in ((1823, 781), (4096, 12672)):
        x = torch.randn(*shape, device=DEVICE, generator=g)
        out = mod.softmax(x).to(dt).float()
        w, _, _ = post_rows(x, out, triton.next_power_of_2(shape[1]))
        results.append(w > 1.0)
        print(f"{'ok ' if w > 1.0 else 'BAD'} output rounded to {dt} {shape}: worst {w:.1f} (must exceed 1)",
              flush=True)

# A wrong denominator must fail Post: the sum without the row's last column (exact otherwise).
for shape in ((1823, 781), (4096, 12672)):
    x = torch.randn(*shape, device=DEVICE, generator=g)
    xd = x.double()
    e = torch.exp(xd - xd.max(dim=1, keepdim=True).values)
    out = (e / (e.sum(dim=1, keepdim=True) - e[:, -1:])).float()
    w, _, _ = post_rows(x, out, triton.next_power_of_2(shape[1]))
    results.append(w > 1.0)
    print(f"{'ok ' if w > 1.0 else 'BAD'} denominator drops the last column {shape}: worst {w:.1f} (must exceed 1)",
          flush=True)
    del x, xd, e, out

print(f"worst error / tolerance over every RowOk row: {worst_all:.4f}")
print(f"{sum(results)}/{len(results)} cases pass")
assert all(results)
