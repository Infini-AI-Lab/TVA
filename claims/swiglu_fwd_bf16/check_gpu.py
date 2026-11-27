"""GPU check of the claim `Claims.SwigluFwd` (Claim.lean).

Runs the kernel's own host (source/swiglu.py, imported by path) on the launches SETUP.md asks for
and on launches at the edges of `Inst.ok` and `Pre`, and checks `Post` exactly as it is stated, in
float64, with a = x[r, c] and b = y[r, c] the exact bfloat16 inputs:

* z[r, c] finite;
* for a <= -89: z[r, c] == 0;
* for a > -89: |z[r, c] - zRef| <= (2^-8 + 2^-13) * |zRef| + 2^-133, zRef = a * b / (1 + exp(-a));
* the frame: x and y unchanged, and (for the direct launches, whose output sits inside a larger
  buffer filled with a sentinel) no element outside the written ones changed.

Only elements inside `Pre` (|zRef| <= 2^127) are checked; each case says how many lie outside.
The reference is computed in float64 (relative error about 1e-16, negligible against 2^-8). Each
line reports the largest error / tolerance. Negative tests: wrong outputs (sigmoid(x) * y without
the factor x, x * y, rounding toward zero into bfloat16, the product x * sigmoid(x) rounded to
bfloat16 before the second product, sigmoid with exp(+x)) must exceed the tolerance.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "swiglu.py")
spec = importlib.util.spec_from_file_location("swiglu_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import triton  # noqa: E402

DEV = torch.device("cuda")
REL = 2.0 ** -8 + 2.0 ** -13
ABS = 2.0 ** -133
UNDER = -89.0
PRE_MAX = 2.0 ** 127
SENT = -12345.0

worst = [0.0]
failures = []


def zref(a, b):
    """zRef = a * b / (1 + e^-a) in float64 (the claim's exact value; for a <= -89 unused)."""
    return a * b / (1.0 + torch.exp(-a))


def check_post(name, x, y, z):
    """Post on flattened rows: x, y, z of one shape (bfloat16), z the kernel's output. Checked in
    chunks of 2^24 elements, to keep the float64 copies small."""
    xs, ys, zs = x.reshape(-1), y.reshape(-1), z.reshape(-1)
    e, fin, zero_ok, n_under, n_out = 0.0, True, True, 0, 0
    CH = 1 << 24
    for lo in range(0, xs.numel(), CH):
        a = xs[lo:lo + CH].double()
        b = ys[lo:lo + CH].double()
        o = zs[lo:lo + CH].double()
        ref = zref(a, b)
        under = a <= UNDER
        pre = under | (ref.abs() <= PRE_MAX)
        n_out += int((~pre).sum().item())
        n_under += int((under & pre).sum().item())
        fin = fin and bool(torch.isfinite(o[pre]).all().item())
        zero_ok = zero_ok and bool((o[under & pre] == 0).all().item())
        sel = pre & ~under
        if sel.any():
            tol = REL * ref[sel].abs() + ABS
            e = max(e, ((o[sel] - ref[sel]).abs() / tol).max().item())
        del a, b, o, ref
    worst[0] = max(worst[0], e)
    good = fin and zero_ok and e <= 1
    if not good:
        failures.append(name)
    print(f"{'ok  ' if good else 'FAIL'} {name}: err/tol {e:.4f}, finite {fin}, zero for a <= -89 "
          f"{zero_ok} ({n_under} elements), outside Pre {n_out}")
    return good


def host_case(name, x, y, **kw):
    """The host's own launch (`swiglu_fwd`, B = 4096 by default) on x, y of one shape."""
    x0, y0 = x.clone(), y.clone()
    z = mod.swiglu_fwd(x, y, **kw)
    torch.cuda.synchronize()
    frame = torch.equal(x, x0) and torch.equal(y, y0) and z.shape == x.shape and z.is_contiguous()
    if not frame:
        failures.append(name + " (frame)")
        print(f"FAIL {name}: inputs changed or output not contiguous")
    check_post(name, x, y, z)


def direct_case(name, x2, sx, y2, sy, N, D, B, num_warps=4, T_arg=None, grid=None, check=True):
    """A direct launch: x2, y2 flat buffers read at r * s + c; z inside a sentinel-filled buffer.
    T_arg overrides the runtime T (to exercise its int64 class); grid overrides the grid."""
    T = N * D
    pad = 4096
    nz = T if grid is None else min(T, grid[0] * B)
    zbig = torch.full((nz + 2 * pad,), SENT, dtype=torch.bfloat16, device=DEV)
    z = zbig[pad:pad + nz]
    x0, y0 = x2.clone(), y2.clone()
    g = grid if grid is not None else (triton.cdiv(T, B),)
    mod.swiglu_fwd_kernel[g](x2, y2, z, sx, sy, D, T if T_arg is None else T_arg, D=D, B=B,
                             num_warps=num_warps)
    torch.cuda.synchronize()
    frame = (torch.equal(x2, x0) and torch.equal(y2, y0)
             and bool((zbig[:pad] == SENT).all()) and bool((zbig[pad + nz:] == SENT).all()))
    if not frame:
        failures.append(name + " (frame)")
    print(f"     frame {'holds' if frame else 'BROKEN'}")
    if check:
        k = torch.arange(nz, device=DEV)
        r, c = k // D, k % D
        check_post(name, x2[r * sx + c], y2[r * sy + c], z)


def bf(t):
    return t.to(device=DEV, dtype=torch.bfloat16)


def all_bf16():
    """Every finite bfloat16."""
    p = torch.arange(0, 65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return p[torch.isfinite(p.float())].to(DEV)


def negatives(x, y):
    """Wrong outputs that Post must reject: each exceeds the tolerance somewhere."""
    z = mod.swiglu_fwd(x, y)
    torch.cuda.synchronize()
    a, b = x.reshape(-1).double(), y.reshape(-1).double()
    ref = zref(a, b)
    sel = (a > UNDER) & (ref.abs() <= PRE_MAX)
    tol = REL * ref.abs() + ABS
    xf, yf = x.reshape(-1).float(), y.reshape(-1).float()
    sg = torch.sigmoid(xf)
    p = xf * sg
    wrong = {
        "z = sigmoid(x) * y (no factor x)": (sg * yf).double(),
        "z = x * y": (a * b),
        "rounding toward zero into bfloat16": bf_rz(p * yf),
        "x * sigmoid(x) rounded to bfloat16 first": (p.to(torch.bfloat16).float() * yf).to(torch.bfloat16).double(),
        "sigmoid with exp(+x)": (xf / (1 + torch.exp(xf)) * yf).to(torch.bfloat16).double(),
        "z = 0": torch.zeros_like(a),
    }
    ok = check_post("negatives: the kernel itself", x, y, z)
    for k, got in wrong.items():
        e = ((got[sel] - ref[sel]).abs() / tol[sel]).max().item()
        bad = e > 1
        if not bad:
            failures.append(f"negative: {k}")
        print(f"{'ok  ' if bad else 'FAIL'} negative {k}: err/tol {e:.3e} -> "
              f"{'rejected' if bad else 'ACCEPTED'}")
    return ok


def bf_rz(v):
    """float32 -> bfloat16 rounding toward zero (truncate the low 16 bits), as float64."""
    bits = v.float().contiguous().view(torch.int32) & ~0xFFFF
    return bits.view(torch.float32).double()


def main():
    gen = torch.Generator().manual_seed(0)
    rn = lambda *s, scale=1.0: bf(torch.randn(s, generator=gen) * scale)

    print("== SETUP.md: the source's test (randn inputs)")
    for shape in [(1, 1), (7, 33), (4, 1024, 512), (2, 2048, 2752)]:
        host_case(f"host {shape}", rn(*shape), rn(*shape))
    gy = rn(4, 1024, 2 * 688)
    g, yh = gy.chunk(2, dim=-1)
    host_case("host chunked (4, 1024, 688) of (4, 1024, 1376), read in place", g, yh)
    assert mod.prepare_input(g)[1] == 1376

    print("== SETUP.md: fla's GatedMLP (D = 5632 and 11008), training batches and one token")
    host_case("host GatedMLP 2048: (8, 2048, 5632)", rn(8, 2048, 5632, scale=2.0), rn(8, 2048, 5632))
    host_case("host GatedMLP 2048 decode: (1, 1, 5632)", rn(1, 1, 5632), rn(1, 1, 5632))
    host_case("host GatedMLP 4096: (4, 4096, 11008)", rn(4, 4096, 11008, scale=3.0), rn(4, 4096, 11008))
    host_case("host SwiGLULinear's call (output_contiguous=True)", rn(3, 77, 5632), rn(3, 77, 5632),
              output_contiguous=True)

    print("== SETUP.md: every B of fla's autotune list, with several num_warps")
    x, y = rn(4096, 2752, scale=4.0), rn(4096, 2752)
    for B in [512, 1024, 2048, 4096, 8192]:
        for nw in [1, 4, 32]:
            host_case(f"host (4096, 2752) B {B} num_warps {nw}", x, y, B=B, num_warps=nw)

    print("== edges of Pre: every finite bfloat16 x against y in a range of magnitudes")
    xa = all_bf16()
    for yv in [1.0, -3.0, 2.0 ** -133, 2.0 ** -120, 1e-20, 2.0 ** 100, 2.0 ** 126, -(2.0 ** 127)]:
        host_case(f"host every x, y = {yv:.3g}", xa, torch.full_like(xa, yv))
    print("== edges of Pre: every finite bfloat16 y against x near the overflow of tl.exp(-x)")
    ya = all_bf16()
    for xv in [-88.0, -88.5, -89.0, -90.0, -1e30, 0.0, 2.0 ** -133, -(2.0 ** -133), 1e-3, 88.5, 1e30]:
        host_case(f"host x = {xv:.3g}, every y", torch.full_like(ya, xv), ya)
    print("== edges of Pre: |zRef| up to 2^127")
    a = rn(1 << 16, scale=8.0).double()
    s = (PRE_MAX * (1 - 2.0 ** -9)) / zref(a, torch.ones_like(a)).abs()
    b = bf(s)  # rounds to nearest: may step just past 2^127, those are skipped as outside Pre
    host_case("host |zRef| near 2^127", bf(a), b)

    print("== edges of Inst.ok: layouts (prepare_input's strides)")
    host_case("host D = 1", rn(1000, 1), rn(1000, 1))
    host_case("host 1-D x", rn(4099), rn(4099))
    host_case("host broadcast rows (stride 0)", rn(1, 1536).expand(64, 1536), rn(64, 1536))
    host_case("host transposed y (copied)", rn(96, 80), rn(80, 96).t())
    host_case("host 3-D with a size-1 dim, chunked", *rn(1, 300, 2 * 129).chunk(2, dim=-1))
    host_case("host empty: (0, 64)", rn(0, 64), rn(0, 64))
    print("== edges of Inst.ok: direct launches (other strides, B), frame checked")
    for B in [1, 64, 512, 8192, 2 ** 15]:
        for (N, D, sx, sy) in [(5, 7, 7, 9), (3, 100, 0, 250), (17, 33, 64, 33)]:
            x2 = rn((N - 1) * sx + D)
            y2 = rn((N - 1) * sy + D)
            direct_case(f"direct B {B} N {N} D {D} sx {sx} sy {sy}", x2, sx, y2, sy, N, D, B)
    print("== T in the int64 class (T >= 2^31): a direct launch on the first 2 programs")
    N, D = 2 ** 20 + 3, 2048  # T = 2^31 + 6144
    x2, y2 = rn(4 * 4096), rn(4 * 4096)
    direct_case("direct T = 2^31 + 6144 (int64), sx = sy = 0, grid (2,)", x2[:D], 0, y2[:D], 0, N, D,
                4096, T_arg=N * D, grid=(2,))

    print(f"\nworst error / tolerance: {worst[0]:.4f} (margin {1 / worst[0]:.2f}x)")

    print("\n== negative tests (randn inputs, and every bfloat16 x against y = 3)")
    negatives(rn(1024, 1024, scale=3.0), rn(1024, 1024))
    negatives(xa, torch.full_like(xa, 3.0))

    print("\n== outside the claim (README, Exclusions)")
    x = bf(torch.tensor([2.0 ** 100, -89.0, -100.0]))
    y = bf(torch.tensor([2.0 ** 100, 2.0 ** 100, 1.0]))
    z = mod.swiglu_fwd(x, y)
    torch.cuda.synchronize()
    print(f"x = 2^100, y = 2^100 (|zRef| = 2^200 > 2^127): z = {z[0].item()}")
    print(f"x = -89, y = 2^100: z = {z[1].item()} (zRef = {zref(x.double(), y.double())[1].item():.3e}; "
          f"Post asks z = 0)")
    print(f"x = -100, y = 1: z = {z[2].item()} (zRef = {zref(x.double(), y.double())[2].item():.3e})")

    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("all checks hold")


if __name__ == "__main__":
    main()
