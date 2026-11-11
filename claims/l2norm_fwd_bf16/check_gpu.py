"""GPU check of the claim `Claims.L2NormFwd` (Claim.lean).

Runs the kernel's own host (source/l2norm.py, imported by path) on the
launches SETUP.md asks for and on launches at the edges of `Inst.ok` and `Pre`, and checks `Post`
exactly as it is stated, in float64:

* rstd[t] finite and |rstd[t] - rstdRef| <= 2^-12 * rstdRef, rstdRef = 1 / sqrt(sum_j a_j^2 + eps)
  (eps the host's double);
* y[t, j] finite and |y[t, j] - a_j * rstdRef| <= 2^-7 * |a_j| * rstdRef + 2^-132 for a bfloat16
  y (output_dtype = None), <= 2^-11 * |a_j| * rstdRef + 2^-149 for a float32 y
  (output_dtype = torch.float32);
* the frame: x unchanged, and (for the direct launches, whose outputs sit inside larger buffers
  filled with a sentinel) no byte outside the written elements changed.

The reference uses the exact bfloat16 inputs; the sum of squares is computed in float64 (relative
error about 1e-16, negligible against 2^-12). Each line reports the largest error / tolerance.
Negative tests: wrong outputs (y = 0, y = x, y rounded to bfloat16 in a float32 y, rstd off by
2^-11, the next row's rstd) must exceed their tolerances.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "l2norm.py")
spec = importlib.util.spec_from_file_location("l2norm_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import triton  # noqa: E402

DEV = torch.device("cuda")
RSTD_TOL = 2.0 ** -12
Y_TOL = 2.0 ** -7
Y_FLOOR = 2.0 ** -132
Y_TOL32 = 2.0 ** -11
Y_FLOOR32 = 2.0 ** -149

worst = {"rstd": 0.0, "y": 0.0, "y32": 0.0}
failures = []


def check_post(name, x2d, y2d, rstd1d, eps):
    """Post on rows of x2d (bfloat16 [T, D]) against outputs y2d, rstd1d, in float64."""
    xd = x2d.double()
    s = (xd * xd).sum(-1)
    ref = 1.0 / torch.sqrt(s + eps)
    rc = rstd1d.double()
    ok_r = torch.isfinite(rc).all().item()
    er = ((rc - ref).abs() / (RSTD_TOL * ref)).max().item()
    yc = y2d.double()
    ok_y = torch.isfinite(yc).all().item()
    yref = xd * ref[:, None]
    f32 = y2d.dtype == torch.float32
    assert f32 or y2d.dtype == torch.bfloat16
    if f32:
        tol = Y_TOL32 * xd.abs() * ref[:, None] + Y_FLOOR32
    else:
        tol = Y_TOL * xd.abs() * ref[:, None] + Y_FLOOR
    ey = ((yc - yref).abs() / tol).max().item()
    worst["rstd"] = max(worst["rstd"], er)
    worst["y32" if f32 else "y"] = max(worst["y32" if f32 else "y"], ey)
    good = ok_r and ok_y and er <= 1 and ey <= 1
    if not good:
        failures.append(name)
    print(f"{'ok  ' if good else 'FAIL'} {name}: rstd err/tol {er:.3e} (finite {ok_r}), "
          f"y err/tol {ey:.3e} (finite {ok_y})")
    return good


def host_case(name, x, eps=1e-6, output_dtype=None):
    """The host's own launch (BT = 16) on x (any shape [..., D])."""
    x0 = x.clone()
    y, rstd = mod.l2norm_fwd(x, eps=eps, output_dtype=output_dtype)
    torch.cuda.synchronize()
    D = x.shape[-1]
    assert torch.equal(x, x0), "x changed"
    check_post(name, x.reshape(-1, D), y.reshape(-1, D), rstd.reshape(-1), eps)


SENT_BF16 = -12345.0
SENT_F32 = -54321.0


def direct_case(name, x2d, eps, BT, T_arg=None, grid=None, ydt=torch.bfloat16):
    """A direct launch of the kernel with block size BT (the host always passes 16; Inst.ok covers
    every power-of-two BT). The outputs sit inside larger sentinel-filled buffers, to check the
    frame. T_arg overrides the runtime T (to exercise its int64 class); grid overrides the grid."""
    T, D = x2d.shape
    BD = triton.next_power_of_2(D)
    pad = 4096
    ybig = torch.full((T * D + 2 * pad,), SENT_BF16, dtype=ydt, device=DEV)
    rbig = torch.full((T + 2 * pad,), SENT_F32, dtype=torch.float32, device=DEV)
    y = ybig[pad:pad + T * D].view(T, D)
    r = rbig[pad:pad + T]
    x0 = x2d.clone()
    Targ = T if T_arg is None else T_arg
    g = grid if grid is not None else (triton.cdiv(T, BT),)
    mod.l2norm_fwd_kernel[g](x=x2d, y=y, rstd=r, eps=eps, T=Targ, D=D, BD=BD,
                             NB=triton.cdiv(T, 2048 * 32), BT=BT, num_warps=4)
    torch.cuda.synchronize()
    frame = (torch.equal(x2d, x0)
             and bool((ybig[:pad] == SENT_BF16).all()) and bool((ybig[pad + T * D:] == SENT_BF16).all())
             and bool((rbig[:pad] == SENT_F32).all()) and bool((rbig[pad + T:] == SENT_F32).all()))
    if not frame:
        failures.append(name + " (frame)")
    print(f"     frame {'holds' if frame else 'BROKEN'}")
    check_post(name, x2d, y, r, eps)


def bf(t):
    return t.to(device=DEV, dtype=torch.bfloat16)


def negatives(x, output_dtype):
    """Wrong outputs that Post must reject: each exceeds its tolerance somewhere."""
    D = x.shape[-1]
    y, rstd = mod.l2norm_fwd(x, eps=1e-6, output_dtype=output_dtype)
    torch.cuda.synchronize()
    x2, y2, r1 = x.reshape(-1, D).double(), y.reshape(-1, D).double(), rstd.reshape(-1).double()
    ref = 1.0 / torch.sqrt((x2 * x2).sum(-1) + 1e-6)
    yref = x2 * ref[:, None]
    f32 = output_dtype == torch.float32
    ytol = (Y_TOL32 if f32 else Y_TOL) * x2.abs() * ref[:, None] + (Y_FLOOR32 if f32 else Y_FLOOR)
    rtol = RSTD_TOL * ref
    wrong = {
        "y = 0": (torch.zeros_like(y2), yref, ytol),
        "y = x (no normalisation)": (x2, yref, ytol),
        "y with the next row's rstd": (x2 * torch.roll(r1, 1)[:, None], yref, ytol),
        "rstd * (1 + 2^-11)": (r1 * (1 + 2.0 ** -11), ref, rtol),
        "rstd = 1 / sum x^2 (no sqrt)": (1.0 / ((x2 * x2).sum(-1) + 1e-6), ref, rtol),
    }
    if f32:
        wrong["y rounded to bfloat16"] = (y.reshape(-1, D).to(torch.bfloat16).double(), yref, ytol)
    tag = "float32" if f32 else "bfloat16"
    for k, (got, r, tol) in wrong.items():
        e = ((got - r).abs() / tol).max().item()
        bad = e > 1
        if not bad:
            failures.append(f"negative {tag}: {k}")
        print(f"{'ok  ' if bad else 'FAIL'} negative ({tag} y) {k}: err/tol {e:.3e} -> "
              f"{'rejected' if bad else 'ACCEPTED'}")


def main():
    gen = torch.Generator().manual_seed(0)
    assert struct.pack(">d", 1e-6).hex() == "3eb0c6f7a0b5ed8d"  # Claim.eps1e6

    print("== SETUP.md: the host's test cases (B, T, H, D), its inputs randn * 0.5 + 0.3, eps 1e-6")
    for B, T, H, D in [(1, 1, 1, 1), (1, 1, 1, 128), (1, 63, 1, 60), (1, 65, 2, 512), (2, 500, 4, 64),
                       (2, 1000, 2, 100), (3, 1024, 4, 128), (2, 333, 3, 256), (4, 4096, 16, 128)]:
        x = bf(torch.randn((B, T, H, D), generator=gen) * 0.5 + 0.3)
        host_case(f"host ({B}, {T}, {H}, {D})", x)
    print("== SETUP.md: GatedDeltaNet, 4 x 4096 tokens x 6 heads of 256")
    host_case("host (4, 4096, 6, 256)", bf(torch.randn((4, 4096, 6, 256), generator=gen)))
    print("== SETUP.md: PaTH attention's w, output_dtype = float32, 32 heads of 64")
    F32 = torch.float32
    host_case("host (4, 4096, 32, 64) float32 y", bf(torch.randn((4, 4096, 32, 64), generator=gen)),
              output_dtype=F32)
    host_case("host (8, 1, 32, 64) float32 y", bf(torch.randn((8, 1, 32, 64), generator=gen)),
              output_dtype=F32)

    print("== edges of Inst.ok: every D class, T not a multiple of 16")
    for D in [1, 2, 3, 5, 16, 17, 31, 33, 64, 65, 127, 129, 255, 257, 300, 511, 512]:
        for T in [1, 15, 17, 1001]:
            host_case(f"host T {T} D {D}", bf(torch.randn((T, D), generator=gen) * 3.0))
            if T == 17:
                host_case(f"host T {T} D {D} float32 y", bf(torch.randn((T, D), generator=gen) * 3.0),
                          output_dtype=F32)

    print("== edges of Inst.ok: eps from 2^-125 to 2^100, zero rows, tiny rows")
    for eps in [2.0 ** -125, 1e-30, 1e-12, 1e-6, 1e-5, 1.0, 1e10, 1e30, 2.0 ** 100]:
        for D in [1, 7, 128, 512]:
            x = torch.randn((64, D), generator=gen)
            x[0] = 0.0  # a zero row: rstd = eps^-1/2
            x[1] = 2.0 ** -133  # the smallest bfloat16 subnormal
            x[2] = 2.0 ** -70  # squares below float32's subnormals
            x[3, :] = 0.0
            x[3, 0] = 2.0 ** -60
            x[4] *= 2.0 ** 40
            x[5] *= 2.0 ** -40
            if eps <= 2.0 ** 100:
                host_case(f"host eps {eps:.3g} D {D}", bf(x), eps=eps)
                host_case(f"host eps {eps:.3g} D {D} float32 y", bf(x), eps=eps, output_dtype=F32)
    # eps = 2^126, the largest covered: Pre then allows only zero rows
    host_case("host eps 2^126, zero rows", bf(torch.zeros((17, 64))), eps=2.0 ** 126)
    print("== edges of Pre: sum x^2 + eps up to 2^126")
    for D in [1, 2, 64, 512]:
        x = torch.randn((32, D), generator=gen).abs() + 0.5
        x = x / x.pow(2).sum(-1, keepdim=True).sqrt()  # unit rows
        x = bf(x.double() * 2.0 ** 62.99).double()
        # scale every row so that its exact sum of squares is just below 2^126 - eps
        s = (x * x).sum(-1, keepdim=True)
        x = bf(x * torch.sqrt((2.0 ** 126) * (1 - 2 ** -7) / s))
        assert ((x.double() ** 2).sum(-1) + 1e-6 <= 2.0 ** 126).all()
        host_case(f"host near 2^126 D {D}", x)
        host_case(f"host near 2^126 D {D} float32 y", x, output_dtype=F32)
        # mixed scales: one large element, the rest tiny
        xm = torch.full((16, D), 2.0 ** -100)
        xm[:, 0] = 2.0 ** 60
        xm[1::2, 0] = -(2.0 ** 60)
        host_case(f"host mixed scales D {D}", bf(xm))
        host_case(f"host mixed scales D {D} float32 y", bf(xm), output_dtype=F32)

    print("== edges of Inst.ok: other BT (direct launches; the host passes 16), frame checked")
    for BT in [1, 2, 8, 16, 32, 64, 128]:
        for D in [1, 100, 512]:
            T = 3 * BT + 1
            direct_case(f"direct BT {BT} T {T} D {D}", bf(torch.randn((T, D), generator=gen)), 1e-6, BT)
            direct_case(f"direct BT {BT} T {T} D {D} float32 y", bf(torch.randn((T, D), generator=gen)),
                        1e-6, BT, ydt=F32)
    print("== T in the int64 class (T >= 2^31): a direct launch on the first 2 programs' rows")
    x = bf(torch.randn((32, 64), generator=gen))
    direct_case("direct T = 2^31 + 7 (int64), grid (2,)", x, 1e-6, 16, T_arg=2 ** 31 + 7, grid=(2,))
    direct_case("direct T = 2^31 + 7 (int64), grid (2,), float32 y", x, 1e-6, 16, T_arg=2 ** 31 + 7,
                grid=(2,), ydt=F32)

    print(f"\nworst error / tolerance: rstd {worst['rstd']:.3e} (margin {1 / worst['rstd']:.1f}x), "
          f"y {worst['y']:.3e} (margin {1 / worst['y']:.1f}x), "
          f"float32 y {worst['y32']:.3e} (margin {1 / worst['y32']:.1f}x)")

    print("\n== negative tests (DeltaNet's 16 heads of 128, PaTH's 32 heads of 64)")
    negatives(bf(torch.randn((2, 512, 16, 128), generator=gen)), None)
    negatives(bf(torch.randn((2, 512, 32, 64), generator=gen)), F32)

    print("\n== tol / |y| at real shapes (median, worst over y != 0)")
    for shape, odt in (((4, 4096, 16, 128), None), ((4, 4096, 32, 64), F32)):
        x = bf(torch.randn(shape, generator=gen)).reshape(-1, shape[-1])[:65536].double()
        ref = 1.0 / torch.sqrt((x * x).sum(-1) + 1e-6)
        yref = (x * ref[:, None]).abs()
        tol = ((Y_TOL32 if odt else Y_TOL) * yref + (Y_FLOOR32 if odt else Y_FLOOR))[yref > 0] / yref[yref > 0]
        print(f"{shape} {'float32' if odt else 'bfloat16'} y: median {tol.median().item():.3e}, "
              f"worst {tol.max().item():.3e}; rstd {RSTD_TOL:.3e} relative")

    print("\n== outside the claim (README, Exclusions): the kernel's limits")
    # eps below float32's normals: a zero row's sum + eps is flushed by sqrt.approx.ftz
    x = bf(torch.zeros((1, 4)))
    y, r = mod.l2norm_fwd(x, eps=1e-39)
    torch.cuda.synchronize()
    print(f"eps = 1e-39, a zero row: rstd {r.item()} (reference {1 / math.sqrt(1e-39):.3e}), "
          f"y {y[0, 0].item()} (reference 0)")
    # sum of squares beyond float32's range
    x = bf(torch.full((1, 1), 2.0 ** 64))
    y, r = mod.l2norm_fwd(x)
    torch.cuda.synchronize()
    print(f"x = 2^64, D = 1: rstd {r.item()} (reference {2.0 ** -64:.3e}), y {y[0, 0].item()} (reference 1)")

    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("all checks hold")


if __name__ == "__main__":
    main()
