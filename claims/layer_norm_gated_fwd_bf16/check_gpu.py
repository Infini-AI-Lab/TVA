"""GPU check of the claim `Claims.LayerNormGatedFwd` (Claim.lean).

Runs the kernel's own host (source/fused_norm_gate.py, imported by path) on the launches SETUP.md
asks for and on launches at the edges of `Inst.ok` and `Pre`, and checks `Post` exactly as it is
stated, in float64:

* rstd[t] finite and |rstd[t] - rstdRef| <= 2^-12 * rstdRef,
  rstdRef = 1 / sqrt(sum_j x_j^2 / D + eps) (eps the host's double);
* y[t, j] finite and |y[t, j] - yRef| <= 2^-7 * |yRef| + 2^-126 * (1 + |zRef|) * (1 + |w_j|) * gMag,
  zRef = x_j * rstdRef, yRef = zRef * w_j * gate, gate = g * sigmoid(g) (swish) or sigmoid(g)
  (sigmoid), gMag = 1 + |g| (swish) or 1 (sigmoid);
* the frame: x, g and weight unchanged, and (for the direct launches, whose outputs sit inside
  larger buffers filled with a sentinel) no byte outside the written elements changed.

The reference uses the exact bfloat16 inputs, in float64 (relative error about 1e-15, negligible
against the tolerances). Each line reports the largest error / tolerance. Negative tests: wrong
outputs (y = 0, no gate, the other activation, the gate g alone, LayerNorm instead of RMSNorm, the
next row's rstd, rstd off by 2^-11, rstd without the division by D) must exceed their tolerances.

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
SRC = os.path.join(HERE, "source", "fused_norm_gate.py")
spec = importlib.util.spec_from_file_location("fused_norm_gate_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import triton  # noqa: E402

DEV = torch.device("cuda")
RSTD_TOL = 2.0 ** -12
Y_TOL = 2.0 ** -7
Y_FLOOR = 2.0 ** -126

worst = {"rstd": 0.0, "y": 0.0}
failures = []
CHUNK = 1 << 14  # rows per float64 chunk, to keep each case's memory small


def refs(x, g, w, eps, swish):
    """rstdRef, yRef and the y tolerance (float64) of bfloat16 rows x, g [R, D] and weight w [D]."""
    xd, gd, wd = x.double(), g.double(), w.double()
    D = x.shape[-1]
    rref = 1.0 / torch.sqrt((xd * xd).sum(-1) / D + eps)
    z = xd * rref[:, None]
    sig = torch.sigmoid(gd)
    gate = gd * sig if swish else sig
    yref = z * wd[None, :] * gate
    gmag = (1.0 + gd.abs()) if swish else torch.ones_like(gd)
    ytol = Y_TOL * yref.abs() + Y_FLOOR * (1.0 + z.abs()) * (1.0 + wd.abs())[None, :] * gmag
    return rref, yref, ytol


def check_post(name, x2d, g2d, w, y2d, rstd1d, eps, swish):
    """Post on the rows of x2d, g2d (bfloat16 [T, D]) against the outputs y2d, rstd1d."""
    assert y2d.dtype == torch.bfloat16 and rstd1d.dtype == torch.float32
    T = x2d.shape[0]
    er = ey = 0.0
    ok_r = ok_y = True
    for s in range(0, T, CHUNK):
        e = min(T, s + CHUNK)
        rref, yref, ytol = refs(x2d[s:e], g2d[s:e], w, eps, swish)
        rc = rstd1d[s:e].double()
        yc = y2d[s:e].double()
        ok_r = ok_r and bool(torch.isfinite(rc).all())
        ok_y = ok_y and bool(torch.isfinite(yc).all())
        er = max(er, ((rc - rref).abs() / (RSTD_TOL * rref)).max().item())
        ey = max(ey, ((yc - yref).abs() / ytol).max().item())
    worst["rstd"] = max(worst["rstd"], er)
    worst["y"] = max(worst["y"], ey)
    good = ok_r and ok_y and er <= 1 and ey <= 1
    if not good:
        failures.append(name)
    print(f"{'ok  ' if good else 'FAIL'} {name}: rstd err/tol {er:.3e} (finite {ok_r}), "
          f"y err/tol {ey:.3e} (finite {ok_y})")
    return good


def act_name(swish):
    return "swish" if swish else "sigmoid"


def host_case(name, x, g, w, eps=1e-6, activation="swish"):
    """The host's own launch (BT = 16) on x, g [T, D] and w [D] (RMSNorm, no bias, no residual)."""
    x0, g0, w0 = x.clone(), g.clone(), w.clone()
    y, mean, rstd, _ = mod.layer_norm_gated_fwd(x, g, w, None, activation=activation, eps=eps,
                                                is_rms_norm=True)
    torch.cuda.synchronize()
    assert mean is None
    if not (torch.equal(x, x0) and torch.equal(g, g0) and torch.equal(w, w0)):
        failures.append(name + " (inputs changed)")
        print(f"FAIL {name}: inputs changed")
    check_post(f"{name} [{activation}]", x, g, w, y, rstd, eps, activation in ("swish", "silu"))


SENT_BF16 = -12345.0
SENT_F32 = -54321.0


def direct_case(name, x2d, g2d, w, eps, BT, swish=True, T_arg=None, grid=None, num_warps=4):
    """A direct launch of the kernel with block size BT (the host passes 16; Inst.ok covers every
    power-of-two BT). The outputs sit inside larger sentinel-filled buffers, to check the frame.
    T_arg overrides the runtime T (to exercise its int64 class); grid overrides the grid."""
    T, D = x2d.shape
    BD = triton.next_power_of_2(D)
    pad = 4096
    ybig = torch.full((T * D + 2 * pad,), SENT_BF16, dtype=torch.bfloat16, device=DEV)
    rbig = torch.full((T + 2 * pad,), SENT_F32, dtype=torch.float32, device=DEV)
    y = ybig[pad:pad + T * D].view(T, D)
    r = rbig[pad:pad + T]
    x0, g0, w0 = x2d.clone(), g2d.clone(), w.clone()
    Targ = T if T_arg is None else T_arg
    gr = grid if grid is not None else (triton.cdiv(T, BT),)
    mod.layer_norm_gated_fwd_kernel[gr](
        x=x2d, g=g2d, y=y, w=w, b=None, residual=None, residual_out=None, mean=None, rstd=r,
        eps=eps, T=Targ, D=D, BD=BD, NB=triton.cdiv(T, 2048 * 32),
        ACTIVATION="swish" if swish else "sigmoid", IS_RMS_NORM=True, STORE_RESIDUAL_OUT=False,
        HAS_RESIDUAL=False, HAS_WEIGHT=True, HAS_BIAS=False, BT=BT, num_warps=num_warps)
    torch.cuda.synchronize()
    frame = (torch.equal(x2d, x0) and torch.equal(g2d, g0) and torch.equal(w, w0)
             and bool((ybig[:pad] == SENT_BF16).all()) and bool((ybig[pad + T * D:] == SENT_BF16).all())
             and bool((rbig[:pad] == SENT_F32).all()) and bool((rbig[pad + T:] == SENT_F32).all()))
    if not frame:
        failures.append(name + " (frame)")
    print(f"     frame {'holds' if frame else 'BROKEN'}")
    check_post(f"{name} [{act_name(swish)}]", x2d, g2d, w, y, r, eps, swish)


def bf(t):
    return t.to(device=DEV, dtype=torch.bfloat16)


def layer(gen, rows, D, gscale=2.0):
    """Random rows as a layer feeds the norm: x ~ N(0, 1), g ~ N(0, gscale^2), w ~ 1 + 0.1 N(0, 1)."""
    x = bf(torch.randn((rows, D), generator=gen))
    g = bf(gscale * torch.randn((rows, D), generator=gen))
    w = bf(1 + 0.1 * torch.randn((D,), generator=gen))
    return x, g, w


def negatives(gen, rows, D, swish, eps=1e-6):
    """Wrong outputs that Post must reject: each exceeds its tolerance somewhere."""
    x, g, w = layer(gen, rows, D)
    act = "swish" if swish else "sigmoid"
    y, _, rstd, _ = mod.layer_norm_gated_fwd(x, g, w, None, activation=act, eps=eps, is_rms_norm=True)
    torch.cuda.synchronize()
    xd, gd, wd, yd, rd = x.double(), g.double(), w.double(), y.double(), rstd.double()
    rref, yref, ytol = refs(x, g, w, eps, swish)
    rtol = RSTD_TOL * rref
    z = xd * rref[:, None]
    sig = torch.sigmoid(gd)
    other = sig if swish else gd * sig
    xc = xd - xd.mean(-1, keepdim=True)
    ln = xc / torch.sqrt((xc * xc).mean(-1, keepdim=True) + eps)
    wrong = {
        "y = 0": (torch.zeros_like(yd), yref, ytol),
        "y without the gate": (z * wd[None, :], yref, ytol),
        "y with the other activation": (z * wd[None, :] * other, yref, ytol),
        "y gated by g alone": (z * wd[None, :] * gd, yref, ytol),
        "y with LayerNorm (mean subtracted)": (ln * wd[None, :] * (gd * sig if swish else sig), yref, ytol),
        "y without the weight": (z * (gd * sig if swish else sig), yref, ytol),
        "y with the next row's rstd": (yd / rd[:, None] * torch.roll(rd, 1)[:, None], yref, ytol),
        "rstd * (1 + 2^-11)": (rd * (1 + 2.0 ** -11), rref, rtol),
        "rstd without / D": (1.0 / torch.sqrt((xd * xd).sum(-1) + eps), rref, rtol),
    }
    for k, (got, r, tol) in wrong.items():
        e = ((got - r).abs() / tol).max().item()
        bad = e > 1
        if not bad:
            failures.append(f"negative {act}: {k}")
        print(f"{'ok  ' if bad else 'FAIL'} negative ({act}, {rows} x {D}) {k}: err/tol {e:.3e} -> "
              f"{'rejected' if bad else 'ACCEPTED'}")


def main():
    gen = torch.Generator().manual_seed(0)
    assert struct.pack(">d", 1e-6).hex() == "3eb0c6f7a0b5ed8d"  # Claim.eps1e6
    assert struct.pack(">d", 1e-5).hex() == "3ee4f8b588e368f1"  # Claim.eps1e5

    print("== SETUP.md: the host's test cases (T, D, activation, eps)")
    for T, D, act, eps in [(4, 64, "silu", 1e-5), (2048, 128, "sigmoid", 1e-5), (200, 50, "sigmoid", 1e-5),
                           (98304, 512, "swish", 1e-6), (262144, 128, "sigmoid", 1e-5), (1, 1, "swish", 1e-5),
                           (33, 300, "swish", 1e-5)]:
        x, g, w = layer(gen, T, D)
        host_case(f"host T {T} D {D} eps {eps:g}", x, g, w, eps=eps, activation=act)
    print("== SETUP.md: the layers (4 x 4096 tokens, one token of a batch of 8; eps 1e-6)")
    for name, T, D, act in [("GatedDeltaNet train", 98304, 512, "swish"), ("GatedDeltaNet decode", 48, 512, "swish"),
                            ("GLA train", 65536, 512, "swish"), ("GLA decode", 32, 512, "swish"),
                            ("KDA train", 262144, 128, "sigmoid"), ("KDA decode", 128, 128, "sigmoid")]:
        x, g, w = layer(gen, T, D)
        host_case(f"host {name} T {T} D {D}", x, g, w, eps=1e-6, activation=act)

    print("== edges of Inst.ok: every D class, T not a multiple of 16, both activations")
    for D in [1, 2, 3, 5, 16, 17, 31, 33, 64, 65, 127, 129, 255, 257, 300, 511, 512]:
        for T in [1, 15, 17, 1001]:
            for act in ["swish", "sigmoid"]:
                x, g, w = layer(gen, T, D, gscale=4.0)
                host_case(f"host T {T} D {D}", x, g, w, activation=act)

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
            g = bf(3 * torch.randn((64, D), generator=gen))
            w = bf(1 + 0.1 * torch.randn((D,), generator=gen))
            for act in ["swish", "sigmoid"]:
                host_case(f"host eps {eps:.3g} D {D}", bf(x), g, w, eps=eps, activation=act)
    # eps = 2^125, the largest Pre allows with T >= 1: only zero rows
    z64 = bf(torch.zeros((17, 64)))
    host_case("host eps 2^125, zero rows", z64, bf(torch.randn((17, 64), generator=gen)),
              bf(torch.ones(64)), eps=2.0 ** 125)

    print("== edges of Pre: sum x^2 + eps up to 2^125, mixed scales")
    for D in [1, 2, 64, 512]:
        x = torch.randn((32, D), generator=gen).abs() + 0.5
        x = x / x.pow(2).sum(-1, keepdim=True).sqrt()  # unit rows
        x = bf(x.double() * 2.0 ** 62).double()
        s = (x * x).sum(-1, keepdim=True)
        x = bf(x * torch.sqrt((2.0 ** 125) * (1 - 2 ** -7) / s))
        assert ((x.double() ** 2).sum(-1) + 1e-6 <= 2.0 ** 125).all()
        g = bf(3 * torch.randn((32, D), generator=gen))
        w = bf(1 + 0.1 * torch.randn((D,), generator=gen))
        for act in ["swish", "sigmoid"]:
            host_case(f"host near 2^125 D {D}", x, g, w, activation=act)
            xm = torch.full((16, D), 2.0 ** -100)
            xm[:, 0] = 2.0 ** 60
            xm[1::2, 0] = -(2.0 ** 60)
            host_case(f"host mixed scales D {D}", bf(xm), g[:16], w, activation=act)

    print("== edges of Pre: gates and weights from 0 to 2^56, overflowing exp, subnormal products")
    for D in [1, 64, 512]:
        rows = 64
        x = torch.randn((rows, D), generator=gen)
        g = 4 * torch.randn((rows, D), generator=gen)
        g[0] = 2.0 ** 56
        g[1] = -(2.0 ** 56)
        g[2] = -88.5  # the last gate before tl.exp(-g) overflows
        g[3] = -89.0  # tl.exp(-g) = inf, the kernel's sigmoid is 0
        g[4] = -100.0
        g[5] = 0.0
        g[6] = 2.0 ** -133
        g[7] = -(2.0 ** -133)
        g[8] = 88.5
        g[9] = 20.0
        g[10] = -20.0
        g[11] = torch.linspace(-120, 120, D)
        g[12] = -88.0
        x[13] = 2.0 ** -133  # tiny normalised values (with a large row elsewhere below)
        x[13, 0] = 2.0 ** 40
        x[14] = 1.0
        g[14] = torch.linspace(-90, -80, D)
        for wname, wv in [("w random", 1 + 0.1 * torch.randn((D,), generator=gen)), ("w = 0", torch.zeros(D)),
                          ("w = 2^56", torch.full((D,), 2.0 ** 56)), ("w = -2^56", torch.full((D,), -(2.0 ** 56))),
                          ("w = 2^-133", torch.full((D,), 2.0 ** -133))]:
            for act in ["swish", "sigmoid"]:
                host_case(f"host extreme gates D {D} {wname}", bf(x), bf(g), bf(wv), activation=act)

    print("== edges of Inst.ok: other BT and num_warps (direct launches; the host passes 16, 4), frame checked")
    for BT in [1, 2, 8, 16, 32, 64]:
        for D in [1, 100, 512]:
            T = 3 * BT + 1
            for sw in [True, False]:
                x, g, w = layer(gen, T, D)
                direct_case(f"direct BT {BT} T {T} D {D}", x, g, w, 1e-6, BT, swish=sw)
    for nw in [8, 16]:
        for BT in [16, 32, 64]:
            x, g, w = layer(gen, 3 * BT + 5, 256)
            direct_case(f"direct BT {BT} num_warps {nw} D 256", x, g, w, 1e-6, BT, num_warps=nw)
    print("== T in the int64 class (T >= 2^31): a direct launch on the first 2 programs' rows")
    x, g, w = layer(gen, 32, 64)
    for sw in [True, False]:
        direct_case("direct T = 2^31 + 7 (int64), grid (2,)", x, g, w, 1e-6, 16, swish=sw,
                    T_arg=2 ** 31 + 7, grid=(2,))

    print(f"\nworst error / tolerance: rstd {worst['rstd']:.3e} (margin {1 / worst['rstd']:.1f}x), "
          f"y {worst['y']:.3e} (margin {1 / worst['y']:.1f}x)")

    print("\n== negative tests (GatedDeltaNet's 6 heads of 512, KDA's 16 heads of 128)")
    negatives(gen, 6 * 512, 512, True)
    negatives(gen, 16 * 512, 128, False)

    print("\n== tol / |y| at real shapes (median, worst over |yRef| > 2^-100)")
    for D, sw in ((512, True), (128, False)):
        x, g, w = layer(gen, 65536, D)
        _, yref, ytol = refs(x, g, w, 1e-6, sw)
        m = yref.abs() > 2.0 ** -100
        rel = ytol[m] / yref.abs()[m]
        print(f"D {D} {act_name(sw)}: median {rel.median().item():.3e}, worst {rel.max().item():.3e}; "
              f"rstd {RSTD_TOL:.3e} relative")

    print("\n== outside the claim (README, Exclusions): the kernel's limits")
    one = bf(torch.ones(4))
    # eps = 0 (fla's own test): a zero row gives 0 * inf
    y, _, r, _ = mod.layer_norm_gated_fwd(bf(torch.zeros((1, 4))), bf(torch.ones((1, 4))), one, None,
                                          eps=0.0, is_rms_norm=True)
    torch.cuda.synchronize()
    print(f"eps = 0, a zero row: rstd {r.item()} (reference inf), y {y[0, 0].item()} (reference 0)")
    # eps below float32's normals: a zero row's var + eps is flushed by sqrt.approx.ftz
    y, _, r, _ = mod.layer_norm_gated_fwd(bf(torch.zeros((1, 4))), bf(torch.ones((1, 4))), one, None,
                                          eps=1e-39, is_rms_norm=True)
    torch.cuda.synchronize()
    print(f"eps = 1e-39, a zero row: rstd {r.item()} (reference {1 / math.sqrt(1e-39):.3e}), "
          f"y {y[0, 0].item()} (reference 0)")
    # sum of squares beyond float32's range
    y, _, r, _ = mod.layer_norm_gated_fwd(bf(torch.full((1, 1), 2.0 ** 64)), bf(torch.ones((1, 1))),
                                          bf(torch.ones(1)), None, eps=1e-6, is_rms_norm=True)
    torch.cuda.synchronize()
    print(f"x = 2^64, D = 1: rstd {r.item()} (reference {2.0 ** -64:.3e}), y {y[0, 0].item()} "
          f"(reference {1 / (1 + math.exp(-1)):.4f})")
    # a product w * g beyond float32's range
    y, _, r, _ = mod.layer_norm_gated_fwd(bf(torch.ones((1, 1))), bf(torch.full((1, 1), 2.0 ** 100)),
                                          bf(torch.full((1,), 2.0 ** 100)), None, eps=1e-6, is_rms_norm=True)
    torch.cuda.synchronize()
    print(f"w = g = 2^100, D = 1: y {y[0, 0].item()} (reference {2.0 ** 200:.3e}, beyond bfloat16)")

    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("all checks hold")


if __name__ == "__main__":
    main()
