"""GPU check of the claim `Claims.L2NormBwd` (Claim.lean).

Runs the kernel's own host (source/l2norm.py, imported by path) on the
launches SETUP.md asks for and on launches at the edges of `Inst.ok` and `Pre`, and checks `Post`
exactly as it is stated, in float64. For each row, with a, g the row's y and dy values, r its
rstd, S = sum_k g_k a_k, A = sum_k |g_k a_k|, ref_j = g_j r - S a_j r, M_j = |r| (|g_j| + A |a_j|)
and F_j = (1 + |r|) (1 + |a_j|):

* dx[t, j] finite and |dx[t, j] - ref_j| <= 2^-18 M_j + 2^-138 F_j for a float32 dx, and
  <= 2^-8 |ref_j| + 2^-17 M_j + 2^-133 F_j for a bfloat16 dx;
* the frame: y, rstd, dy unchanged, and (for the direct launches, whose output sits inside a
  larger buffer filled with a sentinel) no byte outside the written elements changed.

Every case also checks that its inputs satisfy `Pre`: finite, and
(1 + |r|) (1 + |a_j|) (1 + |g_j| + A) <= 2^124. The reference uses the exact inputs; the sums are
computed in float64 (relative error about 1e-16 of A, far below 2^-18). Each line reports the
largest error / tolerance. Negative tests: wrong outputs (dx = 0, dx = dy * rstd, the projection
added instead of subtracted, the next row's rstd, a float32 dx rounded to bfloat16, dx off by
2^-6) must exceed their tolerances.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
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
BF16, F32 = torch.bfloat16, torch.float32
TOL32 = 2.0 ** -18
FLOOR32 = 2.0 ** -138
RND16 = 2.0 ** -8
TOL16 = 2.0 ** -17
FLOOR16 = 2.0 ** -133
ROOM = 2.0 ** 124

worst = {"dx16": 0.0, "dx32": 0.0}
failures = []


def parts(y2d, r1d, dy2d):
    """ref, M, F in float64 for rows y2d, r1d, dy2d."""
    a, r, g = y2d.double(), r1d.double()[:, None], dy2d.double()
    S = (g * a).sum(-1, keepdim=True)
    A = (g * a).abs().sum(-1, keepdim=True)
    ref = g * r - S * a * r
    M = r.abs() * (g.abs() + A * a.abs())
    F = (1 + r.abs()) * (1 + a.abs())
    return ref, M, F, A


def tol_of(ref, M, F, f32):
    if f32:
        return TOL32 * M + FLOOR32 * F
    return RND16 * ref.abs() + TOL16 * M + FLOOR16 * F


def check_pre(name, y2d, r1d, dy2d):
    a, r, g = y2d.double(), r1d.double()[:, None], dy2d.double()
    fin = bool(torch.isfinite(a).all() and torch.isfinite(r).all() and torch.isfinite(g).all())
    A = (g * a).abs().sum(-1, keepdim=True)
    room = ((1 + r.abs()) * (1 + a.abs()) * (1 + g.abs() + A)).max().item()
    ok = fin and room <= ROOM
    if not ok:
        failures.append(name + " (inputs outside Pre)")
        print(f"FAIL {name}: inputs outside Pre (finite {fin}, room {room:.3e})")
    return ok


def check_post(name, y2d, r1d, dy2d, dx2d):
    """Post on rows (y2d [T, D], r1d [T], dy2d [T, D]) against the output dx2d, in float64."""
    ref, M, F, _ = parts(y2d, r1d, dy2d)
    f32 = dx2d.dtype == F32
    assert f32 or dx2d.dtype == BF16
    c = dx2d.double()
    ok_f = bool(torch.isfinite(c).all())
    e = ((c - ref).abs() / tol_of(ref, M, F, f32)).max().item() if c.numel() else 0.0
    key = "dx32" if f32 else "dx16"
    worst[key] = max(worst[key], e)
    good = ok_f and e <= 1
    if not good:
        failures.append(name)
    print(f"{'ok  ' if good else 'FAIL'} {name}: dx err/tol {e:.3e} (finite {ok_f})")
    return good


def host_case(name, y, rstd, dy):
    """The host's own launch (BT = 16) on y, dy (any shape [..., D]) and rstd (shape [...])."""
    D = y.shape[-1]
    check_pre(name, y.reshape(-1, D), rstd.reshape(-1), dy.reshape(-1, D))
    y0, r0, g0 = y.clone(), rstd.clone(), dy.clone()
    dx = mod.l2norm_bwd(y, rstd, dy)
    torch.cuda.synchronize()
    assert dx.dtype == y.dtype and dx.shape == y.shape
    if not (torch.equal(y, y0) and torch.equal(rstd, r0) and torch.equal(dy, g0)):
        failures.append(name + " (inputs changed)")
        print("     inputs CHANGED")
    check_post(name, y.reshape(-1, D), rstd.reshape(-1), dy.reshape(-1, D), dx.reshape(-1, D))


SENT = {BF16: -12345.0, F32: -54321.0}


def direct_case(name, y2d, r1d, dy2d, BT, dxdt=None, T_arg=None, grid=None):
    """A direct launch of the kernel with block size BT (the host always passes 16; Inst.ok covers
    every power-of-two BT) and a dx of dtype dxdt (the host's is y's). dx sits inside a larger
    sentinel-filled buffer, to check the frame. T_arg overrides the runtime T (to exercise its
    int64 class); grid overrides the grid."""
    T, D = y2d.shape
    dxdt = dxdt or y2d.dtype
    check_pre(name, y2d, r1d, dy2d)
    BD = triton.next_power_of_2(D)
    pad = 4096
    big = torch.full((T * D + 2 * pad,), SENT[dxdt], dtype=dxdt, device=DEV)
    dx = big[pad:pad + T * D].view(T, D)
    y0, r0, g0 = y2d.clone(), r1d.clone(), dy2d.clone()
    Targ = T if T_arg is None else T_arg
    g = grid if grid is not None else (triton.cdiv(T, BT),)
    mod.l2norm_bwd_kernel[g](y=y2d, rstd=r1d, dy=dy2d, dx=dx, eps=1e-6, T=Targ, D=D, BD=BD,
                             NB=triton.cdiv(T, 2048 * 32), BT=BT, num_warps=4)
    torch.cuda.synchronize()
    frame = (torch.equal(y2d, y0) and torch.equal(r1d, r0) and torch.equal(dy2d, g0)
             and bool((big[:pad] == SENT[dxdt]).all()) and bool((big[pad + T * D:] == SENT[dxdt]).all()))
    if not frame:
        failures.append(name + " (frame)")
    print(f"     frame {'holds' if frame else 'BROKEN'}")
    check_post(name, y2d, r1d, dy2d, dx)


def fwd(x, eps=1e-6, ydt=BF16):
    """The forward's outputs as fla's l2norm_fwd makes them: rstd in float32, y = x * rstd."""
    xf = x.to(device=DEV, dtype=F32)
    rstd = (1.0 / torch.sqrt((xf * xf).sum(-1) + eps)).to(F32)
    y = (xf * rstd[..., None]).to(ydt)
    return y, rstd


def on(t, dt):
    return t.to(device=DEV, dtype=dt)


def negatives(y, rstd, dy):
    """Wrong outputs that Post must reject: each exceeds its tolerance somewhere."""
    D = y.shape[-1]
    dx = mod.l2norm_bwd(y, rstd, dy)
    torch.cuda.synchronize()
    a, r, g = y.reshape(-1, D), rstd.reshape(-1), dy.reshape(-1, D)
    ref, M, F, _ = parts(a, r, g)
    f32 = dx.dtype == F32
    tol = tol_of(ref, M, F, f32)
    ad, rd, gd = a.double(), r.double()[:, None], g.double()
    S = (gd * ad).sum(-1, keepdim=True)
    c = dx.reshape(-1, D).double()
    wrong = {
        "dx = 0": torch.zeros_like(c),
        "dx = dy * rstd (no projection)": gd * rd,
        "projection added": gd * rd + S * ad * rd,
        "the next row's rstd": gd * torch.roll(rd, 1, 0) - S * ad * torch.roll(rd, 1, 0),
        "dx * (1 + 2^-6)": c * (1 + 2.0 ** -6),
    }
    if f32:
        wrong["dx rounded to bfloat16"] = dx.reshape(-1, D).to(BF16).double()
    tag = "float32" if f32 else "bfloat16"
    for k, got in wrong.items():
        e = ((got - ref).abs() / tol).max().item()
        bad = e > 1
        if not bad:
            failures.append(f"negative {tag}: {k}")
        print(f"{'ok  ' if bad else 'FAIL'} negative ({tag} dx) {k}: err/tol {e:.3e} -> "
              f"{'rejected' if bad else 'ACCEPTED'}")


def main():
    gen = torch.Generator().manual_seed(0)
    assert struct.pack(">d", 1e-6).hex() == "3eb0c6f7a0b5ed8d"  # Claim.eps1e6

    print("== SETUP.md: the host's test cases (B, T, H, D), y from x = randn * 0.5 + 0.3, dy = randn")
    for B, T, H, D in [(1, 1, 1, 1), (1, 1, 1, 128), (1, 63, 1, 60), (1, 65, 2, 512), (2, 500, 4, 64),
                       (2, 1000, 2, 100), (3, 1024, 4, 128), (2, 333, 3, 256), (4, 4096, 16, 128)]:
        y, rstd = fwd(torch.randn((B, T, H, D), generator=gen) * 0.5 + 0.3)
        dy = on(torch.randn((B, T, H, D), generator=gen), BF16)
        host_case(f"host ({B}, {T}, {H}, {D})", y, rstd, dy)
    print("== SETUP.md: GatedDeltaNet, 4 x 4096 tokens x 6 heads of 256")
    y, rstd = fwd(torch.randn((4, 4096, 6, 256), generator=gen))
    host_case("host (4, 4096, 6, 256)", y, rstd, on(torch.randn((4, 4096, 6, 256), generator=gen), BF16))
    print("== SETUP.md: PaTH attention's w, float32 y, dy, dx, 32 heads of 64")
    y, rstd = fwd(torch.randn((4, 4096, 32, 64), generator=gen), ydt=F32)
    host_case("host (4, 4096, 32, 64) float32", y, rstd, on(torch.randn((4, 4096, 32, 64), generator=gen), F32))
    print("== SETUP.md: a float32 dy into a bfloat16 y (dx bfloat16), and a bfloat16 dy into a float32 y")
    y, rstd = fwd(torch.randn((4, 4096, 16, 128), generator=gen))
    host_case("host (4, 4096, 16, 128) float32 dy", y, rstd,
              on(torch.randn((4, 4096, 16, 128), generator=gen) * 1e-3, F32))
    y, rstd = fwd(torch.randn((2, 512, 32, 64), generator=gen), ydt=F32)
    host_case("host (2, 512, 32, 64) float32 y, bfloat16 dy", y, rstd,
              on(torch.randn((2, 512, 32, 64), generator=gen), BF16))

    print("== edges of Inst.ok: every D class, T not a multiple of 16, the four dtype pairs")
    for D in [1, 2, 3, 5, 16, 17, 31, 33, 64, 65, 127, 129, 255, 257, 300, 511, 512]:
        for T in [1, 15, 17, 1001]:
            for ydt, gdt in ((BF16, BF16), (F32, F32), (BF16, F32), (F32, BF16)):
                if T != 17 and (ydt, gdt) != (BF16, BF16):
                    continue
                y, rstd = fwd(torch.randn((T, D), generator=gen) * 3.0, ydt=ydt)
                dy = on(torch.randn((T, D), generator=gen) * 3.0, gdt)
                host_case(f"host T {T} D {D} y {str(ydt)[6:]} dy {str(gdt)[6:]}", y, rstd, dy)

    print("== edges of Pre: cancellation, scales, zero rows, subnormals, room near 2^124")
    for D in [1, 7, 64, 128, 512]:
        for ydt, gdt in ((BF16, BF16), (F32, F32)):
            T = 64
            y, rstd = fwd(torch.randn((T, D), generator=gen), ydt=ydt)
            g = torch.randn((T, D), generator=gen).to(DEV)
            g[0] = y[0].float() * 3.0           # dy along y: dx = 0 up to rounding
            g[1] = y[1].float() * -1e5          # the same, large
            g[2] = 0.0                          # zero gradient
            g[3] *= 2.0 ** 40
            g[4] *= 2.0 ** -40
            g[5] = 2.0 ** -133                  # bfloat16's smallest subnormal
            g[6] = 2.0 ** -149 if gdt == F32 else 2.0 ** -133
            dy = g.to(gdt)
            rstd = rstd.clone()
            rstd[7] = 2.0 ** 60                 # a zero row's rstd (eps tiny), with y = 0
            y = y.clone()
            y[7] = 0.0
            rstd[8] = 2.0 ** -60
            y[9] = 2.0 ** -133 if ydt == BF16 else 2.0 ** -149
            rstd[9] = 2.0 ** 40
            host_case(f"host edges D {D} y {str(ydt)[6:]} dy {str(gdt)[6:]}", y, rstd, dy)
        # room near 2^124: (1 + r)(1 + a)(1 + g + A) close to the bound
        y = on(torch.full((8, D), 1.0 / D ** 0.5), BF16)
        rstd = on(torch.full((8,), 2.0 ** 30), F32)
        g = torch.randn((8, D), generator=gen).abs() + 1.0
        A = (g * y.double().cpu()).abs().sum(-1, keepdim=True)
        a = y.double().cpu().abs()
        scale = (ROOM * (1 - 2 ** -6)) / ((1 + 2.0 ** 30) * (1 + a) * (g + A)).max()
        dy = on(g * scale, BF16)
        host_case(f"host room near 2^124 D {D}", y, rstd, dy)

    print("== edges of Inst.ok: other BT, a dx of another dtype (direct launches), frame checked")
    for BT in [1, 2, 8, 16, 32, 64, 128]:
        for D in [1, 100, 512]:
            T = 3 * BT + 1
            for ydt, gdt, xdt in ((BF16, BF16, BF16), (F32, F32, F32), (BF16, BF16, F32), (F32, BF16, BF16)):
                if BT not in (1, 16) and (ydt, gdt, xdt) != (BF16, BF16, BF16):
                    continue
                y, rstd = fwd(torch.randn((T, D), generator=gen), ydt=ydt)
                dy = on(torch.randn((T, D), generator=gen), gdt)
                direct_case(f"direct BT {BT} T {T} D {D} y {str(ydt)[6:]} dy {str(gdt)[6:]} "
                            f"dx {str(xdt)[6:]}", y, rstd, dy, BT, dxdt=xdt)
    print("== T in the int64 class (T >= 2^31): a direct launch on the first 2 programs' rows")
    for ydt in (BF16, F32):
        y, rstd = fwd(torch.randn((32, 64), generator=gen), ydt=ydt)
        dy = on(torch.randn((32, 64), generator=gen), ydt)
        direct_case(f"direct T = 2^31 + 7 (int64), grid (2,), {str(ydt)[6:]}", y, rstd, dy, 16,
                    T_arg=2 ** 31 + 7, grid=(2,))

    print(f"\nworst error / tolerance: bfloat16 dx {worst['dx16']:.3e} (margin {1 / worst['dx16']:.1f}x), "
          f"float32 dx {worst['dx32']:.3e} (margin {1 / worst['dx32']:.1f}x)")

    print("\n== negative tests (DeltaNet's 16 heads of 128, PaTH's 32 heads of 64)")
    y, rstd = fwd(torch.randn((2, 512, 16, 128), generator=gen))
    negatives(y, rstd, on(torch.randn((2, 512, 16, 128), generator=gen), BF16))
    y, rstd = fwd(torch.randn((2, 512, 32, 64), generator=gen), ydt=F32)
    negatives(y, rstd, on(torch.randn((2, 512, 32, 64), generator=gen), F32))

    print("\n== tol / max(|dx|) at real shapes (median and worst over rows)")
    for shape, dt in (((4, 4096, 16, 128), BF16), ((4, 4096, 32, 64), F32)):
        x = torch.randn(shape, generator=gen)
        y, rstd = fwd(x, ydt=dt)
        dy = on(torch.randn(shape, generator=gen), dt)
        D = shape[-1]
        ref, M, F, _ = parts(y.reshape(-1, D)[:65536], rstd.reshape(-1)[:65536], dy.reshape(-1, D)[:65536])
        tol = tol_of(ref, M, F, dt == F32)
        rel = tol.max(-1).values / ref.abs().max(-1).values
        print(f"{shape} {str(dt)[6:]}: median {rel.median().item():.3e}, worst {rel.max().item():.3e}")

    print("\n== outside the claim (README, Exclusions): the kernel's limits")
    # dy * rstd beyond float32's range
    y = on(torch.ones((1, 4)) * 0.5, BF16)
    rstd = on(torch.full((1,), 2.0 ** 100), F32)
    dy = on(torch.full((1, 4), 2.0 ** 40), BF16)
    dx = mod.l2norm_bwd(y, rstd, dy)
    torch.cuda.synchronize()
    ref, _, _, _ = parts(y, rstd, dy)
    print(f"r = 2^100, dy = 2^40, y = 0.5, D = 4: dx {dx[0, 0].item()} (reference {ref[0, 0].item():.3e})")

    print(f"\npeak device memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("all checks hold")


if __name__ == "__main__":
    main()
