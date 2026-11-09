"""GPU check of the claim in Claim.lean.

Runs the kernel's own host, `layer_norm_fwd` of source/layernorm.py
(imported by path), on the setup launches of Claim.lean (LayerNorm with a residual, GroupNorm and
LayerNorm without one) and on launches at the edges of `Inst.ok`, and
computes `Post`'s reference and tolerances in float64 exactly as `Post` states them:

* with a residual, res_out[t, j] == bfloat16(s[t, j]) bit for bit, s = float32(x) + float32(res)
  (IEEE float32); without, s = float32(x) and res_out is not written;
* row t uses the weight and bias of group t % G;
* |mean[t] - meanRef| <= meanTol, |rstd[t] - rstdRef| <= rstdTol, |y[t, j] - yRef| <= yTol,
  every stored value finite;
* the frame: a direct launch with the host's arguments into outputs surrounded by guard bands
  leaves the guards and the inputs unchanged.

Prints, for each launch and output, the largest error / tolerance, and fails if any is >= 1; then
checks that wrong outputs (negative tests) exceed their tolerances, and prints tol / |output| at
real shapes.
Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "layernorm.py")
spec = importlib.util.spec_from_file_location("layernorm_host", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = triton.runtime.driver.active.get_active_torch_device()
P = lambda e: 2.0 ** e  # noqa: E731


def reference(x, res, w, b, eps, BD, G):
    """Post's reference and tolerances, in float64, row by row; row t uses group t % G."""
    T, D = x.shape
    s32 = x.float() + res.float() if res is not None else x.float()  # sv: the IEEE float32 sum, or x
    s = s32.double()
    grp = torch.arange(T, device=x.device) % G
    wd = w.double().view(G, D)[grp]
    bd = b.double().view(G, D)[grp]
    L = int(math.log2(BD))                    # depth = Nat.clog 2 BD
    mean_ref = s.sum(-1) / D
    abs_mean = s.abs().sum(-1) / D
    var_ref = ((s - mean_ref[:, None]) ** 2).sum(-1) / D
    rstd_ref = 1.0 / torch.sqrt(var_ref + eps)
    z_ref = (s - mean_ref[:, None]) * rstd_ref[:, None]
    y_ref = z_ref * wd + bd
    mean_tol = (L + 4) * P(-23) * abs_mean + P(-140)
    rms_s = torch.sqrt((s ** 2).sum(-1) / D)
    e_tol = mean_tol[:, None] + P(-23) * (s.abs() + abs_mean[:, None])
    cond = (mean_tol + P(-23) * (abs_mean + rms_s)) * rstd_ref
    gam = (L + 10) * P(-23) + P(-136) / eps
    rstd_tol = torch.where(4 * cond <= 1, 2 * (cond + gam) * rstd_ref,
                           torch.full_like(cond, 2.0 / math.sqrt(eps)))
    z_tol = (s - mean_ref[:, None]).abs() * rstd_tol[:, None] + e_tol * (rstd_ref + rstd_tol)[:, None]
    y_tol = P(-7) * y_ref.abs() + 4 * wd.abs() * z_tol + P(-21) * bd.abs() + P(-120)
    return dict(s32=s32, mean_ref=mean_ref, rstd_ref=rstd_ref, y_ref=y_ref, mean_tol=mean_tol,
                rstd_tol=rstd_tol, y_tol=y_tol, cond=cond, z_ref=z_ref)


def ratio(got, ref, tol):
    got = got.double()
    assert torch.isfinite(got).all(), "a stored value is not finite"
    return ((got - ref).abs() / tol).max().item()


def frame_check(x, res, w, b, eps, G):
    """A direct launch, with the host's arguments, into outputs inside guard bands."""
    T, D = x.shape
    GB = 4096
    def band(n, dt):
        return torch.full((n + 2 * GB,), -12345.0, dtype=dt, device=DEV)
    yb, rob = band(T * D, torch.bfloat16), band(T * D, torch.bfloat16)
    mb, rb = band(T, torch.float32), band(T, torch.float32)
    x0, w0, b0 = x.clone(), w.clone(), b.clone()
    r0 = res.clone() if res is not None else None
    BD = min(65536 // x.element_size(), triton.next_power_of_2(D))
    has_res = res is not None
    mod.layer_norm_fwd_kernel[(triton.cdiv(T, 32),)](
        x=x, y=yb[GB:GB + T * D].view(T, D), w=w, b=b, res=res,
        res_out=rob[GB:GB + T * D].view(T, D) if has_res else None,
        mean=mb[GB:GB + T], rstd=rb[GB:GB + T], eps=eps, T=T, G=G, D=D, BD=BD, NB=triton.cdiv(T, 2048),
        IS_RMS_NORM=False, HAS_RESIDUAL=has_res, STORE_RESIDUAL_OUT=has_res, HAS_WEIGHT=True,
        HAS_BIAS=True, BT=32, num_warps=4)
    torch.cuda.synchronize()
    bands = [yb, mb, rb] + ([rob] if has_res else [])
    ok = all(torch.equal(t[:GB], torch.full_like(t[:GB], -12345.0)) and
             torch.equal(t[-GB:], torch.full_like(t[-GB:], -12345.0)) for t in bands)
    if not has_res:  # res_out is not written: its whole buffer keeps the sentinel
        ok = ok and torch.equal(rob, torch.full_like(rob, -12345.0))
    ok = ok and torch.equal(x, x0) and torch.equal(w, w0) and torch.equal(b, b0)
    if has_res:
        ok = ok and torch.equal(res, r0)
    return ok


CHUNK = 1 << 16  # rows per reference chunk (keeps the float64 reference small)


def check_post(x, res, w, b, eps, G, y, mean, rstd, res_out):
    """Post on a launch's outputs, chunk by chunk: (res_out exact, mean, rstd, y ratios, max cond)."""
    T, D = x.shape
    BD = triton.next_power_of_2(D)
    exact, rm, rr, ry, cmax = True, 0.0, 0.0, 0.0, 0.0
    for t0 in range(0, T, CHUNK):
        t1 = min(T, t0 + CHUNK)
        # chunks start at multiples of CHUNK, a multiple of G for every G used here
        assert t0 % G == 0
        rs = res[t0:t1] if res is not None else None
        R = reference(x[t0:t1], rs, w, b, eps, BD, G)
        if res is not None:
            exact = exact and torch.equal(res_out[t0:t1].view(torch.int16),
                                          R["s32"].to(torch.bfloat16).view(torch.int16))
        rm = max(rm, ratio(mean[t0:t1], R["mean_ref"], R["mean_tol"]))
        rr = max(rr, ratio(rstd[t0:t1], R["rstd_ref"], R["rstd_tol"]))
        ry = max(ry, ratio(y[t0:t1], R["y_ref"], R["y_tol"]))
        cmax = max(cmax, R["cond"].max().item())
    return exact, rm, rr, ry, cmax


def fmt(r):
    """An error / tolerance: three decimals, or nine near 1."""
    return f"{r:.3f}" if r < 0.999 else f"{r:.9f}"


def run(name, x, res, w, b, eps, G=1):
    T, D = x.shape
    y, mean, rstd, res_out = mod.layer_norm_fwd(x, w, b, eps=eps, residual=res, num_groups=G)
    torch.cuda.synchronize()
    exact, rm, rr, ry, cmax = check_post(x, res, w, b, eps, G, y, mean, rstd, res_out)
    fr = frame_check(x, res, w, b, eps, G)
    tag = "res" if res is not None else "no res"
    print(f"{name:38s} T {T:6d} D {D:3d} G {G:2d} {tag:6s}: res_out exact {exact}, frame {fr}, "
          f"mean {fmt(rm)}, rstd {fmt(rr)}, y {fmt(ry)}  (max cond {cmax:.2e})")
    return exact and fr and max(rm, rr, ry) < 1, (rm, rr, ry)


def negatives(x, w, b, eps, G):
    """Wrong outputs that Post must reject (each must exceed its tolerance somewhere)."""
    T, D = x.shape
    BD = triton.next_power_of_2(D)
    y, mean, rstd, _ = mod.layer_norm_fwd(x, w, b, eps=eps, residual=None, num_groups=G)
    torch.cuda.synchronize()
    R = reference(x, None, w, b, eps, BD, G)
    w0 = w.double().view(G, D)[0][None, :]
    b0 = b.double().view(G, D)[0][None, :]
    grp = torch.arange(T, device=x.device) % G
    wd = w.double().view(G, D)[grp]
    bd = b.double().view(G, D)[grp]
    wrong = {
        "y = 0": (torch.zeros_like(y), R["y_ref"], R["y_tol"]),
        "y with group 0's weight and bias": ((R["z_ref"] * w0 + b0).to(torch.bfloat16), R["y_ref"], R["y_tol"]),
        "y without the bias": ((R["z_ref"] * wd).to(torch.bfloat16), R["y_ref"], R["y_tol"]),
        "y with x for x - mean": ((x.double() * R["rstd_ref"][:, None] * wd + bd).to(torch.bfloat16),
                                  R["y_ref"], R["y_tol"]),
        "mean * (1 + 2^-12)": (mean.double() * (1 + P(-12)), R["mean_ref"], R["mean_tol"]),
        "rstd * (1 + 2^-14)": (rstd.double() * (1 + P(-14)), R["rstd_ref"], R["rstd_tol"]),
        "rstd without eps": (1.0 / torch.sqrt(((x.double() - R["mean_ref"][:, None]) ** 2).mean(-1)),
                             R["rstd_ref"], R["rstd_tol"]),
    }
    ok = True
    for k, (got, ref, tol) in wrong.items():
        r = ((got.double() - ref).abs() / tol).max().item()
        print(f"  negative {k:34s}: worst error / tolerance {r:10.3f} -> {'rejected' if r > 1 else 'ACCEPTED'}")
        ok = ok and r > 1
    return ok


def meaning(x, res, w, b, eps, G, label):
    """tol / |output| at a real shape: median and worst, per output."""
    T, D = x.shape
    BD = triton.next_power_of_2(D)
    R = reference(x[:CHUNK], res[:CHUNK] if res is not None else None, w, b, eps, BD, G)
    def q(t):
        t = t.flatten()
        return f"median {t.median().item():.2e}, worst {t.max().item():.2e}"
    print(f"  tol / |output| at {label}: mean (vs mean |s|) "
          f"{q(R['mean_tol'] / (x[:CHUNK].double().abs().mean(-1) + 1e-300))}; rstd {q(R['rstd_tol'] / R['rstd_ref'])}; "
          f"y {q(R['y_tol'] / R['y_ref'].abs().clamp(min=1e-300))}")


def bf(t):
    return t.to(device=DEV, dtype=torch.bfloat16)


def main():
    gen = torch.Generator().manual_seed(0)
    rn = lambda *s: torch.randn(s, generator=gen)  # noqa: E731
    ok, worst = True, [0.0, 0.0, 0.0]

    def go(name, x, res, w, b, eps=1e-5, G=1):
        nonlocal ok
        good, rs = run(name, bf(x), bf(res) if res is not None else None, bf(w), bf(b), eps, G)
        ok = ok and good
        for i in range(3):
            worst[i] = max(worst[i], rs[i])

    # The host's test launches (source lines 203-219): LayerNorm with a residual, G = 1.
    for T, D in [(1, 1), (1, 64), (31, 50), (2048, 50), (2048, 64), (2048, 128), (33, 512),
                 (1000, 300), (65536, 256)]:
        go(f"setup ({T}, {D})", rn(T, D), rn(T, D), 1 + 0.1 * rn(D), 0.1 * rn(D))

    # GroupNorm without a residual (setup_rwkv7_*, setup_rwkv6_*, setup_gn_test_*).
    eps7 = 64 * 1e-5
    go("setup rwkv7 g_norm, 4 x 4096 tokens", rn(524288, 64), None, 1 + 0.1 * rn(32 * 64),
       0.1 * rn(32 * 64), eps=eps7, G=32)
    go("setup rwkv7 g_norm, one token", rn(32, 64), None, 1 + 0.1 * rn(32 * 64),
       0.1 * rn(32 * 64), eps=eps7, G=32)
    go("setup rwkv6 g_norm, 4 x 4096 tokens", rn(65536, 512), None, 1 + 0.1 * rn(4 * 512),
       0.1 * rn(4 * 512), G=4)
    go("setup fla GroupNorm test, hidden 512", rn(4096, 128), None, rn(4 * 128), rn(4 * 128), G=4)
    go("setup fla GroupNorm test, hidden 2048", rn(4096, 512), None, rn(4 * 512), rn(4 * 512), G=4)
    for hid, G in ((64, 1), (128, 1), (512, 1), (64, 4), (128, 4), (1024, 4)):
        D = hid // G
        go(f"setup fla GroupNorm test, hidden {hid}, G {G}", rn(1024 * G, D), None, rn(hid), rn(hid), G=G)
    # LayerNorm without a residual (setup_ln_*): fla's LayerNorm test, D in 50, 64, 128.
    for D in (50, 64, 128):
        go(f"setup fla LayerNorm test, D {D}", rn(2048, D), None, rn(D), rn(D))

    # Edges of Inst.ok and Pre.
    T = 100
    go("D=1, many rows", rn(T, 1), rn(T, 1), 1 + 0.1 * rn(1), 0.1 * rn(1))
    go("D=2", rn(T, 2), rn(T, 2), 1 + 0.1 * rn(2), 0.1 * rn(2))
    go("D=257 (BD=512)", rn(T, 257), rn(T, 257), 1 + 0.1 * rn(257), 0.1 * rn(257))
    go("D=512, T=1", rn(1, 512), rn(1, 512), 1 + 0.1 * rn(512), 0.1 * rn(512))
    c = torch.full((T, 300), 1000.5)
    go("constant rows, offset 1000", c, torch.zeros(T, 300), 1 + 0.1 * rn(300), 0.1 * rn(300))
    go("near-constant rows, offset 100", 100 + 0.01 * rn(T, 300), 0.01 * rn(T, 300),
       1 + 0.1 * rn(300), 0.1 * rn(300))
    go("near-constant rows, offset 3e4", 30000 + 64 * rn(T, 333), 64 * rn(T, 333),
       1 + 0.1 * rn(333), 0.1 * rn(333))
    go("ill-conditioned rows, D=3, offset 3e4", torch.full((4096, 3), 30000.0), 1e-3 * rn(4096, 3),
       1 + 0.1 * rn(3), 0.1 * rn(3))
    go("ill-conditioned rows, D=3, offset 3e4, eps 2^-125", torch.full((4096, 3), 30000.0),
       1e-3 * rn(4096, 3), 1 + 0.1 * rn(3), 0.1 * rn(3), eps=P(-125))
    go("near-constant rows, no res, G=3", 30000 + 64 * rn(T, 333), None,
       1 + 0.1 * rn(3 * 333), 0.1 * rn(3 * 333), G=3)
    go("tiny rows (1e-30)", 1e-30 * rn(T, 64), 1e-30 * rn(T, 64), 1 + 0.1 * rn(64), 0.1 * rn(64))
    go("subnormal rows", 1e-39 * rn(T, 64), 1e-39 * rn(T, 64), 1 + 0.1 * rn(64), 0.1 * rn(64))
    go("subnormal rows, no res, G=4", 1e-39 * rn(T, 64), None, 1 + 0.1 * rn(256), 0.1 * rn(256), G=4)
    sgn = lambda *s: torch.where(torch.rand(s, generator=gen) < 0.5, -1.0, 1.0)  # noqa: E731
    go("|x| = |res| = 2^56, D=512", P(56) * sgn(T, 512), P(56) * sgn(T, 512),
       1 + 0.1 * rn(512), 0.1 * rn(512))
    go("|x| = 2^56, no res, D=512, G=2", P(56) * sgn(T, 512), None,
       1 + 0.1 * rn(1024), 0.1 * rn(1024), G=2)
    go("|x|,|res| <= 2^56 mixed scales", P(56) * rn(T, 512).clamp(-1, 1) * (rn(T, 512) > 1),
       rn(T, 512), 1 + 0.1 * rn(512), 0.1 * rn(512))
    go("|w| = |b| = 2^16", rn(T, 128), rn(T, 128), P(16) * sgn(128), P(16) * sgn(128))
    go("|w| = |b| = 2^16, no res, G=5", rn(T, 128), None, P(16) * sgn(640), P(16) * sgn(640), G=5)
    go("|w| = |b| = 2^16, ill-conditioned, eps 2^-125", torch.full((T, 3), 30000.0), 1e-3 * rn(T, 3),
       P(16) * sgn(3), P(16) * sgn(3), eps=P(-125))
    go("w = 0, b = 0", rn(T, 128), rn(T, 128), torch.zeros(128), torch.zeros(128))
    go("eps = 2^-125, random rows", rn(T, 64), rn(T, 64), 1 + 0.1 * rn(64), 0.1 * rn(64), eps=P(-125))
    go("eps = 2^-125, constant rows", torch.full((T, 50), 3.0), torch.zeros(T, 50),
       1 + 0.1 * rn(50), 0.1 * rn(50), eps=P(-125))
    go("eps = 2^-125, constant rows, no res", torch.full((T, 50), 3.0), None,
       1 + 0.1 * rn(50), 0.1 * rn(50), eps=P(-125))
    go("eps = 2^-125, tiny rows", 1e-38 * rn(T, 64), 1e-38 * rn(T, 64), 1 + 0.1 * rn(64),
       0.1 * rn(64), eps=P(-125))
    go("eps = 2^126", rn(T, 64), rn(T, 64), 1 + 0.1 * rn(64), 0.1 * rn(64), eps=P(126))
    go("eps = 1.0", rn(T, 64), rn(T, 64), 1 + 0.1 * rn(64), 0.1 * rn(64), eps=1.0)
    go("T = 33 (partial block), D = 7", rn(33, 7), rn(33, 7), 1 + 0.1 * rn(7), 0.1 * rn(7))
    go("T = 37, G = 7 (T not a multiple of G)", rn(37, 64), None, rn(7 * 64), rn(7 * 64), G=7)
    go("T = 5 < G = 32", rn(5, 64), None, rn(32 * 64), rn(32 * 64), eps=eps7, G=32)
    go("G = 1024, D = 1", rn(4096, 1), None, rn(1024), rn(1024), G=1024)

    print(f"worst error / tolerance: mean {fmt(worst[0])}, rstd {fmt(worst[1])}, y {fmt(worst[2])}")

    # Negative tests: wrong outputs at RWKV-7's GroupNorm shape (8192 rows of 64, 32 groups).
    print("negative tests (RWKV-7 g_norm shape, 8192 rows):")
    xn, wn, bn = bf(rn(8192, 64)), bf(1 + 0.1 * rn(32 * 64)), bf(0.1 * rn(32 * 64))
    neg_ok = negatives(xn, wn, bn, eps7, 32)
    ok = ok and neg_ok

    # Meaningfulness: tol / |output| at real shapes.
    meaning(bf(rn(65536, 64)), None, bf(1 + 0.1 * rn(32 * 64)), bf(0.1 * rn(32 * 64)), eps7, 32,
            "RWKV-7 g_norm (D 64, G 32)")
    meaning(bf(rn(65536, 512)), None, bf(1 + 0.1 * rn(4 * 512)), bf(0.1 * rn(4 * 512)), 1e-5, 4,
            "RWKV-6 g_norm (D 512, G 4)")
    meaning(bf(rn(2048, 64)), bf(rn(2048, 64)), bf(1 + 0.1 * rn(64)), bf(0.1 * rn(64)), 1e-5, 1,
            "LayerNorm with a residual (2048 x 64)")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
