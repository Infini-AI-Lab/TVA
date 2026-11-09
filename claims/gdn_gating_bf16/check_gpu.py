"""GPU check of the prepared claim for `gdn_gating_bf16` (claims/README.md, check 4).

Runs the kernel's own host (`source/fused_gdn_gating.py`) on the setup's
launches (the host's test cases and SETUP.md's configurations) and on launches at the edges of
`Inst.ok` and `Pre` (every finite `bfloat16` value of `a` and of `b`, `A_log` from `-3e38` to the
largest value `ExpRoom` admits, sums `a + dt_bias` at the softplus threshold, near `+2^128` and
rounding to `-inf`), and checks `Post` exactly as `Claim.lean` states it, in `float64` from the
host's inputs:

* `g`: finite and `|g - gRef| <= gTol` (`gTol = gEps * gMag`), or `-inf` with
  `maxVal <= |gRef| + gTol`; and that README Tolerances' fine bound lies below `gTol`;
* `beta_output`: finite and `|beta - sigRef| <= bTol`;
* the frame: in direct launches with the host's own launch line, on outputs carved from a
  random-filled pool, every guard byte and every input is unchanged.

Inputs outside `Pre` (sums that overflow to `+inf`, `exp(A_log)` or `exp(-b)` without room) are
run too and reported separately (they are the Exclusions). Run with
`CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "fused_gdn_gating.py")
spec = importlib.util.spec_from_file_location("gdn_gating_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
MAXF32 = (2 - 2.0 ** -23) * 2.0 ** 127
D = torch.float64


def exp_room(t):
    """`ExpRoom t` in float64 (an overflowing `exp` is `inf`, so no room)."""
    with torch.no_grad():
        return (1 + (2.0 ** -22 + 2.0 ** -23 * t.abs())) * torch.exp(t) + 2.0 ** -140 <= MAXF32


def sp_ref(x):
    return torch.where(x <= 20, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x)


def g_ref(A, x):
    return -(torch.exp(A) * sp_ref(x))


def g_tol(A, x):
    """`gTol = gEps A * gMag A x` of Claim.lean."""
    eps = 2.0 ** -23 * (64 + torch.clamp(A.abs(), max=128.0))
    return eps * (torch.exp(A) + 2.0 ** -121) * (sp_ref(x) + 2.0 ** -6)


def g_fine(A, x):
    """The fine bound of README Tolerances (before the relaxation to `gTol`), for the comparison."""
    rel = 2.0 ** -19 + 2.0 ** -23 * (A.abs() + 2 * torch.clamp(-x, min=0.0, max=20.0))
    return rel * g_ref(A, x).abs() + 2.0 ** -23 * torch.exp(A) + 2.0 ** -139 * sp_ref(x) + 2.0 ** -148


def sig_ref(b):
    return 1 / (1 + torch.exp(-b))


def b_tol(b):
    return 2.0 ** -7 * sig_ref(b).abs() + 2.0 ** -133


class Stats:
    def __init__(self):
        self.g = 0.0
        self.b = 0.0
        self.n = 0
        self.ninf = 0
        self.fail = 0
        self.excl = 0
        self.excl_bad = 0
        self.gx = {}
        self.fine_ok = True      # the fine bound of README Tolerances lies below gTol on every covered cell
        self.real = {}           # tolerance / |output| samples on the setup's launches
        self.realmax = {}

    def check(self, name, A_log, a, b, dt_bias, g, beta, real=False):
        """`Post`'s value clauses on one launch; the inputs are `[batch, H]` views (as the host
        passes them), the outputs `[1, batch, H]`."""
        A = A_log.double()[None, :]
        x = a.double() + dt_bias.double()[None, :]
        bb = b.double()
        gg = g[0].double()
        be = beta[0].double()
        covered = exp_room(A) & (x < 2.0 ** 128 - 2.0 ** 103) & exp_room(-bb)
        ref = g_ref(A, x)
        tol = g_tol(A, x)
        fin = torch.isfinite(gg)
        ok_fin = fin & ((gg - ref).abs() <= tol)
        ok_inf = (gg == -math.inf) & (MAXF32 <= ref.abs() + tol)
        ok_g = ok_fin | ok_inf
        r = torch.where(fin, (gg - ref).abs() / tol, torch.zeros_like(tol))
        bref = sig_ref(bb)
        ok_b = torch.isfinite(be) & ((be - bref).abs() <= b_tol(bb))
        rb = (be - bref).abs() / b_tol(bb)
        cov_g = covered & ok_g
        cov_b = covered & ok_b
        bad = covered & ~(ok_g & ok_b)
        self.n += int(covered.sum())
        self.ninf += int((covered & ok_inf & ~ok_fin).sum())
        if covered.any():
            self.g = max(self.g, float(r[covered].max()))
            self.b = max(self.b, float(rb[covered].max()))
        fine = g_fine(A, x)
        self.fine_ok = self.fine_ok and bool((fine[covered] <= tol[covered] * (1 + 1e-12)).all())
        if real:
            sel = covered & (ref != 0)
            for key, t in (("g", tol), ("g fine", fine), ("beta", b_tol(bb))):
                rr = (t / (bref if key == "beta" else ref).abs())[sel]
                self.real.setdefault(key, []).append(rr.flatten()[::max(1, rr.numel() // 200000)].cpu())
                self.realmax[key] = max(self.realmax.get(key, 0.0), float(rr.max()))
        for lo, hi in ((-100.0, -16.0), (-16.0, -8.0), (-8.0, 0.0), (0.0, 20.0), (20.0, 1e40)):
            sel = covered & fin & (x >= lo) & (x < hi) & (A.abs() <= 20)
            if sel.any():
                e = float(((gg - ref).abs() / ref.abs().clamp(min=1e-300))[sel].max())
                t = float((tol / ref.abs().clamp(min=1e-300))[sel].max())
                o = self.gx.get((lo, hi), (0.0, 0.0))
                self.gx[(lo, hi)] = (max(o[0], e), max(o[1], t))
        nbad = int(bad.sum())
        self.fail += nbad
        ex = ~covered
        self.excl += int(ex.sum())
        self.excl_bad += int((ex & ~(ok_g & ok_b)).sum())
        gmax = float(r[covered].max()) if covered.any() else 0.0
        bmax = float(rb[covered].max()) if covered.any() else 0.0
        print(f"{name}: {int(covered.sum())} covered cells, g err/tol {gmax:.3f}, beta err/tol "
              f"{bmax:.3f}, -inf cells {int((covered & ok_inf & ~ok_fin).sum())}, failures {nbad}; "
              f"outside Pre {int(ex.sum())} (of them outside Post {int((ex & ~(ok_g & ok_b)).sum())})")
        if nbad:
            idx = bad.nonzero()[:5]
            for i, h in idx.tolist():
                print("   FAIL", float(A[0, h]), float(x[i, h]), float(gg[i, h]), float(ref[i, h]),
                      float(tol[i, h]), float(bb[i, h]), float(be[i, h]))


def inputs(gen, batch, H, split, scale=8.0):
    A_log = torch.randn(H, generator=gen).to(device=DEV, dtype=torch.float32)
    dt_bias = torch.randn(H, generator=gen).to(device=DEV, dtype=torch.bfloat16)
    if split:
        mixed = (torch.randn(batch, 2 * H, generator=gen) * scale).to(device=DEV, dtype=torch.bfloat16)
        b, a = mixed.split([H, H], dim=-1)
    else:
        a = (torch.randn(batch, H, generator=gen) * scale).to(device=DEV, dtype=torch.bfloat16)
        b = (torch.randn(batch, H, generator=gen) * scale).to(device=DEV, dtype=torch.bfloat16)
    return A_log, a, b, dt_bias


def run_host(st, name, A_log, a, b, dt_bias, real=False):
    copies = [t.clone() for t in (A_log, a, b, dt_bias)]
    g, beta = mod.fused_gdn_gating(A_log, a, b, dt_bias)
    torch.cuda.synchronize()
    for t, c in zip((A_log, a, b, dt_bias), copies):
        assert torch.equal(t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32),
                           c.view(torch.int16) if c.dtype == torch.bfloat16 else c.view(torch.int32)), \
            f"{name}: an input changed"
    st.check(name, A_log, a, b, dt_bias, g, beta, real)


def run_direct(st, name, A_log, a, b, dt_bias, gen):
    """The host's launch line on outputs carved from a random-filled pool with guards."""
    batch, H = a.shape
    n = batch * H
    G = 1024
    pool = torch.randint(-2 ** 31, 2 ** 31 - 1, (2 * n + 3 * G,), generator=gen, dtype=torch.int32).to(DEV)
    init = pool.clone()
    g = pool[G:G + n].view(torch.float32).view(1, batch, H)
    beta = pool[2 * G + n:2 * G + 2 * n].view(torch.float32).view(1, batch, H)
    copies = [t.clone() for t in (A_log, a, b, dt_bias)]
    grid = (batch, 1, triton_cdiv(H, 8))
    mod.fused_gdn_gating_kernel[grid](g, beta, A_log, a, b, dt_bias, 1, a.stride(0), b.stride(0), H,
                                      1.0, 20.0, 8, num_warps=1, num_stages=3)
    torch.cuda.synchronize()
    guard = torch.ones_like(pool, dtype=torch.bool)
    guard[G:G + n] = False
    guard[2 * G + n:2 * G + 2 * n] = False
    assert torch.equal(pool[guard], init[guard]), f"{name}: a guard byte changed"
    for t, c in zip((A_log, a, b, dt_bias), copies):
        assert torch.equal(t.float(), c.float()) or torch.equal(t.isnan(), c.isnan()), f"{name}: input changed"
    st.check(name + " (direct, guards)", A_log, a, b, dt_bias, g, beta)


def triton_cdiv(x, y):
    return (x + y - 1) // y


def all_bf16_finite():
    pats = torch.arange(-2 ** 15, 2 ** 15, dtype=torch.int32).to(torch.int16)
    v = pats.view(torch.bfloat16)
    return v[torch.isfinite(v)]


def main():
    gen = torch.Generator().manual_seed(0)
    st = Stats()
    # The host's test cases (source lines 134-155), with its values past the threshold.
    for batch, H, split in [(1, 32, False), (1, 48, True), (4, 8, True), (16, 48, True), (32, 24, True),
                            (7, 4, False), (513, 16, False), (8192, 32, False), (32768, 32, True)]:
        A_log, a, b, dt_bias = inputs(gen, batch, H, split)
        a[0, :2] = torch.tensor([25.0, -30.0], dtype=torch.bfloat16)
        run_host(st, f"test batch {batch} heads {H} split {split}", A_log, a, b, dt_bias, real=True)
    # SETUP.md's configurations (Qwen3-Next 32 heads at TP 1, 2, 4, 8; Qwen3.5 48 and 24).
    for batch, H, split in [(65536, 32, True), (65536, 16, False), (32768, 8, True), (65536, 4, True),
                            (65536, 48, True), (32768, 24, False)]:
        A_log, a, b, dt_bias = inputs(gen, batch, H, split)
        A_log = torch.log(torch.empty(H).uniform_(1e-3, 16, generator=gen)).to(DEV)  # A ~ U(0, 16)
        run_host(st, f"config batch {batch} heads {H} split {split}", A_log, a, b, dt_bias, real=True)
    # Edges: every finite bfloat16 `a` (one per token) against a spread of `A_log` (one per head),
    # dt_bias 0 or small; every finite bfloat16 `b`.
    vals = all_bf16_finite().to(DEV)
    T = vals.numel()
    A_vals = torch.tensor([-3e38, -1000.0, -150.0, -100.0, -20.0, -2.0, 0.0, 1.0, 2.77, 10.0, 20.0, 50.0,
                           88.0, 88.72282, 88.72283, 89.0], dtype=torch.float32)
    H = A_vals.numel()
    for bias_kind in ("zero", "random", "huge"):
        A_log = A_vals.to(DEV)
        if bias_kind == "zero":
            dt_bias = torch.zeros(H, dtype=torch.bfloat16, device=DEV)
        elif bias_kind == "random":
            dt_bias = (torch.randn(H, generator=gen) * 4).to(device=DEV, dtype=torch.bfloat16)
        else:
            dt_bias = torch.tensor([3.38e38, -3.38e38] * (H // 2), dtype=torch.bfloat16, device=DEV)
        a = vals[:, None].expand(T, H).contiguous()
        b = vals[torch.randperm(T, generator=gen).to(DEV)][:, None].expand(T, H).contiguous()
        run_host(st, f"edge: every finite a, every finite b, dt_bias {bias_kind}", A_log, a, b, dt_bias)
    # The softplus threshold: sums `a + dt_bias` within a few float32 ulps of 20 (a = 20 and its
    # bfloat16 neighbours, dt_bias tiny of both signs, so that the float32 sum rounds).
    a = torch.tensor([19.875, 20.0, 20.125], dtype=torch.bfloat16).repeat_interleave(64)[:, None]
    tiny = torch.tensor([2.0 ** -k for k in range(14, 30)] + [-(2.0 ** -k) for k in range(14, 30)],
                        dtype=torch.bfloat16)
    H = tiny.numel()
    a = a.expand(a.shape[0], H).contiguous().to(DEV)
    A_log = torch.zeros(H, dtype=torch.float32, device=DEV)
    b = torch.zeros_like(a)
    run_host(st, "edge: threshold", A_log, a, b, tiny.to(DEV))
    # Direct launches with guards (the frame), contiguous and split.
    for batch, H, split in [(513, 16, False), (1000, 48, True)]:
        A_log, a, b, dt_bias = inputs(gen, batch, H, split)
        run_direct(st, f"frame batch {batch} heads {H} split {split}", A_log, a, b, dt_bias, gen)
    print(f"TOTAL: {st.n} covered cells; worst g err/tol {st.g:.4f}, worst beta err/tol {st.b:.4f}; "
          f"{st.ninf} cells -inf by the overflow clause; failures {st.fail}; outside Pre {st.excl} "
          f"(outside Post there: {st.excl_bad})")
    for key in ("g", "g fine", "beta"):
        x = torch.cat(st.real[key])
        print(f"  setup launches, tolerance/|output| {key:6s}: median {float(x.median()):.3g}, worst {st.realmax[key]:.3g}")
    print(f"  the fine bound of README Tolerances lies below gTol on every covered cell: {st.fine_ok}")
    for (lo, hi), (e, t) in sorted(st.gx.items()):
        print(f"  x in [{lo:g}, {hi:g}), |A_log| <= 20: largest |g - gRef|/|gRef| {e:.3e}, largest gTol/|gRef| {t:.3e}")
    assert st.fail == 0 and st.fine_ok
    print("Post holds on every covered cell")


if __name__ == "__main__":
    main()
