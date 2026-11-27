"""GPU check of the claim for `gelu_and_mul_sparse_bf16` (`Claim.lean`).

Runs the kernel's own host, `gelu_and_mul_sparse_triton` of source/activation.py (imported by
path), on the setup launches of Claim.lean (Gemma 3n's sparse MLP shapes, the source's test
shapes, both multipliers) and on launches at the edges of `Inst.ok` and `Pre`, and computes
`Post`'s reference and tolerances in float64 exactly as `Post` states them, from the host's
inputs (`g`, `u` the row's `bfloat16` gate and up, `m` the host's double):

* `M`, `A`, `Sd`: the exact mean, mean magnitude and biased standard deviation of the gate;
  `cRef = M + m*Sd`; `cutTol = 2^-6 (|M| + |m| Sd) + 2^-17 (1 + |m|) A + 2^-130`;
* `sRef = max(g - cRef, 0)`, `yRef = geluRef(sRef) * u` with
  `geluRef x = x/2 (1 + tanh(0.7978845608028654 (x + 0.044715 x^3)))`;
* every output finite and `|out - yRef| <= yTol = 2^-5 |yRef| + 2 |u| cutTol + 2^-132`;
* where `g <= cRef - cutTol`, the output is a zero;
* `Pre` of each input: every element finite with `|v| <= 2^32` (`InOk`) and every row
  `2^-18 A + 2^-62 <= Sd` (`RowOk`); a launch with an input outside `Pre` is reported as such and
  its outputs are not checked against `Post`;
* the frame: a direct launch with the host's arguments into an output inside guard bands leaves
  the guards and `x` unchanged, and writes the host's output bit for bit.

Prints one line per launch (the largest error / tolerance, and the share of elements under the
zero clause), fails if any ratio is >= 1; then checks that wrong outputs (negative tests) exceed
their tolerances, and prints tol / |yRef| at a real shape. Each case keeps its device memory under
4 GB.
Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "activation.py")
spec = importlib.util.spec_from_file_location("activation_host", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
P = lambda e: 2.0 ** e  # noqa: E731
K, A3 = 0.7978845608028654, 0.044715
M_GEMMA = 1.6448535919189453   # mGemma: Normal(0, 1).icdf(torch.tensor(0.95)).item()
M_GEN = 1.6448536269514722     # mGen: the default of vLLM's input generator
CHUNK = 512                    # rows per float64 reference chunk


def gelu_ref(x):
    return x / 2 * (1 + torch.tanh(K * (x + A3 * x ** 3)))


def row_stats(g):
    """`meanRef`, `absMean`, `stdRef` of each row (float64, keepdim)."""
    M = g.mean(-1, keepdim=True)
    A = g.abs().mean(-1, keepdim=True)
    Sd = ((g - M) ** 2).mean(-1, keepdim=True).sqrt()
    return M, A, Sd


def reference(xc, m):
    """Post's reference and tolerances for a chunk of rows of `x` (bfloat16 -> float64)."""
    xd = xc.double()
    d = xd.shape[-1] // 2
    g, u = xd[:, :d], xd[:, d:]
    M, A, Sd = row_stats(g)
    c = M + m * Sd
    cut = P(-6) * (M.abs() + abs(m) * Sd) + P(-17) * (1 + abs(m)) * A + P(-130)
    s = (g - c).clamp(min=0)
    y = gelu_ref(s) * u
    tol = P(-5) * y.abs() + 2 * u.abs() * cut + P(-132)
    in_ok = bool((torch.isfinite(xd) & (xd.abs() <= P(32))).all())
    row_ok = bool((P(-18) * A + P(-62) <= Sd).all())
    return dict(g=g, u=u, c=c, cut=cut, s=s, y=y, tol=tol, pre=in_ok and row_ok,
                in_ok=in_ok, row_ok=row_ok)


def check_post(x, m, out):
    """Post on a launch's output: (in Pre, worst ratio, zero clause holds, share under it)."""
    T = x.shape[0]
    pre, worst, zero_ok, nz, n = True, 0.0, True, 0, 0
    for t0 in range(0, T, CHUNK):
        R = reference(x[t0:t0 + CHUNK], m)
        pre = pre and R["pre"]
        o = out[t0:t0 + CHUNK].double()
        if not torch.isfinite(o).all():
            return pre, float("inf"), False, 0.0
        worst = max(worst, ((o - R["y"]).abs() / R["tol"]).max().item())
        below = R["g"] <= R["c"] - R["cut"]
        zero_ok = zero_ok and bool((o[below] == 0).all())
        nz += int(below.sum())
        n += below.numel()
    return pre, worst, zero_ok, nz / n


def frame_check(x, m, out_host):
    """A direct launch with the host's arguments into an output inside guard bands."""
    T, d = x.shape[0], x.shape[1] // 2
    GB = 4096
    band = torch.full((T * d + 2 * GB,), -12345.0, dtype=torch.bfloat16, device=DEV)
    x0 = x.clone()
    BS = triton.next_power_of_2(d)
    mod._gelu_and_mul_sparse_kernel[(T,)](
        x, band[GB:GB + T * d], d=d, std_multiplier=m, BLOCK_SIZE=BS,
        num_warps=min(max(BS // 256, 1), 8))
    torch.cuda.synchronize()
    sent = torch.full((GB,), -12345.0, dtype=torch.bfloat16, device=DEV)
    ok = torch.equal(band[:GB], sent) and torch.equal(band[-GB:], sent) and torch.equal(x, x0)
    ok = ok and torch.equal(band[GB:GB + T * d].view(torch.int16),
                            out_host.reshape(-1).view(torch.int16))
    del band, x0
    return ok


def fmt(r):
    return f"{r:.3f}" if r < 0.999 else f"{r:.9f}"


class Run:
    def __init__(self):
        self.ok = True
        self.worst = 0.0
        self.outside = []

    def go(self, name, x, m):
        T, d = x.shape[0], x.shape[1] // 2
        assert mod._supports_gelu_and_mul_sparse(x, m, "tanh")
        out = mod.gelu_and_mul_sparse_triton(x, m, "tanh")
        torch.cuda.synchronize()
        pre, r, zok, zs = check_post(x, m, out)
        fr = frame_check(x, m, out)
        if not pre:
            print(f"{name:52s} T {T:5d} d {d:5d} m {m:+.4f}: OUTSIDE Pre (not checked), frame {fr}")
            self.outside.append(name)
            self.ok = self.ok and fr
        else:
            good = fr and zok and r < 1
            print(f"{name:52s} T {T:5d} d {d:5d} m {m:+.4f}: y {fmt(r)}, zero clause {zok} "
                  f"({100 * zs:5.1f}% under it), frame {fr}{'' if good else '  <-- FAIL'}")
            self.ok = self.ok and good
            self.worst = max(self.worst, r)
        del out
        torch.cuda.empty_cache()


def rand_x(T, d, gen, gscale=1.0, gmean=0.0, uscale=1.0):
    """`[T, 2d]` bfloat16: gate `gmean + gscale*N(0,1)`, up `uscale*N(0,1)`, built row-chunk-wise."""
    x = torch.empty(T, 2 * d, dtype=torch.bfloat16, device=DEV)
    for t0 in range(0, T, CHUNK):
        t1 = min(T, t0 + CHUNK)
        g = torch.randn(t1 - t0, d, generator=gen, device=DEV) * gscale + gmean
        u = torch.randn(t1 - t0, d, generator=gen, device=DEV) * uscale
        x[t0:t1, :d] = g.to(torch.bfloat16)
        x[t0:t1, d:] = u.to(torch.bfloat16)
    return x


def near_cutoff(T, d, m, gen, frac=0.1, iters=12):
    """Rows where a share `frac` of the gate lies within a few cutTol of the row's own cRef (a
    fixed-point iteration: the other elements are N(0,1), so cRef settles)."""
    x = rand_x(T, d, gen)
    n = max(1, int(frac * d))
    noise = torch.randn(T, n, generator=gen, device=DEV, dtype=torch.float64)
    for _ in range(iters):
        R = reference(x, m)
        x[:, :n] = (R["c"] + 2 * R["cut"] * noise).to(torch.bfloat16)
    R = reference(x, m)
    near = ((R["g"] - R["c"]).abs() <= 3 * R["cut"]).double().mean().item()
    print(f"  (near-cutoff rows: {100 * near:.1f}% of the gate within 3 cutTol of cRef)")
    return x


def negatives(x, m):
    """Wrong outputs that Post must reject (each must exceed its tolerance somewhere)."""
    out = mod.gelu_and_mul_sparse_triton(x, m, "tanh").double()
    R = reference(x, m)
    g, u = R["g"], R["u"]
    M, A, Sd = row_stats(g)
    info = {
        "exact (erf) GELU": torch.nn.functional.gelu(R["s"]) * u,
        "no cube term (0.044715 -> 0)": R["s"] / 2 * (1 + torch.tanh(K * R["s"])) * u,
    }
    wrong = {
        "no sparsity: geluRef(gate) * up": gelu_ref(g) * u,
        "cutoff = mean (no std * m)": gelu_ref((g - M).clamp(min=0)) * u,
        "cutoff = mean - m * std": gelu_ref((g - (M - m * Sd)).clamp(min=0)) * u,
        "gate and up swapped": None,
        "output * (1 + 2^-3)": out * (1 + P(-3)),
        "output = 0": torch.zeros_like(out),
    }
    xs = torch.cat([x[:, x.shape[1] // 2:], x[:, :x.shape[1] // 2]], dim=1)
    wrong["gate and up swapped"] = reference(xs, m)["y"]
    ok = True
    for k, got in wrong.items():
        r = ((got - R["y"]).abs() / R["tol"]).max().item()
        print(f"  negative {k:34s}: worst error / tolerance {r:12.3f} -> "
              f"{'rejected' if r > 1 else 'ACCEPTED'}")
        ok = ok and r > 1
    for k, got in info.items():
        r = ((got - R["y"]).abs() / R["tol"]).max().item()
        print(f"  (within the tolerance, not a negative) {k:28s}: worst error / tolerance {r:.3f}")
    return ok


def meaning(x, m, label):
    """tol / |yRef| on the elements above the cutoff, at a real shape."""
    R = reference(x, m)
    keep = R["s"] > 0
    q = (R["tol"] / R["y"].abs().clamp(min=1e-300))[keep & (R["y"] != 0)]
    q1 = (R["tol"] / R["y"].abs().clamp(min=1e-300))[(R["s"] >= 1) & (R["y"] != 0)]
    print(f"  tol / |yRef| above the cutoff at {label}: median {q.median().item():.3e}, "
          f"90th percentile {q.quantile(0.9).item():.3e}; where sRef >= 1: median "
          f"{q1.median().item():.3e}; cutTol / Sd median "
          f"{(R['cut'] / row_stats(R['g'])[2]).median().item():.3e}")


def main():
    gen = torch.Generator(device=DEV).manual_seed(0)
    run = Run()

    # setup_*: Gemma 3n's sparse MLP (E4B d = 16384 / tp, E2B d = 8192 / tp), decode and prefill.
    for name, T, d in [("setup e4b tp1 decode", 1, 16384), ("setup e4b tp1 prefill", 8192, 16384),
                       ("setup e4b tp2 prefill", 8192, 8192), ("setup e4b tp4 prefill", 8192, 4096),
                       ("setup e4b tp8 prefill", 8192, 2048), ("setup e2b tp1 decode", 1, 8192),
                       ("setup e2b tp1 prefill", 8192, 8192), ("setup e2b tp8 prefill", 8192, 1024)]:
        run.go(name, rand_x(T, d, gen), M_GEMMA)
    # The source's test shapes and the input generator's multiplier.
    for T, d in [(1, 8192), (7, 16384), (33, 4096), (5, 100)]:
        run.go(f"setup test ({T}, {d})", rand_x(T, d, gen), M_GEMMA)
    run.go("setup gen (8192, 16384), mGen", rand_x(8192, 16384, gen), M_GEN)

    # vLLM's tests of this kernel (tests/kernels/ir/test_activation.py, bfloat16, mGen): Inst.ok
    # covers them (mkInst_ok); the constant rows of its constant-row test are outside Pre.
    for T in (1, 8, 64):
        for d in (2048, 4096, 8192, 13824, 16384, 32768):
            run.go(f"vllm test ({T}, {d})", rand_x(T, d, gen), M_GEN)
    for d in (31, 128):
        run.go(f"vllm small-size test (5, {d})", rand_x(5, d, gen), M_GEN)
    run.go("vllm second-seed test (32, 8192)", rand_x(32, 8192, gen), M_GEN)
    xk = rand_x(4, 8192, gen); xk[:, :8192] = 3.0
    run.go("vllm constant-row test (gate 3.0)", xk, M_GEN)

    # Scales and offsets.
    for sc in (0.02, 10.0, 1000.0):
        run.go(f"gate and up scaled by {sc}", rand_x(256, 4096, gen, gscale=sc, uscale=sc), M_GEMMA)
    for mu in (100.0, -100.0):
        run.go(f"gate mean {mu}", rand_x(256, 4096, gen, gmean=mu), M_GEMMA)
    run.go("gate mean 3, std 0.5", rand_x(256, 4096, gen, gscale=0.5, gmean=3.0), M_GEMMA)

    # Multipliers.
    for m in (0.0, 64.0, -64.0, 0.1, -M_GEMMA, 8.0, -38.5):
        run.go(f"multiplier {m}", rand_x(256, 2048, gen), m)

    # Row lengths.
    run.go("d = 1 (constant rows)", rand_x(64, 1, gen), M_GEMMA)
    for d in (2, 3, 7, 100, 257, 32767):
        run.go(f"d = {d}", rand_x(64, d, gen), M_GEMMA)
    run.go("d = 32768 (BLOCK_SIZE 32768)", rand_x(64, 32768, gen), M_GEMMA)
    run.go("d = 32768, mean 50", rand_x(64, 32768, gen, gmean=50.0), M_GEMMA)

    # Gates near the cutoff.
    for m in (M_GEMMA, 0.0, -M_GEMMA):
        run.go("10% of the gate near cRef", near_cutoff(256, 4096, m, gen), m)
    run.go("one gate element near cRef, d = 7", near_cutoff(512, 7, M_GEMMA, gen, frac=1 / 7), M_GEMMA)

    # Magnitudes at the edge of InOk: |gate|, |up| up to 2^32.
    sgn = lambda *s: torch.where(torch.rand(s, generator=gen, device=DEV) < 0.5, -1.0, 1.0)  # noqa: E731
    xb = (P(32) * sgn(128, 2 * 4096) * (0.5 + 0.5 * torch.rand(128, 2 * 4096, generator=gen, device=DEV)))
    xb = xb.clamp(-P(32), P(32)).to(torch.bfloat16)
    for m in (M_GEMMA, -64.0, 64.0):
        run.go("|gate|, |up| in [2^31, 2^32]", xb, m)
    xh = rand_x(128, 4096, gen, gscale=P(29), uscale=P(29)).double().clamp(-P(32), P(32)).to(torch.bfloat16)
    run.go("gate, up 2^29 N(0,1) clamped to 2^32, m = -64", xh, -64.0)

    # Rows that just meet RowOk: one value repeated but one neighbour.
    for d, v in ((32768, 1.0), (32768, -3.0), (4096, 1000.0)):
        x = torch.empty(16, 2 * d, dtype=torch.bfloat16, device=DEV)
        x[:, :d] = v
        nb = torch.nextafter(torch.tensor(v, dtype=torch.bfloat16),
                             torch.tensor(float("inf"), dtype=torch.bfloat16))
        x[:, d - 1] = nb.item()
        x[:, d:] = torch.randn(16, d, generator=gen, device=DEV).to(torch.bfloat16)
        for m in (M_GEMMA, 0.0, -M_GEMMA):
            run.go(f"row of {v} with one neighbour", x, m)
    # Constant rows and an all-zero row: outside Pre (RowOk).
    xc = rand_x(16, 1024, gen); xc[:, :1024] = 0.5
    run.go("constant gate rows", xc, M_GEMMA)
    xz = rand_x(16, 1024, gen); xz[:, :1024] = 0.0
    run.go("all-zero gate rows", xz, M_GEMMA)

    # Tiny rows (RowOk holds above 2^-62).
    run.go("gate 2^-40 N(0,1)", rand_x(128, 4096, gen, gscale=P(-40)), M_GEMMA)
    run.go("gate 2^-40 N(0,1), up 2^-40 N(0,1)", rand_x(128, 4096, gen, gscale=P(-40), uscale=P(-40)),
           M_GEMMA)
    run.go("gate 2^-58 N(0,1)", rand_x(128, 4096, gen, gscale=P(-58)), M_GEMMA)
    run.go("gate 1e-30 N(0,1) (below RowOk's floor)", rand_x(128, 4096, gen, gscale=1e-30), M_GEMMA)

    # Rows of several shapes stacked (ndim > 2 flattens to rows).
    for shp in ((4, 33), (2, 3)):  # (2, 3): vLLM's n-d test, [2, 3, 16384]
        dd = 2048 if shp == (4, 33) else 8192
        x3 = rand_x(shp[0] * shp[1], dd, gen).view(*shp, 2 * dd)
        out3 = mod.gelu_and_mul_sparse_triton(x3, M_GEN, "tanh")
        pre, r, zok, _ = check_post(x3.view(-1, 2 * dd), M_GEN, out3.view(-1, dd))
        good = pre and r < 1 and zok and out3.shape == (*shp, dd)
        name = f"3-D input [{shp[0]}, {shp[1]}, 2*{dd}]"
        print(f"{name:52s}: y {fmt(r)}, zero clause {zok}, shape {tuple(out3.shape)}"
              f"{'' if good else '  <-- FAIL'}")
        run.ok = run.ok and good
        run.worst = max(run.worst, r)

    print(f"worst error / tolerance inside Pre: {fmt(run.worst)}")
    print(f"launches outside Pre (not checked): {len(run.outside)}: {', '.join(run.outside)}")
    expected_outside = {"d = 1 (constant rows)", "vllm constant-row test (gate 3.0)", "constant gate rows", "all-zero gate rows",
                        "gate 1e-30 N(0,1) (below RowOk's floor)"}
    if set(run.outside) != expected_outside:
        print("unexpected set of launches outside Pre  <-- FAIL")
        run.ok = False

    print("negative tests (Gemma 3n E4B tp1 shape, 256 rows):")
    neg_ok = negatives(rand_x(256, 16384, gen), M_GEMMA)
    print("negative tests (10% of the gate near the cutoff, 256 rows of 4096):")
    neg_ok = negatives(near_cutoff(256, 4096, M_GEMMA, gen), M_GEMMA) and neg_ok
    run.ok = run.ok and neg_ok

    meaning(rand_x(512, 16384, gen), M_GEMMA, "Gemma 3n E4B (d 16384), N(0,1) gate")

    print(f"peak device memory: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    print("PASS" if run.ok else "FAIL")
    sys.exit(0 if run.ok else 1)


if __name__ == "__main__":
    main()
