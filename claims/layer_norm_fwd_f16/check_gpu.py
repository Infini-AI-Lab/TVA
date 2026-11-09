"""GPU check of the prepared claim `Claims.LayerNormFwdF16` (claims/README.md, check 4).

Runs the kernel's own host, `layer_norm` (`LayerNorm.apply`, whose forward launches
`_layer_norm_fwd_fused`) of `source/05-layer-norm.py`, imported by path (the module's top level also
runs the tutorial's test and its benchmark: the script loads it with `benchmark.run` disabled and
makes the benchmark's forward launches itself), on the launches of `SETUP.md` and on launches at the
edges of `Inst.ok` and `Pre`. For every row it computes `Post`'s reference and tolerances from the
host's inputs in float64 exactly as `Claim.lean` states them (`meanRef`, `varRef`, `rstdRef`,
`zRef`, `yRef`, `meanTol`, `cond`, `gam`, `rstdTol`, `zTol`, `yTol`), checks `YRoom` (`Pre`), and
reports, for `mean`, `rstd` and `y`, the largest error / tolerance and that every element is
finite. The frame: the host's `torch.empty_like` and `torch.empty` are made to return views into
larger buffers filled with a sentinel (NaN), whose guard bands must be unchanged and whose output
elements must all be written; the inputs must be unchanged.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch
import triton
import triton.testing

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "05-layer-norm.py")
_orig_run = triton.testing.Mark.run
triton.testing.Mark.run = lambda self, *a, **k: None
spec = importlib.util.spec_from_file_location("layer_norm_src", SRC)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)   # also runs the module's test_layer_norm(1151, 8192, float16)
except AssertionError as e:
    # The tutorial's own test (line 374) also checks the backward kernels' gradients (lines
    # 320-323, not this claim's kernels), whose float16 `db` check fails on some runs. The forward's
    # definitions (lines 48-297) are bound before line 374, so the check goes on without it.
    print(f"(the tutorial's test at import failed an assert of the backward pass: {e!r}; "
          "the forward is checked below)", flush=True)
finally:
    triton.testing.Mark.run = _orig_run

DEV = "cuda"
GUARD = 4096


def next_pow2(n):
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------- the tolerances (Claim.lean)
def tolerances(x, w, b, eps, BLOCK):
    """Per row (x: [rows, N] float64): references and tolerances, as Claim.lean states them."""
    N = x.shape[1]
    D = float(BLOCK.bit_length() - 1)   # depth = log2 BLOCK_SIZE (Claim.lean, `depth`)
    meanRef = x.sum(1) / N
    absMean = x.abs().sum(1) / N
    varRef = ((x - meanRef[:, None]) ** 2).sum(1) / N
    rstdRef = 1.0 / torch.sqrt(varRef + eps)
    zRef = (x - meanRef[:, None]) * rstdRef[:, None]
    yRef = zRef * w[None, :] + b[None, :]
    meanTol = (D + 4) * 2.0 ** -23 * absMean + 2.0 ** -140
    cond = meanTol * rstdRef
    gam = (D + 8) * (2.0 ** -23 + 2.0 ** -145 / eps)
    mTerm = cond / 8 + cond ** 2 / 4
    reg = cond <= 0.5          # the two cases of rstdTol and zTol
    rstdTol = torch.where(reg, rstdRef * (gam + mTerm), (1 + gam) / math.sqrt(eps)) + 2.0 ** -140
    zTol = torch.where(
        reg[:, None],
        (1 + 2 * gam) * (zRef.abs() * (gam + mTerm[:, None]) + cond[:, None]) + 2.0 ** -140 * (1 + rstdRef[:, None]),
        zRef.abs() + (1 + 2 * gam) * ((x - meanRef[:, None]).abs() + meanTol[:, None]) / math.sqrt(eps) + 2.0 ** -140 * (1 + 1 / math.sqrt(eps)))
    yTol = 2.0 ** -10 * yRef.abs() + (1 + 2.0 ** -10) * w.abs()[None, :] * zTol + \
        2.0 ** -21 * (w.abs()[None, :] * zRef.abs() + b.abs()[None, :]) + 2.0 ** -24
    room = (w.abs()[None, :] * (zRef.abs() + zTol) + b.abs()[None, :] <= 2.0 ** 15).all(1)
    return meanRef, rstdRef, yRef, meanTol, rstdTol, yTol, room


# ---------------------------------------------------------------- running the host
GUARDED = []


def run_host(x, w, b, eps):
    """`layer_norm(x, w_shape, w, b, eps)` with `y`, `mean`, `rstd` allocated inside sentinel buffers."""
    real_empty_like, real_empty = torch.empty_like, torch.empty
    GUARDED.clear()

    def guarded(numel, dtype, shape, device):
        big = real_empty(numel + 2 * GUARD, dtype=dtype, device=device).fill_(float("nan"))
        GUARDED.append(big)
        return big[GUARD:GUARD + numel].view(shape)

    def empty_like(t, *a, **kw):
        return guarded(t.numel(), kw.get("dtype", t.dtype), t.shape, t.device)

    def empty(shape, *a, dtype=None, device=None, **kw):
        shape = tuple(shape) if isinstance(shape, (tuple, list, torch.Size)) else (shape,)
        return guarded(math.prod(shape), dtype or torch.float32, shape, device)

    torch.empty_like, torch.empty = empty_like, empty
    try:
        y = mod.layer_norm(x, (x.shape[-1],), w, b, eps)
    finally:
        torch.empty_like, torch.empty = real_empty_like, real_empty
    torch.cuda.synchronize()
    return y


WORST = {"mean": 0.0, "rstd": 0.0, "y": 0.0}
ALL_OK = True


def check_launch(name, x, w, b, eps=1e-5, chunk=None):
    global ALL_OK
    M, N = x.shape
    BLOCK = next_pow2(N)
    x0, w0, b0 = x.clone(), w.clone(), b.clone()
    y = run_host(x, w, b, eps)
    ybig, mbig, rbig = GUARDED[0], GUARDED[1], GUARDED[2]
    mean = mbig[GUARD:GUARD + M]
    rstd = rbig[GUARD:GUARD + M]
    guards = all(bool(torch.isnan(g[:GUARD].float()).all() and torch.isnan(g[-GUARD:].float()).all())
                 for g in (ybig, mbig, rbig))
    inputs = bool(torch.equal(x, x0) and torch.equal(w, w0) and torch.equal(b, b0))
    w64, b64 = w.double(), b.double()
    rM = rR = rY = 0.0
    finite = True
    noroom = 0
    chunk = chunk or max(1, 2 ** 24 // N)
    for r0 in range(0, M, chunk):
        r1 = min(M, r0 + chunk)
        x64 = x[r0:r1].double()
        mR, sR, yR, mT, sT, yT, room = tolerances(x64, w64, b64, eps, BLOCK)
        noroom += int((~room).sum())
        mc, sc, yc = mean[r0:r1].double(), rstd[r0:r1].double(), y[r0:r1].double()
        finite = finite and bool(torch.isfinite(mc).all() and torch.isfinite(sc).all())
        rM = max(rM, float(((mc - mR).abs() / mT).max()))
        rR = max(rR, float(((sc - sR).abs() / sT).max()))
        if room.any():
            yc, yR, yT = yc[room], yR[room], yT[room]
            finite = finite and bool(torch.isfinite(yc).all())
            rY = max(rY, float(((yc - yR).abs() / yT).max()))
    ok = guards and inputs and finite and rM < 1 and rR < 1 and rY < 1
    ALL_OK = ALL_OK and ok
    for k, v in (("mean", rM), ("rstd", rR), ("y", rY)):
        WORST[k] = max(WORST[k], v)
    print(f"{name:34s} M={M} N={N} BLOCK={BLOCK} eps={eps:.3g}: mean {rM:.3f} rstd {rR:.3f} y {rY:.3f} "
          f"finite={finite} guards+inputs={guards and inputs} rows-without-room={noroom} "
          f"{'ok' if ok else 'FAIL'}", flush=True)
    del y, ybig, mbig, rbig
    torch.cuda.empty_cache()


def host_inputs(M, N, seed=0):
    """As `test_layer_norm` and `bench_layer_norm` build them (lines 302-306, 340-344)."""
    torch.manual_seed(seed)
    w = torch.rand((N,), dtype=torch.float16, device=DEV)
    b = torch.rand((N,), dtype=torch.float16, device=DEV)
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=torch.float16, device=DEV)
    return x, w, b


def outside(name, x, w, b, eps):
    """A launch outside `Inst.ok`: report what the kernel writes."""
    class Ctx:
        def save_for_backward(self, *ts):
            self.saved = ts
    ctx = Ctx()
    y = mod.LayerNorm.forward(ctx, x, (x.shape[-1],), w, b, eps)
    torch.cuda.synchronize()
    rstd = ctx.saved[4]
    print(f"{name:34s} (outside Inst.ok) eps={eps:.3g}: rstd[0]={float(rstd[0]):.4g}, "
          f"y finite={bool(torch.isfinite(y).all())}, y[0,:3]={y[0, :3].tolist()}", flush=True)


NEG_OK = True


def negatives(name, M, N, seed=0):
    """Wrong outputs `Post` must reject, on the host's inputs (float64 values of three wrong kernels:
    a mean that drops one term, a variance that drops one term, the unbiased variance with `N - 1`;
    an all-zero `y`; `y` with the weights shifted by one column). Each must fail `Post`'s clause on
    some row; also prints tol / |output| (median, worst) on these rows."""
    global NEG_OK
    x, w, b = host_inputs(M, N, seed)
    BLOCK = next_pow2(N)
    x64, w64, b64 = x.double(), w.double(), b.double()
    mR, sR, yR, mT, sT, yT, room = tolerances(x64, w64, b64, 1e-5, BLOCK)
    dev2 = (x64 - mR[:, None]) ** 2
    wrong = {
        "mean drops a term": ((x64[:, :-1].sum(1) / N - mR).abs() / mT),
        "variance drops a term": ((1 / torch.sqrt(dev2[:, :-1].sum(1) / N + 1e-5) - sR).abs() / sT),
        "unbiased variance (N - 1)": ((1 / torch.sqrt(dev2.sum(1) / (N - 1) + 1e-5) - sR).abs() / sT),
        "y all zero": (yR.abs() / yT).max(1).values,
        "y, weights shifted": (((((x64 - mR[:, None]) * sR[:, None]) * w64.roll(1) + b64).half().double()
                                - yR).abs() / yT).max(1).values,
    }
    for k, r in wrong.items():
        rej = bool((r > 1).any())
        NEG_OK = NEG_OK and rej
        print(f"negative {name} M={M} N={N}: {k:28s} worst err/tol {float(r.max()):9.2f}, rows rejected "
              f"{float((r > 1).double().mean()):.3f}: {'rejected' if rej else 'ACCEPTED'}", flush=True)
    q = lambda t: f"median {float(t.median()):.2e} worst {float(t.max()):.2e}"
    print(f"tol/|output| {name} M={M} N={N}: mean {q(mT / mR.abs())}; rstd {q(sT / sR)}; "
          f"y {q(yT / yR.abs().clamp_min(2.0 ** -24))}", flush=True)


def main():
    # --- the test (line 374)
    check_launch("test", *host_inputs(1151, 8192))
    # --- the benchmark's forward launches (lines 326-371): M = 4096, N = 512 i, 2 <= i <= 31
    for i in range(2, 32):
        check_launch("bench", *host_inputs(4096, 512 * i, seed=i))
    # --- edges of Inst.ok
    for N in (2, 3, 1000, 32767, 32768):
        check_launch("edge N", *host_inputs(33, N, seed=N))
    check_launch("edge M=1", *host_inputs(1, 4096))
    check_launch("edge M=2^16, N=4096", *host_inputs(2 ** 16, 4096))
    x, w, b = host_inputs(64, 1024)
    check_launch("edge eps=2^-125", x, w, b, eps=2.0 ** -125)
    check_launch("edge eps=1", x, w, b, eps=1.0)
    check_launch("edge eps=2^126", x, w, b, eps=2.0 ** 126)
    # --- edges of Pre (values)
    c = torch.full((16, 2048), 1000.0, dtype=torch.float16, device=DEV)
    check_launch("edge constant rows (1000)", c, w[:1].repeat(2048), b[:1].repeat(2048))
    check_launch("edge constant rows eps=2^-125", c, w[:1].repeat(2048), b[:1].repeat(2048), eps=2.0 ** -125)
    nc = c.clone()
    nc[:, ::2] = 1000.5
    check_launch("edge near-constant rows", nc, w[:1].repeat(2048), b[:1].repeat(2048))
    big = (torch.randn((16, 2048), device=DEV) * 20000).clamp(-65504, 65504).half()
    check_launch("edge |x| up to 65504", big, w[:1].repeat(2048), b[:1].repeat(2048))
    tiny = (torch.randn((16, 2048), device=DEV) * 2.0 ** -20).half()
    check_launch("edge subnormal x", tiny, w[:1].repeat(2048), b[:1].repeat(2048))
    z = torch.zeros(2048, dtype=torch.float16, device=DEV)
    check_launch("edge w = b = 0", host_inputs(16, 2048)[0], z, z)
    spike = torch.zeros((8, 1024), dtype=torch.float16, device=DEV)
    spike[:, 0] = 1.0
    check_launch("edge spike rows, |w| = 960", spike, torch.full((1024,), 960.0, dtype=torch.float16, device=DEV),
                 torch.full((1024,), 1000.0, dtype=torch.float16, device=DEV))
    xs, ws, bs = host_inputs(16, 4096)
    check_launch("edge |w| = |b| = 2^10", xs, (ws * 1024).half(), (bs * -1024).half())
    # --- outside Inst.ok: the kernel limits of README, Exclusions
    outside("eps = 1e-39, constant rows", c[:2], w[:1].repeat(2048), b[:1].repeat(2048), 1e-39)
    outside("eps = 1.5 * 2^-126, constant rows", c[:2], w[:1].repeat(2048), b[:1].repeat(2048), 1.5 * 2.0 ** -126)
    outside("eps = 3.4e38, x = +-2^15", torch.tensor([[32768.0, -32768.0]] * 2, dtype=torch.float16, device=DEV),
            torch.ones(2, dtype=torch.float16, device=DEV), torch.zeros(2, dtype=torch.float16, device=DEV),
            3.4028234663852886e38)
    # --- wrong outputs Post must reject: the test's shape and the benchmark's largest
    negatives("test", 1151, 8192)
    negatives("bench", 512, 15872, seed=31)
    print(f"\nworst error / tolerance: mean {WORST['mean']:.4f}, rstd {WORST['rstd']:.4f}, y {WORST['y']:.4f}; "
          f"{'ALL OK' if ALL_OK else 'SOME FAILED'}; negatives {'all rejected' if NEG_OK else 'SOME ACCEPTED'}")
    return 0 if ALL_OK and NEG_OK else 1


if __name__ == "__main__":
    sys.exit(main())
