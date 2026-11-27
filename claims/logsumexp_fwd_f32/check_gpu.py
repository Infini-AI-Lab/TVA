"""GPU check of the claim `Claims.LogsumexpFwd` (Claim.lean).

Runs the kernel's own host (`logsumexp_fwd` of source/ops.py, imported by path) on the launches
SETUP.md asks for, and direct launches of `logsumexp_fwd_kernel` at the edges of `Inst.ok`, and
checks `Post` as it is stated, block by block, in float64:

* every `z[n, i]` is written (the host's `z` is read back from the launch's own buffer, a spy on
  the kernel's launch keeps it; a direct launch writes into a buffer filled with a sentinel);
* if the block is `BlockOk` (every input finite, `|y_j| <= 2^126` without a scale, `<= 2^20` with
  one, `y_j = x[n, j] * scale` with the host's double `scale`), `z[n, i]` is finite and
  `|z[n, i] - lseRef| <= 2^-20 * M + 2^-16`, `lseRef = log sum_j e^(y_j)` over the block's columns
  (computed as `m + log sum_j e^(y_j - m)` in float64, `m` the block's largest `y_j`) and `M` the
  block's largest `|y_j|` (the smallest bound `Post` allows). Each line reports the largest error
  / tolerance;
* the frame: `x` is unchanged (bitwise), and a direct launch leaves every element of its sentinel
  buffer outside `z`'s `N * ND` elements unchanged.

Negative tests: plausible wrong outputs (the log-sum-exp without `+ m`, with the block's largest
column left out, over the whole row instead of the block, in base 2, without the scale, rounded to
bfloat16 or float16, the next block's value) must exceed the tolerance. A last section runs blocks
that are not `BlockOk` (outside the accuracy claim) and prints what the kernel stores.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "ops.py")
spec = importlib.util.spec_from_file_location("lse_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import triton  # noqa: E402

DEV = torch.device("cuda")
TOL_M = 2.0 ** -20   # Claim.lean, `zTolM`
TOL_0 = 2.0 ** -16   # `zTol0`
SENT = -777.25

KERNEL = mod.logsumexp_fwd_kernel


class Spy:
    """Stands for `logsumexp_fwd_kernel` in the host's module: records the last launch's grid and
    arguments (so the host's own `z` can be checked), then launches the kernel unchanged."""

    def __init__(self, k):
        self.k = k
        self.last = None

    def __getitem__(self, grid):
        def run(**kw):
            self.last = (grid, kw)
            return self.k[grid](**kw)
        return run


SPY = Spy(KERNEL)
mod.logsumexp_fwd_kernel = SPY

worst = {"host": 0.0, "direct": 0.0}
failures = []


def host_bd(D):
    return min(triton.next_power_of_2(D), 64 * 1024)


def block_stats(x2d, scale, BD, ND, rows=None):
    """Per block (float64): (lseRef, M, BlockOk, base-2 lse, lse without the max term, lse without
    the block's largest column, the row's lse) for the rows `rows` of x2d ([N, D] float32)."""
    xr = x2d if rows is None else x2d[rows]
    N, D = xr.shape
    W = ND * BD
    xd = torch.zeros((N, W), dtype=torch.float64, device=DEV)
    xd[:, :D] = xr.double()
    valid = torch.zeros((N, W), dtype=torch.bool, device=DEV)
    valid[:, :D] = True
    sig = 1.0 if scale is None else scale
    y = xd * sig
    y = y.view(N, ND, BD)
    valid = valid.view(N, ND, BD)
    finite = torch.isfinite(y) | ~valid
    ymax = 2.0 ** 126 if scale is None else 2.0 ** 20
    absy = torch.where(valid, y.abs(), torch.zeros_like(y))
    xfin = torch.isfinite(xd.view(N, ND, BD)) | ~valid
    ok = (xfin & finite).all(-1) & (torch.where(valid & finite, absy, torch.zeros_like(y)) <= ymax).all(-1)
    neg = torch.full_like(y, -math.inf)
    ys = torch.where(valid & finite, y, neg)
    m = ys.max(-1).values
    msafe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(ys - msafe[..., None])
    S = e.sum(-1)
    ref = msafe + torch.log(S)
    M = torch.where(valid & finite, absy, torch.zeros_like(y)).max(-1).values
    # wrong outputs
    lse2 = msafe + torch.log2((torch.exp2((ys - msafe[..., None]) * 1.0)).sum(-1))
    nomax = torch.log(S)
    arg = ys.argmax(-1, keepdim=True)
    e_drop = e.scatter(-1, arg, 0.0)
    drop = msafe + torch.log(e_drop.sum(-1))
    rowm = msafe.max(-1, keepdim=True).values
    rowlse = rowm + torch.log((S * torch.exp(msafe - rowm)).sum(-1, keepdim=True))
    rowlse = rowlse.expand_as(ref)
    return ref, M, ok, lse2, nomax, drop, rowlse


def err_tol(c, ref, M, ok):
    """(largest error / tolerance over the BlockOk blocks, every BlockOk output finite)."""
    c = c.double()
    fin = bool(torch.isfinite(c[ok]).all().item())
    tol = TOL_M * M + TOL_0
    r = ((c - ref).abs() / tol)[ok]
    w = r.max().item() if r.numel() else 0.0
    return w, fin


def check_blocks(name, kind, x2d, z2d, scale, BD, chunk_elems=2 ** 24):
    """Post on every block of x2d against z2d ([N, ND]), in chunks of rows."""
    N, D = x2d.shape
    ND = triton.cdiv(D, BD)
    rows = max(1, chunk_elems // (ND * BD))
    w_all, fin_all, nok = 0.0, True, 0
    for r0 in range(0, N, rows):
        sl = slice(r0, min(N, r0 + rows))
        ref, M, ok, *_ = block_stats(x2d, scale, BD, ND, sl)
        w, fin = err_tol(z2d[sl], ref, M, ok)
        w_all, fin_all = max(w_all, w), fin_all and fin
        nok += int((~ok).sum().item())
    worst[kind] = max(worst[kind], w_all)
    return w_all, fin_all, nok


def host_case(name, x, scale=None):
    """The host's own launch on x ([..., D] float32): Post on the kernel's z (kept by the spy)."""
    before = x.clone()
    SPY.last = None
    out = mod.logsumexp_fwd(x, scale=scale)
    torch.cuda.synchronize()
    D = x.shape[-1]
    x2d = x.reshape(-1, D)
    N = x2d.shape[0]
    BD = host_bd(D)
    ND = triton.cdiv(D, BD)
    grid, kw = SPY.last
    z = kw["z"]
    good = tuple(grid) == (N, ND) and kw["BD"] == BD and z.shape == (N, ND)
    w, fin, nok = check_blocks(name, "host", x2d, z, scale, BD) if N else (0.0, True, 0)
    same = torch.equal(x.view(torch.int32), before.view(torch.int32))
    good = good and fin and w <= 1 and same and out.shape == x.shape[:-1]
    if not good:
        failures.append(name)
    print(f"{'ok  ' if good else 'FAIL'} host {name}: N {N} D {D} BD {BD} ND {ND} scale {scale}: "
          f"err/tol {w:.3e}, finite {fin}, blocks not BlockOk {nok}, x unchanged {same}", flush=True)
    return z


def direct_case(name, x2d, scale, BD, guard=1024):
    """A direct launch with block size BD, z inside a sentinel buffer."""
    before = x2d.clone()
    N, D = x2d.shape
    ND = triton.cdiv(D, BD)
    buf = torch.full((guard + N * ND + guard,), SENT, device=DEV)
    z = buf[guard:guard + N * ND].view(N, ND)
    KERNEL[(N, ND)](x=x2d, z=z, scale=scale, softcapping=None, D=D, BD=BD,
                    HAS_SCALE=scale is not None, HAS_SOFTCAPPING=False)
    torch.cuda.synchronize()
    frame = bool((buf[:guard] == SENT).all().item() and (buf[guard + N * ND:] == SENT).all().item())
    w, fin, nok = check_blocks(name, "direct", x2d, z, scale, BD) if N else (0.0, True, 0)
    same = torch.equal(x2d.view(torch.int32), before.view(torch.int32))
    good = fin and w <= 1 and same and frame
    if not good:
        failures.append(name)
    print(f"{'ok  ' if good else 'FAIL'} direct {name}: N {N} D {D} BD {BD} ND {ND} scale {scale}: "
          f"err/tol {w:.3e}, finite {fin}, blocks not BlockOk {nok}, frame {frame}, x unchanged {same}",
          flush=True)
    return z


def negatives(name, x2d, z, scale, BD):
    """Plausible wrong outputs, each must exceed the tolerance on some BlockOk block."""
    N, D = x2d.shape
    ND = triton.cdiv(D, BD)
    ref, M, ok, lse2, nomax, drop, rowlse = block_stats(x2d, scale, BD, ND)
    cand = {
        "without + m": nomax,
        "largest column left out": drop,
        "base 2 (exp2/log2)": lse2,
        "bfloat16-rounded": z.to(torch.bfloat16).double(),
        "float16-rounded": z.to(torch.float16).double(),
        "next block (cyclic)": z.double().reshape(-1).roll(-1).view(N, ND),
    }
    if ND > 1:
        cand["whole row, not the block"] = rowlse
    if scale is not None and scale != 1.0:
        r2, *_ = block_stats(x2d, None, BD, ND)
        cand["scale not applied"] = r2
    for k, v in cand.items():
        e, _ = err_tol(v, ref, M, ok)
        bad = e > 1
        if not bad:
            failures.append(f"negative {name}: {k}")
        print(f"{'ok  ' if bad else 'FAIL'} negative {name}: {k}: err/tol {e:.3e} -> "
              f"{'rejected' if bad else 'ACCEPTED'}", flush=True)


def main():
    gen = torch.Generator().manual_seed(0)

    def rnd(*shape, s=4.0):
        return (torch.randn(shape, generator=gen) * s).to(device=DEV, dtype=torch.float32)

    print("== the host's launches (SETUP.md)")
    # fla's test shape (tests/modules/test_cross_entropy.py:415-421) without softcapping, the
    # host test's shapes, and chunks of logits over common vocabulary sizes.
    for N, D, scale in [
        (1, 1, None), (2, 1025, None), (2, 1025, 1.0), (2, 1025, 0.0),
        (63, 50257, None), (64, 65536, 0.5), (32, 128256, None), (16, 151936, 1.0),
        (8, 262144, 30.0),
        (2048, 32000, 1.0), (1024, 128256, None), (512, 151936, 1.0), (256, 262144, None),
    ]:
        host_case(f"{N}x{D}", rnd(N, D), scale)
    # fla's test input itself (zeros), and a 3-D input (the host flattens the leading dimensions)
    host_case("fla test zeros 2x1025", torch.zeros((2, 1025), device=DEV), 1.0)
    host_case("3-D 4x33x777", rnd(4, 33, 777), None)
    host_case("N = 0", torch.zeros((0, 300), device=DEV), None)

    print("\n== direct launches at the edges of Inst.ok")
    for D, BD in [(1, 1), (1, 2 ** 16), (2, 1), (3, 2), (1000, 16), (65536, 2 ** 16), (65537, 2 ** 16),
                  (70000, 1024), (2 ** 20, 2 ** 16), (5000, 2 ** 16)]:
        for scale in (None, 0.5):
            direct_case(f"D {D} BD {BD}", rnd(5, D), scale, BD)
    direct_case("many rows, ND = 1", rnd(2 ** 20, 1), None, 1)
    direct_case("many rows, ND = 3", rnd(2 ** 16, 40), 1.0, 16)

    print("\n== values at the edges of BlockOk")
    D = 4096
    big = 2.0 ** 126
    x = torch.full((3, D), big, device=DEV)
    x[1] = -big
    x[2, ::2] = -big
    direct_case("|x| = 2^126 (no scale)", x, None, 1024)
    x = rnd(4, D) * 2.0 ** 100
    x = x.clamp(-big, big)
    direct_case("x ~ 2^102 (no scale)", x, None, 4096)
    x = (torch.rand((4, D), generator=gen) * 2 - 1).to(DEV) * (2.0 ** 20)
    direct_case("|x * scale| ~ 2^20 (scale 1)", x, 1.0, 4096)
    direct_case("|x * scale| ~ 2^20 (scale 2^-10)", x * 1024, 2.0 ** -10, 1024)
    direct_case("|x * scale| ~ 2^20 (scale -1)", x, -1.0, 4096)
    direct_case("scale 2^-126, x ~ 2^120", rnd(2, D) * 2.0 ** 120, 2.0 ** -126, 4096)
    direct_case("scale 2^126, tiny x", rnd(2, D) * 2.0 ** -140, 2.0 ** 126, 4096)
    direct_case("scale 2^-140 (float32 subnormal), x ~ 2^120", rnd(2, D) * 2.0 ** 120, 2.0 ** -140, 4096)
    direct_case("scale 1e-300 (float32 0)", rnd(2, D) * 2.0 ** 120, 1e-300, 4096)
    direct_case("scale -2^126, tiny x", rnd(2, D) * 2.0 ** -140, -(2.0 ** 126), 4096)
    direct_case("scale 30, logits", rnd(8, 32000, s=1.0), 30.0, 32768)
    direct_case("subnormal inputs", rnd(3, D) * 2.0 ** -140, None, 4096)
    direct_case("subnormal inputs, scale 0.1", rnd(3, D) * 2.0 ** -140, 0.1, 4096)
    direct_case("all equal (zeros)", torch.zeros((2, 2 ** 16), device=DEV), None, 2 ** 16)
    direct_case("all equal (zeros), scale 0", rnd(2, 2 ** 16), 0.0, 2 ** 16)
    direct_case("all equal (-3)", torch.full((2, 2 ** 16), -3.0, device=DEV), 1.0, 2 ** 16)
    x = rnd(4, D)
    x[:, 7] = 60.0
    direct_case("one dominant column", x, None, 4096)
    x = torch.linspace(-30, 0, D, device=DEV).repeat(4, 1)
    direct_case("gaps up to 30", x, 1.0, 4096)
    x = torch.linspace(-2000, 0, D, device=DEV).repeat(4, 1)
    direct_case("gaps up to 2000", x, None, 4096)

    print("\n== negative tests")
    x = rnd(64, 70000)
    z = direct_case("negatives base (scale 0.5)", x, 0.5, 2 ** 16)
    negatives("64x70000 scale 0.5", x, z, 0.5, 2 ** 16)
    x = rnd(32, 50257, s=2.0)
    z = direct_case("negatives base (no scale)", x, None, 2 ** 16)
    negatives("32x50257 no scale", x, z, None, 2 ** 16)
    x = rnd(32, 1025)
    z = direct_case("negatives base (BD 256)", x, 1.0, 256)
    negatives("32x1025 BD 256 scale 1", x, z, 1.0, 256)

    print("\n== outside BlockOk (no accuracy claimed): what the kernel stores")
    D = 300
    x = rnd(6, D)
    x[0, 5] = float("nan")
    x[1, 5] = float("inf")
    x[2, 5] = -float("inf")
    x[3, :] = -float("inf")
    x[4, :] = 2.0 ** 127
    z = direct_case("not BlockOk (NaN, +inf, -inf, all -inf, 2^127)", x, None, 512)
    print("   z per row:", [v for v in z.view(-1).tolist()])
    x = rnd(2, D) * 2.0 ** 24
    z = direct_case("|x * scale| ~ 2^26 with a scale (beyond 2^20)", x, 1.0, 512)
    print("   z per row:", [v for v in z.view(-1).tolist()])

    print(f"\nworst err/tol: host {worst['host']:.3e}, direct {worst['direct']:.3e}")
    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("all checks hold")


if __name__ == "__main__":
    main()
