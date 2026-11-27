"""GPU check of the claim `Claims.LogsigmoidFwd` (Claim.lean).

Runs the kernel's own host (source/logsigmoid.py, imported by path) on the launches SETUP.md asks
for and on launches at the edges of `Inst.ok` and `Pre`, and checks `Post` exactly as it is
stated, in float64:

* y[k] finite and |y[k] - ref| <= yTol, ref = lsRef(v) / t, lsRef(v) = -log(1 + e^-v) (float64's
  logsigmoid, relative error about 1e-16), yTol = (2^-8 + 2^-19)|ref| + 2^-23/|t| + 2^-133, with
  v the bfloat16 input and t the host's double temperature;
* the frame: x unchanged, and (for the direct launches, whose output sits inside a larger buffer
  filled with a sentinel) no byte outside the written elements changed.

Cells outside `Pre` (|v| + 1 > 2^125 |t|) are counted separately. Each line reports the largest
error / tolerance. Negative tests: wrong outputs (y = 0, no temperature, no log term, the log term's
sign flipped, the neighbouring element, one bfloat16 ulp off) must exceed the tolerance.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os
import struct

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "logsigmoid.py")
spec = importlib.util.spec_from_file_location("logsigmoid_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import triton  # noqa: E402

DEV = torch.device("cuda")
REL = 2.0 ** -8 + 2.0 ** -19
ABS_T = 2.0 ** -23
FLOOR = 2.0 ** -133

worst = {"y": 0.0}
failures = []
counts = {"cells": 0, "outside": 0, "outside_bad": 0}


def ref_tol(v, t):
    """ref and yTol at the float64 inputs v and the double t."""
    ref = torch.nn.functional.logsigmoid(v) / t
    tol = REL * ref.abs() + ABS_T / abs(t) + FLOOR
    return ref, tol


def check_post(name, xv, yv, t, quiet=False):
    """Post on matching flat views: xv the inputs (bfloat16, element k of y's input), yv the
    outputs (bfloat16)."""
    v = xv.double().reshape(-1)
    yc = yv.double().reshape(-1)
    cov = v.abs() + 1 <= 2.0 ** 125 * abs(t)
    ref, tol = ref_tol(v, t)
    fin = torch.isfinite(yc)
    err = torch.where(cov, (yc - ref).abs() / tol, torch.zeros_like(tol))
    e = err.max().item() if err.numel() else 0.0
    ok = bool((fin | ~cov).all()) and e <= 1
    counts["cells"] += int(cov.sum())
    counts["outside"] += int((~cov).sum())
    counts["outside_bad"] += int((~cov & ~(fin & ((yc - ref).abs() <= tol))).sum())
    worst["y"] = max(worst["y"], e)
    if not ok:
        failures.append(name)
    if not quiet or not ok:
        extra = f", outside Pre {int((~cov).sum())}" if (~cov).any() else ""
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {int(cov.sum())} cells, err/tol {e:.3e}{extra}")
    return ok, int(cov.sum())


def host_case(name, x, t=1.0, chunk=2 ** 26):
    """The host's own launch (B = 1024) on x (any shape); the comparison runs in chunks."""
    x0 = x.clone()
    y = mod.logsigmoid_fwd(x, t)
    torch.cuda.synchronize()
    assert torch.equal(x, x0), "x changed"
    assert y.is_contiguous() and y.shape == x.shape and y.dtype == torch.bfloat16
    xf, yf = x.reshape(-1), y.reshape(-1)
    ok, ncov = True, 0
    e0 = worst["y"]
    worst["y"] = 0.0
    for s in range(0, max(xf.numel(), 1), chunk):
        o, n = check_post(name, xf[s:s + chunk], yf[s:s + chunk], t, quiet=True)
        ok &= o
        ncov += n
    e = worst["y"]
    worst["y"] = max(e0, e)
    print(f"{'ok  ' if ok else 'FAIL'} {name} (t = {t}): {ncov} cells in Pre of {xf.numel()}, "
          f"err/tol {e:.3e}")
    del y, x0


SENT = -12345.0


def direct_case(name, xbuf, N, D, sx, B, t, T_arg=None, grid=None):
    """A direct launch of the kernel on rows of `xbuf` at row stride `sx`, block size `B`, the output
    inside a larger sentinel-filled buffer. T_arg overrides the runtime T; grid overrides the grid."""
    T = N * D
    pad = 4096
    ybig = torch.full((T + 2 * pad,), SENT, dtype=torch.bfloat16, device=DEV)
    y = ybig[pad:pad + T]
    x0 = xbuf.clone()
    g = grid if grid is not None else (triton.cdiv(T, B),)
    mod.logsigmoid_fwd_kernel[g](x=xbuf, y=y, stride_x_row=sx, stride_y_row=D, temperature=t,
                                 T=T if T_arg is None else T_arg, D=D, B=B, num_warps=4)
    torch.cuda.synchronize()
    frame = (torch.equal(xbuf, x0) and bool((ybig[:pad] == SENT).all())
             and bool((ybig[pad + T:] == SENT).all()))
    if not frame:
        failures.append(name + " (frame)")
    k = torch.arange(T, device=DEV)
    xv = xbuf[(k // D) * sx + k % D] if T else xbuf[:0]
    ok, _ = check_post(name, xv, y, t, quiet=True)
    print(f"{'ok  ' if ok and frame else 'FAIL'} {name}: {T} cells, frame {'holds' if frame else 'BROKEN'}")


def bf(t):
    return t.to(device=DEV, dtype=torch.bfloat16)


def all_bf16():
    """Every finite bfloat16 value."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    v = bits.view(torch.bfloat16)
    return v[torch.isfinite(v.float())].to(DEV)


def negatives(gen):
    """Wrong outputs that Post must reject: each exceeds the tolerance somewhere."""
    x = bf(torch.randn((8192,), generator=gen) * 4)
    x = torch.cat([x, all_bf16()])
    for t in (0.5, 16.0):
        y = mod.logsigmoid_fwd(x, t)
        torch.cuda.synchronize()
        v = x.double()
        ref, tol = ref_tol(v, t)
        cov = v.abs() + 1 <= 2.0 ** 125 * t
        y16 = y.contiguous()
        up = (y16.view(torch.int16) + 1).view(torch.bfloat16).double()  # one ulp away from 0 in magnitude
        wrong = {
            "y = 0": torch.zeros_like(ref),
            "no temperature": torch.nn.functional.logsigmoid(v),
            "no log term (min(0, x) / t)": torch.clamp(v, max=0) / t,
            "log term with the wrong sign": (torch.clamp(v, max=0) + torch.log1p(torch.exp(-v.abs()))) / t,
            "the neighbouring element": torch.roll(y.double(), 1),
            "one bfloat16 ulp off": torch.where(torch.isfinite(up), up, y.double()),
        }
        for k, got in wrong.items():
            e = ((got - ref).abs() / tol)[cov].max().item()
            bad = e > 1
            if not bad:
                failures.append(f"negative: {k} (t = {t})")
            print(f"{'ok  ' if bad else 'FAIL'} negative {k} (t = {t}): err/tol {e:.3e} -> "
                  f"{'rejected' if bad else 'ACCEPTED'}")


def main():
    gen = torch.Generator().manual_seed(0)
    for t, h in ((1.0, "3ff0000000000000"), (0.5, "3fe0000000000000"), (2.0, "4000000000000000"),
                 (16.0, "4030000000000000")):
        assert struct.pack(">d", t).hex() == h  # Claim.t1, t05, t2, t16

    print("== SETUP.md: fla's test cases (bfloat16 inputs, randn)")
    for shape, t, nc in [((1, 1, 64), 1.0, False), ((2, 500, 128), 0.5, False), ((2, 512, 128), 0.5, False),
                         ((3, 2048, 1200), 2.0, False), ((2, 500, 128), 0.5, True), ((2, 512, 128), 1.0, True),
                         ((3, 2048, 1200), 2.0, True), ((63,), 1.0, False), ((63,), 1.0, True)]:
        if nc and len(shape) == 1:
            x = bf(torch.randn(shape[0] * 3, generator=gen))[::3]  # prepare_input copies it
        elif nc:
            x = bf(torch.randn(*shape[:-1], 2 * shape[-1], generator=gen)).chunk(2, dim=-1)[0]
        else:
            x = bf(torch.randn(shape, generator=gen))
        host_case(f"host {shape}{' noncontiguous' if nc else ''}", x, t)
    print("== SETUP.md: GLA's gate logits (t = 16), its output gate, a long sequence")
    host_case("host gla gk (4, 2048, 1024)", bf(torch.randn((4, 2048, 1024), generator=gen) * 3), 16.0)
    host_case("host gla gate (4, 2048, 2048)", bf(torch.randn((4, 2048, 2048), generator=gen) * 3), 1.0)
    host_case("host long (1, 65536, 4096)", bf(torch.randn((1, 65536, 4096), generator=gen) * 3), 1.0)

    print("== edges of Pre: every finite bfloat16 input, temperatures from 2^-126 to 2^127")
    xa = all_bf16()
    for t in [1.0, 0.5, 2.0, 16.0, -1.0, -16.0, 0.3, 1 / 3, 1e-3, 1e3, 2.0 ** -126, 1.5 * 2.0 ** -126,
              2.0 ** -100, 1e-30, 1e30, 2.0 ** 100, 3.0 ** 70, 2.0 ** 127, -(2.0 ** 127)]:
        host_case(f"host all bfloat16", xa, t)
    print("== edges of Inst.ok: shapes, strides (direct launches, B from 1 to 2^16), frame checked")
    host_case("host shape (2, 0)", bf(torch.randn((2, 0), generator=gen)))
    host_case("host shape (0,)", bf(torch.randn((0,), generator=gen)))
    host_case("host shape (5, 1)", bf(torch.randn((5, 1), generator=gen)))
    host_case("host expand (rows read twice, stride 0)", bf(torch.randn((1, 300), generator=gen)).expand(7, 300))
    host_case("host (3, 4, 5)[:, ::2] (one row stride)", bf(torch.randn((3, 4, 5), generator=gen))[:, ::2])
    pool = bf(torch.randn((1 << 20,), generator=gen) * 8)
    for B in [1, 16, 512, 1024, 2048, 4096, 8192, 65536]:
        for N, D, sx in [(3, 1, 1), (3, 1, 7), (17, 33, 33), (17, 33, 66), (17, 33, 40), (9, 100, 0),
                         (5, 1200, 2400), (1, 5000, 5000), (300, 128, 128)]:
            direct_case(f"direct B {B} N {N} D {D} stride {sx}", pool, N, D, sx, B, 1.0 if B != 512 else 16.0)
    direct_case("direct T = 0", pool, 0, 64, 64, 1024, 1.0)
    print("== integer classes: stride_x_row >= 2^31 (int64) with one row; T = 2^31 + 7 (int64)")
    direct_case("direct stride 2^31 + 5, N 1", pool, 1, 3000, 2 ** 31 + 5, 1024, 1.0)
    direct_case("direct T = 2^31 + 7, grid (2,), D 64", pool, 32, 64, 64, 1024, 1.0, T_arg=2 ** 31 + 7,
                grid=(2,))

    print(f"\ncells checked {counts['cells']}, worst error / tolerance {worst['y']:.3e} "
          f"(margin {1 / worst['y']:.3f}x); cells outside Pre {counts['outside']} (of them outside Post "
          f"{counts['outside_bad']})")

    print("\n== negative tests")
    negatives(gen)

    print("\n== tol / |ref| at real inputs (randn * 3, t = 1): median and the share of cells where "
          "2^-23/|t| exceeds the relative term")
    v = bf(torch.randn((1 << 20,), generator=gen) * 3).double()
    ref, tol = ref_tol(v, 1.0)
    r = (tol / ref.abs())
    print(f"median {r.median().item():.4e}, share where the absolute term dominates "
          f"{(ABS_T > REL * ref.abs()).double().mean().item():.4f}")

    print("\n== outside the claim (README, Exclusions)")
    x = bf(torch.tensor([-(2.0 ** 127), -3e38, 1.0]))
    y = mod.logsigmoid_fwd(x, 2.0 ** -2)
    torch.cuda.synchronize()
    print(f"|x| + 1 > 2^125 |t| (x = -2^127, -3e38 at t = 0.25): y {y.tolist()} "
          f"(reference {(torch.nn.functional.logsigmoid(x.double()) / 0.25).tolist()})")
    x = bf(torch.tensor([-1.0, 1.0]))
    y = mod.logsigmoid_fwd(x, 1e-40)
    torch.cuda.synchronize()
    print(f"t = 1e-40 (a float32 subnormal): y {y.tolist()} "
          f"(reference {(torch.nn.functional.logsigmoid(x.double()) / 1e-40).tolist()})")
    x = bf(torch.tensor([float('-inf'), float('inf'), float('nan')]))
    y = mod.logsigmoid_fwd(x, 1.0)
    torch.cuda.synchronize()
    print(f"non-finite inputs (-inf, inf, nan): y {y.tolist()}")

    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("all checks hold")


if __name__ == "__main__":
    main()
