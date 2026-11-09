"""GPU check of the prepared claim `Claims.VecAdd` (claims/README.md, check 4).

Runs the kernel's own host, `add` of `source/01-vector-add.py` (imported by
path; the module's top level also runs the tutorial's own test and benchmark, so this script
loads it with `benchmark.run` disabled), on the launches of `SETUP.md` and on launches at the
edges of `Inst.ok`, and checks `Post` as it is stated:

* for every `j < n`, the output is the IEEE `float32` sum `x[j] + y[j]` rounded to nearest even, up
  to NaN payload. The reference is computed from the host's inputs in `float64` (the exact sum of
  two `float32` values, then one rounding to `float32`, which is the correctly rounded `float32`
  sum since `53 ≥ 2·24 + 2`), compared bit for bit (both NaN counts as equal);
* the frame: the inputs are unchanged (bitwise), and a direct launch of the module's kernel with the
  host's arguments but `output` a slice of a larger sentinel-filled buffer leaves every element
  outside the first `n` unchanged; the same launch with other `BLOCK_SIZE` (covered by `Inst.ok`).

Usage: `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch
import triton
import triton.testing

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "01-vector-add.py")

# The module ends with `benchmark.run(print_data=True, show_plots=True)`; keep its definitions but
# do not run the benchmark (or plot) at import: the launches it makes are checked below.
_orig_run = triton.testing.Mark.run
triton.testing.Mark.run = lambda self, *a, **k: None
spec = importlib.util.spec_from_file_location("vec_add_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
triton.testing.Mark.run = _orig_run

DEVICE = triton.runtime.driver.active.get_active_torch_device()
CHUNK = 1 << 24


def ref_bits(x, y):
    """`FloatV.add a b` from the inputs: the exact sum in `float64`, rounded once to `float32`."""
    out = torch.empty_like(x)
    for s in range(0, x.numel(), CHUNK):
        out[s:s + CHUNK] = (x[s:s + CHUNK].double() + y[s:s + CHUNK].double()).float()
    return out


def post_values(x, y, out):
    """Number of elements `j < n` whose output is not `FloatV.add` up to NaN payload."""
    bad = 0
    for s in range(0, x.numel(), CHUNK):
        r = (x[s:s + CHUNK].double() + y[s:s + CHUNK].double()).float()
        o = out[s:s + CHUNK]
        same = r.view(torch.int32) == o.view(torch.int32)
        both_nan = torch.isnan(r) & torch.isnan(o)
        bad += int((~(same | both_nan)).sum().item())
    return bad


def direct(x, y, block, guard=4096):
    """The host's launch (lines 64-75) with `output` a slice of a guarded buffer."""
    n = x.numel()
    SENT = -12345.5
    buf = torch.full((guard + n + guard,), SENT, dtype=torch.float32, device=x.device)
    out = buf[guard:guard + n]
    mod.add_kernel[(triton.cdiv(n, block),)](x, y, out, n, BLOCK_SIZE=block)
    torch.cuda.synchronize()
    g = torch.cat([buf[:guard], buf[guard + n:]])
    return out, bool((g == SENT).all().item())


results = []


def check(name, x, y, blocks=(), host=True):
    xb, yb = x.clone(), y.clone()
    ok = True
    parts = []
    if host:
        out = mod.add(x, y)
        torch.cuda.synchronize()
        bad = post_values(x, y, out)
        ok &= bad == 0
        parts.append(f"host: {bad} mismatches")
        del out
    for b in blocks:
        out, frame = direct(x, y, b)
        bad = post_values(x, y, out)
        ok &= bad == 0 and frame
        parts.append(f"BLOCK {b}: {bad} mismatches, frame {frame}")
        del out
    ok &= torch.equal(x.view(torch.int32), xb.view(torch.int32))
    ok &= torch.equal(y.view(torch.int32), yb.view(torch.int32))
    results.append(ok)
    print(f"{'ok ' if ok else 'BAD'} {name} (n={x.numel()}): " + "; ".join(parts), flush=True)


# The test (lines 84-89): seed 0, `torch.rand(98432)`.
torch.manual_seed(0)
x = torch.rand(98432, device=DEVICE)
y = torch.rand(98432, device=DEVICE)
check("test", x, y, blocks=(1024, 1, 64, 2 ** 15))

# The benchmark (lines 107-129): `torch.rand(size, dtype=float32)`, size 2^12 ... 2^27.
for e in range(12, 28):
    x = torch.rand(2 ** e, device=DEVICE, dtype=torch.float32)
    y = torch.rand(2 ** e, device=DEVICE, dtype=torch.float32)
    check(f"bench 2^{e}", x, y, blocks=(1024,) if e <= 20 else ())
    del x, y
torch.cuda.empty_cache()

# Edges of Inst.ok: sizes not a multiple of the block, n = 1, every float32 bit pattern class
# (NaN, infinities, signed zeros, subnormals, overflow, ties), and add(x, x).
g = torch.Generator().manual_seed(1)
for n in (1, 2, 1023, 1025, 4097, 98433):
    bits = torch.randint(-2 ** 31, 2 ** 31, (n,), generator=g, dtype=torch.int64).to(torch.int32)
    bits2 = torch.randint(-2 ** 31, 2 ** 31, (n,), generator=g, dtype=torch.int64).to(torch.int32)
    x = bits.view(torch.float32).to(DEVICE)
    y = bits2.view(torch.float32).to(DEVICE)
    check(f"random bit patterns n={n}", x, y, blocks=(1024, 1, 128, 2 ** 15))
special = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, 2.0 ** -126,
                        2.0 ** -149, -2.0 ** -149, 3.4028234663852886e38, 1.0 + 2.0 ** -23,
                        2.0 ** -24, 1.5 * 2.0 ** -24], dtype=torch.float32)
xs = special.repeat_interleave(special.numel()).to(DEVICE)
ys = special.repeat(special.numel()).to(DEVICE)
check("special pairs", xs, ys, blocks=(1024, 1, 16))
check("ties to even", torch.full((3000,), 1.0, device=DEVICE),
      torch.full((3000,), 2.0 ** -24, device=DEVICE), blocks=(1024,))
x = torch.rand(5000, device=DEVICE)
check("add(x, x)", x, x, blocks=(1024,))

print(f"{sum(results)}/{len(results)} cases pass")
assert all(results)
