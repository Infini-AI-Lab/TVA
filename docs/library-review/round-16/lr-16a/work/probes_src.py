"""The kernel sources of lr-16a's Part U, shared by probes.py (the map check) and unused_gpu.py (the device).

In both sources the kernel lines are 1-17 (the imports, the module-level names the kernel reads, the kernel and
its helper), `triple` and `_unused_helper` are in `source.unused`, and `run` (27-31) is host code. The unused
function's default rebinds, when Python executes its `def` statement at import, a name the kernel reads: `SCALE`
(a `tl.constexpr` global, which the supported subset excludes) or `double` (a `@triton.jit` helper, a
module-level name the supported subset admits)."""

SCALE_SRC = '''import torch
import triton
import triton.language as tl

SCALE = tl.constexpr(2)


@triton.jit
def k(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, x * SCALE)





@triton.jit
def triple(x):
    return x * 3


def _unused_helper(x=(SCALE := tl.constexpr(3))):
    return x


def run():
    x = torch.arange(16, dtype=torch.float32, device="cuda")
    o = torch.empty(16, dtype=torch.float32, device="cuda")
    k[(1,)](x, o)
    return o
'''

JIT_SRC = '''import torch
import triton
import triton.language as tl


@triton.jit
def double(x):
    return x * 2


@triton.jit
def k(x_ptr, o_ptr):
    i = tl.arange(0, 16)
    x = tl.load(x_ptr + i)
    tl.store(o_ptr + i, double(x))


@triton.jit
def triple(x):
    return x * 3


def _unused_helper(f=(double := triple)):
    return f


def run():
    x = torch.arange(16, dtype=torch.float32, device="cuda")
    o = torch.empty(16, dtype=torch.float32, device="cuda")
    k[(1,)](x, o)
    return o
'''

SOURCES = {"SCALE": SCALE_SRC, "JIT": JIT_SRC}
MANIFEST = {"source": {"ranges": [[1, 17]], "host": [[27, 31]], "unused": ["triple", "_unused_helper"]}}
