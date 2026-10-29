"""lr-18a, round 18: a kernel module whose own statements bind each module-level name once (the supported subset's
rule, SEMANTICS "The supported subset", Names): `double` by its `def`. rebind_ext.py, another module, rebinds it."""
import triton
import triton.language as tl


@triton.jit
def double(x):
    return x * 2


@triton.jit
def triple(x):
    return x * 3


@triton.jit
def k(x_ptr, o_ptr):
    x = tl.load(x_ptr + tl.arange(0, 16))
    tl.store(o_ptr + tl.arange(0, 16), double(x))
