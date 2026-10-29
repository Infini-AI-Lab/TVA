"""lr-19a: a Python NaN passed to a helper's parameter (in the supported subset: SEMANTICS, "Parameters
and arguments", "A helper's parameter holds what the call passes"). The front end names each
specialised helper by `mangle_fn` (primitives/triton-3.8.0-code-generator.py:32-41, 1369-1371),
which writes a constexpr argument as `repr(value)` (primitives/triton-3.8.0-core.py:196-201), and
`repr` of every NaN is `nan`. So two calls of one helper, one with `float('nan')` and one with
`-float('nan')`, get one function: the second call runs the first's body.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I helper_nan.py
"""
import struct
import torch
import triton
import triton.language as tl


@triton.jit
def store_c(p, c):
    tl.store(p + tl.arange(0, 1), tl.full([1], c, tl.float32))


@triton.jit
def store_c_bf16(p, c):
    tl.store(p + tl.arange(0, 1), tl.full([1], c, tl.bfloat16))


@triton.jit
def store_where(p, c):
    i = tl.arange(0, 1)
    x = tl.load(p + i)
    tl.store(p + i, tl.where(i < 0, x, c))


@triton.jit
def k_pos_neg(o):
    store_c(o, float('nan'))
    store_c(o + 1, -float('nan'))


@triton.jit
def k_neg_pos(o):
    store_c(o, -float('nan'))
    store_c(o + 1, float('nan'))


@triton.jit
def k_named(o):
    A: tl.constexpr = float('nan')
    B: tl.constexpr = float('inf') - float('inf')
    store_c(o, A)
    store_c(o + 1, B)


@triton.jit
def k_keyword(o):
    store_c(o, c=float('nan'))
    store_c(p=o + 1, c=-float('nan'))


@triton.jit
def k_bf16(o):
    store_c_bf16(o, float('nan'))
    store_c_bf16(o + 1, -float('nan'))


@triton.jit
def k_where(o):
    store_where(o, -float('nan'))
    store_where(o + 1, float('nan'))


@triton.jit
def k_inline(o):
    # control: no helper, the same two constants
    tl.store(o + tl.arange(0, 1), tl.full([1], float('nan'), tl.float32))
    tl.store(o + 1 + tl.arange(0, 1), tl.full([1], -float('nan'), tl.float32))


@triton.jit
def k_inf(o):
    # control: infinities through the helper (repr 'inf' and '-inf' differ)
    store_c(o, float('inf'))
    store_c(o + 1, -float('inf'))


def bits(t):
    w = {torch.float32: (torch.int32, 32), torch.bfloat16: (torch.int16, 16)}[t.dtype]
    return [f"{v & ((1 << w[1]) - 1):x}" for v in t.view(w[0]).tolist()]


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("repr(float('nan')) =", repr(float('nan')), "; repr(-float('nan')) =", repr(-float('nan')))
    cases = [
        ("H1 store_c(o, nan); store_c(o+1, -nan)", k_pos_neg, torch.float32, ["7fc00000", "ffc00000"]),
        ("H2 store_c(o, -nan); store_c(o+1, nan)", k_neg_pos, torch.float32, ["ffc00000", "7fc00000"]),
        ("H3 A: constexpr = nan, B: constexpr = inf - inf; store_c(o, A); store_c(o+1, B)", k_named,
         torch.float32, ["7fc00000", "ffc00000"]),
        ("H4 keywords: store_c(o, c=nan); store_c(p=o+1, c=-nan)", k_keyword, torch.float32,
         ["7fc00000", "ffc00000"]),
        ("H5 bfloat16: store_c_bf16(o, nan); store_c_bf16(o+1, -nan)", k_bf16, torch.bfloat16, ["7fc0", "ffc0"]),
        ("H6 tl.where in a helper: store_where(o, -nan); store_where(o+1, nan)", k_where, torch.float32,
         ["ffc00000", "7fc00000"]),
        ("C1 control, no helper: tl.full(nan), tl.full(-nan)", k_inline, torch.float32, ["7fc00000", "ffc00000"]),
        ("C2 control, infinities through the helper", k_inf, torch.float32, ["7f800000", "ff800000"]),
    ]
    for name, k, dt, want in cases:
        o = torch.zeros(2, dtype=dt, device="cuda")
        k[(1,)](o)
        torch.cuda.synchronize()
        got = bits(o)
        verdict = "as SEMANTICS 'Constants' states" if got == want else "DIFFERS from SEMANTICS 'Constants'"
        print(f"{name}: stored {got}, library (litF32/litBF16 of each call's double) {want}: {verdict}")


if __name__ == "__main__":
    main()
