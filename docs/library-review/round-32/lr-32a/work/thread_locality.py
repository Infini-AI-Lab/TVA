"""lr-32a: Triton 3.8.0's OptimizeThreadLocality rewrite, beyond the author's evidence/reduce-tree part L.

All inputs are small integers, so every float sum and product here is exact in float32 and every
association of the source gives the same value: a result other than the source's is not a
reassociation but a different computation.

  U1-U6  the update operation after the reduction is not the reduction's combine (or is `-`):
         acc - sum, sum - acc, acc + max, maximum(acc, sum), acc * sum, acc + min
  F1-F7  near-miss source forms: is the pattern visible as "a tile produced directly by tl.load"?
         .to(same dtype), * 1.0, block pointer, helper function, float16 tl.sum (upcast), float16
         tl.max, a host-descriptor load
Each line: the stored values, the source's value, whether the TTGIR has the rewrite (a 3-D reshape
and a tt.reduce after the loop).
Run: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I thread_locality.py
"""
import re

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEV = "cuda"
M, N, T = 64, 64, 4


@triton.jit
def k_upd(x_ptr, o_ptr, OP: tl.constexpr, T: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    if OP == 4:
        acc = tl.full([M], 1.0, tl.float32)
    else:
        acc = tl.zeros([M], dtype=tl.float32)
    for t in range(T):
        x = tl.load(x_ptr + t * M * N + rm[:, None] * N + rn[None, :])
        if OP == 0:
            acc = acc - tl.sum(x, axis=1)
        elif OP == 1:
            acc = tl.sum(x, axis=1) - acc
        elif OP == 2:
            acc = acc + tl.max(x, axis=1)
        elif OP == 3:
            acc = tl.maximum(acc, tl.sum(x, axis=1))
        elif OP == 4:
            acc = acc * tl.sum(x, axis=1)
        elif OP == 5:
            acc = acc + tl.min(x, axis=1)
        else:
            acc = acc + tl.sum(x, axis=1)
    tl.store(o_ptr + rm, acc)


@triton.jit
def helper_load(p):
    return tl.load(p)


@triton.jit
def k_form(x_ptr, h_ptr, o_ptr, desc, FORM: tl.constexpr, T: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    acc = tl.zeros([M], dtype=tl.float32)
    for t in range(T):
        offs = t * M * N + rm[:, None] * N + rn[None, :]
        if FORM == 0:
            acc = acc + tl.sum(tl.load(x_ptr + offs).to(tl.float32), axis=1)
        elif FORM == 1:
            acc = acc + tl.sum(tl.load(x_ptr + offs) * 1.0, axis=1)
        elif FORM == 2:
            bp = tl.make_block_ptr(x_ptr, (T * M, N), (N, 1), (t * M, 0), (M, N), (1, 0))
            acc = acc + tl.sum(tl.load(bp), axis=1)
        elif FORM == 3:
            acc = acc + tl.sum(helper_load(x_ptr + offs), axis=1)
        elif FORM == 4:
            acc = acc + tl.sum(tl.load(h_ptr + offs), axis=1)
        elif FORM == 5:
            acc = acc + tl.max(tl.load(h_ptr + offs), axis=1)
        elif FORM == 6:
            acc = acc + tl.sum(desc.load([t * M, 0]), axis=1)
        else:
            acc = acc + tl.sum(tl.load(x_ptr + offs), axis=1)
    tl.store(o_ptr + rm, acc)


def rewritten(ttgir):
    return bool(re.search(r"tt\.reshape .*tensor<\d+x\d+x\d+x", ttgir))


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    x = torch.ones(T, M, N, dtype=torch.float32, device=DEV)
    x[:, :, 1] = 2.0  # max 2, min 1, sum N + 1 per row and trip
    s = float(N + 1)
    names = {0: ("acc - sum", -T * s), 1: ("sum - acc", None), 2: ("acc + max", T * 2.0),
             3: ("maximum(acc, sum)", s), 4: ("acc * sum (from 1)", s ** T), 5: ("acc + min", T * 1.0),
             6: ("acc + sum", T * s)}
    # sum - acc: a_{t+1} = s - a_t from 0 gives s, 0, s, 0, ... ; after T = 4 trips, 0
    names[1] = ("sum - acc", 0.0)
    for op, (nm, want) in names.items():
        o = torch.full((M,), -7.0, dtype=torch.float32, device=DEV)
        h = k_upd[(1,)](x, o, OP=op, T=T, M=M, N=N, num_warps=4)
        vals = sorted(set(o.cpu().tolist()))
        print(f"U{op + 1} {nm:20s}: stored {vals}, source {want}, rewrite {rewritten(h.asm['ttgir'])}")
    xh = x.to(torch.float16)
    desc = TensorDescriptor(x.view(T * M, N), shape=[T * M, N], strides=[N, 1], block_shape=[M, N])
    forms = {0: ".to(float32) of a float32 load", 1: "load * 1.0", 2: "block pointer load",
             3: "load in a helper", 4: "float16 tl.sum", 5: "float16 tl.max", 6: "host descriptor load",
             7: "plain load (control)"}
    want = {4: T * s, 5: T * 2.0}
    for f, nm in forms.items():
        o = torch.full((M,), -7.0, dtype=torch.float32, device=DEV)
        h = k_form[(1,)](x, xh, o, desc, FORM=f, T=T, M=M, N=N, num_warps=4)
        vals = sorted(set(o.cpu().tolist()))
        print(f"F{f + 1} {nm:32s}: stored {vals}, source {want.get(f, T * s)}, rewrite {rewritten(h.asm['ttgir'])}")


if __name__ == "__main__":
    main()
