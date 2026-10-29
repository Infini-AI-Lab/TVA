"""lr-20e: is a float32 `tl.dot(..., input_precision="ieee")` a sequence of `fma`s with `enable_fp_fusion=False`?

docs/rules/num.md ("`tl.dot`", `DotF32Ieee`) says Triton runs `K` sequential `fma.rn.f32` (evidence/num-dot Part
4, with the default fusion). One element where an `fma` and a rounded product then an addition differ:
`a = b = 1 + 2^-12`, `acc = -(1 + 2^-11)`: `fma` gives `2^-24`, `mul` then `add` gives `0`. Both are within
`DotF32Ieee`'s bound; this only says which instruction runs.

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I ieee_dot_fusion.py
"""
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def dk(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, c, input_precision="ieee"))


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    M = N = K = 16
    a = torch.zeros((M, K), device="cuda")
    b = torch.zeros((K, N), device="cuda")
    a[:, 0] = 1 + 2.0 ** -12
    b[0, :] = 1 + 2.0 ** -12
    c = torch.full((M, N), -(1 + 2.0 ** -11), device="cuda")
    for fusion in (True, False):
        for warps in (1, 4):
            o = torch.empty_like(c)
            h = dk[(1,)](a, b, c, o, M=M, N=N, K=K, num_warps=warps, enable_fp_fusion=fusion)
            torch.cuda.synchronize()
            ptx = h.asm["ptx"]
            print(f"enable_fp_fusion={fusion}, num_warps={warps}: element {float(o[0, 0])!r} "
                  f"(fma: {2.0 ** -24!r}, mul then add: 0.0); PTX has fma.rn.f32: {'fma.rn.f32' in ptx}, "
                  f"mul.rn.f32: {'mul.rn.f32' in ptx}")


if __name__ == "__main__":
    main()
