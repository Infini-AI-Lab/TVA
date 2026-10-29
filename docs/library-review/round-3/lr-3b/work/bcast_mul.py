"""Library review round 3 (lr-3b), survey area `transcription`: TRANSCRIPTION.md [sum-bcast-mul]
names one spelling, `tl.sum(x[:, :, None] * y[None, :, :], axis=1)`. Triton's combine pass matches
the IR (primitives/triton-3.8.0-Combine.cpp:112-198): a rank-3 `tt.reduce` on axis 1 with one
`arith.addf`, of an `arith.mulf` of two broadcasts of `expand_dims` on axes 2 and 0, M, N >= 16.
So other spellings that give the same IR are rewritten too. With `enable_fp_fusion=False`, a
reviewer who reads the rule literally applies [reduce] and [fusion] instead (`tl.sumc`, whose
nodes are IEEE additions when fusion is off), and the rewritten `tl.dot` (IEEE precision, an FMA
loop on sm_90) may give a result no IEEE tree gives.

K = 2, so every IEEE tree gives round(round(x0*y0) + round(x1*y1)). The script counts the outputs
that differ from it, and reports whether the TTIR has `tt.dot` and the PTX `fma.rn.f32`.

Triton 3.8.0 (~/fpv/tva/.venv), H200 NVL (sm_90), GPU 0. Run from the TVA root:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-3/lr-3b/work/bcast_mul.py
"""
import torch
import triton
import triton.language as tl

M, K, N = 16, 2, 16


@triton.jit
def k_index(x_ptr, y_ptr, o_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    y = tl.load(y_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    r = tl.sum(x[:, :, None] * y[None, :, :], axis=1)
    tl.store(o_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], r)


@triton.jit
def k_expand(x_ptr, y_ptr, o_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    y = tl.load(y_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    r = tl.sum(tl.expand_dims(x, 2) * tl.expand_dims(y, 0), axis=1)
    tl.store(o_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], r)


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def k_reduce(x_ptr, y_ptr, o_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    y = tl.load(y_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    r = tl.reduce(tl.expand_dims(x, 2) * tl.expand_dims(y, 0), 1, _add)
    tl.store(o_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], r)


def main():
    print("Triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    g = torch.Generator(device="cuda").manual_seed(0)
    # products that are inexact in float32, so that a fused and an unfused sum often differ
    x = (1 + torch.rand(M, K, device="cuda", generator=g)).float()
    y = (1 + torch.rand(K, N, device="cuda", generator=g)).float()
    p0 = x[:, 0:1] * y[0:1, :]
    p1 = x[:, 1:2] * y[1:2, :]
    ieee = p0 + p1  # float32 IEEE: round(round(x0 y0) + round(x1 y1)), the only IEEE tree for K = 2
    for name, k in (("x[:, :, None] * y[None, :, :] (the rule's spelling)", k_index),
                    ("tl.expand_dims(x, 2) * tl.expand_dims(y, 0)", k_expand),
                    ("tl.reduce(expand_dims product, 1, add)", k_reduce)):
        for fusion in (True, False):
            o = torch.empty(M, N, device="cuda")
            c = k[(1,)](x, y, o, M, K, N, enable_fp_fusion=fusion)
            ttir, ptx = c.asm["ttir"], c.asm["ptx"]
            differ = (o != ieee).sum().item()
            print(f"{name}, enable_fp_fusion={fusion}: tt.dot in TTIR: {'tt.dot' in ttir}; "
                  f"fma.rn.f32 in PTX: {'fma.rn.f32' in ptx}; outputs that no IEEE tree gives: {differ}/{M * N}")


if __name__ == "__main__":
    main()
