"""lr-34a, L149: tl.where(c, tl.load(p, mask=c), other) with a scalar runtime condition c.

Triton's combine pass rewrites select(c, load(p, splat(c)), other) into load(p, splat(c), other)
(primitives/triton-3.8.0-Combine.cpp:68-107), so OptimizeThreadLocality then sees a tt.load operand.
Data as lr-33a's otl_wording33.py: two trips of [64, 64] tiles; trip 0 has 2^24 in column 0 and 1 in
column 63, trip 1 -2^24 and 1. "rewritten" = the TTGIR holds a tt.reshape to rank 3.
"""
import re
import torch
import triton
import triton.language as tl

M, N, T = 64, 64, 2


@triton.jit
def k(x_ptr, o_ptr, cv, FORM: tl.constexpr, IV: tl.constexpr, T: tl.constexpr, M: tl.constexpr,
      N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    c = cv > 0
    acc = tl.zeros([M], dtype=tl.float32) + IV
    for t in range(T):
        p = x_ptr + t * M * N + rm[:, None] * N + rn[None, :]
        if FORM == 0:  # where on the load's own scalar mask, other 0.0
            x = tl.where(c, tl.load(p, mask=c), 0.0)
        elif FORM == 1:  # where on the load's own scalar mask, other 5.0
            x = tl.where(c, tl.load(p, mask=c), 5.0)
        else:  # control: the load with other=0.0
            x = tl.load(p, mask=c, other=0.0)
        acc = acc + tl.sum(x, axis=1)
    tl.store(o_ptr + rm, acc)


def main():
    x = torch.zeros(T, M, N, dtype=torch.float32, device="cuda")
    x[0, :, 0] = 2.0 ** 24
    x[0, :, 63] = 1.0
    x[1, :, 0] = -2.0 ** 24
    x[1, :, 63] = 1.0
    for form, name in [(0, "tl.where(c, tl.load(p, mask=c), 0.0)"), (1, "tl.where(c, tl.load(p, mask=c), 5.0)"),
                       (2, "control: tl.load(p, mask=c, other=0.0)")]:
        for iv in [0.0, 1.0]:
            for cv in [5, 0]:
                o = torch.full((M,), -7.0, device="cuda")
                h = k[(1,)](x, o, cv, FORM=form, IV=iv, T=T, M=M, N=N, num_warps=4)
                torch.cuda.synchronize()
                rw = bool(re.search(r"tt\.reshape [^\n]*-> tensor<\d+x\d+x\d+x", h.asm["ttgir"]))
                if cv > 0:
                    src = iv + 1.0  # trips 2^24 + 1 -> 2^24 (rounding), then -2^24 + 1 -> 1.0 ... see below
                    src = None
                else:
                    src = iv + (0.0 if form != 1 else T * N * 5.0)
                vals = sorted(set(o.tolist()))
                print(f"{name}; acc starts {iv}; cv={cv} (c {'true' if cv > 0 else 'false'}): rewritten={rw}, "
                      f"stored {vals}" + ("" if src is None else f", the source gives {src}"))


if __name__ == "__main__":
    main()
