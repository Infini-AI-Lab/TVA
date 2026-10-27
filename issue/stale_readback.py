"""Library issue L15: one Triton program is not sequential for global memory.

Each program writes its block of `x` to `buf`, then reads the same block back reversed and
writes that to `out`. If the program ran in order, every value read back would be one just
written, so `out` would be `x` reversed per block and would never hold the sentinel that
`buf` starts with. A program is many warps, though, and nothing makes the warp that reads
`buf[j]` wait for the warp that writes it. Each sentinel in `out` is a read that ran first.
A `tl.debug_barrier()` between the two steps removes them.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I issue/stale_readback.py
"""
import torch
import triton
import triton.language as tl

SENTINEL = -12345.0


@triton.jit
def readback(x_ptr, buf_ptr, out_ptr, B: tl.constexpr, BARRIER: tl.constexpr):
    base = tl.program_id(0) * B
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + base + offs)
    tl.store(buf_ptr + base + offs, x)            # step 1: write buf[base, base + B)
    if BARRIER:
        tl.debug_barrier()
    r = tl.load(buf_ptr + base + (B - 1 - offs))  # step 2: read the same block, reversed
    tl.store(out_ptr + base + offs, r)


def main():
    B, programs = 1024, 4096
    print(f"triton {triton.__version__}, {torch.cuda.get_device_name(0)}")
    x = torch.randn(programs * B, device="cuda")
    want = x.view(programs, B).flip(1).reshape(-1)
    for barrier in (False, True):
        buf = torch.full_like(x, SENTINEL)
        out = torch.empty_like(x)
        readback[(programs,)](x, buf, out, B, barrier)
        torch.cuda.synchronize()
        wrong = (out != want).sum().item()
        stale = (out == SENTINEL).sum().item()
        print(f"barrier={barrier!s:5}: {wrong:8d} of {out.numel()} elements wrong, "
              f"{stale} of them the sentinel")


if __name__ == "__main__":
    main()
