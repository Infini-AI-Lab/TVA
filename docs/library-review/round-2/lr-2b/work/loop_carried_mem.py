"""lr-2b, library review round 2, semantics survey: a loop whose iteration reads what the previous
iteration stored, at the same offsets (each thread reads its own writes).

Sequentially, iteration `i` loads row `i` of a buffer, which iteration `i - 1` stored, and stores
`row + 1` into row `i + 1`. So row `i` ends as `row 0 + i`. If the compiler's software pipelining
issues a later iteration's load before an earlier iteration's store, the load reads the initial
contents (`0`) instead. Run with `num_stages` 1 and 3, with and without a dot in the loop (Triton
pipelines loads that feed a `tl.dot`), and with `tl.range(..., num_stages=3)`, also with a
`tl.debug_barrier()` after the store. A control loads and stores back the same block in every
iteration (a loop-invariant address).

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-2/lr-2b/work/loop_carried_mem.py
"""
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def chain(buf_ptr, n, B: tl.constexpr):
    offs = tl.arange(0, B)
    for i in range(0, n):
        x = tl.load(buf_ptr + i * B + offs)
        tl.store(buf_ptr + (i + 1) * B + offs, x + 1.0)


@triton.jit
def chain_range(buf_ptr, n, B: tl.constexpr):
    offs = tl.arange(0, B)
    for i in tl.range(0, n, num_stages=3):
        x = tl.load(buf_ptr + i * B + offs)
        tl.store(buf_ptr + (i + 1) * B + offs, x + 1.0)


@triton.jit
def chain_range_barrier(buf_ptr, n, B: tl.constexpr):
    offs = tl.arange(0, B)
    for i in tl.range(0, n, num_stages=3):
        x = tl.load(buf_ptr + i * B + offs)
        tl.store(buf_ptr + (i + 1) * B + offs, x + 1.0)
        tl.debug_barrier()


@triton.jit
def bump_in_place(buf_ptr, n, B: tl.constexpr):
    # a loop-invariant address: every iteration loads the block and stores it back plus one
    offs = tl.arange(0, B)
    for i in range(0, n):
        x = tl.load(buf_ptr + offs)
        tl.store(buf_ptr + offs, x + 1.0)


@triton.jit
def chain_dot(buf_ptr, w_ptr, n, B: tl.constexpr):
    # rows of a B x B tile: iteration i loads tile i, multiplies by w (the identity), stores tile i + 1
    r = tl.arange(0, B)
    tile = r[:, None] * B + r[None, :]
    w = tl.load(w_ptr + tile)
    for i in range(0, n):
        x = tl.load(buf_ptr + i * B * B + tile)
        y = tl.dot(x, w, input_precision="ieee")
        tl.store(buf_ptr + (i + 1) * B * B + tile, y + 1.0)


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}", flush=True)
    n, B = 64, 1024
    for kern, name in ((chain, "range"), (chain_range, "tl.range(num_stages=3)"),
                       (chain_range_barrier, "... + tl.debug_barrier()")):
        for stages in (1, 3):
            buf = torch.zeros((n + 1) * B, device=DEV, dtype=torch.float32)
            kern[(1,)](buf, n, B, num_stages=stages)
            torch.cuda.synchronize()
            want = torch.arange(n + 1, device=DEV, dtype=torch.float32)[:, None].expand(n + 1, B)
            bad = int((buf.view(n + 1, B) != want).sum())
            print(f"{name:24s} num_stages={stages}: {bad} of {(n + 1) * B} elements differ from the sequential result",
                  flush=True)
    for stages in (1, 3):
        buf = torch.zeros(B, device=DEV, dtype=torch.float32)
        bump_in_place[(1,)](buf, n, B, num_stages=stages)
        torch.cuda.synchronize()
        bad = int((buf != float(n)).sum())
        print(f"{'in place, same address':24s} num_stages={stages}: {bad} of {B} elements differ from the sequential "
              f"result ({n})", flush=True)
    B2 = 32
    for stages in (1, 3, 4):
        buf = torch.zeros((n + 1) * B2 * B2, device=DEV, dtype=torch.float32)
        w = torch.eye(B2, device=DEV, dtype=torch.float32)
        k = chain_dot[(1,)](buf, w, n, B2, num_stages=stages)
        torch.cuda.synchronize()
        want = torch.arange(n + 1, device=DEV, dtype=torch.float32)[:, None].expand(n + 1, B2 * B2)
        bad = int((buf.view(n + 1, B2 * B2) != want).sum())
        print(f"{'range with tl.dot':24s} num_stages={stages}: {bad} of {(n + 1) * B2 * B2} elements differ from the "
              f"sequential result; cp.async in PTX: {'cp.async' in k.asm['ptx']}", flush=True)


if __name__ == "__main__":
    main()
