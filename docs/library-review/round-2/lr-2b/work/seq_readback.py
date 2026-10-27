"""lr-2b, library review round 2, semantics survey: is one program sequential for global memory?

SEMANTICS.md ("Programs", "No barrier semantics") says a program is sequential: each step sees the
memory left by the program's earlier steps, so "a program that writes and then reads back its own
data sees the write", with no barrier in the model. On the device a program is many threads. When
the read-back maps elements to other threads than the write did, nothing in the source orders one
thread's store before another thread's load.

Method. Each program stores a block of `float32` values to its own slice of a scratch buffer (which
starts at a sentinel), then loads the same slice back in reversed order (`BLOCK - 1 - offs`), so
each thread reads elements other threads (other warps) wrote, and stores what it read. In the
sequential semantics every value read is the value written: the output is the reversed input, and no
sentinel survives. Variant `barrier` adds `tl.debug_barrier()` between the two steps (a control).
Variant `same` reads back the same offsets (each thread reads its own writes; a control).
Two more conflicts within one program: `war` loads a block reversed and stores it in place
(`buf[offs] = buf[B - 1 - offs] + 1`; sequentially, every load comes before every store), and `waw`
stores `y` at `offs` and then `y + 1` at `B - 1 - offs` (sequentially, the second store wins at every
element).

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-2/lr-2b/work/seq_readback.py
"""
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"
SENTINEL = -12345.0


@triton.jit
def readback(x_ptr, buf_ptr, out_ptr, B: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + pid * B + offs)
    # some per-element work, so that warps reach the store at different times
    y = x
    for _ in range(8):
        y = tl.where(y > 1e30, y, y * 1.0000001 + 0.5)
    tl.store(buf_ptr + pid * B + offs, y)
    if MODE == 1:
        tl.debug_barrier()
    if MODE == 2:
        r = tl.load(buf_ptr + pid * B + offs)
    else:
        r = tl.load(buf_ptr + pid * B + (B - 1 - offs))
    tl.store(out_ptr + pid * B + offs, r)


@triton.jit
def war(buf_ptr, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    r = tl.load(buf_ptr + pid * B + (B - 1 - offs))
    tl.store(buf_ptr + pid * B + offs, r + 1.0)


@triton.jit
def waw(x_ptr, buf_ptr, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    y = tl.load(x_ptr + pid * B + offs)
    tl.store(buf_ptr + pid * B + offs, y)
    tl.store(buf_ptr + pid * B + (B - 1 - offs), y + 1.0)


def run_war(B, warps, progs, reps):
    wrong = 0
    for _ in range(reps):
        x = torch.randn(progs * B, device=DEV, dtype=torch.float32)
        buf = x.clone()
        war[(progs,)](buf, B, num_warps=warps)
        torch.cuda.synchronize()
        wrong += int((buf.view(progs, B) != x.view(progs, B).flip(1) + 1.0).sum())
    return wrong


def run_waw(B, warps, progs, reps):
    wrong = 0
    x = torch.randn(progs * B, device=DEV, dtype=torch.float32)
    for _ in range(reps):
        buf = torch.full((progs * B,), SENTINEL, device=DEV, dtype=torch.float32)
        waw[(progs,)](x, buf, B, num_warps=warps)
        torch.cuda.synchronize()
        wrong += int((buf.view(progs, B) != (x.view(progs, B) + 1.0).flip(1)).sum())
    return wrong


def run(mode, B, warps, progs, reps):
    x = torch.randn(progs * B, device=DEV, dtype=torch.float32)
    stale = wrong = 0
    for _ in range(reps):
        buf = torch.full((progs * B,), SENTINEL, device=DEV, dtype=torch.float32)
        out = torch.empty_like(buf)
        readback[(progs,)](x, buf, out, B, mode, num_warps=warps)
        torch.cuda.synchronize()
        want = buf.view(progs, B)
        if mode != 2:
            want = want.flip(1)
        got = out.view(progs, B)
        stale += int((got == SENTINEL).sum())
        wrong += int((got != want).sum())
    return stale, wrong


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}", flush=True)
    names = {0: "reversed read-back, no barrier", 1: "reversed read-back, tl.debug_barrier()",
             2: "same-offset read-back, no barrier"}
    progs, reps = 4096, 20
    for B, warps in ((1024, 4), (4096, 8), (8192, 16)):
        for mode in (0, 1, 2):
            stale, wrong = run(mode, B, warps, progs, reps)
            total = progs * B * reps
            print(f"B={B:5d} num_warps={warps:2d} {names[mode]:40s}: {wrong:9d} of {total} elements differ "
                  f"from what the program wrote ({stale} still hold the sentinel)", flush=True)
        total = progs * B * reps
        print(f"B={B:5d} num_warps={warps:2d} {'war: in-place reversed copy':40s}: {run_war(B, warps, progs, reps):9d} "
              f"of {total} elements differ from the sequential result", flush=True)
        print(f"B={B:5d} num_warps={warps:2d} {'waw: second store reversed':40s}: {run_waw(B, warps, progs, reps):9d} "
              f"of {total} elements differ from the sequential result", flush=True)


if __name__ == "__main__":
    main()
