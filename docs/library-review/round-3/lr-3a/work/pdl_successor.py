"""lr-3a, library review round 3: can a kernel launched *without* programmatic dependent launch
(`launch_pdl=False`, so `Launch.Valid`) see its memory changed during its run, when it calls
`gdc_launch_dependents()` (dropped by `[no-op:gdc]`) and the next kernel on the stream is
launched with `launch_pdl=True`?

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-3/lr-3a/work/pdl_successor.py

Kernel A (the kernel under claim, default `launch_pdl`): optionally calls
`gdc_launch_dependents()` first, spins, then loads `x` and stores `y = x + 1`. Its model drops the
call, and from `init` it ends with `y = x_init + 1` and `x = x_init`.
Kernel B (the next launch on the stream): stores `x[i] = -1000` (it never touches `y`).
Counted, per variant, over 20 runs of `2^22` elements: elements with `y != x_init + 1`, i.e. A read
an `x` that was not its initial memory.
"""
import time

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents

DEV = "cuda"


@triton.jit
def kernel_a(x_ptr, y_ptr, n, iters, B: tl.constexpr, SIGNAL: tl.constexpr):
    if SIGNAL:
        gdc_launch_dependents()
    offs = tl.program_id(0) * B + tl.arange(0, B)
    acc = offs.to(tl.float32)
    for _ in range(iters):
        acc = acc * 1.0000001 + 1e-30
    x = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(y_ptr + offs, x + 1.0 + (acc - acc), mask=offs < n)


@triton.jit
def kernel_b(x_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(x_ptr + offs, tl.full([B], -1000.0, tl.float32), mask=offs < n)


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"{time.strftime('%Y-%m-%d')}", flush=True)
    B, n, iters, reps = 1024, 1 << 22, 20000, 20
    for signal, pdl in ((True, False), (False, True), (True, True)):
        bad_runs, bad = 0, 0
        for _ in range(reps):
            x0 = torch.arange(n, device=DEV, dtype=torch.float32)
            x = x0.clone()
            y = torch.full((n,), -7.0, device=DEV)
            torch.cuda.synchronize()
            ka = kernel_a[(triton.cdiv(n, B),)](x, y, n, iters, B, signal)
            kernel_b[(triton.cdiv(n, B),)](x, n, B, launch_pdl=pdl)
            torch.cuda.synchronize()
            k = int((y != x0 + 1.0).sum())
            bad_runs += k > 0
            bad += k
        print(f"A calls gdc_launch_dependents={signal!s:5} (A's launch_pdl={ka.metadata.launch_pdl}), "
              f"B launch_pdl={pdl!s:5}: runs where A's y != x_init + 1: {bad_runs} of {reps}; "
              f"elements {bad} of {reps * n}", flush=True)


if __name__ == "__main__":
    main()
