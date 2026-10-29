"""TVA library evidence (library issue L17): programmatic dependent launch (`launch_pdl`) and the
grid-dependency-control calls `gdc_wait` / `gdc_launch_dependents`.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/pdl/run.py
Output: evidence/pdl/out.txt (README.md says what it shows).

Part A. A kernel `y = 2 x + 1` with and without `gdc_wait()` (first statement) and
`gdc_launch_dependents()` (after the load), launched with the default `launch_pdl` (False): the
PTX of the gdc variant holds the `griddepcontrol` instructions, and the two variants give the same
output, compared element by element (bit for bit) over several sizes.

Part B. The same kernels launched with `launch_pdl=True`: does the launcher accept it, and is the
output still right when the kernel follows nothing?

Part C (adversarial). A producer kernel signals `gdc_launch_dependents()` at once, spins, then
writes `buf[i] = i + 1`. The consumer, launched next on the same stream, copies `buf` to `out`.
With `launch_pdl=True` and no `gdc_wait`, the consumer may start before the producer has finished
and read the old contents (0). Counted: elements of `out` that differ from `buf`'s final value, for
the consumer without `gdc_wait` and `launch_pdl` False or True, and with `gdc_wait` and
`launch_pdl=True`.

Part D (round 4, library issue L17: the other side). The kernel under claim, A, is launched with
the default `launch_pdl` (a valid launch). It optionally calls `gdc_launch_dependents()` first,
spins, then reads `x` and writes `y = x + 1`. The next launch on the stream, B, writes `x` (and
never `y`), with `launch_pdl` False or True. A's model drops the call and, run alone, ends with
`y = x_init + 1`. Counted: elements with `y != x_init + 1`, that is, A read an `x` other than its
initial memory because B ran while A still ran.
"""
import time

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

DEV = "cuda"


@triton.jit
def affine(x_ptr, y_ptr, n, B: tl.constexpr, GDC: tl.constexpr):
    if GDC:
        gdc_wait()
    offs = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(x_ptr + offs, mask=offs < n)
    if GDC:
        gdc_launch_dependents()
    tl.store(y_ptr + offs, x * 2.0 + 1.0, mask=offs < n)


@triton.jit
def producer(buf_ptr, n, iters, B: tl.constexpr):
    gdc_launch_dependents()
    offs = tl.program_id(0) * B + tl.arange(0, B)
    acc = offs.to(tl.float32)
    for _ in range(iters):
        acc = acc * 1.0000001 + 1e-30
    v = (offs + 1).to(tl.float32) + (acc - acc)
    tl.store(buf_ptr + offs, v, mask=offs < n)


@triton.jit
def consumer(buf_ptr, out_ptr, n, B: tl.constexpr, WAIT: tl.constexpr):
    if WAIT:
        gdc_wait()
    offs = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(out_ptr + offs, tl.load(buf_ptr + offs, mask=offs < n), mask=offs < n)


@triton.jit
def claimed(x_ptr, y_ptr, n, iters, B: tl.constexpr, SIGNAL: tl.constexpr):
    if SIGNAL:
        gdc_launch_dependents()
    offs = tl.program_id(0) * B + tl.arange(0, B)
    acc = offs.to(tl.float32)
    for _ in range(iters):
        acc = acc * 1.0000001 + 1e-30
    x = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(y_ptr + offs, x + 1.0 + (acc - acc), mask=offs < n)


@triton.jit
def overwrite(x_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(x_ptr + offs, tl.full([B], -1000.0, tl.float32), mask=offs < n)


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}", flush=True)
    B = 1024

    print("\n== Part A: launch_pdl default ==")
    x = torch.empty(1, device=DEV)
    for gdc in (False, True):
        k = affine[(1,)](x, torch.empty(1, device=DEV), 1, B, gdc)
        ptx = k.asm["ptx"]
        print(f"GDC={gdc!s:5}: metadata launch_pdl={k.metadata.launch_pdl}; PTX griddepcontrol.wait: "
              f"{ptx.count('griddepcontrol.wait')}, griddepcontrol.launch_dependents: "
              f"{ptx.count('griddepcontrol.launch_dependents')}")
    g = torch.Generator(device=DEV).manual_seed(20261027)
    total_diff = 0
    for n in (1, 7, 1023, 1024, 1025, 65536, 1000003):
        x = torch.randn(n, device=DEV, generator=g) * 1e3
        x[::97] = float("nan")
        outs = []
        for gdc in (False, True):
            y = torch.full((n,), -7.0, device=DEV)
            affine[(triton.cdiv(n, B),)](x, y, n, B, gdc)
            torch.cuda.synchronize()
            outs.append(y.view(torch.int32).clone())
        diff = int((outs[0] != outs[1]).sum())
        total_diff += diff
        print(f"n={n:8d}: elements whose bits differ between GDC=False and GDC=True: {diff}")
    print(f"total differing elements: {total_diff}")

    print("\n== Part B: launch_pdl=True ==")
    for gdc in (False, True):
        n = 1000003
        x = torch.randn(n, device=DEV, generator=g)
        want = torch.full((n,), -7.0, device=DEV)
        affine[(triton.cdiv(n, B),)](x, want, n, B, gdc)
        y = torch.full((n,), -7.0, device=DEV)
        try:
            k = affine[(triton.cdiv(n, B),)](x, y, n, B, gdc, launch_pdl=True)
            torch.cuda.synchronize()
            diff = int((y.view(torch.int32) != want.view(torch.int32)).sum())
            print(f"GDC={gdc!s:5}: launch_pdl=True accepted (compiled kernel's metadata launch_pdl="
                  f"{k.metadata.launch_pdl}); elements differing from the default launch: {diff}")
        except Exception as e:  # noqa: BLE001
            print(f"GDC={gdc!s:5}: launch_pdl=True raised {type(e).__name__}: {e}")

    print("\n== Part C: producer, then consumer on the same stream ==")
    n = 1 << 22
    iters = 20000
    for wait, pdl in ((False, False), (False, True), (True, True)):
        stale_runs, stale_elems = 0, 0
        reps = 20
        for _ in range(reps):
            buf = torch.zeros(n, device=DEV)
            out = torch.full((n,), -1.0, device=DEV)
            torch.cuda.synchronize()
            producer[(triton.cdiv(n, B),)](buf, n, iters, B)
            consumer[(triton.cdiv(n, B),)](buf, out, n, B, wait, launch_pdl=pdl)
            torch.cuda.synchronize()
            bad = int((out != buf).sum())
            stale_runs += bad > 0
            stale_elems += bad
        print(f"consumer gdc_wait={wait!s:5} launch_pdl={pdl!s:5}: runs with stale reads {stale_runs} of {reps}; "
              f"stale elements {stale_elems} of {reps * n}", flush=True)

    print("\n== Part D: the claimed kernel (default launch_pdl), then a launch that writes its input ==")
    for signal, pdl in ((False, False), (True, False), (False, True), (True, True)):
        bad_runs, bad = 0, 0
        reps = 20
        for _ in range(reps):
            x0 = torch.arange(n, device=DEV, dtype=torch.float32)
            x = x0.clone()
            y = torch.full((n,), -7.0, device=DEV)
            torch.cuda.synchronize()
            ka = claimed[(triton.cdiv(n, B),)](x, y, n, iters, B, signal)
            overwrite[(triton.cdiv(n, B),)](x, n, B, launch_pdl=pdl)
            torch.cuda.synchronize()
            k = int((y != x0 + 1.0).sum())
            bad_runs += k > 0
            bad += k
        print(f"claimed kernel calls gdc_launch_dependents={signal!s:5} (its launch_pdl={ka.metadata.launch_pdl}), "
              f"next launch launch_pdl={pdl!s:5}: runs where y != x_init + 1: {bad_runs} of {reps}; "
              f"elements {bad} of {reps * n}", flush=True)


if __name__ == "__main__":
    main()
