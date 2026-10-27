"""lr-1, library review round 1, L6: what does a `tl.static_assert` line do in Triton 3.8.0, and what do the
lines dropped as `no-op:print` do?

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-1/lr-1/work/l6_static_assert.py
(Triton 3.8.0, NVIDIA H200 NVL, sm_90.)

The front end (code_generator.py, `execute_static_assert`, and `static_executor` for
`tl.static_print`) evaluates the call's arguments as ordinary expressions before it looks at them.
So a holding assertion has no effect only when its condition contains no operation with an effect.
"""
import re
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def with_or_without(x_ptr, out_ptr, B: tl.constexpr, WITH: tl.constexpr):
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + offs)
    if WITH:
        tl.static_assert(B % 16 == 0, "B must be a multiple of 16")
        tl.static_assert(x.dtype == tl.float32)
        tl.static_assert(x.shape[0] == B)
    tl.store(out_ptr + offs, x + 1.0)


@triton.jit
def runtime_condition(x_ptr, out_ptr, B: tl.constexpr):
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + offs)
    tl.static_assert(tl.sum(x) > 0.0)
    tl.store(out_ptr + offs, x)


@triton.jit
def store_in_condition(x_ptr, out_ptr, B: tl.constexpr):
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + offs)
    # the assertion holds (a store returns a value that is not None), and the store is emitted
    tl.static_assert(tl.store(out_ptr + B + offs, x * 2.0) is not None)
    tl.store(out_ptr + offs, x)


@triton.jit
def _helper(p, B: tl.constexpr):
    tl.store(p + tl.arange(0, B), tl.full((B,), 3.0, tl.float32))


@triton.jit
def helper_in_condition(x_ptr, out_ptr, B: tl.constexpr):
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + offs)
    tl.static_assert(_helper(out_ptr + B, B) is None)
    tl.store(out_ptr + offs, x)


@triton.jit
def atomic_in_static_print(cnt_ptr, B: tl.constexpr):
    tl.static_print(tl.atomic_add(cnt_ptr, 1))


@triton.jit
def atomic_in_device_assert(cnt_ptr, B: tl.constexpr):
    tl.device_assert(tl.atomic_add(cnt_ptr, 1) >= 0)


@triton.jit
def atomic_in_device_print(cnt_ptr, B: tl.constexpr):
    tl.device_print("old", tl.atomic_add(cnt_ptr, 1))


def strip_loc(ir):
    ir = re.sub(r"\s*loc\([^()]*(\([^()]*\))*[^()]*\)", "", ir)
    return "\n".join(l for l in ir.splitlines() if not l.startswith("#loc"))


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}")
    B = 16
    x = torch.arange(1, B + 1, device=DEV, dtype=torch.float32)

    print("\n== S1: holding assertions (constexpr, dtype, shape) change nothing ==")
    out = torch.zeros(B, device=DEV)
    k1 = with_or_without.warmup(x, out, B, True, grid=(1,))
    k0 = with_or_without.warmup(x, out, B, False, grid=(1,))
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        same = strip_loc(k1.asm[stage]) == strip_loc(k0.asm[stage])
        print(f"    {stage}: with and without the assertions identical (locations stripped): {same}")
    with_or_without[(1,)](x, out, B, True)
    print(f"    out == x + 1: {bool(torch.equal(out, x + 1))}")

    print("\n== S2: a condition that is not a compile-time bool is rejected on the host ==")
    out = torch.zeros(B, device=DEV)
    try:
        runtime_condition[(1,)](x, out, B)
        print("    host error: none")
    except Exception as e:  # noqa: BLE001
        print(f"    host error: {type(e).__name__}; out untouched: {bool((out == 0).all())}")

    print("\n== S3: a holding tl.static_assert whose condition stores ==")
    out = torch.zeros(2 * B, device=DEV)
    store_in_condition[(1,)](x, out, B)
    torch.cuda.synchronize()
    print(f"    compiles: True; out[B:2B] == 2x (written by the assertion's condition): "
          f"{bool(torch.equal(out[B:], 2 * x))}")
    k = store_in_condition.warmup(x, out, B, grid=(1,))
    print(f"    tt.store ops in the TTIR: {strip_loc(k.asm['ttir']).count('tt.store')}")

    print("\n== S4: a holding tl.static_assert whose condition calls a @triton.jit helper that stores ==")
    out = torch.zeros(2 * B, device=DEV)
    try:
        helper_in_condition[(1,)](x, out, B)
        torch.cuda.synchronize()
        print(f"    compiles: True; out[B:2B] == 3 (written by the helper): {bool((out[B:] == 3.0).all())}")
    except Exception as e:  # noqa: BLE001
        print(f"    host error: {type(e).__name__}: {str(e).splitlines()[0][:100] if str(e) else ''}")

    print("\n== S5: lines the transcription drops as no-op:print, with an atomic in an argument (debug off) ==")
    for name, kern in (("tl.static_print", atomic_in_static_print), ("tl.device_assert", atomic_in_device_assert),
                       ("tl.device_print", atomic_in_device_print)):
        cnt = torch.zeros(1, device=DEV, dtype=torch.int32)
        try:
            kern[(4,)](cnt, B)
            torch.cuda.synchronize()
            print(f"    {name}(tl.atomic_add(cnt, 1) ...), grid 4: cnt = {int(cnt.item())} (0 if the line had no effect)")
        except Exception as e:  # noqa: BLE001
            print(f"    {name}: host error {type(e).__name__}: {str(e).splitlines()[0][:100] if str(e) else ''}")


if __name__ == "__main__":
    main()
