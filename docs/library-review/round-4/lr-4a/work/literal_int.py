"""lr-4a, L23: `[literal]` says a Python number in `tl.full(shape, c, d)` with an integer dtype `d`
is "`c` itself". What does Triton 3.8.0 build when `c` is not a value of `d`, or is a float?
And what do the binary operators do with such a number (they check the range)?

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/literal_int.py
"""
import torch
import triton
import triton.language as tl

TD = {"int8": (tl.int8, torch.int8), "uint8": (tl.uint8, torch.uint8), "int16": (tl.int16, torch.int16),
      "int32": (tl.int32, torch.int32), "uint32": (tl.uint32, torch.uint32), "int64": (tl.int64, torch.int64),
      "float32": (tl.float32, torch.float32), "bfloat16": (tl.bfloat16, torch.bfloat16)}


@triton.jit
def k_full(o, C: tl.constexpr, DT: tl.constexpr):
    tl.store(o + tl.arange(0, 1), tl.full([1], C, DT))


@triton.jit
def k_add(x, o, C: tl.constexpr):
    tl.store(o + tl.arange(0, 1), tl.load(x + tl.arange(0, 1)) + C)


def run_full(c, d):
    tdt, torch_dt = TD[d]
    o = torch.zeros(1, dtype=torch_dt, device="cuda")
    try:
        k_full[(1,)](o, C=c, DT=tdt)
        torch.cuda.synchronize()
        return f"stores {o.item()!r}"
    except Exception as e:  # noqa: BLE001
        return f"rejected on the host ({type(e).__name__}: {str(e).splitlines()[-1][:90]})"


def run_add(c, d):
    tdt, torch_dt = TD[d]
    x = torch.zeros(1, dtype=torch_dt, device="cuda")
    o = torch.zeros(1, dtype=torch_dt, device="cuda")
    try:
        k_add[(1,)](x, o, C=c)
        torch.cuda.synchronize()
        return f"stores {o.item()!r}"
    except Exception as e:  # noqa: BLE001
        return f"rejected on the host ({type(e).__name__}: {str(e).splitlines()[-1][:90]})"


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    for c, d in [(300, "int8"), (-1, "uint8"), (256, "uint8"), (70000, "int16"), (2**31, "int32"), (2**40, "int32"),
                 (-1, "uint32"), (2**63, "int64"), (1.5, "int32"), (-2.7, "int8"), (True, "float32"), (2**40 + 1, "float32"),
                 (3.0, "int32")]:
        print(f"tl.full([1], {c!r}, tl.{d}): {run_full(c, d)}")
    for c, d in [(300, "int8"), (2**31, "int32"), (-1, "uint8"), (1.5, "int32")]:
        print(f"x + {c!r} with x a tl.{d} zero: {run_add(c, d)}")


if __name__ == "__main__":
    main()
