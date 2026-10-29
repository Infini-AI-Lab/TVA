"""TVA library evidence (library issue L35, the integer case of `[literal]`): what `tl.full(shape, c, d)`
holds for a Python `int` `c` in an integer dtype `d`, and what an operator does with an `int` operand
outside `d`'s range.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/literals/int_full.py
Output: evidence/literals/int_full.out.txt (README.md, "Integer constants").

For each integer dtype, the constants are the range's ends, one past each end, `±2^w`, `2^w ± 1`, 300,
`-300`, and 40 random integers in `[-2^(w+8), 2^(w+8)]`. Each is stored by `tl.full([1], c, d)`, and the
stored bits are compared with `c` modulo `2^w` (the library's `IntV.ofInt c`, `BitVec.ofInt w c`). A
compile error is a launch rejected on the host. Then `x + c` with `x` a tensor of dtype `d` is compiled
for `c` one past each end.
"""
import os
import random
import tempfile

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="tva-int-full-")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

DTYPES = [("int8", tl.int8, torch.int8, 8, True), ("int16", tl.int16, torch.int16, 16, True),
          ("int32", tl.int32, torch.int32, 32, True), ("int64", tl.int64, torch.int64, 64, True),
          ("uint8", tl.uint8, torch.uint8, 8, False), ("uint16", tl.uint16, torch.uint16, 16, False),
          ("uint32", tl.uint32, torch.uint32, 32, False), ("uint64", tl.uint64, torch.uint64, 64, False)]


@triton.jit
def full_kernel(o_ptr, C: tl.constexpr, D: tl.constexpr):
    tl.store(o_ptr + tl.arange(0, 1), tl.full([1], C, D))


@triton.jit
def add_kernel(x_ptr, o_ptr, C: tl.constexpr):
    tl.store(o_ptr, tl.load(x_ptr) + C)


def bits(t, w):
    """The stored element's bit pattern, as an unsigned integer."""
    v = int(t.view(torch.int64 if w == 64 else torch.int32 if w == 32 else torch.int16 if w == 16 else torch.int8)
            .cpu()[0].item())
    return v % (1 << w)


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    rng = random.Random(5)
    for name, d, td, w, signed in DTYPES:
        lo, hi = (-(1 << (w - 1)), (1 << (w - 1)) - 1) if signed else (0, (1 << w) - 1)
        cs = [lo, hi, lo - 1, hi + 1, 1 << w, -(1 << w), (1 << w) + 1, (1 << w) - 1, 300, -300]
        cs += [rng.randint(-(1 << (w + 8)), 1 << (w + 8)) for _ in range(40)]
        same = rejected = differ = 0
        examples = []
        for c in cs:
            o = torch.zeros(1, dtype=td, device="cuda")
            try:
                full_kernel[(1,)](o, C=c, D=d)
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                rejected += 1
                if len(examples) < 3:
                    examples.append(f"{c}: rejected")
                continue
            if bits(o, w) == c % (1 << w):
                same += 1
            else:
                differ += 1
                examples.append(f"{c}: stored bits {bits(o, w):#x}, c mod 2^w {c % (1 << w):#x}")
        print(f"tl.full in {name}: {len(cs)} constants, stored c mod 2^{w}: {same}, otherwise: {differ}, rejected on the "
              f"host: {rejected}; " + "; ".join(examples[:4]))
        for c in (lo - 1, hi + 1):
            x = torch.zeros(1, dtype=td, device="cuda")
            o = torch.zeros(1, dtype=td, device="cuda")
            try:
                add_kernel[(1,)](x, o, C=c)
                torch.cuda.synchronize()
                print(f"  x + {c} in {name}: compiled, stored bits {bits(o, w):#x}")
            except Exception as e:  # noqa: BLE001
                msg = str(e).splitlines()
                print(f"  x + {c} in {name}: rejected on the host ({type(e).__name__}: "
                      f"{next((m for m in msg if 'range' in m or 'unsigned' in m), msg[0])[:90].strip()})")


if __name__ == "__main__":
    main()
