"""TVA library evidence (library issue L16): the result types and values of integer reductions.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/int-reductions/run.py
Output: evidence/int-reductions/out.txt (README.md says what it shows).

For each integer dtype (int8, int16, uint8, uint16, int1, int32, uint32) and each reduction
(`tl.max(x)`, `tl.min(x)`, `tl.sum(x)`, `tl.max(x, 0, return_indices=True)`, `tl.argmax(x, 0)`), one
program reduces one block. The script reads the result's element type from the TTIR (the type of the
`tt.reduce` result; MLIR integer types carry no sign) and stores the result widened to `int64` with
`.to(tl.int64)`, which sign-extends a signed and zero-extends an unsigned result, so the stored value
shows the result's signedness. It also stores `r + 1000` into an `int32` buffer: a result still in 8
or 16 bits would not hold it (for the value of `return_indices=True`, which keeps the input's type,
it stores `r.to(tl.int32) + 1000`, and part C shows that `r + 1000` is rejected).

Inputs: every value of the 8-bit types (256) and of the 16-bit types (65536), in one block each,
shuffled with a fixed seed; for `int1`, 256 booleans; for the 32-bit types, 65536 values that include
the extremes, 0, -1 and 1, and random values. A second part sums blocks of 2^17 large values, so
that the sum leaves 32 bits and shows how it wraps.
"""
import re
import time

import numpy as np
import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def red(x_ptr, o64_ptr, o32_ptr, N: tl.constexpr, OP: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, N))
    if OP == 0:
        r = tl.max(x)
    elif OP == 1:
        r = tl.min(x)
    elif OP == 2:
        r = tl.sum(x)
    elif OP == 3:
        r, idx = tl.max(x, 0, return_indices=True)
        tl.store(o64_ptr + 2, idx.to(tl.int64))
    elif OP == 4:
        r = tl.argmax(x, 0)
    else:
        r, idx = tl.max(x, 0, return_indices=True)
    tl.store(o64_ptr, r.to(tl.int64))
    if OP == 3:
        # the value keeps the input's type: `r + 1000` does not compile for 8-bit inputs (part C)
        tl.store(o32_ptr, r.to(tl.int32) + 1000)
    else:
        tl.store(o32_ptr, (r + 1000).to(tl.int32))


OPS = ["tl.max(x)", "tl.min(x)", "tl.sum(x)", "tl.max(x, 0, return_indices=True)", "tl.argmax(x, 0)",
       "tl.max(x, 0, return_indices=True)[0] + 1000"]


def reduce_types(ttir):
    """The result types of the `tt.reduce` operations of a TTIR."""
    return re.findall(r"\}\) : \([^)]*\) -> (\(?[^\n]*?\)?)(?:\s+loc\(.*\))?$", ttir, re.M)


def values(name, rng):
    if name == "int8":
        v = np.arange(-128, 128, dtype=np.int64)
    elif name == "uint8":
        v = np.arange(0, 256, dtype=np.int64)
    elif name == "int16":
        v = np.arange(-32768, 32768, dtype=np.int64)
    elif name == "uint16":
        v = np.arange(0, 65536, dtype=np.int64)
    elif name == "int1":
        v = np.array([0, 1] * 128, dtype=np.int64)
    elif name == "int32":
        v = rng.integers(-2**31, 2**31, size=65536, dtype=np.int64)
        v[:5] = [-2**31, 2**31 - 1, 0, -1, 1]
    else:  # uint32
        v = rng.integers(0, 2**32, size=65536, dtype=np.int64)
        v[:4] = [0, 2**32 - 1, 1, 2**31]
    return rng.permutation(v)


def to_torch(name, v):
    if name == "int1":
        return torch.tensor(v.astype(bool), device=DEV)
    if name in ("int8", "uint8", "int16", "int32"):
        return torch.tensor(v, dtype={"int8": torch.int8, "uint8": torch.uint8, "int16": torch.int16,
                                      "int32": torch.int32}[name], device=DEV)
    if name == "uint16":
        return torch.tensor(v.astype(np.uint16).view(np.int16), dtype=torch.int16, device=DEV).view(torch.uint16)
    return torch.tensor(v.astype(np.uint32).view(np.int32), dtype=torch.int32, device=DEV).view(torch.uint32)


def wrap(x, bits, signed):
    x %= 2**bits
    return x - 2**bits if signed and x >= 2**(bits - 1) else x


def expect(name, op, v):
    """What the library states: max/min/argmax on the exact values (sign- or zero-extended), the
    sum in int32 (signed inputs narrower than 32 bits) or uint32 (unsigned ones, int1 included)."""
    signed = name in ("int8", "int16", "int32")
    if op in (0, 3):
        return int(v.max())
    if op == 1:
        return int(v.min())
    if op == 2:
        return wrap(int(v.sum()), 32, signed)
    return int(np.flatnonzero(v == v.max())[0])  # argmax, tie_break_left: the first maximum


def run(name, op, v):
    x = to_torch(name, v)
    o64 = torch.zeros(3, dtype=torch.int64, device=DEV)
    o32 = torch.zeros(1, dtype=torch.int32, device=DEV)
    k = red[(1, )](x, o64, o32, len(v), op, num_warps=8)
    torch.cuda.synchronize()
    return reduce_types(k.asm["ttir"]), int(o64[0].item()), int(o32[0].item()), int(o64[2].item())


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}")
    rng = np.random.default_rng(20261027)
    print("\n== A. Every value of each small type in one block (32-bit types: 65536 adversarial values) ==")
    print("dtype   | reduction                            | TTIR result type | value as int64 | r + 1000 (int32) "
          "| expected | ok")
    bad = 0
    for name in ("int8", "int16", "uint8", "uint16", "int1", "int32", "uint32"):
        v = values(name, rng)
        for op in range(5):
            ty, r64, r32, idx = run(name, op, v)
            want = expect(name, op, v)
            ok = r64 == want and r32 == wrap(want + 1000, 32, True)
            if op == 3:
                ok = ok and 0 <= idx < len(v) and int(v[idx]) == want
            bad += not ok
            print(f"{name:7} | {OPS[op]:36} | {', '.join(ty):16} | {r64:14} | {r32:16} | {want:8} | {ok}")
    print(f"A: {bad} result(s) differ from the library's statement")

    print("\n== B. Sums that leave 32 bits: 2^17 copies of a large value ==")
    bad = 0
    for name, val in (("int8", 127), ("int8", -128), ("int16", 32767), ("int16", -32768), ("uint8", 255),
                      ("uint16", 65535), ("int32", 2**31 - 1), ("uint32", 2**32 - 1)):
        v = np.full(2**17, val, dtype=np.int64)
        ty, r64, _, _ = run(name, 2, v)
        signed = name in ("int8", "int16", "int32")
        want = wrap(val * 2**17, 32, signed)
        ok = r64 == want
        bad += not ok
        print(f"{name:7} x {val:11} * 2^17: TTIR {', '.join(ty)}; sum as int64 {r64:12}; "
              f"{'int32' if signed else 'uint32'} wrap of the exact sum {want:12}: {ok}")
    print(f"B: {bad} result(s) differ")

    print("\n== C. `r + 1000` on the value of tl.max(x, 0, return_indices=True) ==")
    for name in ("int8", "uint8", "int16"):
        v = values(name, rng)
        try:
            ty, r64, r32, _ = run(name, 5, v)
            res = f"compiles; TTIR {', '.join(ty)}; r + 1000 = {r32}"
        except Exception as e:  # noqa: BLE001
            lines = [l.strip() for l in str(e).strip().splitlines() if l.strip()]
            res = f"rejected on the host: {type(e).__name__}: {lines[-1][:120] if lines else ''}"
        print(f"{name:7}: {res}")


if __name__ == "__main__":
    main()
