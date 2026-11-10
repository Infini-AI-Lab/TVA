"""L185: Triton 3.8.0's integer `x.to(d')` on the GPU against the library's `IntV.cast` (the value modulo
2^w', read as signed or unsigned by d'): every pair of int8..int64, uint8..uint64 (no int1), on boundary
values of the source. Usage: CUDA_VISIBLE_DEVICES=0 python -I int_cast_gpu.py"""
import itertools
import torch
import triton
import triton.language as tl

TY = {"i8": (True, 8), "i16": (True, 16), "i32": (True, 32), "i64": (True, 64),
      "u8": (False, 8), "u16": (False, 16), "u32": (False, 32), "u64": (False, 64)}
TL = {"i8": tl.int8, "i16": tl.int16, "i32": tl.int32, "i64": tl.int64,
      "u8": tl.uint8, "u16": tl.uint16, "u32": tl.uint32, "u64": tl.uint64}
# torch has no uint16..uint64 kernels everywhere; carry every value in a signed tensor of the same width
TT = {8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64}


@triton.jit
def cast_kernel(x_ptr, y_ptr, n, SRC: tl.constexpr, DST: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(x_ptr.to(tl.pointer_type(SRC)) + o, mask=m)
    tl.store(y_ptr.to(tl.pointer_type(DST)) + o, x.to(DST), mask=m)


def to_int(v, s, w):  # the integer a w-bit pattern means
    v %= 1 << w
    return v - (1 << w) if s and v >= 1 << (w - 1) else v


def model(a, s2, w2):  # IntV.cast_eq_ofInt: ofInt a.toInt, read in d'
    return to_int(a, s2, w2)


bad = total = 0
for src, dst in itertools.product(TY, TY):
    s, w = TY[src]
    s2, w2 = TY[dst]
    lo, hi = (-(1 << (w - 1)), (1 << (w - 1)) - 1) if s else (0, (1 << w) - 1)
    vals = sorted({lo, lo + 1, -1 if s else 1, 0, 1, 2, 127, 128, 255, 256, 32767, 32768, 65535, 65536,
                   2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**40 + 5, 2**63 - 1, hi - 1, hi, -129, -32769, -2**31 - 1})
    vals = [v for v in vals if lo <= v <= hi]
    x = torch.tensor([to_int(v, True, w) for v in vals], dtype=TT[w], device="cuda")
    y = torch.zeros(len(vals), dtype=TT[w2], device="cuda")
    cast_kernel[(1,)](x, y, len(vals), SRC=TL[src], DST=TL[dst], BLOCK=32)
    got = [to_int(int(g), s2, w2) for g in y.cpu().tolist()]
    for v, g in zip(vals, got):
        total += 1
        if g != model(v, s2, w2):
            bad += 1
            print(f"MISMATCH {src}->{dst}: {v} gives {g}, model {model(v, s2, w2)}")
print(f"triton {triton.__version__}, {torch.cuda.get_device_name(0)}: {total} values over 64 pairs, {bad} mismatches")
