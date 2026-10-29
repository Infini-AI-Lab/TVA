"""lr-14a: what an fp8 constant built from a Python number gives on the device (L64's
"a constant converted into fp8 is folded only when exact, so the CvtSat facts cover it").

For each Python number c and fp8 dtype: (a) tl.full([16], c, dtype); (b) the same value as a
float32 tensor loaded from memory, then .to(dtype) (the device conversion, tl.cvt8); (c) c as a
float32 constant tensor then .to(dtype). Prints the result bits and the conversion instructions
of (a). (d) tl.load(fp8 pointer, mask all false, other=c). A cell "error" is a compile error on the host.
Run: CUDA_VISIBLE_DEVICES=0 python -I fp8_const.py
"""
import re

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def k_full(out_ptr, C: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    v = tl.full([16], C, D)
    tl.store(out_ptr + offs, v.to(tl.int8, bitcast=True))


@triton.jit
def k_load(x_ptr, out_ptr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x.to(D).to(tl.int8, bitcast=True))


@triton.jit
def k_f32const(out_ptr, C: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    v = tl.full([16], C, tl.float32)
    tl.store(out_ptr + offs, v.to(D).to(tl.int8, bitcast=True))


@triton.jit
def k_where(x_ptr, out_ptr, C: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs).to(D)
    v = tl.where(offs < 0, x, C)
    tl.store(out_ptr + offs, v.to(tl.int8, bitcast=True))


@triton.jit
def k_other(p8_ptr, out_ptr, C: tl.constexpr):
    offs = tl.arange(0, 16)
    v = tl.load(p8_ptr + offs, mask=offs < 0, other=C)
    tl.store(out_ptr + offs, v.to(tl.int8, bitcast=True))


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    consts = [float("inf"), float("-inf"), float("nan"), 1e6, -1e6, 1000.0, 470.0, 448.0, 460.0,
              65536.0, 61440.0, 60000.0, 57344.0, 0.1, 1.0, 3.0e38, 1e39]
    dts = [("float8e5", tl.float8e5), ("float8e4nv", tl.float8e4nv)]
    print("| c | dtype | tl.full(c) | f32 loaded .to | f32 const .to | tl.where(.., c) | load other=c | ops in tl.full PTX |")
    print("|---|---|---|---|---|---|---|---|")
    for c in consts:
        for name, d in dts:
            o = torch.empty(16, dtype=torch.int8, device="cuda")
            row = [repr(c), name]
            try:
                h = k_full[(1,)](o, C=c, D=d)
                torch.cuda.synchronize()
                row.append(f"`{int(o[0]) & 0xff:02x}`")
                ops = sorted(set(re.findall(r"\b(?:cvt|mov\.b16|mov\.u16)[a-z0-9.]*", h.asm["ptx"])))
            except Exception as e:  # noqa: BLE001
                row.append("error: " + type(e).__name__)
                ops = []
            with np.errstate(over="ignore"):
                c32 = float(np.float32(c))  # the float32 nearest to c (1e39 -> inf)
            x = torch.full((16,), c32, dtype=torch.float32, device="cuda")
            k_load[(1,)](x, o, D=d); torch.cuda.synchronize()
            row.append(f"`{int(o[0]) & 0xff:02x}`")
            try:
                k_f32const[(1,)](o, C=c, D=d); torch.cuda.synchronize()
                row.append(f"`{int(o[0]) & 0xff:02x}`")
            except Exception as e:  # noqa: BLE001
                row.append("error: " + type(e).__name__)
            try:
                k_where[(1,)](x, o, C=c, D=d); torch.cuda.synchronize()
                row.append(f"`{int(o[0]) & 0xff:02x}`")
            except Exception as e:  # noqa: BLE001
                row.append("error: " + type(e).__name__)
            tdt = torch.float8_e5m2 if name == "float8e5" else torch.float8_e4m3fn
            p8 = torch.zeros(16, dtype=tdt, device="cuda")
            try:
                k_other[(1,)](p8, o, C=c); torch.cuda.synchronize()
                row.append(f"`{int(o[0]) & 0xff:02x}`")
            except Exception as e:  # noqa: BLE001
                row.append("error: " + type(e).__name__)
            row.append(", ".join(f"`{s}`" for s in ops) or "none")
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
