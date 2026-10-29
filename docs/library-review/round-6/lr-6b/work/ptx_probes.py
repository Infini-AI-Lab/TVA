"""Library review round 6 (lr-6b), survey area `semantics`: PTX probes of SEMANTICS.md's float
statements ("Float results are exact up to NaN payload", "Float `+` and `-`: IEEE or fusable").

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-6/lr-6b/work/ptx_probes.py
Output: ptx_probes.out.txt next to this file.

Cases (compiled for the device; the PTX instructions of the arithmetic are listed):
  R1  `tl.fma` on bfloat16 and float16: one rounding needs a native `fma.rn.bf16`/`fma.rn.f16`; an fma in
      float32 followed by a conversion would round twice, which `tl.fma` (one rounding) does not allow.
  R2  `(a * b).to(tl.float32) + c` with float16 `a`, `b`: is the widened product fused into a float32
      `fma`? `AddRel` (for the float32 addition) does not cover that result, since the float16-rounded
      product is not the float32 rounding of the exact product.
  R3  bfloat16 `+`, `-`, `*` and float16 `a * b + c` (fusion, covered by `tl.mul_add`).
"""
import os
import re
import tempfile

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="lr6b-ptx-")

import torch
import triton
import triton.language as tl

print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")


def ptx_ops(kernel, pat=r"\b(fma|add|sub|mul|cvt)\.[a-z0-9.]+"):
    cache = kernel.device_caches[0][0]
    ptx = list(cache.values())[-1].asm["ptx"]
    body = [l.strip() for l in ptx.splitlines()]
    ops = []
    for l in body:
        m = re.match(r"^(fma|add|sub|mul|cvt)\.([a-z0-9.]+)", l)
        if m and ("f16" in m.group(2) or "f32" in m.group(2) or "bf16" in m.group(2)):
            ops.append(m.group(1) + "." + m.group(2))
    return sorted(set(ops))


@triton.jit
def r1_kernel(a_ptr, b_ptr, c_ptr, o_ptr):
    offs = tl.arange(0, 16)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    c = tl.load(c_ptr + offs)
    tl.store(o_ptr + offs, tl.fma(a, b, c))


@triton.jit
def r2_kernel(a_ptr, b_ptr, c_ptr, o_ptr):
    offs = tl.arange(0, 16)
    a = tl.load(a_ptr + offs)  # float16
    b = tl.load(b_ptr + offs)  # float16
    c = tl.load(c_ptr + offs)  # float32
    tl.store(o_ptr + offs, (a * b).to(tl.float32) + c)


@triton.jit
def r3_kernel(a_ptr, b_ptr, c_ptr, o_ptr):
    offs = tl.arange(0, 16)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    c = tl.load(c_ptr + offs)
    tl.store(o_ptr + offs, a + b)
    tl.store(o_ptr + 16 + offs, a - b)
    tl.store(o_ptr + 32 + offs, a * b)
    tl.store(o_ptr + 48 + offs, a * b + c)


def run(kernel, dts, odt, n=16):
    args = [torch.ones(64, dtype=d, device="cuda") for d in dts]
    out = torch.zeros(64, dtype=odt, device="cuda")
    kernel[(1,)](*args, out)
    torch.cuda.synchronize()
    return ptx_ops(kernel)


print("R1 tl.fma bfloat16:", run(r1_kernel, [torch.bfloat16] * 3, torch.bfloat16))
r1_kernel.device_caches.clear() if hasattr(r1_kernel.device_caches, "clear") else None
print("R1 tl.fma float16:", run(r1_kernel, [torch.float16] * 3, torch.float16))
print("R2 (f16 a * f16 b).to(f32) + f32 c:", run(r2_kernel, [torch.float16, torch.float16, torch.float32], torch.float32))
print("R3 bfloat16 +, -, *, a*b+c:", run(r3_kernel, [torch.bfloat16] * 3, torch.bfloat16))
print("R3 float16 +, -, *, a*b+c:", run(r3_kernel, [torch.float16] * 3, torch.float16))
