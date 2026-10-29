"""lr-14a: a Python number stored through a float8e5 pointer (a stored value: first a float32
tensor, then converted), and a float32 tensor stored through it (the implicit conversion of
tl.store), against tl.full in float8e5 (built directly in the dtype).
Run: CUDA_VISIBLE_DEVICES=0 python -I fp8_store.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_store_const(p_ptr, C: tl.constexpr):
    offs = tl.arange(0, 16)
    tl.store(p_ptr + offs, C)


@triton.jit
def k_store_f32(x_ptr, p_ptr):
    offs = tl.arange(0, 16)
    tl.store(p_ptr + offs, tl.load(x_ptr + offs))


@triton.jit
def k_full(p_ptr, C: tl.constexpr):
    offs = tl.arange(0, 16)
    tl.store(p_ptr + offs, tl.full([16], C, tl.float8e5))


print(f"Triton {triton.__version__}, {torch.cuda.get_device_name(0)}")
print("| c | tl.store(p8, c) | tl.store(p8, float32 tensor c) | tl.store(p8, tl.full([16], c, tl.float8e5)) |")
print("|---|---|---|---|")
for c in [float("inf"), float("-inf"), 1e39, -1e39, 1e6, 57344.0]:
    row = [repr(c)]
    p = torch.zeros(16, dtype=torch.float8_e5m2, device="cuda")
    for f in ("const", "f32", "full"):
        try:
            if f == "const":
                k_store_const[(1,)](p, C=c)
            elif f == "f32":
                x = torch.full((16,), c, dtype=torch.float64, device="cuda").to(torch.float32)
                k_store_f32[(1,)](x, p)
            else:
                k_full[(1,)](p, C=c)
            torch.cuda.synchronize()
            row.append(f"`{int(p.view(torch.uint8)[0]):02x}`")
        except Exception as e:  # noqa: BLE001
            row.append(f"rejected: {type(e).__name__}")
    print("| " + " | ".join(row) + " |")
