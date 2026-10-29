"""Print the PTX of `y = m + tl.math.log2(l)` on sm_90 (Triton 3.8.0), to see whether the outer
addition is contracted into the libdevice polynomial's last multiplication.

Usage: CUDA_VISIBLE_DEVICES=0 ~/fpv/tva/.venv/bin/python -I math_log2_ptx.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k_log2(m_ptr, l_ptr, y_ptr, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(y_ptr + o, tl.load(m_ptr + o) + tl.math.log2(tl.load(l_ptr + o)))


@triton.jit
def k_log2_alone(l_ptr, y_ptr, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(y_ptr + o, tl.math.log2(tl.load(l_ptr + o)))


m = torch.rand(32, device="cuda")
l = torch.rand(32, device="cuda") + 0.5
y = torch.empty_like(m)
for name, c in [("m + log2(l)", k_log2.warmup(m, l, y, B=32, grid=(1,), num_warps=1)),
                ("log2(l)", k_log2_alone.warmup(l, y, B=32, grid=(1,), num_warps=1))]:
    print("=====", name)
    on = False
    for ln in c.asm["ptx"].splitlines():
        s = ln.strip()
        if s.startswith("ld.global"):
            on = True
        if on and s and not s.startswith("//") and not s.startswith("$L") and not s.startswith("."):
            print("  ", s)
        if s.startswith("st.global"):
            on = False
