"""What Triton 3.8.0 emits on sm_90 for the approximate math functions the tutorials use
(`tl.exp`, `tl.math.exp2`, `tl.math.log2`, `tl.sqrt`), and whether an addition that consumes their
result is contracted with something inside the lowering (an `fma` in the PTX).

SEMANTICS says an addition is fusable when a *product* feeds it, "library functions included".
06's `m_i += tl.math.log2(l_i)` and the softmax sums add such results. This prints the PTX
instructions between the loads and the store of `y = m + f(l)` for each function.

Usage: CUDA_VISIBLE_DEVICES=0 ~/fpv/tva/.venv/bin/python -I math_lowering.py
"""
import re

import torch
import triton
import triton.language as tl


@triton.jit
def k_exp(m_ptr, l_ptr, y_ptr, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(y_ptr + o, tl.load(m_ptr + o) + tl.exp(tl.load(l_ptr + o)))


@triton.jit
def k_exp2(m_ptr, l_ptr, y_ptr, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(y_ptr + o, tl.load(m_ptr + o) + tl.math.exp2(tl.load(l_ptr + o)))


@triton.jit
def k_log2(m_ptr, l_ptr, y_ptr, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(y_ptr + o, tl.load(m_ptr + o) + tl.math.log2(tl.load(l_ptr + o)))


@triton.jit
def k_sqrt(m_ptr, l_ptr, y_ptr, B: tl.constexpr):
    o = tl.arange(0, B)
    tl.store(y_ptr + o, tl.load(m_ptr + o) + tl.sqrt(tl.load(l_ptr + o)))


m = torch.rand(128, device="cuda")
l = torch.rand(128, device="cuda") + 0.5
y = torch.empty_like(m)
for k in [k_exp, k_exp2, k_log2, k_sqrt]:
    c = k.warmup(m, l, y, B=128, grid=(1,))
    ptx = c.asm["ptx"]
    body = [ln.strip() for ln in ptx.splitlines()
            if re.match(r"\s*(ex2|lg2|sqrt|rsqrt|fma|mul|add|sub|neg|setp|selp|mov\.f32|call|rcp|div)", ln)]
    ops = sorted({re.split(r"\s", ln)[0] for ln in body})
    print(f"{k.fn.__name__}: {ops}")
