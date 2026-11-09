"""lr-34b: which PTX does each one-product / mixed form give (L155)?  Prints the arithmetic lines."""
import re
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def k(a_ptr, b_ptr, c_ptr, m_ptr, o_ptr, FORM: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    a = tl.load(a_ptr + j)
    b = tl.load(b_ptr + j)
    c = tl.load(c_ptr + j)
    m = tl.load(m_ptr + j) != 0
    if FORM == 0:
        r = a * b + c
    elif FORM == 1:
        r = c + a * b
    elif FORM == 2:
        r = a * b - c
    elif FORM == 3:
        r = c - a * b
    elif FORM == 4:
        r = -(a * b) + c
    elif FORM == 5:
        r = tl.where(m, a * b, 0.0) + c
    elif FORM == 6:
        r = tl.where(m, a * b, c) + c
    elif FORM == 7:  # product in a,b's dtype, widened, added in c's dtype
        r = (a * b).to(c.dtype) + c
    elif FORM == 8:
        r = a / b + c
    elif FORM == 9:
        r = tl.maximum(a * b, 0.0) + c
    elif FORM == 10:
        r = (a * b).to(c.dtype) - c
    elif FORM == 11:
        r = c - (a * b).to(c.dtype)
    tl.store(o_ptr + j, r)


def ptx_of(form, dab, dc):
    N = 256
    a = torch.ones(N, dtype=dab, device="cuda")
    c = torch.ones(N, dtype=dc, device="cuda")
    m = torch.ones(N, dtype=torch.int32, device="cuda")
    o = torch.empty(N, dtype=dc if form in (7, 10, 11) else dab, device="cuda")
    h = k[(1,)](a, a, c, m, o, form, N)
    ptx = h.asm["ptx"].split(".entry", 1)[1]
    ops = sorted({x for x in re.findall(r"\b((?:add|sub|fma|mul|neg|cvt|div|rcp|selp)\.[a-z0-9.]+)", ptx)})
    return ops


if __name__ == "__main__":
    for form in range(12):
        combos = [(torch.float32, torch.float32), (torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)]
        if form in (7, 10, 11):
            combos = [(torch.float16, torch.float32), (torch.bfloat16, torch.float32), (torch.float32, torch.float64),
                      (torch.float16, torch.float64)]
        for dab, dc in combos:
            try:
                print(form, str(dab)[6:], str(dc)[6:], ptx_of(form, dab, dc), flush=True)
            except Exception as e:  # noqa: BLE001
                print(form, str(dab)[6:], str(dc)[6:], "ERR", type(e).__name__, str(e)[:200])
