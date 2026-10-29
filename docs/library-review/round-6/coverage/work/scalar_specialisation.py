"""How Triton 3.8.0 passes scalar arguments (library issue L29): the specialization of an int
equal to 1, an int at or above 2^31, a float and a bool, as the JIT's binder computes it.

Usage: CUDA_VISIBLE_DEVICES=0 ~/fpv/tva/.venv/bin/python -I scalar_specialisation.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k(out, n, x):
    tl.store(out, n)


out = torch.empty(1, device="cuda", dtype=torch.int64)
for n in [1, 7, 16, 2**31, -1]:
    for x in [0.5, True]:
        c = k.warmup(out, n, x, grid=(1,))
        print(f"n={n!r:>12} x={x!r:>5}: signature={dict(c.src.signature)} constants={dict(c.src.constants)} "
              f"attrs={c.src.attrs}")
