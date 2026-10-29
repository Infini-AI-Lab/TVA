"""lr-10a, round 10 (L50): does Triton 3.8.0 run a helper made by calling `triton.jit(f)`, and a kernel under an
aliased decorator? Neither is a decorator that names `jit`, so the gate's `jit_functions` finds neither
(work/gate_probes.out, G5 and G6). Each kernel stores what it computes into one int32; the expected value
shows which code ran.

Usage: CUDA_VISIBLE_DEVICES=0 <TVA's .venv python> -I work/jit_spellings.py
"""
import torch
import triton
import triton.language as tl
from triton import jit as J


def _h(x):
    return x * 10 + 7


h = triton.jit(_h)  # a JITFunction made by a call, as a helper


@triton.jit
def k_calls_h(p):
    tl.store(p, h(tl.load(p)))


@J  # an aliased decorator
def k_alias(p):
    tl.store(p, tl.load(p) + 100)


def _k(p):
    tl.store(p, tl.load(p) * 3)


k_call = triton.jit(_k)  # the kernel itself made by a call


if __name__ == "__main__":
    print(f"triton {triton.__version__}, {torch.cuda.get_device_name(0)}")
    for name, kern, want in (("k_calls_h (helper h = triton.jit(_h))", k_calls_h, 5 * 10 + 7),
                             ("k_alias (@J, J = triton.jit)", k_alias, 105),
                             ("k_call = triton.jit(_k)", k_call, 15)):
        x = torch.full((1,), 5, dtype=torch.int32, device="cuda")
        kern[(1,)](x)
        torch.cuda.synchronize()
        print(f"- {name}: type {type(kern).__name__}; stored {int(x.item())} (expected {want})")
