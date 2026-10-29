"""lr-14a: does a float32 pointer tl.atomic_add give the same new values for every sem and scope?
AtomicAddF32 says "any sem"; evidence/num-atomic-add runs acq_rel and relaxed at scope gpu. Each
lane adds its own operand to its own cell (2^22 lanes: random patterns, subnormals, near
cancellations), for sem in {acq_rel, acquire, release, relaxed} and scope in {gpu, cta, sys};
the new cells are compared bit for bit with acq_rel/gpu. Prints the atom instructions.
Run: CUDA_VISIBLE_DEVICES=0 python -I atomic_sem.py
"""
import re

import torch
import triton
import triton.language as tl


@triton.jit
def k(c_ptr, v_ptr, n, SEM: tl.constexpr, SCOPE: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    tl.atomic_add(c_ptr + offs, tl.load(v_ptr + offs, mask=m), mask=m, sem=SEM, scope=SCOPE)


def inputs(n, g):
    o = torch.randint(-(1 << 31), 1 << 31, (n,), generator=g, dtype=torch.int64).to(torch.int32).view(torch.float32)
    v = torch.randint(-(1 << 31), 1 << 31, (n,), generator=g, dtype=torch.int64).to(torch.int32).view(torch.float32)
    q = n // 4
    sub = torch.randint(-(1 << 23), 1 << 23, (2, q), generator=g, dtype=torch.int64).to(torch.int32).view(torch.float32)
    o[:q], v[:q] = sub[0], sub[1]                                  # subnormal operands
    o[q:2 * q] = torch.randn(q, generator=g)
    v[q:2 * q] = -o[q:2 * q] * (1 + torch.randint(-8, 9, (q,), generator=g) * 2.0 ** -23)  # cancellation
    return o.cuda(), v.cuda()


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    n = 1 << 22
    o, v = inputs(n, torch.Generator().manual_seed(1414))
    ref = None
    print("| sem | scope | instruction | cells differing from acq_rel/gpu |")
    print("|---|---|---|---|")
    for sem in ("acq_rel", "acquire", "release", "relaxed"):
        for scope in ("gpu", "cta", "sys"):
            c = o.clone()
            h = k[(n // 1024,)](c, v, n, SEM=sem, SCOPE=scope, B=1024)
            torch.cuda.synchronize()
            bits = c.view(torch.int32)
            if ref is None:
                ref = bits.clone()
            diff = int((bits != ref).sum())
            ins = sorted(set(re.findall(r"\batom\.[a-z0-9_.:]+", h.asm["ptx"])))
            print(f"| {sem} | {scope} | {', '.join(f'`{i}`' for i in ins)} | {diff} |")


if __name__ == "__main__":
    main()
