"""lr-4a, L25: `[config]` (and `[pure]`) say that the operations of a condition that is not pure
are model lines. Does every operation of such a condition run? Python's `and`/`or` with a
compile-time operand that decides the result, and a conditional expression with a compile-time
condition, leave the rest unevaluated in Triton's front end, so their operations do not run.

Each kernel runs 4 programs; `c` counts the atomics that ran (4 = every program ran it).

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/short_circuit.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k1(c, o, FLAG: tl.constexpr):
    # [config]: `if FLAG and <atomic> is not None` with FLAG False
    if FLAG and tl.atomic_add(c, 1) is not None:
        v = 1
    else:
        v = 2
    tl.store(o + tl.program_id(0), v)


@triton.jit
def k2(c, o, FLAG: tl.constexpr):
    # [config]: `if FLAG or <atomic> is None` with FLAG True
    if FLAG or tl.atomic_add(c, 1) is None:
        v = 1
    else:
        v = 2
    tl.store(o + tl.program_id(0), v)


@triton.jit
def k3(c, o, FLAG: tl.constexpr):
    # [config] with an elif whose condition is impure, after a taken `if`: never visited
    if FLAG:
        v = 1
    elif tl.atomic_add(c, 1) is not None:
        v = 2
    else:
        v = 3
    tl.store(o + tl.program_id(0), v)


@triton.jit
def k4(c, o, FLAG: tl.constexpr):
    # [pure] (not pure argument of tl.static_print): `FLAG and <atomic>` with FLAG False
    tl.static_print(FLAG and tl.atomic_add(c, 1))
    tl.store(o + tl.program_id(0), 1)


@triton.jit
def k5(c, o, FLAG: tl.constexpr):
    # [pure] (not pure argument of tl.device_print): a conditional expression with a constexpr condition
    tl.device_print("v", tl.atomic_add(c, 1) if FLAG else 0)
    tl.store(o + tl.program_id(0), 1)


@triton.jit
def k6(c, o, FLAG: tl.constexpr):
    # control: the atomic first, then the constexpr
    if tl.atomic_add(c, 1) is not None and FLAG:
        v = 1
    else:
        v = 2
    tl.store(o + tl.program_id(0), v)


CASES = [
    ("K1 if FLAG and <atomic> is not None (FLAG=False)", k1, False),
    ("K2 if FLAG or <atomic> is None (FLAG=True)", k2, True),
    ("K3 if FLAG: ... elif <atomic> is not None: (FLAG=True)", k3, True),
    ("K4 tl.static_print(FLAG and <atomic>) (FLAG=False)", k4, False),
    ("K5 tl.device_print('v', <atomic> if FLAG else 0) (FLAG=False)", k5, False),
    ("K6 control: if <atomic> is not None and FLAG (FLAG=False)", k6, False),
]


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    for name, k, flag in CASES:
        c = torch.zeros(1, dtype=torch.int32, device="cuda")
        o = torch.zeros(4, dtype=torch.int32, device="cuda")
        h = k[(4,)](c, o, FLAG=flag)
        torch.cuda.synchronize()
        n_at = h.asm["ttir"].count("tt.atomic_rmw")
        print(f"{name}: atomics that ran {c.item()} of 4; outputs {o.tolist()}; tt.atomic_rmw in TTIR: {n_at}")


if __name__ == "__main__":
    main()
