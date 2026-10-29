"""lr-4b, library review round 4, L25 and [pure]: an operation that [pure] would transcribe as a
model line, inside a condition or an argument, that the front end never evaluates.

[config] (round 4) says: when the condition is not pure, "its operations are model lines, as
[pure] says", and [pure] says the argument expressions "are transcribed as model lines, in
Python's order (left to right), as if each stood alone". But the front end short-circuits `and`
and `or` at compile time when a non-tensor value decides (`visit_BoolOp`,
code-generator.py:1481-1501), and evaluates only one side of a conditional expression whose
condition is a compile-time value (`visit_IfExp`). So some operations of an impure expression never
run. Grid of 4 programs; the counter shows how many atomics ran.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I <this file>
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k1(cnt_ptr, o_ptr, FLAG: tl.constexpr):
    # [config] on an impure condition: `FLAG and <atomic> is not None`, FLAG = False
    if FLAG and tl.atomic_add(cnt_ptr, 1) is not None:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def k2(cnt_ptr, o_ptr):
    # two atomics in an impure condition; the first decides at compile time (`is None` is False)
    if (tl.atomic_add(cnt_ptr, 1) is None) and (tl.atomic_add(cnt_ptr, 1) is None):
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


@triton.jit
def k3(cnt_ptr, o_ptr, FLAG: tl.constexpr):
    # [compile-check] on an impure condition: `FLAG or <atomic> is None`, FLAG = True
    tl.static_assert(FLAG or tl.atomic_add(cnt_ptr, 1) is None)
    tl.store(o_ptr + tl.program_id(0), 1)


@triton.jit
def k4(cnt_ptr, o_ptr, FLAG: tl.constexpr):
    # [no-op:print] on an impure argument: a conditional expression with a compile-time condition
    tl.static_print(tl.atomic_add(cnt_ptr, 1) if FLAG else 0)
    tl.store(o_ptr + tl.program_id(0), 1)


@triton.jit
def k5(cnt_ptr, o_ptr, FLAG: tl.constexpr):
    # [config] on a conditional expression inside the condition
    if (tl.atomic_add(cnt_ptr, 1) if FLAG else None) is None:
        tl.store(o_ptr + tl.program_id(0), 1)
    else:
        tl.store(o_ptr + tl.program_id(0), 2)


def run(name, k, kw, as_pure):
    cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    o = torch.zeros(4, dtype=torch.int32, device="cuda")
    h = k[(4,)](cnt, o, **kw)
    torch.cuda.synchronize()
    ttir = h.asm["ttir"]
    print(f"{name}: counter {cnt.item()} (transcribing every operation, as [pure] says: {as_pure}); "
          f"outputs {o.tolist()}; tt.atomic_rmw in TTIR: {ttir.count('tt.atomic_rmw')}; "
          f"scf.if in TTIR: {ttir.count('scf.if')}")


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    run("K1 if FLAG and <atomic> is not None (FLAG=False)", k1, {"FLAG": False}, 4)
    run("K2 if (<atomic> is None) and (<atomic> is None)", k2, {}, 8)
    run("K3 tl.static_assert(FLAG or <atomic> is None) (FLAG=True)", k3, {"FLAG": True}, 4)
    run("K4 tl.static_print(<atomic> if FLAG else 0) (FLAG=False)", k4, {"FLAG": False}, 4)
    run("K5 if (<atomic> if FLAG else None) is None (FLAG=False)", k5, {"FLAG": False}, 4)


if __name__ == "__main__":
    main()
