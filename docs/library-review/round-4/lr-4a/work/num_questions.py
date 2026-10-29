"""lr-4a: device statements of rules/num.md ("Device questions", MathAt's docstring) on the device.

  Q1  tl.exp and tl.exp2 on float32 at -inf, +inf, NaN, and at inputs whose exact result is
      subnormal (MathAt's example exp(-inf) = +0; the flush-to-zero L4 names for exp2).
  Q2  float32 tl.atomic_add flushes subnormal operands and results ("atom.add.f32 ... flushes
      subnormal inputs and results to zero", so AtomicAddAcc's abs >= 2^-125).
  Q3  a scalar tl.atomic_add(p, 0.0) (and -0.0) with sem relaxed/acquire leaves memory as it was
      (lowered to a plain load), seen on an old value -0.0 (an IEEE add of +0 would give +0).

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/num_questions.py
"""
import struct

import torch
import triton
import triton.language as tl


def bits(v):
    return hex(struct.unpack("<I", struct.pack("<f", v))[0])


@triton.jit
def k_math(x, o1, o2, B: tl.constexpr):
    i = tl.arange(0, B)
    v = tl.load(x + i)
    tl.store(o1 + i, tl.exp(v))
    tl.store(o2 + i, tl.exp2(v))


@triton.jit
def k_atomic(p, v, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.atomic_add(p + i, tl.load(v + i))


@triton.jit
def k_atomic0(p, SEM: tl.constexpr, NEG: tl.constexpr):
    if NEG:
        tl.atomic_add(p, -0.0, sem=SEM)
    else:
        tl.atomic_add(p, 0.0, sem=SEM)


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    xs = [float("-inf"), float("inf"), float("nan"), -100.0, -126.0, -127.0, -140.0, -149.0, -150.0, -87.5, -103.0]
    x = torch.tensor(xs + [0.0] * (16 - len(xs)), dtype=torch.float32, device="cuda")
    o1, o2 = torch.empty_like(x), torch.empty_like(x)
    k_math[(1,)](x, o1, o2, B=16)
    for i, v in enumerate(xs):
        print(f"Q1 x={v!r}: tl.exp -> {o1[i].item()!r} ({bits(o1[i].item())}), tl.exp2 -> {o2[i].item()!r} ({bits(o2[i].item())})")
    # Q2: old value and operand both subnormal (2^-127 + 2^-127 = 2^-126, a normal), and
    # 0.75*2^-126 + 0.75*2^-126 = 1.5*2^-126; and a normal minus a normal giving a subnormal.
    sub = 2.0 ** -127
    cases = [(sub, sub), (0.75 * 2.0 ** -126, 0.75 * 2.0 ** -126), (1.5 * 2.0 ** -126, -2.0 ** -126), (1.0, sub), (sub, 0.0)]
    p = torch.tensor([a for a, _ in cases], dtype=torch.float32, device="cuda")
    v = torch.tensor([b for _, b in cases], dtype=torch.float32, device="cuda")
    pp = torch.zeros(8, dtype=torch.float32, device="cuda"); pp[:len(cases)] = p
    vv = torch.zeros(8, dtype=torch.float32, device="cuda"); vv[:len(cases)] = v
    k_atomic[(1,)](pp, vv, B=8)
    for i, (a, b) in enumerate(cases):
        print(f"Q2 atomic_add old {a!r} + {b!r}: memory {pp[i].item()!r} (exact sum {a + b!r})")
    for sem in ["relaxed", "acquire", "release", "acq_rel"]:
        for neg in [False, True]:
            q = torch.tensor([-0.0], dtype=torch.float32, device="cuda")
            h = k_atomic0[(1,)](q, SEM=sem, NEG=neg)
            ptx = h.asm["ptx"]
            kind = "atom/red" if ("atom." in ptx or "red." in ptx) else "no atomic instruction"
            print(f"Q3 scalar atomic_add(p, {'-0.0' if neg else '0.0'}, sem={sem}) on -0.0: memory {bits(q.item())} ({kind} in the PTX)")


if __name__ == "__main__":
    main()
