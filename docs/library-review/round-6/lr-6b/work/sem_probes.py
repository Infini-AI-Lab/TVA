"""Library review round 6 (lr-6b), survey area `semantics`: device and compiler probes of
SEMANTICS.md statements that the batch's evidence does not settle.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-6/lr-6b/work/sem_probes.py
Output: sem_probes.out.txt next to this file (what each case shows is printed with it).

Cases:
  Q1  `/` with two integer operands: SEMANTICS.md does not say what it computes. Triton converts both
      operands to float32 (`semantic.truediv`), also int64 ones, so `(2**40 + 1) / 1` is 2**40 and
      `16777217 / 1` is 16777216, unlike Python's float64 division.
  Q2  `-0.0` as a "tensor of its own" (`other=`, a stored scalar, an argument of `tl.fma`): SEMANTICS.md
      gives `FloatV.litF32`, which is `+0` for a zero. The `tl.fma(-1, +0, c)` case tells `+0` from `-0`.
  Q3  The order in which the front end evaluates a call's arguments: keyword arguments first
      (`visit_Call`), not Python's left-to-right order. Shown by the order of two loads in the TTIR.
  Q4  `x if c else y` with a tensor condition: only the taken side runs on the device (an `scf.if`),
      not both (SEMANTICS.md: "both sides run inside a runtime branch").
  Q5  `and` on two `int32` scalar tensors: rejected on the host (`logical_and` bitcasts to `int1`);
      on two `int1` tensors it is the logical and. SEMANTICS.md says only which operands are
      evaluated, not what `and`/`or`/`not` compute.
  Q6  The dtype an operator with a Python float computes in: `x_i32 * 1e-40` in float64 (the
      number's own type), `x_f32 * 1e-40` in float32 with a subnormal constant, `x_f16 * 1e-40` with
      the constant `+0`.
  Q7  Integer `//` and `%` on negative operands truncate (C semantics).
  Q8  A `while` whose condition holds an atomic, and a `for` whose body holds one: the front end's
      dry run of the body (to find the carried names) leaves no extra atomic on the device; the
      condition runs once per test, the last (false) one included.
"""
import os
import sys
import struct
import tempfile

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="lr6b-sem-")

import torch
import triton
import triton.language as tl

print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")


def bits32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def ttir(kernel):
    """The TTIR of the last compilation of `kernel` (first device)."""
    cache = kernel.device_caches[0][0]
    return list(cache.values())[-1].asm["ttir"]


# ---------------------------------------------------------------- Q1
@triton.jit
def q1_kernel(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr):
    a = tl.load(a_ptr)  # int64
    b = tl.load(b_ptr)  # int64
    c = tl.load(c_ptr)  # int32
    d = tl.load(d_ptr)  # int32
    tl.store(out_ptr + 0, a / b)
    tl.store(out_ptr + 1, c / d)
    tl.store(out_ptr + 2, c / 3)


def q1():
    a = torch.tensor([2**40 + 1], dtype=torch.int64, device="cuda")
    b = torch.tensor([1], dtype=torch.int64, device="cuda")
    c = torch.tensor([16777217], dtype=torch.int32, device="cuda")
    d = torch.tensor([1], dtype=torch.int32, device="cuda")
    out = torch.zeros(3, dtype=torch.float64, device="cuda")
    q1_kernel[(1,)](a, b, c, d, out)
    torch.cuda.synchronize()
    ir = ttir(q1_kernel)
    divs = [l.strip() for l in ir.splitlines() if "arith.divf" in l]
    print("Q1 int / int")
    print(f"  (2**40+1) / 1 (int64): {out[0].item()!r}  Python: {(2**40 + 1) / 1!r}")
    print(f"  16777217 / 1 (int32): {out[1].item()!r}  Python: {16777217 / 1!r}")
    print(f"  16777217 / 3 (int32 / Python int): {out[2].item()!r}  float32 of Python's: "
          f"{struct.unpack('<f', struct.pack('<f', 16777217 / 3))[0]!r}")
    print(f"  TTIR divisions: {[('f32' if ': f32' in l else 'f64' if ': f64' in l else '?') for l in divs]}")


# ---------------------------------------------------------------- Q2
@triton.jit
def q2_kernel(x_ptr, y_ptr, out_ptr):
    offs = tl.arange(0, 2)
    x = tl.load(x_ptr + offs)  # [-1.0, -1.0]
    y = tl.load(y_ptr + offs)  # [0.0, 0.0]
    tl.store(out_ptr + 0 + tl.arange(0, 1), tl.load(x_ptr + tl.arange(0, 1), mask=tl.arange(0, 1) < 0, other=-0.0))
    tl.store(out_ptr + 1, -0.0)
    f = tl.fma(x, y, -0.0)  # (-1)*(+0) + c: -0 iff c is -0
    tl.store(out_ptr + 2 + offs, f)
    tl.store(out_ptr + 4 + tl.arange(0, 1), tl.full([1], -0.0, tl.float32))


def q2():
    x = torch.tensor([-1.0, -1.0], dtype=torch.float32, device="cuda")
    y = torch.tensor([0.0, 0.0], dtype=torch.float32, device="cuda")
    out = torch.full((5,), float("nan"), dtype=torch.float32, device="cuda")
    q2_kernel[(1,)](x, y, out)
    torch.cuda.synchronize()
    v = out.tolist()
    print("Q2 -0.0 as a tensor of its own")
    print(f"  other=-0.0: {hex(bits32(v[0]))}; stored -0.0: {hex(bits32(v[1]))}; "
          f"tl.fma(-1, +0, -0.0): {hex(bits32(v[2]))}; tl.full(-0.0): {hex(bits32(v[4]))}  (0x0 = +0, 0x80000000 = -0)")


# ---------------------------------------------------------------- Q3
@triton.jit
def q3_kernel(a_ptr, m_ptr, out_ptr):
    tl.store(out_ptr, tl.load(a_ptr), mask=tl.load(m_ptr) > 0)


def q3():
    a = torch.ones(1, dtype=torch.float32, device="cuda")
    m = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.float32, device="cuda")
    q3_kernel[(1,)](a, m, out)
    torch.cuda.synchronize()
    ir = ttir(q3_kernel)
    loads = [l.strip() for l in ir.splitlines() if "tt.load" in l]
    order = ["m (keyword)" if ": !tt.ptr<i32>" in l else "a (positional)" for l in loads]
    print("Q3 the order of a call's arguments")
    print(f"  source: tl.store(out_ptr, tl.load(a_ptr), mask=tl.load(m_ptr) > 0); TTIR load order: {order}")


# ---------------------------------------------------------------- Q4
@triton.jit
def q4_kernel(t_ptr, cnt_ptr, out_ptr):
    t = tl.load(t_ptr)  # 1: a runtime (tensor) condition
    v = tl.atomic_add(cnt_ptr, 1) if t > 0 else tl.atomic_add(cnt_ptr, 10)
    tl.store(out_ptr + tl.program_id(0), v)


def q4():
    t = torch.ones(1, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    q4_kernel[(4,)](t, cnt, out)
    torch.cuda.synchronize()
    ir = ttir(q4_kernel)
    print("Q4 a conditional expression with a tensor condition, 4 programs")
    print(f"  counter {cnt.item()} (4 = only the taken side runs, 44 = both); scf.if in TTIR: {'scf.if' in ir}; "
          f"atomics in TTIR: {ir.count('tt.atomic_rmw')}")


# ---------------------------------------------------------------- Q5
@triton.jit
def q5_int_kernel(x_ptr, y_ptr, out_ptr):
    x = tl.load(x_ptr)
    y = tl.load(y_ptr)
    tl.store(out_ptr, (x and y).to(tl.int32))


@triton.jit
def q5_bool_kernel(x_ptr, y_ptr, out_ptr):
    x = tl.load(x_ptr) != 0
    y = tl.load(y_ptr) != 0
    tl.store(out_ptr, (x and y).to(tl.int32))
    tl.store(out_ptr + 1, (x or y).to(tl.int32))
    tl.store(out_ptr + 2, (not x).to(tl.int32))


def q5():
    x = torch.tensor([2], dtype=torch.int32, device="cuda")
    y = torch.tensor([0], dtype=torch.int32, device="cuda")
    out = torch.zeros(3, dtype=torch.int32, device="cuda")
    print("Q5 `and` on scalar tensors")
    try:
        q5_int_kernel[(1,)](x, y, out)
        torch.cuda.synchronize()
        print(f"  int32 `2 and 0`: compiled, result {out[0].item()}")
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()[-1]
        print(f"  int32 `x and y`: rejected on the host ({type(e).__name__}: {msg[:90]})")
    q5_bool_kernel[(1,)](x, y, out)
    torch.cuda.synchronize()
    print(f"  int1 (x = 2 != 0, y = 0 != 0): and {out[0].item()}, or {out[1].item()}, not x {out[2].item()}")


# ---------------------------------------------------------------- Q6
@triton.jit
def q6_kernel(xi_ptr, xf_ptr, xh_ptr, oi_ptr, of_ptr, oh_ptr):
    xi = tl.load(xi_ptr)
    xf = tl.load(xf_ptr)
    xh = tl.load(xh_ptr)
    tl.store(oi_ptr, xi * 1e-40)
    tl.store(of_ptr, xf * 1e-40)
    tl.store(oh_ptr, xh * 1e-40)


def q6():
    xi = torch.tensor([3], dtype=torch.int32, device="cuda")
    xf = torch.tensor([3.0], dtype=torch.float32, device="cuda")
    xh = torch.tensor([3.0], dtype=torch.float16, device="cuda")
    oi = torch.zeros(1, dtype=torch.float64, device="cuda")
    of = torch.zeros(1, dtype=torch.float32, device="cuda")
    oh = torch.zeros(1, dtype=torch.float16, device="cuda")
    q6_kernel[(1,)](xi, xf, xh, oi, of, oh)
    torch.cuda.synchronize()
    ir = ttir(q6_kernel)
    muls = [("f64" if ": f64" in l else "f32" if ": f32" in l else "f16" if ": f16" in l else "?")
            for l in ir.splitlines() if "arith.mulf" in l]
    print("Q6 the dtype of an operator with a Python float")
    print(f"  TTIR products: {muls}; int32 3 * 1e-40 = {oi.item()!r} (stored to float64); "
          f"float32 3 * 1e-40 = {of.item()!r}; float16 3 * 1e-40 = {oh.item()!r}")


# ---------------------------------------------------------------- Q7
@triton.jit
def q7_kernel(a_ptr, b_ptr, out_ptr):
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    tl.store(out_ptr, a // b)
    tl.store(out_ptr + 1, a % b)


def q7():
    a = torch.tensor([-7], dtype=torch.int32, device="cuda")
    b = torch.tensor([2], dtype=torch.int32, device="cuda")
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    q7_kernel[(1,)](a, b, out)
    torch.cuda.synchronize()
    print(f"Q7 integer -7 // 2 = {out[0].item()}, -7 % 2 = {out[1].item()} (C: -3, -1; Python: -4, 1)")


# ---------------------------------------------------------------- Q8
@triton.jit
def q8_kernel(cnt_ptr, cnt2_ptr, out_ptr):
    i = 0
    while tl.atomic_add(cnt_ptr, 1) < 3:
        i += 1
    for j in range(3):
        tl.atomic_add(cnt2_ptr, 1)
    tl.store(out_ptr, i)


def q8():
    cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    cnt2 = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    q8_kernel[(1,)](cnt, cnt2, out)
    torch.cuda.synchronize()
    print(f"Q8 while-condition atomic: counter {cnt.item()} (4 = tests 0,1,2,3), iterations {out.item()}; "
          f"for-body atomic x3: counter {cnt2.item()}")


for q in (q1, q2, q3, q4, q5, q6, q7, q8):
    try:
        q()
    except Exception as e:  # noqa: BLE001
        print(f"{q.__name__}: FAILED {type(e).__name__}: {str(e).strip().splitlines()[-1][:200]}")
