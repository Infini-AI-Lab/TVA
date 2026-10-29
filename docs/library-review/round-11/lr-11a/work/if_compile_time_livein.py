"""lr-11a: a runtime `if` that assigns a name whose value before the `if` is not a tensor
(a `tl.constexpr` parameter, `None`, a dtype, a name annotated `: tl.constexpr`).

SEMANTICS "A runtime `if`" (round 11, L61): "After the `if`, the names it merges have the taken
branch's value: a name defined before the `if` that a branch assigns, and a name new to the `if` that
both branches define. A name defined before the `if` must keep its dtype and shape in each branch that
assigns it, also when both do ... otherwise the kernel is rejected on the host."

`visit_then_else_blocks` (code_generator.py:855-868) skips a live-in that is not a Triton value
(`if not _is_triton_value(value): continue`; a `tl.constexpr` is one, `core.py:210`, but `None` and a
dtype are not). Such a name is merged only when both branches define it (as a new name, :876-889),
and when one branch does, it keeps its value from before the `if` (`enter_sub_region` restores the
scope, :136-140).

  I1  FACTOR: tl.constexpr = 2 (a parameter); if n > 0: FACTOR = 3 (one branch); store x * FACTOR
      (a constexpr is a Triton value: the type check applies)
  I2  y = None; if n > 0: y = tl.load(p) (one branch); if y is not None: store y else store -1
      SEMANTICS: merged (stores the load, 10) or rejected; the source: y is still None: stores -1
  I3  y = None; if n > 0: y = tl.load(p) else: y = tl.load(q) (both); store y
      SEMANTICS: rejected (y's type changes from None); the source: merged, runs
  I4  c: tl.constexpr = 2; if n > 0: c = 3 (one branch); store x * c
  I5  control: c = 2 (a tensor, L59); if n > 0: c = 3; store x * c  -> 3x where n > 0
  I6  d = tl.float16; if n > 0: d = tl.float32 (one branch); store x.to(d) of x = 1 + 2^-12
      SEMANTICS: merged (float32: 1.000244140625) or rejected; the source: d is still float16 (1.0)
Every case runs with n = 2 (not 1, which the launcher makes a compile-time constant).
Run: CUDA_VISIBLE_DEVICES=0 <venv>/bin/python -I if_compile_time_livein.py
"""
import json, os, subprocess, sys

import torch
import triton
import triton.language as tl


@triton.jit
def k1(x_ptr, out_ptr, n, FACTOR: tl.constexpr):
    x = tl.load(x_ptr)
    if n > 0:
        FACTOR = 3
    tl.store(out_ptr, x * FACTOR)


@triton.jit
def k2(p, q, out_ptr, n):
    y = None
    if n > 0:
        y = tl.load(p)
    if y is not None:
        tl.store(out_ptr, y)
    else:
        tl.store(out_ptr, -1.0)


@triton.jit
def k3(p, q, out_ptr, n):
    y = None
    if n > 0:
        y = tl.load(p)
    else:
        y = tl.load(q)
    tl.store(out_ptr, y)


@triton.jit
def k4(x_ptr, out_ptr, n):
    x = tl.load(x_ptr)
    c: tl.constexpr = 2
    if n > 0:
        c = 3
    tl.store(out_ptr, x * c)


@triton.jit
def k5(x_ptr, out_ptr, n):
    x = tl.load(x_ptr)
    c = 2
    if n > 0:
        c = 3
    tl.store(out_ptr, x * c)


@triton.jit
def k6(x_ptr, out_ptr, n):
    x = tl.load(x_ptr)
    d = tl.float16
    if n > 0:
        d = tl.float32
    tl.store(out_ptr, x.to(d).to(tl.float32))


def run_case(name):
    x = torch.tensor([10.0], device="cuda")
    q = torch.tensor([20.0], device="cuda")
    out = torch.zeros(1, device="cuda")
    n = 2
    if name == "I1":
        k1[(1,)](x, out, n, FACTOR=2)
    elif name == "I2":
        k2[(1,)](x, q, out, n)
    elif name == "I3":
        k3[(1,)](x, q, out, n)
    elif name == "I4":
        k4[(1,)](x, out, n)
    elif name == "I5":
        k5[(1,)](x, out, n)
    else:
        x6 = torch.tensor([1.0 + 2.0 ** -12], device="cuda")
        k6[(1,)](x6, out, n)
    torch.cuda.synchronize()
    return f"compiled and ran with n = 2; stored {out.item()!r}"


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        try:
            res = run_case(sys.argv[2])
        except Exception as e:
            msg = [l for l in str(e).strip().splitlines() if l.strip()]
            res = "rejected on the host: " + type(e).__name__ + ": " + (msg[-1] if msg else "")[:160]
        print("RESULT " + json.dumps(res))
        return
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for name in ["I1", "I2", "I3", "I4", "I5", "I6"]:
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[7:]) if line else "process failed: " + (p.stderr.strip().splitlines() or ["?"])[-1][:200]
        print(f"{name}: {res}")


if __name__ == "__main__":
    main()
