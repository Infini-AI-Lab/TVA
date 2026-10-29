"""Calls of Python's builtins in a kernel.

Library evidence for library issue L75 (round 11). The front end looks a name up in the function's
locals, then its module's globals, then `builtin_namespace`
(`primitives/triton-3.8.0-code-generator.py` lines 357-365, 381-427): `len`, `list`, `range`, `float`,
`int`, `isinstance`, `getattr` and `hasattr` are Python's own, and `print`, `min` and `max` are
Triton's (`tl.device_print`, `builtin_min`, `builtin_max`). A call of `float`, `isinstance`, `list`,
`getattr` or `hasattr` runs the Python function on the arguments, each `tl.constexpr` unwrapped
(lines 1444-1448), and `int` and `len` run through the static executor (lines 1638-1659); the result
is wrapped as a compile-time value (`normalize_value`, lines 95-100). No other builtin is visible.
Each case is compiled and run on GPU 0 in its own process.

  F1  tl.store(out, -float('inf')) and tl.load(p, mask, other=-float('inf')) (tutorial 02): the bits
  F2  tl.zeros([4], tl.float32) - float("inf") (tutorial 06): the bits
  F3  tl.static_assert(float("inf") > 1e308); tl.static_assert(int(2.7) == 2); tl.static_assert(len((1, 2, 3)) == 3)
  F4  isinstance(p, tl.tensor) and isinstance(p, tl.tensor_descriptor) for a pointer argument and a host
      TensorDescriptor argument (tutorial 06's _maybe_make_tensor_desc), deciding a compile-time if
  F5  getattr(x, "dtype") == tl.float32 and hasattr(x, "dtype") in tl.static_assert
  F6  len(x.shape) == 1 for a block x; list(x.shape) == [4]
  F7  isinstance(tl.atomic_add(c, 1), tl.tensor) in a tl.static_assert (8 programs): does the atomic run?
  F8  float(x) of a loaded scalar (a tensor)
  F9  abs(-3) (a builtin outside builtin_namespace)
  F10 int(n) of a runtime int32 argument

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/py-builtins/run.py  (from TVA's root)
"""
import json, os, struct, subprocess, sys

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def k_inf(p, out_ptr, CASE: tl.constexpr):
    i = tl.arange(0, 4)
    if CASE == 1:
        tl.store(out_ptr, -float('inf'))
        tl.store(out_ptr + 1 + i, tl.load(p + i, mask=i < 2, other=-float('inf')))
    else:
        tl.store(out_ptr + i, tl.zeros([4], dtype=tl.float32) - float("inf"))


@triton.jit
def k_static(x_ptr, out_ptr, n, c_ptr, CASE: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, 4))
    if CASE == 3:
        tl.static_assert(float("inf") > 1e308)
        tl.static_assert(int(2.7) == 2)
        tl.static_assert(len((1, 2, 3)) == 3)
    elif CASE == 5:
        tl.static_assert(getattr(x, "dtype") == tl.float32)
        tl.static_assert(hasattr(x, "dtype"))
    elif CASE == 6:
        tl.static_assert(len(x.shape) == 1)
        tl.static_assert(list(x.shape) == [4])
    elif CASE == 7:
        tl.static_assert(isinstance(tl.atomic_add(c_ptr, 1), tl.tensor))
    elif CASE == 8:
        tl.static_assert(float(tl.load(x_ptr)) > 0)
    elif CASE == 9:
        tl.static_assert(abs(-3) == 3)
    else:
        tl.static_assert(int(n) >= 0)
    tl.store(out_ptr, 1.0)


@triton.jit
def k_isinstance(src, out_ptr):
    if isinstance(src, tl.tensor_descriptor):
        tl.store(out_ptr, 1.0)
    else:
        tl.store(out_ptr, 2.0)
    if isinstance(src, tl.tensor):
        tl.store(out_ptr + 1, 1.0)
    else:
        tl.store(out_ptr + 1, 2.0)


CASES = {
    "F1": "-float('inf') stored, and as other= of a masked load",
    "F2": "tl.zeros([4], tl.float32) - float('inf')",
    "F3": "float, int and len of compile-time values in tl.static_assert",
    "F4": "isinstance on a pointer argument and on a host TensorDescriptor argument",
    "F5": "getattr(x, 'dtype') and hasattr(x, 'dtype') of a block",
    "F6": "len(x.shape) and list(x.shape) of a block",
    "F7": "isinstance(tl.atomic_add(c, 1), tl.tensor) in tl.static_assert, 8 programs",
    "F8": "float(x) of a loaded scalar",
    "F9": "abs(-3)",
    "F10": "int(n) of a runtime int32 argument",
}


def bits(v):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", v))[0]


def run_case(name):
    dev = "cuda"
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
    out = torch.full((5,), 7.0, device=dev)
    if name in ("F1", "F2"):
        k_inf[(1,)](x, out, CASE=1 if name == "F1" else 2)
        torch.cuda.synchronize()
        n = 5 if name == "F1" else 4
        return "stored " + ", ".join(f"{v!r} ({bits(v)})" for v in out.tolist()[:n])
    if name == "F4":
        res = []
        for label, arg in (("pointer", x),
                           ("TensorDescriptor", TensorDescriptor(x.view(1, 4), shape=[1, 4], strides=[4, 1],
                                                                  block_shape=[1, 4]))):
            o = torch.full((2,), 7.0, device=dev)
            ck = k_isinstance[(1,)](arg, o)
            torch.cuda.synchronize()
            stores = sum(1 for l in ck.asm["ttir"].splitlines() if "tt.store" in l)
            res.append(f"{label}: is a tl.tensor_descriptor -> {o[0].item()}, is a tl.tensor -> {o[1].item()} "
                       f"(1.0 yes, 2.0 no); stores in the TTIR: {stores}")
        return "; ".join(res)
    c = torch.zeros(1, device=dev, dtype=torch.int32)
    k_static[(8,)](x, out, 5, c, CASE=int(name[1:]))
    torch.cuda.synchronize()
    return f"compiled and ran; atomics: {c.item()}"


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
    for name, desc in CASES.items():
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--case", name],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[len("RESULT "):]) if line else "process failed: " + p.stderr.strip()[-160:]
        print(f"{name}: {desc}\n  {res}")


if __name__ == "__main__":
    main()
