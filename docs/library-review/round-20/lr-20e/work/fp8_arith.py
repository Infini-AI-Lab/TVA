"""lr-20e: does Triton 3.8.0 reject on the host every arithmetic operator with an fp8 tensor operand?

SEMANTICS ("Constants", the fp8 case, and "Operators and promotion", last item) says: "With an fp8 tensor an
arithmetic operator is rejected anyway" and "Arithmetic on two operands of the same fp8 dtype type-checks,
but the backend rejects it on the host." MLIR's arith folders rewrite `x * 1 -> x`, `x / 1 -> x`,
`x - (+0) -> x` before the backend sees them. This script compiles and runs each form, each kernel in a
file of its own, on every fp8 pattern of x, and prints what is stored (or the host error).

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I fp8_arith.py
"""
import importlib.util
import os
import sys
import tempfile

import torch
import triton

FORMS = ["x * 1.0", "1.0 * x", "x / 1.0", "x - 0.0", "x * 1", "x + 0.0", "x * 2.0", "x + x", "x * x",
         "x - x", "x / 2.0", "tl.where(i < 0, x, 1.0) * 1.0", "x * -1.0", "-x",
         "tl.full([256], 1.5, DT) * tl.full([256], 2.0, DT)", "tl.full([256], 1.5, DT) + 0.5"]


def load_kernel(tmp, name, expr, dt):
    src = "\n".join([
        "import triton", "import triton.language as tl", "", "@triton.jit",
        f"def {name}(x_ptr, o_ptr):",
        "    i = tl.arange(0, 256)",
        f"    x = tl.load(x_ptr + i).to({dt}, bitcast=True)",
        f"    y = {expr.replace('DT', dt)}",
        "    tl.store(o_ptr + i, y.to(tl.uint8, bitcast=True))", ""])
    path = os.path.join(tmp, name + ".py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    tmp = tempfile.mkdtemp(prefix="lr20e-fp8a-")
    x = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    for dt in ("tl.float8e5", "tl.float8e4nv"):
        for k, expr in enumerate(FORMS):
            name = f"k_{dt.split('.')[1]}_{k}"
            kern = load_kernel(tmp, name, expr, dt)
            o = torch.full((256,), 0x55, dtype=torch.uint8, device="cuda")
            try:
                kern[(1,)](x, o)
                torch.cuda.synchronize()
            except Exception as ex:
                msg = str(ex).strip().splitlines()
                print(f"{dt:14s} {expr:32s} rejected on the host: {type(ex).__name__}: "
                      f"{(msg[-1] if msg else '')[:100]}")
                continue
            got = o.tolist()
            same = sum(1 for a, b in zip(range(256), got) if a == b)
            print(f"{dt:14s} {expr:32s} RAN: {same}/256 patterns stored unchanged; "
                  f"e.g. 0x3c->{got[0x3c]:02x}, 0x40->{got[0x40]:02x}, 0x80->{got[0x80]:02x}, 0x01->{got[0x01]:02x}")


if __name__ == "__main__":
    main()
