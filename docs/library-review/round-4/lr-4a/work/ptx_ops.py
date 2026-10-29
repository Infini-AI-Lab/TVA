"""lr-4a: which PTX instruction each "exact" float operation of rules/num.md compiles to.

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/ptx_ops.py
Prints, per operation and dtype, the float instructions of the kernel's PTX (ld/st/mov/cvt of
addresses filtered out), so that `.ftz` or an approximate instruction shows.
"""
import re
import torch
import triton
import triton.language as tl

DT = {"f16": (torch.float16, tl.float16), "bf16": (torch.bfloat16, tl.bfloat16),
      "f32": (torch.float32, tl.float32), "f64": (torch.float64, tl.float64)}


@triton.jit
def k_add(x, y, o, OP: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    a = tl.load(x + i)
    b = tl.load(y + i)
    if OP == "add":
        r = a + b
    elif OP == "sub":
        r = a - b
    elif OP == "mul":
        r = a * b
    elif OP == "div":
        r = a / b
    elif OP == "div_rn":
        r = tl.div_rn(a, b)
    elif OP == "sqrt_rn":
        r = tl.sqrt_rn(a)
    elif OP == "sqrt":
        r = tl.sqrt(a)
    elif OP == "exp":
        r = tl.exp(a)
    elif OP == "exp2":
        r = tl.exp2(a)
    elif OP == "fma":
        r = tl.fma(a, b, a)
    tl.store(o + i, r)


@triton.jit
def k_cvt(x, o, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.store(o + i, tl.load(x + i).to(o.dtype.element_ty))


KEEP = re.compile(r"^\s*(add|sub|mul|fma|div|sqrt|rcp|ex2|lg2|cvt\.r|cvt\.f|cvt\.rn|neg|abs|min|max)\S*")


def show(name, ptx):
    ins = sorted({l.strip().split()[0] for l in ptx.splitlines() if KEEP.match(l)
                  and not re.match(r"^\s*(add|mul|sub)\.(s|u|lo|wide|hi)", l)
                  and not re.match(r"^\s*(add|mul|sub)\.\w*(s32|u32|s64|u64|b32|b64)\b", l)
                  and not re.match(r"^\s*cvt\.\w*(s32|u32|s64|u64)", l)})
    print(f"{name:28s} {' '.join(ins)}")


def main():
    print("triton", triton.__version__, torch.cuda.get_device_name(0))
    B = 128
    for op in ["add", "sub", "mul", "fma", "div", "div_rn", "sqrt_rn", "sqrt", "exp", "exp2"]:
        for d in ["f16", "bf16", "f32", "f64"]:
            td, _ = DT[d]
            x = torch.ones(B, dtype=td, device="cuda")
            y = torch.ones(B, dtype=td, device="cuda")
            outd = td
            if op == "div" and d in ("f16", "bf16"):
                outd = torch.float32
            o = torch.empty(B, dtype=outd, device="cuda")
            try:
                h = k_add[(1,)](x, y, o, OP=op, B=B)
                show(f"{op} {d}", h.asm["ptx"])
            except Exception as e:  # noqa: BLE001
                print(f"{op + ' ' + d:28s} rejected: {type(e).__name__}")
    for s, t in [("f32", "f16"), ("f32", "bf16"), ("f64", "f32"), ("f64", "f16"), ("f64", "bf16"),
                 ("f16", "bf16"), ("bf16", "f16"), ("f16", "f32"), ("bf16", "f32"), ("f32", "f64"),
                 ("f16", "f64"), ("bf16", "f64")]:
        x = torch.ones(B, dtype=DT[s][0], device="cuda")
        o = torch.empty(B, dtype=DT[t][0], device="cuda")
        try:
            h = k_cvt[(1,)](x, o, B=B)
            show(f"cvt {s}->{t}", h.asm["ptx"])
        except Exception as e:  # noqa: BLE001
            print(f"cvt {s}->{t}: rejected: {type(e).__name__}")


if __name__ == "__main__":
    main()
