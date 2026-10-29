"""lr-14b, round 14 (L64): Python constants `±inf` and NaN built in `float8e5` (and `float8e4nv`).

SEMANTICS ("Constants") builds a Python number in an fp8 dtype as `FloatV.litF32`, then
`tl.cvt8`, and docs/rules/num.md says such a constant is folded only when the conversion is exact,
"so the facts cover it too". With `CvtSatF32E5m2`, `tl.cvt8` of `±inf` is `±57344` (`0x7b`,
`0xfb`). This probe builds `±inf` and NaN constants in `float8e5` in every place SEMANTICS lists
(`tl.full`, a value of `tl.where`, an arithmetic operand) and in the places that first make a
`float32` tensor (`other=` of `tl.load`, a stored value, an explicit `.to`), and prints the byte the
device stores and the fp8 instructions of the PTX. A kernel Triton rejects on the host is listed as
such (no outcome).

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I cvt8_inf.py   (output: cvt8_inf.out)
"""
import torch
import triton
import triton.language as tl

B = 128


@triton.jit
def k_full(o, C: tl.constexpr, DT: tl.constexpr, B: tl.constexpr):
    tl.store(o + tl.arange(0, B), tl.full([B], C, DT).to(tl.uint8, bitcast=True))


@triton.jit
def k_where(o, x_ptr, C: tl.constexpr, DT: tl.constexpr, B: tl.constexpr):
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + offs)  # fp8 values (zeros)
    r = tl.where(offs >= 0, C, x)  # every lane takes the constant
    tl.store(o + offs, r.to(tl.uint8, bitcast=True))


@triton.jit
def k_arith(o, x_ptr, C: tl.constexpr, DT: tl.constexpr, B: tl.constexpr):
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + offs)  # fp8 zeros
    r = x + C
    tl.store(o + offs, r.to(tl.uint8, bitcast=True))


@triton.jit
def k_other(o, x_ptr, C: tl.constexpr, DT: tl.constexpr, B: tl.constexpr):
    offs = tl.arange(0, B)
    r = tl.load(x_ptr + offs, mask=offs < 0, other=C)  # every lane masked: `other`
    tl.store(o + offs, r.to(tl.uint8, bitcast=True))


@triton.jit
def k_tostore(o8, C: tl.constexpr, DT: tl.constexpr, B: tl.constexpr):
    tl.store(o8 + tl.arange(0, B), tl.full([B], C, tl.float32))


@triton.jit
def k_to(o, C: tl.constexpr, DT: tl.constexpr, B: tl.constexpr):
    tl.store(o + tl.arange(0, B), tl.full([B], C, tl.float32).to(DT).to(tl.uint8, bitcast=True))


def main():
    lines = [f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}", ""]
    lines.append("| dtype | c | where it is built | byte stored | fp8 instructions in the PTX |")
    lines.append("|---|---|---|---|---|")
    for dname, dt, tdt in (("float8e5", tl.float8e5, torch.float8_e5m2), ("float8e4nv", tl.float8e4nv, torch.float8_e4m3fn)):
        for c in (float("inf"), float("-inf"), float("nan"), 1000.0, 60000.0):
            for kname, kern, extra in (("tl.full(shape, c, fp8)", k_full, False),
                                       ("tl.where(m, c, x_fp8)", k_where, True),
                                       ("x_fp8 + c", k_arith, True),
                                       ("tl.load(p_fp8, mask, other=c)", k_other, True),
                                       ("tl.store(p_fp8, float32 tensor of c)", k_tostore, None),
                                       ("tl.full(shape, c, float32).to(fp8)", k_to, False)):
                try:
                    if extra is None:
                        o8 = torch.zeros(B, dtype=tdt, device="cuda")
                        h = kern[(1,)](o8, C=c, DT=dt, B=B)
                        o = o8.view(torch.uint8)
                    else:
                        o = torch.zeros(B, dtype=torch.uint8, device="cuda")
                        if extra:
                            x = torch.zeros(B, dtype=tdt, device="cuda")
                            h = kern[(1,)](o, x, C=c, DT=dt, B=B)
                        else:
                            h = kern[(1,)](o, C=c, DT=dt, B=B)
                    vals = sorted(set(o.tolist()))
                    byte = ", ".join(f"`{v:02x}`" for v in vals)
                    ins = sorted({l.split()[0] for l in h.asm["ptx"].splitlines()
                                  if l.strip().startswith("cvt") and ("e4m3" in l or "e5m2" in l)})
                    ptx = ", ".join(f"`{i}`" for i in ins) if ins else "none: a folded constant"
                except Exception as ex:
                    msg = str(ex).strip().splitlines()[-1][:90] if str(ex).strip() else ""
                    byte, ptx = f"rejected on the host ({type(ex).__name__})", msg
                lines.append(f"| {dname} | {c!r} | `{kname}` | {byte} | {ptx} |")
    lines.append("")
    lines.append("`CvtSatF32E5m2` (`tl.cvt8` of the float32 constant): +inf -> `7b`, -inf -> `fb`, NaN -> a NaN, "
                 "1000 -> `64`, 60000 -> `7b`; `CvtSatF32E4m3`: +inf, 1000, 60000 -> `7e`, -inf -> `fe`, NaN -> `7f`.")
    text = "\n".join(lines)
    print(text)
    with open("cvt8_inf.out", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
