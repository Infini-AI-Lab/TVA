"""lr-20e: the approximate-function facts of docs/rules/num.md on every float32 input, independently of
evidence/num-math-f32.

For each fact (`ExpF32`, `Exp2F32`, `LogF32`, `Log2F32`, `SqrtF32`, `RsqrtF32`, `TVA.Rules.Num.Sm90`), every one
of the 2^32 float32 bit patterns is passed through the Triton function in a kernel of its own (one `FN` per
compilation), and the result is checked against `MathAcc`'s statement (`TVA.Rules.Num`): for a finite `x` in the
fact's domain and in the function's natural domain, with room (`(1 + rel)·|ref x| + abs ≤ maxVal`), the result is
finite and `|r − ref x| ≤ rel·|ref x| + abs`. `ExpF32` is a family over `X ≥ 0` on `[-X, X]`; for each `x` its
strongest member is `X = |x|` (a larger `X` only weakens both the bound and the room), so `rel = 2^-22 + 2^-23·|x|`.
The reference is float64 (torch on the GPU, within a few ulp of float64); a violation is counted only beyond the
bound plus `2^-50·|ref|`. It also checks `ExpF32NegInf` and `Exp2F32NegInf` (the bits of `+0` at `-inf`).

Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I math_f32_all.py [nofusion]
With `nofusion`, every launch passes `enable_fp_fusion=False` (`Launch.fusion = false`, which scope 4 admits):
the facts do not depend on it.
"""
import math
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def fk(x_ptr, o_ptr, n, FN: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    if FN == 0:
        y = tl.exp(x)
    elif FN == 1:
        y = tl.math.exp2(x)
    elif FN == 2:
        y = tl.log(x)
    elif FN == 3:
        y = tl.math.log2(x)
    elif FN == 4:
        y = tl.sqrt(x)
    else:
        y = tl.math.rsqrt(x)
    tl.store(o_ptr + offs, y, mask=m)


MAXV = float.fromhex("0x1.fffffep+127")
N2 = 2.0 ** -126
# name, FN, reference, domain test (on float64 x), rel (function of float64 x), abs
FACTS = [
    ("ExpF32", 0, torch.exp, lambda x: torch.ones_like(x, dtype=torch.bool),
     lambda x: 2.0 ** -22 + 2.0 ** -23 * x.abs(), 2.0 ** -140),
    ("Exp2F32", 1, torch.exp2, lambda x: torch.ones_like(x, dtype=torch.bool),
     lambda x: torch.full_like(x, 2.0 ** -22), 2.0 ** -126),
    ("LogF32", 2, torch.log, lambda x: x > 0, lambda x: torch.full_like(x, 2.0 ** -22), 2.0 ** -149),
    ("Log2F32", 3, torch.log2, lambda x: x > 0, lambda x: torch.full_like(x, 2.0 ** -22), 2.0 ** -149),
    ("SqrtF32", 4, torch.sqrt, lambda x: (x == 0) | (x >= N2), lambda x: torch.full_like(x, 2.0 ** -22), 2.0 ** -149),
    ("RsqrtF32", 5, torch.rsqrt, lambda x: x >= N2, lambda x: torch.full_like(x, 2.0 ** -22), 2.0 ** -149),
]


FUSION = not (len(sys.argv) > 1 and sys.argv[1] == "nofusion")


def main():
    print(f"enable_fp_fusion={FUSION}")
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"Python {sys.version.split()[0]}")
    CH = 1 << 26
    B = 1024
    stats = {f[0]: dict(checked=0, viol=0, worst=0.0, worst_x=None, nonfinite=0) for f in FACTS}
    for c in range(1 << 32 >> 26):
        bits = torch.arange(c * CH, (c + 1) * CH, dtype=torch.int64, device="cuda")
        bits = torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32)
        x = bits.view(torch.float32)
        x64 = x.double()
        fin = torch.isfinite(x64)
        for name, fn, ref_f, dom, relf, ab in FACTS:
            y = torch.empty_like(x)
            fk[(CH // B,)](x, y, CH, FN=fn, B=B, enable_fp_fusion=FUSION)
            ref = ref_f(x64)
            rel = relf(x64)
            sel = fin & dom(x64)
            room = (1 + rel) * ref.abs() + ab <= MAXV
            sel = sel & room & torch.isfinite(ref)
            y64 = y.double()
            bound = rel * ref.abs() + ab
            err = (y64 - ref).abs()
            okfin = torch.isfinite(y64)
            bad = sel & (~okfin | (err > bound + 2.0 ** -50 * ref.abs()))
            st = stats[name]
            st["checked"] += int(sel.sum())
            st["viol"] += int(bad.sum())
            st["nonfinite"] += int((sel & ~okfin).sum())
            ratio = torch.where(sel & okfin, err / bound, torch.zeros_like(err))
            mx, ix = ratio.max(dim=0)
            if float(mx) > st["worst"]:
                st["worst"] = float(mx)
                st["worst_x"] = (int(bits[ix]) & 0xFFFFFFFF, float(x[ix]), float(y[ix]), float(ref[ix]))
            if int(bad.sum()) and st.get("first") is None:
                i = int(torch.nonzero(bad)[0])
                st["first"] = (int(bits[i]) & 0xFFFFFFFF, float(x[i]), float(y[i]), float(ref[i]))
    print("| fact | inputs checked | outside the fact | non-finite where finite is required | largest error / bound | at x (pattern, x, result, float64 ref) |")
    print("|---|---|---|---|---|---|")
    total = 0
    for name, *_ in FACTS:
        st = stats[name]
        total += st["viol"]
        w = st["worst_x"]
        ws = f"`{w[0]:08x}`, {w[1]!r}, {w[2]!r}, {w[3]!r}" if w else "-"
        print(f"| {name} | {st['checked']} | {st['viol']} | {st['nonfinite']} | {st['worst']:.4f} | {ws} |"
              + (f" first outside: {st['first']}" if st.get("first") else ""))
    # MathAt at -inf
    x = torch.tensor([-math.inf], dtype=torch.float32, device="cuda")
    for name, fn in (("ExpF32NegInf", 0), ("Exp2F32NegInf", 1)):
        y = torch.empty_like(x)
        fk[(1,)](x, y, 1, FN=fn, B=B, enable_fp_fusion=FUSION)
        b = int(y.view(torch.int32)[0]) & 0xFFFFFFFF
        ok = b == 0
        total += 0 if ok else 1
        print(f"{name}: result bits {b:08x} ({'+0, as the fact says' if ok else 'NOT +0'})")
    print(f"outside the facts in all: {total}")


if __name__ == "__main__":
    main()
