"""lr-14b, round 14 (L64): the fp8 byte a NaN or an infinity converts to, from each source format.

The `CvtSat…` docstrings (lean/TVA/Rules/Num/Sm90.lean) say `.satfinite` sends a NaN to the NaN
`0x7f` and `±inf` to `±maxVal`; evidence/num-cvt8 shows this from `float32` only. This prints the
bytes from `float16`, `bfloat16` and `float32` NaNs of both signs, quiet and signalling, and
`±inf`, at run time.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I cvt8_nan.py   (output: cvt8_nan.out)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def k(o, x_ptr, DT: tl.constexpr, B: tl.constexpr):
    offs = tl.arange(0, B)
    x = tl.load(x_ptr + offs)
    tl.store(o + offs, x.to(DT).to(tl.uint8, bitcast=True))


PATS = {
    "float32": (torch.int32, torch.float32, [0x7FC00000, 0x7F800001, 0x7FFFFFFF, -0x00400000, -0x007FFFFF,
                                              0x7F800000, -0x00800000]),
    "float16": (torch.int16, torch.float16, [0x7E00, 0x7C01, 0x7FFF, -0x0200, -0x03FF, 0x7C00, -0x0400]),
    "bfloat16": (torch.int16, torch.bfloat16, [0x7FC0, 0x7F81, 0x7FFF, -0x0040, -0x007F, 0x7F80, -0x0080]),
}
NAMES = ["+qNaN", "+sNaN", "+NaN all ones", "-qNaN", "-NaN (low payload)", "+inf", "-inf"]


def main():
    lines = [f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}", ""]
    lines.append("| from | input | pattern | to float8e4nv | to float8e5 |")
    lines.append("|---|---|---|---|---|")
    for src, (it, ft, pats) in PATS.items():
        B = 8
        raw = torch.tensor(pats + [0] * (B - len(pats)), dtype=it, device="cuda")
        x = raw.view(ft)
        outs = {}
        for name, dt in (("e4m3", tl.float8e4nv), ("e5m2", tl.float8e5)):
            o = torch.zeros(B, dtype=torch.uint8, device="cuda")
            k[(1,)](o, x, DT=dt, B=B)
            outs[name] = o.tolist()
        width = 8 if it == torch.int32 else 4
        for j, p in enumerate(pats):
            lines.append(f"| {src} | {NAMES[j]} | `{p & ((1 << (4 * width)) - 1):0{width}x}` | "
                         f"`{outs['e4m3'][j]:02x}` | `{outs['e5m2'][j]:02x}` |")
    text = "\n".join(lines)
    print(text)
    with open("cvt8_nan.out", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
