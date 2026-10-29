"""lr-9b, library review round 9 (L4): the device's fp8 conversions against the Lean rounding that
`Rules.Num.CvtRn` names (`Format.cvt f g .rn false .canonical`, evaluated by `CvtEval.lean`).

Run from the TVA root:
  (cd lean && lake env lean ../docs/library-review/round-9/lr-9b/work/CvtEval.lean) > lean.out
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-9/lr-9b/work/cvt_compare.py lean.out

For each input in range (finite, |x| <= maxVal of the fp8 format), the device's `x.to(fp8)` must
equal the Lean rounding bit for bit (CvtRn is up to NaN payload, and no NaN arises in range).
"""
import sys
import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def conv(x_ptr, o_ptr, n, OUT: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(o_ptr + offs, x.to(OUT), mask=m)


SRC = {"F16": (torch.int16, torch.float16), "BF16": (torch.int16, torch.bfloat16), "F32": (torch.int32, torch.float32)}
DST = {"E4": (tl.float8e4nv, torch.float8_e4m3fn, 448.0), "E5": (tl.float8e5, torch.float8_e5m2, 57344.0)}


def main(path):
    lines = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            lines[parts[0]] = parts[1]
    print("| from | to | inputs | in range | differ from Lean |")
    print("|---|---|---|---|---|")
    total = 0
    for key, hexs in lines.items():
        src, dst = key[:-2], key[-2:]
        ity, fty = SRC[src]
        oty, tdt, mx = DST[dst]
        lean = torch.tensor(list(bytes.fromhex(hexs)), dtype=torch.int64)
        n = lean.numel()
        if src == "F32":
            pats = [(i * 65521) % 4294967296 for i in range(n)]
            p = torch.tensor(pats, dtype=torch.int64)
            p = torch.where(p >= 2 ** 31, p - 2 ** 32, p).to(torch.int32)
        else:
            p = torch.arange(n, dtype=torch.int64)
            p = torch.where(p >= 2 ** 15, p - 2 ** 16, p).to(torch.int16)
        x = p.to(DEV).view(fty)
        o = torch.empty(n, dtype=tdt, device=DEV)
        conv[(triton.cdiv(n, 1024),)](x, o, n, OUT=oty, B=1024)
        dev = o.view(torch.uint8).to(torch.int64).cpu()
        xf = x.float().cpu()
        inr = torch.isfinite(xf) & (xf.abs() <= mx)
        diff = int(((dev != lean) & inr).sum())
        total += diff
        print(f"| {src} | {dst} | {n} | {int(inr.sum())} | {diff} |")
        if diff:
            idx = torch.nonzero((dev != lean) & inr).flatten()[:5]
            for i in idx.tolist():
                print(f"  x={xf[i].item()!r} pattern={int(p[i]) & ((1 << (8 * p.element_size())) - 1):#x} device={int(dev[i]):#04x} lean={int(lean[i]):#04x}")
    print(f"\nTotal in-range inputs where the device differs from the Lean rounding: {total}")


if __name__ == "__main__":
    main(sys.argv[1])
