"""lr-9a: Rules.Num.CvtRn against the device, through the library's own definition.

evidence/num-cvt8 compares the device with a Python round-to-nearest-even. CvtRn states equality
with the Lean `FloatV.cvtWith .rn` (= `Format.cvt src dst .rn false .canonical`). This compares
the device with the Lean definition's own output (`cvt_lean.lean`, run with `lake env lean --run`)
bit for bit, for every finite input with |x| <= maxVal of the destination: every float16 and
bfloat16 pattern, and 393216 float32 patterns (any high half, low half in {0, 1, 0x7fff, 0x8000,
0x8001, 0xffff}: the ties of both fp8 formats and their neighbours).

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I cvt_lean.py <the Lean program's output>
"""
import sys
import torch
import triton
import triton.language as tl

LOWS = [0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff]


@triton.jit
def cvt(x_ptr, e4_ptr, e5_ptr, n, SRC: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    raw = tl.load(x_ptr + offs, mask=m)
    if SRC == 0:
        x = raw.to(tl.float32, bitcast=True)
    elif SRC == 1:
        x = raw.to(tl.float16, bitcast=True)
    else:
        x = raw.to(tl.bfloat16, bitcast=True)
    tl.store(e4_ptr + offs, x.to(tl.float8e4nv).to(tl.int8, bitcast=True), mask=m)
    tl.store(e5_ptr + offs, x.to(tl.float8e5).to(tl.int8, bitcast=True), mask=m)


def main():
    lean = {"f16": {}, "bf16": {}, "f32": {}, "s32": {}}
    for line in open(sys.argv[1]):
        parts = line.split()
        if len(parts) != 3 or parts[0] not in lean:
            continue
        lean[parts[0]][int(parts[1])] = parts[2]
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("| from | to | finite inputs with abs(x) <= maxVal compared | differ from the Lean `cvtWith .rn` | first difference (input bits, device, Lean) |")
    print("|---|---|---|---|---|")
    for src, name, idt, tdt in [(1, "f16", torch.int16, torch.float16), (2, "bf16", torch.int16, torch.bfloat16),
                                (0, "f32", torch.int32, torch.float32)]:
        if name == "f32":
            pats = [hi * 65536 + lo for hi in range(65536) for lo in LOWS]
            leanb = []
            for hi in range(65536):
                s = lean["f32"][hi]
                for j in range(len(LOWS)):
                    leanb.append(s[4 * j:4 * j + 4])
        else:
            pats = list(range(65536))
            leanb = [lean[name][i] for i in pats]
        nbits = 32 if name == "f32" else 16
        b64 = torch.tensor(pats, dtype=torch.int64)
        bits = torch.where(b64 >= 2 ** (nbits - 1), b64 - 2 ** nbits, b64).to(idt).cuda()
        n = bits.numel()
        e4 = torch.empty(n, dtype=torch.int8, device="cuda"); e5 = torch.empty(n, dtype=torch.int8, device="cuda")
        cvt[(triton.cdiv(n, 1024),)](bits, e4, e5, n, SRC=src, B=1024)
        xf = bits.view(tdt).double().cpu().tolist()
        e4 = (e4.to(torch.int64) & 0xff).cpu().tolist(); e5 = (e5.to(torch.int64) & 0xff).cpu().tolist()
        for fmt, dev, off, maxv in [("e4m3", e4, 0, 448.0), ("e5m2", e5, 2, 57344.0)]:
            cmp = diff = 0
            first = None
            for i in range(n):
                v = xf[i]
                if not (abs(v) <= maxv):   # NaN, infinities and out-of-range values are outside CvtRn
                    continue
                lb = leanb[i][off:off + 2]
                assert lb != "--"
                cmp += 1
                if int(lb, 16) != dev[i]:
                    diff += 1
                    if first is None:
                        first = (hex(pats[i]), hex(dev[i]), lb)
            print(f"| {name} | {fmt} | {cmp} | {diff} | {first} |")
    main16(lean)


@triton.jit
def cvt16(x_ptr, hn, hz, bn, bz, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m).to(tl.float32, bitcast=True)
    tl.store(hn + offs, x.to(tl.float16).to(tl.int16, bitcast=True), mask=m)
    tl.store(hz + offs, x.to(tl.float16, fp_downcast_rounding="rtz").to(tl.int16, bitcast=True), mask=m)
    tl.store(bn + offs, x.to(tl.bfloat16).to(tl.int16, bitcast=True), mask=m)
    tl.store(bz + offs, x.to(tl.bfloat16, fp_downcast_rounding="rtz").to(tl.int16, bitcast=True), mask=m)


def main16(lean):
    """float32 into float16/bfloat16 (rn, rz) against the Lean `Format.cvt`, bit for bit, on the
    same patterns; NaN inputs are skipped (tl.cvt is up to NaN payload)."""
    pats = [hi * 65536 + lo for hi in range(65536) for lo in LOWS]
    b64 = torch.tensor(pats, dtype=torch.int64)
    bits = torch.where(b64 >= 2 ** 31, b64 - 2 ** 32, b64).to(torch.int32).cuda()
    n = bits.numel()
    outs = [torch.empty(n, dtype=torch.int16, device="cuda") for _ in range(4)]
    cvt16[(triton.cdiv(n, 1024),)](bits, *outs, n, B=1024)
    outs = [(o.to(torch.int64) & 0xffff).cpu().tolist() for o in outs]
    xf = bits.view(torch.float32).double().cpu().tolist()
    print("\n| from | to, rounding | non-NaN inputs compared | differ from the Lean `Format.cvt` | first difference |")
    print("|---|---|---|---|---|")
    for j, name in enumerate(["float16, rn", "float16, rz", "bfloat16, rn", "bfloat16, rz"]):
        cmp = diff = 0
        first = None
        for i in range(n):
            if xf[i] != xf[i]:
                continue
            s = lean["s32"][pats[i] // 65536]
            k = LOWS.index(pats[i] % 65536)
            lb = s[16 * k + 4 * j:16 * k + 4 * j + 4]
            cmp += 1
            if int(lb, 16) != outs[j][i]:
                diff += 1
                if first is None:
                    first = (hex(pats[i]), hex(outs[j][i]), lb)
        print(f"| f32 | {name} | {cmp} | {diff} | {first} |")


if __name__ == "__main__":
    main()
