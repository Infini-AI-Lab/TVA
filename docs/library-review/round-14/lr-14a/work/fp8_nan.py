"""lr-14a: what the fp8 conversions give for every NaN pattern and for +-inf of float16 and
bfloat16 (all NaN patterns of each), and for NaN patterns of float32 (every payload of the top
12 bits and 2^16 random payloads, both signs). num-cvt8 runs only the canonical float32 NaN and
float32 +-inf; rules/num.md says "NaN gives the NaN 0x7f".
Run: CUDA_VISIBLE_DEVICES=0 python -I fp8_nan.py
"""
import torch
import triton
import triton.language as tl


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


def run(bits, src):
    n = bits.numel()
    e4 = torch.empty(n, dtype=torch.int8, device="cuda")
    e5 = torch.empty(n, dtype=torch.int8, device="cuda")
    cvt[(triton.cdiv(n, 1024),)](bits, e4, e5, n, SRC=src, B=1024)
    torch.cuda.synchronize()
    return (e4.to(torch.int64) & 0xff), (e5.to(torch.int64) & 0xff)


def show(name, bits, src):
    e4, e5 = run(bits, src)
    print(f"| {name} | {bits.numel()} | {sorted(set(e4.tolist()))} | {sorted(set(e5.tolist()))} |")


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("| inputs | count | float8e4nv results (bits) | float8e5 results (bits) |")
    print("|---|---|---|---|")
    for name, w, ebits, mbits, src, idt in [("float16", 16, 5, 10, 1, torch.int16), ("bfloat16", 16, 8, 7, 2, torch.int16)]:
        allp = torch.arange(0, 1 << w, dtype=torch.int64)
        e = (allp >> mbits) & ((1 << ebits) - 1)
        frac = allp & ((1 << mbits) - 1)
        sign = allp >> (w - 1)
        nan = (e == (1 << ebits) - 1) & (frac != 0)
        inf = (e == (1 << ebits) - 1) & (frac == 0)
        to16 = lambda t: torch.where(t >= (1 << 15), t - (1 << 16), t).to(idt).cuda()
        show(f"{name} NaN, sign +", to16(allp[nan & (sign == 0)]), src)
        show(f"{name} NaN, sign -", to16(allp[nan & (sign == 1)]), src)
        show(f"{name} +inf", to16(allp[inf & (sign == 0)]), src)
        show(f"{name} -inf", to16(allp[inf & (sign == 1)]), src)
    g = torch.Generator().manual_seed(14)
    top = torch.arange(1, 1 << 12, dtype=torch.int64) << 11          # payloads in the top 12 fraction bits
    rnd = torch.randint(1, 1 << 23, (1 << 16,), generator=g, dtype=torch.int64)
    for sgn in (0, 1):
        pats = (torch.cat([top, rnd]) | (0xFF << 23) | (sgn << 31))
        pats = torch.where(pats >= (1 << 31), pats - (1 << 32), pats).to(torch.int32).cuda()
        show(f"float32 NaN, sign {'+-'[sgn]}", pats, 0)


if __name__ == "__main__":
    main()
