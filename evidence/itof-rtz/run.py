"""TVA library evidence (library issue L24): an int-to-float value narrowed with
`fp_downcast_rounding="rtz"`.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/itof-rtz/run.py
Output: evidence/itof-rtz/out.txt (README.md says what it shows).

For every `int32` x (all 2^32, in chunks), the device computes
- `x.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtz")` and the same into `tl.float16`;
- `x.to(tl.float32).to(tl.bfloat16)` and `.to(tl.float16)`, the default rounding, as controls.
Each is compared, bit for bit, with
- the round-4 rule: `tl.itof` to `float32` (one rounding to nearest even), then `tl.cvt` toward
  zero into the narrow format;
- the round-3 rule: one `tl.itof` to the narrow format, which `tl.ItofRel` lets round once to
  nearest even, or twice through `float32` or `float64`.
The references are computed on the GPU with torch: the conversion of an `int32` to `float32` and
`float64` is IEEE (to nearest even); toward zero into `bfloat16` keeps the high 16 bits of the
`float32` pattern; toward zero into `float16` is the nearest `float16` stepped one place toward
zero when it is larger in magnitude (overflow gives the largest finite value, 65504).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def chains(x_ptr, bz_ptr, hz_ptr, bn_ptr, hn_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    f = x.to(tl.float32)
    tl.store(bz_ptr + offs, f.to(tl.bfloat16, fp_downcast_rounding="rtz"), mask=m)
    tl.store(hz_ptr + offs, f.to(tl.float16, fp_downcast_rounding="rtz"), mask=m)
    tl.store(bn_ptr + offs, x.to(tl.float32).to(tl.bfloat16), mask=m)
    tl.store(hn_ptr + offs, x.to(tl.float32).to(tl.float16), mask=m)


def round_int(x, p):
    """The integers `x` (int32) rounded once, to nearest even, to `p` significant bits, as float32
    (exact: at most 25 significant bits are kept and the value is at most 2^31)."""
    a = x.to(torch.int64).abs()
    _, e = torch.frexp(a.to(torch.float64))           # a = m 2^e, m in [1/2, 1): e significant bits
    shift = (e.to(torch.int64) - p).clamp(min=0)
    q = a >> shift
    r = a - (q << shift)
    half = torch.where(shift > 0, torch.ones_like(shift) << (shift - 1).clamp(min=0), torch.zeros_like(shift))
    up = (shift > 0) & ((r > half) | ((r == half) & (q % 2 == 1)))
    v = ((q + up.to(torch.int64)) << shift).to(torch.float64)
    return torch.where(x < 0, -v, v).to(torch.float32)


def rtz_f16(f):
    h = f.to(torch.float16)
    up = h.float().abs() > f.abs()
    return torch.where(up, torch.nextafter(h, torch.zeros_like(h)), h)


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    B, CH = 1024, 1 << 26
    k = chains.warmup(torch.empty(1, dtype=torch.int32, device="cuda"), *[torch.empty(1, device="cuda", dtype=d) for d in
                      (torch.bfloat16, torch.float16, torch.bfloat16, torch.float16)], 1, B, grid=(1,))
    ttir = k.asm["ttir"]
    print(f"TTIR: tt.fp_to_fp {ttir.count('tt.fp_to_fp')}, arith.sitofp {ttir.count('arith.sitofp')}, "
          f"arith.truncf {ttir.count('arith.truncf')}")
    tot = {key: 0 for key in ("bz_new", "bz_old", "hz_new", "hz_old", "bn_new_ctrl", "hn_new_ctrl")}
    examples = []
    for start in range(-(1 << 31), 1 << 31, CH):
        x = torch.arange(start, start + CH, dtype=torch.int64, device="cuda").to(torch.int32)
        outs = [torch.empty(CH, dtype=d, device="cuda") for d in (torch.bfloat16, torch.float16, torch.bfloat16, torch.float16)]
        chains[(CH // B,)](x, *outs, CH, B)
        torch.cuda.synchronize()
        bz, hz, bn, hn = (o.view(torch.int16) for o in outs)
        f32 = x.to(torch.float32)
        new_b = (f32.view(torch.int32) >> 16).to(torch.int16)          # rtz: the high half of the pattern
        new_h = rtz_f16(f32).view(torch.int16)
        # tl.ItofRel: one rounding of the integer to nearest even, or two through float32 or float64
        # (an int32 is exact in float64, so through float64 is the single rounding)
        old_b = [round_int(x, 8).to(torch.bfloat16).view(torch.int16), f32.to(torch.bfloat16).view(torch.int16)]
        old_h = [round_int(x, 11).to(torch.float16).view(torch.int16), f32.to(torch.float16).view(torch.int16)]
        tot["bz_new"] += int((bz != new_b).sum())
        tot["hz_new"] += int((hz != new_h).sum())
        bad_b = (bz != old_b[0]) & (bz != old_b[1])
        bad_h = (hz != old_h[0]) & (hz != old_h[1])
        tot["bz_old"] += int(bad_b.sum())
        tot["hz_old"] += int(bad_h.sum())
        tot["bn_new_ctrl"] += int(((bn != old_b[0]) & (bn != old_b[1])).sum())
        tot["hn_new_ctrl"] += int(((hn != old_h[0]) & (hn != old_h[1])).sum())
        if not examples and bad_b.any():
            i = int(bad_b.nonzero()[0])
            examples.append((int(x[i]), float(outs[0][i]), float(f32[i].to(torch.bfloat16))))
    n = 1 << 32
    print(f"int32 inputs: {n}")
    print(f"bfloat16, rtz chain vs the round-4 rule (itof to float32, then cvt toward zero): {tot['bz_new']} differ")
    print(f"float16,  rtz chain vs the round-4 rule: {tot['hz_new']} differ")
    print(f"bfloat16, rtz chain vs the round-3 rule (one tl.itof, tl.ItofRel): {tot['bz_old']} outside it")
    print(f"float16,  rtz chain vs the round-3 rule: {tot['hz_old']} outside it")
    print(f"controls, default rounding vs tl.ItofRel: bfloat16 {tot['bn_new_ctrl']} outside, float16 {tot['hn_new_ctrl']} outside")
    for x, got, near in examples:
        print(f"e.g. x = {x}: the rtz chain gives {got}, the nearest bfloat16 is {near}")


if __name__ == "__main__":
    main()
