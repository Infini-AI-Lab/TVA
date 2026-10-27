"""lr-2b, library review round 2, semantics survey: smaller checks of SEMANTICS.md and TVA.Lang.

M1. `tl.max`/`tl.min` of a small integer tensor: SEMANTICS.md lists the reductions' promotions as
    "tl.max/tl.min reductions widen floats narrower than 32 bits to fp32; tl.sum widens only small
    integers". The kernel checks the result dtype with `tl.static_assert`, and shows the consequence
    of the dtype: `tl.max(x) + 100` for an `int8` `x` whose maximum is 100.
M2. Fusion across a widening: `(a * b).to(tl.float32) + c` with `float16`/`bfloat16` `a`, `b`. The
    library's fusable form `tl.addc` admits `round(p + c)` only for exact `p` that round *in float32*
    to the widened product; `fma(a, b, c)` in float32 is not of that form. Counts how many elements
    equal the unfused result and how many equal the float32 `fma`.
M3. Float `tl.atomic_max`/`tl.atomic_min` on signed zeros, negative values and NaNs, checked against
    `tl.fBound` (one of the two values, and when neither is NaN, not below/above either).

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-2/lr-2b/work/misc_probe.py
"""
import itertools
import math
import struct
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def max_small_int(x_ptr, out_ptr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, N))
    m = tl.max(x, axis=0)
    tl.static_assert(m.dtype == tl.int32)
    tl.store(out_ptr, m + 100)


@triton.jit
def widen_add(a_ptr, b_ptr, c_ptr, out_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m)
    b = tl.load(b_ptr + offs, mask=m)
    c = tl.load(c_ptr + offs, mask=m)
    tl.store(out_ptr + offs, (a * b).to(tl.float32) + c, mask=m)


@triton.jit
def amax(p_ptr, v_ptr, IS_MAX: tl.constexpr):
    i = tl.program_id(0)
    v = tl.load(v_ptr + i)
    if IS_MAX:
        tl.atomic_max(p_ptr + i, v)
    else:
        tl.atomic_min(p_ptr + i, v)


def m1():
    x = torch.tensor([100] + [-5] * 15, device=DEV, dtype=torch.int8)
    out = torch.zeros(1, device=DEV, dtype=torch.int32)
    try:
        max_small_int[(1,)](x, out, 16)
        torch.cuda.synchronize()
        print(f"M1 tl.max of an int8 tensor compiles with `tl.static_assert(m.dtype == tl.int32)`; "
              f"tl.max(x) + 100 = {int(out.item())} (int8 arithmetic would wrap to {((200 + 128) % 256) - 128})",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"M1 {type(e).__name__}: {str(e).strip().splitlines()[-1][:160]}", flush=True)


def m2():
    n, B = 1 << 20, 1024
    g = torch.Generator(device="cpu").manual_seed(2)
    for dt in (torch.float16, torch.bfloat16):
        a = torch.randn(n, generator=g).to(dt).to(DEV)
        b = torch.randn(n, generator=g).to(dt).to(DEV)
        # c cancels the product, so that the two roundings differ often
        c = (-(a.float() * b.float()) + torch.randn(n, generator=g).to(DEV) * 1e-3).float()
        for fusion in (True, False):
            out = torch.empty(n, device=DEV, dtype=torch.float32)
            k = widen_add[(triton.cdiv(n, B),)](a, b, c, out, n, B, enable_fp_fusion=fusion)
            torch.cuda.synchronize()
            unfused = (a * b).float() + c                      # IEEE product in dt, widened, IEEE sum
            fused = torch.addcmul(c.double(), a.double(), b.double()).float()  # a*b exact in fp64, one rounding
            differ = unfused != fused
            print(f"M2 {str(dt).replace('torch.', ''):8s} fusion={str(fusion):5s}: of {int(differ.sum())} elements where "
                  f"the two differ, the device gives the unfused result at {int((out == unfused)[differ].sum())} and "
                  f"the float32 fma at {int((out == fused)[differ].sum())}; "
                  f"fma.rn.f32 in PTX: {'fma.rn.f32' in k.asm['ptx']}", flush=True)


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def le(x, y):
    return x <= y


def m3():
    vals = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x00000001, 0x80000001, 0x7F800000, 0xFF800000,
            0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001]
    pairs = list(itertools.product(vals, vals))
    for is_max in (True, False):
        olds = torch.tensor([o for o, _ in pairs], dtype=torch.int64).to(torch.int32).view(torch.float32)
        vs = torch.tensor([v for _, v in pairs], dtype=torch.int64).to(torch.int32).view(torch.float32)
        p = olds.clone().to(DEV)
        amax[(len(pairs),)](p, vs.to(DEV), is_max)
        torch.cuda.synchronize()
        res = p.cpu().view(torch.int32).tolist()
        bad = []
        for (o, v), r in zip(pairs, res):
            r &= 0xFFFFFFFF
            ok = r in (o, v)
            fo, fv, fr = f32(o), f32(v), f32(r)
            if ok and not (math.isnan(fo) or math.isnan(fv)):
                ok = (le(fo, fr) and le(fv, fr)) if is_max else (le(fr, fo) and le(fr, fv))
            if not ok:
                bad.append(f"{o:08x},{v:08x}->{r:08x}")
        print(f"M3 float32 atomic_{'max' if is_max else 'min'}: {len(pairs)} pairs, {len(bad)} outside tl.fBound"
              f"{': ' + ' '.join(bad[:8]) if bad else ''}", flush=True)


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}", flush=True)
    m1()
    m2()
    m3()


if __name__ == "__main__":
    main()
