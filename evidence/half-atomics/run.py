"""TVA evidence for library issue L172 (round 38; library review lr-37b's work/half_atomics.py, round 37):
float16 / bfloat16 atomic_add at 2-byte granularity on the H200.

On sm_90 a float16 or bfloat16 `atom.add.noftz` lane compiles to a 32-bit `ATOMG.E.CAS` loop on
the aligned word holding the half (`sass_dump.py`), so it rewrites the neighbouring half.
  P. packing: a 1024-lane float16 / bfloat16 tensor atomic_add (8 elements per thread), with no mask
     and with a mask that disables odd lanes, and with the result used: the PTX and SASS atomics
     and the new values against one IEEE addition per lane (old + val in float64, rounded once).
  N. neighbours: 64 programs; program p < 32 does R rounds of atomic_add(+1) on the even halves of
     a 2048-half buffer; program p >= 32 does R rounds of plain tl.store of round-dependent values
     to the odd halves (each odd half written by one program only). The model has no race (other
     bytes). Checks: every even half = 32 * R (R = 64 for float16, 8 for bfloat16, so every partial sum is exact), every odd half = the
     last value its program stored. 20 launches, float16 and bfloat16.
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import os
import re
import subprocess
import tempfile

import torch
import triton
import triton.language as tl


@triton.jit
def k_pack(p, v, o, MASK: tl.constexpr, USE: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    x = tl.load(v + j)
    if MASK:
        r = tl.atomic_add(p + j, x, mask=(j % 2) == 0)
    else:
        r = tl.atomic_add(p + j, x)
    if USE:
        tl.store(o + j, r)


@triton.jit
def k_nb(buf, R: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    if pid < 32:
        for r in range(R):
            tl.atomic_add(buf + 2 * j, tl.full([BLOCK], 1.0, buf.dtype.element_ty), sem="relaxed")
    else:
        q = pid - 32
        cells = buf + 2 * (q * (BLOCK // 32) + j % (BLOCK // 32)) + 1
        mine = j < (BLOCK // 32)
        for r in range(R):
            tl.store(cells, ((r + 1) * 1.0 + q * 0.0).to(buf.dtype.element_ty), mask=mine)


def sass_atoms(h):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "k.cubin")
        with open(f, "wb") as fh:
            fh.write(h.asm["cubin"])
        out = subprocess.run([triton.knobs.nvidia.cuobjdump.path, "-sass", f], capture_output=True,
                             text=True).stdout
    return sorted(set(m.group(1) for m in re.finditer(r"\b((?:ATOMG|REDG|ATOM|RED)\.[A-Z0-9_.]+)", out)))


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    g = torch.Generator(device="cuda").manual_seed(37)
    N = 1024
    print("## P. packing")
    for dt in (torch.float16, torch.bfloat16):
        for mask in (False, True):
            for use in (False, True):
                old = (torch.randn(N, device="cuda", generator=g) * 50).to(dt)
                val = (torch.randn(N, device="cuda", generator=g) * 3).to(dt)
                p = old.clone()
                o = torch.empty_like(p)
                h = k_pack[(1,)](p, val, o, mask, use, N)
                torch.cuda.synchronize()
                exp = (old.double() + val.double()).to(dt)
                if mask:
                    exp = torch.where(torch.arange(N, device="cuda") % 2 == 0, exp, old)
                bad = int((p.view(torch.int16) != exp.view(torch.int16)).sum())
                rbad = ""
                if use:
                    sel = torch.arange(N, device="cuda") % 2 == 0 if mask else torch.ones(N, dtype=torch.bool, device="cuda")
                    rbad = f"; returned != old on {int((o.view(torch.int16) != old.view(torch.int16))[sel].sum())}"
                ptx = sorted(set(re.findall(r"\b(?:atom|red)\.[a-z0-9._]+", h.asm["ptx"])))
                print(f"{str(dt)[6:]} mask={mask} used={use}: PTX {ptx}; SASS {sass_atoms(h)}; "
                      f"new value != one IEEE add on {bad} of {N}{rbad}")
    print("## N. neighbours")
    BLOCK = 1024
    for dt, R in ((torch.float16, 64), (torch.bfloat16, 8)):   # 32 * R exact in the format
        bad_even = bad_odd = 0
        for t in range(20):
            buf = torch.zeros(2 * BLOCK, device="cuda", dtype=dt)
            k_nb[(64,)](buf, R, 2 * BLOCK, BLOCK)
            torch.cuda.synchronize()
            bad_even += int((buf[0::2].float() != 32.0 * R).sum())
            bad_odd += int((buf[1::2].float() != float(R)).sum())
        print(f"{str(dt)[6:]}: 20 launches: even halves != {32 * R} on {bad_even} of {20 * BLOCK}; "
              f"odd halves != {R} on {bad_odd} of {20 * BLOCK}")


if __name__ == "__main__":
    main()
