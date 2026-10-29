"""evidence/float-layer, part E: the memory encoding of every dtype, library against device.

Write: a kernel stores values of dtype d (from random bit patterns) into a buffer of dtype d; a
second kernel reads that memory back byte by byte through a uint8 pointer. Read: random bytes
are written into memory; a kernel loads them as elements of dtype d and returns their bit
patterns (int1: the loaded boolean). Bytes.lean computes the same with the library's
DType.toBits / DType.ofBits and TVA.Math.Bytes.leBytes / leValue.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I bytes.py <work folder>
(it runs `lake env lean --run` in ../../lean; one result line per dtype is printed).
"""
import os
import subprocess
import sys

import numpy as np
import torch
import triton
import triton.language as tl

D = {  # name: (torch dtype, triton dtype, bytes, float?)
    "i1": (torch.bool, tl.int1, 1, False), "i8": (torch.int8, tl.int8, 1, False),
    "i16": (torch.int16, tl.int16, 2, False), "i32": (torch.int32, tl.int32, 4, False),
    "i64": (torch.int64, tl.int64, 8, False), "u8": (torch.uint8, tl.uint8, 1, False),
    "u16": (torch.uint16, tl.uint16, 2, False), "u32": (torch.uint32, tl.uint32, 4, False),
    "u64": (torch.uint64, tl.uint64, 8, False), "f16": (torch.float16, tl.float16, 2, True),
    "bf16": (torch.bfloat16, tl.bfloat16, 2, True), "f32": (torch.float32, tl.float32, 4, True),
    "f64": (torch.float64, tl.float64, 8, True), "f8e5": (torch.float8_e5m2, tl.float8e5, 1, True),
    "f8e4nv": (torch.float8_e4m3fn, tl.float8e4nv, 1, True),
}
UT = {1: tl.uint8, 2: tl.uint16, 4: tl.uint32, 8: tl.uint64}


@triton.jit
def store_k(bits_ptr, buf_ptr, n, DT: tl.constexpr, U: tl.constexpr, NB: tl.constexpr,
            ISF: tl.constexpr, ISB: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    b = tl.load(bits_ptr + offs, mask=m)
    if ISB:
        v = b != 0
    elif ISF:
        if NB == 8:
            v = b.to(tl.float64, bitcast=True)
        else:
            v = b.to(U).to(DT, bitcast=True)
    else:
        v = b.to(DT)
    tl.store(buf_ptr + offs, v, mask=m)


@triton.jit
def bytes_k(src_ptr, out_ptr, n, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(src_ptr + offs, mask=m).to(tl.int32), mask=m)


@triton.jit
def load_k(buf_ptr, out_ptr, n, DT: tl.constexpr, U: tl.constexpr, NB: tl.constexpr,
           ISF: tl.constexpr, ISB: tl.constexpr, B: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    v = tl.load(buf_ptr + offs, mask=m)
    if ISB:
        r = v.to(tl.int64)
    elif ISF:
        if NB == 8:
            r = v.to(tl.int64, bitcast=True)
        else:
            r = v.to(U, bitcast=True).to(tl.int64)
    else:
        if NB == 8:
            r = v.to(tl.int64, bitcast=True)
        else:
            r = v.to(U, bitcast=True).to(tl.int64)
    tl.store(out_ptr + offs, r, mask=m)


def main():
    work = sys.argv[1]
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(20261028)
    N = 512
    lines, expect = [], []
    for name, (tdt, tldt, nb, isf) in D.items():
        # write
        if name == "i1":
            pats = np.array([0, 1] * (N // 2), dtype=np.uint64)
        else:
            pats = rng.integers(0, 1 << 63, N, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, N, dtype=np.uint64)
            pats = pats & np.uint64((1 << (8 * nb)) - 1 if nb < 8 else (1 << 64) - 1)
        bits = torch.from_numpy(pats.view(np.int64)).cuda()
        buf = torch.zeros(N, dtype=tdt, device="cuda")
        store_k[(1,)](bits, buf, N, DT=tldt, U=UT[nb], NB=nb, ISF=isf, ISB=name == "i1", B=N)
        raw = buf.view(torch.uint8)
        got = torch.empty(raw.numel(), dtype=torch.int32, device="cuda")
        bytes_k[(triton.cdiv(raw.numel(), 1024),)](raw, got, raw.numel(), B=1024)
        got = got.cpu().numpy().reshape(N, nb)
        for p, g in zip(pats.tolist(), got):
            lines.append(f"w {name} {p:x}")
            expect.append(("w", name, f"{p:x}", " ".join(f"{int(x):02x}" for x in g)))
        # read
        rb = rng.integers(0, 256, N * nb, dtype=np.uint8)
        if name == "i1":
            rb = np.arange(N, dtype=np.uint64).astype(np.uint8)
        mem = torch.from_numpy(rb).cuda()
        out = torch.empty(N, dtype=torch.int64, device="cuda")
        load_k[(1,)](mem.view(tdt), out, N, DT=tldt, U=UT[nb], NB=nb, ISF=isf, ISB=name == "i1", B=N)
        for k, v in enumerate(out.cpu().tolist()):
            bs = "".join(f"{int(x):02x}" for x in rb[k * nb:(k + 1) * nb])
            lines.append(f"r {name} {bs}")
            expect.append(("r", name, bs, f"{v & ((1 << 64) - 1):x}"))
    inp = os.path.join(os.path.abspath(work), "bytes.in")
    with open(inp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([os.path.expanduser("~/.elan/bin/lake"), "env", "lean", "--run",
                        os.path.join(here, "Bytes.lean"), inp],
                       cwd=os.path.join(here, "..", "..", "lean"), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)
    lib = r.stdout.splitlines()
    assert len(lib) == len(expect), (len(lib), len(expect))
    per = {}
    for (kind, name, arg, dev), l in zip(expect, lib):
        k = (name, kind)
        ok, bad = per.get(k, (0, []))
        if l.strip() == dev:
            per[k] = (ok + 1, bad)
        else:
            per[k] = (ok, bad + [f"{arg}: library {l.strip()}, device {dev}"])
    for name in D:
        for kind in ("w", "r"):
            ok, bad = per[(name, kind)]
            what = "write (bytes in memory)" if kind == "w" else "read (value of bytes)"
            print(f"E.{name:7s} {what:24s} {ok + len(bad):4d} values  "
                  f"{'agree' if not bad else 'DIFFER: ' + '; '.join(bad[:3])}")


if __name__ == "__main__":
    main()
