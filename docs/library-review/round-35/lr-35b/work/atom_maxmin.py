"""lr-35b (library issue L164): what a float `tl.atomic_max` / `tl.atomic_min` compiles to, and what it
does with a NaN, a -0.0 and a negative operand on the device (Triton 3.8.0, sm_90).

Run from the TVA root:  CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
Output: atom_maxmin.txt beside this file."""
import os
import re
import struct
import subprocess

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
CUOBJDUMP = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


@triton.jit
def k(p, v, o, MAX: tl.constexpr, N: tl.constexpr):
    j = tl.arange(0, N)
    if MAX:
        r = tl.atomic_max(p + j, tl.load(v + j), sem="relaxed")
    else:
        r = tl.atomic_min(p + j, tl.load(v + j), sem="relaxed")
    tl.store(o + j, r)


def bits32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f_of(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def sass(h):
    path = os.path.join(HERE, "_k.cubin")
    with open(path, "wb") as fh:
        fh.write(h.asm["cubin"])
    try:
        return subprocess.run([CUOBJDUMP, "-sass", path], capture_output=True, text=True, timeout=60).stdout
    finally:
        os.remove(path)


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    NAN, NNAN = 0x7fc00000, 0xffc00000
    vals = [NAN, NNAN, bits32(-0.0), bits32(0.0), bits32(-1.0), bits32(2.0), 0x7f800000, 0xff800000]
    olds = [bits32(1.0), bits32(-1.0), bits32(0.0), bits32(-0.0), NAN, NNAN, bits32(-3.0), bits32(5.0)]
    pairs = [(o, v) for o in olds for v in vals]
    N = 64
    assert len(pairs) == N
    for is_max in (True, False):
        name = "atomic_max" if is_max else "atomic_min"
        p = torch.tensor([o for o, _ in pairs], dtype=torch.int64).to(torch.int32).view(torch.float32).cuda()
        v = torch.tensor([x for _, x in pairs], dtype=torch.int64).to(torch.int32).view(torch.float32).cuda()
        o = torch.empty(N, dtype=torch.float32, device="cuda")
        h = k[(1,)](p, v, o, is_max, N)
        torch.cuda.synchronize()
        ptx = h.asm["ptx"]
        atoms = re.findall(r"(@%p\d+ atom\.[^;]*;)", ptx)
        setps = re.findall(r"setp\.[^;]*;", ptx)
        out(f"## float32 {name}, sem=relaxed")
        out("PTX atoms: " + " | ".join(a.strip() for a in atoms))
        out("PTX setp: " + " | ".join(s.strip() for s in setps))
        s = sass(h)
        out("SASS atoms: " + " | ".join(re.findall(r"(@!?P\d\s+ATOMG[^;]*;|\bATOMG[^;]*;)", s)))
        newv = p.view(torch.int32).cpu().tolist()
        ret = o.view(torch.int32).cpu().tolist()
        out("| old | operand | new (device) | new is old | new is operand | returned is old |")
        out("|---|---|---|---|---|---|")
        bad = 0
        for i, (ob, vb) in enumerate(pairs):
            nb = newv[i] & 0xffffffff
            rb = ret[i] & 0xffffffff
            if nb not in (ob, vb):
                bad += 1
            if vb in (NAN, NNAN) or ob in (NAN, NNAN) or vb == bits32(-0.0) or i % 9 == 0:
                out(f"| {ob:#010x} ({f_of(ob)}) | {vb:#010x} ({f_of(vb)}) | {nb:#010x} | {nb == ob} | {nb == vb} | {rb == ob} |")
        out(f"new value neither old nor operand: {bad} of {N}")
        nan_writes = sum(1 for i, (ob, vb) in enumerate(pairs) if vb in (NAN, NNAN) and (newv[i] & 0xffffffff) == vb
                         and vb != ob)
        out(f"NaN operands that were written into the cell: {nan_writes} of {sum(1 for _, vb in pairs if vb in (NAN, NNAN))}")
        out("")
    with open(os.path.join(HERE, "atom_maxmin.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
