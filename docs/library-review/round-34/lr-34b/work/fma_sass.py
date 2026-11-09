"""lr-34b: SASS of the forms in fma_ptx.py whose PTX keeps a separate mul/div and add (L155)."""
import os
import re
import subprocess
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fma_ptx import k  # noqa: E402

TOOL = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
HERE = os.path.dirname(os.path.abspath(__file__))


def sass(form, dab, dc):
    N = 256
    a = torch.ones(N, dtype=dab, device="cuda")
    c = torch.ones(N, dtype=dc, device="cuda")
    m = torch.ones(N, dtype=torch.int32, device="cuda")
    o = torch.empty(N, dtype=dc if form in (7, 10, 11) else dab, device="cuda")
    h = k[(1,)](a, a, c, m, o, form, N)
    p = os.path.join(HERE, "_k.cubin")
    open(p, "wb").write(h.asm["cubin"])
    s = subprocess.run([TOOL, "-sass", p], capture_output=True, text=True, timeout=60).stdout
    os.remove(p)
    return [ln.strip() for ln in s.splitlines() if re.search(r"\b(F|D|H)(ADD|FMA|MUL|SEL)|MUFU|FSEL|HADD2|HFMA2|HMUL2|F2F|FCHK", ln)]


for form, dab, dc in [(5, torch.float32, torch.float32), (6, torch.float32, torch.float32), (9, torch.float32, torch.float32),
                      (8, torch.float32, torch.float32), (7, torch.float16, torch.float32), (7, torch.float32, torch.float64),
                      (5, torch.float16, torch.float16)]:
    print("FORM", form, str(dab)[6:], str(dc)[6:])
    for ln in sass(form, dab, dc):
        print("   ", ln)
