"""lr-34b: PTX/SASS of `a / b + c` (fma_ptx.py form 8) in float16 and float32 (L155 survey)."""
import os
import subprocess
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fma_ptx import k  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
dt = {"f16": torch.float16, "f32": torch.float32}[sys.argv[1]]
N = 2
a = torch.ones(N, dtype=dt, device="cuda")
m = torch.ones(N, dtype=torch.int32, device="cuda")
o = torch.empty(N, dtype=dt, device="cuda")
h = k[(1,)](a, a, a, m, o, 8, N, num_warps=1)
print(h.asm["ttgir"].split("tt.func", 1)[1][:3000])
print(h.asm["ptx"].split(".entry", 1)[1])
p = os.path.join(HERE, "_k.cubin")
open(p, "wb").write(h.asm["cubin"])
print(subprocess.run([TOOL, "-sass", p], capture_output=True, text=True, timeout=60).stdout)
os.remove(p)
