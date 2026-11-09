"""lr-37b: print the SASS of a float16 / bfloat16 tensor atomic_add (result unused), 64 lanes.
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file> <float16|bfloat16>"""
import os, re, subprocess, sys, tempfile
import torch, triton, triton.language as tl


@triton.jit
def k(p, v, N: tl.constexpr):
    j = tl.arange(0, N)
    tl.atomic_add(p + j, tl.load(v + j))


dt = getattr(torch, sys.argv[1])
p = torch.zeros(64, device="cuda", dtype=dt)
h = k[(1,)](p, torch.ones_like(p), 64)
with tempfile.TemporaryDirectory() as d:
    f = os.path.join(d, "k.cubin")
    open(f, "wb").write(h.asm["cubin"])
    out = subprocess.run([triton.knobs.nvidia.cuobjdump.path, "-sass", f], capture_output=True, text=True).stdout
for line in out.splitlines():
    m = re.search(r"/\*[0-9a-f]{4}\*/\s+(.*?;)", line)
    if m:
        print(m.group(1))
