"""Tutorial 09's partial tiles: `offs_am = tl.where(offs_am < M, offs_am, 0)` followed by
`tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)`.

With `M < BLOCK_SIZE_M` (the tutorial's own `validate(32, 32, 32)`), `offs_am` is
`[0, 1, …, M-1, 0, 0, …]`, which is not `BLOCK_SIZE_M / C` runs of `C` consecutive values: the
`max_contiguous` promise is false, and the library reads a false hint as undefined behaviour
(SEMANTICS "Compiler hints", `TVA.tl.max_contiguous`). This script runs `matmul_kernel` and
`matmul_kernel_persistent` through the tutorial's own host code at that size, and prints the
largest difference from torch, and the values of `offs_am` the hint is about.

Usage: CUDA_VISIBLE_DEVICES=0 ~/fpv/tva/.venv/bin/python -I t09_partial_tiles.py <tutorials dir>
"""
import importlib.util
import os
import sys

import torch

sys.dont_write_bytecode = True  # leave no cache in the tutorials folder

tut = os.path.join(sys.argv[1], "09-persistent-matmul.py")
spec = importlib.util.spec_from_file_location("t09", tut)
t09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t09)

torch.manual_seed(0)
for (M, N, K) in [(32, 32, 32), (200, 136, 64)]:
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    ref = (a.float() @ b.float())
    for name, fn in [("matmul_kernel", t09.matmul), ("matmul_kernel_persistent", t09.matmul_persistent)]:
        c = fn(a, b)
        torch.cuda.synchronize()
        print(f"M,N,K={M},{N},{K} {name}: max |c - ref| = {(c.float() - ref).abs().max().item():.3e}")
BM = 128
offs = torch.arange(BM)
offs = torch.where(offs < 32, offs, 0)
print("offs_am for M = 32, BLOCK_SIZE_M = 128:", offs[:4].tolist(), "...", offs[30:36].tolist(), "...")
