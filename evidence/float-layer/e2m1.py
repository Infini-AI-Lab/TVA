"""evidence/float-layer, part A: the values of the 16 E2M1 (fp4) patterns.

Triton has no fp4 dtype; fp4 operands only enter `tl.dot_scaled`. A dot of fp4 rows (all scales
1, that is e8m0 127) with the bfloat16 identity gives each fp4 value converted exactly to float32 (a zero as +0: the dot adds +0 products).
The library's values are `Format.cvt e2m1 binary32` of each pattern (driver spec
`cvt:e2m1:f32:rn:0`).

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I e2m1.py <build folder>
"""
import os
import struct
import subprocess
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def k(a_ptr, as_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk2 = tl.arange(0, K // 2)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * (K // 2) + rk2[None, :])
    sa = tl.load(as_ptr + rm[:, None] * (K // 32) + tl.arange(0, K // 32)[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.dot_scaled(a, sa, "e2m1", b, None, "bf16")
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


def main():
    driver = os.path.join(os.path.abspath(sys.argv[1]), "driver")
    M, N, K = 16, 32, 32
    # row r holds pattern (r + j) mod 16 at column j: low nibble first in each byte
    codes = torch.tensor([[(r + j) % 16 for j in range(K)] for r in range(M)], dtype=torch.int32)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).cuda()
    scales = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")
    eye = torch.eye(K, N, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(M, N, dtype=torch.float32, device="cuda")
    k[(1,)](packed, scales, eye, c, M=M, N=N, K=K)
    dev = c.view(torch.int32).cpu()
    r = subprocess.run([driver, "cvt:e2m1:f32:rn:0", "dump-all", "0", "16"], capture_output=True,
                       text=True, check=True)
    lib = {int(a): int(b, 16) for a, b in (l.split() for l in r.stdout.splitlines())}
    bad = []
    for i in range(M):
        for j in range(N):
            p = int(codes[i, j])
            d = int(dev[i, j]) & 0xffffffff
            # a dot's sum of products gives +0 for -0 (IEEE: -0 + +0 = +0), so zeros compare as values
            if d != lib[p] and not (d & 0x7fffffff == 0 and lib[p] & 0x7fffffff == 0):
                bad.append(f"pattern {p:x}: library {lib[p]:x}, device {d:x}")
    vals = ", ".join(f"{p:x}:{struct.unpack('<f', lib[p].to_bytes(4, 'little'))[0]:g}" for p in range(16))
    print(f"A.e2m1.f32 (tl.dot_scaled)   16 patterns x {M * N // 16} positions  "
          f"{'agree' if not bad else 'DIFFER: ' + '; '.join(bad[:4])}  [{vals}]")


if __name__ == "__main__":
    main()
