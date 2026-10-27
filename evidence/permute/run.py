"""TVA library evidence (library issue L17): `tl.permute` and `tl.trans` of any rank.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/permute/run.py
Output: evidence/permute/out.txt (README.md says what it shows).

One program loads a block of distinct values (the element's row-major position), permutes it with
`tl.permute(x, dims)`, `tl.trans(x, dims)`, `tl.trans(x)` (no dims) or `x.T`, and stores the result
in row-major order. The script compares the stored block with `torch.permute`, whose definition is
the library's: output dimension `k` is input dimension `dims[k]`, and `out[o] = x[i]` with
`i[dims[k]] = o[k]`. Every permutation of rank-3 blocks `[2, 4, 8]` and `[16, 32, 64]` and of
rank-4 blocks `[2, 2, 4, 8]` and `[4, 8, 16, 32]` is tried (the larger blocks span several warps).
"""
import itertools
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def perm2(x_ptr, y_ptr, S0: tl.constexpr, S1: tl.constexpr, MODE: tl.constexpr):
    a0 = tl.arange(0, S0)[:, None]
    a1 = tl.arange(0, S1)[None, :]
    x = tl.load(x_ptr + a0 * S1 + a1)
    if MODE == 0:
        y = tl.trans(x)
    else:
        y = x.T
    b0 = tl.arange(0, S1)[:, None]
    b1 = tl.arange(0, S0)[None, :]
    tl.store(y_ptr + b0 * S0 + b1, y)


@triton.jit
def perm3(x_ptr, y_ptr, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, D0: tl.constexpr,
          D1: tl.constexpr, D2: tl.constexpr, O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
          MODE: tl.constexpr):
    a0 = tl.arange(0, S0)[:, None, None]
    a1 = tl.arange(0, S1)[None, :, None]
    a2 = tl.arange(0, S2)[None, None, :]
    x = tl.load(x_ptr + a0 * (S1 * S2) + a1 * S2 + a2)
    if MODE == 0:
        y = tl.permute(x, (D0, D1, D2))
    elif MODE == 1:
        y = tl.trans(x, (D0, D1, D2))
    elif MODE == 2:
        y = tl.permute(x, D0, D1, D2)
    else:
        y = tl.trans(x)
    b0 = tl.arange(0, O0)[:, None, None]
    b1 = tl.arange(0, O1)[None, :, None]
    b2 = tl.arange(0, O2)[None, None, :]
    tl.store(y_ptr + b0 * (O1 * O2) + b1 * O2 + b2, y)


@triton.jit
def perm4(x_ptr, y_ptr, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
          D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, O0: tl.constexpr,
          O1: tl.constexpr, O2: tl.constexpr, O3: tl.constexpr, MODE: tl.constexpr):
    a0 = tl.arange(0, S0)[:, None, None, None]
    a1 = tl.arange(0, S1)[None, :, None, None]
    a2 = tl.arange(0, S2)[None, None, :, None]
    a3 = tl.arange(0, S3)[None, None, None, :]
    x = tl.load(x_ptr + a0 * (S1 * S2 * S3) + a1 * (S2 * S3) + a2 * S3 + a3)
    if MODE == 0:
        y = tl.permute(x, (D0, D1, D2, D3))
    elif MODE == 1:
        y = tl.trans(x, (D0, D1, D2, D3))
    elif MODE == 2:
        y = tl.permute(x, D0, D1, D2, D3)
    else:
        y = tl.trans(x)
    b0 = tl.arange(0, O0)[:, None, None, None]
    b1 = tl.arange(0, O1)[None, :, None, None]
    b2 = tl.arange(0, O2)[None, None, :, None]
    b3 = tl.arange(0, O3)[None, None, None, :]
    tl.store(y_ptr + b0 * (O1 * O2 * O3) + b1 * (O2 * O3) + b2 * O3 + b3, y)


MODES = {0: "tl.permute(x, dims)", 1: "tl.trans(x, dims)", 2: "tl.permute(x, *dims)", 3: "tl.trans(x)"}


def check(shape, dims, mode):
    n = 1
    for s in shape:
        n *= s
    x = torch.arange(n, device=DEV, dtype=torch.int32).reshape(shape)
    out_shape = [shape[d] for d in dims]
    y = torch.empty(out_shape, device=DEV, dtype=torch.int32)
    if len(shape) == 3:
        perm3[(1, )](x, y, *shape, *dims, *out_shape, mode)
    else:
        perm4[(1, )](x, y, *shape, *dims, *out_shape, mode)
    torch.cuda.synchronize()
    return bool(torch.equal(y, x.permute(*dims)))


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}")
    for shape in ([2, 4, 8], [16, 32, 64], [2, 2, 4, 8], [4, 8, 16, 32]):
        r = len(shape)
        print(f"\n== Rank {r}, shape {shape}: every permutation ==")
        for mode in (0, 1, 2):
            perms = list(itertools.permutations(range(r)))
            good = [p for p in perms if check(shape, list(p), mode)]
            bad = [p for p in perms if p not in good]
            print(f"{MODES[mode]:22}: {len(good)} of {len(perms)} permutations equal torch.permute"
                  + (f"; differ: {bad}" if bad else ""))
        swap = list(range(r - 2)) + [r - 1, r - 2]
        print(f"{MODES[3]:22}: equals torch.permute with dims {swap} (the last two axes swapped): "
              f"{check(shape, swap, 3)}")

    print("\n== Rank 2, shape [8, 16] ==")
    x = torch.arange(128, device=DEV, dtype=torch.int32).reshape(8, 16)
    for mode, name in ((0, "tl.trans(x)"), (1, "x.T")):
        y = torch.empty(16, 8, device=DEV, dtype=torch.int32)
        perm2[(1, )](x, y, 8, 16, mode)
        torch.cuda.synchronize()
        print(f"{name:22}: equals torch.permute with dims [1, 0]: {bool(torch.equal(y, x.permute(1, 0)))}")


if __name__ == "__main__":
    main()
