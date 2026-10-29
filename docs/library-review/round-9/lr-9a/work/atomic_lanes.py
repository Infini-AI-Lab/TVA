"""lr-9a: one rmw per enabled lane, as the shared-atomics kit (TVA.Lemmas.SharedAcc, SafeA.atomic)
and SEMANTICS.md ("Atomics") say, also when Triton's layout holds an element in several threads
(a tensor smaller than the number of threads, a broadcast tensor).

Each program adds 1.0 (float32) and 1 (int32) to cells through `tl.atomic_add` on tensors of
1 to 512 elements (1-D, and 2-D with a broadcast), with and without a mask, num_warps 1 to 8,
and a scalar atomic; every enabled cell must be incremented exactly once per program.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I atomic_lanes.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add1d(f_ptr, i_ptr, n, B: tl.constexpr, MASKED: tl.constexpr):
    offs = tl.arange(0, B)
    if MASKED:
        m = offs < n
        tl.atomic_add(f_ptr + offs, tl.full([B], 1.0, tl.float32), mask=m)
        tl.atomic_add(i_ptr + offs, tl.full([B], 1, tl.int32), mask=m)
    else:
        tl.atomic_add(f_ptr + offs, tl.full([B], 1.0, tl.float32))
        tl.atomic_add(i_ptr + offs, tl.full([B], 1, tl.int32))


@triton.jit
def add2d(f_ptr, i_ptr, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)[:, None]
    c = tl.arange(0, C)[None, :]
    offs = r * C + c
    tl.atomic_add(f_ptr + offs, tl.full([R, C], 1.0, tl.float32))
    tl.atomic_add(i_ptr + offs, tl.full([R, C], 1, tl.int32))


@triton.jit
def add_scalar(f_ptr, i_ptr):
    tl.atomic_add(f_ptr, 1.0)
    tl.atomic_add(i_ptr, 1)


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    G = 64
    bad = 0
    rows = 0
    for nw in [1, 2, 4, 8]:
        for B in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            for masked in [False, True]:
                n = max(1, B - 3) if masked else B
                f = torch.zeros(B, device="cuda"); i = torch.zeros(B, dtype=torch.int32, device="cuda")
                add1d[(G,)](f, i, n, B=B, MASKED=masked, num_warps=nw)
                want = torch.zeros(B, device="cuda"); want[:n] = G
                ok = bool(torch.equal(f, want)) and bool(torch.equal(i, want.to(torch.int32)))
                rows += 1
                if not ok:
                    bad += 1
                    print(f"1-D B={B} masked={masked} num_warps={nw}: float {f[:4].tolist()} int {i[:4].tolist()}")
        for (R, C) in [(1, 16), (16, 1), (2, 8), (8, 2), (4, 4), (16, 16), (1, 1), (32, 4)]:
            f = torch.zeros(R * C, device="cuda"); i = torch.zeros(R * C, dtype=torch.int32, device="cuda")
            add2d[(G,)](f, i, R=R, C=C, num_warps=nw)
            ok = bool((f == G).all()) and bool((i == G).all())
            rows += 1
            if not ok:
                bad += 1
                print(f"2-D {R}x{C} num_warps={nw}: float {f[:4].tolist()} int {i[:4].tolist()}")
        f = torch.zeros(1, device="cuda"); i = torch.zeros(1, dtype=torch.int32, device="cuda")
        add_scalar[(G,)](f, i, num_warps=nw)
        rows += 1
        if not (f.item() == G and i.item() == G):
            bad += 1
            print(f"scalar num_warps={nw}: float {f.item()} int {i.item()}")
    print(f"configurations: {rows}; with a cell not incremented exactly once per program: {bad}")


if __name__ == "__main__":
    main()
