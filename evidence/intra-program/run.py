"""TVA library evidence (library issue L15): conflicting global accesses within one program.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/intra-program/run.py
Output: evidence/intra-program/out.txt (README.md says what it shows).

A Triton program is many threads. Nothing in the source orders one thread's access to a global
address before another thread's access to the same address in the same program. Each part below
counts results that a program run in source order could not give.

Part A, one block written and then read back. Each program stores its block of `x` to `buf`
(which starts at a sentinel), then loads the block back and stores what it read to `out`.
- `reversed`: reads `buf[B - 1 - j]`, an element another thread wrote. In source order no sentinel
  can be read.
- `reversed + debug_barrier`: the same with `tl.debug_barrier()` between the two steps.
- `same offsets`: reads `buf[j]`, the element the same thread wrote.
- `war`: `buf[j] = buf[B - 1 - j] + 1` in place (loads first in source order).
- `waw`: stores `y` at `j`, then `y + 1` at `B - 1 - j` (in source order the second store wins).
Before the store, each value goes through a few dependent operations, so that warps reach it at
different times.

Part B, a loop that reads what the previous iteration stored. Iteration `t` loads row `t` of
`buf` and stores it plus one into row `t + 1` (same offsets, so each thread reads its own writes
when the loop runs in order). In source order row `t` ends as `t`. Variants: `range` and
`tl.range(..., num_stages=3)`, each at `num_stages` 1 and 3, the second also with a
`tl.debug_barrier()` after the store; and a loop whose row goes through `tl.dot` with the identity
(Triton pipelines loads that feed a dot) at `num_stages` 1, 3 (the default) and 4.

Part C, two atomics on the same address in one program. `tl.atomic_xchg(p + j, 1)` and then
`tl.atomic_xchg(p + (B - 1 - j), 2)`: each address gets one exchange from each operation, from
different lanes. In source order every address ends as `2`; an address that ends as `1` had the
second operation's exchange first. And `o1 = tl.atomic_add(q + j, 1)`, then
`o2 = tl.atomic_add(q + (B - 1 - j), 1)`: in source order `o2` is `1` everywhere; `o2 = 0` means the
second add came first.
"""
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"
SENTINEL = -12345.0


@triton.jit
def skew(x):
    y = x
    for _ in range(8):
        y = tl.where(y > 1e30, y, y * 1.0000001 + 0.25)
    return y


@triton.jit
def block_rw(x_ptr, buf_ptr, out_ptr, B: tl.constexpr, MODE: tl.constexpr):
    base = tl.program_id(0) * B
    j = tl.arange(0, B)
    y = skew(tl.load(x_ptr + base + j))
    if MODE == 3:  # war
        r = tl.load(buf_ptr + base + (B - 1 - j))
        tl.store(buf_ptr + base + j, r + 1.0)
    elif MODE == 4:  # waw
        tl.store(buf_ptr + base + j, y)
        tl.store(buf_ptr + base + (B - 1 - j), y + 1.0)
    else:
        tl.store(buf_ptr + base + j, y)
        if MODE == 1:
            tl.debug_barrier()
        if MODE == 2:
            r = tl.load(buf_ptr + base + j)
        else:
            r = tl.load(buf_ptr + base + (B - 1 - j))
        tl.store(out_ptr + base + j, r)


@triton.jit
def chain(buf_ptr, n, B: tl.constexpr, RANGE: tl.constexpr, BARRIER: tl.constexpr):
    j = tl.arange(0, B)
    if RANGE:
        for t in tl.range(0, n, num_stages=3):
            x = tl.load(buf_ptr + t * B + j)
            tl.store(buf_ptr + (t + 1) * B + j, x + 1.0)
            if BARRIER:
                tl.debug_barrier()
    else:
        for t in range(0, n):
            x = tl.load(buf_ptr + t * B + j)
            tl.store(buf_ptr + (t + 1) * B + j, x + 1.0)


@triton.jit
def chain_dot(buf_ptr, w_ptr, n, B: tl.constexpr):
    r = tl.arange(0, B)
    tile = r[:, None] * B + r[None, :]
    w = tl.load(w_ptr + tile)
    for t in range(0, n):
        x = tl.load(buf_ptr + t * B * B + tile)
        tl.store(buf_ptr + (t + 1) * B * B + tile, tl.dot(x, w, input_precision="ieee") + 1.0)


@triton.jit
def two_xchg(p_ptr, x_ptr, B: tl.constexpr):
    base = tl.program_id(0) * B
    j = tl.arange(0, B)
    one = (skew(tl.load(x_ptr + base + j)) * 0.0 + 1.0).to(tl.int32)
    tl.atomic_xchg(p_ptr + base + j, one)
    tl.atomic_xchg(p_ptr + base + (B - 1 - j), one * 2)


@triton.jit
def two_add(q_ptr, x_ptr, o2_ptr, B: tl.constexpr):
    base = tl.program_id(0) * B
    j = tl.arange(0, B)
    one = (skew(tl.load(x_ptr + base + j)) * 0.0 + 1.0).to(tl.int32)
    tl.atomic_add(q_ptr + base + j, one)
    o2 = tl.atomic_add(q_ptr + base + (B - 1 - j), one)
    tl.store(o2_ptr + base + j, o2)


def main():
    print(f"triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, {time.strftime('%Y-%m-%d')}", flush=True)
    g = torch.Generator(device=DEV).manual_seed(20261027)

    print("\n== Part A: one block written, then read back ==")
    names = {0: "reversed", 1: "reversed + debug_barrier", 2: "same offsets", 3: "war (in place, reversed)",
             4: "waw (second store reversed)"}
    for B, warps, programs in ((1024, 4, 8192), (4096, 8, 4096)):
        x = torch.randn(programs * B, device=DEV, generator=g)
        # the values the program writes, as the device computes them: one `same offsets` run's buf
        ydev = torch.full_like(x, SENTINEL)
        block_rw[(programs,)](x, ydev, torch.empty_like(x), B, 2, num_warps=warps)
        torch.cuda.synchronize()
        assert int((ydev == SENTINEL).sum()) == 0
        for mode in range(5):
            buf = torch.full_like(x, SENTINEL)
            out = torch.full_like(x, SENTINEL)
            if mode == 3:
                buf = x.clone()
                start = buf.clone()
            block_rw[(programs,)](x, buf, out, B, mode, num_warps=warps)
            torch.cuda.synchronize()
            n = programs * B
            if mode in (0, 1):
                want = ydev.view(programs, B).flip(1).reshape(-1)
                bad = int((out != want).sum())
                note = f"{int((out == SENTINEL).sum())} of them the sentinel"
            elif mode == 2:
                bad = int((out != ydev).sum())
                note = f"{int((out == SENTINEL).sum())} of them the sentinel"
            elif mode == 3:
                want = start.view(programs, B).flip(1).reshape(-1) + 1.0
                bad = int((buf != want).sum())
                note = "elements of buf that are not the source-order result"
            else:
                want = (ydev + 1.0).view(programs, B).flip(1).reshape(-1)
                bad = int((buf != want).sum())
                note = "elements where the first store won"
            print(f"B={B:5d} num_warps={warps:2d} {names[mode]:28s}: {bad:9d} of {n} elements differ from source "
                  f"order ({bad / n:.1%}); {note}", flush=True)

    print("\n== Part B: a loop reading what the previous iteration stored ==")
    n, B = 64, 1024
    want = torch.arange(n + 1, device=DEV, dtype=torch.float32)[:, None].expand(n + 1, B)
    for rng, barrier, label in ((False, False, "range"), (True, False, "tl.range(num_stages=3)"),
                                (True, True, "tl.range(num_stages=3) + debug_barrier")):
        for stages in (1, 3):
            buf = torch.zeros((n + 1) * B, device=DEV)
            chain[(1,)](buf, n, B, rng, barrier, num_stages=stages)
            torch.cuda.synchronize()
            bad = int((buf.view(n + 1, B) != want).sum())
            print(f"{label:40s} num_stages={stages}: {bad:6d} of {(n + 1) * B} elements differ from source order "
                  f"({bad / ((n + 1) * B):.1%})", flush=True)
    B2 = 32
    want2 = torch.arange(n + 1, device=DEV, dtype=torch.float32)[:, None].expand(n + 1, B2 * B2)
    for stages in (1, 3, 4):
        buf = torch.zeros((n + 1) * B2 * B2, device=DEV)
        w = torch.eye(B2, device=DEV)
        k = chain_dot[(1,)](buf, w, n, B2, num_stages=stages)
        torch.cuda.synchronize()
        bad = int((buf.view(n + 1, B2 * B2) != want2).sum())
        print(f"{'range, row through tl.dot':40s} num_stages={stages}: {bad:6d} of {(n + 1) * B2 * B2} elements "
              f"differ from source order ({bad / ((n + 1) * B2 * B2):.1%}); cp.async in PTX: "
              f"{'cp.async' in k.asm['ptx']}", flush=True)

    print("\n== Part C: two atomics on the same address in one program ==")
    for B, warps, programs in ((1024, 4, 8192), (4096, 8, 2048)):
        x = torch.randn(programs * B, device=DEV, generator=g)
        reps = 10
        rev_x, rev_a = 0, 0
        for _ in range(reps):
            p = torch.zeros(programs * B, device=DEV, dtype=torch.int32)
            two_xchg[(programs,)](p, x, B, num_warps=warps)
            q = torch.zeros(programs * B, device=DEV, dtype=torch.int32)
            o2 = torch.full((programs * B,), -1, device=DEV, dtype=torch.int32)
            two_add[(programs,)](q, x, o2, B, num_warps=warps)
            torch.cuda.synchronize()
            assert int((q != 2).sum()) == 0
            rev_x += int((p == 1).sum())
            rev_a += int((o2 == 0).sum())
        tot = reps * programs * B
        print(f"B={B:5d} num_warps={warps:2d}: atomic_xchg pair: {rev_x} of {tot} addresses end as 1 (second "
              f"operation's exchange first); atomic_add pair: {rev_a} of {tot} second-add old values are 0 "
              f"(second operation's add first); every counter ends at 2", flush=True)


if __name__ == "__main__":
    main()
