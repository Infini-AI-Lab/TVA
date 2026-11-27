"""TVA library evidence, L40 (sm_89): which `k` each fp8 `tl.dot` instruction adds, and in what order.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/l40/num-dot-chunk8-order/run.py
Output (stdout): kept as evidence/l40/num-dot-chunk8-order/run.out.

The fp8 chunk facts (`DotF8ChunkLoss`, `TVA.Device.Dot8Chunk`) state the accumulation per chunk of 32
consecutive `k` (`k / 32 = j`), the chunks added in increasing order, each to the accumulator as it
stands. On the H200 that is `wgmma`'s `k32` instruction. On the L40 Triton 3.8.0 emits the native fp8
`mma.sync…m16n8k32…e4m3/e5m2…` (listed below), whose window keeps 13 bits below the largest term
(`evidence/num-dot` Part 1 on the L40: `s <= 13`).

**Probe.** For distinct chunks `x < y` and `k`, row `i` of `a` holds `+B` at `32x + o1`, `-B` at
`32y + o2` and `S` at `32k + o3` (random offsets `o`), `b` the matching ones, the accumulator 0. With
`B = 2^14`, `S = 1` (`float8e5`; `B = 2^8`, `S = 2^-6` for `float8e4nv`), `S` is 14 binades below
`B`: an instruction that adds `S` while `B` is in its window (a `B` in the same instruction, or the
accumulator holding `B`) drops it. So the result is nonzero exactly when `S`'s chunk is added after
both `B`'s, if every instruction adds one aligned chunk of 32 consecutive `k` and the chunks go in
increasing order: `nonzero == (k > y)`. The triples with `k` in its own chunk (`k` not `x` or `y`)
test the order of the chunks; those with `k = x` or `k = y` test whether one instruction is one
window (`S` shares an instruction with a `B`). Every triple of 8 chunks (`K = 256`) and a sample of 32 chunks
(`K = 1024`) where the shape fits; also with `k` inside the chunk of `x` or `y`.
"""
import random
import re
import torch
import triton
import triton.language as tl

DEV = "cuda"
random.seed(20261124)
FMT = {"float8e5": (torch.float8_e5m2, tl.float8e5, 2.0 ** 14, 1.0),
       "float8e4nv": (torch.float8_e4m3fn, tl.float8e4nv, 2.0 ** 8, 2.0 ** -6)}


@triton.jit
def dotk(a_ptr, b_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.zeros((M, N), dtype=tl.float32)
    o = tl.dot(a, b, acc)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], o)


def run(f, a, b, nw):
    dt = FMT[f][0]
    M, K = a.shape
    N = b.shape[1]
    o = torch.empty(M, N, device=DEV)
    h = dotk[(1,)](a.to(dt).contiguous(), b.to(dt).contiguous(), o, M, N, K, num_warps=nw)
    return o, sorted(set(re.findall(r"mma\.sync[\w.]*", h.asm["ptx"])))


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("| operands | M x N, warps | K | chunks | triples, k in its own chunk | agree | triples, k = x or y | agree | instructions |")
    print("|---|---|---|---|---|---|---|---|---|")
    bad = 0
    bad_same = 0
    for f, (_, _, B, S) in FMT.items():
        for (M, N, nw) in ((16, 16, 1), (32, 32, 2), (64, 64, 4), (16, 64, 1)):
            for nch in (2, 4, 8, 32):
                K = 32 * nch
                trip = [(x, y, k) for x in range(nch) for y in range(nch) for k in range(nch) if x < y]
                if len(trip) > 4000:
                    trip = random.sample(trip, 4000)
                agree, ins = [0, 0], None
                try:
                    for s0 in range(0, len(trip), M):
                        tt = trip[s0:s0 + M]
                        a = torch.zeros(M, K, device=DEV); b = torch.zeros(K, N, device=DEV)
                        for i, (x, y, k) in enumerate(tt):
                            o1, o2, o3 = random.sample(range(32), 3)
                            a[i, 32 * x + o1] = B; a[i, 32 * y + o2] = -B; a[i, 32 * k + o3] = S
                            b[32 * x + o1, i % N] = 1.0; b[32 * y + o2, i % N] = 1.0; b[32 * k + o3, i % N] = 1.0
                        # one row per column i % N: rows sharing a column would mix in b, so M <= N here
                        o, ins = run(f, a, b, nw)
                        for i, (x, y, k) in enumerate(tt):
                            agree[int(k in (x, y))] += int((o[i, i % N].item() != 0.0) == (k > y))
                except triton.runtime.errors.OutOfResources:
                    print(f"| {f} | {M} x {N}, {nw} | {K} | {nch} | rejected: out of resources on this device | | | | |")
                    continue
                n_own = sum(1 for (x, y, k) in trip if k not in (x, y))
                bad += n_own - agree[0]
                bad_same += (len(trip) - n_own) - agree[1]
                print(f"| {f} | {M} x {N}, {nw} | {K} | {nch} | {n_own} | {agree[0]} | {len(trip) - n_own} | {agree[1]} | {', '.join(ins)} |")
    print(f"\nTriples with k in its own chunk where the result disagrees with aligned chunks of 32 added in increasing order: {bad}")
    print(f"Triples with k = x or y where S survives a B in its chunk (the instruction is not one window): {bad_same}")


if __name__ == "__main__":
    main()
