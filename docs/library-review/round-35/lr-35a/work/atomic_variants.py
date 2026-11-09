"""lr-35a (library round 35, L161): other shapes of launch B of evidence/atomic-result-order.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
One program, 512 lanes, two atomic_add(+1) calls on one cell Y0[0], element 0 delayed by a chain of
loads (as in evidence/atomic-result-order). The semantics: every lane of the first call is performed
before a step that uses its result, and that step before every lane of the second call, so the first
call's old values are all < 512 and the second's all >= 512. Variants:
  barrier: tl.debug_barrier() between the store of r and the second call;
  loop:    the same call run twice in a loop, each result stored;
  addr:    the first result used only in an address (tl.store(OUT + r, x)), the second stored;
  unused2: the first result stored, the second call's result unused; the first call's values are
           then read back, the semantics still has every r < 512 (the second call follows the store).
"""
import sys
import torch, triton, triton.language as tl


@triton.jit
def kern(Y0, OUT, OUT2, P, N: tl.constexpr, BLOCK: tl.constexpr, V: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = offs
    for i in range(N):
        x = tl.load(P + x)
    if V == 0:      # barrier
        r = tl.atomic_add(Y0 + 0 * x, tl.full([BLOCK], 1, tl.int32))
        tl.store(OUT + x, r)
        tl.debug_barrier()
        s = tl.atomic_add(Y0 + 0 * x, tl.full([BLOCK], 1, tl.int32))
        tl.store(OUT2 + x, s)
    elif V == 1:    # loop
        for it in range(2):
            r = tl.atomic_add(Y0 + 0 * x, tl.full([BLOCK], 1, tl.int32))
            tl.store(OUT + it * BLOCK + x, r)
    elif V == 2:    # addr
        r = tl.atomic_add(Y0 + 0 * x, tl.full([BLOCK], 1, tl.int32))
        tl.store(OUT + r, x)
        s = tl.atomic_add(Y0 + 0 * x, tl.full([BLOCK], 1, tl.int32))
        tl.store(OUT2 + x, s)
    else:           # unused2
        r = tl.atomic_add(Y0 + 0 * x, tl.full([BLOCK], 1, tl.int32))
        tl.store(OUT + x, r)
        tl.atomic_add(Y0 + 0 * x, tl.full([BLOCK], 1, tl.int32))


def chain(L, n, dev, base):
    return torch.randperm(L // 32 - 64, device=dev)[:n].to(torch.int32) * 32 + base


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    BLOCK, L, N = 512, 1 << 25, 400
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    dev = "cuda"
    z = torch.zeros(1, device=dev, dtype=torch.int32)
    print("| variant | launches | outside the semantics | last: max first-call value, min second-call value |")
    print("|---|---|---|---|")
    for V, name in enumerate(["barrier", "loop", "addr", "unused2"]):
        bad, worst = 0, None
        for t in range(trials):
            P = torch.arange(L, dtype=torch.int32, device=dev)
            c = torch.cat([z, chain(L, N - 1, dev, 2048), z])
            P[c[:-1].long()] = c[1:]
            Y0 = torch.zeros(1, dtype=torch.int32, device=dev)
            OUT = torch.full((4 * BLOCK,), -1, dtype=torch.int32, device=dev)
            OUT2 = torch.full((BLOCK,), -1, dtype=torch.int32, device=dev)
            kern[(1,)](Y0, OUT, OUT2, P, N=N, BLOCK=BLOCK, V=V, num_warps=4)
            torch.cuda.synchronize()
            assert int(Y0[0]) == 2 * BLOCK
            o, o2 = OUT.cpu(), OUT2.cpu()
            if V == 0:
                first, second = o[:BLOCK], o2
            elif V == 1:
                first, second = o[:BLOCK], o[BLOCK:2 * BLOCK]
            elif V == 2:
                first = torch.nonzero(o >= 0).flatten()   # the addresses r the first call returned
                second = o2
            else:
                first, second = o[:BLOCK], torch.full((1,), 10 ** 6)
            assert first.numel() == BLOCK
            mx, mn = int(first.max()), int(second.min())
            if mx >= BLOCK or mn < BLOCK:
                bad += 1
                worst = (mx, mn)
        print(f"| {name} | {trials} | {bad} | {worst} |", flush=True)


if __name__ == "__main__":
    main()
