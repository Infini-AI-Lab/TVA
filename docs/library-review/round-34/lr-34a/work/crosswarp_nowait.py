"""lr-34a: crosswarp.py without a program that waits on another (docs/exclusions.md excludes those).

Program 0 as in crosswarp.py: a dependent chain of N loads (slow for element 0 only, held by warp 0),
r = tl.atomic_add(Y + x, 0) (acq_rel, used), tl.store(OUT + x, r), then tl.atomic_add(C + x, 1,
sem=SEM), result unused (tl.atomic_add_unused under L157).
Program 1: a dependent chain of M loads of its own (a delay; no other program writes them), then
q = tl.atomic_add(C + W, 0) (acq_rel, used); if q == 1: tl.atomic_add(Y, 1) (acq_rel); store q.

Shared cells (Y, C) are reached by acq_rel gpu atomics only (Event.Strong); no IntraRace; no
program waits. In the semantics program 0's rmw on Y[0] precedes its store (the stored memory
depends on the old value, so Step.defer cannot pass it), the store precedes its rmw on C[W], and
program 1's rmw on Y[0] happens only after its rmw on C[W] returned 1 (control dependence; with
FIXED, `tl.atomic_add(Y, q)`, a data dependence and a footprint fixed in advance). So no
outcome of the semantics has Q = 1 and OUT[0] = 1.
"""
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def xw(Y, C, OUT, Q, P, PB, N: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr,
       SEM: tl.constexpr, W: tl.constexpr, FIXED: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        x = offs
        for i in range(N):
            x = tl.load(P + x)
        r = tl.atomic_add(Y + x, tl.zeros([BLOCK], tl.int32))
        tl.store(OUT + x, r)
        tl.atomic_add(C + x, tl.full([BLOCK], 1, tl.int32), sem=SEM)
    else:
        j = tl.load(PB)
        for i in range(M):
            j = tl.load(PB + j)
        q = tl.atomic_add(C + W + j, 0)
        if FIXED:
            tl.atomic_add(Y, q)  # footprint fixed; the added value depends on q
        else:
            if q == 1:
                tl.atomic_add(Y, 1)
        tl.store(Q, q)


def chain(L, n, dev, base):
    far = torch.randperm(L // 32 - 64, device=dev)[:n].to(torch.int32) * 32 + base
    return far


def main():
    BLOCK, L = 512, 1 << 25
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    dev = "cuda"
    for FIXED, W in [(False, BLOCK - 1), (True, BLOCK - 1), (False, 0)]:
        for sem in ["acq_rel", "relaxed"]:
            both = seen = 0
            for t in range(trials):
                P = torch.arange(L, dtype=torch.int32, device=dev)
                z = torch.zeros(1, device=dev, dtype=torch.int32)
                c = torch.cat([z, chain(L, N - 1, dev, 2048), z])
                P[c[:-1].long()] = c[1:]
                PB = torch.zeros(L, dtype=torch.int32, device=dev)
                cb = torch.cat([z, chain(L, M, dev, 2048)])
                PB[cb[:-1].long()] = cb[1:]          # PB[last] = 0: after M+1 loads j = 0
                Y = torch.zeros(BLOCK, dtype=torch.int32, device=dev)
                C = torch.zeros(BLOCK, dtype=torch.int32, device=dev)
                OUT = torch.full((BLOCK,), -1, dtype=torch.int32, device=dev)
                Q = torch.full((1,), -1, dtype=torch.int32, device=dev)
                xw[(2,)](Y, C, OUT, Q, P, PB, N=N, M=M, BLOCK=BLOCK, SEM=sem, W=W, FIXED=FIXED, num_warps=4)
                torch.cuda.synchronize()
                q = int(Q[0])
                assert q in (0, 1) and int(C.sum()) == BLOCK and bool((OUT[1:] == 0).all())
                assert int(Y[0]) == q
                seen += q == 1
                both += q == 1 and int(OUT[0]) == 1
            print(f"{'tl.atomic_add(Y, q)' if FIXED else 'if q == 1: tl.atomic_add(Y, 1)'}; watched C[{W}] (warp {W // 128}) sem={sem} N={N} M={M}: Q=1 in {seen}/{trials}, "
                  f"Q=1 and OUT[0]=1 (no outcome of the semantics) in {both}/{trials}")


if __name__ == "__main__":
    main()
