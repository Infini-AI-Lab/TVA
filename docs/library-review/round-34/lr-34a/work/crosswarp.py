"""lr-34a: does the model's order of one program's atomics hold on the device?

Program 0: a dependent chain of loads (slow only for element 0, which warp 0 holds), then
r = tl.atomic_add(Y + x, 0) (acq_rel, its result used), tl.store(OUT + x, r), then
tl.atomic_add(C + offs, 1, sem=SEM) (result unused: tl.atomic_add_unused under L157).
Program 1: spins on tl.atomic_add(C + W, 0) (acq_rel, used) until it is 1, then
tl.atomic_add(Y, 1) (acq_rel).

Every cell two programs share is reached only by Event.Strong atomics (acq_rel, gpu), and no
program has an IntraRace, so the launch is well behaved. In the semantics program 0's rmw on Y[0]
cannot be deferred past the store (the stored memory depends on its old value), so it precedes the
rmw on C[BLOCK-1]; program 1 increments Y[0] only after reading C[BLOCK-1] = 1. So every outcome
of the semantics has OUT[0] = 0. OUT[0] = 1 on the device is an outcome the semantics does not give.
"""
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def xw(Y, C, OUT, P, N: tl.constexpr, BLOCK: tl.constexpr, SEM: tl.constexpr, W: tl.constexpr):
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
        q = tl.atomic_add(C + W, 0)
        while q == 0:
            q = tl.atomic_add(C + W, 0)
        tl.atomic_add(Y, 1)


def main():
    BLOCK, L = 512, 1 << 25
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    dev = "cuda"
    P = torch.arange(L, dtype=torch.int32, device=dev)
    for W in [BLOCK - 1, 0]:
        for sem in ["acq_rel", "relaxed"]:
            hits = 0
            for t in range(trials):
                far = torch.randperm(L // 32 - 64, device=dev)[: N - 1].to(torch.int32) * 32 + 2048
                z = torch.zeros(1, device=dev, dtype=torch.int32)
                chain = torch.cat([z, far, z])
                P[chain[:-1].long()] = chain[1:]
                Y = torch.zeros(BLOCK, dtype=torch.int32, device=dev)
                C = torch.zeros(BLOCK, dtype=torch.int32, device=dev)
                OUT = torch.full((BLOCK,), -1, dtype=torch.int32, device=dev)
                k = xw[(2,)](Y, C, OUT, P, N=N, BLOCK=BLOCK, SEM=sem, W=W, num_warps=4)
                torch.cuda.synchronize()
                assert int(C.sum()) == BLOCK and int(Y[0]) == 1, (C.sum(), Y[0])
                assert bool((OUT[1:] == 0).all())
                hits += int(OUT[0]) == 1
                P[chain[:-1].long()] = chain[:-1]
            print(f"watched C[{W}] (warp {W // 128}) sem={sem} N={N}: OUT[0]=1 in {hits}/{trials} launches")
            if sem == "relaxed" and W == 0:
                for ln in k.asm["ptx"].splitlines():
                    if any(w in ln for w in ("atom.", "bar.", "membar", "fence", "st.global", "ld.global")):
                        print("   ", ln.strip())


if __name__ == "__main__":
    main()
