"""TVA library evidence (library issue L161): an atomic's result used, and the order of a program's atomics.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/atomic-result-order/run.py
Output: evidence/atomic-result-order/out.txt (README.md says what it shows).

`Step.defer` (lean/TVA/Sem/Step.lean) lets a pending atomic be passed only by a later step that is
the same for every old value. A tensor `tl.store` is one step, so a store that uses any lane of an
atomic's result holds back every later atomic of the program, while on the device each thread orders
only its own accesses. Two launches, each well behaved in the semantics (every cell two accesses
share is reached by `acq_rel` gpu atomics only, no program has an `IntraRace`, no program waits):

A (library review lr-34a's counterexample, two programs). Program 0: a dependent chain of N loads that
  is slow for element 0 only (held by warp 0; a delay: no program writes the chain, and on these data
  it leaves x = offs), r = tl.atomic_add(Y + x, 0), tl.store(OUT + x, r), tl.atomic_add(C + x, 1).
  Program 1: a chain of M loads (a delay that ends at j = 0), q = tl.atomic_add(C + 511 + j, 0),
  tl.atomic_add(Y, q), tl.store(Q, q). In the semantics program 0's atomic on Y[0] precedes its store,
  which precedes its atomic on C[511], and program 1's atomic on Y[0] follows its atomic on C[511]; so
  no outcome has Q = 1 and OUT[0] = 1.
B (one program). The same chain, r = tl.atomic_add(Y0 + 0 * x, 1) (512 lanes on the one cell Y[0]),
  tl.store(OUT + x, r), s = tl.atomic_add(Y0 + 0 * x, 1), tl.store(OUT2 + x, s). In the semantics
  every lane of the first call precedes the store, which precedes every lane of the second call; so
  from Y[0] = 0 every r is below 512 and every s at least 512.
Each runs with `sem="acq_rel"` and `sem="relaxed"` on every atomic, 200 launches each.
"""
import os
import sys

import torch
import triton
import triton.language as tl

OUT_LINES = []
HERE = os.path.dirname(os.path.abspath(__file__))


def out(s=""):
    print(s, flush=True)
    OUT_LINES.append(s)


@triton.jit
def two_programs(Y, C, OUT, Q, P, PB, N: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr, SEM: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        x = offs
        for i in range(N):
            x = tl.load(P + x)
        r = tl.atomic_add(Y + x, tl.zeros([BLOCK], tl.int32), sem=SEM)
        tl.store(OUT + x, r)
        tl.atomic_add(C + x, tl.full([BLOCK], 1, tl.int32), sem=SEM)
    else:
        j = tl.load(PB)
        for i in range(M):
            j = tl.load(PB + j)
        q = tl.atomic_add(C + BLOCK - 1 + j, 0, sem=SEM)
        tl.atomic_add(Y, q, sem=SEM)
        tl.store(Q, q)


@triton.jit
def one_program(Y0, OUT, OUT2, P, N: tl.constexpr, BLOCK: tl.constexpr, SEM: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = offs
    for i in range(N):
        x = tl.load(P + x)
    r = tl.atomic_add(Y0 + 0 * x, tl.full([BLOCK], 1, tl.int32), sem=SEM)
    tl.store(OUT + x, r)
    s = tl.atomic_add(Y0 + 0 * x, tl.full([BLOCK], 1, tl.int32), sem=SEM)
    tl.store(OUT2 + x, s)


def chain(L, n, dev, base):
    return torch.randperm(L // 32 - 64, device=dev)[:n].to(torch.int32) * 32 + base


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    BLOCK, L = 512, 1 << 25
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    N, M = 400, 60
    dev = "cuda"
    z = torch.zeros(1, device=dev, dtype=torch.int32)
    out(f"BLOCK = {BLOCK}, num_warps = 4, a chain of N = {N} loads for element 0, M = {M} for program 1; "
        f"{trials} launches per row")
    out("")
    out("| launch | sem | launches | outside the semantics | detail |")
    out("|---|---|---|---|---|")
    for sem in ["acq_rel", "relaxed"]:
        both = seen = 0
        for t in range(trials):
            P = torch.arange(L, dtype=torch.int32, device=dev)
            c = torch.cat([z, chain(L, N - 1, dev, 2048), z])
            P[c[:-1].long()] = c[1:]
            PB = torch.zeros(L, dtype=torch.int32, device=dev)
            cb = torch.cat([z, chain(L, M, dev, 2048)])
            PB[cb[:-1].long()] = cb[1:]
            Y = torch.zeros(BLOCK, dtype=torch.int32, device=dev)
            C = torch.zeros(BLOCK, dtype=torch.int32, device=dev)
            OUT = torch.full((BLOCK,), -1, dtype=torch.int32, device=dev)
            Q = torch.full((1,), -1, dtype=torch.int32, device=dev)
            two_programs[(2,)](Y, C, OUT, Q, P, PB, N=N, M=M, BLOCK=BLOCK, SEM=sem, num_warps=4)
            torch.cuda.synchronize()
            q = int(Q[0])
            assert q in (0, 1) and int(C.sum()) == BLOCK and int(Y[0]) == q and bool((OUT[1:] == 0).all())
            seen += q == 1
            both += q == 1 and int(OUT[0]) == 1
        out(f"| A, two programs | {sem} | {trials} | {both} | Q = 1 in {seen}; Q = 1 and OUT[0] = 1 in {both} |")
    for sem in ["acq_rel", "relaxed"]:
        bad = 0
        worst = None
        for t in range(trials):
            P = torch.arange(L, dtype=torch.int32, device=dev)
            c = torch.cat([z, chain(L, N - 1, dev, 2048), z])
            P[c[:-1].long()] = c[1:]
            Y0 = torch.zeros(1, dtype=torch.int32, device=dev)
            OUT = torch.full((BLOCK,), -1, dtype=torch.int32, device=dev)
            OUT2 = torch.full((BLOCK,), -1, dtype=torch.int32, device=dev)
            one_program[(1,)](Y0, OUT, OUT2, P, N=N, BLOCK=BLOCK, SEM=sem, num_warps=4)
            torch.cuda.synchronize()
            r, s = OUT.cpu(), OUT2.cpu()
            assert int(Y0[0]) == 2 * BLOCK
            assert sorted(torch.cat([r, s]).tolist()) == list(range(2 * BLOCK))
            if int(r.max()) >= BLOCK or int(s.min()) < BLOCK:
                bad += 1
                worst = (int(r.max()), int(s.min()), int(r[0]))
        out(f"| B, one program | {sem} | {trials} | {bad} | in the last such launch: max r = {worst[0] if worst else '-'}, "
            f"min s = {worst[1] if worst else '-'}, r[0] = {worst[2] if worst else '-'} (the semantics: max r < {BLOCK} <= min s) |")
    with open(os.path.join(HERE, "out.txt"), "w") as fh:
        fh.write("\n".join(OUT_LINES) + "\n")


if __name__ == "__main__":
    main()
