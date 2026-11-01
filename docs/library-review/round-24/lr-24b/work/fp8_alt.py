"""lr-24b: is the fed-back accumulator's compounding under the fp8 chunk facts the device's behaviour
(library issue L124's round-24 entry), or slack in the facts?

The chunk facts bound each chunk's step by rho*Mj + sigma*|s_j + c_j| + alpha (rho = 2^-7,
sigma = 2^-12, alpha = 2^-126), Mj >= |s_j|, so an accumulator that dominates its chunks is charged
2^-7 + 2^-12 of itself per chunk. By rules/num.md's own model of a wgmma chunk (each term truncated
toward zero to a multiple of u, then the sum cut to 14 bits), a truncated term loses at most
min(|term|, u), so a chunk's error is also below

    (1 + 2^-13) * (sum_chunk |a_k b_k| + u) + 2^-13 * |s_j + c_j|,   u <= 2^-13 * Mj + 2^-139,

which gives the alternative step bound

    B_alt = (1 + 2^-13) * sum_chunk |a_k b_k| + 2^-12 * Mj + sigma * |s_j + c_j| + alpha

(Mj = max(|s_j|, the chunk's largest lifted product)). Under B_alt a dominating accumulator is charged
2^-12 + 2^-12 of itself per chunk instead of 2^-7 + 2^-12.

This script measures the device against B_alt:
  * single-chunk dots (K = 32), where the chunk's partial values are the accumulator and the result,
    so the check needs no model, on both instruction paths;
  * chains of tl.dot(p, v, acc) as in tutorial 06 (p from a softmax, or adversarial blocks whose
    products sit just below the window's unit under a dominating accumulator), K = 64 per dot, every
    accumulator stored, on wgmma: the device equals the model on every dot, and each chunk of the
    model's chain is checked against B_alt and against the facts' own step bound.

Run from this folder:  CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I fp8_alt.py > fp8_alt.out
"""
import math
import random
import time
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

random.seed(24124)
torch.manual_seed(24124)
SC = 200
RHO, SIG, ALP = Fr(1, 2 ** 7), Fr(1, 2 ** 12), Fr(1, 2 ** 126)
FMT = {"e5": (torch.float8_e5m2, tl.float8e5, -14), "e4": (torch.float8_e4m3fn, tl.float8e4nv, -6)}
NAME = {"e5": "float8e5", "e4": "float8e4nv"}


def finite_values(f):
    v = torch.arange(256, dtype=torch.uint8).view(FMT[f][0]).float()
    return sorted(set(float(x) for x in v if math.isfinite(float(x))))


VALS = {f: finite_values(f) for f in FMT}


@triton.jit
def dot1(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, c))


@triton.jit
def chain(p_ptr, v_ptr, c_ptr, accs_ptr, T, M: tl.constexpr, N: tl.constexpr, KC: tl.constexpr,
          ADT: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, KC)
    acc = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    for t in range(T):
        p = tl.load(p_ptr + t * M * KC + rm[:, None] * KC + rk[None, :]).to(ADT)
        v = tl.load(v_ptr + t * KC * N + rk[:, None] * N + rn[None, :])
        acc = tl.dot(p, v, acc)
        tl.store(accs_ptr + t * M * N + rm[:, None] * N + rn[None, :], acc)


def toi(x):
    return int(math.ldexp(x, SC))


def flog2(x):
    return abs(x).bit_length() - 1 - SC


def trunc0(t, sh):
    if sh <= 0 or t == 0:
        return t
    m = (abs(t) >> sh) << sh
    return m if t > 0 else -m


def cut_sig(x, p):
    if x == 0:
        return 0
    q = abs(x).bit_length()
    return trunc0(x, q - p) if q > p else x


def lift(x, f):
    return 0 if x == 0 else max(abs(x), 1 << (FMT[f][2] + SC))


def model_chunk(s, ai, bi, da, db):
    terms = [(s, max(flog2(s), -126) if s else None)]
    for x, y in zip(ai, bi):
        terms.append(((x * y) >> SC, max(flog2(x), FMT[da][2]) + max(flog2(y), FMT[db][2]) if (x and y) else None))
    fs = [e for (t, e) in terms if e is not None]
    if not fs:
        return 0
    sh = max(fs) - 13 + SC
    return cut_sig(sum(trunc0(t, sh) for t, _ in terms), 14)


def step_ratios(s, s2, ai, bi, da, db):
    """The chunk step s -> s2 against the facts' bound and B_alt (both at the smallest Mj)."""
    D = 2 ** SC
    prods = [(x * y) >> SC for x, y in zip(ai, bi)]
    mj = Fr(max([abs(s)] + [(lift(x, da) * lift(y, db)) >> SC for x, y in zip(ai, bi)]), D)
    c = sum(prods)
    err = Fr(abs(s2 - (s + c)), D)
    new = Fr(abs(s + c), D)
    facts = RHO * mj + SIG * new + ALP
    alt = (1 + Fr(1, 2 ** 13)) * Fr(sum(abs(p) for p in prods), D) + Fr(1, 2 ** 12) * mj + SIG * new + ALP
    return float(err / facts), float(err / alt), float(Fr(abs(s), D) / mj) if mj else 0.0


def allset_below(f, target):
    c = [v for v in VALS[f] if 0 < v < target]
    return max(c) if c else 0.0


def rs():
    return 1.0 if random.random() < 0.5 else -1.0


def gen_single(kind, M, N, da, db):
    K = 32
    A = [[0.0] * K for _ in range(M)]
    B = [[0.0] * N for _ in range(K)]
    C = [[0.0] * N for _ in range(M)]
    va, vb = VALS[da], VALS[db]
    for k in range(K):
        for n in range(N):
            B[k][n] = 1.0 if kind == "accdom1" else rs() * random.choice([1.0, 0.5, 2.0])
    for i in range(M):
        e = random.randint(-20, 20)
        if kind == "accdom1":      # one sign: products just below the accumulator's window unit, all lost
            u = 2.0 ** (e - 13)
            for k in range(K):
                A[i][k] = allset_below(da, u)
            for n in range(N):
                C[i][n] = (2 - 2.0 ** -23) * 2.0 ** e
        elif kind == "accdom":     # products just below the accumulator's window unit 2^(e-13)
            u = 2.0 ** (e - 13)
            for k in range(K):
                A[i][k] = rs() * allset_below(da, u / 2)
            for n in range(N):
                C[i][n] = rs() * (2 - 2.0 ** -23) * 2.0 ** e
        elif kind == "accsmallp":  # products far below the accumulator (the attention regime)
            for k in range(K):
                A[i][k] = rs() * random.choice(va[len(va) // 4:len(va) // 2])
            for n in range(N):
                C[i][n] = rs() * random.uniform(1, 2) * 2.0 ** (e + 10)
        else:                      # random operands, accumulators of any size
            for k in range(K):
                A[i][k] = rs() * random.choice(va)
            for n in range(N):
                C[i][n] = rs() * random.uniform(1, 2) * 2.0 ** random.randint(-30, 30)
    C = [[float(torch.tensor(x, dtype=torch.float32)) for x in row] for row in C]
    return A, B, C


def run_single(stats):
    for (da, db) in [("e4", "e4"), ("e5", "e5"), ("e4", "e5"), ("e5", "e4")]:
        for (M, N, nw) in [(64, 64, 4), (128, 64, 8), (32, 32, 2), (16, 64, 1)]:
            for kind in ("accdom1", "accdom", "accsmallp", "random"):
                A, B, C = gen_single(kind, M, N, da, db)
                a = torch.tensor(A).to(FMT[da][0]).cuda()
                b = torch.tensor(B).to(FMT[db][0]).cuda()
                c = torch.tensor(C, dtype=torch.float32).cuda()
                o = torch.empty(M, N, device="cuda")
                h = dot1[(1,)](a, b, c, o, M, N, 32, num_warps=nw)
                path = "wgmma" if "wgmma" in h.asm["ptx"] else "mma.sync"
                O = o.double().cpu().tolist()
                Bc = [[toi(B[k][n]) for k in range(32)] for n in range(N)]
                wf = wa = 0.0
                n_el = 0
                for i in range(M):
                    ai = [toi(x) for x in A[i]]
                    for n in range(N):
                        rf, ra, _ = step_ratios(toi(C[i][n]), toi(O[i][n]), ai, Bc[n], da, db)
                        wf, wa = max(wf, rf), max(wa, ra)
                        n_el += 1
                key = (NAME[da], NAME[db], path, "single K=32")
                st = stats.setdefault(key, [0, 0.0, 0.0, 0, 0])
                st[0] += n_el
                st[1], st[2] = max(st[1], wf), max(st[2], wa)
                print(f"| {NAME[da]} x {NAME[db]} | single, {kind}, {M} x {N}, {nw} warps | {path} | {n_el} | "
                      f"{wf:.4f} | {wa:.4f} |", flush=True)


def run_chains(stats):
    KC = 64
    for (da, db) in [("e4", "e4"), ("e5", "e5"), ("e4", "e5"), ("e5", "e4")]:
        for (M, N, nw) in [(128, 64, 4), (128, 128, 4)]:
            for kind in ("softmax", "accdom"):
                T = 32
                P = torch.zeros(T, M, KC)
                V = torch.zeros(T, KC, N)
                vvals = torch.tensor([v for v in VALS[db] if 0.25 <= abs(v) <= 4])
                for t in range(T):
                    if kind == "softmax":
                        s = torch.randn(M, KC) * 4
                        P[t] = torch.exp2(s - s.max(dim=1, keepdim=True).values).to(FMT[da][0]).float()
                        V[t] = vvals[torch.randint(len(vvals), (KC, N))] * torch.sign(torch.randn(KC, N))
                    else:
                        # after the first block: products far below the accumulator, one third just below
                        # an accumulator-scale window unit
                        P[t] = torch.tensor([[allset_below(da, 2.0 ** random.randint(-9, -4)) for _ in range(KC)]
                                             for _ in range(M)])
                        V[t] = torch.sign(torch.randn(KC, N))
                c0 = torch.randn(M, N).float() * (1.0 if kind == "softmax" else 2.0 ** 6)
                Pin = P.cuda().contiguous()
                V8 = V.to(FMT[db][0]).cuda().contiguous()
                accs = torch.empty(T, M, N, device="cuda")
                h = chain[(1,)](Pin, V8, c0.cuda(), accs, T, M, N, KC, FMT[da][1], num_warps=nw)
                path = "wgmma" if "wgmma" in h.asm["ptx"] else "mma.sync"
                A8 = P.to(FMT[da][0]).float().double()
                ACC = accs.double().cpu()
                wf = wa = 0.0
                nok = nd = nch = 0
                dom = 0.0
                rows = random.sample(range(M), 8)
                cols = random.sample(range(N), 8)
                for i in rows:
                    for n in cols:
                        s = toi(float(c0[i, n]))
                        for t in range(T):
                            ai = [toi(float(x)) for x in A8[t, i]]
                            bi = [toi(float(x)) for x in V[t, :, n]]
                            for j in range(KC // 32):
                                s2 = model_chunk(s, ai[32 * j:32 * j + 32], bi[32 * j:32 * j + 32], da, db)
                                rf, ra, d = step_ratios(s, s2, ai[32 * j:32 * j + 32], bi[32 * j:32 * j + 32], da, db)
                                wf, wa, dom = max(wf, rf), max(wa, ra), max(dom, d)
                                nch += 1
                                s = s2
                            nd += 1
                            nok += 1 if s == toi(float(ACC[t, i, n])) else 0
                key = (NAME[da], NAME[db], path, "chains K=64")
                st = stats.setdefault(key, [0, 0.0, 0.0, 0, 0])
                st[0] += nch
                st[1], st[2] = max(st[1], wf), max(st[2], wa)
                st[3] += nok
                st[4] += nd
                print(f"| {NAME[da]} x {NAME[db]} | chain, {kind}, {M} x {N}, T = {T} | {path} | {nch} chunks, "
                      f"model = device on {nok}/{nd} dots | {wf:.4f} | {wa:.4f} |", flush=True)


def main():
    t0 = time.time()
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("| operands | case | path | elements or chunks | largest step / facts' bound | largest step / B_alt |")
    print("|---|---|---|---|---|---|")
    stats = {}
    run_single(stats)
    run_chains(stats)
    print("\nTotals: elements or chunks, largest ratio to the facts' step bound, largest ratio to B_alt "
          "(chains: model = device dots)")
    for key, st in sorted(stats.items()):
        extra = f", model = device {st[3]}/{st[4]}" if st[4] else ""
        print(f"| {key[0]} x {key[1]} | {key[2]} | {key[3]} | {st[0]} | {st[1]:.4f} | {st[2]:.4f}{extra} |")
    print(f"\n(run time {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
