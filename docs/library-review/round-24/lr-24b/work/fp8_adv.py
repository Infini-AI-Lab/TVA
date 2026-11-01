"""lr-24b: adversarial tests of the fp8 chunk facts (DotE4m3Chunk, DotE5m2Chunk, DotE4m3E5m2Chunk,
DotE5m2E4m3Chunk) on sm_90 with Triton 3.8.0, written independently of evidence/num-dot-chunk.

Run from this folder:  CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I fp8_adv.py > fp8_adv.out

What it checks, per output element (i, n) of each tl.dot:
  * the facts: the device result lies in the interval [lo_n, hi_n] of values reachable from the
    accumulator by n = ceil(K/32) steps, each within rho*max(|s_j|, m_j) + sigma*|s_j + c_j| + alpha of
    s_j + c_j (rho = 2^-7, sigma = 2^-12, alpha = 2^-126, m_j the chunk's largest lifted product). The step
    bounds are monotone in s_j, so this interval is exactly the set the facts allow;
  * the lemma Dot8Chunk.DotF8Chunk.liftAcc (relLift8 K * L + absLift8 K, L on lifted magnitudes);
  * my own bit-level model of one wgmma chunk (written from rules/num.md's description): device = model.
    For mma.sync only the facts are checked (the fp8 operands go through float16 instructions).

Configurations not in the author's evidence: other block shapes and warp counts (m64n16, m64n32,
m64 with 8 warps, M = 256, N = 256, mma.sync with 8 warps), `a` transposed in the kernel, both operands
transposed, and a tutorial-03 K loop (`acc = tl.dot(a, b, acc)` and `acc += tl.dot(a, b)`, num_stages 3
and 4). Inputs: random bit patterns, window cases with mixed signs, all-negative subnormal products,
operands at the subnormal/normal edge, accumulator-dominated chunks, cancellation to near zero,
subnormal accumulators, operands near the maxima.
"""
import math
import random
import sys
import time
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

random.seed(24024)
torch.manual_seed(24024)
SC = 200                      # every value is held as an integer multiple of 2^-SC
RHO, SIG, ALP = Fr(1, 2 ** 7), Fr(1, 2 ** 12), Fr(1, 2 ** 126)
F32MAX = Fr((2 ** 24 - 1) * 2 ** 104)
FMT = {  # torch dtype, triton dtype, fraction bits, emin, max
    "e5": (torch.float8_e5m2, tl.float8e5, 2, -14, 57344.0),
    "e4": (torch.float8_e4m3fn, tl.float8e4nv, 3, -6, 448.0),
}
NAME = {"e5": "float8e5", "e4": "float8e4nv"}


def finite_values(f):
    v = torch.arange(256, dtype=torch.uint8).view(FMT[f][0]).float()
    return sorted(set(float(x) for x in v if math.isfinite(float(x))))


VALS = {f: finite_values(f) for f in FMT}


# ------------------------------------------------------------------------------------------------ kernels

@triton.jit
def dotk(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         A_T: tl.constexpr, B_T: tl.constexpr, A_REG: tl.constexpr, ADT: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    if A_T:
        a = tl.trans(tl.load(a_ptr + rk[:, None] * M + rm[None, :]))
    elif A_REG:
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :]).to(ADT)
    else:
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    if B_T:
        b = tl.trans(tl.load(b_ptr + rn[:, None] * K + rk[None, :]))
    else:
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, c))


@triton.jit
def kloop(a_ptr, b_ptr, c_ptr, o_ptr, KT, M: tl.constexpr, N: tl.constexpr, BK: tl.constexpr,
          PLUS: tl.constexpr, NS: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, BK)
    acc = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    for kk in tl.range(0, KT, BK, num_stages=NS):
        a = tl.load(a_ptr + rm[:, None] * KT + (kk + rk)[None, :])
        b = tl.load(b_ptr + (kk + rk)[:, None] * N + rn[None, :])
        if PLUS:
            acc += tl.dot(a, b)
        else:
            acc = tl.dot(a, b, acc)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc)


def instr(h):
    ptx = h.asm["ptx"]
    names = set()
    for line in ptx.splitlines():
        t = line.strip().split(" ")[0]
        if t.startswith("wgmma.mma_async") or t.startswith("mma.sync") or t.startswith("mma."):
            names.add(t)
    if not names and "fma.rn.f32" in ptx:
        names.add("fma(only)")
    return ",".join(sorted(names))


def path_of(names):
    return "wgmma" if "wgmma" in names else ("mma.sync" if "mma.sync" in names else "other")


# ------------------------------------------------------------------------------------------- arithmetic

def toi(x):
    return int(math.ldexp(x, SC))


def flog2(x):            # floor(log2|x|) of a nonzero integer held in units of 2^-SC
    return abs(x).bit_length() - 1 - SC


def field8(x, f):
    return max(flog2(x), FMT[f][3])


def trunc0(t, sh):       # t toward zero to a multiple of 2^sh (integer units)
    if sh <= 0 or t == 0:
        return t
    m = (abs(t) >> sh) << sh
    return m if t > 0 else -m


def cut_sig(x, p):       # x toward zero to p significant bits
    if x == 0:
        return 0
    q = abs(x).bit_length()
    return trunc0(x, q - p) if q > p else x


def lift(x, f):          # liftMag, integer units
    return 0 if x == 0 else max(abs(x), 1 << (FMT[f][3] + SC))


def model_chunk(s, ai, bi, da, db):
    """One wgmma chunk (my reading of rules/num.md): fields of the nonzero terms, E the largest,
    u = 2^(E-13), every term toward zero to a multiple of u, the exact sum toward zero to 14 bits."""
    terms = [(s, max(flog2(s), -126) if s else None)]
    for x, y in zip(ai, bi):
        p = (x * y) >> SC
        terms.append((p, field8(x, da) + field8(y, db) if (x and y) else None))
    fs = [f for (t, f) in terms if f is not None]
    if not fs:
        return 0
    sh = max(fs) - 13 + SC
    return cut_sig(sum(trunc0(t, sh) for t, _ in terms), 14)


def rn32(x):            # x to the nearest float32, ties to even (no overflow here)
    if x == 0:
        return 0
    sh = max(abs(x).bit_length() - 24, SC - 149)
    if sh <= 0:
        return x
    m, r = divmod(abs(x), 1 << sh)
    half = 1 << (sh - 1)
    if r > half or (r == half and m % 2 == 1):
        m += 1
    return (m << sh) if x > 0 else -(m << sh)


def check_element(ai, bi, ci, ri, da, db, K, want_model, plus_bk=None):
    """ai, bi lists of K integers, ci, ri integers. Returns (in_room, ratio_facts, ratio_lift, model_ok,
    max_chunk_ratio)."""
    prods = [(x * y) >> SC for x, y in zip(ai, bi)]
    lifts = [(lift(x, da) * lift(y, db)) >> SC for x, y in zip(ai, bi)]
    n = (K + 31) // 32
    unit = Fr(1, 2 ** SC)
    vc = Fr(ci) * unit
    vr = Fr(ri) * unit
    mag = abs(vc) + Fr(sum(abs(p) for p in prods), 2 ** SC)
    rel8 = (1 + RHO) ** n - 1
    abs8 = n * (1 + RHO) ** n * ALP
    if mag + rel8 * mag + abs8 > F32MAX:
        return (False, None, None, None, None)
    exact = vc + Fr(sum(prods), 2 ** SC)
    lo = hi = vc
    for j in range(n):
        cj = Fr(sum(prods[32 * j:32 * j + 32])) * unit
        mj = Fr(max(lifts[32 * j:32 * j + 32])) * unit
        hi = hi + cj + RHO * max(abs(hi), mj) + SIG * abs(hi + cj) + ALP
        lo = lo + cj - RHO * max(abs(lo), mj) - SIG * abs(lo + cj) - ALP
    if vr >= exact:
        rf = (vr - exact) / (hi - exact)
    else:
        rf = (exact - vr) / (exact - lo)
    L = abs(vc) + Fr(sum(lifts), 2 ** SC)
    rl8 = (1 + RHO + SIG) ** n - 1
    al8 = n * (1 + RHO + SIG) ** n * ALP
    rlift = abs(vr - exact) / (rl8 * L + al8)
    mok = None
    cmax = 0.0
    if want_model:
        # without plus_bk: one chain of chunks from the accumulator; with it (`acc += tl.dot(a, b)`, which
        # the combine pass leaves unfused for fp8: maxNumImpreciseAcc != 0): per block of plus_bk, the
        # chunks from 0, then a float32 addition to nearest into the accumulator
        acc = ci
        s = ci if plus_bk is None else 0
        for j in range(n):
            if plus_bk is not None and j > 0 and (32 * j) % plus_bk == 0:
                acc, s = rn32(acc + s), 0
            s2 = model_chunk(s, ai[32 * j:32 * j + 32], bi[32 * j:32 * j + 32], da, db)
            cj = sum(prods[32 * j:32 * j + 32])
            mj = max(abs(s), max(lifts[32 * j:32 * j + 32]))
            bound = RHO * Fr(mj) + SIG * Fr(abs(s + cj)) + ALP * 2 ** SC
            if bound > 0:
                cmax = max(cmax, float(Fr(abs(s2 - (s + cj))) / bound))
            s = s2
        if plus_bk is not None:
            s = rn32(acc + s)
        mok = (s == ri)
    return (True, float(rf), float(rlift), mok, cmax)


# --------------------------------------------------------------------------------------------- inputs

def rand_sign():
    return 1.0 if random.random() < 0.5 else -1.0


def pick(vals, lo, hi):
    c = [v for v in vals if lo <= abs(v) <= hi and v > 0]
    return random.choice(c)


def allset_below(f, target):
    """The largest-significand value of format f strictly below target (positive), or 0."""
    c = [v for v in VALS[f] if 0 < v < target]
    return max(c) if c else 0.0


def gen(kind, M, N, K, da, db):
    """Returns a [M, K], b [K, N], c [M, N] as float lists (a, b representable in da, db)."""
    A = [[0.0] * K for _ in range(M)]
    B = [[0.0] * N for _ in range(K)]
    C = [[0.0] * N for _ in range(M)]
    va, vb = VALS[da], VALS[db]
    mna, mnb = 2.0 ** FMT[da][3], 2.0 ** FMT[db][3]
    if kind == "bits":
        for i in range(M):
            for k in range(K):
                A[i][k] = random.choice(va)
        for k in range(K):
            for n in range(N):
                B[k][n] = random.choice(vb)
        for i in range(M):
            for n in range(N):
                C[i][n] = random.choice([0.0, rand_sign() * 2.0 ** random.randint(-40, 30)])
        return A, B, C
    if kind in ("winmix", "negsub", "edge"):
        # b: per chunk one big partner at kb (same for every column), powers of two elsewhere (sign per
        # element); a: per row the operand of the big product (subnormal, at the edge or normal) and
        # values with every fraction bit set chosen so that the products fall just below u
        nch = (K + 31) // 32
        kbs = [32 * j + random.randrange(min(32, K - 32 * j)) for j in range(nch)]
        bigb = pick(vb, FMT[db][4] / 4, FMT[db][4])
        tk = [random.randint(-4, 2) for _ in range(K)]
        for k in range(K):
            for n in range(N):
                if k in kbs:
                    B[k][n] = bigb if kind != "negsub" else -bigb
                elif kind == "negsub":
                    B[k][n] = random.choice([v for v in vb if 0 < v < mnb])
                else:
                    B[k][n] = (rand_sign() if kind == "winmix" else 1.0) * 2.0 ** tk[k]
        for i in range(M):
            if kind == "edge":
                # the two largest subnormals and the two smallest normals
                cand = sorted(v for v in va if 0 < v < mna)[-2:] + sorted(v for v in va if v >= mna)[:2]
            else:
                cand = [v for v in va if 0 < v < mna] if (kind == "negsub" or random.random() < 0.6) else \
                    [v for v in va if mna <= v <= 4 * mna]
            for j, kb in enumerate(kbs):
                x = random.choice(cand)
                A[i][kb] = x if kind != "negsub" else x
                fa = max(math.frexp(x)[1] - 1, FMT[da][3])
                fb = math.frexp(bigb)[1] - 1
                u = 2.0 ** (fa + fb - 13)
                for k in range(32 * j, min(32 * j + 32, K)):
                    if k == kb:
                        continue
                    # |a*b| just below u (or between u and 2u for a third of them): a = allset * 2^e
                    tgt = u if random.random() < 0.67 else 2 * u
                    av = allset_below(da, tgt / 2.0 ** tk[k])
                    if kind == "negsub":
                        av = random.choice([v for v in va if 0 < v < mna])
                        A[i][k] = -av
                    else:
                        A[i][k] = av * (rand_sign() if kind == "winmix" else 1.0)
            fa0 = max(math.frexp(A[i][kbs[0]])[1] - 1, FMT[da][3]) if A[i][kbs[0]] else 0
            u0 = 2.0 ** (fa0 + math.frexp(bigb)[1] - 1 - 13)
            for n in range(N):
                r = random.random()
                if r < 0.25:
                    C[i][n] = 0.0
                elif r < 0.5:
                    C[i][n] = rand_sign() * u0 * (1 - 2.0 ** -23)
                elif r < 0.75:
                    C[i][n] = rand_sign() * u0 * random.random()
                else:
                    C[i][n] = rand_sign() * abs(A[i][kbs[0]] * bigb) * (1 + random.random())
        C = [[float(torch.tensor(x, dtype=torch.float32)) for x in row] for row in C]
        return A, B, C
    if kind == "accdom":
        # an accumulator with every bit set, products with random signs just below its window's unit
        for i in range(M):
            e = random.randint(-20, 20)
            for n in range(N):
                C[i][n] = rand_sign() * (2 - 2.0 ** -23) * 2.0 ** e
            u = 2.0 ** (e - 13)
            for k in range(K):
                A[i][k] = rand_sign() * allset_below(da, u)
        for k in range(K):
            for n in range(N):
                B[k][n] = rand_sign() * 1.0
        return A, B, C
    if kind == "cancel":
        # random products, accumulator = minus their exact sum rounded to float32 (s + c near zero)
        for i in range(M):
            for k in range(K):
                A[i][k] = rand_sign() * random.choice(va[len(va) // 2:])
        for k in range(K):
            for n in range(N):
                B[k][n] = rand_sign() * random.choice(vb[len(vb) // 2:])
        S = -(torch.tensor(A, dtype=torch.float64) @ torch.tensor(B, dtype=torch.float64))
        C = S.to(torch.float32).double().tolist()
        return A, B, C
    if kind == "subacc":
        # subnormal float32 accumulators, the smallest products (or none)
        for i in range(M):
            for k in range(K):
                A[i][k] = rand_sign() * random.choice([0.0] + [v for v in va if 0 < v < 2 * mna])
        for k in range(K):
            for n in range(N):
                B[k][n] = rand_sign() * random.choice([v for v in vb if 0 < v < 2 * mnb])
        for i in range(M):
            for n in range(N):
                C[i][n] = rand_sign() * random.randint(1, 2 ** 23 - 1) * 2.0 ** -149
        return A, B, C
    if kind == "huge":
        for i in range(M):
            for k in range(K):
                A[i][k] = rand_sign() * random.choice(va[-6:])
        for k in range(K):
            for n in range(N):
                B[k][n] = rand_sign() * random.choice(vb[-6:])
        for i in range(M):
            for n in range(N):
                C[i][n] = rand_sign() * random.random() * 2.0 ** 100
        C = [[float(torch.tensor(x, dtype=torch.float32)) for x in row] for row in C]
        return A, B, C
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------- runs

def to_dev(X, dt):
    return torch.tensor(X, dtype=torch.float32).to(dt).cuda()


def run_case(kind, M, N, K, nw, da, db, layout, stats, max_elems=2048):
    A, B, C = gen(kind, M, N, K, da, db)
    a32 = torch.tensor(A, dtype=torch.float32)
    b32 = torch.tensor(B, dtype=torch.float32)
    c32 = torch.tensor(C, dtype=torch.float32).cuda()
    a8 = a32.to(FMT[da][0])
    b8 = b32.to(FMT[db][0])
    assert torch.equal(a8.float(), a32) and torch.equal(b8.float(), b32)
    o = torch.empty(M, N, device="cuda")
    if layout == "aT":
        ain, bin_ = a8.t().contiguous().cuda(), b8.contiguous().cuda()
        h = dotk[(1,)](ain, bin_, c32, o, M, N, K, True, False, False, FMT[da][1], num_warps=nw)
    elif layout == "abT":
        ain, bin_ = a8.t().contiguous().cuda(), b8.t().contiguous().cuda()
        h = dotk[(1,)](ain, bin_, c32, o, M, N, K, True, True, False, FMT[da][1], num_warps=nw)
    elif layout == "areg16":
        ain, bin_ = a32.to(torch.float16).cuda(), b8.contiguous().cuda()
        h = dotk[(1,)](ain, bin_, c32, o, M, N, K, False, False, True, FMT[da][1], num_warps=nw)
    else:
        ain, bin_ = a8.contiguous().cuda(), b8.contiguous().cuda()
        h = dotk[(1,)](ain, bin_, c32, o, M, N, K, False, False, False, FMT[da][1], num_warps=nw)
    names = instr(h)
    path = path_of(names)
    return finish(A, B, C, o, da, db, K, path, names, stats, max_elems, (kind, M, N, K, nw, layout))


def finish(A, B, C, o, da, db, K, path, names, stats, max_elems, label, plus_bk=None):
    M, N = len(A), len(B[0])
    O = o.double().cpu().tolist()
    Ai = [[toi(x) for x in row] for row in A]
    Bi = [[toi(B[k][n]) for k in range(K)] for n in range(N)]
    idx = [(i, n) for i in range(M) for n in range(N)]
    if len(idx) > max_elems:
        idx = random.sample(idx, max_elems)
    worst = (0.0, 0.0, 0.0)
    nchk = nroom = nmod = 0
    bad = []
    for (i, n) in idx:
        inr, rf, rl, mok, cr = check_element(Ai[i], Bi[n], toi(C[i][n]), toi(O[i][n]), da, db, K,
                                             path == "wgmma", plus_bk)
        if not inr:
            continue
        nchk += 1
        if rf > 1 or rl > 1:
            bad.append((i, n, rf, rl))
        if mok is not None:
            nmod += 1 if mok else 0
        worst = (max(worst[0], rf), max(worst[1], rl), max(worst[2], cr))
    key = (NAME[da], NAME[db], path)
    st = stats.setdefault(key, [0, 0, 0, 0.0, 0.0, 0.0])
    st[0] += nchk
    st[1] += len(bad)
    st[2] += nmod
    st[3] = max(st[3], worst[0])
    st[4] = max(st[4], worst[1])
    st[5] = max(st[5], worst[2])
    mod = f"{nmod}/{nchk}" if path == "wgmma" else "-"
    print(f"| {NAME[da]} x {NAME[db]} | {label} | {names} | {nchk} | {len(bad)} | {mod} | "
          f"{worst[0]:.4f} | {worst[1]:.4f} | {worst[2]:.4f} |", flush=True)
    for b in bad[:5]:
        print("   OUTSIDE:", b, flush=True)
    return bad


def run_kloop(kind, M, N, BK, KT, nw, ns, plus, da, db, stats):
    A, B, C = gen(kind, M, N, KT, da, db)
    a8 = torch.tensor(A, dtype=torch.float32).to(FMT[da][0]).cuda()
    b8 = torch.tensor(B, dtype=torch.float32).to(FMT[db][0]).cuda()
    c32 = torch.tensor(C, dtype=torch.float32).cuda()
    o = torch.empty(M, N, device="cuda")
    h = kloop[(1,)](a8, b8, c32, o, KT, M, N, BK, plus, ns, num_warps=nw)
    names = instr(h)
    return finish(A, B, C, o, da, db, KT, path_of(names), names, stats, 2048,
                  ("kloop" + ("+=" if plus else ""), kind, M, N, f"BK={BK}", f"K={KT}", nw, f"ns={ns}"),
                  BK if plus else None)


def main():
    t0 = time.time()
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("lr-24b fp8 adversarial tests (facts interval, liftAcc lemma, model on wgmma)\n")
    print("| operands | case (kind, M, N, K, warps, layout) | instructions | elements | outside | model = device |"
          " ratio (facts) | ratio (liftAcc) | chunk ratio (model) |")
    print("|---|---|---|---|---|---|---|---|---|")
    stats = {}
    pairs = [("e4", "e4"), ("e5", "e5"), ("e4", "e5"), ("e5", "e4")]
    configs = [(64, 16, 4), (64, 32, 4), (64, 64, 8), (128, 32, 4), (256, 64, 8), (64, 256, 4),
               (16, 32, 2), (32, 16, 1), (32, 128, 4), (64, 64, 2), (16, 16, 8)]
    kinds = ["winmix", "negsub", "edge", "accdom", "cancel", "subacc", "huge", "bits"]
    if "--quick" in sys.argv:    # a smoke test, not the evidence
        pairs, configs = [("e4", "e5")], [(64, 16, 4), (16, 32, 2)]
    for (da, db) in pairs:
        for (M, N, nw) in configs:
            for K in (32, 128):
                for kind in kinds:
                    run_case(kind, M, N, K, nw, da, db, "plain", stats, max_elems=512)
        for (M, N, nw) in [(64, 64, 4), (128, 64, 4), (32, 32, 2)]:
            for layout in ("aT", "abT", "areg16"):
                for K in (64, 256):
                    for kind in ("winmix", "negsub", "edge"):
                        run_case(kind, M, N, K, nw, da, db, layout, stats, max_elems=512)
        for (M, N, nw) in [(128, 64, 4), (64, 128, 4), (32, 64, 2)]:
            for ns in (3, 4):
                for plus in (False, True):
                    for kind in ("winmix", "negsub", "accdom"):
                        run_kloop(kind, M, N, 64, 512, nw, ns, plus, da, db, stats)
    print("\nTotals per operand pair and path: elements, outside the facts or the lemma, model = device "
          "(wgmma), largest ratio (facts), largest ratio (liftAcc), largest chunk ratio (model)")
    for key, st in sorted(stats.items()):
        print(f"| {key[0]} x {key[1]} | {key[2]} | {st[0]} | {st[1]} | {st[2] if key[2] == 'wgmma' else '-'} | "
              f"{st[3]:.4f} | {st[4]:.4f} | {st[5]:.4f} |")
    print(f"\n(run time {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
