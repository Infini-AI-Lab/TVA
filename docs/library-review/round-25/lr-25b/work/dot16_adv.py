"""lr-25b: adversarial tests of the 16-bit tl.dot facts DotF16 and DotBF16 as restated in round 25
(DotLiftAcc, relDot16 K = (K+2)*2^-22, absDot16 K = (K+2)*2^-126, against
L = |acc| + sum liftMag a * liftMag b) on sm_90 with Triton 3.8.0, written independently of
evidence/num-dot-sub16-round25.

Run from this folder:  CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I dot16_adv.py > dot16_adv.out

Per output element (i, n) of each tl.dot, with room (L + rel*L + abs <= float32 max):
  * the fact: |r - (acc + sum a*b)| <= relDot16 K * L + absDot16 K (`ratio`);
  * the bound before round 25, relative to |acc| + sum |a*b| (`old ratio`, for information);
  * my own bit-level model of one instruction, written from rules/num.md's description (16 products and
    the accumulator aligned to the largest exponent field, a subnormal operand read as the smallest
    normal, 25 bits kept below it, each term toward zero, the sum toward zero to float32), chunks of 16
    consecutive k in increasing order: device = model, on both paths.

Configurations not in the author's evidence: other block shapes and warp counts, `a` transposed in the
kernel, both operands transposed, `a` converted in registers from float32, and a tutorial-03 K loop
(`acc = tl.dot(a, b, acc)` and `acc += tl.dot(a, b)`, num_stages 3 and 4). Inputs: window cases with
mixed signs (subnormal or normal operand of the large product, in `a` or in `b`), all-negative products
of subnormal operands, operands at the subnormal/normal edge, both operands subnormal, accumulator-
dominated chunks, cancellation to near zero, subnormal accumulators, operands near the maxima, random
bit patterns, and (bfloat16) products below float32's subnormal range among normal ones.
"""
import math
import random
import sys
import time
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

random.seed(25025)
torch.manual_seed(25025)
SC = 300                      # every value is held as an integer multiple of 2^-SC
F32MAX = Fr((2 ** 24 - 1) * 2 ** 104)
FMT = {  # torch dtype, triton dtype, fraction bits, emin, smallest subnormal exponent, max exponent
    "f16": (torch.float16, tl.float16, 10, -14, -24, 15),
    "bf16": (torch.bfloat16, tl.bfloat16, 7, -126, -133, 127),
}
NAME = {"f16": "float16", "bf16": "bfloat16"}


def rel16(K):
    return Fr(K + 2, 2 ** 22)


def abs16(K):
    return Fr(K + 2, 2 ** 126)


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
        if t.startswith("wgmma.mma_async") or t.startswith("mma."):
            names.add(t)
    if not names and "fma.rn.f32" in ptx:
        names.add("fma(only)")
    return ",".join(sorted(names))


def path_of(names):
    return "wgmma" if "wgmma" in names else ("mma.sync" if "mma.sync" in names else "other")


# ------------------------------------------------------------------------------------------- arithmetic

def toi(x):
    return int(math.ldexp(x, SC))


def flog2(x):
    return abs(x).bit_length() - 1 - SC


def trunc0(t, sh):
    if sh <= 0 or t == 0:
        return t
    m = (abs(t) >> sh) << sh
    return m if t > 0 else -m


def to_f32_rz(x):        # toward zero to float32 (24 bits; grid 2^-149 below 2^-126)
    if x == 0:
        return 0
    return trunc0(x, max(abs(x).bit_length() - 24, SC - 149))


def lift(x, f):
    return 0 if x == 0 else max(abs(x), 1 << (FMT[f][3] + SC))


def model_instr(s, ai, bi, f):
    terms = [(s, max(flog2(s), -126) if s else None)]
    for x, y in zip(ai, bi):
        p = (x * y) >> SC
        terms.append((p, max(flog2(x), FMT[f][3]) + max(flog2(y), FMT[f][3]) if (x and y) else None))
    fs = [e for (t, e) in terms if e is not None]
    if not fs:
        return 0
    sh = max(fs) - 25 + SC
    return to_f32_rz(sum(trunc0(t, sh) for t, _ in terms))


def check_element(ai, bi, ci, ri, f, K):
    prods = [(x * y) >> SC for x, y in zip(ai, bi)]
    lifts = [(lift(x, f) * lift(y, f)) >> SC for x, y in zip(ai, bi)]
    D = 2 ** SC
    vc = Fr(ci, D)
    L = abs(vc) + Fr(sum(lifts), D)
    if L + rel16(K) * L + abs16(K) > F32MAX:
        return (False, None, None, None)
    if ri is None:       # a non-finite result with room: outside the fact
        return (True, float("inf"), float("inf"), False)
    vr = Fr(ri, D)
    exact = vc + Fr(sum(prods), D)
    err = abs(vr - exact)
    ratio = err / (rel16(K) * L + abs16(K))
    mag = abs(vc) + Fr(sum(abs(p) for p in prods), D)
    old = err / (rel16(K) * mag + abs16(K))
    s = ci
    for j in range((K + 15) // 16):
        s = model_instr(s, ai[16 * j:16 * j + 16], bi[16 * j:16 * j + 16], f)
    return (True, float(ratio), float(old), s == ri)


# --------------------------------------------------------------------------------------------- inputs

def rs():
    return 1.0 if random.random() < 0.5 else -1.0


def rnd_val(f, e_lo, e_hi, allset=False):
    """A positive value of format f with exponent in [e_lo, e_hi] (normal) and a random (or all-set)
    fraction; e below emin gives a subnormal 2^e * (1 + frac) truncated to the subnormal grid."""
    frac, emin, esub = FMT[f][2], FMT[f][3], FMT[f][4]
    e = random.randint(e_lo, e_hi)
    m = (2 ** frac - 1) if allset else random.randrange(2 ** frac)
    if e >= emin:
        return (1 + m / 2 ** frac) * 2.0 ** e
    # subnormal: multiple of 2^esub below 2^emin
    q = random.randint(1, 2 ** (emin - esub) - 1)
    return q * 2.0 ** esub


def sub_val(f):
    emin, esub = FMT[f][3], FMT[f][4]
    return random.randint(1, 2 ** (emin - esub) - 1) * 2.0 ** esub


def allset_below(f, target):
    """The largest value of format f strictly below target (> 0)."""
    frac, emin, esub = FMT[f][2], FMT[f][3], FMT[f][4]
    e = math.frexp(target)[1] - 1          # target in [2^e, 2^(e+1))
    if target == 2.0 ** e:
        e -= 1
        cand = (2 - 2.0 ** -frac) * 2.0 ** e
    else:
        cand = math.floor(target / 2.0 ** (e - frac)) * 2.0 ** (e - frac)
        if cand >= target:
            cand -= 2.0 ** (e - frac)
    if e < emin:                           # subnormal grid
        cand = math.floor(target / 2.0 ** esub) * 2.0 ** esub
        if cand >= target:
            cand -= 2.0 ** esub
    return max(cand, 0.0)


def f32(x):
    return float(torch.tensor(x, dtype=torch.float32))


def gen(kind, M, N, K, f):
    A = [[0.0] * K for _ in range(M)]
    B = [[0.0] * N for _ in range(K)]
    C = [[0.0] * N for _ in range(M)]
    frac, emin, esub, emax = FMT[f][2], FMT[f][3], FMT[f][4], FMT[f][5]
    mn = 2.0 ** emin
    if kind == "bits":
        dt = FMT[f][0]

        def rb():
            while True:
                v = float(torch.tensor([random.randrange(65536)], dtype=torch.int32).to(torch.int16).view(dt)[0])
                if math.isfinite(v):
                    return v
        for i in range(M):
            for k in range(K):
                A[i][k] = rb()
        for k in range(K):
            for n in range(N):
                B[k][n] = rb()
        for i in range(M):
            for n in range(N):
                C[i][n] = random.choice([0.0, rs() * 2.0 ** random.randint(-60, 40)])
        return A, B, C
    if kind in ("winmix", "winmixT", "negsub", "edge"):
        # per chunk one large product at kb whose a-side (b-side for winmixT) operand is subnormal, at
        # the edge or normal; the other products just below (or within twice) the window's unit
        nch = (K + 15) // 16
        kbs = [16 * j + random.randrange(16) for j in range(nch)]
        bige = min(emax - 1, 15 if f == "f16" else 120)
        big = rnd_val(f, bige - 3, bige)
        tk = [random.randint(-3, 2) for _ in range(K)]
        side_b = kind == "winmixT"
        for k in range(K):
            for n in range(N):
                if k in kbs:
                    v = big
                elif kind == "negsub":
                    v = sub_val(f)
                else:
                    v = (rs() if kind != "edge" else 1.0) * 2.0 ** tk[k]
                B[k][n] = v
        for i in range(M):
            for j, kb in enumerate(kbs):
                if kind == "edge":
                    x = random.choice([mn - 2.0 ** esub, mn - 2 * 2.0 ** esub, mn, mn + 2.0 ** (emin - frac)])
                elif kind == "negsub" or random.random() < 0.6:
                    x = sub_val(f)
                else:
                    x = rnd_val(f, emin, emin + 3)
                A[i][kb] = x
                ex = max(math.frexp(x)[1] - 1, emin)
                eb = math.frexp(big)[1] - 1
                u = 2.0 ** (ex + eb - 25)
                for k in range(16 * j, 16 * j + 16):
                    if k == kb:
                        continue
                    if kind == "negsub":
                        A[i][k] = -sub_val(f)
                        continue
                    tgt = u if random.random() < 0.67 else 2 * u
                    av = allset_below(f, tgt / 2.0 ** tk[k])
                    A[i][k] = av * (rs() if kind != "edge" else 1.0)
            if kind == "negsub":
                for kb in kbs:
                    A[i][kb] = -A[i][kb]
            ex0 = max(math.frexp(A[i][kbs[0]])[1] - 1, emin) if A[i][kbs[0]] else emin
            u0 = 2.0 ** (ex0 + math.frexp(big)[1] - 1 - 25)
            for n in range(N):
                r = random.random()
                if r < 0.25:
                    C[i][n] = 0.0
                elif r < 0.5:
                    C[i][n] = rs() * u0 * (1 - 2.0 ** -23)
                elif r < 0.75:
                    C[i][n] = rs() * u0 * random.random()
                else:
                    C[i][n] = rs() * abs(A[i][kbs[0]] * big) * (1 + random.random())
        if side_b:      # move the structure to the b side: a' = b^T, b' = a^T, for a square block
            assert M == N
            A, B = [[B[k][i] for k in range(K)] for i in range(M)], [[A[n][k] for n in range(N)] for k in range(K)]
            C = [[C[n][i] for n in range(N)] for i in range(M)]
        C = [[f32(x) for x in row] for row in C]
        return A, B, C
    if kind == "subsub":
        for i in range(M):
            for k in range(K):
                A[i][k] = rs() * sub_val(f)
        for k in range(K):
            for n in range(N):
                B[k][n] = rs() * (sub_val(f) if random.random() < 0.5 else rnd_val(f, emin, emin + 4))
        for i in range(M):
            for n in range(N):
                C[i][n] = random.choice([0.0, rs() * random.randint(1, 2 ** 20) * 2.0 ** -149])
        return A, B, C
    if kind == "accdom":
        for i in range(M):
            e = random.randint(-30, 30)
            for n in range(N):
                C[i][n] = rs() * (2 - 2.0 ** -23) * 2.0 ** e
            u = 2.0 ** (e - 25)
            for k in range(K):
                A[i][k] = rs() * allset_below(f, u)
        for k in range(K):
            for n in range(N):
                B[k][n] = rs()
        return A, B, C
    if kind == "cancel":
        for i in range(M):
            for k in range(K):
                A[i][k] = rs() * rnd_val(f, -6, 6)
        for k in range(K):
            for n in range(N):
                B[k][n] = rs() * rnd_val(f, -6, 6)
        S = -(torch.tensor(A, dtype=torch.float64) @ torch.tensor(B, dtype=torch.float64))
        C = S.to(torch.float32).double().tolist()
        return A, B, C
    if kind == "subacc":
        for i in range(M):
            for k in range(K):
                A[i][k] = rs() * random.choice([0.0, sub_val(f), mn])
        for k in range(K):
            for n in range(N):
                B[k][n] = rs() * random.choice([sub_val(f), mn, 2.0 ** -20])
        for i in range(M):
            for n in range(N):
                C[i][n] = rs() * random.randint(1, 2 ** 23 - 1) * 2.0 ** -149
        return A, B, C
    if kind == "huge":
        top = 15 if f == "f16" else 55
        for i in range(M):
            for k in range(K):
                A[i][k] = rs() * rnd_val(f, top - 2, top, allset=random.random() < 0.5)
        for k in range(K):
            for n in range(N):
                B[k][n] = rs() * rnd_val(f, top - 2, top, allset=random.random() < 0.5)
        for i in range(M):
            for n in range(N):
                C[i][n] = f32(rs() * random.random() * 2.0 ** (2 * top - 4))
        return A, B, C
    if kind == "under":
        # bfloat16 (and float16): products below float32's subnormal range among normal ones
        for i in range(M):
            for k in range(K):
                A[i][k] = rs() * (rnd_val(f, emin, emin + 2) if random.random() < 0.7 else rnd_val(f, -3, 3))
        for k in range(K):
            for n in range(N):
                B[k][n] = rs() * rnd_val(f, emin, emin + 2)
        for i in range(M):
            for n in range(N):
                C[i][n] = random.choice([0.0, rs() * random.randint(1, 2 ** 23) * 2.0 ** -149])
        return A, B, C
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------- runs

def finish(A, B, C, o, f, K, names, stats, max_elems, label):
    M, N = len(A), len(B[0])
    O = o.double().cpu().tolist()
    Ai = [[toi(x) for x in row] for row in A]
    Bi = [[toi(B[k][n]) for k in range(K)] for n in range(N)]
    idx = [(i, n) for i in range(M) for n in range(N)]
    if len(idx) > max_elems:
        idx = random.sample(idx, max_elems)
    nchk = nmod = nold = 0
    worst = worst_old = 0.0
    bad = []
    for (i, n) in idx:
        ri = toi(O[i][n]) if math.isfinite(O[i][n]) else None
        inr, r, ro, mok = check_element(Ai[i], Bi[n], toi(C[i][n]), ri, f, K)
        if not inr:
            continue
        nchk += 1
        nmod += 1 if mok else 0
        if r > 1:
            bad.append((i, n, r))
        if ro > 1:
            nold += 1
        worst, worst_old = max(worst, r), max(worst_old, ro)
    path = path_of(names)
    st = stats.setdefault((NAME[f], path), [0, 0, 0, 0, 0.0, 0.0])
    st[0] += nchk
    st[1] += len(bad)
    st[2] += nmod
    st[3] += nold
    st[4] = max(st[4], worst)
    st[5] = max(st[5], worst_old)
    print(f"| {NAME[f]} | {label} | {names} | {nchk} | {len(bad)} | {nmod}/{nchk} | {worst:.4f} | {nold} | "
          f"{worst_old:.3f} |", flush=True)
    for b in bad[:5]:
        print("   OUTSIDE:", b, flush=True)


def run_case(kind, M, N, K, nw, f, layout, stats, max_elems=512):
    A, B, C = gen(kind, M, N, K, f)
    dt = FMT[f][0]
    a32, b32 = torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)
    a16, b16 = a32.to(dt), b32.to(dt)
    assert torch.equal(a16.float(), a32) and torch.equal(b16.float(), b32), kind
    c32 = torch.tensor(C, dtype=torch.float32).cuda()
    o = torch.empty(M, N, device="cuda")
    at = layout in ("aT", "abT")
    bt = layout == "abT"
    areg = layout == "areg32"
    ain = a32.cuda() if areg else (a16.t().contiguous().cuda() if at else a16.contiguous().cuda())
    bin_ = b16.t().contiguous().cuda() if bt else b16.contiguous().cuda()
    h = dotk[(1,)](ain, bin_, c32, o, M, N, K, at, bt, areg, FMT[f][1], num_warps=nw)
    finish(A, B, C, o, f, K, instr(h), stats, max_elems, (kind, M, N, K, nw, layout))


def run_kloop(kind, M, N, BK, KT, nw, ns, plus, f, stats):
    A, B, C = gen(kind, M, N, KT, f)
    dt = FMT[f][0]
    a16 = torch.tensor(A, dtype=torch.float32).to(dt).cuda()
    b16 = torch.tensor(B, dtype=torch.float32).to(dt).cuda()
    c32 = torch.tensor(C, dtype=torch.float32).cuda()
    o = torch.empty(M, N, device="cuda")
    h = kloop[(1,)](a16, b16, c32, o, KT, M, N, BK, plus, ns, num_warps=nw)
    finish(A, B, C, o, f, KT, instr(h), stats, 2048,
           ("kloop" + ("+=" if plus else ""), kind, M, N, f"BK={BK}", f"K={KT}", nw, f"ns={ns}"))


def main():
    t0 = time.time()
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("lr-25b 16-bit adversarial tests (DotLiftAcc bound, old bound, model on both paths)\n")
    print("| format | case (kind, M, N, K, warps, layout) | instructions | elements | outside | model = device |"
          " ratio | outside old | old ratio |")
    print("|---|---|---|---|---|---|---|---|---|")
    stats = {}
    configs = [(64, 16, 4), (64, 32, 4), (64, 64, 8), (128, 32, 4), (256, 64, 8), (64, 256, 4),
               (16, 32, 2), (32, 16, 1), (32, 128, 4), (64, 64, 2), (16, 16, 8)]
    kinds = ["winmix", "negsub", "edge", "subsub", "accdom", "cancel", "subacc", "huge", "bits", "under"]
    fmts = ["f16", "bf16"]
    if "--quick" in sys.argv:    # a smoke test, not the evidence
        configs = [(64, 16, 4), (16, 32, 2)]
    for f in fmts:
        for (M, N, nw) in configs:
            for K in (16, 64, 256):
                for kind in kinds + (["winmixT"] if M == N else []):
                    run_case(kind, M, N, K, nw, f, "plain", stats)
        for (M, N, nw) in [(64, 64, 4), (128, 128, 8), (32, 32, 2)]:
            for layout in ("aT", "abT", "areg32"):
                for K in (32, 128):
                    for kind in ("winmix", "winmixT", "negsub", "edge"):
                        run_case(kind, M, N, K, nw, f, layout, stats)
        for (M, N, nw) in [(128, 64, 4), (64, 128, 4), (128, 128, 8), (32, 64, 2)]:
            for ns in (3, 4):
                for plus in (False, True):
                    for kind in ("winmix", "negsub", "accdom"):
                        run_kloop(kind, M, N, 64, 512, nw, ns, plus, f, stats)
    print("\nTotals per format and path: elements, outside the fact, model = device, outside the old bound, "
          "largest ratio, largest old ratio")
    for key, st in sorted(stats.items()):
        print(f"| {key[0]} | {key[1]} | {st[0]} | {st[1]} | {st[2]}/{st[0]} | {st[3]} | {st[4]:.4f} | {st[5]:.3f} |")
    print(f"\n(run time {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
