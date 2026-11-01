"""lr-24a: adversarial tests of the fp8 tl.dot chunk facts DotE4m3Chunk, DotE5m2Chunk, DotE4m3E5m2Chunk and
DotE5m2E4m3Chunk (library issue L124), and of the whole-dot bound derived from them (L125).

Written for this review, independently of the author's evidence script: its own kernels, its own model of the
wgmma instruction (written from docs/rules/num.md's description, not copied), its own inputs and shapes.

The facts (lean/TVA/Rules/Num/Dot8Chunk.lean, DotChunkAcc with rho = 2^-7, sigma = 2^-12, alpha = 2^-126): per output
element, partial values s_0 = acc, ..., s_n = result (n = ceil(K/32)), chunk j the k with k / 32 = j, in increasing j,
and |s_{j+1} - (s_j + c_j)| <= rho M_j + sigma |s_j + c_j| + alpha for every M_j >= |s_j| and >= each lifted product
of the chunk (lift(x) = 0 for x = 0, else max(|x|, 2^emin): 2^-14 float8e5, 2^-6 float8e4nv). The reachable results
form the interval [lo_n, hi_n] with hi_{j+1} = hi_j + c_j + rho max(|hi_j|, m_j) + sigma |hi_j + c_j| + alpha (the step
is increasing in s_j), lo likewise; a result in it has such a chain, one outside has none.

The wgmma model checked bit for bit (docs/rules/num.md): per instruction, the accumulator and the 32 products of a
chunk; field e(a)+e(b), e(x) = max(floor(log2|x|), emin), and max(floor(log2|s|), -126) for the accumulator;
u = 2^(E-13); each term truncated toward zero to a multiple of u, summed exactly; the sum truncated toward zero to 14
significant bits.

Parts:
  P  input_precision: the PTX for every value against the default's, all four pairs, both paths.
  O  chunk membership and order on wgmma, K = 256: random positions of +B, -B and a small s among random products;
     a loaded, loaded transposed, or converted from float32 in registers; b loaded or transposed; 4 and 8 warps.
  S  shapes and operand forms outside the evidence (M or N in {1, 2, 4, 8}, 64 x 8 and 64 x 256 with 4 warps, 64 x 64
     with 2 and 8 warps, 256 x 128 with 8 warps, 32 x 128 with 4 warps), seven kinds of inputs of my own; device
     against the facts' interval (both paths) and the model (wgmma).
  L  a K loop as tutorial 03 runs it in fp8 (acc = tl.dot(a, b, acc), operands loaded in fp8, num_stages 3 and 4),
     every accumulator stored: each dot against the model and the interval.
  X  extremes: an accumulator near the float32 maximum, operands at the formats' maxima, subnormal float32
     accumulators, both operands subnormal.
  W  the whole-dot lemma's bound (Dot8Chunk.DotF8Chunk.liftAcc: relLift8 K L + absLift8 K) on every element of S.
Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I adv8.py > adv8.out
"""
import random
import re
import time
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

torch.manual_seed(24240)
random.seed(24240)
DEV = "cuda"
F32MAX = float(torch.finfo(torch.float32).max)
RHO, SIG, ALP = 2.0 ** -7, 2.0 ** -12, 2.0 ** -126
FMT = {  # torch dtype, triton dtype, emin, fraction bits, max
    "e5": (torch.float8_e5m2, tl.float8e5, -14, 2, 57344.0),
    "e4": (torch.float8_e4m3fn, tl.float8e4nv, -6, 3, 448.0),
}
PAIRS = [("e5", "e5"), ("e4", "e4"), ("e4", "e5"), ("e5", "e4")]
NEG = -(1 << 30)


@triton.jit
def kdot(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, AMODE: tl.constexpr,
         BMODE: tl.constexpr, ADT: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    if AMODE == 0:
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    elif AMODE == 1:
        a = tl.trans(tl.load(a_ptr + rk[:, None] * M + rm[None, :]))
    else:
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :]).to(ADT)
    if BMODE == 0:
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    else:
        b = tl.trans(tl.load(b_ptr + rn[:, None] * K + rk[None, :]))
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, c, input_precision=PREC))


@triton.jit
def kloop(a_ptr, b_ptr, c_ptr, accs_ptr, o_ptr, T, M: tl.constexpr, N: tl.constexpr, KB: tl.constexpr,
          STORE: tl.constexpr, NS: tl.constexpr):
    # tutorial 03's K loop: a [M, T*KB], b [T*KB, N], both fp8 in memory
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, KB)
    acc = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    for t in tl.range(0, T, num_stages=NS):
        a = tl.load(a_ptr + rm[:, None] * (T * KB) + (t * KB + rk)[None, :])
        b = tl.load(b_ptr + (t * KB + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
        if STORE:
            tl.store(accs_ptr + t * M * N + rm[:, None] * N + rn[None, :], acc)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc)


def path_of(ptx):
    if "wgmma.mma_async" in ptx:
        return "wgmma"
    if "mma.sync" in ptx:
        return "mma.sync"
    return "other"


def run(A, B, C, da, db, nw, amode=0, bmode=0, prec="tf32", want_ptx=False):
    M, K = A.shape
    N = B.shape[1]
    a8, b8 = A.to(FMT[da][0]), B.to(FMT[db][0])
    assert torch.equal(a8.float(), A) and torch.equal(b8.float(), B)
    ain = A.contiguous() if amode == 2 else (a8.t().contiguous() if amode == 1 else a8.contiguous())
    bin_ = b8.t().contiguous() if bmode == 1 else b8.contiguous()
    o = torch.empty(M, N, device=DEV)
    h = kdot[(1,)](ain, bin_, C.contiguous(), o, M, N, K, amode, bmode, FMT[da][1], prec, num_warps=nw)
    ptx = h.asm["ptx"]
    return (o, path_of(ptx), ptx) if want_ptx else (o, path_of(ptx))


# ------------------------------------------------------------------------------------------------ the model

def field(x, emin):
    return torch.clamp(torch.frexp(x.abs())[1].to(torch.int64) - 1, min=emin)


def model(A, B, C, da, db):
    """The wgmma model, float64 tensors (exact: every truncated term is an integer multiple of u below 2^16 u)."""
    M, K = A.shape
    ea, eb = field(A, FMT[da][2]), field(B, FMT[db][2])
    s = C.clone()
    for k0 in range(0, K, 32):
        a, b = A[:, k0:k0 + 32], B[k0:k0 + 32, :]
        P = a[:, None, :] * b.t()[None, :, :]
        EP = torch.where(P != 0, ea[:, k0:k0 + 32][:, None, :] + eb[k0:k0 + 32, :].t()[None, :, :],
                         torch.full(P.shape, NEG, dtype=torch.int64, device=DEV))
        es = torch.where(s != 0, torch.clamp(torch.frexp(s.abs())[1].to(torch.int64) - 1, min=-126),
                         torch.full(s.shape, NEG, dtype=torch.int64, device=DEV))
        E = torch.maximum(EP.amax(-1), es)
        live = E > NEG // 2
        u = torch.ldexp(torch.ones_like(s), torch.where(live, E - 13, torch.zeros_like(E)))
        S = (torch.trunc(s / u) + torch.trunc(P / u[..., None]).sum(-1)) * u
        S = torch.where(live, S, torch.zeros_like(S))
        e = torch.frexp(S.abs())[1].to(torch.int64) - 1
        g = torch.ldexp(torch.ones_like(S), e - 13)
        s = torch.where(S == 0, torch.zeros_like(S), torch.trunc(S / g) * g)
    return s


# ------------------------------------------------------------------------------------------------ the facts

def lift(X, f):
    return torch.where(X != 0, torch.clamp(X.abs(), min=2.0 ** FMT[f][2]), torch.zeros_like(X))


def interval(A, B, C, da, db):
    """float64: exact sums S, half-widths dp, dm of the facts' interval, the float64 error bound of S, the whole-dot
    lemma's bound, the room (relDot8, absDot8 on |acc| + sum |a b|), and L."""
    M, K = A.shape
    La, Lb = lift(A, da), lift(B, db)
    S = C.clone()
    dp = torch.zeros_like(S); dm = torch.zeros_like(S)
    mag = C.abs().clone(); L = C.abs().clone()
    for k0 in range(0, K, 32):
        P = A[:, None, k0:k0 + 32] * B[k0:k0 + 32, :].t()[None, :, :]
        c = P.sum(-1)
        m = (La[:, None, k0:k0 + 32] * Lb[k0:k0 + 32, :].t()[None, :, :]).amax(-1)
        L = L + (La[:, None, k0:k0 + 32] * Lb[k0:k0 + 32, :].t()[None, :, :]).sum(-1)
        mag = mag + P.abs().sum(-1)
        dp = dp + RHO * torch.maximum((S + dp).abs(), m) + SIG * (S + c + dp).abs() + ALP
        dm = dm + RHO * torch.maximum((S - dm).abs(), m) + SIG * (S + c - dm).abs() + ALP
        S = S + c
    n = (K + 31) // 32
    rel8 = (1 + RHO) ** n - 1; abs8 = n * (1 + RHO) ** n * ALP
    room = mag * (1 + 2.0 ** -40) * (1 + rel8) + abs8 <= F32MAX
    relL = (1 + RHO + SIG) ** n - 1; absL = n * (1 + RHO + SIG) ** n * ALP
    whole = relL * L + absL
    slack = (K + 2) * 2.0 ** -52 * mag
    return S, dp, dm, slack, room, whole


def exact_ratio(arow, bcol, c, r, da, db):
    """The interval in exact rationals, for one element: the error over the half-width on its side."""
    K = len(arow)
    rho, sig, alp = Fr(1, 2 ** 7), Fr(1, 2 ** 12), Fr(1, 2 ** 126)
    la = [0 if x == 0 else max(abs(Fr(x)), Fr(2) ** FMT[da][2]) for x in arow]
    lb = [0 if y == 0 else max(abs(Fr(y)), Fr(2) ** FMT[db][2]) for y in bcol]
    S = Fr(c); dp = dm = Fr(0)
    for k0 in range(0, K, 32):
        ks = range(k0, min(K, k0 + 32))
        cj = sum(Fr(arow[k]) * Fr(bcol[k]) for k in ks)
        mj = max(la[k] * lb[k] for k in ks)
        dp = dp + rho * max(abs(S + dp), mj) + sig * abs(S + cj + dp) + alp
        dm = dm + rho * max(abs(S - dm), mj) + sig * abs(S + cj - dm) + alp
        S = S + cj
    e = Fr(r) - S
    return e / dp if e >= 0 else -e / dm


class Tally:
    def __init__(self):
        self.n = 0; self.room = 0; self.out = 0; self.nonfin = 0; self.eq = 0; self.cmp = 0; self.worst = 0.0
        self.exact = 0; self.where = None; self.wout = 0; self.wworst = 0.0

    def add(self, A, B, C, R, da, db, path, label):
        Ad, Bd, Cd, Rd = A.double(), B.double(), C.double(), R.double()
        S, dp, dm, slack, room, whole = interval(Ad, Bd, Cd, da, db)
        fin = torch.isfinite(Rd)
        err = Rd - S
        ratio = torch.where(err >= 0, (err.abs() + slack) / (dp * (1 - 2.0 ** -40)), (err.abs() + slack) / (dm * (1 - 2.0 ** -40)))
        ratio = torch.where(room & fin, ratio, torch.zeros_like(ratio))
        self.n += R.numel(); self.room += int(room.sum()); self.nonfin += int((room & ~fin).sum())
        for i, n in (ratio > 0.9).nonzero().tolist():
            ratio[i, n] = float(exact_ratio(A[i].tolist(), B[:, n].tolist(), float(C[i, n]), float(R[i, n]), da, db))
            self.exact += 1
        self.out += int((ratio > 1).sum())
        self.worst = max(self.worst, float(ratio.max()))
        wr = torch.where(room & fin, (err.abs() + slack) / whole, torch.zeros_like(whole))
        self.wout += int((wr > 1).sum()); self.wworst = max(self.wworst, float(wr.max()))
        if path == "wgmma":
            sim = model(Ad, Bd, Cd, da, db)
            self.cmp += R.numel(); self.eq += int((sim == Rd).sum())
            if int((sim != Rd).sum()) and self.where is None:
                i, n = (sim != Rd).nonzero()[0].tolist()
                self.where = (label, i, n, float(Rd[i, n]), float(sim[i, n]))

    def merge(self, t):
        for k in ("n", "room", "out", "nonfin", "eq", "cmp", "exact", "wout"):
            setattr(self, k, getattr(self, k) + getattr(t, k))
        self.worst = max(self.worst, t.worst); self.wworst = max(self.wworst, t.wworst)


# ------------------------------------------------------------------------------------------------ inputs

def q(x, f):
    return x.clamp(-FMT[f][4], FMT[f][4]).to(FMT[f][0]).float()


def sgn(*shape):
    return torch.where(torch.rand(*shape, device=DEV) < 0.5, -1.0, 1.0)


def pow2(lo, hi, *shape):
    return torch.exp2(torch.randint(lo, hi + 1, shape, device=DEV).float())


def rsub(f, *shape):
    """random nonzero subnormal values of format f, random sign"""
    fr, emin = FMT[f][3], FMT[f][2]
    return torch.randint(1, 2 ** fr, shape, device=DEV).float() * 2.0 ** (emin - fr) * sgn(*shape)


def gen(kind, M, N, K, da, db):
    ea, eb = FMT[da][2], FMT[db][2]
    if kind == "randn":
        a = torch.randn(M, K, device=DEV) * pow2(-2, 2, M, K); b = torch.randn(K, N, device=DEV) * pow2(-2, 2, K, N)
        c = torch.randn(M, N, device=DEV) * pow2(-4, 6, M, N)
    elif kind == "subw":
        # random subnormal a at random k per row, large powers of two in b at random k per column; the other
        # products random, straddling the unit of the window the lifted product sets; random accumulators
        bigexp = 8 if db == "e4" else 15
        big = pow2(bigexp - 3, bigexp, K, N) * sgn(K, N)
        b = torch.where(torch.rand(K, N, device=DEV) < 0.15, big, (1 + torch.rand(K, N, device=DEV)) * pow2(-6, -2, K, N) * sgn(K, N))
        E = ea + bigexp - 1
        # a small: products a*b around 2^(E-13): a ~ 2^(E-13+4+d)
        a = (1 + torch.rand(M, K, device=DEV)) * pow2(E - 13 + 2, E - 13 + 6, M, K) * sgn(M, K)
        a = torch.where(torch.rand(M, K, device=DEV) < 0.2, rsub(da, M, K), a)
        mode = torch.randint(0, 3, (M, N), device=DEV)
        c = torch.where(mode == 0, torch.zeros(M, N, device=DEV),
                        torch.where(mode == 1, 2.0 ** (E - 13) * torch.rand(M, N, device=DEV) * sgn(M, N),
                                    2.0 ** E * (1 + torch.rand(M, N, device=DEV)) * sgn(M, N)))
    elif kind == "dsub":
        # both operands subnormal at many places; some normal operands
        a = torch.where(torch.rand(M, K, device=DEV) < 0.6, rsub(da, M, K), torch.randn(M, K, device=DEV) * 2.0 ** (ea + 2))
        b = torch.where(torch.rand(K, N, device=DEV) < 0.6, rsub(db, K, N), torch.randn(K, N, device=DEV) * 2.0 ** (eb + 2))
        c = torch.randint(-20, 21, (M, N), device=DEV).float() * 2.0 ** -140 * (torch.rand(M, N, device=DEV) < 0.5)
    elif kind == "mixsign":
        a = (1 + torch.rand(M, K, device=DEV)) * pow2(-6, -1, M, K) * sgn(M, K)
        b = (1 + torch.rand(K, N, device=DEV)) * pow2(-8, -4, K, N) * sgn(K, N)
        for j in range(0, K, 32):
            p1, p2 = random.sample(range(j, min(K, j + 32)), 2)
            v = 2.0 ** random.randint(0, 4)
            a[:, p1] = v; a[:, p2] = -v; b[p1, :] = 2.0 ** random.randint(0, 3); b[p2, :] = b[p1, :]
        c = torch.randn(M, N, device=DEV) * pow2(-14, 4, M, N)
    elif kind == "accneg":
        a = torch.rand(M, K, device=DEV) + 0.5; b = torch.rand(K, N, device=DEV) + 0.5
        a, b = q(a, da), q(b, db)
        c = -(a @ b) * (0.875 + 0.25 * torch.rand(M, N, device=DEV))
    elif kind == "accwin":
        # an accumulator that sets the window (largest field) against products just below its unit
        a = (1 + torch.rand(M, K, device=DEV)) * pow2(-4, 0, M, K) * sgn(M, K)
        b = (1 + torch.rand(K, N, device=DEV)) * pow2(-4, 0, K, N)
        c = (1 + torch.rand(M, N, device=DEV)) * 2.0 ** 11 * sgn(M, N)
    elif kind == "rbits":
        def rb(f, *shape):
            x = torch.randint(0, 256, shape, device=DEV, dtype=torch.uint8).view(FMT[f][0]).float()
            return torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        a, b = rb(da, M, K), rb(db, K, N)
        c = torch.randn(M, N, device=DEV) * pow2(-40, 30, M, N)
    else:
        raise ValueError(kind)
    return q(a, da).contiguous(), q(b, db).contiguous(), c.float().contiguous()


KINDS = ["randn", "subw", "dsub", "mixsign", "accneg", "accwin", "rbits"]
FORM = {0: "ld", 1: "ldT", 2: "reg"}


# ------------------------------------------------------------------------------------------------ parts

def part_p(out):
    out("## Part P: input_precision\n")
    out("| operands | M x N, warps | K | path | input_precision | PTX equal to the default's |")
    out("|---|---|---|---|---|---|")
    for da, db in PAIRS:
        for (M, N, nw, K) in [(64, 64, 4, 64), (16, 16, 1, 64)]:
            A = q(torch.randn(M, K, device=DEV), da); B = q(torch.randn(K, N, device=DEV), db)
            C = torch.zeros(M, N, device=DEV)
            _, path, p0 = run(A, B, C, da, db, nw, want_ptx=True)
            for prec in ("tf32x3", "ieee", "bf16x3", "bf16x6"):
                _, _, p1 = run(A, B, C, da, db, nw, prec=prec, want_ptx=True)
                out(f"| {da} x {db} | {M} x {N}, {nw} | {K} | {path} | {prec} | {p1 == p0} |")
    out("")


def part_o(out):
    out("## Part O: chunk membership and order on wgmma (K = 256)\n")
    out("Per element (row of a): +B and -B at random k, a small s at a random k, the other products random and smaller;"
        " B/s = 2^14 (s lost unless its chunk comes after both B's chunks) or 2^8 (kept). Device against the model.\n")
    out("| operands | M x N, warps | a | b | B/s | path | elements | model = device |")
    out("|---|---|---|---|---|---|---|---|")
    K = 256
    for da, db in PAIRS:
        for (M, N, nw) in [(64, 64, 4), (128, 64, 8)]:
            for (am, bm) in [(0, 0), (1, 0), (2, 1), (1, 1)]:
                for lg in (14, 8):
                    eq = tot = 0
                    path = None
                    for rep in range(2):
                        A = (1 + torch.rand(M, K, device=DEV)) * 2.0 ** -5 * sgn(M, K)
                        B = torch.ones(K, N, device=DEV) * torch.where(torch.rand(K, 1, device=DEV) < 0.5, 1.0, 2.0)
                        for i in range(M):
                            p1, p2, p3 = random.sample(range(K), 3)
                            A[i, p1] = 2.0 ** 7; A[i, p2] = -(2.0 ** 7); A[i, p3] = 2.0 ** (7 - lg)
                        A, B = q(A, da), q(B, db)
                        C = torch.zeros(M, N, device=DEV)
                        R, path = run(A, B, C, da, db, nw, am, bm)
                        sim = model(A.double(), B.double(), C.double(), da, db)
                        eq += int((sim == R.double()).sum()); tot += R.numel()
                    out(f"| {da} x {db} | {M} x {N}, {nw} | {FORM[am]} | {FORM[bm]} | 2^{lg} | {path} | {tot} | {eq} |")
    out("")


def part_s(out):
    out("## Part S: shapes and operand forms outside the evidence\n")
    out("| operands | M x N, warps | K | a | b | path | elements | in room | outside the facts | model = device | largest ratio | whole-dot lemma: outside (largest ratio) |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|")
    shapes = [(1, 16, 1), (4, 16, 1), (8, 8, 1), (16, 2, 1), (16, 16, 8), (32, 128, 4), (64, 64, 2), (64, 8, 4),
              (64, 256, 4), (64, 64, 8), (256, 128, 8)]
    forms = [(0, 0), (1, 0), (2, 1), (1, 1)]
    tot = Tally()
    for da, db in PAIRS:
        for (M, N, nw) in shapes:
            for K in (32, 128, 512):
                if M * N >= 64 * 64 and K > 128:
                    continue
                for (am, bm) in forms:
                    t = Tally()
                    path = None
                    try:
                        for kind in KINDS:
                            for rep in range(2):
                                A, B, C = gen(kind, M, N, K, da, db)
                                R, path = run(A, B, C, da, db, nw, am, bm)
                                t.add(A, B, C, R, da, db, path, (da, db, M, N, nw, K, am, bm, kind))
                    except triton.runtime.errors.OutOfResources:
                        out(f"| {da} x {db} | {M} x {N}, {nw} | {K} | {FORM[am]} | {FORM[bm]} | rejected (out of resources) | | | | | | |")
                        continue
                    mtxt = f"{t.eq}/{t.cmp}" if t.cmp else "-"
                    out(f"| {da} x {db} | {M} x {N}, {nw} | {K} | {FORM[am]} | {FORM[bm]} | {path} | {t.n} | {t.room} | {t.out + t.nonfin} | {mtxt} | {t.worst:.4f} | {t.wout} ({t.wworst:.4f}) |"
                        + (f" first mismatch {t.where}" if t.where else ""))
                    tot.merge(t)
    out(f"\nPart S totals: {tot.n} elements, {tot.room} in room, {tot.out + tot.nonfin} outside the facts, model = device"
        f" on {tot.eq}/{tot.cmp} (wgmma), largest ratio {tot.worst:.4f}, {tot.exact} recomputed exactly; whole-dot"
        f" lemma's bound: {tot.wout} outside, largest ratio {tot.wworst:.4f}\n")


def part_l(out):
    out("## Part L: tutorial 03's K loop in fp8 (operands loaded in fp8, acc = tl.dot(a, b, acc))\n")
    out("| operands | M x N x KB, warps, stages | T | kind | path | dots x elements | outside the facts | model = device (per dot) | largest ratio | final equal without the stores |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for da, db in PAIRS:
        for (M, N, KB, nw, ns) in [(128, 256, 128, 8, 3), (128, 128, 64, 4, 4), (64, 64, 128, 4, 3), (32, 32, 64, 2, 3)]:
            for kind in ("randn", "subw", "accwin"):
                T = 8
                A = torch.zeros(M, T * KB, device=DEV); B = torch.zeros(T * KB, N, device=DEV)
                for t in range(T):
                    a, b, _ = gen(kind, M, N, KB, da, db)
                    A[:, t * KB:(t + 1) * KB] = a; B[t * KB:(t + 1) * KB, :] = b
                C = torch.randn(M, N, device=DEV) * 4
                a8 = A.to(FMT[da][0]).contiguous(); b8 = B.to(FMT[db][0]).contiguous()
                accs = torch.empty(T, M, N, device=DEV); o = torch.empty(M, N, device=DEV); o2 = torch.empty(M, N, device=DEV)
                try:
                    h = kloop[(1,)](a8, b8, C, accs, o, T, M, N, KB, True, ns, num_warps=nw)
                    kloop[(1,)](a8, b8, C, accs, o2, T, M, N, KB, False, ns, num_warps=nw)
                except triton.runtime.errors.OutOfResources:
                    out(f"| {da} x {db} | {M} x {N} x {KB}, {nw}, {ns} | {T} | {kind} | rejected (out of resources) | | | | | |")
                    continue
                path = path_of(h.asm["ptx"])
                tl_ = Tally()
                prev = C
                for t in range(T):
                    tl_.add(A[:, t * KB:(t + 1) * KB], B[t * KB:(t + 1) * KB, :], prev, accs[t], da, db, path, ("loop", t))
                    prev = accs[t]
                mtxt = f"{tl_.eq}/{tl_.cmp}" if tl_.cmp else "-"
                out(f"| {da} x {db} | {M} x {N} x {KB}, {nw}, {ns} | {T} | {kind} | {path} | {T} x {M * N} | {tl_.out + tl_.nonfin} | {mtxt} | {tl_.worst:.4f} | {torch.equal(o, o2) and torch.equal(o, accs[T - 1])} |")
    out("")


def part_x(out):
    out("## Part X: extremes\n")
    out("| case | operands | path | K | in room | outside the facts | non-finite in room | model = device | largest ratio |")
    out("|---|---|---|---|---|---|---|---|---|")
    for da, db in PAIRS:
        for (M, N, nw) in [(64, 64, 4), (16, 16, 1)]:
            for K in (32, 256):
                n = (K + 31) // 32
                rel8 = (1 + RHO) ** n - 1
                cases = []
                cmax = F32MAX * (1 - 2 * rel8)
                A = q(torch.rand(M, K, device=DEV) + 1, da); B = q(torch.rand(K, N, device=DEV) + 1, db)
                C = (torch.full((M, N), cmax, device=DEV) * sgn(M, N))
                cases.append(("accumulator near the float32 maximum", A, B, C))
                A = q((torch.rand(M, K, device=DEV) * 0.5 + 0.5) * FMT[da][4] * sgn(M, K), da)
                B = q((torch.rand(K, N, device=DEV) * 0.5 + 0.5) * FMT[db][4], db)
                C = torch.randn(M, N, device=DEV) * 2.0 ** 30
                cases.append(("operands near the maxima", A, B, C))
                A = q(rsub(da, M, K), da); B = q(rsub(db, K, N), db)
                C = torch.randint(-2 ** 20, 2 ** 20, (M, N), device=DEV).float() * 2.0 ** -149
                cases.append(("both subnormal, subnormal float32 accumulators", A, B, C))
                A = q(torch.where(torch.rand(M, K, device=DEV) < 0.03, torch.randn(M, K, device=DEV), torch.zeros(M, K, device=DEV)), da)
                B = q(torch.randn(K, N, device=DEV), db)
                C = torch.randint(-2 ** 22, 2 ** 22, (M, N), device=DEV).float() * 2.0 ** -149
                cases.append(("mostly zero products, subnormal accumulators", A, B, C))
                for (name, A, B, C) in cases:
                    t = Tally()
                    R, path = run(A, B, C, da, db, nw)
                    t.add(A, B, C, R, da, db, path, name)
                    mtxt = f"{t.eq}/{t.cmp}" if t.cmp else "-"
                    out(f"| {name} | {da} x {db} | {path} | {K} | {t.room} | {t.out} | {t.nonfin} | {mtxt} | {t.worst:.4f} |"
                        + (f" first mismatch {t.where}" if t.where else ""))
    out("")


def main():
    t0 = time.time()

    def out(s=""):
        print(s, flush=True)

    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    part_p(out)
    part_x(out)
    part_o(out)
    part_l(out)
    part_s(out)
    out(f"(run time {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
