"""lr-25a: adversarial tests of the restated 16-bit tl.dot facts DotF16 and DotBF16 (library issue L126).

Written for this review, independently of the author's evidence script: its own kernel, its own model of the
instruction (written from docs/rules/num.md's description, not copied), its own inputs and shapes.

The facts (lean/TVA/Rules/Num/Sm90.lean, DotLiftAcc): per output element, with L = |acc| + sum_k lift(a_k) lift(b_k),
lift(x) = 0 for x = 0 and max(|x|, 2^emin) otherwise (2^-14 float16, 2^-126 bfloat16), and room
L + rel L + abs <= float32 max:  |r - (acc + sum a b)| <= rel L + abs,  rel = (K+2) 2^-22,  abs = (K+2) 2^-126.

The model checked bit for bit (docs/rules/num.md, "tl.dot (the chunk model)"): per instruction, the accumulator and
16 products of 16 consecutive k; field e(a)+e(b) with e(x) = max(floor(log2|x|), emin), and max(floor(log2|s|), -126)
for the accumulator; u = 2^(E-25) for the largest field E; each term truncated toward zero to a multiple of u, summed
exactly, the sum truncated toward zero to float32; instructions in increasing k.

Parts:
  S  shapes and operand forms the evidence does not run: M or N in {1, 2, 4, 8} (padded mma.sync), 64 x 8 and
     64 x 256 with 4 warps, 64 x 64 with 2 and 8 warps, 16 x 16 with 8 warps, 32 x 128 with 4 warps, 256 x 128 with 8
     warps; a loaded, loaded transposed ([K, M] then tl.trans), or converted from float32 in registers; b loaded or
     transposed. Seven kinds of inputs of my own (below). Device against the model and against the fact.
  O  chunk membership and order: random positions of +B, -B and a small s (B/s = 2^28 or 2^20) among random small
     products, K = 256, both paths, every operand form; device against the model.
  X  extremes: an accumulator near the float32 maximum (the room's edge), bfloat16 products near 2^127, subnormal
     float32 results, both operands subnormal.
  P  input_precision: the PTX for every value against the default's.
Run from this folder: CUDA_VISIBLE_DEVICES=0 <tva>/.venv/bin/python -I adv16.py > adv16.out
"""
import random
import re
import sys
import time
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

torch.manual_seed(25250)
random.seed(25250)
DEV = "cuda"
F32MAX = float(torch.finfo(torch.float32).max)
FMT = {  # torch dtype, triton dtype, emin, fraction bits, largest finite exponent
    "f16": (torch.float16, tl.float16, -14, 10, 15),
    "bf16": (torch.bfloat16, tl.bfloat16, -126, 7, 127),
}
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


def path_of(ptx):
    if "wgmma.mma_async" in ptx:
        return "wgmma"
    if "mma.sync" in ptx:
        return "mma.sync"
    return "other"


def run(A, B, C, f, nw, amode=0, bmode=0, prec="tf32", want_ptx=False):
    """A [M,K], B [K,N] float32 holding values of format f exactly; C [M,N] float32."""
    M, K = A.shape
    N = B.shape[1]
    dt = FMT[f][0]
    a16, b16 = A.to(dt), B.to(dt)
    assert torch.equal(a16.float(), A) and torch.equal(b16.float(), B)
    ain = A.contiguous() if amode == 2 else (a16.t().contiguous() if amode == 1 else a16.contiguous())
    bin_ = b16.t().contiguous() if bmode == 1 else b16.contiguous()
    o = torch.empty(M, N, device=DEV)
    h = kdot[(1,)](ain, bin_, C.contiguous(), o, M, N, K, amode, bmode, FMT[f][1], prec, num_warps=nw)
    ptx = h.asm["ptx"]
    return (o, path_of(ptx), ptx) if want_ptx else (o, path_of(ptx))


# ------------------------------------------------------------------------------------------------ the model

def field(x, emin):
    return torch.clamp(torch.frexp(x.abs())[1].to(torch.int64) - 1, min=emin)


def to_f32_trunc(S):
    e = torch.frexp(S.abs())[1].to(torch.int64) - 1
    g = torch.ldexp(torch.ones_like(S), torch.clamp(e - 23, min=-149))
    return torch.where(S == 0, torch.zeros_like(S), torch.trunc(S / g) * g)


def model(A, B, C, f):
    """float64 tensors; exact (terms are integer multiples of u below 2^28 u)."""
    emin = FMT[f][2]
    M, K = A.shape
    ea, eb = field(A, emin), field(B, emin)
    s = C.clone()
    for k0 in range(0, K, 16):
        a, b = A[:, k0:k0 + 16], B[k0:k0 + 16, :]
        P = a[:, None, :] * b.t()[None, :, :]
        EP = torch.where(P != 0, ea[:, k0:k0 + 16][:, None, :] + eb[k0:k0 + 16, :].t()[None, :, :],
                         torch.full(P.shape, NEG, dtype=torch.int64, device=DEV))
        es = torch.where(s != 0, torch.clamp(torch.frexp(s.abs())[1].to(torch.int64) - 1, min=-126),
                         torch.full(s.shape, NEG, dtype=torch.int64, device=DEV))
        E = torch.maximum(EP.amax(-1), es)
        live = E > NEG // 2
        u = torch.ldexp(torch.ones_like(s), torch.where(live, E - 25, torch.zeros_like(E)))
        S = (torch.trunc(s / u) + torch.trunc(P / u[..., None]).sum(-1)) * u
        s = to_f32_trunc(torch.where(live, S, torch.zeros_like(S)))
    return s


# ------------------------------------------------------------------------------------------------ the fact

def lift(X, f):
    return torch.where(X != 0, torch.clamp(X.abs(), min=2.0 ** FMT[f][2]), torch.zeros_like(X))


def exact_ratio(arow, bcol, c, r, f):
    mn = Fr(2) ** FMT[f][2]
    K = len(arow)
    S = Fr(c) + sum(Fr(x) * Fr(y) for x, y in zip(arow, bcol))
    L = abs(Fr(c)) + sum((0 if x == 0 else max(abs(Fr(x)), mn)) * (0 if y == 0 else max(abs(Fr(y)), mn))
                         for x, y in zip(arow, bcol))
    rel = Fr(K + 2, 2 ** 22)
    ab = Fr(K + 2, 1) / Fr(2) ** 126
    room = L + rel * L + ab <= Fr(F32MAX)
    return abs(Fr(r) - S) / (rel * L + ab), room


class Tally:
    def __init__(self):
        self.n = 0; self.room = 0; self.out = 0; self.nonfin = 0; self.eq = 0; self.cmp = 0; self.worst = 0.0
        self.exact = 0; self.where = None

    def add(self, A, B, C, R, f, label, check_model=True):
        Ad, Bd, Cd, Rd = A.double(), B.double(), C.double(), R.double()
        K = A.shape[1]
        P = Ad[:, None, :] * Bd.t()[None, :, :]
        S = Cd + P.sum(-1)
        mag = Cd.abs() + P.abs().sum(-1)
        L = Cd.abs() + (lift(Ad, f)[:, None, :] * lift(Bd, f).t()[None, :, :]).sum(-1)
        rel, ab = (K + 2) * 2.0 ** -22, (K + 2) * 2.0 ** -126
        slack = (K + 2) * 2.0 ** -52 * mag          # float64 summation error of S (and of L, below)
        room = L * (1 + 2.0 ** -40) * (1 + rel) + ab <= F32MAX
        bound = (rel * L * (1 - (K + 2) * 2.0 ** -52) + ab)
        fin = torch.isfinite(Rd)
        ratio = torch.where(room & fin, ((Rd - S).abs() + slack) / bound, torch.zeros_like(S))
        self.n += R.numel(); self.room += int(room.sum()); self.nonfin += int((room & ~fin).sum())
        if check_model:
            sim = model(Ad, Bd, Cd, f)
            self.cmp += R.numel(); self.eq += int((sim == Rd).sum())
            if int((sim != Rd).sum()) and self.where is None:
                i, n = (sim != Rd).nonzero()[0].tolist()
                self.where = (label, i, n, float(Rd[i, n]), float(sim[i, n]))
        hot = (ratio > 0.9).nonzero().tolist()
        for i, n in hot:
            rq, rm = exact_ratio(A[i].tolist(), B[:, n].tolist(), float(C[i, n]), float(R[i, n]), f)
            self.exact += 1
            ratio[i, n] = float(rq) if rm else 0.0
        self.out += int((ratio > 1).sum())
        m = float(ratio.max())
        if m > self.worst:
            self.worst = m


# ------------------------------------------------------------------------------------------------ inputs

def q(x, f):
    mx = float(torch.finfo(FMT[f][0]).max)
    return x.clamp(-mx, mx).to(FMT[f][0]).float()


def sgn(*shape):
    return torch.where(torch.rand(*shape, device=DEV) < 0.5, -1.0, 1.0)


def pow2(lo, hi, *shape):
    return torch.exp2(torch.randint(lo, hi + 1, shape, device=DEV).float())


def gen(kind, M, N, K, f):
    dt, _, emin, frac, emax = FMT[f]
    mn = 2.0 ** emin
    smin = 2.0 ** (emin - frac)
    if kind == "randn":
        a = torch.randn(M, K, device=DEV) * pow2(-3, 3, M, K)
        b = torch.randn(K, N, device=DEV) * pow2(-3, 3, K, N)
        c = torch.randn(M, N, device=DEV) * pow2(-6, 6, M, N)
    elif kind == "soft":
        # exp2 of minus a scaled |randn|, cast: many subnormal and zero operands in a (tutorial 06's p)
        a = torch.exp2(-torch.randn(M, K, device=DEV).abs() * pow2(2, 7, M, 1))
        b = torch.randn(K, N, device=DEV)
        c = torch.randn(M, N, device=DEV) * (torch.rand(M, N, device=DEV) < 0.5)
    elif kind in ("subw", "subwacc"):
        # per row, random subnormal a at random k; per column, random large powers of two b at random k: an element
        # meets a subnormal x large product where they coincide; the rest random small products straddling the unit
        # of the lifted window, with random bits and signs; accumulators 0, near u, or near the large product
        eb = (12 if f == "f16" else 100)
        big = pow2(eb, emax, K, N) * sgn(K, N)
        bigmask = torch.rand(K, N, device=DEV) < 0.2
        sub = torch.randint(1, 2 ** frac, (M, K), device=DEV).float() * smin * sgn(M, K)
        submask = torch.rand(M, K, device=DEV) < 0.2
        Etyp = emin + (eb + emax) // 2                 # a typical lifted field
        # small products near 2^(Etyp-25), with random bits and signs: a = (1..2) 2^ea, ea in e0 .. e0+2, times
        # b = (1..2) 2^(Etyp-25-(e0+1)+d), d in -3 .. 3
        e0 = max(emin, Etyp - 25 - 12)
        ea = torch.randint(e0, e0 + 3, (M, K), device=DEV).float()
        a = torch.where(submask, sub, (1 + torch.rand(M, K, device=DEV)) * torch.exp2(ea) * sgn(M, K))
        bsmall = (1 + torch.rand(K, N, device=DEV)) * torch.exp2(Etyp - 25 - (e0 + 1) + torch.randint(-3, 4, (K, N), device=DEV).float())
        b = torch.where(bigmask, big, bsmall)
        u0 = 2.0 ** (Etyp - 25)
        mode = torch.randint(0, 4, (M, N), device=DEV)
        c = torch.where(mode == 0, torch.zeros(M, N, device=DEV),
                        torch.where(mode == 1, u0 * torch.rand(M, N, device=DEV) * sgn(M, N),
                                    2.0 ** Etyp * (1 + torch.rand(M, N, device=DEV)) * sgn(M, N)))
        if kind == "subwacc":
            c = c * 2.0 ** 6
    elif kind == "dsub":
        # both operands subnormal at random places, a few larger products to set the window
        a = torch.where(torch.rand(M, K, device=DEV) < 0.5, torch.randint(-2 ** frac + 1, 2 ** frac, (M, K), device=DEV).float() * smin,
                        torch.randn(M, K, device=DEV) * mn * 4)
        b = torch.where(torch.rand(K, N, device=DEV) < 0.5, torch.randint(-2 ** frac + 1, 2 ** frac, (K, N), device=DEV).float() * smin,
                        torch.randn(K, N, device=DEV) * mn * 4)
        bigk = torch.rand(K, device=DEV) < 0.1
        b[bigk] = b[bigk] * 2.0 ** (10 if f == "f16" else 100) * (1.0 if f == "f16" else 2.0 ** 100)
        c = torch.randn(M, N, device=DEV) * mn * mn * (torch.rand(M, N, device=DEV) < 0.5) if f == "f16" else \
            torch.randint(-5, 6, (M, N), device=DEV).float() * 2.0 ** -149
    elif kind == "mixsign":
        # +B and -B in random places (cancelling), random small products of both signs just around B's window unit
        a = (1 + torch.rand(M, K, device=DEV)) * pow2(-6, 0, M, K) * sgn(M, K)
        b = (1 + torch.rand(K, N, device=DEV)) * pow2(-20, -14, K, N) * sgn(K, N)
        for j in range(0, K, 16):
            p1, p2 = random.sample(range(j, min(K, j + 16)), 2)
            a[:, p1] = 2.0 ** random.randint(-2, 4); a[:, p2] = -a[:, p1]
            b[p1, :] = 1.0; b[p2, :] = 1.0
        c = torch.randn(M, N, device=DEV) * pow2(-22, 4, M, N)
    elif kind == "accneg":
        # a large accumulator against products of the opposite sign
        a = torch.rand(M, K, device=DEV) + 0.5
        b = torch.rand(K, N, device=DEV) + 0.5
        c = -(a.sum(1, keepdim=True) @ torch.ones(1, N, device=DEV)) * (0.75 + 0.5 * torch.rand(M, N, device=DEV))
        c = c * (1 + 2.0 ** -20 * torch.randint(0, 64, (M, N), device=DEV))
    elif kind == "rbits":
        def rb(*shape):
            x = torch.randint(-32768, 32768, shape, device=DEV, dtype=torch.int16).view(dt).float()
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
            if f == "bf16":
                x = torch.where(x.abs() > 2.0 ** 40, x * 2.0 ** -90, x)
            return x
        a, b = rb(M, K), rb(K, N)
        c = torch.randn(M, N, device=DEV) * pow2(-40, 40, M, N)
    else:
        raise ValueError(kind)
    return q(a, f).contiguous(), q(b, f).contiguous(), c.float().contiguous()


KINDS = ["randn", "soft", "subw", "subwacc", "dsub", "mixsign", "accneg", "rbits"]


# ------------------------------------------------------------------------------------------------ parts

def part_s(out):
    out("## Part S: shapes and operand forms outside the evidence\n")
    out("| format | M x N, warps | K | a | b | path | elements | in room | outside the fact | model = device | largest ratio |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    shapes = [(1, 16, 1), (2, 16, 1), (4, 16, 1), (8, 16, 1), (16, 1, 1), (16, 2, 1), (16, 4, 1), (16, 8, 1), (8, 8, 1),
              (16, 16, 8), (32, 128, 4), (64, 64, 2), (64, 8, 4), (64, 256, 4), (64, 64, 8), (256, 128, 8)]
    forms = [(0, 0), (1, 0), (2, 1), (1, 1)]
    allc = Tally()
    for f in ("f16", "bf16"):
        for (M, N, nw) in shapes:
            for K in (16, 64, 256):
                if M * N >= 64 * 64 and K > 64:
                    continue
                for (am, bm) in forms:
                    t = Tally()
                    path = None
                    try:
                        for kind in KINDS:
                            for rep in range(2):
                                A, B, C = gen(kind, M, N, K, f)
                                R, path = run(A, B, C, f, nw, am, bm)
                                t.add(A, B, C, R, f, (f, M, N, nw, K, am, bm, kind))
                    except triton.runtime.errors.OutOfResources:
                        out(f"| {f} | {M} x {N}, {nw} | {K} | {am} | {bm} | rejected (out of resources) | | | | | |")
                        continue
                    out(f"| {f} | {M} x {N}, {nw} | {K} | {'ld,ldT,reg'.split(',')[am]} | {'ld,ldT'.split(',')[bm]} | {path} | {t.n} | {t.room} | {t.out + t.nonfin} | {t.eq}/{t.cmp} | {t.worst:.4f} |"
                        + (f" first mismatch {t.where}" if t.where else ""))
                    for k in ("n", "room", "out", "nonfin", "eq", "cmp", "exact"):
                        setattr(allc, k, getattr(allc, k) + getattr(t, k))
                    allc.worst = max(allc.worst, t.worst)
    out(f"\nPart S totals: {allc.n} elements, {allc.room} in room, {allc.out + allc.nonfin} outside the fact,"
        f" model = device on {allc.eq}/{allc.cmp}, largest ratio {allc.worst:.4f}, {allc.exact} recomputed exactly\n")


def part_o(out):
    out("## Part O: chunk membership and order (K = 256)\n")
    out("Per element: +B and -B at random k, a small s at a random k (B/s = 2^28: lost unless added after both; 2^20:"
        " kept), and random products far below. Device against the model (16 consecutive k per instruction, increasing).\n")
    out("| format | M x N, warps | a | b | B/s | path | elements | model = device |")
    out("|---|---|---|---|---|---|---|---|")
    K = 256
    for f in ("f16", "bf16"):
        for (M, N, nw) in [(64, 64, 4), (128, 64, 8), (16, 16, 1), (32, 32, 4)]:
            for (am, bm) in [(0, 0), (1, 0), (2, 1), (1, 1)]:
                for lg in (28, 20):
                    eq = tot = 0
                    path = None
                    for rep in range(2):
                        A = torch.zeros(M, K, device=DEV); B = torch.zeros(K, N, device=DEV)
                        # a different placement per row of a; b is all ones on the placements' k
                        A += (1 + torch.rand(M, K, device=DEV)) * 2.0 ** -14 * sgn(M, K)
                        B += 1.0
                        for i in range(M):
                            p1, p2, p3 = random.sample(range(K), 3)
                            A[i, p1] = 2.0 ** 8; A[i, p2] = -(2.0 ** 8); A[i, p3] = 2.0 ** (8 - lg)
                        A = q(A, f)
                        C = torch.zeros(M, N, device=DEV)
                        R, path = run(A, B, C, f, nw, am, bm)
                        sim = model(A.double(), B.double(), C.double(), f)
                        eq += int((sim == R.double()).sum()); tot += R.numel()
                    out(f"| {f} | {M} x {N}, {nw} | {'ld,ldT,reg'.split(',')[am]} | {'ld,ldT'.split(',')[bm]} | 2^{lg} | {path} | {tot} | {eq} |")
    out("")


def part_x(out):
    out("## Part X: extremes\n")
    out("| case | format | path | K | elements in room | outside the fact | non-finite in room | model = device | largest ratio |")
    out("|---|---|---|---|---|---|---|---|---|")
    for f in ("f16", "bf16"):
        for (M, N, nw) in [(64, 64, 4), (16, 16, 1)]:
            for K in (16, 256):
                rel = (K + 2) * 2.0 ** -22
                cases = []
                # an accumulator just inside the room, positive products pushing toward the maximum
                cmax = F32MAX * (1 - 2 * rel) / (1 + 2.0 ** -30)
                A = (torch.rand(M, K, device=DEV) + 1); B = (torch.rand(K, N, device=DEV) + 1)
                if f == "bf16":
                    A = A * 2.0 ** 50; B = B * 2.0 ** 50
                C = torch.full((M, N), cmax, device=DEV) * torch.where(torch.rand(M, N, device=DEV) < 0.5, 1.0, -1.0)
                C = torch.where(C > 0, C - 2.0 ** 104 * 64 * K, C + 2.0 ** 104 * 64 * K)
                cases.append(("accumulator near the float32 maximum", A, B, C))
                if f == "bf16":
                    # single products near 2^127 (field 126), the rest zero or tiny
                    A = torch.zeros(M, K, device=DEV); B = torch.zeros(K, N, device=DEV)
                    A[:, 0] = 1.5 * 2.0 ** 63 * sgn(M); B[0, :] = (1 + torch.randint(0, 64, (N,), device=DEV) / 128.0) * 2.0 ** 63
                    A[:, 1:] = torch.randn(M, K - 1, device=DEV) * 2.0 ** 30; B[1:, :] = torch.randn(K - 1, N, device=DEV) * 2.0 ** 30
                    cases.append(("products near 2^127", A, B, torch.zeros(M, N, device=DEV)))
                    # subnormal float32 results
                    A = torch.randint(-8, 9, (M, K), device=DEV).float() * 2.0 ** -70
                    B = torch.randint(-8, 9, (K, N), device=DEV).float() * 2.0 ** -72
                    C = torch.randint(-50, 51, (M, N), device=DEV).float() * 2.0 ** -149
                    cases.append(("subnormal float32 results", A, B, C))
                # both operands subnormal everywhere, accumulator subnormal float32 or zero
                fr = FMT[f][3]
                smin = 2.0 ** (FMT[f][2] - fr)
                A = torch.randint(-2 ** fr + 1, 2 ** fr, (M, K), device=DEV).float() * smin
                B = torch.randint(-2 ** fr + 1, 2 ** fr, (K, N), device=DEV).float() * smin
                C = torch.randint(-9, 10, (M, N), device=DEV).float() * 2.0 ** -149
                cases.append(("both operands subnormal", A, B, C))
                for (name, A, B, C) in cases:
                    A, B = q(A, f), q(B, f)
                    t = Tally()
                    R, path = run(A, B, C, f, nw)
                    t.add(A, B, C, R, f, name)
                    out(f"| {name} | {f} | {path} | {K} | {t.room} | {t.out} | {t.nonfin} | {t.eq}/{t.cmp} | {t.worst:.4f} |")
    out("")


def part_p(out):
    out("## Part P: input_precision\n")
    out("| format | M x N, warps | K | input_precision | PTX equal to the default's |")
    out("|---|---|---|---|---|")
    for f in ("f16", "bf16"):
        for (M, N, nw, K) in [(64, 64, 4, 64), (16, 16, 1, 64), (128, 128, 8, 32)]:
            A = q(torch.randn(M, K, device=DEV), f); B = q(torch.randn(K, N, device=DEV), f)
            C = torch.zeros(M, N, device=DEV)
            _, _, p0 = run(A, B, C, f, nw, want_ptx=True)
            for prec in ("tf32x3", "ieee", "bf16x3", "bf16x6"):
                _, _, p1 = run(A, B, C, f, nw, prec=prec, want_ptx=True)
                out(f"| {f} | {M} x {N}, {nw} | {K} | {prec} | {p1 == p0} |")
    out("")


def main():
    t0 = time.time()

    def out(s=""):
        print(s, flush=True)

    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    part_p(out)
    part_x(out)
    part_o(out)
    part_s(out)
    out(f"(run time {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
