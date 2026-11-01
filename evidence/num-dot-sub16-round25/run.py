"""TVA library evidence (library issue L126): a 16-bit `tl.dot` into float32 on sm_90, a subnormal operand included.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-sub16-round25/run.py
Output: evidence/num-dot-sub16-round25/out.txt (README.md says what it shows). `--quick` runs a smoke test
(written to quick.txt, not the evidence).

The facts checked (docs/rules/num.md, round 25): `DotF16` and `DotBF16`, for float16 (bfloat16) operands into
float32, `max_num_imprecise_acc = 0`, any `input_precision`, per output element with values a_k, b_k, acc:

    |r - (acc + sum_k a_k b_k)| <= relDot16(K) * L + absDot16(K),
    L = |acc| + sum_k lift(a_k) lift(b_k),   relDot16(K) = (K+2) 2^-22,   absDot16(K) = (K+2) 2^-126,

when L + relDot16(K) L + absDot16(K) is at most the float32 maximum (room); lift(x) = 0 for x = 0 and
max(|x|, 2^emin) otherwise, 2^emin the format's smallest normal (2^-14 for float16, 2^-126 for bfloat16). Before
round 25 the facts said the same with |a_k b_k| in place of lift(a_k) lift(b_k) (the "old bound" below), which
fails when a product with a subnormal operand sets the window (evidence/num-dot-sub16).

The bit-level model of one instruction (Part A establishes it, Parts B and D check it on both paths): the terms are
the accumulator s (float32) and the 16 exact products p_k = a_k b_k of the chunk (16 consecutive k, Part E). Each
nonzero term has an exponent field: e(a_k) + e(b_k) for a product, with e(x) = max(floor(log2|x|), emin) (a
subnormal operand reads as the smallest normal), and max(floor(log2|s|), -126) for the accumulator. With E the
largest field and u = 2^(E-25), every term is truncated toward zero to a multiple of u, the truncated terms are
summed exactly, and the sum is truncated toward zero to float32 (24 significant bits, and the grid 2^-149 below
2^-126). That is the new accumulator; the instructions run in increasing order of the chunks.

The derivation of the constants from the model (README, "Derivation"): per chunk the error is below
5.25 * 2^-23 * (|s_j| + A_j) + 5.25 * 2^-149, A_j the chunk's sum of lift(a_k) lift(b_k); `chunk ratio` below is
the largest step of the model's chain over that bound.

Parts:
  A. The window: structured single elements against the model, both paths, both formats.
  B, C. The model against the device bit for bit, and the facts, on adversarial inputs at every K that compiles,
     seven block shapes, both paths.
  D. The kernels' shapes (tutorial 06's attention, tutorials 03 and 08's tiles) and chains of
     tl.dot(p.to(fp16), v, acc) and of a K loop.
  E. The order of the chunks.
  F. L126's inputs (evidence/num-dot-sub16, probe.py) against the old and the new bound.
  G. Other facts: input_precision on these inputs; the fp8 chunk facts on mma.sync (fp8 upcast to float16) with a
     subnormal float8e5 operand; tl.dot_scaled's emulation (a bfloat16 dot of normal values) on window inputs.
"""
import math
import random
import sys
import time
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

torch.manual_seed(20261101)
random.seed(20261101)
DEV = "cuda"
F64 = torch.float64
BIG = 1 << 20
F32MAX = 3.4028234663852886e38
QUICK = "--quick" in sys.argv
ZSIGN = [0, 0]      # zero results equal to the model's, and those of them whose sign differs

# torch dtype, Triton dtype, fraction bits, emin, max value, largest significand, largest exponent
FMT = {
    "f16": (torch.float16, tl.float16, 10, -14, 65504.0, 2 - 2.0 ** -10, 15),
    "bf16": (torch.bfloat16, tl.bfloat16, 7, -126, 3.3895313892515355e38, 2 - 2.0 ** -7, 127),
}
NAME = {"f16": "float16", "bf16": "bfloat16"}
FACT = {"f16": "DotF16", "bf16": "DotBF16"}
W = 25          # the window: bits kept below the largest field


def rel16(K):
    return (K + 2) * 2.0 ** -22


def abs16(K):
    return (K + 2) * 2.0 ** -126


# ----------------------------------------------------------------------------------------------- kernels

@triton.jit
def dotk(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         USE_ACC: tl.constexpr, A_REG: tl.constexpr, B_T: tl.constexpr, ADT: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    if A_REG:
        # `a` arrives as float32 values the format holds exactly and is converted in registers, as
        # `p.to(tl.float16)` in tutorial 06's _attn_fwd
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :]).to(ADT)
    else:
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    if B_T:
        # `b` stored as [N, K] and transposed, as `k.T` in _attn_fwd
        b = tl.trans(tl.load(b_ptr + rn[:, None] * K + rk[None, :]))
    else:
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    if USE_ACC:
        c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
        o = tl.dot(a, b, c, input_precision=PREC)
    else:
        o = tl.dot(a, b, input_precision=PREC)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], o)


@triton.jit
def chain(a_ptr, b_ptr, c_ptr, al_ptr, accs_ptr, o_ptr, T, M: tl.constexpr, N: tl.constexpr,
          KC: tl.constexpr, ADT: tl.constexpr, A_REG: tl.constexpr, SCALE: tl.constexpr, STORE: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, KC)
    acc = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    for t in range(T):
        if SCALE:
            acc = acc * tl.load(al_ptr + t)
        if A_REG:
            p = tl.load(a_ptr + t * M * KC + rm[:, None] * KC + rk[None, :]).to(ADT)
        else:
            p = tl.load(a_ptr + t * M * KC + rm[:, None] * KC + rk[None, :])
        v = tl.load(b_ptr + t * KC * N + rk[:, None] * N + rn[None, :])
        acc = tl.dot(p, v, acc)
        if STORE:
            tl.store(accs_ptr + t * M * N + rm[:, None] * N + rn[None, :], acc)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc)


def path_of(h):
    ptx = h.asm["ptx"]
    return "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "?")


def instr_of(h):
    import re
    ptx = h.asm["ptx"]
    m = re.findall(r"(wgmma\.mma_async[\w.]*|mma\.sync[\w.]*)", ptx)
    return sorted(set(m))


def run_dot(a, b, c, f, nw, use_acc=True, a_reg=False, b_t=False, prec="tf32", handle=False):
    """a [M,K], b [K,N] float32 tensors of values the format holds exactly; c [M,N] float32."""
    M, K = a.shape
    N = b.shape[1]
    dt = FMT[f][0]
    a16 = a.to(dt)
    b16 = b.to(dt)
    assert torch.equal(a16.float(), a) and torch.equal(b16.float(), b), "operand not representable"
    ain = a.contiguous() if a_reg else a16.contiguous()
    bin_ = b16.t().contiguous() if b_t else b16.contiguous()
    o = torch.empty(M, N, device=DEV)
    h = dotk[(1,)](ain, bin_, c.contiguous(), o, M, N, K, use_acc, a_reg, b_t, FMT[f][1], prec, num_warps=nw)
    if handle:
        return o, path_of(h), h
    return o, path_of(h)


# ------------------------------------------------------------------------------------------ the model

def rawexp(x, emin):
    e = torch.frexp(x.abs())[1].to(torch.int64) - 1
    return torch.clamp(e, min=emin)


def trunc_f32(S):
    """Truncate float64 values toward zero to float32 (24 significant bits; the grid 2^-149 below 2^-126)."""
    er = torch.frexp(S.abs())[1].to(torch.int64) - 1
    g = torch.ldexp(torch.ones_like(S), torch.clamp(er - 23, min=-149))
    return torch.where(S == 0, S, torch.trunc(S / g) * g)


def model(A, B, C, f, use_acc=True, partials=False):
    """The model on float64 tensors (exact: each truncated term is an integer multiple of u below 2^27 u, and 17 of
    them sum below 2^32 u). Returns the result and, optionally, the partial values and each chunk's lifted sum."""
    M, K = A.shape
    emin = FMT[f][3]
    ea = rawexp(A, emin)
    eb = rawexp(B, emin)
    s = C.clone() if use_acc else torch.zeros_like(C)
    ps = [s]
    for k0 in range(0, K, 16):
        k1 = min(K, k0 + 16)
        P = A[:, None, k0:k1] * B[k0:k1, :].T[None, :, :]
        EP = ea[:, None, k0:k1] + eb[k0:k1, :].T[None, :, :]
        EP = torch.where(P == 0, torch.full_like(EP, -BIG), EP)
        es = torch.clamp(torch.frexp(s.abs())[1].to(torch.int64) - 1, min=-126)
        es = torch.where(s == 0, torch.full_like(es, -BIG), es)
        E = torch.maximum(EP.amax(-1), es)
        u = torch.ldexp(torch.ones_like(s), torch.clamp(E, min=-1000) - W)
        S1 = (torch.trunc(s / u) + torch.trunc(P / u[..., None]).sum(-1)) * u
        S1 = torch.where(E <= -BIG // 2, torch.zeros_like(S1), S1)
        s = trunc_f32(S1)
        ps.append(s)
    return (s, ps) if partials else s


def model_py(acc, prods, emin):
    """One chunk of the model in exact rationals (Part A's table): prods = [(a, b)] python floats."""
    def ef(x):
        return max(math.frexp(abs(x))[1] - 1, emin)
    terms = [(Fr(acc), max(math.frexp(abs(acc))[1] - 1, -126))] if acc != 0 else []
    terms += [(Fr(x) * Fr(y), ef(x) + ef(y)) for x, y in prods if x != 0 and y != 0]
    if not terms:
        return 0.0
    E = max(e for _, e in terms)
    u = Fr(2) ** (E - W)
    S1 = sum((t / u).__trunc__() for t, _ in terms) * u
    if S1 == 0:
        return 0.0
    er = math.floor(math.log2(abs(S1)))
    while Fr(2) ** er > abs(S1):
        er -= 1
    while Fr(2) ** (er + 1) <= abs(S1):
        er += 1
    g = Fr(2) ** max(er - 23, -149)
    return float((S1 / g).__trunc__() * g)


# --------------------------------------------------------------------------------- the bound (Part C)

def lift(X, f):
    return torch.where(X != 0, torch.clamp(X.abs(), min=2.0 ** FMT[f][3]), torch.zeros_like(X))


def lsb(X, f):
    e = rawexp(X, FMT[f][3])
    return torch.where(X != 0, e - FMT[f][2], torch.full_like(e, BIG))


def bound_check(A, B, C, R, f, use_acc=True, ps=None):
    """Per element: the error against the new bound (relDot16 L + absDot16) and the old one (relDot16 mag +
    absDot16), with room; float64 sums with the reference's rounding error added where float64 does not hold every
    partial sum exactly (`exact`); and, given the model's partial values `ps`, each chunk's step against the
    derivation's per-chunk bound 5.25 2^-23 (|s_j| + A_j) + 5.25 2^-149."""
    M, K = A.shape
    C0 = C if use_acc else torch.zeros_like(C)
    La, Lb = lift(A, f), lift(B, f)
    qa, qb = lsb(A, f), lsb(B, f)
    ec = torch.clamp(torch.frexp(C0.abs())[1].to(torch.int64) - 1, min=-126) - 23
    q = torch.where(C0 != 0, ec, torch.full_like(ec, BIG))
    tot = C0.abs().clone()
    L = C0.abs().clone()
    S = C0.clone()
    step = torch.zeros_like(C0)
    for j, k0 in enumerate(range(0, K, 16)):
        k1 = min(K, k0 + 16)
        P = A[:, None, k0:k1] * B[k0:k1, :].T[None, :, :]
        Pl = La[:, None, k0:k1] * Lb[k0:k1, :].T[None, :, :]
        cj = P.sum(-1)
        Aj = Pl.sum(-1)
        if ps is not None:
            s0, s1 = ps[j], ps[j + 1]
            ej = (s1 - (s0 + cj)).abs()
            bj = 5.25 * 2.0 ** -23 * (s0.abs() + Aj) + 5.25 * 2.0 ** -149
            step = torch.maximum(step, ej / bj)
        S = S + cj
        tot = tot + P.abs().sum(-1)
        L = L + Aj
        Q = qa[:, None, k0:k1] + qb[k0:k1, :].T[None, :, :]
        q = torch.minimum(q, torch.where(P != 0, Q, torch.full_like(Q, BIG)).amin(-1))
    lim = torch.ldexp(torch.ones_like(tot), torch.clamp(q, max=900) + 53)
    exact = (tot * (1 + 2.0 ** -40) < lim) | (q >= BIG)
    slack = torch.where(exact, torch.zeros_like(tot), (K + 2) * 2.0 ** -52 * tot)
    err = (R - S).abs() + slack
    rel, ab = rel16(K), abs16(K)
    Lc = L * (1 - (K + 2) * 2.0 ** -52)          # a lower bound on the exact L
    new_b = rel * Lc + ab
    old_b = rel * tot * (1 - (K + 2) * 2.0 ** -52) + ab
    room = L * (1 + 2.0 ** -40) + rel * L * (1 + 2.0 ** -40) + ab <= F32MAX
    return dict(new=err / new_b, old=err / old_b, room=room, fin=torch.isfinite(R), exact=exact, S=S, L=L,
                step=step)


def exact_sum(xs):
    """The exact sum of python floats (dyadic rationals)."""
    num, den = 0, 1
    for x in xs:
        n, d = float(x).as_integer_ratio()
        if d > den:
            num *= d // den
            den = d
        num += n * (den // d)
    return Fr(num, den)


def exact_ratio(arow, bcol, c, r, f):
    """The new and old ratios of one element in exact rationals (python floats in)."""
    K = len(arow)
    mn = 2.0 ** FMT[f][3]
    prods = [x * y for x, y in zip(arow, bcol)]          # exact in float64 (at most 22 significant bits)
    assert all(Fr(x) * Fr(y) == Fr(p) for x, y, p in zip(arow, bcol, prods))
    S = exact_sum([c] + prods)
    mag = exact_sum([abs(c)] + [abs(p) for p in prods])
    lifts = [(0.0 if x == 0 else max(abs(x), mn)) * (0.0 if y == 0 else max(abs(y), mn)) for x, y in zip(arow, bcol)]
    L = exact_sum([abs(c)] + lifts)
    rel = Fr(K + 2, 2 ** 22)
    ab = Fr(K + 2) / Fr(2) ** 126
    err = abs(Fr(r) - S)
    return err / (rel * L + ab), err / (rel * mag + ab), S


# ------------------------------------------------------------------------------------------- inputs

def q16(x, f):
    """Round float32 values to the format (saturating), back to float32."""
    mx = FMT[f][4]
    return x.clamp(-mx, mx).to(FMT[f][0]).float()


def sgn(shape):
    return torch.where(torch.rand(shape, device=DEV) < 0.5, -1.0, 1.0)


def gen(kind, M, N, K, f):
    dt, _, frac, emin, fmax, sig, emax = FMT[f]
    mn = 2.0 ** emin
    smin = 2.0 ** (emin - frac)
    if kind == "normal":
        a = torch.randn(M, K, device=DEV); b = torch.randn(K, N, device=DEV); c = torch.randn(M, N, device=DEV)
    elif kind == "softmax":
        # p = exp2(score - row max) as tutorial 06 makes it, then p.to(format): many subnormal and zero values
        sc = torch.randn(M, K, device=DEV) * torch.tensor([1.0, 4.0, 16.0, 64.0], device=DEV)[torch.randint(0, 4, (M, 1), device=DEV)]
        a = torch.exp2(sc - sc.amax(1, keepdim=True))
        b = torch.randn(K, N, device=DEV)
        c = torch.randn(M, N, device=DEV) * torch.randint(0, 2, (M, N), device=DEV)
    elif kind == "spread":
        a = torch.randn(M, K, device=DEV) * torch.exp2(torch.randint(-6, 6, (M, K), device=DEV).float())
        b = torch.randn(K, N, device=DEV) * torch.exp2(torch.randint(-6, 6, (K, N), device=DEV).float())
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-10, 10, (M, N), device=DEV).float())
    elif kind == "accbig":
        a = torch.rand(M, K, device=DEV) + 1; b = torch.rand(K, N, device=DEV) + 1
        c = torch.exp2(torch.randint(4, 20, (M, N), device=DEV).float()) * (2 - 2.0 ** -23) * sgn((M, N))
    elif kind == "allones":
        # same-sign products with all fraction bits set, against an accumulator with all bits set
        a = torch.full((M, K), sig, device=DEV) * torch.exp2(-torch.randint(0, 4, (M, K), device=DEV).float())
        b = torch.full((K, N), sig, device=DEV)
        c = torch.exp2(torch.randint(0, 9, (M, N), device=DEV).float()) * (2 - 2.0 ** -23) * sgn((M, N))
    elif kind == "onebig":
        # one large term per chunk among small ones with all fraction bits set
        t = torch.randint(1, 13, (M, K), device=DEV).float()
        a = sig * torch.exp2(-t)
        pos = torch.randint(0, 16, (M, (K + 15) // 16), device=DEV)
        for j in range(pos.shape[1]):
            a[torch.arange(M, device=DEV), (j * 16 + pos[:, j]).clamp(max=K - 1)] = sig * 4
        b = torch.full((K, N), sig, device=DEV) * torch.where(torch.rand(K, N, device=DEV) < 0.1, -1.0, 1.0)
        c = torch.zeros(M, N, device=DEV)
    elif kind == "cancel":
        # pairwise cancellation: products k and k+1 cancel exactly, or nearly (b scaled by 1.25)
        a = q16(torch.randn(M, K, device=DEV), f); b = q16(torch.randn(K, N, device=DEV), f)
        a[:, 1::2] = -a[:, 0::2]
        near = torch.rand(1).item() < 0.5
        b[1::2, :] = q16(b[0::2, :] * (1.25 if near else 1.0), f)
        c = torch.randn(M, N, device=DEV) * 2.0 ** -20
    elif kind == "tiny":
        # multiples of the smallest subnormals; subnormal float32 accumulators
        a = torch.randint(-3, 4, (M, K), device=DEV).float() * smin
        b = torch.randint(-3, 4, (K, N), device=DEV).float() * (smin if f == "f16" else 2.0 ** -10)
        c = torch.randint(-3, 4, (M, N), device=DEV).float() * 2.0 ** -149 * torch.exp2(torch.randint(0, 20, (M, N), device=DEV).float())
    elif kind == "huge":
        if f == "f16":
            a = (torch.rand(M, K, device=DEV) + 0.5) * fmax / 2 * sgn((M, K))
            b = (torch.rand(K, N, device=DEV) + 0.5) * fmax / 2
        else:
            e = 62 - math.ceil(math.log2(K) / 2)
            a = (torch.rand(M, K, device=DEV) + 0.5) * 2.0 ** e * sgn((M, K))
            b = (torch.rand(K, N, device=DEV) + 0.5) * 2.0 ** e
        c = torch.zeros(M, N, device=DEV)
    elif kind == "sub":
        # many subnormal operands, on both sides
        a = torch.randn(M, K, device=DEV) * mn
        b = torch.randn(K, N, device=DEV) * torch.where(torch.rand(K, N, device=DEV) < 0.5, 4.0, mn)
        c = torch.randn(M, N, device=DEV) * mn * torch.randint(0, 2, (M, N), device=DEV)
    elif kind in ("subwin", "winworst"):
        # per chunk: the largest field E on one product (`subwin`: a subnormal operand, the smallest or a random
        # one, times 2^eb, so the product is 2^-t of 2^E; `winworst`: a normal power of two times 2^eb), on the a
        # side or the b side; the other 15 products just below the window's unit u = 2^(E-25) with all fraction
        # bits set (lost whole), every product of one sign per row; the accumulator 0, just below the first
        # chunk's u (lost), or near the big product, of either sign
        a = torch.zeros(M, K, device=DEV); b = torch.zeros(K, N, device=DEV)
        if kind == "subwin":
            lo, hi = (12, emax) if f == "f16" else (100, emax)
        else:
            lo, hi = (10, emax) if f == "f16" else (40, 100)
        E0 = None
        bigs = None
        for j in range((K + 15) // 16):
            k0 = j * 16
            eb = random.randint(lo, hi)
            if kind == "subwin":
                sv = smin * (1 if random.random() < 0.5 else random.randint(1, 2 ** frac - 1))
                E = emin + eb
            else:
                sv = 1.0
                E = eb
            u = 2.0 ** (E - W)
            xa = max(emin, (E - 26) // 2) if f == "bf16" else emin
            xb = E - 26 - xa
            assert emin <= xb <= emax, (f, E, xa, xb)
            side = random.random() < 0.5
            pos = k0 + random.randrange(min(16, K - k0))
            for k in range(k0, min(K, k0 + 16)):
                if k == pos:
                    if side:
                        a[:, k] = sv; b[k, :] = 2.0 ** eb
                    else:
                        a[:, k] = 2.0 ** eb; b[k, :] = sv
                else:
                    a[:, k] = sig * 2.0 ** xa
                    b[k, :] = 2.0 ** xb
            if j == 0:
                E0, bigs = E, sv * 2.0 ** eb
        sgnrow = sgn((M, 1))
        a = a * sgnrow
        u0 = 2.0 ** (E0 - W)
        mode = torch.randint(0, 4, (M, N), device=DEV)
        c = torch.where(mode == 0, torch.zeros(M, N, device=DEV),
                        torch.where(mode == 1, sgnrow * u0 * (1 - 2.0 ** -23),
                                    torch.where(mode == 2, -sgnrow * bigs * (1 - 2.0 ** -23),
                                                sgnrow * bigs * (1 - 2.0 ** -23))))
    elif kind == "subrand":
        # randomized: in each chunk a few products with a subnormal operand and a random large partner, the rest
        # random products near the lifted window's unit (some lose bits, some not), accumulators of any size
        a = torch.randn(M, K, device=DEV) * torch.exp2(torch.randint(-14, 0, (M, K), device=DEV).float())
        if f == "bf16":
            a = a * 2.0 ** -100
        b = torch.randn(K, N, device=DEV) * torch.exp2(torch.randint(-14, 0, (K, N), device=DEV).float())
        lo = 8 if f == "f16" else 90
        msk = torch.rand(M, K, device=DEV) < 0.08
        a = torch.where(msk, torch.randint(1, 2 ** frac, (M, K), device=DEV).float() * smin, a)
        bigb = torch.exp2(torch.randint(lo, emax + 1, (K, N), device=DEV).float()) * sgn((K, N))
        msk2 = torch.rand(K, N, device=DEV) < 0.15
        b = torch.where(msk2, bigb, b)
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-40, 0, (M, N), device=DEV).float()) * torch.randint(0, 2, (M, N), device=DEV)
    elif kind == "trunc24":
        # sums just above a power of two with bits just below the result's 24-bit unit
        a = torch.full((M, K), 1.0, device=DEV) * torch.where(torch.rand(M, K, device=DEV) < 0.03, -1.0, 1.0)
        b = torch.full((K, N), 1.0, device=DEV)
        c = (1 + torch.randint(1, 2 ** 22, (M, N), device=DEV).float() * 2.0 ** -23) * sgn((M, N))
    elif kind == "bits":
        # every finite value of the format, uniformly by bit pattern; accumulators over a wide range
        def rbits(shape):
            x = torch.randint(-32768, 32768, shape, device=DEV, dtype=torch.int16).view(dt).float()
            return torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        a = rbits((M, K)); b = rbits((K, N))
        if f == "bf16":
            a = torch.where(a.abs() > 2.0 ** 50, a * 2.0 ** -80, a)
            b = torch.where(b.abs() > 2.0 ** 50, b * 2.0 ** -80, b)
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-30, 30, (M, N), device=DEV).float())
    else:
        raise ValueError(kind)
    return q16(a, f).contiguous(), q16(b, f).contiguous(), c.float().contiguous()


KINDS = ["normal", "softmax", "spread", "accbig", "allones", "onebig", "cancel", "tiny", "huge", "sub", "subwin",
         "winworst", "subrand", "trunc24", "bits"]


# ------------------------------------------------------------------------------------------------ parts

class Agg:
    def __init__(self):
        self.n = 0; self.room = 0; self.viol = 0; self.nonfin = 0; self.inexact = 0
        self.new = 0.0; self.old = 0.0; self.oldviol = 0; self.step = 0.0
        self.worst = None; self.sim_n = 0; self.sim_eq = 0; self.oldkinds = {}

    def add(self, A, B, C, R, f, use_acc, cfg, sim=None, ps=None):
        bc = bound_check(A, B, C, R, f, use_acc, ps)
        room, fin, exact = bc["room"], bc["fin"], bc["exact"]
        ok = room & fin
        self.n += R.numel(); self.room += int(room.sum())
        self.nonfin += int((room & ~fin).sum())
        self.inexact += int((ok & ~exact).sum())
        new = torch.where(ok, bc["new"], torch.zeros_like(bc["new"]))
        old = torch.where(ok, bc["old"], torch.zeros_like(bc["old"]))
        # the largest of the run, and any element near the bound, in exact rationals
        idx = (new > 0.9).nonzero().tolist()
        top = int(torch.argmax(new).item())
        i0, n0 = divmod(top, R.shape[1])
        if [i0, n0] not in idx:
            idx.append([i0, n0])
        exS = {}
        for i, n in idx:
            cv = float(C[i, n]) if use_acc else 0.0
            rq, oq, S = exact_ratio(A[i].tolist(), B[:, n].tolist(), cv, float(R[i, n]), f)
            new[i, n] = float(rq)
            exS[(i, n)] = S
            if rq > 1:
                self.viol += 1
        rmax = float(new.max().item())
        if rmax >= self.new:
            t = int(torch.argmax(new).item())
            i, n = divmod(t, R.shape[1])
            self.new = rmax
            S = exS.get((i, n), bc["S"][i, n].item())
            self.worst = (cfg, i, n, float(C[i, n]) if use_acc else 0.0, float(R[i, n]), float(S), float(bc["L"][i, n]))
        self.old = max(self.old, float(old.max())); nold = int((old > 1).sum()); self.oldviol += nold
        if nold and len(cfg) > 8:
            self.oldkinds[cfg[8]] = self.oldkinds.get(cfg[8], 0) + nold
        if ps is not None:
            self.step = max(self.step, float(torch.where(room, bc["step"], torch.zeros_like(R)).max()))
        if sim is not None:
            self.sim_n += R.numel(); self.sim_eq += int((sim == R).sum())
            # zero results: the sign of the device's zero against the model's (the facts speak of values only)
            z = (sim == R) & (R == 0)
            ZSIGN[0] += int(z.sum()); ZSIGN[1] += int((z & (torch.signbit(sim) != torch.signbit(R))).sum())


def part_a(out):
    out("## Part A: the window (one output element, one chunk)\n")
    out("Each row: one element of `tl.dot(a, b, acc)` with K = 16, on `wgmma` (64 x 64 block, 4 warps) and `mma.sync`"
        " (16 x 16, 1 warp); the products and the accumulator as listed, the rest zero. `model`: the bit-level model"
        " (module docstring). E is the largest field, u = 2^(E-25) the window's unit.\n")
    out("| operands | path | accumulator | products | exact | device | model | what it shows |")
    out("|---|---|---|---|---|---|---|---|")
    cases = []

    def add(f, acc, prs, what):
        cases.append((f, acc, prs, what))

    P = lambda t: (2.0 ** -12, 2.0 ** (12 - t))          # 2^-t from normal float16 operands (t <= 26)
    for f in ("f16", "bf16"):
        for t in (25, 26):
            add(f, 0.0, [(1.0, 1.0), (-1.0, 1.0), P(t)], f"1 - 1 + 2^-{t}: E = 0, u = 2^-25; one window for the chunk")
        for t in (25, 26):
            add(f, 0.0, [(1.5, 1.5), (-1.5, 1.5), P(t)], f"2.25 - 2.25 + 2^-{t}: the field of 1.5 * 1.5 is 0, not 1")
        add(f, 0.0, [(FMT[f][5], FMT[f][5]), (-FMT[f][5], FMT[f][5]), P(25)], "the largest significands' product: field 0")
        add(f, 0.0, [(1.0, 1.0), (-1.0, 1.0), (1.5 * 2.0 ** -12, 2.0 ** -13)], "1.5 u truncated to u")
        add(f, 0.0, [(1.0, 1.0), (-1.0, 1.0), (-1.5 * 2.0 ** -12, 2.0 ** -13)], "-1.5 u truncated toward zero, to -u")
        add(f, 0.0, [(1.0, 1.0), (-1.0, 1.0)] + [P(26)] * 14, "14 terms u/2: each dropped, no partial sums")
        add(f, 1.0, [(-1.0, 1.0), P(25)], "the accumulator 1 (field 0) is in the window: 2^-25 kept")
        add(f, 1.0, [(-1.0, 1.0), P(26)], "and 2^-26 dropped")
        add(f, 3.0, [(-3.0, 1.0), P(24)], "accumulator 3 (field 1): u = 2^-24, kept")
        add(f, 3.0, [(-3.0, 1.0), P(25)], "2^-25 < u dropped")
        add(f, 1.0 + 2.0 ** -23, [], "the accumulator alone passes unchanged")
        add(f, 0.0, [(1.0, 1.0), P(24), P(25)], "1 + 3 2^-25: kept in the window, truncated to float32: 1")
        add(f, 0.0, [(-1.0, 1.0), (-2.0 ** -12, 2.0 ** -12), (-2.0 ** -12, 2.0 ** -13)], "the same, negative: toward zero")
        add(f, 0.0, [(1.0, 1.0), (1.0, 1.0), P(23), P(24)], "2 + 3 2^-24: truncated to 2")
        add(f, 0.0, [(1.0, 1.0), P(23), P(24), P(25)], "1 + 7 2^-25: truncated to 1 + 2^-23")
    # subnormal operands
    for t in (25, 26):
        add("f16", 0.0, [(2.0 ** -23, 2.0 ** 14), (-2.0 ** -23, 2.0 ** 14), P(t)], f"float16 subnormal 2^-23 reads as 2^-14: E = 0; 2^-{t}")
        add("f16", 0.0, [(2.0 ** 14, 2.0 ** -23), (-2.0 ** 14, 2.0 ** -23), P(t)], f"the same with the subnormal in b; 2^-{t}")
    add("f16", 0.0, [(2.0 ** -23, 2.0 ** 14), (-2.0 ** -23, 2.0 ** 14), (2.0 ** -14, 2.0 ** -14)],
        "float16 subnormal 2^-23 times 2^14: E = 0, u = 2^-25, so 2^-28 is dropped")
    add("f16", 0.0, [(2.0 ** -9, 1.0), (-2.0 ** -9, 1.0), (2.0 ** -14, 2.0 ** -14)],
        "the same product 2^-9 from normals: E = -9, u = 2^-34, 2^-28 kept")
    for t in (27, 28):
        add("bf16", 0.0, [(2.0 ** -133, 2.0 ** 124), (-2.0 ** -133, 2.0 ** 124), (2.0 ** -14, 2.0 ** (14 - t))],
            f"bfloat16 subnormal 2^-133 reads as 2^-126: E = -2; 2^-{t}")
        add("bf16", 0.0, [(2.0 ** 124, 2.0 ** -133), (-2.0 ** 124, 2.0 ** -133), (2.0 ** -14, 2.0 ** (14 - t))],
            f"the same with the subnormal in b; 2^-{t}")
    add("f16", 0.0, [(2.0 ** -23, 2.0 ** 14)] + [((2 - 2.0 ** -10) * 2.0 ** -12, 2.0 ** -12)] * 15,
        "L126's float16 case: each small product loses its bits below 2^-25")
    add("bf16", 0.0, [(2.0 ** -133, 2.0 ** 124)] + [((2 - 2.0 ** -7) * 2.0 ** -13, 2.0 ** -13)] * 15,
        "L126's bfloat16 case: below 2^-27 lost")
    # subnormal accumulators and results (bfloat16 reaches them)
    for accv in (2.0 ** -149, (2.0 ** 23 - 1) * 2.0 ** -149):
        add("bf16", accv, [], "a subnormal accumulator alone passes unchanged")
    add("bf16", 3 * 2.0 ** -149, [(2.0 ** -76, 2.0 ** -76)] * 16, "a subnormal accumulator's field is -126: u = 2^-151, the 2^-152 dropped")
    add("bf16", 0.0, [(2.0 ** -76, 2.0 ** -76)] * 16, "the same products alone: E = -152, kept, 2^-148")
    add("bf16", 0.0, [(2.0 ** -70, 2.0 ** -70)], "a subnormal result is kept (no flush)")
    add("bf16", 0.0, [(2.0 ** -70 * 1.0078125, 2.0 ** -70 * 1.0078125)], "truncated to the grid 2^-149")
    add("bf16", 0.0, [(2.0 ** -100, 2.0 ** -100)], "2^-200: below float32, 0")
    add("bf16", 0.0, [(2.0 ** -63, 2.0 ** -63), (-2.0 ** -63, 2.0 ** -63)] + [(2.0 ** -75, 2.0 ** -76)] * 14,
        "E = -126: 14 terms 2^-151 kept, the sum 3.5 2^-149 truncated")
    add("bf16", -1.5 * 2.0 ** -126, [(2.0 ** -63, 2.0 ** -63)], "-1.5 2^-126 + 2^-126 = -2^-127, subnormal")
    groups = {}
    for cs in cases:
        groups.setdefault(cs[0], []).append(cs)
    allok = True
    instrs = set()
    for f, cc in groups.items():
        for (M, nw) in ((64, 4), (16, 1)):
            for s in range(0, len(cc), M):
                part = cc[s:s + M]
                a = torch.zeros(M, 16, device=DEV); b = torch.zeros(16, M, device=DEV); c = torch.zeros(M, M, device=DEV)
                for i, (_, acc, prs, _) in enumerate(part):
                    c[i, i] = acc
                    for k, (x, y) in enumerate(prs):
                        a[i, k] = x; b[k, i] = y
                o, path, h = run_dot(a, b, c, f, nw, handle=True)
                instrs.update(instr_of(h))
                for i, (_, acc, prs, what) in enumerate(part):
                    accv = c[i, i].item()
                    ex = Fr(accv) + sum(Fr(x) * Fr(y) for x, y in prs)
                    mv = model_py(accv, prs, FMT[f][3])
                    dv = o[i, i].item()
                    allok &= (mv == dv)
                    prs_s = ", ".join(f"{x!r}*{y!r}" for x, y in prs[:3]) + (f", ... ({len(prs)})" if len(prs) > 3 else "") if prs else "none"
                    out(f"| {NAME[f]} | {path} | {accv!r} | {prs_s} | {float(ex)!r} | {dv!r} | {mv!r} | {what} |")
    out(f"\nEvery device result equal to the model: {allok}\n")
    out("Instructions in the PTX: " + ", ".join(f"`{x}`" for x in sorted(instrs)) + "\n")


def ks_for(M, N):
    if QUICK:
        return [16, 64]
    return [16, 32, 64, 128, 256, 512, 1024, 2048, 4096] if M * N <= 32 * 32 else [16, 32, 64, 128, 256, 512, 1024, 2048]


SHAPES = [(16, 16, 1), (32, 32, 2), (16, 64, 4), (64, 64, 4), (128, 64, 4), (64, 128, 8), (128, 128, 8)]
if QUICK:
    SHAPES = [(16, 16, 1), (64, 64, 4)]
FORMATS = ["f16", "bf16"]


SUBKINDS = {"softmax", "tiny", "sub", "subwin", "winworst", "subrand", "bits"}


def reps_of(kind):
    """Draws per kind and configuration: 16 for the kinds with subnormal operands or the window's worst case, 6 for
    the others (1 each in a smoke test)."""
    if QUICK:
        return 1
    return 16 if kind in SUBKINDS else 6


def run_configs(out, configs, title, totals, worst, oldkinds, kinds=KINDS):
    rejected = set()
    for f in FORMATS:
        for (M, N, nw, K, a_reg, b_t, use_acc, label) in configs:
            if (f, M, N, nw, a_reg, b_t, use_acc) in rejected:
                continue        # a larger K than one rejected on the host
            variant = label or (("a in registers" if a_reg else "") + ("b transposed" if b_t else "")
                                + ("" if use_acc else ", no acc")).strip(", ") or "plain"
            agg = Agg()
            path = None
            try:
                for kind in kinds:
                    if kind == "bits" and K > 1024 and M * N > 64 * 64:
                        continue
                    for rep in range(reps_of(kind)):
                        A, B, C = gen(kind, M, N, K, f)
                        R, path = run_dot(A, B, C, f, nw, use_acc, a_reg, b_t)
                        Ad, Bd, Cd, Rd = A.double(), B.double(), C.double(), R.double()
                        sim, ps = model(Ad, Bd, Cd, f, use_acc, partials=True)
                        agg.add(Ad, Bd, Cd, Rd, f, use_acc, (NAME[f], M, N, nw, variant, path, K, title, kind), sim, ps)
            except triton.runtime.errors.OutOfResources:
                out(f"| {NAME[f]} | {M} x {N}, {nw} | {variant} | rejected | {K} | (out of resources) | | | | | | |")
                rejected.add((f, M, N, nw, a_reg, b_t, use_acc))
                continue
            out(f"| {NAME[f]} | {M} x {N}, {nw} | {variant} | {path} | {K} | {agg.room} | {agg.viol + agg.nonfin} | {agg.sim_eq}/{agg.sim_n} | {agg.new:.4f} | {agg.step:.4f} | {agg.oldviol} | {agg.old:.4f} |")
            key = (f, path)
            t = totals.setdefault(key, dict(room=0, viol=0, sim_n=0, sim_eq=0, new=0.0, step=0.0, inexact=0, oldviol=0, old=0.0))
            t["room"] += agg.room; t["viol"] += agg.viol + agg.nonfin; t["sim_n"] += agg.sim_n; t["sim_eq"] += agg.sim_eq
            t["inexact"] += agg.inexact; t["oldviol"] += agg.oldviol; t["old"] = max(t["old"], agg.old)
            if agg.new >= t["new"]:
                t["new"] = agg.new; worst[key] = agg.worst
            t["step"] = max(t["step"], agg.step)
            for kd, nk in agg.oldkinds.items():
                oldkinds[(f, path, kd)] = oldkinds.get((f, path, kd), 0) + nk


TABLE_HEAD = ("| operands | M x N, warps | variant | path | K | elements in range | outside the new bound | model = device |"
              " new ratio | chunk ratio | outside the old bound | old ratio |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|")


def totals_out(out, totals, worst, oldkinds):
    out("\n### Totals\n")
    out("| operands | path | elements in range | outside the new bound | model = device | largest new ratio | largest chunk ratio | elements with float64 sums (rounding added) | outside the old bound | largest old ratio |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for (f, path), t in sorted(totals.items()):
        out(f"| {NAME[f]} | {path} | {t['room']} | {t['viol']} | {t['sim_eq']}/{t['sim_n']} | {t['new']:.4f} | {t['step']:.4f} | {t['inexact']} | {t['oldviol']} | {t['old']:.4f} |")
    out("\n### Elements outside the old bound, by kind of input\n")
    out("| operands | path | kind | elements |")
    out("|---|---|---|---|")
    for (f, path, kd), nk in sorted(oldkinds.items()):
        out(f"| {NAME[f]} | {path} | `{kd}` | {nk} |")
    if not oldkinds:
        out("| (none) | | | |")
    out("\n### The inputs with the largest new ratio\n")
    for key, w in sorted(worst.items()):
        if w is None:
            continue
        cfg, i, n, c, r, S, L = w
        out(f"- {cfg[0]}, {cfg[5]}: {cfg[1]} x {cfg[2]}, {cfg[3]} warps, {cfg[4]}, K = {cfg[6]}, kind `{cfg[8]}`,"
            f" element ({i}, {n}): accumulator {c!r}, exact result {S!r}, device {r!r}, L = {L!r}")


def part_bc(out):
    out("\n## Parts B and C: the model and the facts, every K that compiles\n")
    out("Each row: one format, block shape and K; 15 kinds of inputs (README), 16 draws of each kind with subnormal"
        " operands or the window's worst case (`softmax`, `tiny`, `sub`, `subwin`, `winworst`, `subrand`, `bits`)"
        " and 6 of the others. `model = device`:"
        " elements whose device result equals the model's (as values). `new ratio`: the largest error over the new"
        " bound relDot16(K) L + absDot16(K) (1 = on the bound). `chunk ratio`: the largest step of the model's chain"
        " over the derivation's per-chunk bound 5.25 2^-23 (|s_j| + A_j) + 5.25 2^-149. `old ratio`: the error over"
        " the bound before round 25, relDot16(K) (|acc| + sum |a b|) + absDot16(K).\n")
    out(TABLE_HEAD)
    totals, worst, oldkinds = {}, {}, {}
    configs = []
    for (M, N, nw) in SHAPES:
        for K in ks_for(M, N):
            configs.append((M, N, nw, K, False, False, True, None))
    run_configs(out, configs, "B", totals, worst, oldkinds)
    totals_out(out, totals, worst, oldkinds)


def part_d(out):
    out("\n## Part D: the kernels' shapes\n")
    out("### D1: single dots at the kernels' shapes\n")
    out("Tutorial 06's `_attn_fwd` and `_attn_bwd` (`[128, D] x [D, 64]` with `b` transposed, with and without an"
        " accumulator; `[128, 64] x [64, D]` with `a` converted to the format in registers, `p.to(...)`), `D` in"
        " {64, 128, 256}, and tutorials 03 and 08's tiles (`128 x 128 x 64`, `128 x 256 x 64`), each with 4 and 8"
        " warps; the columns as in Parts B and C.\n")
    out(TABLE_HEAD)
    totals, worst, oldkinds = {}, {}, {}
    configs = []
    for nw in ((4,) if QUICK else (4, 8)):
        for D in ((64,) if QUICK else (64, 128, 256)):
            configs.append((128, 64, nw, D, False, True, True, None))
            configs.append((128, 64, nw, D, False, True, False, None))
            configs.append((128, D, nw, 64, True, False, True, None))
        for (BM, BN, BK) in ((128, 128, 64), (128, 256, 64)):
            configs.append((BM, BN, nw, BK, False, False, True, f"tile {BM}x{BN}x{BK}"))
    run_configs(out, configs, "D1", totals, worst, oldkinds)
    totals_out(out, totals, worst, oldkinds)

    out("\n### D2: chains\n")
    out("`for t in range(T): [acc = acc * alpha_t]; acc = tl.dot(p_t, v_t, acc)`, every acc_t stored (the result"
        " without the stores is checked equal). `p.to`: p_t [M, 64] arrives as float32 and is converted in registers"
        " (tutorial 06's `p.to(fp16)`), from a softmax of random scores (`exp2` of scores minus the row maximum, scale"
        " 1, 4, 16 or 64: many subnormal and zero values), or `subwin` blocks; `K loop`: p_t loaded in the format"
        " (tutorial 03's `acc = tl.dot(a, b, acc)`), `normal` or `subwin` blocks. `per dot`: each dot from the stored"
        " accumulator before it (times alpha_t, in float32) against the facts and the model. `whole chain` (no"
        " rescale): the final result against the facts for one dot of K = 64 T from the initial accumulator, and"
        " against the model run over the concatenated operands.\n")
    out("| operands | M x N, warps | path | p, v | rescale | T | elements | per dot: model = device | per dot: new ratio | per dot: old ratio | whole chain: model = device | whole chain: new ratio | whole chain: old ratio |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    KC = 64
    tot_viol = 0
    maxr = 0.0
    shapes = [(128, 64, 4), (32, 64, 2)] if QUICK else [(128, 64, 4), (128, 128, 4), (128, 256, 4), (128, 64, 8), (128, 128, 8), (32, 64, 2)]
    for f in FORMATS:
        for (M, N, nw) in shapes:
            for pk in ["p.to softmax", "p.to subwin", "K loop normal", "K loop subwin"]:
                a_reg = pk.startswith("p.to")
                for scale in ((False,) if not a_reg else (False, True)):
                    for T in ([1, 4] if QUICK else ([1, 4, 16, 64, 256] if not scale else [16, 64])):
                        if "subwin" in pk and T > 64:
                            continue
                        P = torch.zeros(T, M, KC, device=DEV); V = torch.zeros(T, KC, N, device=DEV)
                        for t in range(T):
                            if pk == "p.to softmax":
                                p, _, _ = gen("softmax", M, N, KC, f)
                                v = q16(torch.randn(KC, N, device=DEV), f)
                            elif "subwin" in pk:
                                p, v, _ = gen("subwin", M, N, KC, f)
                            else:
                                p, v, _ = gen("normal", M, N, KC, f)
                            P[t] = p; V[t] = v
                        c0 = torch.randn(M, N, device=DEV) * (0 if "subwin" in pk else 1)
                        al = (torch.rand(T, device=DEV) * 0.75 + 0.25) if scale else torch.ones(T, device=DEV)
                        accs = torch.empty(T, M, N, device=DEV)
                        o = torch.empty(M, N, device=DEV)
                        V16 = V.to(FMT[f][0]).contiguous()
                        Pin = P.contiguous() if a_reg else P.to(FMT[f][0]).contiguous()
                        h = chain[(1,)](Pin, V16, c0, al, accs, o, T, M, N, KC, FMT[f][1], a_reg, scale, True, num_warps=nw)
                        path = path_of(h)
                        assert torch.equal(o, accs[T - 1])
                        o2 = torch.empty(M, N, device=DEV)
                        chain[(1,)](Pin, V16, c0, al, accs, o2, T, M, N, KC, FMT[f][1], a_reg, scale, False, num_warps=nw)
                        same_nostore = torch.equal(o2, o)
                        agg = Agg()
                        prev = c0
                        for t in range(T):
                            vc = prev * al[t] if scale else prev
                            Ad, Bd, Cd, Rd = P[t].double(), V[t].double(), vc.double(), accs[t].double()
                            sim = model(Ad, Bd, Cd, f)
                            agg.add(Ad, Bd, Cd, Rd, f, True, ("chain", t), sim)
                            prev = accs[t]
                        whole_new = whole_sim = whole_old = "-"
                        if not scale:
                            Aall = torch.cat([P[t] for t in range(T)], 1).double()
                            Ball = torch.cat([V[t] for t in range(T)], 0).double()
                            agg2 = Agg()
                            sim2 = model(Aall, Ball, c0.double(), f)
                            agg2.add(Aall, Ball, c0.double(), o.double(), f, True, ("whole",), sim2)
                            whole_new = f"{agg2.new:.4f}" + (" (outside!)" if agg2.viol + agg2.nonfin else "")
                            whole_sim = f"{agg2.sim_eq}/{agg2.sim_n}"
                            whole_old = f"{agg2.old:.4f}"
                            tot_viol += agg2.viol + agg2.nonfin
                            maxr = max(maxr, agg2.new)
                        tot_viol += agg.viol + agg.nonfin
                        maxr = max(maxr, agg.new)
                        out(f"| {NAME[f]} | {M} x {N}, {nw} | {path} | {pk} | {'yes' if scale else 'no'} | {T} | {M * N} | {agg.sim_eq}/{agg.sim_n} | {agg.new:.4f} | {agg.old:.4f} | {whole_sim} | {whole_new} | {whole_old} |"
                            + ("" if same_nostore else " (result differs without the stores)"))
    out(f"\nElements outside the new bound in D2: {tot_viol}; largest new ratio {maxr:.4f}")


def part_e(out):
    out("\n## Part E: the order of the chunks\n")
    out("For chunks x < y and k (all distinct, chunks of 16 consecutive k): +B at a random position of chunk x, -B in"
        " chunk y, a small product s in chunk k, with B/s = 2^28, beyond the window. s survives exactly when chunk k"
        " is accumulated after both x and y (before, it is cut against B; between, too). `agree`: triples whose"
        " outcome is that of increasing k.\n")
    out("| operands | M x N, warps | path | chunks | triples | agree with increasing k |")
    out("|---|---|---|---|---|---|")
    for f in FORMATS:
        for (M, N, nw) in [(64, 64, 4), (16, 16, 1)]:
            for nch in ((8,) if QUICK else (8, 32)):
                (Ba, Bb), (Sa, Sb) = (2.0 ** 14, 2.0 ** 14), (1.0, 1.0)
                K = nch * 16
                triples = [(x, y, k) for x in range(nch) for y in range(nch) for k in range(nch)
                           if len({x, y, k}) == 3 and x < y]
                if len(triples) > 3000:
                    triples = random.sample(triples, 3000)
                agree = 0
                path = None
                D = min(M, N)
                for s in range(0, len(triples), D):
                    tt = triples[s:s + D]
                    a = torch.zeros(M, K, device=DEV); b = torch.zeros(K, N, device=DEV); c = torch.zeros(M, N, device=DEV)
                    for i, (x, y, k) in enumerate(tt):
                        o1, o2, o3 = random.sample(range(16), 3)
                        a[i, x * 16 + o1] = Ba; b[x * 16 + o1, i] = Bb
                        a[i, y * 16 + o2] = -Ba; b[y * 16 + o2, i] = Bb
                        a[i, k * 16 + o3] = Sa; b[k * 16 + o3, i] = Sb
                    o, path = run_dot(a, b, c, f, nw)
                    for i, (x, y, k) in enumerate(tt):
                        agree += int((o[i, i].item() != 0.0) == (k > max(x, y)))
                out(f"| {NAME[f]} | {M} x {N}, {nw} | {path} | {nch} | {len(triples)} | {agree} |")


def part_f(out):
    out("\n## Part F: L126's inputs\n")
    out("evidence/num-dot-sub16's `probe.py` cases (K = 16, the large product at k = 0, 7, 15), and the same at K = 64"
        " and 256 with the chunk repeated: the error against the bound before round 25 (`old`) and the new one.\n")
    out("| case | path | K | k | exact | device | model | error / old bound | error / new bound |")
    out("|---|---|---|---|---|---|---|---|---|")
    cases = [
        ("f16", "float16, subnormal a = 2^-23, b = 2^14", 2.0 ** -23, 2.0 ** 14, (2 - 2 ** -10) * 2 ** -12, 2.0 ** -12),
        ("f16", "float16, normal a = 2^-14, b = 2^5", 2.0 ** -14, 2.0 ** 5, (2 - 2 ** -10) * 2 ** -12, 2.0 ** -12),
        ("bf16", "bfloat16, subnormal a = 2^-133, b = 2^124", 2.0 ** -133, 2.0 ** 124, (2 - 2 ** -7) * 2 ** -13, 2.0 ** -13),
        ("bf16", "bfloat16, normal a = 2^-126, b = 2^117", 2.0 ** -126, 2.0 ** 117, (2 - 2 ** -7) * 2 ** -13, 2.0 ** -13),
    ]
    worst = 0.0
    for f, label, ba, bb, sa, sb in cases:
        for (M, nw) in ((64, 4), (16, 1)):
            for K in (16, 64, 256):
                for pos in ((0, 7, 15) if K == 16 else (0,)):
                    a = torch.zeros(M, K, device=DEV); b = torch.zeros(K, 64, device=DEV); c = torch.zeros(M, 64, device=DEV)
                    for j in range(K // 16):
                        for k in range(16):
                            a[0, j * 16 + k] = ba if k == pos else sa
                            b[j * 16 + k, 0] = bb if k == pos else sb
                    o, path = run_dot(a, b, c, f, nw)
                    r = float(o[0, 0])
                    rq, oq, S = exact_ratio(a[0].tolist(), b[:, 0].tolist(), 0.0, r, f)
                    mv = float(model(a.double()[:1], b.double()[:, :1], c.double()[:1, :1], f)[0, 0])
                    worst = max(worst, float(rq))
                    out(f"| {label} | {path} | {K} | {pos} | {float(S)!r} | {r!r} | {mv!r} | {float(oq):.3f} | {float(rq):.4f} |")
    out(f"\nLargest error / new bound in Part F: {worst:.4f}")


# ---- Part G helpers: the fp8 chunk facts' interval (as evidence/num-dot-chunk computes it) and dot_scaled

F8 = {"e5": (torch.float8_e5m2, tl.float8e5, 2, -14), "e4": (torch.float8_e4m3fn, tl.float8e4nv, 3, -6)}


def chunk8_ratio(arow, bcol, c, r, da, db):
    """The fp8 chunk facts (DotE5m2Chunk, ...): the interval their per-chunk bound composed over the chunks implies,
    in exact rationals; the error over its half-width on its side (1 = on the bound)."""
    K = len(arow)
    RHO, SIG, ALP = Fr(1, 2 ** 7), Fr(1, 2 ** 12), Fr(1, 2 ** 126)
    la = [0 if x == 0 else max(Fr(abs(x)), Fr(2) ** F8[da][3]) for x in arow]
    lb = [0 if y == 0 else max(Fr(abs(y)), Fr(2) ** F8[db][3]) for y in bcol]
    S = Fr(c)
    dp = dm = Fr(0)
    for k0 in range(0, K, 32):
        ks = range(k0, min(K, k0 + 32))
        cj = sum(Fr(arow[k]) * Fr(bcol[k]) for k in ks)
        mj = max(la[k] * lb[k] for k in ks)
        S1 = S + cj
        dp = dp + RHO * max(abs(S + dp), mj) + SIG * abs(S1 + dp) + ALP
        dm = dm + RHO * max(abs(S - dm), mj) + SIG * abs(S1 - dm) + ALP
        S = S1
    err = Fr(r) - S
    return (err / dp if err >= 0 else -err / dm), S


@triton.jit
def dot8(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, c))


@triton.jit
def dsk(a_ptr, sa_ptr, b_ptr, sb_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        FM: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    rs = tl.arange(0, K // 32)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    sa = tl.load(sa_ptr + rm[:, None] * (K // 32) + rs[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    sb = tl.load(sb_ptr + rn[:, None] * (K // 32) + rs[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    o = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", c, fast_math=FM)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], o)


def part_g(out):
    out("\n## Part G: other facts\n")
    # G1: input_precision
    out("### G1: `input_precision` on these inputs\n")
    out("Every `input_precision` against the default (`tf32`), bit for bit, on the kinds with subnormal operands.\n")
    out("| operands | M x N, warps | K | input_precision | kinds | elements | equal to the default |")
    out("|---|---|---|---|---|---|---|")
    for f in FORMATS:
        for (M, N, nw) in [(64, 64, 4), (16, 16, 1)]:
            for K in (16, 64, 256):
                for prec in ("tf32x3", "ieee", "bf16x3", "bf16x6"):
                    n = eq = 0
                    for kind in ("subwin", "sub", "softmax", "subrand", "bits"):
                        A, B, C = gen(kind, M, N, K, f)
                        R0, _ = run_dot(A, B, C, f, nw)
                        R1, _ = run_dot(A, B, C, f, nw, prec=prec)
                        n += R0.numel()
                        eq += int(((R0 == R1) | (torch.isnan(R0) & torch.isnan(R1))).sum())
                    out(f"| {NAME[f]} | {M} x {N}, {nw} | {K} | {prec} | subwin, sub, softmax, subrand, bits | {n} | {eq} |")
    # G2: fp8 on mma.sync with a subnormal float8e5 operand
    out("\n### G2: the fp8 chunk facts on `mma.sync` with a subnormal `float8e5` operand\n")
    out("On `mma.sync` Triton converts fp8 operands to `float16` (a subnormal `float8e5` value is a subnormal `float16`"
        " value) and runs two `float16` instructions per chunk of 32. One product per chunk of 16 has a subnormal"
        " `float8e5` operand and a large partner, so its field is the smallest normal's; the other products lie just"
        " below that `float16` window's unit and are lost (only `float8e5` by `float8e5` has normal products that"
        " small). The error against `DotE5m2Chunk`'s implied interval, on `16 x 16` with 1 warp.\n")
    out("| operands | path | K | rows | largest error / chunk facts' bound | largest error / magnitude sum |")
    out("|---|---|---|---|---|---|")
    for (da, db) in (("e5", "e5"),):
        for K in (32, 64, 256):
            M = N = 16
            a = torch.zeros(M, K, device=DEV); b = torch.zeros(K, N, device=DEV); c = torch.zeros(M, N, device=DEV)
            big_b = 2.0 ** 15 if db == "e5" else 2.0 ** 8
            for i in range(M):
                for j in range(K // 16):
                    pos = j * 16 + random.randrange(16)
                    for k in range(j * 16, j * 16 + 16):
                        if k == pos:
                            a[i, k] = 2.0 ** -16 * random.choice([1, 2, 3]); b[k, i] = big_b
                        else:
                            # field of the big product: -14 + e(big_b); small products 1.75 * 2^(E - 26)
                            E = -14 + int(math.log2(big_b))
                            a[i, k] = 1.75 * 2.0 ** (E - 26 - (-6 if db == "e4" else -12))
                            b[k, i] = 2.0 ** (-6 if db == "e4" else -12)
            a8 = a.to(F8[da][0]); b8 = b.to(F8[db][0])
            assert torch.equal(a8.float(), a) and torch.equal(b8.float(), b)
            o = torch.empty(M, N, device=DEV)
            h = dot8[(1,)](a8, b8, c, o, M, N, K, num_warps=1)
            path = path_of(h)
            worst = 0.0; worst_rel = 0.0
            for i in range(M):
                rq, S = chunk8_ratio(a[i].tolist(), b[:, i].tolist(), 0.0, float(o[i, i]), da, db)
                mag = sum(abs(Fr(x) * Fr(y)) for x, y in zip(a[i].tolist(), b[:, i].tolist()))
                worst = max(worst, float(rq)); worst_rel = max(worst_rel, float(abs(Fr(float(o[i, i])) - S) / mag))
            out(f"| {da} x {db} | {path} | {K} | {M} | {worst:.6f} | {worst_rel:.3g} |")
    # G3: dot_scaled
    out("\n### G3: `tl.dot_scaled` (`\"e4m3\"` by `\"e4m3\"`) on window inputs\n")
    out("`DotScaledE4m3`'s domain `scaledDomBF16` admits only `0` and normal `bfloat16` scaled values, so the emulated"
        " `bfloat16` dot has no subnormal operand. Per chunk of 16 one product `448 * 448` (scale `2^0`) and 15 products"
        " of `1.875 * 2^-6` (normal `e4m3`, and a subnormal `e4m3` element `2^-9`, which `bfloat16` holds as a normal"
        " value), lost below the window; the error against the fact's bound `relDot16 K mag + absDot16 K`.\n")
    out("| fast_math | K | rows | largest error / bound | every scaled value in scaledDomBF16 |")
    out("|---|---|---|---|---|")
    for FM in (False, True):
        for K in (32, 64, 256):
            M = N = 16
            av = torch.zeros(M, K, device=DEV); bv = torch.zeros(K, N, device=DEV)
            for i in range(M):
                for j in range(K // 16):
                    pos = j * 16 + random.randrange(16)
                    for k in range(j * 16, j * 16 + 16):
                        if k == pos:
                            av[i, k] = 448.0; bv[k, i] = 448.0
                        else:
                            av[i, k] = 1.875 * 2.0 ** -6 if (k % 3) else 2.0 ** -9
                            bv[k, i] = 1.875 * 2.0 ** -6
            a8 = av.to(torch.float8_e4m3fn); b8 = bv.to(torch.float8_e4m3fn)
            assert torch.equal(a8.float(), av) and torch.equal(b8.float(), bv)
            sa = torch.full((M, K // 32), 127, dtype=torch.uint8, device=DEV)
            sb = torch.full((N, K // 32), 127, dtype=torch.uint8, device=DEV)
            c = torch.zeros(M, N, device=DEV)
            o = torch.empty(M, N, device=DEV)
            dsk[(1,)](a8.view(torch.uint8), sa, b8.view(torch.uint8), sb, c, o, M, N, K, FM, num_warps=4)
            worst = 0.0
            indom = bool(((av == 0) | (av.abs() >= 2.0 ** -126)).all() and ((bv == 0) | (bv.abs() >= 2.0 ** -126)).all())
            for i in range(M):
                prods = [Fr(float(x)) * Fr(float(y)) for x, y in zip(av[i].tolist(), bv[:, i].tolist())]
                S = sum(prods); mag = sum(abs(p) for p in prods)
                err = abs(Fr(float(o[i, i])) - S)
                worst = max(worst, float(err / (Fr(K + 2, 2 ** 22) * mag + Fr(K + 2) / Fr(2) ** 126)))
            out(f"| {FM} | {K} | {M} | {worst:.6f} | {indom} |")


def main():
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    part_a(out)
    part_e(out)
    part_f(out)
    part_g(out)
    part_d(out)
    part_bc(out)
    out(f"\nZero results equal to the model's (all parts): {ZSIGN[0]}; of them with the opposite sign: {ZSIGN[1]}")
    out(f"\n(run time {time.time() - t0:.0f} s)")
    with open("evidence/num-dot-sub16-round25/" + ("quick.txt" if QUICK else "out.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
