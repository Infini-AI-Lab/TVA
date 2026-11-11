"""TVA library evidence (library issue L150, round 32): a 16-bit `tl.dot` into float32 on sm_90, chunk by chunk.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=2 .venv/bin/python -I evidence/num-dot-chunk16/run.py
Output: evidence/num-dot-chunk16/out.txt (README.md says what it shows). `--quick` runs a smoke test (written to
quick.txt, not the evidence).

The facts checked (lean/TVA/Rules/Num/Dot16Chunk.lean): `DotF16Chunk` and `DotBF16Chunk`, for float16 (bfloat16)
operands into float32, `max_num_imprecise_acc = 0`, any `input_precision`. Per output element there are partial
values s_0 = acc, ..., s_n = result (n = ceil(K/16)), one step per chunk of 16 consecutive k in increasing order, with

    |s_{j+1} - (s_j + c_j)| <= min(2^-20 Mj, 2 A_j + 2^-24 Mj) + 2^-22 |s_j + c_j| + 2^-145

for every Mj at least |s_j| and every lifted product lift(a_k) lift(b_k) of the chunk (lift(x) = 0 for x = 0 and
max(|x|, 2^emin) otherwise); c_j is the chunk's exact sum, A_j = sum |a_k b_k| its exact magnitude. Each element is
checked at the smallest Mj (the larger of |s_j| and the chunk's largest lifted product), where the bound is tightest.

The per-instruction model is evidence/num-dot16-model's (copied below, with its input kinds): the terms are the
accumulator and the 16 exact products, aligned to the largest exponent field E (a subnormal operand reads as the
smallest normal; the accumulator's field is max(floor(log2|s|), -126)), each truncated toward zero to a multiple of
u = 2^(E-25), summed exactly, and the sum truncated toward zero to float32.

Parts:
  S. Single chunks (K = 16) on the device: the partial values are the accumulator and the result, so each element
     is checked against the step bound with no model; the model is also compared with the device.
  C. Chains at every K that compiles: the device result against the model; each step of the model's chain (equal to
     the device's result) against the step bound; and the result against the composed interval errM from the exact
     inputs (Lemmas/Dot16Chunk.lean, `DotChunk16Loss.err`), next to DotF16's bound.
  L. Fed-back accumulators: tutorial 03's K loop and tutorial 06's `acc = tl.dot(p.to(fp16), v, acc)`, T steps of
     K = 64, every acc_t stored: each dot's steps from the stored accumulator before it, each dot against errM, and
     the whole chain (no rescale) against errM over 4T chunks, next to DotF16 composed step by step.
"""
import math
import random
import sys
import time
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

torch.manual_seed(20261108)
random.seed(20261108)
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


# ================================================================================ the chunk facts (round 32)

RHO, SIG, ALPHA, KAPPA, RHOP = 2.0 ** -20, 2.0 ** -22, 2.0 ** -145, 2.0, 2.0 ** -24
RHO_F, SIG_F, ALPHA_F, KAPPA_F, RHOP_F = Fr(1, 2 ** 20), Fr(1, 2 ** 22), Fr(1, 2 ** 145), Fr(2), Fr(1, 2 ** 24)
SLACK = 2.0 ** -48      # float64 slack per unit of magnitude (17 terms at 2^-53 each, with margin)


def chunks16(K):
    return (K + 15) // 16


def rel_room(K):
    return (1 + RHO) ** chunks16(K) - 1


def abs_room(K):
    return chunks16(K) * (1 + RHO) ** chunks16(K) * ALPHA


def gen2(kind, M, N, K, f):
    """The round-32 kinds (the window's worst cases for the step bound), else evidence/num-dot16-model's."""
    dt, _, frac, emin, fmax, sig, emax = FMT[f]
    if kind in ("accdom", "accdom2"):
        # the accumulator sets the window (field E_i per row), 2^E_i times 1, 1 + 2^-23 or 2 - 2^-23; every product
        # of the row just below u = 2^(E_i - 25) (`accdom`: lost whole) or just below 2u (`accdom2`: truncated to u,
        # losing almost u), of the accumulator's sign or the opposite one
        if f == "f16":
            Es = torch.randint(12, 42, (M, 1), device=DEV)
        else:
            Es = torch.randint(-100, 121, (M, 1), device=DEV)
        xa = Es - (26 if kind == "accdom" else 25)
        a = sig * torch.exp2(xa.float()).expand(M, K).clone()
        b = torch.ones(K, N, device=DEV)
        m = torch.tensor([1.0, 1 + 2.0 ** -23, 2 - 2.0 ** -23], device=DEV)[torch.randint(0, 3, (M, N), device=DEV)]
        rs = sgn((M, 1))
        c = rs * torch.exp2(Es.float()) * m
        a = a * torch.where(torch.rand(M, 1, device=DEV) < 0.5, rs, -rs)
        return q16(a, f).contiguous(), b.contiguous(), c.float().contiguous()
    if kind == "subacc":
        # bfloat16: a subnormal float32 accumulator sets the window (field -126, u = 2^-151); products just below u
        if f != "bf16":
            return gen("tiny", M, N, K, f)
        a = torch.full((M, K), sig * 2.0 ** -76, device=DEV) * sgn((M, 1))
        b = torch.full((K, N), 2.0 ** -76, device=DEV)
        c = torch.randint(1, 2 ** 22, (M, N), device=DEV).float() * 2.0 ** -149 * sgn((M, N))
        return q16(a, f).contiguous(), b.contiguous(), c.contiguous()
    return gen(kind, M, N, K, f)


KINDS2 = KINDS + ["accdom", "accdom2", "subacc"]
WORSTK = {"subwin", "winworst", "accdom", "accdom2", "subacc", "subrand", "bits", "sub", "softmax", "tiny"}


def chunk_terms(Ad, Bd, f, k0, k1):
    P = Ad[:, None, k0:k1] * Bd[k0:k1, :].T[None, :, :]
    La, Lb = lift(Ad, f), lift(Bd, f)
    PL = La[:, None, k0:k1] * Lb[k0:k1, :].T[None, :, :]
    return P, PL


def step_ratios(Ad, Bd, ps, f):
    """Per element, the largest step over its chunks against the bound (at the smallest Mj), and against each arm
    alone; float64 values with SLACK times the magnitudes added to the error (an upper bound on the ratio)."""
    M, K = Ad.shape
    rmin = torch.zeros(M, Bd.shape[1], dtype=F64, device=DEV)
    r1 = torch.zeros_like(rmin); r2 = torch.zeros_like(rmin)
    for j, k0 in enumerate(range(0, K, 16)):
        P, PL = chunk_terms(Ad, Bd, f, k0, min(K, k0 + 16))
        s0, s1 = ps[j], ps[j + 1]
        cj = P.sum(-1); Aj = P.abs().sum(-1)
        Mj = torch.maximum(s0.abs(), PL.amax(-1))
        err = (s1 - (s0 + cj)).abs() + SLACK * (s0.abs() + Aj)
        tail = SIG * (s0 + cj).abs() + ALPHA
        a1 = RHO * Mj; a2 = KAPPA * Aj + RHOP * Mj
        rmin = torch.maximum(rmin, err / (torch.minimum(a1, a2) + tail))
        r1 = torch.maximum(r1, err / (a1 + tail)); r2 = torch.maximum(r2, err / (a2 + tail))
    return rmin, r1, r2


def exact_steps(arow, bcol, ps_el, f):
    """One element's largest step ratio in exact rationals (python floats in; ps_el the partial values)."""
    mn = 2.0 ** FMT[f][3]
    L = lambda x: Fr(0) if x == 0 else max(Fr(abs(x)), Fr(mn))
    worst = Fr(0)
    for j, k0 in enumerate(range(0, len(arow), 16)):
        prods = [(arow[k], bcol[k]) for k in range(k0, min(len(arow), k0 + 16))]
        c = sum((Fr(x) * Fr(y) for x, y in prods), Fr(0))
        A = sum((abs(Fr(x) * Fr(y)) for x, y in prods), Fr(0))
        s0, s1 = Fr(ps_el[j]), Fr(ps_el[j + 1])
        Mj = max([abs(s0)] + [L(x) * L(y) for x, y in prods])
        bnd = min(RHO_F * Mj, KAPPA_F * A + RHOP_F * Mj) + SIG_F * abs(s0 + c) + ALPHA_F
        worst = max(worst, abs(s1 - (s0 + c)) / bnd)
    return worst


def errm_bound(Ad, Bd, C0, f):
    """errM from the exact inputs (Lemmas/Dot16Chunk.lean): float64, with the partial sums' own rounding added
    (delta); also the exact sum S_n, delta, and DotF16's whole-dot bound relDot16(K) L + absDot16(K)."""
    M, K = Ad.shape
    S = C0.clone(); e = torch.zeros_like(C0)
    mag = C0.abs().clone(); Lt = C0.abs().clone()
    for j, k0 in enumerate(range(0, K, 16)):
        P, PL = chunk_terms(Ad, Bd, f, k0, min(K, k0 + 16))
        mag = mag + P.abs().sum(-1); Lt = Lt + PL.sum(-1)
    delta = (K + 17) * 2.0 ** -52 * mag
    for j, k0 in enumerate(range(0, K, 16)):
        P, PL = chunk_terms(Ad, Bd, f, k0, min(K, k0 + 16))
        X = torch.maximum(S.abs(), PL.amax(-1)) + delta
        Aj = P.abs().sum(-1)
        S1 = S + P.sum(-1)
        e = (1 + SIG) * e + torch.minimum(RHO * (X + e), KAPPA * Aj + RHOP * (X + e)) + SIG * (S1.abs() + delta) + ALPHA
        S = S1
    old = rel16(K) * Lt + abs16(K)
    return e * (1 + 2.0 ** -30), S, delta, old, mag


class Agg2:
    def __init__(self):
        self.n = self.viol = self.nonfin = self.sim_n = self.sim_eq = 0
        self.r = self.r1 = self.r2 = self.rm = self.rold = 0.0
        self.worst = None
        self.exact_checked = 0

    def add(self, Ad, Bd, C0, Rd, f, ps_dev, label, sim=None, composed=True):
        K = Ad.shape[1]
        mag = C0.abs() + (Ad.abs() @ Bd.abs())
        room = mag * (1 + 2.0 ** -40) * (1 + rel_room(K)) + abs_room(K) <= F32MAX
        fin = torch.isfinite(Rd)
        self.n += int(room.sum()); self.nonfin += int((room & ~fin).sum())
        if sim is not None:
            same = (sim == Rd) | (torch.isnan(sim) & torch.isnan(Rd))
            self.sim_n += int(room.sum()); self.sim_eq += int((same & room).sum())
        ok = room & fin
        if not bool(ok.any()):
            return
        rmin, r1, r2 = step_ratios(Ad, Bd, ps_dev, f)
        rmin = torch.where(ok, rmin, torch.zeros_like(rmin))
        # every element above 0.95 of the bound (at most 64 per call) and the largest one, in exact rationals
        flag = (rmin > 0.95).nonzero().tolist()[:64]
        imax = divmod(int(rmin.argmax()), rmin.shape[1])
        for (i, n) in flag + [list(imax)]:
            ex = exact_steps(Ad[i].tolist(), Bd[:, n].tolist(), [float(p[i, n]) for p in ps_dev], f)
            self.exact_checked += 1
            if ex > 1:
                self.viol += 1
            rmin[i, n] = float(ex)
        m = float(rmin.max())
        if m >= self.r:
            self.r = m; self.worst = (label, imax)
        self.r1 = max(self.r1, float(torch.where(ok, r1, torch.zeros_like(r1)).max()))
        self.r2 = max(self.r2, float(torch.where(ok, r2, torch.zeros_like(r2)).max()))
        if composed:
            e, S, delta, old, _ = errm_bound(Ad, Bd, C0, f)
            err = (Rd - S).abs() + delta
            rm = torch.where(ok, err / e, torch.zeros_like(e))
            ro = torch.where(ok, err / old, torch.zeros_like(e))
            if bool((rm > 1).any()):
                self.viol += int((rm > 1).sum())
            self.rm = max(self.rm, float(rm.max())); self.rold = max(self.rold, float(ro.max()))


def reps2(kind):
    if QUICK:
        return 1
    return 6 if kind in WORSTK else 3


FORMATS = ["f16", "bf16"]
SHAPES2 = [(16, 16, 1), (32, 32, 2), (64, 64, 2), (64, 64, 4), (128, 64, 4), (128, 128, 8)]
if QUICK:
    SHAPES2 = [(16, 16, 1), (64, 64, 4)]


def part_s(out):
    out("\n## Part S: single chunks on the device (K = 16)\n")
    out("Each element against the step bound with the device's own partial values (the accumulator and the result),"
        " at the smallest `Mj`; `arm 1`, `arm 2`: the ratio to each arm alone (plus the `2^-22`, `2^-145` terms);"
        " every element above 0.95 and each call's largest recomputed in exact rationals. `model = device`: as values.\n")
    out("| operands | M x N, warps | variant | path | elements in range | outside | model = device | largest ratio | arm 1 | arm 2 | worst kind |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    tot = {}
    for f in FORMATS:
        for (M, N, nw) in SHAPES2:
            for (use_acc, a_reg, b_t, label) in [(True, False, False, "acc"), (False, False, False, "no acc"),
                                                  (True, True, True, "a in registers, b transposed")]:
                if QUICK and label != "acc":
                    continue
                agg = Agg2(); path = None
                for kind in KINDS2:
                    for _ in range(reps2(kind) * 2):
                        A, B, C = gen2(kind, M, N, 16, f)
                        R, path = run_dot(A, B, C, f, nw, use_acc, a_reg, b_t)
                        Ad, Bd, Cd, Rd = A.double(), B.double(), C.double(), R.double()
                        C0 = Cd if use_acc else torch.zeros_like(Cd)
                        sim = model(Ad, Bd, Cd, f, use_acc)
                        agg.add(Ad, Bd, C0, Rd, f, [C0, Rd], kind, sim, composed=False)
                wk = agg.worst[0] if agg.worst else "-"
                out(f"| {NAME[f]} | {M} x {N}, {nw} | {label} | {path} | {agg.n} | {agg.viol + agg.nonfin} | {agg.sim_eq}/{agg.sim_n} | {agg.r:.4f} | {agg.r1:.4f} | {agg.r2:.4f} | {wk} |")
                t = tot.setdefault((f, path), [0, 0, 0, 0, 0.0, 0.0, 0.0, 0])
                t[0] += agg.n; t[1] += agg.viol + agg.nonfin; t[2] += agg.sim_eq; t[3] += agg.sim_n
                t[4] = max(t[4], agg.r); t[5] = max(t[5], agg.r1); t[6] = max(t[6], agg.r2); t[7] += agg.exact_checked
    out("\nTotals:\n")
    out("| operands | path | elements in range | outside | model = device | largest ratio | arm 1 | arm 2 | exact rechecks |")
    out("|---|---|---|---|---|---|---|---|---|")
    for (f, path), t in sorted(tot.items()):
        out(f"| {NAME[f]} | {path} | {t[0]} | {t[1]} | {t[2]}/{t[3]} | {t[4]:.4f} | {t[5]:.4f} | {t[6]:.4f} | {t[7]} |")
    return tot


def ks2(M, N):
    if QUICK:
        return [32, 128]
    return [32, 64, 128, 256, 512, 1024, 2048] if M * N <= 32 * 32 else [32, 64, 128, 256, 512, 1024]


def part_c(out):
    out("\n## Part C: chains at every K that compiles\n")
    out("The device result against the model; each step of the model's chain (equal to the device's result) against"
        " the step bound; the result against the composed `errM` from the exact inputs (`errM ratio`); `DotF16 ratio`:"
        " the same error against `relDot16 K L + absDot16 K`. Rejected launches (out of resources) are recorded.\n")
    out("| operands | M x N, warps | path | K | elements in range | outside | model = device | step ratio | errM ratio | DotF16 ratio |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    tot = {}
    for f in FORMATS:
        for (M, N, nw) in SHAPES2:
            for K in ks2(M, N):
                agg = Agg2(); path = None
                try:
                    for kind in KINDS2:
                        if kind == "bits" and K > 512:
                            continue
                        for _ in range(reps2(kind)):
                            A, B, C = gen2(kind, M, N, K, f)
                            R, path = run_dot(A, B, C, f, nw, True)
                            Ad, Bd, Cd, Rd = A.double(), B.double(), C.double(), R.double()
                            sim, ps = model(Ad, Bd, Cd, f, True, partials=True)
                            agg.add(Ad, Bd, Cd, Rd, f, ps, kind, sim)
                except triton.runtime.errors.OutOfResources:
                    out(f"| {NAME[f]} | {M} x {N}, {nw} | rejected | {K} | (out of resources) | | | | | |")
                    break
                out(f"| {NAME[f]} | {M} x {N}, {nw} | {path} | {K} | {agg.n} | {agg.viol + agg.nonfin} | {agg.sim_eq}/{agg.sim_n} | {agg.r:.4f} | {agg.rm:.4f} | {agg.rold:.4f} |")
                t = tot.setdefault((f, path), [0, 0, 0, 0, 0.0, 0.0])
                t[0] += agg.n; t[1] += agg.viol + agg.nonfin; t[2] += agg.sim_eq; t[3] += agg.sim_n
                t[4] = max(t[4], agg.r); t[5] = max(t[5], agg.rm)
    out("\nTotals:\n")
    out("| operands | path | elements in range | outside | model = device | step ratio | errM ratio |")
    out("|---|---|---|---|---|---|---|")
    for (f, path), t in sorted(tot.items()):
        out(f"| {NAME[f]} | {path} | {t[0]} | {t[1]} | {t[2]}/{t[3]} | {t[4]:.4f} | {t[5]:.4f} |")
    return tot


def part_l(out):
    out("\n## Part L: fed-back accumulators\n")
    out("`for t in range(T): [acc = acc * alpha_t]; acc = tl.dot(p_t, v_t, acc)`, K = 64 per dot, every acc_t"
        " stored. `K loop`: tutorial 03 (p_t loaded in the format); `p.to`: tutorial 06 (p_t converted in registers"
        " from a softmax). Per dot, from the stored accumulator before it: the model against the device, each step"
        " against the step bound, the dot against errM. Whole chain (no rescale): the final result against errM over"
        " 4T chunks from the exact inputs (`whole errM ratio`); `bound / DotF16 composed`: errM over the whole chain"
        " divided by the bound a kernel gets by composing DotF16 step by step (e_{t+1} = e_t + relDot16(64)"
        " (|S_t| + e_t + L_t) + absDot16(64)), the largest over the elements.\n")
    out("| operands | M x N, warps | path | p, v | rescale | T | elements | outside | model = device | step ratio | per-dot errM ratio | whole errM ratio | bound / DotF16 composed |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    KC = 64
    viol = 0; mr = 0.0
    shapes = [(128, 64, 4), (32, 64, 2)] if QUICK else [(128, 64, 4), (128, 128, 8), (64, 64, 4), (32, 64, 2), (16, 64, 1)]
    for f in FORMATS:
        for (M, N, nw) in shapes:
            for pk in ["K loop normal", "K loop accdom", "p.to softmax"]:
                a_reg = pk.startswith("p.to")
                for scale in ((False,) if not a_reg else (False, True)):
                    for T in ([1, 8] if QUICK else ([4, 32, 256] if not scale else [32])):
                        P = torch.zeros(T, M, KC, device=DEV); V = torch.zeros(T, KC, N, device=DEV)
                        c0 = torch.randn(M, N, device=DEV)
                        for t in range(T):
                            if pk == "p.to softmax":
                                p, _, _ = gen("softmax", M, N, KC, f)
                                v = q16(torch.randn(KC, N, device=DEV), f)
                            elif pk == "K loop accdom":
                                p, v, cc = gen2("accdom", M, N, KC, f)
                                if t == 0:
                                    c0 = cc
                            else:
                                p, v, _ = gen("normal", M, N, KC, f)
                            P[t] = p; V[t] = v
                        al = (torch.rand(T, device=DEV) * 0.75 + 0.25) if scale else torch.ones(T, device=DEV)
                        accs = torch.empty(T, M, N, device=DEV); o = torch.empty(M, N, device=DEV)
                        V16 = V.to(FMT[f][0]).contiguous()
                        Pin = P.contiguous() if a_reg else P.to(FMT[f][0]).contiguous()
                        h = chain[(1,)](Pin, V16, c0, al, accs, o, T, M, N, KC, FMT[f][1], a_reg, scale, True, num_warps=nw)
                        path = path_of(h)
                        assert torch.equal(o, accs[T - 1])
                        agg = Agg2(); prev = c0
                        for t in range(T):
                            vc = prev * al[t] if scale else prev
                            Ad, Bd, Cd, Rd = P[t].double(), V[t].double(), vc.double(), accs[t].double()
                            sim, ps = model(Ad, Bd, Cd, f, True, partials=True)
                            agg.add(Ad, Bd, Cd, Rd, f, ps, ("chain", t), sim)
                            prev = accs[t]
                        whole = cmp_ = "-"
                        if not scale:
                            Aall = torch.cat([P[t] for t in range(T)], 1).double()
                            Ball = torch.cat([V[t] for t in range(T)], 0).double()
                            e, S, delta, _, _ = errm_bound(Aall, Ball, c0.double(), f)
                            err = (o.double() - S).abs() + delta
                            fin = torch.isfinite(o)
                            rw = float(torch.where(fin, err / e, torch.zeros_like(e)).max())
                            viol += int(((err / e) > 1).sum())
                            mr = max(mr, rw)
                            # DotF16 composed step by step from the exact partial sums
                            eo = torch.zeros_like(e); St = c0.double()
                            for t in range(T):
                                Lt = lift(P[t].double(), f) @ lift(V[t].double(), f)
                                eo = eo + rel16(KC) * (St.abs() + eo + Lt) + abs16(KC)
                                St = St + P[t].double() @ V[t].double()
                            whole = f"{rw:.4f}"
                            cmp_ = f"{float((e / eo).max()):.4f}"
                        viol += agg.viol + agg.nonfin
                        out(f"| {NAME[f]} | {M} x {N}, {nw} | {path} | {pk} | {'yes' if scale else 'no'} | {T} | {M * N} | {agg.viol + agg.nonfin} | {agg.sim_eq}/{agg.sim_n} | {agg.r:.4f} | {agg.rm:.4f} | {whole} | {cmp_} |")
    out(f"\nElements outside in Part L: {viol}; largest whole-chain errM ratio {mr:.4f}")
    return viol


def main():
    path = "evidence/num-dot-chunk16/" + ("quick.txt" if QUICK else "out.txt")
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)
    t0 = time.time()
    out("# num-dot-chunk16: output")
    out(f"\nTriton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}"
        f" (sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
    part_s(out)
    part_c(out)
    part_l(out)
    out(f"\nRun time: {time.time() - t0:.0f} s")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
