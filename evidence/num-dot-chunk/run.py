"""TVA library evidence (library issue L124): an fp8 `tl.dot` into float32, chunk by chunk, on sm_90.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-chunk/run.py
Output: evidence/num-dot-chunk/out.txt (README.md says what it shows).

The facts checked (docs/rules/num.md, "`tl.dot` of fp8, chunk by chunk"): for an fp8 x fp8 `tl.dot` into
float32 with Triton's default max_num_imprecise_acc (2^30 on sm_90), per output element, with `K` products
in chunks of 32 consecutive k (chunk j: 32j <= k < 32j + 32), there are partial values s_0 = acc, ...,
s_n = result (n = ceil(K/32)) such that each step is within

    rho * M_j + sigma * |s_j + c_j| + alpha,     rho = 2^-7, sigma = 2^-12, alpha = 2^-126,

of s_j + c_j, where c_j is the chunk's exact sum and M_j is any bound on |s_j| and on every lifted product
L(a_k) L(b_k) of the chunk (L(x) = 0 for x = 0, else max(|x|, 2^emin), the format's smallest normal).

Parts:
  A. The window: structured inputs, against the bit-level model below (`model`); A2, the mma.sync path.
  B. The model against the device, bit for bit (wgmma path), on adversarial inputs at every K that compiles.
  C. The facts on both instruction paths: every element against the bound the facts imply for it (the
     per-chunk bound composed over the chunks with the exact partial sums and the worst case of the
     unknown partial values), and, on wgmma, each chunk of the model's chain against the per-chunk bound.
  D. Chains of tl.dot(p, v, acc) feeding the accumulator back (with p converted to fp8 in registers, as
     tutorial 06's _attn_fwd does), with and without a rescale of the accumulator between the dots.
  E. The order of the chunks.
  F. The existing facts DotE4m3, ... (relDot8) on the inputs of Part A's subnormal window.

The bit-level model of one wgmma chunk (Part A establishes it, Part B checks it): the terms are the
accumulator s (float32) and the 32 exact products p_k = a_k b_k. Each nonzero term has a raw exponent:
e(p_k) = e(a_k) + e(b_k), with e(x) = max(floor(log2|x|), emin) for an fp8 operand (a subnormal counts at
emin), and e(s) = max(floor(log2|s|), -126). With E the largest raw exponent and u = 2^(E-13), every term is
truncated toward zero to a multiple of u, the truncated terms are summed exactly, and the sum is truncated
toward zero to 14 significant bits. That is the new accumulator.
"""
import math
import random
import sys
import time
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

torch.manual_seed(20261031)
random.seed(20261031)
DEV = "cuda"
F64 = torch.float64

RHO, SIG, ALP = 2.0 ** -7, 2.0 ** -12, 2.0 ** -126
RHO_Q, SIG_Q, ALP_Q = Fr(1, 2 ** 7), Fr(1, 2 ** 12), Fr(1, 2 ** 126)
FMT = {  # torch dtype, Triton dtype, fraction bits, emin, max value, largest significand
    "e5": (torch.float8_e5m2, tl.float8e5, 2, -14, 57344.0, 1.75),
    "e4": (torch.float8_e4m3fn, tl.float8e4nv, 3, -6, 448.0, 1.875),
}
NAME = {"e5": "float8e5", "e4": "float8e4nv"}
FACT = {("e4", "e4"): "DotE4m3Chunk", ("e5", "e5"): "DotE5m2Chunk", ("e4", "e5"): "DotE4m3E5m2Chunk",
        ("e5", "e4"): "DotE5m2E4m3Chunk"}
PAIRS = [("e5", "e5"), ("e4", "e4"), ("e4", "e5"), ("e5", "e4")]
BIG = 1 << 20
QUICK = "--quick" in sys.argv       # a smoke test on a few configurations (not the evidence)


# ----------------------------------------------------------------------------------------------- kernels

@triton.jit
def dotk(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         USE_ACC: tl.constexpr, A_REG: tl.constexpr, B_T: tl.constexpr, ADT: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    if A_REG:
        # `a` arrives as float32 values that fp8 represents exactly and is converted in registers, as
        # `p.to(tl.float8e5)` in _attn_fwd
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
        o = tl.dot(a, b, c)
    else:
        o = tl.dot(a, b)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], o)


@triton.jit
def chain(a_ptr, b_ptr, c_ptr, al_ptr, accs_ptr, o_ptr, T, M: tl.constexpr, N: tl.constexpr,
          KC: tl.constexpr, ADT: tl.constexpr, SCALE: tl.constexpr, STORE: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, KC)
    acc = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    for t in range(T):
        if SCALE:
            acc = acc * tl.load(al_ptr + t)
        p = tl.load(a_ptr + t * M * KC + rm[:, None] * KC + rk[None, :]).to(ADT)
        v = tl.load(b_ptr + t * KC * N + rk[:, None] * N + rn[None, :])
        acc = tl.dot(p, v, acc)
        if STORE:
            tl.store(accs_ptr + t * M * N + rm[:, None] * N + rn[None, :], acc)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc)


def path_of(h):
    ptx = h.asm["ptx"]
    return "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "?")


def run_dot(a, b, c, da, db, nw, use_acc=True, a_reg=False, b_t=False):
    """a [M,K], b [K,N] float32 tensors of values the formats represent exactly; c [M,N] float32."""
    M, K = a.shape
    N = b.shape[1]
    a8 = a.to(FMT[da][0])
    b8 = b.to(FMT[db][0])
    assert torch.equal(a8.float(), a) and torch.equal(b8.float(), b), "operand not representable"
    ain = a.contiguous() if a_reg else a8.contiguous()
    bin_ = b8.t().contiguous() if b_t else b8.contiguous()
    o = torch.empty(M, N, device=DEV)
    h = dotk[(1,)](ain, bin_, c.contiguous(), o, M, N, K, use_acc, a_reg, b_t, FMT[da][1], num_warps=nw)
    return o, path_of(h)


# ------------------------------------------------------------------------------------------ the model

def rawexp8(x, f):
    e = torch.frexp(x.abs())[1].to(torch.int64) - 1
    return torch.clamp(e, min=FMT[f][3])


def model(A, B, C, da, db, use_acc=True, partials=False):
    """The wgmma model on float64 tensors (exact here: each truncated term is an integer multiple of u below
    2^17 u, and 33 of them sum below 2^23 u). Returns the result and, optionally, the partial values."""
    M, K = A.shape
    N = B.shape[1]
    ea = rawexp8(A, da)
    eb = rawexp8(B, db)
    s = C.clone() if use_acc else torch.zeros_like(C)
    ps = [s]
    for k0 in range(0, K, 32):
        k1 = min(K, k0 + 32)
        P = A[:, None, k0:k1] * B[k0:k1, :].T[None, :, :]
        EP = ea[:, None, k0:k1] + eb[k0:k1, :].T[None, :, :]
        EP = torch.where(P == 0, torch.full_like(EP, -BIG), EP)
        es = torch.clamp(torch.frexp(s.abs())[1].to(torch.int64) - 1, min=-126)
        es = torch.where(s == 0, torch.full_like(es, -BIG), es)
        E = torch.maximum(EP.amax(-1), es)
        u = torch.ldexp(torch.ones_like(s), torch.clamp(E, min=-1000) - 13)
        S1 = torch.trunc(s / u) + torch.trunc(P / u[..., None]).sum(-1)
        S1 = S1 * u
        S1 = torch.where(E <= -BIG // 2, torch.zeros_like(S1), S1)
        er = torch.frexp(S1.abs())[1].to(torch.int64) - 1
        u2 = torch.ldexp(torch.ones_like(s), er - 13)
        s = torch.where(S1 == 0, S1, torch.trunc(S1 / u2) * u2)
        ps.append(s)
    return (s, ps) if partials else s


def model_py(acc, prods_ea_eb):
    """One chunk of the model in exact rationals (for Part A's table): prods_ea_eb = [(p, e(p))]."""
    terms = [(Fr(acc), max(math.frexp(abs(acc))[1] - 1, -126))] if acc != 0 else []
    terms += [(Fr(p), e) for p, e in prods_ea_eb if p != 0]
    if not terms:
        return 0.0
    E = max(e for _, e in terms)
    u = Fr(2) ** (E - 13)
    S1 = sum((t / u).__trunc__() for t, _ in terms) * u
    if S1 == 0:
        return 0.0
    er = math.frexp(float(abs(S1)))[1] - 1
    if Fr(2) ** er > abs(S1):
        er -= 1
    u2 = Fr(2) ** (er - 13)
    return float((S1 / u2).__trunc__() * u2)


# --------------------------------------------------------------------------------- the bound (Part C)

def lift(X, f):
    return torch.where(X != 0, torch.clamp(X.abs(), min=2.0 ** FMT[f][3]), torch.zeros_like(X))


def lsb8(X, f):
    e = torch.frexp(X.abs())[1].to(torch.int64) - 1
    return torch.where(X != 0, torch.clamp(e, min=FMT[f][3]) - FMT[f][2], torch.full_like(e, BIG))


def chunk_data(A, B, C, da, db, use_acc=True):
    """Per element: the exact chunk sums c_j, lifted chunk maxima m_j, |acc| + sum |a b|, and whether float64
    holds every partial sum exactly (all terms multiples of 2^q with total magnitude below 2^(q+53))."""
    M, K = A.shape
    N = B.shape[1]
    C0 = C if use_acc else torch.zeros_like(C)
    La, Lb = lift(A, da), lift(B, db)
    qa, qb = lsb8(A, da), lsb8(B, db)
    ec = torch.clamp(torch.frexp(C0.abs())[1].to(torch.int64) - 1, min=-126) - 23
    q = torch.where(C0 != 0, ec, torch.full_like(ec, BIG))
    tot = C0.abs().clone()
    cs, ms = [], []
    for k0 in range(0, K, 32):
        k1 = min(K, k0 + 32)
        P = A[:, None, k0:k1] * B[k0:k1, :].T[None, :, :]
        cs.append(P.sum(-1))
        tot = tot + P.abs().sum(-1)
        ms.append((La[:, None, k0:k1] * Lb[k0:k1, :].T[None, :, :]).amax(-1))
        Q = qa[:, None, k0:k1] + qb[k0:k1, :].T[None, :, :]
        q = torch.minimum(q, torch.where(P != 0, Q, torch.full_like(Q, BIG)).amin(-1))
    lim = torch.ldexp(torch.ones_like(tot), torch.clamp(q, max=900) + 53)
    exact = (tot * (1 + 2.0 ** -40) < lim) | (q >= BIG)
    return C0, cs, ms, tot, exact


def bound_check(A, B, C, R, da, db, use_acc=True, sim_partials=None):
    """The facts' implied interval for every element, in float64 (exact sums where `exact`); returns a dict
    of tensors. ratio: the error over the half-width of the interval on its side (1 = on the bound)."""
    K = A.shape[1]
    C0, cs, ms, tot, exact = chunk_data(A, B, C, da, db, use_acc)
    S = C0.clone()
    dp = torch.zeros_like(S)
    dm = torch.zeros_like(S)
    e = torch.zeros_like(S)
    step_ratio = torch.zeros_like(S)
    for j, (c, m) in enumerate(zip(cs, ms)):
        S1 = S + c
        dp = dp + RHO * torch.maximum((S + dp).abs(), m) + SIG * (S1 + dp).abs() + ALP
        dm = dm + RHO * torch.maximum((S - dm).abs(), m) + SIG * (S1 - dm).abs() + ALP
        e = (1 + RHO + SIG) * e + RHO * torch.maximum(S.abs(), m) + SIG * S1.abs() + ALP
        if sim_partials is not None:
            s0, s1 = sim_partials[j], sim_partials[j + 1]
            st = (s1 - (s0 + c)).abs() / (RHO * torch.maximum(s0.abs(), m) + SIG * (s0 + c).abs() + ALP)
            step_ratio = torch.maximum(step_ratio, st)
        S = S1
    err = R - S
    ratio = torch.where(err >= 0, err / dp, -err / dm)
    n = (K + 31) // 32
    rel8 = (1 + 2.0 ** -7) ** n - 1
    abs8 = n * (1 + 2.0 ** -7) ** n * 2.0 ** -126
    room = tot + rel8 * tot + abs8 <= 3.4028234663852886e38
    old = err.abs() / (rel8 * tot + abs8)
    return dict(ratio=ratio, ratio_e=err.abs() / e, exact=exact, room=room, fin=torch.isfinite(R),
                old=old, step=step_ratio, S=S, err=err)


def exact_ratio(arow, bcol, c, r, da, db):
    """The same interval in exact rationals for one element (python floats in; the products as integers at
    the scale of the formats' smallest subnormals, the rest as Fractions)."""
    K = len(arow)
    sa, sb = FMT[da][2] - FMT[da][3], FMT[db][2] - FMT[db][3]     # 2^sa x is an integer for x in da
    ai = [int(x * 2.0 ** sa) for x in arow]
    bi = [int(y * 2.0 ** sb) for y in bcol]
    assert all(x * 2.0 ** sa == v for x, v in zip(arow, ai)) and all(y * 2.0 ** sb == v for y, v in zip(bcol, bi))
    scale = 1 << (sa + sb)
    la = [0 if v == 0 else max(abs(v), 1 << FMT[da][2]) for v in ai]     # lifted, at the same scale
    lb = [0 if v == 0 else max(abs(v), 1 << FMT[db][2]) for v in bi]
    S = Fr(c)
    dp = dm = e = Fr(0)
    for k0 in range(0, K, 32):
        ks = range(k0, min(K, k0 + 32))
        cj = Fr(sum(ai[k] * bi[k] for k in ks), scale)
        mj = Fr(max(la[k] * lb[k] for k in ks), scale)
        S1 = S + cj
        dp = dp + RHO_Q * max(abs(S + dp), mj) + SIG_Q * abs(S1 + dp) + ALP_Q
        dm = dm + RHO_Q * max(abs(S - dm), mj) + SIG_Q * abs(S1 - dm) + ALP_Q
        e = (1 + RHO_Q + SIG_Q) * e + RHO_Q * max(abs(S), mj) + SIG_Q * abs(S1) + ALP_Q
        S = S1
    err = Fr(r) - S
    return (err / dp if err >= 0 else -err / dm), abs(err) / e, S


# ------------------------------------------------------------------------------------------- inputs

def q8(x, f):
    """Round float32 values to the format (saturating), back to float32."""
    mx = FMT[f][4]
    return x.clamp(-mx, mx).to(FMT[f][0]).float()


def sgn(shape):
    return torch.where(torch.rand(shape, device=DEV) < 0.5, -1.0, 1.0)


def gen(kind, M, N, K, da, db):
    fa, fb = FMT[da], FMT[db]
    ma, mb = fa[5], fb[5]
    na, nb = 2.0 ** fa[3], 2.0 ** fb[3]
    if kind == "normal":
        a = torch.randn(M, K, device=DEV); b = torch.randn(K, N, device=DEV); c = torch.randn(M, N, device=DEV)
    elif kind == "spread":
        a = torch.randn(M, K, device=DEV) * torch.exp2(torch.randint(-4, 4, (M, K), device=DEV).float())
        b = torch.randn(K, N, device=DEV) * torch.exp2(torch.randint(-4, 4, (K, N), device=DEV).float())
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-8, 8, (M, N), device=DEV).float())
    elif kind == "accbig":
        a = torch.rand(M, K, device=DEV) + 1; b = torch.rand(K, N, device=DEV) + 1
        c = torch.exp2(torch.randint(4, 20, (M, N), device=DEV).float()) * (2 - 2.0 ** -23) * sgn((M, N))
    elif kind == "allones":
        # same-sign products with all fraction bits set, against a large accumulator with all bits set
        a = torch.full((M, K), ma, device=DEV) * torch.exp2(-torch.randint(0, 4, (M, K), device=DEV).float())
        b = torch.full((K, N), mb, device=DEV)
        c = torch.exp2(torch.randint(0, 9, (M, N), device=DEV).float()) * (2 - 2.0 ** -23) * sgn((M, N))
    elif kind == "onebig":
        # one large term per chunk among small ones with all fraction bits set
        t = torch.randint(1, 13, (M, K), device=DEV).float()
        a = ma * torch.exp2(-t)
        pos = torch.randint(0, 32, (M, (K + 31) // 32), device=DEV)
        for j in range(pos.shape[1]):
            a[torch.arange(M, device=DEV), j * 32 + pos[:, j]] = ma * 4
        b = torch.full((K, N), mb, device=DEV) * torch.where(torch.rand(K, N, device=DEV) < 0.1, -1.0, 1.0)
        c = torch.zeros(M, N, device=DEV)
    elif kind == "cancel":
        # pairwise cancellation: products k and k+1 cancel exactly, or nearly (b scaled by 1.25)
        a = torch.randn(M, K, device=DEV); b = torch.randn(K, N, device=DEV)
        a = q8(a, da); b = q8(b, db)
        a[:, 1::2] = -a[:, 0::2]
        near = torch.rand(1).item() < 0.5
        b[1::2, :] = q8(b[0::2, :] * (1.25 if near else 1.0), db)
        c = torch.randn(M, N, device=DEV) * 2.0 ** -12
    elif kind == "tiny":
        a = torch.randint(-3, 4, (M, K), device=DEV).float() * (na * 2.0 ** -fa[2])
        b = torch.randint(-3, 4, (K, N), device=DEV).float() * (nb * 2.0 ** -fb[2])
        c = torch.randint(-3, 4, (M, N), device=DEV).float() * 2.0 ** -140
    elif kind == "huge":
        a = (torch.rand(M, K, device=DEV) + 0.5) * fa[4] / 2 * sgn((M, K))
        b = (torch.rand(K, N, device=DEV) + 0.5) * fb[4] / 2
        c = torch.zeros(M, N, device=DEV)
    elif kind == "sub":
        # many subnormal operands
        a = torch.randn(M, K, device=DEV) * na; b = torch.randn(K, N, device=DEV) * nb * 4
        c = torch.randn(M, N, device=DEV) * na * nb * torch.randint(0, 2, (M, N), device=DEV)
    elif kind in ("subwin", "winworst"):
        # per chunk: the largest raw exponent E on one product (`subwin`: the smallest subnormal times
        # 2^eb, whose magnitude is 2^-m of 2^E; `winworst`: 1 times 2^eb, magnitude 2^E), the other 31
        # products just below the window's unit u = 2^(E-13) with all fraction bits set (lost whole), every
        # product of one sign per row; the accumulator 0, just below u (lost, adding to the products' loss),
        # or near the big product with bits below u, of either sign
        emax = {"e5": 15, "e4": 8}
        eb = emax[db] - int(torch.randint(0, 3, (1,)).item())
        E = (fa[3] if kind == "subwin" else 0) + eb
        u = 2.0 ** (E - 13)
        # a small product ma * 2^(E-14) < u as a * b with a = ma * 2^xa (normal in da), b = 2^xb (in db)
        xa = max(fa[3], E - 14 - emax[db])
        xa = min(xa, emax[da])
        xb = E - 14 - xa
        assert fb[3] - fb[2] <= xb <= emax[db], (da, db, kind, E, xa, xb)
        a = torch.zeros(M, K, device=DEV); b = torch.zeros(K, N, device=DEV)
        for j in range((K + 31) // 32):
            k0 = j * 32
            pos = k0 + int(torch.randint(0, min(32, K - k0), (1,)).item())
            for k in range(k0, min(K, k0 + 32)):
                if k == pos:
                    a[:, k] = na * 2.0 ** -fa[2] if kind == "subwin" else 1.0
                    b[k, :] = 2.0 ** eb
                else:
                    a[:, k] = ma * 2.0 ** xa
                    b[k, :] = 2.0 ** xb
        sgnrow = sgn((M, 1))
        a = a * sgnrow
        big = (na * 2.0 ** -fa[2] if kind == "subwin" else 1.0) * 2.0 ** eb
        mode = torch.randint(0, 4, (M, N), device=DEV)
        c = torch.where(mode == 0, torch.zeros(M, N, device=DEV),
                        torch.where(mode == 1, sgnrow * u * (1 - 2.0 ** -23),
                                    torch.where(mode == 2, -sgnrow * big * (1 - 2.0 ** -23),
                                                sgnrow * big * (1 - 2.0 ** -23))))
    elif kind == "trunc14":
        # sums just above a power of two with bits just below the result's 14-bit unit
        a = torch.full((M, K), 1.0, device=DEV) * torch.where(torch.rand(M, K, device=DEV) < 0.03, -1.0, 1.0)
        b = torch.full((K, N), 1.0, device=DEV)
        c = (1 + torch.randint(1, 2 ** 12, (M, N), device=DEV).float() * 2.0 ** -13) * sgn((M, N))
    elif kind == "bits":
        # every finite value of the formats, uniformly by bit pattern; accumulators over a wide range
        def rbits(shape, f):
            t = FMT[f][0]
            x = torch.randint(0, 256, shape, device=DEV, dtype=torch.uint8).view(t).float()
            return torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        a = rbits((M, K), da); b = rbits((K, N), db)
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-30, 30, (M, N), device=DEV).float())
    else:
        raise ValueError(kind)
    return q8(a, da).contiguous(), q8(b, db).contiguous(), c.contiguous()


KINDS = ["normal", "spread", "accbig", "allones", "onebig", "cancel", "tiny", "huge", "sub", "subwin",
         "winworst", "trunc14", "bits"]


# ------------------------------------------------------------------------------------------------ parts

class Agg:
    def __init__(self):
        self.n = 0; self.room = 0; self.viol = 0; self.nonfin = 0; self.inexact = 0
        self.ratio = 0.0; self.ratio_e = 0.0; self.step = 0.0; self.old = 0.0; self.oldviol = 0
        self.worst = None; self.sim_n = 0; self.sim_eq = 0; self.oldkinds = {}

    def add(self, A, B, C, R, da, db, use_acc, cfg, sim=None, sim_ps=None):
        bc = bound_check(A, B, C, R, da, db, use_acc, sim_ps)
        room, fin, exact = bc["room"], bc["fin"], bc["exact"]
        self.n += R.numel(); self.room += int(room.sum())
        self.nonfin += int((room & ~fin).sum())
        ratio = torch.where(room & fin & exact, bc["ratio"], torch.zeros_like(bc["ratio"]))
        # elements float64 cannot hold exactly, and any near the bound: exact rationals
        redo = (room & fin & ~exact) | (ratio > 0.99)
        self.inexact += int((room & fin & ~exact).sum())
        idx = redo.nonzero().tolist()
        top = int(torch.argmax(ratio).item())
        i0, n0 = divmod(top, R.shape[1])
        if [i0, n0] not in idx:
            idx.append([i0, n0])
        exS = {}
        for i, n in idx:
            cv = float(C[i, n]) if use_acc else 0.0
            rq, req, S = exact_ratio(A[i].tolist(), B[:, n].tolist(), cv, float(R[i, n]), da, db)
            ratio[i, n] = float(rq)
            exS[(i, n)] = S
            if rq > 1:
                self.viol += 1
        rmax = float(ratio.max().item())
        if rmax >= self.ratio:
            t = int(torch.argmax(ratio).item())
            i, n = divmod(t, R.shape[1])
            self.ratio = rmax
            S = exS.get((i, n), bc["S"][i, n].item())
            self.worst = (cfg, i, n, float(C[i, n]) if use_acc else 0.0, float(R[i, n]), float(S))
        self.ratio_e = max(self.ratio_e, float(torch.where(room & fin & exact, bc["ratio_e"], torch.zeros_like(R)).max()))
        self.step = max(self.step, float(torch.where(room & exact, bc["step"], torch.zeros_like(R)).max()))
        old = torch.where(room & fin & exact, bc["old"], torch.zeros_like(R))
        self.old = max(self.old, float(old.max())); self.oldviol += int((old > 1).sum())
        nold = int((old > 1).sum())
        if nold and len(cfg) > 8:
            self.oldkinds[cfg[8]] = self.oldkinds.get(cfg[8], 0) + nold
        if sim is not None:
            self.sim_n += R.numel(); self.sim_eq += int((sim == R).sum())


def part_a(out):
    out("## Part A: the window (one output element, one chunk)\n")
    out("Each row: one element of `tl.dot(a, b, acc)` with K = 32 on wgmma (64 x 64 block, 4 warps); the products"
        " and the accumulator as listed, the rest zero. `model`: the bit-level model (module docstring). u is the"
        " window's unit 2^(E-13).\n")
    out("| operands | accumulator | products | exact | device | model | what it shows |")
    out("|---|---|---|---|---|---|---|")
    M = 64
    cases = []
    e5, e4 = "e5", "e4"

    def add(fa, fb, acc, prs, what):
        cases.append((fa, fb, acc, prs, what))

    add(e5, e5, 0.0, [(1.0, 1.0), (2.0 ** -13, 1.0)], "largest term 1: 2^-13 = u kept")
    add(e5, e5, 0.0, [(1.0, 1.0), (2.0 ** -14, 1.0)], "2^-14 < u dropped")
    add(e5, e5, 0.0, [(1.0, 1.0), (1.5 * 2.0 ** -13, 1.0)], "1.5 u truncated to u")
    add(e5, e5, 0.0, [(1.0, 1.0), (-1.5 * 2.0 ** -13, 1.0)], "-1.5 u truncated toward zero, to -u")
    add(e5, e5, 0.0, [(1.0, 1.0)] + [(2.0 ** -14, 1.0)] * 31, "31 terms u/2: each dropped, no group sums")
    add(e5, e5, 0.0, [(1.0, 1.0), (-1.0, 1.0), (2.0 ** -16, 1.0)], "one window for the chunk: 1 - 1 + 2^-16 loses 2^-16")
    add(e5, e5, 1.0, [(-1.0, 1.0), (2.0 ** -16, 1.0)], "the accumulator is in the same window")
    add(e5, e5, 0.0, [(1.75, 1.75), (2.0 ** -12, 1.0)], "1.75*1.75: e = 0, u = 2^-13; 2^-12 kept")
    add(e5, e5, 0.0, [(1.75, 1.75), (2.0 ** -13, 1.0)], "kept in the window, then cut by the result's 14 bits")
    add(e5, e5, 0.0, [(1.0, 1.0), (1.0, 1.0), (2.0 ** -13, 1.0)], "result 2 + 2^-13: 14 bits keep down to 2^-12")
    add(e5, e5, 0.0, [(1.0, 1.0)] * 4 + [(2.0 ** -11, 1.0)], "result 4 + 2^-11: kept")
    add(e5, e5, 0.0, [(1.0, 1.0)] * 4 + [(2.0 ** -12, 1.0)], "result 4 + 2^-12: cut")
    add(e5, e5, 1.0 + 2.0 ** -14, [], "the accumulator alone is cut to 14 bits")
    add(e5, e5, -(1.0 + 2.0 ** -13 + 2.0 ** -14), [], "toward zero for a negative accumulator")
    add(e5, e5, 3.0, [(2.0 ** -12, 1.0)], "accumulator 3 (e = 1): u = 2^-12, kept")
    add(e5, e5, 3.0, [(2.0 ** -13, 1.0)], "2^-13 < u dropped")
    add(e5, e5, 0.0, [(2.0 ** -16, 2.0 ** 15), (1.75 * 2.0 ** -13, 1.0)], "subnormal 2^-16 counts as 2^-14: e = 1, u = 2^-12")
    add(e5, e5, 0.0, [(2.0 ** -1, 1.0), (1.75 * 2.0 ** -13, 1.0)], "the same product 2^-1 from normals: u = 2^-14")
    add(e4, e4, 0.0, [(2.0 ** -9, 2.0 ** 8), (1.875 * 2.0 ** -6, 2.0 ** -6)], "e4m3 subnormal 2^-9 counts as 2^-6: u = 2^-11")
    add(e4, e4, 0.0, [(2.0 ** -1, 1.0), (1.875 * 2.0 ** -6, 2.0 ** -6)], "the same product from normals: u = 2^-14")
    add(e4, e5, 0.0, [(2.0 ** -9, 2.0 ** 8), (1.875 * 2.0 ** -6, 2.0 ** -6)], "e4m3 x e5m2, subnormal e4m3 operand")
    add(e5, e4, 0.0, [(2.0 ** -16, 2.0 ** 8), (1.75 * 2.0 ** -14, 2.0 ** -6)], "e5m2 x e4m3, subnormal e5m2 operand: u = 2^-19")
    add(e5, e5, 2.0 ** -127, [], "subnormal accumulator alone: e = -126, u = 2^-139, kept")
    add(e5, e5, 2.0 ** -149, [], "2^-149 < u = 2^-139: dropped")
    add(e5, e5, (2.0 ** 23 - 1) * 2.0 ** -149, [], "largest subnormal: cut to a multiple of 2^-139")
    add(e5, e5, 2.0 ** -126 * (1 + 2.0 ** -20), [], "smallest normal and a bit: 14 bits")
    add(e5, e5, 2.0 ** -127, [(1.0, 1.0), (-1.0, 1.0)], "subnormal accumulator against products: dropped")
    groups = {}
    for cs in cases:
        groups.setdefault((cs[0], cs[1]), []).append(cs)
    allok = True
    for (fa, fb), cc in groups.items():
        for s in range(0, len(cc), M):
            part = cc[s:s + M]
            a = torch.zeros(M, 32, device=DEV); b = torch.zeros(32, M, device=DEV); c = torch.zeros(M, M, device=DEV)
            for i, (_, _, acc, prs, _) in enumerate(part):
                c[i, i] = acc
                for k, (x, y) in enumerate(prs):
                    a[i, k] = x; b[k, i] = y
            o, path = run_dot(a, b, c, fa, fb, 4)
            assert path == "wgmma"
            for i, (_, _, acc, prs, what) in enumerate(part):
                accv = c[i, i].item()
                ex = Fr(accv) + sum(Fr(x) * Fr(y) for x, y in prs)
                pe = []
                for x, y in prs:
                    ex_ = max(math.frexp(abs(x))[1] - 1, FMT[fa][3]) + max(math.frexp(abs(y))[1] - 1, FMT[fb][3]) if x * y != 0 else 0
                    pe.append((x * y, ex_))
                mv = model_py(accv, pe)
                dv = o[i, i].item()
                allok &= (mv == dv)
                prs_s = ", ".join(f"{x!r}*{y!r}" for x, y in prs) if prs else "none"
                out(f"| {NAME[fa]} x {NAME[fb]} | {accv!r} | {prs_s} | {float(ex)!r} | {dv!r} | {mv!r} | {what} |")
    out(f"\nEvery device result equal to the model: {allok}\n")


def part_a2(out):
    out("## Part A2: the mma.sync path (one output element)\n")
    out("Blocks with M < 64 or fewer than 4 warps: Triton converts each fp8 operand to float16 (`cvt.rn.f16x2.e5m2x2`,"
        " `cvt.rn.f16x2.e4m3x2`, exact) and runs `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, two per chunk"
        " of 32 (the PTX of each case below). Rows: K = 32, 16 x 16 block, 1 warp.\n")
    out("| operands | accumulator | products | exact | device | what it shows |")
    out("|---|---|---|---|---|---|")
    M = 16
    cases = []
    for accv in [2.0 ** -149, 2.0 ** -130 * 1.375, (2.0 ** 23 - 1) * 2.0 ** -149, 2.0 ** -126 * (1 + 2.0 ** -20),
                 1 + 2.0 ** -23, -(1 + 2.0 ** -22 + 2.0 ** -23)]:
        cases.append(("e5", "e5", accv, [], "the accumulator alone passes unchanged (subnormals included)"))
    for t in (22, 23, 24):
        cases.append(("e5", "e5", 0.0, [(1.0, 1.0), (2.0 ** -12, 2.0 ** (12 - t))], f"a term 2^-{t} against a term 1"))
    for sexp in (25, 26):
        cases.append(("e5", "e5", 2.0 ** sexp, [(1.0, 1.0)] * 16, f"16 terms 1 against an accumulator 2^{sexp}"))
    cases.append(("e4", "e4", 0.0, [(2.0 ** -9, 2.0 ** -9), (1.0, 1.0)], "the smallest e4m3 product 2^-18 against 1 (kept)"))
    cases.append(("e5", "e5", 0.0, [(2.0 ** -16, 2.0 ** -16)], "the smallest e5m2 product 2^-32 alone (no flush)"))
    for (fa, fb) in sorted({(c[0], c[1]) for c in cases}):
        part = [c for c in cases if (c[0], c[1]) == (fa, fb)]
        a = torch.zeros(M, 32, device=DEV); b = torch.zeros(32, M, device=DEV); c = torch.zeros(M, M, device=DEV)
        for i, (_, _, acc, prs, _) in enumerate(part):
            c[i, i] = acc
            for k, (x, y) in enumerate(prs):
                a[i, k] = x; b[k, i] = y
        a8 = a.to(FMT[fa][0]); b8 = b.to(FMT[fb][0])
        o = torch.empty(M, M, device=DEV)
        h = dotk[(1,)](a8, b8, c, o, M, M, 32, True, False, False, FMT[fa][1], num_warps=1)
        ptx = h.asm["ptx"]
        assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx and "wgmma" not in ptx
        for i, (_, _, acc, prs, what) in enumerate(part):
            accv = c[i, i].item()
            ex = Fr(accv) + sum(Fr(x) * Fr(y) for x, y in prs)
            prs_s = ", ".join(f"{x!r}*{y!r}" for x, y in prs[:3]) + (f", ... ({len(prs)})" if len(prs) > 3 else "") if prs else "none"
            out(f"| {NAME[fa]} x {NAME[fb]} | {accv!r} | {prs_s} | {float(ex)!r} | {o[i, i].item()!r} | {what} |")
    out("")


def ks_for(M, N):
    if QUICK:
        return [32, 256]
    return [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] if M * N <= 32 * 32 else [32, 64, 128, 256, 512, 1024, 2048]


SHAPES = [(16, 16, 1), (32, 32, 2), (16, 64, 4), (64, 64, 4), (128, 64, 4), (64, 128, 8), (128, 128, 8)]
if QUICK:
    SHAPES = [(16, 16, 1), (64, 64, 4)]


def part_bc(out):
    out("\n## Parts B and C: the model and the facts, every K that compiles\n")
    out("Each row: one operand pair, block shape and K; 13 kinds of inputs (README), two draws each (`bits` on blocks"
        " up to 32 x 32 and K up to 1024 only). `model = device`: elements whose device result equals the model's, bit"
        " for bit (wgmma only). `ratio`: the largest error over the bound the facts imply for the element (the"
        " interval of the per-chunk bound composed over the chunks with the exact partial sums and the worst case of"
        " the unknown partial values; 1 = on the bound). `ratio (closed form)`: over the composed bound e_n of"
        " `Dot8Chunk.err_le` instead. `chunk ratio`: the largest error of one chunk of the model's chain over the"
        " per-chunk bound (wgmma). `relDot8 ratio`: the error over the existing facts' bound (DotE5m2, ...), for"
        " comparison.\n")
    out("| operands | M x N, warps | variant | path | K | elements in range | outside | model = device | ratio | ratio (closed form) | chunk ratio | relDot8 ratio |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|")
    totals = {}
    worst = {}
    oldkinds = {}
    rejected = set()
    configs = []
    for (M, N, nw) in SHAPES:
        for K in ks_for(M, N):
            configs.append((M, N, nw, K, False, False, True))
    # the kernel's shapes (tutorial 06, BLOCK_M = 128, BLOCK_N = 64, num_warps = 4)
    for D in (64, 128, 256):
        configs.append((128, 64, 4, D, False, True, True))    # qk = tl.dot(q, k.T): [128, D] x [D, 64]
        configs.append((128, 64, 4, D, False, True, False))   # the same, tl.dot(q, k.T) with no accumulator
        configs.append((128, D, 4, 64, True, False, True))    # acc = tl.dot(p, v, acc): [128, 64] x [64, D]
    for da, db in PAIRS:
        for (M, N, nw, K, a_reg, b_t, use_acc) in configs:
            variant = ("a in registers" if a_reg else "") + ("b transposed" if b_t else "") + ("" if use_acc else ", no acc")
            variant = variant.strip(", ") or "plain"
            if (da, db, M, N, nw, a_reg, b_t, use_acc) in rejected:
                continue        # a larger K than one rejected on the host
            agg = Agg()
            path = None
            try:
                for kind in KINDS:
                    if kind == "bits" and (M * N > 32 * 32 or K > 1024):
                        continue
                    for rep in range(2):
                        A, B, C = gen(kind, M, N, K, da, db)
                        R, path = run_dot(A, B, C, da, db, nw, use_acc, a_reg, b_t)
                        Ad, Bd, Cd, Rd = A.double(), B.double(), C.double(), R.double()
                        sim = sps = None
                        if path == "wgmma":
                            sim, sps = model(Ad, Bd, Cd, da, db, use_acc, partials=True)
                        agg.add(Ad, Bd, Cd, Rd, da, db, use_acc, (NAME[da], NAME[db], M, N, nw, variant, path, K, kind),
                                sim, sps)
            except triton.runtime.errors.OutOfResources as e:
                out(f"| {NAME[da]} x {NAME[db]} | {M} x {N}, {nw} | {variant} | rejected | {K} | (out of resources) | | | | | | |")
                rejected.add((da, db, M, N, nw, a_reg, b_t, use_acc))
                continue
            simtxt = f"{agg.sim_eq}/{agg.sim_n}" if path == "wgmma" else "-"
            steptxt = f"{agg.step:.4f}" if path == "wgmma" else "-"
            out(f"| {NAME[da]} x {NAME[db]} | {M} x {N}, {nw} | {variant} | {path} | {K} | {agg.room} | {agg.viol + agg.nonfin} | {simtxt} | {agg.ratio:.4f} | {agg.ratio_e:.4f} | {steptxt} | {agg.old:.4f} |")
            key = (da, db, path)
            t = totals.setdefault(key, dict(room=0, viol=0, sim_n=0, sim_eq=0, ratio=0.0, step=0.0, inexact=0, oldviol=0, old=0.0))
            t["room"] += agg.room; t["viol"] += agg.viol + agg.nonfin; t["sim_n"] += agg.sim_n; t["sim_eq"] += agg.sim_eq
            t["inexact"] += agg.inexact; t["oldviol"] += agg.oldviol; t["old"] = max(t["old"], agg.old)
            if agg.ratio >= t["ratio"]:
                t["ratio"] = agg.ratio; worst[key] = agg.worst
            t["step"] = max(t["step"], agg.step)
            for kd, nk in agg.oldkinds.items():
                oldkinds[(da, db, path, kd)] = oldkinds.get((da, db, path, kd), 0) + nk
    out("\n### Totals\n")
    out("| operands | path | elements in range | outside the bound | model = device | largest ratio | largest chunk ratio | elements recomputed exactly (float64 not exact) | relDot8: elements outside | relDot8: largest ratio |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for (da, db, path), t in sorted(totals.items()):
        sim = f"{t['sim_eq']}/{t['sim_n']}" if path == "wgmma" else "-"
        out(f"| {NAME[da]} x {NAME[db]} | {path} | {t['room']} | {t['viol']} | {sim} | {t['ratio']:.4f} | {t['step']:.4f} | {t['inexact']} | {t['oldviol']} | {t['old']:.4f} |")
    out("\n### Elements outside the existing facts' bound (relDot8), by kind of input\n")
    out("| operands | path | kind | elements |")
    out("|---|---|---|---|")
    for (da, db, path, kd), nk in sorted(oldkinds.items()):
        out(f"| {NAME[da]} x {NAME[db]} | {path} | `{kd}` | {nk} |")
    if not oldkinds:
        out("| (none) | | | |")
    out("\n### The inputs with the largest ratio\n")
    for key, w in sorted(worst.items()):
        if w is None:
            continue
        cfg, i, n, c, r, S = w
        out(f"- {cfg[0]} x {cfg[1]}, {cfg[6]}: {cfg[2]} x {cfg[3]}, {cfg[4]} warps, {cfg[5]}, K = {cfg[7]}, kind `{cfg[8]}`,"
            f" element ({i}, {n}): accumulator {c!r}, exact result {S!r}, device {r!r}")


def part_d(out):
    out("\n## Part D: chains of tl.dot(p, v, acc)\n")
    out("`for t in range(T): [acc = acc * alpha_t]; acc = tl.dot(p_t.to(fp8), v_t, acc)`, p_t [M, 64] converted in"
        " registers, v_t [64, N]; every acc_t stored. `per dot`: each dot from the stored accumulator before it (times"
        " alpha_t, in float32) against the facts and the model. `whole chain` (no rescale): the final result against"
        " the facts composed over all T x 2 chunks from the initial accumulator, and against the model run over the"
        " concatenated operands.\n")
    out("| operands | M x N, warps | path | p, v | rescale | T | elements | per dot: model = device | per dot: ratio | whole chain: model = device | whole chain: ratio | whole chain: relDot8 ratio |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|")
    KC = 64
    tot_viol = 0
    for da, db in PAIRS:
        for (M, N, nw) in ([(128, 64, 4), (32, 64, 2)] if QUICK else [(128, 64, 4), (128, 128, 4), (128, 256, 4), (32, 64, 2)]):
            for pk in ["softmax", "allones", "subwin"]:
                for scale in (False, True):
                    for T in ([1, 4] if QUICK else [1, 4, 16, 64, 256] if not scale else [16, 64]):
                        if pk != "softmax" and T > 64:
                            continue
                        P = torch.zeros(T, M, KC, device=DEV); V = torch.zeros(T, KC, N, device=DEV)
                        for t in range(T):
                            if pk == "softmax":
                                sc = torch.randn(M, KC, device=DEV) * [1.0, 4.0, 16.0][t % 3]
                                p = torch.exp2(sc - sc.amax(1, keepdim=True))
                                v = torch.randn(KC, N, device=DEV)
                            elif pk == "allones":
                                p = torch.full((M, KC), FMT[da][5], device=DEV) * torch.exp2(-torch.randint(0, 3, (M, KC), device=DEV).float())
                                v = torch.full((KC, N), FMT[db][5], device=DEV)
                            else:
                                p, v, _ = gen("subwin", M, N, KC, da, db)
                            P[t] = q8(p, da); V[t] = q8(v, db)
                        c0 = torch.randn(M, N, device=DEV) * (0 if pk == "subwin" else 1)
                        al = (torch.rand(T, device=DEV) * 0.75 + 0.25) if scale else torch.ones(T, device=DEV)
                        accs = torch.empty(T, M, N, device=DEV)
                        o = torch.empty(M, N, device=DEV)
                        V8 = V.to(FMT[db][0])
                        h = chain[(1,)](P.contiguous(), V8.contiguous(), c0, al, accs, o, T, M, N, KC, FMT[da][1], scale, True,
                                        num_warps=nw)
                        path = path_of(h)
                        assert torch.equal(o, accs[T - 1])
                        o2 = torch.empty(M, N, device=DEV)
                        chain[(1,)](P.contiguous(), V8.contiguous(), c0, al, accs, o2, T, M, N, KC, FMT[da][1], scale, False,
                                    num_warps=nw)
                        same_nostore = torch.equal(o2, o)
                        # per dot
                        agg = Agg()
                        prev = c0
                        for t in range(T):
                            vc = prev * al[t] if scale else prev
                            Ad, Bd, Cd, Rd = P[t].double(), V[t].double(), vc.double(), accs[t].double()
                            sim = model(Ad, Bd, Cd, da, db) if path == "wgmma" else None
                            agg.add(Ad, Bd, Cd, Rd, da, db, True, ("chain", t), sim)
                            prev = accs[t]
                        whole_txt = whole_sim = whole_old = "-"
                        if not scale:
                            Aall = torch.cat([P[t] for t in range(T)], 1).double()
                            Ball = torch.cat([V[t] for t in range(T)], 0).double()
                            agg2 = Agg()
                            sim2 = model(Aall, Ball, c0.double(), da, db) if path == "wgmma" else None
                            agg2.add(Aall, Ball, c0.double(), o.double(), da, db, True, ("whole",), sim2)
                            whole_txt = f"{agg2.ratio:.4f}" + (" (outside!)" if agg2.viol + agg2.nonfin else "")
                            whole_sim = f"{agg2.sim_eq}/{agg2.sim_n}" if path == "wgmma" else "-"
                            whole_old = f"{agg2.old:.4f}"
                            tot_viol += agg2.viol + agg2.nonfin
                        tot_viol += agg.viol + agg.nonfin
                        simtxt = f"{agg.sim_eq}/{agg.sim_n}" if path == "wgmma" else "-"
                        out(f"| {NAME[da]} x {NAME[db]} | {M} x {N}, {nw} | {path} | {pk} | {'yes' if scale else 'no'} | {T} | {M * N} | {simtxt} | {agg.ratio:.4f} | {whole_sim} | {whole_txt} | {whole_old} |"
                            + ("" if same_nostore else " (result differs without the stores)"))
    out(f"\nElements outside the bound in Part D: {tot_viol}")


def part_e(out):
    out("\n## Part E: the order of the chunks\n")
    out("For chunks x < y and k (all distinct): +B at a random position of chunk x, -B in chunk y, a small product s"
        " in chunk k, with B/s beyond the window (2^14 on wgmma; 2^30, or 2^34 with a float8e4nv operand, on"
        " mma.sync). s survives exactly when chunk k"
        " is accumulated after both x and y (before, it is cut against B; between, too). `agree`: triples whose"
        " outcome is that of increasing k. With chunks of 16 on mma.sync (two instructions per chunk of 32) the"
        " outcome is not that of increasing k: Triton's fp8-to-fp16 register layout interleaves k within each chunk"
        " of 32.\n")
    out("| operands | M x N, warps | path | chunk | chunks | triples | agree with increasing k |")
    out("|---|---|---|---|---|---|---|")
    for da, db in PAIRS:
        for (M, N, nw, pth) in [(64, 64, 4, "wgmma"), (16, 16, 1, "mma.sync")]:
            for nch, CH in [(8, 32), (32, 32), (8, 16)]:
                if CH == 16 and pth == "wgmma":
                    continue
                # B = Ba * Bb, s = Sa * Sb, every factor held by its format: B / s = 2^14 on wgmma, at least
                # 2^30 on mma.sync
                if pth == "wgmma":
                    (Ba, Bb), (Sa, Sb) = (2.0 ** 8, 2.0 ** 6), (1.0, 1.0)
                elif (da, db) == ("e5", "e5"):
                    (Ba, Bb), (Sa, Sb) = (2.0 ** 13, 2.0 ** 13), (2.0 ** -2, 2.0 ** -2)
                else:
                    (Ba, Bb), (Sa, Sb) = (2.0 ** 8, 2.0 ** 8), (2.0 ** -9, 2.0 ** -9)
                K = nch * CH
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
                        o1, o2, o3 = random.sample(range(CH), 3)
                        a[i, x * CH + o1] = Ba; b[x * CH + o1, i] = Bb
                        a[i, y * CH + o2] = -Ba; b[y * CH + o2, i] = Bb
                        a[i, k * CH + o3] = Sa; b[k * CH + o3, i] = Sb
                    o, path = run_dot(a, b, c, da, db, nw)
                    for i, (x, y, k) in enumerate(tt):
                        agree += int((o[i, i].item() != 0.0) == (k > max(x, y)))
                out(f"| {NAME[da]} x {NAME[db]} | {M} x {N}, {nw} | {path} | {CH} | {nch} | {len(triples)} | {agree} |")


def part_f(out):
    out("\n## Part F: the existing facts (relDot8) on a subnormal operand\n")
    out("The window's raw exponent counts a subnormal fp8 operand at the format's smallest normal, so a product with"
        " a subnormal operand can set a unit u above 2^-13 of every term. One chunk: that product, 31 products just"
        " below u (lost whole), and an accumulator just below u (lost). The bound of DotE4m3, DotE5m2, ... is"
        " relDot8(K) (|acc| + sum |a b|) + absDot8(K); the new facts' bound uses the lifted products.\n")
    out("| operands | path | K | accumulator | exact | device | error | relDot8 bound | error / relDot8 bound | error / new facts' bound |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for (da, db, sub, big, small) in [
            ("e5", "e5", (2.0 ** -16, 2.0 ** 15), None, (1.75 * 2.0 ** -13, 1.0)),
            ("e4", "e4", (2.0 ** -9, 2.0 ** 8), None, (1.875 * 2.0 ** -6, 2.0 ** -6)),
            ("e4", "e5", (2.0 ** -9, 2.0 ** 8), None, (1.875 * 2.0 ** -6, 2.0 ** -6)),
            ("e5", "e4", (2.0 ** -16, 2.0 ** 8), None, (1.75 * 2.0 ** -14, 2.0 ** -6))]:
        E = FMT[da][3] + int(math.log2(sub[1]))
        u = 2.0 ** (E - 13)
        for K in (32, 64, 256):
            for accv in (0.0, u * (1 - 2.0 ** -23)):
                for (M, N, nw) in [(64, 64, 4), (16, 16, 1)]:
                    a = torch.zeros(M, K, device=DEV); b = torch.zeros(K, N, device=DEV); c = torch.zeros(M, N, device=DEV)
                    for j in range(K // 32):
                        a[0, j * 32] = sub[0]; b[j * 32, 0] = sub[1]
                        for k in range(j * 32 + 1, j * 32 + 32):
                            a[0, k] = small[0]; b[k, 0] = small[1]
                    c[0, 0] = accv
                    o, path = run_dot(a, b, c, da, db, nw)
                    accq = c[0, 0].item()
                    prs = [(a[0, k].item(), b[k, 0].item()) for k in range(K)]
                    ex = Fr(accq) + sum(Fr(x) * Fr(y) for x, y in prs)
                    mag = abs(Fr(accq)) + sum(abs(Fr(x) * Fr(y)) for x, y in prs)
                    n = K // 32
                    rel8 = (1 + Fr(1, 128)) ** n - 1
                    abs8 = n * (1 + Fr(1, 128)) ** n * Fr(2) ** -126
                    r = o[0, 0].item()
                    err = abs(Fr(r) - ex)
                    rq, _, _ = exact_ratio([x for x, _ in prs], [y for _, y in prs], accq, r, da, db)
                    out(f"| {NAME[da]} x {NAME[db]} | {path} | {K} | {accq!r} | {float(ex)!r} | {r!r} | {float(err):.6g} | {float(rel8 * mag + abs8):.6g} | {float(err / (rel8 * mag + abs8)):.4f} | {float(rq):.4f} |")


def main():
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    part_a(out)
    part_a2(out)
    part_e(out)
    part_f(out)
    part_d(out)
    part_bc(out)
    out(f"\n(run time {time.time() - t0:.0f} s)")
    with open("evidence/num-dot-chunk/" + ("quick.txt" if QUICK else "out.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
