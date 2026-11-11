"""TVA library evidence (library issues L124 and L128, round 26): the fp8 `tl.dot` chunk facts with each chunk's
loss bounded two ways, and `input_precision` on fp8 operands, on sm_90.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-chunk-loss/run.py
Output: evidence/num-dot-chunk-loss/out.txt (README.md says what it shows).

The facts checked (docs/rules/num.md, "`tl.dot` of fp8, chunk by chunk"): `DotE4m3ChunkLoss`, `DotE5m2ChunkLoss`,
`DotE4m3E5m2ChunkLoss`, `DotE5m2E4m3ChunkLoss` (`DotChunkLoss` with `rhoChunk8`, `sigmaChunk8`, `alphaChunk8`,
`kappaChunk8`, `rhoLossChunk8`). Per output element, with K products in chunks of 32 consecutive k (chunk j:
32j <= k < 32j + 32), there are partial values s_0 = acc, ..., s_n = result (n = ceil(K/32)) such that each step is
within

    min(rho * M_j, kappa * A_j + rho2 * M_j) + sigma * |s_j + c_j| + alpha,
    rho = 2^-7, kappa = 2, rho2 = 2^-12, sigma = 2^-12, alpha = 2^-126,

of s_j + c_j, where c_j is the chunk's exact sum, A_j = sum_chunk |a_k b_k| its exact magnitude (not lifted), and
M_j any bound on |s_j| and on every lifted product L(a_k) L(b_k) of the chunk (L(x) = 0 for x = 0, else
max(|x|, 2^emin), the format's smallest normal). The first arm alone is the round-24 facts `DotE4m3Chunk`, ...
(`DotChunkAcc`); the new facts state both arms of the same partial values.

The derivation (README, "Derivation"): by the bit-level model of a wgmma chunk (evidence/num-dot-chunk), a product
truncated toward zero to a multiple of the window's unit u loses at most min(|p|, u), the accumulator less than u,
and the cut to 14 bits less than 2^-13 of the truncated sum; so a chunk's error is below
(1 + 2^-13) * (A_j + u) + 2^-13 * |s_j + c_j|, with u <= 2^-13 * M_j + 2^-139.

Parts:
  P. input_precision: the PTX of every value against the default's (and against a kernel that omits the argument),
     and the results bit for bit, all four operand pairs, both instruction paths (library issue L128).
  S. Single chunks (K = 32) on the device, both paths: s_0 = acc and s_1 = the result, so each element is checked
     against the step bound directly, with no model.
  B. Every K that compiles on four block shapes: on wgmma the model against the device bit for bit and each chunk
     of the model's chain against the step bound; on both paths every element against the interval the facts
     imply (the step bound composed over the chunks with the exact partial sums and the worst case of the unknown
     partial values).
  C. Chains of tl.dot(p, v, acc) feeding the accumulator back, as tutorial 06's _attn_fwd does (p converted to fp8
     in registers, with and without a rescale of the accumulator between the dots): each dot as in Part B, and each
     whole chain against the facts composed over all its chunks, with the composed bound's half-width against the
     round-24 facts' (the sharpening).

Arithmetic: float64 tensors. The products of fp8 values are exact in float64; a sum of n terms is within
n * 2^-53 of their magnitudes' sum, and every step bound is at least 2^-12 of M_j >= the largest term, so a step
ratio computed in float64 is within 2^-30 of the exact one, and an interval ratio within 10^-4 for the longest
chains here. Every row's largest element, and any element above 0.9 of a bound, is recomputed in exact rationals.
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

RHO, SIG, ALP, KAP, RHO2 = 2.0 ** -7, 2.0 ** -12, 2.0 ** -126, 2.0, 2.0 ** -12
RHO_Q, SIG_Q, ALP_Q, KAP_Q, RHO2_Q = Fr(1, 2 ** 7), Fr(1, 2 ** 12), Fr(1, 2 ** 126), Fr(2), Fr(1, 2 ** 12)
FMT = {  # torch dtype, Triton dtype, fraction bits, emin, max value, largest significand, emax
    "e5": (torch.float8_e5m2, tl.float8e5, 2, -14, 57344.0, 1.75, 15),
    "e4": (torch.float8_e4m3fn, tl.float8e4nv, 3, -6, 448.0, 1.875, 8),
}
NAME = {"e5": "float8e5", "e4": "float8e4nv"}
PAIRS = [("e5", "e5"), ("e4", "e4"), ("e4", "e5"), ("e5", "e4")]
F32MAX = 3.4028234663852886e38
QUICK = "--quick" in sys.argv       # a smoke test on a few configurations (not the evidence)


# ----------------------------------------------------------------------------------------------- kernels

@triton.jit
def dotk(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         USE_ACC: tl.constexpr, A_REG: tl.constexpr, B_T: tl.constexpr, ADT: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    if A_REG:
        # `a` arrives as float32 values that fp8 holds exactly and is converted in registers (`p.to(...)`)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :]).to(ADT)
    else:
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    if B_T:
        # `b` stored as [N, K] and transposed in the kernel (`k.T`)
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
def dotp(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, PREC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, c, input_precision=PREC))


@triton.jit
def dotd(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, c))


@triton.jit
def chain(a_ptr, b_ptr, c_ptr, al_ptr, accs_ptr, T, M: tl.constexpr, N: tl.constexpr,
          KC: tl.constexpr, ADT: tl.constexpr, SCALE: tl.constexpr):
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
        tl.store(accs_ptr + t * M * N + rm[:, None] * N + rn[None, :], acc)


def ptx_code(h, name):
    """The kernel's PTX without its debug information (`.loc`, `.file` and the `.debug` sections, which name source
    lines) and with the entry name replaced, for comparing two compilations."""
    keep = []
    for line in h.asm["ptx"].splitlines():
        t = line.strip()
        if t.startswith(".section") and ".debug" in t:
            break
        if t.startswith(".loc") or t.startswith(".file"):
            continue
        keep.append(line.replace(name, "KERNEL"))
    return "\n".join(keep)


def path_of(h):
    ptx = h.asm["ptx"]
    return "wgmma" if "wgmma" in ptx else ("mma.sync" if "mma.sync" in ptx else "?")


def run_dot(a, b, c, da, db, nw, use_acc=True, a_reg=False, b_t=False):
    """a [M,K], b [K,N] float32 tensors of values the formats hold exactly; c [M,N] float32."""
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


BIG = 1 << 20


def model(A, B, C, da, db):
    """The wgmma model of evidence/num-dot-chunk on float64 tensors (exact: each truncated term is an integer
    multiple of u below 2^17 u, and 33 of them sum below 2^23 u). Returns the partial values s_0, ..., s_n."""
    K = A.shape[1]
    ea = rawexp8(A, da)
    eb = rawexp8(B, db)
    s = C.clone()
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
        S1 = (torch.trunc(s / u) + torch.trunc(P / u[..., None]).sum(-1)) * u
        S1 = torch.where(E <= -BIG // 2, torch.zeros_like(S1), S1)
        er = torch.frexp(S1.abs())[1].to(torch.int64) - 1
        u2 = torch.ldexp(torch.ones_like(s), er - 13)
        s = torch.where(S1 == 0, S1, torch.trunc(S1 / u2) * u2)
        ps.append(s)
    return ps


# --------------------------------------------------------------------------------------------- bounds

def lift(X, f):
    return torch.where(X != 0, torch.clamp(X.abs(), min=2.0 ** FMT[f][3]), torch.zeros_like(X))


def chunk_data(A, B, da, db):
    """Per element and chunk: the exact sum c_j, the magnitude A_j = sum |a b| and the largest lifted product m_j;
    and |acc|-free total magnitude sum |a b|."""
    K = A.shape[1]
    La, Lb = lift(A, da), lift(B, db)
    cs, As, ms = [], [], []
    for k0 in range(0, K, 32):
        k1 = min(K, k0 + 32)
        P = A[:, None, k0:k1] * B[k0:k1, :].T[None, :, :]
        cs.append(P.sum(-1))
        As.append(P.abs().sum(-1))
        ms.append((La[:, None, k0:k1] * Lb[k0:k1, :].T[None, :, :]).amax(-1))
    return cs, As, ms


def step3(s0, c, a, m):
    """The step bounds at the smallest M_j = max(|s0|, m): (round-24 arm, loss arm, the facts' minimum)."""
    Mj = torch.maximum(s0.abs(), m)
    tail = SIG * (s0 + c).abs() + ALP
    win = RHO * Mj
    loss = KAP * a + RHO2 * Mj
    return win + tail, loss + tail, torch.minimum(win, loss) + tail


def intervals(C0, cs, As, ms):
    """The facts' implied interval, composed over the chunks with the exact partial sums S_j and the worst case of
    the unknown partial values: the step s -> s + c + B(s) is increasing in s (B moves by at most rho + sigma < 1
    per unit of s), so the upper end follows the upper end. Returns S_n and the half-widths (above, below) of the
    new facts' interval and of the round-24 facts'."""
    S = C0.clone()
    z = torch.zeros_like(S)
    up_n, dn_n, up_o, dn_o = z.clone(), z.clone(), z.clone(), z.clone()
    for c, a, m in zip(cs, As, ms):
        S1 = S + c
        for (d, sg, new) in ((up_n, 1, True), (dn_n, -1, True), (up_o, 1, False), (dn_o, -1, False)):
            Mj = torch.maximum((S + sg * d).abs(), m)
            arm = RHO * Mj
            if new:
                arm = torch.minimum(arm, KAP * a + RHO2 * Mj)
            d += arm + SIG * (S1 + sg * d).abs() + ALP
        S = S1
    return S, up_n, dn_n, up_o, dn_o


def side_ratio(err, up, dn):
    return torch.where(err >= 0, err / up, -err / dn)


def exact_interval(arow, bcol, c, r, da, db):
    """The same intervals in exact rationals for one element: (new ratio, old ratio, half-width new / old)."""
    S = Fr(c)
    un = dn = uo = do = Fr(0)
    K = len(arow)
    la = [Fr(0) if x == 0 else max(abs(Fr(x)), Fr(2) ** FMT[da][3]) for x in arow]
    lb = [Fr(0) if y == 0 else max(abs(Fr(y)), Fr(2) ** FMT[db][3]) for y in bcol]
    for k0 in range(0, K, 32):
        ks = range(k0, min(K, k0 + 32))
        cj = sum((Fr(arow[k]) * Fr(bcol[k]) for k in ks), Fr(0))
        aj = sum((abs(Fr(arow[k]) * Fr(bcol[k])) for k in ks), Fr(0))
        mj = max(la[k] * lb[k] for k in ks)
        S1 = S + cj
        vals = []
        for (d, sg, new) in ((un, 1, True), (dn, -1, True), (uo, 1, False), (do, -1, False)):
            Mj = max(abs(S + sg * d), mj)
            arm = RHO_Q * Mj
            if new:
                arm = min(arm, KAP_Q * aj + RHO2_Q * Mj)
            vals.append(d + arm + SIG_Q * abs(S1 + sg * d) + ALP_Q)
        un, dn, uo, do = vals
        S = S1
    err = Fr(r) - S
    rn = err / un if err >= 0 else -err / dn
    ro = err / uo if err >= 0 else -err / do
    return float(rn), float(ro), float(max(un, dn) / max(uo, do)), S


def exact_steps(arow, bcol, ps, da, db):
    """The largest step ratios of a chain of partial values (python floats) in exact rationals."""
    K = len(arow)
    la = [Fr(0) if x == 0 else max(abs(Fr(x)), Fr(2) ** FMT[da][3]) for x in arow]
    lb = [Fr(0) if y == 0 else max(abs(Fr(y)), Fr(2) ** FMT[db][3]) for y in bcol]
    wn = wl = wo = 0.0
    for j, k0 in enumerate(range(0, K, 32)):
        ks = range(k0, min(K, k0 + 32))
        cj = sum((Fr(arow[k]) * Fr(bcol[k]) for k in ks), Fr(0))
        aj = sum((abs(Fr(arow[k]) * Fr(bcol[k])) for k in ks), Fr(0))
        mj = max(la[k] * lb[k] for k in ks)
        s0, s1 = Fr(ps[j]), Fr(ps[j + 1])
        Mj = max(abs(s0), mj)
        tail = SIG_Q * abs(s0 + cj) + ALP_Q
        e = abs(s1 - (s0 + cj))
        wo = max(wo, float(e / (RHO_Q * Mj + tail)))
        wl = max(wl, float(e / (KAP_Q * aj + RHO2_Q * Mj + tail)))
        wn = max(wn, float(e / (min(RHO_Q * Mj, KAP_Q * aj + RHO2_Q * Mj) + tail)))
    return wn, wl, wo


# ------------------------------------------------------------------------------------------- inputs

def q8(x, f):
    """Round float32 values to the format (saturating), back to float32."""
    mx = FMT[f][4]
    return x.clamp(-mx, mx).to(FMT[f][0]).float()


def sgn(shape):
    return torch.where(torch.rand(shape, device=DEV) < 0.5, -1.0, 1.0)


def small_pair(da, db, t):
    """Normal operands (a, b) with a * b = ma * 2^t (ma the largest significand of da), b a power of two."""
    xb = max(FMT[db][3], min(FMT[db][6], t // 2))
    xa = t - xb
    assert FMT[da][3] <= xa <= FMT[da][6], (da, db, t)
    return FMT[da][5] * 2.0 ** xa, 2.0 ** xb


def gen_acc_window(kind, M, N, K, da, db):
    """The accumulator sets each element's window (field e, unit u = 2^(e-13)); every product is ma * 2^(e-14), just
    below u, with normal operands (so its field is below e). `accone`: the accumulator (2 - 2^-23) 2^e, every product
    of its sign; `lossacc`: the accumulator 2^e + u - 2^(e-23) (bits below u all set, magnitude near 2^e, so M_j is
    smallest against u), every product of its sign; `lossmix`: as `lossacc` with random signs and a quarter of the
    products zero. Each row of `a` has its own e; `b` is shared, so `b` is a power of two per k and the row's scale
    is in `a`."""
    a = torch.zeros(M, K, device=DEV)
    c = torch.zeros(M, N, device=DEV)
    xb = -3
    b = torch.full((K, N), 2.0 ** xb, device=DEV)
    for i in range(M):
        # e in [5, 18]: a = ma * 2^(e-11) is a normal value of both formats, b = 2^-3, so the product's field is
        # e - 14, below the accumulator's
        e = random.randint(5, 18)
        u = 2.0 ** (e - 13)
        av = FMT[da][5] * 2.0 ** (e - 14 - xb)
        s = 1.0 if random.random() < 0.5 else -1.0
        if kind == "accone":
            a[i, :] = s * av
            c[i, :] = s * (2 - 2.0 ** -23) * 2.0 ** e
        elif kind == "lossacc":
            a[i, :] = s * av
            c[i, :] = s * (2.0 ** e + u - 2.0 ** (e - 23))
        else:
            a[i, :] = av * sgn((K,)) * (torch.rand(K, device=DEV) < 0.75).float()
            c[i, :] = sgn((N,)) * (2.0 ** e + u - 2.0 ** (e - 23))
    return a, b, c


def gen(kind, M, N, K, da, db):
    fa, fb = FMT[da], FMT[db]
    ma, mb = fa[5], fb[5]
    na, nb = 2.0 ** fa[3], 2.0 ** fb[3]
    if kind in ("accone", "lossacc", "lossmix"):
        a, b, c = gen_acc_window(kind, M, N, K, da, db)
    elif kind == "accdom":
        # products just below half the accumulator's window unit, random signs; the accumulator with all bits set
        a, b, c = gen_acc_window("accone", M, N, K, da, db)
        a = q8(a * 0.5, da) * sgn((M, K))
        c = c * sgn((M, N))
    elif kind == "accsmall":
        # products far below the accumulator (the attention regime after many blocks)
        a = torch.randn(M, K, device=DEV); b = torch.randn(K, N, device=DEV)
        c = (torch.rand(M, N, device=DEV) + 1) * torch.exp2(torch.randint(8, 16, (M, N), device=DEV).float()) * sgn((M, N))
    elif kind == "normal":
        a = torch.randn(M, K, device=DEV); b = torch.randn(K, N, device=DEV); c = torch.randn(M, N, device=DEV)
    elif kind == "spread":
        a = torch.randn(M, K, device=DEV) * torch.exp2(torch.randint(-4, 4, (M, K), device=DEV).float())
        b = torch.randn(K, N, device=DEV) * torch.exp2(torch.randint(-4, 4, (K, N), device=DEV).float())
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-8, 8, (M, N), device=DEV).float())
    elif kind == "accbig":
        a = torch.rand(M, K, device=DEV) + 1; b = torch.rand(K, N, device=DEV) + 1
        c = torch.exp2(torch.randint(4, 20, (M, N), device=DEV).float()) * (2 - 2.0 ** -23) * sgn((M, N))
    elif kind == "allones":
        a = torch.full((M, K), ma, device=DEV) * torch.exp2(-torch.randint(0, 4, (M, K), device=DEV).float())
        b = torch.full((K, N), mb, device=DEV)
        c = torch.exp2(torch.randint(0, 9, (M, N), device=DEV).float()) * (2 - 2.0 ** -23) * sgn((M, N))
    elif kind == "onebig":
        t = torch.randint(1, 13, (M, K), device=DEV).float()
        a = ma * torch.exp2(-t)
        pos = torch.randint(0, 32, (M, (K + 31) // 32), device=DEV)
        for j in range(pos.shape[1]):
            a[torch.arange(M, device=DEV), j * 32 + pos[:, j]] = ma * 4
        b = torch.full((K, N), mb, device=DEV) * torch.where(torch.rand(K, N, device=DEV) < 0.1, -1.0, 1.0)
        c = torch.zeros(M, N, device=DEV)
    elif kind == "cancel":
        a = q8(torch.randn(M, K, device=DEV), da); b = q8(torch.randn(K, N, device=DEV), db)
        a[:, 1::2] = -a[:, 0::2]
        near = random.random() < 0.5
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
        a = torch.randn(M, K, device=DEV) * na; b = torch.randn(K, N, device=DEV) * nb * 4
        c = torch.randn(M, N, device=DEV) * na * nb * torch.randint(0, 2, (M, N), device=DEV)
    elif kind in ("subwin", "winworst"):
        # per chunk: the largest field on one product (`subwin`: the smallest subnormal times 2^eb; `winworst`:
        # 1 times 2^eb), the other 31 products just below the window's unit u with all fraction bits set, of one
        # sign per row; the accumulator 0, just below u, or near the big product, of either sign
        eb = fb[6] - random.randint(0, 2)
        E = (fa[3] if kind == "subwin" else 0) + eb
        u = 2.0 ** (E - 13)
        xa = min(max(fa[3], E - 14 - fb[6]), fa[6])
        xb = E - 14 - xa
        a = torch.zeros(M, K, device=DEV); b = torch.zeros(K, N, device=DEV)
        for j in range((K + 31) // 32):
            k0 = j * 32
            pos = k0 + random.randrange(min(32, K - k0))
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
        a = torch.full((M, K), 1.0, device=DEV) * torch.where(torch.rand(M, K, device=DEV) < 0.03, -1.0, 1.0)
        b = torch.full((K, N), 1.0, device=DEV)
        c = (1 + torch.randint(1, 2 ** 12, (M, N), device=DEV).float() * 2.0 ** -13) * sgn((M, N))
    elif kind == "bits":
        def rbits(shape, f):
            x = torch.randint(0, 256, shape, device=DEV, dtype=torch.uint8).view(FMT[f][0]).float()
            return torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        a = rbits((M, K), da); b = rbits((K, N), db)
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-30, 30, (M, N), device=DEV).float())
    else:
        raise ValueError(kind)
    c = c.float()   # float32 accumulators (values float32 holds)
    return q8(a, da).contiguous(), q8(b, db).contiguous(), c.contiguous()


KINDS = ["normal", "spread", "accbig", "allones", "onebig", "cancel", "tiny", "huge", "sub", "subwin", "winworst",
         "trunc14", "bits", "accone", "accdom", "accsmall", "lossacc", "lossmix"]


# ------------------------------------------------------------------------------------------- tallies

class Agg:
    """Tallies of one row: elements in range, outside, the largest ratios and where."""

    def __init__(self):
        self.n = self.room = self.out = self.nonfin = self.grow = self.sim_n = self.sim_eq = 0
        self.r_new = self.r_old = self.st_new = self.st_loss = self.st_old = 0.0
        self.w_min = None       # the smallest composed half-width, new over old
        self.worst = None

    def add(self, A, B, C, R, da, db, cfg, ps=None, sim=None):
        """A, B, C, R float64 [M,K], [K,N], [M,N], [M,N] (C the accumulator, zero without one); ps: the model's
        partial values (wgmma), sim: the model's result."""
        K = A.shape[1]
        cs, As, ms = chunk_data(A, B, da, db)
        mag = C.abs() + sum(As)
        n = (K + 31) // 32
        rel8 = (1 + 2.0 ** -7) ** n - 1
        abs8 = n * (1 + 2.0 ** -7) ** n * 2.0 ** -126
        room = mag + rel8 * mag + abs8 <= F32MAX
        fin = torch.isfinite(R)
        self.n += R.numel()
        self.room += int(room.sum())
        self.nonfin += int((room & ~fin).sum())
        ok = room & fin
        S, un, dn, uo, do = intervals(C, cs, As, ms)
        err = R - S
        rn = torch.where(ok, side_ratio(err, un, dn), torch.zeros_like(R))
        ro = torch.where(ok, side_ratio(err, uo, do), torch.zeros_like(R))
        hw = torch.maximum(un, dn) / torch.maximum(uo, do)
        # the device's magnitude against |acc| + sum |a b| (truncation toward zero never increases a magnitude)
        self.grow += int((ok & (R.abs() > mag * (1 + 2.0 ** -40))).sum())
        # exact rationals: the largest element and any above 0.9
        idx = ((rn > 0.9) | (ro > 0.9)).nonzero().tolist()
        top = int(torch.argmax(rn).item())
        i0, n0 = divmod(top, R.shape[1])
        if [i0, n0] not in idx:
            idx.append([i0, n0])
        for i, nn in idx:
            if not bool(ok[i, nn]):
                continue
            q_new, q_old, q_hw, Sx = exact_interval(A[i].tolist(), B[:, nn].tolist(), float(C[i, nn]),
                                                     float(R[i, nn]), da, db)
            rn[i, nn], ro[i, nn] = q_new, q_old
            if q_new > 1 or q_old > 1:
                self.out += 1
            if (i, nn) == (i0, n0) and q_new >= self.r_new:
                self.worst = (cfg, i, nn, float(C[i, nn]), float(R[i, nn]), float(Sx))
        self.r_new = max(self.r_new, float(rn.max()))
        self.r_old = max(self.r_old, float(ro.max()))
        hwv = float(torch.where(ok, hw, torch.full_like(hw, 9.0)).min())
        self.w_min = hwv if self.w_min is None else min(self.w_min, hwv)
        if ps is not None:
            sn = sl = so = torch.zeros_like(R)
            for j in range(n):
                bo, bl, bn = step3(ps[j], cs[j], As[j], ms[j])
                e = (ps[j + 1] - (ps[j] + cs[j])).abs()
                sn = torch.maximum(sn, e / bn); sl = torch.maximum(sl, e / bl); so = torch.maximum(so, e / bo)
            sn = torch.where(ok, sn, torch.zeros_like(sn))
            sl = torch.where(ok, sl, torch.zeros_like(sl))
            so = torch.where(ok, so, torch.zeros_like(so))
            flag = ((sn > 0.9) | (so > 0.9)).nonzero().tolist()
            t = int(torch.argmax(sn).item())
            if [t // R.shape[1], t % R.shape[1]] not in flag:
                flag.append([t // R.shape[1], t % R.shape[1]])
            for i, nn in flag:
                wn, wl, wo = exact_steps(A[i].tolist(), B[:, nn].tolist(), [float(p[i, nn]) for p in ps], da, db)
                sn[i, nn], sl[i, nn], so[i, nn] = wn, wl, wo
                if wn > 1 or wo > 1:
                    self.out += 1
            self.st_new = max(self.st_new, float(sn.max()))
            self.st_loss = max(self.st_loss, float(sl.max()))
            self.st_old = max(self.st_old, float(so.max()))
        if sim is not None:
            self.sim_n += R.numel()
            self.sim_eq += int((sim == R).sum())


def fmt_worst(w):
    if w is None:
        return "-"
    cfg, i, n, c, r, S = w
    return f"{cfg}, element ({i}, {n}): accumulator {c!r}, exact {S!r}, device {r!r}"


# ------------------------------------------------------------------------------------------------ parts

PRECS = [None, "tf32", "tf32x3", "ieee", "bf16x3", "bf16x6"]


def part_p(out):
    out("## Part P: input_precision (library issue L128)\n")
    out("One `tl.dot(a, b, acc, input_precision=PREC)`, K = 64, for each PREC. `PTX = default`: the PTX equals that of"
        " PREC = None (the default); `default = omitted`: that PTX equals the PTX of a kernel that omits the argument"
        " (debug line information and entry names aside); `results`: elements equal to the default's bit for bit, on one draw each of `normal`,"
        " `sub`, `subwin` and `bits` inputs.\n")
    out("| operands | M x N, warps | path | input_precision | PTX = default | default = omitted | results equal |")
    out("|---|---|---|---|---|---|---|")
    allok = True
    for da, db in PAIRS:
        for (M, N, nw) in [(64, 64, 4), (128, 64, 8), (16, 16, 1), (32, 32, 2)]:
            K = 64
            ins = []
            for kind in ("normal", "sub", "subwin", "bits"):
                A, B, C = gen(kind, M, N, K, da, db)
                ins.append((A.to(FMT[da][0]), B.to(FMT[db][0]), C))
            o = torch.empty(M, N, device=DEV)
            hd = dotd[(1,)](ins[0][0], ins[0][1], ins[0][2], o, M, N, K, num_warps=nw)
            ptx_omit = ptx_code(hd, "dotd")
            ref, ref_ptx, path = [], None, None
            for (a8, b8, c) in ins:
                o = torch.empty(M, N, device=DEV)
                h = dotp[(1,)](a8, b8, c, o, M, N, K, None, num_warps=nw)
                ref.append(o)
                ref_ptx = ptx_code(h, "dotp")
                path = path_of(h)
            same_omit = ref_ptx == ptx_omit
            for prec in PRECS[1:]:
                eq = tot = 0
                pt = None
                for (a8, b8, c), r0 in zip(ins, ref):
                    o = torch.empty(M, N, device=DEV)
                    h = dotp[(1,)](a8, b8, c, o, M, N, K, prec, num_warps=nw)
                    pt = ptx_code(h, "dotp")
                    eq += int((o.view(torch.int32) == r0.view(torch.int32)).sum())
                    tot += o.numel()
                ok = (pt == ref_ptx) and same_omit and eq == tot
                allok &= ok
                out(f"| {NAME[da]} x {NAME[db]} | {M} x {N}, {nw} | {path} | {prec} | {pt == ref_ptx} | {same_omit} |"
                    f" {eq}/{tot} |")
    out(f"\nEvery input_precision gives the default's PTX and bits: {allok}\n")


SHAPES_S = [(64, 64, 4, False, False), (128, 64, 8, False, False), (128, 64, 4, False, True), (128, 64, 4, True, False),
            (16, 16, 1, False, False), (32, 32, 2, False, False), (16, 64, 4, False, False)]


def variant_name(a_reg, b_t):
    return ("a in registers" if a_reg else "") + ("b transposed" if b_t else "") or "plain"


def part_s(out, totals):
    out("\n## Part S: single chunks on the device (K = 32)\n")
    out("Each row: one operand pair, block shape and variant, every kind of input (README), two draws each. With one"
        " chunk the partial values are the accumulator and the device's result, so each element is checked against the"
        " step bound with no model, at the smallest M_j (the larger of |acc| and the chunk's largest lifted product)."
        " `new`: the error over the new facts' bound (the minimum of the two arms); `loss arm`: over the second arm"
        " alone, kappa A_j + rho2 M_j + sigma |s + c| + alpha; `round 24`: over the round-24 facts' bound;"
        " `magnitude`: elements whose result exceeds |acc| + sum |a b|.\n")
    out("| operands | M x N, warps | variant | path | kind | elements in range | outside | new | loss arm | round 24 | magnitude |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    for da, db in PAIRS:
        for (M, N, nw, a_reg, b_t) in (SHAPES_S[:2] + SHAPES_S[4:5] if QUICK else SHAPES_S):
            for kind in (KINDS[:3] + KINDS[13:] if QUICK else KINDS):
                agg = Agg()
                path = None
                for rep in range(2):
                    A, B, C = gen(kind, M, N, 32, da, db)
                    R, path = run_dot(A, B, C, da, db, nw, True, a_reg, b_t)
                    Ad, Bd, Cd, Rd = A.double(), B.double(), C.double(), R.double()
                    # the single chunk: the step is the interval; the partial values are the accumulator and result
                    agg.add(Ad, Bd, Cd, Rd, da, db, (NAME[da], NAME[db], M, N, nw, variant_name(a_reg, b_t), path, 32, kind),
                            ps=[Cd, Rd])
                out(f"| {NAME[da]} x {NAME[db]} | {M} x {N}, {nw} | {variant_name(a_reg, b_t)} | {path} | `{kind}` |"
                    f" {agg.room} | {agg.out + agg.nonfin} | {agg.st_new:.4f} | {agg.st_loss:.4f} | {agg.st_old:.4f} |"
                    f" {agg.grow} |")
                key = (da, db, path, "S")
                t = totals.setdefault(key, dict(room=0, out=0, new=0.0, loss=0.0, old=0.0, grow=0, worst=None,
                                                sim_n=0, sim_eq=0, hw=9.0, chunk=0.0))
                t["room"] += agg.room; t["out"] += agg.out + agg.nonfin; t["grow"] += agg.grow
                if agg.st_new >= t["new"]:
                    t["new"] = agg.st_new; t["worst"] = agg.worst
                t["loss"] = max(t["loss"], agg.st_loss); t["old"] = max(t["old"], agg.st_old)
                t["hw"] = min(t["hw"], agg.w_min)


SHAPES_B = [(64, 64, 4, 2048), (128, 128, 8, 512), (16, 16, 1, 2048), (32, 32, 2, 1024)]


def part_b(out, totals):
    out("\n## Part B: every K that compiles\n")
    out("Each row: one operand pair, block shape and K, every kind of input, one draw each. `interval`: the largest"
        " error over the half-width (on its side) of the interval the facts imply (`new`, and `round 24` for the"
        " round-24 facts); `chunk`: on wgmma, the largest error of one chunk of the model's chain over the new step"
        " bound; `half-width`: the smallest ratio of the new composed bound to the round-24 one; `model = device`:"
        " elements whose device result equals the model's, bit for bit (wgmma).\n")
    out("| operands | M x N, warps | path | K | elements in range | outside | model = device | interval: new | interval: round 24 | chunk: new | chunk: loss arm | half-width new / round 24 | magnitude |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for da, db in PAIRS:
        for (M, N, nw, kmax) in ([(64, 64, 4, 128), (16, 16, 1, 128)] if QUICK else SHAPES_B):
            K = 64
            while K <= kmax:
                agg = Agg()
                path = None
                try:
                    for kind in (KINDS[:3] + KINDS[13:] if QUICK else KINDS):
                        A, B, C = gen(kind, M, N, K, da, db)
                        R, path = run_dot(A, B, C, da, db, nw)
                        Ad, Bd, Cd, Rd = A.double(), B.double(), C.double(), R.double()
                        ps = model(Ad, Bd, Cd, da, db) if path == "wgmma" else None
                        agg.add(Ad, Bd, Cd, Rd, da, db, (NAME[da], NAME[db], M, N, nw, "plain", path, K, kind),
                                ps=ps, sim=ps[-1] if ps is not None else None)
                except triton.runtime.errors.OutOfResources:
                    out(f"| {NAME[da]} x {NAME[db]} | {M} x {N}, {nw} | rejected | {K} | (out of resources) | | | | | | | | |")
                    break
                sim = f"{agg.sim_eq}/{agg.sim_n}" if path == "wgmma" else "-"
                ch = f"{agg.st_new:.4f} | {agg.st_loss:.4f}" if path == "wgmma" else "- | -"
                out(f"| {NAME[da]} x {NAME[db]} | {M} x {N}, {nw} | {path} | {K} | {agg.room} | {agg.out + agg.nonfin} |"
                    f" {sim} | {agg.r_new:.4f} | {agg.r_old:.4f} | {ch} | {agg.w_min:.4f} | {agg.grow} |")
                key = (da, db, path, "B")
                t = totals.setdefault(key, dict(room=0, out=0, new=0.0, loss=0.0, old=0.0, grow=0, worst=None,
                                                sim_n=0, sim_eq=0, hw=9.0, chunk=0.0))
                t["room"] += agg.room; t["out"] += agg.out + agg.nonfin; t["grow"] += agg.grow
                t["sim_n"] += agg.sim_n; t["sim_eq"] += agg.sim_eq
                if agg.r_new >= t["new"]:
                    t["new"] = agg.r_new; t["worst"] = agg.worst
                t["old"] = max(t["old"], agg.r_old); t["chunk"] = max(t["chunk"], agg.st_new)
                t["loss"] = max(t["loss"], agg.st_loss); t["hw"] = min(t["hw"], agg.w_min)
                K *= 2


def part_c(out, totals):
    out("\n## Part C: chains of tl.dot(p, v, acc) (tutorial 06)\n")
    out("`for t in range(T): [acc = acc * alpha_t]; acc = tl.dot(p_t.to(fp8), v_t, acc)`, p_t [M, 64] converted in"
        " registers, v_t [64, N], every acc_t stored. `per dot`: each dot from the stored accumulator before it (times"
        " alpha_t in float32) against the new facts' interval (and, on wgmma, the model, and each chunk of the model's"
        " chain against the new step bound). `whole chain` (no rescale): the final result against the facts composed"
        " over all 2T chunks from the initial accumulator; `half-width`: that composed bound, new over round 24, for"
        " the element where the ratio is largest (`max`) and smallest (`min`). Inputs: `softmax` (p = exp2 of random"
        " scores minus the row maximum, v = randn), `accdom` (after a randn * 2^6 accumulator, products just below the"
        " accumulator's window unit), `subwin` (num-dot-chunk's window case).\n")
    out("| operands | M x N, warps | path | p, v | rescale | T | per dot: model = device | per dot: new | per dot: chunk new | whole chain: model = device | whole chain: new | whole chain: round 24 | half-width min | half-width max |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    KC = 64
    for da, db in PAIRS:
        for (M, N, nw) in ([(128, 64, 4), (32, 64, 2)] if QUICK else [(128, 64, 4), (128, 128, 4), (32, 64, 2)]):
            for pk in ["softmax", "accdom", "subwin"]:
                for scale in (False, True):
                    for T in ([4] if QUICK else ([1, 4, 16, 64, 256] if (pk == "softmax" and not scale) else [16, 64])):
                        P = torch.zeros(T, M, KC, device=DEV); V = torch.zeros(T, KC, N, device=DEV)
                        for t in range(T):
                            if pk == "softmax":
                                sc = torch.randn(M, KC, device=DEV) * [1.0, 4.0, 16.0][t % 3]
                                p = torch.exp2(sc - sc.amax(1, keepdim=True))
                                v = torch.randn(KC, N, device=DEV)
                            elif pk == "accdom":
                                r = torch.randint(4, 10, (M, KC), device=DEV).float()
                                p = FMT[da][5] * torch.exp2(-r - 1)
                                v = torch.sign(torch.randn(KC, N, device=DEV))
                                v = torch.where(v == 0, torch.ones_like(v), v)
                            else:
                                p, v, _ = gen("subwin", M, N, KC, da, db)
                            P[t] = q8(p, da); V[t] = q8(v, db)
                        if pk == "softmax":
                            c0 = torch.randn(M, N, device=DEV)
                        elif pk == "accdom":
                            c0 = torch.randn(M, N, device=DEV) * 2.0 ** 6
                        else:
                            c0 = torch.zeros(M, N, device=DEV)
                        al = (torch.rand(T, device=DEV) * 0.75 + 0.25) if scale else torch.ones(T, device=DEV)
                        accs = torch.empty(T, M, N, device=DEV)
                        V8 = V.to(FMT[db][0])
                        h = chain[(1,)](P.contiguous(), V8.contiguous(), c0, al, accs, T, M, N, KC, FMT[da][1], scale,
                                        num_warps=nw)
                        path = path_of(h)
                        agg = Agg()
                        prev = c0
                        for t in range(T):
                            vc = prev * al[t] if scale else prev
                            Ad, Bd, Cd, Rd = P[t].double(), V[t].double(), vc.double(), accs[t].double()
                            ps = model(Ad, Bd, Cd, da, db) if path == "wgmma" else None
                            agg.add(Ad, Bd, Cd, Rd, da, db, ("chain", NAME[da], NAME[db], M, N, pk, t), ps=ps,
                                    sim=ps[-1] if ps is not None else None)
                            prev = accs[t]
                        whole = "- | - | - | - | -"
                        if not scale:
                            Aall = torch.cat([P[t] for t in range(T)], 1).double()
                            Ball = torch.cat([V[t] for t in range(T)], 0).double()
                            agg2 = Agg()
                            ps2 = model(Aall, Ball, c0.double(), da, db) if path == "wgmma" else None
                            Rf = accs[T - 1].double()
                            agg2.add(Aall, Ball, c0.double(), Rf, da, db, ("whole chain", NAME[da], NAME[db], M, N, pk, T),
                                     sim=ps2[-1] if ps2 is not None else None)
                            cs, As, ms = chunk_data(Aall, Ball, da, db)
                            S, un, dn, uo, do = intervals(c0.double(), cs, As, ms)
                            hw = torch.maximum(un, dn) / torch.maximum(uo, do)
                            sim2 = f"{agg2.sim_eq}/{agg2.sim_n}" if path == "wgmma" else "-"
                            whole = (f"{sim2} | {agg2.r_new:.4f} | {agg2.r_old:.4f} | {float(hw.min()):.4g} |"
                                     f" {float(hw.max()):.4g}")
                            key = (da, db, path, "C whole")
                            t2 = totals.setdefault(key, dict(room=0, out=0, new=0.0, loss=0.0, old=0.0, grow=0, worst=None,
                                                             sim_n=0, sim_eq=0, hw=9.0, chunk=0.0))
                            t2["room"] += agg2.room; t2["out"] += agg2.out + agg2.nonfin; t2["grow"] += agg2.grow
                            t2["sim_n"] += agg2.sim_n; t2["sim_eq"] += agg2.sim_eq
                            if agg2.r_new >= t2["new"]:
                                t2["new"] = agg2.r_new; t2["worst"] = agg2.worst
                            t2["old"] = max(t2["old"], agg2.r_old); t2["hw"] = min(t2["hw"], agg2.w_min)
                        sim = f"{agg.sim_eq}/{agg.sim_n}" if path == "wgmma" else "-"
                        chn = f"{agg.st_new:.4f}" if path == "wgmma" else "-"
                        out(f"| {NAME[da]} x {NAME[db]} | {M} x {N}, {nw} | {path} | {pk} | {'yes' if scale else 'no'} | {T} |"
                            f" {sim} | {agg.r_new:.4f} | {chn} | {whole} |")
                        key = (da, db, path, "C dot")
                        t1 = totals.setdefault(key, dict(room=0, out=0, new=0.0, loss=0.0, old=0.0, grow=0, worst=None,
                                                         sim_n=0, sim_eq=0, hw=9.0, chunk=0.0))
                        t1["room"] += agg.room; t1["out"] += agg.out + agg.nonfin; t1["grow"] += agg.grow
                        t1["sim_n"] += agg.sim_n; t1["sim_eq"] += agg.sim_eq
                        if agg.r_new >= t1["new"]:
                            t1["new"] = agg.r_new; t1["worst"] = agg.worst
                        t1["old"] = max(t1["old"], agg.r_old); t1["chunk"] = max(t1["chunk"], agg.st_new)
                        t1["loss"] = max(t1["loss"], agg.st_loss); t1["hw"] = min(t1["hw"], agg.w_min)


def main():
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    totals = {}
    part_p(out)
    part_s(out, totals)
    part_b(out, totals)
    part_c(out, totals)
    out("\n## Totals\n")
    out("Part S: single chunks (the step bound on the device's own values); Part B: every K; Part C: chains, per dot and"
        " whole. `new`: the largest error over the new facts' bound (step bound in Part S, interval otherwise);"
        " `loss arm`: the step over the second arm alone (Part S: the device's step; Parts B and C: each chunk of the"
        " model's chain, wgmma); `round 24`: as `new` against the round-24 facts; `chunk`: each chunk of the model's"
        " chain over the new step bound (wgmma); `half-width`: the smallest ratio of the new composed bound to the"
        " round-24 one; `magnitude`: results above |acc| + sum |a b|.\n")
    out("| operands | path | part | elements in range | outside | model = device | new | loss arm | round 24 | chunk | half-width | magnitude |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (da, db, path, part), t in sorted(totals.items()):
        sim = f"{t['sim_eq']}/{t['sim_n']}" if t["sim_n"] else "-"
        model_cols = path == "wgmma" and part in ("B", "C dot")
        loss = f"{t['loss']:.4f}" if (part == "S" or model_cols) else "-"
        chunk = f"{t['chunk']:.4f}" if model_cols else "-"
        out(f"| {NAME[da]} x {NAME[db]} | {path} | {part} | {t['room']} | {t['out']} | {sim} | {t['new']:.4f} |"
            f" {loss} | {t['old']:.4f} | {chunk} | {t['hw']:.4g} | {t['grow']} |")
    out("\n### The inputs with the largest new ratio\n")
    for key, t in sorted(totals.items()):
        out(f"- {NAME[key[0]]} x {NAME[key[1]]}, {key[2]}, Part {key[3]}: {t['new']:.4f}, {fmt_worst(t['worst'])}")
    out(f"\n(run time {time.time() - t0:.0f} s)")
    with open("evidence/num-dot-chunk-loss/" + ("quick.txt" if QUICK else "out.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
