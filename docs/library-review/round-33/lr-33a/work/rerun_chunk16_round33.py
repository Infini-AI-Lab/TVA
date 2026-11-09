"""TVA library evidence (library issue L150, round 33): which k each instruction of a 16-bit `tl.dot` adds on sm_90,
and the chunk facts `DotF16Chunk`, `DotBF16Chunk` restated per chunk of 64 consecutive k.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-chunk16-round33/run.py
Output: evidence/num-dot-chunk16-round33/output.txt (README.md says what it shows).

The facts checked (lean/TVA/Rules/Num/Dot16Chunk.lean, as of round 33): for float16 (bfloat16) operands into float32,
`max_num_imprecise_acc = 0`, per output element there are partial values s_0 = acc, ..., s_n = result
(n = ceil(K/64)), one step per chunk of 64 consecutive k (all K of them when K < 64), with

    |s_{j+1} - (s_j + c_j)| <= min(2^-13 Mj, 2 A_j + 2^-22 Mj) + 2^-20 |s_j + c_j| + 2^-143

for every Mj at least |s_j| and every lifted product of the chunk; c_j is the chunk's exact sum, A_j its sum of |a b|.

The per-instruction model is evidence/num-dot-sub16-round25's: the terms are the accumulator and the instruction's 16
exact products, aligned to the largest exponent field E (a subnormal operand reads as the smallest normal; the
accumulator's field is max(floor(log2|s|), -126)), each truncated toward zero to a multiple of u = 2^(E-25), summed
exactly, and the sum truncated toward zero to float32. Here the 16 k of an instruction are those the device uses.

Parts:
  W. kWidth and path for each operand origin (loaded in the format, float32, float64, int16, int8, uint8, bool,
     float8e5, float8e4nv, an fp8 load unpacked by tl.join), shape and warp count, from the TTGIR and the PTX.
  P. The k each instruction adds, found on the device (a product 1 among products 2^-27, the k found so far zeroed),
     against the rule used by the model, at K = 16 to 256.
  M. The model, with the instructions' k, against the device on every element; the chunk facts' step at the
     partial values the model's chain gives at each chunk's end (its last value is the device's result), in exact
     rationals at the smallest Mj; and the round-32 facts (16 consecutive k) on the same elements.
  X. Round 32's counterexample (lr-32b): K = 32, a converted from float8e5 on mma.sync.
"""
import itertools
import math
import random
import re
import sys
from fractions import Fraction as Fr

import torch
import triton
import triton.language as tl

torch.manual_seed(20261108)
random.seed(20261108)
DEV = "cuda"
BIG = 1 << 20
W = 25
FMT = {"f16": (torch.float16, 10, -14), "bf16": (torch.bfloat16, 7, -126)}
SRC = {"fmt": 0, "e5": 1, "i8": 2, "join": 3, "f32": 4, "e4": 5, "u8": 6, "i16": 7, "f64": 8, "bool": 9}
OUT = []


def out(s=""):
    print(s, flush=True)
    OUT.append(s)


@triton.jit
def kd(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, SA: tl.constexpr,
       SB: tl.constexpr, BF: tl.constexpr, ACC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    if SA == 3:
        rk2 = tl.arange(0, K // 2)
        lo = tl.load(a_ptr + rm[:, None] * K + 2 * rk2[None, :])
        hi = tl.load(a_ptr + rm[:, None] * K + 2 * rk2[None, :] + 1)
        if BF:
            a = tl.join(lo.to(tl.bfloat16), hi.to(tl.bfloat16)).reshape(M, K)
        else:
            a = tl.join(lo.to(tl.float16), hi.to(tl.float16)).reshape(M, K)
    else:
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        if BF:
            a = a.to(tl.bfloat16)
        else:
            a = a.to(tl.float16)
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    if BF:
        b = b.to(tl.bfloat16)
    else:
        b = b.to(tl.float16)
    if ACC:
        r = tl.dot(a, b, tl.load(c_ptr + rm[:, None] * N + rn[None, :]))
    else:
        r = tl.dot(a, b)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], r)


TORCH_SRC = {"e5": torch.float8_e5m2, "e4": torch.float8_e4m3fn, "i8": torch.int8, "u8": torch.uint8,
             "i16": torch.int16, "f32": torch.float32, "f64": torch.float64, "bool": torch.bool, "join": torch.float8_e5m2}


def dev_tensor(x, src, f):
    """x: float32 values exactly representable in the source dtype."""
    if src == "fmt":
        return x.to(FMT[f][0]).contiguous()
    return x.to(TORCH_SRC[src]).contiguous()


def launch(A, B, C, f, sa, sb, nw, acc=True):
    M, K = A.shape
    N = B.shape[1]
    o = torch.empty(M, N, dtype=torch.float32, device=DEV)
    h = kd[(1,)](dev_tensor(A, sa, f), dev_tensor(B, sb, f), C.contiguous(), o, M, N, K, SRC[sa], SRC[sb],
                 int(f == "bf16"), int(acc), num_warps=nw)
    kw = sorted(set(int(x) for x in re.findall(r"kWidth\s*=\s*(\d+)", h.asm["ttgir"])))
    ptx = h.asm["ptx"]
    path = "wgmma" if "wgmma.mma_async" in ptx else ("mma.sync" if "mma.sync" in ptx else "other")
    return o, path, kw


# ------------------------------------------------------------------------------------------- the partitions

def rule(path, kw, K):
    """The k of each instruction, in order, as the model uses them (checked against the device in Part P)."""
    if path == "wgmma" or kw in ([], [2]) or K <= 16:
        return [list(range(k0, min(K, k0 + 16))) for k0 in range(0, K, 16)]
    w = kw[0]
    if w == 4:
        return [[b0 + 4 * c + 2 * i + e + 16 * h for h in range(2) for c in range(4) for e in range(2)]
                for b0 in range(0, K, 32) for i in range(2)]
    if w == 8 and K == 32:
        return [[8 * c + 4 * i + e for c in range(4) for e in range(4)] for i in range(2)]
    if w == 8:
        return [[b0 + 8 * c + 2 * i + e + 32 * h for h in range(2) for c in range(4) for e in range(2)]
                for b0 in range(0, K, 64) for i in range(4)]
    raise ValueError((path, kw, K))


def probe_partition(f, sa, M, N, K, nw):
    """Part P's probe: row values a = 2^-14, b = 2^-13 (products 2^-27) but a 1 at k = big; zeroed k removed. The
    result is above 1 iff an instruction before the one holding big held 16 small products (2^-23 above 1 after
    rounding is kept only when 16 products 2^-27 are added in one instruction before the 1 sets the window)."""
    info = {}

    def once(big, zero):
        A = torch.full((M, K), 2.0 ** -14, device=DEV)
        B = torch.full((K, N), 2.0 ** -13, device=DEV)
        A[:, big] = 1.0
        B[big, :] = 1.0
        for z in zero:
            A[:, z] = 0.0
        o, path, kw = launch(A, B, torch.zeros(M, N, device=DEV), f, sa, "fmt", nw, acc=False)
        info["path"], info["kw"] = path, kw
        v = set(o.flatten().tolist())
        assert len(v) == 1, v
        return v.pop() == 1.0

    left, zero, parts = list(range(K)), [], []
    while left:
        first = [k for k in left if once(k, zero)]
        if len(first) == len(left) or not first:
            parts.append(left)
            break
        parts.append(first)
        zero += first
        left = [k for k in left if k not in first]
    return parts, info["path"], info["kw"]


# ------------------------------------------------------------------------------------------- the model

def rawexp(x, emin):
    e = torch.frexp(x.abs())[1].to(torch.int64) - 1
    return torch.clamp(e, min=emin)


def trunc_f32(S):
    er = torch.frexp(S.abs())[1].to(torch.int64) - 1
    g = torch.ldexp(torch.ones_like(S), torch.clamp(er - 23, min=-149))
    return torch.where(S == 0, S, torch.trunc(S / g) * g)


def model(A, B, C, f, parts, acc=True):
    """float64 tensors in (exact: every truncated term is an integer multiple of u below 2^27 u). Returns the
    result and the value after each instruction."""
    emin = FMT[f][2]
    ea, eb = rawexp(A, emin), rawexp(B, emin)
    s = C.clone() if acc else torch.zeros_like(C)
    after = []
    for idx in parts:
        ix = torch.tensor(idx, device=A.device)
        P = A[:, None, ix] * B[ix, :].T[None, :, :]
        EP = ea[:, None, ix] + eb[ix, :].T[None, :, :]
        EP = torch.where(P == 0, torch.full_like(EP, -BIG), EP)
        es = torch.clamp(torch.frexp(s.abs())[1].to(torch.int64) - 1, min=-126)
        es = torch.where(s == 0, torch.full_like(es, -BIG), es)
        E = torch.maximum(EP.amax(-1), es)
        u = torch.ldexp(torch.ones_like(s), torch.clamp(E, min=-1000) - W)
        S1 = (torch.trunc(s / u) + torch.trunc(P / u[..., None]).sum(-1)) * u
        S1 = torch.where(E <= -BIG // 2, torch.zeros_like(S1), S1)
        s = trunc_f32(S1)
        after.append(s)
    return s, after


# ------------------------------------------------------------------------------------------- the bound

RHO, SIG, ALPHA, KAPPA, RHOP = Fr(1, 2 ** 13), Fr(1, 2 ** 20), Fr(1, 2 ** 143), Fr(2), Fr(1, 2 ** 22)
OLD = (Fr(1, 2 ** 20), Fr(1, 2 ** 22), Fr(1, 2 ** 145), Fr(2), Fr(1, 2 ** 24))


def lift(x, emin):
    return Fr(0) if x == 0 else max(abs(x), Fr(2) ** emin)


def step_ratio(s0, s1, prods, lifts, consts):
    rho, sig, alpha, kap, rhop = consts
    c = sum(prods, Fr(0))
    A = sum((abs(p) for p in prods), Fr(0))
    Mj = max([abs(s0)] + lifts)
    bnd = min(rho * Mj, kap * A + rhop * Mj) + sig * abs(s0 + c) + alpha
    return abs(s1 - (s0 + c)) / bnd


def chunk_steps(arow, bcol, c0, ends, emin, size, consts):
    """The steps of one element: ends[j] = the partial value after chunk j (chunks of `size` consecutive k)."""
    K = len(arow)
    prods = [x * y for x, y in zip(arow, bcol)]
    lifts = [lift(x, emin) * lift(y, emin) for x, y in zip(arow, bcol)]
    s, worst = c0, Fr(0)
    for j, k0 in enumerate(range(0, K, size)):
        k1 = min(K, k0 + size)
        r = step_ratio(s, ends[j], prods[k0:k1], lifts[k0:k1], consts)
        worst = max(worst, r)
        s = ends[j]
    return worst


def feasible_old(arow, bcol, c0, r, emin):
    """Do partial values of 16-chunks exist that meet the round-32 bound (one free value per inner boundary)?
    Only checked for K = 32 (one inner value): the slack is piecewise linear in s1, so its kinks and ends decide."""
    K = len(arow)
    assert K == 32
    rho, sig, alpha, kap, rhop = OLD
    prods = [x * y for x, y in zip(arow, bcol)]
    lifts = [lift(x, emin) * lift(y, emin) for x, y in zip(arow, bcol)]
    c0s, c1s = sum(prods[:16], Fr(0)), sum(prods[16:], Fr(0))
    A0, A1 = sum((abs(p) for p in prods[:16]), Fr(0)), sum((abs(p) for p in prods[16:]), Fr(0))
    M0 = max([abs(c0)] + lifts[:16])
    B0 = min(rho * M0, kap * A0 + rhop * M0) + sig * abs(c0 + c0s) + alpha
    lo, hi = c0 + c0s - B0, c0 + c0s + B0
    ml = max(lifts[16:])

    def ok(s1):
        M1 = max(abs(s1), ml)
        return abs(r - (s1 + c1s)) <= min(rho * M1, kap * A1 + rhop * M1) + sig * abs(s1 + c1s) + alpha
    sw = kap * A1 / (rho - rhop)
    cands = [lo, hi] + [x for x in (r - c1s, Fr(0), ml, -ml, -c1s, sw, -sw) if lo <= x <= hi]
    return any(ok(x) for x in cands)


# ------------------------------------------------------------------------------------------- inputs

def q_src(x, src, f):
    """Round float32 values into the source dtype's values (and the format's, since they are converted)."""
    if src == "fmt":
        return x.to(FMT[f][0]).float()
    if src in ("e5", "join"):
        y = x.clamp(-57344, 57344).to(torch.float8_e5m2).float()
    elif src == "i8":
        y = x.round().clamp(-128, 127)
    else:
        raise ValueError(src)
    return y.to(FMT[f][0]).float()


def sgn(shape):
    return torch.where(torch.rand(shape, device=DEV) < 0.5, -1.0, 1.0)


def gen(kind, M, N, K, f, sa, parts, sb="fmt"):
    dt, frac, emin = FMT[f]
    if kind == "normal":
        sc = 32.0 if sa == "i8" else 1.0
        a = torch.randn(M, K, device=DEV) * sc
        b = torch.randn(K, N, device=DEV)
        c = torch.randn(M, N, device=DEV) * torch.randint(0, 2, (M, N), device=DEV)
    elif kind == "spread":
        a = torch.randn(M, K, device=DEV) * torch.exp2(torch.randint(-8, 8, (M, K), device=DEV).float())
        if sa == "i8":
            a = a * 16
        b = torch.randn(K, N, device=DEV) * torch.exp2(torch.randint(-8, 8, (K, N), device=DEV).float())
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-12, 12, (M, N), device=DEV).float())
    elif kind == "bits":
        if sa in ("e5", "join"):
            a = torch.randint(0, 256, (M, K), device=DEV, dtype=torch.uint8).view(torch.float8_e5m2).float()
            a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
        elif sa == "i8":
            a = torch.randint(-128, 128, (M, K), device=DEV).float()
        else:
            a = torch.randint(-32768, 32768, (M, K), device=DEV, dtype=torch.int16).view(dt).float()
            a = torch.where(torch.isfinite(a) & (a.abs() < 2.0 ** 40), a, torch.zeros_like(a))
        b = torch.randn(K, N, device=DEV) * torch.exp2(torch.randint(-20, 4, (K, N), device=DEV).float())
        c = torch.randn(M, N, device=DEV) * torch.exp2(torch.randint(-30, 20, (M, N), device=DEV).float())
    elif kind == "accdom":
        # an accumulator 2^20 (times 1, 1 + 2^-23 or 2 - 2^-23) with every product just below the window's unit
        # u = 2^-5 (lost whole): the second arm's worst case
        a = sgn((M, 1)) * torch.ones(M, K, device=DEV)
        b = torch.full((K, N), 2.0 ** -5 - 2.0 ** -16, device=DEV)
        if sa == "i8":
            b = b / 2
            a = a * 2
        m = torch.randint(0, 3, (M, N), device=DEV)
        c = 2.0 ** 20 * torch.where(m == 0, 1.0, torch.where(m == 1, 1 + 2.0 ** -23, 2 - 2.0 ** -23)) * sgn((M, N))
    elif kind == "grow":
        # per chunk: the accumulator 1, then the products of the chunk's first instructions 1 each (so the
        # accumulator grows to 49 by the last one), the last instruction's products just below its unit 2^-20
        # (lost whole): the first arm's case, where the accumulator outgrows Mj within the chunk
        a = torch.zeros(M, K, device=DEV)
        b = torch.zeros(K, N, device=DEV)
        for j0 in range(0, K, 64):
            ins = [p for p in parts if min(p) >= j0 and max(p) < j0 + 64]
            for t, p in enumerate(ins):
                for k in p:
                    if t < len(ins) - 1:
                        a[:, k] = 1.0
                        b[k, :] = 1.0
                    else:
                        a[:, k] = 2.0 ** -7 if sa != "i8" else 1.0
                        b[k, :] = (2 - 2.0 ** -10) * 2.0 ** -14 if sa != "i8" else (2 - 2.0 ** -7) * 2.0 ** -21
        c = torch.ones(M, N, device=DEV) * sgn((M, 1))
        a = a * sgn((M, 1)).abs()
    elif kind == "sub":
        # subnormal results of the conversion and subnormal partners: lifted products far above the products
        if sa in ("e5", "join"):
            a = torch.randint(1, 4, (M, K), device=DEV).float() * 2.0 ** -16 * sgn((M, K))
            a = torch.where(torch.rand(M, K, device=DEV) < 0.2, torch.randn(M, K, device=DEV) * 64, a)
        elif sa == "i8":
            a = torch.randint(-3, 4, (M, K), device=DEV).float()
        else:
            a = torch.randn(M, K, device=DEV) * 2.0 ** emin
        b = torch.where(torch.rand(K, N, device=DEV) < 0.5, torch.randn(K, N, device=DEV) * 2.0 ** 10,
                        torch.randn(K, N, device=DEV) * 2.0 ** emin)
        c = torch.randn(M, N, device=DEV) * 2.0 ** -20 * torch.randint(0, 2, (M, N), device=DEV)
    else:
        raise ValueError(kind)
    a = q_src(a, sa, f)
    b = q_src(b, sb, f)
    return a.contiguous(), b.contiguous(), c.float().contiguous()


KINDS = ["normal", "spread", "bits", "accdom", "grow", "sub"]


# ------------------------------------------------------------------------------------------- parts

def part_w():
    out("== Part W: kWidth and path by operand origin (Triton's choice; mma.sync unless M >= 64 with 4 warps)")
    shapes = [(16, 16, 1), (32, 32, 2), (16, 64, 1), (32, 64, 4), (64, 32, 2), (64, 64, 4), (128, 64, 4), (128, 128, 8)]
    seen = {}
    for f in ("f16", "bf16"):
        for sa, sb in [("fmt", "fmt"), ("f32", "fmt"), ("f32", "f32"), ("f64", "f64"), ("i16", "fmt"), ("i8", "fmt"),
                       ("u8", "fmt"), ("bool", "fmt"), ("e5", "fmt"), ("e4", "fmt"), ("fmt", "e5"), ("e5", "e5"),
                       ("fmt", "i8"), ("join", "fmt"), ("join", "e5")]:
            row = []
            for (M, N, nw) in shapes:
                for K in (32, 128):
                    A = torch.zeros(M, K, device=DEV)
                    B = torch.zeros(K, N, device=DEV)
                    try:
                        _, path, kw = launch(A, B, torch.zeros(M, N, device=DEV), f, sa, sb, nw)
                        row.append(f"{M}x{N}x{K}/{nw}w:{path}:kW{kw}")
                        seen.setdefault((path, tuple(kw)), []).append((f, sa, sb, M, N, K, nw))
                    except Exception as e:  # noqa: BLE001
                        row.append(f"{M}x{N}x{K}/{nw}w:{type(e).__name__}")
            out(f"W {f} a={sa} b={sb}: " + " ".join(row))
    out("W kWidth values seen (path, kWidth): " + ", ".join(f"{p} {list(k)}: {len(v)}" for (p, k), v in sorted(seen.items())))
    return seen


def part_p():
    out("== Part P: the k of each instruction, probed on the device, against the model's rule")
    cfgs = []
    for f in ("f16", "bf16"):
        for sa in ("fmt", "e5", "i8", "join"):
            for (M, N, nw) in ((16, 16, 1), (32, 32, 2), (16, 64, 1), (32, 64, 4)):
                for K in (16, 32, 64, 128, 256):
                    if K == 256 and (M, N, nw) != (16, 16, 1):
                        continue
                    cfgs.append((f, sa, M, N, K, nw))
    cfgs.append(("f16", "e5", 64, 64, 128, 4))
    cfgs.append(("f16", "fmt", 64, 64, 128, 4))
    bad = tot = 0
    worst_span = 0
    for (f, sa, M, N, K, nw) in cfgs:
        if sa == "i8":
            continue  # the probe's values are float8e5 values; int8 sources share the kWidth 4 layout (Part W, M)
        parts, path, kw = probe_partition(f, sa, M, N, K, nw)
        r = rule(path, kw, K)
        same = parts == r
        sizes = sorted(set(len(p) for p in parts))
        blocks = [(min(p) // 64, max(p) // 64) for p in parts]
        in64 = all(b0 == b1 for b0, b1 in blocks) and all(blocks[i][0] <= blocks[i + 1][0] for i in range(len(blocks) - 1))
        span = max(max(p) - min(p) + 1 for p in parts)
        worst_span = max(worst_span, span)
        tot += 1
        bad += (not same) or (not in64)
        out(f"P {f} a={sa} {M}x{N}x{K} {nw}w {path} kW{kw}: {len(parts)} instructions of {sizes} k, widest span {span}, "
            f"each in one aligned block of 64, blocks in order: {in64}; equal to the rule: {same}"
            + ("" if K > 64 or kw != [8] else f"; {parts}"))
    out(f"P total: {tot} configurations, {bad} with a partition other than the rule or outside one block of 64; "
        f"widest span of one instruction's k: {worst_span}")


def part_m():
    out("== Part M: the model (the instructions' k) against the device; the chunk facts' steps (chunks of 64)")
    cfgs = []
    for f in ("f16", "bf16"):
        for sa in ("fmt", "e5", "i8", "join"):
            for (M, N, nw) in ((16, 16, 1), (32, 32, 2), (16, 64, 1), (64, 64, 4)):
                if (M, N, nw) == (64, 64, 4) and sa in ("i8", "join"):
                    continue
                for K in (16, 32, 64, 128, 256):
                    cfgs.append((f, sa, "fmt", M, N, K, nw))
        for sa in ("fmt", "e5"):
            for (M, N, nw) in ((16, 16, 1), (32, 32, 2), (64, 64, 4)):
                for K in (32, 128):
                    cfgs.append((f, sa, "e5", M, N, K, nw))
    stats = {}
    tot_el = tot_eq = tot_ex = tot_out = old_out = 0
    for (f, sa, sb, M, N, K, nw) in cfgs:
        emin = FMT[f][2]
        for kind in KINDS:
            reps = 2
            for rep in range(reps):
                # the partition first (needed by `grow`), from a launch with zeros
                _, path, kw = launch(torch.zeros(M, K, device=DEV), torch.zeros(K, N, device=DEV),
                                     torch.zeros(M, N, device=DEV), f, sa, sb, nw)
                parts = rule(path, kw, K)
                A, B, C = gen(kind, M, N, K, f, sa, parts, sb)
                use_acc = kind != "grow" or True
                o, path, kw = launch(A, B, C, f, sa, sb, nw, acc=use_acc)
                R, after = model(A.double(), B.double(), C.double(), f, parts, acc=use_acc)
                Rd = o.double()
                fin = torch.isfinite(Rd) & torch.isfinite(R)
                eq = ((R == Rd) | (~torch.isfinite(R) & ~torch.isfinite(Rd))).sum().item()
                tot_el += R.numel()
                tot_eq += eq
                # the partial values at each chunk's end (its last instruction) from the model's chain
                ends_idx = []
                for j0 in range(0, K, 64):
                    last = max(i for i, p in enumerate(parts) if min(p) < j0 + 64)
                    ends_idx.append(last)
                ends16_idx = [max(i for i, p in enumerate(parts) if min(p) < j0 + 16) for j0 in range(0, K, 16)]
                key = (f, path, tuple(kw), "b converted" if sb != "fmt" else "", kind)
                st = stats.setdefault(key, [0, 0, Fr(0), 0])
                els = [(i, n) for i in range(0, M, max(1, M // 8)) for n in range(0, N, max(1, N // 4))]
                for (i, n) in els:
                    if not bool(fin[i, n]):
                        continue
                    arow = [Fr(x) for x in A[i].tolist()]
                    bcol = [Fr(x) for x in B[:, n].tolist()]
                    c0 = Fr(C[i, n].item()) if use_acc else Fr(0)
                    ends = [Fr(after[e][i, n].item()) for e in ends_idx]
                    assert ends[-1] == Fr(Rd[i, n].item()) or R[i, n] != Rd[i, n]
                    w = chunk_steps(arow, bcol, c0, ends, emin, 64, (RHO, SIG, ALPHA, KAPPA, RHOP))
                    st[0] += 1
                    st[1] += w > 1
                    st[2] = max(st[2], w)
                    tot_ex += 1
                    tot_out += w > 1
                    # the round-32 facts, on the same partial values, where the instructions are 16 consecutive k
                    if parts == rule("wgmma", [], K):
                        ends16 = [Fr(after[e][i, n].item()) for e in ends16_idx]
                        w16 = chunk_steps(arow, bcol, c0, ends16, emin, 16, OLD)
                        old_out += w16 > 1
                    elif K == 32:
                        st[3] += not feasible_old(arow, bcol, c0, Fr(Rd[i, n].item()), emin)
    out("M model = device: %d of %d element results" % (tot_eq, tot_el))
    for (f, path, kw, bc, kind), (n, nout, w, nold) in sorted(stats.items()):
        out(f"M {f} {path} kW{list(kw)}{' ' + bc if bc else ''} {kind:7s}: {n} elements checked exactly, {nout} outside the chunk-of-64 bound, "
            f"largest ratio {float(w):.4f}" + (f"; at K = 32, {nold} with no partial values of 16-chunks meeting "
                                              f"the round-32 bound" if kw and kw[0] > 2 else ""))
    out(f"M total: {tot_ex} elements checked exactly against the chunks of 64, {tot_out} outside; on consecutive "
        f"partitions {old_out} outside the round-32 bound (16-chunks)")


def part_x():
    out("== Part X: round 32's counterexample (lr-32b), K = 32, float16, a converted from float8e5, mma.sync")
    u = Fr(1, 2 ** 25)
    K, M, N = 32, 16, 16
    A = torch.full((M, K), 2.0 ** -13, device=DEV)
    B = torch.full((K, N), (2 - 2.0 ** -10) * 2.0 ** -13, device=DEV)
    A[:, 0] = 1.0; B[0, :] = 1.0
    A[:, 2] = -1.0; B[2, :] = 1.0
    A[:, 1] = 1.5 * 2.0 ** -12; B[1, :] = 2.0 ** -12
    C = torch.full((M, N), (1 - 2.0 ** -24) * 2.0 ** -25, device=DEV)
    for sa in ("e5", "fmt"):
        o, path, kw = launch(A, B, C, "f16", sa, "fmt", 1)
        vals = sorted(set(o.flatten().tolist()))
        arow = [Fr(x) for x in A[0].tolist()]
        bcol = [Fr(x) for x in B[:, 0].tolist()]
        c0 = Fr(C[0, 0].item())
        r = Fr(vals[0])
        old = feasible_old(arow, bcol, c0, r, -14)
        new = chunk_steps(arow, bcol, c0, [r], -14, 64, (RHO, SIG, ALPHA, KAPPA, RHOP))
        exact = c0 + sum((x * y for x, y in zip(arow, bcol)), Fr(0))
        out(f"X a={sa} {path} kW{kw}: device {vals}, exact {float(exact / u):.4f} u, error {float((exact - r) / u):.4f} u; "
            f"round-32 facts (16-chunks) satisfiable: {old}; round-33 fact (one chunk of 64): ratio {float(new):.6f} "
            f"(holds: {new <= 1})")


def main():
    out(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
    part_w()
    part_p()
    part_x()
    part_m()
    with open(__file__.rsplit("/", 1)[0] + "/rerun_chunk16_round33.out", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
