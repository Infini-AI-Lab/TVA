"""lr-32b, round 32, L150: a launch on which DotF16Chunk (Rules/Num/Dot16Chunk.lean) is false.

A float16 tl.dot on mma.sync whose `a` operand is converted in registers from float8e5 (kWidth = 4)
adds, per instruction, the k sets {0,1,4,5,...,28,29} then {2,3,6,7,...,30,31} of each 32-k block
(dot_partition.out), not 16 consecutive k.

1. The per-instruction model of evidence/num-dot-sub16-round25 with that partition, against the
   device, on random inputs (float16, bfloat16; a or b converted).
2. The counterexample (K = 32, one tl.dot, u = 2^-25): the accumulator s0 = (1 - 2^-24) u;
   k = 0: 1 * 1; k = 2: -1 * 1; k = 1: (1.5 * 2^-12) * 2^-12 = 3u; every other k:
   2^-13 * (2 - 2^-10) 2^-13 = (1 - 2^-11) u. The device: the first instruction holds 1, 3u and 14
   small products: 1 + 3u truncates to 1, the rest is lost; the second holds -1 and 15 small
   products with accumulator 1: 0. The fact's chunks are k = 0..15 (c0 = 3u + 13 small, M0 = 1)
   and k = 16..31 (16 small). Step 0 forces s1 >= s0 + c0 - B0, step 1 forces s1 within B1 of
   result - c1; the script computes both in exact rationals and reports whether any s1 exists.

Run from TVA's root: CUDA_VISIBLE_DEVICES=2 .venv/bin/python -I docs/library-review/round-32/lr-32b/work/dot_counter.py
"""
import os, sys, tempfile
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="lr32b-"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import torch, triton, triton.language as tl
from dot_kwidth import instr, consec


@triton.jit
def kdot(a_ptr, b_ptr, c_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         CA: tl.constexpr, CB: tl.constexpr, BF: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    if BF:
        if CA:
            a = a.to(tl.bfloat16)
        if CB:
            b = b.to(tl.bfloat16)
    else:
        if CA:
            a = a.to(tl.float16)
        if CB:
            b = b.to(tl.float16)
    acc = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], tl.dot(a, b, acc))


def obs_part(K):
    out = []
    for blk in range(K // 32):
        base = 32 * blk
        out.append([base + 4 * q + i for q in range(8) for i in (0, 1)])
        out.append([base + 4 * q + 2 + i for q in range(8) for i in (0, 1)])
    return out


def launch(a, b, c, CA, CB, BF, nw=1):
    M, K = a.shape; N = b.shape[1]
    fd = torch.bfloat16 if BF else torch.float16
    ad = (a.to(torch.float8_e5m2) if CA else a.to(fd)).cuda()
    bd = (b.to(torch.float8_e5m2) if CB else b.to(fd)).cuda()
    o = torch.empty(M, N, dtype=torch.float32, device="cuda")
    h = kdot[(1,)](ad, bd, c.float().cuda(), o, M, N, K, CA, CB, BF, num_warps=nw)
    path = "wgmma" if "wgmma" in h.asm["ptx"] else "mma.sync"
    import re
    kw = sorted(set(re.findall(r"kWidth\s*=\s*(\d+)", h.asm["ttgir"])))
    return o.cpu(), path, kw


def chain(s, ar, br, part, emin):
    for ch in part:
        s = instr(s, [(ar[k], br[k]) for k in ch], emin)
    return s


def part1():
    g = torch.Generator().manual_seed(11)
    for BF in (0, 1):
        emin = -126 if BF else -14
        fd = torch.bfloat16 if BF else torch.float16
        for CA, CB in ((1, 0), (0, 1), (1, 1)):
            for (M, N, K, nw) in ((16, 16, 64, 1), (32, 32, 128, 2), (64, 64, 64, 4)):
                tot = eq_obs = eq_con = 0
                for t in range(6):
                    e1 = torch.randint(-14, 4, (M, K), generator=g).float()
                    a = torch.pow(2.0, e1) * torch.where(torch.rand(M, K, generator=g) < .5, -1., 1.)
                    a = a * (1 + torch.randint(0, 4, (M, K), generator=g).float() / 4)
                    e2 = torch.randint(-14, 4, (K, N), generator=g).float()
                    b = (torch.rand(K, N, generator=g) + 1) * torch.pow(2.0, e2)
                    a = (a.to(torch.float8_e5m2) if CA else a.to(fd)).float()
                    b = (b.to(torch.float8_e5m2) if CB else b.to(fd)).float()
                    c = torch.randn(M, N, generator=g) * 4
                    o, path, kw = launch(a, b, c, CA, CB, BF, nw)
                    for i in range(0, M, max(1, M // 4)):
                        ar = [Fr(x) for x in a[i].tolist()]
                        for n in range(0, N, max(1, N // 4)):
                            br = [Fr(x) for x in b[:, n].tolist()]
                            s0 = Fr(c[i, n].item()); d = Fr(o[i, n].item()); tot += 1
                            eq_obs += chain(s0, ar, br, obs_part(K), emin) == d
                            eq_con += chain(s0, ar, br, consec(K), emin) == d
                print(f"part1 {'bf16' if BF else 'f16'} convert a={CA} b={CB} M={M} N={N} K={K} warps={nw} "
                      f"path={path} kWidth={kw}: {tot} elements; equal to the model with the observed "
                      f"partition {eq_obs}, with 16 consecutive k {eq_con}", flush=True)


def part2():
    u = Fr(1, 2 ** 25)
    K, M, N = 32, 16, 16
    sm_a, sm_b = 2.0 ** -13, (2 - 2.0 ** -10) * 2.0 ** -13
    a = torch.full((M, K), sm_a); b = torch.zeros(K, N)
    b[:, :] = sm_b
    a[:, 0] = 1.0; b[0, :] = 1.0
    a[:, 2] = -1.0; b[2, :] = 1.0
    a[:, 1] = 1.5 * 2.0 ** -12; b[1, :] = 2.0 ** -12
    s0f = (1 - 2.0 ** -24) * 2.0 ** -25
    c = torch.full((M, N), s0f)
    for CA, CB in ((1, 0), (0, 0)):
        o, path, kw = launch(a, b, c, CA, CB, 0, 1)
        vals = sorted(set(o.flatten().tolist()))
        ar = [Fr(x) for x in a[0].tolist()]; br = [Fr(x) for x in b[:, 0].tolist()]
        s0 = Fr(s0f)
        assert s0 == (1 - Fr(1, 2 ** 24)) * u
        vr = Fr(vals[0])
        mobs = chain(s0, ar, br, obs_part(K), -14)
        prods = [ar[k] * br[k] for k in range(K)]
        lift = [abs(p) for p in prods]  # every operand normal: liftMag = |x|
        c0 = sum(prods[:16]); c1 = sum(prods[16:])
        A0 = sum(abs(p) for p in prods[:16]); A1 = sum(abs(p) for p in prods[16:])
        rho, sig, alp, kap, rho2 = Fr(1, 2 ** 20), Fr(1, 2 ** 22), Fr(1, 2 ** 145), 2, Fr(1, 2 ** 24)
        M0 = max(abs(s0), max(lift[:16]))
        B0 = min(rho * M0, kap * A0 + rho2 * M0) + sig * abs(s0 + c0) + alp
        lo, hi = s0 + c0 - B0, s0 + c0 + B0
        # step 1 at s1: |vr - (s1 + c1)| <= min(rho M1, kap A1 + rho2 M1) + sig |s1 + c1| + alp,
        # M1 = max(|s1|, max lift[16:]). The slack is 1-Lipschitz-dominated: check s1 = lo, hi and
        # the point vr - c1 if inside [lo, hi].
        def ok(s1):
            M1 = max(abs(s1), max(lift[16:]))
            return abs(vr - (s1 + c1)) <= min(rho * M1, kap * A1 + rho2 * M1) + sig * abs(s1 + c1) + alp
        # g(s1) = bound - |error| is piecewise linear: its maximum on [lo, hi] is at an end or a kink
        ml = max(lift[16:]); sw = kap * A1 / (rho - rho2)
        kinks = [vr - c1, Fr(0), ml, -ml, -c1, sw, -sw]
        cands = [lo, hi] + [x for x in kinks if lo <= x <= hi]
        feasible = any(ok(s) for s in cands)
        exact = s0 + c0 + c1
        print(f"part2 convert a={CA} path={path} kWidth={kw}: device results {vals}; model (observed "
              f"partition) {float(mobs)!r}; exact {float(exact / u):.4f} u; device error "
              f"{float((exact - vr) / u):.4f} u; step 0 allows s1 in [{float(lo / u):.4f} u, "
              f"{float(hi / u):.4f} u]; step 1 needs s1 near {float((vr - c1) / u):.4f} u; "
              f"partial values exist (DotF16Chunk holds here): {feasible}", flush=True)


if __name__ == "__main__":
    part2()
    part1()
