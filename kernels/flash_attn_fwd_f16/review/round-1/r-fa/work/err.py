from fractions import Fraction as F
import math, sys

def consts(N, HD, BN, R, V, C, absE=F(1, 2**149)):
    u = F(1, 2**24); eta = F(1, 2**150); u16 = F(1, 2**11); eta16 = F(1, 2**25)
    relE = F(1, 2**22)
    relL = F(1, 2**22); absL = F(1, 2**22); relDiv = F(1, 2**22); absDiv = F(1, 2**148)
    relDot = lambda K: (K + 2) * F(1, 2**22)
    absDot = lambda K: (K + 2) * F(1, 2**126)
    R = F(R); V = F(V); C = F(C)
    S = HD * R**2
    e1 = relDot(HD) * S + absDot(HD)
    Q = S + e1
    Mmu = C * Q * (1 + u) + eta
    d = C * e1 + u * (C * Q) + eta
    dx = d + 6 * u * Mmu + 4 * eta
    kp = dx + relE * (1 + dx)
    g = 2 * BN * u
    ks = kp + g * (1 + kp)
    ns = BN * absE * (1 + g) + 2 * BN * eta
    da = 2 * u * Mmu + eta
    ka = da + relE * (1 + da)
    Lmax = 2 * N
    T = N // BN
    nl = 3 * absE * Lmax + 2 * ns + 4 * eta
    rl0 = ks + 6 * u + 2 * nl
    dl = 2 * ka + 12 * u + 2 * nl
    rL = rl0 + T * dl
    k16 = kp + u16 * (1 + kp)
    n16 = absE * (1 + u16) + eta16
    th = relDot(BN)
    na = 3 * absE * V * Lmax + 2 * eta + 2 * V * BN * n16 + absDot(BN)
    rv0 = V * (k16 + 2 * th) + 2 * na
    dv = V * (3 * ka + 2 * u + 3 * th) + 2 * na
    rV = rv0 + T * dv
    ok = {
        'CQ<=2^60': C * Q <= 2**60,
        'dx<=1/4': dx <= F(1, 4),
        'ka<=1/4': ka <= F(1, 4),
        'rL<=1/2': rL <= F(1, 2),
        'rV<=V': rV <= V,
        'R<=65504': R <= 65504, 'V<=2^13': V <= 2**13,
    }
    if rL < 1:
        eO = (rV + V * rL) / (1 - rL)
        errO = eO + relDiv * (V + eO) + absDiv + u16 * ((V + eO) * (1 + relDiv) + absDiv) + eta16
        eL = 2 * rL / (1 - rL)
        Lg = math.log2(2 * N) + float(eL)
        errM = float(u) * (float(Mmu) + (1 + float(relL)) * Lg + float(absL)) + float(eta) + float(relL) * Lg + float(absL) + float(eL)
    else:
        eO = errO = errM = None
    return dict(e1=float(e1), d=float(d), Mmu=float(Mmu), rL=float(rL), rV=float(rV),
                errO=None if errO is None else float(errO), errM=errM, ok=all(ok.values()),
                fails=[k for k, v in ok.items() if not v])

def show(tag, **kw):
    r = consts(**kw)
    eo = r['errO']
    print(f"{tag:42s} ok={r['ok']!s:5s} fails={r['fails']} e1={r['e1']:.3g} Mmu={r['Mmu']:.3g} rL={r['rL']:.3g} rV={r['rV']:.3g} errO={eo if eo is None else f'{eo:.4g}'} errO/V={'-' if eo is None else f'{eo/kw['V']:.3%}'} errM={r['errM'] if r['errM'] is None else f'{r['errM']:.4g}'}")

if __name__ == '__main__':
    import itertools
    # doc table
    show('doc N1024 HD64 BN64 R4 V4 C.2', N=1024, HD=64, BN=64, R=4, V=4, C=0.2)
    show('doc N1024 HD64 BN128 R4 V4 C.2', N=1024, HD=64, BN=128, R=4, V=4, C=0.2)
    show('doc N1024 HD64 BN64 R2 V2 C.2', N=1024, HD=64, BN=64, R=2, V=2, C=0.2)
    show('doc N1024 HD64 BN64 R8 V4 C.2', N=1024, HD=64, BN=64, R=8, V=4, C=0.2)
    # realistic: HD128, sm_scale 1/sqrt(128) -> c = 0.1275
    c128 = 1/math.sqrt(128)*1.4426950408889634*1.0001
    c64 = 1/8*1.4426950408889634*1.0001
    c256 = 1/16*1.4426950408889634*1.0001
    for N in [1024, 4096, 16384, 65536, 131072, 1 << 20]:
        for R in [2, 4, 8, 16]:
            show(f'HD128 BN128 N{N} R{R} V4', N=N, HD=128, BN=128, R=R, V=4, C=c128)
    for N in [4096, 65536]:
        for R in [4, 8, 16]:
            show(f'HD64 BN64 N{N} R{R} V4', N=N, HD=64, BN=64, R=R, V=4, C=c64)
            show(f'HD256 BN64 N{N} R{R} V4', N=N, HD=256, BN=64, R=R, V=4, C=c256)
    # tutorial test inputs: normal(0, 0.5) fp16 -> |x| up to ~ 3 (5 sigma ~2.5); sm_scale 0.5
    for HD in [64, 128]:
        c = 0.5 * 1.4426950408889634 * 1.0001
        for N in [128, 1024, 4096]:
            show(f'tutorial test HD{HD} N{N} R3 V3 sm.5', N=N, HD=HD, BN=64, R=3, V=3, C=c)
    # largest N for which Bnd.Ok holds, R=4, HD=128
    # largest R at HD=128, N=16384
    for R in [16, 20, 24, 28, 32]:
        show(f'HD128 N16384 R{R}', N=16384, HD=128, BN=128, R=R, V=4, C=c128)
    # absE = 2^-126 (ftz) at the same points
    show('ftz absE: HD128 N131072 R8', N=131072, HD=128, BN=128, R=8, V=4, C=c128, absE=F(1, 2**126))
    show('149 absE: HD128 N131072 R8', N=131072, HD=128, BN=128, R=8, V=4, C=c128)
