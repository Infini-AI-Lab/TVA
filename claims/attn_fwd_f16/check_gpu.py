"""GPU check of the prepared claim of `attn_fwd_f16` (claims/README.md, check 4).

Runs the kernel's own host (`source/06-fused-attention.py`, imported by path:
`_attention.forward`, the function `attention` applies, called with a small context object so that
the log-sum-exp `M` it saves can be read) on the launches of SETUP.md (the test's and the
benchmark's `float16` launches) and on launches at the edges of `Inst.ok` and of `HeadOk`; computes
`Post`'s reference and tolerances from the inputs in float64 exactly as `Claim.lean` states them
(`score`, `scoreMag` with `liftMag`, the row's smallest `X`, `attnRef`, `attnAbs`, `vAbsSum`,
`attnMag`, `lseRef`, `epsO`, `epsM`); and reports, for `o` and `M`, the largest
error / tolerance, the tolerance / |output| (median and worst) on the file's launches, for `Post`'s
tolerance and for the fine bound of README Tolerances (which must lie below it), that every element is written and finite, and the frame: the host's
`torch.empty_like` and `torch.empty` are made to return `o` and `M` as views into larger NaN-filled
buffers, whose guard bands must stay unchanged, and the inputs must be unchanged.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py [--quick]
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "06-fused-attention.py")
spec = importlib.util.spec_from_file_location("fused_attention_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = "cuda"
QUICK = "--quick" in sys.argv

# ---------------------------------------------------------------- the tolerances (Claim.lean)
u32 = 2.0 ** -24


def relDot16(K):
    return (K + 2) * 2.0 ** -22


def eps_post(D, N, X):
    """`epsO D N X` and `epsM D X` of Claim.lean ("The tolerances"), in float64; X a tensor."""
    return (D * X + N + 2 ** 10) * 2.0 ** -20, (D * X + 2 ** 10) * 2.0 ** -20


def fine_tolerances(D, N, X):
    """The fine bound of README Tolerances (`tolO`, `tolS`, `tolF`, `tolM`, steps 1-5), in float64;
    X a tensor. Only for the comparison: `Post` states `eps_post`."""
    yb = 1.5 * X
    expErr = (relDot16(D) + 2.0 ** -20) * yb + 2.0 ** -50
    psiP = (1 + expErr) * (1 + 2.0 ** -22) - 1
    psi16 = (1 + psiP) * (1 + 2.0 ** -11) - 1
    psiL = (1 + u32) ** 63 * (1 + psiP) - 1
    drift = 1 + 4 * u32 * (yb + expErr)
    stepAcc = (1 + relDot16(64)) * (1 + 2.0 ** -150) * ((1 + 2.0 ** -22) * (1 + u32))
    stepL = (1 + 3 * u32) * (1 + 2.0 ** -150) * (1 + 2.0 ** -22)
    nb = N // 64
    facc = (1 + psi16) * drift * stepAcc ** nb
    fl = (1 + psiL) * drift * stepL ** nb
    absL, absA, kapA = 2.0 ** -100, 2.0 ** -37, 2.0 ** -29
    omega0 = 2.0 ** -25 + 2.0 ** -125 + 66 * 2.0 ** -36
    wlow = 1 - expErr
    betaL = (fl - 1) + fl * (nb * absL) / wlow
    denQ = 1 - betaL
    relQ = (facc - 1 + betaL) / denQ
    absQS = facc * omega0 / (wlow * denQ)
    absQ0 = facc * (nb * absA) / (wlow * denQ) + facc * kapA / denQ
    thetaW = 2.0 ** -20 * X
    lastRel = 2.0 ** -11 * (1 + 2.0 ** -22) + 2.0 ** -22
    tolO = lastRel * (1 + relQ) * (1 + thetaW) + relQ * (1 + thetaW) + thetaW
    tolS = absQS * (1 + lastRel)
    tolF = absQ0 * (1 + lastRel) + 2.0 ** -11 * 2.0 ** -148 + 2.0 ** -25 + 2.0 ** -148
    tolM = (1 + u32) * (17 * 2.0 ** -22 + 2 * betaL + 2.0 ** -21 * X + 2.0 ** -149) + 2.0 ** -150
    return tolO, tolS, tolF, tolM


def lift(x):
    """`Rules.Num.liftMag .f16`: 0 for 0, else max(|x|, 2^-14)."""
    a = x.abs()
    return torch.where(x == 0, torch.zeros_like(a), torch.clamp(a, min=2.0 ** -14))


# ---------------------------------------------------------------- running the host
class Ctx:
    def save_for_backward(self, *ts):
        self.saved = ts


GUARD = 4096
GUARDED = []


def run_host(q, k, v, causal, sm):
    """`_attention.forward` with `o` and `M` allocated inside larger NaN-filled buffers: every
    element must be written, and the guard bands around them must stay NaN (the frame). With
    `warp_specialize = False` the host passes host `TensorDescriptor`s, and the allocator it sets with
    `triton.set_allocator` is never called: `o` and `M` are the only tensors allocated."""
    real_empty_like, real_empty = torch.empty_like, torch.empty
    GUARDED.clear()

    def guarded(numel, dtype, shape, device):
        big = real_empty(numel + 2 * GUARD, dtype=dtype, device=device).fill_(float("nan"))
        GUARDED.append(big)
        return big[GUARD:GUARD + numel].view(shape)

    def empty_like(t, *a, **kw):
        return guarded(t.numel(), kw.get("dtype", t.dtype), t.shape, t.device)

    def empty(shape, *a, dtype=None, device=None, **kw):
        shape = tuple(shape) if isinstance(shape, (tuple, list, torch.Size)) else (shape,)
        return guarded(math.prod(shape), dtype or torch.float32, shape, device)

    torch.empty_like, torch.empty = empty_like, empty
    try:
        ctx = Ctx()
        o = mod._attention.forward(ctx, q, k, v, causal, sm, False)
    finally:
        torch.empty_like, torch.empty = real_empty_like, real_empty
    torch.cuda.synchronize()
    return o, ctx.saved[4]


def guards_ok():
    return all(bool(torch.isnan(g[:GUARD].float()).all() and torch.isnan(g[-GUARD:].float()).all())
               for g in GUARDED)


# ---------------------------------------------------------------- checking Post on one head
def check_head(q, k, v, o, M, sigma, causal):
    """Post on one head (`[N, D]` slices). Returns (ratioO, ratioM, finite, headok, Xmax)."""
    N, D = q.shape
    q64, k64, v64 = q.double(), k.double(), v.double()
    lq, lk = lift(q64), lift(k64)
    fin = bool(torch.isfinite(o.float()).all() and torch.isfinite(M).all())
    headok = bool(torch.isfinite(q64).all() and torch.isfinite(k64).all() and torch.isfinite(v64).all()
                  and (v64.abs() <= 2.0 ** 15).all())
    rO = rM = 0.0
    Xmax = 0.0
    rows_per_chunk = max(64, 2 ** 23 // N)   # each [rows, N] float64 block at most 64 MiB
    vabs = v64.abs()
    for r0 in range(0, N, rows_per_chunk):
        r1 = min(N, r0 + rows_per_chunk)
        s = q64[r0:r1] @ k64.T                         # score
        smag = sigma * (lq[r0:r1] @ lk.T)              # sigma * scoreMag
        headok = headok and bool((smag <= 2.0 ** 10).all())
        if causal:
            n = torch.arange(r0, r1, device=q.device)[:, None]
            j = torch.arange(N, device=q.device)[None, :]
            keep = j <= n
        else:
            keep = torch.ones(r1 - r0, N, dtype=torch.bool, device=q.device)
        X = torch.where(keep, smag, torch.zeros_like(smag)).amax(1)   # the row's smallest X
        Xmax = max(Xmax, float(X.max()))
        z = torch.where(keep, sigma * s, torch.full_like(s, -math.inf))
        zmax = z.amax(1, keepdim=True)
        w = torch.exp(z - zmax)                        # weight / exp(zmax)
        W = w.sum(1, keepdim=True)
        ref = (w @ v64) / W                            # attnRef
        absr = (w @ vabs) / W                          # attnAbs
        vsum = keep.double() @ vabs                    # vAbsSum
        lse = (zmax[:, 0] + torch.log(W[:, 0])) / math.log(2.0)   # lseRef (log2)
        eO, eM = eps_post(D, N, X)
        nkeys = keep.double().sum(1, keepdim=True)
        mag = torch.maximum(absr, vsum / nkeys)        # attnMag
        errO = (o[r0:r1].double() - ref).abs()
        boundO = eO[:, None] * mag + 2.0 ** -24
        errM = (M[r0:r1].double() - lse).abs()
        boundM = eM
        rO = max(rO, float((errO / boundO).max()))
        rM = max(rM, float((errM / boundM).max()))
        tO, tS, tF, tM = fine_tolerances(D, N, X)
        fineO = tO[:, None] * absr + tS[:, None] * vsum + tF[:, None]
        fineM = tM + u32 * lse.abs()
        global FINE_OK
        FINE_OK = FINE_OK and bool((fineO <= boundO).all()) and bool((fineM <= boundM).all())
        if STATS is not None:
            nz = ref.abs() > 0
            for key, b, r in (("o", boundO, ref.abs()), ("o fine", fineO, ref.abs()),
                              ("M", boundM[:, None], lse.abs()[:, None]), ("M fine", fineM[:, None], lse.abs()[:, None])):
                ratio = (b.expand_as(r) / r)[(r > 0).expand_as(r) if key[0] == "M" else nz]
                STATS.setdefault(key, []).append(ratio.flatten()[::max(1, ratio.numel() // 20000)].cpu())
                WORSTRATIO[key] = max(WORSTRATIO.get(key, 0.0), float(ratio.max()))
    return rO, rM, fin, headok, Xmax


WORST = {"o": 0.0, "M": 0.0}
ALL_OK = True
FINE_OK = True      # the fine bound of README Tolerances lies below Post's tolerance everywhere
STATS = None        # tolerance / |output| samples on the file's launches (a dict while they run)
WORSTRATIO = {}


def check_launch(name, Z, H, N, D, causal, sm, make, heads=None, Xtarget=None):
    """One launch: inputs from `make`, the host, then Post on the chosen heads (all by default).
    With `Xtarget`, `sm_scale` is chosen so that the largest `sm_scale·scoreMag` is `Xtarget`."""
    global ALL_OK
    q, k, v = make(Z, H, N, D)
    if Xtarget is not None:
        mx = max(float((lift(q[z, h].double()) @ lift(k[z, h].double()).T).max())
                 for z in range(Z) for h in range(H))
        sm = Xtarget / mx
    q0, k0, v0 = q.cpu(), k.cpu(), v.cpu()   # the inputs' copies for the frame check, on the host
    try:
        o, M = run_host(q, k, v, causal, sm)
    except Exception as e:  # rejected on the host (compilation): no outcome
        print(f"{name:34s} Z={Z} H={H} N={N} D={D} causal={int(causal)}: "
              f"rejected on the host ({type(e).__name__}): no outcome", flush=True)
        return
    frame = bool(torch.equal(q.cpu(), q0) and torch.equal(k.cpu(), k0) and torch.equal(v.cpu(), v0)) and guards_ok() \
        and len(GUARDED) == 2
    allfin = all(bool(torch.isfinite(o[z]).all()) for z in range(Z)) and bool(torch.isfinite(M).all())   # every element written
    hs = range(Z * H) if heads is None else heads
    rO = rM = Xm = 0.0
    ok = frame and allfin
    for bh in hs:
        z, h = divmod(bh, H)
        a, b, c, f, x = check_head(q[z, h], k[z, h], v[z, h], o[z, h], M[z, h], sm, causal)
        if not f:
            print(f"  head {bh} is not covered (HeadOk false): only 'written' is checked", flush=True)
            continue
        ok = ok and c
        rO, rM, Xm = max(rO, a), max(rM, b), max(Xm, x)
    ok = ok and rO < 1 and rM < 1
    ALL_OK = ALL_OK and ok
    WORST["o"], WORST["M"] = max(WORST["o"], rO), max(WORST["M"], rM)
    print(f"{name:34s} Z={Z} H={H} N={N} D={D} causal={int(causal)} sm={sm:.4g}: "
          f"o {rO:.3f}  M {rM:.3f}  X<={Xm:.1f}  written+finite={allfin} frame(guards+inputs)={frame} "
          f"{'ok' if ok else 'FAIL'}", flush=True)
    del q, k, v, o, M, q0, k0, v0
    torch.cuda.empty_cache()


def normal(std):
    def make(Z, H, N, D):
        g = torch.Generator(device=DEV).manual_seed(20)
        mk = lambda: torch.empty((Z, H, N, D), dtype=torch.float16, device=DEV).normal_(0.0, std, generator=g)
        return mk(), mk(), mk()
    return make


def randn(Z, H, N, D):
    g = torch.Generator(device=DEV).manual_seed(1)
    mk = lambda: torch.randn((Z, H, N, D), dtype=torch.float16, device=DEV, generator=g)
    return mk(), mk(), mk()


def custom(fq, fk, fv):
    def make(Z, H, N, D):
        g = torch.Generator(device=DEV).manual_seed(7)
        shape = (Z, H, N, D)
        return (fq(shape, g).half().contiguous(), fk(shape, g).half().contiguous(),
                fv(shape, g).half().contiguous())
    return make


def report_stats():
    """Tolerance / |output| over the file's launches (test and benchmark): `Post`'s and the fine one's."""
    global STATS
    for key in ("o", "o fine", "M", "M fine"):
        x = torch.cat(STATS[key])
        print(f"tolerance/|output| {key:7s}: median {float(x.median()):.3g}  worst {WORSTRATIO[key]:.3g}")
    STATS = None


def main():
    global STATS
    torch.manual_seed(0)
    STATS = {}
    # --- the test (source lines 631-695): every float16 forward it runs (sm_scale 0.5, normal(0, 0.5))
    for Z in [1, 4]:
        for H in [2, 48]:
            for N in [128, 1024, 4096]:
                for D in [64, 128]:
                    for causal in [False, True]:
                        if QUICK and (Z, H) != (1, 2):
                            continue
                        heads = None if Z * H <= 8 else [0, 1, Z * H // 2, Z * H - 1]
                        check_launch("test", Z, H, N, D, causal, 0.5, normal(0.5), heads)
    # --- the benchmark (lines 698-780): Z=4, H=32, N=2^10..2^14, randn, sm_scale 1.3
    for i in range(10, 15):
        N = 2 ** i
        for D in [64, 128]:
            for causal in [True, False]:
                if QUICK and i > 11:
                    continue
                heads = [0, 77, 127] if N >= 8192 else [0, 1, 64, 127]
                check_launch("bench", 4, 32, N, D, causal, 1.3, randn, heads)
    report_stats()
    # --- edges of Inst.ok
    check_launch("edge D=256", 1, 2, 1024, 256, False, 0.5, normal(0.5))
    check_launch("edge D=256 causal", 1, 2, 1024, 256, True, 0.5, normal(0.5))
    check_launch("edge N=2^16 D=256", 1, 1, 2 ** 16, 256, False, 1 / 16, normal(1.0), heads=[0])
    check_launch("edge N=2^16 causal", 1, 1, 2 ** 16, 64, True, 0.125, normal(1.0), heads=[0])
    check_launch("edge Z*H=4096", 64, 64, 128, 64, True, 0.5, normal(0.5), heads=[0, 2049, 4095])
    check_launch("edge sm=2^-60", 1, 2, 256, 64, False, 2.0 ** -60, normal(1.0))
    # --- edges of HeadOk (the values)
    big = lambda s, g: torch.randn(s, generator=g, device=DEV) * 3.0
    check_launch("edge X=1000 (scores)", 1, 1, 512, 128, False, None,
                 custom(big, big, lambda s, g: torch.randn(s, generator=g, device=DEV)), Xtarget=1000.0)
    check_launch("edge X=1000 causal", 1, 1, 512, 128, True, None,
                 custom(big, big, lambda s, g: torch.randn(s, generator=g, device=DEV)), Xtarget=1000.0)
    check_launch("edge X=1000 D=256 N=4096", 1, 1, 4096, 256, False, None,
                 custom(big, big, lambda s, g: torch.randn(s, generator=g, device=DEV)), Xtarget=1000.0)
    check_launch("edge |v|=2^15", 1, 2, 1024, 64, False, 0.5,
                 custom(lambda s, g: torch.randn(s, generator=g, device=DEV) * 0.5,
                        lambda s, g: torch.randn(s, generator=g, device=DEV) * 0.5,
                        lambda s, g: torch.full(s, 32768.0, device=DEV) * torch.sign(torch.randn(s, generator=g, device=DEV))))
    check_launch("edge uniform, v=1", 1, 1, 2 ** 14, 64, False, 0.5,
                 custom(lambda s, g: torch.zeros(s, device=DEV), lambda s, g: torch.zeros(s, device=DEV),
                        lambda s, g: torch.ones(s, device=DEV)))
    check_launch("edge uniform, v=1/3", 1, 1, 2 ** 14, 64, True, 0.5,
                 custom(lambda s, g: torch.zeros(s, device=DEV), lambda s, g: torch.zeros(s, device=DEV),
                        lambda s, g: torch.full(s, 1 / 3, device=DEV)))
    check_launch("edge subnormal q,k,v", 1, 2, 512, 64, False, 0.5,
                 custom(lambda s, g: torch.randn(s, generator=g, device=DEV) * 2.0 ** -16,
                        lambda s, g: torch.randn(s, generator=g, device=DEV),
                        lambda s, g: torch.randn(s, generator=g, device=DEV) * 2.0 ** -18))
    print(f"\nworst error / tolerance: o {WORST['o']:.4f}, M {WORST['M']:.4f}; "
          f"fine bound below Post's: {FINE_OK}; {'ALL OK' if ALL_OK and FINE_OK else 'SOME FAILED'}")
    print(f"peak device memory: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    return 0 if ALL_OK and FINE_OK else 1


if __name__ == "__main__":
    sys.exit(main())
