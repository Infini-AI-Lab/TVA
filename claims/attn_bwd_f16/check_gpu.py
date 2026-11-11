"""GPU check of the claim of `attn_bwd_f16`.

Runs the kernel's own host (`source/06-fused-attention.py`, imported by path): the forward
`attention(q, k, v, causal, sm_scale)`, then `.backward(dout)`, which launches `_attn_bwd` at lines
608-621. While the host runs, the module's `_attn_bwd` is replaced by a recorder whose
`kernel[grid](...)` keeps the launch's tensors (`q`, `arg_k`, `v`, `do`, `dq`, `dk`, `dv`, `M`,
`delta`, `sm_scale`, `CAUSAL`) and runs the real JIT function's launch unchanged; `torch.empty_like`
is made to give `dq`, `dk`, `dv`, `delta` views of longer buffers whose tails hold a sentinel.

For every head checked it computes, in float64 and exactly as `Claim.lean` states them, the pair
quantities (`sQK`, `aQK` with `lm`, `sVD`, `aVD`, `xE`, `eX`, `yE`, `eY`, `yB`, `pBr`, `pE`,
`dsE`, `p16Err`, `p16B`, `p16L` with `lift16`, `dsErr`, `dsB`, `ds16Err`, `ds16B`, `ds16L`), the
step counts `stepsKV`, `stepsQ`, the exact values `dVE`, `dKE`, `dQE` and the tolerances `dVTol`,
`dKTol`, `dQTol` (`accB`, `rhoA`, `alphaA`); it checks `HeadV.Ok`, and reports for each output the
largest `|c - E| / (T + (|E| + T)·2^-11 + 2^-25)` (`OutSpec`'s second clause), and that `c` is the
`float16` rounding of some value within `T` of `E` (its first clause: the distance from `E` to the
interval of values that round to `c` is at most `T`; that distance over `T` is reported too); and the frame: every input
unchanged bit for bit, every sentinel kept.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "06-fused-attention.py")
spec = importlib.util.spec_from_file_location("fused_attention_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = "cuda"
U32, ETA32, U16, ETA16 = 2.0 ** -24, 2.0 ** -150, 2.0 ** -11, 2.0 ** -25
REL16_32 = 34 * 2.0 ** -22                       # relDot16 32
ABS16_32 = 34 * 2.0 ** -126                      # absDot16 32
RHOA = REL16_32 + 3 * U32 * (1 + REL16_32)        # DotLoop.stepRel relDot16 32
ALPHAA = ABS16_32 * (1 + 3 * U32) + 4 * ETA32     # DotLoop.stepAbs absDot16 32
LN2F = 1453635 / 2097152                          # ln2F: float32(0.6931471824645996)
SENT16 = -12345                                   # an int16 pattern, a finite float16
SENT32 = -7.0e30

WORST = {"dV": 0.0, "dK": 0.0, "dQ": 0.0, "dV32": 0.0, "dK32": 0.0, "dQ32": 0.0, "ok": True, "at": {}, "heads": 0, "notok": 0}


def same(a, b):
    """Bitwise equality (a NaN equals itself)."""
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def lm(x):
    """`lm` (`Device.liftMag .f16`): 0 for 0, else max(|x|, 2^-14)."""
    a = x.abs()
    return torch.where(x == 0, torch.zeros_like(a), torch.clamp(a, min=2.0 ** -14))


def lift16(B):
    """`lift16`: 0 below 2^-24, else max(B, 2^-14)."""
    return torch.where(B < 2.0 ** -24, torch.zeros_like(B), torch.clamp(B, min=2.0 ** -14))


def accB(T, B):
    """`accB T B = ((1 + rhoA)^T - 1)·B + T·(1 + rhoA)^T·alphaA` (T a tensor broadcast over B)."""
    g = (1 + RHOA) ** T
    return (g - 1) * B + T * g * ALPHAA


def eSub(D, s, a, x):
    """`eSub d s a x = rH·a + aH + 3u32·(|s| + rH·a + aH + |x|) + 4eta32`."""
    rH, aH = (D + 2) * 2.0 ** -22, (D + 2) * 2.0 ** -126
    return rH * a + aH + 3 * U32 * (s.abs() + rH * a + aH + x.abs()) + 4 * ETA32


def pairs(D, sQK, aQK, L, sVD, aVD, Dl, allowed):
    """The pair quantities of `Claim.lean` on a block of pairs (all float64 tensors of one shape)."""
    xE, eX = sQK - L, eSub(D, sQK, aQK, L)
    yE, eY = sVD - Dl, eSub(D, sVD, aVD, Dl)
    yB = yE.abs() + eY
    p2 = torch.exp2(xE)
    pBr = (1 + 2.0 ** -22) * torch.exp2(xE + eX) + 2.0 ** -126
    z = torch.zeros_like(xE)
    pE = torch.where(allowed, p2, z)
    dsE = pE * yE
    p16Err = torch.where(allowed, pBr - p2 + U16 * pBr + ETA16, z)
    p16L = lift16(torch.where(allowed, (1 + U16) * pBr + ETA16, z))
    dsErr = (pBr - p2) * yB + p2 * eY + U32 * (pBr * yB) + ETA32
    dsB = (1 + U32) * (pBr * yB) + ETA32
    ds16Err = torch.where(allowed, dsErr + U16 * dsB + ETA16, z)
    ds16L = lift16(torch.where(allowed, (1 + U16) * dsB + ETA16, z))
    fits = ~allowed | ((xE + eX <= 15) & (pBr * yB <= 2.0 ** 15))
    return pE, dsE, p16Err, p16L, ds16Err, ds16L, bool(fits.all())


def f16_spacing(c):
    """Half the float16 spacing below and above each value of `c` (float16), as float64."""
    c16 = c.to(torch.float16)
    lo = (c16.double() - torch.nextafter(c16, torch.full_like(c16, -float("inf"))).double()).abs() / 2
    hi = (torch.nextafter(c16, torch.full_like(c16, float("inf"))).double() - c16.double()).abs() / 2
    return lo, hi


def judge(key, name, c, E, T):
    """`OutSpec` on a block: `c` float16 outputs, `E` exact values, `T` tolerances (float64)."""
    c64 = c.double()
    lo, hi = f16_spacing(c)
    # first clause: some v with |v - E| <= T rounds to c: c lies within half a spacing of [E-T, E+T]
    below = (E - T) - c64          # > 0 when c is below the interval
    above = c64 - (E + T)          # > 0 when c is above it
    first_ok = bool(((below <= hi) & (above <= lo)).all()) and bool(torch.isfinite(c64).all())
    # the float32 error the claim must explain: the distance from E to c's rounding interval
    # [c - lo, c + hi], over T (at most 1 exactly when the first clause holds)
    need = torch.clamp(torch.maximum((c64 - lo) - E, E - (c64 + hi)), min=0)
    r32 = float((need / T).max())
    room = E.abs() + T <= 65504
    tol16 = T + (E.abs() + T) * U16 + ETA16
    ratio = torch.where(room, (c64 - E).abs() / tol16, torch.zeros_like(E))
    r = float(ratio.max())
    i = int(ratio.argmax())
    at = (name, float((c64 - E).abs().flatten()[i]), float(tol16.flatten()[i]), float(E.flatten()[i]))
    return r, first_ok, at, r32


def commit(key, name, r, first_ok, at, r32):
    WORST[key + "32"] = max(WORST[key + "32"], r32)
    if not first_ok:
        print(f"  {name}: {key}: OUTSPEC FIRST CLAUSE FAILS")
        WORST["ok"] = False
    if r > WORST[key]:
        WORST[key] = r
        WORST["at"][key] = at


def check_head(name, D, causal, s, Q, K, V, dO, L, Dl, dq, dk, dv, chunk):
    """`Post` for one head: `Q`, `K`, `V`, `dO` `[N, D]` float16; `L`, `Dl` `[N]` float32;
    outputs `[N, D]` float16."""
    N = Q.shape[0]
    chunk = min(chunk, max(64, 2 ** 21 // N))   # each [chunk, N] float64 block at most 16 MiB
    Qd, Kd, Vd, dOd = Q.double(), K.double(), V.double(), dO.double()
    lQ, lK, lV, ldO = lm(Qd), lm(Kd), lm(Vd), lm(dOd)
    Ld, Dd = L.double(), Dl.double()
    if not (all(bool(torch.isfinite(x).all()) for x in (Qd, Kd, Vd, dOd, Ld, Dd))
            and bool((Ld.abs() <= 2.0 ** 126).all()) and bool((Dd.abs() <= 2.0 ** 126).all())):
        WORST["notok"] += 1
        return {}
    ok = True
    r = {"dV": 0.0, "dK": 0.0, "dQ": 0.0}
    res = []
    idx = torch.arange(N, device=DEV)
    # key rows n (dV, dK): pairs [n, m]
    for a in range(0, N, chunk):
        n = idx[a:a + chunk]
        allowed = torch.ones((n.numel(), N), dtype=torch.bool, device=DEV) if not causal else \
            (n[:, None] <= idx[None, :])
        pE, dsE, p16Err, p16L, ds16Err, ds16L, fits = pairs(
            D, Kd[a:a + chunk] @ Qd.T, lK[a:a + chunk] @ lQ.T, Ld[None, :],
            Vd[a:a + chunk] @ dOd.T, lV[a:a + chunk] @ ldO.T, Dd[None, :], allowed)
        ok = ok and fits
        T = (8 + (N - 128 * (n // 128) - 128) // 32) if causal else torch.full_like(n, N // 32)
        T = T.double()[:, None]
        dVE = pE @ dOd
        dVTol = p16Err @ dOd.abs() + accB(T, p16L @ ldO)
        dKS = dsE @ Qd
        dKAcc = ds16Err @ Qd.abs() + accB(T, ds16L @ lQ)
        dKE = s * dKS
        dKTol = abs(s) * dKAcc + U32 * (abs(s) * (dKS.abs() + dKAcc)) + ETA32
        res.append(("dV", judge("dV", name, dv[a:a + chunk], dVE, dVTol)))
        res.append(("dK", judge("dK", name, dk[a:a + chunk], dKE, dKTol)))
    # query rows m (dQ): pairs [m, n] (the transpose)
    for a in range(0, N, chunk):
        m = idx[a:a + chunk]
        allowed = torch.ones((m.numel(), N), dtype=torch.bool, device=DEV) if not causal else \
            (idx[None, :] <= m[:, None])
        pE, dsE, p16Err, p16L, ds16Err, ds16L, fits = pairs(
            D, Qd[a:a + chunk] @ Kd.T, lQ[a:a + chunk] @ lK.T, Ld[a:a + chunk, None],
            dOd[a:a + chunk] @ Vd.T, ldO[a:a + chunk] @ lV.T, Dd[a:a + chunk, None], allowed)
        ok = ok and fits
        T = (8 + 128 * (m // 128) // 32) if causal else torch.full_like(m, N // 32)
        T = T.double()[:, None]
        dQS = dsE @ Kd
        dQAcc = ds16Err @ Kd.abs() + accB(T, ds16L @ lK)
        dQE = LN2F * dQS
        dQTol = LN2F * dQAcc + U32 * (LN2F * (dQS.abs() + dQAcc)) + ETA32
        res.append(("dQ", judge("dQ", name, dq[a:a + chunk], dQE, dQTol)))
    if not ok:
        WORST["notok"] += 1  # HeadV.Ok fails: Post says nothing of this head's accuracy
        return {}
    WORST["heads"] += 1
    for key, (rr, first_ok, at, r32) in res:
        commit(key, name, rr, first_ok, at, r32)
        r[key] = max(r[key], rr)
    return r


class Recorder:
    """Stands for the module's `_attn_bwd` while the host runs."""

    def __init__(self, real):
        self.real, self.calls = real, []

    def __getitem__(self, grid):
        launcher = self.real[grid]

        def run(Q, K, V, sm_scale, DO, DQ, DK, DV, M, D, *a, **kw):
            snap = [x.detach().cpu() for x in (Q, K, V, DO, M, D)]   # on the host, for the frame check
            out = launcher(Q, K, V, sm_scale, DO, DQ, DK, DV, M, D, *a, **kw)
            torch.cuda.synchronize()
            self.calls.append(dict(t=(Q, K, V, DO, M, D), out=(DQ, DK, DV), snap=snap, sm=sm_scale,
                                   causal=kw["CAUSAL"], D=kw["HEAD_DIM"]))
            return out

        return run


def host_forward(q, k, v, causal, sm):
    """The forward's `o` and `M` computed in float64 (for head dims whose forward kernel the H200
    rejects): `M` the base-2 log-sum-exp of `q·kᵀ·sm·RCP_LN2` over the allowed keys (`float32`),
    `o` the softmax-weighted mean of `v` (`float16`)."""
    N = q.shape[2]
    S = (q.double() @ k.double().transpose(-1, -2)) * (sm * 1.4426950408889634)
    if causal:
        S = S.masked_fill(torch.ones(N, N, dtype=torch.bool, device=DEV).triu(1), -float("inf"))
    mx = S.max(-1, keepdim=True).values
    P = torch.exp2(S - mx)
    l = P.sum(-1, keepdim=True)
    M = (mx + torch.log2(l))[..., 0].float()
    o = ((P / l) @ v.double()).half()
    return o, M


def run_case(name, Z, H, N, D, causal, sm, std, heads=None, chunk=1024, qk_scale=1.0, seed=20,
             fwd=True):
    """One launch: inputs drawn as the test (`normal(0, std)`) or the benchmark (`randn`, std 1)."""
    torch.manual_seed(seed)
    dtype = torch.float16
    q = (torch.empty((Z, H, N, D), dtype=dtype, device=DEV).normal_(0.0, std * qk_scale)).requires_grad_()
    k = (torch.empty((Z, H, N, D), dtype=dtype, device=DEV).normal_(0.0, std * qk_scale)).requires_grad_()
    v = (torch.empty((Z, H, N, D), dtype=dtype, device=DEV).normal_(0.0, std)).requires_grad_()
    dout = torch.randn_like(q)
    rec = Recorder(mod._attn_bwd)
    real_empty_like = torch.empty_like
    tails = []

    def empty_like(x, *a, **kw):
        if x.dtype in (torch.float16, torch.float32) and not a and not kw:
            n = x.numel()
            store = torch.empty(n + 512, dtype=x.dtype, device=x.device)
            if x.dtype == torch.float16:
                store.view(torch.int16).fill_(SENT16)
            else:
                store.fill_(SENT32)
            tails.append(store)
            return store[:n].view(x.shape)
        return real_empty_like(x, *a, **kw)

    mod._attn_bwd = rec
    try:
        if fwd:
            o = mod.attention(q, k, v, causal, sm, False)
        else:
            # `_attention.backward` itself, with the forward's saved tensors computed on the host
            ob, Mb = host_forward(q.detach(), k.detach(), v.detach(), causal, sm)
            ctx = types.SimpleNamespace(saved_tensors=(q.detach(), k.detach(), v.detach(), ob, Mb),
                                        sm_scale=sm, HEAD_DIM=D, causal=causal)
            o = None
        torch.empty_like = empty_like
        try:
            if fwd:
                o.backward(dout)
            else:
                mod._attention.backward(ctx, dout)
        finally:
            torch.empty_like = real_empty_like
    except Exception as ex:  # rejected on the host (e.g. out of resources): no outcome
        mod._attn_bwd = rec.real
        torch.empty_like = real_empty_like
        print(f"{name}: rejected on the host ({type(ex).__name__}: {str(ex).splitlines()[0][:80]})")
        return
    finally:
        mod._attn_bwd = rec.real
    torch.cuda.synchronize()
    call, = rec.calls
    if not all(same(a, b.detach().cpu()) for a, b in zip(call["snap"], call["t"])):
        print(f"  {name}: FRAME VIOLATED (an input changed)")
        WORST["ok"] = False
    for st in tails:
        tail = st[-512:]
        good = (tail.view(torch.int16) == SENT16).all() if st.dtype == torch.float16 else (tail == SENT32).all()
        if not bool(good):
            print(f"  {name}: FRAME VIOLATED (a byte past an output changed)")
            WORST["ok"] = False
    Qt, Kt, Vt, DOt, Mt, Dt = call["t"]
    dq, dk, dv = call["out"]
    assert call["causal"] == causal and call["D"] == D
    s = float(torch.tensor(call["sm"], dtype=torch.float32))  # smF: the float32 the kernel receives
    G = Z * H
    hs = range(G) if heads is None else sorted(set(h % G for h in heads))
    res = {"dV": 0.0, "dK": 0.0, "dQ": 0.0}
    with torch.no_grad():
        for g in hs:
            z, h = divmod(g, H)
            r = check_head(name, D, causal, s, Qt[z, h], Kt[z, h], Vt[z, h], DOt[z, h], Mt[z, h], Dt[z, h],
                           dq[z, h], dk[z, h], dv[z, h], chunk)
            for key in r:
                res[key] = max(res[key], r[key])
    print(f"{name}: heads checked {len(hs)}/{G}; err/tol dV {res['dV']:.3e} dK {res['dK']:.3e} "
          f"dQ {res['dQ']:.3e}")
    del q, k, v, dout, o, rec, call, tails
    torch.cuda.empty_cache()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "test"):
        # the test's launches (bwd, triton-fp16): std 0.5, sm_scale 0.5
        for Z in (1, 4):
            for H in (2, 48):
                for N in (128, 1024, 4096):
                    for D in (64, 128):
                        for causal in (False, True):
                            heads = None if Z * H * N <= 2 * 4096 else [0, 1, 37, Z * H - 1]
                            run_case(f"test Z{Z} H{H} N{N} D{D} causal={causal}", Z, H, N, D, causal,
                                     0.5, 0.5, heads=heads)
    if which in ("all", "bench"):
        # the benchmark's sizes (randn inputs, sm_scale 1.3), at BATCH 1, N_HEADS 2 (the full
        # BATCH 4, N_HEADS 32 tensors at N_CTX 2^14 exceed the 4 GB budget; heads are independent)
        for e in range(10, 15):
            for D in (64, 128):
                for causal in (False, True):
                    run_case(f"bench N{2 ** e} D{D} causal={causal}", 1, 2, 2 ** e, D, causal, 1.3, 1.0,
                             heads=[0], chunk=512)
        # one launch at the benchmark's head count
        run_case("bench Z4 H32 N1024 D128 causal=True", 4, 32, 1024, 128, True, 1.3, 1.0, heads=[0, 127])
    if which in ("all", "edge"):
        # edges of Inst.ok: the other head dims the forward admits
        # (the forward kernel fails a static assertion at 16 and 32, so the file's own path never
        # reaches the backward there; `_attention.backward` is called on host-computed `o`, `M`)
        for D in (16, 32):
            for causal in (False, True):
                run_case(f"edge D{D} causal={causal}", 1, 2, 1024, D, causal, 0.5, 0.5, fwd=False)
        for causal in (False, True):
            run_case(f"edge D256 causal={causal}", 1, 2, 1024, 256, causal, 0.5, 0.5)
        run_case("edge ZH1 N128", 1, 1, 128, 64, True, 0.5, 0.5)
        # edges of the values: sharper attention (scores 4x), a small sm_scale
        run_case("pre-sharp", 1, 2, 2048, 128, True, 1.3, 2.0, heads=[0, 1])
        run_case("pre-smallscale", 1, 2, 1024, 64, False, 2.0 ** -6, 1.0)
    keys = ("dV", "dK", "dQ")
    print("WORST float32 part (distance from E to c's rounding interval) / T: " +
          ", ".join(f"{k} {WORST[k + '32']:.3e}" for k in keys))
    print("WORST: " + ", ".join(f"{k} {WORST[k]:.3e} at {WORST['at'].get(k)}" for k in keys) +
          f"; heads checked {WORST['heads']}, of which HeadV.Ok fails on {WORST['notok']}; "
          f"first clause, finiteness and frame {'hold' if WORST['ok'] else 'FAIL'}")
    print(f"peak device memory: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    sys.exit(0 if WORST["ok"] and max(WORST[k] for k in keys) < 1 else 1)


if __name__ == "__main__":
    main()
