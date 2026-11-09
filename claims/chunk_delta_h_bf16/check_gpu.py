"""GPU check of the prepared claim `Claims.ChunkDeltaH` (Claim.lean).

Runs the kernel's own host (`source/chunk_delta_h.py`, imported by path) on the launches SETUP.md
names and on launches at the edges of `Inst.ok`, and checks `Post` as it is stated, computed in
float64 from the host's inputs:

* both layouts of the state (`STATE_V_FIRST` false and true: `h`, `h0`, `ht` as `[K, V]` or
  `[V, K]`, compared in the claim's `hIdx` / `sIdx` order), with and without the gate (`USE_G`);
* `Post`'s float32 states `Y t` (the existential of each column): the witness is the kernel's own
  state, read by a copy of the kernel with one more store per state block (`instrumented`: the
  `float32` `b_h` stored next to each `bfloat16` store of line 185 …, run with the final state); its
  `h`, `v_new` and `ht` are checked to equal the host's bit for bit;
* `Y 0` equal to the initial state (`0` without one); `h[t] = bf16(Y t)` bit for bit; per chunk,
  `||Y (t + 1) - StepY t (Y t)|| <= errStep t` (column norm), with `StepY t y = 2^g_last y +
  k^T (2^(g_last - g) (u - w h[t]))` from the stored state `h[t]` (the output, read back) and the
  tolerance `errStep` (the `bfloat16` / `float32` rounding terms, `Num.Exp2F32`, `Num.DotBF16` at
  K = 64 chained over the K blocks, the spectral norm `sKn` of `k^T`; `dotHvf` with
  `STATE_V_FIRST`); `v_new` within `tolV` of `vrefS t = u - w h[t]` element by element; `ht` equal
  to the last `Y` bit for bit; each chunk's clauses under `StOk t'` for every `t' <= t` (`h[t']`
  finite, `||h[t']|| <= 2^40`, which is checked and holds on every launch); every output element finite;
  `Vals` (the inputs' finiteness and caps, the gate bounds `g_last <= 8`, `g_last - g <= 8`,
  `padB t <= 2^100`) is checked too;
* the frame: one direct launch with the host's arguments writes `h`, `v_new`, `ht` placed between
  guard regions filled with a NaN pattern; the guards and every input stay unchanged, no output
  element keeps the pattern, and the outputs equal the host's bit for bit;
* the host's `chunk_offsets` equal the claim's `cbase`.

Rejection (negative tests) is shown without a witness: `Post` needs `bf16(Y t) = h[t]`, so `Y t`
lies in the box of `float32` values that round to `h[t]` (a point at `t = 0`, `Y 0 = h0`, and at the
end, `Y = ht`); when even the nearest pair of points of two consecutive boxes is farther than
`errStep t` from one step (`box` below 1 fails), or `h[0] != bf16(h0)` as values (`-0 = +0`), or a `v_new` element is
out of its tolerance, no `Y` meets `Post`. Negative tests, in both layouts, on a launch with grouped
value heads (`H = 2`, `HV = 4`), and without the gate: one element of `h`, `v_new`, `ht` moved by 4
times its tolerance, two sequences' chunks of `h` exchanged, and the outputs of the exact
recurrence with one kernel bug (no decay of the state; a chunk's update dropped; the tokens' gate
`2^(g_last - g)` dropped; `k` read at the key head `hv % H`; every token gated from the chunk's
first token; `g_last` read one token early; `ht` the state before the last chunk; every state
multiplied by `1 + 2^-7` or `1 - 2^-7` each chunk); and a guard byte changed: each must be rejected.
Also printed (not a test): what the check finds for the exact recurrence with the state carried in
`float64` or in `bfloat16` (a witness by nearest points, a refutation, or neither).

Prints, per case, the largest error / tolerance of each output (below 1 with its margin).
Usage: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import linecache
import math
import os
import re
import types

import torch
import triton

torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "chunk_delta_h.py")
spec = importlib.util.spec_from_file_location("chunk_delta_h_src", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

DEV = "cuda"
D = torch.float64
BT, BV = 64, 32
# Claim.lean, "The tolerance": the library's constants.
U32, E32, UBF, EBF, MU = 2.0 ** -24, 2.0 ** -150, 2.0 ** -8, 2.0 ** -134, 2.0 ** -126
RD, AD = 66 * 2.0 ** -22, 66 * 2.0 ** -126  # Rules.Num.relDot16 64, absDot16 64
NAN_BF16, NAN_F32 = 0x7FC1, 0x7FC00001  # guard / sentinel patterns
REFUTE = 1 + 1e-6  # a refutation needs a lower bound above the tolerance by this factor


def instrumented():
    """The kernel with one more store after each `bfloat16` store of the state (lines 185, 193,
    201, 209): the `float32` block itself, into `hf` (laid out as `h`). Built in memory from
    source/; nothing else changes."""
    text = open(SRC).read()
    text = text.replace("    h,\n    h0,\n", "    h,\n    hf,\n    h0,\n", 1)
    text = text.replace("    h += (boh * HV + i_h).to(tl.int64) * K*V\n",
                        "    h += (boh * HV + i_h).to(tl.int64) * K*V\n"
                        "    hf += (boh * HV + i_h).to(tl.int64) * K*V\n", 1)
    text = re.sub(r"^(\s*)p_h(\d) = h \+ (.*)$", r"\1p_h\2 = h + \3\n\1q_h\2 = hf + \3", text, flags=re.M)
    text = re.sub(r"^(\s*)tl\.store\(p_h(\d), b_h\d\.to\(p_h\d\.dtype\.element_ty\), mask=m_h\d\)$",
                  r"\g<0>\n\1tl.store(q_h\2, b_h\2, mask=m_h\2)", text, flags=re.M)
    assert text.count("tl.store(q_h") == 4 and text.count("q_h1 = hf +") == 2
    fname = "<chunk_delta_h instrumented>"
    linecache.cache[fname] = (len(text), None, text.splitlines(True), fname)
    mod = types.ModuleType("chunk_delta_h_instr")
    mod.__file__ = fname
    exec(compile(text, fname, "exec"), mod.__dict__)
    return mod.chunk_gated_delta_rule_fwd_kernel_h_blockdim64


KI = instrumented()


def lift(x):
    """Rules.Num.liftMag .bf16 (0 at 0, else max(|x|, 2^-126))."""
    return torch.where(x == 0, torch.zeros_like(x), x.abs().clamp(min=MU))


def chain_err(nb, L):
    """chainErr nb L = ((1 + rD)^nb - 1) L + nb (1 + rD)^nb aD."""
    return ((1 + RD) ** nb - 1) * L + nb * (1 + RD) ** nb * AD


def spec2(A):
    """specNorm: the largest singular value (batched), the square root of the largest eigenvalue
    of `AᵀA` (or of `AAᵀ`, the smaller one)."""
    G = A.transpose(-1, -2) @ A if A.shape[-1] <= A.shape[-2] else A @ A.transpose(-1, -2)
    return torch.linalg.eigvalsh(G)[..., -1].clamp(min=0).sqrt()


def bf16_box(hb):
    """The `float32` values whose `bfloat16` rounding (nearest even) is exactly `hb` (finite), as
    float64 `(lo, hi)`: between the midpoints to the neighbours, a midpoint included when `hb`'s
    last mantissa bit is even; `±0` give `[0, m]` and `[-m, 0]`."""
    bits = hb.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    neg = (bits >> 15) == 1
    mb = bits & 0x7FFF

    def val(mm):
        v = mm.clamp(0, 0x7F80).to(torch.int16).view(torch.bfloat16).to(D)
        return torch.where(torch.isinf(v), torch.full_like(v, 2.0 ** 128), v)
    v0, vm, vp = val(mb), val(mb - 1), val(mb + 1)
    lo = torch.where(mb == 0, torch.zeros_like(v0), (vm + v0) / 2).float()
    hi = ((v0 + vp) / 2).float()
    odd = (mb & 1) == 1
    lo = torch.where(odd, torch.nextafter(lo, torch.full_like(lo, math.inf)), lo).to(D)
    hi = torch.where(odd, torch.nextafter(hi, torch.zeros_like(hi)), hi).to(D)
    return torch.where(neg, -hi, lo), torch.where(neg, -lo, hi)


def seqs(x):
    """(row, start, len) of each sequence, as rowOf / startOf / lenOf."""
    B, T = x["k"].shape[:2]
    cu = x["cu_seqlens"]
    if cu is None:
        return [(b, 0, T) for b in range(B)]
    c = cu.tolist()
    return [(0, c[i], c[i + 1] - c[i]) for i in range(len(c) - 1)]


def cbases(x):
    S = seqs(x)
    if x["cu_seqlens"] is None:
        NT = (x["k"].shape[1] + BT - 1) // BT
        return [n * NT for n in range(len(S))]
    out, acc = [], 0
    for (_, _, L) in S:
        out.append(acc)
        acc += (L + BT - 1) // BT
    return out + [acc]


REL = None  # set to a list to collect (chunk, errStep / ||h[t+1]||, error / ||h[t+1]||)
TOLS = None  # set to a dict to collect the tolerances: ("h", n, t + 1), ("v_new", n, t), ("ht", n)


def check_post(x, h, vn, ht, use_h0, store_ht, vf=False, use_g=True, Y=None):
    """Post's numeric clauses and Vals, for the host's inputs `x` (`h0` in the `[K, V]` layout) and
    outputs (`h`, `ht` in the `[V, K]` layout when `vf`). `Y`: the witness, a list per sequence of
    its `float32` states `[nch + 1, HV, K, V]`, or `"greedy"` (each `Y (t + 1)` the
    point of `h[t + 1]`'s box nearest to the step from `Y t`), or None (refutation only).
    Returns `worst`: `h0` (0, or inf when `Y 0 != h0`), `store` (0, or inf when some
    `bf16(Y t) != h[t]`), `step` (the largest `||Y (t+1) - StepY t|| / errStep t`), `v_new`, `ht` (0,
    or inf when `ht != Y nch`), and `box` (the largest lower bound / errStep over consecutive
    boxes, or inf when `h[0] != bf16(h0)` as values: above 1 refutes `Post`); whether Vals holds; and whether
    every stored state is in range (`StOk`) and every output element checked is finite."""
    k, w, u = (x[n].to(D) for n in ("k", "w", "u"))
    g = x["g"].to(D) if use_g else None
    B, T, H, K = k.shape
    HV, V = u.shape[2], u.shape[3]
    G = HV // H
    nb = (K + 63) // 64
    sK = math.sqrt(K)
    h0 = x["h0"].to(D) if use_h0 else None
    hF = h.reshape(-1, HV, V, K) if vf else h.reshape(-1, HV, K, V)

    def hB(c):  # h[c] as bfloat16, `[HV, K, V]`
        return hF[c].transpose(-1, -2) if vf else hF[c]
    htB = (ht.transpose(-1, -2) if vf else ht) if store_ht else None
    worst = {"h0": 0.0, "store": 0.0, "step": 0.0, "v_new": 0.0, "ht": 0.0, "box": 0.0}
    vals = True
    ok = True  # StOk at every chunk, every checked output finite
    ins = [x["k"], x["w"], x["u"]] + ([x["g"]] if use_g else [])
    for t_, cap in zip(ins, (2 ** 16, 2 ** 16, 2 ** 16, 2 ** 24)):
        vals &= bool(torch.isfinite(t_.float()).all() and (t_.to(D).abs() <= cap).all())
    if use_h0:
        vals &= bool(torch.isfinite(x["h0"]).all() and (h0.abs() <= 2 ** 16).all())

    def ratio(err, tol):
        r = torch.where(err == 0, torch.zeros_like(err), err / tol)
        return r.max().item() if r.numel() else 0.0

    def bits_ne(a, b):  # float32 values a (float64) and bfloat16 b: bf16(a) != b
        return bool((a.float().bfloat16().view(torch.int16) != b.view(torch.int16)).any())

    inf = float("inf")
    cb = cbases(x)
    for n, (b, s0, L) in enumerate(seqs(x)):
        kh = torch.arange(HV, device=DEV) // G
        H0 = h0[n] if use_h0 else torch.zeros(HV, K, V, dtype=D, device=DEV)  # h0v
        pad = torch.zeros(HV, dtype=D, device=DEV)  # padB
        nchn = (L + BT - 1) // BT
        Yn = Y[n] if isinstance(Y, list) else None
        if Yn is not None and bool((Yn[0].to(D) != H0).any()):
            worst["h0"] = inf
        if nchn == 0:
            if store_ht:  # ht = Y 0 = the initial state
                if Yn is not None and not torch.equal(htB[n].float(), Yn[0].float()):
                    worst["ht"] = inf
                if bool((htB[n].to(D) != H0).any()):
                    worst["box"] = inf
            continue
        Hb = hB(cb[n])
        Hc = Hb.to(D)
        ok &= bool(torch.isfinite(Hc).all())
        # h[0] = bf16(Y 0) with Y 0 equal to h0 as a real (Post: `(Y 0 c).toReal = h0v`), so -0 in h0
        # may be +0 in Y 0 and h[0]: compare values, not bits
        if bool((H0.float().bfloat16().double() != Hb.double()).any()):
            worst["box"] = inf
            worst["store"] = inf
        Yc = Yn[0].to(D) if Yn is not None else (H0 if Y == "greedy" else None)
        blo, bhi = H0, H0  # Y 0 is the initial state
        upto = torch.ones(HV, V, dtype=torch.bool, device=DEV)  # StOk t' for every t' <= t
        for t in range(nchn):
            stok = torch.isfinite(Hc).all(dim=1) & (Hc.norm(dim=1) <= 2.0 ** 40)  # StOk t, [HV, V]
            ok &= bool(stok.all())
            upto &= stok
            Hc = torch.where(stok[:, None, :], Hc, torch.zeros_like(Hc))
            if Yc is not None and Yn is not None and t > 0 and bits_ne(Yc, Hb):
                worst["store"] = inf
            a, c = s0 + BT * t, min(BT, L - BT * t)
            Kc = k[b, a:a + c][:, kh].permute(1, 0, 2)          # [HV, c, K]
            Wc = w[b, a:a + c].permute(1, 0, 2)                 # [HV, c, K]
            Uc = u[b, a:a + c].permute(1, 0, 2)                 # [HV, c, V]
            hn = Hc.norm(dim=1)
            hl = lift(Hc).norm(dim=1)
            rh = (UBF * hn + sK * EBF) / (1 - UBF)
            YB = hn + rh
            wl = lift(Wc).norm(dim=2)
            dW = chain_err(nb, wl[:, :, None] * hl[:, None, :])  # [HV, c, V]
            WH = Wc @ Hc
            vrefS = Uc - WH
            bvB = WH.abs() + dW
            sB = 3 * U32 * (Uc.abs() + bvB) + 4 * E32
            vnB = vrefS.abs() + dW + sB
            tolV = dW + sB + UBF * vnB + EBF
            vo = vn[b, a:a + c].to(D).permute(1, 0, 2)
            ok &= bool(torch.isfinite(vo).all())
            worst["v_new"] = max(worst["v_new"], ratio(((vo - vrefS).abs() * upto[:, None, :]), tolV))
            if TOLS is not None:
                TOLS[("v_new", n, t)] = tolV.clone()
            if use_g:
                gc = g[b, a:a + c].permute(1, 0)                # [HV, c]
                gl = gc[:, -1]
                dg = gl[:, None] - gc
                vals &= bool((gl <= 8).all() and (dg <= 8).all())
                Gm = torch.exp2(dg)
                dgE = U32 * dg.abs() + E32
                errF = (1 + 2.0 ** -22) * torch.exp2(dg + dgE) - Gm + MU
                fB = Gm + errF
                vgB = fB[:, :, None] * vnB * (1 + U32) + E32
                dB = (UBF * vgB + EBF + U32 * fB[:, :, None] * vnB + E32 + errF[:, :, None] * vnB
                      + Gm[:, :, None] * (dW + sB))
                gam = torch.exp2(gl)
                errE = 2.0 ** -22 * gam + MU
                eB = gam + errE
                dec = errE[:, None] * YB + U32 * eB[:, None] * YB + sK * E32
                YpB = eB[:, None] * (1 + U32) * YB + sK * E32
            else:  # no gate: Gam = gam = 1, no product, no exp2
                Gm = torch.ones(HV, c, dtype=D, device=DEV)
                vgB = vnB
                dB = UBF * vgB + EBF + dW + sB
                gam = torch.ones(HV, dtype=D, device=DEV)
                eB = gam
                dec = torch.zeros_like(YB)
                YpB = YB
            vgbB = (1 + UBF) * vgB + EBF
            kl = lift(Kc).pow(2).sum(dim=(1, 2)).sqrt()
            Lk = kl[:, None] * (vgbB + MU).norm(dim=1)
            if vf:  # dotHvf: the dot from +0, then the fusable `+` of its transpose
                dotH = RD * Lk + sK * AD + 3 * U32 * (YpB + (1 + RD) * Lk + sK * AD) + 4 * sK * E32
            else:
                dotH = RD * (YpB + Lk) + sK * AD
            pad = (1 + RD + 3 * U32) * (eB * (1 + U32) * pad + E32 + AD) + 4 * E32
            vals &= bool((pad <= 2.0 ** 100).all())
            sKn = spec2(Kc)
            errStep = dec + sKn[:, None] * dB.norm(dim=1) + dotH        # [HV, V]
            beta = Kc.transpose(1, 2) @ (vrefS * Gm[:, :, None])        # StepY t y = gam y + beta
            last = t + 1 == nchn
            if not last:
                Hb = hB(cb[n] + t + 1)
                Hn = Hb.to(D)
                ok &= bool(torch.isfinite(Hn).all())
                nlo, nhi = bf16_box(Hb)
            elif store_ht:
                ok &= bool(torch.isfinite(htB[n]).all())
                nlo = nhi = htB[n].to(D)
            # refutation: the nearest points of Y t's box and Y (t + 1)'s box
            if not last or store_ht:
                g_ = gam[:, None, None]
                dist = torch.clamp(torch.maximum(beta - (nhi - g_ * blo), (nlo - g_ * bhi) - beta), min=0)
                worst["box"] = max(worst["box"], ratio(dist.norm(dim=1) * upto, errStep))
            # the witness
            if Yc is not None:
                tgt = gam[:, None, None] * Yc + beta
                if Yn is not None:
                    Yx = Yn[t + 1].to(D)
                elif not last:
                    Yx = torch.minimum(torch.maximum(tgt, nlo), nhi).float().to(D)
                elif store_ht:
                    Yx = htB[n].to(D)
                else:
                    Yx = tgt.float().to(D)
                fin = torch.isfinite(Yc).all(dim=1) & torch.isfinite(Yx).all(dim=1)
                err = (Yx - tgt).norm(dim=1)
                err = torch.where(fin, err, torch.full_like(err, inf)) * upto
                worst["step"] = max(worst["step"], ratio(err, errStep))
                if REL is not None and not last:
                    hnn = Hn.norm(dim=1).clamp_min(1e-300)
                    REL.append((t + 1, (errStep / hnn).flatten(), (err / hnn).flatten()))
                Yc = Yx
            if TOLS is not None:
                TOLS[("h", n, t + 1) if not last else ("ht", n)] = errStep.clone()
            if not last:
                blo, bhi = nlo, nhi
                Hc = Hn
        if store_ht and Yn is not None and not torch.equal(htB[n].float(), Yn[-1].float()):
            worst["ht"] = inf
    return worst, vals, ok


def accepted(worst):
    return all(worst[o] < 1 for o in ("h0", "store", "step", "v_new", "ht"))


def refuted(worst):
    return worst["box"] >= REFUTE or worst["v_new"] >= REFUTE


def h0_of(x, vf):
    """The initial state as the host takes it: `[N, HV, K, V]`, or `[N, HV, V, K]` when `vf`."""
    return x["h0"].transpose(-1, -2).contiguous() if vf else x["h0"]


def host_run(x, use_h0, store_ht, vf=False, use_g=True):
    return m.chunk_gated_delta_rule_fwd_h(
        x["k"], x["w"], x["u"], g=x["g"] if use_g else None,
        initial_state=h0_of(x, vf) if use_h0 else None,
        output_final_state=store_ht, state_v_first=vf, cu_seqlens=x["cu_seqlens"])


def launch_args(x, vf):
    k, cu = x["k"], x["cu_seqlens"]
    B, T, H, K = k.shape
    HV, V = x["u"].shape[2], x["u"].shape[3]
    if cu is None:
        N, NT, co = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu) - 1, len(m.prepare_chunk_indices(cu, BT))
        co = m.prepare_chunk_offsets(cu, BT)
    hshape = (B, NT, HV, V, K) if vf else (B, NT, HV, K, V)
    sshape = (N, HV, V, K) if vf else (N, HV, K, V)
    return B, T, H, K, HV, V, N, NT, co, hshape, sshape


def witness(x, use_h0, store_ht, ref, vf=False, use_g=True):
    """The kernel's own `float32` states, from the instrumented copy (run with the final state):
    per sequence `[nch + 1, HV, K, V]` (`float32`). Also whether its `h`, `v_new` (and `ht`, when
    the host stored it) equal the host's bit for bit."""
    k, w, u, g, cu = x["k"], x["w"], x["u"], x["g"], x["cu_seqlens"]
    B, T, H, K, HV, V, N, NT, co, hshape, sshape = launch_args(x, vf)
    h = torch.empty(hshape, dtype=torch.bfloat16, device=DEV)
    hf = torch.empty(hshape, dtype=torch.float32, device=DEV)
    vn = torch.empty_like(u)
    ht = torch.zeros(sshape, dtype=torch.float32, device=DEV)
    KI[(triton.cdiv(V, BV) * N * HV,)](
        k=k, v=u, w=w, v_new=vn, g=g if use_g else None, gk=None, h=h, hf=hf,
        h0=h0_of(x, vf) if use_h0 else None, ht=ht, cu_seqlens=cu, chunk_offsets=co, T=T, H=H,
        HV=HV, K=K, V=V, BT=BT, BV=BV, USE_G=use_g, USE_GK=False, USE_INITIAL_STATE=use_h0,
        STORE_FINAL_STATE=True, SAVE_NEW_VALUE=True, STATE_V_FIRST=vf, IS_VARLEN=cu is not None,
        num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    rh, rv, rt = ref
    same = torch.equal(h.view(torch.int16), rh.view(torch.int16)) and \
        torch.equal(vn.view(torch.int16), rv.view(torch.int16))
    if store_ht:
        same &= torch.equal(ht.view(torch.int32), rt.view(torch.int32))
    hfF = hf.reshape(-1, HV, V, K) if vf else hf.reshape(-1, HV, K, V)
    if vf:
        hfF, ht = hfF.transpose(-1, -2), ht.transpose(-1, -2)
    cb = cbases(x)
    Ys = []
    for n, (_, _, L) in enumerate(seqs(x)):
        nchn = (L + BT - 1) // BT
        Ys.append(torch.cat([hfF[cb[n]:cb[n] + nchn], ht[n][None]]))
    return Ys, same


def guarded(shape, dtype, pat, guard=4096):
    """A tensor of `shape` inside a buffer with `guard` elements before and after, all set to
    the bit pattern `pat`."""
    numel = math.prod(shape)
    idt = torch.int16 if dtype == torch.bfloat16 else torch.int32
    buf = torch.full((numel + 2 * guard,), pat, dtype=idt, device=DEV).view(dtype)
    return buf, buf[guard:guard + numel].view(shape), guard


def direct_frame(x, use_h0, store_ht, ref, vf=False, use_g=True):
    """One launch with the host's arguments, the outputs between guards. True when the guards and
    inputs are unchanged, no output element keeps the pattern, and the outputs equal `ref`."""
    k, w, u, g, cu = x["k"], x["w"], x["u"], x["g"], x["cu_seqlens"]
    h0 = h0_of(x, vf) if use_h0 else None
    B, T, H, K, HV, V, N, NT, co, hshape, sshape = launch_args(x, vf)
    before = [t.clone() for t in (k, w, u, g, x["h0"])] + ([cu.clone(), co.clone()] if cu is not None else [])
    hb, h, gh = guarded(hshape, torch.bfloat16, NAN_BF16)
    vb, vn, gv = guarded(tuple(u.shape), torch.bfloat16, NAN_BF16)
    tb, ht, gt = guarded(sshape, torch.float32, NAN_F32)
    if store_ht:
        ht.zero_()  # the host's new_zeros
    grid = (triton.cdiv(V, BV) * N * HV,)
    m.chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k, v=u, w=w, v_new=vn, g=g if use_g else None, gk=None, h=h, h0=h0,
        ht=ht if store_ht else None,
        cu_seqlens=cu, chunk_offsets=co, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BV=BV,
        USE_G=use_g, USE_GK=False, USE_INITIAL_STATE=use_h0, STORE_FINAL_STATE=store_ht,
        SAVE_NEW_VALUE=True, STATE_V_FIRST=vf, IS_VARLEN=cu is not None,
        num_warps=4, num_stages=3)
    torch.cuda.synchronize()
    ok = True
    for buf, gd, pat, idt in ((hb, gh, NAN_BF16, torch.int16), (vb, gv, NAN_BF16, torch.int16),
                             (tb, gt, NAN_F32, torch.int32)):
        bits = buf.view(idt)
        ok &= bool((bits[:gd] == pat).all() and (bits[-gd:] == pat).all())
    ok &= not bool((h.view(torch.int16) == NAN_BF16).any()) and not bool((vn.view(torch.int16) == NAN_BF16).any())
    if not store_ht:
        ok &= bool((ht.view(torch.int32) == NAN_F32).all())
    after = [k, w, u, g, x["h0"]] + ([cu, co] if cu is not None else [])
    ok &= all(torch.equal(p.view(torch.uint8) if p.dtype.is_floating_point else p,
                          q.view(torch.uint8) if q.dtype.is_floating_point else q)
              for p, q in zip(before, after))
    rh, rv, rt = ref
    ok &= torch.equal(h.view(torch.int16), rh.view(torch.int16)) and torch.equal(vn.view(torch.int16), rv.view(torch.int16))
    if store_ht:
        ok &= torch.equal(ht.view(torch.int32), rt.view(torch.int32))
    meta = True
    if cu is not None:
        meta = co.tolist() == cbases(x)
    return ok, meta, (hb, gh)


def to_dev(x, cu32=False):
    if cu32 and x["cu_seqlens"] is not None:
        x["cu_seqlens"] = x["cu_seqlens"].to(torch.int32)
    return x


def run_case(name, lens, H, HV, K, V, varlen, flags=(True, True), cu32=False, vf=False, use_g=True,
             neg0=False, **kw):
    use_h0, store_ht = flags
    if not use_g:
        kw = dict(kw, gate_scale=0.0)  # w, u of the delta rule without a gate (g = 0)
    x = to_dev(m.make_inputs(lens, H, HV, K, V, varlen=varlen, **kw), cu32)
    if neg0:  # every third column of h0 is -0 (the kernel's state starts from +0 there)
        x["h0"][..., ::3] = -0.0
    outs = host_run(x, use_h0, store_ht, vf, use_g)
    torch.cuda.synchronize()
    Ys, same = witness(x, use_h0, store_ht, outs, vf, use_g)
    worst, vals, fin = check_post(x, *outs, use_h0, store_ht, vf, use_g, Y=Ys)
    frame, meta, _ = direct_frame(x, use_h0, store_ht, outs, vf, use_g)
    ok = vals and fin and frame and meta and same and accepted(worst) and not refuted(worst)
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    name = name + (" [V,K]" if vf else "") + ("" if use_g else " no gate")
    print(f"{'ok  ' if ok else 'FAIL'} {name}: err/tol step {worst['step']:.3f}, v_new {worst['v_new']:.3f}"
          f" (box {worst['box']:.3f}); Y 0 = h0 {worst['h0'] == 0}, h = bf16(Y) {worst['store'] == 0}, "
          f"ht = Y {worst['ht'] == 0}; witness = kernel {same}; Vals {vals}, StOk+finite {fin}, "
          f"frame {frame}, chunk_offsets {meta}; peak {mem:.2f} GiB", flush=True)
    return ok, worst, x, outs


def buggy(x, outs, kind, vf, use_g=True):
    """The outputs of the exact recurrence (`float64`, then stored as the kernel stores them; the new
    values from the stored `bfloat16` state, as the kernel's `w` dots read it) with one plausible
    kernel bug: `no decay` drops the state's `2^g_last`; `drop chunk` skips chunk 1's
    update; `no gate` drops the tokens' `2^(g_last - g)`; `key head` reads `k` at head `hv % H`
    instead of `hv // (HV / H)`; `first-token gate` gates every token of a chunk by
    `2^(g_last - g_first)`; `g_last early` takes the chunk's `g_last` at its second-to-last token;
    `ht early` stores the state before the last chunk as `ht`; `bias +` / `bias -` multiply every
    state by `1 + 2^-7` / `1 - 2^-7` each chunk (the step taken from the stored state). Not bugs
    (printed, not tested): `exact` (the state carried in `float64`, the new values from it too) and `bf16 state` (the state
    carried between chunks rounded to `bfloat16`)."""
    h, vn, ht = outs
    k, w, u = (x[n].to(D) for n in ("k", "w", "u"))
    g = x["g"].to(D) if use_g else torch.zeros(u.shape[:3], dtype=D, device=DEV)
    _, _, H, K = k.shape
    HV, V = u.shape[2], u.shape[3]
    kh = torch.arange(HV, device=DEV) % H if kind == "key head" else torch.arange(HV, device=DEV) // (HV // H)
    hh = torch.zeros(h.numel() // (HV * K * V), HV, K, V, dtype=D, device=DEV)
    vv = torch.zeros_like(u)
    tt = torch.zeros(ht.shape[0], HV, K, V, dtype=D, device=DEV)
    cb = cbases(x)
    for n, (b, s0, L) in enumerate(seqs(x)):
        S = x["h0"][n].to(D).clone()
        prev = S
        for t in range((L + BT - 1) // BT):
            a, c = s0 + BT * t, min(BT, L - BT * t)
            if kind in ("bf16 state", "bias +", "bias -"):
                S = S.float().bfloat16().to(D)
            hh[cb[n] + t] = S
            Kc = k[b, a:a + c][:, kh].permute(1, 0, 2)
            Sr = S if kind == "exact" else S.float().bfloat16().to(D)  # what the `w` dots read
            vt = u[b, a:a + c].permute(1, 0, 2) - w[b, a:a + c].permute(1, 0, 2) @ Sr
            vv[b, a:a + c] = vt.permute(1, 0, 2)
            gc = g[b, a:a + c].permute(1, 0)
            gl = gc[:, max(c - 2, 0)] if kind == "g_last early" else gc[:, -1]
            fac = torch.exp2(gl[:, None] - (gc[:, :1] if kind == "first-token gate" else gc))
            if kind == "no gate":
                fac = torch.ones_like(fac)
            dec = torch.ones_like(gl) if kind == "no decay" else torch.exp2(gl)
            prev = S
            if not (kind == "drop chunk" and t == 1):
                S = dec[:, None, None] * S + Kc.transpose(1, 2) @ (vt * fac[:, :, None])
            if kind in ("bias +", "bias -"):
                S = S * (1 + (2.0 ** -7 if kind == "bias +" else -2.0 ** -7))
        tt[n] = prev if kind == "ht early" else S
    if vf:
        hh, tt = hh.transpose(-1, -2), tt.transpose(-1, -2)
    return (hh.reshape(h.shape).bfloat16(), vv.bfloat16(),
            tt.reshape(ht.shape).float())


def negatives(x, outs, use_h0=True, store_ht=True, vf=False, use_g=True):
    """Each corrupted output must be refuted (returns (name, rejected)); and what the check finds
    for the exact recurrence's outputs, with the state carried in `float64` or in `bfloat16`. An
    element move is 4 times its own tolerance: one element of chunk 1 of sequence 0 (value head
    0, column 0, state row 5) by 4 errStep 0 plus the width of its box, one of `v_new` (sequence
    0, token 7) by 4 tolV, one of `ht` (sequence 0, row 3) by 4 errStep."""
    global TOLS
    h, vn, ht = outs
    HV, V, K = vn.shape[2], vn.shape[3], x["k"].shape[3]
    TOLS = {}
    check_post(x, h, vn, ht, use_h0, store_ht, vf, use_g)
    tols, TOLS = TOLS, None
    c1 = cbases(x)[0] + 1
    res = []
    kinds = ["h", "v_new", "ht", "swap", "no decay", "drop chunk", "no gate", "key head",
             "first-token gate", "g_last early", "ht early", "bias +", "bias -"]
    if not use_g:
        kinds = [q for q in kinds if q not in ("no decay", "no gate", "first-token gate", "g_last early")]
    H_ = x["k"].shape[2]
    ar = torch.arange(HV)
    if torch.equal(ar % H_, ar // (HV // H_)):  # one key head per group: the bug changes nothing
        kinds.remove("key head")
    for which in kinds:
        h2, v2, t2 = h.clone(), vn.clone(), ht.clone()
        if which == "h":
            hv_ = h2.view(-1, HV, V, K)[c1, 0, 0, 5] if vf else h2.view(-1, HV, K, V)[c1, 0, 5, 0]
            d = 4 * tols[("h", 0, 1)][0, 0].item() + 2.0 ** -6 * abs(hv_.item())
            if vf:
                h2.view(-1, HV, V, K)[c1, 0, 0, 5] += d
            else:
                h2.view(-1, HV, K, V)[c1, 0, 5, 0] += d
        elif which == "v_new":
            b0, s00, _ = seqs(x)[0]
            v2[b0, s00 + 7, 0, 0] += 4 * tols[("v_new", 0, 0)][0, 7, 0].item()
        elif which == "ht":
            d = 4 * tols[("ht", 0)][0, 0].item()
            if vf:
                t2.view(-1, HV, V, K)[0, 0, 0, 3] += d
            else:
                t2.view(-1, HV, K, V)[0, 0, 3, 0] += d
        elif which == "swap":
            hh = h2.view(-1, *h2.shape[2:])
            hh[[0, -1]] = hh[[-1, 0]]
        else:
            h2, v2, t2 = buggy(x, outs, which, vf, use_g)
        w_, vals, fin = check_post(x, h2, v2, t2, use_h0, store_ht, vf, use_g)
        res.append((which, refuted(w_)))
    acc = {}
    for kind in ("exact", "bf16 state"):
        o2 = buggy(x, outs, kind, vf, use_g)
        w_, _, _ = check_post(x, *o2, use_h0, store_ht, vf, use_g, Y="greedy")
        acc[kind] = ("accepted (witness)" if accepted(w_) else
                     "refuted" if refuted(w_) else "neither")
    return res, acc


if __name__ == "__main__":
    torch.manual_seed(0)
    all_ok = True
    F = (True, True)
    NG = {"use_g": False}
    cases = [
        # fla's test_chunk shapes (fixed length), bfloat16, h0 and ht (tests/ops/test_gdn.py:149-169)
        ("fixed B4 T1024 H4 HV4 D128", [1024] * 4, 4, 4, 128, 128, False, F, False, {}),
        ("fixed B2 T75 H4 HV4 D64 gate/0.01", [75] * 2, 4, 4, 64, 64, False, F, False, {"gate_scale": 100.0}),
        ("fixed B2 T500 H3 HV3 D60", [500] * 2, 3, 3, 60, 60, False, F, False, {}),
        ("fixed B2 T1000 H3 HV3 D64 gate/0.1", [1000] * 2, 3, 3, 64, 64, False, F, False, {"gate_scale": 10.0}),
        ("fixed B3 T1024 H4 HV4 D100 gate/0.1", [1024] * 3, 4, 4, 100, 100, False, F, False, {"gate_scale": 10.0}),
        ("fixed B2 T1500 H4 HV4 D128 gate/10", [1500] * 2, 4, 4, 128, 128, False, F, False, {"gate_scale": 0.1}),
        ("fixed B4 T2048 H8 HV8 D64", [2048] * 4, 8, 8, 64, 64, False, F, False, {}),
        ("fixed B2 T256 H2 HV4 D64", [256] * 2, 2, 4, 64, 64, False, F, False, {}),
        ("fixed B2 T512 H2 HV8 D64 gate/0.1", [512] * 2, 2, 8, 64, 64, False, F, False, {"gate_scale": 10.0}),
        ("fixed B2 T1024 H4 HV8 D128", [1024] * 2, 4, 8, 128, 128, False, F, False, {}),
        ("fixed B1 T64 H2 HV4 D32", [64], 2, 4, 32, 32, False, F, False, {}),
        # Qwen3-Next's heads, the four flag combinations
        ("qwen3-next fixed 2x4096 no h0/ht", [4096] * 2, 16, 32, 128, 128, False, (False, False), False, {"gate_scale": 0.05}),
        ("qwen3-next fixed 2x4096 ht", [4096] * 2, 16, 32, 128, 128, False, (False, True), False, {"gate_scale": 0.05}),
        ("qwen3-next fixed 2x4096 h0", [4096] * 2, 16, 32, 128, 128, False, (True, False), False, {"gate_scale": 0.05}),
        ("qwen3-next fixed 2x4096 h0 ht", [4096] * 2, 16, 32, 128, 128, False, F, False, {"gate_scale": 0.05}),
        ("fla GatedDeltaNet default fixed 2x2048 H6 K256 V512", [2048] * 2, 6, 6, 256, 512, False, F, False, {"gate_scale": 0.1}),
        # variable length (test_gdn.py:752-771, 848-863)
        ("varlen [15] H4 D60", [15], 4, 4, 60, 60, True, F, False, {}),
        ("varlen [256,244,500] H4 D64", [256, 244, 500], 4, 4, 64, 64, True, F, False, {}),
        ("varlen [15,85,200,900,800] H4 D100", [15, 85, 200, 900, 800], 4, 4, 100, 100, True, F, False, {}),
        ("varlen [256,244,500] H2 HV4 D64", [256, 244, 500], 2, 4, 64, 64, True, F, False, {}),
        ("varlen [256,244,500] H2 HV8 D64", [256, 244, 500], 2, 8, 64, 64, True, F, False, {}),
        ("varlen [8192] H4 D60", [8192], 4, 4, 60, 60, True, F, False, {}),
        ("varlen int32 [256,244,500] H4 D64", [256, 244, 500], 4, 4, 64, 64, True, F, True, {}),
        ("qwen3-next varlen int32 no h0/ht", [1000, 3000, 0, 1, 4096], 16, 32, 128, 128, True, (False, False), True, {"gate_scale": 0.05}),
        ("qwen3-next varlen int32 ht", [1000, 3000, 0, 1, 4096], 16, 32, 128, 128, True, (False, True), True, {"gate_scale": 0.05}),
        ("qwen3-next varlen h0", [1000, 3000, 0, 1, 4096], 16, 32, 128, 128, True, (True, False), False, {"gate_scale": 0.05}),
        ("qwen3-next varlen h0 ht", [1000, 3000, 0, 1, 4096], 16, 32, 128, 128, True, F, False, {"gate_scale": 0.05}),
        ("fla GatedDeltaNet default varlen int32 H6 K256 V512", [2000, 48, 2048], 6, 6, 256, 512, True, F, True, {"gate_scale": 0.1}),
        # STATE_V_FIRST (the [V, K] layout): test_chunk_state_v_first's shapes (test_gdn.py:414-425),
        # fla's GatedDeltaNet layer (6 heads, K = 256, V = 512) with the four flag combinations,
        # Qwen3-Next's heads, edges
        ("vf fixed B1 T63 H1 D64", [63], 1, 1, 64, 64, False, F, False, {"vf": True}),
        ("vf fixed B2 T500 H3 D60", [500] * 2, 3, 3, 60, 60, False, F, False, {"vf": True}),
        ("vf fixed B3 T1024 H4 D128", [1024] * 3, 4, 4, 128, 128, False, F, False, {"vf": True}),
        ("vf fixed B4 T2048 H8 D64", [2048] * 4, 8, 8, 64, 64, False, F, False, {"vf": True}),
        ("vf GatedDeltaNet varlen int32 no h0/ht", [2000, 48, 2048], 6, 6, 256, 512, True, (False, False), True, {"vf": True, "gate_scale": 0.1}),
        ("vf GatedDeltaNet varlen int32 ht", [2000, 48, 2048], 6, 6, 256, 512, True, (False, True), True, {"vf": True, "gate_scale": 0.1}),
        ("vf GatedDeltaNet varlen int32 h0", [2000, 48, 2048], 6, 6, 256, 512, True, (True, False), True, {"vf": True, "gate_scale": 0.1}),
        ("vf GatedDeltaNet varlen int32 h0 ht", [2000, 48, 2048], 6, 6, 256, 512, True, F, True, {"vf": True, "gate_scale": 0.1}),
        ("vf GatedDeltaNet fixed 2x2048 ht", [2048] * 2, 6, 6, 256, 512, False, (False, True), False, {"vf": True, "gate_scale": 0.1}),
        ("vf qwen3-next varlen h0 ht", [1000, 3000, 0, 1, 4096], 16, 32, 128, 128, True, F, False, {"vf": True, "gate_scale": 0.05}),
        ("vf edge K=192 V=33 HV=3", [300, 1, 63, 64, 65], 1, 3, 192, 33, True, F, False, {"vf": True}),
        ("vf edge K=100 no decay 4096 tokens", [4096], 2, 2, 100, 100, True, F, False, {"vf": True, "gate_scale": 0.0}),
        ("vf edge empty sequences only h0->ht", [0, 0], 2, 2, 64, 64, True, F, False, {"vf": True}),
        ("vf varlen [256,244,500] H2 HV4 D64", [256, 244, 500], 2, 4, 64, 64, True, F, False, {"vf": True}),
        # USE_G = False (g = None): the delta rule's forward (fla/ops/delta_rule/chunk.py:39-49):
        # test_chunk's shapes (tests/ops/test_delta.py:21-26), test_chunk_varlen's sequences
        # (test_delta.py:167-170), fla's default DeltaNet layer (16 heads, K = V = 128)
        ("delta fixed B1 T63 H1 D64", [63], 1, 1, 64, 64, False, F, False, NG),
        ("delta fixed B2 T100 H4 D60", [100] * 2, 4, 4, 60, 60, False, F, False, NG),
        ("delta fixed B2 T1000 H3 D128", [1000] * 2, 3, 3, 128, 128, False, F, False, NG),
        ("delta fixed B2 T1024 H4 D128", [1024] * 2, 4, 4, 128, 128, False, F, False, NG),
        ("delta fixed B3 T2000 H4 D128", [2000] * 3, 4, 4, 128, 128, False, F, False, NG),
        ("delta fixed B4 T2048 H8 D64", [2048] * 4, 8, 8, 64, 64, False, F, False, NG),
        ("delta fixed B1 T64 H2 D32", [64], 2, 2, 32, 32, False, F, False, NG),
        ("delta varlen int32 [15] H2 D64", [15], 2, 2, 64, 64, True, F, True, NG),
        ("delta varlen int32 [111,389] H3 D60", [111, 389], 3, 3, 60, 60, True, F, True, NG),
        ("delta varlen int32 [256,244,400,100] H3 D64", [256, 244, 400, 100], 3, 3, 64, 64, True, F, True, NG),
        ("delta varlen int32 [15,85,200,900,399,201,200] H4 D100", [15, 85, 200, 900, 399, 201, 200], 4, 4, 100, 100, True, F, True, NG),
        ("delta DeltaNet default fixed 2x4096 H16 D128 no h0/ht", [4096] * 2, 16, 16, 128, 128, False, (False, False), False, NG),
        ("delta DeltaNet default varlen int32 no h0/ht", [1000, 3000, 0, 1, 4096], 16, 16, 128, 128, True, (False, False), True, NG),
        ("delta DeltaNet default varlen int32 h0 ht", [1000, 3000, 0, 1, 4096], 16, 16, 128, 128, True, F, True, NG),
        ("delta DeltaNet default varlen int32 ht", [1000, 3000, 0, 1, 4096], 16, 16, 128, 128, True, (False, True), True, NG),
        ("delta DeltaNet default varlen h0", [1000, 3000, 0, 1, 4096], 16, 16, 128, 128, True, (True, False), False, NG),
        ("delta varlen [256,244,500] H2 HV4 D64", [256, 244, 500], 2, 4, 64, 64, True, F, False, NG),
        ("delta edge K=192 V=33 HV=3", [300, 1, 63, 64, 65], 1, 3, 192, 33, True, F, False, NG),
        ("delta edge K=256 16384 tokens", [16384], 1, 1, 256, 64, True, F, False, NG),
        ("delta edge empty sequences only h0->ht", [0, 0], 2, 2, 64, 64, True, F, False, NG),
        ("delta edge [V,K] varlen", [256, 244, 500], 2, 4, 64, 64, True, F, False, dict(NG, vf=True)),
        # edges of Inst.ok and of Vals
        ("edge K=1 V=1 T=1", [1], 1, 1, 1, 1, False, F, False, {}),
        ("edge K=192 V=33 HV=3", [300, 1, 63, 64, 65], 1, 3, 192, 33, True, F, False, {}),
        ("edge K=256 V=100", [129, 0, 0, 200], 2, 2, 256, 100, True, F, False, {}),
        ("edge empty sequences only h0->ht", [0, 0], 2, 2, 64, 64, True, F, False, {}),
        ("edge no decay 4096 tokens", [4096], 2, 2, 128, 128, True, F, False, {"gate_scale": 0.0}),
        ("edge slow decay 16384 tokens", [16384], 1, 2, 128, 128, True, F, False, {"gate_scale": 0.01}),
        ("edge strong decay", [1000, 777], 2, 4, 128, 128, True, F, False, {"gate_scale": 20.0}),
        ("edge beta in (0, 2)", [512, 300], 2, 2, 128, 128, True, F, False, {"beta_scale": 2.0, "gate_scale": 0.05}),
        ("edge large state h0 x 5e3", [700], 2, 2, 128, 128, True, F, False, {"h0_scale": 5.0e3}),
        ("edge tiny state h0 x 1e-30", [700], 2, 2, 128, 128, True, F, False, {"h0_scale": 1.0e-30, "gate_scale": 3.0}),
        ("edge h0 with -0 entries", [300, 200], 2, 2, 64, 64, True, F, False, {"neg0": True}),
        ("vf edge h0 with -0 entries", [300, 200], 2, 2, 64, 64, True, F, False, {"neg0": True, "vf": True}),
        ("edge 256 sequences of 1..64", [1 + (i * 37) % 64 for i in range(256)], 1, 2, 64, 64, True, F, True, {}),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    negs = {}
    for (name, lens, H, HV, K, V, varlen, flags, cu32, kw) in cases:
        if only and only not in name:
            continue
        torch.cuda.reset_peak_memory_stats()
        ok, worst, x, outs = run_case(name, lens, H, HV, K, V, varlen, flags, cu32, **kw)
        all_ok &= ok
        if name in ("varlen [256,244,500] H2 HV4 D64", "vf varlen [256,244,500] H2 HV4 D64",
                    "delta varlen [256,244,500] H2 HV4 D64", "edge slow decay 16384 tokens"):
            negs[name] = (x, outs, kw.get("vf", False), kw.get("use_g", True))
        del x, outs
        torch.cuda.empty_cache()
    if only:
        sys.exit(0 if all_ok else 1)
    # non-monotone gate within Vals (g_last - g up to about 2, g_last up to about 1)
    x = m.make_inputs([600, 300], 2, 2, 64, 64, varlen=True)
    x["g"] = (torch.rand_like(x["g"]) * 3.0 - 2.0).contiguous()
    outs = host_run(x, True, True)
    Ys, same = witness(x, True, True, outs)
    worst, vals, fin = check_post(x, *outs, True, True, Y=Ys)
    frame, meta, _ = direct_frame(x, True, True, outs)
    ok = vals and fin and frame and meta and same and accepted(worst) and not refuted(worst)
    all_ok &= ok
    print(f"{'ok  ' if ok else 'FAIL'} edge non-monotone gate: err/tol step {worst['step']:.3f}, "
          f"v_new {worst['v_new']:.3f}; Vals {vals}, finite {fin}, frame {frame}", flush=True)
    # negative tests
    for name, (x, outs, vf, use_g) in negs.items():
        res, acc = negatives(x, outs, vf=vf, use_g=use_g)
        for which, rejected in res:
            all_ok &= rejected
            print(f"{'ok  ' if rejected else 'FAIL'} negative ({which}; {name}): rejected {rejected}", flush=True)
        for kind, how in (("exact", "float64"), ("bf16 state", "bfloat16")):
            print(f"info the exact recurrence ({name}), state carried in {how}: {acc[kind]}", flush=True)
    x, outs, _, _ = negs["varlen [256,244,500] H2 HV4 D64"]
    hb_ok, _, (hb, gh) = direct_frame(x, True, True, outs)
    hb.view(torch.int16)[gh - 1] = 0  # a guard element changed: the frame check must notice
    bits = hb.view(torch.int16)
    rejected = not bool((bits[:gh] == NAN_BF16).all())
    all_ok &= hb_ok and rejected
    print(f"{'ok  ' if rejected else 'FAIL'} negative (guard byte changed): rejected {rejected}", flush=True)
    # Vals' padded-row clause: K = 60 (four padded rows), g = 8 in every token of 33 chunks, u = 0,
    # k_i = e_i and w_i = 256 e_i for the chunk's tokens i < 60 (k_i = w_i = 0 for the other four),
    # so the real rows' chunk map is 2^8 I - 256 I = 0. Every other clause of Vals holds; padB grows
    # by about 2^8 per chunk and passes 2^100, so Vals must fail (the dot facts would allow the
    # padded rows to overflow).
    L, Kp, Vp = 33 * BT, 60, 32
    i_loc = torch.arange(L, device=DEV) % BT
    eye = torch.cat([torch.eye(Kp, device=DEV), torch.zeros(BT - Kp, Kp, device=DEV)])[i_loc]
    xp = {"k": eye[None, :, None].bfloat16(), "w": (256 * eye)[None, :, None].bfloat16(),
          "u": torch.zeros(1, L, 1, Vp, device=DEV).bfloat16(),
          "g": torch.full((1, L, 1), 8.0, device=DEV), "h0": torch.zeros(1, 1, Kp, Vp, device=DEV),
          "cu_seqlens": torch.tensor([0, L], device=DEV)}
    zh = torch.zeros(1, 33, 1, Kp, Vp, device=DEV).bfloat16()
    _, vals_p, _ = check_post(xp, zh, xp["u"], xp["h0"].float(), True, True)
    xp["g"].zero_()  # g = 0, w_i = k_i (the real rows' map again 0): padB stays tiny, Vals holds
    xp["w"] = xp["k"].clone()
    _, vals_q, _ = check_post(xp, zh, xp["u"], xp["h0"].float(), True, True)
    all_ok &= vals_q
    print(f"{'ok  ' if vals_q else 'FAIL'} Vals holds with g = 0, w_i = k_i: {vals_q}", flush=True)
    all_ok &= not vals_p
    print(f"{'ok  ' if not vals_p else 'FAIL'} Vals fails on growing padded rows (K = 60, g = 8): {not vals_p}", flush=True)
    print("ALL OK" if all_ok else "SOME CHECK FAILED")
    sys.exit(0 if all_ok else 1)
