"""GPU check of the claim `Claims.SsdChunkScan` (Claim.lean).

Runs the kernel's own host `chunk_scan_fwd` (source/ssd_chunk_scan.py, imported by path) on the
source's eleven test cases and on launches at the edges of `Inst.ok` and `Vals`, and checks, in
float64, `Pre` and `Post` exactly as the claim states them:

* `Pre`'s metadata: the host's `seq_idx`, `chunk_indices` and `chunk_offsets` equal the claim's
  `reqOf`, `hostIdx` and `hostOff` (a transcription of the Lean definitions below, from
  `query_start_loc` alone), `len(chunk_offsets)` equals `hostN`, and the strides are the ones
  `model` passes;
* `Vals`: every input the reference reads finite, every `dA_cumsum` entry <= 0, and the range
  conditions (`dSt <= 64`, `dS <= 64`, `wMag <= 2^120`, `lSig`, `hSt * (mSig + eSig)`,
  `aSig + lW`, `aAcc + |x D|` <= 2^120);
* `Post`: every `out[0, t, h, p]` finite and within `tol` of `yRef`, both computed by the claim's
  definitions from the host's inputs (the request boundaries from `query_start_loc`); the byte
  frame: the inputs unchanged bit for bit, and guard bands of 4096 elements on each side of `out`
  (placed inside a sentinel-filled buffer) unchanged.

`tol = (nIt + 256) 2^-15 mag + 2 expErr + 2^-110 (1 + hSt)`, `mag = |yRef| + hSt lSig + lW`
(Claim.lean). Each line reports the largest |error| / tol over the launch's outputs, the median
tol / |yRef|, and the same median for the fine bound of README Tolerances (steps 1-6,
`(1 + 2^-8) errPre + 2^-8 |yRef| + 2^-134`, which `Post` does not use) with the largest
tol / fine; the check also fails if tol is below the fine bound anywhere on a covered launch. The float64 evaluation of
`yRef` and `tol` has a relative error near 1e-15, far below the tolerance (about 1e-2 relative).
Negative tests then check that `Post` rejects wrong outputs: a launch with wrong `chunk_offsets`,
an output moved by twice its tolerance, and three plausible wrong readings of the kernel (the
first request's state in chunk 0 taken from `initial_states`, no decay boundary, no `D` term).

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "ssd_chunk_scan.py")
spec = importlib.util.spec_from_file_location("ssd_chunk_scan_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import triton  # noqa: E402

DEV = torch.device("cuda")
uB, uF, mB = 2.0 ** -8, 2.0 ** -24, 2.0 ** -126
etaF, etaB = 2.0 ** -150, 2.0 ** -134
RHO = 2.0 ** -23 + 2.0 ** -47      # tl.mul_add's constant (the fine bound only)
BK = 32
GUARD = 4096
SENTINEL = 0x7FC1                # a bfloat16 NaN pattern no store of the kernel writes
BIG = 2.0 ** 120
failures = []
worst = [0.0]
fine_stats = {"min tol - fine": float("inf"), "tol/fine": []}


def relD(K):
    return (K + 2) * 2.0 ** -22


def absD(K):
    return (K + 2) * 2.0 ** -126


def errE(x, dlt):
    rel = 2.0 ** -22 + 2.0 ** -23 * (x.abs() + dlt)
    return (1 + rel) * torch.exp(x + dlt) - torch.exp(x) + 2.0 ** -140


def eHat(x, dlt):
    return torch.exp(x) + errE(x, dlt)


# --- the claim's host metadata (Claim.lean: laterReqs, reqOf, hostN, metaStep, hostMeta) -------

def req_of_all(qsl, nreq, T):
    r = torch.zeros(T, dtype=torch.int64)
    for q in range(1, nreq):
        r[qsl[q]:] += 1
    return r


def host_n(qsl, nreq, T, cs):
    return (T + cs - 1) // cs + sum(1 for r in range(1, nreq) if qsl[r] % cs != 0)


def host_meta(qsl, nreq, cs, n):
    idx = list(range(n))
    off = {}
    p = 0
    for r in range(1, nreq):
        s, e = qsl[r], qsl[r + 1]
        p += 1 if s % cs > 0 else 0
        lo_k = s // cs + p
        hi_k = e // cs + p + (1 if e % cs > 0 else 0)
        for k in range(lo_k, min(hi_k, n)):
            idx[k] = idx[k] - p
        off[lo_k] = s % cs
    return idx, [off.get(k, 0) for k in range(n)]


def fin(t):
    return bool(torch.isfinite(t).all())


def run_and_check(name, inp, H, P, N, G, cs, has_init=True, chunk_offsets=None, variant=None,
                  perturb=False, expect="hold"):
    """Launch the host on `inp` (with `initial_states` and the metadata when `has_init`, else
    `None` as SGLang passes them; `chunk_offsets` replaced when given) and check `Pre`, `Vals` and
    `Post` (with the reference `variant` when given). `expect` is "hold" (a covered launch: `Pre`,
    `Vals` and `Post` hold), "reject" (`Post` must fail) or "excluded" (`Vals` must fail: the
    claim says nothing of the values; the output is reported). Returns whether `Post` held."""
    x, C, cb, dt, dA = inp["x"], inp["C"], inp["cb"], inp["dt"], inp["dA_cumsum"]
    states, init, D = inp["states"], inp["initial_states"], inp["D"]
    T = x.shape[1]
    qsl = [int(v) for v in inp["query_start_loc"].tolist()]
    nreq = len(qsl) - 1
    nc = triton.cdiv(T, cs)
    BSD = max(triton.next_power_of_2(N), 16)
    grp = torch.arange(H, device=DEV) // (H // G)

    # Pre: the metadata are the host's, as the claim transcribes them
    ref_req = req_of_all(qsl, nreq, T)
    meta_ok = inp["seq_idx"][0].cpu().long().equal(ref_req)
    if has_init:
        n = host_n(qsl, nreq, T, cs)
        idx, off = host_meta(qsl, nreq, cs, n)
        meta_ok = (meta_ok and len(inp["chunk_offsets"]) == n and len(inp["chunk_indices"]) == n
                   and inp["chunk_indices"].cpu().tolist() == idx
                   and inp["chunk_offsets"].cpu().tolist() == off)
    else:
        n = nc
    # the strides `model` passes
    xT, cT = x.stride(1), C.stride(1)
    strides_ok = (x.stride() == (T * xT, xT, P, 1) and C.stride() == (T * cT, cT, N, 1)
                  and cb.is_contiguous() and dt.is_contiguous() and dA.is_contiguous()
                  and states.is_contiguous() and init.is_contiguous() and D.is_contiguous()
                  and H * P <= xT and G * N <= cT)
    # CbZero: without initial states, cb is a zero on the pairs s <= t of one physical chunk
    # that belong to different requests
    cbzero_ok = True
    if not has_init:
        reqd = ref_req.to(DEV)
        for c in range(nc):
            lo, hi = c * cs, min(T, (c + 1) * cs)
            rq = reqd[lo:hi]
            cross = (rq[:, None] != rq[None, :]).tril()                    # [t, s], s <= t
            blk = cb[0, c][:, :hi - lo, :hi - lo]
            cbzero_ok = cbzero_ok and bool((blk[:, cross] == 0).all())
    if not (meta_ok and strides_ok and cbzero_ok):
        failures.append(f"{name}: Pre (metadata {meta_ok}, strides {strides_ok}, "
                        f"cb zero across requests {cbzero_ok})")
        return False

    # out inside a sentinel-filled buffer; clones of the inputs for the frame
    buf = torch.full((2 * GUARD + T * H * P,), SENTINEL, dtype=torch.int16, device=DEV)
    out = buf[GUARD:GUARD + T * H * P].view(torch.bfloat16).view(1, T, H, P)
    ins = [x, C, cb, dt, dA, states, init, D, inp["seq_idx"], inp["chunk_indices"],
           inp["chunk_offsets"]]
    before = [t.clone() for t in ins]
    co = inp["chunk_offsets"] if chunk_offsets is None else chunk_offsets
    mod.chunk_scan_fwd(cb, x, dt, dA, C, states, D=D, z=None, seq_idx=inp["seq_idx"],
                       chunk_indices=inp["chunk_indices"] if has_init else None,
                       chunk_offsets=co if has_init else None,
                       initial_states=init if has_init else None, out=out)
    torch.cuda.synchronize()
    frame_ok = (bool((buf[:GUARD] == SENTINEL).all()) and bool((buf[-GUARD:] == SENTINEL).all())
                and all(torch.equal(a.view(torch.uint8) if a.dtype.is_floating_point else a,
                                    b.view(torch.uint8) if b.dtype.is_floating_point else b)
                        for a, b in zip(ins, before)))

    # Vals: the finite inputs (cb on the used pairs below), the states the kernel rounds to
    # bfloat16 at most 2^120, dA <= 0
    def fin120(t):
        return fin(t) and bool((t.double().abs() <= BIG).all())
    vals_ok = (fin(C[0, :T]) and fin(x[0, :T]) and fin(D) and fin120(states)
               and (fin120(init) or not has_init) and fin(dt) and fin(dA)
               and bool((dA <= 0).all()))
    vals_why = [] if vals_ok else ["inputs"]

    x64 = x[0].double()
    C64 = C[0].double()
    D64 = D.double()
    dA64 = dA[0].double()
    dt64 = dt[0].double()
    nIt = (cs + BK - 1) // BK
    worst_l = 0.0
    viol = 0
    nonfin = 0
    rels = []
    fmax = 0.0
    out64 = out[0].double()
    if perturb:
        out64 = out64.clone()
    for r in range(nreq):
        start, end = qsl[r], qsl[r + 1]
        for c in range(start // cs, (end - 1) // cs + 1):
            lo, hi = max(start, c * cs), min(end, (c + 1) * cs)
            o_ = lo - c * cs
            i0, i1 = o_, hi - c * cs
            L = hi - lo
            from_init = r >= 1 and start >= c * cs
            if variant == "init0" and r == 0 and c == 0:
                from_init = True
            gated = from_init and not has_init
            # Sv: the reference's state; SvL: the state the kernel reads out (line 361)
            SL = (init[r] if (from_init and has_init) else states[0, c]).double()   # [H, P, N]
            S = torch.zeros_like(SL) if (gated and variant != "nogate") else SL
            Ct = C64[lo:hi][:, grp]                                          # [L, H, N]
            aC, aS = Ct.abs(), S.abs()
            sig = torch.einsum("lhn,hpn->lhp", Ct, S)
            mSig = torch.einsum("lhn,hpn->lhp", aC, aS)
            cSum = aC.sum(-1)[:, :, None]
            lSig = torch.einsum("lhn,hpn->lhp", aC + mB, (1 + uB) * aS + etaB + mB)
            lSigL = torch.einsum("lhn,hpn->lhp", aC + mB, (1 + uB) * SL.abs() + etaB + mB)
            dAi = dA64[:, c, i0:i1]                                          # [H, L]
            use_b = o_ > 0 and variant != "noboundary"
            dB = dA64[:, c, o_ - 1][:, None] if use_b else torch.zeros_like(dAi[:, :1])
            dSt = (dAi - dB).T[:, :, None]                                   # [L, H, 1]
            dErrSt = (uF * dSt.abs() + etaF) if o_ > 0 else torch.zeros_like(dSt)
            eSt, hSt = errE(dSt, dErrSt), eHat(dSt, dErrSt)
            eSig = uB * mSig + etaB * cSum + relD(BSD) * lSig + absD(BSD)
            errSig = (1 + uF) * eSig * hSt + uF * mSig * hSt + mSig * eSt + etaF
            aSig = (1 + uF) * (mSig + eSig) * hSt + etaF
            # the intra-chunk weights, on the pairs s <= t of the logical chunk (toks)
            causal = torch.ones(L, L, dtype=torch.bool, device=DEV).tril()[None]
            cbm = cb[0, c].double()[grp][:, i0:i1, i0:i1]                    # [H, L(t), L(s)]
            vals_ok_cb = fin(torch.where(causal, cbm, 0.0))
            dtm = dt64[:, c, i0:i1][:, None, :]
            dS = dAi[:, :, None] - dAi[:, None, :]
            dErrS = uF * dS.abs() + etaF

            def m(v):
                return torch.where(causal, v, 0.0)
            w = m(cbm * torch.exp(dS) * dtm)
            wMag = m(cbm.abs() * eHat(dS, dErrS) * dtm.abs())
            wAbs = m(2.0 ** -133 + 2.0 ** -148 * dtm.abs())
            eW = m(cbm.abs() * dtm.abs() * errE(dS, dErrS))
            ewS = eW + (uB + 3 * uF) * wMag + wAbs
            wHat = (1 + uB + 3 * uF) * wMag + wAbs
            # the pairs the loop computes (kToks): without initial states, from the physical
            # chunk's start (cb zero across requests: Pre)
            k0 = i0 if has_init else 0
            Lk = i1 - k0
            ck = torch.ones(L, Lk, dtype=torch.bool, device=DEV).tril(i0 - k0)[None]
            cbk = cb[0, c].double()[grp][:, i0:i1, k0:i1]                    # [H, L(t), Lk(s)]
            dAk = dA64[:, c, k0:i1]
            dSk = dAi[:, :, None] - dAk[:, None, :]
            ceK = torch.where(ck, cbk.abs() * eHat(dSk, uF * dSk.abs() + etaF), 0.0)
            wMagK = ceK * dt64[:, c, k0:i1][:, None, :].abs()
            xs = x64[lo:hi]                                                  # [L, H, P]
            ax = xs.abs()
            errW = torch.einsum("hts,shp->thp", ewS, ax)
            mW = torch.einsum("hts,shp->thp", wHat, ax)
            lW = torch.einsum("hts,shp->thp", m(wHat + mB), ax + mB)
            intra = torch.einsum("hts,shp->thp", w, xs)
            errDot = 2 * nIt * ((relD(BK) + 2.0 ** -21) * (aSig + lW) + absD(BK) + 2.0 ** -140)
            aAcc = aSig + mW + errDot
            xD = xs * D64[None, :, None]
            if variant == "noD":
                xD = torch.zeros_like(xD)
            errMA = RHO * (xD.abs() + aAcc) + 3 * etaF
            errPre = errSig + errW + errDot + errMA
            y = torch.exp(dSt) * sig + intra + xD
            # the fine bound of README Tolerances (steps 1-6), for the comparison only
            fine = (1 + uB) * errPre + uB * y.abs() + etaB
            # Post's tolerance: (nIt + 256) 2^-15 mag + 2 expErr + 2^-110 (1 + hSt)
            mag = y.abs() + hSt * lSig + lW
            expErr = mSig * eSt + torch.einsum("hts,shp->thp", eW + wAbs, ax)
            tol = (nIt + 256) * 2.0 ** -15 * mag + 2 * expErr + 2.0 ** -110 * (1 + hSt)
            if expect == "hold":
                fine_stats["min tol - fine"] = min(fine_stats["min tol - fine"],
                                                   float((tol - fine).min()))
                rf = (tol / fine).flatten()
                fine_stats["tol/fine"].append(rf[::101].float().cpu())
                fine_stats["max"] = max(fine_stats.get("max", 0.0), float(rf.max()))
            conds = {
                "cb": vals_ok_cb,
                "dSt": bool((dSt <= 64).all()),
                "dS": bool((torch.where(ck, dSk, 0.0) <= 64).all()),
                "cb exp": bool((ceK <= BIG).all()),
                "wMag": bool((wMagK <= BIG).all()),
                "lSigL": bool((lSigL <= BIG).all()),
                "state term": bool((hSt * (mSig + eSig) <= BIG).all()),
                "aSig + lW": bool((aSig + lW <= BIG).all()),
                "aAcc + |xD|": bool((aAcc + xD.abs() <= BIG).all()),
            }
            for k_, v_ in conds.items():
                if not v_ and k_ not in vals_why:
                    vals_why.append(k_)
            vals_ok = vals_ok and all(conds.values())
            if perturb and r == nreq - 1 and c == (end - 1) // cs:
                # move the last element by twice its tolerance (rounded to bfloat16)
                out64[hi - 1, H - 1, P - 1] = (y[-1, -1, -1] + 2 * tol[-1, -1, -1] +
                                               4 * uB * y[-1, -1, -1].abs()).to(
                                                   torch.bfloat16).double()
            o = out64[lo:hi]
            nonfin += int((~torch.isfinite(o)).sum())
            good = torch.isfinite(o) & ((o - y).abs() <= tol)
            viol += int((~good).sum())
            ratio = ((o - y).abs() / tol).nan_to_num(nan=float("inf"))
            worst_l = max(worst_l, float(ratio.max()))
            rels.append((float((tol / y.abs()).median()), float((fine / y.abs()).median())))
            fmax = max(fmax, float((tol / fine).max()))
    post_ok = viol == 0 and frame_ok
    ok = post_ok and vals_ok
    print(f"{name}: tokens {T} requests {nreq} heads {H} headdim {P} dstate {N} groups {G} "
          f"chunk {cs} initial states {has_init} (grid y {n}): worst err/tol {worst_l:.3g}, "
          f"median tol/|yRef| {sorted(rels)[len(rels) // 2][0]:.1e} "
          f"(fine bound {sorted(rels)[len(rels) // 2][1]:.1e}; tol/fine at most {fmax:.3g}), "
          f"violations {viol} "
          f"(non-finite {nonfin}), Vals {vals_ok}{'' if vals_ok else ' ' + str(vals_why)}, "
          f"frame {frame_ok}: " + ("holds" if ok else "FAILS"))
    if expect == "hold":
        worst[0] = max(worst[0], worst_l)
        if not ok:
            failures.append(name)
    elif expect == "reject":
        if ok:
            failures.append(f"{name}: a wrong output was accepted")
    else:
        if vals_ok:
            failures.append(f"{name}: Vals holds on a launch it must exclude")
        if not frame_ok:
            failures.append(f"{name}: frame")
    return ok


def inputs(seqlens, H, P, N, G, cs, gen, contig=False, a_scale=2.0, zero_decay=False,
           nonmono=False, x_scale=1.0, st_scale=1.0, c_scale=1.0, adversarial=False,
           st_fill=0.0, cb_upper=False, dt_big_cb0=False):
    """Inputs as SGLang's Mamba-2 prefill builds them (the source's `make_inputs`: `cb` zero
    across requests, NaN past `seqlen`), with knobs for the edges: contiguous `x`/`C` (SGLang's
    tests), strong or no decay, a non-monotone `dA_cumsum`, scaled values, states whose
    `bfloat16` roundings all go the same way, states filled with a value, `cb` at the float32
    maximum and NaN on the pairs `s > t`, and `dt` at the float32 maximum on tokens whose `cb`
    column is zero."""
    T = sum(seqlens)
    nc = triton.cdiv(T, cs)
    conv_dim = H * P + 2 * G * N
    conv = (torch.randn((T, conv_dim), generator=gen) * torch.tensor(
        [x_scale] * (H * P) + [1.0] * (G * N) + [c_scale] * (G * N))).to(torch.bfloat16)
    if adversarial:
        conv[:, H * P + G * N:] = 1.0
    conv = conv.to(DEV)
    xf, Bf, Cf = torch.split(conv, [H * P, G * N, G * N], dim=-1)
    if contig:
        xf, Cf = xf.contiguous(), Cf.contiguous()
    x = xf.view(1, T, H, P)
    B = Bf.reshape(1, T, G, N)
    C = Cf.view(1, T, G, N)
    qsl = torch.tensor([0] + list(seqlens)).cumsum(0)
    seq_idx = torch.repeat_interleave(torch.arange(len(seqlens), dtype=torch.int32),
                                      torch.tensor(seqlens)).unsqueeze(0)
    ci, co = mod.query_start_loc_to_chunk_indices_offsets(qsl, cs, T)
    A = -torch.exp(torch.rand(H, generator=gen) * a_scale)
    if zero_decay:
        A = torch.zeros(H)
    dt_tok = torch.nn.functional.softplus(torch.randn((H, nc * cs), generator=gen) - 3.0)
    dt_tok[:, T:] = 0.0
    dt = dt_tok.reshape(1, H, nc, cs)
    if nonmono:
        dA = -torch.rand((1, H, nc, cs), generator=gen) * 5.0
    else:
        dA = (A[:, None] * dt_tok).reshape(1, H, nc, cs).cumsum(-1)
    pad = nc * cs - T
    Bc = torch.nn.functional.pad(B[0].float().cpu(), (0, 0, 0, 0, 0, pad)).reshape(nc, cs, G, N)
    Cc = torch.nn.functional.pad(C[0].float().cpu(), (0, 0, 0, 0, 0, pad)).reshape(nc, cs, G, N)
    cb = torch.einsum("cign,cjgn->cgij", Cc.double(), Bc.double()).float()
    sid = torch.nn.functional.pad(seq_idx[0], (0, pad), value=-1).reshape(nc, cs)
    same = (sid[:, :, None] == sid[:, None, :]) & (sid[:, :, None] >= 0)
    cb = torch.where(same[:, None], cb, 0.0)
    valid = sid >= 0
    cb = torch.where(valid[:, None, :, None] & valid[:, None, None, :], cb, float("nan")).contiguous()
    if cb_upper:
        upper = torch.ones(cs, cs, dtype=torch.bool).triu(1)
        cb = torch.where(upper, torch.finfo(torch.float32).max, cb)
        cb[:, :, :, 1::2] = torch.where(upper[:, 1::2], float("nan"), cb[:, :, :, 1::2])
    if dt_big_cb0:
        # the last token of each request in a chunk, s: cb[t, s] = 0 for every t, dt[s] huge
        ends = [e - 1 for e in qsl[1:].tolist()]
        for e in ends:
            c, j = divmod(e, cs)
            cb[c, :, :, j] = 0.0
            dt[0, :, c, j] = torch.finfo(torch.float32).max
    states = torch.randn((1, nc, H, P, N), generator=gen) * st_scale + st_fill
    init = torch.randn((len(seqlens), H, P, N), generator=gen) * st_scale + st_fill
    init = init * (torch.rand(len(seqlens), generator=gen) < 0.7)[:, None, None, None]
    if adversarial and st_fill == 0.0:
        # every state rounds down to 1 in bfloat16 (just below the midpoint 1 + 2^-8)
        states = torch.full_like(states, 1.0 + 2.0 ** -8 - 2.0 ** -20)
        init = torch.full_like(init, 1.0 + 2.0 ** -8 - 2.0 ** -20)
    D = (torch.rand(H, generator=gen) + 0.5).to(torch.bfloat16)
    return dict(cb=cb.unsqueeze(0).to(DEV), x=x, dt=dt.to(DEV), dA_cumsum=dA.to(DEV), C=C,
                states=states.to(DEV), D=D.to(DEV), seq_idx=seq_idx.to(DEV),
                chunk_indices=ci.to(torch.int32).to(DEV), chunk_offsets=co.to(torch.int32).to(DEV),
                initial_states=init.to(DEV), query_start_loc=qsl)


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # the source's test cases (lines 781-793), with the source's own make_inputs, each with
    # initial states (setup_1 ... setup_11) and without (setup_12 ... setup_22)
    cases = [
        ([1], 8, 64, 128, 8, 256),
        ([256], 8, 64, 128, 8, 256),
        ([16, 2, 8, 13], 16, 64, 64, 16, 8),
        ([270, 88, 212, 203], 16, 64, 64, 16, 256),
        ([16, 20], 4, 32, 32, 4, 8),
        ([5, 1, 1, 1], 13, 5, 5, 13, 256),
        ([138, 225, 138, 225], 8, 21, 21, 8, 128),
        ([64, 32, 16, 8, 8], 4, 16, 16, 4, 8),
        ([1000, 1, 2049, 64, 64, 300], 32, 80, 128, 8, 128),
        ([4096, 4096], 64, 64, 128, 8, 256),
        ([3] * 40 + [500], 16, 64, 128, 1, 256),
    ]
    for i, (lens, H, P, N, G, cs) in enumerate(cases, 1):
        inp = mod.make_inputs(lens, H, P, N, G, cs, gen, DEV)
        run_and_check(f"setup_{i}", inp, H, P, N, G, cs)
        run_and_check(f"setup_{i + len(cases)}", inp, H, P, N, G, cs, has_init=False)
        del inp
        torch.cuda.empty_cache()

    # edges of Inst.ok and Vals, each with and without initial states
    edges = [
        ("chunk 2", [3, 1, 4], 4, 16, 16, 2, 2, {}),
        ("chunk 16", [40, 9, 33], 4, 32, 32, 4, 16, {}),
        ("chunk 1024", [1500, 548, 1], 4, 64, 64, 1, 1024, {}),
        ("dstate 1", [300, 77], 8, 64, 1, 8, 256, {}),
        ("dstate 17", [300, 77], 8, 64, 17, 2, 128, {}),
        ("dstate 100", [300, 77], 8, 64, 100, 4, 128, {}),
        ("headdim 1", [129, 130], 8, 1, 64, 8, 64, {}),
        ("headdim 65", [129, 130], 4, 65, 64, 4, 64, {}),
        ("headdim 128", [129, 130], 4, 128, 128, 1, 128, {}),
        ("one group, 128 heads", [500, 300], 128, 64, 128, 1, 256, {}),
        ("200 one-token requests", [1] * 200, 8, 64, 128, 8, 256, {}),
        ("chunk-aligned starts", [256, 256, 512, 3], 8, 64, 128, 8, 256, {}),
        ("seven starts in one chunk", [3, 5, 7, 2, 1, 4, 30, 400], 8, 64, 128, 8, 256, {}),
        ("contiguous x and C", [270, 88, 212, 203], 16, 64, 128, 16, 256, {"contig": True}),
        ("Nemotron-H rank, 8192 tokens", [3000, 2000, 3192], 128, 64, 128, 8, 128, {}),
        ("Nemotron Nano 2 rank", [5000, 3192], 128, 80, 128, 8, 128, {}),
        ("24 heads, one group, 8192 tokens", [6000, 2192], 24, 64, 128, 1, 256, {}),
        ("32768 tokens", [20000, 12768], 16, 64, 128, 8, 256, {}),
        ("strong decay", [700, 300], 8, 64, 128, 8, 256, {"a_scale": 6.0}),
        ("no decay", [700, 300], 8, 64, 128, 8, 256, {"zero_decay": True}),
        ("non-monotone dA_cumsum", [700, 300], 8, 64, 128, 8, 256, {"nonmono": True}),
        ("values near 1e15", [700, 300], 8, 64, 128, 8, 256,
         {"x_scale": 1e15, "st_scale": 1e15}),
        ("states at 2^120, C near 2^-130", [300, 77], 8, 64, 128, 8, 256,
         {"st_scale": 0.0, "st_fill": 2.0 ** 120, "c_scale": 2.0 ** -130}),
        ("subnormal x and C", [300, 200], 8, 64, 128, 8, 256,
         {"x_scale": 1e-39, "c_scale": 1e-39}),
        ("states rounding one way", [700, 300], 8, 64, 128, 8, 256,
         {"adversarial": True, "x_scale": 0.0}),
        ("cb at float32 max and NaN on pairs s > t", [300, 77, 140], 8, 64, 128, 8, 256,
         {"cb_upper": True}),
        ("dt at float32 max where cb is 0", [300, 77, 140], 8, 64, 128, 8, 256,
         {"dt_big_cb0": True}),
    ]
    for name, lens, H, P, N, G, cs, kw in edges:
        inp = inputs(lens, H, P, N, G, cs, gen, **kw)
        run_and_check(name, inp, H, P, N, G, cs)
        run_and_check(name + ", no initial states", inp, H, P, N, G, cs, has_init=False)
        del inp
        torch.cuda.empty_cache()

    # launches Vals excludes (overflow paths of the kernel's float32 arithmetic): the claim says
    # nothing of their values; each must fail Vals (the output is reported)
    big = torch.finfo(torch.float32).max
    lens, H, P, N, G, cs = [300, 77], 8, 64, 128, 8, 256
    inp = inputs(lens, H, P, N, G, cs, torch.Generator().manual_seed(1))
    inp["states"][..., 0] = big          # finite float32, rounds to bfloat16 inf at line 360
    inp["initial_states"][..., 0] = big
    inp["C"][..., 0] = 0.0               # 0 * inf at line 361
    run_and_check("excluded: states at float32 max, C column 0", inp, H, P, N, G, cs,
                  expect="excluded")
    run_and_check("excluded: states at float32 max, C column 0, no initial states", inp,
                  H, P, N, G, cs, has_init=False, expect="excluded")
    inp = inputs(lens, H, P, N, G, cs, torch.Generator().manual_seed(2), nonmono=True)
    inp["dt"][0, :, 0, 0] = 0.0          # token 0's dt = 0: wMag(t, 0) = 0
    inp["dA_cumsum"][0, :, 0, 0] = -5.0  # dS(t, 0) > 0
    inp["cb"][0, 0, :, 1:300, 0] = big   # cb * exp overflows at line 408, inf * 0 at line 410
    run_and_check("excluded: cb at float32 max with dt = 0", inp, H, P, N, G, cs,
                  expect="excluded")
    run_and_check("excluded: cb at float32 max with dt = 0, no initial states", inp,
                  H, P, N, G, cs, has_init=False, expect="excluded")
    inp = inputs([200, 100], H, P, N, G, cs, gen)
    inp["dA_cumsum"][0, :, 0, 150:200] = -100.0   # request 0's last tokens far below request 1's
    inp["dA_cumsum"][0, :, 0, 200:] = 0.0         # exp(dS) overflows on the cross-request pairs,
    run_and_check("excluded: cross-request decay above 64, no initial states", inp,
                  H, P, N, G, cs, has_init=False, expect="excluded")    # times cb = 0
    run_and_check("excluded: decay boundary above 64", inp, H, P, N, G, cs, expect="excluded")
    inp = inputs(lens, H, P, N, G, cs, gen, st_scale=0.0, st_fill=2.0 ** 120, adversarial=True)
    inp["C"].mul_(4.0)                   # Σ C·S above float32 max at line 361 (times 0 when gated)
    run_and_check("excluded: state read-out above float32 max, no initial states", inp,
                  H, P, N, G, cs, has_init=False, expect="excluded")
    inp = inputs(lens, H, P, N, G, cs, gen)
    inp["x"].fill_(3.0e38)               # finite bfloat16; x * D overflows where D > 1 (line 442)
    run_and_check("excluded: x near bfloat16 max", inp, H, P, N, G, cs, expect="excluded")
    del inp
    torch.cuda.empty_cache()

    # negative tests: Post must reject these
    lens, H, P, N, G, cs = [270, 88, 212, 203], 8, 64, 128, 8, 256
    inp = inputs(lens, H, P, N, G, cs, gen)
    run_and_check("wrong chunk_offsets (all 0)", inp, H, P, N, G, cs,
                  chunk_offsets=torch.zeros_like(inp["chunk_offsets"]), expect="reject")
    run_and_check("an output moved by 2 tol", inp, H, P, N, G, cs, perturb=True, expect="reject")
    run_and_check("an output moved by 2 tol, no initial states", inp, H, P, N, G, cs,
                  has_init=False, perturb=True, expect="reject")
    run_and_check("reading: request 0 from initial_states in chunk 0", inp, H, P, N, G, cs,
                  variant="init0", expect="reject")
    run_and_check("reading: no decay boundary", inp, H, P, N, G, cs, variant="noboundary",
                  expect="reject")
    run_and_check("reading: no D term", inp, H, P, N, G, cs, variant="noD", expect="reject")
    run_and_check("reading: no initial states, a starting request continues states[0, c]", inp,
                  H, P, N, G, cs, has_init=False, variant="nogate", expect="reject")

    print(f"worst err/tol over the covered launches: {worst[0]:.3f}")
    q = torch.cat(fine_stats["tol/fine"])
    print(f"tol / fine bound over the covered launches' outputs: median {float(q.median()):.3f} "
          f"(a sample) max {fine_stats['max']:.3f}; "
          f"min (tol - fine) {fine_stats['min tol - fine']:.3g}")
    if fine_stats["min tol - fine"] < 0:
        failures.append("tol below the fine bound")
    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("all checks hold")
