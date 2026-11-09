"""GPU check of the claim `Claims.SsdChunkState` (Claim.lean).

Runs the kernel's own host `chunk_state_fwd` (source/ssd_chunk_state.py, imported by path) on the
source's ten test cases, on launches at the edges of `Inst.ok` and `Vals`, and checks, in float64,
`Pre` and `Post` exactly as the claim states them:

* `Pre`'s metadata: the host's `seq_idx` equals the claim's `reqOf` (a transcription of the Lean
  definition, from `query_start_loc` alone);
* `Vals`: every `x`, `B`, `dt`, `dA_cumsum` entry finite, every `dA_cumsum` entry <= 0, and the room
  conditions (`dRef <= 64`, `eHat * |dt| <= 2^126`, `|B| * sHat <= 2^126`, `dotMag <= 2^120`);
* `Post`: every element of `states` finite and within `tol` of `sRef`, both computed by the claim's
  definitions (`toks`, `dRef`, `wRef`, `dErr`, `expRel`, `errE`, `eHat`, `errS`, `sHat`, `errP`,
  `pHat`, `errB`, `bHat`, `magB`, `dotMag`, `steps`, `inMag`; `tol = 2^-8·(1 + steps·2^-6)·dotMag +
  2^-100·inMag`) from the host's inputs; the byte
  frame: the inputs unchanged, and guard bands of 4096 elements on each side of `states` (the host's
  `torch.empty` allocation is placed inside a sentinel-filled buffer during the call) unchanged.

The launches include one whose `x.stride(0)` and `B.stride(0)` are above `2^31` (`int64`
arguments), which `Inst.ok` covers. Each line reports the largest |error| / tol over all elements
and tol / |sRef| (median and largest over the elements with sRef != 0).
Negative tests then check that the same comparison rejects wrong outputs: one element moved by
just over its tolerance, the states of a launch without `seq_idx` (which also sums the other
requests' tokens), the states computed with the wrong group of `B`, and the states of kernels
that miss a token or decay wrongly (computed in float64 from the inputs): the last token of each
chunk dropped, the decay taken to position `chunk_size - 2`, `dt` read one position off, a middle
token dropped, and the first token of each chunk's request segment (`toks c`) dropped under a
weak decay. Under the inputs' own decay that dropped token is accepted: its weight
`exp(dA[chunk_size - 1] - dA[i])` is e^-20.1 to e^-9.5 in chunks 0 and 1 (`i = 0`) and e^-12.8 to
e^-7.5 in chunk 2 (token 600, `i = 88`), so it moves no element by its tolerance (printed, not a
failure).

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "ssd_chunk_state.py")
spec = importlib.util.spec_from_file_location("ssd_chunk_state_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import triton  # noqa: E402

DEV = torch.device("cuda")
BK = 64
GUARD = 4096
SENTINEL = 0x7FC0DEAD           # a NaN pattern no store of the kernel writes
worst = [0.0]
failures = []


def req_of(qsl, nreq, t):
    """Claim.lean `reqOf`: the number of later requests starting at or before `t`."""
    return sum(1 for r in range(1, nreq) if qsl[r] <= t)


def mag_b(v):
    """Claim.lean `magB`: 0 for 0, otherwise max(|v|, 2^-126)."""
    a = v.abs()
    return torch.where(a == 0, torch.zeros_like(a), a.clamp(min=2.0 ** -126))


def post_check(states, x, B, dt, dA, seqlens, cs, extra=None):
    """Post's numeric clause for every element (and Vals); returns (worst err/tol, finite, vals,
    tol / |sRef| over the elements with sRef != 0)."""
    _, seqlen, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    nc = dt.shape[2]
    nreq = len(seqlens)
    qsl = [0]
    for s in seqlens:
        qsl.append(qsl[-1] + s)
    grp = torch.arange(H, device=x.device) // (H // G)
    vals_ok = bool(torch.isfinite(x).all()) and bool(torch.isfinite(B).all())
    vals_ok = vals_ok and bool(torch.isfinite(dt).all()) and bool(torch.isfinite(dA).all())
    vals_ok = vals_ok and bool((dA <= 0).all())
    fin_ok = True
    w_max = 0.0
    ratios = []
    for c in range(nc):
        lo = c * cs
        end_tok = min((c + 1) * cs, seqlen) - 1
        e = req_of(qsl, nreq, end_tok)
        t0 = max(lo, qsl[e])                                  # toks: lo..end_tok with reqOf t = e
        i = torch.arange(t0 - lo, end_tok + 1 - lo, device=x.device)
        dAc = dA[0, :, c].double()                            # [H, cs]
        D = dAc[:, cs - 1:cs] - dAc[:, i]                     # dRef [H, nt]
        dErr = D.abs() * 2.0 ** -24 + 2.0 ** -150
        rel = 2.0 ** -22 + 2.0 ** -23 * (D.abs() + dErr)
        eD = torch.exp(D)
        errE = eD * ((1 + rel) * torch.expm1(dErr) + rel) + 2.0 ** -140
        eHat = eD + errE
        dtv = dt[0, :, c].double()[:, i]
        w = eD * dtv                                          # wRef
        errS = dtv.abs() * errE + 2.0 ** -24 * (eHat * dtv.abs()) + 2.0 ** -150
        sHat = w.abs() + errS
        xs = x[0, t0:end_tok + 1].double()                    # [nt, H, P]
        Bv = B[0, t0:end_tok + 1].double()[:, grp]            # [nt, H, N]
        Bw = Bv * w.t()[:, :, None]
        errP = Bv.abs() * errS.t()[:, :, None] + 2.0 ** -24 * (Bv.abs() * sHat.t()[:, :, None]) + 2.0 ** -150
        pHat = Bw.abs() + errP
        errB = errP + 2.0 ** -8 * pHat + 2.0 ** -134
        bHat = Bw.abs() + errB
        ref = torch.einsum("thp,thn->hpn", xs, Bw)
        A = torch.einsum("thp,thn->hpn", mag_b(xs), bHat.clamp(min=2.0 ** -126))
        T = (end_tok + 1 - lo + BK - 1) // BK
        inMag = torch.einsum("thp,thn->hpn", 1 + xs.abs(), (1 + Bv.abs()) * (1 + dtv.abs()).t()[:, :, None])
        tol = 2.0 ** -8 * (1 + T * 2.0 ** -6) * A + 2.0 ** -100 * inMag
        vals_ok = vals_ok and bool((D <= 64).all()) and float((eHat * dtv.abs()).max()) <= 2.0 ** 126
        vals_ok = vals_ok and float((Bv.abs() * sHat.t()[:, :, None]).max()) <= 2.0 ** 126
        vals_ok = vals_ok and float(A.max()) <= 2.0 ** 120
        got = states[0, c].double()
        fin_ok = fin_ok and bool(torch.isfinite(got).all())
        r = float(((got - ref).abs() / tol).max())
        w_max = max(w_max, r)
        nz = ref != 0
        ratios.append((tol[nz] / ref[nz].abs()).flatten())
        if extra is not None:
            extra(c, ref, tol)
    rat = torch.cat(ratios) if ratios else torch.zeros(1, dtype=torch.float64)
    return w_max, fin_ok, vals_ok, rat


def run(name, seqlens, H, P, N, G, cs, rows, gen, transform=None, record=True):
    x, B, dt, dA, seq_idx = mod.make_inputs(seqlens, H, P, N, G, cs, DEV, gen, rows)
    if transform is not None:
        x, B, dt, dA = transform(x, B, dt, dA)
    seqlen = sum(seqlens)
    qsl = [0]
    for s in seqlens:
        qsl.append(qsl[-1] + s)
    # Pre: MetaOk
    meta_ok = seq_idx[0].tolist() == [req_of(qsl, len(seqlens), t) for t in range(seqlen)]
    base = x._base if x._base is not None else x
    keep = [t.clone() for t in (base, B, dt, dA, seq_idx)]
    # the host allocates states with torch.empty (line 243): during the call, the allocation is placed
    # inside a larger buffer whose guard bands (GUARD elements on each side) hold a sentinel
    orig_empty = torch.empty
    bufs = []

    def empty_guarded(shape, device=None, dtype=None):
        n = math.prod(shape)
        big = orig_empty((n + 2 * GUARD,), device=device, dtype=dtype)
        big.view(torch.int32).fill_(SENTINEL)
        bufs.append(big)
        return big[GUARD:GUARD + n].view(shape)

    torch.empty = empty_guarded
    try:
        states = mod.chunk_state_fwd(B, x, dt, dA, seq_idx=seq_idx, states_in_fp32=True)
    finally:
        torch.empty = orig_empty
    torch.cuda.synchronize()
    frame_ok = len(bufs) == 1 and all(torch.equal(a, b) for a, b in zip((base, B, dt, dA, seq_idx), keep))
    for big in bufs:
        g = big.view(torch.int32)
        frame_ok = frame_ok and bool((g[:GUARD] == SENTINEL).all()) and bool((g[-GUARD:] == SENTINEL).all())
    r, fin_ok, vals_ok, rat = post_check(states, x, B, dt, dA, seqlens, cs)
    good = meta_ok and vals_ok and fin_ok and frame_ok and r <= 1
    if record:
        worst[0] = max(worst[0], r)
    print(f"{'ok ' if good else 'BAD'} {name}: requests {len(seqlens)} tokens {seqlen} heads {H} headdim {P} "
          f"dstate {N} groups {G} chunk {cs} {'rows' if rows else 'contiguous'}: err/tol {r:.4f}, "
          f"tol/|sRef| median {rat.median().item():.3g} max {rat.max().item():.3g}; "
          f"seq_idx {meta_ok}, Vals {vals_ok}, finite {fin_ok}, frame {frame_ok}", flush=True)
    if not good:
        failures.append(name)
    return states, (x, B, dt, dA, seq_idx)


def wrong_states(x, B, dt, dA, seqlens, cs, drop=None, end_pos=None, dt_shift=0):
    """The chunk states of a wrong kernel, in float64 from the inputs (`sRef` with one change):
    `drop(t0, end_tok)` the tokens of `toks c` it leaves out, `end_pos` the position it decays to
    (instead of `chunk_size - 1`), `dt_shift` the shift of the `dt` position it reads (cyclic
    within the chunk). Returns float32 states shaped as the host's."""
    _, seqlen, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    nc = dt.shape[2]
    qsl = [0]
    for s_ in seqlens:
        qsl.append(qsl[-1] + s_)
    grp = torch.arange(H, device=x.device) // (H // G)
    end_pos = cs - 1 if end_pos is None else end_pos
    out = torch.zeros((1, nc, H, P, N), dtype=torch.float64, device=x.device)
    for c in range(nc):
        lo = c * cs
        end_tok = min((c + 1) * cs, seqlen) - 1
        t0 = max(lo, qsl[req_of(qsl, len(seqlens), end_tok)])
        toks = [t for t in range(t0, end_tok + 1) if drop is None or t not in drop(t0, end_tok)]
        if not toks:
            continue
        i = torch.tensor([t - lo for t in toks], device=x.device)
        dAc = dA[0, :, c].double()
        D = dAc[:, end_pos:end_pos + 1] - dAc[:, i]
        dtv = torch.roll(dt[0, :, c].double(), dt_shift, dims=-1)[:, i]
        w = torch.exp(D) * dtv
        xs = x[0, toks].double()
        Bw = B[0, toks].double()[:, grp] * w.t()[:, :, None]
        out[0, c] = torch.einsum("thp,thn->hpn", xs, Bw)
    return out.float()


def negative_tokens(gen):
    """A kernel that misses a token or decays to the wrong position fails `Post`. On a launch of
    768 tokens in chunks of 256 (two requests, 600 + 168: the tokens of chunks 0 and 1 are the
    first request's, those of chunk 2 the second request's, from token 600), each wrong kernel's
    states (`wrong_states`) are compared with the claim's reference and tolerance."""
    sl, H, P, N, G, cs = [600, 168], 8, 64, 128, 2, 256
    _, (x, B, dt, dA, _) = run("token tests' launch", sl, H, P, N, G, cs, True, gen)
    tests = [
        ("last token of each chunk dropped", dict(drop=lambda t0, e: {e}), None),
        ("decay to position chunk_size - 2", dict(end_pos=cs - 2), None),
        ("dt shifted by one position", dict(dt_shift=1), None),
        ("a middle token of each chunk dropped", dict(drop=lambda t0, e: {(t0 + e) // 2}), None),
        ("first token of each chunk's request segment dropped, weak decay (dA x 0.01)",
         dict(drop=lambda t0, e: {t0}), 0.01),
    ]
    for name, kw, dA_scale in tests:
        dAv = dA if dA_scale is None else dA * dA_scale
        bad = wrong_states(x, B, dt, dAv, sl, cs, **kw)
        r = post_check(bad, x, B, dt, dAv, sl, cs)[0]
        ok = r > 1
        print(f"{'ok ' if ok else 'BAD'} negative, {name}: err/tol {r:.3g} (must exceed 1)")
        if not ok:
            failures.append("negative " + name)
    # Not rejected, and not a failure of the check: under the inputs' own decay the first token of
    # each chunk's request segment carries the weight exp(dA[cs - 1] - dA[i]), e^-20.1 to e^-9.5
    # across the heads in chunks 0 and 1 (i = 0) and e^-12.8 to e^-7.5 in chunk 2 (i = 88), so
    # leaving it out moves every element by less than its tolerance.
    bad = wrong_states(x, B, dt, dA, sl, cs, drop=lambda t0, e: {t0})
    r = post_check(bad, x, B, dt, dA, sl, cs)[0]
    lw = (dA[0, :, :2, cs - 1] - dA[0, :, :2, 0]).double()
    lw2 = (dA[0, :, 2, cs - 1] - dA[0, :, 2, 600 - 2 * cs]).double()
    print(f"    (first token of each chunk's request segment dropped, the inputs' decay: err/tol "
          f"{r:.3g}, accepted; its decay exponent is {lw.max().item():.1f} to {lw.min().item():.1f} "
          f"in chunks 0 and 1 (position 0), {lw2.max().item():.1f} to {lw2.min().item():.1f} in "
          f"chunk 2 (token 600, position 88))")


def negative(gen):
    """The comparison must reject wrong outputs."""
    sl, H, P, N, G, cs = [100, 37, 200], 8, 64, 128, 2, 64
    states, (x, B, dt, dA, seq_idx) = run("negative tests' launch", sl, H, P, N, G, cs, True, gen)
    # 1. one element moved by 1.01 x its tolerance
    bad = states.clone()
    hit = {}

    def grab(c, ref, tol):
        if c == 1:
            hit["v"] = (ref[3, 5, 7] + 1.01 * tol[3, 5, 7]).item()
    post_check(states, x, B, dt, dA, sl, cs, extra=grab)
    bad[0, 1, 3, 5, 7] = hit["v"]
    r1 = post_check(bad, x, B, dt, dA, sl, cs)[0]
    # 2. the launch without seq_idx: sums the earlier requests' tokens of a chunk too
    r2 = post_check(mod.chunk_state_fwd(B, x, dt, dA, seq_idx=None), x, B, dt, dA, sl, cs)[0]
    # 3. B's groups swapped (head h reads group 1 - g)
    Bsw = B.flip(2).contiguous()
    r3 = post_check(mod.chunk_state_fwd(Bsw, x, dt, dA, seq_idx=seq_idx), x, B, dt, dA, sl, cs)[0]
    for name, r in (("one element + 1.01 tol", r1), ("no seq_idx", r2), ("groups swapped", r3)):
        ok = r > 1
        print(f"{'ok ' if ok else 'BAD'} negative, {name}: err/tol {r:.3g} (must exceed 1)")
        if not ok:
            failures.append("negative " + name)


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # SETUP.md's launches: the source's ten test cases (setup_1 ... setup_10)
    cases = [
        ([112], 3, 5, 5, 3, 16, False),
        ([128], 32, 128, 128, 32, 32, False),
        ([16, 20], 16, 64, 64, 16, 8, False),
        ([270, 88, 212, 203], 16, 64, 64, 16, 256, False),
        ([138, 225], 13, 32, 32, 13, 128, False),
        ([5, 30], 4, 21, 21, 4, 256, False),
        ([1000, 1, 2049, 64, 64, 300], 80, 64, 128, 1, 256, True),
        ([8192], 128, 64, 128, 8, 256, True),
        ([3] * 40 + [500], 128, 80, 128, 8, 256, True),
        ([300, 1700], 64, 64, 256, 1, 256, True),
    ]
    for k, (sl, H, P, N, G, cs, rows) in enumerate(cases):
        run(f"setup_{k + 1}", sl, H, P, N, G, cs, rows, gen)
    # edges of Inst.ok (setup_e1 ... setup_e6)
    edges = [
        ("setup_e1 32768 tokens", [8000, 24768], 128, 64, 128, 8, 256, True),
        ("setup_e2 200 one-token requests", [1] * 200, 8, 64, 128, 1, 256, True),
        ("setup_e3 chunk 1", [3, 2, 4], 4, 16, 16, 2, 1, True),
        ("setup_e4 chunk 2", [7, 1, 8], 4, 16, 16, 4, 2, False),
        ("setup_e5 headdim = dstate = 1", [37], 1, 1, 1, 1, 16, False),
        ("setup_e6 headdim 65, dstate 129", [100, 157], 6, 65, 129, 3, 64, True),
    ]
    for name, sl, H, P, N, G, cs, rows in edges:
        run(name, sl, H, P, N, G, cs, rows, gen)
    # edges of Vals, on the shapes of setup_7 and setup_e6
    big = ([1000, 1, 2049, 64, 64, 300], 80, 64, 128, 1, 256, True)
    run("strong decay (dA x 200)", *big, gen, transform=lambda x, B, dt, dA: (x, B, dt, dA * 200.0))
    run("no decay (dA = 0)", *big, gen, transform=lambda x, B, dt, dA: (x, B, dt, dA * 0.0))
    run("large dt (dt x 1000)", *big, gen, transform=lambda x, B, dt, dA: (x, B, dt * 1000.0, dA))

    def nonmono(x, B, dt, dA):
        bump = torch.zeros_like(dA)
        bump[..., : dA.shape[-1] // 2] = -3.0
        return x, B, dt, (dA + bump).clamp(max=0.0)
    run("non-monotone dA_cumsum (decays up to +3)", *big, gen, transform=nonmono)
    small = ([100, 157], 6, 65, 129, 3, 64, True)
    run("values near 1e15", *small, gen,
        transform=lambda x, B, dt, dA: ((x.float() * 1e15).bfloat16(), (B.float() * 1e15).bfloat16(), dt, dA))
    run("values near 1e-20 (products below float32's normal range)", *small, gen,
        transform=lambda x, B, dt, dA: ((x.float() * 1e-20).bfloat16(), (B.float() * 1e-20).bfloat16(), dt, dA))

    def int64_batch_strides(x, B, dt, dA):
        # x.stride(0) and B.stride(0) above 2^31 (int64 arguments; batch = 1, so no element is
        # read through them), as SGLang's layer passes them from about 210,000 tokens on
        s0 = 2 ** 31 + 4096
        return (x.as_strided(x.shape, (s0,) + x.stride()[1:], x.storage_offset()),
                B.as_strided(B.shape, (s0,) + B.stride()[1:], B.storage_offset()), dt, dA)
    run("int64 x.stride(0), B.stride(0)", *big, gen, transform=int64_batch_strides)

    def adversarial(x, B, dt, dA):
        # every scaled B = dt = 1 + 2^-8 - 2^-20 rounds down to 1 in bfloat16 (just under half an ulp),
        # with x = B = 1 and no decay: the error adds up coherently over the chunk
        dt2 = torch.where(dt > 0, torch.full_like(dt, 1 + 2.0 ** -8 - 2.0 ** -20), dt)
        return torch.ones_like(x), torch.ones_like(B), dt2, dA * 0.0
    run("adversarial bfloat16 rounding (coherent)", [256, 300], 4, 64, 128, 1, 256, True, gen,
        transform=adversarial)
    negative(gen)
    negative_tokens(gen)
    print(f"worst err/tol over the covered launches: {worst[0]:.4f}")
    if failures:
        print("FAILED:", ", ".join(failures))
        raise SystemExit(1)
    print("all checks hold")
