"""GPU check of the claim `Claims.SsdStatePassing` (Claim.lean).

Runs the kernel's own host `state_passing_fwd` (source/
ssd_state_passing.py, imported by path) on the source's nine test cases and on launches at the edges
of `Inst.ok` and `Vals`, each with initial states (`initial_states` and `chunk_offsets` given,
`hasInit`) and without (both `None`), and checks, in float64, `Pre` and `Post` exactly as the claim
states them:

* `Pre`'s metadata: the host's `seq_idx` equals the claim's `reqOf`, and with initial states
  `chunk_offsets` equals `hostOff` and `len(chunk_offsets)` equals `hostN` (a transcription of the
  Lean definitions below, from `query_start_loc` alone);
* `Vals`: every input finite, every `dA_cumsum` entry <= 0, and the room conditions
  (`dRef <= 64`, `eHat * (|xRef| + tolIn) + |states| <= 2^126`);
* `Post`: `out[0, 0]` equals `initial_states[0]` bit for bit (the `float32` `+0` without initial
  states); for `1 <= c < nchunks` `out[0, c]` and at `c = nchunks` `final_states[0]` are finite and
  within `tol c` of `sRef c`, with `sRef` and `tol` computed by the claim's recursions from the
  host's inputs (a request starting in a chunk starts from its initial state, or from `0`); the
  byte frame: the inputs unchanged, and guard bands of 4096 elements on each side of `out` and
  `final_states` (the host's two `torch.empty` allocations are placed inside sentinel-filled
  buffers during the call) unchanged; every element of the outputs is checked against the
  reference.

Negative tests then check that the same comparison rejects wrong outputs: an output moved by twice
its tolerance on a chunk where a request starts (without initial states), the outputs of a launch
with initial states checked as a launch without (the state not restarted from `0`), and a launch
with wrong `chunk_offsets` (all `0`, so the decay of a request starting inside a chunk counts the
earlier tokens too).

Each line reports, per output, the largest |error| / tol. The float64 evaluation of `sRef` has a
relative error near 1e-16 per chunk, far below the tolerance (about 1e-7 relative per chunk).

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "ssd_state_passing.py")
spec = importlib.util.spec_from_file_location("ssd_state_passing_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import triton  # noqa: E402

DEV = torch.device("cuda")
RHO = 2.0 ** -23 + 2.0 ** -47
ETA = 2.0 ** -150
GUARD = 4096                    # guard elements on each side of out and final_states
SENTINEL = 0x7FC0DEAD           # a NaN pattern no store of the kernel writes
worst = {"out": 0.0, "final": 0.0}
failures = []


# --- the claim's host metadata (Claim.lean: laterReqs, reqOf, hostN, offStep, hostOff) ---------

def req_of(qsl, nreq, t):
    return sum(1 for r in range(1, nreq) if qsl[r] <= t)


def host_n(qsl, nreq, seqlen, cs):
    return (seqlen + cs - 1) // cs + sum(1 for r in range(1, nreq) if qsl[r] % cs != 0)


def host_off(qsl, nreq, cs, n):
    f = {}
    p = 0
    for r in range(1, nreq):
        s = qsl[r]
        p += 1 if s % cs > 0 else 0
        f[s // cs + p] = s % cs
    return [f.get(k, 0) for k in range(n)]


def check(name, seqlens, states, dA_cumsum, init, cs, has_init=True, launch_init=None, offsets=None,
          perturb=False, expect="hold"):
    """One launch: `has_init` is the claim's `hasInit` for `Pre` and `Post`; the launch itself has
    initial states when `launch_init` (default `has_init`); `offsets` replaces `chunk_offsets`;
    `perturb` moves one output on a chunk where a request starts by twice its tolerance. `expect`
    is "hold" (`Pre`, `Vals` and `Post` hold) or "reject" (`Pre` and `Vals` hold, `Post` fails)."""
    if launch_init is None:
        launch_init = has_init
    nreq = len(seqlens)
    seqlen = sum(seqlens)
    nchunks = triton.cdiv(seqlen, cs)
    _, _, nheads, dim = states.shape
    qsl = [0]
    for s in seqlens:
        qsl.append(qsl[-1] + s)
    # the host's metadata, as the source's make_inputs builds it
    seq_idx = torch.repeat_interleave(torch.arange(nreq, dtype=torch.int32), torch.tensor(seqlens)).unsqueeze(0)
    _, chunk_offsets = mod.query_start_loc_to_chunk_indices_offsets(torch.tensor(qsl), cs, seqlen)
    chunk_offsets = chunk_offsets.to(torch.int32)
    # Pre: MetaOk and Inst.ok's nmeta_eq
    req = [req_of(qsl, nreq, t) for t in range(seqlen)]
    assert seq_idx[0].tolist() == req, f"{name}: seq_idx differs from reqOf"
    assert len(chunk_offsets) == host_n(qsl, nreq, seqlen, cs), f"{name}: N differs from hostN"
    assert chunk_offsets.tolist() == host_off(qsl, nreq, cs, len(chunk_offsets)), f"{name}: hostOff"
    if offsets is not None:
        chunk_offsets = offsets
    # the launch (without initial states the host gets None for initial_states and chunk_offsets,
    # and passes chunk_meta_num = 0: Inst.ok's nmeta_eq)
    args = [t.to(DEV) for t in (states, dA_cumsum, init, seq_idx, chunk_offsets)]
    keep = [a.clone() for a in args]
    # the host allocates out and final_states with torch.empty (lines 274-279): during the call,
    # each allocation is placed inside a larger buffer whose guard bands (GUARD elements on each
    # side) hold a sentinel, so that Post's byte frame can be checked around the outputs too
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
        out, final = mod.state_passing_fwd(args[0], args[1],
                                           initial_states=args[2] if launch_init else None,
                                           seq_idx=args[3], chunk_size=cs, out_dtype=torch.float32,
                                           is_cont_batched=True,
                                           chunk_offsets=args[4] if launch_init else None)
    finally:
        torch.empty = orig_empty
    torch.cuda.synchronize()
    frame_ok = all(torch.equal(a, b) for a, b in zip(args, keep)) and len(bufs) == 2
    for big in bufs:
        g = big.view(torch.int32)
        frame_ok = frame_ok and bool((g[:GUARD] == SENTINEL).all()) and bool((g[-GUARD:] == SENTINEL).all())
    # the reference and the tolerance (Claim.lean: endTok, eReq, isReset, offR, hasSub, dRef, sRef,
    # dErr, expRel, errE, eHat, tol, tolIn), from the host's inputs, in float64
    A = args[1][0].double()            # [nheads, nchunks, cs]
    I = args[2].double()               # [nreq, nheads, dim]
    S = args[0][0]                     # [nchunks, nheads, dim] (float32; sliced per chunk)
    if not has_init:
        I = torch.zeros_like(I)        # startV: every request starts from 0
    vals_ok = bool(torch.isfinite(A).all()) and bool((A <= 0).all()) and bool(torch.isfinite(I).all())
    vals_ok = vals_ok and bool(torch.isfinite(S).all())
    s = I[0].clone()
    tl_ = torch.zeros_like(s)
    o0 = out[0, 0]
    if has_init:
        exact0 = torch.equal(o0.view(torch.int32), args[2][0].view(torch.int32))
    else:
        exact0 = bool((o0.view(torch.int32) == 0).all())
    moved = not perturb
    e_out = 0.0
    e_fin = 0.0
    fin_ok = True
    for c in range(nchunks):
        end_tok = min((c + 1) * cs, seqlen) - 1
        e = req[end_tok]
        reset = c * cs <= qsl[e]
        off = qsl[e] - c * cs
        has_sub = reset and off > 0
        d = A[:, c, cs - 1] - (A[:, c, off - 1] if has_sub else 0.0)       # [nheads]
        derr = d.abs() * 2.0 ** -24 + ETA if has_sub else torch.zeros_like(d)
        rel = 2.0 ** -22 + 2.0 ** -23 * (d.abs() + derr)
        ed = torch.exp(d)
        errE = ed * ((1 + rel) * torch.expm1(derr) + rel) + 2.0 ** -140
        eHat = ed + errE
        x = I[e] if reset else s
        tin = torch.zeros_like(tl_) if reset else tl_
        n = S[c].double()
        room = (eHat[:, None] * (x.abs() + tin) + n.abs()).max().item()
        vals_ok = vals_ok and bool((d <= 64).all()) and room <= 2.0 ** 126
        s = ed[:, None] * x + n
        tl_ = eHat[:, None] * tin + errE[:, None] * x.abs() + RHO * (eHat[:, None] * (x.abs() + tin) + n.abs()) + 3 * ETA
        got = out[0, c + 1] if c < nchunks - 1 else final[0]
        g = got.double()
        if not moved and reset and c >= 1:
            g = g.clone()
            g[0, 0] = s[0, 0] + 2 * tl_[0, 0]
            moved = True
        fin_ok = fin_ok and bool(torch.isfinite(g).all())
        r = ((g - s).abs() / tl_).max().item()
        if c < nchunks - 1:
            e_out = max(e_out, r)
        else:
            e_fin = r
    post = exact0 and fin_ok and frame_ok and e_out <= 1 and e_fin <= 1
    if expect == "hold":
        worst["out"] = max(worst["out"], e_out)
        worst["final"] = max(worst["final"], e_fin)
        good = vals_ok and post
    else:
        good = vals_ok and moved and not post
    print(f"{'ok ' if good else 'BAD'} {name} (initial states {has_init}): requests {nreq} tokens {seqlen} "
          f"chunks {nchunks} heads {nheads} dim {dim} chunk {cs}: err/tol out {e_out:.3g} final {e_fin:.3g}; "
          f"out[0] exact {exact0}, Vals {vals_ok}, frame {frame_ok}" + ("" if expect == "hold" else
                                                                         f"; Post {post} (must be False)"))
    if not good:
        failures.append(name)


def inputs(seqlens, nheads, dim, cs, gen, dscale=1.0, init_scale=1.0, st_scale=1.0):
    states, dA_cumsum, seq_idx, chunk_offsets, dA_tok = mod.make_inputs(seqlens, nheads, dim, cs, gen)
    has_prefix = torch.rand(len(seqlens), generator=gen) < 0.7
    init = torch.randn((len(seqlens), nheads, dim), generator=gen) * has_prefix[:, None, None]
    return states * st_scale, dA_cumsum * dscale, init * init_scale


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    # SETUP.md's launches: the source's nine test cases (lines 430-438), with its generator
    cases = [
        ([1], 8, 8192, 256),
        ([256], 8, 8192, 256),
        ([16, 20], 4, 32 * 16, 8),
        ([270, 88, 212, 203], 4, 5 * 16, 8),
        ([270, 88, 212, 203], 16, 8192, 256),
        ([5, 1, 1, 1, 300, 7, 128, 129], 16, 8192, 128),
        ([1000, 1, 2049, 64, 64, 300], 32, 10240, 128),
        ([4096, 4096], 128, 8192, 256),
        ([3] * 40 + [500], 8, 8192, 64),
    ]
    # (setup_1 ... setup_9 with initial states, setup_10 ... setup_18 the same launches without)
    for i, (sl, H, D, cs) in enumerate(cases):
        st, dA, init = inputs(sl, H, D, cs, gen)
        check(f"setup_{i + 1}", sl, st, dA, init, cs)
        check(f"setup_{i + 10}", sl, st, dA, init, cs, has_init=False)
    # edges of Inst.ok
    edge = [
        ("chunk 1", [3, 1, 5, 2], 2, 64, 1),
        ("chunk 2", [3, 1, 5, 2, 7], 2, 64, 2),
        ("chunk 1024", [700, 1500, 9], 2, 1024, 1024),
        ("dim 1", [9, 1, 30], 3, 1, 8),
        ("dim 1025", [9, 1, 30], 3, 1025, 8),
        ("dim 2049, 1 head", [50, 77], 1, 2049, 16),
        ("1-token requests", [1] * 200, 2, 128, 8),
        ("aligned starts", [64, 128, 64, 256], 2, 256, 64),
        ("many starts in one chunk", [1, 2, 3, 4, 5, 6, 7, 300], 2, 256, 256),
        ("one chunk, 3 requests", [2, 3, 4], 4, 512, 16),
        ("4096 chunks, one request", [32768], 2, 64, 8),
        ("prefill 32768 tokens, 128 heads x 8192", [8192, 1, 16384, 8191], 128, 8192, 256),
    ]
    both = (True, False)
    for name, sl, H, D, cs in edge:
        st, dA, init = inputs(sl, H, D, cs, gen)
        for hi in both:
            check(name, sl, st, dA, init, cs, has_init=hi)
    # edges of Vals: strong decay (dA x 200, exp underflows), no decay over 4096 chunks, large values
    st, dA, init = inputs([2000, 3, 1000], 4, 256, 8, gen, dscale=200.0)
    for hi in both:
        check("strong decay (dA x 200)", [2000, 3, 1000], st, dA, init, 8, has_init=hi)
    st, dA, init = inputs([32768], 2, 64, 8, gen, dscale=0.0)
    for hi in both:
        check("no decay, 4096 chunks", [32768], st, dA, init, 8, has_init=hi)
    st, dA, init = inputs([300, 5, 700], 4, 256, 64, gen, init_scale=1e30, st_scale=1e30)
    for hi in both:
        check("values near 1e30", [300, 5, 700], st, dA, init, 64, has_init=hi)
    # a cumulative sum that is not monotone within a chunk (entries <= 0, decays up to about +3)
    sl, H, D, cs = [37, 100, 91], 4, 256, 32
    st, dA, init = inputs(sl, H, D, cs, gen)
    bump = torch.zeros_like(dA)
    bump[..., : cs // 2] = -3.0
    for hi in both:
        check("non-monotone dA_cumsum", sl, st, (dA + bump).clamp(max=0.0), init, cs, has_init=hi)
    # negative tests: Post must reject these
    sl, H, D, cs = [270, 88, 212, 203], 4, 256, 64
    st, dA, init = inputs(sl, H, D, cs, gen)
    init = torch.randn(init.shape, generator=gen)          # every request with a nonzero initial state
    check("negative: an output moved by 2 tol", sl, st, dA, init, cs, has_init=False, perturb=True,
          expect="reject")
    check("negative: initial states read", sl, st, dA, init, cs, has_init=False, launch_init=True,
          expect="reject")
    n_off = host_n([0, 270, 358, 570, 773], 4, 773, cs)
    check("negative: chunk_offsets all 0", sl, st, dA, init, cs, offsets=torch.zeros(n_off, dtype=torch.int32),
          expect="reject")
    print(f"worst err/tol: out {worst['out']:.3f}, final {worst['final']:.3f}")
    if failures:
        print("FAILED:", ", ".join(failures))
        raise SystemExit(1)
    print("all checks hold")
