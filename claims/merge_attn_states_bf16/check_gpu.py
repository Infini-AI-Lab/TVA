"""GPU check of the claim of `merge_attn_states_bf16`.

Runs the kernel's own host (`source/merge_attn_states.py`, imported by
path) on the host's test launches and on launches at the edges of `Inst.ok` and `Pre`, computes
`Post`'s reference and tolerances from the inputs in float64 exactly as `Claim.lean` states them
(`corr`, `refMax`, `side`, `refSum`, `refLse`, `refOut`, `lseTol`, `bf16Round`, `outTol`,
`absTol`), and reports, per output, the largest error / tolerance and that the exactness clauses
(both sides empty: lse `-inf`, output `+0`; copied tokens: bit for bit) and the frame (every input
unchanged; the outputs are fresh and wholly written) hold. Each launch runs with `return_lse`
`True` and `False` (`OUTPUT_LSE`). Negative tests: wrong outputs (all zero, a side dropped, an
unweighted average, swapped weights, the lse's log term dropped, zeroed copied tokens) that
`Post` must reject.

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source",
                   "merge_attn_states.py")
spec = importlib.util.spec_from_file_location("merge_attn_states_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = "cuda"
LSE_TOL = 2.0 ** -16
BF16_ROUND = 2.0 ** -8
OUT_TOL = 2.0 ** -13
ABS_TOL = 2.0 ** -133


def reference(po, pl, so, sl, ptwc):
    """Post's reference, float64, for the merged tokens t < ptwc: (refLse [H, ptwc], refOut
    [ptwc, H, hs], bothEmpty [H, ptwc]). Inputs are the host's tensors as they were before the
    launch: po/so [T, H, hs] bfloat16, pl/sl [H, T] float32."""
    p = pl[:, :ptwc].double().cpu()
    s = sl[:, :ptwc].double().cpu()
    # corr: +inf is -inf
    p = torch.where(p == math.inf, torch.full_like(p, -math.inf), p)
    s = torch.where(s == math.inf, torch.full_like(s, -math.inf), s)
    ep, es = p == -math.inf, s == -math.inf
    both = ep & es
    # refMax: an empty side does not count
    m = torch.where(ep, s, torch.where(es, p, torch.maximum(p, s)))
    m = torch.where(both, torch.zeros_like(m), m)  # unused where both are empty
    wp = torch.where(ep, torch.zeros_like(p), torch.exp(p - m))
    ws = torch.where(es, torch.zeros_like(s), torch.exp(s - m))
    tot = wp + ws
    ref_lse = torch.log(tot) + m
    pv = po[:ptwc].double().cpu()  # [ptwc, H, hs]
    sv = so[:ptwc].double().cpu()
    wpT, wsT, totT = wp.T[:, :, None], ws.T[:, :, None], tot.T[:, :, None]
    ref_out = (pv * wpT + sv * wsT) / totT
    return ref_lse, ref_out, both, pv, sv


def storage_bytes(t):
    """A tensor's whole storage (the padding lanes of a padded head included), as bytes."""
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage()).clone()


def run_case(name, po, pl, so, sl, ptwc, worst, ret_lse=True):
    """One launch: po/so may be views of padded storage; pl/sl may be transposed views;
    `ret_lse` is the host's `return_lse` (False: `output_lse` is `None`, `OUTPUT_LSE = False`)."""
    snaps = [storage_bytes(t) for t in (po, pl, so, sl)]
    out, out_lse = mod.merge_attn_states(po, pl, so, sl, ret_lse, ptwc)
    torch.cuda.synchronize()
    ok = True
    # frame: every input storage unchanged; the outputs are fresh and every element is written
    for a, t in zip(snaps, (po, pl, so, sl)):
        if not torch.equal(a, storage_bytes(t)):
            print(f"  {name}: FRAME VIOLATED (an input changed)")
            ok = False
    if (out_lse is None) == ret_lse:
        print(f"  {name}: output_lse {'missing' if ret_lse else 'returned'}"); ok = False
    ok2, r, ro = post(name, po, pl, so, sl, ptwc, out, out_lse, verbose=True)
    ok = ok and ok2
    worst["lse"] = max(worst["lse"], r)
    worst["out"] = max(worst["out"], ro)
    T, H, hs = po.shape
    print(f"{name}: T={T} H={H} hs={hs} ptwc={ptwc} return_lse={ret_lse}: lse err/tol {r:.3e}, "
          f"out err/tol {ro:.3e}{'' if ok else '  FAILED'}")
    worst["ok"] = worst["ok"] and ok
    return out, out_lse


def post(name, po, pl, so, sl, ptwc, out, out_lse, verbose=False):
    """`Post` of `Claim.lean` on the outputs `out` (and `out_lse`, `None` without `output_lse`)
    against the inputs: (exactness clauses hold, worst lse err/tol, worst output err/tol)."""
    T, H, hs = po.shape
    k = T if ptwc is None else ptwc
    ok = True
    r = ro = 0.0
    say = print if verbose else (lambda *a: None)
    # copied tokens t >= ptwc: bit for bit
    if k < T:
        if not torch.equal(out[k:].view(torch.int16), so[k:].contiguous().view(torch.int16)):
            say(f"  {name}: copy of output NOT bit-exact"); ok = False
        if out_lse is not None and not torch.equal(out_lse[:, k:].contiguous().view(torch.int32),
                                                   sl[:, k:].contiguous().view(torch.int32)):
            say(f"  {name}: copy of lse NOT bit-exact"); ok = False
    if k > 0:
        ref_lse, ref_out, both, pv, sv = reference(po, pl, so, sl, k)
        cl = out_lse[:, :k].double().cpu() if out_lse is not None else None
        w = out[:k].cpu()
        wd = w.double()
        # both sides empty: lse exactly -inf, output exactly +0 (pattern 0)
        if both.any():
            if cl is not None and not bool((cl[both] == -math.inf).all()):
                say(f"  {name}: both-empty lse not -inf"); ok = False
            bothT = both.T  # [k, H]
            wb = w.view(torch.int16)[bothT]
            if not bool((wb == 0).all()):
                say(f"  {name}: both-empty output not +0"); ok = False
        ne = ~both
        if ne.any():
            if cl is not None:
                if not bool(torch.isfinite(cl[ne]).all()):
                    say(f"  {name}: lse not finite"); ok = False
                tol = LSE_TOL * (1 + ref_lse[ne].abs())
                r = ((cl[ne] - ref_lse[ne]).abs() / tol).max().item()
            neT = ne.T[:, :, None].expand(-1, -1, hs)
            wn, rn = wd[neT], ref_out[neT]
            if not bool(torch.isfinite(wn).all()):
                say(f"  {name}: output not finite"); ok = False
            tolo = BF16_ROUND * rn.abs() + OUT_TOL * torch.maximum(pv[neT].abs(), sv[neT].abs()) + ABS_TOL
            ro = ((wn - rn).abs() / tolo).max().item()
    return ok, r, ro


def negatives(gen, worst):
    """Wrong outputs `Post` must reject, on the host's case (4096, 32, 128, 128): each is the
    kernel's output with one typical mistake."""
    po, pl, so, sl = host_inputs(gen, 4096, 32, 128, 0, False)
    out, out_lse = mod.merge_attn_states(po, pl, so, sl, True, 128)
    torch.cuda.synchronize()
    k = 128
    pc, sc = corr_t(pl), corr_t(sl)
    wrong = {}
    z = out.clone(); z[:k] = 0; wrong["all-zero merged output"] = (z, out_lse)
    a = out.clone(); a[:k] = so[:k]; wrong["suffix output only (prefix dropped)"] = (a, out_lse)
    b = out.clone(); b[:k] = po[:k]; wrong["prefix output only (suffix dropped)"] = (b, out_lse)
    m = torch.maximum(pc, sc)
    hp = torch.where(torch.isfinite(m), torch.exp(pc - m), torch.zeros_like(m))
    hs_ = torch.where(torch.isfinite(m), torch.exp(sc - m), torch.zeros_like(m))
    avg = (po.float() + so.float()) / 2
    c = out.clone(); c[:k] = torch.where(torch.isfinite(m).T[:k, :, None], avg[:k],
                                         torch.zeros_like(avg[:k])).to(out.dtype)
    wrong["unweighted average"] = (c, out_lse)
    tot = hp + hs_
    sw = torch.where(tot > 0, hs_ / tot, torch.zeros_like(tot)).T[:, :, None]
    pw = torch.where(tot > 0, hp / tot, torch.zeros_like(tot)).T[:, :, None]
    e = out.clone(); e[:k] = (po.float()[:k] * sw[:k] + so.float()[:k] * pw[:k]).to(out.dtype)
    wrong["weights swapped"] = (e, out_lse)
    f = out_lse.clone(); f[:, :k] = torch.where(torch.isfinite(m[:, :k]), m[:, :k], f[:, :k])
    wrong["lse = max (log term dropped)"] = (out, f)
    g = out.clone(); g[k:] = 0; wrong["copied tokens zeroed"] = (g, out_lse)
    all_rejected = True
    for nm, (o, ol) in wrong.items():
        ok, r, ro = post(nm, po, pl, so, sl, k, o, ol)
        rejected = (not ok) or r > 1 or ro > 1
        all_rejected = all_rejected and rejected
        print(f"negative '{nm}': exactness {'holds' if ok else 'fails'}, lse err/tol {r:.3e}, "
              f"out err/tol {ro:.3e}: {'rejected' if rejected else 'ACCEPTED (BAD)'}")
    worst["ok"] = worst["ok"] and all_rejected


def host_inputs(gen, num_tokens, num_heads, head_size, pad, lse_t, both_empty_cols=False):
    """The host's test inputs (source lines 346-373)."""
    if lse_t:
        prefix_lse = torch.randn((num_tokens, num_heads), generator=gen).T
        suffix_lse = torch.randn((num_tokens, num_heads), generator=gen).T
    else:
        prefix_lse = torch.randn((num_heads, num_tokens), generator=gen)
        suffix_lse = torch.randn((num_heads, num_tokens), generator=gen)
    mp = torch.rand((num_heads, num_tokens), generator=gen) < 0.1
    ms = torch.rand((num_heads, num_tokens), generator=gen) < 0.1
    both = mp & ms
    mp, ms = mp & ~both, ms & ~both
    prefix_lse[mp] = float("inf")
    suffix_lse[ms] = float("inf")
    r = torch.rand((num_heads, num_tokens), generator=gen)
    prefix_lse[r < 0.05] = float("-inf")
    suffix_lse[r < 0.05] = float("-inf")
    prefix_lse[(r >= 0.05) & (r < 0.075)] = float("-inf")
    suffix_lse[(r >= 0.075) & (r < 0.1)] = float("-inf")
    if both_empty_cols:
        prefix_lse[:, 2:4] = float("-inf")
        suffix_lse[:, 2:4] = float("-inf")
    prefix_lse, suffix_lse = prefix_lse.to(DEV), suffix_lse.to(DEV)
    shape = (num_tokens, num_heads, head_size + pad)
    po = torch.randn(shape, generator=gen).to(device=DEV, dtype=torch.bfloat16)[..., :head_size]
    so = torch.randn(shape, generator=gen).to(device=DEV, dtype=torch.bfloat16)[..., :head_size]
    return po, prefix_lse, so, suffix_lse


def bf16_bits(gen, shape, kind):
    """bfloat16 tensors at the edges of `Pre`: kind 'any' is every bit pattern (NaN and inf
    included: only for copied tokens or both-empty pairs); 'finite' is every finite pattern
    (subnormals and the largest magnitudes included)."""
    bits = torch.randint(0, 1 << 16, shape, generator=gen, dtype=torch.int32).to(torch.int16)
    x = bits.view(torch.bfloat16)
    if kind == "finite":
        bad = ~torch.isfinite(x.float())
        x = torch.where(bad, torch.zeros_like(x), x)
    return x


def f32_lse(gen, shape):
    """float32 lse values at the edges of `Pre`: any non-NaN, mixing ordinary values, huge
    finite values (up to the float32 maximum), subnormals and both infinities."""
    n = math.prod(shape)
    bits = torch.randint(0, 1 << 31, (n,), generator=gen, dtype=torch.int64)
    sign = torch.randint(0, 2, (n,), generator=gen, dtype=torch.int64) << 31
    raw = ((bits | sign) & 0xFFFFFFFF).to(torch.int64)
    raw = torch.where(raw >= (1 << 31), raw - (1 << 32), raw).to(torch.int32)
    x = raw.view(torch.float32)
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    pick = torch.randint(0, 6, (n,), generator=gen)
    ordinary = torch.randn(n, generator=gen) * 4
    x = torch.where(pick == 0, ordinary, x)
    x = torch.where(pick == 1, ordinary + 1e30, x)
    x = torch.where(pick == 2, torch.full_like(x, float("inf")), x)
    x = torch.where(pick == 3, torch.full_like(x, float("-inf")), x)
    return x.view(shape)


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    worst = {"lse": 0.0, "out": 0.0, "ok": True}
    # the host's test cases (source lines 335-345)
    for (T, H, hs, ptwc, pad, lse_t) in [
        (1, 8, 128, None, 0, False), (6, 8, 128, None, 0, False), (256, 4, 32, None, 0, False),
        (613, 16, 48, 128, 0, False), (512, 32, 96, 128, 0, True), (1024, 48, 128, None, 64, True),
        (1536, 64, 256, 1, 0, False), (4096, 32, 128, 128, 0, False), (300, 8, 64, 0, 8, True),
    ]:
        po, pl, so, sl = host_inputs(gen, T, H, hs, pad, lse_t, both_empty_cols=(T == 6))
        run_case(f"host({T},{H},{hs},{ptwc},{pad},{lse_t})", po, pl, so, sl, ptwc, worst)
        # the host's test runs each case without output_lse too (source lines 387-389)
        run_case(f"host-nolse({T},{H},{hs},{ptwc},{pad},{lse_t})", po, pl, so, sl, ptwc, worst,
                 ret_lse=False)
    # the code base's largest shape, with and without output_lse
    po, pl, so, sl = host_inputs(gen, 4096, 64, 256, 0, False)
    run_case("largest(4096,64,256,128)", po, pl, so, sl, 128, worst)
    run_case("largest-nolse(4096,64,256,128)", po, pl, so, sl, 128, worst, ret_lse=False)
    del po, pl, so, sl
    # edges of Inst.ok: one token, one head, head size 1 (padded 1); head size 1 with 2 heads
    for (T, H, hs, ptwc, pad, lse_t) in [(1, 1, 1, None, 0, False), (3, 2, 1, 2, 1, True),
                                          (2, 3, 3, 1, 5, False), (64, 64, 256, 32, 64, True)]:
        po, pl, so, sl = host_inputs(gen, T, H, hs, pad, lse_t)
        run_case(f"edge({T},{H},{hs},{ptwc},{pad},{lse_t})", po, pl, so, sl, ptwc, worst)
        run_case(f"edge-nolse({T},{H},{hs},{ptwc},{pad},{lse_t})", po, pl, so, sl, ptwc, worst,
                 ret_lse=False)
    # edges of Pre: lse any non-NaN (huge, subnormal, infinite), bfloat16 any finite value
    # (subnormal to the largest), on merged pairs; any bits on copied tokens; any bits in the
    # outputs of both-empty pairs
    for rep in range(4):
        T, H, hs = 512, 16, 128
        ptwc = 384
        pl = f32_lse(gen, (H, T)).to(DEV)
        sl = f32_lse(gen, (H, T)).to(DEV)
        po = bf16_bits(gen, (T, H, hs), "finite").to(DEV)
        so = bf16_bits(gen, (T, H, hs), "finite").to(DEV)
        if rep == 1:
            # cancellation: equal lse, opposite outputs
            sl[:, :ptwc] = pl[:, :ptwc]
            so[:ptwc] = -po[:ptwc]
        if rep == 2:
            # the largest finite lse on both sides (m + log 2 rounds to m)
            pl[:, :ptwc] = 3.4028234663852886e38
            sl[:, :ptwc] = 3.4028234663852886e38
        # both-empty pairs: any bits in their outputs
        both = (corr_t(pl[:, :ptwc]) == -math.inf) & (corr_t(sl[:, :ptwc]) == -math.inf)
        anyb = bf16_bits(gen, (ptwc, H, hs), "any").to(DEV)
        bT = both.T[:, :, None].expand(-1, -1, hs)
        po[:ptwc] = torch.where(bT, anyb, po[:ptwc])
        so[:ptwc] = torch.where(bT, anyb.flip(0), so[:ptwc])
        # copied tokens: any bits
        so[ptwc:] = bf16_bits(gen, (T - ptwc, H, hs), "any").to(DEV)
        raw = torch.randint(-(1 << 31), 1 << 31, (H, T - ptwc), generator=gen,
                            dtype=torch.int64).to(torch.int32)
        sl[:, ptwc:] = raw.view(torch.float32).to(DEV)
        run_case(f"pre-edge{rep}", po, pl, so, sl, ptwc, worst)
        run_case(f"pre-edge{rep}-nolse", po, pl, so, sl, ptwc, worst, ret_lse=False)
    negatives(gen, worst)
    print(f"WORST: lse err/tol {worst['lse']:.3e} (margin {1 / max(worst['lse'], 1e-300):.1f}x), "
          f"out err/tol {worst['out']:.3e} (margin {1 / max(worst['out'], 1e-300):.1f}x); "
          f"exactness and frame {'hold' if worst['ok'] else 'FAIL'}")
    sys.exit(0 if worst["ok"] and worst["lse"] < 1 and worst["out"] < 1 else 1)


def corr_t(x):
    return torch.where(x == math.inf, torch.full_like(x, -math.inf), x)


if __name__ == "__main__":
    main()
