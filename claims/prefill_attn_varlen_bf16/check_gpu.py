"""GPU check of the claim of `prefill_attn_varlen_bf16`.

Runs the kernel's own host (`source/prefill_attention.py`,
imported by path) on the host's test launches, on SETUP.md's model shapes (causal, and the vision
encoders' non-causal launches) and on launches at the edges of `Inst.ok` and `Pre`, causal and
not; computes `Post`'s reference and tolerance from the inputs in float64 exactly as `Claim.lean`
states them (`lastKey`, `score`, `scoreMag` with `liftMag`, `Xrow`, `attnRef`, `attnAbs`, `vSum`,
`trips`, `tolRel`, `tolAbs`); and reports the largest error /
tolerance over every written element, that every written element is finite, and the frame (every
input unchanged; every element of `o` outside the sequences keeps its initial bits: the host's
`torch.empty_like` is made to fill `o` with a sentinel pattern). Negative tests: `Post` must
reject typical wrong outputs (all zero, the other mode's output, a key block or a key left out,
the wrong key/value head). It also prints `tol / |attnRef|` (median, 90th percentile) at two
real shapes.

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
                   "prefill_attention.py")
spec = importlib.util.spec_from_file_location("prefill_attention_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = "cuda"
BLOCK = 128
SENTINEL = -12345  # an int16 pattern (a bfloat16 NaN-free value) the fresh `o` starts with


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def tol_rel(BD, T, X):
    """`tolRel BD T X` of Claim.lean, in float64 (T, X tensors):
    2^-6 + T 2^-14 + (BD + 16) X 2^-20."""
    return 2.0 ** -6 + T * 2.0 ** -14 + (BD + 16) * X * 2.0 ** -20


def lift(x):
    """`Device.liftMag .bf16`: 0 for 0, else max(|x|, 2^-126)."""
    a = x.abs()
    return torch.where(x == 0, torch.zeros_like(a), torch.clamp(a, min=2.0 ** -126))


def seq_attn(q, k, v, start, n, sigma, causal, rows=512, drop=None, kvmap=None):
    """Attention of one sequence in float64, head by head (yields `(h, ref, tol, tolRel max)`,
    `ref` and `tol` `[n, Lk]`, so that only one head's float64 blocks are held at a time):
    `attnRef` over the keys
    each query attends (`j <= p` when causal, every `j < n` otherwise), with `Post`'s tolerance
    (`Xrow`, `attnAbs`, `vSum`, `trips (lastKey)`). `drop(p, j)` (a mask of keys to leave out) and
    `kvmap(h)` (the key/value head of query head `h`) give the wrong outputs of the negative
    tests; the tolerance is always `Post`'s."""
    H, HKV, Lk = q.shape[1], k.shape[1], q.shape[2]
    group = H // HKV
    BD = next_pow2(Lk)
    for h in range(H):
        ref = torch.empty((n, Lk), dtype=torch.float64, device=DEV)
        tol = torch.empty_like(ref)
        tolrel_max = 0.0
        g = h // group
        qh = q[start:start + n, h, :].double()
        kh = k[start:start + n, g, :].double()
        vh = v[start:start + n, g, :].double()
        gw = g if kvmap is None else kvmap(h)
        kw = k[start:start + n, gw, :].double()
        vw = v[start:start + n, gw, :].double()
        lq, lk = lift(qh), lift(kh)
        if causal:
            vsum = torch.cumsum(vh.abs(), dim=0)  # vSum[p, e] = sum_{j <= p} |v_j[e]|
        else:
            vsum = vh.abs().sum(dim=0, keepdim=True).expand(n, Lk)  # sum_{j < n} |v_j[e]|
        for r0 in range(0, n, rows):
            r1 = min(n, r0 + rows)
            p = torch.arange(r0, r1, device=DEV)
            j = torch.arange(0, n, device=DEV)
            if causal:
                att = j[None, :] <= p[:, None]
            else:
                att = torch.ones((r1 - r0, n), dtype=torch.bool, device=DEV)
            s = sigma * (qh[r0:r1] @ kh.T)
            mag = sigma * (lq[r0:r1] @ lk.T)
            s = torch.where(att, s, torch.full_like(s, -math.inf))
            mag = torch.where(att, mag, torch.full_like(mag, -math.inf))
            X = mag.max(dim=1).values  # Xrow
            w = torch.softmax(s, dim=1)  # e^{score} / sum e^{score}, over the attended keys
            absw = w @ vh.abs()
            last = p if causal else torch.full_like(p, n - 1)  # lastKey
            T = (last // BLOCK + 1).double()
            tr = tol_rel(BD, T, X)
            tolrel_max = max(tolrel_max, float(tr.max()))
            tol[r0:r1] = tr[:, None] * absw + 2.0 ** -80 * (1 + vsum[r0:r1])
            if drop is None and kvmap is None:
                ref[r0:r1] = w @ vh
            else:
                sw = sigma * (qh[r0:r1] @ kw.T)
                keep = att if drop is None else att & ~drop(p[:, None], j[None, :])
                sw = torch.where(keep, sw, torch.full_like(sw, -math.inf))
                ref[r0:r1] = torch.softmax(sw, dim=1) @ vw
            del s, mag, att, w, absw, tr
        yield h, ref, tol, tolrel_max
        del ref, tol


def post_ratio(o, q, k, v, starts, lens, sigma, causal, rows=512, stats=None):
    """`Post`'s value clause on the output `o`: the largest error / tolerance over every
    written element, whether every written element is finite, and where the worst is."""
    worst, at, fin, trm = 0.0, None, True, 0.0
    for s0, n in zip(starts, lens):
        if n == 0:
            continue
        fin = fin and bool(torch.isfinite(o[s0:s0 + n].float()).all())
        for h, ref, tol, tr in seq_attn(q, k, v, s0, n, sigma, causal, rows):
            trm = max(trm, tr)
            err = (o[s0:s0 + n, h].double() - ref).abs()
            ratio = err / tol
            r = ratio.max().item()
            if r > worst:
                worst = r
                i = int(ratio.flatten().argmax())
                at = (float(err.flatten()[i]), float(tol.flatten()[i]))
            if stats is not None:
                stats.append((tol / ref.abs()).flatten().float().cpu())
            del err, ratio
    return worst, fin, at, trm


def wrong_outputs(q, k, v, starts, lens, sigma, causal, total, other_mode_out):
    """Typical wrong outputs (each in `bfloat16`, outside the sequences `0`): all zero; the
    kernel's output in the other mode (causal for a non-causal launch and back); the last key
    block of every row left out; the first key left out; each query head on the wrong key/value
    head (`h % num_kv_heads`, when grouped)."""
    H, HKV = q.shape[1], k.shape[1]
    outs = {"all-zero": torch.zeros_like(other_mode_out),
            "other-mode": other_mode_out}

    def build(**kw):
        o = torch.zeros(other_mode_out.shape, dtype=torch.bfloat16, device=DEV)
        for s0, n in zip(starts, lens):
            if n > 0:
                for h, ref, _, _ in seq_attn(q, k, v, s0, n, sigma, causal, **kw):
                    o[s0:s0 + n, h] = ref.to(torch.bfloat16)
        return o

    def last_block(p, j):  # the keys of the row's last trip (`lastKey // 128`), but key 0
        last = p if causal else j.max()
        return (j // BLOCK == last // BLOCK) & (j > 0)

    outs["drop-last-key-block"] = build(drop=last_block)
    outs["drop-first-key"] = build(drop=lambda p, j: (j == 0) & ((p > 0) if causal else (j.max() > 0)))
    if HKV > 1 and HKV < H:
        outs["wrong-kv-head"] = build(kvmap=lambda h: h % HKV)
    return outs


def run_case(name, H, HKV, Lk, seq_lens, gen, scale_qk=1.0, v_scale=1.0, sm_scale=None,
             gaps=None, worst=None, rows=512, uniform_v=None, fused=False, causal=True,
             unit_rms=False, negatives=False, stats=None):
    """One host launch, `is_causal=causal`: sequences of `seq_lens`, packed with `gaps` tokens
    before each (default none), `randn` inputs (q, k scaled by `scale_qk`, v by `v_scale`), or
    the views of one fused `qkv` tensor (`fused`), or `q`, `k`, `v` normalised to unit RMS per
    head (`unit_rms`)."""
    gaps = gaps or [0] * len(seq_lens)
    starts, t = [], 0
    for gp, n in zip(gaps, seq_lens):
        t += gp
        starts.append(t)
        t += n
    total = t + 3  # three tokens after the last sequence, in no sequence
    q = (torch.randn((total, H, Lk), generator=gen) * scale_qk).to(device=DEV, dtype=torch.bfloat16)
    k = (torch.randn((total, HKV, Lk), generator=gen) * scale_qk).to(device=DEV, dtype=torch.bfloat16)
    v = (torch.randn((total, HKV, Lk), generator=gen) * v_scale).to(device=DEV, dtype=torch.bfloat16)
    if fused:
        # SETUP.md's strided inputs: q, k, v views of one fused [total, (H + 2*HKV)*Lk] tensor
        qkv = torch.randn((total, (H + 2 * HKV) * Lk), generator=gen).to(device=DEV,
                                                                         dtype=torch.bfloat16)
        q = qkv[:, :H * Lk].view(total, H, Lk)
        k = qkv[:, H * Lk:(H + HKV) * Lk].view(total, HKV, Lk)
        v = qkv[:, (H + HKV) * Lk:].view(total, HKV, Lk)
    if unit_rms:
        q, k, v = (x.float() / x.float().pow(2).mean(-1, keepdim=True).sqrt()
                   for x in (q, k, v))
        q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
    if uniform_v is not None:
        # q = 0: every score 0, uniform weights; v constant along the tokens, one value per
        # component: the bfloat16 roundings of the weights all push the same way
        q = torch.zeros_like(q)
        c = torch.tensor(uniform_v, dtype=torch.float32)[:Lk].to(torch.bfloat16)
        v = c.to(DEV)[None, None, :].expand(total, HKV, Lk).contiguous()
    b_start_loc = torch.tensor(starts, dtype=torch.int32, device=DEV)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    snaps = [x.clone() for x in (q, k, v, b_start_loc, b_seq_len)]
    real_empty_like = torch.empty_like

    def sentinel_empty_like(x, *a, **kw):
        y = real_empty_like(x, *a, **kw)
        y.view(torch.int16).fill_(SENTINEL)
        return y

    torch.empty_like = sentinel_empty_like
    try:
        o = mod.context_attention_fwd(q, k, v, b_start_loc, b_seq_len, max(seq_lens),
                                      is_causal=causal, sm_scale=sm_scale)
    finally:
        torch.empty_like = real_empty_like
    torch.cuda.synchronize()
    ok = True
    for a, b in zip(snaps, (q, k, v, b_start_loc, b_seq_len)):
        if not torch.equal(a, b):
            print(f"  {name}: FRAME VIOLATED (an input changed)")
            ok = False
    covered = torch.zeros(total, dtype=torch.bool, device=DEV)
    for s0, n in zip(starts, seq_lens):
        covered[s0:s0 + n] = True
    if not bool((o.view(torch.int16)[~covered] == SENTINEL).all()):
        print(f"  {name}: FRAME VIOLATED (an element outside the sequences changed)")
        ok = False
    sigma = (1.0 / (Lk ** 0.5)) if sm_scale is None else sm_scale
    r, fin, at, trm = post_ratio(o, q, k, v, starts, seq_lens, sigma, causal, rows, stats)
    if not fin:
        print(f"  {name}: an output is not finite")
        ok = False
    worst["tolrel_max"] = max(worst["tolrel_max"], trm)
    if r > worst["out"]:
        worst["out"], worst["at"] = r, (name,) + at
    print(f"{name}: {'causal' if causal else 'non-causal'} heads {H}/{HKV} Lk {Lk} lens "
          f"{seq_lens if len(seq_lens) < 8 else str(len(seq_lens)) + ' seqs'} maxLen {max(seq_lens)}:"
          f" err/tol {r:.3e}{'' if ok else '  FAILED'}")
    worst["ok"] = worst["ok"] and ok
    if negatives:
        other = mod.context_attention_fwd(q, k, v, b_start_loc, b_seq_len, max(seq_lens),
                                          is_causal=not causal, sm_scale=sm_scale)
        for wname, wo in wrong_outputs(q, k, v, starts, seq_lens, sigma, causal, total,
                                       other).items():
            wr = post_ratio(wo, q, k, v, starts, seq_lens, sigma, causal, rows)[0]
            rejected = wr > 1
            print(f"  negative {wname}: worst err/tol {wr:.3e}: "
                  f"{'rejected' if rejected else 'ACCEPTED (Post too weak)'}")
            worst["ok"] = worst["ok"] and rejected
            worst["neg_min"] = min(worst["neg_min"], wr)
    torch.cuda.empty_cache()


def summary(label, stats):
    x = torch.cat(stats)
    x = x[torch.isfinite(x)]
    print(f"{label}: tol/|attnRef| median {x.median().item():.3g}, p90 "
          f"{x.quantile(0.9).item() if x.numel() < 2 ** 24 else x[::max(1, x.numel() // 2 ** 23)].quantile(0.9).item():.3g}")


def main():
    gen = torch.Generator().manual_seed(0)
    worst = {"out": 0.0, "ok": True, "tolrel_max": 0.0, "at": None, "neg_min": math.inf}
    host = [
        (4, 4, 128, [8, 12]), (4, 4, 96, [8, 12]), (4, 4, 80, [8, 12]), (4, 4, 13, [8, 12]),
        (32, 8, 128, [1, 127, 128, 129, 1000]), (28, 4, 128, [255, 3, 2048]),
        (16, 16, 80, [1024, 64, 1]), (8, 1, 64, [300, 513]),
    ]
    # the host's test cases (source lines 270-279), causal, and the same shapes non-causal
    for causal in (True, False):
        for H, HKV, Lk, lens in host:
            run_case("host", H, HKV, Lk, lens, gen, worst=worst, causal=causal)
    # SETUP.md's strided inputs: a fused qkv projection (Llama-3-8B's heads, the host's lengths)
    run_case("strided-fused-qkv", 32, 8, 128, [1, 127, 128, 129, 1000], gen, fused=True,
             worst=worst, negatives=True)
    # SETUP.md's model shapes at their largest: 64 heads, 8 KV heads, head dim 128, 8192 tokens
    st_causal = []
    run_case("setup-large", 64, 8, 128, [8192, 4096], gen, worst=worst, stats=st_causal)
    # the vision encoders (is_causal=False, fused qkv views)
    grid = [4] * 9 + [1]  # a 37 x 37 merged grid in windows of 4 x 4: 100 windows
    windows = [a * b * 4 for a in grid for b in grid]
    st_vit = []
    run_case("vit-full", 16, 16, 80, [5476], gen, fused=True, causal=False, worst=worst,
             negatives=True, stats=st_vit)
    run_case("vit-window", 16, 16, 80, windows, gen, fused=True, causal=False, worst=worst,
             negatives=True)
    run_case("vit-max-tp8", 2, 2, 80, [65536], gen, fused=True, causal=False, worst=worst,
             rows=256)
    run_case("vit-video", 16, 16, 80, [3072] * 16, gen, fused=True, causal=False, worst=worst)
    run_case("vit-qwen3", 16, 16, 72, [4096, 1024], gen, fused=True, causal=False, worst=worst)
    run_case("vit-mimo", 32, 8, 64, [4096, 1024], gen, fused=True, causal=False, worst=worst,
             negatives=True)
    run_case("vit-scale1", 16, 16, 64, [1024, 256], gen, sm_scale=1.0, unit_rms=True,
             causal=False, worst=worst, negatives=True)
    # edges of Inst.ok: head dim 1 and 256, 64 heads on one KV head, gaps and empty sequences,
    # a custom sm_scale, the longest covered prompt (2^17 tokens)
    # head dims up to 8 (BLOCK_DMODEL < 16) are rejected on the host: `tl.dot` needs K >= 16
    for lk in (1, 8):
        try:
            run_case(f"edge-lk{lk}", 2, 1, lk, [1, 5], gen, worst=worst)
            print(f"edge-lk{lk}: UNEXPECTEDLY COMPILED")
            worst["ok"] = False
        except Exception as ex:  # triton.compiler.errors.CompilationError
            print(f"edge-lk{lk}: rejected on the host as expected ({type(ex).__name__})")
    for causal in (True, False):
        run_case("edge-lk9", 2, 1, 9, [1, 5, 300], gen, worst=worst, causal=causal)
        run_case("edge-lk256", 4, 2, 256, [129, 700], gen, worst=worst, causal=causal)
        run_case("edge-mqa", 64, 1, 64, [200, 1, 257], gen, worst=worst, causal=causal)
        run_case("edge-gaps", 3, 3, 48, [0, 7, 0, 130, 2], gen, gaps=[2, 5, 1, 0, 9],
                 worst=worst, causal=causal)
        run_case("edge-smscale", 4, 2, 64, [333, 64], gen, sm_scale=0.5, worst=worst,
                 causal=causal)
    run_case("edge-long", 1, 1, 64, [2 ** 17], gen, worst=worst, rows=256)
    run_case("edge-long", 1, 1, 64, [2 ** 17], gen, worst=worst, rows=128, causal=False)
    # edges of Pre: large scores (one-hot softmax, `Xrow` in the hundreds), large and tiny values
    cs = [1.0 + i / 128 for i in range(64)]  # every bfloat16 in [1, 1.5), then their negatives
    for causal in (True, False):
        run_case("pre-bigscore", 8, 2, 128, [600, 129], gen, scale_qk=6.0, worst=worst,
                 causal=causal)
        run_case("pre-bigv", 4, 4, 64, [300], gen, v_scale=1e30, worst=worst, causal=causal)
        run_case("pre-tinyv", 4, 4, 64, [300], gen, v_scale=1e-37, worst=worst, causal=causal)
        run_case("pre-uniform", 2, 1, 128, [255, 600, 129], gen,
                 uniform_v=cs + [-c for c in cs], worst=worst, causal=causal)
    summary("causal 64/8 heads, head dim 128, [8192, 4096]", st_causal)
    summary("non-causal 16/16 heads, head dim 80, [5476] (vit-full)", st_vit)
    print(f"WORST: out err/tol {worst['out']:.3e} (margin {1 / max(worst['out'], 1e-300):.2f}x) at "
          f"(case, err, tol) = {worst['at']}; largest tolRel {worst['tolrel_max']:.3e}; "
          f"smallest negative err/tol {worst['neg_min']:.3e}; finiteness, frame and negatives "
          f"{'hold' if worst['ok'] else 'FAIL'}")
    print(f"PEAK device memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    sys.exit(0 if worst["ok"] and worst["out"] < 1 else 1)


if __name__ == "__main__":
    main()
