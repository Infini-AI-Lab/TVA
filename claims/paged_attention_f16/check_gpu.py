"""GPU check of the prepared claim of `paged_attention_f16` (claims/README.md, check 4).

Runs the kernel's own host (`source/paged_attention.py`, imported by path): `paged_attention(q,
k_cache, v_cache, block_tables, seq_lens)` on the test's three configurations (its `make_cache`
and generator, seed 0) and on launches at the edges of `Inst.ok` and `Pre`. For every element it
computes, in float64 and exactly as `Claim.lean` states them, the reference `attnRef`, its scale
`attnAbs` (the softmax weights `exp(scale·score)` over the sequence's tokens read through the block
table, `scale` the double), the sequence's page count `T = cdiv(seq_len, PAGE_SIZE)` and score bound
`X = max_t scale·scoreAbs` (the smallest `X` the claim allows, where its tolerance is tightest),
and the tolerance `tolA T X = (9·T/8 + 8·X + 8)·2^-21`. It reports the largest
`|c - attnRef| / (tolA·(attnAbs + |attnRef|) + 2^-11·|attnRef| + 2^-24)` over the elements of
sequences whose values are covered (`SeqVals`), that those elements are finite, and the frame: every input
unchanged bit for bit, and `out` (from the host's `torch.empty_like`) a view of a longer buffer
whose tail keeps a sentinel. On some launches it also checks that `Post` rejects wrong outputs
(rounded to `float16`): all zeros, the last or the first page dropped, the neighbouring key/value
head, the scores without `scale`; and it reports the tolerance against the output, `tol / |attnRef|`
(median, 90th percentile and largest).

Run: CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py
"""
import sys; sys.dont_write_bytecode = True  # keep source/ free of __pycache__ (python -I ignores PYTHONDONTWRITEBYTECODE)

import importlib.util
import math
import os

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "paged_attention.py")
spec = importlib.util.spec_from_file_location("paged_attention_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = "cuda"
SENT = -12345  # an int16 pattern, a finite float16


def tol_a(T, X):
    """`tolA T X` of Claim.lean: `(9·T/8 + 8·X + 8)·2^-21`."""
    return (9 * T / 8 + 8 * X + 8) / 2.0 ** 21


WORST = {"r": 0.0, "ok": True, "at": None, "tolA_max": 0.0, "elems": 0}
NEGS = ("zero", "drop-last-page", "drop-first-page", "next-kv-head", "no-scale")


def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def check(name, q, k_cache, v_cache, block_tables, seq_lens, out, negatives=False):
    """`Post` on every element of `out` (sequences with covered values); with `negatives`, whether
    `Post` rejects each wrong output (some element outside its tolerance)."""
    S, H, D = q.shape
    P, KV = k_cache.shape[1], k_cache.shape[2]
    group = H // KV
    sigma = D ** -0.5  # the host's `head_dim**-0.5`, a double
    r_case = 0.0
    neg = {k: [0, 0, 0] for k in NEGS}  # rows tried, rows rejected, elements accepted
    rel = []
    for s in range(S):
        n = int(seq_lens[s])
        if n < 1:
            continue  # SeqVals needs a token: Post says only that a float16 is stored
        t = torch.arange(n, device=DEV)
        pages = block_tables[s, t // P].long()
        k = k_cache[pages, t % P].double()  # [n, KV, D]
        v = v_cache[pages, t % P].double()
        qs = q[s].double()  # [H, D]
        T = float((n + P - 1) // P)
        for h in range(H):
            g = h // group
            kk, vv = k[:, g, :], v[:, g, :]
            if not (bool(torch.isfinite(qs[h]).all()) and bool(torch.isfinite(kk).all())
                    and bool(torch.isfinite(vv).all()) and bool((vv.abs() <= 2.0 ** 15).all())):
                continue
            score = kk @ qs[h]
            sabs = kk.abs() @ qs[h].abs()
            X = sigma * float(sabs.max())
            if X > 2.0 ** 10:
                continue
            w = torch.softmax(sigma * score, dim=0)
            ref = w @ vv
            absw = w @ vv.abs()
            tolA = tol_a(T, X)
            tol = tolA * (absw + ref.abs()) + 2.0 ** -11 * ref.abs() + 2.0 ** -24
            rel.append(tol / ref.abs())
            c = out[s, h].double()
            if negatives:
                fakes = {"zero": torch.zeros_like(ref)}
                if T >= 2:
                    nd = (int(T) - 1) * P
                    fakes["drop-last-page"] = torch.softmax(sigma * score[:nd], dim=0) @ vv[:nd]
                    fakes["drop-first-page"] = torch.softmax(sigma * score[P:], dim=0) @ vv[P:]
                if KV > 1:
                    g2 = (g + 1) % KV
                    w2 = torch.softmax(sigma * (k[:, g2, :] @ qs[h]), dim=0)
                    fakes["next-kv-head"] = w2 @ v[:, g2, :]
                fakes["no-scale"] = torch.softmax(score, dim=0) @ vv
                for key, f in fakes.items():
                    f = f.to(torch.float16).double()
                    acc = (f - ref).abs() <= tol
                    neg[key][0] += 1
                    neg[key][1] += int(not bool(acc.all()))
                    neg[key][2] += int(acc.sum())
            if not bool(torch.isfinite(c).all()):
                print(f"  {name}: seq {s} head {h}: an output is not finite  FAILED")
                WORST["ok"] = False
            ratio = (c - ref).abs() / tol
            r = float(ratio.max())
            WORST["elems"] += D
            WORST["tolA_max"] = max(WORST["tolA_max"], float(tolA))
            if r > r_case:
                r_case = r
            if r > WORST["r"]:
                e = int(ratio.argmax())
                WORST["r"] = r
                WORST["at"] = (name, s, h, e, float((c - ref).abs()[e]), float(tol[e]), X, T, float(absw[e]))
    if negatives:
        for key, (rows, rej, acc) in neg.items():
            if rows == 0:
                continue
            ok = rej > 0  # Post (over the whole output) rejects it
            if not ok:
                WORST["ok"] = False
            print(f"  {name}: wrong output {key}: rejected ({rej}/{rows} rows fail, "
                  f"{acc / (rows * D):.3f} of the elements within tolerance)  {'ok' if ok else 'NOT REJECTED'}")
    if rel:
        rel = torch.cat(rel)
        print(f"  {name}: tol / |attnRef| median {float(rel.median()):.3e}, 90th percentile "
              f"{float(rel.quantile(0.9)):.3e}, largest {float(rel.max()):.3e}")
    return r_case


def run(name, H, KV, D, P, seq_lens, gen, num_pages=None, q_scale=1.0, v_scale=1.0, shared=False,
        strided=False, negatives=False):
    """One launch of the host, inputs as the test makes them (`make_cache`)."""
    num_pages = num_pages or sum(triton.cdiv(n, P) for n in seq_lens) + 50
    k_cache, v_cache, block_tables, lens = mod.make_cache(seq_lens, KV, D, P, num_pages, DEV, gen)
    if v_scale != 1.0:
        v_cache = (v_cache.float() * v_scale).clamp(-2.0 ** 15, 2.0 ** 15).to(torch.float16)
    if shared:  # every sequence reads the same pages (allowed: pages need not be distinct)
        block_tables[:] = block_tables[0:1].expand_as(block_tables) if block_tables.shape[0] > 0 else block_tables
    S = len(seq_lens)
    if strided:  # q a view with gaps between heads and sequences (last stride 1)
        big = torch.randn((S, H, D + 24), generator=gen).to(device=DEV, dtype=torch.float16)
        q = big[:, :, 8:8 + D]
    else:
        q = (torch.randn((S, H, D), generator=gen) * q_scale).to(device=DEV, dtype=torch.float16)
    snaps = [x.clone() for x in (q, k_cache, v_cache, block_tables, lens)]
    real = torch.empty_like
    holder = []

    def empty_like(x, *a, **kw):
        y = torch.empty(x.numel() + 256, dtype=x.dtype, device=x.device)
        y.view(torch.int16).fill_(SENT)
        holder.append(y)
        return y[:x.numel()].view(x.shape)

    torch.empty_like = empty_like
    try:
        out = mod.paged_attention(q, k_cache, v_cache, block_tables, lens)
    finally:
        torch.empty_like = real
    torch.cuda.synchronize()
    for a, b in zip(snaps, (q, k_cache, v_cache, block_tables, lens)):
        if not same(a, b):
            print(f"  {name}: FRAME VIOLATED (an input changed)")
            WORST["ok"] = False
    if not bool((holder[0][-256:].view(torch.int16) == SENT).all()):
        print(f"  {name}: FRAME VIOLATED (a byte past out changed)")
        WORST["ok"] = False
    r = check(name, q, k_cache, v_cache, block_tables, lens, out, negatives)
    print(f"{name}: heads {H}/{KV} D {D} page {P} lens "
          f"{seq_lens if len(seq_lens) <= 8 else str(len(seq_lens)) + ' seqs'}: worst err/tol {r:.3e}")
    del k_cache, v_cache, block_tables, lens, q, out, snaps, holder
    torch.cuda.empty_cache()


def main():
    gen = torch.Generator().manual_seed(0)
    # the test's configurations (source lines 140-144), in its order, from its generator
    for H, KV, D, P, lens in [
        (32, 8, 128, 16, [1, 15, 16, 17, 1000, 4096, 333]),
        (8, 8, 64, 16, [5, 64, 2049]),
        (16, 2, 128, 32, [31, 32, 33, 8191]),
    ]:
        run("test", H, KV, D, P, lens, gen, negatives=True)
    # edges of Inst.ok: head dims and pages from the smallest to 256, one KV head, many sequences,
    # pages shared by every sequence, a strided q, the longest sequences (2^16 tokens)
    run("edge-small", 4, 2, 1, 2, [1, 2, 3, 77], gen)
    run("edge-page1", 8, 2, 128, 1, [1, 2, 1000, 4097], gen, negatives=True)
    run("edge-page1-d1", 4, 2, 1, 1, [1, 3, 500], gen)
    run("edge-d256", 4, 1, 256, 256, [255, 256, 257, 3000], gen)
    run("edge-p2", 8, 4, 64, 2, [1, 2, 3, 999], gen)
    run("edge-mqa", 64, 1, 128, 16, [1, 500, 4097], gen)
    run("edge-manyseqs", 4, 4, 64, 16, [1 + (i * 37) % 300 for i in range(200)], gen)
    run("edge-shared", 8, 2, 128, 16, [700, 700, 700], gen, shared=True)
    run("edge-strided", 16, 2, 128, 32, [31, 900, 4100], gen, strided=True)
    run("edge-long", 2, 1, 128, 16, [2 ** 16], gen)
    run("edge-long-page1", 2, 1, 128, 1, [2 ** 16], gen)
    # long decode sequences, 32K to 64K tokens, at the test's head layouts
    run("long", 32, 8, 128, 16, [32768, 49152, 65536], gen, negatives=True)
    run("long-p32", 16, 2, 128, 32, [32768, 65536], gen, negatives=True)
    run("edge-zero-seq", 4, 2, 64, 16, [0, 5, 0, 70], gen)
    # edges of Pre (SeqVals): sharp attention (scores near 2^10/... ), large values (|v| near 2^14)
    run("pre-sharp", 8, 2, 128, 16, [300, 2000], gen, q_scale=8.0)
    run("pre-bigv", 8, 2, 128, 16, [300, 2000], gen, v_scale=2.0 ** 13)
    run("pre-tinyv", 8, 2, 128, 16, [300, 2000], gen, v_scale=2.0 ** -20)
    print(f"WORST: err/tol {WORST['r']:.3e} (margin {1 / max(WORST['r'], 1e-300):.2f}x) at "
          f"(case, seq, head, e, err, tol, X, T, attnAbs) = {WORST['at']}; largest tolA "
          f"{WORST['tolA_max']:.3e}; {WORST['elems']} elements checked; finiteness and frame "
          f"{'hold' if WORST['ok'] else 'FAIL'}")
    sys.exit(0 if WORST["ok"] and WORST["r"] < 1 else 1)


if __name__ == "__main__":
    main()
