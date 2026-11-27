"""GPU check of the claim for `token_bitmask_f32`.

Runs the kernel's own host (`source/bitmask_ops.py`) on the setup's launches and direct launches
of the kernel with the host's launch line at the edges of `Inst.ok` and `Pre`, and checks `Post`
exactly as `Claim.lean` states it, bit for bit:

* for each processed row `r` (`indices[k]` with indices, `k` without) and token `c < vocab_size`
  whose bit (bit `c % 32` of the bitmask's element `r * bitmask_strides + c // 32`) is `0`, the
  logit `r * logits_strides + c` is the `float32` pattern `0xff800000` (`-inf`);
* the frame: every other byte is unchanged (the other logits, the padding past `vocab_size`, the
  rows not processed, the bitmask, the indices, and guard regions around the logits in direct
  launches).

The logits hold random 32-bit patterns (NaNs with payloads, infinities, `-0` included) and the
bitmasks random words (the sign bit included), so the comparison is on the bits. No tolerance:
the claim is exact. Negative tests: plausible wrong outputs (the bits read most significant
first, the mask's polarity inverted, the next word's bits, tokens past `vocab_size` written,
`+inf` stored, the wrong row's bits) must fail `Post`. Each case keeps its device memory under
4 GB: the setup's largest launches (8192 rows at vocabularies 129280 and 262144) are cut to 2048
and 1024 rows. Run with `CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import os
import sys
import warnings

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "bitmask_ops.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("token_bitmask_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
# the host's `torch.tensor(indices)` of a tensor warns; the copy is what the host does
warnings.filterwarnings("ignore", message="To copy construct from a tensor")
NEG_INF_BITS = -8388608  # 0xff800000 as int32
POS_INF_BITS = 0x7F800000
results = []
neg_results = []


def expected_bits(init_bits, bitmask, rows, vocab, ls, bs, variant=None):
    """`Post` as `int32` bit patterns of the flat logits storage, from the initial storage, the
    bitmask's storage, the processed rows and the strides. `variant` builds a plausible wrong
    output instead (the negative tests)."""
    out = init_bits.clone()
    words = bitmask.reshape(-1).to(torch.int64)
    c = torch.arange(vocab, device=DEV, dtype=torch.int64)
    shift = c % 32
    for a in range(0, len(rows), 256):  # in chunks, to stay well under 4 GB of device memory
        r = torch.tensor(rows[a:a + 256], device=DEV, dtype=torch.int64)
        wi = r[:, None] * bs + c[None, :] // 32
        if variant == "next_word":
            wi = (wi + 1).clamp(max=words.numel() - 1)
        if variant == "wrong_row":
            wi = ((r[:, None] + 1) % max(1, words.numel() // max(bs, 1))) * bs + c[None, :] // 32
        w = words[wi]
        sh = (31 - shift) if variant == "msb_first" else shift
        bit = (w >> sh[None, :]) & 1
        masked = (bit == 1) if variant == "polarity" else (bit == 0)
        idx = r[:, None] * ls + c[None, :]
        val = POS_INF_BITS if variant == "pos_inf" else NEG_INF_BITS
        cur = out[idx]
        out[idx] = torch.where(masked, torch.full_like(cur, val), cur)
        if variant == "past_vocab" and ls > vocab:
            pad = r[:, None] * ls + torch.arange(vocab, ls, device=DEV, dtype=torch.int64)[None, :]
            out[pad] = NEG_INF_BITS
    return out


def random_bits(n, g):
    return torch.randint(-2 ** 31, 2 ** 31, (n,), generator=g, dtype=torch.int32).to(DEV)


def check(name, fin_bits, init_bits, bitmask, bm0, rows, vocab, ls, bs, extra_frame=True,
          negatives=False):
    exp = expected_bits(init_bits, bitmask, rows, vocab, ls, bs)
    ok = torch.equal(fin_bits, exp)
    frame = torch.equal(bitmask, bm0) and extra_frame
    n_neg = 0
    if negatives:
        for v in ["msb_first", "polarity", "next_word", "past_vocab", "pos_inf", "wrong_row"]:
            if (v == "past_vocab" and ls <= vocab) or (v == "wrong_row" and bitmask.numel() <= bs):
                continue
            wrong = expected_bits(init_bits, bitmask, rows, vocab, ls, bs, variant=v)
            fails = not torch.equal(wrong, exp)
            neg_results.append((name, v, fails))
            n_neg += 1
    results.append((name, ok, frame, len(rows) * vocab))
    print(f"{name}: {len(rows)} rows, vocab {vocab}, strides ({ls}, {bs}): logits bit-exact {ok}, "
          f"bitmask and frame unchanged {frame}" + (f", {n_neg} wrong outputs tried" if n_neg else ""))


def run_host(name, bsz, width, bw, g, kind="random", indices=None, ndim=2, negatives=False):
    """The host on `[bsz, width]` logits of random bits and a `[bsz, bw]` bitmask."""
    init = random_bits(bsz * width, g)
    logits = init.clone().view(torch.float32).view(bsz, width)
    if kind == "random":
        bitmask = random_bits(bsz * bw, g).view(bsz, bw)
    elif kind == "zeros":
        bitmask = torch.zeros((bsz, bw), dtype=torch.int32, device=DEV)
    elif kind == "ones":
        bitmask = torch.full((bsz, bw), -1, dtype=torch.int32, device=DEV)
    else:  # sparse: a few allowed tokens per row, as grammars give
        bitmask = torch.zeros((bsz, bw), dtype=torch.int32, device=DEV)
        hot = torch.randint(0, bw * 32, (bsz, 8), generator=g).to(DEV)
        for j in range(8):
            w = hot[:, j] // 32
            bitmask[torch.arange(bsz, device=DEV), w] |= (1 << (hot[:, j] % 32)).to(torch.int32)
    bm0 = bitmask.clone()
    lg, bm = logits, bitmask
    if ndim == 1:
        lg, bm = logits.view(width), bitmask.view(bw)
    idx0 = None if indices is None else (indices.clone() if torch.is_tensor(indices) else list(indices))
    mod.apply_token_bitmask_inplace_triton(lg, bm, indices)
    vocab = min(width, bw * 32)
    rows = list(range(bsz)) if indices is None else [int(x) for x in indices]
    same_idx = idx0 is None or (torch.equal(indices, idx0) if torch.is_tensor(indices) else indices == idx0)
    check(name, logits.reshape(-1).view(torch.int32), init, bitmask, bm0, rows, vocab, width, bw,
          extra_frame=same_idx, negatives=negatives)


def run_direct(name, bsz, width, bw, vocab, g, num_sms, grid, guard=4096, indices=None):
    """The host's launch line (source lines 172-184), with any `NUM_SMS` and grid, on logits
    carved from a pool of random bits with guards around them."""
    n = bsz * width
    pool = random_bits(guard + n + guard, g)
    init = pool[guard:guard + n].clone()
    pool0 = pool.clone()
    logits = pool[guard:guard + n].view(torch.float32).view(bsz, width)
    bitmask = random_bits(bsz * bw, g).view(bsz, bw)
    bm0 = bitmask.clone()
    num_rows = bsz if indices is None else indices.numel()
    mod.apply_token_bitmask_inplace_kernel[(grid,)](
        logits, bitmask, indices, num_rows, vocab, width, bw, num_sms, 4096,
        num_warps=4096 // 32 // (16 // logits.element_size()), num_stages=3)
    rows = list(range(num_rows)) if indices is None else indices.tolist()
    guards = torch.equal(pool[:guard], pool0[:guard]) and torch.equal(pool[guard + n:], pool0[guard + n:])
    check(name + f" (direct, NUM_SMS {num_sms}, grid {grid})", pool[guard:guard + n], init, bitmask,
          bm0, rows, vocab, width, bw, extra_frame=guards)


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    sms = mod.get_device_core_count()
    print(f"NUM_SMS = {sms}")
    # The setup (SETUP.md): SGLang's calls, logits [bs, vocab], bitmask [bs, cdiv(vocab, 32)].
    for bsz, vocab in [(1, 32000), (64, 128256), (256, 129280), (1024, 151936), (37, 152064),
                       (16, 202048), (8, 262144), (2048, 129280), (1024, 262144)]:
        run_host(f"setup bs {bsz} vocab {vocab}", bsz, vocab, (vocab + 31) // 32, g,
                 negatives=(bsz <= 64))
    run_host("setup sparse grammar mask bs 512 vocab 151936", 512, 151936, 4748, g, kind="sparse")
    run_host("setup padded width 32064, vocab 32000", 4, 32064, 1000, g, negatives=True)
    run_host("setup indices [5, 0, 3] of 8, vocab 32000", 8, 32000, 1000, g, indices=[5, 0, 3],
             negatives=True)
    # Edges of Inst.ok and Pre through the host.
    run_host("edge every token masked", 17, 50257, 1571, g, kind="zeros")
    run_host("edge no token masked", 17, 50257, 1571, g, kind="ones")
    run_host("edge 1-D logits and bitmask", 1, 4097, 129, g, ndim=1, negatives=True)
    run_host("edge vocab 1", 3, 1, 1, g, kind="zeros")
    run_host("edge vocab 31 (last word partly used)", 5, 31, 1, g, kind="zeros")
    run_host("edge vocab 4096", 9, 4096, 128, g)
    run_host("edge vocab 4097, bits past vocab 0", 9, 4097, 129, g, kind="zeros")
    run_host("edge bitmask narrower than the row (vocab 3200 of width 4100)", 6, 4100, 100, g,
             kind="zeros")
    run_host("edge bitmask width 0 (no token)", 4, 100, 0, g, kind="zeros")
    run_host("edge batch 0", 0, 32000, 1000, g)
    run_host("edge 1000 rows of vocab 1 (items = 1000 > NUM_SMS)", 1000, 1, 1, g)
    run_host("edge 8192 rows of vocab 64", 8192, 64, 2, g)
    run_host("edge indices tensor, all rows reversed", 300, 5000, 157, g,
             indices=torch.arange(299, -1, -1, dtype=torch.int32, device=DEV))
    run_host("edge indices empty list", 4, 1000, 32, g, indices=[])
    run_host("edge indices one row", 4, 1000, 32, g, indices=[3])
    # Direct launches: other NUM_SMS and grids within Inst.ok, guards around the logits.
    run_direct("edge NUM_SMS 1", 5, 9000, 282, 9000, g, 1, 1)
    run_direct("edge NUM_SMS 7, grid 5 (= items)", 5, 4096, 128, 4096, g, 7, 5)
    run_direct("edge NUM_SMS 200", 64, 33000, 1032, 33000, g, 200, 200)
    run_direct("edge no-SM branch: grid = items 20, NUM_SMS 32", 5, 16384, 512, 16384, g, 32, 20)
    run_direct("edge vocab below the bitmask and the stride", 7, 5000, 157, 4999, g, sms, sms)
    run_direct("edge indices", 6, 3000, 94, 3000, g, sms, sms,
               indices=torch.tensor([4, 1], dtype=torch.int32, device=DEV))

    bad = [r for r in results if not (r[1] and r[2])]
    neg_bad = [r for r in neg_results if not r[2]]
    print(f"{len(results)} launches, {sum(r[3] for r in results)} logits checked; worst error / "
          f"tolerance: 0 / 0 (bit-exact) on every element; frame held on every launch"
          if not bad else f"FAILED: {bad}")
    print(f"{len(neg_results) - len(neg_bad)} of {len(neg_results)} wrong outputs fail Post"
          + ("" if not neg_bad else f"; NOT rejected: {neg_bad}"))
    sys.exit(1 if bad or neg_bad else 0)
