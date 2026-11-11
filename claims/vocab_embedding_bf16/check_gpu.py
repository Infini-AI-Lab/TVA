"""GPU check of the claim for `vocab_embedding_bf16`.

Runs the kernel's own host (`source/vocab_parallel_embedding.py`) on
the setup's launches (the host's test cases, with the same generator) and on launches at the edges
of `Inst.ok` and `Pre`, and checks `Post` exactly as `Claim.lean` states it, bit for bit:

* element `r * hidden_dim + col` of the output is the weight's storage element
  `(v - org_vocab_start_index) * weight_stride0 + col` for an id `v` in the original range, element
  `(v - added_offset) * weight_stride0 + col` for an id in the added range (the original range
  first), and the `bfloat16` pattern `0x0000` (`+0`) otherwise;
* the frame: every byte outside the output's `n_tokens * hidden_dim` elements is unchanged (the
  inputs; guard regions around the output in direct launches with the host's own launch line).

The weight holds random 16-bit patterns (NaNs with payloads, infinities, subnormals, `-0`
included), so the comparison is on the bits. No tolerance: the claim is exact. Run with
`CUDA_VISIBLE_DEVICES=<gpu> python -I check_gpu.py`.
"""

import importlib.util
import math
import os
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "vocab_parallel_embedding.py")
sys.dont_write_bytecode = True  # leave no __pycache__ in the kernel's folder
spec = importlib.util.spec_from_file_location("vocab_embedding_src", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEV = torch.device("cuda")
I64_MIN, I64_MAX = -2 ** 63, 2 ** 63 - 1


def added_offset(cfg):
    return (cfg["added_vocab_start_index"] - (cfg["org_vocab_end_index"] - cfg["org_vocab_start_index"])
            - cfg["num_org_vocab_padding"])


def expected_bits(ids, storage, stride, hidden, cfg):
    """`Post`'s first clause as `int16` bit patterns, from the host's inputs: the ids, the weight's
    storage (from its first element) and the compile-time ranges."""
    v = ids.reshape(-1).to(torch.int64)
    os_, oe = cfg["org_vocab_start_index"], cfg["org_vocab_end_index"]
    as_, ae = cfg["added_vocab_start_index"], cfg["added_vocab_end_index"]
    org = (v >= os_) & (v < oe)
    add = (v >= as_) & (v < ae) & ~org
    # TokOk (Pre): an id of either range names a row inside the storage
    rows = torch.where(org, v - os_, torch.where(add, v - added_offset(cfg), torch.zeros_like(v)))
    assert bool(((rows * stride + hidden <= storage.numel()) | ~(org | add)).all())
    assert bool(((rows >= 0) | ~(org | add)).all())
    col = torch.arange(hidden, device=v.device, dtype=torch.int64)
    flat = storage.view(torch.int16)
    out = torch.empty((v.numel(), hidden), dtype=torch.int16, device=v.device)
    for a in range(0, v.numel(), 4096):  # in chunks, to stay well under 4 GB of device memory
        idx = rows[a:a + 4096, None] * stride + col[None, :]
        bits = flat[idx.clamp(0, storage.numel() - 1)]
        out[a:a + 4096] = torch.where((org | add)[a:a + 4096, None], bits, torch.zeros_like(bits))
    return out


results = []


def run_host(name, ids, base, hidden, cfg):
    """The host on `weight = base[:, :hidden]` (row stride `base.shape[1]`)."""
    weight = base[:, :hidden]
    ids0, base0 = ids.clone(), base.clone()
    out = mod.vocab_parallel_embedding(ids, weight, **cfg)
    n = ids.numel()
    assert out.shape == (*ids.shape, hidden) and out.dtype == torch.bfloat16
    if n == 0:
        results.append((name, True, True, 0))
        print(f"{name}: empty input, no launch")
        return
    exp = expected_bits(ids, base.reshape(-1), base.shape[1], hidden, cfg)
    ok = torch.equal(out.reshape(n, hidden).view(torch.int16), exp)
    frame = torch.equal(ids, ids0) and torch.equal(base.view(torch.int16), base0.view(torch.int16))
    results.append((name, ok, frame, n * hidden))
    print(f"{name}: {n} ids, hidden {hidden}, stride {base.shape[1]}: output bit-exact {ok} "
          f"({n * hidden} elements), inputs unchanged {frame}")


def run_guarded(name, ids, base, hidden, cfg, guard=4096):
    """The host's launch line (source lines 126-142) on an output carved from a pool of random
    bits with guards around it."""
    weight = base[:, :hidden]
    n = ids.numel()
    g = torch.Generator(device=DEV).manual_seed(3)
    pool = torch.randint(-2 ** 15, 2 ** 15, (guard + n * hidden + guard,), generator=g, device=DEV,
                         dtype=torch.int16)
    pool0 = pool.clone()
    output = pool[guard:guard + n * hidden].view(torch.bfloat16).view(*ids.shape, hidden)
    block_h = min(1024, triton.next_power_of_2(hidden))
    grid = (n, triton.cdiv(hidden, block_h))
    mod._vocab_parallel_embedding_kernel[grid](
        ids, weight, output, hidden, weight.stride(0), cfg["org_vocab_start_index"],
        cfg["org_vocab_end_index"], cfg["num_org_vocab_padding"], cfg["added_vocab_start_index"],
        cfg["added_vocab_end_index"], BLOCK_H=block_h, num_warps=8, num_stages=3)
    exp = expected_bits(ids, base.reshape(-1), base.shape[1], hidden, cfg)
    ok = torch.equal(output.reshape(n, hidden).view(torch.int16), exp)
    frame = torch.equal(pool[:guard], pool0[:guard]) and torch.equal(pool[guard + n * hidden:],
                                                                      pool0[guard + n * hidden:])
    results.append((name, ok, frame, n * hidden))
    print(f"{name} (guarded direct launch): {n} ids, hidden {hidden}: output bit-exact {ok}, "
          f"guards unchanged {frame}")


def random_bits(rows, cols, g):
    return torch.randint(-2 ** 15, 2 ** 15, (rows, cols), generator=g, dtype=torch.int16).view(
        torch.bfloat16).to(DEV)


def boundary_ids(cfg, vocab):
    os_, oe = cfg["org_vocab_start_index"], cfg["org_vocab_end_index"]
    as_, ae = cfg["added_vocab_start_index"], cfg["added_vocab_end_index"]
    vals = [os_ - 1, os_, oe - 1, oe, as_ - 1, as_, ae - 1, ae, -1, 0, vocab, 2 ** 30, 2 ** 31,
            -2 ** 31, I64_MIN, I64_MAX, I64_MIN + 1, I64_MAX - 1]
    return [x for x in vals if I64_MIN <= x <= I64_MAX]


if __name__ == "__main__":
    # The setup: the host's test cases (source lines 197-226), same generator and ids.
    gen = torch.Generator().manual_seed(0)
    cases = [
        (128256, 128256, 4096, 8, 3, (8192,), torch.int64),
        (129280, 129280, 7168, 8, 7, (1,), torch.int64),
        (151936, 151936, 2048, 4, 0, (3, 77), torch.int64),
        (152064, 152064, 5120, 2, 1, (513,), torch.int32),
        (32000 + 256, 32000, 257, 2, 1, (1000,), torch.int64),
        (50257 + 3, 50257, 6144, 8, 7, (300,), torch.int64),
        (1000, 1000, 7, 2, 0, (64,), torch.int32),
        (1000, 1000, 128, 4, 2, (0,), torch.int64),
    ]
    for vocab, org_vocab, hidden, tp, rank, shape, idt in cases:
        cfg, rows = mod.shard_indices(vocab, org_vocab, rank, tp)
        weight = torch.randn((rows, hidden), generator=gen).to(device=DEV, dtype=torch.bfloat16)
        ids = torch.randint(0, vocab, shape, generator=gen)
        flat = ids.view(-1)
        if flat.numel() >= 4:
            flat[:4] = torch.tensor([-1, cfg["org_vocab_start_index"], vocab, 2 ** 30])
        if flat.numel() >= 6 and idt == torch.int64:
            flat[4:6] = torch.tensor([I64_MIN, I64_MAX])
        elif flat.numel() > 0:
            flat[0] = cfg["org_vocab_end_index"] - 1
        ids = ids.to(device=DEV, dtype=idt)
        run_host(f"setup vocab {vocab} tp {tp} rank {rank} hidden {hidden} {str(idt).split('.')[-1]}",
                 ids, weight, hidden, cfg)

    # Edges: random bit patterns in the weight, every boundary id, both id dtypes, every rank,
    # a column slice (row stride wider than hidden_dim), added vocabularies with padding rows.
    g = torch.Generator().manual_seed(1)
    for vocab, org_vocab, hidden, tp, stride_extra, n, idt in [
        (32000 + 256, 32000, 257, 2, 0, 4096, torch.int64),
        (50257 + 3, 50257, 6144, 8, 0, 512, torch.int64),
        (1000 + 24, 1000, 7, 1, 9, 300, torch.int32),
        (129280, 129280, 7168, 8, 0, 256, torch.int64),
        (151936, 151936, 1000, 4, 24, 2048, torch.int32),
        (128256 + 512, 128256, 8192, 2, 0, 1024, torch.int64),
        (4096, 4096, 1025, 4, 3, 777, torch.int64),
        (64, 64, 1, 1, 0, 100, torch.int64),
    ]:
        for rank in range(tp):
            cfg, rows = mod.shard_indices(vocab, org_vocab, rank, tp)
            base = random_bits(rows, hidden + stride_extra, g)
            ids = torch.randint(-50, vocab + 50, (n,), generator=g, dtype=torch.int64)
            b = boundary_ids(cfg, vocab)
            if idt == torch.int32:
                b = [x for x in b if -2 ** 31 <= x < 2 ** 31]
            ids[:len(b)] = torch.tensor(b, dtype=torch.int64)
            run_host(f"edge vocab {vocab} (+{vocab - org_vocab}) tp {tp} rank {rank} hidden {hidden} "
                     f"stride {hidden + stride_extra} {str(idt).split('.')[-1]}", ids.to(device=DEV, dtype=idt),
                     base, hidden, cfg)
    # Many tokens at a large hidden size: 32768 ids, hidden 8192 (a 512 MiB output).
    cfg, rows = mod.shard_indices(151936, 151936, 1, 8)
    base = random_bits(rows, 8192, g)
    ids = torch.randint(-1000, 151936 + 1000, (32768,), generator=g, dtype=torch.int64).to(DEV)
    run_host("edge 32768 ids, hidden 8192", ids, base, 8192, cfg)
    del base, ids
    # Overlapping ranges (not the layer's, allowed by Inst.ok): the original range wins.
    cfg = dict(org_vocab_start_index=100, org_vocab_end_index=300, num_org_vocab_padding=5,
               added_vocab_start_index=250, added_vocab_end_index=320)
    base = random_bits(400, 64, g)
    ids = torch.arange(80, 340, dtype=torch.int64).to(DEV)
    run_host("edge overlapping ranges", ids, base, 64, cfg)
    # Frame with arbitrary initial output contents.
    cfg, rows = mod.shard_indices(32000 + 256, 32000, 1, 2)
    base = random_bits(rows, 300, g)
    ids = torch.randint(-50, 32300, (999,), generator=g, dtype=torch.int64).to(DEV)
    run_guarded("frame hidden 257", ids, base, 257, cfg)
    cfg, rows = mod.shard_indices(1000, 1000, 0, 2)
    base = random_bits(rows, 7, g)
    ids = torch.randint(-5, 1005, (64,), generator=g, dtype=torch.int64).to(DEV, torch.int32)
    run_guarded("frame hidden 7 int32", ids, base, 7, cfg)

    bad = [r for r in results if not (r[1] and r[2])]
    print(f"{len(results)} host calls, {sum(r[3] for r in results)} output elements checked; "
          f"worst error / tolerance: 0 / 0 (bit-exact) on every element; frame held on every launch"
          if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)
