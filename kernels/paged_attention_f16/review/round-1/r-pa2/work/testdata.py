"""The test's inputs (source lines 137-150: its generator, seed 0, the same draws in the same order
as make_cache and the test), rebuilt on the CPU, and the quantities the claim's tolerance depends
on: the largest scale*scoreAbs over all (sequence, head, token), its mean per head_dim, the
largest attnAbs, the pages visited, and the tolerance bound tolO*attnAbs + tolF at those values.
No GPU and no Triton: the kernel's output is not computed here."""
import math

import torch


def cdiv(a, b):
    return (a + b - 1) // b


def make_cache(seq_lens, num_kv_heads, head_dim, page_size, num_pages, generator):
    seq_lens = [int(n) for n in seq_lens]
    pages_per_seq = [cdiv(n, page_size) for n in seq_lens]
    max_pages = max(pages_per_seq)
    assert sum(pages_per_seq) <= num_pages
    perm = torch.randperm(num_pages, generator=generator)
    block_tables = torch.zeros((len(seq_lens), max_pages), dtype=torch.int32)
    used = 0
    for s, n in enumerate(pages_per_seq):
        block_tables[s, :n] = perm[used:used + n].to(torch.int32)
        used += n
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, generator=generator).to(dtype=torch.float16)
    v_cache = torch.randn(shape, generator=generator).to(dtype=torch.float16)
    return k_cache, v_cache, block_tables, torch.tensor(seq_lens, dtype=torch.int32)


gen = torch.Generator().manual_seed(0)
overall_X = 0.0
overall_A = 0.0
for num_heads, num_kv_heads, head_dim, page_size, seq_lens in [
    (32, 8, 128, 16, [1, 15, 16, 17, 1000, 4096, 333]),
    (8, 8, 64, 16, [5, 64, 2049]),
    (16, 2, 128, 32, [31, 32, 33, 8191]),
]:
    num_pages = sum(cdiv(n, page_size) for n in seq_lens) + 50
    k_cache, v_cache, block_tables, lens = make_cache(seq_lens, num_kv_heads, head_dim, page_size,
                                                      num_pages, gen)
    q = torch.randn((len(seq_lens), num_heads, head_dim), generator=gen).to(dtype=torch.float16)
    scale = head_dim ** -0.5
    group = num_heads // num_kv_heads
    maxX = 0.0
    sumX = 0.0
    cntX = 0
    maxA = 0.0
    maxT = 0
    for s in range(len(seq_lens)):
        n = int(lens[s])
        t = torch.arange(n)
        pages = block_tables[s, t // page_size].long()
        k = k_cache[pages, t % page_size].double().repeat_interleave(group, dim=1)  # [n, H, D]
        v = v_cache[pages, t % page_size].double().repeat_interleave(group, dim=1)
        qs = q[s].double()  # [H, D]
        sabs = torch.einsum("hd,nhd->hn", qs.abs(), k.abs()) * scale  # scale * scoreAbs
        sc = torch.einsum("hd,nhd->hn", qs, k) * scale
        w = torch.softmax(sc, dim=-1)
        attn_abs = torch.einsum("hn,nhd->hd", w, v.abs())
        maxX = max(maxX, sabs.max().item())
        sumX += sabs.sum().item()
        cntX += sabs.numel()
        maxA = max(maxA, attn_abs.max().item())
        maxT = max(maxT, cdiv(n, page_size))
    overall_X = max(overall_X, maxX)
    overall_A = max(overall_A, maxA)
    print(f"heads {num_heads}/{num_kv_heads} head_dim {head_dim} page {page_size}: "
          f"max scale*scoreAbs {maxX:.4f}, mean {sumX / cntX:.4f} (0.6366*sqrt(D) = "
          f"{0.6366 * math.sqrt(head_dim):.4f}), max attnAbs {maxA:.4f}, max pages {maxT}")
print(f"overall: max scale*scoreAbs {overall_X:.4f}, max attnAbs {overall_A:.4f}")
print(f"bound at tolO(128,32,256,11)=1.2488e-3 and the largest attnAbs: "
      f"{1.2488e-3 * overall_A + 2**-25:.4e}")
