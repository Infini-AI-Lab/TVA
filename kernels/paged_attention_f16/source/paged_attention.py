"""
Paged attention (decode)
========================

One query token per sequence attends to that sequence's cached keys and values. The cache is
paged: it is a pool of fixed-size pages, and each sequence's block table lists, in order, the
pages that hold its tokens. Token ``t`` of sequence ``s`` is in page ``block_tables[s, t // PAGE]``,
at row ``t % PAGE``. So the kernel reads its keys and values through indices it loads from memory
(indirect memory access).

Layouts (all contiguous in the last dimension):

* ``q``: ``[num_seqs, num_heads, head_dim]``, ``float16``;
* ``k_cache``, ``v_cache``: ``[num_pages, PAGE, num_kv_heads, head_dim]``, ``float16``;
* ``block_tables``: ``[num_seqs, max_pages_per_seq]``, ``int32``;
* ``seq_lens``: ``[num_seqs]``, ``int32``;
* ``out``: ``[num_seqs, num_heads, head_dim]``, ``float16``.

Query head ``h`` reads key/value head ``h // (num_heads // num_kv_heads)`` (grouped-query
attention). Each program handles one (sequence, query head) pair and runs an online softmax over
the sequence's pages, accumulating in ``float32``.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def paged_attention_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr, block_tables_ptr, seq_lens_ptr,  #
    scale,  #
    stride_qs, stride_qh,  #
    stride_kp, stride_kt, stride_kh,  #
    stride_bs,  #
    stride_os, stride_oh,  #
    QUERIES_PER_KV: tl.constexpr,  #
    PAGE_SIZE: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // QUERIES_PER_KV
    seq_len = tl.load(seq_lens_ptr + seq)

    offs_t = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + seq * stride_qs + head * stride_qh + offs_d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for j in range(0, tl.cdiv(seq_len, PAGE_SIZE)):
        # The page that holds tokens j * PAGE_SIZE ... j * PAGE_SIZE + PAGE_SIZE - 1. The cache can
        # hold more than 2^31 elements, so the offset of a page is computed in int64.
        page = tl.load(block_tables_ptr + seq * stride_bs + j).to(tl.int64)
        mask = j * PAGE_SIZE + offs_t < seq_len
        kv_offs = page * stride_kp + offs_t[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :]
        k = tl.load(k_cache_ptr + kv_offs, mask=mask[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * scale
        s = tl.where(mask, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(v_cache_ptr + kv_offs, mask=mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + seq * stride_os + head * stride_oh + offs_d, out.to(tl.float16))


def paged_attention(q, k_cache, v_cache, block_tables, seq_lens):
    num_seqs, num_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, head_dim_k = k_cache.shape
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.float16
    assert v_cache.shape == k_cache.shape and v_cache.stride() == k_cache.stride()
    assert head_dim_k == head_dim and num_heads % num_kv_heads == 0
    assert block_tables.dtype == seq_lens.dtype == torch.int32
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and block_tables.stride(-1) == 1
    out = torch.empty_like(q)
    grid = (num_seqs, num_heads)
    paged_attention_kernel[grid](
        out, q, k_cache, v_cache, block_tables, seq_lens,  #
        head_dim**-0.5,  #
        q.stride(0), q.stride(1),  #
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),  #
        block_tables.stride(0),  #
        out.stride(0), out.stride(1),  #
        QUERIES_PER_KV=num_heads // num_kv_heads,  #
        PAGE_SIZE=page_size,  #
        HEAD_DIM=head_dim,  #
        num_warps=4,  #
    )
    return out


def make_cache(seq_lens, num_kv_heads, head_dim, page_size, num_pages, device, generator):
    """A random cache, and block tables that give each sequence distinct random pages."""
    seq_lens = [int(n) for n in seq_lens]
    pages_per_seq = [triton.cdiv(n, page_size) for n in seq_lens]
    max_pages = max(pages_per_seq)
    assert sum(pages_per_seq) <= num_pages
    perm = torch.randperm(num_pages, generator=generator)
    block_tables = torch.zeros((len(seq_lens), max_pages), dtype=torch.int32)
    used = 0
    for s, n in enumerate(pages_per_seq):
        block_tables[s, :n] = perm[used:used + n].to(torch.int32)
        used += n
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, generator=generator).to(device=device, dtype=torch.float16)
    v_cache = torch.randn(shape, generator=generator).to(device=device, dtype=torch.float16)
    return k_cache, v_cache, block_tables.to(device), torch.tensor(seq_lens, dtype=torch.int32, device=device)


def reference(q, k_cache, v_cache, block_tables, seq_lens):
    """Attention computed in float64 from the gathered keys and values."""
    num_seqs, num_heads, head_dim = q.shape
    page_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = num_heads // num_kv_heads
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    for s in range(num_seqs):
        n = int(seq_lens[s])
        t = torch.arange(n, device=q.device)
        pages = block_tables[s, t // page_size].long()
        k = k_cache[pages, t % page_size].double()  # [n, num_kv_heads, head_dim]
        v = v_cache[pages, t % page_size].double()
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[s].double(), k) * head_dim**-0.5
        out[s] = torch.einsum("hn,nhd->hd", torch.softmax(scores, dim=-1), v)
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    for num_heads, num_kv_heads, head_dim, page_size, seq_lens in [
        (32, 8, 128, 16, [1, 15, 16, 17, 1000, 4096, 333]),
        (8, 8, 64, 16, [5, 64, 2049]),
        (16, 2, 128, 32, [31, 32, 33, 8191]),
    ]:
        num_pages = sum(triton.cdiv(n, page_size) for n in seq_lens) + 50
        k_cache, v_cache, block_tables, lens = make_cache(seq_lens, num_kv_heads, head_dim, page_size, num_pages,
                                                          DEVICE, gen)
        q = torch.randn((len(seq_lens), num_heads, head_dim), generator=gen).to(device=DEVICE, dtype=torch.float16)
        out = paged_attention(q, k_cache, v_cache, block_tables, lens)
        ref = reference(q, k_cache, v_cache, block_tables, lens)
        err = (out.double() - ref).abs().max().item()
        print(f"heads {num_heads}/{num_kv_heads} head_dim {head_dim} page {page_size} seq_lens {seq_lens}: "
              f"max abs error {err:.2e}")
        assert err < 2e-3, err
    print("✅ paged attention matches the float64 reference")
