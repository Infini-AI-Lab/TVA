"""
Vocabulary-parallel embedding lookup (one tensor-parallel rank's shard)
=======================================================================

With tensor parallelism, each rank holds a contiguous shard of the embedding table: the rows of
the original vocabulary ``[org_vocab_start_index, org_vocab_end_index)``, then
``num_org_vocab_padding`` padding rows, then the rows of the added (LoRA) vocabulary
``[added_vocab_start_index, added_vocab_end_index)``. For each token id, the kernel copies the
shard's row for that id, or writes a row of zeros when the id is outside both ranges (another
rank holds it; the all-reduce that follows sums the ranks' outputs). One program per (token,
block of ``BLOCK_H`` columns). The ranges are compile-time values.

Layouts:

* ``input_``: any shape, contiguous, ``int64`` (or ``int32``) token ids;
* ``weight``: ``[num_embeddings_per_partition, hidden_dim]``, ``bfloat16``, rows of stride
  ``weight.stride(0)`` and unit column stride;
* ``output``: ``[*input_.shape, hidden_dim]``, contiguous, the weight's dtype.

A token id ``t`` in the original range reads local row ``t - org_vocab_start_index``; one in the
added range reads local row ``t - added_offset``, where ``added_offset = added_vocab_start_index
- (org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding``.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/embeddings/vocab_parallel_embedding.py``:

* ``_vocab_parallel_embedding_kernel``: line 9;
* the host ``vocab_parallel_embedding``: line 56.

The test's ``shard_indices`` follows how the layer computes a rank's ranges,
``python/sglang/srt/layers/vocab_parallel_embedding.py``: ``pad_vocab_size`` (line 57),
``vocab_range_from_global_vocab_size`` (line 70), the padding in
``VocabParallelEmbedding.__init__`` (lines 273-298) and ``_get_indices`` (line 353).

Changes from the source:

* ``_vocab_parallel_embedding_kernel`` (``vocab_parallel_embedding.py:9``) is copied verbatim,
  with its ``@triton.jit`` (the source has no autotuning or heuristics on it).
* The host ``vocab_parallel_embedding`` (``vocab_parallel_embedding.py:56``) is copied verbatim
  except for the launch option ``num_stages=3``, added: the
  wrapper passes none, so this is Triton 3.8.0's default, which its launches get. It already
  allocates its output fresh. ``BLOCK_H = min(1024, next_power_of_2(hidden_dim))`` and
  ``num_warps=8`` are the wrapper's.
* ``shard_indices`` (a test helper, adapted from the layer's code cited above), ``reference``
  and the ``__main__`` test are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _vocab_parallel_embedding_kernel(
    input_ptr,
    weight_ptr,
    out_ptr,
    # The scalar params are tl.constexpr on purpose: it lets the compiler fold
    # the vocab-window comparisons into a single range check (measured ~5%
    # faster at large token counts), and each embedding layer has one fixed
    # (hidden_dim, stride, shard-window) tuple, so the specialization costs one
    # compile per layer at warmup.
    hidden_dim: tl.constexpr,
    weight_stride0: tl.constexpr,
    org_vocab_start_index: tl.constexpr,
    org_vocab_end_index: tl.constexpr,
    num_org_vocab_padding: tl.constexpr,
    added_vocab_start_index: tl.constexpr,
    added_vocab_end_index: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < hidden_dim

    token = tl.load(input_ptr + row).to(tl.int64)
    org_vocab_mask = (token >= org_vocab_start_index) & (token < org_vocab_end_index)
    added_vocab_mask = (token >= added_vocab_start_index) & (
        token < added_vocab_end_index
    )
    valid = org_vocab_mask | added_vocab_mask

    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    local_id = tl.where(
        org_vocab_mask, token - org_vocab_start_index, token - added_offset
    )
    local_id = tl.where(valid, local_id, 0)
    vals = tl.load(
        weight_ptr + local_id * weight_stride0 + cols,
        mask=col_mask & valid,
        other=0.0,
    )
    tl.store(out_ptr + row * hidden_dim + cols, vals, mask=col_mask)


def vocab_parallel_embedding(
    input_: torch.Tensor,
    weight: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> torch.Tensor:
    assert input_.is_cuda
    assert input_.is_contiguous()
    assert input_.dtype in (torch.int32, torch.int64)
    assert weight.is_cuda
    assert weight.ndim == 2
    assert weight.stride(1) == 1

    hidden_dim = weight.shape[1]
    output = torch.empty(
        (*input_.shape, hidden_dim), dtype=weight.dtype, device=weight.device
    )
    n_tokens = input_.numel()
    if n_tokens == 0:
        return output

    block_h = min(1024, triton.next_power_of_2(hidden_dim))
    grid = (n_tokens, triton.cdiv(hidden_dim, block_h))
    _vocab_parallel_embedding_kernel[grid](
        input_,
        weight,
        output,
        hidden_dim,
        weight.stride(0),
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
        BLOCK_H=block_h,
        num_warps=8,
        num_stages=3,
    )
    return output


def shard_indices(num_embeddings, org_num_embeddings, tp_rank, tp_size, padding_size=64):
    """A rank's vocabulary ranges, as ``VocabParallelEmbedding`` computes them, and its number of
    weight rows (``num_embeddings_per_partition``)."""

    def pad_vocab_size(vocab_size, pad_to):
        return ((vocab_size + pad_to - 1) // pad_to) * pad_to

    def vocab_range(global_vocab_size, rank, world_size, offset=0):
        assert global_vocab_size % world_size == 0
        per_partition = global_vocab_size // world_size
        return rank * per_partition + offset, (rank + 1) * per_partition + offset

    if pad_vocab_size(org_num_embeddings, padding_size) % tp_size != 0:
        padding_size *= tp_size
    num_added = num_embeddings - org_num_embeddings
    org_padded = pad_vocab_size(org_num_embeddings, padding_size)
    padded = pad_vocab_size(org_padded + num_added, padding_size)
    p_org_start, p_org_end = vocab_range(org_padded, tp_rank, tp_size)
    p_add_start, p_add_end = vocab_range(padded - org_padded, tp_rank, tp_size, offset=org_num_embeddings)
    org_start = min(p_org_start, org_num_embeddings)
    org_end = min(p_org_end, org_num_embeddings)
    add_start = min(p_add_start, num_embeddings)
    add_end = min(p_add_end, num_embeddings)
    cfg = dict(
        org_vocab_start_index=org_start,
        org_vocab_end_index=org_end,
        num_org_vocab_padding=(p_org_end - p_org_start) - (org_end - org_start),
        added_vocab_start_index=add_start,
        added_vocab_end_index=add_end,
    )
    return cfg, padded // tp_size


def reference(input_, weight, cfg):
    """The shard's rows by index, zero rows for ids outside the shard."""
    t = input_.long()
    org = (t >= cfg["org_vocab_start_index"]) & (t < cfg["org_vocab_end_index"])
    added = (t >= cfg["added_vocab_start_index"]) & (t < cfg["added_vocab_end_index"])
    added_offset = (cfg["added_vocab_start_index"]
                    - (cfg["org_vocab_end_index"] - cfg["org_vocab_start_index"])
                    - cfg["num_org_vocab_padding"])
    local = torch.where(org, t - cfg["org_vocab_start_index"], t - added_offset)
    valid = org | added
    local = torch.where(valid, local, 0)
    out = weight[local.reshape(-1)].reshape(*t.shape, weight.shape[1]).clone()
    out[~valid] = 0
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (vocab incl. added, original vocab, hidden, tp_size, tp_rank, input shape, input dtype)
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
        cfg, rows = shard_indices(vocab, org_vocab, rank, tp)
        weight = torch.randn((rows, hidden), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        ids = torch.randint(0, vocab, shape, generator=gen)
        flat = ids.view(-1)
        if flat.numel() >= 4:  # ids another rank holds, and ids no rank holds
            flat[:4] = torch.tensor([-1, cfg["org_vocab_start_index"], vocab, 2**30])
        if flat.numel() >= 6 and idt == torch.int64:
            flat[4:6] = torch.tensor([torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max])
        elif flat.numel() > 0:
            flat[0] = cfg["org_vocab_end_index"] - 1
        ids = ids.to(device=DEVICE, dtype=idt)
        out = vocab_parallel_embedding(ids, weight, **cfg)
        ref = reference(ids, weight, cfg)
        in_shard = int(((out != 0).any(dim=-1)).sum()) if out.numel() else 0
        assert out.shape == ref.shape and torch.equal(out, ref)
        print(f"vocab {vocab} (original {org_vocab}) hidden {hidden} tp {tp} rank {rank} ids {tuple(shape)} "
              f"{str(idt).split('.')[-1]}: shard rows {rows}, ids in the shard {in_shard}, exact")
    print("vocab_parallel_embedding matches the reference exactly")
