"""
Ranks of the sampled tokens (vLLM)
==================================

For each row ``r`` of a ``[batch_size, vocab_size]`` logits tensor and the token ``t_r`` sampled
for it, the rank of ``t_r``: the number of vocabulary entries ``j < vocab_size`` whose logit is at
least the token's own, ``logits[r, j] >= logits[r, t_r]``. Ties count, and the token counts
itself, so the rank is ``1`` for a unique maximum (vLLM's ``selected_token_ranks``, 1-based). One
program per row walks the row in blocks of ``BLOCK_SIZE`` entries; the lanes past ``vocab_size``
in the last block hold ``-inf``.

Layouts and dtypes:

* ``logits``: ``[batch_size, vocab_size]``, ``float32``, ``bfloat16`` or ``float16``, last dimension
  contiguous, row stride ``logits.stride(0) >= vocab_size`` (a slice ``logits[:, :vocab_size]`` of
  a padded vocabulary has a larger stride);
* ``sampled_token_ids``: ``[batch_size]``, ``int64``, each in ``[0, vocab_size)``;
* ``token_ranks`` (output, allocated fresh): ``[batch_size]``, ``int64``.

Provenance: vLLM v0.33.0 (commit ``4ba787ec99``), ``vllm/v1/worker/gpu/sample/logprob.py``:

* ``_ranks_kernel``: ``logprob.py:61-81`` (``@triton.jit`` at line 61, ``def`` at 62);
* the host, from ``compute_topk_scores`` (``logprob.py:111-189``): the allocation of
  ``token_ranks`` and the launch (``logprob.py:173-181``).

Changes from the source:

* ``logprob.py:61-81``: the kernel, copied verbatim with its ``@triton.jit``; it has no autotuning
  or heuristics and calls no helper.
* The host ``compute_token_ranks`` is the part of ``compute_topk_scores`` that computes the ranks:
  ``batch_size, vocab_size = logits.shape`` (``logprob.py:122``), the allocation of
  ``token_ranks`` and the launch (``logprob.py:173-181``), with the same grid and arguments. The
  rest of ``compute_topk_scores`` (the top-k selection, the other kernels of the file, the
  ``LogprobsTensors`` it returns) does not touch the ranks and is dropped, with the file's other
  kernels (``_topk_log_softmax_kernel``, ``_fill_logprob_token_ids_kernel``) and their hosts and
  imports. The launch options the source leaves to Triton's defaults are written out
  (``num_warps=4``, ``num_stages=3``, Triton 3.8.0's defaults), which changes nothing.
* ``reference`` and the ``__main__`` test are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_ranks(logits: torch.Tensor, sampled_token_ids: torch.Tensor) -> torch.Tensor:
    """The ranks part of vLLM's `compute_topk_scores` (`logprob.py:111`)."""
    batch_size, vocab_size = logits.shape

    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
        num_warps=4,
        num_stages=3,
    )
    return token_ranks


def reference(logits, sampled_token_ids):
    """Exact, on the CPU, for a token whose logit is not `-inf`: the number of entries at least the
    token's logit (NaN never counts)."""
    lg = logits.float().cpu()
    t = sampled_token_ids.cpu().long()
    x = lg.gather(1, t[:, None])
    return (lg >= x).sum(dim=1).to(torch.int64)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)

    # Batches of 1 to 1024 rows; vocabularies of Qwen3 (151936), Llama-3 (128256), Llama-2 (32000),
    # Gemma-3 (262144); float32 (processed logits) and the model dtypes bfloat16, float16 (raw).
    for batch_size, vocab_size, dtype in [
        (1, 32000, torch.float32),
        (8, 128256, torch.float32),
        (64, 151936, torch.bfloat16),
        (256, 32000, torch.float16),
        (1024, 8192, torch.float32),
        (3, 262144, torch.bfloat16),
    ]:
        logits = torch.randn(batch_size, vocab_size, generator=gen).to(dtype)
        token_ids = torch.randint(0, vocab_size, (batch_size,), generator=gen, dtype=torch.int64)
        got = compute_token_ranks(logits.to(DEVICE), token_ids.to(DEVICE)).cpu()
        ok = torch.equal(got, reference(logits, token_ids))
        print(f"batch {batch_size} vocab {vocab_size} {dtype}: {'exact' if ok else 'MISMATCH'}")
        assert ok
    print("✅ token ranks match the reference exactly")
