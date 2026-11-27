# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's `_topk_log_softmax_kernel` and its host `compute_token_logprobs`.

From vLLM (commit `4ba787ec99`), `vllm/v1/worker/gpu/sample/logprob.py`. Changes from that file:

* the imports: `from vllm.triton_utils import tl, triton` is `import triton` and
  `import triton.language as tl` (vLLM's `triton_utils` re-exports those two modules); the imports
  of `numpy`, `MAX_LOGPROB_TOKEN_IDS`, `SamplingParams`, `LogprobsTensors`, `StagedWriteTensor`
  and `UvaBackedTensor`, used only by the code removed below, are dropped;
* the other kernels of the file (`_ranks_kernel`, `_fill_logprob_token_ids_kernel`) and the
  class `LogprobTokenIdsState` are removed;
* `compute_topk_scores` keeps only what decides the launch of `_topk_log_softmax_kernel` on its
  fast path (no request asked for custom `logprob_token_ids`, `logits_mode=False`): the token-id
  matrix `[sampled, top-k ids]` and the call of `compute_token_logprobs`. It returns the matrix
  and the scores; the ranks (`_ranks_kernel`) and the `LogprobsTensors` wrapper are removed.

`_MAX_TOPK_BLOCK`, `_topk_log_softmax_kernel` and `compute_token_logprobs` are unchanged.
"""

import torch

import triton
import triton.language as tl

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk_ids_stride,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk_ids_stride + k_offset, mask=k_mask, other=0
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        token_ids.stride(0),
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        TOPK_BLOCK_SIZE=topk_block_size,
    )
    return logprobs


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape

    # Fast path: no request asked for custom logprob_token_ids.
    logprob_token_ids = sampled_token_ids.unsqueeze(-1)
    if num_logprobs > 0:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
    scores = compute_token_logprobs(logits, logprob_token_ids)
    return logprob_token_ids, scores
