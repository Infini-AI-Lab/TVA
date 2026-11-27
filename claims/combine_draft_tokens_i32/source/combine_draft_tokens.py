"""
Combine the sampled and draft tokens into the next step's input (vLLM)
======================================================================

Before a decode step with speculative decoding, each request's slice of the packed ``input_ids``
ends with its tokens to verify: the last token sampled in the previous step (the "bonus" token)
followed by its draft tokens. vLLM keeps both on the GPU (``last_sampled_tokens``,
``draft_tokens``), so this kernel writes them into ``input_ids`` and, in the same pass, computes
``logits_indices``, the rows of the model's hidden states whose logits are sampled.

One program per request ``b`` (its request-state row ``i = idx_mapping[b]``). With
``num_logits = cu_num_logits[b + 1] - cu_num_logits[b]`` (the request's draft tokens plus
``NUM_NEW_SAMPLED_TOKENS``) and ``query_end = query_start_loc[b + 1]``:

* ``logits_indices[cu_num_logits[b] + j] = query_end - num_logits + j`` for ``j < num_logits``:
  the request's last ``num_logits`` query rows;
* if the request is past its prefill (``seq_lens[b] > prefill_len[i]``):
  - with ``NUM_NEW_SAMPLED_TOKENS = 1``, when the first logit's position is past the prefill
    (``seq_lens[b] - num_logits >= prefill_len[i]``), ``input_ids[query_end - num_logits]`` gets
    ``last_sampled_tokens[i]``;
  - ``input_ids[query_end - num_draft_tokens + j] = draft_tokens[i, j]`` for
    ``j < num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS``.

Layouts and dtypes (``vllm/v1/worker/gpu/input_batch.py:23-40``, ``vllm/v1/worker/gpu/states.py:52-80``):

* ``input_ids``: ``[max_num_tokens]``, ``int32``;
* ``idx_mapping``: ``[num_reqs]``, ``int64`` (batch index to request-state row);
* ``last_sampled_tokens``: ``[max_num_reqs, 1]``, ``int64``;
* ``query_start_loc``: ``[num_reqs_padded + 1]``, ``int32``; ``seq_lens``: ``[num_reqs_padded]``,
  ``int32``;
* ``prefill_len``: ``[max_num_reqs]``, ``int32``;
* ``draft_tokens``: ``[max_num_reqs, num_speculative_steps]``, ``int64``, contiguous;
* ``cu_num_logits``: ``[num_reqs + 1]``, ``int32``;
* ``logits_indices`` (output, ``torch.empty``): ``[num_logits]``, ``int64``.

Provenance: vLLM v0.33.0 (commit ``4ba787ec99``), ``vllm/v1/worker/gpu/input_batch.py``:

* ``_combine_sampled_and_draft_tokens_kernel``: ``input_batch.py:414-472``;
* ``combine_sampled_and_draft_tokens``, the host: ``input_batch.py:475-513``;
* its one caller, ``GPUModelRunner.prepare_inputs``: ``vllm/v1/worker/gpu/model_runner.py:1510-1521``.

Changes from the source:

* ``input_batch.py:10``: ``from vllm.triton_utils import tl, triton`` (vLLM's re-export of
  Triton) is written ``import triton`` and ``import triton.language as tl``; the module's other
  imports, classes and kernels are not copied.
* ``input_batch.py:427``: the default of the kernel's last parameter removed,
  ``NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1`` written ``NUM_NEW_SAMPLED_TOKENS: tl.constexpr``
  (the host passes it on every launch). The rest of the kernel's signature and its body are
  unchanged.
* ``input_batch.py:475-513``: the host, verbatim.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Keep prompt-tail slots intact; only rewrite generated-token slots.
    first_logit_seq_pos = seq_len - num_logits
    if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
        # Write the last sampled token ID to input_ids.
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_new_sampled_tokens: int = 1,  # excl accepted draft tokens, a.k.a bonus tokens
) -> torch.Tensor:
    assert num_new_sampled_tokens in (0, 1), (
        f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
    )
    # use idx_mapping.shape[0] for actual request count
    num_reqs = idx_mapping.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    logits_indices = torch.empty(num_logits, dtype=torch.int64, device=input_ids.device)
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        # NOTE(woosuk): Add num_new_sampled_tokens to ensure the block covers the
        # last sampled token in addition to all draft tokens.
        BLOCK_SIZE=triton.next_power_of_2(
            num_speculative_steps + num_new_sampled_tokens
        ),
    )
    return logits_indices
