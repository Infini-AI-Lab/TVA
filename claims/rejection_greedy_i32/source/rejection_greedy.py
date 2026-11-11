"""
Greedy rejection sampling for speculative decoding (vLLM)
=========================================================

A batch of requests, request ``r`` with ``n_r`` draft tokens (``n_r`` may be ``0``), packed one after
another: request ``r``'s drafts are ``draft_token_ids[start_r : end_r]`` with
``end_r = cu_num_draft_tokens[r]`` and ``start_r = cu_num_draft_tokens[r - 1]`` (``0`` for ``r = 0``).
The target model's argmax at the same positions is ``target_argmax[start_r : end_r]``. One program
per request walks its drafts in order: at position ``pos`` it writes the target's token, and stops
after the first position whose draft differs from the target's token (that position's target token
is still written). If every draft matched, it also writes the request's bonus token at position
``n_r``. Positions it does not write keep the placeholder ``-1`` the host filled in.

Requests that are not greedy (``is_greedy[r]`` false) are left to the random sampler: the program
returns at once. vLLM's all-greedy batches pass ``is_greedy = None`` (every request greedy).

Layouts and dtypes:

* ``draft_token_ids``: ``[num_tokens]``, ``int32``, ``num_tokens = sum_r n_r``;
* ``cu_num_draft_tokens``: ``[batch_size]``, ``int32``, the inclusive prefix sums of the ``n_r``;
* ``target_argmax``: ``[num_tokens]``, ``int64`` (``target_logits.argmax(dim=-1)``);
* ``bonus_token_ids``: ``[batch_size, 1]``, ``int32``, contiguous;
* ``is_greedy`` (optional): ``[batch_size]``, ``bool``;
* ``output_token_ids`` (output, allocated fresh and filled with ``-1``):
  ``[batch_size, max_spec_len + 1]``, ``int32``, contiguous; ``max_spec_len = max_r n_r``.

Provenance: vLLM v0.33.0 (commit ``4ba787ec99``), ``vllm/v1/sample/rejection_sampler.py``:

* ``rejection_greedy_sample_kernel``: ``rejection_sampler.py:791-847`` (comment at 791,
  ``@triton.jit(do_not_specialize=["max_spec_len"])`` at 792, ``def`` at 793);
* ``PLACEHOLDER_TOKEN_ID``: ``rejection_sampler.py:43``;
* the host, from ``rejection_sample`` (``rejection_sampler.py:432-536``) and
  ``_rejection_greedy_sample`` (``rejection_sampler.py:874-891``).

Changes from the source:

* ``rejection_sampler.py:791-847``: the kernel, copied verbatim with its decorator
  ``@triton.jit(do_not_specialize=["max_spec_len"])``; it has no autotuning or heuristics and calls
  no helper. Both ``SYNTHETIC_MODE`` paths are kept.
* ``rejection_sampler.py:43-44``: the module constants ``PLACEHOLDER_TOKEN_ID: tl.constexpr = -1``
  and ``GREEDY_TEMPERATURE: tl.constexpr = 0`` are not module-level here (the kernel reads neither):
  ``PLACEHOLDER_TOKEN_ID = -1`` is a local of the host, which is its only user; ``GREEDY_TEMPERATURE``
  is used only by vLLM's computation of ``is_greedy`` (``rejection_sampler.py:484``), which the host
  takes as an argument, so it is dropped.
* The host ``rejection_greedy_sample`` keeps the greedy part of ``rejection_sample``: the asserts on
  the inputs it takes (``rejection_sampler.py:455-466``), the output buffer filled with
  ``PLACEHOLDER_TOKEN_ID`` (``:474-479``), ``is_greedy = None`` when all requests are greedy
  (``:481-484``), and the launch ``_rejection_greedy_sample`` makes (``:874-891``): grid
  ``(batch_size,)``, ``SYNTHETIC_MODE = synthetic_conditional_rates is not None``. It takes
  ``target_argmax`` (vLLM computes it as ``target_logits.argmax(dim=-1)``, ``:511``) and, for
  ``is_greedy``, the per-request flag (vLLM: ``sampling_metadata.temperature ==
  GREEDY_TEMPERATURE``, i.e. ``== 0``). vLLM's launch goes through a warm-up dispatcher
  (``vllm/model_executor/warmup/jit_warmup_triton_helper.py:585-605``), which passes these
  arguments by name with Triton's default launch options; those are written out
  (``num_warps=4``, ``num_stages=3``, Triton 3.8.0's defaults).
* ``reference`` and the ``__main__`` test are new; the cases follow
  ``tests/v1/sample/test_rejection_sampler.py``.
"""

import torch

import triton
import triton.language as tl


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic mode only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            if SYNTHETIC_MODE:
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                # -1 is used for padded draft token ids that should be rejected.
                accepted = (uniform_prob < rate) and draft_token_id >= 0
                token_id = draft_token_id if accepted else target_argmax_id
                rejected = not accepted
            else:
                token_id = target_argmax_id
                rejected = draft_token_id != target_argmax_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


def rejection_greedy_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    target_argmax: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    # [batch_size] or None (all requests greedy)
    is_greedy: torch.Tensor | None = None,
    # synthetic mode only
    uniform_probs: torch.Tensor | None = None,
    synthetic_conditional_rates: torch.Tensor | None = None,
) -> torch.Tensor:
    """The greedy part of vLLM's `rejection_sample` (`rejection_sampler.py:432`)."""
    assert draft_token_ids.ndim == 1
    assert cu_num_draft_tokens.ndim == 1

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    device = target_argmax.device
    assert draft_token_ids.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_argmax.shape == (num_tokens,)

    PLACEHOLDER_TOKEN_ID = -1  # rejection_sampler.py:43

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    rejection_greedy_sample_kernel[(output_token_ids.shape[0],)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
        uniform_probs,
        synthetic_conditional_rates,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        num_warps=4,
        num_stages=3,
    )
    return output_token_ids


def reference(draft, num_draft_tokens, max_spec_len, target_argmax, bonus, is_greedy):
    """Exact, on the CPU."""
    batch_size = len(num_draft_tokens)
    out = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int32)
    start = 0
    for r, n in enumerate(num_draft_tokens):
        if is_greedy is None or bool(is_greedy[r]):
            accepted_all = True
            for pos in range(n):
                t = int(target_argmax[start + pos])
                out[r, pos] = t
                if int(draft[start + pos]) != t:
                    accepted_all = False
                    break
            if accepted_all:
                out[r, n] = int(bonus[r, 0])
        start += n
    return out


def make_batch(num_draft_tokens, vocab_size, accept_rate, generator, pad_drafts=False):
    """Drafts that agree with the target's argmax at each position with probability accept_rate."""
    num_tokens = sum(num_draft_tokens)
    target_argmax = torch.randint(0, vocab_size, (num_tokens,), generator=generator, dtype=torch.int64)
    other = torch.randint(0, vocab_size, (num_tokens,), generator=generator, dtype=torch.int64)
    agree = torch.rand(num_tokens, generator=generator) < accept_rate
    draft = torch.where(agree, target_argmax, other).to(torch.int32)
    if pad_drafts:
        draft[torch.rand(num_tokens, generator=generator) < 0.1] = -1
    bonus = torch.randint(0, vocab_size, (len(num_draft_tokens), 1), generator=generator, dtype=torch.int32)
    cu = torch.tensor(num_draft_tokens, dtype=torch.int64).cumsum(0).to(torch.int32)
    return draft, cu, target_argmax, bonus


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)

    # vLLM's hand-written cases (tests/v1/sample/test_rejection_sampler.py).
    for spec, out_toks, expected in [
        ([[1, 2, 3]], [[1, 2, 3, 4]], [[1, 2, 3, 4]]),             # perfect match
        ([[1, 2, 3]], [[1, 5, 3, 4]], [[1, 5, -1, -1]]),           # early mismatch
        ([[1, 2], [3]], [[1, 2, 5], [3, 4]], [[1, 2, 5], [3, 4, -1]]),
        ([[1]], [[1, 2]], [[1, 2]]),                               # single token
        ([[]], [[5]], [[5]]),                                      # no drafts
        ([[1, 2, 3], [4, 5, 6]], [[1, 2, 3, 4], [4, 7, 6, 8]], [[1, 2, 3, 4], [4, 7, -1, -1]]),
    ]:
        n = [len(s) for s in spec]
        draft = torch.tensor(sum(spec, []), dtype=torch.int32)
        target = torch.tensor(sum([o[:-1] for o in out_toks], []), dtype=torch.int64)
        bonus = torch.tensor([[o[-1]] for o in out_toks], dtype=torch.int32)
        cu = torch.tensor(n, dtype=torch.int64).cumsum(0).to(torch.int32)
        got = rejection_greedy_sample(draft.to(DEVICE), n, max(n), cu.to(DEVICE), target.to(DEVICE),
                                      bonus.to(DEVICE)).cpu()
        ok = torch.equal(got, torch.tensor(expected, dtype=torch.int32))
        print(f"vLLM case {spec}: {'exact' if ok else 'MISMATCH'}")
        assert ok

    # Random batches: batch sizes 1 to 1024, drafts per request 0 to k (k = num_speculative_tokens,
    # 1 to 8 for EAGLE/MTP/n-gram, up to MAX_SPEC_LEN = 128), vocabularies of Llama-3 and Qwen.
    for batch_size, k, vocab, accept, mixed, pad in [
        (1, 1, 128256, 0.7, False, False),
        (8, 3, 128256, 0.7, False, False),
        (64, 5, 151936, 0.8, True, False),
        (256, 4, 32000, 0.5, True, True),
        (1024, 8, 151936, 0.9, True, False),
        (37, 128, 128256, 0.97, True, False),
        (16, 2, 32000, 1.0, False, False),
        (16, 2, 32000, 0.0, True, False),
    ]:
        if mixed:
            n = torch.randint(0, k + 1, (batch_size,), generator=gen).tolist()
        else:
            n = [k] * batch_size
        max_spec_len = max(n)
        draft, cu, target, bonus = make_batch(n, vocab, accept, gen, pad)
        for greedy_mask in [None, torch.rand(batch_size, generator=gen) < 0.5]:
            got = rejection_greedy_sample(draft.to(DEVICE), n, max_spec_len, cu.to(DEVICE), target.to(DEVICE),
                                          bonus.to(DEVICE),
                                          None if greedy_mask is None else greedy_mask.to(DEVICE)).cpu()
            ref = reference(draft, n, max_spec_len, target, bonus, greedy_mask)
            ok = torch.equal(got, ref)
            print(f"batch {batch_size} k {k} vocab {vocab} accept {accept} mixed {mixed} padded drafts {pad} "
                  f"is_greedy {'None' if greedy_mask is None else 'mask'}: {'exact' if ok else 'MISMATCH'}")
            assert ok
    print("✅ greedy rejection sampling matches the reference exactly")
