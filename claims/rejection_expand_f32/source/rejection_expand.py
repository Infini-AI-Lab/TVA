"""
Per-request values expanded to tokens for speculative decoding (vLLM)
=====================================================================

A batch of requests, request ``r`` with ``n_r`` packed tokens (``n_r`` may be ``0``): request ``r``'s
tokens are positions ``start_r : end_r`` with ``end_r = cu_num_tokens[r]`` and
``start_r = cu_num_tokens[r - 1]`` (``0`` for ``r = 0``). ``expand_batch_to_tokens`` writes, at
every position of request ``r``, the request's value ``x[r]``, with ``replace_from`` replaced by
``replace_to``: ``out[p] = replace_to if x[r] == replace_from else x[r]`` for
``start_r <= p < end_r``. One program per request writes its positions with one masked store of
``MAX_NUM_TOKENS = MAX_SPEC_LEN = 128`` lanes.

vLLM's sampler (``apply_sampling_constraints``) calls it three times per non-greedy step, with
``cu_num_tokens = cu_num_draft_tokens`` and ``num_tokens`` the number of draft tokens:

* ``temperature`` (``[batch_size]``, ``float32``) with ``replace_from = 0``, ``replace_to = 1``
  (greedy requests get temperature ``1``);
* ``top_k`` (``[batch_size]``, ``int32``) and ``top_p`` (``[batch_size]``, ``float32``), with the
  defaults ``replace_from = replace_to = 0``.

Layouts and dtypes:

* ``x``: ``[batch_size]``, ``float32`` or ``int32``, contiguous;
* ``cu_num_tokens``: ``[batch_size]``, ``int32``, the inclusive prefix sums of the ``n_r``, each
  ``n_r <= MAX_SPEC_LEN``;
* ``expanded_x`` (output, ``x.new_empty(num_tokens)``): ``[num_tokens]``, the dtype of ``x``,
  ``num_tokens = sum_r n_r``.

Provenance: vLLM v0.33.0 (commit ``4ba787ec99``), ``vllm/v1/sample/rejection_sampler.py``:

* ``MAX_SPEC_LEN``: ``rejection_sampler.py:47``;
* ``expand_batch_to_tokens``: ``rejection_sampler.py:648-686``;
* ``expand_kernel``: ``rejection_sampler.py:1021-1042`` (comment at 1021,
  ``@triton.jit(do_not_specialize=["replace_from", "replace_to"])`` at 1022, ``def`` at 1023);
* ``_expand``: ``rejection_sampler.py:1057-1067``.

Changes from the source:

* ``rejection_sampler.py:1021-1042``: the kernel, copied verbatim with its decorator; it has no
  autotuning or heuristics and calls no helper.
* ``rejection_sampler.py:47``: ``MAX_SPEC_LEN = 128``, copied verbatim.
* ``rejection_sampler.py:648-686``: ``expand_batch_to_tokens``, copied verbatim (docstring
  included).
* ``rejection_sampler.py:1057-1067``: ``_expand`` launches the kernel directly instead of through
  vLLM's warm-up dispatcher (``@triton_kernel_dispatcher_with_warmup``, with its warm-up inputs
  ``_expand_warmup_inputs`` at ``:1045-1054``; the dispatcher's launch is
  ``vllm/model_executor/warmup/jit_warmup_triton_helper.py:585-605`` and ``:315-358``). The
  dispatcher passes the same arguments by name, grid ``(input.shape[0],)`` and
  ``MAX_NUM_TOKENS=MAX_SPEC_LEN`` (the ``DispatchSpec`` of ``:1067``), with Triton's default launch
  options; those are written out (``num_warps=4``, ``num_stages=3``, Triton 3.8.0's defaults).
* ``GREEDY_TEMPERATURE`` (``rejection_sampler.py:44``) is the local ``GREEDY_TEMPERATURE = 0`` of the
  test, which mirrors the calls of ``apply_sampling_constraints`` (``:617-641``).
* ``reference`` and the ``__main__`` test are new.
"""

import torch

import triton
import triton.language as tl

# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.

    Returns:
        expanded_x: [num_tokens] tensor.

    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    _expand(
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
    )
    return expanded_x


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = tl.zeros([], dtype=cu_num_tokens_ptr.dtype.element_ty)
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


def _expand(
    output: torch.Tensor,
    input: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    replace_from: int,
    replace_to: int,
):
    """The launch vLLM's dispatcher makes (rejection_sampler.py:1057-1067)."""
    expand_kernel[(input.shape[0],)](
        output_ptr=output,
        input_ptr=input,
        cu_num_tokens_ptr=cu_num_tokens,
        replace_from=replace_from,
        replace_to=replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,
        num_warps=4,
        num_stages=3,
    )


def reference(x, num_tokens_per_req, replace_from=0, replace_to=0):
    """Exact, on the CPU: ``x[r]`` (``replace_to`` where it equals ``replace_from``) repeated ``n_r`` times."""
    rf = torch.tensor(replace_from, dtype=x.dtype)
    rt = torch.tensor(replace_to, dtype=x.dtype)
    vals = torch.where(x == rf, rt, x)
    counts = torch.tensor(num_tokens_per_req, dtype=torch.int64)
    return torch.repeat_interleave(vals, counts)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    GREEDY_TEMPERATURE = 0  # rejection_sampler.py:44

    # The docstring's example.
    x = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    cu = torch.tensor([2, 5, 6], dtype=torch.int32)
    got = expand_batch_to_tokens(x.to(DEVICE), cu.to(DEVICE), 6).cpu()
    ok = torch.equal(got, torch.tensor([1.5, 1.5, 2.5, 2.5, 2.5, 3.5]))
    print(f"docstring example: {'exact' if ok else 'MISMATCH'}")
    assert ok

    # The three calls of apply_sampling_constraints (rejection_sampler.py:617-641) on random
    # batches: batch sizes 1 to 1024, drafts per request 0 to k (up to MAX_SPEC_LEN).
    for batch_size, k, mixed in [
        (1, 1, False),
        (8, 3, False),
        (64, 5, True),
        (256, 4, True),
        (1024, 8, True),
        (37, 128, True),
        (16, 2, False),
    ]:
        if mixed:
            n = torch.randint(0, k + 1, (batch_size,), generator=gen).tolist()
        else:
            n = [k] * batch_size
        num_tokens = sum(n)
        cu = torch.tensor(n, dtype=torch.int64).cumsum(0).to(torch.int32)
        greedy = torch.rand(batch_size, generator=gen) < 0.3
        temperature = torch.where(greedy, 0.0, torch.rand(batch_size, generator=gen) * 2).float()
        top_k = torch.randint(0, 100, (batch_size,), generator=gen, dtype=torch.int32)
        top_p = torch.where(torch.rand(batch_size, generator=gen) < 0.2, 0.0,
                            torch.rand(batch_size, generator=gen)).float()
        for name, val, rf, rt in [
            ("temperature", temperature, GREEDY_TEMPERATURE, 1),
            ("top_k", top_k, 0, 0),
            ("top_p", top_p, 0, 0),
        ]:
            got = expand_batch_to_tokens(val.to(DEVICE), cu.to(DEVICE), num_tokens,
                                         replace_from=rf, replace_to=rt).cpu()
            ref = reference(val, n, rf, rt)
            ok = got.view(torch.int32).equal(ref.view(torch.int32))
            print(f"batch {batch_size} k {k} mixed {mixed} {name}: {'exact' if ok else 'MISMATCH'}")
            assert ok
    print("✅ expand_batch_to_tokens matches the reference exactly")
