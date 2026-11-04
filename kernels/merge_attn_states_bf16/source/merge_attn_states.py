"""
Merge attention states
======================

Merges two partial attention outputs of the same queries, computed over two disjoint parts of
the keys (a prefix and a suffix: the cached context and the new tokens, or two KV splits), by
their log-sum-exps (section 2.2 of https://www.arxiv.org/pdf/2501.01005). For token ``t`` and
head ``h``, with ``p = prefix_lse[h, t]`` and ``s = suffix_lse[h, t]`` (an ``+inf`` read as
``-inf``) and ``m = max(p, s)``:

    out[t, h]     = (prefix_output[t, h] * e^(p-m) + suffix_output[t, h] * e^(s-m)) / (e^(p-m) + e^(s-m))
    out_lse[h, t] = log(e^(p-m) + e^(s-m)) + m

and, when both sides are empty (``m = -inf``), ``out = 0`` and ``out_lse = -inf``. Tokens
``t >= prefill_tokens_with_context`` have no prefix: their output and lse are the suffix's,
copied. One program handles one (token, head) pair (grid ``(num_tokens, num_heads)``).

Layouts:

* ``prefix_output``, ``suffix_output``: ``[num_tokens, num_heads, head_size]``, ``bfloat16``;
  the last dimension contiguous, the head stride ``prefix_output.stride(1)`` (possibly padded,
  shared by both), and the token stride ``num_heads`` times the head stride;
* ``prefix_lse``, ``suffix_lse``: ``[num_heads, num_tokens]``, ``float32``, any strides;
* ``output``: ``[num_tokens, num_heads, head_size]``, ``bfloat16``, allocated by the host;
* ``output_lse``: ``[num_heads, num_tokens]``, ``float32``, allocated by the host.

Provenance: vLLM v0.33.0, commit ``4ba787ec99``:

* ``merge_attn_states_kernel``: ``vllm/v1/attention/ops/triton_merge_attn_states.py:72-223``,
  copied verbatim except for its parameters' defaults (below);
* ``merge_attn_states`` (the host wrapper): ``vllm/v1/attention/ops/triton_merge_attn_states.py:14-69``,
  adapted below, with the checks of the dispatcher that calls it,
  ``vllm/v1/attention/ops/merge_attn_states.py:111-128``;
* ``float8_info``: ``vllm/v1/attention/ops/triton_merge_attn_states.py:9``, now a local of the
  host.

Changes from the source:

* ``triton_merge_attn_states.py:6-7``: the imports from vLLM (``current_platform``, and ``tl``
  and ``triton`` through ``vllm.triton_utils``) are replaced by ``import triton`` and
  ``import triton.language as tl``.
* ``triton_merge_attn_states.py:94-95``: the defaults of the kernel's parameters are removed
  (``FP8_MIN: tl.constexpr = float8_info.min`` is now ``FP8_MIN: tl.constexpr``, and ``FP8_MAX``
  likewise): TVA's subset has no kernel parameter defaults, and the host passes each parameter.
* ``triton_merge_attn_states.py:9``: the module global
  ``float8_info = torch.finfo(current_platform.fp8_dtype())`` is a local of the host,
  ``torch.finfo(torch.float8_e4m3fn)`` (the fp8 dtype on an H200), and the host passes
  ``FP8_MIN=float8_info.min`` (``-448.0``) and ``FP8_MAX=float8_info.max`` (``448.0``), the
  values the defaults gave.
* ``triton_merge_attn_states.py:14-69``: the host allocates ``output`` (the inputs' dtype, or
  ``float8_e4m3fn`` with an ``output_scale``) and ``output_lse`` (``float32``, when
  ``return_lse``) fresh and returns them, where the source's wrapper takes them from the caller.
  It also checks what the dispatcher ``merge_attn_states.py:113-128`` checks (the two outputs'
  head strides equal; an fp8 output only with an ``output_scale``). The launch is the source's:
  no autotuning, Triton's default launch options (``num_warps = 4``, ``num_stages = 3``), and the
  constexpr ``OUTPUT_LSE = output_lse is not None``, ``USE_FP8 = output_scale is not None``.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    prefix_lse_head_stride,
    prefix_lse_token_stride,
    suffix_lse_head_stride,
    suffix_lse_token_stride,
    output_lse_head_stride,
    output_lse_token_stride,
    output_scale,  # scale tensor or None
    prefill_tokens_with_context,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    prefix_mask = token_idx < prefill_tokens_with_context

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE

    # For tokens without context (token_idx >= prefill_tokens_with_context),
    # directly copy from suffix_output
    if not prefix_mask:
        s_lse = tl.load(
            suffix_lse
            + head_idx * suffix_lse_head_stride
            + token_idx * suffix_lse_token_stride
        )
        if OUTPUT_LSE:
            tl.store(
                output_lse
                + head_idx * output_lse_head_stride
                + token_idx * output_lse_token_stride,
                s_lse,
            )

        s_out = tl.load(
            suffix_output
            + token_idx * num_heads * prefix_head_stride
            + head_idx * prefix_head_stride
            + head_arange,
            mask=head_mask,
        )

        if USE_FP8:
            s_out = s_out * (1.0 / tl.load(output_scale))
            s_out = tl.clamp(s_out, FP8_MIN, FP8_MAX)
            s_out = s_out.to(output.dtype.element_ty)

        tl.store(
            output
            + token_idx * num_heads * output_head_stride
            + head_idx * output_head_stride
            + head_arange,
            s_out,
            mask=head_mask,
        )
        return

    # For tokens with context (token_idx < prefill_tokens_with_context),
    # perform normal merge operation
    p_lse = tl.load(
        prefix_lse
        + head_idx * prefix_lse_head_stride
        + token_idx * prefix_lse_token_stride
    )
    s_lse = tl.load(
        suffix_lse
        + head_idx * suffix_lse_head_stride
        + token_idx * suffix_lse_token_stride
    )

    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    # Will reuse precomputed Exp values for scale factor computation.
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        # Both sides empty (max_lse == -inf) => undefined merge; keep -inf so
        # downstream merges continue to treat the token as empty.
        out_lse = tl.where(max_lse == float("-inf"), float("-inf"), out_lse)
        tl.store(
            output_lse
            + head_idx * output_lse_head_stride
            + token_idx * output_lse_token_stride,
            out_lse,
        )

    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    # If both sides are empty (max_lse == -inf) the scales are 0/0 = NaN; emit
    # zeros rather than NaN.
    out = tl.where(max_lse == float("-inf"), 0.0, out)

    if USE_FP8:
        out = out * (1.0 / tl.load(output_scale))
        out = tl.clamp(out, FP8_MIN, FP8_MAX)
        out = out.to(output.dtype.element_ty)

    tl.store(
        output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
        + head_arange,
        out,
        mask=head_mask,
    )


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    return_lse: bool = True,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
):
    """The host wrapper, from ``triton_merge_attn_states.py:14-69``; it returns ``output`` and
    ``output_lse`` (``None`` unless ``return_lse``)."""
    # merge_attn_states.py:113-117
    assert prefix_output.stride(1) == suffix_output.stride(1), (
        "merge_attn_states requires prefix_output and suffix_output to have "
        f"matching head strides, got {prefix_output.stride(1)} and "
        f"{suffix_output.stride(1)}"
    )
    float8_info = torch.finfo(torch.float8_e4m3fn)  # triton_merge_attn_states.py:9 on an H200
    output_dtype = prefix_output.dtype if output_scale is None else torch.float8_e4m3fn
    output = torch.empty(prefix_output.shape, dtype=output_dtype, device=prefix_output.device)
    output_lse = (
        torch.empty(prefix_lse.shape, dtype=torch.float32, device=prefix_lse.device)
        if return_lse
        else None
    )
    # merge_attn_states.py:124-128
    if output.dtype not in (torch.float32, torch.half, torch.bfloat16):
        assert output_scale is not None, (
            f"output_scale is required when output is {output.dtype}"
        )

    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    # We assume the output stride on num_head is not always as same as the
    # `suffix_output` and `prefix_output`, as them might be padded by the
    # attention backend.
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)
    # lse tensors are [NUM_HEADS, NUM_TOKENS] but may be non-contiguous views
    # (e.g. a transpose of a backend's [NUM_TOKENS, NUM_HEADS] output), so index
    # them by their actual strides rather than assuming a contiguous layout.
    prefix_lse_head_stride = prefix_lse.stride(0)
    prefix_lse_token_stride = prefix_lse.stride(1)
    suffix_lse_head_stride = suffix_lse.stride(0)
    suffix_lse_token_stride = suffix_lse.stride(1)
    output_lse_head_stride = output_lse.stride(0) if output_lse is not None else 0
    output_lse_token_stride = output_lse.stride(1) if output_lse is not None else 0

    # If prefill_tokens_with_context is None, all tokens should use prefix context
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = num_tokens

    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        prefix_lse_head_stride,
        prefix_lse_token_stride,
        suffix_lse_head_stride,
        suffix_lse_token_stride,
        output_lse_head_stride,
        output_lse_token_stride,
        output_scale,
        prefill_tokens_with_context,
        head_size,
        padded_head_size,
        output_lse is not None,
        output_scale is not None,
        FP8_MIN=float8_info.min,
        FP8_MAX=float8_info.max,
    )
    return output, output_lse


def reference(prefix_output, prefix_lse, suffix_output, suffix_lse, prefill_tokens_with_context):
    """The merge in float64 (output before its rounding to bfloat16, and the lse)."""
    num_tokens = prefix_output.shape[0]
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = num_tokens
    p = prefix_lse.double().clone()
    s = suffix_lse.double().clone()
    p[p == float("inf")] = float("-inf")
    s[s == float("inf")] = float("-inf")
    m = torch.maximum(p, s)
    empty = m == float("-inf")
    m_safe = torch.where(empty, torch.zeros_like(m), m)
    pe = torch.exp(p - m_safe)
    se = torch.exp(s - m_safe)
    tot = pe + se
    lse = torch.where(empty, torch.full_like(m, float("-inf")), torch.log(tot) + m_safe)
    ps = torch.where(empty, torch.zeros_like(m), pe / tot).T[:, :, None]  # [T, H, 1]
    ss = torch.where(empty, torch.zeros_like(m), se / tot).T[:, :, None]
    out = prefix_output.double() * ps + suffix_output.double() * ss
    # tokens without context: the suffix's output and lse, copied
    out[prefill_tokens_with_context:] = suffix_output[prefill_tokens_with_context:].double()
    lse[:, prefill_tokens_with_context:] = suffix_lse[:, prefill_tokens_with_context:].double()
    return out, lse


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # tests/kernels/attention/test_merge_attn_states.py:154-156 (tokens 256 ... 4096, heads 4 ...
    # 64, head sizes 32 ... 256, prefill_tokens_with_context None or 128), its both-empty test
    # (6 tokens, 8 heads, 128), and the layouts the wrapper allows (padded head stride,
    # transposed lse views).
    # (num_tokens, num_heads, head_size, prefill_tokens_with_context, head padding, lse transposed)
    for num_tokens, num_heads, head_size, ptwc, pad, lse_t in [
        (1, 8, 128, None, 0, False),
        (6, 8, 128, None, 0, False),
        (256, 4, 32, None, 0, False),
        (613, 16, 48, 128, 0, False),
        (512, 32, 96, 128, 0, True),
        (1024, 48, 128, None, 64, True),
        (1536, 64, 256, 1, 0, False),
        (4096, 32, 128, 128, 0, False),
        (300, 8, 64, 0, 8, True),
    ]:
        if lse_t:
            prefix_lse = torch.randn((num_tokens, num_heads), generator=gen).T
            suffix_lse = torch.randn((num_tokens, num_heads), generator=gen).T
        else:
            prefix_lse = torch.randn((num_heads, num_tokens), generator=gen)
            suffix_lse = torch.randn((num_heads, num_tokens), generator=gen)
        # +inf on 10% of each side (never both), as the code base's test; -inf on both sides
        # (both empty) on another 5%, and -inf on one side on 5%
        mp = torch.rand((num_heads, num_tokens), generator=gen) < 0.1
        ms = torch.rand((num_heads, num_tokens), generator=gen) < 0.1
        both = mp & ms
        mp, ms = mp & ~both, ms & ~both
        prefix_lse[mp] = float("inf")
        suffix_lse[ms] = float("inf")
        r = torch.rand((num_heads, num_tokens), generator=gen)
        prefix_lse[r < 0.05] = float("-inf")
        suffix_lse[r < 0.05] = float("-inf")
        prefix_lse[(r >= 0.05) & (r < 0.075)] = float("-inf")
        suffix_lse[(r >= 0.075) & (r < 0.1)] = float("-inf")
        if num_tokens == 6:
            prefix_lse[:, 2:4] = float("-inf")
            suffix_lse[:, 2:4] = float("-inf")
        prefix_lse, suffix_lse = prefix_lse.to(DEVICE), suffix_lse.to(DEVICE)
        shape = (num_tokens, num_heads, head_size + pad)
        prefix_output = torch.randn(shape, generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        suffix_output = torch.randn(shape, generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        prefix_output = prefix_output[..., :head_size]
        suffix_output = suffix_output[..., :head_size]

        out, out_lse = merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse,
                                         True, ptwc)
        ref, ref_lse = reference(prefix_output, prefix_lse, suffix_output, suffix_lse, ptwc)
        assert not out.isnan().any()
        # bfloat16 output: within one rounding (2^-8 relative) plus a little
        err = ((out.double() - ref).abs() / (ref.abs() + 1e-2)).max().item()
        assert err < 2**-8 + 2e-3, err
        same_inf = torch.equal(torch.isinf(out_lse), torch.isinf(ref_lse)) and torch.equal(
            out_lse[torch.isinf(out_lse)], ref_lse[torch.isinf(ref_lse)].float())
        fin = torch.isfinite(ref_lse)
        lse_err = (out_lse.double()[fin] - ref_lse[fin]).abs().max().item()
        assert same_inf and lse_err < 1e-5, (same_inf, lse_err)
        out2, lse2 = merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse,
                                       False, ptwc)
        assert lse2 is None and torch.equal(out2, out)
        print(f"tokens {num_tokens} heads {num_heads} head_size {head_size} "
              f"prefill_tokens_with_context {ptwc} head padding {pad} lse transposed {lse_t}: "
              f"max rel error {err:.2e}, lse max abs error {lse_err:.2e}")
    print("✅ merge_attn_states matches the float64 reference")
