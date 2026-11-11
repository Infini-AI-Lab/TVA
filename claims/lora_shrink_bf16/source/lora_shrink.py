"""
Multi-LoRA shrink (bfloat16)
============================

The first half of a batched multi-LoRA linear layer (Punica's SGMV shrink): each token ``t`` whose
adapter is ``l = token_lora_mapping[t] ≥ 0`` gets ``out[s, t, :] = scaling · inputs[t, :] @ A_s[l]ᵀ``
for each slice ``s`` (``A_s[l]`` the adapter's ``[rank, hidden]`` LoRA A matrix); tokens with no adapter
(``-1``) keep a zero row. The tokens are grouped by adapter on the host: ``token_indices_sorted_by_lora_ids``
lists the token indices sorted by adapter id, ``lora_ids`` the distinct ids (``-1`` included when present,
then ``-1`` padding), ``num_tokens_per_lora`` and ``lora_token_start_loc`` each group's size and start.
The grid is ``(SPLIT_K · cdiv(M, BLOCK_M) · cdiv(rank, BLOCK_N), num_slices, num_active_loras)``: a
program takes one adapter group (axis 2), one slice (axis 1), and one ``BLOCK_M × BLOCK_N`` tile of that
group's rows (axis 0, grouped by ``GROUP_SIZE_M``); groups of id ``-1`` and tiles past the group's
size return at once. With ``SPLIT_K > 1`` a program sums one part of K and adds it into the output with
``tl.atomic_add``; with ``SPLIT_K = 1`` it stores its tile.

Layouts:

* ``inputs``: ``[M, hidden]``, ``bfloat16`` (or ``float16``), contiguous;
* ``lora_a_weights``: a list of ``num_slices`` tensors ``[max_loras, 1, rank, hidden]`` (or
  ``[max_loras, rank, hidden]``), ``bfloat16``, contiguous, with equal strides; for one slice the kernel
  reads the tensor itself, for several a ``uint64`` tensor of their addresses;
* ``token_lora_mapping``, ``token_indices_sorted_by_lora_ids``: ``[M]``, ``int32``;
* ``num_tokens_per_lora``, ``lora_ids``: ``[max_loras + 1]``, ``int32``; ``lora_token_start_loc``:
  ``[max_loras + 2]``, ``int32``;
* output: ``[num_slices, M, rank]``, ``float32``.

Provenance: vLLM v0.33.0, commit ``4ba787ec99`` (``4ba787ec9959e566d3b587a796dff07673324da5``),
package ``vllm/``:

* ``mm_k``: ``lora/ops/triton_ops/kernel_utils.py:8-107``;
* ``do_shrink_kernel``: ``lora/ops/triton_ops/kernel_utils.py:236-356``;
* ``_lora_shrink_kernel``: ``lora/ops/triton_ops/lora_shrink_op.py:101-216``;
* ``_lora_shrink`` (the host wrapper): ``lora/ops/triton_ops/lora_shrink_op.py:219-408``;
* ``_get_lora_a_ptr``: ``lora/ops/triton_ops/utils.py:25-71``;
* ``get_lora_op_configs`` (the default shrink config): ``lora/ops/triton_ops/utils.py:195-231``;
* ``LoRAKernelMeta.make``, ``prepare_tensors``, ``meta_args``:
  ``lora/ops/triton_ops/lora_kernel_metadata.py:47-205``;
* the caller: ``PunicaWrapperGPU.add_shrink`` and ``add_lora_linear``,
  ``lora/punica_wrapper/punica_gpu.py:90-122, 240-262`` (a ``float32`` output buffer).

Changes from the source:

* ``kernel_utils.py:8-107, 236-356`` and ``lora_shrink_op.py:101-216``: ``mm_k``, ``do_shrink_kernel``
  and ``_lora_shrink_kernel`` are copied verbatim (they have no autotune or heuristics decorator), except
  that the defaults of ``_lora_shrink_kernel``'s parameters are removed (``lora_shrink_op.py:131-134``:
  ``STORE_PARTIALS: tl.constexpr = False``, ``output_split_stride=0``, ``PARTIAL_N: tl.constexpr = 0``,
  ``PARTIAL_K: tl.constexpr = 0`` become ``STORE_PARTIALS: tl.constexpr``, ``output_split_stride``,
  ``PARTIAL_N: tl.constexpr``, ``PARTIAL_K: tl.constexpr``): TVA's subset has no kernel parameter
  defaults; the host passes each, with the default's value.
* ``lora_kernel_metadata.py:47-205``: ``lora_kernel_meta`` builds the same tensors as
  ``LoRAKernelMeta.make`` followed by ``prepare_tensors`` and ``meta_args``, with buffers of exactly
  ``M`` tokens (vLLM's hold ``max_num_tokens`` and are sliced to ``M``) and no CUDA-graph capture
  sizes (``captured_lora_counts`` empty, vLLM's default); ``num_active_loras`` is ``max_loras + 1``
  unless ``specialize_active_lora`` (default ``False``, ``config/lora.py:67``).
* ``utils.py:25-71``: ``_get_lora_a_ptr`` is copied without its cache ``_LORA_A_PTR_DICT``, and with
  ``torch.tensor(…, dtype=torch.uint64, device=device)`` for ``async_tensor_h2d``
  (``utils/torch_utils.py:702``).
* ``lora_shrink_op.py:219-408``: the host ``lora_shrink`` is ``_lora_shrink``, adapted:
  - it allocates its output fresh, ``torch.zeros((num_slices, M, rank), dtype=torch.float32)``, and
    returns it, where vLLM zeroes the caller's ``output_tensor`` (``lora_shrink_op.py:296``) and writes
    into it; with no tokens, or no token with an adapter, it returns the zeros;
  - ``lora_shrink_op.py:306-321``: ``get_lora_op_configs`` is replaced by its default shrink config
    (``utils.py:217-231``, the one taken when ``VLLM_TUNED_CONFIG_FOLDER`` is unset), written in the
    host: ``block_m = 32``, ``block_n = 16``, ``num_warps = 4``, ``num_ctas = 1``, ``group_size_m = 8``,
    ``num_stages = 2``, and ``block_k = 256``, ``split_k = 64`` for a batch of fewer than 128 tokens,
    ``block_k = 32``, ``split_k = 8`` from 128 tokens (``is_batch_invariant`` off);
  - ``lora_shrink_op.py:262-275, 323-326, 344-352, 387-406``: the batch-invariant path
    (``VLLM_BATCH_INVARIANT``, default off: ``partials`` and ``_lora_shrink_reduce_kernel``) is
    dropped;
  - ``lora_shrink_op.py:339-343``: ``use_gdc = False`` (``VLLM_LORA_ENABLE_DUAL_STREAM`` is off by
    default), so ``USE_GDC = False`` and ``launch_pdl = False``;
  - ``EVEN_K``, the grid and the launch's arguments are those of ``lora_shrink_op.py:327-386``.
* ``reference`` and the test are new (after vLLM's ``tests/lora/test_punica_ops.py``).
"""

import torch

import triton
import triton.language as tl


@triton.jit
def mm_k(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    ak_stride: int,
    bk_stride: int,
    offset_k: tl.tensor,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    b_dtype: tl.constexpr,
    USE_GDC: tl.constexpr,
    base_k: int,
):
    """Given a_ptr and b_ptr, that identify the rows of A (m x k) and columns of
    B (k x n), iterate, through the K dimension to compute the partial/complete
    matrix block product.
    If SPLIT_K == 1, the output m x n product is complete.
    If SPLIT_K > 1, the thread block computes partial outputs. The partial
    outputs are then atomically summed in the caller code.

    Args:
        a_ptr: Array of pointers, identifying rows of A
        b_ptr: Array of pointers, identifying columns of B
        ak_stride: K dimension stride of the A matrix
        bk_stride: K dimension stride of the B matrix
        offset_k: Offsets within the current BLOCK_K tile along K
        K: Length of the K dimension
        BLOCK_M: M dimension of the output block m x n
        BLOCK_N: N dimension of the output block m x n
        BLOCK_K: K dimension atom
        EVEN_K: True if the blocks of A and B can be loaded without any
            masking.
        SPLIT_K: Parameter signifying parallelism in the K dimension.
        CAST_TYPE: if True, cast the values from the A matrix to the B
            matrix dtype.
        b_dtype: datatype of the B matrix
        USE_GDC: Whether to use PDL. True indicates use.
        base_k: Base offset along K dimension for current SPLIT_K group

    """
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Step size along K for each iteration
    STEP_K = BLOCK_K * SPLIT_K

    # Total number of iterations (compile-time constant)
    num_iters = tl.cdiv(K, STEP_K)

    for k in range(num_iters):
        # Current iteration's global K offset
        iter_k = k * STEP_K + base_k

        # Check if this iteration is completely valid (no masking needed)
        block_end = iter_k + BLOCK_K

        if EVEN_K:
            # K is divisible by BLOCK_K, no masking ever needed
            # pre-fetch lora weight
            tiled_b = tl.load(b_ptr)
            if USE_GDC:
                tl.extra.cuda.gdc_wait()
            tiled_a = tl.load(a_ptr)
            if CAST_TYPE:
                tiled_a = tiled_a.to(b_dtype)
            accumulator += tl.dot(tiled_a, tiled_b)
        else:
            # Check if we need element-wise masking
            if iter_k >= K:
                # Entire block out of range, skip
                pass
            elif block_end <= K:
                # Entire block in range, no masking needed (fast path)
                tiled_b = tl.load(b_ptr)
                if USE_GDC:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(a_ptr)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator += tl.dot(tiled_a, tiled_b)
            else:
                # Partial block, need masking (only last iteration)
                k_offsets = tl.arange(0, BLOCK_K)
                mask = iter_k + k_offsets < K
                tiled_b = tl.load(b_ptr, mask=mask[:, None], other=0.0)
                if USE_GDC:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(a_ptr, mask=mask[None, :], other=0.0)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator += tl.dot(tiled_a, tiled_b)

        a_ptr += STEP_K * ak_stride
        b_ptr += STEP_K * bk_stride

    return accumulator


@triton.jit
def do_shrink_kernel(
    pid_n,
    pid_sk,
    slice_id,
    lora_index,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    # input strides
    input_d0_stride,
    input_d1_stride,
    # lora strides
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    # output strides
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    output_split_stride,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
    STORE_PARTIALS: tl.constexpr,
):
    """Given an array of integers that identifies the rows of A, ram,
    a lora index that identifies which LoRA to use from lora_ptr, lora_index,
    a slice_id that identifies the input/output slice, compute the
    matrix product and store in the appropriate output location.
    """
    # Identify the lora_ptr from slice_id.
    if SLICE_NUM == 1:
        # current lora ptr
        cur_lora_ptr = lora_ptr
    else:
        # current lora ptr
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty)
        )

    # Identify the column indices of B to process.
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # Identify A and B block pointers
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = (
        input_ptr
        # int64 to keep the row offset from overflowing at long context lengths
        + ram[:, None].to(tl.int64) * input_d0_stride
        + offset_k[None, :] * input_d1_stride
    )
    b_ptr = (
        cur_lora_ptr
        + lora_d0_stride * lora_index
        + rbn[None, :] * lora_d1_stride
        + offset_k[:, None] * lora_d2_stride
    )

    # Compute partial/complete block matrix product.
    accumulator = mm_k(
        a_ptr,
        b_ptr,
        input_d1_stride,
        lora_d2_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        False,
        cur_lora_ptr.dtype.element_ty,
        False,  # USE_GDC is always False in shrink kernel
        base_k=pid_sk * BLOCK_K,
    )
    # GDC launch dependents hints the runtime system to launch dependent kernels.
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
    # Identify the C output pointers to store the results of the accumulator.
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    if STORE_PARTIALS:
        cur_out_ptr = (
            out_ptr
            + slice_id.to(tl.int64) * output_d0_stride
            + pid_sk.to(tl.int64) * output_split_stride
        )
    else:
        cur_out_ptr = (
            out_ptr if SLICE_NUM == 1 else out_ptr + slice_id * output_d0_stride
        )
    c_ptr = (
        cur_out_ptr
        + ram[:, None] * output_d1_stride
        + offset_cn[None, :] * output_d2_stride
    )
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)

    if STORE_PARTIALS:
        # Unscaled FP32 partial per K split; the reducer adds the splits in a
        # fixed order and applies scaling once.
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        accumulator *= scaling
        # handles write-back with reduction-splitting
        if SPLIT_K == 1:
            tl.store(c_ptr, accumulator, mask=c_mask)
        else:
            tl.atomic_add(c_ptr, accumulator, mask=c_mask, sem="relaxed")


@triton.jit
def _lora_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    STORE_PARTIALS: tl.constexpr,
    output_split_stride,
    PARTIAL_N: tl.constexpr,
    PARTIAL_K: tl.constexpr,
):
    if STORE_PARTIALS:
        # The split-K partial stage specializes on N and K.
        N = PARTIAL_N
        K = PARTIAL_K
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K

    pid_m_n = pid_sk_m_n // SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * cta_n_num
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(cta_m_num - first_pid_m, GROUP_SIZE_M)

    # Column-major ordering within groups for better cache reuse
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # num rows this CTA should process.
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    # Identify all rows that this CTA should process.
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )
    # Load all relevant row indices.
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_shrink_kernel(
        pid_n,
        pid_sk,
        slice_id,
        lora_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
        ram,  # array identifying the rows of Input ptr to operate on
        # input strides
        input_d0_stride,
        input_d1_stride,
        # lora strides
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,
        # output strides
        output_d0_stride,
        output_d1_stride,
        output_d2_stride,
        output_split_stride,
        scaling,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        SLICE_NUM,
        USE_GDC,
        STORE_PARTIALS,
    )


def _get_lora_a_ptr(lora_a_weights: list[torch.Tensor], device: torch.device):
    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    tensor_ptrs = []
    for lora_a_weight in lora_a_weights:
        if lora_a_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
            assert lora_a_weight.size(1) == 1
            lora_a_weight = lora_a_weight.squeeze(dim=1)
        else:
            assert lora_a_weight.ndim == 3  # shape:(lora_num,size,rank)
        assert lora_a_weight.is_contiguous()
        tensor_ptrs.append(lora_a_weight.data_ptr())
        lora_strides_d0.append(lora_a_weight.stride(0))
        lora_strides_d1.append(lora_a_weight.stride(1))
        lora_strides_d2.append(lora_a_weight.stride(2))
    if len(lora_a_weights) > 1:
        lora_ptr_tensor = torch.tensor(tensor_ptrs, dtype=torch.uint64, device=device)
    else:
        lora_ptr_tensor = lora_a_weights[0]

    if (
        len(set(lora_strides_d0)) > 1
        or len(set(lora_strides_d1)) > 1
        or len(set(lora_strides_d2)) > 1
    ):
        raise ValueError("All LoRA weights must have the same stride.")

    return (
        lora_ptr_tensor,
        lora_strides_d0[0],
        lora_strides_d1[0],
        lora_strides_d2[0],
    )


def lora_kernel_meta(token_lora_mapping: torch.Tensor, max_loras: int, specialize_active_lora: bool = False):
    """The metadata of ``LoRAKernelMeta`` for one forward pass: ``(token_lora_mapping,
    token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, active_lora_ids,
    no_lora_flag_cpu, num_active_loras)``, as ``meta_args`` returns them."""
    device = token_lora_mapping.device
    num_tokens = token_lora_mapping.size(0)
    # make (lora_kernel_metadata.py:47-98) and _reset (100-105)
    token_indices_sorted_by_lora_ids = torch.empty(num_tokens, dtype=torch.int32, device=device)
    active_lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32, device=device)
    num_tokens_per_lora = torch.zeros(max_loras + 1, dtype=torch.int32, device=device)
    lora_token_start_loc = torch.zeros(max_loras + 2, dtype=torch.int32, device=device)
    no_lora_flag_cpu = torch.tensor([False], dtype=torch.bool, device="cpu")
    num_active_loras_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
    default_num_active_loras_cpu = torch.tensor([max_loras + 1], dtype=torch.int32, device="cpu")

    # prepare_tensors (lora_kernel_metadata.py:107-165)
    no_lora = torch.all(token_lora_mapping == -1)
    no_lora_flag_cpu[0] = no_lora
    if not no_lora:
        _, sorted_indices = torch.sort(token_lora_mapping, stable=True)
        token_indices_sorted_by_lora_ids.copy_(sorted_indices)
        lora_ids, counts = torch.unique(token_lora_mapping, sorted=True, return_counts=True)
        active_lora_ids[: lora_ids.size(0)].copy_(lora_ids)
        num_tokens_per_lora[: counts.size(0)].copy_(counts)
        num_active_loras_cpu[0] = lora_ids.size(0)
        start_loc = torch.cumsum(counts, dim=0)
        lora_token_start_loc[1 : 1 + start_loc.size(0)].copy_(start_loc)

    # meta_args (lora_kernel_metadata.py:167-205)
    num_active_loras = num_active_loras_cpu if specialize_active_lora else default_num_active_loras_cpu
    return (
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
    )


@torch.inference_mode()
def lora_shrink(
    inputs: torch.Tensor,  #  shape [num_tokens, hidden_size]
    lora_a_weights: list[torch.Tensor],  # shape [num_loras, lora_rank, hidden_size]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens]
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    no_lora_flag_cpu: torch.Tensor,  # shape [1]
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    scaling: float,
) -> torch.Tensor:
    """The shrink ``out[s, t] = scaling · inputs[t] @ lora_a_weights[s][token_lora_mapping[t]]ᵀ``, a fresh
    ``[num_slices, num_tokens, rank]`` ``float32`` tensor, zero on tokens with no adapter."""
    N = lora_a_weights[0].shape[-2]  # rank
    output_tensor = torch.zeros(
        (len(lora_a_weights), inputs.size(0), N), dtype=torch.float32, device=inputs.device
    )
    assert no_lora_flag_cpu.numel() == 1
    if inputs.size(0) == 0:
        # An empty first call has no work to do.
        return output_tensor
    if no_lora_flag_cpu.item():
        # None of the inputs require LoRA.
        return output_tensor

    assert inputs.dtype == lora_a_weights[0].dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_a_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(1) == lora_a_weights[0].size(-1)
    inputs = inputs.contiguous()
    assert output_tensor.is_contiguous()

    # metadata sanity check
    M = inputs.size(0)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    (lora_ptr_tensor, lora_strides_d0, lora_strides_d1, lora_strides_d2) = (
        _get_lora_a_ptr(lora_a_weights, inputs.device)
    )
    N, K = lora_a_weights[0].shape[-2:]  # K=hidden_size,N=rank
    NUM_SLICES = len(lora_a_weights)

    # The default shrink config (utils.py:217-231), which depends on the batch M.
    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_K = 256 if M < 128 else 32
    SPLIT_K = 64 if M < 128 else 8
    NUM_WARPS = 4
    NUM_STAGES = 2
    NUM_CTAS = 1
    GROUP_SIZE_M = 8

    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0  # type: ignore

    grid = (
        SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        NUM_SLICES,
        num_active_loras.item(),
    )

    use_gdc = False
    first_out = output_tensor
    _lora_shrink_kernel[grid](
        inputs,
        lora_ptr_tensor,
        first_out,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        first_out.stride(0),
        first_out.stride(-2),
        first_out.stride(-1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        GROUP_SIZE_M,
        NUM_SLICES,
        use_gdc,
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        num_stages=NUM_STAGES,
        launch_pdl=use_gdc,
        STORE_PARTIALS=False,
        output_split_stride=0,
        PARTIAL_N=0,
        PARTIAL_K=0,
    )

    return output_tensor


def reference(inputs, lora_a_weights, token_lora_mapping, scaling):
    """``scaling · inputs[t] @ A_s[l]ᵀ`` in ``float64`` for tokens with an adapter ``l ≥ 0``, zero
    otherwise."""
    M = inputs.size(0)
    out = torch.zeros((len(lora_a_weights), M, lora_a_weights[0].shape[-2]), dtype=torch.float64,
                      device=inputs.device)
    x = inputs.double()
    for s, w in enumerate(lora_a_weights):
        w = w.reshape(w.size(0), w.shape[-2], w.shape[-1]).double()
        for l in range(w.size(0)):
            rows = torch.nonzero(token_lora_mapping == l, as_tuple=True)[0]
            if rows.numel():
                out[s, rows] = scaling * (x[rows] @ w[l].t())
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (tokens, hidden, rank, max_loras, num_slices, fraction of tokens with no adapter, specialize), from
    # vLLM's tests/lora/test_punica_ops.py:320-421 (hidden 128..128256 and their TP shards, 2049; batches
    # of 1..32 sequences of 128 tokens; max_loras 1..128; ranks 1..256; 1 to 3 slices) and the model
    # sizes there (hidden 4096, 5120, 8192, 14336).
    cases = [
        (1, 4096, 16, 4, 1, 0.0, False),
        (5, 2049, 8, 8, 1, 0.3, False),
        (37, 2049, 1, 1, 1, 0.5, True),
        (128, 4096, 64, 32, 1, 0.1, False),
        (512, 2049, 256, 8, 2, 0.2, False),
        (300, 14336, 32, 4, 3, 0.0, True),
        (2048, 8192, 16, 128, 1, 0.1, False),
        (4096, 5120, 128, 8, 1, 0.0, False),
        (64, 4096, 16, 4, 1, 1.0, False),
    ]
    for M, hidden, rank, max_loras, nslices, no_lora_frac, specialize in cases:
        inputs = torch.rand((M, hidden), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        weights = [torch.rand((max_loras, 1, rank, hidden), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
                   for _ in range(nslices)]
        mapping = torch.randint(0, max_loras, (M,), generator=gen, dtype=torch.int32)
        mapping[torch.rand(M, generator=gen) < no_lora_frac] = -1
        mapping = mapping.to(DEVICE)
        scaling = 0.5
        meta = lora_kernel_meta(mapping, max_loras, specialize)
        out = lora_shrink(inputs, weights, *meta, scaling)
        ref = reference(inputs, weights, mapping, scaling)
        err = (out.double() - ref).abs()
        rel = (err / (ref.abs() + ref.abs().mean() + 1e-30)).max().item()
        print(f"tokens {M} hidden {hidden} rank {rank} max_loras {max_loras} slices {nslices} "
              f"no-adapter fraction {no_lora_frac} specialize {specialize}: "
              f"max error / (|ref| + mean |ref|) {rel:.2e}")
        assert out.shape == (nslices, M, rank) and out.dtype == torch.float32
        assert rel < 2e-4, rel  # float32 accumulation over K up to 14336
    print("✅ LoRA shrink matches the float64 reference")
