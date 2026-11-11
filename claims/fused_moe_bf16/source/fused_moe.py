"""
Fused MoE expert GEMM (bfloat16)
================================

The expert GEMMs of a mixture-of-experts layer: each token is routed to ``top_k`` experts. The first
GEMM's row ``t · top_k + j`` is token ``t``'s hidden state times the weights of its ``j``-th expert,
``C[t, j, :] = A[t, :] @ w1[topk_ids[t, j]]ᵀ``; the second GEMM's is row ``t · top_k + j`` of
``intermediate_cache2`` times ``w2[topk_ids[t, j]]ᵀ`` (the kernel's ``top_k = 1``); either is optionally
scaled by the routing weight ``topk_weights[t, j]`` (``MUL_ROUTED_WEIGHT``). The routed rows are sorted
by expert and each expert's list padded to a multiple of ``BLOCK_SIZE_M`` (``moe_align_block_size``):
``sorted_token_ids`` lists the flat indices ``t · top_k + j`` (the padding is ``M · top_k``),
``expert_ids`` the expert of each ``BLOCK_SIZE_M`` block, ``num_tokens_post_padded`` the padded total.
For decode-sized batches (``M · top_k · 4 ≤ E``, no expert map) there is no sorting
(``naive_block_assignment``): block ``f`` is flat index ``f`` alone, of expert ``topk_ids.view(-1)[f]``.
Each program computes one ``BLOCK_SIZE_M × BLOCK_SIZE_N`` tile; programs past the padded total return.

Layouts (all contiguous): ``A`` ``[M, K]`` (or ``intermediate_cache2`` ``[M · top_k, K]``), ``bfloat16``;
``B`` ``[E, N, K]``, ``bfloat16``; ``topk_weights`` ``[M, top_k]``, ``float32``; ``topk_ids``
``[M, top_k]``, ``int32``; ``sorted_token_ids`` ``[max_num_tokens_padded]``, ``int32``; ``expert_ids``
``[cdiv(max_num_tokens_padded, BLOCK_SIZE_M)]`` (or ``topk_ids.view(-1)``), ``int32``;
``num_tokens_post_padded`` ``[1]``, ``int32``; ``C`` ``[M, top_k, N]``, ``bfloat16``.

Provenance: vLLM v0.33.0, commit ``4ba787ec99`` (``4ba787ec9959e566d3b587a796dff07673324da5``),
package ``vllm/``, in ``model_executor/layers/fused_moe/``:

* ``write_zeros_to_output``: ``fused_moe.py:44-61``; ``fused_moe_kernel``: ``fused_moe.py:297-608``;
* ``invoke_fused_moe_triton_kernel`` (the launcher): ``fused_moe.py:761-910``;
* ``fused_experts_impl`` (the caller; its two GEMMs): ``fused_moe.py:1664-1868``;
* ``get_default_config``: ``fused_moe.py:1298-1418``; ``_prepare_expert_assignment``:
  ``fused_moe.py:1549-1599``;
* ``moe_align_block_size``: ``moe_align_block_size.py:55-169``, which calls the CUDA op
  ``ops.moe_align_block_size`` (``csrc/moe/moe_align_sum_kernels.cu:119-229, 231-337, 339-368, 673-787``).

Changes from the source:

* ``fused_moe.py:44-61, 297-608``: ``write_zeros_to_output`` and ``fused_moe_kernel`` are copied verbatim
  (they have no autotune or heuristics decorator), except that the default of ``fused_moe_kernel``'s
  last parameter is removed (``fused_moe.py:353``: ``USE_TD: tl.constexpr = False`` becomes
  ``USE_TD: tl.constexpr``): TVA's subset has no kernel parameter defaults; the host passes it
  (``USE_TD=False``, as vLLM's launcher does).
* ``moe_align_block_size.py:55-169`` and ``csrc/moe/moe_align_sum_kernels.cu``: vLLM's
  ``moe_align_block_size`` launches a CUDA op; ``moe_align_block_size`` here is a torch-only
  equivalent, with the same buffer sizes and contents: ``sorted_token_ids`` filled with
  ``topk_ids.numel()``, then each expert's routed flat indices at its padded offset; ``expert_ids`` the
  expert of each block up to the padded total and ``-1`` after it; ``num_tokens_post_padded`` the
  padded total. Ids outside ``[0, num_experts)``, and with ``ignore_invalid_experts`` ids the expert
  map sends to ``-1``, are not routed (``get_local_expert_id``, ``moe_align_sum_kernels.cu:119-132``).
  Within one expert the indices are in ascending order (the CUDA op's order is its threads'; the
  kernel's result does not depend on it). ``return_scatter_idx`` (unused by the caller) is dropped.
* ``fused_moe.py:1095-1165, 1298-1450``: the configuration as vLLM picks it with no override and no
  ``VLLM_TUNED_CONFIG_FOLDER``: the tuned file ``E=<E>,N=<N>,device_name=<device>.json`` of vLLM's
  ``configs/`` (``source/configs/`` holds vLLM's 22 ``NVIDIA_H200`` files, verbatim; every H200 maps
  to that name), its entry for the batch size nearest ``M``, else ``get_default_config(M, E)``, kept
  for its general ``bfloat16`` branch on CUDA (``fused_moe.py:1372-1416``; the batch-invariant, fp8
  and wna16 branches and the ROCm stage count dropped): ``(16, 64, 128)`` for ``M ≤ 32``,
  ``(32, 64, 128)`` for ``M ≤ 64``, ``(32, 128, 64)`` for ``M ≤ 96``, ``(64, 128, 64)`` for ``M ≤ 512``,
  ``(128, 128, 64)`` above, ``GROUP_SIZE_M`` ``16`` when ``M // E > 128``. The device name is read with
  ``torch.cuda.get_device_name`` (vLLM's platform does the same on CUDA); the imports are local.
* ``fused_moe.py:1549-1599``: ``_prepare_expert_assignment`` for weights in device memory and no
  wna16 block shape (``weights_host_resident`` and the wna16 test false), with
  ``ignore_invalid_experts=True``, as ``fused_experts_impl`` calls it. With ``M · top_k · 4 ≤ E``
  and no expert map it is the naive assignment: ``sorted_token_ids`` is ``None``, ``expert_ids`` is
  ``topk_ids.view(-1)`` and ``num_tokens_post_padded`` holds ``M · top_k · BLOCK_SIZE_M``; otherwise
  ``moe_align_block_size``.
* ``fused_moe.py:1664-1868``: ``fused_moe_gemm1`` is ``fused_experts_impl`` up to and including the
  first GEMM, ``fused_moe_gemm2`` its second GEMM (``fused_moe.py:1836-1861``: ``top_k = 1``,
  ``mul_routed_weight = not apply_router_weight_on_input``), for unquantised activations and weights
  (the quantisation flags false, ``ocp_mx_scheme`` and the scales ``None``, no biases): their asserts;
  ``intermediate_cache1``, ``intermediate_cache3`` allocated fresh as ``[M, top_k, N]``, ``[M, top_k, K]``
  (vLLM views one buffer it shares, ``fused_moe.py:1749-1755``), ``intermediate_cache3`` zeroed with an
  expert map as vLLM does; ``fused_moe_gemm2`` takes ``intermediate_cache2`` as its input and the
  configuration from ``topk_ids.size(0)``, vLLM's ``M``. The activation, the quantisation of the
  inputs and ``moe_sum`` are not part of this file.
* ``fused_moe.py:913-1013`` (``dispatch_fused_moe_kernel``): its asserts are kept; the wna16 branch
  (int4/int8 weights) is dropped.
* ``fused_moe.py:761-910`` (``invoke_fused_moe_triton_kernel``): kept for the unquantised path with no
  bias: ``SWAP_AB = False`` (fp8 only), ``USE_TD = False`` (``resolve_moe_use_td``, ``fused_moe/utils.py:736-748``,
  is false on CUDA with ``VLLM_TRITON_USE_TD`` unset), ``A_scale``, ``B_scale``, ``B_bias`` ``None``, their
  strides ``0``, ``group_n = group_k = 0``, ``HAS_BIAS = False``; the grid computed directly.
* ``reference`` and the test are new (after vLLM's ``tests/kernels/moe/test_moe.py``).
"""

import torch

import triton
import triton.language as tl


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SWAP_AB: tl.constexpr,
    # Tensor-descriptor path for the A gather and B load in the K-loop.
    USE_TD: tl.constexpr,
):
    """Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    # Cast to int64 to prevent overflow in stride*offset products
    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
    offs_token = offs_token.to(tl.int64)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # TD gather and the SWAP_AB accumulator layout are mutually exclusive.
    tl.static_assert(not (USE_TD and SWAP_AB))
    if USE_TD:
        # ``tt.descriptor_gather`` requires block_shape[0] == 1 and i32 idx.
        m_td = num_valid_tokens // top_k
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr,
            shape=(m_td, K),
            strides=(stride_am, stride_ak),
            block_shape=(1, BLOCK_SIZE_K),
        )
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr + off_experts * stride_be,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        )
        gather_idx = (offs_token // top_k).to(tl.int32)
    elif SWAP_AB:
        a_ptrs = a_ptr + (
            offs_k[:, None] * stride_ak + offs_token[None, :] // top_k * stride_am
        )
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        )
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if USE_TD:
            a = a_desc.gather(gather_idx, k * BLOCK_SIZE_K)
            b = b_desc.load([pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]).T
        elif SWAP_AB:
            a_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & token_mask[None, :]
            b_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if SWAP_AB:
                    accumulator += tl.dot(b, a) * b_scale[:, None] * a_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    if SWAP_AB:
                        accumulator = tl.dot(b, a, acc=accumulator)
                    else:
                        accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        if not USE_TD:
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if SWAP_AB:
        accumulator = tl.trans(accumulator, (1, 0))

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A torch-only equivalent of vLLM's ``moe_align_block_size`` (its CUDA op included).

    Returns ``sorted_token_ids`` (each expert's flat indices into ``topk_ids``, padded with
    ``topk_ids.numel()`` to a multiple of ``block_size`` per expert), ``expert_ids`` (the expert of each
    block, ``-1`` past the padded total) and ``num_tokens_post_padded`` (the padded total, ``[1]``).
    """
    numel = topk_ids.numel()
    device = topk_ids.device
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = triton.cdiv(max_num_tokens_padded, block_size) * block_size
    if numel < num_experts:
        max_num_tokens_padded = min(numel * block_size, max_num_tokens_padded)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    padded_num_experts = triton.cdiv(num_experts, 32) * 32
    assert padded_num_experts < 1024, "padded_num_experts must be less than 1024"

    sorted_ids = torch.full((max_num_tokens_padded,), numel, dtype=torch.int32, device=device)
    expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32, device=device)

    # get_local_expert_id (moe_align_sum_kernels.cu:119-132): ids outside [0, num_experts) are not
    # routed; with the expert map (passed only under ignore_invalid_experts), the local id, -1 not routed.
    flat = topk_ids.reshape(-1).to(torch.int64)
    valid = (flat >= 0) & (flat < num_experts)
    local = torch.full_like(flat, -1)
    if expert_map is not None and ignore_invalid_experts:
        local[valid] = expert_map[flat[valid]].to(torch.int64)
    else:
        local[valid] = flat[valid]
    routed = local != -1

    idx = torch.arange(numel, device=device)[routed]
    eid = local[routed]
    counts = torch.bincount(eid, minlength=num_experts)[:num_experts]
    padded = ((counts + block_size - 1) // block_size) * block_size
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    cumsum[1:] = torch.cumsum(padded, dim=0)
    total = int(cumsum[-1])

    order = torch.sort(eid, stable=True).indices
    idx, eid = idx[order], eid[order]
    rank = torch.arange(idx.numel(), device=device) - (torch.cumsum(counts, 0) - counts)[eid]
    sorted_ids[cumsum[eid] + rank] = idx.to(torch.int32)

    blocks = torch.arange(total // block_size, device=device)
    expert_ids[blocks] = (torch.searchsorted(cumsum[1:], blocks * block_size, right=True)).to(torch.int32)
    num_tokens_post_pad = torch.tensor([total], dtype=torch.int32, device=device)

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(M: int, E: int) -> dict:
    """``get_default_config``'s branch for ``bfloat16`` (``fused_moe.py:1372-1416``; CUDA, no
    ``VLLM_BATCH_INVARIANT``): the tile from the batch ``M`` and the local experts ``E``."""
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 512:
        block_m = 64
    else:
        block_m = 128

    block_n = 64 if M <= 64 else 128

    block_k = 128 if M <= 64 else 64

    tokens_per_expert = M // max(E, 1)
    group_m = 16 if tokens_per_expert > 128 else 1

    num_warps = 4 if M <= 128 else 8

    if M <= 32:
        num_stages = 4
    else:
        num_stages = 3

    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "SPLIT_K": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    return config


def get_device_name_as_file_name() -> str:
    """``vllm/utils/platform_utils.py:69-74``, for the current CUDA device."""
    import re

    name = torch.cuda.get_device_name()
    name = re.sub(r"[\s/]+", "_", name)
    return name


def get_config_file_name(E: int, N: int, dtype: str | None) -> str:
    """``get_config_file_name`` (``fused_moe.py:1095-1106``) without a block shape."""
    device_name = get_device_name_as_file_name()
    # Set device_name to H200 if a device from the H200 family is detected
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}.json"


def get_moe_configs(E: int, N: int, dtype: str | None):
    """``get_moe_configs`` (``fused_moe.py:1109-1165``) on vLLM's own files (``configs/``, the
    package's configuration directory; no ``VLLM_TUNED_CONFIG_FOLDER``)."""
    import json
    import os

    json_file_name = get_config_file_name(E, N, dtype)
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            tuned_config = json.load(f)
            tuned_config.pop("triton_version", None)
            return {int(key): val for key, val in tuned_config.items()}
    return None


def try_get_optimal_moe_config(w1_shape, w2_shape, top_k: int, dtype: str | None, M: int) -> dict:
    """``try_get_optimal_moe_config`` (``fused_moe.py:1421-1450``), no override configuration:
    the tuned file's configuration for the batch size nearest ``M``, else
    ``get_default_config``."""
    E, _, N = w2_shape
    configs = get_moe_configs(E, N, dtype)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = get_default_config(M, E)
    return config


def _prepare_expert_assignment(
    topk_ids: torch.Tensor,
    config: dict,
    num_tokens: int,
    top_k_num: int,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
):
    """``_prepare_expert_assignment`` (``fused_moe.py:1549-1599``) for unquantised weights in
    device memory, with ``ignore_invalid_experts=True``."""
    naive_block_assignment = (
        expert_map is None
        and num_tokens * top_k_num * 4 <= global_num_experts
    )

    if naive_block_assignment:
        return (
            None,
            topk_ids.view(-1),
            torch.full(
                (1,),
                topk_ids.numel() * config["BLOCK_SIZE_M"],
                dtype=torch.int32,
                device=topk_ids.device,
            ),
        )

    return moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        global_num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type: tl.dtype,
):
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique,
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]
    grid = (triton.cdiv(EM, config["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), config["BLOCK_SIZE_N"]), )

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")

    fused_moe_kernel[grid](
        A,
        B,
        C,
        None,
        None,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=False,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SWAP_AB=False,
        USE_TD=False,
        **config,
    )


def _compute_type(dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.float32:
        return tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {dtype}")


def fused_moe_gemm1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """The first expert GEMM of vLLM's ``fused_experts_impl``: ``[M, top_k, N]``, row ``(t, j)`` the
    token ``t`` times expert ``topk_ids[t, j]``'s ``w1``."""
    assert hidden_states.size(1) == w1.size(2), (
        f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
    )
    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    M = num_tokens

    # vLLM looks the configuration up by w2's shape, [E, K, N // 2] after the gate-and-up
    # activation (fused_moe.py:1736-1745); the config dtype of bfloat16 is None.
    config = try_get_optimal_moe_config(w1.size(), (E, hidden_states.size(1), N // 2), top_k_num,
                                        None, M)

    intermediate_cache1 = torch.empty(
        (M, top_k_num, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    compute_type = _compute_type(hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids, config, num_tokens, top_k_num, global_num_experts, expert_map,
    )

    # dispatch_fused_moe_kernel (fused_moe.py:936-938)
    assert topk_weights is not None or not apply_router_weight_on_input
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    invoke_fused_moe_triton_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        top_k_num,
        config,
        compute_type=compute_type,
    )
    return intermediate_cache1


def fused_moe_gemm2(
    intermediate_cache2: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """The second expert GEMM of vLLM's ``fused_experts_impl``: ``[M, top_k, K]``, row ``(t, j)``
    row ``t · top_k + j`` of ``intermediate_cache2`` (``[M · top_k, N / 2]``) times expert
    ``topk_ids[t, j]``'s ``w2`` (``[E, K, N / 2]``), times ``topk_weights[t, j]`` unless the
    weights were applied on the input."""
    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert intermediate_cache2.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = topk_ids.size(0)
    E, K, _ = w2.size()
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    assert intermediate_cache2.size() == (num_tokens * top_k_num, w2.size(2))

    M = num_tokens

    # the configuration of the call (fused_moe.py:1736-1745), looked up by w2's shape
    config = try_get_optimal_moe_config((E, 2 * w2.size(2), K), w2.size(), top_k_num, None, M)

    intermediate_cache3 = torch.empty(
        (M, top_k_num, K),
        device=intermediate_cache2.device,
        dtype=intermediate_cache2.dtype,
    )

    compute_type = _compute_type(intermediate_cache2.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids, config, num_tokens, top_k_num, global_num_experts, expert_map,
    )

    if expert_map is not None:
        intermediate_cache3.zero_()

    mul_routed_weight = not apply_router_weight_on_input

    # dispatch_fused_moe_kernel (fused_moe.py:936-938)
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    invoke_fused_moe_triton_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        1,
        config,
        compute_type=compute_type,
    )
    return intermediate_cache3


def reference(hidden_states, w1, topk_weights, topk_ids, mul_routed_weight, expert_map=None,
              second=False):
    """Row ``(t, j)``: ``x @ w[e]ᵀ`` in ``float64``, ``e`` the local expert of ``topk_ids[t, j]`` and
    ``x`` row ``t`` of ``hidden_states`` (the first GEMM) or row ``t · top_k + j`` of
    ``intermediate_cache2`` (``second``); ``NaN`` where the row is not routed to a local expert."""
    M, top_k = topk_ids.shape
    out = torch.full((M, top_k, w1.size(1)), float("nan"), dtype=torch.float64, device=hidden_states.device)
    local = topk_ids.long() if expert_map is None else expert_map[topk_ids.long()].long()
    a = hidden_states.double()
    for e in range(w1.size(0)):
        t, j = torch.nonzero(local == e, as_tuple=True)
        if t.numel():
            out[t, j] = (a[t * top_k + j] if second else a[t]) @ w1[e].double().t()
    if mul_routed_weight:
        out *= topk_weights.double()[:, :, None]
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (M, N, K, E, top_k, apply_router_weight_on_input, local experts or None), from vLLM's
    # tests/kernels/moe/test_moe.py:169-173, 247-253 (m 1..40000, n 128..2048 with N = 2n, k 128..1024 and
    # 511, E 8..256, top_k 1, 2, 6) and model configs at TP8: Mixtral-8x7B (E 8, top 2, hidden 4096,
    # N 2·1792), DeepSeek-V3 (E 256, top 8, hidden 7168, N 2·256), Qwen3-30B-A3B (E 128, top 8,
    # hidden 2048, N 2·768 at TP1), Llama-4-Scout (E 16, top 1, hidden 5120, router weight on input).
    # Each case runs the first GEMM (w1 [E, N, K]) and the second (w2 [E, K, N / 2] on an
    # intermediate_cache2 of [M · top_k, N / 2]).
    cases = [
        (1, 256, 128, 8, 2, False, None),
        (33, 4096, 128, 64, 6, False, None),
        (100, 1022, 511, 8, 2, False, None),
        (222, 512, 7168, 256, 8, False, None),
        (1024, 1536, 2048, 128, 8, False, None),
        (2048, 3584, 4096, 8, 2, False, None),
        (64, 2048, 5120, 16, 1, True, None),
        (300, 1024, 1024, 64, 6, False, 16),
        (2, 512, 7168, 256, 8, False, None),
    ]
    for M, N, K, E, top_k, on_input, local_experts in cases:
        a = (torch.randn((M, K), generator=gen) / 10).to(device=DEVICE, dtype=torch.bfloat16)
        score = torch.randn((M, E), generator=gen).to(DEVICE)
        topk_weights, topk_ids = torch.topk(torch.softmax(score, dim=-1), top_k, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
        expert_map = None
        E_local = E if local_experts is None else local_experts
        w1 = (torch.randn((E_local, N, K), generator=gen) / 10).to(device=DEVICE, dtype=torch.bfloat16)
        w2 = (torch.randn((E_local, K, N // 2), generator=gen) / 10).to(device=DEVICE, dtype=torch.bfloat16)
        if local_experts is not None:
            # expert parallelism: this rank holds experts 0 .. local_experts - 1 of E
            expert_map = torch.full((E,), -1, dtype=torch.int32, device=DEVICE)
            expert_map[:local_experts] = torch.arange(local_experts, dtype=torch.int32, device=DEVICE)
        g = E if expert_map is not None else -1
        x2 = (torch.randn((M * top_k, N // 2), generator=gen) / 10).to(device=DEVICE, dtype=torch.bfloat16)
        for second in (False, True):
            if second:
                out = fused_moe_gemm2(x2, w2, topk_weights, topk_ids, on_input,
                                      global_num_experts=g, expert_map=expert_map)
                ref = reference(x2, w2, topk_weights, topk_ids, not on_input, expert_map, second=True)
            else:
                out = fused_moe_gemm1(a, w1, topk_weights, topk_ids, on_input,
                                      global_num_experts=g, expert_map=expert_map)
                ref = reference(a, w1, topk_weights, topk_ids, on_input, expert_map)
            routed = ~torch.isnan(ref)
            err = (out.double() - ref).abs()[routed]
            rel = (err / (ref.abs()[routed] + ref.abs()[routed].mean())).max().item()
            print(f"GEMM {2 if second else 1} M {M} N {N} K {K} E {E} top_k {top_k} "
                  f"router weight on input {on_input} local experts {local_experts}: "
                  f"max error / (|ref| + mean |ref|) {rel:.2e}, max abs error {err.max().item():.2e}")
            assert out.shape == (M, top_k, K if second else N) and out.dtype == torch.bfloat16
            assert rel < 1e-2, rel
    # the alignment helper on vLLM's documented example (moe_align_block_size.py:112-127, ids 1..4 of 5)
    ids = torch.tensor([[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], dtype=torch.int32, device=DEVICE)
    s, e, n = moe_align_block_size(ids, 4, 5)
    assert s[:int(n)].tolist() == [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12], s
    assert e[:int(n) // 4].tolist() == [1, 2, 3, 4] and int(n) == 16 and (e[4:] == -1).all()
    print("✅ fused MoE expert GEMMs match the float64 reference")
