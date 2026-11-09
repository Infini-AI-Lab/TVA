"""
Prefill attention over packed variable-length sequences
=======================================================

Attention for prompts packed one after another along the token dimension (no padding, page
size 1): sequence ``b`` holds the tokens ``b_start_loc[b] ... b_start_loc[b] + b_seq_len[b] - 1``
of ``q``, ``k``, ``v`` and the output. Each query attends to the keys of its own sequence, with
the causal mask (key position at most the query's) when ``IS_CAUSAL``. One program handles one
sequence, one query head and one block of ``BLOCK_M`` queries (grid ``(batch, num_heads,
cdiv(max_input_len, BLOCK))``); it runs an online softmax over blocks of ``BLOCK_N`` keys, with
``tl.dot`` in ``float32`` and the probabilities cast to the values' dtype before the second
``tl.dot``. Query head ``h`` reads key/value head ``h // kv_group_num`` (grouped-query
attention).

Layouts (last dimension contiguous):

* ``q``: ``[total_tokens, num_heads, head_dim]``, ``k`` and ``v``:
  ``[total_tokens, num_kv_heads, head_dim]``, ``bfloat16``, token and head strides free;
* ``b_start_loc``, ``b_seq_len``: ``[batch]``, ``int32``;
* ``o``: ``[total_tokens, num_heads, head_dim]``, ``bfloat16``, allocated by the host.

``head_dim`` (``Lk``) need not be a power of two: ``BLOCK_DMODEL`` is the power of two at or
above it, and the extra columns are masked.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/attention/prefill_attention.py``:

* ``_fwd_kernel``: lines 34-167, copied verbatim;
* ``context_attention_fwd`` (the host wrapper): lines 170-219, adapted below.

Changes from the source:

* ``prefill_attention.py:25-31``: ``from sglang.srt.utils import is_cuda, is_hip`` and the
  module's device globals are replaced by locals of the host with their values on an H200:
  ``_is_cuda = True``, ``_is_hip = False``, ``CUDA_CAPABILITY = (9, 0)``: the file has no
  module-level assignments.
* ``prefill_attention.py:170-219``: the host allocates the output ``o`` fresh
  (``torch.empty_like(q)``, as the vision encoder's caller does,
  ``python/sglang/srt/layers/attention/vision.py:508``) and returns it, where the source's
  wrapper takes it from the caller. No autotuning in the source: the wrapper's own choice is
  kept, ``BLOCK = 128`` (capability 9 > 8), ``num_warps = 4 if Lk <= 64 else 8``,
  ``num_stages = 1``.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]

    mask_d = offs_d < Lk

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    end_n = (
        cur_batch_seq_len
        if not IS_CAUSAL
        else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
    )
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        if IS_CAUSAL:
            qk += tl.where(
                (start_n + offs_n[None, :] < cur_batch_seq_len)
                & (offs_m[:, None] >= (start_n + offs_n[None, :])),
                0,
                float("-inf"),
            )
        else:
            qk += tl.where(
                (start_n + offs_n[None, :]) < cur_batch_seq_len, 0, float("-inf")
            )

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])
    )


def context_attention_fwd(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal=True, sm_scale=None
):
    """
    q, k, v: [b * s, head, head_dim]
    b_start_loc: [b]
    b_seq_len: [b]
    out: [b * s, head, head_dim], allocated here and returned
    sm_scale: softmax scale, defaults to 1/sqrt(head_dim)
    """
    o = torch.empty_like(q)
    # prefill_attention.py:27-31, on an H200
    _is_cuda = True
    _is_hip = False
    if _is_cuda or _is_hip:
        CUDA_CAPABILITY = (9, 0)  # torch.cuda.get_device_capability()
    if (_is_cuda or _is_hip) and CUDA_CAPABILITY[0] > 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )
    return o


def reference(q, k, v, seq_lens, is_causal=True):
    """Attention in float64, sequence by sequence."""
    num_heads, num_kv_heads, head_dim = q.shape[1], k.shape[1], q.shape[2]
    group = num_heads // num_kv_heads
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    start = 0
    for n in seq_lens:
        qs = q[start:start + n].double().transpose(0, 1)  # [H, n, D]
        ks = k[start:start + n].double().repeat_interleave(group, dim=1).transpose(0, 1)
        vs = v[start:start + n].double().repeat_interleave(group, dim=1).transpose(0, 1)
        s = qs @ ks.transpose(1, 2) * head_dim**-0.5
        if is_causal:
            s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
        out[start:start + n] = (torch.softmax(s, dim=-1) @ vs).transpose(0, 1)
        start += n
    return out


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # test/registered/attention/test_triton_attention_kernels.py:640-680 (4 heads, head dims
    # 128, 96, 80 and 13, sequence lengths 8 and 12), and model shapes: Llama-3-8B (32 heads,
    # 8 KV heads, 128), Qwen2.5-7B (28 / 4, 128), a vision encoder (16 / 16, 80).
    for num_heads, num_kv_heads, head_dim, seq_lens in [
        (4, 4, 128, [8, 12]),
        (4, 4, 96, [8, 12]),
        (4, 4, 80, [8, 12]),
        (4, 4, 13, [8, 12]),
        (32, 8, 128, [1, 127, 128, 129, 1000]),
        (28, 4, 128, [255, 3, 2048]),
        (16, 16, 80, [1024, 64, 1]),
        (8, 1, 64, [300, 513]),
    ]:
        total = sum(seq_lens)
        starts = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
        q = torch.randn((total, num_heads, head_dim), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn((total, num_kv_heads, head_dim), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn((total, num_kv_heads, head_dim), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        b_start_loc = torch.tensor(starts, dtype=torch.int32, device=DEVICE)
        b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
        o = context_attention_fwd(q, k, v, b_start_loc, b_seq_len, max(seq_lens))
        ref = reference(q, k, v, seq_lens)
        err = (o.double() - ref).abs().max().item()
        print(f"heads {num_heads}/{num_kv_heads} head_dim {head_dim} seq_lens {seq_lens}: "
              f"max abs error {err:.2e}")
        assert err < 2e-2, err
    print("✅ context_attention_fwd (causal) matches the float64 reference")
