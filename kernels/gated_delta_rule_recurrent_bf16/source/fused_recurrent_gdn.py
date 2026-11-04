"""
Gated delta rule, fused recurrent forward
=========================================

The gated delta rule run token by token, as gated DeltaNet decodes (and as its short prefills
run). Each (sequence, value head) carries a state ``S`` of ``K x V`` floats; for each token
``t`` of the sequence, with ``q_t``, ``k_t`` of the key head ``h // (HV // H)``, ``v_t`` and
``beta_t`` of the value head ``h``:

    S   <- S * exp(g_t)                          (the decay; also per key or value channel with gk, gv)
    v'  <- beta_t * (v_t - S^T k_t)              (the delta rule's correction)
    S   <- S + k_t v'^T
    o_t <- S^T (scale * q_t)

starting from the initial state (or zeros) and saving the final state. Options of the kernel:
``q`` and ``k`` L2-normalised in the kernel, ``beta`` passed through a sigmoid (and doubled), the
decay computed in the kernel from a raw gate (``-exp(A_log) * softplus(g + dt_bias)``), and the state
stored as ``V x K`` (``STATE_V_FIRST``). The state and all arithmetic are ``float32``.

Layouts (contiguous):

* ``q``, ``k``: ``[B, T, H, K]``, ``bfloat16``; ``v``: ``[B, T, HV, V]``, ``bfloat16``, ``HV`` a
  multiple of ``H`` (grouped value attention when ``HV > H``);
* ``g``: ``[B, T, HV]``, ``float32``, the log decay; ``gk``: ``[B, T, HV, K]``; ``gv``: ``[B, T, HV, V]``;
* ``beta``: ``[B, T, HV]`` (one per head) or ``[B, T, HV, V]``, ``bfloat16``; when ``beta`` is not
  given, the wrapper makes it ``torch.ones_like(q[..., 0])``;
* ``initial_state``: ``[N, HV, K, V]``, ``float32`` (``[N, HV, V, K]`` with ``state_v_first``);
* ``o``: ``[B, T, HV, V]``, of ``v``'s dtype; ``final_state``: like ``initial_state``, ``float32``,
  allocated fresh by the wrapper;
* variable-length batches: ``B = 1``, ``cu_seqlens``: ``[N + 1]`` (``int64``), the sequences'
  boundaries.

One program per (block of ``BV`` value channels, sequence, value head): ``grid = (NV * N * HV,)``,
``BK = next_power_of_2(K)``, ``BV = min(8, next_power_of_2(V))`` (``next_power_of_2(V)`` with ``gv``).

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``:

* ``fused_recurrent_gated_delta_rule_fwd_kernel``: ``fla/ops/gated_delta_rule/fused_recurrent.py:29-181``;
* ``exp``: ``fla/ops/utils/op.py:44-45`` (the default branch of ``op.py:32-53``);
* ``fused_recurrent_gated_delta_rule_fwd``: ``fla/ops/gated_delta_rule/fused_recurrent.py:184-250``;
* ``fused_recurrent_gated_delta_rule``: ``fla/ops/gated_delta_rule/fused_recurrent.py:309-475``,
  through ``FusedRecurrentFunction``: ``fused_recurrent.py:253-306``.

Changes from the source:

* ``fla/ops/gated_delta_rule/fused_recurrent.py:19-28``: the decorator ``@triton.heuristics({'USE_G':
  ..., 'USE_GK': ..., 'USE_GV': ..., 'USE_INITIAL_STATE': ..., 'STORE_FINAL_STATE': ...,
  'IS_VARLEN': ..., 'USE_GATE_IN_KERNEL': ..., 'HAS_DT_BIAS': ...})`` removed;
  ``@triton.jit(do_not_specialize=['T'])`` kept. The kernel's signature and body are unchanged. The
  source has no autotuning: the wrapper launches with ``num_warps=1, num_stages=3``.
* ``fla/ops/utils/op.py:44-45``: the helper ``exp`` copied verbatim. ``op.py`` defines it by the
  environment variable ``FLA_USE_FAST_OPS``: unset (the default, so on an H200 too), it is
  ``tl.exp`` of the argument in ``float32``; with ``FLA_USE_FAST_OPS=1`` it would be libdevice's
  ``fast_expf`` (``op.py:34``), which this file does not take.
* ``fla/ops/utils/softplus.py:111-118``: the helper ``softplus`` (on NVIDIA GPUs, so on an H200,
  ``softplus_nv``, ``softplus.py:68-81``: inline PTX built from the module-level ``tl.constexpr``
  values ``s_softplus``, ``s_constraints``, ``NUM_REG``, ``softplus.py:17-35, 57-65``) is not
  copied, nor are those values and the functions that build them. The kernel names ``softplus``
  only inside ``if USE_GATE_IN_KERNEL:`` (``fused_recurrent.py:132-137``), a compile-time branch
  the proved configuration does not take, where Triton does not resolve names; a launch with
  ``A_log`` given would need it.
* ``fla/ops/gated_delta_rule/fused_recurrent.py:184-250``: the host
  ``fused_recurrent_gated_delta_rule_fwd`` passes, as the removed heuristics did, ``USE_G``,
  ``USE_GK``, ``USE_GV``, ``USE_INITIAL_STATE``, ``STORE_FINAL_STATE``, ``IS_VARLEN``,
  ``USE_GATE_IN_KERNEL`` and ``HAS_DT_BIAS`` (each ``True`` when its tensor, ``g``, ``gk``, ``gv``,
  ``h0``, ``ht``, ``cu_seqlens``, ``A_log``, ``dt_bias``, is not ``None``). Otherwise its argument
  handling, allocation, grid, arguments and launch options are the source's.
* ``fla/ops/gated_delta_rule/fused_recurrent.py:253-306``: ``FusedRecurrentFunction`` (an autograd
  function whose backward raises) is not kept: the entry point calls the host directly, after the
  ``.contiguous()`` calls on the tensor arguments that its ``@input_guard``
  (``fla/utils/decorators.py:97-157``) made.
* ``fla/ops/gated_delta_rule/fused_recurrent.py:309-475``: the entry point
  ``fused_recurrent_gated_delta_rule`` is kept with its defaults and checks (its docstring
  shortened, and the ``DeprecationWarning`` it issues for the old keyword ``transpose_state_layout``,
  ``fused_recurrent.py:423-427``, dropped with the ``warnings`` import).
* The reference (``float64``) and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))


@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, (pid // NV).to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if STATE_V_FIRST:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if STATE_V_FIRST:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = h0 + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta = b_beta * 2

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            if USE_GATE_IN_KERNEL:
                b_A = tl.load(A_log + i_hv).to(tl.float32)
                if HAS_DT_BIAS:
                    b_g = b_g + tl.load(dt_bias + i_hv).to(tl.float32)
                b_g = -exp(b_A) * softplus(b_g)
            b_h *= exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h *= exp(b_gk[None, :])
            else:
                b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h *= exp(b_gv[:, None])
            else:
                b_h *= exp(b_gv[None, :])

        if STATE_V_FIRST:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_v += HV*V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV*K
        if USE_GV:
            p_gv += HV*V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV*V

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    if output_final_state:
        if state_v_first:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = (NV * N * HV,)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=g is not None,  # the removed @triton.heuristics, USE_G to HAS_DT_BIAS
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_VARLEN=cu_seqlens is not None,
        USE_GATE_IN_KERNEL=A_log is not None,
        HAS_DT_BIAS=dt_bias is not None,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
        STATE_V_FIRST=state_v_first,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA (Grouped Value Attention) is applied if `HV > H`, where `HV` must be divisible by `H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
            When `use_gate_in_kernel=False` (default), `g` must be in log space (pre-computed decay).
            When `use_gate_in_kernel=True`, `g` is the raw pre-activation input; the kernel fuses
            `-exp(A_log) * softplus(g + dt_bias)` internally per step.
        gk (torch.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space GDN decay internally.
            When `True`, `g` is the raw input and `A_log` must be provided; the kernel fuses
            gate activation into the recurrence. Default: `False`.
        A_log (Optional[torch.Tensor]):
            Decay parameter of shape `[HV]`. Required when `use_gate_in_kernel=True`.
        dt_bias (Optional[torch.Tensor]):
            Bias added to `g` before activation, of shape `[HV]`.
            Only used when `use_gate_in_kernel=True`.
        use_beta_sigmoid_in_kernel (Optional[bool]):
            Whether to apply `torch.sigmoid(beta)` inside the kernel.
            - If `True`, the passed `beta` acts as the raw beta logits.
            - If `False`, `beta` is expected to already be in post-sigmoid space.
            Default: `False`.
        allow_neg_eigval (Optional[bool]):
            Whether to allow negative eigenvalues by scaling `beta` to `[0, 2)`.
            Only takes effect together with `use_beta_sigmoid_in_kernel=True`, in which case
            the kernel computes `2 * sigmoid(beta)` instead of `sigmoid(beta)`. Default: `False`.
        state_v_first (Optional[bool]):
            Store the recurrent state in V-first ``[V, K]`` layout instead of the default ``[K, V]``. Default: ``False``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    """
    if 'transpose_state_layout' in kwargs:
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
        state_v_first = kwargs.pop('transpose_state_layout')

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
        if g is None:
            raise ValueError("`g` (raw pre-activation) must be provided when `use_gate_in_kernel=True`.")
    else:
        A_log = None
        dt_bias = None
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    # the source's FusedRecurrentFunction.forward, under @input_guard: tensor arguments made contiguous
    q, k, v, g, gk, gv, beta, A_log, dt_bias, initial_state, cu_seqlens = (
        t.contiguous() if isinstance(t, torch.Tensor) else t
        for t in (q, k, v, g, gk, gv, beta, A_log, dt_bias, initial_state, cu_seqlens)
    )
    o, final_state = fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state


def reference(q, k, v, g, beta, scale, initial_state, cu_seqlens=None, use_qk_l2norm=False):
    """The gated delta rule token by token in float64; returns the outputs and the final states."""
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    if cu_seqlens is None:
        bounds = [(b, 0, T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = [(0, cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
    o = torch.zeros(v.shape, dtype=torch.float64, device=v.device)
    hts = []
    for n, (b, bos, eos) in enumerate(bounds):
        S = initial_state[n].double().clone()  # [HV, K, V]
        for t in range(bos, eos):
            qt = q[b, t].double().repeat_interleave(G, dim=0)  # [HV, K]
            kt = k[b, t].double().repeat_interleave(G, dim=0)
            if use_qk_l2norm:
                qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            qt = qt * scale
            S = S * torch.exp(g[b, t].double())[:, None, None]
            vt = beta[b, t].double()[:, None] * (v[b, t].double() - torch.einsum("hkv,hk->hv", S, kt))
            S = S + kt[:, :, None] * vt[:, None, :]
            o[b, t] = torch.einsum("hkv,hk->hv", S, qt)
        hts.append(S)
    return o, torch.stack(hts)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=gen).to(device=DEVICE, dtype=dtype)

    cases = [
        # (B, T, H, HV, K, V, cu_seqlens, beta given): fla's tests/ops/test_gdn.py test_fused_recurrent
        # shapes, fla's GatedDeltaNet defaults (6 heads, K = 256, V = 512), Qwen3-Next's heads (16 key
        # heads, 32 value heads, head dims 128), decoding (T = 1)
        (1, 63, 1, 1, 64, 64, None, True),
        (2, 500, 4, 4, 60, 60, None, True),
        (2, 300, 2, 8, 128, 128, None, True),
        (3, 256, 2, 2, 128, 128, None, False),
        (4, 128, 3, 3, 128, 128, None, False),
        (2, 64, 6, 6, 256, 512, None, True),
        (64, 1, 16, 32, 128, 128, None, True),
        (8, 2, 16, 32, 128, 128, None, True),
        (1, 300, 4, 8, 64, 64, [0, 1, 15, 100, 300], True),
        (1, 200, 4, 4, 128, 128, [0, 7, 64, 65, 200], False),
        (1, 256, 16, 32, 128, 128, [0, 1, 2, 130, 256], True),
    ]
    for B, T, H, HV, K, V, cu, beta_given in cases:
        cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.long, device=DEVICE)
        N = B if cu is None else len(cu) - 1
        v = randn(B, T, HV, V)
        g = torch.nn.functional.logsigmoid(torch.rand((B, T, HV), generator=gen)).to(DEVICE)
        beta = torch.rand((B, T, HV), generator=gen).sigmoid().to(DEVICE, torch.bfloat16) if beta_given else None
        h0 = randn(N, HV, K, V, dtype=torch.float32)
        beta_ref = beta if beta_given else torch.ones((B, T, HV), device=DEVICE)
        for l2 in (True, False):
            # with the in-kernel L2 norm (as fla's layers call it), raw q and k; without it, unit q and k
            q, k = randn(B, T, H, K), randn(B, T, H, K)
            if not l2:
                q, k = (torch.nn.functional.normalize(x.float(), dim=-1).bfloat16() for x in (q, k))
            o, ht = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=h0, output_final_state=True,
                                                     use_qk_l2norm_in_kernel=l2, cu_seqlens=cu_seqlens)
            ref_o, ref_ht = reference(q, k, v, g, beta_ref, K ** -0.5, h0, cu_seqlens=cu_seqlens, use_qk_l2norm=l2)
            assert o.dtype == torch.bfloat16 and ht.dtype == torch.float32 and ht.shape == (N, HV, K, V)
            assert ht.data_ptr() != h0.data_ptr()
            err_o = ((o.double() - ref_o).abs() / (1 + ref_o.abs())).max().item()
            err_ht = ((ht.double() - ref_ht).abs() / (1 + ref_ht.abs())).max().item()
            print(f"B {B} T {T} H {H} HV {HV} K {K} V {V} cu_seqlens {cu} beta {'given' if beta_given else 'default'} "
                  f"qk l2norm {l2}: max error o {err_o:.2e}, final state {err_ht:.2e} (relative to 1 + |ref|)")
            assert err_o < 1e-2 and err_ht < 1e-4, (err_o, err_ht)
    print("✅ fused recurrent gated delta rule matches the float64 reference")
