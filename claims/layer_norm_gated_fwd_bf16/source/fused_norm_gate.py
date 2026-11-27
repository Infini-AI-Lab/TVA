"""
Gated LayerNorm / RMSNorm forward
=================================

Each row of ``x`` is normalised in ``float32`` and multiplied by a gate: RMSNorm (``IS_RMS_NORM``)
computes ``rstd = 1 / sqrt(sum(x^2) / D + eps)`` and ``y = x * rstd * w``, LayerNorm subtracts the
row's mean first (and may add a bias ``b``); then ``y`` is multiplied by ``g * sigmoid(g)``
(``ACTIVATION`` ``"swish"`` or ``"silu"``) or by ``sigmoid(g)`` (``"sigmoid"``), ``g`` the gate
tensor of the same shape as ``x``. ``rstd`` (and LayerNorm's ``mean``) are saved for the backward
pass. One program handles ``BT`` consecutive rows; the row length ``D`` (at most 512 on this path)
is padded to the power of two ``BD``. fla's ``FusedRMSNormGated`` runs it as the output norm of its
linear-attention layers (one row per token and head, ``D`` the value head dimension).

Layouts (contiguous):

* ``x``, ``g``: ``[T, D]``, ``bfloat16``; ``residual`` (optional): ``[T, D]``;
* ``weight``, ``bias`` (each optional): ``[D]``;
* ``y``: ``[T, D]``, the dtype of ``x`` (or ``out_dtype``);
* ``residual_out``: ``[T, D]``, the residual's dtype, allocated when there is a residual (or a
  ``residual_dtype`` other than the dtype of ``x``);
* ``mean`` (LayerNorm only), ``rstd``: ``[T]``, ``float32``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``. Here, the kernel is lines
55-127.

* ``layer_norm_gated_fwd_kernel``: ``fla/modules/norm/fused_norm_gate.py:22-107`` (the kernel's
  ``def`` at line 36).
* ``layer_norm_gated_fwd`` (the host): adapted from ``fla/modules/norm/fused_norm_gate.py:447-525``.

Changes from the source:

* ``fla/modules/norm/fused_norm_gate.py:22-34``: the decorators ``@triton.heuristics({...})``
  (``STORE_RESIDUAL_OUT``, ``HAS_RESIDUAL``, ``HAS_WEIGHT``, ``HAS_BIAS``: whether
  ``residual_out``, ``residual``, ``w``, ``b`` are not ``None``) and ``@triton.autotune(...)``
  (over ``BT`` in ``[16, 32, 64]`` and ``num_warps`` in ``[4, 8, 16]``, key
  ``["D", "NB", "IS_RMS_NORM", "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_WEIGHT"]``) are removed.
  ``@triton.jit(do_not_specialize=['T'])`` and the kernel's signature and body are verbatim.
* ``fla/modules/norm/fused_norm_gate.py:447-525`` (``layer_norm_gated_fwd``): the decorator
  ``@dispatch`` (fla's backend registry, which falls back to this function on NVIDIA GPUs) is
  removed; the launch passes the four flags the removed heuristics computed, and one configuration
  of the removed autotune list, ``BT=16`` and ``num_warps=4``, as explicit arguments, so the grid
  is ``(triton.cdiv(T, 16),)``; the branch for ``D > 512``, which launches another kernel
  (``layer_norm_gated_fwd_kernel_row``, ``fused_norm_gate.py:110-187``), is replaced by an assert.
  The rest (the asserts, the allocations, ``BD``, ``NB`` and the launch's keywords) is the
  wrapper's.
* ``reference`` and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def layer_norm_gated_fwd_kernel(
    x,
    g,
    y,
    w,
    b,
    residual,
    residual_out,
    mean,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_t = o_t < T
    m_x = m_t[:, None] & m_d[None, :]

    p_x = x + o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = residual + o_t[:, None] * D + o_d[None, :]
        b_x += tl.load(p_res, mask=m_x, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = residual_out + o_t[:, None] * D + o_d[None, :]
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), mask=m_x)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = mean + o_t
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), mask=m_t)
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = rstd + o_t
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    # swish/sigmoid output gate
    p_g = g + o_t[:, None] * D + o_d[None, :]
    b_g = tl.load(p_g, mask=m_x, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == "sigmoid":
        b_y = b_y * tl.sigmoid(b_g)

    p_y = y + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)



def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)

    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # fla launches another kernel (layer_norm_gated_fwd_kernel_row) for D > 512.
    assert D <= 512
    # bucket token counts to limit autotuning across sequence lengths.
    NB = triton.cdiv(T, 2048 * 32)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]),)

    # One configuration of the autotune list at fla/modules/norm/fused_norm_gate.py:31: BT = 16,
    # num_warps = 4.
    layer_norm_gated_fwd_kernel[grid](
        x=x,
        g=g,
        y=y,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=residual_out,
        mean=mean,
        rstd=rstd,
        eps=eps,
        T=T,
        D=D,
        BD=BD,
        NB=NB,
        ACTIVATION=activation,
        IS_RMS_NORM=is_rms_norm,
        STORE_RESIDUAL_OUT=residual_out is not None,  # the removed @triton.heuristics
        HAS_RESIDUAL=residual is not None,  # the removed @triton.heuristics
        HAS_WEIGHT=weight is not None,  # the removed @triton.heuristics
        HAS_BIAS=bias is not None,  # the removed @triton.heuristics
        BT=16,
        num_warps=4,
    )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


def reference(x, g, weight, eps, activation):
    """RMSNorm of x times weight, gated by g, in float64 (no bias, no residual)."""
    xd = x.double()
    gd = g.double()
    rstd = 1.0 / torch.sqrt((xd * xd).mean(-1) + eps)
    gate = gd * torch.sigmoid(gd) if activation in ("swish", "silu") else torch.sigmoid(gd)
    y = xd * rstd[:, None] * weight.double() * gate
    return y, rstd


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (T, D, activation, eps): fla's RMSNorm-gated test on this path
    # (tests/modules/test_layernorm_gated.py:72-80: 2 x 2 heads of T tokens, its eps = 0 replaced by
    # 1e-5), GatedDeltaNet's output norm (6 heads of 512 on 4 x 4096 tokens, swish, eps 1e-6),
    # KDA's (16 heads of 128, sigmoid, eps 1e-5), one row, D = 1, rows not a multiple of BT.
    for T, D, act, eps in [
        (4, 64, "silu", 1e-5),
        (2048, 128, "sigmoid", 1e-5),
        (200, 50, "sigmoid", 1e-5),
        (98304, 512, "swish", 1e-6),
        (262144, 128, "sigmoid", 1e-5),
        (1, 1, "swish", 1e-5),
        (33, 300, "swish", 1e-5),
    ]:
        x = torch.randn((T, D), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        g = (2 * torch.randn((T, D), generator=gen)).to(device=DEVICE, dtype=torch.bfloat16)
        w = (1 + 0.1 * torch.randn((D,), generator=gen)).to(device=DEVICE, dtype=torch.bfloat16)
        y, mean, rstd, _ = layer_norm_gated_fwd(x, g, w, None, activation=act, eps=eps, is_rms_norm=True)
        y_ref, rstd_ref = reference(x, g, w, eps, act)
        assert mean is None and y.dtype == torch.bfloat16 and rstd.dtype == torch.float32
        err_y = ((y.double() - y_ref).abs() / (2 ** -20 + y_ref.abs())).max().item()
        err_r = ((rstd.double() - rstd_ref).abs() / rstd_ref).max().item()
        print(f"T {T} D {D} {act}: y max rel error {err_y:.2e}, rstd max rel error {err_r:.2e}")
        # bfloat16 rounds to 2^-9 relative.
        assert err_y < 8e-3, err_y
        assert err_r < 1e-4, err_r
    print("layer_norm_gated_fwd matches the float64 reference")
