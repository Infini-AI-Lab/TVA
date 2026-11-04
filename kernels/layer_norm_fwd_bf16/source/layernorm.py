"""
LayerNorm (or RMSNorm) forward, with a residual
===============================================

Each row of ``x`` (with the row of ``res`` added, when there is a residual) is normalised:
``s = x + res``, ``mean = sum(s) / D``, ``rstd = 1 / sqrt(sum((s - mean)^2) / D + eps)`` and
``y = (s - mean) * rstd * w + b``, all in ``float32`` (RMSNorm, ``IS_RMS_NORM``, skips the mean).
The sum ``s`` is written out (``res_out``, the next layer's residual), and ``mean`` and ``rstd``
are saved for the backward pass. One program handles ``BT`` consecutive rows; the row length ``D``
(at most 512 on this path) is padded to the power of two ``BD``. Row ``t`` uses the weights of group
``t % G`` (``G = 1`` for LayerNorm; fla's GroupNorm uses ``G > 1``).

Layouts (contiguous):

* ``x``: ``[T, D]``, ``bfloat16``; ``residual`` (optional): ``[T, D]``;
* ``weight``, ``bias`` (each optional): ``[G * D]``;
* ``y``: ``[T, D]``, the dtype of ``x`` (or ``out_dtype``);
* ``res_out``: ``[T, D]``, the residual's dtype, allocated when there is a residual (or a
  ``residual_dtype`` other than the dtype of ``x``);
* ``mean`` (LayerNorm only), ``rstd``: ``[T]``, ``float32``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``. Here, the kernel is lines 50-114.

* ``layer_norm_fwd_kernel``: ``fla/modules/norm/layernorm.py:38-111`` (the kernel's ``def`` at
  line 48).
* ``layer_norm_fwd`` (the host): adapted from ``fla/modules/norm/layernorm.py:388-471``.

Changes from the source:

* ``fla/modules/norm/layernorm.py:38-46``: the decorator ``@triton.autotune(...)`` (over ``BT`` in
  ``[32, 64, 128]`` and ``num_warps`` in ``[2, 4, 8]``, key
  ``['D', 'NB', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM']``) is removed.
  ``@triton.jit(do_not_specialize=['T'])`` and the kernel's signature and body are verbatim.
* ``fla/modules/norm/layernorm.py:388-471`` (``layer_norm_fwd``): the decorator ``@dispatch``
  (fla's backend registry, which falls back to this function on NVIDIA GPUs) is removed; the launch
  passes one configuration of the autotune list, ``BT=32`` and ``num_warps=4``, as explicit
  arguments, so the grid is ``(triton.cdiv(T, 32),)``; the branch for ``D > 512`` (and for
  ``D == 512`` on Intel GPUs), which launches another kernel (``layer_norm_fwd_kernel_row``,
  ``layernorm.py:114-179``), is replaced by an assert. The rest (the asserts, the allocations,
  ``BD``, ``NB``, the flags and the launch's keywords) is the wrapper's.
* ``reference`` and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def layer_norm_fwd_kernel(
    x,
    y,
    w,
    b,
    res,
    res_out,
    mean,
    rstd,
    eps,
    T,
    G: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    o_g = o_t % G
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_t = o_t < T
    m_x = m_t[:, None] & m_d[None, :]

    p_x = x + o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = res + o_t[:, None] * D + o_d[None, :]
        b_x += tl.load(p_res, mask=m_x, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = res_out + o_t[:, None] * D + o_d[None, :]
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
        b_w = tl.load(w + o_g[:, None] * D + o_d[None, :], mask=m_d[None, :]).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_g[:, None] * D + o_d[None, :], mask=m_d[None, :]).to(tl.float32)
    b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    p_y = y + o_t[:, None] * D + o_d[None, :]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
    num_groups: int = 1,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D, G = *x.shape, num_groups
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (G * D,)
    if bias is not None:
        assert bias.shape == (G * D,)

    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        res_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        res_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # fla launches another kernel (layer_norm_fwd_kernel_row) for D > 512 (and D == 512 on Intel).
    assert D <= 512
    NB = triton.cdiv(T, 2048)

    def grid(meta):
        return (triton.cdiv(T, meta['BT']),)

    # One configuration of the autotune list at fla/modules/norm/layernorm.py:39-43: BT = 32, num_warps = 4.
    layer_norm_fwd_kernel[grid](
        x=x,
        y=y,
        w=weight,
        b=bias,
        res=residual,
        res_out=res_out,
        mean=mean,
        rstd=rstd,
        eps=eps,
        T=T,
        G=G,
        D=D,
        BD=BD,
        NB=NB,
        IS_RMS_NORM=is_rms_norm,
        HAS_RESIDUAL=residual is not None,
        STORE_RESIDUAL_OUT=res_out is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        BT=32,
        num_warps=4,
    )
    # res_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, res_out if res_out is not None else x


def reference(x, weight, bias, residual, eps):
    """LayerNorm of x + residual, in float64 from the float32 sum (as the kernel forms it)."""
    s32 = x.float() + residual.float()
    s = s32.double()
    mean = s.mean(-1)
    var = ((s - mean[:, None]) ** 2).mean(-1)
    rstd = 1.0 / torch.sqrt(var + eps)
    y = (s - mean[:, None]) * rstd[:, None] * weight.double() + bias.double()
    return y, mean, rstd, s32.to(residual.dtype)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    eps = 1e-5
    # (T, D): fla's tests (tests/modules/test_layernorm.py:20-25: B = 2, H = 2, T = 512, D in 50, 64,
    # 128, so 2048 rows), D = 512 (the largest on this path), one row, D = 1, rows not a multiple of
    # BT, and a long batch.
    for T, D in [
        (1, 1),
        (1, 64),
        (31, 50),
        (2048, 50),
        (2048, 64),
        (2048, 128),
        (33, 512),
        (1000, 300),
        (65536, 256),
    ]:
        x = torch.randn((T, D), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        res = torch.randn((T, D), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        w = (1 + 0.1 * torch.randn((D,), generator=gen)).to(device=DEVICE, dtype=torch.bfloat16)
        b = (0.1 * torch.randn((D,), generator=gen)).to(device=DEVICE, dtype=torch.bfloat16)
        y, mean, rstd, res_out = layer_norm_fwd(x, w, b, eps=eps, residual=res)
        y_ref, mean_ref, rstd_ref, res_out_ref = reference(x, w, b, res, eps)
        assert y.dtype == res_out.dtype == torch.bfloat16 and mean.dtype == rstd.dtype == torch.float32
        assert torch.equal(res_out, res_out_ref)
        err_y = ((y.double() - y_ref).abs() / (1 + y_ref.abs())).max().item()
        err_m = (mean.double() - mean_ref).abs().max().item()
        err_r = ((rstd.double() - rstd_ref).abs() / rstd_ref).max().item()
        print(f"T {T} D {D}: res_out exact, y max error {err_y:.2e} (abs / (1 + |ref|)), "
              f"mean max abs error {err_m:.2e}, rstd max rel error {err_r:.2e}")
        # bfloat16 rounds to 2^-9 relative.
        assert err_y < 8e-3, err_y
        assert err_m < 1e-5, err_m
        assert err_r < 1e-4, err_r
    print("layer_norm_fwd matches the float64 reference")
