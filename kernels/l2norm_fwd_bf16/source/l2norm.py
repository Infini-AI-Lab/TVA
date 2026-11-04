"""
L2 normalisation, forward
=========================

Each row of ``x`` is divided by its L2 norm: ``y[t, :] = x[t, :] * rstd[t]`` with
``rstd[t] = 1 / sqrt(sum_d x[t, d]^2 + eps)``, the sum computed in ``float32``. One program
normalises ``BT`` consecutive rows; the row length ``D`` (at most 512 on this path) is padded to the
power of two ``BD``. fla uses it on the queries and keys of its delta-rule layers (one row per
token and head, ``D`` the head dimension).

Layouts (contiguous):

* ``x``: ``[..., D]``, viewed as ``[T, D]`` (row stride ``D``), ``bfloat16``;
* ``y``: ``[T, D]``, the dtype of ``x`` (or ``output_dtype``), returned with the shape of ``x``;
* ``rstd``: ``[T]``, ``float32``, returned with the shape of ``x`` without its last dimension.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``. Here, the kernel is lines 44-60.

* ``l2norm_fwd_kernel``: ``fla/modules/norm/l2norm.py:63-84`` (the kernel's ``def`` at line 69).
* ``l2norm_fwd`` (the host): adapted from ``fla/modules/norm/l2norm.py:111-139``.

Changes from the source:

* ``fla/modules/norm/l2norm.py:63-67``: the decorator ``@fla_cache_autotune(...)`` (fla's
  autotuner, over ``num_warps`` in ``[1, 2, 4, 8, 16]`` and ``BT`` in ``BT_LIST``, key
  ``["D", "NB"]``) is removed. ``@triton.jit(do_not_specialize=["T"])`` and the kernel's
  signature and body are verbatim.
* ``fla/modules/norm/l2norm.py:111-139`` (``l2norm_fwd``): the decorator ``@dispatch`` (fla's
  backend registry, which falls back to this function on NVIDIA GPUs) is removed; the launch
  passes one configuration of the autotune list, ``BT=16`` and ``num_warps=4``, as explicit
  arguments, so the grid is ``(triton.cdiv(T, 16),)``; the branch for ``D > 512``, which launches
  another kernel (``l2norm_fwd_kernel_row``, ``l2norm.py:26-39``), is replaced by an assert.
  The rest (the view, the output allocation, ``BD``, ``NB`` and the launch's keywords) is the
  wrapper's.
* ``reference`` and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(x, y, rstd, eps, T, D: tl.constexpr, BD: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    p_x = x + o_t[:, None] * D + o_d[None, :]
    p_y = y + o_t[:, None] * D + o_d[None, :]
    p_rstd = rstd + o_t

    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])

    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # fla launches another kernel (l2norm_fwd_kernel_row) for D > 512.
    assert D <= 512
    # bucket token counts to limit autotuning across sequence lengths.
    NB = triton.cdiv(T, 2048 * 32)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]),)

    # One configuration of the autotune list at fla/modules/norm/l2norm.py:64: BT = 16, num_warps = 4.
    l2norm_fwd_kernel[grid](x=x, y=y, rstd=rstd, eps=eps, T=T, D=D, BD=BD, NB=NB, BT=16, num_warps=4)
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def reference(x, eps=1e-6):
    """The normalised rows and 1 / sqrt(sum x^2 + eps), in float64."""
    xd = x.double()
    rstd = 1.0 / torch.sqrt((xd * xd).sum(-1) + eps)
    return xd * rstd[..., None], rstd


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (B, T, H, D): fla's tests (tests/modules/test_l2norm.py:20-29, the D <= 512 cases) and the
    # delta-rule layers' head dims (64, 128, 256), with one row, D = 1 and a long batch.
    for B, T, H, D in [
        (1, 1, 1, 1),
        (1, 1, 1, 128),
        (1, 63, 1, 60),
        (1, 65, 2, 512),
        (2, 500, 4, 64),
        (2, 1000, 2, 100),
        (3, 1024, 4, 128),
        (2, 333, 3, 256),
        (4, 4096, 16, 128),
    ]:
        x = (torch.randn((B, T, H, D), generator=gen) * 0.5 + 0.3).to(device=DEVICE, dtype=torch.bfloat16)
        y, rstd = l2norm_fwd(x)
        y_ref, rstd_ref = reference(x)
        assert y.dtype == torch.bfloat16 and y.shape == x.shape and rstd.shape == x.shape[:-1]
        err_y = (y.double() - y_ref).abs().max().item()
        err_r = ((rstd.double() - rstd_ref).abs() / rstd_ref).max().item()
        print(f"B {B} T {T} H {H} D {D}: y max abs error {err_y:.2e}, rstd max rel error {err_r:.2e}")
        # |y| <= 1, and bfloat16 rounds to 2^-9 relative.
        assert err_y < 4e-3, err_y
        assert err_r < 1e-5, err_r
    print("l2norm_fwd matches the float64 reference")
