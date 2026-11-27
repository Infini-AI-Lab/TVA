"""
L2 normalisation, backward
==========================

The gradient of ``y = x / sqrt(sum_d x^2 + eps)``: from the forward's output ``y``, its factor
``rstd = 1 / sqrt(sum_d x^2 + eps)`` and the incoming gradient ``dy``, each row of
``dx[t, :] = dy[t, :] * rstd[t] - (sum_d dy[t, d] * y[t, d]) * y[t, :] * rstd[t]``, computed in
``float32``. One program handles ``BT`` consecutive rows; the row length ``D`` (at most 512 on this
path) is padded to the power of two ``BD``. fla uses it in the backward of its delta-rule layers
(``use_qk_l2norm_in_kernel``) and of ``l2norm`` (one row per token and head, ``D`` the head
dimension).

Layouts (contiguous):

* ``y``: ``[..., D]``, viewed as ``[T, D]`` (row stride ``D``), ``bfloat16`` (or ``float32`` when the
  forward ran with ``output_dtype=torch.float32``);
* ``rstd``: ``[T]``, ``float32``;
* ``dy``: ``[T, D]``, the gradient of ``y`` (``bfloat16`` or ``float32``);
* ``dx``: ``[T, D]``, the dtype of ``y``, returned with the shape of ``y``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``. Here, the kernel is lines 51-67.

* ``l2norm_bwd_kernel``: ``fla/modules/norm/l2norm.py:87-108`` (the kernel's ``def`` at line 93).
* ``l2norm_bwd`` (the host): adapted from ``fla/modules/norm/l2norm.py:142-168``.

Changes from the source:

* ``fla/modules/norm/l2norm.py:87-91``: the decorator ``@fla_cache_autotune(...)`` (fla's
  autotuner, over ``num_warps`` in ``[1, 2, 4, 8, 16]`` and ``BT`` in ``BT_LIST``, key
  ``["D", "NB"]``) is removed. ``@triton.jit(do_not_specialize=["T"])`` and the kernel's
  signature and body are verbatim.
* ``fla/modules/norm/l2norm.py:142-168`` (``l2norm_bwd``): the decorator ``@dispatch`` (fla's
  backend registry, which falls back to this function on NVIDIA GPUs) is removed; the launch
  passes one configuration of the autotune list, ``BT=16`` and ``num_warps=4``, as explicit
  arguments, so the grid is ``(triton.cdiv(T, 16),)``; the branch for ``D > 512``, which launches
  another kernel (``l2norm_bwd_kernel_row``, ``l2norm.py:42-60``), is replaced by an assert.
  The rest (the views, the output allocation, ``BD``, ``NB`` and the launch's keywords) is the
  wrapper's.
* The other kernels and functions of the file (``l2norm_fwd_kernel_row``,
  ``l2norm_bwd_kernel_row``, ``l2norm_fwd_kernel``, ``l2norm_fwd``, ``L2NormFunction``, ``l2norm``,
  ``L2Norm``) are left out.
* ``reference`` and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def l2norm_bwd_kernel(y, rstd, dy, dx, eps, T, D: tl.constexpr, BD: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    p_y = y + o_t[:, None] * D + o_d[None, :]
    p_rstd = rstd + o_t
    p_dy = dy + o_t[:, None] * D + o_d[None, :]
    p_dx = dx + o_t[:, None] * D + o_d[None, :]

    b_y = tl.load(p_y, mask=m_x, other=0.0).to(tl.float32)
    b_rstd = tl.load(p_rstd, mask=m_t, other=0.0).to(tl.float32)
    b_dy = tl.load(p_dy, mask=m_x, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=m_x)


def l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape

    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]

    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # fla launches another kernel (l2norm_bwd_kernel_row) for D > 512.
    assert D <= 512
    # bucket token counts to limit autotuning across sequence lengths.
    NB = triton.cdiv(T, 2048 * 32)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]),)

    # One configuration of the autotune list at fla/modules/norm/l2norm.py:88: BT = 16, num_warps = 4.
    l2norm_bwd_kernel[grid](y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, T=T, D=D, BD=BD, NB=NB, BT=16, num_warps=4)

    return dx.view(y_shape_og)


def reference(y, rstd, dy):
    """dy * rstd - (sum dy * y) * y * rstd, in float64, from the exact inputs."""
    yd, rd, gd = y.double(), rstd.double()[..., None], dy.double()
    return gd * rd - (gd * yd).sum(-1, keepdim=True) * yd * rd


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
        x = (torch.randn((B, T, H, D), generator=gen) * 0.5 + 0.3).to(device=DEVICE, dtype=torch.float32)
        rstd = (1.0 / torch.sqrt((x * x).sum(-1) + 1e-6)).float()
        y = (x * rstd[..., None]).to(torch.bfloat16)
        dy = torch.randn((B, T, H, D), generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        dx = l2norm_bwd(y, rstd, dy)
        dx_ref = reference(y, rstd, dy)
        assert dx.dtype == torch.bfloat16 and dx.shape == y.shape
        err = ((dx.double() - dx_ref).abs() / (dx_ref.abs() + 1e-3)).max().item()
        print(f"B {B} T {T} H {H} D {D}: dx max error / (|dx| + 1e-3) {err:.2e}")
        # bfloat16 rounds to 2^-8 relative; the float32 error is far smaller.
        assert err < 8e-3, err
    print("l2norm_bwd matches the float64 reference")
