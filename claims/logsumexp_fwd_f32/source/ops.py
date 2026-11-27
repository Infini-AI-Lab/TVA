"""
Block-wise log-sum-exp, forward
===============================

The log-sum-exp of each row of ``x`` along its last dimension, computed in blocks: the row of
length ``D`` is cut into ``ND = cdiv(D, BD)`` blocks of ``BD`` columns, and program ``(n, i)``
stores ``z[n, i] = log(sum_j exp(y_j - m)) + m`` over the columns ``j`` of block ``i`` of row
``n``, with ``y_j = x[n, j] * scale`` (or ``x[n, j]`` without a scale) and ``m`` the block's
largest ``y_j``. The host combines the partial values with ``torch.logsumexp`` over the blocks.
On NVIDIA GPUs fla calls it only from its test; its fused linear cross entropy on Ascend NPUs uses
a counterpart of this kernel for the normaliser of its logits (``D`` the vocabulary size).

Layouts (contiguous):

* ``x``: ``[..., D]``, viewed as ``[N, D]`` (row stride ``D``), ``float32``;
* ``z``: ``[N, ND]``, ``float32`` (the kernel's output); the host returns the shape of ``x``
  without its last dimension.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, package ``fla/``.
Here, the kernel is lines 67-91.

* ``logsumexp_fwd_kernel``: ``fla/modules/fused_linear_cross_entropy/ops.py:45-73`` (the kernel's
  ``def`` at line 50);
* ``exp``, ``log``, ``tanh``: ``fla/ops/utils/op.py:44-45, 48-49, 52-53`` (the branch taken when
  the environment variable ``FLA_USE_FAST_OPS`` is not ``'1'``, the default);
* ``logsumexp_fwd`` (the host): adapted from ``fla/modules/fused_linear_cross_entropy/ops.py:94-107``.

Changes from the source:

* ``fla/modules/fused_linear_cross_entropy/ops.py:45-48``: the decorator
  ``@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None, 'HAS_SOFTCAPPING':
  lambda args: args['softcapping'] is not None})`` is removed. ``@triton.jit`` and the kernel's
  signature and body are verbatim.
* ``fla/ops/utils/op.py:43-53``: the default branch's helpers ``exp``, ``log`` and ``tanh`` (the
  names the source file imports from it, line 26) are copied into this file as top-level
  definitions, with the import of ``triton.language.extra.libdevice`` as ``tldevice`` that
  ``tanh`` uses (``op.py:12``). The branch for ``FLA_USE_FAST_OPS=1`` (``fast_expf``,
  ``fast_logf``, ``fast_tanhf``) is not copied.
* ``fla/modules/fused_linear_cross_entropy/ops.py:94-107`` (``logsumexp_fwd``): the decorator
  ``@dispatch`` (fla's backend registry, which on NVIDIA GPUs calls this function) is removed; the
  launch passes ``HAS_SCALE = scale is not None`` and ``HAS_SOFTCAPPING = softcapping is not
  None``, which the removed heuristics computed. The rest (the view, ``BD``, ``ND``, the
  allocation of ``z``, the grid, the combination over the blocks and the final conversion) is the
  source's.
* ``reference`` and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))


@triton.jit
def log(x): return tl.log(x.to(tl.float32))


@triton.jit
def tanh(x): return tldevice.tanh(x.to(tl.float32))


@triton.jit
def logsumexp_fwd_kernel(
    x,  # [N, D]
    z,  # [N, ND]
    scale,
    softcapping,
    D: tl.constexpr,
    BD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    # [BD]
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    if HAS_SOFTCAPPING:
        b_x = softcapping * tanh(b_x / softcapping)
    b_x = tl.where(m_d, b_x, -float('inf'))
    b_m = tl.max(b_x, 0)
    b_z = log(tl.sum(exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, BD) + i_d, b_z)


def logsumexp_fwd(x, scale: float | None = None, softcapping: float | None = None, dtype: torch.dtype | None = None):
    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    BD = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, BD)

    z = x.new_empty(N, ND, dtype=torch.float)
    # HAS_SCALE and HAS_SOFTCAPPING as the removed @triton.heuristics computed them.
    logsumexp_fwd_kernel[(N, ND)](x=x, z=z, scale=scale, softcapping=softcapping, D=D, BD=BD,
                                  HAS_SCALE=scale is not None, HAS_SOFTCAPPING=softcapping is not None)
    z = z.logsumexp(-1).view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        z = z.to(dtype)
    return z


def reference(x, scale=None):
    """The log-sum-exp of each row of ``x`` (times ``scale``), in float64."""
    xd = x.double()
    if scale is not None:
        xd = xd * scale
    return torch.logsumexp(xd, -1)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (N, D, scale): fla's test shape (tests/modules/test_cross_entropy.py:415-421, 2 x 1025) and
    # logits of a chunk of tokens over common vocabulary sizes (one block up to 65536 columns,
    # several beyond).
    for N, D, scale in [
        (1, 1, None),
        (2, 1025, None),
        (2, 1025, 1.0),
        (2, 1025, 0.0),
        (63, 50257, None),
        (64, 65536, 0.5),
        (32, 128256, None),
        (16, 151936, 1.0),
        (8, 262144, 30.0),
    ]:
        x = (torch.randn((N, D), generator=gen) * 4.0).to(device=DEVICE, dtype=torch.float32)
        z = logsumexp_fwd(x, scale=scale)
        ref = reference(x, scale)
        assert z.dtype == torch.float32 and z.shape == x.shape[:-1]
        err = (z.double() - ref).abs().max().item()
        print(f"N {N} D {D} scale {scale}: max abs error {err:.2e}")
        assert err < 1e-4 * (1 + ref.abs().max().item()), err
    print("logsumexp_fwd matches the float64 reference")
