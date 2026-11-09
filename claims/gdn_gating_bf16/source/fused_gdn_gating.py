"""
Gated DeltaNet gating (the decay ``g`` and the write strength ``beta``)
======================================================================

For each token and value head of a Gated DeltaNet layer (Qwen3-Next's linear attention), the
kernel computes

* ``g = -exp(A_log) * softplus(a + dt_bias)``, with ``softplus(x) = (1 / beta) * log(1 +
  exp(beta * x))`` for ``beta * x <= threshold`` and ``x`` above it, in ``float32``;
* ``beta_output = sigmoid(b)``, computed in ``float32`` and converted to ``b``'s dtype before the
  store.

One program per (token, block of ``BLK_HEADS = 8`` heads); ``seq_len`` is 1, ``beta`` and
``threshold`` are compile-time values (1.0 and 20.0).

Layouts:

* ``A_log``: ``[num_heads]``, ``float32``;
* ``dt_bias``: ``[num_heads]``, ``bfloat16``;
* ``a``, ``b``: ``[batch, num_heads]``, ``bfloat16``, unit stride along the heads, row strides
  ``a.stride(0)`` and ``b.stride(0)`` (``num_heads``, or ``2 * num_heads`` for views split from
  one ``[batch, 2 * num_heads]`` projection);
* ``g``, ``beta_output`` (the outputs): ``[1, batch, num_heads]``, ``float32``, contiguous.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/attention/linear/fused_gdn_gating.py``:

* ``fused_gdn_gating_kernel``: line 11 (with the two comment lines above its decorator, 8-9);
* the host ``fused_gdn_gating``: line 44.

Changes from the source:

* ``fused_gdn_gating_kernel`` (``fused_gdn_gating.py:11``) is copied verbatim, with its
  ``@triton.jit`` and the comment lines above it (the source has no autotuning or heuristics on
  it, and its parameters have no defaults).
* The host ``fused_gdn_gating`` (``fused_gdn_gating.py:44``) is copied verbatim except that its
  return annotation ``Tuple[torch.Tensor, torch.Tensor]`` (``typing.Tuple``, line 1) is written
  ``tuple[torch.Tensor, torch.Tensor]``, and the launch option ``num_stages=3`` is added: the
  wrapper passes none, so this is Triton 3.8.0's default, which its launches get. It already
  allocates its outputs fresh. The grid, ``BLK_HEADS = 8``, ``num_warps=1`` and the defaults
  ``beta=1.0, threshold=20.0`` are the wrapper's.
* ``reference`` and the ``__main__`` test are new.
"""

import torch

import triton
import triton.language as tl


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
# beta_output = b.sigmoid()
@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
    blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    seq_len = 1
    stride_a = a.stride(0)
    stride_b = b.stride(0)
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        stride_a,
        stride_b,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
        num_stages=3,
    )
    return g, beta_output


def reference(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """``g`` and ``sigmoid(b)`` in ``float64`` (``g`` with the same ``softplus`` threshold)."""
    x = a.double() + dt_bias.double()
    softplus_x = torch.where(beta * x <= threshold, (1 / beta) * torch.log1p(torch.exp(beta * x)), x)
    g = -torch.exp(A_log.double()) * softplus_x
    return g.unsqueeze(0), torch.sigmoid(b.double()).unsqueeze(0)


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (batch, num_heads, a and b split from one [batch, 2 * num_heads] projection)
    cases = [
        (1, 32, False),
        (1, 48, True),
        (4, 8, True),
        (16, 48, True),
        (32, 24, True),
        (7, 4, False),
        (513, 16, False),
        (8192, 32, False),
        (32768, 32, True),
    ]
    for batch, num_heads, split in cases:
        A_log = torch.randn(num_heads, generator=gen).to(device=DEVICE, dtype=torch.float32)
        dt_bias = torch.randn(num_heads, generator=gen).to(device=DEVICE, dtype=torch.bfloat16)
        if split:
            mixed_ba = (torch.randn(batch, 2 * num_heads, generator=gen) * 8).to(device=DEVICE, dtype=torch.bfloat16)
            b, a = mixed_ba.split([num_heads, num_heads], dim=-1)
        else:
            a = (torch.randn(batch, num_heads, generator=gen) * 8).to(device=DEVICE, dtype=torch.bfloat16)
            b = (torch.randn(batch, num_heads, generator=gen) * 8).to(device=DEVICE, dtype=torch.bfloat16)
        if batch > 0:  # some values past the softplus threshold
            a[0, :2] = torch.tensor([25.0, -30.0], dtype=torch.bfloat16)
        g, beta_output = fused_gdn_gating(A_log, a, b, dt_bias)
        g_ref, beta_ref = reference(A_log, a, b, dt_bias)
        assert g.shape == g_ref.shape == (1, batch, num_heads) and g.dtype == torch.float32
        g_err = (g.double() - g_ref).abs()
        g_bound = 1e-5 * g_ref.abs() + 1e-6 * torch.exp(A_log.double())
        b_err = (beta_output.double() - beta_ref).abs()
        b_bound = 2.0**-7 * beta_ref.abs() + 1e-7
        print(f"batch {batch} heads {num_heads} strides {a.stride(0)}/{b.stride(0)}: g max abs error "
              f"{g_err.max().item():.2e} (/bound {(g_err / g_bound).max().item():.2f}), beta max abs error "
              f"{b_err.max().item():.2e} (/bound {(b_err / b_bound).max().item():.2f})")
        assert (g_err <= g_bound).all() and (b_err <= b_bound).all()
    print("fused_gdn_gating matches the float64 reference")
