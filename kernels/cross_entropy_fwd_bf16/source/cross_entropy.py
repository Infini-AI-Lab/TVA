"""
Cross entropy, forward
======================

The loss of each row of ``logits`` against its target class, with the row split in blocks of
``BV`` classes: program ``(n, j)`` reads block ``j`` of row ``n`` (classes ``j BV .. j BV + BV - 1``
below ``V``) and computes, in ``float32``, the block's log-sum-exp ``lse`` and its part of the loss:
``-logit[target]`` when the target is in the block (else ``0``), with label smoothing
``(1 - eps) * that - eps * sum(block's logits) / V_TOTAL``. A row whose target is ``ignore_index``
has loss ``0``. When the row is one block (``NV = 1``, no split), the program adds
``lse + z_loss`` to the loss, with ``z_loss = lse_square_scale * lse^2``, and writes ``z_loss``;
when it is split (``NV > 1``), it writes each block's partial loss and ``lse`` and leaves ``z_loss``
alone (fla then combines the blocks in a second kernel, ``cross_entropy_fwd_kernel_reduce``,
which this file does not hold). The logits may be scaled (``logit_scale``) and soft-capped
(``logit_softcapping``, by ``tanh``) first.

Layouts:

* ``logits``: ``[N, V]``, ``bfloat16``, contiguous in the last dimension, row stride ``s_logits``;
* ``target``: ``[N]``, ``int64``;
* ``loss``, ``lse``: ``[N]`` (``NV = 1``) or ``[NV, N]`` (``NV > 1``), ``float32``;
* ``z_loss``: ``[N]``, ``float32``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``. Here, the kernel is lines 67-124; ``exp`` and ``log`` are lines 59-64.

* ``cross_entropy_fwd_kernel``: ``fla/modules/fused_cross_entropy/ops.py:26-83`` (the kernel's
  ``def`` at line 27).
* ``exp``, ``log``: ``fla/ops/utils/op.py:44-45`` and ``48-49``.
* ``cross_entropy_fwd`` (the host): adapted from ``fla/modules/fused_cross_entropy/ops.py:230-295``.

Changes from the source:

* ``fla/modules/fused_cross_entropy/ops.py:26-83``: none; the kernel has no autotuning, and
  ``@triton.jit(do_not_specialize=['N'])`` and its signature and body are verbatim.
* ``fla/ops/utils/op.py:32-53``: fla picks its ``exp``, ``log`` by the environment variable
  ``FLA_USE_FAST_OPS``; this file copies the default definitions (``FLA_USE_FAST_OPS`` unset or
  ``'0'``, the default on an H200), ``exp`` at ``op.py:44-45`` and ``log`` at ``op.py:48-49``.
* ``fla/modules/fused_cross_entropy/ops.py:16``: the import of ``tanh`` from
  ``triton.language.extra.libdevice`` is removed. The kernel calls ``tanh`` only when
  ``logit_softcapping`` is not ``None``, a branch on a compile-time value that the host never takes
  (it passes ``logit_softcapping=None``), so Triton never resolves the name.
* ``fla/modules/fused_cross_entropy/ops.py:230-295`` (``cross_entropy_fwd``): the decorator
  ``@dispatch`` (fla's backend registry, which falls back to this function on NVIDIA GPUs unless
  its Gluon backend is enabled, ``FLA_GLUON``/``FLA_CE_GLUON``, off by default) is removed; the
  argument ``process_group`` is removed, so ``world_size = 1`` and ``rank = 0`` (no vocabulary
  parallelism); the call of ``cross_entropy_reduce`` (``ops.py:173-227``, which launches
  ``cross_entropy_fwd_kernel_reduce``) when the row is split is removed, so the host returns the
  kernel's outputs: per block when ``NV > 1``. The rest (the stride handling, ``BV``,
  ``num_warps``, ``NV``, the allocations, the grid and the launch's keywords) is the wrapper's.
* ``reference`` and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))


@triton.jit
def log(x): return tl.log(x.to(tl.float32))


@triton.jit(do_not_specialize=['N'])
def cross_entropy_fwd_kernel(
    logits,  # [N, V]
    target,  # [N]
    loss,  # [NV, N] or [N] when NV == 1
    z_loss,  # [N]
    lse,  # [NV, N] or [N] when NV == 1
    s_logits,
    label_smoothing: tl.constexpr,
    logit_scale: tl.constexpr,
    lse_square_scale: tl.constexpr,
    logit_softcapping: tl.constexpr,
    ignore_index: tl.constexpr,
    N,
    V: tl.constexpr,
    V_TOTAL: tl.constexpr,
    V_START: tl.constexpr,
    BV: tl.constexpr,
    SPLIT: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    p_logits = logits + i_n * s_logits

    # [BV]
    b_logits = tl.load(p_logits + o_v, mask=m_v, other=-float('inf')).to(tl.float32) * logit_scale
    if logit_softcapping is not None:
        b_logits = logit_softcapping * tanh(b_logits / logit_softcapping)
    # these transforms do not preserve -inf in padded lanes
    if logit_softcapping is not None or logit_scale <= 0:
        b_logits = tl.where(m_v, b_logits, -float('inf'))
    if label_smoothing > 0:
        b_sum = tl.sum(tl.where(m_v, b_logits, 0.0), 0)
    b_max = tl.max(b_logits, 0)
    b_lse = log(tl.sum(exp(b_logits - b_max), 0)) + b_max

    b_target = tl.load(target + i_n).to(tl.int64)
    b_loss = 0.0
    b_z_loss = 0.0
    if b_target != ignore_index:
        b_target -= V_START
        m_target = (b_target >= i_v * BV) & (b_target < tl.minimum(V, (i_v + 1) * BV))
        b_target_logit = tl.load(p_logits + b_target, mask=m_target, other=0).to(tl.float32) * logit_scale
        if logit_softcapping is not None:
            b_target_logit = logit_softcapping * tanh(b_target_logit / logit_softcapping)
        b_loss = -b_target_logit
        if label_smoothing > 0:
            b_loss = (1 - label_smoothing) * b_loss - label_smoothing * b_sum / V_TOTAL
        if not SPLIT:
            b_z_loss = lse_square_scale * b_lse * b_lse
            b_loss += b_lse + b_z_loss

    tl.store(loss + i_v * N + i_n, b_loss)
    tl.store(lse + i_v * N + i_n, b_lse)
    if not SPLIT:
        tl.store(z_loss + i_n, b_z_loss)


def cross_entropy_fwd(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    logit_softcapping: float | None = None,
    ignore_index: int = -100,
):
    N, V = logits.shape
    assert target.shape == (N,)
    # No process group: one rank holds the whole vocabulary.
    world_size = 1
    total_classes = world_size * V
    rank = 0
    class_start_idx = rank * V

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    BV = min(triton.next_power_of_2(V), 64 * 1024)
    # reduce softcap register pressure without introducing an additional split reduction
    if logit_softcapping is not None and V > BV:
        BV = 8 * 1024
    num_warps = 4 if BV < 2048 else (8 if BV < 8192 else 16)
    # vocab partitions contribute partial loss until the global LSE is available
    NV = triton.cdiv(V, BV)
    split = world_size > 1 or NV > 1
    shape = (NV, N) if NV > 1 else (N,)
    loss = logits.new_empty(shape, dtype=torch.float)
    lse = torch.empty_like(loss)
    z_loss = logits.new_empty(N, dtype=torch.float)

    cross_entropy_fwd_kernel[(N, NV)](
        logits=logits,
        target=target,
        loss=loss,
        z_loss=z_loss,
        lse=lse,
        s_logits=logits.stride(0),
        label_smoothing=label_smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        logit_softcapping=logit_softcapping,
        ignore_index=ignore_index,
        N=N,
        V=V,
        V_TOTAL=total_classes,
        V_START=class_start_idx,
        BV=BV,
        SPLIT=split,
        num_warps=num_warps,
    )

    # fla combines the blocks here when split (cross_entropy_reduce, a second kernel).
    return loss, z_loss, lse, total_classes, class_start_idx


def reference(logits, target, BV, label_smoothing=0.0, lse_square_scale=0.0, ignore_index=-100):
    """Each block's lse and loss (and the row's z-loss when the row is one block), in float64."""
    N, V = logits.shape
    NV = triton.cdiv(V, BV)
    x = logits.double()
    lse = torch.empty((NV, N), dtype=torch.float64, device=logits.device)
    loss = torch.zeros((NV, N), dtype=torch.float64, device=logits.device)
    valid = target != ignore_index
    t = torch.where(valid, target, 0)
    for j in range(NV):
        blk = x[:, j * BV:min(V, (j + 1) * BV)]
        lse[j] = torch.logsumexp(blk, dim=1)
        inside = valid & (t >= j * BV) & (t < min(V, (j + 1) * BV))
        tl_logit = torch.where(inside, x.gather(1, t[:, None])[:, 0], 0.0)
        part = -tl_logit
        if label_smoothing > 0:
            part = (1 - label_smoothing) * part - label_smoothing * blk.sum(1) / V
        loss[j] = torch.where(valid, part, 0.0)
    z = torch.where(valid, lse_square_scale * lse[0] ** 2, 0.0)
    if NV == 1:
        loss[0] = torch.where(valid, loss[0] + lse[0] + z, 0.0)
        return loss[0], z, lse[0]
    return loss, None, lse


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (N, V, label_smoothing, row stride padding): fla's tests (tests/modules/test_cross_entropy.py:28-145:
    # 'basic' 1024 x 32000, 'large_vocab' 63 x 131071, 'strided_tail' 127 x 4103, 'smoothing'
    # 1025 x 32000, 'split_boundary' 7 x 65536, 'row_tail_split' 5 x 8193, 'single_class' 2 x 1,
    # 255 x 65537), one row, and model vocabularies (50277, 128256, 151936).
    for N, V, ls, pad in [
        (1, 1, 0.0, 0),
        (2, 1, 0.0, 0),
        (1, 32000, 0.0, 0),
        (1024, 32000, 0.0, 0),
        (1025, 32000, 0.1, 0),
        (127, 4103, 0.1, 64),
        (5, 8193, 0.1, 0),
        (7, 65536, 0.0, 0),
        (255, 65537, 0.1, 0),
        (63, 131071, 0.0, 0),
        (333, 50277, 0.0, 3),
        (512, 128256, 0.0, 0),
        (1024, 151936, 0.1, 0),
    ]:
        buf = (2 * torch.randn((N, V + pad), generator=gen)).to(device=DEVICE, dtype=torch.bfloat16)
        logits = buf[:, :V]
        target = torch.randint(V, (N,), generator=gen)
        target[::3] = -100
        target[N // 2] = V - 1
        target = target.to(DEVICE)
        loss, z_loss, lse, total, start = cross_entropy_fwd(logits, target, label_smoothing=ls)
        BV = min(triton.next_power_of_2(V), 64 * 1024)
        loss_ref, z_ref, lse_ref = reference(logits, target, BV, label_smoothing=ls)
        assert loss.dtype == lse.dtype == torch.float32 and loss.shape == loss_ref.shape
        err_l = (loss.double() - loss_ref).abs().max().item()
        err_s = (lse.double() - lse_ref).abs().max().item()
        err_z = 0.0 if z_ref is None else (z_loss.double() - z_ref).abs().max().item()
        print(f"N {N} V {V} label_smoothing {ls} row stride {V + pad} (NV {triton.cdiv(V, BV)}): "
              f"loss max abs error {err_l:.2e}, lse max abs error {err_s:.2e}, z_loss max abs error {err_z:.2e}")
        assert err_l < 1e-4 and err_s < 1e-4 and err_z == 0.0, (err_l, err_s, err_z)
    print("cross_entropy_fwd matches the float64 reference")
