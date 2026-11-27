"""
Grammar-constrained decoding: apply a packed token bitmask to logits, in place (SGLang)
======================================================================================

Grammar-constrained decoding (xgrammar) gives, for each request of a batch, the set of tokens the
grammar allows next, as a bitmask packed 32 tokens per ``int32``: bit ``c % 32`` of word ``c // 32``
of a row is token ``c``'s bit, ``1`` for allowed and ``0`` for masked. The kernel sets every masked
logit of each row to ``-inf``, in place, and leaves every other logit as it is. With ``indices``,
only the listed rows are processed (logits row ``indices[k]`` with bitmask row ``indices[k]``).

The kernel is a persistent loop: ``NUM_SMS`` programs walk the ``num_rows * cdiv(vocab_size,
BLOCK_SIZE)`` work items (a row and a block of ``BLOCK_SIZE = 4096`` tokens) with stride
``NUM_SMS``. Each item loads the ``BLOCK_SIZE // 32 = 128`` words of its block (masked to the row's
words), expands them to ``4096`` bits (``>>`` by ``0..31`` and ``& 1``) and stores ``-inf`` where
the token is below ``vocab_size`` and its bit is ``0``.

Layouts and dtypes:

* ``logits``: ``[batch, vocab]`` (or ``[vocab]``), contiguous; ``float32`` on SGLang's path (the
  logits processor returns ``float32`` logits); updated in place;
* ``bitmask``: ``[batch, cdiv(vocab, 32)]`` (or one row), ``int32``, contiguous;
* ``indices`` (optional): a list or tensor of distinct rows, made an ``int32`` tensor by the host.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/grammar/bitmask_ops.py`` (adapted from xgrammar v0.1.17):

* ``apply_token_bitmask_inplace_kernel``: line 13 (``@triton.jit``), ``def`` at line 14;
* the host ``apply_token_bitmask_inplace_triton``: line 84;
* ``get_device_core_count``: ``python/sglang/srt/utils/common.py:1161``.

Changes from the source:

* ``apply_token_bitmask_inplace_kernel`` is copied verbatim, with its ``@triton.jit`` (the source
  has no autotuning or heuristics on it).
* ``apply_token_bitmask_inplace_triton`` is copied verbatim.
* The import ``from sglang.srt.utils import get_device_core_count`` is replaced by a copy of that
  function (``common.py:1161-1167``) with its MUSA and XPU branches dropped: on a CUDA device it
  returns the device's multiprocessor count (``132`` on the H200), otherwise ``0``.
* This docstring is new.
"""

from typing import List, Optional, Union

import torch
import triton
import triton.language as tl


def get_device_core_count(device_id: int = 0) -> int:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    return 0


@triton.jit
def apply_token_bitmask_inplace_kernel(
    logits_ptr,
    bitmask_ptr,
    indices_ptr,
    num_rows,
    vocab_size,
    logits_strides,
    bitmask_strides,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply a bitmask to logits in-place using Triton. The bitmask is a 01 bitwise compressed tensor,
    where 0 means the token is masked and 1 means the token is not masked. After applying the bitmask,
    the masked logits will be set to -inf.

    Parameters
    ----------
    logits_ptr : tl.tensor
        Pointer to the logits tensor to apply the bitmask to.

    bitmask_ptr : tl.tensor
        Pointer to the bitmask tensor to apply.

    indices_ptr : Optional[tl.tensor]
        Optional pointer to indices tensor specifying which rows to apply the mask to.

    num_rows : int
        Number of rows to process. If indices_ptr is provided, this is the number of unique indices.

    vocab_size : int
        Size of the vocabulary dimension. If the logits does not have a vocab padding, this is the
        same as the logits's second dimension. Otherwise, this is the actual size of the vocabulary.

    logits_strides : int
        Stride between rows in the logits tensor.

    bitmask_strides : int
        Stride between rows in the bitmask tensor.

    NUM_SMS : int
        Number of streaming multiprocessors to use.

    BLOCK_SIZE : int
        Size of processing blocks.
    """

    pid = tl.program_id(0)
    num_blocks = tl.cdiv(vocab_size, BLOCK_SIZE)
    for work_id in tl.range(pid, num_rows * num_blocks, NUM_SMS):
        row_id = work_id // num_blocks
        block_offset = (work_id % num_blocks) * BLOCK_SIZE
        batch_id = row_id if indices_ptr is None else tl.load(indices_ptr + row_id)
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        bitmask_offsets = block_offset // 32 + tl.arange(0, BLOCK_SIZE // 32)
        vocab_mask = offsets < vocab_size
        packed_bitmask_mask = bitmask_offsets < bitmask_strides
        packed_bitmask = tl.load(
            bitmask_ptr + batch_id * bitmask_strides + bitmask_offsets,
            packed_bitmask_mask,
        )
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE)

        tl.store(
            logits_ptr + batch_id * logits_strides + offsets,
            -float("inf"),
            vocab_mask & bitmask,
        )


def apply_token_bitmask_inplace_triton(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
):
    NUM_SMS = get_device_core_count()
    BLOCK_SIZE = 4096
    BITS_PER_BLOCK = 32

    # Check input dtype
    assert bitmask.dtype == torch.int32, "bitmask must be of type int32"

    # Check input tensor shapes.
    logits_shape = logits.shape
    bitmask_shape = bitmask.shape
    if logits.ndim == 1:
        logits_shape = (1, logits_shape[0])
    if bitmask.ndim == 1:
        bitmask_shape = (1, bitmask_shape[0])

    required_bitmask_width = (logits_shape[1] + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK
    assert required_bitmask_width >= bitmask_shape[1], (
        f"Bitmask width too large: allow at most {required_bitmask_width} int32s for "
        f"logits' width {logits_shape[1]}, but got {bitmask_shape[1]}"
    )

    vocab_size = min(logits_shape[1], bitmask_shape[1] * BITS_PER_BLOCK)

    num_rows = None
    if isinstance(indices, list) or isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, dtype=torch.int32, device=logits.device)
        num_rows = indices.shape[0]
    else:
        assert logits_shape[0] == bitmask_shape[0], (
            f"batch size mismatch: logits {logits_shape[0]} vs bitmask {bitmask_shape[0]}"
        )
        num_rows = logits_shape[0]

    if NUM_SMS > 0:
        grid = (NUM_SMS,)
    else:
        num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
        grid = (num_rows * num_blocks,)
        NUM_SMS = triton.next_power_of_2(grid[0])

    apply_token_bitmask_inplace_kernel[grid](
        logits,
        bitmask,
        indices,
        num_rows,
        vocab_size,
        logits_shape[1],
        bitmask_shape[1],
        NUM_SMS,
        BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32 // (16 // logits.element_size()),
        num_stages=3,
    )
