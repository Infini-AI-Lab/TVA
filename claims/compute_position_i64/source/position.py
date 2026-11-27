"""
Positions of an extend batch (each request's new token positions, flat)
========================================================================

An extend (prefill) batch of ``bs`` requests: request ``b`` already has ``extend_prefix_lens[b]``
tokens in the KV cache and adds ``extend_seq_lens[b]`` new ones. The new tokens of all requests
are laid out one request after another, so request ``b``'s tokens start at the running sum
``extend_start_loc[b] = extend_seq_lens[0] + ... + extend_seq_lens[b - 1]``, and its token ``j``
has position ``extend_prefix_lens[b] + j``. One program per request computes its start (a loop
over the lengths of the requests before it) and writes its positions in blocks of 512
(``BLOCK_SIZE``).

Layouts:

* ``extend_prefix_lens``: ``[bs]``, ``int32`` (``has_prefix`` is ``True``); any other length
  means no prefixes (``has_prefix`` is ``False``, every prefix ``0``);
* ``extend_seq_lens``: ``[bs]``, ``int32``;
* ``positions`` (output): ``[extend_seq_lens_sum]``, ``int64``;
* ``extend_start_loc`` (output): ``[bs]``, ``int32``.

Provenance: SGLang v0.5.23, commit ``297e54fd10``,
``python/sglang/kernels/ops/attention/position.py`` (the whole file). Its callers are
``compute_position`` (``python/sglang/srt/model_executor/forward_batch_info.py:2378``, used by
``ForwardBatch.init_new``, ``two_batch_overlap.py``, ``dflash_worker_v2.py`` and
``dspark_kv_inject.py``) and the code base's test
``test/registered/kernels/ops/speculative/test_next_verify_prep.py:112-134``.

Changes from the source: this docstring is new. The rest of the file (the host
``compute_position_triton`` and the ``@triton.jit`` kernel ``compute_position_kernel``) is
copied verbatim.
"""

import torch
import triton
import triton.language as tl


def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    has_prefix = extend_prefix_lens.shape[0] == batch_size

    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    # Launch kernel
    compute_position_kernel[(batch_size,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
        has_prefix,
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)
