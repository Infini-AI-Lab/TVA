"""
Positions and sequence lengths of a batch (vLLM)
================================================

A batch of ``num_reqs`` requests. Request ``r`` (its *batch index*) has its state in slot
``idx_mapping[r]`` of the request states; ``num_computed_tokens[idx_mapping[r]]`` tokens of it are
already computed, and its ``query_len = query_start_loc[r + 1] - query_start_loc[r]`` new tokens
are packed at ``query_start_loc[r] .. query_start_loc[r + 1] - 1`` of the batch. One program per
request writes

* ``seq_lens[r] = num_computed_tokens[idx_mapping[r]] + query_len``, and
* ``positions[query_start_loc[r] + j] = num_computed_tokens[idx_mapping[r]] + j`` for
  ``j < query_len``, in blocks of ``BLOCK_SIZE = 1024``;

and one more program (the last, ``program_id == num_programs - 1``) writes ``seq_lens[r] = 0``
for the unused entries ``num_reqs <= r < max_num_reqs`` (``max_num_reqs = seq_lens.shape[0]``),
also in blocks of 1024.

Layouts and dtypes (vLLM's ``InputBuffers`` and ``RequestState``):

* ``idx_mapping``: ``[num_reqs]``, ``int64``;
* ``query_start_loc``: ``[num_reqs_padded + 1]``, ``int32``, the exclusive running sums of the
  query lengths from ``0`` (entries past ``num_reqs`` repeat the total);
* ``num_computed_tokens``: ``[max_num_reqs]``, ``int32``, indexed by state slot;
* ``pos`` (output): ``[max_num_tokens]``, ``int64``;
* ``seq_lens`` (output): ``[max_num_reqs]``, ``int32``.

Provenance: vLLM v0.33.0 (commit ``4ba787ec99``), ``vllm/v1/worker/gpu/input_batch.py``:

* ``_prepare_pos_seq_lens_kernel``: ``input_batch.py:356-390`` (``@triton.jit`` at 356, ``def``
  at 357);
* the host ``prepare_pos_seq_lens``: ``input_batch.py:393-412``; its one caller is
  ``vllm/v1/worker/gpu/model_runner.py:1499``.

Changes from the source:

* ``input_batch.py:356-390``: the kernel, copied with its decorator ``@triton.jit``; it has no
  autotuning or heuristics and calls no helper. One comment line of the kernel (line 369, above the
  zeroing loop) is removed; the code is unchanged.
* ``input_batch.py:393-412``: the host, copied; the two comment lines above the launch (lines
  401-402) are removed; the code is unchanged.
* ``input_batch.py:10``: ``from vllm.triton_utils import tl, triton`` is ``import triton`` and
  ``import triton.language as tl`` (vLLM's module re-exports Triton's).
* ``reference``, ``make_batch`` and the ``__main__`` test are new; the batches follow the
  model runner's (``model_runner.py:1400-1505``) and vLLM's serving sizes.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )


def reference(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens):
    """Exact, on the CPU: the new `pos` and `seq_lens` (int64 tensors) from the old ones."""
    num_reqs = idx_mapping.shape[0]
    pos = pos.to(torch.int64).clone()
    seq_lens = seq_lens.to(torch.int64).clone()
    for r in range(num_reqs):
        nc = int(num_computed_tokens[int(idx_mapping[r])])
        start, end = int(query_start_loc[r]), int(query_start_loc[r + 1])
        seq_lens[r] = nc + end - start
        pos[start:end] = torch.arange(nc, nc + end - start, dtype=torch.int64)
    seq_lens[num_reqs:] = 0
    return pos, seq_lens


def make_batch(query_lens, max_num_reqs, max_num_tokens, max_context, generator):
    """A batch as the model runner builds it: request states in random slots of `max_num_reqs`,
    `num_computed_tokens` below `max_context`, `query_start_loc` the running sums of `query_lens`
    from 0 padded with the total, `pos` and `seq_lens` holding stale values."""
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    assert num_reqs <= max_num_reqs and num_tokens <= max_num_tokens
    idx_mapping = torch.randperm(max_num_reqs, generator=generator)[:num_reqs].to(torch.int64)
    num_computed_tokens = torch.randint(0, max_context, (max_num_reqs,), generator=generator,
                                        dtype=torch.int32)
    query_start_loc = torch.full((max_num_reqs + 1,), num_tokens, dtype=torch.int32)
    query_start_loc[0] = 0
    query_start_loc[1:num_reqs + 1] = torch.tensor(query_lens, dtype=torch.int64).cumsum(0).to(torch.int32)
    pos = torch.randint(-2 ** 40, 2 ** 40, (max_num_tokens,), generator=generator, dtype=torch.int64)
    seq_lens = torch.randint(-2 ** 31, 2 ** 31, (max_num_reqs,), generator=generator,
                             dtype=torch.int64).to(torch.int32)
    return idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens


if __name__ == "__main__":
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    gen = torch.Generator().manual_seed(0)
    # (name, query lengths, max_num_reqs, max_num_tokens, max computed tokens)
    for name, query_lens, max_num_reqs, max_num_tokens, max_context in [
        ("decode", [1] * 8, 256, 8192, 4096),
        ("prefill", [1, 1023, 1024, 1025, 3000, 0, 17], 16, 8192, 2048),
        ("full batch", [2] * 128, 128, 512, 1000),
        ("one request", [5], 1, 16, 100),
        ("long padding", [3, 1, 2], 4096, 64, 100),
        ("spec decode", [4] * 64, 1024, 8192, 32768),
        ("serving", [1] * 1000 + [299] * 24, 1024, 8192, 131072),
        ("long prefill", [2 ** 20], 256, 2 ** 20, 2 ** 21),
    ]:
        idx, qsl, nct, pos, seq_lens = make_batch(query_lens, max_num_reqs, max_num_tokens,
                                                  max_context, gen)
        ref_pos, ref_seq_lens = reference(idx, qsl, nct, pos, seq_lens)
        idx_d, qsl_d, nct_d, pos_d, seq_lens_d = (x.to(DEVICE) for x in (idx, qsl, nct, pos, seq_lens))
        prepare_pos_seq_lens(idx_d, qsl_d[:len(query_lens) + 1], nct_d, pos_d, seq_lens_d)
        ok = torch.equal(pos_d.cpu(), ref_pos) and torch.equal(seq_lens_d.cpu().to(torch.int64), ref_seq_lens)
        print(f"{name}: {len(query_lens)} requests, {sum(query_lens)} tokens, max_num_reqs {max_num_reqs}: "
              f"{'exact' if ok else 'MISMATCH'}")
        assert ok
    print("✅ positions and seq_lens match the reference exactly")
