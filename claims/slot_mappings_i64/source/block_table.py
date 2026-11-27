# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Slot mappings of a batch's tokens (vLLM)
========================================

For each KV-cache group ``g`` and each token ``t`` of the batch, the KV-cache slot the token's
key and value are written to: ``t`` belongs to request ``b`` (``query_start_loc[b] <= t <
query_start_loc[b + 1]``), whose row of the group's block table is ``idx_mapping[b]``; with the
token's position ``p = positions[t]`` and the group's block size ``B``, the slot is
``block_table[g][idx_mapping[b], p // B] * B + p % B``. A request with ``idx_mapping[b] < 0``
(a dummy request) and a group whose layers use no slot mapping get ``PAD_SLOT_ID = -1``, and so
does every token from ``query_start_loc[num_reqs]`` to the end of the row. With decode context
parallelism (``CP_SIZE > 1``) and a sharded group, a rank stores only its own tokens (every
``CP_INTERLEAVE`` consecutive positions of a virtual block of ``B * CP_SIZE`` go to the next rank),
at its local position, and the others get ``-1``.

Layouts and dtypes:

* ``idx_mapping``: ``[num_reqs]``, ``int64``;
* ``query_start_loc``: ``[num_reqs + 1]`` (at least), ``int32``, nondecreasing;
* ``positions``: ``[num_tokens]`` (at least), ``int64``;
* per group ``g`` (``num_groups`` of them): the block table, ``[max_num_reqs, max_num_blocks_g]``,
  ``int32``, contiguous, passed by address (``block_table_ptrs``, ``uint64``) with its row stride
  (``block_table_strides``, ``int64``); the block size (``block_sizes``, ``int32``) and the flags
  ``slot_mapping_enabled`` and ``dcp_sharded`` (``bool``);
* ``slot_mappings`` (output): ``[num_groups, max_num_tokens]``, ``int64``.

Provenance: vLLM v0.33.0 (commit ``4ba787ec99``):

* ``_compute_slot_mappings_kernel``: ``vllm/v1/worker/gpu/block_table.py:333-414``;
* ``_load_ptr``: ``vllm/v1/worker/gpu/buffer_utils.py:353-357``;
* ``PAD_SLOT_ID``: ``vllm/v1/attention/backends/utils.py:54``;
* the host, from ``BlockTables`` (``block_table.py:17-271``).

Changes from the source:

* ``block_table.py:333-414``: the kernel, copied verbatim with its decorator ``@triton.jit``; it
  has no autotuning or heuristics.
* ``buffer_utils.py:353-357``: the helper ``_load_ptr`` the kernel calls, copied verbatim into this
  file (vLLM imports it from ``buffer_utils.py``).
* ``utils.py:54``: ``PAD_SLOT_ID = -1`` is defined here (vLLM imports it); only the host reads it.
* The imports: ``triton`` and ``triton.language as tl`` directly (vLLM's ``vllm.triton_utils``
  re-exports them).
* The host ``BlockTables`` keeps what ``compute_slot_mappings`` uses: the constructor's arguments
  ``block_sizes``, ``max_num_reqs``, ``max_num_batched_tokens``, ``max_num_blocks_per_group``,
  ``device``, ``cp_size``, ``cp_rank``, ``cp_interleave``, ``slot_mapping_enabled``,
  ``dcp_sharded``; each group's block table is a plain ``torch.zeros`` tensor (vLLM's
  ``StagedWriteTensor.gpu`` is one, ``buffer_utils.py:161``) written directly by ``set_rows``
  instead of through staged writes; ``self.slot_mappings`` as in vLLM (``block_table.py:90-95``);
  the layout tensors of ``init_block_table_layout_tensors`` (``:105-123``) that the kernel reads.
  ``compute_slot_mappings`` (``:229-260``) is verbatim. The other methods (staged writes, the
  gather of block tables and its kernel, the dummy tables, the sliding-window masks) are dropped.
* ``reference`` and the ``__main__`` test are new.
"""

from collections.abc import Iterable

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mapping_enabled,  # [num_kv_cache_groups]
    dcp_sharded,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    kv_block_size = tl.load(block_sizes + group_id)
    mapping_enabled = tl.load(slot_mapping_enabled + group_id)
    if CP_SIZE != 1:
        sharded = tl.load(dcp_sharded + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    # idx_mapping == -1 marks a dummy (or CUDA-graph padding) request that owns
    # no blocks: never read its block-table row and emit PAD for its tokens.
    is_real_req = req_state_idx >= 0
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            local_positions = positions
            is_local = True
        else:
            # Context parallelism is used.
            virtual_block_size = kv_block_size * CP_SIZE
            virtual_block_indices = positions // virtual_block_size
            virtual_block_offsets = positions % virtual_block_size
            is_local = virtual_block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = virtual_block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = virtual_block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            local_positions = virtual_block_indices * kv_block_size + local_offsets
            local_positions = tl.where(sharded, local_positions, positions)
            is_local = ~sharded | is_local

        block_indices = tl.where(mapping_enabled, local_positions // kv_block_size, 0)
        block_offsets = local_positions % kv_block_size
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=is_local & is_real_req,
            other=0,
        )
        slot_ids = block_numbers * kv_block_size + block_offsets
        if CP_SIZE != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        slot_ids = tl.where(mapping_enabled & is_real_req, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)


class BlockTables:
    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
        slot_mapping_enabled: list[bool] | None = None,
        dcp_sharded: list[bool] | None = None,
    ):
        self.block_sizes = block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups
        if slot_mapping_enabled is None:
            slot_mapping_enabled = [True] * self.num_kv_cache_groups
        assert len(slot_mapping_enabled) == self.num_kv_cache_groups
        self._slot_mapping_enabled = slot_mapping_enabled
        if dcp_sharded is None:
            dcp_sharded = [True] * self.num_kv_cache_groups
        assert len(dcp_sharded) == self.num_kv_cache_groups
        self.dcp_sharded = torch.tensor(dcp_sharded, dtype=torch.bool, device=device)

        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[torch.Tensor] = []
        for max_num_blocks in max_num_blocks_per_group:
            block_table = torch.zeros(
                (self.max_num_reqs, max_num_blocks), dtype=torch.int32, device=device
            )
            self.block_tables.append(block_table)

        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        self.block_table_ptrs = self._make_ptr_tensor(self.block_tables)
        self.block_table_strides = torch.tensor(
            [b.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.slot_mapping_enabled = torch.tensor(
            self._slot_mapping_enabled, dtype=torch.bool, device=self.device
        )

    def set_rows(self, group: int, req_index: int, block_ids: list[int]) -> None:
        """Row ``req_index`` of group ``group``'s block table starts with ``block_ids``."""
        row = torch.tensor(block_ids, dtype=torch.int32)
        self.block_tables[group][req_index, : len(block_ids)] = row.to(self.device)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_kv_cache_groups == 0:
            return (self.slot_mappings if out is None else out)[:, :num_tokens_padded]
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        slot_mappings = self.slot_mappings if out is None else out
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            slot_mappings.shape[1],
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mapping_enabled,
            self.dcp_sharded,
            slot_mappings,
            slot_mappings.stride(0),
            self.cp_rank,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )
        return slot_mappings[:, :num_tokens_padded]


def reference(tables: BlockTables, idx_mapping, query_start_loc, positions, init_out):
    """The slot mappings the kernel must leave, on the CPU, from the initial output ``init_out``
    (``[num_groups, max_num_tokens]``): exact integer arithmetic, token by token."""
    out = init_out.clone().cpu()
    n = out.shape[1]
    idx = idx_mapping.cpu().tolist()
    qsl = query_start_loc.cpu().long().tolist()
    pos = positions.cpu().tolist()
    C, I, r = tables.cp_size, tables.cp_interleave, tables.cp_rank
    for g in range(tables.num_kv_cache_groups):
        bt = tables.block_tables[g].cpu().reshape(-1).tolist()
        st = tables.block_tables[g].stride(0)
        B = tables.block_sizes[g]
        en = tables._slot_mapping_enabled[g]
        sh = bool(tables.dcp_sharded[g])
        for b in range(len(idx)):
            for t in range(qsl[b], qsl[b + 1]):
                p = pos[t]
                slot = PAD_SLOT_ID
                if en and idx[b] >= 0:
                    local, lp = True, p
                    if C != 1 and sh:
                        vbo = p % (B * C)
                        local = vbo // I % C == r
                        lp = p // (B * C) * B + vbo // (I * C) * I + vbo % I
                    if local:
                        slot = bt[idx[b] * st + lp // B] * B + lp % B
                out[g, t] = slot
        out[g, qsl[len(idx)] : n] = PAD_SLOT_ID
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda")
    for cp_size, cp_rank, cp_interleave in [(1, 0, 1), (2, 1, 1), (4, 2, 8)]:
        tables = BlockTables(
            block_sizes=[16, 64],
            max_num_reqs=8,
            max_num_batched_tokens=4096,
            max_num_blocks_per_group=[64, 16],
            device=dev,
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_interleave=cp_interleave,
            slot_mapping_enabled=[True, True],
            dcp_sharded=[True, False],
        )
        for g in range(2):
            for req in range(8):
                tables.set_rows(g, req, list(range(req * 100 + g, req * 100 + g + 16)))
        lens = [5, 0, 1000, 3]
        idx_mapping = torch.tensor([3, 1, 7, -1], dtype=torch.int64, device=dev)
        qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
        positions = torch.cat([torch.arange(10, 10 + n) for n in lens]).to(dev)
        init = tables.slot_mappings.clone()
        tables.compute_slot_mappings(idx_mapping, qsl, positions, 2048)
        ref = reference(tables, idx_mapping, qsl, positions, init)
        assert torch.equal(tables.slot_mappings.cpu(), ref), (cp_size, cp_rank, cp_interleave)
    print("ok")
