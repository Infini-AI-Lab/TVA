import TVA

/-!
# `slot_mappings_i64`: the KV-cache slots of a batch's tokens, vLLM

The kernel is `_compute_slot_mappings_kernel` of `source/block_table.py` (lines 73-154), with the
helper `_load_ptr` it calls (lines 66-70); both read the module-level names `triton` and `tl`
(lines 60-61). The host `BlockTables.compute_slot_mappings` (lines 233-264) launches it on the grid
`(num_groups, num_reqs + 1)`: program `(g, b)` with `b < num_reqs` computes the slots of request
`b`'s tokens in KV-cache group `g`, and program `(g, num_reqs)` fills the rest of group `g`'s row
with `-1`. Its arguments: the `int64` `idx_mapping` (`[num_reqs]`, each request's row of the
block tables, negative for a dummy request), the `int32` `query_start_loc` (the requests' token
ranges), the `int64` `positions`, the per-group tables (the block tables' addresses as `uint64`,
their `int64` row strides, the `int32` block sizes, the `bool` flags `slot_mapping_enabled` and
`dcp_sharded`), the `int64` output `slot_mappings` (`[num_groups, max_num_tokens]`, row stride
`slot_mappings.stride(0)`), the rank `cp_rank`, and the constexprs `CP_SIZE`, `CP_INTERLEAVE`,
`PAD_ID = -1`, `TRITON_BLOCK_SIZE = 1024`. Every `CP_SIZE ≥ 1` (decode context parallelism when
`CP_SIZE > 1`) and `CP_INTERLEAVE ≥ 1` is covered. `SETUP.md` lists the launches.

What is claimed: every covered launch ends normally; for every group `g`, request `b` and token
`t` of `b` (`query_start_loc[b] ≤ t < query_start_loc[b + 1]`), element `(g, t)` of
`slot_mappings` holds `slot d init g b t`: with `B` the group's block size, `p = positions[t]`
and `p'` its local position (`p` itself without context parallelism or for an unsharded group),
`block_table[g][idx_mapping[b], p' // B] * B + p' % B` when the group uses slot mappings, the
request is real (`idx_mapping[b] ≥ 0`) and the token is the rank's (always without context
parallelism or for an unsharded group), and `-1` otherwise; every element `(g, t)` with
`query_start_loc[num_reqs] ≤ t < max_num_tokens` holds `-1`; every other byte is unchanged.

The conditions: at least one group, fewer than `2^31` programs; at most `8192` requests;
`max_num_tokens ≤ 2^22`, at most the output's row stride, which is below `2^31` with
`(num_groups − 1) · stride < 2^31` (the kernel's `int32` row offset); `CP_SIZE ≥ 1`,
`CP_INTERLEAVE ≥ 1`, `CP_INTERLEAVE · CP_SIZE < 2^31`, `cp_rank < 2^31`; each block size at least
`1` with `block_size · CP_SIZE < 2^31` (the kernel's `int32` virtual block size); the buffers hold
the per-group entries and the batch's, are valid, and the output shares no byte with an input;
each block table's address is a multiple of 16 (the kernel's hint); `query_start_loc` is
nondecreasing from a nonnegative start to at most `max_num_tokens` and the number of positions;
a real request with tokens has a row in each group's block table; and for each token of a real
request in a group that uses slot mappings: its position is nonnegative and, when it is the
rank's, its block lies in the block table and `block_number · B` fits `int32` (the kernel's
`int32` product, a limit of the kernel). Any positions of dummy requests and any block-table
entries the kernel does not use for a slot are covered.

The specification is exact: no float operation, no device fact.
-/
namespace Claims.SlotMappings

open TVA

/-! ## The model -/

/-- The pad value: `PAD_ID = -1`, the host's `PAD_SLOT_ID` (source line 63), as an `int64`. -/
def PAD : I64 := IntV.ofInt (-1)

/-- `_load_ptr(ptr_to_ptr, tl.int32)` (source lines 66-70), inlined at its call (line 108). -/
def _load_ptr (ptr_to_ptr : Ptr .u64) : Prog (Ptr .i32) := do
  -- line 68: one `uint64` element
  let ptr ← tl.load1 ptr_to_ptr
  -- line 69: `tl.cast(ptr, tl.pointer_type(elem_dtype))` with `elem_dtype = tl.int32`
  -- (`inttoptr`)
  let ptr := tl.intToPtr .i32 ptr
  -- line 70: `tl.multiple_of(ptr, 16)`, a hint on the value the cast defines: its address is a
  -- multiple of 16 bytes; the call returns it
  tl.multiple_of1 ptr 16

/-- `_compute_slot_mappings_kernel` (source lines 73-154). `max_num_tokens`,
`slot_mappings_stride` and `cp_rank` are runtime `int32`s (integer arguments in `[0, 2^31)`; one
equal to `1` is the compile-time constant `1`, whose every use (an operand of `*` or `-` with an
`int32` tensor, a comparison with one, a bound of `range` with one) builds `1` in `int32`, the
same value, see `model`). `CP_SIZE` and `CP_INTERLEAVE` are `tl.constexpr`s (Lean parameters);
`PAD_ID = -1` and `TRITON_BLOCK_SIZE = 1024` are the host's constexprs, written as `-1` and
`1024` below. The pointers: `idx_mapping`, `pos` and the output `*i64`; `query_start_loc` and
`block_sizes` `*i32`; `block_table_ptrs` `*u64`; `block_table_strides` `*i64`;
`slot_mapping_enabled` and `dcp_sharded` `*i1` (`torch.bool`). -/
def _compute_slot_mappings_kernel (max_num_tokens : I32) (idx_mapping : Ptr .i64)
    (query_start_loc : Ptr .i32) (pos : Ptr .i64) (block_table_ptrs : Ptr .u64)
    (block_table_strides : Ptr .i64) (block_sizes : Ptr .i32) (slot_mapping_enabled : Ptr .i1)
    (dcp_sharded : Ptr .i1) (slot_mappings_ptr : Ptr .i64) (slot_mappings_stride : I32)
    (cp_rank : I32) (CP_SIZE CP_INTERLEAVE : Nat) : Kernel where
  body ctx := do
    -- lines 93-94: axes 0 and 1 of the grid, `int32`s
    let group_id := tl.program_id ctx 0
    let batch_idx := tl.program_id ctx 1
    -- line 95: the wrapping `int32` product, added to the pointer
    let slot_mapping_ptr := slot_mappings_ptr + group_id * slot_mappings_stride
    -- line 97: a runtime `if` on `batch_idx == tl.num_programs(1) - 1` (`1` built in `int32`)
    if IntV.eq batch_idx (tl.num_programs ctx 1 - (IntV.ofInt 1 : I32)) then do
      -- line 102: one `int32` element
      let actual_num_tokens ← tl.load1 (query_start_loc + batch_idx)
      -- line 103: `for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE)`, `int32`
      -- bounds and the constexpr step
      let _ ← tl.forRange actual_num_tokens max_num_tokens (IntV.ofNat 1024 : I32) ()
        fun i _ => do
          -- line 104
          let offset : Tensor [1024] I32 := i + tl.arange 0 1024
          -- line 105: the stored Python `-1` becomes an `int32` tensor, broadcast, then
          -- converted to the pointer's `int64` (sign extension)
          tl.store (slot_mapping_ptr + offset)
            (Tensor.castInt (tl.full [1024] (IntV.ofInt (-1) : I32)))
            (some (offset.ltS max_num_tokens))
      -- line 106: the programs of the last row return
      return ()
    else do
      -- line 108: `_load_ptr`, inlined
      let block_table_ptr ← _load_ptr (block_table_ptrs + group_id)
      -- lines 109-111: one `int64`, one `int32` and one `int1` element
      let block_table_stride ← tl.load1 (block_table_strides + group_id)
      let kv_block_size ← tl.load1 (block_sizes + group_id)
      let mapping_enabled ← tl.load1 (slot_mapping_enabled + group_id)
      -- lines 112-113: a compile-time `if`. With `CP_SIZE ≠ 1`, one `int1` element; with
      -- `CP_SIZE = 1` the line is not visited, no load runs, and the name is never read (the
      -- branches below that read it are not visited either): the model binds `false`
      let sharded : Bool ← if CP_SIZE ≠ 1 then tl.load1 (dcp_sharded + group_id) else pure false
      -- line 115: one `int64` element
      let req_state_idx ← tl.load1 (idx_mapping + batch_idx)
      -- line 118: `req_state_idx >= 0`, the `0` an `int32` tensor widened to `int64`
      let is_real_req : Bool := IntV.le (IntV.ofInt 0 : I64) req_state_idx
      -- lines 119-120: two `int32` elements; `+ 1` adds the `int32` constant `1` to the pointer
      let start_idx ← tl.load1 (query_start_loc + batch_idx)
      let end_idx ← tl.load1 (query_start_loc + batch_idx + (IntV.ofInt 1 : I32))
      -- line 121: `for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE)`, `int32` bounds; the
      -- loop ends the kernel
      tl.forRange start_idx end_idx (IntV.ofNat 1024 : I32) () fun i _ => do
        -- line 122
        let offset : Tensor [1024] I32 := i + tl.arange 0 1024
        -- line 123: `other=0`, an `int32` tensor converted to the pointer's `int64`
        let positions : Tensor [1024] I64 ← tl.load (pos + offset) (some (offset.ltS end_idx))
          (some (Tensor.castInt (tl.full [1024] (IntV.ofInt 0 : I32))))
        -- `kv_block_size` promoted to `int64` (sign extension), as lines 138, 142 and 143 each
        -- do for their operation with an `int64` tensor
        let bs64 : I64 := IntV.cast kv_block_size
        -- lines 125-140: a compile-time `if` on `CP_SIZE == 1`
        let (local_positions, is_local) ← (if CP_SIZE = 1 then
            -- lines 127-128: `is_local` is the Python `True`; its one use, `True &
            -- is_real_req` (line 146), is the `int1` scalar `is_real_req`, broadcast by the load
            -- to every lane: the model holds `True` in every lane
            pure (positions, tl.full [1024] true)
          else do
            -- line 131: the `int32` scalar times the constexpr built in `int32`, wrapping
            let virtual_block_size : I32 := kv_block_size * (IntV.ofNat CP_SIZE : I32)
            -- lines 132-133: `int64` `//` and `%` (truncating) by the `int32` scalar widened
            let virtual_block_indices ← tl.idiv positions (bc (IntV.cast virtual_block_size : I64))
            let virtual_block_offsets ← tl.irem positions (bc (IntV.cast virtual_block_size : I64))
            -- line 134: `(virtual_block_offsets // CP_INTERLEAVE) % CP_SIZE == cp_rank`, the
            -- constexprs built in `int64`, the `int32` `cp_rank` widened for the comparison
            let q ← tl.idiv virtual_block_offsets (bc (IntV.ofNat CP_INTERLEAVE : I64))
            let q ← tl.irem q (bc (IntV.ofNat CP_SIZE : I64))
            let is_local := Tensor.eq q (bc (IntV.cast cp_rank : I64))
            -- line 135: the Python product of the constexprs, built in `int64`
            let rounds ← tl.idiv virtual_block_offsets
              (bc (IntV.ofNat (CP_INTERLEAVE * CP_SIZE) : I64))
            -- line 136
            let remainder ← tl.irem virtual_block_offsets (bc (IntV.ofNat CP_INTERLEAVE : I64))
            -- line 137: wrapping `int64`
            let local_offsets := rounds * bc (IntV.ofNat CP_INTERLEAVE : I64) + remainder
            -- line 138: wrapping `int64`
            let local_positions := virtual_block_indices * bc bs64 + local_offsets
            -- line 139: the `int1` scalar condition broadcast
            let local_positions := tl.where (bc sharded) local_positions positions
            -- line 140: `~sharded` of an `int1` is its negation
            let is_local := Tensor.or (bc (!sharded)) is_local
            pure (local_positions, is_local))
        -- line 142: `tl.where(mapping_enabled, local_positions // kv_block_size, 0)`: the
        -- truncating `int64` `//`, the scalar condition broadcast, `0` built in `int64`
        let block_indices := tl.where (bc mapping_enabled)
          (← tl.idiv local_positions (bc bs64)) (bc (IntV.ofInt 0 : I64))
        -- line 143
        let block_offsets ← tl.irem local_positions (bc bs64)
        -- lines 144-148: `(block_table_ptr + req_state_idx * block_table_stride) +
        -- block_indices`, the wrapping `int64` product; the mask `is_local & is_real_req`;
        -- `other=0`, an `int32` tensor in the pointer's dtype
        let block_numbers : Tensor [1024] I32 ← tl.load
          (block_table_ptr + req_state_idx * block_table_stride + block_indices)
          (some (is_local.and (bc is_real_req))) (some (tl.full [1024] (IntV.ofInt 0 : I32)))
        -- line 149: the wrapping `int32` product (`int32` tensor times `int32` scalar),
        -- widened to `int64` for the sum with the `int64` offsets
        let slot_ids : Tensor [1024] I64 :=
          Tensor.castInt (block_numbers * bc kv_block_size) + block_offsets
        -- lines 150-151: a compile-time `if`; `PAD_ID` built in `int64`
        let slot_ids := if CP_SIZE ≠ 1 then tl.where is_local slot_ids (bc PAD) else slot_ids
        -- line 153: the `int1` scalars' `&`, broadcast; `PAD_ID` built in `int64`
        let slot_ids := tl.where (bc (mapping_enabled && is_real_req)) slot_ids (bc PAD)
        -- line 154: the values are `int64`, the pointer's dtype
        tl.store (slot_mapping_ptr + offset) slot_ids (some (offset.ltS end_idx))

/-! ## The conditions -/

/-- What the host chooses: one launch through `compute_slot_mappings` (lines 233-264). The
tensors are given by their storage (`Buf`); each group `g` has its block table `bt g`
(`[max_num_reqs, max_num_blocks_g]`, its element `(r, j)` at `r * bstride g + j`), its block size
`bsz g` and its flags `en g` (`slot_mapping_enabled`) and `sh g` (`dcp_sharded`), which the
host's per-group tensors hold (`Pre`). -/
structure Inst where
  /-- `num_kv_cache_groups`: the grid's `x`. -/
  G : Nat
  /-- `num_reqs = idx_mapping.shape[0]`: the grid's `y` is `R + 1`. -/
  R : Nat
  /-- `max_num_tokens = slot_mappings.shape[1]`. -/
  N : Nat
  /-- `slot_mappings.stride(0)`. -/
  S : Nat
  /-- The constexprs `CP_SIZE`, `CP_INTERLEAVE` and the rank `cp_rank`. -/
  CP_SIZE : Nat
  CP_INTERLEAVE : Nat
  cp_rank : Nat
  /-- Group `g`'s block size (`block_sizes[g]`). -/
  bsz : Nat → Nat
  /-- Group `g`'s block-table row stride (`block_table_strides[g]`). -/
  bstride : Nat → Nat
  /-- Group `g`'s `slot_mapping_enabled[g]`. -/
  en : Nat → Bool
  /-- Group `g`'s `dcp_sharded[g]`. -/
  sh : Nat → Bool
  /-- The storage of group `g`'s block table (`int32`), whose address `block_table_ptrs[g]`
  holds. -/
  bt : Nat → Buf .i32
  /-- The storage of `idx_mapping` (`int64`). -/
  idx_mapping : Buf .i64
  /-- The storage of `query_start_loc` (`int32`). -/
  query_start_loc : Buf .i32
  /-- The storage of `positions` (`int64`). -/
  positions : Buf .i64
  /-- The storage of the per-group tables. -/
  block_table_ptrs : Buf .u64
  block_table_strides : Buf .i64
  block_sizes : Buf .i32
  slot_mapping_enabled : Buf .i1
  dcp_sharded : Buf .i1
  /-- The storage of the output `slot_mappings` (`int64`). -/
  slot_mappings : Buf .i64

/-- The signed value element `i` of an `int64` buffer holds in `m` (`0` when absent). -/
def rd64 (m : Mem) (b : Buf .i64) (i : Nat) : Int :=
  match m.read .i64 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- The signed value element `i` of an `int32` buffer holds in `m` (`0` when absent). -/
def rd32 (m : Mem) (b : Buf .i32) (i : Nat) : Int :=
  match m.read .i32 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- `query_start_loc[b]`. -/
def qsl (d : Inst) (m : Mem) (b : Nat) : Int := rd32 m d.query_start_loc b

/-- `idx_mapping[b]`: request `b`'s row of the block tables (negative for a dummy request). -/
def idx (d : Inst) (m : Mem) (b : Nat) : Int := rd64 m d.idx_mapping b

/-- `positions[t]`. -/
def pos (d : Inst) (m : Mem) (t : Nat) : Int := rd64 m d.positions t

/-- Whether group `g` stores position `p` on this rank (lines 134, 140, 146): always without
context parallelism (`CP_SIZE = 1`) or for an unsharded group; otherwise when
`(p % (B·CP_SIZE)) / CP_INTERLEAVE % CP_SIZE = cp_rank`, `B` the group's block size. -/
def cpLocal (d : Inst) (g : Nat) (p : Int) : Prop :=
  d.CP_SIZE = 1 ∨ d.sh g = false ∨
    p % ((d.bsz g * d.CP_SIZE : Nat) : Int) / (d.CP_INTERLEAVE : Int) % (d.CP_SIZE : Int) =
      (d.cp_rank : Int)

/-- Position `p`'s local position in group `g` (lines 131-139): `p` without context parallelism
or for an unsharded group; otherwise, with `V = B·CP_SIZE` and `I = CP_INTERLEAVE`,
`(p / V)·B + ((p % V) / (I·CP_SIZE))·I + (p % V) % I`. -/
def cpPos (d : Inst) (g : Nat) (p : Int) : Int :=
  if d.CP_SIZE = 1 ∨ d.sh g = false then p
  else
    p / ((d.bsz g * d.CP_SIZE : Nat) : Int) * (d.bsz g : Int) +
      p % ((d.bsz g * d.CP_SIZE : Nat) : Int) / ((d.CP_INTERLEAVE * d.CP_SIZE : Nat) : Int) *
        (d.CP_INTERLEAVE : Int) +
      p % ((d.bsz g * d.CP_SIZE : Nat) : Int) % (d.CP_INTERLEAVE : Int)

/-- The element of group `g`'s block table that token `t` of request `b` reads: row
`idx_mapping[b]`, column `cpPos / B`. -/
def btIdx (d : Inst) (m : Mem) (g b t : Nat) : Int :=
  idx d m b * (d.bstride g : Int) + cpPos d g (pos d m t) / (d.bsz g : Int)

/-- Token `t` of request `b` gets a slot in group `g`: the group uses slot mappings, the request
is real, and the token is the rank's. -/
def HasSlot (d : Inst) (m : Mem) (g b t : Nat) : Prop :=
  d.en g = true ∧ 0 ≤ idx d m b ∧ cpLocal d g (pos d m t)

instance (d : Inst) (g : Nat) (p : Int) : Decidable (cpLocal d g p) := by
  unfold cpLocal; infer_instance

instance (d : Inst) (m : Mem) (g b t : Nat) : Decidable (HasSlot d m g b t) := by
  unfold HasSlot; infer_instance

/-- **The slot** of token `t` of request `b` in group `g`: the block number times the block size
plus the offset of the local position in its block, or `-1` (`PAD_ID`) without a slot. -/
def slot (d : Inst) (m : Mem) (g b t : Nat) : Int :=
  if HasSlot d m g b t then
    rd32 m (d.bt g) (btIdx d m g b t).toNat * (d.bsz g : Int) +
      cpPos d g (pos d m t) % (d.bsz g : Int)
  else -1

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- At least one group (the host launches nothing without one), and the grid's `x` within
  CUDA's limits. -/
  G_pos : 1 ≤ d.G
  G_lt : d.G < 2 ^ 31
  /-- `num_reqs` is a batch size: at most `8192` (so the grid's `y`, `R + 1`, is below `2^16`). -/
  R_le : d.R ≤ 8192
  /-- Fewer than `2^31` programs (Triton's launcher). -/
  grid_lt : d.G * (d.R + 1) < 2 ^ 31
  /-- `max_num_tokens` is a token count: at most `2^22`; at most the output's row stride. -/
  N_le : d.N ≤ 2 ^ 22
  N_S : d.N ≤ d.S
  /-- The row stride is an `int32` runtime argument, and the last group's row offset
  `(G − 1)·S`, an `int32` product in the kernel (line 95), does not wrap: a limit of the
  kernel. -/
  S_lt : d.S < 2 ^ 31
  GS_lt : (d.G - 1) * d.S < 2 ^ 31
  /-- The context-parallel constexprs: `CP_SIZE ≥ 1`, `CP_INTERLEAVE ≥ 1`, and their product an
  `int32` Python integer (real: both at most a few thousand); `cp_rank` an `int32` argument. -/
  cp_pos : 1 ≤ d.CP_SIZE
  il_pos : 1 ≤ d.CP_INTERLEAVE
  il_cp : d.CP_INTERLEAVE * d.CP_SIZE < 2 ^ 31
  rank_lt : d.cp_rank < 2 ^ 31
  /-- Each block size is at least `1`, and the virtual block size `block_size · CP_SIZE`, an
  `int32` product in the kernel (line 131), does not wrap: a limit of the kernel. -/
  bsz_pos : ∀ g, g < d.G → 1 ≤ d.bsz g
  bsz_cp : ∀ g, g < d.G → d.bsz g * d.CP_SIZE < 2 ^ 31
  /-- Each block-table row stride is an `int64`. -/
  bstride_lt : ∀ g, g < d.G → d.bstride g < 2 ^ 63
  /-- The buffers hold the batch's entries and the groups'. -/
  idx_len : d.R ≤ d.idx_mapping.len
  qsl_len : d.R + 1 ≤ d.query_start_loc.len
  ptrs_len : d.G ≤ d.block_table_ptrs.len
  strides_len : d.G ≤ d.block_table_strides.len
  sizes_len : d.G ≤ d.block_sizes.len
  en_len : d.G ≤ d.slot_mapping_enabled.len
  sh_len : d.G ≤ d.dcp_sharded.len
  out_len : (d.G - 1) * d.S + d.N ≤ d.slot_mappings.len
  /-- The output shares no byte with an input (fresh allocations). -/
  disj_idx : Disjoint d.slot_mappings.bytes d.idx_mapping.bytes
  disj_qsl : Disjoint d.slot_mappings.bytes d.query_start_loc.bytes
  disj_pos : Disjoint d.slot_mappings.bytes d.positions.bytes
  disj_ptrs : Disjoint d.slot_mappings.bytes d.block_table_ptrs.bytes
  disj_strides : Disjoint d.slot_mappings.bytes d.block_table_strides.bytes
  disj_sizes : Disjoint d.slot_mappings.bytes d.block_sizes.bytes
  disj_en : Disjoint d.slot_mappings.bytes d.slot_mapping_enabled.bytes
  disj_sh : Disjoint d.slot_mappings.bytes d.dcp_sharded.bytes
  disj_bt : ∀ g, g < d.G → Disjoint d.slot_mappings.bytes (d.bt g).bytes

/-- The model on an instance: the kernel's arguments are the buffers' first elements, the row
length, the row stride and the rank. At `max_num_tokens = 1`, `stride = 1` or `cp_rank = 1` the
argument is the compile-time constant `1`: as an operand of `*` with the `int32` `group_id`
(line 95), of `<` with an `int32` tensor (line 105), of `==` with the `int64` tensor (line 134,
the `int32` constant `1` widened) or as a bound of `range` with the `int32` `actual_num_tokens`
(line 103), it is `1` in `int32`, the model's value. -/
def model (d : Inst) : Kernel :=
  _compute_slot_mappings_kernel (IntV.ofNat d.N) (d.idx_mapping.ptr 0) (d.query_start_loc.ptr 0)
    (d.positions.ptr 0) (d.block_table_ptrs.ptr 0) (d.block_table_strides.ptr 0)
    (d.block_sizes.ptr 0) (d.slot_mapping_enabled.ptr 0) (d.dcp_sharded.ptr 0)
    (d.slot_mappings.ptr 0) (IntV.ofNat d.S) (IntV.ofNat d.cp_rank) d.CP_SIZE d.CP_INTERLEAVE

/-- The host's launch (line 242): the grid `(num_groups, num_reqs + 1, 1)`; the source's one block
shape `[1024]`; `enable_fp_fusion` is Triton's default (`true`; no float operation); the launch
passes no option: `num_warps = 4`, `num_stages = 3` change no result, `num_ctas = 1`, no
`launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch := { grid := ![d.G, d.R + 1, 1], blocks := [[1024]] }

/-- Request `b`'s tokens are its token range, nonempty, and the request is real: its row exists
in every group's block table (lines 144-148 read it even for a token without a slot). -/
def RowOk (d : Inst) (m : Mem) (b : Nat) : Prop :=
  0 ≤ idx d m b → qsl d m b < qsl d m (b + 1) →
    ∀ g, g < d.G → idx d m b * (d.bstride g : Int) < ((d.bt g).len : Int)

/-- Token `t` of real request `b` in group `g` using slot mappings: its position is nonnegative,
and if it gets a slot, the block it reads lies in the block table and the `int32` product
`block_number · B` (line 149) does not wrap. -/
def TokOk (d : Inst) (m : Mem) (g b t : Nat) : Prop :=
  0 ≤ pos d m t ∧
    (HasSlot d m g b t →
      btIdx d m g b t < ((d.bt g).len : Int) ∧
      -2 ^ 31 ≤ rd32 m (d.bt g) (btIdx d m g b t).toNat * (d.bsz g : Int) ∧
      rd32 m (d.bt g) (btIdx d m g b t).toNat * (d.bsz g : Int) < 2 ^ 31)

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte present);
the per-group tensors hold each group's block-table address (a multiple of 16, the hint of line
70), row stride, block size and flags; `query_start_loc` is nondecreasing over the batch from a
nonnegative start, its last entry at most `max_num_tokens` and the number of positions; every
real request with tokens has a row in each block table (`RowOk`), and each token of a real
request in a group using slot mappings satisfies `TokOk`. Any other positions and any
block-table entries the kernel does not use for a slot are covered; the output's initial
contents are any. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.idx_mapping.Valid init ∧ d.query_start_loc.Valid init ∧ d.positions.Valid init ∧
    d.block_table_ptrs.Valid init ∧ d.block_table_strides.Valid init ∧
    d.block_sizes.Valid init ∧ d.slot_mapping_enabled.Valid init ∧ d.dcp_sharded.Valid init ∧
    d.slot_mappings.Valid init ∧ (∀ g, g < d.G → (d.bt g).Valid init) ∧
    (∀ g, g < d.G →
      init.read .u64 (d.block_table_ptrs.ptr g).addr = some (IntV.ofInt (d.bt g).base) ∧
      (16 : Int) ∣ (d.bt g).base ∧
      init.read .i64 (d.block_table_strides.ptr g).addr = some (IntV.ofNat (d.bstride g)) ∧
      init.read .i32 (d.block_sizes.ptr g).addr = some (IntV.ofNat (d.bsz g)) ∧
      init.read .i1 (d.slot_mapping_enabled.ptr g).addr = some (d.en g) ∧
      init.read .i1 (d.dcp_sharded.ptr g).addr = some (d.sh g)) ∧
    0 ≤ qsl d init 0 ∧ (∀ b, b < d.R → qsl d init b ≤ qsl d init (b + 1)) ∧
    qsl d init d.R ≤ (d.N : Int) ∧ qsl d init d.R ≤ (d.positions.len : Int) ∧
    (∀ b, b < d.R → RowOk d init b) ∧
    (∀ g, g < d.G → d.en g = true → ∀ b, b < d.R → 0 ≤ idx d init b →
      ∀ t : Nat, qsl d init b ≤ (t : Int) → (t : Int) < qsl d init (b + 1) → TokOk d init g b t)

/-! ## The specification -/

/-- The output elements the launch writes: `(g, t)` for every group and every `t` from
`query_start_loc[0]` to `max_num_tokens`. -/
def outW (d : Inst) (init : Mem) : Set Nat :=
  {j : Nat | ∃ g : Nat, g < d.G ∧ ∃ t : Nat, qsl d init 0 ≤ (t : Int) ∧ t < d.N ∧
    j = g * d.S + t}

/-- What the launch leaves: for every group `g`, request `b` and token `t` of `b`, element
`(g, t)` of `slot_mappings` (`g * stride + t`) holds `slot d init g b t`; every element `(g, t)`
from `query_start_loc[num_reqs]` to `max_num_tokens` holds `-1`; every byte outside these
elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ g, g < d.G → ∀ b, b < d.R → ∀ t : Nat, qsl d init b ≤ (t : Int) →
      (t : Int) < qsl d init (b + 1) →
      fin.read .i64 (d.slot_mappings.ptr ((g * d.S + t : Nat) : Int)).addr =
        some (IntV.ofInt (slot d init g b t))) ∧
    (∀ g, g < d.G → ∀ t : Nat, qsl d init d.R ≤ (t : Int) → t < d.N →
      fin.read .i64 (d.slot_mappings.ptr ((g * d.S + t : Nat) : Int)).addr = some PAD) ∧
    ∀ z, z ∉ d.slot_mappings.elems (outW d init) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_compute_slot_mappings_kernel` ends normally with
`Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- Two buffers whose byte ranges are ordered do not overlap. -/
theorem bytes_disjoint_of_le {d d' : DType} {b : Buf d} {c : Buf d'}
    (h : b.base + b.len * d.bytes ≤ c.base) : Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hcx => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hcx
  have hb := d.bytes_pos
  linarith

/-- The host's layout (lines 157-226): `G` groups with block sizes `bs g`, `rows` rows of
`cols g` blocks each (contiguous: row stride `cols g`), flags `en g`, `sh g`; `R` requests;
an output of `G × N` (stride `N`), or of a wider row `S`; the context-parallel constexprs `C`,
`I` and rank `r`. The inputs lie one after another from `2^30` with `2^20`-byte gaps, the block
tables from `2^40`, `2^36` bytes apart, the output at `2^52`. Each input takes 8 bytes per
element, so every base is a multiple of 8. -/
def mk (G R N S : Nat) (C I r : Nat) (bs cols : Nat → Nat) (en sh : Nat → Bool) (rows : Nat) :
    Inst :=
  let i0 : Int := 2 ^ 30
  let q0 : Int := i0 + R * 8 + 2 ^ 20
  let p0 : Int := q0 + (R + 1) * 8 + 2 ^ 20
  let t0 : Int := p0 + N * 8 + 2 ^ 20
  { G := G, R := R, N := N, S := S, CP_SIZE := C, CP_INTERLEAVE := I, cp_rank := r,
    bsz := bs, bstride := cols, en := en, sh := sh,
    bt := fun g => ⟨2 ^ 40 + (g : Int) * 2 ^ 36, rows * cols g⟩,
    idx_mapping := ⟨i0, R⟩, query_start_loc := ⟨q0, R + 1⟩, positions := ⟨p0, N⟩,
    block_table_ptrs := ⟨t0, G⟩, block_table_strides := ⟨t0 + G * 8 + 2 ^ 20, G⟩,
    block_sizes := ⟨t0 + 2 * (G * 8 + 2 ^ 20), G⟩,
    slot_mapping_enabled := ⟨t0 + 3 * (G * 8 + 2 ^ 20), G⟩,
    dcp_sharded := ⟨t0 + 4 * (G * 8 + 2 ^ 20), G⟩,
    slot_mappings := ⟨2 ^ 52, G * S⟩ }

theorem mk_ok (G R N S C I r : Nat) (bs cols : Nat → Nat) (en sh : Nat → Bool) (rows : Nat)
    (h1 : 1 ≤ G) (h2 : G ≤ 2 ^ 10) (h3 : R ≤ 8192) (h4 : N ≤ 2 ^ 22) (h5 : N ≤ S)
    (h6 : S < 2 ^ 31) (h7 : (G - 1) * S < 2 ^ 31) (h8 : 1 ≤ C) (h9 : 1 ≤ I)
    (h10 : I * C < 2 ^ 31) (h11 : r < 2 ^ 31) (h12 : ∀ g, g < G → 1 ≤ bs g)
    (h13 : ∀ g, g < G → bs g * C < 2 ^ 31) (h14 : ∀ g, g < G → cols g ≤ 2 ^ 20)
    (h15 : rows ≤ 2 ^ 13) :
    (mk G R N S C I r bs cols en sh rows).ok := by
  have hS : (G - 1) * S + N ≤ G * S := by
    have : G = (G - 1) + 1 := by omega
    calc (G - 1) * S + N ≤ (G - 1) * S + S := by omega
      _ = G * S := by
        conv_rhs => rw [this]
        ring
  have hlt : (G : Int) * 8 + 2 ^ 20 ≤ 2 ^ 24 := by
    have : (G : Int) ≤ 2 ^ 10 := by exact_mod_cast h2
    linarith
  have hR : (R : Int) ≤ 8192 := by exact_mod_cast h3
  have hN : (N : Int) ≤ 2 ^ 22 := by exact_mod_cast h4
  have hgrid : G * (R + 1) < 2 ^ 31 := by
    have : G * (R + 1) ≤ 2 ^ 10 * 8193 := Nat.mul_le_mul h2 (by omega)
    omega
  refine ⟨h1, (show G < 2 ^ 31 by omega), h3, hgrid, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13,
    fun g hg => ?_,
    le_refl _, le_refl _, le_refl _, le_refl _, le_refl _, le_refl _, le_refl _, hS,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, fun g hg => ?_⟩
  · have := h14 g hg; show cols g < 2 ^ 63; omega
  all_goals try
    (refine (bytes_disjoint_of_le ?_).symm
     simp only [mk, DType.bytes]; push_cast; nlinarith)
  refine (bytes_disjoint_of_le ?_).symm
  have hg2 : g < G := hg
  simp only [mk, DType.bytes]
  have hc : ((rows * cols g : Nat) : Int) ≤ 2 ^ 33 := by
    have := h14 g hg2
    have : rows * cols g ≤ 2 ^ 13 * 2 ^ 20 := Nat.mul_le_mul h15 this
    have h' : ((rows * cols g : Nat) : Int) ≤ ((2 ^ 13 * 2 ^ 20 : Nat) : Int) := by exact_mod_cast this
    norm_num at h' ⊢
    linarith
  have hg' : (g : Int) ≤ 2 ^ 10 := by exact_mod_cast (show g ≤ 2 ^ 10 by omega)
  push_cast at hc ⊢
  nlinarith

/-- The memory whose every byte is present: `1` at address `c`, `0` elsewhere. -/
def oneAt (c : Int) : Mem := fun a => some (if a = c then 1 else 0)

/-- A buffer of aligned bounds is valid in `oneAt c`: every byte is present. -/
theorem valid_oneAt {d : DType} (b : Buf d) (c : Int) (h0 : 0 ≤ b.base)
    (h1 : b.base + b.len * d.bytes ≤ 2 ^ 64) (h2 : b.base % d.bytes = 0) : b.Valid (oneAt c) :=
  ⟨h0, h1, h2, fun _ _ => by simp [oneAt]⟩

/-- A covered instance and memory: one group of block size `1`, no request (only the padding
program runs), the output of `2^20` elements. -/
def sat : Inst :=
  { mk 1 0 (2 ^ 20) (2 ^ 20) 1 1 0 (fun _ => 1) (fun _ => 0) (fun _ => false) (fun _ => false) 0
    with bt := fun _ => ⟨0, 0⟩ }

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨sat, oneAt sat.block_sizes.base, ?_, ?_⟩
  · refine ⟨by decide, by decide, by decide, by decide, by decide, le_refl _, by decide, by decide,
      le_refl _,
      le_refl _, by decide, by decide, fun _ _ => le_refl _,
      fun _ _ => by simp [sat, mk], fun _ _ => by simp [sat, mk], le_refl _, le_refl _, le_refl _, le_refl _, le_refl _, le_refl _,
      le_refl _, by decide, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, fun g hg => ?_⟩
    all_goals
      (refine (bytes_disjoint_of_le ?_).symm
       simp only [sat, mk, DType.bytes]; norm_num)
  · have hq0 : qsl sat (oneAt sat.block_sizes.base) 0 = 0 := by decide
    have hv : ∀ {d : DType} (b : Buf d), 0 ≤ b.base → b.base + b.len * d.bytes ≤ 2 ^ 64 →
        b.base % d.bytes = 0 → b.Valid (oneAt sat.block_sizes.base) :=
      fun b h0 h1 h2 => valid_oneAt b _ h0 h1 h2
    refine ⟨hv _ (by decide) (by decide) (by decide), hv _ (by decide) (by decide) (by decide),
      hv _ (by decide) (by decide) (by decide), hv _ (by decide) (by decide) (by decide),
      hv _ (by decide) (by decide) (by decide), hv _ (by decide) (by decide) (by decide),
      hv _ (by decide) (by decide) (by decide), hv _ (by decide) (by decide) (by decide),
      hv _ (by decide) (by decide) (by decide), fun g _ => hv (sat.bt g) (show (0 : Int) ≤ 0 by norm_num)
        (show (0 : Int) + ((0 : Nat) : Int) * 4 ≤ 2 ^ 64 by norm_num) (show (0 : Int) % 4 = 0 by norm_num),
      fun g hg => ?_, by rw [hq0], fun b hb => absurd hb (Nat.not_lt_zero b),
      by show qsl sat _ 0 ≤ _; rw [hq0]; decide, by show qsl sat _ 0 ≤ _; rw [hq0]; decide,
      fun b hb => absurd hb (Nat.not_lt_zero b), fun g hg hen => ?_⟩
    · obtain rfl : g = 0 := by have : g < 1 := hg; omega
      exact ⟨by decide, by decide, by decide, by decide, by decide, by decide⟩
    · obtain rfl : g = 0 := by have : g < 1 := hg; omega
      exact absurd hen (by decide)

/-! ## The setup's launches

The launches of `SETUP.md`, each through `mk` (the host's contiguous layout): the groups' block
sizes, blocks per row, flags; the requests; the output's row; the context-parallel constexprs.
`Inst.ok` does not depend on the batch's token counts or positions, which are `Pre`'s. -/

/-- vLLM's model runner, one attention group: block size 16, 1024 rows of 8192 blocks, 256
requests, 8192 tokens. -/
def setup_runner_one_group : Inst :=
  mk 1 256 8192 8192 1 1 0 (fun _ => 16) (fun _ => 8192) (fun _ => true) (fun _ => true) 1024
theorem setup_runner_one_group_ok : setup_runner_one_group.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)

/-- A hybrid model's groups: full attention (block size 16), sliding window (32), a recurrent group
without slot mappings (block size 1056, one block per row); 512 rows, 300 requests, 16384
tokens. -/
def setup_runner_hybrid : Inst :=
  mk 3 300 16384 16384 1 1 0 (fun g => [16, 32, 1056].getD g 1) (fun g => [2048, 1024, 1].getD g 1)
    (fun g => [true, true, false].getD g false) (fun g => [true, true, false].getD g false) 512
theorem setup_runner_hybrid_ok : setup_runner_hybrid.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)

/-- Decode context parallelism over `C` ranks, rank `r`, interleave `I`, `G` groups of block size
64 with 512 blocks per row (the first sharded, the others as `sh` says); 256 rows, 100 requests,
8192 tokens. -/
def setupDcp (G C r I : Nat) (sh : Nat → Bool) : Inst :=
  mk G 100 8192 8192 C I r (fun _ => 64) (fun _ => 512) (fun _ => true) sh 256

/-- Two ranks, rank 0, interleave 1. -/
def setup_dcp2_rank0 : Inst := setupDcp 1 2 0 1 (fun _ => true)
theorem setup_dcp2_rank0_ok : setup_dcp2_rank0.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)
/-- Two ranks, rank 1, interleave 1. -/
def setup_dcp2_rank1 : Inst := setupDcp 1 2 1 1 (fun _ => true)
theorem setup_dcp2_rank1_ok : setup_dcp2_rank1.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)
/-- Eight ranks, rank 5, interleave 64, a second group replicated (unsharded). -/
def setup_dcp8_rank5_interleave64 : Inst := setupDcp 2 8 5 64 (fun g => decide (g = 0))
theorem setup_dcp8_rank5_interleave64_ok : setup_dcp8_rank5_interleave64.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)
/-- Four ranks, rank 1, interleave 16, two sharded groups. -/
def setup_dcp4_rank1_interleave16 : Inst := setupDcp 2 4 1 16 (fun _ => true)
theorem setup_dcp4_rank1_interleave16_ok : setup_dcp4_rank1_interleave16.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)

/-- A dummy batch (`InputBatch.make_dummy`): 100 requests over 4096 tokens, two groups (block
sizes 16 and 64), 128 rows; the same layout serves the padded batch with dummy requests (64
requests). -/
def setup_dummy_batch : Inst :=
  mk 2 100 4096 4096 1 1 0 (fun g => [16, 64].getD g 1) (fun g => [256, 64].getD g 1)
    (fun _ => true) (fun _ => true) 128
theorem setup_dummy_batch_ok : setup_dummy_batch.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)
def setup_dummy_requests : Inst :=
  mk 2 64 4096 4096 1 1 0 (fun g => [16, 64].getD g 1) (fun g => [256, 64].getD g 1)
    (fun _ => true) (fun _ => true) 128
theorem setup_dummy_requests_ok : setup_dummy_requests.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)

/-- A speculator's output `[G, max_num_tokens + max_num_reqs]` (4096 + 256), two groups, 200
requests. -/
def setup_speculator_expanded : Inst :=
  mk 2 200 (4096 + 256) (4096 + 256) 1 1 0 (fun g => [16, 32].getD g 1)
    (fun g => [1024, 512].getD g 1) (fun _ => true) (fun _ => true) 256
theorem setup_speculator_expanded_ok : setup_speculator_expanded.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)

/-- The source's own test (source lines 299-326): two groups (block sizes 16 and 64, the second
unsharded), 8 rows, 4 requests, 4096 tokens, at `CP_SIZE` 1, 2 (rank 1) and 4 (rank 2,
interleave 8). -/
def setupSourceTest (C r I : Nat) : Inst :=
  mk 2 4 4096 4096 C I r (fun g => [16, 64].getD g 1) (fun g => [64, 16].getD g 1)
    (fun _ => true) (fun g => decide (g = 0)) 8
def setup_source_test_cp1 : Inst := setupSourceTest 1 0 1
theorem setup_source_test_cp1_ok : setup_source_test_cp1.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)
def setup_source_test_cp2 : Inst := setupSourceTest 2 1 1
theorem setup_source_test_cp2_ok : setup_source_test_cp2.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)
def setup_source_test_cp4 : Inst := setupSourceTest 4 2 8
theorem setup_source_test_cp4_ok : setup_source_test_cp4.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)

/-- The real ranges' edge: 8192 requests, `2^22` tokens, eight groups. -/
def setup_edge_max : Inst :=
  mk 8 8192 (2 ^ 22) (2 ^ 22) 1 1 0 (fun _ => 16) (fun _ => 2 ^ 18) (fun _ => true)
    (fun _ => true) 8192
theorem setup_edge_max_ok : setup_edge_max.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by decide) (by decide) (by decide) (by norm_num)

end Claims.SlotMappings
