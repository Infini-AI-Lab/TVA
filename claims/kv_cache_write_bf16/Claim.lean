import TVA

/-!
# `kv_cache_write_bf16`: KV cache write, flash layout, vLLM

The kernel is `reshape_and_cache_kernel_flash` of `source/reshape_and_cache_flash.py` (lines
73-195; it reads the module-level names `triton` and `tl`, lines 60-61). The host
`reshape_and_cache_flash` (lines 198-298) launches it with one program per (token, tile of
`TILE_SIZE` of the token's `num_heads * head_size` values), on `key` and `value`
(`[num_tokens, num_heads, head_size]`, each token's values contiguous, any token stride), the
caches `key_cache` and `value_cache` (`[num_blocks, block_size, num_heads, head_size]`, last
dimension contiguous, the other strides those of the key cache), and the `int64` `slot_mapping`,
in three configurations (`Mode`): `bfloat16` inputs and caches (`kv_cache_dtype = "auto"`);
`bfloat16` inputs and `float8e4nv` caches (`kv_cache_dtype = "fp8"`, the inputs divided by the
`float32` scales `k_scale`, `v_scale`); `float8e4nv` inputs and caches (`"fp8"`, the inputs stored
as they are). The caches are the caller's: in both of vLLM's layouts they are two views of ONE
allocation whose byte ranges overlap (the flash backends' `kv[:, 0]`, `kv[:, 1]`; the Triton
backend's `kv_cache.transpose(1, 2).split(head_size, -1)`, where keys and values interleave).
The covered configuration is `SETUP.md`'s: `USE_HEAD_MAJOR_LAYOUT = False`,
`INTERLEAVED_V_PACK_FACTOR = 0`, `x = 1`, `dim_stride_k = dim_stride_v = 0`, and
`FP8_KV_CACHE = False` with a `bfloat16` cache, `True` with a `float8e4nv` cache.

What is claimed: every covered launch ends normally; for every token `t` with a slot `s ≥ 0`,
every head `h` and component `e`, the key cache's element `(s // block_size, s % block_size, h, e)`
holds what `Stored` says of `key[t, h, e]` (the value itself, bit for bit, except with `bfloat16`
inputs and an fp8 cache: the saturating fp8 rounding of a `float32` within `DivFull`'s bound of
`key[t, h, e] / k_scale`), and the value cache's likewise of `value[t, h, e]` (with `v_scale`);
every byte outside those elements is unchanged.

The conditions: `TILE_SIZE` a power of two, at most `2^20`; at most `2^22` tokens and the grid
within CUDA's limits; `num_heads·head_size < 2^31`, the strides and the buffers' lengths below
`2^63`; the buffers hold every element the kernel touches, and are valid; the cache view is
non-overlapping; the caches share no byte with `key`, `value`, `slot_mapping` and the scales, and no
element of the key cache shares a byte with one of the value cache; every slot is negative
(padding) or a slot of the cache, and the non-negative slots of two tokens differ. Any input values
and scales are covered.

The fp8 configuration rests on the device facts for `/` on `float32` and the conversion
`float32 → float8e4nv` (`Device`).
-/

namespace Claims.KvCacheWrite

open TVA

/-! ## The model -/

/-- The covered configurations: the dtypes of `key` and `value` and of the caches, and the
constexpr `FP8_KV_CACHE` the host sets (`_is_quantized_kv_cache(kv_cache_dtype)`, source line
260). -/
inductive Mode
  /-- `kv_cache_dtype = "auto"`: `bfloat16` inputs and caches, `FP8_KV_CACHE = False`. -/
  | bf16
  /-- `kv_cache_dtype = "fp8"`: `bfloat16` inputs, `float8e4nv` caches (the host's view as
  `torch.float8_e4m3fn`, source lines 247-259), `FP8_KV_CACHE = True`. -/
  | fp8
  /-- `kv_cache_dtype = "fp8"` with `float8e4nv` inputs and caches, `FP8_KV_CACHE = True`. -/
  | fp8in
  deriving DecidableEq

/-- The dtype of `key` and `value`. -/
abbrev Mode.kd : Mode → DType
  | .fp8in => .f8e4nv
  | _ => .bf16

/-- The dtype of the caches. -/
abbrev Mode.cd : Mode → DType
  | .bf16 => .bf16
  | _ => .f8e4nv

set_option linter.unusedVariables false in
/-- `reshape_and_cache_kernel_flash` (source lines 73-195), in configuration `m`. `key` and
`value` are `*m.kd` pointers, the caches `*m.cd` pointers, `slot_mapping` an `*i64` pointer,
`k_scale` and `v_scale` `*fp32` pointers (read only in `Mode.fp8`); the seven strides are
`tl.int64`-annotated runtime arguments (never constants, `TVA.Triton.Promote`).
`num_heads`, `head_size`, `block_size` and `TILE_SIZE` are `tl.constexpr`s (Lean parameters); the
constexprs `x`, `USE_HEAD_MAJOR_LAYOUT` and `INTERLEAVED_V_PACK_FACTOR` are used only in the
compile-time condition of line 118 and the branch it cuts; `FP8_KV_CACHE` (set by `m`) and the
dtype predicates `key_load.dtype.is_fp8()`, `value_load.dtype.is_fp8()` (compile-time `bool`s of
the loads' dtype `m.kd`, `TVA.Triton.Promote`) decide lines 164-183, so the model
gives lines 161-194 once per configuration. -/
def reshape_and_cache_kernel_flash (m : Mode)
    (key_ptr value_ptr : Ptr m.kd) (key_cache_ptr value_cache_ptr : Ptr m.cd)
    (slot_mapping_ptr : Ptr .i64) (k_scale v_scale : Ptr .f32)
    (key_stride value_stride block_stride head_stride dim_stride_k dim_stride_v page_stride : I64)
    (num_heads head_size block_size TILE_SIZE : Nat) : Kernel where
  body ctx := do
    -- line 103: axis 0 of the grid, an `int32`
    let token_idx := tl.program_id ctx 0
    -- line 104: one `int64` element of `slot_mapping`, `.to(tl.int64)` (the identity)
    let slot_idx := (← tl.convert .i64 .i64
      (Tensor.scalar (← tl.load1 (slot_mapping_ptr + token_idx)))).item
    -- lines 105-107: a runtime `if` on the `int64` comparison with `0` (built in `int64`); the
    -- programs of padding tokens return; only one branch returns, so the rest stays in the subset
    if IntV.lt slot_idx (IntV.ofInt 0 : I64) then
      return ()
    else do
      -- lines 109-110: `int64` `//` and `%` (truncating) by the constexpr built in `int64`
      let block_idx :=
        (← tl.idiv (Tensor.scalar slot_idx) (Tensor.scalar (IntV.ofNat block_size : I64))).item
      let block_offset :=
        (← tl.irem (Tensor.scalar slot_idx) (Tensor.scalar (IntV.ofNat block_size : I64))).item
      -- lines 112-114: the tile of `TILE_SIZE` positions, `int32` (wrapping)
      let tile_i := tl.program_id ctx 1
      let tile_offs := tl.arange 0 TILE_SIZE
      let tile_pos : Tensor [TILE_SIZE] I32 :=
        tile_i * (IntV.ofNat TILE_SIZE : I32) + tile_offs
      -- lines 115-116: the `int32` `token_idx` promoted to `int64`, the `int64` products
      let src_key_idx : I64 :=
        (← tl.convert .i32 .i64 (Tensor.scalar token_idx)).item * key_stride
      let src_value_idx : I64 :=
        (← tl.convert .i32 .i64 (Tensor.scalar token_idx)).item * value_stride
      -- lines 118-148: `if USE_HEAD_MAJOR_LAYOUT:` on the constexpr `False`: not visited
      -- lines 150-151: `int32` `//` and `%` by the constexpr `head_size` built in `int32`
      let cur_head ← tl.idiv tile_pos (tl.full [TILE_SIZE] (IntV.ofNat head_size : I32))
      let cur_dim ← tl.irem tile_pos (tl.full [TILE_SIZE] (IntV.ofNat head_size : I32))
      -- lines 152-157: `((block_idx * block_stride + block_offset * page_stride) + cur_head *
      -- head_stride) + cur_dim`: the `int64` scalar sum, then the `int32` blocks promoted to
      -- `int64` (broadcasting the scalar), wrapping `int64` arithmetic
      let head64 ← tl.convert .i32 .i64 cur_head
      let dim64 ← tl.convert .i32 .i64 cur_dim
      let tgt_idx_k : Tensor [TILE_SIZE] I64 :=
        block_idx * block_stride + block_offset * page_stride + head64 * head_stride + dim64
      -- line 158
      let tgt_idx_v : Tensor [TILE_SIZE] I64 := tgt_idx_k
      -- the mask of lines 162, 173, 188 and 193: `tile_pos < (num_heads * head_size)`, the
      -- constexpr product an `int32` constant (`< 2^31`, `Inst.ok`)
      let mask : Tensor [TILE_SIZE] Bool :=
        tile_pos.ltS (IntV.ofNat (num_heads * head_size) : I32)
      -- lines 161-195, by the configuration
      (match m, key_ptr, value_ptr, key_cache_ptr, value_cache_ptr with
      | .bf16, key_ptr, value_ptr, key_cache_ptr, value_cache_ptr => do
        -- lines 161-163: `(key_ptr + src_key_idx) + tile_pos`; a masked load without `other`
        let key_load ← tl.load (key_ptr + src_key_idx + tile_pos) (some mask)
        -- lines 164, 168-169: `if FP8_KV_CACHE:` on the constexpr `False`: `key_tile = key_load`
        let key_tile := key_load
        -- lines 172-174
        let value_load ← tl.load (value_ptr + src_value_idx + tile_pos) (some mask)
        -- lines 175, 182-183: as lines 164, 168-169: `value_tile = value_load`
        let value_tile := value_load
        -- lines 185-189: the values are already `bfloat16`, the pointer's dtype
        tl.store (key_cache_ptr + tgt_idx_k) key_tile (some mask)
        -- lines 190-194
        tl.store (value_cache_ptr + tgt_idx_v) value_tile (some mask)
        -- line 195
        return ()
      | .fp8, key_ptr, value_ptr, key_cache_ptr, value_cache_ptr => do
        -- lines 161-163
        let key_load ← tl.load (key_ptr + src_key_idx + tile_pos) (some mask)
        -- lines 164, 167: `if FP8_KV_CACHE:` on the constexpr `True`; `key_load.dtype.is_fp8()` is
        -- `False` (`bfloat16`), so the `else` side: `key_load / tl.load(k_scale)`, the scalar
        -- `float32` load, the `bfloat16` tile widened to `float32` (exact; `/` computes in the
        -- promoted `float32`), the scalar broadcast, the platform's division
        let key_tile ← tl.div (← tl.cvt (g := .binary32) key_load) (bc (← tl.load1 k_scale))
        -- lines 172-174
        let value_load ← tl.load (value_ptr + src_value_idx + tile_pos) (some mask)
        -- lines 175-176, 178, 181: `value_load.dtype.is_fp8()` is `False`: `value_load /
        -- tl.load(v_scale)`, as line 167
        let value_tile ← tl.div (← tl.cvt (g := .binary32) value_load) (bc (← tl.load1 v_scale))
        -- lines 185-189: the `float32` tile stored to the `float8e4nv` pointer: converted by the
        -- saturating rounding to nearest even (`tl.cvt8 .rn`, `TVA.Triton.Memory`)
        tl.store (key_cache_ptr + tgt_idx_k) (← tl.cvt8 (g := .e4m3) key_tile .rn) (some mask)
        -- lines 190-194
        tl.store (value_cache_ptr + tgt_idx_v) (← tl.cvt8 (g := .e4m3) value_tile .rn) (some mask)
        -- line 195
        return ()
      | .fp8in, key_ptr, value_ptr, key_cache_ptr, value_cache_ptr => do
        -- lines 161-163
        let key_load ← tl.load (key_ptr + src_key_idx + tile_pos) (some mask)
        -- lines 164, 167: `FP8_KV_CACHE` is `True`; `key_load.dtype.is_fp8()` is `True`
        -- (`float8e4nv`): `key_tile = key_load` (the `else` side is not visited)
        let key_tile := key_load
        -- lines 172-174
        let value_load ← tl.load (value_ptr + src_value_idx + tile_pos) (some mask)
        -- lines 175-177: `value_load.dtype.is_fp8()` is `True`: `value_tile = value_load`
        let value_tile := value_load
        -- lines 185-189: the values are already `float8e4nv`, the pointer's dtype
        tl.store (key_cache_ptr + tgt_idx_k) key_tile (some mask)
        -- lines 190-194
        tl.store (value_cache_ptr + tgt_idx_v) value_tile (some mask)
        -- line 195
        return ())

/-! ## The conditions -/

/-- What the host chooses: one launch through `reshape_and_cache_flash` on the covered
configuration. The tensors are given by their storage (`Buf`, from each tensor's first element)
and their strides in elements: `key` and `value` have each token's `num_heads * head_size` values
contiguous; the caches have their last dimension contiguous and the value cache has the key
cache's strides (as the kernel assumes, SETUP.md), so one set of cache strides describes both. -/
structure Inst where
  /-- The configuration: the dtypes and `FP8_KV_CACHE`. -/
  mode : Mode
  /-- `num_tokens`: the grid's `x` (`slot_mapping.shape[0]`). -/
  num_tokens : Nat
  /-- `num_heads`, `head_size` (`key.shape[1]`, `key.shape[2]`). -/
  num_heads : Nat
  head_size : Nat
  /-- `block_size`, `num_blocks` (`key_cache.shape[1]`, `key_cache.shape[0]`). -/
  block_size : Nat
  num_blocks : Nat
  /-- `TILE_SIZE` (the host's `min(2048, next_power_of_2(num_heads * head_size))`). -/
  TILE_SIZE : Nat
  /-- `key.stride(0)`, `value.stride(0)`. -/
  key_stride : Nat
  value_stride : Nat
  /-- `key_cache.stride(0)`, `stride(1)`, `stride(2)` (the value cache's too). -/
  block_stride : Nat
  page_stride : Nat
  head_stride : Nat
  /-- The storage of `key`, `value` (`mode.kd`) and of `key_cache`, `value_cache` (`mode.cd`). -/
  key : Buf mode.kd
  value : Buf mode.kd
  key_cache : Buf mode.cd
  value_cache : Buf mode.cd
  /-- The storage of `slot_mapping` (`int64`). -/
  slot_mapping : Buf .i64
  /-- The one-element `float32` tensors `k_scale`, `v_scale` (read in `Mode.fp8` only). -/
  k_scale : Buf .f32
  v_scale : Buf .f32

/-- `num_heads * head_size`: the values of one token's key (and value) row. -/
def numVals (d : Inst) : Nat := d.num_heads * d.head_size

/-- `num_blocks * block_size`: the cache's slots. -/
def numSlots (d : Inst) : Nat := d.num_blocks * d.block_size

/-- The grid's `y`: `cdiv(num_heads * head_size, TILE_SIZE)`. -/
def numTiles (d : Inst) : Nat := (numVals d + d.TILE_SIZE - 1) / d.TILE_SIZE

/-- Element `(t, p)` of `key` (token `t`, position `p = h * head_size + e` of its row). -/
def keyIdx (d : Inst) (t p : Nat) : Nat := t * d.key_stride + p

/-- Element `(t, p)` of `value`. -/
def valueIdx (d : Inst) (t p : Nat) : Nat := t * d.value_stride + p

/-- Element `(b, r, h, e)` of either cache: block `b`, row `r`, head `h`, component `e`. -/
def kvIdx (d : Inst) (b r h e : Nat) : Nat :=
  b * d.block_stride + r * d.page_stride + h * d.head_stride + e

/-- The `int64` value of `slot_mapping[t]` in memory `m` (`0` when absent). -/
def slotAt (d : Inst) (m : Mem) (t : Nat) : Int :=
  match m.read .i64 (d.slot_mapping.ptr (t : Int)).addr with
  | some v => v.toInt
  | none => 0

/-- The cache element that receives component `e` of head `h` of token `t`: block
`slot // block_size`, row `slot % block_size`. -/
def dstIdx (d : Inst) (m : Mem) (t h e : Nat) : Nat :=
  kvIdx d ((slotAt d m t).toNat / d.block_size) ((slotAt d m t).toNat % d.block_size) h e

/-- The cache elements a launch from `m` writes: those of every non-padding token's row. -/
def dstSet (d : Inst) (m : Mem) : Set Nat :=
  {k | ∃ (t h e : Nat), t < d.num_tokens ∧ 0 ≤ slotAt d m t ∧ h < d.num_heads ∧ e < d.head_size ∧
    k = dstIdx d m t h e}

/-- The bytes a launch from `m` writes: the written elements of both caches. -/
def writeSet (d : Inst) (m : Mem) : Set Int :=
  d.key_cache.elems (dstSet d m) ∪ d.value_cache.elems (dstSet d m)

/-- The instances covered: SETUP.md's configuration with any shapes; buffers holding every
element the kernel addresses; a cache view that is non-overlapping (`kv_inj`); and the key and
value caches' elements in different bytes (`disj_caches`), which both of vLLM's layouts meet
although the two caches' byte ranges overlap. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `TILE_SIZE` is a power of two of at most `2^20` (the host's is, and `tl.arange` accepts
  nothing else). -/
  tile_pow2 : ∃ e, d.TILE_SIZE = 2 ^ e
  tile_le : d.TILE_SIZE ≤ 2 ^ 20
  /-- The grid `(num_tokens, cdiv(n, TILE_SIZE))` within CUDA's limits, fewer than `2^31`
  programs (`Launch.Valid`). -/
  tokens_lt : d.num_tokens < 2 ^ 31
  /-- `num_tokens`, the tokens of the batch, is a context length: at most `2^22`. -/
  tokens_ctx : d.num_tokens ≤ 2 ^ 22
  tiles_lt : numTiles d < 2 ^ 16
  programs_lt : d.num_tokens * numTiles d < 2 ^ 31
  /-- `num_heads * head_size < 2^31`: the mask's constant is an `int32`, and the `int32` tile
  positions do not wrap (a limit of the kernel, README "Exclusions"). -/
  vals_lt : numVals d < 2 ^ 31
  /-- `head_size < 2^31`: the divisor of lines 150-151 is an `int32` constant, whatever
  `num_heads` (README "Exclusions"). -/
  hs_lt : d.head_size < 2 ^ 31
  /-- `block_size` is an `int64` constant. -/
  block_size_lt : d.block_size < 2 ^ 63
  /-- `slot_mapping` has an element per token. -/
  slot_len : d.num_tokens ≤ d.slot_mapping.len
  /-- `key` and `value` hold every element the kernel reads. -/
  key_len : ∀ t p, t < d.num_tokens → p < numVals d → keyIdx d t p < d.key.len
  value_len : ∀ t p, t < d.num_tokens → p < numVals d → valueIdx d t p < d.value.len
  /-- The caches hold every element of the `[num_blocks, block_size, num_heads, head_size]`
  view. -/
  kv_len : ∀ b r h e, b < d.num_blocks → r < d.block_size → h < d.num_heads → e < d.head_size →
    kvIdx d b r h e < d.key_cache.len ∧ kvIdx d b r h e < d.value_cache.len
  /-- The cache view is non-overlapping: distinct indices of the view name distinct elements (as
  in every view vLLM allocates). -/
  kv_inj : ∀ b r h e b' r' h' e', b < d.num_blocks → r < d.block_size → h < d.num_heads →
    e < d.head_size → b' < d.num_blocks → r' < d.block_size → h' < d.num_heads →
    e' < d.head_size → kvIdx d b r h e = kvIdx d b' r' h' e' → b = b' ∧ r = r' ∧ h = h' ∧ e = e'
  /-- The strides are `int64` values. -/
  strides_lt : d.key_stride < 2 ^ 63 ∧ d.value_stride < 2 ^ 63 ∧ d.block_stride < 2 ^ 63 ∧
    d.page_stride < 2 ^ 63 ∧ d.head_stride < 2 ^ 63
  /-- `key`, `value` and the caches have fewer than `2^63` elements, so the `int64` offsets of an
  enabled lane (`token_idx * key_stride`, line 115; `block_idx * block_stride + block_offset *
  page_stride + cur_head * head_stride + cur_dim`, lines 152-157), sums of non-negative terms at
  most the element's index (`key_len`, `value_len`, `kv_len`), do not wrap (README
  "Exclusions"). -/
  lens_lt : d.key.len < 2 ^ 63 ∧ d.value.len < 2 ^ 63 ∧ d.key_cache.len < 2 ^ 63 ∧
    d.value_cache.len < 2 ^ 63
  /-- The caches' storage shares no byte with the inputs' (the kernel only reads `key`, `value`
  and `slot_mapping`, which vLLM keeps apart from the cache). -/
  disj_kc_key : Disjoint d.key_cache.bytes d.key.bytes
  disj_kc_value : Disjoint d.key_cache.bytes d.value.bytes
  disj_kc_slot : Disjoint d.key_cache.bytes d.slot_mapping.bytes
  disj_vc_key : Disjoint d.value_cache.bytes d.key.bytes
  disj_vc_value : Disjoint d.value_cache.bytes d.value.bytes
  disj_vc_slot : Disjoint d.value_cache.bytes d.slot_mapping.bytes
  /-- The two caches' elements occupy different bytes: every element of the key cache's view
  shares no byte with any element of the value cache's view. The caches' storage ranges may
  overlap (both of vLLM's layouts interleave them in one allocation). -/
  disj_caches : ∀ b r h e b' r' h' e', b < d.num_blocks → r < d.block_size → h < d.num_heads →
    e < d.head_size → b' < d.num_blocks → r' < d.block_size → h' < d.num_heads →
    e' < d.head_size →
    Disjoint (Mem.span d.mode.cd (d.key_cache.ptr (kvIdx d b r h e : Int)).addr)
      (Mem.span d.mode.cd (d.value_cache.ptr (kvIdx d b' r' h' e' : Int)).addr)
  /-- With an fp8 cache and `bfloat16` inputs (`Mode.fp8`) the kernel reads `k_scale` and
  `v_scale`: each holds its element, and the caches share no byte with them (vLLM's
  `layer._k_scale`, `layer._v_scale` are tensors of their own). -/
  scales : d.mode = .fp8 → 1 ≤ d.k_scale.len ∧ 1 ≤ d.v_scale.len ∧
    Disjoint d.key_cache.bytes d.k_scale.bytes ∧ Disjoint d.key_cache.bytes d.v_scale.bytes ∧
    Disjoint d.value_cache.bytes d.k_scale.bytes ∧ Disjoint d.value_cache.bytes d.v_scale.bytes

/-- The model on an instance: the configuration, the buffers' first elements, the strides,
`dim_stride_k = dim_stride_v = 0` (the host's on this path) and the constexprs. -/
def model (d : Inst) : Kernel :=
  reshape_and_cache_kernel_flash d.mode (d.key.ptr 0) (d.value.ptr 0) (d.key_cache.ptr 0)
    (d.value_cache.ptr 0) (d.slot_mapping.ptr 0) (d.k_scale.ptr 0) (d.v_scale.ptr 0)
    (IntV.ofNat d.key_stride) (IntV.ofNat d.value_stride) (IntV.ofNat d.block_stride)
    (IntV.ofNat d.head_stride) (IntV.ofNat 0) (IntV.ofNat 0)
    (IntV.ofNat d.page_stride) d.num_heads d.head_size d.block_size d.TILE_SIZE

/-- The host's launch (source lines 262-298): the grid `(num_tokens, cdiv(n, TILE_SIZE), 1)`, the
source's block shape `[TILE_SIZE]`; `enable_fp_fusion` Triton's default (`true`); `num_warps = 16`
and `num_stages = 10` change no result; `num_ctas = 1`, no `launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch :=
  { grid := ![d.num_tokens, numTiles d, 1], fusion := true, blocks := [[d.TILE_SIZE]] }

/-- The inputs covered: `key`, `value`, both caches and `slot_mapping` are valid (aligned, inside
`[0, 2^64)`, every byte present), and in `Mode.fp8` `k_scale` and `v_scale` too; every slot is
negative (padding; vLLM's `-1`) or a slot of the cache, and the non-negative slots of two tokens
differ (vLLM gives each token its own slot). Any input values, NaN and infinities included, and
any scales. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.key.Valid init ∧ d.value.Valid init ∧ d.key_cache.Valid init ∧ d.value_cache.Valid init ∧
  d.slot_mapping.Valid init ∧
  (∀ t, t < d.num_tokens → slotAt d init t < (numSlots d : Int)) ∧
  (∀ t t', t < d.num_tokens → t' < d.num_tokens → 0 ≤ slotAt d init t →
    slotAt d init t = slotAt d init t' → t = t') ∧
  (d.mode = .fp8 → d.k_scale.Valid init ∧ d.v_scale.Valid init)

/-! ## The specification -/

/-- The `float32` value of a one-element scale tensor in memory `m` (`+0` when absent). -/
def scaleAt (m : Mem) (b : Buf .f32) : F32 :=
  match m.read .f32 (b.ptr 0).addr with
  | some v => v
  | none => FloatV.ofPat 0

/-- **What a cache element holds** for an input element `x`, in configuration `m`, with scale
`s`: `x` itself, bit for bit, when the cache has the inputs' dtype (`Mode.bf16`, `Mode.fp8in`);
with `bfloat16` inputs and a `float8e4nv` cache (`Mode.fp8`), when `x` and `s` are finite, `s ≠ 0`
and `|x / s| ≤ 2^126`, the saturating rounding to nearest even (`FloatV.cvtSat .rn`) of a finite
`float32` within `2^-22·|x/s| + 2^-148` of `x / s` (the default division's bound, `DivFull`): one
of the at most two fp8 values next to `x / s` (`±448` beyond the range); otherwise nothing more
than that the element holds a `float8e4nv` value. -/
def Stored : (m : Mode) → F32 → m.kd.Val → m.cd.Val → Prop
  | .bf16, _, x, v => v = x
  | .fp8, s, x, v => (x : BF16).isFinite = true → s.isFinite = true → s.toRat ≠ 0 →
      |(x : BF16).toRat / s.toRat| ≤ 2 ^ 126 →
      ∃ r : F32, r.isFinite = true ∧
        |r.toRat - (x : BF16).toRat / s.toRat| ≤
          2 ^ (-22 : ℤ) * |(x : BF16).toRat / s.toRat| + 2 ^ (-148 : ℤ) ∧
        (v : F8E4) = FloatV.cvtSat .rn r
  | .fp8in, _, x, v => v = x

/-- What the launch leaves: for every token `t` with a slot `s ≥ 0`, every head `h` and component
`e`, the key cache's element `(s // block_size, s % block_size, h, e)` holds a value `Stored` says
of the initial `key[t, h, e]` (with the initial `k_scale`), and the value cache's element of the
initial `value[t, h, e]` (with `v_scale`); every byte outside the written elements of the two
caches is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t, t < d.num_tokens → 0 ≤ slotAt d init t → ∀ h e, h < d.num_heads → e < d.head_size →
      (∃ x v, init.read d.mode.kd (d.key.ptr (keyIdx d t (h * d.head_size + e) : Int)).addr = some x ∧
        fin.read d.mode.cd (d.key_cache.ptr (dstIdx d init t h e : Int)).addr = some v ∧
        Stored d.mode (scaleAt init d.k_scale) x v) ∧
      (∃ x v, init.read d.mode.kd (d.value.ptr (valueIdx d t (h * d.head_size + e) : Int)).addr = some x ∧
        fin.read d.mode.cd (d.value_cache.ptr (dstIdx d init t h e : Int)).addr = some v ∧
        Stored d.mode (scaleAt init d.v_scale) x v)) ∧
    ∀ z, z ∉ writeSet d init → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `reshape_and_cache_kernel_flash` ends normally with
`Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-! ### vLLM's two cache layouts

Both are views of one `bfloat16` allocation at `c0` (`make_caches`, source lines 301-320), with
`n = num_heads * head_size`:
* `flash`: `kv` is `[num_blocks, 2, block_size, num_heads, head_size]`, `key_cache = kv[:, 0]`
  and `value_cache = kv[:, 1]`: strides `(2 * block_size * n, n, head_size, 1)`, the value cache
  `block_size * n` elements after the key cache;
* `triton_attn`: `kv` is `[num_blocks, num_heads, block_size, 2 * head_size]`, transposed to
  `[num_blocks, block_size, num_heads, 2 * head_size]` and split along the last dimension: strides
  `(2 * block_size * n, 2 * head_size, 2 * block_size * head_size, 1)`, the value cache
  `head_size` elements after the key cache.

`key` and `value` are `qkv[:, 1]` and `qkv[:, 2]` of a `[num_tokens, 3, num_heads, head_size]`
tensor at `q0` (source lines 360-362), token stride `3 * n`. The buffers lie `2^40` bytes apart. -/

/-- Mixed-radix uniqueness: `a * m + x` determines `a` and `x` when `x < m`. -/
theorem mr {a a' x x' m : Nat} (hx : x < m) (hx' : x' < m) (h : a * m + x = a' * m + x') :
    a = a' ∧ x = x' := by
  have hm : 0 < m := by omega
  have e1 : (a * m + x) / m = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hm, Nat.div_eq_of_lt hx, Nat.zero_add]
  have e2 : (a' * m + x') / m = a' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hm, Nat.div_eq_of_lt hx', Nat.zero_add]
  have e3 : (a * m + x) % m = x := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hx]
  have e4 : (a' * m + x') % m = x' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hx']
  exact ⟨by rw [← e1, ← e2, h], by rw [← e3, ← e4, h]⟩

/-- Mixed-radix bound. -/
theorem mr_lt {a A x m : Nat} (ha : a < A) (hx : x < m) : a * m + x < A * m := by
  have : (a + 1) * m ≤ A * m := Nat.mul_le_mul_right _ ha
  nlinarith

/-- Two `bfloat16` elements at even addresses that differ share no byte. -/
theorem span_disj {a a' : Int} (h : a ≠ a') (ha : a % 2 = 0) (ha' : a' % 2 = 0) :
    Disjoint (Mem.span .bf16 a) (Mem.span .bf16 a') := by
  refine Set.disjoint_left.mpr fun x hx hx' => ?_
  simp only [Mem.span, DType.bytes, Set.mem_ofPred_eq] at hx hx'
  omega

/-- Two buffers whose byte ranges are ordered do not overlap. -/
theorem bytes_disjoint_of_le {d d' : DType} {b : Buf d} {c : Buf d'}
    (h : b.base + b.len * d.bytes ≤ c.base) : Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hcx => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hcx
  have hb := d.bytes_pos
  linarith

theorem bytes_disjoint_of_ge {d d' : DType} {b : Buf d} {c : Buf d'}
    (h : c.base + c.len * d'.bytes ≤ b.base) : Disjoint b.bytes c.bytes :=
  (bytes_disjoint_of_le h).symm

/-- The cache layout of an instance: `flash` or `triton_attn`. -/
inductive Layout
  | flash
  | triton_attn

/-- Two elements of size `dt.bytes` at different indices from one base share no byte. -/
theorem span_disj_idx {dt : DType} (c : Int) {i j : Nat} (h : i ≠ j) :
    Disjoint (Mem.span dt (c + (i : Int) * dt.bytes)) (Mem.span dt (c + (j : Int) * dt.bytes)) := by
  refine Set.disjoint_left.mpr fun x hx hx' => ?_
  simp only [Mem.span, Set.mem_ofPred_eq] at hx hx'
  have hb : (1 : Int) ≤ dt.bytes := by exact_mod_cast dt.bytes_pos
  rcases Nat.lt_or_gt_of_ne h with hij | hij
  · have : (i : Int) + 1 ≤ j := by exact_mod_cast hij
    nlinarith
  · have : (j : Int) + 1 ≤ i := by exact_mod_cast hij
    nlinarith

/-- The offset of the value cache's first element after the key cache's, in elements: `kv[:, 1]`
is `block_size * n` after `kv[:, 0]` (flash); the value half of a row is `head_size` after the
key half (triton_attn). -/
def vOff (L : Layout) (bs n hs : Nat) : Nat :=
  match L with | .flash => bs * n | .triton_attn => hs

/-- An instance with vLLM's buffers, in configuration `m`: `T` tokens, `nh` heads of `hs`, blocks
of `bs`, `nb` blocks, tile `TILE`, the caches in layout `L`. Offsets inside an allocation are in
elements times the element's bytes. -/
def mk (m : Mode) (L : Layout) (T nh hs bs nb TILE : Nat) : Inst :=
  let n := nh * hs
  let q0 : Int := 2 ^ 40
  let s0 : Int := 2 * 2 ^ 40
  let c0 : Int := 3 * 2 ^ 40
  let k0 : Int := 4 * 2 ^ 40
  { mode := m, num_tokens := T, num_heads := nh, head_size := hs, block_size := bs,
    num_blocks := nb, TILE_SIZE := TILE, key_stride := 3 * n, value_stride := 3 * n,
    block_stride := 2 * bs * n,
    page_stride := (match L with | .flash => n | .triton_attn => 2 * hs),
    head_stride := (match L with | .flash => hs | .triton_attn => 2 * bs * hs),
    key := ⟨q0 + (m.kd.bytes : Int) * n, T * (3 * n)⟩,
    value := ⟨q0 + (m.kd.bytes : Int) * (2 * n), T * (3 * n)⟩,
    key_cache := ⟨c0, nb * (2 * bs * n)⟩,
    value_cache := ⟨c0 + (m.cd.bytes : Int) * vOff L bs n hs, nb * (2 * bs * n) - vOff L bs n hs⟩,
    slot_mapping := ⟨s0, T⟩, k_scale := ⟨k0, 1⟩, v_scale := ⟨k0 + 4, 1⟩ }

/-- The element of the view as a mixed-radix number, in each layout. -/
theorem kvIdx_mk_flash (m : Mode) (T nh hs bs nb TILE b r h e : Nat) :
    kvIdx (mk m .flash T nh hs bs nb TILE) b r h e = ((b * (2 * bs) + r) * nh + h) * hs + e := by
  simp only [kvIdx, mk]; ring

theorem kvIdx_mk_triton (m : Mode) (T nh hs bs nb TILE b r h e : Nat) :
    kvIdx (mk m .triton_attn T nh hs bs nb TILE) b r h e =
      ((b * nh + h) * bs + r) * (2 * hs) + e := by
  simp only [kvIdx, mk]; ring

/-- Every instance `mk` builds is covered, given its sizes' bounds (`hq`, `hc`: the inputs and the
cache allocation each within its `2^40`-byte region; at most `2^22` tokens). -/
theorem mk_ok (m : Mode) (L : Layout) (T nh hs bs nb TILE : Nat) (ht : ∃ e, TILE = 2 ^ e)
    (htl : TILE ≤ 2 ^ 20)
    (h1 : T < 2 ^ 31) (h2 : (nh * hs + TILE - 1) / TILE < 2 ^ 16)
    (h3 : T * ((nh * hs + TILE - 1) / TILE) < 2 ^ 31) (h4 : nh * hs < 2 ^ 31) (h5 : bs < 2 ^ 31)
    (h6 : hs < 2 ^ 31) (hq : T * (3 * (nh * hs)) + 2 * (nh * hs) < 2 ^ 37)
    (hc : (nb + 1) * (2 * bs * (nh * hs)) + 2 * hs < 2 ^ 37) (hTc : T ≤ 2 ^ 22) :
    (mk m L T nh hs bs nb TILE).ok := by
  have hX : 2 * bs * (nh * hs) ≤ (nb + 1) * (2 * bs * (nh * hs)) :=
    Nat.le_mul_of_pos_left _ (Nat.succ_pos _)
  have hbh : 2 * bs * hs < 2 ^ 63 := by
    have : bs * hs < 2 ^ 31 * 2 ^ 31 := Nat.mul_lt_mul'' h5 h6
    have e : 2 * bs * hs = 2 * (bs * hs) := by ring
    omega
  have hvo : vOff L bs (nh * hs) hs ≤ bs * (nh * hs) + hs := by
    cases L <;> simp only [vOff] <;> omega
  refine ⟨ht, htl, h1, hTc, h2, h3, h4, h6, ?_, le_refl _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_⟩
  · -- block_size_lt
    show bs < 2 ^ 63
    omega
  · -- key_len
    intro t p ht hp
    show t * (3 * (nh * hs)) + p < T * (3 * (nh * hs))
    have hp' : p < nh * hs := hp
    have h' := Nat.mul_le_mul_right (3 * (nh * hs)) (show t + 1 ≤ T from ht)
    rw [Nat.succ_mul] at h'
    omega
  · -- value_len
    intro t p ht hp
    show t * (3 * (nh * hs)) + p < T * (3 * (nh * hs))
    have hp' : p < nh * hs := hp
    have h' := Nat.mul_le_mul_right (3 * (nh * hs)) (show t + 1 ≤ T from ht)
    rw [Nat.succ_mul] at h'
    omega
  · -- kv_len
    intro b r h e hb hr hh he
    have hb' : b < nb := hb
    have hr' : r < bs := hr
    have hh' : h < nh := hh
    have he' : e < hs := he
    cases L with
    | flash =>
      rw [kvIdx_mk_flash]
      have := mr_lt (mr_lt (mr_lt hb' (show r < 2 * bs by omega)) hh') he'
      have hv := mr_lt (mr_lt (mr_lt hb' (show bs + r < 2 * bs by omega)) hh') he'
      have e1 : nb * (2 * bs) * nh * hs = nb * (2 * bs * (nh * hs)) := by ring
      have e2 : ((b * (2 * bs) + (bs + r)) * nh + h) * hs + e =
          ((b * (2 * bs) + r) * nh + h) * hs + e + bs * (nh * hs) := by ring
      exact ⟨by show _ < nb * (2 * bs * (nh * hs)); omega,
        by show _ < nb * (2 * bs * (nh * hs)) - vOff .flash bs (nh * hs) hs; simp only [vOff]; omega⟩
    | triton_attn =>
      rw [kvIdx_mk_triton]
      have := mr_lt (mr_lt (mr_lt hb' hh') hr') (show e < 2 * hs by omega)
      have hv := mr_lt (mr_lt (mr_lt hb' hh') hr') (show hs + e < 2 * hs by omega)
      have e1 : nb * nh * bs * (2 * hs) = nb * (2 * bs * (nh * hs)) := by ring
      exact ⟨by show _ < nb * (2 * bs * (nh * hs)); omega,
        by show _ < nb * (2 * bs * (nh * hs)) - vOff .triton_attn bs (nh * hs) hs; simp only [vOff]
           omega⟩
  · -- kv_inj
    intro b r h e b' r' h' e' hb hr hh he hb' hr' hh' he' heq
    have hr1 : r < bs := hr
    have hr2 : r' < bs := hr'
    have hh1 : h < nh := hh
    have hh2 : h' < nh := hh'
    have he1 : e < hs := he
    have he2 : e' < hs := he'
    cases L with
    | flash =>
      rw [kvIdx_mk_flash, kvIdx_mk_flash] at heq
      obtain ⟨q1, rfl⟩ := mr he1 he2 heq
      obtain ⟨q2, rfl⟩ := mr hh1 hh2 q1
      obtain ⟨rfl, rfl⟩ := mr (show r < 2 * bs by omega) (show r' < 2 * bs by omega) q2
      exact ⟨rfl, rfl, rfl, rfl⟩
    | triton_attn =>
      rw [kvIdx_mk_triton, kvIdx_mk_triton] at heq
      obtain ⟨q1, rfl⟩ := mr (show e < 2 * hs by omega) (show e' < 2 * hs by omega) heq
      obtain ⟨q2, rfl⟩ := mr hr1 hr2 q1
      obtain ⟨rfl, rfl⟩ := mr hh1 hh2 q2
      exact ⟨rfl, rfl, rfl, rfl⟩
  · -- strides_lt
    cases L <;> exact ⟨show 3 * (nh * hs) < 2 ^ 63 by omega, show 3 * (nh * hs) < 2 ^ 63 by omega,
      show 2 * bs * (nh * hs) < 2 ^ 63 by omega, by simp only [mk]; omega,
      by simp only [mk]; omega⟩
  · -- lens_lt: inside the `2^37`-element regions
    have hX2 : nb * (2 * bs * (nh * hs)) ≤ (nb + 1) * (2 * bs * (nh * hs)) :=
      Nat.mul_le_mul_right _ (Nat.le_succ nb)
    simp only [mk]
    omega
  -- the caches lie after the inputs: ordered regions
  · refine bytes_disjoint_of_ge ?_; cases m <;> simp only [mk, DType.bytes] <;> omega
  · refine bytes_disjoint_of_ge ?_; cases m <;> simp only [mk, DType.bytes] <;> omega
  · refine bytes_disjoint_of_ge ?_; cases m <;> simp only [mk, DType.bytes] <;> omega
  · refine bytes_disjoint_of_ge ?_
    cases m <;> simp only [mk, DType.bytes] <;> omega
  · refine bytes_disjoint_of_ge ?_
    cases m <;> simp only [mk, DType.bytes] <;> omega
  · refine bytes_disjoint_of_ge ?_
    cases m <;> simp only [mk, DType.bytes] <;> omega
  · -- disj_caches: the key element's index differs from the value element's index plus the
    -- value cache's offset, in elements from the allocation's first element
    intro b r h e b' r' h' e' hb hr hh he hb' hr' hh' he'
    have hr1 : r < bs := hr
    have hr2 : r' < bs := hr'
    have hh1 : h < nh := hh
    have hh2 : h' < nh := hh'
    have he1 : e < hs := he
    have he2 : e' < hs := he'
    have hne : kvIdx (mk m L T nh hs bs nb TILE) b r h e ≠
        vOff L bs (nh * hs) hs + kvIdx (mk m L T nh hs bs nb TILE) b' r' h' e' := by
      intro heq
      cases L with
      | flash =>
        rw [kvIdx_mk_flash, kvIdx_mk_flash] at heq
        simp only [vOff] at heq
        have heq' : ((b * (2 * bs) + r) * nh + h) * hs + e =
            ((b' * (2 * bs) + (bs + r')) * nh + h') * hs + e' := by rw [heq]; ring
        obtain ⟨q1, -⟩ := mr he1 he2 heq'
        obtain ⟨q2, -⟩ := mr hh1 hh2 q1
        obtain ⟨-, q3⟩ := mr (show r < 2 * bs by omega) (show bs + r' < 2 * bs by omega) q2
        omega
      | triton_attn =>
        rw [kvIdx_mk_triton, kvIdx_mk_triton] at heq
        simp only [vOff] at heq
        have heq' : ((b * nh + h) * bs + r) * (2 * hs) + e =
            ((b' * nh + h') * bs + r') * (2 * hs) + (hs + e') := by rw [heq]; ring
        obtain ⟨-, q1⟩ := mr (show e < 2 * hs by omega) (show hs + e' < 2 * hs by omega) heq'
        omega
    have e1 : ((mk m L T nh hs bs nb TILE).key_cache.ptr
        (kvIdx (mk m L T nh hs bs nb TILE) b r h e : Int)).addr =
        3 * 2 ^ 40 + ((kvIdx (mk m L T nh hs bs nb TILE) b r h e : Nat) : Int) * m.cd.bytes := by
      simp only [Buf.ptr_addr, mk]
    have e2 : ((mk m L T nh hs bs nb TILE).value_cache.ptr
        (kvIdx (mk m L T nh hs bs nb TILE) b' r' h' e' : Int)).addr =
        3 * 2 ^ 40 + ((vOff L bs (nh * hs) hs + kvIdx (mk m L T nh hs bs nb TILE) b' r' h' e' : Nat) :
          Int) * m.cd.bytes := by
      simp only [Buf.ptr_addr, mk]; push_cast; ring
    show Disjoint (Mem.span m.cd _) (Mem.span m.cd _)
    rw [e1, e2]
    exact span_disj_idx _ hne
  · -- scales
    intro hm
    subst hm
    have hX2 : nb * (2 * bs * (nh * hs)) ≤ (nb + 1) * (2 * bs * (nh * hs)) :=
      Nat.mul_le_mul_right _ (Nat.le_succ nb)
    have hl : bs * (nh * hs) ≤ 2 * bs * (nh * hs) := by
      rw [show 2 * bs * (nh * hs) = bs * (nh * hs) + bs * (nh * hs) by ring]; omega
    refine ⟨le_refl _, le_refl _, bytes_disjoint_of_le ?_, bytes_disjoint_of_le ?_,
      bytes_disjoint_of_le ?_, bytes_disjoint_of_le ?_⟩ <;>
      simp only [mk, DType.bytes] <;> omega

/-- The memory every byte of which is present and `0`. -/
def zeroMem : Mem := fun _ => some 0

theorem readBytes_zeroMem : ∀ (n : Nat) (a : Int),
    zeroMem.readBytes a n = some (List.replicate n 0) := by
  intro n
  induction n with
  | zero => intro a; rfl
  | succ n ih =>
    intro a
    show (match zeroMem a, zeroMem.readBytes (a + 1) n with
      | some b, some bs => some (b :: bs)
      | _, _ => none) = _
    rw [ih (a + 1)]
    rfl

theorem zeroMem_read {d : DType} {a : Int} (h : Mem.Aligned d a) :
    zeroMem.read d a = some (DType.ofBits d 0) := by
  simp only [Mem.read, h, ↓reduceIte, readBytes_zeroMem, Option.map_some]
  have hlv : ∀ n : Nat, Bytes.leValue (List.replicate n 0) = 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih => simp only [List.replicate, Bytes.leValue, ih]; norm_num
  rw [hlv]

/-- Every slot of the all-zero memory is `0`. -/
theorem slotAt_zeroMem (d : Inst) (t : Nat) : slotAt d zeroMem t = 0 := by
  unfold slotAt
  by_cases h : Mem.Aligned .i64 (d.slot_mapping.ptr (t : Int)).addr
  · rw [zeroMem_read h]; decide
  · simp only [Mem.read, h, ↓reduceIte]

/-! ## The setup's launches

The host's test cases (source lines 344-353; `TILE_SIZE = min(2048, next_power_of_2(n))`), in
their layouts, and SETUP.md's serving shapes, with a `bfloat16` cache; vLLM's fp8 test cases
(`tests/kernels/attention/test_cache.py:181`, `kv_cache_dtype="fp8"`) and serving shapes with a
`float8e4nv` cache, from `bfloat16` inputs and from `float8e4nv` inputs. -/

/-- 21 tokens, 8 heads of 64, blocks of 16, 512 blocks, flash. -/
def setup_t21_h64_flash : Inst := mk .bf16 .flash 21 8 64 16 512 512
theorem setup_t21_h64_flash_ok : setup_t21_h64_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- 21 tokens, 8 heads of 80, blocks of 8, 512 blocks, flash. -/
def setup_t21_h80_flash : Inst := mk .bf16 .flash 21 8 80 8 512 1024
theorem setup_t21_h80_flash_ok : setup_t21_h80_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- 21 tokens, 8 heads of 256, blocks of 32, 5000 blocks, flash. -/
def setup_t21_h256_flash : Inst := mk .bf16 .flash 21 8 256 32 5000 2048
theorem setup_t21_h256_flash_ok : setup_t21_h256_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨11, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- 1 token, 8 heads of 128, blocks of 16, 64 blocks, flash. -/
def setup_t1_flash : Inst := mk .bf16 .flash 1 8 128 16 64 1024
theorem setup_t1_flash_ok : setup_t1_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- 1 token (padding), 8 heads of 128, blocks of 16, 64 blocks, triton_attn. -/
def setup_t1_triton : Inst := mk .bf16 .triton_attn 1 8 128 16 64 1024
theorem setup_t1_triton_ok : setup_t1_triton.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- 256 tokens (64 padding), 8 heads of 128, blocks of 16, 64 blocks, triton_attn. -/
def setup_t256_triton : Inst := mk .bf16 .triton_attn 256 8 128 16 64 1024
theorem setup_t256_triton_ok : setup_t256_triton.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- 300 tokens (17 padding), 4 heads of 96, blocks of 64, 20 blocks, triton_attn. -/
def setup_t300_h96_triton : Inst := mk .bf16 .triton_attn 300 4 96 64 20 512
theorem setup_t300_h96_triton_ok : setup_t300_h96_triton.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- 4096 tokens (100 padding), 32 heads of 128 (two tiles), blocks of 16, 300 blocks, flash. -/
def setup_t4096_h32_flash : Inst := mk .bf16 .flash 4096 32 128 16 300 2048
theorem setup_t4096_h32_flash_ok : setup_t4096_h32_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨11, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Serving (SETUP.md): Llama-3's 8 KV heads of 128, vLLM's default blocks of 16, 65536 blocks,
8192 batched tokens, in each layout. -/
def setup_serving_flash : Inst := mk .bf16 .flash 8192 8 128 16 65536 1024
theorem setup_serving_flash_ok : setup_serving_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_serving_triton : Inst := mk .bf16 .triton_attn 8192 8 128 16 65536 1024
theorem setup_serving_triton_ok : setup_serving_triton.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Serving: 32 KV heads of 256 (`n = 8192`, four tiles), blocks of 32. -/
def setup_serving_h32x256 : Inst := mk .bf16 .flash 4096 32 256 32 4096 2048
theorem setup_serving_h32x256_ok : setup_serving_h32x256.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨11, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The fp8 cache from `bfloat16` inputs (`Mode.fp8`): vLLM's test shapes (21 tokens, 8 heads of
64, 80 and 256, blocks of 16, 8 and 32, 512 and 5000 blocks, the flash layout) and serving
(8 KV heads of 128, blocks of 16, 65536 blocks, 8192 tokens, in each layout). -/
def setup_fp8_t21_h64_flash : Inst := mk .fp8 .flash 21 8 64 16 512 512
theorem setup_fp8_t21_h64_flash_ok : setup_fp8_t21_h64_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨9, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_fp8_t21_h80_flash : Inst := mk .fp8 .flash 21 8 80 8 512 1024
theorem setup_fp8_t21_h80_flash_ok : setup_fp8_t21_h80_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_fp8_t21_h256_flash : Inst := mk .fp8 .flash 21 8 256 32 5000 2048
theorem setup_fp8_t21_h256_flash_ok : setup_fp8_t21_h256_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨11, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_fp8_serving_flash : Inst := mk .fp8 .flash 8192 8 128 16 65536 1024
theorem setup_fp8_serving_flash_ok : setup_fp8_serving_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_fp8_serving_triton : Inst := mk .fp8 .triton_attn 8192 8 128 16 65536 1024
theorem setup_fp8_serving_triton_ok : setup_fp8_serving_triton.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The fp8 cache from `float8e4nv` inputs (`Mode.fp8in`, the `is_fp8()` side): 8 KV heads of
128, blocks of 16, in each layout. -/
def setup_fp8in_flash : Inst := mk .fp8in .flash 8192 8 128 16 65536 1024
theorem setup_fp8in_flash_ok : setup_fp8in_flash.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_fp8in_triton : Inst := mk .fp8in .triton_attn 256 8 128 16 64 1024
theorem setup_fp8in_triton_ok : setup_fp8in_triton.ok :=
  mk_ok _ _ _ _ _ _ _ _ ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- A covered instance and memory: the test's one-token flash case on the all-zero memory (the
token's slot `0`). -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨setup_t1_flash, zeroMem, setup_t1_flash_ok, ?_⟩
  refine ⟨⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ?_, ?_, ?_⟩
  rotate_left 15
  · intro t _
    rw [slotAt_zeroMem]
    simp [numSlots, setup_t1_flash, mk]
  · intro t t' ht ht' _ _
    simp only [setup_t1_flash, mk] at ht ht'
    omega
  · intro h
    exact absurd h (by decide)
  all_goals (simp only [setup_t1_flash, mk, vOff, DType.bytes]; norm_num)

end Claims.KvCacheWrite
