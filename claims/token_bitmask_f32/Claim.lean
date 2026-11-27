import TVA

/-!
# `token_bitmask_f32`: grammar bitmask applied to `float32` logits in place, SGLang

The kernel is `apply_token_bitmask_inplace_kernel` of `source/bitmask_ops.py` (lines 56-124; it
reads the module-level names `triton` and `tl`, lines 45-46). The host
`apply_token_bitmask_inplace_triton` (lines 127-184, with `get_device_core_count`, lines 49-53)
launches it with `NUM_SMS` programs (the device's multiprocessor count) and `BLOCK_SIZE = 4096`,
on the `float32` logits `[batch, width]` (contiguous), the `int32` bitmask
`[batch', cdiv(width, 32)]` (at most that wide, contiguous) and `indices` either `None` or an
`int32` tensor of rows. The programs walk the `num_rows * cdiv(vocab_size, 4096)` work items (a row,
a block of `4096` tokens) with stride `NUM_SMS`. `SETUP.md` lists the launches.

What is claimed: every covered launch ends normally; for each processed row `r` (the `k`-th index
with `indices`, row `k` without) and each token `c < vocab_size` whose bit, bit `c % 32` of the
bitmask's element `r * bitmask_strides + c / 32`, is `0`, the logit
`r * logits_strides + c` holds the `float32` `-∞` (`0xff800000`); every other byte, every other
logit included, is unchanged.

The conditions: a grid of `NUM_SMS` programs or fewer, enough to reach every work item, and
`NUM_SMS` positive unless there is no item; the
`int32` arithmetic does not wrap (`vocab_size + 4095 < 2^31`, `num_rows * cdiv(vocab_size, 4096) +
NUM_SMS ≤ 2^31`, each buffer's rows within `2^31` elements); `vocab_size` within the logits' row
stride and the bitmask's `32 * bitmask_strides` bits; the buffers hold their rows, are valid, and
the logits share no byte with the bitmask or the indices; with `indices`, every index is a row of
both tensors and the indices are distinct. Any logits and bitmask values are covered.

The specification is exact: the only float value is the stored constant, no device fact.
-/

namespace Claims.TokenBitmask

open TVA

/-! ## The model -/

/-- `-float("inf")` of line 122: a Python `float` stored through a `float32` pointer, so a tensor
of its own dtype, `float32` (an infinity), already the pointer's dtype: the constant Triton builds
for the double `-∞` (`litF32` of its pattern; `TVA.Triton.Promote`). -/
def negInfC : F32 := FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)

/-- `apply_token_bitmask_inplace_kernel` (source lines 56-124) with the host's
`BLOCK_SIZE = 4096`; `NUM_SMS` is a Lean parameter. `logits_ptr` is `*fp32`, `bitmask_ptr`
`*int32`; `indices_ptr` is `none` when the host passes `None` (line 108's `is None` is then a
compile-time `True`) and `some p` for the `int32` tensor. `num_rows`, `vocab_size`,
`logits_strides` and `bitmask_strides` are runtime `int32` arguments (values in `[0, 2^31)`). An
argument equal to `1` is the compile-time constant `1`: every use of `num_rows`,
`logits_strides` and `bitmask_strides` is an operand of `*` or `<` with an `int32` tensor, which
builds `1` in `int32`, and `vocab_size = 1` makes line 104 Python's `cdiv(1, 4096) = 1` and line
111 a comparison with the `int32` `1`: the same values as the model's `int32` arguments. -/
def apply_token_bitmask_inplace_kernel (logits_ptr : Ptr .f32) (bitmask_ptr : Ptr .i32)
    (indices_ptr : Option (Ptr .i32)) (num_rows vocab_size logits_strides bitmask_strides : I32)
    (NUM_SMS : Nat) : Kernel where
  body ctx := do
    -- line 103: axis 0 of the 1-D grid, an `int32`
    let pid := tl.program_id ctx 0
    -- line 104: `tl.cdiv(vocab_size, BLOCK_SIZE)` with a tensor operand: Triton's body
    -- `(x + (div - 1)) // div`, the constexpr `4096` built in `int32`
    let num_blocks := (← tl.cdiv (Tensor.scalar vocab_size)
      (Tensor.scalar (IntV.ofNat 4096 : I32))).item
    -- line 105: `tl.range(pid, num_rows * num_blocks, NUM_SMS)`: `int32` bounds (the wrapping
    -- product), the constexpr step built in `int32`; the loop carries nothing
    tl.forRange pid (num_rows * num_blocks) (IntV.ofNat NUM_SMS : I32) () fun work_id _ => do
      -- line 106: truncating `int32` `//`
      let row_id := (← tl.idiv (Tensor.scalar work_id) (Tensor.scalar num_blocks)).item
      -- line 107: truncating `%`, then the product with the constexpr built in `int32`
      let block_offset :=
        (← tl.irem (Tensor.scalar work_id) (Tensor.scalar num_blocks)).item * (IntV.ofNat 4096 : I32)
      -- line 108: `row_id if indices_ptr is None else tl.load(indices_ptr + row_id)`: `is` gives
      -- a compile-time value, so one side is evaluated; the load is one `int32` element
      let batch_id : I32 ←
        match indices_ptr with
        | none => pure row_id
        | some indices_ptr => tl.load1 (indices_ptr + row_id)
      -- line 109
      let offsets := block_offset + tl.arange 0 4096
      -- line 110: `block_offset // 32`, the Python `32` built in `int32`; `BLOCK_SIZE // 32` is
      -- the compile-time `128`
      let bitmask_offsets :=
        (← tl.idiv (Tensor.scalar block_offset) (Tensor.scalar (IntV.ofNat 32 : I32))).item +
          tl.arange 0 128
      -- lines 111-112: signed `int32` comparisons with the scalars
      let vocab_mask := offsets.ltS vocab_size
      let packed_bitmask_mask := bitmask_offsets.ltS bitmask_strides
      -- lines 113-116: `(bitmask_ptr + batch_id * bitmask_strides) + bitmask_offsets`, left to
      -- right: the `int32` product, then the `int32` lanes; a masked load without `other`
      let packed_bitmask : Tensor [128] I32 ←
        tl.load (bitmask_ptr + batch_id * bitmask_strides + bitmask_offsets)
          (some packed_bitmask_mask)
      -- line 117: `packed_bitmask[:, None] >> tl.arange(0, 32)[None, :]`, both broadcast to
      -- `[128, 32]`: an arithmetic shift (the left operand, the dispatching one, is `int32`), by
      -- amounts `0..31`; `& 1` with the Python `1` built in `int32`; `== 0` compares with the
      -- `int32` `0` (the Python number a tensor of its own dtype)
      let shifted : Tensor [128, 32] I32 ←
        tl.ashr (packed_bitmask.col 32) ((tl.arange 0 32 : Tensor [32] I32).row 128)
      let bitmask : Tensor [128, 32] Bool :=
        (Tensor.map2 IntV.and shifted (bc (IntV.ofInt 1 : I32))).eq (bc (IntV.ofInt 0 : I32))
      -- line 118: `reshape` (row-major, `can_reorder=False`): element `(i, j)` is lane `32 i + j`
      let bitmask : Tensor [4096] Bool := bitmask.reshape [4096] (by decide)
      -- lines 120-124: `(logits_ptr + batch_id * logits_strides) + offsets`: the `int32` product,
      -- then the `int32` lanes; the stored value `negInfC` broadcast to the block (already the
      -- pointer's dtype); the mask `vocab_mask & bitmask`
      tl.store (logits_ptr + batch_id * logits_strides + offsets) (tl.full [4096] negInfC)
        (some (vocab_mask.and bitmask))

/-! ## The conditions -/

/-- What the host chooses: one launch of the kernel through `apply_token_bitmask_inplace_triton`
or directly with the host's launch line. The tensors are given by their storage (`Buf`). -/
structure Inst where
  /-- `num_rows`: the rows processed (`len(indices)` with indices, else the logits' rows). -/
  num_rows : Nat
  /-- `vocab_size`: the tokens processed in each row (`min(width, 32 * bitmask width)`). -/
  vocab_size : Nat
  /-- `logits_strides`: the logits' row stride in elements (the host's `logits_shape[1]`). -/
  logits_strides : Nat
  /-- `bitmask_strides`: the bitmask's row stride in elements (the host's `bitmask_shape[1]`). -/
  bitmask_strides : Nat
  /-- The logits' rows (`logits_shape[0]`). -/
  logits_rows : Nat
  /-- The bitmask's rows (`bitmask_shape[0]`). -/
  bitmask_rows : Nat
  /-- `indices` is a tensor (`true`) or `None` (`false`). -/
  hasIdx : Bool
  /-- `NUM_SMS`, the constexpr: the device's multiprocessor count (`132` on the H200). -/
  NUM_SMS : Nat
  /-- The grid's `x` (`NUM_SMS` on a CUDA device). -/
  grid : Nat
  /-- The storage of `logits` (`float32`), updated in place. -/
  logits : Buf .f32
  /-- The storage of `bitmask` (`int32`). -/
  bitmask : Buf .i32
  /-- The storage of `indices` (`int32`; unused without it). -/
  indices : Buf .i32

/-- The work items: `num_rows * cdiv(vocab_size, 4096)`. -/
def items (d : Inst) : Nat := d.num_rows * ((d.vocab_size + 4095) / 4096)

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `NUM_SMS` is positive (the loop's step), or there is no work item (the host's branch for a
  device that reports no multiprocessors launches an empty grid with `NUM_SMS = 0` then); and it
  is an `int32` constant. -/
  sms_pos : 0 < d.NUM_SMS ∨ items d = 0
  sms_lt : d.NUM_SMS < 2 ^ 31
  /-- At most `NUM_SMS` programs (program `p` takes the items `p, p + NUM_SMS, …`: a program past
  `NUM_SMS` would repeat another's items), and enough of them to reach every item: the host's
  `NUM_SMS` (a CUDA device), or at least one per item. -/
  grid_le : d.grid ≤ d.NUM_SMS
  grid_cover : min d.NUM_SMS (items d) ≤ d.grid
  /-- `vocab_size` is an `int32` whose `cdiv` addition `vocab_size + 4095` does not wrap (a limit
  of the kernel, README, Exclusions). -/
  vocab_lt : d.vocab_size + 4095 < 2 ^ 31
  /-- `num_rows` is an `int32`, and the `int32` product `num_rows * num_blocks` and the loop's last
  increment do not wrap (`tl.RangeOk`; a limit of the kernel). -/
  rows_lt : d.num_rows < 2 ^ 31
  range_le : items d + d.NUM_SMS ≤ 2 ^ 31
  /-- The strides are `int32` arguments. -/
  ls_lt : d.logits_strides < 2 ^ 31
  bs_lt : d.bitmask_strides < 2 ^ 31
  /-- A row's tokens lie in its logits row and its bitmask row (the host's `min`). -/
  vocab_ls : d.vocab_size ≤ d.logits_strides
  vocab_bs : d.vocab_size ≤ 32 * d.bitmask_strides
  /-- The buffers hold their rows. -/
  logits_len : d.logits_rows * d.logits_strides ≤ d.logits.len
  bitmask_len : d.bitmask_rows * d.bitmask_strides ≤ d.bitmask.len
  /-- The `int32` row offsets `batch_id * logits_strides` and `batch_id * bitmask_strides` do not
  wrap: each tensor has at most `2^31` elements (a limit of the kernel). -/
  logits_fit : d.logits_rows * d.logits_strides ≤ 2 ^ 31
  bitmask_fit : d.bitmask_rows * d.bitmask_strides ≤ 2 ^ 31
  /-- Without `indices`, the rows processed are rows of both tensors (the host asserts equal row
  counts). -/
  rows_le : d.hasIdx = false → d.num_rows ≤ d.logits_rows ∧ d.num_rows ≤ d.bitmask_rows
  /-- With `indices`, it holds the `num_rows` indices. -/
  idx_len : d.hasIdx = true → d.num_rows ≤ d.indices.len
  /-- The logits share no byte with the bitmask or the indices (the host makes `indices` a fresh
  tensor). -/
  disj_bitmask : Disjoint d.logits.bytes d.bitmask.bytes
  disj_idx : d.hasIdx = true → Disjoint d.logits.bytes d.indices.bytes

/-- The value of index `k` in `m` (`0` when absent). -/
def idxVal (d : Inst) (m : Mem) (k : Nat) : Int :=
  match m.read .i32 (d.indices.ptr (k : Int)).addr with
  | some v => v.toInt
  | none => 0

/-- The row the `k`-th processed row is: `indices[k]` with indices, `k` without. -/
def rowOf (d : Inst) (m : Mem) (k : Nat) : Int :=
  if d.hasIdx then idxVal d m k else (k : Int)

/-- Token `c`'s bit in row `r` of the bitmask in `m`: bit `c % 32` (the least significant bit
is bit `0`) of the `int32` element `r * bitmask_strides + c / 32` (`true` when absent). -/
def bitOf (d : Inst) (m : Mem) (r : Int) (c : Nat) : Bool :=
  match m.read .i32 (d.bitmask.ptr (r * (d.bitmask_strides : Int) + ((c / 32 : Nat) : Int))).addr with
  | some v => v.bits.getLsbD (c % 32)
  | none => true

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte present;
`indices` when given); with `indices`, every index is a row of both tensors and the indices are
distinct (the kernel's "unique indices"). Any logits and bitmask values. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.logits.Valid init ∧ d.bitmask.Valid init ∧
    (d.hasIdx = true → d.indices.Valid init) ∧
    (d.hasIdx = true → ∀ k, k < d.num_rows →
      0 ≤ idxVal d init k ∧ idxVal d init k < d.logits_rows ∧ idxVal d init k < d.bitmask_rows) ∧
    (d.hasIdx = true → ∀ k k', k < d.num_rows → k' < d.num_rows →
      idxVal d init k = idxVal d init k' → k = k')

/-! ## The specification -/

/-- The logits the launch sets to `-∞`: token `c < vocab_size` of a processed row whose bit is
`0`, at element `row * logits_strides + c`. -/
def maskedElems (d : Inst) (init : Mem) : Set Nat :=
  {e | ∃ k, k < d.num_rows ∧ ∃ c, c < d.vocab_size ∧ bitOf d init (rowOf d init k) c = false ∧
    (e : Int) = rowOf d init k * (d.logits_strides : Int) + (c : Int)}

/-- What the launch leaves: for every processed row and token `c < vocab_size` whose bit is `0`
in the initial bitmask, the logit holds the `float32` `-∞`; every byte outside those logits is
unchanged (the other logits, the bitmask, the indices). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ k, k < d.num_rows → ∀ c, c < d.vocab_size → bitOf d init (rowOf d init k) c = false →
      fin.read .f32 (d.logits.ptr (rowOf d init k * (d.logits_strides : Int) + (c : Int))).addr =
        some (FloatV.negInf : F32)) ∧
    ∀ z, z ∉ d.logits.elems (maskedElems d init) → fin z = init z

/-- The model on an instance: the kernel's arguments are the buffers' first elements (and
`indices_ptr` `none` without indices) and the four integers. -/
def model (d : Inst) : Kernel :=
  apply_token_bitmask_inplace_kernel (d.logits.ptr 0) (d.bitmask.ptr 0)
    (if d.hasIdx then some (d.indices.ptr 0) else none) (IntV.ofNat d.num_rows)
    (IntV.ofNat d.vocab_size) (IntV.ofNat d.logits_strides) (IntV.ofNat d.bitmask_strides)
    d.NUM_SMS

/-- The host's launch (lines 172-184): a 1-D grid of `grid` programs; the source's block shapes;
`enable_fp_fusion` is Triton's default (`true`); `num_warps = 32` (`float32`) and
`num_stages = 3` change no result; `num_ctas = 1`, no `launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 d.grid [[4096], [128], [32], [128, 32]] (fusion := true)

/-! ## The claim -/

/-- **The claim**: every covered launch of `apply_token_bitmask_inplace_kernel` ends normally with
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

/-- An instance with the host's layout and no indices: `bs` rows of `float32` logits of width
`width` and of an `int32` bitmask of width `bw`, `vocab = min(width, 32 * bw)`, on a device of `S`
multiprocessors (`grid = NUM_SMS = S`); the logits from `2^30`, the bitmask and the (unused)
indices after it with `2^20`-byte gaps. -/
def mk (bs width bw S : Nat) : Inst :=
  let l0 : Int := 2 ^ 30
  let b0 : Int := l0 + bs * width * 4 + 2 ^ 20
  let i0 : Int := b0 + bs * bw * 4 + 2 ^ 20
  { num_rows := bs, vocab_size := min width (32 * bw), logits_strides := width,
    bitmask_strides := bw, logits_rows := bs, bitmask_rows := bs, hasIdx := false,
    NUM_SMS := S, grid := S,
    logits := ⟨l0, bs * width⟩, bitmask := ⟨b0, bs * bw⟩, indices := ⟨i0, 0⟩ }

theorem mk_ok (bs width bw S : Nat) (h0 : 0 < S) (h1 : min width (32 * bw) + 4095 < 2 ^ 31)
    (h2 : bs < 2 ^ 31) (h3 : bs * ((min width (32 * bw) + 4095) / 4096) + S ≤ 2 ^ 31)
    (h4 : width < 2 ^ 31) (h5 : bw < 2 ^ 31) (h6 : bs * width ≤ 2 ^ 31) (h7 : bs * bw ≤ 2 ^ 31)
    (h8 : S < 2 ^ 31 := by norm_num) :
    (mk bs width bw S).ok := by
  refine ⟨Or.inl h0, h8, le_refl _, min_le_left _ _, h1, h2, h3, h4, h5,
    min_le_left _ _, min_le_right _ _, le_refl _, le_refl _, h6, h7,
    fun _ => ⟨le_refl _, le_refl _⟩, fun h => by simp [mk] at h, ?_, fun h => by simp [mk] at h⟩
  refine bytes_disjoint_of_le ?_
  simp only [mk, DType.bytes]; push_cast; nlinarith

/-- The memory every byte of which is present and `0`. -/
def zeroMem : Mem := fun _ => some 0

/-- A covered instance and memory: SGLang's call at batch 1 and vocabulary 32000, 132
multiprocessors, on the all-zero memory (every token masked). -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 1 32000 1000 132, zeroMem,
    mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num), ?_⟩
  refine ⟨⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    fun h => by simp [mk] at h, fun h => by simp [mk] at h, fun h => by simp [mk] at h⟩
  all_goals (simp only [mk, DType.bytes]; norm_num)

/-! ## The setup's launches

SGLang's calls (`SETUP.md`): the sampler's `float32` logits `[bs, vocab]` and the grammar's bitmask
`[bs, cdiv(vocab, 32)]`, no indices, on the H200 (`NUM_SMS = 132`); then the host's other
launches: a padded row, the branch for a device without multiprocessors, and indices. -/

/-- Batch 1, vocabulary 32000 (Llama 2, Mistral). -/
def setup_bs1_v32000 : Inst := mk 1 32000 1000 132
theorem setup_bs1_v32000_ok : setup_bs1_v32000.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- Batch 64, vocabulary 128256 (Llama 3). -/
def setup_bs64_v128256 : Inst := mk 64 128256 4008 132
theorem setup_bs64_v128256_ok : setup_bs64_v128256.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- Batch 256, vocabulary 129280 (DeepSeek-V3). -/
def setup_bs256_v129280 : Inst := mk 256 129280 4040 132
theorem setup_bs256_v129280_ok : setup_bs256_v129280.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- Batch 1024, vocabulary 151936 (Qwen 2, Qwen 3). -/
def setup_bs1024_v151936 : Inst := mk 1024 151936 4748 132
theorem setup_bs1024_v151936_ok : setup_bs1024_v151936.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- Batch 37, vocabulary 152064 (Qwen 2.5). -/
def setup_bs37_v152064 : Inst := mk 37 152064 4752 132
theorem setup_bs37_v152064_ok : setup_bs37_v152064.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- Batch 16, vocabulary 202048 (Llama 4). -/
def setup_bs16_v202048 : Inst := mk 16 202048 6314 132
theorem setup_bs16_v202048_ok : setup_bs16_v202048.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- Batch 8, vocabulary 262144 (Gemma 3). -/
def setup_bs8_v262144 : Inst := mk 8 262144 8192 132
theorem setup_bs8_v262144_ok : setup_bs8_v262144.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- Speculative decoding: 1024 requests with 7 draft tokens each, one row per candidate token
(8192 rows), vocabulary 129280. -/
def setup_spec_bs8192_v129280 : Inst := mk 8192 129280 4040 132
theorem setup_spec_bs8192_v129280_ok : setup_spec_bs8192_v129280.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- The largest batch at the largest vocabulary: 8192 rows of 262144 (`2^31` logits). -/
def setup_bs8192_v262144 : Inst := mk 8192 262144 8192 132
theorem setup_bs8192_v262144_ok : setup_bs8192_v262144.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
/-- A padded logits row: width 32064 (an lm head padded to a multiple of 64) with a bitmask of the
vocabulary's 1000 words, so `vocab_size = 32000` and the 64 padding logits are not processed. -/
def setup_padded_v32000 : Inst := mk 4 32064 1000 132
theorem setup_padded_v32000_ok : setup_padded_v32000.ok :=
  mk_ok _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)

/-- The host's branch for a device that reports no multiprocessors (`get_device_core_count()`
is `0`): one program per item, `grid = items` and `NUM_SMS = next_power_of_2(items)`. -/
def mkG (bs width bw S G : Nat) : Inst := { mk bs width bw S with grid := G }

theorem mkG_ok (bs width bw S G : Nat) (h0 : 0 < S ∨ bs * ((min width (32 * bw) + 4095) / 4096) = 0)
    (h1 : min width (32 * bw) + 4095 < 2 ^ 31)
    (h2 : bs < 2 ^ 31) (h3 : bs * ((min width (32 * bw) + 4095) / 4096) + S ≤ 2 ^ 31)
    (h4 : width < 2 ^ 31) (h5 : bw < 2 ^ 31) (h6 : bs * width ≤ 2 ^ 31) (h7 : bs * bw ≤ 2 ^ 31)
    (h8 : S < 2 ^ 31) (h9 : G ≤ S) (h10 : min S (bs * ((min width (32 * bw) + 4095) / 4096)) ≤ G) :
    (mkG bs width bw S G).ok := by
  refine ⟨h0, h8, h9, h10, h1, h2, h3, h4, h5,
    min_le_left _ _, min_le_right _ _, le_refl _, le_refl _, h6, h7,
    fun _ => ⟨le_refl _, le_refl _⟩, fun h => by simp [mkG, mk] at h, ?_,
    fun h => by simp [mkG, mk] at h⟩
  refine bytes_disjoint_of_le ?_
  simp only [mkG, mk, DType.bytes]; push_cast; nlinarith

/-- That branch at batch 5, vocabulary 16384: 20 items, `NUM_SMS = 32`. -/
def setup_nosm_bs5_v16384 : Inst := mkG 5 16384 512 32 20
theorem setup_nosm_bs5_v16384_ok : setup_nosm_bs5_v16384.ok :=
  mkG_ok _ _ _ _ _ (Or.inl (by norm_num)) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- That branch with no work (batch 0): an empty grid and `NUM_SMS = next_power_of_2(0) = 0`. -/
def setup_nosm_empty : Inst := mkG 0 32000 1000 0 0
theorem setup_nosm_empty_ok : setup_nosm_empty.ok :=
  mkG_ok _ _ _ _ _ (Or.inr (by norm_num)) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The host with `indices`: `n` rows given by a list of distinct indices of a `[bs, width]` logits
tensor and its bitmask, on a device of `S` multiprocessors; the indices' storage after the
bitmask. -/
def mkIdx (bs width bw n S : Nat) : Inst := { mk bs width bw S with
  num_rows := n, hasIdx := true, indices := ⟨(mk bs width bw S).indices.base, n⟩ }

theorem mkIdx_ok (bs width bw n S : Nat) (h0 : 0 < S) (h1 : min width (32 * bw) + 4095 < 2 ^ 31)
    (h2 : n < 2 ^ 31) (h3 : n * ((min width (32 * bw) + 4095) / 4096) + S ≤ 2 ^ 31)
    (h4 : width < 2 ^ 31) (h5 : bw < 2 ^ 31) (h6 : bs * width ≤ 2 ^ 31) (h7 : bs * bw ≤ 2 ^ 31)
    (h8 : S < 2 ^ 31 := by norm_num) :
    (mkIdx bs width bw n S).ok := by
  refine ⟨Or.inl h0, h8, le_refl _, min_le_left _ _, h1, h2, h3, h4, h5,
    min_le_left _ _, min_le_right _ _, le_refl _, le_refl _, h6, h7,
    fun h => by simp [mkIdx] at h, fun _ => le_refl _, ?_, fun _ => ?_⟩
  · refine bytes_disjoint_of_le ?_
    simp only [mkIdx, mk, DType.bytes]; push_cast; nlinarith
  · refine bytes_disjoint_of_le ?_
    simp only [mkIdx, mk, DType.bytes]; push_cast; nlinarith

/-- The host with a list of 3 of 8 rows, vocabulary 32000 (the xgrammar interface's `indices`). -/
def setup_indices_v32000 : Inst := mkIdx 8 32000 1000 3 132
theorem setup_indices_v32000_ok : setup_indices_v32000.ok :=
  mkIdx_ok _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

end Claims.TokenBitmask
