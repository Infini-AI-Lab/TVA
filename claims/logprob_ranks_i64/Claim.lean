import TVA

/-!
# `logprob_ranks_i64`: ranks of the sampled tokens, vLLM

The kernel is `_ranks_kernel` of `source/logprob_ranks.py` (lines 47–67). Its host
`compute_token_ranks` (lines 70–85) allocates the `int64` output `token_ranks` and launches the
kernel on a 1-D grid of `batch_size` programs with `BLOCK_SIZE = 8192`; `SETUP.md` lists the
launches.

Program `r` loads the row's sampled token id `t_r` and its logit `x = logits[r, t_r]`, then walks
the row in blocks of `BLOCK_SIZE` lanes, the lanes past `vocab_size` holding `-inf`, and counts in
an `int32` the lanes whose logit is at least `x` (`logits >= x`, ordered: a NaN on either side
does not count); it stores the count, widened to `int64`, at `token_ranks[r]`.

What is claimed: every covered launch ends normally; element `r` of `token_ranks` is, for every
row `r < batch_size`, the number of entries `j < vocab_size` with `x ≤ logits[r, j]`, plus, when
`x` is `-inf`, the number of padding lanes of the last block (`padLanes`: `-inf ≤ -inf` holds);
every other byte is unchanged. It is exact: comparisons and integer counts only. A NaN `x` gives
`0`; a NaN entry is never counted.

The conditions: the logits are `float32`, `bfloat16` or `float16`; `BLOCK_SIZE` a power of two at
most `2^20`; `vocab_size + BLOCK_SIZE ≤ 2^31`; the row stride below `2^31`; fewer than `2^31` rows;
the buffers hold what the launch touches; the output shares no byte with an input; every token id
is in `[0, vocab_size)`. Every logit value is covered. The kernel has no approximate operation, so
the specification uses no device fact.
-/

namespace Claims.LogprobRanks

open TVA

/-! ## Helpers of the transcription

The logits may be `float32`, `bfloat16` or `float16` (a separate compilation per dtype). The model
is one function of the dtype, with the dtype-dependent comparison written explicitly; `Inst.ok`
restricts the dtype to the three covered ones. -/

/-- `logits >= x` (source line 66) on the loaded block and the token's logit, both of the logits'
dtype: a float compared with its own dtype is not promoted (`BinOp.typing?`), and `x` is broadcast
to the block. Triton's `>=` on floats is the ordered `fcmp oge`, written here as the ordered
`x <= logits` (`Tensor.fle`), which is the same comparison: false when either side is NaN. -/
def geX {n : Nat} : (dt : DType) → Tensor [n] dt.Val → dt.Val → Tensor [n] Bool
  | .f32, v, x => Tensor.fle (bc x) v
  | .bf16, v, x => Tensor.fle (bc x) v
  | .f16, v, x => Tensor.fle (bc x) v
  | _, _, _ => Tensor.full false

/-- The Python float `-inf` of `other=float("-inf")` (source line 65): a Python `float` beside a
`tl.load` first becomes a `float32` tensor of its own type (`Operand.ownType?`; an infinite float
is `float32`), the `float32` constant of the double `0xFFF0000000000000` (`litF32`, `0xFF800000`),
broadcast to the block. -/
def negInfLit {n : Nat} : Tensor [n] F32 := bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000))

/-! ## The model -/

/-- `_ranks_kernel` (source lines 47–67). `output_ptr` points at the `int64` output,
`logits_ptr` at the logits (dtype `dt`), `token_ids_ptr` at the `int64` token ids;
`logits_stride` and `vocab_size` are the `int32` runtime arguments (at the value `1` they are
compile-time constants, which give the same values: README, "Modelling decisions");
`BLOCK_SIZE` is a `tl.constexpr`, a Lean parameter. -/
def _ranks_kernel (dt : DType) (output_ptr : Ptr .i64) (logits_ptr : Ptr dt)
    (logits_stride : I32) (token_ids_ptr : Ptr .i64) (vocab_size : I32) (BLOCK_SIZE : Nat) :
    Kernel where
  body ctx := do
    -- line 56: `tl.program_id(0).to(tl.int64)`: the `int32` program id, sign-extended
    let req_idx : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 57: `req_idx * logits_stride`: an `int64` times an `int32`, the `int32` stride
    -- sign-extended first (`BinOp.typing? .mul`), then the wrapping `int64` product; the pointer
    -- advanced by it
    let stride64 : I64 := (← tl.convert .i32 .i64 (Tensor.scalar logits_stride)).item
    let row_ptr := logits_ptr + req_idx * stride64
    -- line 59: the scalar load of the row's token id (`*i64`)
    let token_id ← tl.load1 (token_ids_ptr + req_idx)
    -- line 60: the token's logit, at the row pointer advanced by the `int64` id
    let x ← tl.load1 (row_ptr + token_id)
    -- line 62: `n = 0`: the Python `0` assigned to a name becomes the `int32` scalar `0`
    let n : I32 := 0
    -- lines 63–66: `for i in range(0, vocab_size, BLOCK_SIZE)`: the bounds `0` and `BLOCK_SIZE`
    -- (Python numbers) become `int32`, so the loop is the `int32` `tl.forRange` with step
    -- `BLOCK_SIZE`; it carries `n`
    let n ← tl.forRange (0 : I32) vocab_size (IntV.ofNat BLOCK_SIZE : I32) n fun i n => do
      -- line 64: the `int32` scalar plus the `int32` `tl.arange` (wrapping addition)
      let block := i + tl.arange 0 BLOCK_SIZE
      -- line 65: the masked load (`block < vocab_size`, signed `int32`); `other` is converted to
      -- the pointer's dtype (`tl.convert .f32 dt`, the identity for `float32`)
      let other ← tl.convert .f32 dt negInfLit
      let logits ← tl.load (row_ptr + block) (some (block.ltS vocab_size)) (some other)
      -- line 66: `(logits >= x).to(tl.int32)`: `int1` to `int32`; `tl.sum` of the `int32` block
      -- is the wrapping `int32` sum (`foldAll`, no widening); `n += …` the wrapping `int32`
      -- addition
      let c ← tl.convert .i1 .i32 (geX dt logits x)
      pure (n + Tensor.foldAll (· + ·) (0 : I32) c)
    -- line 67: the `int32` count converted to the pointer's dtype, `int64` (sign extension)
    let n64 : I64 := (← tl.convert .i32 .i64 (Tensor.scalar n)).item
    tl.store1 (output_ptr + req_idx) n64

/-! ## The conditions -/

/-- What the host chooses: one launch through `compute_token_ranks`, with the contents of its two
input tensors. -/
structure Inst where
  /-- The dtype of `logits`. -/
  dt : DType
  /-- `batch_size = logits.shape[0]`: the grid, one program per row. -/
  B : Nat
  /-- `vocab_size = logits.shape[1]` (the argument `vocab_size`). -/
  V : Nat
  /-- `logits.stride(0)` (the argument `logits_stride`), in elements; the last dimension is
  contiguous. -/
  stride : Nat
  /-- `BLOCK_SIZE` (the host's is `8192`). -/
  BLOCK : Nat
  /-- `logits`: element `(r, j)` at `r * stride + j`. -/
  logits : Buf dt
  /-- The entries of `logits` (flat). -/
  L : Nat → dt.Val
  /-- `sampled_token_ids`: `[batch_size]`, `int64`, contiguous. -/
  tok : Buf .i64
  /-- The entries of `sampled_token_ids`. -/
  T : Nat → I64
  /-- `token_ranks`, allocated fresh by the host: `[batch_size]`, `int64`. -/
  out : Buf .i64

/-- The instances covered: the three logits dtypes, every block size up to `2^20`, every
vocabulary whose `int32` loop does not overflow, every `int32` stride, every grid CUDA accepts
(`0` rows included), buffers that hold what the launch touches, and an output sharing no byte with
an input. The inputs may overlap one another, and the rows may overlap (any stride). -/
structure Inst.ok (d : Inst) : Prop where
  /-- The logits are `float32`, `bfloat16` or `float16`. -/
  dt_ok : d.dt = .f32 ∨ d.dt = .bf16 ∨ d.dt = .f16
  /-- `BLOCK_SIZE` is a power of two (`tl.arange`). -/
  blk_pow2 : ∃ e, d.BLOCK = 2 ^ e
  /-- `BLOCK_SIZE ≤ 2^20` (Triton's largest block). -/
  blk_le : d.BLOCK ≤ 2 ^ 20
  /-- `vocab_size + BLOCK_SIZE ≤ 2^31`: `vocab_size` is an `int32` and the `int32` loop's last
  increment `i + BLOCK_SIZE` does not overflow (`tl.RangeOk`; a limit of the kernel: README,
  Exclusions). -/
  V_blk : d.V + d.BLOCK ≤ 2 ^ 31
  /-- The stride is an `int32` runtime argument. -/
  stride_lt : d.stride < 2 ^ 31
  /-- Fewer than `2^31` rows (the grid, `Launch.Valid`). -/
  B_lt : d.B < 2 ^ 31
  /-- `logits` holds every row: element `(r, j)`, `j < vocab_size`, for `r < batch_size`. -/
  logits_len : ∀ r, r < d.B → r * d.stride + d.V ≤ d.logits.len
  /-- `sampled_token_ids` has one entry per row. -/
  tok_len : d.B ≤ d.tok.len
  /-- The output has one entry per row. -/
  out_len : d.B ≤ d.out.len
  /-- The output shares no byte with `logits` (the host allocates it). -/
  out_logits : Disjoint d.out.bytes d.logits.bytes
  /-- The output shares no byte with `sampled_token_ids`. -/
  out_tok : Disjoint d.out.bytes d.tok.bytes

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte present)
and hold the instance's entries; every sampled token id is a vocabulary index, in
`[0, vocab_size)`. Every logit value (NaN and infinities included) is covered. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.logits.Valid init ∧ d.tok.Valid init ∧ d.out.Valid init ∧
    d.logits.Holds init d.L ∧ d.tok.Holds init d.T ∧
    ∀ r, r < d.B → 0 ≤ (d.T r).toInt ∧ (d.T r).toInt < d.V

/-- The model on an instance: the buffers' first elements, the two scalar arguments as `int32`,
and the constexpr. -/
def model (d : Inst) : Kernel :=
  _ranks_kernel d.dt (d.out.ptr 0) (d.logits.ptr 0) (IntV.ofNat d.stride) (d.tok.ptr 0)
    (IntV.ofNat d.V) d.BLOCK

/-- The host's launch (source lines 75–84): a 1-D grid of `batch_size` programs, the block shape
`[BLOCK_SIZE]`; `enable_fp_fusion = true` (Triton's default; the kernel has no float arithmetic),
`debug = false`, `num_ctas = 1`, no `launch_pdl`. `num_warps = 4` is performance-only;
`num_stages = 3` changes no outcome of a model without a conflict inside a program. -/
def launch (d : Inst) : Launch := Launch.grid1 d.B [[d.BLOCK]]

/-! ## The specification -/

/-- `x ≤ y` on two values of the logits' dtype: the ordered comparison (`FloatV.le`), false when
either is NaN. -/
def geB : (dt : DType) → dt.Val → dt.Val → Bool
  | .f32, x, y => FloatV.le x y
  | .bf16, x, y => FloatV.le x y
  | .f16, x, y => FloatV.le x y
  | _, _, _ => false

/-- `-inf` in the logits' dtype. -/
def negInf : (dt : DType) → dt.Val
  | .f32 => FloatV.negInf
  | .bf16 => FloatV.negInf
  | .f16 => FloatV.negInf
  | dt => dt.ofBits 0

/-- Row `r`'s sampled token id, as a vocabulary index. -/
def tokOf (d : Inst) (r : Nat) : Nat := (d.T r).toInt.toNat

/-- Entry `j` of row `r` of the initial logits. -/
def logitAt (d : Inst) (r j : Nat) : d.dt.Val := d.L (r * d.stride + j)

/-- The logit `x` of row `r`'s sampled token. -/
def xOf (d : Inst) (r : Nat) : d.dt.Val := logitAt d r (tokOf d r)

/-- The number of trips of the loop, `⌈vocab_size / BLOCK_SIZE⌉`. -/
def numTrips (d : Inst) : Nat := (d.V + d.BLOCK - 1) / d.BLOCK

/-- The lanes of the last block past `vocab_size`, which hold `-inf`. -/
def padLanes (d : Inst) : Nat := numTrips d * d.BLOCK - d.V

/-- The number of entries `j < vocab_size` of row `r` at least the token's logit. -/
def countGe (d : Inst) (r : Nat) : Nat :=
  ((Finset.range d.V).filter (fun j => geB d.dt (xOf d r) (logitAt d r j) = true)).card

/-- What the kernel stores for row `r`: `countGe`, plus the padding lanes when the token's logit
is `-inf` (exactly when `x ≤ -inf`). -/
def rank (d : Inst) (r : Nat) : Nat :=
  countGe d r + (if geB d.dt (xOf d r) (negInf d.dt) = true then padLanes d else 0)

/-- What the launch leaves: element `r` of `token_ranks` is the `int64` `rank d r` for every row
`r < batch_size`; every byte outside those elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r : Nat, r < d.B → ∃ v : I64,
      fin.read .i64 (d.out.ptr r).addr = some v ∧ v.toInt = (rank d r : Int)) ∧
    ∀ z, z ∉ d.out.elems {r | r < d.B} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_ranks_kernel` ends normally with `Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- An instance of the host's launch with the buffers at fixed, separate addresses (the output at
`2^32`, the token ids at `2^36`, the logits at `2^40`) and zero contents: `dt` the logits' dtype,
`B` rows of `V` entries with row stride `S`, and `BLOCK_SIZE = 2^e`. -/
def mk (dt : DType) (B V S e : Nat) : Inst where
  dt := dt
  B := B
  V := V
  stride := S
  BLOCK := 2 ^ e
  logits := ⟨2 ^ 40, B * S + V⟩
  L := fun _ => dt.ofBits 0
  tok := ⟨2 ^ 36, B⟩
  T := fun _ => DType.ofBits .i64 0
  out := ⟨2 ^ 32, B⟩

theorem bytes_disj {d d' : DType} (b : Buf d) (c : Buf d')
    (h : b.base + b.len * d.bytes ≤ c.base ∨ c.base + c.len * d'.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem mk_ok {dt : DType} {B V S e : Nat} (hdt : dt = .f32 ∨ dt = .bf16 ∨ dt = .f16)
    (he : e ≤ 20) (hV : V + 2 ^ e ≤ 2 ^ 31) (hS : S < 2 ^ 31) (hB : B * 8 ≤ 2 ^ 32) :
    (mk dt B V S e).ok where
  dt_ok := hdt
  blk_pow2 := ⟨e, rfl⟩
  blk_le := Nat.pow_le_pow_right (by norm_num) he
  V_blk := hV
  stride_lt := hS
  B_lt := by simp only [mk]; omega
  logits_len := fun r hr => by
    simp only [mk]
    have : r * S ≤ B * S := Nat.mul_le_mul_right _ (le_of_lt hr)
    omega
  tok_len := le_refl _
  out_len := le_refl _
  out_logits := bytes_disj _ _ (Or.inl (by simp only [mk, DType.bytes]; push_cast; omega))
  out_tok := bytes_disj _ _ (Or.inl (by simp only [mk, DType.bytes]; push_cast; omega))

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

theorem valid_zeroMem {d : DType} (b : Buf d) (h0 : 0 ≤ b.base)
    (h1 : b.base + b.len * d.bytes ≤ 2 ^ 64) (h2 : b.base % d.bytes = 0) : b.Valid zeroMem :=
  ⟨h0, h1, h2, fun _ _ => by simp [zeroMem]⟩

theorem holds_zeroMem {d : DType} (b : Buf d) (hv : b.Valid zeroMem) :
    b.Holds zeroMem (fun _ => d.ofBits 0) :=
  fun _ hj => zeroMem_read (b.aligned hv hj)

/-- One `float32` row of one entry, token `0`, on the all-zero memory. -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk .f32 1 1 1 13, zeroMem,
    mk_ok (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  have hl : (mk .f32 1 1 1 13).logits.Valid zeroMem := by
    apply valid_zeroMem <;> simp only [mk, DType.bytes] <;> norm_num
  have ht : (mk .f32 1 1 1 13).tok.Valid zeroMem := by
    apply valid_zeroMem <;> simp only [mk, DType.bytes] <;> norm_num
  have ho : (mk .f32 1 1 1 13).out.Valid zeroMem := by
    apply valid_zeroMem <;> simp only [mk, DType.bytes] <;> norm_num
  refine ⟨hl, ht, ho, holds_zeroMem _ hl, holds_zeroMem _ ht, ?_⟩
  intro r hr
  simp only [mk] at hr ⊢
  decide

/-! ## The setup's launches

The host's test cases (source lines 101–117) and the code base's launches (`SETUP.md`), all with
the host's `BLOCK_SIZE = 8192 = 2^13`. Names: `<case>_<rows>x<vocab>[_s<stride>]_<dtype>`. -/

/-- Host test: 1 row of Llama-2's vocabulary, `float32`. -/
def setup_test_1x32000_f32 : Inst := mk .f32 1 32000 32000 13
theorem setup_test_1x32000_f32_ok : setup_test_1x32000_f32.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Host test: 8 rows of Llama-3's vocabulary, `float32`. -/
def setup_test_8x128256_f32 : Inst := mk .f32 8 128256 128256 13
theorem setup_test_8x128256_f32_ok : setup_test_8x128256_f32.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Host test: 64 rows of Qwen3's vocabulary, `bfloat16`. -/
def setup_test_64x151936_bf16 : Inst := mk .bf16 64 151936 151936 13
theorem setup_test_64x151936_bf16_ok : setup_test_64x151936_bf16.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Host test: 256 rows of Llama-2's vocabulary, `float16`. -/
def setup_test_256x32000_f16 : Inst := mk .f16 256 32000 32000 13
theorem setup_test_256x32000_f16_ok : setup_test_256x32000_f16.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Host test: 1024 rows of one block (`vocab_size = BLOCK_SIZE`), `float32`. -/
def setup_test_1024x8192_f32 : Inst := mk .f32 1024 8192 8192 13
theorem setup_test_1024x8192_f32_ok : setup_test_1024x8192_f32.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Host test: 3 rows of Gemma-3's vocabulary, `bfloat16`. -/
def setup_test_3x262144_bf16 : Inst := mk .bf16 3 262144 262144 13
theorem setup_test_3x262144_bf16_ok : setup_test_3x262144_bf16.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- vLLM's regression test of the `int64` row offset (`tests/v1/sample/test_logprobs.py`):
`2^31 / 131072 + 64 = 16448` rows of `131072`, `float32` (`batch_size · vocab_size > 2^31`). -/
def setup_vllm_int64_offset_16448x131072_f32 : Inst := mk .f32 16448 131072 131072 13
theorem setup_vllm_int64_offset_16448x131072_f32_ok : setup_vllm_int64_offset_16448x131072_f32.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- The sampler on a decode batch of 1024 requests, processed `float32` logits of Llama-3. -/
def setup_sampler_1024x128256_f32 : Inst := mk .f32 1024 128256 128256 13
theorem setup_sampler_1024x128256_f32_ok : setup_sampler_1024x128256_f32.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- The sampler on raw `bfloat16` logits of Qwen3, 256 requests. -/
def setup_sampler_256x151936_bf16 : Inst := mk .bf16 256 151936 151936 13
theorem setup_sampler_256x151936_bf16_ok : setup_sampler_256x151936_bf16.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- The sampler on raw `float16` logits of Llama-2, 256 requests. -/
def setup_sampler_256x32000_f16 : Inst := mk .f16 256 32000 32000 13
theorem setup_sampler_256x32000_f16_ok : setup_sampler_256x32000_f16.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- The sampler on the slice `logits[:, :vocab_size]` of a padded vocabulary (GPT-2's `50257`
padded to `50304`), 64 requests, `float32`. -/
def setup_sampler_64x50257_s50304_f32 : Inst := mk .f32 64 50257 50304 13
theorem setup_sampler_64x50257_s50304_f32_ok : setup_sampler_64x50257_s50304_f32.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- The rejection sampler on a speculative batch: 512 requests with 4 drafts each give `2560`
rows (`num_reqs · (k + 1)`), `float32`, Llama-3. -/
def setup_spec_2560x128256_f32 : Inst := mk .f32 2560 128256 128256 13
theorem setup_spec_2560x128256_f32_ok : setup_spec_2560x128256_f32.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Prompt logprobs: a chunk of `CHUNK_SIZE = 1024` prompt positions, raw `bfloat16` logits of
Qwen3. -/
def setup_prompt_1024x151936_bf16 : Inst := mk .bf16 1024 151936 151936 13
theorem setup_prompt_1024x151936_bf16_ok : setup_prompt_1024x151936_bf16.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Prompt logprobs: the last, shorter chunk (`37` positions), `float32`, Gemma-3. -/
def setup_prompt_37x262144_f32 : Inst := mk .f32 37 262144 262144 13
theorem setup_prompt_37x262144_f32_ok : setup_prompt_37x262144_f32.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- An empty batch (`batch_size = 0`): the grid has no program. -/
def setup_empty_0x32000_f32 : Inst := mk .f32 0 32000 32000 13
theorem setup_empty_0x32000_f32_ok : setup_empty_0x32000_f32.ok :=
  mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Claims.LogprobRanks
