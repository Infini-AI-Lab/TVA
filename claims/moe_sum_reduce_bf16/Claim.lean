import TVA

/-!
# `moe_sum_reduce_bf16`: the claim for SGLang's `_moe_sum_reduce_kernel`

Source: `source/moe_sum_reduce.py`. The kernel is
`_moe_sum_reduce_kernel` (lines 50–104); it reads the module-level names `triton` and `tl`
(lines 46–47). The host `moe_sum_reduce` (lines 107–146) launches it on a contiguous `bfloat16`
input `[token_num, topk_num, hidden_dim]` and a fresh contiguous `bfloat16` output
`[token_num, hidden_dim]`, with `BLOCK_M = 1`, `BLOCK_DIM = 2048`, `NUM_STAGE = 1`,
`USE_GDC = False`, on the grid `(cdiv(token_num, BLOCK_M), cdiv(hidden_dim, BLOCK_DIM))`.

For every `(token t, column j)` the kernel sums the token's `topk_num` rows in `float32`, multiplies
by `routed_scaling_factor` (a compile-time `float`, built as a `float32` constant) and stores the
`bfloat16` rounding.

The claim (`Statement`): on every covered instance, every launch that runs ends normally; every
output element `(t, j)` holds a `bfloat16` value, and when its `topk_num` inputs are finite, the
scale is finite and the magnitudes leave room in the formats (`InRange`), that value is finite and
within `tol` of the exact `σ · Σ_k input[t, k, j]` (`σ` the host's double); every other byte is
unchanged.

* The model: `moe_sum_reduce_kernel`, `model`, `launch` (the model map is `model-map.json`).
* The conditions: `Inst`, `Inst.ok`, `Pre`. The specification: `xv`, `refSum`, `absSum`,
  `InRange`, `tol`, `Post`, `Spec`.
* The assumptions: one `[rule]` field, `Compiler.Faithful` (the float operations are IEEE `+`,
  `*` and conversions, which need no device fact).
* Proved here: `consistent`, `satisfiable`, and the coverage of the host's launches. `Statement`
  is the claim to prove.
-/

namespace Claims.MoeSumReduce

open TVA TVA.Math.Float

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `_moe_sum_reduce_kernel` (source lines 50–104). The pointers are `*bfloat16`. The strides are
unannotated integer arguments: an `int32` (at the value `1` a compile-time constant, which
`tl.cast(…, dtype=tl.int64)` makes the same `int64` `1`); `input_stride_2` and `output_stride_1`
are never read. `token_num`, `topk_num`, `hidden_dim` are annotated `int`: runtime `int32`s.
`routed_scaling_factor` is a `tl.constexpr` Python `float`: as an operand of `*` with the `float32`
accumulator it is the `float32` constant `FloatV.litF32` of the double (`model`). `BLOCK_M`,
`BLOCK_DIM` are Lean parameters; `NUM_STAGE` (a `tl.range` scheduling keyword) changes no outcome,
and `USE_GDC` is `False` on every covered launch (lines 85–87 not visited). -/
def moe_sum_reduce_kernel (input_ptr : Ptr .bf16)
    (input_stride_0 input_stride_1 input_stride_2 : I32)
    (output_ptr : Ptr .bf16) (output_stride_0 output_stride_1 : I32)
    (token_num topk_num hidden_dim : I32)
    (routed_scaling_factor : F32) (BLOCK_M BLOCK_DIM : Nat) : Kernel where
  body ctx := do
    -- line 68: `tl.cast(input_stride_0, dtype=tl.int64)`: `semantic.cast` of the `int32` scalar to
    -- `int64`, the conversion `tl.convert .i32 .i64` (sign extension; SEMANTICS.md, `tl.cast`)
    let s0 := (← tl.convert .i32 .i64 (Tensor.scalar input_stride_0)).item
    -- line 69: the same for `input_stride_1`
    let s1 := (← tl.convert .i32 .i64 (Tensor.scalar input_stride_1)).item
    -- line 70: the same for `output_stride_0`
    let os0 := (← tl.convert .i32 .i64 (Tensor.scalar output_stride_0)).item
    -- lines 72–73: the two grid axes, `int32`
    let token_block_id := tl.program_id ctx 0
    let dim_block_id := tl.program_id ctx 1
    -- lines 75–76: the constexprs built in `int32`; wrapping `int32` products and sums
    let offs_token := token_block_id * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
    let offs_dim := dim_block_id * (IntV.ofNat BLOCK_DIM : I32) + tl.arange 0 BLOCK_DIM
    -- lines 78–79: signed `int32` comparisons with the `int32` scalars
    let mask_token := offs_token.ltS token_num
    let mask_dim := offs_dim.ltS hidden_dim
    -- line 81: `offs_token[:, None] * input_stride_0 + offs_dim[None, :]`: the `int32` offsets
    -- promoted to the `int64` of the cast strides (sign extension), the wrapping `int64` product
    -- and sum, broadcast to `[BLOCK_M, BLOCK_DIM]`, added to the pointer in elements
    let offs0 : Tensor [BLOCK_M] I64 ← tl.convert .i32 .i64 offs_token
    let offs1 : Tensor [BLOCK_DIM] I64 ← tl.convert .i32 .i64 offs_dim
    let base_ptrs := input_ptr + (offs0.col BLOCK_DIM * s0 + offs1.row BLOCK_M)
    -- line 83: the `float32` zero block
    let accumulator := tl.zeros [BLOCK_M, BLOCK_DIM] .f32
    -- lines 85–87: not visited (`USE_GDC = False`)
    -- line 89: `tl.range(0, topk_num, num_stages=NUM_STAGE)`, the `int32` induction variable
    -- `i`; the loop carries `accumulator`
    let accumulator ← tl.forRange (IntV.ofInt 0 : I32) topk_num (IntV.ofInt 1 : I32) accumulator
      fun i accumulator => do
        -- line 91: `i * input_stride_1`: `i` promoted to `int64`, the wrapping `int64` product
        let i64 := (← tl.convert .i32 .i64 (Tensor.scalar i)).item
        -- lines 90–94: the masked load; `other=0.0` is the `float32` `+0` converted to `bfloat16`
        let tile ← tl.load (base_ptrs + (bc (i64 * s1) : Tensor [BLOCK_M, BLOCK_DIM] I64))
          (some ((mask_token.col BLOCK_DIM).and (mask_dim.row BLOCK_M)))
          (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_M, BLOCK_DIM] (FloatV.litF32 (FloatV.ofPat 0)))))
        -- line 95: `tile.to(tl.float32)` (exact), then the IEEE `+` (no product feeds it)
        let tile32 : Tensor [BLOCK_M, BLOCK_DIM] F32 ← tl.cvt tile
        let acc ← tl.add accumulator tile32
        return acc
    -- line 96: the IEEE `float32` product with the constant (no product feeds a `+` here)
    let acc2 ← tl.mul accumulator (bc routed_scaling_factor)
    -- line 102: `accumulator.to(input_ptr.dtype.element_ty)`: one rounding to nearest even
    let out : Tensor [BLOCK_M, BLOCK_DIM] BF16 ← tl.cvt acc2
    -- line 99: the store's pointers, as line 81 with `output_stride_0`
    let store_ptrs := output_ptr + (offs0.col BLOCK_DIM * os0 + offs1.row BLOCK_M)
    -- lines 100–104: the masked store
    tl.store store_ptrs out (some ((mask_token.col BLOCK_DIM).and (mask_dim.row BLOCK_M)))

/-! ## The conditions -/

/-- What the host chooses: one launch through `moe_sum_reduce(input, routed_scaling_factor)`,
with the input's contents. -/
structure Inst where
  /-- `token_num` (`input.shape[0]`). -/
  T : Nat
  /-- `topk_num` (`input.shape[1]`). -/
  K : Nat
  /-- `hidden_dim` (`input.shape[2]`). -/
  D : Nat
  /-- `BLOCK_M` (the host's `1`). -/
  BLOCK_M : Nat
  /-- `BLOCK_DIM` (the host's `2048`). -/
  BLOCK_DIM : Nat
  /-- `routed_scaling_factor`: the bit pattern of the Python `float` (a double). -/
  sbits : Nat
  /-- `input`, contiguous: element `(t, k, j)` is `t·K·D + k·D + j`. -/
  input : Buf .bf16
  /-- Its contents. -/
  inVal : Nat → BF16
  /-- `output`, fresh and contiguous: element `(t, j)` is `t·D + j`. -/
  output : Buf .bf16

/-- The scale as the host passes it: the double. -/
def sigma (d : Inst) : F64 := FloatV.ofPat d.sbits

/-- The scale as the kernel multiplies by it: the `float32` constant Triton builds for the
constexpr (`FloatV.litF32` of the double: rounding to nearest even, `+0` for a zero). -/
def sigma32 (d : Inst) : F32 := FloatV.litF32 (sigma d)

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `BLOCK_M`, `BLOCK_DIM` powers of two (`tl.arange`), each block at most `2^20` elements. -/
  bm_pow2 : ∃ e, d.BLOCK_M = 2 ^ e
  bd_pow2 : ∃ e, d.BLOCK_DIM = 2 ^ e
  block_le : d.BLOCK_M * d.BLOCK_DIM ≤ 2 ^ 20
  /-- At least one token, expert row and column (`SETUP.md`: `token_num` from 1, `topk_num` 1 to
  9, `hidden_dim` the model's hidden size). -/
  T_pos : 0 < d.T
  K_pos : 0 < d.K
  D_pos : 0 < d.D
  /-- The `int` arguments are `int32` values. -/
  T_lt : d.T < 2 ^ 31
  K_lt : d.K < 2 ^ 31
  /-- `topk_num ≤ 2^20` (real use: at most 9): keeps `(1+2^-24)^K` below `2`, so the
  accumulation's error bound leaves room in `float32` (README, Exclusions). -/
  K_le : d.K ≤ 2 ^ 20
  D_lt : d.D < 2 ^ 31
  /-- `hidden_dim` is a model dimension: a multiple of `128`, at most `2^17` (README, Exclusions). -/
  D_dim : d.D % 128 = 0 ∧ d.D ≤ 2 ^ 17
  /-- `token_num`, the tokens of the batch, is a context length: at most `2^22` (README,
  Exclusions). -/
  T_ctx : d.T ≤ 2 ^ 22
  /-- `input.stride(0) = topk_num·hidden_dim` is passed as an `int32` (README, Exclusions). -/
  stride_lt : d.K * d.D < 2 ^ 31
  /-- The grid's axis 1, `cdiv(hidden_dim, BLOCK_DIM)`, is below CUDA's `2^16` (a launch limit:
  README, Exclusions). -/
  grid1_lt : (d.D + d.BLOCK_DIM - 1) / d.BLOCK_DIM < 2 ^ 16
  /-- The number of programs, `cdiv(token_num, BLOCK_M) · cdiv(hidden_dim, BLOCK_DIM)`, is below
  `2^31` (a launch limit: README, Exclusions). -/
  grid_size_lt : (d.T + d.BLOCK_M - 1) / d.BLOCK_M * ((d.D + d.BLOCK_DIM - 1) / d.BLOCK_DIM) < 2 ^ 31
  /-- The double's pattern is 64 bits. -/
  sbits_lt : d.sbits < 2 ^ 64
  /-- The buffers hold the tensors. -/
  input_len : d.T * d.K * d.D ≤ d.input.len
  output_len : d.T * d.D ≤ d.output.len
  /-- `output` is a fresh allocation: it shares no byte with `input`. -/
  disj : Disjoint d.output.bytes d.input.bytes

/-- The inputs covered: the buffers are valid and `input` holds the instance's values. Every
`bfloat16` value is covered (NaN and infinities included: `Post` constrains such elements less). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.input.Valid init ∧ d.output.Valid init ∧ d.input.Holds init d.inVal

/-- The model on an instance: the host's contiguous strides `(K·D, D, 1)` and `(D, 1)`, the sizes
as `int32`, the scale's `float32` constant. -/
def model (d : Inst) : Kernel :=
  moe_sum_reduce_kernel (d.input.ptr 0) (IntV.ofNat (d.K * d.D)) (IntV.ofNat d.D) (IntV.ofNat 1)
    (d.output.ptr 0) (IntV.ofNat d.D) (IntV.ofNat 1)
    (IntV.ofNat d.T) (IntV.ofNat d.K) (IntV.ofNat d.D) (sigma32 d) d.BLOCK_M d.BLOCK_DIM

/-- The host's launch (lines 125–146): the grid `(cdiv(T, BLOCK_M), cdiv(D, BLOCK_DIM), 1)`; the
block shapes `[BLOCK_M]`, `[BLOCK_DIM]`, `[BLOCK_M, 1]`, `[1, BLOCK_DIM]`, `[BLOCK_M, BLOCK_DIM]`;
`enable_fp_fusion = true` (the default), `debug = false`, `num_ctas = 1`, no `launch_pdl`.
`num_warps = 16` is performance-only; `num_stages = 3` changes no outcome of a model without a
conflict inside a program. -/
def launch (d : Inst) : Launch where
  grid := ![(d.T + d.BLOCK_M - 1) / d.BLOCK_M, (d.D + d.BLOCK_DIM - 1) / d.BLOCK_DIM, 1]
  fusion := true
  blocks := [[d.BLOCK_M], [d.BLOCK_DIM], [d.BLOCK_M, 1], [1, d.BLOCK_DIM], [d.BLOCK_M, d.BLOCK_DIM]]

/-! ## The specification

The rounding constants are the library's (`Format.binary32`, `Format.bfloat16`: `u`, `eta`). -/

/-- One rounding of `float32`: `2^-24` relative (`Format.binary32.u`), `2^-150` absolute
(`Format.binary32.eta`). -/
def u32 : ℚ := 2 ^ (-24 : ℤ)
def eta32 : ℚ := 2 ^ (-150 : ℤ)
/-- One rounding of `bfloat16`: `2^-8` relative, `2^-134` absolute. -/
def ubf : ℚ := 2 ^ (-8 : ℤ)
def etabf : ℚ := 2 ^ (-134 : ℤ)

theorem u32_eq : u32 = Format.binary32.u := by unfold u32; rw [Format.binary32_u]
theorem eta32_eq : eta32 = Format.binary32.eta := by unfold eta32; rw [Format.binary32_eta]
theorem ubf_eq : ubf = Format.bfloat16.u := by unfold ubf; rw [Format.bfloat16_u]
theorem etabf_eq : etabf = Format.bfloat16.eta := by unfold etabf; rw [Format.bfloat16_eta]

/-- Input `(t, k, j)` as the host's tensor holds it. -/
def xv (d : Inst) (t k j : Nat) : BF16 := d.inVal (t * d.K * d.D + k * d.D + j)

/-- The exact sum over the token's expert rows: `Σ_k input[t, k, j]`. -/
def refSum (d : Inst) (t j : Nat) : ℚ := ∑ k ∈ Finset.range d.K, (xv d t k j).toRat

/-- The sum of their magnitudes, `Σ_k |input[t, k, j]|`. -/
def absSum (d : Inst) (t j : Nat) : ℚ := ∑ k ∈ Finset.range d.K, |(xv d t k j).toRat|

/-- The element is in range: its inputs and the scale are finite, and the magnitudes leave room
in `float32` and `bfloat16` (`Σ|x| ≤ 2^126` and `|σ₃₂|·Σ|x| ≤ 2^126`; both formats overflow
near `2^128`). -/
def InRange (d : Inst) (t j : Nat) : Prop :=
  (∀ k, k < d.K → (xv d t k j).isFinite = true) ∧ (sigma d).isFinite = true ∧
    (sigma32 d).isFinite = true ∧ absSum d t j ≤ 2 ^ 126 ∧
    |(sigma32 d).toRat| * absSum d t j ≤ 2 ^ 126

/-- The error of the `float32` accumulation of `K` values from `+0` (`K` IEEE additions, a chain
of depth `K`; `Tree.errBound`): `((1+u)^K − 1)·A + η·K·(1+u)^K`. -/
def accErr (K : Nat) (A : ℚ) : ℚ := Tree.errBound u32 eta32 K K A

/-- **The tolerance** of an output element, from `S = Σx` (exact), `A = Σ|x|`, `σ` (the double)
and `σ₃₂` (its `float32` constant):
* `E1 = accErr K A`, the accumulation's error, so `|acc| ≤ |S| + E1`;
* `Em = u·|σ₃₂|·(|S| + E1) + η`, one `float32` rounding of `acc·σ₃₂`;
* `Q = |σ₃₂|·|S| + |σ₃₂|·E1 + Em`, a bound on the product `p`;
* `Eb = u_bf·Q + η_bf`, one `bfloat16` rounding of `p`;
* `tol = Eb + Em + |σ₃₂|·E1 + |σ − σ₃₂|·|S|`: the last term is the scale's own conversion
  (the reference uses the host's double). -/
def tol (K : Nat) (S A σ σ32 : ℚ) : ℚ :=
  let E1 := accErr K A
  let Em := u32 * (|σ32| * (|S| + E1)) + eta32
  let Q := |σ32| * |S| + |σ32| * E1 + Em
  (ubf * Q + etabf) + Em + |σ32| * E1 + |σ - σ32| * |S|

/-- What the launch leaves: every output element `(t, j)` holds a `bfloat16` value, which, when the
element is `InRange`, is finite and within `tol` of `σ · Σ_k input[t, k, j]`; every byte outside
the output's `T·D` elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t j, t < d.T → j < d.D → ∃ c : BF16,
      fin.read .bf16 (d.output.ptr (t * d.D + j)).addr = some c ∧
      (InRange d t j → c.isFinite = true ∧
        |c.toRat - (sigma d).toRat * refSum d t j| ≤
          tol d.K (refSum d t j) (absSum d t j) (sigma d).toRat (sigma32 d).toRat)) ∧
    ∀ z, z ∉ d.output.elems {e | e < d.T * d.D} → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)

/-- **The claim to prove.** -/
def Statement : Prop := ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

/-- An instance of the host's launch (`BLOCK_M = 2^em`, `BLOCK_DIM = 2^ed`) with `input` at
`2^32`, `output` at `2^36` and zero contents. -/
def mk (T K D sbits em ed : Nat) : Inst where
  T := T
  K := K
  D := D
  BLOCK_M := 2 ^ em
  BLOCK_DIM := 2 ^ ed
  sbits := sbits
  input := ⟨2 ^ 32, T * K * D⟩
  inVal := fun _ => FloatV.ofPat 0
  output := ⟨2 ^ 36, T * D⟩

theorem mk_ok {T K D sbits em ed : Nat} (hb : 2 ^ em * 2 ^ ed ≤ 2 ^ 20) (hT : 0 < T) (hK : 0 < K)
    (hD : 0 < D) (hT' : T < 2 ^ 31) (hK' : K < 2 ^ 31) (hKle : K ≤ 2 ^ 20) (hD' : D < 2 ^ 31) (hKD : K * D < 2 ^ 31)
    (hg : (D + 2 ^ ed - 1) / 2 ^ ed < 2 ^ 16)
    (hgs : (T + 2 ^ em - 1) / 2 ^ em * ((D + 2 ^ ed - 1) / 2 ^ ed) < 2 ^ 31) (hs : sbits < 2 ^ 64) (hsz : T * K * D * 2 ≤ 2 ^ 35)
    (hDd : D % 128 = 0 ∧ D ≤ 2 ^ 17) (hTc : T ≤ 2 ^ 22) :
    (mk T K D sbits em ed).ok where
  bm_pow2 := ⟨em, rfl⟩
  bd_pow2 := ⟨ed, rfl⟩
  block_le := hb
  T_pos := hT
  K_pos := hK
  D_pos := hD
  T_lt := hT'
  K_lt := hK'
  K_le := hKle
  D_lt := hD'
  D_dim := hDd
  T_ctx := hTc
  stride_lt := hKD
  grid1_lt := hg
  grid_size_lt := hgs
  sbits_lt := hs
  input_len := le_refl _
  output_len := le_refl _
  disj := by
    rw [Set.disjoint_left]
    intro x hx hy
    simp only [mk, Buf.bytes, Set.mem_ofPred_eq, DType.bytes] at hx hy
    have : ((T * K * D : Nat) : Int) * 2 ≤ 2 ^ 35 := by exact_mod_cast hsz
    push_cast at this hx hy
    omega

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨mk 1 1 128 4607182418800017408 0 11, zeroMem,
    mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num), ?_⟩
  have hb2 : (DType.bf16.bytes : Int) = 2 := rfl
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_⟩, ?_⟩
  · simp [mk]
  · simp only [mk, hb2]; norm_num
  · simp only [mk, hb2]; decide
  · intro x _; simp [zeroMem]
  · simp [mk]
  · simp only [mk, hb2]; norm_num
  · simp only [mk, hb2]; decide
  · intro x _; simp [zeroMem]
  · intro i hi
    simp only [mk] at hi
    have ha : Mem.Aligned .bf16 ((2 : Int) ^ 32 + (i : Int) * (DType.bf16.bytes : Int)) := by
      unfold Mem.Aligned; simp only [hb2]; omega
    have hb : zeroMem.readBytes ((2 : Int) ^ 32 + (i : Int) * (DType.bf16.bytes : Int)) 2 =
        some [0, 0] := by simp [Mem.readBytes, zeroMem]
    show zeroMem.read .bf16 ((2 : Int) ^ 32 + (i : Int) * (DType.bf16.bytes : Int)) =
      some (FloatV.ofPat 0)
    unfold Mem.read
    simp only [ha, ↓reduceIte]
    rw [show DType.bf16.bytes = 2 from rfl] at hb ⊢
    rw [hb]
    decide

/-! ## Coverage of the host's launches

The host's test launches (source lines 158–168), `(token_num, topk_num, hidden_dim, scale)`,
with the host's `BLOCK_M = 1`, `BLOCK_DIM = 2048` and the scale's double pattern; and the extremes
`SETUP.md` names (a batch of `2^16` tokens of top-9 at hidden size 7168 for the Marlin and LoRA
runners). The contents are not part of `Inst.ok`. -/

def setup_1_8_7168 : Inst := mk 1 8 7168 4612811918334230528 0 11
theorem setup_1_8_7168_ok : setup_1_8_7168.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num)
def setup_7_9_4096 : Inst := mk 7 9 4096 4599075939470750515 0 11
theorem setup_7_9_4096_ok : setup_7_9_4096.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num)
def setup_32_8_2048 : Inst := mk 32 8 2048 4607182418800017408 0 11
theorem setup_32_8_2048_ok : setup_32_8_2048.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num)
def setup_33_6_5120 : Inst := mk 33 6 5120 4607182418800017408 0 11
theorem setup_33_6_5120_ok : setup_33_6_5120.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num)
def setup_1024_9_4096 : Inst := mk 1024 9 4096 4599075939470750515 0 11
theorem setup_1024_9_4096_ok : setup_1024_9_4096.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num)
def setup_8192_8_7168 : Inst := mk 8192 8 7168 4612811918334230528 0 11
theorem setup_8192_8_7168_ok : setup_8192_8_7168.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num)
def setup_65536_9_7168 : Inst := mk 65536 9 7168 4612811918334230528 0 11
theorem setup_65536_9_7168_ok : setup_65536_9_7168.ok := mk_ok (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num)

end Claims.MoeSumReduce
