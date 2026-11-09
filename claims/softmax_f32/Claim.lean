import TVA

/-!
# `softmax_f32`: the prepared claim for tutorial 02's `softmax_kernel`

Source: `source/02-fused-softmax.py`. The kernel is `softmax_kernel` (lines
84–109); it reads the module-level names `triton` (its decorator) and `tl`, bound by the imports of
lines 26–27. Its host function `softmax` (lines 124–175) launches it, through the `CompiledKernel`
of `softmax_kernel.warmup(…)`, on `float32` matrices: `1823 × 781` (the test, line 187) and
`4096 × N` for `N = 256, 384, …, 12672` (the benchmark, lines 203–229); `SETUP.md` lists them.

The claim (`Statement`): on every covered instance, every launch that runs ends normally; every
output element `y[r, j]` holds a `float32` value; for every row whose inputs are finite or `-∞`
with at least one finite (`RowOk`), there is one factor `sc` for the row, within
`sumTol BLOCK_SIZE` of `1`, such that each output of the row is finite and within
`(elemTol + gapTol · gap_j) · σ_j + floorTol · σ_max` of `sc · σ_j`, `σ` the exact softmax of the
row; every other byte is unchanged.

* The model: `softmax_kernel`, `model`, `launch` (the model map is `model-map.json`).
* The conditions: `Inst`, `Inst.ok`, `Pre`. The specification: `RowOk`, `softmaxRef`,
  `softmaxMax`, `softmaxGap`, the tolerances `sumTol`, `elemTol`, `gapTol`, `floorTol`, `Post`,
  `Spec`.
* The assumptions: seven `[rule]` fields (`Compiler.Faithful`, the library's facts for
  `tl.exp` and `/` on `float32`, and the depth of a `tl.sum`'s tree).
* Proved here: `consistent`, `satisfiable`, and the coverage of the script's launches
  (`setup_*_ok`). `Statement` is the claim to prove.
-/

namespace Claims.Softmax

open TVA TVA.Math.Float

/-! ## The model -/

set_option linter.unusedVariables false in
/-- `softmax_kernel` (source lines 84–109). The pointers are `*fp32` (the host passes `float32`
tensors). `input_row_stride`, `output_row_stride`, `n_rows` and `n_cols` are `int32`: each is an
integer argument in `[0, 2^31)`, a runtime `int32` except when it equals `1`, where Triton passes
the compile-time constant `1`, which each use makes an `int32` tensor first, so the model is the
same (`model`'s docstring). `BLOCK_SIZE` and `num_stages` are `tl.constexpr` parameters, Lean
parameters here; `num_stages` is used only as the scheduling keyword of `tl.range`, which changes no
outcome of a model with no conflict within a program (SEMANTICS.md, "Loops"), so the model does not
use it. -/
def softmax_kernel (output_ptr input_ptr : Ptr .f32)
    (input_row_stride output_row_stride n_rows n_cols : I32)
    (BLOCK_SIZE : Nat) (num_stages : Nat) : Kernel where
  body ctx := do
    let row_start := tl.program_id ctx 0
    let row_step := tl.num_programs ctx 0
    -- `tl.range(row_start, n_rows, row_step, num_stages=num_stages)`: the `int32` loop, no carried
    -- name (every name the body assigns is first defined there and not used after the loop)
    tl.forRange row_start n_rows row_step () fun row_idx _ => do
      -- `row_idx * input_row_stride`: two `int32` values, the wrapping `int32` product
      let row_start_ptr := input_ptr + row_idx * input_row_stride
      let col_offsets := tl.arange 0 BLOCK_SIZE
      let input_ptrs := row_start_ptr + col_offsets
      -- the `int32` tensor against the `int32` scalar `n_cols`, signed
      let mask := col_offsets.ltS n_cols
      -- `other=-float('inf')`: the Python `float` `-inf` (the double `0xFFF0000000000000`)
      -- becomes a `float32` tensor, `litF32` of it (`0xFF800000`), already the pointer's dtype
      let row ← tl.load input_ptrs (some mask) (some (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000))))
      -- `row - tl.max(row, axis=0)`: the reduction of the `float32` block (no widening), then
      -- the scalar broadcast to every lane and the IEEE `-` (no product feeds it)
      let row_minus_max ← tl.sub row (bc (← tl.maxAll true row))
      let numerator ← tl.exp row_minus_max
      -- the input is `tl.exp`'s result, not a product: IEEE additions
      let denominator ← tl.sum numerator
      -- `float32 / float32`: the platform's division, the scalar broadcast to every lane
      let softmax_output ← tl.div numerator (bc denominator)
      let output_row_start_ptr := output_ptr + row_idx * output_row_stride
      let output_ptrs := output_row_start_ptr + col_offsets
      -- `softmax_output` is already `float32`, the pointer's dtype: no conversion
      tl.store output_ptrs softmax_output (some mask)

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch of `softmax_kernel`. -/
structure Inst where
  /-- `n_rows` (the host passes `x.shape[0]`). -/
  n_rows : Nat
  /-- `n_cols` (the host passes `x.shape[1]`). -/
  n_cols : Nat
  /-- `input_row_stride` (the host passes `x.stride(0)`), in elements. -/
  input_row_stride : Nat
  /-- `output_row_stride` (the host passes `y.stride(0)`), in elements. -/
  output_row_stride : Nat
  /-- `BLOCK_SIZE` (the host passes `triton.next_power_of_2(n_cols)`). -/
  BLOCK_SIZE : Nat
  /-- `num_stages` (the host passes `4` or `2`). -/
  num_stages : Nat
  /-- The number of programs, the grid `(num_programs, 1, 1)` (the host passes
  `min(NUM_SM * occupancy, n_rows)`). -/
  num_programs : Nat
  /-- `input_ptr`: the `float32` tensor `x`. -/
  x : Buf .f32
  /-- `output_ptr`: the `float32` tensor `y`. -/
  y : Buf .f32

/-- Element `j` of row `r` of the input: `x[r, j]` at `r * input_row_stride + j`. -/
def inIdx (d : Inst) (r j : Nat) : Nat := r * d.input_row_stride + j

/-- Element `j` of row `r` of the output: `y[r, j]` at `r * output_row_stride + j`. -/
def outIdx (d : Inst) (r j : Nat) : Nat := r * d.output_row_stride + j

/-- The input elements the launch reads: `x[r, j]` for `r < n_rows`, `j < n_cols`. -/
def inSet (d : Inst) : Set Nat := {k | ∃ r j, r < d.n_rows ∧ j < d.n_cols ∧ k = inIdx d r j}

/-- The output elements the launch writes: `y[r, j]` for `r < n_rows`, `j < n_cols`. -/
def outSet (d : Inst) : Set Nat := {k | ∃ r j, r < d.n_rows ∧ j < d.n_cols ∧ k = outIdx d r j}

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `BLOCK_SIZE` is a power of two (Triton requires it of `tl.arange`). -/
  block_pow2 : ∃ e, d.BLOCK_SIZE = 2 ^ e
  /-- `BLOCK_SIZE ≤ 2^14 = 16384`, the block size of the host's largest row (`12672` columns). -/
  block_le : d.BLOCK_SIZE ≤ 2 ^ 14
  /-- Each row fits in one block (the host's `BLOCK_SIZE` is the next power of two). -/
  cols_le : d.n_cols ≤ d.BLOCK_SIZE
  /-- `n_cols` is a model dimension (the width of the row-wise softmax): a positive multiple of
  `128`, at most `2^17` (README, Exclusions). -/
  cols_dim : 0 < d.n_cols ∧ d.n_cols % 128 = 0 ∧ d.n_cols ≤ 2 ^ 17
  /-- With a row to process, there is a program (the host's `min(NUM_SM * occupancy, n_rows)`
  is positive then). -/
  grid_pos : d.n_rows ≠ 0 → 0 < d.num_programs
  /-- The grid has fewer than `2^31` programs. -/
  grid_lt : d.num_programs < 2 ^ 31
  /-- `n_rows + num_programs ≤ 2^31`: `n_rows` is an `int32`, and the `int32` loop
  `tl.range(row_start, n_rows, row_step)` does not overflow its last increment. -/
  rows_grid : d.n_rows + d.num_programs ≤ 2 ^ 31
  /-- `input_row_stride` is an `int32`. -/
  in_stride_lt : d.input_row_stride < 2 ^ 31
  /-- `output_row_stride` is an `int32`. -/
  out_stride_lt : d.output_row_stride < 2 ^ 31
  /-- The `int32` product `row_idx * input_row_stride` does not wrap for any row. -/
  in_off : (d.n_rows - 1) * d.input_row_stride < 2 ^ 31
  /-- The `int32` product `row_idx * output_row_stride` does not wrap for any row. -/
  out_off : (d.n_rows - 1) * d.output_row_stride < 2 ^ 31
  /-- `x` holds every element the launch reads. -/
  x_len : ∀ r j, r < d.n_rows → j < d.n_cols → inIdx d r j < d.x.len
  /-- `y` holds every element the launch writes. -/
  y_len : ∀ r j, r < d.n_rows → j < d.n_cols → outIdx d r j < d.y.len
  /-- The output rows do not overlap: one row, or rows at least `n_cols` apart. -/
  out_rows : d.n_rows ≤ 1 ∨ d.n_cols ≤ d.output_row_stride
  /-- The output elements share no byte with the input elements. -/
  y_x : Disjoint (d.y.elems (outSet d)) (d.x.elems (inSet d))

/-- The model on an instance: the kernel's arguments are the buffers' first elements, the four
integers as `int32`, and the two constexprs. An integer argument equal to `1` is passed as the
compile-time constant `1`: `row_idx * 1` builds it in `int32` (the dtype of the tensor operand),
`col_offsets < 1` and the bound `1` of `tl.range` make it an `int32` tensor (SEMANTICS.md,
"Constants", "Loops"), so the model is the same. -/
def model (d : Inst) : Kernel :=
  softmax_kernel (d.y.ptr 0) (d.x.ptr 0) (IntV.ofNat d.input_row_stride)
    (IntV.ofNat d.output_row_stride) (IntV.ofNat d.n_rows) (IntV.ofNat d.n_cols) d.BLOCK_SIZE
    d.num_stages

/-- The host's launch (source line 174): the grid `(num_programs, 1, 1)`. The source's only block
shape is `[BLOCK_SIZE]` (`col_offsets`, `input_ptrs`, `mask`, `row`, `row_minus_max`,
`numerator`, `softmax_output`, `output_ptrs`). `enable_fp_fusion` is Triton's default (`true`);
`grid1` keeps the defaults `num_ctas = 1`, `launch_pdl = False` and `debug = false`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 d.num_programs [[d.BLOCK_SIZE]] (fusion := true)

/-- The inputs covered: both buffers are valid (aligned, inside `[0, 2^64)`, every byte present).
Every `float32` input value is covered, NaN and infinities included: `Post` constrains a row whose
values are not `RowOk` less. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.y.Valid init

/-! ## The specification -/

/-- A row `xs 0, …, xs (n-1)` whose softmax is defined: every value is a finite `float32` or `-∞`,
and, when the row is not empty, one value is finite. -/
def RowOk (xs : Nat → F32) (n : Nat) : Prop :=
  (∀ k, k < n → (xs k).isFinite = true ∨ xs k = FloatV.negInf) ∧
    (0 < n → ∃ k, k < n ∧ (xs k).isFinite = true)

/-- `e^a` of an input element `a`, with `e^(-∞) = 0`. -/
noncomputable def expIn (a : F32) : ℝ := if a = FloatV.negInf then 0 else Real.exp a.toReal

/-- The softmax of the row `xs 0, …, xs (n-1)` at column `j`: `e^(xs j) / Σ_{k<n} e^(xs k)`,
exactly, in the reals. -/
noncomputable def softmaxRef (xs : Nat → F32) (n j : Nat) : ℝ :=
  expIn (xs j) / ∑ k ∈ Finset.range n, expIn (xs k)

/-- The row's largest exact softmax value, `max_k softmaxRef xs n k` (the softmax at the row's
largest input, `e^(x_max) / Σ_k e^(x_k)`); `0` for an empty row. -/
noncomputable def softmaxMax (xs : Nat → F32) (n : Nat) : ℝ := ⨆ k : Fin n, softmaxRef xs n k

/-- How far column `j` lies below the row's largest output, on a log scale:
`log(softmaxMax / σ_j)`, which is `x_max − x_j` for a finite input `x_j` (and `0` for an input
`-∞`, whose `σ_j` is `0`). -/
noncomputable def softmaxGap (xs : Nat → F32) (n j : Nat) : ℝ :=
  Real.log (softmaxMax xs n / softmaxRef xs n j)

/-- The tolerance of a row's common factor at block size `B`: `(⌈log₂ B⌉ + 56)·2^-24` (`≈ 4.2·10^-6`
at `B = 2^14`). It is the row sum's error, along a tree of depth at most `⌈log₂ B⌉`
(`Rules.Num.SumDepth`: `⌈log₂ B⌉·2^-24` and second-order terms), and the lanes' `tl.exp` errors
summed over the row (`53·2^-24`); one factor for the whole row. -/
def sumTol (B : Nat) : ℚ := ((Nat.clog 2 B : ℚ) + 56) * 2 ^ (-24 : ℤ)

/-- The relative tolerance of each output: `2^-21 + 2^-30` (`tl.exp`'s `2^-22` and the division's
`2^-22`). -/
def elemTol : ℚ := 2 ^ (-21 : ℤ) + 2 ^ (-30 : ℤ)

/-- The relative tolerance per unit of `softmaxGap` (`x_max − x_j`): `3·2^-24 + 2^-30` (the
rounding of `x_j − x_max` and the part of `tl.exp`'s error that grows with its argument). -/
def gapTol : ℚ := 3 * 2 ^ (-24 : ℤ) + 2 ^ (-30 : ℤ)

/-- The tolerance relative to the row's largest output: `2^-133` (`tl.exp`'s absolute `2^-140`
and the division's `2^-148`). -/
def floorTol : ℚ := 2 ^ (-133 : ℤ)

/-- What the launch leaves: for each row `r < n_rows`, with `xs` the row's input values as read in
the initial memory, every output element `y[r, j]`, `j < n_cols`, holds a `float32` value; when the
row is `RowOk`, there is one factor `sc` for the whole row, within `sumTol BLOCK_SIZE` of `1`, such
that each output element is a finite `float32` within
`(elemTol + gapTol · softmaxGap) · σ + floorTol · softmaxMax` of `sc · σ`, where
`σ = softmaxRef xs n_cols j` is the exact softmax; every byte outside the output elements is
unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r, r < d.n_rows → ∃ xs : Nat → F32,
      (∀ k, k < d.n_cols → init.read .f32 (d.x.ptr (inIdx d r k)).addr = some (xs k)) ∧
      (∀ j, j < d.n_cols → ∃ c : F32, fin.read .f32 (d.y.ptr (outIdx d r j)).addr = some c) ∧
      (RowOk xs d.n_cols → ∃ sc : ℝ, |sc - 1| ≤ (sumTol d.BLOCK_SIZE : ℝ) ∧
        ∀ j, j < d.n_cols → ∃ c : F32, fin.read .f32 (d.y.ptr (outIdx d r j)).addr = some c ∧
          c.isFinite = true ∧
          |c.toReal - sc * softmaxRef xs d.n_cols j| ≤
            ((elemTol : ℝ) + (gapTol : ℝ) * softmaxGap xs d.n_cols j) * softmaxRef xs d.n_cols j +
              (floorTol : ℝ) * softmaxMax xs d.n_cols)) ∧
    ∀ z, z ∉ d.y.elems (outSet d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from which
  the model's launch is well behaved, is one of the outcomes of the model's launch. The rule covers
  a launch within its scope: part (A), the kernel and its launches, which the host's launches meet
  (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.MathTotal`: `tl.exp` on `float32` has a result for every input. -/
  exp_total : Rules.Num.MathTotal P .exp .binary32
  /-- [rule] `Num.ExpF32`: the accuracy of `tl.exp` on `float32`. -/
  exp_acc : Rules.Num.ExpF32 P
  /-- [rule] `Num.ExpF32NegInf`: `tl.exp(-∞) = +0` on `float32`. -/
  exp_ninf : Rules.Num.ExpF32NegInf P
  /-- [rule] `Num.DivTotal`: `/` on `float32` has a result for every pair of inputs. -/
  div_total : Rules.Num.DivTotal P .binary32
  /-- [rule] `Num.DivFull`: `/` on `float32` is `div.full.f32`. -/
  div_full : Rules.Num.DivFull P
  /-- [rule] `Num.SumDepth`: the tree of the float `tl.sum` (line 104) over its `BLOCK_SIZE`
  positions has depth at most `⌈log₂ BLOCK_SIZE⌉`. Its result is stored, not carried: no loop
  updates a carried value with a reduction. -/
  sum_depth : Rules.Num.SumDepth P

/-- **The claim to prove.** -/
def Statement : Prop := ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.ref.mathTotal,
    Platform.sm90ref.expF32, Platform.sm90ref.expF32NegInf, Platform.ref.divTotal _,
    Platform.ref.divFull, Platform.sm90ref.sumDepth⟩⟩

/-- The host's launch on a contiguous `R × C` matrix (`x.stride(0) = y.stride(0) = C`), with
`BLOCK_SIZE = 2^e`, `num_stages` `ns` and `np` programs: `x` at `2^34` and a fresh `y` at `2^35`,
`R · C` elements each. -/
def mk (R C e ns np : Nat) : Inst where
  n_rows := R
  n_cols := C
  input_row_stride := C
  output_row_stride := C
  BLOCK_SIZE := 2 ^ e
  num_stages := ns
  num_programs := np
  x := ⟨2 ^ 34, R * C⟩
  y := ⟨2 ^ 35, R * C⟩

theorem idx_lt {R C r j : Nat} (hr : r < R) (hj : j < C) : r * C + j < R * C := by
  have : (r + 1) * C ≤ R * C := Nat.mul_le_mul_right _ hr
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- Every such launch is covered when the block holds the row, the row is a positive multiple of
`128` columns, there are between `1` and `R` programs (and `R + np ≤ 2^31`), the last row starts
below element `2^31`, and the matrix fits below `2^34` bytes. -/
theorem mk_ok {R C e ns np : Nat} (he : e ≤ 14) (hC : C ≤ 2 ^ e) (hnp : 0 < np) (hnpR : np ≤ R)
    (hRnp : R + np ≤ 2 ^ 31) (hoff : (R - 1) * C < 2 ^ 31) (hsz : R * C * 4 ≤ 2 ^ 34)
    (hCd : 0 < C ∧ C % 128 = 0) :
    (mk R C e ns np).ok := by
  have hC14 : C ≤ 2 ^ 14 := by
    have : 2 ^ e ≤ 2 ^ 14 := Nat.pow_le_pow_right (by norm_num) he
    omega
  have hC31 : C < 2 ^ 31 := by omega
  refine ⟨⟨e, rfl⟩, Nat.pow_le_pow_right (by norm_num) he, hC, ⟨hCd.1, hCd.2, by simp only [mk]; omega⟩,
    fun _ => hnp, by simp only [mk]; omega,
    hRnp, hC31, hC31, hoff, hoff, fun r j hr hj => idx_lt hr hj, fun r j hr hj => idx_lt hr hj,
    Or.inr (le_refl _), ?_⟩
  refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
  · rw [Set.disjoint_left]
    intro z hz hz'
    simp only [Buf.bytes, Set.mem_ofPred_eq, mk, DType.bytes] at hz hz'
    have : ((R * C : Nat) : Int) * 4 ≤ 2 ^ 34 := by exact_mod_cast hsz
    push_cast at this hz hz'
    omega
  · rintro k ⟨r, j, hr, hj, rfl⟩; exact idx_lt hr hj
  · rintro k ⟨r, j, hr, hj, rfl⟩; exact idx_lt hr hj

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨mk 4096 256 8 4 1056, testMem,
    mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num), ?_, ?_⟩
  all_goals
    exact ⟨by norm_num [mk], by norm_num [mk, DType.bytes], by norm_num [mk, DType.bytes],
      fun _ _ => by simp [testMem]⟩

/-! ## Coverage of the script's launches (`SETUP.md`)

`softmax(x)` on a contiguous `float32` matrix: `BLOCK_SIZE = next_power_of_2(n_cols)`,
`num_stages = 4` (the H200's `max_shared_mem` is above `200000`), and `num_programs =
min(NUM_SM · occupancy, n_rows)`, which depends on the compiled kernel's registers; the instances
take the H200's values (`check_gpu.py` prints them), and `mk_ok` covers every `num_programs` from
`1` to `n_rows`. -/

def setup_bench_4096x256 : Inst := mk 4096 256 8 4 1056
theorem setup_bench_4096x256_ok : setup_bench_4096x256.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x384 : Inst := mk 4096 384 9 4 1056
theorem setup_bench_4096x384_ok : setup_bench_4096x384.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x512 : Inst := mk 4096 512 9 4 1056
theorem setup_bench_4096x512_ok : setup_bench_4096x512.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x640 : Inst := mk 4096 640 10 4 1056
theorem setup_bench_4096x640_ok : setup_bench_4096x640.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x768 : Inst := mk 4096 768 10 4 1056
theorem setup_bench_4096x768_ok : setup_bench_4096x768.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x896 : Inst := mk 4096 896 10 4 1056
theorem setup_bench_4096x896_ok : setup_bench_4096x896.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x1024 : Inst := mk 4096 1024 10 4 1056
theorem setup_bench_4096x1024_ok : setup_bench_4096x1024.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x1152 : Inst := mk 4096 1152 11 4 1056
theorem setup_bench_4096x1152_ok : setup_bench_4096x1152.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x1280 : Inst := mk 4096 1280 11 4 1056
theorem setup_bench_4096x1280_ok : setup_bench_4096x1280.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x1408 : Inst := mk 4096 1408 11 4 1056
theorem setup_bench_4096x1408_ok : setup_bench_4096x1408.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x1536 : Inst := mk 4096 1536 11 4 1056
theorem setup_bench_4096x1536_ok : setup_bench_4096x1536.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x1664 : Inst := mk 4096 1664 11 4 1056
theorem setup_bench_4096x1664_ok : setup_bench_4096x1664.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x1792 : Inst := mk 4096 1792 11 4 1056
theorem setup_bench_4096x1792_ok : setup_bench_4096x1792.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x1920 : Inst := mk 4096 1920 11 4 1056
theorem setup_bench_4096x1920_ok : setup_bench_4096x1920.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x2048 : Inst := mk 4096 2048 11 4 1056
theorem setup_bench_4096x2048_ok : setup_bench_4096x2048.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x2176 : Inst := mk 4096 2176 12 4 528
theorem setup_bench_4096x2176_ok : setup_bench_4096x2176.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x2304 : Inst := mk 4096 2304 12 4 528
theorem setup_bench_4096x2304_ok : setup_bench_4096x2304.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x2432 : Inst := mk 4096 2432 12 4 528
theorem setup_bench_4096x2432_ok : setup_bench_4096x2432.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x2560 : Inst := mk 4096 2560 12 4 528
theorem setup_bench_4096x2560_ok : setup_bench_4096x2560.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x2688 : Inst := mk 4096 2688 12 4 528
theorem setup_bench_4096x2688_ok : setup_bench_4096x2688.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x2816 : Inst := mk 4096 2816 12 4 528
theorem setup_bench_4096x2816_ok : setup_bench_4096x2816.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x2944 : Inst := mk 4096 2944 12 4 528
theorem setup_bench_4096x2944_ok : setup_bench_4096x2944.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x3072 : Inst := mk 4096 3072 12 4 528
theorem setup_bench_4096x3072_ok : setup_bench_4096x3072.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x3200 : Inst := mk 4096 3200 12 4 528
theorem setup_bench_4096x3200_ok : setup_bench_4096x3200.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x3328 : Inst := mk 4096 3328 12 4 528
theorem setup_bench_4096x3328_ok : setup_bench_4096x3328.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x3456 : Inst := mk 4096 3456 12 4 528
theorem setup_bench_4096x3456_ok : setup_bench_4096x3456.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x3584 : Inst := mk 4096 3584 12 4 528
theorem setup_bench_4096x3584_ok : setup_bench_4096x3584.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x3712 : Inst := mk 4096 3712 12 4 528
theorem setup_bench_4096x3712_ok : setup_bench_4096x3712.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x3840 : Inst := mk 4096 3840 12 4 528
theorem setup_bench_4096x3840_ok : setup_bench_4096x3840.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x3968 : Inst := mk 4096 3968 12 4 528
theorem setup_bench_4096x3968_ok : setup_bench_4096x3968.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x4096 : Inst := mk 4096 4096 12 4 528
theorem setup_bench_4096x4096_ok : setup_bench_4096x4096.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x4224 : Inst := mk 4096 4224 13 4 264
theorem setup_bench_4096x4224_ok : setup_bench_4096x4224.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x4352 : Inst := mk 4096 4352 13 4 264
theorem setup_bench_4096x4352_ok : setup_bench_4096x4352.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x4480 : Inst := mk 4096 4480 13 4 264
theorem setup_bench_4096x4480_ok : setup_bench_4096x4480.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x4608 : Inst := mk 4096 4608 13 4 264
theorem setup_bench_4096x4608_ok : setup_bench_4096x4608.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x4736 : Inst := mk 4096 4736 13 4 264
theorem setup_bench_4096x4736_ok : setup_bench_4096x4736.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x4864 : Inst := mk 4096 4864 13 4 264
theorem setup_bench_4096x4864_ok : setup_bench_4096x4864.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x4992 : Inst := mk 4096 4992 13 4 264
theorem setup_bench_4096x4992_ok : setup_bench_4096x4992.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x5120 : Inst := mk 4096 5120 13 4 264
theorem setup_bench_4096x5120_ok : setup_bench_4096x5120.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x5248 : Inst := mk 4096 5248 13 4 264
theorem setup_bench_4096x5248_ok : setup_bench_4096x5248.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x5376 : Inst := mk 4096 5376 13 4 264
theorem setup_bench_4096x5376_ok : setup_bench_4096x5376.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x5504 : Inst := mk 4096 5504 13 4 264
theorem setup_bench_4096x5504_ok : setup_bench_4096x5504.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x5632 : Inst := mk 4096 5632 13 4 264
theorem setup_bench_4096x5632_ok : setup_bench_4096x5632.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x5760 : Inst := mk 4096 5760 13 4 264
theorem setup_bench_4096x5760_ok : setup_bench_4096x5760.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x5888 : Inst := mk 4096 5888 13 4 264
theorem setup_bench_4096x5888_ok : setup_bench_4096x5888.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x6016 : Inst := mk 4096 6016 13 4 264
theorem setup_bench_4096x6016_ok : setup_bench_4096x6016.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x6144 : Inst := mk 4096 6144 13 4 264
theorem setup_bench_4096x6144_ok : setup_bench_4096x6144.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x6272 : Inst := mk 4096 6272 13 4 264
theorem setup_bench_4096x6272_ok : setup_bench_4096x6272.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x6400 : Inst := mk 4096 6400 13 4 264
theorem setup_bench_4096x6400_ok : setup_bench_4096x6400.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x6528 : Inst := mk 4096 6528 13 4 264
theorem setup_bench_4096x6528_ok : setup_bench_4096x6528.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x6656 : Inst := mk 4096 6656 13 4 264
theorem setup_bench_4096x6656_ok : setup_bench_4096x6656.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x6784 : Inst := mk 4096 6784 13 4 264
theorem setup_bench_4096x6784_ok : setup_bench_4096x6784.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x6912 : Inst := mk 4096 6912 13 4 264
theorem setup_bench_4096x6912_ok : setup_bench_4096x6912.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x7040 : Inst := mk 4096 7040 13 4 264
theorem setup_bench_4096x7040_ok : setup_bench_4096x7040.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x7168 : Inst := mk 4096 7168 13 4 264
theorem setup_bench_4096x7168_ok : setup_bench_4096x7168.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x7296 : Inst := mk 4096 7296 13 4 264
theorem setup_bench_4096x7296_ok : setup_bench_4096x7296.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x7424 : Inst := mk 4096 7424 13 4 264
theorem setup_bench_4096x7424_ok : setup_bench_4096x7424.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x7552 : Inst := mk 4096 7552 13 4 264
theorem setup_bench_4096x7552_ok : setup_bench_4096x7552.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x7680 : Inst := mk 4096 7680 13 4 264
theorem setup_bench_4096x7680_ok : setup_bench_4096x7680.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x7808 : Inst := mk 4096 7808 13 4 264
theorem setup_bench_4096x7808_ok : setup_bench_4096x7808.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x7936 : Inst := mk 4096 7936 13 4 264
theorem setup_bench_4096x7936_ok : setup_bench_4096x7936.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x8064 : Inst := mk 4096 8064 13 4 264
theorem setup_bench_4096x8064_ok : setup_bench_4096x8064.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x8192 : Inst := mk 4096 8192 13 4 264
theorem setup_bench_4096x8192_ok : setup_bench_4096x8192.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x8320 : Inst := mk 4096 8320 14 4 132
theorem setup_bench_4096x8320_ok : setup_bench_4096x8320.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x8448 : Inst := mk 4096 8448 14 4 132
theorem setup_bench_4096x8448_ok : setup_bench_4096x8448.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x8576 : Inst := mk 4096 8576 14 4 132
theorem setup_bench_4096x8576_ok : setup_bench_4096x8576.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x8704 : Inst := mk 4096 8704 14 4 132
theorem setup_bench_4096x8704_ok : setup_bench_4096x8704.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x8832 : Inst := mk 4096 8832 14 4 132
theorem setup_bench_4096x8832_ok : setup_bench_4096x8832.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x8960 : Inst := mk 4096 8960 14 4 132
theorem setup_bench_4096x8960_ok : setup_bench_4096x8960.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x9088 : Inst := mk 4096 9088 14 4 132
theorem setup_bench_4096x9088_ok : setup_bench_4096x9088.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x9216 : Inst := mk 4096 9216 14 4 132
theorem setup_bench_4096x9216_ok : setup_bench_4096x9216.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x9344 : Inst := mk 4096 9344 14 4 132
theorem setup_bench_4096x9344_ok : setup_bench_4096x9344.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x9472 : Inst := mk 4096 9472 14 4 132
theorem setup_bench_4096x9472_ok : setup_bench_4096x9472.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x9600 : Inst := mk 4096 9600 14 4 132
theorem setup_bench_4096x9600_ok : setup_bench_4096x9600.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x9728 : Inst := mk 4096 9728 14 4 132
theorem setup_bench_4096x9728_ok : setup_bench_4096x9728.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x9856 : Inst := mk 4096 9856 14 4 132
theorem setup_bench_4096x9856_ok : setup_bench_4096x9856.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x9984 : Inst := mk 4096 9984 14 4 132
theorem setup_bench_4096x9984_ok : setup_bench_4096x9984.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x10112 : Inst := mk 4096 10112 14 4 132
theorem setup_bench_4096x10112_ok : setup_bench_4096x10112.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x10240 : Inst := mk 4096 10240 14 4 132
theorem setup_bench_4096x10240_ok : setup_bench_4096x10240.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x10368 : Inst := mk 4096 10368 14 4 132
theorem setup_bench_4096x10368_ok : setup_bench_4096x10368.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x10496 : Inst := mk 4096 10496 14 4 132
theorem setup_bench_4096x10496_ok : setup_bench_4096x10496.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x10624 : Inst := mk 4096 10624 14 4 132
theorem setup_bench_4096x10624_ok : setup_bench_4096x10624.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x10752 : Inst := mk 4096 10752 14 4 132
theorem setup_bench_4096x10752_ok : setup_bench_4096x10752.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x10880 : Inst := mk 4096 10880 14 4 132
theorem setup_bench_4096x10880_ok : setup_bench_4096x10880.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x11008 : Inst := mk 4096 11008 14 4 132
theorem setup_bench_4096x11008_ok : setup_bench_4096x11008.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x11136 : Inst := mk 4096 11136 14 4 132
theorem setup_bench_4096x11136_ok : setup_bench_4096x11136.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x11264 : Inst := mk 4096 11264 14 4 132
theorem setup_bench_4096x11264_ok : setup_bench_4096x11264.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x11392 : Inst := mk 4096 11392 14 4 132
theorem setup_bench_4096x11392_ok : setup_bench_4096x11392.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x11520 : Inst := mk 4096 11520 14 4 132
theorem setup_bench_4096x11520_ok : setup_bench_4096x11520.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x11648 : Inst := mk 4096 11648 14 4 132
theorem setup_bench_4096x11648_ok : setup_bench_4096x11648.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x11776 : Inst := mk 4096 11776 14 4 132
theorem setup_bench_4096x11776_ok : setup_bench_4096x11776.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x11904 : Inst := mk 4096 11904 14 4 132
theorem setup_bench_4096x11904_ok : setup_bench_4096x11904.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x12032 : Inst := mk 4096 12032 14 4 132
theorem setup_bench_4096x12032_ok : setup_bench_4096x12032.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x12160 : Inst := mk 4096 12160 14 4 132
theorem setup_bench_4096x12160_ok : setup_bench_4096x12160.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x12288 : Inst := mk 4096 12288 14 4 132
theorem setup_bench_4096x12288_ok : setup_bench_4096x12288.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x12416 : Inst := mk 4096 12416 14 4 132
theorem setup_bench_4096x12416_ok : setup_bench_4096x12416.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x12544 : Inst := mk 4096 12544 14 4 132
theorem setup_bench_4096x12544_ok : setup_bench_4096x12544.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_bench_4096x12672 : Inst := mk 4096 12672 14 4 132
theorem setup_bench_4096x12672_ok : setup_bench_4096x12672.ok := mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Claims.Softmax
