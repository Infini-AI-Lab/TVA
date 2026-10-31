import TVA

/-!
# `softmax_f32`: the model, the conditions, the specification and the assumptions

Source: `source/02-fused-softmax.py`, tutorial 02. The kernel is `softmax_kernel` (lines 84–109);
it reads the module-level names `triton` (its decorator) and `tl`, bound by the imports of lines
26–27. Its host function `softmax` (lines 124–175) launches it, through the `CompiledKernel` of
`softmax_kernel.warmup(…)`, on `float32` matrices: `1823 × 781` (the test, line 187) and
`4096 × N` for `N = 256, 384, …, 12672` (the benchmark, lines 203–229).

* The model: `softmax_kernel`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with the reference `softmaxRef` (its row maximum
  `softmaxMax`, the gap `softmaxGap`) and the tolerances `sumTol`, `elemTol`, `gapTol`,
  `floorTol`.
* The assumptions: `Assumptions` (six `[rule]` fields).

The proof is in `Softmax.Err` (the error of one row) and `Softmax.Proof`.
-/

namespace Kernels.Softmax

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

/-- An input value covered: a finite `float32` of magnitude at most `2^126`, or `-∞`. -/
def InputOk (a : F32) : Prop :=
  (a.isFinite = true ∧ |a.toRat| ≤ 2 ^ 126) ∨ a = FloatV.negInf

/-- The inputs covered: both buffers are valid (aligned, inside `[0, 2^64)`, every byte present);
every input element the launch reads is finite with magnitude at most `2^126`, or `-∞`; and each
row with a column has a finite element. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.y.Valid init ∧
    ∀ r, r < d.n_rows →
      (∀ j, j < d.n_cols → ∃ a : F32, init.read .f32 (d.x.ptr (inIdx d r j)).addr = some a ∧ InputOk a) ∧
      (0 < d.n_cols → ∃ j, j < d.n_cols ∧ ∃ a : F32,
        init.read .f32 (d.x.ptr (inIdx d r j)).addr = some a ∧ a.isFinite = true)

/-! ## The specification -/

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

/-- The tolerance of a row's common factor at block size `B`: `(B + 86)·2^-24` (`≈ 9.8·10^-4` at
`B = 2^14`). It is the row sum's error, along a tree of depth up to `B − 1` (`(B − 1)·2^-24` and
second-order terms) and the lanes' `tl.exp` errors summed over the row (`53·2^-24`); one factor
for the whole row. -/
def sumTol (B : Nat) : ℚ := ((B : ℚ) + 86) * 2 ^ (-24 : ℤ)

/-- The relative tolerance of each output: `2^-21 + 2^-30` (`tl.exp`'s `2^-22` and the division's
`2^-22`). -/
def elemTol : ℚ := 2 ^ (-21 : ℤ) + 2 ^ (-30 : ℤ)

/-- The relative tolerance per unit of `softmaxGap` (`x_max − x_j`): `3·2^-24 + 2^-30` (the
rounding of `x_j − x_max` and the part of `tl.exp`'s error that grows with its argument). -/
def gapTol : ℚ := 3 * 2 ^ (-24 : ℤ) + 2 ^ (-30 : ℤ)

/-- The tolerance relative to the row's largest output: `2^-133` (`tl.exp`'s absolute `2^-140`
and the division's `2^-148`). -/
def floorTol : ℚ := 2 ^ (-133 : ℤ)

/-- What the launch leaves: for each row `r < n_rows`, with `xs` the row's input values, there is
one factor `sc` for the whole row, within `sumTol BLOCK_SIZE` of `1`, such that each output
element `y[r, j]`, `j < n_cols`, is a finite `float32` within
`(elemTol + gapTol · softmaxGap) · σ + floorTol · softmaxMax` of `sc · σ`, where
`σ = softmaxRef xs n_cols j` is the exact softmax; every byte outside the output elements is
unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r, r < d.n_rows → ∃ xs : Nat → F32,
      (∀ k, k < d.n_cols → init.read .f32 (d.x.ptr (inIdx d r k)).addr = some (xs k)) ∧
      ∃ sc : ℝ, |sc - 1| ≤ (sumTol d.BLOCK_SIZE : ℝ) ∧
        ∀ j, j < d.n_cols → ∃ c : F32, fin.read .f32 (d.y.ptr (outIdx d r j)).addr = some c ∧
          c.isFinite = true ∧
          |c.toReal - sc * softmaxRef xs d.n_cols j| ≤
            ((elemTol : ℝ) + (gapTol : ℝ) * softmaxGap xs d.n_cols j) * softmaxRef xs d.n_cols j +
              (floorTol : ℝ) * softmaxMax xs d.n_cols) ∧
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

end Kernels.Softmax
