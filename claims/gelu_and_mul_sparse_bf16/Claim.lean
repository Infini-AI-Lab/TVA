import TVA

/-!
# `gelu_and_mul_sparse_bf16`: Gaussian-sparse GELU-and-multiply (Gemma 3n), vLLM

The kernel is `_gelu_and_mul_sparse_kernel` of `source/activation.py` (lines 56–92; it reads the
module-level names `triton` and `tl`, bound by the imports of lines 49–50). Its host
`_gelu_and_mul_sparse_triton_op` (lines 113–134) launches it on the grid `(n_rows,)`, one program
per row of `x = [gate | up]` (`2·d` elements, `d = x.shape[-1] // 2`), with
`BLOCK_SIZE = next_power_of_2(d)`, into a fresh output of `n_rows·d` elements. vLLM's Gemma 3n MLP
(`GeluAndMulSparse`) calls it on `bfloat16` activations; `SETUP.md` lists the launches.

What is claimed: every covered launch ends normally, and for every row `t < n_rows` and column
`j < d`, with `g j`, `u j` the row's initial `gate` and `up` values, `M`, `A`, `Sd` the exact mean,
mean magnitude and (biased) standard deviation of `g`, `m` the host's multiplier and
`cRef = M + m·Sd` the exact cutoff:
* `output[t, j]` is a finite `bfloat16` within `yTol j = 2^-5·|yRef j| + 2·|u j|·cutTol + 2^-132`
  of `yRef j = geluRef (max (g j − cRef) 0) · u j`, the exact tanh-approximate GELU of the exact
  sparse gate times `up` (`cutTol` bounds the kernel's rounded cutoff against `cRef`);
* where `g j ≤ cRef − cutTol` (the gate is below every cutoff the kernel may compute),
  `output[t, j]` is a zero;
* every other byte is unchanged.

The conditions: `d` from 1 to `32768` (the host's limit) with `BLOCK_SIZE` its next power of two;
`n_rows` from 1 to `2^31 − 1` (CUDA's grid limit); the multiplier a finite double with `|m| ≤ 64`;
`x` holds `n_rows·2d` elements and `output` `n_rows·d`, sharing no byte; every element of `x` is a
finite `bfloat16` with `|v| ≤ 2^32`; every row's gate is not nearly constant:
`2^-18·A + 2^-62 ≤ Sd`.

The tolerances rest on the device facts for `tl.sqrt`, `tl.exp` and `/` on `float32`
(`Device.SqrtF32`, `Device.ExpF32`, `Device.DivFull`) and on the depth of a `tl.sum`'s tree
(`Device.SumDepth`); README, Tolerances derives them.
-/

namespace Claims.GeluAndMulSparse

open TVA

/-! ## The model -/

/-- The `float32` constant Triton builds for a Python `float` operand of a `float32` operation
(`FloatV.litF32` of the double's `float64` pattern). -/
def c32 (bits : Nat) : F32 := FloatV.litF32 (FloatV.ofPat bits)

/-- `_gelu_and_mul_sparse_kernel` (source lines 56–92). `x_ptr` and `output_ptr` are `*bf16` (the
host's `x` and its fresh `output`, of `x`'s dtype). `std_multiplier` is a Python `float` argument:
a runtime `float32` scalar, `FloatV.hostF32` of the host's double (`model`). `d` and `BLOCK_SIZE`
are `tl.constexpr`s, Lean parameters. `input_dtype = x_ptr.dtype.element_ty` (line 74) is the
compile-time dtype `bfloat16`: each `.to(input_dtype)` is a conversion to `bfloat16`. -/
def _gelu_and_mul_sparse_kernel (x_ptr output_ptr : Ptr .bf16) (d : Nat) (std_multiplier : F32)
    (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    -- line 64: axis 0 of the grid, an `int32` scalar, sign-extended to `int64`
    let row : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 65: an `int32` block
    let offsets : Tensor [BLOCK_SIZE] I32 := tl.arange 0 BLOCK_SIZE
    -- line 66: the constexpr `d` becomes an `int32` tensor (a comparison operand), the signed
    -- comparison
    let mask := offsets.ltS (IntV.ofNat d : I32)
    -- line 67: the Python `2` and the constexpr `d` built in `int64` (the dtype of `row`), the
    -- `int64` products
    let row_start : I64 := row * (IntV.ofNat 2 : I64) * (IntV.ofNat d : I64)
    -- line 69: `x_ptr + row_start` advances the pointer by the `int64` scalar, then by the `int32`
    -- block; `other=0.0` is the `float32` `+0`, converted to the pointer's dtype: `+0` in
    -- `bfloat16` (exact; the value `litBF16` of `0` gives); `.to(tl.float32)` widens exactly
    let gate : Tensor [BLOCK_SIZE] F32 ← tl.cvt (← tl.load (x_ptr + row_start + offsets) (some mask)
      (some (bc (FloatV.litBF16 (FloatV.ofPat 0)))))
    -- line 72: `((x_ptr + row_start) + d) + offsets`, left to right: the constexpr `d` added to a
    -- pointer is an `int32` constant
    let up : Tensor [BLOCK_SIZE] F32 ← tl.cvt (← tl.load
      (x_ptr + row_start + (IntV.ofNat d : I32) + offsets) (some mask)
      (some (bc (FloatV.litBF16 (FloatV.ofPat 0)))))
    -- line 75: `tl.sum(gate, axis=0)` of the rank-1 `float32` block, which no product feeds:
    -- IEEE additions along a tree (`tl.sum`, a scalar); `/ d`: the constexpr `d` built in the
    -- division's dtype `float32`, the platform's division
    let mean0 : F32 := (← tl.div (Tensor.scalar (← tl.sum gate))
      (Tensor.scalar (FloatV.litF32 (FloatV.ofRat (d : ℚ))))).item
    -- line 76: `gate - mean`: `mean` is a `float32` quotient, a product for the fusion rule, so
    -- the fusable `-` (`tl.subc`), the scalar broadcast; the lanes `offsets ≥ d` hold the
    -- constant `0.0`, built in `float32`
    let centered := tl.where mask (← tl.subc gate (bc mean0)) (bc (FloatV.litF32 (FloatV.ofPat 0)))
    -- line 77: the IEEE product, summed along a tree whose additions it may feed (`tl.sumc`);
    -- `/ d` as line 75
    let variance : F32 := (← tl.div (Tensor.scalar (← tl.sumc (← tl.mul centered centered)))
      (Tensor.scalar (FloatV.litF32 (FloatV.ofRat (d : ℚ))))).item
    -- line 79: `mean.to(bfloat16)`, one rounding to nearest even, then the exact widening
    let mean16 : Tensor [] BF16 ← tl.cvt (Tensor.scalar mean0)
    let mean : Tensor [] F32 ← tl.cvt mean16
    -- line 80: `tl.sqrt`, the platform's square root, then the two conversions
    let std16 : Tensor [] BF16 ← tl.cvt (← tl.sqrt (Tensor.scalar variance))
    let std : Tensor [] F32 ← tl.cvt std16
    -- line 81: the IEEE product with the runtime `float32` scalar (it feeds a conversion, not an
    -- addition), then the two conversions
    let scaled16 : Tensor [] BF16 ← tl.cvt (← tl.mul std (Tensor.scalar std_multiplier))
    let scaled_std : Tensor [] F32 ← tl.cvt scaled16
    -- line 82: the IEEE `+` (both operands are conversions, not products), then the two
    -- conversions
    let cutoff16 : Tensor [] BF16 ← tl.cvt (← tl.add mean scaled_std)
    let cutoff : Tensor [] F32 ← tl.cvt cutoff16
    -- line 83: the IEEE `-` (no product feeds it), the scalar broadcast, then the two conversions
    let sg16 : Tensor [BLOCK_SIZE] BF16 ← tl.cvt (← tl.sub gate (bc cutoff.item))
    let sg0 : Tensor [BLOCK_SIZE] F32 ← tl.cvt sg16
    -- line 84: `sparse_gate < 0.0`: the Python `0.0` a `float32` tensor, the ordered comparison;
    -- the value `0.0` built in `float32`
    let sparse_gate := tl.where (sg0.flt (bc (c32 0))) (bc (c32 0)) sg0
    -- line 87: `0.044715 * sparse_gate * sparse_gate * sparse_gate`, left to right: the constant
    -- built in `float32`, two IEEE products; then `sparse_gate + (… * sparse_gate)`, an addition
    -- the last product feeds: `tl.mul_add` (the IEEE sum is symmetric)
    let p1 ← tl.mul (bc (c32 0x3FA6E4E26D4801F7)) sparse_gate
    let p2 ← tl.mul p1 sparse_gate
    let z ← tl.mul_add p2 sparse_gate sparse_gate
    -- line 86: the constant `0.7978845608028654` built in `float32`, the IEEE product
    let inner ← tl.mul (bc (c32 0x3FE9884533D43651)) z
    -- line 89: `2.0 * inner` (the IEEE product); `tl.sigmoid`, Triton's `1 / (1 + exp(-x))`;
    -- `2.0 * …` (the IEEE product); `… - 1.0`, a subtraction the product feeds (`tl.subc`)
    let sig ← tl.sigmoid (← tl.mul (bc (c32 0x4000000000000000)) inner)
    let tanh ← tl.subc (← tl.mul (bc (c32 0x4000000000000000)) sig) (bc (c32 0x3FF0000000000000))
    -- line 90: `0.5 * sparse_gate` (the IEEE product); `1.0 + tanh`, the IEEE `+` (`tanh` is a
    -- subtraction's result, not a product); their IEEE product
    let act0 ← tl.mul (← tl.mul (bc (c32 0x3FE0000000000000)) sparse_gate)
      (← tl.add (bc (c32 0x3FF0000000000000)) tanh)
    -- line 91: the two conversions
    let act16 : Tensor [BLOCK_SIZE] BF16 ← tl.cvt act0
    let activated : Tensor [BLOCK_SIZE] F32 ← tl.cvt act16
    -- line 92: `row * d` in `int64`; the IEEE product `activated * up` (no addition consumes it),
    -- converted to the pointer's dtype `bfloat16` (one rounding to nearest even); the masked store
    let y16 : Tensor [BLOCK_SIZE] BF16 ← tl.cvt (← tl.mul activated up)
    tl.store (output_ptr + row * (IntV.ofNat d : I64) + offsets) y16 (some mask)

/-! ## The conditions -/

/-- What the host chooses: one launch of `_gelu_and_mul_sparse_kernel` through
`_gelu_and_mul_sparse_triton_op` (source lines 113–134) on a contiguous `bfloat16` `x` viewed as
`[n_rows, 2·d]` and its fresh `output`, viewed as `[n_rows, d]`. -/
structure Inst where
  /-- `n_rows` (`x.numel() // x.shape[-1]`, line 119): the grid. -/
  T : Nat
  /-- `d` (`x.shape[-1] // 2`, line 114): the row's `gate` (and `up`) length. -/
  d : Nat
  /-- `BLOCK_SIZE`: the host passes `triton.next_power_of_2(d)` (line 124). -/
  B : Nat
  /-- `std_multiplier`: the Python `float` (a double); the kernel gets `FloatV.hostF32 m`. -/
  m : F64
  /-- `x`, `bfloat16`: row `t`'s `gate` is elements `t·2d … t·2d + d − 1`, its `up` the next `d`. -/
  x : Buf .bf16
  /-- `output` (`torch.empty`, line 115), `bfloat16`: row `t` is elements `t·d … t·d + d − 1`. -/
  out : Buf .bf16

/-- Element `j` of row `t`'s `gate` in `x`. -/
def gidx (d : Inst) (t j : Nat) : Nat := t * (2 * d.d) + j

/-- Element `j` of row `t`'s `up` in `x`. -/
def uidx (d : Inst) (t j : Nat) : Nat := t * (2 * d.d) + d.d + j

/-- Element `j` of row `t` of `output`. -/
def oidx (d : Inst) (t j : Nat) : Nat := t * d.d + j

/-- The elements of `x` the launch reads. -/
def inSet (d : Inst) : Set Nat := {k | k < d.T * (2 * d.d)}

/-- The elements of `output` the launch writes. -/
def outSet (d : Inst) : Set Nat := {k | k < d.T * d.d}

/-- The instances covered: `d` from 1 to `32768`, `BLOCK_SIZE` its next power of two, from 1 to
`2^31 − 1` rows, a finite multiplier with `|m| ≤ 64`, buffers holding what the launch touches, and
the output sharing no byte with `x`. -/
structure Inst.ok (d : Inst) : Prop where
  /-- A row has at least one `gate` element (the host requires `x.shape[-1] > 0` and even). -/
  d_pos : 1 ≤ d.d
  /-- `d ≤ 32768`: the host's `_MAX_INTERMEDIATE_SIZE` (line 109). -/
  d_le : d.d ≤ 32768
  /-- `BLOCK_SIZE` is a power of two. -/
  b_pow2 : ∃ e, d.B = 2 ^ e
  /-- `d ≤ BLOCK_SIZE`. -/
  d_le_b : d.d ≤ d.B
  /-- `BLOCK_SIZE < 2·d`: with `b_pow2` and `d_le_b`, `BLOCK_SIZE = next_power_of_2(d)`. -/
  b_lt : d.B < 2 * d.d
  /-- At least one row (the host returns before the launch when `output` is empty, line 116). -/
  t_pos : 1 ≤ d.T
  /-- `n_rows < 2^31`: CUDA's limit on the grid's first axis. -/
  t_lt : d.T < 2 ^ 31
  /-- The multiplier is finite. -/
  m_fin : d.m.isFinite = true
  /-- `|m| ≤ 64` (README, Exclusions: `Normal(0, 1).icdf` of a `float32` sparsity in `(0, 1)` is
  within `±40`). -/
  m_le : |d.m.toRat| ≤ 64
  /-- `x` holds `n_rows·2d` elements. -/
  x_len : d.T * (2 * d.d) ≤ d.x.len
  /-- `output` holds `n_rows·d` elements. -/
  out_len : d.T * d.d ≤ d.out.len
  /-- The output shares no byte with `x` (`torch.empty` allocates it). -/
  out_x : Disjoint (d.out.elems (outSet d)) (d.x.elems (inSet d))

/-- The model on an instance: the buffers' first elements, the multiplier as the launcher's
`float32` (`FloatV.hostF32` of the Python double), the constexprs `d` and `BLOCK_SIZE`. -/
def model (d : Inst) : Kernel :=
  _gelu_and_mul_sparse_kernel (d.x.ptr 0) (d.out.ptr 0) d.d (FloatV.hostF32 d.m) d.B

/-- The host's launch (lines 121–133): the grid `(n_rows,)`. The source's block shape is
`[BLOCK_SIZE]` (the offsets, mask, pointers, loads and the elementwise arithmetic); the reductions'
results are scalars. `enable_fp_fusion` is Triton's default (`true`); `grid1` keeps
`num_ctas = 1`, `launch_pdl = False` and `debug = false`; `num_warps` is performance-only. -/
def launch (d : Inst) : Launch := Launch.grid1 d.T [[d.B]]

/-- The `bfloat16` element `i` of a buffer in memory `m` (`+0` where absent; `Pre` makes every
element the launch reads present). -/
def rval (m : Mem) (p : Buf .bf16) (i : Nat) : BF16 :=
  match m.read .bf16 (p.ptr (i : Int)).addr with
  | some v => v
  | none => FloatV.ofPat 0

/-- `gate[t, j]` and `up[t, j]` in the initial memory, as reals. -/
noncomputable def gv (d : Inst) (init : Mem) (t j : Nat) : ℝ := (rval init d.x (gidx d t j)).toReal
noncomputable def uv (d : Inst) (init : Mem) (t j : Nat) : ℝ := (rval init d.x (uidx d t j)).toReal

/-- The exact mean of row `t`'s gate. -/
noncomputable def meanRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (∑ j ∈ Finset.range d.d, gv d init t j) / d.d

/-- The exact mean magnitude of row `t`'s gate, `Σ_j |g j| / d`. -/
noncomputable def absMean (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  (∑ j ∈ Finset.range d.d, |gv d init t j|) / d.d

/-- The exact (biased) standard deviation of row `t`'s gate, `sqrt(Σ_j (g j − M)² / d)` (PyTorch's
`std(unbiased=False)`). -/
noncomputable def stdRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  Real.sqrt ((∑ j ∈ Finset.range d.d, (gv d init t j - meanRef d init t) ^ 2) / d.d)

/-- An element of `x` covered: a finite `bfloat16` of magnitude at most `2^32` (README,
Exclusions). -/
def InOk (a : BF16) : Prop := a.isFinite = true ∧ |a.toRat| ≤ 2 ^ 32

/-- Row `t`'s gate is not nearly constant: `2^-18·A + 2^-62 ≤ Sd` (README, Exclusions: on a nearly
constant row the kernel's variance may be subnormal, where the library's `tl.sqrt` fact says
nothing). Every row that is not constant and whose largest `|gate|` is at least `2^-44` meets
it. -/
def RowOk (d : Inst) (init : Mem) (t : Nat) : Prop :=
  2 ^ (-18 : ℤ) * absMean d init t + 2 ^ (-62 : ℤ) ≤ stdRef d init t

/-- The inputs covered: `x` and `output` are valid (aligned, inside `[0, 2^64)`, every byte
present), every element of `x` the launch reads is `InOk`, and every row is `RowOk`. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.out.Valid init ∧
    (∀ k, k < d.T * (2 * d.d) → InOk (rval init d.x k)) ∧
    ∀ t, t < d.T → RowOk d init t

/-! ## The specification

For row `t`, from the host's inputs (`g j = gate[t, j]`, `u j = up[t, j]`, `m` the host's double):
`M = meanRef`, `A = absMean`, `Sd = stdRef`, the exact cutoff `cRef = M + m·Sd`, the exact sparse
gate `sRef j = max (g j − cRef) 0`, and `yRef j = geluRef (sRef j) · u j`, with
`geluRef x = x/2 · (1 + tanh(0.7978845608028654·(x + 0.044715·x³)))` (the source's constants,
lines 86–87, as decimals). The tolerances (README, Tolerances):

* `cutTol = 2^-6·(|M| + |m|·Sd) + 2^-17·(1 + |m|)·A + 2^-130`, a bound on the kernel's cutoff
  (rounded to `bfloat16` at each step) against `cRef`;
* `yTol j = 2^-5·|yRef j| + 2·|u j|·cutTol + 2^-132`.
-/

/-- The host's multiplier as a real. -/
noncomputable def mR (d : Inst) : ℝ := d.m.toReal

/-- The exact cutoff `M + m·Sd`. -/
noncomputable def cutRef (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  meanRef d init t + mR d * stdRef d init t

/-- The tolerance of the cutoff. -/
noncomputable def cutTol (d : Inst) (init : Mem) (t : Nat) : ℝ :=
  2 ^ (-6 : ℤ) * (|meanRef d init t| + |mR d| * stdRef d init t) +
    2 ^ (-17 : ℤ) * (1 + |mR d|) * absMean d init t + 2 ^ (-130 : ℤ)

/-- The tanh-approximate GELU with the source's constants:
`x/2 · (1 + tanh(0.7978845608028654·(x + 0.044715·x³)))`. -/
noncomputable def geluRef (x : ℝ) : ℝ :=
  x / 2 * (1 + Real.tanh (0.7978845608028654 * (x + 0.044715 * x ^ 3)))

/-- The exact sparse gate `max (g j − cRef) 0`. -/
noncomputable def sRef (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  max (gv d init t j - cutRef d init t) 0

/-- The exact output `geluRef (sRef j) · u j`. -/
noncomputable def yRef (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  geluRef (sRef d init t j) * uv d init t j

/-- The tolerance of each output element. -/
noncomputable def yTol (d : Inst) (init : Mem) (t j : Nat) : ℝ :=
  2 ^ (-5 : ℤ) * |yRef d init t j| + 2 * |uv d init t j| * cutTol d init t + 2 ^ (-132 : ℤ)

/-- What the launch leaves: for every row `t < n_rows` and column `j < d`, `output[t, j]` is a
finite `bfloat16` within `yTol j` of `yRef j`, and a zero where `g j ≤ cRef − cutTol`; every byte
outside the elements of `output` the launch writes is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t, t < d.T → ∀ j, j < d.d → ∃ c : BF16,
      fin.read .bf16 (d.out.ptr (oidx d t j : Int)).addr = some c ∧ c.isFinite = true ∧
      |c.toReal - yRef d init t j| ≤ yTol d init t j ∧
      (gv d init t j ≤ cutRef d init t - cutTol d init t → c.toReal = 0)) ∧
    ∀ z, z ∉ d.out.elems (outSet d) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_gelu_and_mul_sparse_kernel` ends normally with
`Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The double `1.6448535919189453` (`0x3FFA515200000000`): Gemma 3n's multiplier, vLLM's
`Normal(0, 1).icdf(torch.tensor(0.95)).item()` (the `float32` quantile, as a Python `float`). -/
def mGemma : F64 := FloatV.ofPat 0x3FFA515200000000

theorem mGemma_facts : mGemma.isFinite = true ∧ |mGemma.toRat| ≤ 64 := by
  refine ⟨by decide +kernel, ?_⟩
  have : |(FloatV.ofPat 0x3FFA515200000000 : F64).toRat| ≤ 64 := by decide +kernel
  exact this

/-- The double `1.6448536269514722` (`0x3FFA515209676ABB`), the default multiplier of vLLM's input
generator for this operation (`vllm/ir/ops/activation.py:30`). -/
def mGen : F64 := FloatV.ofPat 0x3FFA515209676ABB

theorem mGen_facts : mGen.isFinite = true ∧ |mGen.toRat| ≤ 64 := by
  refine ⟨by decide +kernel, ?_⟩
  have : |(FloatV.ofPat 0x3FFA515209676ABB : F64).toRat| ≤ 64 := by decide +kernel
  exact this

/-- A launch of the host's: `T` rows of `d` columns, `BLOCK_SIZE = B`, the multiplier `m`; `x` at
`2^40` and `output` at `2^41` (each below `2^40` bytes long). -/
def mkInst (T d B : Nat) (m : F64) : Inst where
  T := T
  d := d
  B := B
  m := m
  x := ⟨2 ^ 40, T * (2 * d)⟩
  out := ⟨2 ^ 41, T * d⟩

theorem mkInst_ok {T d B : Nat} {m : F64} (hd : 1 ≤ d) (hd' : d ≤ 32768) (hb : ∃ k, B = 2 ^ k)
    (hdb : d ≤ B) (hb2 : B < 2 * d) (ht : 1 ≤ T) (ht' : T < 2 ^ 31) (htd : T * d ≤ 2 ^ 37)
    (hm : m.isFinite = true ∧ |m.toRat| ≤ 64) : (mkInst T d B m).ok := by
  refine ⟨hd, hd', hb, hdb, hb2, ht, ht', hm.1, hm.2, le_rfl, le_rfl, ?_⟩
  refine Buf.elems_disjoint_of _ _ ?_ (fun j hj => hj) (fun j hj => hj)
  refine Set.disjoint_left.mpr fun z hz hz' => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq, mkInst, DType.bytes] at hz hz'
  have h1 : ((T * d : Nat) : ℤ) ≤ 2 ^ 37 := by exact_mod_cast htd
  push_cast at h1 hz hz'
  nlinarith

/-- A memory in which every byte is present: `0`, but the byte at `a1`, which is `0x3F`. -/
def oneMem (a1 : Int) : Mem := fun a => some (if a = a1 then 0x3F else 0)

/-- Rows of one element pair: the instance `T = 1`, `d = 2`, `BLOCK_SIZE = 2`, whose `gate` is
`(0, 0.5)` in `oneMem` (`0x3F00` is `0.5` in `bfloat16`). -/
def satInst : Inst := mkInst 1 2 2 mGemma

theorem satInst_read (k : Nat) (hk : k < 4) :
    rval (oneMem (2 ^ 40 + 3)) satInst.x k =
      if k = 1 then (FloatV.ofPat 0x3F00 : BF16) else FloatV.ofPat 0 := by
  interval_cases k <;> decide +kernel

theorem satInst_gate (j : Nat) (hj : j < 2) :
    gv satInst (oneMem (2 ^ 40 + 3)) 0 j = if j = 1 then 1 / 2 else 0 := by
  have h : gidx satInst 0 j = j := by simp [gidx]
  unfold gv
  rw [h, satInst_read j (by omega)]
  interval_cases j
  · have : (FloatV.ofPat 0 : BF16).toRat = 0 := by decide +kernel
    simp [FloatV.toReal, this]
  · have : (FloatV.ofPat 0x3F00 : BF16).toRat = 1 / 2 := by decide +kernel
    simp [FloatV.toReal, this]

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨satInst, oneMem (2 ^ 40 + 3),
    mkInst_ok (by norm_num) (by norm_num) ⟨1, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) mGemma_facts, ?_⟩
  have hv : ∀ (b : Buf .bf16), 0 ≤ b.base → b.base + b.len * 2 ≤ 2 ^ 64 → b.base % 2 = 0 →
      b.Valid (oneMem (2 ^ 40 + 3)) :=
    fun b h1 h2 h3 => ⟨h1, h2, h3, fun _ _ => by simp [oneMem]⟩
  refine ⟨hv _ (by simp [satInst, mkInst]) (by simp [satInst, mkInst]) (by simp [satInst, mkInst]),
    hv _ (by simp [satInst, mkInst]) (by simp [satInst, mkInst]) (by simp [satInst, mkInst]),
    fun k hk => ?_, fun t ht => ?_⟩
  · change k < 1 * (2 * 2) at hk
    rw [satInst_read k (by omega)]
    split_ifs
    · exact ⟨by decide +kernel, by decide +kernel⟩
    · exact ⟨by decide +kernel, by decide +kernel⟩
  · change t < 1 at ht
    obtain rfl : t = 0 := by omega
    have hs : ∀ j ∈ Finset.range 2, gv satInst (oneMem (2 ^ 40 + 3)) 0 j =
        if j = 1 then 1 / 2 else 0 := fun j hj => satInst_gate j (Finset.mem_range.mp hj)
    have hdd : satInst.d = 2 := rfl
    have hd : (satInst.d : ℝ) = 2 := by rw [hdd]; norm_num
    have hM : meanRef satInst (oneMem (2 ^ 40 + 3)) 0 = 1 / 4 := by
      unfold meanRef
      rw [hdd, Finset.sum_congr rfl hs]
      simp; norm_num
    have hA : absMean satInst (oneMem (2 ^ 40 + 3)) 0 = 1 / 4 := by
      unfold absMean
      rw [hdd, Finset.sum_congr rfl (fun j hj => by rw [hs j hj])]
      simp [Finset.sum_range_succ]; norm_num
    have hS : stdRef satInst (oneMem (2 ^ 40 + 3)) 0 = 1 / 4 := by
      unfold stdRef
      rw [hM, hdd, Finset.sum_congr rfl (fun j hj => by rw [hs j hj])]
      simp only [Finset.sum_range_succ, Finset.sum_range_zero]
      norm_num
    unfold RowOk
    rw [hA, hS]
    norm_num

/-! ## Coverage

`SETUP.md`'s launches: Gemma 3n's sparse MLP layers (the first ten layers of each model,
`activation_sparsity = 0.95`) in `bfloat16` with the model's multiplier `mGemma`: E4B
(`intermediate_size = 16384`) and E2B (`8192`), each row local to a tensor-parallel shard, so
`d = 16384 / tp` or `8192 / tp` for `tp` in 1, 2, 4, 8; one decoding token and a prefill chunk of
8192 tokens. The source's test launches (`(tokens, d)` = `(1, 8192)`, `(7, 16384)`, `(33, 4096)`,
`(5, 100)`), and vLLM's input generator's multiplier `mGen` with the shapes of vLLM's tests of this
operation (`tests/kernels/ir/test_activation.py`: 1 to 64 tokens, `d` from 31 to 32768). -/

def setup_e4b_tp1_decode : Inst := mkInst 1 16384 16384 mGemma
def setup_e4b_tp1_prefill : Inst := mkInst 8192 16384 16384 mGemma
def setup_e4b_tp2_prefill : Inst := mkInst 8192 8192 8192 mGemma
def setup_e4b_tp4_prefill : Inst := mkInst 8192 4096 4096 mGemma
def setup_e4b_tp8_prefill : Inst := mkInst 8192 2048 2048 mGemma
def setup_e2b_tp1_decode : Inst := mkInst 1 8192 8192 mGemma
def setup_e2b_tp1_prefill : Inst := mkInst 8192 8192 8192 mGemma
def setup_e2b_tp8_prefill : Inst := mkInst 8192 1024 1024 mGemma
def setup_test_1x8192 : Inst := mkInst 1 8192 8192 mGemma
def setup_test_7x16384 : Inst := mkInst 7 16384 16384 mGemma
def setup_test_33x4096 : Inst := mkInst 33 4096 4096 mGemma
def setup_test_5x100 : Inst := mkInst 5 100 128 mGemma
def setup_gen_8192x16384 : Inst := mkInst 8192 16384 16384 mGen
def setup_vtest_64x13824 : Inst := mkInst 64 13824 16384 mGen
def setup_vtest_64x32768 : Inst := mkInst 64 32768 32768 mGen
def setup_vtest_32x8192 : Inst := mkInst 32 8192 8192 mGen
def setup_vtest_5x31 : Inst := mkInst 5 31 32 mGen
def setup_vtest_5x128 : Inst := mkInst 5 128 128 mGen

theorem setup_e4b_tp1_decode_ok : setup_e4b_tp1_decode.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨14, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_e4b_tp1_prefill_ok : setup_e4b_tp1_prefill.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨14, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_e4b_tp2_prefill_ok : setup_e4b_tp2_prefill.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨13, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_e4b_tp4_prefill_ok : setup_e4b_tp4_prefill.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨12, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_e4b_tp8_prefill_ok : setup_e4b_tp8_prefill.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨11, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_e2b_tp1_decode_ok : setup_e2b_tp1_decode.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨13, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_e2b_tp1_prefill_ok : setup_e2b_tp1_prefill.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨13, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_e2b_tp8_prefill_ok : setup_e2b_tp8_prefill.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨10, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_test_1x8192_ok : setup_test_1x8192.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨13, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_test_7x16384_ok : setup_test_7x16384.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨14, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_test_33x4096_ok : setup_test_33x4096.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨12, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_test_5x100_ok : setup_test_5x100.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGemma_facts
theorem setup_gen_8192x16384_ok : setup_gen_8192x16384.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨14, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGen_facts
theorem setup_vtest_64x13824_ok : setup_vtest_64x13824.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨14, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGen_facts
theorem setup_vtest_64x32768_ok : setup_vtest_64x32768.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨15, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGen_facts
theorem setup_vtest_32x8192_ok : setup_vtest_32x8192.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨13, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGen_facts
theorem setup_vtest_5x31_ok : setup_vtest_5x31.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨5, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGen_facts
theorem setup_vtest_5x128_ok : setup_vtest_5x128.ok :=
  mkInst_ok (by norm_num) (by norm_num) ⟨7, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) mGen_facts

end Claims.GeluAndMulSparse
