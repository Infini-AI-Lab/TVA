import TVA

/-!
# Prepared claim: `gdn_gating_bf16`

Source: `source/fused_gdn_gating.py`, SGLang v0.5.23's
`fused_gdn_gating_kernel` (lines 53-84; it reads the module-level names `triton` and `tl`, bound by
the imports of lines 47-48), launched by its host `fused_gdn_gating` (lines 87-119) on the grid
`(batch, 1, cdiv(num_heads, 8))` with `seq_len = 1`, `BLK_HEADS = 8`, `num_warps=1`,
`num_stages=3` and the wrapper's defaults `beta = 1.0`, `threshold = 20.0` (compile-time values).

For every token `i < batch` and head `h < num_heads` the kernel stores
* `g = -exp(A_log[h]) * softplus(a[i, h] + dt_bias[h])`, `softplus(x) = log(1 + exp(x))` for
  `x <= 20` and `x` above it, in `float32`;
* `beta_output = sigmoid(b[i, h])`, computed in `float32`, rounded to `bfloat16`, and stored into
  the `float32` output.

The file holds the model (`fused_gdn_gating_kernel`, `model`, `launch`), the conditions (`Inst`,
`Inst.ok`, `Pre`), the specification (`Post`, `Spec`), the assumptions (`Assumptions`) and the
claim `Statement`, to be proved. It proves only `consistent`, `satisfiable` and the coverage of the
setup's launches (`setup_*_ok`). The README gives the derivation of the tolerances.
-/

namespace Claims.GdnGating

open TVA

/-! ## The model -/

/-- The `float32` constant Triton builds for the Python number `1.0` (and the integer `1`): its
`float64` pattern `0x3FF0000000000000`, in the dtype of the operation (`float32`). It is `beta`,
`1 / beta` (the compile-time `1.0`), and the `1` of `1 + tl.exp(...)` and of `tl.sigmoid`'s
`1 / (1 + math.exp(-x))`. -/
def oneF32 : F32 := FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)

/-- The `float32` constant Triton builds for the Python number `20.0`, the default `threshold`
(`float64` pattern `0x4034000000000000`). -/
def twentyF32 : F32 := FloatV.litF32 (FloatV.ofPat 0x4034000000000000)

/-- `fused_gdn_gating_kernel` (source lines 53-84). The pointers `g`, `beta_output`, `A_log` are
`*fp32`; `a`, `b`, `dt_bias` are `*bf16` (the host's tensors, SETUP.md). `seq_len`, `stride_a`,
`stride_b` are integer arguments, `int32` at every use (an argument equal to `1`, as `seq_len`
always is, is the compile-time constant `1`, which each product first makes an `int32` tensor, so
both classes give this model). `NUM_HEADS` and `BLK_HEADS` are `tl.constexpr`s, Lean parameters.
`beta = 1.0` and `threshold = 20.0` are compile-time values (the wrapper's defaults, which every
caller uses): the model writes the constants Triton builds for them, `oneF32` and `twentyF32`. -/
def fused_gdn_gating_kernel (g beta_output A_log : Ptr .f32) (a b dt_bias : Ptr .bf16)
    (seq_len stride_a stride_b : I32) (NUM_HEADS BLK_HEADS : Nat) : Kernel where
  body ctx := do
    -- line 69
    let i_b := tl.program_id ctx 0
    let i_s := tl.program_id ctx 1
    let i_d := tl.program_id ctx 2
    -- line 70: the constexpr `BLK_HEADS` is built in `int32` (the tensor's dtype); wrapping `int32`
    let head_off := i_d * (IntV.ofNat BLK_HEADS : I32) + tl.arange 0 BLK_HEADS
    -- line 71: `int32` products and sums (wrapping); `NUM_HEADS` built in `int32`
    let off := i_b * seq_len * (IntV.ofNat NUM_HEADS : I32) + i_s * (IntV.ofNat NUM_HEADS : I32) +
      head_off
    -- line 72: the `int32` tensor against the constant `NUM_HEADS` in `int32`, signed
    let mask := head_off.ltS (IntV.ofNat NUM_HEADS : I32)
    -- lines 73-76: masked loads without `other`: the disabled lanes are unspecified
    let blk_A_log ← tl.load (A_log + head_off) (some mask)
    let blk_a ← tl.load (a + i_b * stride_a + head_off) (some mask)
    let blk_b ← tl.load (b + i_b * stride_b + head_off) (some mask)
    let blk_bias ← tl.load (dt_bias + head_off) (some mask)
    -- line 77: the exact widenings `bfloat16 -> float32`, then the IEEE `+` (no product feeds it)
    let xa ← tl.cvt (g := .binary32) blk_a
    let xb ← tl.cvt (g := .binary32) blk_bias
    let x ← tl.add xa xb
    -- line 79, the condition (evaluated first): `beta * x`, the IEEE product with the constant
    -- `1.0` built in `float32`; `<= threshold`, the constant `20.0` a `float32` tensor, the
    -- ordered comparison
    let t1 ← tl.mul (bc oneF32) x
    let cond := t1.fle (bc twentyF32)
    -- line 79, the value: the source writes `beta * x` a second time; `tl.exp` of it; `1 + ...`,
    -- the IEEE `+` (its operand is `tl.exp`'s result, not a product); `tl.log`; `(1 / beta) * ...`,
    -- the compile-time `1.0` built in `float32`, the IEEE product (no addition consumes it)
    let t2 ← tl.mul (bc oneF32) x
    let e ← tl.exp t2
    let s ← tl.add (bc oneF32) e
    let l ← tl.log s
    let l2 ← tl.mul (bc oneF32) l
    -- lines 78-80: `tl.where` of two `float32` values on the `int1` condition
    let softplus_x := tl.where cond l2 x
    -- line 81: `blk_A_log.to(tl.float32)` is no conversion (`A_log` is `float32`); `tl.exp`; the
    -- unary `-`; the IEEE product (no addition consumes it)
    let eA ← tl.exp blk_A_log
    let neA ← tl.neg eA
    let blk_g ← tl.mul neA softplus_x
    -- line 82: `blk_g.to(g.dtype.element_ty)` is no conversion (`float32` to `float32`)
    tl.store (g + off) blk_g (some mask)
    -- line 83: the exact widening, then `tl.sigmoid`'s body `1 / (1 + math.exp(-x))`, inlined
    -- (SEMANTICS.md, `tl.sigmoid`): `-x`, `tl.exp`, the IEEE `+` of the constant `1`, and the
    -- default `float32` division of the constant `1` by the sum
    let blk_b32 ← tl.cvt (g := .binary32) blk_b
    let nb ← tl.neg blk_b32
    let eb ← tl.exp nb
    let den ← tl.add (bc oneF32) eb
    let blk_beta_output ← tl.div (bc oneF32) den
    -- line 84: `.to(b.dtype.element_ty)` rounds to `bfloat16` (to nearest even); the store into the
    -- `float32` pointer converts it back (exact)
    let beta16 ← tl.cvt (g := .bfloat16) blk_beta_output
    let beta32 ← tl.cvt (g := .binary32) beta16
    tl.store (beta_output + off) beta32 (some mask)

/-! ## The conditions -/

/-- What the host chooses: the instance of one launch. `seq_len = 1`, `BLK_HEADS = 8`,
`beta = 1.0` and `threshold = 20.0` are the wrapper's own values, fixed in `model`. -/
structure Inst where
  /-- `batch` (`a.shape[0]`), the grid's first axis. -/
  batch : Nat
  /-- `num_heads` (`a.shape[1]`), the constexpr `NUM_HEADS`. -/
  H : Nat
  /-- `stride_a` (`a.stride(0)`), in elements. -/
  stride_a : Nat
  /-- `stride_b` (`b.stride(0)`), in elements. -/
  stride_b : Nat
  /-- `A_log`: `float32`, `[num_heads]`. -/
  A_log : Buf .f32
  /-- `a`: `bfloat16`, `[batch, num_heads]` at row stride `stride_a`, unit stride along heads. -/
  a : Buf .bf16
  /-- `b`: `bfloat16`, `[batch, num_heads]` at row stride `stride_b`. -/
  b : Buf .bf16
  /-- `dt_bias`: `bfloat16`, `[num_heads]`. -/
  dt_bias : Buf .bf16
  /-- `g`: `float32`, `[1, batch, num_heads]`, contiguous (the output). -/
  g : Buf .f32
  /-- `beta_output`: `float32`, `[1, batch, num_heads]`, contiguous (the output). -/
  bout : Buf .f32

/-- The number of head blocks, `triton.cdiv(num_heads, 8)`, the grid's third axis. -/
def Bn (d : Inst) : Nat := (d.H + 7) / 8

/-- The instances covered: every launch of the wrapper whose grid CUDA accepts and whose `int32`
offsets do not wrap (each such condition is a limit of the kernel, README, Exclusions), with buffers
holding every element the launch reads or writes, and outputs sharing no byte with the inputs or
with each other. -/
structure Inst.ok (d : Inst) : Prop where
  /-- The grid's first axis `batch` is within CUDA's limit (`2^31 - 1`). -/
  batch_lt : d.batch < 2 ^ 31
  /-- The grid's third axis `cdiv(num_heads, 8)` is within CUDA's limit (`65535`). -/
  grid_z : Bn d < 2 ^ 16
  /-- The `int32` output offset `i_b * num_heads + h` of every written element is below `2^31`
  (the last is `batch * num_heads - 1`). -/
  flat_le : d.batch * d.H ≤ 2 ^ 31
  /-- `stride_a` is an `int32` argument (below `2^31`). -/
  stride_a_lt : d.stride_a < 2 ^ 31
  /-- `stride_b` is an `int32` argument. -/
  stride_b_lt : d.stride_b < 2 ^ 31
  /-- The `int32` offset `i_b * stride_a + h` of every element of `a` read is below `2^31`. -/
  arow_le : (d.batch - 1) * d.stride_a + d.H ≤ 2 ^ 31
  /-- The `int32` offset `i_b * stride_b + h` of every element of `b` read is below `2^31`. -/
  brow_le : (d.batch - 1) * d.stride_b + d.H ≤ 2 ^ 31
  /-- `A_log` holds `num_heads` elements. -/
  A_len : d.H ≤ d.A_log.len
  /-- `dt_bias` holds `num_heads` elements. -/
  bias_len : d.H ≤ d.dt_bias.len
  /-- Row `i` of `a` is in the buffer (contiguous rows, or a view split from one
  `[batch, 2 * num_heads]` projection: any row stride). -/
  a_len : ∀ i, i < d.batch → i * d.stride_a + d.H ≤ d.a.len
  /-- Row `i` of `b` is in the buffer. -/
  b_len : ∀ i, i < d.batch → i * d.stride_b + d.H ≤ d.b.len
  /-- `g` holds `batch * num_heads` elements. -/
  g_len : d.batch * d.H ≤ d.g.len
  /-- `beta_output` holds `batch * num_heads` elements. -/
  bout_len : d.batch * d.H ≤ d.bout.len
  /-- `g` shares no byte with the inputs or with `beta_output` (the wrapper allocates it). -/
  g_disj : Disjoint d.g.bytes
    (d.A_log.bytes ∪ d.a.bytes ∪ d.b.bytes ∪ d.dt_bias.bytes ∪ d.bout.bytes)
  /-- `beta_output` shares no byte with the inputs (the wrapper allocates it). -/
  bout_disj : Disjoint d.bout.bytes (d.A_log.bytes ∪ d.a.bytes ∪ d.b.bytes ∪ d.dt_bias.bytes)
  /-- `batch` is the number of tokens of the extend batch: a context length, at most `2^22`. -/
  batch_ctx : d.batch ≤ 2 ^ 22

/-- The model on an instance: the buffers' first elements, `seq_len = 1`, the strides as `int32`,
`NUM_HEADS = num_heads`, `BLK_HEADS = 8`. -/
def model (d : Inst) : Kernel :=
  fused_gdn_gating_kernel (d.g.ptr 0) (d.bout.ptr 0) (d.A_log.ptr 0) (d.a.ptr 0) (d.b.ptr 0)
    (d.dt_bias.ptr 0) (IntV.ofNat 1) (IntV.ofNat d.stride_a) (IntV.ofNat d.stride_b) d.H 8

/-- The host's launch (source lines 99, 102-118): the grid `(batch, 1, cdiv(num_heads, 8))`; the
source's only block shape is `[BLK_HEADS] = [8]`; `num_warps=1` and `num_stages=3` are
performance-only; `enable_fp_fusion` is Triton's default (`true`); `num_ctas = 1`, no `launch_pdl`,
`debug = false`. -/
def launch (d : Inst) : Launch :=
  { grid := ![d.batch, 1, Bn d], fusion := true, blocks := [[8]] }

/-- `tl.exp` of `t` has room in `float32` by the library's card `Num.ExpF32` taken at `X = |t|`:
`(1 + 2^-22 + 2^-23·|t|)·e^t + 2^-140 ≤ maxVal`. It holds for every `t ≤ 88.72282` and fails from
`88.72283` (`ln maxVal ≈ 88.722839`, where `exp` itself overflows). -/
def ExpRoom (t : ℝ) : Prop :=
  (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t|)) * Real.exp t + 2 ^ (-140 : ℤ) ≤
    (TVA.Math.Float.Format.binary32.maxVal : ℝ)

/-- The input values covered: every element read is finite (SETUP.md: any finite values), and
* `exp(A_log[h])` does not overflow (`ExpRoom`: `A_log[h] ≤ 88.72282`);
* the `float32` sum `a[i, h] + dt_bias[h]` does not overflow to `+∞` (its exact value is below
  `2^128 - 2^103`, from which it rounds to `+∞`); it may round to `-∞`;
* `exp(-b[i, h])` does not overflow (`ExpRoom (-b)`: `b[i, h] ≥ -88.72282`).
Each is a limit of the kernel or of the library (README, Exclusions). -/
def InOk (d : Inst) (m : Mem) : Prop :=
  (∀ h, h < d.H → ∃ A : F32, m.read .f32 (d.A_log.ptr h).addr = some A ∧
      A.isFinite = true ∧ ExpRoom A.toReal) ∧
  (∀ i, i < d.batch → ∀ h, h < d.H → ∃ av bv : BF16,
      m.read .bf16 (d.a.ptr (i * d.stride_a + h)).addr = some av ∧
      m.read .bf16 (d.dt_bias.ptr h).addr = some bv ∧
      av.isFinite = true ∧ bv.isFinite = true ∧
      av.toReal + bv.toReal < 2 ^ 128 - 2 ^ 103) ∧
  (∀ i, i < d.batch → ∀ h, h < d.H → ∃ bv : BF16,
      m.read .bf16 (d.b.ptr (i * d.stride_b + h)).addr = some bv ∧
      bv.isFinite = true ∧ ExpRoom (-bv.toReal))

/-- The inputs covered: the six buffers are valid (aligned, inside `[0, 2^64)`, every byte present)
and the values are as `InOk` says. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.A_log.Valid init ∧ d.a.Valid init ∧ d.b.Valid init ∧ d.dt_bias.Valid init ∧
    d.g.Valid init ∧ d.bout.Valid init ∧ InOk d init

/-! ## The specification -/

/-- The exact softplus with the kernel's constants (`beta = 1`, `threshold = 20`). -/
noncomputable def spRef (x : ℝ) : ℝ := if x ≤ 20 then Real.log (1 + Real.exp x) else x

/-- The exact decay `-e^A · softplus(x)`. -/
noncomputable def gRef (A x : ℝ) : ℝ := -(Real.exp A * spRef x)

/-- The exact write strength `sigmoid(b) = 1 / (1 + e^(-b))`. -/
noncomputable def sigRef (b : ℝ) : ℝ := 1 / (1 + Real.exp (-b))

/-- The scale of `g`'s tolerance: the magnitudes of its two factors, `e^A` (with a floor `2^-121`
for `tl.exp`'s absolute error `2^-140`) times `softplus(x)` (with a floor `2^-6` for the rounding of
`1 + exp(x)`, an absolute error of `2^-24` in `softplus`). -/
noncomputable def gMag (A x : ℝ) : ℝ := (Real.exp A + 2 ^ (-121 : ℤ)) * (spRef x + 2 ^ (-6 : ℤ))

/-- The relative tolerance of `g`: `2^-23·(64 + |A|)`, `|A|` capped at `128` (`tl.exp(A)`'s card at
`X = |A|` grows with `|A|`; below `A = -128` every term is far below the floors). -/
noncomputable def gEps (A : ℝ) : ℝ := 2 ^ (-23 : ℤ) * (64 + min |A| 128)

/-- The tolerance of `g` at `A = A_log[h]` and `x = a[i, h] + dt_bias[h]` (exact):
`gEps A · gMag A x` (README, Tolerances). -/
noncomputable def gTol (A x : ℝ) : ℝ := gEps A * gMag A x

/-- The tolerance of `beta_output`: twice the `bfloat16` rounding (`2^-8` relative) and `2^-133`
(the `bfloat16` grid below its normal range is `2^-133`). -/
noncomputable def bTol (b : ℝ) : ℝ := 2 ^ (-7 : ℤ) * |sigRef b| + 2 ^ (-133 : ℤ)

/-- What the launch leaves. For each token `i < batch` and head `h < num_heads`, with `A`, `av`,
`bv`, `bb` the initial `A_log[h]`, `a[i, h]`, `dt_bias[h]`, `b[i, h]`:
* element `i * num_heads + h` of `g` is a finite `float32` within `gTol` of `gRef A (av + bv)`, or
  `-∞` where the exact value is at the edge of `float32`'s range (`maxVal ≤ |gRef| + gTol`: the
  product overflows);
* element `i * num_heads + h` of `beta_output` is a finite `float32` within `bTol` of `sigRef bb`;
* every byte outside the two outputs' first `batch * num_heads` elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ i, i < d.batch → ∀ h, h < d.H →
      ∃ A : F32, init.read .f32 (d.A_log.ptr h).addr = some A ∧
      ∃ av : BF16, init.read .bf16 (d.a.ptr (i * d.stride_a + h)).addr = some av ∧
      ∃ bv : BF16, init.read .bf16 (d.dt_bias.ptr h).addr = some bv ∧
      ∃ c : F32, fin.read .f32 (d.g.ptr (i * d.H + h)).addr = some c ∧
        ((c.isFinite = true ∧
            |c.toReal - gRef A.toReal (av.toReal + bv.toReal)| ≤
              gTol A.toReal (av.toReal + bv.toReal)) ∨
          (c = FloatV.negInf ∧
            (TVA.Math.Float.Format.binary32.maxVal : ℝ) ≤
              |gRef A.toReal (av.toReal + bv.toReal)| + gTol A.toReal (av.toReal + bv.toReal)))) ∧
    (∀ i, i < d.batch → ∀ h, h < d.H →
      ∃ bb : BF16, init.read .bf16 (d.b.ptr (i * d.stride_b + h)).addr = some bb ∧
      ∃ c : F32, fin.read .f32 (d.bout.ptr (i * d.H + h)).addr = some c ∧
        c.isFinite = true ∧ |c.toReal - sigRef bb.toReal| ≤ bTol bb.toReal) ∧
    ∀ z, z ∉ d.g.elems {j | j < d.batch * d.H} ∪ d.bout.elems {j | j < d.batch * d.H} →
      fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from which
  the model's launch is well behaved, is one of the outcomes of the model's launch. The rule covers
  a launch within its scope: part (A), the kernel and its launches, which the host's launches meet
  (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.MathTotal`: `tl.exp` on `float32` has a result for every input (also on the
  masked lanes, and where `tl.where` discards it). -/
  exp_total : Rules.Num.MathTotal P .exp .binary32
  /-- [rule] `Num.ExpF32`: the accuracy of `tl.exp` on `float32`. -/
  exp_acc : Rules.Num.ExpF32 P
  /-- [rule] `Num.ExpF32NegInf`: `tl.exp(-∞) = +0` on `float32` (the softplus of a sum
  `a + dt_bias` that rounds to `-∞`). -/
  exp_ninf : Rules.Num.ExpF32NegInf P
  /-- [rule] `Num.MathTotal`: `tl.log` on `float32` has a result for every input. -/
  log_total : Rules.Num.MathTotal P .log .binary32
  /-- [rule] `Num.LogF32`: the accuracy of `tl.log` on `float32`. -/
  log_acc : Rules.Num.LogF32 P
  /-- [rule] `Num.DivTotal`: `/` on `float32` has a result for every pair of inputs. -/
  div_total : Rules.Num.DivTotal P .binary32
  /-- [rule] `Num.DivFull`: `/` on `float32` is `div.full.f32`. -/
  div_full : Rules.Num.DivFull P

/-- The claim to prove. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.ref.mathTotal,
    Platform.sm90ref.expF32, Platform.sm90ref.expF32NegInf, Platform.ref.mathTotal,
    Platform.sm90ref.logF32, Platform.ref.divTotal _, Platform.ref.divFull⟩⟩

/-! ## Instances

`mk batch H split`: the host's tensors at fixed, 16-byte-aligned, disjoint addresses. Contiguous
(`split = false`): `a` and `b` of `batch * num_heads` elements, row stride `num_heads`. Split
(`split = true`): `b` and `a` the two halves of one `[batch, 2 * num_heads]` projection
(`mixed_ba.split([num_heads, num_heads], dim=-1)`, source line 150), row stride `2 * num_heads`,
`a` starting `num_heads` elements after `b`. -/

def mk (batch H : Nat) (split : Bool) : Inst where
  batch := batch
  H := H
  stride_a := if split then 2 * H else H
  stride_b := if split then 2 * H else H
  A_log := ⟨2 ^ 30, H⟩
  dt_bias := ⟨2 ^ 30 + 2 ^ 20, H⟩
  b := if split then ⟨2 ^ 31, (batch - 1) * (2 * H) + H⟩ else ⟨2 ^ 31, batch * H⟩
  a := if split then ⟨2 ^ 31 + H * 2, (batch - 1) * (2 * H) + H⟩ else ⟨2 ^ 32, batch * H⟩
  g := ⟨2 ^ 33, batch * H⟩
  bout := ⟨2 ^ 34, batch * H⟩

theorem bytes_disj {d₁ d₂ : DType} (x : Buf d₁) (y : Buf d₂)
    (h : x.base + x.len * d₁.bytes ≤ y.base ∨ y.base + y.len * d₂.bytes ≤ x.base) :
    Disjoint x.bytes y.bytes := by
  rw [Set.disjoint_left]
  intro z hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem rows_fit {batch s H len : Nat} (h : (batch - 1) * s + H ≤ len) :
    ∀ i, i < batch → i * s + H ≤ len := by
  intro i hi
  have : i * s ≤ (batch - 1) * s := Nat.mul_le_mul_right _ (by omega)
  omega

theorem mk_ok (batch H : Nat) (split : Bool) (hB : 1 ≤ batch) (hb : batch ≤ 2 ^ 22)
    (hH : H ≤ 2 ^ 16) (hBH : batch * H ≤ 2 ^ 26) : (mk batch H split).ok := by
  have hBH2 : (batch - 1) * (2 * H) + H ≤ 2 * (batch * H) := by
    have : (batch - 1) * (2 * H) + 2 * H = 2 * (batch * H) := by
      rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel hB]; ring
    omega
  have hBH1 : (batch - 1) * H + H ≤ batch * H := by
    have : (batch - 1) * H + H = batch * H := by
      rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel hB]
    omega
  have bf : DType.bytes DType.bf16 = 2 := rfl
  have b4 : DType.bytes DType.f32 = 4 := rfl
  cases split <;>
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, rows_fit ?_, rows_fit ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
  simp only [mk, Bn, Bool.false_eq_true, ↓reduceIte] <;>
  try omega
  all_goals
    simp only [Set.disjoint_union_right]
    and_intros
  all_goals (apply bytes_disj; simp only [bf, b4]; omega)

/-- A memory in which every byte is present (and `0`). -/
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
  have hlv : ∀ n : Nat, Math.Bytes.leValue (List.replicate n 0) = 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih => simp only [List.replicate, Math.Bytes.leValue, ih]; norm_num
  rw [hlv]

theorem valid_zeroMem {d : DType} (b : Buf d) (h0 : 0 ≤ b.base)
    (h1 : b.base + b.len * d.bytes ≤ 2 ^ 64) (h2 : b.base % d.bytes = 0) : b.Valid zeroMem :=
  ⟨h0, h1, h2, fun _ _ => by simp [zeroMem]⟩

theorem maxVal_ge : (2 : ℝ) ≤ (TVA.Math.Float.Format.binary32.maxVal : ℝ) := by
  have : (2 : ℚ) ≤ TVA.Math.Float.Format.binary32.maxVal := by decide +kernel
  exact_mod_cast this

theorem expRoom_zero : ExpRoom 0 := by
  unfold ExpRoom
  rw [Real.exp_zero, abs_zero]
  refine le_trans ?_ maxVal_ge
  norm_num

theorem zero_f32 : (DType.ofBits .f32 0 : F32).isFinite = true ∧
    (DType.ofBits .f32 0 : F32).toReal = 0 := by
  refine ⟨by decide, ?_⟩
  have : (DType.ofBits .f32 0 : F32).toRat = 0 := by decide +kernel
  simp [FloatV.toReal, this]

theorem zero_bf16 : (DType.ofBits .bf16 0 : BF16).isFinite = true ∧
    (DType.ofBits .bf16 0 : BF16).toReal = 0 := by
  refine ⟨by decide, ?_⟩
  have : (DType.ofBits .bf16 0 : BF16).toRat = 0 := by decide +kernel
  simp [FloatV.toReal, this]

/-- The source's first test case (`batch = 1`, `num_heads = 32`, contiguous) on the all-zero
memory: every input is `0`. -/
theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have bf : DType.bytes DType.bf16 = 2 := rfl
  have b4 : DType.bytes DType.f32 = 4 := rfl
  refine ⟨mk 1 32 false, zeroMem, mk_ok 1 32 false (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals try (apply valid_zeroMem <;> simp only [mk, bf, b4, Bool.false_eq_true, ↓reduceIte] <;>
    norm_num)
  · intro h hh
    simp only [mk] at hh
    refine ⟨_, zeroMem_read ⟨?_, ?_, ?_⟩, zero_f32.1, by rw [zero_f32.2]; exact expRoom_zero⟩ <;>
      simp only [mk, Buf.ptr_addr, b4] <;> omega
  · intro i hi h hh
    simp only [mk] at hi hh
    refine ⟨_, _, zeroMem_read ⟨?_, ?_, ?_⟩, zeroMem_read ⟨?_, ?_, ?_⟩, zero_bf16.1, zero_bf16.1,
      by rw [zero_bf16.2]; norm_num⟩ <;>
      simp only [mk, Buf.ptr_addr, bf, Bool.false_eq_true, ↓reduceIte] <;> push_cast <;> omega
  · intro i hi h hh
    simp only [mk] at hi hh
    refine ⟨_, zeroMem_read ⟨?_, ?_, ?_⟩, zero_bf16.1,
      by rw [zero_bf16.2, neg_zero]; exact expRoom_zero⟩ <;>
      simp only [mk, Buf.ptr_addr, bf, Bool.false_eq_true, ↓reduceIte] <;> push_cast <;> omega

/-! ## The setup's launches

The host's test cases (source lines 134-144: `(batch, num_heads, split)`), then SETUP.md's
configurations: Qwen3-Next's 32 value heads at TP 1, 2, 4 and 8 (32, 16, 8, 4 heads per rank) and
Qwen3.5-27B's 48 and 24, each at a prefill chunk of 32768 tokens and at 65536 tokens, contiguous and
split. -/

def setup_b1_h32 : Inst := mk 1 32 false
theorem setup_b1_h32_ok : setup_b1_h32.ok := mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_b1_h48_split : Inst := mk 1 48 true
theorem setup_b1_h48_split_ok : setup_b1_h48_split.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_b4_h8_split : Inst := mk 4 8 true
theorem setup_b4_h8_split_ok : setup_b4_h8_split.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_b16_h48_split : Inst := mk 16 48 true
theorem setup_b16_h48_split_ok : setup_b16_h48_split.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_b32_h24_split : Inst := mk 32 24 true
theorem setup_b32_h24_split_ok : setup_b32_h24_split.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_b7_h4 : Inst := mk 7 4 false
theorem setup_b7_h4_ok : setup_b7_h4.ok := mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_b513_h16 : Inst := mk 513 16 false
theorem setup_b513_h16_ok : setup_b513_h16.ok := mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_b8192_h32 : Inst := mk 8192 32 false
theorem setup_b8192_h32_ok : setup_b8192_h32.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_b32768_h32_split : Inst := mk 32768 32 true
theorem setup_b32768_h32_split_ok : setup_b32768_h32_split.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def setup_qwen3next_tp1 : Inst := mk 65536 32 true
theorem setup_qwen3next_tp1_ok : setup_qwen3next_tp1.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_qwen3next_tp2 : Inst := mk 65536 16 false
theorem setup_qwen3next_tp2_ok : setup_qwen3next_tp2.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_qwen3next_tp4 : Inst := mk 32768 8 true
theorem setup_qwen3next_tp4_ok : setup_qwen3next_tp4.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_qwen3next_tp8 : Inst := mk 65536 4 true
theorem setup_qwen3next_tp8_ok : setup_qwen3next_tp8.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_qwen35_tp1 : Inst := mk 65536 48 true
theorem setup_qwen35_tp1_ok : setup_qwen35_tp1.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
def setup_qwen35_tp2 : Inst := mk 32768 24 false
theorem setup_qwen35_tp2_ok : setup_qwen35_tp2.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Claims.GdnGating
