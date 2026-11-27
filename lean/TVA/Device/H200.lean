import TVA.Device.Weaken
import TVA.Device.Ref.Sm90

/-!
# The H200's device facts (sm_90, Triton 3.8.0)

`Facts P`: every fact TVA states about the platform `P` of an NVIDIA H200 (sm_90) running code
compiled by Triton 3.8.0, one field per fact. Each field's statement is defined in
`TVA.Device.Sm90`, `TVA.Device.Dot8Chunk` or `TVA.Device.Dot16Chunk`, whose docstrings say what
the device does, the bound, and the folder of `evidence/` where it was measured on an H200 NVL.

`ref_ok`: the reference platform (`Platform.ref`) satisfies every fact, so the facts are
consistent (`TVA.Device.Ref.Sm90`).

`TVA.Device` (the selector) makes these the facts every claim assumes when the H200 is the target.
-/
namespace TVA.Device.H200

open Device

/-- **The H200's device facts**: every fact TVA states about the platform `P` of an H200 (sm_90)
running Triton 3.8.0's code. Each field is one fact; its docstring in `TVA.Device.Sm90`,
`TVA.Device.Dot8Chunk` or `TVA.Device.Dot16Chunk` says what the device does, the bound, and the
evidence. -/
structure Facts (P : Platform) : Prop where
  /-- Every approximate function has a result on every input of every format. -/
  mathTotal : ∀ fn f, Device.MathTotal P fn f
  /-- The default division has a result for every pair of operands. -/
  divTotal : ∀ f, Device.DivTotal P f
  /-- Every conversion has a result. -/
  cvtTotal : ∀ f g rnd, Device.CvtTotal P f g rnd
  /-- `tl.dot` has a result for all operands. A configuration Triton rejects never reaches the
  platform: the model's `tl.dot` checks it first. -/
  dotTotal : ∀ cfg, Device.DotTotal P cfg
  /-- A float `tl.atomic_add` has a result for every old value and operand. -/
  atomicAddTotal : ∀ f, Device.AtomicAddTotal P f
  /-- `tl.dot_scaled` has a result for all operands. -/
  dotScaledTotal : ∀ cfg, Device.DotScaledTotal P cfg
  /-- `tl.exp` on `float32` (`Device.ExpF32`). -/
  expF32 : Device.ExpF32 P
  /-- `tl.math.exp2` on `float32` (`Device.Exp2F32`). -/
  exp2F32 : Device.Exp2F32 P
  /-- `tl.log` on `float32` (`Device.LogF32`). -/
  logF32 : Device.LogF32 P
  /-- `tl.math.log2` on `float32` (`Device.Log2F32`). -/
  log2F32 : Device.Log2F32 P
  /-- `tl.sqrt` on `float32` (`Device.SqrtF32`). -/
  sqrtF32 : Device.SqrtF32 P
  /-- `tl.math.rsqrt` on `float32` (`Device.RsqrtF32`). -/
  rsqrtF32 : Device.RsqrtF32 P
  /-- `tl.exp(-∞) = +0` on `float32` (`Device.ExpF32NegInf`). -/
  expF32NegInf : Device.ExpF32NegInf P
  /-- `tl.math.exp2(-∞) = +0` on `float32` (`Device.Exp2F32NegInf`). -/
  exp2F32NegInf : Device.Exp2F32NegInf P
  /-- `tl.exp` overflows to `+∞` on `float32` (`Device.ExpF32Overflow`). -/
  expF32Overflow : Device.ExpF32Overflow P
  /-- `/` on `float32` is `div.full.f32` (`Device.DivFull`). -/
  divFull : Device.DivFull P
  /-- `x / ±∞ = ±0` on `float32` (`Device.DivF32InfDivisor`). -/
  divF32InfDivisor : Device.DivF32InfDivisor P
  /-- `/` on `float64` is IEEE division (`Device.DivF64`). -/
  divF64 : Device.DivF64 P
  /-- `float32 → float8e4nv` saturates (`Device.CvtSatF32E4m3`). -/
  cvtSatF32E4m3 : Device.CvtSatF32E4m3 P
  /-- `float32 → float8e5` saturates (`Device.CvtSatF32E5m2`). -/
  cvtSatF32E5m2 : Device.CvtSatF32E5m2 P
  /-- `float16 → float8e4nv` saturates (`Device.CvtSatF16E4m3`). -/
  cvtSatF16E4m3 : Device.CvtSatF16E4m3 P
  /-- `float16 → float8e5` saturates (`Device.CvtSatF16E5m2`). -/
  cvtSatF16E5m2 : Device.CvtSatF16E5m2 P
  /-- `bfloat16 → float8e4nv` saturates (`Device.CvtSatBF16E4m3`). -/
  cvtSatBF16E4m3 : Device.CvtSatBF16E4m3 P
  /-- `bfloat16 → float8e5` saturates (`Device.CvtSatBF16E5m2`). -/
  cvtSatBF16E5m2 : Device.CvtSatBF16E5m2 P
  /-- `tl.dot` of `float16` into `float32`, relative to lifted magnitudes (`Device.DotF16`). -/
  dotF16 : Device.DotF16 P
  /-- `tl.dot` of `bfloat16` into `float32`, relative to lifted magnitudes (`Device.DotBF16`). -/
  dotBF16 : Device.DotBF16 P
  /-- `tl.dot` of `float32` with `input_precision="ieee"` (`Device.DotF32Ieee`). -/
  dotF32Ieee : Device.DotF32Ieee P
  /-- `tl.dot` of `int8` into `int32` is exact in range (`Device.DotI8`). -/
  dotI8 : Device.DotI8 P
  /-- `tl.dot` of `float16`, chunk by chunk (`Device.DotF16Chunk`). -/
  dotF16Chunk : Device.DotF16Chunk P
  /-- `tl.dot` of `bfloat16`, chunk by chunk (`Device.DotBF16Chunk`). -/
  dotBF16Chunk : Device.DotBF16Chunk P
  /-- `tl.dot` of `float8e4nv`, chunk by chunk (`Device.DotE4m3ChunkLoss`). -/
  dotE4m3ChunkLoss : Device.DotE4m3ChunkLoss P
  /-- `tl.dot` of `float8e5`, chunk by chunk (`Device.DotE5m2ChunkLoss`). -/
  dotE5m2ChunkLoss : Device.DotE5m2ChunkLoss P
  /-- `tl.dot` of `float8e4nv` by `float8e5`, chunk by chunk (`Device.DotE4m3E5m2ChunkLoss`). -/
  dotE4m3E5m2ChunkLoss : Device.DotE4m3E5m2ChunkLoss P
  /-- `tl.dot` of `float8e5` by `float8e4nv`, chunk by chunk (`Device.DotE5m2E4m3ChunkLoss`). -/
  dotE5m2E4m3ChunkLoss : Device.DotE5m2E4m3ChunkLoss P
  /-- `tl.dot_scaled` of MXFP8 operands (`Device.DotScaledE4m3`). -/
  dotScaledE4m3 : Device.DotScaledE4m3 P
  /-- `tl.dot_scaled` of MXFP8 by MXFP4 (`Device.DotScaledE4m3E2m1`). -/
  dotScaledE4m3E2m1 : Device.DotScaledE4m3E2m1 P
  /-- `tl.atomic_add` on `float32` (`Device.AtomicAddF32`). -/
  atomicAddF32 : Device.AtomicAddF32 P
  /-- The trees of a float reduction are balanced (`Device.SumDepth`). -/
  sumDepth : Device.SumDepth P

/-- The H200's reference platform: `Platform.ref` (IEEE arithmetic rounded once). -/
noncomputable abbrev refPlatform : Platform := Platform.ref

/-- **The H200's facts are jointly satisfiable**: the reference platform satisfies every one. -/
theorem ref_ok : Facts Platform.ref where
  mathTotal := Platform.ref.mathTotal
  divTotal := Platform.ref.divTotal
  cvtTotal := Platform.ref.cvtTotal
  dotTotal := Platform.ref.dotTotal
  atomicAddTotal := Platform.ref.atomicAddTotal
  dotScaledTotal := Platform.ref.dotScaledTotal
  expF32 := Platform.ref.expF32
  exp2F32 := Platform.ref.exp2F32
  logF32 := Platform.ref.logF32
  log2F32 := Platform.ref.log2F32
  sqrtF32 := Platform.ref.sqrtF32
  rsqrtF32 := Platform.ref.rsqrtF32
  expF32NegInf := Platform.ref.expF32NegInf
  exp2F32NegInf := Platform.ref.exp2F32NegInf
  expF32Overflow := Platform.ref.expF32Overflow
  divFull := Platform.ref.divFull
  divF32InfDivisor := Platform.ref.divF32InfDivisor
  divF64 := Platform.ref.divF64
  cvtSatF32E4m3 := Platform.ref.cvtSatF32E4m3
  cvtSatF32E5m2 := Platform.ref.cvtSatF32E5m2
  cvtSatF16E4m3 := Platform.ref.cvtSatF16E4m3
  cvtSatF16E5m2 := Platform.ref.cvtSatF16E5m2
  cvtSatBF16E4m3 := Platform.ref.cvtSatBF16E4m3
  cvtSatBF16E5m2 := Platform.ref.cvtSatBF16E5m2
  dotF16 := Platform.ref.dotF16
  dotBF16 := Platform.ref.dotBF16
  dotF32Ieee := Platform.ref.dotF32Ieee
  dotI8 := Platform.ref.dotI8
  dotF16Chunk := Platform.ref.dotF16Chunk
  dotBF16Chunk := Platform.ref.dotBF16Chunk
  dotE4m3ChunkLoss := Platform.ref.dotE4m3ChunkLoss
  dotE5m2ChunkLoss := Platform.ref.dotE5m2ChunkLoss
  dotE4m3E5m2ChunkLoss := Platform.ref.dotE4m3E5m2ChunkLoss
  dotE5m2E4m3ChunkLoss := Platform.ref.dotE5m2E4m3ChunkLoss
  dotScaledE4m3 := Platform.ref.dotScaledE4m3
  dotScaledE4m3E2m1 := Platform.ref.dotScaledE4m3E2m1
  atomicAddF32 := Platform.ref.atomicAddF32
  sumDepth := Platform.ref.sumDepth

/-! ## Facts implied by others -/

variable {P : Platform}

/-- `float32 → float8e4nv` rounds to nearest even in range (`CvtF32E4m3`, from the saturating
`cvtSatF32E4m3`; false on the L40, which rounds twice). -/
theorem Facts.cvtF32E4m3 (h : Facts P) : CvtF32E4m3 P := h.cvtSatF32E4m3.cvtRn Format.e4m3_regular

/-- `float32 → float8e5` rounds to nearest even in range (`CvtF32E5m2`; as `cvtF32E4m3`). -/
theorem Facts.cvtF32E5m2 (h : Facts P) : CvtF32E5m2 P := h.cvtSatF32E5m2.cvtRn Format.e5m2_regular

end TVA.Device.H200
