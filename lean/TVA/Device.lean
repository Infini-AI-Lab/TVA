import TVA.Device.Weaken
import TVA.Device.Ref.Sm90

/-!
# The device facts: `Device P`

Some Triton operations have no exact meaning: the compiler lowers them to instructions whose
results the hardware documents only up to an accuracy, or not at all (`exp`, the default `/`,
`tl.dot`, conversions into fp8, float `atomic_add`, `tl.dot_scaled`, the tree of a float
reduction). TVA leaves them to a platform `P` (`TVA.Platform`), a relation per operation, and
every claim quantifies over every `P` that satisfies `Device P`: the conjunction of all the
facts TVA states about the device, an NVIDIA H200 (sm_90) running code compiled by Triton 3.8.0.

**What a fact says.** Each fact is one shape of `TVA.Device.Shapes` (accuracy on a domain, an
exact value, totality) at one operation, format and configuration, with its constants written in
(`TVA.Device.Sm90`, `TVA.Device.Dot8Chunk`, `TVA.Device.Dot16Chunk`). Its docstring names the
instructions Triton emits and the bound.

**Why they are trusted.** A fact is not proved: it is a statement about hardware, which Lean
cannot see. It is part of what a reader of a claim must trust (`TVA.Claim`), with Lean, Mathlib,
the claim's transcription of the kernel and the compiler. Each fact is kept as narrow as the
claims need, and each is stated with slack over what was observed (typically a factor of 2 on
each constant), so that it is not an exact model of one instruction sequence but a bound that
the instructions Triton may choose all meet.

**How they were measured.** Each fact cites a folder of `evidence/` (`evidence/num-math-f32`,
`evidence/num-dot`, …). It holds the script that ran the operation through Triton 3.8.0 on an
NVIDIA H200 NVL, its output, and a README saying what it shows. Where the input space is small
the measurement is exhaustive (every `float32` input of `exp`, every `float16` input of a
conversion); for `tl.dot` it combines a bit-level model of the tensor-core instructions, checked
against the device on every element tested, with adversarial and random inputs at every size
that compiles; and the bound is derived from the model, then checked against the device.

**Why `Device.ref_ok` matters.** Every claim assumes `Device P`. If the facts contradicted each
other, no platform would satisfy them, and every claim would hold vacuously, whatever its kernel
does. `Device.ref_ok` proves that one platform, `Platform.ref` (`TVA.Device.Ref.Platform`: IEEE
arithmetic rounded once, as accurate as a float result can be), satisfies every fact at once.
So the facts are consistent, and a claim's proof cannot draw anything from them that a real
platform as accurate as `Platform.ref` would not give.

A fact implied by others is a theorem here rather than a field: the round-to-nearest
conversions in range (`Device.cvtF32E4m3`, …, from the saturating ones) and the first arm of the
fp8 chunk facts (`Device.dotE4m3Chunk`, …). Weaker forms of any fact follow by
`TVA.Device.Weaken`.
-/
namespace TVA

open Device

/-- **The device facts**: every fact TVA states about the platform `P` of an H200 (sm_90) running
Triton 3.8.0's code. Every claim assumes it (`TVA.Claim`). Each field is one fact; its docstring
in `TVA.Device.Sm90`, `TVA.Device.Dot8Chunk` or `TVA.Device.Dot16Chunk` says what the device
does, the bound, and the evidence. -/
structure Device (P : Platform) : Prop where
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

namespace Device

variable {P : Platform}

/-! ## Facts implied by others -/

/-- `float32 → float8e4nv` rounds to nearest even in range (`CvtF32E4m3`). -/
theorem cvtF32E4m3 (h : Device P) : CvtF32E4m3 P := h.cvtSatF32E4m3.cvtRn Format.e4m3_regular

/-- `float32 → float8e5` rounds to nearest even in range (`CvtF32E5m2`). -/
theorem cvtF32E5m2 (h : Device P) : CvtF32E5m2 P := h.cvtSatF32E5m2.cvtRn Format.e5m2_regular

/-- `float16 → float8e4nv` rounds to nearest even in range (`CvtF16E4m3`). -/
theorem cvtF16E4m3 (h : Device P) : CvtF16E4m3 P := h.cvtSatF16E4m3.cvtRn Format.e4m3_regular

/-- `float16 → float8e5` rounds to nearest even in range (`CvtF16E5m2`). -/
theorem cvtF16E5m2 (h : Device P) : CvtF16E5m2 P := h.cvtSatF16E5m2.cvtRn Format.e5m2_regular

/-- `bfloat16 → float8e4nv` rounds to nearest even in range (`CvtBF16E4m3`). -/
theorem cvtBF16E4m3 (h : Device P) : CvtBF16E4m3 P :=
  h.cvtSatBF16E4m3.cvtRn Format.e4m3_regular

/-- `bfloat16 → float8e5` rounds to nearest even in range (`CvtBF16E5m2`). -/
theorem cvtBF16E5m2 (h : Device P) : CvtBF16E5m2 P :=
  h.cvtSatBF16E5m2.cvtRn Format.e5m2_regular

/-- `tl.dot` of `float8e4nv`, chunk by chunk, first arm (`DotE4m3Chunk`). -/
theorem dotE4m3Chunk (h : Device P) : DotE4m3Chunk P := DotF8ChunkLoss.chunk h.dotE4m3ChunkLoss

/-- `tl.dot` of `float8e5`, chunk by chunk, first arm (`DotE5m2Chunk`). -/
theorem dotE5m2Chunk (h : Device P) : DotE5m2Chunk P := DotF8ChunkLoss.chunk h.dotE5m2ChunkLoss

/-- `tl.dot` of `float8e4nv` by `float8e5`, chunk by chunk, first arm (`DotE4m3E5m2Chunk`). -/
theorem dotE4m3E5m2Chunk (h : Device P) : DotE4m3E5m2Chunk P :=
  DotF8ChunkLoss.chunk h.dotE4m3E5m2ChunkLoss

/-- `tl.dot` of `float8e5` by `float8e4nv`, chunk by chunk, first arm (`DotE5m2E4m3Chunk`). -/
theorem dotE5m2E4m3Chunk (h : Device P) : DotE5m2E4m3Chunk P :=
  DotF8ChunkLoss.chunk h.dotE5m2E4m3ChunkLoss

/-! ## The facts are consistent -/

/-- **The device facts are jointly satisfiable**: the reference platform satisfies every one, so
no claim holds vacuously through its assumption `Device P`. -/
theorem ref_ok : Device Platform.ref where
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

end Device

end TVA
