import TVA.Device.Weaken
import TVA.Device.Ref.Sm90

/-!
# The L40's device facts (sm_89, Triton 3.8.0)

`Facts P`: every fact TVA states about the platform `P` of an NVIDIA L40 (Ada, sm_89) running code
compiled by Triton 3.8.0, with the same fields as the H200's (`TVA.Device.H200`). Every evidence
script of `evidence/` was run on an L40 (`evidence/l40/README.md` lists each run, its output and
what changed), with the same Triton 3.8.0 and torch 2.11.0+cu130.

**What is the same.** Most facts are the H200's statements unchanged, because the L40 meets them
with the slack the library keeps (about a factor of 2 on each constant): the approximate
`float32` functions and their special values (every input; the outputs equal the H200's bit for
bit), `div.full.f32` and the infinite divisor, `float64` division (`div.rn.f64`), the conversions
into fp8 from `float16` and `bfloat16`, `tl.dot` of 16-bit operands as a whole (`DotF16`, ratios
`0.25`), `float32` with `input_precision="ieee"` (bit for bit the sequential `fma`), `int8`, the
fp8 chunk facts (ratios up to `0.52`), `tl.dot_scaled` (the same `bfloat16` emulation), float
`atomic_add` and the depth of a reduction's tree.

The 16-bit chunk facts (`DotF16Chunk`, `DotBF16Chunk`) also keep the H200's statement, with less
slack: the L40's `mma.sync…m16n8k16` keeps one bit less of the accumulator (`evidence/num-dot` part
1 on the L40: `s ≤ 24`, not `25`) and is not the H200's bit-level model (`evidence/num-dot16-model`
part A), and the largest ratio to the bound over the H200's adversarial inputs is `0.80`, not `0.50`
(accumulator-dominated inputs, where the arm `2·A_j + 2^-22·Mj` is the smaller one). No element was
outside (34.3 million in single chunks, 19.0 million in chains, every shape that fits in the L40's
shared memory), and that arm holds of any window that truncates: a product below the window loses at
most its own magnitude. Evidence: `evidence/l40/num-dot-chunk16`,
`evidence/l40/num-dot-chunk16-order` (the chunks of 64 in increasing order, as on the H200).

**What differs** (the statements below):

* **`float32 → fp8` is two roundings.** Triton 3.8.0 lowers it on sm_89 to `cvt.rz.f16.f32`
  (toward zero, into `float16`) and then `cvt.rn.satfinite.e4m3x2.f16x2` (`e5m2x2`); on sm_90 it
  is one `cvt.rn.satfinite.e4m3x2.f32`. The result is `cvtSat .rn` of the `float16` truncation,
  which differs from one rounding on 1032066 (`float8e4nv`) and 1146756 (`float8e5`) of the
  `float32` inputs in range: `x` just above `2^-10` gives `0`, not the smallest subnormal
  `2^-9`. Evidence: `evidence/l40/num-cvt8` (every `float32` input against the composition,
  0 differences; the library's own `FloatV.cvtSat .rn (FloatV.cvtWith .rz x)` against the device
  on 1409006 inputs, 0 differences). So the H200's `CvtSatF32E4m3`, `CvtSatF32E5m2` and the
  round-to-nearest facts derived from them (`CvtF32E4m3`, `CvtF32E5m2`) are false here.

The fp8 `tl.dot` uses the native `mma.sync…m16n8k32…e4m3/e5m2` on sm_89 (sm_90's `mma.sync`
converts to `float16`), with a 13-bit window like sm_90's `wgmma`; its chunks of 32 are added in
increasing order, but one instruction is not one window (`evidence/l40/num-dot-chunk8-order`). The
fp8 chunk facts are kept: the measured ratios leave the factor 2.

`ref_ok`: one platform satisfies every fact: `refPlatform`, the reference platform with the L40's
conversion from `float32` into fp8. One rounding there (`Platform.ref`) does not satisfy it.
-/
namespace TVA.Device.L40

open Device

variable (P : Platform)

/-! ## The L40's own statements -/

/-- `float32 → fp8` on the L40 (`x.to(g)` with the default rounding): toward zero into `float16`
(`cvt.rz.f16.f32`; beyond `65504` the largest finite `float16`), then the saturating rounding to
nearest even (`cvt.rn.satfinite…f16x2`): the result is
`FloatV.cvtSat .rn (FloatV.cvtWith (g := .binary16) .rz x)`, up to NaN payload. -/
def CvtSatF32Two (g : Format) : Prop :=
  ∀ x : F32, ∀ r : FloatV g, P.cvt .binary32 g .rn x r →
    FloatV.NaNEq (FloatV.cvtSat .rn (FloatV.cvtWith (g := .binary16) .rz x)) r

/-- **`float32 → float8e4nv`** on the L40: two roundings (`CvtSatF32Two`). Evidence:
`evidence/l40/num-cvt8`. -/
def CvtSatF32E4m3 : Prop := CvtSatF32Two P .e4m3

/-- **`float32 → float8e5`** on the L40: two roundings (`CvtSatF32Two`). Evidence:
`evidence/l40/num-cvt8`. -/
def CvtSatF32E5m2 : Prop := CvtSatF32Two P .e5m2

/-! ## The facts -/

/-- **The L40's device facts**: every fact TVA states about the platform `P` of an L40 (sm_89)
running Triton 3.8.0's code, with the H200's field names. The fields whose statements are the
L40's own are `cvtSatF32E4m3` and `cvtSatF32E5m2`; the others are
the shared statements of `TVA.Device.Sm90`, `TVA.Device.Dot8Chunk`, `TVA.Device.Dot16Chunk`. -/
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
  /-- `float32 → float8e4nv`: two roundings, toward zero to `float16`, then saturating to
  nearest even (`L40.CvtSatF32E4m3`). -/
  cvtSatF32E4m3 : L40.CvtSatF32E4m3 P
  /-- `float32 → float8e5`: two roundings (`L40.CvtSatF32E5m2`). -/
  cvtSatF32E5m2 : L40.CvtSatF32E5m2 P
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

/-! ## A platform satisfying them -/

/-- The conversions the L40 does in two roundings: `float32` into fp8, to nearest. -/
def TwoRoundings (f g : Format) (rnd : RoundMode) : Prop :=
  f = .binary32 ∧ (g = .e4m3 ∨ g = .e5m2) ∧ rnd = .rn

/-- The reference platform with the L40's conversion from `float32` into fp8: one rounding
everywhere (`Platform.ref`), except the two roundings of `CvtSatF32Two`. -/
noncomputable def refPlatform : Platform :=
  { Platform.ref with
    cvt := fun f g rnd x r =>
      (TwoRoundings f g rnd →
        FloatV.NaNEq (FloatV.cvtSat .rn (FloatV.cvtWith (g := .binary16) .rz x)) r) ∧
      (¬ TwoRoundings f g rnd → Platform.ref.cvt f g rnd x r) }

/-- Every conversion of `refPlatform` has a result. -/
theorem cvtTotal (f g : Format) (rnd : RoundMode) : CvtTotal refPlatform f g rnd := fun x => by
  by_cases hc : TwoRoundings f g rnd
  · exact ⟨_, fun _ => Or.inl rfl, fun h => absurd hc h⟩
  · obtain ⟨r, hr⟩ := Platform.ref.cvtTotal f g rnd x
    exact ⟨r, fun h => absurd h hc, fun _ => hr⟩

/-- A conversion from a format other than `float32` is `Platform.ref`'s on `refPlatform`. -/
theorem oneRounding {f g : Format} {rnd : RoundMode} {x : FloatV f} {r : FloatV g}
    (hf : f ≠ .binary32) (h : refPlatform.cvt f g rnd x r) : Platform.ref.cvt f g rnd x r :=
  h.2 fun h' => hf h'.1

/-- **The L40's facts are jointly satisfiable**: `refPlatform` satisfies every one. -/
theorem ref_ok : Facts refPlatform where
  mathTotal := Platform.ref.mathTotal
  divTotal := Platform.ref.divTotal
  cvtTotal := cvtTotal
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
  cvtSatF32E4m3 := fun _ _ h => h.1 ⟨rfl, by decide, rfl⟩
  cvtSatF32E5m2 := fun _ _ h => h.1 ⟨rfl, by decide, rfl⟩
  cvtSatF16E4m3 := fun x r h => Platform.ref.cvtSatF16E4m3 x r (oneRounding (by decide) h)
  cvtSatF16E5m2 := fun x r h => Platform.ref.cvtSatF16E5m2 x r (oneRounding (by decide) h)
  cvtSatBF16E4m3 := fun x r h => Platform.ref.cvtSatBF16E4m3 x r (oneRounding (by decide) h)
  cvtSatBF16E5m2 := fun x r h => Platform.ref.cvtSatBF16E5m2 x r (oneRounding (by decide) h)
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

end TVA.Device.L40
