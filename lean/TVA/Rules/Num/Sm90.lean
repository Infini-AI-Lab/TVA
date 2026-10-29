import TVA.Rules.Num
import TVA.Lemmas.FloatMax

/-!
# The device facts the library states (sm_90, Triton 3.8.0)

Page: `docs/rules/num.md` ("Facts the library states"). Each constant here is one fact: an
instance of a `Rules.Num` definition, for one operation, format and configuration, with its
constants written into the definition. A kernel assumes a fact by naming it, as a `[rule]` field
(`exp2 : Rules.Num.Exp2F32 P`), and never restates its constants. The lowering behind each fact
and TVA's evidence (`evidence/num-*`, Triton 3.8.0 on an NVIDIA H200 NVL) are on the page.

Every fact here holds of `Platform.sm90ref` (`TVA.Lemmas.Sm90`), so a bundle that cites any of
them proves `Claim.Consistent` with that platform.
-/
namespace TVA.Rules.Num

open TVA.Math.Float

variable (P : Platform)

/-! ## Approximate functions on `float32` -/

/-- **`tl.exp` on `float32`.** Triton computes `x · log2 e` with `mul.f32` (the constant rounded
to `float32`), then `ex2.approx.f32` without `.ftz`, so subnormal results are kept
(`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:634-658`). Rounding the product moves the
exponent by up to about `2^-24·|x|`, so the relative error grows with `|x|`: for every `X ≥ 0` and
finite `x` in `[-X, X]` (with room in `float32`), the result is finite and within
`(2^-22 + 2^-23·X)·e^x + 2^-140` of `e^x`. Evidence: `evidence/num-math-f32` (every `float32`
input). -/
def ExpF32 : Prop :=
  ∀ X : ℝ, 0 ≤ X →
    MathAcc P .exp .binary32 (Set.Icc (-X) X) (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * X) (2 ^ (-140 : ℤ))

/-- **`tl.math.exp2` on `float32`.** Triton calls libdevice's `__nv_exp2f`, which with Triton's
default flush-to-zero setting is `ex2.approx.ftz.f32`: results below `2^-126` are flushed to `0`.
Every finite input (with room in `float32`) gives a finite result within `2^-22·2^x + 2^-126` of
`2^x`. Evidence: `evidence/num-math-f32` (every `float32` input). -/
def Exp2F32 : Prop := MathAcc P .exp2 .binary32 Set.univ (2 ^ (-22 : ℤ)) (2 ^ (-126 : ℤ))

/-- **`tl.log` on `float32`** (libdevice's `__nv_logf`): for finite `x > 0`, within
`2^-22·|ln x| + 2^-149` of `ln x`. Evidence: `evidence/num-math-f32` (every `float32` input). -/
def LogF32 : Prop := MathAcc P .log .binary32 Set.univ (2 ^ (-22 : ℤ)) (2 ^ (-149 : ℤ))

/-- **`tl.math.log2` on `float32`** (libdevice's `__nv_log2f`, a polynomial with `fma.rn.ftz`
that scales subnormal inputs first): for finite `x > 0`, within `2^-22·|log₂ x| + 2^-149` of
`log₂ x`. Evidence: `evidence/num-math-f32` (every `float32` input). -/
def Log2F32 : Prop := MathAcc P .log2 .binary32 Set.univ (2 ^ (-22 : ℤ)) (2 ^ (-149 : ℤ))

/-- The inputs on which `tl.sqrt` is accurate: `0` and the normal positive floats. -/
def sqrtDom : Set ℝ := {x | x = 0 ∨ 2 ^ (-126 : ℤ) ≤ x}

/-- **`tl.sqrt` on `float32`** (`sqrt.approx.ftz.f32`): for `x = 0` or a normal `x > 0`, within
`2^-22·√x + 2^-149` of `√x`. A subnormal input is flushed to `0` and gives `0`, so subnormal
inputs are outside the domain. Evidence: `evidence/num-math-f32` (every `float32` input). -/
def SqrtF32 : Prop := MathAcc P .sqrt .binary32 sqrtDom (2 ^ (-22 : ℤ)) (2 ^ (-149 : ℤ))

/-- **`tl.math.rsqrt` on `float32`** (`rsqrt.approx.ftz.f32`): for a normal `x > 0`, within
`2^-22/√x + 2^-149` of `1/√x`. A subnormal input is flushed and gives `+∞`. Evidence:
`evidence/num-math-f32` (every `float32` input). -/
def RsqrtF32 : Prop := MathAcc P .rsqrt .binary32 (Set.Ici (2 ^ (-126 : ℤ))) (2 ^ (-22 : ℤ)) (2 ^ (-149 : ℤ))

/-- `-∞` in `float32` (pattern `0xff800000`). -/
def negInfF32 : F32 := FloatV.ofPat 0xff800000

/-- `+0` in `float32`. -/
def zeroF32 : F32 := FloatV.ofPat 0

/-- `negInfF32` is the library's `−∞` of `float32` (`FloatV.negInf`, `TVA.Lemmas.FloatMax`), so
the facts at `negInfF32` combine directly with the lemmas about `FloatV.negInf`. -/
theorem negInfF32_eq_negInf : negInfF32 = FloatV.negInf := by decide

/-- **`tl.exp(-∞) = +0`** on `float32` (`ex2.approx.f32` of `-∞`). Softmax over masked lanes
loaded as `-inf` needs it. Evidence: `evidence/num-math-f32` (special values). -/
def ExpF32NegInf : Prop := MathAt P .exp .binary32 negInfF32 zeroF32

/-- **`tl.math.exp2(-∞) = +0`** on `float32` (`ex2.approx.ftz.f32` of `-∞`). The online softmax
needs it when the running maximum starts at `-inf`. Evidence: `evidence/num-math-f32`. -/
def Exp2F32NegInf : Prop := MathAt P .exp2 .binary32 negInfF32 zeroF32

/-! ## Conversions into fp8 -/

/-- **`float32 → float8e4nv`** (`x.to(tl.float8e4nv)`, `cvt.rn.satfinite.e4m3x2.f32`): a finite
value with `|x| ≤ 448` rounds to nearest even. Evidence: `evidence/num-cvt8` (every `float32`
input, bit for bit). -/
def CvtF32E4m3 : Prop := CvtRn P .binary32 .e4m3

/-- **`float32 → float8e5`** (`cvt.rn.satfinite.e5m2x2.f32`): a finite value with `|x| ≤ 57344`
rounds to nearest even. Evidence: `evidence/num-cvt8` (every `float32` input). -/
def CvtF32E5m2 : Prop := CvtRn P .binary32 .e5m2

/-- **`float16 → float8e4nv`** (`cvt.rn.satfinite.e4m3x2.f16x2`): in range, round to nearest
even. Evidence: `evidence/num-cvt8` (every `float16` input). -/
def CvtF16E4m3 : Prop := CvtRn P .binary16 .e4m3

/-- **`float16 → float8e5`** (`cvt.rn.satfinite.e5m2x2.f16x2`): in range, round to nearest even.
Evidence: `evidence/num-cvt8` (every `float16` input). -/
def CvtF16E5m2 : Prop := CvtRn P .binary16 .e5m2

/-- **`bfloat16 → float8e4nv`** (exact widening to `float32`, then
`cvt.rn.satfinite.e4m3x2.f32`): in range, round to nearest even. Evidence: `evidence/num-cvt8`
(every `bfloat16` input). -/
def CvtBF16E4m3 : Prop := CvtRn P .bfloat16 .e4m3

/-- **`bfloat16 → float8e5`** (exact widening, then `cvt.rn.satfinite.e5m2x2.f32`): in range,
round to nearest even. Evidence: `evidence/num-cvt8` (every `bfloat16` input). -/
def CvtBF16E5m2 : Prop := CvtRn P .bfloat16 .e5m2

/-! ## Conversions into fp8 beyond the range (saturation)

The same instructions on every input: `.satfinite` sends a finite value beyond `maxVal` (448,
57344) and `±∞` to `±maxVal`, and a NaN to the NaN `0x7f`. Each fact is `CvtSat`: the result is
`FloatV.cvtSat .rn x` up to NaN payload, and it implies the `CvtRn` fact of its pair
(`CvtSat.cvtRn`). Evidence: `evidence/float-layer` (part A: the library's `Format.cvt … .rn true`
against the device on every `float16`, `bfloat16` and `float32` input, bit for bit up to NaN
payload) and `evidence/num-cvt8` (every input: `run.py` the results above `maxVal`, `specials.py`
every `±∞` and NaN pattern of the three source formats). -/

/-- **`float32 → float8e4nv`, every input** (`cvt.rn.satfinite.e4m3x2.f32`): `CvtSat`. -/
def CvtSatF32E4m3 : Prop := CvtSat P .binary32 .e4m3

/-- **`float32 → float8e5`, every input** (`cvt.rn.satfinite.e5m2x2.f32`): `CvtSat`. -/
def CvtSatF32E5m2 : Prop := CvtSat P .binary32 .e5m2

/-- **`float16 → float8e4nv`, every input** (`cvt.rn.satfinite.e4m3x2.f16x2`): `CvtSat`. -/
def CvtSatF16E4m3 : Prop := CvtSat P .binary16 .e4m3

/-- **`float16 → float8e5`, every input** (`cvt.rn.satfinite.e5m2x2.f16x2`): `CvtSat`. -/
def CvtSatF16E5m2 : Prop := CvtSat P .binary16 .e5m2

/-- **`bfloat16 → float8e4nv`, every input** (exact widening to `float32`, then
`cvt.rn.satfinite.e4m3x2.f32`): `CvtSat`. -/
def CvtSatBF16E4m3 : Prop := CvtSat P .bfloat16 .e4m3

/-- **`bfloat16 → float8e5`, every input** (exact widening, then `cvt.rn.satfinite.e5m2x2.f32`):
`CvtSat`. -/
def CvtSatBF16E5m2 : Prop := CvtSat P .bfloat16 .e5m2

/-! ## `tl.dot` into `float32` -/

/-- The relative constant of a 16-bit `tl.dot` with `K` products: `(K + 2)·2^-22`. -/
def relDot16 (K : ℕ) : ℚ := (K + 2) * 2 ^ (-22 : ℤ)

/-- The absolute constant of a 16-bit `tl.dot`: `(K + 2)·2^-126`. -/
def absDot16 (K : ℕ) : ℚ := (K + 2) * 2 ^ (-126 : ℤ)

/-- **`tl.dot` of `float16` operands into `float32`** (`max_num_imprecise_acc = 0`, any
`input_precision`, which Triton ignores for these operands). Triton chains `mma.sync` or
`wgmma` instructions of `16` products each along `K`. Each instruction keeps every term's bits
down to at least `2^-23` of the largest term, the accumulator included (truncating below), so a
chunk of `j` nonzero terms is within `j·2^-23` times their magnitudes' sum; over the `⌈K/16⌉`
instructions this stays below the constants. Per output element:
within `(K+2)·2^-22·(|acc| + Σ|a·b|) + (K+2)·2^-126` of `acc + Σ a·b`, with room in `float32`.
Evidence: `evidence/num-dot` (the accumulation window, and adversarial inputs at every `K` that
compiles, on both instruction paths). -/
def DotF16 : Prop :=
  ∀ p : InputPrecision, DotAcc P ⟨.f16, .f16, .f32, p, 0⟩ (fun _ => True) relDot16 absDot16

/-- **`tl.dot` of `bfloat16` operands into `float32`**: as `DotF16`, with the same constants
(the absolute term covers products below `float32`'s normal range). Evidence: `evidence/num-dot`. -/
def DotBF16 : Prop :=
  ∀ p : InputPrecision, DotAcc P ⟨.bf16, .bf16, .f32, p, 0⟩ (fun _ => True) relDot16 absDot16

/-- The number of `32`-product instructions an fp8 `tl.dot` with `K` products takes:
`⌈K/32⌉`. -/
def chunks8 (K : ℕ) : ℕ := (K + 31) / 32

/-- The relative constant of an fp8 `tl.dot`: `(1 + 2^-7)^⌈K/32⌉ − 1`. -/
def relDot8 (K : ℕ) : ℚ := (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K - 1

/-- The absolute constant of an fp8 `tl.dot`: `⌈K/32⌉·(1 + 2^-7)^⌈K/32⌉·2^-126`. -/
def absDot8 (K : ℕ) : ℚ := chunks8 K * (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K * 2 ^ (-126 : ℤ)

/-- `tl.dot` of fp8 operands `a`, `b` into `float32` with Triton's default
`max_num_imprecise_acc` on sm_90 (`2^30`, so the accumulation is imprecise throughout), any
`input_precision`. -/
def DotF8 (a b : DType) : Prop :=
  ∀ p : InputPrecision, DotAcc P ⟨a, b, .f32, p, 2 ^ 30⟩ (fun _ => True) relDot8 absDot8

/-- **`tl.dot` of `float8e4nv` operands into `float32`** (`DotF8`). On the `wgmma` path each
instruction of `32` products keeps only the terms' bits down to `2^-13` of the largest, the
accumulator included, so each is within `33·2^-13` times its terms' magnitudes' sum, plus a
rounding, below `2^-7`; on the
`mma.sync` path Triton upcasts to `float16`. Per output element: within
`((1 + 2^-7)^⌈K/32⌉ − 1)·(|acc| + Σ|a·b|) + ⌈K/32⌉·(1 + 2^-7)^⌈K/32⌉·2^-126` of `acc + Σ a·b`.
Evidence: `evidence/num-dot`. -/
def DotE4m3 : Prop := DotF8 P .f8e4nv .f8e4nv

/-- **`tl.dot` of `float8e5` operands into `float32`** (`DotF8`, as `DotE4m3`). Evidence:
`evidence/num-dot`. -/
def DotE5m2 : Prop := DotF8 P .f8e5 .f8e5

/-- **`tl.dot` of `float8e4nv` by `float8e5` into `float32`** (`DotF8`). Evidence:
`evidence/num-dot`. -/
def DotE4m3E5m2 : Prop := DotF8 P .f8e4nv .f8e5

/-- **`tl.dot` of `float8e5` by `float8e4nv` into `float32`** (`DotF8`). Evidence:
`evidence/num-dot`. -/
def DotE5m2E4m3 : Prop := DotF8 P .f8e5 .f8e4nv

/-- **`tl.dot` of `float32` operands with `input_precision="ieee"`** (no tensor cores): each
output element is a sequence of `K` `fma.rn.f32`, `k = 0, …, K-1`, from the accumulator. Per
element: within `(K+1)·2^-23·(|acc| + Σ|a·b|) + (K+1)·2^-149` of `acc + Σ a·b`, with room. This is
also the dot that `tl.sum_bcast_mul` may become. Evidence: `evidence/num-dot` (bit-for-bit
against the sequential `fma`, and adversarial inputs). -/
def DotF32Ieee : Prop :=
  DotAcc P ⟨.f32, .f32, .f32, .ieee, 0⟩ (fun _ => True)
    (fun K => (K + 1) * 2 ^ (-23 : ℤ)) (fun K => (K + 1) * 2 ^ (-149 : ℤ))

/-! ## `tl.dot_scaled` on sm_90 -/

/-- The scaled values the sm_90 emulation multiplies exactly in `bfloat16`: `0`, or a magnitude
in `bfloat16`'s normal range. -/
def scaledDomBF16 (q : ℚ) : Prop :=
  q = 0 ∨ ((2 : ℚ) ^ (-126 : ℤ) ≤ |q| ∧ |q| ≤ Format.bfloat16.maxVal)

/-- The `e8m0` scale bits the emulation decodes as the OCP value: `1 … 254` (`0x00` is read as
`0`, `0xFF` is NaN). -/
def e8m0Dom (b : ℕ) : Prop := 1 ≤ b ∧ b ≤ 254

/-- `tl.dot_scaled` with `e8m0` scales on both sides, groups of `32`, `K`-packed operands and
either `fast_math`, as sm_90 runs it: each element and its scale are converted to `bfloat16` and
multiplied there (exact on `scaledDomBF16`), then a `bfloat16` `tl.dot` into `float32`
(`DotBF16`'s constants). -/
def DotScaledMx (lhs rhs : MXFmt) : Prop :=
  ∀ fm : Bool, DotScaledAcc P ⟨lhs, rhs, .e8m0, .e8m0, 32, true, true, fm⟩ scaledDomBF16 e8m0Dom
    relDot16 absDot16

/-- **`tl.dot_scaled(…, "e4m3", …, "e4m3", …)`** (MXFP8) on sm_90: Triton's emulation
(`triton-3.8.0-DecomposeScaledBlocked.cpp:191-228`), as `DotScaledMx`. Per output element, for
scaled values in `scaledDomBF16` and scale bits in `e8m0Dom`: within `(K+2)·2^-22·mag + (K+2)·2^-126`.
Evidence: `evidence/num-dot-scaled`. -/
def DotScaledE4m3 : Prop := DotScaledMx P .e4m3 .e4m3

/-- **`tl.dot_scaled(…, "e4m3", …, "e2m1", …)`** (MXFP8 by MXFP4, tutorial 10's mixed format) on
sm_90, as `DotScaledMx`. Evidence: `evidence/num-dot-scaled`. -/
def DotScaledE4m3E2m1 : Prop := DotScaledMx P .e4m3 .e2m1

/-! ## Float `atomic_add` -/

/-- **`tl.atomic_add` on `float32`**: `AtomicAddAcc` with `rel = 2^-23`, `abs = 2^-124`. It
covers the pointer atomic (`atom.global.<scope>.<sem>.add.f32` and its `.v4` form, every `sem`
and `scope`: all 12 bit-identical, `evidence/num-atomic-add` `sem_scope.py`) and a
descriptor's `atomic_add` (the TMA reduction `cp.reduce.async.bulk.tensor ….add`). Rounding to
nearest with subnormal inputs and results flushed to zero, as `atom.add.f32` does, stays within
`2^-24·|o + v| + 2^-125`; the fact keeps a factor `2` on both. Evidence:
`evidence/num-atomic-add`. -/
def AtomicAddF32 : Prop := AtomicAddAcc P .binary32 (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ))

end TVA.Rules.Num
