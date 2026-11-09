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

/-- `+∞` in `float32` (pattern `0x7f800000`). -/
def posInfF32 : F32 := FloatV.ofPat 0x7f800000

/-- **`tl.exp` overflows to `+∞` on `float32`**: every input from `88.72283935546875` (pattern
`0x42b17218`, `1453635/16384`) up, and `+∞`, gives `+∞`. Triton computes `x · log2 e` with
`mul.f32`, then `ex2.approx.f32` (`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:634-658`), which is
`+∞` once the product reaches `128`; the largest input with a finite result is `0x42b17217`
(`88.72283172607422`, result `0x7f7fff4f`). `ExpF32` says nothing there (its room fails).
Evidence: `evidence/num-special-f32` E1 (every `float32` pattern from `+64` to `+∞`; library
issue L154). -/
def ExpF32Overflow : Prop :=
  ∀ x : F32, ((x.isFinite = true ∧ (1453635 / 16384 : ℚ) ≤ x.toRat) ∨ x = posInfF32) →
    ∀ r, P.math .exp .binary32 x r → r = posInfF32

/-! ## Division by an infinite divisor on `float32` -/

/-- **`x / ±∞ = ±0`** on `float32` (`div.full.f32`): for every finite `x` and infinite `y`, the
result is the zero whose sign is `sign x xor sign y` (`1 / ∞ = +0`), as IEEE division gives it
(`Format.div`). `DivFull` says nothing at an infinite divisor. Evidence: `evidence/num-special-f32`
D1 (every finite `float32` `x`, by `+∞` and by `-∞`; library issue L154). -/
def DivF32InfDivisor : Prop :=
  ∀ x y : F32, x.isFinite = true → y.isInf = true → ∀ r, P.div .binary32 x y r →
    r = FloatV.ofPat (Format.binary32.zeroPat (x.sign != y.sign))

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
`input_precision`, which Triton ignores for these operands), relative to lifted magnitudes
(`DotLiftAcc`; library issue L126, round 25). Triton chains `mma.sync` or `wgmma` instructions of
`16` products each along `K`, in increasing `k`. Each instruction aligns the accumulator and its
products to the largest exponent field `E` (a product's field is its operands' fields added, a
subnormal operand's field read as the smallest normal's, `2^-14`), truncates every term toward zero
to a multiple of `2^(E-25)`, sums them exactly and truncates the sum toward zero to `float32`. So a
chunk's error is below `5.25·2^-23·(|s| + A) + 5.25·2^-149`, with `s` the accumulator before it
(which never grows past `|acc| + Σ|a·b|`) and `A` the chunk's sum of `liftMag a·liftMag b`; over
the `⌈K/16⌉` instructions this stays below the constants. Per output element, with `L = |acc| + Σ liftMag a·liftMag b` and room in `float32`:
within `(K+2)·2^-22·L + (K+2)·2^-126` of `acc + Σ a·b`. Before round 25 the fact had `Σ|a·b|` in
place of the lifted sum, which a subnormal operand with a large partner breaks. Evidence:
`evidence/num-dot-sub16-round25` (a bit-exact model of both instruction paths, and adversarial
inputs at every `K` that compiles), `evidence/num-dot`. -/
def DotF16 : Prop :=
  ∀ p : InputPrecision, DotLiftAcc P ⟨.f16, .f16, .f32, p, 0⟩ relDot16 absDot16

/-- **`tl.dot` of `bfloat16` operands into `float32`**: as `DotF16`, with the same constants, a
subnormal operand counted at `2^-126` (the absolute term covers products below `float32`'s normal
range). Evidence: `evidence/num-dot-sub16-round25`, `evidence/num-dot`. -/
def DotBF16 : Prop :=
  ∀ p : InputPrecision, DotLiftAcc P ⟨.bf16, .bf16, .f32, p, 0⟩ relDot16 absDot16

/- The fp8 `tl.dot` facts are `DotE4m3Chunk`, …, `DotE5m2E4m3Chunk` (`TVA.Rules.Num.Dot8Chunk`).
The whole-dot facts `DotE4m3`, `DotE5m2`, `DotE4m3E5m2`, `DotE5m2E4m3` (and their family `DotF8`)
were withdrawn in round 24 (library issue L125): a subnormal operand breaks their bound. -/

/-- **`tl.dot` of `float32` operands with `input_precision="ieee"`** (no tensor cores): each
output element is a sequence of `K` `fma.rn.f32`, `k = 0, …, K-1`, from the accumulator. Per
element: within `(K+1)·2^-23·(|acc| + Σ|a·b|) + (K+1)·2^-149` of `acc + Σ a·b`, with room. This is
also the dot that `tl.sum_bcast_mul` may become. Evidence: `evidence/num-dot` (bit-for-bit
against the sequential `fma`, and adversarial inputs). -/
def DotF32Ieee : Prop :=
  DotAcc P ⟨.f32, .f32, .f32, .ieee, 0⟩ (fun _ => True)
    (fun K => (K + 1) * 2 ^ (-23 : ℤ)) (fun K => (K + 1) * 2 ^ (-149 : ℤ))

/-- **`tl.dot` of `int8` operands into `int32`** (library issue L141, round 31), with or without an
accumulator, any `input_precision` (Triton ignores it for these operands) and
`max_num_imprecise_acc = 0` (the default for operands that are not fp8, `semantic.py:1516-1520`):
exact. Triton 3.8.0's only integer `tl.dot` is this one (`semantic.py:1483-1486`: both operands
`int8`, an `int32` result whatever `out_dtype` says). Triton chains `mma.sync …satfinite.s32.s8.s8.s32`
(`m16n8k32`) or `wgmma …s32.s8.s8` (`k32`) instructions along `K`, which sum in integers: the first
saturates an `int32` partial sum, the second wraps it. When `|acc| + Σ|a·b|` is at most `2^31 - 1`,
no partial sum in any order leaves the `int32` range, so the result is `acc + Σ a·b` exactly. Beyond
that the two paths differ, and the fact says nothing. Evidence: `evidence/num-dot-int8` (random and
extreme operands on both paths and every precision, the room at its edge, and the paths' behaviour
beyond it). -/
def DotI8 : Prop :=
  ∀ p : InputPrecision, DotAcc P ⟨.i8, .i8, .i32, p, 0⟩ (fun _ => True) (fun _ => 0) (fun _ => 0)

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
(`DotBF16`'s constants). On `scaledDomBF16` no operand of that dot is subnormal, so its lifted
magnitudes are `|a·b|` and the bound is relative to `|acc| + Σ|a·b|`. -/
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

/-! ## The depth of a float reduction's tree (library issue L149) -/

namespace TVA.Rules.Num

/-- **The trees of a float `tl.sum` on sm_90 are balanced**: `P.sumDepth n ≤ ⌈log₂ n⌉`, so a float
reduction (`tl.sum`, `tl.sumc` and their axis forms) over the `n` positions of its axis combines
them along a tree of depth at most `⌈log₂ n⌉` (`SumDepth.depth_le`): `log₂ n` for the power of two
`n` of a Triton axis. Triton 3.8.0 lowers a reduction (`ReduceOpToLLVM.cpp`) as a balanced binary
tree over the axis's elements held by one thread (`treeReduce`, arity 2 on NVIDIA), then a
butterfly of `shfl.sync.bfly` over the lanes along the axis, then, through shared memory, the same
butterfly over the warps along the axis: each stage halves the positions left, whatever the layout.
Evidence: `evidence/reduce-tree`. It is not stated for a reduction that the compiler rewrites across
loop iterations (`OptimizeThreadLocality.cpp`, `docs/exclusions.md`). -/
def SumDepth (P : Platform) : Prop := ∀ n, P.sumDepth n ≤ Nat.clog 2 n

/-- With `SumDepth`, a reduction's tree (`t.depth ≤ tl.sumBound P n`) has depth at most `⌈log₂ n⌉`. -/
theorem SumDepth.depth_le {P : Platform} (h : SumDepth P) {n : Nat} {t : Tree}
    (ht : t.depth ≤ tl.sumBound P n) : t.depth ≤ Nat.clog 2 n :=
  ht.trans (max_le (h n) le_rfl)

/-- `⌈log₂ (2^k)⌉ = k`: the depth bound for an axis of `2^k` positions. -/
theorem clog_two_pow (k : Nat) : Nat.clog 2 (2 ^ k) = k := Nat.clog_pow 2 k (by norm_num)

end TVA.Rules.Num
