import TVA.Triton.Dot
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# What a device fact can say

A `Platform` (`TVA.Platform`) gives, as relations, the operations Triton leaves to the device:
approximate math, the default float division, `tl.dot`, conversions into fp8, float
`tl.atomic_add`, `tl.dot_scaled`, and the depth of a float reduction's tree. A device fact is a
property of a platform `P`. This file defines the *shapes* such a property takes; each fact of
`TVA.Device.Sm90` and `TVA.Device.Dot8Chunk` and `TVA.Device.Dot16Chunk` is one shape at one operation, format and
configuration, with its constants written in.

* **Totality** (`MathTotal`, `DivTotal`, `CvtTotal`, `DotTotal`, `AtomicAddTotal`,
  `DotScaledTotal`): the operation has a result for every operand. On the device an instruction
  always produces something; a proof needs this to know that a program does not get stuck.
* **Accuracy** (`MathAcc`, `DivAcc`, `AtomicAddAcc`, `DotAcc`, `DotLiftAcc`, `DotScaledAcc`):
  on operands in a domain, and when the exact result leaves room in the format, every result is
  finite and within `rel · magnitude + abs` of the exact result.
* **Exact results** (`MathAt`, `DivRn`, `CvtRn`, `CvtSat`): at given inputs, or for every input,
  the result is a stated value (up to NaN payload where the device's NaN is not fixed).

The magnitude a `tl.dot` error is relative to is `|acc| + Σ |a·b|`, or, for the 16-bit facts,
`|acc| + Σ liftMag a · liftMag b` (`liftMag`): the tensor core aligns the products of an
instruction to the largest exponent among them, and reads a subnormal operand at the smallest
normal magnitude, so a subnormal operand weighs as if it were that large.

Weaker forms of these shapes (larger constants, smaller domains) follow by the lemmas of
`TVA.Device.Weaken`. `TVA.Device` gathers the facts; `TVA.Device.Ref` shows a platform satisfying
all of them.
-/
namespace TVA

/-- The real function an approximate `tl.*` function stands for. `erf` has none here, so its
accuracy cannot be stated with `MathAcc`. -/
noncomputable def MathFn.ref : MathFn → Option (ℝ → ℝ)
  | .exp => some Real.exp
  | .exp2 => some (fun x => (2 : ℝ) ^ x)
  | .log => some Real.log
  | .log2 => some (Real.logb 2)
  | .sin => some Real.sin
  | .cos => some Real.cos
  | .sqrt => some Real.sqrt
  | .rsqrt => some (fun x => 1 / Real.sqrt x)
  | .erf => none

/-- Where the real function is defined. Outside it Mathlib gives junk values (`Real.log 0 = 0`,
`Real.sqrt (-4) = 0`, `1 / √0 = 0`), which no device result approximates. -/
def MathFn.natDom : MathFn → Set ℝ
  | .log | .log2 | .rsqrt => Set.Ioi 0
  | .sqrt => Set.Ici 0
  | _ => Set.univ

/-- The largest magnitude a dtype holds: the format's `maxVal` for floats, `2^(w-1) - 1` or
`2^w - 1` for integers, `0` for `tl.int1`. A `tl.dot` fact asks the exact result plus its error
to stay below the accumulator's bound. -/
def DType.bound : DType → ℚ
  | .f16 => Format.binary16.maxVal | .bf16 => Format.bfloat16.maxVal
  | .f32 => Format.binary32.maxVal | .f64 => Format.binary64.maxVal
  | .f8e5 => Format.e5m2.maxVal | .f8e4nv => Format.e4m3.maxVal
  | .i8 => 2 ^ 7 - 1 | .i16 => 2 ^ 15 - 1 | .i32 => 2 ^ 31 - 1 | .i64 => 2 ^ 63 - 1
  | .u8 => 2 ^ 8 - 1 | .u16 => 2 ^ 16 - 1 | .u32 => 2 ^ 32 - 1 | .u64 => 2 ^ 64 - 1
  | .i1 => 0

namespace Device

/-- The magnitude below which the tensor core reads a nonzero operand of dtype `d` at this
magnitude: the smallest normal magnitude of the 16- and 8-bit float dtypes (`2^-14` for
`float16` and `float8e5`, `2^-6` for `float8e4nv`, `2^-126` for `bfloat16`), and `0` (no lift)
for the other dtypes. -/
def liftFloor : DType → ℚ
  | .f8e5 => 2 ^ (-14 : ℤ)
  | .f8e4nv => 2 ^ (-6 : ℤ)
  | .f16 => 2 ^ (-14 : ℤ)
  | .bf16 => 2 ^ (-126 : ℤ)
  | _ => 0

/-- The magnitude the tensor core aligns an operand value `q` of dtype `d` by: `0` for `0`, and
otherwise `|q|`, or `liftFloor d` when `|q|` is below it (a subnormal `float16`, `bfloat16` or
fp8 value reads as the smallest normal). For zero and normal values it is `|q|`. -/
def liftMag (d : DType) (q : ℚ) : ℚ := if q = 0 then 0 else max |q| (liftFloor d)

variable (P : Platform)

/-! ## Totality -/

/-- `P.math fn` has a result for every input of format `f`. -/
def MathTotal (fn : MathFn) (f : Format) : Prop := ∀ x : FloatV f, ∃ r, P.math fn f x r

/-- `P.div` has a result for every pair of inputs of format `f`. -/
def DivTotal (f : Format) : Prop := ∀ x y : FloatV f, ∃ r, P.div f x y r

/-- `P.cvt f g rnd` has a result for every input. -/
def CvtTotal (f g : Format) (rnd : RoundMode := .rn) : Prop :=
  ∀ x : FloatV f, ∃ r, P.cvt f g rnd x r

/-- `P.dot cfg` has a result for all operands of all sizes. -/
def DotTotal (cfg : DotCfg) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc : Tensor [M, N] cfg.acc.Val), ∃ r, P.dot cfg a b acc r

/-- `P.atomicAdd f` has a result for every old value and operand. -/
def AtomicAddTotal (f : Format) : Prop := ∀ o v : FloatV f, ∃ n, P.atomicAdd f o v n

/-- `P.dotScaled cfg` has a result for all operands of all sizes. -/
def DotScaledTotal (cfg : ScaledCfg) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] Nat) (sa : Option (Tensor [M, K] Nat))
    (b : Tensor [K, N] Nat) (sb : Option (Tensor [K, N] Nat)) (acc : Tensor [M, N] F32),
    ∃ r, P.dotScaled cfg a sa b sb acc r

/-! ## Approximate functions -/

/-- **Accuracy of an approximate function.** On finite inputs `x` in `dom` and in the function's
natural domain (`MathFn.natDom`), whose reference value leaves room in the format
(`(1 + rel)·|ref x| + abs ≤ maxVal`), every result of `P.math fn` is finite and within
`rel · |ref x| + abs` of the real function (`MathFn.ref`). Without the room condition, `exp 126`
(above `2^181`) would need a finite result in `float32`, which does not exist. -/
def MathAcc (fn : MathFn) (f : Format) (dom : Set ℝ) (rel abs : ℝ) : Prop :=
  ∃ ref, fn.ref = some ref ∧ ∀ x r : FloatV f, P.math fn f x r → x.isFinite = true →
    x.toReal ∈ dom → x.toReal ∈ fn.natDom → (1 + rel) * |ref x.toReal| + abs ≤ (f.maxVal : ℝ) →
    r.isFinite = true ∧ |r.toReal - ref x.toReal| ≤ rel * |ref x.toReal| + abs

/-- **A special value.** Every result of `P.math fn` at the input `x` is `y` (for example
`exp(-∞) = +0`). -/
def MathAt (fn : MathFn) (f : Format) (x y : FloatV f) : Prop := ∀ r, P.math fn f x r → r = y

/-! ## Division -/

/-- **Accuracy of the default division** (`/`, `tl.fdiv` with `ieee_rounding=False`). For finite
`x`, `y`, `y ≠ 0` and `|x/y| ≤ qmax`, every result is finite and within `rel · |x/y| + abs` of
`x / y`. -/
def DivAcc (f : Format) (qmax rel abs : ℚ) : Prop :=
  ∀ x y r : FloatV f, P.div f x y r → x.isFinite = true → y.isFinite = true → y.toRat ≠ 0 →
    |x.toRat / y.toRat| ≤ qmax →
    r.isFinite = true ∧ |r.toRat - x.toRat / y.toRat| ≤ rel * |x.toRat / y.toRat| + abs

/-- **The default division on format `f` is IEEE division**, rounded to nearest even
(`FloatV.divRn`), up to NaN payload. -/
def DivRn (f : Format) : Prop :=
  ∀ x y r : FloatV f, P.div f x y r → FloatV.NaNEq (FloatV.divRn x y) r

/-! ## Conversions -/

/-- **A conversion rounds to nearest even in range.** A finite value within `g`'s range converts,
with the default rounding `.rn`, to its rounding to nearest even (`FloatV.cvtWith .rn`), up to
NaN payload. It says nothing about inputs beyond `g.maxVal`, infinities or NaN; `CvtSat` does. -/
def CvtRn (f g : Format) : Prop :=
  ∀ x : FloatV f, ∀ r : FloatV g, P.cvt f g .rn x r → x.isFinite = true →
    |x.toRat| ≤ g.maxVal → FloatV.NaNEq (FloatV.cvtWith .rn x) r

/-- **A conversion saturates.** Every input converts, with the default rounding `.rn`, to its
saturating rounding to nearest even (`FloatV.cvtSat .rn`), up to NaN payload: a finite value
within `g`'s range to its rounding, a finite value beyond the range and `±∞` to `±maxVal` of `g`
(PTX `.satfinite`), a NaN to a NaN. It implies `CvtRn` (`CvtSat.cvtRn`). -/
def CvtSat (f g : Format) : Prop :=
  ∀ x : FloatV f, ∀ r : FloatV g, P.cvt f g .rn x r → FloatV.NaNEq (FloatV.cvtSat .rn x) r

/-- **A conversion is within one rounding in range**: a finite value within `g`'s range converts
(with `.rn`) to a finite value within `|x|·u + η` of `x`. It is what `CvtRn` gives
(`CvtRn.cvtAcc`); no fact states it directly. -/
def CvtAcc (f g : Format) : Prop :=
  ∀ x : FloatV f, ∀ r : FloatV g, P.cvt f g .rn x r → x.isFinite = true →
    |x.toRat| ≤ g.maxVal →
    r.isFinite = true ∧ |r.toRat - x.toRat| ≤ |x.toRat| * g.u + g.eta

/-! ## Float `atomic_add` -/

/-- **Accuracy of a float `tl.atomic_add`.** If the old value `o` and the operand `v` are finite
and their exact sum is in range, the new value is finite and within `rel · |o + v| + abs` of the
sum. -/
def AtomicAddAcc (f : Format) (rel abs : ℚ) : Prop :=
  ∀ o v n : FloatV f, P.atomicAdd f o v n → o.isFinite = true → v.isFinite = true →
    |o.toRat + v.toRat| ≤ f.maxVal →
    n.isFinite = true ∧ |n.toRat - (o.toRat + v.toRat)| ≤ rel * |o.toRat + v.toRat| + abs

/-! ## `tl.dot` -/

/-- **Accuracy of `tl.dot`**, per output element `(i, n)`. Suppose row `i` of `a` and column `n`
of `b` have values `va`, `vb` (`DType.val?`: finite floats, integers) in `dom`, and the
accumulator element has value `vc`. Then the result element has a value within
`rel K · mag + abs K` of `vc + Σₖ va·vb`, where `mag = |vc| + Σₖ |va·vb|`, provided
`mag + rel K · mag + abs K` is at most the accumulator's `DType.bound`. With an integer
accumulator and `rel = abs = 0` it says that an exact sum in range is the result. -/
def DotAcc (cfg : DotCfg) (dom : ℚ → Prop) (rel abs : ℕ → ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc r : Tensor [M, N] cfg.acc.Val), P.dot cfg a b acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ),
      (∀ k, cfg.a.val? (a.get2 i k) = some (va k)) → (∀ k, cfg.b.val? (b.get2 k n) = some (vb k)) →
      (∀ k, dom (va k) ∧ dom (vb k)) → cfg.acc.val? (acc.get2 i n) = some vc →
      |vc| + ∑ k, |va k * vb k| + (rel K * (|vc| + ∑ k, |va k * vb k|) + abs K) ≤ cfg.acc.bound →
      ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
        |vr - (vc + ∑ k, va k * vb k)| ≤ rel K * (|vc| + ∑ k, |va k * vb k|) + abs K

/-- **Accuracy of `tl.dot` relative to lifted magnitudes**, per output element `(i, n)`. As
`DotAcc` on every operand value, with `L = |vc| + Σₖ liftMag cfg.a (va k) · liftMag cfg.b (vb k)`
in place of `|vc| + Σₖ |va·vb|`: provided `L + rel K · L + abs K` is at most the accumulator's
`DType.bound`, the result element has a value within `rel K · L + abs K` of `vc + Σₖ va·vb`. For
zero and normal operands `L` is `DotAcc`'s magnitude. -/
def DotLiftAcc (cfg : DotCfg) (rel abs : ℕ → ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc r : Tensor [M, N] cfg.acc.Val), P.dot cfg a b acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ),
      (∀ k, cfg.a.val? (a.get2 i k) = some (va k)) → (∀ k, cfg.b.val? (b.get2 k n) = some (vb k)) →
      cfg.acc.val? (acc.get2 i n) = some vc →
      |vc| + ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k) +
          (rel K * (|vc| + ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k)) + abs K) ≤
        cfg.acc.bound →
      ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
        |vr - (vc + ∑ k, va k * vb k)| ≤
          rel K * (|vc| + ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k)) + abs K

/-! ## `tl.dot_scaled` -/

/-- **Accuracy of `tl.dot_scaled`**, per output element `(i, n)`. Suppose row `i` of `lhs` and
column `n` of `rhs` have finite scaled values `va`, `vb` (`scaledValue`) in `dom`, their scale
bits are in `sdom`, and the accumulator element is finite. Then the result element is finite and
within `rel K · mag + abs K` of `acc + Σₖ va·vb`, where `mag = |acc| + Σₖ |va·vb|`, provided
`mag + rel K · mag + abs K` is within the `float32` range. `dom` and `sdom` carry what the
device's path needs (the sm_90 emulation multiplies each element by its scale in `bfloat16`, so
`dom` keeps those products exact; it reads `e8m0` scale `0x00` as `0`, so `sdom` excludes it). -/
def DotScaledAcc (cfg : ScaledCfg) (dom : ℚ → Prop) (sdom : Nat → Prop) (rel abs : ℕ → ℚ) :
    Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] Nat) (sa : Option (Tensor [M, K] Nat))
    (b : Tensor [K, N] Nat) (sb : Option (Tensor [K, N] Nat)) (acc r : Tensor [M, N] F32),
    P.dotScaled cfg a sa b sb acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ),
      (∀ k, scaledValue cfg.lhs cfg.lhsScale (a.get2 i k) (sa.map fun t => t.get2 i k) =
        some (va k)) →
      (∀ k, scaledValue cfg.rhs cfg.rhsScale (b.get2 k n) (sb.map fun t => t.get2 k n) =
        some (vb k)) →
      (∀ k, dom (va k) ∧ dom (vb k)) →
      (∀ t, sa = some t → ∀ k, sdom (t.get2 i k)) → (∀ t, sb = some t → ∀ k, sdom (t.get2 k n)) →
      (acc.get2 i n).isFinite = true →
        |(acc.get2 i n).toRat| + ∑ k, |va k * vb k| +
            (rel K * (|(acc.get2 i n).toRat| + ∑ k, |va k * vb k|) + abs K) ≤
          Format.binary32.maxVal →
        (r.get2 i n).isFinite = true ∧
          |(r.get2 i n).toRat - ((acc.get2 i n).toRat + ∑ k, va k * vb k)| ≤
            rel K * (|(acc.get2 i n).toRat| + ∑ k, |va k * vb k|) + abs K

end Device

end TVA
