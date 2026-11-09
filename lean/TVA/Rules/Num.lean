import TVA.Lang.Platform
import TVA.Lang.Scaled
import TVA.Lemmas.FloatErr
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# Rules about the platform (`Rules.Num.*`)

Page: `docs/rules/num.md`.

The platform `P` gives the operations the Triton documentation leaves to the device:
- approximate math (`P.math`);
- the default float division (`P.div`);
- `tl.dot` (`P.dot`);
- fp8 conversions (`P.cvt`);
- float `atomic_add` (`P.atomicAdd`).

A claim quantifies over `P`. These definitions are the shapes of the facts about `P` a bundle
may assume, as fields of its `Assumptions` with kind `[rule]`. The facts themselves, each one
instance for one operation, format and configuration with its constants written into the
definition, are the constants of `TVA.Rules.Num.Sm90` (`Exp2F32`, `DotF16`, …), together with
`DivFull`, `DivRn` and the `…Total` rules here; `docs/rules/num.md` lists them. A bundle assumes
only the facts it uses. Library review judges each fact against the device and Triton 3.8.0's
lowering, with TVA's evidence; a kernel reviewer checks only that a bundle cites the library's
fact for the operation its model uses.

* `MathTotal`, `DivTotal`, `CvtTotal`, `DotTotal`, `AtomicAddTotal`: the operation has a result
  (on the device there is always one). The `Safe` rules need these.
* `MathAt fn f x y`: at the input `x` (typically an infinity), the result is exactly `y`.
* `MathAcc fn f dom rel abs`: on finite inputs in `dom` and in the function's natural domain
  (`MathFn.natDom`), whose reference value leaves room in the format
  (`(1 + rel)·|ref x| + abs ≤ maxVal`), results are finite and within `rel · |ref x| + abs` of
  the real function (`MathFn.ref`).
* `DivAcc f qmax rel abs`: the same for `x / y`, `y ≠ 0`, `|x/y| ≤ qmax`. Two instances name
  what Triton 3.8.0 compiles `/` to on NVIDIA:
  - `DivFull`: `float32` `/` is `div.full.f32`, within `2^-22·|x/y| + 2^-148` of `x/y` for
    `|x/y| ≤ 2^126`;
  - `DivRn f`: `/` is IEEE division (`float64`: `div.rn.f64`). `DivRn.divAcc` turns it into
    `DivAcc` with one rounding's bound (`u`, `η`), and `DivFull.of_divRn` shows that `DivFull` is
    satisfiable.
* `CvtRn f g`: a conversion with the default rounding of a finite value in `g`'s range is its
  rounding to nearest even; `CvtRn.cvtAcc` turns it into `CvtAcc f g`: finite and within one
  rounding's bound (`|x|·u + η`) of the target.
* `CvtSat f g`: a conversion with the default rounding of any input is its saturating rounding
  to nearest even (`FloatV.cvtSat .rn`: `±maxVal` beyond the range and for `±∞`), up to NaN
  payload; `CvtSat.cvtRn` gives `CvtRn f g`.
* `DotAcc cfg dom rel abs`: an output element of `tl.dot` whose row of `a` and column of `b`
  have values (`DType.val?`) in `dom` is within `rel K · (|acc| + Σ |a·b|) + abs K` of
  `acc + Σ a·b`, with room below the accumulator's `DType.bound`. With an integer accumulator
  and `rel = abs = 0`, an exact sum in range is the result.
* `DotLiftAcc cfg rel abs`: the same for every operand value, relative to
  `|acc| + Σ liftMag a · liftMag b` (library issue L126): `liftMag d q` counts a nonzero operand
  below the smallest normal magnitude `minNormal8 d` at that magnitude, as the tensor core aligns
  it (a subnormal `float16`, `bfloat16` or fp8 operand reads as the smallest normal).
* `DotScaledTotal cfg`, `DotScaledAcc cfg dom sdom rel abs`: `tl.dot_scaled` has a result; an
  output element whose row of `lhs` and column of `rhs` have scaled values in `dom` and scale
  bits in `sdom` is within `rel K · (|acc| + Σ |a·b|) + abs K` of `acc + Σ a·b`.
* `AtomicAddAcc f rel abs`: a float `atomic_add` whose exact sum is in range is finite and
  within `rel · |o + v| + abs`. On NVIDIA, `atom.add.f32` flushes subnormal inputs and results.
  Two operands just below the smallest normal then sum to almost `2 · 2^emin` and give `0`, so
  `abs` must be at least `2^(emin + 1)` (`2^-125` for fp32).
* `MathAcc.mono`, `DotAcc.mono`, `DotLiftAcc.mono`, `DotScaledAcc.mono`: a fact implies the same
  statement with larger constants or a smaller domain; `CvtRn.cvtAcc`, `CvtSat.cvtRn`,
  `DivRn.divAcc`, `DotAcc.liftAcc`: a fact implies a weaker shape. They state nothing new about the
  device.
-/
namespace TVA.Rules.Num

open TVA.Math.Float

/-- The real function an approximate `tl.*` function stands for. `erf` has none here, so its
accuracy cannot be stated with `MathAcc`. -/
noncomputable def _root_.TVA.MathFn.ref : MathFn → Option (ℝ → ℝ)
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
def _root_.TVA.MathFn.natDom : MathFn → Set ℝ
  | .log | .log2 | .rsqrt => Set.Ioi 0
  | .sqrt => Set.Ici 0
  | _ => Set.univ

/-- The rational value of a dtype element: a finite float's value, or an integer's. `none` for
NaN, infinities and `int1`. -/
def _root_.TVA.DType.val? : (d : DType) → d.Val → Option ℚ
  | .f16, x => if x.isFinite = true then some x.toRat else none
  | .bf16, x => if x.isFinite = true then some x.toRat else none
  | .f32, x => if x.isFinite = true then some x.toRat else none
  | .f64, x => if x.isFinite = true then some x.toRat else none
  | .f8e5, x => if x.isFinite = true then some x.toRat else none
  | .f8e4nv, x => if x.isFinite = true then some x.toRat else none
  | .i8, x => some x.toInt | .i16, x => some x.toInt | .i32, x => some x.toInt
  | .i64, x => some x.toInt
  | .u8, x => some x.toInt | .u16, x => some x.toInt | .u32, x => some x.toInt
  | .u64, x => some x.toInt
  | .i1, _ => none

/-- The largest magnitude a dtype holds: the format's `maxVal` for floats, `2^(w-1) - 1` or
`2^w - 1` for integers. -/
def _root_.TVA.DType.bound : DType → ℚ
  | .f16 => Format.binary16.maxVal | .bf16 => Format.bfloat16.maxVal
  | .f32 => Format.binary32.maxVal | .f64 => Format.binary64.maxVal
  | .f8e5 => Format.e5m2.maxVal | .f8e4nv => Format.e4m3.maxVal
  | .i8 => 2 ^ 7 - 1 | .i16 => 2 ^ 15 - 1 | .i32 => 2 ^ 31 - 1 | .i64 => 2 ^ 63 - 1
  | .u8 => 2 ^ 8 - 1 | .u16 => 2 ^ 16 - 1 | .u32 => 2 ^ 32 - 1 | .u64 => 2 ^ 64 - 1
  | .i1 => 0

variable (P : Platform)

/-- `P.math fn` has a result for every input of format `f`. -/
def MathTotal (fn : MathFn) (f : Format) : Prop := ∀ x : FloatV f, ∃ r, P.math fn f x r

/-- `P.div` has a result for every pair of inputs of format `f`. -/
def DivTotal (f : Format) : Prop := ∀ x y : FloatV f, ∃ r, P.div f x y r

/-- `P.cvt f g rnd` has a result for every input. -/
def CvtTotal (f g : Format) (rnd : RoundMode := .rn) : Prop := ∀ x : FloatV f, ∃ r, P.cvt f g rnd x r

/-- `P.dot cfg` has a result for all operands of all sizes. -/
def DotTotal (cfg : DotCfg) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc : Tensor [M, N] cfg.acc.Val), ∃ r, P.dot cfg a b acc r

/-- `P.atomicAdd f` has a result for every old value and operand. -/
def AtomicAddTotal (f : Format) : Prop := ∀ o v : FloatV f, ∃ n, P.atomicAdd f o v n

/-- **Accuracy of an approximate function.** On finite inputs `x` with `x ∈ dom`, in the
function's natural domain, and with room in the format for the bound
(`(1 + rel)·|ref x| + abs ≤ maxVal`), every result of `P.math fn` is finite and within
`rel · |ref x| + abs` of the real function. Without the room condition, `exp 126` (above
`2^181`) could not have a finite result in `float32`, and the rule would contradict
`MathTotal`. -/
def MathAcc (fn : MathFn) (f : Format) (dom : Set ℝ) (rel abs : ℝ) : Prop :=
  ∃ ref, fn.ref = some ref ∧ ∀ x r : FloatV f, P.math fn f x r → x.isFinite = true →
    x.toReal ∈ dom → x.toReal ∈ fn.natDom → (1 + rel) * |ref x.toReal| + abs ≤ (f.maxVal : ℝ) →
    r.isFinite = true ∧ |r.toReal - ref x.toReal| ≤ rel * |ref x.toReal| + abs

/-- **A special value.** Every result of `P.math fn` at the input `x` is `y`. For example
`exp(-∞) = +0` in `float32`: Triton computes `exp` as `ex2.approx` of `x · log2 e`
(`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:913-917`), and `ex2.approx` gives `+0` for `-∞`.
Softmax over masked rows (loaded as `-inf`) needs it. -/
def MathAt (fn : MathFn) (f : Format) (x y : FloatV f) : Prop := ∀ r, P.math fn f x r → r = y

/-- **Accuracy of the default division.** For finite `x`, `y`, `y ≠ 0` and `|x/y| ≤ qmax`, every
result is finite and within `rel · |x/y| + abs` of `x / y`. -/
def DivAcc (f : Format) (qmax rel abs : ℚ) : Prop :=
  ∀ x y r : FloatV f, P.div f x y r → x.isFinite = true → y.isFinite = true → y.toRat ≠ 0 →
    |x.toRat / y.toRat| ≤ qmax →
    r.isFinite = true ∧ |r.toRat - x.toRat / y.toRat| ≤ rel * |x.toRat / y.toRat| + abs

/-- **`/` on `float32` is `div.full.f32`**, within `2^-22·|x/y| + 2^-148` of `x/y` for
`|x/y| ≤ 2^126`. Triton 3.8.0 lowers `arith.divf` on `f32` to `llvm.nvvm.div.full`
(`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:553-580`), which LLVM selects as `div.full.f32`
(`llvm-NVPTXIntrinsics.td:1536-1537`). The PTX ISA documents it as "a relatively fast, full-range
approximation that scales operands to achieve better accuracy", with a maximum error of 2 ulp. The
fact is the relative bound, not "2 ulp": near the top of a binade, where the bound is almost 4 ulp
of the quotient's binade, errors reach 2.11 ulp. TVA's evidence: `evidence/num-div-f32`, about
`3.1·10^10` pairs (every divisor mantissa at many exponents, subnormal operands, random pairs), all
within the bound, at most `0.651` of it (`docs/rules/num.md`, "Division"). -/
def DivFull : Prop := DivAcc P .binary32 (2 ^ 126) (2 ^ (-22 : ℤ)) (2 ^ (-148 : ℤ))

/-- **`/` on format `f` is IEEE division**, up to NaN payload. Triton 3.8.0 lowers `arith.divf`
on `f64` to `llvm.nvvm.div.rn.d` (`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:569-571`), which
is `div.rn.f64` or, after LLVM's `instcombine`, an IEEE `fdiv` (`llvm-NVPTXTargetTransformInfo.cpp:298`).
So `DivRn .binary64` holds on NVIDIA. TVA's evidence: `evidence/num-div-f64`, `x / y` and
`tl.fdiv(x, y)` on `float64` against `FloatV.divRn` itself on about `4.5·10^8` pairs (random
patterns, subnormal dividends, quotients and divisors, divisors near `1`, quotients next to
midpoints, constant divisors), all equal up to NaN payload (`docs/rules/num.md`, "Division"). -/
def DivRn (f : Format) : Prop :=
  ∀ x y r : FloatV f, P.div f x y r → FloatV.NaNEq (FloatV.divRn x y) r

/-- **A round-to-nearest conversion is accurate in range.** A finite value within `g`'s range
converts (with the default `.rn`) to a finite value within `|x|·u + η` of `x`, the bound
`FloatV.cvt_err` gives for one rounding. It is judged per source format and architecture: before
sm_90, `float32` goes to fp8 through a truncation to `float16`, which is not always the rounding to
nearest but stays within this bound. -/
def CvtAcc (f g : Format) : Prop :=
  ∀ x : FloatV f, ∀ r : FloatV g, P.cvt f g .rn x r → x.isFinite = true → |x.toRat| ≤ g.maxVal →
    r.isFinite = true ∧ |r.toRat - x.toRat| ≤ |x.toRat| * g.u + g.eta

/-- **Accuracy of a float `atomic_add`.** If the old value and the operand are finite and their
exact sum is in range, the new value is finite and within `rel · |o + v| + abs` of the sum. -/
def AtomicAddAcc (f : Format) (rel abs : ℚ) : Prop :=
  ∀ o v n : FloatV f, P.atomicAdd f o v n → o.isFinite = true → v.isFinite = true →
    |o.toRat + v.toRat| ≤ f.maxVal →
    n.isFinite = true ∧ |n.toRat - (o.toRat + v.toRat)| ≤ rel * |o.toRat + v.toRat| + abs

/-- `P.dotScaled cfg` has a result for all operands of all sizes. -/
def DotScaledTotal (cfg : ScaledCfg) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] Nat) (sa : Option (Tensor [M, K] Nat)) (b : Tensor [K, N] Nat)
    (sb : Option (Tensor [K, N] Nat)) (acc : Tensor [M, N] F32), ∃ r, P.dotScaled cfg a sa b sb acc r

/-- **Accuracy of `tl.dot_scaled`**, per output element `(i, n)`. Suppose row `i` of `lhs`
and column `n` of `rhs` have finite scaled values `va`, `vb` (`scaledValue`) in `dom`, their
scale bits are in `sdom`, and the accumulator element is finite. Then the result element is
finite and within `rel K · mag + abs K` of `acc + Σₖ va·vb`, where `mag = |acc| + Σₖ |va·vb|`,
provided `mag + rel K · mag + abs K` is within the `float32` range. `dom` and `sdom` carry
what the target's path needs. The sm_90 emulation multiplies each element by its scale in
`bfloat16`/`float16` (`triton-3.8.0-DecomposeScaledBlocked.cpp:191-228`), so `dom` must keep
those products in range, and it reads `e8m0` scale `0x00` as `0`. -/
def DotScaledAcc (cfg : ScaledCfg) (dom : ℚ → Prop) (sdom : Nat → Prop) (rel abs : ℕ → ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] Nat) (sa : Option (Tensor [M, K] Nat)) (b : Tensor [K, N] Nat)
    (sb : Option (Tensor [K, N] Nat)) (acc r : Tensor [M, N] F32), P.dotScaled cfg a sa b sb acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ),
      (∀ k, scaledValue cfg.lhs cfg.lhsScale (a.get2 i k) (sa.map fun t => t.get2 i k) = some (va k)) →
      (∀ k, scaledValue cfg.rhs cfg.rhsScale (b.get2 k n) (sb.map fun t => t.get2 k n) = some (vb k)) →
      (∀ k, dom (va k) ∧ dom (vb k)) →
      (∀ t, sa = some t → ∀ k, sdom (t.get2 i k)) → (∀ t, sb = some t → ∀ k, sdom (t.get2 k n)) →
      (acc.get2 i n).isFinite = true →
        |(acc.get2 i n).toRat| + ∑ k, |va k * vb k| +
            (rel K * (|(acc.get2 i n).toRat| + ∑ k, |va k * vb k|) + abs K) ≤ Format.binary32.maxVal →
        (r.get2 i n).isFinite = true ∧
          |(r.get2 i n).toRat - ((acc.get2 i n).toRat + ∑ k, va k * vb k)| ≤
            rel K * (|(acc.get2 i n).toRat| + ∑ k, |va k * vb k|) + abs K

/-- **Accuracy of `tl.dot`**, per output element `(i, n)`. Suppose row `i` of `a` and
column `n` of `b` have values `va`, `vb` (`DType.val?`: finite floats, integers) in `dom`, and
the accumulator element has value `vc`. Then the result element has a value within
`rel K · mag + abs K` of `vc + Σₖ va·vb`, where `mag = |vc| + Σₖ |va·vb|`, provided
`mag + rel K · mag + abs K` is at most the accumulator's `DType.bound`. The constants hold for
one `cfg`, and library review judges them with its input precision (`tf32` keeps 10 fraction bits
of `float32` operands) and its `maxImpreciseAcc` (how many fp8 products sm_90 may sum in the
tensor core's reduced-precision accumulator before adding them into `float32`). With an
integer accumulator and `rel = abs = 0`, the rule says that an exact sum in range is the
result; Triton 3.8.0's only integer `tl.dot` is `int8 × int8 → int32`
(`semantic.py:1483-1486`), and the library states this for it as `Rules.Num.DotI8` (library issue
L141). -/
def DotAcc (cfg : DotCfg) (dom : ℚ → Prop) (rel abs : ℕ → ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc r : Tensor [M, N] cfg.acc.Val), P.dot cfg a b acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ),
      (∀ k, cfg.a.val? (a.get2 i k) = some (va k)) → (∀ k, cfg.b.val? (b.get2 k n) = some (vb k)) →
      (∀ k, dom (va k) ∧ dom (vb k)) → cfg.acc.val? (acc.get2 i n) = some vc →
      |vc| + ∑ k, |va k * vb k| + (rel K * (|vc| + ∑ k, |va k * vb k|) + abs K) ≤ cfg.acc.bound →
      ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
        |vr - (vc + ∑ k, va k * vb k)| ≤ rel K * (|vc| + ∑ k, |va k * vb k|) + abs K

/-- The smallest normal magnitude of a dtype whose subnormal values the tensor core reads at it:
`2^-14` for `float8e5` and `float16`, `2^-6` for `float8e4nv`, `2^-126` for `bfloat16`; `0` for the
other dtypes (no lift). The name is from round 24, when only the fp8 facts used it; round 25 added
`float16` and `bfloat16` (library issue L126). -/
def minNormal8 : DType → ℚ
  | .f8e5 => 2 ^ (-14 : ℤ)
  | .f8e4nv => 2 ^ (-6 : ℤ)
  | .f16 => 2 ^ (-14 : ℤ)
  | .bf16 => 2 ^ (-126 : ℤ)
  | _ => 0

/-- The magnitude the tensor core aligns an operand value `q` of dtype `d` by: `0` for `0`, and
otherwise `|q|`, or the smallest normal magnitude when `|q|` is below it (a subnormal `float16`,
`bfloat16` or fp8 value). For normal values and zero it is `|q|`. -/
def liftMag (d : DType) (q : ℚ) : ℚ := if q = 0 then 0 else max |q| (minNormal8 d)

/-- **Accuracy of `tl.dot` relative to lifted magnitudes**, per output element `(i, n)`. Suppose
row `i` of `a` and column `n` of `b` have values `va`, `vb` (`DType.val?`: any finite floats), and
the accumulator element has value `vc`. Let `L = |vc| + Σₖ liftMag cfg.a (va k) · liftMag cfg.b (vb k)`.
Then, provided `L + rel K · L + abs K` is at most the accumulator's `DType.bound`, the result
element has a value within `rel K · L + abs K` of `vc + Σₖ va·vb`. It is `DotAcc` on every value
with `L` in place of `|vc| + Σₖ |va·vb|`: the tensor core aligns a chunk's terms to the largest
exponent field, and an operand below the smallest normal reads as `minNormal8`, so a product with a
subnormal operand can set the window while far smaller (library issue L126). For zero and normal
operands `L` is `|vc| + Σₖ |va·vb|`. -/
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

/-- **A conversion rounds to nearest even in range.** A finite value within `g`'s range
converts, with the default `.rn`, to its rounding to nearest even (`FloatV.cvtWith .rn`, up to
NaN payload). It says nothing about inputs beyond `g.maxVal` (Triton's fp8 conversions saturate
them), infinities or NaN; `CvtSat` does. `CvtRn.cvtAcc` turns it into `CvtAcc`. -/
def CvtRn (f g : Format) : Prop :=
  ∀ x : FloatV f, ∀ r : FloatV g, P.cvt f g .rn x r → x.isFinite = true → |x.toRat| ≤ g.maxVal →
    FloatV.NaNEq (FloatV.cvtWith .rn x) r

/-- **A conversion saturates.** Every input converts, with the default `.rn`, to its saturating
rounding to nearest even (`FloatV.cvtSat .rn`, up to NaN payload): a finite value within `g`'s
range to its rounding (as `CvtRn` says), a finite value beyond the range and `±∞` to `±maxVal` of
`g` (PTX `.satfinite`), and a NaN to a NaN. `CvtSat.cvtRn` gives `CvtRn`; `FloatV.cvtSat_rn_err`
bounds the result against the clamped value, and `FloatV.cvtSat_of_gt`, `cvtSat_of_isInf` give
`±maxVal`. -/
def CvtSat (f g : Format) : Prop :=
  ∀ x : FloatV f, ∀ r : FloatV g, P.cvt f g .rn x r → FloatV.NaNEq (FloatV.cvtSat .rn x) r

/-! ## Weakening a fact

A kernel may use a fact with larger constants or a smaller domain than the library states. These
lemmas derive the weaker statement; they state nothing new about the device. -/

section Mono
variable {P}

/-- `MathAcc` on a smaller domain with a bound that is at least as large at every point. -/
theorem MathAcc.mono {fn : MathFn} {f : Format} {dom dom' : Set ℝ} {rel abs rel' abs' : ℝ}
    (h : MathAcc P fn f dom rel abs) (hdom : dom' ⊆ dom)
    (hb : ∀ g, fn.ref = some g → ∀ x ∈ dom', x ∈ fn.natDom →
      rel * |g x| + abs ≤ rel' * |g x| + abs') :
    MathAcc P fn f dom' rel' abs' := by
  obtain ⟨g, hg, h⟩ := h
  refine ⟨g, hg, fun x r hr hx hd hn hroom => ?_⟩
  have hb' := hb g hg _ hd hn
  obtain ⟨hfin, herr⟩ := h x r hr hx (hdom hd) hn (by linarith)
  exact ⟨hfin, herr.trans hb'⟩

/-- `DotAcc` on a smaller domain with constants at least as large. -/
theorem DotAcc.mono {cfg : DotCfg} {dom dom' : ℚ → Prop} {rel abs rel' abs' : ℕ → ℚ}
    (h : DotAcc P cfg dom rel abs) (hdom : ∀ q, dom' q → dom q) (hrel : ∀ K, rel K ≤ rel' K)
    (habs : ∀ K, abs K ≤ abs' K) : DotAcc P cfg dom' rel' abs' := by
  intro M N K a b acc r hr i n va vb vc hva hvb hd hvc hroom
  have hmag : 0 ≤ |vc| + ∑ k, |va k * vb k| :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
  have hm := mul_le_mul_of_nonneg_right (hrel K) hmag
  obtain ⟨vr, hvr, herr⟩ := h a b acc r hr i n va vb vc hva hvb
    (fun k => ⟨hdom _ (hd k).1, hdom _ (hd k).2⟩) hvc (by linarith [habs K])
  exact ⟨vr, hvr, by linarith [habs K]⟩

private theorem liftMag_nonneg' (d : DType) (q : ℚ) : 0 ≤ liftMag d q := by
  unfold liftMag
  split_ifs
  · exact le_rfl
  · exact (abs_nonneg q).trans (le_max_left _ _)

private theorem abs_mul_le_liftMag' (da db : DType) (x y : ℚ) :
    |x * y| ≤ liftMag da x * liftMag db y := by
  have hx : |x| ≤ liftMag da x := by
    unfold liftMag; split_ifs with h
    · rw [h, abs_zero]
    · exact le_max_left _ _
  have hy : |y| ≤ liftMag db y := by
    unfold liftMag; split_ifs with h
    · rw [h, abs_zero]
    · exact le_max_left _ _
  rw [abs_mul]
  exact mul_le_mul hx hy (abs_nonneg y) (liftMag_nonneg' da x)

/-- `DotLiftAcc` with constants at least as large. -/
theorem DotLiftAcc.mono {cfg : DotCfg} {rel abs rel' abs' : ℕ → ℚ} (h : DotLiftAcc P cfg rel abs)
    (hrel : ∀ K, rel K ≤ rel' K) (habs : ∀ K, abs K ≤ abs' K) : DotLiftAcc P cfg rel' abs' := by
  intro M N K a b acc r hr i n va vb vc hva hvb hvc hroom
  have hL : 0 ≤ |vc| + ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k) :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ =>
      mul_nonneg (liftMag_nonneg' _ _) (liftMag_nonneg' _ _))
  have hm := mul_le_mul_of_nonneg_right (hrel K) hL
  obtain ⟨vr, hvr, herr⟩ := h a b acc r hr i n va vb vc hva hvb hvc (by linarith [habs K])
  exact ⟨vr, hvr, by linarith [habs K]⟩

/-- A `DotAcc` fact on every value gives `DotLiftAcc` with the same constants, since
`|va·vb| ≤ liftMag va · liftMag vb`: the lifted room implies `DotAcc`'s, and `DotAcc`'s bound is
within the lifted one. -/
theorem DotAcc.liftAcc {cfg : DotCfg} {rel abs : ℕ → ℚ} (h : DotAcc P cfg (fun _ => True) rel abs)
    (hrel : ∀ K, 0 ≤ rel K) : DotLiftAcc P cfg rel abs := by
  intro M N K a b acc r hr i n va vb vc hva hvb hvc hroom
  have hle : ∑ k, |va k * vb k| ≤ ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k) :=
    Finset.sum_le_sum fun k _ => abs_mul_le_liftMag' _ _ _ _
  have hm := mul_le_mul_of_nonneg_left (add_le_add_left hle |vc|) (hrel K)
  obtain ⟨vr, hvr, herr⟩ := h a b acc r hr i n va vb vc hva hvb (fun _ => ⟨trivial, trivial⟩) hvc
    (by linarith)
  exact ⟨vr, hvr, by linarith⟩

/-- `DotScaledAcc` on smaller domains with constants at least as large. -/
theorem DotScaledAcc.mono {cfg : ScaledCfg} {dom dom' : ℚ → Prop} {sdom sdom' : Nat → Prop}
    {rel abs rel' abs' : ℕ → ℚ} (h : DotScaledAcc P cfg dom sdom rel abs)
    (hdom : ∀ q, dom' q → dom q) (hsdom : ∀ b, sdom' b → sdom b) (hrel : ∀ K, rel K ≤ rel' K)
    (habs : ∀ K, abs K ≤ abs' K) : DotScaledAcc P cfg dom' sdom' rel' abs' := by
  intro M N K a sa b sb acc r hr i n va vb hva hvb hd hsa hsb hacc hroom
  have hmag : 0 ≤ |(acc.get2 i n).toRat| + ∑ k, |va k * vb k| :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
  have hm := mul_le_mul_of_nonneg_right (hrel K) hmag
  obtain ⟨hfin, herr⟩ := h a sa b sb acc r hr i n va vb hva hvb
    (fun k => ⟨hdom _ (hd k).1, hdom _ (hd k).2⟩) (fun t ht k => hsdom _ (hsa t ht k))
    (fun t ht k => hsdom _ (hsb t ht k)) hacc (by linarith [habs K])
  exact ⟨hfin, by linarith [habs K]⟩

/-- Round-to-nearest in range gives `CvtAcc` (one rounding: `FloatV.cvt_err`). -/
theorem CvtRn.cvtAcc {f g : Format} (hg : g.Regular) (h : CvtRn P f g) : CvtAcc P f g := by
  intro x r hr hx hmax
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err hg hx hmax
  have hne := h x r hr hx hmax
  rw [FloatV.cvtWith_rn] at hne
  rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
  exact ⟨hfin, herr⟩

/-- A saturating conversion rounds to nearest even in range (`FloatV.cvtSat_of_le`). -/
theorem CvtSat.cvtRn {f g : Format} (hg : g.Regular) (h : CvtSat P f g) : CvtRn P f g := by
  intro x r hr hx hmax
  have := h x r hr
  rwa [FloatV.cvtSat_of_le hg .rn hx hmax] at this

end Mono

/-! ## Consequences of the division rules -/

section Div
variable {P}

/-- IEEE division meets `DivAcc` with one rounding's bound (`u`, `η`), over the whole range of the
format. -/
theorem DivRn.divAcc {f : Format} (hf : f.Regular) (h : DivRn P f) : DivAcc P f f.maxVal f.u f.eta := by
  intro x y r hr hx hy hy0 hmax
  have hz : f.isZero y.pat = false := by
    cases hzy : f.isZero y.pat
    · rfl
    · exact absurd (FloatV.toRat_of_isZero hzy) hy0
  obtain ⟨hfin, herr⟩ := FloatV.divRn_err hf hx hy hz hmax
  rw [(h x y r hr).eq (FloatV.isFinite_iff.mp hfin).1]
  exact ⟨hfin, by rw [mul_comm]; exact herr⟩

/-- IEEE `float32` division meets `DivFull`. So `DivFull` is satisfiable, and a proof that
cites it also holds when `/` is IEEE. -/
theorem DivFull.of_divRn (h : DivRn P .binary32) : DivFull P := by
  intro x y r hr hx hy hy0 hq
  have hmax : (2 : ℚ) ^ 126 ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]; norm_num
  obtain ⟨hfin, herr⟩ := DivRn.divAcc Format.binary32_regular h x y r hr hx hy hy0 (hq.trans hmax)
  refine ⟨hfin, herr.trans ?_⟩
  have hu : Format.binary32.u ≤ 2 ^ (-22 : ℤ) := by
    rw [Format.binary32_u]; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num)
  have he : Format.binary32.eta ≤ 2 ^ (-148 : ℤ) := by
    rw [Format.binary32_eta]; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num)
  exact add_le_add (mul_le_mul_of_nonneg_right hu (abs_nonneg _)) he

end Div

end TVA.Rules.Num
