import TVA.Lang.Platform
import TVA.Lang.Scaled
import TVA.Lemmas.Num
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Rules about the platform (`Rules.Num.*`)

Page: `docs/rules/num.md`.

The platform `P` gives the operations the Triton documentation leaves to the device:
- approximate math (`P.math`);
- the default float division (`P.div`);
- `tl.dot` (`P.dot`);
- fp8 conversions (`P.cvt`);
- float `atomic_add` (`P.atomicAdd`).

A claim quantifies over `P`. These definitions are the facts about `P` a bundle may assume, as
fields of its `Assumptions` with kind `[rule]`. Each is stated for one operation and one format,
so a bundle assumes only what it uses. The reviewer judges each instance against the device and
Triton 3.8.0's lowering, with the evidence it asks for.

* `MathTotal`, `DivTotal`, `CvtTotal`, `DotTotal`, `AtomicAddTotal`: the operation has a result
  (on the device there is always one). The `Safe` rules need these.
* `MathAt fn f x y`: at the input `x` (typically an infinity), the result is exactly `y`.
* `MathAcc fn f dom rel abs`: on finite inputs in `dom` and in the function's natural domain
  (`MathFn.natDom`), whose reference value leaves room in the format
  (`(1 + rel)·|ref x| + abs ≤ maxVal`), results are finite and within `rel · |ref x| + abs` of
  the real function (`MathFn.ref`).
* `DivAcc f qmax rel abs`: the same for `x / y`, `y ≠ 0`, `|x/y| ≤ qmax`. Two instances name
  what Triton 3.8.0 compiles `/` to on NVIDIA:
  - `DivFull`: `float32` `/` is `div.full.f32`, at most 2 ulp from `x/y`;
  - `DivRn f`: `/` is IEEE division (`float64`: `div.rn.f64`). `DivRn.divAcc` turns it into
    `DivAcc` with half an ulp, and `DivFull.of_divRn` shows that `DivFull` is satisfiable.
* `CvtAcc f g`: an fp8 (or other) conversion of a finite value in range is finite and within
  half an ulp of the target.
* `DotAcc cfg dom rel abs`: an output element of `tl.dot` whose row of `a` and column of `b`
  have values (`DType.val?`) in `dom` is within `rel K · (|acc| + Σ |a·b|) + abs K` of
  `acc + Σ a·b`, with room below the accumulator's `DType.bound`. With an integer accumulator
  and `rel = abs = 0`, an exact sum in range is the result.
* `DotScaledTotal cfg`, `DotScaledAcc cfg dom sdom rel abs`: `tl.dot_scaled` has a result; an
  output element whose row of `lhs` and column of `rhs` have scaled values in `dom` and scale
  bits in `sdom` is within `rel K · (|acc| + Σ |a·b|) + abs K` of `acc + Σ a·b`.
* `AtomicAddAcc f rel abs`: a float `atomic_add` whose exact sum is in range is finite and
  within `rel · |o + v| + abs`. On NVIDIA, `atom.add.f32` flushes subnormal inputs and results.
  Two operands just below the smallest normal then sum to almost `2 · 2^emin` and give `0`, so
  `abs` must be at least `2^(emin + 1)` (`2^-125` for fp32).
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

/-- **`/` on `float32` is `div.full.f32`.** Triton 3.8.0 lowers `arith.divf` on `f32` to
`llvm.nvvm.div.full` (`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:553-580`), which LLVM selects
as `div.full.f32` (`llvm-NVPTXIntrinsics.td:1536-1537`). The PTX ISA documents it as "a
relatively fast, full-range approximation that scales operands to achieve better accuracy", at
most 2 ulp from `x/y`. For `|x/y| ≤ 2^126`, 2 ulp is at most `2^-22·|x/y|` for a normal result
and `2^-148` for a subnormal one. TVA's own measurement of it on the target is library issue
L4 (`docs/library-review/ISSUES.md`). -/
def DivFull : Prop := DivAcc P .binary32 (2 ^ 126) (2 ^ (-22 : ℤ)) (2 ^ (-148 : ℤ))

/-- **`/` on format `f` is IEEE division**, up to NaN payload. Triton 3.8.0 lowers `arith.divf`
on `f64` to `llvm.nvvm.div.rn.d` (`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:569-571`), which
is `div.rn.f64` or, after LLVM's `instcombine`, an IEEE `fdiv` (`llvm-NVPTXTargetTransformInfo.cpp:298`).
So `DivRn .binary64` holds on NVIDIA. -/
def DivRn (f : Format) : Prop :=
  ∀ x y r : FloatV f, P.div f x y r → FloatV.NaNEq (FloatV.divRn x y) r

/-- **A round-to-nearest conversion is accurate in range.** A finite value within `g`'s range
converts (with the default `.rn`) to a finite value within `|x|·u + η` of `x`, the bound
`FloatV.cvt_err` gives for one rounding. It is judged per source format and architecture: before
sm_90, `float32` goes to fp8 through a truncation to `float16`, which can exceed half an ulp but
stays within this bound. -/
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
one `cfg`, and the reviewer judges them with its input precision (`tf32` keeps 10 fraction bits
of `float32` operands) and its `maxImpreciseAcc` (how many fp8 products sm_90 may sum in the
tensor core's reduced-precision accumulator before adding them into `float32`). With an
integer accumulator and `rel = abs = 0`, the rule says that an exact sum in range is the
result; Triton 3.8.0's only integer `tl.dot` is `int8 × int8 → int32`
(`semantic.py:1483-1486`). -/
def DotAcc (cfg : DotCfg) (dom : ℚ → Prop) (rel abs : ℕ → ℚ) : Prop :=
  ∀ ⦃M N K : Nat⦄ (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc r : Tensor [M, N] cfg.acc.Val), P.dot cfg a b acc r →
    ∀ (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ) (vc : ℚ),
      (∀ k, cfg.a.val? (a.get2 i k) = some (va k)) → (∀ k, cfg.b.val? (b.get2 k n) = some (vb k)) →
      (∀ k, dom (va k) ∧ dom (vb k)) → cfg.acc.val? (acc.get2 i n) = some vc →
      |vc| + ∑ k, |va k * vb k| + (rel K * (|vc| + ∑ k, |va k * vb k|) + abs K) ≤ cfg.acc.bound →
      ∃ vr, cfg.acc.val? (r.get2 i n) = some vr ∧
        |vr - (vc + ∑ k, va k * vb k)| ≤ rel K * (|vc| + ∑ k, |va k * vb k|) + abs K

/-! ## Consequences of the division rules -/

section Div
variable {P}

/-- IEEE division meets `DivAcc` with half an ulp, over the whole range of the format. -/
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
    norm_num [Format.maxVal, Format.binary32, Format.maxSig, Format.emax, Format.m, Format.expMax, Format.bias]
  obtain ⟨hfin, herr⟩ := DivRn.divAcc Format.binary32_regular h x y r hr hx hy hy0 (hq.trans hmax)
  refine ⟨hfin, herr.trans ?_⟩
  have hu : Format.binary32.u ≤ 2 ^ (-22 : ℤ) := by
    norm_num [Format.u, Format.binary32, Format.m]
  have he : Format.binary32.eta ≤ 2 ^ (-148 : ℤ) := by
    norm_num [Format.eta, Format.binary32, Format.emin, Format.m, Format.bias]
  exact add_le_add (mul_le_mul_of_nonneg_right hu (abs_nonneg _)) he

end Div

end TVA.Rules.Num
