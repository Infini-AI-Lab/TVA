import Mathlib.Analysis.Complex.ExponentialBounds
import TVA.Lemmas.Consistency
import TVA.Rules.Num.Sm90

/-!
# The sm_90 facts are satisfiable together

`Platform.sm90ref` is `Platform.ref` with the tightest table: every approximate function within
one rounding's bound (`u·|y| + η`) wherever a float that accurate exists, `exp(-∞) = exp2(-∞) = +0`
in `float32`, and `exp x = +∞` in `float32` from `1453635/16384` (where no float is that accurate) and
at `+∞`; its conversions to nearest saturate, as sm_90's into fp8 do. Every fact of
`TVA.Rules.Num.Sm90` holds of it (`Platform.sm90ref.expF32`, …, `.cvtSatF32E4m3`, …, `.dotF16`,
`.dotBF16` through `DotAcc.liftAcc`, `.dotI8` through `Platform.ref.dotAcc_int`), and so do
`DivFull`, `DivRn`, every `…Total` rule, every `CvtSat` and every `CvtAcc` (the lemmas of
`Platform.ref`). A bundle that cites any of these facts proves `Claim.Consistent` with this one
platform.
-/
namespace TVA

open TVA.Math.Float

/-- The special values of `Platform.sm90ref`: `exp` and `exp2` of `-∞` are `+0`; `exp` in `float32`
is `+∞` at `+∞` and at the finite inputs from `1453635/16384` (`88.72283935546875`), where no float is
within one rounding's bound of `e^x` (`farExp`); nothing else is fixed. -/
def sm90Special : (fn : MathFn) → (f : Format) → FloatV f → Option (FloatV f) := fun fn f x =>
  if (fn = .exp ∨ fn = .exp2) ∧ x.pat = f.setSign true f.infMag then some (FloatV.ofPat 0)
  else if fn = .exp ∧ f = Format.binary32 ∧
      ((x.isFinite = true ∧ (1453635 / 16384 : ℚ) ≤ x.toRat) ∨ x.pat = 0x7f800000) then
    some (FloatV.ofPat 0x7f800000)
  else none

/-- The reference platform for the sm_90 facts. -/
noncomputable def Platform.sm90ref : Platform :=
  Platform.ref (fun _ f => (f.u : ℝ)) (fun _ f => (f.eta : ℝ)) sm90Special

namespace Platform.sm90ref

open Rules.Num

private theorem u32 : (Format.binary32.u : ℝ) = 2 ^ (-24 : ℤ) := by
  rw [Format.binary32_u]; push_cast; rfl

private theorem eta32 : (Format.binary32.eta : ℝ) = 2 ^ (-150 : ℤ) := by
  rw [Format.binary32_eta]; push_cast; rfl

/-- One rounding's bound (`u·|y| + η`) on any domain, for every approximate function with a real
reference. -/
private theorem base {fn : MathFn} {g : ℝ → ℝ} (hg : fn.ref = some g) :
    MathAcc Platform.sm90ref fn .binary32 Set.univ (Format.binary32.u : ℝ) (Format.binary32.eta : ℝ) :=
  Platform.ref.mathAcc_of_eta (rel := fun _ f => (f.u : ℝ)) (abs := fun _ f => (f.eta : ℝ)) hg
    Format.binary32_regular le_rfl le_rfl Set.univ

/-- A fact with constant `rel ≥ 2^-24` and `abs ≥ 2^-150` follows from one rounding's bound. -/
private theorem of_const {fn : MathFn} {g : ℝ → ℝ} (hg : fn.ref = some g) (dom : Set ℝ) {rel abs : ℝ}
    (hrel : (2 : ℝ) ^ (-24 : ℤ) ≤ rel) (habs : (2 : ℝ) ^ (-150 : ℤ) ≤ abs) :
    MathAcc Platform.sm90ref fn .binary32 dom rel abs := by
  refine (base hg).mono (Set.subset_univ _) fun g' hg' x _ _ => ?_
  rw [hg] at hg'
  cases hg'
  rw [u32, eta32]
  have := mul_le_mul_of_nonneg_right hrel (abs_nonneg (g x))
  linarith

theorem expF32 : ExpF32 Platform.sm90ref := by
  intro X hX
  refine of_const (g := Real.exp) rfl _ ?_ (by norm_num)
  have : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) * X := mul_nonneg (by positivity) hX
  have h : (2 : ℝ) ^ (-24 : ℤ) ≤ 2 ^ (-22 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by norm_num)
  linarith

theorem exp2F32 : Exp2F32 Platform.sm90ref :=
  of_const (g := fun x => (2 : ℝ) ^ x) rfl _ (zpow_le_zpow_right₀ (by norm_num) (by norm_num))
    (zpow_le_zpow_right₀ (by norm_num) (by norm_num))

theorem logF32 : LogF32 Platform.sm90ref :=
  of_const (g := Real.log) rfl _ (zpow_le_zpow_right₀ (by norm_num) (by norm_num))
    (zpow_le_zpow_right₀ (by norm_num) (by norm_num))

theorem log2F32 : Log2F32 Platform.sm90ref :=
  of_const (g := Real.logb 2) rfl _ (zpow_le_zpow_right₀ (by norm_num) (by norm_num))
    (zpow_le_zpow_right₀ (by norm_num) (by norm_num))

theorem sqrtF32 : SqrtF32 Platform.sm90ref :=
  of_const (g := Real.sqrt) rfl _ (zpow_le_zpow_right₀ (by norm_num) (by norm_num))
    (zpow_le_zpow_right₀ (by norm_num) (by norm_num))

theorem rsqrtF32 : RsqrtF32 Platform.sm90ref :=
  of_const (g := fun x => 1 / Real.sqrt x) rfl _ (zpow_le_zpow_right₀ (by norm_num) (by norm_num))
    (zpow_le_zpow_right₀ (by norm_num) (by norm_num))

private theorem negInf_not_finite : negInfF32.isFinite = false := by decide

private theorem special_exp : sm90Special .exp .binary32 negInfF32 = some zeroF32 := by decide

private theorem special_exp2 : sm90Special .exp2 .binary32 negInfF32 = some zeroF32 := by decide

theorem expF32NegInf : ExpF32NegInf Platform.sm90ref :=
  Platform.ref.mathAt negInf_not_finite special_exp

theorem exp2F32NegInf : Exp2F32NegInf Platform.sm90ref :=
  Platform.ref.mathAt negInf_not_finite special_exp2

/-- From `1453635/16384` up, `e^x ≥ 2^128·(1 + 2^-23)`: `128·log 2 + 2^-23 < 1453635/16384`
(`Real.log_two_lt_d9`). -/
private theorem exp_ge_of_ge {x : ℝ} (hx : ((1453635 / 16384 : ℚ) : ℝ) ≤ x) :
    (2 : ℝ) ^ 128 * (1 + 2 ^ (-23 : ℤ)) ≤ Real.exp x := by
  have hlog := Real.log_two_lt_d9
  have h1 : 128 * Real.log 2 + 2 ^ (-23 : ℤ) ≤ x := by
    have : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
    rw [this]; push_cast at hx; norm_num at hlog; linarith
  have h2 : Real.exp (128 * Real.log 2) = 2 ^ 128 := by
    have : (128 : ℝ) * Real.log 2 = ((128 : ℕ) : ℝ) * Real.log 2 := by norm_num
    rw [this, Real.exp_nat_mul, Real.exp_log (by norm_num)]
  have h3 : 1 + (2 : ℝ) ^ (-23 : ℤ) ≤ Real.exp (2 ^ (-23 : ℤ)) := by
    have := Real.add_one_le_exp ((2 : ℝ) ^ (-23 : ℤ)); linarith
  calc (2 : ℝ) ^ 128 * (1 + 2 ^ (-23 : ℤ))
      ≤ Real.exp (128 * Real.log 2) * Real.exp (2 ^ (-23 : ℤ)) := by
        rw [h2]; exact mul_le_mul_of_nonneg_left h3 (by positivity)
    _ = Real.exp (128 * Real.log 2 + 2 ^ (-23 : ℤ)) := (Real.exp_add _ _).symm
    _ ≤ Real.exp x := Real.exp_le_exp.mpr h1

/-- No finite `float32` is within one rounding's bound of `e^x` from `1453635/16384` up. -/
private theorem farExp {x : F32} (hx : (1453635 / 16384 : ℚ) ≤ x.toRat) :
    ¬ ∃ ref, MathFn.exp.ref = some ref ∧ ∃ r' : F32, r'.isFinite = true ∧
      |r'.toReal - ref x.toReal| ≤ (Format.binary32.u : ℝ) * |ref x.toReal| + (Format.binary32.eta : ℝ) := by
  rintro ⟨g, hg, r', hr', hclose⟩
  have hge : g = Real.exp := by
    have h : MathFn.exp.ref = some Real.exp := rfl
    rw [h] at hg; exact (Option.some.inj hg).symm
  subst hge
  have he := exp_ge_of_ge (x := x.toReal) (by unfold FloatV.toReal; exact_mod_cast hx)
  have hmax : |r'.toReal| ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
    have h := Format.abs_toRat_le_maxVal Format.binary32_regular hr'
    rw [Format.binary32_maxVal] at h
    unfold FloatV.toReal FloatV.toRat
    exact_mod_cast h
  rw [u32, eta32, abs_of_pos (Real.exp_pos _)] at hclose
  have h1 : Real.exp x.toReal - r'.toReal ≤ |r'.toReal - Real.exp x.toReal| := by
    rw [abs_sub_comm]; exact le_abs_self _
  have h2 := le_abs_self r'.toReal
  have hz1 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
  have hz2 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
  have hz3 : (0 : ℝ) ≤ 2 ^ (-150 : ℤ) ∧ (2 : ℝ) ^ (-150 : ℤ) ≤ 1 := ⟨by positivity, by
    exact zpow_le_one_of_nonpos₀ (by norm_num) (by norm_num)⟩
  rw [hz1] at hclose; rw [hz2] at he
  norm_num at he hmax
  linarith [h1, h2, hz3.1, hz3.2]

private theorem posInf_not_finite : posInfF32.isFinite = false := by decide

private theorem special_exp_posInf : sm90Special .exp .binary32 posInfF32 = some posInfF32 := by decide

theorem expF32Overflow : ExpF32Overflow Platform.sm90ref := by
  intro x hx r hr
  rcases hx with ⟨hfin, hT⟩ | rfl
  · have hne : ¬ ((MathFn.exp = .exp ∨ MathFn.exp = .exp2) ∧
        x.pat = Format.binary32.setSign true Format.binary32.infMag) := by
      rintro ⟨_, hp⟩
      have : x.isFinite = false := by unfold FloatV.isFinite; rw [hp]; decide
      rw [this] at hfin; exact Bool.false_ne_true hfin
    have hsp : sm90Special .exp .binary32 x = some posInfF32 := by
      unfold sm90Special
      rw [ite_eq_right_of_eq_false _ _ (eq_false hne),
        ite_eq_left_of_eq_true _ _ (eq_true ⟨rfl, rfl, Or.inl ⟨hfin, hT⟩⟩)]; rfl
    exact Platform.ref.mathAt_far hfin (farExp hT) hsp r hr
  · exact Platform.ref.mathAt posInf_not_finite special_exp_posInf r hr

/-- The platform's conversions to nearest saturate (`Platform.ref.cvtSat`), so every `CvtSat`
holds, and every `CvtRn` into a regular format (`CvtSat.cvtRn`). -/
theorem cvtSat (f g : Format) : CvtSat Platform.sm90ref f g := Platform.ref.cvtSat f g

theorem cvtRn (f : Format) {g : Format} (hg : g.Regular) : CvtRn Platform.sm90ref f g :=
  (cvtSat f g).cvtRn hg

theorem cvtF32E4m3 : CvtF32E4m3 Platform.sm90ref := cvtRn _ Format.e4m3_regular
theorem cvtF32E5m2 : CvtF32E5m2 Platform.sm90ref := cvtRn _ Format.e5m2_regular
theorem cvtF16E4m3 : CvtF16E4m3 Platform.sm90ref := cvtRn _ Format.e4m3_regular
theorem cvtF16E5m2 : CvtF16E5m2 Platform.sm90ref := cvtRn _ Format.e5m2_regular
theorem cvtBF16E4m3 : CvtBF16E4m3 Platform.sm90ref := cvtRn _ Format.e4m3_regular
theorem cvtBF16E5m2 : CvtBF16E5m2 Platform.sm90ref := cvtRn _ Format.e5m2_regular

theorem cvtSatF32E4m3 : CvtSatF32E4m3 Platform.sm90ref := cvtSat _ _
theorem cvtSatF32E5m2 : CvtSatF32E5m2 Platform.sm90ref := cvtSat _ _
theorem cvtSatF16E4m3 : CvtSatF16E4m3 Platform.sm90ref := cvtSat _ _
theorem cvtSatF16E5m2 : CvtSatF16E5m2 Platform.sm90ref := cvtSat _ _
theorem cvtSatBF16E4m3 : CvtSatBF16E4m3 Platform.sm90ref := cvtSat _ _
theorem cvtSatBF16E5m2 : CvtSatBF16E5m2 Platform.sm90ref := cvtSat _ _

private theorem u32q : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u

private theorem eta32q : Format.binary32.eta = 2 ^ (-150 : ℤ) := Format.binary32_eta

private theorem dot16 (cfg : DotCfg) (hacc : cfg.acc = .f32) :
    DotAcc Platform.sm90ref cfg (fun _ => True) relDot16 absDot16 := by
  refine Platform.ref.dotAcc (g := Format.binary32) (by rw [hacc]; rfl) _ (fun K _ => ?_) (fun K _ => ?_)
    (by unfold relDot16; positivity) (by unfold absDot16; positivity)
  · rw [u32q]; unfold relDot16
    have : (2 : ℚ) ≤ K + 2 := by have := Nat.cast_nonneg (α := ℚ) K; linarith
    have h1 : (2 : ℚ) ^ (-24 : ℤ) ≤ 2 ^ (-22 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by norm_num)
    nlinarith [zpow_pos (show (0 : ℚ) < 2 by norm_num) (-22 : ℤ)]
  · rw [eta32q]; unfold absDot16
    have : (2 : ℚ) ≤ K + 2 := by have := Nat.cast_nonneg (α := ℚ) K; linarith
    have h1 : (2 : ℚ) ^ (-150 : ℤ) ≤ 2 ^ (-126 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by norm_num)
    nlinarith [zpow_pos (show (0 : ℚ) < 2 by norm_num) (-126 : ℤ)]

private theorem relDot16_nonneg (K : ℕ) : 0 ≤ relDot16 K := by unfold relDot16; positivity

theorem dotF16 : DotF16 Platform.sm90ref := fun _ => (dot16 _ rfl).liftAcc relDot16_nonneg
theorem dotBF16 : DotBF16 Platform.sm90ref := fun _ => (dot16 _ rfl).liftAcc relDot16_nonneg

theorem dotF32Ieee : DotF32Ieee Platform.sm90ref := by
  refine Platform.ref.dotAcc (g := Format.binary32) rfl _ (fun K _ => ?_) (fun K _ => ?_)
    (by positivity) (by positivity)
  · rw [u32q]
    have : (1 : ℚ) ≤ K + 1 := by have := Nat.cast_nonneg (α := ℚ) K; linarith
    have h1 : (2 : ℚ) ^ (-24 : ℤ) ≤ 2 ^ (-23 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by norm_num)
    nlinarith [zpow_pos (show (0 : ℚ) < 2 by norm_num) (-23 : ℤ)]
  · rw [eta32q]
    have : (1 : ℚ) ≤ K + 1 := by have := Nat.cast_nonneg (α := ℚ) K; linarith
    have h1 : (2 : ℚ) ^ (-150 : ℤ) ≤ 2 ^ (-149 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by norm_num)
    nlinarith [zpow_pos (show (0 : ℚ) < 2 by norm_num) (-149 : ℤ)]

theorem dotI8 : DotI8 Platform.sm90ref := fun _ =>
  Platform.ref.dotAcc_int rfl rfl rfl _ (fun _ => le_refl 0) (fun _ => le_refl 0)

private theorem dotScaled (lhs rhs : MXFmt) : DotScaledMx Platform.sm90ref lhs rhs := by
  intro fm
  refine Platform.ref.dotScaledAcc _ _ _ (fun K _ => ?_) (fun K _ => ?_)
    (by unfold relDot16; positivity) (by unfold absDot16; positivity)
  · rw [u32q]; unfold relDot16
    have : (2 : ℚ) ≤ K + 2 := by have := Nat.cast_nonneg (α := ℚ) K; linarith
    have h1 : (2 : ℚ) ^ (-24 : ℤ) ≤ 2 ^ (-22 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by norm_num)
    nlinarith [zpow_pos (show (0 : ℚ) < 2 by norm_num) (-22 : ℤ)]
  · rw [eta32q]; unfold absDot16
    have : (2 : ℚ) ≤ K + 2 := by have := Nat.cast_nonneg (α := ℚ) K; linarith
    have h1 : (2 : ℚ) ^ (-150 : ℤ) ≤ 2 ^ (-126 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by norm_num)
    nlinarith [zpow_pos (show (0 : ℚ) < 2 by norm_num) (-126 : ℤ)]

theorem dotScaledE4m3 : DotScaledE4m3 Platform.sm90ref := dotScaled _ _
theorem dotScaledE4m3E2m1 : DotScaledE4m3E2m1 Platform.sm90ref := dotScaled _ _

theorem divFull : DivFull Platform.sm90ref := Platform.ref.divFull
theorem divRn (f : Format) : DivRn Platform.sm90ref f := Platform.ref.divRn f

/-- IEEE division (`Format.div`) of a finite value by an infinite one is the signed zero, so
`DivF32InfDivisor` holds (library issue L154). -/
theorem divF32InfDivisor : DivF32InfDivisor Platform.sm90ref := by
  intro x y hx hy r hr
  have hxn : x.isNaN = false := (FloatV.isFinite_iff.mp hx).1
  have hyn : y.isNaN = false := by
    unfold FloatV.isInf at hy; unfold FloatV.isNaN
    have hs : Format.binary32.specials = .ieee := rfl
    simp only [Format.isInf, Format.isNaN, hs, Bool.and_eq_true, beq_iff_eq] at hy ⊢
    simp [hy.2]
  have hxi : x.isInf = false := (FloatV.isFinite_iff.mp hx).2
  have hd : FloatV.divRn x y = FloatV.ofPat (Format.binary32.zeroPat (x.sign != y.sign)) := by
    unfold FloatV.divRn FloatV.sign
    unfold FloatV.isNaN at hxn hyn; unfold FloatV.isInf at hy hxi
    simp only [Format.div, hxn, hyn, hxi, hy, Bool.or_self, Bool.false_eq_true, ite_false, ite_true]
  simp only [Platform.sm90ref, Platform.ref] at hr
  rw [← hd]
  refine FloatV.NaNEq.eq hr ?_
  rw [hd]; unfold FloatV.isNaN FloatV.ofPat FloatV.pat
  cases x.sign != y.sign <;> decide

theorem atomicAddF32 : AtomicAddF32 Platform.sm90ref :=
  Platform.ref.atomicAddAcc Format.binary32_regular
    (by rw [u32q]; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num))
    (by rw [eta32q]; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num))

end Platform.sm90ref

/-- `Platform.sm90ref` (`Platform.ref`) reduces along balanced trees: `Rules.Num.SumDepth` holds
(library issue L149). -/
theorem Platform.sm90ref.sumDepth : Rules.Num.SumDepth Platform.sm90ref := fun _ => le_rfl

end TVA
