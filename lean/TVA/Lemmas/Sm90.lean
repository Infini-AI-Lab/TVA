import TVA.Lemmas.Consistency
import TVA.Rules.Num.Sm90

/-!
# The sm_90 facts are satisfiable together

`Platform.sm90ref` is `Platform.ref` with the tightest table: every approximate function within
one rounding's bound (`u·|y| + η`) wherever a float that accurate exists, and `exp(-∞) = exp2(-∞) = +0`
in `float32`; its conversions to nearest saturate, as sm_90's into fp8 do. Every fact of
`TVA.Rules.Num.Sm90` holds of it (`Platform.sm90ref.expF32`, …, `.cvtSatF32E4m3`, …), and so do
`DivFull`, `DivRn`, every `…Total` rule, every `CvtSat` and every `CvtAcc` (the lemmas of
`Platform.ref`). A bundle that cites any of these facts proves `Claim.Consistent` with this one
platform.
-/
namespace TVA

open TVA.Math.Float

/-- The special values of `Platform.sm90ref`: `exp` and `exp2` of `-∞` are `+0`; nothing else is
fixed. -/
def sm90Special : (fn : MathFn) → (f : Format) → FloatV f → Option (FloatV f) := fun fn f x =>
  if (fn = .exp ∨ fn = .exp2) ∧ x.pat = f.setSign true f.infMag then some (FloatV.ofPat 0) else none

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

theorem dotF16 : DotF16 Platform.sm90ref := fun _ => dot16 _ rfl
theorem dotBF16 : DotBF16 Platform.sm90ref := fun _ => dot16 _ rfl

private theorem chunks8_pos {K : ℕ} (hK : 0 < K) : 1 ≤ chunks8 K := by
  unfold chunks8; omega

private theorem dot8 (a b : DType) : DotF8 Platform.sm90ref a b := by
  intro p
  have hd : (0 : ℚ) ≤ 2 ^ (-7 : ℤ) := by positivity
  refine Platform.ref.dotAcc (g := Format.binary32) rfl _ (fun K hK => ?_) (fun K hK => ?_)
    (by simp [relDot8, chunks8]) (by simp [absDot8, chunks8])
  · rw [u32q]; unfold relDot8
    have hn := chunks8_pos hK
    have hb := one_add_mul_le_pow (a := (2 : ℚ) ^ (-7 : ℤ)) (by linarith) (chunks8 K)
    have hn' : (1 : ℚ) ≤ chunks8 K := by exact_mod_cast hn
    have h1 : (2 : ℚ) ^ (-24 : ℤ) ≤ 2 ^ (-7 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by norm_num)
    nlinarith
  · rw [eta32q]; unfold absDot8
    have hn' : (1 : ℚ) ≤ chunks8 K := by exact_mod_cast chunks8_pos hK
    have hp : (1 : ℚ) ≤ (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K := one_le_pow₀ (by linarith)
    have h1 : (2 : ℚ) ^ (-150 : ℤ) ≤ 2 ^ (-126 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by norm_num)
    have h2 : (1 : ℚ) ≤ (chunks8 K : ℚ) * (1 + 2 ^ (-7 : ℤ)) ^ chunks8 K := by nlinarith
    have h3 := zpow_pos (show (0 : ℚ) < 2 by norm_num) (-126 : ℤ)
    nlinarith

theorem dotE4m3 : DotE4m3 Platform.sm90ref := dot8 _ _
theorem dotE5m2 : DotE5m2 Platform.sm90ref := dot8 _ _
theorem dotE4m3E5m2 : DotE4m3E5m2 Platform.sm90ref := dot8 _ _
theorem dotE5m2E4m3 : DotE5m2E4m3 Platform.sm90ref := dot8 _ _

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

theorem atomicAddF32 : AtomicAddF32 Platform.sm90ref :=
  Platform.ref.atomicAddAcc Format.binary32_regular
    (by rw [u32q]; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num))
    (by rw [eta32q]; exact zpow_le_zpow_right₀ (by norm_num) (by norm_num))

end Platform.sm90ref

end TVA
