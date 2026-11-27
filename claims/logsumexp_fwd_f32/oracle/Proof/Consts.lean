import Claim
import TVA.Proof

/-!
# Constants and facts about `exp` and `log` in the reals

`uR = 2^-24` and `etaR = 2^-150` (one `float32` rounding), the bound `|s|·e^s ≤ 16·e^s + 2^-23`
for `s ≤ 1/2`, the log-sum-exp's Lipschitz bound (`lse_lip`: moving each exponent by at most `c`
moves the log-sum-exp by at most `c`), and the log of a relatively close positive number
(`abs_log_sub_le`).
-/

namespace Claims.LogsumexpFwd

open TVA

/-- One `float32` rounding, relative: `2^-24`. -/
noncomputable def uR : ℝ := 2 ^ (-24 : ℤ)

/-- One `float32` rounding, absolute: `2^-150`. -/
noncomputable def etaR : ℝ := 2 ^ (-150 : ℤ)

lemma uR_pos : 0 < uR := by unfold uR; positivity
lemma etaR_pos : 0 < etaR := by unfold etaR; positivity
lemma uR_le : uR ≤ 1 / 2 ^ 20 := by unfold uR; norm_num
lemma etaR_le : etaR ≤ uR / 2 ^ 120 := by unfold uR etaR; norm_num

lemma b32_reg : Format.binary32.Regular := Format.binary32_regular
lemma b32_ieee : Format.binary32.specials = .ieee := rfl
lemma b32_nf : Format.binary32.specials ≠ .finite := by decide

/-- `float32`'s rounding constants as reals. -/
lemma b32_u_cast : ((Format.binary32.u : ℚ) : ℝ) = uR := by
  unfold uR; rw [Format.binary32_u]; norm_num

lemma b32_eta_cast : ((Format.binary32.eta : ℚ) : ℝ) = etaR := by
  unfold etaR; rw [Format.binary32_eta]; norm_num

lemma b32_maxVal_cast :
    ((Format.binary32.maxVal : ℚ) : ℝ) = ((2 : ℝ) ^ 24 - 1) * 2 ^ 104 := by
  rw [Format.binary32_maxVal]; norm_num

lemma b32_maxVal_ge : (2 : ℝ) ^ 127 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
  rw [b32_maxVal_cast]; norm_num

lemma b32_maxVal_le : ((Format.binary32.maxVal : ℚ) : ℝ) ≤ 2 ^ 128 := by
  rw [b32_maxVal_cast]; norm_num

/-! ## `exp` -/

/-- `w ≤ e^w`. -/
lemma le_exp_self (w : ℝ) : w ≤ Real.exp w := by
  have := Real.add_one_le_exp w; linarith

/-- `w·e^{-w} ≤ 1` for every `w`. -/
lemma mul_exp_neg_le_one (w : ℝ) : w * Real.exp (-w) ≤ 1 := by
  have h1 := le_exp_self w
  have h2 := (Real.exp_pos (-w)).le
  have h3 : Real.exp w * Real.exp (-w) = 1 := by rw [← Real.exp_add]; simp
  nlinarith

/-- `e^-16 ≤ 2^-23`. -/
lemma exp_neg16_le : Real.exp (-(16 : ℝ)) ≤ (2 : ℝ) ^ (-23 : ℤ) := by
  have h2 : (2.718 : ℝ) ≤ Real.exp 1 := le_of_lt (lt_trans (by norm_num) Real.exp_one_gt_d9)
  have h3 : (2.718 : ℝ) ^ 16 ≤ (Real.exp 1) ^ 16 := pow_le_pow_left₀ (by norm_num) h2 16
  have h4 : (2 : ℝ) ^ 23 ≤ (2.718 : ℝ) ^ 16 := by norm_num
  have h1 : Real.exp (16 : ℝ) = (Real.exp 1) ^ 16 := by
    rw [← Real.exp_nat_mul]; norm_num
  have h5 : (2 : ℝ) ^ 23 ≤ Real.exp (16 : ℝ) := by rw [h1]; linarith
  have h8 : Real.exp (-(16 : ℝ)) = (Real.exp (16 : ℝ))⁻¹ := by rw [← Real.exp_neg]
  have h7 : (2 : ℝ) ^ (-23 : ℤ) = ((2 : ℝ) ^ 23)⁻¹ := by norm_num
  rw [h8, h7]
  exact (inv_le_inv₀ (Real.exp_pos 16) (by norm_num)).mpr h5

/-- `w·e^{-w} ≤ 16·e^{-w} + 2^-23` for `w ≥ 0`. -/
lemma mul_exp_neg_le16 {w : ℝ} (_hw : 0 ≤ w) :
    w * Real.exp (-w) ≤ 16 * Real.exp (-w) + (2 : ℝ) ^ (-23 : ℤ) := by
  rcases le_or_gt w 16 with h | h
  · have h1 : w * Real.exp (-w) ≤ 16 * Real.exp (-w) :=
      mul_le_mul_of_nonneg_right h (Real.exp_pos (-w)).le
    have h2 : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) := by positivity
    linarith
  · have h3 : Real.exp (-w) = Real.exp (-(w - 16)) * Real.exp (-16) := by
      rw [← Real.exp_add]; ring_nf
    have h4 := mul_exp_neg_le_one (w - 16)
    have h9 : (w - 16) * Real.exp (-w) ≤ Real.exp (-16) := by
      rw [h3]
      have heq : (w - 16) * (Real.exp (-(w - 16)) * Real.exp (-16))
          = ((w - 16) * Real.exp (-(w - 16))) * Real.exp (-16) := by ring
      rw [heq]
      calc ((w - 16) * Real.exp (-(w - 16))) * Real.exp (-16) ≤ 1 * Real.exp (-16) :=
            mul_le_mul_of_nonneg_right h4 (Real.exp_pos (-16)).le
        _ = Real.exp (-16) := by ring
    have h10 := exp_neg16_le
    have hsplit : w * Real.exp (-w) = 16 * Real.exp (-w) + (w - 16) * Real.exp (-w) := by ring
    linarith

/-- `|s|·e^s ≤ 16·e^s + 2^-23` for `s ≤ 1/2`. -/
lemma abs_mul_exp_le {s : ℝ} (hs : s ≤ 1 / 2) :
    |s| * Real.exp s ≤ 16 * Real.exp s + (2 : ℝ) ^ (-23 : ℤ) := by
  have h0 : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) := by positivity
  rcases le_or_gt 0 s with h | h
  · rw [abs_of_nonneg h]
    have : s * Real.exp s ≤ 16 * Real.exp s :=
      mul_le_mul_of_nonneg_right (by linarith) (Real.exp_pos s).le
    linarith
  · rw [abs_of_neg h]
    have := mul_exp_neg_le16 (w := -s) (by linarith)
    rwa [neg_neg] at this

/-- `e^{1/2} ≤ 2`. -/
lemma exp_half_le : Real.exp (1 / 2) ≤ 2 := by
  have h := Real.exp_bound' (x := 1 / 2) (by norm_num) (by norm_num) (n := 3) (by norm_num)
  norm_num [Finset.sum_range_succ, Nat.factorial] at h
  linarith

/-- `1/2 ≤ e^{-1/2}`. -/
lemma half_le_exp_neg_half : (1 / 2 : ℝ) ≤ Real.exp (-(1 / 2)) := by
  have h := Real.add_one_le_exp (-(1 / 2 : ℝ))
  linarith

/-! ## The log-sum-exp -/

/-- **Moving every exponent by at most `c` moves the log-sum-exp by at most `c`.** -/
lemma lse_lip {S : Finset Nat} (hS : S.Nonempty) {f g : Nat → ℝ} {c : ℝ}
    (h : ∀ j ∈ S, |f j - g j| ≤ c) :
    |Real.log (∑ j ∈ S, Real.exp (f j)) - Real.log (∑ j ∈ S, Real.exp (g j))| ≤ c := by
  have hpos : ∀ h' : Nat → ℝ, 0 < ∑ j ∈ S, Real.exp (h' j) :=
    fun h' => Finset.sum_pos (fun j _ => Real.exp_pos _) hS
  have hle : ∀ (a b : Nat → ℝ), (∀ j ∈ S, a j ≤ b j + c) →
      Real.log (∑ j ∈ S, Real.exp (a j)) ≤ Real.log (∑ j ∈ S, Real.exp (b j)) + c := by
    intro a b hab
    have h1 : ∑ j ∈ S, Real.exp (a j) ≤ Real.exp c * ∑ j ∈ S, Real.exp (b j) := by
      rw [Finset.mul_sum]
      refine Finset.sum_le_sum fun j hj => ?_
      rw [← Real.exp_add]
      exact Real.exp_le_exp.mpr (by linarith [hab j hj])
    have h2 := Real.log_le_log (hpos a) h1
    rw [Real.log_mul (Real.exp_pos c).ne' (hpos b).ne', Real.log_exp] at h2
    linarith
  rw [abs_le]
  constructor
  · have := hle g f (fun j hj => by have := (abs_le.mp (h j hj)).1; linarith)
    linarith
  · have := hle f g (fun j hj => by have := (abs_le.mp (h j hj)).2; linarith)
    linarith

/-- The log of a positive number within `ε·A` of `A > 0` (`ε ≤ 1/2`) is within `2ε` of `log A`. -/
lemma abs_log_sub_le {S A ε : ℝ} (hA : 0 < A) (hε : ε ≤ 1 / 2) (h : |S - A| ≤ ε * A) :
    0 < S ∧ |Real.log S - Real.log A| ≤ ε / (1 - ε) := by
  have h1 := (abs_le.mp h).1
  have h2 := (abs_le.mp h).2
  have hε0 : 0 ≤ ε := by
    by_contra hn
    push Not at hn
    have : ε * A < 0 := mul_neg_of_neg_of_pos hn hA
    linarith [abs_nonneg (S - A)]
  have hSpos : 0 < S := by nlinarith
  refine ⟨hSpos, ?_⟩
  have h1ε : 0 < 1 - ε := by linarith
  rw [abs_le]
  constructor
  · -- log S ≥ log A − ε/(1−ε): S ≥ (1−ε)A, log(1−ε) ≥ −ε/(1−ε)
    have hS : (1 - ε) * A ≤ S := by linarith
    have hl := Real.log_le_log (by positivity) hS
    rw [Real.log_mul h1ε.ne' hA.ne'] at hl
    have hlog : -(ε / (1 - ε)) ≤ Real.log (1 - ε) := by
      have := Real.one_sub_inv_le_log_of_pos h1ε
      have e : 1 - (1 - ε)⁻¹ = -(ε / (1 - ε)) := by field_simp; ring
      linarith
    linarith
  · have hS : S ≤ (1 + ε) * A := by linarith
    have hl := Real.log_le_log hSpos hS
    rw [Real.log_mul (by linarith) hA.ne'] at hl
    have hlog : Real.log (1 + ε) ≤ ε := by
      have := Real.log_le_sub_one_of_pos (show (0 : ℝ) < 1 + ε by linarith)
      linarith
    have : ε ≤ ε / (1 - ε) := by
      rw [le_div_iff₀ h1ε]; nlinarith
    linarith

end Claims.LogsumexpFwd
