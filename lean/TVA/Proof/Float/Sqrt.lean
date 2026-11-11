import TVA.Proof.Float.Ops
import Mathlib.Analysis.Real.Sqrt

/-!
# Square roots

`Format.sqrtMag rnd a` rounds `√a` for a rational `a > 0` without computing a real number: the
quantum exponent comes from `⌊log₂ a⌋ / 2`, the integer part of `√a / 2^Q` is an integer square
root, and its fraction is compared with `1/2` by squaring. This file shows that this is the
rounding to nearest of the real `√a`:

* `sqrt_rint_err`: the rounded integer square root is within `1/2` of `√t`;
* `sqrtMag_rn_spec`: for `0 < a ≤ maxVal`, a finite magnitude within `√a·u + η` of `√a`, the bound
  of one rounding to nearest;
* `sqrt_rn_err`: the pattern operation `sqrt .rn` of a finite `x ≥ 0` (`tl.sqrt_rn`) is finite,
  nonnegative and within `√x·u + η` of `√x`.

`FloatV.sqrtRn_err` (`TVA.Proof.Float.Error`) restates the last for `FloatV`.
-/
namespace TVA.Format

variable {f : Format}

/-- The significand `sqrtMag` computes, `rint` of the integer square root, is within `1/2` of
`√t`. -/
theorem sqrt_rint_err {t : ℚ} (ht : 0 ≤ t) :
    |((rint .rn (Nat.sqrt (Rat.floor t).toNat)
        (cmpq t (((Nat.sqrt (Rat.floor t).toNat : ℕ) + 1 / 2) * ((Nat.sqrt (Rat.floor t).toNat : ℕ) + 1 / 2)))
        : ℕ) : ℝ) - Real.sqrt t| ≤ 1 / 2 := by
  rw [floor_toNat]
  set k := Nat.sqrt ⌊t⌋₊ with hk
  have hk1 : ((k : ℚ)) ^ 2 ≤ t := by
    have h1 : k ^ 2 ≤ ⌊t⌋₊ := Nat.sqrt_le' _
    have h2 : (⌊t⌋₊ : ℚ) ≤ t := Nat.floor_le ht
    calc ((k : ℚ)) ^ 2 = ((k ^ 2 : ℕ) : ℚ) := by push_cast; ring
      _ ≤ (⌊t⌋₊ : ℚ) := by exact_mod_cast h1
      _ ≤ t := h2
  have hk2 : t < ((k : ℚ) + 1) ^ 2 := by
    have h1 : ⌊t⌋₊ < (k + 1) ^ 2 := Nat.lt_succ_sqrt' _
    have h2 : t < (⌊t⌋₊ : ℚ) + 1 := Nat.lt_floor_add_one t
    have h3 : (⌊t⌋₊ : ℚ) + 1 ≤ ((k + 1) ^ 2 : ℕ) := by exact_mod_cast h1
    calc t < (⌊t⌋₊ : ℚ) + 1 := h2
      _ ≤ ((k + 1) ^ 2 : ℕ) := h3
      _ = ((k : ℚ) + 1) ^ 2 := by push_cast; ring
  have htR : (0 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have hk1R : (k : ℝ) ≤ Real.sqrt t := by
    apply Real.le_sqrt_of_sq_le
    have : ((k : ℚ) ^ 2 : ℚ) ≤ t := hk1
    exact_mod_cast this
  have hk2R : Real.sqrt t < (k : ℝ) + 1 := by
    rw [Real.sqrt_lt' (by positivity)]
    have : t < ((k : ℚ) + 1) ^ 2 := hk2
    exact_mod_cast this
  have hpos : (0 : ℝ) < (k : ℝ) + 1 / 2 := by positivity
  unfold cmpq
  rw [abs_le]
  split_ifs with h1 h2
  · -- below the midpoint: `k`
    have : Real.sqrt t < (k : ℝ) + 1 / 2 := by
      rw [Real.sqrt_lt' hpos]
      have h1' := (Rat.cast_lt (K := ℝ)).mpr h1
      push_cast at h1'
      nlinarith
    simp only [rint]; constructor <;> linarith
  · -- the midpoint: `k` or `k + 1`, half away
    have : Real.sqrt t = (k : ℝ) + 1 / 2 := by
      have e : (t : ℝ) = ((k : ℝ) + 1 / 2) ^ 2 := by
        have : (t : ℝ) = (((k : ℚ) + 1 / 2) * ((k : ℚ) + 1 / 2) : ℚ) := by exact_mod_cast h2
        rw [this]; push_cast; ring
      rw [e, Real.sqrt_sq hpos.le]
    simp only [rint]
    split_ifs <;> push_cast <;> constructor <;> linarith
  · -- above the midpoint: `k + 1`
    have hgt : ((k : ℚ) + 1 / 2) * ((k : ℚ) + 1 / 2) < t := lt_of_le_of_ne (not_lt.mp h1) (Ne.symm h2)
    have : (k : ℝ) + 1 / 2 < Real.sqrt t := by
      rw [Real.lt_sqrt hpos.le]
      have h1' := (Rat.cast_lt (K := ℝ)).mpr hgt
      push_cast at h1'
      nlinarith
    simp only [rint]; push_cast; constructor <;> linarith

/-- The largest value is at least `1` (at least `2^(topExp − bias)`). -/
theorem one_le_maxVal (hf : f.Regular) : 1 ≤ f.maxVal := by
  have hE := topExp_pos hf
  have h := magVal_ge_of_exp (f := f) (M := f.maxMag) (by rw [maxMag_div hf]; exact hE)
  rw [magVal_maxMag hf, maxMag_div hf] at h
  have hb : 0 ≤ ((f.topExp : ℕ) : ℤ) - f.bias := by
    have h4 : 4 ≤ 2 ^ f.e := two_pow_e_ge hf
    have h2 : (2 : ℤ) ^ f.e = 2 * 2 ^ (f.e - 1) := by
      rw [← pow_succ']; congr 1; have := hf.1; omega
    have h1 : (1 : ℤ) ≤ 2 ^ (f.e - 1) := one_le_pow₀ (by norm_num)
    have ht : 2 ^ f.e - 2 ≤ f.topExp := by unfold topExp; split <;> omega
    have ht' : ((2 ^ f.e - 2 : ℕ) : ℤ) = (2 : ℤ) ^ f.e - 2 := by
      push_cast [show 2 ≤ 2 ^ f.e by omega]; ring
    have : ((2 ^ f.e - 2 : ℕ) : ℤ) ≤ (f.topExp : ℤ) := by exact_mod_cast ht
    unfold bias
    linarith
  calc (1 : ℚ) = 2 ^ (0 : ℤ) := by simp
    _ ≤ 2 ^ (((f.topExp : ℕ) : ℤ) - f.bias) := zpow_le_zpow_right₀ (by norm_num) hb
    _ ≤ f.maxVal := h

/-- **`sqrtMag` rounds `√a` to nearest**: for `0 < a ≤ maxVal` the magnitude is finite (at most
`maxMag`) and its value is within `√a·u + η` of `√a`, the bound of one rounding to nearest (at least
half an ulp, and up to about a whole one: not the half-ulp bound itself). -/
theorem sqrtMag_rn_spec (hf : f.Regular) {a : ℚ} (ha : 0 < a) (hmax : a ≤ f.maxVal) :
    f.sqrtMag .rn a ≤ f.maxMag ∧
      |(f.magVal (f.sqrtMag .rn a) : ℝ) - Real.sqrt a| ≤ Real.sqrt a * f.u + f.eta := by
  obtain ⟨hL1, hL2⟩ := qlog2_spec ha
  set L := qlog2 a with hL
  set j : ℤ := L / 2 with hj
  have hj1 : 2 * j ≤ L := by omega
  have hj2 : L ≤ 2 * j + 1 := by omega
  set Q : ℤ := max j f.emin - f.m with hQ
  set t : ℚ := a * pow2 (-2 * Q) with ht
  have ht' : t = a * (2 : ℚ) ^ (-2 * Q) := by rw [ht, pow2_eq]
  have ht0 : 0 ≤ t := by rw [ht']; positivity
  set k := Nat.sqrt (Rat.floor t).toNat with hk
  set N := rint .rn k (cmpq t (((k : ℚ) + 1 / 2) * ((k : ℚ) + 1 / 2))) with hN
  have hsm : f.sqrtMag .rn a = f.encode .rn false N Q := rfl
  have herr : |(N : ℝ) - Real.sqrt t| ≤ 1 / 2 := sqrt_rint_err ht0
  -- the real square roots
  set σ := Real.sqrt a with hσ
  set τ := Real.sqrt t with hτ
  have h2 : (0 : ℝ) < 2 := by norm_num
  have haR : (0 : ℝ) < (a : ℝ) := by exact_mod_cast ha
  have hτσ : τ = σ * (2 : ℝ) ^ (-Q) := by
    have e1 : (t : ℝ) = (a : ℝ) * ((2 : ℝ) ^ (-Q)) ^ 2 := by
      rw [ht']; push_cast
      rw [← zpow_natCast, ← zpow_mul]; ring_nf
    rw [hτ, e1, Real.sqrt_mul haR.le, Real.sqrt_sq (by positivity)]
  have hστ : σ = τ * (2 : ℝ) ^ Q := by
    rw [hτσ, mul_assoc, ← zpow_add₀ h2.ne']; simp
  have hσlo : (2 : ℝ) ^ j ≤ σ := by
    apply Real.le_sqrt_of_sq_le
    have : (2 : ℚ) ^ (2 * j) ≤ a := (zpow_le_zpow_right₀ (by norm_num) hj1).trans hL1
    have : ((2 : ℚ) ^ (2 * j) : ℝ) ≤ (a : ℝ) := by exact_mod_cast this
    push_cast at this
    rw [← zpow_natCast, ← zpow_mul, show j * ((2 : ℕ) : ℤ) = 2 * j by push_cast; ring]; exact this
  have hσhi : σ < (2 : ℝ) ^ (j + 1) := by
    rw [hσ, Real.sqrt_lt' (by positivity)]
    have : a < (2 : ℚ) ^ (2 * j + 2) := hL2.trans_le (zpow_le_zpow_right₀ (by norm_num) (by omega))
    have : (a : ℝ) < ((2 : ℚ) ^ (2 * j + 2) : ℝ) := by exact_mod_cast this
    push_cast at this
    rw [← zpow_natCast, ← zpow_mul]
    convert this using 2; push_cast; ring
  have hmax1 := one_le_maxVal hf
  have hσmax : σ ≤ (f.maxVal : ℝ) := by
    have hm1 : (1 : ℝ) ≤ (f.maxVal : ℝ) := by exact_mod_cast hmax1
    have hle : (a : ℝ) ≤ (f.maxVal : ℝ) ^ 2 := by
      have : (a : ℝ) ≤ (f.maxVal : ℝ) := by exact_mod_cast hmax
      nlinarith
    calc σ ≤ Real.sqrt ((f.maxVal : ℝ) ^ 2) := Real.sqrt_le_sqrt hle
      _ = (f.maxVal : ℝ) := Real.sqrt_sq (by linarith)
  have hjQ : j ≤ Q + f.m := by rw [hQ]; omega
  have hτ0 : 0 ≤ τ := Real.sqrt_nonneg _
  -- `N ≤ 2^(m+1)`
  have hτhi : τ < (2 : ℝ) ^ ((f.m : ℤ) + 1) := by
    rw [hτσ]
    calc σ * (2 : ℝ) ^ (-Q) < (2 : ℝ) ^ (j + 1) * (2 : ℝ) ^ (-Q) :=
          mul_lt_mul_of_pos_right hσhi (by positivity)
      _ = (2 : ℝ) ^ (j + 1 - Q) := by rw [← zpow_add₀ h2.ne']; ring_nf
      _ ≤ (2 : ℝ) ^ ((f.m : ℤ) + 1) := zpow_le_zpow_right₀ (by norm_num) (by omega)
  have hN1 : N ≤ 2 ^ (f.m + 1) := by
    have h1 : (N : ℝ) < ((2 ^ (f.m + 1) : ℕ) : ℝ) + 1 := by
      have : (2 : ℝ) ^ ((f.m : ℤ) + 1) = ((2 ^ (f.m + 1) : ℕ) : ℝ) := by
        rw [show ((f.m : ℤ) + 1) = ((f.m + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]; push_cast; rfl
      rw [abs_le] at herr; linarith
    have : N < 2 ^ (f.m + 1) + 1 := by exact_mod_cast h1
    omega
  have hQ0 : f.emin - f.m ≤ Q := by rw [hQ]; have := le_max_right j f.emin; linarith
  have hsub : N < 2 ^ f.m → Q = f.emin - f.m := by
    intro hlt
    by_contra hne
    have hjm : f.emin ≤ j := by
      by_contra h'; exact hne (by rw [hQ, max_eq_right (not_le.mp h').le])
    have hQj : Q = j - f.m := by rw [hQ, max_eq_left hjm]
    have hτlo : (2 : ℝ) ^ (f.m : ℤ) ≤ τ := by
      rw [hτσ]
      calc (2 : ℝ) ^ (f.m : ℤ) = (2 : ℝ) ^ j * (2 : ℝ) ^ (-Q) := by
            rw [← zpow_add₀ h2.ne', hQj]; ring_nf
        _ ≤ σ * (2 : ℝ) ^ (-Q) := mul_le_mul_of_nonneg_right hσlo (by positivity)
    have h1 : ((2 ^ f.m : ℕ) : ℝ) - 1 < (N : ℝ) := by
      have : (2 : ℝ) ^ (f.m : ℤ) = ((2 ^ f.m : ℕ) : ℝ) := by rw [zpow_natCast]; push_cast; rfl
      rw [abs_le] at herr; linarith
    have : 2 ^ f.m < N + 1 := by
      have : ((2 ^ f.m : ℕ) : ℝ) < (N : ℝ) + 1 := by linarith
      exact_mod_cast this
    omega
  -- no overflow: `N · 2^Q ≤ maxVal`
  set T : ℤ := (f.topExp : ℤ) - f.bias - f.m with hT
  have hQT : Q ≤ T := by
    have hjT : j < (f.topExp : ℤ) + 1 - f.bias := by
      have hl : (2 : ℝ) ^ j < (2 : ℝ) ^ ((f.topExp : ℤ) + 1 - f.bias) := by
        have := maxVal_lt hf
        have : ((f.maxVal : ℚ) : ℝ) < (((2 : ℚ) ^ ((f.topExp : ℤ) + 1 - f.bias) : ℚ) : ℝ) := by
          exact_mod_cast this
        push_cast at this
        linarith
      exact (zpow_lt_zpow_iff_right₀ (by norm_num)).mp hl
    have hE : f.emin ≤ (f.topExp : ℤ) - f.bias := by
      rw [emin_eq]; have := topExp_pos hf; omega
    rw [hQ, hT]; omega
  set K : ℕ := f.topSig * 2 ^ (T - Q).toNat with hK
  have hKQ : (K : ℚ) * (2 : ℚ) ^ Q = f.maxVal := by
    rw [hK, maxVal_eq]; push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), mul_assoc, ← zpow_add₀ (by norm_num)]
    congr 2; ring
  have hτK : τ ≤ (K : ℝ) := by
    have hKQR : (K : ℝ) * (2 : ℝ) ^ Q = (f.maxVal : ℝ) := by
      have := congrArg (fun x : ℚ => (x : ℝ)) hKQ
      push_cast at this; exact this
    rw [hτσ]
    calc σ * (2 : ℝ) ^ (-Q) ≤ (f.maxVal : ℝ) * (2 : ℝ) ^ (-Q) := mul_le_mul_of_nonneg_right hσmax (by positivity)
      _ = (K : ℝ) := by rw [← hKQR, mul_assoc, ← zpow_add₀ h2.ne']; simp
  have hNK : N ≤ K := by
    have : (N : ℝ) < (K : ℝ) + 1 := by rw [abs_le] at herr; linarith
    have : N < K + 1 := by exact_mod_cast this
    omega
  have hNmax : (N : ℚ) * (2 : ℚ) ^ Q ≤ f.maxVal := by
    rw [← hKQ]; exact mul_le_mul_of_nonneg_right (by exact_mod_cast hNK) (by positivity)
  have henc : f.encode .rn false N Q = f.pack N Q := by
    unfold encode; rw [pow2_eq, ite_eq_left hNmax]
  refine ⟨by rw [hsm, henc]; exact pack_le_maxMag hf hN1 hQ0 hsub hNmax, ?_⟩
  rw [hsm, henc, magVal_pack hN1 hQ0 hsub]
  -- the error: `2^Q · |N − τ| ≤ 2^Q / 2 ≤ σ·u + η`
  have hval : (((N : ℚ) * (2 : ℚ) ^ Q : ℚ) : ℝ) - σ = ((N : ℝ) - τ) * (2 : ℝ) ^ Q := by
    rw [hστ]; push_cast; ring
  have hpQ : (0 : ℝ) < (2 : ℝ) ^ Q := by positivity
  rw [hval, abs_mul, abs_of_pos hpQ]
  have hhalf : (2 : ℝ) ^ Q / 2 ≤ σ * (f.u : ℝ) + (f.eta : ℝ) := by
    have hu0 : (0 : ℝ) ≤ (f.u : ℝ) := by exact_mod_cast u_nonneg
    have he0 : (0 : ℝ) ≤ (f.eta : ℝ) := by exact_mod_cast eta_nonneg
    by_cases hjm : f.emin ≤ j
    · have hQj : Q = j - f.m := by rw [hQ, max_eq_left hjm]
      have hq := half_quantum_eq (f := f) Q
      have hq' : (2 : ℝ) ^ Q / 2 = (f.u : ℝ) * (2 : ℝ) ^ ((f.m : ℤ) + Q) := by
        have := congrArg (fun x : ℚ => (x : ℝ)) hq; push_cast at this; exact this
      rw [hq', show (f.m : ℤ) + Q = j by rw [hQj]; ring]
      nlinarith
    · have hQe : Q = f.emin - f.m := by rw [hQ, max_eq_right (not_le.mp hjm).le]
      have he : (f.eta : ℝ) = (2 : ℝ) ^ Q / 2 := by
        rw [eta_eq]; push_cast
        rw [hQe, eq_div_iff (by norm_num), ← zpow_add_one₀ (by norm_num)]; ring_nf
      have : 0 ≤ σ * (f.u : ℝ) := mul_nonneg (Real.sqrt_nonneg _) hu0
      linarith
  calc |(N : ℝ) - τ| * (2 : ℝ) ^ Q ≤ 1 / 2 * (2 : ℝ) ^ Q := mul_le_mul_of_nonneg_right herr hpQ.le
    _ = (2 : ℝ) ^ Q / 2 := by ring
    _ ≤ σ * (f.u : ℝ) + (f.eta : ℝ) := hhalf

/-- **`√x` to nearest** (`sqrt .rn`) of a finite pattern with a nonnegative value: a finite pattern,
with a nonnegative value, within `√x·u + η` of `√x` (`sqrt(−0) = −0`, value `0`). -/
theorem sqrt_rn_err (hf : f.Regular) {x : ℕ} (hx : f.isFinite x = true) (h0 : 0 ≤ f.toRat x) :
    f.sqrt .rn x < 2 ^ f.width ∧ f.isFinite (f.sqrt .rn x) = true ∧ 0 ≤ f.toRat (f.sqrt .rn x) ∧
      |(f.toRat (f.sqrt .rn x) : ℝ) - Real.sqrt (f.toRat x)| ≤ Real.sqrt (f.toRat x) * f.u + f.eta := by
  obtain ⟨hxn, hxi⟩ := (isFinite_iff x).mp hx
  have hu0 : (0 : ℝ) ≤ (f.u : ℝ) := by exact_mod_cast u_nonneg
  have he0 : (0 : ℝ) ≤ (f.eta : ℝ) := by exact_mod_cast eta_nonneg
  by_cases hz : f.isZero x = true
  · have hv : f.toRat x = 0 := toRat_of_isZero hz
    have hs : f.sqrt .rn x = f.zeroPat (f.sign x) := by unfold sqrt; simp [hxn, hz]
    have hzv : f.toRat (f.zeroPat (f.sign x)) = 0 := by
      unfold zeroPat; rw [toRat_setSign]; unfold mag; simp [magVal_zero]
    rw [hs, hzv, hv]
    refine ⟨setSign_lt _ _, by unfold zeroPat; rw [isFinite_setSign]; exact isFinite_zero hf, le_rfl, ?_⟩
    simp only [Rat.cast_zero, Real.sqrt_zero, sub_zero, abs_zero, zero_mul, zero_add]
    exact he0
  · have hz' : f.isZero x = false := by simpa using hz
    have hpos : 0 < f.toRat x := lt_of_le_of_ne h0 (fun h => hz (toRat_eq_zero_iff.mp h.symm))
    have hsx : f.sign x = false := by
      by_contra hs
      have hs' : f.sign x = true := by simpa using hs
      have hm : 0 < f.mag x := by
        rcases Nat.eq_zero_or_pos (f.mag x) with h | h
        · exfalso; apply hz; unfold isZero; simp [h]
        · exact h
      have : f.toRat x < 0 := by
        rw [toRat_eq, ite_eq_left hs']; exact neg_neg_of_pos (magVal_pos hm)
      linarith
    have hs : f.sqrt .rn x = f.sqrtMag .rn (f.toRat x) := by unfold sqrt; simp [hxn, hz', hsx, hxi]
    have hmax : f.toRat x ≤ f.maxVal := (le_abs_self _).trans (abs_toRat_le_maxVal hf hx)
    obtain ⟨hle, herr⟩ := sqrtMag_rn_spec hf hpos hmax
    have hlt : f.sqrtMag .rn (f.toRat x) < f.signBit := hle.trans_lt (maxMag_lt hf)
    have hmag : f.mag (f.sqrtMag .rn (f.toRat x)) = f.sqrtMag .rn (f.toRat x) := mag_of_lt hlt
    have hval : f.toRat (f.sqrtMag .rn (f.toRat x)) = f.magVal (f.sqrtMag .rn (f.toRat x)) := by
      have := toRat_mag (f := f) (f.sqrtMag .rn (f.toRat x)); rwa [hmag] at this
    rw [hs, hval]
    exact ⟨hlt.trans signBit_lt, isFinite_of_mag_le hf (by rw [hmag]; exact hle), magVal_nonneg _, herr⟩

end TVA.Format
