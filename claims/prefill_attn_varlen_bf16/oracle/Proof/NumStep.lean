import Proof.NumBase

section
/-!
# Bounds on the invariant's state

`RowInv.bounds`: the sum `l` of a row that has run is between `1/2` and `2^18`; the first trip's state satisfies the invariant.
-/

namespace Claims.PrefillAttnVarlen

namespace Num

open Finset

/-- `exp x ≤ 1 + 2x` on `[0, 1/2]`. -/
private theorem exp_le_lin {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1 / 2) : Real.exp x ≤ 1 + 2 * x := by
  rcases h0.eq_or_lt with h | h
  · subst h; simp
  · have h3 := Real.exp_bound_div_one_sub_of_interval' h (by linarith)
    have h2 : 1 / (1 - x) ≤ 1 + 2 * x := by
      rw [div_le_iff₀ (by linarith)]; nlinarith
    linarith

/-- The growth factor of a small exponent: `e^ω ≤ 1.1` and `e^ω (1 + B) − 1 ≤ 0.12`. -/
private theorem growth {ω B : ℝ} (h0 : 0 ≤ ω) (h1 : ω ≤ 1 / 20) (hB0 : 0 ≤ B) (hB : B ≤ 1 / 70) :
    Real.exp ω ≤ 11 / 10 ∧ Real.exp ω * (1 + B) - 1 ≤ 3 / 25 := by
  have h := exp_le_lin h0 (by linarith)
  refine ⟨by linarith, ?_⟩
  nlinarith [mul_nonneg (sub_nonneg.mpr (show Real.exp ω ≤ 11 / 10 by linarith)) (show (0:ℝ) ≤ 1 + B by linarith)]

/-- The running maximum of a row that has run is within `Y`. -/
private theorem m_bounds {L t : ℕ} {sc sh : ℕ → ℝ} {δ Y m l : ℝ} (inv : RowInv L t sc sh δ Y m l)
    (hY : ∀ j, j ≤ L → |sc j| + δ ≤ Y) (ht0 : 0 < t) : -Y ≤ m ∧ m ≤ Y := by
  obtain ⟨j, hj, hjm⟩ := inv.attain ht0
  have hjL : j ≤ L := by
    unfold keysN at hj; simp only [mem_range] at hj; omega
  have h1 := abs_le.mp (inv.close_sh j hj)
  have h2 := hY j hjL
  rw [← hjm]
  constructor <;> linarith [le_abs_self (sc j), neg_abs_le (sc j)]

/-- The exact sum at the maximum is between `1` and `2^17`. -/
private theorem Lam_bounds {L t : ℕ} {sc sh : ℕ → ℝ} {δ Y m l : ℝ} (inv : RowInv L t sc sh δ Y m l)
    (ht0 : 0 < t) (ht : t ≤ 1024) : 1 ≤ Lam sh (keysN L t) m ∧ Lam sh (keysN L t) m ≤ 2 ^ 17 := by
  obtain ⟨j, hj, hjm⟩ := inv.attain ht0
  have hle : ∀ i ∈ keysN L t, Real.exp (sh i - m) ≤ 1 := fun i hi =>
    Real.exp_le_one_iff.mpr (by linarith [inv.le_m i hi])
  constructor
  · calc (1 : ℝ) = Real.exp (sh j - m) := by rw [hjm]; simp
      _ ≤ Lam sh (keysN L t) m :=
        Finset.single_le_sum (f := fun i => Real.exp (sh i - m)) (fun i _ => (Real.exp_pos _).le) hj
  · have hc : (keysN L t).card ≤ 128 * 1024 := by
      unfold keysN; rw [card_range]; omega
    have hc' : ((keysN L t).card : ℝ) ≤ 128 * 1024 := by exact_mod_cast hc
    calc Lam sh (keysN L t) m ≤ ∑ _i ∈ keysN L t, (1 : ℝ) := sum_le_sum hle
      _ = (keysN L t).card := by simp
      _ ≤ 2 ^ 17 := by linarith

private theorem omL_bounds {t : ℕ} {Y m : ℝ} (h2 : m ≤ Y) (hY5 : Y ≤ 5000)
    (ht : t ≤ 1024) : 0 ≤ omL Y t m ∧ omL Y t m ≤ 1 / 20 := by
  have hmax : max (m + Y) 0 ≤ 10000 := max_le (by linarith) (by norm_num)
  have h0 : 0 ≤ max (m + Y) 0 := le_max_right _ _
  have ht' : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg t
  unfold omL cL0 cL1
  constructor <;> norm_num <;> nlinarith

private theorem omA_bounds {t : ℕ} {Y m : ℝ} (h2 : m ≤ Y) (hY5 : Y ≤ 5000)
    (ht : t ≤ 1024) : 0 ≤ omA Y t m ∧ omA Y t m ≤ 1 / 20 := by
  have hmax : max (m + Y) 0 ≤ 10000 := max_le (by linarith) (by norm_num)
  have h0 : 0 ≤ max (m + Y) 0 := le_max_right _ _
  have ht' : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg t
  unfold omA cA0 cA1
  constructor <;> norm_num <;> nlinarith

/-- The sum `l` is within `12%` of the exact sum, up to `2^-80`. -/
private theorem l_rel {L t : ℕ} {sc sh : ℕ → ℝ} {δ Y m l : ℝ} (inv : RowInv L t sc sh δ Y m l)
    (hY : ∀ j, j ≤ L → |sc j| + δ ≤ Y) (hY5 : Y ≤ 5000) (ht0 : 0 < t) (ht : t ≤ 1024) :
    22 / 25 * Lam sh (keysN L t) m - 2 ^ (-80 : ℤ) ≤ l ∧
      l ≤ 28 / 25 * Lam sh (keysN L t) m + 2 ^ (-80 : ℤ) := by
  obtain ⟨hm1, hm2⟩ := m_bounds inv hY ht0
  obtain ⟨hw0, hw1⟩ := omL_bounds hm2 hY5 ht
  have hY0 : 0 ≤ Y := by linarith
  have hB0 : 0 ≤ bL Y := by unfold bL; positivity
  have hB1 : bL Y ≤ 1 / 70 := by unfold bL; norm_num; linarith
  obtain ⟨g1, g2⟩ := growth hw0 hw1 hB0 hB1
  have hLam := (Lam_bounds inv ht0 ht).1
  have ht' : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have hη : (0 : ℝ) ≤ ηL := by unfold ηL; positivity
  have tiny : Real.exp (omL Y t m) * (t * ηL) ≤ 2 ^ (-80 : ℤ) := by
    calc Real.exp (omL Y t m) * (t * ηL) ≤ 11 / 10 * (1024 * ηL) :=
          mul_le_mul g1 (mul_le_mul_of_nonneg_right ht' hη) (by positivity) (by norm_num)
      _ ≤ 2 ^ (-80 : ℤ) := by unfold ηL; norm_num
  have cl := inv.l_close
  unfold Close at cl
  have hG := mul_le_mul_of_nonneg_right g2 (by linarith : (0 : ℝ) ≤ Lam sh (keysN L t) m)
  obtain ⟨c1, c2⟩ := abs_le.mp cl
  generalize (2 : ℝ) ^ (-80 : ℤ) = e at tiny ⊢
  constructor <;> linarith

/-- **The sum is bounded** on a row that has run: `1/2 ≤ l ≤ 2^18`, and `|m| ≤ Y`. -/
theorem RowInv.bounds {L t : ℕ} {sc sh : ℕ → ℝ} {δ Y m l : ℝ} (inv : RowInv L t sc sh δ Y m l)
    (hδ : 0 ≤ δ) (hY : ∀ j, j ≤ L → |sc j| + δ ≤ Y) (hY5 : Y ≤ 5000)
    (ht0 : 0 < t) (ht : t ≤ 1024) : 1 / 2 ≤ l ∧ l ≤ 2 ^ 18 ∧ -Y ≤ m ∧ m ≤ Y := by
  have _ := hδ
  obtain ⟨hm1, hm2⟩ := m_bounds inv hY ht0
  obtain ⟨r1, r2⟩ := l_rel inv hY hY5 ht0 ht
  obtain ⟨l1, l2⟩ := Lam_bounds inv ht0 ht
  have e : (2 : ℝ) ^ (-80 : ℤ) ≤ 1 / 100 := by norm_num
  refine ⟨by linarith, by linarith, hm1, hm2⟩

/-- **The accumulator is a weighted average of the values**: `|a| ≤ 2^121` when every `|v| ≤ 2^120`
(`a` the normalised accumulator after `t ≥ 1` trips). -/
theorem acc_le {L t : ℕ} {sc sh vv : ℕ → ℝ} {δ Y V m l a : ℝ}
    (hδ : 0 ≤ δ) (hY : ∀ j, j ≤ L → |sc j| + δ ≤ Y) (hY5 : Y ≤ 5000)
    (ht0 : 0 < t) (ht : t ≤ 1024) (inv : RowInv L t sc sh δ Y m l)
    (hvv : ∀ j, j ≤ L → |vv j| ≤ 2 ^ 120) (hV0 : 0 ≤ V) (hV : ∑ j ∈ range (L + 1), |vv j| ≤ V) (hVu : V ≤ 2 ^ 138)
    (cinv : Close (l * a) (Asum sh vv (keysN L t) m) (Aabs sh vv (keysN L t) m) (omA Y t m)
      (bA Y) (ηA * (1 + V)) t) :
    |a| ≤ 2 ^ 121 ∧ |l * a| ≤ 4 * (1 + V) ∧ |a| ≤ 8 * (1 + V) := by
  have _ := hδ
  obtain ⟨hm1, hm2⟩ := m_bounds inv hY ht0
  obtain ⟨r1, -⟩ := l_rel inv hY hY5 ht0 ht
  obtain ⟨l1, l2⟩ := Lam_bounds inv ht0 ht
  have e : (2 : ℝ) ^ (-80 : ℤ) ≤ 1 / 100 := by norm_num
  have hl : 1 / 2 ≤ l := by linarith
  obtain ⟨hw0, hw1⟩ := omA_bounds hm2 hY5 ht
  have hY0 : 0 ≤ Y := by linarith
  have hB0 : 0 ≤ bA Y := by unfold bA; positivity
  have hB1 : bA Y ≤ 1 / 70 := by unfold bA; norm_num; linarith
  obtain ⟨g1, g2⟩ := growth hw0 hw1 hB0 hB1
  have ht' : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have hV1 : (0 : ℝ) ≤ 1 + V := by linarith
  have hη : (0 : ℝ) ≤ ηA * (1 + V) := by unfold ηA; positivity
  have tiny : Real.exp (omA Y t m) * (t * (ηA * (1 + V))) ≤ 2 ^ (-85 : ℤ) * (1 + V) := by
    calc Real.exp (omA Y t m) * (t * (ηA * (1 + V)))
        ≤ 11 / 10 * (1024 * (ηA * (1 + V))) :=
          mul_le_mul g1 (mul_le_mul_of_nonneg_right ht' hη) (by positivity) (by norm_num)
      _ ≤ 2 ^ (-85 : ℤ) * (1 + V) := by unfold ηA; norm_num; nlinarith
  have hsub : keysN L t ⊆ range (L + 1) := by
    intro j hj; unfold keysN at hj; simp only [mem_range] at hj ⊢; omega
  have hAabs : Aabs sh vv (keysN L t) m ≤ V := by
    calc Aabs sh vv (keysN L t) m ≤ ∑ j ∈ keysN L t, |vv j| := by
          unfold Aabs
          refine sum_le_sum fun j hj => ?_
          have : Real.exp (sh j - m) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith [inv.le_m j hj])
          nlinarith [abs_nonneg (vv j)]
      _ ≤ ∑ j ∈ range (L + 1), |vv j| :=
          sum_le_sum_of_subset_of_nonneg hsub fun _ _ _ => abs_nonneg _
      _ ≤ V := hV
  have hAabs' : Aabs sh vv (keysN L t) m ≤ 2 ^ 120 * Lam sh (keysN L t) m := by
    unfold Aabs Lam
    rw [mul_sum]
    refine sum_le_sum fun j hj => ?_
    have hj' : j ≤ L := by
      have := hsub hj; simp only [mem_range] at this; omega
    have := hvv j hj'
    nlinarith [Real.exp_pos (sh j - m)]
  have hAnn : 0 ≤ Aabs sh vv (keysN L t) m :=
    sum_nonneg fun j _ => mul_nonneg (Real.exp_pos _).le (abs_nonneg _)
  have hAs : |Asum sh vv (keysN L t) m| ≤ Aabs sh vv (keysN L t) m := by
    unfold Asum Aabs
    refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun j _ => ?_)
    rw [abs_mul, abs_of_pos (Real.exp_pos _)]
  have cl := cinv
  unfold Close at cl
  have hG := mul_le_mul_of_nonneg_right g2 hAnn
  have key : |l * a| ≤ 12 / 5 * Aabs sh vv (keysN L t) m / 2 + 2 ^ (-85 : ℤ) * (1 + V) := by
    have := abs_sub_abs_le_abs_sub (l * a) (Asum sh vv (keysN L t) m)
    generalize (2 : ℝ) ^ (-85 : ℤ) = e at tiny ⊢
    linarith
  have e85 : (2 : ℝ) ^ (-85 : ℤ) * (1 + V) ≤ 1 / 100 * (1 + V) := by
    have : (2 : ℝ) ^ (-85 : ℤ) ≤ 1 / 100 := by norm_num
    exact mul_le_mul_of_nonneg_right this hV1
  have h2 : |l * a| ≤ 4 * (1 + V) := by nlinarith
  have habs : |l * a| = l * |a| := by rw [abs_mul, abs_of_pos (by linarith : 0 < l)]
  refine ⟨?_, h2, ?_⟩
  · have hLm : Lam sh (keysN L t) m ≤ 25 / 22 * (l + 1 / 100) := by linarith
    have hV2 : 2 ^ (-85 : ℤ) * (1 + V) ≤ (2 : ℝ) ^ 55 := by
      have : (2 : ℝ) ^ (-85 : ℤ) * (1 + V) ≤ 2 ^ (-85 : ℤ) * (1 + 2 ^ 138) :=
        mul_le_mul_of_nonneg_left (by linarith) (by positivity)
      refine this.trans ?_
      norm_num
    have hlA : l * |a| ≤ 12 / 5 / 2 * (2 ^ 120 * Lam sh (keysN L t) m) + 2 ^ 55 := by
      rw [← habs]; nlinarith
    have h120 : (0 : ℝ) < 2 ^ 120 := by positivity
    nlinarith [abs_nonneg a]
  · nlinarith [abs_nonneg a]

/-- A fresh row (no trip yet) satisfies the invariant. -/
theorem RowInv.zero {L : ℕ} {sc sh : ℕ → ℝ} {δ Y m : ℝ} : RowInv L 0 sc sh δ Y m 0 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [keysN_zero, Close, Lam]

/-- A fresh column: the accumulator is `0`. -/
theorem Close.zero_col {L : ℕ} {sh vv : ℕ → ℝ} {Y V m : ℝ} :
    Close (0 * 0) (Asum sh vv (keysN L 0) m) (Aabs sh vv (keysN L 0) m) (omA Y 0 m)
      (bA Y) (ηA * (1 + V)) 0 := by
  simp [keysN_zero, Close, Asum, Aabs]

end Num

end Claims.PrefillAttnVarlen

end

section
/-!
# The row's step

`rowstep`: one trip keeps `RowInv` (from `trip_l`, `prop_step` and the splitting of the exact sums).
-/

namespace Claims.PrefillAttnVarlen

namespace Num

open Finset

/-- **The row's step.** Trip `t` (block `t` of the keys `≤ L`): the lane scores `qs` within `δ` of
the exact ones, `mij` their maximum, `m'` the new running maximum; the device's errors for
`α'`, `β'`, `p`, `lij`, `l'` as in `trip_l`. The first trip (`t = 0`) has `α = 0`, `l = 0`. -/
theorem rowstep {L t : ℕ} {sc sh qs p : ℕ → ℝ} {δ Y m l m' mij Δ α α' β β' l' lij : ℝ}
    (hδ : 0 ≤ δ) (hY : ∀ j, j ≤ L → |sc j| + δ ≤ Y) (hY5 : Y ≤ 5000)
    (hL : 128 * t ≤ L) (ht : t < 1024)
    (inv : RowInv L t sc sh δ Y m l)
    (hqs : ∀ c ∈ blk L t, |qs c - sc (128 * t + c)| ≤ δ)
    (hmij : (∃ c ∈ blk L t, qs c = mij) ∧ ∀ c ∈ blk L t, qs c ≤ mij)
    (hm' : (t = 0 ∧ m' = mij ∧ Δ = 0 ∧ α = 0) ∨
      (0 < t ∧ m' = max m mij ∧ Δ = m' - m ∧ α = Real.exp (m - m')))
    (hα' : |α' - α| ≤ rA Δ * α + ηE)
    (hβ : β = Real.exp (mij - m')) (hβ' : |β' - β| ≤ rE Y * β + ηE)
    (hp0 : ∀ c ∈ range 128, c ∉ blk L t → p c = 0)
    (hp : ∀ c ∈ blk L t, |p c - Real.exp (qs c - mij)| ≤ rE Y * Real.exp (qs c - mij) + ηE)
    (hlij : |lij - ∑ c ∈ range 128, p c| ≤ rS * ∑ c ∈ range 128, |p c| + ηS)
    (hl' : |l' - (α' * l + β' * lij)| ≤ rO * (|α' * l| + |β' * lij|) + ηO) :
    RowInv L (t + 1) sc (shNext t sh qs) δ Y m' l' := by
  have hY0 : 0 ≤ Y := by
    have := hY 0 (Nat.zero_le _)
    have := abs_nonneg (sc 0)
    linarith
  have hbl : ∀ c ∈ blk L t, c < 128 ∧ 128 * t + c ≤ L := by
    intro c hc
    simpa [blk] using hc
  have hqsY : ∀ c ∈ blk L t, |qs c| ≤ Y := by
    intro c hc
    have h1 := hqs c hc
    have h2 := hY (128 * t + c) (hbl c hc).2
    have h3 : |qs c| ≤ |sc (128 * t + c)| + |qs c - sc (128 * t + c)| := by
      calc |qs c| = |sc (128 * t + c) + (qs c - sc (128 * t + c))| := by ring_nf
        _ ≤ _ := abs_add_le _ _
    linarith
  obtain ⟨⟨c0, hc0, hc0e⟩, hle⟩ := hmij
  have hmijY : -Y ≤ mij ∧ mij ≤ Y := by
    rw [← hc0e]; exact abs_le.mp (hqsY c0 hc0)
  have hmij_le : mij ≤ m' := by
    rcases hm' with ⟨-, h, -, -⟩ | ⟨-, h, -, -⟩
    · exact h.ge
    · rw [h]; exact le_max_right _ _
  have hpos : 0 < t → 1 / 2 ≤ l ∧ l ≤ 2 ^ 18 ∧ -Y ≤ m ∧ m ≤ Y := fun h =>
    inv.bounds hδ hY hY5 h ht.le
  have hm'Y : -Y ≤ m' ∧ m' ≤ Y := by
    refine ⟨by linarith [hmijY.1], ?_⟩
    rcases hm' with ⟨-, h, -, -⟩ | ⟨h0, h, -, -⟩
    · rw [h]; exact hmijY.2
    · rw [h]; exact max_le (hpos h0).2.2.2 hmijY.2
  have hl0 : t = 0 → l = 0 := by
    intro h0
    subst h0
    have := inv.l_close
    simp only [keysN_zero, Lam, sum_empty, Close, Nat.cast_zero, mul_zero, zero_mul, sub_zero,
      mul_zero, add_zero] at this
    exact abs_nonpos_iff.mp this
  have hold : ∀ j, j ∈ keysN L (t + 1) → (j < 128 * t) ∨ ∃ c ∈ blk L t, j = 128 * t + c := by
    intro j hj
    simp only [keysN, mem_range, lt_min_iff] at hj
    by_cases h : j < 128 * t
    · exact Or.inl h
    · exact Or.inr ⟨j - 128 * t, by simp only [blk, mem_filter, mem_range]; omega, by omega⟩
  have hkeys : ∀ j, j < 128 * t → j ∈ keysN L t := by
    intro j hj; rw [keysN_eq_range hL]; exact mem_range.mpr hj
  have hkeys' : ∀ j ∈ keysN L t, j < 128 * t := by
    intro j hj; rw [keysN_eq_range hL] at hj; exact mem_range.mp hj
  have hkeysS : ∀ j, j < 128 * t → j ∈ keysN L (t + 1) := by
    intro j hj
    simp only [keysN, mem_range, lt_min_iff]; omega
  have hkeysN : ∀ c ∈ blk L t, 128 * t + c ∈ keysN L (t + 1) := by
    intro c hc
    have := hbl c hc
    simp only [keysN, mem_range, lt_min_iff]; omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro j hj
    rcases hold j hj with h | ⟨c, hc, rfl⟩
    · rw [shNext_old h]; exact inv.close_sh j (hkeys j h)
    · rw [shNext_new]; exact hqs c hc
  · intro _
    rcases hm' with ⟨h0, h, -, -⟩ | ⟨h0, h, -, -⟩
    · exact ⟨128 * t + c0, hkeysN c0 hc0, by rw [shNext_new, hc0e, h]⟩
    · rcases le_total m mij with hmm | hmm
      · refine ⟨128 * t + c0, hkeysN c0 hc0, ?_⟩
        rw [shNext_new, hc0e, h, max_eq_right hmm]
      · obtain ⟨j, hj, hje⟩ := inv.attain h0
        refine ⟨j, hkeysS j (hkeys' j hj), ?_⟩
        rw [shNext_old (hkeys' j hj), hje, h, max_eq_left hmm]
  · intro j hj
    rcases hold j hj with h | ⟨c, hc, rfl⟩
    · rw [shNext_old h]
      have h1 := inv.le_m j (hkeys j h)
      have h0 : 0 < t := by
        rcases Nat.eq_zero_or_pos t with h0 | h0
        · subst h0; omega
        · exact h0
      rcases hm' with ⟨h0', -⟩ | ⟨-, h', -, -⟩
      · omega
      · rw [h']; exact h1.trans (le_max_left _ _)
    · rw [shNext_new]; exact (hle c hc).trans hmij_le
  · -- the sum
    set Zs := ∑ c ∈ blk L t, Real.exp (qs c - m') with hZs
    have hlam := Lam_succ hL sh qs m m'
    have hLam0 : 0 ≤ Lam sh (keysN L t) m := sum_nonneg fun j _ => (Real.exp_pos _).le
    have hLam0' : 0 ≤ Lam (shNext t sh qs) (keysN L (t + 1)) m' :=
      sum_nonneg fun j _ => (Real.exp_pos _).le
    have hZ0 : 0 ≤ Zs := sum_nonneg fun c _ => (Real.exp_pos _).le
    have hB0 : 0 ≤ bL Y := by unfold bL; positivity
    -- the data of the two cases
    have hcase : (0 ≤ α ∧ α ≤ 1) ∧ 0 ≤ Δ ∧ Δ ≤ 2 * Y ∧ 0 ≤ l ∧ l ≤ 2 ^ 18 ∧
        Lam (shNext t sh qs) (keysN L (t + 1)) m' = α * Lam sh (keysN L t) m + Zs ∧
        (α = 0 ∨ (0 ≤ omL Y t m ∧ omL Y t m + (cL0 + cL1 * Δ) ≤ omL Y (t + 1) m')) := by
      rcases hm' with ⟨h0, h, hΔe, hαe⟩ | ⟨h0, h, hΔe, hαe⟩
      · subst h0
        have hl := hl0 rfl
        refine ⟨⟨?_, ?_⟩, ?_, ?_, ?_, ?_, ?_, Or.inl hαe⟩
        · rw [hαe]
        · rw [hαe]; norm_num
        · rw [hΔe]
        · rw [hΔe]; exact mul_nonneg (by norm_num) hY0
        · rw [hl]
        · rw [hl]; norm_num
        · rw [hlam, hαe]
          simp [Lam, keysN_zero, hZs]
      · obtain ⟨hl1, hl2, hm1, hm2⟩ := hpos h0
        have hmm : m ≤ m' := by rw [h]; exact le_max_left _ _
        refine ⟨⟨?_, ?_⟩, ?_, ?_, ?_, hl2, ?_, Or.inr ⟨?_, ?_⟩⟩
        · rw [hαe]; exact (Real.exp_pos _).le
        · rw [hαe]; exact Real.exp_le_one_iff.mpr (by linarith)
        · rw [hΔe]; linarith
        · rw [hΔe]; linarith [hm'Y.2]
        · linarith
        · rw [hlam, hαe]
        · unfold omL
          have : (0 : ℝ) ≤ cL0 * t := by unfold cL0; positivity
          have : 0 ≤ cL1 * max (m + Y) 0 := by unfold cL1; positivity
          linarith
        · unfold omL
          rw [Nat.cast_succ, hΔe]
          have e1 : max (m + Y) 0 = m + Y := max_eq_left (by linarith)
          have e2 : m' + Y ≤ max (m' + Y) 0 := le_max_left _ _
          have : cL1 * (m' - m) + cL1 * (m + Y) ≤ cL1 * max (m' + Y) 0 := by
            have : (0 : ℝ) ≤ cL1 := by unfold cL1; positivity
            nlinarith
          rw [e1]; linarith
    obtain ⟨⟨hα0, hα1⟩, hΔ0, hΔ, hl0', hl2, hX', hωc⟩ := hcase
    have hmα : α = 0 ∨ α = Real.exp (m - m') := by
      rcases hm' with ⟨-, -, -, h⟩ | ⟨-, -, -, h⟩
      · exact Or.inl h
      · exact Or.inr h
    have htl : |l' - (α * l + Zs)| ≤ (cL0 + cL1 * Δ) * (α * |l|) + bL Y * Zs + ηL := by
      have hβ0 : 0 ≤ β := by rw [hβ]; exact (Real.exp_pos _).le
      have hβ1 : β ≤ 1 := by rw [hβ]; exact Real.exp_le_one_iff.mpr (by linarith)
      have h := trip_l (C := blk L t) (e := fun c => Real.exp (qs c - mij)) (p := p)
        (β := β) (β' := β') (α := α) (α' := α') (l := l) (l' := l') (lij := lij) (Δ := Δ)
        (Y := Y) (fun c hc => by simpa [blk] using (mem_filter.mp hc).1) hY0 hY5 hΔ0 hΔ hα0 hα1
        hβ0 hβ1 (fun c _ => (Real.exp_pos _).le)
        (fun c hc => Real.exp_le_one_iff.mpr (by linarith [hle c hc])) hp0 hp hβ' hα' hl0' hl2
        hlij hl'
      have e : ∑ c ∈ blk L t, Real.exp (qs c - mij) * β = Zs := by
        refine sum_congr rfl fun c _ => ?_
        rw [hβ, ← Real.exp_add]; congr 1; ring
      rwa [e] at h
    refine (prop_step (x := l) (x' := l') (X := Lam sh (keysN L t) m)
      (X' := Lam (shNext t sh qs) (keysN L (t + 1)) m') (M := Lam sh (keysN L t) m)
      (M' := Lam (shNext t sh qs) (keysN L (t + 1)) m') (Y := Zs) (Z := Zs) (α := α)
      (ra := cL0 + cL1 * Δ) (rb := bL Y) (η := ηL) (B := bL Y) (ω := omL Y t m)
      (ω' := omL Y (t + 1) m') (n := t) hα0 hα1 (by unfold cL0 cL1; have := hΔ0; positivity)
      le_rfl hB0 (by unfold ηL; positivity) hX' hX' (abs_of_nonneg hLam0).le (abs_of_nonneg hZ0).le
      ⟨?_, hωc⟩ htl ?_)
    · unfold omL
      have : (0 : ℝ) ≤ cL0 * ((t + 1 : ℕ) : ℝ) := by unfold cL0; positivity
      have : 0 ≤ cL1 * max (m' + Y) 0 := by unfold cL1; positivity
      linarith
    · rcases hmα with h | h
      · exact Or.inl h
      · exact Or.inr inv.l_close

end Num

end Claims.PrefillAttnVarlen

end

section
/-!
# The column's step

`colstep`: one trip keeps the accumulator's running bound (from `trip_a`, `prop_step` and the splitting of the exact sums).
-/

namespace Claims.PrefillAttnVarlen

namespace Num

open Finset

/-- **The column's step.** One column of values `vv` (`|vv j| ≤ 2^120`, `V` bounds `Σ_{j ≤ L} |vv j|`),
the old unnormalised accumulator `l·a` within the running bound, the trip's data as in `rowstep` and
`trip_a`; then `l'·acc2` is within the running bound at `t + 1`. -/
theorem colstep {L t : ℕ} {sc sh qs p p2 vv : ℕ → ℝ}
    {δ Y V m l m' mij Δ α α' β β' l' ps S a acc1 acc2 : ℝ}
    (hδ : 0 ≤ δ) (hY : ∀ j, j ≤ L → |sc j| + δ ≤ Y) (hY5 : Y ≤ 5000)
    (hL : 128 * t ≤ L) (ht : t < 1024)
    (inv : RowInv L t sc sh δ Y m l) (inv' : RowInv L (t + 1) sc (shNext t sh qs) δ Y m' l')
    (hvv : ∀ j, j ≤ L → |vv j| ≤ 2 ^ 120) (hV0 : 0 ≤ V) (hV : ∑ j ∈ range (L + 1), |vv j| ≤ V) (hVu : V ≤ 2 ^ 138)
    (cinv : Close (l * a) (Asum sh vv (keysN L t) m) (Aabs sh vv (keysN L t) m) (omA Y t m)
      (bA Y) (ηA * (1 + V)) t)
    (ha0 : t = 0 → l = 0 ∧ a = 0)
    (hqs : ∀ c ∈ blk L t, |qs c - sc (128 * t + c)| ≤ δ)
    (hmij : (∃ c ∈ blk L t, qs c = mij) ∧ ∀ c ∈ blk L t, qs c ≤ mij)
    (hm' : (t = 0 ∧ m' = mij ∧ Δ = 0 ∧ α = 0) ∨
      (0 < t ∧ m' = max m mij ∧ Δ = m' - m ∧ α = Real.exp (m - m')))
    (hα' : |α' - α| ≤ rA Δ * α + ηE)
    (hβ : β = Real.exp (mij - m')) (hβ' : |β' - β| ≤ rE Y * β + ηE)
    (hp : ∀ c ∈ blk L t, |p c - Real.exp (qs c - mij)| ≤ rE Y * Real.exp (qs c - mij) + ηE)
    (hps : |ps - β' / l'| ≤ 2 ^ (-22 : ℤ) * (β' / l') + 2 ^ (-148 : ℤ))
    (hp2 : ∀ c ∈ blk L t,
      |p2 c - p c * ps| ≤ (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ)) * |p c * ps| + 2 ^ (-130 : ℤ))
    (hS : |S - (l / l') * α'| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ)) * ((l / l') * α') + 2 ^ (-140 : ℤ))
    (ha1 : |acc1 - a * S| ≤ 2 ^ (-24 : ℤ) * |a * S| + 2 ^ (-149 : ℤ))
    (ha2 : |acc2 - (acc1 + ∑ c ∈ blk L t, p2 c * vv (128 * t + c))| ≤
      rD * (|acc1| + ∑ c ∈ blk L t, |p2 c * vv (128 * t + c)|) +
        (adc0 + adc1 * ∑ c ∈ blk L t, |vv (128 * t + c)|)) :
    Close (l' * acc2) (Asum (shNext t sh qs) vv (keysN L (t + 1)) m')
      (Aabs (shNext t sh qs) vv (keysN L (t + 1)) m') (omA Y (t + 1) m') (bA Y)
      (ηA * (1 + V)) (t + 1) := by
  have hY0 : 0 ≤ Y := by
    have := hY 0 (Nat.zero_le _)
    have := abs_nonneg (sc 0)
    linarith
  have hbl : ∀ c ∈ blk L t, c < 128 ∧ 128 * t + c ≤ L := by
    intro c hc
    simpa [blk] using hc
  have hqsY : ∀ c ∈ blk L t, |qs c| ≤ Y := by
    intro c hc
    have h1 := hqs c hc
    have h2 := hY (128 * t + c) (hbl c hc).2
    have h3 : |qs c| ≤ |sc (128 * t + c)| + |qs c - sc (128 * t + c)| := by
      calc |qs c| = |sc (128 * t + c) + (qs c - sc (128 * t + c))| := by ring_nf
        _ ≤ _ := abs_add_le _ _
    linarith
  obtain ⟨⟨c0, hc0, hc0e⟩, hle⟩ := hmij
  have hmijY : -Y ≤ mij ∧ mij ≤ Y := by
    rw [← hc0e]; exact abs_le.mp (hqsY c0 hc0)
  have hmij_le : mij ≤ m' := by
    rcases hm' with ⟨-, h, -, -⟩ | ⟨-, h, -, -⟩
    · exact h.ge
    · rw [h]; exact le_max_right _ _
  have hpos : 0 < t → 1 / 2 ≤ l ∧ l ≤ 2 ^ 18 ∧ -Y ≤ m ∧ m ≤ Y := fun h =>
    inv.bounds hδ hY hY5 h ht.le
  have hm'Y : -Y ≤ m' ∧ m' ≤ Y := by
    refine ⟨by linarith [hmijY.1], ?_⟩
    rcases hm' with ⟨-, h, -, -⟩ | ⟨h0, h, -, -⟩
    · rw [h]; exact hmijY.2
    · rw [h]; exact max_le (hpos h0).2.2.2 hmijY.2
  have hl'b := inv'.bounds hδ hY hY5 (Nat.succ_pos t) ht
  have hsize : (0 ≤ l ∧ l ≤ 2 ^ 18) ∧ |l * a| ≤ 4 * (1 + V) ∧ |a| ≤ 8 * (1 + V) := by
    rcases Nat.eq_zero_or_pos t with h0 | h0
    · obtain ⟨hl, ha⟩ := ha0 h0
      subst hl; subst ha
      refine ⟨⟨le_rfl, by norm_num⟩, ?_, ?_⟩ <;> simp <;> linarith
    · obtain ⟨-, h2, h3⟩ := acc_le hδ hY hY5 h0 ht.le inv hvv hV0 hV hVu cinv
      exact ⟨⟨by linarith [(hpos h0).1], (hpos h0).2.1⟩, h2, h3⟩
  obtain ⟨⟨hl0, hl2⟩, hla, ha8⟩ := hsize
  have hvsum : ∑ c ∈ blk L t, |vv (128 * t + c)| ≤ V := by
    have h1 := sum_keysN_succ hL (fun j => |vv j|)
    have h2 : ∑ j ∈ keysN L (t + 1), |vv j| ≤ ∑ j ∈ range (L + 1), |vv j| :=
      sum_le_sum_of_subset_of_nonneg
        (fun j hj => by simp only [keysN, mem_range, lt_min_iff] at hj ⊢; omega)
        (fun j _ _ => abs_nonneg _)
    have h3 : 0 ≤ ∑ j ∈ keysN L t, |vv j| := sum_nonneg fun j _ => abs_nonneg _
    linarith
  have hAs := Asum_succ hL sh qs vv m m'
  have hAb := Aabs_succ hL sh qs vv m m'
  set Ys := ∑ c ∈ blk L t, Real.exp (qs c - m') * vv (128 * t + c) with hYs
  set Zs := ∑ c ∈ blk L t, Real.exp (qs c - m') * |vv (128 * t + c)| with hZs
  have hcase : (0 ≤ α ∧ α ≤ 1) ∧ 0 ≤ Δ ∧ Δ ≤ 2 * Y ∧
      Asum (shNext t sh qs) vv (keysN L (t + 1)) m' = α * Asum sh vv (keysN L t) m + Ys ∧
      Aabs (shNext t sh qs) vv (keysN L (t + 1)) m' = α * Aabs sh vv (keysN L t) m + Zs ∧
      (α = 0 ∨ (0 ≤ omA Y t m ∧ omA Y t m + (cA0 + cA1 * Δ) ≤ omA Y (t + 1) m')) := by
    rcases hm' with ⟨h0, h, hΔe, hαe⟩ | ⟨h0, h, hΔe, hαe⟩
    · subst h0
      refine ⟨⟨?_, ?_⟩, ?_, ?_, ?_, ?_, Or.inl hαe⟩
      · rw [hαe]
      · rw [hαe]; norm_num
      · rw [hΔe]
      · rw [hΔe]; exact mul_nonneg (by norm_num) hY0
      · rw [hAs, hαe]; simp [Asum, keysN_zero]
      · rw [hAb, hαe]; simp [Aabs, keysN_zero]
    · obtain ⟨hl1, -, hm1, hm2⟩ := hpos h0
      have hmm : m ≤ m' := by rw [h]; exact le_max_left _ _
      refine ⟨⟨?_, ?_⟩, ?_, ?_, ?_, ?_, Or.inr ⟨?_, ?_⟩⟩
      · rw [hαe]; exact (Real.exp_pos _).le
      · rw [hαe]; exact Real.exp_le_one_iff.mpr (by linarith)
      · rw [hΔe]; linarith
      · rw [hΔe]; linarith [hm'Y.2]
      · rw [hAs, hαe]
      · rw [hAb, hαe]
      · unfold omA
        have : (0 : ℝ) ≤ cA0 * t := by unfold cA0; positivity
        have : 0 ≤ cA1 * max (m + Y) 0 := by unfold cA1; positivity
        linarith
      · unfold omA
        rw [Nat.cast_succ, hΔe]
        have e1 : max (m + Y) 0 = m + Y := max_eq_left (by linarith)
        have e2 : m' + Y ≤ max (m' + Y) 0 := le_max_left _ _
        have : cA1 * (m' - m) + cA1 * (m + Y) ≤ cA1 * max (m' + Y) 0 := by
          have : (0 : ℝ) ≤ cA1 := by unfold cA1; positivity
          nlinarith
        rw [e1]; linarith
  obtain ⟨⟨hα0, hα1⟩, hΔ0, hΔ, hX', hM', hωc⟩ := hcase
  have hmα : α = 0 ∨ α = Real.exp (m - m') := by
    rcases hm' with ⟨-, -, -, h⟩ | ⟨-, -, -, h⟩
    · exact Or.inl h
    · exact Or.inr h
  have hβ0 : 0 ≤ β := by rw [hβ]; exact (Real.exp_pos _).le
  have hβ1 : β ≤ 1 := by rw [hβ]; exact Real.exp_le_one_iff.mpr (by linarith)
  have hstep : |l' * acc2 - (α * (l * a) + Ys)| ≤
      (cA0 + cA1 * Δ) * (α * |l * a|) + bA Y * Zs + ηA * (1 + V) := by
    have h := trip_a (C := blk L t) (e := fun c => Real.exp (qs c - mij)) (p := p) (p2 := p2)
      (v := fun c => vv (128 * t + c)) (β := β) (β' := β') (α := α) (α' := α') (l := l)
      (l' := l') (ps := ps) (S := S) (a := a) (acc1 := acc1) (acc2 := acc2) (Δ := Δ) (Y := Y)
      (V := V) (fun c hc => by simpa [blk] using (mem_filter.mp hc).1) hY0 hY5 hΔ0 hΔ hα0 hα1
      hβ0 hβ1 (fun c _ => (Real.exp_pos _).le)
      (fun c hc => Real.exp_le_one_iff.mpr (by linarith [hle c hc])) hp hβ' hα' hl0 hl2
      hl'b.1 hl'b.2.1 hps hp2 hS ha1 ha2 (fun c hc => hvv _ (hbl c hc).2) hV0 hvsum hla ha8
    have e : ∀ f : ℕ → ℝ, ∑ c ∈ blk L t, Real.exp (qs c - mij) * β * f c =
        ∑ c ∈ blk L t, Real.exp (qs c - m') * f c := fun f => by
      refine sum_congr rfl fun c _ => ?_
      rw [mul_assoc, ← mul_assoc _ β, hβ, ← Real.exp_add]; congr 2; ring
    have e1 := e (fun c => vv (128 * t + c))
    have e2 := e (fun c => |vv (128 * t + c)|)
    rwa [e1, e2] at h
  have hB0 : 0 ≤ bA Y := by unfold bA; positivity
  refine prop_step (x := l * a) (x' := l' * acc2) (X := Asum sh vv (keysN L t) m)
    (X' := Asum (shNext t sh qs) vv (keysN L (t + 1)) m') (M := Aabs sh vv (keysN L t) m)
    (M' := Aabs (shNext t sh qs) vv (keysN L (t + 1)) m') (Y := Ys) (Z := Zs) (α := α)
    (ra := cA0 + cA1 * Δ) (rb := bA Y) (η := ηA * (1 + V)) (B := bA Y) (ω := omA Y t m)
    (ω' := omA Y (t + 1) m') (n := t) hα0 hα1 (by unfold cA0 cA1; have := hΔ0; positivity)
    le_rfl hB0 (by unfold ηA; positivity) hX' hM' ?_ ?_ ⟨?_, hωc⟩ hstep ?_
  · unfold Asum Aabs
    exact (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun j _ => by
      rw [abs_mul, abs_of_pos (Real.exp_pos _)])
  · exact (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun c _ => by
      rw [abs_mul, abs_of_pos (Real.exp_pos _)])
  · unfold omA
    have : (0 : ℝ) ≤ cA0 * ((t + 1 : ℕ) : ℝ) := by unfold cA0; positivity
    have : 0 ≤ cA1 * max (m' + Y) 0 := by unfold cA1; positivity
    linarith
  · rcases hmα with h | h
    · exact Or.inl h
    · exact Or.inr cinv

end Num

end Claims.PrefillAttnVarlen

end
