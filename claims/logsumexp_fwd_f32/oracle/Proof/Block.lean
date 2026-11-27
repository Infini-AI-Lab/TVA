import Proof.Exp

/-!
# One block: the sum of the lanes' `exp` and its `log`

`block_log`: the first `n` lanes' exponents `s_i` are finite, at most `κ ≤ 1/2`, one of them at
least `−κ`; the other lanes' are `−∞` or below `−2^126`. Then the `tl.sum` of the lanes'
`tl.exp` along a tree of depth at most `⌈log₂ B⌉ ≤ 16`, and its `tl.log`, give a finite `L` within
`100u` of `log Σ_{i<n} e^(s_i)`, with `|L| ≤ 12`.
-/

namespace Claims.LogsumexpFwd

open TVA

/-- A list sum over `range n` is the `Finset` sum. -/
lemma list_range_sum_eq {M : Type*} [AddCommMonoid M] (f : Nat → M) (n : Nat) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ m ih =>
      rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ]
      simp [ih]

/-- `ln 2^16 ≤ 11.1`. -/
lemma log_pow16_le : Real.log ((2 : ℝ) ^ 16) ≤ 11.1 := by
  rw [Real.log_pow]
  have := Real.log_two_lt_d9
  push_cast
  linarith

/-- The depth bound of a `tl.sum` over at most `2^16` lanes. -/
lemma sumBound_le {P : Platform} (hsd : Device.SumDepth P) {B : Nat} (hB : B ≤ 2 ^ 16) :
    tl.sumBound P B ≤ 16 := by
  have h1 : Nat.clog 2 B ≤ 16 := Nat.clog_le_of_le_pow hB
  unfold tl.sumBound
  exact max_le (le_trans (hsd B) h1) h1

/-- `(1 + u)^16 ≤ 1 + 16.001·u`. -/
lemma pow16_le : ((1 : ℚ) + 2 ^ (-24 : ℤ)) ^ 16 ≤ 1 + 16.001 * 2 ^ (-24 : ℤ) := by norm_num

set_option maxHeartbeats 1000000 in
/-- **The sum and the log of one block.** -/
theorem block_log {P : Platform} (hexp : Device.ExpF32 P) (hninf : Device.ExpF32NegInf P)
    (hlog : Device.LogF32 P) (hsd : Device.SumDepth P)
    {B n : Nat} (hn : 0 < n) (hnB : n ≤ B) (hB : B ≤ 2 ^ 16)
    {κ : ℝ} (hκ0 : 0 ≤ κ) (hκ : κ ≤ 1 / 2) {s e : Nat → F32} {S L : F32}
    (hsv : ∀ i, i < n → (s i).isFinite = true ∧ (s i).toReal ≤ κ)
    (hstar : ∃ i, i < n ∧ -κ ≤ (s i).toReal)
    (hsm : ∀ i, n ≤ i → i < B →
      s i = FloatV.negInf ∨ ((s i).isFinite = true ∧ (s i).toReal ≤ -2 ^ 126))
    (he : ∀ i, i < B → P.math .exp .binary32 (s i) (e i))
    (hS : ∃ t : Tree, t.Covers B ∧ t.depth ≤ tl.sumBound P B ∧ t.Eval (AddRel false) e S)
    (hL : P.math .log .binary32 S L) :
    L.isFinite = true ∧
      |L.toReal - Real.log (∑ i ∈ Finset.range n, Real.exp (s i).toReal)| ≤ 100 * uR ∧
      |L.toReal| ≤ 12 := by
  have hu := uR_pos
  have hu1 := uR_le
  -- the lanes
  have hlv : ∀ i, i < n → (e i).isFinite = true ∧ |(e i).toReal - Real.exp (s i).toReal| ≤
      (4 * uR + 2 * uR * |(s i).toReal|) * Real.exp (s i).toReal + 2 ^ (-140 : ℤ) := by
    intro i hi
    obtain ⟨hf, hle⟩ := hsv i hi
    exact exp_valid hexp hf (le_trans hle hκ) (he i (lt_of_lt_of_le hi hnB))
  have hlm : ∀ i, n ≤ i → i < B → (e i).isFinite = true ∧ |(e i).toReal| ≤ 2 ^ (-100 : ℤ) :=
    fun i h1 h2 => exp_masked hexp hninf (hsm i h1 h2) (he i h2)
  have hefin : ∀ i, i < B → (e i).isFinite = true := by
    intro i hi
    by_cases h : i < n
    · exact (hlv i h).1
    · exact (hlm i (not_lt.mp h) hi).1
  -- the exact sum `A'` of the valid lanes
  set A := ∑ i ∈ Finset.range n, Real.exp (s i).toReal with hAdef
  obtain ⟨i0, hi0, hi0κ⟩ := hstar
  have hA_lo : (1 : ℝ) / 2 ≤ A := by
    have h1 : Real.exp (s i0).toReal ≤ A := by
      rw [hAdef]
      exact Finset.single_le_sum (f := fun i => Real.exp (s i).toReal)
        (fun i _ => (Real.exp_pos _).le) (Finset.mem_range.mpr hi0)
    have h2 : Real.exp (-(1 / 2)) ≤ Real.exp (s i0).toReal :=
      Real.exp_le_exp.mpr (by linarith)
    linarith [half_le_exp_neg_half]
  have hApos : 0 < A := by linarith
  have hA_hi : A ≤ (n : ℝ) * Real.exp κ := by
    have h1 : A ≤ ∑ i ∈ Finset.range n, Real.exp κ := by
      rw [hAdef]
      exact Finset.sum_le_sum fun i hi =>
        Real.exp_le_exp.mpr (hsv i (Finset.mem_range.mp hi)).2
    simpa using h1
  -- the lanes' errors, summed
  have hsx : ∑ i ∈ Finset.range n, |(s i).toReal| * Real.exp (s i).toReal ≤
      16 * A + n * (2 : ℝ) ^ (-23 : ℤ) := by
    have h1 : ∑ i ∈ Finset.range n, |(s i).toReal| * Real.exp (s i).toReal ≤
        ∑ i ∈ Finset.range n, (16 * Real.exp (s i).toReal + (2 : ℝ) ^ (-23 : ℤ)) :=
      Finset.sum_le_sum fun i hi =>
        abs_mul_exp_le (le_trans (hsv i (Finset.mem_range.mp hi)).2 hκ)
    rw [Finset.sum_add_distrib, ← Finset.mul_sum] at h1
    simpa using h1
  have hnR : (n : ℝ) ≤ 2 ^ 16 := by exact_mod_cast le_trans hnB hB
  have hBR : (B : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hB
  have hvalid : ∑ i ∈ Finset.range n, |(e i).toReal - Real.exp (s i).toReal| ≤ 36.1 * uR * A := by
    have h1 : ∑ i ∈ Finset.range n, |(e i).toReal - Real.exp (s i).toReal| ≤
        ∑ i ∈ Finset.range n, ((4 * uR) * Real.exp (s i).toReal +
          (2 * uR) * (|(s i).toReal| * Real.exp (s i).toReal) + (2 : ℝ) ^ (-140 : ℤ)) :=
      Finset.sum_le_sum fun i hi => by
        have := (hlv i (Finset.mem_range.mp hi)).2
        linarith
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum] at h1
    simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at h1
    rw [← hAdef] at h1
    have h2 : (2 * uR) * ∑ i ∈ Finset.range n, |(s i).toReal| * Real.exp (s i).toReal ≤
        (2 * uR) * (16 * A + n * (2 : ℝ) ^ (-23 : ℤ)) :=
      mul_le_mul_of_nonneg_left hsx (by positivity)
    have h3 : (2 * uR) * (n * (2 : ℝ) ^ (-23 : ℤ)) ≤ (2 * uR) * (2 ^ 16 * (2 : ℝ) ^ (-23 : ℤ)) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hnR (by positivity)) (by positivity)
    have h4 : (n : ℝ) * (2 : ℝ) ^ (-140 : ℤ) ≤ 2 ^ 16 * (2 : ℝ) ^ (-140 : ℤ) :=
      mul_le_mul_of_nonneg_right hnR (by positivity)
    have h5 : (2 * uR) * (2 ^ 16 * (2 : ℝ) ^ (-23 : ℤ)) + 2 ^ 16 * (2 : ℝ) ^ (-140 : ℤ) ≤
        0.1 * uR * (1 / 2) := by unfold uR; norm_num
    have h6 : 0.1 * uR * (1 / 2) ≤ 0.1 * uR * A := mul_le_mul_of_nonneg_left hA_lo (by positivity)
    clear_value A
    generalize (2 : ℝ) ^ (-140 : ℤ) = w at *
    generalize (2 : ℝ) ^ (-23 : ℤ) = z at *
    linarith
  have hmasked : ∑ i ∈ Finset.Ico n B, |(e i).toReal| ≤ 2 ^ (-84 : ℤ) := by
    have h1 : ∑ i ∈ Finset.Ico n B, |(e i).toReal| ≤ ∑ i ∈ Finset.Ico n B, (2 : ℝ) ^ (-100 : ℤ) :=
      Finset.sum_le_sum fun i hi => by
        obtain ⟨h1, h2⟩ := Finset.mem_Ico.mp hi
        exact (hlm i h1 h2).2
    simp only [Finset.sum_const, Nat.card_Ico, nsmul_eq_mul] at h1
    have h2 : ((B - n : Nat) : ℝ) ≤ 2 ^ 16 := by
      have : B - n ≤ 2 ^ 16 := by omega
      exact_mod_cast this
    have h3 : ((B - n : Nat) : ℝ) * (2 : ℝ) ^ (-100 : ℤ) ≤ 2 ^ 16 * (2 : ℝ) ^ (-100 : ℤ) :=
      mul_le_mul_of_nonneg_right h2 (by positivity)
    have h4 : (2 : ℝ) ^ 16 * (2 : ℝ) ^ (-100 : ℤ) = 2 ^ (-84 : ℤ) := by norm_num
    linarith
  -- the exact sum of all lanes' values, and of their magnitudes
  have hsplitR : ∀ g : Nat → ℝ, ∑ i ∈ Finset.range B, g i =
      ∑ i ∈ Finset.range n, g i + ∑ i ∈ Finset.Ico n B, g i := by
    intro g
    rw [Finset.range_eq_Ico, Finset.range_eq_Ico]
    exact (Finset.sum_Ico_consecutive g (Nat.zero_le n) hnB).symm
  set E := ∑ i ∈ Finset.range B, (e i).toReal with hEdef
  set Eabs := ∑ i ∈ Finset.range B, |(e i).toReal| with hEabsdef
  have hEA : |E - A| ≤ 36.2 * uR * A := by
    have h1 : E - A = ∑ i ∈ Finset.range n, ((e i).toReal - Real.exp (s i).toReal) +
        ∑ i ∈ Finset.Ico n B, (e i).toReal := by
      rw [hEdef, hsplitR, hAdef, Finset.sum_sub_distrib]; ring
    have h2 := Finset.abs_sum_le_sum_abs (fun i => (e i).toReal - Real.exp (s i).toReal)
      (Finset.range n)
    have h3 := Finset.abs_sum_le_sum_abs (fun i => (e i).toReal) (Finset.Ico n B)
    have h4 : (2 : ℝ) ^ (-84 : ℤ) ≤ 0.1 * uR * (1 / 2) := by unfold uR; norm_num
    have h6 : 0.1 * uR * (1 / 2) ≤ 0.1 * uR * A := mul_le_mul_of_nonneg_left hA_lo (by positivity)
    rw [h1]
    calc _ ≤ |∑ i ∈ Finset.range n, ((e i).toReal - Real.exp (s i).toReal)| +
          |∑ i ∈ Finset.Ico n B, (e i).toReal| := abs_add_le _ _
      _ ≤ 36.2 * uR * A := by
        clear_value A
        generalize (2 : ℝ) ^ (-84 : ℤ) = w at *
        linarith
  have hEabs : Eabs ≤ (1 + 36.2 * uR) * A := by
    have h1 : Eabs = ∑ i ∈ Finset.range n, |(e i).toReal| + ∑ i ∈ Finset.Ico n B, |(e i).toReal| := by
      rw [hEabsdef, hsplitR]
    have h2 : ∑ i ∈ Finset.range n, |(e i).toReal| ≤
        ∑ i ∈ Finset.range n, (Real.exp (s i).toReal + |(e i).toReal - Real.exp (s i).toReal|) :=
      Finset.sum_le_sum fun i _ => by
        have := abs_add_le (Real.exp (s i).toReal) ((e i).toReal - Real.exp (s i).toReal)
        rw [add_sub_cancel, abs_of_pos (Real.exp_pos _)] at this
        exact this
    rw [Finset.sum_add_distrib, ← hAdef] at h2
    have h4 : (2 : ℝ) ^ (-84 : ℤ) ≤ 0.1 * uR * (1 / 2) := by unfold uR; norm_num
    have h6 : 0.1 * uR * (1 / 2) ≤ 0.1 * uR * A := mul_le_mul_of_nonneg_left hA_lo (by positivity)
    have h7 : 36.1 * uR * A + 0.1 * uR * A = 36.2 * uR * A := by ring
    rw [h1]
    clear_value A
    generalize (2 : ℝ) ^ (-84 : ℤ) = w at *
    linarith
  have hEabs_hi : Eabs ≤ 2 ^ 18 := by
    have h1 : Real.exp κ ≤ 2 := le_trans (Real.exp_le_exp.mpr hκ) exp_half_le
    have h2 : A ≤ 2 ^ 16 * 2 := le_trans hA_hi (mul_le_mul hnR h1 (Real.exp_pos _).le (by positivity))
    have h3 : (1 + 36.2 * uR) ≤ 2 := by unfold uR; norm_num
    nlinarith
  -- the tree sum
  obtain ⟨t, hcov, hdep, hev⟩ := hS
  have hD := le_trans hdep (sumBound_le hsd hB)
  have hEabsQ : ((List.range B).map fun i => |(e i).toRat|).sum ≤ 2 ^ 18 := by
    rw [list_range_sum_eq]
    have : ((∑ i ∈ Finset.range B, |(e i).toRat| : ℚ) : ℝ) = Eabs := by
      rw [hEabsdef]; push_cast; rfl
    have h2 : ((∑ i ∈ Finset.range B, |(e i).toRat| : ℚ) : ℝ) ≤ 2 ^ 18 := by rw [this]; exact hEabs_hi
    exact_mod_cast h2
  have hbnd : ∀ X : ℚ, 0 ≤ X → X ≤ 2 ^ 18 →
      Tree.errBound Format.binary32.u Format.binary32.eta 16 (B - 1) X ≤ 16.001 * 2 ^ (-24 : ℤ) * X + 2 ^ (-133 : ℤ) := by
    intro X hX0 hX1
    unfold Tree.errBound
    rw [Format.binary32_u, Format.binary32_eta]
    have hp := pow16_le
    have hp1 : (1 : ℚ) ≤ (1 + 2 ^ (-24 : ℤ)) ^ 16 := one_le_pow₀ (by norm_num)
    have hBq : ((B - 1 : Nat) : ℚ) ≤ 2 ^ 16 := by
      have : B - 1 ≤ 2 ^ 16 := by omega
      exact_mod_cast this
    have e1 : ((1 + (2 : ℚ) ^ (-24 : ℤ)) ^ 16 - 1) * X ≤ (16.001 * 2 ^ (-24 : ℤ)) * X :=
      mul_le_mul_of_nonneg_right (by linarith) hX0
    have e2 : (2 : ℚ) ^ (-150 : ℤ) * ((B - 1 : Nat) : ℚ) * (1 + 2 ^ (-24 : ℤ)) ^ 16 ≤
        2 ^ (-150 : ℤ) * 2 ^ 16 * 2 := by
      have h1 : (2 : ℚ) ^ (-150 : ℤ) * ((B - 1 : Nat) : ℚ) ≤ 2 ^ (-150 : ℤ) * 2 ^ 16 :=
        mul_le_mul_of_nonneg_left hBq (by positivity)
      have h2 : (1 + (2 : ℚ) ^ (-24 : ℤ)) ^ 16 ≤ 2 := by norm_num
      exact mul_le_mul h1 h2 (by positivity) (by positivity)
    have e3 : (2 : ℚ) ^ (-150 : ℤ) * 2 ^ 16 * 2 ≤ 2 ^ (-133 : ℤ) := by norm_num
    linarith
  have hEabs0 : (0 : ℚ) ≤ ((List.range B).map fun i => |(e i).toRat|).sum := by
    rw [list_range_sum_eq]; exact Finset.sum_nonneg fun i _ => abs_nonneg _
  have hroom : ((List.range B).map fun i => |(e i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta 16 (B - 1)
        ((List.range B).map fun i => |(e i).toRat|).sum ≤ Format.binary32.maxVal := by
    have h1 := hbnd _ hEabs0 hEabsQ
    rw [Format.binary32_maxVal]
    have : (16.001 : ℚ) * 2 ^ (-24 : ℤ) * ((List.range B).map fun i => |(e i).toRat|).sum ≤
        16.001 * 2 ^ (-24 : ℤ) * 2 ^ 18 := mul_le_mul_of_nonneg_left hEabsQ (by positivity)
    norm_num at this h1 ⊢
    linarith
  obtain ⟨hSf, hSerr⟩ := sum_err_depth b32_reg e hcov hD hev hefin hroom
  have hSerr2 := le_trans hSerr (hbnd _ hEabs0 hEabsQ)
  rw [list_range_sum_eq, list_range_sum_eq] at hSerr2
  have hSerrR : |S.toReal - E| ≤ 16.001 * uR * Eabs + 2 ^ (-133 : ℤ) := by
    have h1 : ((|S.toRat - ∑ i ∈ Finset.range B, (e i).toRat| : ℚ) : ℝ) ≤
        ((16.001 * 2 ^ (-24 : ℤ) * ∑ i ∈ Finset.range B, |(e i).toRat| + 2 ^ (-133 : ℤ) : ℚ) : ℝ) := by
      exact_mod_cast hSerr2
    push_cast at h1
    rw [hEdef, hEabsdef, FloatV.toReal_eq]
    have e1 : (2 : ℝ) ^ (-24 : ℤ) = uR := rfl
    simp only [FloatV.toReal_eq]
    rw [e1] at h1
    exact h1
  -- `S` against `A`
  have hSA : |S.toReal - A| ≤ 52.3 * uR * A := by
    have h1 : |S.toReal - A| ≤ |S.toReal - E| + |E - A| := by
      have := abs_add_le (S.toReal - E) (E - A)
      rwa [show S.toReal - E + (E - A) = S.toReal - A by ring] at this
    have h2 : 16.001 * uR * Eabs ≤ 16.001 * uR * ((1 + 36.2 * uR) * A) :=
      mul_le_mul_of_nonneg_left hEabs (by positivity)
    have h3 : (2 : ℝ) ^ (-133 : ℤ) ≤ 0.01 * uR * (1 / 2) := by unfold uR; norm_num
    have h6 : 0.01 * uR * (1 / 2) ≤ 0.01 * uR * A := mul_le_mul_of_nonneg_left hA_lo (by positivity)
    have h7 : 16.001 * uR * ((1 + 36.2 * uR) * A) ≤ 16.08 * uR * A := by
      have : 16.001 * (1 + 36.2 * uR) ≤ 16.08 := by unfold uR; norm_num
      have h8 := mul_le_mul_of_nonneg_right this hApos.le
      have h9 : 16.001 * uR * ((1 + 36.2 * uR) * A) = uR * (16.001 * (1 + 36.2 * uR) * A) := by ring
      have h10 : 16.08 * uR * A = uR * (16.08 * A) := by ring
      rw [h9, h10]
      exact mul_le_mul_of_nonneg_left h8 hu.le
    linarith
  have hε : 52.3 * uR ≤ 1 / 2 := by unfold uR; norm_num
  obtain ⟨hSpos, hlogSA⟩ := abs_log_sub_le hApos hε hSA
  have hlogSA2 : |Real.log S.toReal - Real.log A| ≤ 52.4 * uR := by
    refine le_trans hlogSA ?_
    rw [div_le_iff₀ (by unfold uR; norm_num)]
    have : 52.3 * uR ≤ 52.4 * uR * (1 - 52.3 * uR) := by
      have : 52.4 * 52.3 * uR ≤ 0.1 := by unfold uR; norm_num
      nlinarith
    linarith
  -- `log A` and `log S` are small
  have hlogA_hi : Real.log A ≤ 11.6 := by
    have h1 := Real.log_le_log hApos hA_hi
    have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
    rw [Real.log_mul hn0.ne' (Real.exp_pos κ).ne', Real.log_exp] at h1
    have h2 : Real.log (n : ℝ) ≤ Real.log (2 ^ 16) := Real.log_le_log hn0 hnR
    have h3 := log_pow16_le
    calc Real.log A ≤ Real.log n + κ := h1
      _ ≤ Real.log (2 ^ 16) + 1 / 2 := add_le_add h2 hκ
      _ ≤ 11.1 + 1 / 2 := by linarith
      _ = 11.6 := by norm_num
  have hlogA_lo : -1 ≤ Real.log A := by
    have h1 := Real.log_le_log (by norm_num) hA_lo
    have h2 : Real.log (1 / 2 : ℝ) = -Real.log 2 := by
      rw [one_div, Real.log_inv]
    have h3 := Real.log_two_lt_d9
    linarith
  have hlogS : |Real.log S.toReal| ≤ 11.7 := by
    have := (abs_le.mp hlogSA2)
    have h52 : 52.4 * uR ≤ 0.001 := by unfold uR; norm_num
    rw [abs_le]; constructor <;> linarith
  -- the log
  obtain ⟨ref, href, hacc⟩ := hlog
  simp only [MathFn.ref, Option.some.injEq] at href
  subst href
  have hmv := b32_maxVal_ge
  have hLr := hacc S L hL hSf (Set.mem_univ _) (by simp [MathFn.natDom]; exact hSpos) (by
    have : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.log S.toReal| ≤ (1 + 2 ^ (-22 : ℤ)) * 11.7 :=
      mul_le_mul_of_nonneg_left hlogS (by positivity)
    have h2 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * 11.7 + 2 ^ (-149 : ℤ) ≤ 2 ^ 127 := by norm_num
    generalize (2 : ℝ) ^ (-22 : ℤ) = w at *
    generalize (2 : ℝ) ^ (-149 : ℤ) = w' at *
    linarith)
  obtain ⟨hLf, hLerr⟩ := hLr
  refine ⟨hLf, ?_, ?_⟩
  · have h1 : (2 : ℝ) ^ (-22 : ℤ) * |Real.log S.toReal| ≤ 2 ^ (-22 : ℤ) * 11.7 :=
      mul_le_mul_of_nonneg_left hlogS (by positivity)
    have h2 : (2 : ℝ) ^ (-22 : ℤ) * 11.7 + 2 ^ (-149 : ℤ) + 52.4 * uR ≤ 100 * uR := by
      unfold uR; norm_num
    have h3 : |L.toReal - Real.log A| ≤ |L.toReal - Real.log S.toReal| +
        |Real.log S.toReal - Real.log A| := by
      have := abs_add_le (L.toReal - Real.log S.toReal) (Real.log S.toReal - Real.log A)
      rwa [show L.toReal - Real.log S.toReal + (Real.log S.toReal - Real.log A) =
        L.toReal - Real.log A by ring] at this
    clear_value A
    generalize (2 : ℝ) ^ (-22 : ℤ) = w at *
    generalize (2 : ℝ) ^ (-149 : ℤ) = w' at *
    linarith
  · have h1 : (2 : ℝ) ^ (-22 : ℤ) * |Real.log S.toReal| ≤ 2 ^ (-22 : ℤ) * 11.7 :=
      mul_le_mul_of_nonneg_left hlogS (by positivity)
    have h3 : |L.toReal| ≤ |Real.log S.toReal| + |L.toReal - Real.log S.toReal| := by
      have := abs_add_le (Real.log S.toReal) (L.toReal - Real.log S.toReal)
      rwa [add_sub_cancel] at this
    have h2 : (11.7 : ℝ) + 2 ^ (-22 : ℤ) * 11.7 + 2 ^ (-149 : ℤ) ≤ 12 := by norm_num
    clear_value A
    generalize (2 : ℝ) ^ (-22 : ℤ) = w at *
    generalize (2 : ℝ) ^ (-149 : ℤ) = w' at *
    linarith

end Claims.LogsumexpFwd
