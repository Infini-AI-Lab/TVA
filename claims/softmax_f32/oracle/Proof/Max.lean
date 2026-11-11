import Proof.Exp

/-!
# The maximum of the block and the exact quantities of a row

The max `mv` of the block is a lane's value, and a finite input on a `RowOk` row (`max_facts`). The
exact quantities `tR`, `ER`, `ZR` of the row relative to `mv`, and the claim's `softmaxRef`,
`softmaxMax` and `softmaxGap` written in them.
-/
namespace Claims.Softmax

open TVA

/-! ## The maximum of the block -/

/-- A format fact about `float32`. -/
lemma b32_reg : Format.binary32.Regular := Format.binary32_regular
/-- A format fact about `float32`. -/
lemma b32_ieee : Format.binary32.specials = .ieee := rfl

/-- The rational value of `-∞` in `float32` is `-2^128`. -/
lemma negInf_toRat : (FloatV.negInf : F32).toRat = -((2 : ℚ) ^ 128) := by
  show Format.binary32.toRat (FloatV.negInf : F32).pat = _
  rw [FloatV.negInf_pat]
  have hisinf : Format.binary32.isInf
      (Format.binary32.setSign true Format.binary32.infMag) = true := by
    rw [Format.isInf_setSign]; exact Format.isInf_infMag b32_ieee
  rw [Format.toRat_of_isInf b32_reg hisinf, Format.sign_setSign]
  simp only [ite_true]
  have hb : Format.binary32.bias + 1 = (128 : Int) := by decide
  rw [hb]
  norm_num

/-- The largest finite `float32` is below `2^128`. -/
lemma maxVal_lt_two_pow : Format.binary32.maxVal < (2 : ℚ) ^ 128 := by
  have h := Format.maxVal_lt_inf b32_reg b32_ieee
  have hb : Format.binary32.bias = 127 := rfl
  rw [hb] at h
  simpa using h

/-- On a `RowOk` row, the block's max `mv` is the finite input of the row's largest lane, and every
row lane is at most it. -/
lemma max_facts {n B : Nat} (hn : 0 < n) (hnB : n ≤ B)
    {v : Tensor [B] F32} {mv : F32} {xs : Nat → F32}
    (hv : ∀ (i : Nat) (hi : i < B),
      v.get1 ⟨i, hi⟩ = if _h : i < n then xs i else FloatV.negInf)
    (hmv : ∃ t : Tree, t.Covers B ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < B then v.get1 ⟨j, h⟩ else default) mv)
    (hok : RowOk xs n) :
    ∃ i0 : Nat, i0 < n ∧ i0 < B ∧ mv = xs i0 ∧ mv.isFinite = true ∧ mv.isNaN = false ∧
      (∀ (i : Nat), i < n → (xs i).isFinite = true ∨ xs i = FloatV.negInf) ∧
      (∀ (i : Nat), i < n → (xs i).toRat ≤ mv.toRat) := by
  obtain ⟨t, hcov, hev⟩ := hmv
  have hleaf_lt : ∀ i ∈ t.leaves, i < B := by
    intro i hi
    have h : i ∈ List.range B := hcov.subset hi
    simpa using h
  have hxnan : ∀ i ∈ t.leaves,
      ((fun j => if h : j < B then v.get1 ⟨j, h⟩ else default) i).isNaN = false := by
    intro i hi
    have hiB := hleaf_lt i hi
    have hxi : ((fun j => if h : j < B then v.get1 ⟨j, h⟩ else default) i)
        = (if h : i < B then v.get1 ⟨i, h⟩ else default) := rfl
    rw [hxi, dite_eq_left hiB, hv i hiB]
    by_cases hin : i < n
    · rw [dite_eq_left hin]
      rcases hok.1 i hin with hf' | hni
      · exact (FloatV.isFinite_iff.mp hf').1
      · rw [hni]; exact FloatV.negInf_isNaN b32_ieee
    · rw [dite_eq_right hin]; exact FloatV.negInf_isNaN b32_ieee
  obtain ⟨⟨i0, hi0, hmv_eq⟩, hle⟩ := Tree.eval_maxRel hev hxnan
  have hi0B := hleaf_lt i0 hi0
  have hmvnan : mv.isNaN = false := by
    rw [hmv_eq]; exact hxnan i0 hi0
  have hmv_lane : v.get1 ⟨i0, hi0B⟩ = mv := by
    rw [hmv_eq, dite_eq_left hi0B]
  -- a finite lane of the row
  obtain ⟨k0, hk0, hk0fin⟩ := hok.2 hn
  have hk0B : k0 < B := lt_of_lt_of_le hk0 hnB
  have hle_at : ∀ (i : Nat) (hi : i < B), FloatV.le (v.get1 ⟨i, hi⟩) mv = true := by
    intro i hi
    have hmem : i ∈ t.leaves := hcov.symm.subset (List.mem_range.mpr hi)
    have hh := hle i hmem
    simpa only [ite_true, dite_eq_left hi] using hh
  have hlanenan : ∀ (i : Nat) (hi : i < B), (v.get1 ⟨i, hi⟩).isNaN = false := by
    intro i hi
    rw [hv i hi]
    by_cases hin : i < n
    · rw [dite_eq_left hin]
      rcases hok.1 i hin with hf' | hni
      · exact (FloatV.isFinite_iff.mp hf').1
      · rw [hni]; exact FloatV.negInf_isNaN b32_ieee
    · rw [dite_eq_right hin]; exact FloatV.negInf_isNaN b32_ieee
  have hle_all : ∀ (i : Nat) (hi : i < B), (v.get1 ⟨i, hi⟩).toRat ≤ mv.toRat := by
    intro i hi
    exact (FloatV.le_iff (hlanenan i hi) hmvnan).mp (hle_at i hi)
  -- `mv` is not `-∞`: a finite lane is at most it
  have hmv_neginf : mv ≠ FloatV.negInf := by
    intro hneg
    have hration := hle_all k0 hk0B
    have hinft : mv.toRat = -((2 : ℚ) ^ 128) := by
      rw [hneg]; exact negInf_toRat
    have hk0v : (v.get1 ⟨k0, hk0B⟩).isFinite = true := by
      rw [hv k0 hk0B, dite_eq_left hk0]; exact hk0fin
    have hbound := FloatV.abs_toRat_le_maxVal b32_reg hk0v
    rw [abs_le] at hbound
    rw [hinft] at hration
    linarith [hbound.1, maxVal_lt_two_pow]
  have hi0n : i0 < n := by
    by_contra hin
    apply hmv_neginf
    rw [← hmv_lane, hv i0 hi0B, dite_eq_right hin]
  have hmv_eq_xs : mv = xs i0 := by
    rw [← hmv_lane, hv i0 hi0B, dite_eq_left hi0n]
  have hi0fin : (xs i0).isFinite = true := by
    rcases hok.1 i0 hi0n with h | h
    · exact h
    · exact absurd (hmv_eq_xs.trans h) hmv_neginf
  have hmvfin : mv.isFinite = true := by
    rw [hmv_eq_xs]; exact hi0fin
  exact ⟨i0, hi0n, hi0B, hmv_eq_xs, hmvfin, hmvnan, hok.1, by
    intro i hi
    have hiB : i < B := lt_of_lt_of_le hi hnB
    have h := hle_all i hiB
    rwa [hv i hiB, dite_eq_left hi] at h⟩

/-! ## The exact quantities of one row

`mR` is the max `mv` as a real, `tR mv x = x - mv`, `ER mv x` is `e^(x - mv)` (`0` for `-∞`), and
`ZR` is the exact sum of the `ER` over the row. -/

/-- The block maximum `mv` as a real. -/
noncomputable def mR (mv : F32) : ℝ := (mv.toRat : ℝ)

/-- `x - mv` as a real: the exact exponent of lane `x`. -/
noncomputable def tR (mv x : F32) : ℝ := (x.toRat : ℝ) - mR mv

/-- `e^(x - mv)`, and `0` for `x = -∞`. -/
noncomputable def ER (mv x : F32) : ℝ :=
  if x = FloatV.negInf then 0 else Real.exp (tR mv x)

/-- The exact sum of `ER mv (xs k)` over the row. -/
noncomputable def ZR (mv : F32) (xs : Nat → F32) (n : Nat) : ℝ :=
  ∑ k ∈ Finset.range n, ER mv (xs k)

/-- `e^a = e^b · e^(a - b)`. -/
lemma exp_eq_mul_exp_sub (a b : ℝ) : Real.exp a = Real.exp b * Real.exp (a - b) := by
  rw [← Real.exp_add]; ring_nf

/-- A property of `ER`. -/
lemma ER_eq_exp {mv x : F32} (h : x ≠ FloatV.negInf) : ER mv x = Real.exp (tR mv x) := by
  unfold ER; rw [ite_eq_right h]

/-- A property of `ER`. -/
lemma ER_eq_zero {mv x : F32} (h : x = FloatV.negInf) : ER mv x = 0 := by
  unfold ER; rw [ite_eq_left h]

/-- A property of `ER`. -/
lemma ER_nonneg (mv x : F32) : 0 ≤ ER mv x := by
  unfold ER
  by_cases h : x = FloatV.negInf
  · rw [ite_eq_left h]
  · rw [ite_eq_right h]; exact (Real.exp_pos _).le

/-- A property of `ER`. -/
lemma ER_pos {mv x : F32} (h : x ≠ FloatV.negInf) : 0 < ER mv x := by
  rw [ER_eq_exp h]; exact Real.exp_pos _

/-- A property of `ER`. -/
lemma ER_le_one {mv x : F32} (h : x ≠ FloatV.negInf)
    (hle : (x.toRat : ℝ) ≤ mR mv) : ER mv x ≤ 1 := by
  rw [ER_eq_exp h, ← Real.exp_zero]
  apply Real.exp_le_exp.mpr
  show (x.toRat : ℝ) - mR mv ≤ 0
  linarith

/-- A property of `ER`. -/
lemma ER_self {mv : F32} (h : mv ≠ FloatV.negInf) : ER mv mv = 1 := by
  rw [ER_eq_exp h]
  have h0 : tR mv mv = 0 := by
    show (mv.toRat : ℝ) - mR mv = 0
    unfold mR; ring
  rw [h0, Real.exp_zero]

/-- The claim's `expIn x` is `e^mv · ER mv x`. -/
lemma expIn_eq (mv x : F32) : expIn x = Real.exp (mR mv) * ER mv x := by
  unfold expIn ER
  by_cases h : x = FloatV.negInf
  · rw [ite_eq_left h, ite_eq_left h]; ring
  · rw [ite_eq_right h, ite_eq_right h]
    exact exp_eq_mul_exp_sub _ _

/-- The exact denominator is `e^mv · ZR`. -/
lemma expIn_sum (mv : F32) (xs : Nat → F32) (n : Nat) :
    (∑ k ∈ Finset.range n, expIn (xs k)) = Real.exp (mR mv) * ZR mv xs n := by
  unfold ZR
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun k _ => expIn_eq mv (xs k))

/-- The exact softmax is `ER / ZR`. -/
lemma softmaxRef_eq {mv : F32} {xs : Nat → F32} {n j : Nat} (_hz : ZR mv xs n ≠ 0) :
    softmaxRef xs n j = ER mv (xs j) / ZR mv xs n := by
  unfold softmaxRef
  rw [expIn_eq mv (xs j), expIn_sum mv xs n,
    mul_div_mul_left (ER mv (xs j)) (ZR mv xs n) (Real.exp_pos (mR mv)).ne']

/-- The sum `ZR` is at least `1` (the max lane contributes `1`). -/
lemma one_le_ZR {mv : F32} {xs : Nat → F32} {n i0 : Nat} (hi0 : i0 < n)
    (hmax : ER mv (xs i0) = 1) : 1 ≤ ZR mv xs n := by
  unfold ZR
  have h := Finset.single_le_sum (f := fun k => ER mv (xs k))
    (fun i _ => ER_nonneg mv (xs i)) (Finset.mem_range.mpr hi0)
  rwa [hmax] at h

/-- The sum `ZR` is at most `n`. -/
lemma ZR_le_n {mv : F32} {xs : Nat → F32} {n : Nat}
    (hle1 : ∀ k, k < n → ER mv (xs k) ≤ 1) : ZR mv xs n ≤ (n : ℝ) := by
  unfold ZR
  have h := Finset.sum_le_sum (fun k hk => hle1 k (Finset.mem_range.mp hk))
  simpa using h

/-- The row's largest softmax value is `1 / ZR`. -/
lemma softmaxMax_eq {mv : F32} {xs : Nat → F32} {n i0 : Nat} (hi0 : i0 < n)
    (hpos : 0 < ZR mv xs n) (hmax : ER mv (xs i0) = 1)
    (hle1 : ∀ k, k < n → ER mv (xs k) ≤ 1) :
    softmaxMax xs n = 1 / ZR mv xs n := by
  have hz := hpos.ne'
  have hn : 0 < n := lt_of_le_of_lt (Nat.zero_le i0) hi0
  have : Nonempty (Fin n) := ⟨⟨i0, hi0⟩⟩
  unfold softmaxMax
  apply le_antisymm
  · apply ciSup_le
    intro k
    have h1 : softmaxRef xs n (k : Nat) ≤ 1 / ZR mv xs n := by
      rw [softmaxRef_eq hz]
      exact div_le_div_of_nonneg_right (hle1 (k : Nat) k.isLt) hpos.le
    exact h1
  · have hb : BddAbove (Set.range (fun k : Fin n => softmaxRef xs n (k : Nat))) := by
      refine ⟨1 / ZR mv xs n, ?_⟩
      rintro _ ⟨k, rfl⟩
      have h1 : softmaxRef xs n (k : Nat) ≤ 1 / ZR mv xs n := by
        rw [softmaxRef_eq hz]
        exact div_le_div_of_nonneg_right (hle1 (k : Nat) k.isLt) hpos.le
      exact h1
    have h := le_ciSup hb ⟨i0, hi0⟩
    simp only at h
    rwa [softmaxRef_eq hz, hmax] at h

/-- The claim's gap of a finite lane is `-tR`. -/
lemma softmaxGap_eq_fin {mv : F32} {xs : Nat → F32} {n i0 j : Nat} (hi0 : i0 < n)
    (hpos : 0 < ZR mv xs n) (hmax : ER mv (xs i0) = 1)
    (hle1 : ∀ k, k < n → ER mv (xs k) ≤ 1) (_hj : j < n)
    (hne : xs j ≠ FloatV.negInf) :
    softmaxGap xs n j = -tR mv (xs j) := by
  have hz := hpos.ne'
  have hn : 0 < n := lt_of_le_of_lt (Nat.zero_le i0) hi0
  unfold softmaxGap
  rw [softmaxMax_eq hi0 hpos hmax hle1, softmaxRef_eq hz]
  have hER : ER mv (xs j) ≠ 0 := (ER_pos hne).ne'
  have h1 : (1 / ZR mv xs n) / (ER mv (xs j) / ZR mv xs n) = 1 / ER mv (xs j) := by
    field_simp
  rw [h1, Real.log_div one_ne_zero hER, Real.log_one, ER_eq_exp hne, Real.log_exp]
  ring

/-- The claim's gap of a `-∞` lane is `0`. -/
lemma softmaxGap_eq_neginf {mv : F32} {xs : Nat → F32} {n j : Nat}
    (hz : ZR mv xs n ≠ 0) (hne : xs j = FloatV.negInf) : softmaxGap xs n j = 0 := by
  unfold softmaxGap
  rw [softmaxRef_eq hz, hne, ER_eq_zero rfl, zero_div, div_zero, Real.log_zero]

end Claims.Softmax
