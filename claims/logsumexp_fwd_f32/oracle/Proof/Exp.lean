import Proof.Lane

/-!
# The platform's `exp` of a lane, and the block's max

* `exp_valid`: for a finite `s ≤ 1/2`, `tl.exp(s)` is finite and within
  `(4u + 2u·|s|)·e^s + 2^-140` of `e^s` (`Device.ExpF32` at `X = |s|`);
* `exp_masked`: for `s = −∞` or a finite `s ≤ −2^126`, `tl.exp(s)` is finite and at most `2^-100`
  in magnitude;
* `max_facts`: the max of a block whose first `n` lanes are finite and whose others are `−∞` is a
  finite lane among the first `n`, at least every one of them.
-/

namespace Claims.LogsumexpFwd

open TVA

/-- The `exp` fact at `X = |s|`, unpacked. -/
lemma exp_acc {P : Platform} (hexp : Device.ExpF32 P) {s e : F32} (hs : s.isFinite = true)
    (hroom : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |s.toReal|)) * Real.exp s.toReal
      + 2 ^ (-140 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ))
    (he : P.math .exp .binary32 s e) :
    e.isFinite = true ∧ |e.toReal - Real.exp s.toReal| ≤
      (4 * uR + 2 * uR * |s.toReal|) * Real.exp s.toReal + 2 ^ (-140 : ℤ) := by
  obtain ⟨ref, href, h⟩ := hexp |s.toReal| (abs_nonneg _)
  simp only [MathFn.ref, Option.some.injEq] at href
  subst href
  have hdom : s.toReal ∈ Set.Icc (-|s.toReal|) |s.toReal| :=
    ⟨neg_abs_le _, le_abs_self _⟩
  have hr := h s e he hs hdom (by simp [MathFn.natDom])
    (by rw [abs_of_pos (Real.exp_pos _)]; exact hroom)
  rw [abs_of_pos (Real.exp_pos _)] at hr
  refine ⟨hr.1, ?_⟩
  have e1 : (2 : ℝ) ^ (-22 : ℤ) = 4 * uR := by unfold uR; norm_num
  have e2 : (2 : ℝ) ^ (-23 : ℤ) = 2 * uR := by unfold uR; norm_num
  rw [e1, e2] at hr
  have := hr.2
  linarith

/-- **`tl.exp` of a valid lane.** -/
lemma exp_valid {P : Platform} (hexp : Device.ExpF32 P) {s e : F32} (hs : s.isFinite = true)
    (hs2 : s.toReal ≤ 1 / 2) (he : P.math .exp .binary32 s e) :
    e.isFinite = true ∧ |e.toReal - Real.exp s.toReal| ≤
      (4 * uR + 2 * uR * |s.toReal|) * Real.exp s.toReal + 2 ^ (-140 : ℤ) := by
  refine exp_acc hexp hs ?_ he
  have hmv := b32_maxVal_ge
  have hX := abs_mul_exp_le hs2
  have hE : Real.exp s.toReal ≤ 2 := le_trans (Real.exp_le_exp.mpr hs2) exp_half_le
  have hE0 := (Real.exp_pos s.toReal).le
  have e1 : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |s.toReal|)) * Real.exp s.toReal
      = (1 + 2 ^ (-22 : ℤ)) * Real.exp s.toReal + 2 ^ (-23 : ℤ) * (|s.toReal| * Real.exp s.toReal) := by
    ring
  rw [e1]
  have h3 : (2 : ℝ) ^ (-23 : ℤ) * (|s.toReal| * Real.exp s.toReal) ≤
      2 ^ (-23 : ℤ) * (16 * Real.exp s.toReal + 2 ^ (-23 : ℤ)) :=
    mul_le_mul_of_nonneg_left hX (by positivity)
  have h4 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * Real.exp s.toReal ≤ (1 + 2 ^ (-22 : ℤ)) * 2 :=
    mul_le_mul_of_nonneg_left hE (by positivity)
  have h5 : (2 : ℝ) ^ (-23 : ℤ) * (16 * Real.exp s.toReal + 2 ^ (-23 : ℤ)) ≤
      2 ^ (-23 : ℤ) * (16 * 2 + 2 ^ (-23 : ℤ)) :=
    mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have h6 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * 2 + 2 ^ (-23 : ℤ) * (16 * 2 + 2 ^ (-23 : ℤ)) +
      2 ^ (-140 : ℤ) ≤ 2 ^ 127 := by norm_num
  linarith

/-- `e^{-X} ≤ 4/X²` for `X > 0`. -/
lemma exp_neg_le_sq {X : ℝ} (hX : 0 < X) : Real.exp (-X) ≤ 4 / X ^ 2 := by
  have h1 : X / 2 ≤ Real.exp (X / 2) := le_exp_self _
  have h2 : (X / 2) ^ 2 ≤ Real.exp (X / 2) ^ 2 := pow_le_pow_left₀ (by positivity) h1 2
  have h3 : Real.exp (X / 2) ^ 2 = Real.exp X := by
    rw [← Real.exp_nat_mul]; ring_nf
  rw [h3] at h2
  rw [Real.exp_neg, le_div_iff₀ (by positivity)]
  have h4 : (Real.exp X)⁻¹ * Real.exp X = 1 := inv_mul_cancel₀ (Real.exp_pos X).ne'
  have h5 : 0 < (Real.exp X)⁻¹ := inv_pos.mpr (Real.exp_pos X)
  nlinarith

/-- **`tl.exp` of a masked lane**: `−∞`, or a finite value at most `−2^126`. -/
lemma exp_masked {P : Platform} (hexp : Device.ExpF32 P) (hninf : Device.ExpF32NegInf P)
    {s e : F32} (hs : s = FloatV.negInf ∨ (s.isFinite = true ∧ s.toReal ≤ -2 ^ 126))
    (he : P.math .exp .binary32 s e) :
    e.isFinite = true ∧ |e.toReal| ≤ 2 ^ (-100 : ℤ) := by
  rcases hs with hs | ⟨hsf, hs2⟩
  · subst hs
    have := hninf e he
    subst this
    refine ⟨FloatV.isFinite_ofPat_zero b32_reg, ?_⟩
    rw [FloatV.toReal_ofPat_zero]; simp
  · set X := -s.toReal with hXdef
    have hX : (2 : ℝ) ^ 126 ≤ X := by rw [hXdef]; linarith
    have hX0 : 0 < X := lt_of_lt_of_le (by positivity) hX
    have habs : |s.toReal| = X := by rw [abs_of_neg (by linarith), hXdef]
    have hes : Real.exp s.toReal = Real.exp (-X) := by rw [hXdef, neg_neg]
    have hE := exp_neg_le_sq hX0
    have hX2 : (2 : ℝ) ^ 252 ≤ X ^ 2 := by
      have := pow_le_pow_left₀ (by positivity) hX 2
      rw [← pow_mul] at this; exact this
    have hE2 : Real.exp (-X) ≤ 2 ^ (-250 : ℤ) := by
      have h1 : 4 / X ^ 2 ≤ 4 / 2 ^ 252 := div_le_div_of_nonneg_left (by norm_num) (by positivity) hX2
      have h2 : (4 : ℝ) / 2 ^ 252 = 2 ^ (-250 : ℤ) := by norm_num
      exact le_trans hE (le_trans h1 h2.le)
    have hXE : X * Real.exp (-X) ≤ 2 ^ (-124 : ℤ) := by
      have h1 : X * Real.exp (-X) ≤ X * (4 / X ^ 2) := mul_le_mul_of_nonneg_left hE hX0.le
      have h2 : X * (4 / X ^ 2) = 4 / X := by field_simp
      have h3 : 4 / X ≤ 4 / 2 ^ 126 := div_le_div_of_nonneg_left (by norm_num) (by positivity) hX
      have h4 : (4 : ℝ) / 2 ^ 126 = 2 ^ (-124 : ℤ) := by norm_num
      exact le_trans h1 (by rw [h2]; exact le_trans h3 h4.le)
    have hE0 := (Real.exp_pos (-X)).le
    have hmv := b32_maxVal_ge
    have hroom : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |s.toReal|)) * Real.exp s.toReal
        + 2 ^ (-140 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [habs, hes]
      have e1 : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * X)) * Real.exp (-X)
          = (1 + 2 ^ (-22 : ℤ)) * Real.exp (-X) + 2 ^ (-23 : ℤ) * (X * Real.exp (-X)) := by ring
      rw [e1]
      have h1 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * Real.exp (-X) ≤ (1 + 2 ^ (-22 : ℤ)) * 2 ^ (-250 : ℤ) :=
        mul_le_mul_of_nonneg_left hE2 (by positivity)
      have h2 : (2 : ℝ) ^ (-23 : ℤ) * (X * Real.exp (-X)) ≤ 2 ^ (-23 : ℤ) * 2 ^ (-124 : ℤ) :=
        mul_le_mul_of_nonneg_left hXE (by positivity)
      have h3 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * 2 ^ (-250 : ℤ) + 2 ^ (-23 : ℤ) * 2 ^ (-124 : ℤ) +
          2 ^ (-140 : ℤ) ≤ 2 ^ 127 := by norm_num
      linarith
    obtain ⟨hef, herr⟩ := exp_acc hexp hsf hroom he
    refine ⟨hef, ?_⟩
    rw [habs, hes] at herr
    have hu := uR_pos
    have h1 : (4 * uR + 2 * uR * X) * Real.exp (-X) = 4 * uR * Real.exp (-X) + 2 * uR * (X * Real.exp (-X)) := by ring
    rw [h1] at herr
    have h2 : 4 * uR * Real.exp (-X) ≤ 4 * uR * 2 ^ (-250 : ℤ) :=
      mul_le_mul_of_nonneg_left hE2 (by positivity)
    have h3 : 2 * uR * (X * Real.exp (-X)) ≤ 2 * uR * 2 ^ (-124 : ℤ) :=
      mul_le_mul_of_nonneg_left hXE (by positivity)
    have h4 : |e.toReal| ≤ Real.exp (-X) + |e.toReal - Real.exp (-X)| := by
      have := abs_add_le (Real.exp (-X)) (e.toReal - Real.exp (-X))
      rw [add_sub_cancel, abs_of_nonneg hE0] at this; exact this
    have h5 : (2 : ℝ) ^ (-250 : ℤ) + (4 * uR * 2 ^ (-250 : ℤ) + 2 * uR * 2 ^ (-124 : ℤ)) +
        2 ^ (-140 : ℤ) ≤ 2 ^ (-100 : ℤ) := by unfold uR; norm_num
    linarith

/-! ## The block's max -/

/-- **The block's max.** If the first `n` lanes (`n ≥ 1`) of `v` are finite and the others are
`−∞`, a max along a covering tree is one of the first `n` lanes, finite, and at least each of
them. -/
lemma max_facts {n B : Nat} (hn : 0 < n) (hnB : n ≤ B) {v : Tensor [B] F32} {mv : F32}
    (hfin : ∀ (i : Nat) (hi : i < B), i < n → (v.get1 ⟨i, hi⟩).isFinite = true)
    (hneg : ∀ (i : Nat) (hi : i < B), ¬ i < n → v.get1 ⟨i, hi⟩ = FloatV.negInf)
    (hmv : ∃ t : Tree, t.Covers B ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < B then v.get1 ⟨j, h⟩ else default) mv) :
    ∃ (i0 : Nat) (hi0 : i0 < B), i0 < n ∧ mv = v.get1 ⟨i0, hi0⟩ ∧ mv.isFinite = true ∧
      ∀ (i : Nat) (hi : i < B), i < n → (v.get1 ⟨i, hi⟩).toRat ≤ mv.toRat := by
  obtain ⟨t, hcov, hev⟩ := hmv
  have hleaf_lt : ∀ i ∈ t.leaves, i < B := by
    intro i hi
    have h : i ∈ List.range B := hcov.subset hi
    simpa using h
  have hlanenan : ∀ (i : Nat) (hi : i < B), (v.get1 ⟨i, hi⟩).isNaN = false := by
    intro i hi
    by_cases hin : i < n
    · exact (FloatV.isFinite_iff.mp (hfin i hi hin)).1
    · rw [hneg i hi hin]; exact negInf_isNaN32
  have hxnan : ∀ i ∈ t.leaves,
      ((fun j => if h : j < B then v.get1 ⟨j, h⟩ else default) i).isNaN = false := by
    intro i hi
    have hiB := hleaf_lt i hi
    show (if h : i < B then v.get1 ⟨i, h⟩ else default).isNaN = false
    rw [dite_eq_left hiB]
    exact hlanenan i hiB
  obtain ⟨⟨i0, hi0, hmv_eq⟩, hle⟩ := Tree.eval_maxRel hev hxnan
  have hi0B := hleaf_lt i0 hi0
  have hmv_lane : mv = v.get1 ⟨i0, hi0B⟩ := by
    rw [hmv_eq]; show (if h : i0 < B then v.get1 ⟨i0, h⟩ else default) = _
    rw [dite_eq_left hi0B]
  have hmvnan : mv.isNaN = false := by rw [hmv_lane]; exact hlanenan i0 hi0B
  have hle_all : ∀ (i : Nat) (hi : i < B), (v.get1 ⟨i, hi⟩).toRat ≤ mv.toRat := by
    intro i hi
    have hmem : i ∈ t.leaves := hcov.symm.subset (List.mem_range.mpr hi)
    have hh := hle i hmem
    simp only [ite_true, dite_eq_left hi] at hh
    exact (FloatV.le_iff (hlanenan i hi) hmvnan).mp hh
  have h0B : 0 < B := lt_of_lt_of_le hn hnB
  have hi0n : i0 < n := by
    by_contra hin
    have hm : mv = FloatV.negInf := by rw [hmv_lane, hneg i0 hi0B hin]
    have h1 := hle_all 0 h0B
    have hf0 := hfin 0 h0B hn
    have hbound := FloatV.abs_toRat_le_maxVal b32_reg hf0
    rw [abs_le] at hbound
    rw [hm, negInf_toRat] at h1
    linarith [hbound.1, maxVal_lt_two_pow]
  exact ⟨i0, hi0B, hi0n, hmv_lane, by rw [hmv_lane]; exact hfin i0 hi0B hi0n,
    fun i hi _ => hle_all i hi⟩

end Claims.LogsumexpFwd
