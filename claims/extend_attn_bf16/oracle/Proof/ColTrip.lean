import Proof.DotLemmas
import Proof.ScoreLane

/-!
# The accumulator update of one element, from the float facts

`col_pref` (`acc * re_scale + dot * v_scale`) and `col_ext` (`tl.dot_add (p16, v, acc * re_scale)`):
the new accumulator element against `re_scale · acc + Σ_valid p16 · v`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- The sum of the lifted magnitudes of a row's lanes: masked lanes (`p16 = 0`) count `0`; a valid
lane counts at most `|p16||v| + 2^-126 (|p16| + |v| + 2^-126)`. -/
theorem liftA_le {valid : Fin 64 → Bool} {p16 v : Fin 64 → BF16}
    (hz : ∀ n, valid n = false → (p16 n).toReal = 0) (hpb : ∀ n, valid n = true → |(p16 n).toReal| ≤ 2) :
    ∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ) ≤
      (∑ n : Fin 64, if valid n = true then |(p16 n).toReal| * |(v n).toReal| else 0) +
        1 / 2 ^ 118 * (1 + ∑ n : Fin 64, if valid n = true then |(v n).toReal| else 0) := by
  have hterm : ∀ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ) ≤
      (if valid n = true then |(p16 n).toReal| * |(v n).toReal| else 0) +
        (if valid n = true then 1 / 2 ^ 125 + 1 / 2 ^ 126 * |(v n).toReal| + 1 / 2 ^ 252 else 0) := by
    intro n
    by_cases hv : valid n = true
    · simp only [hv, ite_true]
      have h := DotLift.liftMag_mul_le .bf16 .bf16 (p16 n).toRat (v n).toRat
      have hf : Device.liftFloor .bf16 = 1 / 2 ^ 126 := by
        rw [Device.liftFloor_bf16]; norm_num
      rw [hf] at h
      have h2 := hpb n hv
      have h' : ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ) ≤
          |(p16 n).toReal * (v n).toReal| + (1 / 2 ^ 126 : ℝ) * |(p16 n).toReal| +
            1 / 2 ^ 126 * |(v n).toReal| + 1 / 2 ^ 126 * (1 / 2 ^ 126) := by
        have := (Rat.cast_le (K := ℝ)).mpr h
        push_cast at this
        unfold FloatV.toReal
        push_cast
        simpa [abs_mul] using this
      rw [abs_mul] at h'
      have : (1 / 2 ^ 126 : ℝ) * |(p16 n).toReal| ≤ 1 / 2 ^ 126 * 2 :=
        mul_le_mul_of_nonneg_left h2 (by positivity)
      have e1 : (1 : ℝ) / 2 ^ 126 * 2 = 1 / 2 ^ 125 := by norm_num
      have e2 : (1 : ℝ) / 2 ^ 126 * (1 / 2 ^ 126) = 1 / 2 ^ 252 := by norm_num
      linarith
    · have hvf : valid n = false := by simpa using hv
      simp only [hv]
      have : (p16 n).toRat = 0 := by
        have := hz n hvf
        unfold FloatV.toReal at this
        exact_mod_cast this
      rw [this]; simp
  calc _ ≤ ∑ n : Fin 64, ((if valid n = true then |(p16 n).toReal| * |(v n).toReal| else 0) +
        (if valid n = true then 1 / 2 ^ 125 + 1 / 2 ^ 126 * |(v n).toReal| + 1 / 2 ^ 252 else 0)) :=
        Finset.sum_le_sum fun n _ => hterm n
    _ = (∑ n : Fin 64, if valid n = true then |(p16 n).toReal| * |(v n).toReal| else 0) +
        ∑ n : Fin 64, (if valid n = true then 1 / 2 ^ 125 + 1 / 2 ^ 126 * |(v n).toReal| + 1 / 2 ^ 252 else 0) :=
        Finset.sum_add_distrib
    _ ≤ _ := by
        have h2 : ∑ n : Fin 64, (if valid n = true then 1 / 2 ^ 125 + 1 / 2 ^ 126 * |(v n).toReal| + 1 / 2 ^ 252 else 0)
            ≤ 64 * (1 / 2 ^ 125 + 1 / 2 ^ 252) + 1 / 2 ^ 126 * ∑ n : Fin 64, (if valid n = true then |(v n).toReal| else 0) := by
          have e : ∀ n : Fin 64, (if valid n = true then 1 / 2 ^ 125 + 1 / 2 ^ 126 * |(v n).toReal| + 1 / 2 ^ 252 else (0 : ℝ)) ≤
              (1 / 2 ^ 125 + 1 / 2 ^ 252) + 1 / 2 ^ 126 * (if valid n = true then |(v n).toReal| else 0) := by
            intro n
            by_cases hv : valid n = true
            · simp only [hv, ite_true]; linarith
            · simp only [hv]; norm_num
          calc _ ≤ ∑ n : Fin 64, ((1 / 2 ^ 125 + 1 / 2 ^ 252) + 1 / 2 ^ 126 * (if valid n = true then |(v n).toReal| else 0)) :=
                Finset.sum_le_sum fun n _ => e n
            _ = 64 * (1 / 2 ^ 125 + 1 / 2 ^ 252) + 1 / 2 ^ 126 * ∑ n : Fin 64, (if valid n = true then |(v n).toReal| else 0) := by
                rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
                  Fintype.card_fin]
                simp; ring
        have hV : 0 ≤ ∑ n : Fin 64, (if valid n = true then |(v n).toReal| else 0) :=
          Finset.sum_nonneg fun n _ => by by_cases hv : valid n = true <;> simp [hv, abs_nonneg]
        have h3 : (64 : ℝ) * (1 / 2 ^ 125 + 1 / 2 ^ 252) ≤ 1 / 2 ^ 118 := by norm_num
        have h4 : (1 : ℝ) / 2 ^ 126 * ∑ n : Fin 64, (if valid n = true then |(v n).toReal| else 0) ≤
            1 / 2 ^ 118 * ∑ n : Fin 64, (if valid n = true then |(v n).toReal| else 0) :=
          mul_le_mul_of_nonneg_right (by norm_num) hV
        linarith

/-- The sums of a row's lanes over the valid ones, as used by the accumulator's error. -/
def sgSum (valid : Fin 64 → Bool) (p16 v : Fin 64 → BF16) : ℝ :=
  ∑ n : Fin 64, if valid n = true then (p16 n).toReal * (v n).toReal else 0
/-- `Σ_valid |p16||v|`. -/
def lpSum (valid : Fin 64 → Bool) (p16 v : Fin 64 → BF16) : ℝ :=
  ∑ n : Fin 64, if valid n = true then |(p16 n).toReal| * |(v n).toReal| else 0
/-- `Σ_valid |v|`. -/
def v1Sum (valid : Fin 64 → Bool) (v : Fin 64 → BF16) : ℝ :=
  ∑ n : Fin 64, if valid n = true then |(v n).toReal| else 0

/-- If invalid lanes carry zero probability, the full 64-lane dot product equals the masked sum `sgSum`. -/
theorem sgSum_eq {valid : Fin 64 → Bool} {p16 v : Fin 64 → BF16}
    (hz : ∀ n, valid n = false → (p16 n).toReal = 0) :
    ∑ n : Fin 64, (p16 n).toReal * (v n).toReal = sgSum valid p16 v := by
  unfold sgSum
  refine Finset.sum_congr rfl fun n _ => ?_
  by_cases hv : valid n = true
  · simp [hv]
  · have hvf : valid n = false := by simpa using hv
    simp [hvf, hz n hvf]

/-- The absolute value of the masked signed sum is at most the masked sum of absolute values. -/
theorem sgSum_le_lpSum (valid : Fin 64 → Bool) (p16 v : Fin 64 → BF16) :
    |sgSum valid p16 v| ≤ lpSum valid p16 v := by
  unfold sgSum lpSum
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun n _ => ?_)
  by_cases hv : valid n = true
  · simp [hv, abs_mul]
  · simp [hv]

/-- The masked sum of absolute values `lpSum` is nonnegative. -/
theorem lpSum_nonneg (valid : Fin 64 → Bool) (p16 v : Fin 64 → BF16) : 0 ≤ lpSum valid p16 v :=
  Finset.sum_nonneg fun n _ => by by_cases hv : valid n = true <;> simp [hv]; positivity

/-- The masked sum `v1Sum` is nonnegative. -/
theorem v1Sum_nonneg (valid : Fin 64 → Bool) (v : Fin 64 → BF16) : 0 ≤ v1Sum valid v :=
  Finset.sum_nonneg fun n _ => by by_cases hv : valid n = true <;> simp [hv]

/-- A `float32` product of finite operands with room: finite, within one rounding. -/
theorem mul_real {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hb : |x.toReal * y.toReal| ≤ 2 ^ 126) (h : FloatV.NaNEq (FloatV.mul x y) r) :
    r.isFinite = true ∧ |r.toReal - x.toReal * y.toReal| ≤ 1 / 2 ^ 24 * |x.toReal * y.toReal| + 1 / 2 ^ 150 := by
  have hm : (2 : ℝ) ^ 126 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal]; norm_num
  obtain ⟨h1, h2⟩ := FloatV.mul_err_real Format.binary32_regular hx hy (hb.trans hm)
  obtain ⟨hf, -, hr⟩ := h.exact_of_isFinite h1
  refine ⟨hf, ?_⟩
  rw [hr]
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  have e24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have e150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  rw [e24, e150] at h2
  have := mul_comm (|x.toReal * y.toReal|) (1 / 2 ^ 24 : ℝ)
  linarith

/-- A fusable `float32` addition of finite operands with room: finite, within `3u (|x| + |y|) + 4η`. -/
theorem addrel_real {fusion : Bool} {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hb : |x.toReal| + |y.toReal| ≤ 2 ^ 126) (h : AddRel fusion x y r) :
    r.isFinite = true ∧ |r.toReal - (x.toReal + y.toReal)| ≤
      3 * (1 / 2 ^ 24) * (|x.toReal| + |y.toReal|) + 4 * (1 / 2 ^ 150) := by
  have hM : |x.toRat| + |y.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have : ((|x.toRat| + |y.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast; unfold FloatV.toReal at hb; exact hb
    have h2 : (|x.toRat| + |y.toRat| : ℚ) ≤ 2 ^ 126 := by exact_mod_cast this
    refine h2.trans ?_
    rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num
  obtain ⟨hf, he⟩ := FloatV.AddRel.err Format.binary32_regular (by decide) (by rw [Format.binary32_u]; norm_num)
    h hx hy hM
  refine ⟨hf, ?_⟩
  have he' := (Rat.cast_le (K := ℝ)).mpr he
  rw [Format.binary32_u, Format.binary32_eta] at he'
  push_cast at he'
  unfold FloatV.toReal
  norm_num at he' ⊢
  linarith

/-- **A cached-part trip's accumulator element** `acc * re_scale + dot * v_scale`. -/
theorem col_pref {P : Platform} (_hP : Device P) {fusion : Bool} {valid : Fin 64 → Bool}
    {p16 v : Fin 64 → BF16} {a re x D y a' : F32}
    (hz : ∀ n, valid n = false → (p16 n).toReal = 0) (hpb : ∀ n, valid n = true → |(p16 n).toReal| ≤ 2)
    (hvb : ∀ n, valid n = true → |(v n).toReal| ≤ 2 ^ 100)
    (haf : a.isFinite = true) (hab : |a.toReal| ≤ 2 ^ 119) (hre : re.isFinite = true)
    (hreb : |re.toReal| ≤ 2) (hDf : D.isFinite = true)
    (hD : |D.toReal - ∑ n : Fin 64, (p16 n).toReal * (v n).toReal| ≤
      66 / 2 ^ 22 * (∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat *
        Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ)) + 66 / 2 ^ 126)
    (hx : FloatV.NaNEq (FloatV.mul a re) x) (hy : FloatV.NaNEq (FloatV.mul D (FloatV.hostF32 one64)) y)
    (ha' : AddRel fusion x y a') :
    a'.isFinite = true ∧
      |a'.toReal - (re.toReal * a.toReal + sgSum valid p16 v)| ≤
        epsP1 * |re.toReal * a.toReal| + eps2 * lpSum valid p16 v + eps1 * (1 + v1Sum valid v) := by
  classical
  obtain ⟨hks_f, hks_e⟩ := hostF32_err one64_spec.1 (by rw [one64_spec.2]; norm_num)
    (by rw [one64_spec.2]; norm_num)
  rw [one64_spec.2] at hks_e
  rw [mul_one] at hks_e
  have hSg := sgSum_eq (v := v) hz
  have hLp0 := lpSum_nonneg valid p16 v
  have hV10 := v1Sum_nonneg valid v
  have hAle := liftA_le hz hpb (v := v)
  have hA0 : 0 ≤ ∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ) :=
    Finset.sum_nonneg fun n _ => by
      have := mul_nonneg (Device.liftMag_nonneg .bf16 (p16 n).toRat) (Device.liftMag_nonneg .bf16 (v n).toRat)
      exact_mod_cast this
  -- bounds
  have hLp : lpSum valid p16 v ≤ 2 ^ 107 := by
    unfold lpSum
    calc _ ≤ ∑ _n : Fin 64, (2 : ℝ) * 2 ^ 100 := Finset.sum_le_sum fun n _ => by
          by_cases hv : valid n = true
          · simp only [hv, ite_true]
            exact mul_le_mul (hpb n hv) (hvb n hv) (abs_nonneg _) (by norm_num)
          · simp only [hv]; norm_num
      _ = 2 ^ 107 := by simp; norm_num
  have hV1 : v1Sum valid v ≤ 2 ^ 106 := by
    unfold v1Sum
    calc _ ≤ ∑ _n : Fin 64, (2 : ℝ) ^ 100 := Finset.sum_le_sum fun n _ => by
          by_cases hv : valid n = true
          · simp only [hv, ite_true]; exact hvb n hv
          · simp only [hv]; norm_num
      _ = 2 ^ 106 := by simp; norm_num
  have hAb : ∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ) ≤ 2 ^ 108 := by
    have h1 : (1 : ℝ) / 2 ^ 118 * (1 + v1Sum valid v) ≤ 1 / 2 ^ 118 * (1 + 2 ^ 106) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have h2 : (1 : ℝ) / 2 ^ 118 * (1 + 2 ^ 106) ≤ 1 := by norm_num
    have : (∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ)) ≤
        lpSum valid p16 v + 1 / 2 ^ 118 * (1 + v1Sum valid v) := hAle
    have h3 : (2 : ℝ) ^ 107 + 1 ≤ 2 ^ 108 := by norm_num
    linarith
  have hDb : |D.toReal| ≤ 2 ^ 108 + 2 ^ 107 := by
    have h0 := abs_sub_abs_le_abs_sub D.toReal (∑ n : Fin 64, (p16 n).toReal * (v n).toReal)
    rw [hSg] at h0 hD
    have h1 := sgSum_le_lpSum valid p16 v
    have h2 : 66 / 2 ^ 22 * (∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ)) ≤
        66 / 2 ^ 22 * 2 ^ 108 := mul_le_mul_of_nonneg_left hAb (by positivity)
    have h3 : (66 / 2 ^ 22 * 2 ^ 108 : ℝ) + 66 / 2 ^ 126 + 2 ^ 107 ≤ 2 ^ 108 + 2 ^ 107 := by norm_num
    linarith
  have hksb : |(FloatV.hostF32 one64).toReal| ≤ 2 := by
    have := abs_sub_abs_le_abs_sub (FloatV.hostF32 one64).toReal 1
    simp only [abs_one] at this
    have h00 : (1 : ℝ) / 2 ^ 24 * 1 + 1 / 2 ^ 150 + 1 ≤ 2 := by norm_num
    linarith
  obtain ⟨hxf, hxe⟩ := mul_real haf hre (by
    rw [abs_mul]
    calc |a.toReal| * |re.toReal| ≤ 2 ^ 119 * 2 := mul_le_mul hab hreb (abs_nonneg _) (by positivity)
      _ ≤ 2 ^ 126 := by norm_num) hx
  obtain ⟨hyf, hye⟩ := mul_real hDf hks_f (by
    rw [abs_mul]
    calc |D.toReal| * |(FloatV.hostF32 one64).toReal| ≤ (2 ^ 108 + 2 ^ 107) * 2 :=
          mul_le_mul hDb hksb (abs_nonneg _) (by positivity)
      _ ≤ 2 ^ 126 := by norm_num) hy
  have hxb : |x.toReal| ≤ 2 ^ 121 := by
    have h0 := abs_sub_abs_le_abs_sub x.toReal (a.toReal * re.toReal)
    have h1 : |a.toReal * re.toReal| ≤ 2 ^ 120 := by
      rw [abs_mul]
      calc |a.toReal| * |re.toReal| ≤ 2 ^ 119 * 2 := mul_le_mul hab hreb (abs_nonneg _) (by positivity)
        _ = 2 ^ 120 := by norm_num
    have h2 : 1 / 2 ^ 24 * |a.toReal * re.toReal| ≤ 1 / 2 ^ 24 * 2 ^ 120 :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    have h3 : (2 : ℝ) ^ 120 + 1 / 2 ^ 24 * 2 ^ 120 + 1 / 2 ^ 150 ≤ 2 ^ 121 := by norm_num
    linarith
  have hyb : |y.toReal| ≤ 2 ^ 111 := by
    have h0 := abs_sub_abs_le_abs_sub y.toReal (D.toReal * (FloatV.hostF32 one64).toReal)
    have h1 : |D.toReal * (FloatV.hostF32 one64).toReal| ≤ (2 ^ 108 + 2 ^ 107) * 2 := by
      rw [abs_mul]; exact mul_le_mul hDb hksb (abs_nonneg _) (by positivity)
    have h2 : 1 / 2 ^ 24 * |D.toReal * (FloatV.hostF32 one64).toReal| ≤ 1 / 2 ^ 24 * ((2 ^ 108 + 2 ^ 107) * 2) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    have h3 : (2 : ℝ) ^ 108 * 3 + 1 / 2 ^ 24 * ((2 ^ 108 + 2 ^ 107) * 2) + 1 / 2 ^ 150 ≤ 2 ^ 111 := by norm_num
    linarith
  obtain ⟨hf', he'⟩ := addrel_real hxf hyf (by linarith [(by norm_num : (2 : ℝ) ^ 121 + 2 ^ 111 ≤ 2 ^ 126)]) ha'
  refine ⟨hf', ?_⟩
  have hreal := acc_pref_real (ra := re.toReal * a.toReal) (x := x.toReal) (Dd := D.toReal)
    (ks := (FloatV.hostF32 one64).toReal) (y := y.toReal) (a' := a'.toReal) (Sg := sgSum valid p16 v)
    (A := ∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ))
    (Lp := lpSum valid p16 v) (V1 := v1Sum valid v)
    (by rw [mul_comm] at hxe; exact hxe) (by rw [hSg] at hD; exact hD) hks_e hye he' hAle hA0 hV10
    (sgSum_le_lpSum valid p16 v) hLp0
  exact hreal


/-- **An extend trip's accumulator element** `tl.dot_add(p16, v, acc * re_scale)`. -/
theorem col_ext {P : Platform} (_hP : Device P) {valid : Fin 64 → Bool}
    {p16 v : Fin 64 → BF16} {a re c a' : F32}
    (hz : ∀ n, valid n = false → (p16 n).toReal = 0) (hpb : ∀ n, valid n = true → |(p16 n).toReal| ≤ 2)
    (haf : a.isFinite = true) (hab : |a.toReal| ≤ 2 ^ 119) (hre : re.isFinite = true)
    (hreb : |re.toReal| ≤ 2) (hcx : FloatV.NaNEq (FloatV.mul a re) c)
    (hd : a'.isFinite = true ∧
      |a'.toReal - (c.toReal + ∑ n : Fin 64, (p16 n).toReal * (v n).toReal)| ≤
        (66 / 2 ^ 22 + 3 * (1 / 2 ^ 24) * (1 + 66 / 2 ^ 22)) *
            (|c.toReal| + ∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat *
              Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ)) +
          (66 / 2 ^ 126 * (1 + 3 * (1 / 2 ^ 24)) + 4 * (1 / 2 ^ 150))) :
    a'.isFinite = true ∧
      |a'.toReal - (re.toReal * a.toReal + sgSum valid p16 v)| ≤
        epsE1 * |re.toReal * a.toReal| + eps2 * lpSum valid p16 v + eps1 * (1 + v1Sum valid v) := by
  classical
  have hSg := sgSum_eq (v := v) hz
  have hLp0 := lpSum_nonneg valid p16 v
  have hV10 := v1Sum_nonneg valid v
  have hAle := liftA_le hz hpb (v := v)
  have hA0 : 0 ≤ ∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ) :=
    Finset.sum_nonneg fun n _ => by
      have := mul_nonneg (Device.liftMag_nonneg .bf16 (p16 n).toRat) (Device.liftMag_nonneg .bf16 (v n).toRat)
      exact_mod_cast this
  obtain ⟨hcf, hce⟩ := mul_real haf hre (by
    rw [abs_mul]
    calc |a.toReal| * |re.toReal| ≤ 2 ^ 119 * 2 := mul_le_mul hab hreb (abs_nonneg _) (by positivity)
      _ ≤ 2 ^ 126 := by norm_num) hcx
  refine ⟨hd.1, ?_⟩
  rw [hSg] at hd
  exact acc_ext_real (ra := re.toReal * a.toReal) (c := c.toReal) (a' := a'.toReal)
    (Sg := sgSum valid p16 v)
    (A := ∑ n : Fin 64, ((Device.liftMag .bf16 (p16 n).toRat * Device.liftMag .bf16 (v n).toRat : ℚ) : ℝ))
    (Lp := lpSum valid p16 v) (V1 := v1Sum valid v)
    (by rw [mul_comm] at hce; exact hce) hd.2 hAle hA0 hV10 hLp0


end

end Claims.ExtendAttn
