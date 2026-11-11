import Proof.Math

/-!
# The value of one output element

The kernel's `float32` accumulation of the `K` widened inputs, its multiplication by the `float32`
scale, and the final `bfloat16` rounding: `bf16_value_bound` chains the three error bounds
(`foldF_err`, `FloatV.mul_err`, `FloatV.cvt_err`) into `Fb`, and `room` supplies the room in the
formats that those bounds need from `InRange`.
-/

namespace Claims.MoeSumReduce

open TVA

/-- A `bfloat16` element widened to `float32` keeps its value: `bfloat16 → float32` is exact. -/
theorem cvt_widen {d : Inst} {t k j : Nat} (h : (xv d t k j).isFinite = true) :
    (FloatV.cvt (xv d t k j) : F32).isFinite = true ∧
      (FloatV.cvt (xv d t k j) : F32).toRat = (xv d t k j).toRat := by
  have hh := FloatV.toRat_cvt_of_wider (f := Format.bfloat16) (g := Format.binary32)
    Format.bfloat16_regular Format.binary32_regular Format.wider_bfloat16_binary32 .rn h
  simpa only [FloatV.cvtWith_rn] using hh

/-- Every initial segment of the sum of magnitudes is at most the whole sum. -/
theorem absSum_partial (d : Inst) (t j : Nat) :
    ∀ t', t' ≤ d.K →
      |∑ k ∈ Finset.range t', (xv d t k j).toRat| ≤ absSum d t j := by
  intro t' ht'
  calc |∑ k ∈ Finset.range t', (xv d t k j).toRat|
      ≤ ∑ k ∈ Finset.range t', |(xv d t k j).toRat| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k ∈ Finset.range d.K, |(xv d t k j).toRat| := by
        refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono ht') ?_
        intro i _ _
        exact abs_nonneg _
    _ = absSum d t j := rfl

/-- In-range inputs leave room: the accumulation, scaled, stays inside `bfloat16`'s range. -/
theorem room (d : Inst) (hd : d.ok) {t j : Nat} (h : InRange d t j) :
    (absSum d t j + E1 d.K (absSum d t j)) * (|(sigma32 d).toRat| + 1) * (1 + u32) + eta32
      ≤ Format.bfloat16.maxVal := by
  obtain ⟨_, _, hσ32, hA2, hσA⟩ := h
  have hu0 : (0 : ℚ) ≤ u32 := by rw [u32_eq]; exact Format.u_nonneg
  have he0 : (0 : ℚ) ≤ eta32 := by rw [eta32_eq]; exact Format.eta_nonneg
  have hA0 : (0 : ℚ) ≤ absSum d t j := by
    unfold absSum
    exact Finset.sum_nonneg (fun k _ => abs_nonneg _)
  have hs0 : (0 : ℚ) ≤ |(sigma32 d).toRat| := abs_nonneg _
  have hKq : (d.K : ℚ) ≤ (2 : ℚ) ^ 20 := by
    exact_mod_cast hd.K_le
  have hKu : (d.K : ℚ) * u32 ≤ 1 / 16 := by
    have hu : u32 = (2 : ℚ) ^ (-24 : ℤ) := rfl
    rw [hu]
    have h2 : (2 : ℚ) ^ 20 * (2 : ℚ) ^ (-24 : ℤ) = 1 / 16 := by norm_num
    calc (d.K : ℚ) * (2 : ℚ) ^ (-24 : ℤ)
        ≤ (2 : ℚ) ^ 20 * (2 : ℚ) ^ (-24 : ℤ) :=
          mul_le_mul_of_nonneg_right hKq (by norm_num)
      _ = 1 / 16 := h2
  have hE1le : E1 d.K (absSum d t j)
      ≤ (16 / 15 : ℚ) * (d.K : ℚ) * u32 * absSum d t j + 2 * eta32 * (d.K : ℚ) :=
    E1_le d.K (absSum d t j) hd.K_le hA0
  have h1 : (16 / 15 : ℚ) * (d.K : ℚ) * u32 * absSum d t j ≤ absSum d t j / 15 := by
    have hh := mul_le_mul_of_nonneg_right hKu hA0
    nlinarith
  have h2 : 2 * eta32 * (d.K : ℚ) ≤ (1 / 2 : ℚ) ^ 129 := by
    have he : eta32 = (2 : ℚ) ^ (-150 : ℤ) := rfl
    rw [he]
    have h3 : 2 * (2 : ℚ) ^ (-150 : ℤ) * (2 : ℚ) ^ 20 = (1 / 2 : ℚ) ^ 129 := by norm_num
    calc 2 * (2 : ℚ) ^ (-150 : ℤ) * (d.K : ℚ)
        ≤ 2 * (2 : ℚ) ^ (-150 : ℤ) * (2 : ℚ) ^ 20 := by
          apply mul_le_mul_of_nonneg_left hKq
          norm_num
      _ = (1 / 2 : ℚ) ^ 129 := h3
  have hAE : absSum d t j + E1 d.K (absSum d t j)
      ≤ (16 / 15 : ℚ) * absSum d t j + (1 / 2 : ℚ) ^ 129 := by
    nlinarith [hE1le, h1, h2]
  have hsmax : |(sigma32 d).toRat| ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
    have hh := FloatV.abs_toRat_le_maxVal (f := Format.binary32) Format.binary32_regular hσ32
    rwa [Format.binary32_maxVal] at hh
  have hs2 : (1 / 2 : ℚ) ^ 129 * |(sigma32 d).toRat| ≤ 1 / 2 := by
    have h1 : (1 / 2 : ℚ) ^ 129 * |(sigma32 d).toRat|
        ≤ (1 / 2 : ℚ) ^ 129 * (((2 : ℚ) ^ 24 - 1) * (2 : ℚ) ^ 104) :=
      mul_le_mul_of_nonneg_left hsmax (by norm_num)
    have h2 : (1 / 2 : ℚ) ^ 129 * (((2 : ℚ) ^ 24 - 1) * (2 : ℚ) ^ 104) ≤ 1 / 2 := by
      norm_num
    linarith
  have ha1 : (16 / 15 : ℚ) * (absSum d t j) * |(sigma32 d).toRat| ≤ (16 / 15 : ℚ) * (2 : ℚ) ^ 126 := by
    have hh := mul_le_mul_of_nonneg_left hσA (show (0 : ℚ) ≤ 16 / 15 by norm_num)
    nlinarith [hh]
  have ha2 : (16 / 15 : ℚ) * (absSum d t j) ≤ (16 / 15 : ℚ) * (2 : ℚ) ^ 126 := by
    have hh := mul_le_mul_of_nonneg_left hA2 (show (0 : ℚ) ≤ 16 / 15 by norm_num)
    linarith
  have hX1 : ((16 / 15 : ℚ) * absSum d t j + (1 / 2 : ℚ) ^ 129)
        * (|(sigma32 d).toRat| + 1)
      ≤ (32 / 15 : ℚ) * (2 : ℚ) ^ 126 + 1 / 2 + (1 / 2 : ℚ) ^ 129 := by
    nlinarith [ha1, ha2, hs2]
  have hm := mul_le_mul_of_nonneg_right hAE (by linarith : (0 : ℚ) ≤ |(sigma32 d).toRat| + 1)
  have hX2 : (absSum d t j + E1 d.K (absSum d t j)) * (|(sigma32 d).toRat| + 1) * (1 + u32)
      ≤ ((32 / 15 : ℚ) * (2 : ℚ) ^ 126 + 1 / 2 + (1 / 2 : ℚ) ^ 129) * (1 + u32) := by
    apply mul_le_mul_of_nonneg_right _ (by linarith : (0 : ℚ) ≤ 1 + u32)
    linarith
  have hfin : ((32 / 15 : ℚ) * (2 : ℚ) ^ 126 + 1 / 2 + (1 / 2 : ℚ) ^ 129) * (1 + u32) + eta32
      ≤ (2 ^ 8 - 1) * (2 : ℚ) ^ 120 := by
    have hu : u32 = (2 : ℚ) ^ (-24 : ℤ) := rfl
    have he : eta32 = (2 : ℚ) ^ (-150 : ℤ) := rfl
    rw [hu, he]
    norm_num
  rw [Format.bfloat16_maxVal]
  linarith [hX2, hfin]

/-- The value of one output element: the `float32` left fold `x`, multiplied by the `float32`
scale `σ₃₂` and rounded to `bfloat16`, is within `tol` of `σ · S`. -/
theorem value_bound (K : Nat) (S A σ σ32 : ℚ)
    (hK1 : 1 ≤ K) (hKle : K ≤ 2 ^ 20)
    (hA0 : 0 ≤ A) (hS : |S| ≤ A)
    (hσrel : |σ - σ32| ≤ u32 * |σ32| + eta32)
    (x : Nat → F32) (hxf : ∀ k, k < K → (x k).isFinite = true)
    (hSe : S = ∑ k ∈ Finset.range K, (x k).toRat)
    (hxS : ∀ t, t ≤ K → |∑ k ∈ Finset.range t, (x k).toRat| ≤ A)
    (hroom : (A + E1 K A) * (|σ32| + 1) * (1 + u32) + eta32 ≤ Format.bfloat16.maxVal) :
    (foldF x K).isFinite = true ∧
      ∀ (σ32v : F32), σ32v.toRat = σ32 → σ32v.isFinite = true →
        (FloatV.mul (foldF x K) σ32v).isFinite = true ∧
          ∀ out : BF16, FloatV.NaNEq (FloatV.cvt (FloatV.mul (foldF x K) σ32v)) out →
            out.isFinite = true ∧ |out.toRat - σ * S| ≤ tol K S A σ := by
  have hu0 : (0 : ℚ) ≤ u32 := by rw [u32_eq]; exact Format.u_nonneg
  have he0 : (0 : ℚ) ≤ eta32 := by rw [eta32_eq]; exact Format.eta_nonneg
  have hv0 : (0 : ℚ) ≤ ubf := by rw [ubf_eq]; exact Format.u_nonneg
  have hs0 : (0 : ℚ) ≤ |σ32| := abs_nonneg σ32
  have hK0 : 0 < K := by omega
  have hE10 : (0 : ℚ) ≤ E1 K A := E1_nonneg K A hA0
  have hAE0 : (0 : ℚ) ≤ A + E1 K A := by linarith
  have hbfle : Format.bfloat16.maxVal ≤ Format.binary32.maxVal := by
    rw [Format.bfloat16_maxVal, Format.binary32_maxVal]
    norm_num
  have hroomAE : A + E1 K A ≤ Format.bfloat16.maxVal := by
    have hp0 : (0 : ℚ) ≤ (A + E1 K A) * |σ32| := mul_nonneg hAE0 hs0
    have hp1 : (0 : ℚ) ≤ (A + E1 K A) * (|σ32| + 1) * u32 :=
      mul_nonneg (mul_nonneg hAE0 (by linarith)) hu0
    nlinarith [hroom, hp0, hp1, he0]
  have hM : A + E1 K A ≤ Format.binary32.maxVal := le_trans hroomAE hbfle
  obtain ⟨haccfin, haccraw⟩ := foldF_err x K A hK0 hKle hxf hA0 hxS hM
  have hacc : |(foldF x K).toRat - S| ≤ E1 K A := by
    rw [hSe]
    exact haccraw
  have haccabs : |(foldF x K).toRat| ≤ |S| + E1 K A := by
    have hsplit : (foldF x K).toRat = ((foldF x K).toRat - S) + S := by ring
    calc |(foldF x K).toRat| = |((foldF x K).toRat - S) + S| := congrArg abs hsplit
      _ ≤ |(foldF x K).toRat - S| + |S| := abs_add_le _ _
      _ ≤ E1 K A + |S| := by linarith
      _ = |S| + E1 K A := by ring
  refine ⟨haccfin, ?_⟩
  intro σ32v hσv hσvfin
  have hsv : |σ32v.toRat| = |σ32| := by rw [hσv]
  have hmulmax : |(foldF x K).toRat * σ32v.toRat| ≤ Format.binary32.maxVal := by
    have h1 : |(foldF x K).toRat * σ32v.toRat| ≤ (|S| + E1 K A) * |σ32| := by
      rw [abs_mul, hsv]
      exact mul_le_mul_of_nonneg_right haccabs hs0
    have h2 : (|S| + E1 K A) * |σ32| ≤ (A + E1 K A) * (|σ32| + 1) := by
      have h3 : (|S| + E1 K A) * |σ32| ≤ (A + E1 K A) * |σ32| :=
        mul_le_mul_of_nonneg_right (by linarith) hs0
      have h4 : (A + E1 K A) * |σ32| ≤ (A + E1 K A) * (|σ32| + 1) := by
        nlinarith [hAE0, hs0]
      linarith
    have h5 : (A + E1 K A) * (|σ32| + 1) ≤ Format.bfloat16.maxVal := by
      have hp0 : (0 : ℚ) ≤ (A + E1 K A) * (|σ32| + 1) := mul_nonneg hAE0 (by linarith)
      have hp1 : (0 : ℚ) ≤ (A + E1 K A) * (|σ32| + 1) * u32 := mul_nonneg hp0 hu0
      nlinarith [hroom, hp1, he0]
    linarith [h1, h2, h5, hbfle]
  obtain ⟨hprfin, hprerr⟩ :=
    FloatV.mul_err (f := Format.binary32) Format.binary32_regular haccfin hσvfin hmulmax
  rw [← u32_eq, ← eta32_eq] at hprerr
  have hab : |(foldF x K).toRat * σ32v.toRat| ≤ (|S| + E1 K A) * |σ32| := by
    rw [abs_mul, hsv]
    exact mul_le_mul_of_nonneg_right haccabs hs0
  have hprQ : |(FloatV.mul (foldF x K) σ32v).toRat| ≤ Qb |σ32| S (E1 K A) := by
    have hsplit : (FloatV.mul (foldF x K) σ32v).toRat
        = ((FloatV.mul (foldF x K) σ32v).toRat - (foldF x K).toRat * σ32v.toRat)
          + (foldF x K).toRat * σ32v.toRat := by ring
    have h1 : |(FloatV.mul (foldF x K) σ32v).toRat|
        ≤ |(FloatV.mul (foldF x K) σ32v).toRat - (foldF x K).toRat * σ32v.toRat|
          + |(foldF x K).toRat * σ32v.toRat| := by
      calc |(FloatV.mul (foldF x K) σ32v).toRat|
          = |((FloatV.mul (foldF x K) σ32v).toRat - (foldF x K).toRat * σ32v.toRat)
              + (foldF x K).toRat * σ32v.toRat| := congrArg abs hsplit
        _ ≤ _ := abs_add_le _ _
    have h3 : |(foldF x K).toRat * σ32v.toRat| * u32 ≤ ((|S| + E1 K A) * |σ32|) * u32 :=
      mul_le_mul_of_nonneg_right hab hu0
    unfold Qb Em
    nlinarith [h1, hprerr, h3, hab]
  have hmax2 : |(FloatV.mul (foldF x K) σ32v).toRat| ≤ Format.bfloat16.maxVal := by
    have hQb_eq : Qb |σ32| S (E1 K A) = |σ32| * (|S| + E1 K A) * (1 + u32) + eta32 := by
      unfold Qb Em
      ring
    have hQb_le : Qb |σ32| S (E1 K A) ≤ Format.bfloat16.maxVal := by
      rw [hQb_eq]
      have h1 : |σ32| * (|S| + E1 K A) * (1 + u32) ≤ |σ32| * (A + E1 K A) * (1 + u32) := by
        apply mul_le_mul_of_nonneg_right _ (by linarith : (0 : ℚ) ≤ 1 + u32)
        exact mul_le_mul_of_nonneg_left (by linarith) hs0
      have h2 : |σ32| * (A + E1 K A) * (1 + u32) + eta32
          ≤ (A + E1 K A) * (|σ32| + 1) * (1 + u32) + eta32 := by
        have h3 : (0 : ℚ) ≤ (A + E1 K A) * (1 + u32) :=
          mul_nonneg hAE0 (by linarith)
        nlinarith
      linarith [h1, h2, hroom]
    linarith [hprQ, hQb_le]
  obtain ⟨hcvtfin, hcvterr⟩ :=
    FloatV.cvt_err (f := Format.binary32) (g := Format.bfloat16) Format.bfloat16_regular
      hprfin hmax2
  rw [← ubf_eq, ← etabf_eq] at hcvterr
  refine ⟨hprfin, ?_⟩
  intro out hout
  have hnan : (FloatV.cvt (FloatV.mul (foldF x K) σ32v) : BF16).isNaN = false :=
    (FloatV.isFinite_iff.mp hcvtfin).1
  have houth : out = FloatV.cvt (FloatV.mul (foldF x K) σ32v) := FloatV.NaNEq.eq hout hnan
  refine ⟨?_, ?_⟩
  · rw [houth]
    exact hcvtfin
  · rw [houth]
    have htri : |(FloatV.cvt (FloatV.mul (foldF x K) σ32v) : BF16).toRat - σ * S|
        ≤ Fb |σ32| S (E1 K A) σ σ32 := by
      have e : (FloatV.cvt (FloatV.mul (foldF x K) σ32v) : BF16).toRat - σ * S
          = ((FloatV.cvt (FloatV.mul (foldF x K) σ32v) : BF16).toRat
                - (FloatV.mul (foldF x K) σ32v).toRat)
            + ((FloatV.mul (foldF x K) σ32v).toRat
                - (foldF x K).toRat * σ32v.toRat)
            + ((foldF x K).toRat * σ32v.toRat - σ32 * S)
            + (σ32 * S - σ * S) := by ring
      rw [e]
      have a1 := abs_add_le
        (((FloatV.cvt (FloatV.mul (foldF x K) σ32v) : BF16).toRat
                - (FloatV.mul (foldF x K) σ32v).toRat)
            + ((FloatV.mul (foldF x K) σ32v).toRat
                - (foldF x K).toRat * σ32v.toRat)
            + ((foldF x K).toRat * σ32v.toRat - σ32 * S))
        (σ32 * S - σ * S)
      have a2 := abs_add_le
        (((FloatV.cvt (FloatV.mul (foldF x K) σ32v) : BF16).toRat
                - (FloatV.mul (foldF x K) σ32v).toRat)
            + ((FloatV.mul (foldF x K) σ32v).toRat
                - (foldF x K).toRat * σ32v.toRat))
        ((foldF x K).toRat * σ32v.toRat - σ32 * S)
      have a3 := abs_add_le
        ((FloatV.cvt (FloatV.mul (foldF x K) σ32v) : BF16).toRat
                - (FloatV.mul (foldF x K) σ32v).toRat)
        ((FloatV.mul (foldF x K) σ32v).toRat
                - (foldF x K).toRat * σ32v.toRat)
      have b1 : |(FloatV.cvt (FloatV.mul (foldF x K) σ32v) : BF16).toRat
            - (FloatV.mul (foldF x K) σ32v).toRat|
          ≤ ubf * Qb |σ32| S (E1 K A) + etabf := by
        have hq := mul_le_mul_of_nonneg_right hprQ hv0
        linarith [hcvterr, hq]
      have b2 : |(FloatV.mul (foldF x K) σ32v).toRat - (foldF x K).toRat * σ32v.toRat|
          ≤ Em |σ32| S (E1 K A) := by
        have hq := mul_le_mul_of_nonneg_right hab hu0
        unfold Em
        nlinarith [hprerr, hq]
      have b3 : |(foldF x K).toRat * σ32v.toRat - σ32 * S| ≤ |σ32| * E1 K A := by
        rw [hσv,
          show (foldF x K).toRat * σ32 - σ32 * S = σ32 * ((foldF x K).toRat - S) by ring,
          abs_mul]
        exact mul_le_mul_of_nonneg_left hacc hs0
      have b4 : |σ32 * S - σ * S| ≤ |σ - σ32| * |S| := by
        rw [show σ32 * S - σ * S = -((σ - σ32) * S) by ring, abs_neg, abs_mul]
      unfold Fb
      linarith [a1, a2, a3, b1, b2, b3, b4]
    have hFx : Fb |σ32| S (E1 K A) σ σ32 = Fx |σ32| S (E1 K A) σ σ32 :=
      Fb_eq_Fx _ _ _ _ _
    have htol := Fx_le_tol_weak K S A σ σ32 hK1 hKle hA0 hS hσrel
    rw [hFx] at htri
    exact le_trans htri htol

end Claims.MoeSumReduce
