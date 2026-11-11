import Proof.NumericF

/-!
# `lora_shrink_bf16`: the numeric part

Pure arithmetic on the claim's tolerances (no program): the parts of `K` add up to the dot product,
one scaled part is a `PartOk` operand, the atomic additions have room, and the bound on the
element's value is at most `tol`.
-/

namespace Claims.LoraShrink

open TVA

/-- The `SPLIT_K` parts of `K` partition the `k < K`: the parts' exact sums add up to the dot
product. -/
theorem dotSum_eq_sum_parts (d : Inst) (hd : d.ok) (va vb : Nat → BF16) :
    dotSum d va vb = ∑ p ∈ Finset.range (sk d), partSum d va vb p :=
  dotSum_eq_parts d hd va vb

/-- **One scaled part.** The accumulator `c` of part `p` is a finite `float32` within `errPart p` of
`partSum p`; `w` is its product with the kernel's `scaling` (`tl.mul`, an IEEE product up to NaN
payload). If `dotMag` and the room hold, `w` is a `PartOk` operand. -/
theorem partOk_of_mul (d : Inst) (hd : d.ok) (va vb : Nat → BF16) (p : Nat) (hp : p < sk d)
    (hroom : vMag d va vb + atomTol d va vb ≤ Format.binary32.maxVal)
    (c w : F32) (hc : c.isFinite = true)
    (herr : |c.toRat - partSum d va vb p| ≤ errPart d va vb p)
    (hw : FloatV.NaNEq (FloatV.mul c (scaling32 d)) w) : PartOk d va vb p w := by
  have hu := Format.u_nonneg (f := Format.binary32)
  have he := Format.eta_nonneg (f := Format.binary32)
  have hsc : (scaling32 d).isFinite = true := hd.sc_fin.2
  set P := partSum d va vb p with hP
  set e := errPart d va vb p with he'
  have hcP : |c.toRat| ≤ |P| + e := by
    have := abs_sub_abs_le_abs_sub c.toRat P
    linarith
  have hterm : |s32 d| * (|P| + e) ≤ vMag d va vb := by
    have h1 : |P| + e ≤ partMag d va vb :=
      Finset.single_le_sum (f := fun p => |partSum d va vb p| + errPart d va vb p)
        (fun p _ => add_nonneg (abs_nonneg _) (errPart_nonneg d va vb p)) (Finset.mem_range.mpr hp)
    have h2 := mul_le_mul_of_nonneg_left h1 (abs_nonneg (s32 d))
    have h3 : 0 ≤ mulTol d va vb := by
      have := partMag_nonneg d va vb
      have : 0 ≤ Format.binary32.u * |s32 d| * partMag d va vb :=
        mul_nonneg (mul_nonneg hu (abs_nonneg _)) this
      have : 0 ≤ (sk d : ℚ) * Format.binary32.eta := mul_nonneg (Nat.cast_nonneg _) he
      show 0 ≤ Format.binary32.u * |s32 d| * partMag d va vb + (sk d : ℚ) * Format.binary32.eta
      linarith
    have : vMag d va vb = |s32 d| * partMag d va vb + mulTol d va vb := rfl
    linarith
  have hx : |c.toRat * (scaling32 d).toRat| ≤ |s32 d| * (|P| + e) := by
    rw [abs_mul, mul_comm]
    exact mul_le_mul_of_nonneg_left hcP (abs_nonneg _)
  have hmax : |c.toRat * (scaling32 d).toRat| ≤ Format.binary32.maxVal :=
    hx.trans (hterm.trans ((le_add_of_nonneg_right (atomTol_nonneg d va vb)).trans hroom))
  obtain ⟨hfin, herr2⟩ := FloatV.mul_err Format.binary32_regular hc hsc hmax
  obtain ⟨hwfin, hwv, -⟩ := FloatV.NaNEq.exact_of_isFinite hw hfin
  unfold PartOk
  rw [hwv]
  have e1 : |c.toRat * (scaling32 d).toRat| * Format.binary32.u ≤
      |s32 d| * (|P| + e) * Format.binary32.u := mul_le_mul_of_nonneg_right hx hu
  have hs32 : (scaling32 d).toRat = s32 d := rfl
  refine ⟨hwfin, ?_, ?_⟩
  · have h1 : |(FloatV.mul c (scaling32 d)).toRat - s32 d * P| ≤
        |(FloatV.mul c (scaling32 d)).toRat - c.toRat * s32 d| + |s32 d| * |c.toRat - P| := by
      have : (FloatV.mul c (scaling32 d)).toRat - s32 d * P =
          ((FloatV.mul c (scaling32 d)).toRat - c.toRat * s32 d) + s32 d * (c.toRat - P) := by ring
      rw [this]
      refine (abs_add_le _ _).trans ?_
      rw [abs_mul]
    have h2 : |s32 d| * |c.toRat - P| ≤ |s32 d| * e := mul_le_mul_of_nonneg_left herr (abs_nonneg _)
    rw [hs32] at herr2 e1
    unfold mulErr
    nlinarith
  · have h1 : |(FloatV.mul c (scaling32 d)).toRat| ≤
        |c.toRat * s32 d| + (|c.toRat * s32 d| * Format.binary32.u + Format.binary32.eta) := by
      rw [hs32] at herr2
      have := abs_sub_abs_le_abs_sub (FloatV.mul c (scaling32 d)).toRat (c.toRat * s32 d)
      linarith
    rw [hs32] at hx e1
    unfold mulErr
    nlinarith

/-- **Room for the atomic additions.** If each part's operand `v p` (`p < SPLIT_K`) satisfies the
bounds of `PartOk` (as rationals), their magnitudes add up to at most `vMag`, so the atomic
additions of all parts into the element have room. -/
theorem atom_room (d : Inst) (va vb : Nat → BF16) (v : Nat → ℚ)
    (hv : ∀ p, p < sk d → |v p| ≤ |s32 d| * (|partSum d va vb p| + errPart d va vb p) + mulErr d va vb p)
    (hroom : vMag d va vb + atomTol d va vb ≤ Format.binary32.maxVal) :
    ∑ p ∈ Finset.range (sk d), |v p| ≤ vMag d va vb ∧
      ∑ p ∈ Finset.range (sk d), |v p| +
        DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (sk d) (∑ p ∈ Finset.range (sk d), |v p|) ≤
        Format.binary32.maxVal := by
  have h := sum_abs_le_vMag d va vb v hv
  refine ⟨h, ?_⟩
  have h2 : DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (sk d) (∑ p ∈ Finset.range (sk d), |v p|) ≤
      atomTol d va vb := bound_mono_B (by positivity) _ h
  linarith

set_option maxHeartbeats 4000000 in
/-- **The element is within `tol`, for given constants of a configuration.** The numeric
hypotheses (`rho ≤ r`, the number of steps `≤ Tm`, ...) are those of one launch configuration. -/
theorem total_case (d : Inst) (hd : d.ok) (va vb : Nat → BF16) (v : Nat → ℚ)
    (hv : ∀ p, p < sk d →
      |v p - s32 d * partSum d va vb p| ≤ |s32 d| * errPart d va vb p + mulErr d va vb p ∧
      |v p| ≤ |s32 d| * (|partSum d va vb p| + errPart d va vb p) + mulErr d va vb p)
    (y : ℚ)
    (hy : |y - ∑ p ∈ Finset.range (sk d), v p| ≤
      DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (sk d) (∑ p ∈ Finset.range (sk d), |v p|))
    (r a1 Tm : ℚ) (hT0 : (stepsOf d 0 : ℚ) ≤ Tm) (hTm : Tm ≤ 512)
    (hskT : (sk d : ℚ) * Tm ≤ 4096) (hrho : rho d ≤ r) (hTr : Tm * r ≤ 1 / 100) (hr0 : 0 ≤ r)
    (ha1 : 102 / 100 * (sk d : ℚ) * 2 ^ (-23 : ℤ) ≤ a1) (ha1' : a1 ≤ 1 / 100000)
    (hW : (bk d : ℚ) * 2 ^ (-21 : ℤ) * 512 ≤ 1 / 16)
    (hnum : (1 + 1 / 8388608) * (102 / 100 * r * (1 + a1 * (1 + 1 / 16777216) + 1 / 16777216) +
      a1 * (1 + 1 / 16777216) + 2 / 16777216) ≤ (bk d : ℚ) * 2 ^ (-21 : ℤ)) :
    |y - d.scaling.toRat * dotSum d va vb| ≤ tol d va vb := by
  have hK := hd.K_pos
  have hu := Format.u_nonneg (f := Format.binary32)
  have he := Format.eta_nonneg (f := Format.binary32)
  have hskle : (sk d : ℚ) ≤ 64 := by
    rcases cfg_cases d with h | h <;> rw [h.2.2.2.2] <;> norm_num
  have hT1 : (1 : ℚ) ≤ stepsOf d 0 := by exact_mod_cast steps_pos d hK
  have hrho0 := rho_nonneg d
  have hT0r : (stepsOf d 0 : ℚ) * rho d ≤ 1 / 100 :=
    (mul_le_mul hT0 hrho hrho0 (by linarith)).trans hTr
  have hm := sum_errPart_le d hd va vb hT0r
  have hE := error_sum d hd va vb v hv y hy
  have hs32 := scaling32_err d hd
  have hdm0 := dotMag_nonneg d va vb
  have hσ : |s32 d| ≤ (1 + 2 * Format.binary32.u) * (|d.scaling.toRat| + Format.binary32.eta) := by
    have h1 : |s32 d| ≤ |d.scaling.toRat| + (|s32 d| * Format.binary32.u + Format.binary32.eta) := by
      have := abs_sub_abs_le_abs_sub (s32 d) d.scaling.toRat
      linarith
    have hu2 : Format.binary32.u ≤ 1 / 16777216 := by rw [Format.binary32_u]; norm_num
    have h2 : |s32 d| * (1 - Format.binary32.u) ≤ |d.scaling.toRat| + Format.binary32.eta := by
      linarith
    have h3 : |s32 d| * (1 - Format.binary32.u) ≤
        (1 + 2 * Format.binary32.u) * (|d.scaling.toRat| + Format.binary32.eta) * (1 - Format.binary32.u) := by
      have : (|d.scaling.toRat| + Format.binary32.eta) ≤
          (1 + 2 * Format.binary32.u) * (|d.scaling.toRat| + Format.binary32.eta) * (1 - Format.binary32.u) := by
        have hx : 0 ≤ |d.scaling.toRat| + Format.binary32.eta := by positivity
        have : 1 ≤ (1 + 2 * Format.binary32.u) * (1 - Format.binary32.u) := by nlinarith
        nlinarith
      linarith
    exact le_of_mul_le_mul_right h3 (by linarith)
  have hPM : partMag d va vb ≤ dotMag d va vb + ∑ p ∈ Finset.range (sk d), errPart d va vb p := by
    unfold partMag
    rw [← sum_partLift d hd va vb, ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun p _ => add_le_add (abs_partSum_le d va vb p) le_rfl
  have hPM0 := partMag_nonneg d va vb
  have hm0 : 0 ≤ ∑ p ∈ Finset.range (sk d), errPart d va vb p :=
    Finset.sum_nonneg fun p _ => errPart_nonneg d va vb p
  have hE' : |y - d.scaling.toRat * dotSum d va vb| ≤
      102 / 100 * (sk d : ℚ) * 2 ^ (-23 : ℤ) *
          (|s32 d| * partMag d va vb + Format.binary32.u * |s32 d| * partMag d va vb +
            (sk d : ℚ) * Format.binary32.eta) +
        102 / 100 * (sk d : ℚ) * 2 ^ (-124 : ℤ) +
      (|s32 d| * ∑ p ∈ Finset.range (sk d), errPart d va vb p +
          Format.binary32.u * |s32 d| * partMag d va vb + (sk d : ℚ) * Format.binary32.eta) +
      (Format.binary32.u * |s32 d| + Format.binary32.eta) * dotMag d va vb := by
    refine hE.trans (le_of_eq ?_)
    rw [vMag_eq]; unfold mulTol; simp only [s32]; ring
  have hstep := steps_pos d hK
  have hα := alpha_nonneg d
  have hα' := alpha_le_num d
  have hsk0 : (0 : ℚ) ≤ sk d := Nat.cast_nonneg _
  have hmaster := master (u := Format.binary32.u) (η := Format.binary32.eta)
    (A1 := 102 / 100 * (sk d : ℚ) * 2 ^ (-23 : ℤ)) (A2 := 102 / 100 * (sk d : ℚ) * 2 ^ (-124 : ℤ))
    (skη := (sk d : ℚ) * Format.binary32.eta) (σ := |s32 d|) (s := |d.scaling.toRat|)
    (B := dotMag d va vb) (m := ∑ p ∈ Finset.range (sk d), errPart d va vb p)
    (PM := partMag d va vb) (g := 102 / 100 * (sk d : ℚ) * (stepsOf d 0 : ℚ) * alpha d)
    (M1 := 102 / 100 * (stepsOf d 0 : ℚ) * rho d) (k1 := 102 / 100 * r) (a1 := a1)
    (T0 := (stepsOf d 0 : ℚ)) (W1 := (bk d : ℚ) * 2 ^ (-21 : ℤ))
    (E := |y - d.scaling.toRat * dotSum d va vb|)
    hu (by rw [Format.binary32_u]; norm_num) he (by rw [Format.binary32_eta]) (by positivity) ha1
    (by
      calc 102 / 100 * (sk d : ℚ) * 2 ^ (-124 : ℤ) ≤ 102 / 100 * 64 * 2 ^ (-124 : ℤ) := by
            have := mul_le_mul_of_nonneg_right hskle (by positivity : (0 : ℚ) ≤ 2 ^ (-124 : ℤ))
            nlinarith
        _ ≤ 2 ^ (-117 : ℤ) := by norm_num)
    (mul_nonneg hsk0 he)
    (by
      rw [Format.binary32_eta]
      calc (sk d : ℚ) * 2 ^ (-150 : ℤ) ≤ 64 * 2 ^ (-150 : ℤ) :=
            mul_le_mul_of_nonneg_right hskle (by positivity)
        _ = 2 ^ (-144 : ℤ) := by norm_num)
    ha1' (abs_nonneg _) hdm0 (abs_nonneg _) hPM0
    (by positivity)
    (by positivity)
    (by
      have h1 : (sk d : ℚ) * (stepsOf d 0 : ℚ) ≤ 4096 :=
        (mul_le_mul_of_nonneg_left hT0 hsk0).trans hskT
      have h2 : 102 / 100 * ((sk d : ℚ) * (stepsOf d 0 : ℚ)) * alpha d ≤
          102 / 100 * 4096 * 2 ^ (-117 : ℤ) :=
        mul_le_mul (by linarith) hα' hα (by norm_num)
      calc 102 / 100 * (sk d : ℚ) * (stepsOf d 0 : ℚ) * alpha d
          = 102 / 100 * ((sk d : ℚ) * (stepsOf d 0 : ℚ)) * alpha d := by ring
        _ ≤ 102 / 100 * 4096 * 2 ^ (-117 : ℤ) := h2
        _ ≤ 2 ^ (-104 : ℤ) := by norm_num)
    hT1 (hT0.trans hTm) hσ hm
    (by
      have := mul_le_mul_of_nonneg_left hrho (by positivity : (0 : ℚ) ≤ 102 / 100 * (stepsOf d 0 : ℚ))
      nlinarith)
    hPM (by positivity) hE' hW hnum
  have htol : tol d va vb = (bk d : ℚ) * 2 ^ (-21 : ℤ) * (stepsOf d 0 : ℚ) *
      (|d.scaling.toRat| * dotMag d va vb) +
        2 ^ (-100 : ℤ) * (1 + |d.scaling.toRat| + dotMag d va vb) := by
    unfold tol partLen
    push_cast
    ring
  rw [htol]
  exact hmaster

/-- **The element is within `tol`.** If the operands `v p` (`p < SPLIT_K`) are within the `PartOk`
bounds of `s32 · partSum p`, and the element `y` is within `bound` (the atomic additions' error) of
their sum, then `y` is within `tol` of `scaling · dotSum`. -/
theorem total_bound (d : Inst) (hd : d.ok) (va vb : Nat → BF16) (_hdm : dotMag d va vb ≤ 2 ^ 120)
    (v : Nat → ℚ)
    (hv : ∀ p, p < sk d →
      |v p - s32 d * partSum d va vb p| ≤ |s32 d| * errPart d va vb p + mulErr d va vb p ∧
      |v p| ≤ |s32 d| * (|partSum d va vb p| + errPart d va vb p) + mulErr d va vb p)
    (y : ℚ)
    (hy : |y - ∑ p ∈ Finset.range (sk d), v p| ≤
      DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (sk d) (∑ p ∈ Finset.range (sk d), |v p|)) :
    |y - d.scaling.toRat * dotSum d va vb| ≤ tol d va vb := by
  obtain ⟨hs1, hs2⟩ := steps_zero_le d hd.K_top
  rcases cfg_cases d with h | h
  · have hsk := h.2.2.2.2
    have hbk := h.2.1
    refine total_case d hd va vb v hv y hy (259 * 2 ^ (-22 : ℤ)) (8 / 1000000) 8
      (by exact_mod_cast hs1 h.1) (by norm_num) (by rw [hsk]; norm_num)
      ((rho_le_num d).1 h.1) (by norm_num) (by norm_num) (by rw [hsk]; norm_num) (by norm_num)
      (by rw [hbk]; norm_num) (by rw [hbk]; norm_num)
  · have hsk := h.2.2.2.2
    have hbk := h.2.1
    refine total_case d hd va vb v hv y hy (35 * 2 ^ (-22 : ℤ)) (1 / 1000000) 512
      (by exact_mod_cast hs2 h.1) (by norm_num) (by rw [hsk]; norm_num)
      ((rho_le_num d).2 h.1) (by norm_num) (by norm_num) (by rw [hsk]; norm_num) (by norm_num)
      (by rw [hbk]; norm_num) (by rw [hbk]; norm_num)

end Claims.LoraShrink
