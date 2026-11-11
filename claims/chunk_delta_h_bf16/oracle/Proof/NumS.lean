import Proof.NumG
import Proof.NumV

/-!
# `chunk_delta_h_bf16`: one step of the state (lines 251–316)

One value column, one chunk: the state `Y` entering it is scaled by the decay (`S`), the gated
values `vb` go through `kᵀ` into the update `Yn`, and the claim's `errStep` bounds the distance of
`Yn` from the exact step.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA


/-- `√cl ≤ 8` for `cl ≤ 64`. -/
theorem step_sqrt_le {n m : ℕ} (hn : n ≤ m) (c : ℝ) (hc : 0 ≤ c) (hm : (m : ℝ) ≤ c ^ 2) :
    Real.sqrt n ≤ c := by
  refine Real.sqrt_le_iff.mpr ⟨hc, ?_⟩
  exact le_trans (by exact_mod_cast hn) hm

/-- **The gated values.** For the tokens `i < cl`: `|vb i| ≤ 2^71`; the rounding errors
`vb i − Gm i·vref i` have norm at most `2^-8·GN + 2^-60`; the lifted magnitudes of `vb` have norm
at most `(1 + 2^-8)·GN + 2^-59`. -/
theorem step_tokens {cl : ℕ} (hcl : cl ≤ 64) (Gm vref Z : ℕ → ℝ)
    (hGm : ∀ i, i < cl → 0 < Gm i ∧ Gm i ≤ 2 ^ 8)
    (hZ : ∀ i, i < cl → 0 ≤ Z i ∧ Z i ≤ 2 ^ 61) (hvr : ∀ i, i < cl → |vref i| ≤ Z i)
    (vb : ℕ → BF16)
    (hvb : ∀ i, i < cl → |(vb i).toReal - Gm i * vref i| ≤
      2 ^ (-8 : ℤ) * (Gm i * vAbsA (vref i) (Z i)) + 2 ^ (-63 : ℤ))
    (GN : ℝ) (hGN : GN = colNorm cl (fun i => Gm i * vAbsA (vref i) (Z i))) :
    (∀ i, i < cl → |(vb i).toReal| ≤ 2 ^ 71) ∧
      colNorm cl (fun i => (vb i).toReal - Gm i * vref i) ≤ 2 ^ (-8 : ℤ) * GN + 2 ^ (-60 : ℤ) ∧
      colNorm cl (fun i => lmB (vb i)) ≤ (1 + 2 ^ (-8 : ℤ)) * GN + 2 ^ (-59 : ℤ) := by
  have hG : ∀ i, i < cl → 0 ≤ Gm i * vAbsA (vref i) (Z i) ∧
      |Gm i * vref i| ≤ Gm i * vAbsA (vref i) (Z i) ∧ Gm i * vAbsA (vref i) (Z i) ≤ 2 ^ 70 := by
    intro i hi
    obtain ⟨g0, g1⟩ := hGm i hi
    obtain ⟨z0, z1⟩ := hZ i hi
    have hv := hvr i hi
    unfold vAbsA
    have e5 : (2 : ℝ) ^ (-5 : ℤ) = 1 / 32 := by norm_num
    rw [e5]
    have hV0 : 0 ≤ |vref i| + 1 / 32 * Z i := by positivity
    refine ⟨by positivity, ?_, ?_⟩
    · rw [abs_mul, abs_of_pos g0]
      exact mul_le_mul_of_nonneg_left (by linarith) g0.le
    · have : |vref i| + 1 / 32 * Z i ≤ 2 ^ 62 := by nlinarith
      calc Gm i * (|vref i| + 1 / 32 * Z i) ≤ 2 ^ 8 * 2 ^ 62 :=
            mul_le_mul g1 this hV0 (by positivity)
        _ = 2 ^ 70 := by norm_num
  have e8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 := by norm_num
  have e63 : (2 : ℝ) ^ (-63 : ℤ) = 1 / 2 ^ 63 := by norm_num
  have e126 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have hs8 : Real.sqrt cl ≤ 8 := step_sqrt_le hcl 8 (by norm_num) (by norm_num)
  refine ⟨fun i hi => ?_, ?_, ?_⟩
  · obtain ⟨h0, h1, h2⟩ := hG i hi
    have := hvb i hi
    rw [e8, e63] at this
    have h3 := abs_sub_abs_le_abs_sub (vb i).toReal (Gm i * vref i)
    have : |(vb i).toReal| ≤ 2 ^ 71 := by nlinarith
    exact this
  · have := colNorm_lin2_le (n := cl) (x := fun i => (vb i).toReal - Gm i * vref i)
      (y := fun i => Gm i * vAbsA (vref i) (Z i)) (z := fun _ => 0) (a := 2 ^ (-8 : ℤ)) (a' := 0)
      (b := 2 ^ (-63 : ℤ)) (by positivity) le_rfl (by positivity) (fun i hi => by
        have := hvb i hi
        have h0 := (hG i hi).1
        rw [abs_of_nonneg h0]; simpa using this)
    rw [← hGN] at this
    have h8 : Real.sqrt cl * 2 ^ (-63 : ℤ) ≤ 2 ^ (-60 : ℤ) := by
      rw [e63]; have : (2 : ℝ) ^ (-60 : ℤ) = 8 * (1 / 2 ^ 63) := by norm_num
      rw [this]; exact mul_le_mul_of_nonneg_right hs8 (by positivity)
    linarith
  · have := colNorm_lin2_le (n := cl) (x := fun i => lmB (vb i))
      (y := fun i => Gm i * vAbsA (vref i) (Z i)) (z := fun _ => 0) (a := 1 + 2 ^ (-8 : ℤ)) (a' := 0)
      (b := 2 ^ (-63 : ℤ) + 2 ^ (-126 : ℤ)) (by positivity) le_rfl (by positivity) (fun i hi => by
        obtain ⟨h0, h1, h2⟩ := hG i hi
        have hv := hvb i hi
        have hl := lmB_le (vb i)
        have h3 := abs_sub_abs_le_abs_sub (vb i).toReal (Gm i * vref i)
        rw [abs_of_nonneg (lmB_nonneg _), abs_of_nonneg h0]
        simp only [abs_zero, mul_zero, add_zero]
        rw [e8, e63, e126] at *
        nlinarith)
    rw [← hGN] at this
    have h8 : Real.sqrt cl * (2 ^ (-63 : ℤ) + 2 ^ (-126 : ℤ)) ≤ 2 ^ (-59 : ℤ) := by
      have : (2 : ℝ) ^ (-59 : ℤ) = 8 * (1 / 2 ^ 62) := by norm_num
      rw [this]
      calc _ ≤ 8 * (2 ^ (-63 : ℤ) + 2 ^ (-126 : ℤ)) :=
            mul_le_mul_of_nonneg_right hs8 (by positivity)
        _ ≤ _ := by norm_num
    linarith


/-- The lifted magnitudes of `k` over the chunk: bounded, zero outside `i < cl`, `c < K`, and their
Frobenius norm `klv` is at most `2^23 + 1`. -/
theorem step_klv {K nb cl : ℕ} (hKle : K ≤ 256) (hnb : K ≤ 64 * nb) (hcl : cl ≤ 64)
    (kk : ℕ → ℕ → BF16)
    (hk : ∀ i c, i < 64 → c < 64 * nb → (kk i c).isFinite = true ∧ |(kk i c).toReal| ≤ 2 ^ 16)
    (klv : ℝ) (hkl : klv = Real.sqrt (∑ i ∈ Finset.range cl, ∑ c ∈ Finset.range K, lmB (kk i c) ^ 2)) :
    klv ≤ 2 ^ 23 + 1 := by
  have e126 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have hb : ∀ i ∈ Finset.range cl, ∀ c ∈ Finset.range K, lmB (kk i c) ^ 2 ≤ (2 ^ 16 + 1 / 2 ^ 100) ^ 2 := by
    intro i hi c hc
    have hi' := Finset.mem_range.mp hi
    have hc' := Finset.mem_range.mp hc
    have h1 := lmB_le (kk i c)
    have h2 := (hk i c (by omega) (by omega)).2
    have h3 := lmB_nonneg (kk i c)
    rw [e126] at h1
    have h4 : (1 : ℝ) / 2 ^ 126 ≤ 1 / 2 ^ 100 := by
      apply one_div_le_one_div_of_le (by positivity); norm_num
    exact pow_le_pow_left₀ h3 (by linarith) 2
  have hsum : ∑ i ∈ Finset.range cl, ∑ c ∈ Finset.range K, lmB (kk i c) ^ 2 ≤
      (cl : ℝ) * ((K : ℝ) * (2 ^ 16 + 1 / 2 ^ 100) ^ 2) := by
    calc _ ≤ ∑ i ∈ Finset.range cl, (K : ℝ) * (2 ^ 16 + 1 / 2 ^ 100) ^ 2 := by
          refine Finset.sum_le_sum fun i hi => ?_
          calc _ ≤ ∑ c ∈ Finset.range K, (2 ^ 16 + 1 / 2 ^ 100 : ℝ) ^ 2 :=
                Finset.sum_le_sum fun c hc => hb i hi c hc
            _ = _ := by simp
      _ = _ := by simp
  rw [hkl]
  refine Real.sqrt_le_iff.mpr ⟨by positivity, hsum.trans ?_⟩
  have hc : (cl : ℝ) ≤ 64 := by exact_mod_cast hcl
  have hK : (K : ℝ) ≤ 256 := by exact_mod_cast hKle
  have hq : 0 ≤ ((2 : ℝ) ^ 16 + 1 / 2 ^ 100) ^ 2 := by positivity
  calc (cl : ℝ) * ((K : ℝ) * (2 ^ 16 + 1 / 2 ^ 100) ^ 2)
      ≤ 64 * (256 * (2 ^ 16 + 1 / 2 ^ 100) ^ 2) := by
        refine mul_le_mul hc (mul_le_mul_of_nonneg_right hK hq) (by positivity) (by positivity)
    _ ≤ _ := by norm_num

/-- **The `k` dot's lifted terms through the norm**: the vector of the lifted dot sums over the
tokens has norm at most `klv·‖lm vb‖` (Cauchy–Schwarz on each row, then over the rows). -/
theorem step_T {K nb cl : ℕ} (hnb : K ≤ 64 * nb) (hcl : cl ≤ 64) (kk : ℕ → ℕ → BF16)
    (hkz : ∀ i c, i < 64 → c < 64 * nb → (cl ≤ i ∨ K ≤ c) → (kk i c).toReal = 0)
    (vb : ℕ → BF16) (klv : ℝ)
    (hkl : klv = Real.sqrt (∑ i ∈ Finset.range cl, ∑ c ∈ Finset.range K, lmB (kk i c) ^ 2)) :
    colNorm K (fun c => ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i)) ≤
      klv * colNorm cl (fun i => lmB (vb i)) := by
  have hrow : ∀ c, c < K → ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i) =
      ∑ i ∈ Finset.range cl, lmB (kk i c) * lmB (vb i) := by
    intro c hc
    symm
    refine Finset.sum_subset (Finset.range_subset_range.mpr hcl) fun i hi hni => ?_
    have h1 := Finset.mem_range.mp hi
    have h2 := mt Finset.mem_range.mpr hni
    rw [lmB_zero (hkz i c h1 (by omega) (Or.inl (by omega))), zero_mul]
  set L := colNorm cl (fun i => lmB (vb i)) with hL
  set r : ℕ → ℝ := fun c => colNorm cl (fun i => lmB (kk i c)) with hr
  have hbound := colNorm_lin2_le (n := K)
    (x := fun c => ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i)) (y := r) (z := fun _ => 0)
    (a := L) (a' := 0) (b := 0) (colNorm_nonneg _ _) le_rfl le_rfl (fun c hc => by
      rw [hrow c hc]
      have hnn : 0 ≤ ∑ i ∈ Finset.range cl, lmB (kk i c) * lmB (vb i) :=
        Finset.sum_nonneg fun i _ => mul_nonneg (lmB_nonneg _) (lmB_nonneg _)
      rw [abs_of_nonneg hnn]
      have := sum_abs_mul_le_colNorm cl (fun i => lmB (kk i c)) (fun i => lmB (vb i))
      simp only [abs_of_nonneg (lmB_nonneg _)] at this
      have hr0 : 0 ≤ r c := colNorm_nonneg _ _
      rw [abs_of_nonneg hr0]
      simp only [abs_zero, mul_zero, add_zero]
      linarith [mul_comm L (r c)])
  have hrk : colNorm K r = klv := by
    rw [hkl]
    unfold colNorm
    congr 1
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun c _ => ?_
    simp only [hr]
    unfold colNorm
    rw [Real.sq_sqrt (Finset.sum_nonneg fun i _ => sq_nonneg _)]
  have h0 : colNorm K (fun _ => (0 : ℝ)) = 0 := by unfold colNorm; simp
  rw [hrk, h0] at hbound
  simp only [mul_zero, add_zero] at hbound
  linarith [mul_comm L klv]


/-- **The arithmetic of the state step**: the five norm bounds put together. `Ny` the norm of the
state, `eB` the decay bound, `ns` the norm of the scaled state, `nA`, `nB`, `nC`, `nD` the norms of
the update, scaling, decay and token-rounding errors, `nT` that of the lifted terms. -/
theorem step_arith {gam errE eB ρ2 α2 Ny hn GN klv sK L nA nB nC nD nT ns sq : ℝ}
    (hgam : 0 < gam) (hgam8 : gam ≤ 2 ^ 8) (herrE0 : 0 ≤ errE) (herrE : errE ≤ 2 ^ (-22 : ℤ) * gam + mu)
    (heB : eB = gam + errE) (hρ2 : 0 ≤ ρ2 ∧ ρ2 ≤ 16 / 10 ^ 6) (hα2 : 0 ≤ α2 ∧ α2 ≤ 2 ^ (-110 : ℤ))
    (hn0 : 0 ≤ hn) (hn40 : hn ≤ 2 ^ 40) (hNy0 : 0 ≤ Ny) (hNy : Ny ≤ 1004 / 1000 * hn + 2 ^ (-129 : ℤ))
    (hsq : sq ≤ 16)
    (hns : ns ≤ (1 + u32) * eB * Ny + sq * e32) (hnB : nB ≤ u32 * eB * Ny + sq * e32)
    (hnC : nC ≤ errE * Ny) (hnA : nA ≤ ρ2 * ns + ρ2 * nT + sq * α2) (hnT : nT ≤ klv * L)
    (hL : L ≤ (1 + 2 ^ (-8 : ℤ)) * GN + 2 ^ (-59 : ℤ))
    (hnD : nD ≤ sK * (2 ^ (-8 : ℤ) * GN + 2 ^ (-60 : ℤ)))
    (hGN0 : 0 ≤ GN) (hklv0 : 0 ≤ klv) (hklv : klv ≤ 2 ^ 23 + 1) (hsK : sK ≤ klv)
    (hnT0 : 0 ≤ nT) (hns0 : 0 ≤ ns) :
    nA + nB + nC + nD ≤
      2 ^ (-8 : ℤ) * sK * GN + 2 ^ (-15 : ℤ) * (klv * GN + gam * hn) + 2 ^ (-36 : ℤ) := by
  have hmu : mu = 1 / 2 ^ 126 := by unfold mu; norm_num
  have hu : u32 = 1 / 2 ^ 24 := by unfold u32; norm_num
  have he : e32 = 1 / 2 ^ 150 := by unfold e32; norm_num
  have e8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 := by norm_num
  have e15 : (2 : ℝ) ^ (-15 : ℤ) = 1 / 2 ^ 15 := by norm_num
  have e22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
  have e36 : (2 : ℝ) ^ (-36 : ℤ) = 1 / 2 ^ 36 := by norm_num
  have e59 : (2 : ℝ) ^ (-59 : ℤ) = 1 / 2 ^ 59 := by norm_num
  have e60 : (2 : ℝ) ^ (-60 : ℤ) = 1 / 2 ^ 60 := by norm_num
  have e110 : (2 : ℝ) ^ (-110 : ℤ) = 1 / 2 ^ 110 := by norm_num
  have e129 : (2 : ℝ) ^ (-129 : ℤ) = 1 / 2 ^ 129 := by norm_num
  rw [hmu, e22] at herrE
  rw [hu, he] at hns hnB
  rw [e8, e59] at hL
  rw [e8, e60] at hnD
  rw [e110] at hα2
  rw [e129] at hNy
  rw [e8, e15, e36]
  set P := gam * hn with hP
  have hP0 : 0 ≤ P := by positivity
  have hNy2 : Ny ≤ 2 ^ 41 := by
    have : (1004 / 1000 : ℝ) * hn ≤ 1004 / 1000 * 2 ^ 40 := by nlinarith
    linarith [show (1 : ℝ) / 2 ^ 129 ≤ 1 by norm_num, show (1004 / 1000 : ℝ) * 2 ^ 40 ≤ 2 ^ 41 - 1 by norm_num]
  have heB0 : 0 ≤ eB := by rw [heB]; linarith
  have heBle : eB ≤ (1 + 1 / 2 ^ 22) * gam + 1 / 2 ^ 126 := by rw [heB]; linarith
  -- the decay times the state
  have hF1 : eB * Ny ≤ 10041 / 10000 * P + 1 / 2 ^ 80 := by
    have h1 : eB * Ny ≤ ((1 + 1 / 2 ^ 22) * gam + 1 / 2 ^ 126) * (1004 / 1000 * hn + 1 / 2 ^ 129) :=
      mul_le_mul heBle hNy hNy0 (by positivity)
    have h2 : gam * (1 / 2 ^ 129 : ℝ) ≤ 2 ^ 8 * (1 / 2 ^ 129) :=
      mul_le_mul_of_nonneg_right hgam8 (by positivity)
    have h3 : (1 / 2 ^ 126 : ℝ) * hn ≤ 1 / 2 ^ 126 * 2 ^ 40 :=
      mul_le_mul_of_nonneg_left hn40 (by positivity)
    nlinarith
  have hF2 : errE * Ny ≤ 1 / 2 ^ 21 * P + 1 / 2 ^ 80 := by
    have h1 : errE * Ny ≤ (1 / 2 ^ 22 * gam + 1 / 2 ^ 126) * (1004 / 1000 * hn + 1 / 2 ^ 129) :=
      mul_le_mul herrE hNy hNy0 (by positivity)
    have h2 : gam * (1 / 2 ^ 129 : ℝ) ≤ 2 ^ 8 * (1 / 2 ^ 129) :=
      mul_le_mul_of_nonneg_right hgam8 (by positivity)
    have h3 : (1 / 2 ^ 126 : ℝ) * hn ≤ 1 / 2 ^ 126 * 2 ^ 40 :=
      mul_le_mul_of_nonneg_left hn40 (by positivity)
    nlinarith
  have hsq1 : sq * (1 / 2 ^ 150 : ℝ) ≤ 16 * (1 / 2 ^ 150) :=
    mul_le_mul_of_nonneg_right hsq (by positivity)
  have hns' : ns ≤ 10042 / 10000 * P + 1 / 2 ^ 70 := by
    have : (1 + 1 / 2 ^ 24 : ℝ) * (eB * Ny) ≤ (1 + 1 / 2 ^ 24) * (10041 / 10000 * P + 1 / 2 ^ 80) :=
      mul_le_mul_of_nonneg_left hF1 (by positivity)
    nlinarith
  have hnB' : nB ≤ 1 / 2 ^ 23 * P + 1 / 2 ^ 70 := by
    have : (1 / 2 ^ 24 : ℝ) * (eB * Ny) ≤ (1 / 2 ^ 24) * (10041 / 10000 * P + 1 / 2 ^ 80) :=
      mul_le_mul_of_nonneg_left hF1 (by positivity)
    nlinarith
  have hkL : klv * L ≤ klv * ((1 + 1 / 256) * GN + 1 / 2 ^ 59) :=
    mul_le_mul_of_nonneg_left hL hklv0
  have hk59 : klv * (1 / 2 ^ 59 : ℝ) ≤ (2 ^ 23 + 1) * (1 / 2 ^ 59) :=
    mul_le_mul_of_nonneg_right hklv (by positivity)
  have hρns : ρ2 * ns ≤ 16 / 10 ^ 6 * (10042 / 10000 * P + 1 / 2 ^ 70) :=
    mul_le_mul hρ2.2 hns' hns0 (by norm_num)
  have hρnT : ρ2 * nT ≤ 16 / 10 ^ 6 * (klv * ((1 + 1 / 256) * GN + 1 / 2 ^ 59)) :=
    mul_le_mul hρ2.2 (hnT.trans hkL) hnT0 (by norm_num)
  have hsα : sq * α2 ≤ 16 * (1 / 2 ^ 110) := mul_le_mul hsq hα2.2 hα2.1 (by norm_num)
  have hsK59 : sK * (1 / 2 ^ 60 : ℝ) ≤ (2 ^ 23 + 1) * (1 / 2 ^ 60) :=
    mul_le_mul (hsK.trans hklv) le_rfl (by positivity) (by positivity)
  have hQ : 0 ≤ klv * GN := mul_nonneg hklv0 hGN0
  have hR : sK * (1 / 256 * GN) = 1 / 256 * sK * GN := by ring
  nlinarith

/-- **The room of the update**: for a state value at most `2^115` in magnitude, the update's room
condition holds. -/
theorem step_room {K nb cl : ℕ} (kk : ℕ → ℕ → BF16)
    (hk : ∀ i c, i < 64 → c < 64 * nb → (kk i c).isFinite = true ∧ |(kk i c).toReal| ≤ 2 ^ 16)
    (hkz : ∀ i c, i < 64 → c < 64 * nb → (cl ≤ i ∨ K ≤ c) → (kk i c).toReal = 0)
    (vb : ℕ → BF16) (hv : ∀ i, i < cl → |(vb i).toReal| ≤ 2 ^ 71) {c : ℕ} (hc : c < 64 * nb)
    {sv : ℝ} (hsv : |sv| ≤ 2 ^ 115) :
    |sv| + ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i) ≤ 2 ^ 120 := by
  have e126 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have hterm : ∀ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i) ≤ 2 ^ 89 := by
    intro i hi
    have hi' := Finset.mem_range.mp hi
    by_cases hz : cl ≤ i ∨ K ≤ c
    · rw [lmB_zero (hkz i c hi' hc hz), zero_mul]; positivity
    · rw [not_or, not_le, not_le] at hz
      have h1 := lmB_le (kk i c)
      have h2 := (hk i c hi' hc).2
      have h3 := lmB_le (vb i)
      have h4 := hv i hz.1
      rw [e126] at h1 h3
      have h5 : lmB (kk i c) ≤ 2 ^ 17 := by
        have : (1 : ℝ) / 2 ^ 126 ≤ 1 := by norm_num
        linarith [show (2 : ℝ) ^ 16 + 1 ≤ 2 ^ 17 by norm_num]
      have h6 : lmB (vb i) ≤ 2 ^ 72 := by
        have : (1 : ℝ) / 2 ^ 126 ≤ 1 := by norm_num
        linarith [show (2 : ℝ) ^ 71 + 1 ≤ 2 ^ 72 by norm_num]
      calc lmB (kk i c) * lmB (vb i) ≤ 2 ^ 17 * 2 ^ 72 :=
            mul_le_mul h5 h6 (lmB_nonneg _) (by positivity)
        _ = 2 ^ 89 := by norm_num
  have hsum : ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i) ≤ 64 * 2 ^ 89 := by
    calc _ ≤ ∑ i ∈ Finset.range 64, (2 : ℝ) ^ 89 := Finset.sum_le_sum hterm
      _ = _ := by simp
  linarith [show (2 : ℝ) ^ 115 + 64 * 2 ^ 89 ≤ 2 ^ 120 by norm_num]


/-- **One step of the state, for a nonnegative `sK`.** -/
theorem step_core {K nb cl : ℕ} (hKle : K ≤ 256) (hnb : K ≤ 64 * nb) (hcl : cl ≤ 64)
    (Y : ℕ → F32) (hh : ℕ → ℝ)
    (hY : ∀ c, c < K → (Y c).isFinite = true ∧
      |hh c - (Y c).toReal| ≤ 2 ^ (-8 : ℤ) * |(Y c).toReal| + 2 ^ (-134 : ℤ))
    (hhn : colNorm K hh ≤ 2 ^ 40) (pb : ℝ) (hpb : pb ≤ 2 ^ 100)
    (hpad : ∀ c, K ≤ c → c < 64 * nb → (Y c).isFinite = true ∧ |(Y c).toReal| ≤ pb)
    (gam errE gt : ℝ) (hgam : 0 < gam) (hgam8 : gam ≤ 2 ^ 8)
    (herrE : 0 ≤ errE ∧ errE ≤ 2 ^ (-22 : ℤ) * gam + mu) (hgt : |gt - gam| ≤ errE)
    (S : ℕ → F32)
    (hS : ∀ c, c < 64 * nb → (S c).isFinite = true ∧
      |(S c).toReal - gt * (Y c).toReal| ≤ u32 * |gt * (Y c).toReal| + e32)
    (Gm vref Z : ℕ → ℝ) (hGm : ∀ i, i < cl → 0 < Gm i ∧ Gm i ≤ 2 ^ 8)
    (hZ : ∀ i, i < cl → 0 ≤ Z i ∧ Z i ≤ 2 ^ 61) (hvr : ∀ i, i < cl → |vref i| ≤ Z i)
    (vb : ℕ → BF16)
    (hvb : ∀ i, i < cl → |(vb i).toReal - Gm i * vref i| ≤
      2 ^ (-8 : ℤ) * (Gm i * vAbsA (vref i) (Z i)) + 2 ^ (-63 : ℤ))
    (kk : ℕ → ℕ → BF16)
    (hk : ∀ i c, i < 64 → c < 64 * nb → (kk i c).isFinite = true ∧ |(kk i c).toReal| ≤ 2 ^ 16)
    (hkz : ∀ i c, i < 64 → c < 64 * nb → (cl ≤ i ∨ K ≤ c) → (kk i c).toReal = 0)
    (ρ2 α2 : ℝ) (hρ2 : 0 ≤ ρ2 ∧ ρ2 ≤ 16 / 10 ^ 6) (hα2 : 0 ≤ α2 ∧ α2 ≤ 2 ^ (-110 : ℤ))
    (Yn : ℕ → F32)
    (hupd : ∀ c, c < 64 * nb → (S c).isFinite = true →
      |(S c).toReal| + ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i) ≤ 2 ^ 120 →
      (Yn c).isFinite = true ∧
        |(Yn c).toReal - ((S c).toReal + ∑ i ∈ Finset.range 64, (kk i c).toReal * (vb i).toReal)| ≤
          ρ2 * (|(S c).toReal| + ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i)) + α2)
    (sK klv GN : ℝ) (hsK0 : 0 ≤ sK)
    (hsK : ∀ x : ℕ → ℝ, colNorm K (fun c => ∑ i ∈ Finset.range cl, (kk i c).toReal * x i) ≤
      sK * colNorm cl x)
    (hsKle : sK ≤ klv)
    (hkl : klv = Real.sqrt (∑ i ∈ Finset.range cl, ∑ c ∈ Finset.range K, lmB (kk i c) ^ 2))
    (hGN : GN = colNorm cl (fun i => Gm i * vAbsA (vref i) (Z i))) :
    (∀ c, c < 64 * nb → (Yn c).isFinite = true) ∧
      colNorm K (fun c => (Yn c).toReal -
          (gam * (Y c).toReal + ∑ i ∈ Finset.range cl, (kk i c).toReal * Gm i * vref i)) ≤
        2 ^ (-8 : ℤ) * sK * GN + 2 ^ (-15 : ℤ) * (klv * GN + gam * colNorm K hh) + 2 ^ (-36 : ℤ) := by
  obtain ⟨herrE0, herrE1⟩ := herrE
  have e133 : (2 : ℝ) ^ (-133 : ℤ) = 1 / 2 ^ 133 := by norm_num
  have e134 : (2 : ℝ) ^ (-134 : ℤ) = 1 / 2 ^ 134 := by norm_num
  have e129 : (2 : ℝ) ^ (-129 : ℤ) = 1 / 2 ^ 129 := by norm_num
  have e8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 := by norm_num
  have e22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
  have hu : u32 = 1 / 2 ^ 24 := by unfold u32; norm_num
  have he : e32 = 1 / 2 ^ 150 := by unfold e32; norm_num
  have hmu : mu = 1 / 2 ^ 126 := by unfold mu; norm_num
  have hsqK : Real.sqrt K ≤ 16 := step_sqrt_le hKle 16 (by norm_num) (by norm_num)
  have hsqK0 : 0 ≤ Real.sqrt K := Real.sqrt_nonneg _
  -- the state
  have hy1 : ∀ c, c < K → |(Y c).toReal| ≤ 1004 / 1000 * |hh c| + 2 ^ (-133 : ℤ) := by
    intro c hc
    have h1 := (hY c hc).2
    have h2 := abs_sub_abs_le_abs_sub (Y c).toReal (hh c)
    rw [abs_sub_comm] at h2
    rw [e8, e134] at h1
    rw [e133]
    have := abs_nonneg (hh c)
    linarith
  have hn0 : 0 ≤ colNorm K hh := colNorm_nonneg _ _
  have hNy : colNorm K (fun c => (Y c).toReal) ≤ 1004 / 1000 * colNorm K hh + 2 ^ (-129 : ℤ) := by
    have := colNorm_lin2_le (n := K) (x := fun c => (Y c).toReal) (y := hh) (z := fun _ => 0)
      (a := 1004 / 1000) (a' := 0) (b := 2 ^ (-133 : ℤ)) (by norm_num) le_rfl (by positivity)
      (fun i hi => by simp only [abs_zero, mul_zero, add_zero]; exact hy1 i hi)
    simp only [zero_mul, add_zero] at this
    have h2 : Real.sqrt K * 2 ^ (-133 : ℤ) ≤ 16 * (1 / 2 ^ 133) := by
      rw [e133]; exact mul_le_mul_of_nonneg_right hsqK (by positivity)
    rw [e129]; linarith
  have hyall : ∀ c, c < 64 * nb → |(Y c).toReal| ≤ 2 ^ 100 := by
    intro c hc
    by_cases hcK : c < K
    · have h1 := hy1 c hcK
      have h2 := (abs_le_colNorm hh hcK).trans hhn
      rw [e133] at h1
      nlinarith [abs_nonneg (hh c)]
    · exact ((hpad c (not_lt.mp hcK) hc).2).trans hpb
  -- the decay
  have hgt1 : |gt| ≤ gam + errE := by
    have := abs_sub_abs_le_abs_sub gt gam
    rw [abs_of_pos hgam] at this
    linarith
  have hgt9 : |gt| ≤ 2 ^ 9 := by
    have h1 := herrE1
    rw [e22, hmu] at h1
    have h2 : (1 : ℝ) / 2 ^ 22 * gam ≤ 1 / 2 ^ 22 * 2 ^ 8 :=
      mul_le_mul_of_nonneg_left hgam8 (by positivity)
    linarith [show (2 : ℝ) ^ 8 + 1 / 2 ^ 22 * 2 ^ 8 + 1 / 2 ^ 126 ≤ 2 ^ 9 by norm_num]
  -- the tokens
  obtain ⟨hv71, hδ, hL⟩ := step_tokens hcl Gm vref Z hGm hZ hvr vb hvb GN hGN
  have hklv : klv ≤ 2 ^ 23 + 1 := step_klv hKle hnb hcl kk hk klv hkl
  have hklv0 : 0 ≤ klv := by rw [hkl]; exact Real.sqrt_nonneg _
  -- the scaled state
  have hsabs : ∀ c, c < 64 * nb → |(S c).toReal| ≤ (1 + u32) * |gt| * |(Y c).toReal| + e32 := by
    intro c hc
    have h1 := (hS c hc).2
    rw [abs_mul] at h1
    have h2 := abs_sub_abs_le_abs_sub (S c).toReal (gt * (Y c).toReal)
    rw [abs_mul] at h2
    nlinarith
  have hs115 : ∀ c, c < 64 * nb → |(S c).toReal| ≤ 2 ^ 115 := by
    intro c hc
    have h1 := hsabs c hc
    have h2 : |gt| * |(Y c).toReal| ≤ 2 ^ 9 * 2 ^ 100 :=
      mul_le_mul hgt9 (hyall c hc) (abs_nonneg _) (by positivity)
    rw [hu, he] at h1
    nlinarith
  -- finiteness
  have hfin : ∀ c, c < 64 * nb → (Yn c).isFinite = true ∧
      |(Yn c).toReal - ((S c).toReal + ∑ i ∈ Finset.range 64, (kk i c).toReal * (vb i).toReal)| ≤
        ρ2 * (|(S c).toReal| + ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i)) + α2 := fun c hc =>
    hupd c hc (hS c hc).1 (step_room kk hk hkz vb hv71 hc (hs115 c hc))
  refine ⟨fun c hc => (hfin c hc).1, ?_⟩
  -- the error decomposition
  obtain ⟨a, ha⟩ : ∃ a : ℕ → ℝ, ∀ c, a c = (Yn c).toReal -
      ((S c).toReal + ∑ i ∈ Finset.range 64, (kk i c).toReal * (vb i).toReal) := ⟨_, fun _ => rfl⟩
  obtain ⟨b, hb⟩ : ∃ b : ℕ → ℝ, ∀ c, b c = (S c).toReal - gt * (Y c).toReal := ⟨_, fun _ => rfl⟩
  obtain ⟨cc, hcc⟩ : ∃ cc : ℕ → ℝ, ∀ c, cc c = (gt - gam) * (Y c).toReal := ⟨_, fun _ => rfl⟩
  obtain ⟨d, hd⟩ : ∃ d : ℕ → ℝ, ∀ c, d c =
      ∑ i ∈ Finset.range cl, (kk i c).toReal * ((vb i).toReal - Gm i * vref i) := ⟨_, fun _ => rfl⟩
  obtain ⟨T, hT⟩ : ∃ T : ℕ → ℝ, ∀ c, T c =
      ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i) := ⟨_, fun _ => rfl⟩
  have hsum64 : ∀ c, c < K → ∑ i ∈ Finset.range 64, (kk i c).toReal * (vb i).toReal =
      ∑ i ∈ Finset.range cl, (kk i c).toReal * (vb i).toReal := by
    intro c hc
    symm
    refine Finset.sum_subset (Finset.range_subset_range.mpr hcl) fun i hi hni => ?_
    have h1 := Finset.mem_range.mp hi
    have h2 := mt Finset.mem_range.mpr hni
    rw [hkz i c (by omega) (by omega) (Or.inl (by omega)), zero_mul]
  have hcong : colNorm K (fun c => (Yn c).toReal -
      (gam * (Y c).toReal + ∑ i ∈ Finset.range cl, (kk i c).toReal * Gm i * vref i)) =
      colNorm K (fun c => a c + b c + cc c + d c) := by
    refine colNorm_congr fun c hc => ?_
    rw [ha, hb, hcc, hd, hsum64 c hc]
    have : ∑ i ∈ Finset.range cl, (kk i c).toReal * ((vb i).toReal - Gm i * vref i) =
        ∑ i ∈ Finset.range cl, (kk i c).toReal * (vb i).toReal -
          ∑ i ∈ Finset.range cl, (kk i c).toReal * Gm i * vref i := by
      rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl fun i _ => by ring
    rw [this]; ring
  rw [hcong]
  -- the five norms
  have hρ0 := hρ2.1
  have hα0 := hα2.1
  have hnA : colNorm K a ≤ ρ2 * colNorm K (fun c => (S c).toReal) + ρ2 * colNorm K T +
      Real.sqrt K * α2 :=
    colNorm_lin2_le hρ0 hρ0 hα0 (fun c hc => by
      have h := (hfin c (by omega)).2
      have hnn : 0 ≤ T c := by
        rw [hT]; exact Finset.sum_nonneg fun i _ => mul_nonneg (lmB_nonneg _) (lmB_nonneg _)
      rw [ha, abs_of_nonneg hnn, hT]
      rw [hT] at hnn
      linarith)
  have hnT : colNorm K T ≤ klv * colNorm cl (fun i => lmB (vb i)) := by
    have : T = fun c => ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i) := funext hT
    rw [this]; exact step_T hnb hcl kk hkz vb klv hkl
  have hns : colNorm K (fun c => (S c).toReal) ≤
      (1 + u32) * (gam + errE) * colNorm K (fun c => (Y c).toReal) + Real.sqrt K * e32 := by
    have hu0 : 0 ≤ u32 := by rw [hu]; positivity
    have := colNorm_lin2_le (n := K) (x := fun c => (S c).toReal) (y := fun c => (Y c).toReal)
      (z := fun _ => 0) (a := (1 + u32) * (gam + errE)) (a' := 0) (b := e32)
      (by positivity) le_rfl (by rw [he]; positivity) (fun c hc => by
        have h1 := hsabs c (by omega)
        have h2 : (1 + u32) * |gt| ≤ (1 + u32) * (gam + errE) :=
          mul_le_mul_of_nonneg_left hgt1 (by positivity)
        have h3 := mul_le_mul_of_nonneg_right h2 (abs_nonneg (Y c).toReal)
        simp only [abs_zero, mul_zero, add_zero]
        nlinarith)
    simpa using this
  have hnB : colNorm K b ≤ u32 * (gam + errE) * colNorm K (fun c => (Y c).toReal) +
      Real.sqrt K * e32 := by
    have hu0 : 0 ≤ u32 := by rw [hu]; positivity
    have := colNorm_lin2_le (n := K) (x := b) (y := fun c => (Y c).toReal)
      (z := fun _ => 0) (a := u32 * (gam + errE)) (a' := 0) (b := e32)
      (by positivity) le_rfl (by rw [he]; positivity) (fun c hc => by
        have h1 := (hS c (by omega)).2
        rw [hb, abs_mul] at *
        have h2 : u32 * |gt| ≤ u32 * (gam + errE) := mul_le_mul_of_nonneg_left hgt1 hu0
        have h3 := mul_le_mul_of_nonneg_right h2 (abs_nonneg (Y c).toReal)
        simp only [abs_zero, mul_zero, add_zero]
        nlinarith)
    simpa using this
  have hnC : colNorm K cc ≤ errE * colNorm K (fun c => (Y c).toReal) := by
    have := colNorm_lin2_le (n := K) (x := cc) (y := fun c => (Y c).toReal)
      (z := fun _ => 0) (a := errE) (a' := 0) (b := 0)
      herrE0 le_rfl le_rfl (fun c hc => by
        rw [hcc, abs_mul]
        simp only [abs_zero, mul_zero, add_zero]
        exact mul_le_mul_of_nonneg_right hgt (abs_nonneg _))
    simpa using this
  have hnD : colNorm K d ≤ sK * (2 ^ (-8 : ℤ) * GN + 2 ^ (-60 : ℤ)) := by
    have h1 := hsK (fun i => (vb i).toReal - Gm i * vref i)
    have h2 : d = fun c => ∑ i ∈ Finset.range cl,
        (kk i c).toReal * ((vb i).toReal - Gm i * vref i) := funext hd
    rw [h2]
    exact h1.trans (mul_le_mul_of_nonneg_left hδ hsK0)
  have hcsum : colNorm K (fun c => a c + b c + cc c + d c) ≤
      colNorm K a + colNorm K b + colNorm K cc + colNorm K d := by
    have h1 := colNorm_add_le K (fun c => a c + b c + cc c) d
    have h2 := colNorm_add_le K (fun c => a c + b c) cc
    have h3 := colNorm_add_le K a b
    linarith
  refine hcsum.trans ?_
  exact step_arith (eB := gam + errE) (sq := Real.sqrt K) hgam hgam8 herrE0 herrE1 rfl hρ2 hα2
    hn0 hhn (colNorm_nonneg _ _) hNy hsqK hns hnB hnC hnA hnT hL hnD
    (by rw [hGN]; exact colNorm_nonneg _ _) hklv0 hklv hsKle (colNorm_nonneg _ _)
    (colNorm_nonneg _ _)


/-- The spectral bound `sK` of a nonempty chunk is nonnegative. -/
theorem step_sK_nonneg {K cl : ℕ} (hcl : 0 < cl) (k : ℕ → ℕ → ℝ) {sK : ℝ}
    (hsK : ∀ x : ℕ → ℝ, colNorm K (fun c => ∑ i ∈ Finset.range cl, k i c * x i) ≤
      sK * colNorm cl x) : 0 ≤ sK := by
  have h := hsK (fun i => if i = 0 then 1 else 0)
  have hn : colNorm cl (fun i => if i = 0 then (1 : ℝ) else 0) = 1 := by
    unfold colNorm
    have : ∀ i : ℕ, (if i = 0 then (1 : ℝ) else 0) ^ 2 = if i = 0 then 1 else 0 := by
      intro i; split_ifs <;> simp
    simp only [this, Finset.sum_ite_eq', Finset.mem_range, hcl, ite_true, Real.sqrt_one]
  rw [hn, mul_one] at h
  exact (colNorm_nonneg _ _).trans h

/-- **One step of the state.**
* `Y c` the state (`c < K`: finite, its stored `bfloat16` value `hh c` within the rounding error,
  `‖hh‖ ≤ 2^40`; `K ≤ c < 64·nb`: the padded rows, finite, at most `pb ≤ 2^100`);
* the decay: `gt` the computed `exp2(g_last)` (or `1`) within `errE` of `gam` (`0 < gam ≤ 2^8`),
  `S c` the IEEE product `Y c·gt`;
* the tokens: `kk i c` the loaded `k` (zero when `cl ≤ i` or `K ≤ c`), `vb i` the rounded gated
  value (`i < cl`: within `2^-8·Gm·vAbs + 2^-63` of `Gm·vref`, `Gm ≤ 2^8`, `Z ≤ 2^61`; every `vb`
  finite);
* the update `Yn` (`hupd`, which the walk proves from the dot fact for the dot-with-accumulator,
  or from the dot fact and the fusable addition: constants `ρ2 ≤ 1.6·10^-5`, `α2 ≤ 2^-110`; its
  room condition is discharged here).
Then every `Yn c` is finite and `(Yn c) - (gam·Y c + Σ_i kk i c·Gm i·vref i)`, `c < K`, has norm at
most `errStep`'s expression, with `sK` the spectral norm bound of `k`, `klv` its lifted Frobenius
norm and `GN = ‖Gm·vAbs‖`. -/
theorem numS {K nb cl : ℕ} (hK : 0 < K) (hKle : K ≤ 256) (hnb : K ≤ 64 * nb) (hnb' : 64 * nb < K + 64)
    (hcl : cl ≤ 64)
    (Y : ℕ → F32) (hh : ℕ → ℝ)
    (hY : ∀ c, c < K → (Y c).isFinite = true ∧
      |hh c - (Y c).toReal| ≤ 2 ^ (-8 : ℤ) * |(Y c).toReal| + 2 ^ (-134 : ℤ))
    (hhn : colNorm K hh ≤ 2 ^ 40) (pb : ℝ) (hpb : pb ≤ 2 ^ 100)
    (hpad : ∀ c, K ≤ c → c < 64 * nb → (Y c).isFinite = true ∧ |(Y c).toReal| ≤ pb)
    (gam errE gt : ℝ) (hgam : 0 < gam) (hgam8 : gam ≤ 2 ^ 8)
    (herrE : 0 ≤ errE ∧ errE ≤ 2 ^ (-22 : ℤ) * gam + mu) (hgt : |gt - gam| ≤ errE)
    (S : ℕ → F32)
    (hS : ∀ c, c < 64 * nb → (S c).isFinite = true ∧
      |(S c).toReal - gt * (Y c).toReal| ≤ u32 * |gt * (Y c).toReal| + e32)
    (Gm vref Z : ℕ → ℝ) (hGm : ∀ i, i < cl → 0 < Gm i ∧ Gm i ≤ 2 ^ 8)
    (hZ : ∀ i, i < cl → 0 ≤ Z i ∧ Z i ≤ 2 ^ 61) (hvr : ∀ i, i < cl → |vref i| ≤ Z i)
    (vb : ℕ → BF16) (hvbf : ∀ i, i < 64 → (vb i).isFinite = true)
    (hvb : ∀ i, i < cl → |(vb i).toReal - Gm i * vref i| ≤
      2 ^ (-8 : ℤ) * (Gm i * vAbsA (vref i) (Z i)) + 2 ^ (-63 : ℤ))
    (kk : ℕ → ℕ → BF16)
    (hk : ∀ i c, i < 64 → c < 64 * nb → (kk i c).isFinite = true ∧ |(kk i c).toReal| ≤ 2 ^ 16)
    (hkz : ∀ i c, i < 64 → c < 64 * nb → (cl ≤ i ∨ K ≤ c) → (kk i c).toReal = 0)
    (ρ2 α2 : ℝ) (hρ2 : 0 ≤ ρ2 ∧ ρ2 ≤ 16 / 10 ^ 6) (hα2 : 0 ≤ α2 ∧ α2 ≤ 2 ^ (-110 : ℤ))
    (Yn : ℕ → F32)
    (hupd : ∀ c, c < 64 * nb → (S c).isFinite = true →
      |(S c).toReal| + ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i) ≤ 2 ^ 120 →
      (Yn c).isFinite = true ∧
        |(Yn c).toReal - ((S c).toReal + ∑ i ∈ Finset.range 64, (kk i c).toReal * (vb i).toReal)| ≤
          ρ2 * (|(S c).toReal| + ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i)) + α2)
    (sK klv GN : ℝ)
    (hsK : ∀ x : ℕ → ℝ, colNorm K (fun c => ∑ i ∈ Finset.range cl, (kk i c).toReal * x i) ≤
      sK * colNorm cl x)
    (hsKle : sK ≤ klv)
    (hkl : klv = Real.sqrt (∑ i ∈ Finset.range cl, ∑ c ∈ Finset.range K, lmB (kk i c) ^ 2))
    (hGN : GN = colNorm cl (fun i => Gm i * vAbsA (vref i) (Z i))) :
    (∀ c, c < 64 * nb → (Yn c).isFinite = true) ∧
      colNorm K (fun c => (Yn c).toReal -
          (gam * (Y c).toReal + ∑ i ∈ Finset.range cl, (kk i c).toReal * Gm i * vref i)) ≤
        2 ^ (-8 : ℤ) * sK * GN + 2 ^ (-15 : ℤ) * (klv * GN + gam * colNorm K hh) + 2 ^ (-36 : ℤ) := by
  have hklv0 : 0 ≤ klv := by rw [hkl]; exact Real.sqrt_nonneg _
  have _ := hK
  have _ := hnb'
  have _ := hvbf
  have hsK' : ∀ x : ℕ → ℝ, colNorm K (fun c => ∑ i ∈ Finset.range cl, (kk i c).toReal * x i) ≤
      max sK 0 * colNorm cl x := fun x =>
    (hsK x).trans (mul_le_mul_of_nonneg_right (le_max_left _ _) (colNorm_nonneg _ _))
  have hmax : max sK 0 * GN = sK * GN := by
    rcases Nat.eq_zero_or_pos cl with h0 | h0
    · have : GN = 0 := by rw [hGN, h0]; unfold colNorm; simp
      rw [this]; simp
    · rw [max_eq_left (step_sK_nonneg h0 (fun i c => (kk i c).toReal) hsK)]
  have key := step_core hKle hnb hcl Y hh hY hhn pb hpb hpad gam errE gt hgam hgam8 herrE hgt S hS
    Gm vref Z hGm hZ hvr vb hvb kk hk hkz ρ2 α2 hρ2 hα2 Yn hupd (max sK 0) klv GN (le_max_right _ _)
    hsK' (max_le hsKle hklv0) hkl hGN
  refine ⟨key.1, key.2.trans (le_of_eq ?_)⟩
  rw [mul_assoc, hmax, ← mul_assoc]

end Claims.ChunkDeltaH
