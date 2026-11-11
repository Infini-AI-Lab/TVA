import Proof.NumericD

/-!
# `lora_shrink_bf16`: the final inequality

Pure arithmetic on rationals: the sum of the error terms (the atomic additions, the scaled parts'
accumulator errors and roundings, the rounding of `scaling`) is at most the claim's `tol`.
-/

namespace Claims.LoraShrink

set_option maxHeartbeats 4000000 in
/-- **The error terms add up to at most `tol`.** `E` is at most the atomic additions' error (a
multiple `A1` of `vMag` plus `A2`), plus `σ·m` plus the roundings of the products (`u·σ·PM + skη`),
plus the rounding of `scaling` (`(u·σ + η)·B`), where `m ≤ M1·B + g` is the parts' error, `M1 ≤ k1·T₀`,
`PM ≤ B + m` the parts' magnitude, `σ ≤ (1 + 2u)(s + η)` the rounded scaling. The numeric
hypotheses are those of the two launch configurations. -/
theorem master {u η A1 A2 skη σ s B m PM g M1 k1 a1 T0 W1 E : ℚ}
    (hu : 0 ≤ u) (hu' : u ≤ 1 / 16777216) (hη : 0 ≤ η) (hη' : η ≤ 2 ^ (-150 : ℤ))
    (hA1 : 0 ≤ A1) (hA1' : A1 ≤ a1) (hA2' : A2 ≤ 2 ^ (-117 : ℤ))
    (hsk : 0 ≤ skη) (hsk' : skη ≤ 2 ^ (-144 : ℤ)) (ha1 : a1 ≤ 1 / 100000)
    (hs : 0 ≤ s) (hB : 0 ≤ B) (hσ0 : 0 ≤ σ) (hPM0 : 0 ≤ PM) (hg0 : 0 ≤ g)
    (hM10 : 0 ≤ M1) (hg : g ≤ 2 ^ (-104 : ℤ)) (hT0 : 1 ≤ T0) (hT0' : T0 ≤ 512)
    (hσ : σ ≤ (1 + 2 * u) * (s + η)) (hm : m ≤ M1 * B + g) (hM1 : M1 ≤ k1 * T0)
    (hPM : PM ≤ B + m) (hk1 : 0 ≤ k1)
    (hE : E ≤ A1 * (σ * PM + u * σ * PM + skη) + A2 + (σ * m + u * σ * PM + skη) +
      (u * σ + η) * B)
    (hW1 : W1 * 512 ≤ 1 / 16)
    (hnum : (1 + 1 / 8388608) * (k1 * (1 + a1 * (1 + 1 / 16777216) + 1 / 16777216) +
      a1 * (1 + 1 / 16777216) + 2 / 16777216) ≤ W1) :
    E ≤ W1 * T0 * (s * B) + 2 ^ (-100 : ℤ) * (1 + s + B) := by
  have hY0 : 0 ≤ σ * B := mul_nonneg hσ0 hB
  have ha10 : 0 ≤ a1 := hA1.trans hA1'
  -- Z = σ·PM
  have hZ : σ * PM ≤ (1 + M1) * (σ * B) + g * σ := by
    have h1 : σ * PM ≤ σ * (B + m) := mul_le_mul_of_nonneg_left hPM hσ0
    have h2 : σ * m ≤ σ * (M1 * B + g) := mul_le_mul_of_nonneg_left hm hσ0
    nlinarith
  have hZm : σ * m ≤ M1 * (σ * B) + g * σ := by
    have h2 : σ * m ≤ σ * (M1 * B + g) := mul_le_mul_of_nonneg_left hm hσ0
    nlinarith
  have hZ0 : 0 ≤ σ * PM := mul_nonneg hσ0 hPM0
  obtain ⟨cZ, hcZ⟩ : ∃ x : ℚ, x = A1 * (1 + u) + u := ⟨_, rfl⟩
  have hcZ0 : 0 ≤ cZ := by rw [hcZ]; positivity
  have hE2 : E ≤ cZ * (σ * PM) + (A1 * skη + A2 + skη) + σ * m + u * (σ * B) + η * B := by
    have : u * σ * PM = u * (σ * PM) := by ring
    have e : cZ * (σ * PM) = A1 * (σ * PM) + A1 * u * (σ * PM) + u * (σ * PM) := by rw [hcZ]; ring
    have e2 : A1 * (σ * PM + u * σ * PM + skη) = A1 * (σ * PM) + A1 * u * (σ * PM) + A1 * skη := by ring
    rw [e2] at hE
    linarith
  have hE3 : cZ * (σ * PM) ≤ cZ * ((1 + M1) * (σ * B) + g * σ) :=
    mul_le_mul_of_nonneg_left hZ hcZ0
  obtain ⟨C1, hC1⟩ : ∃ x : ℚ, x = M1 * (1 + A1 * (1 + u) + u) + A1 * (1 + u) + 2 * u := ⟨_, rfl⟩
  have hE4 : E ≤ C1 * (σ * B) + (cZ + 1) * (g * σ) + (A1 * skη + A2 + skη) + η * B := by
    have : C1 * (σ * B) = cZ * ((1 + M1) * (σ * B)) + M1 * (σ * B) + u * (σ * B) := by
      rw [hC1, hcZ]; ring
    have e2 : cZ * ((1 + M1) * (σ * B) + g * σ) = cZ * ((1 + M1) * (σ * B)) + cZ * (g * σ) := by ring
    have e3 : (cZ + 1) * (g * σ) = cZ * (g * σ) + g * σ := by ring
    linarith
  -- bound C1
  have hC1' : (1 + 1 / 8388608) * C1 ≤ W1 * T0 := by
    have hq : 1 + A1 * (1 + u) + u ≤ 1 + a1 * (1 + 1 / 16777216) + 1 / 16777216 := by
      have : A1 * (1 + u) ≤ a1 * (1 + 1 / 16777216) :=
        mul_le_mul hA1' (by linarith) (by linarith) (by linarith [hA1, hA1'])
      linarith
    have hq0 : 0 ≤ 1 + A1 * (1 + u) + u := by positivity
    have hpa0 : (0 : ℚ) ≤ a1 * (1 + 1 / 16777216) + 2 / 16777216 := by nlinarith [hA1, hA1']
    have hp : A1 * (1 + u) + 2 * u ≤ a1 * (1 + 1 / 16777216) + 2 / 16777216 := by
      have : A1 * (1 + u) ≤ a1 * (1 + 1 / 16777216) :=
        mul_le_mul hA1' (by linarith) (by linarith) (by linarith [hA1, hA1'])
      linarith
    have hMq : M1 * (1 + A1 * (1 + u) + u) ≤
        (k1 * T0) * (1 + a1 * (1 + 1 / 16777216) + 1 / 16777216) :=
      mul_le_mul hM1 hq hq0 (by positivity)
    have hC : C1 ≤ (k1 * (1 + a1 * (1 + 1 / 16777216) + 1 / 16777216) +
        a1 * (1 + 1 / 16777216) + 2 / 16777216) * T0 := by
      have h1 : A1 * (1 + u) + 2 * u ≤ (a1 * (1 + 1 / 16777216) + 2 / 16777216) * T0 :=
        hp.trans (le_mul_of_one_le_right hpa0 hT0)
      have h2 : C1 = M1 * (1 + A1 * (1 + u) + u) + (A1 * (1 + u) + 2 * u) := by rw [hC1]; ring
      rw [h2]
      calc M1 * (1 + A1 * (1 + u) + u) + (A1 * (1 + u) + 2 * u)
          ≤ (k1 * T0) * (1 + a1 * (1 + 1 / 16777216) + 1 / 16777216) +
            (a1 * (1 + 1 / 16777216) + 2 / 16777216) * T0 := add_le_add hMq h1
        _ = _ := by ring
    have hnum' := mul_le_mul_of_nonneg_right hnum (by linarith : (0 : ℚ) ≤ T0)
    calc (1 + 1 / 8388608) * C1
        ≤ (1 + 1 / 8388608) * ((k1 * (1 + a1 * (1 + 1 / 16777216) + 1 / 16777216) +
          a1 * (1 + 1 / 16777216) + 2 / 16777216) * T0) :=
          mul_le_mul_of_nonneg_left hC (by norm_num)
      _ = ((1 + 1 / 8388608) * (k1 * (1 + a1 * (1 + 1 / 16777216) + 1 / 16777216) +
          a1 * (1 + 1 / 16777216) + 2 / 16777216)) * T0 := by ring
      _ ≤ W1 * T0 := hnum'
  have hC10 : 0 ≤ C1 := by rw [hC1]; positivity
  -- σ·B ≤ (1 + 2u)(s + η)·B
  have hσB : σ * B ≤ (1 + 1 / 8388608) * (s * B + η * B) := by
    have : σ ≤ (1 + 1 / 8388608) * (s + η) := by
      refine hσ.trans (mul_le_mul_of_nonneg_right ?_ (by linarith))
      linarith
    nlinarith
  have hgσ : g * σ ≤ g * ((1 + 1 / 8388608) * (s + η)) := by
    refine mul_le_mul_of_nonneg_left (hσ.trans (mul_le_mul_of_nonneg_right ?_ (by linarith))) hg0
    linarith
  have hT : C1 * (σ * B) ≤ W1 * T0 * (s * B) + W1 * T0 * (η * B) := by
    have h1 : C1 * (σ * B) ≤ C1 * ((1 + 1 / 8388608) * (s * B + η * B)) :=
      mul_le_mul_of_nonneg_left hσB hC10
    have h2 : C1 * ((1 + 1 / 8388608) * (s * B + η * B)) =
        ((1 + 1 / 8388608) * C1) * (s * B + η * B) := by ring
    have h3 : ((1 + 1 / 8388608) * C1) * (s * B + η * B) ≤ (W1 * T0) * (s * B + η * B) :=
      mul_le_mul_of_nonneg_right hC1' (by positivity)
    have h4 : (W1 * T0) * (s * B + η * B) = W1 * T0 * (s * B) + W1 * T0 * (η * B) := by ring
    linarith
  have hW : W1 * T0 ≤ 1 / 16 := by
    have hW1' : W1 ≤ 1 / 16 / 512 := by linarith
    have : 0 ≤ W1 := by
      have : 0 ≤ (1 + 1 / 8388608) * (k1 * (1 + a1 * (1 + 1 / 16777216) + 1 / 16777216) +
        a1 * (1 + 1 / 16777216) + 2 / 16777216) := by
        have : (0 : ℚ) ≤ a1 * (1 + 1 / 16777216) + 2 / 16777216 := by nlinarith [hA1, hA1']
        positivity
      linarith
    have := mul_le_mul_of_nonneg_left hT0' this
    linarith
  have hcZ' : cZ + 1 ≤ 2 := by
    have : A1 * (1 + u) ≤ a1 * (1 + 1 / 16777216) :=
      mul_le_mul hA1' (by linarith) (by linarith) (by linarith [hA1, hA1'])
    rw [hcZ]
    have : a1 * (1 + 1 / 16777216) ≤ 1 / 99999 := by nlinarith
    linarith
  have hgs : (cZ + 1) * (g * σ) ≤ 2 * (g * ((1 + 1 / 8388608) * (s + η))) := by
    have h0 : 0 ≤ g * σ := mul_nonneg hg0 hσ0
    have := mul_le_mul_of_nonneg_right hcZ' h0
    linarith
  have hg2 : g * ((1 + 1 / 8388608) * (s + η)) ≤ 2 ^ (-104 : ℤ) * (2 * (s + η)) := by
    have : (1 + 1 / 8388608 : ℚ) * (s + η) ≤ 2 * (s + η) := by nlinarith
    exact mul_le_mul hg this (by positivity) (by positivity)
  have hK : A1 * skη ≤ 2 ^ (-144 : ℤ) := by
    have : A1 ≤ 1 := by linarith
    nlinarith
  have e1 : (2 : ℚ) ^ (-117 : ℤ) = 1 / 2 ^ 117 := by norm_num
  have e2 : (2 : ℚ) ^ (-144 : ℤ) = 1 / 2 ^ 144 := by norm_num
  have e3 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have e4 : (2 : ℚ) ^ (-104 : ℤ) = 1 / 2 ^ 104 := by norm_num
  have e5 : (2 : ℚ) ^ (-100 : ℤ) = 1 / 2 ^ 100 := by norm_num
  rw [e1] at hA2'
  rw [e2] at hsk' hK
  rw [e3] at hη'
  rw [e4] at hg2
  rw [e5]
  have hηB : η * B ≤ 1 / 2 ^ 150 * B := mul_le_mul_of_nonneg_right hη' hB
  have hηB2 : W1 * T0 * (η * B) ≤ 1 / 2 ^ 150 * B := by
    have : W1 * T0 * (η * B) ≤ 1 * (η * B) := by
      refine mul_le_mul_of_nonneg_right ?_ (by positivity)
      linarith
    linarith
  have hs2 : 2 ^ 0 * (1 / (2 : ℚ) ^ 104) * (2 * (s + η)) ≤ 1 / 2 ^ 101 * (s + 1) := by
    have : η ≤ 1 := by
      have : (1 : ℚ) / 2 ^ 150 ≤ 1 := by norm_num
      linarith
    have h4 : (1 / (2 : ℚ) ^ 104) * (2 * (s + η)) ≤ (1 / (2 : ℚ) ^ 104) * (2 * (s + 1)) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    nlinarith
  nlinarith [hs2, hηB, hηB2, hT, hgs, hg2, hE4]

end Claims.LoraShrink
