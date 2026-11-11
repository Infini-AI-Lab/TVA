import Claim
import TVA.Proof
import Proof.NumDefs

/-!
# Numerics 2: the gated terms and the epilogue's chain

Pure facts about reals: how the errors of the loop's dots, the gates, the scores' rounding, the
last dot, the scale, the sum and the store add up to the claim's tolerance.
-/

namespace Claims.ChunkFwdO

open TVA

/-! ## The gates -/

/-- Generic error propagation for a gated product: `ob` near `Q`, `Gh` near `G`, `oh` a rounding. -/
theorem gate_core {ob Q P G Gh e s oh : ℝ} (hP : 0 ≤ P) (hQ : |Q| ≤ P) (hG : 0 ≤ G)
    (he : 0 ≤ e) (hs : 0 ≤ s)
    (hob : |ob - Q| ≤ 33 / 500000 * P + 2 ^ (-116 : ℤ))
    (hGh : |Gh - G| ≤ e * G + s)
    (hoh : |oh - ob * Gh| ≤ 2 ^ (-24 : ℤ) * |ob * Gh| + 2 ^ (-150 : ℤ)) :
    |ob * Gh| ≤ ((1 + 33 / 500000) * P + 2 ^ (-116 : ℤ)) * (G * (1 + e) + s) ∧
    |oh - G * Q| ≤ 2 ^ (-24 : ℤ) * (((1 + 33 / 500000) * P + 2 ^ (-116 : ℤ)) * (G * (1 + e) + s))
      + 2 ^ (-150 : ℤ) + (33 / 500000 * P + 2 ^ (-116 : ℤ)) * (G * (1 + e) + s)
      + P * (e * G + s) := by
  have h116 : (0 : ℝ) ≤ 2 ^ (-116 : ℤ) := by positivity
  have hB : |ob| ≤ (1 + 33 / 500000) * P + 2 ^ (-116 : ℤ) := by
    have h1 : |ob| ≤ |ob - Q| + |Q| := by
      have := abs_add_le (ob - Q) Q
      simpa using this
    linarith
  have hH : |Gh| ≤ G * (1 + e) + s := by
    have h1 : |Gh| ≤ |Gh - G| + |G| := by
      have := abs_add_le (Gh - G) G
      simpa using this
    rw [abs_of_nonneg hG] at h1
    linarith
  have hB0 : 0 ≤ (1 + 33 / 500000) * P + 2 ^ (-116 : ℤ) := by positivity
  have hprod : |ob * Gh| ≤ ((1 + 33 / 500000) * P + 2 ^ (-116 : ℤ)) * (G * (1 + e) + s) := by
    rw [abs_mul]
    exact mul_le_mul hB hH (abs_nonneg _) hB0
  refine ⟨hprod, ?_⟩
  have hH0 : 0 ≤ G * (1 + e) + s := by positivity
  have t2 : |(ob - Q) * Gh| ≤ (33 / 500000 * P + 2 ^ (-116 : ℤ)) * (G * (1 + e) + s) := by
    rw [abs_mul]
    exact mul_le_mul hob hH (abs_nonneg _) (by positivity)
  have t3 : |Q * (Gh - G)| ≤ P * (e * G + s) := by
    rw [abs_mul]
    exact mul_le_mul hQ hGh (abs_nonneg _) hP
  have hsplit : oh - G * Q = (oh - ob * Gh) + (ob - Q) * Gh + Q * (Gh - G) := by ring
  have h24 : (0 : ℝ) ≤ 2 ^ (-24 : ℤ) := by positivity
  have t1 : |oh - ob * Gh| ≤ 2 ^ (-24 : ℤ) * (((1 + 33 / 500000) * P + 2 ^ (-116 : ℤ)) * (G * (1 + e) + s))
      + 2 ^ (-150 : ℤ) := by
    exact hoh.trans (add_le_add_left (mul_le_mul_of_nonneg_left hprod h24) _)
  rw [hsplit]
  calc _ ≤ |oh - ob * Gh| + |(ob - Q) * Gh| + |Q * (Gh - G)| := abs_add_three _ _ _
    _ ≤ _ := by linarith

/-- **The inter-chunk term times its gate**: `ob` within the loop's bound of `Q`
(`|Q| ≤ P ≤ 2^100`), `Gh` within `(2^-21 + Γ/2)·G + 2^-126` of `G` (`G ≤ 2^64`, `G·P ≤ 2^100`):
the product `ob·Gh` is below `2^102` (room), and any rounding `oh` of it is within
`(cO + 0.6·Γ)·G·P + fl·(1 + G + P)` of `G·Q`. -/
theorem gate_o_num {ob Q P G Gh Γ : ℝ} (hP : 0 ≤ P) (hQ : |Q| ≤ P) (hG : 0 ≤ G)
    (hG64 : G ≤ 2 ^ 64) (hGP : G * P ≤ 2 ^ 100) (hP100 : P ≤ 2 ^ 100) (hΓ : 0 ≤ Γ)
    (hΓ' : Γ ≤ 2 ^ (-9 : ℤ))
    (hob : |ob - Q| ≤ 33 / 500000 * P + 2 ^ (-116 : ℤ))
    (hGh : |Gh - G| ≤ (2 ^ (-21 : ℤ) + Γ / 2) * G + 2 ^ (-126 : ℤ)) :
    |ob * Gh| ≤ 2 ^ 102 ∧ ∀ oh : ℝ, |oh - ob * Gh| ≤ 2 ^ (-24 : ℤ) * |ob * Gh| + 2 ^ (-150 : ℤ) →
      |oh - G * Q| ≤ (cO + 6 / 10 * Γ) * G * P + fl * (1 + G + P) := by
  have hΓ9 : Γ ≤ 1 / 2 ^ 9 := by
    have : (2:ℝ) ^ (-9 : ℤ) = 1 / 2 ^ 9 := by norm_num
    exact hΓ'.trans_eq this
  have hfl : fl = 1 / 2 ^ 112 := by unfold fl; norm_num
  have h116 : (2:ℝ) ^ (-116 : ℤ) = 1 / 2 ^ 116 := by norm_num
  have h150 : (2:ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have h24 : (2:ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have hek : (2:ℝ) ^ (-21 : ℤ) = 1 / 2 ^ 21 := by norm_num
  have hsk : (2:ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have hΓG : 0 ≤ Γ * G := mul_nonneg hΓ hG
  have hGP' : 0 ≤ G * P := mul_nonneg hG hP
  have hΓGP : 0 ≤ Γ * G * P := mul_nonneg hΓG hP
  have hΓG9 : Γ * G ≤ 1 / 2 ^ 9 * G := mul_le_mul_of_nonneg_right hΓ9 hG
  have hΓGP9 : Γ * G * P ≤ 1 / 2 ^ 9 * (G * P) := by nlinarith
  have hPs : P * (1 / 2 ^ 126) ≤ 2 ^ 100 * (1 / 2 ^ 126) := by nlinarith
  have hGd : G * (1 / 2 ^ 116) ≤ 2 ^ 64 * (1 / 2 ^ 116) := by nlinarith
  have hcore := fun oh hoh => gate_core (e := 2 ^ (-21 : ℤ) + Γ / 2) (s := 2 ^ (-126 : ℤ)) (oh := oh)
    hP hQ hG (by positivity) (by positivity) hob hGh hoh
  refine ⟨?_, fun oh hoh => ?_⟩
  · have := (hcore (ob * Gh) (by rw [sub_self, abs_zero]; positivity)).1
    rw [h116, hek, hsk] at this
    nlinarith
  · have := (hcore oh hoh).2
    rw [h116, h24, h150, hek, hsk] at this
    rw [hfl]
    unfold cO
    nlinarith

/-- **A score times its weight**: `ab` within the loop's bound of `A` (`|A| ≤ PA ≤ 2^100`), `wh`
within `(2^-16 + 0.72·Γ)·w + 2^-123` of `w` (`w ≤ 2^64`, `w·PA ≤ 2^100`): the product `ab·wh` is
below `2^102`, and any rounding `ah` of it is within `(cA + 0.8·Γ)·w·PA + fl·(1 + w + PA)` of
`A·w`. -/
theorem gate_a_num {ab A PA w wh Γ : ℝ} (hPA : 0 ≤ PA) (hA : |A| ≤ PA) (hw : 0 ≤ w)
    (hw64 : w ≤ 2 ^ 64) (hwPA : w * PA ≤ 2 ^ 100) (hPA100 : PA ≤ 2 ^ 100) (hΓ : 0 ≤ Γ)
    (hΓ' : Γ ≤ 2 ^ (-9 : ℤ))
    (hab : |ab - A| ≤ 33 / 500000 * PA + 2 ^ (-116 : ℤ))
    (hwh : |wh - w| ≤ (2 ^ (-16 : ℤ) + 72 / 100 * Γ) * w + 2 ^ (-123 : ℤ)) :
    |ab * wh| ≤ 2 ^ 102 ∧ ∀ ah : ℝ, |ah - ab * wh| ≤ 2 ^ (-24 : ℤ) * |ab * wh| + 2 ^ (-150 : ℤ) →
      |ah - A * w| ≤ (cA + 8 / 10 * Γ) * w * PA + fl * (1 + w + PA) := by
  have hΓ9 : Γ ≤ 1 / 2 ^ 9 := by
    have : (2:ℝ) ^ (-9 : ℤ) = 1 / 2 ^ 9 := by norm_num
    exact hΓ'.trans_eq this
  have hfl : fl = 1 / 2 ^ 112 := by unfold fl; norm_num
  have h116 : (2:ℝ) ^ (-116 : ℤ) = 1 / 2 ^ 116 := by norm_num
  have h150 : (2:ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have h24 : (2:ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have hek : (2:ℝ) ^ (-16 : ℤ) = 1 / 2 ^ 16 := by norm_num
  have hsk : (2:ℝ) ^ (-123 : ℤ) = 1 / 2 ^ 123 := by norm_num
  have hΓG : 0 ≤ Γ * w := mul_nonneg hΓ hw
  have hGP' : 0 ≤ w * PA := mul_nonneg hw hPA
  have hΓGP : 0 ≤ Γ * w * PA := mul_nonneg hΓG hPA
  have hΓG9 : Γ * w ≤ 1 / 2 ^ 9 * w := mul_le_mul_of_nonneg_right hΓ9 hw
  have hΓGP9 : Γ * w * PA ≤ 1 / 2 ^ 9 * (w * PA) := by nlinarith
  have hPs : PA * (1 / 2 ^ 123) ≤ 2 ^ 100 * (1 / 2 ^ 123) := by nlinarith
  have hGd : w * (1 / 2 ^ 116) ≤ 2 ^ 64 * (1 / 2 ^ 116) := by nlinarith
  have hcore := fun oh hoh => gate_core (e := 2 ^ (-16 : ℤ) + 72 / 100 * Γ) (s := 2 ^ (-123 : ℤ)) (oh := oh)
    hPA hA hw (by positivity) (by positivity) hab hwh hoh
  refine ⟨?_, fun oh hoh => ?_⟩
  · have := (hcore (ab * wh) (by rw [sub_self, abs_zero]; positivity)).1
    rw [h116, hek, hsk] at this
    nlinarith
  · have := (hcore oh hoh).2
    rw [h116, h24, h150, hek, hsk] at this
    rw [hfl, mul_comm A w]
    unfold cA
    nlinarith

/-- Without a gate (`G = 1`) the loop's bound already gives the gated one. -/
theorem gate_o_none {ob Q P Γ : ℝ} (hP : 0 ≤ P) (hΓ : 0 ≤ Γ)
    (hob : |ob - Q| ≤ 33 / 500000 * P + 2 ^ (-116 : ℤ)) :
    |ob - 1 * Q| ≤ (cO + 6 / 10 * Γ) * 1 * P + fl * (1 + 1 + P) := by
  have hfl : fl = 1 / 2 ^ 112 := by unfold fl; norm_num
  have h116 : (2:ℝ) ^ (-116 : ℤ) = 1 / 2 ^ 116 := by norm_num
  rw [h116] at hob
  rw [hfl, one_mul Q]
  unfold cO
  have : 0 ≤ Γ * P := mul_nonneg hΓ hP
  nlinarith

/-- Without a gate (`w = 1`) the loop's bound already gives the gated one. -/
theorem gate_a_none {ab A PA Γ : ℝ} (hPA : 0 ≤ PA) (hΓ : 0 ≤ Γ)
    (hab : |ab - A| ≤ 33 / 500000 * PA + 2 ^ (-116 : ℤ)) :
    |ab - A * 1| ≤ (cA + 8 / 10 * Γ) * 1 * PA + fl * (1 + 1 + PA) := by
  have hfl : fl = 1 / 2 ^ 112 := by unfold fl; norm_num
  have h116 : (2:ℝ) ^ (-116 : ℤ) = 1 / 2 ^ 116 := by norm_num
  rw [h116] at hab
  rw [hfl, mul_one A]
  unfold cA
  have : 0 ≤ Γ * PA := mul_nonneg hΓ hPA
  nlinarith

end Claims.ChunkFwdO
