import Claim
import TVA.Proof
import Proof.NumDefs

/-!
# Numerics 3: the epilogue's chain

How the errors of the gated terms (`ot`, `At`), the scores' rounding to `bfloat16` (`a`), the last
dot (`D`), the scale (`σh`), the two products (`m1`, `m2`), the sum (`x`) and the store's rounding
(`c`) add up to the claim's tolerance. Indices `i < n` are the keys `s ≤ t` of the chunk.
-/

namespace Claims.ChunkFwdO

open TVA

/-- `Mag` over `n` keys: `G·PQS + Σ w·PA·Lv`. -/
noncomputable def epMag (n : ℕ) (G PQS : ℝ) (w PA Lv : ℕ → ℝ) : ℝ :=
  G * PQS + ∑ i ∈ Finset.range n, w i * PA i * Lv i

/-- `Flo` over `n` keys: `1 + G + PQS + Σ (1 + w + PA)·Lv`. -/
noncomputable def epFlo (n : ℕ) (G PQS : ℝ) (w PA Lv : ℕ → ℝ) : ℝ :=
  1 + G + PQS + ∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i

/-- `Mabs` over `n` keys: `Σ |A|·w·|v|`. -/
noncomputable def epMabs (n : ℕ) (A w v : ℕ → ℝ) : ℝ :=
  ∑ i ∈ Finset.range n, |A i| * w i * |v i|

/-- The hypotheses the epilogue's chain starts from: the scale `σh` of the kernel is `σ` rounded;
`G` the gate, `QS` the inter-chunk dot (`PQS` its magnitude), `A i`, `w i`, `PA i`, `v i`, `Lv i`
the score, its weight and magnitude, the value and its lifted magnitude of key `i < n`; `ot` the
gated inter-chunk term, `At i` the gated score, `a i` its `bfloat16` rounding; and the room
(`InRange`). -/
structure EpIn (n : ℕ) (σ σh G Γ QS PQS ot : ℝ) (A w PA v Lv At a : ℕ → ℝ) : Prop where
  hσ : (2 : ℝ) ^ (-64 : ℤ) ≤ σ ∧ σ ≤ 2 ^ 64
  hσh : |σh - σ| ≤ 2 ^ (-24 : ℤ) * σ + 2 ^ (-150 : ℤ)
  hG : 0 ≤ G
  hG64 : G ≤ 2 ^ 64
  hΓ : 0 ≤ Γ
  hΓ' : Γ ≤ 2 ^ (-9 : ℤ)
  hPQS : 0 ≤ PQS
  hQS : |QS| ≤ PQS
  hw : ∀ i < n, 0 ≤ w i
  hw64 : ∀ i < n, w i ≤ 2 ^ 64
  hPA : ∀ i < n, 0 ≤ PA i
  hAPA : ∀ i < n, |A i| ≤ PA i
  hLv : ∀ i < n, 0 ≤ Lv i
  hv : ∀ i < n, |v i| ≤ Lv i
  hGPQS : G * PQS ≤ 2 ^ 100
  hPQS100 : PQS ≤ 2 ^ 100
  hwPA : ∀ i < n, w i * PA i ≤ 2 ^ 100
  hPA100 : ∀ i < n, PA i ≤ 2 ^ 100
  hroom : epMag n G PQS w PA Lv + epFlo n G PQS w PA Lv ≤ 2 ^ 100
  hroomσ : σ * (epMag n G PQS w PA Lv + epFlo n G PQS w PA Lv) ≤ 2 ^ 100
  hot : |ot - G * QS| ≤ (cO + 6 / 10 * Γ) * G * PQS + fl * (1 + G + PQS)
  hAt : ∀ i < n, |At i - A i * w i| ≤ (cA + 8 / 10 * Γ) * w i * PA i + fl * (1 + w i + PA i)
  ha : ∀ i < n, |a i - At i| ≤ 2 ^ (-8 : ℤ) * |At i| + 2 ^ (-134 : ℤ)

/-- `cO` as a number. -/
theorem ep_cO : cO = 7 / 100000 := rfl

/-- `cA` as a number. -/
theorem ep_cA : cA = 9 / 100000 := rfl

/-- `fl` as a number. -/
theorem ep_fl : fl = 1 / 2 ^ 112 := by unfold fl; norm_num

/-- The pure arithmetic of the epilogue's chain, with the sums as atoms. -/
theorem ep_alg {σ σh Γ P1 P2 Fl1 Fl2 Mabs GQS S ot D m1 m2 x c : ℝ}
    (hσ : 0 < σ) (hσh : |σh - σ| ≤ 1 / 10 ^ 7 * σ)
    (hΓ : 0 ≤ Γ)
    (hP1 : 0 ≤ P1) (hP2 : 0 ≤ P2) (hFl1 : 1 ≤ Fl1) (hFl2 : 0 ≤ Fl2)
    (hMabs : 0 ≤ Mabs) (hMP : Mabs ≤ P2)
    (hGQS : |GQS| ≤ P1) (hS : |S| ≤ Mabs)
    (hot : |ot - GQS| ≤ (7 / 100000 + 6 / 10 * Γ) * P1 + 1 / 2 ^ 112 * Fl1)
    (hota : |ot| ≤ 1002 / 1000 * P1 + 1 / 10 ^ 32 * Fl1)
    (hDS : |D - S| ≤ 1 / 256 * Mabs + 257 / 256 * (9 / 100000 + 8 / 10 * Γ) * P2
        + (257 / 256 / 2 ^ 112 + 1 / 2 ^ 134) * Fl2
        + 66 / 2 ^ 22 * (101 / 100 * P2 + 1 / 10 ^ 32 * Fl2) + 66 / 2 ^ 126)
    (hDa : |D| ≤ 102 / 100 * P2 + 2 / 10 ^ 32 * (1 + Fl2))
    (hm1 : |m1 - ot * σh| ≤ 1 / 2 ^ 24 * |ot * σh| + 1 / 2 ^ 150)
    (hm2 : |m2 - D * σh| ≤ 1 / 2 ^ 24 * |D * σh| + 1 / 2 ^ 150)
    (hx : |x - (m1 + m2)| ≤ 3 / 2 ^ 24 * (|m1| + |m2|) + 4 / 2 ^ 150)
    (hc : |c - x| ≤ 1 / 2 ^ 8 * |x| + 1 / 2 ^ 134) :
    |c - σ * (GQS + S)| ≤ 1 / 2 ^ 8 * |σ * (GQS + S)| +
      (1 + 1 / 2 ^ 8) * σ * (1 / 2 ^ 8 * Mabs + (1 / 2 ^ 13 + Γ) * (P1 + P2) +
        1 / 2 ^ 100 * (Fl1 + Fl2)) + 1 / 2 ^ 130 := by
  have hσ0 := hσ.le
  have hb : |σh| ≤ (1 + 1 / 10 ^ 7) * σ := by
    have := abs_le.mp hσh
    rw [abs_le]
    constructor <;> linarith [this.1, this.2]
  have hσP1 : 0 ≤ σ * P1 := mul_nonneg hσ0 hP1
  have hσP2 : 0 ≤ σ * P2 := mul_nonneg hσ0 hP2
  have hσF2 : 0 ≤ σ * Fl2 := mul_nonneg hσ0 hFl2
  have hσF1 : σ ≤ σ * Fl1 := le_mul_of_one_le_right hσ0 hFl1
  have hσΓP1 : 0 ≤ σ * Γ * P1 := mul_nonneg (mul_nonneg hσ0 hΓ) hP1
  have hσΓP2 : 0 ≤ σ * Γ * P2 := mul_nonneg (mul_nonneg hσ0 hΓ) hP2
  have hσM : σ * Mabs ≤ σ * P2 := mul_le_mul_of_nonneg_left hMP hσ0
  have hσMn : 0 ≤ σ * Mabs := mul_nonneg hσ0 hMabs
  have hO : |ot * σh| ≤ 1003 / 1000 * (σ * P1) + 1 / 10 ^ 31 * (σ * Fl1) := by
    rw [abs_mul]
    have := mul_le_mul hota hb (abs_nonneg _)
      (by linarith : (0 : ℝ) ≤ 1002 / 1000 * P1 + 1 / 10 ^ 32 * Fl1)
    linarith
  have hV : |D * σh| ≤ 1021 / 1000 * (σ * P2) + 3 / 10 ^ 32 * (σ + σ * Fl2) := by
    rw [abs_mul]
    have := mul_le_mul hDa hb (abs_nonneg _)
      (by linarith : (0 : ℝ) ≤ 102 / 100 * P2 + 2 / 10 ^ 32 * (1 + Fl2))
    linarith
  have hm1' : |m1| ≤ (1 + 1 / 2 ^ 24) * |ot * σh| + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub m1 (ot * σh)
    linarith
  have hm2' : |m2| ≤ (1 + 1 / 2 ^ 24) * |D * σh| + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub m2 (D * σh)
    linarith
  have hxr : |x - (ot * σh + D * σh)| ≤ 25 / 10 ^ 8 * (|ot * σh| + |D * σh|) + 1 / 2 ^ 147 := by
    have e : x - (ot * σh + D * σh) =
        (x - (m1 + m2)) + (m1 - ot * σh) + (m2 - D * σh) := by ring
    rw [e]
    have := abs_add_three (x - (m1 + m2)) (m1 - ot * σh) (m2 - D * σh)
    have n1 := abs_nonneg (ot * σh)
    have n2 := abs_nonneg (D * σh)
    linarith
  have hy : |ot * σh + D * σh - σ * (GQS + S)| ≤
      |σh| * |ot - GQS| + |σh| * |D - S| + |σh - σ| * |GQS + S| := by
    have e : ot * σh + D * σh - σ * (GQS + S) =
        σh * (ot - GQS) + σh * (D - S) + (σh - σ) * (GQS + S) := by ring
    rw [e]
    calc _ ≤ |σh * (ot - GQS)| + |σh * (D - S)| + |(σh - σ) * (GQS + S)| := abs_add_three _ _ _
      _ = _ := by rw [abs_mul, abs_mul, abs_mul]
  have y1 := mul_le_mul hb hot (abs_nonneg _) (mul_nonneg (by norm_num) hσ0)
  have y2 := mul_le_mul hb hDS (abs_nonneg _) (mul_nonneg (by norm_num) hσ0)
  have hGS : |GQS + S| ≤ P1 + P2 := by
    have := abs_add_le GQS S
    linarith
  have y3 := mul_le_mul hσh hGS (abs_nonneg _) (mul_nonneg (by norm_num) hσ0)
  have hxfin : |x - σ * (GQS + S)| ≤
      σ * (1 / 2 ^ 8 * Mabs + (1 / 2 ^ 13 + Γ) * (P1 + P2) + 1 / 2 ^ 100 * (Fl1 + Fl2))
        + 1 / 2 ^ 140 := by
    have t := abs_sub_le x (ot * σh + D * σh) (σ * (GQS + S))
    linarith
  have t1 := abs_sub_le c x (σ * (GQS + S))
  have t2 := abs_sub_abs_le_abs_sub x (σ * (GQS + S))
  linarith

section
variable {n : ℕ} {σ σh G Γ QS PQS ot : ℝ} {A w PA v Lv At a : ℕ → ℝ}

/-- The scale is positive. -/
theorem ep_σpos (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) : 0 < σ :=
  lt_of_lt_of_le (by positivity) h.hσ.1

/-- `Γ ≤ 1/512`. -/
theorem ep_G9 (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) : Γ ≤ 1 / 512 := by
  have e : (2 : ℝ) ^ (-9 : ℤ) = 1 / 512 := by norm_num
  have := h.hΓ'
  rw [e] at this
  exact this

/-- The rounded scale is within `10^-7` (relative) of the scale. -/
theorem ep_sh_dev (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    |σh - σ| ≤ 1 / 10 ^ 7 * σ := by
  have h1 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have h2 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have h3 : (2 : ℝ) ^ (-64 : ℤ) = 1 / 2 ^ 64 := by norm_num
  have h4 := h.hσ.1
  have h5 := h.hσh
  rw [h1, h2] at h5
  rw [h3] at h4
  linarith

/-- The rounded scale is below `(1 + 10^-7)·σ`. -/
theorem ep_hb (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    |σh| ≤ (1 + 1 / 10 ^ 7) * σ := by
  have h1 := abs_le.mp (ep_sh_dev h)
  have h2 := ep_σpos h
  rw [abs_le]
  constructor <;> linarith [h1.1, h1.2]

/-- The sum `Σ w·PA·Lv` is nonnegative. -/
theorem ep_P2_nn (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    0 ≤ ∑ i ∈ Finset.range n, w i * PA i * Lv i :=
  Finset.sum_nonneg fun i hi => by
    have hi' := Finset.mem_range.mp hi
    exact mul_nonneg (mul_nonneg (h.hw i hi') (h.hPA i hi')) (h.hLv i hi')

/-- The sum `Σ (1+w+PA)·Lv` is nonnegative. -/
theorem ep_F2_nn (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    0 ≤ ∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i :=
  Finset.sum_nonneg fun i hi => by
    have hi' := Finset.mem_range.mp hi
    have := h.hw i hi'
    have := h.hPA i hi'
    exact mul_nonneg (by linarith) (h.hLv i hi')

/-- `Mabs` is nonnegative. -/
theorem ep_Mabs_nn (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    0 ≤ epMabs n A w v :=
  Finset.sum_nonneg fun i hi => by
    have hi' := Finset.mem_range.mp hi
    exact mul_nonneg (mul_nonneg (abs_nonneg _) (h.hw i hi')) (abs_nonneg _)

/-- `Mabs ≤ Σ w·PA·Lv`. -/
theorem ep_Mabs_le (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    epMabs n A w v ≤ ∑ i ∈ Finset.range n, w i * PA i * Lv i := by
  unfold epMabs
  apply Finset.sum_le_sum
  intro i hi
  have hi' := Finset.mem_range.mp hi
  have h1 : |A i| * w i ≤ PA i * w i :=
    mul_le_mul_of_nonneg_right (h.hAPA i hi') (h.hw i hi')
  have h2 : |A i| * w i * |v i| ≤ PA i * w i * Lv i :=
    mul_le_mul h1 (h.hv i hi') (abs_nonneg _) (mul_nonneg (h.hPA i hi') (h.hw i hi'))
  linarith

/-- `|Σ A·w·v| ≤ Mabs`. -/
theorem ep_S_le (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    |∑ i ∈ Finset.range n, A i * w i * v i| ≤ epMabs n A w v := by
  unfold epMabs
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i hi => ?_)
  have hw := h.hw i (Finset.mem_range.mp hi)
  rw [abs_mul, abs_mul, abs_of_nonneg hw]

/-- A gated score, split into the exact product and its error. -/
theorem ep_at_split (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) {i : ℕ} (hi : i < n) :
    |At i| ≤ |A i| * w i +
      ((9 / 100000 + 8 / 10 * Γ) * w i * PA i + 1 / 2 ^ 112 * (1 + w i + PA i)) := by
  have h1 := h.hAt i hi
  rw [ep_cA, ep_fl] at h1
  have hw := h.hw i hi
  have h2 := abs_sub_abs_le_abs_sub (At i) (A i * w i)
  rw [abs_mul, abs_of_nonneg hw] at h2
  linarith

/-- A gated score is at most `1.002·w·PA` plus a tiny floor. -/
theorem ep_at_abs (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) {i : ℕ} (hi : i < n) :
    |At i| ≤ 1002 / 1000 * (w i * PA i) + 1 / 2 ^ 112 * (1 + w i + PA i) := by
  have h1 := ep_at_split h hi
  have hw := h.hw i hi
  have hPA := h.hPA i hi
  have h2 : |A i| * w i ≤ PA i * w i :=
    mul_le_mul_of_nonneg_right (h.hAPA i hi) hw
  have h3 : Γ * (w i * PA i) ≤ 1 / 512 * (w i * PA i) :=
    mul_le_mul_of_nonneg_right (ep_G9 h) (mul_nonneg hw hPA)
  have h4 := mul_nonneg hw hPA
  linarith

/-- A gated score is below `2^101` (room for its conversion to `bfloat16`). -/
theorem ep_room_a (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) {i : ℕ} (hi : i < n) :
    |At i| ≤ 2 ^ 101 := by
  have h1 := ep_at_abs h hi
  have h2 := h.hwPA i hi
  have h3 := h.hw64 i hi
  have h4 := h.hPA100 i hi
  linarith

/-- The bfloat16 score plus the floor, per key. -/
theorem ep_a_abs (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) {i : ℕ} (hi : i < n) :
    |a i| + 2 ^ (-126 : ℤ) ≤ 101 / 100 * (w i * PA i) + 1 / 10 ^ 32 * (1 + w i + PA i) := by
  have h1 := ep_at_abs h hi
  have h2 := h.ha i hi
  have e8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 2 ^ 8 := by norm_num
  have e134 : (2 : ℝ) ^ (-134 : ℤ) = 1 / 2 ^ 134 := by norm_num
  have e126 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  rw [e8, e134] at h2
  have h3 := abs_sub_abs_le_abs_sub (a i) (At i)
  have hw := h.hw i hi
  have hPA := h.hPA i hi
  have h4 := mul_nonneg hw hPA
  rw [e126]
  linarith

/-- The sum of the lifted magnitudes of the last dot's products, bounded by the two sums. -/
theorem ep_sum1 (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    ∑ i ∈ Finset.range n, (|a i| + 2 ^ (-126 : ℤ)) * Lv i ≤
      101 / 100 * (∑ i ∈ Finset.range n, w i * PA i * Lv i) +
        1 / 10 ^ 32 * (∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i) := by
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro i hi
  have hi' := Finset.mem_range.mp hi
  have h1 := ep_a_abs h hi'
  have hLv := h.hLv i hi'
  calc (|a i| + 2 ^ (-126 : ℤ)) * Lv i
      ≤ (101 / 100 * (w i * PA i) + 1 / 10 ^ 32 * (1 + w i + PA i)) * Lv i :=
        mul_le_mul_of_nonneg_right h1 hLv
    _ = _ := by ring

/-- The sum of the lifted magnitudes of the last dot's products is below `2^101` (room for the
dot into `float32`). -/
theorem ep_room_D (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    ∑ i ∈ Finset.range n, (|a i| + 2 ^ (-126 : ℤ)) * Lv i ≤ 2 ^ 101 := by
  have h1 := ep_sum1 h
  have hr := h.hroom
  unfold epMag epFlo at hr
  have hP1 : 0 ≤ G * PQS := mul_nonneg h.hG h.hPQS
  have hP2 := ep_P2_nn h
  have hF2 := ep_F2_nn h
  have hG := h.hG
  have hPQS := h.hPQS
  generalize (∑ i ∈ Finset.range n, (|a i| + 2 ^ (-126 : ℤ)) * Lv i) = S1 at h1 ⊢
  linarith

/-- One key's error between the rounded and the exact product. -/
theorem ep_term (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) {i : ℕ} (hi : i < n) :
    |a i * v i - A i * w i * v i| ≤ 1 / 256 * (|A i| * w i * |v i|) +
      257 / 256 * (9 / 100000 + 8 / 10 * Γ) * (w i * PA i * Lv i) +
      (257 / 256 / 2 ^ 112 + 1 / 2 ^ 134) * ((1 + w i + PA i) * Lv i) := by
  have hat := ep_at_split h hi
  have h2 := h.ha i hi
  have e8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 2 ^ 8 := by norm_num
  have e134 : (2 : ℝ) ^ (-134 : ℤ) = 1 / 2 ^ 134 := by norm_num
  rw [e8, e134] at h2
  have hw := h.hw i hi
  have hPA := h.hPA i hi
  have hLv := h.hLv i hi
  have hv := h.hv i hi
  have hΓ := h.hΓ
  have h3 := h.hAt i hi
  rw [ep_cA, ep_fl] at h3
  have hB : |a i - A i * w i| ≤ 1 / 256 * (|A i| * w i) +
      257 / 256 * ((9 / 100000 + 8 / 10 * Γ) * w i * PA i + 1 / 2 ^ 112 * (1 + w i + PA i))
        + 1 / 2 ^ 134 := by
    have := abs_sub_le (a i) (At i) (A i * w i)
    linarith
  have he0 : 0 ≤ (9 / 100000 + 8 / 10 * Γ) * w i * PA i + 1 / 2 ^ 112 * (1 + w i + PA i) := by
    have h1 := mul_nonneg (mul_nonneg (show (0 : ℝ) ≤ 9 / 100000 + 8 / 10 * Γ by linarith) hw) hPA
    have h2 : 0 ≤ 1 / 2 ^ 112 * (1 + w i + PA i) := mul_nonneg (by norm_num) (by linarith)
    linarith
  have e : a i * v i - A i * w i * v i = (a i - A i * w i) * v i := by ring
  rw [e, abs_mul]
  have h5 := mul_le_mul_of_nonneg_right hB (abs_nonneg (v i))
  have h6 := mul_le_mul_of_nonneg_left hv he0
  have h7 : 1 / 2 ^ 134 * |v i| ≤ 1 / 2 ^ 134 * Lv i :=
    mul_le_mul_of_nonneg_left hv (by norm_num)
  have h8 : 0 ≤ w i * Lv i := mul_nonneg hw hLv
  have h9 : 0 ≤ PA i * Lv i := mul_nonneg hPA hLv
  linarith

/-- The error between the last dot's exact value with `a` and with `A·w`. -/
theorem ep_diff (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    |∑ i ∈ Finset.range n, a i * v i - ∑ i ∈ Finset.range n, A i * w i * v i| ≤
      1 / 256 * (∑ i ∈ Finset.range n, |A i| * w i * |v i|) +
      257 / 256 * (9 / 100000 + 8 / 10 * Γ) * (∑ i ∈ Finset.range n, w i * PA i * Lv i)
      + (257 / 256 / 2 ^ 112 + 1 / 2 ^ 134) * (∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i) := by
  rw [← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  have e : ∑ i ∈ Finset.range n, (1 / 256 * (|A i| * w i * |v i|) +
      257 / 256 * (9 / 100000 + 8 / 10 * Γ) * (w i * PA i * Lv i) +
      (257 / 256 / 2 ^ 112 + 1 / 2 ^ 134) * ((1 + w i + PA i) * Lv i)) =
      1 / 256 * (∑ i ∈ Finset.range n, |A i| * w i * |v i|) +
      257 / 256 * (9 / 100000 + 8 / 10 * Γ) * (∑ i ∈ Finset.range n, w i * PA i * Lv i)
      + (257 / 256 / 2 ^ 112 + 1 / 2 ^ 134) *
        (∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i) := by
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
      ← Finset.mul_sum]
  rw [← e]
  exact Finset.sum_le_sum fun i hi => ep_term h (Finset.mem_range.mp hi)

/-- The inter-chunk term is at most `1.002·G·PQS` plus a tiny floor. -/
theorem ep_ot_abs (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    |ot| ≤ 1002 / 1000 * (G * PQS) + 1 / 10 ^ 32 * (1 + G + PQS) := by
  have h1 := h.hot
  rw [ep_cO, ep_fl] at h1
  have hGQS : |G * QS| ≤ G * PQS := by
    rw [abs_mul, abs_of_nonneg h.hG]
    exact mul_le_mul_of_nonneg_left h.hQS h.hG
  have h2 := abs_sub_abs_le_abs_sub ot (G * QS)
  have h3 : Γ * (G * PQS) ≤ 1 / 512 * (G * PQS) :=
    mul_le_mul_of_nonneg_right (ep_G9 h) (mul_nonneg h.hG h.hPQS)
  have hG := h.hG
  have hPQS := h.hPQS
  have hP1 : 0 ≤ G * PQS := mul_nonneg hG hPQS
  linarith

/-- The scaled inter-chunk term is below `2^102` (room for its product with the scale). -/
theorem ep_room_ot (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) :
    |ot * σh| ≤ 2 ^ 102 := by
  have hot := ep_ot_abs h
  have hb := ep_hb h
  have hσ0 := ep_σpos h
  have hr := h.hroomσ
  unfold epMag epFlo at hr
  have hG := h.hG
  have hPQS := h.hPQS
  have hP1 : 0 ≤ G * PQS := mul_nonneg hG hPQS
  have hP2 := ep_P2_nn h
  have hF2 := ep_F2_nn h
  have s1 : 0 ≤ σ * (G * PQS) := mul_nonneg hσ0.le hP1
  have s2 : 0 ≤ σ * G := mul_nonneg hσ0.le hG
  have s3 : 0 ≤ σ * PQS := mul_nonneg hσ0.le hPQS
  have s4 : 0 ≤ σ * (∑ i ∈ Finset.range n, w i * PA i * Lv i) := mul_nonneg hσ0.le hP2
  have s5 : 0 ≤ σ * (∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i) := mul_nonneg hσ0.le hF2
  rw [abs_mul]
  have := mul_le_mul hot hb (abs_nonneg _)
    (by linarith : (0 : ℝ) ≤ 1002 / 1000 * (G * PQS) + 1 / 10 ^ 32 * (1 + G + PQS))
  linarith

/-- The last dot's error bound gives `|D| ≤ 1.02·Σ w·PA·Lv` plus a tiny floor. -/
theorem ep_D_abs (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) {D : ℝ}
    (hD : |D - ∑ i ∈ Finset.range n, a i * v i| ≤
      66 * 2 ^ (-22 : ℤ) * ∑ i ∈ Finset.range n, (|a i| + 2 ^ (-126 : ℤ)) * Lv i +
        66 * 2 ^ (-126 : ℤ)) :
    |D| ≤ 102 / 100 * (∑ i ∈ Finset.range n, w i * PA i * Lv i) +
      2 / 10 ^ 32 * (1 + ∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i) := by
  have e22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
  have e126 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have hT : |∑ i ∈ Finset.range n, a i * v i| ≤
      ∑ i ∈ Finset.range n, (|a i| + 2 ^ (-126 : ℤ)) * Lv i := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i hi => ?_)
    have hi' := Finset.mem_range.mp hi
    have hLv := h.hLv i hi'
    rw [abs_mul]
    have h1 : |a i| * |v i| ≤ |a i| * Lv i :=
      mul_le_mul_of_nonneg_left (h.hv i hi') (abs_nonneg _)
    have h2 : (0 : ℝ) ≤ 2 ^ (-126 : ℤ) * Lv i := mul_nonneg (by positivity) hLv
    linarith
  have h1 := abs_sub_abs_le_abs_sub D (∑ i ∈ Finset.range n, a i * v i)
  have hs1 := ep_sum1 h
  have hF2 := ep_F2_nn h
  have hP2 := ep_P2_nn h
  generalize (∑ i ∈ Finset.range n, (|a i| + 2 ^ (-126 : ℤ)) * Lv i) = S1 at hD hT hs1
  rw [e22, e126] at hD
  linarith

/-- The scaled last dot is below `2^102` (room for its product with the scale). -/
theorem ep_room_Dm (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) {D : ℝ}
    (hD : |D - ∑ i ∈ Finset.range n, a i * v i| ≤
      66 * 2 ^ (-22 : ℤ) * ∑ i ∈ Finset.range n, (|a i| + 2 ^ (-126 : ℤ)) * Lv i +
        66 * 2 ^ (-126 : ℤ)) :
    |D * σh| ≤ 2 ^ 102 := by
  have hDa := ep_D_abs h hD
  have hb := ep_hb h
  have hσ0 := ep_σpos h
  have hr := h.hroomσ
  unfold epMag epFlo at hr
  have hG := h.hG
  have hPQS := h.hPQS
  have hP1 : 0 ≤ G * PQS := mul_nonneg hG hPQS
  have hP2 := ep_P2_nn h
  have hF2 := ep_F2_nn h
  have s1 : 0 ≤ σ * (G * PQS) := mul_nonneg hσ0.le hP1
  have s2 : 0 ≤ σ * G := mul_nonneg hσ0.le hG
  have s3 : 0 ≤ σ * PQS := mul_nonneg hσ0.le hPQS
  have s4 : 0 ≤ σ * (∑ i ∈ Finset.range n, w i * PA i * Lv i) := mul_nonneg hσ0.le hP2
  have s5 : 0 ≤ σ * (∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i) := mul_nonneg hσ0.le hF2
  rw [abs_mul]
  have := mul_le_mul hDa hb (abs_nonneg _)
    (by linarith : (0 : ℝ) ≤ 102 / 100 * (∑ i ∈ Finset.range n, w i * PA i * Lv i) +
      2 / 10 ^ 32 * (1 + ∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i))
  linarith

/-- The two products are below `2^104` together and their sum below `2^105` (room for the sum
and the conversion of the result to `bfloat16`). -/
theorem ep_room_x {m1 m2 x ot' D' : ℝ} (hro1 : |ot'| ≤ 2 ^ 102) (hro2 : |D'| ≤ 2 ^ 102)
    (hm1 : |m1 - ot'| ≤ 2 ^ (-24 : ℤ) * |ot'| + 2 ^ (-150 : ℤ))
    (hm2 : |m2 - D'| ≤ 2 ^ (-24 : ℤ) * |D'| + 2 ^ (-150 : ℤ))
    (hx : |x - (m1 + m2)| ≤ 3 * 2 ^ (-24 : ℤ) * (|m1| + |m2|) + 4 * 2 ^ (-150 : ℤ)) :
    |m1| + |m2| ≤ 2 ^ 104 ∧ |x| ≤ 2 ^ 105 := by
  have e24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have e150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  rw [e24, e150] at hm1 hm2 hx
  have b1 := abs_sub_abs_le_abs_sub m1 ot'
  have b2 := abs_sub_abs_le_abs_sub m2 D'
  have b3 := abs_sub_abs_le_abs_sub x (m1 + m2)
  have b4 := abs_add_le m1 m2
  refine ⟨?_, ?_⟩
  · norm_num at *; linarith
  · norm_num at *; linarith

/-- **The epilogue's chain**: the stored value `c` is within the claim's tolerance of
`σ·(G·QS + Σ A·w·v)`. -/
theorem ep_final (h : EpIn n σ σh G Γ QS PQS ot A w PA v Lv At a) {D m1 m2 x c : ℝ}
    (hD : |D - ∑ i ∈ Finset.range n, a i * v i| ≤
      66 * 2 ^ (-22 : ℤ) * ∑ i ∈ Finset.range n, (|a i| + 2 ^ (-126 : ℤ)) * Lv i +
        66 * 2 ^ (-126 : ℤ))
    (hm1 : |m1 - ot * σh| ≤ 2 ^ (-24 : ℤ) * |ot * σh| + 2 ^ (-150 : ℤ))
    (hm2 : |m2 - D * σh| ≤ 2 ^ (-24 : ℤ) * |D * σh| + 2 ^ (-150 : ℤ))
    (hx : |x - (m1 + m2)| ≤ 3 * 2 ^ (-24 : ℤ) * (|m1| + |m2|) + 4 * 2 ^ (-150 : ℤ))
    (hc : |c - x| ≤ 2 ^ (-8 : ℤ) * |x| + 2 ^ (-134 : ℤ)) :
    |c - σ * (G * QS + ∑ i ∈ Finset.range n, A i * w i * v i)| ≤
      2 ^ (-8 : ℤ) * |σ * (G * QS + ∑ i ∈ Finset.range n, A i * w i * v i)| +
        (1 + 2 ^ (-8 : ℤ)) * σ * (2 ^ (-8 : ℤ) * epMabs n A w v +
          (2 ^ (-13 : ℤ) + Γ) * epMag n G PQS w PA Lv +
          2 ^ (-100 : ℤ) * epFlo n G PQS w PA Lv) + 2 ^ (-130 : ℤ) := by
  have e8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 2 ^ 8 := by norm_num
  have e13 : (2 : ℝ) ^ (-13 : ℤ) = 1 / 2 ^ 13 := by norm_num
  have e22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
  have e24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have e100 : (2 : ℝ) ^ (-100 : ℤ) = 1 / 2 ^ 100 := by norm_num
  have e126 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have e130 : (2 : ℝ) ^ (-130 : ℤ) = 1 / 2 ^ 130 := by norm_num
  have e134 : (2 : ℝ) ^ (-134 : ℤ) = 1 / 2 ^ 134 := by norm_num
  have e150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have hσ0 := ep_σpos h
  have hsh := ep_sh_dev h
  have hG := h.hG
  have hΓ := h.hΓ
  have hPQS := h.hPQS
  have hP1 : 0 ≤ G * PQS := mul_nonneg hG hPQS
  have hP2 := ep_P2_nn h
  have hF2 := ep_F2_nn h
  have hMabs := ep_Mabs_nn h
  have hMP := ep_Mabs_le h
  have hS := ep_S_le h
  have hGQS : |G * QS| ≤ G * PQS := by
    rw [abs_mul, abs_of_nonneg hG]
    exact mul_le_mul_of_nonneg_left h.hQS hG
  have hot : |ot - G * QS| ≤ (7 / 100000 + 6 / 10 * Γ) * (G * PQS) + 1 / 2 ^ 112 * (1 + G + PQS) := by
    have := h.hot
    rw [ep_cO, ep_fl] at this
    linarith
  have hota := ep_ot_abs h
  have hDa := ep_D_abs h hD
  have hDS : |D - ∑ i ∈ Finset.range n, A i * w i * v i| ≤
      1 / 256 * (∑ i ∈ Finset.range n, |A i| * w i * |v i|) +
      257 / 256 * (9 / 100000 + 8 / 10 * Γ) * (∑ i ∈ Finset.range n, w i * PA i * Lv i)
      + (257 / 256 / 2 ^ 112 + 1 / 2 ^ 134) * (∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i)
      + 66 / 2 ^ 22 * (101 / 100 * (∑ i ∈ Finset.range n, w i * PA i * Lv i) +
        1 / 10 ^ 32 * (∑ i ∈ Finset.range n, (1 + w i + PA i) * Lv i)) + 66 / 2 ^ 126 := by
    have h1 := abs_sub_le D (∑ i ∈ Finset.range n, a i * v i)
      (∑ i ∈ Finset.range n, A i * w i * v i)
    have h2 := ep_diff h
    rw [e22] at hD
    have h3 := ep_sum1 h
    generalize (∑ i ∈ Finset.range n, (|a i| + 2 ^ (-126 : ℤ)) * Lv i) = S1 at hD h3
    rw [e126] at hD
    linarith
  rw [e24, e150] at hm1 hm2 hx
  rw [e8, e134] at hc
  have hx' : |x - (m1 + m2)| ≤ 3 / 2 ^ 24 * (|m1| + |m2|) + 4 / 2 ^ 150 := by linarith
  have key := ep_alg hσ0 hsh hΓ hP1 hP2
    (show (1 : ℝ) ≤ 1 + G + PQS by linarith) hF2 hMabs hMP hGQS hS hot hota hDS hDa hm1 hm2 hx' hc
  unfold epMag epFlo epMabs
  rw [e8, e13, e100, e130]
  unfold epMabs at key
  linarith

end

end Claims.ChunkFwdO
