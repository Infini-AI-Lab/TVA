import Proof.LaneG2

/-!
# One lane's output (lines 91–92) and the whole lane

`lane_out`: `activated.to(bfloat16).to(float32)`, the product with `up` and the store's `bfloat16`
rounding are within `(2^-7 + 2^-12)·|G·u| + 2^-133·|u| + 2^-133` of `G·u`. `lane_comb` moves the
bound from the kernel's sparse gate to the exact one with the `2`-Lipschitz `geluRef`, and
`lane_ok` chains the lane's operations from the gate and the cutoff to the stored element.
-/

namespace Claims.GeluAndMulSparse

open TVA

set_option maxHeartbeats 1600000

/-- **The lane's output** (lines 91–92). -/
theorem lane_out {act av pr u : F32} {actb y : BF16} {G : ℝ}
    (hact : act.isFinite = true) (hG0 : 0 ≤ G) (hG1 : G ≤ 2 ^ 41)
    (hae : |act.toReal - G| ≤ 2 ^ (-15 : ℤ) * G + 2 ^ (-145 : ℤ))
    (hu : u.isFinite = true) (hub : |u.toReal| ≤ 2 ^ 32)
    (h1 : FloatV.NaNEq (FloatV.cvtWith .rn act) actb) (h2 : FloatV.NaNEq (FloatV.cvtWith .rn actb) av)
    (h3 : FloatV.NaNEq (FloatV.mul av u) pr) (h4 : FloatV.NaNEq (FloatV.cvtWith .rn pr) y) :
    y.isFinite = true ∧
      |y.toReal - G * u.toReal| ≤
        (2 ^ (-7 : ℤ) + 2 ^ (-12 : ℤ)) * |G * u.toReal| + 2 ^ (-133 : ℤ) * |u.toReal| + 2 ^ (-133 : ℤ) ∧
      (act.toRat = 0 → y.toRat = 0) := by
  have h15 : (2 : ℝ) ^ (-15 : ℤ) = 1 / 32768 := by norm_num
  rw [h15] at hae
  have hQ0 : (0 : ℝ) < 2 ^ (-133 : ℤ) := by positivity
  have hQ1 : (2 : ℝ) ^ (-133 : ℤ) ≤ 1 / 10 ^ 40 := by norm_num
  have h145 : (2 : ℝ) ^ (-145 : ℤ) ≤ 2 ^ (-133 : ℤ) / 4000 := by norm_num
  have hub8 : ub = 1 / 256 := by unfold ub; norm_num
  have heb : etab ≤ 2 ^ (-133 : ℤ) / 2 := by unfold etab; norm_num
  have heb0 : 0 ≤ etab := by unfold etab; positivity
  have hu32 : u32 = 1 / 16777216 := by unfold u32; norm_num
  have he : eta32 ≤ 2 ^ (-133 : ℤ) / 10 ^ 4 := by unfold eta32; norm_num
  have he0 : 0 ≤ eta32 := by unfold eta32; positivity
  generalize (2 : ℝ) ^ (-133 : ℤ) = Q at hQ0 hQ1 h145 heb he ⊢
  have hae' : |act.toReal - G| ≤ 1 / 32768 * G + Q / 4000 := by linarith
  have hab : |act.toReal| ≤ 2 ^ 42 := by
    have := abs_le_of_err hae'; rw [abs_of_nonneg hG0] at this; linarith
  obtain ⟨abf, abe⟩ := cvt16_lane hact (hab.trans (by norm_num)) h1
  obtain ⟨avf, avv⟩ := cvt32_lane abf h2
  rw [hub8] at abe
  rw [← avv] at abe
  have hacta : |act.toReal| ≤ G + (1 / 32768 * G + Q / 4000) := by
    have := abs_le_of_err hae'; rw [abs_of_nonneg hG0] at this; exact this
  -- av against G
  have hBG : |av.toReal - G| ≤ (1 / 256 + 1 / 32768 + 1 / (256 * 32768)) * G + 6 / 10 * Q := by
    have : av.toReal - G = (av.toReal - act.toReal) + (act.toReal - G) := by ring
    rw [this]
    have t := abs_add_le (av.toReal - act.toReal) (act.toReal - G)
    have q : 1 / 256 * |act.toReal| ≤ 1 / 256 * (G + (1 / 32768 * G + Q / 4000)) := by gcongr
    linarith
  have hBb : |av.toReal| ≤ 2 ^ 43 := by
    have := abs_le_of_err hBG; rw [abs_of_nonneg hG0] at this; linarith
  have r3 : |av.toReal * u.toReal| ≤ 2 ^ 75 := by
    rw [abs_mul]
    calc |av.toReal| * |u.toReal| ≤ 2 ^ 43 * 2 ^ 32 := mul_le_mul hBb hub (abs_nonneg _) (by positivity)
      _ = 2 ^ 75 := by norm_num
  obtain ⟨prf, pre⟩ := mul_lane avf hu (r3.trans (by norm_num)) h3
  rw [hu32] at pre
  have prb : |pr.toReal| ≤ 2 ^ 120 := by
    have := abs_le_of_err pre; linarith
  obtain ⟨yf, ye⟩ := cvt16_lane prf prb h4
  rw [hub8] at ye
  refine ⟨yf, ?_, fun h0 => ?_⟩
  · have hu0 := abs_nonneg u.toReal
    have hGu : |G * u.toReal| = G * |u.toReal| := by rw [abs_mul, abs_of_nonneg hG0]
    rw [hGu]
    have hGu0 : 0 ≤ G * |u.toReal| := by positivity
    have hBu : |av.toReal * u.toReal - G * u.toReal| ≤
        (1 / 256 + 1 / 32768 + 1 / (256 * 32768)) * (G * |u.toReal|) + 6 / 10 * (Q * |u.toReal|) := by
      have : av.toReal * u.toReal - G * u.toReal = (av.toReal - G) * u.toReal := by ring
      rw [this, abs_mul]
      have := mul_le_mul_of_nonneg_right hBG hu0
      linarith
    have hQu : Q * |u.toReal| ≤ |u.toReal| / 10 ^ 40 := by
      have := mul_le_mul_of_nonneg_right hQ1 hu0; linarith
    have hBu' : |av.toReal * u.toReal| ≤ G * |u.toReal| * (1 + 1 / 128) + Q * |u.toReal| := by
      have := abs_le_of_err hBu; rw [hGu] at this
      have hQu0 : 0 ≤ Q * |u.toReal| := by positivity
      linarith
    have hpr : |pr.toReal - G * u.toReal| ≤
        (1 / 256 + 1 / 32768 + 1 / (256 * 32768)) * (G * |u.toReal|) + 6 / 10 * (Q * |u.toReal|) +
        1 / 16777216 * (G * |u.toReal| * (1 + 1 / 128) + Q * |u.toReal|) + eta32 := by
      have : pr.toReal - G * u.toReal = (pr.toReal - av.toReal * u.toReal) +
          (av.toReal * u.toReal - G * u.toReal) := by ring
      rw [this]
      have t := abs_add_le (pr.toReal - av.toReal * u.toReal) (av.toReal * u.toReal - G * u.toReal)
      have : 1 / 16777216 * |av.toReal * u.toReal| ≤
          1 / 16777216 * (G * |u.toReal| * (1 + 1 / 128) + Q * |u.toReal|) := by gcongr
      linarith
    have hprG : |pr.toReal| ≤ G * |u.toReal| + |pr.toReal - G * u.toReal| := by
      have := abs_sub_abs_le_abs_sub pr.toReal (G * u.toReal)
      rw [abs_mul, abs_of_nonneg hG0] at this; linarith
    have ye' : |y.toReal - pr.toReal| ≤ 1 / 256 * (G * |u.toReal|) +
        1 / 256 * |pr.toReal - G * u.toReal| + etab := by
      have := mul_le_mul_of_nonneg_left hprG (by norm_num : (0 : ℝ) ≤ 1 / 256); linarith
    have hsplit : y.toReal - G * u.toReal = (y.toReal - pr.toReal) + (pr.toReal - G * u.toReal) := by ring
    rw [hsplit]
    have t := abs_add_le (y.toReal - pr.toReal) (pr.toReal - G * u.toReal)
    have hc1 : (2 : ℝ) ^ (-7 : ℤ) + 2 ^ (-12 : ℤ) = 1 / 128 + 1 / 4096 := by norm_num
    rw [hc1]
    have hQu0 : 0 ≤ Q * |u.toReal| := by positivity
    have hpr0 := abs_nonneg (pr.toReal - G * u.toReal)
    nlinarith
  · obtain ⟨-, hb0⟩ := cvt16_zero hact h0 h1
    have hav0 : av.toRat = 0 := by
      have h5 : av.toReal = 0 := by rw [avv, FloatV.toReal_eq, hb0]; norm_num
      rw [FloatV.toReal_eq] at h5; exact_mod_cast h5
    obtain ⟨-, hpr0⟩ := mul_zero_left avf hav0 hu h3
    exact (cvt16_zero prf hpr0 h4).2

/-- **From the kernel's sparse gate to the exact one**: `geluRef` is `2`-Lipschitz on `[0, ∞)` and
at least half its argument, so the lane's bound against `geluRef s · u` becomes the claim's `yTol`
against `geluRef (max (g − cRef) 0) · u`. -/
theorem lane_comb {g c cR cT s y u : ℝ} (hcT : 2 ^ (-130 : ℤ) ≤ cT)
    (hc : |c - cR| ≤ 7 / 8 * cT - 2 ^ (-131 : ℤ)) (hs0 : 0 ≤ s)
    (hs : |s - max (g - c) 0| ≤ (2 ^ (-8 : ℤ) + 2 ^ (-20 : ℤ)) * max (g - c) 0 + 2 ^ (-131 : ℤ))
    (hy : |y - geluRef s * u| ≤
      (2 ^ (-7 : ℤ) + 2 ^ (-12 : ℤ)) * |geluRef s * u| + 2 ^ (-133 : ℤ) * |u| + 2 ^ (-133 : ℤ)) :
    |y - geluRef (max (g - cR) 0) * u| ≤
      2 ^ (-5 : ℤ) * |geluRef (max (g - cR) 0) * u| + 2 * |u| * cT + 2 ^ (-132 : ℤ) := by
  obtain ⟨sR, hsR⟩ : ∃ x, x = max (g - cR) 0 := ⟨_, rfl⟩
  obtain ⟨m, hm⟩ : ∃ x, x = max (g - c) 0 := ⟨_, rfl⟩
  obtain ⟨e, he⟩ : ∃ x, x = |c - cR| := ⟨_, rfl⟩
  have hsR0 : 0 ≤ sR := by rw [hsR]; exact le_max_right _ _
  have hm0 : 0 ≤ m := by rw [hm]; exact le_max_right _ _
  have he0 : 0 ≤ e := by rw [he]; exact abs_nonneg _
  have hmax : |m - sR| ≤ e := by
    rw [hm, hsR, he]
    have := abs_max_sub_max_le_abs (g - c) (g - cR) 0
    have h2 : |g - c - (g - cR)| = |c - cR| := by rw [abs_sub_comm]; ring_nf
    linarith
  have hmle : m ≤ sR + e := by have := (abs_le.mp hmax).2; linarith
  rw [← hm] at hs
  rw [← he] at hc
  rw [← hsR]
  have hk : (2 : ℝ) ^ (-8 : ℤ) + 2 ^ (-20 : ℤ) = 1 / 256 + 1 / 1048576 := by norm_num
  have h7 : (2 : ℝ) ^ (-7 : ℤ) + 2 ^ (-12 : ℤ) = 1 / 128 + 1 / 4096 := by norm_num
  have hp130 : (2 : ℝ) ^ (-130 : ℤ) = 2 * 2 ^ (-131 : ℤ) := by norm_num
  have hp133 : (2 : ℝ) ^ (-133 : ℤ) = 2 ^ (-131 : ℤ) / 4 := by norm_num
  have hp132 : (2 : ℝ) ^ (-132 : ℤ) = 2 ^ (-131 : ℤ) / 2 := by norm_num
  have hp5 : (2 : ℝ) ^ (-5 : ℤ) = 1 / 32 := by norm_num
  rw [hk] at hs
  rw [h7, hp133] at hy
  rw [hp130] at hcT
  rw [hp132, hp5]
  obtain ⟨P, hP⟩ : ∃ x, x = (2 : ℝ) ^ (-131 : ℤ) := ⟨_, rfl⟩
  rw [← hP] at hs hy hcT hc ⊢
  have hP0 : 0 ≤ P := by rw [hP]; positivity
  obtain ⟨Gs, hGs⟩ : ∃ x, x = geluRef s := ⟨_, rfl⟩
  obtain ⟨Gr, hGr⟩ : ∃ x, x = geluRef sR := ⟨_, rfl⟩
  have hGr0 : 0 ≤ Gr := by rw [hGr]; exact geluRef_nonneg hsR0
  have hsRG : sR ≤ 2 * Gr := by have := geluRef_ge hsR0; rw [hGr]; linarith
  have hlip : |Gs - Gr| ≤ 2 * |s - sR| := by rw [hGs, hGr]; exact geluRef_lip hs0 hsR0
  rw [← hGs] at hy
  rw [← hGr]
  have hss : |s - sR| ≤ (1 / 256 + 1 / 1048576) * (sR + e) + P + e := by
    have : s - sR = (s - m) + (m - sR) := by ring
    rw [this]
    have t := abs_add_le (s - m) (m - sR)
    have : (1 / 256 + 1 / 1048576) * m ≤ (1 / 256 + 1 / 1048576) * (sR + e) :=
      mul_le_mul_of_nonneg_left hmle (by norm_num)
    linarith
  have hu0 := abs_nonneg u
  have hGu : |Gr * u| = Gr * |u| := by rw [abs_mul, abs_of_nonneg hGr0]
  rw [hGu]
  have hD : |Gs - Gr| ≤ 4 * (1 / 256 + 1 / 1048576) * Gr + 2 * (1 + 1 / 256 + 1 / 1048576) * e + 2 * P := by
    have : (1 / 256 + 1 / 1048576 : ℝ) * sR ≤ (1 / 256 + 1 / 1048576) * (2 * Gr) :=
      mul_le_mul_of_nonneg_left hsRG (by norm_num)
    linarith
  have hD' : |Gs - Gr| ≤ 4 * (1 / 256 + 1 / 1048576) * Gr + 2 * (1 + 1 / 256 + 1 / 1048576) * (7 / 8 * cT) := by
    linarith
  obtain ⟨D, hDd⟩ : ∃ x, x = |Gs - Gr| := ⟨_, rfl⟩
  rw [← hDd] at hD'
  have hD0 : 0 ≤ D := by rw [hDd]; exact abs_nonneg _
  have hGsu : |Gs * u| ≤ Gr * |u| + D * |u| := by
    have : Gs * u = Gr * u + (Gs - Gr) * u := by ring
    rw [this]
    have := abs_add_le (Gr * u) ((Gs - Gr) * u)
    rw [abs_mul (Gs - Gr), hGu, ← hDd] at this
    exact this
  have hsplit : y - Gr * u = (y - Gs * u) + (Gs - Gr) * u := by ring
  rw [hsplit]
  have t := abs_add_le (y - Gs * u) ((Gs - Gr) * u)
  rw [abs_mul (Gs - Gr), ← hDd] at t
  have hDu : D * |u| ≤ 4 * (1 / 256 + 1 / 1048576) * (Gr * |u|) +
      2 * (1 + 1 / 256 + 1 / 1048576) * (7 / 8 * cT) * |u| := by
    have := mul_le_mul_of_nonneg_right hD' hu0
    linarith
  have hy' : |y - Gs * u| ≤ (1 / 128 + 1 / 4096) * (Gr * |u| + D * |u|) + P / 4 * |u| + P / 4 := by
    have := mul_le_mul_of_nonneg_left hGsu (by norm_num : (0 : ℝ) ≤ 1 / 128 + 1 / 4096)
    linarith
  have hPu : P * |u| ≤ cT * |u| / 2 := by
    have := mul_le_mul_of_nonneg_right hcT hu0; linarith
  have hGu0 : 0 ≤ Gr * |u| := mul_nonneg hGr0 hu0
  have hcTu : 0 ≤ cT * |u| := mul_nonneg (by linarith) hu0
  nlinarith

end Claims.GeluAndMulSparse
