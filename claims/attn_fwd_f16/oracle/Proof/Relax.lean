import Proof.Tol

/-!
# The fine bound is below the claim's tolerance

For `D ∈ {64, 128, 256}`, `T` key blocks (`2 ≤ T ≤ 1024`, so `64·T ≤ N ≤ 2^16`) and
`0 ≤ X ≤ 2^10`: the fine bound of `Proof.Tol` is at most `epsO` (with the part proportional to
`Σ|v| ≤ N·attnMag` counted), `2^-24` for its constant part, and `epsM`.
-/

namespace Claims.AttnFwd

/-- The head sizes and score bounds the claim covers. -/
structure Par (D : ℕ) (X : ℝ) : Prop where
  hD : D = 64 ∨ D = 128 ∨ D = 256
  hX : 0 ≤ X ∧ X ≤ 1024

/-- The box of parameters the claim covers: `Par`, and `T` key blocks. -/
structure Box (D : ℕ) (T X : ℝ) : Prop where
  hD : D = 64 ∨ D = 128 ∨ D = 256
  hT : 2 ≤ T ∧ T ≤ 1024
  hX : 0 ≤ X ∧ X ≤ 1024

/-- A `Box` bound implies the `Par` bound. -/
theorem Box.par {D : ℕ} {T X : ℝ} (h : Box D T X) : Par D X := ⟨h.hD, h.hX⟩

/-! ### Helper lemmas -/

theorem q22_pos : (0 : ℝ) < q22 := by unfold q22; norm_num

/-- `xi` is nonnegative for nonnegative `X`. -/
theorem xi_nonneg_of {D : ℕ} {X : ℝ} (hX : 0 ≤ X) : 0 ≤ xi D X := by
  unfold xi rD
  positivity

/-- `xi` is monotone in `X`. -/
theorem xi_mono {D : ℕ} {X Y : ℝ} (h : X ≤ Y) : xi D X ≤ xi D Y := by
  unfold xi rD
  have hc : 0 ≤ 14428 / 10000 * (((D : ℝ) + 2) / 2 ^ 22) + 1 / 2 ^ 20 := by positivity
  nlinarith [mul_le_mul_of_nonneg_right h hc]

/-- `rhoP` is nonnegative when `xi` is. -/
theorem rhoP_nonneg_of {D : ℕ} {X : ℝ} (h0 : 0 ≤ xi D X) : 0 ≤ rhoP D X := by
  unfold rhoP
  have := q22_pos
  positivity

/-- `rhoP` is bounded affinely in terms of a bound on `xi`. -/
theorem rhoP_aff {D : ℕ} {X xm : ℝ} (h0 : 0 ≤ xi D X) (h1 : xi D X ≤ xm) :
    rhoP D X ≤ xi D X * (1 + xm + q22 * (1 + xm)) + q22 := by
  unfold rhoP
  have hq : 0 < q22 := q22_pos
  have h2 : xi D X * xi D X ≤ xi D X * xm := mul_le_mul_of_nonneg_left h1 h0
  have h3 : q22 * (xi D X * xi D X) ≤ q22 * (xi D X * xm) :=
    mul_le_mul_of_nonneg_left h2 hq.le
  nlinarith

/-- `rho16` equals `rhoP * (1 + u16) + u16`. -/
theorem rho16_eq (D : ℕ) (X : ℝ) : rho16 D X = rhoP D X * (1 + u16) + u16 := by
  unfold rho16; ring

/-- `rho16` is nonnegative when `xi` is. -/
theorem rho16_nonneg_of {D : ℕ} {X : ℝ} (h0 : 0 ≤ xi D X) : 0 ≤ rho16 D X := by
  rw [rho16_eq]
  have := rhoP_nonneg_of h0
  have : (0 : ℝ) ≤ u16 := by unfold u16; norm_num
  positivity

/-- `rho16` is bounded affinely in terms of a bound on `xi`. -/
theorem rho16_aff {D : ℕ} {X xm : ℝ} (h0 : 0 ≤ xi D X) (h1 : xi D X ≤ xm) :
    rho16 D X ≤ (xi D X * (1 + xm + q22 * (1 + xm)) + q22) * (1 + u16) + u16 := by
  rw [rho16_eq]
  have hp := rhoP_aff h0 h1
  have : (0 : ℝ) ≤ 1 + u16 := by unfold u16; norm_num
  nlinarith [mul_le_mul_of_nonneg_right hp this]

/-- `rbarA` is bounded affinely in terms of a bound on `xi`. -/
theorem rbarA_aff {D : ℕ} {X xm : ℝ} (h0 : 0 ≤ xi D X) (h1 : xi D X ≤ xm) :
    rbarA D X ≤ ((xi D X * (1 + xm + q22 * (1 + xm)) + q22) * (1 + u16) + u16) * (1 + r64) := by
  unfold rbarA
  exact mul_le_mul_of_nonneg_right (rho16_aff h0 h1) (by unfold r64; norm_num)

/-- `rbarA` is nonnegative when `xi` is. -/
theorem rbarA_nonneg_of {D : ℕ} {X : ℝ} (h0 : 0 ≤ xi D X) : 0 ≤ rbarA D X := by
  unfold rbarA
  have := rho16_nonneg_of (X := X) h0
  have : (0 : ℝ) ≤ 1 + r64 := by unfold r64; norm_num
  positivity

/-- `rhoS` is nonnegative. -/
theorem rhoS_nonneg : (0 : ℝ) ≤ rhoS := by
  unfold rhoS u32
  have : (1 : ℝ) ≤ (1 + 1 / 2 ^ 24) ^ 6 := one_le_pow₀ (by norm_num)
  linarith

/-- Unfolded form of `rbarL`. -/
theorem rbarL_eq (D : ℕ) (X : ℝ) :
    rbarL D X = (rhoP D X * (1 + rhoS) + rhoS) * (1 + 3 * u32) := by
  unfold rbarL; ring

/-- `rbarL` is bounded affinely in terms of a bound on `xi`. -/
theorem rbarL_aff {D : ℕ} {X xm : ℝ} (h0 : 0 ≤ xi D X) (h1 : xi D X ≤ xm) :
    rbarL D X ≤ ((xi D X * (1 + xm + q22 * (1 + xm)) + q22) * (1 + rhoS) + rhoS) *
      (1 + 3 * u32) := by
  rw [rbarL_eq]
  have hp := rhoP_aff h0 h1
  have : (0 : ℝ) ≤ 1 + rhoS := by linarith [rhoS_nonneg]
  have h5 : (0 : ℝ) ≤ 1 + 3 * u32 := by unfold u32; norm_num
  exact mul_le_mul_of_nonneg_right (by nlinarith [mul_le_mul_of_nonneg_right hp this]) h5

/-- `rbarL` is nonnegative when `xi` is. -/
theorem rbarL_nonneg_of {D : ℕ} {X : ℝ} (h0 : 0 ≤ xi D X) : 0 ≤ rbarL D X := by
  rw [rbarL_eq]
  have := rhoP_nonneg_of h0
  have := rhoS_nonneg
  have : (0 : ℝ) ≤ 1 + 3 * u32 := by unfold u32; norm_num
  positivity

/-- `EA` is monotone on nonnegative arguments. -/
theorem EA_mono {t a : ℝ} (h0 : 0 ≤ t) (h : t ≤ a) : EA t ≤ EA a := by
  unfold EA
  nlinarith [mul_self_le_mul_self h0 h]

/-- `EA` is nonnegative on nonnegative arguments. -/
theorem EA_nonneg {t : ℝ} (h0 : 0 ≤ t) : 0 ≤ EA t := by
  unfold EA; positivity

/-- `lamL` is bounded affinely in terms of bounds on `xi` and the maximum. -/
theorem lamL_aff {D : ℕ} {T X xm lm : ℝ} (hx0 : 0 ≤ xi D X) (hxm : xi D X ≤ xm)
    (hxm1 : xm < 1) (hT : 0 ≤ T) (hL0 : 0 ≤ ThL T X) (hLm : ThL T X ≤ lm) :
    lamL D T X ≤ EA lm * (ThL T X + rbarL D X) + EA lm * etaAL * T / (1 - xm) := by
  unfold lamL
  have hw : 1 - xm ≤ wlow D X := by unfold wlow; linarith
  have hw0 : 0 < 1 - xm := by linarith
  have hE := EA_mono hL0 hLm
  have hE0 : 0 ≤ EA (ThL T X) := EA_nonneg hL0
  have heta : 0 ≤ etaAL := by unfold etaAL; norm_num
  have hR := rbarL_nonneg_of (X := X) hx0
  have hE1 : 0 ≤ EA lm := le_trans hE0 hE
  apply add_le_add
  · exact mul_le_mul hE (by linarith) (by linarith) hE1
  · apply div_le_div₀ (by positivity) _ hw0 hw
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hE heta) hT


/-- `lamL` is nonnegative under the `xi` bounds. -/
theorem lamL_nonneg_of {D : ℕ} {T X xm : ℝ} (hx0 : 0 ≤ xi D X) (hxm : xi D X ≤ xm)
    (hxm1 : xm < 1) (hT : 0 ≤ T) (hL0 : 0 ≤ ThL T X) : 0 ≤ lamL D T X := by
  unfold lamL
  have hw : 0 < wlow D X := by unfold wlow; linarith
  have hE0 : 0 ≤ EA (ThL T X) := EA_nonneg hL0
  have heta : 0 ≤ etaAL := by unfold etaAL; norm_num
  have hR := rbarL_nonneg_of (X := X) hx0
  positivity

/-- `tolO` is bounded affinely in terms of the `xi`, `a` and `l` bounds. -/
theorem tolO_aff {D : ℕ} {T X xm am lm lmax : ℝ} (hx0 : 0 ≤ xi D X) (hxm : xi D X ≤ xm)
    (hxm1 : xm < 1) (hT : 0 ≤ T) (hA0 : 0 ≤ ThA T X) (hAm : ThA T X ≤ am)
    (hL0 : 0 ≤ ThL T X) (hLm : ThL T X ≤ lm) (hlam : lamL D T X ≤ lmax) (hlmax : lmax < 1) :
    tolO D T X * (1 - lmax) ≤ EA am * (ThA T X + rbarA D X) +
      (EA lm * (ThL T X + rbarL D X) + EA lm * etaAL * T / (1 - xm)) := by
  have hl0 := lamL_nonneg_of hx0 hxm hxm1 hT hL0
  have hla := lamL_aff hx0 hxm hxm1 hT hL0 hLm
  have hd : 0 < 1 - lamL D T X := by linarith
  have hRA := rbarA_nonneg_of (X := X) hx0
  have hE := EA_mono hA0 hAm
  have hE0 := EA_nonneg hA0
  have hn1 : EA (ThA T X) * (ThA T X + rbarA D X) ≤ EA am * (ThA T X + rbarA D X) :=
    mul_le_mul_of_nonneg_right hE (by linarith)
  have hn0 : 0 ≤ EA (ThA T X) * (ThA T X + rbarA D X) + lamL D T X := by positivity
  unfold tolO
  calc (EA (ThA T X) * (ThA T X + rbarA D X) + lamL D T X) / (1 - lamL D T X) * (1 - lmax)
      ≤ (EA (ThA T X) * (ThA T X + rbarA D X) + lamL D T X) / (1 - lamL D T X) *
        (1 - lamL D T X) :=
        mul_le_mul_of_nonneg_left (by linarith) (div_nonneg hn0 hd.le)
    _ = EA (ThA T X) * (ThA T X + rbarA D X) + lamL D T X := div_mul_cancel₀ _ hd.ne'
    _ ≤ _ := by linarith

/-- `tolS` is bounded affinely in terms of the `xi`, `a` and `l` bounds. -/
theorem tolS_aff {D : ℕ} {T X xm am lmax : ℝ} (hx0 : 0 ≤ xi D X) (hxm : xi D X ≤ xm)
    (hxm1 : xm < 1) (hT : 0 ≤ T) (hA0 : 0 ≤ ThA T X) (hAm : ThA T X ≤ am)
    (hL0 : 0 ≤ ThL T X) (hlam : lamL D T X ≤ lmax) (hlmax : lmax < 1) :
    tolS D T X * ((1 - xm) * (1 - lmax)) ≤ EA am * etavA := by
  have hl0 := lamL_nonneg_of hx0 hxm hxm1 hT hL0
  have hE := EA_mono hA0 hAm
  have hE0 := EA_nonneg hA0
  have heta : 0 ≤ etavA := by unfold etavA eta16 r64; norm_num
  have hw : 1 - xm ≤ wlow D X := by unfold wlow; linarith
  have hd : 0 < wlow D X * (1 - lamL D T X) := by
    have : 0 < wlow D X := by linarith
    have : 0 < 1 - lamL D T X := by linarith
    positivity
  have hle : (1 - xm) * (1 - lmax) ≤ wlow D X * (1 - lamL D T X) :=
    mul_le_mul hw (by linarith) (by linarith) (by linarith)
  unfold tolS
  calc EA (ThA T X) * etavA / (wlow D X * (1 - lamL D T X)) * ((1 - xm) * (1 - lmax))
      ≤ EA (ThA T X) * etavA / (wlow D X * (1 - lamL D T X)) *
        (wlow D X * (1 - lamL D T X)) :=
        mul_le_mul_of_nonneg_left hle (div_nonneg (by positivity) hd.le)
    _ = EA (ThA T X) * etavA := div_mul_cancel₀ _ hd.ne'
    _ ≤ _ := mul_le_mul_of_nonneg_right hE heta

/-- `tolF0` is bounded affinely in terms of the `xi`, `a` and `l` bounds. -/
theorem tolF0_aff {D : ℕ} {T X xm am lmax : ℝ} (hx0 : 0 ≤ xi D X) (hxm : xi D X ≤ xm)
    (hxm1 : xm < 1) (hT : 0 ≤ T) (hA0 : 0 ≤ ThA T X) (hAm : ThA T X ≤ am)
    (hL0 : 0 ≤ ThL T X) (hlam : lamL D T X ≤ lmax) (hlmax : lmax < 1)
    (hk : 0 ≤ kapA D X) :
    tolF0 D T X * ((1 - xm) * (1 - lmax)) ≤ EA am * kapA D X + EA am * etaAA * T := by
  have hl0 := lamL_nonneg_of hx0 hxm hxm1 hT hL0
  have hE := EA_mono hA0 hAm
  have hE0 := EA_nonneg hA0
  have heta : 0 ≤ etaAA := by unfold etaAA; norm_num
  have hxm0 : 0 ≤ xm := le_trans hx0 hxm
  have hw : 1 - xm ≤ wlow D X := by unfold wlow; linarith
  have hw1 : wlow D X ≤ 1 := by unfold wlow; linarith
  have hd1 : 0 < 1 - lamL D T X := by linarith
  have hd : 0 < wlow D X * (1 - lamL D T X) := by
    have : 0 < wlow D X := by linarith
    positivity
  have hle : (1 - xm) * (1 - lmax) ≤ wlow D X * (1 - lamL D T X) :=
    mul_le_mul hw (by linarith) (by linarith) (by linarith)
  have hle1 : (1 - xm) * (1 - lmax) ≤ 1 - lamL D T X := by nlinarith
  have hA : EA (ThA T X) * kapA D X / (1 - lamL D T X) * ((1 - xm) * (1 - lmax)) ≤
      EA am * kapA D X := by
    calc EA (ThA T X) * kapA D X / (1 - lamL D T X) * ((1 - xm) * (1 - lmax))
        ≤ EA (ThA T X) * kapA D X / (1 - lamL D T X) * (1 - lamL D T X) :=
          mul_le_mul_of_nonneg_left hle1 (div_nonneg (by positivity) hd1.le)
      _ = EA (ThA T X) * kapA D X := div_mul_cancel₀ _ hd1.ne'
      _ ≤ _ := mul_le_mul_of_nonneg_right hE hk
  have hB : EA (ThA T X) * etaAA * T / (wlow D X * (1 - lamL D T X)) *
      ((1 - xm) * (1 - lmax)) ≤ EA am * etaAA * T := by
    calc EA (ThA T X) * etaAA * T / (wlow D X * (1 - lamL D T X)) * ((1 - xm) * (1 - lmax))
        ≤ EA (ThA T X) * etaAA * T / (wlow D X * (1 - lamL D T X)) *
          (wlow D X * (1 - lamL D T X)) :=
          mul_le_mul_of_nonneg_left hle (div_nonneg (by positivity) hd.le)
      _ = EA (ThA T X) * etaAA * T := div_mul_cancel₀ _ hd.ne'
      _ ≤ _ := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hE heta) hT
  unfold tolF0
  nlinarith [hA, hB]

/-- `tolO` is nonnegative under the `xi` bounds. -/
theorem tolO_nonneg_of {D : ℕ} {T X xm : ℝ} (hx0 : 0 ≤ xi D X) (hxm : xi D X ≤ xm)
    (hxm1 : xm < 1) (hT : 0 ≤ T) (hA0 : 0 ≤ ThA T X) (hL0 : 0 ≤ ThL T X)
    (hlam : lamL D T X ≤ 1 / 2) : 0 ≤ tolO D T X := by
  have hl0 := lamL_nonneg_of hx0 hxm hxm1 hT hL0
  have hRA := rbarA_nonneg_of (X := X) hx0
  have hE0 := EA_nonneg hA0
  unfold tolO
  have : 0 < 1 - lamL D T X := by linarith
  positivity

/-- `tolS` is nonnegative under the `xi` bounds. -/
theorem tolS_nonneg_of {D : ℕ} {T X xm : ℝ} (hx0 : 0 ≤ xi D X) (hxm : xi D X ≤ xm)
    (hxm1 : xm < 1) (hT : 0 ≤ T) (hA0 : 0 ≤ ThA T X) (hL0 : 0 ≤ ThL T X)
    (hlam : lamL D T X ≤ 1 / 2) : 0 ≤ tolS D T X := by
  have hl0 := lamL_nonneg_of hx0 hxm hxm1 hT hL0
  have hE0 := EA_nonneg hA0
  have heta : 0 ≤ etavA := by unfold etavA eta16 r64; norm_num
  have hw : 0 < wlow D X := by unfold wlow; linarith
  unfold tolS
  have : 0 < 1 - lamL D T X := by linarith
  positivity

/-- `tolF0` is nonnegative under the `xi` bounds. -/
theorem tolF0_nonneg_of {D : ℕ} {T X xm : ℝ} (hx0 : 0 ≤ xi D X) (hxm : xi D X ≤ xm)
    (hxm1 : xm < 1) (hT : 0 ≤ T) (hA0 : 0 ≤ ThA T X) (hL0 : 0 ≤ ThL T X)
    (hlam : lamL D T X ≤ 1 / 2) (hk : 0 ≤ kapA D X) : 0 ≤ tolF0 D T X := by
  have hl0 := lamL_nonneg_of hx0 hxm hxm1 hT hL0
  have hE0 := EA_nonneg hA0
  have heta : 0 ≤ etaAA := by unfold etaAA; norm_num
  have hw : 0 < wlow D X := by unfold wlow; linarith
  unfold tolF0
  have : 0 < 1 - lamL D T X := by linarith
  positivity

/-- Ratio bound relating `lamL` to its maximum under the `xi` bounds. -/
theorem lamRatio {D : ℕ} {T X xm lm lmax : ℝ} (hx0 : 0 ≤ xi D X) (hxm : xi D X ≤ xm)
    (hxm1 : xm < 1) (hT : 0 ≤ T) (hL0 : 0 ≤ ThL T X) (hLm : ThL T X ≤ lm)
    (hlam : lamL D T X ≤ lmax) (hlmax : lmax < 1) :
    lamL D T X / (1 - lamL D T X) * (1 - lmax) ≤
      EA lm * (ThL T X + rbarL D X) + EA lm * etaAL * T / (1 - xm) := by
  have hl0 := lamL_nonneg_of hx0 hxm hxm1 hT hL0
  have hla := lamL_aff hx0 hxm hxm1 hT hL0 hLm
  have hd : 0 < 1 - lamL D T X := by linarith
  calc lamL D T X / (1 - lamL D T X) * (1 - lmax)
      ≤ lamL D T X / (1 - lamL D T X) * (1 - lamL D T X) :=
        mul_le_mul_of_nonneg_left (by linarith) (div_nonneg hl0 hd.le)
    _ = lamL D T X := div_mul_cancel₀ _ hd.ne'
    _ ≤ _ := hla


/-! ### The uniform bounds on the box -/

theorem xi_le_unif {D : ℕ} {X : ℝ} (h : Par D X) : xi D X ≤ 3 / 32 := by
  refine (xi_mono (D := D) h.hX.2).trans ?_
  rcases h.hD with rfl | rfl | rfl <;> norm_num [xi, rD]

/-- `ThA` is at most 17/1000 on the covered head sizes and scores. -/
theorem thA_unif {D : ℕ} {T X : ℝ} (h : Box D T X) : ThA T X ≤ 17 / 1000 := by
  have := h.hT; have := h.hX
  unfold ThA EPSA CD u32 Bm
  linarith

/-- `ThA` is nonnegative on the covered head sizes and scores. -/
theorem thA_nn {D : ℕ} {T X : ℝ} (h : Box D T X) : 0 ≤ ThA T X := by
  have := h.hT; have := h.hX
  unfold ThA EPSA CD u32 Bm
  linarith

/-- `ThL` is at most 6/10000 on the covered head sizes and scores. -/
theorem thL_unif {D : ℕ} {T X : ℝ} (h : Box D T X) : ThL T X ≤ 6 / 10000 := by
  have := h.hT; have := h.hX
  unfold ThL EPSL CD u32 Bm
  linarith

/-- `ThL` is nonnegative on the covered head sizes and scores. -/
theorem thL_nn {D : ℕ} {T X : ℝ} (h : Box D T X) : 0 ≤ ThL T X := by
  have := h.hT; have := h.hX
  unfold ThL EPSL CD u32 Bm
  linarith

/-- `rho16` is at most 104/1000 on the covered head sizes and scores. -/
theorem rho16_unif {D : ℕ} {X : ℝ} (h : Par D X) : rho16 D X ≤ 104 / 1000 := by
  have h0 : 0 ≤ xi D X := xi_nonneg_of h.hX.1
  have h1 := xi_le_unif h
  have h2 := rho16_aff h0 h1
  simp only [q22, u16] at h2 ⊢
  linarith

/-- `rbarA` is at most 104/1000 on the covered head sizes and scores. -/
theorem rbarA_unif {D : ℕ} {X : ℝ} (h : Par D X) : rbarA D X ≤ 104 / 1000 := by
  have h0 : 0 ≤ xi D X := xi_nonneg_of h.hX.1
  have h1 := xi_le_unif h
  have h2 := rbarA_aff h0 h1
  simp only [q22, u16, r64] at h2 ⊢
  linarith

/-- `rbarL` is at most 103/1000 on the covered head sizes and scores. -/
theorem rbarL_unif {D : ℕ} {X : ℝ} (h : Par D X) : rbarL D X ≤ 103 / 1000 := by
  have h0 : 0 ≤ xi D X := xi_nonneg_of h.hX.1
  have h1 := xi_le_unif h
  have h2 := rbarL_aff h0 h1
  have h3 : rhoS ≤ 1 / 10 ^ 6 := by
    unfold rhoS u32
    norm_num
  simp only [q22, u32] at h2 ⊢
  nlinarith

/-- `lamL` is at most 11/100 on the covered head sizes and scores. -/
theorem lamL_unif {D : ℕ} {T X : ℝ} (h : Box D T X) : lamL D T X ≤ 11 / 100 := by
  have h0 : 0 ≤ xi D X := xi_nonneg_of h.hX.1
  have h1 := xi_le_unif h.par
  have h2 := lamL_aff (T := T) (lm := 6 / 10000) h0 h1 (by norm_num) (by linarith [h.hT.1])
    (thL_nn h) (thL_unif h)
  have h3 := rbarL_unif h.par
  have h4 := thL_unif h
  have h5 := h.hT.2
  simp only [EA, etaAL] at h2
  norm_num at h2
  linarith

/-! ### The weights' errors on the head sizes and scores covered -/

theorem Par.xi_nonneg {D : ℕ} {X : ℝ} (h : Par D X) : 0 ≤ xi D X := xi_nonneg_of h.hX.1
/-- `xi` is at most 1/10 on the covered head sizes and scores. -/
theorem Par.xi_le {D : ℕ} {X : ℝ} (h : Par D X) : xi D X ≤ 1 / 10 := by
  linarith [xi_le_unif h]
/-- `rhoP` is nonnegative on the covered head sizes and scores. -/
theorem Par.rhoP_nonneg {D : ℕ} {X : ℝ} (h : Par D X) : 0 ≤ rhoP D X :=
  rhoP_nonneg_of h.xi_nonneg
/-- `rhoP` is at most 1/5 on the covered head sizes and scores. -/
theorem Par.rhoP_le {D : ℕ} {X : ℝ} (h : Par D X) : rhoP D X ≤ 1 / 5 := by
  have h2 := rhoP_aff h.xi_nonneg (xi_le_unif h)
  have h1 := xi_le_unif h
  simp only [q22] at h2 ⊢
  norm_num at h2
  linarith
/-- `rho16` is nonnegative on the covered head sizes and scores. -/
theorem Par.rho16_nonneg {D : ℕ} {X : ℝ} (h : Par D X) : 0 ≤ rho16 D X :=
  rho16_nonneg_of h.xi_nonneg
/-- `rho16` is at most 1/5 on the covered head sizes and scores. -/
theorem Par.rho16_le {D : ℕ} {X : ℝ} (h : Par D X) : rho16 D X ≤ 1 / 5 := by
  linarith [rho16_unif h]
/-- `rbarL` is nonnegative on the covered head sizes and scores. -/
theorem Par.rbarL_nonneg {D : ℕ} {X : ℝ} (h : Par D X) : 0 ≤ rbarL D X :=
  rbarL_nonneg_of h.xi_nonneg
/-- `rbarL` is at most 1/5 on the covered head sizes and scores. -/
theorem Par.rbarL_le {D : ℕ} {X : ℝ} (h : Par D X) : rbarL D X ≤ 1 / 5 := by
  linarith [rbarL_unif h]
/-- `rbarA` is nonnegative on the covered head sizes and scores. -/
theorem Par.rbarA_nonneg {D : ℕ} {X : ℝ} (h : Par D X) : 0 ≤ rbarA D X :=
  rbarA_nonneg_of h.xi_nonneg
/-- `rbarA` is at most 1/5 on the covered head sizes and scores. -/
theorem Par.rbarA_le {D : ℕ} {X : ℝ} (h : Par D X) : rbarA D X ≤ 1 / 5 := by
  linarith [rbarA_unif h]
/-- `kapA` is nonnegative on the covered head sizes and scores. -/
theorem Par.kapA_nonneg {D : ℕ} {X : ℝ} (h : Par D X) : 0 ≤ kapA D X := by
  have := h.rho16_nonneg
  unfold kapA r64
  positivity
/-- `kapA` is at most 2/10^9 on the covered head sizes and scores. -/
theorem kapA_tight {D : ℕ} {X : ℝ} (h : Par D X) : kapA D X ≤ 2 / 10 ^ 9 := by
  have := rho16_unif h
  unfold kapA r64
  norm_num
  linarith
/-- `kapA` is at most 1/2^10 on the covered head sizes and scores. -/
theorem Par.kapA_le {D : ℕ} {X : ℝ} (h : Par D X) : kapA D X ≤ 1 / 2 ^ 10 := by
  have := kapA_tight h
  norm_num at this ⊢
  linarith
/-- `wlow` is at least 9/10 on the covered head sizes and scores. -/
theorem Par.wlow_ge {D : ℕ} {X : ℝ} (h : Par D X) : 9 / 10 ≤ wlow D X := by
  unfold wlow
  linarith [xi_le_unif h]

/-- `ThA` is at most 1/32 on the covered box. -/
theorem Box.thA_le {D : ℕ} {T X : ℝ} (h : Box D T X) : ThA T X ≤ 1 / 32 := by
  linarith [thA_unif h]
/-- `ThL` is at most 1/32 on the covered box. -/
theorem Box.thL_le {D : ℕ} {T X : ℝ} (h : Box D T X) : ThL T X ≤ 1 / 32 := by
  linarith [thL_unif h]
/-- `lamL` is nonnegative on the covered box. -/
theorem Box.lamL_nonneg {D : ℕ} {T X : ℝ} (h : Box D T X) : 0 ≤ lamL D T X :=
  lamL_nonneg_of h.par.xi_nonneg (xi_le_unif h.par) (by norm_num) (by linarith [h.hT.1])
    (thL_nn h)
/-- `lamL` is at most 1/4 on the covered box. -/
theorem Box.lamL_le {D : ℕ} {T X : ℝ} (h : Box D T X) : lamL D T X ≤ 1 / 4 := by
  linarith [lamL_unif h]

/-- `tolO` is at most 1/2 on the covered box. -/
theorem tolO_unif {D : ℕ} {T X : ℝ} (h : Box D T X) : tolO D T X ≤ 1 / 2 := by
  have h2 := tolO_aff (T := T) (am := 17 / 1000) (lm := 6 / 10000) (lmax := 11 / 100)
    h.par.xi_nonneg (xi_le_unif h.par) (by norm_num) (by linarith [h.hT.1]) (thA_nn h)
    (thA_unif h) (thL_nn h) (thL_unif h) (lamL_unif h) (by norm_num)
  have h3 := rbarL_unif h.par
  have h4 := rbarA_unif h.par
  have h5 := thA_unif h
  have h6 := thL_unif h
  have h7 := h.hT.2
  simp only [EA, etaAL] at h2
  norm_num at h2
  linarith
/-- `tolO` is at most 1/2 on the covered box. -/
theorem Box.tolO_le {D : ℕ} {T X : ℝ} (h : Box D T X) : tolO D T X ≤ 1 / 2 := tolO_unif h
/-- `tolO` is nonnegative on the covered box. -/
theorem Box.tolO_nonneg {D : ℕ} {T X : ℝ} (h : Box D T X) : 0 ≤ tolO D T X :=
  tolO_nonneg_of h.par.xi_nonneg (xi_le_unif h.par) (by norm_num) (by linarith [h.hT.1])
    (thA_nn h) (thL_nn h) (by linarith [lamL_unif h])

/-- `tolS` is at most 1/10^7 on the covered box. -/
theorem tolS_unif {D : ℕ} {T X : ℝ} (h : Box D T X) : tolS D T X ≤ 1 / 10 ^ 7 := by
  have h2 := tolS_aff (T := T) (am := 17 / 1000) (lmax := 11 / 100)
    h.par.xi_nonneg (xi_le_unif h.par) (by norm_num) (by linarith [h.hT.1]) (thA_nn h)
    (thA_unif h) (thL_nn h) (lamL_unif h) (by norm_num)
  simp only [EA, etavA, eta16, r64] at h2
  norm_num at h2
  linarith
/-- `tolS` is at most 1/2^22 on the covered box. -/
theorem Box.tolS_le {D : ℕ} {T X : ℝ} (h : Box D T X) : tolS D T X ≤ 1 / 2 ^ 22 := by
  have := tolS_unif h
  norm_num at this ⊢
  linarith
/-- `tolS` is nonnegative on the covered box. -/
theorem Box.tolS_nonneg {D : ℕ} {T X : ℝ} (h : Box D T X) : 0 ≤ tolS D T X :=
  tolS_nonneg_of h.par.xi_nonneg (xi_le_unif h.par) (by norm_num) (by linarith [h.hT.1])
    (thA_nn h) (thL_nn h) (by linarith [lamL_unif h])

/-- `tolF0` is at most 2/10^8 on the covered box. -/
theorem tolF0_unif {D : ℕ} {T X : ℝ} (h : Box D T X) : tolF0 D T X ≤ 2 / 10 ^ 8 := by
  have h2 := tolF0_aff (T := T) (am := 17 / 1000) (lmax := 11 / 100)
    h.par.xi_nonneg (xi_le_unif h.par) (by norm_num) (by linarith [h.hT.1]) (thA_nn h)
    (thA_unif h) (thL_nn h) (lamL_unif h) (by norm_num) h.par.kapA_nonneg
  have h3 := kapA_tight h.par
  have h7 := h.hT.2
  simp only [EA, etaAA] at h2
  norm_num at h2
  linarith
/-- `tolF0` is at most 1 on the covered box. -/
theorem Box.tolF0_le {D : ℕ} {T X : ℝ} (h : Box D T X) : tolF0 D T X ≤ 1 := by
  linarith [tolF0_unif h]
/-- `tolF0` is nonnegative on the covered box. -/
theorem Box.tolF0_nonneg {D : ℕ} {T X : ℝ} (h : Box D T X) : 0 ≤ tolF0 D T X :=
  tolF0_nonneg_of h.par.xi_nonneg (xi_le_unif h.par) (by norm_num) (by linarith [h.hT.1])
    (thA_nn h) (thL_nn h) (by linarith [lamL_unif h]) h.par.kapA_nonneg


/-- The scale factor `1 + xm + q22 * (1 + xm)` of the relaxed recurrence. -/
noncomputable def rlxk (xm : ℝ) : ℝ := 1 + xm + q22 * (1 + xm)
/-- Relaxed one-step bound on the accumulator path, with rounding terms `u16` and `r64`. -/
noncomputable def rlxRA (xm x : ℝ) : ℝ := ((x * rlxk xm + q22) * (1 + u16) + u16) * (1 + r64)
/-- Relaxed one-step bound on the row-sum path, with rounding terms `rhoS` and `u32`. -/
noncomputable def rlxRL (xm x : ℝ) : ℝ := ((x * rlxk xm + q22) * (1 + rhoS) + rhoS) * (1 + 3 * u32)

/-- `rlxRL` is monotone in its second argument for nonnegative first argument. -/
theorem rlxRL_mono {xm x y : ℝ} (hxm : 0 ≤ xm) (h : x ≤ y) : rlxRL xm x ≤ rlxRL xm y := by
  unfold rlxRL rlxk
  have := q22_pos
  have := rhoS_nonneg
  have h5 : (0 : ℝ) ≤ 1 + 3 * u32 := by unfold u32; norm_num
  have hk : 0 ≤ 1 + xm + q22 * (1 + xm) := by positivity
  have hk' := mul_le_mul_of_nonneg_right h hk
  have h6 := mul_le_mul_of_nonneg_right
    (by linarith : x * (1 + xm + q22 * (1 + xm)) + q22 ≤ y * (1 + xm + q22 * (1 + xm)) + q22)
    (by linarith : (0 : ℝ) ≤ 1 + rhoS)
  exact mul_le_mul_of_nonneg_right (by linarith) h5

/-- `lamL` is bounded using an explicit bound `xm` on `xi` with `xm < 1`. -/
theorem lamL_xm {D : ℕ} {T X xm lmax : ℝ} (h : Box D T X) (hxm : xi D X ≤ xm) (hxm1 : xm < 1)
    (hlam : EA (6 / 10000) * (6 / 10000 + rlxRL xm xm) + EA (6 / 10000) * etaAL * 1024 / (1 - xm)
      ≤ lmax) : lamL D T X ≤ lmax := by
  have h0 := h.par.xi_nonneg
  have hxm0 : 0 ≤ xm := le_trans h0 hxm
  have hT0 : 0 ≤ T := by linarith [h.hT.1]
  have h1 := lamL_aff (T := T) (lm := 6 / 10000) h0 hxm hxm1 hT0 (thL_nn h) (thL_unif h)
  have hRL : rbarL D X ≤ rlxRL xm xm := (rbarL_aff h0 hxm).trans (rlxRL_mono hxm0 hxm)
  have hE := EA_nonneg (t := 6 / 10000) (by norm_num)
  have heta : 0 ≤ etaAL := by unfold etaAL; norm_num
  have h2 : EA (6 / 10000) * (ThL T X + rbarL D X) ≤ EA (6 / 10000) * (6 / 10000 + rlxRL xm xm) :=
    mul_le_mul_of_nonneg_left (by linarith [thL_unif h]) hE
  have h3 : EA (6 / 10000) * etaAL * T / (1 - xm) ≤ EA (6 / 10000) * etaAL * 1024 / (1 - xm) :=
    div_le_div₀ (by positivity) (mul_le_mul_of_nonneg_left h.hT.2 (by positivity))
      (by linarith) le_rfl
  linarith

/-- Finite-parameter bound on the output tolerance in terms of `xm` and `lmax`. -/
theorem relax_o_fin {D : ℕ} {T X N xm lmax : ℝ} (h : Box D T X) (hN : 64 * T ≤ N)
    (hxm : xi D X ≤ xm) (hxm1 : xm < 1) (hlmax : lmax < 1)
    (hlam : EA (6 / 10000) * (6 / 10000 + rlxRL xm xm) + EA (6 / 10000) * etaAL * 1024 / (1 - xm)
      ≤ lmax)
    (hfin : (1 + lastRel) * ((EA (17 / 1000) * (ThA T X + rlxRA xm (xi D X)) +
        (EA (6 / 10000) * (ThL T X + rlxRL xm (xi D X)) +
          EA (6 / 10000) * etaAL * T / (1 - xm))) / (1 - lmax)) + lastRel +
        N * (EA (17 / 1000) * etavA / ((1 - xm) * (1 - lmax)) * (1 + lastRel))
      ≤ epsO' D N X) :
    tolO2 D T X + N * tolS2 D T X ≤ epsO' D N X := by
  have h0 := h.par.xi_nonneg
  have hxm0 : 0 ≤ xm := le_trans h0 hxm
  have hT0 : 0 ≤ T := by linarith [h.hT.1]
  have hlamL := lamL_xm h hxm hxm1 hlam
  have hO := tolO_aff (T := T) (am := 17 / 1000) (lm := 6 / 10000) (lmax := lmax)
    h0 hxm hxm1 hT0 (thA_nn h) (thA_unif h) (thL_nn h) (thL_unif h) hlamL hlmax
  have hS := tolS_aff (T := T) (am := 17 / 1000) (lmax := lmax)
    h0 hxm hxm1 hT0 (thA_nn h) (thA_unif h) (thL_nn h) hlamL hlmax
  have hRA : rbarA D X ≤ rlxRA xm (xi D X) := rbarA_aff h0 hxm
  have hRL : rbarL D X ≤ rlxRL xm (xi D X) := rbarL_aff h0 hxm
  have hE1 := EA_nonneg (t := 17 / 1000) (by norm_num)
  have hE2 := EA_nonneg (t := 6 / 10000) (by norm_num)
  have hd : 0 < 1 - lmax := by linarith
  have hd2 : 0 < (1 - xm) * (1 - lmax) := by
    have : 0 < 1 - xm := by linarith
    positivity
  have hO2 : tolO D T X ≤ (EA (17 / 1000) * (ThA T X + rlxRA xm (xi D X)) +
        (EA (6 / 10000) * (ThL T X + rlxRL xm (xi D X)) +
          EA (6 / 10000) * etaAL * T / (1 - xm))) / (1 - lmax) := by
    rw [le_div_iff₀ hd]
    have a1 : EA (17 / 1000) * (ThA T X + rbarA D X) ≤
        EA (17 / 1000) * (ThA T X + rlxRA xm (xi D X)) :=
      mul_le_mul_of_nonneg_left (by linarith) hE1
    have a2 : EA (6 / 10000) * (ThL T X + rbarL D X) ≤
        EA (6 / 10000) * (ThL T X + rlxRL xm (xi D X)) :=
      mul_le_mul_of_nonneg_left (by linarith) hE2
    linarith
  have hS2 : tolS D T X ≤ EA (17 / 1000) * etavA / ((1 - xm) * (1 - lmax)) := by
    rw [le_div_iff₀ hd2]; exact hS
  have hN0 : 0 ≤ N := by linarith
  have hlr : 0 ≤ lastRel := by unfold lastRel u16 q22; norm_num
  have b1 : (1 + lastRel) * tolO D T X ≤ (1 + lastRel) * ((EA (17 / 1000) *
      (ThA T X + rlxRA xm (xi D X)) + (EA (6 / 10000) * (ThL T X + rlxRL xm (xi D X)) +
        EA (6 / 10000) * etaAL * T / (1 - xm))) / (1 - lmax)) :=
    mul_le_mul_of_nonneg_left hO2 (by linarith)
  have b2 : N * (tolS D T X * (1 + lastRel)) ≤
      N * (EA (17 / 1000) * etavA / ((1 - xm) * (1 - lmax)) * (1 + lastRel)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hS2 (by linarith)) hN0
  unfold tolO2 tolS2
  nlinarith

/-- Finite-parameter bound on the statistics tolerance in terms of `xm` and `lmax`. -/
theorem relax_m_fin {D : ℕ} {T X xm lmax : ℝ} (h : Box D T X)
    (hxm : xi D X ≤ xm) (hxm1 : xm < 1) (hlmax : lmax < 1)
    (hlam : EA (6 / 10000) * (6 / 10000 + rlxRL xm xm) + EA (6 / 10000) * etaAL * 1024 / (1 - xm)
      ≤ lmax)
    (hfin : 14427 / 10000 * ((EA (6 / 10000) * (ThL T X + rlxRL xm (xi D X)) +
          EA (6 / 10000) * etaAL * T / (1 - xm)) / (1 - lmax)) + 18 * q22 + 1 / 2 ^ 149 +
        u32 * (Bm X + 19) + 1 / 2 ^ 150 ≤ epsM' D X) :
    tolM D T X ≤ epsM' D X := by
  have h0 := h.par.xi_nonneg
  have hT0 : 0 ≤ T := by linarith [h.hT.1]
  have hlamL := lamL_xm h hxm hxm1 hlam
  have hl0 := Box.lamL_nonneg h
  have hR := lamRatio (T := T) (lm := 6 / 10000) h0 hxm hxm1 hT0 (thL_nn h) (thL_unif h)
    hlamL hlmax
  have hRL : rbarL D X ≤ rlxRL xm (xi D X) := rbarL_aff h0 hxm
  have hE2 := EA_nonneg (t := 6 / 10000) (by norm_num)
  have hd : 0 < 1 - lmax := by linarith
  have hO2 : lamL D T X / (1 - lamL D T X) ≤ (EA (6 / 10000) * (ThL T X + rlxRL xm (xi D X)) +
          EA (6 / 10000) * etaAL * T / (1 - xm)) / (1 - lmax) := by
    rw [le_div_iff₀ hd]
    have a2 : EA (6 / 10000) * (ThL T X + rbarL D X) ≤
        EA (6 / 10000) * (ThL T X + rlxRL xm (xi D X)) :=
      mul_le_mul_of_nonneg_left (by linarith) hE2
    linarith
  unfold tolM
  linarith

/-- The output tolerance is bounded on the covered box for context lengths up to 2^16. -/
theorem relax_o {D : ℕ} {T X N : ℝ} (h : Box D T X) (hN : 64 * T ≤ N) (hN2 : N ≤ 2 ^ 16) :
    tolO2 D T X + N * tolS2 D T X ≤ epsO' D N X := by
  have _ := hN2
  have hT := h.hT
  have hX := h.hX
  have hD := h.hD
  rcases hD with rfl | rfl | rfl <;> rcases le_total X 16 with hX16 | hX16
  · -- D = 64, X ≤ 16
    refine relax_o_fin h hN (xm := 1 / 2500)
      (lmax := 11 / 10000) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [lastRel, EA, ThA, ThL, xi, rD, rlxRA, rlxRL, rlxk, etavA, eta16, r64, epsO',
        q22, u16, u32, rhoS, EPSA, EPSL, CD, Bm, etaAL]
      norm_num
      linarith
  · -- D = 64, 16 ≤ X
    refine relax_o_fin h hN (xm := 243 / 10000)
      (lmax := 16 / 625) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [lastRel, EA, ThA, ThL, xi, rD, rlxRA, rlxRL, rlxk, etavA, eta16, r64, epsO',
        q22, u16, u32, rhoS, EPSA, EPSL, CD, Bm, etaAL]
      norm_num
      linarith
  · -- D = 128, X ≤ 16
    refine relax_o_fin h hN (xm := 3 / 4000)
      (lmax := 7 / 5000) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [lastRel, EA, ThA, ThL, xi, rD, rlxRA, rlxRL, rlxk, etavA, eta16, r64, epsO',
        q22, u16, u32, rhoS, EPSA, EPSL, CD, Bm, etaAL]
      norm_num
      linarith
  · -- D = 128, 16 ≤ X
    refine relax_o_fin h hN (xm := 47 / 1000)
      (lmax := 499 / 10000) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [lastRel, EA, ThA, ThL, xi, rD, rlxRA, rlxRL, rlxk, etavA, eta16, r64, epsO',
        q22, u16, u32, rhoS, EPSA, EPSL, CD, Bm, etaAL]
      norm_num
      linarith
  · -- D = 256, X ≤ 16
    refine relax_o_fin h hN (xm := 3 / 2000)
      (lmax := 11 / 5000) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [lastRel, EA, ThA, ThL, xi, rD, rlxRA, rlxRL, rlxk, etavA, eta16, r64, epsO',
        q22, u16, u32, rhoS, EPSA, EPSL, CD, Bm, etaAL]
      norm_num
      linarith
  · -- D = 256, 16 ≤ X
    refine relax_o_fin h hN (xm := 23 / 250)
      (lmax := 253 / 2500) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [lastRel, EA, ThA, ThL, xi, rD, rlxRA, rlxRL, rlxk, etavA, eta16, r64, epsO',
        q22, u16, u32, rhoS, EPSA, EPSL, CD, Bm, etaAL]
      norm_num
      linarith

/-- The final tolerance `tolF2` is at most 1/2^24 on the covered box. -/
theorem relax_f {D : ℕ} {T X : ℝ} (h : Box D T X) : tolF2 D T X ≤ 1 / 2 ^ 24 := by
  have h1 := tolF0_unif h
  unfold tolF2 lastRel u16 q22 e16
  norm_num
  norm_num at h1
  linarith

/-- The statistics tolerance `tolM` is at most `epsM'` on the covered box. -/
theorem relax_m {D : ℕ} {T X : ℝ} (h : Box D T X) : tolM D T X ≤ epsM' D X := by
  have hT := h.hT
  have hX := h.hX
  have hD := h.hD
  rcases hD with rfl | rfl | rfl <;> rcases le_total X 16 with hX16 | hX16
  · -- D = 64, X ≤ 16
    refine relax_m_fin h (xm := 1 / 2500)
      (lmax := 11 / 10000) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [EA, ThL, xi, rD, rlxRL, rlxk, epsM', q22, u32, rhoS, CD, Bm, EPSL, etaAL]
      norm_num
      linarith
  · -- D = 64, 16 ≤ X
    refine relax_m_fin h (xm := 243 / 10000)
      (lmax := 16 / 625) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [EA, ThL, xi, rD, rlxRL, rlxk, epsM', q22, u32, rhoS, CD, Bm, EPSL, etaAL]
      norm_num
      linarith
  · -- D = 128, X ≤ 16
    refine relax_m_fin h (xm := 3 / 4000)
      (lmax := 7 / 5000) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [EA, ThL, xi, rD, rlxRL, rlxk, epsM', q22, u32, rhoS, CD, Bm, EPSL, etaAL]
      norm_num
      linarith
  · -- D = 128, 16 ≤ X
    refine relax_m_fin h (xm := 47 / 1000)
      (lmax := 499 / 10000) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [EA, ThL, xi, rD, rlxRL, rlxk, epsM', q22, u32, rhoS, CD, Bm, EPSL, etaAL]
      norm_num
      linarith
  · -- D = 256, X ≤ 16
    refine relax_m_fin h (xm := 3 / 2000)
      (lmax := 11 / 5000) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [EA, ThL, xi, rD, rlxRL, rlxk, epsM', q22, u32, rhoS, CD, Bm, EPSL, etaAL]
      norm_num
      linarith
  · -- D = 256, 16 ≤ X
    refine relax_m_fin h (xm := 23 / 250)
      (lmax := 253 / 2500) ?_ (by norm_num) (by norm_num) ?_ ?_
    · unfold xi rD
      norm_num
      linarith
    · simp only [EA, etaAL, rlxRL, rlxk, rhoS, q22, u32]
      norm_num
    · simp only [EA, ThL, xi, rD, rlxRL, rlxk, epsM', q22, u32, rhoS, CD, Bm, EPSL, etaAL]
      norm_num
      linarith

end Claims.AttnFwd
