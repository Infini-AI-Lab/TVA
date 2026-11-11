import Proof.NumG
import Proof.NumUtil
import Proof.NumV
import Proof.GlueAux

/-!
# `chunk_delta_h_bf16`: the decay of the state

The device's `exp2(g_last)`, the `float32` product of a state value with it, and the bound on a
state value from its `bfloat16` rounding.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- **The decay `exp2(g_last)`**: finite, `gam = 2^g_last` positive and at most `2^8`, and the
device's value within `errE` of it (`errE ≤ 2^-22·gam + mu`). -/
theorem gt_facts {d : Inst} (hd : d.ok) {P : Platform} (hP : Device.Exp2F32 P) {init : Mem}
    (hV : Vals init d) {n hv t : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    (hG : d.useG = true) {bgl gt : F32}
    (hbgl : rdF init d.g (gTok d n hv (BT * t + cl d n t - 1)) = some bgl)
    (he : P.math .exp2 .binary32 bgl gt) :
    gt.isFinite = true ∧ 0 < gam init d n hv t ∧ gam init d n hv t ≤ 2 ^ 8 ∧
      |gt.toReal - gam init d n hv t| ≤ errE init d n hv t ∧ 0 ≤ errE init d n hv t ∧
      errE init d n hv t ≤ 2 ^ (-22 : ℤ) * gam init d n hv t + mu := by
  have hlast : BT * t + cl d n t - 1 < lenOf d n := by
    have : 0 < cl d n t := by simp only [cl, BT, nch] at *; omega
    simp only [cl, BT] at *; omega
  obtain ⟨gl, hgl, hglf, hglb⟩ := vals_g hd hV hG hn hhv hlast
  rw [hbgl] at hgl
  cases hgl
  have hgl8 := (hV.2.2.2.2.2.1 n hv t hn hhv ht).1
  have e1 : valF init d.g (gIdx d (rowOf d n) (startOf d n + (BT * t + cl d n t - 1)) hv) = bgl.toRat :=
    valF_of_rd hbgl
  have hglast : glast init d n hv t = bgl.toReal := by
    unfold glast gq
    rw [ite_eq_left hG, e1]; rfl
  have hle : bgl.toReal ≤ 8 := by rw [← hglast]; exact hgl8
  obtain ⟨hef, hee⟩ := gate_exp hP hglf (by linarith) he
  have hgam : gam init d n hv t = (2 : ℝ) ^ bgl.toReal := by
    unfold gam; rw [hglast]; rfl
  have hpos : (0 : ℝ) < (2 : ℝ) ^ bgl.toReal := Real.rpow_pos_of_pos (by norm_num) _
  have hErr : errE init d n hv t = 2 ^ (-22 : ℤ) * gam init d n hv t + mu := by
    unfold errE; rw [ite_eq_left hG]
  refine ⟨hef, by rw [hgam]; exact hpos, ?_, ?_, ?_, le_of_eq hErr⟩
  · rw [hgam]
    have := two_rpow_le_zpow (x := bgl.toReal) (n := 8) (by push_cast; exact hle)
    simpa using this
  · rw [hErr, hgam]; exact hee
  · rw [hErr]; unfold mu; have := mul_nonneg (zpow_pos (by norm_num : (0:ℝ) < 2) (-22 : ℤ)).le hpos.le
    have h2 : (0:ℝ) < 2 ^ (-126 : ℤ) := zpow_pos (by norm_num) _
    rw [hgam]; linarith

/-- **A state value times the decay**: the `float32` product is finite and within `2^-24` relatively
(plus the floor) of the exact product. -/
theorem scale_facts {y gt s : F32} (hy : y.isFinite = true) (hgt : gt.isFinite = true)
    (hb : |y.toReal * gt.toReal| ≤ 2 ^ 120) (hs : FloatV.NaNEq (FloatV.mul y gt) s) :
    s.isFinite = true ∧ |s.toReal - gt.toReal * y.toReal| ≤ u32 * |gt.toReal * y.toReal| + e32 := by
  obtain ⟨hf, he⟩ := FloatV.mul_err_real Format.binary32_regular hy hgt (by
    rw [Format.binary32_maxVal_real]; exact hb.trans (by norm_num))
  rw [Format.binary32_u_real, Format.binary32_eta_real] at he
  rw [hs.eq_of_isFinite hf]
  refine ⟨hf, ?_⟩
  rw [mul_comm gt.toReal y.toReal]
  unfold u32 e32
  rw [mul_comm (2 ^ (-24 : ℤ) : ℝ)]
  exact he

/-- **A state value from its `bfloat16` rounding**: if the rounding is finite and of norm at most
`2^40`, the value is at most `2^41`. -/
theorem Y_le {y : F32} {h : BF16} (hy : y.isFinite = true) (hh : FloatV.NaNEq (FloatV.cvt y) h)
    (hhf : h.isFinite = true) (hb : |h.toReal| ≤ 2 ^ 40) : |y.toReal| ≤ 2 ^ 41 := by
  have := cvt_bf16_err_of_finite hy hh hhf
  have h8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 := by norm_num
  have h134 : (2 : ℝ) ^ (-134 : ℤ) ≤ 1 := by norm_num
  rw [h8] at this
  have h1 : |y.toReal| ≤ |h.toReal| + |h.toReal - y.toReal| := by
    have := abs_sub_abs_le_abs_sub y.toReal h.toReal
    rw [abs_sub_comm] at this
    linarith
  have : |y.toReal| ≤ 2 ^ 40 + (1 / 256 * |y.toReal| + 1) := by linarith [abs_nonneg y.toReal]
  have h41 : (2 : ℝ) ^ 41 = 2 * 2 ^ 40 := by norm_num
  have h40 : (1 : ℝ) ≤ 2 ^ 40 := by norm_num
  nlinarith [abs_nonneg y.toReal]

/-- **The padded rows' step, arithmetic**: the next value of a padded row (state value `ys` within
`pb`, decay `gtv` within `eB`, product `S`, then the update, whose `k` terms vanish) stays within the
next `padB`. -/
theorem pad_arith {ys gtv S Yn pb eB ρ α : ℝ} (hys : |ys| ≤ pb) (hgt : |gtv| ≤ eB)
    (hS : |S - gtv * ys| ≤ u32 * |gtv * ys| + e32) (hY : |Yn| ≤ (1 + ρ) * |S| + α)
    (hρ0 : 0 ≤ ρ) (hρ : ρ ≤ rD + 3 * u32) (hα : α ≤ (1 + rD + 3 * u32) * aD + 4 * e32) :
    |Yn| ≤ (1 + rD + 3 * u32) * (eB * (1 + u32) * pb + e32 + aD) + 4 * e32 := by
  have hu : 0 ≤ u32 := by unfold u32; positivity
  have he : 0 ≤ e32 := by unfold e32; positivity
  have hrD : 0 ≤ rD := by rw [rD_eq]; positivity
  have hpb : 0 ≤ pb := (abs_nonneg _).trans hys
  have heB : 0 ≤ eB := (abs_nonneg _).trans hgt
  have h1 : |gtv * ys| ≤ eB * pb := by
    rw [abs_mul]; exact mul_le_mul hgt hys (abs_nonneg _) heB
  have h2 : |S| ≤ (1 + u32) * (eB * pb) + e32 := by
    have h3 := abs_sub_abs_le_abs_sub S (gtv * ys)
    nlinarith
  have hR : 0 ≤ rD + 3 * u32 := by positivity
  have h4 : (1 + ρ) * |S| ≤ (1 + rD + 3 * u32) * ((1 + u32) * (eB * pb) + e32) := by
    apply mul_le_mul (by linarith) h2 (abs_nonneg _) (by linarith)
  nlinarith [mul_nonneg hR (mul_nonneg heB hpb)]

/-- **The `bfloat16` rounding of a finite `float32` is not a NaN** (it is finite or an infinity). -/
theorem cvt_f32_bf16_notNaN {y : F32} (hy : y.isFinite = true) : (FloatV.cvt y : BF16).isNaN = false := by
  have hreg := Format.bfloat16_regular
  have hpat : (FloatV.cvt y : BF16) = FloatV.ofPat
      (Format.bfloat16.roundRat .rn false (Format.binary32.sign y.pat) y.toRat) :=
    congrArg FloatV.ofPat (Format.cvt_of_finite _ _ _ hy)
  have hlt := roundRat_lt hreg (by decide) (Format.binary32.sign y.pat) (q := y.toRat)
  rw [hpat]
  by_cases hq0 : y.toRat = 0
  · have := (Format.roundRat_zero_spec (f := Format.bfloat16) (rnd := .rn) (sat := false)
      (zs := Format.binary32.sign y.pat) hreg)
    rw [hq0]
    have hfin : (FloatV.ofPat (Format.bfloat16.roundRat .rn false (Format.binary32.sign y.pat) 0) : BF16).isFinite = true := by
      rw [FloatV.isFinite_ofPat this.1]; exact this.2.1
    exact (FloatV.isFinite_iff.1 hfin).1
  · have ha : 0 < |y.toRat| := abs_pos.mpr hq0
    obtain ⟨hN, hQ, hsub⟩ := Format.roundSig_bounds (f := Format.bfloat16) .rn ha
    by_cases hover : (Format.roundSig .rn |y.toRat| (Format.bfloat16.qexp |y.toRat|) : ℚ) *
        (2 : ℚ) ^ (Format.bfloat16.qexp |y.toRat|) ≤ Format.bfloat16.maxVal
    · have hle := Format.pack_le_maxMag hreg hN hQ hsub hover
      have hlt' := hle.trans_lt (Format.maxMag_lt hreg)
      have hpat2 : Format.bfloat16.roundRat .rn false (Format.binary32.sign y.pat) y.toRat =
          Format.bfloat16.setSign (decide (y.toRat < 0)) (Format.bfloat16.pack
            (Format.roundSig .rn |y.toRat| (Format.bfloat16.qexp |y.toRat|)) (Format.bfloat16.qexp |y.toRat|)) := by
        rw [Format.roundRat_of_ne hq0]
        unfold Format.roundMag Format.encode
        rw [Format.pow2_eq, ite_eq_left hover, Format.setSign_of_lt hlt']
        by_cases h : y.toRat < 0 <;> simp [h]
      rw [hpat2, FloatV.isNaN_ofPat (Format.setSign_lt _ _), Format.isNaN_setSign]
      have hfin : Format.bfloat16.isFinite (Format.bfloat16.pack
            (Format.roundSig .rn |y.toRat| (Format.bfloat16.qexp |y.toRat|)) (Format.bfloat16.qexp |y.toRat|)) = true :=
        Format.isFinite_of_mag_le hreg (by rw [Format.mag_of_lt hlt']; exact hle)
      simp only [Format.isFinite, Bool.and_eq_true, Bool.not_eq_true'] at hfin
      exact hfin.1
    · have hsp : Format.bfloat16.specials = .ieee := rfl
      have hmag : Format.bfloat16.roundMag .rn false |y.toRat| = Format.bfloat16.infMag := by
        unfold Format.roundMag Format.encode
        rw [Format.pow2_eq, ite_eq_right hover]
        simp [hsp]
      rw [Format.roundRat_of_ne hq0, hmag]
      have hlt2 := FloatV.infMag_lt_signBit (f := Format.bfloat16)
      have hI := Format.isInf_infMag (f := Format.bfloat16) hsp
      have hpos : (if y.toRat < 0 then Format.bfloat16.signBit else 0) + Format.bfloat16.infMag =
          Format.bfloat16.setSign (decide (y.toRat < 0)) Format.bfloat16.infMag := by
        rw [Format.setSign_of_lt hlt2]; by_cases h : y.toRat < 0 <;> simp [h]
      rw [hpos, FloatV.isNaN_ofPat (Format.setSign_lt _ _), Format.isNaN_setSign]
      have hw : Format.bfloat16.infMag < 2 ^ Format.bfloat16.width := by
        have := Format.setSign_lt (f := Format.bfloat16) false Format.bfloat16.infMag
        rwa [Format.setSign_of_lt hlt2] at this
      have h2 : (FloatV.ofPat Format.bfloat16.infMag : BF16).isInf = true := by
        rw [FloatV.isInf_ofPat hw]; exact hI
      have h3 := FloatV.isNaN_of_isInf h2
      rwa [FloatV.isNaN_ofPat hw] at h3

/-- **The `[V, K]` layout's update, one lane**: the dot `D` (within `rD·L + aD` of the sum `Sg`,
`|Sg| ≤ L`) added to the decayed state `S` (a fusable `+`): finite and within
`(rD + 4·2^-24)(|S| + L) + ((1 + 3·2^-24)·aD + 4·2^-150)` of `S + Sg`. -/
theorem upd_vf {fusion : Bool} {S D Yn : F32} {Sg L : ℝ} (hS : S.isFinite = true)
    (hD : D.isFinite = true) (hroom : |S.toReal| + L ≤ 2 ^ 120) (hL0 : 0 ≤ L) (hSg : |Sg| ≤ L)
    (hdD : |D.toReal - Sg| ≤ rD * L + aD) (hadd : AddRel fusion S D Yn) :
    Yn.isFinite = true ∧ |Yn.toReal - (S.toReal + Sg)| ≤
      (rD + 4 * u32) * (|S.toReal| + L) + ((1 + 3 * u32) * aD + 4 * e32) := by
  have hr : rD = 66 * 2 ^ (-22 : ℤ) := rD_eq
  have ha : aD = 66 * 2 ^ (-126 : ℤ) := aD_eq
  have hu : u32 = 2 ^ (-24 : ℤ) := rfl
  have he : e32 = 2 ^ (-150 : ℤ) := rfl
  have hrD : rD ≤ 1 / 100 := by rw [hr]; norm_num
  have hrD0 : 0 ≤ rD := by rw [hr]; positivity
  have haD : aD ≤ 1 := by rw [ha]; norm_num
  have haD0 : 0 ≤ aD := by rw [ha]; positivity
  have hu0 : 0 ≤ u32 := by rw [hu]; positivity
  have hu1 : u32 ≤ 1 / 100 := by rw [hu]; norm_num
  have he0 : 0 ≤ e32 := by rw [he]; positivity
  have hD1 : |D.toReal| ≤ (1 + rD) * L + aD := by
    have := abs_sub_abs_le_abs_sub D.toReal Sg
    nlinarith [abs_nonneg D.toReal]
  have hS0 := abs_nonneg S.toReal
  have hM : |S.toReal| + |D.toReal| ≤ 2 ^ 125 := by
    have : (1 + rD) * L ≤ 2 * 2 ^ 120 := by nlinarith
    nlinarith [show (2 : ℝ) ^ 121 = 2 * 2 ^ 120 by norm_num, show (2 : ℝ) ^ 125 = 16 * 2 ^ 121 by norm_num]
  obtain ⟨hf, hb⟩ := add_f32 hS hD hM hadd
  refine ⟨hf, ?_⟩
  rw [← hu, ← he] at hb
  have h1 := abs_sub_abs_le_abs_sub (Yn.toReal - (S.toReal + D.toReal)) (Sg - D.toReal)
  have h2 : |Yn.toReal - (S.toReal + Sg)| ≤ |Yn.toReal - (S.toReal + D.toReal)| + |D.toReal - Sg| := by
    have : Yn.toReal - (S.toReal + Sg) = (Yn.toReal - (S.toReal + D.toReal)) + (D.toReal - Sg) := by ring
    rw [this]; exact abs_add_le _ _
  have hLu : 0 ≤ u32 * L := mul_nonneg hu0 hL0
  have hLr : 0 ≤ rD * L := mul_nonneg hrD0 hL0
  nlinarith [mul_nonneg hu0 hS0, mul_nonneg hu0 hrD0, mul_nonneg (mul_nonneg hu0 hrD0) hL0, mul_nonneg hu0 haD0]

/-- The same for a padded row (`L = 0`, `Sg = 0`): within `3·2^-24` relatively of `S`. -/
theorem upd_vf_pad {fusion : Bool} {S D Yn : F32} (hS : S.isFinite = true) (hD : D.isFinite = true)
    (hroom : |S.toReal| ≤ 2 ^ 120) (hdD : |D.toReal| ≤ aD) (hadd : AddRel fusion S D Yn) :
    Yn.isFinite = true ∧ |Yn.toReal| ≤ (1 + 3 * u32) * |S.toReal| + ((1 + 3 * u32) * aD + 4 * e32) := by
  have ha : aD = 66 * 2 ^ (-126 : ℤ) := aD_eq
  have hu : u32 = 2 ^ (-24 : ℤ) := rfl
  have he : e32 = 2 ^ (-150 : ℤ) := rfl
  have haD : aD ≤ 1 := by rw [ha]; norm_num
  have hu0 : 0 ≤ u32 := by rw [hu]; positivity
  have haD0 : 0 ≤ aD := by rw [ha]; positivity
  have hS0 := abs_nonneg S.toReal
  have hM : |S.toReal| + |D.toReal| ≤ 2 ^ 125 := by
    nlinarith [show (2 : ℝ) ^ 125 = 32 * 2 ^ 120 by norm_num, show (1 : ℝ) ≤ 2 ^ 120 by norm_num]
  obtain ⟨hf, hb⟩ := add_f32 hS hD hM hadd
  refine ⟨hf, ?_⟩
  rw [← hu, ← he] at hb
  have h1 := abs_sub_abs_le_abs_sub Yn.toReal (S.toReal + D.toReal)
  have h2 := abs_add_le S.toReal D.toReal
  nlinarith [mul_nonneg hu0 hS0, mul_nonneg hu0 haD0]

/-- **`0 + x = x`** in `float32`, for a finite `x`: the sum is finite and has the value of `x`. -/
theorem add_zero_left {x r : F32} (hx : x.isFinite = true)
    (h : FloatV.NaNEq (FloatV.add (FloatV.ofPat 0) x) r) : r.isFinite = true ∧ r.toRat = x.toRat := by
  have hreg := Format.binary32_regular
  have hxn' : Format.binary32.isNaN x.pat = false := (FloatV.isFinite_iff.1 hx).1
  have hxi' : Format.binary32.isInf x.pat = false := (FloatV.isFinite_iff.1 hx).2
  have hx' : Format.binary32.isFinite x.pat = true := hx
  have h0n : Format.binary32.isNaN 0 = false := by
    have := FloatV.isNaN_ofPat_zero (f := Format.binary32) hreg
    rwa [show (FloatV.ofPat 0 : F32).isNaN = Format.binary32.isNaN 0 from
      FloatV.isNaN_ofPat (by decide)] at this
  have h0i : Format.binary32.isInf 0 = false := by
    have := FloatV.isInf_ofPat_zero (f := Format.binary32) hreg
    rwa [show (FloatV.ofPat 0 : F32).isInf = Format.binary32.isInf 0 from
      FloatV.isInf_ofPat (by decide)] at this
  have h00 : Format.binary32.toRat 0 = 0 := FloatV.toRat_ofPat_zero
  have hadd : (FloatV.add (FloatV.ofPat 0) x : F32) = FloatV.ofPat
      (Format.binary32.roundRat .rn false (Format.binary32.sign 0 && Format.binary32.sign x.pat)
        (Format.binary32.toRat x.pat)) := by
    unfold FloatV.add Format.add
    simp [FloatV.pat_ofPat_zero, h0n, h0i, hxn', hxi', h00]
  have hlt := roundRat_lt hreg (by decide) (Format.binary32.sign 0 && Format.binary32.sign x.pat)
    (q := Format.binary32.toRat x.pat)
  have key : (FloatV.add (FloatV.ofPat 0) x : F32).isFinite = true ∧
      (FloatV.add (FloatV.ofPat 0) x : F32).toRat = x.toRat := by
    rw [hadd, FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
    by_cases hq : Format.binary32.toRat x.pat = 0
    · have hx0 : x.toRat = 0 := hq
      rw [hx0, hq]
      have := Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn) (sat := false)
        (zs := Format.binary32.sign 0 && Format.binary32.sign x.pat) hreg
      exact ⟨this.2.1, this.2.2⟩
    · have hc := Format.cvt_of_wider hreg hreg (by unfold Format.Wider; exact ⟨le_rfl, le_rfl, le_rfl⟩) .rn false hx'
      rw [Format.cvt_of_finite _ _ _ hx'] at hc
      obtain ⟨hcf, hcr⟩ := hc
      have e : Format.binary32.roundRat .rn false (Format.binary32.sign 0 && Format.binary32.sign x.pat)
          (Format.binary32.toRat x.pat) = Format.binary32.roundRat .rn false (Format.binary32.sign x.pat)
          (Format.binary32.toRat x.pat) := by
        rw [Format.roundRat_of_ne hq, Format.roundRat_of_ne hq]
      rw [e]
      exact ⟨hcf, hcr⟩
  have hf := key.1
  have heq := h.eq_of_isFinite hf
  rw [heq]
  exact key

end Claims.ChunkDeltaH
