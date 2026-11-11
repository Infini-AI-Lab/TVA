import Proof.NumV
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

/-!
# `chunk_delta_h_bf16`: the gated values (lines 247–250, 288)

One valid token, one value column: `exp2(g_last - g)` (the IEEE `-`, then the device's `exp2`),
the IEEE `*` with the new value `bv` and the `bfloat16` rounding `vb` that the state's dot takes.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- `2^e - 1 ≤ e` for `0 ≤ e ≤ 1`. -/
theorem two_rpow_sub_one_le {e : ℝ} (h0 : 0 ≤ e) (h1 : e ≤ 1) : (2 : ℝ) ^ e - 1 ≤ e := by
  have := rpow_one_add_le_one_add_mul_self (s := 1) (by norm_num) h0 h1
  norm_num at this
  linarith

/-- A power of two is at most `2^n` when the exponent is at most `n`. -/
theorem two_rpow_le_zpow {x : ℝ} {n : ℤ} (h : x ≤ n) : (2 : ℝ) ^ x ≤ 2 ^ n :=
  calc (2 : ℝ) ^ x ≤ (2 : ℝ) ^ (n : ℝ) := Real.rpow_le_rpow_of_exponent_le (by norm_num) h
    _ = 2 ^ n := Real.rpow_intCast 2 n

/-- **Two powers of two whose exponents are within `t` differ by at most `2^a·(2^t - 1)`.** -/
theorem rpow_two_gap {a b t : ℝ} (h : |b - a| ≤ t) :
    |(2 : ℝ) ^ b - (2 : ℝ) ^ a| ≤ (2 : ℝ) ^ a * ((2 : ℝ) ^ t - 1) := by
  have h2a : (0 : ℝ) < (2 : ℝ) ^ a := Real.rpow_pos_of_pos (by norm_num) a
  have ht : 0 ≤ t := le_trans (abs_nonneg _) h
  have hge1 : ∀ s : ℝ, 0 ≤ s → 1 ≤ (2 : ℝ) ^ s := fun s hs => Real.one_le_rpow (by norm_num) hs
  rcases le_total a b with hab | hab
  · have hb : (2 : ℝ) ^ b = (2 : ℝ) ^ a * (2 : ℝ) ^ (b - a) := by
      rw [← Real.rpow_add (by norm_num)]; congr 1; ring
    have hmono : (2 : ℝ) ^ (b - a) ≤ (2 : ℝ) ^ t :=
      Real.rpow_le_rpow_of_exponent_le (by norm_num)
        (by rw [abs_of_nonneg (by linarith)] at h; exact h)
    have h1 := hge1 (b - a) (by linarith)
    rw [abs_of_nonneg (by rw [hb]; nlinarith), hb]
    nlinarith
  · have hb : (2 : ℝ) ^ a = (2 : ℝ) ^ b * (2 : ℝ) ^ (a - b) := by
      rw [← Real.rpow_add (by norm_num)]; congr 1; ring
    have hmono : (2 : ℝ) ^ (a - b) ≤ (2 : ℝ) ^ t :=
      Real.rpow_le_rpow_of_exponent_le (by norm_num)
        (by rw [abs_of_nonpos (by linarith)] at h; linarith)
    have h1 := hge1 (a - b) (by linarith)
    have h2b : (0 : ℝ) < (2 : ℝ) ^ b := Real.rpow_pos_of_pos (by norm_num) b
    set s := (2 : ℝ) ^ (a - b) with hs
    have hspos : 0 < s := by linarith
    set v := 1 / s with hv
    have hvs : v * s = 1 := by rw [hv]; field_simp
    have hv0 : 0 < v := by positivity
    have hinv : (2 : ℝ) ^ b = (2 : ℝ) ^ a * v := by
      have : (2 : ℝ) ^ b * s = (2 : ℝ) ^ a := hb.symm
      nlinarith
    have hv1 : v ≤ 1 := by nlinarith
    have h3 : 1 - v ≤ s - 1 := by nlinarith [mul_nonneg hv0.le (sq_nonneg (s - 1))]
    rw [hinv, abs_of_nonpos (by nlinarith)]
    nlinarith


/-- The exponent's `float32` difference: finite and within `|dg|·2^-24 + 2^-150` of `dg`. -/
theorem gate_sub {bgl bg xs : F32} (hgl : bgl.isFinite = true ∧ |bgl.toReal| ≤ 2 ^ 24)
    (hg : bg.isFinite = true ∧ |bg.toReal| ≤ 2 ^ 24)
    (hxs : FloatV.NaNEq (FloatV.sub bgl bg) xs) :
    xs.isFinite = true ∧
      |xs.toReal - (bgl.toReal - bg.toReal)| ≤
        |bgl.toReal - bg.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.sub_err_real Format.binary32_regular hgl.1 hg.1 (by
    rw [Format.binary32_maxVal_real]
    have := abs_sub bgl.toReal bg.toReal
    have : (2 : ℝ) ^ 24 + 2 ^ 24 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith [hgl.2, hg.2])
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  rw [hxs.eq_of_isFinite h1]
  exact ⟨h1, h2⟩

/-- The device's `exp2` of a finite argument at most `9`: finite and within `2^-22·2^x + 2^-126`
of `2^x`. -/
theorem gate_exp {P : Platform} (hP : Device.Exp2F32 P) {xs e : F32} (hx : xs.isFinite = true)
    (hxb : xs.toReal ≤ 9) (he : P.math .exp2 .binary32 xs e) :
    e.isFinite = true ∧
      |e.toReal - (2 : ℝ) ^ xs.toReal| ≤ 2 ^ (-22 : ℤ) * (2 : ℝ) ^ xs.toReal + 2 ^ (-126 : ℤ) := by
  obtain ⟨g, hg, h⟩ := hP
  have hgf : g = fun x : ℝ => (2 : ℝ) ^ x := by
    simpa [MathFn.ref] using hg.symm
  subst hgf
  have h9 : (2 : ℝ) ^ xs.toReal ≤ 2 ^ (9 : ℤ) := two_rpow_le_zpow (by push_cast; exact hxb)
  have hpos : (0 : ℝ) < (2 : ℝ) ^ xs.toReal := Real.rpow_pos_of_pos (by norm_num) _
  have hroom : (1 + 2 ^ (-22 : ℤ)) * |(fun x : ℝ => (2 : ℝ) ^ x) xs.toReal| + 2 ^ (-126 : ℤ) ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    beta_reduce
    rw [Format.binary32_maxVal_real, abs_of_pos hpos]
    norm_num at h9 ⊢
    nlinarith
  have key := h xs e he hx (Set.mem_univ _) (Set.mem_univ _) hroom
  beta_reduce at key
  rwa [abs_of_pos hpos] at key


/-- **The arithmetic of the gate away from underflow** (`2^dg` not below `2^-128`): `e` is within
`8·10^-6` of `G`, relatively. -/
theorem gate_rel {G Z vref bv e p vb : ℝ} (hG0 : 0 < G) (hG8 : G ≤ 2 ^ 8) (hZ0 : 0 ≤ Z)
    (hZ : Z ≤ 2 ^ 61) (hvr : |vref| ≤ Z)
    (hbv : |bv - vref| ≤ (3 / 5) * 2 ^ (-13 : ℤ) * Z + 2 ^ (-110 : ℤ))
    (he : |e - G| ≤ G * (8 / 10 ^ 6) + 2 ^ (-126 : ℤ))
    (hp : |p - bv * e| ≤ 2 ^ (-24 : ℤ) * |bv * e| + 2 ^ (-150 : ℤ))
    (hvb : |vb - p| ≤ 2 ^ (-8 : ℤ) * |p| + 2 ^ (-134 : ℤ)) :
    |vb - G * vref| ≤ 2 ^ (-8 : ℤ) * (G * (|vref| + 2 ^ (-5 : ℤ) * Z)) + 2 ^ (-63 : ℤ) := by
  set δ0 : ℝ := (3 / 5) * 2 ^ (-13 : ℤ) * Z + 2 ^ (-110 : ℤ) with hδ0
  have hδ0n : 0 ≤ δ0 := by positivity
  have hB : |bv| ≤ Z + δ0 := by
    have := abs_add_le vref (bv - vref)
    simp only [add_sub_cancel] at this
    linarith
  set D2 : ℝ := G * ((Z + δ0) * (8 / 10 ^ 6) + δ0) + (Z + δ0) * 2 ^ (-126 : ℤ) with hD2
  have hD2' : |bv * e - G * vref| ≤ D2 := by
    have hsplit : bv * e - G * vref = bv * (e - G) + G * (bv - vref) := by ring
    rw [hsplit]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul, abs_of_pos hG0]
    have h1 : |bv| * |e - G| ≤ (Z + δ0) * (G * (8 / 10 ^ 6) + 2 ^ (-126 : ℤ)) :=
      mul_le_mul hB he (abs_nonneg _) (by positivity)
    have h2 : G * |bv - vref| ≤ G * δ0 := mul_le_mul_of_nonneg_left hbv hG0.le
    rw [hD2]
    nlinarith
  have hGV : G * |vref| ≤ G * Z := mul_le_mul_of_nonneg_left hvr hG0.le
  have hGv : |G * vref| = G * |vref| := by rw [abs_mul, abs_of_pos hG0]
  have hbe : |bv * e| ≤ G * |vref| + D2 := by
    have := abs_add_le (G * vref) (bv * e - G * vref)
    rw [add_sub_cancel, hGv] at this
    linarith
  have hpG : |p - G * vref| ≤ 2 ^ (-24 : ℤ) * (G * |vref| + D2) + 2 ^ (-150 : ℤ) + D2 := by
    have h1 : p - G * vref = (p - bv * e) + (bv * e - G * vref) := by ring
    rw [h1]
    refine (abs_add_le _ _).trans ?_
    have : 2 ^ (-24 : ℤ) * |bv * e| ≤ 2 ^ (-24 : ℤ) * (G * |vref| + D2) :=
      mul_le_mul_of_nonneg_left hbe (by positivity)
    linarith
  have hpabs : |p| ≤ G * |vref| + |p - G * vref| := by
    have := abs_add_le (G * vref) (p - G * vref)
    rw [add_sub_cancel, hGv] at this
    linarith
  have hfin : |vb - G * vref| ≤ |vb - p| + |p - G * vref| := by
    have : vb - G * vref = (vb - p) + (p - G * vref) := by ring
    rw [this]; exact abs_add_le _ _
  have hGc : G * 2 ^ (-110 : ℤ) ≤ 2 ^ 8 * 2 ^ (-110 : ℤ) :=
    mul_le_mul_of_nonneg_right hG8 (by positivity)
  have hZc : Z * 2 ^ (-126 : ℤ) ≤ 2 ^ 61 * 2 ^ (-126 : ℤ) :=
    mul_le_mul_of_nonneg_right hZ (by positivity)
  have hVZ : |vref| ≤ Z := hvr
  have hV0 := abs_nonneg vref
  rw [hD2, hδ0] at hpG
  norm_num at hpG hpabs hfin hvb hGc hZc hGV ⊢
  nlinarith


/-- **The arithmetic of the gate under underflow** (`2^dg ≤ 2^-128`, the computed `2^xs` at most
`2^-127`): the rounded product is at most `2^-63` in magnitude, and so is the exact one. -/
theorem gate_abs {G X Z vref bv e p vb : ℝ} (hG0 : 0 < G) (hG : G ≤ 2 ^ (-128 : ℤ))
    (hX0 : 0 < X) (hX : X ≤ 2 ^ (-127 : ℤ)) (hZ0 : 0 ≤ Z) (hZ : Z ≤ 2 ^ 61) (hvr : |vref| ≤ Z)
    (hbv : |bv - vref| ≤ (3 / 5) * 2 ^ (-13 : ℤ) * Z + 2 ^ (-110 : ℤ))
    (he : |e - X| ≤ 2 ^ (-22 : ℤ) * X + 2 ^ (-126 : ℤ))
    (hp : |p - bv * e| ≤ 2 ^ (-24 : ℤ) * |bv * e| + 2 ^ (-150 : ℤ))
    (hvb : |vb - p| ≤ 2 ^ (-8 : ℤ) * |p| + 2 ^ (-134 : ℤ)) :
    |vb - G * vref| ≤ 2 ^ (-8 : ℤ) * (G * (|vref| + 2 ^ (-5 : ℤ) * Z)) + 2 ^ (-63 : ℤ) := by
  have hB : |bv| ≤ 2 ^ 61 + 2 ^ 48 := by
    have h1 := abs_add_le vref (bv - vref)
    simp only [add_sub_cancel] at h1
    have h2 : (3 / 5 : ℝ) * 2 ^ (-13 : ℤ) * Z ≤ 3 / 5 * 2 ^ (-13 : ℤ) * 2 ^ 61 :=
      mul_le_mul_of_nonneg_left hZ (by positivity)
    norm_num at h2 hbv ⊢
    linarith
  have hE : |e| ≤ 2 ^ (-127 : ℤ) * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-126 : ℤ) := by
    have h1 := abs_add_le X (e - X)
    simp only [add_sub_cancel] at h1
    rw [abs_of_pos hX0] at h1
    have h2 : (2 : ℝ) ^ (-22 : ℤ) * X ≤ 2 ^ (-22 : ℤ) * 2 ^ (-127 : ℤ) :=
      mul_le_mul_of_nonneg_left hX (by positivity)
    norm_num at h2 hX h1 he ⊢
    linarith
  have hbe : |bv * e| ≤ (2 ^ 61 + 2 ^ 48) * (2 ^ (-127 : ℤ) * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-126 : ℤ)) := by
    rw [abs_mul]
    exact mul_le_mul hB hE (abs_nonneg _) (by positivity)
  have hGv : G * |vref| ≤ 2 ^ (-128 : ℤ) * 2 ^ 61 :=
    mul_le_mul hG (hvr.trans hZ) (abs_nonneg _) (by positivity)
  have hpa : |p| ≤ |bv * e| + |p - bv * e| := by
    have := abs_add_le (bv * e) (p - bv * e)
    rwa [add_sub_cancel] at this
  have hfin : |vb - G * vref| ≤ |vb - p| + |p| + G * |vref| := by
    have h1 : vb - G * vref = (vb - p) + p - G * vref := by ring
    have h2 := abs_sub (vb - p + p) (G * vref)
    rw [h1]
    rw [abs_mul, abs_of_pos hG0] at h2
    have h3 := abs_add_le (vb - p) p
    linarith
  have hRHS : 0 ≤ 2 ^ (-8 : ℤ) * (G * (|vref| + 2 ^ (-5 : ℤ) * Z)) := by positivity
  norm_num at hbe hpa hfin hvb hp hGv hRHS ⊢
  nlinarith


/-- **The gated value of one token.** `bv` is within `0.6·2^-13·Z + 2^-110` of `vref`
(`|vref| ≤ Z ≤ 2^61`, `numV`); the gate values `bgl`, `bg` are finite, at most `2^24` and
`bgl - bg ≤ 8`; `xs` is the IEEE difference, `e` the device's `exp2` of it, `p` the IEEE product
`bv·e`, `vb` its `bfloat16` rounding. Then `vb` is finite and within
`2^-8·2^dg·vAbs + 2^-63` of `2^dg·vref`, `dg = bgl - bg`. -/
theorem numG {P : Platform} (hP : Device.Exp2F32 P) {vref Z : ℝ} (hZ : Z ≤ 2 ^ 61)
    (hvr : |vref| ≤ Z) (bv : F32)
    (hbv : bv.isFinite = true ∧
      |bv.toReal - vref| ≤ (3 / 5) * 2 ^ (-13 : ℤ) * Z + 2 ^ (-110 : ℤ))
    (bgl bg xs e p : F32) (vb : BF16)
    (hgl : bgl.isFinite = true ∧ |bgl.toReal| ≤ 2 ^ 24)
    (hg : bg.isFinite = true ∧ |bg.toReal| ≤ 2 ^ 24) (hd8 : bgl.toReal - bg.toReal ≤ 8)
    (hxs : FloatV.NaNEq (FloatV.sub bgl bg) xs) (he : P.math .exp2 .binary32 xs e)
    (hp : FloatV.NaNEq (FloatV.mul bv e) p) (hvb : FloatV.NaNEq (FloatV.cvt p) vb) :
    vb.isFinite = true ∧
      |vb.toReal - Real.rpow 2 (bgl.toReal - bg.toReal) * vref| ≤
        2 ^ (-8 : ℤ) * (Real.rpow 2 (bgl.toReal - bg.toReal) * vAbsA vref Z) + 2 ^ (-63 : ℤ) := by
  obtain ⟨hbvf, hbvb⟩ := hbv
  have hZ0 : 0 ≤ Z := (abs_nonneg _).trans hvr
  obtain ⟨hxf, hxe⟩ := gate_sub hgl hg hxs
  set dg : ℝ := bgl.toReal - bg.toReal with hdg
  have hdgb : |dg| ≤ 2 ^ 25 := by
    have := abs_sub bgl.toReal bg.toReal
    have h2 : (2 : ℝ) ^ 24 + 2 ^ 24 = 2 ^ 25 := by norm_num
    linarith [hgl.2, hg.2]
  have hxs9 : xs.toReal ≤ 9 := by
    have h1 := (abs_le.mp hxe).2
    rcases le_total 0 dg with h | h
    · rw [abs_of_nonneg h] at h1
      norm_num at h1
      linarith
    · rw [abs_of_nonpos h] at h1
      norm_num at h1
      linarith
  obtain ⟨hef, hee⟩ := gate_exp hP hxf hxs9 he
  set X : ℝ := (2 : ℝ) ^ xs.toReal with hXdef
  have hX0 : 0 < X := Real.rpow_pos_of_pos (by norm_num) _
  have hX9 : X ≤ 2 ^ 9 := by
    have := two_rpow_le_zpow (x := xs.toReal) (n := 9) (by push_cast; exact hxs9)
    norm_num at this ⊢
    exact this
  have hBv : |bv.toReal| ≤ 2 ^ 62 := by
    have h1 := abs_add_le vref (bv.toReal - vref)
    simp only [add_sub_cancel] at h1
    have h2 : (3 / 5 : ℝ) * 2 ^ (-13 : ℤ) * Z ≤ 3 / 5 * 2 ^ (-13 : ℤ) * 2 ^ 61 :=
      mul_le_mul_of_nonneg_left hZ (by positivity)
    norm_num at h2 hbvb ⊢
    linarith
  have hEe : |e.toReal| ≤ 2 ^ 10 := by
    have h1 := abs_add_le X (e.toReal - X)
    simp only [add_sub_cancel] at h1
    rw [abs_of_pos hX0] at h1
    have h2 : (2 : ℝ) ^ (-22 : ℤ) * X ≤ 2 ^ (-22 : ℤ) * 2 ^ 9 :=
      mul_le_mul_of_nonneg_left hX9 (by positivity)
    norm_num at h2 hX9 h1 hee ⊢
    linarith
  have hprod : |bv.toReal * e.toReal| ≤ 2 ^ 72 := by
    rw [abs_mul]
    calc |bv.toReal| * |e.toReal| ≤ 2 ^ 62 * 2 ^ 10 :=
          mul_le_mul hBv hEe (abs_nonneg _) (by positivity)
      _ = 2 ^ 72 := by norm_num
  obtain ⟨hmf, hme⟩ := FloatV.mul_err_real Format.binary32_regular hbvf hef (by
    rw [Format.binary32_maxVal_real]
    exact hprod.trans (by norm_num))
  rw [Format.binary32_u_real, Format.binary32_eta_real] at hme
  have hpeq := hp.eq_of_isFinite hmf
  subst hpeq
  have hpb : |(FloatV.mul bv e).toReal| ≤ 2 ^ 100 := by
    have h1 := abs_sub_abs_le_abs_sub (FloatV.mul bv e).toReal (bv.toReal * e.toReal)
    have h2 : |bv.toReal * e.toReal| * 2 ^ (-24 : ℤ) ≤ 2 ^ 72 * 2 ^ (-24 : ℤ) :=
      mul_le_mul_of_nonneg_right hprod (by positivity)
    have h3 : (2 : ℝ) ^ 72 + 2 ^ 72 * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) ≤ 2 ^ 100 := by norm_num
    linarith
  obtain ⟨hvf, hvbe⟩ := cvt_bf16 hmf hpb hvb
  refine ⟨hvf, ?_⟩
  simp only [Real.rpow_eq_pow]
  have hG0 : 0 < (2 : ℝ) ^ dg := Real.rpow_pos_of_pos (by norm_num) _
  have hme' : |(FloatV.mul bv e).toReal - bv.toReal * e.toReal| ≤
      2 ^ (-24 : ℤ) * |bv.toReal * e.toReal| + 2 ^ (-150 : ℤ) := by
    rw [mul_comm (2 ^ (-24 : ℤ) : ℝ)]; exact hme
  rcases lt_or_ge dg (-128) with hlow | hhigh
  · have hxlow : xs.toReal ≤ ((-127 : ℤ) : ℝ) := by
      have h1 := (abs_le.mp hxe).2
      rw [abs_of_nonpos (by linarith : dg ≤ 0)] at h1
      push_cast
      norm_num at h1
      linarith
    have hX127 : X ≤ 2 ^ (-127 : ℤ) := two_rpow_le_zpow hxlow
    have hG128 : (2 : ℝ) ^ dg ≤ 2 ^ (-128 : ℤ) := two_rpow_le_zpow (by push_cast; linarith)
    exact gate_abs hG0 hG128 hX0 hX127 hZ0 hZ hvr hbvb hee hme' hvbe
  · have hdg128 : |dg| ≤ 128 := abs_le.mpr ⟨by linarith, by linarith⟩
    obtain ⟨t0, ht0⟩ : ∃ t0 : ℝ, t0 = 128 * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := ⟨_, rfl⟩
    have ht0s : t0 ≤ 77 / 10 ^ 7 := by rw [ht0]; norm_num
    have ht0n : 0 ≤ t0 := by rw [ht0]; positivity
    have ht : |xs.toReal - dg| ≤ t0 := by
      refine hxe.trans ?_
      have := mul_le_mul_of_nonneg_right hdg128 (by positivity : (0 : ℝ) ≤ 2 ^ (-24 : ℤ))
      rw [ht0]
      linarith
    have hgap := rpow_two_gap ht
    have ht1 : (2 : ℝ) ^ t0 - 1 ≤ t0 := two_rpow_sub_one_le ht0n (by linarith)
    have hG8 : (2 : ℝ) ^ dg ≤ 2 ^ 8 := by
      have := two_rpow_le_zpow (x := dg) (n := 8) (by push_cast; linarith)
      simpa using this
    have hgap' : |X - (2 : ℝ) ^ dg| ≤ (2 : ℝ) ^ dg * t0 :=
      hgap.trans (mul_le_mul_of_nonneg_left ht1 hG0.le)
    have hXG : X ≤ (2 : ℝ) ^ dg * (1 + t0) := by
      have := (abs_le.mp hgap').2
      linarith
    have hX22 : (2 : ℝ) ^ (-22 : ℤ) * X ≤ 2 ^ (-22 : ℤ) * ((2 : ℝ) ^ dg * (1 + t0)) :=
      mul_le_mul_of_nonneg_left hXG (by positivity)
    have he' : |e.toReal - (2 : ℝ) ^ dg| ≤ (2 : ℝ) ^ dg * (8 / 10 ^ 6) + 2 ^ (-126 : ℤ) := by
      have h1 := abs_add_le (e.toReal - X) (X - (2 : ℝ) ^ dg)
      have h2 : e.toReal - X + (X - (2 : ℝ) ^ dg) = e.toReal - (2 : ℝ) ^ dg := by ring
      rw [h2] at h1
      have hmul : (2 : ℝ) ^ dg * t0 ≤ (2 : ℝ) ^ dg * (77 / 10 ^ 7) :=
        mul_le_mul_of_nonneg_left ht0s hG0.le
      have hmul2 : (2 : ℝ) ^ (-22 : ℤ) * ((2 : ℝ) ^ dg * (1 + t0)) ≤
          2 ^ (-22 : ℤ) * ((2 : ℝ) ^ dg * (1 + 77 / 10 ^ 7)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (by linarith) hG0.le) (by positivity)
      have c22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
      have c126 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
      rw [c22] at hee hX22 hmul2
      rw [c126] at hee ⊢
      generalize (2 : ℝ) ^ dg = G at *
      norm_num at hmul2 hmul ⊢
      linarith
    exact gate_rel hG0 hG8 hZ0 hZ hvr hbvb he' hme' hvbe


/-- **A masked token** (`tl.where` gives the factor `0`): the product and its rounding are
finite. -/
theorem numG_masked (bv z p : F32) (vb : BF16) (hbv : bv.isFinite = true)
    (hz : z.isFinite = true ∧ z.toReal = 0) (hp : FloatV.NaNEq (FloatV.mul bv z) p)
    (hvb : FloatV.NaNEq (FloatV.cvt p) vb) : vb.isFinite = true := by
  obtain ⟨hmf, hme⟩ := FloatV.mul_err_real Format.binary32_regular hbv hz.1 (by
    rw [hz.2, mul_zero, abs_zero, Format.binary32_maxVal_real]
    norm_num)
  rw [hz.2, mul_zero, abs_zero, zero_mul, zero_add, sub_zero, Format.binary32_eta_real] at hme
  have hpeq := hp.eq_of_isFinite hmf
  subst hpeq
  have hpb : |(FloatV.mul bv z).toReal| ≤ 2 ^ 100 := by
    have : (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ 100 := by norm_num
    linarith
  exact (cvt_bf16 hmf hpb hvb).1

end Claims.ChunkDeltaH
