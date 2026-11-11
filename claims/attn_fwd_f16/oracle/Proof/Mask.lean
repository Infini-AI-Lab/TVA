import Proof.Plain

/-!
# The masked trip

The exponents and weights of a trip with the causal mask (`STAGE = 2`), one row, and the walk of
the trip's body.
-/

namespace Claims.AttnFwd

open TVA

noncomputable section

/-- Rounding a rational of magnitude below `2^-25` to `float16` gives zero. -/
theorem cvt16_tiny_aux (zs : Bool) {q : ℚ} (hs : |q| < 1 / 2 ^ 25) :
    Format.binary16.toRat (Format.binary16.roundRat .rn false zs q) = 0 := by
  have hg : Format.binary16.Regular := Format.Std.regular Format.binary16_std
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨-, -, h3⟩ := Format.roundRat_zero_spec (f := Format.binary16) (rnd := .rn) (sat := false) (zs := zs) hg
    exact h3
  have hmax : |q| ≤ Format.binary16.maxVal := by
    refine hs.le.trans ?_
    rw [Format.binary16_maxVal]; norm_num
  obtain ⟨-, -, hval⟩ := Format.roundRat_spec hg .rn false zs hq0 hmax
  have ha : 0 < |q| := abs_pos.mpr hq0
  have herr := Format.roundSig_rn_err ha (Format.binary16.qexp |q|)
  have hQ : (2 : ℚ) ^ (-24 : ℤ) ≤ (2 : ℚ) ^ (Format.binary16.qexp |q|) := by
    apply zpow_le_zpow_right₀ (by norm_num)
    rw [Format.qexp_eq ha]
    have : Format.binary16.emin - Format.binary16.m ≤ max (Int.log 2 |q|) Format.binary16.emin - Format.binary16.m := by
      have := le_max_right (Int.log 2 |q|) Format.binary16.emin
      omega
    refine le_trans ?_ this
    decide
  have hN : (Format.roundSig .rn |q| (Format.binary16.qexp |q|) : ℚ) = 0 := by
    set N := (Format.roundSig .rn |q| (Format.binary16.qexp |q|) : ℚ) with hNd
    have hN0 : (0 : ℚ) ≤ N := by positivity
    have h2 := (abs_le.mp herr).2
    have h24 : (2 : ℚ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
    have hp : (0:ℚ) < (2 : ℚ) ^ (Format.binary16.qexp |q|) := by positivity
    have : N * (2 : ℚ) ^ (Format.binary16.qexp |q|) < (2 : ℚ) ^ (Format.binary16.qexp |q|) := by
      have : |q| < (2 : ℚ) ^ (Format.binary16.qexp |q|) / 2 := by
        have : (1:ℚ) / 2 ^ 25 ≤ (2 : ℚ) ^ (Format.binary16.qexp |q|) / 2 := by
          rw [h24] at hQ; linarith
        linarith
      linarith
    have hN1 : N < 1 := by
      by_contra hc
      push Not at hc
      nlinarith
    have : (Format.roundSig .rn |q| (Format.binary16.qexp |q|)) = 0 := by
      have : (Format.roundSig .rn |q| (Format.binary16.qexp |q|) : ℚ) < 1 := hN1
      exact_mod_cast Nat.lt_one_iff.mp (by exact_mod_cast this)
    rw [hNd, this]; simp
  rw [hval, hN]; simp

/-- The nearest `float16` to a `float32` of magnitude below `2^-25` is zero. -/
theorem cvt16_tiny {x : F32} {r : F16} (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r)
    (hx : x.isFinite = true) (hs : |x.toRat| < 1 / 2 ^ 25) : r.toRat = 0 := by
  have hf : (FloatV.cvt x : F16).isFinite = true :=
    (FloatV.cvt_err (g := Format.binary16) (Format.Std.regular Format.binary16_std) hx (by
      rw [Format.binary16_maxVal]; refine hs.le.trans ?_; norm_num)).1
  have hr : r = FloatV.cvt x := by
    have := h.eq_of_isFinite (by rw [FloatV.cvtWith_rn]; exact hf)
    rw [FloatV.cvtWith_rn] at this
    exact this
  subst hr
  unfold FloatV.cvt
  rw [Format.cvt_of_finite _ .rn false hx]
  have hmax : |x.toRat| ≤ Format.binary16.maxVal := by
    refine hs.le.trans ?_; rw [Format.binary16_maxVal]; norm_num
  obtain ⟨hlt, -, -⟩ := Format.roundRat_rn_err (Format.Std.regular Format.binary16_std) false
    (Format.binary32.sign x.pat) hmax
  change (FloatV.ofPat (Format.binary16.roundRat .rn false (Format.binary32.sign x.pat) x.toRat) : F16).toRat = 0
  rw [FloatV.toRat_ofPat hlt]
  exact cvt16_tiny_aux _ hs

/-- A masked trip's exponent is within `ξ` of the exact one: the sum `c·d̂ + w` is rounded by a
`mul_add` (within `3u`), and the difference with the maximum once (within `u`). -/
theorem expo_round_m {σ sc dh c m1 prod y X : ℝ} (D : ℕ) (hX : 0 ≤ X ∧ X ≤ 1024) (hrD0 : 0 ≤ rD D)
    (hBc : |c * dh| ≤ 14429 / 10000 * X + 1 / 2 ^ 44)
    (hscore : |c * dh - σ * sc / Real.log 2| ≤ 14427 / 10000 * rD D * X + 5 * u32 * X + 1 / 2 ^ 43)
    (hm1 : |m1| ≤ Bm X) (hp : |prod - c * dh| ≤ 3 * u32 * |c * dh| + 4 / 2 ^ 150)
    (hy : |y - (prod - m1)| ≤ u32 * |prod - m1| + 1 / 2 ^ 150) :
    |y - (σ * sc / Real.log 2 - m1)| ≤ xi D X := by
  have hu : u32 = 1 / 2 ^ 24 := rfl
  have hBm : |m1| ≤ 1443 / 1000 * X + 1 / 2 ^ 40 := hm1
  have hcdh' : |c * dh| ≤ 14429 / 10000 * X + 1 / 2 ^ 40 := hBc.trans (by norm_num)
  have hprod : |prod| ≤ 14429 / 10000 * X + 1 / 2 ^ 40 + (3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 40) + 4 / 2 ^ 150) := by
    have := abs_sub_abs_le_abs_sub prod (c * dh)
    have h4 : 3 * u32 * |c * dh| ≤ 3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 40) :=
      mul_le_mul_of_nonneg_left hcdh' (by rw [hu]; norm_num)
    linarith
  have hd : |prod - m1| ≤ |prod| + |m1| := abs_sub _ _
  have h5 : u32 * |prod - m1| ≤ u32 * (|prod| + |m1|) :=
    mul_le_mul_of_nonneg_left hd (by rw [hu]; norm_num)
  have h6 : u32 * (|prod| + |m1|) ≤ u32 * ((14429 / 10000 * X + 1 / 2 ^ 40 +
      (3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 40) + 4 / 2 ^ 150)) + (1443 / 1000 * X + 1 / 2 ^ 40)) :=
    mul_le_mul_of_nonneg_left (by linarith) (by rw [hu]; norm_num)
  have h4 : 3 * u32 * |c * dh| ≤ 3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 40) :=
    mul_le_mul_of_nonneg_left hcdh' (by rw [hu]; norm_num)
  have hround : |y - (c * dh - m1)| ≤ u32 * ((14429 / 10000 * X + 1 / 2 ^ 40 +
      (3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 40) + 4 / 2 ^ 150)) + (1443 / 1000 * X + 1 / 2 ^ 40)) +
      1 / 2 ^ 150 + (3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 40) + 4 / 2 ^ 150) := by
    have e : y - (c * dh - m1) = (y - (prod - m1)) + (prod - c * dh) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    linarith
  have e : y - (σ * sc / Real.log 2 - m1) = (y - (c * dh - m1)) + (c * dh - σ * sc / Real.log 2) := by ring
  rw [e]
  refine (abs_add_le _ _).trans ?_
  have hxi : xi D X = X * (14428 / 10000 * rD D + 1 / 2 ^ 20) + 1 / 2 ^ 40 := rfl
  rw [hxi]
  have hrX := mul_nonneg hrD0 hX.1
  rw [hu] at hround hscore
  nlinarith [hround, hscore, hrX, hX.1]

/-- `expo_up` with the `mul_add` rounding. -/
theorem expo_up_m {σ sc dh c m1 X : ℝ} (D : ℕ) (hX : 0 ≤ X ∧ X ≤ 1024) (hrD0 : 0 ≤ rD D)
    (_hBc : |c * dh| ≤ 14429 / 10000 * X + 1 / 2 ^ 44)
    (hscore : |c * dh - σ * sc / Real.log 2| ≤ 14427 / 10000 * rD D * X + 5 * u32 * X + 1 / 2 ^ 43)
    (hmax : c * dh - m1 ≤ 3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 44) + 4 / 2 ^ 150) :
    σ * sc / Real.log 2 - m1 ≤ xi D X := by
  have hxi : xi D X = X * (14428 / 10000 * rD D + 1 / 2 ^ 20) + 1 / 2 ^ 40 := rfl
  rw [hxi]
  have hrX := mul_nonneg hrD0 hX.1
  have hu : u32 = 1 / 2 ^ 24 := rfl
  rw [hu] at hmax hscore
  have := (abs_le.mp hscore).1
  nlinarith [hrX, hX.1]

/-- `expo_down` with the `mul_add` rounding. -/
theorem expo_down_m {σ sc dh c m1 X : ℝ} (D : ℕ) (hX : 0 ≤ X ∧ X ≤ 1024) (hrD0 : 0 ≤ rD D)
    (hscore : |c * dh - σ * sc / Real.log 2| ≤ 14427 / 10000 * rD D * X + 5 * u32 * X + 1 / 2 ^ 43)
    (hmax : m1 - c * dh ≤ 3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 44) + 4 / 2 ^ 150) :
    m1 - xi D X ≤ σ * sc / Real.log 2 := by
  have hxi : xi D X = X * (14428 / 10000 * rD D + 1 / 2 ^ 20) + 1 / 2 ^ 40 := rfl
  rw [hxi]
  have hrX := mul_nonneg hrD0 hX.1
  have hu : u32 = 1 / 2 ^ 24 := rfl
  rw [hu] at hmax hscore
  have := (abs_le.mp hscore).2
  nlinarith [hrX, hX.1]

/-- The two constants of the mask (`tl.where(mask, 0, -1.0e6)`), as `float32` values. -/
theorem mask_consts : (FloatV.litF32 (FloatV.ofPat 0) : F32).isFinite = true ∧
    (FloatV.litF32 (FloatV.ofPat 0) : F32).toReal = 0 ∧
    (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000) : F32).isFinite = true ∧
    (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000) : F32).toReal = -1000000 := by
  have h1 : (FloatV.litF32 (FloatV.ofPat 0) : F32).isFinite = true := by decide +kernel
  have h2 : (FloatV.litF32 (FloatV.ofPat 0) : F32).toRat = 0 := by decide +kernel
  have h3 : (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000) : F32).isFinite = true := by decide +kernel
  have h4 : (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000) : F32).toRat = -1000000 := by decide +kernel
  refine ⟨h1, ?_, h3, ?_⟩
  · unfold FloatV.toReal; rw [h2]; norm_num
  · unfold FloatV.toReal; rw [h4]; norm_num

/-- **The reference and the exponents of a masked trip** (one row). The lanes `ok` are attended
(the mask holds, `w = 0`), the others get `w = −10^6`. -/
theorem trip_expo_m {fusion : Bool} {D : ℕ} (hD : D = 64 ∨ D = 128 ∨ D = 256) {X σ : ℝ}
    (hX : 0 ≤ X ∧ X ≤ 1024) (hσ : 1 / 2 ^ 64 ≤ σ ∧ σ ≤ 2 ^ 64)
    {qkr q1r yr w : Fin 64 → F32} {ok : Fin 64 → Prop} {sc Λ : Fin 64 → ℝ} {c mx m0 m1 : F32}
    (hqk : ∀ j, (qkr j).isFinite = true ∧
      |(qkr j).toReal - sc j| ≤ rD D * Λ j + ((D : ℝ) + 2) / 2 ^ 126)
    (hΛ0 : ∀ j, 0 ≤ Λ j) (hΛ : ∀ j, σ * Λ j ≤ 1024) (hΛX : ∀ j, ok j → σ * Λ j ≤ X)
    (hsc : ∀ j, |sc j| ≤ Λ j)
    (hcf : c.isFinite = true) (_hcpos : 0 < c.toReal)
    (hc : |c.toReal - σ / Real.log 2| ≤ 31 / 10 * u32 * (σ / Real.log 2))
    (hw0 : ∀ j, ok j → w j = FloatV.litF32 (FloatV.ofPat 0))
    (hw1 : ∀ j, ¬ ok j → w j = FloatV.litF32 (FloatV.ofPat 0xC12E848000000000))
    (hq1 : ∀ j, FloatV.NaNEq (FloatV.add (FloatV.mul (qkr j) c) (w j)) (q1r j) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (qkr j) c (w j)) (q1r j)))
    (hmx : ∃ t : Tree, t.Covers 64 ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < 64 then q1r ⟨j, h⟩ else default) mx)
    (hm1 : FloatV.MaxRel true m0 mx m1)
    (hm0 : m0 = FloatV.negInf ∨ (m0.isFinite = true ∧ |m0.toReal| ≤ Bm X))
    (hne : (∃ j, ok j) ∨ m0.isFinite = true)
    (hy : ∀ j, FloatV.NaNEq (FloatV.sub (q1r j) m1) (yr j)) :
    m1.isFinite = true ∧ |m1.toReal| ≤ Bm X ∧ (m0.isFinite = true → m0.toReal ≤ m1.toReal) ∧
      (∀ j, ok j → (yr j).isFinite = true ∧
        |(yr j).toReal - (σ * sc j / Real.log 2 - m1.toReal)| ≤ xi D X ∧
        σ * sc j / Real.log 2 - m1.toReal ≤ xi D X) ∧
      (∀ j, ¬ ok j → (yr j).isFinite = true ∧ (yr j).toReal ≤ -900000) ∧
      ((m0.isFinite = true ∧ m1.toReal = m0.toReal) ∨
        ∃ j, ok j ∧ m1.toReal - xi D X ≤ σ * sc j / Real.log 2) := by
  classical
  obtain ⟨hz1, hz2, hn1, hn2⟩ := mask_consts
  have hrD0 : 0 ≤ rD D := by unfold rD; positivity
  have hu : u32 = 1 / 2 ^ 24 := rfl
  have hu0 : 0 ≤ u32 := by rw [hu]; norm_num
  have hwb : ∀ j, (w j).isFinite = true ∧ (ok j → (w j).toReal = 0) ∧ (¬ ok j → (w j).toReal = -1000000) := by
    intro j
    by_cases hj : ok j
    · rw [hw0 j hj]; exact ⟨hz1, fun _ => hz2, fun h => absurd hj h⟩
    · rw [hw1 j hj]; exact ⟨hn1, fun h => absurd h hj, fun _ => hn2⟩
  have h1024 : (0 : ℝ) ≤ 1024 ∧ (1024 : ℝ) ≤ 1024 := ⟨by norm_num, le_rfl⟩
  have hcd : ∀ j, |c.toReal * (qkr j).toReal| ≤ 1500 := fun j =>
    (expo_score D hD h1024 hσ (hΛ0 j) (hΛ j) (hsc j) (hqk j).2 hc).1.trans (by norm_num)
  have hq1l : ∀ j, (q1r j).isFinite = true ∧
      |(q1r j).toReal - ((qkr j).toReal * c.toReal + (w j).toReal)| ≤
        3 * u32 * (|(qkr j).toReal * c.toReal| + |(w j).toReal|) + 4 / 2 ^ 150 := by
    intro j
    refine mulAdd_real (hq1 j) (hqk j).1 hcf (hwb j).1 ?_
    have h1 : |(qkr j).toReal * c.toReal| ≤ 1500 := by rw [mul_comm]; exact hcd j
    have h2 : (1002000 : ℝ) ≤ 2 ^ 100 := by norm_num
    by_cases hj : ok j
    · rw [(hwb j).2.1 hj, abs_zero]; linarith
    · rw [(hwb j).2.2 hj, abs_neg, abs_of_pos (by norm_num : (0:ℝ) < 1000000)]; linarith
  -- masked lanes
  have hmk : ∀ j, ¬ ok j → (q1r j).toReal ≤ -998000 ∧ -1002000 ≤ (q1r j).toReal := by
    intro j hj
    have h := (hq1l j).2
    have h1 := hcd j
    rw [mul_comm] at h1
    rw [(hwb j).2.2 hj] at h
    have h2 := abs_le.mp h1
    have h3 : 3 * u32 * (|(qkr j).toReal * c.toReal| + |(-1000000 : ℝ)|) + 4 / 2 ^ 150 ≤ 1 := by
      have : |(qkr j).toReal * c.toReal| + |(-1000000 : ℝ)| ≤ 1002000 := by
        rw [abs_neg, abs_of_pos (by norm_num : (0:ℝ) < 1000000)]; linarith [abs_le.mp h1]
      have := mul_le_mul_of_nonneg_left this (show 0 ≤ 3 * u32 by positivity)
      rw [hu] at this ⊢; norm_num at this ⊢; linarith
    have h4 := abs_le.mp h
    constructor <;> linarith [h4.1, h4.2, h2.1, h2.2]
  -- attended lanes
  have hok : ∀ j, ok j → |c.toReal * (qkr j).toReal| ≤ 14429 / 10000 * X + 1 / 2 ^ 44 ∧
      |c.toReal * (qkr j).toReal - σ * sc j / Real.log 2| ≤
        14427 / 10000 * rD D * X + 5 * u32 * X + 1 / 2 ^ 43 :=
    fun j hj => expo_score D hD hX hσ (hΛ0 j) (hΛX j hj) (hsc j) (hqk j).2 hc
  have hokq : ∀ j, ok j → |(q1r j).toReal - c.toReal * (qkr j).toReal| ≤
      3 * u32 * |c.toReal * (qkr j).toReal| + 4 / 2 ^ 150 := by
    intro j hj
    have h := (hq1l j).2
    rw [(hwb j).2.1 hj, add_zero, abs_zero, add_zero, mul_comm (qkr j).toReal] at h
    exact h
  have hBcle : (14429 / 10000 * X + 1 / 2 ^ 44 : ℝ) ≤ 1500 := by nlinarith [hX.2]
  have hokabs : ∀ j, ok j → |(q1r j).toReal| ≤ 1443 / 1000 * X + 1 / 2 ^ 40 := by
    intro j hj
    obtain ⟨h1, -⟩ := hok j hj
    have h2 := hokq j hj
    have h3 := abs_sub_abs_le_abs_sub (q1r j).toReal (c.toReal * (qkr j).toReal)
    have h4 : 3 * u32 * |c.toReal * (qkr j).toReal| ≤ 3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 44) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    rw [hu] at h4 h2
    nlinarith [hX.1]
  have hokge : ∀ j, ok j → -1600 ≤ (q1r j).toReal := by
    intro j hj
    have := abs_le.mp (hokabs j hj)
    nlinarith [hX.2, hX.1]
  have hnanq : ∀ i, i < 64 → (if h : i < 64 then q1r ⟨i, h⟩ else (default : F32)).isNaN = false := by
    intro i hi
    simp only [dite_eq_left hi]
    exact (FloatV.isFinite_iff.mp (hq1l ⟨i, hi⟩).1).1
  -- the maximum of the row
  obtain ⟨t, hcov, hev⟩ := hmx
  have hlv : ∀ i, i ∈ t.leaves ↔ i < 64 := fun i => by rw [hcov.mem_iff, List.mem_range]
  obtain ⟨⟨i0, hi0, hmxeq⟩, hle⟩ := Tree.eval_maxRel_toReal hev (fun i hi => hnanq i ((hlv i).mp hi))
  have hi0' : i0 < 64 := (hlv i0).mp hi0
  set j0 : Fin 64 := ⟨i0, hi0'⟩ with hj0
  have hmxj : mx = q1r j0 := by rw [hmxeq]; simp [hj0, hi0']
  have hmxf : mx.isFinite = true := by rw [hmxj]; exact (hq1l j0).1
  have hqle : ∀ j, (q1r j).toReal ≤ mx.toReal := by
    intro j
    have := hle j.1 ((hlv j.1).mpr j.2)
    simpa [dite_eq_left j.2] using this
  have hmxnan : mx.isNaN = false := (FloatV.isFinite_iff.mp hmxf).1
  -- the maximum is an attended lane when there is one
  have hmxok : (∃ j, ok j) → ∃ jm, ok jm ∧ mx = q1r jm := by
    rintro ⟨j1, hj1⟩
    by_cases hj : ok j0
    · exact ⟨j0, hj, hmxj⟩
    · exfalso
      have h1 := (hmk j0 hj).1
      have h2 := hqle j1
      have h3 := hokge j1 hj1
      rw [hmxj] at h2
      linarith
  have hBm0 : 0 ≤ Bm X := by unfold Bm; have := hX.1; positivity
  have hBm1500 : Bm X ≤ 1500 := by unfold Bm; nlinarith [hX.2]
  have hm0b : m0.isFinite = true → |m0.toReal| ≤ Bm X := by
    intro hf
    rcases hm0 with h0 | ⟨-, hb⟩
    · exfalso
      have hninf : (FloatV.negInf : F32).isFinite = false := by decide +kernel
      rw [h0, hninf] at hf; exact absurd hf (by simp)
    · exact hb
  -- the cases of the new maximum
  have hcases : (m1 = mx ∧ mx.toReal ≤ m1.toReal) ∨
      (m0.isFinite = true ∧ m1.toReal = m0.toReal ∧ mx.toReal ≤ m1.toReal) := by
    rcases hm0 with h0 | ⟨h0f, -⟩
    · left
      have : m1 = mx := FloatV.MaxRel.negInf_left binary32_reg (Format.Std.ieee Format.binary32_std)
        (h0 ▸ hm1) hmxnan
      exact ⟨this, by rw [this]⟩
    · have hm0nan : m0.isNaN = false := (FloatV.isFinite_iff.mp h0f).1
      have hmax := hm1.toReal_eq_max hm0nan hmxnan
      rcases hm1.mem hm0nan hmxnan with h | h
      · right; exact ⟨h0f, by rw [h], by rw [hmax]; exact le_max_right _ _⟩
      · left; exact ⟨h, by rw [hmax]; exact le_max_right _ _⟩
  have hm1f : m1.isFinite = true := by
    rcases hcases with ⟨h, -⟩ | ⟨h0f, h, -⟩
    · rw [h]; exact hmxf
    · have hm0nan : m0.isNaN = false := (FloatV.isFinite_iff.mp h0f).1
      rcases hm1.mem hm0nan hmxnan with h' | h'
      · rw [h']; exact h0f
      · rw [h']; exact hmxf
  have hmono : m0.isFinite = true → m0.toReal ≤ m1.toReal := by
    intro h0f
    have hm0nan : m0.isNaN = false := (FloatV.isFinite_iff.mp h0f).1
    have hmax := hm1.toReal_eq_max hm0nan hmxnan
    rw [hmax]; exact le_max_left _ _
  have hge : mx.toReal ≤ m1.toReal := by rcases hcases with ⟨-, h⟩ | ⟨-, -, h⟩ <;> exact h
  have hm1B : |m1.toReal| ≤ Bm X := by
    rcases hcases with ⟨h, -⟩ | ⟨h0f, h, -⟩
    · rw [h]
      by_cases hex : ∃ j, ok j
      · obtain ⟨jm, hjm, hmj⟩ := hmxok hex
        rw [hmj]; exact hokabs jm hjm
      · exfalso
        have hm0f : m0.isFinite = true := hne.resolve_left hex
        have h1 := hmono hm0f
        have h2 := (hmk j0 (fun hh => hex ⟨j0, hh⟩)).1
        have h3 := (abs_le.mp (hm0b hm0f)).1
        rw [h] at h1
        rw [hmxj] at h1
        linarith
    · rw [h]; exact hm0b h0f
  have hm1r : |m1.toReal| ≤ 1500 := hm1B.trans hBm1500
  refine ⟨hm1f, hm1B, hmono, fun j hj => ?_, fun j hj => ?_, ?_⟩
  · obtain ⟨hs1, hs2⟩ := hok j hj
    have hq1b := hokabs j hj
    obtain ⟨hyf, hye⟩ := sub_real (hy j) (hq1l j).1 hm1f (by
      have := abs_sub (q1r j).toReal m1.toReal
      nlinarith [hX.2, hX.1])
    refine ⟨hyf, ?_, ?_⟩
    · refine expo_round_m D hX hrD0 (c := c.toReal) (dh := (qkr j).toReal) (prod := (q1r j).toReal)
        (y := (yr j).toReal) hs1 hs2 hm1B (hokq j hj) hye
    · refine expo_up_m D hX hrD0 hs1 hs2 ?_
      have h1 := hokq j hj
      have h2 := (abs_le.mp h1).1
      have h3 := hqle j
      have h4 : 3 * u32 * |c.toReal * (qkr j).toReal| ≤ 3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 44) :=
        mul_le_mul_of_nonneg_left hs1 (by positivity)
      linarith
  · obtain ⟨hlo, hhi⟩ := hmk j hj
    obtain ⟨hyf, hye⟩ := sub_real (hy j) (hq1l j).1 hm1f (by
      have h1 : |(q1r j).toReal| ≤ 1002000 := abs_le.mpr ⟨by linarith, by linarith⟩
      have h2 := abs_sub (q1r j).toReal m1.toReal
      have h3 : (1003500 : ℝ) ≤ 2 ^ 100 := by norm_num
      linarith)
    refine ⟨hyf, ?_⟩
    have h1 := (abs_le.mp hye).2
    have h2 : u32 * |(q1r j).toReal - m1.toReal| ≤ 1 := by
      have : |(q1r j).toReal - m1.toReal| ≤ 1004000 := by
        rw [abs_le]; have := abs_le.mp hm1r; constructor <;> linarith
      have := mul_le_mul_of_nonneg_left this hu0
      rw [hu] at this ⊢; norm_num at this ⊢; linarith
    have h3 : (1 : ℝ) / 2 ^ 150 ≤ 1 := by norm_num
    have := abs_le.mp hm1r
    linarith
  · rcases hcases with ⟨hEq, -⟩ | ⟨h0f, h, -⟩
    · by_cases hex : ∃ j, ok j
      · right
        obtain ⟨jm, hjm, hmj⟩ := hmxok hex
        refine ⟨jm, hjm, ?_⟩
        obtain ⟨hs1, hs2⟩ := hok jm hjm
        refine expo_down_m D hX hrD0 hs2 ?_
        have h1 := hokq jm hjm
        have h2 := (abs_le.mp h1).2
        have h4 : 3 * u32 * |c.toReal * (qkr jm).toReal| ≤ 3 * u32 * (14429 / 10000 * X + 1 / 2 ^ 44) :=
          mul_le_mul_of_nonneg_left hs1 (by positivity)
        have : m1.toReal = (q1r jm).toReal := by rw [hEq, hmj]
        rw [this]
        linarith
      · exfalso
        have hm0f : m0.isFinite = true := hne.resolve_left hex
        have h1 := hmono hm0f
        have h2 := (hmk j0 (fun hh => hex ⟨j0, hh⟩)).1
        have h3 := (abs_le.mp (hm0b hm0f)).1
        rw [hEq, hmxj] at h1
        linarith
    · left; exact ⟨h0f, h⟩

/-- **A masked lane's weight is negligible**: with exponent at most `−900000` the `exp2` is below
`2^-125` and rounds to `0` in `float16`. -/
theorem masked_weight {P : Platform} (hP : Device P) {y p : F32} {p16 : F16}
    (hy : y.isFinite = true) (hyle : y.toReal ≤ -900000) (hp : P.math .exp2 .binary32 y p)
    (h16 : FloatV.NaNEq (FloatV.cvtWith .rn p) p16) :
    p.isFinite = true ∧ |p.toReal| ≤ 1 / 2 ^ 125 ∧ p16.toRat = 0 := by
  obtain ⟨hpf, hpe⟩ := exp2_real hP hp hy (by linarith)
  have h2 : (2 : ℝ) ^ y.toReal ≤ 2 ^ (-200 : ℝ) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
  have h3 : (2 : ℝ) ^ (-200 : ℝ) = 1 / 2 ^ 200 := by norm_num
  have h4 : (0 : ℝ) < (2 : ℝ) ^ y.toReal := Real.rpow_pos_of_pos (by norm_num) _
  have hq : q22 ≤ 1 := by unfold q22; norm_num
  have hb : |p.toReal| ≤ 1 / 2 ^ 125 := by
    have h5 := abs_sub_abs_le_abs_sub p.toReal ((2 : ℝ) ^ y.toReal)
    rw [abs_of_pos h4] at h5
    have h6 : q22 * (2 : ℝ) ^ y.toReal ≤ 1 * (1 / 2 ^ 200) :=
      mul_le_mul hq (h3 ▸ h2) h4.le (by norm_num)
    have h8 : (2 : ℝ) ^ y.toReal ≤ 1 / 2 ^ 200 := h3 ▸ h2
    have h7 : (1 : ℝ) / 2 ^ 200 + 1 * (1 / 2 ^ 200) + 1 / 2 ^ 126 ≤ 1 / 2 ^ 125 := by norm_num
    linarith
  refine ⟨hpf, hb, cvt16_tiny h16 hpf ?_⟩
  have : |p.toReal| < 1 / 2 ^ 25 := hb.trans_lt (by norm_num)
  unfold FloatV.toReal at this
  have h8 : ((|p.toRat| : ℚ) : ℝ) < ((1 / 2 ^ 25 : ℚ) : ℝ) := by push_cast; exact this
  exact_mod_cast h8

/-- **The exponents of a masked trip**, one row: `ExpoOK`, and a masked lane's exponent is below
`−900000`. -/
theorem expo_masked {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    {bh sm τ : Nat} (hsm : sm < G d) (hτ : 64 * τ + 64 ≤ d.N_CTX)
    (h : HeadOk d init bh) (r : Fin 128) {X : ℝ} (hPar : Par d.HEAD_DIM X)
    (hkeys : ∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X)
    {c : F32} (hc : c.isFinite = true ∧ 0 < c.toReal ∧
      |c.toReal - (sigma d : ℝ) / Real.log 2| ≤ 31 / 10 * u32 * ((sigma d : ℝ) / Real.log 2))
    {q : Tensor [128, d.HEAD_DIM] F16} {kt : Tensor [64, d.HEAD_DIM] F16}
    (hq : ∀ (r : Fin 128) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.q.ptr (idx d bh (sm * 128 + r) e : Int)).addr = some (q.get2 r e))
    (hk : ∀ (kk : Fin 64) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.k.ptr (idx d bh (64 * τ + kk) e : Int)).addr = some (kt.get2 kk e))
    {fusion : Bool} {mt : Tensor [128] F32} {qk qk1 y : Tensor [128, 64] F32}
    {w : Fin 64 → F32} {mx m1 : Tensor [128] F32}
    (hdot : P.dot cfg16 q kt.trans (tl.zeros [128, 64] DType.f32) qk)
    (hm0 : mt.get1 r = FloatV.negInf ∨ ((mt.get1 r).isFinite = true ∧ |(mt.get1 r).toReal| ≤ Bm X))
    (hw0 : ∀ j : Fin 64, 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r) → w j = FloatV.litF32 (FloatV.ofPat 0))
    (hw1 : ∀ j : Fin 64, 64 * τ + (j : Nat) ∉ keys d (sm * 128 + r) →
      w j = FloatV.litF32 (FloatV.ofPat 0xC12E848000000000))
    (hq1 : ∀ j : Fin 64, FloatV.NaNEq (FloatV.add (FloatV.mul (qk.get2 r j) c) (w j)) (qk1.get2 r j) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (qk.get2 r j) c (w j)) (qk1.get2 r j)))
    (hne : (∃ j : Fin 64, 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r)) ∨ (mt.get1 r).isFinite = true)
    (hmx : ∃ t : Tree, t.Covers 64 ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < 64 then qk1.get2 r ⟨j, h⟩ else default) (mx.get1 r))
    (hm1 : FloatV.MaxRel true (mt.get1 r) (mx.get1 r) (m1.get1 r))
    (hy : ∀ j : Fin 64, FloatV.NaNEq (FloatV.sub (qk1.get2 r j) (m1.get1 r)) (y.get2 r j)) :
    ExpoOK d init bh (sm * 128 + r) τ X (mt.get1 r) (m1.get1 r) (fun j => y.get2 r j) ∧
      ∀ j : Fin 64, 64 * τ + (j : Nat) ∉ keys d (sm * 128 + r) →
        (y.get2 r j).isFinite = true ∧ (y.get2 r j).toReal ≤ -900000 := by
  have hσ : 1 / 2 ^ 64 ≤ (sigma d : ℝ) ∧ (sigma d : ℝ) ≤ 2 ^ 64 := by
    obtain ⟨-, hlo, hhi⟩ := hd.scale
    refine ⟨?_, ?_⟩
    · have : ((1 / 2 ^ 64 : ℚ) : ℝ) ≤ (sigma d : ℝ) := by exact_mod_cast hlo
      simpa using this
    · have : (sigma d : ℝ) ≤ ((2 ^ 64 : ℚ) : ℝ) := by exact_mod_cast hhi
      simpa using this
  have hn : sm * 128 + (r : Nat) < d.N_CTX := sm_row_lt hd hsm r.isLt
  have hlane := fun j : Fin 64 => lane_dot1 hd hP h hτ hsm hq hk hdot r j
  have hΛ0 : ∀ j : Fin 64, 0 ≤ (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ) := fun j => by
    exact_mod_cast scoreMag_nonneg (d := d) init bh (sm * 128 + r) (64 * τ + j)
  have hΛ : ∀ j : Fin 64, (sigma d : ℝ) * (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ) ≤ 1024 := fun j => by
    have := h.2.2 (sm * 128 + r) (64 * τ + j) hn (by have := j.isLt; omega)
    exact_mod_cast this
  have hΛX : ∀ j : Fin 64, 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r) →
      (sigma d : ℝ) * (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ) ≤ X := fun j hj => by
    have := hkeys _ hj
    exact_mod_cast this
  have hsc : ∀ j : Fin 64, |(score d init bh (sm * 128 + r) (64 * τ + j) : ℝ)| ≤
      (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ) := fun j => by
    have := abs_score_le (d := d) init bh (sm * 128 + r) (64 * τ + j)
    exact_mod_cast this
  obtain ⟨hm1f, hm1B, hmono, hlanes, hmasked, hwit⟩ := trip_expo_m (fusion := fusion) (D := d.HEAD_DIM)
    hd.head_dim hPar.hX hσ (qkr := fun j => qk.get2 r j) (q1r := fun j => qk1.get2 r j)
    (yr := fun j => y.get2 r j) (w := w) (ok := fun j => 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r))
    (sc := fun j => (score d init bh (sm * 128 + r) (64 * τ + j) : ℝ))
    (Λ := fun j => (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ)) (c := c) (mx := mx.get1 r)
    (m0 := mt.get1 r) (m1 := m1.get1 r)
    (fun j => hlane j) hΛ0 hΛ hΛX hsc hc.1 hc.2.1 hc.2.2 hw0 hw1 hq1 hmx hm1 hm0 hne hy
  exact ⟨⟨hm1f, hm1B, hmono, hlanes, hwit⟩, hmasked⟩


/-- The body of the key loop for `STAGE = 2` (source lines 70–109): the trip with the causal mask,
`start_n` the induction variable. -/
def maskedTrip (d : Inst) (q : Tensor [128, d.HEAD_DIM] F16) (desc_k desc_v : TensorDesc .f16)
    (c : F32) (offs_m : Tensor [128] I32) (offs_n : Tensor [64] I32) (start_n : I32) (x : St d) :
    Prog (St d) := do
  let kt ← tl.desc_load (bs := [64, d.HEAD_DIM]) desc_k [x.2.2.2.1, IntV.ofInt 0]
  let qk ← tl.dot cfg16 q kt.trans (tl.zeros [128, 64] DType.f32)
  let mask := (start_n + offs_n.row 128).le (offs_m.col 64)
  let qk1 ← tl.mul_add qk (bc c)
    (Tensor.where_ mask (bc (FloatV.litF32 (FloatV.ofPat 0)))
      (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000))))
  let mx ← tl.max1 true qk1
  let m1 ← tl.maximum x.2.2.1 mx
  let y ← tl.sub qk1 (m1.col 64)
  let p ← tl.exp2 y
  let s ← tl.sub x.2.2.1 m1
  let al ← tl.exp2 s
  let lij ← tl.sum1 p
  let acc1 ← tl.mul x.1 (al.col d.HEAD_DIM)
  let vt ← tl.desc_load (bs := [64, d.HEAD_DIM]) desc_v [x.2.2.2.2, IntV.ofInt 0]
  let p16 : Tensor [128, 64] F16 ← tl.cvt p
  let acc' ← tl.dot cfg16 p16 vt acc1
  let l' ← tl.mul_add x.2.1 al lij
  pure (acc', l', m1, x.2.2.2.1 + IntV.ofNat 64, x.2.2.2.2 + IntV.ofNat 64)

/-- **The masked trip**: from the state after `τ` blocks (`τ = 2·sm` or `2·sm + 1`, so
`start_n = 64τ` is in the query tile), the body leaves the state after `τ + 1` blocks. -/
theorem safe_masked_trip {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d)
    (hcau : d.causal = true)
    {q : Tensor [128, d.HEAD_DIM] F16} {c : F32}
    (hq : ∀ (r : Fin 128) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.q.ptr (idx d bh (sm * 128 + r) e : Int)).addr = some (q.get2 r e))
    (hc : c.isFinite = true ∧ 0 < c.toReal ∧
      |c.toReal - (sigma d : ℝ) / Real.log 2| ≤ 31 / 10 * u32 * ((sigma d : ℝ) / Real.log 2))
    {offs_m : Tensor [128] I32} {offs_n : Tensor [64] I32} {start_n : I32}
    (hoffm : ∀ r : Fin 128, (offs_m.get1 r).toInt = ((sm * 128 + r : Nat) : Int))
    (hoffn : ∀ j : Fin 64, (offs_n.get1 j).toInt = ((j : Nat) : Int))
    (τ : Nat) (hτ : 64 * τ + 64 ≤ d.N_CTX) (hτ1 : τ < 1024)
    (hrow : 64 * τ ≤ sm * 128 ∨ 1 ≤ τ)
    (hstart : start_n.toInt = ((64 * τ : Nat) : Int))
    (st : St d) (ho : st.2.2.2.1.toInt = ((bh * d.N_CTX + 64 * τ : Nat) : Int))
    (ho2 : st.2.2.2.2.toInt = ((bh * d.N_CTX + 64 * τ : Nat) : Int))
    (hst : StateOK d init bh sm τ st.2.2.1 st.2.1 st.1) :
    Safe ((launch d).env P) (Rset d) ∅
      (fun x m' => m' = init ∧ StateOK d init bh sm (τ + 1) x.2.2.1 x.2.1 x.1 ∧
        x.2.2.2.1.toInt = ((bh * d.N_CTX + 64 * (τ + 1) : Nat) : Int) ∧
        x.2.2.2.2.toInt = ((bh * d.N_CTX + 64 * (τ + 1) : Nat) : Int)) init
      (maskedTrip d q (hostDesc d d.k 64) (hostDesc d d.v 64) c offs_m offs_n start_n st) := by
  have hbN := row_bound' hd hbh
  have hN32 := hd.n_le
  have hrl := hd.rows_lt
  unfold maskedTrip
  refine safe_tile_load hd (b := d.k) (by simpa [Buf.ptr] using hd.align.2.1) hd.lens.2.1 hpre.2.1 64 _ ?_ ?_ ?_ fun kt hkt => ?_
  · rw [ho]; positivity
  · rw [ho]; push_cast; omega
  · exact Set.subset_union_right.trans (Set.subset_union_left)
  refine safe_dot (hP.dotTotal _ _ _ _) fun qk hqk => ?_
  refine safe_mul_add fun qk1 hqk1 => ?_
  refine safe_max1 (by norm_num) true fun mx hmx => ?_
  refine safe_maximum fun m1 hm1 => ?_
  refine safe_sub fun y hy => ?_
  refine safe_math (fn := .exp2) (fun i => hP.mathTotal _ _ _) fun p hp => ?_
  refine safe_sub fun s hs => ?_
  refine safe_math (fn := .exp2) (fun i => hP.mathTotal _ _ _) fun al hal => ?_
  refine safe_sum1 (by norm_num) fun lij hl => ?_
  refine safe_mul fun acc1 ha1 => ?_
  refine safe_tile_load hd (b := d.v) (by simpa [Buf.ptr] using hd.align.2.2.1) hd.lens.2.2.1 hpre.2.2.1 64 _ ?_ ?_ ?_ fun vt hvt => ?_
  · rw [ho2]; positivity
  · rw [ho2]; push_cast; omega
  · exact Set.subset_union_right
  refine safe_cvt (g := .binary16) ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩ fun p16 hp16 => ?_
  refine safe_dot (hP.dotTotal _ _ _ _) fun acc' hacc' => ?_
  refine safe_mul_add fun l' hl' => ?_
  refine safe_ret ?_
  have hk' : ∀ (kk : Fin 64) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.k.ptr (idx d bh (64 * τ + kk) e : Int)).addr = some (kt.get2 kk e) := by
    intro kk e
    have := hkt kk e
    rw [ho, Int.toNat_natCast] at this
    rw [← this]; unfold idx; congr 3; ring
  have hv' : ∀ (kk : Fin 64) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.v.ptr (idx d bh (64 * τ + kk) e : Int)).addr = some (vt.get2 kk e) := by
    intro kk e
    have := hvt kk e
    rw [ho2, Int.toNat_natCast] at this
    rw [← this]; unfold idx; congr 3; ring
  -- the mask of lane `(r, j)` is `64τ + j ≤ sm·128 + r`, i.e. the key is attended
  have hmaskl : ∀ (r : Fin 128) (j : Fin 64),
      (((start_n + offs_n.row 128).le (offs_m.col 64)).get ((r, j, ()) : Idx [128, 64])) = true ↔
        64 * τ + (j : Nat) ∈ keys d (sm * 128 + r) := by
    intro r j
    have hs : (start_n + (offs_n.row 128).get ((r, j, ()) : Idx [128, 64])).toInt = ((64 * τ + (j : Nat) : Nat) : Int) :=
      IntV.I32.add_nat hstart (hoffn j) (by omega)
    rw [show ((start_n + offs_n.row 128).le (offs_m.col 64)).get ((r, j, ()) : Idx [128, 64]) = IntV.le ((start_n + offs_n.row 128).get ((r, j, ()) : Idx [128, 64])) ((offs_m.col 64).get ((r, j, ()) : Idx [128, 64])) from rfl, IntV.le_eq, decide_eq_true_iff]
    show (start_n + offs_n.get1 j).toInt ≤ (offs_m.get1 r).toInt ↔ _
    rw [show (start_n + offs_n.get1 j) = start_n + (offs_n.row 128).get ((r, j, ()) : Idx [128, 64]) from rfl, hs, hoffm r]
    simp only [keys, hcau, Finset.mem_filter, Finset.mem_range]
    have := j.isLt
    constructor
    · intro hh; refine ⟨by omega, Or.inr (by omega)⟩
    · rintro ⟨-, h1 | h1⟩
      · simp at h1
      · omega
  have hW0 : ∀ (r : Fin 128) (j : Fin 64), 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r) →
      (Tensor.where_ ((start_n + offs_n.row 128).le (offs_m.col 64))
        (bc (FloatV.litF32 (FloatV.ofPat 0))) (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000)))).get ((r, j, ()) : Idx [128, 64]) =
        FloatV.litF32 (FloatV.ofPat 0) := by
    intro r j hj
    refine (Tensor.get_where _ _ _ _).trans ((ite_eq_left ((hmaskl r j).mpr hj)).trans rfl)
  have hW1 : ∀ (r : Fin 128) (j : Fin 64), 64 * τ + (j : Nat) ∉ keys d (sm * 128 + r) →
      (Tensor.where_ ((start_n + offs_n.row 128).le (offs_m.col 64))
        (bc (FloatV.litF32 (FloatV.ofPat 0))) (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000)))).get ((r, j, ()) : Idx [128, 64]) =
        FloatV.litF32 (FloatV.ofPat 0xC12E848000000000) := by
    intro r j hj
    have : ((start_n + offs_n.row 128).le (offs_m.col 64)).get ((r, j, ()) : Idx [128, 64]) = false := by
      by_contra hh
      exact hj ((hmaskl r j).mp (by simpa using hh))
    refine (Tensor.get_where _ _ _ _).trans ((ite_eq_right (by rw [this]; simp)).trans rfl)
  have hX1024 : Par d.HEAD_DIM (1024 : ℝ) := ⟨hd.head_dim, by norm_num, le_rfl⟩
  have hexp : ∀ (h : HeadOk d init bh) (r : Fin 128) (X : ℝ), Par d.HEAD_DIM X →
      (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
      ExpoOK d init bh (sm * 128 + r) τ X (st.2.2.1.get1 r) (m1.get1 r) (fun j => y.get2 r j) ∧
      ∀ j : Fin 64, 64 * τ + (j : Nat) ∉ keys d (sm * 128 + r) →
        (y.get2 r j).isFinite = true ∧ (y.get2 r j).toReal ≤ -900000 := by
    intro h r X hX hk
    refine expo_masked hd hP hsm hτ h r hX hk hc hq hk' hqk (w := fun j => (Tensor.where_
        ((start_n + offs_n.row 128).le (offs_m.col 64)) (bc (FloatV.litF32 (FloatV.ofPat 0)))
        (bc (FloatV.litF32 (FloatV.ofPat 0xC12E848000000000)))).get ((r, j, ()) : Idx [128, 64]))
      (state_m0 hst h r hX hk) (hW0 r) (hW1 r) (fun j => hqk1 (r, j, ())) ?_ (hmx r) (hm1 (r, ()))
      (fun j => hy (r, j, ()))
    rcases hrow with h1 | h1
    · left
      refine ⟨⟨0, by norm_num⟩, ?_⟩
      simp only [keys, hcau, Finset.mem_filter, Finset.mem_range]
      have := r.isLt
      exact ⟨by omega, Or.inr (by omega)⟩
    · right
      rcases hst h r with ⟨h0, -⟩ | ⟨-, hf, -⟩
      · omega
      · exact hf
  refine ⟨rfl, ?_, ?_, ?_⟩
  · refine trip_state hd hP hsm hτ hτ1 hv' (fun h r X hX hk => (hexp h r X hX hk).1)
      (fun h r j hj => ?_) ⟨hp, hs, hal, hl, ha1, hp16, hacc', hl'⟩ hst
    have hn : sm * 128 + (r : Nat) < d.N_CTX := sm_row_lt hd hsm r.isLt
    obtain ⟨hyf, hyle⟩ := (hexp h r 1024 hX1024 (keys_1024 h hn)).2 j hj
    exact masked_weight hP hyf hyle (hp (r, j, ())) (hp16 (r, j, ()))
  · exact (IntV.I32.add_nat ho (IntV.I32.toInt_ofNat (by norm_num)) (by omega)).trans (by push_cast; ring_nf)
  · exact (IntV.I32.add_nat ho2 (IntV.I32.toInt_ofNat (by norm_num)) (by omega)).trans (by push_cast; ring_nf)

end

end Claims.AttnFwd
