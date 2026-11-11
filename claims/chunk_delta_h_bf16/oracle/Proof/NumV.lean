import Proof.NumUtil

/-!
# `chunk_delta_h_bf16`: the new values (lines 214–243)

One token, one value column: the chain of `tl.dot`s over the blocks of the state (`b_v`), the
fusable `u - b_v`, and the `bfloat16` rounding of the store. The error is `δ = 0.6·2^-13·Z` before
the store and within `tolV` after it.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- A lifted magnitude is nonnegative. -/
theorem lmB_nonneg (x : BF16) : 0 ≤ lmB x := by
  unfold lmB; exact_mod_cast Device.liftMag_nonneg _ _

/-- A lifted magnitude is at least the magnitude. -/
theorem abs_toReal_le_lmB (x : BF16) : |x.toReal| ≤ lmB x := by
  unfold lmB FloatV.toReal; exact_mod_cast Device.abs_le_liftMag _ _

/-- The lifted magnitude of a zero is `0`. -/
theorem lmB_zero {x : BF16} (h : x.toReal = 0) : lmB x = 0 := by
  have h0 : x.toRat = 0 := by
    unfold FloatV.toReal at h; exact_mod_cast h
  unfold lmB; simp [Device.liftMag, h0]

/-- A lifted magnitude exceeds the magnitude by at most `2^-126`. -/
theorem lmB_le (x : BF16) : lmB x ≤ |x.toReal| + 2 ^ (-126 : ℤ) := by
  unfold lmB FloatV.toReal
  unfold Device.liftMag
  split_ifs with h
  · simp only [Rat.cast_zero]; positivity
  · simp only [Device.liftFloor]
    push_cast
    rw [max_le_iff]
    constructor
    · linarith [abs_nonneg (x.toRat : ℝ), (by positivity : (0:ℝ) < 2 ^ (-126 : ℤ))]
    · linarith [abs_nonneg (x.toRat : ℝ)]

/-- A product's magnitude is at most the product of the lifted magnitudes. -/
theorem abs_mul_le_lmB (x y : BF16) : |x.toReal * y.toReal| ≤ lmB x * lmB y := by
  unfold lmB FloatV.toReal
  have := Device.abs_mul_le_liftMag .bf16 .bf16 x.toRat y.toRat
  exact_mod_cast this


/-- `rD` as a number. -/
theorem rD_eq : rD = 66 * 2 ^ (-22 : ℤ) := by
  unfold rD Device.relDot16; push_cast; norm_num

/-- `aD` as a number. -/
theorem aD_eq : aD = 66 * 2 ^ (-126 : ℤ) := by
  unfold aD Device.absDot16; push_cast; norm_num

/-- A fusable `u + y` of two `float32` values with room: finite, within `3·2^-24·(|u| + |y|) +
4·2^-150` of the exact sum. -/
theorem add_f32 {fusion : Bool} {u y bv : F32} (hu : u.isFinite = true) (hy : y.isFinite = true)
    (hM : |u.toReal| + |y.toReal| ≤ 2 ^ 125) (h : AddRel fusion u y bv) :
    bv.isFinite = true ∧
      |bv.toReal - (u.toReal + y.toReal)| ≤
        3 * 2 ^ (-24 : ℤ) * (|u.toReal| + |y.toReal|) + 4 * 2 ^ (-150 : ℤ) := by
  have hMq : |u.toRat| + |y.toRat| ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have h1 : ((|u.toRat| + |y.toRat| : ℚ) : ℝ) ≤
        ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      push_cast
      rw [Format.binary32_maxVal_real, Format.binary32_eta_real]
      have e1 : ((u.toRat : ℚ) : ℝ) = u.toReal := rfl
      have e2 : ((y.toRat : ℚ) : ℝ) = y.toReal := rfl
      rw [e1, e2]
      have : (2 : ℝ) ^ 125 + 2 * 2 ^ (-150 : ℤ) ≤ ((2 ^ 24 - 1) * 2 ^ 104) / 2 := by norm_num
      linarith
    exact_mod_cast h1
  obtain ⟨hf, he⟩ := FloatV.AddRel.err (f := Format.binary32) Format.binary32_regular
    (by rw [Format.Std.ieee Format.binary32_std]; decide)
    (by rw [Format.binary32_u]; norm_num [zpow_neg]) h hu hy hMq
  refine ⟨hf, ?_⟩
  have h2 := Rat.cast_le (K := ℝ) |>.mpr he
  push_cast at h2
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  have e1 : ((u.toRat : ℚ) : ℝ) = u.toReal := rfl
  have e2 : ((y.toRat : ℚ) : ℝ) = y.toReal := rfl
  have e3 : ((bv.toRat : ℚ) : ℝ) = bv.toReal := rfl
  rw [e1, e2, e3] at h2
  exact h2

/-- A fusable `u - r` of two `float32` values with room: finite, within `3·2^-24·(|u| + |r|) +
4·2^-150` of the exact difference. -/
theorem sub_f32 {fusion : Bool} {u r bv : F32} (hu : u.isFinite = true) (hr : r.isFinite = true)
    (hM : |u.toReal| + |r.toReal| ≤ 2 ^ 125) (h : AddRel fusion u (FloatV.neg r) bv) :
    bv.isFinite = true ∧
      |bv.toReal - (u.toReal - r.toReal)| ≤
        3 * 2 ^ (-24 : ℤ) * (|u.toReal| + |r.toReal|) + 4 * 2 ^ (-150 : ℤ) := by
  have hrn : (FloatV.neg r).isFinite = true := by rw [FloatV.isFinite_neg]; exact hr
  have := add_f32 hu hrn (by rw [FloatV.toReal_neg, abs_neg]; exact hM) h
  rw [FloatV.toReal_neg, abs_neg] at this
  refine ⟨this.1, ?_⟩
  have e : bv.toReal - (u.toReal - r.toReal) = bv.toReal - (u.toReal + -r.toReal) := by ring
  rw [e]; exact this.2

/-- The `bfloat16` rounding of a finite `float32` value in range: finite, within `2^-8·|y| +
2^-134`. -/
theorem cvt_bf16 {y : F32} {x : BF16} (hy : y.isFinite = true) (hb : |y.toReal| ≤ 2 ^ 100)
    (hx : FloatV.NaNEq (FloatV.cvt y) x) :
    x.isFinite = true ∧ |x.toReal - y.toReal| ≤ 2 ^ (-8 : ℤ) * |y.toReal| + 2 ^ (-134 : ℤ) := by
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hy
    (by rw [Format.bfloat16_maxVal_real]; exact hb.trans (by norm_num))
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at h2
  rw [hx.eq_of_isFinite h1]
  exact ⟨h1, by linarith⟩


/-- `√K ≤ 16` for `K ≤ 256`. -/
theorem sqrt_le_sixteen {K : ℕ} (h : K ≤ 256) : Real.sqrt K ≤ 16 := by
  have : (K : ℝ) ≤ 256 := by exact_mod_cast h
  exact Real.sqrt_le_iff.mpr ⟨by norm_num, by nlinarith⟩

/-- The lifted `w` row has norm at most `2^20 + 1` when its entries are at most `2^16`. -/
theorem lm_w_norm {K : ℕ} (hKle : K ≤ 256) (w : ℕ → BF16)
    (hw : ∀ c, c < K → |(w c).toReal| ≤ 2 ^ 16) :
    colNorm K (fun c => lmB (w c)) ≤ 2 ^ 20 + 1 := by
  have h1 : colNorm K (fun c => lmB (w c)) ≤ colNorm K (fun _ => (2 : ℝ) ^ 16 + 2 ^ (-126 : ℤ)) :=
    colNorm_mono_abs (fun i hi => by
      rw [abs_of_nonneg (lmB_nonneg _), abs_of_nonneg (by positivity)]
      linarith [lmB_le (w i), hw i hi])
  rw [colNorm_const, abs_of_nonneg (by positivity)] at h1
  calc _ ≤ _ := h1
    _ ≤ 16 * ((2 : ℝ) ^ 16 + 2 ^ (-126 : ℤ)) :=
        mul_le_mul_of_nonneg_right (sqrt_le_sixteen hKle) (by positivity)
    _ ≤ 2 ^ 20 + 1 := by norm_num

/-- The lifted state column has norm at most `2^40 + 1` when the column has norm at most `2^40`. -/
theorem lm_h_norm {K : ℕ} (hKle : K ≤ 256) (hs : ℕ → BF16)
    (hhn : colNorm K (fun c => (hs c).toReal) ≤ 2 ^ 40) :
    colNorm K (fun c => lmB (hs c)) ≤ 2 ^ 40 + 1 := by
  have h1 : colNorm K (fun c => lmB (hs c)) ≤
      colNorm K (fun c => |(hs c).toReal| + 2 ^ (-126 : ℤ)) :=
    colNorm_mono_abs (fun i _ => by
      rw [abs_of_nonneg (lmB_nonneg _), abs_of_nonneg (by positivity)]
      exact lmB_le (hs i))
  have h2 : colNorm K (fun c => |(hs c).toReal| + 2 ^ (-126 : ℤ)) ≤
      colNorm K (fun c => |(hs c).toReal|) + colNorm K (fun _ => (2 : ℝ) ^ (-126 : ℤ)) :=
    colNorm_add_le K _ _
  have h3 : colNorm K (fun c => |(hs c).toReal|) = colNorm K (fun c => (hs c).toReal) := by
    unfold colNorm; simp [sq_abs]
  rw [colNorm_const, abs_of_nonneg (by positivity), h3] at h2
  have h4 : Real.sqrt K * 2 ^ (-126 : ℤ) ≤ 16 * 2 ^ (-126 : ℤ) :=
    mul_le_mul_of_nonneg_right (sqrt_le_sixteen hKle) (by positivity)
  have h5 : (16 : ℝ) * 2 ^ (-126 : ℤ) ≤ 1 := by norm_num
  linarith

/-- Cauchy–Schwarz for the lifted magnitudes. -/
theorem sum_lm_le (K : ℕ) (w hs : ℕ → BF16) :
    ∑ c ∈ Finset.range K, lmB (w c) * lmB (hs c) ≤
      colNorm K (fun c => lmB (w c)) * colNorm K (fun c => lmB (hs c)) := by
  have := sum_abs_mul_le_colNorm K (fun c => lmB (w c)) (fun c => lmB (hs c))
  simpa [abs_of_nonneg (lmB_nonneg _)] using this

/-- The sum over `64·(b + 1)` terms is the sum over `64·b` terms and the block. -/
theorem sum_block (f : ℕ → ℝ) (b : ℕ) :
    ∑ c ∈ Finset.range (64 * (b + 1)), f c =
      ∑ c ∈ Finset.range (64 * b), f c + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), f c := by
  rw [show 64 * (b + 1) = 64 * b + 64 by ring, Finset.range_eq_Ico, Finset.range_eq_Ico]
  exact (Finset.sum_Ico_consecutive f (Nat.zero_le _) (by omega)).symm

/-- A dot product is at most the sum of the lifted products. -/
theorem abs_sum_le_lm (n : ℕ) (w hs : ℕ → BF16) :
    |∑ c ∈ Finset.range n, (w c).toReal * (hs c).toReal| ≤
      ∑ c ∈ Finset.range n, lmB (w c) * lmB (hs c) :=
  (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun c _ => abs_mul_le_lmB (w c) (hs c))

/-- **The chain of dots.** After `b ≤ nb ≤ 4` blocks the running dot `r b` is finite and within
`b·1.0001·(rD·A + aD)` of the exact sum of the first `64·b` products, `A` an upper bound on the
sum of the lifted products and at most `2^61`; the room condition of each step is discharged. -/
theorem dot_chain {nb : ℕ} (hnb : nb ≤ 4) (w hs : ℕ → BF16) (r : ℕ → F32) {A : ℝ}
    (hA : ∑ c ∈ Finset.range (64 * nb), lmB (w c) * lmB (hs c) ≤ A) (hA1 : A ≤ 2 ^ 61)
    (hr0 : (r 0).isFinite = true ∧ (r 0).toReal = 0)
    (hstep : ∀ b, b < nb → (r b).isFinite = true →
      |(r b).toReal| + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (w c) * lmB (hs c) ≤ 2 ^ 120 →
      (r (b + 1)).isFinite = true ∧
        |(r (b + 1)).toReal - ((r b).toReal +
            ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), (w c).toReal * (hs c).toReal)| ≤
          rD * (|(r b).toReal| + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (w c) * lmB (hs c)) +
            aD) :
    ∀ b, b ≤ nb → (r b).isFinite = true ∧
      |(r b).toReal - ∑ c ∈ Finset.range (64 * b), (w c).toReal * (hs c).toReal| ≤
        b * (10001 / 10000) * (rD * A + aD) := by
  have hrD : rD = 66 * 2 ^ (-22 : ℤ) := rD_eq
  have haD : aD = 66 * 2 ^ (-126 : ℤ) := aD_eq
  have hrD0 : 0 ≤ rD := by rw [hrD]; positivity
  have hrD1 : rD ≤ 1 / 50000 := by rw [hrD]; norm_num
  have haD0 : 0 ≤ aD := by rw [haD]; positivity
  have hlm0 : ∀ c, 0 ≤ lmB (w c) * lmB (hs c) := fun c => mul_nonneg (lmB_nonneg _) (lmB_nonneg _)
  have hA0 : 0 ≤ A := (Finset.sum_nonneg fun c _ => hlm0 c).trans hA
  have hq0 : 0 ≤ rD * A + aD := by positivity
  have hq1 : rD * A + aD ≤ 2 ^ 50 := by
    have : rD * A ≤ 1 / 50000 * 2 ^ 61 := mul_le_mul hrD1 hA1 hA0 (by norm_num)
    have : aD ≤ 1 := by rw [haD]; norm_num
    linarith [(by norm_num : (1 / 50000 * 2 ^ 61 : ℝ) + 1 ≤ 2 ^ 50)]
  intro b
  induction b with
  | zero =>
    intro _
    refine ⟨hr0.1, ?_⟩
    simp [hr0.2]
  | succ b ih =>
    intro hb
    obtain ⟨hf, hE⟩ := ih (by omega)
    have hbr : (b : ℝ) ≤ 3 := by exact_mod_cast (by omega : b ≤ 3)
    have hb0 : (0 : ℝ) ≤ b := Nat.cast_nonneg b
    set g : ℝ := b * (10001 / 10000) * (rD * A + aD) with hg
    have hg0 : 0 ≤ g := by positivity
    have hg1 : g ≤ 2 ^ 53 := by
      have : (b : ℝ) * (10001 / 10000) ≤ 3 * (10001 / 10000) :=
        mul_le_mul_of_nonneg_right hbr (by norm_num)
      calc g ≤ 3 * (10001 / 10000) * (rD * A + aD) := mul_le_mul_of_nonneg_right this hq0
        _ ≤ 3 * (10001 / 10000) * 2 ^ 50 := mul_le_mul_of_nonneg_left hq1 (by norm_num)
        _ ≤ 2 ^ 53 := by norm_num
    have hS := abs_sum_le_lm (64 * b) w hs
    have hsplit : ∑ c ∈ Finset.range (64 * b), lmB (w c) * lmB (hs c) +
        ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (w c) * lmB (hs c) ≤ A := by
      rw [← sum_block]
      refine le_trans ?_ hA
      exact Finset.sum_le_sum_of_subset_of_nonneg
        (Finset.range_mono (by omega)) (fun c _ _ => hlm0 c)
    have hrb : |(r b).toReal| ≤ |∑ c ∈ Finset.range (64 * b), (w c).toReal * (hs c).toReal| + g := by
      have := abs_sub_abs_le_abs_sub (r b).toReal
        (∑ c ∈ Finset.range (64 * b), (w c).toReal * (hs c).toReal)
      linarith
    have hblk0 : 0 ≤ ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (w c) * lmB (hs c) :=
      Finset.sum_nonneg fun c _ => hlm0 c
    obtain ⟨hf', hs'⟩ := hstep b (by omega) hf (by
      have : (2 : ℝ) ^ 53 + 2 ^ 61 ≤ 2 ^ 120 := by norm_num
      linarith)
    refine ⟨hf', ?_⟩
    rw [sum_block]
    have hdec : (r (b + 1)).toReal - (∑ c ∈ Finset.range (64 * b), (w c).toReal * (hs c).toReal +
          ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), (w c).toReal * (hs c).toReal) =
        ((r (b + 1)).toReal - ((r b).toReal +
            ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), (w c).toReal * (hs c).toReal)) +
          ((r b).toReal - ∑ c ∈ Finset.range (64 * b), (w c).toReal * (hs c).toReal) := by ring
    rw [hdec]
    have hmul : rD * (|(r b).toReal| + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64),
        lmB (w c) * lmB (hs c)) ≤ rD * (A + g) :=
      mul_le_mul_of_nonneg_left (by linarith) hrD0
    have hkey : rD * ((b : ℝ) * (10001 / 10000)) ≤ 1 / 10000 := by
      have : (b : ℝ) * (10001 / 10000) ≤ 3 * (10001 / 10000) :=
        mul_le_mul_of_nonneg_right hbr (by norm_num)
      calc rD * ((b : ℝ) * (10001 / 10000)) ≤ 1 / 50000 * (3 * (10001 / 10000)) :=
            mul_le_mul hrD1 this (by positivity) (by norm_num)
        _ ≤ 1 / 10000 := by norm_num
    have hq : rD * ((b : ℝ) * (10001 / 10000)) * (rD * A + aD) ≤ 1 / 10000 * (rD * A + aD) :=
      mul_le_mul_of_nonneg_right hkey hq0
    have hfin : ((b + 1 : ℕ) : ℝ) * (10001 / 10000) * (rD * A + aD) =
        g + 1 / 10000 * (rD * A + aD) + (rD * A + aD) := by
      rw [hg]; push_cast; ring
    rw [hfin]
    calc _ ≤ |(r (b + 1)).toReal - ((r b).toReal +
            ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), (w c).toReal * (hs c).toReal)| +
          |(r b).toReal - ∑ c ∈ Finset.range (64 * b), (w c).toReal * (hs c).toReal| :=
          abs_add_le _ _
      _ ≤ rD * (A + g) + aD + g := by linarith
      _ ≤ _ := by nlinarith


/-- The arithmetic of the subtraction: the chain's error and the fusable `u - r` give the bound
on `bv`. -/
theorem sub_budget {u S r bv WH : ℝ} (hWH : 0 ≤ WH) (hS : |S| ≤ WH)
    (hE : |r - S| ≤ 4 * (10001 / 10000) * (rD * WH + aD))
    (hsub : |bv - (u - r)| ≤ 3 * 2 ^ (-24 : ℤ) * (|u| + |r|) + 4 * 2 ^ (-150 : ℤ)) :
    |bv - (u - S)| ≤ (3 / 5) * 2 ^ (-13 : ℤ) * (|u| + WH) + 2 ^ (-110 : ℤ) := by
  rw [rD_eq, aD_eq] at hE
  have hr : |r| ≤ WH + |r - S| := by
    have := abs_sub_abs_le_abs_sub r S
    linarith
  have hd : |bv - (u - S)| ≤ |bv - (u - r)| + |r - S| := by
    have : bv - (u - S) = (bv - (u - r)) - (r - S) := by ring
    rw [this]; exact abs_sub _ _
  have hu0 := abs_nonneg u
  norm_num at hE hsub ⊢
  nlinarith

/-- The arithmetic of the store: the `bfloat16` rounding of `bv` is within the claim's tolerance
of `vref`. -/
theorem store_budget {vref bv x Z : ℝ} (hvr : |vref| ≤ Z)
    (hbv : |bv - vref| ≤ (3 / 5) * 2 ^ (-13 : ℤ) * Z + 2 ^ (-110 : ℤ))
    (hx : |x - bv| ≤ 2 ^ (-8 : ℤ) * |bv| + 2 ^ (-134 : ℤ)) :
    |x - vref| ≤ 2 ^ (-8 : ℤ) * vAbsA vref Z + 2 ^ (-100 : ℤ) := by
  unfold vAbsA
  have hb : |bv| ≤ |vref| + |bv - vref| := by
    have := abs_add_le vref (bv - vref)
    simpa using this
  have hd : |x - vref| ≤ |x - bv| + |bv - vref| := by
    have : x - vref = (x - bv) + (bv - vref) := by ring
    rw [this]; exact abs_add_le _ _
  have hZ0 : 0 ≤ Z := (abs_nonneg _).trans hvr
  norm_num at hbv hx hd hb ⊢
  nlinarith

/-- **The new value of one token.** `nb` blocks (`K ≤ 64·nb < K + 64`); `w c` the loaded `w` row
(zero past `K`), `hs c` the `bfloat16` state rounded for the dots (any finite value past `K`); the
chain `r 0 = 0, r 1, …, r nb` of dot results (each from the previous, `hstep`: the `Device.DotBF16`
fact, whose room condition the proof discharges), `bv` the fusable `u - r nb` (`hsub`), `x` its
`bfloat16` rounding. With `vref = u - Σ_c w_c·hs_c` and `Z = |u| + ‖liftMag w‖·‖liftMag hs‖`:
`|vref| ≤ Z ≤ 2^61`, `bv` is within `0.6·2^-13·Z + 2^-110` of `vref`, and `x` is finite and
within `2^-8·vAbs + 2^-100` of `vref`. -/
theorem numV {K nb : ℕ} (hK : 0 < K) (hKle : K ≤ 256) (hnb : K ≤ 64 * nb) (hnb' : 64 * nb < K + 64)
    (w hs : ℕ → BF16) (u : F32) (r : ℕ → F32) (bv : F32) (x : BF16) {fusion : Bool}
    (hw : ∀ c, c < 64 * nb → (w c).isFinite = true ∧ |(w c).toReal| ≤ 2 ^ 16)
    (hwz : ∀ c, K ≤ c → c < 64 * nb → (w c).toReal = 0)
    (hhs : ∀ c, c < 64 * nb → (hs c).isFinite = true)
    (hhn : colNorm K (fun c => (hs c).toReal) ≤ 2 ^ 40)
    (hu : u.isFinite = true ∧ |u.toReal| ≤ 2 ^ 16)
    (hr0 : (r 0).isFinite = true ∧ (r 0).toReal = 0)
    (hstep : ∀ b, b < nb → (r b).isFinite = true →
      |(r b).toReal| + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (w c) * lmB (hs c) ≤ 2 ^ 120 →
      (r (b + 1)).isFinite = true ∧
        |(r (b + 1)).toReal - ((r b).toReal +
            ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), (w c).toReal * (hs c).toReal)| ≤
          rD * (|(r b).toReal| + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (w c) * lmB (hs c)) + aD)
    (hsub : AddRel fusion u (FloatV.neg (r nb)) bv) (hx : FloatV.NaNEq (FloatV.cvt bv) x) :
    |u.toReal - ∑ c ∈ Finset.range K, (w c).toReal * (hs c).toReal| ≤
        |u.toReal| + colNorm K (fun c => lmB (w c)) * colNorm K (fun c => lmB (hs c)) ∧
      |u.toReal| + colNorm K (fun c => lmB (w c)) * colNorm K (fun c => lmB (hs c)) ≤ 2 ^ 61 ∧
      bv.isFinite = true ∧
      |bv.toReal - (u.toReal - ∑ c ∈ Finset.range K, (w c).toReal * (hs c).toReal)| ≤
        (3 / 5) * 2 ^ (-13 : ℤ) *
          (|u.toReal| + colNorm K (fun c => lmB (w c)) * colNorm K (fun c => lmB (hs c))) +
          2 ^ (-110 : ℤ) ∧
      x.isFinite = true ∧
      |x.toReal - (u.toReal - ∑ c ∈ Finset.range K, (w c).toReal * (hs c).toReal)| ≤
        2 ^ (-8 : ℤ) * vAbsA (u.toReal - ∑ c ∈ Finset.range K, (w c).toReal * (hs c).toReal)
          (|u.toReal| + colNorm K (fun c => lmB (w c)) * colNorm K (fun c => lmB (hs c))) +
          2 ^ (-100 : ℤ) := by
  have _hfin := hhs
  have hnb4 : nb ≤ 4 := by omega
  set WH := colNorm K (fun c => lmB (w c)) * colNorm K (fun c => lmB (hs c)) with hWHdef
  have hW := lm_w_norm hKle w (fun c hc => (hw c (by omega)).2)
  have hH := lm_h_norm hKle hs hhn
  have hW0 : 0 ≤ colNorm K (fun c => lmB (w c)) := colNorm_nonneg _ _
  have hH0 : 0 ≤ colNorm K (fun c => lmB (hs c)) := colNorm_nonneg _ _
  have hWH0 : 0 ≤ WH := mul_nonneg hW0 hH0
  have hWH1 : WH ≤ 2 ^ 60 + 2 ^ 41 := by
    calc WH ≤ (2 ^ 20 + 1) * (2 ^ 40 + 1) := mul_le_mul hW hH hH0 (by positivity)
      _ ≤ 2 ^ 60 + 2 ^ 41 := by norm_num
  have hZ : |u.toReal| + WH ≤ 2 ^ 61 := by
    have : (2 : ℝ) ^ 16 + (2 ^ 60 + 2 ^ 41) ≤ 2 ^ 61 := by norm_num
    linarith [hu.2]
  have hAeq : ∑ c ∈ Finset.range (64 * nb), lmB (w c) * lmB (hs c) =
      ∑ c ∈ Finset.range K, lmB (w c) * lmB (hs c) := by
    symm
    refine Finset.sum_subset (Finset.range_mono hnb) (fun c hc hnot => ?_)
    have hcK : K ≤ c := by simpa using hnot
    rw [lmB_zero (hwz c hcK (by simpa using hc)), zero_mul]
  have hAWH : ∑ c ∈ Finset.range (64 * nb), lmB (w c) * lmB (hs c) ≤ WH := by
    rw [hAeq]; exact sum_lm_le K w hs
  have hSeq : ∑ c ∈ Finset.range (64 * nb), (w c).toReal * (hs c).toReal =
      ∑ c ∈ Finset.range K, (w c).toReal * (hs c).toReal := by
    symm
    refine Finset.sum_subset (Finset.range_mono hnb) (fun c hc hnot => ?_)
    have hcK : K ≤ c := by simpa using hnot
    rw [hwz c hcK (by simpa using hc), zero_mul]
  obtain ⟨hrf, hrE⟩ := dot_chain hnb4 w hs r hAWH (hWH1.trans (by norm_num)) hr0 hstep nb le_rfl
  rw [hSeq] at hrE
  set S := ∑ c ∈ Finset.range K, (w c).toReal * (hs c).toReal with hSdef
  have hSWH : |S| ≤ WH := (abs_sum_le_lm K w hs).trans (sum_lm_le K w hs)
  have hnbR : (nb : ℝ) ≤ 4 := by exact_mod_cast hnb4
  have hq0 : 0 ≤ rD * WH + aD := by
    rw [rD_eq, aD_eq]; positivity
  have hE : |(r nb).toReal - S| ≤ 4 * (10001 / 10000) * (rD * WH + aD) :=
    hrE.trans (mul_le_mul_of_nonneg_right
      (mul_le_mul_of_nonneg_right hnbR (by norm_num)) hq0)
  have hEsmall : 4 * (10001 / 10000) * (rD * WH + aD) ≤ 2 ^ 50 := by
    rw [rD_eq, aD_eq]
    have : WH ≤ 2 ^ 61 := hWH1.trans (by norm_num)
    norm_num
    nlinarith
  have hr1 : |(r nb).toReal| ≤ 2 ^ 62 := by
    have := abs_sub_abs_le_abs_sub (r nb).toReal S
    have : (2 : ℝ) ^ 61 + 2 ^ 50 ≤ 2 ^ 62 := by norm_num
    linarith [hWH1.trans (by norm_num : (2 : ℝ) ^ 60 + 2 ^ 41 ≤ 2 ^ 61)]
  obtain ⟨hbf, hbe⟩ := sub_f32 hu.1 hrf (by
    have : (2 : ℝ) ^ 16 + 2 ^ 62 ≤ 2 ^ 100 := by norm_num
    linarith [hu.2]) hsub
  have hbv := sub_budget hWH0 hSWH hE hbe
  have hvr : |u.toReal - S| ≤ |u.toReal| + WH := by
    calc |u.toReal - S| ≤ |u.toReal| + |S| := abs_sub _ _
      _ ≤ _ := by linarith
  have hbb : |bv.toReal| ≤ 2 ^ 100 := by
    have h1 := abs_add_le (u.toReal - S) (bv.toReal - (u.toReal - S))
    have h2 : |bv.toReal - (u.toReal - S)| ≤ 2 ^ 80 := by
      refine hbv.trans ?_
      have : (3 / 5 : ℝ) * 2 ^ (-13 : ℤ) * (|u.toReal| + WH) ≤ 3 / 5 * 2 ^ (-13 : ℤ) * 2 ^ 61 :=
        mul_le_mul_of_nonneg_left hZ (by positivity)
      norm_num at this ⊢
      linarith
    have : (u.toReal - S) + (bv.toReal - (u.toReal - S)) = bv.toReal := by ring
    rw [this] at h1
    have : (2 : ℝ) ^ 61 + 2 ^ 80 ≤ 2 ^ 100 := by norm_num
    linarith
  obtain ⟨hxf, hxe⟩ := cvt_bf16 hbf hbb hx
  exact ⟨hvr, hZ, hbf, hbv, hxf, store_budget hvr hbv hxe⟩

end Claims.ChunkDeltaH
