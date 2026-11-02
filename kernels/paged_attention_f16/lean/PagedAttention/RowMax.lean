import PagedAttention.Lane

/-!
# `paged_attention_f16`: one program's row, the scores, the running maximum and the exponents

For one program (sequence `s`, query head `h`) the analysis tracks the sequence's tokens
`j < n` among the tokens of the pages processed so far, `j < P·t`. `RowCtx` holds the row's data:
`scale` (`c`, the double's value), the exact scores `sv j`, their magnitudes `bv j`, the bounds `Y`
(on `c·bv j`) and `E` (on each exponent's error), `n` and `P`.

* `score_lane`: one lane's score `tl.sum(k * q[None, :]) * scale` (line 61) against `c·sv`.
* `MInv`: the running maximum `m_i` after `t ≥ 1` pages is within `E` of some token's `c·sv_j`,
  and every token's `c·sv_j` is at most `m_i + E`.
* `mstep`: one page's new maximum `m_new` (line 63), each lane's exponent `s − m_new` (line 65,
  within `E` of `c·sv_j − m_new`, or `−∞` on a masked lane), and the input of `alpha` (line 64).
-/

namespace Kernels.PagedAttention

open TVA TVA.Math.Float

/-- One program's row data. -/
structure RowCtx where
  /-- `scale` (the double), as a real number. -/
  c : ℝ
  /-- The bound on `c·bv j` over the sequence's tokens. -/
  Y : ℝ
  /-- The bound on each exponent's error. -/
  E : ℝ
  /-- The relative error of a score's row sum, `rhoD HEAD_DIM`. -/
  rhoD : ℝ
  /-- The exact scores `q·k_j`. -/
  sv : ℕ → ℝ
  /-- Their magnitudes `Σ_e |q·k_j|`. -/
  bv : ℕ → ℝ
  /-- The sequence's length. -/
  n : ℕ
  /-- `PAGE_SIZE`. -/
  P : ℕ

/-- What the analysis needs of a row's data. -/
structure RowCtx.Ok (R : RowCtx) : Prop where
  c_lo : 1 / 2 ^ 64 ≤ R.c
  c_hi : R.c ≤ 2 ^ 64
  Y_nonneg : 0 ≤ R.Y
  Y_le : R.Y ≤ 2 ^ 10
  rhoD_nonneg : 0 ≤ R.rhoD
  rhoD_le : R.rhoD ≤ 1 / 2 ^ 14
  E_eq : R.E = (R.rhoD + 8 / 2 ^ 24) * R.Y + 1 / 2 ^ 50
  sv_le : ∀ j, |R.sv j| ≤ R.bv j
  al_le : ∀ j, j < R.n → R.c * R.bv j ≤ R.Y
  P_pos : 0 < R.P
  P_le : R.P ≤ 256

theorem RowCtx.Ok.c_pos {R : RowCtx} (h : R.Ok) : 0 < R.c :=
  lt_of_lt_of_le (by norm_num) h.c_lo

theorem RowCtx.Ok.E_nonneg {R : RowCtx} (h : R.Ok) : 0 ≤ R.E := by
  rw [h.E_eq]; have := h.Y_nonneg; have := h.rhoD_nonneg; positivity

theorem RowCtx.Ok.E_le_Y {R : RowCtx} (h : R.Ok) : R.E ≤ R.Y / 2 ^ 13 + 1 / 2 ^ 50 := by
  rw [h.E_eq]
  have := h.Y_nonneg
  have h1 : R.rhoD + 8 / 2 ^ 24 ≤ 1 / 2 ^ 13 := by have := h.rhoD_le; linarith
  nlinarith

theorem RowCtx.Ok.E_le {R : RowCtx} (h : R.Ok) : R.E ≤ 1 / 8 := by
  rw [h.E_eq]
  have h1 := h.rhoD_le
  have h2 := h.Y_le
  have h3 := h.Y_nonneg
  have h4 := h.rhoD_nonneg
  have : (R.rhoD + 8 / 2 ^ 24) * R.Y ≤ (1 / 2 ^ 14 + 8 / 2 ^ 24) * 2 ^ 10 :=
    mul_le_mul (by linarith) h2 h3 (by norm_num)
  norm_num at this ⊢
  linarith

/-- The error of a lane's score `s` (line 61) against `c·sv`. -/
noncomputable def RowCtx.Es (R : RowCtx) : ℝ := (R.rhoD + 3 / 2 ^ 24) * R.Y + 1 / 2 ^ 70

theorem RowCtx.Ok.Es_le {R : RowCtx} (h : R.Ok) : R.Es ≤ R.E := by
  unfold RowCtx.Es; rw [h.E_eq]; have := h.Y_nonneg; nlinarith

theorem RowCtx.Ok.Es_nonneg {R : RowCtx} (h : R.Ok) : 0 ≤ R.Es := by
  unfold RowCtx.Es; have := h.Y_nonneg; have := h.rhoD_nonneg; positivity

/-- A token's `c·sv` is at most `Y` in magnitude. -/
theorem RowCtx.Ok.csv_le {R : RowCtx} (h : R.Ok) {j : ℕ} (hj : j < R.n) : |R.c * R.sv j| ≤ R.Y := by
  rw [abs_mul, abs_of_pos h.c_pos]
  exact (mul_le_mul_of_nonneg_left (h.sv_le j) h.c_pos.le).trans (h.al_le j hj)

/-! ## A lane's score (line 61) -/

/-- **A lane's row sum** (line 61): the rounded products `pr` of `D ≤ 256` pairs of `float16`
values (as `float32`), added along a tree of fusable additions, against the exact sum `S` of the
products, relative to `B = Σ|k·q|`. -/
theorem row_sum_err {D : ℕ} (hD : D ≤ 256) {fusion : Bool} {kf qf pr : Fin D → F32} {srow : F32}
    (hkf : ∀ e, (kf e).isFinite = true) (hqf : ∀ e, (qf e).isFinite = true)
    (hkb : ∀ e, |(kf e).toReal| ≤ 65504) (hqb : ∀ e, |(qf e).toReal| ≤ 65504)
    (hpr : ∀ e, FloatV.NaNEq (FloatV.mul (kf e) (qf e)) (pr e))
    (hsum : ∃ tr : Tree, tr.Covers D ∧ tr.Eval (AddRel fusion) (laneSeq pr) srow)
    (hB : ∑ e, |(kf e).toReal * (qf e).toReal| ≤ 2 ^ 74) :
    srow.isFinite = true ∧
      |srow.toReal - ∑ e, (kf e).toReal * (qf e).toReal| ≤
        ((1 + 1 / 2 ^ 24) * (1 + 3 / 2 ^ 24) ^ (D - 1) - 1) * ∑ e, |(kf e).toReal * (qf e).toReal| +
          1 / 2 ^ 139 := by
  set B := ∑ e, |(kf e).toReal * (qf e).toReal| with hBd
  set S := ∑ e, (kf e).toReal * (qf e).toReal with hSd
  have hB0 : 0 ≤ B := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hprl : ∀ e, (pr e).isFinite = true ∧
      |(pr e).toReal - (kf e).toReal * (qf e).toReal| ≤ |(kf e).toReal * (qf e).toReal| / 2 ^ 24 + 1 / 2 ^ 150 := by
    intro e
    refine f32_mul (hpr e) (hkf e) (hqf e) ?_
    rw [abs_mul]
    have := mul_le_mul (hkb e) (hqb e) (abs_nonneg _) (by norm_num)
    linarith
  have hprb : ∀ e, |(pr e).toReal| ≤ (1 + 1 / 2 ^ 24) * |(kf e).toReal * (qf e).toReal| + 1 / 2 ^ 150 := by
    intro e
    have := abs_sub_abs_le_abs_sub (pr e).toReal ((kf e).toReal * (qf e).toReal)
    have := (hprl e).2
    linarith
  have hA : ∑ e, |(pr e).toReal| ≤ (1 + 1 / 2 ^ 24) * B + D / 2 ^ 150 := by
    calc ∑ e, |(pr e).toReal| ≤ ∑ e, ((1 + 1 / 2 ^ 24) * |(kf e).toReal * (qf e).toReal| + 1 / 2 ^ 150) :=
          Finset.sum_le_sum fun e _ => hprb e
      _ = (1 + 1 / 2 ^ 24) * B + D / 2 ^ 150 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
            Fintype.card_fin, nsmul_eq_mul]; ring
  have hDr : (D : ℝ) ≤ 256 := by exact_mod_cast hD
  have hD150 : (D : ℝ) / 2 ^ 150 ≤ 256 / 2 ^ 150 := div_le_div_of_nonneg_right hDr (by norm_num)
  have hA1 : ∑ e, |(pr e).toReal| ≤ 2 ^ 100 := by
    have : (1 + 1 / 2 ^ 24) * B ≤ (1 + 1 / 2 ^ 24) * 2 ^ 74 := mul_le_mul_of_nonneg_left hB (by norm_num)
    norm_num at this hD150 ⊢
    linarith
  obtain ⟨hsf, hse⟩ := laneSumc_err hD hsum (fun e => (hprl e).1) hA1
  refine ⟨hsf, ?_⟩
  set K := ((1 : ℝ) + 3 / 2 ^ 24) ^ (D - 1) with hK
  have hK1 : 1 ≤ K := one_le_pow₀ (by norm_num)
  have hK2 : K ≤ 1 + 1 / 2 ^ 12 := pow255_le hD
  have hn1 : ((D - 1 : ℕ) : ℝ) ≤ 256 := by exact_mod_cast (show D - 1 ≤ 256 by omega)
  have hsum_pr : |∑ e, (pr e).toReal - S| ≤ B / 2 ^ 24 + D / 2 ^ 150 := by
    rw [hSd, ← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ e, |(pr e).toReal - (kf e).toReal * (qf e).toReal|
        ≤ ∑ e, (|(kf e).toReal * (qf e).toReal| / 2 ^ 24 + 1 / 2 ^ 150) :=
          Finset.sum_le_sum fun e _ => (hprl e).2
      _ = B / 2 ^ 24 + D / 2 ^ 150 := by
          rw [Finset.sum_add_distrib, ← Finset.sum_div, Finset.sum_const, Finset.card_univ,
            Fintype.card_fin, nsmul_eq_mul]; ring
  have t := abs_sub_le srow.toReal (∑ e, (pr e).toReal) S
  have k1 : (K - 1) * ∑ e, |(pr e).toReal| ≤ (K - 1) * ((1 + 1 / 2 ^ 24) * B + D / 2 ^ 150) :=
    mul_le_mul_of_nonneg_left hA (by linarith)
  have k2 : 4 * ((D - 1 : ℕ) : ℝ) / 2 ^ 150 * K ≤ 4 * 256 / 2 ^ 150 * (1 + 1 / 2 ^ 12) := by
    apply mul_le_mul _ hK2 (by linarith) (by norm_num)
    apply div_le_div_of_nonneg_right _ (by norm_num); linarith
  have k3 : (K - 1) * (D / 2 ^ 150) ≤ 1 / 2 ^ 12 * (256 / 2 ^ 150) :=
    mul_le_mul (by linarith) hD150 (by positivity) (by norm_num)
  have e1 : (K - 1) * ((1 + 1 / 2 ^ 24) * B + D / 2 ^ 150) + B / 2 ^ 24 =
      ((1 + 1 / 2 ^ 24) * K - 1) * B + (K - 1) * (D / 2 ^ 150) := by ring
  norm_num at k2 k3 hD150 ⊢
  linarith

/-- **One lane's score**: `tl.sum(k * q[None, :], axis=1) * scale` for one row `r` of the page:
the row sum `srow` (`row_sum_err`) times the `float32` scale `c32` (within one rounding of `c`). -/
theorem score_lane {D : ℕ} (hD : D ≤ 256) {fusion : Bool} {kf qf pr : Fin D → F32}
    {srow c32 sc : F32} {c : ℝ} (hc0 : 1 / 2 ^ 64 ≤ c) (hc1 : c ≤ 2 ^ 64)
    (hkf : ∀ e, (kf e).isFinite = true) (hqf : ∀ e, (qf e).isFinite = true)
    (hkb : ∀ e, |(kf e).toReal| ≤ 65504) (hqb : ∀ e, |(qf e).toReal| ≤ 65504)
    (hpr : ∀ e, FloatV.NaNEq (FloatV.mul (kf e) (qf e)) (pr e))
    (hsum : ∃ tr : Tree, tr.Covers D ∧ tr.Eval (AddRel fusion) (laneSeq pr) srow)
    (hc32f : c32.isFinite = true) (hc32 : |c32.toReal - c| ≤ c / 2 ^ 24 + 1 / 2 ^ 150)
    (hsc : FloatV.NaNEq (FloatV.mul srow c32) sc)
    (hB : c * ∑ e, |(kf e).toReal * (qf e).toReal| ≤ 2 ^ 10) :
    sc.isFinite = true ∧
      |sc.toReal - c * ∑ e, (kf e).toReal * (qf e).toReal| ≤
        ((1 + 1 / 2 ^ 24) * (1 + 3 / 2 ^ 24) ^ (D - 1) - 1 + 3 / 2 ^ 24) *
          (c * ∑ e, |(kf e).toReal * (qf e).toReal|) + 1 / 2 ^ 70 := by
  have hc : 0 < c := lt_of_lt_of_le (by norm_num) hc0
  set B := ∑ e, |(kf e).toReal * (qf e).toReal| with hBd
  set S := ∑ e, (kf e).toReal * (qf e).toReal with hSd
  have hB0 : 0 ≤ B := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hBle : B ≤ 2 ^ 74 := by
    have h1 : 1 / 2 ^ 64 * B ≤ c * B := mul_le_mul_of_nonneg_right hc0 hB0
    have h2 : (1 : ℝ) / 2 ^ 64 * B ≤ 2 ^ 10 := by linarith
    have e : B = 2 ^ 64 * (1 / 2 ^ 64 * B) := by ring
    rw [e]
    have := mul_le_mul_of_nonneg_left h2 (by norm_num : (0 : ℝ) ≤ 2 ^ 64)
    norm_num at this ⊢
    linarith
  obtain ⟨hsf, hrow⟩ := row_sum_err hD hkf hqf hkb hqb hpr hsum hBle
  set K := ((1 : ℝ) + 3 / 2 ^ 24) ^ (D - 1) with hK
  have hK1 : 1 ≤ K := one_le_pow₀ (by norm_num)
  have hK2 : K ≤ 1 + 1 / 2 ^ 12 := pow255_le hD
  set r := (1 + 1 / 2 ^ 24) * K - 1 with hr
  have hr0 : 0 ≤ r := by rw [hr]; nlinarith
  have hr1 : r ≤ 1 / 2 ^ 11 := by rw [hr]; nlinarith
  have hSB : |S| ≤ B := Finset.abs_sum_le_sum_abs _ _
  have hsrow : |srow.toReal| ≤ (1 + r) * B + 1 / 2 ^ 139 := by
    have := abs_sub_abs_le_abs_sub srow.toReal S
    have e : (1 + r) * B = B + r * B := by ring
    linarith
  set X := c * B with hX
  have hcB0 : 0 ≤ X := mul_nonneg hc.le hB0
  have hrr : r * X ≤ 1 / 2 ^ 11 * X := mul_le_mul_of_nonneg_right hr1 hcB0
  have hrr0 : 0 ≤ r * X := mul_nonneg hr0 hcB0
  -- `|srow|·c` and `|srow|`
  have hsrow_c : |srow.toReal| * c ≤ X + r * X + 1 / 2 ^ 75 := by
    have k1 := mul_le_mul_of_nonneg_right hsrow hc.le
    have k2 : 1 / 2 ^ 139 * c ≤ 1 / 2 ^ 139 * 2 ^ 64 := mul_le_mul_of_nonneg_left hc1 (by norm_num)
    have e : ((1 + r) * B + 1 / 2 ^ 139) * c = X + r * X + 1 / 2 ^ 139 * c := by rw [hX]; ring
    norm_num at k2 ⊢
    linarith
  have hrB : r * B ≤ 1 / 2 ^ 11 * 2 ^ 74 := mul_le_mul hr1 hBle hB0 (by norm_num)
  have hsrow_e : |srow.toReal| ≤ 2 ^ 76 := by
    have e : (1 + r) * B = B + r * B := by ring
    norm_num at hrB ⊢
    linarith
  have q3 : |srow.toReal| * (1 / 2 ^ 150) ≤ 2 ^ 76 * (1 / 2 ^ 150) :=
    mul_le_mul_of_nonneg_right hsrow_e (by norm_num)
  have hc32b : |c32.toReal| ≤ (1 + 1 / 2 ^ 24) * c + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub c32.toReal c
    rw [abs_of_pos hc] at this; linarith
  have hsc32 : |srow.toReal * c32.toReal| ≤
      (1 + 1 / 2 ^ 24) * (X + r * X + 1 / 2 ^ 75) + 2 ^ 76 * (1 / 2 ^ 150) := by
    rw [abs_mul]
    have q1 : |srow.toReal| * |c32.toReal| ≤ |srow.toReal| * ((1 + 1 / 2 ^ 24) * c + 1 / 2 ^ 150) :=
      mul_le_mul_of_nonneg_left hc32b (abs_nonneg _)
    have q2 : (1 + 1 / 2 ^ 24) * (|srow.toReal| * c) ≤ (1 + 1 / 2 ^ 24) * (X + r * X + 1 / 2 ^ 75) :=
      mul_le_mul_of_nonneg_left hsrow_c (by norm_num)
    have e : |srow.toReal| * ((1 + 1 / 2 ^ 24) * c + 1 / 2 ^ 150) =
        (1 + 1 / 2 ^ 24) * (|srow.toReal| * c) + |srow.toReal| * (1 / 2 ^ 150) := by ring
    linarith
  have hroom : |srow.toReal * c32.toReal| ≤ 2 ^ 120 := by
    norm_num at hsc32 hrr ⊢
    linarith
  obtain ⟨hscf, hsce⟩ := f32_mul hsc hsf hc32f hroom
  refine ⟨hscf, ?_⟩
  have p1 : |sc.toReal - srow.toReal * c32.toReal| ≤
      ((1 + 1 / 2 ^ 24) * (X + r * X + 1 / 2 ^ 75) + 2 ^ 76 * (1 / 2 ^ 150)) / 2 ^ 24 + 1 / 2 ^ 150 := by
    refine hsce.trans ?_
    have := div_le_div_of_nonneg_right hsc32 (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
    linarith
  have p2 : |srow.toReal * c32.toReal - srow.toReal * c| ≤
      1 / 2 ^ 24 * (X + r * X + 1 / 2 ^ 75) + 2 ^ 76 * (1 / 2 ^ 150) := by
    rw [← mul_sub, abs_mul]
    have k : |srow.toReal| * |c32.toReal - c| ≤ |srow.toReal| * (c / 2 ^ 24 + 1 / 2 ^ 150) :=
      mul_le_mul_of_nonneg_left hc32 (abs_nonneg _)
    have k2 : 1 / 2 ^ 24 * (|srow.toReal| * c) ≤ 1 / 2 ^ 24 * (X + r * X + 1 / 2 ^ 75) :=
      mul_le_mul_of_nonneg_left hsrow_c (by norm_num)
    have e : |srow.toReal| * (c / 2 ^ 24 + 1 / 2 ^ 150) =
        1 / 2 ^ 24 * (|srow.toReal| * c) + |srow.toReal| * (1 / 2 ^ 150) := by ring
    linarith
  have p3 : |srow.toReal * c - c * S| ≤ r * X + 1 / 2 ^ 75 := by
    rw [show srow.toReal * c - c * S = c * (srow.toReal - S) by ring, abs_mul, abs_of_pos hc]
    have k := mul_le_mul_of_nonneg_left hrow hc.le
    have k2 : c * (1 / 2 ^ 139) ≤ 2 ^ 64 * (1 / 2 ^ 139) := mul_le_mul_of_nonneg_right hc1 (by norm_num)
    have e : c * (r * B + 1 / 2 ^ 139) = r * X + c * (1 / 2 ^ 139) := by rw [hX]; ring
    norm_num at k2 ⊢
    linarith
  have tri : |sc.toReal - c * S| ≤ |sc.toReal - srow.toReal * c32.toReal| +
      |srow.toReal * c32.toReal - srow.toReal * c| + |srow.toReal * c - c * S| := by
    have := abs_sub_le sc.toReal (srow.toReal * c32.toReal) (c * S)
    have := abs_sub_le (srow.toReal * c32.toReal) (srow.toReal * c) (c * S)
    linarith
  have goal_eq : (r + 3 / 2 ^ 24) * X = r * X + 3 / 2 ^ 24 * X := by ring
  show |sc.toReal - c * S| ≤ (r + 3 / 2 ^ 24) * X + 1 / 2 ^ 70
  rw [goal_eq]
  norm_num at p1 p2 p3 hrr ⊢
  linarith

/-! ## The running maximum -/

/-- The running maximum after `t ≥ 1` pages. -/
structure MInv (R : RowCtx) (t : ℕ) (m : F32) : Prop where
  fin : m.isFinite = true
  bound : |m.toReal| ≤ R.Y + R.E
  attained : ∃ j, j < R.P * t ∧ j < R.n ∧ |m.toReal - R.c * R.sv j| ≤ R.E
  above : ∀ j, j < R.P * t → j < R.n → R.c * R.sv j ≤ m.toReal + R.E

/-- The input `m_i − m_new` of `alpha` (line 64): `−∞` before the first page; afterwards a finite
value within one rounding of `m_i − m_new ≤ 0`. -/
def AlphaIn (t : ℕ) (m m' dm : F32) : Prop :=
  (t = 0 ∧ dm = FloatV.negInf) ∨
  (0 < t ∧ m.isFinite = true ∧ m.toReal ≤ m'.toReal ∧ dm.isFinite = true ∧
    |dm.toReal - (m.toReal - m'.toReal)| ≤ (m'.toReal - m.toReal) / 2 ^ 24 + 1 / 2 ^ 150)

/-- Each lane's exponent (line 65): for a token `j = P·t + r`, finite and within `E` of
`c·sv_j − m_new`; for a masked lane, `−∞`. -/
def LaneX (R : RowCtx) (t : ℕ) (m' : F32) (x : Fin R.P → F32) : Prop :=
  ∀ r : Fin R.P, (R.P * t + r < R.n → (x r).isFinite = true ∧
      |(x r).toReal - (R.c * R.sv (R.P * t + r) - m'.toReal)| ≤ R.E) ∧
    (¬ R.P * t + r < R.n → x r = FloatV.negInf)

theorem le_toReal {x y : F32} (hx : x.isNaN = false) (hy : y.isNaN = false) (h : FloatV.le x y = true) :
    x.toReal ≤ y.toReal := by
  have := (FloatV.le_iff hx hy).mp h
  simp only [toReal_def]; exact_mod_cast this

/-- `tl.maximum` of non-NaN values: one of them, not below either. -/
theorem maxRel_facts {x y r : F32} (h : FloatV.MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : (r = x ∨ r = y) ∧ x.toReal ≤ r.toReal ∧ y.toReal ≤ r.toReal := by
  have hmem := h.mem hx hy
  have hr : r.isNaN = false := by rcases hmem with rfl | rfl <;> assumption
  obtain ⟨h1, h2⟩ := h.le hx hy
  exact ⟨hmem, le_toReal hx hr h1, le_toReal hy hr h2⟩

theorem negInf32_isNaN : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN (by decide)

theorem negInf32_not_fin : (FloatV.negInf : F32).isFinite = false := by
  have h := FloatV.negInf_isInf (f := Format.binary32) (by decide)
  cases hf : (FloatV.negInf : F32).isFinite
  · rfl
  · have := (FloatV.isFinite_iff.mp hf).2; rw [h] at this; cases this

/-- A key `j < P·(t+1)` lies in the first `t` pages or in page `t`. -/
theorem key_split {P t j : ℕ} (h : j < P * (t + 1)) : j < P * t ∨ ∃ r : Fin P, j = P * t + r := by
  by_cases hj : j < P * t
  · exact Or.inl hj
  · refine Or.inr ⟨⟨j - P * t, ?_⟩, ?_⟩
    · rw [Nat.mul_succ] at h; omega
    · simp only; omega

/-- **`tl.max` of a page's lanes** (any tree): the lanes are finite on the tokens and `−∞` on
the masked lanes, and lane `0` is a token. The maximum is a token's lane, finite, and not below
any token's lane. -/
theorem maxP {R : RowCtx} {t : ℕ} {sc : Fin R.P → F32} {mx : F32}
    (h : ∃ tr : Tree, tr.Covers R.P ∧ tr.Eval (FloatV.MaxRel true) (laneSeq sc) mx)
    (hS : ∀ r : Fin R.P, (R.P * t + r < R.n → (sc r).isFinite = true) ∧
      (¬ R.P * t + r < R.n → sc r = FloatV.negInf))
    (r0 : Fin R.P) (h0 : R.P * t + r0 < R.n) :
    ∃ r1 : Fin R.P, R.P * t + r1 < R.n ∧ mx = sc r1 ∧
      ∀ r : Fin R.P, R.P * t + r < R.n → (sc r).toReal ≤ mx.toReal := by
  obtain ⟨tr, hcov, hev⟩ := h
  have hmem : ∀ {i : ℕ}, i ∈ tr.leaves ↔ i < R.P := fun {i} => by rw [hcov.mem_iff, List.mem_range]
  have hls : ∀ i (hi : i < R.P), laneSeq sc i = sc ⟨i, hi⟩ := fun i hi => by simp [laneSeq, hi]
  have hnan : ∀ r : Fin R.P, (sc r).isNaN = false := fun r => by
    by_cases hr : R.P * t + r < R.n
    · exact isNaN_of_fin ((hS r).1 hr)
    · rw [(hS r).2 hr]; exact negInf32_isNaN
  obtain ⟨⟨i, hi, hmi⟩, hle⟩ := Tree.eval_maxRel hev fun i hi => by
    rw [hls i (hmem.mp hi)]; exact hnan _
  have hi' := hmem.mp hi
  rw [hls i hi'] at hmi
  have hle' : ∀ r : Fin R.P, FloatV.le (sc r) mx = true := fun r => by
    have := hle r (hmem.mpr r.2)
    simp only [ite_true] at this
    rwa [hls r r.2] at this
  have hmn : mx.isNaN = false := by rw [hmi]; exact hnan _
  refine ⟨⟨i, hi'⟩, ?_, hmi, fun r hr => le_toReal (hnan r) hmn (hle' r)⟩
  by_contra hne
  have hneg : mx = FloatV.negInf := by rw [hmi]; exact (hS _).2 hne
  have h1 := le_toReal (hnan r0) hmn (hle' r0)
  rw [hneg] at h1
  obtain ⟨h2, h3⟩ := FloatV.negInf_toRat_le (f := Format.binary32) Format.binary32_regular (by decide)
    (hnan r0)
  have h4 : (FloatV.negInf : F32).toRat = (sc r0).toRat := by
    simp only [toReal_def] at h1
    exact le_antisymm h2 (by exact_mod_cast h1)
  have h5 := h3 h4
  have h6 := (hS r0).1 h0
  rw [h5, negInf32_not_fin] at h6
  cases h6

/-- The new maximum: `tl.maximum(m_i, a)` is `a` before the first page (`m_i = −∞`), and
afterwards `m_i` or `a`, not below either. -/
theorem newmax {R : RowCtx} {t : ℕ} {m a m' : F32} (haf : a.isFinite = true)
    (hm' : FloatV.MaxRel true m a m') (hold : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m)) :
    m'.isFinite = true ∧ (m' = a ∨ (0 < t ∧ MInv R t m ∧ m' = m)) ∧
      (0 < t → m.toReal ≤ m'.toReal) ∧ a.toReal ≤ m'.toReal := by
  rcases hold with ⟨rfl, rfl⟩ | ⟨ht, hmi⟩
  · have := FloatV.MaxRel.negInf_left Format.binary32_regular (by decide) hm' (isNaN_of_fin haf)
    subst this
    exact ⟨haf, Or.inl rfl, fun h => absurd h (lt_irrefl 0), le_rfl⟩
  · obtain ⟨hmem, h1, h2⟩ := maxRel_facts hm' (isNaN_of_fin hmi.fin) (isNaN_of_fin haf)
    rcases hmem with rfl | rfl
    · exact ⟨hmi.fin, Or.inr ⟨ht, hmi, rfl⟩, fun _ => h1, h2⟩
    · exact ⟨haf, Or.inl rfl, fun _ => h1, h2⟩

/-- `alpha`'s input `m_i − m_new` (line 64) from the old and new maxima. -/
theorem alphaIn_of {R : RowCtx} (hR : R.Ok) {t : ℕ} {m m' dm : F32} (hm'fin : m'.isFinite = true)
    (hm'b : |m'.toReal| ≤ R.Y + R.E) (hmle : 0 < t → m.toReal ≤ m'.toReal)
    (hdm : FloatV.NaNEq (FloatV.sub m m') dm)
    (hold : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m)) : AlphaIn t m m' dm := by
  rcases hold with ⟨rfl, rfl⟩ | ⟨ht, hmi⟩
  · left
    refine ⟨rfl, ?_⟩
    rw [FloatV.sub_negInf_left (by decide) hm'fin] at hdm
    exact hdm.eq negInf32_isNaN
  · right
    have hle := hmle ht
    have hY := hR.Y_le
    have hE := hR.E_le
    have hroom : |m.toReal - m'.toReal| ≤ 2 ^ 120 := by
      have := hmi.bound
      have := abs_sub (m.toReal) (m'.toReal); linarith
    obtain ⟨hdf, hde⟩ := f32_sub hdm hmi.fin hm'fin hroom
    rw [abs_of_nonpos (by linarith : m.toReal - m'.toReal ≤ 0)] at hde
    exact ⟨ht, hmi.fin, hle, hdf, by linarith⟩

/-- **A page's maximum and exponents** (lines 62–65): `sc` the lanes of line 62 (a token's within
`Es` of its `c·sv`, a masked one `−∞`), `mx` their maximum, `m'` the new maximum, `x` the
exponents, `dm` the input of `alpha`. Lane `0` of the page is a token. -/
theorem mstep {R : RowCtx} (hR : R.Ok) {t : ℕ} {m mx m' dm : F32} {sc x : Fin R.P → F32}
    (hS : ∀ r : Fin R.P, (R.P * t + r < R.n → (sc r).isFinite = true ∧
        |(sc r).toReal - R.c * R.sv (R.P * t + r)| ≤ R.Es) ∧
      (¬ R.P * t + r < R.n → sc r = FloatV.negInf))
    (h0 : R.P * t < R.n)
    (hmx : ∃ tr : Tree, tr.Covers R.P ∧ tr.Eval (FloatV.MaxRel true) (laneSeq sc) mx)
    (hm' : FloatV.MaxRel true m mx m')
    (hx : ∀ r, FloatV.NaNEq (FloatV.sub (sc r) m') (x r))
    (hdm : FloatV.NaNEq (FloatV.sub m m') dm)
    (hold : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m)) :
    MInv R (t + 1) m' ∧ LaneX R t m' x ∧ AlphaIn t m m' dm := by
  have hE0 := hR.E_nonneg
  have hEY := hR.E_le_Y
  have hE8 := hR.E_le
  have hY0 := hR.Y_nonneg
  have hY1 := hR.Y_le
  have hEs := hR.Es_le
  have hEs0 := hR.Es_nonneg
  have hP := hR.P_pos
  obtain ⟨r1, hr1, rfl, hmxle⟩ := maxP hmx (fun r => ⟨fun h => ((hS r).1 h).1, (hS r).2⟩) ⟨0, hP⟩
    (by simpa using h0)
  obtain ⟨hmxf, hmxe⟩ := (hS r1).1 hr1
  obtain ⟨hm'fin, hcase, hmle, hale⟩ := newmax hmxf hm' hold
  have hMI : MInv R (t + 1) m' := by
    have hatt : ∃ j, j < R.P * (t + 1) ∧ j < R.n ∧ |m'.toReal - R.c * R.sv j| ≤ R.E := by
      rcases hcase with rfl | ⟨-, hmi, rfl⟩
      · exact ⟨R.P * t + r1, by rw [Nat.mul_succ]; omega, hr1, hmxe.trans hEs⟩
      · obtain ⟨j, hj, hjn, hjd⟩ := hmi.attained
        exact ⟨j, by rw [Nat.mul_succ]; omega, hjn, hjd⟩
    refine ⟨hm'fin, ?_, hatt, ?_⟩
    · obtain ⟨j, -, hj, hd⟩ := hatt
      have := hR.csv_le hj
      have := abs_sub_abs_le_abs_sub m'.toReal (R.c * R.sv j)
      linarith
    · intro j hj hjn
      rcases key_split hj with hj' | ⟨r, rfl⟩
      · rcases hold with ⟨rfl, -⟩ | ⟨ht, hmi⟩
        · simp at hj'
        · have := hmi.above j hj' hjn; have := hmle ht; linarith
      · have h1 := (hS r).1 hjn
        have h2 := hmxle r hjn
        have := neg_abs_le ((sc r).toReal - R.c * R.sv (R.P * t + r))
        linarith
  refine ⟨hMI, ?_, alphaIn_of hR hm'fin hMI.bound hmle hdm hold⟩
  intro r
  refine ⟨fun hr => ?_, fun hr => ?_⟩
  · obtain ⟨hscf, hsce⟩ := (hS r).1 hr
    have hcs := hR.csv_le hr
    have hm'b := hMI.bound
    have hscb : |(sc r).toReal| ≤ R.Y + R.Es := by
      have := abs_sub_abs_le_abs_sub (sc r).toReal (R.c * R.sv (R.P * t + r)); linarith
    have hd : |(sc r).toReal - m'.toReal| ≤ R.Y + R.Es + (R.Y + R.E) := by
      have := abs_sub (sc r).toReal m'.toReal; linarith
    have hroom : |(sc r).toReal - m'.toReal| ≤ 2 ^ 120 := by linarith
    obtain ⟨hxf, hxe⟩ := f32_sub (hx r) hscf hm'fin hroom
    refine ⟨hxf, ?_⟩
    have e : (x r).toReal - (R.c * R.sv (R.P * t + r) - m'.toReal) =
        ((x r).toReal - ((sc r).toReal - m'.toReal)) + ((sc r).toReal - R.c * R.sv (R.P * t + r)) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    have h2 : |(sc r).toReal - m'.toReal| / 2 ^ 24 ≤ (R.Y + R.Es + (R.Y + R.E)) / 2 ^ 24 :=
      div_le_div_of_nonneg_right hd (by norm_num)
    have hEe := hR.E_eq
    unfold RowCtx.Es at hsce h2 hEs
    rw [hEe] at h2 ⊢
    have := hR.rhoD_nonneg
    have := hR.rhoD_le
    nlinarith
  · have hsc := (hS r).2 hr
    have h := hx r
    rw [hsc, FloatV.sub_negInf_left (by decide) hm'fin] at h
    exact h.eq negInf32_isNaN

end Kernels.PagedAttention
