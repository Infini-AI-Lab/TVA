import AttnFwdE5m2.Lane

/-!
# `attn_fwd_e5m2`: one row, the running maximum and the exponents

For one query row, the analysis tracks the keys the row attends to (`RowCtx.al`) among the keys of
the blocks processed so far, `j < 64·t`. `RowCtx` holds the row's data: `qk_scale` (`c`), the exact
base-2 scale `w = sm_scale / ln 2` (`sm_scale·log₂ e`), the exact scores `sv j` (`q·k_j`), their
magnitudes `bv j` (`Σ|q·k_j|`), and the bounds `Y` (on `c·bv j` and `w·bv j`) and `E` (on each
exponent's error). Key `j`'s exact exponent is `w·s_j`: its exact weight `exp(sm_scale·s_j)` is
`2^(w·s_j)`.

* `MInv`: the running maximum `m_i` after `t ≥ 1` blocks is within `E` of some attended key's
  `w·s_j`, and every attended key's `w·s_j` is at most `m_i + E`.
* `mstep_plain` (lines 80–81, `STAGE` 1 or 3) and `mstep_masked` (lines 75–78, `STAGE` 2): one
  block's new maximum `m_ij`, each lane's exponent `qk` (within `E` of `w·s_j − m_ij`, at most
  `2^-10`, and below `−2^19` for a masked lane), and the input of `alpha` (line 84); and either the
  maximum is kept or some lane's exponent is within `2^-11` of `0` (for `Mag.lmin_real`).
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-- One query row's data. -/
structure RowCtx where
  /-- `qk_scale` (line 219), as a real number. -/
  c : ℝ
  /-- The exact base-2 scale `sm_scale / ln 2`. -/
  w : ℝ
  /-- The bound on `c·Σ|q·k_j|` and `w·Σ|q·k_j|` over the attended keys. -/
  Y : ℝ
  /-- The bound on each exponent's error. -/
  E : ℝ
  /-- `relDot8 HEAD_DIM`. -/
  rhoD : ℝ
  /-- `absDot8 HEAD_DIM`. -/
  aD : ℝ
  /-- The exact scores `q·k_j`. -/
  sv : ℕ → ℝ
  /-- Their magnitudes `Σ_e |q_e·k_{j,e}|`. -/
  bv : ℕ → ℝ
  /-- The keys the row attends to. -/
  al : ℕ → Bool

/-- What the analysis needs of a row's data. -/
structure RowCtx.Ok (R : RowCtx) : Prop where
  c_pos : 0 < R.c
  c_le : R.c ≤ 2 ^ 65
  w_pos : 0 < R.w
  cw : |R.c - R.w| ≤ R.w / 2 ^ 22
  Y_nonneg : 0 ≤ R.Y
  Y_le : R.Y ≤ 768
  rhoD_nonneg : 0 ≤ R.rhoD
  rhoD_le : R.rhoD ≤ 1 / 8
  aD_nonneg : 0 ≤ R.aD
  aD_le : R.aD ≤ 1 / 2 ^ 117
  E_eq : R.E = (R.rhoD + 1 / 2 ^ 19) * R.Y + 1 / 2 ^ 50
  E_le : R.E ≤ 13
  sv_le : ∀ j, |R.sv j| ≤ R.bv j
  al_le : ∀ j, R.al j = true → R.c * R.bv j ≤ R.Y ∧ R.w * R.bv j ≤ R.Y

theorem RowCtx.Ok.E_nonneg {R : RowCtx} (h : R.Ok) : 0 ≤ R.E := by
  rw [h.E_eq]; have := h.Y_nonneg; have := h.rhoD_nonneg; positivity

theorem RowCtx.Ok.E_le_Y {R : RowCtx} (h : R.Ok) : R.E ≤ R.Y / 4 + 1 / 2 ^ 50 := by
  rw [h.E_eq]
  have := h.Y_nonneg
  have h1 : R.rhoD + 1 / 2 ^ 19 ≤ 1 / 4 := by
    have := h.rhoD_le; linarith
  nlinarith

theorem RowCtx.Ok.bv_nonneg {R : RowCtx} (h : R.Ok) (j : ℕ) : 0 ≤ R.bv j :=
  (abs_nonneg _).trans (h.sv_le j)

/-- The running maximum after `t ≥ 1` blocks. -/
structure MInv (R : RowCtx) (t : ℕ) (m : F32) : Prop where
  fin : m.isFinite = true
  bound : |m.toReal| ≤ R.Y + R.E
  attained : ∃ j, j < 64 * t ∧ R.al j = true ∧ |m.toReal - R.w * R.sv j| ≤ R.E
  above : ∀ j, j < 64 * t → R.al j = true → R.w * R.sv j ≤ m.toReal + R.E

/-- The input `m_i − m_ij` of `alpha` (line 84): `−∞` before the first block; afterwards a finite
value within one rounding of `m_i − m_ij ≤ 0`. -/
def AlphaIn (t : ℕ) (m m' dm : F32) : Prop :=
  (t = 0 ∧ dm = FloatV.negInf) ∨
  (0 < t ∧ m.isFinite = true ∧ m.toReal ≤ m'.toReal ∧ dm.isFinite = true ∧
    |dm.toReal - (m.toReal - m'.toReal)| ≤ (m'.toReal - m.toReal) / 2 ^ 24 + 1 / 2 ^ 150)

/-- Each lane's exponent (the `qk` of line 82): finite and at most `2^-10`; for an attended key
`j = 64t + jj`, within `E` of `w·s_j − m_ij`; for a masked one, below `−2^19`. -/
def LaneX (R : RowCtx) (t : ℕ) (m' : F32) (x : Fin 64 → F32) : Prop :=
  ∀ jj : Fin 64, (x jj).isFinite = true ∧ (x jj).toReal ≤ 1 / 2 ^ 10 ∧
    (R.al (64 * t + jj) = true → |(x jj).toReal - (R.w * R.sv (64 * t + jj) - m'.toReal)| ≤ R.E) ∧
    (R.al (64 * t + jj) = false → (x jj).toReal ≤ -2 ^ 19)

theorem toReal_neg (x : F32) : (FloatV.neg x).toReal = -x.toReal := by
  simp only [toReal_def, FloatV.toRat_neg]; push_cast; ring

theorem isNaN_of_fin {f : Format} {x : FloatV f} (h : x.isFinite = true) : x.isNaN = false :=
  (FloatV.isFinite_iff.mp h).1

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

/-- The bound on `c·S` of a lane with `c·bv ≤ B`. -/
theorem cS_bound {R : RowCtx} (hR : R.Ok) {S : F32} {j : ℕ} {B : ℝ}
    (hS : |S.toReal - R.sv j| ≤ R.rhoD * R.bv j + R.aD) (hB : R.c * R.bv j ≤ B) :
    |R.c * S.toReal - R.c * R.sv j| ≤ R.rhoD * B + 1 / 2 ^ 52 ∧
      |R.c * S.toReal| ≤ (1 + R.rhoD) * B + 1 / 2 ^ 52 := by
  have hc := hR.c_pos
  have hbv : 0 ≤ R.bv j := hR.bv_nonneg j
  have hca : R.c * R.aD ≤ 1 / 2 ^ 52 := by
    have := mul_le_mul hR.c_le hR.aD_le hR.aD_nonneg (by norm_num)
    norm_num at this ⊢; linarith
  have h1 : |R.c * S.toReal - R.c * R.sv j| ≤ R.rhoD * B + 1 / 2 ^ 52 := by
    rw [← mul_sub, abs_mul, abs_of_pos hc]
    have := mul_le_mul_of_nonneg_left hS hc.le
    have h2 : R.c * (R.rhoD * R.bv j) ≤ R.rhoD * B := by
      have := mul_le_mul_of_nonneg_left hB hR.rhoD_nonneg; nlinarith
    nlinarith
  refine ⟨h1, ?_⟩
  have hsv : |R.c * R.sv j| ≤ B := by
    rw [abs_mul, abs_of_pos hc]; exact (mul_le_mul_of_nonneg_left (hR.sv_le j) hc.le).trans hB
  have := abs_sub_abs_le_abs_sub (R.c * S.toReal) (R.c * R.sv j)
  nlinarith [hR.rhoD_nonneg]

/-- The change from `qk_scale` to the exact `w` at a key with `w·bv ≤ Y`. -/
theorem cw_bound {R : RowCtx} (hR : R.Ok) {j : ℕ} (hw : R.w * R.bv j ≤ R.Y) :
    |R.c * R.sv j - R.w * R.sv j| ≤ R.Y / 2 ^ 22 := by
  rw [← sub_mul, abs_mul]
  have h1 := mul_le_mul hR.cw (hR.sv_le j) (abs_nonneg _) (by have := hR.w_pos; positivity)
  have : R.w / 2 ^ 22 * R.bv j = R.w * R.bv j / 2 ^ 22 := by ring
  rw [this] at h1
  have := div_le_div_of_nonneg_right hw (by norm_num : (0 : ℝ) ≤ 2 ^ 22)
  linarith

/-- A key `j < 64(t+1)` lies in the first `t` blocks or in block `t`. -/
theorem key_split {t j : ℕ} (h : j < 64 * (t + 1)) : j < 64 * t ∨ ∃ jj : Fin 64, j = 64 * t + jj := by
  by_cases hj : j < 64 * t
  · exact Or.inl hj
  · exact Or.inr ⟨⟨j - 64 * t, by omega⟩, by simp; omega⟩

/-- The new maximum is `MInv` when it is within `E` of an attended key's `w·s`, and every attended
key so far is at most it plus `E`. -/
theorem minv_of {R : RowCtx} (hR : R.Ok) {t : ℕ} {m' : F32} (hfin : m'.isFinite = true)
    (hatt : ∃ j, j < 64 * (t + 1) ∧ R.al j = true ∧ |m'.toReal - R.w * R.sv j| ≤ R.E)
    (habove : ∀ j, j < 64 * (t + 1) → R.al j = true → R.w * R.sv j ≤ m'.toReal + R.E) :
    MInv R (t + 1) m' := by
  refine ⟨hfin, ?_, hatt, habove⟩
  obtain ⟨j, -, hj, hd⟩ := hatt
  have hsv : |R.w * R.sv j| ≤ R.Y := by
    rw [abs_mul, abs_of_pos hR.w_pos]
    exact (mul_le_mul_of_nonneg_left (hR.sv_le j) hR.w_pos.le).trans (hR.al_le j hj).2
  have := abs_sub_abs_le_abs_sub m'.toReal (R.w * R.sv j)
  linarith

/-- The new maximum: `tl.maximum(m_i, a)` is `a` before the first block (`m_i = −∞`), and
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

/-- `alpha`'s input `m_i − m_ij` (line 84) from the old and new maxima. -/
theorem alphaIn_of {R : RowCtx} {t : ℕ} {m m' dm : F32} (hm'fin : m'.isFinite = true)
    (hm'b : |m'.toReal| ≤ R.Y + R.E) (hY : R.Y ≤ 768) (hE : R.E ≤ 13)
    (hmle : 0 < t → m.toReal ≤ m'.toReal) (hdm : FloatV.NaNEq (FloatV.sub m m') dm)
    (hold : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m)) : AlphaIn t m m' dm := by
  rcases hold with ⟨rfl, rfl⟩ | ⟨ht, hmi⟩
  · left
    refine ⟨rfl, ?_⟩
    rw [FloatV.sub_negInf_left (by decide) hm'fin] at hdm
    exact hdm.eq negInf32_isNaN
  · right
    have hle := hmle ht
    have hroom : |m.toReal - m'.toReal| ≤ 2 ^ 120 := by
      have := hmi.bound
      have := abs_sub (m.toReal) (m'.toReal); linarith
    obtain ⟨hdf, hde⟩ := f32_sub hdm hmi.fin hm'fin hroom
    rw [abs_of_nonpos (by linarith : m.toReal - m'.toReal ≤ 0)] at hde
    exact ⟨ht, hmi.fin, hle, hdf, by linarith⟩

set_option maxHeartbeats 800000 in
/-- **The block's maximum and exponents, `STAGE` 1 or 3** (lines 80–81, 84): every lane of the
block is attended. `S` is the block's row of `tl.dot(q, k)`, `mx` its row maximum, `a` that times
`qk_scale` (`cq`), `m'` the new maximum, `Sc` the lanes times `qk_scale`, `x` the exponents, `dm`
the input of `alpha`. -/
theorem mstep_plain {R : RowCtx} (hR : R.Ok) {t : ℕ} {fusion : Bool} {cq m mx a m' dm : F32}
    {S Sc x : Fin 64 → F32}
    (hcq : cq.isFinite = true) (hcqv : cq.toReal = R.c)
    (hS : ∀ jj : Fin 64, (S jj).isFinite = true ∧
      |(S jj).toReal - R.sv (64 * t + jj)| ≤ R.rhoD * R.bv (64 * t + jj) + R.aD)
    (hal : ∀ jj : Fin 64, R.al (64 * t + jj) = true)
    (hmx : (∃ jj, mx = S jj) ∧ ∀ jj, (S jj).toReal ≤ mx.toReal)
    (ha : FloatV.NaNEq (FloatV.mul mx cq) a) (hm' : FloatV.MaxRel true m a m')
    (hSc : ∀ jj, FloatV.NaNEq (FloatV.mul (S jj) cq) (Sc jj))
    (hx : ∀ jj, AddRel fusion (Sc jj) (FloatV.neg m') (x jj))
    (hdm : FloatV.NaNEq (FloatV.sub m m') dm)
    (hold : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m)) :
    MInv R (t + 1) m' ∧ LaneX R t m' x ∧ AlphaIn t m m' dm ∧
      ((0 < t ∧ m' = m) ∨ ∃ jj : Fin 64, |(x jj).toReal| ≤ 1 / 2 ^ 11) := by
  have hc := hR.c_pos
  have hE0 := hR.E_nonneg
  have hEY := hR.E_le_Y
  have hEe := hR.E_eq
  have hE13 := hR.E_le
  have hY0 := hR.Y_nonneg
  have hY1 := hR.Y_le
  have hrho0 := hR.rhoD_nonneg
  have hrho1 := hR.rhoD_le
  have hrY : R.rhoD * R.Y ≤ R.Y / 8 := by nlinarith
  -- each lane's product with `qk_scale`, and the change to `w`
  have hcS : ∀ jj : Fin 64, |R.c * (S jj).toReal - R.c * R.sv (64 * t + jj)| ≤ R.rhoD * R.Y + 1 / 2 ^ 52 ∧
      |R.c * (S jj).toReal| ≤ (1 + R.rhoD) * R.Y + 1 / 2 ^ 52 := fun jj =>
    cS_bound hR (hS jj).2 (hR.al_le _ (hal jj)).1
  have hcw : ∀ jj : Fin 64, |R.c * R.sv (64 * t + jj) - R.w * R.sv (64 * t + jj)| ≤ R.Y / 2 ^ 22 :=
    fun jj => cw_bound hR (hR.al_le _ (hal jj)).2
  have hcSY : ∀ jj : Fin 64, |R.c * (S jj).toReal| ≤ 9 / 8 * R.Y + 1 / 2 ^ 52 := fun jj => by
    have := (hcS jj).2; nlinarith
  -- the row maximum is a lane
  obtain ⟨⟨jj0, rfl⟩, hmxle⟩ := hmx
  have hmxf := (hS jj0).1
  obtain ⟨hcS0, hcS0'⟩ := hcS jj0
  -- `a = fl(mx · qk_scale)`
  have hroomA : |(S jj0).toReal * cq.toReal| ≤ 2 ^ 120 := by
    rw [hcqv, mul_comm]; have := hcSY jj0; nlinarith
  obtain ⟨haf, hae⟩ := f32_mul ha hmxf hcq hroomA
  rw [hcqv, mul_comm (S jj0).toReal] at hae
  have hae' : |a.toReal - R.c * (S jj0).toReal| ≤ (9 / 8 * R.Y + 1 / 2 ^ 52) / 2 ^ 24 + 1 / 2 ^ 150 := by
    have := div_le_div_of_nonneg_right (hcSY jj0) (by norm_num : (0 : ℝ) ≤ 2 ^ 24); linarith
  -- `a` is within `E` of the key `jj0`'s `w·s`
  have haj : |a.toReal - R.w * R.sv (64 * t + jj0)| ≤ R.E := by
    have e : a.toReal - R.w * R.sv (64 * t + jj0) = (a.toReal - R.c * (S jj0).toReal) +
        (R.c * (S jj0).toReal - R.c * R.sv (64 * t + jj0)) +
        (R.c * R.sv (64 * t + jj0) - R.w * R.sv (64 * t + jj0)) := by ring
    rw [e]; refine (abs_add_le _ _).trans ((add_le_add (abs_add_le _ _) le_rfl).trans ?_)
    have := hcw jj0
    rw [hEe]; nlinarith
  -- every lane's `w·s` is at most `a + E`
  have haabove : ∀ jj : Fin 64, R.w * R.sv (64 * t + jj) ≤ a.toReal + R.E := by
    intro jj
    have h1 : R.c * (S jj).toReal ≤ R.c * (S jj0).toReal := mul_le_mul_of_nonneg_left (hmxle jj) hc.le
    have h2 := (hcS jj).1
    have h3 := neg_abs_le (R.c * (S jj).toReal - R.c * R.sv (64 * t + jj))
    have h4 := neg_abs_le (a.toReal - R.c * (S jj0).toReal)
    have h5 := neg_abs_le (R.c * R.sv (64 * t + jj) - R.w * R.sv (64 * t + jj))
    have h6 := hcw jj
    rw [hEe]; nlinarith
  -- the new maximum
  obtain ⟨hm'fin, hcase, hmle, hale⟩ := newmax haf hm' hold
  have hMI : MInv R (t + 1) m' := by
    refine minv_of hR hm'fin ?_ ?_
    · rcases hcase with rfl | ⟨-, hmi, rfl⟩
      · exact ⟨64 * t + jj0, by omega, hal jj0, haj⟩
      · obtain ⟨j, hj, hjal, hjd⟩ := hmi.attained
        exact ⟨j, by omega, hjal, hjd⟩
    · intro j hj hjal
      rcases key_split hj with hj' | ⟨jj, rfl⟩
      · rcases hold with ⟨rfl, -⟩ | ⟨ht, hmi⟩
        · omega
        · have := hmi.above j hj' hjal; have := hmle ht; linarith
      · have := haabove jj; linarith
  -- a lane at the new maximum, or the maximum kept
  have hnew : (0 < t ∧ m' = m) ∨ ∃ jj : Fin 64, |(x jj).toReal| ≤ 1 / 2 ^ 11 := by
    rcases hcase with hma | ⟨ht0, -, hmm⟩
    · right
      refine ⟨jj0, ?_⟩
      have hSf := (hS jj0).1
      have hroomS : |(S jj0).toReal * cq.toReal| ≤ 2 ^ 120 := by
        rw [hcqv, mul_comm]; have := hcSY jj0; nlinarith
      obtain ⟨hScf, hSce⟩ := f32_mul (hSc jj0) hSf hcq hroomS
      rw [hcqv, mul_comm (S jj0).toReal] at hSce
      have hSce' : |(Sc jj0).toReal - R.c * (S jj0).toReal| ≤ (9 / 8 * R.Y + 1 / 2 ^ 52) / 2 ^ 24 + 1 / 2 ^ 150 := by
        have := div_le_div_of_nonneg_right (hcSY jj0) (by norm_num : (0 : ℝ) ≤ 2 ^ 24); linarith
      have hnegf : (FloatV.neg m').isFinite = true := by rw [FloatV.isFinite_neg]; exact hm'fin
      have hm'b := hMI.bound
      have hScb : |(Sc jj0).toReal| ≤ 5 / 4 * R.Y + 1 / 2 ^ 51 := by
        have := abs_sub_abs_le_abs_sub (Sc jj0).toReal (R.c * (S jj0).toReal)
        have : (9 / 8 * R.Y + 1 / 2 ^ 52) / 2 ^ 24 ≤ R.Y / 2 ^ 20 + 1 / 2 ^ 76 := by
          rw [div_le_iff₀ (by norm_num)]; nlinarith
        have := hcSY jj0
        nlinarith
      have hroomX : |(Sc jj0).toReal| + |(FloatV.neg m').toReal| ≤ 2 ^ 120 := by
        rw [toReal_neg, abs_neg]; nlinarith
      obtain ⟨-, hxe⟩ := f32_addRel (hx jj0) hScf hnegf hroomX
      rw [toReal_neg, abs_neg] at hxe
      have hsum : 3 * (|(Sc jj0).toReal| + |m'.toReal|) / 2 ^ 24 ≤
          3 * (5 / 4 * R.Y + 1 / 2 ^ 51 + (R.Y + R.E)) / 2 ^ 24 :=
        div_le_div_of_nonneg_right (by linarith) (by norm_num)
      have ham : |m'.toReal - R.c * (S jj0).toReal| ≤ (9 / 8 * R.Y + 1 / 2 ^ 52) / 2 ^ 24 + 1 / 2 ^ 150 := by
        rw [hma]; exact hae'
      obtain ⟨k1, k2⟩ := abs_le.mp hxe
      obtain ⟨k3, k4⟩ := abs_le.mp hSce'
      obtain ⟨k5, k6⟩ := abs_le.mp ham
      rw [abs_le]; constructor <;> linarith
    · exact Or.inl ⟨ht0, hmm⟩
  refine ⟨hMI, ?_, alphaIn_of hm'fin hMI.bound hY1 hE13 hmle hdm hold, hnew⟩
  -- the exponents
  intro jj
  have hSf := (hS jj).1
  obtain ⟨hcSj, hcSj'⟩ := hcS jj
  have hcSYj := hcSY jj
  have hroomS : |(S jj).toReal * cq.toReal| ≤ 2 ^ 120 := by rw [hcqv, mul_comm]; nlinarith
  obtain ⟨hScf, hSce⟩ := f32_mul (hSc jj) hSf hcq hroomS
  rw [hcqv, mul_comm (S jj).toReal] at hSce
  have hSce' : |(Sc jj).toReal - R.c * (S jj).toReal| ≤ (9 / 8 * R.Y + 1 / 2 ^ 52) / 2 ^ 24 + 1 / 2 ^ 150 := by
    have := div_le_div_of_nonneg_right hcSYj (by norm_num : (0 : ℝ) ≤ 2 ^ 24); linarith
  have hnegf : (FloatV.neg m').isFinite = true := by rw [FloatV.isFinite_neg]; exact hm'fin
  have hm'b := hMI.bound
  have hScb : |(Sc jj).toReal| ≤ 5 / 4 * R.Y + 1 / 2 ^ 51 := by
    have := abs_sub_abs_le_abs_sub (Sc jj).toReal (R.c * (S jj).toReal)
    have : (9 / 8 * R.Y + 1 / 2 ^ 52) / 2 ^ 24 ≤ R.Y / 2 ^ 20 + 1 / 2 ^ 76 := by
      rw [div_le_iff₀ (by norm_num)]; nlinarith
    nlinarith
  have hroomX : |(Sc jj).toReal| + |(FloatV.neg m').toReal| ≤ 2 ^ 120 := by
    rw [toReal_neg, abs_neg]; nlinarith
  obtain ⟨hxf, hxe⟩ := f32_addRel (hx jj) hScf hnegf hroomX
  rw [toReal_neg, abs_neg] at hxe
  have hsum : 3 * (|(Sc jj).toReal| + |m'.toReal|) / 2 ^ 24 ≤ 3 * (5 / 4 * R.Y + 1 / 2 ^ 51 + (R.Y + R.E)) / 2 ^ 24 :=
    div_le_div_of_nonneg_right (by linarith) (by norm_num)
  refine ⟨hxf, ?_, fun _ => ?_, fun h => absurd (hal jj) (by rw [h]; decide)⟩
  · -- `x ≤ 2^-10`: `Sc ≤ a + rounding` and `a ≤ m'`
    have h1 : R.c * (S jj).toReal ≤ R.c * (S jj0).toReal := mul_le_mul_of_nonneg_left (hmxle jj) hc.le
    have h2 := le_abs_self ((x jj).toReal - ((Sc jj).toReal + -m'.toReal))
    have h3 := le_abs_self ((Sc jj).toReal - R.c * (S jj).toReal)
    have h4 := neg_abs_le (a.toReal - R.c * (S jj0).toReal)
    nlinarith
  · have e : (x jj).toReal - (R.w * R.sv (64 * t + jj) - m'.toReal) =
        ((x jj).toReal - ((Sc jj).toReal + -m'.toReal)) + ((Sc jj).toReal - R.c * (S jj).toReal) +
          (R.c * (S jj).toReal - R.c * R.sv (64 * t + jj)) +
          (R.c * R.sv (64 * t + jj) - R.w * R.sv (64 * t + jj)) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ((add_le_add (abs_add_le _ _) le_rfl).trans
      ((add_le_add (add_le_add (abs_add_le _ _) le_rfl) le_rfl).trans ?_))
    have := hcw jj
    rw [hEe] at hsum ⊢
    nlinarith

set_option maxHeartbeats 800000 in
/-- **The block's maximum and exponents, `STAGE` 2** (lines 75–78, 84): the masked block. `w` is
the `tl.where` of line 76 (`+0` on an attended lane, `float32(-1e6)` on a masked one), `x'` the
`tl.mul_add`, `mx` its row maximum, `m'` the new maximum, `x` the exponents (line 78). Every lane's
`c·Σ|q·k|` is at most `1536`, which keeps the masked lanes below the attended ones. Before the first
block, the block's first lane is attended. -/
theorem mstep_masked {R : RowCtx} (hR : R.Ok) {t : ℕ} {fusion : Bool} {cq m mx m' dm : F32}
    {S w x' x : Fin 64 → F32}
    (hcq : cq.isFinite = true) (hcqv : cq.toReal = R.c)
    (hS : ∀ jj : Fin 64, (S jj).isFinite = true ∧
      |(S jj).toReal - R.sv (64 * t + jj)| ≤ R.rhoD * R.bv (64 * t + jj) + R.aD)
    (hglob : ∀ jj : Fin 64, R.c * R.bv (64 * t + jj) ≤ 1536)
    (hw : ∀ jj : Fin 64, (w jj).isFinite = true ∧ (R.al (64 * t + jj) = true → (w jj).toReal = 0) ∧
      (R.al (64 * t + jj) = false → (w jj).toReal = -1000000))
    (hx' : ∀ jj, FloatV.NaNEq (FloatV.add (FloatV.mul (S jj) cq) (w jj)) (x' jj) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (S jj) cq (w jj)) (x' jj)))
    (hmx : (∃ jj, mx = x' jj) ∧ ∀ jj, (x' jj).toReal ≤ mx.toReal)
    (hm' : FloatV.MaxRel true m mx m')
    (hx : ∀ jj, FloatV.NaNEq (FloatV.sub (x' jj) m') (x jj))
    (hdm : FloatV.NaNEq (FloatV.sub m m') dm)
    (hold : (t = 0 ∧ m = FloatV.negInf ∧ R.al (64 * t) = true) ∨ (0 < t ∧ MInv R t m)) :
    MInv R (t + 1) m' ∧ LaneX R t m' x ∧ AlphaIn t m m' dm ∧
      ((0 < t ∧ m' = m) ∨ ∃ jj : Fin 64, |(x jj).toReal| ≤ 1 / 2 ^ 11) := by
  have hc := hR.c_pos
  have hE0 := hR.E_nonneg
  have hEY := hR.E_le_Y
  have hEe := hR.E_eq
  have hE13 := hR.E_le
  have hY0 := hR.Y_nonneg
  have hY1 := hR.Y_le
  have hrho0 := hR.rhoD_nonneg
  have hrho1 := hR.rhoD_le
  have hrY : R.rhoD * R.Y ≤ R.Y / 8 := by nlinarith
  have hold' : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m) := by
    rcases hold with ⟨h1, h2, -⟩ | h
    · exact Or.inl ⟨h1, h2⟩
    · exact Or.inr h
  -- the error of an attended lane's `x'` against `c·s`
  set e1 : ℝ := R.rhoD * R.Y + 1 / 2 ^ 52 + 3 * ((1 + R.rhoD) * R.Y + 1 / 2 ^ 52) / 2 ^ 24 + 4 / 2 ^ 150
    with he1
  have he1b : e1 ≤ R.Y / 7 + 1 / 2 ^ 51 := by rw [he1]; nlinarith
  -- each lane's `x'`
  have hX : ∀ jj : Fin 64, (x' jj).isFinite = true ∧ |(x' jj).toReal| ≤ 1001730 ∧
      (R.al (64 * t + jj) = true → |(x' jj).toReal - R.c * R.sv (64 * t + jj)| ≤ e1) ∧
      (R.al (64 * t + jj) = false → (x' jj).toReal ≤ -998000) := by
    intro jj
    obtain ⟨hSf, hSe⟩ := hS jj
    obtain ⟨hwf, hw0, hw1⟩ := hw jj
    obtain ⟨-, hg'⟩ := cS_bound hR hSe (hglob jj)
    have hg'' : |R.c * (S jj).toReal| ≤ 1729 := by nlinarith
    have hwb : |(w jj).toReal| ≤ 1000000 := by
      cases hal : R.al (64 * t + jj)
      · rw [hw1 hal]; norm_num
      · rw [hw0 hal]; norm_num
    have hroom : |(S jj).toReal * cq.toReal| + |(w jj).toReal| ≤ 2 ^ 120 := by
      rw [hcqv, mul_comm]; nlinarith
    obtain ⟨hxf, hxe⟩ := f32_mulAdd (hx' jj) hSf hcq hwf hroom
    rw [hcqv, mul_comm (S jj).toReal] at hxe
    have hxb : |(x' jj).toReal| ≤ 1001730 := by
      have := abs_sub_abs_le_abs_sub (x' jj).toReal (R.c * (S jj).toReal + (w jj).toReal)
      have := abs_add_le (R.c * (S jj).toReal) (w jj).toReal
      nlinarith
    refine ⟨hxf, hxb, fun hal => ?_, fun hal => ?_⟩
    · obtain ⟨hcS1, hcS1'⟩ := cS_bound hR hSe (hR.al_le _ hal).1
      rw [hw0 hal, add_zero, abs_zero, add_zero] at hxe
      have e : (x' jj).toReal - R.c * R.sv (64 * t + jj) =
          ((x' jj).toReal - R.c * (S jj).toReal) + (R.c * (S jj).toReal - R.c * R.sv (64 * t + jj)) := by ring
      rw [e, he1]; refine (abs_add_le _ _).trans ?_
      have := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hcS1' (by norm_num : (0 : ℝ) ≤ 3))
        (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
      nlinarith
    · rw [hw1 hal, show |(-1000000 : ℝ)| = 1000000 by norm_num] at hxe
      have := le_abs_self ((x' jj).toReal - (R.c * (S jj).toReal + -1000000))
      have := le_abs_self (R.c * (S jj).toReal)
      nlinarith
  -- the row maximum is a lane
  obtain ⟨⟨jj0, rfl⟩, hmxle⟩ := hmx
  obtain ⟨hmxf, hmxb, -, -⟩ := hX jj0
  obtain ⟨hm'fin, hcase, hmle, hale⟩ := newmax hmxf hm' hold'
  -- an attended lane's `x'` is at least `-1000`
  have hlow : ∀ jj : Fin 64, R.al (64 * t + jj) = true → -1000 ≤ (x' jj).toReal := by
    intro jj hal
    have h1 := (hX jj).2.2.1 hal
    have hsv : |R.c * R.sv (64 * t + jj)| ≤ R.Y := by
      rw [abs_mul, abs_of_pos hc]
      exact (mul_le_mul_of_nonneg_left (hR.sv_le _) hc.le).trans (hR.al_le _ hal).1
    have := neg_abs_le ((x' jj).toReal - R.c * R.sv (64 * t + jj))
    have := neg_abs_le (R.c * R.sv (64 * t + jj))
    linarith
  -- the maximum's lane is attended when it is the new maximum
  have hj0 : m' = x' jj0 → R.al (64 * t + jj0) = true := by
    intro hm'eq
    by_contra hne
    have hne' : R.al (64 * t + jj0) = false := by simpa using hne
    have h1 := (hX jj0).2.2.2 hne'
    rcases hold with ⟨rfl, rfl, hal0⟩ | ⟨ht, hmi⟩
    · have := hlow ⟨0, by norm_num⟩ (by simpa using hal0)
      have := hmxle ⟨0, by norm_num⟩
      linarith
    · have h2 := hmle ht
      have h3 := hmi.bound
      have h4 := neg_abs_le m.toReal
      rw [hm'eq] at h2
      linarith
  have hMI : MInv R (t + 1) m' := by
    refine minv_of hR hm'fin ?_ ?_
    · rcases hcase with hcase | ⟨-, hmi, rfl⟩
      · have hal := hj0 hcase
        refine ⟨64 * t + jj0, by omega, hal, ?_⟩
        rw [hcase]
        have h1 := (hX jj0).2.2.1 hal
        have h2 := cw_bound hR (hR.al_le _ hal).2
        have e : (x' jj0).toReal - R.w * R.sv (64 * t + jj0) = ((x' jj0).toReal - R.c * R.sv (64 * t + jj0)) +
            (R.c * R.sv (64 * t + jj0) - R.w * R.sv (64 * t + jj0)) := by ring
        rw [e]; refine (abs_add_le _ _).trans ?_
        rw [he1] at h1; rw [hEe]; nlinarith
      · obtain ⟨j, hj, hjal, hjd⟩ := hmi.attained
        exact ⟨j, by omega, hjal, hjd⟩
    · intro j hj hjal
      rcases key_split hj with hj' | ⟨jj, rfl⟩
      · rcases hold' with ⟨rfl, -⟩ | ⟨ht, hmi⟩
        · omega
        · have := hmi.above j hj' hjal; have := hmle ht; linarith
      · have h1 := (hX jj).2.2.1 hjal
        have h2 := cw_bound hR (hR.al_le _ hjal).2
        have := hmxle jj
        have := neg_abs_le ((x' jj).toReal - R.c * R.sv (64 * t + jj))
        have := neg_abs_le (R.c * R.sv (64 * t + jj) - R.w * R.sv (64 * t + jj))
        rw [he1] at h1; rw [hEe]; nlinarith
  -- a lane at the new maximum, or the maximum kept
  have hnew : (0 < t ∧ m' = m) ∨ ∃ jj : Fin 64, |(x jj).toReal| ≤ 1 / 2 ^ 11 := by
    rcases hcase with hma | ⟨ht0, -, hmm⟩
    · right
      refine ⟨jj0, ?_⟩
      have hm'b := hMI.bound
      have hroom : |(x' jj0).toReal - m'.toReal| ≤ 2 ^ 120 := by
        have := abs_sub (x' jj0).toReal m'.toReal; linarith
      obtain ⟨-, hxe⟩ := f32_sub (hx jj0) hmxf hm'fin hroom
      rw [← hma, sub_self, abs_zero, sub_zero, zero_div, zero_add] at hxe
      exact hxe.trans (by norm_num)
    · exact Or.inl ⟨ht0, hmm⟩
  refine ⟨hMI, ?_, alphaIn_of hm'fin hMI.bound hY1 hE13 hmle hdm hold', hnew⟩
  -- the exponents
  intro jj
  obtain ⟨hx'f, hx'b, hx'a, hx'm⟩ := hX jj
  have hm'b := hMI.bound
  have hroom : |(x' jj).toReal - m'.toReal| ≤ 2 ^ 120 := by
    have := abs_sub (x' jj).toReal m'.toReal; linarith
  obtain ⟨hxf, hxe⟩ := f32_sub (hx jj) hx'f hm'fin hroom
  have hx'le : (x' jj).toReal ≤ m'.toReal := (hmxle jj).trans hale
  rw [abs_of_nonpos (by linarith : (x' jj).toReal - m'.toReal ≤ 0)] at hxe
  refine ⟨hxf, ?_, fun hal => ?_, fun hal => ?_⟩
  · have := le_abs_self ((x jj).toReal - ((x' jj).toReal - m'.toReal))
    have : (0 : ℝ) ≤ -((x' jj).toReal - m'.toReal) := by linarith
    nlinarith
  · have h1 := hx'a hal
    have h2 := cw_bound hR (hR.al_le _ hal).2
    have hsv : |R.c * R.sv (64 * t + jj)| ≤ R.Y := by
      rw [abs_mul, abs_of_pos hc]
      exact (mul_le_mul_of_nonneg_left (hR.sv_le _) hc.le).trans (hR.al_le _ hal).1
    have hxa : |(x' jj).toReal| ≤ R.Y + e1 := by
      have := abs_sub_abs_le_abs_sub (x' jj).toReal (R.c * R.sv (64 * t + jj)); linarith
    have hd : -((x' jj).toReal - m'.toReal) ≤ R.Y + e1 + (R.Y + R.E) := by
      have := neg_abs_le (x' jj).toReal; have := le_abs_self m'.toReal; linarith
    have e : (x jj).toReal - (R.w * R.sv (64 * t + jj) - m'.toReal) =
        ((x jj).toReal - ((x' jj).toReal - m'.toReal)) + ((x' jj).toReal - R.c * R.sv (64 * t + jj)) +
          (R.c * R.sv (64 * t + jj) - R.w * R.sv (64 * t + jj)) := by ring
    rw [e]; refine (abs_add_le _ _).trans ((add_le_add (abs_add_le _ _) le_rfl).trans ?_)
    have h3 : -((x' jj).toReal - m'.toReal) / 2 ^ 24 ≤ (R.Y + e1 + (R.Y + R.E)) / 2 ^ 24 :=
      div_le_div_of_nonneg_right hd (by norm_num)
    rw [he1] at h1 h3
    rw [hEe] at h3 ⊢
    nlinarith
  · have h1 := hx'm hal
    have hneg : (x' jj).toReal - m'.toReal ≤ -997000 := by
      have := neg_abs_le m'.toReal; linarith
    have := le_abs_self ((x jj).toReal - ((x' jj).toReal - m'.toReal))
    have hd : -((x' jj).toReal - m'.toReal) ≤ 1001730 + (R.Y + R.E) := by
      have := neg_abs_le (x' jj).toReal; have := le_abs_self m'.toReal; linarith
    norm_num
    nlinarith

end Kernels.AttnFwdE5m2
