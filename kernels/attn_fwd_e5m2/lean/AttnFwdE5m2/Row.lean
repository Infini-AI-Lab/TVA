import AttnFwdE5m2.Facts

/-!
# `attn_fwd_e5m2`: one query row of the instance

* `alOf`, `rowCtx`: the keys a row attends to, and its `RowCtx` at a bound `X ≤ Xh HEAD_DIM` on its
  scaled score magnitudes; `rowCtx_ok`;
* `max64`: the row maximum of a block (`tl.max`, any tree);
* `esR`, `ebR`: the accumulator's error bounds block by block (`Model.errS`, `Model.errB` in the
  row's terms);
* `RowSt`: a row's `m_i`, `l_i` and `acc` after `t` blocks: their initial values when `t = 0`,
  and `MInv`, `LInv`, `AInv` and the second bound (`Mag`) afterwards;
* `mag_tail`: one block of the second bound;
* `row_step_plain`, `row_step_masked`: one block of the loop (lines 72–109) keeps `RowSt`.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## The row's keys and context -/

/-- Key `j` is one query `n` attends to (`keys d n`). -/
def alOf (d : Inst) (n j : ℕ) : Bool := decide (j < d.N_CTX ∧ (d.causal = false ∨ j ≤ n))

theorem mem_keys {d : Inst} {n j : ℕ} : j ∈ keys d n ↔ alOf d n j = true := by
  unfold keys alOf; simp

/-- The library's constants of the score's `tl.dot` at `HEAD_DIM = 64, 128, 256`, and the bound
`expErr ≤ 13` at the coverage cap `Xh`. -/
theorem dot8_consts {D : ℕ} (hD : D = 64 ∨ D = 128 ∨ D = 256) :
    0 ≤ Rules.Num.relDot8 D ∧ Rules.Num.relDot8 D ≤ 1 / 8 ∧ 0 ≤ Rules.Num.absDot8 D ∧
      Rules.Num.absDot8 D ≤ 1 / 2 ^ 117 ∧ Xh D ≤ 512 ∧
      (Rules.Num.relDot8 D + 1 / 2 ^ 19) * (3 / 2 * Xh D) + 1 / 2 ^ 50 ≤ 13 := by
  rcases hD with rfl | rfl | rfl <;>
    simp only [Rules.Num.relDot8, Rules.Num.absDot8, Xh, Rules.Num.chunks8] <;> norm_num

/-- Query `n` of head `bh` at the bound `X`: `qk_scale` `c`, the exact base-2 scale
`sm_scale / ln 2`, the exact scores and their magnitudes, the dot's constants for `HEAD_DIM`
products, `Y = 3X/2` and `E = expErr`. -/
noncomputable def rowCtx (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X : ℚ) : RowCtx where
  c := c
  w := (sigma d : ℝ) / Real.log 2
  Y := 3 / 2 * (X : ℝ)
  E := (expErr d.HEAD_DIM X : ℝ)
  rhoD := (Rules.Num.relDot8 d.HEAD_DIM : ℝ)
  aD := (Rules.Num.absDot8 d.HEAD_DIM : ℝ)
  sv j := (score d init bh n j : ℝ)
  bv j := (scoreAbs d init bh n j : ℝ)
  al := alOf d n

theorem score_le (d : Inst) (init : Mem) (bh n j : ℕ) :
    |(score d init bh n j : ℝ)| ≤ (scoreAbs d init bh n j : ℝ) := by
  have : |score d init bh n j| ≤ scoreAbs d init bh n j := Finset.abs_sum_le_sum_abs _ _
  exact_mod_cast this

theorem scoreAbs_nonneg (d : Inst) (init : Mem) (bh n j : ℕ) : 0 ≤ scoreAbs d init bh n j :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

theorem sigma_pos {d : Inst} (hd : d.ok) : 0 < sigma d := by
  have := hd.scale.2.1; unfold sigma; linarith [show (0 : ℚ) < 1 / 2 ^ 64 by positivity]

theorem w_le {σ : ℝ} (hσ : 0 ≤ σ) : σ / Real.log 2 ≤ 3 / 2 * σ := by
  have h1 := Real.log_two_gt_d9
  have hl : 0 < Real.log 2 := by linarith
  rw [div_le_iff₀ hl]; nlinarith

/-- **The row's context is `RowCtx.Ok`** for a bound `X ≤ Xh HEAD_DIM` on its keys' scaled score
magnitudes. -/
theorem rowCtx_ok {d : Inst} (hd : d.ok) (init : Mem) {c : ℝ} (hc0 : 0 < c) (hc65 : c ≤ 2 ^ 65)
    (hc32 : c ≤ 3 / 2 * (sigma d : ℝ))
    (hcw : |c - (sigma d : ℝ) / Real.log 2| ≤ (sigma d : ℝ) / Real.log 2 / 2 ^ 22)
    (bh : ℕ) {n : ℕ} (hn : n < d.N_CTX) {X : ℚ} (hX : X ≤ Xh d.HEAD_DIM)
    (hXk : ∀ j ∈ keys d n, sigma d * scoreAbs d init bh n j ≤ X) :
    (rowCtx d init c bh n X).Ok ∧ 0 ≤ X := by
  have hσ := sigma_pos hd
  have hσ' : (0 : ℝ) < (sigma d : ℝ) := by exact_mod_cast hσ
  have hX0 : 0 ≤ X := by
    have h0 : (0 : ℕ) ∈ keys d n := mem_keys.mpr (by unfold alOf; simp; omega)
    have := hXk 0 h0
    have := mul_nonneg hσ.le (scoreAbs_nonneg d init bh n 0)
    linarith
  obtain ⟨hr0, hr1, ha0, ha1, hXh, hEh⟩ := dot8_consts hd.head_dim
  have hX0' : (0 : ℝ) ≤ X := by exact_mod_cast hX0
  have hX512 : (X : ℝ) ≤ 512 := by exact_mod_cast hX.trans hXh
  have hl : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hE : expErr d.HEAD_DIM X ≤ 13 := by
    unfold expErr ybound
    have : (Rules.Num.relDot8 d.HEAD_DIM + 1 / 2 ^ 19) * (3 / 2 * X) ≤
        (Rules.Num.relDot8 d.HEAD_DIM + 1 / 2 ^ 19) * (3 / 2 * Xh d.HEAD_DIM) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    linarith
  refine ⟨⟨hc0, hc65, div_pos hσ' hl, hcw, by show (0 : ℝ) ≤ 3 / 2 * (X : ℝ); positivity,
    by show 3 / 2 * (X : ℝ) ≤ 768; linarith,
    by show (0 : ℝ) ≤ (Rules.Num.relDot8 d.HEAD_DIM : ℝ); exact_mod_cast hr0,
    by show (Rules.Num.relDot8 d.HEAD_DIM : ℝ) ≤ 1 / 8
       have := (Rat.cast_le (K := ℝ)).mpr hr1; push_cast at this; exact this,
    by show (0 : ℝ) ≤ (Rules.Num.absDot8 d.HEAD_DIM : ℝ); exact_mod_cast ha0,
    by show (Rules.Num.absDot8 d.HEAD_DIM : ℝ) ≤ 1 / 2 ^ 117
       have := (Rat.cast_le (K := ℝ)).mpr ha1; push_cast at this; exact this, ?_,
    by show (expErr d.HEAD_DIM X : ℝ) ≤ 13; exact_mod_cast hE,
    fun j => score_le d init bh n j, ?_⟩, hX0⟩
  · show (expErr d.HEAD_DIM X : ℝ) =
      ((Rules.Num.relDot8 d.HEAD_DIM : ℝ) + 1 / 2 ^ 19) * (3 / 2 * (X : ℝ)) + 1 / 2 ^ 50
    unfold expErr ybound; push_cast; ring
  · intro j hj
    show c * (scoreAbs d init bh n j : ℝ) ≤ 3 / 2 * (X : ℝ) ∧
      (sigma d : ℝ) / Real.log 2 * (scoreAbs d init bh n j : ℝ) ≤ 3 / 2 * (X : ℝ)
    have h1 := hXk j (mem_keys.mpr hj)
    have h1' : (sigma d : ℝ) * (scoreAbs d init bh n j : ℝ) ≤ X := by exact_mod_cast h1
    have h2 : (0 : ℝ) ≤ (scoreAbs d init bh n j : ℝ) := by exact_mod_cast scoreAbs_nonneg d init bh n j
    have hw := w_le hσ'.le
    constructor
    · have := mul_le_mul_of_nonneg_right hc32 h2
      nlinarith
    · have := mul_le_mul_of_nonneg_right hw h2
      nlinarith

/-- With every attended key in the first `T` blocks, the row's sums over those blocks are its
sums over its keys. -/
theorem psum_keys {d : Inst} {n T : ℕ} (hT : ∀ j, alOf d n j = true → j < 64 * T) (f : ℕ → ℝ) :
    psum (alOf d n) T f = ∑ j ∈ keys d n, f j := by
  unfold psum
  rw [← Finset.sum_filter]
  refine Finset.sum_congr ?_ fun _ _ => rfl
  ext j
  simp only [Finset.mem_filter, Finset.mem_range]
  constructor
  · rintro ⟨-, h⟩; exact mem_keys.mpr h
  · intro h; have h' := mem_keys.mp h; exact ⟨hT j h', h'⟩

/-! ## A block's row maximum -/

theorem mem_leaves_64 {tr : Tree} (h : tr.Covers 64) {i : ℕ} : i ∈ tr.leaves ↔ i < 64 := by
  rw [h.mem_iff, List.mem_range]

/-- **`tl.max` of a block's row** (any tree) of finite lanes: one of them, not below any. -/
theorem max64 {S : Fin 64 → F32} {mx : F32}
    (h : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (FloatV.MaxRel true) (laneSeq S) mx)
    (hf : ∀ jj, (S jj).isFinite = true) : (∃ jj, mx = S jj) ∧ ∀ jj, (S jj).toReal ≤ mx.toReal := by
  obtain ⟨tr, hcov, hev⟩ := h
  have hls : ∀ i (hi : i < 64), laneSeq S i = S ⟨i, hi⟩ := fun i hi => by simp [laneSeq, hi]
  obtain ⟨⟨i, hi, hmi⟩, hle⟩ := Tree.eval_maxRel hev fun i hi => by
    rw [hls i ((mem_leaves_64 hcov).mp hi)]; exact isNaN_of_fin (hf _)
  have hi' := (mem_leaves_64 hcov).mp hi
  rw [hls i hi'] at hmi
  have hmf : mx.isFinite = true := by rw [hmi]; exact hf _
  refine ⟨⟨⟨i, hi'⟩, hmi⟩, fun jj => ?_⟩
  have := hle jj ((mem_leaves_64 hcov).mpr jj.2)
  simp only [ite_true] at this
  rw [hls jj jj.2] at this
  exact le_toReal (isNaN_of_fin (hf _)) (isNaN_of_fin hmf) this

/-! ## The accumulator's error bounds -/

/-- The accumulator's error bound against the exact partial sums, block by block (in the units of
the exact weights): `Model.errS` in the row's terms. -/
noncomputable def esR (R : RowCtx) (ve : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => (1 + rhoAR R) * esR R ve t + rhoAR R * |Sx R ve t| + psiAR R * Mg R ve t

/-- The accumulator's absolute error bound (in the units of the running maximum): `Model.errB`
in the row's terms. -/
noncomputable def ebR (R : RowCtx) (ve : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => (1 + rhoAR R) * ebR R ve t + (1 + rho8r) * om8r * AbV R ve t + absAr

theorem esR_nonneg {R : RowCtx} (hR : R.Ok) (ve : ℕ → ℝ) : ∀ t, 0 ≤ esR R ve t
  | 0 => le_rfl
  | t + 1 => by
    have h1 := esR_nonneg hR ve t
    have h2 := rhoAR_nonneg hR
    have h3 := psiAR_nonneg hR
    have h4 := Mg_nonneg R ve t
    show 0 ≤ (1 + rhoAR R) * esR R ve t + rhoAR R * |Sx R ve t| + psiAR R * Mg R ve t
    positivity

theorem ebR_nonneg {R : RowCtx} (hR : R.Ok) (ve : ℕ → ℝ) : ∀ t, 0 ≤ ebR R ve t
  | 0 => le_rfl
  | t + 1 => by
    have h1 := ebR_nonneg hR ve t
    have h2 := rhoAR_nonneg hR
    have h4 := AbV_nonneg R ve t
    show 0 ≤ (1 + rhoAR R) * ebR R ve t + (1 + rho8r) * om8r * AbV R ve t + absAr
    unfold rho8r om8r absAr
    positivity

/-! ## The state of a row -/

/-- A row's `m_i`, `l_i` and `acc` (an element per column `e`, whose `v` values are `ve e`) after `t`
blocks: before the first, `-∞`, `1.0` and `0.0`; afterwards `MInv`, `LInv` and `AInv` with the
bounds `esR`, `ebR`, and the second bound: `l_i ≥ lminB t`, and `|acc − r·l_i| ≤ magA·l_i + magD`
for every `r` and every bound `K` on `|v_j − r|` over the row's keys. -/
def RowSt (R : RowCtx) {D : ℕ} (ve : Fin D → ℕ → ℝ) (t : ℕ) (m l : F32) (acc : Fin D → F32) : Prop :=
  (t = 0 ∧ m = FloatV.negInf ∧ l.isFinite = true ∧ l.toReal = 1 ∧
      ∀ e, (acc e).isFinite = true ∧ (acc e).toReal = 0) ∨
    (0 < t ∧ MInv R t m ∧ LInv R t m l ∧
      (∀ e, AInv R (ve e) t m (acc e) (esR R (ve e) t) (ebR R (ve e) t)) ∧ lminB t ≤ l.toReal ∧
      ∀ e (r K : ℝ), 0 ≤ K → (∀ j, R.al j = true → |ve e j - r| ≤ K) →
        |(acc e).toReal - r * l.toReal| ≤ magA K |r| t * l.toReal + magD K |r| t)

theorem RowSt.old {R : RowCtx} {D : ℕ} {ve : Fin D → ℕ → ℝ} {t : ℕ} {m l : F32} {acc : Fin D → F32}
    (h : RowSt R ve t m l acc) : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m) := by
  rcases h with ⟨h0, hm, -⟩ | ⟨ht, hmi, -⟩
  · exact Or.inl ⟨h0, hm⟩
  · exact Or.inr ⟨ht, hmi⟩

/-- **The second bound, one block** (lines 82–106): `l_i ≥ lminB (t + 1)`, and `|acc − r·l_i| ≤
magA·l_i + magD` for every `r` and every bound `K` on `|v_j − r|` over the row's keys. `hcomb` is
the dot's error (`astep`); `hnew`: the maximum is kept, or a lane's exponent is near `0`. -/
theorem mag_tail {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve : Fin D → ℕ → ℝ}
    {m m' dm al l lij l' : F32} {x p : Fin 64 → F32} {q : Fin 64 → F8E5} {vt : Fin 64 → Fin D → F8E5}
    {acc a1 acc' : Fin D → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x) (hin : AlphaIn t m m' dm)
    (hnew : (0 < t ∧ m' = m) ∨ ∃ jj : Fin 64, |(x jj).toReal| ≤ 1 / 2 ^ 11)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj)) (hal : P.math .exp2 .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij)
    (hq : ∀ jj, P.cvt Format.binary32 Format.e5m2 .rn (p jj) (q jj))
    (ha1 : ∀ e, FloatV.NaNEq (FloatV.mul (acc e) al) (a1 e))
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj))
    (hcomb : ∀ e, |(acc' e).toReal - ((a1 e).toReal + ∑ jj, (q jj).toReal * (vt jj e).toReal)| ≤
      rho8r * (|(a1 e).toReal| + ∑ jj, |(q jj).toReal * (vt jj e).toReal|) + 16641 / 2 ^ 139)
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hold : RowSt R ve t m l acc) :
    lminB (t + 1) ≤ l'.toReal ∧ ∀ e (r K : ℝ), 0 ≤ K → (∀ j, R.al j = true → |ve e j - r| ≤ K) →
      |(acc' e).toReal - r * l'.toReal| ≤ magA K |r| (t + 1) * l'.toReal + magD K |r| (t + 1) := by
  -- the lanes
  have hpq := lane_pq hexp hcvt hR hX hp hq
  have hpf : ∀ jj, (p jj).isFinite = true := fun jj => (hpq jj).1
  have hpb : ∀ jj, |(p jj).toReal| ≤ 2 := fun jj => by have := (hpq jj).2.1; linarith
  have hp0 : ∀ jj, -(1 / 2 ^ 125) ≤ (p jj).toReal := fun jj => (hpq jj).2.2.1
  -- `l_ij` (line 85)
  have hSA : ∑ jj, |(p jj).toReal| ≤ 2 ^ 100 := by
    have := lane_sum_le (a := fun jj => |(p jj).toReal|) (c := fun _ => (0 : ℝ)) (k := 0) (e := 2)
      fun jj => by simp only [zero_mul, zero_add]; exact hpb jj
    norm_num at this ⊢; linarith
  obtain ⟨hlijf, hlije⟩ := sum64_err hsum hpf hSA
  rw [show ((1 : ℝ) + 1 / 2 ^ 24) ^ 63 - 1 = gamr from rfl] at hlije
  obtain ⟨-, -, -, hlijb⟩ := block_l hexp hR hX hp hsum
  -- `alpha` (line 84)
  have hmi : 0 < t → MInv R t m := fun ht0 => by
    rcases hold with ⟨h, -⟩ | ⟨-, hmi, -⟩
    · omega
    · exact hmi
  obtain ⟨haf, hab, ha0, hae⟩ := alpha_facts hexp hni hR hM' hin hal hmi
  have hα : -(1 / 2 ^ 126) ≤ al.toReal ∧ |al.toReal| ≤ 1 + 1 / 2 ^ 12 ∧
      (0 < t → m' = m → 1 - 1 / 2 ^ 12 ≤ al.toReal) := by
    rcases Nat.eq_zero_or_pos t with h0 | ht0
    · rw [ha0 h0]; exact ⟨by norm_num, by norm_num, fun h => absurd h (by omega)⟩
    · obtain ⟨hle, hae'⟩ := hae ht0
      have hl1 : (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 := two_rpow_le_one (by linarith)
      have hl0 := two_rpow_pos (m.toReal - m'.toReal)
      have hd0 := dmaxR_nonneg hR
      have hd1 := dmaxR_le hR
      have he0 : 0 ≤ (1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1 := by nlinarith
      have he1 : (1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1 ≤ 1 / 2 ^ 12 - 1 / 2 ^ 125 := by nlinarith
      have h1 : ((1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) ≤
          (1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1 := mul_le_of_le_one_right he0 hl1
      have h2 : ((1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1) * (2 : ℝ) ^ (m.toReal - m'.toReal) ≤
          (2 : ℝ) ^ (m.toReal - m'.toReal) := mul_le_of_le_one_left hl0.le (by linarith)
      have hk := abs_le.mp hae'
      refine ⟨by linarith, ?_, fun _ hmm => ?_⟩
      · rw [abs_le]; constructor <;> linarith
      · have hk1 := hk.1
        rw [hmm, sub_self, Real.rpow_zero] at hk1
        linarith
  -- the old `l_i`
  have hlb : l.isFinite = true ∧ 0 ≤ l.toReal ∧ l.toReal ≤ 2 ^ 44 := by
    rcases hold with ⟨-, -, hf, h1, -⟩ | ⟨ht0, hmi', hL, -⟩
    · exact ⟨hf, by rw [h1]; norm_num, by rw [h1]; norm_num⟩
    · exact ⟨hL.1, hL.2.1.le, le_of_abs_le (hL.bound hR (by omega) hmi')⟩
  -- line 106
  have hroomL : |l.toReal * al.toReal| + |lij.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul, abs_of_nonneg hlb.2.1]
    have := mul_le_mul hlb.2.2 hab (abs_nonneg _) (by norm_num)
    linarith
  obtain ⟨-, hl'e⟩ := f32_mulAdd hl' hlb.1 haf hlijf hroomL
  -- `l_i` from below
  have hcase : (lminB t ≤ l.toReal ∧ 1 - 1 / 2 ^ 12 ≤ al.toReal) ∨
      ∃ jj, 1 - 1 / 2 ^ 10 ≤ (p jj).toReal := by
    rcases hnew with ⟨ht0, hmm⟩ | ⟨jj, hxj⟩
    · left
      refine ⟨?_, hα.2.2 ht0 hmm⟩
      rcases hold with ⟨h0, -⟩ | ⟨-, -, -, -, hlm, -⟩
      · omega
      · exact hlm
    · exact Or.inr ⟨jj, p_big hexp (hX jj).1 hxj (hp jj)⟩
  refine ⟨lmin_real (by omega) hlb.2.1 hlb.2.2 hα.1 hp0 hpb hlije hl'e hcase, fun e r K hK hKv => ?_⟩
  -- the column's lanes
  have hlE : ∀ jj, |(q jj).toReal * (vt jj e).toReal - r * (p jj).toReal| ≤
        (9 / 8 * K + |r| / 8) * |(p jj).toReal| + (K + |r|) / 2 ^ 17 ∧
      |(q jj).toReal * (vt jj e).toReal| ≤ (K + |r|) * (9 / 8 * |(p jj).toReal| + 1 / 2 ^ 17) := fun jj =>
    lane_E hK (fun h => ⟨(hpq jj).2.2.2.2.1 h, by rw [(hvt jj e).2]; exact hKv _ h⟩) (hpq jj).2.2.2.2.2
  -- the old element
  have hacc : (acc e).isFinite = true ∧ |(acc e).toReal| ≤ 2 ^ 58 ∧
      |al.toReal| * |(acc e).toReal - r * l.toReal| ≤
        |al.toReal| * magA K |r| t * l.toReal + (1 + 1 / 2 ^ 12) * magD K |r| t := by
    rcases hold with ⟨h0, -, -, -, ha⟩ | ⟨ht0, -, -, hA, -, hE⟩
    · refine ⟨(ha e).1, by rw [(ha e).2]; norm_num, ?_⟩
      rw [ha0 h0, h0, show magD K |r| 0 = 0 from rfl]
      simp
    · refine ⟨(hA e).1, (hA e).2.1.trans (crA_le (by omega)), ?_⟩
      have h1 := mul_le_mul_of_nonneg_left (hE e r K hK hKv) (abs_nonneg al.toReal)
      have h2 : |al.toReal| * magD K |r| t ≤ (1 + 1 / 2 ^ 12) * magD K |r| t :=
        mul_le_mul_of_nonneg_right hα.2.1 (magD_nonneg hK (abs_nonneg r) t)
      have e1 : |al.toReal| * (magA K |r| t * l.toReal + magD K |r| t) =
          |al.toReal| * magA K |r| t * l.toReal + |al.toReal| * magD K |r| t := by ring
      linarith
  -- line 95
  have hroom1 : |(acc e).toReal * al.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul]; have := mul_le_mul hacc.2.1 hab (abs_nonneg _) (by norm_num); linarith
  obtain ⟨-, ha1e⟩ := f32_mul (ha1 e) hacc.1 haf hroom1
  exact estep_real (q := fun jj => (q jj).toReal) (p := fun jj => (p jj).toReal)
    (v := fun jj => (vt jj e).toReal) hK (magA_nonneg hK (abs_nonneg r) t) hlb.2.1 hlb.2.2 hα.1 hp0
    (fun jj => (hlE jj).1) (fun jj => (hlE jj).2) hacc.2.2 ha1e (hcomb e) hlije hl'e

/-- **The rest of a block** (lines 82–109), once the new maximum and the exponents are known. -/
theorem row_tail {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve : Fin D → ℕ → ℝ} (hve : ∀ e j, |ve e j| ≤ 57344)
    {m m' dm al l lij l' : F32} {x p : Fin 64 → F32} {q : Fin 64 → F8E5} {vt : Fin 64 → Fin D → F8E5}
    {acc a1 acc' : Fin D → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x) (hin : AlphaIn t m m' dm)
    (hnew : (0 < t ∧ m' = m) ∨ ∃ jj : Fin 64, |(x jj).toReal| ≤ 1 / 2 ^ 11)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj)) (hal : P.math .exp2 .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij)
    (hq : ∀ jj, P.cvt Format.binary32 Format.e5m2 .rn (p jj) (q jj))
    (ha1 : ∀ e, FloatV.NaNEq (FloatV.mul (acc e) al) (a1 e))
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj))
    (hdot : ∀ e, (a1 e).isFinite = true → (∀ k, (q k).isFinite = true) →
      |(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat| +
        (Rules.Num.relDot8 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot8 64) ≤
          Format.binary32.maxVal →
      (acc' e).isFinite = true ∧ |(acc' e).toRat - ((a1 e).toRat + ∑ k, (q k).toRat * (vt k e).toRat)| ≤
        Rules.Num.relDot8 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot8 64)
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hold : RowSt R ve t m l acc) : RowSt R ve (t + 1) m' l' acc' := by
  have hA : ∀ e, AInv R (ve e) (t + 1) m' (acc' e) (esR R (ve e) (t + 1)) (ebR R (ve e) (t + 1)) ∧
      |(acc' e).toReal - ((a1 e).toReal + ∑ jj, (q jj).toReal * (vt jj e).toReal)| ≤
        rho8r * (|(a1 e).toReal| + ∑ jj, |(q jj).toReal * (vt jj e).toReal|) + 16641 / 2 ^ 139 := fun e => by
    refine astep hexp hni hcvt hR ht (hve e) (esR_nonneg hR _ t) (ebR_nonneg hR _ t) hM' hX hp hq hin hal
      (ha1 e) (fun jj => hvt jj e) (hdot e) ?_
    rcases hold with ⟨h0, -, -, -, ha⟩ | ⟨ht0, hmi, -, hA, -⟩
    · exact Or.inl ⟨h0, (ha e).1, (ha e).2⟩
    · exact Or.inr ⟨ht0, hmi, hA e⟩
  obtain ⟨hlm, hE⟩ := mag_tail hexp hni hcvt hR ht hM' hX hin hnew hp hal hsum hq ha1 hvt
    (fun e => (hA e).2) hl' hold
  right
  refine ⟨by omega, hM', ?_, fun e => (hA e).1, hlm, hE⟩
  refine lstep hexp hni hR ht hM' hX hp hin hal hsum hl' ?_
  rcases hold with ⟨h0, -, hlf, hlb, -⟩ | ⟨ht0, hmi, hL, -⟩
  · exact Or.inl ⟨h0, hlf, hlb⟩
  · exact Or.inr ⟨ht0, hmi, hL⟩

/-- **One block, `STAGE` 1 or 3** (lines 72–73, 80–109): every key attended. -/
theorem row_step_plain {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve : Fin D → ℕ → ℝ} (hve : ∀ e j, |ve e j| ≤ 57344)
    {cq m mx a m' dm al l lij l' : F32} {S Sc x p : Fin 64 → F32} {q : Fin 64 → F8E5}
    {vt : Fin 64 → Fin D → F8E5} {acc a1 acc' : Fin D → F32}
    (hcq : cq.isFinite = true) (hcqv : cq.toReal = R.c)
    (hS : ∀ jj : Fin 64, (S jj).isFinite = true ∧
      |(S jj).toReal - R.sv (64 * t + jj)| ≤ R.rhoD * R.bv (64 * t + jj) + R.aD)
    (hall : ∀ jj : Fin 64, R.al (64 * t + jj) = true)
    (hmx : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (FloatV.MaxRel true) (laneSeq S) mx)
    (ha : FloatV.NaNEq (FloatV.mul mx cq) a) (hm' : FloatV.MaxRel true m a m')
    (hSc : ∀ jj, FloatV.NaNEq (FloatV.mul (S jj) cq) (Sc jj))
    (hx : ∀ jj, AddRel fusion (Sc jj) (FloatV.neg m') (x jj))
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj))
    (hdm : FloatV.NaNEq (FloatV.sub m m') dm) (hal : P.math .exp2 .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij)
    (hq : ∀ jj, P.cvt Format.binary32 Format.e5m2 .rn (p jj) (q jj))
    (ha1 : ∀ e, FloatV.NaNEq (FloatV.mul (acc e) al) (a1 e))
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj))
    (hdot : ∀ e, (a1 e).isFinite = true → (∀ k, (q k).isFinite = true) →
      |(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat| +
        (Rules.Num.relDot8 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot8 64) ≤
          Format.binary32.maxVal →
      (acc' e).isFinite = true ∧ |(acc' e).toRat - ((a1 e).toRat + ∑ k, (q k).toRat * (vt k e).toRat)| ≤
        Rules.Num.relDot8 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot8 64)
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hold : RowSt R ve t m l acc) : RowSt R ve (t + 1) m' l' acc' := by
  obtain ⟨hM', hX, hin, hnew⟩ := mstep_plain hR hcq hcqv hS hall (max64 hmx fun jj => (hS jj).1) ha hm' hSc hx
    hdm hold.old
  exact row_tail hexp hni hcvt hR ht hve hM' hX hin hnew hp hal hsum hq ha1 hvt hdot hl' hold

/-- **One block, `STAGE` 2** (lines 72–78, 82–109): the causal mask. Before the first block, the
block's first key is attended. -/
theorem row_step_masked {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve : Fin D → ℕ → ℝ} (hve : ∀ e j, |ve e j| ≤ 57344)
    {cq m mx m' dm al l lij l' : F32} {S w x' x p : Fin 64 → F32} {q : Fin 64 → F8E5}
    {vt : Fin 64 → Fin D → F8E5} {acc a1 acc' : Fin D → F32}
    (hcq : cq.isFinite = true) (hcqv : cq.toReal = R.c)
    (hS : ∀ jj : Fin 64, (S jj).isFinite = true ∧
      |(S jj).toReal - R.sv (64 * t + jj)| ≤ R.rhoD * R.bv (64 * t + jj) + R.aD)
    (hglob : ∀ jj : Fin 64, R.c * R.bv (64 * t + jj) ≤ 1536)
    (hw : ∀ jj : Fin 64, (w jj).isFinite = true ∧ (R.al (64 * t + jj) = true → (w jj).toReal = 0) ∧
      (R.al (64 * t + jj) = false → (w jj).toReal = -1000000))
    (hx' : ∀ jj, FloatV.NaNEq (FloatV.add (FloatV.mul (S jj) cq) (w jj)) (x' jj) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (S jj) cq (w jj)) (x' jj)))
    (hmx : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (FloatV.MaxRel true) (laneSeq x') mx)
    (hm' : FloatV.MaxRel true m mx m')
    (hx : ∀ jj, FloatV.NaNEq (FloatV.sub (x' jj) m') (x jj))
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj))
    (hdm : FloatV.NaNEq (FloatV.sub m m') dm) (hal : P.math .exp2 .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij)
    (hq : ∀ jj, P.cvt Format.binary32 Format.e5m2 .rn (p jj) (q jj))
    (ha1 : ∀ e, FloatV.NaNEq (FloatV.mul (acc e) al) (a1 e))
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj))
    (hdot : ∀ e, (a1 e).isFinite = true → (∀ k, (q k).isFinite = true) →
      |(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat| +
        (Rules.Num.relDot8 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot8 64) ≤
          Format.binary32.maxVal →
      (acc' e).isFinite = true ∧ |(acc' e).toRat - ((a1 e).toRat + ∑ k, (q k).toRat * (vt k e).toRat)| ≤
        Rules.Num.relDot8 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot8 64)
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (h0 : t = 0 → R.al (64 * t) = true)
    (hold : RowSt R ve t m l acc) : RowSt R ve (t + 1) m' l' acc' := by
  have hx'f : ∀ jj, (x' jj).isFinite = true := fun jj => by
    have hc := hR.c_pos
    obtain ⟨hSf, hSe⟩ := hS jj
    have hroom : |(S jj).toReal * cq.toReal| + |(w jj).toReal| ≤ 2 ^ 120 := by
      rw [abs_mul, hcqv, abs_of_pos hc]
      have hb := hR.sv_le (64 * t + jj)
      have hbv : 0 ≤ R.bv (64 * t + jj) := (abs_nonneg _).trans hb
      have h1 := abs_sub_abs_le_abs_sub (S jj).toReal (R.sv (64 * t + jj))
      have h2 : |(S jj).toReal| ≤ (1 + R.rhoD) * R.bv (64 * t + jj) + R.aD := by
        have := hR.rhoD_nonneg; nlinarith
      have h3 : R.c * |(S jj).toReal| ≤ R.c * ((1 + R.rhoD) * R.bv (64 * t + jj) + R.aD) :=
        mul_le_mul_of_nonneg_left h2 hc.le
      have h4 := hglob jj
      have h5 := hR.rhoD_le
      have h6 := hR.aD_le
      have h7 := hR.c_le
      have h8 : R.c * R.aD ≤ 2 ^ 65 * (1 / 2 ^ 117) := mul_le_mul h7 h6 hR.aD_nonneg (by norm_num)
      have h9 : |(w jj).toReal| ≤ 1000000 := by
        obtain ⟨-, hw1, hw2⟩ := hw jj
        cases hal : R.al (64 * t + jj)
        · rw [hw2 hal]; norm_num
        · rw [hw1 hal]; norm_num
      have : R.c * ((1 + R.rhoD) * R.bv (64 * t + jj) + R.aD) =
          (1 + R.rhoD) * (R.c * R.bv (64 * t + jj)) + R.c * R.aD := by ring
      have h10 : (1 + R.rhoD) * (R.c * R.bv (64 * t + jj)) ≤ 2 * 1536 :=
        mul_le_mul (by linarith) h4 (by positivity) (by norm_num)
      nlinarith
    exact (f32_mulAdd (hx' jj) hSf hcq (hw jj).1 hroom).1
  have hold' : (t = 0 ∧ m = FloatV.negInf ∧ R.al (64 * t) = true) ∨ (0 < t ∧ MInv R t m) := by
    rcases hold.old with ⟨ht0, hm⟩ | h
    · exact Or.inl ⟨ht0, hm, h0 ht0⟩
    · exact Or.inr h
  obtain ⟨hM', hX, hin, hnew⟩ := mstep_masked hR hcq hcqv hS hglob hw hx' (max64 hmx hx'f) hm' hx hdm hold'
  exact row_tail hexp hni hcvt hR ht hve hM' hX hin hnew hp hal hsum hq ha1 hvt hdot hl' hold

end Kernels.AttnFwdE5m2
