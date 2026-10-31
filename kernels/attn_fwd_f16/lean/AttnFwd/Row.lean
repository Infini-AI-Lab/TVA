import AttnFwd.Facts

/-!
# `attn_fwd_f16`: one query row of the instance

* `alOf`, `rowCtx`: the keys a row attends to, and its `RowCtx` at a bound `X` on its scaled score
  magnitudes; `rowCtx_ok`;
* `max64`: the row maximum of a block (`tl.max`, any tree);
* `RowSt`: a row's `m_i`, `l_i` and `acc` after `t` blocks: their initial values when `t = 0`,
  and `MInv`, `LInv`, `AInv` afterwards;
* `row_step_plain`, `row_step_masked`: one block of the loop (lines 72–107) keeps `RowSt`.
-/

namespace Kernels.AttnFwd

open TVA TVA.Math.Float

/-! ## The row's keys and context -/

/-- Key `j` is one query `n` attends to (`keys d n`). -/
def alOf (d : Inst) (n j : ℕ) : Bool := decide (j < d.N_CTX ∧ (d.causal = false ∨ j ≤ n))

theorem mem_keys {d : Inst} {n j : ℕ} : j ∈ keys d n ↔ alOf d n j = true := by
  unfold keys alOf; simp

/-- Query `n` of head `bh` at the bound `X`: `qk_scale` `c`, the exact scores and their
magnitudes, the dot's constants for `HEAD_DIM` products, `Y = 3X/2` and `E = expErr`. -/
noncomputable def rowCtx (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X : ℚ) : RowCtx where
  c := c
  Y := 3 / 2 * (X : ℝ)
  E := (expErr d.HEAD_DIM X : ℝ)
  rhoD := (Rules.Num.relDot16 d.HEAD_DIM : ℝ)
  aD := (Rules.Num.absDot16 d.HEAD_DIM : ℝ)
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

/-- **The row's context is `RowCtx.Ok`** for a bound `X ≤ 2^10` on its keys' scaled score
magnitudes. -/
theorem rowCtx_ok {d : Inst} (hd : d.ok) (init : Mem) {c : ℝ} (hc0 : 0 < c) (hc65 : c ≤ 2 ^ 65)
    (hc32 : c ≤ 3 / 2 * (sigma d : ℝ)) (bh : ℕ) {n : ℕ} (hn : n < d.N_CTX) {X : ℚ} (hX : X ≤ 2 ^ 10)
    (hXk : ∀ j ∈ keys d n, sigma d * scoreAbs d init bh n j ≤ X) :
    (rowCtx d init c bh n X).Ok ∧ 0 ≤ X := by
  have hσ := sigma_pos hd
  have hX0 : 0 ≤ X := by
    have h0 : (0 : ℕ) ∈ keys d n := mem_keys.mpr (by unfold alOf; simp; omega)
    have := hXk 0 h0
    have := mul_nonneg hσ.le (scoreAbs_nonneg d init bh n 0)
    linarith
  have hD := headDim_le hd
  have hD' : (d.HEAD_DIM : ℝ) ≤ 256 := by exact_mod_cast hD
  have hX0' : (0 : ℝ) ≤ X := by exact_mod_cast hX0
  have hX' : (X : ℝ) ≤ 2 ^ 10 := by exact_mod_cast hX
  obtain ⟨hE0, hE1⟩ := expErr_le hD hX0 hX
  refine ⟨⟨hc0, hc65, by show (0 : ℝ) ≤ 3 / 2 * (X : ℝ); positivity,
    by show 3 / 2 * (X : ℝ) ≤ 1536; linarith, ?_, ?_, ?_, ?_, ?_, hE1, fun j => score_le d init bh n j, ?_⟩, hX0⟩
  · show (0 : ℝ) ≤ (Rules.Num.relDot16 d.HEAD_DIM : ℝ); rw [relDot16_cast]; positivity
  · show (Rules.Num.relDot16 d.HEAD_DIM : ℝ) ≤ 1 / 2 ^ 13
    rw [relDot16_cast, div_le_div_iff₀ (by norm_num) (by norm_num)]; nlinarith
  · show (0 : ℝ) ≤ (Rules.Num.absDot16 d.HEAD_DIM : ℝ); rw [absDot16_cast]; positivity
  · show (Rules.Num.absDot16 d.HEAD_DIM : ℝ) ≤ 1 / 2 ^ 117
    rw [absDot16_cast, div_le_div_iff₀ (by norm_num) (by norm_num)]; nlinarith
  · show (expErr d.HEAD_DIM X : ℝ) = ((Rules.Num.relDot16 d.HEAD_DIM : ℝ) + 1 / 2 ^ 20) * (3 / 2 * (X : ℝ)) + 1 / 2 ^ 50
    exact expErr_cast _ _
  · intro j hj
    show c * (scoreAbs d init bh n j : ℝ) ≤ 3 / 2 * (X : ℝ)
    have h1 := hXk j (mem_keys.mpr hj)
    have h1' : (sigma d : ℝ) * (scoreAbs d init bh n j : ℝ) ≤ X := by exact_mod_cast h1
    have h2 : (0 : ℝ) ≤ (scoreAbs d init bh n j : ℝ) := by exact_mod_cast scoreAbs_nonneg d init bh n j
    have := mul_le_mul_of_nonneg_right hc32 h2
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

/-! ## The state of a row -/

/-- A row's `m_i`, `l_i` and `acc` (an element per column `e`, whose `v` values are `ve e`) after `t`
blocks: before the first, `-∞`, `1.0` and `0.0`; afterwards `MInv`, `LInv` and `AInv`. -/
def RowSt (R : RowCtx) {D : ℕ} (ve : Fin D → ℕ → ℝ) (t : ℕ) (m l : F32) (acc : Fin D → F32) : Prop :=
  (t = 0 ∧ m = FloatV.negInf ∧ l.isFinite = true ∧ |l.toReal| ≤ 1 ∧
      ∀ e, (acc e).isFinite = true ∧ (acc e).toReal = 0) ∨
    (0 < t ∧ MInv R t m ∧ LInv R t m l ∧ ∀ e, AInv R (ve e) t m (acc e))

theorem RowSt.old {R : RowCtx} {D : ℕ} {ve : Fin D → ℕ → ℝ} {t : ℕ} {m l : F32} {acc : Fin D → F32}
    (h : RowSt R ve t m l acc) : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m) := by
  rcases h with ⟨h0, hm, -⟩ | ⟨ht, hmi, -⟩
  · exact Or.inl ⟨h0, hm⟩
  · exact Or.inr ⟨ht, hmi⟩

/-- **The rest of a block** (lines 82–106), once the new maximum and the exponents are known. -/
theorem row_tail {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve : Fin D → ℕ → ℝ} (hve : ∀ e j, |ve e j| ≤ 2 ^ 15)
    {m m' dm al l lij l' : F32} {x p : Fin 64 → F32} {q : Fin 64 → F16} {vt : Fin 64 → Fin D → F16}
    {acc a1 acc' : Fin D → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x) (hin : AlphaIn t m m' dm)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj)) (hal : P.math .exp2 .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij)
    (hq : ∀ jj, FloatV.NaNEq (FloatV.cvt (p jj)) (q jj))
    (ha1 : ∀ e, FloatV.NaNEq (FloatV.mul (acc e) al) (a1 e))
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj))
    (hdot : ∀ e, (a1 e).isFinite = true → (∀ k, (q k).isFinite = true) →
      |(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat| +
        (Rules.Num.relDot16 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot16 64) ≤
          Format.binary32.maxVal →
      (acc' e).isFinite = true ∧ |(acc' e).toRat - ((a1 e).toRat + ∑ k, (q k).toRat * (vt k e).toRat)| ≤
        Rules.Num.relDot16 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot16 64)
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hold : RowSt R ve t m l acc) : RowSt R ve (t + 1) m' l' acc' := by
  right
  refine ⟨by omega, hM', ?_, fun e => ?_⟩
  · refine lstep hexp hni hR ht hM' hX hp hin hal hsum hl' ?_
    rcases hold with ⟨h0, -, hlf, hlb, -⟩ | ⟨ht0, hmi, hL, -⟩
    · exact Or.inl ⟨h0, hlf, hlb⟩
    · exact Or.inr ⟨ht0, hmi, hL⟩
  · refine astep hexp hni hR ht (hve e) hM' hX hp hq hin hal (ha1 e) (fun jj => hvt jj e) (hdot e) ?_
    rcases hold with ⟨h0, -, -, -, ha⟩ | ⟨ht0, hmi, -, hA⟩
    · exact Or.inl ⟨h0, (ha e).1, (ha e).2⟩
    · exact Or.inr ⟨ht0, hmi, hA e⟩

/-- **One block, `STAGE` 1 or 3** (lines 72–73, 80–107): every key attended. -/
theorem row_step_plain {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve : Fin D → ℕ → ℝ} (hve : ∀ e j, |ve e j| ≤ 2 ^ 15)
    {cq m mx a m' dm al l lij l' : F32} {S Sc x p : Fin 64 → F32} {q : Fin 64 → F16}
    {vt : Fin 64 → Fin D → F16} {acc a1 acc' : Fin D → F32}
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
    (hq : ∀ jj, FloatV.NaNEq (FloatV.cvt (p jj)) (q jj))
    (ha1 : ∀ e, FloatV.NaNEq (FloatV.mul (acc e) al) (a1 e))
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj))
    (hdot : ∀ e, (a1 e).isFinite = true → (∀ k, (q k).isFinite = true) →
      |(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat| +
        (Rules.Num.relDot16 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot16 64) ≤
          Format.binary32.maxVal →
      (acc' e).isFinite = true ∧ |(acc' e).toRat - ((a1 e).toRat + ∑ k, (q k).toRat * (vt k e).toRat)| ≤
        Rules.Num.relDot16 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot16 64)
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hold : RowSt R ve t m l acc) : RowSt R ve (t + 1) m' l' acc' := by
  obtain ⟨hM', hX, hin⟩ := mstep_plain hR hcq hcqv hS hall (max64 hmx fun jj => (hS jj).1) ha hm' hSc hx hdm
    hold.old
  exact row_tail hexp hni hR ht hve hM' hX hin hp hal hsum hq ha1 hvt hdot hl' hold

/-- **One block, `STAGE` 2** (lines 72–78, 82–107): the causal mask. Before the first block, the
block's first key is attended. -/
theorem row_step_masked {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve : Fin D → ℕ → ℝ} (hve : ∀ e j, |ve e j| ≤ 2 ^ 15)
    {cq m mx m' dm al l lij l' : F32} {S w x' x p : Fin 64 → F32} {q : Fin 64 → F16}
    {vt : Fin 64 → Fin D → F16} {acc a1 acc' : Fin D → F32}
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
    (hq : ∀ jj, FloatV.NaNEq (FloatV.cvt (p jj)) (q jj))
    (ha1 : ∀ e, FloatV.NaNEq (FloatV.mul (acc e) al) (a1 e))
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj))
    (hdot : ∀ e, (a1 e).isFinite = true → (∀ k, (q k).isFinite = true) →
      |(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat| +
        (Rules.Num.relDot16 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot16 64) ≤
          Format.binary32.maxVal →
      (acc' e).isFinite = true ∧ |(acc' e).toRat - ((a1 e).toRat + ∑ k, (q k).toRat * (vt k e).toRat)| ≤
        Rules.Num.relDot16 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot16 64)
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
  obtain ⟨hM', hX, hin⟩ := mstep_masked hR hcq hcqv hS hglob hw hx' (max64 hmx hx'f) hm' hx hdm hold'
  exact row_tail hexp hni hR ht hve hM' hX hin hp hal hsum hq ha1 hvt hdot hl' hold

end Kernels.AttnFwd
