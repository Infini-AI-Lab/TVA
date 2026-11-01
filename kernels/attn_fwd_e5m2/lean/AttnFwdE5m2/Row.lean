import AttnFwdE5m2.Facts

/-!
# `attn_fwd_e5m2`: one query row of the instance

* `alOf`, `rowCtx`: the keys a row attends to, and its `RowCtx` at a bound `X ≤ Xh HEAD_DIM` on its
  scaled lifted score magnitudes and a bound `Z` on its scaled score errors; `rowCtx_ok`;
* `max64`: the row maximum of a block (`tl.max`, any tree);
* `RowSt`: a row's `m_i`, `l_i` and `acc` after `t` blocks: their initial values when `t = 0`,
  and `MInv`, `LInv`, `ACr`, `l_i ≥ lminB t`, the centred bound (`Cen.CInv`) and the second bound
  (`Mag`) afterwards;
* `mag_tail`, `cen_tail`: one block of the second bound and of the centred bound;
* `row_step_plain`, `row_step_masked`: one block of the loop (lines 72–109) keeps `RowSt`.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## The row's keys and context -/

/-- Key `j` is one query `n` attends to (`keys d n`). -/
def alOf (d : Inst) (n j : ℕ) : Bool := decide (j < d.N_CTX ∧ (d.causal = false ∨ j ≤ n))

theorem mem_keys {d : Inst} {n j : ℕ} : j ∈ keys d n ↔ alOf d n j = true := by
  unfold keys alOf; simp

/-- The library's constants of the score's `tl.dot` at `HEAD_DIM = 64, 128, 256` (its whole-dot
bound from the chunk facts, `Dot8Chunk.DotF8Chunk.liftAcc`), and the bound `expErr ≤ 13` at the
coverage cap `Xh`. -/
theorem dot8_consts {D : ℕ} (hD : D = 64 ∨ D = 128 ∨ D = 256) :
    0 ≤ Dot8Chunk.relLift8 D ∧ Dot8Chunk.relLift8 D ≤ 1 / 8 ∧ 0 ≤ Dot8Chunk.absLift8 D ∧
      Dot8Chunk.absLift8 D ≤ 1 / 2 ^ 122 ∧ Xh D ≤ 512 ∧
      Dot8Chunk.relLift8 D * (3 / 2 * Xh D) + 1 / 2 ^ 55 + 3 / 2 * Xh D / 2 ^ 19 + 1 / 2 ^ 50 ≤ 13 := by
  rcases hD with rfl | rfl | rfl <;>
    simp only [Dot8Chunk.relLift8, Dot8Chunk.absLift8, Xh, Rules.Num.chunks8, Rules.Num.rhoChunk8,
      Rules.Num.sigmaChunk8, Rules.Num.alphaChunk8] <;> norm_num

/-- Query `n` of head `bh` at the bounds `X` (on `sm_scale·scoreLift`) and `Z` (on
`sm_scale·scoreErr`): `qk_scale` `c`, the exact base-2 scale `sm_scale / ln 2`, the exact scores,
their lifted magnitudes and the bounds on their `tl.dot` errors (the smaller of `scoreErr` and the
whole-dot bound), `Y = 3X/2`, `Dl = scoreE` and `E = expErr`. -/
noncomputable def rowCtx (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) : RowCtx where
  c := c
  w := (sigma d : ℝ) / Real.log 2
  Y := 3 / 2 * (X : ℝ)
  E := (expErr d.HEAD_DIM X Z : ℝ)
  Dl := (scoreE d.HEAD_DIM X Z : ℝ)
  sv j := (score d init bh n j : ℝ)
  bv j := (scoreLift d init bh n j : ℝ)
  de j := ((min (scoreErr d init bh n j)
    (Dot8Chunk.relLift8 d.HEAD_DIM * scoreLift d init bh n j + Dot8Chunk.absLift8 d.HEAD_DIM) : ℚ) : ℝ)
  al := alOf d n

/-- The exact score as a sum over the lanes of `q`'s and `k`'s rows. -/
theorem score_eq (d : Inst) (init : Mem) (bh n j : ℕ) :
    score d init bh n j = ∑ e, qrow d init bh n e * krow d init bh j e := by
  unfold score qrow krow
  exact Finset.sum_range (fun e => val8 init d.q (idx d bh n e) * val8 init d.k (idx d bh j e))

theorem score_le' (d : Inst) (init : Mem) (bh n j : ℕ) : |score d init bh n j| ≤ scoreLift d init bh n j := by
  rw [score_eq]
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun e _ => Dot8Chunk.abs_mul_le_liftMag _ _ _ _)

theorem score_le (d : Inst) (init : Mem) (bh n j : ℕ) :
    |(score d init bh n j : ℝ)| ≤ (scoreLift d init bh n j : ℝ) := by
  exact_mod_cast score_le' d init bh n j

theorem scoreLift_nonneg (d : Inst) (init : Mem) (bh n j : ℕ) : 0 ≤ scoreLift d init bh n j :=
  Finset.sum_nonneg fun _ _ => mul_nonneg (Dot8Chunk.liftMag_nonneg _ _) (Dot8Chunk.liftMag_nonneg _ _)

theorem scoreErr_nonneg (d : Inst) (init : Mem) (bh n j : ℕ) : 0 ≤ scoreErr d init bh n j :=
  Dot8Chunk.errB_nonneg Dot8Chunk.rhoChunk8_nonneg Dot8Chunk.sigmaChunk8_nonneg Dot8Chunk.alphaChunk8_nonneg
    _ _ _ _

theorem sigma_pos {d : Inst} (hd : d.ok) : 0 < sigma d := by
  have := hd.scale.2.1; unfold sigma; linarith [show (0 : ℚ) < 1 / 2 ^ 64 by positivity]

theorem w_le {σ : ℝ} (hσ : 0 ≤ σ) : σ / Real.log 2 ≤ 3 / 2 * σ := by
  have h1 := Real.log_two_gt_d9
  have hl : 0 < Real.log 2 := by linarith
  rw [div_le_iff₀ hl]; nlinarith

/-- A key's `tl.dot` error bound times `qk_scale`: at most `scoreE`, when the key's scaled lifted
magnitude is at most `X` and its scaled score error at most `Z`. -/
theorem c_de_le {d : Inst} (hd : d.ok) (init : Mem) {c : ℝ} (hc0 : 0 < c) (hc65 : c ≤ 2 ^ 65)
    (hc32 : c ≤ 3 / 2 * (sigma d : ℝ)) {bh n j : ℕ} {X Z : ℚ}
    (hX : sigma d * scoreLift d init bh n j ≤ X) (hZ : sigma d * scoreErr d init bh n j ≤ Z) :
    c * ((min (scoreErr d init bh n j)
      (Dot8Chunk.relLift8 d.HEAD_DIM * scoreLift d init bh n j + Dot8Chunk.absLift8 d.HEAD_DIM) : ℚ) : ℝ) ≤
      (scoreE d.HEAD_DIM X Z : ℝ) := by
  obtain ⟨hr0, hr1, ha0, ha1, -, -⟩ := dot8_consts hd.head_dim
  have hσ : (0 : ℝ) ≤ (sigma d : ℝ) := by exact_mod_cast (sigma_pos hd).le
  have hL0 : (0 : ℝ) ≤ (scoreLift d init bh n j : ℝ) := by exact_mod_cast scoreLift_nonneg d init bh n j
  have hE0 : (0 : ℝ) ≤ (scoreErr d init bh n j : ℝ) := by exact_mod_cast scoreErr_nonneg d init bh n j
  have hX' : (sigma d : ℝ) * (scoreLift d init bh n j : ℝ) ≤ X := by exact_mod_cast hX
  have hZ' : (sigma d : ℝ) * (scoreErr d init bh n j : ℝ) ≤ Z := by exact_mod_cast hZ
  have hr0' : (0 : ℝ) ≤ (Dot8Chunk.relLift8 d.HEAD_DIM : ℝ) := by exact_mod_cast hr0
  have ha0' : (0 : ℝ) ≤ (Dot8Chunk.absLift8 d.HEAD_DIM : ℝ) := by exact_mod_cast ha0
  have ha1' : (Dot8Chunk.absLift8 d.HEAD_DIM : ℝ) ≤ 1 / 2 ^ 122 := by
    have := (Rat.cast_le (K := ℝ)).mpr ha1; push_cast at this; exact this
  unfold scoreE ybound
  push_cast
  rw [mul_min_of_nonneg _ _ hc0.le]
  apply min_le_min
  · have := mul_le_mul_of_nonneg_right hc32 hE0
    nlinarith
  · have h1 := mul_le_mul_of_nonneg_right hc32 hL0
    have h2 : c * (Dot8Chunk.absLift8 d.HEAD_DIM : ℝ) ≤ 2 ^ 65 * (1 / 2 ^ 122) :=
      mul_le_mul hc65 ha1' ha0' (by norm_num)
    have h3 : c * (scoreLift d init bh n j : ℝ) ≤ 3 / 2 * (X : ℝ) := by nlinarith
    have h4 := mul_le_mul_of_nonneg_left h3 hr0'
    have e : c * ((Dot8Chunk.relLift8 d.HEAD_DIM : ℝ) * (scoreLift d init bh n j : ℝ) +
        (Dot8Chunk.absLift8 d.HEAD_DIM : ℝ)) = (Dot8Chunk.relLift8 d.HEAD_DIM : ℝ) *
        (c * (scoreLift d init bh n j : ℝ)) + c * (Dot8Chunk.absLift8 d.HEAD_DIM : ℝ) := by ring
    rw [e]
    norm_num at h2 ⊢
    linarith

/-- **The row's context is `RowCtx.Ok`** for bounds `X ≤ Xh HEAD_DIM` and `Z` on its keys' scaled
lifted score magnitudes and scaled score errors. -/
theorem rowCtx_ok {d : Inst} (hd : d.ok) (init : Mem) {c : ℝ} (hc0 : 0 < c) (hc65 : c ≤ 2 ^ 65)
    (hc32 : c ≤ 3 / 2 * (sigma d : ℝ))
    (hcw : |c - (sigma d : ℝ) / Real.log 2| ≤ (sigma d : ℝ) / Real.log 2 / 2 ^ 22)
    (bh : ℕ) {n : ℕ} (hn : n < d.N_CTX) {X Z : ℚ} (hX : X ≤ Xh d.HEAD_DIM)
    (hXk : ∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ X)
    (hZk : ∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤ Z) :
    (rowCtx d init c bh n X Z).Ok ∧ 0 ≤ X := by
  have hσ := sigma_pos hd
  have hσ' : (0 : ℝ) < (sigma d : ℝ) := by exact_mod_cast hσ
  have h0 : (0 : ℕ) ∈ keys d n := mem_keys.mpr (by unfold alOf; simp; omega)
  have hX0 : 0 ≤ X := by
    have := hXk 0 h0
    have := mul_nonneg hσ.le (scoreLift_nonneg d init bh n 0)
    linarith
  have hZ0 : 0 ≤ Z := by
    have := hZk 0 h0
    have := mul_nonneg hσ.le (scoreErr_nonneg d init bh n 0)
    linarith
  obtain ⟨hr0, hr1, ha0, ha1, hXh, hEh⟩ := dot8_consts hd.head_dim
  have hX0' : (0 : ℝ) ≤ X := by exact_mod_cast hX0
  have hX512 : (X : ℝ) ≤ 512 := by exact_mod_cast hX.trans hXh
  have hl : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hsE : scoreE d.HEAD_DIM X Z ≤ Dot8Chunk.relLift8 d.HEAD_DIM * (3 / 2 * X) + 1 / 2 ^ 55 := by
    unfold scoreE ybound; exact min_le_right _ _
  have hsE0 : 0 ≤ scoreE d.HEAD_DIM X Z := by
    unfold scoreE ybound; apply le_min (by linarith); positivity
  have hE : expErr d.HEAD_DIM X Z ≤ 13 := by
    unfold expErr
    have h1 : Dot8Chunk.relLift8 d.HEAD_DIM * (3 / 2 * X) ≤ Dot8Chunk.relLift8 d.HEAD_DIM * (3 / 2 * Xh d.HEAD_DIM) :=
      mul_le_mul_of_nonneg_left (by linarith) hr0
    have h2 : ybound X / 2 ^ 19 ≤ 3 / 2 * Xh d.HEAD_DIM / 2 ^ 19 := by
      unfold ybound; exact div_le_div_of_nonneg_right (by linarith) (by norm_num)
    linarith
  have hDl : (scoreE d.HEAD_DIM X Z : ℝ) ≤ 3 / 2 * (X : ℝ) / 8 + 1 / 2 ^ 52 := by
    have h1 : scoreE d.HEAD_DIM X Z ≤ 3 / 2 * X / 8 + 1 / 2 ^ 55 := by
      have := mul_le_mul_of_nonneg_right hr1 (by positivity : (0 : ℚ) ≤ 3 / 2 * X)
      linarith
    have := (Rat.cast_le (K := ℝ)).mpr h1; push_cast at this
    have : (1 : ℝ) / 2 ^ 55 ≤ 1 / 2 ^ 52 := by norm_num
    linarith
  refine ⟨⟨hc0, hc65, div_pos hσ' hl, hcw, by show (0 : ℝ) ≤ 3 / 2 * (X : ℝ); positivity,
    by show 3 / 2 * (X : ℝ) ≤ 768; linarith,
    by show (0 : ℝ) ≤ (scoreE d.HEAD_DIM X Z : ℝ); exact_mod_cast hsE0,
    by show (scoreE d.HEAD_DIM X Z : ℝ) ≤ 3 / 2 * (X : ℝ) / 8 + 1 / 2 ^ 52; exact hDl, ?_,
    by show (expErr d.HEAD_DIM X Z : ℝ) ≤ 13; exact_mod_cast hE,
    fun j => score_le d init bh n j, fun j => ?_, ?_⟩, hX0⟩
  · show (expErr d.HEAD_DIM X Z : ℝ) = (scoreE d.HEAD_DIM X Z : ℝ) + 3 / 2 * (X : ℝ) / 2 ^ 19 + 1 / 2 ^ 50
    unfold expErr ybound; push_cast; ring
  · show (0 : ℝ) ≤ ((min (scoreErr d init bh n j)
      (Dot8Chunk.relLift8 d.HEAD_DIM * scoreLift d init bh n j + Dot8Chunk.absLift8 d.HEAD_DIM) : ℚ) : ℝ)
    have := scoreErr_nonneg d init bh n j
    have := scoreLift_nonneg d init bh n j
    have : (0 : ℚ) ≤ min (scoreErr d init bh n j)
        (Dot8Chunk.relLift8 d.HEAD_DIM * scoreLift d init bh n j + Dot8Chunk.absLift8 d.HEAD_DIM) :=
      le_min ‹_› (by positivity)
    exact_mod_cast this
  · intro j hj
    have hjk := mem_keys.mpr hj
    show c * (scoreLift d init bh n j : ℝ) ≤ 3 / 2 * (X : ℝ) ∧
      (sigma d : ℝ) / Real.log 2 * (scoreLift d init bh n j : ℝ) ≤ 3 / 2 * (X : ℝ) ∧
      c * ((min (scoreErr d init bh n j)
        (Dot8Chunk.relLift8 d.HEAD_DIM * scoreLift d init bh n j + Dot8Chunk.absLift8 d.HEAD_DIM) : ℚ) : ℝ) ≤
        (scoreE d.HEAD_DIM X Z : ℝ)
    have h1 := hXk j hjk
    have h1' : (sigma d : ℝ) * (scoreLift d init bh n j : ℝ) ≤ X := by exact_mod_cast h1
    have h2 : (0 : ℝ) ≤ (scoreLift d init bh n j : ℝ) := by exact_mod_cast scoreLift_nonneg d init bh n j
    have hw := w_le hσ'.le
    refine ⟨?_, ?_, c_de_le hd init hc0 hc65 hc32 h1 (hZk j hjk)⟩
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

/-! ## The state of a row -/

/-- A row's `m_i`, `l_i` and `acc` (an element per column `e`, whose `v` values are `ve e` and their
lifted magnitudes `vl e`) after `t` blocks: before the first, `-∞`, `1.0` and `0.0`; afterwards
`MInv`, `LInv`, `ACr`, `l_i ≥ lminB t`, the centred bound `CInv`, and the second bound:
`|acc − r·l_i| ≤ magA·l_i + magD` for every `r` and every bound `K` on `|v_j − r|` over the row's
keys. -/
def RowSt (R : RowCtx) {D : ℕ} (ve vl : Fin D → ℕ → ℝ) (t : ℕ) (m l : F32) (acc : Fin D → F32) : Prop :=
  (t = 0 ∧ m = FloatV.negInf ∧ l.isFinite = true ∧ l.toReal = 1 ∧
      ∀ e, (acc e).isFinite = true ∧ (acc e).toReal = 0) ∨
    (0 < t ∧ MInv R t m ∧ LInv R t m l ∧ (∀ e, ACr t (acc e)) ∧ lminB t ≤ l.toReal ∧
      (∀ e, CInv R (ve e) (vl e) t m (acc e) l) ∧
      ∀ e (r K : ℝ), 0 ≤ K → (∀ j, R.al j = true → |ve e j - r| ≤ K) →
        |(acc e).toReal - r * l.toReal| ≤ magA K |r| t * l.toReal + magD K |r| t)

theorem RowSt.old {R : RowCtx} {D : ℕ} {ve vl : Fin D → ℕ → ℝ} {t : ℕ} {m l : F32} {acc : Fin D → F32}
    (h : RowSt R ve vl t m l acc) : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m) := by
  rcases h with ⟨h0, hm, -⟩ | ⟨ht, hmi, -⟩
  · exact Or.inl ⟨h0, hm⟩
  · exact Or.inr ⟨ht, hmi⟩

/-- A lifted fp8 magnitude in real numbers: `liftMag x ≤ |x| + 2^-14`. -/
theorem liftMag_r (x : ℚ) : ((Rules.Num.liftMag .f8e5 x : ℚ) : ℝ) ≤ |(x : ℝ)| + 1 / 2 ^ 14 := by
  have h := Dot8Chunk.liftMag_le .f8e5 x
  have hmn : Rules.Num.minNormal8 .f8e5 = 1 / 2 ^ 14 := by simp [Rules.Num.minNormal8]
  rw [hmn] at h
  have := (Rat.cast_le (K := ℝ)).mpr h
  push_cast at this
  exact this

theorem liftMag_r_nonneg (x : ℚ) : (0 : ℝ) ≤ ((Rules.Num.liftMag .f8e5 x : ℚ) : ℝ) := by
  exact_mod_cast Dot8Chunk.liftMag_nonneg .f8e5 x

/-- **A nonzero `float8e5` value is at least `2^-16` in magnitude** (the smallest subnormal), by
evaluating the format's value of each of the `256` patterns. -/
theorem e5m2_abs_ge_aux : ∀ n < 256, Format.e5m2.toRat n ≠ 0 → (1 : ℚ) / 2 ^ 16 ≤ |Format.e5m2.toRat n| := by
  decide +kernel

theorem f8_abs_ge (x : F8E5) (h : x.toRat ≠ 0) : (1 : ℚ) / 2 ^ 16 ≤ |x.toRat| :=
  e5m2_abs_ge_aux x.pat (by unfold FloatV.pat; exact x.bits.isLt) h

/-- One lane's lifted product, for the second bound: at most
`|q·v| + iotaV K |r|·2^-14·(9/8·|p| + K + |r| + 2^-13)` (for a masked lane `q = 0`; for `v = 0` the
product is `0`, and otherwise `|v| ≥ 2^-16` makes `iotaV = 1`). -/
theorem lane_lift {q v : ℚ} {p r K : ℝ} {al : Bool} (hK : 0 ≤ K) (hv8 : v ≠ 0 → 1 / 2 ^ 16 ≤ |v|)
    (hatt : al = true → |(q : ℝ) - p| ≤ |p| / 8 + 1 / 2 ^ 17 ∧ |(v : ℝ) - r| ≤ K)
    (hmask : al = false → q = 0) :
    ((Rules.Num.liftMag .f8e5 q * Rules.Num.liftMag .f8e5 v : ℚ) : ℝ) ≤
      |(q : ℝ) * (v : ℝ)| + iotaV K |r| * (9 / 8 * |p| + (K + |r| + 1 / 2 ^ 13)) / 2 ^ 14 := by
  have hι0 := iotaV_nonneg hK (abs_nonneg r)
  have h0 : 0 ≤ iotaV K |r| * (9 / 8 * |p| + (K + |r| + 1 / 2 ^ 13)) / 2 ^ 14 := by positivity
  cases al
  · rw [hmask rfl, Dot8Chunk.liftMag_zero]; simp only [zero_mul, Rat.cast_zero]
    have := abs_nonneg ((0 : ℝ) * (v : ℝ)); linarith
  · obtain ⟨hq, hv⟩ := hatt rfl
    by_cases hv0 : v = 0
    · rw [hv0, Dot8Chunk.liftMag_zero]; simp only [mul_zero, Rat.cast_zero]
      have := abs_nonneg ((q : ℝ) * ((0 : ℚ) : ℝ)); linarith
    have h16 : (1 : ℝ) / 2 ^ 16 ≤ |(v : ℝ)| := by
      have := (Rat.cast_le (K := ℝ)).mpr (hv8 hv0); push_cast at this; exact this
    have hvb' : |(v : ℝ)| ≤ K + |r| := by have := abs_sub_abs_le_abs_sub (v : ℝ) r; linarith
    have hι1 : iotaV K |r| = 1 := min_eq_left (by nlinarith)
    rw [hι1, one_mul]
    push_cast
    have h1 := liftMag_r q
    have h2 := liftMag_r v
    have h3 := liftMag_r_nonneg q
    have h4 := liftMag_r_nonneg v
    have hqb : |(q : ℝ)| ≤ 9 / 8 * |p| + 1 / 2 ^ 17 := by
      have := abs_sub_abs_le_abs_sub (q : ℝ) p; linarith
    have hvb : |(v : ℝ)| ≤ K + |r| := by have := abs_sub_abs_le_abs_sub (v : ℝ) r; linarith
    have k1 := mul_le_mul h1 h2 h4 (by positivity)
    rw [abs_mul]
    have e : (|(q : ℝ)| + 1 / 2 ^ 14) * (|(v : ℝ)| + 1 / 2 ^ 14) =
        |(q : ℝ)| * |(v : ℝ)| + (|(q : ℝ)| + |(v : ℝ)| + 1 / 2 ^ 14) / 2 ^ 14 := by ring
    have k0 : (1 : ℝ) / 2 ^ 17 + 1 / 2 ^ 14 ≤ 1 / 2 ^ 13 := by norm_num
    have k2 : (|(q : ℝ)| + |(v : ℝ)| + 1 / 2 ^ 14) / 2 ^ 14 ≤ (9 / 8 * |p| + (K + |r| + 1 / 2 ^ 13)) / 2 ^ 14 :=
      div_le_div_of_nonneg_right (by linarith) (by norm_num)
    linarith

/-- The `p·v` `tl.dot`'s chunks' largest lifted products are at most their sum. -/
theorem cl2_le_sum (q v : Fin 64 → ℚ) :
    cl2 q v ≤ ∑ k, ((Rules.Num.liftMag .f8e5 (q k) * Rules.Num.liftMag .f8e5 (v k) : ℚ) : ℝ) := by
  have h : ∀ c, Dot8Chunk.chunkLift8 .f8e5 .f8e5 q v c ≤ Dot8Chunk.chunkLiftSum8 .f8e5 .f8e5 q v c := fun c => by
    unfold Dot8Chunk.chunkLift8 Dot8Chunk.chunkLiftSum8
    rw [Finset.fold_max_le]
    refine ⟨Finset.sum_nonneg fun k _ => mul_nonneg (Dot8Chunk.liftMag_nonneg _ _) (Dot8Chunk.liftMag_nonneg _ _),
      fun k hk => ?_⟩
    exact Finset.single_le_sum (f := fun k => Rules.Num.liftMag .f8e5 (q k) * Rules.Num.liftMag .f8e5 (v k))
      (fun k _ => mul_nonneg (Dot8Chunk.liftMag_nonneg _ _) (Dot8Chunk.liftMag_nonneg _ _)) hk
  have hs := Dot8Chunk.sum_chunkLiftSum8 .f8e5 .f8e5 q v
  have h2 : Rules.Num.chunks8 64 = 2 := rfl
  rw [h2, Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at hs
  have h3 : Dot8Chunk.chunkLift8 .f8e5 .f8e5 q v 0 + Dot8Chunk.chunkLift8 .f8e5 .f8e5 q v 1 ≤
      ∑ k, Rules.Num.liftMag .f8e5 (q k) * Rules.Num.liftMag .f8e5 (v k) := by
    linarith [h 0, h 1]
  have := (Rat.cast_le (K := ℝ)).mpr h3
  unfold cl2
  push_cast at this ⊢
  exact this

/-- **The second bound, one block** (lines 82–106): `l_i ≥ lminB (t + 1)`, and `|acc − r·l_i| ≤
magA·l_i + magD` for every `r` and every bound `K` on `|v_j − r|` over the row's keys. `hcomb` is
the dot's error (`astep`); `hnew`: the maximum is kept, or a lane's exponent is near `0`. -/
theorem mag_tail {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve vl : Fin D → ℕ → ℝ}
    {m m' dm al l lij l' : F32} {x p : Fin 64 → F32} {q : Fin 64 → F8E5} {vt : Fin 64 → Fin D → F8E5}
    {acc a1 acc' : Fin D → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x) (hin : AlphaIn t m m' dm)
    (hnew : (0 < t ∧ m' = m) ∨ ∃ jj : Fin 64, |(x jj).toReal| ≤ 1 / 2 ^ 11)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj)) (hal : P.math .exp2 .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij)
    (hq : ∀ jj, P.cvt Format.binary32 Format.e5m2 .rn (p jj) (q jj))
    (ha1 : ∀ e, |(a1 e).toReal - (acc e).toReal * al.toReal| ≤ |(acc e).toReal * al.toReal| / 2 ^ 24 + 1 / 2 ^ 150)
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj))
    (hcomb : ∀ e, |(acc' e).toReal - ((a1 e).toReal + ∑ jj, (q jj).toReal * (vt jj e).toReal)| ≤
      (cS8 : ℝ) * |(a1 e).toReal| + (cC8 : ℝ) * ∑ jj, |(q jj).toReal * (vt jj e).toReal| +
        (cP8 : ℝ) * cl2 (fun k => (q k).toRat) (fun k => (vt k e).toRat) + (cA8 : ℝ))
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hold : RowSt R ve vl t m l acc) :
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
    obtain ⟨-, -, -, hlijb⟩ := block_l hexp hR hX hp hsum
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
  -- the dot's error, in the form `estep_real` takes
  have hcomb' : |(acc' e).toReal - ((a1 e).toReal + ∑ jj, (q jj).toReal * (vt jj e).toReal)| ≤
      (cS8 : ℝ) * (|(a1 e).toReal| + ∑ jj, |(q jj).toReal * (vt jj e).toReal|) +
        (cP8 : ℝ) * iotaV K |r| * (9 / 8 * ∑ jj, |(p jj).toReal| + 64 * (K + |r| + 1 / 2 ^ 13)) / 2 ^ 14 +
          (cA8 : ℝ) := by
    have hlift : ∀ jj, ((Rules.Num.liftMag .f8e5 (q jj).toRat * Rules.Num.liftMag .f8e5 (vt jj e).toRat : ℚ) : ℝ) ≤
        |(q jj).toReal * (vt jj e).toReal| +
          iotaV K |r| * (9 / 8 * |(p jj).toReal| + (K + |r| + 1 / 2 ^ 13)) / 2 ^ 14 :=
      fun jj => lane_lift (p := (p jj).toReal) hK (f8_abs_ge (vt jj e))
        (fun h => ⟨(hpq jj).2.2.2.2.1 h, by rw [← toReal_def, (hvt jj e).2]; exact hKv _ h⟩)
        (fun h => by have := ((hpq jj).2.2.2.2.2 h).1; rw [toReal_def] at this; exact_mod_cast this)
    have hcl0 := (cl2_le_sum (fun k => (q k).toRat) (fun k => (vt k e).toRat)).trans
      (Finset.sum_le_sum fun jj _ => hlift jj)
    have hsum2 : ∑ jj, (|(q jj).toReal * (vt jj e).toReal| +
        iotaV K |r| * (9 / 8 * |(p jj).toReal| + (K + |r| + 1 / 2 ^ 13)) / 2 ^ 14) =
        ∑ jj, |(q jj).toReal * (vt jj e).toReal| +
          iotaV K |r| * (9 / 8 * ∑ jj, |(p jj).toReal| + 64 * (K + |r| + 1 / 2 ^ 13)) / 2 ^ 14 := by
      rw [Finset.sum_add_distrib, ← Finset.sum_div, ← Finset.mul_sum, Finset.sum_add_distrib, ← Finset.mul_sum]
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_ofNat]
    rw [hsum2] at hcl0
    have k1 := mul_le_mul_of_nonneg_left hcl0 (by rw [cP8_r]; norm_num : (0 : ℝ) ≤ (cP8 : ℝ))
    have := hcomb e
    simp only [cS8_r, cC8_r, cP8_r, cA8_r] at this k1 ⊢
    linarith only [this, k1]
  -- the old element
  have hacc : (acc e).isFinite = true ∧ |(acc e).toReal| ≤ 2 ^ 58 ∧
      |al.toReal| * |(acc e).toReal - r * l.toReal| ≤
        |al.toReal| * magA K |r| t * l.toReal + (1 + 1 / 2 ^ 12) * magD K |r| t := by
    rcases hold with ⟨h0, -, -, -, ha⟩ | ⟨ht0, -, -, hA, -, -, hE⟩
    · refine ⟨(ha e).1, by rw [(ha e).2]; norm_num, ?_⟩
      rw [ha0 h0, h0, show magD K |r| 0 = 0 from rfl]
      simp
    · refine ⟨(hA e).1, (hA e).2.trans (crA_le (by omega)), ?_⟩
      have h1 := mul_le_mul_of_nonneg_left (hE e r K hK hKv) (abs_nonneg al.toReal)
      have h2 : |al.toReal| * magD K |r| t ≤ (1 + 1 / 2 ^ 12) * magD K |r| t :=
        mul_le_mul_of_nonneg_right hα.2.1 (magD_nonneg hK (abs_nonneg r) t)
      have e1 : |al.toReal| * (magA K |r| t * l.toReal + magD K |r| t) =
          |al.toReal| * magA K |r| t * l.toReal + |al.toReal| * magD K |r| t := by ring
      linarith
  exact estep_real (q := fun jj => (q jj).toReal) (p := fun jj => (p jj).toReal)
    (v := fun jj => (vt jj e).toReal) hK (magA_nonneg hK (abs_nonneg r) t) hlb.2.1 hlb.2.2 hα.1 hp0
    (fun jj => (hlE jj).1) (fun jj => (hlE jj).2) hacc.2.2 (ha1 e) hcomb' hlije hl'e

/-! ## One block of the centred bound -/

/-- One attended lane's fp8 `p` against its scaled weight `w`: between `a8·w − eps8` and
`b8·w + eps8`. -/
theorem lane_q8 {q p w pe : ℝ} (hpe : 1 ≤ pe) (hw : 0 < w)
    (hp1 : |p - w| ≤ (pe * (1 + 1 / 2 ^ 22) - 1) * w + 1 / 2 ^ 126)
    (hp2 : w * (1 - 1 / 2 ^ 22) / pe - 1 / 2 ^ 126 ≤ p) (hqp : |q - p| ≤ |p| / 8 + 1 / 2 ^ 17) :
    a8 pe * w - (eps8 : ℝ) ≤ q ∧ q ≤ b8 pe * w + (eps8 : ℝ) := by
  have hpe0 : 0 < pe := by linarith
  have he : (eps8 : ℝ) = 1 / 2 ^ 17 + 1 / 2 ^ 125 := eps8_r
  have ha : a8 pe * w = 7 / 8 * (w * (1 - 1 / 2 ^ 22) / pe) := by unfold a8; field_simp
  have hb : b8 pe * w = 9 / 8 * (pe * (1 + 1 / 2 ^ 22) * w) := by unfold b8; ring
  have hu : 0 ≤ w * (1 - 1 / 2 ^ 22) / pe := by
    apply div_nonneg _ hpe0.le; nlinarith
  have hp3 : p ≤ pe * (1 + 1 / 2 ^ 22) * w + 1 / 2 ^ 126 := by
    have := le_abs_self (p - w); nlinarith
  rw [ha, hb, he]
  obtain ⟨hq1, hq2⟩ := abs_le.mp hqp
  rcases le_or_gt 0 p with h | h
  · rw [abs_of_nonneg h] at hq1 hq2
    constructor <;> nlinarith
  · rw [abs_of_neg h] at hq1 hq2
    constructor <;> nlinarith

/-- **The centred bound, one block** (lines 82–106): from the old row state, the block's lanes,
`alpha`, `acc * alpha` (`ha1`), the dot's error (`hcomb`) and the new `l_i` (`hl'`, at least
`lminB (t + 1)`), every element keeps `CInv` (`Cen.cstep_real`). -/
theorem cen_tail {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve vl : Fin D → ℕ → ℝ}
    {m m' dm al l lij l' : F32} {x p : Fin 64 → F32} {q : Fin 64 → F8E5} {vt : Fin 64 → Fin D → F8E5}
    {acc a1 acc' : Fin D → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x) (hin : AlphaIn t m m' dm)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj)) (hal : P.math .exp2 .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij)
    (hq : ∀ jj, P.cvt Format.binary32 Format.e5m2 .rn (p jj) (q jj))
    (ha1 : ∀ e, |(a1 e).toReal - (acc e).toReal * al.toReal| ≤ |(acc e).toReal * al.toReal| / 2 ^ 24 + 1 / 2 ^ 150)
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj) ∧
      ((Rules.Num.liftMag .f8e5 (vt jj e).toRat : ℚ) : ℝ) = vl e (64 * t + jj))
    (hcomb : ∀ e, |(acc' e).toReal - ((a1 e).toReal + ∑ jj, (q jj).toReal * (vt jj e).toReal)| ≤
      (cS8 : ℝ) * |(a1 e).toReal| + (cC8 : ℝ) * ∑ jj, |(q jj).toReal * (vt jj e).toReal| +
        (cP8 : ℝ) * cl2 (fun k => (q k).toRat) (fun k => (vt k e).toRat) + (cA8 : ℝ))
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hlm' : lminB (t + 1) ≤ l'.toReal)
    (hold : RowSt R ve vl t m l acc) :
    ∀ e, CInv R (ve e) (vl e) (t + 1) m' (acc' e) l' := by
  intro e r
  have hpe := pER_ge hR
  have hdm0 := dmaxR_nonneg hR
  have hdm1 := dmaxR_le hR
  set K' := (2 : ℝ) ^ (-m'.toReal) with hK'd
  have hK'0 : 0 < K' := two_rpow_pos _
  -- the lanes
  have hpq := lane_pq hexp hcvt hR hX hp hq
  have hpf : ∀ jj, (p jj).isFinite = true := fun jj => (hpq jj).1
  have hpb : ∀ jj, |(p jj).toReal| ≤ 2 := fun jj => by have := (hpq jj).2.1; linarith
  have hp0 : ∀ jj, -(1 / 2 ^ 125) ≤ (p jj).toReal := fun jj => (hpq jj).2.2.1
  have hw : ∀ jj : Fin 64, wt R m'.toReal (64 * t + jj) = K' * ew R (64 * t + jj) := fun jj => wt_eq _ _ _
  have hatt : ∀ jj : Fin 64, R.al (64 * t + jj) = true →
      a8 (pER R) * wt R m'.toReal (64 * t + jj) - (eps8 : ℝ) ≤ (q jj).toReal ∧
        (q jj).toReal ≤ b8 (pER R) * wt R m'.toReal (64 * t + jj) + (eps8 : ℝ) ∧
        |(q jj).toReal - (p jj).toReal| ≤ |(p jj).toReal| / 8 + 1 / 2 ^ 17 := fun jj ha => by
    obtain ⟨hxf, hxle, hxa, hxm⟩ := hX jj
    obtain ⟨-, -, hpa, -⟩ := lane_p hexp hR hxf hxle hxa hxm (hp jj)
    have hqp := (hpq jj).2.2.2.2.1 ha
    exact ⟨(lane_q8 hpe (wt_pos _ _ _) (hpa ha).1 (hpa ha).2 hqp).1,
      (lane_q8 hpe (wt_pos _ _ _) (hpa ha).1 (hpa ha).2 hqp).2, hqp⟩
  have hmask : ∀ jj : Fin 64, R.al (64 * t + jj) = false → (q jj).toReal = 0 ∧ |(p jj).toReal| ≤ 1 / 2 ^ 125 :=
    fun jj ha => (hpq jj).2.2.2.2.2 ha
  have he8 : (0 : ℝ) ≤ (eps8 : ℝ) := by rw [eps8_r]; norm_num
  have ha8 := (a8_pos hpe).le
  have hab : a8 (pER R) ≤ b8 (pER R) := by have := a8_le hpe; have := b8_ge hpe; linarith
  obtain ⟨hTb, hQAb⟩ := block_cen (q := fun jj => (q jj).toReal) (p := fun jj => (p jj).toReal)
    (v := fun jj => (vt jj e).toReal) (w := fun jj => wt R m'.toReal (64 * t + jj))
    (al := fun jj => R.al (64 * t + jj)) (r := r) he8 ha8 hab (fun jj => (wt_pos _ _ _).le) hatt hmask
  beta_reduce at hTb hQAb
  -- the chunks' largest lifted products
  have hlift : ∀ jj : Fin 64, ((Rules.Num.liftMag .f8e5 (q jj).toRat * Rules.Num.liftMag .f8e5 (vt jj e).toRat : ℚ) : ℝ) ≤
      if R.al (64 * t + jj) = true then
        (b8 (pER R) * wt R m'.toReal (64 * t + jj) + (eps8 : ℝ) + 1 / 2 ^ 14) * vl e (64 * t + jj) else 0 := by
    intro jj
    push_cast
    rw [(hvt jj e).2.2]
    have hvl0 : 0 ≤ vl e (64 * t + jj) := by rw [← (hvt jj e).2.2]; exact liftMag_r_nonneg _
    split_ifs with ha
    · obtain ⟨h1, h2, -⟩ := hatt jj ha
      have hq0 : |(q jj).toReal| ≤ b8 (pER R) * wt R m'.toReal (64 * t + jj) + (eps8 : ℝ) := by
        have := mul_nonneg ha8 (wt_pos R m'.toReal (64 * t + jj)).le
        have := mul_nonneg (ha8.trans hab) (wt_pos R m'.toReal (64 * t + jj)).le
        rw [abs_le]; constructor <;> linarith
      have hlq := liftMag_r (q jj).toRat
      rw [← toReal_def] at hlq
      exact mul_le_mul_of_nonneg_right (by linarith) hvl0
    · have ha' : R.al (64 * t + jj) = false := by simpa using ha
      have h0 := (hmask jj ha').1
      have : (q jj).toRat = 0 := by rw [toReal_def] at h0; exact_mod_cast h0
      rw [this, Dot8Chunk.liftMag_zero]; simp
  have hcl := cl2_cen (al := fun jj => R.al (64 * t + jj)) (w := fun jj => wt R m'.toReal (64 * t + jj))
    (vl := fun jj => vl e (64 * t + jj)) (b8_ge hpe |>.trans' (by norm_num)) he8
    (fun jj => (wt_pos _ _ _).le) (fun jj => by rw [← (hvt jj e).2.2]; exact liftMag_r_nonneg _) hlift
  beta_reduce at hcl
  have hBl : ∑ c ∈ Finset.range 2, (Rules.Num.chunk8 64 c).fold max 0
      (fun k : Fin 64 => if R.al (64 * t + k) = true then wt R m'.toReal (64 * t + k) * vl e (64 * t + k) else 0) =
      K' * bLW R (vl e) t := by
    have hf : (fun k : Fin 64 => if R.al (64 * t + k) = true then wt R m'.toReal (64 * t + k) * vl e (64 * t + k) else 0) =
        fun k : Fin 64 => K' * (if R.al (64 * t + k) = true then ew R (64 * t + k) * vl e (64 * t + k) else 0) := by
      funext k
      split_ifs
      · rw [hw k]; ring
      · simp
    unfold bLW
    rw [hf, Finset.mul_sum]
    exact Finset.sum_congr rfl fun c _ => fold_max_mul _ _ hK'0.le
  rw [hBl] at hcl
  -- the block's sums
  have hBc : ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then wt R m'.toReal (64 * t + jj) * |(vt jj e).toReal - r| else 0) =
      K' * bC R (ve e) r t := by
    unfold bC; rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun jj _ => ?_
    split_ifs
    · rw [hw jj, (hvt jj e).2.1]; ring
    · simp
  have hBa : ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then |(vt jj e).toReal - r| else 0) = bA R (ve e) r t := by
    unfold bA
    refine Finset.sum_congr rfl fun jj _ => ?_
    rw [(hvt jj e).2.1]
  have hCb : ∑ jj : Fin 64, (if R.al (64 * t + jj) = true then wt R m'.toReal (64 * t + jj) * ((vt jj e).toReal - r) else 0) =
      K' * (Cx R (ve e) r (t + 1) - Cx R (ve e) r t) := by
    unfold Cx; rw [psum_succ, add_sub_cancel_left, Finset.mul_sum]
    refine Finset.sum_congr rfl fun jj _ => ?_
    split_ifs
    · rw [hw jj, (hvt jj e).2.1]; ring
    · simp
  rw [hBc, hBa, hCb] at hTb
  rw [hBc, hBa] at hQAb
  -- `l_ij` (line 85)
  have hSA : ∑ jj, |(p jj).toReal| ≤ 2 ^ 100 := by
    have := lane_sum_le (a := fun jj => |(p jj).toReal|) (c := fun _ => (0 : ℝ)) (k := 0) (e := 2)
      fun jj => by simp only [zero_mul, zero_add]; exact hpb jj
    norm_num at this ⊢; linarith
  obtain ⟨hlijf, hlije⟩ := sum64_err hsum hpf hSA
  rw [show ((1 : ℝ) + 1 / 2 ^ 24) ^ 63 - 1 = gamr from rfl] at hlije
  have hlije' : |lij.toReal - ∑ jj, (p jj).toReal| ≤ gamr * ∑ jj, |(p jj).toReal| + 1 / 2 ^ 142 := by
    have : 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24 : ℝ) ^ 63 ≤ 1 / 2 ^ 142 := by norm_num
    linarith
  have hsp := sum_p_ge hp0
  have hsabs : |∑ jj, (p jj).toReal| ≤ ∑ jj, |(p jj).toReal| := Finset.abs_sum_le_sum_abs _ _
  -- `alpha` (line 84)
  have hmi : 0 < t → MInv R t m := fun ht0 => by
    rcases hold with ⟨h, -⟩ | ⟨-, hmi, -⟩
    · omega
    · exact hmi
  obtain ⟨haf, hab2, ha0, hae⟩ := alpha_facts hexp hni hR hM' hin hal hmi
  -- the old state, scaled to the new maximum
  obtain ⟨lam, hlam0, hlam1, hαl, hlb, hacc, hI, hLu⟩ : ∃ lam : ℝ, 0 ≤ lam ∧ lam ≤ 1 ∧
      |al.toReal - lam| ≤ etaA (dmaxR R) * lam + 1 / 2 ^ 126 ∧
      (l.isFinite = true ∧ 0 ≤ l.toReal ∧ l.toReal ≤ 2 ^ 44) ∧ |(acc e).toReal| ≤ 2 ^ 58 ∧
      |lam * (acc e).toReal - r * (lam * l.toReal) - kap (pER R) * (K' * Cx R (ve e) r t)| ≤
        K' * ecR R (ve e) (vl e) r t + acR R (ve e) (vl e) r t * (lam * l.toReal) ∧
      lam * l.toReal ≤ K' * (FhR R t * psum R.al t (ew R)) + FhR R t * (t * absLr) := by
    rcases hold with ⟨h0, -, hlf, hl1, ha⟩ | ⟨ht0, hmi', hL, hA, -, hC, -⟩
    · subst h0
      refine ⟨0, le_rfl, zero_le_one, ?_, ⟨hlf, by rw [hl1]; norm_num, by rw [hl1]; norm_num⟩,
        by rw [(ha e).2]; norm_num, ?_, ?_⟩
      · rw [ha0 rfl]; norm_num
      · simp [Cx, psum_zero, ecR, acR]
      · simp [psum_zero]
    · obtain ⟨hle, hae'⟩ := hae ht0
      have hl1 : (2 : ℝ) ^ (m.toReal - m'.toReal) ≤ 1 := two_rpow_le_one (by linarith)
      have hl0 := two_rpow_pos (m.toReal - m'.toReal)
      have hKK : (2 : ℝ) ^ (m.toReal - m'.toReal) * (2 : ℝ) ^ (-m.toReal) = K' := by
        rw [hK'd, ← two_rpow_add]; congr 1; ring
      refine ⟨(2 : ℝ) ^ (m.toReal - m'.toReal), hl0.le, hl1, hae',
        ⟨hL.1, hL.2.1.le, le_of_abs_le (hL.bound hR (by omega) hmi')⟩,
        (hA e).2.trans (crA_le (by omega)), ?_, ?_⟩
      · have h0 := mul_le_mul_of_nonneg_left (hC e r) hl0.le
        have h1 : |(2 : ℝ) ^ (m.toReal - m'.toReal) * ((acc e).toReal - r * l.toReal -
            kap (pER R) * ((2 : ℝ) ^ (-m.toReal) * Cx R (ve e) r t))| ≤ (2 : ℝ) ^ (m.toReal - m'.toReal) *
            ((2 : ℝ) ^ (-m.toReal) * ecR R (ve e) (vl e) r t + acR R (ve e) (vl e) r t * l.toReal) := by
          rw [abs_mul, abs_of_pos hl0]; exact h0
        have e1 : (2 : ℝ) ^ (m.toReal - m'.toReal) * ((acc e).toReal - r * l.toReal -
            kap (pER R) * ((2 : ℝ) ^ (-m.toReal) * Cx R (ve e) r t)) =
            (2 : ℝ) ^ (m.toReal - m'.toReal) * (acc e).toReal - r * ((2 : ℝ) ^ (m.toReal - m'.toReal) * l.toReal) -
              kap (pER R) * (K' * Cx R (ve e) r t) := by rw [← hKK]; ring
        have e2 : (2 : ℝ) ^ (m.toReal - m'.toReal) * ((2 : ℝ) ^ (-m.toReal) * ecR R (ve e) (vl e) r t +
            acR R (ve e) (vl e) r t * l.toReal) = K' * ecR R (ve e) (vl e) r t +
              acR R (ve e) (vl e) r t * ((2 : ℝ) ^ (m.toReal - m'.toReal) * l.toReal) := by rw [← hKK]; ring
        rw [e1, e2] at h1
        exact h1
      · obtain ⟨-, -, hlu, -⟩ := hL
        have hW := psum_wt_eq R t m.toReal
        rw [hW] at hlu
        have h1 := mul_le_mul_of_nonneg_left hlu hl0.le
        have hF0 : 0 ≤ FhR R t * (t * absLr) := by
          have := FhR_ge hR t; unfold absLr; positivity
        have h2 : (2 : ℝ) ^ (m.toReal - m'.toReal) * (FhR R t * (t * absLr)) ≤ FhR R t * (t * absLr) :=
          mul_le_of_le_one_left hF0 hl1
        have e1 : (2 : ℝ) ^ (m.toReal - m'.toReal) * (FhR R t * ((2 : ℝ) ^ (-m.toReal) * psum R.al t (ew R) +
            t * absLr)) = K' * (FhR R t * psum R.al t (ew R)) +
              (2 : ℝ) ^ (m.toReal - m'.toReal) * (FhR R t * (t * absLr)) := by rw [← hKK]; ring
        linarith
  -- `acc * alpha` (line 95)
  have hα' : |al.toReal - lam| ≤ ((1 + dmaxR R) * (1 + 1 / 2 ^ 22) - 1) * lam + 1 / 2 ^ 126 := hαl
  have hs := scale_err hlam0 hα' (ha1 e)
  have htau : |(acc e).toReal| * (1 + 1 / 2 ^ 24) / 2 ^ 126 + 1 / 2 ^ 150 ≤ 1 / 2 ^ 66 := by
    have k1 : |(acc e).toReal| * (1 + 1 / 2 ^ 24) ≤ 2 ^ 58 * 2 :=
      mul_le_mul hacc (by norm_num) (by norm_num) (by norm_num)
    have k2 : |(acc e).toReal| * (1 + 1 / 2 ^ 24) / 2 ^ 126 ≤ 2 ^ 58 * 2 / 2 ^ 126 :=
      div_le_div_of_nonneg_right k1 (by norm_num)
    norm_num at k2 ⊢; linarith
  -- line 106
  have hroomL : |l.toReal * al.toReal| + |lij.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul, abs_of_nonneg hlb.2.1]
    have := mul_le_mul hlb.2.2 hab2 (abs_nonneg _) (by norm_num)
    obtain ⟨-, -, -, hlijb⟩ := block_l hexp hR hX hp hsum
    linarith
  obtain ⟨-, hl'e⟩ := f32_mulAdd hl' hlb.1 haf hlijf hroomL
  have hLab : FhR R t * (t * absLr) ≤ 1 / 2 ^ 56 := by
    have hF := FhR_le hR (t := t) (by omega)
    have htr : (t : ℝ) ≤ 1024 := by exact_mod_cast (show t ≤ 1024 by omega)
    unfold absLr
    have := mul_le_mul hF (mul_le_mul_of_nonneg_right htr (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 80))
      (by positivity) (by norm_num)
    norm_num at this ⊢; linarith
  have hLu0 : 0 ≤ K' * (FhR R t * psum R.al t (ew R)) := by
    have := FhR_ge hR t
    have := psum_nonneg (al := R.al) (t := t) (f := ew R) fun j => (ew_pos R j).le
    positivity
  have hLab0 : 0 ≤ FhR R t * (t * absLr) := by have := FhR_ge hR t; unfold absLr; positivity
  have hstep := cstep_real hpe hdm0 hdm1 hlam0 hlb.2.1 hlb.2.2 hαl (by positivity) htau hs
    (acR_nonneg hR _ _ r t) (mul_nonneg hK'0.le (ecR_nonneg hR _ _ r t)) hI hLu0 hLab0 hLab hLu (hcomb e)
    (mul_nonneg hK'0.le (bC_nonneg R (ve e) r t)) (mul_nonneg hK'0.le (bLW_nonneg R (vl e) t))
    (bA_nonneg R (ve e) r t) (bVL_nonneg R (vl e) t) hQAb hcl hTb
    (Finset.sum_nonneg fun _ _ => abs_nonneg _) hsp hsabs hlije' hl'e
    ((lminB_half (by omega)).trans hlm')
  rw [errNext_scale _ _ _ _ _ _ _ _ _ hK'0.le] at hstep
  have e1 : K' * Cx R (ve e) r t + K' * (Cx R (ve e) r (t + 1) - Cx R (ve e) r t) = K' * Cx R (ve e) r (t + 1) := by
    ring
  rw [e1] at hstep
  exact hstep

/-- **The rest of a block** (lines 82–109), once the new maximum and the exponents are known. -/
theorem row_tail {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve vl : Fin D → ℕ → ℝ}
    {m m' dm al l lij l' : F32} {x p : Fin 64 → F32} {q : Fin 64 → F8E5} {vt : Fin 64 → Fin D → F8E5}
    {acc a1 acc' : Fin D → F32}
    (hM' : MInv R (t + 1) m') (hX : LaneX R t m' x) (hin : AlphaIn t m m' dm)
    (hnew : (0 < t ∧ m' = m) ∨ ∃ jj : Fin 64, |(x jj).toReal| ≤ 1 / 2 ^ 11)
    (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj)) (hal : P.math .exp2 .binary32 dm al)
    (hsum : ∃ tr : Tree, tr.Covers 64 ∧ tr.Eval (AddRel false) (laneSeq p) lij)
    (hq : ∀ jj, P.cvt Format.binary32 Format.e5m2 .rn (p jj) (q jj))
    (ha1 : ∀ e, FloatV.NaNEq (FloatV.mul (acc e) al) (a1 e))
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj) ∧
      ((Rules.Num.liftMag .f8e5 (vt jj e).toRat : ℚ) : ℝ) = vl e (64 * t + jj))
    (hdot : ∀ e, (a1 e).isFinite = true → (∀ k, (q k).isFinite = true) →
      |(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat| +
        (Rules.Num.relDot8 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot8 64) ≤
          Format.binary32.maxVal →
      (acc' e).isFinite = true ∧ |(acc' e).toReal - ((a1 e).toReal + ∑ k, (q k).toReal * (vt k e).toReal)| ≤
        (cS8 : ℝ) * |(a1 e).toReal| + (cC8 : ℝ) * ∑ k, |(q k).toReal * (vt k e).toReal| +
          (cP8 : ℝ) * cl2 (fun k => (q k).toRat) (fun k => (vt k e).toRat) + (cA8 : ℝ))
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hold : RowSt R ve vl t m l acc) : RowSt R ve vl (t + 1) m' l' acc' := by
  have hA : ∀ e, (a1 e).isFinite = true ∧
      |(a1 e).toReal - (acc e).toReal * al.toReal| ≤ |(acc e).toReal * al.toReal| / 2 ^ 24 + 1 / 2 ^ 150 ∧
      ACr (t + 1) (acc' e) ∧
      |(acc' e).toReal - ((a1 e).toReal + ∑ jj, (q jj).toReal * (vt jj e).toReal)| ≤
        (cS8 : ℝ) * |(a1 e).toReal| + (cC8 : ℝ) * ∑ jj, |(q jj).toReal * (vt jj e).toReal| +
          (cP8 : ℝ) * cl2 (fun k => (q k).toRat) (fun k => (vt k e).toRat) + (cA8 : ℝ) := fun e => by
    refine astep hexp hni hcvt hR ht hM' hX hp hq hin hal (ha1 e) (fun jj => (hvt jj e).1) (hdot e) ?_
    rcases hold with ⟨h0, -, -, -, ha⟩ | ⟨ht0, hmi, -, hA, -⟩
    · exact Or.inl ⟨h0, (ha e).1, (ha e).2⟩
    · exact Or.inr ⟨ht0, hmi, hA e⟩
  obtain ⟨hlm, hE⟩ := mag_tail hexp hni hcvt hR ht hM' hX hin hnew hp hal hsum hq (fun e => (hA e).2.1)
    (fun jj e => ⟨(hvt jj e).1, (hvt jj e).2.1⟩) (fun e => (hA e).2.2.2) hl' hold
  have hC := cen_tail hexp hni hcvt hR ht hM' hX hin hp hal hsum hq (fun e => (hA e).2.1) hvt
    (fun e => (hA e).2.2.2) hl' hlm hold
  right
  refine ⟨by omega, hM', ?_, fun e => (hA e).2.2.1, hlm, hC, hE⟩
  refine lstep hexp hni hR ht hM' hX hp hin hal hsum hl' ?_
  rcases hold with ⟨h0, -, hlf, hlb, -⟩ | ⟨ht0, hmi, hL, -⟩
  · exact Or.inl ⟨h0, hlf, hlb⟩
  · exact Or.inr ⟨ht0, hmi, hL⟩

/-- **One block, `STAGE` 1 or 3** (lines 72–73, 80–109): every key attended. -/
theorem row_step_plain {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve vl : Fin D → ℕ → ℝ}
    {cq m mx a m' dm al l lij l' : F32} {S Sc x p : Fin 64 → F32} {q : Fin 64 → F8E5}
    {vt : Fin 64 → Fin D → F8E5} {acc a1 acc' : Fin D → F32}
    (hcq : cq.isFinite = true) (hcqv : cq.toReal = R.c)
    (hS : ∀ jj : Fin 64, (S jj).isFinite = true ∧ |(S jj).toReal - R.sv (64 * t + jj)| ≤ R.de (64 * t + jj))
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
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj) ∧
      ((Rules.Num.liftMag .f8e5 (vt jj e).toRat : ℚ) : ℝ) = vl e (64 * t + jj))
    (hdot : ∀ e, (a1 e).isFinite = true → (∀ k, (q k).isFinite = true) →
      |(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat| +
        (Rules.Num.relDot8 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot8 64) ≤
          Format.binary32.maxVal →
      (acc' e).isFinite = true ∧ |(acc' e).toReal - ((a1 e).toReal + ∑ k, (q k).toReal * (vt k e).toReal)| ≤
        (cS8 : ℝ) * |(a1 e).toReal| + (cC8 : ℝ) * ∑ k, |(q k).toReal * (vt k e).toReal| +
          (cP8 : ℝ) * cl2 (fun k => (q k).toRat) (fun k => (vt k e).toRat) + (cA8 : ℝ))
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (hold : RowSt R ve vl t m l acc) : RowSt R ve vl (t + 1) m' l' acc' := by
  obtain ⟨hM', hX, hin, hnew⟩ := mstep_plain hR hcq hcqv hS hall (max64 hmx fun jj => (hS jj).1) ha hm' hSc hx
    hdm hold.old
  exact row_tail hexp hni hcvt hR ht hM' hX hin hnew hp hal hsum hq ha1 hvt hdot hl' hold

/-- **One block, `STAGE` 2** (lines 72–78, 82–109): the causal mask. Before the first block, the
block's first key is attended. -/
theorem row_step_masked {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hni : Rules.Num.Exp2F32NegInf P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} (ht : t + 1 ≤ 1024) {fusion : Bool} {D : ℕ}
    {ve vl : Fin D → ℕ → ℝ}
    {cq m mx m' dm al l lij l' : F32} {S w x' x p : Fin 64 → F32} {q : Fin 64 → F8E5}
    {vt : Fin 64 → Fin D → F8E5} {acc a1 acc' : Fin D → F32}
    (hcq : cq.isFinite = true) (hcqv : cq.toReal = R.c)
    (hS : ∀ jj : Fin 64, (S jj).isFinite = true ∧ |(S jj).toReal - R.sv (64 * t + jj)| ≤ R.de (64 * t + jj))
    (hglob : ∀ jj : Fin 64, R.c * R.bv (64 * t + jj) ≤ 1536 ∧ R.c * R.de (64 * t + jj) ≤ 192)
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
    (hvt : ∀ jj e, (vt jj e).isFinite = true ∧ (vt jj e).toReal = ve e (64 * t + jj) ∧
      ((Rules.Num.liftMag .f8e5 (vt jj e).toRat : ℚ) : ℝ) = vl e (64 * t + jj))
    (hdot : ∀ e, (a1 e).isFinite = true → (∀ k, (q k).isFinite = true) →
      |(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat| +
        (Rules.Num.relDot8 64 * (|(a1 e).toRat| + ∑ k, |(q k).toRat * (vt k e).toRat|) + Rules.Num.absDot8 64) ≤
          Format.binary32.maxVal →
      (acc' e).isFinite = true ∧ |(acc' e).toReal - ((a1 e).toReal + ∑ k, (q k).toReal * (vt k e).toReal)| ≤
        (cS8 : ℝ) * |(a1 e).toReal| + (cC8 : ℝ) * ∑ k, |(q k).toReal * (vt k e).toReal| +
          (cP8 : ℝ) * cl2 (fun k => (q k).toRat) (fun k => (vt k e).toRat) + (cA8 : ℝ))
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lij) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lij) l'))
    (h0 : t = 0 → R.al (64 * t) = true)
    (hold : RowSt R ve vl t m l acc) : RowSt R ve vl (t + 1) m' l' acc' := by
  have hx'f : ∀ jj, (x' jj).isFinite = true := fun jj => by
    have hc := hR.c_pos
    obtain ⟨hSf, hSe⟩ := hS jj
    have hroom : |(S jj).toReal * cq.toReal| + |(w jj).toReal| ≤ 2 ^ 120 := by
      rw [abs_mul, hcqv, abs_of_pos hc]
      obtain ⟨-, hg⟩ := cS_bound hR hSe (hglob jj).1 (hglob jj).2
      rw [abs_mul, abs_of_pos hc] at hg
      have h9 : |(w jj).toReal| ≤ 1000000 := by
        obtain ⟨-, hw1, hw2⟩ := hw jj
        cases hal : R.al (64 * t + jj)
        · rw [hw2 hal]; norm_num
        · rw [hw1 hal]; norm_num
      have : |(S jj).toReal| * R.c = R.c * |(S jj).toReal| := by ring
      linarith
    exact (f32_mulAdd (hx' jj) hSf hcq (hw jj).1 hroom).1
  have hold' : (t = 0 ∧ m = FloatV.negInf ∧ R.al (64 * t) = true) ∨ (0 < t ∧ MInv R t m) := by
    rcases hold.old with ⟨ht0, hm⟩ | h
    · exact Or.inl ⟨ht0, hm, h0 ht0⟩
    · exact Or.inr h
  obtain ⟨hM', hX, hin, hnew⟩ := mstep_masked hR hcq hcqv hS hglob hw hx' (max64 hmx hx'f) hm' hx hdm hold'
  exact row_tail hexp hni hcvt hR ht hM' hX hin hnew hp hal hsum hq ha1 hvt hdot hl' hold

end Kernels.AttnFwdE5m2
