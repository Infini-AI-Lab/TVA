import PagedAttention.Final

/-!
# `paged_attention_f16`: facts for the program walk

* integer arithmetic in `int32` and `int64` without wrap-around;
* the constants: `other=0.0` converted to `float16`, `-inf` and `0.0` in `float32`, `scale`;
* `rowCtx`, `veOf`, `RowSt`, `StInv`: one program's row data and state over the loop;
* `row_step`: one page of the loop (lines 61–68) keeps `RowSt`;
* `row_final`: the end of the row (lines 71–72).
-/

namespace Kernels.PagedAttention

open TVA TVA.Math.Float

/-! ## Integers -/

theorem i32_add {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x + y) (h1 : x + y < 2 ^ 31) : (a + b).toInt = x + y := by
  rw [IntV.toInt_add (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_mul {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x * y) (h1 : x * y < 2 ^ 31) : (a * b).toInt = x * y := by
  rw [IntV.toInt_mul (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_nat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n := IntV.I32.toInt_ofNat h

theorem i32_zero : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)

theorem i32_one : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)

theorem i32_lt {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    IntV.lt a b = decide (x < y) := by
  rw [IntV.lt_eq, ha, hb]

theorem i64_add {a b : I64} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : 0 ≤ x + y) (h1 : x + y < 2 ^ 63) : (a + b).toInt = x + y := by
  rw [IntV.toInt_add (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i64_mul {a b : I64} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : 0 ≤ x * y) (h1 : x * y < 2 ^ 63) : (a * b).toInt = x * y := by
  rw [IntV.toInt_mul (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

/-- `x.to(tl.int64)` of an `int32`: the same value. -/
theorem conv64 {P : Platform} {x : I32} {r : I64} (h : ConvRel P .i32 .i64 x r) : r.toInt = x.toInt := by
  change r = IntV.ofInt x.toInt at h
  rw [h]
  have := IntV.sfits_toInt x
  exact IntV.toInt_ofInt (by norm_num) (by
    unfold IntV.SFits at this ⊢; constructor <;> norm_num at this ⊢ <;> omega)

theorem conv64_ex (P : Platform) (x : I32) : ∃ r, ConvRel P .i32 .i64 x r := ⟨IntV.ofInt x.toInt, rfl⟩

/-! ## The constants -/

theorem f32_zero_lit : (FloatV.litF32 (FloatV.ofPat 0) : F32) = FloatV.ofPat 0 := by
  unfold FloatV.litF32
  rw [ite_eq_left]
  rw [FloatV.isZero_ofPat (by norm_num)]
  decide

theorem f16_cvt_zero : (FloatV.cvt (FloatV.ofPat 0 : F32) : F16).isFinite = true ∧
    (FloatV.cvt (FloatV.ofPat 0 : F32) : F16).toRat = 0 := by
  have h0 : (FloatV.ofPat 0 : F32).pat = 0 := FloatV.pat_ofPat (by norm_num)
  have hfin : Math.Float.Format.binary32.isFinite 0 = true := by decide
  have hsign : Math.Float.Format.binary32.sign 0 = false := by decide
  have htr : Math.Float.Format.binary32.toRat 0 = 0 :=
    Math.Float.Format.toRat_of_isZero (by decide)
  unfold FloatV.cvt
  rw [h0, Math.Float.Format.cvt_of_finite _ _ _ hfin, hsign, htr]
  obtain ⟨hlt, hf, hr⟩ := Math.Float.Format.roundRat_zero_spec (f := Math.Float.Format.binary16)
    (rnd := .rn) (sat := false) (zs := false) Math.Float.Format.binary16_regular
  rw [FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨hf, hr⟩

/-- The source's `other=0.0` converted to `float16` (lines 60, 67): some result exists. -/
theorem zero_conv_ex (P : Platform) :
    ∃ r, ConvRel P .f32 .f16 (FloatV.litF32 (FloatV.ofPat 0) : F32) r :=
  ⟨FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0) : F32), by
    show FloatRel P Math.Float.Format.binary32 Math.Float.Format.binary16 _ _
    unfold FloatRel
    rw [ite_eq_right (by decide)]
    exact Or.inl ⟨by unfold Math.Float.Format.Std; simp, by unfold Math.Float.Format.Std; simp,
      FloatV.NaNEq.refl _⟩⟩

/-- Every result of the conversion is a finite zero. -/
theorem zero_conv {P : Platform} {r : F16}
    (h : ConvRel P .f32 .f16 (FloatV.litF32 (FloatV.ofPat 0) : F32) r) :
    r.isFinite = true ∧ r.toRat = 0 := by
  change FloatRel P Math.Float.Format.binary32 Math.Float.Format.binary16 _ r at h
  unfold FloatRel at h
  rw [ite_eq_right (by decide), f32_zero_lit] at h
  obtain ⟨hf, hr⟩ := f16_cvt_zero
  rcases h with ⟨-, -, h⟩ | ⟨-, hF8, -⟩ | ⟨hF8, -, -⟩
  · have hn : (FloatV.cvt (FloatV.ofPat 0 : F32) : F16).isNaN = false :=
      (FloatV.isFinite_iff.mp hf).1
    rw [h.eq hn]
    exact ⟨hf, hr⟩
  · exact absurd hF8 (by unfold Math.Float.Format.F8; decide)
  · exact absurd hF8 (by unfold Math.Float.Format.F8; decide)

/-- Line 51's and line 62's `-inf` in `float32`. -/
theorem ninf_lit : (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000 : F64) : F32) = FloatV.negInf := by
  decide +kernel

/-- The widening of a finite `float16` to `float32` is finite with the same value. -/
theorem widen {x : F16} {r : F32} (hx : x.isFinite = true) (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r) :
    r.isFinite = true ∧ r.toRat = x.toRat := by
  obtain ⟨hf, hv⟩ := FloatV.toRat_cvt_of_wider Format.binary16_regular Format.binary32_regular
    Format.wider_binary16_binary32 .rn hx
  rw [h.eq (FloatV.isFinite_iff.mp hf).1]
  exact ⟨hf, hv⟩

theorem f16_f32_ok : Format.binary16.Std ∧ Format.binary32.Std ∧ tl.RoundOk Format.binary16 Format.binary32 .rn :=
  ⟨by unfold Format.Std; simp, by unfold Format.Std; simp, Or.inl rfl⟩

theorem f32_f16_ok : Format.binary32.Std ∧ Format.binary16.Std ∧ tl.RoundOk Format.binary32 Format.binary16 .rn :=
  ⟨by unfold Format.Std; simp, by unfold Format.Std; simp, Or.inl rfl⟩

theorem f16_abs_le {x : F16} (hx : x.isFinite = true) : |x.toReal| ≤ 65504 := by
  have h := FloatV.abs_toRat_le_maxVal Format.binary16_regular hx
  have : ((|x.toRat| : ℚ) : ℝ) ≤ (Format.binary16.maxVal : ℝ) := by exact_mod_cast h
  rw [maxVal16] at this; push_cast at this; rw [← toReal_def] at this; exact this

/-- **`scale` in `float32`** (`FloatV.hostF32` of the double): finite, within one rounding of
`scale`. -/
theorem scale_facts {d : Inst} (hd : d.ok) :
    (FloatV.hostF32 d.scale).isFinite = true ∧
      |(FloatV.hostF32 d.scale).toReal - (sigma d : ℝ)| ≤ (sigma d : ℝ) / 2 ^ 24 + 1 / 2 ^ 150 := by
  obtain ⟨hsf, hs1, hs2⟩ := hd.scale
  have hσd : ((sigma d : ℚ) : ℝ) = d.scale.toReal := rfl
  rw [hσd]
  have hσ1 : 1 / 2 ^ 64 ≤ d.scale.toReal := by
    have := (Rat.cast_le (K := ℝ)).mpr hs1; push_cast at this; exact this
  have hσ2 : d.scale.toReal ≤ 2 ^ 64 := by
    have := (Rat.cast_le (K := ℝ)).mpr hs2; push_cast at this; exact this
  have hσ0 : 0 < d.scale.toReal := lt_of_lt_of_le (by positivity) hσ1
  have hnan : d.scale.isNaN = false := isNaN_of_fin hsf
  have hh : FloatV.hostF32 d.scale = FloatV.cvt d.scale := by
    unfold FloatV.hostF32; simp [hnan]
  have hmax : |d.scale.toRat| ≤ Format.binary32.maxVal := by
    have : ((|d.scale.toRat| : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      rw [maxVal32]; push_cast; rw [← toReal_def, abs_of_pos hσ0]; nlinarith
    exact_mod_cast this
  obtain ⟨hs32f, hs32e⟩ := FloatV.cvt_err (g := Format.binary32) Format.binary32_regular hsf hmax
  rw [← hh] at hs32f hs32e
  refine ⟨hs32f, ?_⟩
  rw [u32_eq, eta32_eq] at hs32e
  have := (Rat.cast_le (K := ℝ)).mpr hs32e
  push_cast at this
  rw [← toReal_def, ← toReal_def, abs_of_pos hσ0] at this
  norm_num at this ⊢; linarith

/-! ## The row's data and state -/

/-- Program `(s, h)`'s row data for a bound `X` on its scaled score magnitudes. -/
noncomputable def rowCtx (d : Inst) (init : Mem) (s h : ℕ) (X : ℚ) : RowCtx where
  c := (sigma d : ℝ)
  Y := (X : ℝ)
  E := (expErr d.HEAD_DIM X : ℝ)
  rhoD := (rhoD d.HEAD_DIM : ℝ)
  sv := fun j => (score d init s h j : ℝ)
  bv := fun j => (scoreAbs d init s h j : ℝ)
  n := seqLen d init s
  P := d.PAGE_SIZE

/-- The values of column `e` of `v` the row sums: token `j`'s (`0` past the sequence). -/
noncomputable def veOf (d : Inst) (init : Mem) (s h : ℕ) (e : ℕ) : ℕ → ℝ :=
  fun j => if j < seqLen d init s then (val16 init d.v_cache (tokElem d init s h j e) : ℝ) else 0

/-- The row's state after `t` pages: the initial values, or the invariants. -/
def RowSt (R : RowCtx) {D : ℕ} (ve : Fin D → ℕ → ℝ) (t : ℕ) (m l : F32) (acc : Fin D → F32) : Prop :=
  (t = 0 ∧ m = FloatV.negInf ∧ l.isFinite = true ∧ l.toReal = 0 ∧
      ∀ e, (acc e).isFinite = true ∧ (acc e).toReal = 0) ∨
    (0 < t ∧ MInv R t m ∧ LInv R t m l ∧ ∀ e, AInv R (ve e) t m (acc e))

/-- Program `(s, h)` after `t` pages: for covered values and every bound `X ≤ 2^10` on the
sequence's scaled score magnitudes, `RowSt` of its row data. -/
def StInv (d : Inst) (init : Mem) (s h t : ℕ) (acc : Tensor [d.HEAD_DIM] F32) (l m : F32) : Prop :=
  SeqVals d init s h → ∀ X : ℚ, X ≤ 2 ^ 10 →
    (∀ j, j < seqLen d init s → sigma d * scoreAbs d init s h j ≤ X) →
    RowSt (rowCtx d init s h X) (fun e => veOf d init s h e) t m l (fun e => acc.get1 e)

theorem headDim_le {d : Inst} (hd : d.ok) : d.HEAD_DIM ≤ 256 := by
  obtain ⟨a, ha, he⟩ := hd.head_dim
  rw [he]; calc 2 ^ a ≤ 2 ^ 8 := Nat.pow_le_pow_right (by norm_num) ha
    _ = 256 := by norm_num

theorem pageSize_le {d : Inst} (hd : d.ok) : d.PAGE_SIZE ≤ 256 := by
  obtain ⟨a, -, ha, he⟩ := hd.page_size
  rw [he]; calc 2 ^ a ≤ 2 ^ 8 := Nat.pow_le_pow_right (by norm_num) ha
    _ = 256 := by norm_num

theorem pageSize_pos {d : Inst} (hd : d.ok) : 0 < d.PAGE_SIZE := by
  obtain ⟨a, -, -, he⟩ := hd.page_size
  rw [he]; positivity

theorem headDim_pos {d : Inst} (hd : d.ok) : 0 < d.HEAD_DIM := by
  obtain ⟨a, -, he⟩ := hd.head_dim
  rw [he]; positivity

theorem sigma_bounds {d : Inst} (hd : d.ok) : 1 / 2 ^ 64 ≤ (sigma d : ℝ) ∧ (sigma d : ℝ) ≤ 2 ^ 64 := by
  obtain ⟨-, hs1, hs2⟩ := hd.scale
  constructor
  · have := (Rat.cast_le (K := ℝ)).mpr hs1; push_cast at this; exact this
  · have := (Rat.cast_le (K := ℝ)).mpr hs2; push_cast at this; exact this

theorem score_le (d : Inst) (init : Mem) (s h j : ℕ) :
    |(score d init s h j : ℝ)| ≤ (scoreAbs d init s h j : ℝ) := by
  have : |score d init s h j| ≤ scoreAbs d init s h j := Finset.abs_sum_le_sum_abs _ _
  exact_mod_cast this

theorem scoreAbs_nonneg (d : Inst) (init : Mem) (s h j : ℕ) : 0 ≤ scoreAbs d init s h j :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- The row data is as the analysis needs, for a sequence with a token. -/
theorem rowCtx_ok {d : Inst} (hd : d.ok) (init : Mem) {s h : ℕ} {X : ℚ} (hn : 1 ≤ seqLen d init s)
    (hX : X ≤ 2 ^ 10) (hXk : ∀ j, j < seqLen d init s → sigma d * scoreAbs d init s h j ≤ X) :
    (rowCtx d init s h X).Ok ∧ 0 ≤ X := by
  have hσ := sigma_bounds hd
  have hσ0 : 0 ≤ sigma d := by
    have : (0 : ℝ) ≤ (sigma d : ℝ) := le_trans (by norm_num) hσ.1
    exact_mod_cast this
  have hX0 : 0 ≤ X := le_trans (mul_nonneg hσ0 (scoreAbs_nonneg d init s h 0)) (hXk 0 (by omega))
  obtain ⟨hr0, hr1⟩ := rhoD_bounds (headDim_le hd)
  refine ⟨⟨hσ.1, hσ.2, (by show (0 : ℝ) ≤ (X : ℝ); exact_mod_cast hX0),
    (by show (X : ℝ) ≤ 2 ^ 10; exact_mod_cast hX), hr0, hr1, ?_, fun j => score_le d init s h j,
    fun j hj => ?_, pageSize_pos hd, pageSize_le hd⟩, hX0⟩
  · show (expErr d.HEAD_DIM X : ℝ) = ((rhoD d.HEAD_DIM : ℝ) + 8 / 2 ^ 24) * (X : ℝ) + 1 / 2 ^ 50
    exact expErr_cast _ _
  · show (sigma d : ℝ) * (scoreAbs d init s h j : ℝ) ≤ (X : ℝ)
    exact_mod_cast hXk j hj

theorem veOf_le {d : Inst} {init : Mem} {s h : ℕ} (hH : SeqVals d init s h) (e : ℕ) (he : e < d.HEAD_DIM)
    (j : ℕ) : |veOf d init s h e j| ≤ 2 ^ 15 := by
  unfold veOf
  split_ifs with hj
  · have := (hH.2.2.1 j e hj he).2.2; exact_mod_cast this
  · simp

/-! ## One page of the loop -/

/-- **One page** (lines 62–68) from the lanes' scores: `sc` the lanes of line 62 (`RowMax.mstep`'s
form), `mx`, `m'`, `dm`, `al`, `x`, `p`, `lsum`, `l'` lines 63–66, `vt` the page's `v` (a token's
value, `0` on a masked lane), `pv`, `vsum`, `acc'` line 68. -/
theorem row_step {P : Platform} (hexp : Rules.Num.ExpF32 P) (hni : Rules.Num.ExpF32NegInf P)
    {R : RowCtx} (hR : R.Ok) {D : ℕ} {t : ℕ} (ht : t + 1 ≤ 2 ^ 16) (h0 : R.P * t < R.n)
    {fusion : Bool} {ve : Fin D → ℕ → ℝ} (hve : ∀ e j, |ve e j| ≤ 2 ^ 15)
    {sc x p : Fin R.P → F32} {m mx m' dm al lsum l l' : F32}
    {vt pv : Fin R.P → Fin D → F32} {vsum acc acc' : Fin D → F32}
    (hS : ∀ r : Fin R.P, (R.P * t + r < R.n → (sc r).isFinite = true ∧
        |(sc r).toReal - R.c * R.sv (R.P * t + r)| ≤ R.Es) ∧
      (¬ R.P * t + r < R.n → sc r = FloatV.negInf))
    (hmx : ∃ tr : Tree, tr.Covers R.P ∧ tr.Eval (FloatV.MaxRel true) (laneSeq sc) mx)
    (hm' : FloatV.MaxRel true m mx m')
    (hdm : FloatV.NaNEq (FloatV.sub m m') dm) (hal : P.math .exp .binary32 dm al)
    (hx : ∀ r, FloatV.NaNEq (FloatV.sub (sc r) m') (x r)) (hp : ∀ r, P.math .exp .binary32 (x r) (p r))
    (hlsum : ∃ tr : Tree, tr.Covers R.P ∧ tr.Eval (AddRel false) (laneSeq p) lsum)
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l al) lsum) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l al lsum) l'))
    (hvt : ∀ r e, (vt r e).isFinite = true ∧
      (R.P * t + r < R.n → (vt r e).toReal = ve e (R.P * t + r)) ∧
      (¬ R.P * t + r < R.n → (vt r e).toReal = 0))
    (hpv : ∀ r e, FloatV.NaNEq (FloatV.mul (p r) (vt r e)) (pv r e))
    (hvsum : ∀ e, ∃ tr : Tree, tr.Covers R.P ∧ tr.Eval (AddRel fusion) (laneSeq (fun r => pv r e)) (vsum e))
    (hacc' : ∀ e, FloatV.NaNEq (FloatV.add (FloatV.mul (acc e) al) (vsum e)) (acc' e) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (acc e) al (vsum e)) (acc' e)))
    (hold : RowSt R ve t m l acc) :
    RowSt R ve (t + 1) m' l' acc' := by
  have hm0 : (t = 0 ∧ m = FloatV.negInf) ∨ (0 < t ∧ MInv R t m) := by
    rcases hold with ⟨h1, h2, -⟩ | ⟨h1, h2, -⟩
    · exact Or.inl ⟨h1, h2⟩
    · exact Or.inr ⟨h1, h2⟩
  obtain ⟨hM', hX, hin⟩ := mstep hR hS h0 hmx hm' hx hdm hm0
  refine Or.inr ⟨by omega, hM', ?_, fun e => ?_⟩
  · refine lstep hexp hni hR ht hM' hX hp hin hal hlsum hl' ?_
    rcases hold with ⟨h1, -, h3, h4, -⟩ | ⟨h1, h2, h3, -⟩
    · exact Or.inl ⟨h1, h3, h4⟩
    · exact Or.inr ⟨h1, h2, h3⟩
  · refine astep hexp hni hR ht (hve e) hM' hX hp hin hal (fun r => hvt r e) (fun r => hpv r e) (hvsum e)
      (hacc' e) ?_
    rcases hold with ⟨h1, -, -, -, h5⟩ | ⟨h1, h2, -, h4⟩
    · exact Or.inl ⟨h1, (h5 e).1, (h5 e).2⟩
    · exact Or.inr ⟨h1, h2, h4 e⟩

end Kernels.PagedAttention
