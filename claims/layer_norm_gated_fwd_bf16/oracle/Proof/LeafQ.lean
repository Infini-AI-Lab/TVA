import Proof.LeafRow

/-!
# `layer_norm_gated_fwd_bf16`: one program's rows satisfy `RowDone`

`qn`: from the facts the walk gives (`Proof.Walk`), each row `t` of program `p` has the `rstd`
and the `y` values `RowDone` asks for, in the final memory `m`.
-/

set_option maxHeartbeats 8000000

namespace Claims.LayerNormGatedFwd

open TVA
open TVA TVA.Format

/-- The `[1, BD]` mask block of the model. -/
abbrev mdRow (d : Inst) : Tensor [d.BT, d.BD] Bool :=
  ((tl.arange 0 d.BD).ltS (IntV.ofNat d.D : I32)).row d.BT

/-- **One program**: from the values the walk computed and the facts the `Safe` rules gave, the
final memory satisfies `Qn`. -/
theorem qn {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (p : Nat) (m : Mem)
    (P : Platform) (hsqrt_acc : Device.SqrtF32 P) (hdiv_full : Device.DivFull P)
    (hexp : Device.ExpF32 P) (hovf : Device.ExpF32Overflow P) (hinf : Device.DivF32InfDivisor P)
    {fusion : Bool}
    (b_x16 : Tensor [d.BT, d.BD] BF16) (b_x b_xx : Tensor [d.BT, d.BD] F32)
    (s1 b_var s2 sq b_rstd : Tensor [d.BT] F32) (b_w16 : Tensor [d.BD] BF16)
    (b_w : Tensor [d.BD] F32) (b_x_hat b_y0 : Tensor [d.BT, d.BD] F32)
    (b_g16 : Tensor [d.BT, d.BD] BF16) (b_g b_y : Tensor [d.BT, d.BD] F32)
    (b_y16 : Tensor [d.BT, d.BD] BF16)
    (hx16 : ∀ j : Idx [d.BT, d.BD], enx d p j = true →
      init.read .bf16 (d.x.ptr (gx d p j)).addr = some (b_x16.get j))
    (hz : ∀ j : Idx [d.BT, d.BD], enx d p j = false →
      b_x16.get j = FloatV.litBF16 (FloatV.ofPat 0))
    (hcvt : ∀ j : Idx [d.BT, d.BD], FloatV.NaNEq (FloatV.cvtWith .rn (b_x16.get j)) (b_x.get j))
    (hmul : ∀ j : Idx [d.BT, d.BD], FloatV.NaNEq
      (FloatV.mul ((tl.where (mdRow d) b_x (bc (FloatV.litF32 (FloatV.ofPat 0)))).get j)
        ((tl.where (mdRow d) b_x (bc (FloatV.litF32 (FloatV.ofPat 0)))).get j)) (b_xx.get j))
    (htree : ∀ r : Fin d.BT, ∃ t : Tree, t.Covers d.BD ∧
      t.Eval (AddRel fusion)
        (fun j => if h : j < d.BD then b_xx.get2 r ⟨j, h⟩ else default) (s1.get1 r))
    (hdv : ∀ j : Idx [d.BT], P.div .binary32 (s1.get j)
      ((bc (FloatV.litF32 (FloatV.ofRat (d.D : ℚ)))).get j) (b_var.get j))
    (haddc : ∀ j : Idx [d.BT],
      AddRel fusion (b_var.get j) ((bc (FloatV.hostF32 d.eps)).get j) (s2.get j))
    (hsqrt : ∀ j : Idx [d.BT], P.math .sqrt .binary32 (s2.get j) (sq.get j))
    (hdiv : ∀ j : Idx [d.BT],
      P.div .binary32 ((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))).get j) (sq.get j)
        (b_rstd.get j))
    (hw16 : ∀ j : Idx [d.BD], enW d j = true →
      init.read .bf16 (d.w.ptr (gW j)).addr = some (b_w16.get j))
    (hcvtw : ∀ j : Idx [d.BD], FloatV.NaNEq (FloatV.cvtWith .rn (b_w16.get j)) (b_w.get j))
    (hmA : ∀ j : Idx [d.BT, d.BD],
      FloatV.NaNEq (FloatV.mul (b_x.get j) ((b_rstd.col d.BD).get j)) (b_x_hat.get j))
    (hmB : ∀ j : Idx [d.BT, d.BD],
      FloatV.NaNEq (FloatV.mul (b_x_hat.get j) ((b_w.row d.BT).get j)) (b_y0.get j))
    (hg16 : ∀ j : Idx [d.BT, d.BD], enx d p j = true →
      init.read .bf16 (d.g.ptr (gx d p j)).addr = some (b_g16.get j))
    (hcvtg : ∀ j : Idx [d.BT, d.BD], FloatV.NaNEq (FloatV.cvtWith .rn (b_g16.get j)) (b_g.get j))
    (hgate : ∀ j : Idx [d.BT, d.BD], GateOut d.swish P (b_y0.get j) (b_g.get j) (b_y.get j))
    (hcvt16 : ∀ j : Idx [d.BT, d.BD], FloatV.NaNEq (FloatV.cvtWith .rn (b_y.get j)) (b_y16.get j))
    (hyread : ∀ j : Idx [d.BT, d.BD], enx d p j = true →
      m.read .bf16 (d.y.ptr (gx d p j)).addr = some (b_y16.get j))
    (hrread : ∀ j : Idx [d.BT], enT d p j = true →
      m.read .f32 (d.rstd.ptr (gT d p j)).addr = some (b_rstd.get j)) :
    Qn d init p m := by
  intro t ht
  have htT : t < d.T := ht.1
  have hlo : p * d.BT ≤ t := ht.2.1
  have hhi : t < p * d.BT + d.BT := ht.2.2
  have hsub : t - p * d.BT < d.BT := by omega
  have hD31 : d.D < 2 ^ 31 := by have := hd.d_le; omega
  have hBD31 : d.BD < 2 ^ 31 := by have := hd.bd_le; omega
  let i0 : Fin d.BT := ⟨t - p * d.BT, hsub⟩
  let a : Nat → BF16 := fun j =>
    if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else FloatV.litBF16 (FloatV.ofPat 0)
  have htm : t ∈ tileRow d p := ⟨htT, hlo, hhi⟩
  -- ## the row's inputs
  have haD : ∀ j (hj : j < d.D), a j = b_x16.get2 i0 ⟨j, col_lt_bd hd hj⟩ := by
    intro j hj
    show (if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else
      FloatV.litBF16 (FloatV.ofPat 0)) = _
    rw [dite_eq_left_of_eq_true (eq_true hj)]
  have haPad : ∀ j, ¬ j < d.D → a j = FloatV.ofPat 0 := by
    intro j hj
    show (if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else
      FloatV.litBF16 (FloatV.ofPat 0)) = _
    rw [dite_eq_right_of_eq_false (eq_false hj), FloatV.litBF16_ofPat_zero]
  have hxv : ∀ j, j < d.D → xv d init t j = a j := by
    intro j hj
    have h := hx16 (i0, (⟨j, col_lt_bd hd hj⟩, ())) (lane_enx htm hj)
    rw [lane_gx htm] at h
    rw [haD j hj]
    exact rval_of_read h
  have hrow : rowSum d init t = rs a d.D := by
    unfold rowSum rs
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [hxv j (Finset.mem_range.mp hj)]
  have hrsBD : rs a d.BD = rs a d.D := by
    unfold rs
    refine (Finset.sum_subset (Finset.range_subset_range.mpr hd.d_bd) ?_).symm
    intro x _ hx
    have hxd : ¬ x < d.D := fun h' => hx (Finset.mem_range.mpr h')
    rw [haPad x hxd, zero16.2]; norm_num
  have hAe := hpre.2.2.2.2.2.2.2 t htT
  rw [hrow] at hAe
  have hA0 : (0 : ℝ) ≤ rs a d.D := Finset.sum_nonneg (fun j _ => sq_nonneg _)
  have he : (2 : ℝ) ^ (-125 : ℤ) ≤ d.eps.toReal := by
    have h := Rat.cast_le (K := ℝ).mpr hd.eps_lo
    push_cast at h
    simpa only [FloatV.toReal] using h
  have he0 : (0 : ℝ) < d.eps.toReal := lt_of_lt_of_le (by norm_num) he
  have hsqle : ∀ j, j < d.D → (a j).toReal ^ 2 ≤ rs a d.D := fun j hj =>
    Finset.single_le_sum (f := fun k => (a k).toReal ^ 2) (fun k _ => sq_nonneg _)
      (Finset.mem_range.mpr hj)
  -- ## the per-lane squares
  have hcol : ∀ j (hj : j < d.BD),
      (b_xx.get2 i0 ⟨j, hj⟩).isFinite = true ∧
      |(b_xx.get2 i0 ⟨j, hj⟩).toReal - (a j).toReal ^ 2|
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ) := by
    intro j hj
    have hwh : (tl.where (mdRow d) b_x (bc (FloatV.litF32 (FloatV.ofPat 0)))).get
        (i0, (⟨j, hj⟩, ())) = if j < d.D then b_x.get (i0, (⟨j, hj⟩, ())) else FloatV.ofPat 0 := by
      show (if (mdRow d).get (i0, (⟨j, hj⟩, ())) = true then b_x.get (i0, (⟨j, hj⟩, ()))
        else (bc (FloatV.litF32 (FloatV.ofPat 0)) : Tensor [d.BT, d.BD] F32).get
          (i0, (⟨j, hj⟩, ()))) = _
      have hm : (mdRow d).get (i0, (⟨j, hj⟩, ())) = decide (j < d.D) :=
        get_md_row hD31 hBD31 (i0, (⟨j, hj⟩, ()))
      rw [hm]
      simp only [bc, FloatV.litF32_ofPat_zero, decide_eq_true_eq]
      rfl
    by_cases hjd : j < d.D
    · rw [if_pos hjd] at hwh
      obtain ⟨hfinx, -⟩ := hpre.2.2.2.2.2.1 t htT j hjd
      rw [hxv j hjd] at hfinx
      have hb16 : b_x16.get (i0, (⟨j, hj⟩, ())) = a j := (haD j hjd).symm
      obtain ⟨hbf, hbR, -⟩ := widen16 (hcvt (i0, (⟨j, hj⟩, ()))) (by rw [hb16]; exact hfinx)
      rw [hb16] at hbR
      have hm := hmul (i0, (⟨j, hj⟩, ()))
      rw [hwh] at hm
      have hroom : (a j).toReal ^ 2 ≤ (2 : ℝ) ^ (127 : ℤ) := by
        have := hsqle j hjd
        have e : (2 : ℝ) ^ (125 : ℤ) ≤ (2 : ℝ) ^ (127 : ℤ) := by norm_num
        linarith
      exact lane_sq hfinx ⟨hbf, hbR⟩ hroom hm
    · rw [if_neg hjd] at hwh
      have hm := hmul (i0, (⟨j, hj⟩, ()))
      rw [hwh] at hm
      rw [haPad j hjd]
      exact lane_sq zero16.1 ⟨zero32.1, by rw [zero32.2, zero16.2]⟩
        (by rw [zero16.2]; norm_num) hm
  -- ## the row sum and `rstd`
  have hAeBD : rs a d.BD + d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ) := by
    rw [hrsBD]
    have e : (2 : ℝ) ^ (125 : ℤ) ≤ (2 : ℝ) ^ (126 : ℤ) := by norm_num
    have : rs a d.D + d.eps.toReal ≤ (2 : ℝ) ^ (125 : ℤ) := by exact_mod_cast hAe
    linarith
  obtain ⟨hs1fin, hs1err⟩ := sum_err_bd (n := d.BD) hd.bd_le (e := d.eps.toReal) fusion
    (bxx := fun j => if h : j < d.BD then b_xx.get2 i0 ⟨j, h⟩ else default) (a := a)
    (fun j hj => by rw [dite_eq_left_of_eq_true (eq_true hj)]; exact (hcol j hj).1)
    (fun j hj => by rw [dite_eq_left_of_eq_true (eq_true hj)]; exact (hcol j hj).2)
    hAeBD he (htree i0)
  rw [hrsBD] at hs1err
  have hAe' : rs a d.D + d.eps.toReal ≤ (2 : ℝ) ^ (125 : ℤ) := by exact_mod_cast hAe
  obtain ⟨hrfin, hrerr⟩ := rstd_row hd hsqrt_acc hdiv_full hA0 hAe' hs1fin hs1err
    (hdv (i0, ())) (haddc (i0, ())) (hsqrt (i0, ())) (hdiv (i0, ()))
  have hRdef : rstdRef d init t = (Real.sqrt (rs a d.D / d.D + d.eps.toReal))⁻¹ := by
    rw [rstdRef, hrow]
  have hRpos : 0 < rstdRef d init t := by
    rw [hRdef]
    apply inv_pos.mpr
    apply Real.sqrt_pos.mpr
    have := div_nonneg hA0 (by positivity : (0 : ℝ) ≤ d.D)
    linarith
  refine ⟨⟨b_rstd.get1 i0, ?_, hrfin, ?_⟩, ?_⟩
  · have h := hrread (i0, ()) (lane_enT htm (j := 0) (h1 := hsub) (_h2 := bd_pos hd))
    rw [lane_gT htm (j := 0) (_h2 := bd_pos hd)] at h
    exact h
  · rw [hRdef, show ((rstdTol : ℚ) : ℝ) = (2 : ℝ) ^ (-12 : ℤ) by rw [rstdTol]; norm_num]
    exact hrerr
  -- ## each output element
  intro j hj
  have hjbd : j < d.BD := col_lt_bd hd hj
  have hen : enx d p (i0, (⟨j, hjbd⟩, ())) = true := lane_enx htm hj
  have hgx : gx d p (i0, (⟨j, hjbd⟩, ())) = idx d t j := lane_gx htm
  refine ⟨b_y16.get (i0, (⟨j, hjbd⟩, ())), ?_, ?_⟩
  · have h := hyread _ hen
    rw [hgx] at h
    exact h
  -- the inputs of the element
  obtain ⟨hfinx, hgok⟩ := hpre.2.2.2.2.2.1 t htT j hj
  have hwok := hpre.2.2.2.2.2.2.1 j hj
  have hb16 : b_x16.get (i0, (⟨j, hjbd⟩, ())) = xv d init t j := by rw [hxv j hj, haD j hj]; rfl
  obtain ⟨hbf, hbR, -⟩ := widen16 (hcvt (i0, (⟨j, hjbd⟩, ()))) (by rw [hb16]; exact hfinx)
  rw [hb16] at hbR
  have hgb : b_g16.get (i0, (⟨j, hjbd⟩, ())) = gv d init t j := by
    have h := hg16 _ hen
    rw [hgx] at h
    exact (rval_of_read h).symm
  obtain ⟨hgf, hgR, hgQ⟩ := widen16 (hcvtg (i0, (⟨j, hjbd⟩, ()))) (by rw [hgb]; exact hgok.1)
  rw [hgb] at hgR hgQ
  have hwb : b_w16.get (⟨j, hjbd⟩, ()) = wv d init j := by
    have h := hw16 (⟨j, hjbd⟩, ()) (by simpa only [enW, decide_eq_true_eq] using hj)
    exact (rval_of_read h).symm
  obtain ⟨hwf, hwR, -⟩ := widen16 (hcvtw (⟨j, hjbd⟩, ())) (by rw [hwb]; exact hwok.1)
  rw [hwb] at hwR
  have hw56 : |(wv d init j).toReal| ≤ (2 : ℝ) ^ (56 : ℤ) := by
    rw [FloatV.toReal_eq, show (2 : ℝ) ^ (56 : ℤ) = (((2 : ℚ) ^ 56 : ℚ) : ℝ) by norm_num,
      ← Rat.cast_abs]
    exact Rat.cast_le.mpr hwok.2
  have hg56 : |(gv d init t j).toReal| ≤ (2 : ℝ) ^ (56 : ℤ) := by
    rw [FloatV.toReal_eq, show (2 : ℝ) ^ (56 : ℤ) = (((2 : ℚ) ^ 56 : ℚ) : ℝ) by norm_num,
      ← Rat.cast_abs]
    exact Rat.cast_le.mpr hgok.2
  -- `|x·rstdRef| ≤ 23`
  have hx2 : (xv d init t j).toReal ^ 2 ≤ rs a d.D := by rw [hxv j hj]; exact hsqle j hj
  have hD1 : (1 : ℝ) ≤ d.D := by exact_mod_cast hd.d_pos
  have hD512 : (d.D : ℝ) ≤ 512 := by exact_mod_cast hd.d_le
  have hZ : |(xv d init t j).toReal * rstdRef d init t| ≤ 23 := by
    rw [hRdef]; exact z_bound hx2 hD1 hD512 he0
  have hr' : |((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toReal - rstdRef d init t| ≤
      (2 : ℝ) ^ (-12 : ℤ) * rstdRef d init t := by
    rw [Tensor.get_col, hRdef]; exact hrerr
  have hrfin' : ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).isFinite = true := by
    rw [Tensor.get_col]; exact hrfin
  have hwf' : ((b_w.row d.BT).get (i0, (⟨j, hjbd⟩, ()))).isFinite = true := by
    rw [Tensor.get_row]; exact hwf
  have hwR' : ((b_w.row d.BT).get (i0, (⟨j, hjbd⟩, ()))).toReal = (wv d init j).toReal := by
    rw [Tensor.get_row]; exact hwR
  -- the gate
  obtain ⟨nx, ex, den, sv, hnx, hex, hden, hsv, hy⟩ := hgate (i0, (⟨j, hjbd⟩, ()))
  obtain ⟨hsfin, hserr⟩ := gate_err hexp hovf hdiv_full hinf hgok.1 hg56 hgf hgR hgQ hnx hex hden hsv
  have hcv := hcvt16 (i0, (⟨j, hjbd⟩, ()))
  unfold yTol yRef zRef gateRef gMag
  cases hsw : d.swish with
  | true =>
    rw [hsw, if_pos rfl] at hy
    obtain ⟨yg, hyg, hy4⟩ := hy
    simp only [if_true]
    exact elem_swish hRpos hr' hrfin' hZ hbf hbR hwf' hwR' hw56 hgf hgR hg56 hsfin hserr
      (hmA _) (hmB _) hyg hy4 hcv
  | false =>
    rw [hsw, if_neg (by decide)] at hy
    simp only [Bool.false_eq_true, if_false]
    exact elem_sigmoid hRpos hr' hrfin' hZ hbf hbR hwf' hwR' hw56 hsfin hserr
      (hmA _) (hmB _) hy hcv

end Claims.LayerNormGatedFwd
