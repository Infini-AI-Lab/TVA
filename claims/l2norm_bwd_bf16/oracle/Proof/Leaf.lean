import Proof.Defs
import Proof.Num
import Proof.RowSum

/-!
# `l2norm_bwd_bf16`: the result of one program

After the walk (`Proof.Walk`), the program has loaded its tiles of `y` and `dy` (widened to
`float32`: `LoadOk`) and its rows of `rstd`, computed `t1`, the products `pr`, the row sums `s`,
`t2`, `t3` and `b_dx`, and stored `b_dx` (rounded to `bfloat16`, or as it is: `Stored`). This is
the leaf: from those facts, the final memory satisfies `Qn` (`leaf`).

Per row: the row's values `a`, `g`, `r` are the loaded lanes; the room condition of `Pre` bounds
every product (`room`); the row sum is within `29u·A + 2600η` of `S` (`rowsum_err`); then per
column the three products, the fusable subtraction (`mulR`, `subcR`, with room from `room_sa`,
`room_t2r`, `room_sub`) and their composition (`dx_err32`), and for a `bfloat16` `dx` the last
rounding (`cvt16R`, `room_dx`, `dx_err16`).
-/

set_option maxHeartbeats 1000000

namespace Claims.L2NormBwd

open TVA

/-- What a load of a `[BT, BD]` tile of `y` or `dy` (`b`), widened to `float32`, gives: each
enabled lane is the finite value of its element in `init`, and each disabled lane is a finite
zero. -/
def LoadOk (b : Sum (Buf .bf16) (Buf .f32)) (init : Mem) (d : Inst) (p : Nat)
    (v : Tensor [d.BT, d.BD] F32) : Prop :=
  (∀ i, enx d p i = true → ValAt b init (gx d p i) (v.get i).toReal ∧ (v.get i).isFinite = true) ∧
    (∀ i, enx d p i = false → (v.get i).isFinite = true ∧ (v.get i).toReal = 0)

/-- Element `k` of `dx` in memory `m` holds the store of the `float32` value `x`: its rounding to
`bfloat16` (up to NaN payload), or `x` itself. -/
def Stored (b : Sum (Buf .bf16) (Buf .f32)) (m : Mem) (k : Nat) (x : F32) : Prop :=
  match b with
  | .inl b => ∃ c : BF16, m.read .bf16 (b.ptr k).addr = some c ∧
      FloatV.NaNEq (FloatV.cvtWith .rn x) c
  | .inr b => m.read .f32 (b.ptr k).addr = some x

/-- Two reads of one element give one value. -/
theorem valAt_unique {b : Sum (Buf .bf16) (Buf .f32)} {m : Mem} {k : Nat} {v w : ℝ}
    (hv : ValAt b m k v) (hw : ValAt b m k w) : v = w := by
  cases b with
  | inl b =>
    obtain ⟨c, hc, -, rfl⟩ := hv; obtain ⟨c', hc', -, rfl⟩ := hw
    rw [hc] at hc'; cases hc'; rfl
  | inr b =>
    obtain ⟨c, hc, -, rfl⟩ := hv; obtain ⟨c', hc', -, rfl⟩ := hw
    rw [hc] at hc'; cases hc'; rfl

/-- The depth bound of a sum over `BD ≤ 512` lanes on a platform with balanced trees is `9`. -/
theorem sumBound_le {P : Platform} (hP : Device.SumDepth P) {n : Nat} (hn : n ≤ 512) :
    tl.sumBound P n ≤ 9 := by
  have hc : Nat.clog 2 n ≤ 9 := Nat.clog_le_of_le_pow (by simpa using hn)
  unfold tl.sumBound
  exact max_le ((hP n).trans hc) hc

/-- `|S| ≤ A`. -/
theorem rowDot_abs (a g : Nat → ℝ) (D : Nat) : |rowDot a g D| ≤ rowAbs a g D :=
  Finset.abs_sum_le_sum_abs _ _

/-- `A ≥ 0`. -/
theorem rowAbs_nonneg (a g : Nat → ℝ) (D : Nat) : 0 ≤ rowAbs a g D :=
  Finset.sum_nonneg (fun _ _ => abs_nonneg _)

/-- **One element** of `dx`: the facts of its lane give `DxOk`. -/
theorem elem_ok {d : Inst} {m : Mem} {k : Nat} {fusion : Bool}
    {gv yv rv sv t1 t2 t3 dx : F32} {g a r S A : ℝ}
    (hgv : gv.isFinite = true) (hyv : yv.isFinite = true) (hrv : rv.isFinite = true)
    (hsv : sv.isFinite = true)
    (hg : gv.toReal = g) (ha : yv.toReal = a) (hr : rv.toReal = r)
    (hA : 0 ≤ A) (hS : |S| ≤ A) (hs : |sv.toReal - S| ≤ 29 / 2 ^ 24 * A + 2600 / 2 ^ 150)
    (hP : (1 + |r|) * (1 + |a|) * (1 + |g| + A) ≤ 2 ^ 124)
    (h1 : FloatV.NaNEq (FloatV.mul gv rv) t1)
    (h2 : FloatV.NaNEq (FloatV.mul sv yv) t2)
    (h3 : FloatV.NaNEq (FloatV.mul t2 rv) t3)
    (hsub : AddRel fusion t1 (FloatV.neg t3) dx)
    (hst : Stored d.dx m k dx) :
    DxOk d m k (g * r - S * a * r) (|r| * (|g| + A * |a|)) ((1 + |r|) * (1 + |a|)) := by
  obtain ⟨hA', ha', hr', hAa, hX, hY, hZ, hXY⟩ := room hA hP
  subst hg ha hr
  -- `t1 = fl(g·r)`
  obtain ⟨ht1f, ht1⟩ := mulR h1 hgv hrv (by rw [abs_mul]; linarith)
  -- `t2 = fl(s·a)`
  have hsa := room_sa hS hs hAa ha'
  obtain ⟨ht2f, ht2⟩ := mulR h2 hsv hyv (by linarith)
  -- `t3 = fl(t2·r)`
  obtain ⟨ht2rB, ht2r⟩ := room_t2r hS hs ht2 hY hZ hr'
  obtain ⟨ht3f, ht3⟩ := mulR h3 ht2f hrv (by linarith)
  -- the fusable `t1 − t3`
  have hroom : |t1.toReal| + |t3.toReal| ≤ 2 ^ 126 := by
    refine room_sub (X := |gv.toReal| * |rv.toReal|) le_rfl ht1 ht3 ht2rB ?_
    have hZ0 : 0 ≤ |yv.toReal| * |rv.toReal| := by positivity
    have : (1 + 1 / 2 ^ 24 : ℝ) * ((1 + 29 / 2 ^ 24) * (A * |yv.toReal| * |rv.toReal|) +
        2600 / 2 ^ 150 * (|yv.toReal| * |rv.toReal|)) + 1 / 2 ^ 150 * |rv.toReal| ≤
        (1 + 1 / 2 ^ 24) * ((1 + 29 / 2 ^ 24) * (A * |yv.toReal| * |rv.toReal|)) +
          2 ^ 124 := by nlinarith
    nlinarith
  obtain ⟨hdxf, hdx⟩ := subcR hsub ht1f ht3f hroom
  have hE := dx_err32 hA hS hs ht1 ht2 ht3 hdx
  unfold DxOk
  unfold Stored at hst
  cases hb : d.dx with
  | inl b =>
    rw [hb] at hst
    obtain ⟨c, hc, hcv⟩ := hst
    obtain ⟨hcf, hce⟩ := cvt16R hcv hdxf (room_dx hA hS hE hP)
    refine ⟨c, hc, hcf, ?_⟩
    have hM : 0 ≤ |rv.toReal| * (|gv.toReal| + A * |yv.toReal|) := by positivity
    have hF : 1 ≤ (1 + |rv.toReal|) * (1 + |yv.toReal|) := by nlinarith [abs_nonneg rv.toReal, abs_nonneg yv.toReal]
    have h := dx_err16 hM hF hE hce
    simp only [dxRnd16, dxTol16, dxFloor16]
    push_cast
    norm_num [zpow_neg] at h ⊢
    linarith
  | inr b =>
    rw [hb] at hst
    refine ⟨dx, hst, hdxf, ?_⟩
    simp only [dxTol32, dxFloor32]
    push_cast
    norm_num [zpow_neg] at hE ⊢
    linarith

/-- **One program**: from the facts the walk gives about its values, the final memory satisfies
`Qn`. -/
theorem leaf {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat} {m : Mem}
    {fusion : Bool} {P : Platform} (hsd : Device.SumDepth P)
    (b_y b_dy t1 pr t2 t3 b_dx : Tensor [d.BT, d.BD] F32) (b_rstd s : Tensor [d.BT] F32)
    (hy : LoadOk d.y init d p b_y) (hg : LoadOk d.dy init d p b_dy)
    (hr : ∀ i, enT d p i = true → init.read .f32 (d.rstd.ptr (gT d p i)).addr = some (b_rstd.get i))
    (h1 : ∀ i, FloatV.NaNEq (FloatV.mul (b_dy.get i) ((b_rstd.col d.BD).get i)) (t1.get i))
    (hp : ∀ i, FloatV.NaNEq (FloatV.mul (b_dy.get i) (b_y.get i)) (pr.get i))
    (hs : ∀ i, ∃ t : Tree, t.Covers d.BD ∧ t.depth ≤ tl.sumBound P d.BD ∧
      t.Eval (AddRel fusion) (fun j => if h : j < d.BD then pr.get2 i ⟨j, h⟩ else default)
        (s.get1 i))
    (h2 : ∀ i, FloatV.NaNEq (FloatV.mul ((s.col d.BD).get i) (b_y.get i)) (t2.get i))
    (h3 : ∀ i, FloatV.NaNEq (FloatV.mul (t2.get i) ((b_rstd.col d.BD).get i)) (t3.get i))
    (hsub : ∀ i, AddRel fusion (t1.get i) (FloatV.neg (t3.get i)) (b_dx.get i))
    (hst : ∀ i, enx d p i = true → Stored d.dx m (gx d p i) (b_dx.get i)) :
    Qn d init p m := by
  intro t ht
  have hsub' : t - p * d.BT < d.BT := row_sub_lt hd ht
  set i0 : Fin d.BT := ⟨t - p * d.BT, hsub'⟩ with hi0
  -- the lane of column `j`
  let L : ∀ j, j < d.BD → Idx [d.BT, d.BD] := fun j hj => (i0, (⟨j, hj⟩, ()))
  have hLen : ∀ j (hj : j < d.D), enx d p (L j (col_lt_bd hd hj)) = true :=
    fun j hj => lane_enx ht hj
  have hLg : ∀ j (hj : j < d.BD), gx d p (L j hj) = xIdx d t j := fun j hj => lane_gx ht
  have hLoff : ∀ j (hj : j < d.BD), d.D ≤ j → enx d p (L j hj) = false := by
    intro j hj hDj
    cases h : enx d p (L j hj)
    · rfl
    · have := enx_col h; simp only [L] at this; omega
  -- the row's values
  let a : Nat → ℝ := fun j => if h : j < d.D then (b_y.get (L j (col_lt_bd hd h))).toReal else 0
  let g : Nat → ℝ := fun j => if h : j < d.D then (b_dy.get (L j (col_lt_bd hd h))).toReal else 0
  let r : ℝ := (b_rstd.get (i0, ())).toReal
  have ha : ∀ j (hj : j < d.D), a j = (b_y.get (L j (col_lt_bd hd hj))).toReal := by
    intro j hj; simp only [a, dif_pos hj]
  have hgj : ∀ j (hj : j < d.D), g j = (b_dy.get (L j (col_lt_bd hd hj))).toReal := by
    intro j hj; simp only [g, dif_pos hj]
  have hva : ∀ j, j < d.D → ValAt d.y init (xIdx d t j) (a j) := by
    intro j hj
    have := (hy.1 _ (hLen j hj)).1
    rwa [hLg, ← ha j hj] at this
  have hvg : ∀ j, j < d.D → ValAt d.dy init (xIdx d t j) (g j) := by
    intro j hj
    have := (hg.1 _ (hLen j hj)).1
    rwa [hLg, ← hgj j hj] at this
  -- the row's `rstd`
  have hrread : init.read .f32 (d.rstd.ptr t).addr = some (b_rstd.get (i0, ())) := by
    have := hr (i0, ()) (lane_enT ht)
    rwa [lane_gT ht] at this
  have hrfin : (b_rstd.get (i0, ())).isFinite = true := by
    obtain ⟨r', c, hc, hcf, -⟩ := (hpre.2.2.2.2.1 t ht.1).1
    rw [hrread] at hc; cases hc; exact hcf
  have hvr : RstdAt d init t r := ⟨_, hrread, hrfin, rfl⟩
  refine ⟨a, g, r, hva, hvg, hvr, ?_⟩
  -- the room of the row
  have hroom := hpre.2.2.2.2.2 t ht.1 a g r hva hvg hvr
  have hA0 := rowAbs_nonneg a g d.D
  have hAle : rowAbs a g d.D ≤ 2 ^ 124 := (room hA0 (hroom 0 hd.d_pos)).1
  -- the row sum
  obtain ⟨tr, hcov, hdep, hev⟩ := hs i0
  have hsrow := rowsum_err (fusion := fusion) hd.d_bd hd.bd_le (sumBound_le hsd hd.bd_le)
    (fun k => if h : k < d.BD then b_dy.get (L k h) else default)
    (fun k => if h : k < d.BD then b_y.get (L k h) else default)
    (fun j => if h : j < d.BD then pr.get2 i0 ⟨j, h⟩ else default) a g
    (fun k hk => by simp only [dif_pos hk]; exact hp _) hcov hdep hev
    (fun k hk => by
      simp only [dif_pos hk]
      by_cases hkD : k < d.D
      · exact (hg.1 _ (hLen k hkD)).2
      · exact (hg.2 _ (hLoff k hk (by omega))).1)
    (fun k hk => by
      simp only [dif_pos hk]
      by_cases hkD : k < d.D
      · exact (hy.1 _ (hLen k hkD)).2
      · exact (hy.2 _ (hLoff k hk (by omega))).1)
    (fun k hk => by simp only [dif_pos (col_lt_bd hd hk)]; exact (hgj k hk).symm)
    (fun k hk => by simp only [dif_pos (col_lt_bd hd hk)]; exact (ha k hk).symm)
    (fun k hDk hk => by simp only [dif_pos hk]; exact (hg.2 _ (hLoff k hk hDk)).2)
    hAle
  obtain ⟨hsf, hserr⟩ := hsrow
  -- each column
  intro j hj
  have hjB := col_lt_bd hd hj
  have hLj := hLen j hj
  have hcolr : (b_rstd.col d.BD).get (L j hjB) = b_rstd.get (i0, ()) := rfl
  have hcols : (s.col d.BD).get (L j hjB) = s.get1 i0 := rfl
  have e := elem_ok (d := d) (m := m) (k := xIdx d t j) (fusion := fusion)
    (gv := b_dy.get (L j hjB)) (yv := b_y.get (L j hjB)) (rv := b_rstd.get (i0, ()))
    (sv := s.get1 i0) (t1 := t1.get (L j hjB)) (t2 := t2.get (L j hjB)) (t3 := t3.get (L j hjB))
    (dx := b_dx.get (L j hjB)) (g := g j) (a := a j) (r := r) (S := rowDot a g d.D)
    (A := rowAbs a g d.D)
    (hg.1 _ hLj).2 (hy.1 _ hLj).2 hrfin hsf (hgj j hj).symm (ha j hj).symm rfl hA0
    (rowDot_abs a g d.D) hserr (hroom j hj)
    (by rw [← hcolr]; exact h1 _) (by rw [← hcols]; exact h2 _)
    (by rw [← hcolr]; exact h3 _) (hsub _)
    (by rw [← hLg j hjB]; exact hst _ hLj)
  exact e

end Claims.L2NormBwd
