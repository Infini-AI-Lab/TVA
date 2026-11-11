import Proof.Common
import Proof.Basic
import Proof.Meta
import Proof.Tol

/-!
# The interleaved tail of one program

The rotation of adjacent channel pairs `2 k`, `2 k + 1`: the swapped partner is read with the
lane permutation of `body_eq`, the pair index with a division by two. The hypotheses are what the
prologue of `body_eq` establishes.
-/

namespace Claims.RotaryVarlen

open TVA

set_option maxHeartbeats 1000000

/-! ## Element access at a concrete index

The library's `[simp]` lemmas for elementwise operations are stated with an `Idx s` argument
whose shape is implicit. At a concrete tuple index such as `(l, c, ())` the elaborator cannot
see through `Idx` at `implicit` transparency, so `rw` and `simp` reject them. The lemmas below
restate them with the index components explicit; each is `rfl`. -/

/-- The value of the pair index `o_d_repeat = o_d / 2`. -/
private theorem oDRepeat_toInt {d : Inst} (hd : d.ok) {o_d_repeat : Tensor [d.BD] I32}
    (hr : ∀ c : Fin d.BD,
      ((tl.arange 0 d.BD).get (c, ())).div?
          ((bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get (c, ()))
        = some (o_d_repeat.get (c, ())))
    (c : Fin d.BD) : (o_d_repeat.get (c, ())).toInt = ((c.val / 2 : Nat) : Int) := by
  have h := hr c
  rw [g_full1] at h
  have hc : ((tl.arange 0 d.BD).get (c, ())).toInt = (c.val : Int) := oD_toInt hd c
  have ha : 0 ≤ ((tl.arange 0 d.BD).get (c, ())).toInt := by
    rw [hc]; exact_mod_cast Nat.zero_le c.val
  have hv := divVal ha ofNat_two_pos h
  rw [hc, ofNat_two_I32] at hv
  rw [hv]
  exact (Int.natCast_ediv c.val 2).symm

/-- The value of `r_par = o_d % 2`. -/
private theorem rPar_toInt {d : Inst} (hd : d.ok) {r_par : Tensor [d.BD] I32}
    (hr : ∀ c : Fin d.BD,
      ((tl.arange 0 d.BD).get (c, ())).rem?
          ((bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get (c, ()))
        = some (r_par.get (c, ())))
    (c : Fin d.BD) : (r_par.get (c, ())).toInt = ((c.val % 2 : Nat) : Int) := by
  have h := hr c
  rw [g_full1] at h
  have hc : ((tl.arange 0 d.BD).get (c, ())).toInt = (c.val : Int) := oD_toInt hd c
  have ha : 0 ≤ ((tl.arange 0 d.BD).get (c, ())).toInt := by
    rw [hc]; exact_mod_cast Nat.zero_le c.val
  have hv := remVal ha ofNat_two_pos h
  rw [hc, ofNat_two_I32] at hv
  rw [hv]
  exact (Int.natCast_emod c.val 2).symm

/-- The value of `r_swap = (o_d + 1) % 2`: the parity of `c + 1`. -/
private theorem rSwap_toInt {d : Inst} (hd : d.ok) {r_swap : Tensor [d.BD] I32}
    (hr : ∀ idx : Idx [d.BD],
      ((tl.arange 0 d.BD + (IntV.ofNat 1 : I32)).get idx).rem?
        ((bc (IntV.ofNat 2 : I32)).get idx) = some (r_swap.get idx)) :
    ∀ c : Fin d.BD, (r_swap.get (c, ())).toInt = (((c.val + 1) % 2 : Nat) : Int) := by
  intro c
  have h := hr (c, ())
  rw [g_full1] at h
  have hx : ((tl.arange 0 d.BD + (IntV.ofNat 1 : I32)).get (c, ())).toInt
      = ((c.val + 1 : Nat) : Int) := by
    rw [g_adds1]
    refine IntV.I32.add_nat (oD_toInt hd c) ofNat_one_I32 ?_
    have h1 : c.val < d.BD := c.isLt
    have h2 : d.BD ≤ 2 ^ 20 := bd_le hd
    omega
  have ha : 0 ≤ ((tl.arange 0 d.BD + (IntV.ofNat 1 : I32)).get (c, ())).toInt := by
    rw [hx]; exact_mod_cast Nat.zero_le (c.val + 1)
  have hv := remVal ha ofNat_two_pos h
  rw [hx, ofNat_two_I32] at hv
  rw [hv]
  exact (Int.natCast_emod (c.val + 1) 2).symm

/-- The swapped channel of `c` is `partner d c`. -/
private theorem oDSwap_toInt {d : Inst} (hd : d.ok) (hint : d.interleaved = true)
    {r_swap : Tensor [d.BD] I32}
    (hrs : ∀ c : Fin d.BD, (r_swap.get (c, ())).toInt = (((c.val + 1) % 2 : Nat) : Int))
    (c : Fin d.BD) :
    ((tl.arange 0 d.BD + r_swap * (IntV.ofNat 2 : I32) - bc (IntV.ofNat 1 : I32)).get (c, ())).toInt
      = (partner d c.val : Int) := by
  have hpar := Nat.mod_two_eq_zero_or_one c.val
  have hstep : ((tl.arange 0 d.BD + r_swap * (IntV.ofNat 2 : I32)).get (c, ())).toInt
      = ((c.val + ((c.val + 1) % 2) * 2 : Nat) : Int) := by
    rw [g_add1]
    refine IntV.I32.add_nat (oD_toInt hd c) ?_ ?_
    · rw [g_muls1]
      refine IntV.I32.mul_nat (hrs c) ofNat_two_I32 ?_
      have h1 : (c.val + 1) % 2 < 2 := Nat.mod_lt _ (by norm_num)
      have h2 : c.val < d.BD := c.isLt
      have h3 : d.BD ≤ 2 ^ 20 := bd_le hd
      omega
    · have h1 : (c.val + 1) % 2 < 2 := Nat.mod_lt _ (by norm_num)
      have h2 : c.val < d.BD := c.isLt
      have h3 : d.BD ≤ 2 ^ 20 := bd_le hd
      omega
  have hsub : ((tl.arange 0 d.BD + r_swap * (IntV.ofNat 2 : I32) - bc (IntV.ofNat 1 : I32)).get (c, ())).toInt
      = ((c.val + ((c.val + 1) % 2) * 2 - 1 : Nat) : Int) := by
    rw [g_sub1]
    refine IntV.I32.sub_nat hstep (by rw [g_full1]; exact ofNat_one_I32) ?_ ?_
    · have h1 : (c.val + 1) % 2 < 2 := Nat.mod_lt _ (by norm_num)
      omega
    · have h1 : (c.val + 1) % 2 < 2 := Nat.mod_lt _ (by norm_num)
      have h2 : c.val < d.BD := c.isLt
      have h3 : d.BD ≤ 2 ^ 20 := bd_le hd
      omega
  rw [hsub]
  rcases hpar with hpar | hpar
  · have hone : (c.val + 1) % 2 = 1 := by
      rw [Nat.add_mod, hpar]
    rw [hone]
    have hp : partner d c.val = c.val + 1 := by
      simp only [partner, hint, hpar, ite_eq_left]
    rw [hp]
    push_cast
    omega
  · have hone : (c.val + 1) % 2 = 0 := by
      rw [Nat.add_mod, hpar]
    rw [hone]
    have hc1 : 1 ≤ c.val := by omega
    have hp : partner d c.val = c.val - 1 := by
      simp only [partner, hint, ite_true]
      rw [ite_eq_right (by omega)]
    rw [hp]
    push_cast
    omega

/-- The `even` mask of a channel is `isFirst d c`. -/
private theorem even_get {d : Inst} (hint : d.interleaved = true) {r_par : Tensor [d.BD] I32}
    (hr : ∀ c : Fin d.BD, (r_par.get (c, ())).toInt = ((c.val % 2 : Nat) : Int))
    (l : Fin d.BT) (c : Fin d.BD) :
    ((r_par.eq (bc (IntV.ofInt 0 : I32))).row d.BT).get (l, c, ()) = isFirst d c.val := by
  rw [g_row1, g_eq1, g_full1, isFirst, hint]
  simp only [↓reduceIte]
  have h0 : (IntV.ofInt (0 : Int) : I32).toInt = 0 := ofInt_zero_I32
  have hiff : (IntV.eq (r_par.get (c, ())) (IntV.ofInt (0 : Int) : I32) = true) ↔ (c.val % 2 = 0) := by
    rw [IntV.eq_decide, hr c, h0, decide_eq_true_iff]
    constructor
    · intro h; exact_mod_cast h
    · intro h; rw [h]; norm_num
  cases he : (IntV.eq (r_par.get (c, ())) (IntV.ofInt (0 : Int) : I32)) with
  | false =>
    have h1 : decide (c.val % 2 = 0) = false := by
      rw [decide_eq_false_iff_not]
      intro hp
      have h2 := hiff.mpr hp
      rw [he] at h2
      exact Bool.false_eq_true.mp h2
    rw [h1]
  | true =>
    have h1 : decide (c.val % 2 = 0) = true := by
      rw [decide_eq_true_iff]
      exact hiff.mp he
    rw [h1]

/-- `lt` on integer vectors is the order of the values. -/
private theorem IntV.lt_decide {s : Bool} {w : Nat} (a b : IntV s w) :
    a.lt b = decide (a.toInt < b.toInt) := by
  by_cases h : a.toInt < b.toInt
  · rw [decide_eq_true h]
    exact (IntV.cmpKey_lt a b).mpr h
  · rw [decide_eq_false h]
    cases hl : a.lt b
    · rfl
    · exact absurd ((IntV.cmpKey_lt a b).mp hl) h

private theorem decide_and_eq (p q : Prop) [Decidable p] [Decidable q] :
    (decide p && decide q) = decide (p ∧ q) := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq]

private theorem decide_not_eq (p : Prop) [Decidable p] :
    (!decide p) = decide (¬p) := by
  by_cases hp : p <;> simp [hp]

/-! ## The lanes of the tile -/

/-- The lanes the tile reads and writes: the lane is on and its channel is a rotated one. -/
private def dLane (d : Inst) (init : Mem) (i : Nat) (idx : Idx [d.BT, d.BD]) : Prop :=
  laneOn d init i idx.1.val ∧ idx.2.1.val < 2 * d.R

/-- The tile's enable predicate `dLane` as a `Bool`. -/
private def dEn (d : Inst) (init : Mem) (i : Nat) (idx : Idx [d.BT, d.BD]) : Bool :=
  @decide (dLane d init i idx) (by unfold dLane laneOn; infer_instance)

private theorem decide_true_aux (p : Prop) (inst : Decidable p) :
    @decide p inst = true ↔ p :=
  @decide_eq_true_iff p inst

private theorem decide_false_aux (p : Prop) (inst : Decidable p) :
    @decide p inst = false ↔ ¬p :=
  @decide_eq_false_iff_not p inst

private theorem dEn_true {d : Inst} {init : Mem} {i : Nat} {idx : Idx [d.BT, d.BD]} :
    dEn d init i idx = true ↔ dLane d init i idx := by
  unfold dEn; exact decide_true_aux _ _

private theorem dEn_false {d : Inst} {init : Mem} {i : Nat} {idx : Idx [d.BT, d.BD]} :
    dEn d init i idx = false ↔ ¬ dLane d init i idx := by
  unfold dEn; exact decide_false_aux _ _

/-- The table row element the cosine of a lane is read from. -/
private def dGCos (d : Inst) (init : Mem) (i : Nat) (idx : Idx [d.BT, d.BD]) : Nat :=
  (posI d init i idx.1.val).toNat * d.R + idx.2.1.val / 2

/-- The table row element the sine of a lane is read from. -/
private def dGSin (d : Inst) (init : Mem) (i : Nat) (idx : Idx [d.BT, d.BD]) : Nat :=
  (posI d init i idx.1.val).toNat * d.R + idx.2.1.val / 2

/-- The `x` element of a lane. -/
private def dGX0 (d : Inst) (init : Mem) (i : Nat) (idx : Idx [d.BT, d.BD]) : Nat :=
  tokIdx d (tokI d init i idx.1.val) (ax2 d i) idx.2.1.val

/-- The `x` element of a lane's partner. -/
private def dGX1 (d : Inst) (init : Mem) (i : Nat) (idx : Idx [d.BT, d.BD]) : Nat :=
  tokIdx d (tokI d init i idx.1.val) (ax2 d i) (partner d idx.2.1.val)

/-- The `y` element of a lane. -/
private def dGY (d : Inst) (init : Mem) (i : Nat) (idx : Idx [d.BT, d.BD]) : Nat :=
  tokIdx d (tokI d init i idx.1.val) (ax2 d i) idx.2.1.val

private theorem bool_eq_of_iff (a b : Bool) (h : (a = true ↔ b = true)) : a = b := by
  by_cases ha : a <;> by_cases hb : b <;> simp_all

/-! ## Bounds for the tail -/

private theorem R_le {d : Inst} (hd : d.ok) : d.R ≤ 128 := by
  have h1 := hd.r_d
  have h2 := hd.d_le
  omega

private theorem R_lt {d : Inst} (hd : d.ok) : d.R < 2 ^ 31 := by
  have h := R_le hd
  omega

private theorem TR_lt63 {d : Inst} (hd : d.ok) : d.TR < 2 ^ 63 := by
  have h1 := hd.tr_lt
  have h2 := hd.r_pos
  have h3 : d.TR ≤ d.TR * d.R := Nat.le_mul_of_pos_right _ h2
  omega

private theorem ofNat_R_I64 {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.R : I64).toInt = (d.R : Int) := by
  refine ofNat_toInt  ?_
  have h2 : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  rw [h2]
  have h1 : (d.R : Int) ≤ 128 := by exact_mod_cast R_le hd
  have h3 : (2 : Int) ^ 7 < 2 ^ 63 := by norm_num
  linarith

/-! ## The program's mask as a definition -/

/-- The token part of the tile's mask: the position is in the sequence and in the table, as the kernel computes it. -/
private def dMt (d : Inst) (i_t T_n : I64) (o_cs : Tensor [d.BT] I64) : Tensor [d.BT] Bool :=
  ((((i_t * (IntV.ofNat d.BT : I64)
      + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).ltS (IntV.ofInt 0 : I64)).not.and
    ((i_t * (IntV.ofNat d.BT : I64)
      + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).ltS T_n)).and
    (o_cs.ltS (IntV.ofInt 0 : I64)).not).and (o_cs.ltS (IntV.ofNat d.TR : I64))

/-- The tile's mask as the kernel computes it: the token part broadcast down the columns, anded with the channel test `c / 2 < R`. -/
private def dMask (d : Inst) (i_t T_n : I64) (o_cs : Tensor [d.BT] I64)
    (o_d_repeat : Tensor [d.BD] I32) : Tensor [d.BT, d.BD] Bool :=
  (dMt d i_t T_n o_cs).col d.BD
    |>.and ((o_d_repeat.ltS (IntV.ofNat d.R : I32)).row d.BT)

private theorem dMt_true {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64}
    (hot : ∀ k : Idx [d.BT], ((i_t * (IntV.ofNat d.BT : I64)
        + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).get k).toInt
          = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hTR : d.TR < 2 ^ 63) (k : Idx [d.BT]) :
    (dMt d i_t T_n o_cs).get k = true ↔ laneOn d init i k.1.val := by
  simp only [dMt, Tensor.get_and, Tensor.get_not, Tensor.get_ltS, Bool.and_eq_true,
    Bool.not_eq_true', IntV.lt_eq]
  rw [hot k, hT, hocs k]
  rw [ofNat_toInt (w := 64) (n := d.TR)  (by
    have h1 : (d.TR : ℤ) < (2 : ℤ) ^ 63 := by exact_mod_cast hTR
    omega)]
  rw [IntV.toInt_ofInt (w := 64) (x := 0) (by norm_num) (by
    simp only [IntV.SFits]; constructor <;> norm_num)]
  simp only [decide_eq_true_eq, decide_eq_false_iff_not]
  constructor
  · rintro ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩
    exact ⟨h2, by omega, h4⟩
  · rintro ⟨h1, h2, h3⟩
    exact ⟨⟨⟨by omega, h1⟩, by omega⟩, h3⟩

private theorem or_lt (d : Inst) (c : Fin d.BD) (hBD : d.BD ≤ 2 ^ 20) (hR : d.R < 2 ^ 31) :
    ((tl.arange 0 d.BD).ltS (IntV.ofNat d.R : I32)).get (c, ())
      = decide ((c.val : ℤ) < (d.R : ℤ)) := by
  have h1 : ((tl.arange 0 d.BD).ltS (IntV.ofNat d.R : I32)).get (c, ())
      = ((tl.arange 0 d.BD).get (c, ())).lt (IntV.ofNat d.R : I32) :=
    Tensor.get_ltS _ _ _
  rw [h1]
  have har : (tl.arange 0 d.BD).get (c, ())
      = (Tensor.arange (0 : ℤ) d.BD).get (c, ()) := rfl
  rw [har, Tensor.get_arange, IntV.lt_eq]
  rw [IntV.toInt_ofInt (w := 32) (by norm_num) (by
    simp only [IntV.SFits]; constructor <;> push_cast <;> omega)]
  rw [ofNat_toInt (w := 32) (n := d.R)  (by
    have h1 : (d.R : ℤ) < (2 : ℤ) ^ 31 := by exact_mod_cast hR
    have h2 : (2 : ℤ) ^ 31 < (2 : ℤ) ^ 63 := by norm_num
    omega)]
  norm_num

private theorem mRow_true {d : Inst} (hd : d.ok) {o_d_repeat : Tensor [d.BD] I32}
    (hr : ∀ c : Fin d.BD, (o_d_repeat.get (c, ())).toInt = ((c.val / 2 : Nat) : Int))
    (kk : Idx [d.BT, d.BD]) :
    ((o_d_repeat.ltS (IntV.ofNat d.R : I32)).row d.BT).get kk = true
      ↔ kk.2.1.val / 2 < d.R := by
  rw [row_get' _ kk, g_ltS1, IntV.lt_eq, hr kk.2.1, ofNat_R_I32 hd,
    decide_eq_true_iff]
  constructor <;> intro h <;> omega

private theorem decide_congr' {p q : Prop} [Decidable p] [Decidable q] (h : p ↔ q) :
    decide p = decide q := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq] at *

private theorem decide_congr2 {p q : Prop} (i : Decidable p) (j : Decidable q) (h : p ↔ q) :
    @decide p i = @decide q j := by
  cases i with
  | isTrue hp =>
      cases j with
      | isTrue hq => rfl
      | isFalse hq => exact absurd (h.mp hp) hq
  | isFalse hp =>
      cases j with
      | isTrue hq => exact absurd (h.mpr hq) hp
      | isFalse hq => rfl

private theorem div2_lt_iff (c R : Nat) : c / 2 < R ↔ c < 2 * R := by
  constructor <;> intro h <;> omega

private theorem dMask_get {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64} {o_d_repeat : Tensor [d.BD] I32}
    (hot : ∀ k : Idx [d.BT], ((i_t * (IntV.ofNat d.BT : I64)
        + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).get k).toInt
          = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hd : d.ok)
    (hr : ∀ c : Fin d.BD, (o_d_repeat.get (c, ())).toInt = ((c.val / 2 : Nat) : Int))
    (kk : Idx [d.BT, d.BD]) :
    (dMask d i_t T_n o_cs o_d_repeat).get kk = true ↔ dLane d init i kk := by
  have hget : (dMask d i_t T_n o_cs o_d_repeat).get kk
      = (((dMt d i_t T_n o_cs).col d.BD).get kk
        && ((o_d_repeat.ltS (IntV.ofNat d.R : I32)).row d.BT).get kk) := by
    show (Tensor.and ((dMt d i_t T_n o_cs).col d.BD)
      ((o_d_repeat.ltS (IntV.ofNat d.R : I32)).row d.BT)).get kk = _
    rw [show kk = (kk.1, kk.2.1, ()) from rfl]
    exact g2_and _ _ kk.1 kk.2.1
  rw [hget, Bool.and_eq_true, col_get' (dMt d i_t T_n o_cs) kk]
  have h1 : (dMt d i_t T_n o_cs).get (kk.1, ()) = true ↔ laneOn d init i kk.1.val :=
    dMt_true hot hT hocs (TR_lt63 hd) (kk.1, ())
  have h2 : ((o_d_repeat.ltS (IntV.ofNat d.R : I32)).row d.BT).get kk = true
      ↔ kk.2.1.val / 2 < d.R := mRow_true hd hr kk
  rw [h1, h2]
  unfold dLane
  simp only [div2_lt_iff]

private theorem dMask_eq {d : Inst} {init : Mem} {i : Nat} {i_t : I64} {T_n : I64}
    {o_cs : Tensor [d.BT] I64} {o_d_repeat : Tensor [d.BD] I32}
    (hot : ∀ k : Idx [d.BT], ((i_t * (IntV.ofNat d.BT : I64)
        + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).get k).toInt
          = (locI d init i k.1.val : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hocs : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val)
    (hd : d.ok)
    (hr : ∀ c : Fin d.BD, (o_d_repeat.get (c, ())).toInt = ((c.val / 2 : Nat) : Int))
    (kk : Idx [d.BT, d.BD]) :
    (dMask d i_t T_n o_cs o_d_repeat).get kk = dEn d init i kk :=
  bool_eq_of_iff _ _ (by rw [dMask_get hot hT hocs hd hr kk, dEn_true])

/-! ## The interleaved program's tensors -/

/-- The tile's token indices, `i_t * BT + arange(0, BT)`. -/
private abbrev di_oT (d : Inst) (i_t : I64) : Tensor [d.BT] I64 :=
  i_t * (IntV.ofNat d.BT : I64) + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)

/-- The tile's channel indices, `arange(0, BD)`. -/
private abbrev di_oR (d : Inst) : Tensor [d.BD] I32 := tl.arange 0 d.BD

/-- The token's `H * D` row offset, broadcast over the channels. -/
private abbrev di_hd (d : Inst) (i_t : I64) : Tensor [d.BT, d.BD] I64 :=
  (di_oT d i_t).col d.BD * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64)

/-- The `cos`/`sin` table offset of each lane, `pos * R + c / 2`. -/
private abbrev di_csOff (d : Inst) (o_cs : Tensor [d.BT] I64)
    (o_d_repeat : Tensor [d.BD] I32) : Tensor [d.BT, d.BD] I64 :=
  o_cs.col d.BD * (IntV.ofNat d.R : I64)
    + (((o_d_repeat.row d.BT).castInt) : Tensor [d.BT, d.BD] I64)

/-- The pointers of the even-channel `x` operand. -/
private abbrev di_pX0 (d : Inst) (x_h : Ptr .bf16) (i_t : I64) :
    Tensor [d.BT, d.BD] (Ptr .bf16) :=
  (x_h + di_hd d i_t) + (di_oR d).row d.BT

/-- The pointers of the swapped-channel `x` operand. -/
private abbrev di_pX1 (d : Inst) (x_h : Ptr .bf16) (i_t : I64) (r_swap : Tensor [d.BD] I32) :
    Tensor [d.BT, d.BD] (Ptr .bf16) :=
  (x_h + di_hd d i_t) +
    (di_oR d + r_swap * (IntV.ofNat 2 : I32) - bc (IntV.ofNat 1 : I32)).row d.BT

/-- The pointers of the `cos` operand. -/
private abbrev di_pCos (d : Inst) (o_cs : Tensor [d.BT] I64)
    (o_d_repeat : Tensor [d.BD] I32) : Tensor [d.BT, d.BD] (Ptr .bf16) :=
  d.cos.ptr 0 + di_csOff d o_cs o_d_repeat

/-- The pointers of the `sin` operand. -/
private abbrev di_pSin (d : Inst) (o_cs : Tensor [d.BT] I64)
    (o_d_repeat : Tensor [d.BD] I32) : Tensor [d.BT, d.BD] (Ptr .bf16) :=
  d.sin.ptr 0 + di_csOff d o_cs o_d_repeat

/-- The pointers of the stored output. -/
private abbrev di_pY (d : Inst) (y_h : Ptr .bf16) (i_t : I64) :
    Tensor [d.BT, d.BD] (Ptr .bf16) :=
  y_h + (di_hd d i_t + (((di_oR d).row d.BT).castInt : Tensor [d.BT, d.BD] I64))

/-- The `H * D` stride tensor takes the token's flat index times `H * D`. -/
private theorem di_hd_val {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    (l : Fin d.BT) (c : Fin d.BD) :
    ((di_hd d i_t).get (l, c, ())).toInt
      = (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) := by
  have hH : (IntV.ofNat d.H : I64).toInt = (d.H : Int) := ofNat_H_I64 hd
  have hD : (IntV.ofNat d.D : I64).toInt = (d.D : Int) := ofNat_D_I64 hd
  have hloc : ((di_oT d i_t).get (l, ())).toInt = (locI d init i l.val : Int) :=
    oT_toInt hd hpre hi hit l
  have hloc0 : (0 : Int) ≤ (locI d init i l.val : Int) := by exact_mod_cast Nat.zero_le _
  have hloc32 : (locI d init i l.val : Int) < 2 ^ 32 := by
    have hcbt := chunkI_mul_BT_lt hd hpre hi
    have hcbt' : (chunkI d init i : Int) * (d.BT : Int) < 2 ^ 31 := by
      have h2 : ((chunkI d init i * d.BT : Nat) : Int) < (2 : Int) ^ 31 := by exact_mod_cast hcbt
      push_cast at h2 ⊢
      linarith
    have hl : (l.val : Int) < 2 ^ 20 := by
      have h1 : (l.val : Int) < (d.BT : Int) := by exact_mod_cast l.isLt
      have h2 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast bt_le hd
      linarith
    simp only [locI]
    push_cast
    linarith
  have hH0 : (0 : Int) ≤ (d.H : Int) := by exact_mod_cast Nat.zero_le d.H
  have hD0 : (0 : Int) ≤ (d.D : Int) := by exact_mod_cast Nat.zero_le d.D
  have hH16 : (d.H : Int) < 2 ^ 16 := by exact_mod_cast hd.grid_h
  have hD256 : (d.D : Int) ≤ 256 := by exact_mod_cast hd.d_le
  have hb1 : IntV.SFits 64 ((locI d init i l.val : Int) * (d.H : Int)) := by
    refine sfits64_of_bounds ?_ ?_
    · nlinarith
    · nlinarith
  have hb2 : IntV.SFits 64 ((locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)) := by
    have hp1 : (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) := mul_nonneg hloc0 hH0
    have hp1' : (locI d init i l.val : Int) * (d.H : Int) < 2 ^ 32 * 2 ^ 16 := by nlinarith
    have hle : (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
        ≤ (2 ^ 32 * 2 ^ 16) * 256 := by
      have h2 : (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
          ≤ (2 ^ 32 * 2 ^ 16) * (d.D : Int) := mul_le_mul_of_nonneg_right hp1'.le hD0
      have h3 : (2 ^ 32 * 2 ^ 16) * (d.D : Int) ≤ (2 ^ 32 * 2 ^ 16) * 256 :=
        mul_le_mul_of_nonneg_left hD256 (by norm_num)
      linarith
    have hp0 : (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) :=
      mul_nonneg hp1 hD0
    have h3 : (2 : Int) ^ 32 * 2 ^ 16 * 256 < 2 ^ 63 := by norm_num
    exact sfits64_of_bounds (by linarith) (by linarith)
  have hs1 : IntV.SFits 64
      (((di_oT d i_t).get (l, ())).toInt * (IntV.ofNat d.H : I64).toInt) := by
    rw [hloc, hH]; exact hb1
  have hs2 : IntV.SFits 64
      ((((di_oT d i_t).get (l, ())) * (IntV.ofNat d.H : I64)).toInt
        * (IntV.ofNat d.D : I64).toInt) := by
    rw [IntV.toInt_mul rfl rfl hs1, hloc, hH, hD]; exact hb2
  have hg : ((di_hd d i_t).get (l, c, ()))
      = ((di_oT d i_t).get (l, ()) * (IntV.ofNat d.H : I64)) * (IntV.ofNat d.D : I64) := by
    simp only [di_hd, g2_muls, Tensor.get_col]
  rw [hg, IntV.toInt_mul rfl rfl hs2,
    IntV.toInt_mul rfl rfl hs1, hloc, hH, hD]

/-- The lane's channel, as an `I32` value. -/
private theorem di_or_val {d : Inst} (hd : d.ok) (l : Fin d.BT) (c : Fin d.BD) :
    (((di_oR d).row d.BT).get (l, c, ())).toInt = (c.val : Int) := by
  have hg : ((di_oR d).row d.BT).get (l, c, ()) = IntV.ofInt (c.val : Int) := by
    rw [di_oR, g_row1, g_arange]
    norm_num
  rw [hg]
  refine IntV.I32.toInt_ofInt ⟨?_, ?_⟩
  · have hc0 : (0 : Int) ≤ (c.val : Int) := by exact_mod_cast Nat.zero_le c.val
    linarith
  · have hcbd : c.val < d.BD := c.isLt
    have hc20 : c.val < 2 ^ 20 := lt_of_lt_of_le hcbd (bd_le hd)
    have hc : (c.val : Int) < 2 ^ 31 := by
      have h1 : (c.val : Int) < (2 : Int) ^ 20 := by exact_mod_cast hc20
      have h2 : (2 : Int) ^ 20 < 2 ^ 31 := by norm_num
      linarith
    linarith

/-- The swapped channel of `c` is `partner d c`. -/
private theorem di_swap_val {d : Inst} (hd : d.ok) (hint : d.interleaved = true)
    {r_swap : Tensor [d.BD] I32}
    (hrs : ∀ c : Fin d.BD, (r_swap.get (c, ())).toInt = (((c.val + 1) % 2 : Nat) : Int))
    (l : Fin d.BT) (c : Fin d.BD) :
    (((di_oR d + r_swap * (IntV.ofNat 2 : I32) - bc (IntV.ofNat 1 : I32)).row d.BT).get
      (l, c, ())).toInt = (partner d c.val : Int) := by
  rw [g_row1]
  exact oDSwap_toInt hd hint hrs c

/-- The repeated channel value, cast to `I64`. -/
private theorem di_cast_repeat_val {d : Inst}
    {o_d_repeat : Tensor [d.BD] I32}
    (hr : ∀ c : Fin d.BD, (o_d_repeat.get (c, ())).toInt = ((c.val / 2 : Nat) : Int))
    (l : Fin d.BT) (c : Fin d.BD) :
    (((o_d_repeat.row d.BT).castInt : Tensor [d.BT, d.BD] I64).get (l, c, ())).toInt
      = ((c.val / 2 : Nat) : Int) := by
  rw [show (((o_d_repeat.row d.BT).castInt : Tensor [d.BT, d.BD] I64).get (l, c, ()))
      = IntV.cast ((o_d_repeat.row d.BT).get (l, c, ())) from rfl]
  rw [IntV.toInt_cast (w := 32) (w' := 64) _ (by norm_num)]
  rw [g_row1, hr c]

/-- The `cos`/`sin` table offset of a lane is `posI * R + c / 2`. -/
private theorem di_csOff_val {d : Inst} (hd : d.ok) {init : Mem} {i : Nat}
    {o_cs : Tensor [d.BT] I64} {o_d_repeat : Tensor [d.BD] I32}
    (hocs : ∀ l : Fin d.BT, (o_cs.get (l, ())).toInt = posI d init i l.val)
    (hr : ∀ c : Fin d.BD, (o_d_repeat.get (c, ())).toInt = ((c.val / 2 : Nat) : Int))
    {l : Fin d.BT} {c : Fin d.BD}
    (hlan : laneOn d init i l.val) (hc : c.val / 2 < d.R) :
    ((di_csOff d o_cs o_d_repeat).get (l, c, ())).toInt
      = posI d init i l.val * (d.R : Int) + ((c.val / 2 : Nat) : Int) := by
  have hR64 : (IntV.ofNat d.R : I64).toInt = (d.R : Int) := ofNat_R_I64 hd
  have hcast := di_cast_repeat_val hr l c
  have hpos0 : (0 : Int) ≤ posI d init i l.val := hlan.2.1
  have hposTR : posI d init i l.val < (d.TR : Int) := hlan.2.2
  have hTR62 : (d.TR : Int) * (d.R : Int) < 2 ^ 62 := by exact_mod_cast hd.tr_lt
  have hc0 : (0 : Int) ≤ ((c.val / 2 : Nat) : Int) := by exact_mod_cast Nat.zero_le _
  have hclt : ((c.val / 2 : Nat) : Int) < (d.R : Int) := by exact_mod_cast hc
  have hR0 : (0 : Int) ≤ (d.R : Int) := by exact_mod_cast Nat.zero_le d.R
  have hb1 : IntV.SFits 64 (posI d init i l.val * (d.R : Int)) := by
    refine sfits64_of_bounds ?_ ?_
    · nlinarith
    · nlinarith
  have hb2 : IntV.SFits 64 (posI d init i l.val * (d.R : Int) + ((c.val / 2 : Nat) : Int)) := by
    refine sfits64_of_bounds ?_ ?_
    · nlinarith
    · nlinarith
  have hs1 : IntV.SFits 64
      ((o_cs.get (l, ())).toInt * (IntV.ofNat d.R : I64).toInt) := by
    rw [hocs l, hR64]; exact hb1
  have hs2 : IntV.SFits 64
      (((o_cs.get (l, ())) * (IntV.ofNat d.R : I64)).toInt
        + (((o_d_repeat.row d.BT).castInt : Tensor [d.BT, d.BD] I64).get (l, c, ())).toInt) := by
    rw [IntV.toInt_mul rfl rfl hs1, hocs l, hR64, hcast]; exact hb2
  have hg : ((di_csOff d o_cs o_d_repeat).get (l, c, ()))
      = (o_cs.get (l, ())) * (IntV.ofNat d.R : I64)
        + (((o_d_repeat.row d.BT).castInt : Tensor [d.BT, d.BD] I64)).get (l, c, ()) := by
    simp only [di_csOff, g2_add, g2_muls, g_col1]
  rw [hg, IntV.toInt_add rfl rfl hs2,
    IntV.toInt_mul rfl rfl hs1, hocs l, hR64, hcast]

/-- The value of the `x` pointer's index, as an `Int`. -/
private theorem dGX0_cast {d : Inst} {init : Mem} {i : Nat} (l : Fin d.BT)
    (c : Fin d.BD) :
    ((dGX0 d init i (l, c, ()) : Nat) : Int)
      = (((bosI d init i * d.H + ax2 d i) * d.D : Nat) : Int)
        + (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) + (c.val : Int) := by
  simp only [dGX0, tokIdx, tokI]
  push_cast
  ring

/-- The value of the partner index, as an `Int`. -/
private theorem dGX1_cast {d : Inst} {init : Mem} {i : Nat} (l : Fin d.BT)
    (c : Fin d.BD) :
    ((dGX1 d init i (l, c, ()) : Nat) : Int)
      = (((bosI d init i * d.H + ax2 d i) * d.D : Nat) : Int)
        + (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) + (partner d c.val : Int) := by
  simp only [dGX1, tokIdx, tokI]
  push_cast
  ring

/-- The value of the `y` index, as an `Int`. -/
private theorem dGY_cast {d : Inst} {init : Mem} {i : Nat} (l : Fin d.BT)
    (c : Fin d.BD) :
    ((dGY d init i (l, c, ()) : Nat) : Int)
      = (((bosI d init i * d.H + ax2 d i) * d.D : Nat) : Int)
        + (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) + (c.val : Int) := by
  simp only [dGY, tokIdx, tokI]
  push_cast
  ring

/-! ## Bounds on the lane stride -/

private theorem di_val_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) (l : Fin d.BT) :
    (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
      ∧ (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) < 2 ^ 62 := by
  have hloc32 : (locI d init i l.val : Int) < 2 ^ 32 := by
    have hcbt : chunkI d init i * d.BT < 2 ^ 31 := chunkI_mul_BT_lt hd hpre hi
    have hcbt' : (chunkI d init i : Int) * (d.BT : Int) < 2 ^ 31 := by
      have h2 : ((chunkI d init i * d.BT : Nat) : Int) < (2 : Int) ^ 31 := by exact_mod_cast hcbt
      push_cast at h2 ⊢
      linarith
    have hl : (l.val : Int) < 2 ^ 20 := by
      have h1 : (l.val : Int) < (d.BT : Int) := by exact_mod_cast l.isLt
      have h2 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast bt_le hd
      linarith
    simp only [locI]
    push_cast
    linarith
  have hloc0 : (0 : Int) ≤ (locI d init i l.val : Int) := by exact_mod_cast Nat.zero_le _
  have hH0 : (0 : Int) ≤ (d.H : Int) := by exact_mod_cast Nat.zero_le d.H
  have hD0 : (0 : Int) ≤ (d.D : Int) := by exact_mod_cast Nat.zero_le d.D
  have hH16 : (d.H : Int) < 2 ^ 16 := by exact_mod_cast hd.grid_h
  have hD256 : (d.D : Int) ≤ 256 := by exact_mod_cast hd.d_le
  have hp1 : (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) := mul_nonneg hloc0 hH0
  have hp1' : (locI d init i l.val : Int) * (d.H : Int) < 2 ^ 32 * 2 ^ 16 := by nlinarith
  have hle : (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
      ≤ (2 ^ 32 * 2 ^ 16) * 256 := by
    have h2 : (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int)
        ≤ (2 ^ 32 * 2 ^ 16) * (d.D : Int) := mul_le_mul_of_nonneg_right hp1'.le hD0
    have h3 : (2 ^ 32 * 2 ^ 16) * (d.D : Int) ≤ (2 ^ 32 * 2 ^ 16) * 256 :=
      mul_le_mul_of_nonneg_left hD256 (by norm_num)
    linarith
  have hp0 : (0 : Int) ≤ (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) :=
    mul_nonneg hp1 hD0
  have h3 : (2 : Int) ^ 32 * 2 ^ 16 * 256 < 2 ^ 62 := by norm_num
  exact ⟨hp0, by linarith⟩

/-- The `H * D` row stride tensor plus the channel: its value. -/
private theorem di_hdR_val {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    (l : Fin d.BT) (c : Fin d.BD) :
    (((di_hd d i_t)
        + (((di_oR d).row d.BT).castInt : Tensor [d.BT, d.BD] I64)).get (l, c, ())).toInt
      = (locI d init i l.val : Int) * (d.H : Int) * (d.D : Int) + (c.val : Int) := by
  have hhd := di_hd_val hd hpre hi hit l c
  have hor : (((((di_oR d).row d.BT).castInt : Tensor [d.BT, d.BD] I64)).get (l, c, ())).toInt
      = (c.val : Int) := by
    rw [show (((di_oR d).row d.BT).castInt : Tensor [d.BT, d.BD] I64).get (l, c, ())
        = IntV.cast (((di_oR d).row d.BT).get (l, c, ())) from rfl]
    rw [IntV.toInt_cast (w := 32) (w' := 64) _ (by norm_num), di_or_val hd l c]
  obtain ⟨hL0, hL63⟩ := di_val_lt hd hpre hi l
  have hc0 : (0 : Int) ≤ (c.val : Int) := by exact_mod_cast Nat.zero_le c.val
  have hc20 : (c.val : Int) < 2 ^ 20 := by
    have h1 : (c.val : Int) < (d.BD : Int) := by exact_mod_cast c.isLt
    have h2 : (d.BD : Int) ≤ 2 ^ 20 := by exact_mod_cast bd_le hd
    linarith
  have hs : IntV.SFits 64 (((di_hd d i_t).get (l, c, ())).toInt
      + ((((di_oR d).row d.BT).castInt : Tensor [d.BT, d.BD] I64).get (l, c, ())).toInt) := by
    rw [hhd, hor]
    refine sfits64_of_bounds ?_ ?_
    · have h2 : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
      rw [h2]; linarith
    · linarith
  have hg : ((di_hd d i_t)
        + (((di_oR d).row d.BT).castInt : Tensor [d.BT, d.BD] I64)).get (l, c, ())
      = (di_hd d i_t).get (l, c, ())
        + (((di_oR d).row d.BT).castInt : Tensor [d.BT, d.BD] I64).get (l, c, ()) :=
    g2_add _ _ l c
  rw [hg, IntV.toInt_add rfl rfl hs, hhd, hor]

/-! ## Pointer addresses -/

/-- The `cos` pointer of a lane. -/
private theorem di_pCos_get {d : Inst} (hd : d.ok) {init : Mem} {i : Nat}
    {o_cs : Tensor [d.BT] I64} {o_d_repeat : Tensor [d.BD] I32}
    (hocs : ∀ l : Fin d.BT, (o_cs.get (l, ())).toInt = posI d init i l.val)
    (hr : ∀ c : Fin d.BD, (o_d_repeat.get (c, ())).toInt = ((c.val / 2 : Nat) : Int))
    {l : Fin d.BT} {c : Fin d.BD} (hlan : laneOn d init i l.val) (hc : c.val / 2 < d.R) :
    (di_pCos d o_cs o_d_repeat).get (l, c, ()) = d.cos.ptr (dGCos d init i (l, c, ())) := by
  have h0 : (0 : Int) ≤ posI d init i l.val * (d.R : Int) + ((c.val / 2 : Nat) : Int) := by
    have h2 : (0 : Int) ≤ (d.R : Int) := by exact_mod_cast Nat.zero_le d.R
    have h3 : (0 : Int) ≤ ((c.val / 2 : Nat) : Int) := by exact_mod_cast Nat.zero_le _
    nlinarith [hlan.2.1]
  have hcast : ((dGCos d init i (l, c, ()) : Nat) : Int)
      = posI d init i l.val * (d.R : Int) + ((c.val / 2 : Nat) : Int) := by
    simp only [dGCos]
    push_cast
    rw [Int.toNat_of_nonneg hlan.2.1]
  refine ptr_eq_of_addr ?_
  have hg : (di_pCos d o_cs o_d_repeat).get (l, c, ())
      = (d.cos.ptr 0).offset (((di_csOff d o_cs o_d_repeat).get (l, c, ())).toInt) := by
    simp only [di_pCos, g2_ptrAdd]
  rw [hg, Ptr.offset_addr, di_csOff_val hd hocs hr hlan hc, Buf.ptr_addr, hcast,
    Buf.ptr_addr]
  ring

/-- The `sin` pointer of a lane. -/
private theorem di_pSin_get {d : Inst} (hd : d.ok) {init : Mem} {i : Nat}
    {o_cs : Tensor [d.BT] I64} {o_d_repeat : Tensor [d.BD] I32}
    (hocs : ∀ l : Fin d.BT, (o_cs.get (l, ())).toInt = posI d init i l.val)
    (hr : ∀ c : Fin d.BD, (o_d_repeat.get (c, ())).toInt = ((c.val / 2 : Nat) : Int))
    {l : Fin d.BT} {c : Fin d.BD} (hlan : laneOn d init i l.val) (hc : c.val / 2 < d.R) :
    (di_pSin d o_cs o_d_repeat).get (l, c, ()) = d.sin.ptr (dGSin d init i (l, c, ())) := by
  have h0 : (0 : Int) ≤ posI d init i l.val * (d.R : Int) + ((c.val / 2 : Nat) : Int) := by
    have h2 : (0 : Int) ≤ (d.R : Int) := by exact_mod_cast Nat.zero_le d.R
    have h3 : (0 : Int) ≤ ((c.val / 2 : Nat) : Int) := by exact_mod_cast Nat.zero_le _
    nlinarith [hlan.2.1]
  have hcast : ((dGSin d init i (l, c, ()) : Nat) : Int)
      = posI d init i l.val * (d.R : Int) + ((c.val / 2 : Nat) : Int) := by
    simp only [dGSin]
    push_cast
    rw [Int.toNat_of_nonneg hlan.2.1]
  refine ptr_eq_of_addr ?_
  have hg : (di_pSin d o_cs o_d_repeat).get (l, c, ())
      = (d.sin.ptr 0).offset (((di_csOff d o_cs o_d_repeat).get (l, c, ())).toInt) := by
    simp only [di_pSin, g2_ptrAdd]
  rw [hg, Ptr.offset_addr, di_csOff_val hd hocs hr hlan hc, Buf.ptr_addr, hcast,
    Buf.ptr_addr]
  ring

/-- The `x` pointer of a lane. -/
private theorem di_pX0_get {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    {x_h : Ptr .bf16}
    (hx : x_h.addr = (d.x.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (l : Fin d.BT) (c : Fin d.BD) :
    (di_pX0 d x_h i_t).get (l, c, ()) = d.x.ptr (dGX0 d init i (l, c, ())) := by
  refine ptr_eq_of_addr ?_
  have hg : (di_pX0 d x_h i_t).get (l, c, ())
      = (x_h.offset (((di_hd d i_t).get (l, c, ())).toInt)).offset
          ((((di_oR d).row d.BT).get (l, c, ())).toInt) := by
    simp only [di_pX0, g2_ptrAdd, g2_ptrAddT]
  rw [hg, Ptr.offset_addr, Ptr.offset_addr, hx, Buf.ptr_addr, di_hd_val hd hpre hi hit l c,
    di_or_val hd l c, dGX0_cast (d := d) (init := init) (i := i) l c,
    Buf.ptr_addr]
  push_cast
  ring_nf

/-- The `x` pointer of a lane's partner channel. -/
private theorem di_pX1_get {d : Inst} (hd : d.ok) (hint : d.interleaved = true)
    {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    {x_h : Ptr .bf16}
    (hx : x_h.addr = (d.x.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    {r_swap : Tensor [d.BD] I32}
    (hrs : ∀ c : Fin d.BD, (r_swap.get (c, ())).toInt = (((c.val + 1) % 2 : Nat) : Int))
    (l : Fin d.BT) (c : Fin d.BD) :
    (di_pX1 d x_h i_t r_swap).get (l, c, ()) = d.x.ptr (dGX1 d init i (l, c, ())) := by
  refine ptr_eq_of_addr ?_
  have hg : (di_pX1 d x_h i_t r_swap).get (l, c, ())
      = (x_h.offset (((di_hd d i_t).get (l, c, ())).toInt)).offset
          ((((di_oR d + r_swap * (IntV.ofNat 2 : I32)
            - bc (IntV.ofNat 1 : I32)).row d.BT).get (l, c, ())).toInt) := by
    simp only [di_pX1, g2_ptrAdd, g2_ptrAddT]
  rw [hg, Ptr.offset_addr, Ptr.offset_addr, hx, Buf.ptr_addr, di_hd_val hd hpre hi hit l c,
    di_swap_val hd hint hrs l c, dGX1_cast (d := d) (init := init) (i := i) l c,
    Buf.ptr_addr]
  push_cast
  ring_nf

/-- The `y` pointer of a lane. -/
private theorem di_pY_get {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    {y_h : Ptr .bf16}
    (hy : y_h.addr = (d.y.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (l : Fin d.BT) (c : Fin d.BD) :
    (di_pY d y_h i_t).get (l, c, ()) = d.y.ptr (dGY d init i (l, c, ())) := by
  refine ptr_eq_of_addr ?_
  have hg : (di_pY d y_h i_t).get (l, c, ())
      = y_h.offset (((di_hd d i_t
          + (((di_oR d).row d.BT).castInt : Tensor [d.BT, d.BD] I64)).get (l, c, ())).toInt) := by
    simp only [di_pY, g2_ptrAdd]
  rw [hg, Ptr.offset_addr, hy, Buf.ptr_addr, di_hdR_val hd hpre hi hit l c,
    dGY_cast (d := d) (init := init) (i := i) l c, Buf.ptr_addr]
  push_cast
  ring_nf

/-! ## Index bounds -/

/-- The partner of a rotated channel is a rotated channel. -/
private theorem di_partner_lt {d : Inst} (hint : d.interleaved = true) {c : Nat}
    (hc : c < 2 * d.R) : partner d c < 2 * d.R := by
  rw [partner, hint]
  simp only [↓reduceIte]
  split_ifs with h
  · omega
  · omega

/-- A token index is `c` modulo `D`. -/
private theorem tokIdx_mod' {d : Inst} {t h c : Nat} (hc : c < d.D) :
    tokIdx d t h c % d.D = c := by
  show ((t * d.H + h) * d.D + c) % d.D = c
  rw [Nat.add_comm ((t * d.H + h) * d.D) c, Nat.add_mul_mod_self_right,
    Nat.mod_eq_of_lt hc]

/-- Token indices determine the token, the head and the channel. -/
private theorem tokIdx_inj' {d : Inst} {t t' h h' c c' : Nat} (hc : c < d.D) (hc' : c' < d.D)
    (hh : h < d.H) (hh' : h' < d.H)
    (he : tokIdx d t h c = tokIdx d t' h' c') : t = t' ∧ h = h' ∧ c = c' := by
  have hDpos : 0 < d.D := Nat.pos_of_ne_zero (by intro h; rw [h] at hc'; omega)
  have hmod : c = c' := by
    have h1 : tokIdx d t h c % d.D = c := tokIdx_mod' hc
    have h2 : tokIdx d t' h' c' % d.D = c' := tokIdx_mod' hc'
    rw [he] at h1
    omega
  have hlin : (t * d.H + h) * d.D = (t' * d.H + h') * d.D := by
    have e : (t * d.H + h) * d.D + c = (t' * d.H + h') * d.D + c' := he
    omega
  have hdiv : t * d.H + h = t' * d.H + h' := Nat.eq_of_mul_eq_mul_right hDpos hlin
  have hh_eq : h = h' := by
    have h1 : (t * d.H + h) % d.H = h := by
      rw [Nat.mul_comm t d.H, Nat.mul_add_mod, Nat.mod_eq_of_lt hh]
    have h2 : (t' * d.H + h') % d.H = h' := by
      rw [Nat.mul_comm t' d.H, Nat.mul_add_mod, Nat.mod_eq_of_lt hh']
    rw [hdiv] at h1
    omega
  refine ⟨?_, hh_eq, hmod⟩
  have hHpos : 0 < d.H := Nat.pos_of_ne_zero (by intro h; rw [h] at hh; omega)
  have hxx : t * d.H = t' * d.H := by
    rw [hh_eq] at hdiv
    exact Nat.add_right_cancel hdiv
  exact Nat.eq_of_mul_eq_mul_right hHpos hxx

/-- The token of a lane is determined by its `tokI` index. -/
private theorem di_tokI_inj {d : Inst} {init : Mem} {i : Nat} {l l' : Fin d.BT}
    (he : tokI d init i l.val = tokI d init i l'.val) : l = l' := by
  apply Fin.ext
  have h1 : locI d init i l.val = locI d init i l'.val := by
    apply Nat.add_left_cancel (n := bosI d init i)
    exact he
  have h2 : chunkI d init i * d.BT + l.val = chunkI d init i * d.BT + l'.val := h1
  exact Nat.add_left_cancel h2

/-- The `cos`/`sin` element of a lane is inside the tables. -/
private theorem di_gCS_lt {d : Inst} (_hd : d.ok) {init : Mem} {i : Nat}
    {len : Nat} (hlen : d.TR * d.R ≤ len)
    {l : Fin d.BT} {c : Fin d.BD} (hlan : laneOn d init i l.val) (hc : c.val / 2 < d.R) :
    dGCos d init i (l, c, ()) < len := by
  have hR0 : (0 : Int) ≤ (d.R : Int) := by exact_mod_cast Nat.zero_le d.R
  have hclt : ((c.val / 2 : Nat) : Int) < (d.R : Int) := by exact_mod_cast hc
  have hpos0 := hlan.2.1
  have hposTR := hlan.2.2
  have hsum : posI d init i l.val * (d.R : Int) + ((c.val / 2 : Nat) : Int)
      < (d.TR : Int) * (d.R : Int) := by
    nlinarith
  have h0 : (0 : Int) ≤ posI d init i l.val * (d.R : Int) + ((c.val / 2 : Nat) : Int) := by
    nlinarith
  have hg : ((dGCos d init i (l, c, ()) : Nat) : Int)
      = posI d init i l.val * (d.R : Int) + ((c.val / 2 : Nat) : Int) := by
    simp only [dGCos]
    push_cast
    rw [Int.toNat_of_nonneg hlan.2.1]
  have hlen' : ((d.TR * d.R : Nat) : Int) ≤ ((len : Nat) : Int) := by exact_mod_cast hlen
  have hlt : ((dGCos d init i (l, c, ()) : Nat) : Int) < ((len : Nat) : Int) := by
    push_cast at hlen'
    rw [hg]
    linarith
  exact_mod_cast hlt

/-- A lane's `x` element is inside the token's rows. -/
private theorem di_gX0_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {len : Nat} (hlen : nTok d * d.H * d.D ≤ len)
    {l : Fin d.BT} {c : Fin d.BD} (hlan : laneOn d init i l.val) (hc : c.val < d.D) :
    dGX0 d init i (l, c, ()) < len := by
  have ht := tokI_lt hd hpre hi hlan
  have hH := ax2_lt hi
  have hD := hc
  have hDpos : 0 < d.D := by
    have h1 := hd.r_pos; have h2 := hd.r_d; omega
  have hstep : tokI d init i l.val * d.H + ax2 d i + 1 ≤ nTok d * d.H := by
    have h1 : tokI d init i l.val * d.H + (ax2 d i + 1)
        ≤ tokI d init i l.val * d.H + d.H :=
      Nat.add_le_add_left (Nat.succ_le_of_lt hH) _
    have h2 : tokI d init i l.val * d.H + d.H = (tokI d init i l.val + 1) * d.H := by
      rw [Nat.add_mul, Nat.one_mul]
    have h3 : (tokI d init i l.val + 1) * d.H ≤ nTok d * d.H :=
      Nat.mul_le_mul_right _ ht
    have h4 : tokI d init i l.val * d.H + (ax2 d i + 1) ≤ nTok d * d.H :=
      le_trans (le_trans h1 (le_of_eq h2)) h3
    rw [Nat.add_assoc]
    exact h4
  have hlt : dGX0 d init i (l, c, ()) < nTok d * d.H * d.D := by
    show (tokI d init i l.val * d.H + ax2 d i) * d.D + c.val < nTok d * d.H * d.D
    have hmul := Nat.mul_le_mul_right d.D hstep
    have hone : (tokI d init i l.val * d.H + ax2 d i + 1) * d.D
        = (tokI d init i l.val * d.H + ax2 d i) * d.D + d.D := by
      rw [Nat.add_mul, Nat.one_mul]
    have h6 : (tokI d init i l.val * d.H + ax2 d i) * d.D + c.val
        < (tokI d init i l.val * d.H + ax2 d i + 1) * d.D := by
      calc (tokI d init i l.val * d.H + ax2 d i) * d.D + c.val
          < (tokI d init i l.val * d.H + ax2 d i) * d.D + d.D :=
            Nat.add_lt_add_left hD _
        _ = (tokI d init i l.val * d.H + ax2 d i + 1) * d.D := hone.symm
    exact lt_of_lt_of_le h6 hmul
  exact lt_of_lt_of_le hlt hlen

/-- A lane's partner element is inside the token's rows. -/
private theorem di_gX1_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {len : Nat} (hlen : nTok d * d.H * d.D ≤ len)
    {l : Fin d.BT} {c : Fin d.BD} (hlan : laneOn d init i l.val)
    (hp : partner d c.val < d.D) :
    dGX1 d init i (l, c, ()) < len := by
  have ht := tokI_lt hd hpre hi hlan
  have hH := ax2_lt hi
  have hDpos : 0 < d.D := by
    have h1 := hd.r_pos; have h2 := hd.r_d; omega
  have hstep : tokI d init i l.val * d.H + ax2 d i + 1 ≤ nTok d * d.H := by
    have h1 : tokI d init i l.val * d.H + (ax2 d i + 1)
        ≤ tokI d init i l.val * d.H + d.H :=
      Nat.add_le_add_left (Nat.succ_le_of_lt hH) _
    have h2 : tokI d init i l.val * d.H + d.H = (tokI d init i l.val + 1) * d.H := by
      rw [Nat.add_mul, Nat.one_mul]
    have h3 : (tokI d init i l.val + 1) * d.H ≤ nTok d * d.H :=
      Nat.mul_le_mul_right _ ht
    have h4 : tokI d init i l.val * d.H + (ax2 d i + 1) ≤ nTok d * d.H :=
      le_trans (le_trans h1 (le_of_eq h2)) h3
    rw [Nat.add_assoc]
    exact h4
  have hlt : dGX1 d init i (l, c, ()) < nTok d * d.H * d.D := by
    show (tokI d init i l.val * d.H + ax2 d i) * d.D + partner d c.val
      < nTok d * d.H * d.D
    have hmul := Nat.mul_le_mul_right d.D hstep
    have hone : (tokI d init i l.val * d.H + ax2 d i + 1) * d.D
        = (tokI d init i l.val * d.H + ax2 d i) * d.D + d.D := by
      rw [Nat.add_mul, Nat.one_mul]
    have h6 : (tokI d init i l.val * d.H + ax2 d i) * d.D + partner d c.val
        < (tokI d init i l.val * d.H + ax2 d i + 1) * d.D := by
      calc (tokI d init i l.val * d.H + ax2 d i) * d.D + partner d c.val
          < (tokI d init i l.val * d.H + ax2 d i) * d.D + d.D :=
            Nat.add_lt_add_left hp _
        _ = (tokI d init i l.val * d.H + ax2 d i + 1) * d.D := hone.symm
    exact lt_of_lt_of_le h6 hmul
  exact lt_of_lt_of_le hlt hlen

/-- The `y` index map is injective on the enabled lanes. -/
private theorem di_gY_inj {d : Inst} (hd : d.ok) {init : Mem} {i : Nat}
    (hi : i < (launch d).size) {k k' : Idx [d.BT, d.BD]}
    (hk : dEn d init i k = true) (hk' : dEn d init i k' = true)
    (he : dGY d init i k = dGY d init i k') : k = k' := by
  obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
  obtain ⟨hlan', hcR'⟩ := dEn_true.mp hk'
  have hD1 : k.2.1.val < d.D := by have h2 := hd.r_d; omega
  have hD2 : k'.2.1.val < d.D := by have h2 := hd.r_d; omega
  have he' : tokIdx d (tokI d init i k.1.val) (ax2 d i) k.2.1.val
      = tokIdx d (tokI d init i k'.1.val) (ax2 d i) k'.2.1.val := he
  obtain ⟨ht, -, hc⟩ := tokIdx_inj' hD1 hD2 (ax2_lt hi) (ax2_lt hi) he'
  have hl : k.1 = k'.1 := di_tokI_inj ht
  have hc2 : k.2.1 = k'.2.1 := Fin.ext hc
  rw [show k = (k.1, k.2.1, ()) from rfl, show k' = (k'.1, k'.2.1, ()) from rfl, hl, hc2]

/-- The `y` index map lands in the write set. -/
private theorem di_imgY_sub {d : Inst} {init : Mem} {i : Nat} :
    Buf.img (dGY d init i) (dEn d init i)
      ⊆ {k | ∃ l c, l < d.BT ∧ c < 2 * d.R ∧ laneOn d init i l ∧
          k = tokIdx d (tokI d init i l) (ax2 d i) c} := by
  rintro z ⟨k, hk, rfl⟩
  obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
  exact ⟨k.1.val, k.2.1.val, k.1.isLt, hcR, hlan, rfl⟩

/-! ## The buffers and the read set -/

private theorem di_x_sub_R (d : Inst) (init : Mem) (i : Nat) : d.x.bytes ⊆ R d init i := by
  intro z hz
  simp only [R, Set.mem_union]
  tauto

private theorem di_cos_sub_R (d : Inst) (init : Mem) (i : Nat) : d.cos.bytes ⊆ R d init i := by
  intro z hz
  simp only [R, Set.mem_union]
  tauto

private theorem di_sin_sub_R (d : Inst) (init : Mem) (i : Nat) : d.sin.bytes ⊆ R d init i := by
  intro z hz
  simp only [R, Set.mem_union]
  tauto

/-! ## Float helpers for the final cells -/

private theorem FloatV.add_comm_fin {f : Format} {x y : FloatV f}
    (hx : x.isFinite = true) (hy : y.isFinite = true) : FloatV.add x y = FloatV.add y x := by
  unfold FloatV.add
  rw [Format.add_of_finite .rn hx hy, Format.add_of_finite .rn hy hx]
  rw [Bool.and_comm (f.sign y.pat) (f.sign x.pat), Rat.add_comm (f.toRat y.pat) (f.toRat x.pat)]

private theorem NaNEq.isFinite' {f : Format} {x y : FloatV f} (h : FloatV.NaNEq x y)
    (hx : x.isFinite = true) : y.isFinite = true := by
  rw [h.eq (FloatV.isFinite_iff.mp hx).1]
  exact hx

private theorem mul_fin_of_bounds {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hxb : |x.toRat| ≤ 2 ^ 125) (hyb : |y.toRat| ≤ 1) :
    (FloatV.mul x y).isFinite = true := by
  refine (FloatV.mul_err Format.binary32_regular hx hy ?_).1
  rw [Format.binary32_maxVal, abs_mul]
  have h1 : |x.toRat| * |y.toRat| ≤ (2 : ℚ) ^ 125 * 1 :=
    mul_le_mul hxb hyb (abs_nonneg _) (by norm_num)
  have h2 : (2 : ℚ) ^ 125 * 1 ≤ ((2 : ℚ) ^ 24 - 1) * 2 ^ 104 := by norm_num
  linarith

private theorem cvt_bf16_toRat (x : BF16) (hx : x.isFinite = true) :
    (FloatV.cvtWith .rn x : F32).toRat = x.toRat :=
  (FloatV.toRat_cvt_of_wider Format.bfloat16_regular Format.binary32_regular
    Format.wider_bfloat16_binary32 .rn hx).2

private theorem cvt_bf16_fin (x : BF16) (hx : x.isFinite = true) :
    (FloatV.cvtWith .rn x : F32).isFinite = true :=
  (FloatV.toRat_cvt_of_wider Format.bfloat16_regular Format.binary32_regular
    Format.wider_bfloat16_binary32 .rn hx).1

private theorem convRel_cvt {P : Platform} {x : BF16} {y : F32}
    (h : ConvRel P .f32 .bf16 y x) : FloatV.NaNEq (FloatV.cvtWith .rn y) x := by
  simp only [ConvRel, ofFloatRel, FloatRel] at h
  rcases h with h | ⟨h, h8, -⟩ | ⟨h8, -, -⟩
  · exact h.2.2
  · exact absurd h8 (by simp only [Format.F8]; decide)
  · exact absurd h8 (by simp only [Format.F8]; decide)

/-- `AddRel` is symmetric when both operands are finite. -/
private theorem AddRel.comm_true {f : Format} {x y r : FloatV f}
    (hx : x.isFinite = true) (hy : y.isFinite = true) (h : AddRel true x y r) :
    AddRel true y x r := by
  rcases h with h | ⟨-, h⟩
  · left
    rw [FloatV.add_comm_fin hy hx]
    exact h
  · right
    rcases h with ⟨p, q, hp, hq, hpq, z, hz⟩ | h3 | h3
    · refine ⟨rfl, Or.inl ⟨q, p, hq, hp, ?_, z, ?_⟩⟩
      · rcases hpq with h | h
        · exact Or.inr h
        · exact Or.inl h
      · rw [Rat.add_comm q p]
        exact hz
    · exact ⟨rfl, Or.inr (Or.inr h3)⟩
    · exact ⟨rfl, Or.inr (Or.inl h3)⟩

/-- A `bfloat16` cast to `float32` keeps its value. -/
private theorem di_bf2f {l : BF16} {b : F32} (hb : FloatV.NaNEq (FloatV.cvtWith .rn l) b)
    (hl : l.isFinite = true) : b.toRat = l.toRat ∧ b.isFinite = true := by
  have hbn : b = FloatV.cvtWith .rn l :=
    hb.eq (FloatV.isFinite_iff.mp (cvt_bf16_fin l hl)).1
  rw [hbn]
  exact ⟨cvt_bf16_toRat l hl, cvt_bf16_fin l hl⟩

/-- The partner channel is a valid channel of the token. -/
private theorem di_partner_lt_D {d : Inst} (hd : d.ok) (hint : d.interleaved = true) {c : Nat}
    (hc : c < 2 * d.R) : partner d c < d.D := by
  have h1 := di_partner_lt hint hc
  have h2 := hd.r_d
  omega

/-- A lane's `y` element is inside the token's rows. -/
private theorem di_gY_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {len : Nat} (hlen : nTok d * d.H * d.D ≤ len)
    {l : Fin d.BT} {c : Fin d.BD} (hlan : laneOn d init i l.val) (hc : c.val < d.D) :
    dGY d init i (l, c, ()) < len :=
  di_gX0_lt hd hpre hi hlen hlan hc

/-- `where` at an `Idx`-typed index. -/
private theorem g2_where_idx {α : Type} {m n : Nat} (c : Tensor [m, n] Bool)
    (x y : Tensor [m, n] α) (i : Idx [m, n]) :
    (tl.where c x y).get i = (if c.get i then x.get i else y.get i) := rfl

/-- **The interleaved tail is safe**: from the prologue's state and the offset line's `o_cs`, the loads, the rotation and the store of the interleaved layout end with `Q`, reading within `R` and writing within `W`. -/
theorem safe_tail_i {P : Platform} {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size)
    {i_t : I64} {x_h y_h : Ptr .bf16} {T_n : I64} {o_cs : Tensor [d.BT] I64}
    (hit : i_t.toInt = (chunkI d init i : Int))
    (hT : T_n.toInt = (lenI d init i : Int))
    (hx : x_h.addr = (d.x.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (hy : y_h.addr = (d.y.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr)
    (hocs : ∀ l : Fin d.BT, (o_cs.get (l, ())).toInt = posI d init i l.val)
    (hint : d.interleaved = true) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      (do
        let o_t : Tensor [d.BT] I64 := i_t * (IntV.ofNat d.BT : I64) + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)
        let m_t : Tensor [d.BT] Bool :=
          (((o_t.ltS (IntV.ofInt 0 : I64)).not.and (o_t.ltS T_n)).and
            (o_cs.ltS (IntV.ofInt 0 : I64)).not).and (o_cs.ltS (IntV.ofNat d.TR : I64))
        let o_d : Tensor [d.BD] I32 := tl.arange 0 d.BD
        let r_swap ← tl.irem (o_d + (IntV.ofNat 1 : I32)) (bc (IntV.ofNat 2 : I32))
        let o_d_swap : Tensor [d.BD] I32 := o_d + r_swap * (IntV.ofNat 2 : I32) - bc (IntV.ofNat 1 : I32)
        let o_d_repeat ← tl.idiv (tl.arange 0 d.BD) (bc (IntV.ofNat 2 : I32))
        let p_x0 : Tensor [d.BT, d.BD] (Ptr .bf16) :=
          x_h + o_t.col d.BD * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + o_d.row d.BT
        let p_x1 : Tensor [d.BT, d.BD] (Ptr .bf16) :=
          x_h + o_t.col d.BD * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + o_d_swap.row d.BT
        let p_cos : Tensor [d.BT, d.BD] (Ptr .bf16) :=
          d.cos.ptr 0 + (o_cs.col d.BD * (IntV.ofNat d.R : I64) + ((o_d_repeat.row d.BT).castInt : Tensor [d.BT, d.BD] I64))
        let p_sin : Tensor [d.BT, d.BD] (Ptr .bf16) :=
          d.sin.ptr 0 + (o_cs.col d.BD * (IntV.ofNat d.R : I64) + ((o_d_repeat.row d.BT).castInt : Tensor [d.BT, d.BD] I64))
        let mask : Tensor [d.BT, d.BD] Bool :=
          (m_t.col d.BD).and ((o_d_repeat.ltS (IntV.ofNat d.R : I32)).row d.BT)
        let one_bf : Tensor [d.BT, d.BD] BF16 ←
          tl.convert .f32 .bf16 (tl.full [d.BT, d.BD] (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
        let l_cos ← tl.load p_cos (some mask) (some one_bf)
        let b_cos : Tensor [d.BT, d.BD] F32 ← tl.cvt l_cos
        let zero_bf : Tensor [d.BT, d.BD] BF16 ←
          tl.convert .f32 .bf16 (tl.full [d.BT, d.BD] (FloatV.litF32 (FloatV.ofPat 0)))
        let l_sin ← tl.load p_sin (some mask) (some zero_bf)
        let b_sin : Tensor [d.BT, d.BD] F32 ← tl.cvt l_sin
        let l_x0 ← tl.load p_x0 (some mask) (some zero_bf)
        let b_x0 : Tensor [d.BT, d.BD] F32 ← tl.cvt l_x0
        let l_x1 ← tl.load p_x1 (some mask) (some zero_bf)
        let b_x1 : Tensor [d.BT, d.BD] F32 ← tl.cvt l_x1
        let b_o0 ← tl.mul b_x0 b_cos
        let b_o1 ← tl.mul b_x1 b_sin
        let r_par ← tl.irem o_d (bc (IntV.ofNat 2 : I32))
        let even : Tensor [d.BT, d.BD] Bool := (Tensor.eq r_par (bc (IntV.ofInt 0 : I32))).row d.BT
        let b_dif ← tl.subc b_o0 b_o1
        let b_sum ← tl.addc b_o0 b_o1
        let b_y : Tensor [d.BT, d.BD] F32 := tl.where even b_dif b_sum
        let p_y : Tensor [d.BT, d.BD] (Ptr .bf16) :=
          y_h + (o_t.col d.BD * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) +
            ((o_d.row d.BT).castInt : Tensor [d.BT, d.BD] I64))
        let s_y : Tensor [d.BT, d.BD] BF16 ← tl.cvt b_y
        tl.store p_y s_y (some mask)) := by
  have hot : ∀ k : Idx [d.BT], ((i_t * (IntV.ofNat d.BT : I64)
      + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).get k).toInt
        = (locI d init i k.1.val : Int) := by
    intro k
    obtain ⟨l, u⟩ := k
    cases u
    exact oT_toInt hd hpre hi hit l
  have hocsP : ∀ k : Idx [d.BT], (o_cs.get k).toInt = posI d init i k.1.val := by
    intro k
    obtain ⟨l, u⟩ := k
    cases u
    exact hocs l
  refine safe_irem ?_ ?_
  · intro idx
    obtain ⟨c, u⟩ := idx
    cases u
    have hx : ((tl.arange 0 d.BD + (IntV.ofNat 1 : I32)).get (c, ())).toInt
        = ((c.val + 1 : Nat) : Int) := by
      rw [g_adds1]
      refine IntV.I32.add_nat (oD_toInt hd c) ofNat_one_I32 ?_
      have h1 : c.val < d.BD := c.isLt
      have h2 : d.BD ≤ 2 ^ 20 := bd_le hd
      omega
    have hb : ((bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get (c, ())).toInt = 2 := by
      rw [g_full1, ofNat_two_I32]
    have ha : 0 ≤ ((tl.arange 0 d.BD + (IntV.ofNat 1 : I32)).get (c, ())).toInt := by
      rw [hx]; exact_mod_cast Nat.zero_le (c.val + 1)
    have hb2 : 0 < ((bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get (c, ())).toInt := by
      rw [hb]; norm_num
    obtain ⟨q, hq, -⟩ :=
      IntV.rem?_of_nonneg (a := (tl.arange 0 d.BD + (IntV.ofNat 1 : I32)).get (c, ()))
        (b := (bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get (c, ())) ha hb2
    exact ⟨q, hq⟩
  · intro r_swap hr_swap
    have hrs : ∀ c : Fin d.BD, (r_swap.get (c, ())).toInt = (((c.val + 1) % 2 : Nat) : Int) := rSwap_toInt hd hr_swap
    refine safe_idiv ?_ ?_
    · intro idx
      obtain ⟨c, u⟩ := idx
      cases u
      have hx : ((tl.arange 0 d.BD).get (c, ())).toInt = (c.val : Int) := oD_toInt hd c
      have hb : ((bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get (c, ())).toInt = 2 := by
        rw [g_full1, ofNat_two_I32]
      have ha : 0 ≤ ((tl.arange 0 d.BD).get (c, ())).toInt := by
        rw [hx]; exact_mod_cast Nat.zero_le c.val
      have hb2 : 0 < ((bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get (c, ())).toInt := by
        rw [hb]; norm_num
      obtain ⟨q, hq, -⟩ :=
        IntV.div?_of_nonneg (a := (tl.arange 0 d.BD).get (c, ()))
          (b := (bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get (c, ())) ha hb2
      exact ⟨q, hq⟩
    · intro o_d_repeat hr_rep
      have hr : ∀ c : Fin d.BD, (o_d_repeat.get (c, ())).toInt = ((c.val / 2 : Nat) : Int) :=
        fun c => oDRepeat_toInt hd (fun c' => hr_rep (c', ())) c
      refine safe_convert ?_ ?_
      · intro idx
        refine ⟨FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)), ?_⟩
        have hg : (tl.full [d.BT, d.BD] (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))).get idx
            = FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) := rfl
        rw [hg]
        show FloatRel P Format.binary32 Format.bfloat16
          (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))
          (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
        exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), FloatV.NaNEq.refl _⟩
      · intro one_bf hone
        refine safe_load_buf (g := dGCos d init i) (en := dEn d init i) d.cos ?_ ?_ hpre.cosv ?_ ?_
        · refine Buf.addrs_of_get d.cos ?_ ?_
          · intro k
            exact dMask_eq hot hT hocsP hd hr k
          · intro k hk
            obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
            rw [show k = (k.1, k.2.1, ()) from rfl]
            exact di_pCos_get hd (fun l => hocsP (l, ())) hr hlan ((div2_lt_iff _ _).mpr hcR)
        · intro k hk
          obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
          rw [show k = (k.1, k.2.1, ()) from rfl]
          exact di_gCS_lt hd hd.cos_len hlan ((div2_lt_iff _ _).mpr hcR)
        · refine Set.Subset.trans (Buf.elems_sub_bytes d.cos ?_) (di_cos_sub_R d init i)
          intro j hj
          obtain ⟨k, hk, rfl⟩ := hj
          obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
          rw [show k = (k.1, k.2.1, ()) from rfl]
          exact di_gCS_lt hd hd.cos_len hlan ((div2_lt_iff _ _).mpr hcR)
        · intro l_cos hl_cos hoff_cos
          refine safe_cvt ?_ ?_
          · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩
          · intro b_cos hb_cos
            refine safe_convert ?_ ?_
            · intro idx
              refine ⟨FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0)), ?_⟩
              have hg : (tl.full [d.BT, d.BD] (FloatV.litF32 (FloatV.ofPat 0))).get idx
                  = FloatV.litF32 (FloatV.ofPat 0) := rfl
              rw [hg]
              show FloatRel P Format.binary32 Format.bfloat16
                (FloatV.litF32 (FloatV.ofPat 0))
                (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0)))
              exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl),
                FloatV.NaNEq.refl _⟩
            · intro zero_bf hzero
              refine safe_load_buf (g := dGSin d init i) (en := dEn d init i) d.sin ?_ ?_ hpre.sinv ?_ ?_
              · refine Buf.addrs_of_get d.sin ?_ ?_
                · intro k
                  exact dMask_eq hot hT hocsP hd hr k
                · intro k hk
                  obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                  rw [show k = (k.1, k.2.1, ()) from rfl]
                  exact di_pSin_get hd (fun l => hocsP (l, ())) hr hlan ((div2_lt_iff _ _).mpr hcR)
              · intro k hk
                obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                rw [show k = (k.1, k.2.1, ()) from rfl]
                exact di_gCS_lt hd hd.sin_len hlan ((div2_lt_iff _ _).mpr hcR)
              · refine Set.Subset.trans (Buf.elems_sub_bytes d.sin ?_) (di_sin_sub_R d init i)
                intro j hj
                obtain ⟨k, hk, rfl⟩ := hj
                obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                rw [show k = (k.1, k.2.1, ()) from rfl]
                exact di_gCS_lt hd hd.sin_len hlan ((div2_lt_iff _ _).mpr hcR)
              · intro l_sin hl_sin hoff_sin
                refine safe_cvt ?_ ?_
                · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩
                · intro b_sin hb_sin
                  refine safe_load_buf (g := dGX0 d init i) (en := dEn d init i) d.x ?_ ?_ hpre.xv ?_ ?_
                  · refine Buf.addrs_of_get d.x ?_ ?_
                    · intro k
                      exact dMask_eq hot hT hocsP hd hr k
                    · intro k hk
                      obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                      rw [show k = (k.1, k.2.1, ()) from rfl]
                      exact di_pX0_get hd hpre hi hit hx k.1 k.2.1
                  · intro k hk
                    obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                    rw [show k = (k.1, k.2.1, ()) from rfl]
                    exact di_gX0_lt hd hpre hi hd.x_len hlan (by have h2 := hd.r_d; omega)
                  · refine Set.Subset.trans (Buf.elems_sub_bytes d.x ?_) (di_x_sub_R d init i)
                    intro j hj
                    obtain ⟨k, hk, rfl⟩ := hj
                    obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                    rw [show k = (k.1, k.2.1, ()) from rfl]
                    exact di_gX0_lt hd hpre hi hd.x_len hlan (by have h2 := hd.r_d; omega)
                  · intro l_x0 hl_x0 hoff_x0
                    refine safe_cvt ?_ ?_
                    · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩
                    · intro b_x0 hb_x0
                      refine safe_load_buf (g := dGX1 d init i) (en := dEn d init i) d.x ?_ ?_ hpre.xv ?_ ?_
                      · refine Buf.addrs_of_get d.x ?_ ?_
                        · intro k
                          exact dMask_eq hot hT hocsP hd hr k
                        · intro k hk
                          obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                          rw [show k = (k.1, k.2.1, ()) from rfl]
                          exact di_pX1_get hd hint hpre hi hit hx hrs k.1 k.2.1
                      · intro k hk
                        obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                        rw [show k = (k.1, k.2.1, ()) from rfl]
                        exact di_gX1_lt hd hpre hi hd.x_len hlan (di_partner_lt_D hd hint hcR)
                      · refine Set.Subset.trans (Buf.elems_sub_bytes d.x ?_) (di_x_sub_R d init i)
                        intro j hj
                        obtain ⟨k, hk, rfl⟩ := hj
                        obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                        rw [show k = (k.1, k.2.1, ()) from rfl]
                        exact di_gX1_lt hd hpre hi hd.x_len hlan (di_partner_lt_D hd hint hcR)
                      · intro l_x1 hl_x1 hoff_x1
                        refine safe_cvt ?_ ?_
                        · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩
                        · intro b_x1 hb_x1
                          refine safe_mul ?_
                          · intro b_o0 hb_o0
                            refine safe_mul ?_
                            · intro b_o1 hb_o1
                              refine safe_irem ?_ ?_
                              · intro idx
                                obtain ⟨c, u⟩ := idx
                                cases u
                                have hx : ((tl.arange 0 d.BD).get (c, ())).toInt
                                    = (c.val : Int) := oD_toInt hd c
                                have hb : ((bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get
                                    (c, ())).toInt = 2 := by
                                  rw [g_full1, ofNat_two_I32]
                                have ha : 0 ≤ ((tl.arange 0 d.BD).get (c, ())).toInt := by
                                  rw [hx]; exact_mod_cast Nat.zero_le c.val
                                have hb2 : 0 < ((bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get
                                    (c, ())).toInt := by
                                  rw [hb]; norm_num
                                obtain ⟨q, hq, -⟩ :=
                                  IntV.rem?_of_nonneg (a := (tl.arange 0 d.BD).get (c, ()))
                                    (b := (bc (IntV.ofNat 2 : I32) : Tensor [d.BD] I32).get
                                      (c, ())) ha hb2
                                exact ⟨q, hq⟩
                              · intro r_par hr_par
                                have hrp : ∀ c : Fin d.BD, (r_par.get (c, ())).toInt = ((c.val % 2 : Nat) : Int) :=
                                  fun c => rPar_toInt hd (fun c' => hr_par (c', ())) c
                                refine safe_subc ?_
                                · intro b_dif hb_dif
                                  refine safe_addc ?_
                                  · intro b_sum hb_sum
                                    refine safe_cvt ?_ ?_
                                    · exact ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩
                                    · intro s_y hs_y
                                      refine safe_last ?_
                                      refine safe_store_buf (g := dGY d init i) (en := dEn d init i) d.y ?_ ?_ ?_ hpre.yv ?_ ?_
                                      · refine Buf.addrs_of_get d.y ?_ ?_
                                        · intro k
                                          exact dMask_eq hot hT hocsP hd hr k
                                        · intro k hk
                                          obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                                          rw [show k = (k.1, k.2.1, ()) from rfl]
                                          exact di_pY_get hd hpre hi hit hy k.1 k.2.1
                                      · intro k hk
                                        obtain ⟨hlan, hcR⟩ := dEn_true.mp hk
                                        rw [show k = (k.1, k.2.1, ()) from rfl]
                                        exact di_gY_lt hd hpre hi hd.y_len hlan (by have h2 := hd.r_d; omega)
                                      · intro k1 k2 hk1 hk2 he
                                        exact di_gY_inj hd hi hk1 hk2 he
                                      · exact Buf.elems_mono d.y (di_imgY_sub (d := d) (init := init) (i := i))
                                      · intro m' hread' hdisj hpres
                                        refine safe_ret ?_
                                        intro l c hl hcR hlan
                                        have hseq : seqI d init i < nSeq d := seqI_lt hd hpre hi
                                        have hloc : locI d init i l < lenAt d init (seqI d init i) := by
                                          have h := hlan.1
                                          simp only [lenI] at h
                                          exact_mod_cast h
                                        have hoff : 0 ≤ (locI d init i l : Int) + offAt d init (seqI d init i) := hlan.2.1
                                        have hD : c < d.D := by
                                          have h2 := hd.r_d
                                          omega
                                        have hpartD : partner d c < d.D := di_partner_lt_D hd hint hcR
                                        have hcBD : c < d.BD := by
                                          have h2 := hd.r_bd
                                          omega
                                        let kk : Idx [d.BT, d.BD] := (⟨l, hl⟩, ⟨c, hcBD⟩, ())
                                        have hen : dEn d init i kk = true := dEn_true.mpr ⟨hlan, hcR⟩
                                        have hcosr : bfAt d.cos init (dGCos d init i kk) = some (l_cos.get kk) := hl_cos kk hen
                                        have hsinr : bfAt d.sin init (dGSin d init i kk) = some (l_sin.get kk) := hl_sin kk hen
                                        have hx0r : bfAt d.x init (dGX0 d init i kk) = some (l_x0.get kk) := hl_x0 kk hen
                                        have hx1r : bfAt d.x init (dGX1 d init i kk) = some (l_x1.get kk) := hl_x1 kk hen
                                        have hmod0 : dGX0 d init i kk % d.D = c := by
                                          show tokIdx d (tokI d init i l) (ax2 d i) c % d.D = c
                                          exact tokIdx_mod' hD
                                        have hmod1 : dGX1 d init i kk % d.D = partner d c := by
                                          show tokIdx d (tokI d init i l) (ax2 d i) (partner d c) % d.D = partner d c
                                          exact tokIdx_mod' hpartD
                                        obtain ⟨a, ha_mem, ha_ok⟩ :=
                                          hpre.x_vals (dGX0 d init i kk)
                                            (di_gX0_lt hd hpre hi (len := nTok d * d.H * d.D) le_rfl hlan hD)
                                            (by rw [hmod0]; exact hcR)
                                        obtain ⟨b, hb_mem, hb_ok⟩ :=
                                          hpre.x_vals (dGX1 d init i kk)
                                            (di_gX1_lt hd hpre hi (len := nTok d * d.H * d.D) le_rfl hlan hpartD)
                                            (by rw [hmod1]; exact di_partner_lt hint hcR)
                                        obtain ⟨cs, hcs_mem, hcs_ok⟩ :=
                                          hpre.cos_vals (seqI d init i) (locI d init i l) (c / 2) hseq hloc hoff
                                            ((div2_lt_iff _ _).mpr hcR)
                                        obtain ⟨sn, hsn_mem, hsn_ok⟩ :=
                                          hpre.sin_vals (seqI d init i) (locI d init i l) (c / 2) hseq hloc hoff
                                            ((div2_lt_iff _ _).mpr hcR)
                                        have hcsm : bfAt d.cos init (dGCos d init i kk) = some cs := hcs_mem
                                        have hsnm : bfAt d.sin init (dGSin d init i kk) = some sn := hsn_mem
                                        have hal : l_x0.get kk = a := Option.some.inj (hx0r.symm.trans ha_mem)
                                        have hbl : l_x1.get kk = b := Option.some.inj (hx1r.symm.trans hb_mem)
                                        have hcl : l_cos.get kk = cs := Option.some.inj (hcosr.symm.trans hcsm)
                                        have hsl : l_sin.get kk = sn := Option.some.inj (hsinr.symm.trans hsnm)
                                        have hx0fin : (l_x0.get kk).isFinite = true := by rw [hal]; exact ha_ok.1
                                        have hx1fin : (l_x1.get kk).isFinite = true := by rw [hbl]; exact hb_ok.1
                                        have hcosfin : (l_cos.get kk).isFinite = true := by rw [hcl]; exact hcs_ok.1
                                        have hsinfin : (l_sin.get kk).isFinite = true := by rw [hsl]; exact hsn_ok.1
                                        have haf := di_bf2f (l := l_x0.get kk) (b := b_x0.get kk) (hb_x0 kk) hx0fin
                                        have hbf := di_bf2f (l := l_x1.get kk) (b := b_x1.get kk) (hb_x1 kk) hx1fin
                                        have hcf := di_bf2f (l := l_cos.get kk) (b := b_cos.get kk) (hb_cos kk) hcosfin
                                        have hsf := di_bf2f (l := l_sin.get kk) (b := b_sin.get kk) (hb_sin kk) hsinfin
                                        have ha_ok' : XOk (l_x0.get kk) := by rw [hal]; exact ha_ok
                                        have hb_ok' : XOk (l_x1.get kk) := by rw [hbl]; exact hb_ok
                                        have hcs_ok' : TabOk (l_cos.get kk) := by rw [hcl]; exact hcs_ok
                                        have hsn_ok' : TabOk (l_sin.get kk) := by rw [hsl]; exact hsn_ok
                                        have hpair : pairOf d c = c / 2 := by simp [pairOf, hint]
                                        have ha_g : bfAt d.x init (tokIdx d (tokI d init i l) (ax2 d i) c) = some (l_x0.get kk) := by
                                          rw [hal]
                                          exact ha_mem
                                        have hb_g : bfAt d.x init (tokIdx d (tokI d init i l) (ax2 d i) (partner d c))
                                            = some (l_x1.get kk) := by
                                          rw [hbl]
                                          exact hb_mem
                                        have hcs_g : bfAt d.cos init ((posI d init i l).toNat * d.R + pairOf d c) = some (l_cos.get kk) := by
                                          rw [hpair, hcl]
                                          exact hcsm
                                        have hsn_g : bfAt d.sin init ((posI d init i l).toNat * d.R + pairOf d c) = some (l_sin.get kk) := by
                                          rw [hpair, hsl]
                                          exact hsnm
                                        have hy_g : bfAt d.y m' (tokIdx d (tokI d init i l) (ax2 d i) c) = some (s_y.get kk) :=
                                          hread' kk hen
                                        have hmain : (s_y.get kk).isFinite = true ∧
                                            |(s_y.get kk).toRat - rotRef (isFirst d c) (l_x0.get kk).toRat (l_x1.get kk).toRat
                                              (l_cos.get kk).toRat (l_sin.get kk).toRat| ≤
                                              tol (isFirst d c) (l_x0.get kk).toRat (l_x1.get kk).toRat
                                                (l_cos.get kk).toRat (l_sin.get kk).toRat := by
                                          by_cases hpar : c % 2 = 0
                                          · have hisF : isFirst d c = true := by simp [isFirst, hint, hpar]
                                            have hev : ((Tensor.eq r_par (bc (IntV.ofInt 0 : I32))).row d.BT).get kk = isFirst d c := by
                                              rw [show kk = (⟨l, hl⟩, ⟨c, hcBD⟩, ()) from rfl]
                                              exact even_get hint hrp (⟨l, hl⟩ : Fin d.BT) (⟨c, hcBD⟩ : Fin d.BD)
                                            have hcon : (tl.where ((Tensor.eq r_par (bc (IntV.ofInt 0 : I32))).row d.BT) b_dif b_sum).get kk
                                                = b_dif.get kk := by
                                              simp only [g2_where_idx, hev]
                                              simp [hisF]
                                            have hres := tol_sub (a := l_x0.get kk) (b := l_x1.get kk) (cs := l_cos.get kk)
                                              (sn := l_sin.get kk) (af := b_x0.get kk) (bf := b_x1.get kk) (cf := b_cos.get kk)
                                              (sf := b_sin.get kk) (p := b_o0.get kk) (q := b_o1.get kk) (r := b_dif.get kk)
                                              (v := s_y.get kk) haf.1 hbf.1 hcf.1 hsf.1 haf.2 hbf.2 hcf.2 hsf.2
                                              (hb_o0 kk) (hb_o1 kk) (hb_dif kk) (hcon ▸ hs_y kk) ha_ok' hb_ok' hcs_ok' hsn_ok'
                                            exact ⟨hres.1, by rw [hisF]; exact hres.2⟩
                                          · have hisF : isFirst d c = false := by simp [isFirst, hint, hpar]
                                            have hbo0fin : (b_o0.get kk).isFinite = true :=
                                              NaNEq.isFinite' (hb_o0 kk)
                                                (mul_fin_of_bounds haf.2 hcf.2 (by rw [haf.1]; exact ha_ok'.2)
                                                  (by rw [hcf.1]; exact hcs_ok'.2))
                                            have hbo1fin : (b_o1.get kk).isFinite = true :=
                                              NaNEq.isFinite' (hb_o1 kk)
                                                (mul_fin_of_bounds hbf.2 hsf.2 (by rw [hbf.1]; exact hb_ok'.2)
                                                  (by rw [hsf.1]; exact hsn_ok'.2))
                                            have hcomm := AddRel.comm_true hbo0fin hbo1fin (hb_sum kk)
                                            have hev : ((Tensor.eq r_par (bc (IntV.ofInt 0 : I32))).row d.BT).get kk = isFirst d c := by
                                              rw [show kk = (⟨l, hl⟩, ⟨c, hcBD⟩, ()) from rfl]
                                              exact even_get hint hrp (⟨l, hl⟩ : Fin d.BT) (⟨c, hcBD⟩ : Fin d.BD)
                                            have hcon : (tl.where ((Tensor.eq r_par (bc (IntV.ofInt 0 : I32))).row d.BT) b_dif b_sum).get kk
                                                = b_sum.get kk := by
                                              simp only [g2_where_idx, hev]
                                              simp [hisF]
                                            have hres := tol_add (a := l_x0.get kk) (b := l_x1.get kk) (cs := l_cos.get kk)
                                              (sn := l_sin.get kk) (af := b_x0.get kk) (bf := b_x1.get kk) (cf := b_cos.get kk)
                                              (sf := b_sin.get kk) (p := b_o1.get kk) (q := b_o0.get kk) (r := b_sum.get kk)
                                              (v := s_y.get kk) haf.1 hbf.1 hcf.1 hsf.1 haf.2 hbf.2 hcf.2 hsf.2
                                              (hb_o1 kk) (hb_o0 kk) hcomm (hcon ▸ hs_y kk) ha_ok' hb_ok' hcs_ok' hsn_ok'
                                            exact ⟨hres.1, by rw [hisF]; exact hres.2⟩
                                        unfold CellOk
                                        exact ⟨l_x0.get kk, l_x1.get kk, l_cos.get kk, l_sin.get kk, s_y.get kk,
                                          ha_g, hb_g, hcs_g, hsn_g, hy_g, hmain.1, hmain.2⟩

end Claims.RotaryVarlen
