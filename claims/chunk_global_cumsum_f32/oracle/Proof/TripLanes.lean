import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Lane

/-!
# The integer side of one trip

The block index `i_t`, the lane offsets `o_t = i_t·BT + arange`, the mask `o_t < T` and the
pointers `s + bos·H + i_h + o_t·H` of trip `t`: lane `l` is position `base + l`, element
`(base + l)·H + head`, enabled when `bt·BT + l < len`. Also the loop's induction variable and
trip count, and the invariant `Inv` the loop keeps.
-/

namespace Claims.ChunkGlobalCumsum

open TVA

variable {d : Inst} {i : Nat}

/-! ## The invariant -/

/-- The elements of the blocks of trips `0, …, t − 1` hold good values. -/
def Done (d : Inst) (i t : Nat) (m : Mem) : Prop :=
  ∀ u, u < t → ∀ l, l < d.BT → bt d i u * d.BT + l < lenOf d i →
    ∃ v : F32, m.read .f32 (d.o.ptr ((base d i u + l) * d.H + headOf d i : Nat)).addr = some v ∧
      Good d (seqOf d i) (base d i u + l) (headOf d i) v

/-- **The loop's invariant** before trip `t`: memory is `init` outside what the trips wrote, no
byte vanished, the running sum is `ZTree`, and the blocks done hold good values. -/
def Inv (d : Inst) (init : Mem) (i t : Nat) (z : F32) (m : Mem) : Prop :=
  (∀ x, x ∉ iterW (Wt d i) t → m x = init x) ∧ (∀ x, init x ≠ none → m x ≠ none) ∧
    ZTree d i t z ∧ Done d i t m

/-! ## Lanes -/

/-- The lane's element. -/
def gE (d : Inst) (i t : Nat) : Idx [d.BT] → Nat :=
  fun j => (base d i t + j.1.val) * d.H + headOf d i

/-- The lane's mask. -/
def enE (d : Inst) (i t : Nat) : Idx [d.BT] → Bool :=
  fun j => decide (bt d i t * d.BT + j.1.val < lenOf d i)

/-- Positions of the sequence are below `2^35`, elements below `2^52`. -/
theorem elem_lt (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) {q : Nat}
    (hq : q < hiOf d i) : q * d.H + headOf d i < d.B * d.T * d.H := by
  have h1 := hiOf_le hd hT hi
  have h2 := headOf_lt hd i
  have : (q + 1) * d.H ≤ d.B * d.T * d.H := Nat.mul_le_mul_right _ (by omega)
  have e : (q + 1) * d.H = q * d.H + d.H := by ring
  omega

theorem BTH_le (hd : d.ok) : d.B * d.T * d.H ≤ 2 ^ 52 := by
  have h1 := BT_le hd
  have h2 := hd.H_le
  have : d.B * d.T * d.H ≤ 2 ^ 35 * 2 ^ 17 := Nat.mul_le_mul h1 h2
  omega

/-- An enabled lane is inside the sequence. -/
theorem lane_hi (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) {t l : Nat}
    (hen : bt d i t * d.BT + l < lenOf d i) : base d i t + l < hiOf d i := by
  have := lo_add_len hT hi
  unfold base; omega

/-- The block index `i_t` of trip `t`. -/
theorem it_toInt (hd : d.ok) {NT ic : I64} {t : Nat} (hNT : NT.toInt = (nbOf d i : Int))
    (hic : ic.toInt = (t : Int)) (ht : t < nbOf d i) (hT : d.varlen = true → TablesOk d)
    (hi : i < gridSize d) :
    (if d.reverse then NT - IntV.ofInt 1 - ic else ic).toInt = (bt d i t : Int) := by
  have hnb : nbOf d i < 2 ^ 63 := by
    have := nb_le_len (i := i) hd; have := lenOf_le hT hi; have := hd.T_le; omega
  unfold bt
  cases hr : d.reverse
  · simp only [Bool.false_eq_true, ite_false]; exact hic
  · simp only [ite_true]
    have h1 : (IntV.ofInt 1 : I64).toInt = ((1 : Nat) : Int) := by
      rw [IntV.toInt_ofInt_one (by norm_num)]; rfl
    have h2 := IntV.I64.sub_nat hNT h1 (by omega) hnb
    exact IntV.I64.sub_nat h2 hic (by omega) (by omega)

/-- `tl.arange` converted to `int64` keeps its values. -/
theorem cvt_arange_toInt {BT : Nat} (hBT : BT ≤ 2 ^ 20) {P : Platform} {c : Tensor [BT] I64}
    (hc : ∀ i, ConvRel P .i32 .i64 ((tl.arange 0 BT).get i) (c.get i)) :
    ∀ i, (c.get i).toInt = (i.1.val : Int) := by
  intro i
  have h1 : c.get i = IntV.ofInt ((tl.arange 0 BT).get i).toInt := hc i
  rw [h1]
  have h2 : ((tl.arange 0 BT).get i).toInt = (i.1.val : Int) := by
    show (IntV.ofInt (0 + (i.1.val : Int)) : I32).toInt = (i.1.val : Int)
    rw [IntV.toInt_ofInt (by norm_num) (by unfold IntV.SFits; constructor <;> omega)]
    omega
  rw [h2]
  rw [IntV.toInt_ofInt (by norm_num) (by unfold IntV.SFits; constructor <;> omega)]

/-- The lane offsets `i_t·BT + arange` have value `bt·BT + l`. -/
theorem o_t_toInt (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d)
    {t : Nat} (ht : t < nbOf d i) {it : I64} (hit : it.toInt = (bt d i t : Int))
    {c : Tensor [d.BT] I64} (hc : ∀ j, (c.get j).toInt = (j.1.val : Int)) (j : Idx [d.BT]) :
    ((bc (it * (IntV.ofNat d.BT : I64)) + c).get j).toInt = ((bt d i t * d.BT + j.1.val : Nat) : Int) := by
  have hb1 := bt_start_lt hd ht
  have hlen := lenOf_le hT hi
  have hT22 := hd.T_le
  have hbt := hd.bt_le
  have hj := j.1.isLt
  have hmul : (it * (IntV.ofNat d.BT : I64)).toInt = ((bt d i t * d.BT : Nat) : Int) :=
    IntV.I64.mul_nat hit (IntV.I64.toInt_ofNat (by omega)) (by omega)
  rw [Tensor.get_add, Tensor.get_full]
  exact IntV.I64.add_nat hmul (hc j) (by omega)

/-- The mask `o_t < T` is the enabled-lane predicate. -/
theorem mask_eq {t : Nat} {TT : I64} (hTT : TT.toInt = (lenOf d i : Int)) {o : Tensor [d.BT] I64}
    (ho : ∀ j, (o.get j).toInt = ((bt d i t * d.BT + j.1.val : Nat) : Int)) (j : Idx [d.BT]) :
    (o.ltS TT).get j = enE d i t j := by
  rw [Tensor.get_ltS, IntV.lt_eq, hTT, ho j]
  unfold enE
  rw [decide_eq_decide]
  constructor <;> intro h <;> omega

/-- The pointer tensor addresses the lane's element. -/
theorem ptr_eq (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) {t : Nat}
    (ht : t < nbOf d i) {bos ih : I64} (hbos : bos.toInt = (loOf d i : Int))
    (hih : ih.toInt = (headOf d i : Int)) {o : Tensor [d.BT] I64}
    (ho : ∀ j, (o.get j).toInt = ((bt d i t * d.BT + j.1.val : Nat) : Int))
    (b : Buf .f32) (j : Idx [d.BT]) :
    (b.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih + o * (IntV.ofNat d.H : I64)).get j
      = b.ptr (gE d i t j : Int) := by
  have hH17 := hd.H_le
  have hH : (IntV.ofNat d.H : I64).toInt = (d.H : Int) := IntV.I64.toInt_ofNat (by omega)
  have hb1 := bt_start_lt hd ht
  have hlen := lenOf_le hT hi
  have hT22 := hd.T_le
  have hbt := hd.bt_le
  have hj := j.1.isLt
  have hlo : loOf d i ≤ d.B * d.T := le_trans (loOf_le_hiOf hT hi) (hiOf_le hd hT hi)
  have hBT := BT_le hd
  have hloH : loOf d i * d.H ≤ 2 ^ 35 * 2 ^ 17 := Nat.mul_le_mul (by omega) hH17
  have hoH : (bt d i t * d.BT + j.1.val) * d.H ≤ 2 ^ 43 * 2 ^ 17 :=
    Nat.mul_le_mul (by omega) hH17
  have hbosH : (bos * (IntV.ofNat d.H : I64)).toInt = ((loOf d i * d.H : Nat) : Int) :=
    IntV.I64.mul_nat hbos hH (by omega)
  have hoH' : ((o * (IntV.ofNat d.H : I64)).get j).toInt
      = (((bt d i t * d.BT + j.1.val) * d.H : Nat) : Int) := by
    rw [Tensor.get_muls]
    exact IntV.I64.mul_nat (ho j) hH (by omega)
  have key : (((b.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih
      + o * (IntV.ofNat d.H : I64)).get j).addr) = (b.ptr (gE d i t j : Int)).addr := by
    simp only [Tensor.get_ptrAdd, Ptr.add_int, Ptr.offset_addr, Buf.ptr_addr]
    rw [hbosH, hih, hoH']
    unfold gE base
    push_cast
    ring
  calc (b.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih + o * (IntV.ofNat d.H : I64)).get j
      = ⟨((b.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih
          + o * (IntV.ofNat d.H : I64)).get j).addr⟩ := rfl
    _ = ⟨(b.ptr (gE d i t j : Int)).addr⟩ := by rw [key]
    _ = b.ptr (gE d i t j : Int) := rfl

/-- Distinct lanes have distinct elements. -/
theorem gE_inj (hd : d.ok) {t : Nat} {x y : Idx [d.BT]} (h : gE d i t x = gE d i t y) : x = y := by
  unfold gE at h
  have hH := hd.H_pos
  have h1 : (base d i t + x.1.val) * d.H = (base d i t + y.1.val) * d.H := by omega
  have h2 : base d i t + x.1.val = base d i t + y.1.val := Nat.eq_of_mul_eq_mul_right hH h1
  have h3 : x.1 = y.1 := Fin.ext (by omega)
  exact Prod.ext h3 (by cases x.2; cases y.2; rfl)

/-- The lanes' elements are the block's elements. -/
theorem img_sub {t : Nat} : Buf.img (gE d i t) (enE d i t) ⊆ blkElems d i (bt d i t) := by
  rintro e ⟨j, hj, rfl⟩
  unfold enE at hj
  exact ⟨j.1.val, j.1.isLt, by simpa using hj, by unfold gE base; ring⟩

end Claims.ChunkGlobalCumsum
