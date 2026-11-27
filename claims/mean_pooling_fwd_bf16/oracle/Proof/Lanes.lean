import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo

/-!
# Sizes and the lanes' integer values

The buffers have at most `2^63` elements (`Buf.Valid`), so every `int64` offset the program
computes is below `2^63` and nothing wraps: the scalar offsets of the load (`eX`) and of the store
(`eO`), the rows `chk·BT + r` and columns `iD·BD + l`, and the divisor `L`.
-/

namespace Claims.MeanPoolingFwd

open TVA

/-- A valid `bfloat16` buffer has at most `2^63` elements. -/
theorem len_le_of_valid {b : Buf .bf16} {m : Mem} (hv : b.Valid m) : b.len ≤ 2 ^ 63 := by
  obtain ⟨hb0, hb1, -, -⟩ := hv
  have hb : (DType.bf16).bytes = 2 := rfl
  rw [hb] at hb1
  have : (b.len : Int) ≤ 2 ^ 63 := by omega
  exact_mod_cast this

/-- The input has at most `2^63` elements: `B·T·H·D ≤ 2^63`. -/
theorem BTHD_le {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) :
    d.B * d.T * d.H * d.D ≤ 2 ^ 63 := le_trans hd.x_len (len_le_of_valid hpre.1)

/-- The output has at most `2^63` elements: `nRows·H·D ≤ 2^63`. -/
theorem nRHD_le {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) :
    nRows d * d.H * d.D ≤ 2 ^ 63 := le_trans hd.o_len (len_le_of_valid hpre.2.1)

/-- `(n + 1)·H·D ≤ M·H·D` when `n < M`. -/
theorem succ_HD_le {n M H D : Nat} (h : n < M) : (n + 1) * H * D ≤ M * H * D :=
  Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ h)

/-- The largest element of the loaded matrix is inside the input: `eX + (lenS − 1)·H·D + D ≤ B·T·H·D`. -/
theorem eX_last {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) :
    eX d i + (lenS d i - 1) * (d.H * d.D) + d.D ≤ d.B * d.T * d.H * d.D := by
  have h1 := bos_len_le hd hT hi
  have h2 := chk_lt hd hT hi
  have hh := (bOf_lt hd hi).2
  have hlen : 1 ≤ lenS d i := by omega
  unfold eX
  have e1 : (bos d i * d.H + hOf d i) * d.D + (lenS d i - 1) * (d.H * d.D) + d.D =
      (bos d i * d.H + hOf d i + 1) * d.D + (lenS d i - 1) * d.H * d.D := by ring
  have e2 : (bos d i * d.H + hOf d i + 1) * d.D ≤ (bos d i + 1) * d.H * d.D := by
    have : bos d i * d.H + hOf d i + 1 ≤ (bos d i + 1) * d.H := by rw [Nat.add_mul]; omega
    exact Nat.mul_le_mul_right _ this
  have e3 : (bos d i + 1) * d.H * d.D + (lenS d i - 1) * d.H * d.D = (bos d i + lenS d i) * d.H * d.D := by
    have : bos d i + 1 + (lenS d i - 1) = bos d i + lenS d i := by omega
    rw [← Nat.add_mul, ← Nat.add_mul, this]
  have e4 : (bos d i + lenS d i) * d.H * d.D ≤ d.B * d.T * d.H * d.D :=
    Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ h1)
  omega

/-- The largest element of the output row is inside the output: `eO + D ≤ nRows·H·D`. -/
theorem eO_last {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) :
    eO d i + d.D ≤ nRows d * d.H * d.D := by
  have h1 := outRow_lt hd hi
  have hh := (bOf_lt hd hi).2
  unfold eO
  have e1 : (outRow d i * d.H + hOf d i) * d.D + d.D = (outRow d i * d.H + hOf d i + 1) * d.D := by ring
  have e2 : outRow d i * d.H + hOf d i + 1 ≤ nRows d * d.H := by
    have := Nat.mul_le_mul_right d.H (show outRow d i + 1 ≤ nRows d by omega)
    rw [Nat.add_mul, Nat.one_mul] at this; omega
  rw [e1]
  exact Nat.mul_le_mul_right _ e2

/-- `bos·H < 2^63`. -/
theorem bosH_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < gx d * gy d) : bos d i * d.H < 2 ^ 63 := by
  have h1 := eX_last hd hT hi
  have h2 := BTHD_le hd hpre
  have hD := hd.D_pos
  have : bos d i * d.H ≤ eX d i := by
    unfold eX
    calc bos d i * d.H ≤ bos d i * d.H + hOf d i := Nat.le_add_right _ _
      _ ≤ (bos d i * d.H + hOf d i) * d.D := Nat.le_mul_of_pos_right _ hD
  omega

/-- `eX < 2^63`, `(lenS − 1)·H·D < 2^63`. -/
theorem eX_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < gx d * gy d) :
    eX d i < 2 ^ 63 ∧ (lenS d i - 1) * (d.H * d.D) < 2 ^ 63 := by
  have h1 := eX_last hd hT hi
  have h2 := BTHD_le hd hpre
  have hD := hd.D_pos
  omega

/-- `outRow·H < 2^63`, `eO < 2^63`. -/
theorem eO_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < gx d * gy d) : outRow d i * d.H < 2 ^ 63 ∧ eO d i < 2 ^ 63 := by
  have h1 := eO_last hd hi
  have h2 := nRHD_le hd hpre
  have hD := hd.D_pos
  have : outRow d i * d.H ≤ eO d i := by
    unfold eO
    calc outRow d i * d.H ≤ outRow d i * d.H + hOf d i := Nat.le_add_right _ _
      _ ≤ (outRow d i * d.H + hOf d i) * d.D := Nat.le_mul_of_pos_right _ hD
  omega

/-- `lenS < 2^31` (at most `T ≤ 2^22`). -/
theorem lenS_le {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) : lenS d i ≤ 2 ^ 22 := by
  have h1 := bos_len_le hd hT hi
  cases hv : d.varlen
  · simp only [lenS, hv, Bool.false_eq_true, ite_false]; exact hd.T_le
  · have hB := hd.varlen_B hv
    rw [hB, Nat.one_mul] at h1
    have := hd.T_le
    omega

/-- The constant `n` in `int64` for `n < 2^63`. -/
theorem i64_ofNat {n : Nat} (h : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = (n : Int) :=
  IntV.I64.toInt_ofNat h

/-- The rows `i_t·BT + arange(BT)` (`i_t` of value `chk`): lane `r` has value `chk·BT + r`. -/
theorem o_t_toInt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) {P : Platform} {it : I64} (hit : it.toInt = (chk d i : Int))
    {c : Tensor [d.BT - 0] I64}
    (hc : ∀ r, ConvRel P .i32 .i64 ((tl.arange 0 d.BT).get r) (c.get r)) (r : Fin d.BT) :
    (((bc (it * (IntV.ofNat d.BT : I64)) + c : Tensor [d.BT - 0] I64)).get1 r).toInt =
      ((rho d i r : Nat) : Int) := by
  have hc' := Tile2.arange64_toInt hc (by have := bt_le hd; omega) r
  have h1 := chk_lt hd hT hi
  have h2 := lenS_le hd hT hi
  have hb := bt_le hd
  have hmul : (it * (IntV.ofNat d.BT : I64)).toInt = ((chk d i * d.BT : Nat) : Int) :=
    IntV.I64.mul_nat hit (i64_ofNat (by omega)) (by omega)
  rw [Tensor.get1_add]
  exact IntV.I64.add_nat (x := chk d i * d.BT) hmul hc' (by have := r.isLt; omega)

/-- The columns `i_d·BD + arange(BD)` (`i_d` of value `iD`): lane `l` has value `iD·BD + l`. -/
theorem o_d_toInt {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) {P : Platform}
    {idv : I64} (hidv : idv.toInt = (iD d i : Int)) {c : Tensor [d.BD - 0] I64}
    (hc : ∀ r, ConvRel P .i32 .i64 ((tl.arange 0 d.BD).get r) (c.get r)) (l : Fin d.BD) :
    (((bc (idv * (IntV.ofNat d.BD : I64)) + c : Tensor [d.BD - 0] I64)).get1 l).toInt =
      ((kap d i l : Nat) : Int) := by
  have hc' := Tile2.arange64_toInt hc (by have := hd.bd_le; omega) l
  have h1 := (rG_lt hd hi).2
  have h2 := nD_le hd
  have hb := hd.bd_le
  have h3 : iD d i * d.BD + d.BD ≤ 2 ^ 13 * 128 + 128 := by
    have := Nat.mul_le_mul h1.le hb
    have := Nat.mul_le_mul_right 128 h2
    nlinarith
  have hmul : (idv * (IntV.ofNat d.BD : I64)).toInt = ((iD d i * d.BD : Nat) : Int) :=
    IntV.I64.mul_nat hidv (i64_ofNat (by omega)) (by omega)
  rw [Tensor.get1_add]
  exact IntV.I64.add_nat (x := iD d i * d.BD) hmul hc' (by have := l.isLt; omega)

/-- A pointer advanced by an integer of value `k` is the buffer's element `k`. -/
theorem ptr_add_eq {dt : DType} (b : Buf dt) {sg : Bool} {w : Nat} {k : IntV sg w} {n : Nat}
    (h : k.toInt = (n : Int)) : b.ptr 0 + k = b.ptr (n : Int) := by
  rw [Ptr.add_int, h, Buf.ptr_offset, zero_add]

end Claims.MeanPoolingFwd
