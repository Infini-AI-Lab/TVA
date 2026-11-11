import Proof.Meta

/-!
# Helpers shared by the program's walks

Integer-value facts about the kernel's constants, element-wise reads of the tensors it builds
(`tl.arange`, `tl.full`, `+`, `*`, `<`, `cast`, pointer arithmetic), and the bounds on `BT`, `BD`
that the two tails (`Proof/TailNE.lean`, `Proof/TailI.lean`) and the offset line (`Proof/Ocs.lean`)
all use.
-/

namespace Claims.RotaryVarlen

open TVA

/-- `BT` is positive. -/
theorem bt_pos {d : Inst} (hd : d.ok) : 0 < d.BT := by
  obtain ⟨e, he⟩ := hd.bt_pow2
  rw [he]
  exact Nat.two_pow_pos e

/-- `BT` is at most `2^20`. -/
theorem bt_le {d : Inst} (hd : d.ok) : d.BT ≤ 2 ^ 20 := by
  have h1 : 1 ≤ d.BD := by
    obtain ⟨e, he⟩ := hd.bd_pow2
    rw [he]
    exact Nat.one_le_two_pow
  calc d.BT = d.BT * 1 := (Nat.mul_one _).symm
    _ ≤ d.BT * d.BD := Nat.mul_le_mul_left _ h1
    _ ≤ 2 ^ 20 := hd.block_le

/-- `BD` is at most `2^20`. -/
theorem bd_le {d : Inst} (hd : d.ok) : d.BD ≤ 2 ^ 20 := by
  have h1 : 1 ≤ d.BT := bt_pos hd
  calc d.BD = 1 * d.BD := (Nat.one_mul _).symm
    _ ≤ d.BT * d.BD := Nat.mul_le_mul_right _ h1
    _ ≤ 2 ^ 20 := hd.block_le

/-- A program's chunk index times `BT` is far below the `int64` range. -/
theorem chunkI_mul_BT_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size) : chunkI d init i * d.BT < 2 ^ 31 := by
  have hBTpos : 0 < d.BT := bt_pos hd
  by_cases hv : d.isVarlen = true
  · have hs := hpre.seqs hv
    have hg : g0 d = d.NT := by simp [g0, hv]
    have hax : ax0 d i < d.NT := by rw [← hg]; exact ax0_lt hi
    obtain ⟨hseqN, hchunk⟩ := hs.ci_ok (ax0 d i) hax
    rw [chunkI_true hv]
    have hmono : cuN d init (chSeq d init (ax0 d i) + 1) ≤ cuN d init d.N :=
      cuN_mono_le hs (by omega) le_rfl
    have hle : seqLen d init (chSeq d init (ax0 d i)) ≤ d.T := by
      have hlast := hs.cu_last
      simp only [seqLen]
      omega
    have hchunk2 : chChunk d init (ax0 d i) * d.BT < d.T := lt_of_lt_of_le hchunk hle
    have ht := hd.t_lt
    omega
  · have hv2 : d.isVarlen = false := by simpa using hv
    rw [chunkI_false hv2]
    have hg : g0 d = (d.T + d.BT - 1) / d.BT := by simp [g0, hv2]
    have hax : ax0 d i < (d.T + d.BT - 1) / d.BT := by
      have := ax0_lt hi
      rwa [hg] at this
    have hmul : ax0 d i * d.BT < ((d.T + d.BT - 1) / d.BT) * d.BT :=
      Nat.mul_lt_mul_of_pos_right hax hBTpos
    have hle := Nat.div_mul_le_self (d.T + d.BT - 1) d.BT
    have hTB : d.T + d.BT < 2 ^ 31 := by
      have := hd.t_ctx
      have := bt_le hd
      omega
    omega

/-- An integer in `[-2^63, 2^63)` fits `int64`. -/
theorem sfits64_of_bounds {x : Int} (h1 : -(2 ^ 63 : Int) ≤ x) (h2 : x < 2 ^ 63) :
    IntV.SFits 64 x := by
  have hp : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  simp only [IntV.SFits, hp]
  exact ⟨h1, h2⟩

/-- An integer in `[-2^31, 2^31)` fits `int32`. -/
theorem sfits32_of_bounds {x : Int} (h1 : -(2 ^ 31 : Int) ≤ x) (h2 : x < 2 ^ 31) :
    IntV.SFits 32 x := by
  have hp : (2 : Int) ^ (32 - 1) = 2 ^ 31 := by norm_num
  simp only [IntV.SFits, hp]
  exact ⟨h1, h2⟩

/-- Element `i` of `i_t * n + arange(0, n)` is `i_t * n` plus the lane index, in `int64`. -/
theorem get_split (n : Nat) (i_t : I64) (i : Idx [n]) :
    (i_t * (IntV.ofNat n : I64) + ((tl.arange 0 n).castInt : Tensor [n] I64)).get i
      = i_t * (IntV.ofNat n : I64) + (IntV.cast (IntV.ofInt (0 + (i.1.val : Int)) : I32) : I64) := by
  rcases i with ⟨j, u⟩
  cases u
  rfl

/-- The value of element `i` of `i_t * n + arange(0, n) + off`, when no step wraps. -/
theorem toInt_split (n : Nat) (i_t : I64) (i : Idx [n]) (off : I64)
    (hn : (n : Int) < 2 ^ 63) (hiv : (0 + (i.1.val : Int)) < 2 ^ 31)
    (h1 : IntV.SFits 64 (i_t.toInt * (n : Int)))
    (h2 : IntV.SFits 64 (i_t.toInt * (n : Int) + (0 + (i.1.val : Int))))
    (h3 : IntV.SFits 64 (i_t.toInt * (n : Int) + (0 + (i.1.val : Int)) + off.toInt)) :
    (((i_t * (IntV.ofNat n : I64) + ((tl.arange 0 n).castInt : Tensor [n] I64)) + off).get i).toInt
      = i_t.toInt * (n : Int) + (0 + (i.1.val : Int)) + off.toInt := by
  rw [Tensor.get_adds, get_split]
  have h1' : IntV.SFits 64 (i_t.toInt * (IntV.ofNat n : I64).toInt) := by
    rw [ofNat_toInt (w := 64) (n := n)  hn]; exact h1
  have hmul : (i_t * (IntV.ofNat n : I64)).toInt = i_t.toInt * (n : Int) := by
    rw [IntV.toInt_mul rfl rfl h1', ofNat_toInt (w := 64) (n := n)  hn]
  have hcast : (IntV.cast (IntV.ofInt (0 + (i.1.val : Int)) : I32) : I64).toInt = (0 + (i.1.val : Int)) := by
    rw [IntV.toInt_cast (w := 32) (w' := 64) _ (by norm_num)]
    refine IntV.I32.toInt_ofInt ⟨?_, hiv⟩
    have hv0 : (0 : Int) ≤ (i.1.val : Int) := by exact_mod_cast Nat.zero_le i.1.val
    linarith
  have h2' : IntV.SFits 64 ((i_t * (IntV.ofNat n : I64)).toInt +
      (IntV.cast (IntV.ofInt (0 + (i.1.val : Int)) : I32) : I64).toInt) := by
    rw [hmul, hcast]; exact h2
  have hA : (i_t * (IntV.ofNat n : I64) + (IntV.cast (IntV.ofInt (0 + (i.1.val : Int)) : I32) : I64)).toInt
      = i_t.toInt * (n : Int) + (0 + (i.1.val : Int)) := by
    rw [IntV.toInt_add rfl rfl h2', hmul, hcast]
  have h3' : IntV.SFits 64 ((i_t * (IntV.ofNat n : I64) + (IntV.cast (IntV.ofInt (0 + (i.1.val : Int)) : I32) : I64)).toInt + off.toInt) := by
    rw [hA]; exact h3
  rw [IntV.toInt_add rfl rfl h3', hA]

/-- An integer that fits `int32` fits `int64`. -/
theorem sfits64_of_sfits32 {x : Int} (h : IntV.SFits 32 x) : IntV.SFits 64 x := by
  obtain ⟨h1, h2⟩ := h
  refine sfits64_of_bounds ?_ ?_
  · have h32 : (2 : Int) ^ (32 - 1) = 2 ^ 31 := by norm_num
    rw [h32] at h1
    have hp : (2 : Int) ^ 31 ≤ 2 ^ 63 := by norm_num
    linarith
  · have h32 : (2 : Int) ^ (32 - 1) = 2 ^ 31 := by norm_num
    rw [h32] at h2
    have hp : (2 : Int) ^ 31 ≤ 2 ^ 63 := by norm_num
    linarith

/-- The value of `IntV.div?` when the divisor is positive and the dividend nonnegative. -/
theorem divVal {a b q : I32} (ha : 0 ≤ a.toInt) (hb : 0 < b.toInt)
    (h : a.div? b = some q) : q.toInt = a.toInt / b.toInt := by
  obtain ⟨q', hq', hv⟩ := IntV.div?_of_nonneg ha hb
  rw [hq'] at h
  have he : q = q' := (Option.some.inj h).symm
  rw [he]; exact hv

/-- The value of `IntV.rem?` when the divisor is positive and the dividend nonnegative. -/
theorem remVal {a b q : I32} (ha : 0 ≤ a.toInt) (hb : 0 < b.toInt)
    (h : a.rem? b = some q) : q.toInt = a.toInt % b.toInt := by
  obtain ⟨q', hq', hv⟩ := IntV.rem?_of_nonneg ha hb
  rw [hq'] at h
  have he : q = q' := (Option.some.inj h).symm
  rw [he]; exact hv

/-- The `int32` constant `2` has value `2`. -/
theorem ofNat_two_I32 : (IntV.ofNat 2 : I32).toInt = 2 := by
  rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num

/-- The `int32` constant `2` is positive. -/
theorem ofNat_two_pos : 0 < (IntV.ofNat 2 : I32).toInt := by
  rw [ofNat_two_I32]; norm_num

/-- The `int32` constant `1` has value `1`. -/
theorem ofNat_one_I32 : (IntV.ofNat 1 : I32).toInt = 1 := by
  rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num

/-- The `int64` constant `0` has value `0`. -/
theorem ofInt_zero_I64 : (IntV.ofInt (0 : Int) : I64).toInt = 0 := by
  refine IntV.toInt_ofInt (by norm_num) ?_
  simp only [IntV.SFits]
  norm_num

/-- The `int32` constant `0` has value `0`. -/
theorem ofInt_zero_I32 : (IntV.ofInt (0 : Int) : I32).toInt = 0 := by
  exact IntV.I32.toInt_ofInt (by norm_num)

/-- The `int64` constant `BT` has value `BT`. -/
theorem ofNat_bt_I64 {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.BT : I64).toInt = (d.BT : Int) := by
  refine ofNat_toInt  ?_
  have h1 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast bt_le hd
  have h2 : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  rw [h2]
  linarith [show (2 : Int) ^ 20 < 2 ^ 63 by norm_num]

/-- The `int64` constant `TR`, the length of the `cos`/`sin` tables, has value `TR`. -/
theorem ofNat_TR_I64 {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.TR : I64).toInt = (d.TR : Int) := by
  refine ofNat_toInt  ?_
  have h2 : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  rw [h2]
  have hR1 : 1 ≤ d.R := hd.r_pos
  have h1 : d.TR ≤ d.TR * d.R := Nat.le_mul_of_pos_right d.TR hR1
  have h2' : d.TR * d.R < 2 ^ 62 := hd.tr_lt
  have h3 : d.TR < 2 ^ 63 := by omega
  exact_mod_cast h3

/-- The `int32` constant `R` has value `R`. -/
theorem ofNat_R_I32 {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.R : I32).toInt = (d.R : Int) := by
  refine IntV.I32.toInt_ofNat ?_
  have h1 : d.R ≤ d.BD := by have := hd.r_bd; omega
  have h2 : d.BD ≤ 2 ^ 20 := bd_le hd
  omega

/-- Lane `i` of `tl.arange(0, BD)` is `i`. -/
theorem g_arange_bd (n : Nat) (i : Fin n) :
    (tl.arange 0 n).get1 i = IntV.ofInt (0 + (i.val : Int)) := rfl

/-- Lane `i` of `tl.arange(0, n)` is `i`. -/
theorem g_arange (n : Nat) (i : Fin n) :
    (tl.arange 0 n).get (i, ()) = IntV.ofInt (0 + (i.val : Int)) := rfl

/-- Every lane of a splat is the splatted value. -/
theorem g_full1 {α : Type} (n : Nat) (a : α) (i : Fin n) :
    (bc a : Tensor [n] α).get (i, ()) = a := rfl

/-- Every lane of a two-dimensional splat is the splatted value. -/
theorem g_full2 {α : Type} {m n : Nat} (a : α) (i : Fin m) (j : Fin n) :
    (bc a : Tensor [m, n] α).get (i, j, ()) = a := rfl

/-- A lane of a tensor plus a scalar, in `int`. -/
theorem g_sadd1 {n : Nat} {sg : Bool} {w : Nat} (a : IntV sg w)
    (x : Tensor [n] (IntV sg w)) (i : Fin n) :
    (a + x).get (i, ()) = a + x.get (i, ()) := rfl

/-- A lane of a sum of two `int32` tensors is the sum of the lanes. -/
theorem g_add1 {n : Nat} (x y : Tensor [n] I32) (i : Fin n) :
    (x + y).get (i, ()) = x.get (i, ()) + y.get (i, ()) := rfl

/-- A lane of an `int32` tensor plus a scalar is the sum. -/
theorem g_adds1 {n : Nat} (x : Tensor [n] I32) (a : I32) (i : Fin n) :
    (x + a).get (i, ()) = x.get (i, ()) + a := rfl

/-- A lane of an `int32` tensor times a scalar is the product. -/
theorem g_muls1 {n : Nat} (x : Tensor [n] I32) (a : I32) (i : Fin n) :
    (x * a).get (i, ()) = x.get (i, ()) * a := rfl

/-- A lane of a difference of two `int32` tensors is the difference of the lanes. -/
theorem g_sub1 {n : Nat} (x y : Tensor [n] I32) (i : Fin n) :
    (x - y).get (i, ()) = x.get (i, ()) - y.get (i, ()) := rfl

/-- A lane of `x < a` is the comparison of the lane with the scalar. -/
theorem g_ltS1 {n : Nat} (x : Tensor [n] I32) (a : I32) (i : Fin n) :
    (x.ltS a).get (i, ()) = IntV.lt (x.get (i, ())) a := rfl

/-- A lane of `not x` is the negation of the lane. -/
theorem g_not1 {n : Nat} (x : Tensor [n] Bool) (i : Fin n) :
    x.not.get (i, ()) = !x.get (i, ()) := rfl

/-- A lane of `x and y` is the conjunction of the lanes. -/
theorem g_and1 {n : Nat} (x y : Tensor [n] Bool) (i : Fin n) :
    (x.and y).get (i, ()) = (x.get (i, ()) && y.get (i, ())) := rfl

/-- A lane of `x == y` is the comparison of the lanes. -/
theorem g_eq1 {n : Nat} (x y : Tensor [n] I32) (i : Fin n) :
    (x.eq y).get (i, ()) = IntV.eq (x.get (i, ())) (y.get (i, ())) := rfl

/-- A lane of a cast is the cast of the lane. -/
theorem g_cast1 {n : Nat} {sg sg' : Bool} {w w' : Nat}
    (x : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x.castInt : Tensor [n] (IntV sg' w')).get (i, ()) = IntV.cast (x.get (i, ())) := rfl

/-- A lane of a row-broadcast tensor is the lane of its column. -/
theorem g_row1 {α : Type} {m n : Nat} (x : Tensor [n] α) (i : Fin m) (j : Fin n) :
    (x.row m).get (i, j, ()) = x.get (j, ()) := rfl

/-- A lane of a column-broadcast tensor is the lane of its row. -/
theorem g_col1 {α : Type} {m n : Nat} (x : Tensor [m] α) (i : Fin m) (j : Fin n) :
    (x.col n).get (i, j, ()) = x.get (i, ()) := rfl

/-- A lane of the conjunction of two tiles is the conjunction of the lanes. -/
theorem g2_and {m n : Nat} (x y : Tensor [m, n] Bool) (i : Fin m) (j : Fin n) :
    (x.and y).get (i, j, ()) = (x.get (i, j, ()) && y.get (i, j, ())) := rfl

/-- A lane of a tile compared with a scalar is the comparison of the lane. -/
theorem g2_ltS {m n : Nat} (x : Tensor [m, n] I32) (a : I32) (i : Fin m) (j : Fin n) :
    (x.ltS a).get (i, j, ()) = IntV.lt (x.get (i, j, ())) a := rfl

/-- A lane of a tile times a scalar is the product. -/
theorem g2_muls {m n : Nat} (x : Tensor [m, n] I64) (a : I64) (i : Fin m) (j : Fin n) :
    (x * a).get (i, j, ()) = x.get (i, j, ()) * a := rfl

/-- A lane of a sum of two tiles is the sum of the lanes. -/
theorem g2_add {m n : Nat} (x y : Tensor [m, n] I64) (i : Fin m) (j : Fin n) :
    (x + y).get (i, j, ()) = x.get (i, j, ()) + y.get (i, j, ()) := rfl

/-- A lane of a cast tile is the cast of the lane. -/
theorem g2_cast {m n : Nat} {sg sg' : Bool} {w w' : Nat}
    (x : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    (x.castInt : Tensor [m, n] (IntV sg' w')).get (i, j, ()) = IntV.cast (x.get (i, j, ())) := rfl

/-- A lane of `tl.where` picks the lane of one of the two branches. -/
theorem g2_where {α : Type} {m n : Nat} (c : Tensor [m, n] Bool)
    (x y : Tensor [m, n] α) (i : Fin m) (j : Fin n) :
    (tl.where c x y).get (i, j, ()) =
      (if c.get (i, j, ()) then x.get (i, j, ()) else y.get (i, j, ())) := rfl

/-- A lane of a base pointer plus an offset tile is the pointer plus the offset. -/
theorem g2_ptrAdd {m n : Nat} {sg : Bool} {w : Nat} (p : Ptr .bf16)
    (offs : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    (p + offs).get (i, j, ()) = p.offset (offs.get (i, j, ())).toInt := rfl

/-- A lane of a pointer tile plus an offset tile is the sum of the lanes. -/
theorem g2_ptrAddT {m n : Nat} {sg : Bool} {w : Nat}
    (ps : Tensor [m, n] (Ptr .bf16)) (offs : Tensor [m, n] (IntV sg w))
    (i : Fin m) (j : Fin n) :
    (ps + offs).get (i, j, ()) = (ps.get (i, j, ())).offset (offs.get (i, j, ())).toInt := rfl

/-- A lane of a base pointer plus an offset vector is the pointer plus the offset. -/
theorem g_ptrAdd {n : Nat} {sg : Bool} {w : Nat} (p : Ptr .bf16)
    (offs : Tensor [n] (IntV sg w)) (i : Fin n) :
    (p + offs).get (i, ()) = p.offset (offs.get (i, ())).toInt := rfl

/-- `get_split` for an index in `Fin n`. -/
theorem get_split_fin (n : Nat) (i_t : I64) (i : Fin n) :
    (i_t * (IntV.ofNat n : I64) + ((tl.arange 0 n).castInt : Tensor [n] I64)).get (i, ())
      = i_t * (IntV.ofNat n : I64) + (IntV.cast (IntV.ofInt (0 + (i.val : Int)) : I32) : I64) := by
  rw [g_sadd1, g_cast1, g_arange]

/-- `IntV.eq` is the equality of the values. -/
theorem IntV.eq_decide {s : Bool} {w : Nat} (a b : IntV s w) :
    a.eq b = decide (a.toInt = b.toInt) := by
  by_cases h : a.toInt = b.toInt
  · rw [decide_eq_true h]
    exact (IntV.cmpKey_eq a b).mpr h
  · rw [decide_eq_false h]
    cases he : a.eq b
    · rfl
    · exact absurd ((IntV.cmpKey_eq a b).mp he) h

/-- The value of `tl.arange 0 BD` on the channels. -/
theorem oD_toInt {d : Inst} (hd : d.ok) (c : Fin d.BD) :
    ((tl.arange 0 d.BD).get (c, ())).toInt = (c.val : Int) := by
  have h0 : (0 : Int) ≤ (c.val : Int) := by exact_mod_cast Nat.zero_le c.val
  have h1 : (c.val : Int) < (d.BD : Int) := by exact_mod_cast c.isLt
  have h2 : (d.BD : Int) ≤ 2 ^ 20 := by exact_mod_cast bd_le hd
  have hv : (IntV.ofInt (0 + (c.val : Int)) : I32).toInt = 0 + (c.val : Int) :=
    IntV.I32.toInt_ofInt ⟨by omega, by omega⟩
  rw [g_arange, hv]
  omega

/-- The `o_t` value of every token lane is `locI`. -/
theorem oT_toInt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size) {i_t : I64} (hit : i_t.toInt = (chunkI d init i : Int))
    (l : Fin d.BT) :
    ((i_t * (IntV.ofNat d.BT : I64) + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)).get (l, ())).toInt
      = (locI d init i l.val : Int) := by
  have hBTle : d.BT ≤ 2 ^ 20 := bt_le hd
  have hcbt : chunkI d init i * d.BT < 2 ^ 31 := chunkI_mul_BT_lt hd hpre hi
  have hcbtI : (chunkI d init i : Int) * (d.BT : Int) < 2 ^ 31 := by
    have h2 : ((chunkI d init i * d.BT : Nat) : Int) < (2 : Int) ^ 31 := by exact_mod_cast hcbt
    push_cast at h2 ⊢
    linarith
  have hc0 : (0 : Int) ≤ (chunkI d init i : Int) := by exact_mod_cast Nat.zero_le _
  have hl31 : (l.val : Int) < 2 ^ 31 := by
    have h1 : (l.val : Int) < (d.BT : Int) := by exact_mod_cast l.isLt
    have h2 : (d.BT : Int) ≤ 2 ^ 20 := by exact_mod_cast hBTle
    linarith
  have h1 : IntV.SFits 64 (i_t.toInt * (d.BT : Int)) := by
    rw [hit]; refine sfits64_of_bounds ?_ ?_
    · nlinarith
    · linarith
  have h2 : IntV.SFits 64 (i_t.toInt * (d.BT : Int) + (0 + (l.val : Int))) := by
    rw [hit]; refine sfits64_of_bounds ?_ ?_
    · nlinarith
    · linarith
  rw [get_split_fin]
  have hbt64 : (IntV.ofNat d.BT : I64).toInt = (d.BT : Int) := ofNat_bt_I64 hd
  have h1' : IntV.SFits 64 (i_t.toInt * (IntV.ofNat d.BT : I64).toInt) := by
    rw [hbt64]; exact h1
  have hmul : (i_t * (IntV.ofNat d.BT : I64)).toInt = i_t.toInt * (d.BT : Int) := by
    rw [IntV.toInt_mul rfl rfl h1', hbt64]
  have hcast : (IntV.cast (IntV.ofInt (0 + (l.val : Int)) : I32) : I64).toInt = (0 + (l.val : Int)) := by
    rw [IntV.toInt_cast (w := 32) (w' := 64) _ (by norm_num)]
    refine IntV.I32.toInt_ofInt ⟨?_, ?_⟩
    · have h0 : (0 : Int) ≤ (l.val : Int) := by exact_mod_cast Nat.zero_le l.val
      linarith
    · linarith
  have h2' : IntV.SFits 64 ((i_t * (IntV.ofNat d.BT : I64)).toInt +
      (IntV.cast (IntV.ofInt (0 + (l.val : Int)) : I32) : I64).toInt) := by
    rw [hmul, hcast]; exact h2
  rw [IntV.toInt_add rfl rfl h2', hmul, hcast]
  rw [hit]
  simp only [locI]
  push_cast
  ring

/-- A lane of a column-broadcast vector is the vector's lane at the row. -/
theorem col_get' {α : Type} {m n : ℕ} (x : Tensor [m] α) (kk : Idx [m, n]) :
    (x.col n).get kk = x.get (kk.1, ()) := by
  rw [show kk = (kk.1, kk.2.1, ()) from rfl]
  exact Tensor.get_col x kk.1 kk.2.1

/-- A lane of a row-broadcast vector is the vector's lane at the column. -/
theorem row_get' {α : Type} {m n : ℕ} (x : Tensor [n] α) (kk : Idx [m, n]) :
    (x.row m).get kk = x.get (kk.2.1, ()) := by
  rw [show kk = (kk.1, kk.2.1, ()) from rfl]
  exact Tensor.get_row x kk.1 kk.2.1

/-- The `int64` constant `H` has value `H`. -/
theorem ofNat_H_I64 {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.H : I64).toInt = (d.H : Int) := by
  refine ofNat_toInt  ?_
  have h2 : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  rw [h2]
  have h1 : (d.H : Int) < 2 ^ 16 := by exact_mod_cast hd.grid_h
  have h3 : (2 : Int) ^ 16 < 2 ^ 63 := by norm_num
  linarith

/-- The `int64` constant `D` has value `D`. -/
theorem ofNat_D_I64 {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.D : I64).toInt = (d.D : Int) := by
  refine ofNat_toInt  ?_
  have h2 : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  rw [h2]
  have h1 : (d.D : Int) ≤ 256 := by exact_mod_cast hd.d_le
  have h3 : (2 : Int) ^ 8 < 2 ^ 63 := by norm_num
  linarith

/-- Two pointers of one dtype with equal addresses are equal. -/
theorem ptr_eq_of_addr {d : DType} {p q : Ptr d} (h : p.addr = q.addr) : p = q := by
  cases p; cases q; simp_all

end Claims.RotaryVarlen
