import TVA.Proof

/-!
# `int32` values of scalars, as naturals

The scalar `int32` arithmetic of the program ids: a value whose `toInt` is a natural below `2^31`
is `IntV.ofNat` of it, and `//`, `%` of such values are the natural `/`, `%`.
-/

namespace TVA

namespace IntV

/-- An `int32` whose value is the natural `n < 2^31` is `IntV.ofNat n`. -/
theorem I32.eq_ofNat {x : I32} {n : Nat} (h : x.toInt = n) (hn : n < 2 ^ 31) :
    x = IntV.ofNat n :=
  IntV.ext_toInt (by rw [h, I32.toInt_ofNat hn])

/-- The result `r` of `a // b` of two naturals below `2^31`. -/
theorem I32.item_idiv {a b : Nat} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) {r : Tensor [] I32}
    (h : r.item.toInt = (IntV.ofNat a : I32).toInt / (IntV.ofNat b : I32).toInt) :
    r.item = IntV.ofNat (a / b) := by
  rw [I32.toInt_ofNat ha, I32.toInt_ofNat hb] at h
  refine I32.eq_ofNat (by rw [h]; exact (Int.natCast_div a b).symm) ?_
  exact lt_of_le_of_lt (Nat.div_le_self a b) ha

/-- The result `r` of `a % b` of two naturals below `2^31`, `b > 0`. -/
theorem I32.item_irem {a b : Nat} (ha : a < 2 ^ 31) (hb0 : 0 < b) (hb : b < 2 ^ 31)
    {r : Tensor [] I32}
    (h : r.item.toInt = (IntV.ofNat a : I32).toInt % (IntV.ofNat b : I32).toInt) :
    r.item = IntV.ofNat (a % b) := by
  rw [I32.toInt_ofNat ha, I32.toInt_ofNat hb] at h
  refine I32.eq_ofNat (by rw [h]; exact (Int.natCast_mod a b).symm) ?_
  exact lt_of_lt_of_le (Nat.mod_lt a hb0) hb.le

/-- `min` of two naturals below `2^31` as scalars. -/
theorem I32.item_iminimum {x y : I32} {a b : Nat} (hx : x.toInt = a) (hy : y.toInt = b)
    (ha : a < 2 ^ 31) (_hb : b < 2 ^ 31) :
    (tl.iminimum (Tensor.scalar x) (Tensor.scalar y)).item = IntV.ofNat (Min.min a b) := by
  refine I32.eq_ofNat ?_ (lt_of_le_of_lt (Nat.min_le_left a b) ha)
  have : (tl.iminimum (Tensor.scalar x) (Tensor.scalar y)).item = x.min y := rfl
  rw [this, IntV.toInt_min, hx, hy]
  omega

/-- `min (a - b) 8`, the group size, as a scalar. -/
theorem I32.item_iminimum_sub {a b c : Nat} (hba : b ≤ a) (ha : a < 2 ^ 31) (hc : c < 2 ^ 31) :
    (tl.iminimum (Tensor.scalar (IntV.ofNat a - IntV.ofNat b : I32))
      (Tensor.scalar (IntV.ofNat c : I32))).item = IntV.ofNat (Min.min (a - b) c) :=
  I32.item_iminimum (IntV.I32.sub_nat (I32.toInt_ofNat ha) (I32.toInt_ofNat (by omega)) hba ha)
    (I32.toInt_ofNat hc) (by omega) hc

/-- `min c (a - b)`, with the constant first (`min(BLOCK_M, lora_m_size - cta_m_offset)`). -/
theorem I32.item_iminimum_sub' {a b c : Nat} (hba : b ≤ a) (ha : a < 2 ^ 31) (hc : c < 2 ^ 31) :
    (tl.iminimum (Tensor.scalar (IntV.ofNat c : I32))
      (Tensor.scalar (IntV.ofNat a - IntV.ofNat b : I32))).item = IntV.ofNat (Min.min c (a - b)) :=
  I32.item_iminimum (I32.toInt_ofNat hc)
    (IntV.I32.sub_nat (I32.toInt_ofNat ha) (I32.toInt_ofNat (by omega)) hba ha) hc (by omega)

end IntV

end TVA
