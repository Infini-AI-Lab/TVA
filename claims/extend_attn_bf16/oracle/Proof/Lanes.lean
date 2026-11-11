import Proof.Scalars

/-!
# Lane values of the offsets

Generic lemmas for the offsets the tiles are built from: the value of
`((a + p) + x) * s0 + c + y` at `qo_indptr`'s type, and a pointer plus an offset.
-/

namespace Claims.ExtendAttn

open TVA

/-- A buffer's base pointer plus an offset of natural value `n` is element `n`. -/
theorem ptr0_add {d : DType} (b : Buf d) {s : Shape} {sg : Bool} {w : Nat}
    (offs : Tensor s (IntV sg w)) (i : Idx s) {n : Nat} (h : (offs.get i).toInt = n) :
    (b.ptr 0 + offs).get i = b.ptr (n : Int) := by
  change (b.ptr 0).offset (offs.get i).toInt = _
  rw [h, Buf.ptr_offset]; simp

/-- The value of the row-major offset `((a + p) + x) * s0 + c + y` at `qo_indptr`'s type, when no
step wraps. -/
theorem qoff_val (qo32 : Bool) {a p : QI qo32} {x s0 c y : I32} {A P X S0 C Y : Nat}
    (ha : a.toInt = A) (hp : p.toInt = P) (hx : x.toInt = X) (hs : s0.toInt = S0)
    (hc : c.toInt = C) (hy : y.toInt = Y) (h1 : A + P + X < 2 ^ (qoW qo32 - 1))
    (h2 : (A + P + X) * S0 + C + Y < 2 ^ (qoW qo32 - 1)) :
    (((a + p + castQ qo32 x) * castQ qo32 s0 + castQ qo32 c + castQ qo32 y : QI qo32)).toInt =
      (((A + P + X) * S0 + C + Y : Nat) : Int) := by
  have hx' : (castQ qo32 x).toInt = X := by rw [castQ_toInt, hx]
  have hs' : (castQ qo32 s0).toInt = S0 := by rw [castQ_toInt, hs]
  have hc' : (castQ qo32 c).toInt = C := by rw [castQ_toInt, hc]
  have hy' : (castQ qo32 y).toInt = Y := by rw [castQ_toInt, hy]
  have e1 := IntV.add_nat ha hp (by omega)
  have e2 := IntV.add_nat e1 hx' h1
  have e3 := IntV.mul_nat e2 hs' (by omega)
  have e4 := IntV.add_nat e3 hc' (by omega)
  exact IntV.add_nat e4 hy' h2

/-- The scalar `.item` of a promoted scalar is the promoted value. -/
theorem toQv_scalar_item (qo32 : Bool) (x : I32) :
    (toQv qo32 (Tensor.scalar x)).item = castQ qo32 x := by cases qo32 <;> rfl

/-- Entry `(i, j)` of the query-width cast of a column broadcast is the cast of entry `i` of the source. -/
theorem toQv_col_get (qo32 : Bool) {m : Nat} (n : Nat) (x : Tensor [m] I32) (i : Fin m) (j : Fin n) :
    (toQv qo32 (x.col n)).get2 i j = castQ qo32 (x.get1 i) := by
  cases qo32 <;> rfl

/-- Entry `(i, j)` of the query-width cast of a row broadcast is the cast of entry `j` of the source. -/
theorem toQv_row_get (qo32 : Bool) {n : Nat} (m : Nat) (x : Tensor [n] I32) (i : Fin m) (j : Fin n) :
    (toQv qo32 (x.row m)).get2 i j = castQ qo32 (x.get1 j) := by
  cases qo32 <;> rfl


/-- An element index `t * s0 + g * s1 + e` of one of the tensors read or written through
`qo_indptr`'s type fits that type (with an `int32` `qo_indptr` the claim bounds the last element
by `2^31`; with an `int64` one the strides and sizes do). -/
theorem idx_fits {d : Inst} (hd : d.ok) {t g e T' G s0 s1 : Nat} (ht : t < T') (hg : g < G)
    (he : e < d.D) (hT : T' ≤ 2 ^ 22) (hG : G ≤ 2 ^ 16) (hs0 : s0 < 2 ^ 31) (hs1 : s1 < 2 ^ 31)
    (hb : d.cm = true → (T' - 1) * s0 + (G - 1) * s1 + d.D ≤ 2 ^ 31) :
    t * s0 + g * s1 + e < 2 ^ (qoW d.cm - 1) := by
  have h1 : t * s0 ≤ (T' - 1) * s0 := Nat.mul_le_mul_right _ (by omega)
  have h2 : g * s1 ≤ (G - 1) * s1 := Nat.mul_le_mul_right _ (by omega)
  have hDle := hd.D_le
  have hbd := hd.bd_le
  cases hcm : d.cm
  · simp only [qoW]
    have h3 : t * s0 ≤ 2 ^ 22 * 2 ^ 31 := Nat.mul_le_mul (by omega) (by omega)
    have h4 : g * s1 ≤ 2 ^ 16 * 2 ^ 31 := Nat.mul_le_mul (by omega) (by omega)
    norm_num at h3 h4 ⊢
    omega
  · have := hb hcm
    simp only [qoW]
    norm_num
    omega

end Claims.ExtendAttn
