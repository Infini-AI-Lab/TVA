import Proof.Defs

/-!
# Per-lane values

The kernel works lane by lane. This file records what a lane's value means: the `int32` value a
lane of a loaded block of ids or of a gathered map has, when a lane's equality test against the
program's expert holds, and when a lane's `geZero` test holds. It also computes the loop's trip
count and the trip index arithmetic.
-/

namespace Claims.CountExpertTokens

open TVA

/-! ## `entryInt`, `m1`, `geZero`, `eqCurr`, lane by lane -/

/-- An `int32` entry reads as its value. -/
theorem entryInt_i32 (x : I32) : entryInt .i32 x = x.toInt := rfl

/-- An `int64` entry reads as its value. -/
theorem entryInt_i64 (x : I64) : entryInt .i64 x = x.toInt := rfl

/-- The `-1` of `m1` reads as the integer `-1`. -/
theorem entryInt_m1 {d : DType} (hd : d = .i32 ∨ d = .i64) : entryInt d (m1 d) = -1 := by
  rcases hd with rfl | rfl
  · show (m1 .i32).toInt = -1
    exact IntV.I32.toInt_ofInt (by norm_num)
  · show (m1 .i64).toInt = -1
    refine IntV.toInt_ofInt (by norm_num) ?_
    simp only [IntV.SFits]
    norm_num

/-- The `int32` zero has value `0`. -/
theorem toInt_zero_i32 : (0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)

/-- The `int64` zero has value `0`. -/
theorem toInt_zero_i64 : (0 : I64).toInt = 0 := by
  exact IntV.toInt_ofNat (by norm_num)

/-! ## Lane lemmas in `get1` form

`Idx` is defined by recursion on the shape, so terms such as `(i, ())` do not rewrite under the
`Tensor.get_*` lemmas. Stated over `get1` with an explicit `Fin`, they hold by `rfl`; the proofs
below can then rewrite with them syntactically. -/

/-- Every lane of a broadcast scalar is the scalar. -/
theorem get1_full {n : Nat} {α : Type} (a : α) (i : Fin n) :
    (Tensor.full a : Tensor [n] α).get1 i = a := rfl

/-- Lane `i` of `tl.where` picks `x` or `y` by the lane of the condition. -/
theorem get1_where {n : Nat} {α : Type} (c : Tensor [n] Bool) (x y : Tensor [n] α) (i : Fin n) :
    (Tensor.where_ c x y).get1 i = if c.get1 i then x.get1 i else y.get1 i := rfl

/-- Lane `i` of `ptrs + offs` moves the lane of `ptrs` by the lane of `offs`. -/
theorem get1_ptrAddT {n : Nat} {d : DType} (p : Tensor [n] (Ptr d)) (o : Tensor [n] I32)
    (i : Fin n) :
    (p + o).get1 i = (p.get1 i).offset (o.get1 i).toInt := rfl

/-- Lane `i` of `p + offs` moves `p` by the lane of `offs`. -/
theorem get1_ptrAdd {n : Nat} {d : DType} (p : Ptr d) (o : Tensor [n] I32) (i : Fin n) :
    (p + o).get1 i = p.offset (o.get1 i).toInt := rfl

/-- `get1` is `get` at `(i, ())`. -/
theorem get1_get {n : Nat} {α : Type} (x : Tensor [n] α) (i : Fin n) :
    x.get1 i = x.get (i, ()) := rfl

/-- `a.lt 0` reads as the decision of `a.toInt < 0`. -/
theorem lt_zero_decide {w : Nat} (a : IntV true w) (h0 : (0 : IntV true w).toInt = 0) :
    (a.lt 0) = decide (a.toInt < 0) := by
  rw [IntV.lt_eq]
  congr 1
  rw [h0]

/-- A lane of `geZero` holds exactly when the lane's entry is not negative. -/
theorem geZero_true {B : Nat} {d : DType} (hd : d = .i32 ∨ d = .i64) (x : Tensor [B] d.Val)
    (i : Fin B) :
    ((geZero d x).get1 i = true) ↔ 0 ≤ entryInt d (x.get1 i) := by
  rcases hd with rfl | rfl
  · show ((!(x.get1 i).lt 0) = true) ↔ 0 ≤ (x.get1 i).toInt
    rw [lt_zero_decide _ toInt_zero_i32, ← decide_not, decide_eq_true_iff]
    omega
  · show ((!(x.get1 i).lt 0) = true) ↔ 0 ≤ (x.get1 i).toInt
    rw [lt_zero_decide _ toInt_zero_i64, ← decide_not, decide_eq_true_iff]
    omega

/-- A lane of `geZero` fails exactly when the lane's entry is negative. -/
theorem geZero_false {B : Nat} {d : DType} (hd : d = .i32 ∨ d = .i64) (x : Tensor [B] d.Val)
    (i : Fin B) :
    ((geZero d x).get1 i = false) ↔ ¬(0 ≤ entryInt d (x.get1 i)) := by
  have h := geZero_true hd x i
  constructor
  · intro hf hc
    have ht : (geZero d x).get1 i = true := h.mpr hc
    rw [hf] at ht
    exact Bool.noConfusion ht
  · intro hc
    cases hb : (geZero d x).get1 i
    · rfl
    · exact absurd (h.mp hb) hc

/-- A lane of `eqCurr` holds exactly when the lane's entry is the program's expert `e`. -/
theorem eqCurr_true {B : Nat} {d : DType} (hd : d = .i32 ∨ d = .i64) (x : Tensor [B] d.Val)
    (curr : I32) (e : Int) (hc : curr.toInt = e) (i : Fin B) :
    ((eqCurr d x curr).get1 i = true) ↔ entryInt d (x.get1 i) = e := by
  rcases hd with rfl | rfl
  · simp only [eqCurr, Tensor.get1_eq, get1_full, entryInt_i32]
    rw [IntV.eq_eq, decide_eq_true_iff, hc]
  · have hc' : (curr.cast : I64).toInt = e := by
      rw [IntV.toInt_cast curr (by norm_num), hc]
    simp only [eqCurr, Tensor.get1_eq, get1_full, entryInt_i64]
    rw [IntV.eq_eq, decide_eq_true_iff, hc']

/-- A lane of a gathered map's pointer block is the map's entry of the lane's index. -/
theorem mapPtrs_get {B : Nat} {d : DType} (hd : d = .i32 ∨ d = .i64) (em : Ptr .i32)
    (v : Tensor [B] d.Val) (i : Fin B) :
    (mapPtrs d em v).get1 i = em.offset (entryInt d (v.get1 i)) := by
  rcases hd with rfl | rfl <;> rfl

/-! ## Counting -/

/-! ## The masked load's lane -/

/-- The mask of the block load at trip `t`: lane `l` is enabled exactly when the position is in
range. -/
theorem mask_lane {d : Inst} (hd : d.ok) {t : Nat} (ht : t < numTrips d) (x : I32)
    (hx : x.toInt = (t : Int)) (l : Fin d.B) :
    IntV.lt ((tl.arange 0 d.B).get (l, ())) (IntV.ofNat d.numel - x * (IntV.ofNat d.B : I32)) =
      decide (t * d.B + l.val < d.numel) := by
  have hB := B_pos hd
  have hB31 := B_lt hd
  have hn31 := numel_lt hd
  have hlt : t * d.B < d.numel := numTrips_pred_mul hd ht
  have hlB : l.val < d.B := l.isLt
  have hlB31 : l.val < 2 ^ 31 := lt_of_lt_of_le hlB (le_of_lt hB31)
  have hl : ((tl.arange 0 d.B).get (l, ())).toInt = (l.val : Int) := by
    rw [tl.arange, Tensor.get_arange]
    norm_num
    exact IntV.I32.toInt_ofInt (by constructor <;> omega)
  have hBv : (IntV.ofNat d.B : I32).toInt = (d.B : Int) := IntV.I32.toInt_ofNat (by omega)
  have h1 : (IntV.ofNat d.numel : I32).toInt = (d.numel : Int) := IntV.I32.toInt_ofNat (by omega)
  have hmul : (x * (IntV.ofNat d.B : I32)).toInt = ((t * d.B : Nat) : Int) := by
    have h := IntV.I32.mul_nat hx hBv (by omega)
    simpa using h
  have hsub : (IntV.ofNat d.numel - x * (IntV.ofNat d.B : I32)).toInt
      = (d.numel : Int) - ((t * d.B : Nat) : Int) := by
    rw [IntV.I32.sub_nat h1 hmul (le_of_lt hlt) (by omega), Nat.cast_sub (le_of_lt hlt)]
  rw [IntV.lt_eq, hl, hsub, decide_eq_decide]
  omega

/-! ## A single-lane store -/

end Claims.CountExpertTokens
