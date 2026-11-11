import Claim
import TVA.Proof

/-!
# Small facts used throughout

Rank-0 tensors are their item, the zero constant is `+0`, and an integer value is its bit
pattern when their values agree.
-/

namespace Claims.ChunkCumsum

open TVA

/-- A rank-0 tensor is the scalar of its item. -/
theorem tensor_scalar_item {α : Type} (r : Tensor [] α) : Tensor.scalar r.item = r := by
  cases r with
  | mk g =>
    have h : (fun _ : Idx [] => g ()) = g := by
      funext i
      exact congrArg g (Subsingleton.elim () i).symm
    simp only [Tensor.scalar, Tensor.item, h]

/-- `+0` holds `0`. -/
theorem ofPat_zero_toRat : (FloatV.ofPat 0 : F32).toRat = 0 := by
  apply FloatV.toRat_of_isZero
  rw [FloatV.pat_ofPat (by decide)]
  exact Format.isZero_zero

/-- `+0` is finite. -/
theorem ofPat_zero_isFinite : (FloatV.ofPat 0 : F32).isFinite = true := by
  rw [FloatV.isFinite_ofPat (by decide)]
  decide

/-- The literal `0` Triton writes is `+0`. -/
theorem litF32_zero : (FloatV.litF32 (FloatV.ofPat 0) : F32) = FloatV.ofPat 0 := by
  have h : (FloatV.ofPat 0 : F64).isZero = true := by decide
  simp only [FloatV.litF32, h, ite_true]

/-- Signed integer values with equal value are equal. -/
theorem intv_eq_of_toInt_eq {w : Nat} {a b : IntV true w} (h : a.toInt = b.toInt) : a = b := by
  have ha : a.toInt = a.bits.toInt := rfl
  have hb : b.toInt = b.bits.toInt := rfl
  have h' : a.bits.toInt = b.bits.toInt := by rw [← ha, ← hb]; exact h
  have h'' : a.bits = b.bits := BitVec.eq_of_toInt_eq h'
  cases a; cases b; simp_all

end Claims.ChunkCumsum
