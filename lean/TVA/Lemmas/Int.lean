import TVA.Lang.Tensor
import Mathlib.Tactic

/-!
# Integer lemmas

Index arithmetic in Triton is `tl.int32` (or `tl.int64`) and wraps. A proof shows it does not
wrap and then computes with mathematical integers. `IntV.SFits w x`: `x` is in the signed
`w`-bit range. Within it:

* `toInt_ofInt`, `toInt_ofNat`: a literal (`IntV.ofInt`, `IntV.ofNat`) in range keeps its value;
* `toInt_add`, `toInt_sub`, `toInt_mul`: operations are the integer ones;
* `lt_eq`, `le_eq`: comparisons of signed integers are the integer ones, in any range (for either
  signedness, unsigned included, `IntV.lt_toInt` for `<` and `IntV.le_toInt` for `<=`, in
  `TVA.Lemmas.SharedAcc`).

And at any value: `toInt_cast` (a widening cast `x.to(tl.int64)` of a signed integer keeps its
value: sign extension), `toInt_cast_unsigned` (the same for an unsigned one: zero extension).
-/
namespace TVA

namespace IntV

/-- `x` lies in the range of signed `w`-bit integers. -/
def SFits (w : Nat) (x : Int) : Prop := -(2 ^ (w - 1) : Int) ≤ x ∧ x < 2 ^ (w - 1)

variable {w : Nat}

theorem bmod_of_sfits {x : Int} (hw : 0 < w) (h : SFits w x) : x.bmod (2 ^ w) = x := by
  apply Int.bmod_eq_of_le_mul_two
  · have : (2 : Int) ^ w = 2 ^ (w - 1) * 2 := by
      rw [← pow_succ]; congr 1; omega
    push_cast; rw [this]; linarith [h.1]
  · have : (2 : Int) ^ w = 2 ^ (w - 1) * 2 := by
      rw [← pow_succ]; congr 1; omega
    push_cast; rw [this]; linarith [h.2]

/-- The value of a signed integer is in range. -/
theorem sfits_toInt (a : IntV true w) : SFits w a.toInt := by
  simp only [toInt, ite_true]
  exact ⟨BitVec.le_toInt _, BitVec.toInt_lt⟩

theorem toInt_ofInt {x : Int} (hw : 0 < w) (h : SFits w x) : (IntV.ofInt x : IntV true w).toInt = x := by
  simp only [toInt, ofInt, ite_true, BitVec.toInt_ofInt]
  exact bmod_of_sfits hw h

theorem toInt_ofNat {n : Nat} (hw : 0 < w) (h : (n : Int) < 2 ^ (w - 1)) :
    (IntV.ofNat n : IntV true w).toInt = n := by
  simp only [toInt, ofNat, ite_true, BitVec.toInt_ofNat']
  have h1 : (0 : Int) ≤ 2 ^ (w - 1) := by positivity
  have h2 : (0 : Int) ≤ n := by positivity
  exact bmod_of_sfits hw ⟨by linarith, h⟩

theorem toInt_add {a b : IntV true w} (hw : 0 < w) (h : SFits w (a.toInt + b.toInt)) :
    (a + b).toInt = a.toInt + b.toInt := by
  have h' := h
  simp only [toInt, ite_true] at h' ⊢
  show (a.bits + b.bits).toInt = _
  rw [BitVec.toInt_add]
  exact bmod_of_sfits hw h'

theorem toInt_sub {a b : IntV true w} (hw : 0 < w) (h : SFits w (a.toInt - b.toInt)) :
    (a - b).toInt = a.toInt - b.toInt := by
  have h' := h
  simp only [toInt, ite_true] at h' ⊢
  show (a.bits - b.bits).toInt = _
  rw [BitVec.toInt_sub]
  exact bmod_of_sfits hw h'

theorem toInt_mul {a b : IntV true w} (hw : 0 < w) (h : SFits w (a.toInt * b.toInt)) :
    (a * b).toInt = a.toInt * b.toInt := by
  have h' := h
  simp only [toInt, ite_true] at h' ⊢
  show (a.bits * b.bits).toInt = _
  rw [BitVec.toInt_mul]
  exact bmod_of_sfits hw h'

/-- Signed comparison is integer comparison. -/
theorem lt_eq (a b : IntV true w) : a.lt b = decide (a.toInt < b.toInt) := by
  simp only [lt, toInt, ite_true]
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  exact BitVec.slt_iff_toInt_lt

theorem le_eq (a b : IntV true w) : a.le b = decide (a.toInt ≤ b.toInt) := by
  simp only [le, toInt, ite_true]
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  exact BitVec.sle_iff_toInt_le

/-- **A widening cast of a signed integer keeps its value** (`IntV.cast`: sign extension). -/
theorem toInt_cast {w' : Nat} (a : IntV true w) (h : w ≤ w') : (a.cast : IntV true w').toInt = a.toInt := by
  simp only [IntV.cast, IntV.toInt, ite_true]
  exact BitVec.toInt_signExtend_of_le h

/-- **A widening cast of an unsigned integer keeps its value** (`IntV.cast`: zero extension). -/
theorem toInt_cast_unsigned {w' : Nat} (a : IntV false w) (h : w ≤ w') :
    (a.cast : IntV false w').toInt = a.toInt := by
  simp only [IntV.cast, IntV.toInt, Bool.false_eq_true, ite_false]
  rw [BitVec.toNat_setWidth,
    Nat.mod_eq_of_lt (lt_of_lt_of_le a.bits.isLt (Nat.pow_le_pow_right (by norm_num) h))]

/-- `int32` facts with the width fixed. -/
theorem I32.toInt_ofInt {x : Int} (h : -(2 ^ 31 : Int) ≤ x ∧ x < 2 ^ 31) :
    (IntV.ofInt x : I32).toInt = x := IntV.toInt_ofInt (by norm_num) (by simpa [SFits] using h)

theorem I32.toInt_ofNat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n :=
  IntV.toInt_ofNat (by norm_num) (by push_cast; exact_mod_cast h)

end IntV

end TVA

namespace TVA.IntV

variable {s : Bool} {w : Nat}

theorem ext_bits {a b : IntV s w} (h : a.bits = b.bits) : a = b := by
  cases a; cases b; cases h; rfl

/-- Integers of a width form a commutative monoid under wrapping addition (sums of atomic
increments, `tl.sum` of integers). -/
instance : AddCommMonoid (IntV s w) where
  add := (· + ·)
  zero := 0
  add_assoc a b c := ext_bits (BitVec.add_assoc a.bits b.bits c.bits)
  zero_add a := ext_bits (by show BitVec.ofNat w 0 + a.bits = a.bits; simp)
  add_zero a := ext_bits (by show a.bits + BitVec.ofNat w 0 = a.bits; simp)
  add_comm a b := ext_bits (BitVec.add_comm a.bits b.bits)
  nsmul := nsmulRec

end TVA.IntV
