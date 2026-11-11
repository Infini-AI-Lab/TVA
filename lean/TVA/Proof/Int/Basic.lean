import TVA.Core

/-!
# Integers: range, constants, equality, comparisons

The facts about one `IntV` value that every integer proof starts from (the conventions are in
`TVA.Proof.Int`):

* `SFits w x`, "`x` fits `w` signed bits", and how to establish it: `sfits_iff`,
  `sfits_of_nonneg`, `sfits_natCast`, `SFits.mono`, `sfits_toInt`; for sums and products of
  naturals `sfits_nat_add`, `sfits_nat_mul` and their `_of_le` forms (a bound per operand);
  `SFits.of_le_of_le` (between two values that fit), `sfits_of_abs_lt`; `I32.sfits_iff`,
  `I64.sfits_iff`, `I32.sfits`, `I64.sfits`, `I32/I64.sfits_of_nonneg`, `I32/I64.sfits_natCast`
  with the bounds written out, `I64.sfits_of_sfits32`.
* The value of a constant: `toInt_ofInt` (`IntV.ofInt x` in range), `toInt_ofNat` (`IntV.ofNat n`
  in range), `toInt_ofNat_lit` (a numeral such as `(128 : I32)`), the unsigned forms, `toInt_zero`,
  `toInt_neg_one`; by name without a bound: `toInt_ofNat_zero`, `toInt_ofInt_zero`, `toInt_one`,
  `toInt_one_unsigned`, `toInt_ofInt_one`, `toInt_ofInt_neg_one`; `I32.toInt_ofInt`,
  `I32.toInt_ofNat` and the `I64` forms with the bounds written out.
* Building a constant: `ofInt_natCast`, `ofInt_toInt`, `ofNat_toInt_toNat`, and the wrapping
  arithmetic of constants (`ofNat_add_ofNat`, `ofInt_sub_ofInt`, …), true at every value.
* Equality and order: `ext_toInt`, `toInt_inj`, `eq_eq` (`==`), `not_eq_eq` (`!=`), `lt_eq` (`<`),
  `le_eq` (`<=`), `toInt_max`, `toInt_min` (`tl.maximum`, `tl.minimum`), for either signedness.

Arithmetic (`+ - * // %`) is `TVA.Proof.Int.Arith`; casts and conversions `TVA.Proof.Int.Cast`.
-/
namespace TVA

namespace IntV

variable {s : Bool} {w : Nat}

/-! ## The signed range -/

/-- `x` lies in the range of signed `w`-bit integers, `[-2^(w-1), 2^(w-1))`: a computation in
`tl.intW` whose exact result satisfies it does not wrap. -/
def SFits (w : Nat) (x : Int) : Prop := -(2 ^ (w - 1) : Int) ≤ x ∧ x < 2 ^ (w - 1)

/-- `SFits` written out; rewrite with it to hand the two bounds to `omega` or `norm_num`. -/
theorem sfits_iff {x : Int} : SFits w x ↔ -(2 ^ (w - 1) : Int) ≤ x ∧ x < 2 ^ (w - 1) := Iff.rfl

/-- `SFits 32` written out: the range of `tl.int32`. -/
theorem I32.sfits_iff {x : Int} : SFits 32 x ↔ -(2 ^ 31 : Int) ≤ x ∧ x < 2 ^ 31 := Iff.rfl

/-- `SFits 64` written out: the range of `tl.int64`. -/
theorem I64.sfits_iff {x : Int} : SFits 64 x ↔ -(2 ^ 63 : Int) ≤ x ∧ x < 2 ^ 63 := Iff.rfl

/-- A nonnegative integer below `2^(w-1)` fits `w` signed bits (an index or an offset). -/
theorem sfits_of_nonneg {x : Int} (h0 : 0 ≤ x) (h : x < 2 ^ (w - 1)) : SFits w x :=
  ⟨le_trans (neg_nonpos.mpr (by positivity)) h0, h⟩

/-- A natural number below `2^(w-1)` fits `w` signed bits. -/
theorem sfits_natCast {n : Nat} (h : n < 2 ^ (w - 1)) : SFits w (n : Int) :=
  sfits_of_nonneg (by positivity) (by exact_mod_cast h)

/-- A value that fits `w` signed bits fits every wider type (`int32` into `int64`). -/
theorem SFits.mono {w' : Nat} {x : Int} (hw : w ≤ w') (h : SFits w x) : SFits w' x := by
  have : (2 : Int) ^ (w - 1) ≤ 2 ^ (w' - 1) := pow_le_pow_right₀ (by norm_num) (by omega)
  exact ⟨by linarith [h.1], by linarith [h.2]⟩

/-- Use when an `int32`/`int64` sum of two naturals must not wrap (`pid*B + i`): the sum of two
naturals below `2^(w-1)` fits, for `IntV.toInt_add`. -/
theorem sfits_nat_add {x y : Nat} (h : x + y < 2 ^ (w - 1)) : SFits w ((x : Int) + y) := by
  have := sfits_natCast (w := w) h
  push_cast at this
  exact this

/-- Use when a product of two naturals (`row * stride`) must not wrap: it fits when it is below
`2^(w-1)`, for `IntV.toInt_mul`. -/
theorem sfits_nat_mul {x y : Nat} (h : x * y < 2 ^ (w - 1)) : SFits w ((x : Int) * y) := by
  have := sfits_natCast (w := w) h
  push_cast at this
  exact this

/-- Use when each summand has its own bound (`r0 ≤ M - 1`, `i ≤ BM - 1`): `x + y` fits when the
sum of the bounds is below `2^(w-1)`. -/
theorem sfits_nat_add_of_le {x y a b : Nat} (hx : x ≤ a) (hy : y ≤ b) (h : a + b < 2 ^ (w - 1)) :
    SFits w ((x : Int) + y) :=
  sfits_nat_add (by omega)

/-- Use when each factor has its own bound (a row index times a stride, each bounded by the
shape): `x * y` fits when the product of the bounds is below `2^(w-1)`. -/
theorem sfits_nat_mul_of_le {x y a b : Nat} (hx : x ≤ a) (hy : y ≤ b) (h : a * b < 2 ^ (w - 1)) :
    SFits w ((x : Int) * y) :=
  sfits_nat_mul (lt_of_le_of_lt (Nat.mul_le_mul hx hy) h)

/-- Use when a value lies between a smallest and a largest value already known to fit (the
offsets of a tile between its first and last lane). -/
theorem SFits.of_le_of_le {lo hi x : Int} (hlo : SFits w lo) (hhi : SFits w hi) (h1 : lo ≤ x)
    (h2 : x ≤ hi) : SFits w x :=
  ⟨hlo.1.trans h1, lt_of_le_of_lt h2 hhi.2⟩

/-- Use when a bound on the magnitude `|x| < 2^(w-1)` is at hand (a signed difference or an
error term): `x` fits. -/
theorem sfits_of_abs_lt {x : Int} (h : |x| < 2 ^ (w - 1)) : SFits w x := by
  rw [abs_lt] at h
  exact ⟨h.1.le, h.2⟩

/-- Use to build `SFits 32 x` from its two `int32` bounds (each by `omega`). -/
theorem I32.sfits {x : Int} (h0 : -(2 ^ 31 : Int) ≤ x) (h1 : x < 2 ^ 31) : SFits 32 x := ⟨h0, h1⟩

/-- Use to build `SFits 64 x` from its two `int64` bounds (each by `omega`). -/
theorem I64.sfits {x : Int} (h0 : -(2 ^ 63 : Int) ≤ x) (h1 : x < 2 ^ 63) : SFits 64 x := ⟨h0, h1⟩

/-- Use when an `int32` value is widened to `int64`: what fits `int32` fits `int64`. -/
theorem I64.sfits_of_sfits32 {x : Int} (h : SFits 32 x) : SFits 64 x := SFits.mono (by norm_num) h

/-- Use for a nonnegative `int32` index or offset below `2^31`: it fits `int32`. -/
theorem I32.sfits_of_nonneg {x : Int} (h0 : 0 ≤ x) (h : x < 2 ^ 31) : SFits 32 x :=
  IntV.sfits_of_nonneg h0 h

/-- Use for a nonnegative `int64` index or offset below `2^63`: it fits `int64`. -/
theorem I64.sfits_of_nonneg {x : Int} (h0 : 0 ≤ x) (h : x < 2 ^ 63) : SFits 64 x :=
  IntV.sfits_of_nonneg h0 h

/-- Use for a natural number (a shape, a lane index) below `2^31` used in `int32`: it fits. -/
theorem I32.sfits_natCast {n : Nat} (h : n < 2 ^ 31) : SFits 32 (n : Int) :=
  IntV.sfits_natCast h

/-- Use for a natural number below `2^63` used in `int64`: it fits. -/
theorem I64.sfits_natCast {n : Nat} (h : n < 2 ^ 63) : SFits 64 (n : Int) :=
  IntV.sfits_natCast h

/-- The value of a signed integer is in its own range. -/
theorem sfits_toInt (a : IntV true w) : SFits w a.toInt := by
  simp only [toInt, ite_true]
  exact ⟨BitVec.le_toInt _, BitVec.toInt_lt⟩

/-- Reducing an integer in range modulo `2^w` (balanced) leaves it unchanged. -/
private theorem bmod_of_sfits {x : Int} (hw : 0 < w) (h : SFits w x) : x.bmod (2 ^ w) = x := by
  have e : (2 : Int) ^ w = 2 ^ (w - 1) * 2 := by
    rw [← pow_succ]; congr 1; omega
  apply Int.bmod_eq_of_le_mul_two
  · push_cast; rw [e]; linarith [h.1]
  · push_cast; rw [e]; linarith [h.2]

/-- `IntV s 0` has one value, whose value is `0`; the lemmas below use it to need no `0 < w`. -/
theorem toInt_width_zero (a : IntV s 0) : a.toInt = 0 := by
  cases s <;> simp [toInt, BitVec.eq_nil a.bits]

/-! ## Constants -/

/-- A Python integer `x` in the signed range, made a `w`-bit constant (`IntV.ofInt x`), keeps its
value. -/
theorem toInt_ofInt {x : Int} (hw : 0 < w) (h : SFits w x) :
    (IntV.ofInt x : IntV true w).toInt = x := by
  simp only [toInt, ofInt, ite_true, BitVec.toInt_ofInt]
  exact bmod_of_sfits hw h

/-- `IntV.ofNat n` with `n < 2^(w-1)` has value `n` in a signed type of any width. -/
@[simp] theorem toInt_ofNat {n : Nat} (h : n < 2 ^ (w - 1)) :
    (IntV.ofNat n : IntV true w).toInt = n := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · rw [toInt_width_zero]; simp at h; omega
  · simp only [toInt, ofNat, ite_true, BitVec.toInt_ofNat']
    exact bmod_of_sfits hw (sfits_natCast h)

/-- `IntV.ofNat n` with `n < 2^w` has value `n` in an unsigned type. -/
@[simp] theorem toInt_ofNat_unsigned {n : Nat} (h : n < 2 ^ w) :
    (IntV.ofNat n : IntV false w).toInt = n := by
  simp only [toInt, ofNat, Bool.false_eq_true, ite_false, BitVec.toNat_ofNat, Nat.mod_eq_of_lt h]

/-- A numeral below `2^(w-1)` used as a signed constant (`(128 : I32)`) has its own value. -/
@[simp] theorem toInt_ofNat_lit {n : Nat} (h : n < 2 ^ (w - 1)) :
    (no_index (OfNat.ofNat n) : IntV true w).toInt = OfNat.ofNat n :=
  toInt_ofNat h

/-- A numeral below `2^w` used as an unsigned constant (`(255 : U8)`) has its own value. -/
@[simp] theorem toInt_ofNat_lit_unsigned {n : Nat} (h : n < 2 ^ w) :
    (no_index (OfNat.ofNat n) : IntV false w).toInt = OfNat.ofNat n :=
  toInt_ofNat_unsigned h

/-- The constant `0` has value `0`, in every integer type. -/
@[simp] theorem toInt_zero : (0 : IntV s w).toInt = 0 := by
  cases s <;> simp [toInt, OfNat.ofNat, ofNat]

/-- The constant `-1` has value `-1` in a signed type of positive width. -/
@[simp] theorem toInt_neg_one (hw : 0 < w) : (-1 : IntV true w).toInt = -1 := by
  show (-(BitVec.ofNat w 1)).toInt = -1
  rw [show -(BitVec.ofNat w 1) = BitVec.allOnes w from BitVec.neg_one_eq_allOnes,
    BitVec.toInt_allOnes]
  simp [hw]

/-- Use for the constant `IntV.ofNat 0` (a zero offset or a loop's first index): its value is
`0` in every integer type. -/
theorem toInt_ofNat_zero : (IntV.ofNat 0 : IntV s w).toInt = 0 := by
  cases s <;> simp [toInt, ofNat]

/-- Use for the constant `IntV.ofInt 0` (as `ConvRel` of `false` from `int1` gives): its value is
`0` in every integer type. -/
theorem toInt_ofInt_zero : (IntV.ofInt 0 : IntV s w).toInt = 0 := by
  cases s <;> simp [toInt, ofInt]

/-- Use for the signed constant `1` (a step `i + 1`) in a type of at least two bits, without
stating `1 < 2^(w-1)`. -/
theorem toInt_one (hw : 2 ≤ w) : (1 : IntV true w).toInt = 1 :=
  toInt_ofNat (Nat.one_lt_two_pow (by omega))

/-- Use for the unsigned constant `1` in a type of positive width. -/
theorem toInt_one_unsigned (hw : 0 < w) : (1 : IntV false w).toInt = 1 :=
  toInt_ofNat_unsigned (Nat.one_lt_two_pow (by omega))

/-- Use for the constant `IntV.ofInt (-1)` (a sentinel index) in a signed type of positive
width. -/
theorem toInt_ofInt_neg_one (hw : 0 < w) : (IntV.ofInt (-1) : IntV true w).toInt = -1 := by
  show (BitVec.ofInt w (-1)).toInt = -1
  rw [BitVec.ofInt_neg, show BitVec.ofInt w 1 = BitVec.ofNat w 1 from BitVec.ofInt_natCast w 1]
  exact toInt_neg_one hw

/-- An `int32` constant `IntV.ofInt x` with `x` in `[-2^31, 2^31)` has value `x`. -/
theorem I32.toInt_ofInt {x : Int} (h : -(2 ^ 31 : Int) ≤ x ∧ x < 2 ^ 31) :
    (IntV.ofInt x : I32).toInt = x := IntV.toInt_ofInt (by norm_num) h

/-- An `int32` constant `IntV.ofNat n` with `n < 2^31` has value `n`. -/
theorem I32.toInt_ofNat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n :=
  IntV.toInt_ofNat h

/-- An `int64` constant `IntV.ofInt x` with `x` in `[-2^63, 2^63)` has value `x`. -/
theorem I64.toInt_ofInt {x : Int} (h : -(2 ^ 63 : Int) ≤ x ∧ x < 2 ^ 63) :
    (IntV.ofInt x : I64).toInt = x := IntV.toInt_ofInt (by norm_num) h

/-- An `int64` constant `IntV.ofNat n` with `n < 2^63` has value `n`. -/
theorem I64.toInt_ofNat {n : Nat} (h : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = n :=
  IntV.toInt_ofNat h

/-! ## Building constants -/

/-- `IntV.ofInt (n : Int)` is `IntV.ofNat n`, in every integer type: rewrite with it to use the
`ofNat` lemmas. -/
theorem ofInt_natCast (n : Nat) : (IntV.ofInt (n : Int) : IntV s w) = IntV.ofNat n :=
  congrArg IntV.mk (BitVec.ofInt_natCast w n)

/-- Use for the constant `IntV.ofInt 1` (as `ConvRel` of `true` from `int1` gives) in a signed type
of at least two bits. -/
theorem toInt_ofInt_one (hw : 2 ≤ w) : (IntV.ofInt 1 : IntV true w).toInt = 1 := by
  have := ofInt_natCast (s := true) (w := w) 1
  simp only [Nat.cast_one] at this
  rw [this]
  exact toInt_ofNat (Nat.one_lt_two_pow (by omega))

/-- Rebuilding an integer from its value gives it back, in every integer type. -/
theorem ofInt_toInt (a : IntV s w) : (IntV.ofInt a.toInt : IntV s w) = a := by
  cases s
  · show (⟨BitVec.ofInt w (a.bits.toNat : Int)⟩ : IntV false w) = a
    rw [BitVec.ofInt_natCast, BitVec.ofNat_toNat, BitVec.setWidth_eq]
  · show (⟨BitVec.ofInt w a.bits.toInt⟩ : IntV true w) = a
    rw [BitVec.ofInt_toInt]

/-- Rebuilding a nonnegative integer from its value as a natural number gives it back. -/
theorem ofNat_toInt_toNat {a : IntV s w} (h : 0 ≤ a.toInt) :
    (IntV.ofNat a.toInt.toNat : IntV s w) = a := by
  rw [← ofInt_natCast, Int.toNat_of_nonneg h, ofInt_toInt]

/-! ## Wrapping arithmetic of constants

`+ - *` wrap modulo `2^w`, and so does building a constant, so these hold at every value. Read
left to right they fold constant arithmetic into one constant, whose value `toInt_ofNat` or
`toInt_ofInt` then gives. -/

/-- The sum of the constants `IntV.ofNat a` and `IntV.ofNat b` is the constant `a + b`. -/
theorem ofNat_add_ofNat (a b : Nat) :
    (IntV.ofNat a + IntV.ofNat b : IntV s w) = IntV.ofNat (a + b) :=
  congrArg IntV.mk (BitVec.ofNat_add_ofNat a b)

/-- The product of the constants `IntV.ofNat a` and `IntV.ofNat b` is the constant `a * b`. -/
theorem ofNat_mul_ofNat (a b : Nat) :
    (IntV.ofNat a * IntV.ofNat b : IntV s w) = IntV.ofNat (a * b) :=
  congrArg IntV.mk (BitVec.ofNat_mul_ofNat a b)

/-- The sum of the constants `IntV.ofInt a` and `IntV.ofInt b` is the constant `a + b`. -/
theorem ofInt_add_ofInt (a b : Int) :
    (IntV.ofInt a + IntV.ofInt b : IntV s w) = IntV.ofInt (a + b) :=
  congrArg IntV.mk (BitVec.ofInt_add a b).symm

/-- The difference of the constants `IntV.ofInt a` and `IntV.ofInt b` is the constant `a - b`. -/
theorem ofInt_sub_ofInt (a b : Int) :
    (IntV.ofInt a - IntV.ofInt b : IntV s w) = IntV.ofInt (a - b) := by
  show (⟨BitVec.ofInt w a - BitVec.ofInt w b⟩ : IntV s w) = ⟨BitVec.ofInt w (a - b)⟩
  rw [sub_eq_add_neg a b, BitVec.ofInt_add, BitVec.ofInt_neg, BitVec.sub_eq_add_neg]

/-- The product of the constants `IntV.ofInt a` and `IntV.ofInt b` is the constant `a * b`. -/
theorem ofInt_mul_ofInt (a b : Int) :
    (IntV.ofInt a * IntV.ofInt b : IntV s w) = IntV.ofInt (a * b) :=
  congrArg IntV.mk (BitVec.ofInt_mul a b).symm

/-- The negation of the constant `IntV.ofInt a` is the constant `-a`. -/
theorem neg_ofInt (a : Int) : (-(IntV.ofInt a) : IntV s w) = IntV.ofInt (-a) :=
  congrArg IntV.mk BitVec.ofInt_neg.symm

/-! ## Equality and order -/

/-- Two integers of one type with the same value are equal: prove an `IntV` equation by values. -/
theorem ext_toInt {a b : IntV s w} (h : a.toInt = b.toInt) : a = b := by
  cases s
  · simp only [toInt, Bool.false_eq_true, ite_false, Int.natCast_inj] at h
    exact IntV.ext (BitVec.eq_of_toNat_eq h)
  · simp only [toInt, ite_true] at h
    exact IntV.ext (BitVec.toInt_inj.mp h)

/-- Two integers of one type are equal exactly when their values are. -/
theorem toInt_inj {a b : IntV s w} : a.toInt = b.toInt ↔ a = b :=
  ⟨ext_toInt, fun h => h ▸ rfl⟩

/-- `a == b` on integers compares values (`a != b` is its negation). -/
theorem eq_eq (a b : IntV s w) : a.eq b = decide (a.toInt = b.toInt) := by
  rw [Bool.eq_iff_iff, decide_eq_true_iff, toInt_inj, IntV.eq, beq_iff_eq, IntV.ext_iff]

/-- Use for `a != b` (written `!(a.eq b)`, a mask `offs != k`): it compares values. -/
theorem not_eq_eq (a b : IntV s w) : (!a.eq b) = decide (a.toInt ≠ b.toInt) := by
  rw [eq_eq]; simp

/-- `a < b` on integers, signed or unsigned, compares values (a mask `offs < n`). -/
theorem lt_eq (a b : IntV s w) : a.lt b = decide (a.toInt < b.toInt) := by
  cases s
  · simp only [IntV.lt, IntV.toInt, Bool.false_eq_true, ite_false]
    rw [BitVec.ult]; simp
  · simp only [IntV.lt, IntV.toInt, ite_true]
    rw [BitVec.slt]

/-- `a <= b` on integers, signed or unsigned, compares values. -/
theorem le_eq (a b : IntV s w) : a.le b = decide (a.toInt ≤ b.toInt) := by
  cases s
  · simp only [IntV.le, IntV.toInt, Bool.false_eq_true, ite_false]
    rw [BitVec.ule]; simp
  · simp only [IntV.le, IntV.toInt, ite_true]
    rw [BitVec.sle]

/-- `tl.maximum(a, b)` of integers has the larger value. -/
theorem toInt_max (a b : IntV s w) : (a.max b).toInt = Max.max a.toInt b.toInt := by
  unfold IntV.max
  rw [lt_eq]
  by_cases h : a.toInt < b.toInt
  · simp [h, max_eq_right h.le]
  · simp [h, max_eq_left (not_lt.mp h)]

/-- `tl.minimum(a, b)` of integers has the smaller value. -/
theorem toInt_min (a b : IntV s w) : (a.min b).toInt = Min.min a.toInt b.toInt := by
  unfold IntV.min
  rw [lt_eq]
  by_cases h : a.toInt < b.toInt
  · simp [h, min_eq_left h.le]
  · simp [h, min_eq_right (not_lt.mp h)]

end IntV

end TVA
