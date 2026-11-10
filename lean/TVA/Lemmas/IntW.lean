import TVA.Lemmas.Scalar

/-!
# Integer facts at any width (library issue L175)

`TVA.Lemmas.Int` gives `toInt_add/sub/mul` under an `SFits` premise, and `TVA.Lemmas.Scalar` the
`int32` forms `IntV.I32.add_nat`, `mul_nat`, `sub_nat`. This file gives the same facts for every
width, and the small facts a proof about index arithmetic needs on the way. All are about
`IntV`'s definitions only (no device fact).

* `SFits` (all in `IntV`): `sfits_iff` (the definition written out), `sfits_of_nonneg`,
  `sfits_natCast` (a natural below `2^(w-1)`), `sfits_nat_add`, `sfits_nat_mul` (a sum or product of
  naturals), `sfits_nat_add_of_le`, `sfits_nat_mul_of_le` (of naturals each below a bound),
  `SFits.mono` (a wider type), `SFits.of_le_of_le` (between two values in range), `sfits_of_abs_lt`;
  `I32.sfits_iff`, `I64.sfits_iff` (`simp`: the bounds written out), `I32.sfits`, `I64.sfits`,
  `I32.sfits_of_nonneg`, `I64.sfits_of_nonneg`, `I32.sfits_natCast`, `I64.sfits_natCast`,
  `I64.sfits_of_sfits32`.
* Tracked values, for any width `w` and no `0 < w` premise: from `a.toInt = x` and `b.toInt = y`,
  `add_nat`, `mul_nat`, `sub_nat` (naturals, one bound) and `add_int`, `sub_int`, `mul_int`
  (integers, `SFits`); with a constant `IntV.ofNat n` or a numeral, `add_ofNat`, `mul_ofNat`,
  `ofNat_mul`, `sub_ofNat`; `add_nat_unsigned`, `mul_nat_unsigned`, `sub_nat_unsigned` for unsigned
  types; `I64.add_nat`, `mul_nat`, `sub_nat`; `I32.toInt_add/sub/mul` and `I64.toInt_add/sub/mul`
  with the bounds written out.
* Constants (`simp`): `toInt_zero`, `toInt_one`, `toInt_neg_one`, `toInt_ofNat_lit` (a numeral in
  range), `toInt_ofNat_of_lt` (`IntV.ofNat n` in range), `toInt_ofNat_zero`, `toInt_ofInt_zero`,
  `toInt_ofInt_one`, `toInt_ofInt_neg_one`, `ofInt_natCast`, `ofInt_toInt`; unsigned:
  `toInt_ofNat_unsigned`, `toInt_ofNat_lit_unsigned`, `toInt_one_unsigned`; `I64.toInt_ofNat`,
  `I64.ofNat_nat`, `I64.toInt_ofInt`; `ofNat_toInt_toNat`, `toInt_width_zero`.
* Wrapping arithmetic of constants, always true (`simp`): `ofNat_add_ofNat`, `ofNat_mul_ofNat`,
  `ofInt_add_ofInt`, `ofInt_sub_ofInt`, `ofInt_mul_ofInt`, `neg_ofInt`.
* Equality: `ext_toInt`, `toInt_inj`, `eq_eq` (`==` is equality of values), `not_eq_eq` (`!=`).
* Division: `div?_nat`, `rem?_nat`, `div?_ofNat`, `rem?_ofNat` (defined, with the value); the value
  of a given result (a lane of `tl.idiv`/`tl.irem`): `toInt_of_div?`, `toInt_of_rem?` (nonnegative
  by positive), `toInt_of_div?_nat`, `toInt_of_rem?_nat`.
* Rebuilding in a type at least as wide: `toInt_ofInt_toInt`, `toInt_ofInt_toInt_unsigned`,
  `toInt_ofInt_toInt_unsigned_le`.
* Conversions (`ConvRel` between integer dtypes): a widening one keeps `toInt`
  (`ConvRel.toInt_i32_i64` and the other pairs of `int8`…`int64`, `uint8`…`uint64` into a type
  holding every source value, and `i32_i32`, `i64_i64`, `u32_u32`, `u64_u64`), a narrowing one in
  range too (`ConvRel.toInt_i64_i32`); `int1` gives `0` or `1` (`toInt_i1_i32`, `toInt_i1_i64`);
  `exists_i32_i64`, `exists_i64_i64`, `exists_i64_i32`.
* `tl.arange` lanes: `I32.toInt_arange`, `I32.toInt_tl_arange`, `I32.toInt_sadd_arange`.
* Lanes of integer vectors (`Tensor`): `get1_add`, `get1_sub`, `get1_mul`, `get1_neg`, `get1_sadd`,
  `get1_adds`, `get1_smul`, `get1_muls`, `get1_ltS`, `get1_lt`, `get1_le`, `get1_eq`,
  `get1_castInt` (`simp`: the `get1` forms of `Tensor.get_*`); `toInt_get1_add`, `toInt_get1_sub`,
  `toInt_get1_mul`, `toInt_get1_sadd`, `toInt_get1_muls` (tracked lane values).
-/

namespace TVA

namespace IntV

variable {s : Bool} {w : Nat}

/-! ## `SFits` -/

/-- `SFits` written out (it is a definition; use this to rewrite it into its two bounds). -/
theorem sfits_iff {x : Int} : SFits w x ↔ -(2 ^ (w - 1) : Int) ≤ x ∧ x < 2 ^ (w - 1) := Iff.rfl

/-- A nonnegative integer below `2^(w-1)` fits `w` signed bits. -/
theorem sfits_of_nonneg {x : Int} (h0 : 0 ≤ x) (h : x < 2 ^ (w - 1)) : SFits w x :=
  ⟨le_trans (neg_nonpos.mpr (by positivity)) h0, h⟩

/-- A natural number below `2^(w-1)` fits `w` signed bits. -/
theorem sfits_natCast {n : Nat} (h : n < 2 ^ (w - 1)) : SFits w (n : Int) :=
  sfits_of_nonneg (by positivity) (by exact_mod_cast h)

/-- A sum of naturals below `2^(w-1)` fits `w` signed bits. -/
theorem sfits_nat_add {x y : Nat} (h : x + y < 2 ^ (w - 1)) : SFits w ((x : Int) + y) := by
  have := sfits_natCast (w := w) h
  push_cast at this
  exact this

/-- A product of naturals below `2^(w-1)` fits `w` signed bits. -/
theorem sfits_nat_mul {x y : Nat} (h : x * y < 2 ^ (w - 1)) : SFits w ((x : Int) * y) := by
  have := sfits_natCast (w := w) h
  push_cast at this
  exact this

/-- A value that fits `w` signed bits fits every wider type (`int32` into `int64`). -/
theorem SFits.mono {w' : Nat} {x : Int} (hw : w ≤ w') (h : SFits w x) : SFits w' x := by
  have : (2 : Int) ^ (w - 1) ≤ 2 ^ (w' - 1) := pow_le_pow_right₀ (by norm_num) (by omega)
  exact ⟨by linarith [h.1], by linarith [h.2]⟩

/-- A value between two values in range is in range. -/
theorem SFits.of_le_of_le {lo hi x : Int} (hlo : SFits w lo) (hhi : SFits w hi) (h1 : lo ≤ x)
    (h2 : x ≤ hi) : SFits w x :=
  ⟨hlo.1.trans h1, lt_of_le_of_lt h2 hhi.2⟩

/-- An integer of magnitude below `2^(w-1)` fits `w` signed bits. -/
theorem sfits_of_abs_lt {x : Int} (h : |x| < 2 ^ (w - 1)) : SFits w x := by
  rw [abs_lt] at h
  exact ⟨h.1.le, h.2⟩

/-- `SFits 32` written out: the range of `int32`. -/
@[simp] theorem I32.sfits_iff {x : Int} : SFits 32 x ↔ -(2 ^ 31 : Int) ≤ x ∧ x < 2 ^ 31 := Iff.rfl

/-- `SFits 64` written out: the range of `int64`. -/
@[simp] theorem I64.sfits_iff {x : Int} : SFits 64 x ↔ -(2 ^ 63 : Int) ≤ x ∧ x < 2 ^ 63 := Iff.rfl

/-- An integer in `[-2^31, 2^31)` fits `int32`. -/
theorem I32.sfits {x : Int} (h0 : -(2 ^ 31 : Int) ≤ x) (h1 : x < 2 ^ 31) : SFits 32 x := ⟨h0, h1⟩

/-- An integer in `[-2^63, 2^63)` fits `int64`. -/
theorem I64.sfits {x : Int} (h0 : -(2 ^ 63 : Int) ≤ x) (h1 : x < 2 ^ 63) : SFits 64 x := ⟨h0, h1⟩

/-- A value that fits `int32` fits `int64`. -/
theorem I64.sfits_of_sfits32 {x : Int} (h : SFits 32 x) : SFits 64 x := SFits.mono (by norm_num) h

/-- A nonnegative integer below `2^31` fits `int32` (an index or offset known to be nonnegative). -/
theorem I32.sfits_of_nonneg {x : Int} (h0 : 0 ≤ x) (h : x < 2 ^ 31) : SFits 32 x :=
  IntV.sfits_of_nonneg h0 h

/-- A nonnegative integer below `2^63` fits `int64`. -/
theorem I64.sfits_of_nonneg {x : Int} (h0 : 0 ≤ x) (h : x < 2 ^ 63) : SFits 64 x :=
  IntV.sfits_of_nonneg h0 h

/-- A natural number below `2^31` fits `int32`. -/
theorem I32.sfits_natCast {n : Nat} (h : n < 2 ^ 31) : SFits 32 (n : Int) := IntV.sfits_natCast h

/-- A natural number below `2^63` fits `int64`. -/
theorem I64.sfits_natCast {n : Nat} (h : n < 2 ^ 63) : SFits 64 (n : Int) := IntV.sfits_natCast h

/-- A sum of naturals `x ≤ a`, `y ≤ b` fits `w` signed bits when `a + b < 2^(w-1)` (a bound on
each term, such as `r0 ≤ (M - 1)` and `i ≤ BM - 1`). -/
theorem sfits_nat_add_of_le {x y a b : Nat} (hx : x ≤ a) (hy : y ≤ b) (h : a + b < 2 ^ (w - 1)) :
    SFits w ((x : Int) + y) :=
  sfits_nat_add (by omega)

/-- A product of naturals `x ≤ a`, `y ≤ b` fits `w` signed bits when `a * b < 2^(w-1)` (a row
index times a stride, each bounded by the shape). -/
theorem sfits_nat_mul_of_le {x y a b : Nat} (hx : x ≤ a) (hy : y ≤ b) (h : a * b < 2 ^ (w - 1)) :
    SFits w ((x : Int) * y) :=
  sfits_nat_mul (lt_of_le_of_lt (Nat.mul_le_mul hx hy) h)

/-! ## Width zero

`IntV s 0` has one value, of `toInt` `0`. The lemmas below treat this case themselves, so that
they need no `0 < w` premise. -/

/-- The only value of width `0` has value `0`. -/
theorem toInt_width_zero (a : IntV s 0) : a.toInt = 0 := by
  cases s <;> simp [toInt, BitVec.eq_nil a.bits]

/-! ## Constants -/

/-- A constant built from a natural number cast to `Int` is the constant built from the natural
number: `IntV.ofInt (n : Int) = IntV.ofNat n`, in any integer type (`simp` turns the first form
into the second, whose value `toInt_ofNat_of_lt` gives). -/
@[simp] theorem ofInt_natCast (n : Nat) : (IntV.ofInt (n : Int) : IntV s w) = IntV.ofNat n :=
  congrArg IntV.mk (BitVec.ofInt_natCast w n)

/-- `IntV.ofNat n` below `2^(w-1)` has value `n`, in a signed type of any width. -/
@[simp] theorem toInt_ofNat_of_lt {n : Nat} (h : n < 2 ^ (w - 1)) :
    (IntV.ofNat n : IntV true w).toInt = n := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · rw [toInt_width_zero]; simp at h; omega
  · exact toInt_ofNat hw (by exact_mod_cast h)

/-- `IntV.ofNat n` below `2^w` has value `n`, in an unsigned type. -/
@[simp] theorem toInt_ofNat_unsigned {n : Nat} (h : n < 2 ^ w) :
    (IntV.ofNat n : IntV false w).toInt = n := by
  simp only [toInt, ofNat, Bool.false_eq_true, ite_false, BitVec.toNat_ofNat, Nat.mod_eq_of_lt h]

/-- A numeral `n ≥ 2` below `2^(w-1)`, as a signed `w`-bit constant (`(128 : I32)`), has value
`n`. -/
@[simp] theorem toInt_ofNat_lit {n : Nat} [n.AtLeastTwo] (h : n < 2 ^ (w - 1)) :
    (no_index (OfNat.ofNat n) : IntV true w).toInt = OfNat.ofNat n :=
  toInt_ofNat_of_lt h

/-- A numeral `n ≥ 2` below `2^w`, as an unsigned `w`-bit constant (`(255 : U8)`), has value `n`. -/
@[simp] theorem toInt_ofNat_lit_unsigned {n : Nat} [n.AtLeastTwo] (h : n < 2 ^ w) :
    (no_index (OfNat.ofNat n) : IntV false w).toInt = OfNat.ofNat n :=
  toInt_ofNat_unsigned h

/-- The constant `1` has value `1` in an unsigned type of positive width. -/
@[simp] theorem toInt_one_unsigned (hw : 0 < w) : (1 : IntV false w).toInt = 1 := by
  have h : 1 < 2 ^ w := Nat.one_lt_two_pow (by omega)
  exact toInt_ofNat_unsigned h

/-- The constant `0` has value `0`, in any integer type. -/
@[simp] theorem toInt_zero : (0 : IntV s w).toInt = 0 := by
  cases s <;> simp [toInt, OfNat.ofNat, ofNat]

/-- `IntV.ofNat 0` has value `0`, in any integer type. -/
@[simp] theorem toInt_ofNat_zero : (IntV.ofNat 0 : IntV s w).toInt = 0 := by
  cases s <;> simp [toInt, ofNat]

/-- `IntV.ofInt 0` has value `0`, in any integer type. -/
@[simp] theorem toInt_ofInt_zero : (IntV.ofInt 0 : IntV s w).toInt = 0 := by
  cases s <;> simp [toInt, ofInt]

/-- The constant `1` has value `1` in a signed type of at least two bits. -/
@[simp] theorem toInt_one (hw : 2 ≤ w) : (1 : IntV true w).toInt = 1 := by
  have h : 1 < 2 ^ (w - 1) := Nat.one_lt_two_pow (by omega)
  exact toInt_ofNat_of_lt h

/-- `IntV.ofInt 1` has value `1` in a signed type of at least two bits. -/
@[simp] theorem toInt_ofInt_one (hw : 2 ≤ w) : (IntV.ofInt 1 : IntV true w).toInt = 1 := by
  have h : 1 < 2 ^ (w - 1) := Nat.one_lt_two_pow (by omega)
  have := ofInt_natCast (s := true) (w := w) 1
  simp only [Nat.cast_one] at this
  rw [this]
  exact toInt_ofNat_of_lt h

/-- The constant `-1` has value `-1` in a signed type of positive width. -/
@[simp] theorem toInt_neg_one (hw : 0 < w) : (-1 : IntV true w).toInt = -1 := by
  show (-(BitVec.ofNat w 1)).toInt = -1
  rw [show -(BitVec.ofNat w 1) = BitVec.allOnes w from BitVec.neg_one_eq_allOnes,
    BitVec.toInt_allOnes]
  simp [hw]

/-- `IntV.ofInt (-1)` has value `-1` in a signed type of positive width. -/
@[simp] theorem toInt_ofInt_neg_one (hw : 0 < w) : (IntV.ofInt (-1) : IntV true w).toInt = -1 := by
  show (BitVec.ofInt w (-1)).toInt = -1
  rw [BitVec.ofInt_neg, show BitVec.ofInt w 1 = BitVec.ofNat w 1 from BitVec.ofInt_natCast w 1]
  exact toInt_neg_one hw

/-- Rebuilding an integer from its value gives it back, in any integer type. -/
@[simp] theorem ofInt_toInt (a : IntV s w) : (IntV.ofInt a.toInt : IntV s w) = a := by
  cases s
  · show (⟨BitVec.ofInt w (a.bits.toNat : Int)⟩ : IntV false w) = a
    rw [BitVec.ofInt_natCast, BitVec.ofNat_toNat, BitVec.setWidth_eq]
  · show (⟨BitVec.ofInt w a.bits.toInt⟩ : IntV true w) = a
    rw [BitVec.ofInt_toInt]

/-- Rebuilding a nonnegative integer from its value as a natural number gives it back. -/
theorem ofNat_toInt_toNat {a : IntV s w} (h : 0 ≤ a.toInt) :
    (IntV.ofNat a.toInt.toNat : IntV s w) = a := by
  rw [← ofInt_natCast, Int.toNat_of_nonneg h, ofInt_toInt]

/-- `int64` constants in range keep their value. -/
theorem I64.toInt_ofInt {x : Int} (h : -(2 ^ 63 : Int) ≤ x ∧ x < 2 ^ 63) :
    (IntV.ofInt x : I64).toInt = x := IntV.toInt_ofInt (by norm_num) h

/-- An `int64` constant `IntV.ofNat n` with `n < 2^63` has value `n`. -/
theorem I64.toInt_ofNat {n : Nat} (h : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = n :=
  toInt_ofNat_of_lt h

/-- An `int64` constant `IntV.ofNat n` with `n < 2^63` has value `n` (the name `I32.ofNat_nat`
uses for `int32`; the same statement as `I64.toInt_ofNat`). -/
theorem I64.ofNat_nat {n : Nat} (h : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = (n : Int) :=
  toInt_ofNat_of_lt h

/-! ## Wrapping arithmetic of constants

`+ - *` wrap modulo `2^w`, and so does building a constant, so these hold with no range premise.
Read left to right they fold constant arithmetic into one constant, whose value
`toInt_ofNat_of_lt` then gives. -/

/-- The sum of the constants `a` and `b` (`IntV.ofNat`) is the constant `a + b`, in any integer
type (both sides are `a + b` modulo `2^w`). -/
@[simp] theorem ofNat_add_ofNat (a b : Nat) :
    (IntV.ofNat a + IntV.ofNat b : IntV s w) = IntV.ofNat (a + b) :=
  congrArg IntV.mk (BitVec.ofNat_add_ofNat a b)

/-- The product of the constants `a` and `b` (`IntV.ofNat`) is the constant `a * b`, in any
integer type (both sides are `a * b` modulo `2^w`). -/
@[simp] theorem ofNat_mul_ofNat (a b : Nat) :
    (IntV.ofNat a * IntV.ofNat b : IntV s w) = IntV.ofNat (a * b) :=
  congrArg IntV.mk (BitVec.ofNat_mul_ofNat a b)

/-- The sum of the constants `a` and `b` (`IntV.ofInt`) is the constant `a + b`, in any integer
type. -/
@[simp] theorem ofInt_add_ofInt (a b : Int) :
    (IntV.ofInt a + IntV.ofInt b : IntV s w) = IntV.ofInt (a + b) :=
  congrArg IntV.mk (BitVec.ofInt_add a b).symm

/-- The product of the constants `a` and `b` (`IntV.ofInt`) is the constant `a * b`, in any
integer type. -/
@[simp] theorem ofInt_mul_ofInt (a b : Int) :
    (IntV.ofInt a * IntV.ofInt b : IntV s w) = IntV.ofInt (a * b) :=
  congrArg IntV.mk (BitVec.ofInt_mul a b).symm

/-- The negation of the constant `a` (`IntV.ofInt`) is the constant `-a`, in any integer type. -/
@[simp] theorem neg_ofInt (a : Int) : (-(IntV.ofInt a) : IntV s w) = IntV.ofInt (-a) :=
  congrArg IntV.mk BitVec.ofInt_neg.symm

/-- The difference of the constants `a` and `b` (`IntV.ofInt`) is the constant `a - b`, in any
integer type. -/
@[simp] theorem ofInt_sub_ofInt (a b : Int) :
    (IntV.ofInt a - IntV.ofInt b : IntV s w) = IntV.ofInt (a - b) := by
  show (⟨BitVec.ofInt w a - BitVec.ofInt w b⟩ : IntV s w) = ⟨BitVec.ofInt w (a - b)⟩
  rw [sub_eq_add_neg a b, BitVec.ofInt_add, BitVec.ofInt_neg, BitVec.sub_eq_add_neg]

/-! ## Equality -/

/-- Two integers of one type with the same value are equal. -/
theorem ext_toInt {a b : IntV s w} (h : a.toInt = b.toInt) : a = b := by
  cases s
  · simp only [toInt, Bool.false_eq_true, ite_false, Int.natCast_inj] at h
    exact ext_bits (BitVec.eq_of_toNat_eq h)
  · simp only [toInt, ite_true] at h
    exact ext_bits (BitVec.toInt_inj.mp h)

/-- Two integers of one type are equal exactly when their values are. -/
theorem toInt_inj {a b : IntV s w} : a.toInt = b.toInt ↔ a = b :=
  ⟨ext_toInt, fun h => h ▸ rfl⟩

/-- `==` on integers compares values. -/
theorem eq_eq (a b : IntV s w) : a.eq b = decide (a.toInt = b.toInt) := by
  have h : a.eq b = decide (a = b) := by
    by_cases hab : a = b
    · subst hab; simp [IntV.eq]
    · have hne : a.bits ≠ b.bits := fun e => hab (ext_bits e)
      simp [IntV.eq, hab, hne]
  rw [h]; simp only [toInt_inj]

/-- `!=` on integers (written `!(a.eq b)`) compares values. -/
theorem not_eq_eq (a b : IntV s w) : (!a.eq b) = decide (a.toInt ≠ b.toInt) := by
  rw [eq_eq]; simp

/-! ## Arithmetic on tracked values, any width

Each takes the values of the operands as hypotheses (`ha : a.toInt = x`), so a proof can carry
`x` and `y` from step to step. The `_nat` forms need one bound; the `_int` forms take `SFits`.
None needs `0 < w`. -/

/-- `+` of signed integers with values `x`, `y` is `x + y` when that fits. -/
theorem add_int {a b : IntV true w} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : SFits w (x + y)) : (a + b).toInt = x + y := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · simp only [toInt_width_zero] at ha hb ⊢; omega
  · subst ha hb; exact toInt_add hw h

/-- `-` of signed integers with values `x`, `y` is `x - y` when that fits. -/
theorem sub_int {a b : IntV true w} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : SFits w (x - y)) : (a - b).toInt = x - y := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · simp only [toInt_width_zero] at ha hb ⊢; omega
  · subst ha hb; exact toInt_sub hw h

/-- `*` of signed integers with values `x`, `y` is `x * y` when that fits. -/
theorem mul_int {a b : IntV true w} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : SFits w (x * y)) : (a * b).toInt = x * y := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · simp only [toInt_width_zero] at ha hb ⊢; subst ha hb; simp
  · subst ha hb; exact toInt_mul hw h

/-- `+` of signed integers with natural values `x`, `y` is `x + y` when `x + y < 2^(w-1)`. -/
theorem add_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x + y < 2 ^ (w - 1)) : (a + b).toInt = ((x + y : Nat) : Int) := by
  rw [add_int ha hb (sfits_nat_add h)]; norm_cast

/-- `*` of signed integers with natural values `x`, `y` is `x * y` when `x * y < 2^(w-1)`. -/
theorem mul_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x * y < 2 ^ (w - 1)) : (a * b).toInt = ((x * y : Nat) : Int) := by
  rw [mul_int ha hb (sfits_nat_mul h)]; norm_cast

/-- `-` of signed integers with natural values `x ≥ y` is `x - y` when `x < 2^(w-1)`. -/
theorem sub_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (hxy : y ≤ x) (h : x < 2 ^ (w - 1)) : (a - b).toInt = ((x - y : Nat) : Int) := by
  have hs : SFits w ((x : Int) - y) := by
    have := sfits_natCast (w := w) (n := x - y) (by omega)
    push_cast [hxy] at this
    exact this
  rw [sub_int ha hb hs]; push_cast [hxy]; rfl

/-- `+` of unsigned integers with values `x`, `y` is `x + y` when `x + y < 2^w`. -/
theorem add_nat_unsigned {a b : IntV false w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x + y < 2 ^ w) : (a + b).toInt = ((x + y : Nat) : Int) := by
  simp only [toInt, Bool.false_eq_true, ite_false, Int.natCast_inj] at ha hb ⊢
  show (a.bits + b.bits).toNat = _
  rw [BitVec.toNat_add, ha, hb, Nat.mod_eq_of_lt h]

/-- `*` of unsigned integers with values `x`, `y` is `x * y` when `x * y < 2^w`. -/
theorem mul_nat_unsigned {a b : IntV false w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x * y < 2 ^ w) : (a * b).toInt = ((x * y : Nat) : Int) := by
  simp only [toInt, Bool.false_eq_true, ite_false, Int.natCast_inj] at ha hb ⊢
  show (a.bits * b.bits).toNat = _
  rw [BitVec.toNat_mul, ha, hb, Nat.mod_eq_of_lt h]

/-- `-` of unsigned integers with values `x ≥ y` is `x - y`. -/
theorem sub_nat_unsigned {a b : IntV false w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (hxy : y ≤ x) : (a - b).toInt = ((x - y : Nat) : Int) := by
  simp only [toInt, Bool.false_eq_true, ite_false, Int.natCast_inj] at ha hb ⊢
  show (a.bits - b.bits).toNat = _
  have hx : x < 2 ^ w := ha ▸ a.bits.isLt
  rw [BitVec.toNat_sub, ha, hb]
  rw [show 2 ^ w - y + x = (x - y) + 2 ^ w by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]

/-- `a + n` for a constant `n` (`IntV.ofNat n`, or a numeral such as `(64 : I32)`) and a signed
`a` of natural value `x`: `x + n`, when `x + n < 2^(w-1)`. -/
theorem add_ofNat {a : IntV true w} {x n : Nat} (ha : a.toInt = x) (h : x + n < 2 ^ (w - 1)) :
    (a + IntV.ofNat n).toInt = ((x + n : Nat) : Int) :=
  add_nat ha (toInt_ofNat_of_lt (by omega)) h

/-- `a * n` for a constant `n` (`IntV.ofNat n`, or a numeral such as `(128 : I32)`) and a signed
`a` of natural value `x`: `x * n`, when `x * n < 2^(w-1)`. (`n` need not fit by itself when
`x = 0`.) -/
theorem mul_ofNat {a : IntV true w} {x n : Nat} (ha : a.toInt = x) (h : x * n < 2 ^ (w - 1)) :
    (a * IntV.ofNat n).toInt = ((x * n : Nat) : Int) := by
  rcases Nat.eq_zero_or_pos x with rfl | hx
  · have h0 : SFits w (((0 : Nat) : Int) * (IntV.ofNat n : IntV true w).toInt) := by
      rw [Nat.cast_zero, zero_mul]; exact sfits_of_nonneg le_rfl (by positivity)
    rw [mul_int ha rfl h0]; simp
  · exact mul_nat ha (toInt_ofNat_of_lt (lt_of_le_of_lt (Nat.le_mul_of_pos_left n hx) h)) h

/-- `n * a` for a constant `n` and a signed `a` of natural value `x`: `n * x`, when
`n * x < 2^(w-1)`. -/
theorem ofNat_mul {a : IntV true w} {x n : Nat} (ha : a.toInt = x) (h : n * x < 2 ^ (w - 1)) :
    (IntV.ofNat n * a).toInt = ((n * x : Nat) : Int) := by
  rcases Nat.eq_zero_or_pos x with rfl | hx
  · have h0 : SFits w ((IntV.ofNat n : IntV true w).toInt * ((0 : Nat) : Int)) := by
      rw [Nat.cast_zero, mul_zero]; exact sfits_of_nonneg le_rfl (by positivity)
    rw [mul_int rfl ha h0]; simp
  · exact mul_nat (toInt_ofNat_of_lt (lt_of_le_of_lt (Nat.le_mul_of_pos_right n hx) h)) ha h

/-- `a - n` for a constant `n ≤ x` and a signed `a` of natural value `x < 2^(w-1)`: `x - n`. -/
theorem sub_ofNat {a : IntV true w} {x n : Nat} (ha : a.toInt = x) (hnx : n ≤ x)
    (h : x < 2 ^ (w - 1)) : (a - IntV.ofNat n).toInt = ((x - n : Nat) : Int) :=
  sub_nat ha (toInt_ofNat_of_lt (by omega)) hnx h

/-- `int64` `+` of natural values: `x + y`, when `x + y < 2^63`. -/
theorem I64.add_nat {a b : I64} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x + y < 2 ^ 63) : (a + b).toInt = ((x + y : Nat) : Int) := IntV.add_nat ha hb h

/-- `int64` `*` of natural values: `x * y`, when `x * y < 2^63`. -/
theorem I64.mul_nat {a b : I64} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x * y < 2 ^ 63) : (a * b).toInt = ((x * y : Nat) : Int) := IntV.mul_nat ha hb h

/-- `int64` `-` of natural values `x ≥ y` with `x < 2^63`: `x - y`. -/
theorem I64.sub_nat {a b : I64} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y) (hxy : y ≤ x)
    (h : x < 2 ^ 63) : (a - b).toInt = ((x - y : Nat) : Int) := IntV.sub_nat ha hb hxy h

/-- `int32` `+` is the integer sum when the sum is in `[-2^31, 2^31)`. -/
theorem I32.toInt_add {a b : I32} (h0 : -(2 ^ 31 : Int) ≤ a.toInt + b.toInt)
    (h1 : a.toInt + b.toInt < 2 ^ 31) : (a + b).toInt = a.toInt + b.toInt :=
  IntV.toInt_add (by norm_num) ⟨h0, h1⟩

/-- `int32` `-` is the integer difference when the difference is in `[-2^31, 2^31)`. -/
theorem I32.toInt_sub {a b : I32} (h0 : -(2 ^ 31 : Int) ≤ a.toInt - b.toInt)
    (h1 : a.toInt - b.toInt < 2 ^ 31) : (a - b).toInt = a.toInt - b.toInt :=
  IntV.toInt_sub (by norm_num) ⟨h0, h1⟩

/-- `int32` `*` is the integer product when the product is in `[-2^31, 2^31)`. -/
theorem I32.toInt_mul {a b : I32} (h0 : -(2 ^ 31 : Int) ≤ a.toInt * b.toInt)
    (h1 : a.toInt * b.toInt < 2 ^ 31) : (a * b).toInt = a.toInt * b.toInt :=
  IntV.toInt_mul (by norm_num) ⟨h0, h1⟩

/-- `int64` `+` is the integer sum when the sum is in `[-2^63, 2^63)`. -/
theorem I64.toInt_add {a b : I64} (h0 : -(2 ^ 63 : Int) ≤ a.toInt + b.toInt)
    (h1 : a.toInt + b.toInt < 2 ^ 63) : (a + b).toInt = a.toInt + b.toInt :=
  IntV.toInt_add (by norm_num) ⟨h0, h1⟩

/-- `int64` `-` is the integer difference when the difference is in `[-2^63, 2^63)`. -/
theorem I64.toInt_sub {a b : I64} (h0 : -(2 ^ 63 : Int) ≤ a.toInt - b.toInt)
    (h1 : a.toInt - b.toInt < 2 ^ 63) : (a - b).toInt = a.toInt - b.toInt :=
  IntV.toInt_sub (by norm_num) ⟨h0, h1⟩

/-- `int64` `*` is the integer product when the product is in `[-2^63, 2^63)`. -/
theorem I64.toInt_mul {a b : I64} (h0 : -(2 ^ 63 : Int) ≤ a.toInt * b.toInt)
    (h1 : a.toInt * b.toInt < 2 ^ 63) : (a * b).toInt = a.toInt * b.toInt :=
  IntV.toInt_mul (by norm_num) ⟨h0, h1⟩

/-! ## Division of naturals -/

/-- `//` of signed integers with natural values `x`, `y > 0`: defined, with value `x / y`. -/
theorem div?_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (hy : 0 < y) :
    ∃ q, a.div? b = some q ∧ q.toInt = ((x / y : Nat) : Int) := by
  obtain ⟨q, hq, hv⟩ := div?_of_nonneg (a := a) (b := b) (by rw [ha]; positivity)
    (by rw [hb]; exact_mod_cast hy)
  exact ⟨q, hq, by rw [hv, ha, hb]; norm_cast⟩

/-- `%` of signed integers with natural values `x`, `y > 0`: defined, with value `x % y`. -/
theorem rem?_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (hy : 0 < y) :
    ∃ q, a.rem? b = some q ∧ q.toInt = ((x % y : Nat) : Int) := by
  obtain ⟨q, hq, hv⟩ := rem?_of_nonneg (a := a) (b := b) (by rw [ha]; positivity)
    (by rw [hb]; exact_mod_cast hy)
  exact ⟨q, hq, by rw [hv, ha, hb]; norm_cast⟩

/-- `//` of constants `n`, `m > 0` in range: the constant `n / m`. -/
theorem div?_ofNat {n m : Nat} (hn : n < 2 ^ (w - 1)) (hm0 : 0 < m) (hm : m < 2 ^ (w - 1)) :
    (IntV.ofNat n : IntV true w).div? (IntV.ofNat m) = some (IntV.ofNat (n / m)) := by
  obtain ⟨q, hq, hv⟩ := div?_nat (toInt_ofNat_of_lt hn) (toInt_ofNat_of_lt hm) hm0
  rw [hq, ext_toInt (hv.trans (toInt_ofNat_of_lt (lt_of_le_of_lt (Nat.div_le_self n m) hn)).symm)]

/-- `%` of constants `n`, `m > 0` in range: the constant `n % m`. -/
theorem rem?_ofNat {n m : Nat} (hn : n < 2 ^ (w - 1)) (hm0 : 0 < m) (hm : m < 2 ^ (w - 1)) :
    (IntV.ofNat n : IntV true w).rem? (IntV.ofNat m) = some (IntV.ofNat (n % m)) := by
  obtain ⟨q, hq, hv⟩ := rem?_nat (toInt_ofNat_of_lt hn) (toInt_ofNat_of_lt hm) hm0
  rw [hq, ext_toInt (hv.trans (toInt_ofNat_of_lt (lt_trans (Nat.mod_lt n hm0) hm)).symm)]

/-- A result `q` of `//` (`a.div? b = some q`, as `tl.idiv` gives for each lane) of a nonnegative
`a` by a positive `b` has value `a.toInt / b.toInt`. -/
theorem toInt_of_div? {a b q : IntV true w} (h : a.div? b = some q) (ha : 0 ≤ a.toInt)
    (hb : 0 < b.toInt) : q.toInt = a.toInt / b.toInt := by
  obtain ⟨q', hq', hv⟩ := div?_of_nonneg ha hb
  rw [hq'] at h
  obtain rfl := Option.some.inj h
  exact hv

/-- A result `q` of `%` (`a.rem? b = some q`, as `tl.irem` gives for each lane) of a nonnegative
`a` by a positive `b` has value `a.toInt % b.toInt`. -/
theorem toInt_of_rem? {a b q : IntV true w} (h : a.rem? b = some q) (ha : 0 ≤ a.toInt)
    (hb : 0 < b.toInt) : q.toInt = a.toInt % b.toInt := by
  obtain ⟨q', hq', hv⟩ := rem?_of_nonneg ha hb
  rw [hq'] at h
  obtain rfl := Option.some.inj h
  exact hv

/-- A result `q` of `//` of signed integers with natural values `x`, `y > 0` has value `x / y`. -/
theorem toInt_of_div?_nat {a b q : IntV true w} {x y : Nat} (h : a.div? b = some q)
    (ha : a.toInt = x) (hb : b.toInt = y) (hy : 0 < y) : q.toInt = ((x / y : Nat) : Int) := by
  rw [toInt_of_div? h (by rw [ha]; positivity) (by rw [hb]; exact_mod_cast hy), ha, hb]
  norm_cast

/-- A result `q` of `%` of signed integers with natural values `x`, `y > 0` has value `x % y`. -/
theorem toInt_of_rem?_nat {a b q : IntV true w} {x y : Nat} (h : a.rem? b = some q)
    (ha : a.toInt = x) (hb : b.toInt = y) (hy : 0 < y) : q.toInt = ((x % y : Nat) : Int) := by
  rw [toInt_of_rem? h (by rw [ha]; positivity) (by rw [hb]; exact_mod_cast hy), ha, hb]
  norm_cast

/-! ## Rebuilding a value in a wider type -/

/-- The value of a signed integer, rebuilt in a signed type at least as wide, is unchanged. -/
theorem toInt_ofInt_toInt {w' : Nat} (a : IntV true w) (h : w ≤ w') :
    (IntV.ofInt a.toInt : IntV true w').toInt = a.toInt := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · rw [toInt_width_zero]; exact toInt_ofInt_zero
  · exact toInt_ofInt (by omega) ((sfits_toInt a).mono h)

/-- The value of an unsigned integer, rebuilt in a strictly wider signed type, is unchanged. -/
theorem toInt_ofInt_toInt_unsigned {w' : Nat} (a : IntV false w) (h : w < w') :
    (IntV.ofInt a.toInt : IntV true w').toInt = a.toInt := by
  have ha : a.toInt = (a.bits.toNat : Int) := rfl
  rw [ha, ofInt_natCast]
  exact toInt_ofNat_of_lt
    (lt_of_lt_of_le a.bits.isLt (Nat.pow_le_pow_right (by norm_num) (by omega)))

/-- The value of an unsigned integer, rebuilt in an unsigned type at least as wide, is
unchanged. -/
theorem toInt_ofInt_toInt_unsigned_le {w' : Nat} (a : IntV false w) (h : w ≤ w') :
    (IntV.ofInt a.toInt : IntV false w').toInt = a.toInt := by
  have ha : a.toInt = (a.bits.toNat : Int) := rfl
  rw [ha, ofInt_natCast]
  exact toInt_ofNat_unsigned (lt_of_lt_of_le a.bits.isLt (Nat.pow_le_pow_right (by norm_num) h))

/-! ## `int32` lanes of `tl.arange` -/

/-- Lane `i` of `Tensor.arange start n` (`tl.arange`, `int32`) has value `start + i` when the
whole range is in `int32`. (`x.get1 i` is `x.get (i, ())` by definition.) -/
theorem I32.toInt_arange {start : Int} {n : Nat} (hlo : -(2 ^ 31 : Int) ≤ start)
    (hhi : start + n ≤ 2 ^ 31) (i : Fin n) :
    ((Tensor.arange start n).get1 i).toInt = start + i := by
  rw [Tensor.get1_arange]
  have := i.isLt
  exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩

/-- Lane `i` of `tl.arange(0, n)` has value `i` when `n ≤ 2^31`. -/
theorem I32.toInt_tl_arange {n : Nat} (hn : n ≤ 2 ^ 31) (i : Fin (n - 0)) :
    ((tl.arange 0 n).get1 i).toInt = (i : Nat) := by
  have := I32.toInt_arange (start := ((0 : Nat) : Int)) (n := n - 0) (by norm_num)
    (by push_cast; omega) i
  exact this.trans (by simp)

/-- Lane `i` of `c + tl.arange(0, n)`, for an `int32` scalar `c` of natural value `s`, has value
`s + i` when `s + n ≤ 2^31`. -/
theorem I32.toInt_sadd_arange {c : I32} {s n : Nat} (hc : c.toInt = s) (h : s + n ≤ 2 ^ 31)
    (i : Fin (n - 0)) : ((c + tl.arange 0 n).get1 i).toInt = ((s + i : Nat) : Int) := by
  have hi := i.isLt
  have hl := I32.toInt_tl_arange (n := n) (by omega) i
  exact IntV.add_nat hc hl (by simp only [Nat.sub_zero] at hi; norm_num; omega)

end IntV

/-! ## Lanes of integer vectors

`x.get1 i` is `x.get (i, ())`. The `Tensor.get_*` rewrites of `TVA.Lang.Tensor` are stated for
`get`; these are their `get1` forms, for a rank-1 tensor, and the lane values of `+ - *` from
tracked lane values `f i`, `g i`. -/

namespace Tensor

variable {n : Nat} {sg : Bool} {w : Nat}

/-- Lane `i` of the sum of integer vectors is the sum of their lanes `i`. -/
@[simp] theorem get1_add (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x + y).get1 i = x.get1 i + y.get1 i := rfl

/-- Lane `i` of the difference of integer vectors is the difference of their lanes `i`. -/
@[simp] theorem get1_sub (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x - y).get1 i = x.get1 i - y.get1 i := rfl

/-- Lane `i` of the product of integer vectors is the product of their lanes `i`. -/
@[simp] theorem get1_mul (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x * y).get1 i = x.get1 i * y.get1 i := rfl

/-- Lane `i` of `-x` is `-(x.get1 i)`. -/
@[simp] theorem get1_neg (x : Tensor [n] (IntV sg w)) (i : Fin n) : (-x).get1 i = -x.get1 i := rfl

/-- Lane `i` of `a + x`, for a scalar `a`, is `a + x.get1 i`. -/
@[simp] theorem get1_sadd (a : IntV sg w) (x : Tensor [n] (IntV sg w)) (i : Fin n) :
    (a + x).get1 i = a + x.get1 i := rfl

/-- Lane `i` of `x + a`, for a scalar `a`, is `x.get1 i + a`. -/
@[simp] theorem get1_adds (x : Tensor [n] (IntV sg w)) (a : IntV sg w) (i : Fin n) :
    (x + a).get1 i = x.get1 i + a := rfl

/-- Lane `i` of `a * x`, for a scalar `a`, is `a * x.get1 i`. -/
@[simp] theorem get1_smul (a : IntV sg w) (x : Tensor [n] (IntV sg w)) (i : Fin n) :
    (a * x).get1 i = a * x.get1 i := rfl

/-- Lane `i` of `x * a`, for a scalar `a`, is `x.get1 i * a`. -/
@[simp] theorem get1_muls (x : Tensor [n] (IntV sg w)) (a : IntV sg w) (i : Fin n) :
    (x * a).get1 i = x.get1 i * a := rfl

/-- Lane `i` of `x < a`, for a scalar `a` (a mask such as `offs < n`), is `(x.get1 i).lt a`. -/
@[simp] theorem get1_ltS (x : Tensor [n] (IntV sg w)) (a : IntV sg w) (i : Fin n) :
    (x.ltS a).get1 i = (x.get1 i).lt a := rfl

/-- Lane `i` of `x < y` is `(x.get1 i).lt (y.get1 i)`. -/
@[simp] theorem get1_lt (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x.lt y).get1 i = (x.get1 i).lt (y.get1 i) := rfl

/-- Lane `i` of `x <= y` is `(x.get1 i).le (y.get1 i)`. -/
@[simp] theorem get1_le (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x.le y).get1 i = (x.get1 i).le (y.get1 i) := rfl

/-- Lane `i` of `x == y` is `(x.get1 i).eq (y.get1 i)`. -/
@[simp] theorem get1_eq (x y : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x.eq y).get1 i = (x.get1 i).eq (y.get1 i) := rfl

/-- Lane `i` of `x.castInt` (the model's own widening step, as in a reduction of a narrow integer
type or the entries of `tl.make_block_ptr`; not `x.to(tl.int64)` or an implicit promotion, which are `tl.convert`, with
`ConvRel.toInt_i32_i64` and the other pairs) is the cast of lane `i`. -/
@[simp] theorem get1_castInt {sg' : Bool} {w' : Nat} (x : Tensor [n] (IntV sg w)) (i : Fin n) :
    (x.castInt : Tensor [n] (IntV sg' w')).get1 i = IntV.cast (x.get1 i) := rfl

/-- Lane `i` of `x + y`, for signed vectors whose lanes have values `f i` and `g i`, has value
`f i + g i` when that fits. -/
theorem toInt_get1_add {x y : Tensor [n] (IntV true w)} {f g : Fin n → Int}
    (hx : ∀ i, (x.get1 i).toInt = f i) (hy : ∀ i, (y.get1 i).toInt = g i)
    (h : ∀ i, IntV.SFits w (f i + g i)) (i : Fin n) : ((x + y).get1 i).toInt = f i + g i := by
  rw [get1_add]; exact IntV.add_int (hx i) (hy i) (h i)

/-- Lane `i` of `x - y`, for signed vectors whose lanes have values `f i` and `g i`, has value
`f i - g i` when that fits. -/
theorem toInt_get1_sub {x y : Tensor [n] (IntV true w)} {f g : Fin n → Int}
    (hx : ∀ i, (x.get1 i).toInt = f i) (hy : ∀ i, (y.get1 i).toInt = g i)
    (h : ∀ i, IntV.SFits w (f i - g i)) (i : Fin n) : ((x - y).get1 i).toInt = f i - g i := by
  rw [get1_sub]; exact IntV.sub_int (hx i) (hy i) (h i)

/-- Lane `i` of `x * y`, for signed vectors whose lanes have values `f i` and `g i`, has value
`f i * g i` when that fits. -/
theorem toInt_get1_mul {x y : Tensor [n] (IntV true w)} {f g : Fin n → Int}
    (hx : ∀ i, (x.get1 i).toInt = f i) (hy : ∀ i, (y.get1 i).toInt = g i)
    (h : ∀ i, IntV.SFits w (f i * g i)) (i : Fin n) : ((x * y).get1 i).toInt = f i * g i := by
  rw [get1_mul]; exact IntV.mul_int (hx i) (hy i) (h i)

/-- Lane `i` of `a + x`, for a signed scalar `a` of value `c` and a vector whose lanes have values
`f i`, has value `c + f i` when that fits. -/
theorem toInt_get1_sadd {a : IntV true w} {x : Tensor [n] (IntV true w)} {c : Int}
    {f : Fin n → Int} (ha : a.toInt = c) (hx : ∀ i, (x.get1 i).toInt = f i)
    (h : ∀ i, IntV.SFits w (c + f i)) (i : Fin n) : ((a + x).get1 i).toInt = c + f i := by
  rw [get1_sadd]; exact IntV.add_int ha (hx i) (h i)

/-- Lane `i` of `x * a`, for a signed scalar `a` of value `c` and a vector whose lanes have values
`f i`, has value `f i * c` when that fits (`offs * stride`). -/
theorem toInt_get1_muls {a : IntV true w} {x : Tensor [n] (IntV true w)} {c : Int}
    {f : Fin n → Int} (ha : a.toInt = c) (hx : ∀ i, (x.get1 i).toInt = f i)
    (h : ∀ i, IntV.SFits w (f i * c)) (i : Fin n) : ((x * a).get1 i).toInt = f i * c := by
  rw [get1_muls]; exact IntV.mul_int (hx i) ha (h i)

end Tensor

/-! ## Conversions between integer dtypes

`ConvRel P d d' x r` for integer dtypes says `r = IntV.ofInt x.toInt` (the value modulo `2^w'`;
`int1` gives `0` or `1`). A conversion into a type that holds every value of the source keeps the
value; a narrowing one keeps it when it is in range. -/

namespace ConvRel

variable {P : Platform}

/-- `x.to(tl.int64)` of an `int32` keeps the value (a widening conversion). -/
theorem toInt_i32_i64 {x : I32} {r : I64} (h : ConvRel P .i32 .i64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt x (by norm_num)

/-- `x.to(tl.int32)` of an `int32` (no change of type) keeps the value. -/
theorem toInt_i32_i32 {x : I32} {r : I32} (h : ConvRel P .i32 .i32 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt x le_rfl

/-- `x.to(tl.int64)` of an `int64` (no change of type) keeps the value. -/
theorem toInt_i64_i64 {x : I64} {r : I64} (h : ConvRel P .i64 .i64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt x le_rfl

/-- `x.to(tl.int16)` of an `int8` keeps the value (a widening conversion). -/
theorem toInt_i8_i16 {x : I8} {r : I16} (h : ConvRel P .i8 .i16 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt x (by norm_num)

/-- `x.to(tl.int32)` of an `int8` keeps the value (a widening conversion). -/
theorem toInt_i8_i32 {x : I8} {r : I32} (h : ConvRel P .i8 .i32 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt x (by norm_num)

/-- `x.to(tl.int32)` of an `int16` keeps the value (a widening conversion). -/
theorem toInt_i16_i32 {x : I16} {r : I32} (h : ConvRel P .i16 .i32 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt x (by norm_num)

/-- `x.to(tl.int64)` of an `int8` keeps the value (a widening conversion). -/
theorem toInt_i8_i64 {x : I8} {r : I64} (h : ConvRel P .i8 .i64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt x (by norm_num)

/-- `x.to(tl.int64)` of an `int16` keeps the value (a widening conversion). -/
theorem toInt_i16_i64 {x : I16} {r : I64} (h : ConvRel P .i16 .i64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt x (by norm_num)

/-- `x.to(tl.int16)` of a `uint8` keeps the value (a widening conversion). -/
theorem toInt_u8_i16 {x : U8} {r : I16} (h : ConvRel P .u8 .i16 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned x (by norm_num)

/-- `x.to(tl.int32)` of a `uint8` keeps the value (a widening conversion). -/
theorem toInt_u8_i32 {x : U8} {r : I32} (h : ConvRel P .u8 .i32 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned x (by norm_num)

/-- `x.to(tl.int32)` of a `uint16` keeps the value (a widening conversion). -/
theorem toInt_u16_i32 {x : U16} {r : I32} (h : ConvRel P .u16 .i32 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned x (by norm_num)

/-- `x.to(tl.int64)` of a `uint8` keeps the value (a widening conversion). -/
theorem toInt_u8_i64 {x : U8} {r : I64} (h : ConvRel P .u8 .i64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned x (by norm_num)

/-- `x.to(tl.int64)` of a `uint16` keeps the value (a widening conversion). -/
theorem toInt_u16_i64 {x : U16} {r : I64} (h : ConvRel P .u16 .i64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned x (by norm_num)

/-- `x.to(tl.int64)` of a `uint32` keeps the value (a widening conversion). -/
theorem toInt_u32_i64 {x : U32} {r : I64} (h : ConvRel P .u32 .i64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned x (by norm_num)

/-- `x.to(tl.uint16)` of a `uint8` keeps the value (a widening conversion). -/
theorem toInt_u8_u16 {x : U8} {r : U16} (h : ConvRel P .u8 .u16 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned_le x (by norm_num)

/-- `x.to(tl.uint32)` of a `uint8` keeps the value (a widening conversion). -/
theorem toInt_u8_u32 {x : U8} {r : U32} (h : ConvRel P .u8 .u32 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned_le x (by norm_num)

/-- `x.to(tl.uint32)` of a `uint16` keeps the value (a widening conversion). -/
theorem toInt_u16_u32 {x : U16} {r : U32} (h : ConvRel P .u16 .u32 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned_le x (by norm_num)

/-- `x.to(tl.uint64)` of a `uint8` keeps the value (a widening conversion). -/
theorem toInt_u8_u64 {x : U8} {r : U64} (h : ConvRel P .u8 .u64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned_le x (by norm_num)

/-- `x.to(tl.uint64)` of a `uint16` keeps the value (a widening conversion). -/
theorem toInt_u16_u64 {x : U16} {r : U64} (h : ConvRel P .u16 .u64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned_le x (by norm_num)

/-- `x.to(tl.uint64)` of a `uint32` keeps the value (a widening conversion). -/
theorem toInt_u32_u64 {x : U32} {r : U64} (h : ConvRel P .u32 .u64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned_le x (by norm_num)

/-- `x.to(tl.uint32)` of a `uint32` (no change of type) keeps the value. -/
theorem toInt_u32_u32 {x : U32} {r : U32} (h : ConvRel P .u32 .u32 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned_le x le_rfl

/-- `x.to(tl.uint64)` of a `uint64` (no change of type) keeps the value. -/
theorem toInt_u64_u64 {x : U64} {r : U64} (h : ConvRel P .u64 .u64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt_toInt_unsigned_le x le_rfl

/-- `x.to(tl.int32)` of an `int64` keeps the value when it is in `int32`'s range. -/
theorem toInt_i64_i32 {x : I64} {r : I32} (h : ConvRel P .i64 .i32 x r)
    (hx : IntV.SFits 32 x.toInt) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact IntV.toInt_ofInt (by norm_num) hx

/-- An `int1` converted to `int32` is `1` for `true` and `0` for `false`. -/
theorem toInt_i1_i32 {x : Bool} {r : I32} (h : ConvRel P .i1 .i32 x r) :
    r.toInt = if x then 1 else 0 := by
  have h' : r = IntV.ofInt (if x then 1 else 0) := h
  subst h'
  cases x
  · exact IntV.toInt_ofInt_zero
  · exact IntV.toInt_ofInt_one (by norm_num)

/-- An `int1` converted to `int64` is `1` for `true` and `0` for `false`. -/
theorem toInt_i1_i64 {x : Bool} {r : I64} (h : ConvRel P .i1 .i64 x r) :
    r.toInt = if x then 1 else 0 := by
  have h' : r = IntV.ofInt (if x then 1 else 0) := h
  subst h'
  cases x
  · exact IntV.toInt_ofInt_zero
  · exact IntV.toInt_ofInt_one (by norm_num)

/-- Every `int32` has a conversion to `int64` (the program does not stop there). -/
theorem exists_i32_i64 (x : I32) : ∃ r : I64, ConvRel P .i32 .i64 x r := ⟨IntV.ofInt x.toInt, rfl⟩

/-- Every `int64` has a conversion to `int64` (`x.to(tl.int64)` of an `int64` does not stop the
program). -/
theorem exists_i64_i64 (x : I64) : ∃ r : I64, ConvRel P .i64 .i64 x r := ⟨IntV.ofInt x.toInt, rfl⟩

/-- Every `int64` has a conversion to `int32` (the value modulo `2^32`; `toInt_i64_i32` gives it
when in range). -/
theorem exists_i64_i32 (x : I64) : ∃ r : I32, ConvRel P .i64 .i32 x r := ⟨IntV.ofInt x.toInt, rfl⟩

end ConvRel

end TVA
