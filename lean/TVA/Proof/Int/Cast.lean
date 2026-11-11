import TVA.Proof.Int.Basic
import TVA.Triton

/-!
# Integer casts and conversions

`x.to(tl.intW)` between integer dtypes other than `int1` sign-extends a signed source and
zero-extends an unsigned one, then truncates (`IntV.cast`, and `Tensor.castInt` on tensors). The
result is the source's value modulo `2^w'` (`cast_eq_ofInt`), so it keeps the value whenever the
target type holds it:

* `toInt_cast_of_sfits` (into a signed type), `toInt_cast_of_lt` (into an unsigned type): the
  value is kept when it is in the target's range;
* `toInt_cast` (signed into a signed type at least as wide), `toInt_cast_unsigned` (unsigned into
  an unsigned type at least as wide), `toInt_cast_unsigned_signed` (unsigned into a strictly wider
  signed type): widening keeps the value.

The model writes such a conversion either as `tl.convert d d' x` (each element as
`ConvRel P d d'` allows) or as `x.castInt`. Between integer dtypes the two agree:
`ConvRel.iff_cast_<d>_<d'>` says `ConvRel P d d' x r ↔ r = IntV.cast x`, for each of the 64 pairs
of `int8` … `int64`, `uint8` … `uint64`. With the rule for `tl.convert` in `TVA.Proof` it turns
`tl.convert .i32 .i64 x >>= k` into `k x.castInt`. `ConvRel.toInt_i32_i64` and
`ConvRel.toInt_i64_i32` give the value of a result directly; `ConvRel.toInt_i1_i32` and
`ConvRel.toInt_i1_i64` the value of a converted `int1` (`0` or `1`).
-/
namespace TVA

namespace IntV

variable {s : Bool} {w : Nat}

/-- The integer cast is the source's value modulo `2^w'`: rewrite `IntV.cast a` (or a lane of
`Tensor.castInt`) into `IntV.ofInt a.toInt` to use the constant lemmas. -/
theorem cast_eq_ofInt {s' : Bool} {w' : Nat} (a : IntV s w) :
    (IntV.cast a : IntV s' w') = IntV.ofInt a.toInt := by
  cases s
  · show IntV.mk _ = IntV.mk _
    simp only [IntV.toInt, Bool.false_eq_true, ite_false]
    congr 1
    apply BitVec.eq_of_toNat_eq
    simp [BitVec.toNat_setWidth]
  · rfl

/-- A cast into a signed type of positive width keeps a value that fits it (a narrowing
`x.to(tl.int32)` of an `int64` index known to be small). -/
theorem toInt_cast_of_sfits {w' : Nat} (a : IntV s w) (hw' : 0 < w') (h : SFits w' a.toInt) :
    (a.cast : IntV true w').toInt = a.toInt := by
  rw [cast_eq_ofInt]; exact toInt_ofInt hw' h

/-- A cast into an unsigned type keeps a value in `[0, 2^w')`. -/
theorem toInt_cast_of_lt {w' : Nat} (a : IntV s w) (h0 : 0 ≤ a.toInt) (h : a.toInt < 2 ^ w') :
    (a.cast : IntV false w').toInt = a.toInt := by
  rw [cast_eq_ofInt, ← Int.toNat_of_nonneg h0, ofInt_natCast]
  exact toInt_ofNat_unsigned (by zify; rw [Int.toNat_of_nonneg h0]; exact_mod_cast h)

/-- A widening cast of a signed integer keeps its value (`x.to(tl.int64)` of an `int32`). -/
theorem toInt_cast {w' : Nat} (a : IntV true w) (h : w ≤ w') :
    (a.cast : IntV true w').toInt = a.toInt := by
  simp only [IntV.cast, IntV.toInt, ite_true]
  exact BitVec.toInt_signExtend_of_le h

/-- A widening cast of an unsigned integer into an unsigned type keeps its value. -/
theorem toInt_cast_unsigned {w' : Nat} (a : IntV false w) (h : w ≤ w') :
    (a.cast : IntV false w').toInt = a.toInt := by
  simp only [IntV.cast, IntV.toInt, Bool.false_eq_true, ite_false]
  rw [BitVec.toNat_setWidth,
    Nat.mod_eq_of_lt (lt_of_lt_of_le a.bits.isLt (Nat.pow_le_pow_right (by norm_num) h))]

/-- A cast of an unsigned integer into a strictly wider signed type keeps its value
(`x.to(tl.int32)` of a `uint8`). -/
theorem toInt_cast_unsigned_signed {w' : Nat} (a : IntV false w) (h : w < w') :
    (a.cast : IntV true w').toInt = a.toInt := by
  have ha : a.toInt = (a.bits.toNat : Int) := rfl
  rw [cast_eq_ofInt, ha, ofInt_natCast]
  exact toInt_ofNat (lt_of_lt_of_le a.bits.isLt (Nat.pow_le_pow_right (by norm_num) (by omega)))

/-- The relation `ConvRel` gives between integer dtypes, `r = IntV.ofInt x.toInt`, is `r` being
the cast; every `ConvRel.iff_cast_*` is this. -/
theorem ofInt_iff_cast {s' : Bool} {w' : Nat} {x : IntV s w} {r : IntV s' w'} :
    r = IntV.ofInt x.toInt ↔ r = IntV.cast x := by
  rw [cast_eq_ofInt]

end IntV

namespace ConvRel

open IntV

variable {P : Platform}

/-- `x.to(tl.int64)` of an `int32` keeps its value. -/
theorem toInt_i32_i64 {x : I32} {r : I64} (h : ConvRel P .i32 .i64 x r) : r.toInt = x.toInt := by
  have h' : r = IntV.cast x := ofInt_iff_cast.mp h
  rw [h']; exact toInt_cast x (by norm_num)

/-- `x.to(tl.int32)` of an `int64` keeps its value when it fits `int32`. -/
theorem toInt_i64_i32 {x : I64} {r : I32} (h : ConvRel P .i64 .i32 x r)
    (hx : SFits 32 x.toInt) : r.toInt = x.toInt := by
  have h' : r = IntV.ofInt x.toInt := h
  rw [h']; exact toInt_ofInt (by norm_num) hx

/-- `x.to(tl.int32)` of an `int1` is `1` for `true` and `0` for `false`. -/
theorem toInt_i1_i32 {x : Bool} {r : I32} (h : ConvRel P .i1 .i32 x r) :
    r.toInt = if x then 1 else 0 := by
  have h' : r = IntV.ofInt (if x then 1 else 0) := h
  subst h'
  cases x <;> exact toInt_ofInt (by norm_num) (by norm_num [sfits_iff])

/-- `x.to(tl.int64)` of an `int1` is `1` for `true` and `0` for `false`. -/
theorem toInt_i1_i64 {x : Bool} {r : I64} (h : ConvRel P .i1 .i64 x r) :
    r.toInt = if x then 1 else 0 := by
  have h' : r = IntV.ofInt (if x then 1 else 0) := h
  subst h'
  cases x <;> exact toInt_ofInt (by norm_num) (by norm_num [sfits_iff])

/-! ### The 64 pairs of integer dtypes -/

/-- `x.to(tl.int8)` of an `int8`: its one result is `IntV.cast x`. -/
theorem iff_cast_i8_i8 {x : I8} {r : I8} : ConvRel P .i8 .i8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int16)` of an `int8`: its one result is `IntV.cast x`. -/
theorem iff_cast_i8_i16 {x : I8} {r : I16} : ConvRel P .i8 .i16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int32)` of an `int8`: its one result is `IntV.cast x`. -/
theorem iff_cast_i8_i32 {x : I8} {r : I32} : ConvRel P .i8 .i32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int64)` of an `int8`: its one result is `IntV.cast x`. -/
theorem iff_cast_i8_i64 {x : I8} {r : I64} : ConvRel P .i8 .i64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint8)` of an `int8`: its one result is `IntV.cast x`. -/
theorem iff_cast_i8_u8 {x : I8} {r : U8} : ConvRel P .i8 .u8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint16)` of an `int8`: its one result is `IntV.cast x`. -/
theorem iff_cast_i8_u16 {x : I8} {r : U16} : ConvRel P .i8 .u16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint32)` of an `int8`: its one result is `IntV.cast x`. -/
theorem iff_cast_i8_u32 {x : I8} {r : U32} : ConvRel P .i8 .u32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint64)` of an `int8`: its one result is `IntV.cast x`. -/
theorem iff_cast_i8_u64 {x : I8} {r : U64} : ConvRel P .i8 .u64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int8)` of an `int16`: its one result is `IntV.cast x`. -/
theorem iff_cast_i16_i8 {x : I16} {r : I8} : ConvRel P .i16 .i8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int16)` of an `int16`: its one result is `IntV.cast x`. -/
theorem iff_cast_i16_i16 {x : I16} {r : I16} : ConvRel P .i16 .i16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int32)` of an `int16`: its one result is `IntV.cast x`. -/
theorem iff_cast_i16_i32 {x : I16} {r : I32} : ConvRel P .i16 .i32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int64)` of an `int16`: its one result is `IntV.cast x`. -/
theorem iff_cast_i16_i64 {x : I16} {r : I64} : ConvRel P .i16 .i64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint8)` of an `int16`: its one result is `IntV.cast x`. -/
theorem iff_cast_i16_u8 {x : I16} {r : U8} : ConvRel P .i16 .u8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint16)` of an `int16`: its one result is `IntV.cast x`. -/
theorem iff_cast_i16_u16 {x : I16} {r : U16} : ConvRel P .i16 .u16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint32)` of an `int16`: its one result is `IntV.cast x`. -/
theorem iff_cast_i16_u32 {x : I16} {r : U32} : ConvRel P .i16 .u32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint64)` of an `int16`: its one result is `IntV.cast x`. -/
theorem iff_cast_i16_u64 {x : I16} {r : U64} : ConvRel P .i16 .u64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int8)` of an `int32`: its one result is `IntV.cast x`. -/
theorem iff_cast_i32_i8 {x : I32} {r : I8} : ConvRel P .i32 .i8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int16)` of an `int32`: its one result is `IntV.cast x`. -/
theorem iff_cast_i32_i16 {x : I32} {r : I16} : ConvRel P .i32 .i16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int32)` of an `int32`: its one result is `IntV.cast x`. -/
theorem iff_cast_i32_i32 {x : I32} {r : I32} : ConvRel P .i32 .i32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int64)` of an `int32`: its one result is `IntV.cast x`. -/
theorem iff_cast_i32_i64 {x : I32} {r : I64} : ConvRel P .i32 .i64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint8)` of an `int32`: its one result is `IntV.cast x`. -/
theorem iff_cast_i32_u8 {x : I32} {r : U8} : ConvRel P .i32 .u8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint16)` of an `int32`: its one result is `IntV.cast x`. -/
theorem iff_cast_i32_u16 {x : I32} {r : U16} : ConvRel P .i32 .u16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint32)` of an `int32`: its one result is `IntV.cast x`. -/
theorem iff_cast_i32_u32 {x : I32} {r : U32} : ConvRel P .i32 .u32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint64)` of an `int32`: its one result is `IntV.cast x`. -/
theorem iff_cast_i32_u64 {x : I32} {r : U64} : ConvRel P .i32 .u64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int8)` of an `int64`: its one result is `IntV.cast x`. -/
theorem iff_cast_i64_i8 {x : I64} {r : I8} : ConvRel P .i64 .i8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int16)` of an `int64`: its one result is `IntV.cast x`. -/
theorem iff_cast_i64_i16 {x : I64} {r : I16} : ConvRel P .i64 .i16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int32)` of an `int64`: its one result is `IntV.cast x`. -/
theorem iff_cast_i64_i32 {x : I64} {r : I32} : ConvRel P .i64 .i32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int64)` of an `int64`: its one result is `IntV.cast x`. -/
theorem iff_cast_i64_i64 {x : I64} {r : I64} : ConvRel P .i64 .i64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint8)` of an `int64`: its one result is `IntV.cast x`. -/
theorem iff_cast_i64_u8 {x : I64} {r : U8} : ConvRel P .i64 .u8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint16)` of an `int64`: its one result is `IntV.cast x`. -/
theorem iff_cast_i64_u16 {x : I64} {r : U16} : ConvRel P .i64 .u16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint32)` of an `int64`: its one result is `IntV.cast x`. -/
theorem iff_cast_i64_u32 {x : I64} {r : U32} : ConvRel P .i64 .u32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint64)` of an `int64`: its one result is `IntV.cast x`. -/
theorem iff_cast_i64_u64 {x : I64} {r : U64} : ConvRel P .i64 .u64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int8)` of a `uint8`: its one result is `IntV.cast x`. -/
theorem iff_cast_u8_i8 {x : U8} {r : I8} : ConvRel P .u8 .i8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int16)` of a `uint8`: its one result is `IntV.cast x`. -/
theorem iff_cast_u8_i16 {x : U8} {r : I16} : ConvRel P .u8 .i16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int32)` of a `uint8`: its one result is `IntV.cast x`. -/
theorem iff_cast_u8_i32 {x : U8} {r : I32} : ConvRel P .u8 .i32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int64)` of a `uint8`: its one result is `IntV.cast x`. -/
theorem iff_cast_u8_i64 {x : U8} {r : I64} : ConvRel P .u8 .i64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint8)` of a `uint8`: its one result is `IntV.cast x`. -/
theorem iff_cast_u8_u8 {x : U8} {r : U8} : ConvRel P .u8 .u8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint16)` of a `uint8`: its one result is `IntV.cast x`. -/
theorem iff_cast_u8_u16 {x : U8} {r : U16} : ConvRel P .u8 .u16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint32)` of a `uint8`: its one result is `IntV.cast x`. -/
theorem iff_cast_u8_u32 {x : U8} {r : U32} : ConvRel P .u8 .u32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint64)` of a `uint8`: its one result is `IntV.cast x`. -/
theorem iff_cast_u8_u64 {x : U8} {r : U64} : ConvRel P .u8 .u64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int8)` of a `uint16`: its one result is `IntV.cast x`. -/
theorem iff_cast_u16_i8 {x : U16} {r : I8} : ConvRel P .u16 .i8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int16)` of a `uint16`: its one result is `IntV.cast x`. -/
theorem iff_cast_u16_i16 {x : U16} {r : I16} : ConvRel P .u16 .i16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int32)` of a `uint16`: its one result is `IntV.cast x`. -/
theorem iff_cast_u16_i32 {x : U16} {r : I32} : ConvRel P .u16 .i32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int64)` of a `uint16`: its one result is `IntV.cast x`. -/
theorem iff_cast_u16_i64 {x : U16} {r : I64} : ConvRel P .u16 .i64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint8)` of a `uint16`: its one result is `IntV.cast x`. -/
theorem iff_cast_u16_u8 {x : U16} {r : U8} : ConvRel P .u16 .u8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint16)` of a `uint16`: its one result is `IntV.cast x`. -/
theorem iff_cast_u16_u16 {x : U16} {r : U16} : ConvRel P .u16 .u16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint32)` of a `uint16`: its one result is `IntV.cast x`. -/
theorem iff_cast_u16_u32 {x : U16} {r : U32} : ConvRel P .u16 .u32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint64)` of a `uint16`: its one result is `IntV.cast x`. -/
theorem iff_cast_u16_u64 {x : U16} {r : U64} : ConvRel P .u16 .u64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int8)` of a `uint32`: its one result is `IntV.cast x`. -/
theorem iff_cast_u32_i8 {x : U32} {r : I8} : ConvRel P .u32 .i8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int16)` of a `uint32`: its one result is `IntV.cast x`. -/
theorem iff_cast_u32_i16 {x : U32} {r : I16} : ConvRel P .u32 .i16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int32)` of a `uint32`: its one result is `IntV.cast x`. -/
theorem iff_cast_u32_i32 {x : U32} {r : I32} : ConvRel P .u32 .i32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int64)` of a `uint32`: its one result is `IntV.cast x`. -/
theorem iff_cast_u32_i64 {x : U32} {r : I64} : ConvRel P .u32 .i64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint8)` of a `uint32`: its one result is `IntV.cast x`. -/
theorem iff_cast_u32_u8 {x : U32} {r : U8} : ConvRel P .u32 .u8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint16)` of a `uint32`: its one result is `IntV.cast x`. -/
theorem iff_cast_u32_u16 {x : U32} {r : U16} : ConvRel P .u32 .u16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint32)` of a `uint32`: its one result is `IntV.cast x`. -/
theorem iff_cast_u32_u32 {x : U32} {r : U32} : ConvRel P .u32 .u32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint64)` of a `uint32`: its one result is `IntV.cast x`. -/
theorem iff_cast_u32_u64 {x : U32} {r : U64} : ConvRel P .u32 .u64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int8)` of a `uint64`: its one result is `IntV.cast x`. -/
theorem iff_cast_u64_i8 {x : U64} {r : I8} : ConvRel P .u64 .i8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int16)` of a `uint64`: its one result is `IntV.cast x`. -/
theorem iff_cast_u64_i16 {x : U64} {r : I16} : ConvRel P .u64 .i16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int32)` of a `uint64`: its one result is `IntV.cast x`. -/
theorem iff_cast_u64_i32 {x : U64} {r : I32} : ConvRel P .u64 .i32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.int64)` of a `uint64`: its one result is `IntV.cast x`. -/
theorem iff_cast_u64_i64 {x : U64} {r : I64} : ConvRel P .u64 .i64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint8)` of a `uint64`: its one result is `IntV.cast x`. -/
theorem iff_cast_u64_u8 {x : U64} {r : U8} : ConvRel P .u64 .u8 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint16)` of a `uint64`: its one result is `IntV.cast x`. -/
theorem iff_cast_u64_u16 {x : U64} {r : U16} : ConvRel P .u64 .u16 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint32)` of a `uint64`: its one result is `IntV.cast x`. -/
theorem iff_cast_u64_u32 {x : U64} {r : U32} : ConvRel P .u64 .u32 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

/-- `x.to(tl.uint64)` of a `uint64`: its one result is `IntV.cast x`. -/
theorem iff_cast_u64_u64 {x : U64} {r : U64} : ConvRel P .u64 .u64 x r ↔ r = IntV.cast x :=
  ofInt_iff_cast

end ConvRel

end TVA
