import TVA.Base.Bytes
import TVA.Base.Int
import TVA.Base.Float

/-!
# Triton's dtypes

`DType` names the scalar types of Triton (`tl.dtype`) that TVA covers, and `DType.Val d` is the
Lean type of a value of `d`: `Bool` for `tl.int1`, `IntV` for the integers, `FloatV` for the
floats. A dtype also fixes how an element sits in memory: `bytes d` bytes holding its bit
pattern `toBits` little-endian (`TVA.Base.Bytes`); `ofBits` reads a pattern back. `val?` is an
element's exact value as a rational, when it has one.

`TVA.Memory` uses `bytes`, `toBits` and `ofBits` for loads and stores; pointers and tensors are
indexed by dtypes (`Ptr d`, `Buf d`).

Not covered: `tl.float8e4b15` (Triton's software format), the AMD formats `float8e4b8` and
`float8e5b16`, and pointer-to-pointer elements.
-/
namespace TVA

/-- The Triton scalar types TVA covers. -/
inductive DType where
  /-- `tl.int1`, a boolean. -/
  | i1
  /-- `tl.int8`. -/
  | i8
  /-- `tl.int16`. -/
  | i16
  /-- `tl.int32`. -/
  | i32
  /-- `tl.int64`. -/
  | i64
  /-- `tl.uint8`. -/
  | u8
  /-- `tl.uint16`. -/
  | u16
  /-- `tl.uint32`. -/
  | u32
  /-- `tl.uint64`. -/
  | u64
  /-- `tl.float16`. -/
  | f16
  /-- `tl.bfloat16`. -/
  | bf16
  /-- `tl.float32`. -/
  | f32
  /-- `tl.float64`. -/
  | f64
  /-- `tl.float8e5` (OCP E5M2). -/
  | f8e5
  /-- `tl.float8e4nv` (OCP E4M3FN). -/
  | f8e4nv
  deriving DecidableEq, Repr

namespace DType

/-- The Lean type of a value of the dtype. -/
@[reducible] def Val : DType → Type
  | .i1 => Bool
  | .i8 => I8 | .i16 => I16 | .i32 => I32 | .i64 => I64
  | .u8 => U8 | .u16 => U16 | .u32 => U32 | .u64 => U64
  | .f16 => F16 | .bf16 => BF16 | .f32 => F32 | .f64 => F64
  | .f8e5 => F8E5 | .f8e4nv => F8E4

/-- The bytes an element occupies in memory (`tl.int1` takes a whole byte). -/
def bytes : DType → Nat
  | .i1 | .i8 | .u8 | .f8e5 | .f8e4nv => 1
  | .i16 | .u16 | .f16 | .bf16 => 2
  | .i32 | .u32 | .f32 => 4
  | .i64 | .u64 | .f64 => 8

/-- Every element takes at least one byte. -/
theorem bytes_pos (d : DType) : 0 < d.bytes := by cases d <;> decide

/-- The bit pattern of a value, as stored in `bytes` little-endian bytes: `1` or `0` for a
boolean. -/
def toBits : (d : DType) → d.Val → Nat
  | .i1, b => if b then 1 else 0
  | .i8, x => x.bits.toNat | .i16, x => x.bits.toNat | .i32, x => x.bits.toNat
  | .i64, x => x.bits.toNat
  | .u8, x => x.bits.toNat | .u16, x => x.bits.toNat | .u32, x => x.bits.toNat
  | .u64, x => x.bits.toNat
  | .f16, x => x.pat | .bf16, x => x.pat | .f32, x => x.pat | .f64, x => x.pat
  | .f8e5, x => x.pat | .f8e4nv, x => x.pat

/-- The value with the given bit pattern (taken modulo `2^bits`); a boolean byte reads as
`true` when it is not zero. -/
def ofBits : (d : DType) → Nat → d.Val
  | .i1, n => n != 0
  | .i8, n => IntV.ofNat n | .i16, n => IntV.ofNat n | .i32, n => IntV.ofNat n
  | .i64, n => IntV.ofNat n
  | .u8, n => IntV.ofNat n | .u16, n => IntV.ofNat n | .u32, n => IntV.ofNat n
  | .u64, n => IntV.ofNat n
  | .f16, n => FloatV.ofPat n | .bf16, n => FloatV.ofPat n | .f32, n => FloatV.ofPat n
  | .f64, n => FloatV.ofPat n
  | .f8e5, n => FloatV.ofPat n | .f8e4nv, n => FloatV.ofPat n

instance (d : DType) : Inhabited d.Val := ⟨d.ofBits 0⟩

instance (d : DType) : DecidableEq d.Val := by
  cases d <;> exact inferInstance

/-- The float format of a float dtype; `none` for the integer and boolean dtypes. -/
def format? : DType → Option Format
  | .f16 => some .binary16 | .bf16 => some .bfloat16 | .f32 => some .binary32
  | .f64 => some .binary64 | .f8e5 => some .e5m2 | .f8e4nv => some .e4m3
  | _ => none

/-- The value is a NaN (never, for the integer and boolean dtypes). -/
def IsNaN : (d : DType) → d.Val → Prop
  | .f16, x => x.isNaN = true | .bf16, x => x.isNaN = true | .f32, x => x.isNaN = true
  | .f64, x => x.isNaN = true | .f8e5, x => x.isNaN = true | .f8e4nv, x => x.isNaN = true
  | _, _ => False

/-- The exact value of an element, if it has one: `0` or `1` for a boolean, the integer
(`IntV.toInt`) for an integer dtype, and `toRat` for a finite float (`none` for `±∞` and NaN). -/
def val? : (d : DType) → d.Val → Option Rat
  | .i1, b => some (if b then 1 else 0)
  | .i8, x => some x.toInt | .i16, x => some x.toInt | .i32, x => some x.toInt
  | .i64, x => some x.toInt
  | .u8, x => some x.toInt | .u16, x => some x.toInt | .u32, x => some x.toInt
  | .u64, x => some x.toInt
  | .f16, x => if x.isFinite then some x.toRat else none
  | .bf16, x => if x.isFinite then some x.toRat else none
  | .f32, x => if x.isFinite then some x.toRat else none
  | .f64, x => if x.isFinite then some x.toRat else none
  | .f8e5, x => if x.isFinite then some x.toRat else none
  | .f8e4nv, x => if x.isFinite then some x.toRat else none

end DType

end TVA
