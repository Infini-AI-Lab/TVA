import TVA.Math.Float.Format
import TVA.Math.Bytes
import Mathlib.Basic.Real.Basic

/-!
# Scalar types and values

The element types of Triton tensors (`tl.dtype`, Triton 3.8.0) and the Lean values that stand
for them. A prover's model of a kernel computes with these values.

* `IntV signed w`: a `w`-bit integer (`tl.int8` … `tl.uint64`), held as its bit pattern. `+ - *`
  wrap modulo `2^w` (Triton lowers them to LLVM `add`/`sub`/`mul` without `nsw`). Division,
  remainder and shifts can be undefined, so they return `Option` (`none` = undefined behaviour).
* `Bool` is `tl.int1`.
* `FloatV f`: a value of the binary format `f`, held as its bit pattern (NaN payloads, signed
  zeros and infinities are kept). The IEEE operations are those of `TVA.Math.Float.Format`
  (proved there) with round-to-nearest-even.
* `DType` names the Triton dtypes the library covers; `DType.Val d` is the Lean type of its
  values; `DType.bytes`, `DType.toBits` and `DType.ofBits` are the memory encoding (little-endian
  bit pattern, `int1` as one byte `0`/`1`, read back as `≠ 0`).

Not covered (`docs/exclusions.md`): `float8e4b15` (Triton's software format), the AMD formats
`float8e4b8` and `float8e5b16`, and pointer-to-pointer element types.
-/
namespace TVA

open TVA.Math.Float TVA.Math.Bytes

/-! ## Integers -/

/-- A `w`-bit integer, signed (`tl.intW`) or unsigned (`tl.uintW`), as its bit pattern. -/
structure IntV (signed : Bool) (w : Nat) where
  bits : BitVec w
  deriving DecidableEq, Repr

namespace IntV

variable {s : Bool} {w : Nat}

/-- The integer with value `v mod 2^w` (two's complement). -/
def ofInt (v : Int) : IntV s w := ⟨BitVec.ofInt w v⟩
/-- The integer with value `v mod 2^w`. -/
def ofNat (v : Nat) : IntV s w := ⟨BitVec.ofNat w v⟩

/-- The mathematical value: two's complement when signed, the natural number otherwise. -/
def toInt (a : IntV s w) : Int := if s then a.bits.toInt else (a.bits.toNat : Int)

instance : Add (IntV s w) := ⟨fun a b => ⟨a.bits + b.bits⟩⟩
instance : Sub (IntV s w) := ⟨fun a b => ⟨a.bits - b.bits⟩⟩
instance : Mul (IntV s w) := ⟨fun a b => ⟨a.bits * b.bits⟩⟩
instance : Neg (IntV s w) := ⟨fun a => ⟨-a.bits⟩⟩
instance {n : Nat} : OfNat (IntV s w) n := ⟨ofNat n⟩
instance : Inhabited (IntV s w) := ⟨⟨0⟩⟩

/-- Bitwise and, or, xor, not (`&`, `|`, `^`, `~`). -/
def and (a b : IntV s w) : IntV s w := ⟨a.bits &&& b.bits⟩
def or (a b : IntV s w) : IntV s w := ⟨a.bits ||| b.bits⟩
def xor (a b : IntV s w) : IntV s w := ⟨a.bits ^^^ b.bits⟩
def not (a : IntV s w) : IntV s w := ⟨~~~a.bits⟩

/-- `<`, `<=` by the type's signedness; `==` and `!=` compare bits. -/
def lt (a b : IntV s w) : Bool := if s then a.bits.slt b.bits else a.bits.ult b.bits
def le (a b : IntV s w) : Bool := if s then a.bits.sle b.bits else a.bits.ule b.bits
def eq (a b : IntV s w) : Bool := a.bits == b.bits

/-- `max`/`min` by the type's signedness (`tl.maximum`, `tl.minimum`). -/
def max (a b : IntV s w) : IntV s w := if lt a b then b else a
def min (a b : IntV s w) : IntV s w := if lt a b then a else b

/-- Division truncating toward zero (`//` on tensors: LLVM `sdiv`/`udiv`, not Python's floor
division). Undefined (`none`) for a zero divisor and for signed `min / -1`. -/
def div? (a b : IntV s w) : Option (IntV s w) :=
  if b.bits = 0 then none
  else if s then
    (if a.bits = BitVec.intMin w ∧ b.bits = -1 then none else some ⟨a.bits.sdiv b.bits⟩)
  else some ⟨a.bits.udiv b.bits⟩

/-- Remainder with the sign of the dividend (`%`: LLVM `srem`/`urem`). Undefined as `div?`. -/
def rem? (a b : IntV s w) : Option (IntV s w) :=
  if b.bits = 0 then none
  else if s then
    (if a.bits = BitVec.intMin w ∧ b.bits = -1 then none else some ⟨a.bits.srem b.bits⟩)
  else some ⟨a.bits.umod b.bits⟩

/-- `a << n`; undefined when `n ≥ w` (LLVM poison). -/
def shl? (a : IntV s w) (n : Nat) : Option (IntV s w) :=
  if n < w then some ⟨a.bits <<< n⟩ else none

/-- `a >> n`, arithmetic (`ashr`); undefined when `n ≥ w`. -/
def ashr? (a : IntV s w) (n : Nat) : Option (IntV s w) :=
  if n < w then some ⟨a.bits.sshiftRight n⟩ else none

/-- `a >> n`, logical (`lshr`); undefined when `n ≥ w`. -/
def lshr? (a : IntV s w) (n : Nat) : Option (IntV s w) :=
  if n < w then some ⟨a.bits >>> n⟩ else none

/-- Integer-to-integer cast (`x.to(tl.intW)`): sign-extend when the *source* is signed,
zero-extend otherwise, truncate when narrower. -/
def cast {s' : Bool} {w' : Nat} (a : IntV s w) : IntV s' w' :=
  ⟨if s then a.bits.signExtend w' else a.bits.zeroExtend w'⟩

end IntV

/-- `tl.int32` values, the type of `tl.program_id`, `tl.arange` and most index arithmetic. -/
abbrev I8 := IntV true 8
abbrev I16 := IntV true 16
abbrev I32 := IntV true 32
abbrev I64 := IntV true 64
abbrev U8 := IntV false 8
abbrev U16 := IntV false 16
abbrev U32 := IntV false 32
abbrev U64 := IntV false 64

/-! ## Floating point -/

/-- A value of the binary format `f`, as its bit pattern. -/
structure FloatV (f : Format) where
  bits : BitVec f.width
  deriving DecidableEq, Repr

namespace FloatV

variable {f : Format}

instance : Inhabited (FloatV f) := ⟨⟨0⟩⟩

/-- The pattern as a natural number (the representation `Format` works on). -/
def pat (x : FloatV f) : Nat := x.bits.toNat
/-- The value with pattern `n mod 2^width`. -/
def ofPat (n : Nat) : FloatV f := ⟨BitVec.ofNat f.width n⟩

def isNaN (x : FloatV f) : Bool := f.isNaN x.pat
def isInf (x : FloatV f) : Bool := f.isInf x.pat
def isFinite (x : FloatV f) : Bool := f.isFinite x.pat
def isZero (x : FloatV f) : Bool := f.isZero x.pat
def sign (x : FloatV f) : Bool := f.sign x.pat

/-- The exact value of a finite pattern (`0` for both zeros); meaningless for NaN and `±∞`. -/
def toRat (x : FloatV f) : Rat := f.toRat x.pat
/-- The exact value as a real number. -/
noncomputable def toReal (x : FloatV f) : ℝ := (x.toRat : ℝ)

/-- The canonical NaN NVIDIA arithmetic produces: positive, all-ones exponent and fraction
(`0x7fffffff` for f32, `0x7fff` for f16 and bf16). -/
def nan : FloatV f := ofPat f.nanMag

/-! ### IEEE operations, round to nearest even

Each is one IEEE 754 operation of the format with one rounding (`Format.add` etc.). These are
the meanings of Triton's `+`, `-`, `*` on one element when no fusion takes place, of `tl.fma`,
`tl.div_rn` and `tl.sqrt_rn` (`TVA.Lang.Ops` decides which source operation means which). -/

def add (x y : FloatV f) : FloatV f := ofPat (f.add .rn f.nanMag x.pat y.pat)
def sub (x y : FloatV f) : FloatV f := ofPat (f.sub .rn f.nanMag x.pat y.pat)
def mul (x y : FloatV f) : FloatV f := ofPat (f.mul .rn f.nanMag x.pat y.pat)
def fma (x y z : FloatV f) : FloatV f := ofPat (f.fma .rn f.nanMag x.pat y.pat z.pat)
def divRn (x y : FloatV f) : FloatV f := ofPat (f.div .rn f.nanMag x.pat y.pat)
def sqrtRn (x : FloatV f) : FloatV f := ofPat (f.sqrt .rn f.nanMag x.pat)

/-- `-x`: flips the sign bit (LLVM `fneg`). The device may return any NaN for a NaN input; the
`tl.*` operations allow that (`FloatV.NaNEq`). -/
def neg (x : FloatV f) : FloatV f := ofPat (f.negate x.pat)
/-- `tl.abs`: clears the sign bit (LLVM `fabs`). -/
def abs (x : FloatV f) : FloatV f := ofPat (f.withSign false x.pat)

/-- The reference `max`/`min` (default `propagate_nan=NONE`): a NaN operand gives the other
operand; `max(-0, +0) = +0`, `min(-0, +0) = -0`. The device may give either zero
(`FloatV.MaxRel` in `TVA.Lang.Ops`). -/
def max (x y : FloatV f) : FloatV f := ofPat (f.fmax f.nanMag x.pat y.pat)
def min (x y : FloatV f) : FloatV f := ofPat (f.fmin f.nanMag x.pat y.pat)
/-- `tl.maximum(…, propagate_nan=ALL)`: a NaN operand gives the canonical NaN. -/
def maxNaN (x y : FloatV f) : FloatV f := ofPat (f.fmaxNaN f.nanMag x.pat y.pat)
def minNaN (x y : FloatV f) : FloatV f := ofPat (f.fminNaN f.nanMag x.pat y.pat)

/-- Comparisons as Triton lowers them: `<`, `<=`, `>`, `>=`, `==` are ordered (false when a NaN
is involved); `!=` is unordered (true when a NaN is involved). `-0 == +0`. -/
def lt (x y : FloatV f) : Bool := f.compare x.pat y.pat == some .lt
def le (x y : FloatV f) : Bool := f.compare x.pat y.pat == some .lt || f.compare x.pat y.pat == some .eq
def eq (x y : FloatV f) : Bool := f.compare x.pat y.pat == some .eq
def ne (x y : FloatV f) : Bool := !(eq x y)

/-- The value nearest to the rational `q` (ties to even), as the conversion of an exact value:
`+0` for `0`, `±∞` on overflow. -/
def ofRat (q : Rat) : FloatV f := ofPat (Format.ofRat f .rn false q)

/-- Conversion between formats with one rounding to nearest even; NaN gives the canonical NaN.
This is `cvt.rn` between standard formats (`x.to(tl.float16)` from `tl.float32`, and every
widening between standard formats, which is exact). Conversions into fp8 are platform
operations (`tl.cvt8`); conversions out of fp8 are `tl.cvtFrom8`. -/
def cvt {g : Format} (x : FloatV f) : FloatV g := ofPat (Format.cvt f g .rn false .canonical x.pat)

/-- Conversion with rounding mode `rnd`: `.rn` (`cvt`) or `.rz` (truncation, `cvt.rz`). -/
def cvtWith {g : Format} (rnd : RoundMode) (x : FloatV f) : FloatV g :=
  ofPat (Format.cvt f g rnd false .canonical x.pat)

theorem cvtWith_rn {g : Format} (x : FloatV f) : (cvtWith .rn x : FloatV g) = cvt x := rfl

end FloatV

/-- The standard float formats: `float16`, `bfloat16`, `float32`, `float64`. -/
def Math.Float.Format.Std (f : Format) : Prop :=
  f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32 ∨ f = .binary64

/-- The fp8 formats the library covers: `float8e5` and `float8e4nv`. -/
def Math.Float.Format.F8 (f : Format) : Prop := f = .e5m2 ∨ f = .e4m3

abbrev F16 := FloatV Format.binary16
abbrev BF16 := FloatV Format.bfloat16
abbrev F32 := FloatV Format.binary32
abbrev F64 := FloatV Format.binary64
abbrev F8E5 := FloatV Format.e5m2
abbrev F8E4 := FloatV Format.e4m3

/-! ## Triton dtypes -/

/-- The Triton scalar types the library covers. -/
inductive DType where
  /-- `tl.int1` (boolean). -/
  | i1
  | i8 | i16 | i32 | i64
  | u8 | u16 | u32 | u64
  /-- `tl.float16`, `tl.bfloat16`, `tl.float32`, `tl.float64`. -/
  | f16 | bf16 | f32 | f64
  /-- `tl.float8e5` (OCP E5M2) and `tl.float8e4nv` (OCP E4M3FN). -/
  | f8e5 | f8e4nv
  deriving DecidableEq, Repr

namespace DType

/-- The Lean type of a dtype's values. -/
@[reducible] def Val : DType → Type
  | .i1 => Bool
  | .i8 => I8 | .i16 => I16 | .i32 => I32 | .i64 => I64
  | .u8 => U8 | .u16 => U16 | .u32 => U32 | .u64 => U64
  | .f16 => F16 | .bf16 => BF16 | .f32 => F32 | .f64 => F64
  | .f8e5 => F8E5 | .f8e4nv => F8E4

/-- Bytes an element occupies in memory (`int1` takes a byte). -/
def bytes : DType → Nat
  | .i1 | .i8 | .u8 | .f8e5 | .f8e4nv => 1
  | .i16 | .u16 | .f16 | .bf16 => 2
  | .i32 | .u32 | .f32 => 4
  | .i64 | .u64 | .f64 => 8

theorem bytes_pos (d : DType) : 0 < d.bytes := by cases d <;> decide

/-- The bit pattern of a value, stored as `bytes` little-endian bytes. -/
def toBits : (d : DType) → d.Val → Nat
  | .i1, b => if b then 1 else 0
  | .i8, x => x.bits.toNat | .i16, x => x.bits.toNat | .i32, x => x.bits.toNat
  | .i64, x => x.bits.toNat
  | .u8, x => x.bits.toNat | .u16, x => x.bits.toNat | .u32, x => x.bits.toNat
  | .u64, x => x.bits.toNat
  | .f16, x => x.pat | .bf16, x => x.pat | .f32, x => x.pat | .f64, x => x.pat
  | .f8e5, x => x.pat | .f8e4nv, x => x.pat

/-- The value with the given bit pattern; an `int1` byte reads as `true` when non-zero. -/
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

/-- The value is a NaN (never, for integer and boolean dtypes). -/
def IsNaN : (d : DType) → d.Val → Prop
  | .f16, x => x.isNaN = true | .bf16, x => x.isNaN = true | .f32, x => x.isNaN = true
  | .f64, x => x.isNaN = true | .f8e5, x => x.isNaN = true | .f8e4nv, x => x.isNaN = true
  | _, _ => False

/-- Its float format, for the floating dtypes. -/
def format? : DType → Option Format
  | .f16 => some .binary16 | .bf16 => some .bfloat16 | .f32 => some .binary32
  | .f64 => some .binary64 | .f8e5 => some .e5m2 | .f8e4nv => some .e4m3
  | _ => none

/-- Every float dtype has a NaN. -/
theorem exists_nan : (d : DType) → d.format?.isSome = true → ∃ z : d.Val, d.IsNaN z
  | .f16, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .bf16, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f32, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f64, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f8e5, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f8e4nv, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .i1, h | .i8, h | .i16, h | .i32, h | .i64, h | .u8, h | .u16, h | .u32, h | .u64, h => by
      simp [format?] at h

end DType

end TVA
