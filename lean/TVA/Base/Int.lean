import Mathlib.Algebra.Group.Defs

/-!
# Fixed-width integers

`IntV signed w` is a value of Triton's integer dtype of `w` bits: `tl.int8` … `tl.int64` when
`signed`, `tl.uint8` … `tl.uint64` otherwise. It is held as its bit pattern (`BitVec w`); the
signedness only says how the pattern is read (`toInt`) and which LLVM instruction an operation
lowers to. `tl.int1` is `Bool` (`TVA.Base.DType`).

Triton lowers integer operations to LLVM (`semantic.py`), and this file gives each its LLVM
meaning:
* `+ - *` and unary `-` wrap modulo `2^w` (`add`, `sub`, `mul` without `nsw`/`nuw`);
* `//` and `%` truncate toward zero (`sdiv`/`udiv`, `srem`/`urem`, `semantic.py:317-359`), not
  Python's floor division; they are undefined for a zero divisor and for `min // -1`, so `div?`
  and `rem?` return `Option` (`none`: undefined behaviour);
* `<<`, `>>` are undefined when the amount is at least the width (`shl?`, `ashr?`, `lshr?`);
* `& | ^ ~` act on the bits (`and`, `or`, `xor`, `not`);
* `<`, `<=`, `tl.maximum`, `tl.minimum` compare as signed or unsigned by the dtype
  (`icmp slt`/`ult`, `semantic.py:529-540`; `maxsi`/`maxui`, `semantic.py:366-398`);
* `x.to(…)` between integer dtypes is `cast` (`semantic.py:870-880`).

Arithmetic on Python integers (`tl.constexpr` values) is done in Lean on `Int`/`Nat` by the model,
not here. `TVA.Triton.Arith` lifts these operations to tensors.
-/
namespace TVA

/-- A value of a `w`-bit Triton integer dtype, signed (`tl.intW`) or unsigned (`tl.uintW`),
held as its bit pattern. -/
@[ext] structure IntV (signed : Bool) (w : Nat) where
  /-- The bit pattern (two's complement when signed). -/
  bits : BitVec w
  deriving DecidableEq, Repr

namespace IntV

variable {s : Bool} {w : Nat}

/-- The integer whose pattern is `v mod 2^w` (two's complement): what a Python integer becomes
in an integer dtype, and the wrapped result of exact integer arithmetic. -/
def ofInt (v : Int) : IntV s w := ⟨BitVec.ofInt w v⟩

/-- The integer whose pattern is `v mod 2^w`. -/
def ofNat (v : Nat) : IntV s w := ⟨BitVec.ofNat w v⟩

/-- The number the value stands for: the two's complement reading of the pattern when signed,
the natural number otherwise. -/
def toInt (a : IntV s w) : Int := if s then a.bits.toInt else (a.bits.toNat : Int)

/-- The smallest value of the dtype: `-2^(w-1)` when signed, `0` otherwise. -/
def minInt (signed : Bool) (w : Nat) : Int := if signed then -2 ^ (w - 1) else 0

/-- The largest value of the dtype: `2^(w-1) - 1` when signed, `2^w - 1` otherwise. -/
def maxInt (signed : Bool) (w : Nat) : Int := if signed then 2 ^ (w - 1) - 1 else 2 ^ w - 1

/-- `a + b`, wrapping modulo `2^w`. -/
instance : Add (IntV s w) := ⟨fun a b => ⟨a.bits + b.bits⟩⟩
/-- `a - b`, wrapping modulo `2^w`. -/
instance : Sub (IntV s w) := ⟨fun a b => ⟨a.bits - b.bits⟩⟩
/-- `a * b`, wrapping modulo `2^w`. -/
instance : Mul (IntV s w) := ⟨fun a b => ⟨a.bits * b.bits⟩⟩
/-- `-a`, wrapping modulo `2^w`. -/
instance : Neg (IntV s w) := ⟨fun a => ⟨-a.bits⟩⟩
/-- A numeral `n` is `ofNat n`. -/
instance {n : Nat} : OfNat (IntV s w) n := ⟨ofNat n⟩
instance : Inhabited (IntV s w) := ⟨⟨0⟩⟩

/-- `a & b`, bitwise. -/
def and (a b : IntV s w) : IntV s w := ⟨a.bits &&& b.bits⟩
/-- `a | b`, bitwise. -/
def or (a b : IntV s w) : IntV s w := ⟨a.bits ||| b.bits⟩
/-- `a ^ b`, bitwise. -/
def xor (a b : IntV s w) : IntV s w := ⟨a.bits ^^^ b.bits⟩
/-- `~a`, bitwise. -/
def not (a : IntV s w) : IntV s w := ⟨~~~a.bits⟩

/-- `a < b`, signed or unsigned by the dtype. -/
def lt (a b : IntV s w) : Bool := if s then a.bits.slt b.bits else a.bits.ult b.bits
/-- `a <= b`, signed or unsigned by the dtype. -/
def le (a b : IntV s w) : Bool := if s then a.bits.sle b.bits else a.bits.ule b.bits
/-- `a == b`: the patterns are equal (`a != b` is its negation). -/
def eq (a b : IntV s w) : Bool := a.bits == b.bits

/-- `tl.maximum(a, b)`, signed or unsigned by the dtype. -/
def max (a b : IntV s w) : IntV s w := if lt a b then b else a
/-- `tl.minimum(a, b)`, signed or unsigned by the dtype. -/
def min (a b : IntV s w) : IntV s w := if lt a b then a else b

/-- `a // b` on integer tensors: truncating toward zero (LLVM `sdiv`/`udiv`). `none`
(undefined behaviour) for a zero divisor and for signed `min // -1`. -/
def div? (a b : IntV s w) : Option (IntV s w) :=
  if b.bits = 0 then none
  else if s then
    (if a.bits = BitVec.intMin w ∧ b.bits = -1 then none else some ⟨a.bits.sdiv b.bits⟩)
  else some ⟨a.bits.udiv b.bits⟩

/-- `a % b` on integer tensors: the remainder of `div?`, with the sign of `a` (LLVM
`srem`/`urem`). Undefined exactly when `div?` is. -/
def rem? (a b : IntV s w) : Option (IntV s w) :=
  if b.bits = 0 then none
  else if s then
    (if a.bits = BitVec.intMin w ∧ b.bits = -1 then none else some ⟨a.bits.srem b.bits⟩)
  else some ⟨a.bits.umod b.bits⟩

/-- `a << n`; `none` (LLVM poison, undefined) when `n ≥ w`. -/
def shl? (a : IntV s w) (n : Nat) : Option (IntV s w) :=
  if n < w then some ⟨a.bits <<< n⟩ else none

/-- `a >> n`, arithmetic (LLVM `ashr`, copies of the sign bit shifted in); `none` when `n ≥ w`.
Which of `ashr?` and `lshr?` a `>>` means is decided by `TVA.Triton.Arith`. -/
def ashr? (a : IntV s w) (n : Nat) : Option (IntV s w) :=
  if n < w then some ⟨a.bits.sshiftRight n⟩ else none

/-- `a >> n`, logical (LLVM `lshr`, zeros shifted in); `none` when `n ≥ w`. -/
def lshr? (a : IntV s w) (n : Nat) : Option (IntV s w) :=
  if n < w then some ⟨a.bits >>> n⟩ else none

/-- `a.to(tl.intW')` between integer dtypes: sign-extended when the *source* is signed,
zero-extended otherwise, truncated when narrower. The result is the value modulo `2^w'`. -/
def cast {s' : Bool} {w' : Nat} (a : IntV s w) : IntV s' w' :=
  ⟨if s then a.bits.signExtend w' else a.bits.zeroExtend w'⟩

/-- Integers of a width form a commutative monoid under wrapping addition, so that sums of
integer elements (`tl.sum`, atomic increments) can be written with `∑`. -/
instance : AddCommMonoid (IntV s w) where
  add := (· + ·)
  zero := 0
  add_assoc a b c := IntV.ext (BitVec.add_assoc a.bits b.bits c.bits)
  zero_add a := IntV.ext (by show BitVec.ofNat w 0 + a.bits = a.bits; simp)
  add_zero a := IntV.ext (by show a.bits + BitVec.ofNat w 0 = a.bits; simp)
  add_comm a b := IntV.ext (BitVec.add_comm a.bits b.bits)
  nsmul := nsmulRec

end IntV

/-- `tl.int8` values. -/
abbrev I8 := IntV true 8
/-- `tl.int16` values. -/
abbrev I16 := IntV true 16
/-- `tl.int32` values: the type of `tl.program_id`, `tl.arange` and most index arithmetic. -/
abbrev I32 := IntV true 32
/-- `tl.int64` values. -/
abbrev I64 := IntV true 64
/-- `tl.uint8` values. -/
abbrev U8 := IntV false 8
/-- `tl.uint16` values. -/
abbrev U16 := IntV false 16
/-- `tl.uint32` values. -/
abbrev U32 := IntV false 32
/-- `tl.uint64` values. -/
abbrev U64 := IntV false 64

end TVA
