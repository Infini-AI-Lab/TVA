/-!
# IEEE-754 binary formats on bit patterns

Field extraction, NaN/Inf/zero classification and the IEEE comparison for binary formats
with `e` exponent and `m` mantissa bits, on natural-number bit patterns. Instantiated for
`bf16` (8, 7), `f16` (5, 10) and `f32` (8, 23). Layer L0: no GPU notions.
-/
namespace TVA.Math.Float

/-- Fields of a binary float with `e` exponent bits and `m` mantissa bits. -/
structure FloatFields where
  sign : Bool
  exp : Nat
  mant : Nat
  deriving Repr, DecidableEq

def floatFields (e m : Nat) (bits : Nat) : FloatFields :=
  { sign := bits.testBit (e + m)
    exp := (bits >>> m) % (2 ^ e)
    mant := bits % (2 ^ m) }

def FloatFields.isNaN (e : Nat) (f : FloatFields) : Bool := f.exp == 2 ^ e - 1 && f.mant != 0
def FloatFields.isInf (e : Nat) (f : FloatFields) : Bool := f.exp == 2 ^ e - 1 && f.mant == 0
def FloatFields.isZero (f : FloatFields) : Bool := f.exp == 0 && f.mant == 0

/-- `bf16` has 8 exponent bits and 7 mantissa bits. -/
def bf16Fields (bits : Nat) : FloatFields := floatFields 8 7 bits
/-- `f32` has 8 exponent bits and 23 mantissa bits. -/
def f32Fields (bits : Nat) : FloatFields := floatFields 8 23 bits
/-- `f16` has 5 exponent bits and 10 mantissa bits. -/
def f16Fields (bits : Nat) : FloatFields := floatFields 5 10 bits

def bf16IsNaN (bits : Nat) : Bool := (bf16Fields bits).isNaN 8
def f32IsNaN (bits : Nat) : Bool := (f32Fields bits).isNaN 8
def f16IsNaN (bits : Nat) : Bool := (f16Fields bits).isNaN 5

/-- `cvt.f32.bf16` is exact: shift the 16 bits into the high half. -/
def bf16ToF32Bits (bits : Nat) : Nat := (bits % 2 ^ 16) <<< 16

/-- Magnitude-ordering key for an IEEE pattern with `e+m+1` bits: maps the bit
pattern to an integer whose order agrees with the numeric order for non-NaN
values (with `-0 < +0`, which is *not* IEEE equality but is what a total order
on bit patterns gives). Used to define comparisons without real numbers. -/
def ieeeOrderKey (e m : Nat) (bits : Nat) : Int :=
  let f := floatFields e m bits
  let mag : Int := (f.exp * 2 ^ m + f.mant : Nat)
  if f.sign then -mag - 1 else mag

/-- IEEE comparison result on bit patterns: `none` when unordered (a NaN operand). -/
def ieeeCompare (e m : Nat) (a b : Nat) : Option Ordering :=
  let fa := floatFields e m a
  let fb := floatFields e m b
  if fa.isNaN e || fb.isNaN e then none
  else if fa.isZero && fb.isZero then some .eq       -- +0 = -0 in IEEE
  else some (compare (ieeeOrderKey e m a) (ieeeOrderKey e m b))

def f32Compare (a b : Nat) : Option Ordering := ieeeCompare 8 23 a b
def bf16Compare (a b : Nat) : Option Ordering := ieeeCompare 8 7 a b
def f16Compare (a b : Nat) : Option Ordering := ieeeCompare 5 10 a b

end TVA.Math.Float
