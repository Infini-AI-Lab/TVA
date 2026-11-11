import TVA.Base.Float.Arith
import Mathlib.Basic.Real.Basic

/-!
# Float values

`FloatV f` is a value of the float dtype with format `f`, held as its bit pattern, so NaN
payloads, signed zeros and infinities are all kept. `F16`, `BF16`, `F32`, `F64`, `F8E5` and `F8E4`
are the values of `tl.float16`, `tl.bfloat16`, `tl.float32`, `tl.float64`, `tl.float8e5` and
`tl.float8e4nv`.

The operations are the IEEE operations of `TVA.Base.Float.Arith` with rounding to nearest even:
`add`, `sub`, `mul`, `fma`, `divRn`, `sqrtRn`, `neg`, `abs`, `max`, `min`, `maxNaN`, `minNaN`,
the comparisons, `ofRat` (rounding an exact value) and the conversions `cvt`, `cvtWith`,
`cvtSat`. `toRat` and `toReal` give the exact value; `NaNEq` is equality up to NaN payload.

These are exact reference results. Which of them a `tl.*` operation returns, and where the device
may differ (fused multiply-add, approximate division, NaN payloads), is said by `TVA.Triton.Arith`
and `TVA.Triton.Convert`.
-/
namespace TVA

/-- A value of the float dtype with format `f`, as its bit pattern. -/
structure FloatV (f : Format) where
  /-- The bit pattern. -/
  bits : BitVec f.width
  deriving DecidableEq, Repr

namespace FloatV

variable {f : Format}

instance : Inhabited (FloatV f) := ⟨⟨0⟩⟩

/-- The pattern as a natural number (what `Format`'s functions work on). -/
def pat (x : FloatV f) : Nat := x.bits.toNat
/-- The value with pattern `n mod 2^width`. -/
def ofPat (n : Nat) : FloatV f := ⟨BitVec.ofNat f.width n⟩

/-- `x` is a NaN. -/
def isNaN (x : FloatV f) : Bool := f.isNaN x.pat
/-- `x` is `+∞` or `-∞`. -/
def isInf (x : FloatV f) : Bool := f.isInf x.pat
/-- `x` is a number (neither NaN nor infinite). -/
def isFinite (x : FloatV f) : Bool := f.isFinite x.pat
/-- `x` is `+0` or `-0`. -/
def isZero (x : FloatV f) : Bool := f.isZero x.pat
/-- The sign bit of `x` (`true`: negative). -/
def sign (x : FloatV f) : Bool := f.sign x.pat

/-- The exact value of `x` when it is finite (`0` for both zeros). Of `±∞` it is `±2^(bias+1)`,
beyond `±maxVal`, so comparing values orders the infinities below and above every number; of a
NaN it is meaningless. A specification states `isFinite` before it uses `toRat`. -/
def toRat (x : FloatV f) : Rat := f.toRat x.pat
/-- The exact value as a real number (`toRat`, cast). -/
noncomputable def toReal (x : FloatV f) : ℝ := (x.toRat : ℝ)

/-- The canonical NaN: positive, all exponent and fraction bits set (`0x7fffffff` in `float32`,
`0x7fff` in `float16` and `bfloat16`, `0x7f` in both fp8 formats). The device gives it for every
NaN result of `float32`, `float16` and `bfloat16` arithmetic, but not always elsewhere (`-x` and
`tl.abs` keep the payload; `float64` arithmetic gives other NaNs). Since the `tl.*` float
operations are specified up to NaN payload (`NaNEq`), it stands for any NaN there. -/
def nan : FloatV f := ofPat f.nanMag

/-- `-∞` (`0xff800000` in `float32`), in a format with infinities. -/
def negInf : FloatV f := ofPat (f.setSign true f.infMag)

/-- `+∞` (`0x7f800000` in `float32`), in a format with infinities. -/
def posInf : FloatV f := ofPat (f.setSign false f.infMag)

/-- `x` and `y` are equal or both NaN: equality up to NaN payload, which is what the device
guarantees for a NaN result. -/
def NaNEq (x y : FloatV f) : Prop := x = y ∨ (x.isNaN = true ∧ y.isNaN = true)

/-! ## IEEE operations, rounding to nearest even

Each is one IEEE 754 operation of the format with one rounding. They are what Triton's `+`, `-`,
`*` on one element mean when no fusion takes place, and `tl.fma`, `tl.div_rn`, `tl.sqrt_rn`. -/

/-- `x + y`, rounded to nearest even. -/
def add (x y : FloatV f) : FloatV f := ofPat (f.add .rn x.pat y.pat)
/-- `x - y`, rounded to nearest even. -/
def sub (x y : FloatV f) : FloatV f := ofPat (f.sub .rn x.pat y.pat)
/-- `x * y`, rounded to nearest even. -/
def mul (x y : FloatV f) : FloatV f := ofPat (f.mul .rn x.pat y.pat)
/-- `x * y + z` with one rounding to nearest even (`tl.fma`). -/
def fma (x y z : FloatV f) : FloatV f := ofPat (f.fma .rn x.pat y.pat z.pat)
/-- `x / y`, correctly rounded to nearest even (`tl.div_rn`). -/
def divRn (x y : FloatV f) : FloatV f := ofPat (f.div .rn x.pat y.pat)
/-- `√x`, correctly rounded to nearest even (`tl.sqrt_rn`). -/
def sqrtRn (x : FloatV f) : FloatV f := ofPat (f.sqrt .rn x.pat)

/-- `-x`: the sign bit flipped (LLVM `fneg`). -/
def neg (x : FloatV f) : FloatV f := ofPat (f.neg x.pat)
/-- `tl.abs(x)`: the sign bit cleared (LLVM `fabs`). -/
def abs (x : FloatV f) : FloatV f := ofPat (f.absPat x.pat)

/-- `tl.maximum(x, y)` (default `propagate_nan=NONE`, LLVM `maxnum`): a NaN operand gives the
other operand; `max(-0, +0) = +0`. -/
def max (x y : FloatV f) : FloatV f := ofPat (f.maxNum x.pat y.pat)
/-- `tl.minimum(x, y)` (default `propagate_nan=NONE`, LLVM `minnum`): a NaN operand gives the
other operand; `min(-0, +0) = -0`. -/
def min (x y : FloatV f) : FloatV f := ofPat (f.minNum x.pat y.pat)
/-- `tl.maximum(x, y, propagate_nan=ALL)` (LLVM `maximum`): a NaN operand gives NaN. -/
def maxNaN (x y : FloatV f) : FloatV f := ofPat (f.maxNaN x.pat y.pat)
/-- `tl.minimum(x, y, propagate_nan=ALL)` (LLVM `minimum`): a NaN operand gives NaN. -/
def minNaN (x y : FloatV f) : FloatV f := ofPat (f.minNaN x.pat y.pat)

/-- `x < y`: ordered (false when a NaN is involved, `fcmp olt`, `semantic.py:529-534`). -/
def lt (x y : FloatV f) : Bool := f.cmp x.pat y.pat == some .lt
/-- `x <= y`: ordered (false when a NaN is involved). -/
def le (x y : FloatV f) : Bool := f.cmp x.pat y.pat == some .lt || f.cmp x.pat y.pat == some .eq
/-- `x == y`: ordered (false when a NaN is involved); `-0 == +0`. -/
def eq (x y : FloatV f) : Bool := f.cmp x.pat y.pat == some .eq
/-- `x != y`: unordered (true when a NaN is involved). -/
def ne (x y : FloatV f) : Bool := !(eq x y)

/-- The value nearest to the rational `q` (ties to even), `+0` for `0`: the rounding of an exact
value. An overflow gives what `f`'s `Specials` gives: `±∞` (`.ieee`), NaN (`.nanOnly`:
`float8e4nv`) or `±maxVal` (`.finite`: `e2m1`). -/
def ofRat (q : Rat) : FloatV f := ofPat (f.roundRat .rn false false q)

/-- Conversion into format `g` with one rounding to nearest even; NaN gives `g`'s canonical NaN,
`±∞` gives `±∞` (NaN in `e4m3`). This is `cvt.rn` between the standard formats (`x.to(tl.float16)`
from `tl.float32`; every widening, which is exact). -/
def cvt {g : Format} (x : FloatV f) : FloatV g := ofPat (Format.cvt f g .rn false x.pat)

/-- Conversion into format `g` with rounding mode `rnd`: `.rn` (`cvt`) or `.rz` (truncation,
`cvt.rz`, `x.to(…, fp_downcast_rounding="rtz")`). -/
def cvtWith {g : Format} (rnd : RoundMode) (x : FloatV f) : FloatV g :=
  ofPat (Format.cvt f g rnd false x.pat)

/-- The saturating conversion into `g` (PTX `.satfinite`): as `cvtWith rnd`, except that a finite
value beyond `g`'s range and `±∞` give `±maxVal` of `g`; NaN gives `g`'s canonical NaN. This is
what the device does when it converts into fp8. -/
def cvtSat {g : Format} (rnd : RoundMode) (x : FloatV f) : FloatV g :=
  ofPat (Format.cvt f g rnd true x.pat)

end FloatV

/-- `tl.float16` values. -/
abbrev F16 := FloatV Format.binary16
/-- `tl.bfloat16` values. -/
abbrev BF16 := FloatV Format.bfloat16
/-- `tl.float32` values. -/
abbrev F32 := FloatV Format.binary32
/-- `tl.float64` values. -/
abbrev F64 := FloatV Format.binary64
/-- `tl.float8e5` values (OCP E5M2). -/
abbrev F8E5 := FloatV Format.e5m2
/-- `tl.float8e4nv` values (OCP E4M3FN). -/
abbrev F8E4 := FloatV Format.e4m3

end TVA
