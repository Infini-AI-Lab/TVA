import TVA.Math.Float.IEEE

/-!
# Generic binary floating-point formats on bit patterns

A `Format` has `e` exponent bits and `m` fraction bits, stored sign–exponent–fraction in
`e + m + 1` bits (sign at bit `e + m`), with the IEEE bias `2^(e-1) − 1`. What the all-ones
exponent holds is the format's `Special` convention:

* `ieee`: `±∞` (zero fraction) and NaNs (non-zero fraction): binary64/32/16, bfloat16, tf32,
  OCP e5m2;
* `fn` ("finite, with NaN"): no infinities; the all-ones exponent holds normal numbers, except
  the all-ones fraction, which is the only NaN (one per sign): OCP e4m3 (`E4M3FN`);
* `finite`: no infinities and no NaN, every pattern is a number: OCP MX e2m1 (fp4), e2m3 and
  e3m2 (fp6).

Rounding (`roundMag`, `roundRat`) takes the exact rational value to the format under a
`RoundMode` (`rn` ties-to-even, `rna` ties-away, `rz`, `ru`, `rd`), keeps subnormals, and
handles overflow by `overflow`: `sat` (PTX `.satfinite`) clamps to the largest finite value;
otherwise a mode that rounds the magnitude up gives `∞` (`ieee`), the NaN (`fn`) or the largest
finite value (`finite`), and the other modes give the largest finite value.

The operations (`add`, `sub`, `mul`, `fma`, `rcp`, `div`, `fmax`, `fmin`) are the IEEE 754
operations of the format, with the NaN a result takes passed as a parameter; conversions
(`cvt`) round the exact value of the source pattern into the destination and take a NaN
rule (`NaNRule`): a canonical pattern, a quiet NaN keeping the sign, or the widened payload.

`fmaxNaN`/`fminNaN` are the `.NaN` variants of `max`/`min`; `negNaN`/`absNaN` flip or clear the
sign and turn a NaN into a given pattern; `clampUnit` (`.sat`) and `relu` (`.relu`) clamp a result;
`toInt` and `satInt` are the saturating float-to-integer and integer-to-integer conversions of
PTX `cvt`.

Layer L0, without Mathlib, so that the conformance harness compiles it natively. The named
instances are at the end (`binary32`, `binary16`, `bfloat16`, `tf32`, `e5m2`, `e4m3`, `e2m1`,
`e2m3`, `e3m2`, `binary64`). `TVA.Math.Float.Format.Bridge` proves that the existing binary32
operations (`TVA.Math.Float.F32`) and conversions (`TVA.Math.Float.Convert`) are the instances
of these definitions. Not covered: AMD's `fnuz` formats (no `-0`, NaN `0x80`), flush-to-zero
variants and the MX scale format `ue8m0` (`tl.dot_scaled` decodes it by `e8m0Value`). `e2m1` is
the element format of `tl.dot_scaled`'s fp4 operands; `e2m3` and `e3m2` are definitions only.
-/
namespace TVA.Math.Float

/-- What the all-ones exponent of a format holds. -/
inductive Special where
  /-- `±∞` with a zero fraction, NaN otherwise (IEEE 754). -/
  | ieee
  /-- No infinities; only the all-ones fraction is a NaN (OCP e4m3). -/
  | fn
  /-- No infinities and no NaN (OCP MX fp4/fp6). -/
  | finite
  deriving Repr, DecidableEq, Inhabited

/-- A binary floating-point format: `e` exponent bits, `m` fraction bits, a sign bit. -/
structure Format where
  e : Nat
  m : Nat
  special : Special := .ieee
  deriving Repr, DecidableEq, Inhabited

/-- Rounding modes. PTX spells them `.rn`, `.rna`, `.rz`, `.rp` (`ru`) and `.rm` (`rd`). -/
inductive RoundMode where
  /-- To nearest, ties to even. -/
  | rn
  /-- To nearest, ties away from zero. -/
  | rna
  /-- Toward zero. -/
  | rz
  /-- Toward `+∞`. -/
  | ru
  /-- Toward `−∞`. -/
  | rd
  deriving Repr, DecidableEq, Inhabited

namespace Format

/-- `⌊log₂ (n / d)⌋` for `n, d > 0` (the same function as `TVA.Math.Float.F32.log2Floor`). -/
def log2Floor (n d : Nat) : Int :=
  let a := Nat.log2 n
  let b := Nat.log2 d
  let ge : Bool := if a ≥ b then n ≥ d * 2 ^ (a - b) else n * 2 ^ (b - a) ≥ d
  if ge then (a : Int) - b else (a : Int) - b - 1

/-- Round the non-negative fraction `num / den` (`den > 0`) to an integer under `mode`; `neg`
is the sign of the value whose magnitude this is (the directed modes depend on it). -/
def roundDiv (mode : RoundMode) (neg : Bool) (num den : Nat) : Nat :=
  let fl := num / den
  let r := num % den
  match mode with
  | .rn => if 2 * r > den then fl + 1 else if 2 * r < den then fl else if fl % 2 == 0 then fl else fl + 1
  | .rna => if 2 * r ≥ den then fl + 1 else fl
  | .rz => fl
  | .ru => if r != 0 && !neg then fl + 1 else fl
  | .rd => if r != 0 && neg then fl + 1 else fl

variable (f : Format)

/-! ## Layout -/

/-- Total number of bits. -/
def width : Nat := f.e + f.m + 1
/-- The exponent bias `2^(e-1) − 1`. -/
def bias : Nat := 2 ^ (f.e - 1) - 1
/-- The all-ones exponent field. -/
def expMax : Nat := 2 ^ f.e - 1
/-- The all-ones fraction field. -/
def fracMax : Nat := 2 ^ f.m - 1
/-- The sign bit, `2^(e+m)`. -/
def signBit : Nat := 2 ^ (f.e + f.m)
/-- The unbiased exponent of the smallest normal (also the scale of the subnormals). -/
def emin : Int := 1 - (f.bias : Int)
/-- The unbiased exponent of the largest finite value. -/
def emax : Int :=
  match f.special with
  | .ieee => (f.expMax : Int) - 1 - f.bias
  | .fn | .finite => (f.expMax : Int) - f.bias

def sign (x : Nat) : Bool := x.testBit (f.e + f.m)
def expField (x : Nat) : Nat := (x >>> f.m) % 2 ^ f.e
def fracField (x : Nat) : Nat := x % 2 ^ f.m

/-! ## Classification and value -/

def isNaN (x : Nat) : Bool :=
  match f.special with
  | .ieee => f.expField x == f.expMax && f.fracField x != 0
  | .fn => f.expField x == f.expMax && f.fracField x == f.fracMax
  | .finite => false

def isInf (x : Nat) : Bool :=
  match f.special with
  | .ieee => f.expField x == f.expMax && f.fracField x == 0
  | .fn | .finite => false

def isZero (x : Nat) : Bool := f.expField x == 0 && f.fracField x == 0
def isFinite (x : Nat) : Bool := !f.isNaN x && !f.isInf x

/-- Exact magnitude of a finite pattern as a rational (`0` for `±0`): `fraction · 2^(emin − m)`
for a subnormal, `(2^m + fraction) · 2^(E − bias − m)` for a normal with exponent field `E`. -/
def magRat (x : Nat) : Rat :=
  let E := f.expField x
  let M := f.fracField x
  if E == 0 then mkRat M (2 ^ (f.bias + f.m - 1))
  else
    let sig := 2 ^ f.m + M
    if E ≥ f.bias + f.m then ((sig * 2 ^ (E - (f.bias + f.m)) : Nat) : Rat)
    else mkRat sig (2 ^ (f.bias + f.m - E))

/-- Exact value of a finite pattern. -/
def toRat (x : Nat) : Rat := if f.sign x then -f.magRat x else f.magRat x

/-! ## Special patterns (magnitudes have the sign bit clear) -/

def negZero : Nat := f.signBit
/-- The infinity magnitude (all-ones exponent, zero fraction); a value only for `ieee`. -/
def infMag : Nat := f.expMax <<< f.m
/-- The all-ones magnitude: a NaN of an `ieee` format and the NaN of an `fn` format. -/
def nanMag : Nat := (f.expMax <<< f.m) ||| f.fracMax
/-- The largest finite magnitude. -/
def maxFinite : Nat :=
  match f.special with
  | .ieee => ((f.expMax - 1) <<< f.m) ||| f.fracMax
  | .fn => (f.expMax <<< f.m) ||| (f.fracMax - 1)
  | .finite => (f.expMax <<< f.m) ||| f.fracMax

/-- Apply a sign to a magnitude. -/
def withSign (s : Bool) (mag : Nat) : Nat := if s then mag ||| f.signBit else mag % f.signBit

def negate (x : Nat) : Nat := x ^^^ f.signBit

/-! ## Rounding -/

/-- The rounded significand `m` (hidden bit included) at unbiased exponent `e'` lies above
the largest finite value. -/
def over (m : Nat) (e' : Int) : Bool :=
  match f.special with
  | .ieee | .finite => decide (e' > f.emax)
  | .fn => decide (e' > f.emax) || (decide (e' = f.emax) && m == 2 ^ (f.m + 1) - 1)

/-- The magnitude an overflowing result of sign `neg` takes under `mode`. -/
def overflow (mode : RoundMode) (sat neg : Bool) : Nat :=
  let toInf : Bool :=
    match mode with
    | .rn | .rna => true
    | .rz => false
    | .ru => !neg
    | .rd => neg
  if sat || !toInf then f.maxFinite
  else
    match f.special with
    | .ieee => f.infMag
    | .fn => f.nanMag
    | .finite => f.maxFinite

/-- Round a non-negative rational to a magnitude pattern of the format under `mode`; `neg` is
the sign of the value (for the directed modes and the overflow), `sat` clamps overflow to the
largest finite value. Subnormals are kept; `0` and negative inputs give `+0`. -/
def roundMag (mode : RoundMode) (sat neg : Bool) (q : Rat) : Nat :=
  if q ≤ 0 then 0
  else
    let n := q.num.natAbs
    let d := q.den
    let e := log2Floor n d
    let e' := max e f.emin
    let k : Int := e' - f.m                     -- quantum exponent
    let (num, den) : Nat × Nat :=
      if k ≥ 0 then (n, d * 2 ^ k.toNat) else (n * 2 ^ (-k).toNat, d)
    let m := roundDiv mode neg num den
    let (m, e') := if m == 2 ^ (f.m + 1) then (2 ^ f.m, e' + 1) else (m, e')
    if f.over m e' then f.overflow mode sat neg
    else if m < 2 ^ f.m then m                    -- zero or subnormal (e' = emin)
    else ((e' + f.bias).toNat <<< f.m) ||| (m - 2 ^ f.m)

/-- Round a signed rational. The sign goes on the rounded magnitude (so a tiny negative value
gives `-0`); an exact zero gives `-0` when `zeroNeg`, `+0` otherwise (IEEE: an exact zero sum
is `+0` except under `rd`). -/
def roundRat (mode : RoundMode) (sat zeroNeg : Bool) (q : Rat) : Nat :=
  if q == 0 then (if zeroNeg then f.negZero else 0)
  else f.withSign (q < 0) (f.roundMag mode sat (q < 0) (if q < 0 then -q else q))

/-! ## Operations (IEEE 754, one rounding each) -/

/-- The sign of an exact zero sum of two operands of signs `a` and `b`: negative when both are,
or when either is and the mode is `rd`. -/
def zeroSumNeg (mode : RoundMode) (a b : Bool) : Bool := (a && b) || (mode == .rd && (a || b))

/-- Exact-zero results that are not sums of two zeros (cancellation) are `+0` except under `rd`. -/
def cancelNeg (mode : RoundMode) : Bool := mode == .rd

def add (mode : RoundMode) (nan x y : Nat) : Nat :=
  if f.isNaN x || f.isNaN y then nan
  else if f.isInf x && f.isInf y then (if f.sign x == f.sign y then x else nan)
  else if f.isInf x then x
  else if f.isInf y then y
  else if f.isZero x && f.isZero y then
    (if zeroSumNeg mode (f.sign x) (f.sign y) then f.negZero else 0)
  else f.roundRat mode false (cancelNeg mode) (f.toRat x + f.toRat y)

def sub (mode : RoundMode) (nan x y : Nat) : Nat := f.add mode nan x (f.negate y)

def mul (mode : RoundMode) (nan x y : Nat) : Nat :=
  if f.isNaN x || f.isNaN y then nan
  else
    let s := f.sign x != f.sign y
    if f.isInf x || f.isInf y then (if f.isZero x || f.isZero y then nan else f.withSign s f.infMag)
    else if f.isZero x || f.isZero y then f.withSign s 0
    else f.withSign s (f.roundMag mode false s (f.magRat x * f.magRat y))

/-- `round(x·y + z)` with a single rounding. -/
def fma (mode : RoundMode) (nan x y z : Nat) : Nat :=
  if f.isNaN x || f.isNaN y || f.isNaN z then nan
  else
    let ps := f.sign x != f.sign y
    if f.isInf x || f.isInf y then
      if f.isZero x || f.isZero y then nan
      else if f.isInf z && f.sign z != ps then nan
      else f.withSign ps f.infMag
    else if f.isInf z then z
    else if f.isZero x || f.isZero y then
      if f.isZero z then (if zeroSumNeg mode ps (f.sign z) then f.negZero else 0) else z
    else f.roundRat mode false (cancelNeg mode) (f.toRat x * f.toRat y + f.toRat z)

def rcp (mode : RoundMode) (nan x : Nat) : Nat :=
  if f.isNaN x then nan
  else if f.isInf x then f.withSign (f.sign x) 0
  else if f.isZero x then f.withSign (f.sign x) f.infMag
  else f.withSign (f.sign x) (f.roundMag mode false (f.sign x) (f.magRat x)⁻¹)

def div (mode : RoundMode) (nan x y : Nat) : Nat :=
  if f.isNaN x || f.isNaN y then nan
  else
    let s := f.sign x != f.sign y
    if f.isInf x then (if f.isInf y then nan else f.withSign s f.infMag)
    else if f.isInf y then f.withSign s 0
    else if f.isZero y then (if f.isZero x then nan else f.withSign s f.infMag)
    else if f.isZero x then f.withSign s 0
    else f.withSign s (f.roundMag mode false s (f.magRat x / f.magRat y))

/-! ### Square roots

A square root is irrational in general, so it is not a `Rat` to give `roundMag`. For a finite
positive `x = M · 2^E` (`sigExp`), with `E` made even (`M' = 2M`, `E' = E - 1` when `E` is odd),
`√x = √M' · 2^(E'/2)`. With `N = M' · 4^j` and `r = ⌊√N⌋` (`Nat.sqrt`), `√x` lies in
`[r, r + 1) · 2^(E'/2 - j)`, at `r` exactly when `r² = N`. For `j = m + 3`, `r ≥ 2^(m+3)`, so a
unit of `r` is below an eighth of an ulp of the result and every rounding boundary (a multiple of
half an ulp) is an integer multiple of `2^(E'/2 - j)`: no boundary lies strictly between `r` and
`r + 1`. So `√x` rounds, under every mode, as `r` does when exact and as `r + 1/2` otherwise
(`sqrtProxy`). The reciprocal square root uses `N = ⌊4^j / M'⌋` (`j = 2m + 4`, so that
`√(4^j / M') ≥ 2^(m+2)` for every `M' < 2^(m+3)`), inexact also when `M'` does not divide `4^j`. -/

/-- `n · 2^e` as a rational. -/
def scale2 (n : Nat) (e : Int) : Rat :=
  if e ≥ 0 then ((n * 2 ^ e.toNat : Nat) : Rat) else mkRat n (2 ^ (-e).toNat)

/-- The integer significand and exponent of a finite pattern: its magnitude is `M · 2^E`. -/
def sigExp (x : Nat) : Nat × Int :=
  if f.expField x = 0 then (f.fracField x, f.emin - f.m)
  else (2 ^ f.m + f.fracField x, (f.expField x : Int) - f.bias - f.m)

/-- A rational with the rounding of `√(M · 2^E)` (`inv = false`) or `1/√(M · 2^E)` (`inv`), for
`M > 0`; see above. -/
def sqrtProxy (inv : Bool) (M : Nat) (E : Int) : Rat :=
  let (M', E') : Nat × Int := if E % 2 = 0 then (M, E) else (2 * M, E - 1)
  let h : Int := E' / 2
  if inv then
    let j := 2 * f.m + 4
    let N := 4 ^ j / M'
    let r := N.sqrt
    let exact := 4 ^ j % M' == 0 && r * r == N
    scale2 (2 * r + if exact then 0 else 1) (-(j : Int) - h - 1)
  else
    let j := f.m + 3
    let N := M' * 4 ^ j
    let r := N.sqrt
    let exact := r * r == N
    scale2 (2 * r + if exact then 0 else 1) (h - j - 1)

/-- `sqrt` with one rounding under `mode` (`sqrt.rn.f32`, `sqrtf` under `-prec-sqrt=true`,
`__fsqrt_rn/rz/rd/ru`): a NaN gives `nan`, `±0` is kept, a negative value (`-∞` included)
gives `nan`, `+∞` is kept. -/
def sqrt (mode : RoundMode) (nan x : Nat) : Nat :=
  if f.isNaN x then nan
  else if f.isZero x then x
  else if f.sign x then nan
  else if f.isInf x then x
  else
    let (M, E) := f.sigExp x
    f.roundMag mode false false (f.sqrtProxy false M E)

/-- `1/sqrt` with one rounding under `mode` (`__frsqrt_rn`): a NaN gives `nan`, `±0` gives `±∞`,
a negative value gives `nan`, `+∞` gives `+0`. -/
def rsqrt (mode : RoundMode) (nan x : Nat) : Nat :=
  if f.isNaN x then nan
  else if f.isZero x then f.withSign (f.sign x) f.infMag
  else if f.sign x then nan
  else if f.isInf x then 0
  else
    let (M, E) := f.sigExp x
    f.roundMag mode false false (f.sqrtProxy true M E)

/-- Round to an integral value of the format under `mode` (`cvt.rni/rzi/rmi/rpi.f32.f32`,
`FRND`: `rintf`/`nearbyintf` `.rn`, `truncf` `.rz`, `floorf` `.rd`, `ceilf` `.ru`; `roundf`, half
away from zero, `.rna`): a NaN gives `nan`, `±∞` and `±0` are kept, a value that rounds to `0`
gives the zero of its sign. -/
def roundInt (mode : RoundMode) (nan x : Nat) : Nat :=
  if f.isNaN x then nan
  else if f.isInf x || f.isZero x then x
  else
    let q := f.magRat x
    let n := roundDiv mode (f.sign x) q.num.natAbs q.den
    f.withSign (f.sign x) (f.roundMag .rn false (f.sign x) n)

/-- `x` with the sign of `y` (`copysignf`; a bit operation: a NaN keeps its payload). -/
def copySign (x y : Nat) : Nat := f.withSign (f.sign y) (x % f.signBit)

/-- Numeric comparison: `none` when a NaN is involved, `-0 = +0`. -/
def compare (x y : Nat) : Option Ordering :=
  if f.isNaN x || f.isNaN y then none
  else if f.isZero x && f.isZero y then some .eq
  else some (Ord.compare (ieeeOrderKey f.e f.m x) (ieeeOrderKey f.e f.m y))

def lt (x y : Nat) : Bool := f.compare x y == some .lt

/-- `max` without `.NaN`: one NaN operand gives the other operand, two give `nan`;
`max(-0, +0) = +0`. -/
def fmax (nan x y : Nat) : Nat :=
  if f.isNaN x && f.isNaN y then nan
  else if f.isNaN x then y
  else if f.isNaN y then x
  else if f.isZero x && f.isZero y then (if f.sign x && f.sign y then f.negZero else 0)
  else if f.lt x y then y else x

/-- `min` without `.NaN`; `min(-0, +0) = -0`. -/
def fmin (nan x y : Nat) : Nat :=
  if f.isNaN x && f.isNaN y then nan
  else if f.isNaN x then y
  else if f.isNaN y then x
  else if f.isZero x && f.isZero y then (if f.sign x || f.sign y then f.negZero else 0)
  else if f.lt x y then x else y

/-- `max` with `.NaN`: a NaN operand gives `nan`; otherwise `fmax`. -/
def fmaxNaN (nan x y : Nat) : Nat := if f.isNaN x || f.isNaN y then nan else f.fmax nan x y

/-- `min` with `.NaN`: a NaN operand gives `nan`; otherwise `fmin`. -/
def fminNaN (nan x y : Nat) : Nat := if f.isNaN x || f.isNaN y then nan else f.fmin nan x y

/-! ## Sign operations and clamps with an explicit NaN result -/

/-- The pattern of `+1`. -/
def one : Nat := f.bias <<< f.m

/-- Sign flip; a NaN operand gives `nan` (instructions that canonicalise, unlike a bit flip). -/
def negNaN (nan x : Nat) : Nat := if f.isNaN x then nan else f.negate x

/-- Sign cleared; a NaN operand gives `nan`. -/
def absNaN (nan x : Nat) : Nat := if f.isNaN x then nan else f.withSign false x

/-- `.sat` (PTX): clamp to `[+0, 1]`. A NaN gives `+0`, and so does every pattern with the sign
set (negative values, `-∞` and `-0`); values above `1` (and `+∞`) give `1`. -/
def clampUnit (x : Nat) : Nat :=
  if f.isNaN x then 0
  else if f.sign x then 0
  else if f.lt f.one x then f.one
  else x

/-- `.relu` (PTX): every pattern with the sign set (negative values, `-∞`, `-0`) gives `+0`; a NaN
gives `nan`; other values are kept. -/
def relu (nan x : Nat) : Nat := if f.isNaN x then nan else if f.sign x then 0 else x

end Format

/-! ## Conversions -/

/-- What a conversion produces for a NaN input `x`. -/
inductive NaNRule where
  /-- The destination's positive all-ones pattern (`0x7fff` for bf16 and f16, `0x7fffffff` for
  f32, `0x7f` for e4m3 and e5m2): what NVIDIA devices produce for the conversions the library
  models, except the exact `bf16 → f32` shift (`docs/evidence/conformance/cvt.md`). -/
  | canonical
  /-- The input's sign, the all-ones exponent and only the top fraction bit (`0x7e00` or `0xfe00`
  for f16): the IEEE-style quiet NaN, kept for other devices; no NVIDIA conversion the library
  models gives it. -/
  | quietSigned
  /-- The input's sign and fraction, the fraction shifted to the top of the wider destination;
  `quiet` also sets the top fraction bit (exact widening, e.g. `bf16 → f32`). -/
  | widen (quiet : Bool)
  /-- A fixed pattern. -/
  | fixed (bits : Nat)
  deriving Repr, DecidableEq

/-- The NaN pattern a conversion from `src` to `dst` produces for the NaN input `x`. -/
def NaNRule.apply (r : NaNRule) (src dst : Format) (x : Nat) : Nat :=
  match r with
  | .canonical => dst.nanMag
  | .quietSigned => dst.withSign (src.sign x) (dst.infMag ||| 2 ^ (dst.m - 1))
  | .widen quiet =>
      dst.withSign (src.sign x)
        (dst.infMag ||| (if quiet then 2 ^ (dst.m - 1) else 0) ||| (src.fracField x <<< (dst.m - src.m)))
  | .fixed b => b

namespace Format

/-- Convert a pattern of `src` into `dst`: NaN by the rule; `±∞` to `±∞` (the NaN of an `fn`
destination, the largest finite value of a `finite` one or under `sat`); zeros keep their sign;
a finite value is rounded under `mode` (with `sat` clamping overflow), the sign kept even when
the magnitude rounds to zero. -/
def cvt (src dst : Format) (mode : RoundMode) (sat : Bool) (nan : NaNRule) (x : Nat) : Nat :=
  let s := src.sign x
  if src.isNaN x then nan.apply src dst x
  else if src.isInf x then
    dst.withSign s
      (if sat then dst.maxFinite
       else match dst.special with
         | .ieee => dst.infMag
         | .fn => dst.nanMag
         | .finite => dst.maxFinite)
  else if src.isZero x then dst.withSign s 0
  else dst.withSign s (dst.roundMag mode sat s (src.magRat x))

/-- Round an exact rational into `dst` (e.g. an integer conversion `cvt.rn.f32.s32`); `+0` for
zero. -/
def ofRat (dst : Format) (mode : RoundMode) (sat : Bool) (q : Rat) : Nat :=
  dst.roundRat mode sat false q

/-- The range `[lo, hi]` of a `width`-bit integer type, signed or unsigned. -/
def intRange (signed : Bool) (width : Nat) : Int × Int :=
  if signed then (-((2 ^ (width - 1) : Nat) : Int), ((2 ^ (width - 1) - 1 : Nat) : Int))
  else (0, ((2 ^ width - 1 : Nat) : Int))

/-- The integer that a pattern of `src` rounds to under `mode`, clamped to the range of the
`width`-bit integer type (`signed` or not), as a `width`-bit two's-complement pattern: PTX
float-to-integer `cvt.rni/rzi/rmi/rpi[.sat]` (which saturates). `±∞` give the bounds and a NaN
gives `0`. -/
def toInt (src : Format) (mode : RoundMode) (signed : Bool) (width : Nat) (x : Nat) : Nat :=
  let (lo, hi) := intRange signed width
  let v : Int :=
    if src.isNaN x then 0
    else if src.isInf x then (if src.sign x then lo else hi)
    else
      let q := src.magRat x
      let n : Int := (roundDiv mode (src.sign x) q.num.natAbs q.den : Nat)
      if src.sign x then -n else n
  (max lo (min hi v) % ((2 ^ width : Nat) : Int)).toNat

/-- An integer clamped to the range of the `width`-bit integer type, as a pattern (PTX integer
`cvt.sat`). -/
def satInt (signed : Bool) (width : Nat) (v : Int) : Nat :=
  let (lo, hi) := intRange signed width
  (max lo (min hi v) % ((2 ^ width : Nat) : Int)).toNat

/-! ## Named formats -/

/-- IEEE binary64 (`double`). -/
def binary64 : Format := ⟨11, 52, .ieee⟩
/-- IEEE binary32 (`float`, `.f32`). -/
def binary32 : Format := ⟨8, 23, .ieee⟩
/-- IEEE binary16 (`__half`, `.f16`). -/
def binary16 : Format := ⟨5, 10, .ieee⟩
/-- bfloat16 (`__nv_bfloat16`, `.bf16`). -/
def bfloat16 : Format := ⟨8, 7, .ieee⟩
/-- TensorFloat-32 (`.tf32`, stored in the high 19 bits of a 32-bit register). -/
def tf32 : Format := ⟨8, 10, .ieee⟩
/-- OCP FP8 E5M2 (`.e5m2`, `__nv_fp8_e5m2`): IEEE-like, max `57344`. -/
def e5m2 : Format := ⟨5, 2, .ieee⟩
/-- OCP FP8 E4M3 (`.e4m3`, `__nv_fp8_e4m3`, "E4M3FN"): no `∞`, NaN `S.1111.111`, max `448`. -/
def e4m3 : Format := ⟨4, 3, .fn⟩
/-- OCP MX FP4 E2M1 (`.e2m1`): no `∞`, no NaN, max `6`. -/
def e2m1 : Format := ⟨2, 1, .finite⟩
/-- OCP MX FP6 E2M3 (`.e2m3`): no `∞`, no NaN, max `7.5`. -/
def e2m3 : Format := ⟨2, 3, .finite⟩
/-- OCP MX FP6 E3M2 (`.e3m2`): no `∞`, no NaN, max `28`. -/
def e3m2 : Format := ⟨3, 2, .finite⟩

end Format

end TVA.Math.Float
