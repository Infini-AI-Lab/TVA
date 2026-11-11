/-!
# Binary floating-point formats and what a bit pattern means

A *format* with `e` exponent bits and `m` fraction bits stores a value in `1 + e + m` bits:
the sign `s` (the top bit), the exponent field `E` (the next `e` bits) and the fraction field
`F` (the low `m` bits). With the bias `b = 2^(e-1) - 1` (IEEE 754-2019 §3.3; the OCP 8-bit and
4-bit formats use the same layout):
- `E = 0`: the *subnormal* value `F · 2^(1 - b - m)`; both zeros (`±0`) are the case `F = 0`;
- `E > 0`: the *normal* value `(2^m + F) · 2^(E - b - m)`, that is `1.F × 2^(E - b)`;
- the sign bit negates the value.

What a format does with the largest exponent field is its `Specials`:
- `.ieee` (`binary16`, `bfloat16`, `binary32`, `binary64`, and OCP `E5M2`): an all-ones exponent
  field is `±∞` when `F = 0` and NaN otherwise;
- `.nanOnly` (OCP `E4M3`, Triton's `float8e4nv`): no infinities; only the all-ones exponent with
  the all-ones fraction is NaN (`0x7f`, `0xff`), the other patterns of that exponent are numbers;
- `.finite` (OCP `E2M1`, the fp4 of `tl.dot_scaled`): every pattern is a number.

Patterns are natural numbers below `2^width` here; `FloatV` (`TVA.Base.Float.Value`) wraps them
as `BitVec`. Every definition is computable on rationals, so constants can be checked by
`decide` or `#eval`.

* `toRat`: the exact value of a pattern. For `±∞` the formula gives `±2^(emax+1)` (`emax = b`),
  beyond the largest finite value `maxVal`, so comparing values orders the infinities below and
  above every number. A NaN's `toRat` means nothing: every operation tests for NaN first.
* `maxVal`, `u = 2^-(m+1)` (the relative error of one rounding to nearest), `eta = 2^(emin-m-1)`
  (its absolute error below the normal range), `emin = 1 - b`.

Rounding into a format is `TVA.Base.Float.Round`; the IEEE operations are `TVA.Base.Float.Arith`.
-/
namespace TVA

/-- `2^k` as a rational: `2^k`, or `1 / 2^-k` built in lowest terms (so that it computes fast). -/
def Format.pow2 (k : Int) : Rat :=
  if 0 ≤ k then ((2 ^ k.toNat : Nat) : Rat)
  else ⟨1, 2 ^ (-k).toNat, Nat.ne_of_gt (Nat.two_pow_pos _), Nat.gcd_one_left _⟩

/-- What a format does with the patterns of its all-ones exponent field. -/
inductive Format.Specials where
  /-- An all-ones exponent is an infinity (zero fraction) or a NaN. -/
  | ieee
  /-- No infinities; only the all-ones exponent with the all-ones fraction is NaN. -/
  | nanOnly
  /-- Every pattern is a finite number. -/
  | finite
  deriving DecidableEq, Repr

/-- A binary floating-point format: `e` exponent bits, `m` fraction bits, and its specials. -/
structure Format where
  /-- Exponent bits. -/
  e : Nat
  /-- Fraction bits (the significand has `m + 1` bits with the implicit leading one). -/
  m : Nat
  /-- What the all-ones exponent field means. -/
  specials : Format.Specials
  deriving DecidableEq, Repr

namespace Format

/-- IEEE `binary16`: `tl.float16`. -/
def binary16 : Format := ⟨5, 10, .ieee⟩
/-- `bfloat16`: `tl.bfloat16`, `binary32`'s exponent range with 7 fraction bits. -/
def bfloat16 : Format := ⟨8, 7, .ieee⟩
/-- IEEE `binary32`: `tl.float32`. -/
def binary32 : Format := ⟨8, 23, .ieee⟩
/-- IEEE `binary64`: `tl.float64`. -/
def binary64 : Format := ⟨11, 52, .ieee⟩
/-- OCP `E5M2`: `tl.float8e5`, with infinities and NaNs as in IEEE; largest value `57344`. -/
def e5m2 : Format := ⟨5, 2, .ieee⟩
/-- OCP `E4M3FN`: `tl.float8e4nv`, no infinities, NaN `0x7f`/`0xff`, largest value `448`. -/
def e4m3 : Format := ⟨4, 3, .nanOnly⟩
/-- OCP `E2M1`: the fp4 elements of `tl.dot_scaled`, every pattern finite, largest value `6`. -/
def e2m1 : Format := ⟨2, 1, .finite⟩

variable (f : Format)

/-- Bits of a pattern: `1 + e + m`. -/
def width : Nat := f.e + f.m + 1
/-- The exponent bias `2^(e-1) - 1`. -/
def bias : Int := 2 ^ (f.e - 1) - 1
/-- The exponent of the smallest normal value, `1 - bias` (also that of the subnormals). -/
def emin : Int := 1 - f.bias
/-- The sign bit, `2^(e+m)`. -/
def signBit : Nat := 2 ^ (f.e + f.m)

/-- The sign of a pattern (`true`: negative). -/
def sign (n : Nat) : Bool := n / f.signBit % 2 == 1
/-- The *magnitude* of a pattern: its exponent and fraction fields, without the sign. -/
def mag (n : Nat) : Nat := n % f.signBit
/-- The exponent field `E`. -/
def expField (n : Nat) : Nat := f.mag n / 2 ^ f.m
/-- The fraction field `F`. -/
def fracField (n : Nat) : Nat := f.mag n % 2 ^ f.m

/-- The pattern is a NaN. -/
def isNaN (n : Nat) : Bool :=
  match f.specials with
  | .ieee => f.expField n == 2 ^ f.e - 1 && f.fracField n != 0
  | .nanOnly => f.mag n == 2 ^ (f.e + f.m) - 1
  | .finite => false

/-- The pattern is `+∞` or `-∞`. -/
def isInf (n : Nat) : Bool :=
  match f.specials with
  | .ieee => f.expField n == 2 ^ f.e - 1 && f.fracField n == 0
  | .nanOnly | .finite => false

/-- The pattern is a number: neither NaN nor infinite. -/
def isFinite (n : Nat) : Bool := !f.isNaN n && !f.isInf n

/-- The pattern is `+0` or `-0`. -/
def isZero (n : Nat) : Bool := f.mag n == 0

/-- The value of a magnitude `M` (exponent field `M / 2^m`, fraction `M mod 2^m`): subnormal
`F · 2^(emin − m)` for exponent field `0`, normal `(2^m + F) · 2^(E − bias − m)` otherwise. -/
def magVal (M : Nat) : Rat :=
  if M / 2 ^ f.m = 0 then ((M % 2 ^ f.m : Nat) : Rat) * pow2 (f.emin - f.m)
  else ((2 ^ f.m + M % 2 ^ f.m : Nat) : Rat) * pow2 (((M / 2 ^ f.m : Nat) : Int) - f.bias - f.m)

/-- The exact value of a pattern: of a finite one, the number it encodes (`0` for both zeros);
of `±∞`, `±2^(bias+1)`, beyond `±maxVal`; of a NaN, a meaningless number. -/
def toRat (n : Nat) : Rat := if f.sign n then -f.magVal (f.mag n) else f.magVal (f.mag n)

/-- The pattern with sign `s` and the magnitude of `n`. -/
def setSign (s : Bool) (n : Nat) : Nat := (if s then f.signBit else 0) + f.mag n

/-- The largest exponent field of a finite value. -/
def topExp : Nat :=
  match f.specials with
  | .ieee => 2 ^ f.e - 2
  | .nanOnly | .finite => 2 ^ f.e - 1

/-- The largest significand (`2^m + F`) of a finite value at exponent field `topExp`. -/
def topSig : Nat :=
  match f.specials with
  | .nanOnly => 2 ^ (f.m + 1) - 2
  | .ieee | .finite => 2 ^ (f.m + 1) - 1

/-- The magnitude of the largest finite value. -/
def maxMag : Nat := f.topExp * 2 ^ f.m + (f.topSig - 2 ^ f.m)
/-- The largest finite value (`65504` in `binary16`, `448` in `e4m3`). -/
def maxVal : Rat := (f.topSig : Rat) * pow2 ((f.topExp : Int) - f.bias - f.m)
/-- The magnitude of an infinity (in an `.ieee` format): all-ones exponent, zero fraction. -/
def infMag : Nat := (2 ^ f.e - 1) * 2 ^ f.m
/-- The magnitude of the canonical NaN: all exponent and fraction bits set (`0x7fffffff` in
`binary32`, `0x7fff` in `binary16` and `bfloat16`, `0x7f` in both fp8 formats). -/
def nanMag : Nat := 2 ^ (f.e + f.m) - 1
/-- Half an ulp of `1`, `2^-(m+1)`: the relative error of one rounding to nearest (`2^-24` in
`binary32`). -/
def u : Rat := pow2 (-(f.m : Int) - 1)
/-- Half the smallest subnormal, `2^(emin - m - 1)`: the absolute error of one rounding to
nearest below the normal range (`2^-150` in `binary32`). -/
def eta : Rat := pow2 (f.emin - f.m - 1)

/-- The formats the error bounds cover: at least two exponent bits and one fraction bit (every
format above). -/
def Regular : Prop := 2 ≤ f.e ∧ 1 ≤ f.m

instance : Decidable f.Regular := by unfold Regular; exact inferInstance

/-- The order key of a pattern: its magnitude, negated when the sign is set (both zeros at `0`).
Non-NaN patterns are ordered by value exactly as by key. -/
def orderKey (n : Nat) : Int := if f.sign n then -(f.mag n : Int) else (f.mag n : Int)

/-- `g` holds every finite value of `f`: at least as many fraction bits, a quantum at least as
fine at the bottom, and a range at least as large. Then a conversion from `f` into `g` is
exact. -/
def Wider (f g : Format) : Prop := f.m ≤ g.m ∧ g.emin - g.m ≤ f.emin - f.m ∧ f.maxVal ≤ g.maxVal

instance (f g : Format) : Decidable (Wider f g) := by unfold Wider; exact inferInstance

/-- The standard float dtypes: `float16`, `bfloat16`, `float32`, `float64`. -/
def Std (f : Format) : Prop :=
  f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32 ∨ f = .binary64

/-- The fp8 dtypes: `float8e5` and `float8e4nv`. -/
def F8 (f : Format) : Prop := f = .e5m2 ∨ f = .e4m3

instance (f : Format) : Decidable f.Std := by unfold Std; exact inferInstance
instance (f : Format) : Decidable f.F8 := by unfold F8; exact inferInstance

end Format

end TVA
