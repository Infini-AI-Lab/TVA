/-!
# TVA's float layer: the definitions

Every definition of the float layer, in core Lean only (no Mathlib), so that the very same
constants run as native code in TVA's evidence (`evidence/float-layer`, which compiles this
module with `lean -c` and compares its results with the device). The proofs about them are in
`TVA.Math.Encoding`, `TVA.Math.Rounding` and `TVA.Math.Arith`, which import Mathlib; the
module docstrings there say what each definition means.

* Native powers: `powFast`, `intPowFast`, what compiled code runs for `^` (proved equal to it,
  `@[csimp]`); `pow2` (`2^k` as a rational).
* Formats and patterns: `Specials`, `Format` (`binary16`, `bfloat16`, `binary32`, `binary64`,
  `e5m2`, `e4m3`, `e2m1`), `width`, `bias`, `emin`, `sign`, `mag`, `expField`, `fracField`,
  `isNaN`, `isInf`, `isFinite`, `isZero`, `magVal`, `toRat`, `setSign`, `maxVal`, `maxMag`,
  `infMag`, `nanMag`, `u`, `eta`, `Regular`, `intRange`, `orderKey`.
* Rounding: `RoundMode`, `cmpq`, `rint`, `rnat`, `qlog2` (`⌊log₂ a⌋`), `qexp`, `pack`, `encode`,
  `roundSig`, `roundMag`, `roundRat`, `sqrtMag`.
* Operations: `infPat`, `zeroPat`, `neg`, `absPat`, `add`, `sub`, `mul`, `fma`, `div`, `sqrt`,
  `maxNum`, `minNum`, `maxNaN`, `minNaN`, `cmp`, `cvt`, `Wider`.
-/
namespace TVA.Math.Float

/-! ## Native powers

Compiled code runs `b ^ e` on natural numbers and integers as `powFast` and `intPowFast`, the
same functions (`pow_eq_powFast`, `intPow_eq_intPowFast`), through `@[csimp]` replacements of
the two `NatPow` instances, which the compiler only accepts with those proofs: `2 ^ e` below
`2^63` is a 64-bit shift instead of a big-number power. Proofs are unaffected. -/

/-- `b ^ e`, with `2 ^ e` for `e < 63` as a 64-bit shift. -/
def powFast (b e : Nat) : Nat :=
  if b = 2 ∧ e < 63 then ((1 : UInt64) <<< UInt64.ofNat e).toNat else Nat.pow b e

theorem pow_eq_powFast : @Nat.pow = @powFast := by
  funext b e
  unfold powFast
  split
  · rename_i h
    obtain ⟨rfl, he⟩ := h
    have h64 : 2 ^ e < 2 ^ 64 := Nat.pow_lt_pow_right (by decide) (by omega)
    rw [UInt64.toNat_shiftLeft, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega : e < 2 ^ 64),
      Nat.mod_eq_of_lt (by omega : e < 64), show (1 : UInt64).toNat = 1 from rfl, Nat.one_shiftLeft,
      Nat.mod_eq_of_lt h64]
    rfl
  · rfl

/-- The `NatPow Nat` instance with `powFast`. -/
@[instance_reducible] def natPowFast : NatPow Nat := ⟨powFast⟩

@[csimp] theorem instNatPowNat_eq_natPowFast : @instNatPowNat = @natPowFast := by
  unfold natPowFast; rw [← pow_eq_powFast]; rfl

/-- `b ^ e` on integers, with `powFast` for a nonnegative base. -/
def intPowFast : Int → Nat → Int
  | .ofNat m, e => .ofNat (powFast m e)
  | b, e => Int.pow b e

theorem intPow_eq_intPowFast : @Int.pow = @intPowFast := by
  funext b e
  cases b with
  | ofNat m => simp only [intPowFast, ← pow_eq_powFast]; rfl
  | negSucc m => rfl

/-- The `NatPow Int` instance with `intPowFast`. -/
@[instance_reducible] def intNatPowFast : NatPow Int := ⟨intPowFast⟩

@[csimp] theorem instNatPowInt_eq_intNatPowFast : @Int.instNatPow = @intNatPowFast := by
  unfold intNatPowFast; rw [← intPow_eq_intPowFast]; rfl

/-- `2^k` as a rational: `2^k`, or `1 / 2^-k` built in lowest terms. -/
def pow2 (k : Int) : Rat :=
  if 0 ≤ k then ((2 ^ k.toNat : Nat) : Rat)
  else ⟨1, 2 ^ (-k).toNat, Nat.ne_of_gt (Nat.two_pow_pos _), Nat.gcd_one_left _⟩

/-- What a format does with the patterns that are not finite numbers. -/
inductive Specials where
  /-- An all-ones exponent is an infinity (zero fraction) or a NaN. -/
  | ieee
  /-- No infinities; only the all-ones exponent with the all-ones fraction is NaN. -/
  | nanOnly
  /-- Every pattern is a finite number. -/
  | finite
  deriving DecidableEq, Repr

/-- A binary floating-point format: `e` exponent bits, `m` fraction bits, and its specials. -/
structure Format where
  e : Nat
  m : Nat
  specials : Specials
  deriving DecidableEq, Repr

namespace Format

/-- IEEE `binary16` (`tl.float16`). -/
def binary16 : Format := ⟨5, 10, .ieee⟩
/-- `bfloat16` (`tl.bfloat16`): `binary32`'s exponent, 7 fraction bits. -/
def bfloat16 : Format := ⟨8, 7, .ieee⟩
/-- IEEE `binary32` (`tl.float32`). -/
def binary32 : Format := ⟨8, 23, .ieee⟩
/-- IEEE `binary64` (`tl.float64`). -/
def binary64 : Format := ⟨11, 52, .ieee⟩
/-- OCP `E5M2` (`tl.float8e5`). -/
def e5m2 : Format := ⟨5, 2, .ieee⟩
/-- OCP `E4M3` (`tl.float8e4nv`): no infinities, NaN `0x7f`/`0xff`, largest value `448`. -/
def e4m3 : Format := ⟨4, 3, .nanOnly⟩
/-- OCP `E2M1` (fp4): every pattern finite, largest value `6`. -/
def e2m1 : Format := ⟨2, 1, .finite⟩

variable (f : Format)

/-- Bits of a pattern. -/
def width : Nat := f.e + f.m + 1
/-- The exponent bias `2^(e-1) - 1`. -/
def bias : Int := 2 ^ (f.e - 1) - 1
/-- The exponent of the smallest normal value, `1 - bias`. -/
def emin : Int := 1 - f.bias
/-- The sign bit, `2^(e+m)`. -/
def signBit : Nat := 2 ^ (f.e + f.m)

/-- The sign of a pattern (`true`: negative). -/
def sign (n : Nat) : Bool := n / f.signBit % 2 == 1
/-- The pattern without its sign: the exponent and fraction fields. -/
def mag (n : Nat) : Nat := n % f.signBit
/-- The exponent field. -/
def expField (n : Nat) : Nat := f.mag n / 2 ^ f.m
/-- The fraction field. -/
def fracField (n : Nat) : Nat := f.mag n % 2 ^ f.m

/-- A NaN pattern. -/
def isNaN (n : Nat) : Bool :=
  match f.specials with
  | .ieee => f.expField n == 2 ^ f.e - 1 && f.fracField n != 0
  | .nanOnly => f.mag n == 2 ^ (f.e + f.m) - 1
  | .finite => false

/-- An infinite pattern. -/
def isInf (n : Nat) : Bool :=
  match f.specials with
  | .ieee => f.expField n == 2 ^ f.e - 1 && f.fracField n == 0
  | .nanOnly | .finite => false

/-- A finite pattern: neither NaN nor infinite. -/
def isFinite (n : Nat) : Bool := !f.isNaN n && !f.isInf n

/-- A zero (`+0` or `-0`). -/
def isZero (n : Nat) : Bool := f.mag n == 0

/-- The value of a magnitude `M`: exponent field `M / 2^m`, fraction `M mod 2^m`; subnormal
`F · 2^(emin − m)` for exponent field `0`, normal `(2^m + F) · 2^(E − bias − m)` otherwise. -/
def magVal (M : Nat) : Rat :=
  if M / 2 ^ f.m = 0 then ((M % 2 ^ f.m : Nat) : Rat) * pow2 (f.emin - f.m)
  else ((2 ^ f.m + M % 2 ^ f.m : Nat) : Rat) * pow2 (((M / 2 ^ f.m : Nat) : Int) - f.bias - f.m)

/-- The exact value of a pattern: of a finite one, its number; of `±∞`, `±2^(emax+1)` with
`emax` the bias (`toRat_of_isInf`), beyond `±maxVal`, which orders the infinities below and above
every number; of a NaN, the formula's value, which means nothing (every operation tests for NaN
first). -/
def toRat (n : Nat) : Rat := if f.sign n then -f.magVal (f.mag n) else f.magVal (f.mag n)

/-- The pattern with sign `s` and the magnitude of `n`. -/
def setSign (s : Bool) (n : Nat) : Nat := (if s then f.signBit else 0) + f.mag n

/-- The largest exponent field of a finite value. -/
def topExp : Nat :=
  match f.specials with
  | .ieee => 2 ^ f.e - 2
  | .nanOnly | .finite => 2 ^ f.e - 1

/-- The largest significand at `topExp`. -/
def topSig : Nat :=
  match f.specials with
  | .nanOnly => 2 ^ (f.m + 1) - 2
  | .ieee | .finite => 2 ^ (f.m + 1) - 1

/-- The magnitude of the largest finite value. -/
def maxMag : Nat := f.topExp * 2 ^ f.m + (f.topSig - 2 ^ f.m)
/-- The largest finite value. -/
def maxVal : Rat := (f.topSig : Rat) * pow2 ((f.topExp : Int) - f.bias - f.m)
/-- The magnitude of an infinity (`.ieee` formats). -/
def infMag : Nat := (2 ^ f.e - 1) * 2 ^ f.m
/-- The magnitude of the library's canonical NaN: all exponent and fraction bits set
(`0x7fffffff` in `binary32`, `0x7fff` in `binary16`/`bfloat16`, `0x7f` in both fp8 formats). -/
def nanMag : Nat := 2 ^ (f.e + f.m) - 1
/-- Half an ulp of `1`: `2^-(m+1)`, the relative error of one rounding to nearest. -/
def u : Rat := pow2 (-(f.m : Int) - 1)
/-- Half the smallest subnormal: `2^(emin - m - 1)`, the absolute error of one rounding to
nearest below the normal range. -/
def eta : Rat := pow2 (f.emin - f.m - 1)

/-- The formats the error lemmas cover: at least two exponent bits and one fraction bit. -/
def Regular : Prop := 2 ≤ f.e ∧ 1 ≤ f.m

instance : Decidable f.Regular := by unfold Regular; exact inferInstance

/-- The range of a `w`-bit integer type, signed (`sg`) or unsigned: `(min, max)`. -/
def intRange (sg : Bool) (w : Nat) : Int × Int :=
  if sg then (-2 ^ (w - 1), 2 ^ (w - 1) - 1) else (0, 2 ^ w - 1)

/-- The order key: the magnitude, negated for a negative pattern (both zeros at `0`). Values are
ordered as their keys (`toRat_lt_iff_orderKey`). -/
def orderKey (n : Nat) : Int := if f.sign n then -(f.mag n : Int) else (f.mag n : Int)

end Format

/-! ## Rounding -/

/-- The rounding modes Triton uses: `"rtne"` and `"rtz"`. -/
inductive RoundMode where
  /-- To nearest, ties to even. -/
  | rn
  /-- Toward zero. -/
  | rz
  deriving DecidableEq, Repr

/-- How `a` compares with `b`. -/
def cmpq (a b : Rat) : Ordering := if a < b then .lt else if a = b then .eq else .gt

/-- Round a nonnegative number to an integer, from its integer part `k` and how its fractional
part compares with `1/2` (`c`). -/
def rint (rnd : RoundMode) (k : Nat) (c : Ordering) : Nat :=
  match rnd, c with
  | .rz, _ => k
  | .rn, .lt => k
  | .rn, .gt => k + 1
  | .rn, .eq => if k % 2 = 0 then k else k + 1

/-- `x ≥ 0` rounded to an integer. -/
def rnat (rnd : RoundMode) (x : Rat) : Nat :=
  rint rnd (Rat.floor x).toNat (cmpq (x - ((Rat.floor x).toNat : Rat)) (1 / 2))

/-- `⌊log₂ a⌋` for `a > 0`, from the bit lengths of the numerator and the denominator. -/
def qlog2 (a : Rat) : Int :=
  let L : Int := (Nat.log2 a.num.toNat : Int) - (Nat.log2 a.den : Int)
  if pow2 L ≤ a then L else L - 1

namespace Format

variable (f : Format)

/-- The exponent of the quantum for a value `a > 0`: `max(⌊log₂ a⌋, emin) − m`. -/
def qexp (a : Rat) : Int := max (qlog2 a) f.emin - f.m

/-- The magnitude of `N · 2^Q`, for a significand from `roundSig`: `N < 2^m` (subnormal, with
`Q = emin − m`), `2^m ≤ N < 2^(m+1)`, or `N = 2^(m+1)` (a carry into the next binade). -/
def pack (N : Nat) (Q : Int) : Nat :=
  if N < 2 ^ f.m then N
  else if N < 2 ^ (f.m + 1) then (Q + f.m + f.bias).toNat * 2 ^ f.m + (N - 2 ^ f.m)
  else (Q + f.m + 1 + f.bias).toNat * 2 ^ f.m

/-- The magnitude of a rounded value `N · 2^Q`: `pack` when it is at most `maxVal`; otherwise an
overflow, which gives the largest finite magnitude when saturating or rounding toward zero, and
otherwise the infinity (`.ieee`) or the NaN (`.nanOnly`). -/
def encode (rnd : RoundMode) (sat : Bool) (N : Nat) (Q : Int) : Nat :=
  if (N : Rat) * pow2 Q ≤ f.maxVal then f.pack N Q
  else if sat || rnd == .rz then f.maxMag
  else match f.specials with
    | .ieee => f.infMag
    | .nanOnly => f.nanMag
    | .finite => f.maxMag

/-- The significand of `a > 0` rounded at the quantum `2^Q`. -/
def roundSig (rnd : RoundMode) (a : Rat) (Q : Int) : Nat := rnat rnd (a * pow2 (-Q))

/-- The magnitude `a > 0` rounds to. -/
def roundMag (rnd : RoundMode) (sat : Bool) (a : Rat) : Nat :=
  f.encode rnd sat (roundSig rnd a (f.qexp a)) (f.qexp a)

/-- **The pattern the exact value `q` rounds to**: to nearest even (`.rn`) or toward zero
(`.rz`); an overflow saturates when `sat`; `zs` is the sign of a zero result of `q = 0`, and a
nonzero `q` keeps its own sign. -/
def roundRat (rnd : RoundMode) (sat zs : Bool) (q : Rat) : Nat :=
  if q = 0 then (if zs then f.signBit else 0)
  else (if q < 0 then f.signBit else 0) + f.roundMag rnd sat (if q < 0 then -q else q)

/-- The magnitude `√a` rounds to, for `a > 0`: the quantum exponent from `⌊log₂ √a⌋ =
⌊⌊log₂ a⌋ / 2⌋`, the integer part of `√a / 2^Q` from the integer square root of
`⌊a / 4^Q⌋`, and its fraction compared with `1/2` by squaring. -/
def sqrtMag (rnd : RoundMode) (a : Rat) : Nat :=
  let Q := max (qlog2 a / 2) f.emin - f.m
  let t := a * pow2 (-2 * Q)
  let k := Nat.sqrt (Rat.floor t).toNat
  f.encode rnd false (rint rnd k (cmpq t (((k : Rat) + 1 / 2) * ((k : Rat) + 1 / 2)))) Q

/-! ## Operations -/

/-- An infinite result with sign `s`: `±∞`; NaN in a `.nanOnly` format; `±maxVal` in a
`.finite` one. -/
def infPat (s : Bool) : Nat :=
  match f.specials with
  | .ieee => f.setSign s f.infMag
  | .nanOnly => f.nanMag
  | .finite => f.setSign s f.maxMag

/-- The zero with sign `s`. -/
def zeroPat (s : Bool) : Nat := f.setSign s 0

/-- `-x`: the sign bit flipped. -/
def neg (x : Nat) : Nat := f.setSign (!f.sign x) x

/-- `|x|`: the sign bit cleared. -/
def absPat (x : Nat) : Nat := f.mag x

/-- `x + y`. -/
def add (rnd : RoundMode) (x y : Nat) : Nat :=
  if f.isNaN x || f.isNaN y then f.nanMag
  else if f.isInf x then
    (if f.isInf y && f.sign x != f.sign y then f.nanMag else f.infPat (f.sign x))
  else if f.isInf y then f.infPat (f.sign y)
  else f.roundRat rnd false (f.sign x && f.sign y) (f.toRat x + f.toRat y)

/-- `x - y`, that is `x + (-y)`. -/
def sub (rnd : RoundMode) (x y : Nat) : Nat := f.add rnd x (f.neg y)

/-- `x · y`. -/
def mul (rnd : RoundMode) (x y : Nat) : Nat :=
  if f.isNaN x || f.isNaN y then f.nanMag
  else if f.isInf x || f.isInf y then
    (if f.isZero x || f.isZero y then f.nanMag else f.infPat (f.sign x != f.sign y))
  else f.roundRat rnd false (f.sign x != f.sign y) (f.toRat x * f.toRat y)

/-- `x · y + z` with one rounding. -/
def fma (rnd : RoundMode) (x y z : Nat) : Nat :=
  if f.isNaN x || f.isNaN y || f.isNaN z then f.nanMag
  else if f.isInf x || f.isInf y then
    (if f.isZero x || f.isZero y then f.nanMag
     else if f.isInf z && f.sign z != (f.sign x != f.sign y) then f.nanMag
     else f.infPat (f.sign x != f.sign y))
  else if f.isInf z then f.infPat (f.sign z)
  else f.roundRat rnd false ((f.sign x != f.sign y) && f.sign z) (f.toRat x * f.toRat y + f.toRat z)

/-- `x / y`. -/
def div (rnd : RoundMode) (x y : Nat) : Nat :=
  if f.isNaN x || f.isNaN y then f.nanMag
  else if f.isInf x then (if f.isInf y then f.nanMag else f.infPat (f.sign x != f.sign y))
  else if f.isInf y then f.zeroPat (f.sign x != f.sign y)
  else if f.isZero y then (if f.isZero x then f.nanMag else f.infPat (f.sign x != f.sign y))
  else f.roundRat rnd false (f.sign x != f.sign y) (f.toRat x / f.toRat y)

/-- `√x`. -/
def sqrt (rnd : RoundMode) (x : Nat) : Nat :=
  if f.isNaN x then f.nanMag
  else if f.isZero x then f.zeroPat (f.sign x)
  else if f.sign x then f.nanMag
  else if f.isInf x then f.infPat false
  else f.sqrtMag rnd (f.toRat x)

/-- `maxNum(x, y)`: a NaN operand gives the other operand; `+0` above `-0`. -/
def maxNum (x y : Nat) : Nat :=
  if f.isNaN x then (if f.isNaN y then f.nanMag else y)
  else if f.isNaN y then x
  else if f.toRat x < f.toRat y then y
  else if f.toRat y < f.toRat x then x
  else if f.sign x then y else x

/-- `minNum(x, y)`: a NaN operand gives the other operand; `-0` below `+0`. -/
def minNum (x y : Nat) : Nat :=
  if f.isNaN x then (if f.isNaN y then f.nanMag else y)
  else if f.isNaN y then x
  else if f.toRat x < f.toRat y then x
  else if f.toRat y < f.toRat x then y
  else if f.sign x then x else y

/-- `maximum(x, y)`: a NaN operand gives NaN. -/
def maxNaN (x y : Nat) : Nat := if f.isNaN x || f.isNaN y then f.nanMag else f.maxNum x y

/-- `minimum(x, y)`: a NaN operand gives NaN. -/
def minNaN (x y : Nat) : Nat := if f.isNaN x || f.isNaN y then f.nanMag else f.minNum x y

/-- The ordered comparison: `none` when an operand is NaN. -/
def cmp (x y : Nat) : Option Ordering :=
  if f.isNaN x || f.isNaN y then none
  else some (if f.toRat x < f.toRat y then .lt else if f.toRat x = f.toRat y then .eq else .gt)

/-- Conversion from format `f` into `g` with one rounding. -/
def cvt (g : Format) (rnd : RoundMode) (sat : Bool) (x : Nat) : Nat :=
  if f.isNaN x then g.nanMag
  else if f.isInf x then (if sat then g.setSign (f.sign x) g.maxMag else g.infPat (f.sign x))
  else g.roundRat rnd sat (f.sign x) (f.toRat x)

/-- `g` holds every finite value of `f`: at least as many fraction bits, a quantum at least as
fine at the bottom, and a range at least as large. -/
def Wider (f g : Format) : Prop := f.m ≤ g.m ∧ g.emin - g.m ≤ f.emin - f.m ∧ f.maxVal ≤ g.maxVal

instance (f g : Format) : Decidable (Wider f g) := by unfold Wider; exact inferInstance

end Format

end TVA.Math.Float
