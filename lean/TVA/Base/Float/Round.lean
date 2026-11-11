import TVA.Base.Float.Format

/-!
# Rounding an exact value into a format

`f.roundRat rnd sat zs q` is the pattern of format `f` that the exact rational `q` rounds to
(IEEE 754-2019 §4.3). Every rounding operation of the library (`+`, `*`, `fma`, `/`, conversions,
constants) computes its exact result and calls it.

How a positive `a = |q|` is rounded:
1. The *quantum* is `2^Q` with `Q = max(⌊log₂ a⌋, emin) − m` (`qexp`): the spacing of the
   representable values around `a` (in the subnormal range, the fixed spacing `2^(emin − m)`).
2. The *significand* `N` is `a / 2^Q` rounded to an integer (`roundSig`): to the nearest one,
   ties to the even one (`.rn`), or toward zero (`.rz`). `N < 2^m` only in the subnormal range,
   and `N = 2^(m+1)` when rounding carries into the next binade.
3. The result is the value `N · 2^Q`, encoded by `pack`, unless it exceeds `maxVal` (`encode`):
   an *overflow* gives `±maxVal` when saturating (`sat`, PTX `.satfinite`) or rounding toward
   zero, and otherwise `±∞` (`.ieee`), NaN (`.nanOnly`) or `±maxVal` (`.finite`).
4. The sign of `q` is put back. An exact `q = 0` has no sign, so the caller says which zero it
   is (`zs`, by IEEE 754-2019 §6.3: `x + y` is `-0` only when both are `-0`); a nonzero `q`
   that rounds to zero keeps its own sign.

`sqrtMag` rounds `√a` the same way, from an integer square root.

The proof kit shows that this is IEEE rounding: the result is the representable value nearest
to `q` (or the next one toward zero), within `|q|·u + eta` of it when it is finite.
-/
namespace TVA

/-- The rounding modes Triton uses: `"rtne"` (the default) and `"rtz"`. -/
inductive RoundMode where
  /-- To nearest, ties to the even significand. -/
  | rn
  /-- Toward zero (truncation). -/
  | rz
  deriving DecidableEq, Repr

namespace Format

/-- How `a` compares with `b`. -/
def cmpq (a b : Rat) : Ordering := if a < b then .lt else if a = b then .eq else .gt

/-- A nonnegative number rounded to an integer, from its integer part `k` and how its
fractional part compares with `1/2` (`c`). -/
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

variable (f : Format)

/-- The exponent of the quantum for a value `a > 0`: `max(⌊log₂ a⌋, emin) − m`. -/
def qexp (a : Rat) : Int := max (qlog2 a) f.emin - f.m

/-- The magnitude of `N · 2^Q`, for a significand from `roundSig`: `N < 2^m` (subnormal, with
`Q = emin − m`), `2^m ≤ N < 2^(m+1)` (normal), or `N = 2^(m+1)` (a carry into the next binade). -/
def pack (N : Nat) (Q : Int) : Nat :=
  if N < 2 ^ f.m then N
  else if N < 2 ^ (f.m + 1) then (Q + f.m + f.bias).toNat * 2 ^ f.m + (N - 2 ^ f.m)
  else (Q + f.m + 1 + f.bias).toNat * 2 ^ f.m

/-- The magnitude of a rounded value `N · 2^Q`: `pack` when it is at most `maxVal`; otherwise an
overflow, which gives the largest finite magnitude when saturating or rounding toward zero, and
otherwise the infinity (`.ieee`), the NaN (`.nanOnly`) or the largest magnitude (`.finite`). -/
def encode (rnd : RoundMode) (sat : Bool) (N : Nat) (Q : Int) : Nat :=
  if (N : Rat) * pow2 Q ≤ f.maxVal then f.pack N Q
  else if sat || rnd == .rz then f.maxMag
  else match f.specials with
    | .ieee => f.infMag
    | .nanOnly => f.nanMag
    | .finite => f.maxMag

/-- The significand of `a > 0` rounded at the quantum `2^Q`: `a / 2^Q` rounded to an integer. -/
def roundSig (rnd : RoundMode) (a : Rat) (Q : Int) : Nat := rnat rnd (a * pow2 (-Q))

/-- The magnitude `a > 0` rounds to. -/
def roundMag (rnd : RoundMode) (sat : Bool) (a : Rat) : Nat :=
  f.encode rnd sat (roundSig rnd a (f.qexp a)) (f.qexp a)

/-- **The pattern the exact value `q` rounds to**: to nearest even (`.rn`) or toward zero
(`.rz`); an overflow saturates when `sat`; `zs` is the sign of the zero that an exact `q = 0`
gives, and a nonzero `q` keeps its own sign. -/
def roundRat (rnd : RoundMode) (sat zs : Bool) (q : Rat) : Nat :=
  if q = 0 then (if zs then f.signBit else 0)
  else (if q < 0 then f.signBit else 0) + f.roundMag rnd sat (if q < 0 then -q else q)

/-- The magnitude `√a` rounds to, for `a > 0`: the quantum exponent from `⌊log₂ √a⌋ =
⌊⌊log₂ a⌋ / 2⌋`, the integer part of `√a / 2^Q` as the integer square root of `⌊a / 4^Q⌋`, and
its fraction compared with `1/2` by squaring. -/
def sqrtMag (rnd : RoundMode) (a : Rat) : Nat :=
  let Q := max (qlog2 a / 2) f.emin - f.m
  let t := a * pow2 (-2 * Q)
  let k := Nat.sqrt (Rat.floor t).toNat
  f.encode rnd false (rint rnd k (cmpq t (((k : Rat) + 1 / 2) * ((k : Rat) + 1 / 2)))) Q

end Format

end TVA
