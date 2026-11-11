import TVA.Base.Float.Round

/-!
# The IEEE operations on patterns

Each operation takes and returns patterns of one format `f` (IEEE 754-2019 §5.4, §6, §9.6):
- `add`, `sub`, `mul`, `fma`, `div`: the special cases first, then one rounding of the exact
  result (`roundRat`). A NaN operand gives the canonical NaN (`nanMag`), and so do `∞ − ∞`,
  `0 · ∞`, `0 / 0` and `∞ / ∞`; an infinite operand gives the infinity the sign rules say. An
  exact zero result has the sign of IEEE 754-2019 §6.3: `x + y` is `-0` only when both are `-0`; a
  product or quotient has the exclusive or of the signs; `fma` treats the product as an addend
  with that sign.
- `sqrt`: `√x` rounded (`sqrtMag`); `sqrt(-0) = -0`; a negative operand gives NaN.
- `neg`, `absPat`: flip or clear the sign bit (a NaN stays a NaN, with its payload).
- `maxNum`, `minNum`: IEEE 754-2008 `maxNum`/`minNum` (LLVM `maxnum`/`minnum`): a NaN operand
  gives the other operand, `max(-0, +0) = +0`, `min(-0, +0) = -0`. `maxNaN`, `minNaN`: a NaN
  operand gives NaN (LLVM `maximum`/`minimum`).
- `cmp`: the ordered comparison, `none` when an operand is NaN; `-0` equals `+0`.
- `cvt f g rnd sat`: from format `f` into `g` with one rounding; NaN gives `g`'s canonical NaN;
  `±∞` gives `±∞` (`±maxVal` when saturating, NaN into `e4m3`, which has no infinity); a finite
  value beyond `g`'s range overflows as `roundRat` says.

In a format without infinities an infinite result (`x / 0`) is what an overflow gives there
(`infPat`). `FloatV` (`TVA.Base.Float.Value`) wraps these as the operations on values.
-/
namespace TVA.Format

variable (f : Format)

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

/-- `x + y`, rounded with `rnd`. -/
def add (rnd : RoundMode) (x y : Nat) : Nat :=
  if f.isNaN x || f.isNaN y then f.nanMag
  else if f.isInf x then
    (if f.isInf y && f.sign x != f.sign y then f.nanMag else f.infPat (f.sign x))
  else if f.isInf y then f.infPat (f.sign y)
  else f.roundRat rnd false (f.sign x && f.sign y) (f.toRat x + f.toRat y)

/-- `x - y`, that is `x + (-y)`. -/
def sub (rnd : RoundMode) (x y : Nat) : Nat := f.add rnd x (f.neg y)

/-- `x · y`, rounded with `rnd`. -/
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

/-- `x / y`, rounded with `rnd`. -/
def div (rnd : RoundMode) (x y : Nat) : Nat :=
  if f.isNaN x || f.isNaN y then f.nanMag
  else if f.isInf x then (if f.isInf y then f.nanMag else f.infPat (f.sign x != f.sign y))
  else if f.isInf y then f.zeroPat (f.sign x != f.sign y)
  else if f.isZero y then (if f.isZero x then f.nanMag else f.infPat (f.sign x != f.sign y))
  else f.roundRat rnd false (f.sign x != f.sign y) (f.toRat x / f.toRat y)

/-- `√x`, rounded with `rnd`. -/
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

/-- The ordered comparison of `x` with `y`: `none` when an operand is NaN. -/
def cmp (x y : Nat) : Option Ordering :=
  if f.isNaN x || f.isNaN y then none
  else some (if f.toRat x < f.toRat y then .lt else if f.toRat x = f.toRat y then .eq else .gt)

/-- Conversion of a pattern of `f` into format `g` with one rounding (`rnd`), saturating an
overflow when `sat`. -/
def cvt (g : Format) (rnd : RoundMode) (sat : Bool) (x : Nat) : Nat :=
  if f.isNaN x then g.nanMag
  else if f.isInf x then (if sat then g.setSign (f.sign x) g.maxMag else g.infPat (f.sign x))
  else g.roundRat rnd sat (f.sign x) (f.toRat x)

end TVA.Format
