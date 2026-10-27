import TVA.Math.Float.IEEE

/-!
# IEEE-754 binary32 arithmetic on bit patterns

Bit-exact definitions of the correctly rounded binary32 operations over `BitVec 32`.
Finite values map to exact rationals, the operation is computed exactly, and the result is
rounded to nearest-even (`roundMag`): subnormals are kept (no flush to zero), overflow rounds
to ±∞, signed zeros follow IEEE 754-2008 §6.3, and every NaN result is the canonical NaN
`0x7fffffff` (NVIDIA GPUs do not propagate payloads).

Layer L0. These are the IEEE operations themselves; which source construct or instruction
denotes which operation is decided by each level's semantics (a CUDA `a + b` on `float` under
`-fmad=false`, a PTX `add.rn.f32`, a Triton `arith.addf`), and approximate units
(`ex2.approx`, `exp2f`) are platform relations here (`TVA.Platform`, `Rules.Num`).

Validation record (ported from v1): bit-for-bit agreement with the device on H200 NVL
(driver 610.57.04, nvcc 13.2, 2026-10-20) for add/sub/mul/div/max/min/fma/rcp on random and
special operands, 0 mismatches.
-/
namespace TVA.Math.Float.F32

open TVA.Math.Float

abbrev Bits := BitVec 32

def canonicalNaN : Bits := 0x7fffffff#32
def posInf : Bits := 0x7f800000#32
def negInf : Bits := 0xff800000#32
def posZero : Bits := 0#32
def negZero : Bits := 0x80000000#32
def one : Bits := 0x3f800000#32

def sign (x : Bits) : Bool := x.toNat ≥ 2 ^ 31
def expField (x : Bits) : Nat := (x.toNat >>> 23) % 256
def mantField (x : Bits) : Nat := x.toNat % 2 ^ 23
def isNaN (x : Bits) : Bool := expField x == 255 && mantField x != 0
def isInf (x : Bits) : Bool := expField x == 255 && mantField x == 0
def isZero (x : Bits) : Bool := expField x == 0 && mantField x == 0
def isFinite (x : Bits) : Bool := expField x != 255

def withSign (s : Bool) (mag : Bits) : Bits := if s then mag ||| 0x80000000#32 else mag &&& 0x7fffffff#32
def negate (x : Bits) : Bits := x ^^^ 0x80000000#32

/-- Exact magnitude of a finite pattern as a rational (`0` for ±0). -/
def magRat (x : Bits) : Rat :=
  let e := expField x
  let m := mantField x
  if e == 0 then mkRat m (2 ^ 149)
  else
    let sig := 2 ^ 23 + m
    if e ≥ 150 then (sig * 2 ^ (e - 150) : Nat) else mkRat sig (2 ^ (150 - e))

/-- Exact value of a finite pattern. -/
def toRat (x : Bits) : Rat := if sign x then -magRat x else magRat x

/-- `⌊log₂ (n / d)⌋` for `n, d > 0`. -/
def log2Floor (n d : Nat) : Int :=
  let a := Nat.log2 n
  let b := Nat.log2 d
  let ge : Bool := if a ≥ b then n ≥ d * 2 ^ (a - b) else n * 2 ^ (b - a) ≥ d
  if ge then (a : Int) - b else (a : Int) - b - 1

/-- Round a non-negative rational to the nearest binary32 magnitude (ties to even):
`0`, a subnormal, a normal, or `+∞` on overflow. The sign is applied by the caller. -/
def roundMag (q : Rat) : Bits :=
  if q ≤ 0 then posZero
  else
    let n := q.num.natAbs
    let d := q.den
    let e := log2Floor n d
    let e' := max e (-126)
    let k : Int := e' - 23                      -- quantum exponent
    let (num, den) : Nat × Nat :=
      if k ≥ 0 then (n, d * 2 ^ k.toNat) else (n * 2 ^ (-k).toNat, d)
    let fl := num / den
    let r := num % den
    let m := if 2 * r > den then fl + 1 else if 2 * r < den then fl else if fl % 2 == 0 then fl else fl + 1
    let (m, e') := if m == 2 ^ 24 then (2 ^ 23, e' + 1) else (m, e')
    if e' > 127 then posInf
    else if m < 2 ^ 23 then BitVec.ofNat 32 m          -- zero or subnormal (e' = -126)
    else BitVec.ofNat 32 (((e' + 127).toNat <<< 23) ||| (m - 2 ^ 23))

/-- Round a signed rational; an exact zero becomes `+0` (the RN rule for cancellation). -/
def roundRat (q : Rat) : Bits :=
  if q == 0 then posZero else withSign (q < 0) (roundMag (if q < 0 then -q else q))

/-- **Card `add.f32` / `add.rn.f32`** (`sub` is `add` of the negated second operand).
Preconditions: none. Transition: IEEE 754 addition, RNE, subnormals kept; NaN input ⇒
canonical NaN; `±∞ + ∓∞` ⇒ NaN; zero results: `(-0) + (-0) = -0`, otherwise `+0`.
Trust: documented-exact. -/
def add (x y : Bits) : Bits :=
  if isNaN x || isNaN y then canonicalNaN
  else if isInf x && isInf y then (if sign x == sign y then x else canonicalNaN)
  else if isInf x then x
  else if isInf y then y
  else if isZero x && isZero y then (if sign x && sign y then negZero else posZero)
  else roundRat (toRat x + toRat y)

/-- **Card `sub.f32` / `sub.rn.f32`**: `x - y = x + (-y)`. Trust: documented-exact. -/
def sub (x y : Bits) : Bits := add x (negate y)

/-- **Card `mul.f32` / `mul.rn.f32`**. Transition: IEEE multiplication, RNE; `0 × ∞` ⇒ NaN;
sign of zero and infinite results is the XOR of the signs. Trust: documented-exact. -/
def mul (x y : Bits) : Bits :=
  if isNaN x || isNaN y then canonicalNaN
  else
    let s := sign x != sign y
    if isInf x || isInf y then (if isZero x || isZero y then canonicalNaN else withSign s posInf)
    else if isZero x || isZero y then withSign s posZero
    else withSign s (roundMag (magRat x * magRat y))

/-- **Card `fma.rn.f32`**: `round(x·y + z)` with a single rounding. NaN input, `0 × ∞`, or
`±∞ ∓ ∞` ⇒ canonical NaN; an infinite product or addend gives the infinity; an exact
zero sum is `-0` only when the product and the addend are both `-0`.
Trust: documented-exact. -/
def fma (x y z : Bits) : Bits :=
  if isNaN x || isNaN y || isNaN z then canonicalNaN
  else
    let ps := sign x != sign y
    if isInf x || isInf y then
      if isZero x || isZero y then canonicalNaN
      else if isInf z && sign z != ps then canonicalNaN
      else withSign ps posInf
    else if isInf z then z
    else if isZero x || isZero y then
      if isZero z then (if ps && sign z then negZero else posZero) else z
    else roundRat (toRat x * toRat y + toRat z)

/-- **Card `rcp.rn.f32`**: `round(1/x)`; `±0 ⇒ ±∞`, `±∞ ⇒ ±0`, NaN ⇒ canonical NaN.
Trust: documented-exact. -/
def rcp (x : Bits) : Bits :=
  if isNaN x then canonicalNaN
  else if isInf x then withSign (sign x) posZero
  else if isZero x then withSign (sign x) posInf
  else withSign (sign x) (roundMag (magRat x)⁻¹)

/-- **Card `div.rn.f32`**: `round(x / y)`; `0/0`, `∞/∞` ⇒ NaN; `x/0 ⇒ ±∞`; `x/∞ ⇒ ±0`.
Trust: documented-exact. -/
def div (x y : Bits) : Bits :=
  if isNaN x || isNaN y then canonicalNaN
  else
    let s := sign x != sign y
    if isInf x then (if isInf y then canonicalNaN else withSign s posInf)
    else if isInf y then withSign s posZero
    else if isZero y then (if isZero x then canonicalNaN else withSign s posInf)
    else if isZero x then withSign s posZero
    else withSign s (roundMag (magRat x / magRat y))

/-- Numeric comparison of two non-NaN patterns, with `-0 = +0`. -/
def lt (x y : Bits) : Bool := f32Compare x.toNat y.toNat == some .lt

/-- **Card `max.f32`** (no `.NaN`): the larger operand; exactly one NaN ⇒ the other operand;
both NaN ⇒ canonical NaN; `max(-0, +0) = +0` (verified on the device, see the test).
Trust: documented-exact. -/
def fmax (x y : Bits) : Bits :=
  if isNaN x && isNaN y then canonicalNaN
  else if isNaN x then y
  else if isNaN y then x
  else if isZero x && isZero y then (if sign x && sign y then negZero else posZero)
  else if lt x y then y else x

/-- **Card `min.f32`** (no `.NaN`): the smaller operand; NaN rules as for `fmax`;
`min(-0, +0) = -0`. Trust: documented-exact. -/
def fmin (x y : Bits) : Bits :=
  if isNaN x && isNaN y then canonicalNaN
  else if isNaN x then y
  else if isNaN y then x
  else if isZero x && isZero y then (if sign x || sign y then negZero else posZero)
  else if lt x y then x else y

/-- Equality up to NaN payloads: equal patterns, or both NaN. Source-level certificates
compare floats this way, because a compiler may legally change a NaN's payload (e.g. by
folding `x * 1.0f` to `x`). -/
def eqUpToNaN (x y : Bits) : Prop := x = y ∨ (isNaN x = true ∧ isNaN y = true)

theorem eqUpToNaN_of_eq {x y : Bits} (h : x = y) : eqUpToNaN x y := Or.inl h

/-- [meaning] When `y` is not a NaN, `x` equals `y` up to NaN exactly when the two bit patterns
are equal. -/
theorem eqUpToNaN_iff_eq {x y : Bits} (hy : isNaN y = false) : eqUpToNaN x y ↔ x = y := by
  constructor
  · rintro (h | ⟨-, h⟩)
    · exact h
    · rw [hy] at h; cases h
  · exact Or.inl

/-- [meaning] When `y` is a NaN, `x` equals `y` up to NaN exactly when `x` is a NaN (with any
payload). -/
theorem eqUpToNaN_iff_isNaN {x y : Bits} (hy : isNaN y = true) : eqUpToNaN x y ↔ isNaN x = true := by
  constructor
  · rintro (rfl | ⟨h, -⟩)
    · exact hy
    · exact h
  · exact fun hx => Or.inr ⟨hx, hy⟩

/-- [meaning] If either operand is a NaN, the f32 sum `add` is a NaN. -/
theorem isNaN_add {x y : Bits} (h : isNaN x = true ∨ isNaN y = true) : isNaN (add x y) = true := by
  have : (isNaN x || isNaN y) = true := by rcases h with h | h <;> simp [h]
  simp only [add, this, ite_true]
  decide

end TVA.Math.Float.F32
