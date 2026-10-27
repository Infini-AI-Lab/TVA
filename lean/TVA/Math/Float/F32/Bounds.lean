import TVA.Math.Float.F32
import TVA.Math.Float.Convert
import Mathlib

/-!
# Definitions of the `f32` rounding-error analysis

The constants (`u`, `minNormal`, `maxNormal`, `subErr`), the round-to-nearest-even of a fraction
(`rne`), the scaled encoding (`expOf`, `scaled`, `encode`), the chains the tensor-core and
atomic twins compute (`fmaChain`, `addChain`, `gemmTwin`) with their exact and absolute sums and
side conditions, `γ`, `δ`, and `roundBf16`. The theorems about them are in the modules indexed
by `TVA.Math.Float.F32.Error` (`Round`, `OpErr`, `ChainErr`, `GemmErr`, `Bf16Round`, `AbsErr`,
`Exact`). This module held the whole analysis until library step A split it; the definitions
stayed here, so their names and modules are unchanged.
-/
set_option autoImplicit false

namespace TVA.Math.Float.F32

open TVA.Math.Float

open Rat


/-- Unit roundoff of binary32. -/
def u : ℚ := 1 / 2 ^ 24
/-- Smallest positive normal, `2^-126`. -/
def minNormal : ℚ := 1 / 2 ^ 126
/-- Largest finite, `(2 − 2^-23)·2^127 = (2^24 − 1)·2^104`. -/
def maxNormal : ℚ := (2 ^ 24 - 1) * 2 ^ 104
/-- Half an ulp of the subnormal grid, `2^-150`. -/
def subErr : ℚ := 1 / 2 ^ 150

/-- The integer rounding inside `roundMag`: `num / den` to the nearest integer, ties to
even. -/
def rne (num den : ℕ) : ℕ :=
  let fl := num / den
  let r := num % den
  if 2 * r > den then fl + 1 else if 2 * r < den then fl else if fl % 2 == 0 then fl else fl + 1

/-- The (clamped) exponent of a positive rational: `max ⌊log₂ q⌋ (-126)`. -/
def expOf (q : ℚ) : ℤ := max (log2Floor q.num.natAbs q.den) (-126)

/-- `q · 2^(23 − expOf q)` as a fraction `num / den` of naturals. -/
def scaled (q : ℚ) : ℕ × ℕ :=
  let n := q.num.natAbs
  let d := q.den
  let k : ℤ := expOf q - 23
  if k ≥ 0 then (n, d * 2 ^ k.toNat) else (n * 2 ^ (-k).toNat, d)

/-- Encode an integer significand `m` (`≤ 2^24`) at quantum exponent `e' − 23`. -/
def encode (m : ℕ) (e' : ℤ) : Bits :=
  let (m, e') := if m == 2 ^ 24 then (2 ^ 23, e' + 1) else (m, e')
  if e' > 127 then posInf
  else if m < 2 ^ 23 then BitVec.ofNat 32 m
  else BitVec.ofNat 32 (((e' + 127).toNat <<< 23) ||| (m - 2 ^ 23))

/-- Left-fold FMA chain: `fmaChain c [(a₁,b₁),…,(aₙ,bₙ)] = fma aₙ bₙ (… (fma a₁ b₁ c))` — the
accumulation order of a scalar FMA loop and of the tensor-core accumulation twin (`mma`). -/
def fmaChain (c : Bits) : List (Bits × Bits) → Bits
  | [] => c
  | (a, b) :: l => fmaChain (fma a b c) l

/-- The exact value of the chain. -/
def exactSum (c : Bits) (l : List (Bits × Bits)) : ℚ :=
  toRat c + (l.map fun p => toRat p.1 * toRat p.2).sum

/-- The condition number's numerator: `|c| + Σ |aᵢ|·|bᵢ|`. -/
def absSum (c : Bits) (l : List (Bits × Bits)) : ℚ :=
  |toRat c| + (l.map fun p => |toRat p.1| * |toRat p.2|).sum

/-- Every step of the chain is a well-conditioned FMA: finite operands and an exact result that is
zero or in the normal range (below overflow). -/
def ChainOK (c : Bits) : List (Bits × Bits) → Prop
  | [] => True
  | (a, b) :: l =>
      isFinite a = true ∧ isFinite b = true ∧
      (toRat a * toRat b + toRat c = 0 ∨ minNormal ≤ |toRat a * toRat b + toRat c|) ∧
      |toRat a * toRat b + toRat c| ≤ maxNormal ∧ ChainOK (fma a b c) l

/-- `γₙ = (1+u)ⁿ − 1`, the classical accumulated-rounding factor. -/
def γ (n : ℕ) : ℚ := (1 + u) ^ n - 1

/-- `addChain acc [p₁,…,pₘ] = add (… (add acc p₁) …) pₘ`: the partial sums landing on an
accumulator one at a time, each with one rounding. -/
def addChain (acc : Bits) : List Bits → Bits
  | [] => acc
  | p :: l => addChain (add acc p) l

def AddOK (acc : Bits) : List Bits → Prop
  | [] => True
  | p :: l =>
      isFinite p = true ∧ (toRat acc + toRat p = 0 ∨ minNormal ≤ |toRat acc + toRat p|) ∧
      |toRat acc + toRat p| ≤ maxNormal ∧ AddOK (add acc p) l

/-- The bf16 GEMM accumulation twin for one output element: the products of each
split (a CTA's k-range) accumulate in an FMA chain from zero, and the split results are
added into the zero-initialized output one by one (the atomic fix-up, in whatever order
the schedule produces). -/
def gemmTwin (chains : List (List (Bits × Bits))) : Bits :=
  addChain posZero (chains.map (fmaChain posZero))

def GemmOK (chains : List (List (Bits × Bits))) : Prop :=
  (∀ ch ∈ chains, ChainOK posZero ch) ∧ AddOK posZero (chains.map (fmaChain posZero))

def gemmExact (chains : List (List (Bits × Bits))) : ℚ :=
  (chains.map fun ch => (ch.map fun p => toRat p.1 * toRat p.2).sum).sum

def gemmAbs (chains : List (List (Bits × Bits))) : ℚ :=
  (chains.map fun ch => (ch.map fun p => |toRat p.1| * |toRat p.2|).sum).sum

/-- Total number of products. -/
def gemmK (chains : List (List (Bits × Bits))) : ℕ := (chains.map List.length).sum

/-- The bf16-rounded value of an f32 pattern, as an f32 pattern. -/
def roundBf16 (x : Bits) : Bits := BitVec.ofNat 32 (bf16ToF32Bits (f32ToBf16RNE x.toNat))

/-- `δₙ = subErr · Σ_{i<n} (1+u)^i`, recursively. -/
def δ : ℕ → ℚ
  | 0 => 0
  | n + 1 => (1 + u) * δ n + subErr

/-- Operands finite and every exact partial result below overflow. -/
def ChainFin (c : Bits) : List (Bits × Bits) → Prop
  | [] => True
  | (a, b) :: l =>
      isFinite a = true ∧ isFinite b = true ∧
      |toRat a * toRat b + toRat c| ≤ maxNormal ∧ ChainFin (fma a b c) l

def AddFin (acc : Bits) : List Bits → Prop
  | [] => True
  | p :: l => isFinite p = true ∧ |toRat acc + toRat p| ≤ maxNormal ∧ AddFin (add acc p) l

end TVA.Math.Float.F32
