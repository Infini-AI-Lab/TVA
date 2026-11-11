import Claim
import TVA.Proof

/-!
# The error analysis of one row: definitions

The online softmax of one query row, over the reals. The kernel keeps, per row, a running maximum
`m` (the reference), a running sum `l` (`deno`) and, per output component `e`, an accumulator
`a`. The analysis follows *ghost weights*: after `t` trips the row has seen the positions `S`; each
seen position `j` has a computed score `ŝ_j` (a ghost real), a ghost weight `u_j` and a ghost
rounding `δ_j` of `p` to `bfloat16`, and

* `l ≈ Σ_S u_j` (`RowInv`), with relative error `expm1 (lam0 + lam1·t)` (the `float32` roundings of
  the sum, the `fma` and the rescaling) and a tiny absolute part;
* `a_e ≈ Σ_S u_j (1 + δ_j) v_j` (`ColInv`), with relative error `expm1 γ`, `γ` growing by a fixed
  amount per trip, and a tiny absolute part;
* `u_j = exp (ŝ_j − m) · F_j` with `F_j` within `expm1 (t·e1 + e2·(m − ŝ_j))` of `1`: the factors
  of the computed `tl.exp`s (each trip rescales every old weight by the computed `re_scale`, a
  factor within `expm1 (e1 + e2·(m' − m))` of `exp (m − m')`); the second term telescopes along
  the running maximum.

A position seen in trip `t_j` enters `S` with `u_j = p̂_j`, the computed `p`, and later trips
multiply it by their `re_scale`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-! ## Constants -/

/-- The unit roundoff of `float32`. -/
def uf : ℝ := 1 / 2 ^ 24
/-- The unit roundoff of `bfloat16`. -/
def ub : ℝ := 1 / 2 ^ 8
/-- A `tl.exp` factor's log-range: `e1` per call. -/
def e1 : ℝ := 257 / 2 ^ 30
/-- A `tl.exp` factor's log-range per unit of exponent magnitude. -/
def e2 : ℝ := 771 / 2 ^ 32
/-- The log of the sum tree's relative error (depth `6`, `(1+u)^6 ≤ exp (6u)`). -/
def lam0 : ℝ := 6 / 2 ^ 24
/-- The log of one `fma` step's relative error (`1 + 3u ≤ exp (3u)`). -/
def lam1 : ℝ := 3 / 2 ^ 24
/-- The tiny absolute error per trip, in `deno`. -/
def a0 : ℝ := 1 / 2 ^ 100
/-- The tiny absolute error per trip, in the accumulator (times `1 + Σ|v|`). -/
def t0 : ℝ := 1 / 2 ^ 100
/-- The accumulator's per-trip exponent on a cached-part trip (`acc * re_scale + dot`). -/
def g1p : ℝ := 9 / 2 ^ 25
/-- The accumulator's per-trip exponent on an extend trip (`tl.dot` with the accumulator). -/
def g1e : ℝ := 68 / 2 ^ 22
/-- The accumulator's once-only exponent (the `tl.dot` error of one trip's weights). -/
def g2 : ℝ := 68 / 2 ^ 22
/-- The largest number of trips of a row. -/
def tmax : ℕ := 2 ^ 13
/-- The largest number of positions of a row. -/
def nmax : ℕ := 2 ^ 18

/-- `exp x − 1`. -/
def expm1 (x : ℝ) : ℝ := Real.exp x - 1

/-! ## The ghost and the invariants -/

/-- The ghost of a row: the computed score `s`, the weight `u` and the `bfloat16` rounding `δ` of
`p` of each seen position. -/
structure Ghost where
  s : ℕ → ℝ
  u : ℕ → ℝ
  δ : ℕ → ℝ

/-- **The row invariant** after `t` trips that have seen the positions `S`, with reference `m`
and sum `l`: the computed scores of the seen positions are within `dl` of the exact scores `sc`
and of magnitude at most `Xp`; `m` is the largest computed score; each weight is `exp (ŝ − m)`
times a factor close to `1`; and `l` is the sum of the weights up to the sum's error. -/
def RowInv (sc : ℕ → ℝ) (dl Xp : ℝ) (t : ℕ) (S : Finset ℕ) (m l : ℝ) (G : Ghost) : Prop :=
  (∀ j ∈ S, |G.s j - sc j| ≤ dl ∧ |G.s j| ≤ Xp ∧ |G.δ j| ≤ ub) ∧
  (S.Nonempty → (∀ j ∈ S, G.s j ≤ m) ∧ (∃ j ∈ S, G.s j = m) ∧
    ∀ j ∈ S, ∃ F : ℝ, |F - 1| ≤ expm1 (t * e1 + e2 * (m - G.s j)) ∧
      G.u j = Real.exp (G.s j - m) * F) ∧
  (S = ∅ → |l| ≤ t * a0) ∧
  |l - ∑ j ∈ S, G.u j| ≤ expm1 (lam0 + lam1 * t) * ∑ j ∈ S, G.u j + t * a0

/-- **The column invariant**: the accumulator `a` of one output component (values `v`) is
`Σ_S u_j (1 + δ_j) v_j` up to a relative error `expm1 γ` of `Σ_S u_j |v_j|` and a tiny absolute
error. -/
def ColInv (v : ℕ → ℝ) (γ : ℝ) (t : ℕ) (S : Finset ℕ) (G : Ghost) (a : ℝ) : Prop :=
  |a - ∑ j ∈ S, G.u j * (1 + G.δ j) * v j| ≤
    expm1 γ * ∑ j ∈ S, G.u j * |v j| + t * t0 * (1 + ∑ j ∈ S, |v j|)

/-- The ghost after a trip: the new positions `V` enter with computed score `s'`, rounding `δ'`
and weight `exp (s' − m') · f`; the old positions' weights are rescaled by `exp (m − m') · g`. -/
def Ghost.step (G : Ghost) (V : Finset ℕ) (s' δ' f : ℕ → ℝ) (m m' g : ℝ) : Ghost where
  s j := if j ∈ V then s' j else G.s j
  u j := if j ∈ V then Real.exp (s' j - m') * f j else Real.exp (m - m') * g * G.u j
  δ j := if j ∈ V then δ' j else G.δ j

/-- The tiny absolute error of the `deno` chain's `float32` operations in one trip (`tl.exp`'s
`2^-140`, the sum's and the `fma`'s `2^-150`-sized parts; with room). -/
def eps0 : ℝ := 1 / 2 ^ 135
/-- The tiny absolute error of the accumulator's operations in one trip, times `1 + Σ|v|` (the
`tl.dot`'s `66 · 2^-126` and the lifted magnitudes of subnormal operands). -/
def eps1 : ℝ := 1 / 2 ^ 118
/-- The absolute error of rounding `p` to `bfloat16` (half the smallest subnormal). -/
def etab : ℝ := 1 / 2 ^ 134

/-! ## One trip, from the float facts

`TripStat` is what the float glue proves about one trip of a row: the new positions `V` with their
computed scores, the new reference `m'`, the computed `re_scale` `r` and `p`, the sum of the
lanes and the new `deno`. -/

/-- The facts about one trip's reference, `re_scale`, `p`, sum and `deno` (see the file docstring):
`s'` the computed scores of the new lanes `V`, `m'` the new reference, `r` the computed
`re_scale = tl.exp (m − m')`, `p` the computed `tl.exp (ŝ − m')`, `f` and `g` their factors,
`Ssum` the computed `tl.sum p` and `l'` the new `deno`. -/
structure TripStat (sc : ℕ → ℝ) (dl Xp : ℝ) (S V : Finset ℕ) (m l : ℝ)
    (s' δ' f p : ℕ → ℝ) (m' g r Ssum l' : ℝ) : Prop where
  /-- the new lanes' computed scores and `bfloat16` roundings -/
  lanes : ∀ j ∈ V, |s' j - sc j| ≤ dl ∧ |s' j| ≤ Xp ∧ |δ' j| ≤ ub
  /-- the new reference is at least every new score and the old reference -/
  ref_ge : (∀ j ∈ V, s' j ≤ m') ∧ (S.Nonempty → m ≤ m')
  /-- the new reference is the old one or a new score -/
  ref_att : (S ∪ V).Nonempty → (S.Nonempty ∧ m' = m) ∨ ∃ j ∈ V, s' j = m'
  /-- `re_scale` is `exp (m − m')` times a factor `g`, up to the absolute part of `tl.exp` -/
  scale : S.Nonempty → |g - 1| ≤ expm1 (e1 + e2 * (m' - m)) ∧ |r - Real.exp (m - m') * g| ≤ eps0
  /-- the rescaling is bounded even when nothing was seen (`tl.exp (−∞) = 0` or `tl.exp (0)`) -/
  scale_bd : |r| ≤ 1 + 1 / 2 ^ 20
  /-- each new weight is `exp (ŝ − m')` times a factor, up to the absolute part of `tl.exp` -/
  weight : ∀ j ∈ V, |f j - 1| ≤ expm1 (e1 + e2 * (m' - s' j)) ∧
    |p j - Real.exp (s' j - m') * f j| ≤ eps0
  /-- the sum of the lanes -/
  sum : |Ssum - ∑ j ∈ V, p j| ≤ expm1 lam0 * ∑ j ∈ V, |p j| + eps0
  /-- the new `deno` -/
  deno : |l' - (r * l + Ssum)| ≤ 3 * uf * (|r * l| + |Ssum|) + eps0



/-- `wref K sc v`: the softmax-weighted mean `Σ_K exp (sc j) · v j / Σ_K exp (sc j)`
(`attnRef`). -/
def wref (K : Finset ℕ) (sc v : ℕ → ℝ) : ℝ :=
  (∑ j ∈ K, Real.exp (sc j) * v j) / ∑ j ∈ K, Real.exp (sc j)

/-- `wabs K sc v`: the same weighted mean of `|v|` (`attnAbs`). -/
def wabs (K : Finset ℕ) (sc v : ℕ → ℝ) : ℝ :=
  (∑ j ∈ K, Real.exp (sc j) * |v j|) / ∑ j ∈ K, Real.exp (sc j)

/-- The relative error of the computed ratio `a / l` against `wref`, in units of `wabs`:
`dl` the scores' error, `Xp` their bound, `t` the trips, `γ` the accumulator's exponent.
`Φ` bounds the weights' factors, `ρ` the ratio of the largest to the smallest weight factor
(relative to the exact weights), `Λ` the error of `deno`. -/
def coreErr (dl Xp : ℝ) (t : ℕ) (γ : ℝ) : ℝ :=
  Real.exp (2 * dl) * ((1 + expm1 (t * e1 + 2 * e2 * Xp)) / (1 - expm1 (t * e1 + 2 * e2 * Xp))) *
      (ub + expm1 γ * (1 + ub) + (expm1 (lam0 + lam1 * t) + 1 / 2 ^ 85)) /
      (1 - (expm1 (lam0 + lam1 * t) + 1 / 2 ^ 85)) +
    (Real.exp (2 * dl) * ((1 + expm1 (t * e1 + 2 * e2 * Xp)) / (1 - expm1 (t * e1 + 2 * e2 * Xp))) - 1)

/-- The ghost before any trip. -/
def Ghost.zero : Ghost := ⟨fun _ => 0, fun _ => 0, fun _ => 0⟩

/-- Before any trip nothing is seen: the invariants hold with `l = 0`, `a = 0`, whatever `m`. -/
theorem rowInv_zero (sc : ℕ → ℝ) (dl Xp m : ℝ) : RowInv sc dl Xp 0 ∅ m 0 Ghost.zero := by
  refine ⟨by simp, by simp, by simp, ?_⟩
  simp [expm1]

/-- The column invariant holds at step 0 with the empty lane set and zero accumulator. -/
theorem colInv_zero (v : ℕ → ℝ) (γ : ℝ) (G : Ghost) : ColInv v γ 0 ∅ G 0 := by
  simp [ColInv, expm1]

end

end Claims.ExtendAttn
