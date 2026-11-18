import TVA.Core

/-!
# The relations of float results

What a float maximum or minimum, and an addition that a product may feed, may return
(`TVA.Triton.Arith` explains fusion). They come before the catalog of sites (`TVA.Site`), which
uses them.
-/
namespace TVA

/-- What `tl.maximum` (`isMax`) or `tl.minimum` may return on two elements: the IEEE
`maxNum`/`minNum` up to NaN payload, or either operand when both are zeros (LLVM `maxnum` leaves
the sign of a zero result open). -/
def FloatV.MaxRel {f : Format} (isMax : Bool) (x y r : FloatV f) : Prop :=
  FloatV.NaNEq (if isMax then FloatV.max x y else FloatV.min x y) r ∨
    (x.isZero = true ∧ y.isZero = true ∧ (r = x ∨ r = y))

/-- `q` is an exact value that rounds (to nearest even) to `x`: it rounds to a finite value
equal to `x`, or to the same infinity (a product that overflowed when rounded, but enters an
`fma` exactly). -/
def RoundsTo {f : Format} (q : Rat) (x : FloatV f) : Prop :=
  (f.isFinite (f.roundRat .rn false false q) = true ∧ x.isFinite = true ∧
      f.toRat (f.roundRat .rn false false q) = x.toRat) ∨
    (f.isInf (f.roundRat .rn false false q) = true ∧ f.roundRat .rn false false q = x.pat)

/-- The results an addition `x + y` may produce when either operand may be a product.

Without fusion, the IEEE sum up to NaN payload. With fusion, also what an `fma` gives: an `fma`
takes one product unrounded and its other operand as it is (sm_90 has no instruction that adds two
unrounded products). The model does not track which operand is the product, so this allows:
- the rounding of `p + q` for exact `p`, `q` that round to the operands (`RoundsTo`), one of them
  the operand's own value (`p = x.toRat` or `q = y.toRat`), with either sign for an exact zero;
- an infinite operand added to an exact other operand, which gives that infinity. -/
def AddRel {f : Format} (fusion : Bool) (x y r : FloatV f) : Prop :=
  FloatV.NaNEq (FloatV.add x y) r ∨
    (fusion = true ∧
      ((∃ p q : Rat, RoundsTo p x ∧ RoundsTo q y ∧ (p = x.toRat ∨ q = y.toRat) ∧ ∃ z : Bool,
          FloatV.NaNEq (FloatV.ofPat (f.roundRat .rn false z (p + q))) r) ∨
        (∃ p : Rat, RoundsTo p x ∧ y.isInf = true ∧ r = y) ∨
        (∃ q : Rat, RoundsTo q y ∧ x.isInf = true ∧ r = x)))

end TVA
