import Softmax

/-! Reviewer r-softmax, round 2: what the new `Post` means, and its tolerances at the covered
block sizes. -/

open TVA Kernels.Softmax

set_option pp.fullNames true in
#print Kernels.Softmax.Post

set_option pp.fullNames true in
#print Kernels.Softmax.softmaxMax

set_option pp.fullNames true in
#print Kernels.Softmax.softmaxGap

#print axioms Kernels.Softmax.main
#print axioms Kernels.Softmax.consistent
#print axioms Kernels.Softmax.satisfiable

/-- `softmaxMax` is at least every `σ_k` of the row (so it is the row's largest, a finite
supremum). -/
example (xs : Nat → F32) (n : Nat) (k : Fin n) : softmaxRef xs n k ≤ softmaxMax xs n :=
  le_ciSup (f := fun k : Fin n => softmaxRef xs n k) (Set.finite_range _).bddAbove k

/-- ... and at most the largest of them. -/
example (xs : Nat → F32) (n : Nat) (hn : 0 < n) (M : ℝ) (h : ∀ k : Fin n, softmaxRef xs n k ≤ M) :
    softmaxMax xs n ≤ M := by
  have : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  exact ciSup_le h

/-- An empty row: `softmaxMax` is `0`. -/
example (xs : Nat → F32) : softmaxMax xs 0 = 0 := by
  unfold softmaxMax; exact Real.iSup_of_isEmpty _

/-- The common factor cancels between two outputs of a row: if both meet `Post`'s per-element
bound with one `sc`, their relative errors differ by at most the sum of the two tolerances. -/
example {sc c0 c1 s0 s1 T0 T1 : ℝ} (h0 : 0 < s0) (h1 : 0 < s1)
    (e0 : |c0 - sc * s0| ≤ T0 * s0) (e1 : |c1 - sc * s1| ≤ T1 * s1) :
    |c0 / s0 - c1 / s1| ≤ T0 + T1 := by
  have a0 : |c0 / s0 - sc| ≤ T0 := by
    rw [show c0 / s0 - sc = (c0 - sc * s0) / s0 by field_simp, abs_div, abs_of_pos h0,
      div_le_iff₀ h0]; exact e0
  have a1 : |c1 / s1 - sc| ≤ T1 := by
    rw [show c1 / s1 - sc = (c1 - sc * s1) / s1 by field_simp, abs_div, abs_of_pos h1,
      div_le_iff₀ h1]; exact e1
  calc |c0 / s0 - c1 / s1| = |(c0 / s0 - sc) - (c1 / s1 - sc)| := by ring_nf
    _ ≤ |c0 / s0 - sc| + |c1 / s1 - sc| := abs_sub _ _
    _ ≤ T0 + T1 := add_le_add a0 a1

/-- For the row `(0, -1)`: `T0 + T1 ≤ 2·elemTol + gapTol + 3·floorTol < 1.2·10^-6`, far below the
`float16` (`5.2·10^-4`) and `bfloat16` (`3.0·10^-3`) differences (work/rejection.py). -/
example : 2 * elemTol + gapTol + 3 * floorTol < 12 / 10 ^ 7 := by
  norm_num [elemTol, gapTol, floorTol]

/-- The common factor's bound is below `bfloat16`'s and `float16`'s unit roundoffs only up to
`B = 2^14` and `B = 2^9`; it is the sum's bound, applied to the whole row at once. -/
example : sumTol (2 ^ 14) < 2 ^ (-8 : ℤ) := by norm_num [sumTol]
example : sumTol (2 ^ 9) < 2 ^ (-11 : ℤ) := by norm_num [sumTol]

/-- At the test's `BLOCK_SIZE = 1024`, `sumTol` is above `6.6·10^-5` and the library's tree bound
`(B − 1)·2^-24` is within `87·2^-24` of it. -/
example : 66 / 10 ^ 6 < sumTol (2 ^ 10) ∧
    sumTol (2 ^ 10) - ((2 ^ 10 : ℚ) - 1) * 2 ^ (-24 : ℤ) = 87 * 2 ^ (-24 : ℤ) := by
  norm_num [sumTol]
