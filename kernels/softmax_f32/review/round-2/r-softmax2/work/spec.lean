import Softmax

/-! r-softmax2, round 2: the new specification with full names, the axioms, and two readings of
`softmaxMax` and `softmaxGap` at their edges (an empty row; an input `-∞`, whose `σ` is `0`). -/

set_option pp.fullNames true

#print Kernels.Softmax.Post
#print Kernels.Softmax.softmaxMax
#print Kernels.Softmax.softmaxGap
#print Kernels.Softmax.sumTol
#print Kernels.Softmax.elemTol
#print Kernels.Softmax.gapTol
#print Kernels.Softmax.floorTol
#print Kernels.Softmax.RowDone
#print axioms Kernels.Softmax.main
#print axioms Kernels.Softmax.consistent
#print axioms Kernels.Softmax.satisfiable

open Kernels.Softmax

-- An empty row: `softmaxMax` is `0` (`⨆` over `Fin 0`).
example (xs : Nat → TVA.F32) : softmaxMax xs 0 = 0 := by
  simp [softmaxMax]

-- A column whose exact softmax is `0` (an input `-∞`) has gap `0` (`Real.log 0 = 0`, `x / 0 = 0`),
-- so `Post` holds its output to `floorTol · softmaxMax` of `0`.
example (xs : Nat → TVA.F32) (n j : Nat) (h : softmaxRef xs n j = 0) : softmaxGap xs n j = 0 := by
  simp [softmaxGap, h]

-- The tolerances at the host's block sizes (the README's figures): `sumTol` at 256, 1024, 2^14,
-- within `87·2^-24` of `(B - 1)·2^-24`; the per-element bound of the README's half-precision
-- argument, `2·elemTol + gapTol + floorTol·(1 + e)` with `e < 3`, below `1.14·10^-6`.
example : (2 / 10 ^ 5 : ℚ) < sumTol 256 ∧ sumTol 256 < 21 / 10 ^ 6 := by norm_num [sumTol]
example : (66 / 10 ^ 6 : ℚ) < sumTol 1024 ∧ sumTol 1024 < 67 / 10 ^ 6 := by norm_num [sumTol]
example : (98 / 10 ^ 5 : ℚ) < sumTol 16384 ∧ sumTol 16384 < 99 / 10 ^ 5 := by norm_num [sumTol]
example (B : Nat) : sumTol B - ((B : ℚ) - 1) * 2 ^ (-24 : ℤ) = 87 * 2 ^ (-24 : ℤ) := by
  unfold sumTol; ring
example : 2 * elemTol + gapTol + floorTol * (1 + 3) < (114 / 10 ^ 8 : ℚ) := by
  norm_num [elemTol, gapTol, floorTol]
-- The test's `torch.allclose` `rtol` (`1e-5`) is below the library's tree term `(B - 1)·2^-24`
-- at `B = 1024` alone, so no tolerance built on `sum_err_covers` reaches it.
example : (1 / 10 ^ 5 : ℚ) < (1023 : ℚ) * 2 ^ (-24 : ℤ) := by norm_num
