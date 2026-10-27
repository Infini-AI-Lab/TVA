import TVA.Sem.Launch

/-! lr-1, library review round 1, L3: the grids of `l3_rejections.py` W1 and W2 are valid launches
(`TVA.Launch.Valid`), with 2^31 and 2^32 programs. On the H200, Triton 3.8.0 runs none of their
programs and reports nothing (`l3_rejections.out`). -/

open TVA

/-- W1: grid (2^30, 2, 1). -/
def w1 : Launch := { grid := ![2 ^ 30, 2, 1], blocks := [] }
/-- W2: grid (2^16, 2^15, 2). -/
def w2 : Launch := { grid := ![2 ^ 16, 2 ^ 15, 2], blocks := [] }

example : w1.Valid := by
  refine ⟨by decide, by decide, by decide, rfl, by simp [w1]⟩
example : w1.size = 2 ^ 31 := by decide
example : w2.Valid := by
  refine ⟨by decide, by decide, by decide, rfl, by simp [w2]⟩
example : w2.size = 2 ^ 32 := by decide
