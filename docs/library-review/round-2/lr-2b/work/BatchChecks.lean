import TVA

/-!
lr-2b, library review round 2: Lean checks of L8 and L9 (statements and boundary cases).
Run from `lean/`: `lake env lean ../docs/library-review/round-2/lr-2b/work/BatchChecks.lean`.
-/
open TVA

-- L8: the relaxed `Fits` holds exactly at the boundary `p * B + B = 2^31` …
example : Block.Fits (2 ^ 21 - 1) (2 ^ 10) := by unfold Block.Fits; norm_num
-- … and fails one past it.
example : ¬ Block.Fits (2 ^ 21) (2 ^ 10) := by unfold Block.Fits; norm_num

-- L8: the last program of `cdiv(n, 1024)` for `n = 2^31 - 1` fits (`n + B ≤ 2^31` fails there,
-- so `fits_of_grid` could not give it).
example : Block.Fits ((2 ^ 31 - 1 + 1024 - 1) / 1024 - 1) 1024 :=
  Block.fits_of_grid_pow2 (n := 2 ^ 31 - 1) (e := 10) rfl (by norm_num) (by norm_num) (by norm_num)

-- L8: lane `B - 1` of that last program is `2^31 - 1` as an `int32`, without wrapping.
example (l : Fin (1024 - 0)) (hl : l.val = 1023) :
    ((Block.offs (2 ^ 21 - 1) 1024).get (l, ())).toInt = 2 ^ 31 - 1 := by
  rw [Block.lane_toInt (by unfold Block.Fits; norm_num), hl]; norm_num

-- L9: the grids the launcher skips are no longer valid.
example : ¬ (({ grid := ![2 ^ 30, 2, 1], blocks := [] } : Launch).Valid) := by
  simp [Launch.Valid, Launch.size]
example : ¬ (({ grid := ![2 ^ 16, 2 ^ 15, 2], blocks := [] } : Launch).Valid) := by
  simp [Launch.Valid, Launch.size]
example : ¬ (({ grid := ![2 ^ 16 + 1, 2 ^ 15, 1], blocks := [] } : Launch).Valid) := by
  simp [Launch.Valid, Launch.size]
-- L9: the largest grids that run are valid.
example : (({ grid := ![2 ^ 31 - 1, 1, 1], blocks := [] } : Launch).Valid) := by
  simp [Launch.Valid, Launch.size]
example : (({ grid := ![65535, 2 ^ 15, 1], blocks := [] } : Launch).Valid) := by
  simp [Launch.Valid, Launch.size]
-- L9: a zero grid is valid and has no program.
example : (({ grid := ![0, 70, 1], blocks := [] } : Launch).Valid) ∧
    ({ grid := ![0, 70, 1], blocks := [] } : Launch).size = 0 := by
  simp [Launch.Valid, Launch.size]

-- L9: `valid_grid1` needs only `n < 2^31` and the block shapes.
example : (Launch.grid1 (2 ^ 31 - 1) [[1024]]).Valid :=
  Launch.valid_grid1 (by norm_num) (by
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    exact ⟨fun k hk => ⟨10, by simp at hk; omega⟩, by norm_num⟩)

#print axioms Block.fits_of_grid_dvd
#print axioms Block.fits_of_grid_pow2
#print axioms Block.lane_toInt
#print axioms Launch.valid_grid1
