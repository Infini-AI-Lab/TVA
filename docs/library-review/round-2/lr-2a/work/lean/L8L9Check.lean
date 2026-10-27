import TVA

/-! lr-2a, round 2: checks of the L8 and L9 changes (`lake env lean`, this file only). -/

open TVA TVA.Block

-- L8: the new bound is exactly "the largest offset fits `int32`".
example : Fits (2 ^ 21 - 1) (2 ^ 10) := by unfold Fits; norm_num
example : ¬ Fits (2 ^ 21) (2 ^ 10) := by unfold Fits; norm_num
-- the largest lane of the last fitting program is `2^31 - 1`, and `lane_toInt` gives it exactly
example : ((offs (2 ^ 21 - 1) (2 ^ 10)).get (⟨2 ^ 10 - 1, by norm_num⟩, ())).toInt = 2 ^ 31 - 1 := by
  rw [lane_toInt (by unfold Fits; norm_num)]; norm_num
-- a grid of `cdiv(n, B)` with `n = 2^31 - 1` and `B = 2^10` fits (the case `fits_of_grid` misses)
example {p : Nat} (hp : p < (2 ^ 31 - 1 + 2 ^ 10 - 1) / 2 ^ 10) : Fits p (2 ^ 10) :=
  fits_of_grid_pow2 rfl (by norm_num) (by norm_num) hp
example : ¬ (2 ^ 31 - 1 + 2 ^ 10 ≤ 2 ^ 31) := by norm_num

-- L9: the silent-skip grids are no longer valid, the largest 1-D grid that runs still is, and
-- the 2-D grid of the evidence's C2 is valid.
example (b : List Shape) : ¬ ({ grid := ![2 ^ 30, 2, 1], blocks := b } : Launch).Valid := by
  intro h; have h4 := h.2.2.2.1; change 2 ^ 30 * 2 * 1 < 2 ^ 31 at h4; norm_num at h4
example (b : List Shape) : ¬ ({ grid := ![2 ^ 16, 2 ^ 15, 2], blocks := b } : Launch).Valid := by
  intro h; have h4 := h.2.2.2.1; change 2 ^ 16 * 2 ^ 15 * 2 < 2 ^ 31 at h4; norm_num at h4
example (b : List Shape) : ¬ ({ grid := ![2 ^ 16 + 1, 2 ^ 15, 1], blocks := b } : Launch).Valid := by
  intro h; have h4 := h.2.2.2.1; change (2 ^ 16 + 1) * 2 ^ 15 * 1 < 2 ^ 31 at h4; norm_num at h4
example : (Launch.grid1 (2 ^ 31 - 1) [[1024]]).Valid :=
  Launch.valid_grid1 (by norm_num) (by
    intro s hs; simp only [List.mem_singleton] at hs; subst hs
    exact ⟨fun k hk => ⟨10, by simp at hk; omega⟩, by norm_num⟩)
example : ({ grid := ![65535, 2 ^ 15, 1], blocks := [] } : Launch).Valid := by
  refine ⟨by simp, by simp, by simp, ?_, rfl, by simp⟩
  change 65535 * 2 ^ 15 * 1 < 2 ^ 31; norm_num
-- the evidence's C8 grid runs every program, but has more than `2^31` programs: not covered
example : ¬ ({ grid := ![6700417, 641, 1], blocks := [] } : Launch).Valid := by
  intro h; have h4 := h.2.2.2.1; change 6700417 * 641 * 1 < 2 ^ 31 at h4; norm_num at h4

#print axioms TVA.Block.fits_of_grid_dvd
#print axioms TVA.Block.fits_of_grid_pow2
#print axioms TVA.Block.lane_toInt
#print axioms TVA.Launch.valid_grid1
