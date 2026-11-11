import Claim
import TVA.Proof
import Proof.Basic
import Proof.Foot

/-!
# Lanes of the `[128, 256]` tile

The `[128, 256]` tile is indexed by `Idx [128, 256]`; the dot's elements are indexed by two
`Fin`s. `get2_lane` bridges the two, and `laneIdx_inj` says the distinct enabled lanes of a
program's `c` tile are distinct elements of `c`.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- The row of a lane, as a `Fin 128`. -/
def laneR (l : Idx [128, 256]) : Fin 128 :=
  ⟨l.coord 0, by have h := l.coord_lt (k := 0) (by simp); simpa [Shape.dim] using h⟩

/-- The column of a lane, as a `Fin 256`. -/
def laneC (l : Idx [128, 256]) : Fin 256 :=
  ⟨l.coord 1, by have h := l.coord_lt (k := 1) (by simp); simpa [Shape.dim] using h⟩

@[simp] theorem laneR_coord (l : Idx [128, 256]) : (laneR l).val = l.coord 0 := rfl

@[simp] theorem laneC_coord (l : Idx [128, 256]) : (laneC l).val = l.coord 1 := rfl

/-- **`get2` at a lane's coordinates**: the dot's element `(row, column)` of a lane is the
tensor's element at the lane. -/
theorem get2_lane {α : Type} (x : Tensor [128, 256] α) (l : Idx [128, 256]) :
    x.get2 (laneR l) (laneC l) = x.get l := by
  rcases l with ⟨a, b, u⟩
  rcases u with ⟨⟩
  rfl

/-- Two lanes with the same coordinates are equal. -/
theorem lane_ext {l l' : Idx [128, 256]} (h0 : l.coord 0 = l'.coord 0)
    (h1 : l.coord 1 = l'.coord 1) : l = l' := by
  refine Idx.ext_coord fun k => ?_
  match k with
  | 0 => exact h0
  | 1 => exact h1
  | _ + 2 => rfl

/-- The row coordinate from the row. -/
theorem laneRow_coord {d : Inst} {i : Fin (launch d).size} {l l' : Idx [128, 256]}
    (h : laneRow d i l = laneRow d i l') : l.coord 0 = l'.coord 0 := by
  have h' := h
  unfold laneRow at h'
  omega

/-- The column coordinate from the column. -/
theorem laneCol_coord {d : Inst} {i : Fin (launch d).size} {l l' : Idx [128, 256]}
    (h : laneCol d i l = laneCol d i l') : l.coord 1 = l'.coord 1 := by
  have h' := h
  unfold laneCol at h'
  omega

/-- **Distinct enabled lanes are distinct elements of `c`.** -/
theorem laneIdx_inj {d : Inst} (hd : d.ok) {i : Fin (launch d).size} {l l' : Idx [128, 256]}
    (hl : laneMask d i l = true) (hl' : laneMask d i l' = true)
    (h : laneIdx d i l = laneIdx d i l') : l = l' := by
  obtain ⟨hrow, hcol⟩ := laneMask_bounds hl
  obtain ⟨hrow', hcol'⟩ := laneMask_bounds hl'
  have hN : 0 < d.N := N_pos hd
  have hcol_eq : laneCol d i l = laneCol d i l' := by
    have e1 : laneIdx d i l % d.N = laneCol d i l := by
      have h1 : laneRow d i l * d.N + laneCol d i l
          = laneCol d i l + laneRow d i l * d.N := Nat.add_comm _ _
      rw [laneIdx, h1, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hcol]
    have e2 : laneIdx d i l' % d.N = laneCol d i l' := by
      have h1 : laneRow d i l' * d.N + laneCol d i l'
          = laneCol d i l' + laneRow d i l' * d.N := Nat.add_comm _ _
      rw [laneIdx, h1, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hcol']
    rw [← e1, ← e2, h]
  have hrow_eq : laneRow d i l = laneRow d i l' := by
    have e : laneRow d i l * d.N + laneCol d i l' = laneRow d i l' * d.N + laneCol d i l' := by
      have h' : laneRow d i l * d.N + laneCol d i l
          = laneRow d i l' * d.N + laneCol d i l' := by
        simpa [laneIdx] using h
      rw [hcol_eq] at h'
      exact h'
    exact Nat.mul_right_cancel hN (Nat.add_right_cancel e)
  exact lane_ext (laneRow_coord hrow_eq) (laneCol_coord hcol_eq)

end Claims.BlockScaledMxfp8
