import Proof.Program

/-!
# `attn_bwd_f16`: the footprints of the programs

Distinct programs own disjoint tiles of the outputs, the outputs share no byte with the inputs,
and every element of the outputs is in the tile of some program.
-/

namespace Claims.AttnBwd

open TVA

/-- `x·H + c = y·H + c'` with `c, c' < H` gives `x = y` and `c = c'`. -/
theorem mul_add_inj {x y c c' H : Nat} (hc : c < H) (hc' : c' < H) (h : x * H + c = y * H + c') :
    x = y ∧ c = c' := by
  rcases lt_trichotomy x y with hxy | hxy | hxy
  · exfalso
    have : (x + 1) * H ≤ y * H := Nat.mul_le_mul_right _ hxy
    nlinarith
  · subst hxy; exact ⟨rfl, by omega⟩
  · exfalso
    have : (y + 1) * H ≤ x * H := Nat.mul_le_mul_right _ hxy
    nlinarith

/-- A program is determined by its coordinates. -/
theorem prog_ext (d : Inst) {i j : Nat}
    (hh : head d i = head d j) (hp : pidX d i = pidX d j) : i = j := by
  have h1 := Nat.div_add_mod i (gx d)
  have h2 := Nat.div_add_mod j (gx d)
  unfold head pidX at *
  rw [hh, hp] at h1
  omega

/-- The elements of two distinct programs' tiles are different. -/
theorem tileSet_disjoint {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gx d * 1 * G d)
    (hj : j < gx d * 1 * G d) (hij : i ≠ j) : Disjoint (tileSet d i) (tileSet d j) := by
  rw [Set.disjoint_left]
  rintro e ⟨r, c, hr, hc, rfl⟩ ⟨r', c', hr', hc', h⟩
  have hHD : 0 < d.HEAD_DIM := by have := hd.hd; omega
  have hpi := pidX_lt hi
  have hpj := pidX_lt hj
  have hg := mul_gx hd
  unfold eIdx at h
  obtain ⟨h1, h2⟩ := mul_add_inj hc' hc h.symm
  obtain ⟨h3, h4⟩ := mul_add_inj (x := head d i) (y := head d j) (c := 128 * pidX d i + r)
    (c' := 128 * pidX d j + r') (H := d.N_CTX) (by nlinarith) (by nlinarith)
    h1.symm
  have hp : pidX d i = pidX d j := by omega
  exact hij (prog_ext d h3 hp)

end Claims.AttnBwd
