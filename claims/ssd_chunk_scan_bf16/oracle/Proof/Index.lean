import Proof.Init
import Proof.Noinit

/-!
# Program indices and tiles

The program index is determined by its tile `(k, pm, pn, h)` and every tuple in range is a program;
an element of `out` determines its token, head and channel.
-/

namespace Claims.SsdChunkScan

open TVA

/-- The program of tile `(k, pm, pn, h)`. -/
def progOf (d : Inst) (k pm pn h : Nat) : Nat := (pm * npn d + pn) + gx d * (k + d.nmeta * h)

/-- The program of an in-range tile is in the grid. -/
theorem progOf_lt {d : Inst} (hd : d.ok) {k pm pn h : Nat} (hk : k < d.nmeta) (hh : h < d.H)
    (hpm : pm < nmt d) (hpn : pn < npn d) : progOf d k pm pn h < gx d * d.nmeta * d.H := by
  have hg := gx_pos hd
  have h1 : pm * npn d + pn < gx d := by
    rw [gx_eq]
    calc pm * npn d + pn < pm * npn d + npn d := by omega
      _ = (pm + 1) * npn d := by ring
      _ ≤ nmt d * npn d := Nat.mul_le_mul_right _ hpm
  have h2 : k + d.nmeta * h < d.nmeta * d.H := by
    calc k + d.nmeta * h < d.nmeta + d.nmeta * h := by omega
      _ = d.nmeta * (h + 1) := by ring
      _ ≤ d.nmeta * d.H := Nat.mul_le_mul_left _ hh
  unfold progOf
  calc pm * npn d + pn + gx d * (k + d.nmeta * h) < gx d + gx d * (k + d.nmeta * h) := by omega
    _ = gx d * (k + d.nmeta * h + 1) := by ring
    _ ≤ gx d * (d.nmeta * d.H) := Nat.mul_le_mul_left _ (by omega)
    _ = _ := by ring

/-- The logical chunk of the program of tile `(k, pm, pn, h)`. -/
theorem kOf_progOf {d : Inst} (hd : d.ok) {k pm pn h : Nat} (hk : k < d.nmeta)
    (hpm : pm < nmt d) (hpn : pn < npn d) : kOf d (progOf d k pm pn h) = k := by
  have hg := gx_pos hd
  have h1 : pm * npn d + pn < gx d := by
    rw [gx_eq]
    calc pm * npn d + pn < pm * npn d + npn d := by omega
      _ = (pm + 1) * npn d := by ring
      _ ≤ nmt d * npn d := Nat.mul_le_mul_right _ hpm
  unfold kOf progOf
  rw [Nat.add_mul_div_left _ _ hg, Nat.div_eq_of_lt h1, zero_add,
    Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hk]

/-- The head of the program of tile `(k, pm, pn, h)`. -/
theorem hOf_progOf {d : Inst} (hd : d.ok) {k pm pn h : Nat} (hk : k < d.nmeta)
    (hpm : pm < nmt d) (hpn : pn < npn d) : hOf d (progOf d k pm pn h) = h := by
  have hg := gx_pos hd
  have hm : 0 < d.nmeta := by omega
  have h1 : pm * npn d + pn < gx d := by
    rw [gx_eq]
    calc pm * npn d + pn < pm * npn d + npn d := by omega
      _ = (pm + 1) * npn d := by ring
      _ ≤ nmt d * npn d := Nat.mul_le_mul_right _ hpm
  unfold hOf progOf
  rw [← Nat.div_div_eq_div_mul, Nat.add_mul_div_left _ _ hg, Nat.div_eq_of_lt h1, zero_add,
    Nat.add_mul_div_left _ _ hm, Nat.div_eq_of_lt hk, zero_add]

/-- The row tile of the program of tile `(k, pm, pn, h)`. -/
theorem pmOf_progOf {d : Inst} (hd : d.ok) {k pm pn h : Nat} (hpm : pm < nmt d)
    (hpn : pn < npn d) : pmOf d (progOf d k pm pn h) = pm := by
  have hg := gx_pos hd
  have hnp := npn_pos hd
  have h1 : pm * npn d + pn < gx d := by
    rw [gx_eq]
    calc pm * npn d + pn < pm * npn d + npn d := by omega
      _ = (pm + 1) * npn d := by ring
      _ ≤ nmt d * npn d := Nat.mul_le_mul_right _ hpm
  unfold pmOf progOf
  rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h1, Nat.add_comm (pm * npn d) pn,
    Nat.add_mul_div_right _ _ hnp, Nat.div_eq_of_lt hpn, zero_add]

/-- The column tile of the program of tile `(k, pm, pn, h)`. -/
theorem pnOf_progOf {d : Inst} (hd : d.ok) {k pm pn h : Nat} (hpm : pm < nmt d)
    (hpn : pn < npn d) : pnOf d (progOf d k pm pn h) = pn := by
  have hg := gx_pos hd
  have hnp := npn_pos hd
  have h1 : pm * npn d + pn < gx d := by
    rw [gx_eq]
    calc pm * npn d + pn < pm * npn d + npn d := by omega
      _ = (pm + 1) * npn d := by ring
      _ ≤ nmt d * npn d := Nat.mul_le_mul_right _ hpm
  unfold pnOf progOf
  rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h1, Nat.add_comm (pm * npn d) pn,
    Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hpn]

/-- A program is determined by its tile. -/
theorem prog_ext {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gx d * d.nmeta * d.H)
    (_hj : j < gx d * d.nmeta * d.H) (hk : kOf d i = kOf d j) (hpm : pmOf d i = pmOf d j)
    (hpn : pnOf d i = pnOf d j) (hh : hOf d i = hOf d j) : i = j := by
  have hg := gx_pos hd
  have hnp := npn_pos hd
  have hm : 0 < d.nmeta := by
    rcases Nat.eq_zero_or_pos d.nmeta with h | h
    · rw [h] at hi; simp at hi
    · exact h
  have key : ∀ i, i = (pmOf d i * npn d + pnOf d i) + gx d * (kOf d i + d.nmeta * hOf d i) := by
    intro i
    unfold pmOf pnOf kOf hOf
    have e1 : i / gx d = d.nmeta * (i / gx d / d.nmeta) + i / gx d % d.nmeta :=
      (Nat.div_add_mod _ _).symm
    have e2 : i / gx d / d.nmeta = i / (gx d * d.nmeta) := by rw [Nat.div_div_eq_div_mul]
    have e3 : i % gx d = npn d * (i % gx d / npn d) + i % gx d % npn d :=
      (Nat.div_add_mod _ _).symm
    have e4 : i = gx d * (i / gx d) + i % gx d := (Nat.div_add_mod _ _).symm
    have e5 : gx d * (i / gx d % d.nmeta + d.nmeta * (i / (gx d * d.nmeta))) = gx d * (i / gx d) := by
      rw [← e2, add_comm, ← e1]
    have e6 : i % gx d / npn d * npn d + i % gx d % npn d = i % gx d := by
      rw [Nat.mul_comm]; exact e3.symm
    omega
  rw [key i, key j, hk, hpm, hpn, hh]

end Claims.SsdChunkScan
