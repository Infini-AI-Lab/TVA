import Proof.Index

/-!
# Footprints

The tiles of distinct programs are disjoint, and every element of `out` is in the tile of the
program that owns it.
-/

namespace Claims.SsdChunkScan

open TVA

/-- An element of `out` determines its token, head and channel. -/
theorem outIdx_dec {d : Inst} {t t' h h' p p' : Nat} (hh : h < d.H) (hh' : h' < d.H)
    (hp : p < d.P) (hp' : p' < d.P) (e : outIdx d t h p = outIdx d t' h' p') :
    t = t' ∧ h = h' ∧ p = p' := by
  have hP : 0 < d.P := by omega
  have hHP : 0 < d.H * d.P := Nat.mul_pos (by omega) hP
  have hlt : h * d.P + p < d.H * d.P := by
    calc h * d.P + p < h * d.P + d.P := by omega
      _ = (h + 1) * d.P := by ring
      _ ≤ d.H * d.P := Nat.mul_le_mul_right _ hh
  have hlt' : h' * d.P + p' < d.H * d.P := by
    calc h' * d.P + p' < h' * d.P + d.P := by omega
      _ = (h' + 1) * d.P := by ring
      _ ≤ d.H * d.P := Nat.mul_le_mul_right _ hh'
  unfold outIdx at e
  have e1 : t * (d.H * d.P) + (h * d.P + p) = t' * (d.H * d.P) + (h' * d.P + p') := by omega
  have d1 : (t * (d.H * d.P) + (h * d.P + p)) / (d.H * d.P) = t := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hHP, Nat.div_eq_of_lt hlt, zero_add]
  have d2 : (t' * (d.H * d.P) + (h' * d.P + p')) / (d.H * d.P) = t' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hHP, Nat.div_eq_of_lt hlt', zero_add]
  have m1 : (t * (d.H * d.P) + (h * d.P + p)) % (d.H * d.P) = h * d.P + p := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hlt]
  have m2 : (t' * (d.H * d.P) + (h' * d.P + p')) % (d.H * d.P) = h' * d.P + p' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hlt']
  rw [e1] at d1 m1
  have htt : t = t' := d1.symm.trans d2
  have hhp : h * d.P + p = h' * d.P + p' := m1.symm.trans m2
  have q1 : (h * d.P + p) / d.P = h := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt hp, zero_add]
  have q2 : (h' * d.P + p') / d.P = h' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt hp', zero_add]
  have hh2 : h = h' := by rw [← q1, ← q2, hhp]
  refine ⟨htt, hh2, ?_⟩
  rw [hh2] at hhp; omega

/-- An element of a tile is the element of a valid row and column of the tile. -/
theorem tile_elem {d : Inst} {k pm pn h x : Nat}
    (hx : Tile2.ElemAt (lc d k * d.cs * (d.H * d.P) + h * d.P) (d.H * d.P) 1
      (fun ii : Fin BM => rowOf d k pm ii) (fun jj : Fin BN => colOf pn jj) (lend d k) d.P x) :
    ∃ ii jj : Nat, ii < BM ∧ jj < BN ∧ RowOk d k pm ii ∧ ColOk d pn jj ∧
      x = outIdx d (tokOf d k pm ii) h (colOf pn jj) := by
  obtain ⟨ii, jj, h1, h2, h3⟩ := hx
  refine ⟨ii, jj, ii.isLt, jj.isLt, h1, h2, ?_⟩
  rw [h3]; unfold outIdx tokOf; ring

/-- The tiles of two programs that share an element are the same tile. -/
theorem tile_eq {d : Inst} (hd : d.ok) {k pm pn h k' pm' pn' h' ii jj ii' jj' : Nat}
    (hk : k < d.nmeta) (hk' : k' < d.nmeta) (hh : h < d.H) (hh' : h' < d.H)
    (hii : ii < BM) (hjj : jj < BN) (hii' : ii' < BM) (hjj' : jj' < BN)
    (hri : RowOk d k pm ii) (hri' : RowOk d k' pm' ii') (hcj : ColOk d pn jj)
    (hcj' : ColOk d pn' jj')
    (e : outIdx d (tokOf d k pm ii) h (colOf pn jj) = outIdx d (tokOf d k' pm' ii') h' (colOf pn' jj')) :
    k = k' ∧ pm = pm' ∧ pn = pn' ∧ h = h' := by
  obtain ⟨ht, hhh, hp⟩ := outIdx_dec hh hh' hcj hcj' e
  have hl1 := lend_le hd hk
  have hl2 := lend_le hd hk'
  have hcs := cs_pos hd
  unfold tokOf at ht
  unfold RowOk at hri hri'
  have hr1 : rowOf d k pm ii < d.cs := by omega
  have hr2 : rowOf d k' pm' ii' < d.cs := by omega
  have hc : lc d k = lc d k' := by
    have h1 : (lc d k * d.cs + rowOf d k pm ii) / d.cs = lc d k := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hcs, Nat.div_eq_of_lt hr1, zero_add]
    have h2 : (lc d k' * d.cs + rowOf d k' pm' ii') / d.cs = lc d k' := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hcs, Nat.div_eq_of_lt hr2, zero_add]
    rw [← h1, ← h2, ht]
  have hr : rowOf d k pm ii = rowOf d k' pm' ii' := by rw [hc] at ht; omega
  have hkk : k = k' := by
    refine unique hd hk hk' hc ⟨?_, hri⟩ ⟨?_, ?_⟩ (r := rowOf d k pm ii)
    · unfold rowOf; omega
    · unfold rowOf at *; omega
    · rw [hr]; exact hri'
  subst hkk
  unfold rowOf at hr
  unfold colOf at hp
  simp only [BM, BN] at *
  refine ⟨trivial, by omega, by omega, hhh⟩

/-- Tiles of distinct programs share no byte of `out`. -/
theorem W_disjoint {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gx d * d.nmeta * d.H)
    (hj : j < gx d * d.nmeta * d.H) (hij : i ≠ j) : Disjoint (W d i) (W d j) := by
  obtain ⟨hk, hh, hpm, hpn, -⟩ := decode_lt hd hi
  obtain ⟨hk', hh', hpm', hpn', -⟩ := decode_lt hd hj
  refine d.out.elems_disjoint ?_
  rw [Set.disjoint_left]
  intro x hx hx'
  obtain ⟨ii, jj, h1, h2, hri, hcj, hxe⟩ := tile_elem hx
  obtain ⟨ii', jj', h1', h2', hri', hcj', hxe'⟩ := tile_elem hx'
  rw [hxe] at hxe'
  obtain ⟨e1, e2, e3, e4⟩ := tile_eq hd hk hk' hh hh' h1 h2 h1' h2' hri hri' hcj hcj' hxe'
  exact hij (prog_ext hd hi hj e1 e2 e3 e4)

end Claims.SsdChunkScan
