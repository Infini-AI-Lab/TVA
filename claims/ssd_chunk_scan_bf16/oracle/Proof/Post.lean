import Proof.Footprint

/-!
# The programs' results give `Post`

Every element of `out` below `seqlen·nheads·headdim` is in the tile of the program that owns it,
and no other byte is written.
-/

namespace Claims.SsdChunkScan

open TVA

/-- Every tile element is an element of `out` below `seqlen · nheads · headdim`. -/
theorem W_sub {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * d.nmeta * d.H) :
    W d i ⊆ d.out.elems {j | j < d.T * (d.H * d.P)} := by
  obtain ⟨hk, hh, hpm, hpn, -⟩ := decode_lt hd hi
  refine d.out.elems_mono ?_
  intro x hx
  obtain ⟨ii, jj, h1, h2, hri, hcj, hxe⟩ := tile_elem hx
  have hrow : lo d (kOf d i) ≤ rowOf d (kOf d i) (pmOf d i) ii ∧
      rowOf d (kOf d i) (pmOf d i) ii < lend d (kOf d i) := ⟨by unfold rowOf; omega, hri⟩
  have hT := (tok_of_row hd hk hrow).1
  have hP := hcj
  rw [hxe]
  show outIdx d _ _ _ < _
  unfold outIdx
  have e1 : (tokOf d (kOf d i) (pmOf d i) ii + 1) * (d.H * d.P) ≤ d.T * (d.H * d.P) :=
    Nat.mul_le_mul_right _ hT
  have e2 : (hOf d i + 1) * d.P ≤ d.H * d.P := Nat.mul_le_mul_right _ hh
  have e3 : (tokOf d (kOf d i) (pmOf d i) ii + 1) * (d.H * d.P) =
      tokOf d (kOf d i) (pmOf d i) ii * (d.H * d.P) + d.H * d.P := by ring
  have e4 : (hOf d i + 1) * d.P = hOf d i * d.P + d.P := by ring
  unfold ColOk at hP
  omega

/-- The element `(t, h, p)` of `out` lies in the tile of the program that owns it, with the tile
coordinates of its row and column. -/
theorem owner {d : Inst} (hd : d.ok) {t : Nat} (p : Nat) (ht : t < d.T) (hp : p < d.P) :
    ∃ k pm pn ii jj : Nat, k < d.nmeta ∧ pm < nmt d ∧ pn < npn d ∧ ii < BM ∧ jj < BN ∧
      RowOk d k pm ii ∧ ColOk d pn jj ∧ tokOf d k pm ii = t ∧ colOf pn jj = p := by
  obtain ⟨k, hk, hc, hlo, hl⟩ := cover hd ht
  have hcs := cs_pos hd
  have hlc : lc d k = t / d.cs := hc
  refine ⟨k, (t % d.cs - lo d k) / BM, p / BN, (t % d.cs - lo d k) % BM, p % BN, hk, ?_, ?_,
    Nat.mod_lt _ (by norm_num [BM]), Nat.mod_lt _ (by norm_num [BN]), ?_, ?_, ?_, ?_⟩
  · unfold nmt
    have := lend_le hd hk
    have := Nat.mod_lt t hcs
    simp only [BM]; clear hlc hc; generalize t % d.cs = r at *; omega
  · unfold npn; simp only [BN]; omega
  · unfold RowOk rowOf; simp only [BM]; omega
  · unfold ColOk colOf; simp only [BN]; omega
  · unfold tokOf rowOf
    have h1 := Nat.div_add_mod t d.cs
    rw [Nat.mul_comm] at h1
    simp only [BM]; rw [hlc]
    generalize t / d.cs * d.cs = q at *
    generalize t % d.cs = r at *
    omega
  · unfold colOf; simp only [BN]; omega

end Claims.SsdChunkScan
