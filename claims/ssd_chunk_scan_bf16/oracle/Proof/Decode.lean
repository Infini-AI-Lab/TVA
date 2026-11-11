import Proof.CoreSafe

/-!
# The program index

`program_id` of program `i` and the decoding of `i` into `(k, pm, pn, h)`.
-/

namespace Claims.SsdChunkScan

open TVA

/-- `cdiv(headdim, 64)` is positive. -/
theorem npn_pos {d : Inst} (hd : d.ok) : 0 < npn d := by
  have := hd.P_pos; unfold npn; simp only [BN]; omega

/-- `cdiv(chunk_size, 64)` is positive. -/
theorem nmt_pos {d : Inst} (hd : d.ok) : 0 < nmt d := by
  have := cs_pos hd; unfold nmt; simp only [BM]; omega

/-- The grid's `x` is positive. -/
theorem gx_pos {d : Inst} (hd : d.ok) : 0 < gx d := by
  rw [gx_eq]; exact Nat.mul_pos (nmt_pos hd) (npn_pos hd)

/-- `program_id(0)` of program `i`. -/
theorem pid0_eq (d : Inst) (i : Nat) :
    tl.program_id ((launch d).ctx i) 0 = IntV.ofNat (i % gx d) := rfl

/-- `program_id(1)` of program `i` is its logical chunk. -/
theorem pid1_eq (d : Inst) (i : Nat) :
    tl.program_id ((launch d).ctx i) 1 = IntV.ofNat (kOf d i) := rfl

/-- `program_id(2)` of program `i` is its head. -/
theorem pid2_eq (d : Inst) (i : Nat) :
    tl.program_id ((launch d).ctx i) 2 = IntV.ofNat (hOf d i) := rfl

/-- The decoding of a program index is inside the grid. -/
theorem decode_lt {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * d.nmeta * d.H) :
    kOf d i < d.nmeta ∧ hOf d i < d.H ∧ pmOf d i < nmt d ∧ pnOf d i < npn d ∧ i % gx d < gx d := by
  have hg := gx_pos hd
  have hnp := npn_pos hd
  have hm : 0 < d.nmeta := by
    rcases Nat.eq_zero_or_pos d.nmeta with h | h
    · rw [h] at hi; simp at hi
    · exact h
  have ha : i % gx d < gx d := Nat.mod_lt _ hg
  refine ⟨Nat.mod_lt _ hm, ?_, ?_, Nat.mod_lt _ hnp, ha⟩
  · unfold hOf
    rw [Nat.div_lt_iff_lt_mul (Nat.mul_pos hg hm)]
    linarith [Nat.mul_comm d.H (gx d * d.nmeta)]
  · unfold pmOf
    rw [Nat.div_lt_iff_lt_mul hnp]
    exact lt_of_lt_of_eq ha (gx_eq d)

end Claims.SsdChunkScan
