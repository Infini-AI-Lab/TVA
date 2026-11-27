import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Lanes

/-!
# Programs and the elements they write

A program is determined by its output row, head and feature block (`progOf`, `prog_eq`), and every
(row, head, block) of the output has a program (`prog_props`). So distinct programs write disjoint
elements of `o` (`wset_disjoint`); `o` shares no byte with `x`, `cu_seqlens` or `chunk_indices`, so
no program writes what another reads (`W_disjoint_R`).
-/

namespace Claims.MeanPoolingFwd

open TVA

/-- The grid row of output row `g`. -/
def rowsOf (d : Inst) (g : Nat) : Nat := if d.varlen then g else g % ntFix d

/-- The batch row of output row `g`. -/
def batchOf (d : Inst) (g : Nat) : Nat := if d.varlen then 0 else g / ntFix d

/-- The program writing output row `g`, head `h`, feature block `jb`. -/
def progOf (d : Inst) (g h jb : Nat) : Nat :=
  (batchOf d g * d.H + h) * gx d + (rowsOf d g * nD d + jb)

/-- **A program is its output row, head and feature block.** -/
theorem prog_eq {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) :
    progOf d (outRow d i) (hOf d i) (iD d i) = i := by
  obtain ⟨-, -, hp1⟩ := coords_lt hi
  have hb : batchOf d (outRow d i) = bOf d i ∧ rowsOf d (outRow d i) = rG d i := by
    cases hv : d.varlen
    · obtain ⟨h1, h2⟩ := outRow_div_mod hd hv hi
      constructor <;> simp [batchOf, rowsOf, hv, h1, h2]
    · have hB := hd.varlen_B hv
      have h1 : bOf d i = 0 := by
        have := (bOf_lt hd hi).1; omega
      constructor <;> simp [batchOf, rowsOf, outRow, hv, h1]
  unfold progOf
  rw [hb.1, hb.2]
  have e1 : bOf d i * d.H + hOf d i = i / gx d := by
    unfold bOf hOf; rw [← hp1, Nat.mul_comm]; exact Nat.div_add_mod _ _
  have e2 : rG d i * nD d + iD d i = i % gx d := by
    unfold rG iD p0; rw [Nat.mul_comm]; exact Nat.div_add_mod _ _
  rw [e1, e2, Nat.mul_comm]; exact Nat.div_add_mod _ _

/-- **Every output row, head and feature block has its program.** -/
theorem prog_props {d : Inst} (hd : d.ok) {g h jb : Nat} (hg : g < nRows d) (hh : h < d.H)
    (hjb : jb < nD d) :
    progOf d g h jb < gx d * gy d ∧ outRow d (progOf d g h jb) = g ∧
      hOf d (progOf d g h jb) = h ∧ iD d (progOf d g h jb) = jb := by
  have hnD := nD_pos hd
  -- the row and batch
  have hrb : rowsOf d g < ntG d ∧ batchOf d g < d.B ∧
      (if d.varlen then rowsOf d g else batchOf d g * ntFix d + rowsOf d g) = g := by
    cases hv : d.varlen
    · simp only [rowsOf, batchOf, ntG, nRows, hv, Bool.false_eq_true, ite_false] at hg ⊢
      have hn : 0 < ntFix d := by
        rcases Nat.eq_zero_or_pos (ntFix d) with h0 | h0
        · rw [h0, Nat.mul_zero] at hg; omega
        · exact h0
      refine ⟨Nat.mod_lt _ hn, (Nat.div_lt_iff_lt_mul hn).mpr hg, ?_⟩
      rw [Nat.mul_comm]; exact Nat.div_add_mod _ _
    · simp only [rowsOf, batchOf, ntG, nRows, hv, ite_true] at hg ⊢
      exact ⟨hg, hd.B_pos, by simp⟩
  obtain ⟨hr, hbB, hgeq⟩ := hrb
  set r := rowsOf d g
  set b := batchOf d g
  have hp0 : r * nD d + jb < gx d := by
    unfold gx
    have := Nat.mul_le_mul_right (nD d) (show r + 1 ≤ ntG d by omega)
    rw [Nat.add_mul, Nat.one_mul] at this
    rw [Nat.mul_comm (nD d)]; omega
  have hp1 : b * d.H + h < gy d := by
    unfold gy
    have := Nat.mul_le_mul_right d.H (show b + 1 ≤ d.B by omega)
    rw [Nat.add_mul, Nat.one_mul] at this; omega
  have hgx : 0 < gx d := by omega
  have hlt : (b * d.H + h) * gx d + (r * nD d + jb) < gx d * gy d := by
    have := Nat.mul_le_mul_right (gx d) (show b * d.H + h + 1 ≤ gy d by omega)
    rw [Nat.add_mul, Nat.one_mul] at this
    rw [Nat.mul_comm (gx d)]; omega
  have hq0 : p0 d ((b * d.H + h) * gx d + (r * nD d + jb)) = r * nD d + jb := by
    unfold p0; rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hp0]
  have hq1 : p1 d ((b * d.H + h) * gx d + (r * nD d + jb)) = b * d.H + h := by
    unfold p1
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hgx, Nat.div_eq_of_lt hp0, Nat.zero_add,
      Nat.mod_eq_of_lt hp1]
  have hiD : iD d ((b * d.H + h) * gx d + (r * nD d + jb)) = jb := by
    unfold iD; rw [hq0, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hjb]
  have hrG : rG d ((b * d.H + h) * gx d + (r * nD d + jb)) = r := by
    unfold rG; rw [hq0, Nat.add_comm, Nat.add_mul_div_right _ _ hnD, Nat.div_eq_of_lt hjb, Nat.zero_add]
  have hbO : bOf d ((b * d.H + h) * gx d + (r * nD d + jb)) = b := by
    unfold bOf; rw [hq1, Nat.add_comm, Nat.add_mul_div_right _ _ hd.H_pos, Nat.div_eq_of_lt hh,
      Nat.zero_add]
  have hhO : hOf d ((b * d.H + h) * gx d + (r * nD d + jb)) = h := by
    unfold hOf; rw [hq1, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hh]
  refine ⟨hlt, ?_, hhO, hiD⟩
  unfold progOf
  show (if d.varlen then _ else _) = g
  rw [hrG, hbO]; exact hgeq

/-! ## The written elements -/

/-- An element program `i` writes decodes to its output row, head and feature block. -/
theorem wset_decode {d : Inst} (hd : d.ok) {i e : Nat} (he : e ∈ wset d i) :
    e / d.D / d.H = outRow d i ∧ e / d.D % d.H = hOf d i ∧ e % d.D / d.BD = iD d i := by
  obtain ⟨l, hl, rfl⟩ := he
  have hD := hd.D_pos
  have hH := hd.H_pos
  have hh := Nat.mod_lt (p1 d i) hH
  have hbd := bd_pos hd
  have e1 : (eO d i + (iD d i * d.BD + l)) / d.D = outRow d i * d.H + hOf d i := by
    unfold eO; rw [Nat.add_comm, Nat.add_mul_div_right _ _ hD, Nat.div_eq_of_lt hl, Nat.zero_add]
  have e2 : (eO d i + (iD d i * d.BD + l)) % d.D = iD d i * d.BD + l := by
    unfold eO; rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]
  rw [e1, e2]
  refine ⟨?_, ?_, ?_⟩
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hH, Nat.div_eq_of_lt (by unfold hOf; exact hh),
      Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by unfold hOf; exact hh)]
  · rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_div_left _ _ hbd, Nat.div_eq_of_lt l.isLt, Nat.zero_add]

/-- Every element program `i` writes is inside the output. -/
theorem wset_lt {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) {e : Nat}
    (he : e ∈ wset d i) : e < d.o.len := by
  obtain ⟨l, hl, rfl⟩ := he
  have h1 := eO_last hd hi
  have h2 := hd.o_len
  omega

/-- **Distinct programs write disjoint elements.** -/
theorem wset_disjoint {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gx d * gy d)
    (hj : j < gx d * gy d) (hij : i ≠ j) : Disjoint (wset d i) (wset d j) := by
  rw [Set.disjoint_left]
  intro e hei hej
  obtain ⟨a1, a2, a3⟩ := wset_decode hd hei
  obtain ⟨b1, b2, b3⟩ := wset_decode hd hej
  apply hij
  rw [← prog_eq hd hi, ← prog_eq hd hj, ← a1, ← a2, ← a3, b1, b2, b3]

/-! ## The read footprint lies in the inputs -/

/-- The read footprint lies in the bytes of `x`, `cu_seqlens` and `chunk_indices`. -/
theorem R_sub {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) : R d i ⊆ d.x.bytes ∪ (if d.varlen then d.cu.bytes ∪ d.ci.bytes else ∅) := by
  intro z hz
  rcases hz with hz | hz
  · refine Or.inl (d.x.tileAt_sub_bytes (Or.inr (Or.inr ?_)) hz)
    have h1 := eX_last hd hT hi
    have h2 := hd.x_len
    have := hd.D_pos
    omega
  · unfold extraReads at hz
    cases hv : d.varlen
    · simp only [hv, Bool.false_eq_true, ite_false] at hz; exact absurd hz (Set.notMem_empty z)
    · simp only [hv, ite_true, Set.mem_union] at hz ⊢
      obtain ⟨hrNT, hsN, -, -, -, -, -⟩ := row_facts hd hv (hT hv) hi
      have hci := hd.ci_len hv
      have hcu := hd.cu_len hv
      refine Or.inr ?_
      rcases hz with ((hz | hz) | hz) | hz
      · exact Or.inr (d.ci.span_sub (j := 2 * rG d i) (by omega) (by push_cast; exact hz))
      · exact Or.inr (d.ci.span_sub (j := 2 * rG d i + 1) (by omega) (by push_cast; exact hz))
      · exact Or.inl (d.cu.span_sub (j := seqN d i) (by omega) hz)
      · exact Or.inl (d.cu.span_sub (j := seqN d i + 1) (by omega) (by push_cast; exact hz))

/-- The write footprint lies in the bytes of `o`. -/
theorem W_sub {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) : W d i ⊆ d.o.bytes :=
  d.o.elems_sub_bytes fun _ he => wset_lt hd hi he

/-- **No program writes what a program reads.** -/
theorem W_disjoint_R {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i j : Nat}
    (hi : i < gx d * gy d) (hj : j < gx d * gy d) : Disjoint (W d i) (R d j) := by
  refine Disjoint.mono (W_sub hd hi) (R_sub hd hT hj) ?_
  refine Disjoint.union_right hd.o_x ?_
  cases hv : d.varlen
  · simp only [Bool.false_eq_true, ite_false]; exact Set.disjoint_empty _
  · simp only [ite_true]; exact Disjoint.union_right (hd.o_cu hv) (hd.o_ci hv)

end Claims.MeanPoolingFwd
