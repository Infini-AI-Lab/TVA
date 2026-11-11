import Claim
import TVA.Proof
import Proof.Geo

/-!
# Bounds of the program geometry

Every program's sequence is a covered sequence, its chunk index and head are inside the grid, and
the element a lane touches is inside the flat `[B, T, H]` tensor whenever the lane is enabled.
-/

namespace Claims.ChunkCumsum

open TVA

/-- The number of rows of the grid (the first axis, before `B·H`). -/
def rowsOf (d : Inst) : Nat := if d.varlen then d.NT else (d.T + d.BT - 1) / d.BT

/-- The grid has `rowsOf · B · H` programs. -/
theorem gridSize_eq (d : Inst) : gridSize d = rowsOf d * d.B * d.H := rfl

/-- `B·H` is positive. -/
theorem BH_pos {d : Inst} (hd : d.ok) : 0 < BH d := Nat.mul_pos hd.B_pos hd.H_pos

/-- A program of the grid has its row below `rowsOf`. -/
theorem row_lt {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gridSize d) : row d i < rowsOf d := by
  rw [gridSize_eq] at hi
  unfold row
  rw [Nat.div_lt_iff_lt_mul (BH_pos hd)]
  simpa [BH, Nat.mul_assoc] using hi

/-- A program's head is below `H`. -/
theorem headOf_lt (d : Inst) (hd : d.ok) (i : Nat) : headOf d i < d.H :=
  Nat.mod_lt _ hd.H_pos

/-- A program's index within its row is below `B·H`. -/
theorem lin_lt {d : Inst} (hd : d.ok) (i : Nat) : lin d i < BH d := by
  unfold lin BH
  exact Nat.mod_lt _ (BH_pos hd)

/-- A program's batch row is below `B`. -/
theorem batchOf_lt {d : Inst} (hd : d.ok) (i : Nat) : batchOf d i < d.B := by
  unfold batchOf
  rw [Nat.div_lt_iff_lt_mul hd.H_pos]
  unfold lin BH
  have h := Nat.mod_lt i (BH_pos hd)
  calc i % (d.B * d.H) < d.B * d.H := h
    _ = d.B * d.H := rfl

/-- A program's sequence is below the number of sequences. -/
theorem nOf_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gridSize d) : nOf d i < nSeq d := by
  cases hvv : d.varlen
  · rw [nOf_false hvv, nSeq_false hvv]
    exact batchOf_lt hd i
  · rw [nOf_true hvv, nSeq_true hvv]
    have hr : row d i < d.NT := by
      have := row_lt hd hi
      simpa [rowsOf, hvv] using this
    obtain ⟨-, -, -, hseq, -⟩ := hT hvv
    obtain ⟨h1, h2, -, -⟩ := hseq (row d i) hr
    omega

/-- A lane is enabled exactly when its offset `chunk·BT + l` is below the sequence length. -/
theorem en_iff {d : Inst} (i l : Nat) : en d i l = true ↔ chkOf d i * d.BT + l < lenOf d i := by
  simp [en]

/-- `cu_seqlens` is nondecreasing up to `N`. -/
theorem cuAt_mono_le {d : Inst} (hT : TablesOk d) {a b : Nat} (hab : a ≤ b) (hb : b ≤ d.N) :
    cuAt d a ≤ cuAt d b := by
  induction b with
  | zero => simp_all
  | succ k ih =>
    rcases Nat.eq_or_lt_of_le hab with h | h
    · subst h; exact le_rfl
    · have hk : k ≤ d.N := by omega
      exact le_trans (ih (by omega) hk) (cuAt_step hT (by omega))

/-- Every sequence ends within the flat tensor's `B·T` positions. -/
theorem hi_le_BT {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {n : Nat}
    (hn : n < nSeq d) : hi d n ≤ d.B * d.T := by
  cases hvv : d.varlen
  · rw [hi_false hvv, nSeq_false hvv] at *
    have hb : n + 1 ≤ d.B := hn
    calc n * d.T + d.T = (n + 1) * d.T := by ring
      _ ≤ d.B * d.T := Nat.mul_le_mul_right _ hb
  · rw [hi_true hvv, nSeq_true hvv] at *
    obtain ⟨-, hmono, hcuT, -, -⟩ := hT hvv
    have h1 : 0 ≤ cuAt d (n + 1) := cuAt_nonneg (hT hvv) (by omega)
    have h2 : cuAt d (n + 1) ≤ cuAt d d.N := cuAt_mono_le (hT hvv) (by omega) (by omega)
    have h3 : (cuAt d (n + 1)).toNat ≤ (cuAt d d.N).toNat := Int.toNat_le_toNat h2
    have h4 : (cuAt d d.N).toNat ≤ d.T := by
      have hnn : (0 : Int) ≤ cuAt d d.N := le_trans h1 h2
      omega
    have h5 : d.T ≤ d.B * d.T := by
      have hb : 1 ≤ d.B := hd.B_pos
      nlinarith
    exact le_trans h3 (le_trans h4 h5)

/-- An enabled lane's position is inside its sequence. -/
theorem pos_lt_hi {d : Inst} (hT : d.varlen = true → TablesOk d) {i l : Nat}
    (hn : nOf d i < nSeq d) (he : en d i l = true) : pos d i l < hi d (nOf d i) := by
  rw [en_iff] at he
  have hbl := baseOf_add_len hT i hn
  unfold pos
  omega

/-- An enabled lane's element is inside the `B·T·H` tensor. -/
theorem elem_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i l : Nat}
    (hn : nOf d i < nSeq d) (he : en d i l = true) : elem d i l < d.B * d.T * d.H := by
  have hp := pos_lt_hi hT hn he
  have hTle := hi_le_BT hd hT hn
  have hh := headOf_lt d hd i
  have hpT : pos d i l < d.B * d.T := lt_of_lt_of_le hp hTle
  unfold elem
  have h1 : pos d i l * d.H + headOf d i < pos d i l * d.H + d.H := by omega
  have h2 : pos d i l * d.H + d.H ≤ d.B * d.T * d.H := by
    have h3 : pos d i l * d.H + d.H = (pos d i l + 1) * d.H := by ring
    rw [h3]
    calc (pos d i l + 1) * d.H ≤ (d.B * d.T) * d.H := Nat.mul_le_mul_right _ hpT
      _ = d.B * d.T * d.H := by ring
  exact lt_of_lt_of_le h1 h2

/-- The element index determines the lane of a program. -/
theorem elem_inj {d : Inst} (hd : d.ok) {i l l' : Nat} (h : elem d i l = elem d i l') : l = l' := by
  have hH := hd.H_pos
  have hh := headOf_lt d hd i
  have hpos : pos d i l = pos d i l' := by
    have h1 : elem d i l / d.H = pos d i l := by
      unfold elem
      rw [Nat.mul_comm (pos d i l) d.H, Nat.mul_add_div hH, Nat.div_eq_of_lt hh, Nat.add_zero]
    have h2 : elem d i l' / d.H = pos d i l' := by
      unfold elem
      rw [Nat.mul_comm (pos d i l') d.H, Nat.mul_add_div hH, Nat.div_eq_of_lt hh, Nat.add_zero]
    rw [← h1, ← h2, h]
  unfold pos at hpos
  omega

/-- The element index determines the head of a program's flat position. -/
theorem elem_mod {d : Inst} (hd : d.ok) (i l : Nat) : elem d i l % d.H = headOf d i := by
  have hh := headOf_lt d hd i
  unfold elem
  rw [Nat.mul_comm (pos d i l) d.H, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hh]

/-- The flat position of an element is its quotient by `H`. -/
theorem elem_div {d : Inst} (hd : d.ok) (i l : Nat) : elem d i l / d.H = pos d i l := by
  have hH := hd.H_pos
  have hh := headOf_lt d hd i
  unfold elem
  rw [Nat.mul_comm (pos d i l) d.H, Nat.mul_add_div hH, Nat.div_eq_of_lt hh, Nat.add_zero]

end Claims.ChunkCumsum
