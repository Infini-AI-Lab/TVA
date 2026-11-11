import Claim
import TVA.Proof
import Proof.Geo

/-!
# Bounds of the program geometry

Every program's sequence is a covered sequence, its value head is inside `HV`, and the token of an
enabled lane is inside its sequence, hence inside the flat `[B·T]` token axis.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- `BT` is positive. -/
theorem bt_pos {d : Inst} (hd : d.ok) : 0 < d.BT := by
  rcases hd.bt_ok with h | h | h <;> omega

/-- A program's value head is below `HV`. -/
theorem hvOf_lt {d : Inst} (hd : d.ok) (bh : Nat) : hvOf d bh < d.HV := Nat.mod_lt _ hd.HV_pos

/-- A program of the grid has its batch row below `B`. -/
theorem bOf_lt {d : Inst} (hd : d.ok) {bh : Nat} (hbh : bh < nRows d) : bOf d bh < d.B := by
  unfold bOf
  rw [Nat.div_lt_iff_lt_mul hd.HV_pos]
  exact hbh

/-- On a variable-length launch the batch row is `0`. -/
theorem bOf_eq_zero {d : Inst} (hd : d.ok) (h : d.varlen = true) {bh : Nat} (hbh : bh < nRows d) :
    bOf d bh = 0 := by
  have hb := bOf_lt hd hbh
  have h1 := hd.varlen_B h
  omega

/-- A program's row is below the number of rows of the table (variable length). -/
theorem row_lt_NT {d : Inst} (hv : d.varlen = true) {r : Nat} (hr : r < nChunks d) : r < d.NT := by
  simpa [nChunks, hv] using hr

/-- A program's sequence is below the number of sequences. -/
theorem nOf_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {r bh : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d) : nOf d r bh < nSeq d := by
  cases hvv : d.varlen
  · rw [nOf_false hvv, nSeq_false hvv]
    exact bOf_lt hd hbh
  · rw [nOf_true hvv, nSeq_true hvv]
    obtain ⟨-, -, -, hseq, -⟩ := hT hvv
    obtain ⟨h1, h2, -, -⟩ := hseq r (row_lt_NT hvv hr)
    omega

/-- A lane is enabled exactly when its offset `chunk·BT + l` is below the sequence length. -/
theorem en_iff {d : Inst} (r bh l : Nat) :
    en d r bh l = true ↔ chkOf d r bh * d.BT + l < lenOf d r bh := by
  simp [en]

/-- Every sequence ends within the flat tensor's `B·T` tokens. -/
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

/-- An enabled lane's token is inside its sequence. -/
theorem pos_lt_hi {d : Inst} (hT : d.varlen = true → TablesOk d) {r bh l : Nat}
    (hn : nOf d r bh < nSeq d) (he : en d r bh l = true) : pos d r bh l < hi d (nOf d r bh) := by
  rw [en_iff] at he
  have hbl := baseOf_add_len hT r bh hn
  unfold pos
  omega

/-- An enabled lane's token is at least its sequence's start. -/
theorem lo_le_pos (d : Inst) (r bh l : Nat) : lo d (nOf d r bh) ≤ pos d r bh l := by
  unfold pos baseOf; omega

/-- An enabled lane's token is inside the flat token axis. -/
theorem pos_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {r bh l : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d) (he : en d r bh l = true) :
    pos d r bh l < d.B * d.T :=
  lt_of_lt_of_le (pos_lt_hi hT (nOf_lt hd hT hr hbh) he)
    (hi_le_BT hd hT (nOf_lt hd hT hr hbh))

/-- Element `(tok, hv, e)` of a `[B·T, HV, V]` tensor is inside it. -/
theorem oIdx_lt {d : Inst} {tok hv e : Nat} (ht : tok < d.B * d.T) (hh : hv < d.HV)
    (he : e < d.V) : oIdx d tok hv e < d.B * d.T * d.HV * d.V := by
  unfold oIdx
  have h1 : (tok * d.HV + hv) * d.V + e < (tok * d.HV + hv) * d.V + d.V := by omega
  have h2 : (tok * d.HV + hv) * d.V + d.V = (tok * d.HV + hv + 1) * d.V := by ring
  have h3 : tok * d.HV + hv + 1 ≤ (tok + 1) * d.HV := by
    have : (tok + 1) * d.HV = tok * d.HV + d.HV := by ring
    omega
  have h4 : (tok + 1) * d.HV ≤ d.B * d.T * d.HV := Nat.mul_le_mul_right _ ht
  calc oIdx d tok hv e < (tok * d.HV + hv + 1) * d.V := by unfold oIdx; omega
    _ ≤ (d.B * d.T * d.HV) * d.V := Nat.mul_le_mul_right _ (le_trans h3 h4)

/-- A program's written elements are inside `dv`'s tensor. -/
theorem wSet_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {r bh : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d) :
    ∀ x ∈ wSet d r bh, x < d.dv.len := by
  rintro x ⟨l, e, hl, he, hev, rfl⟩
  exact lt_of_lt_of_le (oIdx_lt (pos_lt hd hT hr hbh he) (hvOf_lt hd bh) hev) hd.dv_len

/-- A program's write footprint lies in the bytes of `dv`. -/
theorem W_subset_dvBytes {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {r bh : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d) : W d r bh ⊆ d.dv.bytes :=
  Buf.elems_sub_bytes d.dv (wSet_lt hd hT hr hbh)

end Claims.ChunkBwdDvLocal
