import Claim
import TVA.Proof
import Proof.Util

/-!
# The geometry of one program

Program `i` of the grid handles the (chunk, batch row, head) triple `(chkOf, nOf, headOf)`:
`row i` is its row of the grid, `nOf i` its sequence, `chkOf i` its chunk in that sequence and
`headOf i` its head. Its lane `l` reads and writes element `elem i l = pos i l * H + headOf i` of
the flat `[B, T, H]` tensor, where `pos i l` is the flat time index, and the lane is enabled when
its offset `chkOf i * BT + l` is inside the sequence (`en i l`).

The footprints `R` (read) and `W` (write) and the postcondition `Q` are stated with these.
-/

namespace Claims.ChunkCumsum

open TVA

/-- The (batch row, head) pairs of the grid's first axis. -/
def BH (d : Inst) : Nat := d.B * d.H

/-- Program `i`'s row of the grid's first axis. -/
def row (d : Inst) (i : Nat) : Nat := i / BH d

/-- Program `i`'s offset within its row (the (batch row, head) pair). -/
def lin (d : Inst) (i : Nat) : Nat := i % BH d

/-- Program `i`'s batch row (fixed length). -/
def batchOf (d : Inst) (i : Nat) : Nat := lin d i / d.H

/-- Program `i`'s head. -/
def headOf (d : Inst) (i : Nat) : Nat := lin d i % d.H

/-- Program `i`'s sequence. -/
def nOf (d : Inst) (i : Nat) : Nat :=
  if d.varlen then (seqAt d (row d i)).toNat else batchOf d i

/-- Program `i`'s chunk within its sequence. -/
def chkOf (d : Inst) (i : Nat) : Nat :=
  if d.varlen then (chkAt d (row d i)).toNat else row d i

/-- The first flat time index of program `i`'s sequence. -/
def baseOf (d : Inst) (i : Nat) : Nat := lo d (nOf d i)

/-- The length of program `i`'s sequence. -/
def lenOf (d : Inst) (i : Nat) : Nat := hi d (nOf d i) - lo d (nOf d i)

/-- The flat time index of program `i`'s lane `l`. -/
def pos (d : Inst) (i l : Nat) : Nat := baseOf d i + chkOf d i * d.BT + l

/-- Lane `l` of program `i` is enabled: its offset in the sequence is inside it. -/
def en (d : Inst) (i l : Nat) : Bool := decide (chkOf d i * d.BT + l < lenOf d i)

/-- Element of the flat `[B, T, H]` tensor that lane `l` of program `i` touches. -/
def elem (d : Inst) (i l : Nat) : Nat := pos d i l * d.H + headOf d i

/-- The lanes a scan at lane `l` combines: `l + 1` forward, `BT - l` reverse. -/
def leavesOf (rev : Bool) (BT l : Nat) : List Nat :=
  if rev then List.range' l (BT - l) else List.range (l + 1)

/-- The bytes program `i` reads from the int tables (only on a variable-length launch). -/
def extraReads (d : Inst) (i : Nat) : Set Int :=
  if d.varlen then
    Mem.span d.idxD (d.ci.ptr (2 * (row d i : Nat))).addr ∪
      Mem.span d.idxD (d.ci.ptr (2 * (row d i : Nat) + 1)).addr ∪
        Mem.span d.idxD (d.cu.ptr (seqAt d (row d i)).toNat).addr ∪
          Mem.span d.idxD (d.cu.ptr ((seqAt d (row d i)).toNat + 1)).addr
  else ∅

/-- The element each lane of program `i` touches, as a function of the lane index. -/
def elemF (d : Inst) (i : Nat) : Idx [d.BT] → Nat := fun j => elem d i j.1.val

/-- The mask of program `i`, as a function of the lane index. -/
def enF (d : Inst) (i : Nat) : Idx [d.BT] → Bool := fun j => en d i j.1.val

/-- Program `i`'s read footprint. -/
def R (d : Inst) (i : Nat) : Set Int :=
  d.s.elems (Buf.img (elemF d i) (enF d i)) ∪ extraReads d i

/-- Program `i`'s write footprint. -/
def W (d : Inst) (i : Nat) : Set Int :=
  d.o.elems (Buf.img (elemF d i) (enF d i))

/-- What program `i` leaves: every enabled lane of `o` holds the value the scan computes there,
within the tolerance when the element is in range. -/
def Q (d : Inst) (i : Nat) (m : Mem) : Prop :=
  ∀ l, l < d.BT → en d i l = true → ∃ v : F32,
    m.read .f32 (d.o.ptr (elem d i l)).addr = some v ∧
      (InRange d (nOf d i) (pos d i l) (headOf d i) → v.isFinite = true ∧
        |v.toRat - sig d * refSum d (nOf d i) (pos d i l) (headOf d i)| ≤
          tol d (nOf d i) (pos d i l) (headOf d i))

/-! ## The two branches of the layout definitions -/

/-- On a variable-length launch, sequence `n` starts at `cu_seqlens[n]`. -/
theorem lo_true {d : Inst} (h : d.varlen = true) (n : Nat) : lo d n = (cuAt d n).toNat := by
  simp [lo, h]

/-- On a fixed-length launch, sequence (row) `n` starts at `n·T`. -/
theorem lo_false {d : Inst} (h : d.varlen = false) (n : Nat) : lo d n = n * d.T := by
  simp [lo, h]

/-- On a variable-length launch, sequence `n` ends at `cu_seqlens[n+1]`. -/
theorem hi_true {d : Inst} (h : d.varlen = true) (n : Nat) : hi d n = (cuAt d (n + 1)).toNat := by
  simp [hi, h]

/-- On a fixed-length launch, sequence (row) `n` ends at `n·T + T`. -/
theorem hi_false {d : Inst} (h : d.varlen = false) (n : Nat) : hi d n = n * d.T + d.T := by
  simp [hi, h]

/-- A variable-length launch has `N` sequences. -/
theorem nSeq_true {d : Inst} (h : d.varlen = true) : nSeq d = d.N := by
  simp [nSeq, h]

/-- A fixed-length launch has `B` sequences. -/
theorem nSeq_false {d : Inst} (h : d.varlen = false) : nSeq d = d.B := by
  simp [nSeq, h]

/-- On a variable-length launch, the base of program `i` is the start of its sequence. -/
theorem baseOf_true {d : Inst} (h : d.varlen = true) (i : Nat) :
    baseOf d i = (cuAt d (nOf d i)).toNat := by
  simp [baseOf, lo_true h]

/-- On a fixed-length launch, the base of program `i` is `n·T` for its row `n`. -/
theorem baseOf_false {d : Inst} (h : d.varlen = false) (i : Nat) : baseOf d i = nOf d i * d.T := by
  simp [baseOf, lo_false h]

/-- On a variable-length launch, program `i`'s sequence is read from `chunk_indices`. -/
theorem nOf_true {d : Inst} (h : d.varlen = true) (i : Nat) : nOf d i = (seqAt d (row d i)).toNat := by
  simp [nOf, h]

/-- On a fixed-length launch, program `i`'s sequence is its batch row. -/
theorem nOf_false {d : Inst} (h : d.varlen = false) (i : Nat) : nOf d i = batchOf d i := by
  simp [nOf, h]

/-- On a variable-length launch, program `i`'s chunk is read from `chunk_indices`. -/
theorem chkOf_true {d : Inst} (h : d.varlen = true) (i : Nat) : chkOf d i = (chkAt d (row d i)).toNat := by
  simp [chkOf, h]

/-- On a fixed-length launch, program `i`'s chunk is its row of the grid. -/
theorem chkOf_false {d : Inst} (h : d.varlen = false) (i : Nat) : chkOf d i = row d i := by
  simp [chkOf, h]

/-! ## Bounds of the sequence layout -/

/-- `cu_seqlens` is nonnegative through `N`. -/
theorem cuAt_nonneg {d : Inst} (hT : TablesOk d) {n : Nat} (hn : n ≤ d.N) : 0 ≤ cuAt d n := by
  obtain ⟨h0, hmono, -, -, -⟩ := hT
  induction n with
  | zero => exact h0
  | succ k ih => exact le_trans (ih (by omega)) (hmono k (by omega))

/-- `cu_seqlens` is nondecreasing. -/
theorem cuAt_step {d : Inst} (hT : TablesOk d) {n : Nat} (hn : n < d.N) :
    cuAt d n ≤ cuAt d (n + 1) :=
  hT.2.1 n hn

/-- `lo n ≤ hi n` on the covered sequences. -/
theorem lo_le_hi {d : Inst} (hT : d.varlen = true → TablesOk d) {n : Nat} (hn : n < nSeq d) :
    lo d n ≤ hi d n := by
  cases hvv : d.varlen
  · rw [lo_false hvv, hi_false hvv]
    omega
  · rw [lo_true hvv, hi_true hvv, nSeq_true hvv] at *
    exact Int.toNat_le_toNat (cuAt_step (hT hvv) hn)

/-- The length of a sequence adds back to `hi`. -/
theorem baseOf_add_len {d : Inst} (hT : d.varlen = true → TablesOk d) (i : Nat)
    (hn : nOf d i < nSeq d) :
    baseOf d i + lenOf d i = hi d (nOf d i) := by
  unfold baseOf lenOf
  have := lo_le_hi hT hn
  omega

end Claims.ChunkCumsum
