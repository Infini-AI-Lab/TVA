import Claim
import TVA.Proof
import Proof.Util

/-!
# The geometry of one program

Program `(r, bh)` of the grid (`program_id(0) = r`, `program_id(1) = bh`) handles the chunk
`chkOf` of the sequence `nOf` for the value head `hvOf`: on a fixed-length launch the sequence is
the batch row `bh // HV` and the chunk is `r`; on a variable-length launch both are read from row
`r` of `chunk_indices`. Its lane `l` is the token `pos r bh l` of the flat token axis, enabled
when its offset `chkOf * BT + l` is inside the sequence (`en`). It writes the elements
`(pos, hv, e)` of `dv`, `e < V`, of its enabled lanes.

The footprints `R` (read) and `W` (write) and the post-condition `Q` are stated with these.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- Program `(r, bh)`'s value head, `i_bh % HV`. -/
def hvOf (d : Inst) (bh : Nat) : Nat := bh % d.HV

/-- Program `(r, bh)`'s batch row (fixed length), `i_bh // HV`. -/
def bOf (d : Inst) (bh : Nat) : Nat := bh / d.HV

/-- Program `(r, bh)`'s sequence. -/
def nOf (d : Inst) (r bh : Nat) : Nat :=
  if d.varlen then (seqAt d r).toNat else bOf d bh

/-- Program `(r, bh)`'s chunk within its sequence. -/
def chkOf (d : Inst) (r _bh : Nat) : Nat :=
  if d.varlen then (chkAt d r).toNat else r

/-- The first flat token of program `(r, bh)`'s sequence. -/
def baseOf (d : Inst) (r bh : Nat) : Nat := lo d (nOf d r bh)

/-- The length of program `(r, bh)`'s sequence. -/
def lenOf (d : Inst) (r bh : Nat) : Nat := hi d (nOf d r bh) - lo d (nOf d r bh)

/-- The flat token of program `(r, bh)`'s lane `l`. -/
def pos (d : Inst) (r bh l : Nat) : Nat := baseOf d r bh + chkOf d r bh * d.BT + l

/-- Lane `l` of program `(r, bh)` is enabled: its offset in the sequence is inside it. -/
def en (d : Inst) (r bh l : Nat) : Bool := decide (chkOf d r bh * d.BT + l < lenOf d r bh)

/-- The element of `dv` that lane `l`, column `e` of program `(r, bh)` writes. -/
def wEl (d : Inst) (r bh l e : Nat) : Nat := oIdx d (pos d r bh l) (hvOf d bh) e

/-- The elements program `(r, bh)` writes: those of its enabled lanes. -/
def wSet (d : Inst) (r bh : Nat) : Set Nat :=
  {x | ∃ l e, l < d.BT ∧ en d r bh l = true ∧ e < d.V ∧ x = wEl d r bh l e}

/-- The bytes program `(r, bh)` reads: the input buffers (and the tables on a variable-length
launch). -/
def R (d : Inst) (_r _bh : Nat) : Set Int :=
  d.q.bytes ∪ d.k.bytes ∪ d.dO.bytes ∪ (if d.useG then d.g.bytes else ∅) ∪
    (if d.varlen then d.cu.bytes ∪ d.ci.bytes else ∅)

/-- The bytes program `(r, bh)` writes: its elements of `dv`. -/
def W (d : Inst) (r bh : Nat) : Set Int := d.dv.elems (wSet d r bh)

/-- What program `(r, bh)` leaves: every enabled lane and column of `dv` holds a finite
`bfloat16` within the tolerance of the exact value gradient. -/
def Q (d : Inst) (init : Mem) (r bh : Nat) (m : Mem) : Prop :=
  ∀ l, l < d.BT → en d r bh l = true → ∀ e, e < d.V →
    ∃ c : BF16, rdB m d.dv (wEl d r bh l e) = some c ∧ c.isFinite = true ∧
      |c.toReal - ref d init (nOf d r bh) (pos d r bh l) (hvOf d bh) e| ≤
        tol d init (nOf d r bh) (pos d r bh l) (hvOf d bh) e

/-! ## The two branches of the layout definitions -/

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

/-- On a variable-length launch, program `(r, bh)`'s sequence is named by row `r` of `chunk_indices`. -/
theorem nOf_true {d : Inst} (h : d.varlen = true) (r bh : Nat) :
    nOf d r bh = (seqAt d r).toNat := by simp [nOf, h]

/-- On a fixed-length launch, program `(r, bh)`'s sequence is its batch row. -/
theorem nOf_false {d : Inst} (h : d.varlen = false) (r bh : Nat) : nOf d r bh = bOf d bh := by
  simp [nOf, h]

/-- On a variable-length launch, the chunk is named by row `r` of `chunk_indices`. -/
theorem chkOf_true {d : Inst} (h : d.varlen = true) (r bh : Nat) :
    chkOf d r bh = (chkAt d r).toNat := by simp [chkOf, h]

/-- On a fixed-length launch, the chunk is the grid's first coordinate `r`. -/
theorem chkOf_false {d : Inst} (h : d.varlen = false) (r bh : Nat) : chkOf d r bh = r := by
  simp [chkOf, h]

/-! ## Bounds of the sequence layout -/

/-- `cu_seqlens` is nonnegative through `N`. -/
theorem cuAt_nonneg {d : Inst} (hT : TablesOk d) {n : Nat} (hn : n ≤ d.N) : 0 ≤ cuAt d n := by
  obtain ⟨h0, hmono, -, -, -⟩ := hT
  induction n with
  | zero => exact h0
  | succ k ih => exact le_trans (ih (by omega)) (hmono k (by omega))

/-- `cu_seqlens` is nondecreasing step by step. -/
theorem cuAt_step {d : Inst} (hT : TablesOk d) {n : Nat} (hn : n < d.N) :
    cuAt d n ≤ cuAt d (n + 1) :=
  hT.2.1 n hn

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

/-- `lo n ≤ hi n` on the covered sequences. -/
theorem lo_le_hi {d : Inst} (hT : d.varlen = true → TablesOk d) {n : Nat} (hn : n < nSeq d) :
    lo d n ≤ hi d n := by
  cases hvv : d.varlen
  · rw [lo_false hvv, hi_false hvv]
    omega
  · rw [lo_true hvv, hi_true hvv, nSeq_true hvv] at *
    exact Int.toNat_le_toNat (cuAt_step (hT hvv) hn)

/-- The length of a sequence adds back to `hi`. -/
theorem baseOf_add_len {d : Inst} (hT : d.varlen = true → TablesOk d) (r bh : Nat)
    (hn : nOf d r bh < nSeq d) : baseOf d r bh + lenOf d r bh = hi d (nOf d r bh) := by
  unfold baseOf lenOf
  have := lo_le_hi hT hn
  omega

end Claims.ChunkBwdDvLocal
