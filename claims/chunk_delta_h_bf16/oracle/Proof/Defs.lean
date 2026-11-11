import Proof.NumDefs

/-!
# `chunk_delta_h_bf16`: programs, footprints and the per-column statement

A program `i` of the grid `(cdiv(V, 32)·N·HV,)` is decoded as `i = (n·HV + hv)·NV + jv`: sequence
`n`, value head `hv`, block `jv` of `32` value columns. It reads no output buffer, and writes the
columns of its block of `h`, `v_new` and (with `STORE_FINAL_STATE`) `ht`.

* the decoding and the cells a program writes (`hCells`, `vCells`, `tCells`), per chunk `Wt`;
* `Col`: `Post`'s clause for one column; `ColPre`: its first `t` chunks.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-! ## The program index -/

/-- `cdiv(V, 32)`: the blocks of value columns. -/
def NV (d : Inst) : Nat := (d.V + BV - 1) / BV

/-- The block of value columns of program `i`. -/
def jvOf (d : Inst) (i : Nat) : Nat := i % NV d

/-- The sequence of program `i`. -/
def nOf (d : Inst) (i : Nat) : Nat := i / NV d / d.HV

/-- The value head of program `i`. -/
def hvOf (d : Inst) (i : Nat) : Nat := i / NV d % d.HV

/-- The program owning block `jv` of value head `hv` of sequence `n`. -/
def progOf (d : Inst) (n hv jv : Nat) : Nat := (n * d.HV + hv) * NV d + jv

/-! ## Footprints -/

/-- The bytes the program may read: those outside the three outputs. -/
def Rset (d : Inst) : Set Int := {x | x ∉ d.h.bytes ∧ x ∉ d.vn.bytes ∧ x ∉ d.ht.bytes}

/-- The value columns of block `jv`. -/
def inBlk (d : Inst) (jv j : Nat) : Prop := BV * jv ≤ j ∧ j < BV * jv + BV ∧ j < d.V

/-- The elements of `h` stored by chunk `t`: the state of the program's columns. -/
def hCells (d : Inst) (n hv jv t : Nat) : Set Nat :=
  {x | ∃ c j, c < d.K ∧ inBlk d jv j ∧ x = hIdx d (cbase d n + t) hv c j}

/-- The elements of `v_new` stored by chunk `t`. -/
def vCells (d : Inst) (n hv jv t : Nat) : Set Nat :=
  {x | ∃ i j, i < cl d n t ∧ inBlk d jv j ∧
    x = uIdx d (rowOf d n) (startOf d n + BT * t + i) hv j}

/-- The elements of `ht` stored at the end. -/
def tCells (d : Inst) (n hv jv : Nat) : Set Nat :=
  {x | ∃ c j, c < d.K ∧ inBlk d jv j ∧ x = sIdx d n hv c j}

/-- The bytes chunk `t` writes. -/
def Wt (d : Inst) (n hv jv t : Nat) : Set Int :=
  d.h.elems (hCells d n hv jv t) ∪ d.vn.elems (vCells d n hv jv t)

/-- The bytes the final state's store writes. -/
def Wend (d : Inst) (n hv jv : Nat) : Set Int :=
  if d.storeHt then d.ht.elems (tCells d n hv jv) else ∅

/-- The write footprint of program `i`. -/
def W (d : Inst) (i : Nat) : Set Int :=
  iterW (Wt d (nOf d i) (hvOf d i) (jvOf d i)) (nch d (nOf d i)) ∪ Wend d (nOf d i) (hvOf d i) (jvOf d i)

/-! ## The per-column statement -/

/-- `Post`'s clause for column `j` of value head `hv` of sequence `n`, read in `fin`. -/
def Col (d : Inst) (init fin : Mem) (n hv j : Nat) : Prop :=
  (∀ s, s < lenOf d n →
      ∃ x, rdB fin d.vn (uIdx d (rowOf d n) (startOf d n + s) hv j) = some x ∧
        (Vals init d → (∀ t', t' ≤ s / BT → StOk fin d n hv j t') → x.isFinite = true ∧
          |x.toReal - vrefS init fin d n hv j (s / BT) (s % BT)| ≤
            tolV init fin d n hv j (s / BT) (s % BT))) ∧
    (∀ t, t < nch d n → ∀ c, c < d.K → ∃ x, rdB fin d.h (hIdx d (cbase d n + t) hv c j) = some x) ∧
    ∃ Y : Nat → Nat → F32,
      (Vals init d → ∀ c, c < d.K → (Y 0 c).isFinite = true ∧ (Y 0 c).toReal = h0v init d n hv j c) ∧
      (∀ t, t < nch d n → ∀ c, c < d.K → (Y t c).isFinite = true →
        rdB fin d.h (hIdx d (cbase d n + t) hv c j) = some (FloatV.cvt (Y t c))) ∧
      (∀ t, t < nch d n → Vals init d → (∀ t', t' ≤ t → StOk fin d n hv j t') →
        (∀ c, c < d.K → (Y t c).isFinite = true ∧ (Y (t + 1) c).isFinite = true) ∧
        colNorm d.K (fun c => (Y (t + 1) c).toReal -
          Step init fin d n hv j t (fun c' => (Y t c').toReal) c) ≤ errStep init fin d n hv j t) ∧
      (d.storeHt = true → ∀ c, c < d.K → rdF fin d.ht (sIdx d n hv c j) = some (Y (nch d n) c))

/-- **`Post` is `Col` for every column, and the frame.** -/
theorem post_iff (d : Inst) (init fin : Mem) :
    Post d init fin ↔ (∀ n hv j, n < d.N → hv < d.HV → j < d.V → Col d init fin n hv j) ∧
      ∀ z, z ∉ d.h.elems {i | i < d.B * NTt d * d.HV * d.K * d.V} ∪
        d.vn.elems {i | i < d.B * d.T * d.HV * d.V} ∪
        (if d.storeHt then d.ht.elems {i | i < d.N * d.HV * d.K * d.V} else ∅) →
      fin z = init z := Iff.rfl

/-- **`Col` after `t` chunks**: the clauses of `Col` for the first `t` chunks, with the state
sequence `Y` (its last state `Y t` is the one entering chunk `t`). -/
def ColPre (d : Inst) (init m : Mem) (n hv j t : Nat) (Y : Nat → Nat → F32) : Prop :=
  (∀ s, s < min (lenOf d n) (BT * t) →
      ∃ x, rdB m d.vn (uIdx d (rowOf d n) (startOf d n + s) hv j) = some x ∧
        (Vals init d → (∀ t', t' ≤ s / BT → StOk m d n hv j t') → x.isFinite = true ∧
          |x.toReal - vrefS init m d n hv j (s / BT) (s % BT)| ≤
            tolV init m d n hv j (s / BT) (s % BT))) ∧
    (∀ t', t' < t → ∀ c, c < d.K → ∃ x, rdB m d.h (hIdx d (cbase d n + t') hv c j) = some x) ∧
    (Vals init d → ∀ c, c < d.K → (Y 0 c).isFinite = true ∧ (Y 0 c).toReal = h0v init d n hv j c) ∧
    (∀ t', t' < t → ∀ c, c < d.K → (Y t' c).isFinite = true →
      rdB m d.h (hIdx d (cbase d n + t') hv c j) = some (FloatV.cvt (Y t' c))) ∧
    (∀ t', t' < t → Vals init d → (∀ t'', t'' ≤ t' → StOk m d n hv j t'') →
      (∀ c, c < d.K → (Y t' c).isFinite = true ∧ (Y (t' + 1) c).isFinite = true) ∧
      colNorm d.K (fun c => (Y (t' + 1) c).toReal -
        Step init m d n hv j t' (fun c' => (Y t' c').toReal) c) ≤ errStep init m d n hv j t')

/-- After all the chunks, `ColPre` and the final store give `Col`. -/
theorem col_of_pre {d : Inst} {init m : Mem} {n hv j : Nat} {Y : Nat → Nat → F32}
    (h : ColPre d init m n hv j (nch d n) Y)
    (hT : d.storeHt = true → ∀ c, c < d.K → rdF m d.ht (sIdx d n hv c j) = some (Y (nch d n) c)) :
    Col d init m n hv j := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  have hle : lenOf d n ≤ BT * nch d n := by
    unfold nch BT; omega
  refine ⟨fun s hs => h1 s (lt_min hs (lt_of_lt_of_le hs hle)), h2, Y, h3, h4, h5, hT⟩

/-! ## What a program leaves, and what a loop trip may assume -/

/-- The bytes `ColPre ... t` reads in the output buffers: the stored states of the chunks before
`t`, and the new values of the first `t` chunks' tokens, in column `j`. -/
def colCells (d : Inst) (n hv j t : Nat) : Set Int :=
  d.h.elems {x | ∃ t' c, t' < t ∧ c < d.K ∧ x = hIdx d (cbase d n + t') hv c j} ∪
    d.vn.elems {x | ∃ s, s < min (lenOf d n) (BT * t) ∧
      x = uIdx d (rowOf d n) (startOf d n + s) hv j}

/-- **What program `i` leaves in its memory `m`**: `Post`'s clause for each of its columns. -/
def Q (d : Inst) (init : Mem) (i : Nat) (m : Mem) : Prop :=
  ∀ jj, jj < BV → inBlk d (jvOf d i) (BV * jvOf d i + jj) →
    Col d init m (nOf d i) (hvOf d i) (BV * jvOf d i + jj)

/-- A program's memory: the bytes outside the outputs are the initial ones, and every byte
present initially is present. -/
def MemOk (d : Inst) (init m : Mem) : Prop :=
  (∀ x ∈ Rset d, m x = init x) ∧ ∀ x, init x ≠ none → m x ≠ none

end Claims.ChunkDeltaH
