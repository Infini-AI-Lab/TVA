import Claim
import TVA.Proof
import Proof.Basic

/-!
# The geometry of one program

Program `i` handles feature block `iD i`, head `hOf i` and output row `outRow i`: chunk `chk i` of
a sequence that starts at `bos i` of the flat time axis and has `lenS i` positions (on a
variable-length launch, the sequence `seqN i` and chunk named by row `rG i` of `chunk_indices`).
Its load is the 2-D tile of rows `chk·BT + r` (below `lenS`) and columns `iD·BD + l` (below `D`)
of the matrix at element `eX i = (bos·H + h)·D` with strides `H·D` and `1`; its store writes the
columns of output row `outRow i` and head `hOf i`. The footprints `R`, `W` and the postcondition
`Q` are stated with these.
-/

namespace Claims.MeanPoolingFwd

open TVA

/-- Program `i`'s output row `g`. -/
def outRow (d : Inst) (i : Nat) : Nat := if d.varlen then rG d i else bOf d i * ntFix d + rG d i

/-- Program `i`'s sequence (variable length): `chunk_indices[rG, 0]`. -/
def seqN (d : Inst) (i : Nat) : Nat := (seqAt d (rG d i)).toNat

/-- Program `i`'s chunk within its sequence. -/
def chk (d : Inst) (i : Nat) : Nat := if d.varlen then (chkAt d (rG d i)).toNat else rG d i

/-- The first position of program `i`'s sequence on the flat time axis. -/
def bos (d : Inst) (i : Nat) : Nat :=
  if d.varlen then (cuAt d (seqN d i)).toNat else bOf d i * d.T

/-- The length of program `i`'s sequence. -/
def lenS (d : Inst) (i : Nat) : Nat :=
  if d.varlen then (cuAt d (seqN d i + 1)).toNat - (cuAt d (seqN d i)).toNat else d.T

/-- The number of positions of program `i`'s chunk, the divisor `min(BT, T − i_t·BT)`. -/
def Lof (d : Inst) (i : Nat) : Nat := min d.BT (lenS d i - chk d i * d.BT)

/-- The first element of the matrix program `i` loads from: `(bos·H + h)·D`. -/
def eX (d : Inst) (i : Nat) : Nat := (bos d i * d.H + hOf d i) * d.D

/-- The tile's rows: `chk·BT + r`. -/
def rho (d : Inst) (i : Nat) : Fin d.BT → Nat := fun r => chk d i * d.BT + r

/-- The tile's columns: `iD·BD + l`. -/
def kap (d : Inst) (i : Nat) : Fin d.BD → Nat := fun l => iD d i * d.BD + l

/-- The first output element program `i` writes: `(g·H + h)·D`. -/
def eO (d : Inst) (i : Nat) : Nat := (outRow d i * d.H + hOf d i) * d.D

/-- The output elements program `i` writes. -/
def wset (d : Inst) (i : Nat) : Set Nat :=
  {e | ∃ l : Fin d.BD, iD d i * d.BD + l < d.D ∧ e = eO d i + (iD d i * d.BD + l)}

/-- The bytes program `i` reads from the tables (only on a variable-length launch). -/
def extraReads (d : Inst) (i : Nat) : Set Int :=
  if d.varlen then
    Mem.span d.idxD (d.ci.ptr (2 * (rG d i : Nat))).addr ∪
      Mem.span d.idxD (d.ci.ptr (2 * (rG d i : Nat) + 1)).addr ∪
        Mem.span d.idxD (d.cu.ptr (seqN d i)).addr ∪
          Mem.span d.idxD (d.cu.ptr (seqN d i + 1)).addr
  else ∅

/-- Program `i`'s read footprint. -/
def R (d : Inst) (i : Nat) : Set Int :=
  d.x.tileAt (eX d i) (d.H * d.D) 1 (rho d i) (kap d i) (lenS d i) d.D ∪ extraReads d i

/-- Program `i`'s write footprint. -/
def W (d : Inst) (i : Nat) : Set Int := d.o.elems (wset d i)

/-- What program `i` leaves: every column `j` of its block below `D` of output row `outRow i`, head
`hOf i`, holds a value as `Post` says. -/
def Q (d : Inst) (i : Nat) (m : Mem) : Prop :=
  ∀ l : Fin d.BD, iD d i * d.BD + l < d.D → ∃ c : BF16,
    m.read .bf16 (d.o.ptr ((outRow d i * d.H + hOf d i) * d.D + (iD d i * d.BD + l))).addr = some c ∧
      (InRange d (outRow d i) (hOf d i) (iD d i * d.BD + l) → c.isFinite = true ∧
        |c.toRat - mean d (outRow d i) (hOf d i) (iD d i * d.BD + l)| ≤
          tol d.BT (mean d (outRow d i) (hOf d i) (iD d i * d.BD + l))
            (absMean d (outRow d i) (hOf d i) (iD d i * d.BD + l)))

/-! ## The tables -/

/-- `cu_seqlens` is nonnegative through `N`. -/
theorem cuAt_nonneg {d : Inst} (hT : TablesOk d) {n : Nat} (hn : n ≤ d.N) : 0 ≤ cuAt d n := by
  obtain ⟨h0, hmono, -, -⟩ := hT
  induction n with
  | zero => exact h0
  | succ k ih => exact le_trans (ih (by omega)) (hmono k (by omega))

/-- `cu_seqlens` is nondecreasing up to `N`. -/
theorem cuAt_mono_le {d : Inst} (hT : TablesOk d) {a b : Nat} (hab : a ≤ b) (hb : b ≤ d.N) :
    cuAt d a ≤ cuAt d b := by
  induction b with
  | zero => simp_all
  | succ k ih =>
    rcases Nat.eq_or_lt_of_le hab with h | h
    · subst h; exact le_rfl
    · exact le_trans (ih (by omega) (by omega)) (hT.2.1 k (by omega))

/-- The row facts of a variable-length program: its sequence is below `N`, its chunk starts inside
it, and the sequence lies in `[0, T]`. -/
theorem row_facts {d : Inst} (hd : d.ok) (hv : d.varlen = true) (hT : TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) :
    rG d i < d.NT ∧ seqN d i < d.N ∧ 0 ≤ cuAt d (seqN d i) ∧
      cuAt d (seqN d i) ≤ cuAt d (seqN d i + 1) ∧ cuAt d (seqN d i + 1) ≤ d.T ∧
      0 ≤ chkAt d (rG d i) ∧
      chkAt d (rG d i) * d.BT < cuAt d (seqN d i + 1) - cuAt d (seqN d i) := by
  have hr : rG d i < d.NT := by
    have := (rG_lt hd hi).1; simpa [ntG, hv] using this
  have hTN := hT.2.2.1
  obtain ⟨hs0, hsN, hc0, hcl⟩ := hT.2.2.2 (rG d i) hr
  have hn : seqN d i < d.N := by unfold seqN; omega
  have hseq : ((seqAt d (rG d i)).toNat : Int) = seqAt d (rG d i) := Int.toNat_of_nonneg hs0
  refine ⟨hr, hn, cuAt_nonneg hT (by omega), hT.2.1 _ hn,
    le_trans (cuAt_mono_le hT (by omega) le_rfl) hTN, hc0, ?_⟩
  exact hcl

/-! ## The chunk -/

/-- The chunk starts inside its sequence: `chk·BT < lenS`. -/
theorem chk_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) : chk d i * d.BT < lenS d i := by
  cases hv : d.varlen
  · have hr : rG d i < ntFix d := by have := (rG_lt hd hi).1; simpa [ntG, hv] using this
    simp only [chk, lenS, hv, Bool.false_eq_true, ite_false]
    have hb := bt_pos hd
    have h1 : rG d i * d.BT < ntFix d * d.BT := Nat.mul_lt_mul_of_pos_right hr hb
    unfold ntFix at h1 hr
    have h2 : (d.T + d.BT - 1) / d.BT * d.BT ≤ d.T + d.BT - 1 := Nat.div_mul_le_self _ _
    have h3 : rG d i * d.BT + d.BT ≤ (d.T + d.BT - 1) / d.BT * d.BT := by
      have := Nat.mul_le_mul_right d.BT (show rG d i + 1 ≤ (d.T + d.BT - 1) / d.BT by omega)
      rw [Nat.add_mul, Nat.one_mul] at this; exact this
    omega
  · obtain ⟨-, -, h0, hle, -, hc0, hcl⟩ := row_facts hd hv (hT hv) hi
    simp only [chk, lenS, hv, ite_true]
    have e1 : ((cuAt d (seqN d i)).toNat : Int) = cuAt d (seqN d i) := Int.toNat_of_nonneg h0
    have e2 : ((cuAt d (seqN d i + 1)).toNat : Int) = cuAt d (seqN d i + 1) :=
      Int.toNat_of_nonneg (by omega)
    have e3 : ((chkAt d (rG d i)).toNat : Int) = chkAt d (rG d i) := Int.toNat_of_nonneg hc0
    have : (((chkAt d (rG d i)).toNat * d.BT : Nat) : Int) <
        (((cuAt d (seqN d i + 1)).toNat - (cuAt d (seqN d i)).toNat : Nat) : Int) := by
      push_cast
      rw [Nat.cast_sub (by omega), e1, e2, e3]; exact hcl
    exact_mod_cast this

/-- The sequence lies inside the flat time axis: `bos + lenS ≤ B·T`. -/
theorem bos_len_le {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) : bos d i + lenS d i ≤ d.B * d.T := by
  cases hv : d.varlen
  · have hb := (bOf_lt hd hi).1
    simp only [bos, lenS, hv, Bool.false_eq_true, ite_false]
    have := Nat.mul_le_mul_right d.T (show bOf d i + 1 ≤ d.B by omega)
    rw [Nat.add_mul, Nat.one_mul] at this; exact this
  · obtain ⟨-, -, h0, hle, hTe, -, -⟩ := row_facts hd hv (hT hv) hi
    simp only [bos, lenS, hv, ite_true, hd.varlen_B hv, Nat.one_mul]
    omega

/-- `1 ≤ L ≤ BT`. -/
theorem Lof_bounds {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) : 1 ≤ Lof d i ∧ Lof d i ≤ d.BT := by
  have h := chk_lt hd hT hi
  have hb := bt_pos hd
  unfold Lof
  constructor
  · omega
  · exact Nat.min_le_left _ _

/-- A row of the tile is in the sequence exactly when it is below `L`. -/
theorem rho_lt_iff {d : Inst} (i : Nat) (r : Fin d.BT) : rho d i r < lenS d i ↔ (r : Nat) < Lof d i := by
  unfold rho Lof
  have := r.isLt
  omega

/-! ## The output row -/

/-- The output row is below `nRows`. -/
theorem outRow_lt {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) : outRow d i < nRows d := by
  have hr := (rG_lt hd hi).1
  cases hv : d.varlen
  · have hb := (bOf_lt hd hi).1
    simp only [outRow, nRows, ntG, hv, Bool.false_eq_true, ite_false] at hr ⊢
    have := Nat.mul_le_mul_right (ntFix d) (show bOf d i + 1 ≤ d.B by omega)
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  · simp only [outRow, nRows, ntG, hv, ite_true] at hr ⊢
    exact hr

/-- On a fixed-length launch, the output row splits back into the batch row and the chunk. -/
theorem outRow_div_mod {d : Inst} (hd : d.ok) (hv : d.varlen = false) {i : Nat}
    (hi : i < gx d * gy d) : outRow d i / ntFix d = bOf d i ∧ outRow d i % ntFix d = rG d i := by
  have hr : rG d i < ntFix d := by have := (rG_lt hd hi).1; simpa [ntG, hv] using this
  have hn : 0 < ntFix d := by omega
  simp only [outRow, hv, Bool.false_eq_true, ite_false]
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hr, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]

/-- The claim's `start` of program `i`'s output row is `bos + chk·BT`. -/
theorem start_eq {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) :
    start d (outRow d i) = bos d i + chk d i * d.BT := by
  cases hv : d.varlen
  · obtain ⟨h1, h2⟩ := outRow_div_mod hd hv hi
    simp only [start, hv, Bool.false_eq_true, ite_false, h1, h2, bos, chk]
  · simp only [start, outRow, bos, chk, seqN, hv, ite_true]

/-- The claim's `len` of program `i`'s output row is `L`. -/
theorem len_eq {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {i : Nat}
    (hi : i < gx d * gy d) : len d (outRow d i) = Lof d i ∧ stop d (outRow d i) = start d (outRow d i) + Lof d i := by
  have hs := start_eq hd hi
  have hc := chk_lt hd hT hi
  have hstop : stop d (outRow d i) = min (start d (outRow d i) + d.BT) (bos d i + lenS d i) := by
    cases hv : d.varlen
    · obtain ⟨h1, -⟩ := outRow_div_mod hd hv hi
      simp only [stop, hv, Bool.false_eq_true, ite_false, h1, bos, lenS]
    · obtain ⟨-, -, h0, hle, -, -, -⟩ := row_facts hd hv (hT hv) hi
      simp only [stop, hv, ite_true, bos, lenS]
      have : (cuAt d (seqN d i + 1)).toNat - (cuAt d (seqN d i)).toNat + (cuAt d (seqN d i)).toNat =
          (cuAt d (seqN d i + 1)).toNat := by omega
      unfold outRow seqN at *
      simp only [hv, ite_true] at *
      omega
  unfold len Lof
  rw [hstop, hs]
  constructor <;> omega

end Claims.MeanPoolingFwd
