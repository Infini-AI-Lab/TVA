import Claim
import TVA.Proof
import Proof.InvZero

/-!
# The addresses and bounds of the loop's two loads

The `a` and `b` loads of turn `t` read the elements `aIdx`/`bIdx` of their lanes, on the lanes
the masks `maskA`/`maskB` enable. This file turns the invariant's pointer lanes and the mask
values (`Proof/LaneMask.lean`) into the three facts a `safe_load_buf` application needs: the
address map, the element bound, and the read footprint.
-/

open TVA
namespace Claims.SsdBmm

set_option maxHeartbeats 800000

/-! ## The enabled lanes -/

/-- An enabled lane of the `a` load has its row inside the chunk. -/
theorem enA_row {d : Inst} {i : Fin (launch d).size} {t : Nat} {mll kk : Nat}
    (h : enA d i t mll kk = true) : mOf d i * BM + mll < chunkNat d i := by
  unfold enA at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1

/-- An enabled lane of the `a` load has its `k` index below `K`. -/
theorem enA_k {d : Inst} {i : Fin (launch d).size} {t : Nat} {mll kk : Nat}
    (h : enA d i t mll kk = true) : t * BK + kk < d.K := by
  unfold enA at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.2

/-- An enabled lane of the `b` load has its `k` index below `K`. -/
theorem enB_k {d : Inst} {i : Fin (launch d).size} {t : Nat} {kk nn : Nat}
    (h : enB d i t kk nn = true) : t * BK + kk < d.K := by
  unfold enB at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1

/-- An enabled lane of the `b` load has its column inside the chunk. -/
theorem enB_col {d : Inst} {i : Fin (launch d).size} {t : Nat} {kk nn : Nat}
    (h : enB d i t kk nn = true) : nOf d i * BN + nn < chunkNat d i := by
  unfold enB at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.2

/-- An enabled row of the `seq_idx` load is inside the chunk. -/
theorem enM_row {d : Inst} {i : Fin (launch d).size} {mll : Nat}
    (h : enM d i mll = true) : mOf d i * BM + mll < chunkNat d i := by
  unfold enM at h
  exact of_decide_eq_true h

/-- An enabled column of the `seq_idx` load is inside the chunk. -/
theorem enN_col {d : Inst} {i : Fin (launch d).size} {nn : Nat}
    (h : enN d i nn = true) : nOf d i * BN + nn < chunkNat d i := by
  unfold enN at h
  exact of_decide_eq_true h

/-- An enabled lane of the store has its row below `chunk_size`. -/
theorem enS_row {d : Inst} {i : Fin (launch d).size} {mll nn : Nat}
    (h : enS d i mll nn = true) : mOf d i * BM + mll < d.C := by
  unfold enS at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1

/-- An enabled lane of the store has its column below `chunk_size`. -/
theorem enS_col {d : Inst} {i : Fin (launch d).size} {mll nn : Nat}
    (h : enS d i mll nn = true) : nOf d i * BN + nn < d.C := by
  unfold enS at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.2

/-! ## The rows and columns of an enabled lane are inside the batch -/

/-- A row inside the chunk is a token of the sequence. -/
theorem rowA_lt_of_chunk {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {mll : Nat}
    (h : mOf d i * BM + mll < chunkNat d i) : rowA d i mll < d.T := by
  have h1 := cOf_mul_C_lt hd i
  have h2 : mOf d i * BM + mll < d.T - cOf d i * d.C :=
    lt_of_lt_of_le h (min_le_right _ _)
  unfold rowA
  omega

/-- A column inside the chunk is a token of the sequence. -/
theorem colB_lt_of_chunk {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {nn : Nat}
    (h : nOf d i * BN + nn < chunkNat d i) : colB d i nn < d.T := by
  have h1 := cOf_mul_C_lt hd i
  have h2 : nOf d i * BN + nn < d.T - cOf d i * d.C :=
    lt_of_lt_of_le h (min_le_right _ _)
  unfold colB
  omega

/-! ## The two loads' address maps -/

/-- The lane function of the `a` load of turn `t`. -/
def gA (d : Inst) (i : Fin (launch d).size) (t : Nat) : Idx [BM, BK] → Nat :=
  fun idx => aIdx d (rowA d i idx.1.val) (hOf d i) (t * BK + idx.2.1.val)

/-- The lane function of the `b` load of turn `t`. -/
def gB (d : Inst) (i : Fin (launch d).size) (t : Nat) : Idx [BK, BN] → Nat :=
  fun idx => bIdx d (colB d i idx.2.1.val) (hOf d i) (t * BK + idx.1.val)

/-- The lanes of the `a` load address elements of `a`: lane `(mll, kk)` is element `gA` when enabled. -/
theorem aLoad_addrs {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (t : Nat)
    (ht : t < steps d.K) {st : StT} {m init : Mem} (hinv : inv d init i t st m) :
    d.a.Addrs (tl.addrs st.1 (some (maskA d i t))) (gA d i t)
      (fun idx => enA d i t idx.1.val idx.2.1.val) := by
  refine Buf.addrs_of_get d.a (fun idx => ?_) ?_
  · exact maskA_get hd i t ht idx.1 idx.2.1
  · rintro ⟨mll, kk, u⟩ hen
    cases u
    exact hinv.2.1 mll kk (lt_of_lt_of_le (enA_row hen) (min_le_left _ _))

/-- The lanes of the `b` load address elements of `b`. -/
theorem bLoad_addrs {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (t : Nat)
    (ht : t < steps d.K) {st : StT} {m init : Mem} (hinv : inv d init i t st m) :
    d.b.Addrs (tl.addrs st.2.1 (some (maskB d i t))) (gB d i t)
      (fun idx => enB d i t idx.1.val idx.2.1.val) := by
  refine Buf.addrs_of_get d.b (fun idx => ?_) ?_
  · exact maskB_get hd i t ht idx.1 idx.2.1
  · rintro ⟨kk, nn, u⟩ hen
    cases u
    exact hinv.2.2.1 kk nn (lt_of_lt_of_le (enB_col hen) (min_le_left _ _))

/-! ## The loads stay inside their buffers -/

/-- Every enabled lane of the `a` load is inside `a`'s storage. -/
theorem aLoad_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (t : Nat) (idx : Idx [BM, BK])
    (h : enA d i t idx.1.val idx.2.1.val = true) : gA d i t idx < d.a.len :=
  hd.a_len _ _ _ (rowA_lt_of_chunk hd i (enA_row h)) (hOf_lt hd i) (enA_k h)

/-- Every enabled lane of the `b` load is inside `b`'s storage. -/
theorem bLoad_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (t : Nat) (idx : Idx [BK, BN])
    (h : enB d i t idx.1.val idx.2.1.val = true) : gB d i t idx < d.b.len :=
  hd.b_len _ _ _ (colB_lt_of_chunk hd i (enB_col h)) (hOf_lt hd i) (enB_k h)

/-! ## The loads read only from the read footprint -/

/-- The bytes the `a` load reads are in the program's read footprint. -/
theorem aLoad_R {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (t : Nat) {init : Mem} :
    d.a.elems (Buf.img (gA d i t) (fun idx => enA d i t idx.1.val idx.2.1.val)) ⊆ R d init i := by
  refine le_trans (Buf.elems_sub_bytes d.a ?_) ?_
  · rintro j ⟨idx, hen, rfl⟩
    exact aLoad_lt hd i t idx hen
  · intro x hx
    exact Or.inl (Or.inl hx)

/-- The bytes the `b` load reads are in the program's read footprint. -/
theorem bLoad_R {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (t : Nat) {init : Mem} :
    d.b.elems (Buf.img (gB d i t) (fun idx => enB d i t idx.1.val idx.2.1.val)) ⊆ R d init i := by
  refine le_trans (Buf.elems_sub_bytes d.b ?_) ?_
  · rintro j ⟨idx, hen, rfl⟩
    exact bLoad_lt hd i t idx hen
  · intro x hx
    exact Or.inl (Or.inr hx)

end Claims.SsdBmm



