import Claim
import TVA.Proof
import Proof.Geo
import Proof.GeoBound

/-!
# The footprints of the programs are disjoint

The element index `elem d i l` determines the program `i` and the lane `l`: it gives the head
(`% H`) and the flat position (`/ H`), the position gives the sequence (the sequences' position
intervals are disjoint), then the chunk, then the (batch row, head) pair — and in a variable-length
launch the batch row is `0` because `B = 1`. So two programs write different elements of `o`.

A program's reads are from `s` (and, on a variable-length launch, the int tables), which share no
byte with the fresh `o`.
-/

namespace Claims.ChunkCumsum

open TVA

/-! ## The element index determines the program -/

/-- Two lanes with the same element have the same head. -/
theorem headOf_eq_of_elem_eq {d : Inst} (hd : d.ok) {i j l l' : Nat}
    (h : elem d i l = elem d j l') : headOf d i = headOf d j := by
  rw [← elem_mod hd i l, ← elem_mod hd j l', h]

/-- Two lanes with the same element have the same flat position. -/
theorem pos_eq_of_elem_eq {d : Inst} (hd : d.ok) {i j l l' : Nat}
    (h : elem d i l = elem d j l') : pos d i l = pos d j l' := by
  rw [← elem_div hd i l, ← elem_div hd j l', h]

/-- `i` is its (row, batch row, head) triple. -/
theorem row_BH_add {d : Inst} (i : Nat) :
    row d i * BH d + batchOf d i * d.H + headOf d i = i := by
  unfold row batchOf headOf lin BH
  have h1 := Nat.div_add_mod i (d.B * d.H)
  have h2 := Nat.div_add_mod (i % (d.B * d.H)) d.H
  calc (i / (d.B * d.H)) * (d.B * d.H) + (i % (d.B * d.H) / d.H) * d.H
        + (i % (d.B * d.H)) % d.H
      = (d.B * d.H) * (i / (d.B * d.H))
          + (d.H * (i % (d.B * d.H) / d.H) + (i % (d.B * d.H)) % d.H) := by ring
    _ = (d.B * d.H) * (i / (d.B * d.H)) + i % (d.B * d.H) := by rw [h2]
    _ = i := h1

/-- On a variable-length launch the batch row is `0`. -/
theorem batchOf_eq_zero {d : Inst} (hd : d.ok) (h : d.varlen = true) (i : Nat) :
    batchOf d i = 0 := by
  have hb := batchOf_lt hd i
  have h1 := hd.varlen_B h
  omega

/-- The sequence of a program is determined by its lane's flat position. -/
theorem nOf_eq_of_pos_eq {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i j l l' : Nat} (hi : i < gridSize d) (hj : j < gridSize d)
    (hei : en d i l = true) (hej : en d j l' = true) (h : pos d i l = pos d j l') :
    nOf d i = nOf d j := by
  have hni : nOf d i < nSeq d := nOf_lt hd hT hi
  have hnj : nOf d j < nSeq d := nOf_lt hd hT hj
  have hli : lo d (nOf d i) ≤ pos d i l := by unfold pos baseOf; omega
  have hlj : lo d (nOf d j) ≤ pos d j l' := by unfold pos baseOf; omega
  have hhi : pos d i l < Claims.ChunkCumsum.hi d (nOf d i) := pos_lt_hi hT hni hei
  have hhj : pos d j l' < Claims.ChunkCumsum.hi d (nOf d j) := pos_lt_hi hT hnj hej
  cases hvv : d.varlen
  · simp only [lo_false hvv, hi_false hvv, nOf_false hvv] at hli hhi hlj hhj ⊢
    by_contra hne
    rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
    · have h1 : (batchOf d i + 1) * d.T ≤ batchOf d j * d.T :=
        Nat.mul_le_mul_right _ (by omega)
      have h2 : (batchOf d i + 1) * d.T = batchOf d i * d.T + d.T := by ring
      omega
    · have h1 : (batchOf d j + 1) * d.T ≤ batchOf d i * d.T :=
        Nat.mul_le_mul_right _ (by omega)
      have h2 : (batchOf d j + 1) * d.T = batchOf d j * d.T + d.T := by ring
      omega
  · by_contra hne
    have hNi : nOf d i < d.N := by simpa [nSeq_true hvv] using hni
    have hNj : nOf d j < d.N := by simpa [nSeq_true hvv] using hnj
    rw [hi_true hvv (nOf d i)] at hhi
    rw [hi_true hvv (nOf d j)] at hhj
    rw [lo_true hvv (nOf d i)] at hli
    rw [lo_true hvv (nOf d j)] at hlj
    rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
    · have hm : nOf d i + 1 ≤ nOf d j := by omega
      have hmono := cuAt_mono_le (hT hvv) hm (by omega : nOf d j ≤ d.N)
      have hle : (cuAt d (nOf d i + 1)).toNat ≤ (cuAt d (nOf d j)).toNat :=
        Int.toNat_le_toNat hmono
      omega
    · have hm : nOf d j + 1 ≤ nOf d i := by omega
      have hmono := cuAt_mono_le (hT hvv) hm (by omega : nOf d i ≤ d.N)
      have hle : (cuAt d (nOf d j + 1)).toNat ≤ (cuAt d (nOf d i)).toNat :=
        Int.toNat_le_toNat hmono
      omega

/-- The chunk of a program is determined by its lane's flat position and its sequence, provided
the lanes are inside the block (`l, l' < d.BT`). -/
theorem chkOf_eq_of_pos_eq_of_lt {d : Inst} (hd : d.ok) {i j l l' : Nat} (hn : nOf d i = nOf d j)
    (h : pos d i l = pos d j l') (hl : l < d.BT) (hl' : l' < d.BT) :
    chkOf d i = chkOf d j := by
  have hBT : 0 < d.BT := by
    obtain ⟨e, he⟩ := hd.bt_pow2
    rw [he]; positivity
  have h1 : chkOf d i * d.BT + l = chkOf d j * d.BT + l' := by
    unfold pos baseOf at h
    rw [hn] at h
    omega
  have h2 : (chkOf d i * d.BT + l) / d.BT = chkOf d i := by
    rw [Nat.mul_comm, Nat.mul_add_div hBT, Nat.div_eq_of_lt hl, Nat.add_zero]
  have h3 : (chkOf d j * d.BT + l') / d.BT = chkOf d j := by
    rw [Nat.mul_comm, Nat.mul_add_div hBT, Nat.div_eq_of_lt hl', Nat.add_zero]
  rw [h1, h3] at h2
  exact h2.symm

/-- Two enabled lanes touching the same element belong to the same program (they need not be the
same lane: `elem_unique` is false as stated). This is the form used by `img_disjoint`. -/
private theorem same_prog {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i j l l' : Nat} (hi : i < gridSize d) (hj : j < gridSize d)
    (hei : en d i l = true) (hej : en d j l' = true) (hl : l < d.BT) (hl' : l' < d.BT)
    (h : elem d i l = elem d j l') : i = j := by
  have hpos : pos d i l = pos d j l' := pos_eq_of_elem_eq hd h
  have hn : nOf d i = nOf d j := nOf_eq_of_pos_eq hd hT hi hj hei hej hpos
  have hrow : row d i = row d j := by
    have hc : chkOf d i = chkOf d j :=
      chkOf_eq_of_pos_eq_of_lt hd hn hpos hl hl'
    cases hvv : d.varlen
    · rw [chkOf_false hvv i] at hc
      rw [chkOf_false hvv j] at hc
      exact hc
    · have hTab := hT hvv
      have hri : row d i < d.NT := by
        have := row_lt hd hi; simpa [rowsOf, hvv] using this
      have hrj : row d j < d.NT := by
        have := row_lt hd hj; simpa [rowsOf, hvv] using this
      obtain ⟨-, -, -, hseqInfo, hinj, -⟩ := hTab
      have hni : 0 ≤ seqAt d (row d i) := (hseqInfo (row d i) hri).1
      have hnj : 0 ≤ seqAt d (row d j) := (hseqInfo (row d j) hrj).1
      have hci : 0 ≤ chkAt d (row d i) := (hseqInfo (row d i) hri).2.2.1
      have hcj : 0 ≤ chkAt d (row d j) := (hseqInfo (row d j) hrj).2.2.1
      have hn1 : nOf d i = (seqAt d (row d i)).toNat := nOf_true hvv i
      have hn2 : nOf d j = (seqAt d (row d j)).toNat := nOf_true hvv j
      have hc1 : chkOf d i = (chkAt d (row d i)).toNat := chkOf_true hvv i
      have hc2 : chkOf d j = (chkAt d (row d j)).toNat := chkOf_true hvv j
      have hseqeq : seqAt d (row d i) = seqAt d (row d j) := by
        have hnat : (seqAt d (row d i)).toNat = (seqAt d (row d j)).toNat := by
          rw [← hn1, ← hn2]; exact hn
        have hcast : ((seqAt d (row d i)).toNat : Int)
            = ((seqAt d (row d j)).toNat : Int) := by rw [hnat]
        rwa [Int.toNat_of_nonneg hni, Int.toNat_of_nonneg hnj] at hcast
      have hchkeq : chkAt d (row d i) = chkAt d (row d j) := by
        have hnat : (chkAt d (row d i)).toNat = (chkAt d (row d j)).toNat := by
          rw [← hc1, ← hc2]; exact hc
        have hcast : ((chkAt d (row d i)).toNat : Int)
            = ((chkAt d (row d j)).toNat : Int) := by rw [hnat]
        rwa [Int.toNat_of_nonneg hci, Int.toNat_of_nonneg hcj] at hcast
      by_contra hne
      exact hinj (row d i) (row d j) hri hrj hne (by rw [hseqeq, hchkeq])
  have hbatch : batchOf d i = batchOf d j := by
    cases hvv : d.varlen
    · rw [nOf_false hvv i] at hn
      rw [nOf_false hvv j] at hn
      exact hn
    · rw [batchOf_eq_zero hd hvv i, batchOf_eq_zero hd hvv j]
  have hhead : headOf d i = headOf d j := headOf_eq_of_elem_eq hd h
  have h1 : row d i * BH d + batchOf d i * d.H + headOf d i = i := row_BH_add i
  have h2 : row d j * BH d + batchOf d j * d.H + headOf d j = j := row_BH_add j
  rw [hrow, hbatch, hhead] at h1
  omega

/-- Two enabled lanes of the block touching the same element are the same lane of the same
program. -/
theorem elem_unique_of_lt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i j l l' : Nat} (hi : i < gridSize d) (hj : j < gridSize d)
    (hei : en d i l = true) (hej : en d j l' = true) (hl : l < d.BT) (hl' : l' < d.BT)
    (h : elem d i l = elem d j l') : i = j ∧ l = l' := by
  have hij : i = j := same_prog hd hT hi hj hei hej hl hl' h
  refine ⟨hij, ?_⟩
  rw [hij] at h
  exact elem_inj hd h

/-! ## Disjointness of the element sets -/

/-- The elements enabled lanes of distinct programs touch are distinct. -/
theorem img_disjoint {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i j : Nat} (hi : i < gridSize d) (hj : j < gridSize d) (hij : i ≠ j) :
    Disjoint (Buf.img (elemF d i) (enF d i)) (Buf.img (elemF d j) (enF d j)) := by
  rw [Set.disjoint_left]
  rintro x hx hy
  unfold Buf.img elemF enF at hx hy
  obtain ⟨a, hae, rfl⟩ := hx
  obtain ⟨b, hbe, hba⟩ := hy
  exact hij (same_prog hd hT hi hj hae hbe a.1.isLt b.1.isLt hba.symm)

/-- The written elements of a program are inside the sequence's positions. -/
theorem W_subset_written {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) : W d i ⊆ d.o.elems (written d) := by
  have hsub : d.o.elems (Buf.img (elemF d i) (enF d i)) ⊆ d.o.elems (written d) := by
    refine Buf.elems_mono d.o ?_
    intro j hj
    unfold Buf.img elemF enF at hj
    obtain ⟨a, hae, rfl⟩ := hj
    simp only [written, Set.mem_ofPred_eq]
    exact ⟨nOf d i, pos d i a.1.val, headOf d i, nOf_lt hd hT hi,
      by unfold pos baseOf; omega, pos_lt_hi hT (nOf_lt hd hT hi) hae,
      headOf_lt d hd i, rfl⟩
  intro x hx
  unfold W at hx
  exact hsub hx

/-- A program's write footprint lies in the bytes of `o`. -/
theorem W_subset_oBytes {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) : W d i ⊆ d.o.bytes := by
  intro x hx
  unfold W at hx
  obtain ⟨j, hj, hxj⟩ := hx
  unfold Buf.img elemF enF at hj
  obtain ⟨a, hae, rfl⟩ := hj
  refine Buf.span_sub d.o ?_ hxj
  have h1 := elem_lt hd hT (nOf_lt hd hT hi) hae
  have h2 := hd.o_len
  omega

/-- A program's reads are in `s` (and the int tables). -/
theorem R_subset_read {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) :
    R d i ⊆ d.s.bytes ∪ d.cu.bytes ∪ d.ci.bytes := by
  have hsel : d.s.elems (Buf.img (elemF d i) (enF d i)) ⊆ d.s.bytes := by
    refine Buf.elems_sub_bytes d.s ?_
    intro j hj
    unfold Buf.img elemF enF at hj
    obtain ⟨a, hae, rfl⟩ := hj
    have h1 := elem_lt hd hT (nOf_lt hd hT hi) hae
    have h2 := hd.s_len
    omega
  unfold R extraReads
  split
  · rename_i hv
    have hri : row d i < d.NT := by
      have := row_lt hd hi; simpa [rowsOf, hv] using this
    have hci := hd.ci_len hv
    have hcu := hd.cu_len hv
    have hnN : nOf d i < d.N := by
      have := nOf_lt hd hT hi; simpa [nSeq_true hv] using this
    have hnOf : nOf d i = (seqAt d (row d i)).toNat := nOf_true hv i
    have hci1 : 2 * row d i < d.ci.len := by omega
    have hci2 : 2 * row d i + 1 < d.ci.len := by omega
    have hcu1 : (seqAt d (row d i)).toNat < d.cu.len := by omega
    have hcu2 : (seqAt d (row d i)).toNat + 1 < d.cu.len := by omega
    intro x hx
    simp only [Set.mem_union] at hx
    rcases hx with hx | hx
    · exact Or.inl (Or.inl (hsel hx))
    · rcases hx with ((hx | hx) | hx) | hx
      · exact Or.inr (Buf.span_sub d.ci hci1 hx)
      · exact Or.inr (Buf.span_sub d.ci hci2 hx)
      · exact Or.inl (Or.inr (Buf.span_sub d.cu hcu1 hx))
      · exact Or.inl (Or.inr (Buf.span_sub d.cu hcu2 hx))
  · rename_i hv
    intro x hx
    rcases hx with hx | hx
    · exact Or.inl (Or.inl (hsel hx))
    · simp at hx

/-! ## The three disjointness facts -/

/-- Distinct programs write disjoint bytes. -/
theorem W_disjoint {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i j : Nat} (hi : i < gridSize d) (hj : j < gridSize d) (hij : i ≠ j) :
    Disjoint (W d i) (W d j) := by
  have himg := img_disjoint hd hT hi hj hij
  unfold W
  exact Buf.elems_disjoint d.o himg

/-- No program writes a byte any program reads: `o` shares no byte with `s` or the tables. -/
theorem W_disjoint_R {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i j : Nat} (hi : i < gridSize d) (hj : j < gridSize d) :
    Disjoint (W d i) (R d j) := by
  have hW : W d i ⊆ d.o.bytes := W_subset_oBytes hd hT hi
  cases hvv : d.varlen
  · have hR : R d j ⊆ d.s.bytes := by
      intro x hx
      unfold R at hx
      rcases hx with hx | hx
      · refine Buf.elems_sub_bytes d.s ?_ hx
        intro k hk
        unfold Buf.img elemF enF at hk
        obtain ⟨a, hae, rfl⟩ := hk
        have h1 := elem_lt hd hT (nOf_lt hd hT hj) hae
        have h2 := hd.s_len
        omega
      · simp [extraReads, hvv] at hx
    exact Disjoint.mono hW hR hd.o_s
  · have hR : R d j ⊆ d.s.bytes ∪ d.cu.bytes ∪ d.ci.bytes := R_subset_read hd hT hj
    have hd1 : Disjoint (d.s.bytes ∪ d.cu.bytes ∪ d.ci.bytes) d.o.bytes :=
      ((hd.o_s.symm.union_left (hd.o_cu hvv).symm).union_left (hd.o_ci hvv).symm)
    exact Disjoint.mono hW hR hd1.symm

/-- No program reads a byte it writes. -/
theorem R_disjoint_W {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) : Disjoint (R d i) (W d i) := by
  have hW : W d i ⊆ d.o.bytes := W_subset_oBytes hd hT hi
  cases hvv : d.varlen
  · have hR : R d i ⊆ d.s.bytes := by
      intro x hx
      unfold R at hx
      rcases hx with hx | hx
      · refine Buf.elems_sub_bytes d.s ?_ hx
        intro k hk
        unfold Buf.img elemF enF at hk
        obtain ⟨a, hae, rfl⟩ := hk
        have h1 := elem_lt hd hT (nOf_lt hd hT hi) hae
        have h2 := hd.s_len
        omega
      · simp [extraReads, hvv] at hx
    exact Disjoint.mono hR hW hd.o_s.symm
  · have hR : R d i ⊆ d.s.bytes ∪ d.cu.bytes ∪ d.ci.bytes := R_subset_read hd hT hi
    have hd1 : Disjoint (d.s.bytes ∪ d.cu.bytes ∪ d.ci.bytes) d.o.bytes :=
      ((hd.o_s.symm.union_left (hd.o_cu hvv).symm).union_left (hd.o_ci hvv).symm)
    exact Disjoint.mono hR hW hd1

/-- A program's writes are disjoint from what any other program reads or writes. -/
theorem hdisj_all {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i j : Nat} (hi : i < gridSize d) (hj : j < gridSize d) (hij : i ≠ j) :
    Disjoint (W d i) (R d j ∪ W d j) := by
  exact Disjoint.union_right (W_disjoint_R hd hT hi hj) (W_disjoint hd hT hi hj hij)

end Claims.ChunkCumsum
