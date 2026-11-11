import Proof.Defs
import Proof.Lanes

/-!
# Footprints and the final memory

The programs' write footprints are pairwise disjoint and disjoint from every read footprint, and
the results of the independent programs give `Post`.
-/

namespace Claims.VocabEmbedding

open TVA

open Block

/-- The row of program number `r + n_tokens * q` is `r`. -/
theorem prow_mul_add {d : Inst} {r q : Nat} (hr : r < d.n_tokens) :
    prow d (r + d.n_tokens * q) = r := by
  show (r + d.n_tokens * q) % d.n_tokens = r
  rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hr]

/-- The column block of program number `r + n_tokens * q` is `q`. -/
theorem pcol_mul_add {d : Inst} {r q : Nat} (hr : r < d.n_tokens) (hq : q < nColBlocks d) :
    pcol d (r + d.n_tokens * q) = q := by
  have hn : 0 < d.n_tokens := by omega
  show (r + d.n_tokens * q) / d.n_tokens % nColBlocks d = q
  rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt hr, Nat.zero_add, Nat.mod_eq_of_lt hq]

/-- A column of the output lies in a column block of the grid. -/
theorem div_lt_nColBlocks {d : Inst} (hd : d.ok) {col : Nat} (hc : col < d.hidden_dim) :
    col / d.BLOCK_H < nColBlocks d := by
  have hB : 0 < d.BLOCK_H := block_pos hd
  have h1 : (d.hidden_dim + d.BLOCK_H - 1) / d.BLOCK_H * d.BLOCK_H ≤
      d.hidden_dim + d.BLOCK_H - 1 := Nat.div_mul_le_self _ _
  have h2 : d.hidden_dim + d.BLOCK_H - 1 <
      (d.hidden_dim + d.BLOCK_H - 1) / d.BLOCK_H * d.BLOCK_H + d.BLOCK_H :=
    Nat.lt_div_mul_add hB
  have hx : d.hidden_dim ≤ (d.hidden_dim + d.BLOCK_H - 1) / d.BLOCK_H * d.BLOCK_H := by
    omega
  show col / d.BLOCK_H < (d.hidden_dim + d.BLOCK_H - 1) / d.BLOCK_H
  rw [Nat.div_lt_iff_lt_mul hB]
  omega

/-- A program's write footprint is disjoint from any program's read footprint. -/
theorem disjoint_write_read {d : Inst} (hd : d.ok) {init : Mem}
    (i j : Fin (launch d).size) :
    Disjoint (W d (prow d i) (pcol d i)) (R d init (prow d j)) := by
  have hbytes : Disjoint d.output.bytes (d.input_.bytes ∪ d.weight.bytes) :=
    Disjoint.union_right hd.out_input hd.out_weight
  refine Disjoint.mono ?_ ?_ hbytes
  · exact Buf.elems_sub_bytes d.output (fun k hk => outTile_lt hd (prow_lt i) hk)
  · show d.input_.elems {prow d j} ∪ d.weight.bytes ⊆
      d.input_.bytes ∪ d.weight.bytes
    exact Set.union_subset
      ((input_elems_sub_bytes hd (prow_lt j)).trans Set.subset_union_left)
      Set.subset_union_right

/-- A program's write footprint is disjoint from its own read footprint. -/
theorem disjoint_read_write {d : Inst} (hd : d.ok) {init : Mem}
    (i : Fin (launch d).size) :
    Disjoint (R d init (prow d i)) (W d (prow d i) (pcol d i)) :=
  (disjoint_write_read hd i i).symm

/-- The write footprint of a program is disjoint from every other program's reads and writes. -/
theorem disjoint_foot {d : Inst} (hd : d.ok) {init : Mem}
    (i j : Fin (launch d).size) (hij : i ≠ j) :
    Disjoint (W d (prow d i) (pcol d i))
      (R d init (prow d j) ∪ W d (prow d j) (pcol d j)) := by
  have hne : prow d i ≠ prow d j ∨ pcol d i ≠ pcol d j := by
    by_contra h
    push Not at h
    exact hij (prow_pcol_inj h.1 h.2)
  have hWW : Disjoint (W d (prow d i) (pcol d i)) (W d (prow d j) (pcol d j)) :=
    Buf.elems_disjoint d.output (tile_disjoint_out hne)
  exact Disjoint.union_right (disjoint_write_read hd i j) hWW

/-- The independent programs' results give `Post`. -/
theorem post_of_local {d : Inst} (hd : d.ok) (fin init : Mem)
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Q d init (prow d i) (pcol d i) m ∧
      ∀ x ∈ R d init (prow d i) ∪ W d (prow d i) (pcol d i), fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d (prow d i) (pcol d i)) → fin x = init x) :
    Post d init fin := by
  constructor
  · intro r col hr hc
    have hB : 0 < d.BLOCK_H := block_pos hd
    have hq : col / d.BLOCK_H < nColBlocks d := div_lt_nColBlocks hd hc
    obtain ⟨l, hl⟩ := tile_lane (tile_div hB hc)
    have hjt : r + d.n_tokens * (col / d.BLOCK_H) < (launch d).size := by
      rw [launch_size]
      have h2 : d.n_tokens * (col / d.BLOCK_H + 1) ≤ d.n_tokens * nColBlocks d :=
        Nat.mul_le_mul_left _ hq
      rw [Nat.mul_add] at h2
      omega
    let j : Fin (launch d).size := ⟨r + d.n_tokens * (col / d.BLOCK_H), hjt⟩
    have hpi : prow d j = r := prow_mul_add hr
    have hqi : pcol d j = col / d.BLOCK_H := pcol_mul_add hr hq
    obtain ⟨m, hm, hagree⟩ := hloc j
    have hlcol : pcol d j * d.BLOCK_H + l.val < d.hidden_dim := by
      rw [hqi, ← hl]
      exact hc
    obtain ⟨v, htok, c, hread, hout⟩ := hm l hlcol
    have hcolp : pcol d j * d.BLOCK_H + l.val = col := by
      rw [hqi, ← hl]
    have hmem : prow d j * d.hidden_dim + (pcol d j * d.BLOCK_H + l.val) ∈
        outTile d (prow d j) (pcol d j) :=
      ⟨pcol d j * d.BLOCK_H + l.val, ⟨Nat.le_add_right _ _, Nat.add_lt_add_left l.2 _, hlcol⟩, rfl⟩
    have hcongr : fin.read .bf16
        (d.output.ptr ((prow d j * d.hidden_dim + (pcol d j * d.BLOCK_H + l.val) : Nat) : Int)).addr
        = m.read .bf16
        (d.output.ptr ((prow d j * d.hidden_dim + (pcol d j * d.BLOCK_H + l.val) : Nat) : Int)).addr :=
      Mem.read_congr (fun x hx => hagree x (Or.inr
        (Buf.span_sub_elems d.output (S := outTile d (prow d j) (pcol d j)) hmem hx)))
    have hidx : r * d.hidden_dim + col =
        prow d j * d.hidden_dim + (pcol d j * d.BLOCK_H + l.val) := by
      rw [hcolp, hpi]
    refine ⟨v, ?_, c, ?_, ?_⟩
    · rw [← hpi]
      exact htok
    · rw [hidx]
      exact hcongr.trans hread
    · rw [← hcolp]
      exact hout
  · intro z hz
    exact hfr z (fun i hzW =>
      hz (Buf.elems_mono d.output (outTile_subset_outElems (prow_lt i)) hzW))

end Claims.VocabEmbedding
