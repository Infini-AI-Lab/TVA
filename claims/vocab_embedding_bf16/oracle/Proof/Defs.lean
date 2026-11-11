import Claim
import TVA.Proof

/-!
# `vocab_embedding_bf16`: the objects of the proof

The grid (`prow`, `pcol`), the footprints (`R`, `W`), the per-program result (`Q`), the value
the kernel computes for an id (`laneSel`, `wOff`), and the `int64` expressions of the kernel
(`orgT`, `addT`, `validT`, `localIdT`).
-/

namespace Claims.VocabEmbedding

open TVA

open Block

/-! ## The grid -/

/-- The row of program number `i`. -/
def prow (d : Inst) (i : Nat) : Nat := i % d.n_tokens

/-- The column block of program number `i`. -/
def pcol (d : Inst) (i : Nat) : Nat := i / d.n_tokens % nColBlocks d

/-- The grid has `n_tokens * cdiv(hidden_dim, BLOCK_H)` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.n_tokens * nColBlocks d := by
  simp [Launch.size, launch]

/-- `program_id(0)` of program number `i` is its row. -/
theorem launch_pid0 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = prow d i := by
  simp only [Launch.ctx, launch, prow, Matrix.cons_val_zero]

/-- `program_id(1)` of program number `i` is its column block. -/
theorem launch_pid1 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 1 = pcol d i := by
  simp only [Launch.ctx, launch, pcol, Matrix.cons_val_one, Matrix.cons_val_zero]

/-- `BLOCK_H` is positive. -/
theorem block_pos {d : Inst} (hd : d.ok) : 0 < d.BLOCK_H := by
  obtain ⟨e, he, hB⟩ := hd.block
  rw [hB]; exact Nat.two_pow_pos e

/-- The launch is valid: the grid is within CUDA's limits and the block shape is a power of two. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine ⟨hd.tokens_lt, hd.cols_lt, ?_, ?_, rfl, rfl, rfl, ?_⟩
  · show (1 : Nat) < 2 ^ 16
    norm_num
  · rw [launch_size]; simpa using hd.grid_lt
  · intro s hs
    simp only [launch, List.mem_singleton] at hs
    subst hs
    refine ⟨fun k hk => ?_, ?_⟩
    · simp only [List.mem_singleton] at hk
      subst hk
      obtain ⟨e, -, hB⟩ := hd.block
      exact ⟨e, hB⟩
    · obtain ⟨e, he, hB⟩ := hd.block
      rw [hB]
      calc 2 ^ e * 1 ≤ 2 ^ 10 * 1 := by
            exact Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) he)
        _ ≤ 2 ^ 20 := by norm_num

/-- Program number `i` is its row and its column block. -/
theorem prow_pcol {d : Inst} (i : Fin (launch d).size) :
    (i : Nat) = prow d i + d.n_tokens * pcol d i := by
  have hsz : (i : Nat) < d.n_tokens * nColBlocks d := by
    rw [← launch_size]; exact i.isLt
  have hn : 0 < d.n_tokens := by
    rcases Nat.eq_zero_or_pos d.n_tokens with h | h
    · rw [h, Nat.zero_mul] at hsz; exact absurd hsz (Nat.not_lt_zero _)
    · exact h
  have hsz' : (i : Nat) < nColBlocks d * d.n_tokens := by rw [Nat.mul_comm]; exact hsz
  have hlt : (i : Nat) / d.n_tokens < nColBlocks d := by
    rw [Nat.div_lt_iff_lt_mul hn]; exact hsz'
  show (i : Nat) = (i : Nat) % d.n_tokens + d.n_tokens * ((i : Nat) / d.n_tokens % nColBlocks d)
  rw [Nat.mod_eq_of_lt hlt]
  exact (Nat.mod_add_div (i : Nat) d.n_tokens).symm

/-- The row of a program of the grid is a token. -/
theorem prow_lt {d : Inst} (i : Fin (launch d).size) : prow d i < d.n_tokens := by
  have hsz : (i : Nat) < d.n_tokens * nColBlocks d := by
    rw [← launch_size]; exact i.isLt
  rcases Nat.eq_zero_or_pos d.n_tokens with h | h
  · rw [h, Nat.zero_mul] at hsz; exact absurd hsz (Nat.not_lt_zero _)
  · simp only [prow]; exact Nat.mod_lt _ h

/-- The column block of a program of the grid is a column block. -/
theorem pcol_lt {d : Inst} (i : Fin (launch d).size) : pcol d i < nColBlocks d := by
  have hsz : (i : Nat) < nColBlocks d * d.n_tokens := by
    rw [Nat.mul_comm, ← launch_size]; exact i.isLt
  have hn : 0 < d.n_tokens := by
    rcases Nat.eq_zero_or_pos d.n_tokens with h | h
    · rw [h, Nat.mul_zero] at hsz; exact absurd hsz (Nat.not_lt_zero _)
    · exact h
  have hb : 0 < nColBlocks d := Nat.pos_of_mul_pos_right (lt_of_le_of_lt (Nat.zero_le _) hsz)
  simp only [pcol]
  exact Nat.mod_lt _ hb

/-- The grid's programs are their (row, column block) pairs. -/
theorem prow_pcol_inj {d : Inst} {i j : Fin (launch d).size}
    (hp : prow d i = prow d j) (hq : pcol d i = pcol d j) : i = j := by
  refine Fin.ext ?_
  rw [prow_pcol i, prow_pcol j, hp, hq]

/-! ## Footprints and the per-program result -/

/-- The output elements of column block `q` of row `p`. -/
def outTile (d : Inst) (p q : Nat) : Set Nat :=
  {j | ∃ col, Tile q d.BLOCK_H d.hidden_dim col ∧ j = p * d.hidden_dim + col}

/-- The read footprint of program `(p, q)`: its token and the weight. -/
def R (d : Inst) (_init : Mem) (p : Nat) : Set Int :=
  d.input_.elems {p} ∪ d.weight.bytes

/-- The write footprint of program `(p, q)`: its tile of the output. -/
def W (d : Inst) (p q : Nat) : Set Int := d.output.elems (outTile d p q)

/-- What program `(p, q)` leaves in its own memory `m`: each of its columns holds `outVal` of
the id at element `p` of `input_`. -/
def Q (d : Inst) (init : Mem) (p q : Nat) (m : Mem) : Prop :=
  ∀ l : Fin (d.BLOCK_H - 0), q * d.BLOCK_H + l.val < d.hidden_dim →
    ∃ v, tokAt d init p = some v ∧
      ∃ c : BF16,
        m.read .bf16
            (d.output.ptr ((p * d.hidden_dim + (q * d.BLOCK_H + l.val) : Nat) : Int)).addr = some c ∧
          outVal d init v (q * d.BLOCK_H + l.val) c

/-- The elements of the tile of row `p` are output elements. -/
theorem outTile_subset_outElems {d : Inst} {p q : Nat} (hp : p < d.n_tokens) :
    outTile d p q ⊆ outElems d := by
  rintro j ⟨col, ht, rfl⟩
  show p * d.hidden_dim + col < d.n_tokens * d.hidden_dim
  have h1 : p * d.hidden_dim + col < p * d.hidden_dim + d.hidden_dim :=
    Nat.add_lt_add_left ht.2.2 _
  have h2 : p * d.hidden_dim + d.hidden_dim = (p + 1) * d.hidden_dim := by ring
  have h3 : (p + 1) * d.hidden_dim ≤ d.n_tokens * d.hidden_dim :=
    Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hp)
  omega

/-- The elements of the tile of row `p` are elements of the output buffer. -/
theorem outTile_lt {d : Inst} (hd : d.ok) {p q : Nat} (hp : p < d.n_tokens) {j : Nat}
    (hj : j ∈ outTile d p q) : j < d.output.len :=
  lt_of_lt_of_le (outTile_subset_outElems hp hj) hd.out_len

/-- Every element of a tile is a lane of its block. -/
theorem tile_lane {p B n j : Nat} (h : Tile p B n j) : ∃ l : Fin (B - 0), j = p * B + l := by
  obtain ⟨h1, h2, -⟩ := h
  exact ⟨⟨j - p * B, by simp only [Nat.sub_zero]; omega⟩, by simp only; omega⟩

/-- The tile of column block `q` as the image of the block's lanes. -/
theorem img_outTile (d : Inst) (p q : Nat) :
    Buf.img (s := [d.BLOCK_H - 0])
        (fun i => p * d.hidden_dim + (q * d.BLOCK_H + i.1))
        (fun i => decide (q * d.BLOCK_H + i.1 < d.hidden_dim)) =
      outTile d p q := by
  ext j
  simp only [Buf.img, outTile, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨i, hi, rfl⟩
    exact ⟨q * d.BLOCK_H + i.1,
      ⟨Nat.le_add_right _ _, Nat.add_lt_add_left i.1.2 _, by simpa using hi⟩, rfl⟩
  · rintro ⟨col, ⟨h1, h2, h3⟩, rfl⟩
    obtain ⟨l, hl⟩ := tile_lane ⟨h1, h2, h3⟩
    exact ⟨(l, ()), by simpa [← hl] using h3, by simp [← hl]⟩

/-- The token element of row `p` is an element of `input_`. -/
theorem input_elems_sub_bytes {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.n_tokens) :
    d.input_.elems {p} ⊆ d.input_.bytes := by
  refine Buf.elems_sub_bytes _ (fun j hj => ?_)
  simp only [Set.mem_singleton_iff] at hj
  subst hj
  exact lt_of_lt_of_le hp hd.input_len

/-- Output tiles of distinct (row, column block) pairs are disjoint. -/
theorem tile_disjoint_out {d : Inst} {p p' q q' : Nat} (h : p ≠ p' ∨ q ≠ q') :
    Disjoint (outTile d p q) (outTile d p' q') := by
  refine Set.disjoint_left.mpr fun j hj hj' => ?_
  obtain ⟨col, ht, rfl⟩ := hj
  obtain ⟨col', ht', he⟩ := hj'
  have hc : col < d.hidden_dim := ht.2.2
  have hc' : col' < d.hidden_dim := ht'.2.2
  have hdiv : ∀ (a m : Nat), a < d.hidden_dim →
      (m * d.hidden_dim + a) / d.hidden_dim = m := by
    intro a m ha
    have hh : 0 < d.hidden_dim := by omega
    rw [Nat.mul_comm m d.hidden_dim, Nat.mul_add_div hh, Nat.div_eq_of_lt ha, Nat.add_zero]
  have hp : p = p' := by
    calc p = (p * d.hidden_dim + col) / d.hidden_dim := (hdiv col p hc).symm
      _ = (p' * d.hidden_dim + col') / d.hidden_dim := by rw [he]
      _ = p' := hdiv col' p' hc'
  rcases h with h | h
  · exact h hp
  · have hcol : col = col' := by rw [hp] at he; omega
    have hq : q = q' := by
      have e1 : col / d.BLOCK_H = q :=
        Nat.div_eq_of_lt_le ht.1 (by rw [Nat.succ_mul]; exact ht.2.1)
      have e2 : col' / d.BLOCK_H = q' :=
        Nat.div_eq_of_lt_le ht'.1 (by rw [Nat.succ_mul]; exact ht'.2.1)
      have e3 : col' / d.BLOCK_H = q := by rw [← hcol]; exact e1
      exact e3.symm.trans e2
    exact h hq

/-! ## The value the kernel computes for an id -/

/-- The local row of the id `v`: its row in the original range, else its row in the added range,
else `0` (the kernel's `tl.where` chain). -/
def laneSel (d : Inst) (v : Int) : Int :=
  if d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index then v - d.org_vocab_start_index
  else if d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index then v - addedOffset d
  else 0

/-- The weight element read for the id `v` at column `col`. -/
def wOff (d : Inst) (v : Int) (col : Nat) : Int :=
  laneSel d v * (d.weight_stride0 : Int) + (col : Int)

/-- An id in the original range selects row `v - org_vocab_start_index`. -/
theorem laneSel_of_org {d : Inst} {v : Int}
    (h : d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index) :
    laneSel d v = v - d.org_vocab_start_index := by
  simp only [laneSel]
  rw [ite_eq_left h]

/-- An id in the added range only selects row `v - added_offset`. -/
theorem laneSel_of_add {d : Inst} {v : Int}
    (h : d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index)
    (h' : ¬ (d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index)) :
    laneSel d v = v - addedOffset d := by
  simp only [laneSel]
  rw [ite_eq_right h', ite_eq_left h]

/-- An id in neither range selects row `0`. -/
theorem laneSel_of_none {d : Inst} {v : Int}
    (h : ¬ (d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index))
    (h' : ¬ (d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index)) :
    laneSel d v = 0 := by
  simp only [laneSel]
  rw [ite_eq_right h, ite_eq_right h']

/-- The weight element column `col` is where `outVal` reads it. -/
theorem outVal_at {d : Inst} {init : Mem} {v : Int} {col : Nat} {c : BF16}
    (h : d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index ∨
      d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index) :
    outVal d init v col c ↔
      init.read .bf16 (d.weight.ptr (wOff d v col)).addr = some c := by
  rcases h with h | h
  · rw [outVal, ite_eq_left h, wOff, laneSel_of_org h]
  · by_cases h' : d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index
    · rw [outVal, ite_eq_left h', wOff, laneSel_of_org h']
    · rw [outVal, ite_eq_right h', ite_eq_left h, wOff, laneSel_of_add h h']

/-! ## The `int64` expressions of the kernel -/

/-- The original range mask. -/
def orgT (d : Inst) (c : I64) : Tensor [] Bool :=
  Tensor.and ((Tensor.scalar (IntV.ofInt d.org_vocab_start_index : I64)).le (Tensor.scalar c))
    ((Tensor.scalar c).lt (Tensor.scalar (IntV.ofInt d.org_vocab_end_index : I64)))

/-- The added range mask. -/
def addT (d : Inst) (c : I64) : Tensor [] Bool :=
  Tensor.and ((Tensor.scalar (IntV.ofInt d.added_vocab_start_index : I64)).le (Tensor.scalar c))
    ((Tensor.scalar c).lt (Tensor.scalar (IntV.ofInt d.added_vocab_end_index : I64)))

/-- The mask of the ids the kernel looks up. -/
def validT (d : Inst) (c : I64) : Tensor [] Bool := (orgT d c).or (addT d c)

/-- The local row the kernel computes for the id `c`. -/
def localIdT (d : Inst) (c : I64) : Tensor [] I64 :=
  tl.where (validT d c)
    (tl.where (orgT d c) (Tensor.scalar (c - (IntV.ofInt d.org_vocab_start_index : I64)))
      (Tensor.scalar (c - (IntV.ofInt (addedOffset d) : I64))))
    (Tensor.scalar (IntV.ofInt 0 : I64))

end Claims.VocabEmbedding
