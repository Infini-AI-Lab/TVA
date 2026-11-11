import Proof.Defs
import Proof.Lanes
import Proof.Ptr
import Proof.Zero
import Proof.Spec

/-!
# `vocab_embedding_bf16`: proof

Program `(row, col_block)` reads the id at element `row` of `input_` and the weight, and writes
its tile of `BLOCK_H` columns of output row `row`. The proof:

1. `Defs`: the grid (`prow`, `pcol`), the footprints (`R`, `W`) and the result `Q` of a program;
2. `Lanes`: the `int64` expressions of the kernel (range masks, local row times stride) do not
   wrap and equal the claim's integers;
3. `Ptr`: the pointer of each lane of the loads and the store is the element the claim names;
4. `Zero`: the masked-off value `other` is the `bfloat16` `+0`;
5. `safe_program` (here): one program, walked line by line with the rules of `TVA.Proof.Rules`;
6. `Spec`: footprints of distinct programs do not conflict, and the programs' results give `Post`;
7. `main` (here): `correct_of_safe` ties 5 and 6 together.
-/

namespace Claims.VocabEmbedding

open TVA

open Block

/-! ## The body, with the block pattern -/

/-- The kernel's body, line by line, with the block pattern `Block.offs`/`Block.mask` and the
`int64` expressions of `localIdT`/`validT`. -/
def bodyScript (d : Inst) (p q : Nat) : Prog Unit := do
  let row := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat p : I32))).item
  let cols := Block.offs q d.BLOCK_H
  let col_mask := Block.mask q d.BLOCK_H d.hidden_dim
  let t0 ← tl.load1 (d.input_.ptr 0 + row)
  let token := (← tl.convert d.dI .i64 (Tensor.scalar t0)).item
  let other ← tl.convert .f32 .bf16 (tl.full [d.BLOCK_H] (FloatV.litF32 (FloatV.ofPat 0)))
  let vals ← tl.load
    (d.weight.ptr 0 + (localIdT d token).item * (IntV.ofNat d.weight_stride0 : I64) + cols)
    (some (Tensor.and col_mask (bc (validT d token).item))) (some other)
  tl.store (d.output.ptr 0 + row * (IntV.ofNat d.hidden_dim : I64) + cols) vals (some col_mask)

set_option maxHeartbeats 1000000 in
/-- The model's body is the script above (by definition). -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c = bodyScript d (c.pid 0) (c.pid 1) := rfl

/-! ## Conversions whose result a program needs -/

/-- An `int32` or `int64` element has an `int64` conversion: its signed value. -/
theorem convRel_i64_exists {P : Platform} {dt : DType} (hdt : dt = .i32 ∨ dt = .i64)
    (x : dt.Val) : ∃ r : I64, ConvRel P dt .i64 x r := by
  rcases hdt with h | h
  · cases h
    exact ⟨IntV.ofInt x.toInt, rfl⟩
  · cases h
    exact ⟨IntV.ofInt x.toInt, rfl⟩

/-- A `float32` element's conversion to `bfloat16` is its rounding. -/
theorem convRel_bf16_cvt {P : Platform} (x : F32) :
    ConvRel P .f32 .bf16 x (FloatV.cvt x) := by
  exact FloatRel.cvt_of_std (f := .binary32) (g := .bfloat16) (by decide) (by decide) (by decide) x

/-! ## The block mask, lane by lane -/

/-- The mask of a lane of the block pattern, in terms of the column it names. -/
theorem mask_get {d : Inst} {q : Nat} (hfit : Fits q d.BLOCK_H) (hn : d.hidden_dim < 2 ^ 31)
    (i : Idx [d.BLOCK_H - 0]) :
    (Block.mask q d.BLOCK_H d.hidden_dim).get i = decide (q * d.BLOCK_H + i.1 < d.hidden_dim) := by
  obtain ⟨l, ⟨⟩⟩ := i
  exact mask_lane hfit hn l

/-- An enabled lane of the block pattern names a real column. -/
theorem mask_get_lt {d : Inst} {q : Nat} (hfit : Fits q d.BLOCK_H) (hn : d.hidden_dim < 2 ^ 31)
    {i : Idx [d.BLOCK_H - 0]} (h : (Block.mask q d.BLOCK_H d.hidden_dim).get i = true) :
    q * d.BLOCK_H + i.1 < d.hidden_dim := by
  rw [mask_get hfit hn i] at h
  exact of_decide_eq_true h

/-! ## One program -/

/-- **One program.** Program `(p, q)` of the grid, run alone from `init`, is safe, reads its token
and the weight, writes its tile of the output, and leaves there `outVal` of the id at element `p`
of `input_`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {p q : Nat} (hp : p < d.n_tokens) (hq : q < nColBlocks d) :
    Safe E (R d init p) (W d p q) (fun _ m => Q d init p q m) init (bodyScript d p q) := by
  obtain ⟨hin, hw, hout, htoks⟩ := hpre
  have hB : 0 < d.BLOCK_H := block_pos hd
  have hn : d.hidden_dim < 2 ^ 31 := by
    have := hd.hidden_lt
    omega
  have hp31 : p < 2 ^ 31 := lt_of_lt_of_le hp (le_of_lt hd.tokens_lt)
  have hfit : Fits q d.BLOCK_H := fits_of_grid hd.hidden_lt hq
  have hlen : p < d.input_.len := lt_of_lt_of_le hp hd.input_len
  have hdI : d.dI = .i32 ∨ d.dI = .i64 := by
    rcases hd.dI_ok with h | h
    · exact Or.inr h
    · exact Or.inl h
  dsimp only [bodyScript]
  -- line 73: `row = tl.program_id(0).to(tl.int64)`
  refine safe_convert ?_ ?_
  · intro i
    exact ⟨IntV.ofInt ((IntV.ofNat p : I32).toInt), rfl⟩
  intro row hr
  have hrow : row.item.toInt = (p : Int) :=
    (convRel_i64_toInt (P := E.P) (dt := .i32) (Or.inl rfl) (hr ())).trans
      (IntV.I32.toInt_ofNat hp31)
  -- line 78: `t0 = tl.load(input_ptr + row)`
  rw [input_ptr_lane hrow]
  refine safe_load1_buf (d := d.dI) d.input_ hlen hin ?_ ?_
  · exact (Buf.span_sub_elems d.input_ (Set.mem_singleton p)).trans Set.subset_union_left
  intro t0 ht0
  have htok0 : tokAt d init p = some (ivOf t0) := by
    simp only [tokAt, ht0, Option.map_some]
  have hTok : TokOk d (ivOf t0) := htoks p hp (ivOf t0) htok0
  -- line 78: `.to(tl.int64)`
  refine safe_convert ?_ ?_
  · intro i
    exact convRel_i64_exists (P := E.P) hdI ((Tensor.scalar t0).get i)
  intro token htok
  have htokInt : token.item.toInt = ivOf t0 :=
    convRel_i64_toInt (P := E.P) hdI (htok ())
  -- line 94: `other = 0.0`
  refine safe_convert ?_ ?_
  · intro i
    exact ⟨FloatV.cvt ((tl.full [d.BLOCK_H] (FloatV.litF32 (FloatV.ofPat 0))).get i),
      convRel_bf16_cvt _⟩
  intro other hother
  have hother0 : ∀ l : Fin (d.BLOCK_H - 0), other.get (l, ()) = (FloatV.ofPat 0 : BF16) := by
    intro l
    exact convRel_bf16_zero (P := E.P) (hother (l, ()))
  -- line 96: `vals = tl.load(weight_ptr + local_id * weight_stride0 + cols, mask, other)`
  refine safe_load_buf (d := .bf16) (s := [d.BLOCK_H - 0]) (b := d.weight)
    (ptrs := d.weight.ptr 0 + (localIdT d token.item).item * (IntV.ofNat d.weight_stride0 : I64) +
      Block.offs q d.BLOCK_H)
    (mask := some (Tensor.and (Block.mask q d.BLOCK_H d.hidden_dim) (bc (validT d token.item).item)))
    (other := some other)
    (g := fun i => (wOff d (ivOf t0) (q * d.BLOCK_H + i.1)).toNat)
    (en := fun i => (Block.mask q d.BLOCK_H d.hidden_dim).get i && (validT d token.item).item)
    ?_ ?_ hw ?_ ?_
  · -- the lanes address the weight elements
    refine Buf.addrs_of_get d.weight (fun i => rfl) ?_
    intro i hi
    have hsplit := hi
    simp only [Bool.and_eq_true] at hsplit
    have hcol := mask_get_lt hfit hn hsplit.1
    have htile : Tile q d.BLOCK_H d.hidden_dim (q * d.BLOCK_H + i.1) :=
      ⟨Nat.le_add_right _ _, Nat.add_lt_add_left i.1.2 _, hcol⟩
    have hnn : 0 ≤ wOff d (ivOf t0) (q * d.BLOCK_H + i.1) :=
      (weight_lane_lt hd htokInt hTok hsplit.2 i.1 htile).1
    have h1 := weight_ptr_lane hd htokInt hTok hfit i.1
    rw [Int.toNat_of_nonneg hnn]
    exact h1
  · -- the enabled lanes address weight elements
    intro i hi
    have hsplit := hi
    simp only [Bool.and_eq_true] at hsplit
    have hcol := mask_get_lt hfit hn hsplit.1
    have htile : Tile q d.BLOCK_H d.hidden_dim (q * d.BLOCK_H + i.1) :=
      ⟨Nat.le_add_right _ _, Nat.add_lt_add_left i.1.2 _, hcol⟩
    exact (weight_lane_lt hd htokInt hTok hsplit.2 i.1 htile).2
  · -- the read footprint is the weight
    refine (Buf.elems_sub_bytes d.weight (fun j hj => ?_)).trans Set.subset_union_right
    simp only [Buf.img, Set.mem_ofPred_eq] at hj
    obtain ⟨i, hi, rfl⟩ := hj
    have hsplit := hi
    simp only [Bool.and_eq_true] at hsplit
    have hcol := mask_get_lt hfit hn hsplit.1
    have htile : Tile q d.BLOCK_H d.hidden_dim (q * d.BLOCK_H + i.1) :=
      ⟨Nat.le_add_right _ _, Nat.add_lt_add_left i.1.2 _, hcol⟩
    exact (weight_lane_lt hd htokInt hTok hsplit.2 i.1 htile).2
  intro vals hload hdis
  -- line 99: `tl.store(out_ptr + row * hidden_dim + cols, vals, mask=col_mask)`
  refine safe_last (safe_store_buf (d := .bf16) (s := [d.BLOCK_H - 0]) (b := d.output)
    (ptrs := d.output.ptr 0 + row.item * (IntV.ofNat d.hidden_dim : I64) + Block.offs q d.BLOCK_H)
    (mask := some (Block.mask q d.BLOCK_H d.hidden_dim)) (vals := vals)
    (g := fun i => p * d.hidden_dim + (q * d.BLOCK_H + i.1))
    (en := fun i => (Block.mask q d.BLOCK_H d.hidden_dim).get i)
    ?_ ?_ ?_ hout ?_ ?_)
  · -- the lanes address the output elements
    refine Buf.addrs_of_get d.output (fun i => rfl) ?_
    intro i hi
    exact out_ptr_lane hd hrow hp hfit i.1
  · -- the enabled lanes address output elements
    intro i hi
    exact outTile_lt hd hp ⟨q * d.BLOCK_H + i.1,
      ⟨Nat.le_add_right _ _, Nat.add_lt_add_left i.1.2 _, mask_get_lt hfit hn hi⟩, rfl⟩
  · -- distinct lanes address distinct elements
    intro i j hi hj he
    have h1 : q * d.BLOCK_H + i.1 = q * d.BLOCK_H + j.1 := by omega
    exact Prod.ext (Fin.ext (by omega)) rfl
  · -- the write footprint is the tile
    have hen : (fun i : Idx [d.BLOCK_H - 0] =>
        (Block.mask q d.BLOCK_H d.hidden_dim).get i) =
        (fun i => decide (q * d.BLOCK_H + i.1 < d.hidden_dim)) := by
      funext i
      exact mask_get hfit hn i
    rw [hen, img_outTile]
    exact subset_rfl
  intro m' hstore _ _
  refine safe_ret ?_
  -- the tile holds `outVal` of the id at element `p`
  intro l hcol
  have hcol' : q * d.BLOCK_H + l.val < d.hidden_dim := hcol
  have hen : (Block.mask q d.BLOCK_H d.hidden_dim).get (l, ()) = true := by
    rw [mask_get hfit hn (l, ())]
    exact decide_eq_true hcol'
  have htile : Tile q d.BLOCK_H d.hidden_dim (q * d.BLOCK_H + l.val) :=
    ⟨Nat.le_add_right _ _, Nat.add_lt_add_left l.2 _, hcol'⟩
  refine ⟨ivOf t0, htok0, vals.get (l, ()), hstore (l, ()) hen, ?_⟩
  by_cases hv : (validT d token.item).item = true
  · have henl : ((Block.mask q d.BLOCK_H d.hidden_dim).get (l, ()) &&
        (validT d token.item).item) = true := by
      rw [hen, hv, Bool.and_self]
    have hnn : 0 ≤ wOff d (ivOf t0) (q * d.BLOCK_H + l.val) :=
      (weight_lane_lt hd htokInt hTok hv l htile).1
    have hread := hload (l, ()) henl
    rw [Int.toNat_of_nonneg hnn] at hread
    have hdisj : (d.org_vocab_start_index ≤ ivOf t0 ∧ ivOf t0 < d.org_vocab_end_index) ∨
        (d.added_vocab_start_index ≤ ivOf t0 ∧ ivOf t0 < d.added_vocab_end_index) :=
      of_decide_eq_true (by rw [← validT_item hd htokInt]; exact hv)
    exact (outVal_at hdisj).2 hread
  · have hvf : (validT d token.item).item = false := by
      rcases Bool.eq_false_or_eq_true ((validT d token.item).item) with h | h
      · exact absurd h hv
      · exact h
    have hfalse : ((Block.mask q d.BLOCK_H d.hidden_dim).get (l, ()) &&
        (validT d token.item).item) = false := by
      rw [hen, hvf]
      rfl
    have hc0 : vals.get (l, ()) = (FloatV.ofPat 0 : BF16) :=
      (hdis (l, ()) hfalse other rfl).trans (hother0 l)
    have hnot : ¬ ((d.org_vocab_start_index ≤ ivOf t0 ∧ ivOf t0 < d.org_vocab_end_index) ∨
        (d.added_vocab_start_index ≤ ivOf t0 ∧ ivOf t0 < d.added_vocab_end_index)) := by
      intro h
      have hd1 : (validT d token.item).item = true := by
        rw [validT_item hd htokInt]
        exact decide_eq_true h
      rw [hvf] at hd1
      exact Bool.noConfusion hd1
    unfold outVal
    rw [ite_eq_right (fun h => hnot (Or.inl h)), ite_eq_right (fun h => hnot (Or.inr h))]
    exact hc0

/-! ## The claim -/

/-- **The claim.** Each program is safe on its own tile (`safe_program`); the footprints are
disjoint (`Spec`), and the programs' results give `Post` (`post_of_local`). -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => R d init (prow d i)) (fun i => W d (prow d i) (pcol d i))
    (fun i m => Q d init (prow d i) (pcol d i) m) (launch_valid hd) (fun i => ?_)
    (fun i j hij => disjoint_foot hd i j hij) (fun i => disjoint_read_write hd i)
    (fun fin hloc hfr => post_of_local hd fin init hloc hfr)
  refine Eq.mpr (congrArg (Safe _ _ _ _ _) ((body_eq d ((launch d).ctx i)).trans
    (congrArg₂ (bodyScript d) (launch_pid0 d i) (launch_pid1 d i)))) ?_
  exact safe_program hd hpre _ (prow_lt i) (pcol_lt i)

end Claims.VocabEmbedding
