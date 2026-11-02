import PagedAttention

/-! Reviewer r-pa, finding on line 68 at `PAGE_SIZE = 1`: a reduction over one lane is the lane
itself. `Inst.ok.page_size` admits `PAGE_SIZE = 2^0 = 1`; then `tl.sum0c` of
`p[:, None] * v` (shape `[1, HEAD_DIM]`) returns, in each column, exactly the rounded product
`p * v` of its one row, so the `+` of line 68 adds two products. -/

open TVA

/-- Every tree has a leaf. -/
theorem leaves_pos : ∀ t : Tree, 0 < t.leaves.length
  | .leaf _ => by simp [Tree.leaves]
  | .node l _ => by
    have := leaves_pos l
    simp only [Tree.leaves, List.length_append]; omega

/-- A tree that covers one leaf is that leaf. -/
theorem covers_one {t : Tree} (h : t.Covers 1) : t = Tree.leaf 0 := by
  cases t with
  | leaf i =>
    have hm : i ∈ (Tree.leaf i).leaves := by simp [Tree.leaves]
    rw [h.mem_iff] at hm
    simp at hm
    rw [hm]
  | node l r =>
    exfalso
    have hl := leaves_pos l
    have hr := leaves_pos r
    have := h.length_eq
    simp only [Tree.leaves, List.length_append, List.length_range] at this
    omega

/-- So a one-lane `AddRel` reduction gives the lane: no addition is left in it. -/
theorem eval_one {f : Math.Float.Format} {fusion : Bool} {x : Nat → FloatV f} {r : FloatV f}
    (h : ∃ t : Tree, t.Covers 1 ∧ t.Eval (AddRel fusion) x r) : r = x 0 := by
  obtain ⟨t, ht, he⟩ := h
  rw [covers_one ht] at he
  cases he
  rfl

/-- `PAGE_SIZE = 1` is covered by `Inst.ok`'s field `page_size`. -/
example : ∃ b, b ≤ 8 ∧ (1 : Nat) = 2 ^ b := ⟨0, by norm_num, rfl⟩
