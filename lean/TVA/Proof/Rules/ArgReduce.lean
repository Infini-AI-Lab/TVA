import TVA.Proof.Rules.Reduce

/-!
# Rules: index reductions

The proof rules for `tl.argmaxAll`, `tl.argmax1`, `tl.argmax0` (`tl.argmax`, `tl.argmin`,
`tl.max(…, return_indices=True)`; `TVA.Triton.Reduce`), and what their results `tl.ArgRes` allow:

* `tl.ArgRes.index_lt`: the index is a position of the axis;
* `tl.ArgRes.value`: the value is one of the elements (up to NaN payload);
* `tl.ArgRes.tree_of_noNaN`: without NaN, the pair some tree of `argCombine` gives; then
  `Tree.eval_argCombine` (the pair is a leaf's) and, with `tie_break_left` on integers,
  `Tree.eval_argCombine_first` (the first index of the maximum or minimum; `IntV.cmpKey_lt`,
  `IntV.cmpKey_eq` give the integer keys).

How this fits: model → `Safe` per program (the rules of `TVA.Proof.Rules`) → `correct_of_safe`.
-/
namespace TVA

open Prog

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- The pair `argCombine` keeps is one of its two operands. -/
theorem tl.argCombine_eq_or {β : Type} [CmpKey β] (isMax tieLeft : Bool) (a b : β × Nat) :
    tl.argCombine isMax tieLeft a b = a ∨ tl.argCombine isMax tieLeft a b = b := by
  simp only [tl.argCombine]
  split_ifs <;> simp

/-- An index reduction along a tree returns one of its leaves: the index is a leaf of the tree,
and the value is the element at that index. -/
theorem Tree.eval_argCombine {β : Type} [CmpKey β] {isMax tieLeft : Bool} {g : Nat → β} :
    ∀ {t : Tree} {p : β × Nat}, t.Eval (fun a b c => c = tl.argCombine isMax tieLeft a b)
      (fun j => (g j, j)) p → p.2 ∈ t.leaves ∧ p.1 = g p.2
  | .leaf i, _, .leaf _ => ⟨by simp [Tree.leaves], rfl⟩
  | .node l r, _, .node (a := a) (b := b) ha hb hc => by
    obtain ⟨hal, haf⟩ := Tree.eval_argCombine ha
    obtain ⟨hbl, hbf⟩ := Tree.eval_argCombine hb
    subst hc
    rcases tl.argCombine_eq_or isMax tieLeft a b with h | h <;> rw [h]
    · exact ⟨List.mem_append_left _ hal, haf⟩
    · exact ⟨List.mem_append_right _ hbl, hbf⟩

/-- The pair a covering tree gives is a result `ArgRes` allows. -/
theorem tl.ArgRes.of_tree {β : Type} [CmpKey β] {isMax tieLeft : Bool} {n : Nat} {g : Nat → β}
    {p : β × Nat} {t : Tree} (ht : t.Covers n)
    (hp : t.Eval (fun a b c => c = tl.argCombine isMax tieLeft a b) (fun j => (g j, j)) p) :
    tl.ArgRes isMax tieLeft n g (p.1, IntV.ofNat p.2) :=
  Or.inl ⟨p, t, ht, hp, rfl⟩

/-- An index reduction over `n > 0` positions has a result (a sequential tree's). -/
theorem tl.ArgRes.exists {β : Type} [CmpKey β] (isMax tieLeft : Bool) {n : Nat} (hn : 0 < n)
    (g : Nat → β) : ∃ r, tl.ArgRes isMax tieLeft n g r := by
  obtain ⟨p, hp⟩ := Tree.eval_exists (op := fun a b c => c = tl.argCombine isMax tieLeft a b)
    (fun a b => ⟨_, rfl⟩) (fun j => (g j, j)) (Tree.seq n)
  exact ⟨_, tl.ArgRes.of_tree (Tree.seq_covers hn) hp⟩

/-- **Without NaN** among the elements, the result is the pair some covering tree gives. -/
theorem tl.ArgRes.tree_of_noNaN {β : Type} [CmpKey β] {isMax tieLeft : Bool} {n : Nat}
    {g : Nat → β} {r : β × I32} (hno : ∀ j, j < n → CmpKey.nan (g j) = false)
    (h : tl.ArgRes isMax tieLeft n g r) :
    ∃ p : β × Nat, ∃ t : Tree, t.Covers n ∧
      t.Eval (fun a b c => c = tl.argCombine isMax tieLeft a b) (fun j => (g j, j)) p ∧
      r = (p.1, IntV.ofNat p.2) := by
  rcases h with h | ⟨⟨j, hj, hnan⟩, -⟩
  · exact h
  · rw [hno j hj] at hnan; cases hnan

/-- The index is that of a position. -/
theorem tl.ArgRes.index_lt {β : Type} [CmpKey β] {isMax tieLeft : Bool} {n : Nat}
    {g : Nat → β} {r : β × I32} (h : tl.ArgRes isMax tieLeft n g r) :
    ∃ j, j < n ∧ r.2 = IntV.ofNat j := by
  rcases h with ⟨p, t, ht, hp, rfl⟩ | ⟨-, -, hidx⟩
  · have hl := (Tree.eval_argCombine hp).1
    exact ⟨p.2, List.mem_range.mp (ht.mem_iff.mp hl), rfl⟩
  · exact hidx

/-- The value is one of the elements, up to NaN payload. -/
theorem tl.ArgRes.value {β : Type} [CmpKey β] {isMax tieLeft : Bool} {n : Nat}
    {g : Nat → β} {r : β × I32} (h : tl.ArgRes isMax tieLeft n g r) :
    ∃ j, j < n ∧ (r.1 = g j ∨ (CmpKey.nan r.1 = true ∧ CmpKey.nan (g j) = true)) := by
  rcases h with ⟨p, t, ht, hp, rfl⟩ | ⟨-, hval, -⟩
  · obtain ⟨hl, hv⟩ := Tree.eval_argCombine hp
    exact ⟨p.2, List.mem_range.mp (ht.mem_iff.mp hl), Or.inl hv⟩
  · exact hval

/-- Integers have no NaN. -/
@[simp] theorem IntV.cmpKey_nan {sg : Bool} {w : Nat} (a : IntV sg w) : CmpKey.nan a = false :=
  rfl

/-- **`tl.argmax`/`tl.argmin` of a rank-1 tensor** (`tl.argmaxAll`): never undefined; the rest
must be safe for every result `ArgRes` allows. -/
theorem safe_argmaxAll {β : Type} [CmpKey β] [Inhabited β] {N : Nat} (hn : 0 < N)
    (isMax tieLeft : Bool) {x : Tensor [N] β} {k : β × I32 → Prog α}
    (hk : ∀ r : β × I32,
      tl.ArgRes isMax tieLeft N (fun j => if h : j < N then x.get1 ⟨j, h⟩ else default) r →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.argmaxAll isMax tieLeft x >>= k) := by
  simp only [tl.argmaxAll, bind_eq, Prog.bind]
  exact .pick (tl.ArgRes.exists isMax tieLeft hn _) hk

/-- **`tl.argmax(x, 1)`** (`tl.argmax1`): as `safe_argmaxAll` for each row. -/
theorem safe_argmax1 {β : Type} [CmpKey β] [Inhabited β] {M N : Nat} (hn : 0 < N)
    (isMax tieLeft : Bool) {x : Tensor [M, N] β} {k : Tensor [M] β × Tensor [M] I32 → Prog α}
    (hk : ∀ r : Tensor [M] β × Tensor [M] I32, (∀ i, tl.ArgRes isMax tieLeft N
        (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) (r.1.get1 i, r.2.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.argmax1 isMax tieLeft x >>= k) := by
  simp only [tl.argmax1, bind_eq, Prog.bind]
  have hex : ∀ i : Fin M, ∃ r, tl.ArgRes isMax tieLeft N
      (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) r :=
    fun i => tl.ArgRes.exists isMax tieLeft hn _
  refine .pick ⟨(Tensor.of1 fun i => (Classical.choose (hex i)).1,
      Tensor.of1 fun i => (Classical.choose (hex i)).2), fun i => ?_⟩ hk
  simpa using Classical.choose_spec (hex i)

/-- **`tl.argmax(x, 0)`** (`tl.argmax0`): as `safe_argmaxAll` for each column. -/
theorem safe_argmax0 {β : Type} [CmpKey β] [Inhabited β] {M N : Nat} (hm : 0 < M)
    (isMax tieLeft : Bool) {x : Tensor [M, N] β} {k : Tensor [N] β × Tensor [N] I32 → Prog α}
    (hk : ∀ r : Tensor [N] β × Tensor [N] I32, (∀ j, tl.ArgRes isMax tieLeft M
        (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.1.get1 j, r.2.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.argmax0 isMax tieLeft x >>= k) := by
  simp only [tl.argmax0, bind_eq, Prog.bind]
  have hex : ∀ j : Fin N, ∃ r, tl.ArgRes isMax tieLeft M
      (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) r :=
    fun j => tl.ArgRes.exists isMax tieLeft hm _
  refine .pick ⟨(Tensor.of1 fun j => (Classical.choose (hex j)).1,
      Tensor.of1 fun j => (Classical.choose (hex j)).2), fun j => ?_⟩ hk
  simpa using Classical.choose_spec (hex j)

/-- `CmpKey`'s `<` on integers is the order of their values. -/
theorem IntV.cmpKey_lt {sg : Bool} {w : Nat} (a b : IntV sg w) :
    CmpKey.lt a b = true ↔ a.toInt < b.toInt := by
  show IntV.lt a b = true ↔ _
  cases sg
  · simp only [IntV.lt, IntV.toInt, Bool.false_eq_true, ite_false, BitVec.ult_iff_lt]
    simp [BitVec.lt_def]
  · simp only [IntV.lt, IntV.toInt, ite_true]
    exact BitVec.slt_iff_toInt_lt

/-- `CmpKey`'s `==` on integers is the equality of their values. -/
theorem IntV.cmpKey_eq {sg : Bool} {w : Nat} (a b : IntV sg w) :
    CmpKey.eq a b = true ↔ a.toInt = b.toInt := by
  show IntV.eq a b = true ↔ _
  cases sg
  · simp only [IntV.eq, IntV.toInt, Bool.false_eq_true, ite_false, beq_iff_eq, Nat.cast_inj]
    exact ⟨fun h => h ▸ rfl, fun h => BitVec.eq_of_toNat_eq h⟩
  · simp only [IntV.eq, IntV.toInt, ite_true, beq_iff_eq]
    exact ⟨fun h => h ▸ rfl, fun h => BitVec.eq_of_toInt_eq h⟩

section ArgFirst
variable {β : Type} [CmpKey β] (key : β → Int)
  (hlt : ∀ u v : β, CmpKey.lt u v = true ↔ key u < key v)
  (heq : ∀ u v : β, CmpKey.eq u v = true ↔ key u = key v)

include hlt heq in
/-- With `tie_break_left`, on values ordered by `key`, `argCombine` keeps the pair whose value is
better (greater for `isMax`, smaller otherwise), and on equal values the smaller index. -/
theorem tl.argCombine_tieLeft (isMax : Bool) (a b : β × Nat) :
    tl.argCombine isMax true a b =
      if (if isMax then key b.1 < key a.1 else key a.1 < key b.1) ∨
          (key a.1 = key b.1 ∧ a.2 < b.2) then a else b := by
  have hw : ((if isMax then CmpKey.lt b.1 a.1 else CmpKey.lt a.1 b.1) ||
      (true && CmpKey.eq a.1 b.1 && decide (a.2 < b.2))) = true ↔
      ((if isMax then key b.1 < key a.1 else key a.1 < key b.1) ∨
        (key a.1 = key b.1 ∧ a.2 < b.2)) := by
    cases isMax <;> simp [hlt, heq]
  simp only [tl.argCombine]
  by_cases h : (if isMax then key b.1 < key a.1 else key a.1 < key b.1) ∨
      (key a.1 = key b.1 ∧ a.2 < b.2)
  · rw [ite_eq_left (hw.mpr h), ite_eq_left h]
  · have h' : ¬ (((if isMax then CmpKey.lt b.1 a.1 else CmpKey.lt a.1 b.1) ||
        (true && CmpKey.eq a.1 b.1 && decide (a.2 < b.2))) = true) := fun h' => h (hw.mp h')
    rw [ite_eq_right h', ite_eq_right h]

include hlt heq in
/-- **`tl.argmax`/`tl.argmin` with `tie_break_left` on keyed values.** Over any tree whose leaves
are distinct, the reduction of `(g j, j)` gives a leaf `p.2` whose value is a maximum (`isMax`)
or minimum of the leaves' values, and the smallest such leaf. With `t.Covers n` the leaves are
`0, …, n-1`, so `p.2` is the first index of the maximum (minimum). -/
theorem Tree.eval_argCombine_first (isMax : Bool) (g : Nat → β) :
    ∀ {t : Tree} {p : β × Nat}, t.leaves.Nodup →
      t.Eval (fun a b c => c = tl.argCombine isMax true a b) (fun j => (g j, j)) p →
      p.2 ∈ t.leaves ∧ p.1 = g p.2 ∧ ∀ j ∈ t.leaves,
        (if isMax then key (g j) ≤ key (g p.2) else key (g p.2) ≤ key (g j)) ∧
        (key (g j) = key (g p.2) → p.2 ≤ j)
  | .leaf i, _, _, .leaf _ => by
    refine ⟨by simp [Tree.leaves], rfl, fun j hj => ?_⟩
    simp only [Tree.leaves, List.mem_singleton] at hj
    subst hj; cases isMax <;> simp
  | .node l r, _, hnd, .node (a := ⟨va, ia⟩) (b := ⟨vb, ib⟩) ha hb hc => by
    simp only [Tree.leaves, List.nodup_append] at hnd
    obtain ⟨hndl, hndr, hdisj⟩ := hnd
    obtain ⟨hal, haf, hab⟩ := Tree.eval_argCombine_first isMax g hndl ha
    obtain ⟨hbl, hbf, hbb⟩ := Tree.eval_argCombine_first isMax g hndr hb
    simp only at hal haf hab hbl hbf hbb
    subst haf hbf hc
    rw [tl.argCombine_tieLeft key hlt heq]
    simp only [Tree.leaves, List.mem_append]
    by_cases hw : (if isMax then key (g ib) < key (g ia) else key (g ia) < key (g ib)) ∨
        (key (g ia) = key (g ib) ∧ ia < ib)
    · rw [ite_eq_left hw]
      refine ⟨Or.inl hal, rfl, fun j hj => ?_⟩
      rcases hj with hj | hj
      · exact hab j hj
      · obtain ⟨h1, h2⟩ := hbb j hj
        have hjne : ia ≠ j := hdisj _ hal _ hj
        cases isMax <;> simp only [Bool.false_eq_true, ite_false, ite_true] at hw h1 ⊢ <;>
          exact ⟨by omega, fun e => by omega⟩
    · rw [ite_eq_right hw]
      refine ⟨Or.inr hbl, rfl, fun j hj => ?_⟩
      rcases hj with hj | hj
      · obtain ⟨h1, h2⟩ := hab j hj
        have hne : ia ≠ ib := hdisj _ hal _ hbl
        cases isMax <;> simp only [Bool.false_eq_true, ite_false, ite_true] at hw h1 ⊢ <;>
          exact ⟨by omega, fun e => by omega⟩
      · exact hbb j hj

end ArgFirst

end TVA
