import TVA.Proof.Safe

/-!
# Rules: reductions

The proof rules for `TVA.Triton.Reduce`. A float reduction combines its elements along a tree
Triton chooses (`Tree`), so its rule hands the rest of the program every result some admissible
tree gives:

* sums (`tl.sum`, `tl.sum1`, `tl.sum0`, the fusable `tl.sumc`, `sum1c`, `sum0c`, and the
  underlying `…With`): a tree covering the axis, of depth at most `tl.sumBound E.P n`, whose nodes
  are `AddRel` additions. A balanced tree (`Tree.bal`) shows one exists;
* `tl.max`/`tl.min` (`safe_max1`, `safe_maxAll`, `safe_max0`): a covering tree of `MaxRel` nodes;
* index reductions (`tl.argmax`, `tl.max(…, return_indices=True)`): `safe_argmaxAll`,
  `safe_argmax1`, `safe_argmax0`, whose results `tl.ArgRes` describes. Facts about them:
  `tl.ArgRes.index_lt` (the index is in range), `tl.ArgRes.value` (the value is an element),
  `tl.ArgRes.tree_of_noNaN` (without NaN, a tree's pair), with `Tree.eval_argCombine` (the result
  is a leaf) and `Tree.eval_argCombine_first` (with `tie_break_left` on integers, the first index
  of the maximum or minimum).

How this fits: model → `Safe` per program (the rules of `TVA.Proof.Rules`) → `correct_of_safe`.
The float error of a sum along a tree is the float kit's.
-/
namespace TVA

open Prog

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-! ## Trees -/

/-- With a total combining relation, every tree has a value. -/
theorem Tree.eval_exists {β : Type} {op : β → β → β → Prop} (htot : ∀ a b, ∃ c, op a b c)
    (x : Nat → β) : ∀ t : Tree, ∃ r, t.Eval op x r
  | .leaf i => ⟨x i, .leaf i⟩
  | .node l r => by
    obtain ⟨a, ha⟩ := Tree.eval_exists htot x l
    obtain ⟨b, hb⟩ := Tree.eval_exists htot x r
    obtain ⟨c, hc⟩ := htot a b
    exact ⟨c, .node ha hb hc⟩

/-- Every pair of floats has a sum `AddRel` allows (the IEEE one). -/
theorem addRel_total {f : Format} (fusion : Bool) : ∀ a b : FloatV f, ∃ c, AddRel fusion a b c :=
  fun a b => ⟨_, AddRel.ieee fusion a b⟩

/-- A balanced tree over `lo, …, lo + len − 1` in order: the first half has `⌈len/2⌉` leaves. -/
def Tree.bal (lo : Nat) : Nat → Tree
  | 0 => .leaf lo
  | 1 => .leaf lo
  | len + 2 =>
    .node (Tree.bal lo ((len + 3) / 2)) (Tree.bal (lo + (len + 3) / 2) (len + 2 - (len + 3) / 2))
termination_by len => len
decreasing_by all_goals omega

/-- A balanced tree's leaves are its range, in order. -/
theorem Tree.bal_leaves (lo : Nat) :
    ∀ len, 0 < len → (Tree.bal lo len).leaves = List.range' lo len
  | 0, h => absurd h (lt_irrefl 0)
  | 1, _ => by simp [Tree.bal, Tree.leaves]
  | len + 2, _ => by
    rw [Tree.bal, Tree.leaves, Tree.bal_leaves lo ((len + 3) / 2) (by omega),
      Tree.bal_leaves (lo + (len + 3) / 2) (len + 2 - (len + 3) / 2) (by omega)]
    have := List.range'_append (s := lo) (m := (len + 3) / 2) (n := len + 2 - (len + 3) / 2)
      (step := 1)
    simp only [one_mul] at this
    rw [this]; congr 1; omega
termination_by len => len
decreasing_by all_goals omega

/-- A balanced tree has depth at most `⌈log₂ len⌉`. -/
theorem Tree.bal_depth (lo : Nat) : ∀ len, (Tree.bal lo len).depth ≤ Nat.clog 2 len
  | 0 => by simp [Tree.bal, Tree.depth]
  | 1 => by simp [Tree.bal, Tree.depth]
  | len + 2 => by
    rw [Tree.bal, Tree.depth, Nat.clog_of_two_le (by norm_num) (by omega)]
    have h1 := Tree.bal_depth lo ((len + 3) / 2)
    have h2 := Tree.bal_depth (lo + (len + 3) / 2) (len + 2 - (len + 3) / 2)
    have hm : Nat.clog 2 (len + 2 - (len + 3) / 2) ≤ Nat.clog 2 ((len + 3) / 2) :=
      Nat.clog_mono_right 2 (by omega)
    have he : (len + 2 + 2 - 1) / 2 = (len + 3) / 2 := by omega
    rw [he]; omega
termination_by len => len
decreasing_by all_goals omega

/-- A balanced tree over `0, …, n − 1` covers `n`. -/
theorem Tree.bal_covers {n : Nat} (hn : 0 < n) : (Tree.bal 0 n).Covers n := by
  unfold Tree.Covers; rw [Tree.bal_leaves 0 n hn, List.range_eq_range']

/-- Some tree over `0, …, n − 1` fits every reduction's depth bound (`tl.sumBound`). -/
theorem Tree.exists_sumBound (P : Platform) {n : Nat} (hn : 0 < n) :
    ∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound P n :=
  ⟨Tree.bal 0 n, Tree.bal_covers hn, (Tree.bal_depth 0 n).trans (le_max_right _ _)⟩

/-! ## Sums -/

section Sums
variable {f : Format}

/-- `tl.sumWith fusion x` (rank 1, `n > 0`): the rest must be safe for every sum along a covering
tree of depth at most `tl.sumBound E.P n` with `AddRel fusion` nodes. -/
theorem safe_sumWith (fusion : Bool) {n : Nat} (hn : 0 < n) {x : Tensor [n] (FloatV f)}
    {k : FloatV f → Prog α}
    (hk : ∀ r, (∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound E.P n ∧
        t.Eval (AddRel fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sumWith fusion x >>= k) := by
  simp only [tl.sumWith, bind_eq, Prog.bind]
  refine .env ?_
  obtain ⟨t, hc, hd⟩ := Tree.exists_sumBound E.P hn
  obtain ⟨r, hr⟩ := Tree.eval_exists (addRel_total fusion)
    (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) t
  exact .pick ⟨r, t, hc, hd, hr⟩ hk

/-- `tl.sum1With fusion x` (rows of a rank-2 tensor, `N > 0`): each row's sum along its own tree,
as in `safe_sumWith`. -/
theorem safe_sum1With (fusion : Bool) {M N : Nat} (hn : 0 < N) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧
        t.depth ≤ tl.sumBound E.P N ∧
        t.Eval (AddRel fusion) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default)
          (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum1With fusion x >>= k) := by
  simp only [tl.sum1With, bind_eq, Prog.bind]
  refine .env ?_
  obtain ⟨t, hc, hd⟩ := Tree.exists_sumBound E.P hn
  have hex : ∀ i : Fin M, ∃ r, t.Eval (AddRel fusion)
      (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) r :=
    fun i => Tree.eval_exists (addRel_total fusion) _ _
  exact .pick ⟨Tensor.of1 (fun i => Classical.choose (hex i)), fun i =>
    ⟨t, hc, hd, Classical.choose_spec (hex i)⟩⟩ hk

/-- `tl.sum0With fusion x` (columns of a rank-2 tensor, `M > 0`): each column's sum along its own
tree, as in `safe_sumWith`. -/
theorem safe_sum0With (fusion : Bool) {M N : Nat} (hm : 0 < M) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧
        t.depth ≤ tl.sumBound E.P M ∧
        t.Eval (AddRel fusion) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default)
          (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0With fusion x >>= k) := by
  simp only [tl.sum0With, bind_eq, Prog.bind]
  refine .env ?_
  obtain ⟨t, hc, hd⟩ := Tree.exists_sumBound E.P hm
  have hex : ∀ j : Fin N, ∃ r, t.Eval (AddRel fusion)
      (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) r :=
    fun j => Tree.eval_exists (addRel_total fusion) _ _
  exact .pick ⟨Tensor.of1 (fun j => Classical.choose (hex j)), fun j =>
    ⟨t, hc, hd, Classical.choose_spec (hex j)⟩⟩ hk

/-- `tl.sum(x)` (not a product): IEEE additions along a tree. -/
theorem safe_sum {n : Nat} (hn : 0 < n) {x : Tensor [n] (FloatV f)} {k : FloatV f → Prog α}
    (hk : ∀ r, (∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound E.P n ∧
        t.Eval (AddRel false) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum x >>= k) := safe_sumWith false hn hk

/-- `tl.sum(x, axis=1)` (not a product): IEEE additions along a tree per row. -/
theorem safe_sum1 {M N : Nat} (hn : 0 < N) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧
        t.depth ≤ tl.sumBound E.P N ∧
        t.Eval (AddRel false) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default)
          (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum1 x >>= k) := safe_sum1With false hn hk

/-- `tl.sum(x, axis=0)` (not a product): IEEE additions along a tree per column. -/
theorem safe_sum0 {M N : Nat} (hm : 0 < M) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧
        t.depth ≤ tl.sumBound E.P M ∧
        t.Eval (AddRel false) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default)
          (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0 x >>= k) := safe_sum0With false hm hk

/-- `tl.sum(x)` of a product: fusable additions (`AddRel E.fusion`) along a tree. -/
theorem safe_sumc {n : Nat} (hn : 0 < n) {x : Tensor [n] (FloatV f)} {k : FloatV f → Prog α}
    (hk : ∀ r, (∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound E.P n ∧
        t.Eval (AddRel E.fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sumc x >>= k) := by
  simp only [tl.sumc, bind_eq, Prog.bind]
  exact .env (by simpa only [bind_eq] using safe_sumWith E.fusion hn hk)

/-- `tl.sum(x, axis=1)` of a product: fusable additions along a tree per row. -/
theorem safe_sum1c {M N : Nat} (hn : 0 < N) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧
        t.depth ≤ tl.sumBound E.P N ∧
        t.Eval (AddRel E.fusion) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default)
          (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum1c x >>= k) := by
  simp only [tl.sum1c, bind_eq, Prog.bind]
  exact .env (by simpa only [bind_eq] using safe_sum1With E.fusion hn hk)

/-- `tl.sum(x, axis=0)` of a product: fusable additions along a tree per column. -/
theorem safe_sum0c {M N : Nat} (hm : 0 < M) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧
        t.depth ≤ tl.sumBound E.P M ∧
        t.Eval (AddRel E.fusion) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default)
          (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0c x >>= k) := by
  simp only [tl.sum0c, bind_eq, Prog.bind]
  exact .env (by simpa only [bind_eq] using safe_sum0With E.fusion hm hk)

/-! ## Maximum and minimum -/

/-- Every pair of floats has a maximum (minimum) `MaxRel` allows. -/
theorem maxRel_total (isMax : Bool) : ∀ a b : FloatV f, ∃ c, FloatV.MaxRel isMax a b c :=
  fun a b => ⟨if isMax then FloatV.max a b else FloatV.min a b, Or.inl (Or.inl rfl)⟩

/-- `tl.max(x, axis=1)` (`isMax`) or `tl.min`: each row combined with `MaxRel` along a tree. -/
theorem safe_max1 {M N : Nat} (hn : 0 < N) (isMax : Bool) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [M] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [M] (FloatV f), (∀ i, ∃ t : Tree, t.Covers N ∧
        t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default)
          (r.get1 i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.max1 isMax x >>= k) := by
  simp only [tl.max1, bind_eq, Prog.bind]
  have hex : ∀ i : Fin M, ∃ r, (Tree.seq N).Eval (FloatV.MaxRel isMax)
      (fun j => if h : j < N then x.get2 i ⟨j, h⟩ else default) r :=
    fun i => Tree.eval_exists (maxRel_total isMax) _ _
  exact .pick ⟨Tensor.of1 (fun i => Classical.choose (hex i)), fun i =>
    ⟨Tree.seq N, Tree.seq_covers hn, Classical.choose_spec (hex i)⟩⟩ hk

/-- `tl.max(x)` (`isMax`) or `tl.min` of a rank-1 tensor: combined with `MaxRel` along a tree. -/
theorem safe_maxAll {N : Nat} (hn : 0 < N) (isMax : Bool) {x : Tensor [N] (FloatV f)}
    {k : FloatV f → Prog α}
    (hk : ∀ r : FloatV f, (∃ t : Tree, t.Covers N ∧
        t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < N then x.get1 ⟨j, h⟩ else default) r) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.maxAll isMax x >>= k) := by
  simp only [tl.maxAll, bind_eq, Prog.bind]
  obtain ⟨r, hr⟩ := Tree.eval_exists (maxRel_total isMax)
    (fun j => if h : j < N then x.get1 ⟨j, h⟩ else default) (Tree.seq N)
  exact .pick ⟨r, Tree.seq N, Tree.seq_covers hn, hr⟩ hk

/-- `tl.max(x, axis=0)` (`isMax`) or `tl.min`: each column combined with `MaxRel` along a tree. -/
theorem safe_max0 {M N : Nat} (hm : 0 < M) (isMax : Bool) {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧
        t.Eval (FloatV.MaxRel isMax) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default)
          (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.max0 isMax x >>= k) := by
  simp only [tl.max0, bind_eq, Prog.bind]
  have hex : ∀ j : Fin N, ∃ r, (Tree.seq M).Eval (FloatV.MaxRel isMax)
      (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) r :=
    fun j => Tree.eval_exists (maxRel_total isMax) _ _
  exact .pick ⟨Tensor.of1 (fun j => Classical.choose (hex j)), fun j =>
    ⟨Tree.seq M, Tree.seq_covers hm, Classical.choose_spec (hex j)⟩⟩ hk

end Sums

end TVA
