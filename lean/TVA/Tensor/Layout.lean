import TVA.Tensor.Index
import Mathlib.Tactic.Ring

/-!
# Layout operations

Operations that move elements without changing them:

* `tl.join`, `tl.split`, `tl.interleave` (`join`, `split`, `interleave`);
* `tl.reshape` with `can_reorder=False` (`reshape`), which keeps row-major order
  (`semantic.py:639-646`);
* `tl.permute` and `tl.trans` with `dims` (`permute`);
* `tl.broadcast_to` of a block (`broadcastTo`), with Triton's check of the shapes
  (`Shape.bcastOk`).

Each has a lemma reading one element of the result (`get_join_snoc`, `get_reshape`,
`get_permute`, `get_broadcastTo`), and lemmas linking it to the rank-2 forms of
`TVA.Tensor.Basic` (`permute_trans`, `broadcastTo_cols`, `broadcastTo_rows`).
-/
namespace TVA

namespace Tensor

variable {s : Shape} {α : Type}

/-! ### Join, split, interleave, reshape -/

/-- `tl.join(a, b)`: a new last dimension of size 2, with `a` at `0` and `b` at `1`
(`core.py:2160`). Two scalars give a tensor of shape `[2]`. -/
def join (a b : Tensor s α) : Tensor (s ++ [2]) α :=
  ⟨fun i => if (Idx.unsnoc i).2.val = 0 then a.get (Idx.unsnoc i).1 else b.get (Idx.unsnoc i).1⟩

/-- `tl.split(x)`: the two halves along a last dimension of size 2 (`core.py:2198`). -/
def split (x : Tensor (s ++ [2]) α) : Tensor s α × Tensor s α :=
  (⟨fun i => x.get (Idx.snoc i 0)⟩, ⟨fun i => x.get (Idx.snoc i 1)⟩)

/-- `tl.interleave(a, b)`: along the last dimension the elements of `a` and `b` alternate, `a`
first. Triton defines it as `join(a, b).reshape(a.shape[:-1] + [2 * a.shape[-1]])`
(`standard.py:540-560`); `interleave_eq_reshape_join` proves this definition agrees. -/
def interleave {n : Nat} (a b : Tensor (s ++ [n]) α) : Tensor (s ++ [2 * n]) α :=
  ⟨fun i =>
    let half : Fin n := ⟨(Idx.unsnoc i).2.val / 2, Nat.div_lt_of_lt_mul (Idx.unsnoc i).2.2⟩
    if (Idx.unsnoc i).2.val % 2 = 0 then a.get (Idx.snoc (Idx.unsnoc i).1 half)
    else b.get (Idx.snoc (Idx.unsnoc i).1 half)⟩

/-- `tl.reshape(x, s')` with `can_reorder=False`: the same elements in row-major order
(`semantic.py:639-646`). The sizes must agree (`h`). -/
def reshape (x : Tensor s α) (s' : Shape) (h : s'.prod = s.prod) : Tensor s' α :=
  ⟨fun i => x.get (Idx.ofPos s (Idx.pos s' i) (h ▸ Idx.pos_lt s' i))⟩

/-- Element `(i, t)` of `tl.join(a, b)` is `a[i]` when `t = 0` and `b[i]` when `t = 1`. -/
@[simp] theorem get_join_snoc (a b : Tensor s α) (i : Idx s) (t : Fin 2) :
    (join a b).get (Idx.snoc i t) = if t.val = 0 then a.get i else b.get i := by
  simp [join]

/-- `tl.split` of `tl.join(a, b)` gives back `a` and `b`. -/
@[simp] theorem split_join (a b : Tensor s α) : (join a b).split = (a, b) := by
  simp [split, join]

/-- `tl.join` of the two halves from `tl.split(x)` gives back `x`. -/
@[simp] theorem join_split (x : Tensor (s ++ [2]) α) : join x.split.1 x.split.2 = x := by
  ext i
  obtain ⟨j, t, rfl⟩ : ∃ j t, i = Idx.snoc j t := ⟨_, _, (Idx.snoc_unsnoc i).symm⟩
  rw [get_join_snoc]
  simp only [split]
  split_ifs with h0
  · have ht : t = 0 := Fin.ext h0
    rw [ht]
  · have ht : t = 1 := by
      apply Fin.ext
      have := t.2
      simp only [Fin.val_one]
      omega
    rw [ht]

/-- In the last dimension, `tl.interleave(a, b)` holds `a` at `t / 2` for even `t`, else `b`. -/
@[simp] theorem get_interleave_snoc {n : Nat} (a b : Tensor (s ++ [n]) α) (i : Idx s)
    (t : Fin (2 * n)) :
    (interleave a b).get (Idx.snoc i t) =
      if t.val % 2 = 0 then a.get (Idx.snoc i ⟨t.val / 2, Nat.div_lt_of_lt_mul t.2⟩)
      else b.get (Idx.snoc i ⟨t.val / 2, Nat.div_lt_of_lt_mul t.2⟩) := by
  simp [interleave]

/-- An element of a reshape is the input's element at the same row-major position. -/
theorem get_reshape (x : Tensor s α) (s' : Shape) (h : s'.prod = s.prod) (i : Idx s') :
    (x.reshape s' h).get i = x.get (Idx.ofPos s (Idx.pos s' i) (h ▸ Idx.pos_lt s' i)) := rfl

/-- Reshaping back gives the original tensor. -/
@[simp] theorem reshape_reshape (x : Tensor s α) (s' : Shape) (h : s'.prod = s.prod)
    (h' : s.prod = s'.prod) : (x.reshape s' h).reshape s h' = x := by
  ext i
  simp only [get_reshape]
  rw [Idx.ofPos_congr (Idx.pos_ofPos s' _ _) _ (Idx.pos_lt s i), Idx.ofPos_pos]

/-- `interleave` is Triton's `join` followed by the row-major reshape. -/
theorem interleave_eq_reshape_join {n : Nat} (a b : Tensor (s ++ [n]) α)
    (h : (s ++ [2 * n]).prod = ((s ++ [n]) ++ [2]).prod) :
    interleave a b = (join a b).reshape (s ++ [2 * n]) h := by
  ext idx
  obtain ⟨i, t, rfl⟩ : ∃ i t, idx = Idx.snoc i t := ⟨_, _, (Idx.snoc_unsnoc idx).symm⟩
  rw [get_interleave_snoc, get_reshape]
  have ht2 : t.val % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have hpos : Idx.pos ((s ++ [n]) ++ [2])
      (Idx.snoc (Idx.snoc i ⟨t.val / 2, Nat.div_lt_of_lt_mul t.2⟩) ⟨t.val % 2, ht2⟩) =
      Idx.pos (s ++ [2 * n]) (Idx.snoc i t) := by
    rw [Idx.pos_snoc, Idx.pos_snoc, Idx.pos_snoc]
    simp only
    have h2 := Nat.div_add_mod t.val 2
    generalize t.val / 2 = q at h2 ⊢
    generalize t.val % 2 = r at h2 ⊢
    rw [← h2]
    ring
  rw [Idx.ofPos_congr hpos.symm _ (Idx.pos_lt _ _), Idx.ofPos_pos, get_join_snoc]

/-! ### Permutations of the axes

`tl.permute(x, dims)` and `tl.trans(x, dims)` (`core.py:2070-2121`) reorder the axes without
changing the elements: dimension `k` of the result is dimension `dims[k]` of the input
(`semantic.py:701-709`). -/

private theorem getD_of_lt {β : Type} {l : List β} {k : Nat} {d : β} (hk : k < l.length) :
    l.getD k d = l[k] := by
  simp [List.getD_eq_getElem?_getD, hk]

/-- The coordinates `permute` reads are in range. -/
theorem permute_coord_lt (dims : List Nat) (h : dims.Perm (List.range s.length))
    (o : Idx (dims.map (Shape.dim s))) :
    ∀ j, j < s.length → o.coord (dims.idxOf j) < Shape.dim s j := by
  intro j hj
  have hmem : j ∈ dims := h.mem_iff.mpr (List.mem_range.mpr hj)
  have hp : dims.idxOf j < dims.length := List.idxOf_lt_length_of_mem hmem
  have hlt := Idx.coord_lt o (k := dims.idxOf j) (by simpa using hp)
  rw [Shape.dim_eq_getD, getD_of_lt (by simpa using hp), List.getElem_map,
    List.getElem_idxOf] at hlt
  exact hlt

/-- `tl.permute(x, dims)`, and `tl.trans(x, dims)`: the result has shape
`[s[dims[0]], s[dims[1]], …]`, and its element at `o` is the element of `x` whose coordinate
`dims[k]` is `o.coord k`, for every `k` (`get_permute`). `dims` must be a permutation of
`0, …, rank − 1` (Triton rejects any other list at compile time); for a shape of literal rank the
default proof `h` is found by itself.

* A rank-2 `tl.trans(x)` or `x.T` is `Tensor.trans`, which equals `x.permute [1, 0]`
  (`permute_trans`).
* `tl.trans(x)` with no `dims` on a tensor of rank `r ≥ 3` swaps the last two axes
  (`core.py:2091-2097`): the model writes `dims` as a literal list (`x.permute [0, 2, 1]` for
  rank 3), so that the result's shape reduces to the source's. -/
def permute (x : Tensor s α) (dims : List Nat)
    (h : dims.Perm (List.range s.length) := by
      (try simp only [List.length_cons, List.length_nil]); decide) :
    Tensor (dims.map (Shape.dim s)) α :=
  ⟨fun o => x.get (Idx.ofCoords s (fun j => o.coord (dims.idxOf j)) (permute_coord_lt dims h o))⟩

/-- The element of `x.permute dims` at `o` is `x` at the index `i` whose coordinate `dims[k]` is
`o.coord k` for every `k`. -/
theorem get_permute (x : Tensor s α) (dims : List Nat) (h : dims.Perm (List.range s.length))
    (o : Idx (dims.map (Shape.dim s))) (i : Idx s)
    (hi : ∀ k, k < dims.length → i.coord (dims.getD k 0) = o.coord k) :
    (x.permute dims h).get o = x.get i := by
  unfold permute
  refine congrArg x.get (Idx.ext_coord fun j => ?_)
  by_cases hj : j < s.length
  · rw [Idx.coord_ofCoords _ _ _ hj]
    have hmem : j ∈ dims := h.mem_iff.mpr (List.mem_range.mpr hj)
    have hp : dims.idxOf j < dims.length := List.idxOf_lt_length_of_mem hmem
    have := hi (dims.idxOf j) hp
    rw [getD_of_lt hp, List.getElem_idxOf] at this
    exact this.symm
  · rw [Idx.coord_of_ge _ (Nat.le_of_not_lt hj), Idx.coord_of_ge _ (Nat.le_of_not_lt hj)]

/-- On rank 2, `permute [1, 0]` is the transpose `Tensor.trans`. -/
theorem permute_trans {m n : Nat} (x : Tensor [m, n] α) : x.permute [1, 0] = x.trans := by
  ext ⟨j, i, ⟨⟩⟩
  exact (get_permute x [1, 0] _ (j, i, ()) (i, j, ()) fun k hk =>
    match k, hk with
    | 0, _ => rfl
    | 1, _ => rfl).trans rfl

end Tensor

/-! ### Broadcasting to a shape -/

/-- Triton's check that a block of shape `s` broadcasts to `t` (`broadcast_impl_shape`,
`semantic.py:711-725`): the same rank, and each dimension of `s` equal to `t`'s or `1`. Triton
rejects every other pair on the host. -/
def Shape.bcastOk : Shape → Shape → Bool
  | [], [] => true
  | a :: s, b :: t => (a == b || a == 1) && Shape.bcastOk s t
  | _, _ => false

/-- A shape that passes Triton's broadcast check has the same rank as the target shape. -/
theorem Shape.bcastOk_length : ∀ {s t : Shape}, Shape.bcastOk s t = true → s.length = t.length
  | [], [], _ => rfl
  | _ :: s, _ :: t, h => by
      simp only [Shape.bcastOk, Bool.and_eq_true] at h
      simp [Shape.bcastOk_length (s := s) (t := t) h.2]
  | [], _ :: _, h => by simp [Shape.bcastOk] at h
  | _ :: _, [], h => by simp [Shape.bcastOk] at h

/-- If `s` broadcasts to `t`, each dimension of `s` is `t`'s or `1` (or beyond the rank). -/
theorem Shape.bcastOk_dim : ∀ {s t : Shape}, Shape.bcastOk s t = true → ∀ k,
    Shape.dim s k = Shape.dim t k ∨ Shape.dim s k = 1 ∨ s.length ≤ k
  | [], [], _, k => Or.inr (Or.inr (Nat.zero_le k))
  | a :: s, b :: t, h, k => by
      simp only [Shape.bcastOk, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
      cases k with
      | zero =>
        rcases h.1 with h1 | h1
        · exact Or.inl h1
        · exact Or.inr (Or.inl h1)
      | succ k =>
        rcases Shape.bcastOk_dim (s := s) (t := t) h.2 k with h2 | h2 | h2
        · exact Or.inl h2
        · exact Or.inr (Or.inl h2)
        · exact Or.inr (Or.inr (by simpa using h2))
  | [], _ :: _, h, _ => by simp [Shape.bcastOk] at h
  | _ :: _, [], h, _ => by simp [Shape.bcastOk] at h

namespace Tensor

variable {s : Shape} {α : Type}

/-- The coordinates `broadcastTo` reads are in range. -/
theorem bcast_coord_lt {t : Shape} (h : Shape.bcastOk s t = true) (o : Idx t) :
    ∀ k, k < s.length → (if Shape.dim s k = 1 then 0 else o.coord k) < Shape.dim s k := by
  intro k hk
  by_cases h1 : Shape.dim s k = 1
  · simp only [h1, ite_true]; exact Nat.zero_lt_one
  · simp only [h1, ite_false]
    rcases Shape.bcastOk_dim h k with h2 | h2 | h2
    · rw [h2]; exact Idx.coord_lt o (by rw [← Shape.bcastOk_length h]; exact hk)
    · exact absurd h2 h1
    · exact absurd hk (Nat.not_lt_of_le h2)

/-- `tl.broadcast_to(x, t)`, and the method forms `x.broadcast_to(t)`, `x.broadcast_to(*t)`
(`core.py:2049-2066`, `semantic.py:711-725`), of a block `x` of shape `s`: the block of shape `t`
whose element at `o` is `x`'s element at `o`'s coordinates, except that each dimension of size
`1` in `s` is read at `0` (`get_broadcastTo`). Triton accepts only a `t` of the same rank whose
every dimension equals `s`'s or where `s` has `1` (`Shape.bcastOk`, the proof `h`; for shapes of
literal sizes the default proof is found by itself). A scalar's broadcast is `Tensor.full`
(`semantic.py:629-634`). `col`, `row`, `bcastCols` and `bcastRows` are special cases. -/
def broadcastTo (x : Tensor s α) (t : Shape) (h : Shape.bcastOk s t = true := by decide) :
    Tensor t α :=
  ⟨fun o => x.get (Idx.ofCoords s (fun k => if Shape.dim s k = 1 then 0 else o.coord k)
    (bcast_coord_lt h o))⟩

/-- The element of `x.broadcastTo t` at `o` is `x` at the index `i` whose coordinate `k` is `0`
where `s` has size `1`, and `o`'s coordinate `k` otherwise. -/
theorem get_broadcastTo (x : Tensor s α) (t : Shape) (h : Shape.bcastOk s t = true) (o : Idx t)
    (i : Idx s)
    (hi : ∀ k, k < s.length → i.coord k = if Shape.dim s k = 1 then 0 else o.coord k) :
    (x.broadcastTo t h).get o = x.get i := by
  unfold broadcastTo
  refine congrArg x.get (Idx.ext_coord fun k => ?_)
  by_cases hk : k < s.length
  · rw [Idx.coord_ofCoords _ _ _ hk, hi k hk]
  · rw [Idx.coord_of_ge _ (Nat.le_of_not_lt hk), Idx.coord_of_ge _ (Nat.le_of_not_lt hk)]

/-- On rank 2, broadcasting a column `[m, 1]` to `[m, n]` is `bcastCols`. -/
theorem broadcastTo_cols {m n : Nat} (x : Tensor [m, 1] α)
    (h : Shape.bcastOk [m, 1] [m, n] = true) : x.broadcastTo [m, n] h = x.bcastCols n := by
  ext ⟨i, j, ⟨⟩⟩
  rw [get_broadcastTo x [m, n] h (i, j, ()) (i, ⟨0, Nat.zero_lt_one⟩, ()) fun k hk => ?_]
  · rfl
  · match k, hk with
    | 0, _ =>
      show i.1 = if m = 1 then 0 else i.1
      split
      · next hm => have := i.2; omega
      · rfl
    | 1, _ => rfl

/-- On rank 2, broadcasting a row `[1, n]` to `[m, n]` is `bcastRows`. -/
theorem broadcastTo_rows {m n : Nat} (x : Tensor [1, n] α)
    (h : Shape.bcastOk [1, n] [m, n] = true) : x.broadcastTo [m, n] h = x.bcastRows m := by
  ext ⟨i, j, ⟨⟩⟩
  rw [get_broadcastTo x [m, n] h (i, j, ()) (⟨0, Nat.zero_lt_one⟩, j, ()) fun k hk => ?_]
  · rfl
  · match k, hk with
    | 0, _ => rfl
    | 1, _ =>
      show j.1 = if n = 1 then 0 else j.1
      split
      · next hn => have := j.2; omega
      · rfl

end Tensor

end TVA
