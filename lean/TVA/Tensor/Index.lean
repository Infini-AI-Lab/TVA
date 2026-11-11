import TVA.Tensor.Basic

/-!
# Coordinates of an index

Operations that work along any axis (scans, sorting, block pointers, `tl.permute`,
`tl.broadcast_to`) read an index through its coordinates:

* `Shape.dim s k`: the size of dimension `k`;
* `Idx.coord i k`: the `k`-th coordinate of `i`; `Idx.set i k t` replaces it by `t`;
  `Idx.ofCoords` builds an index from its coordinates;
* `Idx.snoc`/`Idx.unsnoc`: append or split off the last coordinate (`tl.join`, `tl.split`);
* `Idx.pos`/`Idx.ofPos`: the row-major position of an index and back (`tl.reshape`).

Used by `TVA.Tensor.Layout` and the axis forms of `TVA.Triton.Reduce`.
-/
namespace TVA

/-- The size of dimension `k` of a shape (`0` beyond the rank). -/
def Shape.dim : Shape → Nat → Nat
  | [], _ => 0
  | n :: _, 0 => n
  | _ :: s, k + 1 => Shape.dim s k

/-- `Shape.dim` is the list's entry, `0` beyond the rank. -/
theorem Shape.dim_eq_getD : ∀ (s : Shape) (k : Nat), Shape.dim s k = s.getD k 0
  | [], _ => rfl
  | _ :: _, 0 => rfl
  | _ :: s, k + 1 => Shape.dim_eq_getD s k

namespace Idx

/-- The `k`-th coordinate of an index (`0` beyond the rank). -/
def coord : {s : Shape} → Idx s → Nat → Nat
  | [], (), _ => 0
  | _ :: _, (i, _), 0 => i.1
  | _ :: s, (_, j), k + 1 => coord (s := s) j k

/-- The index with its `k`-th coordinate replaced by `t`; unchanged when `t` is out of range. -/
def set : {s : Shape} → Idx s → Nat → Nat → Idx s
  | [], (), _, _ => ()
  | n :: _, (i, j), 0, t => (if h : t < n then ⟨t, h⟩ else i, j)
  | _ :: s, (i, j), k + 1, t => (i, set (s := s) j k t)

/-- A coordinate is below its dimension. -/
theorem coord_lt : ∀ {s : Shape} (i : Idx s) {k : Nat}, k < s.length → i.coord k < Shape.dim s k
  | [], (), _, h => absurd h (Nat.not_lt_zero _)
  | _ :: _, (i, _), 0, _ => i.2
  | _ :: s, (_, j), _ + 1, h => coord_lt (s := s) j (Nat.lt_of_succ_lt_succ h)

/-- A coordinate beyond the rank is `0`. -/
theorem coord_of_ge : ∀ {s : Shape} (i : Idx s) {k : Nat}, s.length ≤ k → i.coord k = 0
  | [], (), _, _ => rfl
  | _ :: _, (_, _), 0, h => absurd h (by simp)
  | _ :: s, (_, j), k + 1, h => coord_of_ge (s := s) j (by simpa using h)

/-- After setting coordinate `k` to a value below its dimension, coordinate `k` is that value. -/
theorem coord_set_self : ∀ {s : Shape} (i : Idx s) {k t : Nat}, t < Shape.dim s k →
    (i.set k t).coord k = t
  | [], (), _, _, h => absurd h (Nat.not_lt_zero _)
  | n :: _, (i, j), 0, t, h => by
      have h' : t < n := h
      simp [set, coord, h']
  | _ :: s, (_, j), k + 1, _, h => coord_set_self (s := s) j h

/-- Setting coordinate `k` leaves every other coordinate unchanged. -/
theorem coord_set_ne : ∀ {s : Shape} (i : Idx s) {k k' t : Nat}, k' ≠ k →
    (i.set k t).coord k' = i.coord k'
  | [], (), _, _, _, _ => rfl
  | _ :: _, (_, _), 0, 0, _, h => absurd rfl h
  | _ :: _, (_, _), 0, _ + 1, _, _ => rfl
  | _ :: _, (_, _), _ + 1, 0, _, _ => rfl
  | _ :: s, (_, j), _ + 1, _ + 1, _, h =>
      coord_set_ne (s := s) j (fun e => h (congrArg (· + 1) e))

/-- Setting coordinate `k` to its own value leaves the index unchanged. -/
theorem set_coord_self : ∀ {s : Shape} (i : Idx s) (k : Nat), i.set k (i.coord k) = i
  | [], (), _ => rfl
  | _ :: _, (i, j), 0 => by simp [set, coord, i.2]; rfl
  | _ :: s, (i, j), k + 1 => by simp only [set, coord, set_coord_self (s := s) j k]; rfl

/-- Setting coordinate `k` twice keeps only the second value (when it is below the dimension). -/
theorem set_set : ∀ {s : Shape} (i : Idx s) {k t t' : Nat}, t' < Shape.dim s k →
    (i.set k t).set k t' = i.set k t'
  | [], (), _, _, _, _ => rfl
  | n :: _, (i, j), 0, t, t', h => by
      have h' : t' < n := h
      simp [set, h']; rfl
  | _ :: s, (i, j), k + 1, _, _, h => by simp only [set, set_set (s := s) j h]; rfl

/-- Two indices with the same coordinates are equal. -/
theorem ext_coord : ∀ {s : Shape} {i j : Idx s}, (∀ k, i.coord k = j.coord k) → i = j
  | [], (), (), _ => rfl
  | _ :: _, (a, i), (b, j), h => by
      have h0 := h 0
      simp only [coord] at h0
      rw [Fin.ext h0, ext_coord (fun k => h (k + 1))]

/-- `i` and `j` lie on one line along axis `k` (they agree off `k`): `j` is `i` with its `k`-th
coordinate set to `j`'s. -/
theorem eq_set_of_coord : ∀ {s : Shape} {i j : Idx s} {k : Nat},
    (∀ k', k' ≠ k → i.coord k' = j.coord k') → j = i.set k (j.coord k)
  | [], (), (), _, _ => rfl
  | _ :: _, (a, i), (b, j), 0, h => by
      have hrest : i = j := ext_coord fun k' => h (k' + 1) (Nat.succ_ne_zero _)
      simp [set, coord, b.2, hrest]
      rfl
  | _ :: s, (a, i), (b, j), k + 1, h => by
      have ha : a = b := Fin.ext (h 0 (Nat.succ_ne_zero k).symm)
      have hr := eq_set_of_coord (s := s) (i := i) (j := j) (k := k)
        (fun k' hk => h (k' + 1) (fun e => hk (Nat.succ_injective e)))
      simp only [set, coord, ha] at hr ⊢
      rw [← hr]

/-- The index with coordinates `c 0, c 1, …`, each below its dimension. -/
def ofCoords : (s : Shape) → (c : Nat → Nat) → (∀ k, k < s.length → c k < Shape.dim s k) → Idx s
  | [], _, _ => ()
  | _ :: s, c, h => (⟨c 0, h 0 (Nat.succ_pos _)⟩,
      ofCoords s (fun k => c (k + 1)) (fun k hk => h (k + 1) (Nat.succ_lt_succ hk)))

/-- Coordinate `k` of the index built by `ofCoords` from `c` is `c k`. -/
theorem coord_ofCoords : ∀ (s : Shape) (c : Nat → Nat)
    (h : ∀ k, k < s.length → c k < Shape.dim s k) {k : Nat}, k < s.length →
    (ofCoords s c h).coord k = c k
  | [], _, _, _, hk => absurd hk (Nat.not_lt_zero _)
  | _ :: _, _, _, 0, _ => rfl
  | _ :: s, c, _, _ + 1, hk =>
      coord_ofCoords s (fun k => c (k + 1)) _ (Nat.lt_of_succ_lt_succ hk)

/-! ### The last coordinate -/

/-- The index with a last coordinate `t` appended. -/
def snoc {n : Nat} : {s : Shape} → Idx s → Fin n → Idx (s ++ [n])
  | [], (), t => (t, ())
  | _ :: _, (i, j), t => (i, snoc j t)

/-- The index without its last coordinate, and that coordinate. -/
def unsnoc {n : Nat} : {s : Shape} → Idx (s ++ [n]) → Idx s × Fin n
  | [], (t, ()) => ((), t)
  | _ :: _, (i, j) => ((i, (unsnoc j).1), (unsnoc j).2)

/-- Splitting the last coordinate off `snoc i t` gives back `i` and `t`. -/
@[simp] theorem unsnoc_snoc {n : Nat} : ∀ {s : Shape} (i : Idx s) (t : Fin n),
    unsnoc (snoc i t) = (i, t)
  | [], (), _ => rfl
  | _ :: _, (i, j), t => by simp only [snoc, unsnoc, unsnoc_snoc j t]

/-- Appending back the last coordinate split off by `unsnoc` gives the original index. -/
@[simp] theorem snoc_unsnoc {n : Nat} : ∀ {s : Shape} (i : Idx (s ++ [n])),
    snoc (unsnoc i).1 (unsnoc i).2 = i
  | [], (_, ()) => rfl
  | _ :: _, (i, j) => by simp only [snoc, unsnoc, snoc_unsnoc j]; rfl

/-! ### Row-major positions -/

/-- The row-major position of an index (`0 ≤ pos < s.prod`). -/
def pos : (s : Shape) → Idx s → Nat
  | [], () => 0
  | _ :: s, (i, j) => i.val * s.prod + pos s j

/-- If a position lies below the size of `n :: s`, the inner shape `s` has a nonzero size. -/
theorem prod_pos_of_lt {n : Nat} {s : Shape} {p : Nat} (h : p < (n :: s).prod) : 0 < s.prod := by
  rw [List.prod_cons] at h
  rcases Nat.eq_zero_or_pos s.prod with h0 | h0
  · rw [h0, Nat.mul_zero] at h; exact absurd h (Nat.not_lt_zero _)
  · exact h0

/-- The row-major position of an index lies below the number of elements of its shape. -/
theorem pos_lt : ∀ (s : Shape) (i : Idx s), pos s i < s.prod
  | [], () => Nat.one_pos
  | n :: s, (i, j) => by
      rw [List.prod_cons]
      simp only [pos]
      calc i.val * s.prod + pos s j < i.val * s.prod + s.prod :=
            Nat.add_lt_add_left (pos_lt s j) _
        _ = (i.val + 1) * s.prod := by rw [Nat.succ_mul]
        _ ≤ n * s.prod := Nat.mul_le_mul_right _ i.2

/-- The index at row-major position `p`. -/
def ofPos : (s : Shape) → (p : Nat) → p < s.prod → Idx s
  | [], _, _ => ()
  | n :: s, p, h =>
      (⟨p / s.prod, (Nat.div_lt_iff_lt_mul (prod_pos_of_lt h)).mpr
          (by rw [List.prod_cons] at h; exact h)⟩,
        ofPos s (p % s.prod) (Nat.mod_lt _ (prod_pos_of_lt h)))

/-- `ofPos` at equal positions gives equal indices: rewrites a position under its bound proof. -/
theorem ofPos_congr {s : Shape} {p q : Nat} (h : p = q) (hp : p < s.prod) (hq : q < s.prod) :
    ofPos s p hp = ofPos s q hq := by subst h; rfl

/-- The row-major position of the index at position `p` is `p`. -/
theorem pos_ofPos : ∀ (s : Shape) (p : Nat) (h : p < s.prod), pos s (ofPos s p h) = p
  | [], p, h => by
      have h' : p < 1 := h
      show 0 = p
      omega
  | n :: s, p, h => by
      simp only [ofPos, pos, pos_ofPos s]
      exact Nat.div_add_mod' p s.prod

/-- The index at the row-major position of `i` is `i`. -/
theorem ofPos_pos : ∀ (s : Shape) (i : Idx s) (h : pos s i < s.prod),
    ofPos s (pos s i) h = i
  | [], (), _ => rfl
  | n :: s, (i, j), h => by
      have hP : 0 < s.prod := prod_pos_of_lt h
      have hj := pos_lt s j
      simp only [ofPos, pos]
      have h1 : (i.val * s.prod + pos s j) / s.prod = i.val := by
        rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt hj, Nat.zero_add]
      have h2 : (i.val * s.prod + pos s j) % s.prod = pos s j := by
        rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
      congr 1
      · exact Fin.ext h1
      · rw [ofPos_congr h2 _ hj, ofPos_pos s j]

/-- The position of an index with a last coordinate appended. -/
theorem pos_snoc {n : Nat} : ∀ {s : Shape} (i : Idx s) (t : Fin n),
    pos (s ++ [n]) (snoc i t) = pos s i * n + t
  | [], (), t => by simp [pos, snoc]
  | m :: s, (i, j), t => by
      show i.val * (s ++ [n]).prod + pos (s ++ [n]) (snoc j t) = _
      rw [pos_snoc j t, List.prod_append, List.prod_singleton]
      simp only [pos, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]

end Idx

end TVA
