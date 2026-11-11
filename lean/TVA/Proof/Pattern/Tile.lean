import TVA.Proof.Pattern.TileOffset

/-!
# Two-dimensional tiles: masks, footprints, loads and stores

The access of every matrix operand:

```python
ptrs = base + rows[:, None] * s0 + cols[None, :] * s1
mask = (rows < M)[:, None] & (cols < N)[None, :]     # or one factor, or none
x = tl.load(ptrs, mask=mask, other=…)
tl.store(ptrs, x, mask=mask)
```

Row lane `i` has a natural value `ρ i` and column lane `j` a value `κ j`: `ρ i = r0 + i` for
`rows = r0 + tl.arange(0, BM)`, `ρ i = (r0 + i) % M` for rows wrapped by `% M`. With
`base = b.ptr e`, lane `(i, j)` touches element `e + ρ i * s0 + κ j * s1` of `b` when the mask
enables it, `ρ i < M` and `κ j < N`. A proof gives two facts about the access:

* the pointers on the enabled lanes, `hp` (`ptr_lane_on` and the other lemmas of
  `TVA.Proof.Pattern.TileOffset`);
* the mask, `Tile2.MaskAt mk ρ κ M N` (`maskAt_and`, `maskAt_rows`, `maskAt_cols`, `maskAt_none`,
  `maskAt_of_get`).

From them, `safe_load_tileAt` gives each enabled lane its element and `safe_store_tileAt` writes
them; the footprint is `b.tileAt e s0 s1 ρ κ M N` (`b.tile2 e s0 s1 r0 c0 BM BN M N` for rows
`r0 + i` and columns `c0 + j`). The matrix lies in its buffer when it is empty (`M = 0` or `N = 0`,
as for an empty sequence of a varlen batch at the end of its buffer) or its last element does
(`M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len`). Without a row (column) mask, take an
`M` (`N`) above every row (column).

A store needs a layout that does not alias (`Tile2.Inj`: row-major `inj_rowMajor`, column-major
`inj_colMajor`) and distinct enabled rows and columns (`add_inj` for `r0 + i`), so that its lanes
write distinct elements. Tiles whose rows or columns are apart do not overlap
(`Buf.tileAt_disjoint`, `Buf.tile2_disjoint`, `Buf.tile2_disjoint_grid`), and every tile lies in
the whole matrix (`Buf.tileAt_sub_matrix`).
-/
namespace TVA

namespace Tile2

/-! ## Masks -/

/-- Whether the mask `mk` of a `[BM, BN]` access enables lane `(i, j)`; without a mask every lane
is enabled. -/
def on {BM BN : Nat} (mk : Option (Tensor [BM, BN] Bool)) (i : Fin BM) (j : Fin BN) : Bool :=
  match mk with
  | none => true
  | some m => m.get2 i j

/-- The mask `mk` enables lane `(i, j)` exactly when its row `ρ i` is below `M` and its column
`κ j` below `N`. -/
def MaskAt {BM BN : Nat} (mk : Option (Tensor [BM, BN] Bool)) (ρ : Fin BM → Nat)
    (κ : Fin BN → Nat) (M N : Nat) : Prop :=
  ∀ i j, on mk i j = (decide (ρ i < M) && decide (κ j < N))

section Masks
variable {w BM BN M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

private theorem lt_lane {x mv : IntV true w} {k K : Nat} (hx : x.toInt = (k : Int))
    (hm : mv.toInt = (K : Int)) : IntV.lt x mv = decide (k < K) := by
  rw [IntV.lt_eq, hx, hm]; simp only [Nat.cast_lt]

/-- Any mask, from its lanes. -/
theorem maskAt_of_get {m : Tensor [BM, BN] Bool}
    (h : ∀ i j, m.get2 i j = (decide (ρ i < M) && decide (κ j < N))) :
    MaskAt (some m) ρ κ M N := h

/-- **No mask**: every lane is enabled; it is the mask of a tile whose rows are below `M` and
columns below `N` (for rows `(r0 + i) % M`, `Nat.mod_lt`). -/
theorem maskAt_none (hM : ∀ i, ρ i < M) (hN : ∀ j, κ j < N) : MaskAt none ρ κ M N := by
  intro i j
  simp only [on]; symm; simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨hM i, hN j⟩

/-- **`(rows < M)[:, None] & (cols < N)[None, :]`**
(`((rows.col BN).ltS mv).and ((cols.row BM).ltS nv)`), with `mv`, `nv` of values `M`, `N`. -/
theorem maskAt_and {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)}
    {mv nv : IntV true w} (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int)) (hmv : mv.toInt = M)
    (hnv : nv.toInt = N) :
    MaskAt (some (((rows.col BN).ltS mv).and ((cols.row BM).ltS nv))) ρ κ M N := by
  intro i j
  show (IntV.lt (rows.get1 i) mv && IntV.lt (cols.get1 j) nv) = _
  rw [lt_lane (hr i) hmv, lt_lane (hc j) hnv]

/-- **The row mask alone**, `(rows < M)[:, None]` (`(rows.col BN).ltS mv`), when every column is
below `N`. -/
theorem maskAt_rows {rows : Tensor [BM] (IntV true w)} {mv : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int)) (hmv : mv.toInt = M)
    (hN : ∀ j, κ j < N) : MaskAt (some ((rows.col BN).ltS mv)) ρ κ M N := by
  intro i j
  show IntV.lt (rows.get1 i) mv = _
  rw [lt_lane (hr i) hmv, decide_eq_true (hN j), Bool.and_true]

/-- **The column mask alone**, `(cols < N)[None, :]` (`(cols.row BM).ltS nv`), when every row is
below `M` (rows `(r0 + i) % M`). -/
theorem maskAt_cols {cols : Tensor [BN] (IntV true w)} {nv : IntV true w}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int)) (hnv : nv.toInt = N)
    (hM : ∀ i, ρ i < M) : MaskAt (some ((cols.row BM).ltS nv)) ρ κ M N := by
  intro i j
  show IntV.lt (cols.get1 j) nv = _
  rw [lt_lane (hc j) hnv, decide_eq_true (hM i), Bool.true_and]

end Masks

/-! ## Elements -/

/-- Element `x` is in the tile: `x = e + ρ i * s0 + κ j * s1` for a lane `(i, j)` whose row is
below `M` and column below `N`. -/
def ElemAt {BM BN : Nat} (e s0 s1 : Nat) (ρ : Fin BM → Nat) (κ : Fin BN → Nat) (M N x : Nat) :
    Prop :=
  ∃ i j, ρ i < M ∧ κ j < N ∧ x = e + ρ i * s0 + κ j * s1

/-- Element `x` is in the `M` by `N` matrix at element `e` with strides `s0`, `s1`. -/
def InMatrix (e s0 s1 M N x : Nat) : Prop :=
  ∃ r c, r < M ∧ c < N ∧ x = e + r * s0 + c * s1

/-- The layout does not alias: distinct `(row, column)` pairs below `(M, N)` have distinct offsets
`row * s0 + col * s1`. -/
def Inj (s0 s1 M N : Nat) : Prop :=
  ∀ r c r' c', r < M → c < N → r' < M → c' < N → r * s0 + c * s1 = r' * s0 + c' * s1 →
    r = r' ∧ c = c'

/-- **Row-major** (padded) layouts do not alias: a row's `N` elements, `s1 > 0` apart, fit in the
row stride (`N * s1 ≤ s0`). -/
theorem inj_rowMajor {s0 s1 M N : Nat} (hs1 : 0 < s1) (hN : N * s1 ≤ s0) : Inj s0 s1 M N := by
  intro r c r' c' _ hc _ hc' h
  have hlt : ∀ k, k < N → k * s1 < s0 := fun k hk =>
    lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right hk hs1) hN
  have hs0 : 0 < s0 := lt_of_le_of_lt (Nat.zero_le _) (hlt c hc)
  have hdiv : ∀ k q, k < N → (q * s0 + k * s1) / s0 = q := fun k q hk => by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hs0, Nat.div_eq_of_lt (hlt k hk), Nat.zero_add]
  have hr : r = r' := by rw [← hdiv c r hc, h, hdiv c' r' hc']
  subst hr
  exact ⟨rfl, Nat.eq_of_mul_eq_mul_right hs1 (by omega)⟩

/-- Row-major with a unit column stride does not alias when `N ≤ s0`. -/
theorem inj_rowMajor1 {s0 M N : Nat} (hN : N ≤ s0) : Inj s0 1 M N :=
  inj_rowMajor Nat.one_pos (by simpa using hN)

/-- **Column-major** (padded) layouts do not alias: a column's `M` elements, `s0 > 0` apart, fit
in the column stride (`M * s0 ≤ s1`). -/
theorem inj_colMajor {s0 s1 M N : Nat} (hs0 : 0 < s0) (hM : M * s0 ≤ s1) : Inj s0 s1 M N := by
  intro r c r' c' hr hc hr' hc' h
  have := inj_rowMajor (M := N) hs0 hM c r c' r' hc hr hc' hr' (by linarith)
  exact ⟨this.2, this.1⟩

/-- Column-major with a unit row stride does not alias when `M ≤ s1`. -/
theorem inj_colMajor1 {s1 M N : Nat} (hM : M ≤ s1) : Inj 1 s1 M N :=
  inj_colMajor Nat.one_pos (by simpa using hM)

/-- Rows `r0 + i` (columns `c0 + j`) are distinct: the `hρ` (`hκ`) of `safe_store_tileAt`. -/
theorem add_inj {B r0 L : Nat} :
    ∀ i i' : Fin B, r0 + (i : Nat) < L → r0 + (i' : Nat) < L → r0 + (i : Nat) = r0 + i' → i = i' :=
  fun _ _ _ _ h => Fin.ext (by omega)

section Elems
variable {e s0 s1 BM BN M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

/-- An element of a tile is an element of the matrix. -/
theorem elemAt_matrix {x : Nat} (h : ElemAt e s0 s1 ρ κ M N x) : InMatrix e s0 s1 M N x := by
  obtain ⟨i, j, hiM, hjN, rfl⟩ := h
  exact ⟨ρ i, κ j, hiM, hjN, rfl⟩

/-- The matrix's elements are below `len` when the matrix is empty or its last element is. -/
theorem matrix_lt {x len : Nat} (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < len)
    (h : InMatrix e s0 s1 M N x) : x < len := by
  obtain ⟨r, c, hr, hc, rfl⟩ := h
  rcases hlen with h0 | h0 | hlen
  · omega
  · omega
  have h1 : r * s0 ≤ (M - 1) * s0 := Nat.mul_le_mul_right s0 (by omega)
  have h2 : c * s1 ≤ (N - 1) * s1 := Nat.mul_le_mul_right s1 (by omega)
  omega

/-- **Tiles whose rows (or columns) differ have no common element**, when the layout does not
alias: every row of one differs from every row of the other, or the same for columns. -/
theorem elemAt_disjoint {BM' BN' : Nat} {ρ' : Fin BM' → Nat} {κ' : Fin BN' → Nat}
    (hinj : Inj s0 s1 M N) (hap : (∀ i i', ρ i ≠ ρ' i') ∨ (∀ j j', κ j ≠ κ' j')) :
    Disjoint {x | ElemAt e s0 s1 ρ κ M N x} {x | ElemAt e s0 s1 ρ' κ' M N x} := by
  rw [Set.disjoint_left]
  rintro x ⟨i, j, hi, hj, rfl⟩ ⟨i', j', hi', hj', h⟩
  obtain ⟨h1, h2⟩ := hinj _ _ _ _ hi hj hi' hj' (by omega)
  rcases hap with hap | hap
  · exact hap i i' h1
  · exact hap j j' h2

end Elems

end Tile2

open Tile2

namespace Buf

variable {d : DType} {e s0 s1 BM BN M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

/-- The bytes of the elements of `b` in a 2-D tile (`Tile2.ElemAt`): the footprint of a tile
access. -/
def tileAt {BM BN : Nat} (b : Buf d) (e s0 s1 : Nat) (ρ : Fin BM → Nat) (κ : Fin BN → Nat)
    (M N : Nat) : Set Int :=
  b.elems {x | ElemAt e s0 s1 ρ κ M N x}

/-- The footprint of the tile with rows `r0 + i`, `i < BM`, and columns `c0 + j`, `j < BN`. -/
abbrev tile2 (b : Buf d) (e s0 s1 r0 c0 BM BN M N : Nat) : Set Int :=
  b.tileAt e s0 s1 (fun i : Fin BM => r0 + i) (fun j : Fin BN => c0 + j) M N

/-- The bytes of the `M` by `N` matrix at element `e` of `b` with strides `s0`, `s1`. -/
def matrix (b : Buf d) (e s0 s1 M N : Nat) : Set Int := b.elems {x | InMatrix e s0 s1 M N x}

/-- A tile lies in its matrix. Use it when a footprint holds the whole matrix. -/
theorem tileAt_sub_matrix (b : Buf d) : b.tileAt e s0 s1 ρ κ M N ⊆ b.matrix e s0 s1 M N :=
  b.elems_mono fun _ hx => elemAt_matrix hx

/-- The matrix lies in `b` when it is empty or its last element lies in `b`. -/
theorem matrix_sub_bytes (b : Buf d)
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len) :
    b.matrix e s0 s1 M N ⊆ b.bytes :=
  b.elems_sub_bytes fun _ hx => matrix_lt hlen hx

/-- A tile lies in `b` when its matrix is empty or the matrix's last element lies in `b`. -/
theorem tileAt_sub_bytes (b : Buf d)
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len) :
    b.tileAt e s0 s1 ρ κ M N ⊆ b.bytes :=
  (b.tileAt_sub_matrix).trans (b.matrix_sub_bytes hlen)

/-- The bytes of a tile element lie in the tile's bytes. -/
theorem span_sub_tileAt (b : Buf d) {x : Nat} (h : ElemAt e s0 s1 ρ κ M N x) :
    Mem.span d (b.ptr (x : Int)).addr ⊆ b.tileAt e s0 s1 ρ κ M N :=
  b.span_sub_elems h

/-- **Two tiles of one buffer do not overlap** when the layout does not alias and every row of one
differs from every row of the other (or the same for columns). -/
theorem tileAt_disjoint (b : Buf d) {BM' BN' : Nat} {ρ' : Fin BM' → Nat} {κ' : Fin BN' → Nat}
    (hinj : Inj s0 s1 M N) (hap : (∀ i i', ρ i ≠ ρ' i') ∨ (∀ j j', κ j ≠ κ' j')) :
    Disjoint (b.tileAt e s0 s1 ρ κ M N) (b.tileAt e s0 s1 ρ' κ' M N) :=
  b.elems_disjoint (elemAt_disjoint hinj hap)

/-- **Two tiles `r0 + i`, `c0 + j` whose rows or columns are apart by at least the block** do not
overlap, when the layout does not alias. -/
theorem tile2_disjoint (b : Buf d) {r0 c0 r0' c0' BM' BN' : Nat} (hinj : Inj s0 s1 M N)
    (hap : r0 + BM ≤ r0' ∨ r0' + BM' ≤ r0 ∨ c0 + BN ≤ c0' ∨ c0' + BN' ≤ c0) :
    Disjoint (b.tile2 e s0 s1 r0 c0 BM BN M N) (b.tile2 e s0 s1 r0' c0' BM' BN' M N) := by
  refine b.tileAt_disjoint hinj ?_
  rcases hap with h | h | h | h
  · exact Or.inl fun i i' => by have := i.isLt; have := i'.isLt; omega
  · exact Or.inl fun i i' => by have := i.isLt; have := i'.isLt; omega
  · exact Or.inr fun j j' => by have := j.isLt; have := j'.isLt; omega
  · exact Or.inr fun j j' => by have := j.isLt; have := j'.isLt; omega

private theorem block_ne {p q B i i' : Nat} (hpq : p ≠ q) (hi : i < B) (hi' : i' < B) :
    p * B + i ≠ q * B + i' := by
  intro h
  rcases Nat.lt_or_gt_of_ne hpq with h' | h'
  · have : (p + 1) * B ≤ q * B := Nat.mul_le_mul_right B h'
    rw [Nat.succ_mul] at this; omega
  · have : (q + 1) * B ≤ p * B := Nat.mul_le_mul_right B h'
    rw [Nat.succ_mul] at this; omega

/-- **The tiles of two programs of a 2-D grid do not overlap**: programs `(pm, pn) ≠ (pm', pn')`
handle the tiles at `(pm * BM, pn * BN)` and `(pm' * BM, pn' * BN)`, when the layout does not
alias. -/
theorem tile2_disjoint_grid (b : Buf d) {pm pn pm' pn' : Nat} (hinj : Inj s0 s1 M N)
    (hne : pm ≠ pm' ∨ pn ≠ pn') :
    Disjoint (b.tile2 e s0 s1 (pm * BM) (pn * BN) BM BN M N)
      (b.tile2 e s0 s1 (pm' * BM) (pn' * BN) BM BN M N) := by
  refine b.tileAt_disjoint hinj ?_
  rcases hne with h | h
  · exact Or.inl fun i i' => block_ne h i.isLt i'.isLt
  · exact Or.inr fun j j' => block_ne h j.isLt j'.isLt

end Buf

/-! ## Accesses -/

section Access
variable {d : DType} {e s0 s1 BM BN M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

/-- **The lanes of a 2-D tile access, as buffer elements**: lane `(i, j)` touches element
`e + ρ i * s0 + κ j * s1` when `ρ i < M` and `κ j < N`, from the pointers on the enabled lanes
(`hp`) and the mask (`MaskAt`). -/
theorem addrs_tileAt (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)}
    (hp : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      ptrs.get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int))
    (hm : MaskAt mk ρ κ M N) :
    b.Addrs (tl.addrs ptrs mk) (fun i => e + ρ i.1 * s0 + κ i.2.1 * s1)
      (fun i => decide (ρ i.1 < M) && decide (κ i.2.1 < N)) := by
  rintro ⟨i, j, ⟨⟩⟩
  have hmij := hm i j
  by_cases hij : ρ i < M ∧ κ j < N
  · have hpij : ptrs.get (i, j, ()) = _ := hp i j hij.1 hij.2
    have hen : on mk i j = true := by rw [hmij]; simp [hij.1, hij.2]
    cases mk with
    | none => simp [tl.addrs, hpij, hij.1, hij.2]
    | some m =>
      have hm' : m.get (i, j, ()) = true := hen
      simp [tl.addrs, hm', hpij, hij.1, hij.2]
  · have hd : (decide (ρ i < M) && decide (κ j < N)) = false := by
      simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]; tauto
    have hen : on mk i j = false := by rw [hmij]; exact hd
    cases mk with
    | none => exact absurd hen (by simp [on])
    | some m =>
      have hm' : m.get (i, j, ()) = false := hen
      simp only [tl.addrs, hm', Bool.false_eq_true, ite_false]
      simp [hd]

private theorem tileAt_lane_elem (i : Idx [BM, BN])
    (h : (decide (ρ i.1 < M) && decide (κ i.2.1 < N)) = true) :
    ElemAt e s0 s1 ρ κ M N (e + ρ i.1 * s0 + κ i.2.1 * s1) := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨i.1, i.2.1, h.1, h.2, rfl⟩

/-- The elements a tile access touches are the tile's (`Tile2.ElemAt`). -/
theorem img_tileAt :
    Buf.img (s := [BM, BN]) (fun i => e + ρ i.1 * s0 + κ i.2.1 * s1)
      (fun i => decide (ρ i.1 < M) && decide (κ i.2.1 < N)) =
      {x | ElemAt e s0 s1 ρ κ M N x} := by
  ext x; constructor
  · rintro ⟨⟨i, j, ⟨⟩⟩, h, rfl⟩
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact ⟨i, j, h.1, h.2, rfl⟩
  · rintro ⟨i, j, hiM, hjN, rfl⟩
    exact ⟨(i, j, ()), by simp [hiM, hjN], rfl⟩

/-- The bytes a tile access touches: `b.tileAt`. -/
theorem accBytes_tileAt (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)}
    (hp : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      ptrs.get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int))
    (hm : MaskAt mk ρ κ M N) :
    accBytes d (tl.addrs ptrs mk) = b.tileAt e s0 s1 ρ κ M N := by
  rw [(addrs_tileAt b hp hm).accBytes, img_tileAt]; rfl

private theorem tileAt_inj (hinj : Inj s0 s1 M N)
    (hρ : ∀ i i' : Fin BM, ρ i < M → ρ i' < M → ρ i = ρ i' → i = i')
    (hκ : ∀ j j' : Fin BN, κ j < N → κ j' < N → κ j = κ j' → j = j') (i i' : Idx [BM, BN])
    (hi : (decide (ρ i.1 < M) && decide (κ i.2.1 < N)) = true)
    (hi' : (decide (ρ i'.1 < M) && decide (κ i'.2.1 < N)) = true)
    (h : e + ρ i.1 * s0 + κ i.2.1 * s1 = e + ρ i'.1 * s0 + κ i'.2.1 * s1) : i = i' := by
  obtain ⟨i, j, ⟨⟩⟩ := i
  obtain ⟨i', j', ⟨⟩⟩ := i'
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hi hi'
  dsimp only at h
  obtain ⟨h1, h2⟩ := hinj _ _ _ _ hi.1 hi.2 hi'.1 hi'.2 (by omega)
  rw [hρ i i' hi.1 hi'.1 h1, hκ j j' hi.2 hi'.2 h2]

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **2-D tile load.** Each enabled lane (`ρ i < M`, `κ j < N`) gets its element; a masked-off
lane gets `other` when it is given, and is unspecified otherwise. The pointers are needed on the
enabled lanes only (`hp`). The matrix is empty or `e + (M - 1) * s0 + (N - 1) * s1 < b.len`, and
the tile's bytes are in the read footprint (`Buf.tileAt_sub_matrix` when it holds the matrix). -/
theorem safe_load_tileAt (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)} {other : Option (Tensor [BM, BN] d.Val)}
    (hp : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      ptrs.get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int))
    (hm : MaskAt mk ρ κ M N)
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len)
    (hv : b.Valid m) (hR : b.tileAt e s0 s1 ρ κ M N ⊆ R)
    {k : Tensor [BM, BN] d.Val → Prog α}
    (hk : ∀ v : Tensor [BM, BN] d.Val,
      (∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
        m.read d (b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int)).addr = some (v.get2 i j)) →
      (∀ (i : Fin BM) (j : Fin BN), ¬ (ρ i < M ∧ κ j < N) → ∀ o, other = some o →
        v.get2 i j = o.get2 i j) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs mk other >>= k) :=
  safe_load_buf b (addrs_tileAt b hp hm)
    (fun i h => matrix_lt hlen (elemAt_matrix (tileAt_lane_elem i h))) hv
    (by rw [img_tileAt]; exact hR)
    fun v h1 h2 => hk v (fun i j hi hj => h1 (i, j, ()) (by simp [hi, hj]))
      (fun i j hij o ho => h2 (i, j, ()) (by
        simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]; tauto) o ho)

/-- **2-D tile store.** Afterwards each enabled lane's element holds its value, nothing outside the
tile changed, and every present byte is still present. The layout must not alias (`Tile2.Inj`),
and distinct enabled lanes must have distinct rows (`hρ`) and columns (`hκ`; `add_inj` for
`r0 + i`), so that the lanes write distinct elements. The rest of the program may not write the
tile again. -/
theorem safe_store_tileAt (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)} {vals : Tensor [BM, BN] d.Val}
    (hp : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      ptrs.get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int))
    (hm : MaskAt mk ρ κ M N)
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len)
    (hinj : Inj s0 s1 M N)
    (hρ : ∀ i i' : Fin BM, ρ i < M → ρ i' < M → ρ i = ρ i' → i = i')
    (hκ : ∀ j j' : Fin BN, κ j < N → κ j' < N → κ j = κ j' → j = j')
    (hv : b.Valid m) (hW : b.tileAt e s0 s1 ρ κ M N ⊆ W)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
        m'.read d (b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int)).addr = some (vals.get2 i j)) →
      (∀ x, x ∉ b.tileAt e s0 s1 ρ κ M N → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.tileAt e s0 s1 ρ κ M N) Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals mk >>= k) := by
  refine safe_store_buf b (addrs_tileAt b hp hm)
    (fun i h => matrix_lt hlen (elemAt_matrix (tileAt_lane_elem i h)))
    (fun i i' hi hi' h => tileAt_inj hinj hρ hκ i i' hi hi' h) hv
    (by rw [img_tileAt]; exact hW) fun m' h1 h2 h3 => ?_
  rw [img_tileAt]
  exact hk m' (fun i j hi hj => h1 (i, j, ()) (by simp [hi, hj]))
    (fun x hx => h2 x (by rw [img_tileAt]; exact hx)) h3

end Access

end TVA
