import TVA.Lemmas.Block
import TVA.Lemmas.IntW

/-!
# Two-dimensional strided tiles

The access pattern of every matrix operand:

```python
rows = r0 + tl.arange(0, BM)                 # or pid_m * BM + tl.arange(0, BM)
cols = c0 + tl.arange(0, BN)
ptrs = base + rows[:, None] * s0 + cols[None, :] * s1
mask = (rows < M)[:, None] & (cols < N)[None, :]   # or one factor, or none
x = tl.load(ptrs, mask=mask)
```

Lane `(i, j)` addresses element `e + (r0 + i) * s0 + (c0 + j) * s1` of a buffer `b` (the base
pointer is `b.ptr e`), and the mask enables it when `r0 + i < M` and `c0 + j < N`
(`Tile2.Elem`). This file states, in `int32` or `int64` (any signed width `w`):

* **row and column vectors**: `start_add_toInt` (`a + x`, any width), `arange64_toInt` and
  `start_arange64_toInt` (`tl.arange` converted to `int64` by `tl.convert .i32 .i64`, the model of
  `pid.to(tl.int64) * BM + tl.arange(0, BM)`); in `int32`, `IntV.I32.toInt_tl_arange` and
  `IntV.I32.toInt_sadd_arange` (`pid * BM + tl.arange(0, BM)`), with `IntV.mul_ofNat` for
  `pid * BM` and `Block.fits_of_pow2` for its bound (`TVA.Lemmas.IntW`, `TVA.Lemmas.Block`); rows
  wrapped by `% M` (`irem_ok`, `irem_toInt`, `safe_irem_lanes`: lane `i` of
  `tl.irem (r0 + tl.arange(0, BM)) M` is `(r0 + i) % M`, below `M`);
* **offsets** (`Tile2.off_toInt`, and the terms `colMul_toInt`, `mulCol_toInt`, `col_toInt`,
  `rowMul_toInt`, `mulRow_toInt`, `row_toInt`): each lane's offset is the integer one, with no
  overflow, from a bound on the largest offset over every lane,
  `(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2^(w-1)`; and on the enabled lanes only (`r0 + i < M`,
  `c0 + j < N`) from the matrix's largest offset, `(M - 1) * s0 + (N - 1) * s1 < 2^(w-1)`
  (`off_toInt_on`, `add_toInt_on`, `colMul_toInt_on`, `mulCol_toInt_on`, `rowMul_toInt_on`,
  `mulRow_toInt_on`): a masked-off lane's offset may wrap, since nothing uses it. The `…_on` forms
  take rows and columns of any form, lane values `ρ i` and `κ j` (library issue L184), of which
  `r0 + i` and `c0 + j` are one instance and `(r0 + i) % M` another;
* **pointers** (`Tile2.ptr_lane`, `ptr_lane_seq`, and `ptr_lane_sum_of`, `ptr_lane_seq_of` for any
  form of the two terms, `ptr_lane_sum_swap_of`, `ptr_lane_seq_swap_of` with the column term
  first; on the enabled lanes, from the matrix's bound, the same names with `_on`): the pointer
  tensor's `get2`, for `base + (x + y)` and `base + x + y`; and lane by lane,
  `Tensor.get2_ptrAdd`, `Tensor.get2_ptrAddT`;
* **masks** (`Tile2.MaskOk`: `maskOk_and`, `maskOk_rows`, `maskOk_cols`, `maskOk_none`,
  `maskOk_of_get`);
* **footprints** (`addrs_tile2`, `img_tile2`, `accBytes_tile2`, `Buf.tile2`), and the load and store
  rules `safe_load_tile2`, `safe_store_tile2`, as `Block` does for the 1-D block; the matrix lies in
  its buffer when it is empty (`M = 0` or `N = 0`, as for the last sequence of a varlen batch, of
  length `0`, at the end of its buffer) or its last element does
  (`M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len`);
* **rows and columns of any form** (library issue L184): `MaskAt mk ρ κ M N`, `ElemAt`,
  `Buf.tileAt`, and the footprint and load and store rules `addrs_tileAt`, `accBytes_tileAt`,
  `safe_load_tileAt`, `safe_store_tileAt` (the store with distinct enabled rows and columns); the
  `r0 + i` forms above are the instance `ρ i = r0 + i`, `κ j = c0 + j` (`maskOk_iff_maskAt`,
  `elem_iff_elemAt`, `Buf.tile2_eq_tileAt`), and a tile lies in its whole matrix
  (`Buf.tileAt_sub_matrix`);
* **disjointness** of the elements of two tiles whose rows or columns are apart by at least the
  block (`Tile2.elem_disjoint`, `elem_disjoint_grid`, `Buf.tile2_disjoint`, `Buf.tile2_disjoint_grid`),
  when distinct `(row, column)` pairs give distinct elements (`Tile2.Inj`): row-major
  (`inj_rowMajor`: `N * s1 ≤ s0`, `0 < s1`; `s1 = 1`, `N ≤ s0`: `inj_rowMajor1`) or column-major
  (`inj_colMajor`: `M * s0 ≤ s1`, `0 < s0`; `s0 = 1`, `M ≤ s1`: `inj_colMajor1`).

A lane rule gives the pointers of the enabled lanes,
`hp : ∀ i j, r0 + i < M → c0 + j < N → ptrs.get2 i j = b.ptr (e + (r0 + i) * s0 + (c0 + j) * s1)`
(`ptr_lane_on` and the other `…_on`; from a rule for every lane, `fun i j _ _ => ptr_lane … i j`),
and a mask rule gives `MaskOk mk r0 c0 M N`; the footprint and load and store rules take those
two, so any way of writing the pointers or the mask fits. Without a row (column) mask, choose
`M` (`N`) with `r0 + BM ≤ M` (`c0 + BN ≤ N`): the tile's elements are then the same.
-/
namespace TVA

namespace Tile2

/-! ## Row and column vectors

The scalar arithmetic is `TVA.Lemmas.IntW`'s, for any width and with no `0 < w` premise:
`IntV.mul_nat`, `IntV.add_nat`, `IntV.mul_ofNat` (`pid * BM`, with `IntV.ofNat BM`); the lanes of
`tl.arange(0, n)` are `IntV.I32.toInt_tl_arange`, and of `c + tl.arange(0, n)` in `int32`
`IntV.I32.toInt_sadd_arange`. -/

section IntArith
variable {w : Nat}

/-- `a + x` for a scalar `a` (the first row or column, `r0`) and a vector `x` whose lane `i` has value
`i` (`tl.arange(0, n)`: `IntV.I32.toInt_tl_arange`; that vector converted to `int64`:
`arange64_toInt`): lane `i` is `r0 + i`, when `r0 + n ≤ 2^(w-1)`. -/
theorem start_add_toInt {n r0 : Nat} {a : IntV true w} {x : Tensor [n] (IntV true w)}
    (ha : a.toInt = r0) (hx : ∀ i : Fin n, (x.get1 i).toInt = ((i : Nat) : Int))
    (hfit : r0 + n ≤ 2 ^ (w - 1)) (i : Fin n) :
    ((a + x).get1 i).toInt = ((r0 + i : Nat) : Int) := by
  have hi := i.isLt
  rw [Tensor.get1_sadd]
  exact IntV.add_nat ha (hx i) (by omega)

/-- **`tl.arange(0, n)` converted to `int64`.** The implicit promotion of
`pid.to(tl.int64) * BM + tl.arange(0, BM)` and the explicit `tl.arange(0, BM).to(tl.int64)` are
both `tl.convert .i32 .i64` in the model (SEMANTICS.md, "Operators and promotion"); from the lanes
`safe_convert` gives (`ConvRel P .i32 .i64`), lane `i` has value `i`. With `start_add_toInt`
(`w = 64`), lane `i` of `a + x` is `r0 + i` when `r0 + n ≤ 2^63`. -/
theorem arange64_toInt {P : Platform} {n : Nat} {x : Tensor [n - 0] I64}
    (hx : ∀ i, ConvRel P .i32 .i64 ((tl.arange 0 n).get i) (x.get i)) (hn : n ≤ 2 ^ 31)
    (i : Fin (n - 0)) : (x.get1 i).toInt = ((i : Nat) : Int) := by
  rw [show x.get1 i = x.get (i, ()) from rfl, ConvRel.toInt_i32_i64 (hx (i, ()))]
  exact IntV.I32.toInt_tl_arange hn i

/-- `pid.to(tl.int64) * BM + tl.arange(0, BM)` as the model writes it, `a + x` with `x` the
`arange` converted by `tl.convert .i32 .i64` (`arange64_toInt`): lane `i` is `r0 + i`, when
`r0 + n ≤ 2^63`. -/
theorem start_arange64_toInt {P : Platform} {n r0 : Nat} {a : I64} {x : Tensor [n - 0] I64}
    (ha : a.toInt = r0) (hx : ∀ i, ConvRel P .i32 .i64 ((tl.arange 0 n).get i) (x.get i))
    (hfit : r0 + n ≤ 2 ^ 63) (hn : n ≤ 2 ^ 31) (i : Fin (n - 0)) :
    ((a + x).get1 i).toInt = ((r0 + i : Nat) : Int) :=
  start_add_toInt ha (arange64_toInt hx hn) (by simpa using hfit) i

/-- **Rows wrapped by `% M`**, `(r0 + tl.arange(0, BM)) % M` (tutorial 03's
`offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M`, in `int32` or, converted,
`int64`): `tl.irem x y` with `x`'s lane `i` of value `f i` (`r0 + i`: `start_add_toInt`,
`IntV.I32.toInt_sadd_arange`) and every lane of `y` of value `M > 0` (the scalar `M` broadcast).
Every lane is defined (`safe_irem`'s premise), and every result's lane `i` has value `f i % M`,
which is below `M` (`Nat.mod_lt`): a row in the matrix, so every lane of the tile is enabled
without a row mask (`maskAt_none`, `maskAt_cols`, with `ρ i = f i % M`). -/
theorem irem_ok {n M : Nat} {f : Fin n → Nat} {x y : Tensor [n] (IntV true w)}
    (hx : ∀ i : Fin n, (x.get1 i).toInt = ((f i : Nat) : Int))
    (hy : ∀ i : Fin n, (y.get1 i).toInt = (M : Int)) (hM : 0 < M) :
    ∀ i, ∃ q, (x.get i).rem? (y.get i) = some q := by
  rintro ⟨i, ⟨⟩⟩
  obtain ⟨q, hq, -⟩ := IntV.rem?_nat (hx i) (hy i) hM
  exact ⟨q, hq⟩

/-- The lanes of a result of `tl.irem x y` (`irem_ok`): lane `i` has value `f i % M`. -/
theorem irem_toInt {n M : Nat} {f : Fin n → Nat} {x y r : Tensor [n] (IntV true w)}
    (hx : ∀ i : Fin n, (x.get1 i).toInt = ((f i : Nat) : Int))
    (hy : ∀ i : Fin n, (y.get1 i).toInt = (M : Int)) (hM : 0 < M)
    (hr : ∀ i, (x.get i).rem? (y.get i) = some (r.get i)) (i : Fin n) :
    (r.get1 i).toInt = ((f i % M : Nat) : Int) :=
  IntV.toInt_of_rem?_nat (hr (i, ())) (hx i) (hy i) hM

/-- **`tl.irem x y` with rows wrapped by `% M`, as a rule**: the rest of the program runs on a
vector whose lane `i` has value `f i % M` (below `M`). -/
theorem safe_irem_lanes {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {n M : Nat} {f : Fin n → Nat} {x y : Tensor [n] (IntV true w)}
    {k : Tensor [n] (IntV true w) → Prog α}
    (hx : ∀ i : Fin n, (x.get1 i).toInt = ((f i : Nat) : Int))
    (hy : ∀ i : Fin n, (y.get1 i).toInt = (M : Int)) (hM : 0 < M)
    (hk : ∀ r : Tensor [n] (IntV true w),
      (∀ i : Fin n, (r.get1 i).toInt = ((f i % M : Nat) : Int)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.irem x y >>= k) :=
  safe_irem (irem_ok hx hy hM) fun r hr => hk r (irem_toInt hx hy hM hr)

end IntArith

/-! ## Lane offsets

Two forms. The first (`colMul_toInt`, …, `off_toInt`) bounds every lane, masked-off ones included:
the largest offset `(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1` is below `2^(w-1)`. The second (`…_on`)
bounds the enabled lanes only, those with `r0 + i < M` and `c0 + j < N`, from the matrix's largest
offset `(M - 1) * s0 + (N - 1) * s1 < 2^(w-1)`: a masked-off lane's offset may wrap, and nothing
uses it (`tl.addrs` gives that lane no address, and the mask is computed from `rows` and `cols`,
not from the offset). A claim that bounds only the matrix uses the second form on the last
partial tile. -/

section Offsets
variable {w BM BN r0 c0 s0 s1 M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

private theorem row_le {i : Nat} (hi : i < BM) (s : Nat) : (r0 + i) * s ≤ (r0 + BM - 1) * s :=
  Nat.mul_le_mul_right s (by omega)

private theorem le_pred {k K : Nat} (h : k < K) (s : Nat) : k * s ≤ (K - 1) * s :=
  Nat.mul_le_mul_right s (by omega)

/-- `rows[:, None] * a` (`rows.col BN * a`): lane `(i, j)` is `(r0 + i) * s0`. -/
theorem colMul_toInt {rows : Tensor [BM] (IntV true w)} {a : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)) (ha : a.toInt = s0)
    (hfit : (r0 + BM - 1) * s0 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN * a).get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int) :=
  IntV.mul_nat (hr i) ha (lt_of_le_of_lt (row_le i.isLt s0) hfit)

/-- `a * rows[:, None]` (`a * rows.col BN`): lane `(i, j)` is `(r0 + i) * s0`. -/
theorem mulCol_toInt {rows : Tensor [BM] (IntV true w)} {a : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)) (ha : a.toInt = s0)
    (hfit : (r0 + BM - 1) * s0 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((a * rows.col BN).get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int) := by
  rw [show (((r0 + i) * s0 : Nat) : Int) = ((s0 * (r0 + i) : Nat) : Int) by rw [Nat.mul_comm]]
  exact IntV.mul_nat ha (hr i) (by rw [Nat.mul_comm]; exact lt_of_le_of_lt (row_le i.isLt s0) hfit)

/-- `rows[:, None]` with a unit row stride: lane `(i, j)` is `(r0 + i) * 1`. -/
theorem col_toInt {rows : Tensor [BM] (IntV true w)}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN).get2 i j).toInt = (((r0 + i) * 1 : Nat) : Int) := by
  rw [Nat.mul_one]; exact hr i

/-- `cols[None, :] * a` (`cols.row BM * a`): lane `(i, j)` is `(c0 + j) * s1`. -/
theorem rowMul_toInt {cols : Tensor [BN] (IntV true w)} {a : IntV true w}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (ha : a.toInt = s1)
    (hfit : (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((cols.row BM * a).get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int) :=
  IntV.mul_nat (hc j) ha (lt_of_le_of_lt (row_le j.isLt s1) hfit)

/-- `a * cols[None, :]` (`a * cols.row BM`): lane `(i, j)` is `(c0 + j) * s1`. -/
theorem mulRow_toInt {cols : Tensor [BN] (IntV true w)} {a : IntV true w}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (ha : a.toInt = s1)
    (hfit : (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((a * cols.row BM).get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int) := by
  rw [show (((c0 + j) * s1 : Nat) : Int) = ((s1 * (c0 + j) : Nat) : Int) by rw [Nat.mul_comm]]
  exact IntV.mul_nat ha (hc j) (by rw [Nat.mul_comm]; exact lt_of_le_of_lt (row_le j.isLt s1) hfit)

/-- `cols[None, :]` with a unit column stride: lane `(i, j)` is `(c0 + j) * 1`. -/
theorem row_toInt {cols : Tensor [BN] (IntV true w)}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    ((cols.row BM).get2 i j).toInt = (((c0 + j) * 1 : Nat) : Int) := by
  rw [Nat.mul_one]; exact hc j

/-- **The sum of a row term and a column term** (`x + y`, any form of either): lane `(i, j)` is
`(r0 + i) * s0 + (c0 + j) * s1`, with no overflow when the largest offset,
`(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1`, is below `2^(w-1)`. -/
theorem add_toInt {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int))
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((x + y).get2 i j).toInt = (((r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  IntV.add_nat (hx i j) (hy i j)
    (lt_of_le_of_lt (Nat.add_le_add (row_le i.isLt s0) (row_le j.isLt s1)) hfit)

/-- **The offset of a 2-D tile**, `rows[:, None] * a0 + cols[None, :] * a1` in a signed width `w`
(`int32`: `w = 32`, `int64`: `w = 64`), with `rows` lanes `r0 + i`, `cols` lanes `c0 + j` and the
strides `a0`, `a1` of values `s0`, `s1`: lane `(i, j)` is `(r0 + i) * s0 + (c0 + j) * s1`, with no
overflow when the largest offset, `(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1`, is below `2^(w-1)`. -/
theorem off_toInt {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)}
    {a0 a1 : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN * a0 + cols.row BM * a1).get2 i j).toInt =
      (((r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  add_toInt (colMul_toInt hr ha0 (by omega)) (rowMul_toInt hc ha1 (by omega)) hfit i j

/-! ### Enabled lanes only

Rows and columns of any form (library issue L184): row lane `i` has a value `ρ i` and column lane
`j` a value `κ j`, natural numbers; `ρ i = r0 + i` (`rows = r0 + tl.arange(0, BM)`) is one instance,
`ρ i = (r0 + i) % M` (`rows = (r0 + tl.arange(0, BM)) % M`, `irem_toInt`) another. A lane is enabled
when `ρ i < M` and `κ j < N`. A caller with `hr : ∀ i, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)`
passes it as it is: Lean takes `ρ = fun i => r0 + i`. -/

/-- `rows[:, None] * a` on a lane whose row is in the matrix (`ρ i < M`): `ρ i * s0`, when
`(M - 1) * s0 < 2^(w-1)`. -/
theorem colMul_toInt_on {rows : Tensor [BM] (IntV true w)} {a : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int)) (ha : a.toInt = s0)
    (hfit : (M - 1) * s0 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) (hi : ρ i < M) :
    ((rows.col BN * a).get2 i j).toInt = ((ρ i * s0 : Nat) : Int) :=
  IntV.mul_nat (hr i) ha (lt_of_le_of_lt (le_pred hi s0) hfit)

/-- `a * rows[:, None]` on a lane whose row is in the matrix: `ρ i * s0`, when
`(M - 1) * s0 < 2^(w-1)`. -/
theorem mulCol_toInt_on {rows : Tensor [BM] (IntV true w)} {a : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int)) (ha : a.toInt = s0)
    (hfit : (M - 1) * s0 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) (hi : ρ i < M) :
    ((a * rows.col BN).get2 i j).toInt = ((ρ i * s0 : Nat) : Int) := by
  rw [show ((ρ i * s0 : Nat) : Int) = ((s0 * ρ i : Nat) : Int) by rw [Nat.mul_comm]]
  exact IntV.mul_nat ha (hr i) (by rw [Nat.mul_comm]; exact lt_of_le_of_lt (le_pred hi s0) hfit)

/-- `cols[None, :] * a` on a lane whose column is in the matrix (`κ j < N`): `κ j * s1`, when
`(N - 1) * s1 < 2^(w-1)`. -/
theorem rowMul_toInt_on {cols : Tensor [BN] (IntV true w)} {a : IntV true w}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int)) (ha : a.toInt = s1)
    (hfit : (N - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) (hj : κ j < N) :
    ((cols.row BM * a).get2 i j).toInt = ((κ j * s1 : Nat) : Int) :=
  IntV.mul_nat (hc j) ha (lt_of_le_of_lt (le_pred hj s1) hfit)

/-- `a * cols[None, :]` on a lane whose column is in the matrix: `κ j * s1`, when
`(N - 1) * s1 < 2^(w-1)`. -/
theorem mulRow_toInt_on {cols : Tensor [BN] (IntV true w)} {a : IntV true w}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int)) (ha : a.toInt = s1)
    (hfit : (N - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) (hj : κ j < N) :
    ((a * cols.row BM).get2 i j).toInt = ((κ j * s1 : Nat) : Int) := by
  rw [show ((κ j * s1 : Nat) : Int) = ((s1 * κ j : Nat) : Int) by rw [Nat.mul_comm]]
  exact IntV.mul_nat ha (hc j) (by rw [Nat.mul_comm]; exact lt_of_le_of_lt (le_pred hj s1) hfit)

/-- **The sum of a row term and a column term on an enabled lane** (`ρ i < M`, `κ j < N`):
`ρ i * s0 + κ j * s1`, with no overflow when the matrix's largest offset,
`(M - 1) * s0 + (N - 1) * s1`, is below `2^(w-1)`. The row term need hold only on rows in the
matrix (`colMul_toInt_on`, `mulCol_toInt_on`; `fun i j _ => col_toInt hr i j`), the column term
only on columns in it. -/
theorem add_toInt_on {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → (x.get2 i j).toInt = ((ρ i * s0 : Nat) : Int))
    (hy : ∀ (i : Fin BM) (j : Fin BN), κ j < N → (y.get2 i j).toInt = ((κ j * s1 : Nat) : Int))
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN)
    (hi : ρ i < M) (hj : κ j < N) :
    ((x + y).get2 i j).toInt = ((ρ i * s0 + κ j * s1 : Nat) : Int) :=
  IntV.add_nat (hx i j hi) (hy i j hj)
    (lt_of_le_of_lt (Nat.add_le_add (le_pred hi s0) (le_pred hj s1)) hfit)

/-- **The offset of a 2-D tile on an enabled lane**, `rows[:, None] * a0 + cols[None, :] * a1`:
lane `(i, j)` with `ρ i < M` and `κ j < N` is `ρ i * s0 + κ j * s1`, with no overflow when the
matrix's largest offset, `(M - 1) * s0 + (N - 1) * s1`, is below `2^(w-1)`. -/
theorem off_toInt_on {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)}
    {a0 a1 : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN)
    (hi : ρ i < M) (hj : κ j < N) :
    ((rows.col BN * a0 + cols.row BM * a1).get2 i j).toInt =
      ((ρ i * s0 + κ j * s1 : Nat) : Int) :=
  add_toInt_on (colMul_toInt_on hr ha0 (by omega)) (rowMul_toInt_on hc ha1 (by omega)) hfit i j hi hj
end Offsets

/-! ## Pointers -/

/-- Lane `(i, j)` of `p + offs` (a pointer plus a 2-D offset tensor) is `p` offset by lane
`(i, j)` of `offs`. -/
@[simp] theorem _root_.TVA.Tensor.get2_ptrAdd {m n : Nat} {d : DType} {sg : Bool} {w : Nat} (p : Ptr d)
    (offs : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    (p + offs).get2 i j = p.offset (offs.get2 i j).toInt := rfl

/-- Lane `(i, j)` of `ps + offs` (a 2-D pointer tensor plus a 2-D offset tensor) is lane `(i, j)` of
`ps` offset by lane `(i, j)` of `offs`. -/
@[simp] theorem _root_.TVA.Tensor.get2_ptrAddT {m n : Nat} {d : DType} {sg : Bool} {w : Nat}
    (ps : Tensor [m, n] (Ptr d)) (offs : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    (ps + offs).get2 i j = (ps.get2 i j).offset (offs.get2 i j).toInt := rfl

private theorem row_le' {r0 BM i : Nat} (hi : i < BM) (s : Nat) :
    (r0 + i) * s ≤ (r0 + BM - 1) * s :=
  Nat.mul_le_mul_right s (by omega)

private theorem le_pred' {k K : Nat} (h : k < K) (s : Nat) : k * s ≤ (K - 1) * s :=
  Nat.mul_le_mul_right s (by omega)

section Pointers
variable {d : DType} {w w' BM BN r0 c0 s0 s1 e M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

/-- **The pointers `base + (x + y)`** (one offset tensor, the sum of a row term and a column term
of any form), with `base = b.ptr e`: lane `(i, j)` points at element
`e + (r0 + i) * s0 + (c0 + j) * s1`, when the largest offset is below `2^(w-1)`. -/
theorem ptr_lane_sum_of (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int))
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    (base + (x + y)).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  show base.offset ((x + y).get2 i j).toInt = _
  rw [add_toInt hx hy hfit, hb, Buf.ptr_offset]; push_cast; ring_nf

/-- **The pointers `base + x + y`** (`(base + x) + y`: two pointer additions, a row term and a
column term of any form, possibly of different widths), with `base = b.ptr e`: lane `(i, j)` points
at element `e + (r0 + i) * s0 + (c0 + j) * s1`. Each term's own lemma bounds that term only. -/
theorem ptr_lane_seq_of (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {sg sg' : Bool} {x : Tensor [BM, BN] (IntV sg w)} {y : Tensor [BM, BN] (IntV sg' w')}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    (base + x + y).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  show (base.offset (x.get2 i j).toInt).offset (y.get2 i j).toInt = _
  rw [hx, hy, hb, Buf.ptr_offset, Buf.ptr_offset]; push_cast; ring_nf

/-- `ptr_lane_sum_of` with the column term first, `base + (y + x)`. -/
theorem ptr_lane_sum_swap_of (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int))
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    (base + (y + x)).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  have hs : ((y + x).get2 i j).toInt = (((r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
    rw [show (r0 + (i : Nat)) * s0 + (c0 + j) * s1 = (c0 + j) * s1 + (r0 + i) * s0 by ring]
    exact IntV.add_nat (hy i j) (hx i j)
      (by have := Nat.add_le_add (row_le' (r0 := r0) i.isLt s0) (row_le' (r0 := c0) j.isLt s1); omega)
  show base.offset ((y + x).get2 i j).toInt = _
  rw [hs, hb, Buf.ptr_offset]; push_cast; ring_nf

/-- `ptr_lane_seq_of` with the column term added first, `base + y + x` (as in
`K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d`, a transposed load). -/
theorem ptr_lane_seq_swap_of (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {sg sg' : Bool} {x : Tensor [BM, BN] (IntV sg w)} {y : Tensor [BM, BN] (IntV sg' w')}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    (base + y + x).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  show (base.offset (y.get2 i j).toInt).offset (x.get2 i j).toInt = _
  rw [hx, hy, hb, Buf.ptr_offset, Buf.ptr_offset]; push_cast; ring_nf

/-- **The pointers of a 2-D tile**, `base + (rows[:, None] * a0 + cols[None, :] * a1)` with
`base = b.ptr e`, in a signed width `w`: lane `(i, j)` points at element
`e + (r0 + i) * s0 + (c0 + j) * s1`, when the largest offset,
`(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1`, is below `2^(w-1)`. -/
theorem ptr_lane (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)} {a0 a1 : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    (base + (rows.col BN * a0 + cols.row BM * a1)).get2 i j =
      b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  ptr_lane_sum_of b hb (colMul_toInt hr ha0 (by omega)) (rowMul_toInt hc ha1 (by omega)) hfit i j

/-- **The pointers of a 2-D tile, added one term at a time**,
`base + rows[:, None] * a0 + cols[None, :] * a1` with `base = b.ptr e` (the row term in width `w`,
the column term in width `w'`): lane `(i, j)` points at element `e + (r0 + i) * s0 + (c0 + j) * s1`,
when each term's largest value is below its bound. -/
theorem ptr_lane_seq (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w')} {a0 : IntV true w}
    {a1 : IntV true w'}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit0 : (r0 + BM - 1) * s0 < 2 ^ (w - 1)) (hfit1 : (c0 + BN - 1) * s1 < 2 ^ (w' - 1))
    (i : Fin BM) (j : Fin BN) :
    (base + rows.col BN * a0 + cols.row BM * a1).get2 i j =
      b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  ptr_lane_seq_of b hb (colMul_toInt hr ha0 hfit0) (rowMul_toInt hc ha1 hfit1) i j

/-! ### Enabled lanes only

Each gives the pointers on the lanes with `ρ i < M` and `κ j < N` (rows and columns of any form,
as for the offsets above; `ρ i = r0 + i` when `hr` is about `r0 + i`), the form the footprint and
load and store rules take (`hp`), from the matrix's largest offset. A lemma above gives that form
too: `fun i j _ _ => ptr_lane … i j`. -/

/-- `ptr_lane_sum_of` on the enabled lanes, `base + (x + y)`: from the terms on rows (columns) in
the matrix and `(M - 1) * s0 + (N - 1) * s1 < 2^(w-1)`. -/
theorem ptr_lane_sum_of_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → (x.get2 i j).toInt = ((ρ i * s0 : Nat) : Int))
    (hy : ∀ (i : Fin BM) (j : Fin BN), κ j < N → (y.get2 i j).toInt = ((κ j * s1 : Nat) : Int))
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (w - 1)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + (x + y)).get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) := by
  intro i j hi hj
  show base.offset ((x + y).get2 i j).toInt = _
  rw [add_toInt_on hx hy hfit i j hi hj, hb, Buf.ptr_offset]; push_cast; ring_nf

/-- `ptr_lane_sum_swap_of` on the enabled lanes, `base + (y + x)`. -/
theorem ptr_lane_sum_swap_of_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → (x.get2 i j).toInt = ((ρ i * s0 : Nat) : Int))
    (hy : ∀ (i : Fin BM) (j : Fin BN), κ j < N → (y.get2 i j).toInt = ((κ j * s1 : Nat) : Int))
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (w - 1)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + (y + x)).get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) := by
  intro i j hi hj
  have hs : ((y + x).get2 i j).toInt = ((ρ i * s0 + κ j * s1 : Nat) : Int) := by
    rw [show ρ i * s0 + κ j * s1 = κ j * s1 + ρ i * s0 by ring]
    exact IntV.add_nat (hy i j hj) (hx i j hi)
      (by have := Nat.add_le_add (le_pred' hi s0) (le_pred' hj s1); omega)
  show base.offset ((y + x).get2 i j).toInt = _
  rw [hs, hb, Buf.ptr_offset]; push_cast; ring_nf

/-- `ptr_lane_seq_of` on the enabled lanes, `base + x + y`: each term on rows (columns) in the
matrix, bounded by its own lemma. -/
theorem ptr_lane_seq_of_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {sg sg' : Bool} {x : Tensor [BM, BN] (IntV sg w)} {y : Tensor [BM, BN] (IntV sg' w')}
    (hx : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → (x.get2 i j).toInt = ((ρ i * s0 : Nat) : Int))
    (hy : ∀ (i : Fin BM) (j : Fin BN), κ j < N → (y.get2 i j).toInt = ((κ j * s1 : Nat) : Int)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + x + y).get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) := by
  intro i j hi hj
  show (base.offset (x.get2 i j).toInt).offset (y.get2 i j).toInt = _
  rw [hx i j hi, hy i j hj, hb, Buf.ptr_offset, Buf.ptr_offset]; push_cast; ring_nf

/-- `ptr_lane_seq_swap_of` on the enabled lanes, `base + y + x`. -/
theorem ptr_lane_seq_swap_of_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {sg sg' : Bool} {x : Tensor [BM, BN] (IntV sg w)} {y : Tensor [BM, BN] (IntV sg' w')}
    (hx : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → (x.get2 i j).toInt = ((ρ i * s0 : Nat) : Int))
    (hy : ∀ (i : Fin BM) (j : Fin BN), κ j < N → (y.get2 i j).toInt = ((κ j * s1 : Nat) : Int)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + y + x).get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) := by
  intro i j hi hj
  show (base.offset (y.get2 i j).toInt).offset (x.get2 i j).toInt = _
  rw [hx i j hi, hy i j hj, hb, Buf.ptr_offset, Buf.ptr_offset]; push_cast; ring_nf

/-- **The pointers of a 2-D tile on the enabled lanes**,
`base + (rows[:, None] * a0 + cols[None, :] * a1)` with `base = b.ptr e`: lane `(i, j)` with
`ρ i < M` and `κ j < N` points at element `e + ρ i * s0 + κ j * s1`, when the matrix's largest
offset, `(M - 1) * s0 + (N - 1) * s1`, is below `2^(w-1)`. -/
theorem ptr_lane_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)} {a0 a1 : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (w - 1)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + (rows.col BN * a0 + cols.row BM * a1)).get2 i j =
        b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) :=
  ptr_lane_sum_of_on b hb (colMul_toInt_on hr ha0 (by omega)) (rowMul_toInt_on hc ha1 (by omega)) hfit

/-- **The pointers of a 2-D tile on the enabled lanes, added one term at a time**,
`base + rows[:, None] * a0 + cols[None, :] * a1` (the row term in width `w`, the column term in
width `w'`), when `(M - 1) * s0 < 2^(w-1)` and `(N - 1) * s1 < 2^(w'-1)`. -/
theorem ptr_lane_seq_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w')} {a0 : IntV true w}
    {a1 : IntV true w'}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit0 : (M - 1) * s0 < 2 ^ (w - 1)) (hfit1 : (N - 1) * s1 < 2 ^ (w' - 1)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + rows.col BN * a0 + cols.row BM * a1).get2 i j =
        b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) :=
  ptr_lane_seq_of_on b hb (colMul_toInt_on hr ha0 hfit0) (rowMul_toInt_on hc ha1 hfit1)
end Pointers

/-! ## Masks -/

/-- Whether mask `mk` enables lane `(i, j)`; without a mask every lane is enabled. -/
def on {BM BN : Nat} (mk : Option (Tensor [BM, BN] Bool)) (i : Fin BM) (j : Fin BN) : Bool :=
  match mk with
  | none => true
  | some m => m.get2 i j

/-- The mask `mk` enables lane `(i, j)` exactly when its row `r0 + i` is below `M` and its column
`c0 + j` below `N`. -/
def MaskOk {BM BN : Nat} (mk : Option (Tensor [BM, BN] Bool)) (r0 c0 M N : Nat) : Prop :=
  ∀ i j, on mk i j = (decide (r0 + i < M) && decide (c0 + j < N))

/-- The mask `mk` enables lane `(i, j)` exactly when its row `ρ i` is below `M` and its column
`κ j` below `N`, for rows and columns of any form (library issue L184). `MaskOk mk r0 c0 M N` is
`MaskAt mk (fun i => r0 + i) (fun j => c0 + j) M N` (`maskOk_iff_maskAt`; the two unfold to the
same statement, so a proof of one is a proof of the other). -/
def MaskAt {BM BN : Nat} (mk : Option (Tensor [BM, BN] Bool)) (ρ : Fin BM → Nat)
    (κ : Fin BN → Nat) (M N : Nat) : Prop :=
  ∀ i j, on mk i j = (decide (ρ i < M) && decide (κ j < N))

theorem maskOk_iff_maskAt {BM BN : Nat} {mk : Option (Tensor [BM, BN] Bool)} {r0 c0 M N : Nat} :
    MaskOk mk r0 c0 M N ↔ MaskAt mk (fun i => r0 + i) (fun j => c0 + j) M N := Iff.rfl

section Masks
variable {w BM BN r0 c0 M N : Nat}

/-- Any mask, from its lanes. -/
theorem maskOk_of_get {m : Tensor [BM, BN] Bool}
    (h : ∀ i j, m.get2 i j = (decide (r0 + i < M) && decide (c0 + j < N))) :
    MaskOk (some m) r0 c0 M N := h

/-- No mask: every lane is in the tile when `r0 + BM ≤ M` and `c0 + BN ≤ N`. -/
theorem maskOk_none (hM : r0 + BM ≤ M) (hN : c0 + BN ≤ N) :
    MaskOk (BM := BM) (BN := BN) none r0 c0 M N := by
  intro i j
  have := i.isLt; have := j.isLt
  simp only [on]; symm; simp only [Bool.and_eq_true, decide_eq_true_eq]; omega

private theorem lt_lane {x mv : IntV true w} {k K : Nat} (hx : x.toInt = (k : Int))
    (hm : mv.toInt = (K : Int)) : IntV.lt x mv = decide (k < K) := by
  rw [IntV.lt_eq, hx, hm]; simp only [Nat.cast_lt]

/-- `(rows < M)[:, None] & (cols < N)[None, :]` (`((rows.col BN).ltS mv).and ((cols.row BM).ltS nv)`),
with `mv`, `nv` of values `M`, `N`. -/
theorem maskOk_and {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)}
    {mv nv : IntV true w} (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (hmv : mv.toInt = M)
    (hnv : nv.toInt = N) :
    MaskOk (some (((rows.col BN).ltS mv).and ((cols.row BM).ltS nv))) r0 c0 M N := by
  intro i j
  show (IntV.lt (rows.get1 i) mv && IntV.lt (cols.get1 j) nv) = _
  rw [lt_lane (hr i) hmv, lt_lane (hc j) hnv]

/-- The row mask alone, `(rows < M)[:, None]` (`(rows.col BN).ltS mv`): every column is in the tile
when `c0 + BN ≤ N`. -/
theorem maskOk_rows {rows : Tensor [BM] (IntV true w)} {mv : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)) (hmv : mv.toInt = M)
    (hN : c0 + BN ≤ N) : MaskOk (some ((rows.col BN).ltS mv)) r0 c0 M N := by
  intro i j
  have := j.isLt
  show IntV.lt (rows.get1 i) mv = _
  rw [lt_lane (hr i) hmv, decide_eq_true (by omega : c0 + (j : Nat) < N), Bool.and_true]

/-- The column mask alone, `(cols < N)[None, :]` (`(cols.row BM).ltS nv`): every row is in the tile
when `r0 + BM ≤ M`. -/
theorem maskOk_cols {cols : Tensor [BN] (IntV true w)} {nv : IntV true w}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (hnv : nv.toInt = N)
    (hM : r0 + BM ≤ M) : MaskOk (some ((cols.row BM).ltS nv)) r0 c0 M N := by
  intro i j
  have := i.isLt
  show IntV.lt (cols.get1 j) nv = _
  rw [lt_lane (hc j) hnv, decide_eq_true (by omega : r0 + (i : Nat) < M), Bool.true_and]

/-! ### Rows and columns of any form -/

variable {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

/-- Any mask, from its lanes. -/
theorem maskAt_of_get {m : Tensor [BM, BN] Bool}
    (h : ∀ i j, m.get2 i j = (decide (ρ i < M) && decide (κ j < N))) :
    MaskAt (some m) ρ κ M N := h

/-- No mask: every lane is in the tile when every row is below `M` and every column below `N`
(rows `(r0 + i) % M`: `Nat.mod_lt`). -/
theorem maskAt_none (hM : ∀ i, ρ i < M) (hN : ∀ j, κ j < N) : MaskAt none ρ κ M N := by
  intro i j
  simp only [on]; symm; simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨hM i, hN j⟩

/-- `(rows < M)[:, None] & (cols < N)[None, :]` for rows and columns of any form. -/
theorem maskAt_and {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)}
    {mv nv : IntV true w} (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int)) (hmv : mv.toInt = M)
    (hnv : nv.toInt = N) :
    MaskAt (some (((rows.col BN).ltS mv).and ((cols.row BM).ltS nv))) ρ κ M N := by
  intro i j
  show (IntV.lt (rows.get1 i) mv && IntV.lt (cols.get1 j) nv) = _
  rw [lt_lane (hr i) hmv, lt_lane (hc j) hnv]

/-- The row mask alone, `(rows < M)[:, None]`: every column is in the tile when every `κ j < N`. -/
theorem maskAt_rows {rows : Tensor [BM] (IntV true w)} {mv : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int)) (hmv : mv.toInt = M)
    (hN : ∀ j, κ j < N) : MaskAt (some ((rows.col BN).ltS mv)) ρ κ M N := by
  intro i j
  show IntV.lt (rows.get1 i) mv = _
  rw [lt_lane (hr i) hmv, decide_eq_true (hN j), Bool.and_true]

/-- The column mask alone, `(cols < N)[None, :]`: every row is in the tile when every `ρ i < M`
(rows `(r0 + i) % M`, tutorial 03's `a` tile). -/
theorem maskAt_cols {cols : Tensor [BN] (IntV true w)} {nv : IntV true w}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int)) (hnv : nv.toInt = N)
    (hM : ∀ i, ρ i < M) : MaskAt (some ((cols.row BM).ltS nv)) ρ κ M N := by
  intro i j
  show IntV.lt (cols.get1 j) nv = _
  rw [lt_lane (hc j) hnv, decide_eq_true (hM i), Bool.true_and]

end Masks

/-! ## The tile's elements -/

/-- Element `x` is in the tile: `x = e + (r0 + i) * s0 + (c0 + j) * s1` for a lane `i < BM`,
`j < BN` whose row is below `M` and column below `N`. -/
def Elem (e s0 s1 r0 c0 BM BN M N x : Nat) : Prop :=
  ∃ i j, i < BM ∧ j < BN ∧ r0 + i < M ∧ c0 + j < N ∧ x = e + (r0 + i) * s0 + (c0 + j) * s1

/-- Element `x` is in the tile whose row lane `i` is `ρ i` and column lane `j` is `κ j` (rows and
columns of any form, library issue L184): `x = e + ρ i * s0 + κ j * s1` for a lane whose row is
below `M` and column below `N`. `Elem e s0 s1 r0 c0 BM BN M N` is the instance
`ρ i = r0 + i`, `κ j = c0 + j` (`elem_iff_elemAt`). -/
def ElemAt {BM BN : Nat} (e s0 s1 : Nat) (ρ : Fin BM → Nat) (κ : Fin BN → Nat) (M N x : Nat) :
    Prop :=
  ∃ i j, ρ i < M ∧ κ j < N ∧ x = e + ρ i * s0 + κ j * s1

/-- Distinct `(row, column)` pairs below `(M, N)` give distinct offsets `row * s0 + col * s1`:
the layout does not alias. -/
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

/-- Row-major with unit column stride: `N ≤ s0`. -/
theorem inj_rowMajor1 {s0 M N : Nat} (hN : N ≤ s0) : Inj s0 1 M N :=
  inj_rowMajor Nat.one_pos (by simpa using hN)

/-- **Column-major** (padded) layouts do not alias: a column's `M` elements, `s0 > 0` apart, fit in
the column stride (`M * s0 ≤ s1`). -/
theorem inj_colMajor {s0 s1 M N : Nat} (hs0 : 0 < s0) (hM : M * s0 ≤ s1) : Inj s0 s1 M N := by
  intro r c r' c' hr hc hr' hc' h
  have := inj_rowMajor (M := N) hs0 hM c r c' r' hc hr hc' hr' (by linarith)
  exact ⟨this.2, this.1⟩

/-- Column-major with unit row stride: `M ≤ s1`. -/
theorem inj_colMajor1 {s1 M N : Nat} (hM : M ≤ s1) : Inj 1 s1 M N :=
  inj_colMajor Nat.one_pos (by simpa using hM)

section Elems
variable {e s0 s1 BM BN M N : Nat}

/-- Two tiles whose rows or columns are apart by at least the block (`r0 + BM ≤ r0'`,
`r0' + BM' ≤ r0`, `c0 + BN ≤ c0'` or `c0' + BN' ≤ c0`) have no common element when the layout does
not alias (`Inj`). -/
theorem elem_disjoint {r0 c0 r0' c0' BM' BN' : Nat} (hinj : Inj s0 s1 M N)
    (hap : r0 + BM ≤ r0' ∨ r0' + BM' ≤ r0 ∨ c0 + BN ≤ c0' ∨ c0' + BN' ≤ c0) :
    Disjoint {x | Elem e s0 s1 r0 c0 BM BN M N x} {x | Elem e s0 s1 r0' c0' BM' BN' M N x} := by
  rw [Set.disjoint_left]
  rintro x ⟨i, j, hi, hj, hiM, hjN, rfl⟩ ⟨i', j', hi', hj', hiM', hjN', h⟩
  have := hinj _ _ _ _ hiM hjN hiM' hjN' (by omega)
  omega

/-- The tiles of two programs of a grid, `(pm * BM, pn * BN)` and `(pm' * BM, pn' * BN)`, with
`(pm, pn) ≠ (pm', pn')`, have no common element when the layout does not alias. -/
theorem elem_disjoint_grid {pm pn pm' pn' : Nat} (hinj : Inj s0 s1 M N)
    (hne : pm ≠ pm' ∨ pn ≠ pn') :
    Disjoint {x | Elem e s0 s1 (pm * BM) (pn * BN) BM BN M N x}
      {x | Elem e s0 s1 (pm' * BM) (pn' * BN) BM BN M N x} := by
  rw [Set.disjoint_left]
  rintro x ⟨i, j, hi, hj, hiM, hjN, rfl⟩ ⟨i', j', hi', hj', hiM', hjN', h⟩
  obtain ⟨hr, hc⟩ := hinj _ _ _ _ hiM hjN hiM' hjN' (by omega)
  rcases hne with hne | hne
  · exact hne (Block.tile_unique (B := BM) (n := pm * BM + i + 1) (j := pm * BM + i)
      ⟨by omega, by omega, by omega⟩ ⟨by omega, by omega, by omega⟩)
  · exact hne (Block.tile_unique (B := BN) (n := pn * BN + j + 1) (j := pn * BN + j)
      ⟨by omega, by omega, by omega⟩ ⟨by omega, by omega, by omega⟩)

/-- The tile's elements are below `len` when the matrix is empty (`M = 0` or `N = 0`: the tile has
no element) or its last element is (`e + (M - 1) * s0 + (N - 1) * s1 < len`). -/
theorem elem_lt {r0 c0 x len : Nat} (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < len)
    (h : Elem e s0 s1 r0 c0 BM BN M N x) : x < len := by
  obtain ⟨i, j, -, -, hiM, hjN, rfl⟩ := h
  rcases hlen with h0 | h0 | hlen
  · omega
  · omega
  have h1 : (r0 + i) * s0 ≤ (M - 1) * s0 := Nat.mul_le_mul_right s0 (by omega)
  have h2 : (c0 + j) * s1 ≤ (N - 1) * s1 := Nat.mul_le_mul_right s1 (by omega)
  omega

/-- `Elem` is `ElemAt` with rows `r0 + i` and columns `c0 + j`. -/
theorem elem_iff_elemAt {r0 c0 x : Nat} :
    Elem e s0 s1 r0 c0 BM BN M N x ↔
      ElemAt e s0 s1 (fun i : Fin BM => r0 + i) (fun j : Fin BN => c0 + j) M N x := by
  constructor
  · rintro ⟨i, j, hi, hj, hiM, hjN, rfl⟩
    exact ⟨⟨i, hi⟩, ⟨j, hj⟩, hiM, hjN, rfl⟩
  · rintro ⟨i, j, hiM, hjN, rfl⟩
    exact ⟨i, j, i.isLt, j.isLt, hiM, hjN, rfl⟩

/-- An element of a tile of any rows and columns is an element of the whole matrix (the tile
`Elem e s0 s1 0 0 M N M N`, rows and columns from `0`, `M` by `N`). -/
theorem elemAt_matrix {ρ : Fin BM → Nat} {κ : Fin BN → Nat} {x : Nat}
    (h : ElemAt e s0 s1 ρ κ M N x) : Elem e s0 s1 0 0 M N M N x := by
  obtain ⟨i, j, hiM, hjN, rfl⟩ := h
  exact ⟨ρ i, κ j, hiM, hjN, by omega, by omega, by simp⟩

/-- The elements of a tile of any rows and columns are below `len` when the matrix is empty or its
last element is (`e + (M - 1) * s0 + (N - 1) * s1 < len`). -/
theorem elemAt_lt {ρ : Fin BM → Nat} {κ : Fin BN → Nat} {x len : Nat}
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < len)
    (h : ElemAt e s0 s1 ρ κ M N x) : x < len :=
  elem_lt hlen (elemAt_matrix h)

end Elems

end Tile2

open Tile2

namespace Buf

variable {d : DType} {e s0 s1 BM BN M N : Nat}

/-- The bytes of the elements of `b` in a 2-D tile (`Tile2.Elem`). -/
def tile2 (b : Buf d) (e s0 s1 r0 c0 BM BN M N : Nat) : Set Int :=
  b.elems {x | Elem e s0 s1 r0 c0 BM BN M N x}

/-- The tile's bytes lie in `b` when the matrix is empty (`M = 0` or `N = 0`) or its last element
lies in `b` (`e + (M - 1) * s0 + (N - 1) * s1 < b.len`). -/
theorem tile2_sub_bytes (b : Buf d) {r0 c0 : Nat}
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len) :
    b.tile2 e s0 s1 r0 c0 BM BN M N ⊆ b.bytes :=
  b.elems_sub_bytes fun _ hx => elem_lt hlen hx

/-- The bytes of a tile element lie in the tile's bytes. -/
theorem span_sub_tile2 (b : Buf d) {r0 c0 x : Nat} (h : Elem e s0 s1 r0 c0 BM BN M N x) :
    Mem.span d (b.ptr (x : Int)).addr ⊆ b.tile2 e s0 s1 r0 c0 BM BN M N :=
  b.span_sub_elems h

/-- Two tiles of one buffer whose rows or columns are apart by at least the block do not overlap,
when the layout does not alias (`Tile2.Inj`: `inj_rowMajor`, `inj_colMajor`). -/
theorem tile2_disjoint (b : Buf d) {r0 c0 r0' c0' BM' BN' : Nat} (hinj : Inj s0 s1 M N)
    (hap : r0 + BM ≤ r0' ∨ r0' + BM' ≤ r0 ∨ c0 + BN ≤ c0' ∨ c0' + BN' ≤ c0) :
    Disjoint (b.tile2 e s0 s1 r0 c0 BM BN M N) (b.tile2 e s0 s1 r0' c0' BM' BN' M N) :=
  b.elems_disjoint (elem_disjoint hinj hap)

/-- The tiles of two programs `(pm, pn) ≠ (pm', pn')` of a grid do not overlap, when the layout
does not alias. -/
theorem tile2_disjoint_grid (b : Buf d) {pm pn pm' pn' : Nat} (hinj : Inj s0 s1 M N)
    (hne : pm ≠ pm' ∨ pn ≠ pn') :
    Disjoint (b.tile2 e s0 s1 (pm * BM) (pn * BN) BM BN M N)
      (b.tile2 e s0 s1 (pm' * BM) (pn' * BN) BM BN M N) :=
  b.elems_disjoint (elem_disjoint_grid hinj hne)

/-- Tiles of disjoint buffers do not overlap. -/
theorem tile2_disjoint_of {d' : DType} (b : Buf d) (c : Buf d') (hbc : Disjoint b.bytes c.bytes)
    {r0 c0 e' s0' s1' r0' c0' BM' BN' M' N' : Nat}
    (hb : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len)
    (hc : M' = 0 ∨ N' = 0 ∨ e' + (M' - 1) * s0' + (N' - 1) * s1' < c.len) :
    Disjoint (b.tile2 e s0 s1 r0 c0 BM BN M N) (c.tile2 e' s0' s1' r0' c0' BM' BN' M' N') :=
  hbc.mono (b.tile2_sub_bytes hb) (c.tile2_sub_bytes hc)

/-- The bytes of the elements of `b` in a 2-D tile of any rows and columns (`Tile2.ElemAt`). -/
def tileAt {BM BN : Nat} (b : Buf d) (e s0 s1 : Nat) (ρ : Fin BM → Nat) (κ : Fin BN → Nat)
    (M N : Nat) : Set Int :=
  b.elems {x | ElemAt e s0 s1 ρ κ M N x}

/-- `tile2` is `tileAt` with rows `r0 + i` and columns `c0 + j`. -/
theorem tile2_eq_tileAt (b : Buf d) {r0 c0 : Nat} :
    b.tile2 e s0 s1 r0 c0 BM BN M N =
      b.tileAt e s0 s1 (fun i : Fin BM => r0 + i) (fun j : Fin BN => c0 + j) M N := by
  unfold tile2 tileAt
  congr 1
  ext x
  exact elem_iff_elemAt

/-- A tile of any rows and columns lies in the whole matrix's bytes
(`b.tile2 e s0 s1 0 0 M N M N`). -/
theorem tileAt_sub_matrix (b : Buf d) {ρ : Fin BM → Nat} {κ : Fin BN → Nat} :
    b.tileAt e s0 s1 ρ κ M N ⊆ b.tile2 e s0 s1 0 0 M N M N := by
  rintro y ⟨x, hx, hy⟩
  exact ⟨x, elemAt_matrix hx, hy⟩

/-- The tile's bytes lie in `b` when the matrix is empty or its last element lies in `b`. -/
theorem tileAt_sub_bytes (b : Buf d) {ρ : Fin BM → Nat} {κ : Fin BN → Nat}
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len) :
    b.tileAt e s0 s1 ρ κ M N ⊆ b.bytes :=
  b.elems_sub_bytes fun _ hx => elemAt_lt hlen hx

/-- The bytes of a tile element lie in the tile's bytes. -/
theorem span_sub_tileAt (b : Buf d) {ρ : Fin BM → Nat} {κ : Fin BN → Nat} {x : Nat}
    (h : ElemAt e s0 s1 ρ κ M N x) :
    Mem.span d (b.ptr (x : Int)).addr ⊆ b.tileAt e s0 s1 ρ κ M N :=
  b.span_sub_elems h

end Buf

/-! ## Accesses -/

section Access
variable {d : DType} {e s0 s1 r0 c0 BM BN M N : Nat}

/-- **The lanes of a 2-D tile access, as buffer elements**: lane `(i, j)` touches element
`e + (r0 + i) * s0 + (c0 + j) * s1` when `r0 + i < M` and `c0 + j < N`, from the lanes' pointers
on the enabled lanes (`hp`, by `ptr_lane_on`, `ptr_lane_seq_on`, `ptr_lane_sum_of_on`,
`ptr_lane_seq_of_on`, their `swap` forms, or `fun i j _ _ => ptr_lane … i j`; a masked-off lane's
pointer may be anything) and the mask
(`MaskOk`, by `maskOk_and`, `maskOk_rows`, `maskOk_cols` or `maskOk_none`). -/
theorem addrs_tile2 (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)}
    (hp : ∀ (i : Fin BM) (j : Fin BN), r0 + i < M → c0 + j < N →
      ptrs.get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int))
    (hm : MaskOk mk r0 c0 M N) :
    b.Addrs (tl.addrs ptrs mk) (fun i => e + (r0 + (i.1 : Nat)) * s0 + (c0 + (i.2.1 : Nat)) * s1)
      (fun i => decide (r0 + (i.1 : Nat) < M) && decide (c0 + (i.2.1 : Nat) < N)) := by
  rintro ⟨i, j, ⟨⟩⟩
  have hmij := hm i j
  by_cases hij : r0 + (i : Nat) < M ∧ c0 + (j : Nat) < N
  · have hpij : ptrs.get (i, j, ()) = _ := hp i j hij.1 hij.2
    have hen : on mk i j = true := by rw [hmij]; simp [hij.1, hij.2]
    cases mk with
    | none => simp [tl.addrs, hpij, hij.1, hij.2]
    | some m =>
      have hm' : m.get (i, j, ()) = true := hen
      simp [tl.addrs, hm', hpij, hij.1, hij.2]
  · have hd : (decide (r0 + (i : Nat) < M) && decide (c0 + (j : Nat) < N)) = false := by
      simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]; omega
    have hen : on mk i j = false := by rw [hmij]; exact hd
    cases mk with
    | none => exact absurd hen (by simp [on])
    | some m =>
      have hm' : m.get (i, j, ()) = false := hen
      simp only [tl.addrs, hm', Bool.false_eq_true, ite_false]
      simp [hd]

/-- An enabled lane's element is in the tile. -/
theorem tile2_lane_elem (i : Idx [BM, BN])
    (h : (decide (r0 + (i.1 : Nat) < M) && decide (c0 + (i.2.1 : Nat) < N)) = true) :
    Elem e s0 s1 r0 c0 BM BN M N (e + (r0 + (i.1 : Nat)) * s0 + (c0 + (i.2.1 : Nat)) * s1) := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨i.1, i.2.1, i.1.isLt, i.2.1.isLt, h.1, h.2, rfl⟩

/-- The image of a 2-D tile's lanes is its set of elements (`Tile2.Elem`). -/
theorem img_tile2 :
    Buf.img (s := [BM, BN]) (fun i => e + (r0 + (i.1 : Nat)) * s0 + (c0 + (i.2.1 : Nat)) * s1)
      (fun i => decide (r0 + (i.1 : Nat) < M) && decide (c0 + (i.2.1 : Nat) < N)) =
      {x | Elem e s0 s1 r0 c0 BM BN M N x} := by
  ext x; constructor
  · rintro ⟨⟨i, j, ⟨⟩⟩, h, rfl⟩
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact ⟨i, j, i.isLt, j.isLt, h.1, h.2, rfl⟩
  · rintro ⟨i, j, hi, hj, hiM, hjN, rfl⟩
    exact ⟨(⟨i, hi⟩, ⟨j, hj⟩, ()), by simp [hiM, hjN], rfl⟩

/-- The bytes a 2-D tile access touches: `b.tile2`. -/
theorem accBytes_tile2 (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)}
    (hp : ∀ (i : Fin BM) (j : Fin BN), r0 + i < M → c0 + j < N →
      ptrs.get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int))
    (hm : MaskOk mk r0 c0 M N) :
    accBytes d (tl.addrs ptrs mk) = b.tile2 e s0 s1 r0 c0 BM BN M N := by
  rw [(addrs_tile2 b hp hm).accBytes, img_tile2]; rfl

/-- Two enabled lanes of a tile touch the same element only if they are the same lane, when the
layout does not alias. -/
theorem tile2_inj (hinj : Inj s0 s1 M N) (i i' : Idx [BM, BN])
    (hi : (decide (r0 + (i.1 : Nat) < M) && decide (c0 + (i.2.1 : Nat) < N)) = true)
    (hi' : (decide (r0 + (i'.1 : Nat) < M) && decide (c0 + (i'.2.1 : Nat) < N)) = true)
    (h : e + (r0 + (i.1 : Nat)) * s0 + (c0 + (i.2.1 : Nat)) * s1 =
      e + (r0 + (i'.1 : Nat)) * s0 + (c0 + (i'.2.1 : Nat)) * s1) : i = i' := by
  obtain ⟨i, j, ⟨⟩⟩ := i
  obtain ⟨i', j', ⟨⟩⟩ := i'
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hi hi'
  dsimp only at h
  have := hinj _ _ _ _ hi.1 hi.2 hi'.1 hi'.2 (by omega)
  have h1 : i = i' := Fin.ext (by omega)
  have h2 : j = j' := Fin.ext (by omega)
  rw [h1, h2]

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **2-D tile load.** Each lane of the tile (row below `M`, column below `N`) gets its element;
a masked-off lane gets `other` when given, and is unspecified otherwise. The pointers are needed on
the enabled lanes only (`hp`). The tile's elements must lie in `b` (the matrix is empty, or
`e + (M - 1) * s0 + (N - 1) * s1 < b.len`) and in the read footprint. -/
theorem safe_load_tile2 (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)} {other : Option (Tensor [BM, BN] d.Val)}
    (hp : ∀ (i : Fin BM) (j : Fin BN), r0 + i < M → c0 + j < N →
      ptrs.get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int))
    (hm : MaskOk mk r0 c0 M N)
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len)
    (hv : b.Valid m) (hR : b.tile2 e s0 s1 r0 c0 BM BN M N ⊆ R)
    {k : Tensor [BM, BN] d.Val → Prog α}
    (hk : ∀ v : Tensor [BM, BN] d.Val,
      (∀ (i : Fin BM) (j : Fin BN), r0 + i < M → c0 + j < N →
        m.read d (b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int)).addr = some (v.get2 i j)) →
      (∀ (i : Fin BM) (j : Fin BN), ¬ (r0 + i < M ∧ c0 + j < N) → ∀ o, other = some o →
        v.get2 i j = o.get2 i j) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs mk other >>= k) :=
  safe_load_buf b (addrs_tile2 b hp hm)
    (fun i h => elem_lt hlen (tile2_lane_elem i h)) hv
    (by rw [img_tile2]; exact hR)
    fun v h1 h2 => hk v (fun i j hi hj => h1 (i, j, ()) (by simp [hi, hj]))
      (fun i j hij o ho => h2 (i, j, ()) (by
        simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]; tauto) o ho)

/-- **2-D tile store.** Afterwards each lane of the tile (row below `M`, column below `N`) holds its
value, nothing outside the tile changed, and every present byte is still present. The layout must
not alias (`Tile2.Inj`: `inj_rowMajor`, `inj_colMajor`), so that the lanes write distinct
elements. As for the load, the pointers are needed on the enabled lanes only (`hp`), and the
matrix is empty or `e + (M - 1) * s0 + (N - 1) * s1 < b.len`. The rest of the program may not write
the tile again. -/
theorem safe_store_tile2 (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)} {vals : Tensor [BM, BN] d.Val}
    (hp : ∀ (i : Fin BM) (j : Fin BN), r0 + i < M → c0 + j < N →
      ptrs.get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int))
    (hm : MaskOk mk r0 c0 M N)
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len)
    (hinj : Inj s0 s1 M N) (hv : b.Valid m) (hW : b.tile2 e s0 s1 r0 c0 BM BN M N ⊆ W)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ (i : Fin BM) (j : Fin BN), r0 + i < M → c0 + j < N →
        m'.read d (b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int)).addr =
          some (vals.get2 i j)) →
      (∀ x, x ∉ b.tile2 e s0 s1 r0 c0 BM BN M N → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.tile2 e s0 s1 r0 c0 BM BN M N) Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals mk >>= k) := by
  refine safe_store_buf b (addrs_tile2 b hp hm)
    (fun i h => elem_lt hlen (tile2_lane_elem i h))
    (fun i i' hi hi' h => tile2_inj hinj i i' hi hi' h) hv
    (by rw [img_tile2]; exact hW) fun m' h1 h2 h3 => ?_
  rw [img_tile2]
  exact hk m' (fun i j hi hj => h1 (i, j, ()) (by simp [hi, hj]))
    (fun x hx => h2 x (by rw [img_tile2]; exact hx)) h3

end Access

/-! ## Accesses with rows and columns of any form

The same rules for a tile whose row lane `i` is `ρ i` and column lane `j` is `κ j` (library issue
L184), as tutorial 03's `(r0 + tl.arange(0, BM)) % M` rows with no row mask
(`irem_toInt`, `maskAt_cols`). The rules above are the instance `ρ i = r0 + i`, `κ j = c0 + j`
(`maskOk_iff_maskAt`, `elem_iff_elemAt`, `Buf.tile2_eq_tileAt`). A store needs distinct enabled
rows (`ρ` injective on the rows below `M`) and columns, so that the lanes write distinct elements;
`(r0 + i) % M` rows repeat when `BM > M`, and a load of them needs no such premise. A read
footprint that holds the whole matrix holds the tile (`Buf.tileAt_sub_matrix`). -/

section AccessAt
variable {d : DType} {e s0 s1 BM BN M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

/-- **The lanes of a 2-D tile access, as buffer elements**, rows and columns of any form: lane
`(i, j)` touches element `e + ρ i * s0 + κ j * s1` when `ρ i < M` and `κ j < N`, from the lanes'
pointers on the enabled lanes (`hp`: `ptr_lane_on` and the other `…_on`) and the mask (`MaskAt`). -/
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

/-- An enabled lane's element is in the tile. -/
theorem tileAt_lane_elem (i : Idx [BM, BN])
    (h : (decide (ρ i.1 < M) && decide (κ i.2.1 < N)) = true) :
    ElemAt e s0 s1 ρ κ M N (e + ρ i.1 * s0 + κ i.2.1 * s1) := by
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨i.1, i.2.1, h.1, h.2, rfl⟩

/-- The image of the tile's lanes is its set of elements (`Tile2.ElemAt`). -/
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

/-- The bytes the access touches: `b.tileAt`. -/
theorem accBytes_tileAt (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)}
    (hp : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      ptrs.get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int))
    (hm : MaskAt mk ρ κ M N) :
    accBytes d (tl.addrs ptrs mk) = b.tileAt e s0 s1 ρ κ M N := by
  rw [(addrs_tileAt b hp hm).accBytes, img_tileAt]; rfl

/-- Two enabled lanes touch the same element only if they are the same lane, when the layout does
not alias (`Tile2.Inj`) and distinct enabled lanes have distinct rows (`hρ`) and columns (`hκ`). -/
theorem tileAt_inj (hinj : Inj s0 s1 M N)
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

/-- Rows `r0 + i` (columns `c0 + j`) are distinct: the `hρ` (`hκ`) of `tileAt_inj` and
`safe_store_tileAt` for them. -/
theorem add_inj {B r0 L : Nat} :
    ∀ i i' : Fin B, r0 + (i : Nat) < L → r0 + (i' : Nat) < L → r0 + (i : Nat) = r0 + i' → i = i' :=
  fun _ _ _ _ h => Fin.ext (by omega)

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **2-D tile load, rows and columns of any form.** Each enabled lane (`ρ i < M`, `κ j < N`) gets
its element; a masked-off lane gets `other` when given, and is unspecified otherwise. The pointers
are needed on the enabled lanes only (`hp`). The matrix is empty or
`e + (M - 1) * s0 + (N - 1) * s1 < b.len`, and the tile's bytes are in the read footprint
(`Buf.tileAt_sub_matrix` when it holds the whole matrix). -/
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
    (fun i h => elemAt_lt hlen (tileAt_lane_elem i h)) hv
    (by rw [img_tileAt]; exact hR)
    fun v h1 h2 => hk v (fun i j hi hj => h1 (i, j, ()) (by simp [hi, hj]))
      (fun i j hij o ho => h2 (i, j, ()) (by
        simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]; tauto) o ho)

/-- **2-D tile store, rows and columns of any form.** Afterwards each enabled lane holds its value,
nothing outside the tile changed, and every present byte is still present. The layout must not
alias (`Tile2.Inj`), and distinct enabled lanes must have distinct rows (`hρ`) and columns (`hκ`;
`add_inj` for `r0 + i`), so that the lanes write distinct elements. The rest of the program may not
write the tile again. -/
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
    (fun i h => elemAt_lt hlen (tileAt_lane_elem i h))
    (fun i i' hi hi' h => tileAt_inj hinj hρ hκ i i' hi hi' h) hv
    (by rw [img_tileAt]; exact hW) fun m' h1 h2 h3 => ?_
  rw [img_tileAt]
  exact hk m' (fun i j hi hj => h1 (i, j, ()) (by simp [hi, hj]))
    (fun x hx => h2 x (by rw [img_tileAt]; exact hx)) h3

end AccessAt

end TVA
