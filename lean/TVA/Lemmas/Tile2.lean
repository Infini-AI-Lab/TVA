import TVA.Lemmas.Block

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

* **offsets** (`Tile2.off_toInt`, and the terms `colMul_toInt`, `mulCol_toInt`, `col_toInt`,
  `rowMul_toInt`, `mulRow_toInt`, `row_toInt`): each lane's offset is the integer one, with no
  overflow, from a bound on the largest offset, `(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2^(w-1)`;
  the row and column vectors from `tl.arange` (`arange_toInt`, `start_arange_toInt`,
  `start_arange64_toInt`, `start_add_toInt`, `castInt_toInt`) and the scalar `pid * BM`
  (`toInt_mul_ofNat`);
* **pointers** (`Tile2.ptr_lane`, `ptr_lane_seq`, and `ptr_lane_sum_of`, `ptr_lane_seq_of` for any
  form of the two terms, `ptr_lane_sum_swap_of`, `ptr_lane_seq_swap_of` with the column term
  first): the pointer tensor's `get2`, for `base + (x + y)` and `base + x + y`; and lane by lane,
  `Tensor.get2_ptrAdd`, `Tensor.get2_ptrAddT`;
* **masks** (`Tile2.MaskOk`: `maskOk_and`, `maskOk_rows`, `maskOk_cols`, `maskOk_none`,
  `maskOk_of_get`);
* **footprints** (`addrs_tile2`, `img_tile2`, `accBytes_tile2`, `Buf.tile2`), and the load and store
  rules `safe_load_tile2`, `safe_store_tile2`, as `Block` does for the 1-D block;
* **disjointness** of the elements of two tiles whose rows or columns are apart by at least the
  block (`Tile2.elem_disjoint`, `elem_disjoint_grid`, `Buf.tile2_disjoint`, `Buf.tile2_disjoint_grid`),
  when distinct `(row, column)` pairs give distinct elements (`Tile2.Inj`): row-major
  (`inj_rowMajor`: `N * s1 ≤ s0`, `0 < s1`; `s1 = 1`, `N ≤ s0`: `inj_rowMajor1`) or column-major
  (`inj_colMajor`: `M * s0 ≤ s1`, `0 < s0`; `s0 = 1`, `M ≤ s1`: `inj_colMajor1`).

A lane rule (`ptr_lane…`) gives `hp : ∀ i j, ptrs.get2 i j = b.ptr (e + (r0 + i) * s0 + (c0 + j) * s1)`
and a mask rule gives `MaskOk mk r0 c0 M N`; the footprint and load and store rules take those
two, so any way of writing the pointers or the mask fits. Without a row (column) mask, choose
`M` (`N`) with `r0 + BM ≤ M` (`c0 + BN ≤ N`): the tile's elements are then the same.
-/
namespace TVA

namespace Tile2

/-! ## Integer arithmetic -/

section IntArith
variable {w : Nat}

private theorem sfits_nat {x : Nat} (h : x < 2 ^ (w - 1)) : IntV.SFits w (x : Int) := by
  have h' : (x : Int) < 2 ^ (w - 1) := by exact_mod_cast h
  have : (0 : Int) ≤ 2 ^ (w - 1) := by positivity
  exact ⟨by omega, h'⟩

/-- A product of non-negative values below `2^(w-1)` does not wrap. -/
theorem toInt_mul_nat {a b : IntV true w} {x y : Nat} (hw : 0 < w) (ha : a.toInt = x)
    (hb : b.toInt = y) (h : x * y < 2 ^ (w - 1)) : (a * b).toInt = ((x * y : Nat) : Int) := by
  have hs : IntV.SFits w (a.toInt * b.toInt) := by
    rw [ha, hb]; exact_mod_cast sfits_nat h
  rw [IntV.toInt_mul hw hs, ha, hb]; push_cast; rfl

/-- A sum of non-negative values below `2^(w-1)` does not wrap. -/
theorem toInt_add_nat {a b : IntV true w} {x y : Nat} (hw : 0 < w) (ha : a.toInt = x)
    (hb : b.toInt = y) (h : x + y < 2 ^ (w - 1)) : (a + b).toInt = ((x + y : Nat) : Int) := by
  have hs : IntV.SFits w (a.toInt + b.toInt) := by
    rw [ha, hb]; exact_mod_cast sfits_nat h
  rw [IntV.toInt_add hw hs, ha, hb]; push_cast; rfl

/-- `pid * BM` for a block size `B` (`IntV.ofNat B`): the product `p * B`, when below `2^(w-1)`.
`B` need not fit by itself when `p = 0`. -/
theorem toInt_mul_ofNat {a : IntV true w} {p B : Nat} (hw : 0 < w) (ha : a.toInt = p)
    (h : p * B < 2 ^ (w - 1)) : (a * IntV.ofNat B).toInt = ((p * B : Nat) : Int) := by
  rcases Nat.eq_zero_or_pos p with rfl | hp
  · have hs : IntV.SFits w (a.toInt * (IntV.ofNat B : IntV true w).toInt) := by
      rw [ha]; simp only [Nat.cast_zero, zero_mul]; exact_mod_cast sfits_nat (x := 0) (by omega)
    rw [IntV.toInt_mul hw hs, ha]; simp
  · have hB : B < 2 ^ (w - 1) := lt_of_le_of_lt (Nat.le_mul_of_pos_left B hp) h
    exact toInt_mul_nat hw ha (IntV.toInt_ofNat hw (by exact_mod_cast hB)) h

/-- `tl.arange(0, n)`: lane `i` is `i` (written `0 + i`, the form `r0 + i` of a row or column
vector with `r0 = 0`). -/
theorem arange_toInt {n : Nat} (hn : n ≤ 2 ^ 31) (i : Fin (n - 0)) :
    ((tl.arange 0 n).get1 i).toInt = ((0 + i : Nat) : Int) := by
  have hi := i.isLt
  simp only [Nat.sub_zero] at hi
  show (IntV.ofInt (((0 : Nat) : Int) + (i : Nat)) : I32).toInt = _
  rw [IntV.I32.toInt_ofInt] <;> push_cast
  · ring
  · constructor <;> omega

/-- `a + x` for a scalar `a` (the first row or column, `r0`) and a vector `x` of lanes `0 + i`
(`tl.arange(0, n)`, cast or not): lane `i` is `r0 + i`, when `r0 + n ≤ 2^(w-1)`. -/
theorem start_add_toInt {n r0 : Nat} {a : IntV true w} {x : Tensor [n] (IntV true w)} (hw : 0 < w)
    (ha : a.toInt = r0) (hx : ∀ i : Fin n, (x.get1 i).toInt = ((0 + i : Nat) : Int))
    (hfit : r0 + n ≤ 2 ^ (w - 1)) (i : Fin n) :
    ((a + x).get1 i).toInt = ((r0 + i : Nat) : Int) := by
  have hi := i.isLt
  show (a + x.get1 i).toInt = _
  have h := toInt_add_nat hw ha (hx i) (by omega)
  rw [h]; congr 1; omega

/-- `a + tl.arange(0, n)` in `int32` (`pid * BM + tl.arange(0, BM)`, with `a.toInt = r0`): lane
`i` is `r0 + i`, when `r0 + n ≤ 2^31`. -/
theorem start_arange_toInt {n r0 : Nat} {a : I32} (ha : a.toInt = r0) (hfit : r0 + n ≤ 2 ^ 31)
    (i : Fin (n - 0)) : ((a + tl.arange 0 n).get1 i).toInt = ((r0 + i : Nat) : Int) :=
  start_add_toInt (by norm_num) ha (arange_toInt (by omega)) (by simpa using hfit) i

/-- `castInt` to a wider signed type keeps each lane's value. -/
theorem castInt_toInt {n w' : Nat} {x : Tensor [n] (IntV true w)} {v : Fin n → Int}
    (hx : ∀ i, (x.get1 i).toInt = v i) (h : w ≤ w') (i : Fin n) :
    ((x.castInt : Tensor [n] (IntV true w')).get1 i).toInt = v i := by
  show (IntV.cast (x.get1 i) : IntV true w').toInt = _
  rw [IntV.toInt_cast _ h, hx]

/-- `a + tl.arange(0, n)` in `int64` (`pid.to(tl.int64) * BM + tl.arange(0, BM)`: the `int32`
`arange` is cast to `int64`): lane `i` is `r0 + i`, when `r0 + n ≤ 2^63`. -/
theorem start_arange64_toInt {n r0 : Nat} {a : I64} (ha : a.toInt = r0) (hfit : r0 + n ≤ 2 ^ 63)
    (hn : n ≤ 2 ^ 31) (i : Fin (n - 0)) :
    ((a + ((tl.arange 0 n).castInt : Tensor [n - 0] I64)).get1 i).toInt = ((r0 + i : Nat) : Int) :=
  start_add_toInt (by norm_num) ha (castInt_toInt (arange_toInt hn) (by norm_num))
    (by simpa using hfit) i

end IntArith

/-! ## Lane offsets -/

section Offsets
variable {w BM BN r0 c0 s0 s1 : Nat}

private theorem row_le {i : Nat} (hi : i < BM) (s : Nat) : (r0 + i) * s ≤ (r0 + BM - 1) * s :=
  Nat.mul_le_mul_right s (by omega)

/-- `rows[:, None] * a` (`rows.col BN * a`): lane `(i, j)` is `(r0 + i) * s0`. -/
theorem colMul_toInt {rows : Tensor [BM] (IntV true w)} {a : IntV true w} (hw : 0 < w)
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)) (ha : a.toInt = s0)
    (hfit : (r0 + BM - 1) * s0 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN * a).get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int) :=
  toInt_mul_nat hw (hr i) ha (lt_of_le_of_lt (row_le i.isLt s0) hfit)

/-- `a * rows[:, None]` (`a * rows.col BN`): lane `(i, j)` is `(r0 + i) * s0`. -/
theorem mulCol_toInt {rows : Tensor [BM] (IntV true w)} {a : IntV true w} (hw : 0 < w)
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)) (ha : a.toInt = s0)
    (hfit : (r0 + BM - 1) * s0 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((a * rows.col BN).get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int) := by
  rw [show (((r0 + i) * s0 : Nat) : Int) = ((s0 * (r0 + i) : Nat) : Int) by rw [Nat.mul_comm]]
  exact toInt_mul_nat hw ha (hr i) (by rw [Nat.mul_comm]; exact lt_of_le_of_lt (row_le i.isLt s0) hfit)

/-- `rows[:, None]` with a unit row stride: lane `(i, j)` is `(r0 + i) * 1`. -/
theorem col_toInt {rows : Tensor [BM] (IntV true w)}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN).get2 i j).toInt = (((r0 + i) * 1 : Nat) : Int) := by
  rw [Nat.mul_one]; exact hr i

/-- `cols[None, :] * a` (`cols.row BM * a`): lane `(i, j)` is `(c0 + j) * s1`. -/
theorem rowMul_toInt {cols : Tensor [BN] (IntV true w)} {a : IntV true w} (hw : 0 < w)
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (ha : a.toInt = s1)
    (hfit : (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((cols.row BM * a).get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int) :=
  toInt_mul_nat hw (hc j) ha (lt_of_le_of_lt (row_le j.isLt s1) hfit)

/-- `a * cols[None, :]` (`a * cols.row BM`): lane `(i, j)` is `(c0 + j) * s1`. -/
theorem mulRow_toInt {cols : Tensor [BN] (IntV true w)} {a : IntV true w} (hw : 0 < w)
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (ha : a.toInt = s1)
    (hfit : (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((a * cols.row BM).get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int) := by
  rw [show (((c0 + j) * s1 : Nat) : Int) = ((s1 * (c0 + j) : Nat) : Int) by rw [Nat.mul_comm]]
  exact toInt_mul_nat hw ha (hc j) (by rw [Nat.mul_comm]; exact lt_of_le_of_lt (row_le j.isLt s1) hfit)

/-- `cols[None, :]` with a unit column stride: lane `(i, j)` is `(c0 + j) * 1`. -/
theorem row_toInt {cols : Tensor [BN] (IntV true w)}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    ((cols.row BM).get2 i j).toInt = (((c0 + j) * 1 : Nat) : Int) := by
  rw [Nat.mul_one]; exact hc j

/-- **The sum of a row term and a column term** (`x + y`, any form of either): lane `(i, j)` is
`(r0 + i) * s0 + (c0 + j) * s1`, with no overflow when the largest offset,
`(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1`, is below `2^(w-1)`. -/
theorem add_toInt {x y : Tensor [BM, BN] (IntV true w)} (hw : 0 < w)
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int))
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((x + y).get2 i j).toInt = (((r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  toInt_add_nat hw (hx i j) (hy i j)
    (lt_of_le_of_lt (Nat.add_le_add (row_le i.isLt s0) (row_le j.isLt s1)) hfit)

/-- **The offset of a 2-D tile**, `rows[:, None] * a0 + cols[None, :] * a1` in a signed width `w`
(`int32`: `w = 32`, `int64`: `w = 64`), with `rows` lanes `r0 + i`, `cols` lanes `c0 + j` and the
strides `a0`, `a1` of values `s0`, `s1`: lane `(i, j)` is `(r0 + i) * s0 + (c0 + j) * s1`, with no
overflow when the largest offset, `(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1`, is below `2^(w-1)`. -/
theorem off_toInt {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)}
    {a0 a1 : IntV true w} (hw : 0 < w)
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN * a0 + cols.row BM * a1).get2 i j).toInt =
      (((r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  add_toInt hw (colMul_toInt hw hr ha0 (by omega)) (rowMul_toInt hw hc ha1 (by omega)) hfit i j

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

section Pointers
variable {d : DType} {w w' BM BN r0 c0 s0 s1 e : Nat}

/-- **The pointers `base + (x + y)`** (one offset tensor, the sum of a row term and a column term
of any form), with `base = b.ptr e`: lane `(i, j)` points at element
`e + (r0 + i) * s0 + (c0 + j) * s1`, when the largest offset is below `2^(w-1)`. -/
theorem ptr_lane_sum_of (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {x y : Tensor [BM, BN] (IntV true w)} (hw : 0 < w)
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int))
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    (base + (x + y)).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  show base.offset ((x + y).get2 i j).toInt = _
  rw [add_toInt hw hx hy hfit, hb, Buf.ptr_offset]; push_cast; ring_nf

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
    {x y : Tensor [BM, BN] (IntV true w)} (hw : 0 < w)
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int))
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    (base + (y + x)).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  have hs : ((y + x).get2 i j).toInt = (((r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
    rw [show (r0 + (i : Nat)) * s0 + (c0 + j) * s1 = (c0 + j) * s1 + (r0 + i) * s0 by ring]
    exact toInt_add_nat hw (hy i j) (hx i j)
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
    (hw : 0 < w) (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    (base + (rows.col BN * a0 + cols.row BM * a1)).get2 i j =
      b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  ptr_lane_sum_of b hb hw (colMul_toInt hw hr ha0 (by omega)) (rowMul_toInt hw hc ha1 (by omega))
    hfit i j

/-- **The pointers of a 2-D tile, added one term at a time**,
`base + rows[:, None] * a0 + cols[None, :] * a1` with `base = b.ptr e` (the row term in width `w`,
the column term in width `w'`): lane `(i, j)` points at element `e + (r0 + i) * s0 + (c0 + j) * s1`,
when each term's largest value is below its bound. -/
theorem ptr_lane_seq (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w')} {a0 : IntV true w}
    {a1 : IntV true w'} (hw : 0 < w) (hw' : 0 < w')
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit0 : (r0 + BM - 1) * s0 < 2 ^ (w - 1)) (hfit1 : (c0 + BN - 1) * s1 < 2 ^ (w' - 1))
    (i : Fin BM) (j : Fin BN) :
    (base + rows.col BN * a0 + cols.row BM * a1).get2 i j =
      b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  ptr_lane_seq_of b hb (colMul_toInt hw hr ha0 hfit0) (rowMul_toInt hw' hc ha1 hfit1) i j

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

end Masks

/-! ## The tile's elements -/

/-- Element `x` is in the tile: `x = e + (r0 + i) * s0 + (c0 + j) * s1` for a lane `i < BM`,
`j < BN` whose row is below `M` and column below `N`. -/
def Elem (e s0 s1 r0 c0 BM BN M N x : Nat) : Prop :=
  ∃ i j, i < BM ∧ j < BN ∧ r0 + i < M ∧ c0 + j < N ∧ x = e + (r0 + i) * s0 + (c0 + j) * s1

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

/-- The tile's elements are below `e + (M - 1) * s0 + (N - 1) * s1 + 1`. -/
theorem elem_lt {r0 c0 x len : Nat} (hlen : e + (M - 1) * s0 + (N - 1) * s1 < len)
    (h : Elem e s0 s1 r0 c0 BM BN M N x) : x < len := by
  obtain ⟨i, j, -, -, hiM, hjN, rfl⟩ := h
  have h1 : (r0 + i) * s0 ≤ (M - 1) * s0 := Nat.mul_le_mul_right s0 (by omega)
  have h2 : (c0 + j) * s1 ≤ (N - 1) * s1 := Nat.mul_le_mul_right s1 (by omega)
  omega

end Elems

end Tile2

open Tile2

namespace Buf

variable {d : DType} {e s0 s1 BM BN M N : Nat}

/-- The bytes of the elements of `b` in a 2-D tile (`Tile2.Elem`). -/
def tile2 (b : Buf d) (e s0 s1 r0 c0 BM BN M N : Nat) : Set Int :=
  b.elems {x | Elem e s0 s1 r0 c0 BM BN M N x}

/-- The tile's bytes lie in `b` when its last element does (`e + (M - 1) * s0 + (N - 1) * s1 < b.len`). -/
theorem tile2_sub_bytes (b : Buf d) {r0 c0 : Nat} (hlen : e + (M - 1) * s0 + (N - 1) * s1 < b.len) :
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
    (hb : e + (M - 1) * s0 + (N - 1) * s1 < b.len)
    (hc : e' + (M' - 1) * s0' + (N' - 1) * s1' < c.len) :
    Disjoint (b.tile2 e s0 s1 r0 c0 BM BN M N) (c.tile2 e' s0' s1' r0' c0' BM' BN' M' N') :=
  hbc.mono (b.tile2_sub_bytes hb) (c.tile2_sub_bytes hc)

end Buf

/-! ## Accesses -/

section Access
variable {d : DType} {e s0 s1 r0 c0 BM BN M N : Nat}

/-- **The lanes of a 2-D tile access, as buffer elements**: lane `(i, j)` touches element
`e + (r0 + i) * s0 + (c0 + j) * s1` when `r0 + i < M` and `c0 + j < N`, from the lanes' pointers
(`hp`, by `ptr_lane`, `ptr_lane_seq`, `ptr_lane_sum_of` or `ptr_lane_seq_of`) and the mask
(`MaskOk`, by `maskOk_and`, `maskOk_rows`, `maskOk_cols` or `maskOk_none`). -/
theorem addrs_tile2 (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)}
    (hp : ∀ i j, ptrs.get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int))
    (hm : MaskOk mk r0 c0 M N) :
    b.Addrs (tl.addrs ptrs mk) (fun i => e + (r0 + (i.1 : Nat)) * s0 + (c0 + (i.2.1 : Nat)) * s1)
      (fun i => decide (r0 + (i.1 : Nat) < M) && decide (c0 + (i.2.1 : Nat) < N)) := by
  rintro ⟨i, j, ⟨⟩⟩
  have hmij := hm i j
  have hpij : ptrs.get (i, j, ()) = _ := hp i j
  cases mk with
  | none =>
    simp only [on] at hmij
    simp only [tl.addrs, ← hmij, ite_true, hpij]
  | some m =>
    simp only [on] at hmij
    simp only [tl.addrs]
    rw [show m.get (i, j, ()) = m.get2 i j from rfl, hmij, hpij]

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
    (hp : ∀ i j, ptrs.get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int))
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
a masked-off lane gets `other` when given, and is unspecified otherwise. The tile's elements must
lie in `b` (`e + (M - 1) * s0 + (N - 1) * s1 < b.len`) and in the read footprint. -/
theorem safe_load_tile2 (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)} {other : Option (Tensor [BM, BN] d.Val)}
    (hp : ∀ i j, ptrs.get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int))
    (hm : MaskOk mk r0 c0 M N) (hlen : e + (M - 1) * s0 + (N - 1) * s1 < b.len)
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
elements. The rest of the program may not write the tile again. -/
theorem safe_store_tile2 (b : Buf d) {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)} {vals : Tensor [BM, BN] d.Val}
    (hp : ∀ i j, ptrs.get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int))
    (hm : MaskOk mk r0 c0 M N) (hlen : e + (M - 1) * s0 + (N - 1) * s1 < b.len)
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

end TVA
