import TVA.Proof.Basic
import TVA.Proof.Int

/-!
# Two-dimensional tiles: rows, columns, offsets, pointers

The offsets of every matrix operand:

```python
rows = r0 + tl.arange(0, BM)          # or pid_m * BM + tl.arange(0, BM), or (…) % M
cols = c0 + tl.arange(0, BN)
ptrs = base + rows[:, None] * s0 + cols[None, :] * s1
```

In the model `rows[:, None]` is `rows.col BN`, `cols[None, :]` is `cols.row BM`, and the strides
are scalars `a0`, `a1` of values `s0`, `s1`. With `base = b.ptr e`, lane `(i, j)` points at element
`e + (r0 + i) * s0 + (c0 + j) * s1` of `b`. This file gives that, in `int32` or `int64` (any signed
width `w`); `TVA.Proof.Pattern.Tile` turns it into footprints and load and store rules.

* **Rows and columns.** `start_add_toInt` (`r0 + tl.arange(0, BM)`, any width; in `int32`
  `IntV.I32.toInt_sadd_arange`), `arange64_toInt`, `start_arange64_toInt` (`tl.arange` converted
  to `int64`, the model of `pid.to(tl.int64) * BM + tl.arange(0, BM)`), and rows wrapped by `% M`
  (`safe_irem_lanes`, from `irem_ok`, `irem_toInt` of `TVA.Proof.Int.Lanes`: lane `i` of
  `(r0 + tl.arange(0, BM)) % M` is `(r0 + i) % M`, below `M`).
* **Offsets on every lane** (`colMul_toInt`, `mulCol_toInt`, `rowMul_toInt`, `mulRow_toInt`,
  `col_toInt`, `row_toInt`, `add_toInt`, `off_toInt`): lane `(i, j)` is
  `(r0 + i) * s0 + (c0 + j) * s1` with no overflow when the tile's largest offset,
  `(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1`, is below `2^(w-1)`.
* **Offsets on the enabled lanes** (the `…_on` forms): rows and columns of any form, row lane `i`
  of value `ρ i` and column lane `j` of value `κ j` (`ρ i = r0 + i` and `ρ i = (r0 + i) % M` are two
  instances); lane `(i, j)` with `ρ i < M` and `κ j < N` is `ρ i * s0 + κ j * s1` when the matrix's
  largest offset, `(M - 1) * s0 + (N - 1) * s1`, is below `2^(w-1)`. A masked-off lane's offset may
  wrap: nothing uses it. Use these on the last partial tile when the claim bounds only the matrix.
* **Pointers** (`ptr_lane`, `ptr_lane_seq`, and `ptr_lane_sum_of`, `ptr_lane_seq_of` for any form
  of the two terms, `ptr_lane_sum_swap_of`, `ptr_lane_seq_swap_of` with the column term first; on
  the enabled lanes the same names with `_on`): `base + (x + y)` and `base + x + y`.
  `Tensor.get2_ptrAdd`, `Tensor.get2_ptrAddT` (`simp`) give a pointer lane by lane.

The access rules take the pointers on the enabled lanes,
`∀ i j, ρ i < M → κ j < N → ptrs.get2 i j = b.ptr (e + ρ i * s0 + κ j * s1)`: an `…_on` lemma gives
that directly, an every-lane lemma as `fun i j _ _ => ptr_lane … i j`.
-/
namespace TVA

/-- Lane `(i, j)` of `p + offs` (a pointer plus a 2-D offset tensor) is `p` offset by lane
`(i, j)` of `offs`. -/
@[simp] theorem Tensor.get2_ptrAdd {m n : Nat} {d : DType} {sg : Bool} {w : Nat} (p : Ptr d)
    (offs : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    (p + offs).get2 i j = p.offset (offs.get2 i j).toInt := rfl

/-- Lane `(i, j)` of `ps + offs` (2-D pointers plus 2-D offsets) is lane `(i, j)` of `ps` offset
by lane `(i, j)` of `offs`. -/
@[simp] theorem Tensor.get2_ptrAddT {m n : Nat} {d : DType} {sg : Bool} {w : Nat}
    (ps : Tensor [m, n] (Ptr d)) (offs : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    (ps + offs).get2 i j = (ps.get2 i j).offset (offs.get2 i j).toInt := rfl

namespace Tile2

/-! ## Rows and columns -/

section Vectors
variable {w : Nat}

/-- **`r0 + tl.arange(0, n)` in any width**: `a + x` for a scalar `a` of value `r0` and a vector
`x` whose lane `i` has value `i` (`IntV.I32.toInt_tl_arange`, or `arange64_toInt` in `int64`):
lane `i` is `r0 + i` when `r0 + n ≤ 2^(w-1)`. -/
theorem start_add_toInt {n r0 : Nat} {a : IntV true w} {x : Tensor [n] (IntV true w)}
    (ha : a.toInt = r0) (hx : ∀ i : Fin n, (x.get1 i).toInt = ((i : Nat) : Int))
    (hfit : r0 + n ≤ 2 ^ (w - 1)) (i : Fin n) :
    ((a + x).get1 i).toInt = ((r0 + i : Nat) : Int) := by
  have hi := i.isLt
  rw [Tensor.get1_sadd]
  exact IntV.add_nat ha (hx i) (by omega)

/-- **`tl.arange(0, n)` converted to `int64`** (`tl.convert .i32 .i64`, the model of both
`tl.arange(0, BM).to(tl.int64)` and the implicit promotion in
`pid.to(tl.int64) * BM + tl.arange(0, BM)`): from the lanes `safe_convert` gives, lane `i` has
value `i`. -/
theorem arange64_toInt {P : Platform} {n : Nat} {x : Tensor [n - 0] I64}
    (hx : ∀ i, ConvRel P .i32 .i64 ((tl.arange 0 n).get i) (x.get i)) (hn : n ≤ 2 ^ 31)
    (i : Fin (n - 0)) : (x.get1 i).toInt = ((i : Nat) : Int) := by
  rw [show x.get1 i = x.get (i, ()) from rfl, ConvRel.toInt_i32_i64 (hx (i, ()))]
  exact IntV.I32.toInt_tl_arange hn i

/-- **`pid.to(tl.int64) * BM + tl.arange(0, BM)`** as the model writes it, `a + x` with `x` the
`arange` converted to `int64`: lane `i` is `r0 + i` when `r0 + n ≤ 2^63`. -/
theorem start_arange64_toInt {P : Platform} {n r0 : Nat} {a : I64} {x : Tensor [n - 0] I64}
    (ha : a.toInt = r0) (hx : ∀ i, ConvRel P .i32 .i64 ((tl.arange 0 n).get i) (x.get i))
    (hfit : r0 + n ≤ 2 ^ 63) (hn : n ≤ 2 ^ 31) (i : Fin (n - 0)) :
    ((a + x).get1 i).toInt = ((r0 + i : Nat) : Int) :=
  start_add_toInt ha (arange64_toInt hx hn) (by simpa using hfit) i

/-- **Rows wrapped by `% M`**, `tl.irem x y` with `x`'s lane `i` of natural value `f i` and every
lane of `y` of value `M > 0` (`offs_am = (pid_m * BM + tl.arange(0, BM)) % M`): the rest of the
program runs on a vector whose lane `i` is `f i % M`, below `M`, so every row is in the matrix
(`maskAt_none`, `maskAt_cols` with `ρ i = f i % M`). -/
theorem safe_irem_lanes {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {n M : Nat} {f : Fin n → Nat} {x y : Tensor [n] (IntV true w)}
    {k : Tensor [n] (IntV true w) → Prog α}
    (hx : ∀ i : Fin n, (x.get1 i).toInt = ((f i : Nat) : Int))
    (hy : ∀ i : Fin n, (y.get1 i).toInt = (M : Int)) (hM : 0 < M)
    (hk : ∀ r : Tensor [n] (IntV true w),
      (∀ i : Fin n, (r.get1 i).toInt = ((f i % M : Nat) : Int)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.irem x y >>= k) :=
  safe_irem (irem_ok hx hy hM) fun r hr => hk r (irem_toInt hx hy hM hr)

end Vectors

/-! ## Offsets on every lane -/

section Offsets
variable {w BM BN r0 c0 s0 s1 M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

private theorem row_le {k B : Nat} (hk : k < B) (r s : Nat) : (r + k) * s ≤ (r + B - 1) * s :=
  Nat.mul_le_mul_right s (by omega)

private theorem le_pred {k K : Nat} (h : k < K) (s : Nat) : k * s ≤ (K - 1) * s :=
  Nat.mul_le_mul_right s (by omega)

/-- `rows[:, None] * a` (`rows.col BN * a`): lane `(i, j)` is `(r0 + i) * s0` when
`(r0 + BM - 1) * s0 < 2^(w-1)`. -/
theorem colMul_toInt {rows : Tensor [BM] (IntV true w)} {a : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)) (ha : a.toInt = s0)
    (hfit : (r0 + BM - 1) * s0 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN * a).get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int) :=
  IntV.mul_nat (hr i) ha (lt_of_le_of_lt (row_le i.isLt r0 s0) hfit)

/-- `a * rows[:, None]` (`a * rows.col BN`): lane `(i, j)` is `(r0 + i) * s0` when
`(r0 + BM - 1) * s0 < 2^(w-1)`. -/
theorem mulCol_toInt {rows : Tensor [BM] (IntV true w)} {a : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int)) (ha : a.toInt = s0)
    (hfit : (r0 + BM - 1) * s0 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((a * rows.col BN).get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int) := by
  rw [Nat.mul_comm]
  exact IntV.mul_nat ha (hr i)
    (by rw [Nat.mul_comm]; exact lt_of_le_of_lt (row_le i.isLt r0 s0) hfit)

/-- `rows[:, None]` with a unit row stride, rows of any form: lane `(i, j)` is `ρ i * 1`, on every
lane, with no bound (nothing is multiplied). -/
theorem col_toInt {rows : Tensor [BM] (IntV true w)}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN).get2 i j).toInt = ((ρ i * 1 : Nat) : Int) := by
  rw [Nat.mul_one]; exact hr i

/-- `cols[None, :] * a` (`cols.row BM * a`): lane `(i, j)` is `(c0 + j) * s1` when
`(c0 + BN - 1) * s1 < 2^(w-1)`. -/
theorem rowMul_toInt {cols : Tensor [BN] (IntV true w)} {a : IntV true w}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (ha : a.toInt = s1)
    (hfit : (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((cols.row BM * a).get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int) :=
  IntV.mul_nat (hc j) ha (lt_of_le_of_lt (row_le j.isLt c0 s1) hfit)

/-- `a * cols[None, :]` (`a * cols.row BM`): lane `(i, j)` is `(c0 + j) * s1` when
`(c0 + BN - 1) * s1 < 2^(w-1)`. -/
theorem mulRow_toInt {cols : Tensor [BN] (IntV true w)} {a : IntV true w}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (ha : a.toInt = s1)
    (hfit : (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((a * cols.row BM).get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int) := by
  rw [Nat.mul_comm]
  exact IntV.mul_nat ha (hc j)
    (by rw [Nat.mul_comm]; exact lt_of_le_of_lt (row_le j.isLt c0 s1) hfit)

/-- `cols[None, :]` with a unit column stride, columns of any form: lane `(i, j)` is `κ j * 1`, on
every lane, with no bound. -/
theorem row_toInt {cols : Tensor [BN] (IntV true w)}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    ((cols.row BM).get2 i j).toInt = ((κ j * 1 : Nat) : Int) := by
  rw [Nat.mul_one]; exact hc j

/-- **A row term plus a column term** (`x + y`, each of any form): lane `(i, j)` is
`(r0 + i) * s0 + (c0 + j) * s1` when the tile's largest offset is below `2^(w-1)`. -/
theorem add_toInt {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int))
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((x + y).get2 i j).toInt = (((r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  IntV.add_nat (hx i j) (hy i j)
    (lt_of_le_of_lt (Nat.add_le_add (row_le i.isLt r0 s0) (row_le j.isLt c0 s1)) hfit)

/-- **The offsets of a 2-D tile**, `rows[:, None] * a0 + cols[None, :] * a1`: lane `(i, j)` is
`(r0 + i) * s0 + (c0 + j) * s1` when the tile's largest offset,
`(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1`, is below `2^(w-1)`. -/
theorem off_toInt {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)}
    {a0 a1 : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN * a0 + cols.row BM * a1).get2 i j).toInt =
      (((r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  add_toInt (colMul_toInt hr ha0 (by omega)) (rowMul_toInt hc ha1 (by omega)) hfit i j

/-! ## Offsets on the enabled lanes -/

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
  rw [Nat.mul_comm]
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
  rw [Nat.mul_comm]
  exact IntV.mul_nat ha (hc j) (by rw [Nat.mul_comm]; exact lt_of_le_of_lt (le_pred hj s1) hfit)

/-- **A row term plus a column term on an enabled lane** (`ρ i < M`, `κ j < N`):
`ρ i * s0 + κ j * s1` when the matrix's largest offset, `(M - 1) * s0 + (N - 1) * s1`, is below
`2^(w-1)`. The terms need hold only on rows (columns) in the matrix: `colMul_toInt_on`, …, or with
a unit stride `fun i j _ => col_toInt hr i j`. -/
theorem add_toInt_on {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → (x.get2 i j).toInt = ((ρ i * s0 : Nat) : Int))
    (hy : ∀ (i : Fin BM) (j : Fin BN), κ j < N → (y.get2 i j).toInt = ((κ j * s1 : Nat) : Int))
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN)
    (hi : ρ i < M) (hj : κ j < N) :
    ((x + y).get2 i j).toInt = ((ρ i * s0 + κ j * s1 : Nat) : Int) :=
  IntV.add_nat (hx i j hi) (hy i j hj)
    (lt_of_le_of_lt (Nat.add_le_add (le_pred hi s0) (le_pred hj s1)) hfit)

/-- **The offsets of a 2-D tile on an enabled lane**, `rows[:, None] * a0 + cols[None, :] * a1`:
lane `(i, j)` with `ρ i < M` and `κ j < N` is `ρ i * s0 + κ j * s1` when the matrix's largest
offset, `(M - 1) * s0 + (N - 1) * s1`, is below `2^(w-1)`. -/
theorem off_toInt_on {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)}
    {a0 a1 : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN)
    (hi : ρ i < M) (hj : κ j < N) :
    ((rows.col BN * a0 + cols.row BM * a1).get2 i j).toInt =
      ((ρ i * s0 + κ j * s1 : Nat) : Int) :=
  add_toInt_on (colMul_toInt_on hr ha0 (by omega)) (rowMul_toInt_on hc ha1 (by omega)) hfit
    i j hi hj

end Offsets

/-! ## Pointers -/

section Pointers
variable {d : DType} {w w' BM BN r0 c0 s0 s1 e M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}

private theorem row_le' {r0 BM i : Nat} (hi : i < BM) (s : Nat) :
    (r0 + i) * s ≤ (r0 + BM - 1) * s :=
  Nat.mul_le_mul_right s (by omega)

private theorem le_pred' {k K : Nat} (h : k < K) (s : Nat) : k * s ≤ (K - 1) * s :=
  Nat.mul_le_mul_right s (by omega)

/-- **The pointers `base + (x + y)`** (one offset tensor: a row term plus a column term, each of
any form), `base = b.ptr e`: lane `(i, j)` points at element `e + (r0 + i) * s0 + (c0 + j) * s1`
when the tile's largest offset is below `2^(w-1)`. -/
theorem ptr_lane_sum_of (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int))
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    (base + (x + y)).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  rw [Tensor.get2_ptrAdd, add_toInt hx hy hfit, hb, Buf.ptr_offset]; push_cast; ring_nf

/-- **The pointers `base + x + y`** (two pointer additions, a row term and a column term of any
form, possibly of different widths), `base = b.ptr e`: lane `(i, j)` points at element
`e + (r0 + i) * s0 + (c0 + j) * s1`. Each term's own lemma bounds that term. -/
theorem ptr_lane_seq_of (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {sg sg' : Bool} {x : Tensor [BM, BN] (IntV sg w)} {y : Tensor [BM, BN] (IntV sg' w')}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    (base + x + y).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  rw [Tensor.get2_ptrAddT, Tensor.get2_ptrAdd, hx, hy, hb, Buf.ptr_offset, Buf.ptr_offset]
  push_cast; ring_nf

/-- `ptr_lane_sum_of` with the column term first, `base + (y + x)`. -/
theorem ptr_lane_sum_swap_of (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int))
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    (base + (y + x)).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  have hs : ((y + x).get2 i j).toInt = (((c0 + j) * s1 + (r0 + i) * s0 : Nat) : Int) :=
    IntV.add_nat (hy i j) (hx i j)
      (by have := Nat.add_le_add (row_le' (r0 := r0) i.isLt s0) (row_le' (r0 := c0) j.isLt s1)
          omega)
  rw [Tensor.get2_ptrAdd, hs, hb, Buf.ptr_offset]; push_cast; ring_nf

/-- `ptr_lane_seq_of` with the column term added first, `base + y + x` (as in
`K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d`, a transposed load). -/
theorem ptr_lane_seq_swap_of (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {sg sg' : Bool} {x : Tensor [BM, BN] (IntV sg w)} {y : Tensor [BM, BN] (IntV sg' w')}
    (hx : ∀ i j, (x.get2 i j).toInt = (((r0 + i) * s0 : Nat) : Int))
    (hy : ∀ i j, (y.get2 i j).toInt = (((c0 + j) * s1 : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    (base + y + x).get2 i j = b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) := by
  rw [Tensor.get2_ptrAddT, Tensor.get2_ptrAdd, hx, hy, hb, Buf.ptr_offset, Buf.ptr_offset]
  push_cast; ring_nf

/-- **The pointers of a 2-D tile**, `base + (rows[:, None] * a0 + cols[None, :] * a1)` with
`base = b.ptr e`: lane `(i, j)` points at element `e + (r0 + i) * s0 + (c0 + j) * s1` when the
tile's largest offset, `(r0 + BM - 1) * s0 + (c0 + BN - 1) * s1`, is below `2^(w-1)`. -/
theorem ptr_lane (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)} {a0 a1 : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * s1 < 2 ^ (w - 1)) (i : Fin BM) (j : Fin BN) :
    (base + (rows.col BN * a0 + cols.row BM * a1)).get2 i j =
      b.ptr ((e + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  ptr_lane_sum_of b hb (colMul_toInt hr ha0 (by omega)) (rowMul_toInt hc ha1 (by omega)) hfit i j

/-- **The pointers of a 2-D tile, one term at a time**,
`base + rows[:, None] * a0 + cols[None, :] * a1` (the row term in width `w`, the column term in
width `w'`): lane `(i, j)` points at element `e + (r0 + i) * s0 + (c0 + j) * s1` when each term's
largest value is below its bound. -/
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

/-! ### On the enabled lanes -/

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
  rw [Tensor.get2_ptrAdd, add_toInt_on hx hy hfit i j hi hj, hb, Buf.ptr_offset]
  push_cast; ring_nf

/-- `ptr_lane_sum_swap_of` on the enabled lanes, `base + (y + x)`. -/
theorem ptr_lane_sum_swap_of_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {x y : Tensor [BM, BN] (IntV true w)}
    (hx : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → (x.get2 i j).toInt = ((ρ i * s0 : Nat) : Int))
    (hy : ∀ (i : Fin BM) (j : Fin BN), κ j < N → (y.get2 i j).toInt = ((κ j * s1 : Nat) : Int))
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (w - 1)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + (y + x)).get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) := by
  intro i j hi hj
  have hs : ((y + x).get2 i j).toInt = ((κ j * s1 + ρ i * s0 : Nat) : Int) :=
    IntV.add_nat (hy i j hj) (hx i j hi)
      (by have := Nat.add_le_add (le_pred' hi s0) (le_pred' hj s1); omega)
  rw [Tensor.get2_ptrAdd, hs, hb, Buf.ptr_offset]; push_cast; ring_nf

/-- `ptr_lane_seq_of` on the enabled lanes, `base + x + y`: each term on rows (columns) in the
matrix, bounded by its own lemma. -/
theorem ptr_lane_seq_of_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {sg sg' : Bool} {x : Tensor [BM, BN] (IntV sg w)} {y : Tensor [BM, BN] (IntV sg' w')}
    (hx : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → (x.get2 i j).toInt = ((ρ i * s0 : Nat) : Int))
    (hy : ∀ (i : Fin BM) (j : Fin BN), κ j < N → (y.get2 i j).toInt = ((κ j * s1 : Nat) : Int)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + x + y).get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) := by
  intro i j hi hj
  rw [Tensor.get2_ptrAddT, Tensor.get2_ptrAdd, hx i j hi, hy i j hj, hb, Buf.ptr_offset,
    Buf.ptr_offset]
  push_cast; ring_nf

/-- `ptr_lane_seq_swap_of` on the enabled lanes, `base + y + x`. -/
theorem ptr_lane_seq_swap_of_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {sg sg' : Bool} {x : Tensor [BM, BN] (IntV sg w)} {y : Tensor [BM, BN] (IntV sg' w')}
    (hx : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → (x.get2 i j).toInt = ((ρ i * s0 : Nat) : Int))
    (hy : ∀ (i : Fin BM) (j : Fin BN), κ j < N → (y.get2 i j).toInt = ((κ j * s1 : Nat) : Int)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + y + x).get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) := by
  intro i j hi hj
  rw [Tensor.get2_ptrAddT, Tensor.get2_ptrAdd, hx i j hi, hy i j hj, hb, Buf.ptr_offset,
    Buf.ptr_offset]
  push_cast; ring_nf

/-- **The pointers of a 2-D tile on the enabled lanes**,
`base + (rows[:, None] * a0 + cols[None, :] * a1)` with `base = b.ptr e`: lane `(i, j)` with
`ρ i < M` and `κ j < N` points at element `e + ρ i * s0 + κ j * s1` when the matrix's largest
offset, `(M - 1) * s0 + (N - 1) * s1`, is below `2^(w-1)`. The `hp` of the access rules. -/
theorem ptr_lane_on (b : Buf d) {base : Ptr d} (hb : base = b.ptr (e : Int))
    {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w)} {a0 a1 : IntV true w}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (w - 1)) :
    ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      (base + (rows.col BN * a0 + cols.row BM * a1)).get2 i j =
        b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int) :=
  ptr_lane_sum_of_on b hb (colMul_toInt_on hr ha0 (by omega)) (rowMul_toInt_on hc ha1 (by omega))
    hfit

/-- **The pointers of a 2-D tile on the enabled lanes, one term at a time**,
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

end Tile2

end TVA
