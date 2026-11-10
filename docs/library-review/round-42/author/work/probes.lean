/- Round 42, author: probes of L192 (the unit-stride terms for rows and columns of any form) and L193
(the `safe_convert_iff` example as the docstring now writes it). Run from lean/ with
`lake env lean ../docs/library-review/round-42/author/work/probes.lean`; output in `probes.out`. -/
import TVA
namespace TVA
open Tile2

-- L192 P1: lr-41b's U1, `% N` columns with no stride, now through `row_toInt` itself
example {BM BN N c0 : Nat} {cols : Tensor [BN] I32}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = (((c0 + j) % N : Nat) : Int)) :
    ∀ (i : Fin BM) (j : Fin BN), (c0 + j) % N < N →
      ((cols.row BM).get2 i j).toInt = (((c0 + j) % N * 1 : Nat) : Int) :=
  fun i j _ => row_toInt hc i j

-- L192 P2: `% M` rows with no stride, through `col_toInt`
example {BM BN M r0 : Nat} {rows : Tensor [BM] I64}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = (((r0 + i) % M : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN).get2 i j).toInt = (((r0 + i) % M * 1 : Nat) : Int) :=
  col_toInt hr i j

-- L192 P3: an old `r0 + i` caller gets back the `r0 + i` form unchanged
example {BM BN r0 c0 : Nat} {rows : Tensor [BM] I32} {cols : Tensor [BN] I32}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (i : Fin BM) (j : Fin BN) :
    ((rows.col BN).get2 i j).toInt = (((r0 + i) * 1 : Nat) : Int) ∧
      ((cols.row BM).get2 i j).toInt = (((c0 + j) * 1 : Nat) : Int) :=
  ⟨col_toInt hr i j, row_toInt hc i j⟩

-- L192 P4: `x_ptr + rows[:, None] * s + (offs % N)[None, :]` on an enabled lane, by `add_toInt_on`
-- with `rowMul_toInt_on`'s row term and the unit-stride column term `row_toInt` of `% N` columns
example {BM BN M N r0 c0 s : Nat} {rows : Tensor [BM] I32} {cols : Tensor [BN] I32} {a : I32}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = (((c0 + j) % N : Nat) : Int)) (ha : a.toInt = s)
    (hfit : (M - 1) * s + (N - 1) * 1 < 2 ^ (32 - 1)) (i : Fin BM) (j : Fin BN)
    (hi : r0 + i < M) (hj : (c0 + j) % N < N) :
    ((rows.col BN * a + cols.row BM).get2 i j).toInt = (((r0 + i) * s + (c0 + j) % N * 1 : Nat) : Int) :=
  add_toInt_on (ρ := fun i : Fin BM => r0 + i) (κ := fun j : Fin BN => (c0 + j) % N)
    (colMul_toInt_on hr ha (by omega)) (fun i j _ => row_toInt hc i j) hfit i j hi hj

-- L193 P5: the docstring's form, at any shape
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
example {s : Shape} (x : Tensor s I32) (k : Tensor s I64 → Prog α) :
    Safe E R W Q m (tl.convert .i32 .i64 x >>= k) ↔ Safe E R W Q m (k x.castInt) :=
  safe_convert_iff fun _ _ => ConvRel.iff_cast_i32_i64

-- L193 P6: at shape [n] (lr-41b's V1 shape)
example {n : Nat} (x : Tensor [n] I32) (k : Tensor [n] I64 → Prog α) :
    Safe E R W Q m (tl.convert .i32 .i64 x >>= k) ↔ Safe E R W Q m (k x.castInt) :=
  safe_convert_iff fun _ _ => ConvRel.iff_cast_i32_i64

end TVA

-- the axioms of the two changed lemmas
#print axioms TVA.Tile2.col_toInt
#print axioms TVA.Tile2.row_toInt
