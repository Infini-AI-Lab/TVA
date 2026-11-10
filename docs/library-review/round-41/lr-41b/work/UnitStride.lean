/- lr-41b, round 41: unit-stride terms (`cols[None, :]`, `rows[:, None]` with no stride) for rows
and columns of any form. U1 fails: `row_toInt` takes only `c0 + j` columns. U2 is the hand proof. -/
import TVA
namespace TVA
open Tile2
-- U1: the library's unit-stride column term, for `% N` columns
example {BM BN N c0 : Nat} {cols : Tensor [BN] I32}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = (((c0 + j) % N : Nat) : Int)) :
    ∀ (i : Fin BM) (j : Fin BN), (c0 + j) % N < N →
      ((cols.row BM).get2 i j).toInt = (((c0 + j) % N * 1 : Nat) : Int) :=
  fun i j _ => row_toInt hc i j
-- U2: by hand
example {BM BN N c0 : Nat} {cols : Tensor [BN] I32}
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = (((c0 + j) % N : Nat) : Int)) :
    ∀ (i : Fin BM) (j : Fin BN), (c0 + j) % N < N →
      ((cols.row BM).get2 i j).toInt = (((c0 + j) % N * 1 : Nat) : Int) :=
  fun i j _ => by rw [Nat.mul_one]; exact hc j
end TVA
