import Proof.RowInit
import Proof.Facts

/-!
# The loops' invariants

For the rows of the program's block that lie inside the request, the carried state
`(acc, deno, e_max)` is, after `n` cached-part trips or `k` new-token trips, a `RowFl` of the positions
the row has attended so far (`seenP`, `seenE`), against the exact data of the row (`rowSc`, `rowVc`).
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- The exact scores of new token `i` of request `b`, head `h`, against each position. -/
def rowSc (d : Inst) (m : Mem) (b h i : Nat) : ℕ → ℝ := fun j => score d m b h i j

/-- The exact values: `rowVc e j` is component `e` of the value at position `j`. -/
def rowVc (d : Inst) (m : Mem) (b h : Nat) : ℕ → ℕ → ℝ :=
  fun e j => ((keyV d m b h j e : ℚ) : ℝ)

/-- The positions of the row's keys the cached-part loop has seen after `n` trips. -/
def seenP (d : Inst) (m : Mem) (b i n : Nat) : Finset ℕ :=
  (keys d m b i).filter (· < min (BN * n) (preOf d m b))

/-- The positions of the row's keys below `pre_b + x`: seen once the new-token loop has passed
`x` new tokens. -/
def seenE (d : Inst) (m : Mem) (b i x : Nat) : Finset ℕ :=
  (keys d m b i).filter (· < preOf d m b + x)

/-- The row's score bound `X`, the computed scores' error and bound. -/
def rowX (d : Inst) (init : Mem) (i r : Nat) : ℝ :=
  Xrow d init (bOf d i) (hOf d i) (mOf d i * d.BM + r)

/-- **The cached-part loop's invariant** after `n` trips: memory untouched, and each valid row's state
is a `RowFl` of the keys it has seen. -/
def LoopInvP (d : Inst) (init : Mem) (i : Nat) (n : Nat) (st : LoopSt d) (m : Mem) : Prop :=
  m = init ∧ ∀ r : Fin d.BM, validRow d init i r →
    RowFl (rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
      (rowVc d init (bOf d i) (hOf d i)) d.D d.BD (dlt d.BD (rowX d init i r))
      (rowX d init i r + dlt d.BD (rowX d init i r)) n (g2 + n * g1p)
      (seenP d init (bOf d i) (mOf d i * d.BM + r) n) (st.2.2.get1 r) (st.2.1.get1 r)
      (fun e => st.1.get2 r e)

/-- **The new-token loop's invariant** after `k` trips that started at the new token `es`: memory
untouched, and each valid row's state is a `RowFl` of the keys below `pre + es + 64 k`, after
`Tp + k` trips. -/
def LoopInvE (d : Inst) (init : Mem) (i : Nat) (es k : Nat) (st : LoopSt d) (m : Mem) : Prop :=
  m = init ∧ ∀ r : Fin d.BM, validRow d init i r →
    RowFl (rowSc d init (bOf d i) (hOf d i) (mOf d i * d.BM + r))
      (rowVc d init (bOf d i) (hOf d i)) d.D d.BD (dlt d.BD (rowX d init i r))
      (rowX d init i r + dlt d.BD (rowX d init i r)) (tripsPre d init (bOf d i) + k)
      (g2 + tripsPre d init (bOf d i) * g1p + k * g1e)
      (seenE d init (bOf d i) (mOf d i * d.BM + r) (es + BN * k)) (st.2.2.get1 r) (st.2.1.get1 r)
      (fun e => st.1.get2 r e)

end

end Claims.ExtendAttn
