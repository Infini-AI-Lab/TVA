import Claim
import TVA.Proof
import Proof.Val

/-!
# `streamk_matmul_f16`: the proof

1. **Ranges** (`Proof.Basic`, `Proof.Index`): program `p` runs iterations `[lo p, hi p)` of the
   `tilesM · tilesN · steps K` iterations; inside a tile its run is `[c0, c1)` of the `K` loop.
   A run covering the whole tile is stored, any other run is added to `c` with `tl.atomic_add`.
2. **Lanes and footprints** (`Proof.Lanes`, `Proof.Addrs`, `Proof.Foot`): every address of a tile
   is an element of its matrix; programs store disjoint elements (`Wset`), the split elements are
   shared cells (`Ccells`), and the inputs are never written.
3. **Tolerance** (`Proof.Tol`, `Proof.Count`): a chain of `n` dot steps stays within `fineOne` of
   the exact partial sum, and the runs of a split tile add up, in any order of the atomic
   additions, to within `tol` of the dot product (`tol_close`).
4. **One program alone** (`Proof.Acc`, `Chunk`, `Iter`, `Tail`, `Atomic`, `Step`): the program is
   `SafeA` with the invariant `Q`: the chunk loop (`chunkSafe`), then the store or atomic tail
   (`tail_safe`), then the outer loop (`prog_safe`).
5. **The claim** (`Proof.Val`, here): `correct_of_safeA` combines the programs, and
   `post_of_decomp` turns the replayed log into `Post`.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst} {P : Platform} {init : Mem}

/-- **A covered launch is correct.** The launch is valid; each program is `SafeA` alone (`prog_safe`) with the
shared cells `Ccells`; and any final memory that decomposes as the programs' invariants and the
replay of their atomic steps satisfies `Post` (`post_of_decomp`). -/
theorem correct_launch {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) :
    Correct ((launch d).env P) (model d) (launch d) init (Post d init) := by
  have hdisj : ∀ i j : Fin (launch d).size, i ≠ j →
      Disjoint (Wset d (i : Nat)) (inElems d ∪ Wset d (j : Nat)) := by
    intro i j hij
    have hij' : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    exact (inElems_disjoint_Wset hd (i : Nat)).symm.union_right (Wset_disjoint hd hij')
  have hAc : ∀ i : Fin (launch d).size,
      Disjoint (cellBytes (Ccells d)) (inElems d ∪ Wset d (i : Nat)) := by
    intro i
    rw [cellBytes_Ccells]
    exact (inElems_disjoint_Cbytes hd).symm.union_right (Cbytes_disjoint_Wset hd (i : Nat))
  refine correct_of_safeA (R := fun _ => inElems d) (W := fun i => Wset d (i : Nat))
    (C := Ccells d) (Q := fun i log m => Q d init (i : Nat) log m) (launch_valid hd)
    (fun i => prog_safe hP hd hpre i) (fun i => Q_permInv init (i : Nat)) hdisj
    (fun i => (inElems_disjoint_Wset hd (i : Nat))) hAc
    (fun c hc c' hc' hne => Ccells_distinct hd c hc c' hc' hne)
    (cellBytes_present hd hpre.2.2.1) ?_
  intro fin G hprogs hrep hframe
  rw [cellBytes_Ccells] at hrep hframe
  exact post_of_decomp hP hd hpre hprogs hrep hframe

/-- **The claim**: every covered launch of `streamk_matmul_kernel` ends normally with `Post`. -/
theorem main : claim.Holds :=
  fun _ hP _ hd _ hpre => correct_launch hP hd hpre

end Claims.StreamKMatmul
