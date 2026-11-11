import Claim
import TVA.Proof
import Proof.SafeProg
import Proof.Post

/-!
# The proof of `chunk_bwd_dv_local_bf16`

Program `(r, bh)` of the grid is one chunk of one sequence (`r`, read from `chunk_indices` on a
variable-length launch) for one value head (`bh`). It reads `q`, `k`, `do`, `g` and the two tables
and writes the elements `(token, hv, e)` of `dv` of its chunk's tokens, so the programs are
independent and `correct_of_safe` composes them:

1. `Geo`, `GeoBound`, `GeoCover`, `GeoDisj`, `Basic`, `Post`: the program's geometry (`nOf`,
   `chkOf`, `pos`, `en`), its footprints `R` and `W` and post-condition `Q`; the launch is valid,
   distinct programs write disjoint elements, every token is some program's lane (`main_post`).
2. `Prologue`, `Tail`, `Setup`, `SafeTail`: one program is safe. The prologue's scalars
   (`safe_prologue`), the head's base pointers and the token offsets (`Setup`), then the four
   pieces of the tail.
3. `Spec`, `Vals`, `ValsLane`: what each piece leaves (`GOk`, `KFinal`, `AOk`), the exact sums
   (`kqP`, `kqMagP`) and the covered values (`ValsOk`) read at the program's lanes.
4. `KLoop`, `KStep`, `KDot`, `KTrip`, `LoadLanes`: the gate block and the `K` loop. One trip loads
   two tiles, takes a `bfloat16` dot (`NumDot`) and a fused multiply-add (`NumKStep`); the
   accumulator keeps a relative error that does not grow with the number of trips.
5. `Epi`, `NumGate`, `NumEntry`, `NumCvt`: the gate `exp2` (including an overflowing
   difference), the product, the causal mask and the rounding of `A` to `bfloat16`, entry by entry.
6. `VLoop`, `VTrip`, `VNum`, `NumFinal`: the `V` loop. One trip loads a tile of `do`, takes the
   second dot, rounds and stores; the real-number core (`vlane`, `final_err`) shows the stored
   value is within `tol` of `ref`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- **The claim**: each program is safe (`safe_program`), footprints are disjoint (`GeoDisj`),
and `main_post` turns the programs' results into `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hT := tables_of_pre hpre
  have hs : ∀ i : Fin (launch d).size, (i : Nat) % nChunks d < nChunks d ∧
      (i : Nat) / nChunks d % nRows d < nRows d := fun i => by
    have h : (i : Nat) < nChunks d * nRows d := lt_of_lt_of_eq i.isLt (launch_size d)
    exact ⟨Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 => by simp [h0] at h),
      Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 => by simp [h0] at h)⟩
  refine correct_of_safe (fun i => R d (i % nChunks d) (i / nChunks d % nRows d))
    (fun i => W d (i % nChunks d) (i / nChunks d % nRows d))
    (fun i m => Q d init (i % nChunks d) (i / nChunks d % nRows d) m) (launch_valid hd)
    (fun i => ?_) ?_ ?_ ?_
  · exact safe_program hP hd hpre (hs i).1 (hs i).2 (launch_pid0 d i) (launch_pid1 d i)
  · intro i j hij
    refine W_disjoint_all hd hT (hs i).1 (hs i).2 (hs j).1 (hs j).2 ?_
    by_contra hcon
    push Not at hcon
    apply hij
    apply Fin.ext
    have hi := Nat.div_add_mod (i : Nat) (nChunks d)
    have hj := Nat.div_add_mod (j : Nat) (nChunks d)
    have hdi : (i : Nat) / nChunks d % nRows d = (i : Nat) / nChunks d := by
      have h : (i : Nat) < nChunks d * nRows d := lt_of_lt_of_eq i.isLt (launch_size d)
      exact Nat.mod_eq_of_lt ((Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero fun h0 => by simp [h0] at h)).2 (by rw [Nat.mul_comm]; exact h))
    have hdj : (j : Nat) / nChunks d % nRows d = (j : Nat) / nChunks d := by
      have h : (j : Nat) < nChunks d * nRows d := lt_of_lt_of_eq j.isLt (launch_size d)
      exact Nat.mod_eq_of_lt ((Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero fun h0 => by simp [h0] at h)).2 (by rw [Nat.mul_comm]; exact h))
    rw [← hi, ← hj, hcon.1, ← hdi, ← hdj, hcon.2]
  · intro i
    exact (W_disjoint_R hd hT (hs i).1 (hs i).2).symm
  · intro fin hloc hfr
    exact main_post hd hT hloc hfr

end Claims.ChunkBwdDvLocal
