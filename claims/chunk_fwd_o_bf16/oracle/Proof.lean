import Claim
import TVA.Proof
import Proof.Post

/-!
# The proof of `chunk_fwd_o_bf16`

Program `(pid0, pid1)` of the grid computes one `64 × BV` tile of one chunk's output and writes
only that tile of `o`; programs read only the inputs. So the proof is the independent-programs
route (`correct_of_safe`):

1. `Body`, `Prelude`: the kernel's body is cut after its scalar lines (108–121) and after the
   pointer offsets (123–127); `Prelude` walks the scalars (program ids, the fixed-length
   `NT`, or the table loads of a variable-length batch) into the program's `Geo`.
2. `Loop`, `Dots`, `Lanes`, `Trip`, `LoopSafe`: the loop over `K` keeps, for the lanes the claim's
   `InRange` covers, `b_o` and `b_A` within `DotLoop.bound` of the partial dots.
3. `Gate`: the gates (`g`, `g_gamma`, or none) multiply them by `exp2`, with the error
   `cO`, `cA` (+ `gamErr`) of `Num*`.
4. `Epi`, `EpiLane`: the mask, the scores' rounding to `bfloat16`, the dot with `v`, the scale, the
   sum and the store: each stored element is finite and within `tol` of `ref` (`ep_final`).
5. `Foot`, `Post`: distinct programs write distinct elements of `o`, no program writes what
   another reads, and the programs' results give `Post`.
-/

namespace Claims.ChunkFwdO

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have ht : d.varlen = true → TablesOk d := fun h => (hpre.tables h).2.2.2.2
  refine correct_of_safe (fun _ => Rs d) (fun i => Wg d (geoOf d i))
    (fun i m => Qg d (geoOf d i) (geoOf d i).itg m) (launch_valid hd) (fun i => ?_)
    (fun i j hij => w_disjoint hd ht (lt_size i) (lt_size j) (fun h => hij (Fin.ext h)))
    (fun i => rw_disjoint hd ht (lt_size i)) ?_
  · exact safe_body hP hd hpre (lt_size i)
  · intro fin hloc hfr
    refine post_of_Q hd ht (init := init) (fin := fin) (fun i hi => ?_) (fun x hx => hfr x fun i => hx i (lt_size i))
    obtain ⟨m, hm, hag⟩ := hloc ⟨i, by show i < grid0 d * (d.B * d.HV) * 1; omega⟩
    exact ⟨m, hm, hag⟩

end Claims.ChunkFwdO
