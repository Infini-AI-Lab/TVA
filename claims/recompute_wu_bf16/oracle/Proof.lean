import Claim
import TVA.Proof
import Proof.Valid
import Proof.Post

/-!
# The proof of `recompute_wu_bf16`

Program `(it, b·HV + h)` of the grid owns the tile of `u` and of `w` of the chunk `it` of one
sequence and the value head `h`. It only reads the inputs, so the proof is the
independent-programs route (`correct_of_safe`):

1. `Num`, `Lane`: the rounding chains. `kb = k·beta·2^g` rounded three times is within
   `(2^-7 + 2^-10)·|k beta|·2^g + 2^-125 (|k beta| + 1)` of the exact value; `vb = v·beta` is
   finite and small.
2. `Cell`: an element of the `bfloat16` dot (`Device.DotBF16`) rounded to `bfloat16` meets
   `tolU` (`u_cell`) and `tolW` (`w_cell`) against the exact sum of the operands' products.
3. `Defs`, `Geo`, `Spec`, `Fit`: the programs (chunk, sequence, head), their footprints, the
   lanes each tile holds, and the `int64` arithmetic of the offsets.
4. `Prog`: the kernel's body split into `preProg` (lines 91–98) and `tailProg`, with the
   dtype-dependent lines as parameters; `Steps`, `Lanes`, `Loads`, `VecLoads`, `TileLoads`: what
   each such line and each load gives.
5. `UBound`, `UTrip`, `WTrip`, `Stable`, `Loops`: one trip of each loop meets the cell bounds, the
   loops' invariants (inputs unchanged, outputs present, cells done) and the loops themselves;
   `Tail`, `Pre`, `Body`, `Safe`: the whole body is `Safe`.
6. `Foot`, `Post`, `Valid`: different programs write different tokens; the programs' results
   give `Post`; the launch is valid.
-/

namespace Claims.RecomputeWU

open TVA

/-- **The claim.** Each program is safe on its own tile (`safe_program`); different programs write
disjoint bytes and none writes an input; the programs' results give `Post` (`post_of_Q`). -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => Rall d) (fun i => Wf d init i) (fun i m => Qf d init i m)
    (launch_valid hd) (fun i => ?_) (fun i j hij => ?_) (fun i => ?_) (fun fin hloc hfr => ?_)
  · exact safe_program hP hd hpre i
  · exact Set.disjoint_union_right.mpr ⟨Wf_disjoint_R hd hpre i, Wf_disjoint hd hpre hij⟩
  · exact (Wf_disjoint_R hd hpre i).symm
  · exact post_of_Q hd hpre hloc hfr

end Claims.RecomputeWU
