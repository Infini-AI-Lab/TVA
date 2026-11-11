import Claim
import TVA.Proof
import Proof.Main

/-!
# The proof of `moe_router_bf16`

Program `t` of the grid is token `t`. It reads its row of `x` and all of `w`, and writes
elements `2t` and `2t+1` of `topk_ids` and of `topk_weights`, so the footprints of distinct
programs are disjoint (`Proof.Setup`) and `correct_of_safe` composes the programs:

1. `Proof.Addrs`, `Proof.Cast`, `Proof.Const`: the addresses the program forms, the `bfloat16`
   to `float32` conversion, and the values of the literals;
2. `Proof.Logit`: each logit `x · w_e` is within `tolL` of the exact sum (the fused tree sum);
3. `Proof.Tanh`, `Proof.Cap`, `Proof.CapReal`, `Proof.CapRatio`, `Proof.CapBound`: the soft cap
   `30·tanh(l/30)` computed with `exp` stays within `capEps` of the exact one (real lemmas, then
   the rounding of each operation);
4. `Proof.Argmax`: the two `tl.argmax` calls return the first largest `C` and the first largest
   among the others;
5. `Proof.Topsum`, `Proof.WeightDen`, `Proof.Weight`, `Proof.Final`: the softmax weights within
   their tolerance of the exact ones;
6. `Proof.Spine.safe_program`: one program, walked line by line, ends in the per-token
   postcondition `Q`;
7. `Proof.Main`, `main`: the launch is valid, the footprints are disjoint, and the `Q`s give `Post`.
   `Proof.Setup` has the footprints and that composition step.
-/

namespace Claims.MoeRouter

open TVA

/-- **The claim**, from `Proof.Main`. -/
theorem main : claim.Holds := main_holds

end Claims.MoeRouter
