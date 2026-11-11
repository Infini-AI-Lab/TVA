import Claim
import TVA.Proof
import Proof.Compose

/-!
# The proof of `gdn_gating_bf16`

Program `i` of the grid `(batch, 1, cdiv(H, 8))` handles token `i % batch` and the block of 8 heads
`i / batch`; its lanes are heads `8 (i / batch) + l`, enabled below `H`. Programs touch disjoint
tiles of `g` and `beta_output` and only read the inputs, so `correct_of_safe` composes them:

1. `Proof/Basic.lean`, `Idx.lean`: the launch is valid; the lanes of one program, their `int32`
   offsets (no wrap) and pointers;
2. `Proof/Spec.lean`: the footprints `Rset`, `Wset`, the local postcondition `Q` of one program,
   disjointness of distinct programs' footprints, and `Q` of every program gives `Post`;
3. the numbers of one lane:
   * `Rnd`, `NanFree`, `ExpRoom`, `Bridge`, `FloatStep`: one `float32` rounding, no NaN, the
     constants, and one step of each operation with the device facts (`exp`, `log`, `/`);
   * `Softplus`, `SoftNeg`, `SoftFalse`: the softplus branch `log(1 + exp x)` for `xf ≤ 20`, very
     negative `x`, and the branch `x` for `xf > 20`;
   * `GLow`, `GMain`: the arithmetic of `g = -E · S` within `gTol`, for `A ≤ -89` and above;
     `BetaChain`: the sigmoid chain within `bTol`;
   * `LaneAux`, `LaneG`, `LaneBeta`: `lane_g` and `lane_beta`, the two lanes;
4. `Proof/Safe.lean`: one program, walked line by line, is safe and leaves `Q` (`safe_body`);
   `Program.lean`, `Compose.lean`: the platform's device facts, then `correct` for the launch;
5. `main` below.
-/

namespace Claims.GdnGating

open TVA

/-- **The claim**: every covered launch ends normally with `Post`. -/
theorem main : claim.Holds := fun _ hP _ hd _ hpre => correct_claim hP hd hpre

end Claims.GdnGating
