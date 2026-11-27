import Claim
import TVA.Proof
import Proof.Defs
import Proof.Walk

/-!
# `layer_norm_gated_fwd_bf16`: the proof

The model (`layer_norm_gated_fwd_kernel`), the conditions (`Pre`) and the specification (`Post`) are
in `Claim.lean`. The route is `correct_of_safe`: one program of the grid at a time.

1. Footprints (`Proof.Lanes`, `Proof.Defs`): program `p` owns the rows `p·BT + [0, BT)` below `T`.
   It reads the `[BT, D]` tiles of `x` and `g` and the first `D` elements of `weight` (`R`),
   writes its tile of `y` and its `BT` elements of `rstd` (`W`), and leaves the `rstd` and `y`
   values `Post` asks for on its rows (`Q`). Tiles of distinct programs are disjoint (`hdisj`,
   `hRW`), and the programs' results give `Post` (`hpost`).
2. Numeric bounds:
   - the row sum (`Proof.NumSum`, `Proof.RowSum`): `sum_err_bd`;
   - `/ D`, `+ eps`, `tl.sqrt` and `1 / ·` (`Proof.NumRstd`, `Proof.RstdAux`, `Proof.Consts`,
     `Proof.RstdRow`): `rstd_row`, within `2^-12` relative;
   - the gate `tl.sigmoid` (`Proof.SigReal`, `Proof.SigFloat`, `Proof.Gate`): `gate_err`, within
     `2^-15·σ + 2^-128`, with `tl.exp`'s overflow;
   - the products and the `bfloat16` rounding (`Proof.Chain`, `Proof.Ops`, `Proof.Elem`):
     `elem_swish`, `elem_sigmoid`.
3. The leaf (`Proof.Leaf`, `Proof.LeafRow`, `Proof.LeafQ`): from the values the walk computed,
   the final memory satisfies `Q` (`qn`).
4. The walk (`Proof.Walk`, `safe_program`): each program is `Safe`, one `tl.*` operation at a
   time, with the index arithmetic of the `int64` rows and `int32` columns.
5. `main`: the launch is valid (`launch_valid`), the footprints are disjoint, and
   `correct_of_safe` composes the programs.
-/

namespace Claims.LayerNormGatedFwd

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (R d init) (W d init) (Q d init) (launch_valid hd) (fun i => ?_)
    (fun i j hij => hdisj hd init i j hij) (fun i => hRW hd init i)
    (fun fin hloc hfr => hpost hd hloc hfr)
  exact safe_program hd hpre P (hP.mathTotal _ _) (hP.mathTotal _ _) hP.sqrtF32 (hP.divTotal _)
    hP.divFull hP.expF32 hP.expF32Overflow hP.divF32InfDivisor i

end Claims.LayerNormGatedFwd
