import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.GeoBound
import Proof.GeoCover
import Proof.GeoDisj
import Proof.GeoSum
import Proof.LaneArith
import Proof.Post
import Proof.Safe
import Proof.Util

/-!
# The proof of `chunk_cumsum_f32`

Program `i` of the grid is the (chunk, batch row, head) triple `(chkOf, nOf, headOf)`; its lane `l`
reads and writes element `elem i l` of `s` and `o`, when `en i l` (the lane is inside the
sequence). Its footprints are `R d i` (those bytes of `s`, plus its entries of the two tables) and
`W d i` (those bytes of `o`). `correct_of_safe` composes the programs:

1. `Geo`, `GeoBound`, `LaneArith`: the geometry (`elem`, `en`, `pos`, the sequence bounds from
   `TablesOk`) and the `int64` arithmetic of the body's scalars;
2. `Safe.safe_program`: one program, walked line by line with the rules of `TVA.Proof.Rules`; its
   scan is bounded by `Tail.tail_safe` (`GeoSum`, `NumScan`, `NumArith`: the float error of the
   `tl.cumsum` tree against `tol`);
3. `GeoDisj`: distinct programs write disjoint bytes of `o`, and nobody reads what it writes;
4. `GeoCover`, `Post`: every position of every sequence and every head is some program's lane, so
   the programs' results give `Post`.
-/

namespace Claims.ChunkCumsum

open TVA

/-- Tables of a variable-length launch from the precondition. -/
theorem tables_of_pre {d : Inst} {init : Mem} (hpre : Pre d init) :
    d.varlen = true → TablesOk d :=
  fun hv => (hpre.2.2.2 hv).2.2.2.2

/-- **The claim**: each program is safe (`safe_program`), footprints are disjoint (`GeoDisj`), and
`main_post` turns the programs' results into `Post`. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => R d i) (fun i => W d i) (fun i m => Q d i m)
    (launch_valid hd) ?_ ?_ ?_ ?_
  · intro i
    exact safe_program hd hpre _ (lt_of_lt_of_eq i.isLt (launch_size d)) (launch_pid d i)
  · intro i j hij
    have hi : (i : Nat) < gridSize d := lt_of_lt_of_eq i.isLt (launch_size d)
    have hj : (j : Nat) < gridSize d := lt_of_lt_of_eq j.isLt (launch_size d)
    have hij' : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    exact hdisj_all hd (tables_of_pre hpre) hi hj hij'
  · intro i
    have hi : (i : Nat) < gridSize d := lt_of_lt_of_eq i.isLt (launch_size d)
    exact R_disjoint_W hd (tables_of_pre hpre) hi
  · intro fin hloc hfr
    exact main_post hd (tables_of_pre hpre) hloc hfr

end Claims.ChunkCumsum
