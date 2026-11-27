import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Num
import Proof.Value
import Proof.Lanes
import Proof.Tail
import Proof.Walk
import Proof.Program
import Proof.Disj

/-!
# The proof of `mean_pooling_fwd_bf16`

Program `i` of the grid handles feature block `iD i`, head `hOf i` and output row `outRow i` (a
chunk of `L` positions of one sequence). It reads a `[BT, BD]` tile of `x` (and, on a
variable-length launch, four table entries) and writes `BD` columns of one row of `o`
(`Geo.R`, `Geo.W`). `correct_of_safe` composes the programs:

1. `Basic`, `Geo`, `Lanes`: the grid, the geometry and the `int64` arithmetic;
2. `Program.safe_program`: one program, walked line by line (`Walk.tail_safe`, `Tail.rest_safe`);
   each stored column is bounded by `Value.lane_value` (`Num.mean_err`: the tree sum, the
   division and the rounding against `tol`);
3. `Disj`: distinct programs write disjoint elements of `o`, nobody writes what anyone reads, and
   every element of the output has its program, which gives `Post`.
-/

namespace Claims.MeanPoolingFwd

open TVA

/-- The elements a program writes are output elements. -/
theorem wset_sub {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) :
    wset d i ⊆ {e | e < nRows d * d.H * d.D} := by
  rintro e ⟨l, hl, rfl⟩
  have := eO_last hd hi
  simp only [Set.mem_setOf_eq]
  omega

/-- **The programs' results give `Post`.** -/
theorem main_post {d : Inst} (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Q d i m ∧ ∀ x ∈ R d i ∪ W d i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d i) → fin x = init x) : Post d init fin := by
  have hsz := launch_size d
  constructor
  · intro g h j hg hh hj
    have hjb : j / d.BD < nD d := by
      have := (nD_mul d hd).1
      exact (Nat.div_lt_iff_lt_mul (bd_pos hd)).mpr (by omega)
    obtain ⟨hlt, hro, hho, hid⟩ := prog_props hd hg hh hjb
    let ii : Fin (launch d).size := ⟨progOf d g h (j / d.BD), by rw [hsz]; exact hlt⟩
    have hii : (ii : Nat) = progOf d g h (j / d.BD) := rfl
    obtain ⟨m, hQ, hag⟩ := hloc ii
    have hl : j % d.BD < d.BD := Nat.mod_lt _ (bd_pos hd)
    have hkap : iD d ii * d.BD + j % d.BD = j := by
      rw [hii, hid]; exact Nat.div_add_mod' j d.BD
    obtain ⟨c, hc, hspec⟩ := hQ ⟨j % d.BD, hl⟩ (by show iD d ii * d.BD + j % d.BD < d.D; rw [hkap]; exact hj)
    have hidx : (outRow d ii * d.H + hOf d ii) * d.D + (iD d ii * d.BD + j % d.BD) =
        (g * d.H + h) * d.D + j := by
      rw [hkap, hii, hro, hho]
    have hidxI : ((outRow d ii : Int) * d.H + hOf d ii) * d.D + (iD d ii * d.BD + (j % d.BD : Nat)) =
        ((g : Int) * d.H + h) * d.D + j := by
      have := congrArg (fun n : Nat => (n : Int)) hidx
      push_cast at this; exact this
    simp only [Fin.val_mk] at hc hspec
    rw [hidxI] at hc
    rw [hii, hro, hho] at hspec
    rw [show iD d ii * d.BD + j % d.BD = j from hkap] at hspec
    refine ⟨c, ?_, hspec⟩
    rw [← hc]
    refine Mem.read_congr fun x hx => hag x (Or.inr ?_)
    have hx' : x ∈ Mem.span DType.bf16 (d.o.ptr (((g * d.H + h) * d.D + j : Nat) : Int)).addr := by
      push_cast; exact hx
    refine d.o.span_sub_elems (S := wset d ii) ?_ hx'
    refine ⟨⟨j % d.BD, hl⟩, by show iD d ii * d.BD + j % d.BD < d.D; rw [hkap]; exact hj, ?_⟩
    unfold eO
    rw [hidx]
  · intro z hz
    exact hfr z fun ii hzi => hz (d.o.elems_mono
      (wset_sub hd (lt_of_lt_of_eq ii.isLt hsz)) hzi)

/-- **The claim**: each program is safe (`safe_program`), the footprints are disjoint (`Disj`), and
`main_post` turns the programs' results into `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hT : d.varlen = true → TablesOk d := fun hv => (hpre.2.2.2 hv).2.2.2.2
  have hsz := launch_size d
  refine correct_of_safe (fun i => R d i) (fun i => W d i) (fun i m => Q d i m)
    (launch_valid hd) ?_ ?_ ?_ ?_
  · intro i
    exact safe_program hd hpre (E := (launch d).env P) hP (lt_of_lt_of_eq i.isLt hsz)
      (launch_pid0 d i) (launch_pid1 d i)
  · intro i j hij
    have hi := lt_of_lt_of_eq i.isLt hsz
    have hj := lt_of_lt_of_eq j.isLt hsz
    have hij' : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (W_disjoint_R hd hT hi hj)
      (d.o.elems_disjoint (wset_disjoint hd hi hj hij'))
  · intro i
    have hi := lt_of_lt_of_eq i.isLt hsz
    exact (W_disjoint_R hd hT hi hi).symm
  · intro fin hloc hfr
    exact main_post hd hloc hfr

end Claims.MeanPoolingFwd
