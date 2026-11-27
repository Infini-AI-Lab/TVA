import Proof.Basic
import Proof.Search
import Proof.Block

/-!
# `ep_src2dst_i32`: `compute_src2dst_triton_kernel`

Program `p` reads all `n` sorted ids (its binary search) and its block of `reorder_ids`, and
writes the elements `reorder_ids[j]` of its block of sorted positions `j`. The proof:

1. `Proof.Search`: lines 42–51 read only sorted ids and find the number of leading negative ids
   (`safe_searchIte`, the loop by `safe_while` with the variant `right - left`);
2. `Proof.Block`: lines 53–59 (`safe_rest`) and the whole program (`safe_program`);
3. `main`: by `correct_of_safe`. The written elements of two programs differ because the entries of
   `reorder_ids` are distinct (`Pre`), and the output shares no byte with an input (`Inst.ok`).
-/

namespace Claims.EpSrc2Dst

open TVA Block

/-- The written elements of distinct programs are distinct. -/
theorem outE_disjoint {d : Inst} {init : Mem} (hpre : Pre d init) {p q : Nat} (hpq : p ≠ q) :
    Disjoint (outE d init p) (outE d init q) := by
  rw [Set.disjoint_left]
  rintro k ⟨j, hj, hjk⟩ ⟨j', hj', hjk'⟩
  have e := hpre.2.2.2.2.2 j j' hj.2.2 hj'.2.2 (by rw [hjk, hjk'])
  subst e
  exact hpq (tile_unique hj hj')

/-- The written elements lie in `src2dst`. -/
theorem outE_lt {d : Inst} {init : Mem} (hpre : Pre d init) (p : Nat) :
    ∀ k ∈ outE d init p, k < d.src2dst.len := by
  rintro k ⟨j, hj, hjk⟩
  have := (hpre.2.2.2.2.1 j hj.2.2).2
  omega

/-- No program writes what a program reads. -/
theorem W_disj_R {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (p q : Nat) :
    Disjoint (W d init p) (R d q) := by
  refine Disjoint.union_right ?_ ?_
  · exact d.src2dst.elems_disjoint_of d.sorted hd.out_sorted (outE_lt hpre p)
      (fun j hj => by have := hd.sorted_len; simp only [Set.mem_ofPred_eq] at hj; omega)
  · exact d.src2dst.elems_disjoint_of d.reorder hd.out_reorder (outE_lt hpre p)
      (fun j hj => by have := hd.reorder_len; have := hj.2.2; omega)

/-- **The claim.** Each program is safe (`safe_program`), the footprints are disjoint, and the
programs' results give `Post`. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => R d (i : Nat)) (fun i => W d init (i : Nat))
    (fun i m => Q d init (i : Nat) m) (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · exact safe_program hd hpre (lt_of_lt_of_eq i.isLt (launch_size d)) _ (launch_pid d i)
  · intro i j hij
    refine Disjoint.union_right (W_disj_R hd hpre _ _) (d.src2dst.elems_disjoint (outE_disjoint hpre ?_))
    exact fun h => hij (Fin.ext h)
  · intro i
    exact (W_disj_R hd hpre _ _).symm
  · intro fin hloc hfr
    refine ⟨fun i hi => ?_, fun z hz => hfr z fun q hzq => hz ?_⟩
    · have hti : Tile (i / 512) 512 d.n i := tile_div (by norm_num) hi
      have hq : i / 512 < (launch d).size :=
        lt_of_lt_of_eq (show i / 512 < G d by unfold G; omega) (launch_size d).symm
      obtain ⟨m, hQ, hag⟩ := hloc ⟨i / 512, hq⟩
      have h0 := (hpre.2.2.2.2.1 i hi).1
      have hcast : (((rid d init i).toNat : Nat) : Int) = rid d init i := Int.toNat_of_nonneg h0
      have hmem : (rid d init i).toNat ∈ outE d init (i / 512) := ⟨i, hti, hcast.symm⟩
      have hspan := d.src2dst.span_sub_elems hmem
      rw [hcast] at hspan
      rw [Mem.read_congr fun x hx => hag x (Or.inr (hspan hx))]
      exact hQ i hti
    · obtain ⟨k, ⟨j, hj, hjk⟩, hx⟩ := hzq
      exact ⟨k, ⟨j, hj.2.2, hjk⟩, hx⟩

end Claims.EpSrc2Dst
