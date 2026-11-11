import Claim
import TVA.Proof
import Proof.Footprints

/-!
# The proof of `layer_norm_fwd_f16`

One program of the grid `(M,)` normalises one row; the proof shows each program `Safe` by itself
and composes them with `correct_of_safe`.

1. `Proof.Row`, `Proof.Mean`, `Proof.VarChain`, `Proof.SumSq`, `Proof.SStage`, `Proof.RstdStage`,
   `Proof.ZStage`: the numerics, on plain reals. A `RowP` is one row's values; the mean, the
   variance, the input of `tl.sqrt`, `rstd`, the standardised value and the output are each bounded
   in turn (the rounding errors of the sums, the divisions, the fusable subtractions and `var + eps`
   against the claim's tolerances).
2. `Proof.Glue`, `Proof.OutLane`: the float operations of the model (`AddRel`, `NaNEq`, the device
   facts for `/` and `tl.sqrt`) as bounds on real values; `out_lane` is one output element.
3. `Proof.Lanes`, `Proof.LoadX`, `Proof.Walk1`, `Proof.Walk2`, `Proof.Walk3`, `Proof.Stores`,
   `Proof.Program`: program `t` walked line by line (`safe_program`): the mean, `rstd`, the two
   stores and the output loop; it reads inside `R d t`, writes inside `W d t`, and ends with
   `RowDone`.
4. `Proof.Footprints` and `main`: the programs' footprints are disjoint, and every row's `RowDone`
   gives `Post`.
-/

namespace Claims.LayerNormFwdF16

open TVA

/-- `RowDone` depends only on the bytes the program writes. -/
theorem rowDone_congr {d : Inst} {init m fin : Mem} {t : Nat} (ht : t < d.M)
    (hfin : ∀ z ∈ W d t, fin z = m z) (h : RowDone d init m t) : RowDone d init fin t := by
  obtain ⟨⟨c1, hc1, hf1, hb1⟩, ⟨c2, hc2, hf2, hb2⟩, hy⟩ := h
  refine ⟨⟨c1, ?_, hf1, hb1⟩, ⟨c2, ?_, hf2, hb2⟩, fun j hj => ?_⟩
  · rw [← hc1]
    exact Mem.read_congr fun z hz => hfin z (Or.inl (Or.inr (mean_span_sub ht hz)))
  · rw [← hc2]
    exact Mem.read_congr fun z hz => hfin z (Or.inr (rstd_span_sub hz))
  · obtain ⟨c, hc, hcf, hcb⟩ := hy j hj
    refine ⟨c, ?_, hcf, hcb⟩
    rw [← hc]
    exact Mem.read_congr fun z hz =>
      hfin z (Or.inl (Or.inl (d.y.span_sub_elems (S := rowEls d t) (j := idx d t j) ⟨j, hj, rfl⟩ hz)))

/-- **The claim.** Each program is safe on its own row (`safe_program`); the footprints of distinct
programs are disjoint, and the rows' results give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hsize : (launch d).size = d.M := by simp [launch, Launch.grid1, Launch.size]
  have hM31 : d.M < 2 ^ 31 := by have := hd.m_ctx; omega
  refine correct_of_safe (fun i => R d i) (fun i => W d i) (fun i m => RowDone d init m i) ?_
    (fun i => ?_) ?_ ?_ ?_
  · refine Launch.valid_grid1 hM31 ?_
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    exact ⟨by simpa using hd.block_pow2, by have := hd.block_le; simpa using (by omega : d.BLOCK ≤ 2 ^ 20)⟩
  · have hi : (i : Nat) < d.M := hsize ▸ i.isLt
    refine safe_program hd hpre hi _ hP _ ?_
    simp only [Launch.ctx, launch, Launch.grid1]
    exact Nat.mod_eq_of_lt hi
  · intro i j hij
    have hi : (i : Nat) < d.M := hsize ▸ i.isLt
    have hj : (j : Nat) < d.M := hsize ▸ j.isLt
    exact Disjoint.union_right (W_disjoint_R hd hi hj)
      (W_disjoint_W hd hi hj fun h => hij (Fin.ext h))
  · intro i
    exact R_disjoint_W hd (hsize ▸ i.isLt)
  · intro fin hloc hfr
    refine ⟨fun t ht => ?_, fun z hz => hfr z fun i hzi => hz (W_sub_out (hsize ▸ i.isLt) hzi)⟩
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨t, hsize ▸ ht⟩
    exact rowDone_congr ht (fun z hz => hagree z (Or.inr hz)) hQ

end Claims.LayerNormFwdF16
