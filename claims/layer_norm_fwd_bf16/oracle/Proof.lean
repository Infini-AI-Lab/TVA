import Proof.Body

/-!
# `layer_norm_fwd_bf16`: the proof

1. Program `i` handles rows `32 i … 32 i + 31`. Its footprints are `R` (the bytes of `x`, `residual`,
   `weight`, `bias` it reads) and `W` (the bytes of `res_out`, `mean`, `rstd`, `y` of those rows).
2. `safe_program` walks the kernel body once, line by line: the `float32` sum, `mean`, `rstd` and
   `y` of each row are computed from the lanes, and the row's results meet their tolerances
   (`Proof.Tol*`, `Proof.Y*`), with the device facts for `/`, `sqrt` and the depth of a `tl.sum`.
3. Different programs write different bytes, and none writes what any reads (`R_disjoint_W`,
   `W_disjoint_W`).
4. `correct_of_safe` combines the programs into `Post`.
-/
namespace Claims.LayerNormFwd

open TVA

/-- Program `i` is safe alone, with footprints `R`, `W` and post-condition `Q`. -/
theorem program_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d i) (W d i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) :=
  safe_program hd hpre _ (hP.divTotal _) (hP.mathTotal _ _) hP.sqrtF32 hP.divFull hP.sumDepth i

/-- The programs' results give `Post`. -/
theorem post_of_programs {d : Inst} (hd : d.ok) {init : Mem} (fin : Mem)
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Q d init i m ∧ ∀ x ∈ R d i ∪ W d i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d i) → fin x = init x) : Post d init fin := by
  refine ⟨fun t ht => ?_, fun z hz => hfr z fun i hzi => hz ?_⟩
  · have hp : t / 32 < (launch d).size := by
      rw [launch_size d]
      exact div32_lt hd ht
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨t / 32, hp⟩
    exact rowDone_congr ht rfl hagree (hQ t ht rfl)
  · exact W_subset_outBytes d i hzi

/-- Each program is safe alone, the footprints are disjoint, and together the programs' rows give
`Post`. -/
theorem correct_launch {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) : Correct ((launch d).env P) (model d) (launch d) init (Post d init) :=
  correct_of_safe (fun i => R d i) (fun i => W d i) (fun i m => Q d init i m) (launch_valid hd)
    (program_safe hP hd hpre)
    (fun i j hij => Disjoint.union_right (Disjoint.symm (R_disjoint_W hd j i))
      (W_disjoint_W hd (fun h => hij (Fin.ext h))))
    (fun i => R_disjoint_W hd i i)
    (fun fin hloc hfr => post_of_programs hd fin hloc hfr)

/-- **The claim.** -/
theorem main : claim.Holds := fun _ hP _ hd _ hpre => correct_launch hP hd hpre

end Claims.LayerNormFwd
