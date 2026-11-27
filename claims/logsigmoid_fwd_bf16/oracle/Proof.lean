import Claim
import TVA.Proof
import Proof.Walk

/-!
# The proof of `logsigmoid_fwd_bf16`

Program `p` of the grid `cdiv(T, B)` owns the elements `blk d p` (its block of `B` elements of the
flattened tensor, cut at `T`). It reads the elements `xIdx d k` of `x` for `k` in its block and
writes its block of `y`, so the writes of distinct programs are disjoint, and `correct_of_safe`
composes the programs:

1. `Proof/Real.lean`: the arithmetic of one element in the reals (the error of each step and the
   relaxation to `yTol`);
2. `Proof/Lane.lean`: one element from the operations' results (`lane`), with the device facts for
   `tl.exp`, `tl.log` and `/`;
3. `Proof/Walk.lean`: one program, walked line by line (`safe_program`);
4. `main`: the launch is valid, the footprints are disjoint, and the blocks cover `[0, T)`.
-/

namespace Claims.LogsigmoidFwd

open TVA

/-- The blocks of distinct programs are disjoint. -/
theorem blk_disjoint {d : Inst} {p q : Nat} (hpq : p ≠ q) : Disjoint (blk d p) (blk d q) := by
  rw [Set.disjoint_left]
  rintro j ⟨hp1, hp2, -⟩ ⟨hq1, hq2, -⟩
  have e1 : j / d.B = p := Nat.div_eq_of_lt_le hp1 (by rw [Nat.succ_mul]; exact hp2)
  have e2 : j / d.B = q := Nat.div_eq_of_lt_le hq1 (by rw [Nat.succ_mul]; exact hq2)
  exact hpq (e1.symm.trans e2)

/-- Element `k < T` is in the block of program `k / B`, and that program is in the grid. -/
theorem mem_blk {d : Inst} (hd : d.ok) {k : Nat} (hk : k < d.T) :
    k ∈ blk d (k / d.B) ∧ k / d.B < G d := by
  have hB := B_pos hd
  have h1 := Nat.div_mul_le_self k d.B
  have h2 := Nat.lt_div_mul_add (a := k) hB
  refine ⟨⟨h1, h2, hk⟩, ?_⟩
  apply Nat.lt_of_succ_le
  rw [G, Nat.le_div_iff_mul_le hB, Nat.succ_mul]
  omega

/-- **The claim.** Each program is safe on its own block (`safe_program`); the blocks' footprints
are disjoint, and together they give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hB := B_pos hd
  have hsize : (launch d).size = G d := by simp [launch, G]
  have hWy : ∀ p, W d p ⊆ d.y.elems (ySet d) := fun p => d.y.elems_mono fun _ h => h.2.2
  have hRx : ∀ p, R d p ⊆ d.x.elems (xSet d) := fun p => d.x.elems_mono fun _ ⟨k, hk, he⟩ =>
    ⟨k, hk.2.2, he⟩
  have hWR : ∀ p q, Disjoint (W d p) (R d q) := fun p q => hd.y_x.mono (hWy p) (hRx q)
  refine correct_of_safe (fun i => R d i) (fun i => W d i) (fun i m => Q d init i m) ?_
    (fun i => ?_) ?_ ?_ ?_
  · -- the launch is valid
    refine Launch.valid_grid1 ?_ ?_
    · exact hd.grid_lt
    · intro s hs
      simp only [List.mem_singleton] at hs
      subst hs
      exact ⟨by simpa using hd.block_pow2, by simpa using hd.block_le⟩
  · -- program `i`
    have hi : (i : Nat) < G d := hsize ▸ i.isLt
    refine safe_program hd hpre _ hP.expF32 hP.logF32 hP.divFull (hP.mathTotal _ _)
      (hP.mathTotal _ _) (hP.divTotal _) hi _ ?_
    simp only [Launch.ctx, launch, Launch.grid1]
    exact Nat.mod_eq_of_lt (by simpa [G] using hi)
  · -- no program writes what another reads or writes
    intro i j hij
    refine Disjoint.union_right (hWR i j) (d.y.elems_disjoint (blk_disjoint ?_))
    exact fun h => hij (Fin.ext h)
  · -- no program reads what it writes
    intro i
    exact (hWR i i).symm
  · -- the programs' results give `Post`
    intro fin hloc hfr
    refine ⟨fun k hk => ?_, fun z hz => hfr z fun i hzi => hz (hWy i hzi)⟩
    obtain ⟨hkb, hkG⟩ := mem_blk hd hk
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨k / d.B, hsize ▸ hkG⟩
    obtain ⟨v, hv, c, hc, hcf, hcerr⟩ := hQ k hkb
    refine ⟨v, hv, c, ?_, hcf, hcerr⟩
    rw [Mem.read_congr fun x hx => hagree x (Or.inr (d.y.span_sub_elems hkb hx))]
    exact hc

end Claims.LogsigmoidFwd
