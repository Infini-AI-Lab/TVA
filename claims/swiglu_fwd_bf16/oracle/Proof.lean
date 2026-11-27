import Claim
import TVA.Proof
import Proof.Geo
import Proof.Walk

/-!
# The proof of `swiglu_fwd_bf16`

Program `p` of the grid `cdiv(T, B)` owns the flat elements `blk d p` (its block of `B` elements,
cut at `T = N·D`). It reads their elements of `x` and `y` and writes them in `z`, so the
footprints of distinct programs are disjoint, and `correct_of_safe` composes the programs:

1. `Proof/Real.lean`: the real-number bounds (the `bfloat16` gap below `-88.5`, `e^88.5`, the
   relative errors of `tl.exp(-a)`, `1 + E`, the division, the products and the rounding);
2. `Proof/Lane.lean`: one lane, from the device facts to `ZOk` (`lane`);
3. `Proof/Geo.lean`: the lane maps, the footprints `Rset`, `Wset`, the local postcondition `Q`,
   and the disjointness of the footprints;
4. `Proof/Walk.lean`: one program, walked line by line, is safe and leaves `Q`
   (`safe_program`);
5. `main` below: the launch is valid, the footprints are disjoint, and the blocks cover `[0, T)`.
-/

namespace Claims.SwigluFwd

open TVA

/-- **The claim.** Each program is safe on its own block (`safe_program`); the blocks' footprints
are disjoint, and together they give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hB := b_pos hd
  have hsize : (launch d).size = G d := by simp [launch, G]
  refine correct_of_safe (fun i => Rset d i) (fun i => Wset d i) (fun i m => Q d init i m) ?_
    (fun i => ?_) ?_ ?_ ?_
  · -- the launch is valid
    refine Launch.valid_grid1 hd.grid_lt ?_
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    exact ⟨by simpa using hd.b_pow2, by simpa using hd.b_le⟩
  · -- program `i`
    have hi : (i : Nat) < G d := hsize ▸ i.isLt
    refine safe_program hP hd hpre hi _ ?_
    simp only [Launch.ctx, launch, Launch.grid1]
    exact Nat.mod_eq_of_lt (by simpa [G] using hi)
  · -- no program writes what another reads or writes
    intro i j hij
    refine Disjoint.union_right (w_r_disjoint hd i j) (d.z.elems_disjoint (blk_disjoint hd ?_))
    exact fun h => hij (Fin.ext h)
  · -- no program reads what it writes
    intro i
    exact (w_r_disjoint hd i i).symm
  · -- the programs' results give `Post`
    intro fin hloc hfr
    refine ⟨fun r hr c hc => ?_, fun w hw => hfr w fun i hwi => hw (w_sub d i hwi)⟩
    have hk : r * d.D + c < d.N * d.D := by
      have : (r + 1) * d.D ≤ d.N * d.D := Nat.mul_le_mul_right _ hr
      rw [Nat.succ_mul] at this
      omega
    obtain ⟨hkb, hkG⟩ := mem_blk hd hk
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨(r * d.D + c) / d.B, hsize ▸ hkG⟩
    obtain ⟨a, b, o, ha, hb, ho, hok⟩ := hQ _ hkb
    obtain ⟨hdr, hmc⟩ := div_mod_flat hd (r := r) hc
    have ein : ∀ s : Nat, ((inIdx d s (r * d.D + c) : Nat) : Int) = (r : Int) * (s : Int) + c := by
      intro s; simp only [inIdx, hdr, hmc]; push_cast; ring
    refine ⟨a, b, o, by rw [← ein]; exact ha, by rw [← ein]; exact hb, ?_, hok⟩
    have hz : ((r * d.D + c : Nat) : Int) = (r : Int) * (d.D : Int) + c := by push_cast; ring
    rw [← hz, Mem.read_congr fun x hx => hagree x (Or.inr (d.z.span_sub_elems hkb hx))]
    exact ho

end Claims.SwigluFwd
