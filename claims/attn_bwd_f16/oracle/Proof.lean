import Claim
import TVA.Proof
import Proof.Footprint

/-!
# `attn_bwd_f16`: the claim

Program `i` of the grid `(N_CTX/128, 1, BATCH·N_HEAD)` is the block of `128` key (and query) rows
`128·pidX` of head `head`: it reads inputs (`inBytes`), writes its rows of `dq`, `dk`, `dv` (`Wp`)
and leaves `OutSpec` in them (`Qp`).

1. `Defs`, `Numeric`, `Pair`: the numbers. A computed `float16` `P` and `dS` of one pair are within
   the claim's bounds (`pair_good`); an accumulator lane over the rows processed so far is within
   `accB` of the exact sum (`AccInv`, one step by `AccInv.step`, the whole by `AccInv.final`).
2. `Lanes`, `Scalars`, `Basic`: the index arithmetic: pointer tiles, loads and stores of a head's
   rows, the `int32` head offsets, the launch is valid.
3. `Dkdv`, `Dq`: the two loops (`_attn_bwd_dkdv`, `_attn_bwd_dq`), each walked once for the masked
   and the unmasked call (`dkdv_safe`, `dq_safe`), one trip at a time (`dkdv_good`, `dq_good`).
4. `Stages`, `Outputs`, `Program`: the stages of the two loops, the stored values (`dv_spec`,
   `dk_spec`, `dq_spec`), and `safe_program`, the walk of the kernel body.
5. `Footprint`, here: distinct programs write disjoint tiles, the outputs share no byte with the
   inputs, and `correct_of_safe` gives `main`.
-/

namespace Claims.AttnBwd

open TVA

/-- A program's tile is inside the tensor. -/
theorem tile_sub {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * 1 * G d) :
    tileSet d i ⊆ tSet d := by
  have hp := pidX_lt hi
  have hg := mul_gx hd
  exact tileEl_sub d (head_lt hi) (by nlinarith)

/-- A program writes only output bytes. -/
theorem Wp_sub_out {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * 1 * G d) :
    Wp d i ⊆ outBytes d := by
  have h := tile_sub hd hi
  exact Set.union_subset_union (Set.union_subset_union (d.DQ.elems_mono h) (d.DK.elems_mono h))
    (d.DV.elems_mono h)

/-- Distinct programs write disjoint bytes. -/
theorem Wp_disjoint {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gx d * 1 * G d)
    (hj : j < gx d * 1 * G d) (hij : i ≠ j) : Disjoint (Wp d i) (Wp d j) := by
  have hs := tileSet_disjoint hd hi hj hij
  have ti := tile_sub hd hi
  have tj := tile_sub hd hj
  unfold Wp
  refine Disjoint.union_left (Disjoint.union_left ?_ ?_) ?_ <;>
    refine Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_
  · exact d.DQ.elems_disjoint hs
  · exact hd.dq_dk.mono (d.DQ.elems_mono ti) (d.DK.elems_mono tj)
  · exact hd.dq_dv.mono (d.DQ.elems_mono ti) (d.DV.elems_mono tj)
  · exact hd.dq_dk.symm.mono (d.DK.elems_mono ti) (d.DQ.elems_mono tj)
  · exact d.DK.elems_disjoint hs
  · exact hd.dk_dv.mono (d.DK.elems_mono ti) (d.DV.elems_mono tj)
  · exact hd.dq_dv.symm.mono (d.DV.elems_mono ti) (d.DQ.elems_mono tj)
  · exact hd.dk_dv.symm.mono (d.DV.elems_mono ti) (d.DK.elems_mono tj)
  · exact d.DV.elems_disjoint hs

/-- A memory that agrees with `m` on an element's bytes leaves `OutSpec` of that element unchanged. -/
theorem outSpec_transfer {d : Inst} {init fin m : Mem} {B : Buf .f16} {b r j : Nat}
    {E T : HeadV → ℝ} (h : OutSpec d init m B b r j E T)
    (hag : ∀ x ∈ Mem.span .f16 (B.ptr ((eIdx d b r j : Nat) : Int)).addr, fin x = m x) :
    OutSpec d init fin B b r j E T := by
  obtain ⟨c, hc, hs⟩ := h
  exact ⟨c, by rw [← hc]; exact Mem.read_congr hag, hs⟩

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP (d : Inst) hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun _ => inBytes d) (fun i => Wp d i) (fun i m => Qp d init i m)
    (launch_valid hd) (fun i => safe_program (E := (launch d).env P) hP hd hpre i) ?_ ?_ ?_
  · intro i j hij
    have hi : (i : Nat) < gx d * 1 * G d := lt_of_lt_of_eq i.isLt (launch_size d)
    have hj : (j : Nat) < gx d * 1 * G d := lt_of_lt_of_eq j.isLt (launch_size d)
    refine Disjoint.union_right ?_ (Wp_disjoint hd hi hj (fun h => hij (Fin.ext h)))
    exact hd.out_in.symm.symm.mono (Wp_sub_out hd hi) le_rfl
  · intro i
    have hi : (i : Nat) < gx d * 1 * G d := lt_of_lt_of_eq i.isLt (launch_size d)
    exact hd.out_in.symm.mono le_rfl (Wp_sub_out hd hi)
  · intro fin hloc hfr
    refine ⟨?_, fun z hz => hfr z fun i hzi => hz ?_⟩
    · intro b r j hb hr hj
      have hgx := mul_gx hd
      have hpos : 0 < gx d := by
        have : 0 < d.N_CTX := by omega
        omega
      have hi0 : b * gx d + r / 128 < (launch d).size := by
        rw [launch_size]
        have : r / 128 < gx d := by omega
        calc b * gx d + r / 128 < b * gx d + gx d := by omega
          _ = (b + 1) * gx d := by ring
          _ ≤ G d * gx d := Nat.mul_le_mul_right _ hb
          _ = gx d * 1 * G d := by ring
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨b * gx d + r / 128, hi0⟩
      have hhead : head d (b * gx d + r / 128) = b := by
        unfold head; rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt (by omega)]
        simp
      have hpid : pidX d (b * gx d + r / 128) = r / 128 := by
        unfold pidX; rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
      have hr128 : 128 * (r / 128) + r % 128 = r := Nat.div_add_mod r 128 |>.symm ▸ rfl
      obtain ⟨h1, h2, h3⟩ := hQ (r % 128) j (Nat.mod_lt _ (by norm_num)) hj
      simp only [hhead, hpid, hr128] at h1 h2 h3
      have hi : b * gx d + r / 128 < gx d * 1 * G d := by
        have := hi0; rwa [launch_size] at this
      have hmem : eIdx d b r j ∈ tileSet d (b * gx d + r / 128) :=
        ⟨r % 128, j, Nat.mod_lt _ (by norm_num), hj, by rw [hhead, hpid, hr128]⟩
      refine ⟨outSpec_transfer h1 fun x hx => hagree x (Or.inr ?_),
        outSpec_transfer h2 fun x hx => hagree x (Or.inr ?_),
        outSpec_transfer h3 fun x hx => hagree x (Or.inr ?_)⟩
      · exact Or.inr (d.DV.span_sub_elems hmem hx)
      · exact Or.inl (Or.inr (d.DK.span_sub_elems hmem hx))
      · exact Or.inl (Or.inl (d.DQ.span_sub_elems hmem hx))
    · exact Wp_sub_out hd (lt_of_lt_of_eq i.isLt (launch_size d)) hzi

end Claims.AttnBwd
