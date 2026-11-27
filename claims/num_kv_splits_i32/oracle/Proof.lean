import Proof.Body

/-!
# The proof of `num_kv_splits_i32`

The launch is one program. It reads the batch's lengths (`Rd`) and writes the output's first
`num_seq · num_group` elements (`Wout`) and, with `uneven_batch`, its element `0` (`Wu`), so
`correct_of_safe` applies with a single program. The plan:

1. `Proof/Consts.lean`: the `float32` constants `64.0`, `1.0` and small integers are exact;
2. `Proof/Float.lean`: the `float32` lines give a value at least `C` and within `C·2^-17` of
   `C · max(log₂(L/64), 1)` (`core_chain`, from `DivFull` and `Log2F32`), whose truncation
   satisfies `CoreOk`;
3. `Proof/Defs.lean`: footprints, the post-condition, the folds of `max` and `min`, the lengths'
   bounds;
4. `Proof/Ints.lean`: `tl.cdiv` without wrapping (`safe_cdiv`) and the bounds of the chunk sizes;
5. `Proof/Loads.lean`: the two loads and the values of `tl.max`, `tl.min`;
6. `Proof/Tail.lean`: lines 81-103 in stages;
7. `Proof/Loop.lean`: the store loop, trip `t` writing the elements `j ≡ t (mod num_group)`;
8. `Proof/Body.lean`: the walk (`safe_program`).
-/

namespace Claims.NumKvSplits

open TVA

/-- **The claim.** The one program is safe (`safe_program`) with the footprints `Rd` and
`Wu ∪ Wout`, which are disjoint; its memory gives `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hsize : (launch d).size = 1 := by simp [launch]
  have hRW : Disjoint (Rd d) (Wu d ∪ Wout d) := by
    refine Disjoint.union_right ?_ ?_
    · unfold Wu
      split_ifs with hu
      · refine Disjoint.mono ?_ ?_ (hd.uneven_seq hu).symm
        · exact Buf.elems_sub_bytes d.seq fun j hj => lt_of_lt_of_le hj hd.seq_len
        · exact Buf.elems_sub_bytes d.uneven fun j hj => by
            rw [Set.mem_singleton_iff.mp hj]; have := hd.uneven_len hu; omega
      · exact Set.disjoint_empty _
    · refine Disjoint.mono ?_ ?_ hd.out_seq.symm
      · exact Buf.elems_sub_bytes d.seq fun j hj => lt_of_lt_of_le hj hd.seq_len
      · exact Buf.elems_sub_bytes d.out fun j hj => lt_of_lt_of_le hj hd.out_len
  refine correct_of_safe (fun _ => Rd d) (fun _ => Wu d ∪ Wout d) (fun _ m => Qf d init m) ?_
    (fun i => safe_program hd hpre hP _) ?_ (fun _ => hRW) ?_
  · refine Launch.valid_grid1 (by norm_num) ?_
    intro s hs
    simp only [List.mem_singleton] at hs
    subst hs
    obtain ⟨e, he⟩ := hd.MAX_pow2
    exact ⟨by simpa using ⟨e, he⟩, by simpa using hd.MAX_le⟩
  · intro i j hij
    exact absurd (Fin.ext (by have := i.isLt; have := j.isLt; omega)) hij
  · intro fin hloc hfr
    obtain ⟨m, ⟨⟨E, hcore, hout⟩, hu⟩, hagree⟩ := hloc ⟨0, by rw [hsize]; norm_num⟩
    refine ⟨⟨E, hcore, fun j hj => ?_⟩, fun hU => ?_, fun z hz hzu => ?_⟩
    · obtain ⟨v, hr, hv⟩ := hout j hj
      refine ⟨v, ?_, hv⟩
      rw [Mem.read_congr (m₂ := m) fun x hx => hagree x (Or.inr (Or.inr
        (Buf.span_sub_elems d.out (S := {j | j < d.S * d.G}) hj hx)))]
      exact hr
    · obtain ⟨u, hr, hv⟩ := hu hU
      refine ⟨u, ?_, hv⟩
      rw [Mem.read_congr (m₂ := m) fun x hx => hagree x (Or.inr (Or.inl (by
        simp only [Wu, hU, ite_true]
        exact Buf.span_sub_elems d.uneven (S := {0}) (j := 0) rfl hx)))]
      exact hr
    · refine hfr z fun _ hzW => ?_
      rcases hzW with hzu' | hzo
      · unfold Wu at hzu'
        split_ifs at hzu' with hU
        · exact hzu hU hzu'
        · exact hzu'
      · exact hz hzo

end Claims.NumKvSplits
