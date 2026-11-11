import Proof.Body
import Proof.Final
import Proof.Init

/-!
# `chunk_delta_h_bf16`: the body of a program

`body_rest` (source lines 100–175 and 318–349): the pointers, the lanes, the optional load of `h0`,
the main loop and the final stores. The loop is a hypothesis (`RunLoopSpec`): the trip rule `hT1`
(`TripNonvf`), the invariant before the first trip (`init_nonvf`) and the end (`safe_final_nonvf`)
give the result.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- **The body of a program** (`[K, V]` layout): given the program's sequence, head and block as
`int64` values and the rule of the main loop, `body_rest` leaves `Col` in each of its columns. -/
theorem safe_body_nonvf {d : Inst} (hd : d.ok) {E : Env} (_hP : Device E.P) {init : Mem} (hpre : Pre d init)
    (hvf : d.vf = false) (hT1 : TripNonvf d E init) {n hv jv : Nat} (hn : n < d.N) (hhv : hv < d.HV) (hjv : jv < NV d)
    {i_v i_nh i_h bos boh T : I64} (hiv : i_v.toInt = (jv : Int)) (hinh : i_nh.toInt = ((n * d.HV + hv : Nat) : Int))
    (hih : i_h.toInt = (hv : Int)) (hbos : bos.toInt = (bosOf d n : Int)) (hboh : boh.toInt = (cbase d n : Int))
    (hT : T.toInt = (lenOf d n : Int))
    {runLoop : (I64 → I64 → St BV → Prog (St BV)) → St BV → Prog (St BV)}
    (hrun : RunLoopSpec d E (Rset d) n hv runLoop) :
    Safe E (Rset d) (iterW (Wt d n hv jv) (nch d n) ∪ Wend d n hv jv) (fun (_ : Unit) m => QCol d init n hv jv m) init
      (body_rest (d.k.ptr 0) (d.u.ptr 0) (d.w.ptr 0) (d.vn.ptr 0) (d.h.ptr 0) (d.g.ptr 0) (d.h0.ptr 0) (d.ht.ptr 0)
        d.H d.HV d.K d.V BT BV d.useG d.useH0 d.storeHt i_v i_nh i_h bos boh T runLoop) := by
  obtain ⟨hvk0, hvw0, hvu0, hvg0, hvh0, hvvn0, hvh00, hvht0, -⟩ := hpre
  have hHle : d.H ≤ d.HV := Nat.le_of_dvd hd.HV_pos hd.H_dvd
  have hHp := hd.H_pos
  unfold body_rest
  simp (config := {zeta := false}) only []
  refine safe_idiv1 (x := i_h) (y := (IntV.ofNat (d.HV / d.H) : I64)) (by rw [hih]; omega) ?_ fun kr hkr => ?_
  · rw [IntV.I64.toInt_ofNat (by have := hd.HV_lt; have := Nat.div_le_self d.HV d.H; omega)]
    exact_mod_cast Nat.div_pos hHle hHp
  have hkr' : kr.item.toInt = (kHead d hv : Int) := by
    rw [hkr, hih, IntV.I64.toInt_ofNat (by have := hd.HV_lt; have := Nat.div_le_self d.HV d.H; omega)]
    unfold kHead; push_cast; rfl
  refine safe_convert (fun i => ConvRel.exists_i32_i64 _) fun ot hot0 => ?_
  have hL := lanes_of hd hjv hiv (ot_laneV hot0)
  have hhtp : d.storeHt = true → (d.ht.ptr 0 + i_nh * (IntV.ofNat d.K : I64) * (IntV.ofNat d.V : I64)) =
      d.ht.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int) := fun hst =>
    ptr_off d.ht rfl (nv_ptr hd hn hhv hinh (hlen_ht hd hst hn hhv) (len_le_f32 (hvht0 hst)))
  have hh0p : d.useH0 = true → (d.h0.ptr 0 + i_nh * (IntV.ofNat d.K : I64) * (IntV.ofNat d.V : I64)) =
      d.h0.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int) := fun hU =>
    ptr_off d.h0 rfl (nv_ptr hd hn hhv hinh (hlen_h0 hd hU hn hhv) (len_le_f32 (hvh00 hU)))
  have hvt : d.storeHt = true → d.ht.Valid init := hvht0
  cases hU : d.useH0
  · simp only [Bool.false_eq_true, ite_false]

    refine hrun (fun t st m => Inv d init n hv jv t st m) (Wt d n hv jv) (Wend d n hv jv)
      (fun (_ : Unit) m => QCol d init n hv jv m) init _ _ _ ?_ ?_ ?_ ?_ ?_
    · exact fun t u htu hu => Wt_disjoint hd hn hhv (Nat.ne_of_lt htu) (by omega) hu
    · exact Wend_disjoint_iterW hd hn hhv
    · exact init_nonvf hd hn hhv (Or.inl ⟨hU, rfl, rfl, rfl, rfl⟩)
    · intro t ht iT goff hiT hgoff st m1 hI
      obtain ⟨hpk, hpw, hpu, hpvn, hph⟩ := ptr_facts hd hn hhv (by omega) hbos hboh hih hkr'
        (len_le_bf16 hvk0) (len_le_bf16 hvw0) (len_le_bf16 hvu0) (len_le_bf16 hvvn0) (len_le_bf16 hvh0)
      exact hT1 n hv jv hn hhv hjv _ _ _ _ _ _ bos i_h T _ _ _ _ _ _ _ _ _ _ hpk hpw hpu hpvn hph rfl hbos hih hT
        hL t ht iT goff hiT hgoff st m1 hI
    · intro st m1 hinv
      exact safe_final_nonvf hd hvf hn hhv (hhtp) hL hvt hinv
  · simp only [ite_true]
    have hRh0 : d.h0.bytes ⊆ Rset d := bytes_sub_Rset d.h0 hd.disj_h.2.2.2.2.2.1.symm
      hd.disj_vn.2.2.2.2.1.symm hd.disj_ht.2.2.2.2.1.symm
    have hlen0 := hlen_h0 hd hU hn hhv
    have hv0 := hvh00 hU
    refine safe_h0blk hd (fun _ _ => rfl) hRh0 hv0 hvf hn hhv (by omega) (hh0p hU) hL.ok1 hL.mk1 hL.ov hL.mv
      hlen0 _ ?_
    intro r1 hr1
    refine safe_h0blk_opt hd (fun _ _ => rfl) hRh0 hv0 hvf hn hhv (by omega) (hh0p hU) hL.ok2 hL.mk2 hL.ov hL.mv
      hlen0 _ ?_
    intro r2 hr2
    refine safe_h0blk_opt hd (fun _ _ => rfl) hRh0 hv0 hvf hn hhv (by omega) (hh0p hU) hL.ok3 hL.mk3 hL.ov hL.mv
      hlen0 _ ?_
    intro r3 hr3
    refine safe_h0blk_opt hd (fun _ _ => rfl) hRh0 hv0 hvf hn hhv (by omega) (hh0p hU) hL.ok4 hL.mk4 hL.ov hL.mv
      hlen0 _ ?_
    intro r4 hr4
    refine hrun (fun t st m => Inv d init n hv jv t st m) (Wt d n hv jv) (Wend d n hv jv)
      (fun (_ : Unit) m => QCol d init n hv jv m) init _ _ _ ?_ ?_ ?_ ?_ ?_
    · exact fun t u htu hu => Wt_disjoint hd hn hhv (Nat.ne_of_lt htu) (by omega) hu
    · exact Wend_disjoint_iterW hd hn hhv
    · exact init_nonvf hd hn hhv (Or.inr ⟨hU, hr1, hr2, hr3, hr4⟩)
    · intro t ht iT goff hiT hgoff st m1 hI
      obtain ⟨hpk, hpw, hpu, hpvn, hph⟩ := ptr_facts hd hn hhv (by omega) hbos hboh hih hkr'
        (len_le_bf16 hvk0) (len_le_bf16 hvw0) (len_le_bf16 hvu0) (len_le_bf16 hvvn0) (len_le_bf16 hvh0)
      exact hT1 n hv jv hn hhv hjv _ _ _ _ _ _ bos i_h T _ _ _ _ _ _ _ _ _ _ hpk hpw hpu hpvn hph rfl hbos hih hT
        hL t ht iT goff hiT hgoff st m1 hI
    · intro st m1 hinv
      exact safe_final_nonvf hd hvf hn hhv (hhtp) hL hvt hinv


/-- **The body of a program** (`[V, K]` layout): given the program's sequence, head and block as
`int64` values and the rule of the main loop, `body_rest` leaves `Col` in each of its columns. -/
theorem safe_body_vf {d : Inst} (hd : d.ok) {E : Env} (_hP : Device E.P) {init : Mem} (hpre : Pre d init)
    (hvf : d.vf = true) (hT2 : TripVf d E init) {n hv jv : Nat} (hn : n < d.N) (hhv : hv < d.HV) (hjv : jv < NV d)
    {i_v i_nh i_h bos boh T : I64} (hiv : i_v.toInt = (jv : Int)) (hinh : i_nh.toInt = ((n * d.HV + hv : Nat) : Int))
    (hih : i_h.toInt = (hv : Int)) (hbos : bos.toInt = (bosOf d n : Int)) (hboh : boh.toInt = (cbase d n : Int))
    (hT : T.toInt = (lenOf d n : Int))
    {runLoop : (I64 → I64 → StV BV → Prog (StV BV)) → StV BV → Prog (StV BV)}
    (hrun : RunLoopSpec d E (Rset d) n hv runLoop) :
    Safe E (Rset d) (iterW (Wt d n hv jv) (nch d n) ∪ Wend d n hv jv) (fun (_ : Unit) m => QCol d init n hv jv m) init
      (body_rest_vf (d.k.ptr 0) (d.u.ptr 0) (d.w.ptr 0) (d.vn.ptr 0) (d.h.ptr 0) (d.g.ptr 0) (d.h0.ptr 0) (d.ht.ptr 0)
        d.H d.HV d.K d.V BT BV d.useG d.useH0 d.storeHt i_v i_nh i_h bos boh T runLoop) := by
  obtain ⟨hvk0, hvw0, hvu0, hvg0, hvh0, hvvn0, hvh00, hvht0, -⟩ := hpre
  have hHle : d.H ≤ d.HV := Nat.le_of_dvd hd.HV_pos hd.H_dvd
  have hHp := hd.H_pos
  unfold body_rest_vf
  simp (config := {zeta := false}) only []
  refine safe_idiv1 (x := i_h) (y := (IntV.ofNat (d.HV / d.H) : I64)) (by rw [hih]; omega) ?_ fun kr hkr => ?_
  · rw [IntV.I64.toInt_ofNat (by have := hd.HV_lt; have := Nat.div_le_self d.HV d.H; omega)]
    exact_mod_cast Nat.div_pos hHle hHp
  have hkr' : kr.item.toInt = (kHead d hv : Int) := by
    rw [hkr, hih, IntV.I64.toInt_ofNat (by have := hd.HV_lt; have := Nat.div_le_self d.HV d.H; omega)]
    unfold kHead; push_cast; rfl
  refine safe_convert (fun i => ConvRel.exists_i32_i64 _) fun ot hot0 => ?_
  have hL := lanes_of hd hjv hiv (ot_laneV hot0)
  have hhtp : d.storeHt = true → (d.ht.ptr 0 + i_nh * (IntV.ofNat d.K : I64) * (IntV.ofNat d.V : I64)) =
      d.ht.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int) := fun hst =>
    ptr_off d.ht rfl (nv_ptr hd hn hhv hinh (hlen_ht hd hst hn hhv) (len_le_f32 (hvht0 hst)))
  have hh0p : d.useH0 = true → (d.h0.ptr 0 + i_nh * (IntV.ofNat d.K : I64) * (IntV.ofNat d.V : I64)) =
      d.h0.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int) := fun hU =>
    ptr_off d.h0 rfl (nv_ptr hd hn hhv hinh (hlen_h0 hd hU hn hhv) (len_le_f32 (hvh00 hU)))
  have hvt : d.storeHt = true → d.ht.Valid init := hvht0
  cases hU : d.useH0
  · simp only [Bool.false_eq_true, ite_false]

    refine hrun (fun t st m => InvV d init n hv jv t st m) (Wt d n hv jv) (Wend d n hv jv)
      (fun (_ : Unit) m => QCol d init n hv jv m) init _ _ _ ?_ ?_ ?_ ?_ ?_
    · exact fun t u htu hu => Wt_disjoint hd hn hhv (Nat.ne_of_lt htu) (by omega) hu
    · exact Wend_disjoint_iterW hd hn hhv
    · exact init_vf hd hn hhv (Or.inl ⟨hU, rfl, rfl, rfl, rfl⟩)
    · intro t ht iT goff hiT hgoff st m1 hI
      obtain ⟨hpk, hpw, hpu, hpvn, hph⟩ := ptr_facts hd hn hhv (by omega) hbos hboh hih hkr'
        (len_le_bf16 hvk0) (len_le_bf16 hvw0) (len_le_bf16 hvu0) (len_le_bf16 hvvn0) (len_le_bf16 hvh0)
      exact hT2 n hv jv hn hhv hjv _ _ _ _ _ _ bos i_h T _ _ _ _ _ _ _ _ _ _ hpk hpw hpu hpvn hph rfl hbos hih hT
        hL t ht iT goff hiT hgoff st m1 hI
    · intro st m1 hinv
      exact safe_final_vf hd hvf hjv hn hhv (hhtp) hL hvt hinv
  · simp only [ite_true]
    have hRh0 : d.h0.bytes ⊆ Rset d := bytes_sub_Rset d.h0 hd.disj_h.2.2.2.2.2.1.symm
      hd.disj_vn.2.2.2.2.1.symm hd.disj_ht.2.2.2.2.1.symm
    have hlen0 := hlen_h0 hd hU hn hhv
    have hv0 := hvh00 hU
    refine safe_h0blk_vf hd (fun _ _ => rfl) hRh0 hv0 hvf hn hhv hjv (hh0p hU) hL.ok1 hL.mk1 hL.ov hL.mv
      hlen0 _ ?_
    intro r1 hr1
    refine safe_h0blk_vf_opt hd (fun _ _ => rfl) hRh0 hv0 hvf hn hhv hjv (hh0p hU) hL.ok2 hL.mk2 hL.ov hL.mv
      hlen0 _ ?_
    intro r2 hr2
    refine safe_h0blk_vf_opt hd (fun _ _ => rfl) hRh0 hv0 hvf hn hhv hjv (hh0p hU) hL.ok3 hL.mk3 hL.ov hL.mv
      hlen0 _ ?_
    intro r3 hr3
    refine safe_h0blk_vf_opt hd (fun _ _ => rfl) hRh0 hv0 hvf hn hhv hjv (hh0p hU) hL.ok4 hL.mk4 hL.ov hL.mv
      hlen0 _ ?_
    intro r4 hr4
    refine hrun (fun t st m => InvV d init n hv jv t st m) (Wt d n hv jv) (Wend d n hv jv)
      (fun (_ : Unit) m => QCol d init n hv jv m) init _ _ _ ?_ ?_ ?_ ?_ ?_
    · exact fun t u htu hu => Wt_disjoint hd hn hhv (Nat.ne_of_lt htu) (by omega) hu
    · exact Wend_disjoint_iterW hd hn hhv
    · exact init_vf hd hn hhv (Or.inr ⟨hU, hr1, hr2, hr3, hr4⟩)
    · intro t ht iT goff hiT hgoff st m1 hI
      obtain ⟨hpk, hpw, hpu, hpvn, hph⟩ := ptr_facts hd hn hhv (by omega) hbos hboh hih hkr'
        (len_le_bf16 hvk0) (len_le_bf16 hvw0) (len_le_bf16 hvu0) (len_le_bf16 hvvn0) (len_le_bf16 hvh0)
      exact hT2 n hv jv hn hhv hjv _ _ _ _ _ _ bos i_h T _ _ _ _ _ _ _ _ _ _ hpk hpw hpu hpvn hph rfl hbos hih hT
        hL t ht iT goff hiT hgoff st m1 hI
    · intro st m1 hinv
      exact safe_final_vf hd hvf hjv hn hhv (hhtp) hL hvt hinv


end Claims.ChunkDeltaH
