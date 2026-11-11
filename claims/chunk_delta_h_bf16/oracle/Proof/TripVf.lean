import Proof.GlueVf
import Proof.TripAux

/-!
# `chunk_delta_h_bf16`: the walk of a loop trip (`[V, K]` layout)

As `Trip`, for `chunk_step_vf`: the stores of the `[V, K]` blocks, the `w` dots over the transposed
blocks, the difference, the gate, and the `k` dots into zeros added to the blocks.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA TVA.Tile2

/-- A trip without the gate (`[V, K]` layout). -/
theorem trip_vf_noG {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {init : Mem}
    (hpre : Pre d init) (hvf : d.vf = true) (hG : d.useG = false) : TripVf d E init := by
  intro n hv jv hn hhv hjv kp wp vp vnp hp gp bos i_h T o_v m_v o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4
    hkp hwp hvp hvnp hhp hgp hbos hih hT hL t ht iT goff hiT hgoff st m hinv
  obtain ⟨hmem, YY, hrep, hlane⟩ := hinv
  rcases st with ⟨b1, b2, b3, b4⟩
  obtain ⟨hvk0, hvw0, hvu0, hvg0, hvh0, hvvn0, -⟩ := hpre
  obtain ⟨hRk, hRw, hRu, hRg⟩ := inputs_Rset hd
  have hlenh := hlen_h hd hn ht
  have hsc := sc_lane hd hn ht hiT
  unfold chunk_step_vf
  simp (config := {zeta := false}) only [hG, Bool.false_eq_true, ite_false]
  refine safe_convert (fun i => ConvRel.exists_i32_i64 _) fun ot hot0 => ?_
  have hot := rows_lane hd hn ht hiT (ot_lane hot0)
  -- the stores of the state blocks
  refine safe_hstore_vf hd hvf hhv hjv hhp hsc hL.ok1 hL.mk1 hL.ov hL.mv
    (valid_of_memOk hvh0 hmem) (hBlk_sub_Wt) hlenh ?_
  intro x1 hx1 m1 hr1 hf1 hp1
  have hmem1 := memOk_step hmem (hBlk_not_Rset hd hn hhv ht) hf1 hp1
  refine safe_hstore_vf_opt hd hvf hhv hjv hhp hsc hL.ok2 hL.mk2 hL.ov hL.mv
    (valid_of_memOk hvh0 hmem1) ?_ hlenh ?_
  · intro y hy
    exact ⟨hBlk_sub_Wt hy, fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 0 + 64 ≤ 64)) hy' hy⟩
  intro x2 hx2 m2 hr2 hf2 hp2
  have hmem2 := memOk_step hmem1 (hBlk_not_Rset hd hn hhv ht) hf2 hp2
  refine safe_hstore_vf_opt hd hvf hhv hjv hhp hsc hL.ok3 hL.mk3 hL.ov hL.mv
    (valid_of_memOk hvh0 hmem2) ?_ hlenh ?_
  · intro y hy
    exact ⟨⟨hBlk_sub_Wt hy, fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 0 + 64 ≤ 128)) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 64 + 64 ≤ 128)) hy' hy⟩
  intro x3 hx3 m3 hr3 hf3 hp3
  have hmem3 := memOk_step hmem2 (hBlk_not_Rset hd hn hhv ht) hf3 hp3
  refine safe_hstore_vf_opt hd hvf hhv hjv hhp hsc hL.ok4 hL.mk4 hL.ov hL.mv
    (valid_of_memOk hvh0 hmem3) ?_ hlenh ?_
  · intro y hy
    exact ⟨⟨⟨hBlk_sub_Wt hy, fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 0 + 64 ≤ 192)) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 64 + 64 ≤ 192)) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 128 + 64 ≤ 192)) hy' hy⟩
  intro x4 hx4 m4 hr4 hf4 hp4
  have hmem4 := memOk_step hmem3 (hBlk_not_Rset hd hn hhv ht) hf4 hp4
  -- the dots of the `w` tiles
  refine safe_wblock hd hP hmem4.1 hRw (valid_of_memOk hvw0 hmem4) hn hhv ht hwp hot hT hL.ok1 hL.mk1
    (len_w hd hn) b1.trans (tl.zeros [BT, BV] .f32) ?_
  intro r1 hw1
  refine safe_wblock_opt hd hP hmem4.1 hRw (valid_of_memOk hvw0 hmem4) hn hhv ht hwp hot hT hL.ok2 hL.mk2
    (len_w hd hn) b2.trans r1 ?_
  intro r2 hw2
  refine safe_wblock_opt hd hP hmem4.1 hRw (valid_of_memOk hvw0 hmem4) hn hhv ht hwp hot hT hL.ok3 hL.mk3
    (len_w hd hn) b3.trans r2 ?_
  intro r3 hw3
  refine safe_wblock_opt hd hP hmem4.1 hRw (valid_of_memOk hvw0 hmem4) hn hhv ht hwp hot hT hL.ok4 hL.mk4
    (len_w hd hn) b4.trans r3 ?_
  intro r4 hw4
  -- the `u` load, the difference and the `v_new` store
  refine safe_vblock hd hmem4.1 hRu (valid_of_memOk hvu0 hmem4) (valid_of_memOk hvvn0 hmem4) hn hhv ht hvp hvnp
    hot hT hL.ov hL.mv (len_u hd hn) (len_vn hd hn) ?_ r4 ?_
  · intro y hy
    exact ⟨⟨⟨⟨vCells_sub_Wt hy, fun hy' => Set.disjoint_left.mp (hBlk_vn_disj hd (b0 := 0) hn hhv ht) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_vn_disj hd (b0 := 64) hn hhv ht) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_vn_disj hd (b0 := 128) hn hhv ht) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_vn_disj hd (b0 := 192) hn hhv ht) hy' hy⟩
  intro bv5 m5 hv5 hf5 hp5
  have hmem5 := memOk_step hmem4 (vCells_not_Rset hd hn hhv) hf5 hp5
  -- the rounded values, the `k` tiles, the dots into zeros and the sums into the state
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun vb hcvb => ?_
  refine safe_kdot hd hP hmem5.1 hRk (valid_of_memOk hvk0 hmem5) hn hhv ht hkp hot hT hL.ok1 hL.mk1
    (len_k hd hn) vb (tl.zeros [64, BV] .f32) ?_
  intro d1 hk1
  refine safe_addc ?_
  intro y1 hy1
  refine safe_kupd_vf_opt hd hP hmem5.1 hRk (valid_of_memOk hvk0 hmem5) hn hhv ht hkp hot hT hL.ok2 hL.mk2
    (len_k hd hn) vb b2 ?_
  intro y2 hk2
  refine safe_kupd_vf_opt hd hP hmem5.1 hRk (valid_of_memOk hvk0 hmem5) hn hhv ht hkp hot hT hL.ok3 hL.mk3
    (len_k hd hn) vb b3 ?_
  intro y3 hk3
  refine safe_kupd_vf_opt hd hP hmem5.1 hRk (valid_of_memOk hvk0 hmem5) hn hhv ht hkp hot hT hL.ok4 hL.mk4
    (len_k hd hn) vb b4 ?_
  intro y4 hk4
  refine safe_ret ?_
  exact glue_vf (gt := Tensor.scalar zf32) hd hP hn hhv hjv ht hmem hrep hlane hmem5 hx1 hx2 hx3 hx4 hr1 hr2 hr3 hr4 hf1 hf2 hf3 hf4 hf5 hw1 hw2 hw3 hw4 hv5 (Or.inr ⟨hG, rfl, rfl, rfl, rfl, rfl⟩) (fun i jj => hcvb (i, jj, ())) hk1 (fun jj a => hy1 (jj, a, ())) hk2 hk3 hk4

/-- A trip with the gate (`[V, K]` layout). -/
theorem trip_vf_G {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {init : Mem}
    (hpre : Pre d init) (hvf : d.vf = true) (hG : d.useG = true) : TripVf d E init := by
  intro n hv jv hn hhv hjv kp wp vp vnp hp gp bos i_h T o_v m_v o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4
    hkp hwp hvp hvnp hhp hgp hbos hih hT hL t ht iT goff hiT hgoff st m hinv
  obtain ⟨hmem, YY, hrep, hlane⟩ := hinv
  rcases st with ⟨b1, b2, b3, b4⟩
  obtain ⟨hvk0, hvw0, hvu0, hvg0, hvh0, hvvn0, -⟩ := hpre
  obtain ⟨hRk, hRw, hRu, hRg⟩ := inputs_Rset hd
  have hlenh := hlen_h hd hn ht
  have hsc := sc_lane hd hn ht hiT
  unfold chunk_step_vf
  simp (config := {zeta := false}) only [hG, ite_true]
  refine safe_convert (fun i => ConvRel.exists_i32_i64 _) fun ot hot0 => ?_
  have hot := rows_lane hd hn ht hiT (ot_lane hot0)
  -- the stores of the state blocks
  refine safe_hstore_vf hd hvf hhv hjv hhp hsc hL.ok1 hL.mk1 hL.ov hL.mv
    (valid_of_memOk hvh0 hmem) (hBlk_sub_Wt) hlenh ?_
  intro x1 hx1 m1 hr1 hf1 hp1
  have hmem1 := memOk_step hmem (hBlk_not_Rset hd hn hhv ht) hf1 hp1
  refine safe_hstore_vf_opt hd hvf hhv hjv hhp hsc hL.ok2 hL.mk2 hL.ov hL.mv
    (valid_of_memOk hvh0 hmem1) ?_ hlenh ?_
  · intro y hy
    exact ⟨hBlk_sub_Wt hy, fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 0 + 64 ≤ 64)) hy' hy⟩
  intro x2 hx2 m2 hr2 hf2 hp2
  have hmem2 := memOk_step hmem1 (hBlk_not_Rset hd hn hhv ht) hf2 hp2
  refine safe_hstore_vf_opt hd hvf hhv hjv hhp hsc hL.ok3 hL.mk3 hL.ov hL.mv
    (valid_of_memOk hvh0 hmem2) ?_ hlenh ?_
  · intro y hy
    exact ⟨⟨hBlk_sub_Wt hy, fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 0 + 64 ≤ 128)) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 64 + 64 ≤ 128)) hy' hy⟩
  intro x3 hx3 m3 hr3 hf3 hp3
  have hmem3 := memOk_step hmem2 (hBlk_not_Rset hd hn hhv ht) hf3 hp3
  refine safe_hstore_vf_opt hd hvf hhv hjv hhp hsc hL.ok4 hL.mk4 hL.ov hL.mv
    (valid_of_memOk hvh0 hmem3) ?_ hlenh ?_
  · intro y hy
    exact ⟨⟨⟨hBlk_sub_Wt hy, fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 0 + 64 ≤ 192)) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 64 + 64 ≤ 192)) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_disj hhv (by omega : 128 + 64 ≤ 192)) hy' hy⟩
  intro x4 hx4 m4 hr4 hf4 hp4
  have hmem4 := memOk_step hmem3 (hBlk_not_Rset hd hn hhv ht) hf4 hp4
  -- the dots of the `w` tiles
  refine safe_wblock hd hP hmem4.1 hRw (valid_of_memOk hvw0 hmem4) hn hhv ht hwp hot hT hL.ok1 hL.mk1
    (len_w hd hn) b1.trans (tl.zeros [BT, BV] .f32) ?_
  intro r1 hw1
  refine safe_wblock_opt hd hP hmem4.1 hRw (valid_of_memOk hvw0 hmem4) hn hhv ht hwp hot hT hL.ok2 hL.mk2
    (len_w hd hn) b2.trans r1 ?_
  intro r2 hw2
  refine safe_wblock_opt hd hP hmem4.1 hRw (valid_of_memOk hvw0 hmem4) hn hhv ht hwp hot hT hL.ok3 hL.mk3
    (len_w hd hn) b3.trans r2 ?_
  intro r3 hw3
  refine safe_wblock_opt hd hP hmem4.1 hRw (valid_of_memOk hvw0 hmem4) hn hhv ht hwp hot hT hL.ok4 hL.mk4
    (len_w hd hn) b4.trans r3 ?_
  intro r4 hw4
  -- the `u` load, the difference and the `v_new` store
  refine safe_vblock hd hmem4.1 hRu (valid_of_memOk hvu0 hmem4) (valid_of_memOk hvvn0 hmem4) hn hhv ht hvp hvnp
    hot hT hL.ov hL.mv (len_u hd hn) (len_vn hd hn) ?_ r4 ?_
  · intro y hy
    exact ⟨⟨⟨⟨vCells_sub_Wt hy, fun hy' => Set.disjoint_left.mp (hBlk_vn_disj hd (b0 := 0) hn hhv ht) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_vn_disj hd (b0 := 64) hn hhv ht) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_vn_disj hd (b0 := 128) hn hhv ht) hy' hy⟩,
      fun hy' => Set.disjoint_left.mp (hBlk_vn_disj hd (b0 := 192) hn hhv ht) hy' hy⟩
  intro bv5 m5 hv5 hf5 hp5
  have hmem5 := memOk_step hmem4 (vCells_not_Rset hd hn hhv) hf5 hp5
  -- the gate
  have hgoff' : goff.toInt = (gTok d n hv (BT * t + cl d n t - 1) : Int) := by
    have hcl : 0 < cl d n t := by simp only [cl, BT, nch] at *; omega
    rw [hgoff hG]
    simp only [gTok]
    congr 2
    omega
  refine safe_gate hd hP hmem5.1 hRg (valid_of_memOk (hvg0 hG) hmem5) hn hhv ht hgp hbos hih hot hT hgoff'
    (len_g hd hG hn) bv5 ?_
  intro bv6 gt hgate
  refine safe_scale b1 gt.item ?_
  intro s1 hs1
  refine safe_scale_opt (c := 64 < d.K) b2 gt.item ?_
  intro s2 hs2
  refine safe_scale_opt (c := 128 < d.K) b3 gt.item ?_
  intro s3 hs3
  refine safe_scale_opt (c := 192 < d.K) b4 gt.item ?_
  intro s4 hs4
  -- the rounded values, the `k` tiles, the dots into zeros and the sums into the state
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun vb hcvb => ?_
  refine safe_kdot hd hP hmem5.1 hRk (valid_of_memOk hvk0 hmem5) hn hhv ht hkp hot hT hL.ok1 hL.mk1
    (len_k hd hn) vb (tl.zeros [64, BV] .f32) ?_
  intro d1 hk1
  refine safe_addc ?_
  intro y1 hy1
  refine safe_kupd_vf_opt hd hP hmem5.1 hRk (valid_of_memOk hvk0 hmem5) hn hhv ht hkp hot hT hL.ok2 hL.mk2
    (len_k hd hn) vb s2 ?_
  intro y2 hk2
  refine safe_kupd_vf_opt hd hP hmem5.1 hRk (valid_of_memOk hvk0 hmem5) hn hhv ht hkp hot hT hL.ok3 hL.mk3
    (len_k hd hn) vb s3 ?_
  intro y3 hk3
  refine safe_kupd_vf_opt hd hP hmem5.1 hRk (valid_of_memOk hvk0 hmem5) hn hhv ht hkp hot hT hL.ok4 hL.mk4
    (len_k hd hn) vb s4 ?_
  intro y4 hk4
  refine safe_ret ?_
  exact glue_vf hd hP hn hhv hjv ht hmem hrep hlane hmem5 hx1 hx2 hx3 hx4 hr1 hr2 hr3 hr4 hf1 hf2 hf3 hf4 hf5 hw1 hw2 hw3 hw4 hv5 (Or.inl ⟨hG, hgate, hs1, hs2, hs3, hs4⟩) (fun i jj => hcvb (i, jj, ())) hk1 (fun jj a => hy1 (jj, a, ())) hk2 hk3 hk4

/-- **One trip of the main loop** (`[V, K]` layout). -/
theorem trip_vf {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {init : Mem}
    (hpre : Pre d init) (hvf : d.vf = true) : TripVf d E init := by
  cases hG : d.useG
  · exact trip_vf_noG hd hP hpre hvf hG
  · exact trip_vf_G hd hP hpre hvf hG

end Claims.ChunkDeltaH
