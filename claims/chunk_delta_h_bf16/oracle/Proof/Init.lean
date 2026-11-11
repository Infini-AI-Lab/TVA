import Proof.Seg2
import Proof.GlueVf
import Proof.GlueNonvf

/-!
# `chunk_delta_h_bf16`: the state entering the loop

The blocks after the optional load of `h0` (the zero blocks plus the `h0` tiles) satisfy the loop
invariant before the first trip: the states `Y 0` are `h0`, finite, and the padded rows are zero.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The `h0` tiles over the blocks that exist (`[K, V]` layout). -/
theorem chain_h0 {d : Inst} (hd : d.ok) {init : Mem} {n hv jv : Nat} {z : Tensor [64, BV] F32}
    {r1 r2 r3 r4 : Tensor [64, BV] F32} (hr1 : H0Facts d init n hv jv 0 z r1)
    (hr2 : (64 < d.K ∧ H0Facts d init n hv jv 64 z r2) ∨ (¬ 64 < d.K ∧ r2 = z))
    (hr3 : (128 < d.K ∧ H0Facts d init n hv jv 128 z r3) ∨ (¬ 128 < d.K ∧ r3 = z))
    (hr4 : (192 < d.K ∧ H0Facts d init n hv jv 192 z r4) ∨ (¬ 192 < d.K ∧ r4 = z)) :
    ∀ b, b < nbOf d → H0Facts d init n hv jv (64 * b) z (fam4 r1 r2 r3 r4 b) := by
  intro b hb
  have hnb4 : nbOf d ≤ 4 := by have := hd.K_le; unfold nbOf; omega
  have hb3 : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by omega
  rcases hb3 with rfl | rfl | rfl | rfl
  · exact hr1
  · rcases hr2 with ⟨-, h⟩ | ⟨h, -⟩
    · exact h
    · exact absurd (by unfold nbOf at hb; omega) h
  · rcases hr3 with ⟨-, h⟩ | ⟨h, -⟩
    · exact h
    · exact absurd (by unfold nbOf at hb; omega) h
  · rcases hr4 with ⟨-, h⟩ | ⟨h, -⟩
    · exact h
    · exact absurd (by unfold nbOf at hb; omega) h

/-- The `h0` tiles over the blocks that exist (`[V, K]` layout). -/
theorem chain_h0_vf {d : Inst} (hd : d.ok) {init : Mem} {n hv jv : Nat} {z : Tensor [BV, 64] F32}
    {r1 r2 r3 r4 : Tensor [BV, 64] F32} (hr1 : H0FactsV d init n hv jv 0 z r1)
    (hr2 : (64 < d.K ∧ H0FactsV d init n hv jv 64 z r2) ∨ (¬ 64 < d.K ∧ r2 = z))
    (hr3 : (128 < d.K ∧ H0FactsV d init n hv jv 128 z r3) ∨ (¬ 128 < d.K ∧ r3 = z))
    (hr4 : (192 < d.K ∧ H0FactsV d init n hv jv 192 z r4) ∨ (¬ 192 < d.K ∧ r4 = z)) :
    ∀ b, b < nbOf d → H0FactsV d init n hv jv (64 * b) z (fam4 r1 r2 r3 r4 b) := by
  intro b hb
  have hnb4 : nbOf d ≤ 4 := by have := hd.K_le; unfold nbOf; omega
  have hb3 : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by omega
  rcases hb3 with rfl | rfl | rfl | rfl
  · exact hr1
  · rcases hr2 with ⟨-, h⟩ | ⟨h, -⟩
    · exact h
    · exact absurd (by unfold nbOf at hb; omega) h
  · rcases hr3 with ⟨-, h⟩ | ⟨h, -⟩
    · exact h
    · exact absurd (by unfold nbOf at hb; omega) h
  · rcases hr4 with ⟨-, h⟩ | ⟨h, -⟩
    · exact h
    · exact absurd (by unfold nbOf at hb; omega) h

/-- A lane of the zero block plus the loaded `h0` value: finite, and the initial state's value `h0v`
(a padded row or a masked column: zero). -/
theorem init_val {d : Inst} (_hd : d.ok) {init : Mem} (hV : Vals init d) {n hv j c : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) (hj : j < d.V) (huse : d.useH0 = true) {v r : F32}
    (hvr : (c < d.K → rdF init d.h0 (sIdx d n hv c j) = some v)) (hvz : (¬ c < d.K → v = zf32))
    (hadd : FloatV.NaNEq (FloatV.add (FloatV.ofPat 0) v) r) :
    r.isFinite = true ∧ (c < d.K → r.toReal = h0v init d n hv j c) ∧ (d.K ≤ c → r.toReal = 0) := by
  by_cases hc : c < d.K
  · obtain ⟨x, hx, hf, -⟩ := hV.2.2.2.2.1 huse (sIdx d n hv c j) (sIdx_lt hn hhv hc hj)
    rw [hvr hc] at hx; cases hx
    obtain ⟨hrf, hrr⟩ := add_zero_left hf hadd
    refine ⟨hrf, fun _ => ?_, fun h => absurd h (not_le.2 hc)⟩
    unfold h0v
    rw [ite_eq_left huse, valF_of_rd (hvr hc)]
    unfold FloatV.toReal; rw [hrr]
  · have hz := hvz hc
    subst hz
    obtain ⟨hrf, hrr⟩ := add_zero_left zf_zf32.1 hadd
    refine ⟨hrf, fun h => absurd h hc, fun _ => ?_⟩
    unfold FloatV.toReal
    rw [hrr]
    have := zf_zf32.2
    unfold FloatV.toReal at this
    rw [this]

/-- **The invariant before the first trip** (`[K, V]` layout). -/
theorem init_nonvf {d : Inst} (hd : d.ok) {init : Mem} {n hv jv : Nat} (hn : n < d.N) (hhv : hv < d.HV)
    {r1 r2 r3 r4 : Tensor [64, BV] F32}
    (hr : (d.useH0 = false ∧ r1 = tl.zeros [64, BV] .f32 ∧ r2 = tl.zeros [64, BV] .f32 ∧
        r3 = tl.zeros [64, BV] .f32 ∧ r4 = tl.zeros [64, BV] .f32) ∨
      (d.useH0 = true ∧ H0Facts d init n hv jv 0 (tl.zeros [64, BV] .f32) r1 ∧
        ((64 < d.K ∧ H0Facts d init n hv jv 64 (tl.zeros [64, BV] .f32) r2) ∨
          (¬ 64 < d.K ∧ r2 = tl.zeros [64, BV] .f32)) ∧
        ((128 < d.K ∧ H0Facts d init n hv jv 128 (tl.zeros [64, BV] .f32) r3) ∨
          (¬ 128 < d.K ∧ r3 = tl.zeros [64, BV] .f32)) ∧
        ((192 < d.K ∧ H0Facts d init n hv jv 192 (tl.zeros [64, BV] .f32) r4) ∨
          (¬ 192 < d.K ∧ r4 = tl.zeros [64, BV] .f32)))) :
    Inv d init n hv jv 0 (r1, r2, r3, r4) init := by
  obtain ⟨hnb, hnb'⟩ := nbOf_bounds (d := d)
  refine ⟨⟨fun _ _ => rfl, fun _ h => h⟩, fun jj _ c => flat (fam4 r1 r2 r3 r4) c jj, ?_, ?_⟩
  · intro b hb a jj
    rw [blk_fam4]
    exact (flat_blk _ b a jj).symm
  · intro jj hjj hin
    lift jj to Fin BV using hjj with jf
    have hjV : BV * jv + jf < d.V := hin.2.2
    have hlane : Vals init d → ∀ c, c < 64 * nbOf d → (flat (fam4 r1 r2 r3 r4) c jf).isFinite = true ∧
        (c < d.K → (flat (fam4 r1 r2 r3 r4) c jf).toReal = h0v init d n hv (BV * jv + jf) c) ∧
        (d.K ≤ c → (flat (fam4 r1 r2 r3 r4) c jf).toReal = 0) := by
      intro hV c hc
      obtain ⟨b, a, rfl⟩ := split64 c
      have hb : b < nbOf d := by omega
      rcases hr with ⟨hU, h1, h2, h3, h4⟩ | ⟨hU, h1, h2, h3, h4⟩
      · have hf : fam4 r1 r2 r3 r4 b = tl.zeros [64, BV] .f32 := by
          have : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by have := hd.K_le; unfold nbOf at hb; omega
          rcases this with rfl | rfl | rfl | rfl
          exacts [h1, h2, h3, h4]
        rw [flat_blk, hf]
        obtain ⟨hz1, hz2⟩ := zeros_lane (m := 64) (n := BV) a jf
        refine ⟨hz1, fun _ => ?_, fun _ => hz2⟩
        unfold h0v; rw [ite_eq_right (by simp [hU]), hz2]
      · obtain ⟨v, hv1, hv2, hv3⟩ := chain_h0 hd h1 h2 h3 h4 b hb
        rw [flat_blk]
        have := init_val hd hV hn hhv hjV hU (v := v.get2 a jf) (r := (fam4 r1 r2 r3 r4 b).get2 a jf)
          (c := 64 * b + a)
          (fun hcK => hv1 a jf hcK hjV) (fun hcK => hv2 a jf (fun h => hcK h.1)) (hv3 a jf)
        exact this
    refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
    · intro s hs; exact absurd hs (by simp [BT])
    · intro t' ht'; exact absurd ht' (Nat.not_lt_zero _)
    · intro hV c hc
      have := hlane hV c (by omega)
      exact ⟨this.1, this.2.1 hc⟩
    · intro t' ht'; exact absurd ht' (Nat.not_lt_zero _)
    · intro t' ht'; exact absurd ht' (Nat.not_lt_zero _)
    · intro hV _
      refine ⟨fun c hc => (hlane hV c (by omega)).1, fun c hc1 hc2 => ⟨(hlane hV c hc2).1, ?_⟩⟩
      rw [(hlane hV c hc2).2.2 hc1]
      simp [padB]


/-- **The invariant before the first trip** (`[V, K]` layout). -/
theorem init_vf {d : Inst} (hd : d.ok) {init : Mem} {n hv jv : Nat} (hn : n < d.N) (hhv : hv < d.HV)
    {r1 r2 r3 r4 : Tensor [BV, 64] F32}
    (hr : (d.useH0 = false ∧ r1 = tl.zeros [BV, 64] .f32 ∧ r2 = tl.zeros [BV, 64] .f32 ∧
        r3 = tl.zeros [BV, 64] .f32 ∧ r4 = tl.zeros [BV, 64] .f32) ∨
      (d.useH0 = true ∧ H0FactsV d init n hv jv 0 (tl.zeros [BV, 64] .f32) r1 ∧
        ((64 < d.K ∧ H0FactsV d init n hv jv 64 (tl.zeros [BV, 64] .f32) r2) ∨
          (¬ 64 < d.K ∧ r2 = tl.zeros [BV, 64] .f32)) ∧
        ((128 < d.K ∧ H0FactsV d init n hv jv 128 (tl.zeros [BV, 64] .f32) r3) ∨
          (¬ 128 < d.K ∧ r3 = tl.zeros [BV, 64] .f32)) ∧
        ((192 < d.K ∧ H0FactsV d init n hv jv 192 (tl.zeros [BV, 64] .f32) r4) ∨
          (¬ 192 < d.K ∧ r4 = tl.zeros [BV, 64] .f32)))) :
    InvV d init n hv jv 0 (r1, r2, r3, r4) init := by
  obtain ⟨hnb, hnb'⟩ := nbOf_bounds (d := d)
  refine ⟨⟨fun _ _ => rfl, fun _ h => h⟩, fun jj _ c => flatV (fam4 r1 r2 r3 r4) c jj, ?_, ?_⟩
  · intro b hb a jj
    rw [blkV_fam4]
    exact (flatV_blk _ b a jj).symm
  · intro jj hjj hin
    lift jj to Fin BV using hjj with jf
    have hjV : BV * jv + jf < d.V := hin.2.2
    have hlane : Vals init d → ∀ c, c < 64 * nbOf d → (flatV (fam4 r1 r2 r3 r4) c jf).isFinite = true ∧
        (c < d.K → (flatV (fam4 r1 r2 r3 r4) c jf).toReal = h0v init d n hv (BV * jv + jf) c) ∧
        (d.K ≤ c → (flatV (fam4 r1 r2 r3 r4) c jf).toReal = 0) := by
      intro hV c hc
      obtain ⟨b, a, rfl⟩ := split64 c
      have hb : b < nbOf d := by omega
      rcases hr with ⟨hU, h1, h2, h3, h4⟩ | ⟨hU, h1, h2, h3, h4⟩
      · have hf : fam4 r1 r2 r3 r4 b = tl.zeros [BV, 64] .f32 := by
          have : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by have := hd.K_le; unfold nbOf at hb; omega
          rcases this with rfl | rfl | rfl | rfl
          exacts [h1, h2, h3, h4]
        rw [flatV_blk, hf]
        obtain ⟨hz1, hz2⟩ := zeros_lane (m := BV) (n := 64) jf a
        refine ⟨hz1, fun _ => ?_, fun _ => hz2⟩
        unfold h0v; rw [ite_eq_right (by simp [hU]), hz2]
      · obtain ⟨v, hv1, hv2, hv3⟩ := chain_h0_vf hd h1 h2 h3 h4 b hb
        rw [flatV_blk]
        have := init_val hd hV hn hhv hjV hU (v := v.get2 jf a) (r := (fam4 r1 r2 r3 r4 b).get2 jf a)
          (c := 64 * b + a)
          (fun hcK => hv1 jf a hcK hjV) (fun hcK => hv2 jf a (fun h => hcK h.1)) (hv3 jf a)
        exact this
    refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
    · intro s hs; exact absurd hs (by simp [BT])
    · intro t' ht'; exact absurd ht' (Nat.not_lt_zero _)
    · intro hV c hc
      have := hlane hV c (by omega)
      exact ⟨this.1, this.2.1 hc⟩
    · intro t' ht'; exact absurd ht' (Nat.not_lt_zero _)
    · intro t' ht'; exact absurd ht' (Nat.not_lt_zero _)
    · intro hV _
      refine ⟨fun c hc => (hlane hV c (by omega)).1, fun c hc1 hc2 => ⟨(hlane hV c hc2).1, ?_⟩⟩
      rw [(hlane hV c hc2).2.2 hc1]
      simp [padB]


end Claims.ChunkDeltaH
