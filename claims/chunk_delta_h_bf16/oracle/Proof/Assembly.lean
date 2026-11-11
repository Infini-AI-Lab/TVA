import Proof.Foot
import Proof.Inv

/-!
# `chunk_delta_h_bf16`: extending `ColPre` by one chunk

After a trip, the memory differs from the one before it only in chunk `t`'s cells, which the earlier
chunks' clauses do not read (`ColPre_of_agree`); the new chunk's clauses are the trip's facts.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- **`ColPre` after `t` chunks and the facts of chunk `t` give `ColPre` after `t + 1`.** `m5` is the
memory after the trip, equal to `m` outside chunk `t`'s writes; `Yn` agrees with `Y` up to `t`. -/
theorem colPre_extend {d : Inst} (hd : d.ok) {init m m5 : Mem} {n hv jv j t : Nat}
    {Y Yn : Nat → Nat → F32} (hn : n < d.N) (hhv : hv < d.HV) (hj : inBlk d jv j) (ht : t < nch d n)
    (hold : ColPre d init m n hv j t Y) (hfr : ∀ y, y ∉ Wt d n hv jv t → m5 y = m y)
    (hYn : ∀ t', t' ≤ t → Yn t' = Y t')
    (hvn : ∀ i, i < cl d n t →
      ∃ x, rdB m5 d.vn (uIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv j) = some x ∧
        (Vals init d → (∀ t', t' ≤ t → StOk m5 d n hv j t') → x.isFinite = true ∧
          |x.toReal - vrefS init m5 d n hv j t i| ≤ tolV init m5 d n hv j t i))
    (hh : ∀ c, c < d.K → ∃ x, rdB m5 d.h (hIdx d (cbase d n + t) hv c j) = some x)
    (hcv : ∀ c, c < d.K → (Y t c).isFinite = true →
      rdB m5 d.h (hIdx d (cbase d n + t) hv c j) = some (FloatV.cvt (Y t c)))
    (hstep : Vals init d → (∀ t', t' ≤ t → StOk m5 d n hv j t') →
      (∀ c, c < d.K → (Yn t c).isFinite = true ∧ (Yn (t + 1) c).isFinite = true) ∧
      colNorm d.K (fun c => (Yn (t + 1) c).toReal -
        Step init m5 d n hv j t (fun c' => (Yn t c').toReal) c) ≤ errStep init m5 d n hv j t) :
    ColPre d init m5 n hv j (t + 1) Yn := by
  have hag : ∀ x ∈ colCells d n hv j t, m5 x = m x := fun x hx =>
    hfr x (fun hw => Set.disjoint_left.mp (colCells_disjoint_Wt hd (jv := jv) hn hhv hj ht) hx hw)
  have hold' := ColPre_of_agree hag hold
  obtain ⟨p1, p2, p3, p4, p5⟩ := hold'
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro s hs
    by_cases hst : s < BT * t
    · obtain ⟨x, hx, hr⟩ := p1 s (lt_min (lt_of_lt_of_le hs (min_le_left _ _)) hst)
      exact ⟨x, hx, hr⟩
    · have hsl := lt_of_lt_of_le hs (min_le_left _ _)
      have hs2 := lt_of_lt_of_le hs (min_le_right _ _)
      obtain ⟨x, hx, hr⟩ := hvn (s - BT * t) (by simp only [cl]; unfold BT at *; omega)
      have e1 : BT * t + (s - BT * t) = s := by omega
      have e2 : s / BT = t := by unfold BT at *; omega
      have e3 : s % BT = s - BT * t := by unfold BT at *; omega
      rw [e1] at hx
      refine ⟨x, hx, fun hV hSt => ?_⟩
      rw [e2] at hSt
      rw [e2, e3]
      exact hr hV hSt
  · intro t' ht' c hc
    by_cases hlt : t' < t
    · exact p2 t' hlt c hc
    · have : t' = t := by omega
      subst this; exact hh c hc
  · rw [hYn 0 (Nat.zero_le _)]; exact p3
  · intro t' ht' c hc hfin
    have hle : t' ≤ t := by omega
    by_cases hlt : t' < t
    · rw [hYn t' hle]; exact p4 t' hlt c hc (by rwa [hYn t' hle] at hfin)
    · have : t' = t := by omega
      subst this
      rw [hYn t' le_rfl]
      exact hcv c hc (by rwa [hYn t' le_rfl] at hfin)
  · intro t' ht' hV hSt
    by_cases hlt : t' < t
    · have := p5 t' hlt hV (fun t'' ht'' => hSt t'' (by omega))
      rw [hYn t' (by omega), hYn (t' + 1) (by omega)]
      exact this
    · have : t' = t := by omega
      subst this
      exact hstep hV hSt

end Claims.ChunkDeltaH
