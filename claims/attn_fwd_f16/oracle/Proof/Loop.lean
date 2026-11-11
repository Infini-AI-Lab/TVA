import Proof.Mask

/-!
# The key loop

`safe_keyloop`: a `tl.range(lo, hi, 64)` loop whose body is safe and advances the state by one
trip is safe, and ends in the state after all its trips.
-/

namespace Claims.AttnFwd

open TVA

noncomputable section

/-- **A key loop**: from the state after `t0` blocks, `T` trips of a body that advances the state
by one block each (reading, never writing) leave the state after `t0 + T` blocks. The offsets
start at row `bh·N + lo`. -/
theorem safe_keyloop {d : Inst} {P : Platform} {init : Mem} {bh sm t0 T t1 : Nat} (ht1 : t0 + T = t1)
    (hT : t0 + T ≤ 1024) (hN : bh * d.N_CTX + 64 * (t0 + T) < 2 ^ 31)
    {lo hi offset_y : I32} (hlo : lo.toInt = ((64 * t0 : Nat) : Int))
    (hhi : hi.toInt = ((64 * (t0 + T) : Nat) : Int)) (hoff : offset_y.toInt = ((bh * d.N_CTX : Nat) : Int))
    {acc : Tensor [128, d.HEAD_DIM] F32} {l_i m_i : Tensor [128] F32}
    (hst : StateOK d init bh sm t0 m_i l_i acc)
    {body : I32 → St d → Prog (St d)}
    (hbody : ∀ t, t < T → ∀ (start_n : I32) (st : St d), start_n.toInt = ((64 * (t0 + t) : Nat) : Int) →
      st.2.2.2.1.toInt = ((bh * d.N_CTX + 64 * (t0 + t) : Nat) : Int) →
      st.2.2.2.2.toInt = ((bh * d.N_CTX + 64 * (t0 + t) : Nat) : Int) →
      StateOK d init bh sm (t0 + t) st.2.2.1 st.2.1 st.1 →
      Safe ((launch d).env P) (Rset d) ∅
        (fun x m' => m' = init ∧ StateOK d init bh sm (t0 + t + 1) x.2.2.1 x.2.1 x.1 ∧
          x.2.2.2.1.toInt = ((bh * d.N_CTX + 64 * (t0 + t + 1) : Nat) : Int) ∧
          x.2.2.2.2.toInt = ((bh * d.N_CTX + 64 * (t0 + t + 1) : Nat) : Int)) init (body start_n st)) :
    Safe ((launch d).env P) (Rset d) ∅
      (fun r m' => m' = init ∧ StateOK d init bh sm t1 r.2.2 r.2.1 r.1) init
      (tl.forRange lo hi (IntV.ofNat 64 : I32) (acc, l_i, m_i, offset_y + lo, offset_y + lo) body >>=
        fun x => pure (x.1, x.2.1, x.2.2.1)) := by
  subst ht1
  have h64 : (IntV.ofNat 64 : I32).toInt = 64 := IntV.I32.toInt_ofNat (by norm_num)
  have hrange : tl.RangeOk lo hi (IntV.ofNat 64 : I32) := by
    unfold tl.RangeOk
    rw [h64, hlo, hhi]
    simp only [IntV.maxInt]
    refine ⟨by norm_num, ?_, ?_, by norm_num⟩ <;> push_cast <;> omega
  have htrip : tl.tripCount lo.toInt hi.toInt (IntV.ofNat 64 : I32).toInt = T := by
    rw [h64, hlo, hhi]; push_cast
    unfold tl.tripCount
    have : (64 * ((t0 : Int) + T) - 64 * t0 + 64 - 1) / 64 = T := by omega
    rw [this]; simp
  refine Safe.mono (W := iterW (fun _ => (∅ : Set Int))
    (tl.tripCount lo.toInt hi.toInt (IntV.ofNat 64 : I32).toInt) ∪ ∅) ?_ (fun _ _ h => h) (by simp)
  refine safe_forRange hrange
    (fun t st m' => m' = init ∧ StateOK d init bh sm (t0 + t) st.2.2.1 st.2.1 st.1 ∧
      st.2.2.2.1.toInt = ((bh * d.N_CTX + 64 * (t0 + t) : Nat) : Int) ∧
      st.2.2.2.2.toInt = ((bh * d.N_CTX + 64 * (t0 + t) : Nat) : Int))
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by simp) ?_ ?_ ?_
  · have h1 : (offset_y + lo).toInt = ((bh * d.N_CTX + 64 * (t0 + 0) : Nat) : Int) := by
      have := IntV.I32.add_nat hoff hlo (by omega)
      simpa using this
    exact ⟨rfl, by simpa using hst, h1, h1⟩
  · intro t ht st m1 hI
    obtain ⟨rfl, hs, ho, ho2⟩ := hI
    rw [htrip] at ht
    refine (hbody t ht _ st ?_ ho ho2 hs).mono (fun x m' h => h) le_rfl
    rw [h64, hlo]
    have := IntV.I32.toInt_ofInt (x := ((64 * t0 : Nat) : Int) + t * 64)
      ⟨by omega, by omega⟩
    rw [this]; push_cast; ring
  · intro st m1 hI
    obtain ⟨rfl, hs, -⟩ := hI
    rw [htrip] at hs
    exact .ret ⟨rfl, hs⟩

end

end Claims.AttnFwd
