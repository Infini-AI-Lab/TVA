import Claim
import TVA.Proof
import Proof.Spec
import Proof.Prog

/-!
# The operand steps

Lines 115 and 129–132: the `bfloat16` operands `b_vb` and `b_kbh` from the loaded `b_v`, `b_k`,
`b_b` (and `b_g`), for either dtype of `beta`. Each is a few operations on a tile; the lemma says
what the result's lanes hold.
-/

namespace Claims.RecomputeWU

open TVA

variable {E : Env} {R W : Set Int} {Q : Unit → Mem → Prop} {m : Mem}

/-- Lane `(r, c)` of `b[:, None]` is lane `r` of `b`. -/
theorem lane_col {α : Type} (b : Tensor [64] α) (r c : Fin 64) : (b.col 64).get (r, c, ()) = b.get1 r := rfl

/-- **Line 115 for a `float32` `beta`.** -/
theorem vbStep32_safe {d : Inst} {init : Mem} (hpre : Pre d init) (hbb : d.bbf = false)
    {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d) (hh : h < d.HV)
    {b_b : Tensor [64] F32} {b_v : Tensor [64, 64] BF16}
    {k : Tensor [64, 64] BF16 → Prog Unit}
    (hB : BLane32 d init s0 len it h b_b) (hV : VLane d init s0 len it h t b_v)
    (hk : ∀ b_vb, VbLane d init s0 len it h t b_vb → Safe E R W Q m (k b_vb)) :
    Safe E R W Q m (vbStep32 b_b b_v k) := by
  unfold vbStep32
  refine safe_cvt (g := Format.binary32) ⟨Format.bfloat16_std, Format.binary32_std, Or.inl rfl⟩ ?_
  intro b_vf h1
  refine safe_mul ?_
  intro p h2
  refine safe_cvt (g := Format.bfloat16) ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ ?_
  intro b_vb h3
  refine hk b_vb ?_
  intro r c hc
  have e1 := h1 (r, c, ())
  have e2 := h2 (r, c, ())
  have e3 := h3 (r, c, ())
  rw [lane_col] at e2
  by_cases hr : rowEn it len r = true
  · have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    have htok : tokOf s0 it r < nTok d := by unfold tokOf; omega
    have hvc : t * 64 + c < d.V := by simpa [colEn] using hc
    obtain ⟨hvf, hvb⟩ := v_ok hpre htok hh hvc
    have hbk := beta_ok hpre hbb htok hh
    have eV : b_v.get (r, c, ()) = bfV d.v init (vIdx d (tokOf s0 it r) h (t * 64 + c)) :=
      ((hV r c hc).1 hr)
    have eB : b_b.get1 r = f32V d.beta init (hIdx d (tokOf s0 it r) h) := (hB r).1 hr
    rw [eV, FloatV.cvtWith_rn] at e1
    obtain ⟨f1, -, -⟩ := FloatV.cvt_bf16_f32_exact hvf
    have e1' := e1.eq_of_isFinite f1
    rw [e1', eB] at e2
    obtain ⟨f2, f3, f4⟩ := vb32_ok (⟨hvf, hvb⟩ : KVOk _) hbk
    have e2' := e2.eq_of_isFinite f2
    rw [e2', FloatV.cvtWith_rn] at e3
    have e3' := e3.eq_of_isFinite f3
    have hval : b_vb.get2 r c = vbV d init (tokOf s0 it r) h (t * 64 + c) := by
      show b_vb.get (r, c, ()) = _
      rw [e3']
      simp only [vbV, hbb]; rfl
    refine ⟨?_, ?_, fun _ => hval, fun h' => absurd hr (by simp [h'])⟩
    · rw [hval]; simp only [vbV, hbb, Bool.false_eq_true, ite_false]; exact f3
    · rw [hval]; simp only [vbV, hbb, Bool.false_eq_true, ite_false]; exact f4
  · have hr' : rowEn it len r = false := by simpa using hr
    have eV : b_v.get (r, c, ()) = FloatV.ofPat 0 := ((hV r c hc).2 hr')
    have eB : b_b.get1 r = FloatV.ofPat 0 := (hB r).2 hr'
    rw [eV, FloatV.cvtWith_rn] at e1
    have f1 : (FloatV.cvt (FloatV.ofPat 0 : BF16) : F32).isFinite = true := by decide +kernel
    have e1' := e1.eq_of_isFinite f1
    rw [e1', eB] at e2
    have f2 : (FloatV.mul (FloatV.cvt (FloatV.ofPat 0 : BF16) : F32) (FloatV.ofPat 0)).isFinite
        = true := by decide +kernel
    have e2' := e2.eq_of_isFinite f2
    rw [e2', FloatV.cvtWith_rn] at e3
    have f3 : (FloatV.cvt (FloatV.mul (FloatV.cvt (FloatV.ofPat 0 : BF16) : F32)
        (FloatV.ofPat 0)) : BF16).isFinite = true := by decide +kernel
    have e3' := e3.eq_of_isFinite f3
    have hz : (b_vb.get2 r c).toRat = 0 := by
      show (b_vb.get (r, c, ())).toRat = 0
      rw [e3']; decide +kernel
    refine ⟨?_, ?_, fun h' => absurd h' (by simp [hr']), fun _ => hz⟩
    · show (b_vb.get (r, c, ())).isFinite = true
      rw [e3']; exact f3
    · rw [hz]; norm_num


/-- **Line 115 for a `bfloat16` `beta`.** -/
theorem vbStep16_safe {d : Inst} {init : Mem} (hpre : Pre d init) (hbb : d.bbf = true)
    {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d) (hh : h < d.HV)
    {b_b : Tensor [64] BF16} {b_v : Tensor [64, 64] BF16}
    {k : Tensor [64, 64] BF16 → Prog Unit}
    (hB : BLane16 d init s0 len it h b_b) (hV : VLane d init s0 len it h t b_v)
    (hk : ∀ b_vb, VbLane d init s0 len it h t b_vb → Safe E R W Q m (k b_vb)) :
    Safe E R W Q m (vbStep16 b_b b_v k) := by
  unfold vbStep16
  refine safe_mul ?_
  intro b_vb h2
  refine hk b_vb ?_
  intro r c hc
  have e2 := h2 (r, c, ())
  rw [lane_col] at e2
  by_cases hr : rowEn it len r = true
  · have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    have htok : tokOf s0 it r < nTok d := by unfold tokOf; omega
    have hvc : t * 64 + c < d.V := by simpa [colEn] using hc
    have hv0 := v_ok hpre htok hh hvc
    have hbk := betab_ok hpre hbb htok hh
    have eV : b_v.get (r, c, ()) = bfV d.v init (vIdx d (tokOf s0 it r) h (t * 64 + c)) :=
      ((hV r c hc).1 hr)
    rw [eV, (hB r).1 hr] at e2
    obtain ⟨f2, f4⟩ := vb16_ok hv0 hbk
    have e2' := e2.eq_of_isFinite f2
    have hval : b_vb.get2 r c = vbV d init (tokOf s0 it r) h (t * 64 + c) := by
      show b_vb.get (r, c, ()) = _
      rw [e2']; simp only [vbV, hbb, ite_true]
    refine ⟨?_, ?_, fun _ => hval, fun h' => absurd hr (by simp [h'])⟩
    · rw [hval]; simp only [vbV, hbb, ite_true]; exact f2
    · rw [hval]; simp only [vbV, hbb, ite_true]; exact f4
  · have hr' : rowEn it len r = false := by simpa using hr
    have eV : b_v.get (r, c, ()) = FloatV.ofPat 0 := ((hV r c hc).2 hr')
    rw [eV, (hB r).2 hr'] at e2
    have f2 : (FloatV.mul (FloatV.ofPat 0 : BF16) (FloatV.ofPat 0)).isFinite = true := by
      decide +kernel
    have e2' := e2.eq_of_isFinite f2
    have hz : (b_vb.get2 r c).toRat = 0 := by
      show (b_vb.get (r, c, ())).toRat = 0
      rw [e2']; decide +kernel
    refine ⟨?_, ?_, fun h' => absurd h' (by simp [hr']), fun _ => hz⟩
    · show (b_vb.get (r, c, ())).isFinite = true
      rw [e2']; exact f2
    · rw [hz]; norm_num

/-- The key head `h / (HV / H)` of a value head `h < HV` is below `H`. -/
theorem kHead_lt {d : Inst} (hd : d.ok) {h : Nat} (hh : h < d.HV) : kHead d h < d.H := by
  have hH := hd.h_pos
  obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero hd.hv_div
  have hq0 : 0 < q := by
    rcases Nat.eq_zero_or_pos q with h0 | h0
    · rw [h0] at hq; have := hd.hv_pos; omega
    · exact h0
  unfold kHead
  have : d.HV / d.H = q := by rw [hq]; exact Nat.mul_div_cancel_left q hH
  rw [this]
  exact (Nat.div_lt_iff_lt_mul hq0).mpr (by rw [hq] at hh; linarith [Nat.mul_comm d.H q])

/-- **Lines 129–132 for a `float32` `beta`.** -/
theorem kbStep32_safe {d : Inst} {init : Mem} (hd : d.ok) (hpre : Pre d init) (hbb : d.bbf = false)
    {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d) (hh : h < d.HV)
    {b_b : Tensor [64] F32} {l_g b_g : Tensor [64] F32} {b_k : Tensor [64, 64] BF16}
    {k : Tensor [64, 64] BF16 → Prog Unit}
    (hB : BLane32 d init s0 len it h b_b) (hG : GLane d init s0 len it h l_g)
    (hE : ELane l_g b_g) (hK : KLane d init s0 len it h t b_k)
    (hk : ∀ b_kbh, KbLane d init s0 len it h t b_kbh → Safe E R W Q m (k b_kbh)) :
    Safe E R W Q m (kbStep32 b_b b_g b_k k) := by
  unfold kbStep32
  refine safe_cvt (g := Format.binary32) ⟨Format.bfloat16_std, Format.binary32_std, Or.inl rfl⟩ ?_
  intro b_kf h1
  refine safe_mul ?_
  intro p1 h2
  refine safe_mul ?_
  intro p2 h3
  refine safe_cvt (g := Format.bfloat16) ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ ?_
  intro b_kbh h4
  refine hk b_kbh ?_
  intro r c hc
  have e1 := h1 (r, c, ())
  have e2 := h2 (r, c, ())
  have e3 := h3 (r, c, ())
  have e4 := h4 (r, c, ())
  rw [lane_col] at e2 e3
  obtain ⟨hEf, hEg, hEe⟩ := hE r
  have hef : (b_g.get1 r).isFinite = true := hEf
  by_cases hr : rowEn it len r = true
  · have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    have htok : tokOf s0 it r < nTok d := by unfold tokOf; omega
    have hkc : t * 64 + c < d.K := by simpa [colEn] using hc
    have hk0 := k_ok hpre htok (kHead_lt hd hh) hkc
    have hbk := beta_ok hpre hbb htok hh
    have eg : l_g.get1 r = f32V d.g init (hIdx d (tokOf s0 it r) h) := (hG r).1 hr
    have eK : b_k.get (r, c, ()) =
        bfV d.k init (kIdx d (tokOf s0 it r) (kHead d h) (t * 64 + c)) := ((hK r c hc).1 hr)
    rw [eK, FloatV.cvtWith_rn] at e1
    obtain ⟨f1, -, -⟩ := FloatV.cvt_bf16_f32_exact hk0.1
    have e1' := e1.eq_of_isFinite f1
    rw [e1', (hB r).1 hr] at e2
    rw [eg] at hEg hEe
    obtain ⟨g1, g2, g3, g4⟩ := kb32_ok hk0 hbk hEg hef hEe
    have e2' := e2.eq_of_isFinite g1
    rw [e2'] at e3
    have e3' := e3.eq_of_isFinite g2
    rw [e3', FloatV.cvtWith_rn] at e4
    have e4' := e4.eq_of_isFinite g3
    have hval : b_kbh.get2 r c = kb32 (bfV d.k init (kIdx d (tokOf s0 it r) (kHead d h) (t * 64 + c)))
        (f32V d.beta init (hIdx d (tokOf s0 it r) h)) (b_g.get1 r) := by
      show b_kbh.get (r, c, ()) = _
      rw [e4']; rfl
    refine ⟨by rw [hval]; exact g3, fun _ => ?_, fun h' => absurd hr (by simp [h'])⟩
    rw [hval]
    unfold KbNear kbX gE betaR
    simp only [hbb, Bool.false_eq_true, ite_false]
    exact g4
  · have hr' : rowEn it len r = false := by simpa using hr
    have eK : b_k.get (r, c, ()) = FloatV.ofPat 0 := ((hK r c hc).2 hr')
    rw [eK, FloatV.cvtWith_rn] at e1
    have f1 : (FloatV.cvt (FloatV.ofPat 0 : BF16) : F32).isFinite = true := by decide +kernel
    have e1' := e1.eq_of_isFinite f1
    rw [e1', (hB r).2 hr'] at e2
    have f2 : (FloatV.mul (FloatV.cvt (FloatV.ofPat 0 : BF16) : F32) (FloatV.ofPat 0)).isFinite
        = true := by decide +kernel
    have z2 : (FloatV.mul (FloatV.cvt (FloatV.ofPat 0 : BF16) : F32) (FloatV.ofPat 0)).toRat = 0 := by
      decide +kernel
    have e2' := e2.eq_of_isFinite f2
    rw [e2'] at e3
    obtain ⟨f3, z3⟩ := mul_zero_fin Format.binary32_regular f2 hef (Or.inl z2)
    have e3' := e3.eq_of_isFinite f3
    rw [e3', FloatV.cvtWith_rn] at e4
    obtain ⟨f4, z4⟩ := cvt_zero_fin (g := Format.bfloat16) Format.bfloat16_regular f3 z3
    have e4' := e4.eq_of_isFinite f4
    have hz : (b_kbh.get2 r c).toRat = 0 := by
      show (b_kbh.get (r, c, ())).toRat = 0
      rw [e4']; exact z4
    refine ⟨?_, fun h' => absurd h' (by simp [hr']), fun _ => hz⟩
    show (b_kbh.get (r, c, ())).isFinite = true
    rw [e4']; exact f4


/-- **Lines 129–132 for a `bfloat16` `beta`.** -/
theorem kbStep16_safe {d : Inst} {init : Mem} (hd : d.ok) (hpre : Pre d init) (hbb : d.bbf = true)
    {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d) (hh : h < d.HV)
    {b_b : Tensor [64] BF16} {l_g b_g : Tensor [64] F32} {b_k : Tensor [64, 64] BF16}
    {k : Tensor [64, 64] BF16 → Prog Unit}
    (hB : BLane16 d init s0 len it h b_b) (hG : GLane d init s0 len it h l_g)
    (hE : ELane l_g b_g) (hK : KLane d init s0 len it h t b_k)
    (hk : ∀ b_kbh, KbLane d init s0 len it h t b_kbh → Safe E R W Q m (k b_kbh)) :
    Safe E R W Q m (kbStep16 b_b b_g b_k k) := by
  unfold kbStep16
  refine safe_mul ?_
  intro b_kb16 h1
  refine safe_cvt (g := Format.binary32) ⟨Format.bfloat16_std, Format.binary32_std, Or.inl rfl⟩ ?_
  intro b_kbf h2
  refine safe_mul ?_
  intro p2 h3
  refine safe_cvt (g := Format.bfloat16) ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ ?_
  intro b_kbh h4
  refine hk b_kbh ?_
  intro r c hc
  have e1 := h1 (r, c, ())
  have e2 := h2 (r, c, ())
  have e3 := h3 (r, c, ())
  have e4 := h4 (r, c, ())
  rw [lane_col] at e1 e3
  obtain ⟨hEf, hEg, hEe⟩ := hE r
  have hef : (b_g.get1 r).isFinite = true := hEf
  by_cases hr : rowEn it len r = true
  · have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    have htok : tokOf s0 it r < nTok d := by unfold tokOf; omega
    have hkc : t * 64 + c < d.K := by simpa [colEn] using hc
    have hk0 := k_ok hpre htok (kHead_lt hd hh) hkc
    have hbk := betab_ok hpre hbb htok hh
    have eg : l_g.get1 r = f32V d.g init (hIdx d (tokOf s0 it r) h) := (hG r).1 hr
    have eK : b_k.get (r, c, ()) =
        bfV d.k init (kIdx d (tokOf s0 it r) (kHead d h) (t * 64 + c)) := ((hK r c hc).1 hr)
    rw [eK, (hB r).1 hr] at e1
    rw [eg] at hEg hEe
    obtain ⟨g1, g2, g3, g4⟩ := kb16_ok hk0 hbk hEg hef hEe
    have e1' := e1.eq_of_isFinite g1
    rw [e1', FloatV.cvtWith_rn] at e2
    obtain ⟨f1, -, -⟩ := FloatV.cvt_bf16_f32_exact g1
    have e2' := e2.eq_of_isFinite f1
    rw [e2'] at e3
    have e3' := e3.eq_of_isFinite g2
    rw [e3', FloatV.cvtWith_rn] at e4
    have e4' := e4.eq_of_isFinite g3
    have hval : b_kbh.get2 r c = kb16 (bfV d.k init (kIdx d (tokOf s0 it r) (kHead d h) (t * 64 + c)))
        (bfV d.betab init (hIdx d (tokOf s0 it r) h)) (b_g.get1 r) := by
      show b_kbh.get (r, c, ()) = _
      rw [e4']; rfl
    refine ⟨by rw [hval]; exact g3, fun _ => ?_, fun h' => absurd hr (by simp [h'])⟩
    rw [hval]
    unfold KbNear kbX gE betaR
    simp only [hbb, ite_true]
    exact g4
  · have hr' : rowEn it len r = false := by simpa using hr
    have eK : b_k.get (r, c, ()) = FloatV.ofPat 0 := ((hK r c hc).2 hr')
    rw [eK, (hB r).2 hr'] at e1
    have f1 : (FloatV.mul (FloatV.ofPat 0 : BF16) (FloatV.ofPat 0)).isFinite = true := by
      decide +kernel
    have e1' := e1.eq_of_isFinite f1
    rw [e1', FloatV.cvtWith_rn] at e2
    have f2 : (FloatV.cvt (FloatV.mul (FloatV.ofPat 0 : BF16) (FloatV.ofPat 0)) : F32).isFinite
        = true := by decide +kernel
    have z2 : (FloatV.cvt (FloatV.mul (FloatV.ofPat 0 : BF16) (FloatV.ofPat 0)) : F32).toRat = 0 := by
      decide +kernel
    have e2' := e2.eq_of_isFinite f2
    rw [e2'] at e3
    obtain ⟨f3, z3⟩ := mul_zero_fin Format.binary32_regular f2 hef (Or.inl z2)
    have e3' := e3.eq_of_isFinite f3
    rw [e3', FloatV.cvtWith_rn] at e4
    obtain ⟨f4, z4⟩ := cvt_zero_fin (g := Format.bfloat16) Format.bfloat16_regular f3 z3
    have e4' := e4.eq_of_isFinite f4
    have hz : (b_kbh.get2 r c).toRat = 0 := by
      show (b_kbh.get (r, c, ())).toRat = 0
      rw [e4']; exact z4
    refine ⟨?_, fun h' => absurd h' (by simp [hr']), fun _ => hz⟩
    show (b_kbh.get (r, c, ())).isFinite = true
    rw [e4']; exact f4

end Claims.RecomputeWU
