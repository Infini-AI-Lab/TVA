import Claim
import TVA.Proof
import Proof.Loops

/-!
# The body after `m_t`: lines 103–133

`tail_safe`: the generic tail `tailProg` is safe, with the loads, the `u` loop, `g`'s load and
`exp2`, and the `w` loop, given what the dtype-dependent parts (`beta`'s load and the two operand
steps) do.
-/

namespace Claims.RecomputeWU

open TVA

/-- The elements of `u` of a chunk are in the buffer. -/
theorem UE_lt {d : Inst} (hd : d.ok) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hh : h < d.HV) :
    ∀ x ∈ UE d s0 len it h, x < d.u.len := by
  rintro x ⟨r, c, hr, hrow, hc, rfl⟩
  exact lt_of_lt_of_le (idx_lt (n := nTok d) (by omega) hh hc) hd.u_len

/-- The elements of `w` of a chunk are in the buffer. -/
theorem WE_lt {d : Inst} (hd : d.ok) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hh : h < d.HV) :
    ∀ x ∈ WE d s0 len it h, x < d.w.len := by
  rintro x ⟨r, c, hr, hrow, hc, rfl⟩
  exact lt_of_lt_of_le (idx_lt (n := nTok d) (by omega) hh hc) hd.w_len

/-- The `u` and `w` elements of one chunk are disjoint bytes. -/
theorem UE_WE_disjoint {d : Inst} (hd : d.ok) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d)
    (hh : h < d.HV) : Disjoint (d.u.elems (UE d s0 len it h)) (d.w.elems (WE d s0 len it h)) :=
  d.u.elems_disjoint_of d.w hd.w_u.symm (UE_lt hd hs hh) (WE_lt hd hs hh)

/-- The cells after the `u` loop are all the cells of `u`. -/
theorem UInv_full {d : Inst} {init : Mem} {s0 len it h : Nat} {m : Mem}
    (hI : UInv d init s0 len it h ((d.V + 63) / 64) m) : UFull d init s0 len it h m := by
  intro i hi c hc
  refine hI.2.2 i hi c (lt_min hc ?_)
  omega

/-- **The tail.** -/
theorem tail_safe {E : Env} (hP : Device E.P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) {bos i_h : I64} {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool}
    (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) {β : DType}
    {loadβ : (Tensor [64] β.Val → Prog Unit) → Prog Unit}
    {vbK : Tensor [64] β.Val → Tensor [64, 64] BF16 →
      (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit}
    {kbK : Tensor [64] β.Val → Tensor [64] F32 → Tensor [64, 64] BF16 →
      (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit} {Bl : Tensor [64] β.Val → Prop}
    (hload : ∀ (W : Set Int) (Q : Unit → Mem → Prop) (m : Mem) (k : Tensor [64] β.Val → Prog Unit),
      Ag d init m → (∀ b_b, Bl b_b → Safe E (Rall d) W Q m (k b_b)) →
        Safe E (Rall d) W Q m (loadβ k))
    (hvb : ∀ b_b, Bl b_b → ∀ t, t * 64 < d.V → VbSpec E d init s0 len it h t (vbK b_b))
    (hkb : ∀ b_b l_g b_g, Bl b_b → GLane d init s0 len it h l_g → ELane l_g b_g →
      ∀ t, t * 64 < d.K → KbSpec E d init s0 len it h t (kbK b_b b_g))
    {m : Mem} (hag : Ag d init m) (hpres : Pres d m) :
    Safe E (Rall d) (d.u.elems (UE d s0 len it h) ∪ d.w.elems (WE d s0 len it h))
      (fun _ m' => WInv d init s0 len it h ((d.K + 63) / 64) m') m
      (tailProg (β := β) (d.v.ptr 0) (d.u.ptr 0) (d.A.ptr 0) (d.k.ptr 0) (d.w.ptr 0) (d.g.ptr 0)
        d.H d.HV d.K d.V bos i_h o_t m_t loadβ vbK kbK) := by
  unfold tailProg
  have hHV := HV_small hd
  refine hload _ _ _ _ hag fun b_b hBl => ?_
  -- line 107: the load of `A`
  have hS := base_val hd (X := d.HV) (Y := 64) (s0 := s0) (h := h) (by omega) le_rfl hh (by norm_num)
    (by omega) hbos hih
  have hstr : d.HV * 64 ≤ d.HV * (d.K + d.V + 64) := Nat.mul_le_mul_left _ (by omega)
  have hc : (IntV.ofNat (d.HV * 64) : I64).toInt = ((d.HV * 64 : Nat) : Int) :=
    IntV.I64.toInt_ofNat (lt_of_lt_of_le (stride_lt hd hstr) (by norm_num))
  have hfit : (len - 1) * (d.HV * 64) < 2 ^ 63 :=
    lt_of_lt_of_le (lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.sub_le len 1))
      (fit_len hd hs hstr)) (by norm_num)
  have hlen : len = 0 ∨ 64 = 0 ∨ (s0 * d.HV + h) * 64 + (len - 1) * (d.HV * 64) + (64 - 1) * 1 <
      d.A.len :=
    Or.inr (Or.inr (lt_of_lt_of_le (tile_lt hs (by omega) hh (by norm_num) le_rfl le_rfl)
      hd.A_len))
  have hcols : ∀ j : Fin 64, ((tl.arange 0 64 : Tensor [64] I32).get1 j).toInt = ((0 + j : Nat) : Int) :=
    fun j => by rw [IntV.I32.toInt_tl_arange (by norm_num) j]; simp
  have hnv : (IntV.ofNat 64 : I32).toInt = 64 := IntV.I32.toInt_ofNat (by norm_num)
  refine safe_load_tile_ptr hag d.A (fun x hx => by simp [Rall, hx]) hpre.Av hS hc ho hmt hcols hnv
    hfit hlen ?_
  intro b_A hA1 hA2
  have hA : ALane d init s0 len it h b_A := by
    intro r c
    refine ⟨fun hr => ?_, fun hr => ?_⟩
    · have hrow : it * 64 + r < len := by simpa [rowEn] using hr
      rw [hA1 r c hrow (by simp)]
      congr 1
      unfold aIdx tokOf; ring
    · have hrow : ¬ (it * 64 + r < len) := by simpa [rowEn] using hr
      exact hA2 r c (fun h' => hrow h'.1)
  -- lines 109–117: the `u` loop, then the rest
  refine uLoop_safe hP hd hpre hs hl hh hbos hih ho hmt hA (hvb b_b hBl)
    (UE_WE_disjoint hd hs hh) ?_ ⟨hag, hpres, fun i hi c hc => by simp at hc⟩
  intro m₁ hI
  -- lines 120–121: `g` and `exp2`
  refine loadG_safe hd hpre hs hl hh hbos hih ho hmt hI.1 fun l_g hGl => ?_
  refine exp2_safe hP (GLane_ok hpre hs hh hGl) fun b_g hEl => ?_
  exact wLoop_safe hP hd hpre hs hl hh hbos hih ho hmt hA (hkb b_b l_g b_g hBl hGl hEl)
    ⟨hI.1, hI.2.1, UInv_full hI, fun i hi c hc => by simp at hc⟩

end Claims.RecomputeWU
