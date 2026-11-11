import Claim
import TVA.Proof
import Proof.Loads
import Proof.Spec
import Proof.Geo
import Proof.Fit

/-!
# The `[64]` loads: `beta` (line 104) and `g` (line 121)
-/

namespace Claims.RecomputeWU

open TVA

variable {E : Env} {W : Set Int} {Q : Unit → Mem → Prop}

/-- A row's `beta`/`g` element is in the tensor. -/
theorem hIdx_lt {d : Inst} {s0 len it h r : Nat} (hs : s0 + len ≤ nTok d) (hh : h < d.HV)
    (hr : it * 64 + r < len) : hIdx d (tokOf s0 it r) h < nTok d * d.HV := by
  unfold hIdx tokOf
  have : s0 + it * 64 + r + 1 ≤ nTok d := by omega
  have h1 : (s0 + it * 64 + r + 1) * d.HV ≤ nTok d * d.HV := Nat.mul_le_mul_right _ this
  nlinarith

/-- The pointer lane of `P + o_t · c`, with `P` the buffer's element `e`. -/
theorem ptr_vec {dt : DType} (b : Buf dt) {P : Ptr dt} {o_t : Tensor [64] I64} {c : I64}
    {e s it : Nat} (hP : P = b.ptr (e : Int)) (hc : c.toInt = s)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hfit : (it * 64 + 63) * s < 2 ^ 63) (r : Fin 64) :
    (P + o_t * c).get1 r = b.ptr ((e + (it * 64 + r) * s : Nat) : Int) := by
  have e1 : (P + o_t * c).get1 r = P.offset ((o_t * c).get1 r).toInt := rfl
  rw [e1, hP, Buf.ptr_offset]
  have hr := r.isLt
  have : ((o_t * c).get1 r).toInt = (((it * 64 + r) * s : Nat) : Int) :=
    IntV.mul_nat (ho r) hc (lt_of_le_of_lt (Nat.mul_le_mul_right s (by omega)) hfit)
  rw [this]; push_cast; ring_nf

/-- The pointer `buf.ptr 0 + bos·HV + i_h` is the buffer's element `s0·HV + h`. -/
theorem ptr_base_hv {dt : DType} (b : Buf dt) {d : Inst} (hd : d.ok) {bos i_h : I64} {s0 h : Nat}
    (hs : s0 < nTok d + 64) (hbos : bos.toInt = s0) (hih : i_h.toInt = h) :
    b.ptr 0 + bos * (IntV.ofNat d.HV : I64) + i_h = b.ptr ((s0 * d.HV + h : Nat) : Int) := by
  have h1 : (bos * (IntV.ofNat d.HV : I64)).toInt = ((s0 * d.HV : Nat) : Int) :=
    IntV.mul_nat hbos (IntV.I64.toInt_ofNat (by have := HV_small hd; omega)) (by
      have := fit_idx hd (f := s0) (X := d.HV) (h := 0) (Y := 1) (c := 0) hs le_rfl
        (by have := hd.hv_pos; omega) (by omega) (by omega)
      simp at this; omega)
  rw [ptr_zero_add2 b h1 hih]; push_cast; rfl

/-- **Lines 103–104 for a `float32` `beta`.** -/
theorem loadBeta32_safe {d : Inst} (hd : d.ok) {init m : Mem} (hpre : Pre d init)
    (hbb : d.bbf = false) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) {bos i_h : I64} {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool}
    (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) (hag : Ag d init m)
    {k : Tensor [64] F32 → Prog Unit}
    (hk : ∀ b_b, BLane32 d init s0 len it h b_b → Safe E (Rall d) W Q m (k b_b)) :
    Safe E (Rall d) W Q m (loadBeta32 (d.beta.ptr 0) bos i_h o_t m_t d.HV k) := by
  unfold loadBeta32
  have hHV := HV_small hd
  have hP := ptr_base_hv d.beta hd (by omega) hbos hih
  refine safe_vec_load hag d.beta (g := fun r => hIdx d (tokOf s0 it r) h)
    (fun x hx => by simp [Rall, hx]) (hpre.betav hbb) hmt ?_ ?_ ?_
  · intro r hr
    have := ptr_vec d.beta hP (c := (IntV.ofNat d.HV : I64)) (IntV.I64.toInt_ofNat (by omega)) ho
      (lt_of_lt_of_le (fit_stride hd hs hl (s := d.HV) (by nlinarith [hd.hv_pos])) (by norm_num)) r
    rw [this]
    congr 1
    unfold hIdx tokOf; push_cast; ring
  · intro r hr
    have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    exact lt_of_lt_of_le (hIdx_lt hs hh hrow) (hd.beta_len hbb)
  · intro v h1 h2
    refine hk v fun r => ⟨fun hr => ?_, fun hr => ?_⟩
    · have := h1 r hr
      unfold f32V f32At
      rw [this]; rfl
    · rw [h2 r hr _ rfl]; show FloatV.litF32 (FloatV.ofPat 0) = _
      rw [FloatV.litF32_ofPat_zero]


/-- **Lines 103–104 for a `bfloat16` `beta`.** -/
theorem loadBeta16_safe {d : Inst} (hd : d.ok) {init m : Mem} (hpre : Pre d init)
    (hbb : d.bbf = true) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) {bos i_h : I64} {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool}
    (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) (hag : Ag d init m)
    {k : Tensor [64] BF16 → Prog Unit}
    (hk : ∀ b_b, BLane16 d init s0 len it h b_b → Safe E (Rall d) W Q m (k b_b)) :
    Safe E (Rall d) W Q m (loadBeta16 (d.betab.ptr 0) bos i_h o_t m_t d.HV k) := by
  unfold loadBeta16
  have hHV := HV_small hd
  have hP := ptr_base_hv d.betab hd (by omega) hbos hih
  refine safe_conv_zero fun z hz => ?_
  refine safe_vec_load hag d.betab (g := fun r => hIdx d (tokOf s0 it r) h)
    (fun x hx => by simp [Rall, hx]) (hpre.betabv hbb) hmt ?_ ?_ ?_
  · intro r hr
    have := ptr_vec d.betab hP (c := (IntV.ofNat d.HV : I64)) (IntV.I64.toInt_ofNat (by omega)) ho
      (lt_of_lt_of_le (fit_stride hd hs hl (s := d.HV) (by nlinarith [hd.hv_pos])) (by norm_num)) r
    rw [this]
    congr 1
    unfold hIdx tokOf; push_cast; ring
  · intro r hr
    have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    exact lt_of_lt_of_le (hIdx_lt hs hh hrow) (hd.betab_len hbb)
  · intro v h1 h2
    refine hk v fun r => ⟨fun hr => ?_, fun hr => ?_⟩
    · have := h1 r hr
      unfold bfV bfAt
      rw [this]; rfl
    · rw [h2 r hr _ rfl]; exact hz (r, ())

/-- **Lines 120–121: the load of `g`**, `other=0.0`. -/
theorem loadG_safe {d : Inst} (hd : d.ok) {init m : Mem} (hpre : Pre d init)
    {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) {bos i_h : I64} {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool}
    (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) (hag : Ag d init m)
    {k : Tensor [64] F32 → Prog Unit}
    (hk : ∀ l_g, GLane d init s0 len it h l_g → Safe E (Rall d) W Q m (k l_g)) :
    Safe E (Rall d) W Q m
      (tl.load (d.g.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) +
          o_t * (IntV.ofNat d.HV : I64) : Tensor [64] (Ptr .f32)) (some m_t)
        (some (bc (FloatV.litF32 (FloatV.ofPat 0)))) >>= k) := by
  have hHV := HV_small hd
  have h1 : (bos * (IntV.ofNat d.HV : I64)).toInt = ((s0 * d.HV : Nat) : Int) :=
    IntV.mul_nat hbos (IntV.I64.toInt_ofNat (by omega)) (by
      have := fit_idx hd (f := s0) (X := d.HV) (h := 0) (Y := 1) (c := 0) (by omega) le_rfl
        (by have := hd.hv_pos; omega) (by omega) (by omega)
      simp at this; omega)
  have h2 : (bos * (IntV.ofNat d.HV : I64) + i_h).toInt = ((s0 * d.HV + h : Nat) : Int) :=
    IntV.add_nat h1 hih (by
      have := fit_idx hd (f := s0) (X := d.HV) (h := h) (Y := 1) (c := 0) (by omega) le_rfl hh
        (by omega) (by omega)
      simp at this; omega)
  have hP : d.g.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) = d.g.ptr ((s0 * d.HV + h : Nat) : Int) :=
    ptr_zero_add d.g h2
  refine safe_vec_load hag d.g (g := fun r => hIdx d (tokOf s0 it r) h)
    (fun x hx => by simp [Rall, hx]) hpre.gv hmt ?_ ?_ ?_
  · intro r hr
    have := ptr_vec d.g hP (c := (IntV.ofNat d.HV : I64)) (IntV.I64.toInt_ofNat (by omega)) ho
      (lt_of_lt_of_le (fit_stride hd hs hl (s := d.HV) (by nlinarith [hd.hv_pos])) (by norm_num)) r
    rw [this]
    congr 1
    unfold hIdx tokOf; push_cast; ring
  · intro r hr
    have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    exact lt_of_lt_of_le (hIdx_lt hs hh hrow) (hd.g_len)
  · intro v h1 h2
    refine hk v fun r => ⟨fun hr => ?_, fun hr => ?_⟩
    · have := h1 r hr
      unfold f32V f32At
      rw [this]; rfl
    · rw [h2 r hr _ rfl]; show FloatV.litF32 (FloatV.ofPat 0) = _
      rw [FloatV.litF32_ofPat_zero]


/-- The loaded `g` lanes are finite and at most `8`. -/
theorem GLane_ok {d : Inst} {init : Mem} (hpre : Pre d init) {s0 len it h : Nat}
    (hs : s0 + len ≤ nTok d) (hh : h < d.HV) {l_g : Tensor [64] F32}
    (hG : GLane d init s0 len it h l_g) (r : Fin 64) :
    (l_g.get1 r).isFinite = true ∧ (l_g.get1 r).toReal ≤ 8 := by
  by_cases hr : rowEn it len r = true
  · have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    have htok : tokOf s0 it r < nTok d := by unfold tokOf; omega
    obtain ⟨hf, hle⟩ := g_ok hpre htok hh
    rw [(hG r).1 hr]
    refine ⟨hf, ?_⟩
    unfold FloatV.toReal
    exact_mod_cast hle
  · have hr' : rowEn it len r = false := by simpa using hr
    rw [(hG r).2 hr']
    refine ⟨FloatV.isFinite_zero_f32, ?_⟩
    rw [FloatV.toReal_ofPat_zero]; norm_num

/-- **Line 121: `exp2`** of the loaded `g`. -/
theorem exp2_safe {R : Set Int} {m : Mem} (hP : Device E.P) {l_g : Tensor [64] F32}
    (hl : ∀ r : Fin 64, (l_g.get1 r).isFinite = true ∧ (l_g.get1 r).toReal ≤ 8)
    {k : Tensor [64] F32 → Prog Unit}
    (hk : ∀ b_g, ELane l_g b_g → Safe E R W Q m (k b_g)) :
    Safe E R W Q m (tl.exp2 l_g >>= k) := by
  refine safe_math (fn := .exp2) (fun i => hP.mathTotal _ _ _) ?_
  intro b_g hb
  refine hk b_g fun r => ?_
  obtain ⟨ref, href, hacc⟩ := hP.exp2F32
  have hrf : ref = fun x => (2 : ℝ) ^ x := by
    have : MathFn.ref .exp2 = some (fun x => (2 : ℝ) ^ x) := rfl
    rw [this] at href; exact (Option.some.inj href).symm
  subst hrf
  obtain ⟨hf, h8⟩ := hl r
  have h2 := two_rpow_le h8
  have h0 := two_rpow_pos (l_g.get1 r).toReal
  obtain ⟨h1, h3⟩ := hacc (l_g.get1 r) (b_g.get1 r) (hb (r, ())) hf (Set.mem_univ _) (Set.mem_univ _)
    (by
      rw [Format.binary32_maxVal_real]
      simp only [abs_of_pos h0]
      norm_num
      nlinarith)
  refine ⟨h1, h8, ?_⟩
  simpa [abs_of_pos h0] using h3

end Claims.RecomputeWU
