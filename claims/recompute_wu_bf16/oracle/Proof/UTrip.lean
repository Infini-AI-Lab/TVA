import Claim
import TVA.Proof
import Proof.UBound
import Proof.TileLoads

/-!
# One trip of the `u` loop (lines 110–117)
-/

namespace Claims.RecomputeWU

open TVA

/-- What line 115's step `vbK` does: for any `b_v` with `VLane`, it hands its continuation a
`b_vb` with `VbLane`. -/
def VbSpec (E : Env) (d : Inst) (init : Mem) (s0 len it h t : Nat)
    (vbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit) : Prop :=
  ∀ (W : Set Int) (Q : Unit → Mem → Prop) (m : Mem) (b_v : Tensor [64, 64] BF16)
    (k : Tensor [64, 64] BF16 → Prog Unit), VLane d init s0 len it h t b_v →
    (∀ b_vb, VbLane d init s0 len it h t b_vb → Safe E (Rall d) W Q m (k b_vb)) →
    Safe E (Rall d) W Q m (vbK b_v k)

/-- Lane `j` of `o_v = i_v · 64 + tl.arange(0, 64)` is `t·64 + j`. -/
theorem oV_lane {i_v : I32} {t : Nat} (hiv : i_v.toInt = t) (hfit : t * 64 + 64 ≤ 2 ^ 31)
    (j : Fin 64) : ((i_v * (IntV.ofNat 64 : I32) + tl.arange 0 64).get1 j).toInt =
      ((t * 64 + j : Nat) : Int) :=
  IntV.I32.toInt_sadd_arange (IntV.mul_ofNat hiv (by omega)) hfit j

/-- **One trip of the `u` loop.** -/
theorem uTrip_safe {E : Env} (hP : Device E.P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) (ht : t * 64 < d.V) {bos i_h : I64} {o_t : Tensor [64] I64}
    {m_t : Tensor [64] Bool} {b_A : Tensor [64, 64] BF16}
    (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) (hA : ALane d init s0 len it h b_A)
    {vbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit}
    (hvb : VbSpec E d init s0 len it h t vbK) {i_v : I32} (hiv : i_v.toInt = t)
    {m : Mem} (hag : Ag d init m) (hpres : Pres d m) :
    Safe E (Rall d) (tileU d s0 len it h t) (fun _ m' => UNew d init s0 len it h t m') m
      (uTrip (d.v.ptr 0) (d.u.ptr 0) d.HV d.V bos i_h o_t m_t b_A vbK i_v) := by
  unfold uTrip
  have hHV := HV_small hd
  have hV31 := hd.V_lt
  have hov := oV_lane hiv (by omega)
  have hS := base_val hd (X := d.HV) (Y := d.V) (by omega) le_rfl hh hd.v_pos (by omega) hbos hih
  have hstr : d.HV * d.V ≤ d.HV * (d.K + d.V + 64) := Nat.mul_le_mul_left _ (by omega)
  have hc : (IntV.ofNat (d.HV * d.V) : I64).toInt = ((d.HV * d.V : Nat) : Int) :=
    IntV.I64.toInt_ofNat (lt_of_lt_of_le (stride_lt hd hstr) (by norm_num))
  have hfit : (len - 1) * (d.HV * d.V) < 2 ^ 63 :=
    lt_of_lt_of_le (lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.sub_le len 1))
      (fit_len hd hs hstr)) (by norm_num)
  have hlen : len = 0 ∨ d.V = 0 ∨ eU d s0 h + (len - 1) * (d.HV * d.V) + (d.V - 1) * 1 < d.v.len :=
    Or.inr (Or.inr (lt_of_lt_of_le (tile_lt hs (by omega) hh hd.v_pos le_rfl le_rfl)
      (hd.v_len)))
  have hlenU : len = 0 ∨ d.V = 0 ∨ eU d s0 h + (len - 1) * (d.HV * d.V) + (d.V - 1) * 1 < d.u.len :=
    Or.inr (Or.inr (lt_of_lt_of_le (tile_lt hs (by omega) hh hd.v_pos le_rfl le_rfl)
      (hd.u_len)))
  have hnv : (IntV.ofNat d.V : I32).toInt = d.V := IntV.I32.toInt_ofNat (by omega)
  refine safe_load_tile_ptr hag d.v (fun x hx => by simp [Rall, hx]) hpre.vv hS hc ho hmt hov hnv
    hfit hlen ?_
  intro b_v hv1 hv2
  refine hvb _ _ _ b_v _ (fun r c hc' => ⟨fun hr => ?_, fun hr => ?_⟩) (fun b_vb hVb => ?_)
  · have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    have hcol : t * 64 + c < d.V := by simpa [colEn] using hc'
    rw [hv1 r c hrow hcol]
    congr 1
    unfold vIdx tokOf; ring
  · have hrow : ¬ (it * 64 + r < len) := by simpa [rowEn] using hr
    exact hv2 r c (fun h' => hrow h'.1)
  refine safe_dot (hP.dotTotal cfgU _ _ _) fun r hr => ?_
  refine safe_cvt (g := Format.bfloat16) ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ ?_
  intro s_u hsu
  refine safe_last ?_
  refine safe_store_tile_ptr d.u hpres.1 hS hc ho hmt hov hnv hfit (Nat.le_mul_of_pos_left d.V hd.hv_pos) hlenU s_u (by
      exact fun x hx => hx) ?_
  intro m' hread hfr hpres'
  refine safe_ret ?_
  intro i j hi hj
  have hrow : it * 64 + i < len := by simpa [rowEn] using hi
  have hcol : t * 64 + j < d.V := by simpa [colEn] using hj
  have hb := u_lane_bound hP hpre hs hh hr hA hVb i j hi hj (o := s_u.get2 i j) (by
    have := hsu (i, j, ())
    rw [FloatV.cvtWith_rn] at this
    exact this)
  refine ⟨s_u.get2 i j, ?_, hb.1, hb.2⟩
  have := hread i j hrow hcol
  unfold bfAt
  convert this using 3
  unfold vIdx; push_cast; ring_nf

end Claims.RecomputeWU
