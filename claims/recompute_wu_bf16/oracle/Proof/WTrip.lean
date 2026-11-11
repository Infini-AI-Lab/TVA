import Claim
import TVA.Proof
import Proof.UTrip

/-!
# One trip of the `w` loop (lines 124–133)
-/

namespace Claims.RecomputeWU

open TVA

/-- What lines 129–132's step `kbK` does: for any `b_k` with `KLane`, it hands its continuation a
`b_kbh` with `KbLane`. -/
def KbSpec (E : Env) (d : Inst) (init : Mem) (s0 len it h t : Nat)
    (kbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit) : Prop :=
  ∀ (W : Set Int) (Q : Unit → Mem → Prop) (m : Mem) (b_k : Tensor [64, 64] BF16)
    (k : Tensor [64, 64] BF16 → Prog Unit), KLane d init s0 len it h t b_k →
    (∀ b_kbh, KbLane d init s0 len it h t b_kbh → Safe E (Rall d) W Q m (k b_kbh)) →
    Safe E (Rall d) W Q m (kbK b_k k)

/-- **One trip of the `w` loop.** -/
theorem wTrip_safe {E : Env} (hP : Device E.P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) (ht : t * 64 < d.K) {bos i_h : I64} {o_t : Tensor [64] I64}
    {m_t : Tensor [64] Bool} {b_A : Tensor [64, 64] BF16}
    (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) (hA : ALane d init s0 len it h b_A)
    {kbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit}
    (hkb : KbSpec E d init s0 len it h t kbK) {i_k : I32} (hik : i_k.toInt = t)
    {m : Mem} (hag : Ag d init m) (hpres : Pres d m) :
    Safe E (Rall d) (tileW d s0 len it h t) (fun _ m' => WNew d init s0 len it h t m') m
      (wTrip (d.k.ptr 0) (d.w.ptr 0) d.H d.HV d.K bos i_h o_t m_t b_A kbK i_k) := by
  unfold wTrip
  have hHV := HV_small hd
  have hK31 := hd.K_lt
  have hHle := H_le_HV hd
  have hok := oV_lane hik (by omega)
  -- line 126's `i_h // (HV // H)`
  have hq0 : 0 < d.HV / d.H := by
    obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero hd.hv_div
    have := hd.h_pos
    rw [hq, Nat.mul_div_cancel_left q this]
    rcases Nat.eq_zero_or_pos q with h0 | h0
    · rw [h0] at hq; have := hd.hv_pos; omega
    · exact h0
  have hqle := Nat.div_le_self d.HV d.H
  refine safe_idiv1 (x := i_h) (y := (IntV.ofNat (d.HV / d.H) : I64)) (by rw [hih]; omega)
    (by rw [IntV.I64.toInt_ofNat (by omega)]; exact_mod_cast hq0) ?_
  intro q hq
  rw [hih, IntV.I64.toInt_ofNat (by omega)] at hq
  have hkh : q.item.toInt = (kHead d h : Nat) := by rw [hq]; rfl
  have hkhlt := kHead_lt hd hh
  have hSk := base_val hd (X := d.H) (Y := d.K) (s0 := s0) (h := kHead d h) (by omega) hHle hkhlt
    hd.k_pos (by omega) hbos hkh
  have hSw := base_val hd (X := d.HV) (Y := d.K) (by omega) le_rfl hh hd.k_pos (by omega) hbos hih
  have hstrk : d.H * d.K ≤ d.HV * (d.K + d.V + 64) :=
    le_trans (Nat.mul_le_mul hHle (by omega : d.K ≤ d.K + d.V + 64))
      (Nat.mul_le_mul_left _ (by omega))
  have hstrw : d.HV * d.K ≤ d.HV * (d.K + d.V + 64) := Nat.mul_le_mul_left _ (by omega)
  have hck : (IntV.ofNat (d.H * d.K) : I64).toInt = ((d.H * d.K : Nat) : Int) :=
    IntV.I64.toInt_ofNat (lt_of_lt_of_le (stride_lt hd hstrk) (by norm_num))
  have hcw : (IntV.ofNat (d.HV * d.K) : I64).toInt = ((d.HV * d.K : Nat) : Int) :=
    IntV.I64.toInt_ofNat (lt_of_lt_of_le (stride_lt hd hstrw) (by norm_num))
  have hfitk : (len - 1) * (d.H * d.K) < 2 ^ 63 :=
    lt_of_lt_of_le (lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.sub_le len 1))
      (fit_len hd hs hstrk)) (by norm_num)
  have hfitw : (len - 1) * (d.HV * d.K) < 2 ^ 63 :=
    lt_of_lt_of_le (lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.sub_le len 1))
      (fit_len hd hs hstrw)) (by norm_num)
  have hlenk : len = 0 ∨ d.K = 0 ∨ (s0 * d.H + kHead d h) * d.K + (len - 1) * (d.H * d.K) +
      (d.K - 1) * 1 < d.k.len :=
    Or.inr (Or.inr (lt_of_lt_of_le (tile_lt hs (by omega) hkhlt hd.k_pos le_rfl le_rfl)
      hd.k_len))
  have hlenw : len = 0 ∨ d.K = 0 ∨ eW d s0 h + (len - 1) * (d.HV * d.K) + (d.K - 1) * 1 <
      d.w.len :=
    Or.inr (Or.inr (lt_of_lt_of_le (tile_lt hs (by omega) hh hd.k_pos le_rfl le_rfl)
      hd.w_len))
  have hnv : (IntV.ofNat d.K : I32).toInt = d.K := IntV.I32.toInt_ofNat (by omega)
  refine safe_load_tile_ptr hag d.k (fun x hx => by simp [Rall, hx]) hpre.kv hSk hck ho hmt hok hnv
    hfitk hlenk ?_
  intro b_k hk1 hk2
  refine hkb _ _ _ b_k _ (fun r c hc' => ⟨fun hr => ?_, fun hr => ?_⟩) (fun b_kbh hKb => ?_)
  · have hrow : it * 64 + r < len := by simpa [rowEn] using hr
    have hcol : t * 64 + c < d.K := by simpa [colEn] using hc'
    rw [hk1 r c hrow hcol]
    congr 1
    unfold kIdx tokOf; ring
  · have hrow : ¬ (it * 64 + r < len) := by simpa [rowEn] using hr
    exact hk2 r c (fun h' => hrow h'.1)
  refine safe_dot (hP.dotTotal cfgW _ _ _) fun r hr => ?_
  refine safe_cvt (g := Format.bfloat16) ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ ?_
  intro s_w hsw
  refine safe_last ?_
  refine safe_store_tile_ptr d.w hpres.2 hSw hcw ho hmt hok hnv hfitw
    (Nat.le_mul_of_pos_left d.K hd.hv_pos) hlenw s_w (by exact fun x hx => hx) ?_
  intro m' hread hfr hpres'
  refine safe_ret ?_
  intro i j hi hj
  have hrow : it * 64 + i < len := by simpa [rowEn] using hi
  have hcol : t * 64 + j < d.K := by simpa [colEn] using hj
  have hb := w_lane_bound hP hd hpre hs hh hr hA hKb i j hi hj (o := s_w.get2 i j) (by
    have := hsw (i, j, ())
    rw [FloatV.cvtWith_rn] at this
    exact this)
  refine ⟨s_w.get2 i j, ?_, hb.1, hb.2⟩
  have := hread i j hrow hcol
  unfold bfAt
  convert this using 3
  unfold wIdx; push_cast; ring_nf

end Claims.RecomputeWU
