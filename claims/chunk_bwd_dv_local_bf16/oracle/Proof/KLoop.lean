import Claim
import TVA.Proof
import Proof.Setup
import Proof.Spec
import Proof.NumKStep
import Proof.NumDot
import Proof.ValsLane
import Proof.Prologue
import Proof.KStep

/-!
# The gate block and the `K` loop

Lines 119–136. `safe_gLoadK`: the gate block holds `g` at the enabled lanes. `safe_kLoopK`: the
`K` loop's accumulator is within the bound of `KFinal`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- **The gate block** is safe and gives `GOk`. -/
theorem safe_gLoadK {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r bh : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T)
    {E : Env} {α : Type} {Wk : Set Int} {Q : α → Mem → Prop}
    {cont : Tensor [d.BT] F32 → Prog α}
    (hcont : ∀ b_g, GOk d init r bh b_g → Safe E (R d r bh) Wk Q init (cont b_g)) :
    Safe E (R d r bh) Wk Q init (gLoadK d gb o_t (o_t.ltS T) cont) := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  have hV : ValsOk d init := hpre.2.2.2.2.2.2
  unfold gLoadK
  by_cases hG : d.useG = true
  · simp only [hG, ite_true]
    have hgv : d.g.Valid init := hpre.2.2.2.2.1 hG
    have hlt := g_tile_lt hd hT hr hbh
    have hlen : d.B * d.T * d.HV ≤ d.g.len := hd.g_len hG
    have hglen := buf_len_le hgv
    simp only [DType.bytes] at hglen
    have hHV : d.HV < 2 ^ 31 := by
      have := hd.hvv_lt
      have h' : d.HV ≤ d.HV * d.V := Nat.le_mul_of_pos_right _ hd.V_pos
      omega
    -- the elements of the lanes
    let ge : Idx [d.BT] → Nat := fun i => gOff d r bh + (chkOf d r bh * d.BT + i.1.val) * d.HV
    let ee : Idx [d.BT] → Bool := fun i => en d r bh i.1.val
    have hin : ∀ i, ee i = true → ge i < d.g.len := by
      intro i hi
      have hi' : chkOf d r bh * d.BT + i.1.val ≤ lenOf d r bh - 1 := by
        have := (en_iff r bh i.1.val).mp hi; omega
      have : (chkOf d r bh * d.BT + i.1.val) * d.HV ≤ (lenOf d r bh - 1) * d.HV :=
        Nat.mul_le_mul_right _ hi'
      show gOff d r bh + (chkOf d r bh * d.BT + i.1.val) * d.HV < d.g.len
      omega
    have hge : ∀ l : Fin d.BT, ge (l, ()) =
        gIdx d (pos d r bh l) (hvOf d bh) := by
      intro l
      show gOff d r bh + (chkOf d r bh * d.BT + l.val) * d.HV = _
      unfold gOff gIdx pos baseOf
      ring
    have hsub : d.g.elems (Buf.img ge ee) ⊆ d.g.bytes :=
      Buf.elems_sub_bytes d.g (fun j ⟨i, hi, hj⟩ => hj ▸ hin i hi)
    refine safe_load_buf d.g (g := ge) (en := ee) ?_ hin hgv ?_ ?_
    · refine Buf.addrs_of_get d.g (fun i => ?_) (fun i hi => ?_)
      · exact mt_get S i.1
      · set l : Fin d.BT := i.1 with hl
        have hi' : chkOf d r bh * d.BT + l.val ≤ lenOf d r bh - 1 := by
          have := (en_iff r bh l.val).mp hi; omega
        have hmul : (chkOf d r bh * d.BT + l.val) * d.HV ≤ (lenOf d r bh - 1) * d.HV :=
          Nat.mul_le_mul_right _ hi'
        have hlt' : (chkOf d r bh * d.BT + l.val) * d.HV < 2 ^ 63 := by omega
        have e1 : (o_t.get1 l * (IntV.ofNat d.HV : I64)).toInt =
            (((chkOf d r bh * d.BT + l.val) * d.HV : Nat) : Int) :=
          IntV.I64.mul_nat (S.ot l) (IntV.I64.toInt_ofNat (by omega)) hlt'
        have e2 : ((o_t * (IntV.ofNat d.HV : I64)).get i) = o_t.get1 l * (IntV.ofNat d.HV : I64) := rfl
        rw [Tensor.get_ptrAdd, e2, e1, S.g_eq, Buf.ptr_offset]
        congr 1
    · intro z hz
      have := hsub hz
      unfold R
      simp only [hG, ite_true, Set.mem_union]
      tauto
    · intro v hx hxo
      refine hcont v fun _ l hl => ?_
      have hread := hx (l, ()) hl
      rw [hge l] at hread
      obtain ⟨-, -, hgfin⟩ := vals_tok hd hT hV hr hbh hl
      obtain ⟨xv, hxv, hxfin⟩ := hgfin hG (hvOf d bh) (hvOf_lt hd bh)
      unfold rdF at hxv
      have hvx : v.get (l, ()) = xv := by
        have := hread.symm.trans hxv
        exact Option.some.inj this
      refine ⟨by rw [show v.get1 l = v.get (l, ()) from rfl, hvx]; exact hxfin, ?_⟩
      unfold valF rdF
      rw [hxv]
      show (v.get (l, ())).toRat = xv.toRat
      rw [hvx]
  · have hG' : d.useG = false := by simpa using hG
    simp only [hG', Bool.false_eq_true, ite_false, Prog.pure_eq]
    exact hcont _ fun h => absurd h hG

/-- A loop whose body writes nothing (`safe_forRange` with empty trip footprints). -/
theorem safe_forRange_ro {σ α : Type} {sg : Bool} {w : Nat} {lo hi step : IntV sg w} {init' : σ}
    {body : IntV sg w → σ → Prog σ} {k : σ → Prog α} {E : Env} {R : Set Int}
    {Q : α → Mem → Prop} {m : Mem} {Wk : Set Int}
    (hrange : tl.RangeOk lo hi step) (I : Nat → σ → Mem → Prop) (h0 : I 0 init' m)
    (hstep : ∀ t, t < tl.tripCount lo.toInt hi.toInt step.toInt → ∀ st m₁, I t st m₁ →
      Safe E R ∅ (fun st' m₂ => I (t + 1) st' m₂) m₁
        (body (IntV.ofInt (lo.toInt + t * step.toInt)) st))
    (hk : ∀ st m₁, I (tl.tripCount lo.toInt hi.toInt step.toInt) st m₁ → Safe E R Wk Q m₁ (k st)) :
    Safe E R Wk Q m (tl.forRange lo hi step init' body >>= k) := by
  have h := safe_forRange (E := E) (R := R) (Q := Q) (m := m) hrange I (fun _ => ∅) (Wk := Wk)
    (fun t u _ _ => by simp) (by rw [iterW_empty]; exact Set.empty_disjoint Wk) h0 hstep hk
  rw [iterW_empty, Set.empty_union] at h
  exact h

/-- The number of trips of the `K` loop is at most `16` (`K ≤ 256`, `BK ≥ 16`). -/
theorem trips_le {d : Inst} (hd : d.ok) : (d.K + d.BK - 1) / d.BK ≤ 16 := by
  have h1 := hd.K_le; have h2 := hd.bk_ge
  rw [Nat.div_le_iff_le_mul_add_pred (by omega)]
  omega

/-- **The `K` loop** is safe and gives `KFinal`. -/
theorem safe_kLoopK {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T)
    {α : Type} {Wk : Set Int} {Q : α → Mem → Prop}
    {cont : Tensor [d.BT, d.BT] F32 → Prog α}
    (hcont : ∀ b_A, KFinal d init r bh b_A →
      Safe ((launch d).env P) (R d r bh) Wk Q init (cont b_A)) :
    Safe ((launch d).env P) (R d r bh) Wk Q init (kLoopK d q k o_t (o_t.ltS T) cont) := by
  have hTrips := trips_le hd
  have hK31 : (d.K + d.BK - 1) / d.BK < 2 ^ 31 := by omega
  have hlo : (IntV.ofInt 0 : I32).toInt = 0 := by decide
  have hstp : (IntV.ofInt 1 : I32).toInt = 1 := by decide
  have hhi : (IntV.ofNat ((d.K + d.BK - 1) / d.BK) : I32).toInt = (((d.K + d.BK - 1) / d.BK : Nat) : Int) :=
    IntV.I32.toInt_ofNat hK31
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat ((d.K + d.BK - 1) / d.BK) : I32)
      (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk
    rw [hlo, hstp, hhi]
    have : IntV.maxInt true 32 = 2 ^ 31 - 1 := by decide
    rw [this]
    refine ⟨by norm_num, ?_, by norm_num, by norm_num⟩
    have : (((d.K + d.BK - 1) / d.BK : Nat) : Int) ≤ 16 := by exact_mod_cast hTrips
    omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt
      (IntV.ofNat ((d.K + d.BK - 1) / d.BK) : I32).toInt (IntV.ofInt 1 : I32).toInt =
      (d.K + d.BK - 1) / d.BK := by
    rw [hlo, hstp, hhi]
    unfold tl.tripCount
    generalize (d.K + d.BK - 1) / d.BK = N
    omega
  unfold kLoopK
  refine safe_forRange_ro hrange (fun t st m => m = init ∧ KInv d init r bh t st) ⟨rfl, ?_⟩ ?_ ?_
  · -- the zero accumulator
    intro a b _ _ _
    refine ⟨(FloatV.isFinite_ofPat_zero Format.binary32_regular), ?_⟩
    show |(FloatV.ofPat 0 : F32).toReal - _| ≤ _
    simp [kqP, kqMagP, FloatV.toReal_ofPat_zero]
  · -- one trip
    intro t ht st m₁ ⟨hm, hinv⟩
    obtain rfl : init = m₁ := hm.symm
    rw [htc] at ht
    have htK : t * d.BK < d.K := by
      have hBK0 : 0 < d.BK := by have := hd.bk_ge; omega
      have h1 : (t + 1) * d.BK ≤ d.K + d.BK - 1 := (Nat.le_div_iff_mul_le hBK0).mp ht
      have h2 : (t + 1) * d.BK = t * d.BK + d.BK := by ring
      omega
    have ht15 : t ≤ 15 := by omega
    have hik : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32).toInt
        = (t : Int) := by
      rw [hlo, hstp]
      rw [IntV.toInt_ofInt (by norm_num) (by constructor <;> omega)]
      ring
    unfold kBody
    refine safe_kload_k hd hpre hr hbh S htK hik fun bk hbk => ?_
    refine safe_kload_q hd hpre hr hbh S htK hik fun bq hbq => ?_
    refine safe_dot (hP.dotTotal cfgBF16 bk bq _) fun X hX => ?_
    refine safe_last ?_
    refine safe_mul_add fun r' hr' => ?_
    exact Safe.ret ⟨rfl, kinv_step hP hd hpre hr hbh htK ht15 hbk hbq hX hinv
      (fun a b => hr' (a, b, ()))⟩
  · -- the end of the loop
    intro st m₁ ⟨hm, hinv⟩
    obtain rfl : init = m₁ := hm.symm
    rw [htc] at hinv
    refine hcont st fun a b hae hbe hab => ?_
    obtain ⟨hf, he⟩ := hinv a b hae hbe hab
    have hBK0 : 0 < d.BK := by have := hd.bk_ge; omega
    have hmin : min ((d.K + d.BK - 1) / d.BK * d.BK) d.K = d.K := by
      have h1 := Nat.div_add_mod (d.K + d.BK - 1) d.BK
      have h2 := Nat.mod_lt (d.K + d.BK - 1) hBK0
      have : d.K ≤ (d.K + d.BK - 1) / d.BK * d.BK := by rw [Nat.mul_comm]; omega
      exact Nat.min_eq_right this
    rw [hmin] at he
    refine ⟨hf, he.trans ?_⟩
    have hT16 : (((d.K + d.BK - 1) / d.BK : Nat) : ℝ) ≤ 16 := by exact_mod_cast hTrips
    have hT0 : (0 : ℝ) ≤ (((d.K + d.BK - 1) / d.BK : Nat) : ℝ) := Nat.cast_nonneg _
    have hBKR : (d.BK : ℝ) ≤ 128 := by exact_mod_cast hd.bk_le
    obtain ⟨hs1, -⟩ := sig_bounds hd
    have hσ0 : 0 ≤ sig d := hs1.trans' (by positivity)
    have hMg0 := kqMagP_nonneg d init (pos d r bh a) (pos d r bh b) (hvOf d bh) d.K
    have hc : ((d.BK : ℝ) + 2) / 2 ^ 22 + 5 / 2 ^ 24 + 4 / 2 ^ 24 * (((d.K + d.BK - 1) / d.BK : Nat) : ℝ) ≤
        theta0 := by
      unfold theta0
      have : ((d.BK : ℝ) + 2) / 2 ^ 22 ≤ 130 / 2 ^ 22 := by
        apply div_le_div_of_nonneg_right (by linarith) (by positivity)
      have h4 : 4 / 2 ^ 24 * (((d.K + d.BK - 1) / d.BK : Nat) : ℝ) ≤ 4 / 2 ^ 24 * 16 := by gcongr
      norm_num at this h4 ⊢
      linarith
    have hc2 : (((d.K + d.BK - 1) / d.BK : Nat) : ℝ) / 2 ^ 114 ≤ 1 / 2 ^ 110 := by
      rw [div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith
    have := mul_le_mul_of_nonneg_right hc hMg0
    rw [kqMag_eq]
    exact mul_le_mul_of_nonneg_left (by linarith) hσ0

end Claims.ChunkBwdDvLocal
