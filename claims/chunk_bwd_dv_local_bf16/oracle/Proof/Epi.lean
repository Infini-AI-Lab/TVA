import Claim
import TVA.Proof
import Proof.Setup
import Proof.Spec
import Proof.NumGate
import Proof.NumEntry
import Proof.NumDot
import Proof.NumCvt
import Proof.ValsLane
import Proof.Prologue

/-!
# The epilogue of the matrix `A`

Lines 137–140: the gate `exp2(g[j] - g[p])` (with a gate), the causal mask, the rounding to
`bfloat16`. `safe_epiK` turns the `K` loop's accumulator (`KFinal`) and the gate block (`GOk`)
into the rounded matrix (`AOk`).
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- Casting a `float32` `toRat` bound to the reals, in the form the entry lemmas use. -/
theorem epi_mul_err {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toReal * y.toReal| ≤ 2 ^ 102) :
    (FloatV.mul x y).isFinite = true ∧
      |(FloatV.mul x y).toReal - x.toReal * y.toReal| ≤
        1 / 2 ^ 24 * |x.toReal * y.toReal| + 1 / 2 ^ 150 := by
  have hf := Format.binary32_regular
  have hmaxQ : |x.toRat * y.toRat| ≤ Format.binary32.maxVal := by
    have h : ((|x.toRat * y.toRat| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [Format.binary32_maxVal_real]
      push_cast
      unfold FloatV.toReal at hmax
      refine hmax.trans ?_
      norm_num
    exact_mod_cast h
  obtain ⟨hfin, herr⟩ := FloatV.mul_err hf hx hy hmaxQ
  refine ⟨hfin, ?_⟩
  have h := (by exact_mod_cast herr :
    |((FloatV.mul x y).toRat : ℝ) - (x.toRat : ℝ) * (y.toRat : ℝ)| ≤
      |(x.toRat : ℝ) * (y.toRat : ℝ)| * ((Format.binary32.u : ℚ) : ℝ) +
        ((Format.binary32.eta : ℚ) : ℝ))
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h
  have e1 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have e2 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  rw [e1, e2] at h
  unfold FloatV.toReal
  linarith

set_option maxHeartbeats 1600000 in
/-- **One entry of the matrix, with a gate**: the accumulator `Ahat`, the exp2 of the `float32`
difference of the gate values, their product and its rounding to `bfloat16` give `EntryOk`. -/
theorem epi_entry_gate {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hV : ValsOk d init) (hT : d.varlen = true → TablesOk d) {r bh : Nat} (hr : r < nChunks d)
    (hbh : bh < nRows d) {a b : Fin d.BT} (hbe : en d r bh b = true) (hab : a ≤ b)
    (hG : d.useG = true) {Ahat gx gy s w Pm : F32} {x : BF16}
    (hAf : Ahat.isFinite = true)
    (hAh : |Ahat.toReal - sig d * kq d init (pos d r bh a) (pos d r bh b) (hvOf d bh)| ≤
      sig d * (theta0 * kqMag d init (pos d r bh a) (pos d r bh b) (hvOf d bh) + 1 / 2 ^ 110))
    (hgx : gx.isFinite = true)
    (hgxv : gx.toRat = valF init d.g (gIdx d (pos d r bh b) (hvOf d bh)))
    (hgy : gy.isFinite = true)
    (hgyv : gy.toRat = valF init d.g (gIdx d (pos d r bh a) (hvOf d bh)))
    (hs : FloatV.NaNEq (FloatV.sub gx gy) s) (hw : P.math .exp2 .binary32 s w)
    (hm : FloatV.NaNEq (FloatV.mul Ahat w) Pm) (hc : FloatV.NaNEq (FloatV.cvt Pm : BF16) x) :
    EntryOk d init (pos d r bh a) (pos d r bh b) (hvOf d bh) x := by
  set p := pos d r bh a with hp
  set j := pos d r bh b with hj
  set hv := hvOf d bh with hhv
  obtain ⟨hup, hMg, hroom⟩ := vals_pair hd hT hV hr hbh a.isLt b.isLt hab hbe (hvOf_lt hd bh)
  have hup := hup hG
  obtain ⟨hσ1, hσ2⟩ := sig_bounds hd
  have hσ0 : 0 < sig d := lt_of_lt_of_le (by positivity) hσ1
  obtain ⟨hMg0, hS⟩ := kq_le_kqMag d init p j hv
  -- the gate
  have hdg : dg d init p j hv = (gx.toRat : ℝ) - gy.toRat := by
    unfold dg; rw [hgxv, hgyv]
  have hup' : (gx.toRat : ℝ) - gy.toRat ≤ 64 := by rw [← hdg]; exact hup
  obtain ⟨hwfin, hwerr⟩ := gate_exp2 hP hgx hgy hup' hs hw
  rw [← hdg] at hwerr
  have hwt : wt d init p j hv = (2 : ℝ) ^ dg d init p j hv := by simp [wt, hG]
  have hwE : wErr d init p j hv = wErrR (dg d init p j hv) := by simp [wErr, wErrR, hG]
  have hW0 : 0 < wt d init p j hv := by rw [hwt]; exact Real.rpow_pos_of_pos (by norm_num) _
  have hW64 : wt d init p j hv ≤ 2 ^ 64 := by
    rw [hwt]
    calc (2 : ℝ) ^ dg d init p j hv ≤ (2 : ℝ) ^ (64 : ℝ) :=
          Real.rpow_le_rpow_of_exponent_le (by norm_num) hup
      _ = 2 ^ 64 := by norm_num
  have hEw0 : 0 ≤ wErr d init p j hv := by
    rw [hwE]; exact le_trans (by positivity) (wErrR_ge _)
  have hEw : wErr d init p j hv ≤ wt d init p j hv * (1 / 2 ^ 16) + gam := by
    rw [hwE, hwt]; unfold gam; exact wErrR_le hup
  have hwv : |w.toReal - wt d init p j hv| ≤ wErr d init p j hv := by
    rw [hwE, hwt]; exact hwerr
  have hroom' : sig d * kqMag d init p j hv * (1 + wt d init p j hv + wErr d init p j hv) ≤ 2 ^ 100 := by
    have := hroom; unfold wUp at this; linarith
  have hprod := prod_room hσ2 hσ0 hMg0 hW0 hW64 hEw0 hEw hAh hS hwv hroom'
  -- the product and its rounding
  obtain ⟨hmfin, hmerr⟩ := epi_mul_err hAf hwfin hprod
  have hnan : (FloatV.mul Ahat w).isNaN = false := ((FloatV.isFinite_iff).mp hmfin).1
  have hPm : Pm = FloatV.mul Ahat w := hm.eq hnan
  subst hPm
  have hPmax : |(FloatV.mul Ahat w).toReal| ≤ 2 ^ 103 := by
    have h1 := abs_sub_abs_le_abs_sub (FloatV.mul Ahat w).toReal (Ahat.toReal * w.toReal)
    have h2 : 1 / 2 ^ 24 * |Ahat.toReal * w.toReal| ≤ 1 / 2 ^ 24 * 2 ^ 102 :=
      mul_le_mul_of_nonneg_left hprod (by norm_num)
    norm_num at h2 ⊢
    linarith
  obtain ⟨hxfin0, hxerr⟩ := cvt_bf16_err hmfin (hPmax.trans (by norm_num))
  have hnanx : (FloatV.cvt (FloatV.mul Ahat w) : BF16).isNaN = false :=
    ((FloatV.isFinite_iff).mp hxfin0).1
  have hxe : x = FloatV.cvt (FloatV.mul Ahat w) := hc.eq hnanx
  subst hxe
  -- the entry
  obtain ⟨hE1, hPdev⟩ := entry_err hσ1 hσ2 hMg0 hS hW0 hEw0 hEw hAh hwv hmerr hxerr
  obtain ⟨-, hmag⟩ := entry_mag hσ0 hMg0 hS hW0 hPdev hxerr
  refine ⟨hxfin0, ?_, ?_⟩
  · have : Aex d init p j hv = sig d * kq d init p j hv * wt d init p j hv := rfl
    have h2 : aMag d init p j hv = aM (sig d) (kqMag d init p j hv) (wt d init p j hv) := rfl
    rw [this, h2]
    exact hE1
  · have h1 := Device.liftMag_le .bf16 (FloatV.cvt (FloatV.mul Ahat w) : BF16).toRat
    have h2 : aMag d init p j hv = aM (sig d) (kqMag d init p j hv) (wt d init p j hv) := rfl
    have h3 : ((Device.liftMag .bf16 (FloatV.cvt (FloatV.mul Ahat w) : BF16).toRat : ℚ) : ℝ) ≤
        |(FloatV.cvt (FloatV.mul Ahat w) : BF16).toReal| + 1 / 2 ^ 126 := by
      have := (Rat.cast_le (K := ℝ)).mpr h1
      push_cast at this
      have e : (Device.liftFloor .bf16 : ℚ) = 1 / 2 ^ 126 := by
        rw [Device.liftFloor_bf16]; norm_num
      rw [e] at this
      push_cast at this
      unfold FloatV.toReal
      linarith
    rw [h2]
    linarith

set_option maxHeartbeats 1600000 in
/-- **One entry of the matrix, without a gate**: the accumulator's rounding to `bfloat16`. -/
theorem epi_entry_nogate {d : Inst} (hd : d.ok) {init : Mem}
    (hV : ValsOk d init) (hT : d.varlen = true → TablesOk d) {r bh : Nat} (hr : r < nChunks d)
    (hbh : bh < nRows d) {a b : Fin d.BT} (hbe : en d r bh b = true) (hab : a ≤ b)
    (hG : d.useG = false) {Ahat : F32} {x : BF16}
    (hAf : Ahat.isFinite = true)
    (hAh : |Ahat.toReal - sig d * kq d init (pos d r bh a) (pos d r bh b) (hvOf d bh)| ≤
      sig d * (theta0 * kqMag d init (pos d r bh a) (pos d r bh b) (hvOf d bh) + 1 / 2 ^ 110))
    (hc : FloatV.NaNEq (FloatV.cvt Ahat : BF16) x) :
    EntryOk d init (pos d r bh a) (pos d r bh b) (hvOf d bh) x := by
  set p := pos d r bh a with hp
  set j := pos d r bh b with hj
  set hv := hvOf d bh with hhv
  obtain ⟨-, hMg, hroom⟩ := vals_pair hd hT hV hr hbh a.isLt b.isLt hab hbe (hvOf_lt hd bh)
  obtain ⟨hσ1, hσ2⟩ := sig_bounds hd
  have hσ0 : 0 < sig d := lt_of_lt_of_le (by positivity) hσ1
  obtain ⟨hMg0, hS⟩ := kq_le_kqMag d init p j hv
  have hwt : wt d init p j hv = 1 := by simp [wt, hG]
  have hwE : wErr d init p j hv = 0 := by simp [wErr, hG]
  have hEw : (0 : ℝ) ≤ 1 * (1 / 2 ^ 16) + gam := by unfold gam; positivity
  have hwv : |(1 : ℝ) - 1| ≤ 0 := by norm_num
  have hroom' : sig d * kqMag d init p j hv * (1 + 1 + 0) ≤ 2 ^ 100 := by
    have := hroom; unfold wUp at this; rw [hwt, hwE] at this; linarith
  have hprod := prod_room (W := 1) (Ew := 0) (wv := 1) hσ2 hσ0 hMg0 (by norm_num) (by norm_num)
    le_rfl hEw hAh hS (by simp) hroom'
  have hAmax : |Ahat.toReal| ≤ 2 ^ 103 := by
    rw [mul_one] at hprod
    refine hprod.trans ?_
    norm_num
  obtain ⟨hxfin0, hxerr⟩ := cvt_bf16_err hAf (hAmax.trans (by norm_num))
  have hnanx : (FloatV.cvt Ahat : BF16).isNaN = false := ((FloatV.isFinite_iff).mp hxfin0).1
  have hxe : x = FloatV.cvt Ahat := hc.eq hnanx
  subst hxe
  have hP : |Ahat.toReal - Ahat.toReal * 1| ≤ 1 / 2 ^ 24 * |Ahat.toReal * 1| + 1 / 2 ^ 150 := by
    rw [mul_one, sub_self, abs_zero]; positivity
  obtain ⟨hE1, hPdev⟩ := entry_err (Ew := 0) (wv := 1) hσ1 hσ2 hMg0 hS (by norm_num) le_rfl hEw hAh
    (by simp) hP hxerr
  obtain ⟨-, hmag⟩ := entry_mag hσ0 hMg0 hS (by norm_num : (0 : ℝ) < 1) hPdev hxerr
  refine ⟨hxfin0, ?_, ?_⟩
  · have e1 : Aex d init p j hv = sig d * kq d init p j hv * wt d init p j hv := rfl
    have e2 : aMag d init p j hv = aM (sig d) (kqMag d init p j hv) (wt d init p j hv) := rfl
    rw [e1, e2, hwt]
    exact hE1
  · have h1 := Device.liftMag_le .bf16 (FloatV.cvt Ahat : BF16).toRat
    have e2 : aMag d init p j hv = aM (sig d) (kqMag d init p j hv) (wt d init p j hv) := rfl
    have h3 : ((Device.liftMag .bf16 (FloatV.cvt Ahat : BF16).toRat : ℚ) : ℝ) ≤
        |(FloatV.cvt Ahat : BF16).toReal| + 1 / 2 ^ 126 := by
      have := (Rat.cast_le (K := ℝ)).mpr h1
      push_cast at this
      have e : (Device.liftFloor .bf16 : ℚ) = 1 / 2 ^ 126 := by
        rw [Device.liftFloor_bf16]; norm_num
      rw [e] at this
      push_cast at this
      unfold FloatV.toReal
      linarith
    rw [e2, hwt]
    linarith

/-- **The mask `m_A`**: lane `(a, b)` is set exactly when both lanes are enabled and `a ≤ b`. -/
theorem epi_mask {d : Inst} {r bh : Nat} {o_t : Tensor [d.BT] I64} {m_t : Tensor [d.BT] Bool}
    (hot : ∀ l : Fin d.BT, (o_t.get1 l).toInt = ((chkOf d r bh * d.BT + l : Nat) : Int))
    (hmt : ∀ l : Fin d.BT, m_t.get1 l = en d r bh l) (a b : Fin d.BT) :
    ((((o_t.col d.BT).le (o_t.row d.BT)).and ((m_t.col d.BT).and (m_t.row d.BT))).get2 a b = true) ↔
      (en d r bh a = true ∧ en d r bh b = true ∧ a ≤ b) := by
  show (IntV.le (o_t.get1 a) (o_t.get1 b) && (m_t.get1 a && m_t.get1 b)) = true ↔ _
  rw [IntV.le_eq, hot a, hot b, hmt a, hmt b]
  simp only [Bool.and_eq_true, decide_eq_true_eq, Nat.cast_le]
  constructor
  · rintro ⟨h1, h2, h3⟩
    exact ⟨h2, h3, Fin.le_def.mpr (by nlinarith [a.isLt])⟩
  · rintro ⟨h2, h3, h1⟩
    exact ⟨by have := Fin.le_def.mp h1; omega, h2, h3⟩

/-- Lane `(a, b)` of `tl.where`. -/
theorem where_get2 {m n : Nat} {α : Type} (c : Tensor [m, n] Bool) (x y : Tensor [m, n] α)
    (a : Fin m) (b : Fin n) :
    (tl.where c x y).get2 a b = if c.get2 a b then x.get2 a b else y.get2 a b := rfl

/-- The rounding of `+0` to `bfloat16` is `+0`. -/
theorem epi_cvt_zero {x : BF16} (h : FloatV.NaNEq (FloatV.cvtWith .rn (FloatV.ofPat 0 : F32) : BF16) x) :
    x = FloatV.ofPat 0 := by
  rw [FloatV.cvtWith_ofPat_zero Format.binary32_regular] at h
  exact h.eq (FloatV.isNaN_ofPat_zero Format.bfloat16_regular)

/-- **The epilogue is safe and gives `AOk`.** -/
theorem safe_epiK {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    {o_t : Tensor [d.BT] I64} {m_t : Tensor [d.BT] Bool}
    (hot : ∀ l : Fin d.BT, (o_t.get1 l).toInt = ((chkOf d r bh * d.BT + l : Nat) : Int))
    (hmt : ∀ l : Fin d.BT, m_t.get1 l = en d r bh l)
    {b_g : Tensor [d.BT] F32} {b_A : Tensor [d.BT, d.BT] F32}
    (hG : GOk d init r bh b_g) (hK : KFinal d init r bh b_A)
    {α : Type} {R W : Set Int} {Q : α → Mem → Prop} {m : Mem}
    {cont : Tensor [d.BT, d.BT] BF16 → Prog α}
    (hcont : ∀ A, AOk d init r bh A → Safe ((launch d).env P) R W Q m (cont A)) :
    Safe ((launch d).env P) R W Q m (epiK d o_t m_t b_g b_A cont) := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  have hV : ValsOk d init := hpre.2.2.2.2.2.2
  have hcvt : ∀ x : F32, (FloatV.cvtWith .rn x : BF16) = FloatV.cvt x := fun _ => rfl
  unfold epiK
  by_cases hGu : d.useG = true
  · simp only [hGu, ite_true]
    refine safe_sub fun s hs => ?_
    refine safe_math (fn := MathFn.exp2) (fun i => hP.mathTotal _ _ _) fun w hw => ?_
    refine safe_mul fun m' hm' => ?_
    refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun A hA => ?_
    refine hcont A fun a b => ⟨?_, ?_⟩
    · rintro ⟨hae, hbe, hab⟩
      obtain ⟨hAf, hAh⟩ := hK a b hae hbe hab
      obtain ⟨hgy, hgyv⟩ := hG hGu a hae
      obtain ⟨hgx, hgxv⟩ := hG hGu b hbe
      have hmask := (epi_mask hot hmt a b).mpr ⟨hae, hbe, hab⟩
      have hA' : FloatV.NaNEq (FloatV.cvtWith .rn ((tl.where _ m' (bc zf32)).get2 a b)) (A.get2 a b) :=
        hA (a, b, ())
      simp only [where_get2, hmask, ↓reduceIte, hcvt] at hA'
      exact epi_entry_gate hP hd hV hT hr hbh hbe hab hGu hAf hAh hgx hgxv hgy hgyv (hs (a, b, ()))
        (hw (a, b, ())) (hm' (a, b, ())) hA'
    · intro hn
      have hmask : (((o_t.col d.BT).le (o_t.row d.BT)).and
          ((m_t.col d.BT).and (m_t.row d.BT))).get2 a b = false := by
        by_contra hcon
        exact hn ((epi_mask hot hmt a b).mp (by simpa using hcon))
      have hA' : FloatV.NaNEq (FloatV.cvtWith .rn ((tl.where _ m' (bc zf32)).get2 a b)) (A.get2 a b) :=
        hA (a, b, ())
      simp only [where_get2, hmask, Bool.false_eq_true, ↓reduceIte] at hA'
      exact epi_cvt_zero (by rw [← zf32_eq]; exact hA')
  · have hGu' : d.useG = false := by simpa using hGu
    simp only [hGu', Bool.false_eq_true, ite_false, Prog.pure_eq]
    refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun A hA => ?_
    refine hcont A fun a b => ⟨?_, ?_⟩
    · rintro ⟨hae, hbe, hab⟩
      obtain ⟨hAf, hAh⟩ := hK a b hae hbe hab
      have hmask := (epi_mask hot hmt a b).mpr ⟨hae, hbe, hab⟩
      have hA' : FloatV.NaNEq (FloatV.cvtWith .rn ((tl.where _ b_A (bc zf32)).get2 a b)) (A.get2 a b) :=
        hA (a, b, ())
      simp only [where_get2, hmask, ↓reduceIte, hcvt] at hA'
      exact epi_entry_nogate hd hV hT hr hbh hbe hab hGu' hAf hAh hA'
    · intro hn
      have hmask : (((o_t.col d.BT).le (o_t.row d.BT)).and
          ((m_t.col d.BT).and (m_t.row d.BT))).get2 a b = false := by
        by_contra hcon
        exact hn ((epi_mask hot hmt a b).mp (by simpa using hcon))
      have hA' : FloatV.NaNEq (FloatV.cvtWith .rn ((tl.where _ b_A (bc zf32)).get2 a b)) (A.get2 a b) :=
        hA (a, b, ())
      simp only [where_get2, hmask, Bool.false_eq_true, ↓reduceIte] at hA'
      exact epi_cvt_zero (by rw [← zf32_eq]; exact hA')

end Claims.ChunkBwdDvLocal
