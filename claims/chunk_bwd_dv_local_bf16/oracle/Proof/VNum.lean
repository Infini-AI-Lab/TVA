import Claim
import TVA.Proof
import Proof.Setup
import Proof.Spec
import Proof.ValsLane
import Proof.NumEntry
import Proof.NumFinal
import Proof.NumDot
import Proof.KStep
import Proof.NumCvt
import Proof.LoadLanes

/-!
# One stored element of `dv` in the reals

`vlane` is the numeric core of the `V` loop. Given the rounded matrix `A` (`AOk`), the loaded
tile of `do` (`DoVal`) and any result of the `bfloat16` dot, the `bfloat16` rounding of the dot's
element `(a, j)` is finite and within `tol` of `ref` for the token `pos a` and the column
`t·BV + j`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- What `b_do` holds after trip `t`'s load: at every lane, the exact `do` element when the lane's
token is enabled and its column below `V`, and `+0` otherwise. -/
def DoVal (d : Inst) (init : Mem) (r bh t : Nat) (v : Tensor [d.BT, d.BV] BF16) : Prop :=
  ∀ (k : Fin d.BT) (j : Fin d.BV), (v.get2 k j).isFinite = true ∧
    (v.get2 k j).toRat = if en d r bh k = true ∧ t * d.BV + j.val < d.V then
      valB init d.dO (oIdx d (pos d r bh k) (hvOf d bh) (t * d.BV + j.val)) else 0

/-! ## Facts about the claim's quantities -/

/-- The gate factor is nonnegative. -/
theorem wt_nonneg (d : Inst) (m : Mem) (p j hv : Nat) : 0 ≤ wt d m p j hv := by
  unfold wt
  split
  · exact Real.rpow_nonneg (by norm_num) _
  · norm_num

/-- `aMag` in terms of `aM`. -/
theorem aMag_eq (d : Inst) (m : Mem) (p j hv : Nat) :
    aMag d m p j hv = aM (sig d) (kqMag d m p j hv) (wt d m p j hv) := rfl

/-- `|Aex| ≤ aMag`, and `aMag ≥ 0`. -/
theorem Aex_le_aMag {d : Inst} (hd : d.ok) (m : Mem) (p j hv : Nat) :
    0 ≤ aMag d m p j hv ∧ |Aex d m p j hv| ≤ aMag d m p j hv := by
  have hσ : 0 < sig d := (sig_bounds hd).1.trans_lt' (by positivity)
  obtain ⟨hM0, hS⟩ := kq_le_kqMag d m p j hv
  have hW := wt_nonneg d m p j hv
  rw [aMag_eq]
  refine ⟨aM_nonneg hσ.le hM0 hW, ?_⟩
  have h1 : |Aex d m p j hv| ≤ sig d * kqMag d m p j hv * wt d m p j hv := by
    unfold Aex
    rw [abs_mul, abs_mul, abs_of_pos hσ, abs_of_nonneg hW]
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hS hσ.le) hW
  exact h1.trans (sigma_mag_le_aM hσ.le hM0 hW)

/-! ## The claim's sums over lanes -/

/-- `ref` is a sum over the lanes `k ≥ a` that are enabled. -/
theorem ref_lanes {d : Inst} (hd : d.ok) (m : Mem) (r bh : Nat) (a : Fin d.BT) (e : Nat) :
    ref d m (nOf d r bh) (pos d r bh a) (hvOf d bh) e =
      ∑ k : Fin d.BT, if (a ≤ k ∧ en d r bh k = true) then
        Aex d m (pos d r bh a) (pos d r bh k) (hvOf d bh) * dov d m (pos d r bh k) (hvOf d bh) e
        else 0 :=
  sum_lanes hd r bh a (fun j => Aex d m (pos d r bh a) j (hvOf d bh) * dov d m j (hvOf d bh) e)

/-- `tol` is a sum over the lanes `k ≥ a` that are enabled. -/
theorem tol_lanes {d : Inst} (hd : d.ok) (m : Mem) (r bh : Nat) (a : Fin d.BT) (e : Nat) :
    tol d m (nOf d r bh) (pos d r bh a) (hvOf d bh) e =
      1 / 2 ^ 7 * ∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then
        |Aex d m (pos d r bh a) (pos d r bh k) (hvOf d bh)| * |dov d m (pos d r bh k) (hvOf d bh) e|
        else 0) +
      1 / 2 ^ 13 * ∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then
        aMag d m (pos d r bh a) (pos d r bh k) (hvOf d bh) * doMag d m (pos d r bh k) (hvOf d bh) e
        else 0) + 1 / 2 ^ 100 := by
  unfold tol
  simp only []
  rw [sum_lanes hd r bh a (fun j => |Aex d m (pos d r bh a) j (hvOf d bh)| * |dov d m j (hvOf d bh) e|),
    sum_lanes hd r bh a (fun j => aMag d m (pos d r bh a) j (hvOf d bh) * doMag d m j (hvOf d bh) e)]

/-- A `bfloat16` value of magnitude at most `B ≥ 2^-126` has lifted magnitude at most `B`. -/
theorem liftMag_bf16_le {q B : ℚ} (hq : |q| ≤ B) (hB : (1 : ℚ) / 2 ^ 126 ≤ B) :
    Device.liftMag .bf16 q ≤ B := by
  unfold Device.liftMag
  split_ifs
  · exact (by positivity : (0 : ℚ) ≤ 1 / 2 ^ 126).trans hB
  · exact max_le hq (by simpa [Device.liftFloor] using hB)

/-- The gate error is nonnegative, so `wt ≤ wUp`. -/
theorem wt_le_wUp (d : Inst) (m : Mem) (p j hv : Nat) : wt d m p j hv ≤ wUp d m p j hv := by
  unfold wUp wt wErr
  by_cases hG : d.useG = true
  · simp only [hG, ite_true]
    have h2 := wErrR_ge (dg d m p j hv)
    unfold wErrR at h2
    linarith
  · simp [hG]

set_option maxHeartbeats 4000000 in
/-- **One stored element**: the `bfloat16` rounding of the second dot's element `(a, j)` is finite
and within `tol` of `ref`. -/
theorem vlane {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    {A : Tensor [d.BT, d.BT] BF16} (hA : AOk d init r bh A) {t : Nat}
    {bdo : Tensor [d.BT, d.BV] BF16} (hbdo : DoVal d init r bh t bdo)
    {X : Tensor [d.BT, d.BV] F32} (hX : P.dot cfgBF16 A bdo (tl.zeros [d.BT, d.BV] .f32) X)
    {a : Fin d.BT} (hae : en d r bh a = true) {j : Fin d.BV} (hjV : t * d.BV + j.val < d.V)
    {x : BF16} (hx : FloatV.NaNEq (FloatV.cvt (X.get2 a j) : BF16) x) :
    x.isFinite = true ∧
      |x.toReal - ref d init (nOf d r bh) (pos d r bh a) (hvOf d bh) (t * d.BV + j.val)| ≤
        tol d init (nOf d r bh) (pos d r bh a) (hvOf d bh) (t * d.BV + j.val) := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  have hV : ValsOk d init := hpre.2.2.2.2.2.2
  set p := pos d r bh a with hp
  set hv := hvOf d bh with hhv
  set e := t * d.BV + j.val with he
  -- the lane data
  have hAfin : ∀ k : Fin d.BT, (A.get2 a k).isFinite = true := by
    intro k
    by_cases hk : a ≤ k ∧ en d r bh k = true
    · exact ((hA a k).1 ⟨hae, hk.2, hk.1⟩).1
    · rw [(hA a k).2 (fun h => hk ⟨h.2.2, h.2.1⟩)]; exact (FloatV.isFinite_ofPat_zero Format.bfloat16_regular)
  have hDo : ∀ k : Fin d.BT, en d r bh k = true →
      (bdo.get2 k j).toRat = valB init d.dO (oIdx d (pos d r bh k) hv e) := by
    intro k hk
    have := (hbdo k j).2
    rwa [ite_eq_left ⟨hk, hjV⟩] at this
  have hDo0 : ∀ k : Fin d.BT, ¬ en d r bh k = true → (bdo.get2 k j).toRat = 0 := by
    intro k hk
    have := (hbdo k j).2
    rwa [ite_eq_right (fun h => hk h.1)] at this
  have hdoMag : ∀ k : Fin d.BT, ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) =
      if en d r bh k = true then doMag d init (pos d r bh k) hv e else 0 := by
    intro k
    by_cases hk : en d r bh k = true
    · rw [ite_eq_left hk, hDo k hk]; rfl
    · rw [ite_eq_right hk, hDo0 k hk]; simp
  have hDoBound : ∀ k : Fin d.BT, ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) ≤ 2 ^ 100 := by
    intro k
    by_cases hk : en d r bh k = true
    · obtain ⟨-, hdo, -⟩ := vals_tok hd hT hV hr hbh hk
      obtain ⟨-, hb⟩ := hdo hv e (hvOf_lt hd bh) (by omega)
      have h1 : Device.liftMag .bf16 (bdo.get2 k j).toRat ≤ 2 ^ 100 :=
        liftMag_bf16_le (by rw [hDo k hk]; exact_mod_cast hb) (by norm_num)
      exact_mod_cast h1
    · rw [hDo0 k hk]; simp
  -- the per-lane quantities of `final_err`
  have hpairA : ∀ k : Fin d.BT, (a ≤ k ∧ en d r bh k = true) →
      EntryOk d init p (pos d r bh k) hv (A.get2 a k) :=
    fun k hk => (hA a k).1 ⟨hae, hk.2, hk.1⟩
  have hnpairA : ∀ k : Fin d.BT, ¬ (a ≤ k ∧ en d r bh k = true) → A.get2 a k = FloatV.ofPat 0 :=
    fun k hk => (hA a k).2 (fun h => hk ⟨h.2.2, h.2.1⟩)
  have hzeroA : ∀ k : Fin d.BT, ¬ (a ≤ k ∧ en d r bh k = true) →
      ((A.get2 a k).toRat : ℝ) = 0 ∧ ((Device.liftMag .bf16 (A.get2 a k).toRat : ℚ) : ℝ) = 0 := by
    intro k hk
    rw [hnpairA k hk, FloatV.toRat_ofPat_zero]
    simp
  -- room: the magnitude sums
  have hσ : 0 < sig d := (sig_bounds hd).1.trans_lt' (by positivity)
  have hσ2 : sig d ≤ 2 ^ 32 := (sig_bounds hd).2
  have hSM : ∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then
        aMag d init p (pos d r bh k) hv else 0) * ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ)
        ≤ 2 ^ 120 + 2 ^ 109 := by
    have hF := sum_lanes hd r bh a (fun j' => sig d * kqMag d init p j' hv * wUp d init p j' hv *
      doMag d init j' hv e)
    have hout := vals_out hd hT hV hr hbh hae (hvOf_lt hd bh) (hv := hv) (e := e) (by omega)
    have hterm : ∀ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then aMag d init p (pos d r bh k) hv
        else 0) * ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) ≤
        (if (a ≤ k ∧ en d r bh k = true) then sig d * kqMag d init p (pos d r bh k) hv *
          wUp d init p (pos d r bh k) hv * doMag d init (pos d r bh k) hv e else 0) + 2 ^ 103 := by
      intro k
      by_cases hk : a ≤ k ∧ en d r bh k = true
      · rw [ite_eq_left hk, ite_eq_left hk, hdoMag k, ite_eq_left hk.2]
        have hk' : a.val ≤ k.val := hk.1
        obtain ⟨hup, hMg, -⟩ := vals_pair hd hT hV hr hbh a.isLt k.isLt hk' hk.2 (hvOf_lt hd bh)
        have hsM := sig_kqMag_le hd hT hV hr hbh a.isLt k.isLt hk' hk.2 (hvOf_lt hd bh)
        have hM0 := (kq_le_kqMag d init p (pos d r bh k) hv).1
        have hW0 := wt_nonneg d init p (pos d r bh k) hv
        have hW64 : wt d init p (pos d r bh k) hv ≤ 2 ^ 64 := by
          unfold wt
          by_cases hG : d.useG = true
          · simp only [hG, ite_true]
            calc (2 : ℝ) ^ dg d init p (pos d r bh k) hv ≤ (2 : ℝ) ^ (64 : ℝ) :=
                  Real.rpow_le_rpow_of_exponent_le (by norm_num) (hup hG)
              _ = 2 ^ 64 := by norm_num
          · simp only [hG, ite_false, Bool.false_eq_true]; norm_num
        have hup' := aM_le_up hσ2 hW0 hW64 hsM
        have hmw := wt_le_wUp d init p (pos d r bh k) hv
        have hD0 : 0 ≤ doMag d init (pos d r bh k) hv e := by
          unfold doMag
          exact_mod_cast Device.liftMag_nonneg _ _
        have hDb : doMag d init (pos d r bh k) hv e ≤ 2 ^ 100 := by
          have := hDoBound k
          rwa [hdoMag k, ite_eq_left hk.2] at this
        have h1 : aMag d init p (pos d r bh k) hv ≤ sig d * kqMag d init p (pos d r bh k) hv *
            wUp d init p (pos d r bh k) hv + 8 := by
          rw [aMag_eq]
          have := mul_le_mul_of_nonneg_left hmw (mul_nonneg hσ.le hM0)
          linarith
        nlinarith [mul_le_mul_of_nonneg_right h1 hD0]
      · rw [ite_eq_right hk, ite_eq_right hk]
        simp only [zero_mul, zero_add]
        positivity
    calc _ ≤ ∑ k : Fin d.BT, ((if (a ≤ k ∧ en d r bh k = true) then sig d * kqMag d init p (pos d r bh k) hv *
          wUp d init p (pos d r bh k) hv * doMag d init (pos d r bh k) hv e else 0) + 2 ^ 103) :=
          Finset.sum_le_sum fun k _ => hterm k
      _ = (∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then sig d * kqMag d init p (pos d r bh k) hv *
          wUp d init p (pos d r bh k) hv * doMag d init (pos d r bh k) hv e else 0)) +
          (d.BT : ℝ) * 2 ^ 103 := by
          rw [Finset.sum_add_distrib]; simp
      _ ≤ 2 ^ 120 + 2 ^ 109 := by
          rw [← hF]
          have hBT : (d.BT : ℝ) ≤ 64 := by
            rcases hd.bt_ok with h | h | h <;> rw [h] <;> norm_num
          have := mul_le_mul_of_nonneg_right hBT (by positivity : (0 : ℝ) ≤ 2 ^ 103)
          have e2 : (64 : ℝ) * 2 ^ 103 = 2 ^ 109 := by norm_num
          linarith [hout]
  -- the lifted magnitude of the dot
  have hLterm : ∀ k : Fin d.BT, ((Device.liftMag .bf16 (A.get2 a k).toRat : ℚ) : ℝ) *
      ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) ≤ (1 + 1 / 2 ^ 7) *
      ((if (a ≤ k ∧ en d r bh k = true) then aMag d init p (pos d r bh k) hv else 0) *
        ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ)) := by
    intro k
    have hD0 : (0 : ℝ) ≤ ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) := by
      exact_mod_cast Device.liftMag_nonneg _ _
    by_cases hk : a ≤ k ∧ en d r bh k = true
    · rw [ite_eq_left hk]
      have := (hpairA k hk).2.2
      nlinarith [mul_le_mul_of_nonneg_right this hD0]
    · rw [ite_eq_right hk, (hzeroA k hk).2]; simp
  have hLR : ∑ k : Fin d.BT, ((Device.liftMag .bf16 (A.get2 a k).toRat : ℚ) : ℝ) *
      ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) ≤ 2 ^ 121 := by
    calc _ ≤ ∑ k : Fin d.BT, (1 + 1 / 2 ^ 7) * ((if (a ≤ k ∧ en d r bh k = true) then
            aMag d init p (pos d r bh k) hv else 0) *
            ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ)) :=
          Finset.sum_le_sum fun k _ => hLterm k
      _ = (1 + 1 / 2 ^ 7) * ∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then
            aMag d init p (pos d r bh k) hv else 0) *
            ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) := by rw [Finset.mul_sum]
      _ ≤ (1 + 1 / 2 ^ 7) * (2 ^ 120 + 2 ^ 109) := mul_le_mul_of_nonneg_left hSM (by norm_num)
      _ ≤ 2 ^ 121 := by norm_num
  set Lq : ℚ := ∑ k : Fin d.BT, Device.liftMag .bf16 (A.get2 a k).toRat *
    Device.liftMag .bf16 (bdo.get2 k j).toRat with hLq
  have hLqR : (Lq : ℝ) = ∑ k : Fin d.BT, ((Device.liftMag .bf16 (A.get2 a k).toRat : ℚ) : ℝ) *
      ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) := by
    rw [hLq]; push_cast; rfl
  have hLqle : Lq ≤ 2 ^ 121 := by
    have : (Lq : ℝ) ≤ 2 ^ 121 := by rw [hLqR]; exact hLR
    exact_mod_cast this
  have hLq0 : 0 ≤ Lq := Finset.sum_nonneg fun k _ =>
    mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hBT : d.BT ≤ 64 := by rcases hd.bt_ok with h | h | h <;> omega
  have hrel1 : Device.relDot16 d.BT ≤ 1 := by
    unfold Device.relDot16
    have : ((d.BT : ℚ) + 2) ≤ 66 := by
      have : (d.BT : ℚ) ≤ 64 := by exact_mod_cast hBT
      linarith
    calc ((d.BT : ℚ) + 2) * 2 ^ (-22 : ℤ) ≤ 66 * 2 ^ (-22 : ℤ) :=
          mul_le_mul_of_nonneg_right this (by norm_num)
      _ ≤ 1 := by norm_num
  have habs1 : Device.absDot16 d.BT ≤ 1 := by
    unfold Device.absDot16
    have : ((d.BT : ℚ) + 2) ≤ 66 := by
      have : (d.BT : ℚ) ≤ 64 := by exact_mod_cast hBT
      linarith
    calc ((d.BT : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ 66 * 2 ^ (-126 : ℤ) :=
          mul_le_mul_of_nonneg_right this (by norm_num)
      _ ≤ 1 := by norm_num
  have hrel0 : 0 ≤ Device.relDot16 d.BT := by unfold Device.relDot16; positivity
  have hroom : (1 + Device.relDot16 d.BT) * Lq + Device.absDot16 d.BT ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    have : (1 + Device.relDot16 d.BT) * Lq ≤ 2 * 2 ^ 121 :=
      mul_le_mul (by linarith) hLqle hLq0 (by norm_num)
    have h3 : (2 * 2 ^ 121 + 1 : ℚ) ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hXf, hXe⟩ := dot_lane hP hX a j hAfin (fun k => (hbdo k j).1) hroom
  -- the rounding to `bfloat16`
  have hXmax : |(X.get2 a j).toReal| ≤ 2 ^ 127 := by
    have hS2 : |∑ k : Fin d.BT, (A.get2 a k).toRat * (bdo.get2 k j).toRat| ≤ Lq :=
      slab_abs_le (fun k : Fin d.BT => (A.get2 a k).toRat) (fun k => (bdo.get2 k j).toRat)
    have h1 : |(X.get2 a j).toRat| ≤ Lq + (Device.relDot16 d.BT * Lq + Device.absDot16 d.BT) := by
      have := abs_sub_abs_le_abs_sub (X.get2 a j).toRat (∑ k : Fin d.BT, (A.get2 a k).toRat * (bdo.get2 k j).toRat)
      linarith
    have h2 : (Device.relDot16 d.BT) * Lq ≤ 1 * 2 ^ 121 := mul_le_mul hrel1 hLqle hLq0 (by norm_num)
    have h3 : |(X.get2 a j).toRat| ≤ 2 ^ 124 := by
      have : (2 ^ 121 + (1 * 2 ^ 121 + 1) : ℚ) ≤ 2 ^ 124 := by norm_num
      linarith
    have : |((X.get2 a j).toRat : ℝ)| ≤ 2 ^ 124 := by exact_mod_cast h3
    unfold FloatV.toReal
    exact this.trans (by norm_num)
  obtain ⟨hxfin0, hxerr⟩ := cvt_bf16_err hXf hXmax
  have hnanx : (FloatV.cvt (X.get2 a j) : BF16).isNaN = false := ((FloatV.isFinite_iff).mp hxfin0).1
  have hxe : x = FloatV.cvt (X.get2 a j) := hx.eq hnanx
  subst hxe
  refine ⟨hxfin0, ?_⟩
  -- the sums of the claim, as lane sums
  have hdov : ∀ k : Fin d.BT, en d r bh k = true →
      ((bdo.get2 k j).toRat : ℝ) = dov d init (pos d r bh k) hv e := by
    intro k hk; rw [hDo k hk]; rfl
  have hDmag : ∀ k : Fin d.BT, en d r bh k = true →
      ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) = doMag d init (pos d r bh k) hv e := by
    intro k hk; rw [hdoMag k, ite_eq_left hk]
  rw [ref_lanes hd init r bh a e, tol_lanes hd init r bh a e]
  have hfe := final_err (Finset.univ : Finset (Fin d.BT))
    (fun k => ((A.get2 a k).toRat : ℝ))
    (fun k => if (a ≤ k ∧ en d r bh k = true) then Aex d init p (pos d r bh k) hv else 0)
    (fun k => if (a ≤ k ∧ en d r bh k = true) then aMag d init p (pos d r bh k) hv else 0)
    (fun k => ((bdo.get2 k j).toRat : ℝ))
    (fun k => ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ))
    (fun k => ((Device.liftMag .bf16 (A.get2 a k).toRat : ℚ) : ℝ))
    (v := (X.get2 a j).toReal) (o := (FloatV.cvt (X.get2 a j) : BF16).toReal)
    (ρ2 := ((d.BT : ℝ) + 2) / 2 ^ 22) (α2 := ((d.BT : ℝ) + 2) / 2 ^ 126)
    (fun k _ => by
      by_cases hk : a ≤ k ∧ en d r bh k = true
      · simp only [hk, and_self, ite_true]; exact (Aex_le_aMag hd init _ _ _).1
      · simp only [hk, ite_false]; exact le_refl _)
    (fun k _ => by
      by_cases hk : a ≤ k ∧ en d r bh k = true
      · simp only [hk, and_self, ite_true]; exact (Aex_le_aMag hd init _ _ _).2
      · simp only [hk, ite_false, abs_zero]; exact le_refl _)
    (fun k _ => by exact_mod_cast Device.liftMag_nonneg _ _)
    (fun k _ => by
      have := Device.abs_le_liftMag .bf16 (bdo.get2 k j).toRat
      have h2 : ((|(bdo.get2 k j).toRat| : ℚ) : ℝ) ≤ ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) :=
        by exact_mod_cast this
      simpa using h2)
    (fun k _ => by
      by_cases hk : a ≤ k ∧ en d r bh k = true
      · simp only [hk, and_self, ite_true]
        exact (hpairA k hk).2.1
      · simp only [hk, ite_false, abs_zero, mul_zero, add_zero]
        rw [(hzeroA k hk).1]; simp)
    (fun k _ => by
      have := Device.abs_le_liftMag .bf16 (A.get2 a k).toRat
      have h2 : ((|(A.get2 a k).toRat| : ℚ) : ℝ) ≤ ((Device.liftMag .bf16 (A.get2 a k).toRat : ℚ) : ℝ) :=
        by exact_mod_cast this
      simpa using h2)
    (fun k _ => by
      by_cases hk : a ≤ k ∧ en d r bh k = true
      · simp only [hk, and_self, ite_true]
        exact (hpairA k hk).2.2
      · simp only [hk, ite_false, mul_zero]
        rw [(hzeroA k hk).2])
    (by positivity)
    (by
      have : (d.BT : ℝ) ≤ 64 := by exact_mod_cast hBT
      rw [div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith)
    (by
      have : (d.BT : ℝ) ≤ 64 := by exact_mod_cast hBT
      rw [div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith)
    (by
      have h := (by exact_mod_cast hXe :
        |((X.get2 a j).toRat : ℝ) - ((∑ k : Fin d.BT, (A.get2 a k).toRat * (bdo.get2 k j).toRat : ℚ) : ℝ)| ≤
          ((Device.relDot16 d.BT : ℚ) : ℝ) * (Lq : ℝ) + ((Device.absDot16 d.BT : ℚ) : ℝ))
      have hρ : (((Device.relDot16 d.BT : ℚ) : ℝ)) = ((d.BT : ℝ) + 2) / 2 ^ 22 := by
        unfold Device.relDot16; push_cast; norm_num; ring
      have hα : (((Device.absDot16 d.BT : ℚ) : ℝ)) = ((d.BT : ℝ) + 2) / 2 ^ 126 := by
        unfold Device.absDot16; push_cast; norm_num; ring
      rw [hρ, hα, hLqR] at h
      push_cast at h
      exact h)
    hxerr
  have e1 : ∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then
      Aex d init p (pos d r bh k) hv * dov d init (pos d r bh k) hv e else 0) =
      ∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then Aex d init p (pos d r bh k) hv else 0) *
        ((bdo.get2 k j).toRat : ℝ) := by
    refine Finset.sum_congr rfl fun k _ => ?_
    by_cases hk : a ≤ k ∧ en d r bh k = true
    · simp only [hk, and_self, ite_true]; rw [hdov k hk.2]
    · simp [hk]
  have e2 : ∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then
      |Aex d init p (pos d r bh k) hv| * |dov d init (pos d r bh k) hv e| else 0) =
      ∑ k : Fin d.BT, |(if (a ≤ k ∧ en d r bh k = true) then Aex d init p (pos d r bh k) hv else 0)| *
        |((bdo.get2 k j).toRat : ℝ)| := by
    refine Finset.sum_congr rfl fun k _ => ?_
    by_cases hk : a ≤ k ∧ en d r bh k = true
    · simp only [hk, and_self, ite_true]; rw [hdov k hk.2]
    · simp [hk]
  have e3 : ∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then
      aMag d init p (pos d r bh k) hv * doMag d init (pos d r bh k) hv e else 0) =
      ∑ k : Fin d.BT, (if (a ≤ k ∧ en d r bh k = true) then aMag d init p (pos d r bh k) hv else 0) *
        ((Device.liftMag .bf16 (bdo.get2 k j).toRat : ℚ) : ℝ) := by
    refine Finset.sum_congr rfl fun k _ => ?_
    by_cases hk : a ≤ k ∧ en d r bh k = true
    · simp only [hk, and_self, ite_true]; rw [hDmag k hk.2]
    · simp [hk]
  rw [e1, e2, e3]
  exact hfe

end Claims.ChunkBwdDvLocal
