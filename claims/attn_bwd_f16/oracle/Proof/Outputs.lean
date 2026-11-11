import Proof.Stages
import Proof.Pair

/-!
# `attn_bwd_f16`: from the accumulators to `OutSpec`
-/

namespace Claims.AttnBwd

open TVA

/-- All rows of a block of `128` have the same number of `dk`, `dv` steps. -/
theorem stepsKV_row (d : Inst) {n0 r : Nat} (hn0m : n0 % 128 = 0) (hr : r < 128) :
    stepsKV d (n0 + r) = stepsKV d n0 := by
  unfold stepsKV
  have : (n0 + r) / 128 = n0 / 128 := by omega
  rw [this]

/-- All rows of a block of `128` have the same number of `dq` steps. -/
theorem stepsQ_row (d : Inst) {n0 r : Nat} (hn0m : n0 % 128 = 0) (hr : r < 128) :
    stepsQ d (n0 + r) = stepsQ d n0 := by
  unfold stepsQ
  have : (n0 + r) / 128 = n0 / 128 := by omega
  rw [this]

/-- The exact `dS` of a covered pair is at most `2^15` in magnitude. -/
theorem dsE_abs_le {d : Inst} {hv : HeadV} (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX)
    (hm : m < d.N_CTX) : |dsE d hv n m| ≤ 2 ^ 15 := by
  unfold dsE pE
  split_ifs with hal
  · obtain ⟨hfit1, hfit2⟩ := hok.fits n m hn hm hal
    have he0 : 0 ≤ eX d hv n m := eSub_nonneg d (aQK_nonneg d hv n m)
    have hw : 0 < (2 : ℝ) ^ xE d hv n m := Real.rpow_pos_of_pos (by norm_num) _
    have hwp : (2 : ℝ) ^ xE d hv n m ≤ pBr d hv n m := by
      unfold pBr
      have h1 : (2 : ℝ) ^ xE d hv n m ≤ 2 ^ (xE d hv n m + eX d hv n m) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
      have h2 : (0 : ℝ) ≤ 2 ^ (-22 : ℤ) := by positivity
      have h3 : (0 : ℝ) ≤ 2 ^ (-126 : ℤ) := by positivity
      nlinarith [Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 2) (xE d hv n m + eX d hv n m)]
    have hy : |yE d hv n m| ≤ yB d hv n m := by
      unfold yB eY
      have := eSub_nonneg d (s := sVD d hv n m) (x := (hv.dl m).toReal) (aVD_nonneg d hv n m)
      linarith
    rw [abs_mul, abs_of_pos hw]
    calc 2 ^ xE d hv n m * |yE d hv n m| ≤ pBr d hv n m * yB d hv n m :=
          mul_le_mul hwp hy (abs_nonneg _) (pBr_nonneg d hv n m)
      _ ≤ 2 ^ 15 := hfit2
  · simp

/-- The error bound of the `float16` `dS` of a covered pair is at most `2^17`. -/
theorem ds16Err_le {d : Inst} {hv : HeadV} (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX)
    (hm : m < d.N_CTX) : ds16Err d hv n m ≤ 2 ^ 17 := by
  unfold ds16Err
  split_ifs with hal
  · obtain ⟨hfit1, hfit2⟩ := hok.fits n m hn hm hal
    have he0 : 0 ≤ eX d hv n m := eSub_nonneg d (aQK_nonneg d hv n m)
    have hw : 0 < (2 : ℝ) ^ xE d hv n m := Real.rpow_pos_of_pos (by norm_num) _
    have hwp : (2 : ℝ) ^ xE d hv n m ≤ pBr d hv n m := by
      unfold pBr
      have h1 : (2 : ℝ) ^ xE d hv n m ≤ 2 ^ (xE d hv n m + eX d hv n m) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
      have h2 : (0 : ℝ) ≤ 2 ^ (-22 : ℤ) := by positivity
      have h3 : (0 : ℝ) ≤ 2 ^ (-126 : ℤ) := by positivity
      nlinarith [Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 2) (xE d hv n m + eX d hv n m)]
    have heY : eY d hv n m ≤ yB d hv n m := by
      unfold yB; linarith [abs_nonneg (yE d hv n m)]
    have hyB := yB_nonneg d hv n m
    have hpB := pBr_nonneg d hv n m
    have hpy : pBr d hv n m * yB d hv n m ≤ 2 ^ 15 := hfit2
    have t1 : (pBr d hv n m - 2 ^ xE d hv n m) * yB d hv n m ≤ pBr d hv n m * yB d hv n m :=
      mul_le_mul_of_nonneg_right (by linarith) hyB
    have t2 : 2 ^ xE d hv n m * eY d hv n m ≤ pBr d hv n m * yB d hv n m :=
      mul_le_mul hwp heY (eSub_nonneg d (aVD_nonneg d hv n m)) hpB
    have hu32 := u32_nonneg
    have hu32' := u32_le
    have hu16 := u16_nonneg
    have hu16' := u16_le
    have he32 := eta32_le
    have he16 := eta16_le
    have t3 : u32 * (pBr d hv n m * yB d hv n m) ≤ 1 / 2048 * 2 ^ 15 :=
      mul_le_mul hu32' hpy (mul_nonneg hpB hyB) (by norm_num)
    have t4 : dsB d hv n m ≤ 2 ^ 15 + 32 := by
      unfold dsB
      have : (1 + u32) * (pBr d hv n m * yB d hv n m) ≤ (1 + 1 / 2048) * 2 ^ 15 :=
        mul_le_mul (by linarith) hpy (mul_nonneg hpB hyB) (by norm_num)
      have : (0 : ℝ) ≤ eta32 := eta32_nonneg
      nlinarith
    have t5 : u16 * dsB d hv n m ≤ 1 / 2048 * (2 ^ 15 + 32) := by
      have hdsB0 : 0 ≤ dsB d hv n m := by
        unfold dsB; have := eta32_nonneg; positivity
      exact mul_le_mul hu16' t4 hdsB0 (by norm_num)
    unfold dsErr
    nlinarith
  · positivity

/-- A pair that is not `Allowed` carries nothing. -/
theorem notAllowed_zero {d : Inst} {hv : HeadV} {n m : Nat} (hal : ¬ Allowed d n m) :
    pE d hv n m = 0 ∧ p16Err d hv n m = 0 ∧ p16L d hv n m = 0 ∧ dsE d hv n m = 0 ∧
      ds16Err d hv n m = 0 ∧ ds16L d hv n m = 0 := by
  have h0 : lift16 0 = 0 := by unfold lift16; simp
  have hpE : pE d hv n m = 0 := by unfold pE; rw [ite_eq_right hal]
  refine ⟨hpE, ?_, ?_, ?_, ?_, ?_⟩
  · unfold p16Err; rw [ite_eq_right hal]
  · unfold p16L p16B; rw [ite_eq_right hal, h0]
  · unfold dsE; rw [hpE]; simp
  · unfold ds16Err; rw [ite_eq_right hal]
  · unfold ds16L ds16B; rw [ite_eq_right hal, h0]

/-- **`dV`**: the accumulator of `dv` after both stages, rounded to `float16`, is `OutSpec`'s value
for `dV[n0 + r, j]`. -/
theorem dv_spec (d : Inst) {init : Mem} {b n0 : Nat} (hn0m : n0 % 128 = 0) {dk dv : Tensor [128, d.HEAD_DIM] F32}
    (hinv : DkdvInv d init b n0 (Finset.Ico (lo d n0) d.N_CTX) (stepsKV d n0) dk dv)
    {x : Tensor [128, d.HEAD_DIM] F16}
    (hx : ∀ i : Idx [128, d.HEAD_DIM], (FloatV.cvtWith .rn (dv.get i)).NaNEq (x.get i))
    (r : Fin 128) (j : Fin d.HEAD_DIM) (hv : HeadV) (hH : hv.Holds d init b) (hOk : hv.Ok d) :
    ∃ v : F32, v.isFinite = true ∧ |v.toReal - dVE d hv (n0 + r) j| ≤ dVTol d hv (n0 + r) j ∧
      FloatV.NaNEq (FloatV.cvt v) (x.get2 r j) ∧
      (|dVE d hv (n0 + r) j| + dVTol d hv (n0 + r) j ≤ 65504 →
        (x.get2 r j).isFinite = true ∧
          |(x.get2 r j).toReal - dVE d hv (n0 + r) j| ≤
            dVTol d hv (n0 + r) j + (|dVE d hv (n0 + r) j| + dVTol d hv (n0 + r) j) * u16 + eta16) := by
  have hz : ∀ s < d.N_CTX, s ∉ Finset.Ico (lo d n0) d.N_CTX →
      (famDV d hv).wt ((fun i : Fin 128 => n0 + i) r) s = 0 ∧
      (famDV d hv).err ((fun i : Fin 128 => n0 + i) r) s = 0 ∧
      (famDV d hv).lb ((fun i : Fin 128 => n0 + i) r) s = 0 := by
    intro s hs hns
    have hlo : s < lo d n0 := by
      by_contra hcon
      exact hns (Finset.mem_Ico.mpr ⟨not_lt.mp hcon, hs⟩)
    have hal : ¬ Allowed d (n0 + r) s := by
      unfold lo at hlo
      split_ifs at hlo with hC
      · rintro (h | h)
        · rw [hC] at h; cases h
        · omega
      · omega
    obtain ⟨h1, h2, h3, -⟩ := notAllowed_zero (hv := hv) hal
    exact ⟨h1, h2, h3⟩
  obtain ⟨hf, hb'⟩ := AccInv.final (hinv hv hH hOk).1 r j d.N_CTX
    (fun s hs => (Finset.mem_Ico.mp hs).2) hz
  have hbd : |(dv.get2 r j).toReal - dVE d hv (n0 + r) j| ≤ dVTol d hv (n0 + r) j := by
    unfold dVTol; rw [stepsKV_row d hn0m r.isLt]; exact hb'
  exact out_of_f32 hf hbd (hx (r, j, ()))

/-- A finite `float16` is at most `65504` in magnitude. -/
theorem abs_f16_le {x : F16} (hx : x.isFinite = true) : |x.toReal| ≤ 65504 := by
  have := FloatV.abs_toReal_le_maxVal Format.binary16_regular hx
  rwa [Format.binary16_maxVal_real] at this

/-- The error bound of the `float16` `dS` is nonnegative. -/
theorem ds16Err_nonneg {d : Inst} {hv : HeadV} (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX)
    (hm : m < d.N_CTX) : 0 ≤ ds16Err d hv n m := by
  unfold ds16Err
  split_ifs with hal
  · obtain ⟨hfit1, hfit2⟩ := hok.fits n m hn hm hal
    have he0 : 0 ≤ eX d hv n m := eSub_nonneg d (aQK_nonneg d hv n m)
    have hw : 0 < (2 : ℝ) ^ xE d hv n m := Real.rpow_pos_of_pos (by norm_num) _
    have hwp : (2 : ℝ) ^ xE d hv n m ≤ pBr d hv n m := by
      unfold pBr
      have h1 : (2 : ℝ) ^ xE d hv n m ≤ 2 ^ (xE d hv n m + eX d hv n m) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
      have h2 : (0 : ℝ) ≤ 2 ^ (-22 : ℤ) := by positivity
      have h3 : (0 : ℝ) ≤ 2 ^ (-126 : ℤ) := by positivity
      nlinarith [Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 2) (xE d hv n m + eX d hv n m)]
    have hyB := yB_nonneg d hv n m
    have hpB := pBr_nonneg d hv n m
    have hu32 := u32_nonneg
    have hu16 := u16_nonneg
    have he32 := eta32_nonneg
    have he16 := eta16_nonneg
    have heY : 0 ≤ eY d hv n m := eSub_nonneg d (aVD_nonneg d hv n m)
    have hdsB0 : 0 ≤ dsB d hv n m := by unfold dsB; positivity
    unfold dsErr
    have : 0 ≤ (pBr d hv n m - 2 ^ xE d hv n m) * yB d hv n m := mul_nonneg (by linarith) hyB
    have : 0 ≤ 2 ^ xE d hv n m * eY d hv n m := mul_nonneg hw.le heY
    have : 0 ≤ u32 * (pBr d hv n m * yB d hv n m) := mul_nonneg hu32 (mul_nonneg hpB hyB)
    have : 0 ≤ u16 * dsB d hv n m := mul_nonneg hu16 hdsB0
    linarith
  · exact le_rfl

/-- **Room for the scaling by a scalar**: if every term is bounded, the exact sum and the tolerance
(`Σ err·|u| + accB t (Σ lb·lm u)`) together stay below `2^100`. -/
theorem tol_room {N' t : Nat} (hN : N' ≤ 2 ^ 14) (ht : t ≤ 1024) {wt err lb : Nat → ℝ}
    {u : Nat → F16} (hwt : ∀ s < N', |wt s| ≤ 2 ^ 15) (herr : ∀ s < N', 0 ≤ err s ∧ err s ≤ 2 ^ 17)
    (hlb : ∀ s < N', 0 ≤ lb s ∧ lb s ≤ 2 ^ 16) (hu : ∀ s < N', (u s).isFinite = true) :
    |∑ s ∈ Finset.range N', wt s * (u s).toReal| +
      (∑ s ∈ Finset.range N', err s * |(u s).toReal| +
        accB t (∑ s ∈ Finset.range N', lb s * lm (u s))) ≤ 2 ^ 100 := by
  have hN' : (N' : ℝ) ≤ 2 ^ 14 := by exact_mod_cast hN
  have h1 : |∑ s ∈ Finset.range N', wt s * (u s).toReal| ≤ N' * (2 ^ 15 * 65504) := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ s ∈ Finset.range N', |wt s * (u s).toReal| ≤ ∑ _s ∈ Finset.range N', (2 ^ 15 * 65504 : ℝ) := by
          refine Finset.sum_le_sum fun s hs => ?_
          have hs' := Finset.mem_range.mp hs
          rw [abs_mul]
          exact mul_le_mul (hwt s hs') (abs_f16_le (hu s hs')) (abs_nonneg _) (by norm_num)
      _ = N' * (2 ^ 15 * 65504) := by simp
  have h2 : ∑ s ∈ Finset.range N', err s * |(u s).toReal| ≤ N' * (2 ^ 17 * 65504) := by
    calc ∑ s ∈ Finset.range N', err s * |(u s).toReal| ≤ ∑ _s ∈ Finset.range N', (2 ^ 17 * 65504 : ℝ) := by
          refine Finset.sum_le_sum fun s hs => ?_
          have hs' := Finset.mem_range.mp hs
          exact mul_le_mul (herr s hs').2 (abs_f16_le (hu s hs')) (abs_nonneg _) (by norm_num)
      _ = N' * (2 ^ 17 * 65504) := by simp
  have hB0 : 0 ≤ ∑ s ∈ Finset.range N', lb s * lm (u s) :=
    Finset.sum_nonneg fun s hs => mul_nonneg (hlb s (Finset.mem_range.mp hs)).1 (lm_nonneg _)
  have hB : ∑ s ∈ Finset.range N', lb s * lm (u s) ≤ N' * (2 ^ 16 * 65504) := by
    calc ∑ s ∈ Finset.range N', lb s * lm (u s) ≤ ∑ _s ∈ Finset.range N', (2 ^ 16 * 65504 : ℝ) := by
          refine Finset.sum_le_sum fun s hs => ?_
          have hs' := Finset.mem_range.mp hs
          exact mul_le_mul (hlb s hs').2 (lm_le_65504 (hu s hs')) (lm_nonneg _) (by norm_num)
      _ = N' * (2 ^ 16 * 65504) := by simp
  have h3 := accB_le ht hB0
  have h1' : N' * (2 ^ 15 * 65504 : ℝ) ≤ 2 ^ 14 * (2 ^ 15 * 65504) :=
    mul_le_mul_of_nonneg_right hN' (by norm_num)
  have h2' : N' * (2 ^ 17 * 65504 : ℝ) ≤ 2 ^ 14 * (2 ^ 17 * 65504) :=
    mul_le_mul_of_nonneg_right hN' (by norm_num)
  have hB' : N' * (2 ^ 16 * 65504 : ℝ) ≤ 2 ^ 14 * (2 ^ 16 * 65504) :=
    mul_le_mul_of_nonneg_right hN' (by norm_num)
  have : (2 : ℝ) ^ 14 * (2 ^ 15 * 65504) + (2 ^ 14 * (2 ^ 17 * 65504) + (2 ^ 14 * (2 ^ 16 * 65504) + 1))
      ≤ 2 ^ 100 := by norm_num
  linarith

/-- **`dK`**: the accumulator of `dk` times `sm_scale`, rounded to `float16`, is `OutSpec`'s value
for `dK[n0 + r, j]`. -/
theorem dk_spec (d : Inst) (hd : d.ok) {init : Mem} {b n0 : Nat}
    (hn0 : n0 + 128 ≤ d.N_CTX) (hn0m : n0 % 128 = 0) {dk dv : Tensor [128, d.HEAD_DIM] F32}
    (hinv : DkdvInv d init b n0 (Finset.Ico (lo d n0) d.N_CTX) (stepsKV d n0) dk dv)
    {dkm : Tensor [128, d.HEAD_DIM] F32} {x : Tensor [128, d.HEAD_DIM] F16}
    (hm : ∀ i : Idx [128, d.HEAD_DIM], (FloatV.mul (dk.get i) (smF d)).NaNEq (dkm.get i))
    (hx : ∀ i : Idx [128, d.HEAD_DIM], (FloatV.cvtWith .rn (dkm.get i)).NaNEq (x.get i))
    (r : Fin 128) (j : Fin d.HEAD_DIM) (hv : HeadV) (hH : hv.Holds d init b) (hOk : hv.Ok d) :
    ∃ v : F32, v.isFinite = true ∧ |v.toReal - dKE d hv (n0 + r) j| ≤ dKTol d hv (n0 + r) j ∧
      FloatV.NaNEq (FloatV.cvt v) (x.get2 r j) ∧
      (|dKE d hv (n0 + r) j| + dKTol d hv (n0 + r) j ≤ 65504 →
        (x.get2 r j).isFinite = true ∧
          |(x.get2 r j).toReal - dKE d hv (n0 + r) j| ≤
            dKTol d hv (n0 + r) j + (|dKE d hv (n0 + r) j| + dKTol d hv (n0 + r) j) * u16 + eta16) := by
  have hn : n0 + r < d.N_CTX := by have := r.isLt; omega
  have hz : ∀ s < d.N_CTX, s ∉ Finset.Ico (lo d n0) d.N_CTX →
      (famDK d hv).wt ((fun i : Fin 128 => n0 + i) r) s = 0 ∧
      (famDK d hv).err ((fun i : Fin 128 => n0 + i) r) s = 0 ∧
      (famDK d hv).lb ((fun i : Fin 128 => n0 + i) r) s = 0 := by
    intro s hs hns
    have hlo : s < lo d n0 := by
      by_contra hcon
      exact hns (Finset.mem_Ico.mpr ⟨not_lt.mp hcon, hs⟩)
    have hal : ¬ Allowed d (n0 + r) s := by
      unfold lo at hlo
      split_ifs at hlo with hC
      · rintro (h | h)
        · rw [hC] at h; cases h
        · omega
      · omega
    obtain ⟨-, -, -, h1, h2, h3⟩ := notAllowed_zero (hv := hv) hal
    exact ⟨h1, h2, h3⟩
  obtain ⟨hf, hb'⟩ := AccInv.final (hinv hv hH hOk).2 r j d.N_CTX
    (fun s hs => (Finset.mem_Ico.mp hs).2) hz
  have hbd : |(dk.get2 r j).toReal - dKS d hv (n0 + r) j| ≤ dKAcc d hv (n0 + r) j := by
    unfold dKAcc; rw [stepsKV_row d hn0m r.isLt]; exact hb'
  have hroom : |dKS d hv (n0 + r) j| + dKAcc d hv (n0 + r) j ≤ 2 ^ 100 := by
    have hN := hd.n_le
    have := tol_room (N' := d.N_CTX) (t := stepsKV d (n0 + r)) hN (by
        unfold stepsKV; split_ifs <;> omega)
      (wt := fun s => dsE d hv (n0 + r) s) (err := fun s => ds16Err d hv (n0 + r) s)
      (lb := fun s => ds16L d hv (n0 + r) s) (u := fun s => hv.q s j)
      (fun s hs => dsE_abs_le hOk hn hs) (fun s hs => ⟨ds16Err_nonneg hOk hn hs, ds16Err_le hOk hn hs⟩)
      (fun s hs => ⟨lift16_nonneg _, ds16L_le hOk hn hs⟩) (fun s hs => hOk.q s j hs j.isLt)
    exact this
  have hs16 : |(smF d).toReal| ≤ 2 ^ 16 := by
    have := hd.sm.2
    unfold FloatV.toReal; rw [← Rat.cast_abs]; exact_mod_cast this
  obtain ⟨hwf, hwe⟩ := scale_tol hf hd.sm.1 hbd hroom hs16 (hm (r, j, ()))
  exact out_of_f32 hwf hwe (hx (r, j, ()))

/-- **`dQ`**: the accumulator of `dq` times `LN2`, rounded to `float16`, is `OutSpec`'s value for
`dQ[q0 + r, j]`. -/
theorem dq_spec (d : Inst) (hd : d.ok) {init : Mem} {b q0 : Nat}
    (hq0 : q0 + 128 ≤ d.N_CTX) (hq0m : q0 % 128 = 0) {dq : Tensor [128, d.HEAD_DIM] F32}
    (hinv : DqInv d init b q0 (Finset.Ico 0 (hi d q0)) (stepsQ d q0) dq)
    {dqm : Tensor [128, d.HEAD_DIM] F32} {x : Tensor [128, d.HEAD_DIM] F16}
    (hm : ∀ i : Idx [128, d.HEAD_DIM], (FloatV.mul (dq.get i) ln2F).NaNEq (dqm.get i))
    (hx : ∀ i : Idx [128, d.HEAD_DIM], (FloatV.cvtWith .rn (dqm.get i)).NaNEq (x.get i))
    (r : Fin 128) (j : Fin d.HEAD_DIM) (hv : HeadV) (hH : hv.Holds d init b) (hOk : hv.Ok d) :
    ∃ v : F32, v.isFinite = true ∧ |v.toReal - dQE d hv (q0 + r) j| ≤ dQTol d hv (q0 + r) j ∧
      FloatV.NaNEq (FloatV.cvt v) (x.get2 r j) ∧
      (|dQE d hv (q0 + r) j| + dQTol d hv (q0 + r) j ≤ 65504 →
        (x.get2 r j).isFinite = true ∧
          |(x.get2 r j).toReal - dQE d hv (q0 + r) j| ≤
            dQTol d hv (q0 + r) j + (|dQE d hv (q0 + r) j| + dQTol d hv (q0 + r) j) * u16 + eta16) := by
  have hm' : q0 + r < d.N_CTX := by have := r.isLt; omega
  have hz : ∀ s < d.N_CTX, s ∉ Finset.Ico 0 (hi d q0) →
      (famDQ d hv).wt ((fun i : Fin 128 => q0 + i) r) s = 0 ∧
      (famDQ d hv).err ((fun i : Fin 128 => q0 + i) r) s = 0 ∧
      (famDQ d hv).lb ((fun i : Fin 128 => q0 + i) r) s = 0 := by
    intro s hs hns
    have hhi : hi d q0 ≤ s := by
      by_contra hcon
      exact hns (Finset.mem_Ico.mpr ⟨Nat.zero_le _, not_le.mp hcon⟩)
    have hal : ¬ Allowed d s (q0 + r) := by
      unfold hi at hhi
      split_ifs at hhi with hC
      · rintro (h | h)
        · rw [hC] at h; cases h
        · have := r.isLt; omega
      · omega
    obtain ⟨-, -, -, h1, h2, h3⟩ := notAllowed_zero (hv := hv) hal
    exact ⟨h1, h2, h3⟩
  obtain ⟨hf, hb'⟩ := AccInv.final (hinv hv hH hOk) r j d.N_CTX
    (fun s hs => by
      have := (Finset.mem_Ico.mp hs).2
      unfold hi at this; split_ifs at this <;> omega) hz
  have hbd : |(dq.get2 r j).toReal - dQS d hv (q0 + r) j| ≤ dQAcc d hv (q0 + r) j := by
    unfold dQAcc; rw [stepsQ_row d hq0m r.isLt]; exact hb'
  have hroom : |dQS d hv (q0 + r) j| + dQAcc d hv (q0 + r) j ≤ 2 ^ 100 := by
    have hN := hd.n_le
    have := tol_room (N' := d.N_CTX) (t := stepsQ d (q0 + r)) hN (by
        unfold stepsQ; split_ifs <;> omega)
      (wt := fun s => dsE d hv s (q0 + r)) (err := fun s => ds16Err d hv s (q0 + r))
      (lb := fun s => ds16L d hv s (q0 + r)) (u := fun s => hv.k s j)
      (fun s hs => dsE_abs_le hOk hs hm') (fun s hs => ⟨ds16Err_nonneg hOk hs hm', ds16Err_le hOk hs hm'⟩)
      (fun s hs => ⟨lift16_nonneg _, ds16L_le hOk hs hm'⟩) (fun s hs => hOk.k s j hs j.isLt)
    exact this
  obtain ⟨hwf, hwe⟩ := scale_tol hf ln2F_finite hbd hroom ln2F_abs_le (hm (r, j, ()))
  have hwe' : |(dqm.get2 r j).toReal - dQE d hv (q0 + r) j| ≤ dQTol d hv (q0 + r) j := by
    unfold dQE dQTol
    rw [abs_of_nonneg ln2F_nonneg] at hwe
    exact hwe
  exact out_of_f32 hwf hwe' (hx (r, j, ()))

end Claims.AttnBwd
