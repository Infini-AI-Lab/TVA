import Proof.Row
import Proof.LaneG

/-!
# One row's statistics, the float chain (lines 75–82)

`stats_chain` walks the row's operations from the loaded gate lanes to the `bfloat16`-rounded
cutoff: the sum along a tree, `/ d`, the centered lanes, their squares, the fusable sum, `/ d`,
`tl.sqrt`, the multiplier and the `bfloat16` roundings, with the float lemmas of `Proof.Glue` for
finiteness and each step's error, and the stages of `Proof.Row` for the bounds.
-/

namespace Claims.GeluAndMulSparse

open TVA

set_option maxHeartbeats 1600000

/-- A `float32` with value `0` is `0` as a real. -/
theorem toReal_zero_of {x : F32} (h : x.toRat = 0) : x.toReal = 0 := by
  rw [FloatV.toReal_eq, h]; norm_num

/-- **The row's cutoff**: from the gate lanes (`l j`, the values `gl j`) to the kernel's `cutoff`
(`ck`), finite, within `7/8·cutTol − 2^-131` of `cRef`, and at most `2^40`. -/
theorem stats_chain {P : Platform} (hP : Device P) (r : RowP) (hok : r.Ok) {fusion : Bool}
    (l : ℕ → F32) (hl : ∀ j, j < 2 ^ r.L → (l j).isFinite = true ∧ (l j).toReal = r.gl j)
    {t1 : Tree} {S : F32} (hc1 : t1.Covers (2 ^ r.L)) (hd1 : t1.depth ≤ r.L)
    (he1 : t1.Eval (AddRel false) l S)
    {dv : F32} (hdvf : dv.isFinite = true) (hdv : dv.toReal = r.d)
    {mean0 : F32} (hmean : P.div .binary32 S dv mean0)
    (cent : ℕ → F32)
    (hcent : ∀ j, j < r.d → AddRel fusion (l j) (FloatV.neg mean0) (cent j))
    (hcent0 : ∀ j, r.d ≤ j → j < 2 ^ r.L → cent j = c32 0)
    (q : ℕ → F32) (hq : ∀ j, j < 2 ^ r.L → FloatV.NaNEq (FloatV.mul (cent j) (cent j)) (q j))
    {t2 : Tree} {Q : F32} (hc2 : t2.Covers (2 ^ r.L)) (hd2 : t2.depth ≤ r.L)
    (he2 : t2.Eval (AddRel fusion) q Q)
    {var : F32} (hvar : P.div .binary32 Q dv var)
    {mb16 : BF16} {mb : F32} (hmb1 : FloatV.NaNEq (FloatV.cvtWith .rn mean0) mb16)
    (hmb2 : FloatV.NaNEq (FloatV.cvtWith .rn mb16) mb)
    {sq : F32} (hsq : P.math .sqrt .binary32 var sq)
    {sb16 : BF16} {sb : F32} (hsb1 : FloatV.NaNEq (FloatV.cvtWith .rn sq) sb16)
    (hsb2 : FloatV.NaNEq (FloatV.cvtWith .rn sb16) sb)
    {m32 : F32} (hm32f : m32.isFinite = true) (hm32 : |m32.toReal - r.m| ≤ u32 * |r.m| + eta32)
    {pr : F32} (hpr : FloatV.NaNEq (FloatV.mul sb m32) pr)
    {pb16 : BF16} {pb : F32} (hpb1 : FloatV.NaNEq (FloatV.cvtWith .rn pr) pb16)
    (hpb2 : FloatV.NaNEq (FloatV.cvtWith .rn pb16) pb)
    {cs : F32} (hcs : FloatV.NaNEq (FloatV.add mb pb) cs)
    {ck16 : BF16} {ck : F32} (hck1 : FloatV.NaNEq (FloatV.cvtWith .rn cs) ck16)
    (hck2 : FloatV.NaNEq (FloatV.cvtWith .rn ck16) ck) :
    ck.isFinite = true ∧ |ck.toReal - r.cRef| ≤ 7 / 8 * r.cutTol - 2 ^ (-131 : ℤ) ∧
      |ck.toReal| ≤ 2 ^ 40 := by
  have hL := r.L_le
  have hn : 2 ^ r.L ≤ 2 ^ 15 := Nat.pow_le_pow_right (by norm_num) hL
  have hu : u32 = 1 / 16777216 := RowP.u32_eq
  have he : eta32 ≤ 1 / 10 ^ 45 := by unfold eta32; norm_num
  have he0 : 0 ≤ eta32 := by unfold eta32; positivity
  have hub : ub = 1 / 256 := by unfold ub; norm_num
  have heb : etab ≤ 1 / 10 ^ 40 := by unfold etab; norm_num
  have heb0 : 0 ≤ etab := by unfold etab; positivity
  have hA0 := r.aMean_nonneg
  have hA1 := r.aMean_le
  have hmu := r.mu_le
  have hsd0 := r.sd_nonneg
  have hsd1 := r.sd_le
  have hmle := r.m_le
  -- the sum and the mean
  have hlb : ∀ i, i < 2 ^ r.L → |(l i).toReal| ≤ 2 ^ 40 := by
    intro i hi
    rw [(hl i hi).2]
    unfold RowP.gl
    split_ifs with h
    · exact (r.g_le i h).trans (by norm_num)
    · simp
  obtain ⟨hSf, hSe⟩ := sum_real hL hn l hc1 hd1 he1 (fun i hi => (hl i hi).1) hlb
  have hsum : ∑ i ∈ Finset.range (2 ^ r.L), (l i).toReal = ∑ j ∈ Finset.range (2 ^ r.L), r.gl j :=
    Finset.sum_congr rfl (fun i hi => (hl i (Finset.mem_range.mp hi)).2)
  have hsumabs : ∑ i ∈ Finset.range (2 ^ r.L), |(l i).toReal| =
      ∑ j ∈ Finset.range (2 ^ r.L), |r.gl j| :=
    Finset.sum_congr rfl (fun i hi => by rw [(hl i (Finset.mem_range.mp hi)).2])
  rw [hsum, hsumabs] at hSe
  have hSroom := r.sum_room hSe
  have hd1' : (1 : ℝ) ≤ r.d := by exact_mod_cast r.d_pos
  have hdpos : (0 : ℝ) < r.d := by linarith
  have hq1 : |S.toReal / dv.toReal| ≤ 2 ^ 126 := by
    rw [hdv, abs_div, abs_of_pos hdpos, div_le_iff₀ hdpos]
    have : (2 : ℝ) ^ 48 ≤ 2 ^ 126 * 1 := by norm_num
    nlinarith
  obtain ⟨hm0f, hm0e⟩ := div_real hP hSf hdvf (by rw [hdv]; exact hdpos.ne') hq1 hmean
  rw [hdv] at hm0e
  have hm0 := r.mean_stage hSe hm0e
  have hmE := r.meanErr_le
  rw [hu] at hmE
  have h134 : (2 : ℝ) ^ (-134 : ℤ) ≤ 1 := by norm_num
  have hmb : |mean0.toReal| ≤ 2 ^ 33 := by
    have := abs_le_of_err hm0
    linarith
  -- the centered lanes
  have hnf : (FloatV.neg mean0).isFinite = true := (FloatV.isFinite_neg _).trans hm0f
  have hcf : ∀ j, j < 2 ^ r.L → (cent j).isFinite = true ∧
      (j < r.d → |(cent j).toReal - (r.g j - mean0.toReal)| ≤
        3 * u32 * (|r.g j| + |mean0.toReal|) + 4 * eta32) := by
    intro j hj
    by_cases hjd : j < r.d
    · have hlj := hl j hj
      have hgl : (l j).toReal = r.g j := by rw [hlj.2]; unfold RowP.gl; rw [if_pos hjd]
      have hroom : |(l j).toReal| + |(FloatV.neg mean0).toReal| ≤ 2 ^ 126 + 2 ^ 111 := by
        rw [FloatV.toReal_neg, abs_neg]
        have := hlb j hj
        have : (2 : ℝ) ^ 40 + 2 ^ 33 ≤ 2 ^ 126 + 2 ^ 111 := by norm_num
        linarith
      obtain ⟨hf, hb⟩ := addc_real hlj.1 hnf hroom (hcent j hjd)
      refine ⟨hf, fun _ => ?_⟩
      rw [FloatV.toReal_neg, abs_neg, hgl] at hb
      have e : r.g j + -mean0.toReal = r.g j - mean0.toReal := by ring
      rw [e] at hb
      exact hb
    · rw [hcent0 j (by omega) hj]
      exact ⟨c32_zero_val.1, fun h => absurd h hjd⟩
  set c : ℕ → ℝ := fun j => (cent j).toReal with hcdef
  have hc : ∀ j, j < r.d → |c j - (r.g j - mean0.toReal)| ≤
      3 * u32 * (|r.g j| + |mean0.toReal|) + 4 * eta32 := fun j hj =>
    (hcf j (lt_of_lt_of_le hj r.d_le)).2 hj
  have hc0 : ∀ j, r.d ≤ j → j < 2 ^ r.L → c j = 0 := by
    intro j h1 h2
    simp only [hcdef]
    rw [hcent0 j h1 h2]
    exact toReal_zero_of c32_zero_val.2.1
  have hcb := r.c_le hm0 hc hc0
  -- the squares and their sum
  have hqf : ∀ j, j < 2 ^ r.L → (q j).isFinite = true ∧
      |(q j).toReal - c j * c j| ≤ u32 * |c j * c j| + eta32 := by
    intro j hj
    have hcc : |c j * c j| ≤ 2 ^ 68 := by
      rw [abs_mul]
      calc |c j| * |c j| ≤ 2 ^ 34 * 2 ^ 34 :=
            mul_le_mul (hcb j hj) (hcb j hj) (abs_nonneg _) (by positivity)
        _ = 2 ^ 68 := by norm_num
    exact mul_lane (hcf j hj).1 (hcf j hj).1 (hcc.trans (by norm_num)) (hq j hj)
  have hqb : ∀ j, j < 2 ^ r.L → |(q j).toReal| ≤ 2 ^ 70 := by
    intro j hj
    have hcc : |c j * c j| ≤ 2 ^ 68 := by
      rw [abs_mul]
      calc |c j| * |c j| ≤ 2 ^ 34 * 2 ^ 34 :=
            mul_le_mul (hcb j hj) (hcb j hj) (abs_nonneg _) (by positivity)
        _ = 2 ^ 68 := by norm_num
    have := abs_le_of_err (hqf j hj).2
    rw [hu] at this
    linarith
  obtain ⟨hQf, hQe⟩ := sumc_real hL hn q hc2 hd2 he2 (fun i hi => (hqf i hi).1) hqb
  have hQb := r.Q_le hcb (fun j hj => (hqf j hj).2) hQe
  have hq2 : |Q.toReal / dv.toReal| ≤ 2 ^ 126 := by
    rw [hdv, abs_div, abs_of_pos hdpos, div_le_iff₀ hdpos]
    have : (2 : ℝ) ^ 90 ≤ 2 ^ 126 * 1 := by norm_num
    nlinarith
  obtain ⟨hvf, hve⟩ := div_real hP hQf hdvf (by rw [hdv]; exact hdpos.ne') hq2 hvar
  rw [hdv] at hve
  obtain ⟨hvlo, hvhi, hsv⟩ := r.var_stage hok hm0 hc hc0 (fun j hj => (hqf j hj).2) hQe hve
  -- the square root
  obtain ⟨hsqf, hsqe⟩ := sqrt_real hP hvf hvlo (hvhi.trans (by norm_num)) hsq
  have hsd := r.std_stage hok hvlo hsv hsqe
  have hsE := r.stdErr_le
  rw [hu] at hsE
  have hsqb : |sq.toReal| ≤ 2 ^ 34 := by
    have := abs_le_of_err hsd
    rw [abs_of_nonneg hsd0] at this
    linarith
  -- the roundings and the cutoff
  obtain ⟨hmb16f, hmb16e⟩ := cvt16_lane hm0f (hmb.trans (by norm_num)) hmb1
  obtain ⟨hmbf, hmbv⟩ := cvt32_lane hmb16f hmb2
  rw [← hmbv] at hmb16e
  obtain ⟨hsb16f, hsb16e⟩ := cvt16_lane hsqf (hsqb.trans (by norm_num)) hsb1
  obtain ⟨hsbf, hsbv⟩ := cvt32_lane hsb16f hsb2
  rw [← hsbv] at hsb16e
  have hsbb : |sb.toReal| ≤ 2 ^ 35 := by
    have := abs_le_of_err hsb16e
    rw [hub] at this
    nlinarith
  have hm32b : |m32.toReal| ≤ 65 := by
    have := abs_le_of_err hm32
    rw [hu] at this
    nlinarith
  have hprr : |sb.toReal * m32.toReal| ≤ 2 ^ 42 := by
    rw [abs_mul]
    calc |sb.toReal| * |m32.toReal| ≤ 2 ^ 35 * 65 :=
          mul_le_mul hsbb hm32b (abs_nonneg _) (by positivity)
      _ ≤ 2 ^ 42 := by norm_num
  obtain ⟨hprf, hpre⟩ := mul_lane hsbf hm32f (hprr.trans (by norm_num)) hpr
  have hprb : |pr.toReal| ≤ 2 ^ 43 := by
    have := abs_le_of_err hpre
    rw [hu] at this
    linarith
  obtain ⟨hpb16f, hpb16e⟩ := cvt16_lane hprf (hprb.trans (by norm_num)) hpb1
  obtain ⟨hpbf, hpbv⟩ := cvt32_lane hpb16f hpb2
  rw [← hpbv] at hpb16e
  have hpbb : |pb.toReal| ≤ 2 ^ 44 := by
    have := abs_le_of_err hpb16e
    rw [hub] at this
    linarith
  have hmbb : |mb.toReal| ≤ 2 ^ 34 := by
    have := abs_le_of_err hmb16e
    rw [hub] at this
    linarith
  have hcsr : |mb.toReal + pb.toReal| ≤ 2 ^ 45 := by
    have := abs_add_le mb.toReal pb.toReal
    linarith
  obtain ⟨hcsf, hcse⟩ := add_lane hmbf hpbf (hcsr.trans (by norm_num)) hcs
  have hcsb : |cs.toReal| ≤ 2 ^ 46 := by
    have := abs_le_of_err hcse
    rw [hu] at this
    linarith
  obtain ⟨hck16f, hck16e⟩ := cvt16_lane hcsf (hcsb.trans (by norm_num)) hck1
  obtain ⟨hckf, hckv⟩ := cvt32_lane hck16f hck2
  rw [← hckv] at hck16e
  have hcut := r.cut_stage hok hm0 hsd hm32 hmb16e hsb16e hpre hpb16e hcse hck16e
  refine ⟨hckf, hcut, ?_⟩
  -- the size of the cutoff
  have hcT : r.cutTol ≤ 2 ^ 34 := by
    unfold RowP.cutTol
    have h1 : |r.m| * r.sd ≤ 64 * 2 ^ 33 := mul_le_mul hmle hsd1 hsd0 (by norm_num)
    have h2 : (1 + |r.m|) * r.aMean ≤ 65 * 2 ^ 32 :=
      mul_le_mul (by linarith) hA1 hA0 (by norm_num)
    have h6 : (2 : ℝ) ^ (-6 : ℤ) = 1 / 64 := by norm_num
    have h17 : (2 : ℝ) ^ (-17 : ℤ) = 1 / 131072 := by norm_num
    have h130 : (2 : ℝ) ^ (-130 : ℤ) ≤ 1 := by norm_num
    rw [h6, h17]
    nlinarith
  have hcR : |r.cRef| ≤ 2 ^ 32 + 64 * 2 ^ 33 := by
    unfold RowP.cRef
    have h1 : |r.m * r.sd| ≤ 64 * 2 ^ 33 := by
      rw [abs_mul, abs_of_nonneg hsd0]; exact mul_le_mul hmle hsd1 hsd0 (by norm_num)
    have := abs_add_le r.mu (r.m * r.sd)
    linarith
  have h131 : (0 : ℝ) ≤ 2 ^ (-131 : ℤ) := by positivity
  have := abs_le_of_err hcut
  linarith

end Claims.GeluAndMulSparse
