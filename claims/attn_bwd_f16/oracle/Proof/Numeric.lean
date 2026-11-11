import Proof.Defs

/-!
# `attn_bwd_f16`: the numeric facts

Everything about numbers, none about the program: the `float16` lifted magnitudes, the
accumulation recursion over `ℝ`, the accuracy of one lane of `tl.dot`, the accuracy of the
`P` and `dS` of one pair, one accumulation step of `AccInv`, and the conversions to `OutSpec`.
-/

namespace Claims.AttnBwd

open TVA

/-! ## Lifted magnitudes -/

/-- A lifted magnitude is nonnegative. -/
theorem lm_nonneg (x : F16) : 0 ≤ lm x := by
  unfold lm
  exact_mod_cast Device.liftMag_nonneg _ _

/-- The lifted magnitude of a finite `float16` is at most `65504`. -/
theorem lm_le_65504 {x : F16} (hx : x.isFinite = true) : lm x ≤ 65504 := by
  have h : |x.toRat| ≤ 65504 := by
    have := FloatV.abs_toRat_le_maxVal Format.binary16_regular hx
    rwa [Format.binary16_maxVal] at this
  unfold lm Device.liftMag
  split_ifs with h0
  · norm_num
  · have : max |x.toRat| (Device.liftFloor .f16) ≤ 65504 := by
      refine max_le h ?_
      rw [Device.liftFloor_f16]; norm_num
    exact_mod_cast this

/-- A product's magnitude is at most the product of the lifted magnitudes. -/
theorem abs_mul_le_lm (x y : F16) : |x.toReal * y.toReal| ≤ lm x * lm y := by
  have := Device.abs_mul_le_liftMag .f16 .f16 x.toRat y.toRat
  unfold lm FloatV.toReal
  rw [← Rat.cast_mul, ← Rat.cast_abs, ← Rat.cast_mul]
  exact_mod_cast this

/-- `lift16` is nonnegative. -/
theorem lift16_nonneg (B : ℝ) : 0 ≤ lift16 B := by
  unfold lift16
  split_ifs
  · exact le_rfl
  · exact le_trans (by positivity) (le_max_right _ _)

/-- A finite `float16` of magnitude at most `B` has lifted magnitude at most `lift16 B`. -/
theorem lm_le_of_abs_le {x : F16} {B : ℝ} (h : |x.toReal| ≤ B) :
    lm x ≤ lift16 B := by
  unfold lift16
  by_cases h0 : x.toRat = 0
  · have : lm x = 0 := by
      unfold lm Device.liftMag
      rw [ite_eq_left h0]; simp
    rw [this]
    exact lift16_nonneg B
  · have h24 := FloatV.two_pow_neg24_le_abs_toRat_f16 h0
    have h24' : (2 : ℝ) ^ (-24 : ℤ) ≤ |x.toReal| := by
      unfold FloatV.toReal
      have : ((2 : ℚ) ^ (-24 : ℤ) : ℝ) ≤ ((|x.toRat| : ℚ) : ℝ) := by exact_mod_cast h24
      simpa using this
    have hB : ¬ B < 2 ^ (-24 : ℤ) := not_lt.mpr (h24'.trans h)
    rw [ite_eq_right hB]
    unfold lm Device.liftMag
    rw [ite_eq_right h0]
    have habs : ((|x.toRat| : ℚ) : ℝ) ≤ B := by rw [Rat.cast_abs]; exact h
    have hfl : ((Device.liftFloor .f16 : ℚ) : ℝ) = 2 ^ (-14 : ℤ) := by
      rw [Device.liftFloor_f16]; push_cast; rfl
    rw [Rat.cast_max, hfl]
    exact max_le_max habs le_rfl

/-- `lift16` of a bound at most `2^16` is at most `2^16`. -/
theorem lift16_le {B : ℝ} (h : B ≤ 2 ^ 16) : lift16 B ≤ 2 ^ 16 := by
  unfold lift16
  split_ifs
  · positivity
  · exact max_le h (by norm_num)

/-! ## The accumulation recursion -/

theorem rhoA_q : (DotLoop.stepRel Device.relDot16 32 : ℚ) = 34 * 2 ^ (-22 : ℤ) +
    3 * 2 ^ (-24 : ℤ) * (1 + 34 * 2 ^ (-22 : ℤ)) := by
  unfold DotLoop.stepRel Device.relDot16
  rw [Format.binary32_u]; norm_num

/-- `alphaA` in `ℚ`. -/
theorem alphaA_q : (DotLoop.stepAbs Device.absDot16 32 : ℚ) =
    34 * 2 ^ (-126 : ℤ) * (1 + 3 * 2 ^ (-24 : ℤ)) + 4 * 2 ^ (-150 : ℤ) := by
  unfold DotLoop.stepAbs Device.absDot16
  rw [Format.binary32_u, Format.binary32_eta]; norm_num

/-- The relative constant of one accumulation step is nonnegative. -/
theorem rhoA_nonneg : 0 ≤ rhoA := by
  unfold rhoA; rw [rhoA_q]; norm_num

/-- The relative constant of one accumulation step is below `2^-16`. -/
theorem rhoA_le : rhoA ≤ 1 / 65536 := by
  unfold rhoA; rw [rhoA_q]; norm_num

/-- The absolute constant of one accumulation step is nonnegative. -/
theorem alphaA_nonneg : 0 ≤ alphaA := by
  unfold alphaA; rw [alphaA_q]; norm_num

/-- The absolute constant of one accumulation step is below `2^-100`. -/
theorem alphaA_le : alphaA ≤ 1 / 2048 := by
  unfold alphaA; rw [alphaA_q]; norm_num

/-- No steps, no error. -/
theorem accB_zero (B : ℝ) : accB 0 B = 0 := by
  simp [accB]

/-- `(1 + rhoA)^t ≥ 1`. -/
theorem one_le_pow_rho (t : ℕ) : 1 ≤ (1 + rhoA) ^ t :=
  one_le_pow₀ (by linarith [rhoA_nonneg])

/-- One trip of the recursion over `ℝ` (`DotLoop.accum_step`, with a real magnitude). -/
theorem accB_step {t : ℕ} {a a' St S Bt A : ℝ} (hSt : |St| ≤ Bt) (hSA : |S| ≤ A)
    (he : |a - St| ≤ accB t Bt) (hs : |a' - (a + S)| ≤ rhoA * (|a| + A) + alphaA) :
    |a' - (St + S)| ≤ accB (t + 1) (Bt + A) := by
  have hρ := rhoA_nonneg
  have hα := alphaA_nonneg
  have hA0 : 0 ≤ A := (abs_nonneg _).trans hSA
  have hacc : |a| ≤ Bt + accB t Bt := by
    have := abs_sub_abs_le_abs_sub a St
    linarith
  have htri : |a' - (St + S)| ≤ |a' - (a + S)| + |a - St| := by
    have e2 : a' - (St + S) = (a' - (a + S)) + (a - St) := by ring
    rw [e2]; exact abs_add_le _ _
  set P := (1 + rhoA) ^ t with hP
  have hP1 : 1 ≤ P := one_le_pow_rho t
  unfold accB at he hacc ⊢
  rw [← hP] at he hacc
  rw [pow_succ, ← hP]
  push_cast
  have hBt : 0 ≤ Bt := (abs_nonneg _).trans hSt
  have hm1 : rhoA * |a| ≤ rhoA * (Bt + ((P - 1) * Bt + t * P * alphaA)) :=
    mul_le_mul_of_nonneg_left hacc hρ
  nlinarith [mul_nonneg (mul_nonneg hρ (by linarith : (0 : ℝ) ≤ P - 1)) hA0,
    mul_nonneg (by linarith : (0 : ℝ) ≤ P - 1) hA0, mul_nonneg hρ hA0, mul_nonneg hρ hα,
    mul_nonneg (mul_nonneg hρ (by linarith : (0 : ℝ) ≤ P)) hα, mul_nonneg (Nat.cast_nonneg t) hα,
    mul_nonneg (mul_nonneg (Nat.cast_nonneg t) (by linarith : (0 : ℝ) ≤ P)) hα]

/-- Up to `1024` steps the error bound is below the magnitude plus one. -/
theorem accB_le {t : ℕ} (ht : t ≤ 1024) {B : ℝ} (hB : 0 ≤ B) : accB t B ≤ B + 1 := by
  have hρ := rhoA_nonneg
  have hα := alphaA_nonneg
  have hρle := rhoA_le
  have hαle := alphaA_le
  have hq : (DotLoop.stepRel Device.relDot16 32 : ℚ) * t ≤ 1 / 2 := by
    have : (t : ℚ) ≤ 1024 := by exact_mod_cast ht
    rw [rhoA_q]; norm_num
    nlinarith
  have hpow : (1 + rhoA) ^ t ≤ 1 + 2 * t * rhoA := by
    have := DotLoop.pow_le (x := DotLoop.stepRel Device.relDot16 32) (by rw [rhoA_q]; norm_num) t
      (by rw [mul_comm]; exact hq)
    unfold rhoA
    exact_mod_cast this
  unfold accB
  have ht' : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg t
  have h1 : ((1 + rhoA) ^ t - 1) * B ≤ 2 * t * rhoA * B :=
    mul_le_mul_of_nonneg_right (by linarith) hB
  have hρ' := hρle
  have h2 : 2 * (t : ℝ) * rhoA ≤ 1 / 2 := by
    have : (t : ℝ) * rhoA ≤ 1024 * (1 / 65536) := mul_le_mul ht' hρ' hρ (by norm_num)
    linarith
  have h3 : (1 + rhoA) ^ t ≤ 2 := by linarith
  have h4 : (t : ℝ) * (1 + rhoA) ^ t * alphaA ≤ 1 := by
    have hα' := hαle
    have h5 : (t : ℝ) * (1 + rhoA) ^ t ≤ 1024 * 2 :=
      mul_le_mul ht' h3 (by positivity) (by norm_num)
    have := mul_le_mul h5 hα' hα (by norm_num)
    linarith
  nlinarith

/-! ## One lane of `tl.dot` -/

/-- **One lane of `tl.dot`** of `float16` operands into `float32` (`Device.DotF16`): row `i` of `a`
is `x` and column `j` of `b` is `y`, all finite, over `K ≤ 256` products, from an accumulator lane of
value `c` (`|c| ≤ 2^100`); the result lane is finite and within
`relDot16 K·(|c| + Σ lm x·lm y) + absDot16 K` of `c + Σ x·y`. -/
theorem dot_lane {P : Platform} (hdot : Device.DotF16 P) {M N K : Nat} (hK : K ≤ 256)
    {a : Tensor [M, K] F16} {b : Tensor [K, N] F16} {acc r : Tensor [M, N] F32}
    (hr : P.dot dotCfg a b acc r) (i : Fin M) (j : Fin N) (x y : Nat → F16)
    (hx : ∀ k : Fin K, a.get2 i k = x k) (hy : ∀ k : Fin K, b.get2 k j = y k)
    (hxf : ∀ k : Fin K, (x k).isFinite = true) (hyf : ∀ k : Fin K, (y k).isFinite = true)
    {c : ℚ} (hc : DType.val? .f32 (acc.get2 i j) = some c) (hcb : |c| ≤ 2 ^ 100) :
    ∃ vr : ℚ, DType.val? .f32 (r.get2 i j) = some vr ∧
      |(vr : ℝ) - ((c : ℝ) + ∑ k ∈ Finset.range K, (x k).toReal * (y k).toReal)| ≤
        (Device.relDot16 K : ℝ) * (|(c : ℝ)| + ∑ k ∈ Finset.range K, lm (x k) * lm (y k)) +
          (Device.absDot16 K : ℝ) := by
  have hva : ∀ k : Fin K, DType.val? .f16 (a.get2 i k) = some (x k).toRat := fun k =>
    DType.val?_f16.mpr ⟨by rw [hx k]; exact hxf k, by rw [hx k]⟩
  have hvb : ∀ k : Fin K, DType.val? .f16 (b.get2 k j) = some (y k).toRat := fun k =>
    DType.val?_f16.mpr ⟨by rw [hy k]; exact hyf k, by rw [hy k]⟩
  have hL : ∑ k : Fin K, Device.liftMag .f16 (x k).toRat * Device.liftMag .f16 (y k).toRat ≤
      K * (65504 * 65504) := by
    calc _ ≤ ∑ _k : Fin K, (65504 * 65504 : ℚ) := by
          refine Finset.sum_le_sum fun k _ => ?_
          have h1 : Device.liftMag .f16 (x k).toRat ≤ 65504 := by
            have := lm_le_65504 (hxf k)
            unfold lm at this
            exact_mod_cast this
          have h2 : Device.liftMag .f16 (y k).toRat ≤ 65504 := by
            have := lm_le_65504 (hyf k)
            unfold lm at this
            exact_mod_cast this
          exact mul_le_mul h1 h2 (Device.liftMag_nonneg _ _) (by norm_num)
      _ = K * (65504 * 65504) := by simp
  have hLnn : 0 ≤ ∑ k : Fin K, Device.liftMag .f16 (x k).toRat * Device.liftMag .f16 (y k).toRat :=
    Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hKq : (K : ℚ) ≤ 256 := by exact_mod_cast hK
  have hrel : Device.relDot16 K ≤ 1 := by
    unfold Device.relDot16
    have : (2 : ℚ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
    rw [this]; linarith
  have habs : Device.absDot16 K ≤ 1 := by
    unfold Device.absDot16
    have h : (2 : ℚ) ^ (-126 : ℤ) ≤ 1 / 4194304 := by norm_num
    calc ((K : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ 258 * (1 / 4194304) :=
          mul_le_mul (by linarith) h (by positivity) (by norm_num)
      _ ≤ 1 := by norm_num
  have hrel0 : 0 ≤ Device.relDot16 K := by unfold Device.relDot16; positivity
  have habs0 : 0 ≤ Device.absDot16 K := by unfold Device.absDot16; positivity
  obtain ⟨vr, hvr, he⟩ := hdot .tf32 a b acc r hr i j (fun k => (x k).toRat) (fun k => (y k).toRat)
    c hva hvb hc (by
      show _ ≤ DType.bound .f32
      simp only [DType.bound, Format.binary32_maxVal]
      have hc' : |c| ≤ 2 ^ 100 := hcb
      have hL' : (K : ℚ) * (65504 * 65504) ≤ 2 ^ 40 := by nlinarith
      have hLq : ∑ k : Fin K, Device.liftMag .f16 (x k).toRat * Device.liftMag .f16 (y k).toRat ≤
          2 ^ 40 := hL.trans hL'
      have h1 : |c| + ∑ k : Fin K, Device.liftMag .f16 (x k).toRat * Device.liftMag .f16 (y k).toRat
          ≤ 2 ^ 101 := by
        have : (2 : ℚ) ^ 40 ≤ 2 ^ 100 := by norm_num
        linarith
      have h0 : 0 ≤ |c| + ∑ k : Fin K, Device.liftMag .f16 (x k).toRat * Device.liftMag .f16 (y k).toRat :=
        by positivity
      have h2 : Device.relDot16 K * (|c| + ∑ k : Fin K, Device.liftMag .f16 (x k).toRat *
          Device.liftMag .f16 (y k).toRat) ≤ 2 ^ 101 := by
        calc _ ≤ 1 * (2 ^ 101 : ℚ) := mul_le_mul hrel h1 h0 (by norm_num)
          _ = 2 ^ 101 := one_mul _
      have : (2 : ℚ) ^ 101 + 2 ^ 101 + 1 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
      linarith)
  refine ⟨vr, hvr, ?_⟩
  have he' := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at he'
  have e1 := Fin.sum_univ_eq_sum_range (fun k => (x k).toReal * (y k).toReal) K
  have e2 := Fin.sum_univ_eq_sum_range (fun k => lm (x k) * lm (y k)) K
  rw [← e1, ← e2]
  exact he'



/-- `float16`'s unit roundoff is nonnegative. -/
theorem u16_nonneg : (0 : ℝ) ≤ u16 := by
  unfold u16; rw [Format.binary16_u_real]; positivity

/-- `float32`'s unit roundoff is nonnegative. -/
theorem u32_nonneg : (0 : ℝ) ≤ u32 := by
  unfold u32; rw [Format.binary32_u_real]; positivity

/-- `float16`'s `η` is nonnegative. -/
theorem eta16_nonneg : (0 : ℝ) ≤ eta16 := by
  unfold eta16; rw [Format.binary16_eta_real]; positivity

/-- `float32`'s `η` is nonnegative. -/
theorem eta32_nonneg : (0 : ℝ) ≤ eta32 := by
  unfold eta32; rw [Format.binary32_eta_real]; positivity

/-- `float16`'s unit roundoff is at most `2^-11`. -/
theorem u16_le : u16 ≤ 1 / 2048 := by
  unfold u16; rw [Format.binary16_u_real]; norm_num

/-- `float32`'s unit roundoff is at most `2^-11` (a coarse bound). -/
theorem u32_le : u32 ≤ 1 / 2048 := by
  unfold u32; rw [Format.binary32_u_real]; norm_num

/-- `float16`'s `η` is small. -/
theorem eta16_le : eta16 ≤ 1 / 2048 := by
  unfold eta16; rw [Format.binary16_eta_real]; norm_num

/-- `float32`'s `η` is small. -/
theorem eta32_le : eta32 ≤ 1 / 2048 := by
  unfold eta32; rw [Format.binary32_eta_real]; norm_num

/-- The bound on the computed `P` of a pair that fits is at most `2^15 + 1`. -/
theorem pBr_le {d : Inst} {hv : HeadV} {n m : Nat}
    (h : xE d hv n m + eX d hv n m ≤ 15) : pBr d hv n m ≤ 2 ^ 15 + 1 := by
  unfold pBr
  have h1 : (2 : ℝ) ^ (xE d hv n m + eX d hv n m) ≤ 2 ^ (15 : ℝ) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) h
  have h2 : (2 : ℝ) ^ (15 : ℝ) = 2 ^ 15 := by norm_num
  rw [h2] at h1
  have h3 : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 / 65536 := by norm_num
  have h4 : (2 : ℝ) ^ (-126 : ℤ) ≤ 1 / 2 := by norm_num
  have h5 : 0 ≤ (2 : ℝ) ^ (xE d hv n m + eX d hv n m) := by positivity
  have h6 : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ (xE d hv n m + eX d hv n m) ≤ 1 / 65536 * 2 ^ 15 :=
    mul_le_mul h3 h1 h5 (by norm_num)
  calc (1 + (2 : ℝ) ^ (-22 : ℤ)) * 2 ^ (xE d hv n m + eX d hv n m) + 2 ^ (-126 : ℤ)
      = 2 ^ (xE d hv n m + eX d hv n m) + (2 : ℝ) ^ (-22 : ℤ) * 2 ^ (xE d hv n m + eX d hv n m) +
        2 ^ (-126 : ℤ) := by ring
    _ ≤ 2 ^ 15 + 1 / 65536 * 2 ^ 15 + 1 / 2 := by linarith
    _ ≤ 2 ^ 15 + 1 := by norm_num

/-- The bound on the computed `P` is nonnegative. -/
theorem pBr_nonneg (d : Inst) (hv : HeadV) (n m : Nat) : 0 ≤ pBr d hv n m := by
  unfold pBr; positivity

/-- The lifted bound of the `float16` `P` of a covered pair is at most `2^16`. -/
theorem p16L_le {d : Inst} {hv : HeadV} (hok : hv.Ok d) {n m : Nat}
    (hn : n < d.N_CTX) (hm : m < d.N_CTX) : p16L d hv n m ≤ 2 ^ 16 := by
  unfold p16L
  refine lift16_le ?_
  unfold p16B
  split_ifs with hal
  · have h1 := pBr_le (hok.fits n m hn hm hal).1
    have h2 := pBr_nonneg d hv n m
    have h3 := u16_le
    have h4 := eta16_le
    nlinarith [u16_nonneg]
  · positivity

/-- The relative constant of the `HEAD_DIM`-long dot is nonnegative. -/
theorem rH_nonneg (d : Inst) : 0 ≤ rH d := by
  unfold rH; exact_mod_cast (by unfold Device.relDot16; positivity : (0 : ℚ) ≤ Device.relDot16 d.HEAD_DIM)

/-- The absolute constant of the `HEAD_DIM`-long dot is nonnegative. -/
theorem aH_nonneg (d : Inst) : 0 ≤ aH d := by
  unfold aH; exact_mod_cast (by unfold Device.absDot16; positivity : (0 : ℚ) ≤ Device.absDot16 d.HEAD_DIM)

/-- `aVD` is nonnegative. -/
theorem aVD_nonneg (d : Inst) (hv : HeadV) (n m : Nat) : 0 ≤ aVD d hv n m :=
  Finset.sum_nonneg fun _ _ => mul_nonneg (lm_nonneg _) (lm_nonneg _)

/-- `aQK` is nonnegative. -/
theorem aQK_nonneg (d : Inst) (hv : HeadV) (n m : Nat) : 0 ≤ aQK d hv n m :=
  Finset.sum_nonneg fun _ _ => mul_nonneg (lm_nonneg _) (lm_nonneg _)

/-- `eSub` is nonnegative. -/
theorem eSub_nonneg (d : Inst) {s a x : ℝ} (ha : 0 ≤ a) : 0 ≤ eSub d s a x := by
  unfold eSub
  have := rH_nonneg d
  have := aH_nonneg d
  have := u32_nonneg
  have := eta32_nonneg
  positivity

/-- `yB` is nonnegative. -/
theorem yB_nonneg (d : Inst) (hv : HeadV) (n m : Nat) : 0 ≤ yB d hv n m := by
  unfold yB eY
  have := eSub_nonneg d (s := sVD d hv n m) (x := ((hv.dl m).toReal)) (aVD_nonneg d hv n m)
  positivity

/-- The lifted bound of the `float16` `dS` of a covered pair is at most `2^16`. -/
theorem ds16L_le {d : Inst} {hv : HeadV} (hok : hv.Ok d) {n m : Nat}
    (hn : n < d.N_CTX) (hm : m < d.N_CTX) : ds16L d hv n m ≤ 2 ^ 16 := by
  unfold ds16L
  refine lift16_le ?_
  unfold ds16B
  split_ifs with hal
  · have h1 := (hok.fits n m hn hm hal).2
    have h2 : 0 ≤ pBr d hv n m * yB d hv n m := mul_nonneg (pBr_nonneg d hv n m) (yB_nonneg d hv n m)
    unfold dsB
    have h3 := u16_le
    have h4 := eta16_le
    have h5 := u32_le
    have h6 := eta32_le
    have h7 : (1 + u32) * (pBr d hv n m * yB d hv n m) + eta32 ≤ (1 + 1 / 2048) * 2 ^ 15 + 1 / 2048 :=
      add_le_add (mul_le_mul (by linarith) h1 h2 (by norm_num)) h6
    have h8 : 0 ≤ (1 + u32) * (pBr d hv n m * yB d hv n m) + eta32 := by
      have := u32_nonneg; have := eta32_nonneg; positivity
    calc (1 + u16) * ((1 + u32) * (pBr d hv n m * yB d hv n m) + eta32) + eta16
        ≤ (1 + 1 / 2048) * ((1 + 1 / 2048) * 2 ^ 15 + 1 / 2048) + 1 / 2048 :=
          add_le_add (mul_le_mul (by linarith) h7 h8 (by norm_num)) h4
      _ ≤ 2 ^ 16 := by norm_num
  · positivity

/-! ## One accumulation step -/

/-- The relative constant of one accumulation step over `K ≤ 32` products is at most `rhoA`. -/
theorem stepRel_le {K : ℕ} (hK : K ≤ 32) :
    ((DotLoop.stepRel Device.relDot16 K : ℚ) : ℝ) ≤ rhoA := by
  unfold rhoA
  have : DotLoop.stepRel Device.relDot16 K ≤ DotLoop.stepRel Device.relDot16 32 := by
    unfold DotLoop.stepRel Device.relDot16
    have hu := Format.u_nonneg (f := Format.binary32)
    have hK' : (K : ℚ) ≤ 32 := by exact_mod_cast hK
    have h2 : ((K : ℚ) + 2) * 2 ^ (-22 : ℤ) ≤ ((32 : ℕ) + 2) * 2 ^ (-22 : ℤ) := by
      push_cast
      exact mul_le_mul_of_nonneg_right (by linarith) (by positivity)
    nlinarith
  exact_mod_cast this

/-- The absolute constant of one accumulation step over `K ≤ 32` products is at most `alphaA`. -/
theorem stepAbs_le {K : ℕ} (hK : K ≤ 32) :
    ((DotLoop.stepAbs Device.absDot16 K : ℚ) : ℝ) ≤ alphaA := by
  unfold alphaA
  have : DotLoop.stepAbs Device.absDot16 K ≤ DotLoop.stepAbs Device.absDot16 32 := by
    unfold DotLoop.stepAbs Device.absDot16
    have hu := Format.u_nonneg (f := Format.binary32)
    have hK' : (K : ℚ) ≤ 32 := by exact_mod_cast hK
    have h2 : ((K : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ ((32 : ℕ) + 2) * 2 ^ (-126 : ℤ) := by
      push_cast
      exact mul_le_mul_of_nonneg_right (by linarith) (by positivity)
    nlinarith
  exact_mod_cast this

/-- **One lane of `acc += tl.dot(a, b)`** (either form of `tl.dot_add`), with `K ≤ 32` products: if
the accumulator lane has value `v` (`|v| ≤ 2^48`), the operands are finite and the sum of the
products' lifted magnitudes is at most `Anew ≤ 2^40`, the result lane is finite and within
`rhoA·(|v| + Anew) + alphaA` of `v + Σ a·b`. -/
theorem lane_update {P : Platform} {fusion : Bool} (hdot : Device.DotF16 P) {M N K : Nat}
    (hK : K ≤ 32) {A : Tensor [M, K] F16} {B : Tensor [K, N] F16} {c z r : Tensor [M, N] F32}
    (hr : P.dot dotCfg A B c r ∨
      ∃ d, P.dot dotCfg A B z d ∧ ∀ i, AccAdd fusion .f32 (d.get i) (c.get i) (r.get i))
    (i : Fin M) (j : Fin N) (hz : DType.val? .f32 (z.get2 i j) = some 0) {v : ℚ}
    (hv : DType.val? .f32 (c.get2 i j) = some v) (hvb : |v| ≤ 2 ^ 48)
    (hAf : ∀ k : Fin K, (A.get2 i k).isFinite = true)
    (hBf : ∀ k : Fin K, (B.get2 k j).isFinite = true) {Anew : ℝ}
    (hAnew : ∑ k : Fin K, lm (A.get2 i k) * lm (B.get2 k j) ≤ Anew) (hAnew2 : Anew ≤ 2 ^ 40) :
    ∃ v' : ℚ, DType.val? .f32 (r.get2 i j) = some v' ∧
      |(v' : ℝ) - ((v : ℝ) + ∑ k : Fin K, (A.get2 i k).toReal * (B.get2 k j).toReal)| ≤
        rhoA * (|(v : ℝ)| + Anew) + alphaA := by
  have hva : ∀ k : Fin K, DType.val? .f16 (A.get2 i k) = some (A.get2 i k).toRat := fun k =>
    DType.val?_f16.mpr ⟨hAf k, rfl⟩
  have hvb' : ∀ k : Fin K, DType.val? .f16 (B.get2 k j) = some (B.get2 k j).toRat := fun k =>
    DType.val?_f16.mpr ⟨hBf k, rfl⟩
  obtain ⟨hcfin, hcv⟩ := DType.val?_f32.mp hv
  have hLq : ∑ k : Fin K, Device.liftMag .f16 (A.get2 i k).toRat * Device.liftMag .f16 (B.get2 k j).toRat
      ≤ 2 ^ 40 := by
    have h1 : ((∑ k : Fin K, Device.liftMag .f16 (A.get2 i k).toRat *
        Device.liftMag .f16 (B.get2 k j).toRat : ℚ) : ℝ) ≤ 2 ^ 40 := by
      push_cast
      exact le_trans hAnew hAnew2
    exact_mod_cast h1
  have hLnn : 0 ≤ ∑ k : Fin K, Device.liftMag .f16 (A.get2 i k).toRat *
      Device.liftMag .f16 (B.get2 k j).toRat :=
    Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hKq : (K : ℚ) ≤ 32 := by exact_mod_cast hK
  have hrel : Device.relDot16 K ≤ 1 := by
    unfold Device.relDot16
    have : (2 : ℚ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
    rw [this]; linarith
  have hrel0 : 0 ≤ Device.relDot16 K := by unfold Device.relDot16; positivity
  have habs0 : 0 ≤ Device.absDot16 K := by unfold Device.absDot16; positivity
  have habs : Device.absDot16 K ≤ 1 := by
    unfold Device.absDot16
    have h : (2 : ℚ) ^ (-126 : ℤ) ≤ 1 / 4194304 := by norm_num
    calc ((K : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ 34 * (1 / 4194304) :=
          mul_le_mul (by linarith) h (by positivity) (by norm_num)
      _ ≤ 1 := by norm_num
  have hvq : |v| ≤ 2 ^ 48 := hvb
  have hroom : 2 * ((1 + Device.relDot16 K) * (|(c.get2 i j).toRat| + ∑ k : Fin K,
      Device.liftMag .f16 (A.get2 i k).toRat * Device.liftMag .f16 (B.get2 k j).toRat) +
      Device.absDot16 K + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
    rw [hcv, Format.binary32_maxVal, Format.binary32_eta]
    have h0 : 0 ≤ |v| + ∑ k : Fin K, Device.liftMag .f16 (A.get2 i k).toRat *
        Device.liftMag .f16 (B.get2 k j).toRat := by positivity
    have h1 : |v| + ∑ k : Fin K, Device.liftMag .f16 (A.get2 i k).toRat *
        Device.liftMag .f16 (B.get2 k j).toRat ≤ 2 ^ 49 := by
      have : (2 : ℚ) ^ 40 ≤ 2 ^ 48 := by norm_num
      linarith
    have h2 : (1 + Device.relDot16 K) * (|v| + ∑ k : Fin K, Device.liftMag .f16 (A.get2 i k).toRat *
        Device.liftMag .f16 (B.get2 k j).toRat) ≤ 2 * 2 ^ 49 :=
      mul_le_mul (by linarith) h1 h0 (by norm_num)
    have : (2 : ℚ) ^ 50 + 2 * 1 + 4 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    have he : (2 : ℚ) * 2 ^ (-150 : ℤ) ≤ 1 := by norm_num
    nlinarith
  obtain ⟨hrfin, hrbound⟩ := DotLift.dotAdd_step (fusion := fusion) (hdot .tf32) hrel0 habs0 hr i j hz
    hva hvb' hcfin hroom
  refine ⟨(r.get2 i j).toRat, DType.val?_f32.mpr ⟨hrfin, rfl⟩, ?_⟩
  have hsr := stepRel_le hK
  have hsa := stepAbs_le hK
  have hb1 := (Rat.cast_le (K := ℝ)).mpr hrbound
  push_cast at hb1
  rw [hcv] at hb1
  have hAn : ∑ k : Fin K, ((Device.liftMag .f16 (A.get2 i k).toRat : ℚ) : ℝ) *
      ((Device.liftMag .f16 (B.get2 k j).toRat : ℚ) : ℝ) ≤ Anew := hAnew
  have hrelnn : (0 : ℝ) ≤ ((DotLoop.stepRel Device.relDot16 K : ℚ) : ℝ) := by
    have hu := Format.u_nonneg (f := Format.binary32)
    have h3 := mul_nonneg (mul_nonneg (by norm_num : (0 : ℚ) ≤ 3) hu) (by linarith : 0 ≤ 1 + Device.relDot16 K)
    have : (0 : ℚ) ≤ DotLoop.stepRel Device.relDot16 K := by
      unfold DotLoop.stepRel; linarith
    exact_mod_cast this
  calc _ ≤ ((DotLoop.stepRel Device.relDot16 K : ℚ) : ℝ) * (|(v : ℝ)| + ∑ k : Fin K,
          ((Device.liftMag .f16 (A.get2 i k).toRat : ℚ) : ℝ) *
            ((Device.liftMag .f16 (B.get2 k j).toRat : ℚ) : ℝ)) +
        ((DotLoop.stepAbs Device.absDot16 K : ℚ) : ℝ) := by
        simpa [FloatV.toReal] using hb1
    _ ≤ rhoA * (|(v : ℝ)| + Anew) + alphaA := by
        have h0 : (0 : ℝ) ≤ |(v : ℝ)| := abs_nonneg _
        have h1 : (|(v : ℝ)| + ∑ k : Fin K, ((Device.liftMag .f16 (A.get2 i k).toRat : ℚ) : ℝ) *
            ((Device.liftMag .f16 (B.get2 k j).toRat : ℚ) : ℝ)) ≤ |(v : ℝ)| + Anew := by linarith
        have h2 : 0 ≤ |(v : ℝ)| + ∑ k : Fin K, ((Device.liftMag .f16 (A.get2 i k).toRat : ℚ) : ℝ) *
            ((Device.liftMag .f16 (B.get2 k j).toRat : ℚ) : ℝ) := by
          have : 0 ≤ ∑ k : Fin K, ((Device.liftMag .f16 (A.get2 i k).toRat : ℚ) : ℝ) *
            ((Device.liftMag .f16 (B.get2 k j).toRat : ℚ) : ℝ) :=
            Finset.sum_nonneg fun k _ => mul_nonneg (by exact_mod_cast Device.liftMag_nonneg _ _)
              (by exact_mod_cast Device.liftMag_nonneg _ _)
          linarith
        have h3 := mul_le_mul hsr h1 h2 rhoA_nonneg
        linarith

/-- A zero accumulator satisfies the invariant with no steps. -/
theorem AccInv.zero {M N : Nat} (fam : Fam) (rows : Fin M → Nat) (cols : Fin N → Nat)
    {z : Tensor [M, N] F32} (hz : ∀ i j, DType.val? .f32 (z.get2 i j) = some 0) :
    AccInv fam rows cols ∅ 0 z := fun r =>
  ⟨fun _ => FloatV.ofPat 0, by simp, fun j => ⟨0, hz r j, by simp [accB_zero]⟩⟩

/-- A sum over a range of consecutive indices is a sum over `Fin K`. -/
theorem sum_ico_fin (f : Nat → ℝ) (a K : Nat) :
    ∑ s ∈ Finset.Ico a (a + K), f s = ∑ k : Fin K, f (a + k) := by
  rw [Finset.sum_Ico_eq_sum_range, Nat.add_sub_cancel_left,
    ← Fin.sum_univ_eq_sum_range (fun k => f (a + k)) K]

/-- **One accumulation step keeps the invariant**: the tile `A` of computed coefficients (rows
`rows i`, indices `a, …, a + K - 1`) times the values `B` of the family, added to the accumulator
`c` by `tl.dot_add`, gives an accumulator after `t + 1` steps over the indices `S ∪ [a, a + K)`. -/
theorem AccInv.step {P : Platform} {fusion : Bool} (hdot : Device.DotF16 P) {M N K : Nat}
    (hK : K ≤ 32) (fam : Fam) (rows : Fin M → Nat) (cols : Fin N → Nat) {S : Finset Nat}
    {t a : Nat} (hdisj : ∀ s ∈ S, s < a ∨ a + K ≤ s) (hS : ∀ s ∈ S, s < 2 ^ 14)
    (ht : t < 1024)
    {A : Tensor [M, K] F16} {B : Tensor [K, N] F16} {c z r : Tensor [M, N] F32}
    (hr : P.dot dotCfg A B c r ∨
      ∃ d, P.dot dotCfg A B z d ∧ ∀ i, AccAdd fusion .f32 (d.get i) (c.get i) (r.get i))
    (hz : ∀ i j, DType.val? .f32 (z.get2 i j) = some 0)
    (hc : AccInv fam rows cols S t c)
    (hA : ∀ (i : Fin M) (k : Fin K), (A.get2 i k).isFinite = true ∧
      |(A.get2 i k).toReal - fam.wt (rows i) (a + k)| ≤ fam.err (rows i) (a + k) ∧
      lm (A.get2 i k) ≤ fam.lb (rows i) (a + k))
    (hB : ∀ (k : Fin K) (j : Fin N), B.get2 k j = fam.u (a + k) (cols j))
    (hu : ∀ s, (s ∈ S ∨ (a ≤ s ∧ s < a + K)) → ∀ j : Fin N, (fam.u s (cols j)).isFinite = true)
    (hlb : ∀ (i : Fin M) s, (s ∈ S ∨ (a ≤ s ∧ s < a + K)) → fam.lb (rows i) s ≤ 2 ^ 16) :
    AccInv fam rows cols (S ∪ Finset.Ico a (a + K)) (t + 1) r := by
  intro i
  obtain ⟨cv, hcv, hlane⟩ := hc i
  have hcard : S.card ≤ 2 ^ 14 :=
    calc S.card ≤ (Finset.range (2 ^ 14)).card :=
          Finset.card_le_card fun s hs => Finset.mem_range.mpr (hS s hs)
      _ = 2 ^ 14 := Finset.card_range _
  let cv' : Nat → F16 := fun s =>
    if h : a ≤ s ∧ s < a + K then A.get2 i ⟨s - a, by omega⟩ else cv s
  have hcv'S : ∀ s ∈ S, cv' s = cv s := by
    intro s hs
    simp only [cv']
    rw [dite_eq_right]
    rcases hdisj s hs with h | h <;> omega
  have hcv'I : ∀ k : Fin K, cv' (a + k) = A.get2 i k := by
    intro k
    simp only [cv']
    rw [dite_eq_left ⟨by omega, by have := k.isLt; omega⟩]
    congr 2
    omega
  have hdisjI : Disjoint S (Finset.Ico a (a + K)) := by
    rw [Finset.disjoint_left]
    intro s hs hs'
    have := Finset.mem_Ico.mp hs'
    rcases hdisj s hs with h | h <;> omega
  refine ⟨cv', ?_, fun j => ?_⟩
  · intro s hs
    rcases Finset.mem_union.mp hs with hs | hs
    · rw [hcv'S s hs]; exact hcv s hs
    · obtain ⟨hs1, hs2⟩ := Finset.mem_Ico.mp hs
      obtain ⟨k, hk⟩ : ∃ k : Fin K, s = a + k := ⟨⟨s - a, by omega⟩, by simp; omega⟩
      subst hk
      rw [hcv'I k]; exact hA i k
  · obtain ⟨v, hv, hbd⟩ := hlane j
    have hmem : ∀ k : Fin K, (a ≤ a + k ∧ a + k < a + K) := fun k =>
      ⟨by omega, by have := k.isLt; omega⟩
    -- the pieces
    have hlmu : ∀ s, (s ∈ S ∨ (a ≤ s ∧ s < a + K)) → lm (fam.u s (cols j)) ≤ 65504 :=
      fun s hs => lm_le_65504 (hu s hs j)
    have hlbS : ∀ s ∈ S, 0 ≤ fam.lb (rows i) s := fun s hs =>
      le_trans (lm_nonneg _) (hcv s hs).2.2
    have hlbI : ∀ k : Fin K, 0 ≤ fam.lb (rows i) (a + k) := fun k =>
      le_trans (lm_nonneg _) (hA i k).2.2
    set Bt := ∑ s ∈ S, fam.lb (rows i) s * lm (fam.u s (cols j)) with hBt
    set Anew := ∑ k : Fin K, fam.lb (rows i) (a + k) * lm (fam.u (a + k) (cols j)) with hAnew
    have hterm : ∀ s ∈ S, fam.lb (rows i) s * lm (fam.u s (cols j)) ≤ 2 ^ 32 := by
      intro s hs
      have h1 := hlb i s (Or.inl hs)
      have h2 := hlmu s (Or.inl hs)
      have h3 := lm_nonneg (fam.u s (cols j))
      calc _ ≤ 2 ^ 16 * 65504 := mul_le_mul h1 h2 h3 (by positivity)
        _ ≤ 2 ^ 32 := by norm_num
    have hBt0 : 0 ≤ Bt := Finset.sum_nonneg fun s hs => mul_nonneg (hlbS s hs) (lm_nonneg _)
    have hBt1 : Bt ≤ 2 ^ 46 := by
      calc Bt ≤ ∑ _s ∈ S, (2 : ℝ) ^ 32 := Finset.sum_le_sum hterm
        _ = S.card * 2 ^ 32 := by simp
        _ ≤ 2 ^ 14 * 2 ^ 32 := by
          have : (S.card : ℝ) ≤ 2 ^ 14 := by exact_mod_cast hcard
          exact mul_le_mul_of_nonneg_right this (by positivity)
        _ = 2 ^ 46 := by norm_num
    have hAterm : ∀ k : Fin K, fam.lb (rows i) (a + k) * lm (fam.u (a + k) (cols j)) ≤ 2 ^ 32 := by
      intro k
      have h1 := hlb i (a + k) (Or.inr (hmem k))
      have h2 := hlmu (a + k) (Or.inr (hmem k))
      have h3 := lm_nonneg (fam.u (a + k) (cols j))
      calc _ ≤ 2 ^ 16 * 65504 := mul_le_mul h1 h2 h3 (by positivity)
        _ ≤ 2 ^ 32 := by norm_num
    have hAnew0 : 0 ≤ Anew := Finset.sum_nonneg fun k _ => mul_nonneg (hlbI k) (lm_nonneg _)
    have hAnew1 : Anew ≤ 2 ^ 40 := by
      calc Anew ≤ ∑ _k : Fin K, (2 : ℝ) ^ 32 := Finset.sum_le_sum fun k _ => hAterm k
        _ = K * 2 ^ 32 := by simp
        _ ≤ 32 * 2 ^ 32 := by
          have : (K : ℝ) ≤ 32 := by exact_mod_cast hK
          exact mul_le_mul_of_nonneg_right this (by positivity)
        _ ≤ 2 ^ 40 := by norm_num
    set St := ∑ s ∈ S, (cv s).toReal * (fam.u s (cols j)).toReal with hSt
    have hSt1 : |St| ≤ Bt := by
      refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun s hs => ?_)
      exact (abs_mul_le_lm _ _).trans (mul_le_mul_of_nonneg_right (hcv s hs).2.2 (lm_nonneg _))
    have hv48 : |(v : ℝ)| ≤ 2 ^ 48 := by
      have h1 := accB_le (t := t) (by omega) hBt0
      have h2 : |(v : ℝ)| ≤ |St| + |(v : ℝ) - St| := by
        have := abs_add_le St ((v : ℝ) - St)
        simpa using this
      have h3 : (2 : ℝ) ^ 46 + 1 ≤ 2 ^ 47 := by norm_num
      have h4 : (2 : ℝ) ^ 47 ≤ 2 ^ 48 := by norm_num
      linarith
    have hv48' : |v| ≤ 2 ^ 48 := by exact_mod_cast hv48
    -- the update
    have hAnewle : ∑ k : Fin K, lm (A.get2 i k) * lm (B.get2 k j) ≤ Anew := by
      refine Finset.sum_le_sum fun k _ => ?_
      rw [hB k j]
      exact mul_le_mul_of_nonneg_right (hA i k).2.2 (lm_nonneg _)
    obtain ⟨v', hv', hbd'⟩ := lane_update (fusion := fusion) hdot hK hr i j (hz i j) hv hv48'
      (fun k => (hA i k).1) (fun k => by rw [hB k j]; exact hu _ (Or.inr (hmem k)) j)
      hAnewle hAnew1
    refine ⟨v', hv', ?_⟩
    have hSnew : |∑ k : Fin K, (A.get2 i k).toReal * (B.get2 k j).toReal| ≤ Anew := by
      refine (Finset.abs_sum_le_sum_abs _ _).trans (le_trans ?_ hAnewle)
      exact Finset.sum_le_sum fun k _ => abs_mul_le_lm _ _
    have hstep := accB_step hSt1 hSnew hbd hbd'
    rw [Finset.sum_union hdisjI, Finset.sum_union hdisjI, sum_ico_fin, sum_ico_fin]
    have e1 : ∑ s ∈ S, (cv' s).toReal * (fam.u s (cols j)).toReal = St :=
      Finset.sum_congr rfl fun s hs => by rw [hcv'S s hs]
    have e2 : ∑ k : Fin K, (cv' (a + k)).toReal * (fam.u (a + k) (cols j)).toReal =
        ∑ k : Fin K, (A.get2 i k).toReal * (B.get2 k j).toReal :=
      Finset.sum_congr rfl fun k _ => by rw [hcv'I k, hB k j]
    rw [e1, e2]
    exact hstep

/-! ## From the accumulator to `OutSpec` -/

/-- **The accumulator against the exact sum**: the lane is finite and within
`Σ err·|u| + accB t (Σ lb·lm u)` of `Σ wt·u`, the sums over `s < N'` (the rows outside `S` carry
nothing). -/
theorem AccInv.final {M N : Nat} {fam : Fam} {rows : Fin M → Nat} {cols : Fin N → Nat}
    {S : Finset Nat} {t : Nat} {acc : Tensor [M, N] F32} (h : AccInv fam rows cols S t acc)
    (r : Fin M) (j : Fin N) (N' : Nat) (hS : ∀ s ∈ S, s < N')
    (hz : ∀ s < N', s ∉ S →
      fam.wt (rows r) s = 0 ∧ fam.err (rows r) s = 0 ∧ fam.lb (rows r) s = 0) :
    (acc.get2 r j).isFinite = true ∧
      |(acc.get2 r j).toReal - ∑ s ∈ Finset.range N', fam.wt (rows r) s * (fam.u s (cols j)).toReal| ≤
        ∑ s ∈ Finset.range N', fam.err (rows r) s * |(fam.u s (cols j)).toReal| +
          accB t (∑ s ∈ Finset.range N', fam.lb (rows r) s * lm (fam.u s (cols j))) := by
  obtain ⟨cv, hcv, hlane⟩ := h r
  obtain ⟨v, hv, hb⟩ := hlane j
  obtain ⟨hfin, hval⟩ := DType.val?_f32.mp hv
  refine ⟨hfin, ?_⟩
  have hsub : S ⊆ Finset.range N' := fun s hs => Finset.mem_range.mpr (hS s hs)
  have e1 : ∑ s ∈ S, fam.wt (rows r) s * (fam.u s (cols j)).toReal =
      ∑ s ∈ Finset.range N', fam.wt (rows r) s * (fam.u s (cols j)).toReal :=
    Finset.sum_subset hsub fun s hs hn => by
      rw [(hz s (Finset.mem_range.mp hs) hn).1]; simp
  have e2 : ∑ s ∈ S, fam.err (rows r) s * |(fam.u s (cols j)).toReal| =
      ∑ s ∈ Finset.range N', fam.err (rows r) s * |(fam.u s (cols j)).toReal| :=
    Finset.sum_subset hsub fun s hs hn => by
      rw [(hz s (Finset.mem_range.mp hs) hn).2.1]; simp
  have e3 : ∑ s ∈ S, fam.lb (rows r) s * lm (fam.u s (cols j)) =
      ∑ s ∈ Finset.range N', fam.lb (rows r) s * lm (fam.u s (cols j)) :=
    Finset.sum_subset hsub fun s hs hn => by
      rw [(hz s (Finset.mem_range.mp hs) hn).2.2]; simp
  rw [← e1, ← e2, ← e3]
  have hcast : (acc.get2 r j).toReal = (v : ℝ) := by unfold FloatV.toReal; rw [hval]
  rw [hcast]
  have h1 : |∑ s ∈ S, (cv s).toReal * (fam.u s (cols j)).toReal -
      ∑ s ∈ S, fam.wt (rows r) s * (fam.u s (cols j)).toReal| ≤
      ∑ s ∈ S, fam.err (rows r) s * |(fam.u s (cols j)).toReal| := by
    rw [← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun s hs => ?_)
    rw [← sub_mul, abs_mul]
    exact mul_le_mul_of_nonneg_right (hcv s hs).2.1 (abs_nonneg _)
  have h2 := abs_sub_le (v : ℝ) (∑ s ∈ S, (cv s).toReal * (fam.u s (cols j)).toReal)
    (∑ s ∈ S, fam.wt (rows r) s * (fam.u s (cols j)).toReal)
  linarith

/-- **The stored value** (the `float32` accumulator rounded to `float16`): if `v` is finite and
within `T` of `E`, then `c` (up to NaN payload the rounding of `v`) is within `T` plus one
`float16` rounding of `E`, when there is room in `float16`. -/
theorem out_of_f32 {v : F32} {c : F16} {E T : ℝ} (hv : v.isFinite = true) (hT : |v.toReal - E| ≤ T)
    (hc : FloatV.NaNEq (FloatV.cvt v) c) :
    ∃ v' : F32, v'.isFinite = true ∧ |v'.toReal - E| ≤ T ∧ FloatV.NaNEq (FloatV.cvt v') c ∧
      (|E| + T ≤ 65504 →
        c.isFinite = true ∧ |c.toReal - E| ≤ T + (|E| + T) * u16 + eta16) := by
  refine ⟨v, hv, hT, hc, fun hroom => ?_⟩
  have hvE : |v.toReal| ≤ |E| + T := by
    have := abs_sub_abs_le_abs_sub v.toReal E
    linarith
  have hmax : |v.toReal| ≤ ((Format.binary16.maxVal : ℚ) : ℝ) := by
    rw [Format.binary16_maxVal_real]; linarith
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err_real (g := Format.binary16) Format.binary16_regular hv hmax
  have hcv : c = FloatV.cvt v := hc.eq_of_isFinite hfin
  rw [hcv]
  refine ⟨hfin, ?_⟩
  have h1 : |v.toReal| * u16 ≤ (|E| + T) * u16 := mul_le_mul_of_nonneg_right hvE u16_nonneg
  have h2 := abs_sub_le (FloatV.cvt v : F16).toReal v.toReal E
  have hu : ((Format.binary16.u : ℚ) : ℝ) = u16 := rfl
  have he : ((Format.binary16.eta : ℚ) : ℝ) = eta16 := rfl
  rw [hu, he] at herr
  linarith

/-- **A product by a scalar** (`dk * sm_scale`, `dq * LN2`): for `v` within `T` of `S`, the IEEE
product `w` with a finite `s` of magnitude at most `2^16` is within
`|s|·T + u32·|s|·(|S| + T) + η32` of `s·S`. -/
theorem scale_tol {v s w : F32} (hv : v.isFinite = true) (hs : s.isFinite = true) {S T : ℝ}
    (hT : |v.toReal - S| ≤ T) (hS : |S| + T ≤ 2 ^ 100) (hs16 : |s.toReal| ≤ 2 ^ 16)
    (hw : FloatV.NaNEq (FloatV.mul v s) w) :
    w.isFinite = true ∧
      |w.toReal - s.toReal * S| ≤ |s.toReal| * T + u32 * (|s.toReal| * (|S| + T)) + eta32 := by
  have hT0 : 0 ≤ T := (abs_nonneg _).trans hT
  have hvS : |v.toReal| ≤ |S| + T := by
    have := abs_sub_abs_le_abs_sub v.toReal S
    linarith
  have hprod : |v.toReal * s.toReal| ≤ |s.toReal| * (|S| + T) := by
    rw [abs_mul, mul_comm]
    exact mul_le_mul_of_nonneg_left hvS (abs_nonneg _)
  have hmax : |v.toReal * s.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal]
    push_cast
    have : |s.toReal| * (|S| + T) ≤ 2 ^ 16 * 2 ^ 100 :=
      mul_le_mul hs16 hS (add_nonneg (abs_nonneg _) hT0) (by positivity)
    have h3 : (2 : ℝ) ^ 16 * 2 ^ 100 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hfin, herr⟩ := FloatV.mul_err_real Format.binary32_regular hv hs hmax
  have hwe : w = FloatV.mul v s := hw.eq_of_isFinite hfin
  rw [hwe]
  refine ⟨hfin, ?_⟩
  have h1 : |v.toReal * s.toReal| * u32 ≤ |s.toReal| * (|S| + T) * u32 :=
    mul_le_mul_of_nonneg_right hprod u32_nonneg
  have h2 : |s.toReal * (v.toReal - S)| ≤ |s.toReal| * T := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hT (abs_nonneg _)
  have h3 := abs_sub_le (FloatV.mul v s).toReal (v.toReal * s.toReal) (s.toReal * S)
  have e : v.toReal * s.toReal - s.toReal * S = s.toReal * (v.toReal - S) := by ring
  rw [e] at h3
  have hu : ((Format.binary32.u : ℚ) : ℝ) = u32 := rfl
  have he : ((Format.binary32.eta : ℚ) : ℝ) = eta32 := rfl
  rw [hu, he] at herr
  nlinarith

end Claims.AttnBwd
