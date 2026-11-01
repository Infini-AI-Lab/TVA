import AttnBwd.Lane

/-!
# `attn_bwd_f16`: `tl.dot`, the accumulation and the outputs

* `lm_le_lift16`, `lm_le`, `abs_mul_le_lm`: lifted magnitudes (`lm`), which `Rules.Num.DotF16`
  counts since library round 25; a `float16` value below `2^-24` is `0` (`f16_small_zero`).
* `dot0`: one element of a `tl.dot` from the zero accumulator (lines 293, 307, 345, 353), by
  `Rules.Num.DotF16`, within `relDot16·Σ lm a·lm b + absDot16`.
* `AccI`: what an accumulator element satisfies after `T` steps: it is finite and within
  `accB T Bt` of the exact sum `St` of the products of its `float16` operands, and `St` is within
  `Er` of the exact value `Ex`, with `|St| ≤ Bt ≤ Bm`.
* `accI_dotAdd`: one step `acc += tl.dot(a, b)` over `16` or `32` products (lines 303, 310, 358),
  by `Rules.Num.DotF16` (`DotLift.dotAdd_step`), with the products' lifted magnitudes.
* `out16`, `scale_out`: the products by `sm_scale` and `LN2` (lines 449, 499) and the roundings to
  `float16` of the stores (lines 446, 451, 500).
-/

namespace Kernels.AttnBwd

open TVA
open TVA.Math.Float

/-! ## `float16` values -/

theorem f16_abs_le {x : F16} (hx : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have := FloatV.abs_toRat_le_maxVal Format.binary16_regular hx
  rwa [Format.binary16_maxVal] at this

theorem f16_abs_leR {x : F16} (hx : x.isFinite = true) : |x.toReal| ≤ 65504 := by
  rw [toReal_def]
  have := f16_abs_le hx
  have : ((|x.toRat| : ℚ) : ℝ) ≤ ((65504 : ℚ) : ℝ) := by exact_mod_cast this
  push_cast at this; exact this

theorem val_f16 {x : F16} (hx : x.isFinite = true) : DType.val? .f16 x = some x.toRat := by
  show (if x.isFinite = true then some x.toRat else none) = some x.toRat
  rw [ite_eq_left hx]

theorem prod16_le {x y : F16} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    |x.toRat * y.toRat| ≤ 2 ^ 32 := by
  rw [abs_mul]
  have h1 := f16_abs_le hx
  have h2 := f16_abs_le hy
  have : |x.toRat| * |y.toRat| ≤ 65504 * 65504 := mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
  linarith

theorem sum_prod16_le {K : Nat} (va vb : Fin K → F16) (ha : ∀ k, (va k).isFinite = true)
    (hb : ∀ k, (vb k).isFinite = true) : ∑ k, |(va k).toRat * (vb k).toRat| ≤ K * 2 ^ 32 := by
  calc ∑ k, |(va k).toRat * (vb k).toRat| ≤ ∑ _k : Fin K, (2 : ℚ) ^ 32 :=
        Finset.sum_le_sum fun k _ => prod16_le (ha k) (hb k)
    _ = K * 2 ^ 32 := by simp

/-! ## Lifted magnitudes (`Rules.Num.DotF16` since library round 25) -/

theorem liftMag16_le {x : F16} (hx : x.isFinite = true) : Rules.Num.liftMag .f16 x.toRat ≤ 65504 := by
  have h1 := f16_abs_le hx
  unfold Rules.Num.liftMag
  split_ifs
  · norm_num
  · refine max_le h1 ?_
    rw [DotLift.minNormal8_f16]; norm_num

theorem lm_nn (x : F16) : 0 ≤ lm x := by
  unfold lm; exact_mod_cast Dot8Chunk.liftMag_nonneg .f16 x.toRat

theorem abs_le_lm (x : F16) : |x.toReal| ≤ lm x := by
  unfold lm; rw [toReal_def]; exact_mod_cast Dot8Chunk.abs_le_liftMag .f16 x.toRat

theorem abs_mul_le_lm (x y : F16) : |x.toReal * y.toReal| ≤ lm x * lm y := by
  rw [abs_mul]; exact mul_le_mul (abs_le_lm x) (abs_le_lm y) (abs_nonneg _) (lm_nn x)

theorem lm_le {x : F16} (hx : x.isFinite = true) : lm x ≤ 65504 := by
  unfold lm; exact_mod_cast liftMag16_le hx

theorem lm_mul_le {x y : F16} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    lm x * lm y ≤ 2 ^ 32 := by
  have : lm x * lm y ≤ 65504 * 65504 := mul_le_mul (lm_le hx) (lm_le hy) (lm_nn y) (by norm_num)
  linarith

/-- A `float16` value below `2^-24` (the smallest subnormal) in magnitude is `0`. -/
theorem f16_small_zero {c : F16} (h : |c.toRat| < 1 / 16777216) : c.toRat = 0 := by
  by_contra hne
  have h1 : 1 ≤ Format.binary16.mag c.pat := by
    rcases Nat.eq_zero_or_pos (Format.binary16.mag c.pat) with h0 | h0
    · exfalso; apply hne
      show Format.binary16.toRat c.pat = 0
      rw [Format.toRat_eq, h0, Format.magVal_zero]; simp
    · exact h0
  have h2 := Format.magVal_le_magVal (f := Format.binary16) h1
  have h3 : Format.binary16.magVal 1 = 1 / 16777216 := by decide +kernel
  have h4 : |c.toRat| = Format.binary16.magVal (Format.binary16.mag c.pat) := Format.abs_toRat _
  linarith

/-- A `float16` value of magnitude at most `B` has lifted magnitude at most `lift16 B`. -/
theorem lm_le_lift16 {c : F16} {B : ℝ} (hc : |c.toReal| ≤ B) : lm c ≤ lift16 B := by
  unfold lift16
  split_ifs with hB
  · have h1 : |c.toReal| < 2 ^ (-24 : ℤ) := lt_of_le_of_lt hc hB
    rw [toReal_def, show (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 by norm_num] at h1
    have h2 : |c.toRat| < 1 / 16777216 := by
      have h3 : ((|c.toRat| : ℚ) : ℝ) < ((1 / 16777216 : ℚ) : ℝ) := by push_cast; exact h1
      exact Rat.cast_lt.mp h3
    unfold lm; rw [f16_small_zero h2, Dot8Chunk.liftMag_zero]; simp
  · unfold lm Rules.Num.liftMag
    split_ifs with h0
    · push_cast; exact le_max_of_le_right (by positivity)
    · push_cast
      refine max_le_max ?_ ?_
      · rw [toReal_def] at hc; exact hc
      · rw [DotLift.minNormal8_f16]; push_cast; exact le_rfl

theorem lift16_nn (B : ℝ) : 0 ≤ lift16 B := by
  unfold lift16; split_ifs
  · exact le_rfl
  · exact le_max_of_le_right (by positivity)

theorem lift16_le {B : ℝ} (h : B ≤ 2 ^ 17) : lift16 B ≤ 2 ^ 17 := by
  unfold lift16; split_ifs
  · norm_num
  · exact max_le h (by norm_num)

theorem lift16_zero : lift16 0 = 0 := by
  have h : (0 : ℝ) < 2 ^ (-24 : ℤ) := by positivity
  unfold lift16; simp only [h, ite_true]

/-! ## A `tl.dot` from the zero accumulator -/

/-- One element of `tl.dot(a, b)` (zero accumulator) of finite `float16` operands over `K ≤ 256`
products: finite, within `relDot16 K·Σ lm a·lm b + absDot16 K` of `Σ a·b`. -/
theorem dot0 {P : Platform} (hdot : Rules.Num.DotF16 P) {M N K : Nat} (hK : K ≤ 256)
    {a : Tensor [M, K] F16} {b : Tensor [K, N] F16} {r : Tensor [M, N] F32}
    (hr : P.dot dotCfg a b (tl.zeros [M, N] .f32) r) (i : Fin M) (n : Fin N)
    (ha : ∀ k, (a.get2 i k).isFinite = true) (hb : ∀ k, (b.get2 k n).isFinite = true) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toReal - ∑ k : Fin K, (a.get2 i k).toReal * (b.get2 k n).toReal| ≤
        (Rules.Num.relDot16 K : ℝ) * ∑ k : Fin K, lm (a.get2 i k) * lm (b.get2 k n) +
          (Rules.Num.absDot16 K : ℝ) := by
  have hsum : ∑ k, Rules.Num.liftMag .f16 (a.get2 i k).toRat * Rules.Num.liftMag .f16 (b.get2 k n).toRat ≤
      K * 2 ^ 32 := by
    calc _ ≤ ∑ _k : Fin K, (2 : ℚ) ^ 32 := Finset.sum_le_sum fun k _ => by
          have := mul_le_mul (liftMag16_le (ha k)) (liftMag16_le (hb k))
            (Dot8Chunk.liftMag_nonneg _ _) (by norm_num)
          linarith
      _ = K * 2 ^ 32 := by simp
  have hK' : (K : ℚ) ≤ 256 := by exact_mod_cast hK
  have hrel : Rules.Num.relDot16 K ≤ 1 := by
    unfold Rules.Num.relDot16
    have h2 : (0 : ℚ) ≤ 2 ^ (-22 : ℤ) := by positivity
    have : ((K : ℚ) + 2) * 2 ^ (-22 : ℤ) ≤ 258 * 2 ^ (-22 : ℤ) := mul_le_mul_of_nonneg_right (by linarith) h2
    have : (258 : ℚ) * 2 ^ (-22 : ℤ) ≤ 1 := by norm_num
    linarith
  have hrel0 : 0 ≤ Rules.Num.relDot16 K := by unfold Rules.Num.relDot16; positivity
  have habs : Rules.Num.absDot16 K ≤ 1 := by
    unfold Rules.Num.absDot16
    have h2 : (0 : ℚ) ≤ 2 ^ (-126 : ℤ) := by positivity
    have : ((K : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ 258 * 2 ^ (-126 : ℤ) := mul_le_mul_of_nonneg_right (by linarith) h2
    have : (258 : ℚ) * 2 ^ (-126 : ℤ) ≤ 1 := by norm_num
    linarith
  have hA0 : 0 ≤ ∑ k, Rules.Num.liftMag .f16 (a.get2 i k).toRat * Rules.Num.liftMag .f16 (b.get2 k n).toRat :=
    Finset.sum_nonneg fun _ _ => mul_nonneg (Dot8Chunk.liftMag_nonneg _ _) (Dot8Chunk.liftMag_nonneg _ _)
  obtain ⟨vr, hvr, herr⟩ := hdot .tf32 a b (tl.zeros [M, N] .f32) r hr i n
    (fun k => (a.get2 i k).toRat) (fun k => (b.get2 k n).toRat) 0
    (fun k => val_f16 (ha k)) (fun k => val_f16 (hb k)) DotLoop.zero_val (by
      simp only [DType.bound, abs_zero, zero_add]
      rw [Format.binary32_maxVal]
      have : (K : ℚ) * 2 ^ 32 ≤ 2 ^ 40 := by nlinarith
      nlinarith)
  obtain ⟨hfin, hval⟩ := DotLoop.val?_f32.mp hvr
  refine ⟨hfin, ?_⟩
  rw [← hval] at herr
  simp only [abs_zero, zero_add] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  simpa [toReal_def, lm] using this

/-! ## The accumulation -/

/-- `accB` is nonnegative and grows with `B`. -/
theorem accB_nn (T : ℕ) {B : ℝ} (hB : 0 ≤ B) : 0 ≤ accB T B := by
  unfold accB
  have : 1 ≤ (1 + rhoA) ^ T := one_le_pow₀ (by linarith [rhoA_nn])
  have : 0 ≤ ((1 + rhoA) ^ T - 1) * B := mul_nonneg (by linarith) hB
  have := alphaA_nn
  positivity

theorem accB_mono (T : ℕ) {B B' : ℝ} (h : B ≤ B') : accB T B ≤ accB T B' := by
  unfold accB
  have : 1 ≤ (1 + rhoA) ^ T := one_le_pow₀ (by linarith [rhoA_nn])
  nlinarith

/-- `(1 + rhoA)^T ≤ 1 + 2^-5` for `T ≤ 1024`. -/
theorem pow_rhoA_le {T : ℕ} (hT : T ≤ 1024) : (1 + rhoA) ^ T ≤ 1 + 2 ^ (-5 : ℤ) := by
  have h0 : (0 : ℚ) ≤ DotLoop.stepRel Rules.Num.relDot16 32 := by
    have := rhoA_nn; unfold rhoA at this; exact_mod_cast this
  have hleQ : DotLoop.stepRel Rules.Num.relDot16 32 ≤ 2 ^ (-16 : ℤ) := by
    unfold DotLoop.stepRel Rules.Num.relDot16; rw [Format.binary32_u]; norm_num
  have hT' : (T : ℚ) ≤ 1024 := by exact_mod_cast hT
  have hTρ : (T : ℚ) * DotLoop.stepRel Rules.Num.relDot16 32 ≤ 1 / 2 := by
    have : (T : ℚ) * DotLoop.stepRel Rules.Num.relDot16 32 ≤ 1024 * 2 ^ (-16 : ℤ) :=
      mul_le_mul hT' hleQ h0 (by norm_num)
    refine this.trans ?_; norm_num
  have hp := DotLoop.pow_le h0 T hTρ
  have hp' : ((1 + DotLoop.stepRel Rules.Num.relDot16 32) ^ T : ℚ) ≤ 1 + 2 ^ (-5 : ℤ) := by
    refine hp.trans ?_
    have : 2 * (T : ℚ) * DotLoop.stepRel Rules.Num.relDot16 32 ≤ 2 * 1024 * 2 ^ (-16 : ℤ) := by
      have := mul_le_mul hT' hleQ h0 (by norm_num); linarith
    have e : (2 : ℚ) * 1024 * 2 ^ (-16 : ℤ) = 2 ^ (-5 : ℤ) := by norm_num
    linarith
  have : (((1 + DotLoop.stepRel Rules.Num.relDot16 32) ^ T : ℚ) : ℝ) ≤ ((1 + 2 ^ (-5 : ℤ) : ℚ) : ℝ) := by
    exact_mod_cast hp'
  push_cast at this
  unfold rhoA; exact this

/-- A crude bound: `accB T B ≤ 2^-5·B + 2^-85` for `T ≤ 1024`. -/
theorem accB_le {T : ℕ} (hT : T ≤ 1024) {B : ℝ} (hB : 0 ≤ B) : accB T B ≤ 2 ^ (-5 : ℤ) * B + 2 ^ (-85 : ℤ) := by
  unfold accB
  have hp := pow_rhoA_le hT
  have h1 : 1 ≤ (1 + rhoA) ^ T := one_le_pow₀ (by linarith [rhoA_nn])
  have hT' : (T : ℝ) ≤ 1024 := by exact_mod_cast hT
  have hα := alphaA_le
  have hα0 := alphaA_nn
  have e1 : ((1 + rhoA) ^ T - 1) * B ≤ 2 ^ (-5 : ℤ) * B := mul_le_mul_of_nonneg_right (by linarith) hB
  have e2 : (T : ℝ) * (1 + rhoA) ^ T * alphaA ≤ 1024 * 2 * 2 ^ (-100 : ℤ) := by
    have : (1 + rhoA) ^ T ≤ 2 := by
      have : (2 : ℝ) ^ (-5 : ℤ) ≤ 1 := by norm_num
      linarith
    have h3 : (T : ℝ) * (1 + rhoA) ^ T ≤ 1024 * 2 := mul_le_mul hT' this (by linarith) (by norm_num)
    exact mul_le_mul h3 hα hα0 (by norm_num)
  have : (1024 : ℝ) * 2 * 2 ^ (-100 : ℤ) ≤ 2 ^ (-85 : ℤ) := by norm_num
  linarith

/-- One step of the accumulation over the reals (`DotLoop.accum_step`). -/
theorem accB_step {T : ℕ} {a a' St S Bt A : ℝ} (hSt : |St| ≤ Bt) (hSA : |S| ≤ A)
    (he : |a - St| ≤ accB T Bt) (hs : |a' - (a + S)| ≤ rhoA * (|a| + A) + alphaA) :
    |a' - (St + S)| ≤ accB (T + 1) (Bt + A) := by
  have hρ := rhoA_nn
  have hα := alphaA_nn
  have hA0 : 0 ≤ A := (abs_nonneg _).trans hSA
  have hacc : |a| ≤ Bt + accB T Bt := by
    have := abs_sub_abs_le_abs_sub a St
    linarith
  have htri : |a' - (St + S)| ≤ |a' - (a + S)| + |a - St| := by
    have e2 : a' - (St + S) = (a' - (a + S)) + (a - St) := by ring
    rw [e2]; exact abs_add_le _ _
  set Pw := (1 + rhoA) ^ T with hP
  have hP1 : 1 ≤ Pw := one_le_pow₀ (by linarith)
  unfold accB at he hacc ⊢
  rw [← hP] at he hacc
  rw [pow_succ, ← hP]
  push_cast
  have hBt : 0 ≤ Bt := (abs_nonneg _).trans hSt
  have hm1 : rhoA * |a| ≤ rhoA * (Bt + ((Pw - 1) * Bt + T * Pw * alphaA)) := mul_le_mul_of_nonneg_left hacc hρ
  nlinarith [mul_nonneg (mul_nonneg hρ (by linarith : (0 : ℝ) ≤ Pw - 1)) hA0,
    mul_nonneg (by linarith : (0 : ℝ) ≤ Pw - 1) hA0, mul_nonneg hρ hA0, mul_nonneg hρ hα,
    mul_nonneg (mul_nonneg hρ (by linarith : (0 : ℝ) ≤ Pw)) hα, mul_nonneg (Nat.cast_nonneg T) hα,
    mul_nonneg (mul_nonneg (Nat.cast_nonneg T) (by linarith : (0 : ℝ) ≤ Pw)) hα]

/-- The accumulator invariant after `T` steps: `a` is finite and within `accB T Bt` of the exact
sum `St` of its operands' products, which is within `Er` of the exact value `Ex`, with
`|St| ≤ Bt ≤ Bm`. -/
def AccI (a : F32) (Ex Er Bm : ℝ) (T : ℕ) : Prop :=
  a.isFinite = true ∧ ∃ St Bt : ℝ, |St - Ex| ≤ Er ∧ |St| ≤ Bt ∧ Bt ≤ Bm ∧ |a.toReal - St| ≤ accB T Bt

/-- `tl.zeros`: `+0.0`, before any step. -/
theorem accI_zero : AccI (FloatV.ofPat 0) 0 0 0 0 := by
  refine ⟨zero32.1, 0, 0, by simp, by simp, le_rfl, ?_⟩
  rw [toReal_def, zero32.2]
  simp [accB]

/-- After the last step: within `Er + accB T Bm` of `Ex`. -/
theorem accI_final {a : F32} {Ex Er Bm : ℝ} {T : ℕ} (h : AccI a Ex Er Bm T) :
    a.isFinite = true ∧ |a.toReal - Ex| ≤ Er + accB T Bm := by
  obtain ⟨hf, St, Bt, h1, h2, h3, h4⟩ := h
  refine ⟨hf, ?_⟩
  have e : a.toReal - Ex = (a.toReal - St) + (St - Ex) := by ring
  rw [e]
  have := accB_mono T h3
  have := abs_add_le (a.toReal - St) (St - Ex)
  linarith

/-- `|a| ≤ 2^51` for an accumulator with `Bm ≤ 2^50`, `T ≤ 1024`. -/
theorem accI_abs {a : F32} {Ex Er Bm : ℝ} {T : ℕ} (h : AccI a Ex Er Bm T) (hT : T ≤ 1024)
    (hBm : Bm ≤ 2 ^ 50) : |a.toReal| ≤ 2 ^ 51 := by
  obtain ⟨_, St, Bt, _, h2, h3, h4⟩ := h
  have hBt0 : 0 ≤ Bt := (abs_nonneg _).trans h2
  have := accB_le hT hBt0
  have := abs_sub_abs_le_abs_sub a.toReal St
  have : (2 : ℝ) ^ (-5 : ℤ) * Bt ≤ 2 ^ (-5 : ℤ) * 2 ^ 50 := mul_le_mul_of_nonneg_left (h3.trans hBm) (by positivity)
  have : (2 : ℝ) ^ 50 + 2 ^ (-5 : ℤ) * 2 ^ 50 + 2 ^ (-85 : ℤ) ≤ 2 ^ 51 := by norm_num
  linarith

theorem relDot16_mono {K K' : ℕ} (h : K ≤ K') : Rules.Num.relDot16 K ≤ Rules.Num.relDot16 K' := by
  unfold Rules.Num.relDot16
  have : (K : ℚ) ≤ K' := by exact_mod_cast h
  have : (0 : ℚ) < 2 ^ (-22 : ℤ) := by positivity
  nlinarith

theorem absDot16_mono {K K' : ℕ} (h : K ≤ K') : Rules.Num.absDot16 K ≤ Rules.Num.absDot16 K' := by
  unfold Rules.Num.absDot16
  have : (K : ℚ) ≤ K' := by exact_mod_cast h
  have : (0 : ℚ) < 2 ^ (-126 : ℤ) := by positivity
  nlinarith

/-- **One step `acc += tl.dot(a, b)`** over `K ∈ {16, 32}` products, for one element `(i, j)`:
whether the device ran `tl.dot(a, b, acc)` or `acc + tl.dot(a, b)` (`tl.dot_add`), the invariant
moves to the new products `a[i, k]·b[k, j]`, each within `er k` of `ex k` and of lifted magnitude
`lm a[i, k]·lm b[k, j]` at most `bm k`. -/
theorem accI_dotAdd {P : Platform} (hdot : Rules.Num.DotF16 P) {fusion : Bool} {M N K : Nat}
    (hK : K = 16 ∨ K = 32) {A : Tensor [M, K] F16} {B : Tensor [K, N] F16} {c r : Tensor [M, N] F32}
    (hr : P.dot dotCfg A B c r ∨ ∃ e, P.dot dotCfg A B (Tensor.full (dotCfg.acc.ofBits 0)) e ∧
        ∀ ix, AccAdd fusion dotCfg.acc (e.get ix) (c.get ix) (r.get ix))
    (i : Fin M) (j : Fin N) (hA : ∀ k, (A.get2 i k).isFinite = true)
    (hB : ∀ k, (B.get2 k j).isFinite = true)
    {Ex Er Bm : ℝ} {T : ℕ} (hT : T < 1024) (hinv : AccI (c.get2 i j) Ex Er Bm T)
    (ex er bm : Fin K → ℝ)
    (hx : ∀ k, |(A.get2 i k).toReal * (B.get2 k j).toReal - ex k| ≤ er k)
    (hb : ∀ k, lm (A.get2 i k) * lm (B.get2 k j) ≤ bm k)
    (hcap : Bm + ∑ k, bm k ≤ 2 ^ 50) :
    AccI (r.get2 i j) (Ex + ∑ k, ex k) (Er + ∑ k, er k) (Bm + ∑ k, bm k) (T + 1) := by
  have hK32 : K ≤ 32 := by omega
  have hbm0 : 0 ≤ ∑ k, bm k := Finset.sum_nonneg fun k _ => (mul_nonneg (lm_nn _) (lm_nn _)).trans (hb k)
  have hBm : Bm ≤ 2 ^ 50 := by linarith
  have hcabs := accI_abs hinv (by omega) hBm
  obtain ⟨hcf, St, Bt, h1, h2, h3, h4⟩ := hinv
  -- the step, by `DotF16`
  have hsum : ∑ k, Rules.Num.liftMag .f16 (A.get2 i k).toRat * Rules.Num.liftMag .f16 (B.get2 k j).toRat ≤
      K * 2 ^ 32 := by
    calc _ ≤ ∑ _k : Fin K, (2 : ℚ) ^ 32 := Finset.sum_le_sum fun k _ => by
          have := mul_le_mul (liftMag16_le (hA k)) (liftMag16_le (hB k))
            (Dot8Chunk.liftMag_nonneg _ _) (by norm_num)
          linarith
      _ = K * 2 ^ 32 := by simp
  have hK' : (K : ℚ) ≤ 32 := by exact_mod_cast hK32
  have hcQ : |(c.get2 i j).toRat| ≤ 2 ^ 51 := by
    have : ((|(c.get2 i j).toRat| : ℚ) : ℝ) ≤ ((2 ^ 51 : ℚ) : ℝ) := by
      push_cast; simpa [toReal_def] using hcabs
    exact_mod_cast this
  have hrel1 : Rules.Num.relDot16 K ≤ 1 := (relDot16_mono hK32).trans (by unfold Rules.Num.relDot16; norm_num)
  have habs1 : Rules.Num.absDot16 K ≤ 1 := (absDot16_mono hK32).trans (by unfold Rules.Num.absDot16; norm_num)
  have hrel0 : 0 ≤ Rules.Num.relDot16 K := by unfold Rules.Num.relDot16; positivity
  have habs0 : 0 ≤ Rules.Num.absDot16 K := by unfold Rules.Num.absDot16; positivity
  have hA0 : 0 ≤ ∑ k, Rules.Num.liftMag .f16 (A.get2 i k).toRat * Rules.Num.liftMag .f16 (B.get2 k j).toRat :=
    Finset.sum_nonneg fun _ _ => mul_nonneg (Dot8Chunk.liftMag_nonneg _ _) (Dot8Chunk.liftMag_nonneg _ _)
  obtain ⟨hrf, hre⟩ := DotLift.dotAdd_step (hdot .tf32) hrel0 habs0 hr i j
    (va := fun k => (A.get2 i k).toRat) (vb := fun k => (B.get2 k j).toRat)
    (fun k => val_f16 (hA k)) (fun k => val_f16 (hB k)) hcf (by
      rw [Format.binary32_maxVal, Format.binary32_eta]
      have : (K : ℚ) * 2 ^ 32 ≤ 2 ^ 37 := by nlinarith
      have h5 : |(c.get2 i j).toRat| + ∑ k, Rules.Num.liftMag .f16 (A.get2 i k).toRat *
          Rules.Num.liftMag .f16 (B.get2 k j).toRat ≤ 2 ^ 52 := by
        linarith
      have h6 : (1 + Rules.Num.relDot16 K) * (|(c.get2 i j).toRat| + ∑ k, Rules.Num.liftMag .f16 (A.get2 i k).toRat *
          Rules.Num.liftMag .f16 (B.get2 k j).toRat) ≤ 2 * 2 ^ 52 := by
        have := abs_nonneg (c.get2 i j).toRat
        nlinarith
      have : (2 : ℚ) * (2 * 2 ^ 52 + 1 + 2 * 2 ^ (-150 : ℤ)) ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
      linarith)
  -- the step's constants are at most those of `32` products
  have hsr : DotLoop.stepRel Rules.Num.relDot16 K ≤ DotLoop.stepRel Rules.Num.relDot16 32 := by
    unfold DotLoop.stepRel
    have := relDot16_mono hK32
    have := Format.u_nonneg (f := Format.binary32)
    nlinarith
  have hsa : DotLoop.stepAbs Rules.Num.absDot16 K ≤ DotLoop.stepAbs Rules.Num.absDot16 32 := by
    unfold DotLoop.stepAbs
    have := absDot16_mono hK32
    have := Format.u_nonneg (f := Format.binary32)
    nlinarith
  have hre' : |(r.get2 i j).toRat - ((c.get2 i j).toRat + ∑ k, (A.get2 i k).toRat * (B.get2 k j).toRat)| ≤
      DotLoop.stepRel Rules.Num.relDot16 32 * (|(c.get2 i j).toRat| + ∑ k, Rules.Num.liftMag .f16 (A.get2 i k).toRat *
        Rules.Num.liftMag .f16 (B.get2 k j).toRat) + DotLoop.stepAbs Rules.Num.absDot16 32 := by
    refine hre.trans ?_
    have := abs_nonneg (c.get2 i j).toRat
    nlinarith
  have hreR := (Rat.cast_le (K := ℝ)).mpr hre'
  push_cast at hreR
  set S := ∑ k, (A.get2 i k).toReal * (B.get2 k j).toReal with hS
  set Aa := ∑ k, lm (A.get2 i k) * lm (B.get2 k j) with hAa
  have hstep : |(r.get2 i j).toReal - ((c.get2 i j).toReal + S)| ≤ rhoA * (|(c.get2 i j).toReal| + Aa) + alphaA := by
    simpa [toReal_def, rhoA, alphaA, hS, hAa, lm] using hreR
  have hSA : |S| ≤ Aa :=
    (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => abs_mul_le_lm _ _)
  have hnew := accB_step h2 hSA h4 hstep
  refine ⟨hrf, St + S, Bt + Aa, ?_, ?_, ?_, hnew⟩
  · have e : St + S - (Ex + ∑ k, ex k) = (St - Ex) + ∑ k, ((A.get2 i k).toReal * (B.get2 k j).toReal - ex k) := by
      rw [hS, Finset.sum_sub_distrib]; ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    have := (Finset.abs_sum_le_sum_abs (fun k => (A.get2 i k).toReal * (B.get2 k j).toReal - ex k) Finset.univ).trans
      (Finset.sum_le_sum fun k _ => hx k)
    linarith
  · have := abs_add_le St S; linarith
  · have : Aa ≤ ∑ k, bm k := Finset.sum_le_sum fun k _ => hb k
    linarith

/-! ## The outputs -/

/-- **The last rounding** (lines 446, 451, 500): a finite `v` within `Tl` of `E`, rounded to
`float16` (`c` equal to it up to NaN payload), is finite and within `Tl` plus one rounding when
`|E| + Tl ≤ 65504`. -/
theorem out16 {v : F32} {c : F16} {E Tl : ℝ} (hv : v.isFinite = true) (hE : |v.toReal - E| ≤ Tl)
    (hc : FloatV.NaNEq (FloatV.cvt v) c) (hroom : |E| + Tl ≤ 65504) :
    c.isFinite = true ∧ |c.toReal - E| ≤ Tl + (|E| + Tl) * u16 + eta16 := by
  have hvS : |v.toReal| ≤ |E| + Tl := by
    have := abs_sub_abs_le_abs_sub v.toReal E
    linarith
  obtain ⟨hcf, hce⟩ := cvt16_errR hv (hvS.trans hroom)
  rw [nanEq_fin hc hcf]
  refine ⟨hcf, ?_⟩
  have hu : 0 ≤ u16 := by rw [u16_eq]; positivity
  have e : (FloatV.cvt v : F16).toReal - E = ((FloatV.cvt v : F16).toReal - v.toReal) + (v.toReal - E) := by ring
  rw [e]
  refine (abs_add_le _ _).trans ?_
  have : u16 * |v.toReal| ≤ u16 * (|E| + Tl) := mul_le_mul_of_nonneg_left hvS hu
  linarith

/-- **The product by a scale** (`dk *= sm_scale`, `dq *= LN2`): an accumulator `a` within `Acc`
of `S`, times a finite `s`, with room: finite, within `|s|·Acc + u32·|s|·(|S| + Acc) + eta32` of
`s·S`. -/
theorem scale_out {a s v : F32} {S Acc : ℝ} (ha : a.isFinite = true) (hs : s.isFinite = true)
    (haS : |a.toReal - S| ≤ Acc) (hsize : |s.toReal| * (|S| + Acc) ≤ 2 ^ 126)
    (hv : FloatV.NaNEq (FloatV.mul a s) v) :
    v.isFinite = true ∧
      |v.toReal - s.toReal * S| ≤ |s.toReal| * Acc + u32 * (|s.toReal| * (|S| + Acc)) + eta32 := by
  have haabs : |a.toReal| ≤ |S| + Acc := by
    have := abs_sub_abs_le_abs_sub a.toReal S
    linarith
  have hprod : |a.toReal * s.toReal| ≤ |s.toReal| * (|S| + Acc) := by
    rw [abs_mul, mul_comm]; exact mul_le_mul_of_nonneg_left haabs (abs_nonneg _)
  obtain ⟨hmf, hme⟩ := mul_errR ha hs (hprod.trans hsize)
  rw [nanEq_fin hv hmf]
  refine ⟨hmf, ?_⟩
  have hu : 0 ≤ u32 := by rw [u32_eq]; positivity
  have e : (FloatV.mul a s).toReal - s.toReal * S =
      ((FloatV.mul a s).toReal - a.toReal * s.toReal) + s.toReal * (a.toReal - S) := by ring
  rw [e]
  refine (abs_add_le _ _).trans ?_
  rw [abs_mul (s.toReal)]
  have h1 : |s.toReal| * |a.toReal - S| ≤ |s.toReal| * Acc := mul_le_mul_of_nonneg_left haS (abs_nonneg _)
  have h2 : u32 * |a.toReal * s.toReal| ≤ u32 * (|s.toReal| * (|S| + Acc)) := mul_le_mul_of_nonneg_left hprod hu
  linarith

end Kernels.AttnBwd
