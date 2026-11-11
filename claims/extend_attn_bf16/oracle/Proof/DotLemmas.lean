import Proof.CoreLemmas
import Proof.AccReal
import Proof.FloatOps

/-!
# `tl.dot` of `bfloat16` operands, one output element, over the reals

`dot_zero_real` (a dot from a zero accumulator) and `dot_add_real` (`tl.dot_add`: the folded dot
or the dot plus a fusable addition), for `K = 64` products, from `Device.DotBF16`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- The lifted magnitude sum of one output element, as a real. -/
def liftSum {M N : ℕ} (A : Tensor [M, 64] BF16) (B : Tensor [64, N] BF16) (i : Fin M) (j : Fin N) : ℝ :=
  ∑ k : Fin 64, ((Device.liftMag .bf16 (A.get2 i k).toRat * Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) : ℝ)

/-- The lifted sum `liftSum` of two bf16 tensors is nonnegative at every output position. -/
theorem liftSum_nonneg {M N : ℕ} (A : Tensor [M, 64] BF16) (B : Tensor [64, N] BF16) (i : Fin M)
    (j : Fin N) : 0 ≤ liftSum A B i j :=
  Finset.sum_nonneg fun k _ => by
    have := mul_nonneg (Device.liftMag_nonneg .bf16 (A.get2 i k).toRat)
      (Device.liftMag_nonneg .bf16 (B.get2 k j).toRat)
    exact_mod_cast this

/-- **A dot from a zero accumulator**, one element: finite and within the device's bound. -/
theorem dot_zero_real {P : Platform} (hP : Device P) {M N : ℕ} {A : Tensor [M, 64] BF16}
    {B : Tensor [64, N] BF16} {acc r : Tensor [M, N] F32} (hr : P.dot cfgBF16 A B acc r)
    (i : Fin M) (j : Fin N) (hacc : DType.val? .f32 (acc.get2 i j) = some 0)
    (hA : ∀ k, (A.get2 i k).isFinite = true) (hB : ∀ k, (B.get2 k j).isFinite = true)
    (hroom : liftSum A B i j ≤ 2 ^ 120) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toReal - ∑ k : Fin 64, (A.get2 i k).toReal * (B.get2 k j).toReal| ≤
        66 / 2 ^ 22 * liftSum A B i j + 66 / 2 ^ 126 := by
  have hd := hP.dotBF16 .tf32
  have hva : ∀ k, cfgBF16.a.val? (A.get2 i k) = some (A.get2 i k).toRat := fun k =>
    DType.val?_bf16.mpr ⟨hA k, rfl⟩
  have hvb : ∀ k, cfgBF16.b.val? (B.get2 k j) = some (B.get2 k j).toRat := fun k =>
    DType.val?_bf16.mpr ⟨hB k, rfl⟩
  have hL : (liftSum A B i j) = ((∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat *
      Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) : ℝ) := by
    unfold liftSum; push_cast; rfl
  have hLq : (∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat *
      Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) ≤ 2 ^ 120 := by
    have : ((∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat *
        Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) : ℝ) ≤ ((2 ^ 120 : ℚ) : ℝ) := by
      rw [← hL]; push_cast; exact hroom
    exact_mod_cast this
  have hL0 : (0 : ℚ) ≤ ∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat *
      Device.liftMag .bf16 (B.get2 k j).toRat :=
    Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  obtain ⟨vr, hvr, herr⟩ := hd A B acc r hr i j (fun k => (A.get2 i k).toRat)
    (fun k => (B.get2 k j).toRat) 0 hva hvb hacc (by
      simp only [abs_zero, zero_add, DType.bound]
      have hrel : Device.relDot16 64 ≤ 1 := by unfold Device.relDot16; norm_num
      have habs : Device.absDot16 64 ≤ 1 := by unfold Device.absDot16; norm_num
      have h1 : (Device.relDot16 64) * (∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat *
          Device.liftMag .bf16 (B.get2 k j).toRat) ≤ 1 * 2 ^ 120 :=
        mul_le_mul hrel hLq hL0 (by norm_num)
      show _ + (_ + _) ≤ Format.binary32.maxVal
      rw [Format.binary32_maxVal]
      norm_num at h1 habs ⊢
      linarith)
  obtain ⟨hfin, hval⟩ := DType.val?_f32.mp hvr
  refine ⟨hfin, ?_⟩
  simp only [abs_zero, zero_add] at herr
  have herr' := (Rat.cast_le (K := ℝ)).mpr herr
  unfold FloatV.toReal
  rw [hval]
  push_cast at herr' ⊢
  unfold Device.relDot16 Device.absDot16 at herr'
  rw [hL] at *
  push_cast at herr' ⊢
  norm_num at herr' ⊢
  linarith

/-- **`tl.dot_add`**, one element: the folded dot with accumulator `c`, or the dot from a zero
accumulator `z` then a fusable addition. -/
theorem dot_add_real {P : Platform} (hP : Device P) {fusion : Bool} {M N : ℕ}
    {A : Tensor [M, 64] BF16} {B : Tensor [64, N] BF16} {c z r : Tensor [M, N] F32}
    (hr : P.dot cfgBF16 A B c r ∨ ∃ d, P.dot cfgBF16 A B z d ∧
      ∀ ix, AccAdd fusion .f32 (d.get ix) (c.get ix) (r.get ix))
    (i : Fin M) (j : Fin N) (hz : DType.val? .f32 (z.get2 i j) = some 0)
    (hA : ∀ k, (A.get2 i k).isFinite = true) (hB : ∀ k, (B.get2 k j).isFinite = true)
    (hc : (c.get2 i j).isFinite = true)
    (hroom : |(c.get2 i j).toReal| + liftSum A B i j ≤ 2 ^ 120) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toReal - ((c.get2 i j).toReal +
          ∑ k : Fin 64, (A.get2 i k).toReal * (B.get2 k j).toReal)| ≤
        (66 / 2 ^ 22 + 3 * (1 / 2 ^ 24) * (1 + 66 / 2 ^ 22)) *
            (|(c.get2 i j).toReal| + liftSum A B i j) +
          (66 / 2 ^ 126 * (1 + 3 * (1 / 2 ^ 24)) + 4 * (1 / 2 ^ 150)) := by
  have hva : ∀ k, cfgBF16.a.val? (A.get2 i k) = some (A.get2 i k).toRat := fun k =>
    DType.val?_bf16.mpr ⟨hA k, rfl⟩
  have hvb : ∀ k, cfgBF16.b.val? (B.get2 k j) = some (B.get2 k j).toRat := fun k =>
    DType.val?_bf16.mpr ⟨hB k, rfl⟩
  have hL : (liftSum A B i j) = ((∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat *
      Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) : ℝ) := by
    unfold liftSum; push_cast; rfl
  have hroomq : 2 * ((1 + Device.relDot16 64) * (|(c.get2 i j).toRat| +
      ∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat * Device.liftMag .bf16 (B.get2 k j).toRat) +
      Device.absDot16 64 + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
    have h1 : (|(c.get2 i j).toRat| + ∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat *
        Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) ≤ 2 ^ 120 := by
      have hcabs : ((|(c.get2 i j).toRat| : ℚ) : ℝ) = |(c.get2 i j).toReal| := by
        unfold FloatV.toReal; push_cast; rfl
      have : ((|(c.get2 i j).toRat| + ∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat *
          Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) : ℝ) ≤ ((2 ^ 120 : ℚ) : ℝ) := by
        rw [Rat.cast_add, hcabs, ← hL]
        push_cast
        exact hroom
      exact_mod_cast this
    have hrel : Device.relDot16 64 ≤ 1 := by unfold Device.relDot16; norm_num
    have hL0 : (0 : ℚ) ≤ |(c.get2 i j).toRat| + ∑ k : Fin 64, Device.liftMag .bf16 (A.get2 i k).toRat *
        Device.liftMag .bf16 (B.get2 k j).toRat :=
      add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun k _ =>
        mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _))
    have h3 := mul_le_mul (by linarith : 1 + Device.relDot16 64 ≤ 2) h1 hL0 (by norm_num)
    have habs : Device.absDot16 64 ≤ 1 := by unfold Device.absDot16; norm_num
    rw [Format.binary32_maxVal, Format.binary32_eta]
    norm_num at h3 habs ⊢
    linarith
  have hroomq' : 0 ≤ Device.relDot16 64 := by unfold Device.relDot16; norm_num
  obtain ⟨hfin, herr⟩ := DotLift.dotAdd_step (hP.dotBF16 .tf32) (fusion := fusion) hroomq'
    (by unfold Device.absDot16; norm_num) hr i j hz hva hvb hc hroomq
  refine ⟨hfin, ?_⟩
  have herr' := (Rat.cast_le (K := ℝ)).mpr herr
  unfold DotLoop.stepRel DotLoop.stepAbs at herr'
  rw [Format.binary32_u, Format.binary32_eta] at herr'
  unfold Device.relDot16 Device.absDot16 at herr'
  push_cast at herr'
  unfold FloatV.toReal
  rw [hL]
  push_cast
  norm_num at herr' ⊢
  linarith

/-- The lifted magnitude sum of one output element of a `K`-product dot, as a real. -/
def liftSumK {M N K : ℕ} (A : Tensor [M, K] BF16) (B : Tensor [K, N] BF16) (i : Fin M) (j : Fin N) : ℝ :=
  ∑ k : Fin K, ((Device.liftMag .bf16 (A.get2 i k).toRat * Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) : ℝ)

/-- **A dot of `K ≤ 256` products from a zero accumulator**, one element. -/
theorem dot_zero_realK {P : Platform} (hP : Device P) {M N K : ℕ} (hK : K ≤ 256)
    {A : Tensor [M, K] BF16} {B : Tensor [K, N] BF16} {acc r : Tensor [M, N] F32}
    (hr : P.dot cfgBF16 A B acc r) (i : Fin M) (j : Fin N)
    (hacc : DType.val? .f32 (acc.get2 i j) = some 0)
    (hA : ∀ k, (A.get2 i k).isFinite = true) (hB : ∀ k, (B.get2 k j).isFinite = true)
    (hroom : liftSumK A B i j ≤ 2 ^ 120) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toReal - ∑ k : Fin K, (A.get2 i k).toReal * (B.get2 k j).toReal| ≤
        ((K : ℝ) + 2) / 2 ^ 22 * liftSumK A B i j + ((K : ℝ) + 2) / 2 ^ 126 := by
  have hd := hP.dotBF16 .tf32
  have hva : ∀ k, cfgBF16.a.val? (A.get2 i k) = some (A.get2 i k).toRat := fun k =>
    DType.val?_bf16.mpr ⟨hA k, rfl⟩
  have hvb : ∀ k, cfgBF16.b.val? (B.get2 k j) = some (B.get2 k j).toRat := fun k =>
    DType.val?_bf16.mpr ⟨hB k, rfl⟩
  have hL : (liftSumK A B i j) = ((∑ k : Fin K, Device.liftMag .bf16 (A.get2 i k).toRat *
      Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) : ℝ) := by
    unfold liftSumK; push_cast; rfl
  have hLq : (∑ k : Fin K, Device.liftMag .bf16 (A.get2 i k).toRat *
      Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) ≤ 2 ^ 120 := by
    have : ((∑ k : Fin K, Device.liftMag .bf16 (A.get2 i k).toRat *
        Device.liftMag .bf16 (B.get2 k j).toRat : ℚ) : ℝ) ≤ ((2 ^ 120 : ℚ) : ℝ) := by
      rw [← hL]; push_cast; exact hroom
    exact_mod_cast this
  have hL0 : (0 : ℚ) ≤ ∑ k : Fin K, Device.liftMag .bf16 (A.get2 i k).toRat *
      Device.liftMag .bf16 (B.get2 k j).toRat :=
    Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hKq : (K : ℚ) ≤ 256 := by exact_mod_cast hK
  obtain ⟨vr, hvr, herr⟩ := hd A B acc r hr i j (fun k => (A.get2 i k).toRat)
    (fun k => (B.get2 k j).toRat) 0 hva hvb hacc (by
      simp only [abs_zero, zero_add, DType.bound]
      have hrel : Device.relDot16 K ≤ 1 := by
        unfold Device.relDot16
        have : ((K : ℚ) + 2) * 2 ^ (-22 : ℤ) ≤ (256 + 2) * 2 ^ (-22 : ℤ) :=
          mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
        exact this.trans (by norm_num)
      have habs : Device.absDot16 K ≤ 1 := by
        unfold Device.absDot16
        have : ((K : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ (256 + 2) * 2 ^ (-126 : ℤ) :=
          mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
        exact this.trans (by norm_num)
      have h1 : (Device.relDot16 K) * (∑ k : Fin K, Device.liftMag .bf16 (A.get2 i k).toRat *
          Device.liftMag .bf16 (B.get2 k j).toRat) ≤ 1 * 2 ^ 120 :=
        mul_le_mul hrel hLq hL0 (by norm_num)
      show _ + (_ + _) ≤ Format.binary32.maxVal
      rw [Format.binary32_maxVal]
      norm_num at h1 habs ⊢
      linarith)
  obtain ⟨hfin, hval⟩ := DType.val?_f32.mp hvr
  refine ⟨hfin, ?_⟩
  simp only [abs_zero, zero_add] at herr
  have herr' := (Rat.cast_le (K := ℝ)).mpr herr
  unfold FloatV.toReal
  rw [hval]
  push_cast at herr' ⊢
  unfold Device.relDot16 Device.absDot16 at herr'
  rw [hL] at *
  push_cast at herr' ⊢
  simp only [zpow_neg, zpow_ofNat] at herr' ⊢
  convert herr' using 3 <;> norm_num <;> ring

end

end Claims.ExtendAttn
