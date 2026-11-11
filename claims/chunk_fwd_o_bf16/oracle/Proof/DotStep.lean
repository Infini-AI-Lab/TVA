import Claim
import TVA.Proof
import Proof.Peel

/-!
# One trip of a `bfloat16` dot into a `float32` accumulator

`dotTrip`: from the device fact `Device.DotBF16`, a trip `acc ← tl.dot(a, b, acc)` of `K` products
keeps the loop invariant "the accumulator is within `DotLoop.bound` of the exact partial sum",
for any output element, given room in `float32`.
-/

open TVA
namespace Claims.ChunkFwdO

/-- `cfgBF16` is the `DotBF16` configuration at `input_precision = tf32`. -/
theorem cfgBF16_eq : cfgBF16 = ⟨.bf16, .bf16, .f32, .tf32, 0⟩ := rfl

/-- A `bfloat16` element with value `v`: finite, `toRat = v`. -/
theorem val?_bf16 {x : BF16} {v : ℚ} : DType.val? .bf16 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  show (if x.isFinite = true then some x.toRat else none) = some v ↔ _
  split_ifs with h
  · simp [h]
  · simp [h]

/-- **One trip of the loop's dot** at output element `(i, n)`: if the accumulator is within the
bound of the exact partial sum `S0` (`|S0| ≤ M0`) after `t` trips, the dot of this trip's operands
(values `va k`, `vb k`) leaves it within the bound after `t + 1` trips of the sum `S0 + Σ va·vb`,
with magnitudes `M0 + Σ liftMag·liftMag`; the room is `M0 + Σ ≤ 2^100`. -/
theorem dotTrip {P : Platform} (hP : Device P) {M N K : Nat} (hK : K ≤ 128)
    {a : Tensor [M, K] BF16} {b : Tensor [K, N] BF16} {c r : Tensor [M, N] F32}
    (hr : P.dot cfgBF16 a b c r) (i : Fin M) (n : Fin N) (va vb : Fin K → ℚ)
    (hva : ∀ k, DType.val? .bf16 (a.get2 i k) = some (va k))
    (hvb : ∀ k, DType.val? .bf16 (b.get2 k n) = some (vb k))
    {t : Nat} (ht : (t : ℚ) * Device.relDot16 K ≤ 1 / 2) (ht' : t ≤ 256) {S0 M0 : ℚ}
    (hc : (c.get2 i n).isFinite = true ∧
      |(c.get2 i n).toRat - S0| ≤ DotLoop.bound (Device.relDot16 K) (Device.absDot16 K) t M0)
    (hS0 : |S0| ≤ M0)
    (hroom : M0 + ∑ k, Device.liftMag .bf16 (va k) * Device.liftMag .bf16 (vb k) ≤ 2 ^ 100) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - (S0 + ∑ k, va k * vb k)| ≤
        DotLoop.bound (Device.relDot16 K) (Device.absDot16 K) (t + 1)
          (M0 + ∑ k, Device.liftMag .bf16 (va k) * Device.liftMag .bf16 (vb k)) := by
  set A := ∑ k, Device.liftMag .bf16 (va k) * Device.liftMag .bf16 (vb k) with hA
  set Sk := ∑ k, va k * vb k with hSk
  have hρ : 0 ≤ Device.relDot16 K := by unfold Device.relDot16; positivity
  have hα : 0 ≤ Device.absDot16 K := by unfold Device.absDot16; positivity
  have hA0 : 0 ≤ A := Finset.sum_nonneg fun k _ =>
    mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hSA : |Sk| ≤ A :=
    (Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum fun k _ => Device.abs_mul_le_liftMag _ _ _ _)
  have hM0 : 0 ≤ M0 := (abs_nonneg _).trans hS0
  have hK' : (K : ℚ) ≤ 128 := by exact_mod_cast hK
  have hrK : Device.relDot16 K ≤ 1 / 1000 := by
    unfold Device.relDot16
    have h1 : ((K : ℚ) + 2) * 2 ^ (-22 : ℤ) ≤ 130 * 2 ^ (-22 : ℤ) :=
      mul_le_mul_of_nonneg_right (by linarith) (by positivity)
    refine h1.trans ?_
    norm_num
  have haK : Device.absDot16 K ≤ 1 / 1000 := by
    unfold Device.absDot16
    have h1 : ((K : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ 130 * 2 ^ (-126 : ℤ) :=
      mul_le_mul_of_nonneg_right (by linarith) (by positivity)
    refine h1.trans ?_
    norm_num
  have hbd : DotLoop.bound (Device.relDot16 K) (Device.absDot16 K) t M0 ≤ M0 + 1 := by
    have h1 := DotLoop.bound_le hρ hα (t := t) ht hM0
    have htq : (t : ℚ) ≤ 256 := by exact_mod_cast ht'
    have e1 : 2 * (t : ℚ) * (Device.relDot16 K * M0) ≤ M0 := by
      have : 2 * (t : ℚ) * Device.relDot16 K ≤ 1 := by linarith
      nlinarith
    have e2 : 2 * (t : ℚ) * Device.absDot16 K ≤ 1 := by
      have : 2 * (t : ℚ) * Device.absDot16 K ≤ 2 * 256 * (1 / 1000) := by
        apply mul_le_mul (by linarith) haK hα (by norm_num)
      refine this.trans ?_
      norm_num
    nlinarith
  have hcval : DType.val? .f32 (c.get2 i n) = some (c.get2 i n).toRat :=
    DType.val?_f32.mpr ⟨hc.1, rfl⟩
  have habs : |(c.get2 i n).toRat| ≤ 2 * M0 + 1 := by
    have h1 := abs_sub_abs_le_abs_sub (c.get2 i n).toRat S0
    linarith [hc.2]
  have hf := hP.dotBF16 .tf32 a b c r hr i n va vb (c.get2 i n).toRat hva hvb hcval (by
    simp only [DType.bound]
    rw [Format.binary32_maxVal]
    have h2100 : (2:ℚ) ^ 100 * 8 ≤ 2 ^ 103 := by norm_num
    have hmax : (2:ℚ) ^ 110 ≤ ((2:ℚ)^24 - 1) * 2^104 := by norm_num
    nlinarith [mul_nonneg hρ (add_nonneg (abs_nonneg (c.get2 i n).toRat) hA0)])
  obtain ⟨vr, hvr, herr⟩ := hf
  obtain ⟨hfin, hval⟩ := DType.val?_f32.mp hvr
  have hdot : |(r.get2 i n).toRat - ((c.get2 i n).toRat + Sk)| ≤
      Device.relDot16 K * (|(c.get2 i n).toRat| + A) + Device.absDot16 K := by
    rw [hval]; exact herr
  exact ⟨hfin, DotLoop.accum_step hρ hα hS0 hSA hc.2 hdot⟩

end Claims.ChunkFwdO
