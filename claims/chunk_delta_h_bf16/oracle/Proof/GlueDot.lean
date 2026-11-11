import Proof.NumV
import Proof.Seg

/-!
# `chunk_delta_h_bf16`: one lane of a `tl.dot`

The device fact `Device.DotBF16` read at one output element of a `bfloat16` dot with `64` products
into `float32`: the lane's value is within `rD·(|acc| + Σ lmB·lmB) + aD` of `acc + Σ a·b`.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- **One lane of a `tl.dot`** (`64` `bfloat16` products into `float32`, `Device.DotBF16`): when the
operands and the accumulator are finite and `|acc| + Σ lmB a · lmB b ≤ 2^120`, the result is finite
and within `rD · (|acc| + Σ lmB a · lmB b) + aD` of `acc + Σ a · b`. -/
theorem dot_lane_bound {P : Platform} (hP : Device.DotBF16 P) {M N : Nat} {a : Tensor [M, 64] BF16}
    {b : Tensor [64, N] BF16} {acc r : Tensor [M, N] F32} (h : P.dot cfgBF16 a b acc r)
    (i : Fin M) (n : Fin N) (ha : ∀ k : Fin 64, (a.get2 i k).isFinite = true)
    (hb : ∀ k : Fin 64, (b.get2 k n).isFinite = true) (hacc : (acc.get2 i n).isFinite = true)
    (hroom : |(acc.get2 i n).toReal| + ∑ k : Fin 64, lmB (a.get2 i k) * lmB (b.get2 k n) ≤ 2 ^ 120) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toReal - ((acc.get2 i n).toReal +
          ∑ k : Fin 64, (a.get2 i k).toReal * (b.get2 k n).toReal)| ≤
        rD * (|(acc.get2 i n).toReal| + ∑ k : Fin 64, lmB (a.get2 i k) * lmB (b.get2 k n)) + aD := by
  have hD := hP .tf32 a b acc r h i n (fun k => (a.get2 i k).toRat) (fun k => (b.get2 k n).toRat)
    (acc.get2 i n).toRat (fun k => DType.val?_bf16.2 ⟨ha k, rfl⟩) (fun k => DType.val?_bf16.2 ⟨hb k, rfl⟩)
    (DType.val?_f32.2 ⟨hacc, rfl⟩) ?_
  · obtain ⟨vr, hvr, hbd⟩ := hD
    obtain ⟨hfin, hvr'⟩ := DType.val?_f32.1 hvr
    refine ⟨hfin, ?_⟩
    have hbdR : (|vr - ((acc.get2 i n).toRat + ∑ k : Fin 64, (a.get2 i k).toRat * (b.get2 k n).toRat)| : ℚ) ≤
        Device.relDot16 64 * (|(acc.get2 i n).toRat| + ∑ k : Fin 64,
          Device.liftMag .bf16 (a.get2 i k).toRat * Device.liftMag .bf16 (b.get2 k n).toRat) +
        Device.absDot16 64 := hbd
    have := (Rat.cast_le (K := ℝ)).2 hbdR
    push_cast at this
    unfold FloatV.toReal
    rw [← hvr'] at this
    simpa [rD, aD, lmB] using this
  · refine (Rat.cast_le (K := ℝ)).1 ?_
    have e1 : ((Device.relDot16 64 : ℚ) : ℝ) = 66 * 2 ^ (-22 : ℤ) := rD_eq
    have e2 : ((Device.absDot16 64 : ℚ) : ℝ) = 66 * 2 ^ (-126 : ℤ) := aD_eq
    have e3 : ((DType.f32.bound : ℚ) : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal_real
    push_cast
    rw [e1, e2, e3]
    simp only [lmB] at hroom
    set L : ℝ := |((acc.get2 i n).toRat : ℝ)| + ∑ x : Fin 64, (Device.liftMag .bf16 (a.get2 i x).toRat : ℝ) *
      (Device.liftMag .bf16 (b.get2 x n).toRat : ℝ) with hL
    have hL0 : L ≤ 2 ^ 120 := by
      simpa [hL, FloatV.toReal] using hroom
    have hL1 : 0 ≤ L := by
      rw [hL]
      refine add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun k _ => mul_nonneg ?_ ?_) <;>
        exact_mod_cast Device.liftMag_nonneg _ _
    nlinarith [show (0 : ℝ) < 2 ^ (-22 : ℤ) from by positivity, show (2 : ℝ) ^ (-22 : ℤ) ≤ 1 from by norm_num,
      show (2 : ℝ) ^ (-126 : ℤ) ≤ 1 from by norm_num]

end Claims.ChunkDeltaH
