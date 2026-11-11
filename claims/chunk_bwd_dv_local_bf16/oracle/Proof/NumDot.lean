import Claim
import TVA.Proof
import Proof.Util

/-!
# One element of a `bfloat16` `tl.dot` with a zero accumulator

Both dots of the kernel are `tl.dot(a, b)` of `bfloat16` operands with the default zero
accumulator. `dot_lane` is the device fact `Device.DotBF16` at one output element: the result is
finite and within `relDot16 K · L + absDot16 K` of the exact dot product of the row of `a` and the
column of `b`, where `L` is the sum of the lifted magnitudes.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- `DType.val?` of a finite `bfloat16`. -/
theorem val?_bf16 {x : BF16} (h : x.isFinite = true) : DType.val? .bf16 x = some x.toRat := by
  show (if x.isFinite = true then some x.toRat else none) = some x.toRat
  rw [ite_eq_left h]

/-- `DType.val?` of a finite `float32`. -/
theorem val?_f32 {x : F32} (h : x.isFinite = true) : DType.val? .f32 x = some x.toRat := by
  show (if x.isFinite = true then some x.toRat else none) = some x.toRat
  rw [ite_eq_left h]

/-- **A `bfloat16` dot with a zero accumulator at one element.** -/
theorem dot_lane {P : Platform} (hP : Device P) {M N K : Nat} {a : Tensor [M, K] BF16}
    {b : Tensor [K, N] BF16} {r : Tensor [M, N] F32}
    (hr : P.dot cfgBF16 a b (tl.zeros [M, N] .f32) r) (i : Fin M) (n : Fin N)
    (ha : ∀ k, (a.get2 i k).isFinite = true) (hb : ∀ k, (b.get2 k n).isFinite = true)
    (hroom : (1 + Device.relDot16 K) * ∑ k, Device.liftMag .bf16 (a.get2 i k).toRat *
        Device.liftMag .bf16 (b.get2 k n).toRat + Device.absDot16 K ≤ Format.binary32.maxVal) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - ∑ k, (a.get2 i k).toRat * (b.get2 k n).toRat| ≤
        Device.relDot16 K * ∑ k, Device.liftMag .bf16 (a.get2 i k).toRat *
          Device.liftMag .bf16 (b.get2 k n).toRat + Device.absDot16 K := by
  have hz : DType.val? .f32 ((tl.zeros [M, N] .f32).get2 i n) = some 0 := by
    show DType.val? .f32 (FloatV.ofPat 0) = some 0
    rw [val?_f32 (FloatV.isFinite_ofPat_zero Format.binary32_regular), FloatV.toRat_ofPat_zero]
  have hroom' : |(0 : ℚ)| + ∑ k, Device.liftMag .bf16 (a.get2 i k).toRat *
        Device.liftMag .bf16 (b.get2 k n).toRat +
      (Device.relDot16 K * (|(0 : ℚ)| + ∑ k, Device.liftMag .bf16 (a.get2 i k).toRat *
        Device.liftMag .bf16 (b.get2 k n).toRat) + Device.absDot16 K) ≤ DType.bound .f32 := by
    have : DType.bound .f32 = Format.binary32.maxVal := rfl
    rw [this, abs_zero, zero_add]
    linarith
  obtain ⟨vr, hvr, herr⟩ := hP.dotBF16 .tf32 a b (tl.zeros [M, N] .f32) r hr i n
    (fun k => (a.get2 i k).toRat) (fun k => (b.get2 k n).toRat) 0
    (fun k => val?_bf16 (ha k)) (fun k => val?_bf16 (hb k)) hz hroom'
  have hrf : (r.get2 i n).isFinite = true := by
    by_contra hcon
    have h : DType.val? .f32 (r.get2 i n) = none := by
      show (if (r.get2 i n).isFinite = true then some (r.get2 i n).toRat else none) = none
      rw [ite_eq_right hcon]
    rw [h] at hvr
    cases hvr
  rw [val?_f32 hrf] at hvr
  have hvr' : vr = (r.get2 i n).toRat := (Option.some.inj hvr).symm
  subst hvr'
  refine ⟨hrf, ?_⟩
  simpa only [abs_zero, zero_add, zero_add] using herr

end Claims.ChunkBwdDvLocal
