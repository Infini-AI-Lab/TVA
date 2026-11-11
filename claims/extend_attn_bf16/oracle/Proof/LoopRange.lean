import Proof.Spec

/-!
# The loops' ranges

`range(lo, hi, 64)` over `qo_indptr`'s type or `int32` with natural bounds: the bounds are in
range and the trip count is `⌈(hi - lo) / 64⌉`.
-/

namespace Claims.ExtendAttn

open TVA

/-- `range(lo, 64-step, hi)` of natural bounds `L ≤ H` with `H + 64` below the signed range:
`tl.RangeOk`, and the trip count `(H - L + 63) / 64`. -/
theorem range_ok {w : Nat} {lo hi : IntV true w} {L H : Nat} (hlo : lo.toInt = L)
    (hhi : hi.toInt = H) (hLH : L ≤ H) (hw : H + 64 < 2 ^ (w - 1)) :
    tl.RangeOk lo hi (IntV.ofNat 64 : IntV true w) ∧
      tl.tripCount lo.toInt hi.toInt (IntV.ofNat 64 : IntV true w).toInt = (H - L + 63) / 64 := by
  have h64 : (IntV.ofNat 64 : IntV true w).toInt = 64 := by
    have := IntV.toInt_ofNat (w := w) (n := 64) (by omega)
    simpa using this
  have hmI : IntV.maxInt true w = 2 ^ (w - 1) - 1 := by simp [IntV.maxInt]
  have hpos : (0 : Int) < 2 ^ (w - 1) := by positivity
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
  · rw [h64]; norm_num
  · rw [hhi, h64, hmI]; have : (H : Int) + 64 < 2 ^ (w - 1) := by exact_mod_cast hw
    omega
  · rw [hlo, hmI]; have : (H : Int) + 64 < 2 ^ (w - 1) := by exact_mod_cast hw
    omega
  · rw [h64, hmI]; have : (H : Int) + 64 < 2 ^ (w - 1) := by exact_mod_cast hw
    omega
  · rw [hlo, hhi, h64]
    unfold tl.tripCount
    omega

end Claims.ExtendAttn
