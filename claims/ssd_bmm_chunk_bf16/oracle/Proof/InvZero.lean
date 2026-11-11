import Claim
import TVA.Proof
import Proof.LaneMask

/-!
# The invariant before the loop

The state the kernel enters the main loop with: the two pointer tiles at their first slab, and
the `[BM, BN]` tile of zeros. This file records that it satisfies `inv` at `t = 0`.
-/

open TVA
namespace Claims.SsdBmm

set_option maxHeartbeats 800000

/-- The state the kernel enters the loop with. -/
def st0 (d : Inst) (i : Fin (launch d).size) : StT :=
  (aPtrs d i, bPtrs d i, tl.zeros [BM, BN] .f32)

/-- The invariant holds before the first turn. -/
theorem inv_zero {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {init : Mem} :
    inv d init i 0 (st0 d i) init := by
  show inv d init i 0 (aPtrs d i, bPtrs d i, tl.zeros [BM, BN] .f32) init
  refine ⟨rfl, ?_, ?_, ?_⟩
  · intro mll kk hm
    rw [aPtrs_get hd i mll kk hm]
    norm_num
  · intro kk nn hn
    rw [bPtrs_get hd i kk nn hn]
    norm_num
  · intro mll nn _ _ va vb _ _ _
    rw [zeros_f32_get2 mll nn, f32_zero_toRat]
    have hmin : min (0 * BK) d.K = 0 := by simp
    rw [hmin, dotSum_zero, DotLoop.bound]
    simp

end Claims.SsdBmm



