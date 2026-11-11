import Claim
import TVA.Proof

/-!
# `chunk_delta_h_bf16`: the quantities of the numeric core

Real-valued counterparts of the claim's quantities, so that the numeric lemmas are about numbers.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The lifted magnitude of a `bfloat16` element, as a real. -/
noncomputable def lmB (x : BF16) : ℝ := (Device.liftMag .bf16 x.toRat : ℝ)

/-- The magnitude of a new value for the tolerance: `|vref| + 2^-5·Z`. -/
noncomputable def vAbsA (vref Z : ℝ) : ℝ := |vref| + 2 ^ (-5 : ℤ) * Z

end Claims.ChunkDeltaH
