import Proof.Defs

/-!
# `lora_shrink_bf16`: the two configurations of the host
-/

namespace Claims.LoraShrink

open TVA

/-- The two configurations of the host: `BLOCK_K = 256`, `SPLIT_K = 64`, chunks of 64, four per
dot (below 128 tokens); `BLOCK_K = 32`, `SPLIT_K = 8`, one chunk of 32 per dot (from 128). -/
theorem cfg_cases (d : Inst) :
    (d.M < 128 ∧ bk d = 256 ∧ cw d = 64 ∧ cpb d = 4 ∧ sk d = 64) ∨
      (128 ≤ d.M ∧ bk d = 32 ∧ cw d = 32 ∧ cpb d = 1 ∧ sk d = 8) := by
  by_cases h : d.M < 128
  · left; simp [bk, cw, cpb, sk, h]
  · right; refine ⟨by omega, ?_⟩; simp [bk, cw, cpb, sk, h]

/-- `cw` is positive. -/
theorem cw_pos (d : Inst) : 0 < cw d := by rcases cfg_cases d with h | h <;> simp [h.2.2.1]

/-- `cpb` is positive. -/
theorem cpb_pos (d : Inst) : 0 < cpb d := by rcases cfg_cases d with h | h <;> simp [h.2.2.2.1]

/-- `SPLIT_K` is positive. -/
theorem sk_pos (d : Inst) : 0 < sk d := by rcases cfg_cases d with h | h <;> simp [h.2.2.2.2]

/-- `BLOCK_K` is positive. -/
theorem bk_pos (d : Inst) : 0 < bk d := by rcases cfg_cases d with h | h <;> simp [h.2.1]

/-- The relative constant of one trip is nonnegative. -/
theorem rho_nonneg (d : Inst) : 0 ≤ rho d := by
  unfold rho DotLoop.stepRel Device.relDot16
  rw [Format.binary32_u]; positivity

/-- The absolute constant of one trip is nonnegative. -/
theorem alpha_nonneg (d : Inst) : 0 ≤ alpha d := by
  unfold alpha DotLoop.stepAbs Device.absDot16
  rw [Format.binary32_u, Format.binary32_eta]; positivity

end Claims.LoraShrink
