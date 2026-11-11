import Proof.Defs
import Proof.Cfg

/-!
# `lora_shrink_bf16`: helper facts for one trip of the loop

The configuration (`BLOCK_K`, `SPLIT_K` and the chunk width are one of two triples), the sizes of
the loop (`stepsOf ≤ 512`), the per-trip constants `rho`, `alpha`, and the dictionary between the
`tl.dot` of one trip (chunks of the tile, `Device.chunk16`) and the claim's chunks (`chunkK`).
-/

namespace Claims.LoraShrink

open TVA

/-- The two configurations of the host. -/
theorem cfg4 (d : Inst) :
    (bk d = 256 ∧ cw d = 64 ∧ cpb d = 4 ∧ sk d = 64) ∨ (bk d = 32 ∧ cw d = 32 ∧ cpb d = 1 ∧ sk d = 8) := by
  by_cases h : d.M < 128
  · left; simp [bk, cw, cpb, sk, h]
  · right; simp [bk, cw, cpb, sk, h]

/-- The device's chunks of a `BLOCK_K`-long dot are the claim's `cpb` chunks. -/
theorem chunks16_bk (d : Inst) : Device.chunks16 (bk d) = cpb d := by
  rcases cfg4 d with ⟨h1, -, h3, -⟩ | ⟨h1, -, h3, -⟩ <;> rw [h1, h3] <;> rfl

/-- `BLOCK_K` is `cw · cpb`. -/
theorem bk_eq (d : Inst) : bk d = cw d * cpb d := by
  rcases cfg4 d with ⟨h1, h2, h3, -⟩ | ⟨h1, h2, h3, -⟩ <;> rw [h1, h2, h3]

/-- Inside a block, the device's chunk number `kk / 64` is the claim's `kk / cw`. -/
theorem div64_eq (d : Inst) {kk : Nat} (h : kk < bk d) : kk / 64 = kk / cw d := by
  rcases cfg4 d with ⟨h1, h2, -⟩ | ⟨h1, h2, -⟩
  · rw [h2]
  · rw [h2]; omega

/-- The loop has at most `512` trips. -/
theorem stepsOf_le (d : Inst) (hK : d.K ≤ 2 ^ 17) (p : Nat) : stepsOf d p ≤ 512 := by
  unfold stepsOf blocks
  rcases cfg4 d with ⟨h1, -, -, h4⟩ | ⟨h1, -, -, h4⟩ <;> rw [h1, h4]
  · have : (d.K + 256 - 1) / 256 ≤ 512 := by omega
    omega
  · have : (d.K + 32 - 1) / 32 ≤ 4096 := by omega
    omega

/-- The relative constant of one trip. -/
theorem rho_le (d : Inst) : rho d ≤ 2 ^ (-12 : ℤ) := by
  unfold rho DotLoop.stepRel Device.relDot16
  rw [Format.binary32_u]
  rcases cfg4 d with ⟨h1, -⟩ | ⟨h1, -⟩ <;> rw [h1] <;> norm_num

/-- The absolute constant of one trip. -/
theorem alpha_le (d : Inst) : alpha d ≤ 2 ^ (-100 : ℤ) := by
  unfold alpha DotLoop.stepAbs Device.absDot16
  rw [Format.binary32_u, Format.binary32_eta]
  rcases cfg4 d with ⟨h1, -⟩ | ⟨h1, -⟩ <;> rw [h1] <;> norm_num

end Claims.LoraShrink
