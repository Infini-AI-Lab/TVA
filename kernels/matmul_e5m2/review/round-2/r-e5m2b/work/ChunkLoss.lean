import MatmulE5m2

/-! Re-review after the claim moved to `DotE5m2ChunkLoss`: the cited fact, the new `errAt`, and a
check that the new tolerance is never above the round-24 one (the first arm composed alone). -/

open TVA Kernels.MatmulE5m2

set_option pp.fullNames true

#print Kernels.MatmulE5m2.errAt
#print Kernels.MatmulE5m2.chunkAbs
#print Kernels.MatmulE5m2.tol
#check @Kernels.MatmulE5m2.Assumptions.dot
#print TVA.Rules.Num.DotE5m2ChunkLoss
#print TVA.Rules.Num.DotF8ChunkLoss
#print TVA.Rules.Num.kappaChunk8
#print TVA.Rules.Num.rhoLossChunk8
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable
#print axioms Kernels.MatmulE5m2.tol_spike
#print axioms Kernels.MatmulE5m2.zero_rejected
#print axioms Kernels.MatmulE5m2.drop_rejected
#print axioms Kernels.MatmulE5m2.tol_le_tolA

namespace Check

/-- The round-24 error bound: the first arm composed alone. -/
def errOld (p q : Nat → ℚ) : Nat → ℚ
  | 0 => 0
  | j + 1 => (1 + Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8) * errOld p q j +
      Rules.Num.rhoChunk8 * max |partSum p j| (chunkMax q j) +
      Rules.Num.sigmaChunk8 * |partSum p (j + 1)| + Rules.Num.alphaChunk8

/-- The new error bound is never above the round-24 one, chunk for chunk. -/
theorem errAt_le_errOld (p q : Nat → ℚ) : ∀ j, errAt p q j ≤ errOld p q j
  | 0 => le_rfl
  | j + 1 => by
    have ih := errAt_le_errOld p q j
    have h := errAt_succ_le_acc p q j
    have hρ : (0 : ℚ) ≤ Rules.Num.rhoChunk8 := Dot8Chunk.rhoChunk8_nonneg
    have hσ : (0 : ℚ) ≤ Rules.Num.sigmaChunk8 := Dot8Chunk.sigmaChunk8_nonneg
    have := mul_le_mul_of_nonneg_left ih (by linarith : (0 : ℚ) ≤ 1 + Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8)
    show errAt p q (j + 1) ≤ (1 + Rules.Num.rhoChunk8 + Rules.Num.sigmaChunk8) * errOld p q j +
      Rules.Num.rhoChunk8 * max |partSum p j| (chunkMax q j) +
      Rules.Num.sigmaChunk8 * |partSum p (j + 1)| + Rules.Num.alphaChunk8
    linarith

/-- The second arm is the smaller exactly when `Y = X + E` exceeds `2·A / (2^-7 − 2^-12)`,
`4096·2/31 ≈ 264` times `A` ("about `2^8·A_j`" in the README). -/
example (A Y : ℚ) :
    Rules.Num.kappaChunk8 * A + Rules.Num.rhoLossChunk8 * Y < Rules.Num.rhoChunk8 * Y ↔
      8192 / 31 * A < Y := by
  rw [kappa_eq, rhoL_eq, rho_eq]
  constructor <;> intro h <;> linarith

/-- `tol_spike`'s constant: `(1 + 2^-11)·33·2^-12 + 2·2^-12 = 71713·2^-23`. -/
example : (1 + (1 : ℚ) / 2 ^ 11) * (33 / 2 ^ 12) + 2 / 2 ^ 12 = 71713 / 2 ^ 23 := by norm_num

/-- `partSum` is the library's `psum` from `0` over the loop's chunk sums. -/
theorem partSum_eq_psum (p : Nat → ℚ) :
    ∀ j, partSum p j = Dot8Chunk.psum 0 (fun i => ∑ k : Fin 32, p (32 * i + k)) j
  | 0 => by simp [partSum, Dot8Chunk.psum]
  | j + 1 => by rw [partSum_succ, Dot8Chunk.psum_succ, partSum_eq_psum p j]

/-- `errAt` is exactly the library's `Dot8Chunk.errM` on the loop's chunks, with the fact's
constants, the chunk sums, `chunkAbs` and `chunkMax` (README: "the library's `Dot8Chunk.errM` on the
loop's chunks"). -/
theorem errAt_eq_errM (p q : Nat → ℚ) : ∀ j, errAt p q j =
    Dot8Chunk.errM Rules.Num.rhoChunk8 Rules.Num.sigmaChunk8 Rules.Num.alphaChunk8
      Rules.Num.kappaChunk8 Rules.Num.rhoLossChunk8 0 (fun i => ∑ k : Fin 32, p (32 * i + k))
      (chunkAbs p) (chunkMax q) j
  | 0 => rfl
  | j + 1 => by
    rw [errAt_succ, Dot8Chunk.errM, ← errAt_eq_errM p q j, ← partSum_eq_psum, ← partSum_eq_psum]

end Check
