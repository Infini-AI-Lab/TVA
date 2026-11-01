import MatmulE5m2

/-! Re-review after library round 24: the cited chunk fact, the new tolerance, what it rejects, and
the axioms. -/

set_option pp.fullNames true

#print Kernels.MatmulE5m2.Assumptions
#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.CvtF32E5m2
#print TVA.Rules.Num.CvtTotal
#print TVA.Rules.Num.DotTotal
#print Kernels.MatmulE5m2.tol
#print Kernels.MatmulE5m2.errAt
#print Kernels.MatmulE5m2.liftV
#print Kernels.MatmulE5m2.chunkMax
#check @Kernels.MatmulE5m2.tol_spike
#check @Kernels.MatmulE5m2.zero_rejected
#check @Kernels.MatmulE5m2.drop_rejected
#check @Kernels.MatmulE5m2.tol_le_tolA
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable
#print axioms Kernels.MatmulE5m2.zero_rejected
#print axioms Kernels.MatmulE5m2.drop_rejected

namespace Kernels.MatmulE5m2.Review3

open TVA Kernels.MatmulE5m2

/-- At `K = 4096`, on the one-product input `1 · 1` at index `4095`, a result `S/2` is outside
`OutSpec`'s `float16` bound with the chunk tolerance too. -/
example : tol (spike8 4095 one8) (spike8 4095 one8) 4096 +
    (|dotSum (spike8 4095 one8) (spike8 4095 one8) 4096| + tol (spike8 4095 one8) (spike8 4095 one8) 4096) *
      Math.Float.Format.binary16.u + Math.Float.Format.binary16.eta <
    |dotSum (spike8 4095 one8) (spike8 4095 one8) 4096 / 2 -
      dotSum (spike8 4095 one8) (spike8 4095 one8) 4096| := by
  have hS := dotSum_spike (K := 4096) (by norm_num) one8 one8
  have ht := tol_spike (K := 4096) (by norm_num) (by norm_num) one8 one8
  have ht0 : 0 ≤ tol (spike8 4095 one8) (spike8 4095 one8) 4096 := errAt_nonneg _ _ _
  rw [one8_val, mul_one] at hS
  rw [one8_val, liftMag_one, mul_one] at ht
  have hu : Math.Float.Format.binary16.u = 1 / 2048 := by rw [Math.Float.Format.binary16_u]; norm_num
  have he : Math.Float.Format.binary16.eta = 1 / 33554432 := by
    rw [Math.Float.Format.binary16_eta]; norm_num
  rw [show (4096 - 1 : Nat) = 4095 from rfl] at hS ht
  rw [hS, hu, he, abs_one]
  norm_num
  linarith

end Kernels.MatmulE5m2.Review3
