import MatmulE5m2

/-! Round 2: the new tolerance, what it rejects, and the axioms. -/

set_option pp.fullNames true

#print Kernels.MatmulE5m2.tol
#print Kernels.MatmulE5m2.errAt
#print Kernels.MatmulE5m2.prodV
#print Kernels.MatmulE5m2.partSum
#print Kernels.MatmulE5m2.stepMag
#print Kernels.MatmulE5m2.OutSpec
#check @Kernels.MatmulE5m2.zero_rejected
#check @Kernels.MatmulE5m2.drop_rejected
#check @Kernels.MatmulE5m2.tol_spike
#check @Kernels.MatmulE5m2.tol_le_tolA
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable
#print axioms Kernels.MatmulE5m2.zero_rejected
#print axioms Kernels.MatmulE5m2.drop_rejected

namespace Kernels.MatmulE5m2.Review2

open TVA Kernels.MatmulE5m2

/-- The round-1 counterexample is gone: at `K = 4096`, on the one-product input `1 · 1` in the last
step, a result `S/2` is outside `OutSpec`'s `float16` bound (so are `0` and `−S`, by
`zero_rejected` and the same `tol_spike`). -/
example : tol (spike8 4095 one8) (spike8 4095 one8) 4096 +
    (|dotSum (spike8 4095 one8) (spike8 4095 one8) 4096| + tol (spike8 4095 one8) (spike8 4095 one8) 4096) *
      Math.Float.Format.binary16.u + Math.Float.Format.binary16.eta <
    |dotSum (spike8 4095 one8) (spike8 4095 one8) 4096 / 2 -
      dotSum (spike8 4095 one8) (spike8 4095 one8) 4096| := by
  have hS := dotSum_spike (K := 4096) (by norm_num) one8 one8
  have ht := tol_spike (K := 4096) (by norm_num) (by norm_num) one8 one8
  have ht0 : 0 ≤ tol (spike8 4095 one8) (spike8 4095 one8) 4096 := errAt_nonneg _ _
  rw [one8_val, mul_one] at hS
  rw [one8_val, mul_one, abs_one, mul_one, relDot8_64, absDot8_64] at ht
  have hu : Math.Float.Format.binary16.u = 1 / 2048 := by rw [Math.Float.Format.binary16_u]; norm_num
  have he : Math.Float.Format.binary16.eta = 1 / 33554432 := by
    rw [Math.Float.Format.binary16_eta]; norm_num
  rw [show (4096 - 1 : Nat) = 4095 from rfl] at hS ht
  rw [hS, hu, he, abs_one]
  norm_num
  linarith

end Kernels.MatmulE5m2.Review2
