import AttnFwdE5m2

/-! Reviewer r-e5fwd, round 2: print the changed specification's new constants and `Post` with
full names, to check that every operation is a `TVA.*`, Mathlib or kernel definition whose
meaning was read; and the axioms of the three theorems at the new gate. -/

set_option pp.fullNames true

#print Kernels.AttnFwdE5m2.magCp
#print Kernels.AttnFwdE5m2.magNext
#print Kernels.AttnFwdE5m2.magA
#print Kernels.AttnFwdE5m2.magD
#print Kernels.AttnFwdE5m2.tolV
#print Kernels.AttnFwdE5m2.rho8
#print Kernels.AttnFwdE5m2.Post
#print axioms Kernels.AttnFwdE5m2.main
#print axioms Kernels.AttnFwdE5m2.consistent
#print axioms Kernels.AttnFwdE5m2.satisfiable
