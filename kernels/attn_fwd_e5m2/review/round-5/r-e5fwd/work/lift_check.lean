import AttnFwdE5m2

/-! Reviewer r-e5fwd, library round 25: `liftMag` and `minNormal8` moved to `TVA.Rules.Num`; check
that on `float8e5` they still mean what the kernel's spec and the chunk fact rely on. -/

open TVA TVA.Rules.Num

example : minNormal8 .f8e5 = 2 ^ (-14 : ℤ) := rfl
example (q : ℚ) : liftMag .f8e5 q = if q = 0 then 0 else max |q| (2 ^ (-14 : ℤ)) := rfl

set_option pp.fullNames true
#print TVA.Rules.Num.DotChunkAcc
#print axioms Kernels.AttnFwdE5m2.main
