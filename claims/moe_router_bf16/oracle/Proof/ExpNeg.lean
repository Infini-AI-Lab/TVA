import TVA.Proof

/-!
# The supremum of `t * exp t` on the nonpositive line

The exponential's relative accuracy on a nonpositive argument is `2^-23 * |t|`, and it is
weighted by `exp t` itself. So what the weight bound needs is `|t| * exp t ≤ exp (-1)`,
which is the estimate `log x ≤ x - 1` at `x = |t|`.
-/

namespace TVA

end TVA
