import Claim
import TVA.Proof

/-!
# Constants of the error analysis

The relative constants the intermediate invariants of the proof carry (`cO`, `cA`: the inter-chunk
term and the scores after their gates, apart from the `g_gamma` part), and the absolute floor `fl`.
-/

namespace Claims.ChunkFwdO

open TVA

/-- The relative error of the inter-chunk term after its gate (apart from `g_gamma`'s part). -/
noncomputable def cO : ℝ := 7 / 100000

/-- The relative error of a score after its gate (apart from `g_gamma`'s part). -/
noncomputable def cA : ℝ := 9 / 100000

/-- The coefficient of the absolute floors of the intermediate invariants, `2^-112`. -/
noncomputable def fl : ℝ := (2 : ℝ) ^ (-112 : ℤ)

end Claims.ChunkFwdO
