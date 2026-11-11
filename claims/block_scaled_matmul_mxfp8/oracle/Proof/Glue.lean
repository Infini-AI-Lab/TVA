import Claim
import TVA.Proof
import Proof.Basic
import Proof.ScaleIdx

/-!
# The kernel's reshape and permute of a scale block, read lane by lane.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- The kernel's reshape and permute of `a`'s scale block puts scale `(r, g)` of the tile at position `aU r g`. -/
theorem aScale_reshape {av : Tensor [1, 1, 1, 2, 256] U8}
    (h1 : ([1, 1, 32, 4, 4] : Shape).prod = [1, 1, 1, 2, 256].prod)
    (h2 : ([128, 4] : Shape).prod =
      ([0, 3, 2, 1, 4].map (Shape.dim [1, 1, 32, 4, 4])).prod) :
    ∀ (r : Fin 128) (g : Fin 4),
      ((((av.reshape [1, 1, 32, 4, 4] h1).permute [0, 3, 2, 1, 4]).reshape [128, 4] h2).get2 r g) =
        av.get (Idx.ofPos [1, 1, 1, 2, 256] (aU (r : Nat) (g : Nat)) (aU_lt r.isLt g.isLt)) := by
  intro r g
  exact aScale_get h1 (by decide) h2 r g

/-- The kernel's reshape and permute of `b`'s scale block puts scale `(n, g)` of the tile at position `bPos n g`. -/
theorem bScale_reshape {bv : Tensor [1, 2, 1, 2, 256] U8}
    (h1 : ([2, 1, 32, 4, 4] : Shape).prod = [1, 2, 1, 2, 256].prod)
    (h2 : ([256, 4] : Shape).prod =
      ([0, 3, 2, 1, 4].map (Shape.dim [2, 1, 32, 4, 4])).prod) :
    ∀ (n : Fin 256) (g : Fin 4),
      ((((bv.reshape [2, 1, 32, 4, 4] h1).permute [0, 3, 2, 1, 4]).reshape [256, 4] h2).get2 n g) =
        bv.get (Idx.ofPos [1, 2, 1, 2, 256] (bPos (n : Nat) (g : Nat)) (bPos_lt n.isLt g.isLt)) := by
  intro n g
  exact bScale_get h1 (by decide) h2 n g

end Claims.BlockScaledMxfp8
