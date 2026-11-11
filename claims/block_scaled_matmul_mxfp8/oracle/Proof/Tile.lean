import Claim
import TVA.Proof
import Proof.Basic

/-!
# The loaded tiles' bits

`ScaledCfg.lhsBits` / `rhsBits` of the tiles the loop body loads, as the claim's `xa` / `xbBits`
at the global element they hold.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- **The `lhs` tile**: element `(i, k)` of the loaded `a` block is `a`'s element
`(pm · 128 + i, 128 · t + k)`. -/
theorem aTile_lane {d : Inst} {m : Mem} {pm t : Nat} {av : Tensor [128, 128] F8E4}
    (hav : ∀ (r c : Fin 128), av.get2 r c = xa d m (pm * 128 + (r : Nat)) (128 * t + (c : Nat)))
    (i k : Nat) (hi : i < 128) (hk : k < 128) :
    ScaledCfg.lhsBits (d := .f8e4nv) cfgMixed av i k = (xa d m (pm * 128 + i) (128 * t + k)).pat := by
  simp [ScaledCfg.lhsBits, cfgMixed, Tensor.getD2, hi, hk, hav]
  rfl

/-- **The `rhs` tile, `"mixed"`**: element `(kk, n)` of the transposed `b` block is `b`'s
element `(pn · 256 + n, 128 · t + kk)`. -/
theorem bTile_mixed {d : Inst} (hm : d.mixed = true) {m : Mem} {pn t : Nat}
    {bv : Tensor [256, 64] U8}
    (hbv : ∀ (n : Fin 256) (j : Fin 64),
      DType.toBits .u8 (bv.get2 n j) =
        rdBits m d.b ((pn * 256 + (n : Nat)) * (d.K / 2) + (64 * t + (j : Nat))))
    (l : Idx [256, 64]) (kk : Nat) (hkk : kk < 128) :
    ScaledCfg.rhsBits (d := .u8) cfgMixed bv.trans kk (l.coord 0) =
      xbBits d m (pn * 256 + l.coord 0) (128 * t + kk) := by
  have h1 : kk / 2 < 64 := by omega
  have h2 : l.coord 0 < 256 := by
    have h := l.coord_lt (k := 0) (by simp)
    simpa [Shape.dim] using h
  have hdiv : (128 * t + kk) / 2 = 64 * t + kk / 2 := by omega
  have hmod : (128 * t + kk) % 2 = kk % 2 := by omega
  simp [ScaledCfg.rhsBits, cfgMixed, Tensor.getD2, h1, h2, hbv, xbBits, hm, hdiv, hmod]

/-- **The `rhs` tile, `"mxfp8"`**: element `(kk, n)` of the transposed `b` block is `b`'s
element `(pn · 256 + n, 128 · t + kk)`. -/
theorem bTile_mx {d : Inst} (hm : d.mixed = false) {m : Mem} {pn t : Nat}
    {bv : Tensor [256, 128] F8E4}
    (hbv : ∀ (n : Fin 256) (j : Fin 128),
      (bv.get2 n j).pat = rdBits m d.b ((pn * 256 + (n : Nat)) * d.K + (128 * t + (j : Nat))))
    (l : Idx [256, 128]) (kk : Nat) (hkk : kk < 128) :
    ScaledCfg.rhsBits (d := .f8e4nv) cfgMx bv.trans kk (l.coord 0) =
      xbBits d m (pn * 256 + l.coord 0) (128 * t + kk) := by
  have h2 : l.coord 0 < 256 := by
    have h := l.coord_lt (k := 0) (by simp)
    simpa [Shape.dim] using h
  have hbit : ∀ x : F8E4, DType.toBits .f8e4nv x = x.pat := fun _ => rfl
  simp [ScaledCfg.rhsBits, cfgMx, Tensor.getD2, hkk, h2, hbit, hbv, xbBits, hm]

end Claims.BlockScaledMxfp8
