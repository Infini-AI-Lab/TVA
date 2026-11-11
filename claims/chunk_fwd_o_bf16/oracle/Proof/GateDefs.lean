import Claim
import TVA.Proof
import Proof.Loop
import Proof.NumDefs
import Proof.NumExp
import Proof.NumChain
import Proof.FloatLane

/-!
# The gates: what they leave

`OGate`, `AGate`: after the gates (lines 160–170) `b_o` follows `gate · QS` and `b_A` follows
`Aqk · wgt` with the relative error `cO + 0.6·Γ`, `cA + 0.8·Γ` (`Γ = gamErr`) and the floors `fl`,
on the lanes whose element is `InRange` (`RowIn` for the scores). `Gated1` is the state between
the two gate blocks (only one of which is taken).
-/

open TVA
namespace Claims.ChunkFwdO

/-- After the gates, `b_o` follows the gated inter-chunk term. -/
def OGate (d : Inst) (g : Geo) (r : Nat) (bo : Tensor [BT, d.bk] F32) : Prop :=
  ∀ (i : Fin BT) (c : Fin d.bk), g.rowOk i → g.colOk d c →
    InRange d r (g.lo + i) g.ih (g.iv * d.bk + c) →
    (bo.get2 i c).isFinite = true ∧
      |(bo.get2 i c).toReal - gate d r (g.lo + i) g.ih *
          (QS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ)| ≤
        (cO + 6 / 10 * gamErr d g.ih) * gate d r (g.lo + i) g.ih *
            (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ) +
          fl * (1 + gate d r (g.lo + i) g.ih + (PQS d r (g.lo + i) g.ih (g.iv * d.bk + c) : ℝ))

/-- After the gates, `b_A` follows the gated scores `Aqk · wgt`, `s ≤ t`. -/
def AGate (d : Inst) (g : Geo) (r : Nat) (bA : Tensor [BT, BT] F32) : Prop :=
  ∀ (i i' : Fin BT), g.rowOk i → i' ≤ i → RowIn d r (g.lo + i) g.ih →
    (bA.get2 i i').isFinite = true ∧
      |(bA.get2 i i').toReal - (Aqk d (g.lo + i) (g.lo + i') g.ih : ℝ) *
          wgt d r (g.lo + i) (g.lo + i') g.ih| ≤
        (cA + 8 / 10 * gamErr d g.ih) * wgt d r (g.lo + i) (g.lo + i') g.ih *
            (PA d (g.lo + i) (g.lo + i') g.ih : ℝ) +
          fl * (1 + wgt d r (g.lo + i) (g.lo + i') g.ih + (PA d (g.lo + i) (g.lo + i') g.ih : ℝ))

/-- The state after the `USE_G` block: with `g` the gated facts, otherwise the loop's. -/
def Gated1 (d : Inst) (g : Geo) (r : Nat) (bo : Tensor [BT, d.bk] F32) (bA : Tensor [BT, BT] F32) :
    Prop :=
  (d.useG = true → OGate d g r bo ∧ AGate d g r bA) ∧
    (d.useG = false → OInv d g r (nK d) bo ∧ AInv d g (nK d) bA)

/-! ## One lane of a gate -/

/-- One lane of `b_o * gate`: the product of a loop result and a gate result. -/
theorem og_lane {Q P G Γ : ℝ} {ob Gh oh : F32} (hP : 0 ≤ P) (hQ : |Q| ≤ P) (hG : 0 ≤ G)
    (hG64 : G ≤ 2 ^ 64) (hGP : G * P ≤ 2 ^ 100) (hP100 : P ≤ 2 ^ 100) (hΓ : 0 ≤ Γ)
    (hΓ' : Γ ≤ 2 ^ (-9 : ℤ)) (hobF : ob.isFinite = true)
    (hob : |ob.toReal - Q| ≤ 33 / 500000 * P + 2 ^ (-116 : ℤ)) (hGhF : Gh.isFinite = true)
    (hGh : |Gh.toReal - G| ≤ (2 ^ (-21 : ℤ) + Γ / 2) * G + 2 ^ (-126 : ℤ))
    (hr : FloatV.NaNEq (FloatV.mul ob Gh) oh) :
    oh.isFinite = true ∧
      |oh.toReal - G * Q| ≤ (cO + 6 / 10 * Γ) * G * P + fl * (1 + G + P) := by
  obtain ⟨h102, h⟩ := gate_o_num hP hQ hG hG64 hGP hP100 hΓ hΓ' hob hGh
  obtain ⟨hf, he⟩ := mul32_lane hobF hGhF (h102.trans (by norm_num)) hr
  exact ⟨hf, h _ he⟩

/-- One lane of `b_A * weight`: the product of a score and a weight. -/
theorem ag_lane {A PA w Γ : ℝ} {ab wh ah : F32} (hPA : 0 ≤ PA) (hA : |A| ≤ PA) (hw : 0 ≤ w)
    (hw64 : w ≤ 2 ^ 64) (hwPA : w * PA ≤ 2 ^ 100) (hPA100 : PA ≤ 2 ^ 100) (hΓ : 0 ≤ Γ)
    (hΓ' : Γ ≤ 2 ^ (-9 : ℤ)) (habF : ab.isFinite = true)
    (hab : |ab.toReal - A| ≤ 33 / 500000 * PA + 2 ^ (-116 : ℤ)) (hwhF : wh.isFinite = true)
    (hwh : |wh.toReal - w| ≤ (2 ^ (-16 : ℤ) + 72 / 100 * Γ) * w + 2 ^ (-123 : ℤ))
    (hr : FloatV.NaNEq (FloatV.mul ab wh) ah) :
    ah.isFinite = true ∧
      |ah.toReal - A * w| ≤ (cA + 8 / 10 * Γ) * w * PA + fl * (1 + w + PA) := by
  obtain ⟨h102, h⟩ := gate_a_num hPA hA hw hw64 hwPA hPA100 hΓ hΓ' hab hwh
  obtain ⟨hf, he⟩ := mul32_lane habF hwhF (h102.trans (by norm_num)) hr
  exact ⟨hf, h _ he⟩

end Claims.ChunkFwdO
