import Proof.LaneO

/-!
# One lane, from the gate to the stored element (lines 83–92)

`lane_ok` chains `lane_s` (the kernel's sparse gate), `lane_gelu` (the GELU), `lane_out` (the
product with `up` and the roundings) and `lane_comb` (from the kernel's cutoff to the exact one).
-/

namespace Claims.GeluAndMulSparse

open TVA

set_option maxHeartbeats 1600000

/-- **One lane**: for a gate `g`, an `up` value `u` and the kernel's cutoff `ck` within
`7/8·cT − 2^-131` of `cR`, the stored element is finite, within the claim's tolerance of
`geluRef (max (g − cR) 0) · u`, and a zero when `g ≤ cR − cT`. -/
theorem lane_ok {P : Platform} (hP : Device P) {fusion : Bool} {cR cT : ℝ}
    {g u ck s12 sg0 p1 p2 z inn x2 nx ex den sg tw th h ot act av pr : F32}
    {sg16 actb y : BF16}
    (hg : g.isFinite = true) (hgb : |g.toReal| ≤ 2 ^ 32)
    (hu : u.isFinite = true) (hub : |u.toReal| ≤ 2 ^ 32)
    (hck : ck.isFinite = true) (hckb : |ck.toReal| ≤ 2 ^ 40)
    (hcT : 2 ^ (-130 : ℤ) ≤ cT) (hc : |ck.toReal - cR| ≤ 7 / 8 * cT - 2 ^ (-131 : ℤ))
    (g1 : FloatV.NaNEq (FloatV.sub g ck) s12) (g2 : FloatV.NaNEq (FloatV.cvtWith .rn s12) sg16)
    (g3 : FloatV.NaNEq (FloatV.cvtWith .rn sg16) sg0)
    (h1 : FloatV.NaNEq (FloatV.mul (c32 0x3FA6E4E26D4801F7) (sgSel sg0)) p1)
    (h2 : FloatV.NaNEq (FloatV.mul p1 (sgSel sg0)) p2)
    (h3 : FloatV.NaNEq (FloatV.add (FloatV.mul p2 (sgSel sg0)) (sgSel sg0)) z ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma p2 (sgSel sg0) (sgSel sg0)) z))
    (h4 : FloatV.NaNEq (FloatV.mul (c32 0x3FE9884533D43651) z) inn)
    (h5 : FloatV.NaNEq (FloatV.mul (c32 0x4000000000000000) inn) x2)
    (h6 : FloatV.NaNEq (FloatV.neg x2) nx)
    (h7 : P.math .exp .binary32 nx ex)
    (h8 : FloatV.NaNEq (FloatV.add (FloatV.ofRat 1) ex) den)
    (h9 : P.div .binary32 (FloatV.ofRat 1) den sg)
    (h10 : FloatV.NaNEq (FloatV.mul (c32 0x4000000000000000) sg) tw)
    (h11 : AddRel fusion tw (FloatV.neg (c32 0x3FF0000000000000)) th)
    (h12 : FloatV.NaNEq (FloatV.mul (c32 0x3FE0000000000000) (sgSel sg0)) h)
    (h13 : FloatV.NaNEq (FloatV.add (c32 0x3FF0000000000000) th) ot)
    (h14 : FloatV.NaNEq (FloatV.mul h ot) act)
    (h15 : FloatV.NaNEq (FloatV.cvtWith .rn act) actb)
    (h16 : FloatV.NaNEq (FloatV.cvtWith .rn actb) av)
    (h17 : FloatV.NaNEq (FloatV.mul av u) pr) (h18 : FloatV.NaNEq (FloatV.cvtWith .rn pr) y) :
    y.isFinite = true ∧
      |y.toReal - geluRef (max (g.toReal - cR) 0) * u.toReal| ≤
        2 ^ (-5 : ℤ) * |geluRef (max (g.toReal - cR) 0) * u.toReal| + 2 * |u.toReal| * cT +
          2 ^ (-132 : ℤ) ∧
      (g.toReal ≤ cR - cT → y.toReal = 0) := by
  obtain ⟨hsf, hs0, hs1, hse, hsz⟩ := lane_s hg hck hgb hckb g1 g2 g3
  obtain ⟨hactf, hacte, hact0⟩ := lane_gelu hP hsf hs0 hs1 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12
    h13 h14
  have hG0 := geluRef_nonneg hs0
  have hG1 : geluRef (sgSel sg0).toReal ≤ 2 ^ 41 := (geluRef_le hs0).trans hs1
  obtain ⟨hyf, hye, hy0⟩ := lane_out hactf hG0 hG1 hacte hu hub h15 h16 h17 h18
  refine ⟨hyf, lane_comb hcT hc hs0 hse hye, fun hle => ?_⟩
  have hneg : g.toReal - ck.toReal ≤ -2 ^ (-131 : ℤ) := by
    have h1' := (abs_le.mp hc).1
    have h130 : (0 : ℝ) ≤ 2 ^ (-130 : ℤ) := by positivity
    generalize (2 : ℝ) ^ (-131 : ℤ) = Q at h1' ⊢
    linarith
  have hz := hsz hneg
  have hs0' : (sgSel sg0).toRat = 0 := by rw [hz]; exact c32_zero_val.2.1
  have hy := hy0 (hact0 hs0')
  rw [FloatV.toReal_eq, hy]
  norm_num

end Claims.GeluAndMulSparse
