import TVA

/-! lr-12a: spot check of the axioms behind the float layer, its error lemmas and the facts' lemmas. -/
#print axioms TVA.Math.Float.Format.roundRat_rn_err
#print axioms TVA.Math.Float.Format.roundRat_spec
#print axioms TVA.Math.Float.instNatPowNat_eq_natPowFast
#print axioms TVA.FloatV.add_err
#print axioms TVA.FloatV.fma_err
#print axioms TVA.FloatV.cvtSat_of_le
#print axioms TVA.FloatV.toRat_cvt_of_wider
#print axioms TVA.Rules.Num.DivFull.of_divRn
#print axioms TVA.Rules.Num.CvtRn.cvtAcc
