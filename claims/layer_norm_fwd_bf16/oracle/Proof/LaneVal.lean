import Proof.RowVals

/-! The value of one lane of the row's `float32` sum: the `NaNEq` chain through the two input
conversions and the addition, both for an enabled lane and for a masked one. -/

namespace Claims.LayerNormFwd

open TVA
open Format

noncomputable section


/-- **An enabled lane's sum**: the `float32` value the kernel stores in the lane's temporary is
the sum of the two widened inputs, finite and below `2^57`. -/
theorem lane_sum {a b : BF16} {bx0 br bx : F32}
    (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hab : |a.toRat| + |b.toRat| ≤ (2 : ℚ) ^ 57)
    (h0 : FloatV.NaNEq (FloatV.cvtWith .rn a) bx0)
    (hr : FloatV.NaNEq (FloatV.cvtWith .rn b) br)
    (hx : FloatV.NaNEq (FloatV.add bx0 br) bx) :
    bx = FloatV.add (FloatV.cvt a) (FloatV.cvt b) ∧ bx.isFinite = true ∧
      |bx.toRat| ≤ (2 : ℚ) ^ 57 := by
  have hc0 : (FloatV.cvtWith .rn a : F32).isFinite = true := by
    rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec ha).1
  have hcr : (FloatV.cvtWith .rn b : F32).isFinite = true := by
    rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec hb).1
  have h0' : bx0 = FloatV.cvtWith .rn a := naneq_eq_of_isFinite h0 hc0
  have hr' : br = FloatV.cvtWith .rn b := naneq_eq_of_isFinite hr hcr
  have hfin : (FloatV.add bx0 br).isFinite = true := by
    rw [h0', hr', FloatV.cvtWith_rn, FloatV.cvtWith_rn]
    exact add_cvt_isFinite ha hb hab
  have hx' : bx = FloatV.add bx0 br := naneq_eq_of_isFinite hx hfin
  rw [hx', h0', hr', FloatV.cvtWith_rn, FloatV.cvtWith_rn]
  exact ⟨rfl, add_cvt_isFinite ha hb hab, add_cvt_abs_le ha hb hab⟩

/-- **A masked lane's sum**: both inputs are zero, so the stored value is zero. -/
theorem lane_sum_zero {a b : BF16} {bx0 br bx : F32}
    (ha : a.isFinite = true) (hb : b.isFinite = true) (ha0 : a.toRat = 0) (hb0 : b.toRat = 0)
    (h0 : FloatV.NaNEq (FloatV.cvtWith .rn a) bx0)
    (hr : FloatV.NaNEq (FloatV.cvtWith .rn b) br)
    (hx : FloatV.NaNEq (FloatV.add bx0 br) bx) :
    bx.isFinite = true ∧ bx.toReal = 0 := by
  have hc0 : (FloatV.cvtWith .rn a : F32).isFinite = true := by
    rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec ha).1
  have hcr : (FloatV.cvtWith .rn b : F32).isFinite = true := by
    rw [FloatV.cvtWith_rn]; exact (cvt_bf16_f32_spec hb).1
  have h0' : bx0 = FloatV.cvtWith .rn a := naneq_eq_of_isFinite h0 hc0
  have hr' : br = FloatV.cvtWith .rn b := naneq_eq_of_isFinite hr hcr
  have hfin : (FloatV.add bx0 br).isFinite = true := by
    rw [h0', hr', FloatV.cvtWith_rn, FloatV.cvtWith_rn]
    exact add_cvt_isFinite ha hb (by rw [ha0, hb0]; norm_num)
  have hx' : bx = FloatV.add bx0 br := naneq_eq_of_isFinite hx hfin
  refine ⟨?_, ?_⟩
  · rw [hx', h0', hr', FloatV.cvtWith_rn, FloatV.cvtWith_rn]
    exact add_cvt_isFinite ha hb (by rw [ha0, hb0]; norm_num)
  · rw [hx', h0', hr', FloatV.cvtWith_rn, FloatV.cvtWith_rn]
    unfold FloatV.toReal
    rw [add_cvt_zero ha hb ha0 hb0]
    norm_num

end

end Claims.LayerNormFwd
