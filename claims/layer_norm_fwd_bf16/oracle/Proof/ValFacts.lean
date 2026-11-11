import Proof.Setup

/-! Finiteness and value facts about the bit patterns the row's assembly produces: a rational in
range rounds to a finite pattern, conversions and additions and products of finite values in range
are finite, and the constants `0` and its conversions/additions. -/

namespace Claims.LayerNormFwd

open TVA
open Format

noncomputable section

variable {f g : Format}

/-- A `NaNEq` whose left operand is finite is an equality. -/
theorem FloatV.NaNEq.eq_of_isFinite {x y : FloatV f} (h : FloatV.NaNEq x y)
    (hx : x.isFinite = true) : y = x :=
  h.eq (not_isNaN_of_isFinite hx)

/-- The magnitude of the zero pattern is 0 in every format. -/
theorem Format.mag_zero (f : Format) : f.mag 0 = 0 := by
  unfold Format.mag
  simp

/-- The value of the zero pattern is `0`. -/
theorem Format.toRat_zero (f : Format) : f.toRat 0 = 0 :=
  Format.toRat_of_isZero Format.isZero_zero

/-- The value of a signed zero pattern is `0`. -/
theorem Format.toRat_setSign_zero (f : Format) (s : Bool) : f.toRat (f.setSign s 0) = 0 := by
  rw [Format.toRat_setSign, Format.mag_zero]
  cases s <;> simp [Format.magVal_zero]

/-- A rational in range rounds to a pattern below `2^width`. -/
theorem Format.roundRat_lt (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool) {q : ℚ}
    (hq : |q| ≤ f.maxVal) : f.roundRat rnd sat zs q < 2 ^ f.width := by
  by_cases hq0 : q = 0
  · rw [hq0]; exact (Format.roundRat_zero_spec hf).1
  · exact (Format.roundRat_spec hf rnd sat zs hq0 hq).1

/-- A rational in range rounds to a finite pattern. -/
theorem Format.isFinite_roundRat (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool) {q : ℚ}
    (hq : |q| ≤ f.maxVal) : f.isFinite (f.roundRat rnd sat zs q) = true := by
  by_cases hq0 : q = 0
  · rw [hq0]; exact (Format.roundRat_zero_spec hf).2.1
  · exact (Format.roundRat_spec hf rnd sat zs hq0 hq).2.1

/-- A conversion of a finite value in range is finite. -/
theorem FloatV.isFinite_cvt {x : FloatV f} (hg : g.Regular) (hx : x.isFinite = true)
    (hq : |x.toRat| ≤ g.maxVal) : (FloatV.cvt x : FloatV g).isFinite = true := by
  unfold FloatV.cvt FloatV.isFinite
  rw [Format.cvt_of_finite g .rn false hx]
  rw [FloatV.pat_ofPat (Format.roundRat_lt (f := g) (q := f.toRat x.pat) hg .rn false _ hq)]
  exact Format.isFinite_roundRat (f := g) (q := f.toRat x.pat) hg .rn false _ hq

/-- An addition of finite values in range is finite. -/
theorem FloatV.isFinite_add {x y : FloatV f} (hf : f.Regular) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hq : |x.toRat + y.toRat| ≤ f.maxVal) :
    (FloatV.add x y).isFinite = true := by
  unfold FloatV.add FloatV.isFinite
  rw [Format.add_of_finite .rn hx hy]
  rw [FloatV.pat_ofPat
    (Format.roundRat_lt (f := f) (q := f.toRat x.pat + f.toRat y.pat) hf .rn false _ hq)]
  exact Format.isFinite_roundRat (f := f) (q := f.toRat x.pat + f.toRat y.pat) hf .rn false _ hq

/-- A product of finite values in range is finite. -/
theorem FloatV.isFinite_mul {x y : FloatV f} (hf : f.Regular) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hq : |x.toRat * y.toRat| ≤ f.maxVal) :
    (FloatV.mul x y).isFinite = true := by
  unfold FloatV.mul FloatV.isFinite
  rw [Format.mul_of_finite .rn hx hy]
  rw [FloatV.pat_ofPat
    (Format.roundRat_lt (f := f) (q := f.toRat x.pat * f.toRat y.pat) hf .rn false _ hq)]
  exact Format.isFinite_roundRat (f := f) (q := f.toRat x.pat * f.toRat y.pat) hf .rn false _ hq

/-- The float with zero pattern is finite in a regular format. -/
theorem FloatV.isFinite_ofPat_zero (hf : f.Regular) :
    (FloatV.ofPat (0 : ℕ) : FloatV f).isFinite = true := by
  rw [FloatV.isFinite_ofPat (Nat.two_pow_pos f.width)]
  exact Format.isFinite_zero hf

/-- The float with zero pattern has rational value 0. -/
theorem FloatV.toRat_ofPat_zero : (FloatV.ofPat (0 : ℕ) : FloatV f).toRat = 0 := by
  rw [FloatV.toRat_ofPat (Nat.two_pow_pos f.width)]
  exact Format.toRat_zero f

/-- The float with zero pattern has real value 0. -/
theorem FloatV.toReal_ofPat_zero : (FloatV.ofPat (0 : ℕ) : FloatV f).toReal = 0 := by
  unfold FloatV.toReal
  rw [FloatV.toRat_ofPat_zero]
  norm_num

/-- Converting the zero float between regular formats gives a finite float. -/
theorem FloatV.isFinite_cvt_zero (hf : f.Regular) (hg : g.Regular) :
    (FloatV.cvt (FloatV.ofPat (0 : ℕ) : FloatV f) : FloatV g).isFinite = true := by
  refine FloatV.isFinite_cvt (g := g) hg (FloatV.isFinite_ofPat_zero hf) ?_
  rw [FloatV.toRat_ofPat_zero, abs_zero]
  exact le_of_lt (Format.maxVal_pos hg)

/-- Converting the zero float to another format gives rational value 0. -/
theorem FloatV.toRat_cvt_zero (hf : f.Regular) :
    (FloatV.cvt (FloatV.ofPat (0 : ℕ) : FloatV f) : FloatV g).toRat = 0 := by
  have hpat0 : (FloatV.ofPat (0 : ℕ) : FloatV f).pat = 0 :=
    FloatV.pat_ofPat (Nat.two_pow_pos f.width)
  unfold FloatV.cvt FloatV.toRat
  simp only [hpat0]
  rw [Format.cvt_of_finite g .rn false (Format.isFinite_zero hf)]
  rw [Format.toRat_zero]
  rw [Format.roundRat_zero]
  rw [FloatV.pat_ofPat (Format.setSign_lt (f.sign 0) 0)]
  exact Format.toRat_setSign_zero g (f.sign 0)

/-- Converting the zero float to another format gives real value 0. -/
theorem FloatV.toReal_cvt_zero (hf : f.Regular) :
    (FloatV.cvt (FloatV.ofPat (0 : ℕ) : FloatV f) : FloatV g).toReal = 0 := by
  unfold FloatV.toReal
  rw [FloatV.toRat_cvt_zero hf]
  norm_num

/-- The sum of two zero floats is finite in a regular format. -/
theorem FloatV.isFinite_add_zero (hf : f.Regular) :
    (FloatV.add (FloatV.ofPat (0 : ℕ) : FloatV f) (FloatV.ofPat 0)).isFinite = true := by
  refine FloatV.isFinite_add (f := f) hf (FloatV.isFinite_ofPat_zero hf)
    (FloatV.isFinite_ofPat_zero hf) ?_
  rw [FloatV.toRat_ofPat_zero, add_zero, abs_zero]
  exact le_of_lt (Format.maxVal_pos hf)

/-- The sum of two zero floats has rational value 0. -/
theorem FloatV.toRat_add_zero (hf : f.Regular) :
    (FloatV.add (FloatV.ofPat (0 : ℕ) : FloatV f) (FloatV.ofPat 0)).toRat = 0 := by
  have hpat0 : (FloatV.ofPat (0 : ℕ) : FloatV f).pat = 0 :=
    FloatV.pat_ofPat (Nat.two_pow_pos f.width)
  unfold FloatV.add FloatV.toRat
  simp only [hpat0]
  rw [Format.add_of_finite .rn (Format.isFinite_zero hf) (Format.isFinite_zero hf)]
  simp only [Format.toRat_zero, zero_add]
  rw [Format.roundRat_zero]
  rw [FloatV.pat_ofPat (Format.setSign_lt (f.sign 0 && f.sign 0) 0)]
  exact Format.toRat_setSign_zero f (f.sign 0 && f.sign 0)

/-- The sum of two zero floats has real value 0. -/
theorem FloatV.toReal_add_zero (hf : f.Regular) :
    (FloatV.add (FloatV.ofPat (0 : ℕ) : FloatV f) (FloatV.ofPat 0)).toReal = 0 := by
  unfold FloatV.toReal
  rw [FloatV.toRat_add_zero hf]
  norm_num

/-- The product of two zero floats is finite in a regular format. -/
theorem FloatV.isFinite_mul_zero (hf : f.Regular) :
    (FloatV.mul (FloatV.ofPat (0 : ℕ) : FloatV f) (FloatV.ofPat 0)).isFinite = true := by
  refine FloatV.isFinite_mul (f := f) hf (FloatV.isFinite_ofPat_zero hf)
    (FloatV.isFinite_ofPat_zero hf) ?_
  rw [FloatV.toRat_ofPat_zero, mul_zero, abs_zero]
  exact le_of_lt (Format.maxVal_pos hf)

/-- The product of two zero floats has rational value 0. -/
theorem FloatV.toRat_mul_zero (hf : f.Regular) :
    (FloatV.mul (FloatV.ofPat (0 : ℕ) : FloatV f) (FloatV.ofPat 0)).toRat = 0 := by
  have hpat0 : (FloatV.ofPat (0 : ℕ) : FloatV f).pat = 0 :=
    FloatV.pat_ofPat (Nat.two_pow_pos f.width)
  unfold FloatV.mul FloatV.toRat
  simp only [hpat0]
  rw [Format.mul_of_finite .rn (Format.isFinite_zero hf) (Format.isFinite_zero hf)]
  simp only [Format.toRat_zero, mul_zero]
  rw [Format.roundRat_zero]
  rw [FloatV.pat_ofPat (Format.setSign_lt (f.sign 0 != f.sign 0) 0)]
  exact Format.toRat_setSign_zero f (f.sign 0 != f.sign 0)

/-- The product of two zero floats has real value 0. -/
theorem FloatV.toReal_mul_zero (hf : f.Regular) :
    (FloatV.mul (FloatV.ofPat (0 : ℕ) : FloatV f) (FloatV.ofPat 0)).toReal = 0 := by
  unfold FloatV.toReal
  rw [FloatV.toRat_mul_zero hf]
  norm_num

end

end Claims.LayerNormFwd
