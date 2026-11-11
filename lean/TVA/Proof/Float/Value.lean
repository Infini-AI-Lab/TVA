import TVA.Proof.Float.Sqrt
import TVA.Triton

/-!
# Float values: patterns, NaN payloads and real values

`FloatV f` (`TVA.Base.Float.Value`) holds a pattern of the format `f`; its predicates and its
value are those of the pattern (`x.toRat = f.toRat x.pat`). This file moves facts between the
two levels and states the basic facts about `NaNEq`:

* `ofPat n` has the pattern `n` when it fits (`pat_ofPat`), so its value, finiteness, … are those
  of `n` (`toRat_ofPat`, `isFinite_ofPat`, …); `ext_pat`: two values with one pattern are equal;
* `isFinite_iff` (neither NaN nor infinite), `toRat_of_isZero`, `abs_toRat_le_maxVal`,
  `isNaN_of_isInf`; negation (`toRat_neg`, `isFinite_neg`, `isNaN_neg`, `toReal_neg`);
* `toReal` is the cast of `toRat` (`toReal_eq`, `toReal_eq_toReal_iff`);
* `NaNEq`, equality up to NaN payload, is an equivalence (`NaNEq.refl`, `symm`, `trans`); a value
  `NaNEq` to a non-NaN value is that value (`NaNEq.eq`). Every `tl.*` float operation returns
  its IEEE result up to `NaNEq`, so `NaNEq.exact_of_isFinite` is how a proof gets from "the IEEE
  result is finite" to "the result of the `tl.*` operation is that value".
-/
namespace TVA

namespace FloatV

variable {f : Format}

/-! ## Patterns -/

/-- A value's pattern fits in the format's width. -/
theorem pat_lt (x : FloatV f) : x.pat < 2 ^ f.width := x.bits.isLt

/-- `ofPat n` has the pattern `n` when `n` fits in the width. -/
theorem pat_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).pat = n := by
  simp [ofPat, pat, Nat.mod_eq_of_lt h]

/-- The value of `ofPat n` is `toRat n` when `n` fits in the width. -/
theorem toRat_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).toRat = f.toRat n := by
  unfold toRat; rw [pat_ofPat h]

/-- `ofPat n` is finite as `n` is, when `n` fits in the width. -/
theorem isFinite_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).isFinite = f.isFinite n := by
  unfold isFinite; rw [pat_ofPat h]

/-- `ofPat n` is NaN as `n` is, when `n` fits in the width. -/
theorem isNaN_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).isNaN = f.isNaN n := by
  unfold isNaN; rw [pat_ofPat h]

/-- `ofPat n` is infinite as `n` is, when `n` fits in the width. -/
theorem isInf_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).isInf = f.isInf n := by
  unfold isInf; rw [pat_ofPat h]

/-- `ofPat n` is zero as `n` is, when `n` fits in the width. -/
theorem isZero_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).isZero = f.isZero n := by
  unfold isZero; rw [pat_ofPat h]

/-- `ofPat n` has the sign of `n`, when `n` fits in the width. -/
theorem sign_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).sign = f.sign n := by
  unfold sign; rw [pat_ofPat h]

/-- Finite means neither NaN nor infinite. -/
theorem isFinite_iff {x : FloatV f} : x.isFinite = true ↔ x.isNaN = false ∧ x.isInf = false :=
  Format.isFinite_iff _

/-- A zero pattern (`±0`) has value `0`. -/
theorem toRat_of_isZero {x : FloatV f} (h : f.isZero x.pat = true) : x.toRat = 0 :=
  Format.toRat_of_isZero h

/-- A finite value is at most `maxVal` in magnitude. -/
theorem abs_toRat_le_maxVal (hf : f.Regular) {x : FloatV f} (hx : x.isFinite = true) :
    |x.toRat| ≤ f.maxVal :=
  Format.abs_toRat_le_maxVal hf hx

/-- `ofPat` of a value's pattern is the value. -/
theorem ofPat_pat (x : FloatV f) : ofPat x.pat = x := by
  cases x; simp [ofPat, pat]

/-- Two values with the same pattern are equal. -/
theorem ext_pat {x y : FloatV f} (h : x.pat = y.pat) : x = y := by
  cases x; cases y; simp only [pat] at h; congr; exact BitVec.eq_of_toNat_eq h

/-- An infinite value is not a NaN. -/
theorem isNaN_of_isInf {x : FloatV f} (h : x.isInf = true) : x.isNaN = false := by
  unfold isInf Format.isInf at h
  unfold isNaN Format.isNaN
  cases hs : f.specials
  · simp only [hs, Bool.and_eq_true, beq_iff_eq] at h ⊢
    simp [h.1, h.2]
  · simp [hs] at h
  · simp

/-- A conversion with rounding mode `.rn` is `cvt`. -/
theorem cvtWith_rn {g : Format} (x : FloatV f) : (cvtWith .rn x : FloatV g) = cvt x := rfl

/-! ## Negation and real values -/

/-- `-x` has the value `-x` (every pattern). -/
theorem toRat_neg (x : FloatV f) : (neg x).toRat = -x.toRat := by
  unfold neg; rw [toRat_ofPat (Format.neg_lt _), Format.toRat_neg]; rfl

/-- `-x` is finite exactly when `x` is. -/
theorem isFinite_neg (x : FloatV f) : (neg x).isFinite = x.isFinite := by
  unfold neg; rw [isFinite_ofPat (Format.neg_lt _), Format.isFinite_neg]; rfl

/-- `-x` is NaN exactly when `x` is. -/
theorem isNaN_neg (x : FloatV f) : (neg x).isNaN = x.isNaN := by
  unfold neg; rw [isNaN_ofPat (Format.neg_lt _), Format.isNaN_neg]; rfl

/-- `toReal` is the cast of `toRat`; unfold with it to move a bound between `ℚ` and `ℝ`. -/
theorem toReal_eq (x : FloatV f) : x.toReal = (x.toRat : ℝ) := rfl

/-- Two values have one real value iff they have one rational value. -/
theorem toReal_eq_toReal_iff {g : Format} {x : FloatV f} {y : FloatV g} :
    x.toReal = y.toReal ↔ x.toRat = y.toRat := by
  unfold toReal; exact Rat.cast_inj

/-- `-x` has the real value `-x`. -/
theorem toReal_neg (x : FloatV f) : (neg x).toReal = -x.toReal := by
  unfold toReal; rw [toRat_neg, Rat.cast_neg]

/-! ## Equality up to NaN payload -/

/-- `NaNEq` is reflexive: the IEEE result itself is an allowed result. -/
theorem NaNEq.refl (x : FloatV f) : NaNEq x x := Or.inl rfl

/-- A result `NaNEq` to a value that is not NaN is that value. -/
theorem NaNEq.eq {x y : FloatV f} (h : NaNEq x y) (hx : x.isNaN = false) : y = x := by
  rcases h with h | ⟨h, -⟩
  · exact h.symm
  · rw [hx] at h; cases h

/-- `NaNEq` is symmetric. -/
theorem NaNEq.symm {x y : FloatV f} (h : NaNEq x y) : NaNEq y x := by
  rcases h with h | ⟨h1, h2⟩
  · exact Or.inl h.symm
  · exact Or.inr ⟨h2, h1⟩

/-- `NaNEq` is transitive. -/
theorem NaNEq.trans {x y z : FloatV f} (h1 : NaNEq x y) (h2 : NaNEq y z) : NaNEq x z := by
  rcases h1 with rfl | ⟨hx, hy⟩
  · exact h2
  · rcases h2 with rfl | ⟨-, hz⟩
    · exact Or.inr ⟨hx, hy⟩
    · exact Or.inr ⟨hx, hz⟩

/-- `NaNEq` is an equivalence. -/
theorem NaNEq.equivalence : Equivalence (NaNEq (f := f)) :=
  ⟨NaNEq.refl, NaNEq.symm, NaNEq.trans⟩

/-- Values equal up to NaN payload are both NaN or both not. -/
theorem NaNEq.isNaN_eq {x y : FloatV f} (h : NaNEq x y) : x.isNaN = y.isNaN := by
  rcases h with rfl | ⟨h1, h2⟩
  · rfl
  · rw [h1, h2]

/-- Values equal up to NaN payload are both finite or both not. -/
theorem NaNEq.isFinite_eq {x y : FloatV f} (h : NaNEq x y) : x.isFinite = y.isFinite := by
  rcases h with rfl | ⟨h1, h2⟩
  · rfl
  · have a : x.isFinite = false := by
      unfold isFinite Format.isFinite; unfold isNaN at h1; rw [h1]; rfl
    have b : y.isFinite = false := by
      unfold isFinite Format.isFinite; unfold isNaN at h2; rw [h2]; rfl
    rw [a, b]

/-- **A finite value is exact up to NaN payload**: `NaNEq x y` with `x` finite gives `y = x`. -/
theorem NaNEq.eq_of_isFinite {x y : FloatV f} (h : NaNEq x y) (hx : x.isFinite = true) : y = x :=
  h.eq (isFinite_iff.mp hx).1

/-- **What a `tl.*` operation returns for a finite exact result**: from `NaNEq x y` (`y` the
result, `x` the library's operation) and `x` finite, `y` is finite with `x`'s value. Combine it
with an error bound on `x` (`add_err_real`, …). -/
theorem NaNEq.exact_of_isFinite {x y : FloatV f} (h : NaNEq x y) (hx : x.isFinite = true) :
    y.isFinite = true ∧ y.toRat = x.toRat ∧ y.toReal = x.toReal := by
  rw [h.eq_of_isFinite hx]; exact ⟨hx, rfl, rfl⟩

end FloatV

end TVA
