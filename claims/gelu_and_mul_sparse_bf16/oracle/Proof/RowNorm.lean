import Mathlib

/-!
# The Euclidean norm of a finite family of reals

`√(Σ f²)` over a `Finset`: Cauchy–Schwarz, the triangle inequality (Minkowski), monotonicity in
the magnitudes, scaling, and constants.
-/

namespace Claims.GeluAndMulSparse

open Finset

variable {ι : Type*} (s : Finset ι)

/-- Cauchy–Schwarz: `Σ f·h ≤ √(Σ f²)·√(Σ h²)`. -/
theorem sum_mul_le_sqrt (f h : ι → ℝ) :
    ∑ j ∈ s, f j * h j ≤ Real.sqrt (∑ j ∈ s, f j ^ 2) * Real.sqrt (∑ j ∈ s, h j ^ 2) := by
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq s f h
  have h1 : 0 ≤ ∑ j ∈ s, f j ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have h2 : 0 ≤ ∑ j ∈ s, h j ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  rw [← Real.sqrt_mul h1]
  calc ∑ j ∈ s, f j * h j ≤ |∑ j ∈ s, f j * h j| := le_abs_self _
    _ = Real.sqrt ((∑ j ∈ s, f j * h j) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ _ := Real.sqrt_le_sqrt hcs

/-- Minkowski: `√(Σ (f + h)²) ≤ √(Σ f²) + √(Σ h²)`. -/
theorem sqrt_sum_add_sq_le (f h : ι → ℝ) :
    Real.sqrt (∑ j ∈ s, (f j + h j) ^ 2) ≤
      Real.sqrt (∑ j ∈ s, f j ^ 2) + Real.sqrt (∑ j ∈ s, h j ^ 2) := by
  have h1 : 0 ≤ ∑ j ∈ s, f j ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have h2 : 0 ≤ ∑ j ∈ s, h j ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hc := sum_mul_le_sqrt s f h
  rw [Real.sqrt_le_left (by positivity)]
  have e : ∑ j ∈ s, (f j + h j) ^ 2 =
      ∑ j ∈ s, f j ^ 2 + 2 * ∑ j ∈ s, f j * h j + ∑ j ∈ s, h j ^ 2 := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [e, add_sq, Real.sq_sqrt h1, Real.sq_sqrt h2]
  linarith

/-- The norm grows with the magnitudes: `|f j| ≤ p j` gives `√(Σ f²) ≤ √(Σ p²)`. -/
theorem sqrt_sum_sq_mono {f p : ι → ℝ} (hf : ∀ j ∈ s, |f j| ≤ p j) :
    Real.sqrt (∑ j ∈ s, f j ^ 2) ≤ Real.sqrt (∑ j ∈ s, p j ^ 2) := by
  refine Real.sqrt_le_sqrt (Finset.sum_le_sum fun j hj => ?_)
  have := hf j hj
  rw [← sq_abs (f j)]
  exact pow_le_pow_left₀ (abs_nonneg _) this 2

/-- Scaling: `√(Σ (k·f)²) = |k|·√(Σ f²)`. -/
theorem sqrt_sum_sq_smul (k : ℝ) (f : ι → ℝ) :
    Real.sqrt (∑ j ∈ s, (k * f j) ^ 2) = |k| * Real.sqrt (∑ j ∈ s, f j ^ 2) := by
  have e : ∑ j ∈ s, (k * f j) ^ 2 = k ^ 2 * ∑ j ∈ s, f j ^ 2 := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => by ring
  rw [e, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq_eq_abs]

/-- A constant family: `√(Σ_{j<n} c²) = |c|·√n`. -/
theorem sqrt_sum_sq_const (n : ℕ) (c : ℝ) :
    Real.sqrt (∑ _j ∈ Finset.range n, c ^ 2) = |c| * Real.sqrt n := by
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_comm, Real.sqrt_mul (sq_nonneg _),
    Real.sqrt_sq_eq_abs]

/-- The reverse triangle inequality: `|√(Σ f²) − √(Σ h²)| ≤ √(Σ (f − h)²)`. -/
theorem abs_sqrt_sum_sq_sub_le (f h : ι → ℝ) :
    |Real.sqrt (∑ j ∈ s, f j ^ 2) - Real.sqrt (∑ j ∈ s, h j ^ 2)| ≤
      Real.sqrt (∑ j ∈ s, (f j - h j) ^ 2) := by
  have h1 := sqrt_sum_add_sq_le s (fun j => f j - h j) h
  have h2 := sqrt_sum_add_sq_le s (fun j => h j - f j) f
  simp only [sub_add_cancel] at h1 h2
  have e : ∑ j ∈ s, (h j - f j) ^ 2 = ∑ j ∈ s, (f j - h j) ^ 2 :=
    Finset.sum_congr rfl fun j _ => by ring
  rw [e] at h2
  rw [abs_le]; constructor <;> linarith

end Claims.GeluAndMulSparse
