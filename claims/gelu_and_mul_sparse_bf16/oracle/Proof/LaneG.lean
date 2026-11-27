import Proof.LaneS
import Proof.Gelu

/-!
# One lane's GELU (lines 86–90)

From the lane's sparse gate `s ∈ [0, 2^41]`: the products and `tl.mul_add` of line 87, the product
by `0.7978845608028654` and by `2.0`, `tl.sigmoid` (`-x`, `tl.exp`, `1 + …`, `1 / …`),
`2.0 * … - 1.0`, `1.0 + tanh` and `0.5 * s * …`: the lane's `activated` (before its `bfloat16`
rounding) is finite and within `2^-15·geluRef s + 2^-145` of `geluRef s`, and a zero when `s` is.
-/

namespace Claims.GeluAndMulSparse

open TVA

set_option maxHeartbeats 1600000

/-- A value within `e` of `q` is at most `|q| + e` in magnitude. -/
theorem abs_le_of_err {r q e : ℝ} (h : |r - q| ≤ e) : |r| ≤ |q| + e := by
  have := abs_sub_abs_le_abs_sub r q; linarith

/-- The constants of line 87 against the source's decimals. -/
theorem aF_close : |((12003091 / 268435456 : ℚ) : ℝ) - aR| ≤ 1 / 16777216 * aR := by
  unfold aR; rw [abs_le]; constructor <;> norm_num

/-- The constant of line 86 against the source's decimal. -/
theorem kF_close : |((6693141 / 8388608 : ℚ) : ℝ) - kR| ≤ 1 / 33554432 * kR := by
  unfold kR; rw [abs_le]; constructor <;> norm_num

/-- **The logistic's argument** (lines 86–89): the kernel's `2.0 * inner` against `Xof S`. -/
theorem x2_core {S aF kF p1 p2 z inn x2 : ℝ} (hS0 : 0 ≤ S) (hS1 : S ≤ 2 ^ 41)
    (haF : aF = ((12003091 / 268435456 : ℚ) : ℝ)) (hkF : kF = ((6693141 / 8388608 : ℚ) : ℝ))
    (h1 : |p1 - aF * S| ≤ u32 * |aF * S| + eta32)
    (h2 : |p2 - p1 * S| ≤ u32 * |p1 * S| + eta32)
    (h3 : |z - (p2 * S + S)| ≤ 3 * u32 * (|p2 * S| + |S|) + 4 * eta32)
    (h4 : |inn - kF * z| ≤ u32 * |kF * z| + eta32)
    (h5 : |x2 - 2 * inn| ≤ u32 * |2 * inn| + eta32) :
    |x2 - Xof S| ≤ 2 ^ (-20 : ℤ) * Xof S + 2 ^ (-120 : ℤ) := by
  have ha := aF_close
  have hk := kF_close
  rw [← haF] at ha
  rw [← hkF] at hk
  have hu : u32 = 1 / 16777216 := by unfold u32; norm_num
  rw [hu] at h1 h2 h3 h4 h5
  have haR : aR = 0.044715 := rfl
  have hkR : kR = 0.7978845608028654 := rfl
  have haF' : aF = 12003091 / 268435456 := by rw [haF]; norm_num
  have hkF' : kF = 6693141 / 8388608 := by rw [hkF]; norm_num
  have hE0 : 0 ≤ eta32 := by unfold eta32; positivity
  have hE1 : eta32 ≤ 1 / 10 ^ 45 := by unfold eta32; norm_num
  generalize eta32 = E at *
  have hS2 : 0 ≤ S ^ 2 := by positivity
  have hS3 : 0 ≤ S ^ 3 := by positivity
  have hA0 : 0 ≤ aF * S := by rw [haF']; positivity
  rw [abs_of_nonneg hA0, abs_of_nonneg hS0] at *
  -- (2), (3)
  have q2 : |p1| ≤ aF * S * (1 + 1 / 16777216) + E := by have := abs_le_of_err h1; linarith [abs_of_nonneg hA0]
  have q3 : |p1 * S| ≤ aF * S ^ 3 / S ^ 2 * 0 + (aF * S * (1 + 1 / 16777216) + E) * S := by
    rw [abs_mul, abs_of_nonneg hS0]; nlinarith [mul_le_mul_of_nonneg_right q2 hS0]
  have q3' : |p1 * S| ≤ aF * S ^ 2 * (1 + 1 / 16777216) + E * S := by
    have : (aF * S * (1 + 1 / 16777216) + E) * S = aF * S ^ 2 * (1 + 1 / 16777216) + E * S := by ring
    linarith
  -- (5), (6)
  have q5 : |p2 * S - p1 * S ^ 2| ≤ (1 / 16777216 * |p1 * S| + E) * S := by
    have : p2 * S - p1 * S ^ 2 = (p2 - p1 * S) * S := by ring
    rw [this, abs_mul, abs_of_nonneg hS0]; exact mul_le_mul_of_nonneg_right h2 hS0
  have q5' : (1 / 16777216 * |p1 * S| + E) * S ≤
      1 / 16777216 * (aF * S ^ 3 * (1 + 1 / 16777216)) + 1 / 16777216 * (E * S ^ 2) + E * S := by
    have := mul_le_mul_of_nonneg_right q3' hS0
    have e : (aF * S ^ 2 * (1 + 1 / 16777216) + E * S) * S =
        aF * S ^ 3 * (1 + 1 / 16777216) + E * S ^ 2 := by ring
    have e2 : (1 / 16777216 * |p1 * S| + E) * S = 1 / 16777216 * (|p1 * S| * S) + E * S := by ring
    rw [e2]; linarith
  have q6 : |p1 * S ^ 2 - aF * S ^ 3| ≤ 1 / 16777216 * (aF * S ^ 3) + E * S ^ 2 := by
    have : p1 * S ^ 2 - aF * S ^ 3 = (p1 - aF * S) * S ^ 2 := by ring
    rw [this, abs_mul, abs_of_nonneg hS2]
    have := mul_le_mul_of_nonneg_right h1 hS2
    have e : (1 / 16777216 * (aF * S) + E) * S ^ 2 = 1 / 16777216 * (aF * S ^ 3) + E * S ^ 2 := by ring
    linarith
  -- (7)
  have q7 : |p2| ≤ |p1 * S| * (1 + 1 / 16777216) + E := by have := abs_le_of_err h2; linarith
  have q7' : |p2 * S| ≤ aF * S ^ 3 * (1 + 1 / 16777216) ^ 2 + E * S ^ 2 * (1 + 1 / 16777216) + E * S := by
    rw [abs_mul, abs_of_nonneg hS0]
    have h8 := mul_le_mul_of_nonneg_right q7 hS0
    have h9 := mul_le_mul_of_nonneg_right q3' hS0
    have e : (aF * S ^ 2 * (1 + 1 / 16777216) + E * S) * S = aF * S ^ 3 * (1 + 1 / 16777216) + E * S ^ 2 := by
      ring
    have e2 : (|p1 * S| * (1 + 1 / 16777216) + E) * S = |p1 * S| * S * (1 + 1 / 16777216) + E * S := by ring
    nlinarith
  -- the exact difference of the cubes
  have q8 : |aF * S ^ 3 - aR * S ^ 3| ≤ 1 / 16777216 * (aR * S ^ 3) := by
    have : aF * S ^ 3 - aR * S ^ 3 = (aF - aR) * S ^ 3 := by ring
    rw [this, abs_mul, abs_of_nonneg hS3]
    have := mul_le_mul_of_nonneg_right ha hS3
    linarith
  have q9 : aF * S ^ 3 ≤ aR * S ^ 3 * (1 + 1 / 16777216) := by
    have := (abs_le.mp q8).2; linarith
  -- absolute terms
  have hES2 : E * S ^ 2 ≤ S / 10 ^ 30 := by
    have h10 : S ^ 2 ≤ 2 ^ 41 * S := by rw [sq]; exact mul_le_mul_of_nonneg_right hS1 hS0
    have h11 : E * S ^ 2 ≤ 1 / 10 ^ 45 * (2 ^ 41 * S) :=
      mul_le_mul hE1 h10 (by positivity) (by positivity)
    linarith
  have hES : E * S ≤ S / 10 ^ 40 := by
    have := mul_le_mul_of_nonneg_right hE1 hS0
    linarith
  have hES2' : 0 ≤ E * S ^ 2 := by positivity
  have hES' : 0 ≤ E * S := by positivity
  -- z against Z
  set Z := S + aR * S ^ 3 with hZ
  have haR0 : 0 ≤ aR * S ^ 3 := by rw [haR]; positivity
  have hzZ : |z - Z| ≤ 7 / 16777216 * Z + 5 * E := by
    have hsplit : z - Z = (z - (p2 * S + S)) + (p2 * S - p1 * S ^ 2) + (p1 * S ^ 2 - aF * S ^ 3) +
        (aF * S ^ 3 - aR * S ^ 3) := by rw [hZ]; ring
    rw [hsplit]
    have t1 := abs_add_le ((z - (p2 * S + S)) + (p2 * S - p1 * S ^ 2) + (p1 * S ^ 2 - aF * S ^ 3))
      (aF * S ^ 3 - aR * S ^ 3)
    have t2 := abs_add_le ((z - (p2 * S + S)) + (p2 * S - p1 * S ^ 2)) (p1 * S ^ 2 - aF * S ^ 3)
    have t3 := abs_add_le (z - (p2 * S + S)) (p2 * S - p1 * S ^ 2)
    have h3' : |z - (p2 * S + S)| ≤ 3 * (1 / 16777216) *
        (aF * S ^ 3 * (1 + 1 / 16777216) ^ 2 + E * S ^ 2 * (1 + 1 / 16777216) + E * S + S) + 4 * E := by
      have : 3 * (1 / 16777216) * (|p2 * S| + S) ≤ 3 * (1 / 16777216) *
          (aF * S ^ 3 * (1 + 1 / 16777216) ^ 2 + E * S ^ 2 * (1 + 1 / 16777216) + E * S + S) := by
        gcongr
      linarith
    linarith
  -- inner against kR Z, then x2
  have hZ0 : 0 ≤ Z := by rw [hZ]; linarith
  have hkpos : 0 < kF := by rw [hkF']; norm_num
  have hkR0 : 0 < kR := by rw [hkR]; norm_num
  have hkFb : kF ≤ kR * (1 + 1 / 33554432) := by have := (abs_le.mp hk).2; linarith
  have hkR1 : kR ≤ 1 := by rw [hkR]; norm_num
  have hz0 : |z| ≤ Z * (1 + 7 / 16777216) + 5 * E := by
    have := abs_le_of_err hzZ; rw [abs_of_nonneg hZ0] at this; linarith
  have q10 : |kF * z| ≤ kF * (Z * (1 + 7 / 16777216) + 5 * E) := by
    rw [abs_mul, abs_of_pos hkpos]; exact mul_le_mul_of_nonneg_left hz0 hkpos.le
  have q11 : |kF * (z - Z)| ≤ kF * (7 / 16777216 * Z + 5 * E) := by
    rw [abs_mul, abs_of_pos hkpos]; exact mul_le_mul_of_nonneg_left hzZ hkpos.le
  have q12 : |(kF - kR) * Z| ≤ 1 / 33554432 * kR * Z := by
    rw [abs_mul, abs_of_nonneg hZ0]; exact mul_le_mul_of_nonneg_right hk hZ0
  have hkZ : 0 ≤ kR * Z := by positivity
  have q13 : kF * Z ≤ kR * Z * (1 + 1 / 33554432) := by nlinarith
  have q14 : kF * E ≤ E := by nlinarith [hkFb]
  have hin1 : |inn - kR * Z| ≤ 9 / 16777216 * (kR * Z) + 7 * E := by
    have hsplit : inn - kR * Z = (inn - kF * z) + kF * (z - Z) + (kF - kR) * Z := by ring
    rw [hsplit]
    have t1 := abs_add_le ((inn - kF * z) + kF * (z - Z)) ((kF - kR) * Z)
    have t2 := abs_add_le (inn - kF * z) (kF * (z - Z))
    have h4' : |inn - kF * z| ≤ 1 / 16777216 * (kF * (Z * (1 + 7 / 16777216) + 5 * E)) + E := by
      have : 1 / 16777216 * |kF * z| ≤ 1 / 16777216 * (kF * (Z * (1 + 7 / 16777216) + 5 * E)) := by gcongr
      linarith
    have e1 : kF * (Z * (1 + 7 / 16777216) + 5 * E) = kF * Z * (1 + 7 / 16777216) + 5 * (kF * E) := by ring
    have e2 : kF * (7 / 16777216 * Z + 5 * E) = 7 / 16777216 * (kF * Z) + 5 * (kF * E) := by ring
    have e3 : 1 / 33554432 * kR * Z = 1 / 33554432 * (kR * Z) := by ring
    rw [e1] at h4'
    rw [e2] at q11
    rw [e3] at q12
    have q15 : 1 / 16777216 * (kF * Z * (1 + 7 / 16777216)) + 7 / 16777216 * (kF * Z) ≤
        1 / 16777216 * (kR * Z * (1 + 1 / 33554432) * (1 + 7 / 16777216)) +
          7 / 16777216 * (kR * Z * (1 + 1 / 33554432)) := by
      have := mul_le_mul_of_nonneg_right q13 (by norm_num : (0 : ℝ) ≤ 1 + 7 / 16777216)
      nlinarith
    linarith
  have hX : Xof S = 2 * (kR * Z) := by unfold Xof; rw [hZ]; ring
  have hin0 : |inn| ≤ kR * Z * (1 + 9 / 16777216) + 7 * E := by
    have := abs_le_of_err hin1; rw [abs_of_nonneg hkZ] at this; linarith
  have hx2 : |x2 - 2 * inn| ≤ 2 / 16777216 * (kR * Z * (1 + 9 / 16777216) + 7 * E) + E := by
    rw [abs_mul, abs_two] at h5
    have : 1 / 16777216 * (2 * |inn|) ≤ 1 / 16777216 * (2 * (kR * Z * (1 + 9 / 16777216) + 7 * E)) := by
      gcongr
    linarith
  rw [hX]
  have hsplit : x2 - 2 * (kR * Z) = (x2 - 2 * inn) + 2 * (inn - kR * Z) := by ring
  rw [hsplit]
  have t1 := abs_add_le (x2 - 2 * inn) (2 * (inn - kR * Z))
  rw [abs_mul, abs_two] at t1
  have h120 : (1 : ℝ) / 10 ^ 37 ≤ 2 ^ (-120 : ℤ) := by norm_num
  have h20 : (2 : ℝ) ^ (-20 : ℤ) = 16 / 16777216 := by norm_num
  rw [h20]
  linarith

end Claims.GeluAndMulSparse
