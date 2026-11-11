import Claim
import TVA.Proof
import Proof.Softplus

/-!
# The value stored in `g`, in the reals (`A_log` not very negative)

`E = tl.exp(A_log)` (with the card `ExpF32` at `X = |A|`), `S` the softplus (see
`Proof/Softplus.lean`), and `gv = tl.mul (-E) S` the stored value.  The lemmas bound
`gv` against `gRef A x0 = -(e^A * spRef x0)` by `gTol A x0 = 2^-23*(64 + |A|) * gMag A x0`
(and give the overflow clause when the exact product is at the edge of `float32`).

The bookkeeping of the tolerance (README, `Tolerances`): relative `52` (softplus) `+ (2 + |A|)`
(exp) `+ 1/2` (the final `mul`) against `64 + |A|`.
-/

namespace Claims.GdnGating

open TVA

/-- **The arithmetic of `g` for `A ≥ -89`** (`a = min |A| 128 ≤ 89`): the collected error of `-E · S` and its rounding is at most `2^-23 · (64 + a) · (e + 2^-121) · (p + 2^-6)`, the shape of `gTol`. -/
theorem gmain_arith_bound {e p a : ℝ} (he : 0 < e) (hp : 0 < p) (ha0 : 0 ≤ a) (ha : a ≤ 89) :
    (2:ℝ)^(-24:ℤ) * ((e*(1 + 2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*a) + (2:ℝ)^(-140:ℤ)) *
      (p*(1 + 52*(2:ℝ)^(-23:ℤ)) + (2:ℝ)^(-24:ℤ)*(1 + (2:ℝ)^(-19:ℤ))))
    + ((e*(1 + 2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*a) + (2:ℝ)^(-140:ℤ)) *
      (52*(2:ℝ)^(-23:ℤ)*p + (2:ℝ)^(-24:ℤ)*(1 + (2:ℝ)^(-19:ℤ))))
    + ((e*(2*(2:ℝ)^(-23:ℤ) + (2:ℝ)^(-23:ℤ)*a) + (2:ℝ)^(-140:ℤ))*p)
    + (2:ℝ)^(-150:ℤ)
    ≤ (2:ℝ)^(-23:ℤ)*(64+a)*(e + (2:ℝ)^(-121:ℤ))*(p + (2:ℝ)^(-6:ℤ)) := by
  have h23 : (2:ℝ)^(-23:ℤ) = 1/8388608 := by norm_num
  have h24 : (2:ℝ)^(-24:ℤ) = 1/16777216 := by norm_num
  have h19 : (2:ℝ)^(-19:ℤ) = 1/524288 := by norm_num
  have h140 : (2:ℝ)^(-140:ℤ) = 1/1393796574908163946345982392040522594123776 := by norm_num
  have h150 : (2:ℝ)^(-150:ℤ) = 1/1427247692705959881058285969449495136382746624 := by norm_num
  have h121 : (2:ℝ)^(-121:ℤ) = 1/2658455991569831745807614120560689152 := by norm_num
  have h6 : (2:ℝ)^(-6:ℤ) = 1/64 := by norm_num
  rw [h23, h24, h19, h140, h150, h121, h6]
  have hid : (1/16777216) * ((e*(1 + 2*(1/8388608) + (1/8388608)*a) + 1/1393796574908163946345982392040522594123776) *
      (p*(1 + 52*(1/8388608)) + (1/16777216)*(1 + 1/524288)))
    + ((e*(1 + 2*(1/8388608) + (1/8388608)*a) + 1/1393796574908163946345982392040522594123776) *
      (52*(1/8388608)*p + (1/16777216)*(1 + 1/524288)))
    + ((e*(2*(1/8388608) + (1/8388608)*a) + 1/1393796574908163946345982392040522594123776)*p)
    + 1/1427247692705959881058285969449495136382746624
    = e*p*(16777216⁻¹*(1 + 2*8388608⁻¹ + 8388608⁻¹*a)*(1 + 52*8388608⁻¹) + (1 + 2*8388608⁻¹ + 8388608⁻¹*a)*(52*8388608⁻¹) + (2*8388608⁻¹ + 8388608⁻¹*a))
    + e*(16777216⁻¹*(1 + 2*8388608⁻¹ + 8388608⁻¹*a)*(16777216⁻¹*(1 + 524288⁻¹)) + (1 + 2*8388608⁻¹ + 8388608⁻¹*a)*(16777216⁻¹*(1 + 524288⁻¹)))
    + p*(16777216⁻¹*1393796574908163946345982392040522594123776⁻¹*(1 + 52*8388608⁻¹) + 1393796574908163946345982392040522594123776⁻¹*(52*8388608⁻¹) + 1393796574908163946345982392040522594123776⁻¹)
    + (16777216⁻¹*1393796574908163946345982392040522594123776⁻¹*(16777216⁻¹*(1 + 524288⁻¹)) + 1393796574908163946345982392040522594123776⁻¹*(16777216⁻¹*(1 + 524288⁻¹)) + 1427247692705959881058285969449495136382746624⁻¹) := by ring
  rw [hid]
  have hC1 : 16777216⁻¹*(1 + 2*8388608⁻¹ + 8388608⁻¹*a)*(1 + 52*8388608⁻¹) + (1 + 2*8388608⁻¹ + 8388608⁻¹*a)*(52*8388608⁻¹) + (2*8388608⁻¹ + 8388608⁻¹*a) ≤ 8388608⁻¹*(64+a) := by
    linarith
  have hC2 : 16777216⁻¹*(1 + 2*8388608⁻¹ + 8388608⁻¹*a)*(16777216⁻¹*(1 + 524288⁻¹)) + (1 + 2*8388608⁻¹ + 8388608⁻¹*a)*(16777216⁻¹*(1 + 524288⁻¹)) ≤ 8388608⁻¹*(64+a)*(1/64) := by
    linarith
  have hC3 : 16777216⁻¹*1393796574908163946345982392040522594123776⁻¹*(1 + 52*8388608⁻¹) + 1393796574908163946345982392040522594123776⁻¹*(52*8388608⁻¹) + 1393796574908163946345982392040522594123776⁻¹ ≤ 8388608⁻¹*(64+a)*(1/2658455991569831745807614120560689152) := by
    linarith
  have hC4 : 16777216⁻¹*1393796574908163946345982392040522594123776⁻¹*(16777216⁻¹*(1 + 524288⁻¹)) + 1393796574908163946345982392040522594123776⁻¹*(16777216⁻¹*(1 + 524288⁻¹)) + 1427247692705959881058285969449495136382746624⁻¹ ≤ 8388608⁻¹*(64+a)*(1/170141183460469231731687303715884105728) := by
    linarith
  have hep : 0 ≤ e*p := (mul_pos he hp).le
  have h1 := mul_le_mul_of_nonneg_left hC1 hep
  have h2 := mul_le_mul_of_nonneg_left hC2 he.le
  have h3 := mul_le_mul_of_nonneg_left hC3 hp.le
  nlinarith [h1, h2, h3, hC4]

/-- **`g`, finite case.** -/
theorem g_bound {A x0 E S gv : ℝ}
    (hA : |A| ≤ 89)
    (hE : |E - Real.exp A| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|) * Real.exp A + 2 ^ (-140 : ℤ))
    (hS : |S - spRef x0| ≤ 52 * 2 ^ (-23 : ℤ) * spRef x0 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)))
    (hgv : |gv + E * S| ≤ 2 ^ (-24 : ℤ) * |E * S| + 2 ^ (-150 : ℤ)) :
    |gv - gRef A x0| ≤ gTol A x0 := by
  have h22 : (2:ℝ)^(-22:ℤ) = 1/4194304 := by norm_num
  have h23 : (2:ℝ)^(-23:ℤ) = 1/8388608 := by norm_num
  have h24 : (2:ℝ)^(-24:ℤ) = 1/16777216 := by norm_num
  have h19 : (2:ℝ)^(-19:ℤ) = 1/524288 := by norm_num
  have h140 : (2:ℝ)^(-140:ℤ) = 1/1393796574908163946345982392040522594123776 := by norm_num
  have h150 : (2:ℝ)^(-150:ℤ) = 1/1427247692705959881058285969449495136382746624 := by norm_num
  have h121 : (2:ℝ)^(-121:ℤ) = 1/2658455991569831745807614120560689152 := by norm_num
  have h6 : (2:ℝ)^(-6:ℤ) = 1/64 := by norm_num
  have ha0 : (0:ℝ) ≤ |A| := abs_nonneg A
  have hmin : min |A| 128 = |A| := min_eq_left (by linarith)
  have hepos : 0 < Real.exp A := Real.exp_pos A
  have hppos : 0 < spRef x0 := spRef_pos x0
  have hEle : E - Real.exp A ≤ (1/4194304 + (1/8388608) * |A|) * Real.exp A + 1/1393796574908163946345982392040522594123776 := by
    rw [← h22, ← h23, ← h140]; exact (abs_le.mp hE).2
  have hEge : -((1/4194304 + (1/8388608) * |A|) * Real.exp A + 1/1393796574908163946345982392040522594123776) ≤ E - Real.exp A := by
    rw [← h22, ← h23, ← h140]; exact (abs_le.mp hE).1
  have hEabs : |E| ≤ Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776 := by
    have htri : |E| ≤ |E - Real.exp A| + Real.exp A := by
      have h : |E| = |(E - Real.exp A) + Real.exp A| := congrArg abs (by ring)
      rw [h]
      have h4 := abs_add_le (E - Real.exp A) (Real.exp A)
      rwa [abs_of_pos hepos] at h4
    have h5 : |E - Real.exp A| ≤ (1/4194304 + (1/8388608) * |A|) * Real.exp A + 1/1393796574908163946345982392040522594123776 := by
      rw [abs_le]
      exact ⟨hEge, hEle⟩
    linarith [htri, h5]
  have hEe : |E - Real.exp A| ≤ Real.exp A*(2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776 := by
    rw [abs_le]
    constructor <;> linarith [hEle, hEge]
  have hSle : S - spRef x0 ≤ 52*(1/8388608)*spRef x0 + (1/16777216)*(1 + 1/524288) := by
    rw [← h23, ← h24, ← h19]; exact (abs_le.mp hS).2
  have hSge : -(52*(1/8388608)*spRef x0 + (1/16777216)*(1 + 1/524288)) ≤ S - spRef x0 := by
    rw [← h23, ← h24, ← h19]; exact (abs_le.mp hS).1
  have hSabs : |S| ≤ spRef x0*(1 + 52*(1/8388608)) + (1/16777216)*(1 + 1/524288) := by
    rw [abs_le]
    constructor <;> linarith [hSle, hSge]
  have hSp : |S - spRef x0| ≤ 52*(1/8388608)*spRef x0 + (1/16777216)*(1 + 1/524288) := by
    rw [abs_le]; exact ⟨hSge, hSle⟩
  have htri : |gv - gRef A x0| ≤
      (1/16777216) * |E| * |S| + 1/1427247692705959881058285969449495136382746624
        + |E| * |S - spRef x0| + |E - Real.exp A| * spRef x0 := by
    have habs : gv - gRef A x0 = gv + Real.exp A * spRef x0 := by unfold gRef; ring
    rw [habs]
    have h1 : |gv + Real.exp A * spRef x0| ≤ |gv + E*S| + |E*S - Real.exp A*spRef x0| := by
      have h : gv + Real.exp A * spRef x0 = (gv + E*S) - (E*S - Real.exp A*spRef x0) := by ring
      rw [h]; exact abs_sub _ _
    have h2 : |gv + E*S| ≤ (1/16777216) * |E*S| + 1/1427247692705959881058285969449495136382746624 := by
      have h := hgv
      rw [h24, h150] at h
      exact h
    have h3 : |E*S - Real.exp A*spRef x0| ≤ |E| * |S - spRef x0| + |E - Real.exp A| * spRef x0 := by
      have h : E*S - Real.exp A*spRef x0 = E*(S - spRef x0) + (E - Real.exp A)*spRef x0 := by ring
      rw [h]
      have h4 := abs_add_le (E*(S - spRef x0)) ((E - Real.exp A)*spRef x0)
      rw [abs_mul, abs_mul, abs_of_nonneg hppos.le] at h4
      exact h4
    have h5 : |E*S| = |E| * |S| := abs_mul E S
    rw [h5] at h2
    linarith [h1, h2, h3]
  have hK1 : (0:ℝ) ≤ Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776 := by positivity
  have hK2 : (0:ℝ) ≤ spRef x0*(1 + 52*(1/8388608)) + (1/16777216)*(1 + 1/524288) := by positivity
  have hK3 : (0:ℝ) ≤ 52*(1/8388608)*spRef x0 + (1/16777216)*(1 + 1/524288) := by positivity
  have hK4 : (0:ℝ) ≤ Real.exp A*(2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776 := by positivity
  have e1 : (1/16777216) * |E| * |S| ≤ (1/16777216) *
      ((Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776) *
       (spRef x0*(1 + 52*(1/8388608)) + (1/16777216)*(1 + 1/524288))) := by
    have hp1 : |E| * |S| ≤ (Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776) *
       (spRef x0*(1 + 52*(1/8388608)) + (1/16777216)*(1 + 1/524288)) :=
      mul_le_mul hEabs hSabs (abs_nonneg S) hK1
    calc (1/16777216) * |E| * |S| = (1/16777216) * (|E| * |S|) := by ring
      _ ≤ (1/16777216) * ((Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776) *
       (spRef x0*(1 + 52*(1/8388608)) + (1/16777216)*(1 + 1/524288))) := by
          exact mul_le_mul_of_nonneg_left hp1 (by norm_num)
  have e2 : |E| * |S - spRef x0| ≤ (Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776) *
       (52*(1/8388608)*spRef x0 + (1/16777216)*(1 + 1/524288)) :=
    mul_le_mul hEabs hSp (abs_nonneg _) hK1
  have e3 : |E - Real.exp A| * spRef x0 ≤ (Real.exp A*(2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776) * spRef x0 :=
    mul_le_mul hEe le_rfl hppos.le hK4
  have harith := gmain_arith_bound hepos hppos ha0 hA
  rw [h23, h24, h19, h140, h150, h121, h6] at harith
  have hgoal : gTol A x0 = (1/8388608)*(64+|A|)*(Real.exp A + 1/2658455991569831745807614120560689152)*(spRef x0 + 1/64) := by
    unfold gTol gMag gEps
    rw [hmin, h23, h121, h6]
    ring
  rw [hgoal]
  linarith [htri, e1, e2, e3, harith]

/-- **`g`, overflow.** The exact product is at or above `maxVal`, and the stored value is `-∞`. -/
theorem g_inf {A x0 E S : ℝ}
    (hA : |A| ≤ 89)
    (hE : |E - Real.exp A| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|) * Real.exp A + 2 ^ (-140 : ℤ))
    (hS : |S - spRef x0| ≤ 52 * 2 ^ (-23 : ℤ) * spRef x0 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)))
    (hbig : (1 - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) ≤ |E * S|) :
    (Format.binary32.maxVal : ℝ) ≤ |gRef A x0| + gTol A x0 := by
  have h22 : (2:ℝ)^(-22:ℤ) = 1/4194304 := by norm_num
  have h22b : (2:ℝ)^(-22:ℤ) = 2*(1/8388608) := by norm_num
  have h23 : (2:ℝ)^(-23:ℤ) = 1/8388608 := by norm_num
  have h24 : (2:ℝ)^(-24:ℤ) = 1/16777216 := by norm_num
  have h19 : (2:ℝ)^(-19:ℤ) = 1/524288 := by norm_num
  have h140 : (2:ℝ)^(-140:ℤ) = 1/1393796574908163946345982392040522594123776 := by norm_num
  have h121 : (2:ℝ)^(-121:ℤ) = 1/2658455991569831745807614120560689152 := by norm_num
  have h6 : (2:ℝ)^(-6:ℤ) = 1/64 := by norm_num
  have ha0 : (0:ℝ) ≤ |A| := abs_nonneg A
  have hmin : min |A| 128 = |A| := min_eq_left (by linarith)
  have hepos : 0 < Real.exp A := Real.exp_pos A
  have hppos : 0 < spRef x0 := spRef_pos x0
  have hEle : E - Real.exp A ≤ (1/4194304 + (1/8388608) * |A|) * Real.exp A + 1/1393796574908163946345982392040522594123776 := by
    rw [← h22, ← h23, ← h140]; exact (abs_le.mp hE).2
  have hEge : -((1/4194304 + (1/8388608) * |A|) * Real.exp A + 1/1393796574908163946345982392040522594123776) ≤ E - Real.exp A := by
    rw [← h22, ← h23, ← h140]; exact (abs_le.mp hE).1
  have hEabs : |E| ≤ Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776 := by
    have htri : |E| ≤ |E - Real.exp A| + Real.exp A := by
      have h : |E| = |(E - Real.exp A) + Real.exp A| := congrArg abs (by ring)
      rw [h]
      have h4 := abs_add_le (E - Real.exp A) (Real.exp A)
      rwa [abs_of_pos hepos] at h4
    have h5 : |E - Real.exp A| ≤ (1/4194304 + (1/8388608) * |A|) * Real.exp A + 1/1393796574908163946345982392040522594123776 := by
      rw [abs_le]
      exact ⟨hEge, hEle⟩
    linarith [htri, h5]
  have hSle : S - spRef x0 ≤ 52*(1/8388608)*spRef x0 + (1/16777216)*(1 + 1/524288) := by
    rw [← h23, ← h24, ← h19]; exact (abs_le.mp hS).2
  have hSge : -(52*(1/8388608)*spRef x0 + (1/16777216)*(1 + 1/524288)) ≤ S - spRef x0 := by
    rw [← h23, ← h24, ← h19]; exact (abs_le.mp hS).1
  have hSabs : |S| ≤ spRef x0*(1 + 52*(1/8388608)) + (1/16777216)*(1 + 1/524288) := by
    rw [abs_le]
    constructor <;> linarith [hSle, hSge]
  have hK1 : (0:ℝ) ≤ Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776 := by positivity
  have hbig' : (1 - 2*(1/8388608)) * (Format.binary32.maxVal : ℝ) ≤ |E| * |S| := by
    have h := hbig
    rw [h22b, abs_mul] at h
    exact h
  have hES : (1 - 2*(1/8388608)) * (Format.binary32.maxVal : ℝ) ≤
      (Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776) *
      (spRef x0*(1 + 52*(1/8388608)) + (1/16777216)*(1 + 1/524288)) :=
    le_trans hbig' (mul_le_mul hEabs hSabs (abs_nonneg S) hK1)
  have key : (Real.exp A*(1 + 2*(1/8388608) + (1/8388608)*|A|) + 1/1393796574908163946345982392040522594123776) *
      (spRef x0*(1 + 52*(1/8388608)) + (1/16777216)*(1 + 1/524288)) ≤
      (1 - 2*(1/8388608)) * (Real.exp A * spRef x0 + (1/8388608)*(64+|A|)*(Real.exp A + 1/2658455991569831745807614120560689152)*(spRef x0 + 1/64)) := by
    have hK : 0 ≤ (1 - 2*(1/8388608)) + (1 - 2*(1/8388608))*(1/8388608)*(64+|A|) - (1 + 2*(1/8388608) + (1/8388608)*|A|)*(1 + 52*(1/8388608)) := by linarith
    have hK2 : 0 ≤ (1 - 2*(1/8388608))*(1/8388608)*(64+|A|)*(1/64) - (1 + 2*(1/8388608) + (1/8388608)*|A|)*((1/16777216)*(1 + 1/524288)) := by linarith
    have hK3 : 0 ≤ (1 - 2*(1/8388608))*(1/8388608)*(64+|A|)*(1/2658455991569831745807614120560689152) - (1/1393796574908163946345982392040522594123776)*(1 + 52*(1/8388608)) := by linarith
    have hK4 : 0 ≤ (1 - 2*(1/8388608))*(1/8388608)*(64+|A|)*(1/2658455991569831745807614120560689152)*(1/64) - (1/1393796574908163946345982392040522594123776)*((1/16777216)*(1 + 1/524288)) := by linarith
    have hep : 0 ≤ Real.exp A * spRef x0 := (mul_pos hepos hppos).le
    have h1 := mul_nonneg hK hep
    have h2 := mul_nonneg hK2 hepos.le
    have h3 := mul_nonneg hK3 hppos.le
    nlinarith [h1, h2, h3, hK4]
  have h2 : (1 - 2*(1/8388608)) * (Format.binary32.maxVal : ℝ) ≤
      (1 - 2*(1/8388608)) * (Real.exp A * spRef x0 + (1/8388608)*(64+|A|)*(Real.exp A + 1/2658455991569831745807614120560689152)*(spRef x0 + 1/64)) :=
    le_trans hES key
  have hpos : (0:ℝ) < 1 - 2*(1/8388608) := by norm_num
  have hMle : (Format.binary32.maxVal : ℝ) ≤
      Real.exp A * spRef x0 + (1/8388608)*(64+|A|)*(Real.exp A + 1/2658455991569831745807614120560689152)*(spRef x0 + 1/64) :=
    le_of_mul_le_mul_left h2 hpos
  have hgr : |gRef A x0| = Real.exp A * spRef x0 := by
    unfold gRef
    rw [abs_neg, abs_of_pos (mul_pos hepos hppos)]
  have hgoal : gTol A x0 = (1/8388608)*(64+|A|)*(Real.exp A + 1/2658455991569831745807614120560689152)*(spRef x0 + 1/64) := by
    unfold gTol gMag gEps
    rw [hmin, h23, h121, h6]
    ring
  rw [hgr, hgoal]
  exact hMle

end Claims.GdnGating
