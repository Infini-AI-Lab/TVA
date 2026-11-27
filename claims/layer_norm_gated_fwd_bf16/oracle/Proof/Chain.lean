import TVA.Proof

/-!
# `layer_norm_gated_fwd_bf16`: error propagation through a chain of products

The output `y = bf16(((x·r)·w)·g·s)` is a chain of roundings and products. Each intermediate `a`
is tracked against its exact counterpart `A` by a bound `|a − A| ≤ ε·|A| + κ·m`: a relative part
`ε` and a floor `κ` in units of a magnitude `m ≥ 1` (a product of `1 + |operand|` factors). The
three steps:

* `round_step`: one rounding `f` of `q` (`|f − q| ≤ u'·|q| + η'`);
* `exact_step`: a rounded product `f = fl(a·b)` by an exact operand `b`, the magnitude growing to
  `m·(1 + |b|)`;
* `approx_prod`: a product `a·b` by an approximate `b ≈ B` with `0 ≤ B ≤ 1` (the gate's
  `sigmoid`), the magnitude growing to `K·m` with `|A| ≤ K·m`.
-/

namespace Claims.LayerNormGatedFwd

/-- One rounding of a tracked value: `|f − q| ≤ u'·|q| + η'` and `|q − Q| ≤ ε·|Q| + κ·m` give
`|f − Q| ≤ (ε + u'·(1 + ε))·|Q| + (κ·(1 + u') + η')·m`. -/
theorem round_step {f q Q ε κ m u' η' : ℝ} (hu : 0 ≤ u') (hη0 : 0 ≤ η') (hm : 1 ≤ m) (hκ : 0 ≤ κ)
    (hf : |f - q| ≤ u' * |q| + η') (hq : |q - Q| ≤ ε * |Q| + κ * m) :
    |f - Q| ≤ (ε + u' * (1 + ε)) * |Q| + (κ * (1 + u') + η') * m := by
  have hqa : |q| ≤ |Q| + |q - Q| := by
    have h := abs_add_le Q (q - Q)
    rwa [show Q + (q - Q) = q by ring] at h
  have htri : |f - Q| ≤ |f - q| + |q - Q| := by
    have h := abs_add_le (f - q) (q - Q)
    rwa [show f - q + (q - Q) = f - Q by ring] at h
  have hη : η' ≤ η' * m := by nlinarith
  have h1 : u' * |q| ≤ u' * (|Q| + (ε * |Q| + κ * m)) :=
    mul_le_mul_of_nonneg_left (le_trans hqa (by linarith)) hu
  nlinarith [abs_nonneg Q]

/-- A product by an exact operand: `|a·b − A·b| ≤ ε·|A·b| + κ·(m·(1 + |b|))`. -/
theorem mul_exact {a A b ε κ m : ℝ} (hm : 1 ≤ m) (hκ : 0 ≤ κ) (ha : |a - A| ≤ ε * |A| + κ * m) :
    |a * b - A * b| ≤ ε * |A * b| + κ * (m * (1 + |b|)) := by
  rw [show a * b - A * b = (a - A) * b by ring, abs_mul, abs_mul]
  have h := mul_le_mul_of_nonneg_right ha (abs_nonneg b)
  have h2 : κ * m * |b| ≤ κ * (m * (1 + |b|)) := by
    have : m * |b| ≤ m * (1 + |b|) := mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    nlinarith
  nlinarith

/-- A rounded product by an exact operand: `f = fl(a·b)` tracked against `A·b`, the magnitude
growing to `m·(1 + |b|)`. -/
theorem exact_step {f a A b ε κ m u η : ℝ} (hu : 0 ≤ u) (hη0 : 0 ≤ η) (hm : 1 ≤ m) (hκ : 0 ≤ κ)
    (ha : |a - A| ≤ ε * |A| + κ * m) (hf : |f - a * b| ≤ u * |a * b| + η) :
    |f - A * b| ≤ (ε + u * (1 + ε)) * |A * b| + (κ * (1 + u) + η) * (m * (1 + |b|)) := by
  have hm' : 1 ≤ m * (1 + |b|) := by nlinarith [abs_nonneg b]
  exact round_step hu hη0 hm' hκ hf (mul_exact hm hκ ha)

/-- A product by an approximate operand `b ≈ B` (`|b − B| ≤ γ·B + δ`, `0 ≤ B ≤ 1`), with
`|A| ≤ K·m`: `|a·b − A·B| ≤ (ε·(1 + γ) + γ)·|A·B| + ((1 + ε)·δ + κ·(1 + γ + δ))·(K·m)`. -/
theorem approx_prod {a A b B ε κ m γ δ K : ℝ} (hm : 1 ≤ m) (hK : 1 ≤ K) (hκ : 0 ≤ κ)
    (hε : 0 ≤ ε) (hγ : 0 ≤ γ) (hδ : 0 ≤ δ) (hB0 : 0 ≤ B) (hB1 : B ≤ 1) (hAK : |A| ≤ K * m)
    (ha : |a - A| ≤ ε * |A| + κ * m) (hb : |b - B| ≤ γ * B + δ) :
    |a * b - A * B| ≤ (ε * (1 + γ) + γ) * |A * B| + ((1 + ε) * δ + κ * (1 + γ + δ)) * (K * m) := by
  have hbabs : |b| ≤ (1 + γ) * B + δ := by
    have h := abs_add_le B (b - B)
    rw [show B + (b - B) = b by ring, abs_of_nonneg hB0] at h
    linarith
  have hb1 : |b| ≤ 1 + γ + δ := by nlinarith
  have hdec : a * b - A * B = (a - A) * b + A * (b - B) := by ring
  have htri : |a * b - A * B| ≤ |a - A| * |b| + |A| * |b - B| := by
    rw [hdec]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul]
  have hAB : |A * B| = |A| * B := by rw [abs_mul, abs_of_nonneg hB0]
  have h1 : |a - A| * |b| ≤ (ε * |A|) * ((1 + γ) * B + δ) + κ * m * (1 + γ + δ) := by
    have e1 : |a - A| * |b| ≤ (ε * |A| + κ * m) * |b| :=
      mul_le_mul_of_nonneg_right ha (abs_nonneg _)
    have e2 : ε * |A| * |b| ≤ ε * |A| * ((1 + γ) * B + δ) :=
      mul_le_mul_of_nonneg_left hbabs (by positivity)
    have e3 : κ * m * |b| ≤ κ * m * (1 + γ + δ) :=
      mul_le_mul_of_nonneg_left hb1 (by positivity)
    nlinarith
  have h2 : |A| * |b - B| ≤ |A| * (γ * B + δ) := mul_le_mul_of_nonneg_left hb (abs_nonneg _)
  have hAK' : (ε * δ + δ) * |A| ≤ (ε * δ + δ) * (K * m) :=
    mul_le_mul_of_nonneg_left hAK (by positivity)
  have hm' : κ * m * (1 + γ + δ) ≤ κ * (1 + γ + δ) * (K * m) := by
    have : m ≤ K * m := by nlinarith
    have hp : 0 ≤ κ * (1 + γ + δ) := by positivity
    nlinarith
  rw [hAB]
  nlinarith [abs_nonneg A]

/-- A rounded product by an approximate operand: `approx_prod`, then `round_step`. -/
theorem approx_step {f a A b B ε κ m γ δ K u η : ℝ} (hu : 0 ≤ u) (hη0 : 0 ≤ η) (hm : 1 ≤ m)
    (hK : 1 ≤ K)
    (hκ : 0 ≤ κ) (hε : 0 ≤ ε) (hγ : 0 ≤ γ) (hδ : 0 ≤ δ) (hB0 : 0 ≤ B) (hB1 : B ≤ 1)
    (hAK : |A| ≤ K * m) (ha : |a - A| ≤ ε * |A| + κ * m) (hb : |b - B| ≤ γ * B + δ)
    (hf : |f - a * b| ≤ u * |a * b| + η) :
    |f - A * B| ≤ ((ε * (1 + γ) + γ) + u * (1 + (ε * (1 + γ) + γ))) * |A * B| +
      (((1 + ε) * δ + κ * (1 + γ + δ)) * (1 + u) + η) * (K * m) := by
  have hKm : 1 ≤ K * m := by nlinarith
  have hκ' : 0 ≤ (1 + ε) * δ + κ * (1 + γ + δ) := by positivity
  exact round_step hu hη0 hKm hκ' hf (approx_prod hm hK hκ hε hγ hδ hB0 hB1 hAK ha hb)

/-- The magnitude a tracked value is at most: `|a| ≤ (1 + ε)·|A| + κ·m`. -/
theorem abs_le_of_track {a A ε κ m : ℝ} (ha : |a - A| ≤ ε * |A| + κ * m) :
    |a| ≤ (1 + ε) * |A| + κ * m := by
  have h := abs_add_le A (a - A)
  rw [show A + (a - A) = a by ring] at h
  linarith

/-- Weaken a tracked bound to larger constants. -/
theorem track_mono {a A ε κ m ε' κ' : ℝ} (h : |a - A| ≤ ε * |A| + κ * m) (hε : ε ≤ ε')
    (hκ : κ ≤ κ') (hm : 0 ≤ m) : |a - A| ≤ ε' * |A| + κ' * m := by
  have h1 : ε * |A| ≤ ε' * |A| := mul_le_mul_of_nonneg_right hε (abs_nonneg A)
  have h2 : κ * m ≤ κ' * m := mul_le_mul_of_nonneg_right hκ hm
  linarith

/-- `|a·b| ≤ A·B` from `|a| ≤ A` and `|b| ≤ B`. -/
theorem abs_mul_le_of {a b A B : ℝ} (ha : |a| ≤ A) (hb : |b| ≤ B) : |a * b| ≤ A * B := by
  rw [abs_mul]
  exact mul_le_mul ha hb (abs_nonneg b) (le_trans (abs_nonneg a) ha)

end Claims.LayerNormGatedFwd
