import TVA.Proof.Softmax.Exact

/-!
# The online softmax, with rounding

A kernel's `l` and `acc` are floats: each trip rescales, adds a block of weights and rounds. The
computed state therefore follows the exact recurrence of `TVA.Proof.Softmax.Exact` only up to an
error per trip. This module bounds how those errors add up and what they do to the output ratio.

**The idea.** Take the computed references `m̂` as *the* references: the exact invariant holds for
any references, so the exact sums `lsum`, `asum` at `m̂ t` are the right targets, and the only
errors left are those of the rescaling, the block's weights and the roundings. A trip with a
relative error `ε` (on the magnitude it works with) and an absolute error `η t` multiplies the
error so far by `(1 + ε)·α t` and adds `ε·(magnitude) + η t`. Since the references of a running
maximum never decrease, `α ≤ 1`, so no error is amplified by the rescaling and the total stays
linear in the number of trips.

* `absErr`, `bound`, `alpha`: the running error bound, `((1 + ε)^t − 1)·M + absErr`, where `M`
  is the exact state's magnitude and `absErr` carries each trip's absolute error forward, scaled by
  later `α`s (closed form `absErr_alpha`).
* `accum`: the bound holds for any state `x̂` following `X (t+1) = α t·X t + Y t` up to
  `ε·(α t·|x̂ t| + Z t) + η t` per trip. `l_err` and `acc_err` are its instances for `l` and `acc`.
* `bound_le`: when `α ≤ 1` and `t·ε ≤ 1/2`, the bound is at most `2tε·M + 2·Σ η`.
* `ratio_err`, `ratio_err_half`: the ratio `â/l̂` against `A/L` from the two errors.
* `err_final`, `err_final_mono`: the output `âcc n / l̂ n` against `wavg`.
* One trip from its parts: `scale_err` (`α̂·x`), `block_err`, `block_abs_le` (a block of lanes,
  each within `ρ·p + η` of an exact weight `p ≥ 0`), `block_err_one`, `block_abs_le_one` (the same
  for `l`), `approx_trans` (one approximation after another), `abs_le_of_approx`, `step_err` (the
  scaled state plus the block's contribution, through an addition, an `fma` or a `tl.dot`
  accumulation) and `l_trip` (the trip of `l`, assembled).

**How a kernel uses it.** Per trip, bound the computed `α̂`, the weights `p̂ = exp(…)`
(`abs_exp_round_le`), the block sum or `tl.dot` (the device facts), and the final rounding; combine
them with `step_err` (or `l_trip`) into the step hypothesis of `l_err` and `acc_err`; then
`err_final_mono` gives the output's error, and `wavg_blocks` turns its target into the claim's
flat specification.
-/

namespace TVA

namespace Softmax

open Finset

/-! ## The running bound -/

/-- The absolute part of the running bound: `absErr 0 = 0`,
`absErr (t+1) = (1 + ε)·α t·absErr t + η t`. Each trip's absolute error `η t` is carried to
later trips, scaled by their `α` (`absErr_alpha`). -/
noncomputable def absErr (ε : ℝ) (α η : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => (1 + ε) * α t * absErr ε α η t + η t

/-- **The running bound** after `t` trips, for a magnitude `M`:
`((1 + ε)^t − 1)·M + absErr ε α η t`. -/
noncomputable def bound (ε : ℝ) (α η : ℕ → ℝ) (t : ℕ) (M : ℝ) : ℝ :=
  ((1 + ε) ^ t - 1) * M + absErr ε α η t

/-- The rescaling factor of trip `t` for references `m`: `α t = exp (m t − m (t+1))`, the
`tl.exp(m_old − m_new)` (or `exp2`) an attention kernel multiplies `l` and `acc` by. -/
noncomputable def alpha (m : ℕ → ℝ) (t : ℕ) : ℝ := Real.exp (m t - m (t + 1))

/-- The rescaling factor is positive. -/
theorem alpha_pos (m : ℕ → ℝ) (t : ℕ) : 0 < alpha m t := Real.exp_pos _

/-- `α t ≤ 1` when the reference does not decrease at trip `t` (a running maximum). -/
theorem alpha_le_one {m : ℕ → ℝ} {t : ℕ} (h : m t ≤ m (t + 1)) : alpha m t ≤ 1 :=
  Real.exp_le_one_iff.mpr (by linarith)

/-- The absolute part is non-negative for non-negative factors and errors. -/
theorem absErr_nonneg {ε : ℝ} {α η : ℕ → ℝ} (hε : 0 ≤ ε) :
    ∀ {t : ℕ}, (∀ u < t, 0 ≤ α u) → (∀ u < t, 0 ≤ η u) → 0 ≤ absErr ε α η t
  | 0, _, _ => le_refl _
  | t + 1, hα, hη => by
    simp only [absErr]
    have hc := absErr_nonneg hε (t := t) (fun u hu => hα u (by omega)) (fun u hu => hη u (by omega))
    exact add_nonneg (mul_nonneg (mul_nonneg (by linarith) (hα t (by omega))) hc) (hη t (by omega))

/-- **The absolute part when `α ≤ 1`:** if `0 ≤ α u ≤ 1` (for `0 < u`; `α 0` multiplies `0`) and
`η u ≥ 0`, then `absErr ε α η t ≤ (1 + ε)^t·Σ_{u<t} η u`. -/
theorem absErr_le {ε : ℝ} {α η : ℕ → ℝ} (hε : 0 ≤ ε) :
    ∀ {t : ℕ}, (∀ u < t, 0 ≤ α u) → (∀ u, 0 < u → u < t → α u ≤ 1) → (∀ u < t, 0 ≤ η u) →
      absErr ε α η t ≤ (1 + ε) ^ t * ∑ u ∈ range t, η u
  | 0, _, _, _ => by simp [absErr]
  | t + 1, hα0, hα1, hη => by
    simp only [absErr]
    have ih := absErr_le hε (t := t) (fun u hu => hα0 u (by omega))
      (fun u hu hu' => hα1 u hu (by omega)) (fun u hu => hη u (by omega))
    have hc := absErr_nonneg hε (t := t) (fun u hu => hα0 u (by omega)) (fun u hu => hη u (by omega))
    have hP : 1 ≤ (1 + ε) ^ t := one_le_pow₀ (by linarith)
    have hηt := hη t (by omega)
    have hαc : α t * absErr ε α η t ≤ absErr ε α η t := by
      rcases Nat.eq_zero_or_pos t with rfl | ht
      · simp [absErr]
      · exact mul_le_of_le_one_left hc (hα1 t ht (by omega))
    rw [sum_range_succ, pow_succ]
    have f1 : (1 + ε) * (α t * absErr ε α η t) ≤ (1 + ε) * absErr ε α η t :=
      mul_le_mul_of_nonneg_left hαc (by linarith)
    have f2 : (1 + ε) * absErr ε α η t ≤ (1 + ε) * ((1 + ε) ^ t * ∑ u ∈ range t, η u) :=
      mul_le_mul_of_nonneg_left ih (by linarith)
    have f3 : η t ≤ (1 + ε) ^ t * (1 + ε) * η t :=
      le_mul_of_one_le_left hηt (by nlinarith)
    nlinarith

/-- **The absolute part in closed form**, for the factors `alpha m` of references `m`:
`absErr ε (alpha m) η t = Σ_{u<t} (1 + ε)^(t−1−u)·exp (m (u+1) − m t)·η u`. The error of trip
`u`, made at the reference `m (u+1)`, counts at `exp (m (u+1) − m t) ≤ 1` when the references
never decrease. Use it when an absolute error is large but early (a sharper bound than
`bound_le`). -/
theorem absErr_alpha (ε : ℝ) (m η : ℕ → ℝ) : ∀ t : ℕ,
    absErr ε (alpha m) η t =
      ∑ u ∈ range t, (1 + ε) ^ (t - 1 - u) * Real.exp (m (u + 1) - m t) * η u
  | 0 => by simp [absErr]
  | t + 1 => by
    simp only [absErr]
    rw [absErr_alpha ε m η t, sum_range_succ, mul_sum]
    congr 1
    · refine sum_congr rfl fun u hu => ?_
      have hu' : u < t := mem_range.mp hu
      have hpow : t + 1 - 1 - u = (t - 1 - u) + 1 := by omega
      have he : Real.exp (m (u + 1) - m (t + 1)) =
          Real.exp (m t - m (t + 1)) * Real.exp (m (u + 1) - m t) := by
        rw [← Real.exp_add]; congr 1; ring
      rw [hpow, pow_succ, he, alpha]
      ring
    · simp

/-- The running bound is non-negative for a non-negative magnitude, factors and errors. -/
theorem bound_nonneg {ε M : ℝ} {α η : ℕ → ℝ} {t : ℕ} (hε : 0 ≤ ε) (hM : 0 ≤ M)
    (hα : ∀ u < t, 0 ≤ α u) (hη : ∀ u < t, 0 ≤ η u) : 0 ≤ bound ε α η t M := by
  unfold bound
  have : 1 ≤ (1 + ε) ^ t := one_le_pow₀ (by linarith)
  exact add_nonneg (mul_nonneg (by linarith) hM) (absErr_nonneg hε hα hη)

/-- **The running bound, simplified.** When `0 ≤ α u ≤ 1` (for `0 < u`), `η u ≥ 0` and
`t·ε ≤ 1/2`, the running bound is at most `2tε·M + 2·Σ_{u<t} η u`. -/
theorem bound_le {ε M : ℝ} {α η : ℕ → ℝ} {t : ℕ} (hε : 0 ≤ ε) (ht : t * ε ≤ 1 / 2) (hM : 0 ≤ M)
    (hα0 : ∀ u < t, 0 ≤ α u) (hα1 : ∀ u, 0 < u → u < t → α u ≤ 1) (hη : ∀ u < t, 0 ≤ η u) :
    bound ε α η t M ≤ 2 * t * ε * M + 2 * ∑ u ∈ range t, η u := by
  unfold bound
  have hp := one_add_pow_le hε (n := t) (by linarith)
  have hc := absErr_le hε hα0 hα1 hη
  have hS : 0 ≤ ∑ u ∈ range t, η u := sum_nonneg fun u hu => hη u (mem_range.mp hu)
  have e1 : ((1 + ε) ^ t - 1) * M ≤ 2 * t * ε * M := mul_le_mul_of_nonneg_right (by linarith) hM
  have e2 : (1 + ε) ^ t * ∑ u ∈ range t, η u ≤ 2 * ∑ u ∈ range t, η u :=
    mul_le_mul_of_nonneg_right (by linarith) hS
  linarith

/-- **The running bound of a rescaled accumulation.** Let the exact state follow
`X (t+1) = α t·X t + Y t`, with a magnitude `M (t+1) = α t·M t + Z t`, `|Y t| ≤ Z t`, `α t ≥ 0`,
from `|X 0| ≤ M 0`. If the computed state starts at `x 0 = X 0` and, while its error so far is
within the bound, each trip is within `ε·(α t·|x t| + Z t) + η t` of `α t·x t + Y t`, then after
`t ≤ T` trips `|x t − X t| ≤ bound ε α η t (M t)`. The step may use the bound so far (to bound
`|x t|`, for an absolute error or the format's range). Use `l_err` or `acc_err` for the two states
of an online softmax, and this lemma for another rescaled state. -/
theorem accum {ε : ℝ} {α η X M Y Z x : ℕ → ℝ} {T : ℕ} (hε : 0 ≤ ε) (hα : ∀ t < T, 0 ≤ α t)
    (hx0 : x 0 = X 0) (hXM0 : |X 0| ≤ M 0)
    (hX : ∀ t < T, X (t + 1) = α t * X t + Y t) (hM : ∀ t < T, M (t + 1) = α t * M t + Z t)
    (hYZ : ∀ t < T, |Y t| ≤ Z t)
    (hstep : ∀ t < T, |x t - X t| ≤ bound ε α η t (M t) →
      |x (t + 1) - (α t * x t + Y t)| ≤ ε * (α t * |x t| + Z t) + η t) :
    ∀ t ≤ T, |x t - X t| ≤ bound ε α η t (M t) := by
  suffices key : ∀ t ≤ T, |X t| ≤ M t ∧ |x t - X t| ≤ bound ε α η t (M t) from
    fun t ht => (key t ht).2
  intro t
  induction t with
  | zero => intro _; exact ⟨hXM0, by simp [bound, absErr, hx0]⟩
  | succ t ih =>
    intro ht
    have htT : t < T := by omega
    obtain ⟨hXM, he⟩ := ih (by omega)
    have ha := hα t htT
    have hZ := hYZ t htT
    have hZ0 : 0 ≤ Z t := (abs_nonneg _).trans hZ
    have hs := hstep t htT he
    refine ⟨?_, ?_⟩
    · rw [hX t htT, hM t htT]
      calc |α t * X t + Y t| ≤ |α t * X t| + |Y t| := abs_add_le _ _
        _ ≤ α t * M t + Z t := by
          rw [abs_mul, abs_of_nonneg ha]
          exact add_le_add (mul_le_mul_of_nonneg_left hXM ha) hZ
    · rw [hX t htT, hM t htT]
      have htri : |x (t + 1) - (α t * X t + Y t)| ≤
          |x (t + 1) - (α t * x t + Y t)| + α t * |x t - X t| := by
        have e : x (t + 1) - (α t * X t + Y t) =
            (x (t + 1) - (α t * x t + Y t)) + α t * (x t - X t) := by ring
        rw [e]
        refine (abs_add_le _ _).trans (le_of_eq ?_)
        rw [abs_mul, abs_of_nonneg ha]
      have hxabs : |x t| ≤ M t + |x t - X t| := by
        have := abs_sub_abs_le_abs_sub (x t) (X t); linarith
      unfold bound at he ⊢
      simp only [absErr]
      rw [pow_succ]
      have hP1 : 1 ≤ (1 + ε) ^ t := one_le_pow₀ (by linarith)
      have h1 : ε * (α t * |x t|) ≤ ε * (α t * (M t + |x t - X t|)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hxabs ha) hε
      have h2 : (1 + ε) * (α t * |x t - X t|) ≤
          (1 + ε) * (α t * (((1 + ε) ^ t - 1) * M t + absErr ε α η t)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left he ha) (by linarith)
      have h3 : 0 ≤ ((1 + ε) ^ t - 1) * (1 + ε) * Z t :=
        mul_nonneg (mul_nonneg (by linarith) (by linarith)) hZ0
      nlinarith

/-! ## One trip from its parts -/

/-- **One approximation after another.** If `b` is within `ρ·Z + η` of `Y`, a magnitude `Bm` of `b`
is at most `(1 + ρ)·Z + η`, and `r` is within `ρ'·Bm + η'` of `b`, then `r` is within
`(ρ + ρ'·(1 + ρ))·Z + ((1 + ρ')·η + η')` of `Y`. For example `b = Σ p̂` (lanes within `ρ` of
exact `p`, `block_err_one`) and `r` its float sum (`Bm = Σ|p̂|`, `block_abs_le_one`); or a rounding
after a conversion. -/
theorem approx_trans {ρ ρ' Y Z b Bm r η η' : ℝ} (hρ' : 0 ≤ ρ')
    (hb : |b - Y| ≤ ρ * Z + η) (hBm : Bm ≤ (1 + ρ) * Z + η) (hr : |r - b| ≤ ρ' * Bm + η') :
    |r - Y| ≤ (ρ + ρ' * (1 + ρ)) * Z + ((1 + ρ') * η + η') := by
  have e : r - Y = (r - b) + (b - Y) := by ring
  rw [e]
  have h := mul_le_mul_of_nonneg_left hBm hρ'
  have := abs_add_le (r - b) (b - Y)
  nlinarith

/-- A magnitude from an approximation: `|b| ≤ (1 + ρ)·Z + η` when `|b − Y| ≤ ρ·Z + η` and
`|Y| ≤ Z`. -/
theorem abs_le_of_approx {ρ Y Z b η : ℝ} (hb : |b - Y| ≤ ρ * Z + η) (hYZ : |Y| ≤ Z) :
    |b| ≤ (1 + ρ) * Z + η := by
  have := abs_sub_abs_le_abs_sub b Y
  linarith

/-- **The scaled state.** If `α̂` is within `ρ·α + η` of `α`, then `α̂·x` is within
`ρ·(α·|x|) + η·|x|` of `α·x`: the rescaling `l·α̂`, `acc·α̂[:, None]` before rounding. -/
theorem scale_err {ρ η α α' x : ℝ} (h : |α' - α| ≤ ρ * α + η) :
    |α' * x - α * x| ≤ ρ * (α * |x|) + η * |x| := by
  rw [← sub_mul, abs_mul]
  have := mul_le_mul_of_nonneg_right h (abs_nonneg x)
  linarith

/-- **A block of lanes.** If each lane's `p̂ j` is within `ρ·p j + η` of an exact `p j ≥ 0`, then
`|Σ p̂·v − Σ p·v| ≤ ρ·Σ p·|v| + η·Σ |v|`: the error the weights bring into `acc`'s block term. A
masked lane has `p j = p̂ j = 0`. -/
theorem block_err {κ : Type*} {S : Finset κ} {p p' v : κ → ℝ} {ρ η : ℝ}
    (h : ∀ j ∈ S, |p' j - p j| ≤ ρ * p j + η) :
    |∑ j ∈ S, p' j * v j - ∑ j ∈ S, p j * v j| ≤
      ρ * ∑ j ∈ S, p j * |v j| + η * ∑ j ∈ S, |v j| := by
  rw [← sum_sub_distrib, mul_sum, mul_sum, ← sum_add_distrib]
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun j hj => ?_)
  rw [← sub_mul, abs_mul]
  have := mul_le_mul_of_nonneg_right (h j hj) (abs_nonneg (v j))
  linarith

/-- **The magnitude of a block:** under `block_err`'s conditions (and `p j ≥ 0`),
`Σ |p̂·v| ≤ (1 + ρ)·Σ p·|v| + η·Σ |v|`, the magnitude a `tl.dot` or a reduction is accurate
relative to. -/
theorem block_abs_le {κ : Type*} {S : Finset κ} {p p' v : κ → ℝ} {ρ η : ℝ}
    (hp : ∀ j ∈ S, 0 ≤ p j) (h : ∀ j ∈ S, |p' j - p j| ≤ ρ * p j + η) :
    ∑ j ∈ S, |p' j * v j| ≤ (1 + ρ) * ∑ j ∈ S, p j * |v j| + η * ∑ j ∈ S, |v j| := by
  rw [mul_sum, mul_sum, ← sum_add_distrib]
  refine sum_le_sum fun j hj => ?_
  rw [abs_mul]
  have h1 : |p' j| ≤ (1 + ρ) * p j + η := by
    have := abs_sub_abs_le_abs_sub (p' j) (p j)
    rw [abs_of_nonneg (hp j hj)] at this
    linarith [h j hj]
  have := mul_le_mul_of_nonneg_right h1 (abs_nonneg (v j))
  linarith

/-- **A block of lanes, for the sum `l`:** `|Σ p̂ − Σ p| ≤ ρ·Σ p + η·|S|`. -/
theorem block_err_one {κ : Type*} {S : Finset κ} {p p' : κ → ℝ} {ρ η : ℝ}
    (h : ∀ j ∈ S, |p' j - p j| ≤ ρ * p j + η) :
    |∑ j ∈ S, p' j - ∑ j ∈ S, p j| ≤ ρ * ∑ j ∈ S, p j + η * S.card := by
  have := block_err (v := fun _ => (1 : ℝ)) h
  simpa using this

/-- **The magnitude of a block, for the sum `l`:** `Σ |p̂| ≤ (1 + ρ)·Σ p + η·|S|`. -/
theorem block_abs_le_one {κ : Type*} {S : Finset κ} {p p' : κ → ℝ} {ρ η : ℝ}
    (hp : ∀ j ∈ S, 0 ≤ p j) (h : ∀ j ∈ S, |p' j - p j| ≤ ρ * p j + η) :
    ∑ j ∈ S, |p' j| ≤ (1 + ρ) * ∑ j ∈ S, p j + η * S.card := by
  have := block_abs_le (v := fun _ => (1 : ℝ)) hp h
  simpa using this

/-- **One trip.** The new state `x'` is computed from the scaled state `a` (within
`ρa·(α·|x|) + ηa` of `α·x`: `scale_err`, then `approx_trans` for a rounded product) and the block's
contribution `b` (within `ρb·Z + ηb` of the exact `Y`), through an operation within
`ρo·(|a| + Bm) + ηo` of `a + b`, where the magnitude `Bm` is at most `(1 + ρb)·Z + ηB`: an addition
or `fma` (`Bm = |b|`, `ηB = ηb` by `abs_le_of_approx`), or a `tl.dot` accumulation (`Bm = Σ|p̂·v|`
with `block_abs_le`, or the dot fact's magnitude). Then `x'` is within
`ε·(α·|x| + Z) + ((1 + ρo)·ηa + ηb + ρo·ηB + ηo)` of `α·x + Y`, for any `ε` above both rates. This
is the step hypothesis of `accum`, `l_err` and `acc_err`. -/
theorem step_err {ρa ρb ρo ε α x a b Y Z Bm ηa ηb ηB ηo x' : ℝ} (hρo : 0 ≤ ρo) (hα : 0 ≤ α)
    (hZ : 0 ≤ Z) (hεa : ρa + ρo * (1 + ρa) ≤ ε) (hεb : ρb + ρo * (1 + ρb) ≤ ε)
    (ha : |a - α * x| ≤ ρa * (α * |x|) + ηa) (hb : |b - Y| ≤ ρb * Z + ηb)
    (hBm : Bm ≤ (1 + ρb) * Z + ηB) (ho : |x' - (a + b)| ≤ ρo * (|a| + Bm) + ηo) :
    |x' - (α * x + Y)| ≤ ε * (α * |x| + Z) + ((1 + ρo) * ηa + ηb + ρo * ηB + ηo) := by
  have hαx : 0 ≤ α * |x| := mul_nonneg hα (abs_nonneg x)
  have hax : |a| ≤ (1 + ρa) * (α * |x|) + ηa := by
    have h1 := abs_sub_abs_le_abs_sub a (α * x)
    rw [abs_mul, abs_of_nonneg hα] at h1
    linarith
  have e : x' - (α * x + Y) = (x' - (a + b)) + (a - α * x) + (b - Y) := by ring
  rw [e]
  have h2 := mul_le_mul_of_nonneg_left (add_le_add hax hBm) hρo
  have h3 := (abs_add_le ((x' - (a + b)) + (a - α * x)) (b - Y)).trans
    (add_le_add (abs_add_le (x' - (a + b)) (a - α * x)) le_rfl)
  have h4 : (ρa + ρo * (1 + ρa)) * (α * |x|) ≤ ε * (α * |x|) := mul_le_mul_of_nonneg_right hεa hαx
  have h5 : (ρb + ρo * (1 + ρb)) * Z ≤ ε * Z := mul_le_mul_of_nonneg_right hεb hZ
  nlinarith

/-- **One trip of the sum `l`**, as attention kernels write it: `l' = l·α̂ + r̂` (an `fma` or a
product and a sum, within `ρo·(|α̂·l| + |r̂|) + ηo`), where `α̂` is within `ρα·α + ηα` of `α` and
`r̂` is a float sum of the lanes `p̂ j` (`tl.sum(p, 1)`, within `ρs·Σ|p̂| + ηs`), each within
`ρp·p j + ηp` of an exact `p j ≥ 0`. With `ρb = ρp + ρs·(1 + ρp)` and
`ηb = (1 + ρs)·ηp·|S| + ηs`, `l'` is within `ε·(α·|l| + Σ p) + ((1 + ρo)·(ηα·|l| + ηb) + ηo)` of
`α·l + Σ p`, for any `ε` above `ρα + ρo·(1 + ρα)` and `ρb + ρo·(1 + ρb)`. A masked lane has
`p j = p̂ j = 0`. -/
theorem l_trip {κ : Type*} {S : Finset κ} {p p' : κ → ℝ}
    {α α' l r l' ρp ρs ρα ρo ηp ηs ηα ηo ε : ℝ} (hα : 0 ≤ α) (hρs : 0 ≤ ρs) (hρo : 0 ≤ ρo)
    (hp0 : ∀ j ∈ S, 0 ≤ p j) (hp : ∀ j ∈ S, |p' j - p j| ≤ ρp * p j + ηp)
    (hr : |r - ∑ j ∈ S, p' j| ≤ ρs * ∑ j ∈ S, |p' j| + ηs) (hα' : |α' - α| ≤ ρα * α + ηα)
    (ho : |l' - (α' * l + r)| ≤ ρo * (|α' * l| + |r|) + ηo) (hε1 : ρα + ρo * (1 + ρα) ≤ ε)
    (hε2 : (ρp + ρs * (1 + ρp)) + ρo * (1 + (ρp + ρs * (1 + ρp))) ≤ ε) :
    |l' - (α * l + ∑ j ∈ S, p j)| ≤ ε * (α * |l| + ∑ j ∈ S, p j) +
      ((1 + ρo) * (ηα * |l| + ((1 + ρs) * (ηp * S.card) + ηs)) + ηo) := by
  have hb := approx_trans hρs (block_err_one hp) (block_abs_le_one hp0 hp) hr
  have hYZ : |∑ j ∈ S, p j| ≤ ∑ j ∈ S, p j := (abs_of_nonneg (sum_nonneg hp0)).le
  have := step_err hρo hα (sum_nonneg hp0) hε1 hε2 (scale_err hα') hb (abs_le_of_approx hb hYZ) ho
  linarith

/-! ## The online softmax with computed values -/

section Computed

variable {ι : Type*} {s v : ℕ → ι → ℝ} {B : ℕ → Finset ι}

/-- **The running sum with errors.** Let `m̂` be the computed references and `l̂` the computed sum
(as reals), with `l̂ 0 = 0`. If, while the error so far is within the bound, each trip is within
`ε·(α t·|l̂ t| + R t) + η t` of `α t·l̂ t + R t` (with `α = alpha m̂` and the exact block sum
`R t = Σ_{j ∈ B t} exp (s t j − m̂ (t+1))`), then for `t ≤ n`
`|l̂ t − lsum s B (m̂ t) t| ≤ bound ε (alpha m̂) η t (lsum s B (m̂ t) t)`. -/
theorem l_err {mh lh η : ℕ → ℝ} {ε : ℝ} {n : ℕ} (hε : 0 ≤ ε) (hl0 : lh 0 = 0)
    (hstep : ∀ t < n, |lh t - lsum s B (mh t) t| ≤ bound ε (alpha mh) η t (lsum s B (mh t) t) →
      |lh (t + 1) - (alpha mh t * lh t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))| ≤
        ε * (alpha mh t * |lh t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1))) + η t) :
    ∀ t ≤ n, |lh t - lsum s B (mh t) t| ≤ bound ε (alpha mh) η t (lsum s B (mh t) t) :=
  accum (X := fun t => lsum s B (mh t) t) (M := fun t => lsum s B (mh t) t)
    (Y := fun t => ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))
    (Z := fun t => ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))
    hε (fun t _ => (alpha_pos mh t).le) (by simp [hl0, lsum_zero]) (by simp [lsum_zero])
    (fun t _ => lsum_succ (mh t) (mh (t + 1)) t) (fun t _ => lsum_succ (mh t) (mh (t + 1)) t)
    (fun t _ => (abs_of_nonneg (sum_nonneg fun _ _ => (Real.exp_pos _).le)).le) hstep

/-- **The accumulator with errors** (one output column). With the computed references `m̂` and
accumulator `âcc` (`âcc 0 = 0`): if each trip is within
`ε·(α t·|âcc t| + Σ_{j ∈ B t} exp (s t j − m̂ (t+1))·|v t j|) + η t` of
`α t·âcc t + Σ_{j ∈ B t} exp (s t j − m̂ (t+1))·v t j` (`α = alpha m̂`), then for `t ≤ n`
`|âcc t − asum s v B (m̂ t) t| ≤ bound ε (alpha m̂) η t (asum s |v| B (m̂ t) t)`. -/
theorem acc_err {mh ah η : ℕ → ℝ} {ε : ℝ} {n : ℕ} (hε : 0 ≤ ε) (ha0 : ah 0 = 0)
    (hstep : ∀ t < n, |ah t - asum s v B (mh t) t| ≤
        bound ε (alpha mh) η t (asum s (fun u j => |v u j|) B (mh t) t) →
      |ah (t + 1) - (alpha mh t * ah t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * v t j)| ≤
        ε * (alpha mh t * |ah t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * |v t j|) + η t) :
    ∀ t ≤ n, |ah t - asum s v B (mh t) t| ≤
      bound ε (alpha mh) η t (asum s (fun u j => |v u j|) B (mh t) t) :=
  accum (X := fun t => asum s v B (mh t) t) (M := fun t => asum s (fun u j => |v u j|) B (mh t) t)
    (Y := fun t => ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * v t j)
    (Z := fun t => ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * |v t j|)
    hε (fun t _ => (alpha_pos mh t).le) (by simp [ha0, asum_zero]) (by simp [asum_zero])
    (fun t _ => asum_succ (mh t) (mh (t + 1)) t)
    (fun t _ => asum_succ (v := fun u j => |v u j|) (mh t) (mh (t + 1)) t)
    (fun t _ => (abs_sum_le_sum_abs _ _).trans (le_of_eq (sum_congr rfl fun j _ => by
      rw [abs_mul, abs_of_pos (Real.exp_pos _)]))) hstep

/-- **The ratio of two approximations.** If `l̂` is within `eL < L` of `L > 0`, `â` within `eA` of
`A`, and `|A| ≤ V·L`, then `|â/l̂ − A/L| ≤ (eA + V·eL)/(L − eL)`: the final `acc / l`. -/
theorem ratio_err {L A l a eL eA V : ℝ} (hL : 0 < L) (hl : |l - L| ≤ eL) (heL : eL < L)
    (ha : |a - A| ≤ eA) (hAV : |A| ≤ V * L) : |a / l - A / L| ≤ (eA + V * eL) / (L - eL) := by
  have hd : 0 < L - eL := by linarith
  have hll : L - eL ≤ l := by have := (abs_le.mp hl).1; linarith
  have hlpos : 0 < l := hd.trans_le hll
  have heL0 : 0 ≤ eL := (abs_nonneg _).trans hl
  have heA0 : 0 ≤ eA := (abs_nonneg _).trans ha
  have hV : 0 ≤ V :=
    le_of_mul_le_mul_right (by rw [zero_mul]; exact (abs_nonneg A).trans hAV) hL
  rw [div_sub_div _ _ hlpos.ne' hL.ne', abs_div, abs_of_pos (mul_pos hlpos hL),
    div_le_div_iff₀ (mul_pos hlpos hL) hd]
  have e : a * L - l * A = (a - A) * L + A * (L - l) := by ring
  have h1 : |a * L - l * A| ≤ eA * L + V * L * eL := by
    rw [e]
    calc |(a - A) * L + A * (L - l)| ≤ |(a - A) * L| + |A * (L - l)| := abs_add_le _ _
      _ = |a - A| * L + |A| * |l - L| := by rw [abs_mul, abs_mul, abs_of_pos hL, abs_sub_comm L]
      _ ≤ eA * L + V * L * eL :=
          add_le_add (mul_le_mul_of_nonneg_right ha hL.le)
            (mul_le_mul hAV hl (abs_nonneg _) (mul_nonneg hV hL.le))
  calc |a * L - l * A| * (L - eL) ≤ (eA * L + V * L * eL) * (L - eL) :=
        mul_le_mul_of_nonneg_right h1 hd.le
    _ = (eA + V * eL) * (L - eL) * L := by ring
    _ ≤ (eA + V * eL) * l * L :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hll (add_nonneg heA0 (mul_nonneg hV heL0)))
          hL.le
    _ = (eA + V * eL) * (l * L) := by ring

/-- **The ratio, when `l̂` is within half of `L`:** `|â/l̂ − A/L| ≤ 2·(eA + V·eL)/L`. -/
theorem ratio_err_half {L A l a eL eA V : ℝ} (hL : 0 < L) (hl : |l - L| ≤ eL) (heL : eL ≤ L / 2)
    (ha : |a - A| ≤ eA) (hAV : |A| ≤ V * L) : |a / l - A / L| ≤ 2 * (eA + V * eL) / L := by
  have h := ratio_err hL hl (by linarith) ha hAV
  have heL0 : 0 ≤ eL := (abs_nonneg _).trans hl
  have heA0 : 0 ≤ eA := (abs_nonneg _).trans ha
  have hV : 0 ≤ V :=
    le_of_mul_le_mul_right (by rw [zero_mul]; exact (abs_nonneg A).trans hAV) hL
  refine h.trans ?_
  rw [div_le_div_iff₀ (by linarith) hL]
  have := add_nonneg heA0 (mul_nonneg hV heL0)
  nlinarith

/-- **The output of the online softmax, with errors.** Under the trips of `l_err` (rate `εl`,
absolute errors `ηl ≥ 0`) and `acc_err` (rate `εa`, absolute errors `ηa`), with every `|v u j| ≤ V`
and room `eL < L` (`L = lsum s B (m̂ n) n`, `eL` its running bound), the computed ratio
`âcc n / l̂ n` is within `(eA + V·eL)/(L − eL)` of the softmax-weighted sum over the pairs
(trip, lane) seen, where `eA` is the accumulator's running bound. Use `err_final_mono` instead
for a running maximum. -/
theorem err_final {mh lh ah ηl ηa : ℕ → ℝ} {εl εa V : ℝ} {n : ℕ} (hεl : 0 ≤ εl) (hεa : 0 ≤ εa)
    (hηl : ∀ t < n, 0 ≤ ηl t) (hl0 : lh 0 = 0) (ha0 : ah 0 = 0)
    (hlstep : ∀ t < n, |lh t - lsum s B (mh t) t| ≤ bound εl (alpha mh) ηl t (lsum s B (mh t) t) →
      |lh (t + 1) - (alpha mh t * lh t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))| ≤
        εl * (alpha mh t * |lh t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1))) + ηl t)
    (hastep : ∀ t < n, |ah t - asum s v B (mh t) t| ≤
        bound εa (alpha mh) ηa t (asum s (fun u j => |v u j|) B (mh t) t) →
      |ah (t + 1) - (alpha mh t * ah t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * v t j)| ≤
        εa * (alpha mh t * |ah t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * |v t j|) + ηa t)
    (hV : ∀ u < n, ∀ j ∈ B u, |v u j| ≤ V)
    (hroom : bound εl (alpha mh) ηl n (lsum s B (mh n) n) < lsum s B (mh n) n) :
    |ah n / lh n - wavg ((range n).sigma B) (fun x => s x.1 x.2) (fun x => v x.1 x.2)| ≤
      (bound εa (alpha mh) ηa n (asum s (fun u j => |v u j|) B (mh n) n) +
          V * bound εl (alpha mh) ηl n (lsum s B (mh n) n)) /
        (lsum s B (mh n) n - bound εl (alpha mh) ηl n (lsum s B (mh n) n)) := by
  have hL : 0 < lsum s B (mh n) n :=
    lt_of_le_of_lt (bound_nonneg hεl (lsum_nonneg _ _) (fun u _ => (alpha_pos mh u).le) hηl) hroom
  rw [← asum_div_lsum (mh n) n]
  exact ratio_err hL (l_err hεl hl0 hlstep n le_rfl) hroom (acc_err hεa ha0 hastep n le_rfl)
    ((abs_asum_le _ _).trans (asum_abs_le hV))

/-- **The output of the online softmax, with a running maximum.** As `err_final`, when the
references never decrease after the first trip (`m̂ t ≤ m̂ (t+1)` for `0 < t`, so `α ≤ 1`),
`n·εl ≤ 1/2`, `n·εa ≤ 1/2`, the absolute errors are non-negative, and
`2n·εl·L + 2·Σ ηl ≤ L/2` (`L = lsum s B (m̂ n) n > 0`):
`|âcc n / l̂ n − wavg| ≤ 4n·(εa + εl)·V + 4·(Σ ηa + V·Σ ηl)/L`. With `m̂ n` at most a score seen,
`L ≥ 1` (`one_le_lsum`). This is the final step of an attention kernel's accuracy proof. -/
theorem err_final_mono {mh lh ah ηl ηa : ℕ → ℝ} {εl εa V : ℝ} {n : ℕ} (hεl : 0 ≤ εl)
    (hεa : 0 ≤ εa) (hηl : ∀ t < n, 0 ≤ ηl t) (hηa : ∀ t < n, 0 ≤ ηa t)
    (hmono : ∀ t, 0 < t → t < n → mh t ≤ mh (t + 1)) (hnl : n * εl ≤ 1 / 2) (hna : n * εa ≤ 1 / 2)
    (hl0 : lh 0 = 0) (ha0 : ah 0 = 0)
    (hlstep : ∀ t < n, |lh t - lsum s B (mh t) t| ≤ bound εl (alpha mh) ηl t (lsum s B (mh t) t) →
      |lh (t + 1) - (alpha mh t * lh t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))| ≤
        εl * (alpha mh t * |lh t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1))) + ηl t)
    (hastep : ∀ t < n, |ah t - asum s v B (mh t) t| ≤
        bound εa (alpha mh) ηa t (asum s (fun u j => |v u j|) B (mh t) t) →
      |ah (t + 1) - (alpha mh t * ah t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * v t j)| ≤
        εa * (alpha mh t * |ah t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * |v t j|) + ηa t)
    (hV : ∀ u < n, ∀ j ∈ B u, |v u j| ≤ V) (hL : 0 < lsum s B (mh n) n)
    (hroom : 2 * n * εl * lsum s B (mh n) n + 2 * ∑ t ∈ range n, ηl t ≤ lsum s B (mh n) n / 2) :
    |ah n / lh n - wavg ((range n).sigma B) (fun x => s x.1 x.2) (fun x => v x.1 x.2)| ≤
      4 * n * (εa + εl) * V +
        4 * (∑ t ∈ range n, ηa t + V * ∑ t ∈ range n, ηl t) / lsum s B (mh n) n := by
  set L := lsum s B (mh n) n with hLdef
  set Ma := asum s (fun u j => |v u j|) B (mh n) n with hMa
  have hα0 : ∀ u < n, 0 ≤ alpha mh u := fun u _ => (alpha_pos mh u).le
  have hα1 : ∀ u, 0 < u → u < n → alpha mh u ≤ 1 := fun u hu hun => alpha_le_one (hmono u hu hun)
  have hel := (l_err hεl hl0 hlstep n le_rfl).trans
    (bound_le hεl (by linarith) (lsum_nonneg _ _) hα0 hα1 hηl)
  have hMa0 : 0 ≤ Ma :=
    sum_nonneg fun _ _ => sum_nonneg fun _ _ => mul_nonneg (Real.exp_pos _).le (abs_nonneg _)
  have hMaV : Ma ≤ V * L := asum_abs_le hV
  have hea := (acc_err hεa ha0 hastep n le_rfl).trans
    (bound_le hεa (by linarith) hMa0 hα0 hα1 hηa)
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hea' : |ah n - asum s v B (mh n) n| ≤
      2 * n * εa * (V * L) + 2 * ∑ t ∈ range n, ηa t := by
    refine hea.trans ?_
    have := mul_le_mul_of_nonneg_left hMaV (mul_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) hn0) hεa)
    linarith
  rw [← asum_div_lsum (mh n) n]
  refine (ratio_err_half hL hel hroom hea' ((abs_asum_le _ _).trans (asum_abs_le hV))).trans
    (le_of_eq ?_)
  field_simp
  ring

end Computed

end Softmax

end TVA
