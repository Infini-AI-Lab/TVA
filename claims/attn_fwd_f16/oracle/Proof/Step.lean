import Proof.Tol
import Proof.Online

/-!
# One trip of the row sum and of the accumulator, from its parts

The step hypotheses of `online_step`, proved from what a trip computes: the rescaling factor
`α̂ = exp2(m_old − m_new)` (within `ρα·α + ηα` of the exact `α`), the weights (`p̂` in `float32`,
`p16` in `float16`, each within a relative and an absolute error of the exact `p`), the sum along a
tree, and the rounding of the accumulation (an `fma` for `l`, a `tl.dot` for the accumulator).
-/

namespace Claims.AttnFwd

open Finset

noncomputable section

/-- The per-trip rate of the row sum, for a rescaling factor within `ρα`. -/
def epsL (ρα : ℝ) : ℝ := ρα * (1 + 3 * u32) + 3 * u32
/-- The per-trip rate of the accumulator, for a rescaling factor within `ρα`. -/
def epsA (ρα : ℝ) : ℝ := ρα * ((1 + u32) * (1 + r64)) + (u32 + r64 * (1 + u32))

/-- `epsL` is nonnegative for nonnegative input. -/
theorem epsL_nonneg {ρα : ℝ} (h : 0 ≤ ρα) : 0 ≤ epsL ρα := by
  unfold epsL u32; positivity

/-- `epsA` is nonnegative for nonnegative input. -/
theorem epsA_nonneg {ρα : ℝ} (h : 0 ≤ ρα) : 0 ≤ epsA ρα := by
  unfold epsA u32 r64; positivity

/-- **One trip of the row sum.** `l' = α̂·l + Σ p̂`-ish (an `fma`, or a product then a sum) with
`α̂` within `ρα·α + ηα` of `α`, the lanes `p̂` within `ρp·p + ηp` of the exact weights `p ≥ 0`,
summed along a tree; then `l'` is within `epsL ρα·(α·|l| + Σ p) + rbarL·Σ p + 2^-100` of
`α·l + Σ p`. -/
theorem stepL {D : ℕ} {X α α' ρα ηα l0 lij l' : ℝ} {p p' : Fin 64 → ℝ}
    (hα0 : 0 ≤ α) (hρα : 0 ≤ ρα) (hηα0 : 0 ≤ ηα) (hηα : ηα ≤ 1 / 2 ^ 125)
    (hα' : |α' - α| ≤ ρα * α + ηα) (hl0 : |l0| ≤ 2 ^ 18)
    (_hρp0 : 0 ≤ rhoP D X) (_hρp1 : rhoP D X ≤ 1)
    (hp0 : ∀ j, 0 ≤ p j) (hp : ∀ j, |p' j - p j| ≤ rhoP D X * p j + 1 / 2 ^ 125)
    (hs : |lij - ∑ j, p' j| ≤ rhoS * ∑ j, |p' j| + 1 / 2 ^ 143)
    (hl : |l' - (α' * l0 + lij)| ≤ 3 * u32 * (|α' * l0| + |lij|) + 4 / 2 ^ 150) :
    |l' - (α * l0 + ∑ j, p j)| ≤
      epsL ρα * (α * |l0| + ∑ j, p j) + rbarL D X * ∑ j, p j + etaAL := by
  set rp := rhoP D X with hrp
  set P := ∑ j, p j with hP
  set S' := ∑ j, p' j with hS'
  set Sp := ∑ j, |p' j| with hSp
  have hP0 : 0 ≤ P := sum_nonneg fun j _ => hp0 j
  have hu0 : 0 ≤ u32 := by unfold u32; norm_num
  have hρs0 : 0 ≤ rhoS := by
    unfold rhoS
    have : (1:ℝ) ≤ (1 + u32) ^ 6 := one_le_pow₀ (by linarith)
    linarith
  -- the lanes together
  have hcard : ∑ _j : Fin 64, (1 / 2 ^ 125 : ℝ) = 64 / 2 ^ 125 := by simp; norm_num
  have h1 : |S' - P| ≤ rp * P + 64 / 2 ^ 125 := by
    have : S' - P = ∑ j, (p' j - p j) := by rw [hS', hP, sum_sub_distrib]
    rw [this]
    refine (abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ j, |p' j - p j| ≤ ∑ j, (rp * p j + 1 / 2 ^ 125) := sum_le_sum fun j _ => hp j
      _ = rp * P + 64 / 2 ^ 125 := by rw [sum_add_distrib, ← mul_sum, hcard]
  have h2 : Sp ≤ (1 + rp) * P + 64 / 2 ^ 125 := by
    calc Sp ≤ ∑ j, ((1 + rp) * p j + 1 / 2 ^ 125) := by
          refine sum_le_sum fun j _ => ?_
          have := abs_sub_abs_le_abs_sub (p' j) (p j)
          rw [abs_of_nonneg (hp0 j)] at this
          linarith [hp j]
      _ = (1 + rp) * P + 64 / 2 ^ 125 := by rw [sum_add_distrib, ← mul_sum, hcard]
  have hSS : |S'| ≤ Sp := abs_sum_le_sum_abs _ _
  have hlij : |lij| ≤ (1 + rhoS) * Sp + 1 / 2 ^ 143 := by
    have : |lij| ≤ |S'| + |lij - S'| := by
      have := abs_add_le S' (lij - S')
      rwa [add_sub_cancel] at this
    linarith [abs_nonneg S']
  have hlijP : |lij - P| ≤ rhoS * Sp + 1 / 2 ^ 143 + (rp * P + 64 / 2 ^ 125) := by
    have : lij - P = (lij - S') + (S' - P) := by ring
    rw [this]
    refine (abs_add_le _ _).trans ?_
    linarith
  have hαl : |α' * l0| ≤ (1 + ρα) * (α * |l0|) + ηα * |l0| := by
    have : α' * l0 = α * l0 + (α' - α) * l0 := by ring
    rw [this]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul, abs_of_nonneg hα0]
    have := mul_le_mul_of_nonneg_right hα' (abs_nonneg l0)
    linarith
  have hdl : |(α' - α) * l0| ≤ ρα * (α * |l0|) + ηα * |l0| := by
    rw [abs_mul]
    have := mul_le_mul_of_nonneg_right hα' (abs_nonneg l0)
    linarith
  have hsp : rhoS * Sp ≤ rhoS * ((1 + rp) * P + 64 / 2 ^ 125) := mul_le_mul_of_nonneg_left h2 hρs0
  have hlp : 3 * u32 * |lij| ≤ 3 * u32 * ((1 + rhoS) * ((1 + rp) * P + 64 / 2 ^ 125) + 1 / 2 ^ 143) := by
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    have := mul_le_mul_of_nonneg_left h2 (by linarith : (0:ℝ) ≤ 1 + rhoS)
    linarith
  have hla : 3 * u32 * |α' * l0| ≤ 3 * u32 * ((1 + ρα) * (α * |l0|) + ηα * |l0|) :=
    mul_le_mul_of_nonneg_left hαl (by positivity)
  have hηl : ηα * |l0| ≤ 1 / 2 ^ 107 := by
    calc ηα * |l0| ≤ (1 / 2 ^ 125) * 2 ^ 18 := mul_le_mul hηα hl0 (abs_nonneg _) (by positivity)
      _ = 1 / 2 ^ 107 := by norm_num
  have hsplit : l' - (α * l0 + P) = (l' - (α' * l0 + lij)) + (α' - α) * l0 + (lij - P) := by ring
  have hmain : |l' - (α * l0 + P)| ≤ |l' - (α' * l0 + lij)| + |(α' - α) * l0| + |lij - P| := by
    rw [hsplit]
    have h := abs_add_le (l' - (α' * l0 + lij) + (α' - α) * l0) (lij - P)
    have h' := abs_add_le (l' - (α' * l0 + lij)) ((α' - α) * l0)
    linarith
  have hep : 3 * u32 ≤ epsL ρα := by unfold epsL; nlinarith [mul_nonneg hρα hu0]
  have hR : rbarL D X = (rp + rhoS * (1 + rp)) * (1 + 3 * u32) := rfl
  have hPe : 3 * u32 * P ≤ epsL ρα * P := mul_le_mul_of_nonneg_right hep hP0
  have hu1 : u32 = 1 / 2 ^ 24 := rfl
  have hρs1 : rhoS ≤ 1 / 2 ^ 20 := by unfold rhoS; rw [hu1]; norm_num
  have habs : 3 * u32 * (ηα * |l0|) + ηα * |l0| + 3 * u32 * ((1 + rhoS) * (64 / 2 ^ 125) + 1 / 2 ^ 143)
      + rhoS * (64 / 2 ^ 125) + 1 / 2 ^ 143 + 64 / 2 ^ 125 + 4 / 2 ^ 150 ≤ etaAL := by
    have hη0 : 0 ≤ ηα * |l0| := by positivity
    have e1 : 3 * u32 * (ηα * |l0|) ≤ 3 * (1 / 2 ^ 24) * (1 / 2 ^ 107) := by
      rw [hu1]; exact mul_le_mul_of_nonneg_left hηl (by norm_num)
    have e2 : 3 * u32 * ((1 + rhoS) * (64 / 2 ^ 125) + 1 / 2 ^ 143) ≤
        3 * (1 / 2 ^ 24) * ((1 + 1 / 2 ^ 20) * (64 / 2 ^ 125) + 1 / 2 ^ 143) := by
      rw [hu1]
      refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
      have := mul_le_mul_of_nonneg_right (by linarith : 1 + rhoS ≤ 1 + 1 / 2 ^ 20)
        (by norm_num : (0:ℝ) ≤ 64 / 2 ^ 125)
      linarith
    have e3 : rhoS * (64 / 2 ^ 125) ≤ 1 / 2 ^ 20 * (64 / 2 ^ 125) :=
      mul_le_mul_of_nonneg_right hρs1 (by norm_num)
    unfold etaAL
    norm_num at e1 e2 e3 hηl ⊢
    linarith
  rw [hR]
  unfold epsL
  have hnn : 0 ≤ ρα * (3 * u32) * P := by positivity
  have hnn2 : 0 ≤ ρα * P := by positivity
  linarith [hmain, hl, hdl, hlijP, hsp, hlp, hla, hnn, hnn2, habs]


/-- **One trip of the accumulator** (one output column). `x1 = dot(p16, v, a1)` with `a1` the
rescaled accumulator `α̂·x0` (rounded): `α̂` within `ρα·α + ηα` of `α`, the `float16` weights `p16`
within `rho16·p + eta16` of the exact `p ≥ 0` (and `0`, with lifted magnitude `0`, on the masked
lanes), the dot within `r64` of the lifted magnitudes. Then `x1` is within
`epsA ρα·(α·|x0| + Σ p·|v|) + rbarA·Σ p·|v| + kapA·Σ p + etavA·Σ_{ok}|v| + etaAA` of
`α·x0 + Σ p·v`. `lp`, `lv` are the lifted magnitudes of `p16`, `v`. -/
theorem stepA {D : ℕ} {X α α' ρα ηα x0 a1 x1 : ℝ} {p p16 v lp lv : Fin 64 → ℝ} {ok : Fin 64 → Bool}
    (hα0 : 0 ≤ α) (hρα : 0 ≤ ρα) (_hηα0 : 0 ≤ ηα) (hηα : ηα ≤ 1 / 2 ^ 125)
    (hα' : |α' - α| ≤ ρα * α + ηα) (hx0 : |x0| ≤ 2 ^ 34)
    (ha1 : |a1 - α' * x0| ≤ u32 * |α' * x0| + 1 / 2 ^ 150)
    (_hρ0 : 0 ≤ rho16 D X)
    (hp0 : ∀ j, 0 ≤ p j) (hp16 : ∀ j, |p16 j - p j| ≤ rho16 D X * p j + eta16)
    (hmask : ∀ j, ok j = false → p j = 0 ∧ p16 j = 0 ∧ lp j = 0)
    (hlp : ∀ j, |p16 j| ≤ lp j ∧ lp j ≤ |p16 j| + 1 / 2 ^ 14)
    (hlv : ∀ j, |v j| ≤ lv j ∧ lv j ≤ |v j| + 1 / 2 ^ 14)
    (hdot : |x1 - (a1 + ∑ j, p16 j * v j)| ≤ r64 * (|a1| + ∑ j, lp j * lv j) + 66 / 2 ^ 126) :
    |x1 - (α * x0 + ∑ j, p j * v j)| ≤
      epsA ρα * (α * |x0| + ∑ j, p j * |v j|) + rbarA D X * ∑ j, p j * |v j| +
        kapA D X * ∑ j, p j + etavA * ∑ j, (if ok j = true then |v j| else 0) + etaAA := by
  set rr := rho16 D X with hrr
  set Z := ∑ j, p j * |v j| with hZ
  set R := ∑ j, p j with hR
  set W := ∑ j, (if ok j = true then |v j| else 0) with hW
  set Y := ∑ j, p j * v j with hY
  set Sq := ∑ j, p16 j * v j with hSq
  have hu0 : 0 ≤ u32 := by unfold u32; norm_num
  have hr0 : 0 ≤ r64 := by unfold r64; norm_num
  have he0 : 0 ≤ eta16 := by unfold eta16; norm_num
  have hZ0 : 0 ≤ Z := sum_nonneg fun j _ => mul_nonneg (hp0 j) (abs_nonneg _)
  have hR0 : 0 ≤ R := sum_nonneg fun j _ => hp0 j
  have hW0 : 0 ≤ W := sum_nonneg fun j _ => by split_ifs <;> positivity
  have hcard : ∀ c : ℝ, ∑ _j : Fin 64, c = 64 * c := by intro c; simp
  -- lane by lane
  have L1 : ∀ j, |p16 j * v j - p j * v j| ≤
      rr * (p j * |v j|) + eta16 * (if ok j = true then |v j| else 0) := by
    intro j
    by_cases h : ok j = true
    · rw [ite_eq_left h, ← sub_mul, abs_mul]
      have := mul_le_mul_of_nonneg_right (hp16 j) (abs_nonneg (v j))
      linarith
    · have h' : ok j = false := by simpa using h
      obtain ⟨h1, h2, -⟩ := hmask j h'
      rw [ite_eq_right h, h1, h2]; simp
  have L2 : ∀ j, |p16 j| * |v j| ≤
      (1 + rr) * (p j * |v j|) + eta16 * (if ok j = true then |v j| else 0) := by
    intro j
    by_cases h : ok j = true
    · rw [ite_eq_left h]
      have h1 : |p16 j| ≤ (1 + rr) * p j + eta16 := by
        have := abs_sub_abs_le_abs_sub (p16 j) (p j)
        rw [abs_of_nonneg (hp0 j)] at this
        linarith [hp16 j]
      have := mul_le_mul_of_nonneg_right h1 (abs_nonneg (v j))
      linarith
    · have h' : ok j = false := by simpa using h
      obtain ⟨h1, h2, -⟩ := hmask j h'
      rw [ite_eq_right h, h1, h2]; simp
  have L3 : ∀ j, lp j * lv j ≤ |p16 j| * |v j| +
      1 / 2 ^ 14 * ((if ok j = true then (1:ℝ) else 0) * |p16 j|) +
      1 / 2 ^ 14 * (if ok j = true then |v j| else 0) +
      1 / 2 ^ 28 * (if ok j = true then (1:ℝ) else 0) := by
    intro j
    by_cases h : ok j = true
    · rw [ite_eq_left h, ite_eq_left h]
      obtain ⟨a1, a2⟩ := hlp j
      obtain ⟨b1, b2⟩ := hlv j
      have := mul_le_mul a2 b2 (by linarith [abs_nonneg (v j)]) (by positivity)
      nlinarith
    · have h' : ok j = false := by simpa using h
      obtain ⟨h1, h2, h3⟩ := hmask j h'
      rw [ite_eq_right h, ite_eq_right h, h3, h2]; simp
  have L4 : ∀ j, (if ok j = true then (1:ℝ) else 0) * |p16 j| ≤ (1 + rr) * p j + eta16 := by
    intro j
    by_cases h : ok j = true
    · rw [ite_eq_left h, one_mul]
      have := abs_sub_abs_le_abs_sub (p16 j) (p j)
      rw [abs_of_nonneg (hp0 j)] at this
      linarith [hp16 j]
    · have h' : ok j = false := by simpa using h
      obtain ⟨h1, h2, -⟩ := hmask j h'
      rw [ite_eq_right h, h1]; simp; linarith [mul_nonneg hr0 he0]
  -- the sums
  have S1 : |Sq - Y| ≤ rr * Z + eta16 * W := by
    have : Sq - Y = ∑ j, (p16 j * v j - p j * v j) := by rw [hSq, hY, sum_sub_distrib]
    rw [this]
    refine (abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ j, |p16 j * v j - p j * v j|
        ≤ ∑ j, (rr * (p j * |v j|) + eta16 * (if ok j = true then |v j| else 0)) :=
          sum_le_sum fun j _ => L1 j
      _ = rr * Z + eta16 * W := by rw [sum_add_distrib, ← mul_sum, ← mul_sum]
  have S2 : ∑ j, |p16 j| * |v j| ≤ (1 + rr) * Z + eta16 * W := by
    calc ∑ j, |p16 j| * |v j|
        ≤ ∑ j, ((1 + rr) * (p j * |v j|) + eta16 * (if ok j = true then |v j| else 0)) :=
          sum_le_sum fun j _ => L2 j
      _ = (1 + rr) * Z + eta16 * W := by rw [sum_add_distrib, ← mul_sum, ← mul_sum]
  have S4 : ∑ j, (if ok j = true then (1:ℝ) else 0) * |p16 j| ≤ (1 + rr) * R + 64 * eta16 := by
    calc ∑ j, (if ok j = true then (1:ℝ) else 0) * |p16 j| ≤ ∑ j, ((1 + rr) * p j + eta16) :=
          sum_le_sum fun j _ => L4 j
      _ = (1 + rr) * R + 64 * eta16 := by rw [sum_add_distrib, ← mul_sum, hcard]
  have Scard : ∑ j, (if ok j = true then (1:ℝ) else 0) ≤ 64 := by
    calc ∑ j, (if ok j = true then (1:ℝ) else 0) ≤ ∑ _j : Fin 64, (1:ℝ) :=
          sum_le_sum fun j _ => by split_ifs <;> norm_num
      _ = 64 := by rw [hcard]; norm_num
  have S3 : ∑ j, lp j * lv j ≤ ((1 + rr) * Z + eta16 * W) +
      1 / 2 ^ 14 * ((1 + rr) * R + 64 * eta16) + 1 / 2 ^ 14 * W + 1 / 2 ^ 28 * 64 := by
    calc ∑ j, lp j * lv j ≤ ∑ j, (|p16 j| * |v j| +
        1 / 2 ^ 14 * ((if ok j = true then (1:ℝ) else 0) * |p16 j|) +
        1 / 2 ^ 14 * (if ok j = true then |v j| else 0) +
        1 / 2 ^ 28 * (if ok j = true then (1:ℝ) else 0)) := sum_le_sum fun j _ => L3 j
      _ = ∑ j, |p16 j| * |v j| + 1 / 2 ^ 14 * ∑ j, (if ok j = true then (1:ℝ) else 0) * |p16 j| +
          1 / 2 ^ 14 * W + 1 / 2 ^ 28 * ∑ j, (if ok j = true then (1:ℝ) else 0) := by
          simp only [sum_add_distrib, ← mul_sum]; rfl
      _ ≤ _ := by
          have := mul_le_mul_of_nonneg_left S4 (by norm_num : (0:ℝ) ≤ 1 / 2 ^ 14)
          have := mul_le_mul_of_nonneg_left Scard (by norm_num : (0:ℝ) ≤ 1 / 2 ^ 28)
          linarith
  -- the rescaled accumulator
  have hαx : |α' * x0| ≤ (1 + ρα) * (α * |x0|) + ηα * |x0| := by
    have : α' * x0 = α * x0 + (α' - α) * x0 := by ring
    rw [this]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul, abs_of_nonneg hα0]
    have := mul_le_mul_of_nonneg_right hα' (abs_nonneg x0)
    linarith
  have hdx : |(α' - α) * x0| ≤ ρα * (α * |x0|) + ηα * |x0| := by
    rw [abs_mul]
    have := mul_le_mul_of_nonneg_right hα' (abs_nonneg x0)
    linarith
  have hηx : ηα * |x0| ≤ 1 / 2 ^ 91 := by
    calc ηα * |x0| ≤ (1 / 2 ^ 125) * 2 ^ 34 := mul_le_mul hηα hx0 (abs_nonneg _) (by positivity)
      _ = 1 / 2 ^ 91 := by norm_num
  have hax : |a1 - α * x0| ≤ (ρα + u32 * (1 + ρα)) * (α * |x0|) + (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150 := by
    have e : a1 - α * x0 = (a1 - α' * x0) + (α' - α) * x0 := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    have := mul_le_mul_of_nonneg_left hαx hu0
    nlinarith
  have ha1m : |a1| ≤ (1 + (ρα + u32 * (1 + ρα))) * (α * |x0|) + (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150 := by
    have h1 := abs_add_le (α * x0) (a1 - α * x0)
    rw [add_sub_cancel, abs_mul, abs_of_nonneg hα0] at h1
    linarith
  -- assembly
  have hmain : |x1 - (α * x0 + Y)| ≤ |x1 - (a1 + Sq)| + |a1 - α * x0| + |Sq - Y| := by
    have e : x1 - (α * x0 + Y) = (x1 - (a1 + Sq)) + (a1 - α * x0) + (Sq - Y) := by ring
    rw [e]
    have h := abs_add_le (x1 - (a1 + Sq) + (a1 - α * x0)) (Sq - Y)
    have h' := abs_add_le (x1 - (a1 + Sq)) (a1 - α * x0)
    linarith
  have hd1 : r64 * (|a1| + ∑ j, lp j * lv j) ≤
      r64 * ((1 + (ρα + u32 * (1 + ρα))) * (α * |x0|) + (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150 +
        (((1 + rr) * Z + eta16 * W) + 1 / 2 ^ 14 * ((1 + rr) * R + 64 * eta16) + 1 / 2 ^ 14 * W +
          1 / 2 ^ 28 * 64)) :=
    mul_le_mul_of_nonneg_left (by linarith) hr0
  have hep : r64 ≤ epsA ρα := by
    unfold epsA
    have : 0 ≤ ρα * ((1 + u32) * (1 + r64)) := by positivity
    nlinarith [mul_nonneg hr0 hu0]
  have hZe : r64 * Z ≤ epsA ρα * Z := mul_le_mul_of_nonneg_right hep hZ0
  have hu1 : u32 = 1 / 2 ^ 24 := rfl
  have hr1 : r64 = 66 / 2 ^ 22 := rfl
  have he1 : eta16 = 1 / 2 ^ 25 + 1 / 2 ^ 124 := rfl
  have habs : r64 * ((1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150 + 1 / 2 ^ 14 * (64 * eta16) +
      1 / 2 ^ 28 * 64) + 66 / 2 ^ 126 + (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150 ≤ etaAA := by
    have h1 : (1 + u32) * (ηα * |x0|) ≤ (1 + u32) * (1 / 2 ^ 91) :=
      mul_le_mul_of_nonneg_left hηx (by linarith)
    have h2 : r64 * ((1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150 + 1 / 2 ^ 14 * (64 * eta16) +
        1 / 2 ^ 28 * 64) ≤ r64 * ((1 + u32) * (1 / 2 ^ 91) + 1 / 2 ^ 150 + 1 / 2 ^ 14 * (64 * eta16) +
        1 / 2 ^ 28 * 64) := mul_le_mul_of_nonneg_left (by linarith) hr0
    unfold etaAA
    rw [hu1, hr1, he1] at *
    norm_num at h1 h2 ⊢
    linarith
  have T1 : |x1 - (α * x0 + Y)| ≤ r64 * (|a1| + ∑ j, lp j * lv j) + 66 / 2 ^ 126 +
      ((ρα + u32 * (1 + ρα)) * (α * |x0|) + (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150) +
      (rr * Z + eta16 * W) := by linarith [hmain, hdot, hax, S1]
  have key : r64 * ((1 + (ρα + u32 * (1 + ρα))) * (α * |x0|) + (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150 +
        (((1 + rr) * Z + eta16 * W) + 1 / 2 ^ 14 * ((1 + rr) * R + 64 * eta16) + 1 / 2 ^ 14 * W +
          1 / 2 ^ 28 * 64)) + 66 / 2 ^ 126 +
      ((ρα + u32 * (1 + ρα)) * (α * |x0|) + (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150) +
      (rr * Z + eta16 * W) ≤
      epsA ρα * (α * |x0| + Z) + rbarA D X * Z + kapA D X * R + etavA * W + etaAA := by
    unfold epsA rbarA kapA etavA
    rw [← hrr]
    have e : (epsA ρα * (α * |x0| + Z) + rbarA D X * Z + kapA D X * R + etavA * W + etaAA) -
        (r64 * ((1 + (ρα + u32 * (1 + ρα))) * (α * |x0|) + (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150 +
        (((1 + rr) * Z + eta16 * W) + 1 / 2 ^ 14 * ((1 + rr) * R + 64 * eta16) + 1 / 2 ^ 14 * W +
          1 / 2 ^ 28 * 64)) + 66 / 2 ^ 126 +
        ((ρα + u32 * (1 + ρα)) * (α * |x0|) + (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150) +
        (rr * Z + eta16 * W)) =
        (epsA ρα * Z - r64 * Z) + (etaAA - (r64 * ((1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150 +
          1 / 2 ^ 14 * (64 * eta16) + 1 / 2 ^ 28 * 64) + 66 / 2 ^ 126 +
          (1 + u32) * (ηα * |x0|) + 1 / 2 ^ 150)) := by
      unfold epsA rbarA kapA etavA; rw [← hrr]; ring
    have := sub_nonneg.mpr hZe
    have := sub_nonneg.mpr habs
    unfold epsA rbarA kapA etavA at *
    rw [← hrr] at *
    linarith
  linarith [T1, hd1, key]


end

end Claims.AttnFwd
