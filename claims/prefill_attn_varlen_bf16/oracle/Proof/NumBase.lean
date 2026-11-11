import Mathlib
import TVA.Proof

section
/-!
# The real-number analysis of the online softmax: definitions

The kernel's loop keeps, for each query row, the running maximum `m`, the running sum `l` and the
(normalised) accumulator. This file fixes the exact sums those track (`Lam`, `Asum`, `Aabs`, over
the keys of the first `n` blocks), the shape of the running error bound (`Close`) and the rates
the float operations contribute (the constants of `Num`). Everything is over `ℝ`.
-/

namespace Claims.PrefillAttnVarlen

namespace Num

open Finset

/-- The keys of the first `n` blocks a row attends: `j < 128·n` and `j ≤ L`. -/
def keysN (L n : ℕ) : Finset ℕ := Finset.range (min (128 * n) (L + 1))

/-- The lanes of block `t` that hold an attended key: `c < 128` and `128·t + c ≤ L`. -/
def blk (L t : ℕ) : Finset ℕ := (Finset.range 128).filter fun c => 128 * t + c ≤ L

/-- `Σ_j exp (sh j − μ)` over the keys `K`: the exact running sum at reference `μ`. -/
noncomputable def Lam (sh : ℕ → ℝ) (K : Finset ℕ) (μ : ℝ) : ℝ := ∑ j ∈ K, Real.exp (sh j - μ)

/-- `Σ_j exp (sh j − μ)·v j`: the exact accumulator at reference `μ`. -/
noncomputable def Asum (sh vv : ℕ → ℝ) (K : Finset ℕ) (μ : ℝ) : ℝ :=
  ∑ j ∈ K, Real.exp (sh j - μ) * vv j

/-- `Σ_j exp (sh j − μ)·|v j|`: the magnitude of the accumulator. -/
noncomputable def Aabs (sh vv : ℕ → ℝ) (K : Finset ℕ) (μ : ℝ) : ℝ :=
  ∑ j ∈ K, Real.exp (sh j - μ) * |vv j|

/-- **The running bound**: `x` is within `(e^ω·(1 + B) − 1)·M + e^ω·n·η` of the exact `X`, whose
magnitude is `M`. -/
def Close (x X M ω B η : ℝ) (n : ℕ) : Prop :=
  |x - X| ≤ (Real.exp ω * (1 + B) - 1) * M + Real.exp ω * (n * η)

/-! ## The rates of the float operations (each is proved from a device fact) -/

/-- The relative error of `tl.exp` on an argument of size at most `2·Y + 1`, with the subtraction
that feeds it: `2^-21·(1 + Y)`. -/
noncomputable def rE (Y : ℝ) : ℝ := 2 ^ (-21 : ℤ) * (1 + Y)
/-- The relative error of `tl.exp(m_old − m_new)`, argument of size `Δ`: `2^-21·(1 + Δ)`. -/
noncomputable def rA (Δ : ℝ) : ℝ := 2 ^ (-21 : ℤ) * (1 + Δ)
/-- The absolute error of `tl.exp`, `2^-140`. -/
noncomputable def ηE : ℝ := 2 ^ (-140 : ℤ)
/-- The relative error of the row sum `tl.sum(p, 1)` over 128 lanes: `7` additions, `2^-21`. -/
noncomputable def rS : ℝ := 2 ^ (-21 : ℤ)
/-- Its absolute error. -/
noncomputable def ηS : ℝ := 2 ^ (-140 : ℤ)
/-- The relative error of `alpha * l_i + beta * l_ij` (two products and a fusable sum): `5u`. -/
noncomputable def rO : ℝ := 5 * 2 ^ (-24 : ℤ)
/-- Its absolute error. -/
noncomputable def ηO : ℝ := 2 ^ (-140 : ℤ)
/-- The relative error of one `tl.dot` trip, `acc += tl.dot(p, v)`, `K = 128`: `136·2^-22`. -/
noncomputable def rD : ℝ := 136 * 2 ^ (-22 : ℤ)
/-- The dot trip's absolute error: a constant... -/
noncomputable def adc0 : ℝ := 2 ^ (-118 : ℤ)
/-- ...and a multiple of the magnitudes of the values it meets. -/
noncomputable def adc1 : ℝ := 2 ^ (-140 : ℤ)

/-! ## The running bounds of the sum `l` and the accumulator

`omL`, `omA` are the exponents of the growth `Π (1 + rate)` along the trips: a constant per trip
and a multiple of the running maximum, which rises by at most `2·Y` over the whole loop. -/

/-- `l`'s per-trip rate. -/
noncomputable def cL0 : ℝ := 2 ^ (-19 : ℤ)
/-- `l`'s rate per unit of rise of the running maximum. -/
noncomputable def cL1 : ℝ := 2 ^ (-20 : ℤ)
/-- `l`'s one-time rate (the block's weights and sum). -/
noncomputable def bL (Y : ℝ) : ℝ := 2 ^ (-17 : ℤ) + 2 ^ (-19 : ℤ) * Y
/-- `l`'s absolute error per trip. -/
noncomputable def ηL : ℝ := 2 ^ (-100 : ℤ)
/-- The accumulator's per-trip rate (the dot's `136·2^-22` plus the rescaling's). -/
noncomputable def cA0 : ℝ := 160 * 2 ^ (-22 : ℤ)
/-- The accumulator's rate per unit of rise of the running maximum. -/
noncomputable def cA1 : ℝ := 2 ^ (-20 : ℤ)
/-- The accumulator's one-time rate: `p.to(bfloat16)`'s `2^-8`, the dot, the weights. -/
noncomputable def bA (Y : ℝ) : ℝ := 2 ^ (-8 : ℤ) + 2 ^ (-12 : ℤ) + 2 ^ (-19 : ℤ) * Y
/-- The accumulator's absolute error per trip, times `1 + V` (`V` the sum of `|v|`). -/
noncomputable def ηA : ℝ := 2 ^ (-96 : ℤ)

/-- The exponent of `l`'s growth after `t` trips with running maximum `m`. -/
noncomputable def omL (Y : ℝ) (t : ℕ) (m : ℝ) : ℝ := cL0 * t + cL1 * max (m + Y) 0
/-- The exponent of the accumulator's growth. -/
noncomputable def omA (Y : ℝ) (t : ℕ) (m : ℝ) : ℝ := cA0 * t + cA1 * max (m + Y) 0

/-- The largest `l`-growth over a loop of `T` trips: the running maximum rises by at most `2·Y`. -/
noncomputable def GLmax (Y T : ℝ) : ℝ := Real.exp (cL0 * T + cL1 * (2 * Y)) * (1 + bL Y) - 1
/-- The largest accumulator growth over a loop of `T` trips. -/
noncomputable def GAmax (Y T : ℝ) : ℝ := Real.exp (cA0 * T + cA1 * (2 * Y)) * (1 + bA Y) - 1

end Num

end Claims.PrefillAttnVarlen

end

section
/-!
# One trip of a rescaled accumulation, against the running bound

`prop_step`: if a state `x` is within the running bound `Close` of an exact state `X`, and one trip
`x' ≈ α·x + Y` has rates `ra` (on the rescaled state) and `rb` (on the new block) and an absolute
error `η`, then `x'` is within the running bound, at the larger exponent `ω'`, of `X' = α·X + Y`.
It is the induction step for `l` and for the accumulator.
-/

namespace Claims.PrefillAttnVarlen

namespace Num

/-- **The running bound propagates**. The exact state follows `X' = α·X + Y` with magnitudes
`M' = α·M + Z` (`|X| ≤ M`, `|Y| ≤ Z`, `0 ≤ α ≤ 1`); the computed `x'` is within
`ra·α·|x| + rb·Z + η` of `α·x + Y`; the exponent grows by at least `ra`
(or `α = 0`: the first trip, whose state is not rescaled). -/
theorem prop_step {x x' X X' M M' Y Z α ra rb η B ω ω' : ℝ} {n : ℕ}
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (hra : 0 ≤ ra) (hrb : rb ≤ B) (hrb0 : 0 ≤ rb) (hη : 0 ≤ η)
    (hX' : X' = α * X + Y) (hM' : M' = α * M + Z) (hXM : |X| ≤ M) (hYZ : |Y| ≤ Z)
    (hω : 0 ≤ ω' ∧ (α = 0 ∨ (0 ≤ ω ∧ ω + ra ≤ ω')))
    (hstep : |x' - (α * x + Y)| ≤ ra * (α * |x|) + rb * Z + η)
    (hinv : α = 0 ∨ Close x X M ω B η n) :
    Close x' X' M' ω' B η (n + 1) := by
  unfold Close at *
  have hZ : 0 ≤ Z := (abs_nonneg _).trans hYZ
  have hM : 0 ≤ M := (abs_nonneg _).trans hXM
  have hB : 0 ≤ B := hrb0.trans hrb
  have hA1 : 1 ≤ Real.exp ω' := Real.one_le_exp hω.1
  have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  rw [hM', hX']
  push_cast
  by_cases h0 : α = 0
  · -- the first trip: no rescaled state
    subst h0
    have : |x' - Y| ≤ rb * Z + η := by simpa using hstep
    have h2 : rb * Z ≤ (Real.exp ω' * (1 + B) - 1) * Z :=
      mul_le_mul_of_nonneg_right (by nlinarith) hZ
    have h3 : η ≤ Real.exp ω' * ((n + 1) * η) := by
      have : 0 ≤ (n : ℝ) * η := mul_nonneg hn hη
      nlinarith
    simp only [zero_mul, zero_add]
    nlinarith
  · obtain ⟨hω0, hωω⟩ := hω.2.resolve_left h0
    have hinv' := hinv.resolve_left h0
    have hA : 1 ≤ Real.exp ω := Real.one_le_exp hω0
    have hAA : (1 + ra) * Real.exp ω ≤ Real.exp ω' := by
      calc (1 + ra) * Real.exp ω ≤ Real.exp ra * Real.exp ω :=
            mul_le_mul_of_nonneg_right (by linarith [Real.add_one_le_exp ra]) (by positivity)
        _ = Real.exp (ra + ω) := (Real.exp_add _ _).symm
        _ ≤ Real.exp ω' := Real.exp_le_exp.mpr (by linarith)
    set E := |x - X| with hE
    have hx : |x| ≤ M + E := by
      have := abs_sub_abs_le_abs_sub x X; linarith
    -- the error of the new state
    have e1 : |x' - (α * X + Y)| ≤ ra * (α * (M + E)) + rb * Z + η + α * E := by
      have e : x' - (α * X + Y) = (x' - (α * x + Y)) + α * (x - X) := by ring
      rw [e]
      refine (abs_add_le _ _).trans ?_
      rw [abs_mul, abs_of_nonneg hα0]
      have := mul_le_mul_of_nonneg_left hx (mul_nonneg hra hα0)
      nlinarith
    have hP : 0 ≤ Real.exp ω * (1 + B) - 1 := by nlinarith
    have e2 : (1 + ra) * α * E ≤ (1 + ra) * α * ((Real.exp ω * (1 + B) - 1) * M +
        Real.exp ω * (n * η)) :=
      mul_le_mul_of_nonneg_left hinv' (mul_nonneg (by linarith) hα0)
    have hBnn : 0 ≤ 1 + B := by linarith
    -- coefficient of `α * M`
    have c1 : (1 + ra) * (Real.exp ω * (1 + B) - 1) + ra ≤ Real.exp ω' * (1 + B) - 1 := by
      have := mul_le_mul_of_nonneg_right hAA hBnn
      nlinarith
    have c2 : rb ≤ Real.exp ω' * (1 + B) - 1 := by nlinarith
    have c3 : (1 + ra) * α * (Real.exp ω * (n * η)) + η ≤ Real.exp ω' * ((n + 1) * η) := by
      have h1 : (1 + ra) * Real.exp ω * (n * η) ≤ Real.exp ω' * (n * η) :=
        mul_le_mul_of_nonneg_right hAA (mul_nonneg hn hη)
      have h2 : α * ((1 + ra) * Real.exp ω * (n * η)) ≤ (1 + ra) * Real.exp ω * (n * η) :=
        mul_le_of_le_one_left (mul_nonneg (mul_nonneg (by linarith) (by positivity)) (mul_nonneg hn hη)) hα1
      nlinarith [mul_nonneg hn hη]
    have c4 : α * M * ((1 + ra) * (Real.exp ω * (1 + B) - 1) + ra) ≤
        α * M * (Real.exp ω' * (1 + B) - 1) :=
      mul_le_mul_of_nonneg_left c1 (mul_nonneg hα0 hM)
    have c5 : rb * Z ≤ Z * (Real.exp ω' * (1 + B) - 1) := by nlinarith
    have e3 : ra * (α * (M + E)) + rb * Z + η + α * E =
        α * M * ra + (1 + ra) * α * E + rb * Z + η := by ring
    have e4 : (1 + ra) * α * ((Real.exp ω * (1 + B) - 1) * M + Real.exp ω * (n * η)) =
        α * M * ((1 + ra) * (Real.exp ω * (1 + B) - 1)) +
          (1 + ra) * α * (Real.exp ω * (n * η)) := by ring
    have e5 : (Real.exp ω' * (1 + B) - 1) * (α * M + Z) =
        α * M * (Real.exp ω' * (1 + B) - 1) + Z * (Real.exp ω' * (1 + B) - 1) := by ring
    rw [e5]
    have e6 : α * M * ((1 + ra) * (Real.exp ω * (1 + B) - 1) + ra) =
        α * M * ((1 + ra) * (Real.exp ω * (1 + B) - 1)) + α * M * ra := by ring
    linarith [e1, e2, e3, e4, e6, c3, c4, c5]

end Num

end Claims.PrefillAttnVarlen

end

section
/-!
# The loop invariant over `ℝ`: definitions and the splitting of the key sets

`RowInv` is what the loop keeps true, for one query row, of its running maximum `m` and sum `l`:
the computed scores `sh` are within `δ` of the exact ones, `m` is their maximum so far, and `l` is
within the running bound (`Close`) of the exact sum at reference `m`. For one column, the
unnormalised accumulator `l·a` is within the running bound of the exact `Asum`.

The keys of the first `t + 1` blocks are those of the first `t` plus the lanes `blk L t` of block
`t`; the exact sums split the same way (`Lam_succ`, `Asum_succ`, `Aabs_succ`).
-/

namespace Claims.PrefillAttnVarlen

namespace Num

open Finset

/-- **The row's invariant** after `t` trips: scores `sh` for the attended keys of the first `t`
blocks (within `δ` of the exact `sc`), the running maximum `m` of them, and the sum `l`. -/
structure RowInv (L t : ℕ) (sc sh : ℕ → ℝ) (δ Y m l : ℝ) : Prop where
  close_sh : ∀ j ∈ keysN L t, |sh j - sc j| ≤ δ
  attain : 0 < t → ∃ j ∈ keysN L t, sh j = m
  le_m : ∀ j ∈ keysN L t, sh j ≤ m
  l_close : Close l (Lam sh (keysN L t) m) (Lam sh (keysN L t) m) (omL Y t m) (bL Y) ηL t

/-- The scores after `t + 1` trips: the old ones for the first `t` blocks, the lane scores `qs`
for block `t`. -/
def shNext (t : ℕ) (sh qs : ℕ → ℝ) : ℕ → ℝ := fun j => if j < 128 * t then sh j else qs (j - 128 * t)

/-- No keys are visible after zero trips. -/
theorem keysN_zero (L : ℕ) : keysN L 0 = ∅ := by simp [keysN]

/-- The keys of the first `t` blocks are `[0, 128·t)` when block `t` has a key. -/
theorem keysN_eq_range {L t : ℕ} (h : 128 * t ≤ L) : keysN L t = range (128 * t) := by
  unfold keysN
  rw [min_eq_left (by omega)]

/-- A sum over the keys of the first `t + 1` blocks is the sum over the first `t` plus the
lanes of block `t`. -/
theorem sum_keysN_succ {L t : ℕ} (h : 128 * t ≤ L) (f : ℕ → ℝ) :
    ∑ j ∈ keysN L (t + 1), f j = ∑ j ∈ keysN L t, f j + ∑ c ∈ blk L t, f (128 * t + c) := by
  rw [keysN_eq_range h]
  have h1 : keysN L (t + 1) = range (128 * t) ∪ (blk L t).image (fun c => 128 * t + c) := by
    ext j
    simp only [keysN, blk, mem_range, mem_union, mem_image, mem_filter]
    constructor
    · intro hj
      by_cases hjt : j < 128 * t
      · exact Or.inl hjt
      · exact Or.inr ⟨j - 128 * t, ⟨by omega, by omega⟩, by omega⟩
    · rintro (hj | ⟨c, ⟨hc, hcL⟩, rfl⟩)
      · omega
      · omega
  have h2 : Disjoint (range (128 * t)) ((blk L t).image (fun c => 128 * t + c)) := by
    rw [Finset.disjoint_left]
    intro j hj hj'
    simp only [mem_range, mem_image] at hj hj'
    obtain ⟨c, _, rfl⟩ := hj'
    omega
  rw [h1, sum_union h2, sum_image]
  intro x _ y _ hxy
  simpa using hxy

/-- The new scores agree with the old on the first `t` blocks. -/
theorem shNext_old {t j : ℕ} {sh qs : ℕ → ℝ} (hj : j < 128 * t) : shNext t sh qs j = sh j := by
  simp [shNext, hj]

/-- ...and are the lane scores on block `t`. -/
theorem shNext_new {t c : ℕ} {sh qs : ℕ → ℝ} : shNext t sh qs (128 * t + c) = qs c := by
  simp [shNext]

/-- **The exact sum after one more trip**: rescaled old sum plus the block's weights. -/
theorem Lam_succ {L t : ℕ} (h : 128 * t ≤ L) (sh qs : ℕ → ℝ) (m m' : ℝ) :
    Lam (shNext t sh qs) (keysN L (t + 1)) m' =
      Real.exp (m - m') * Lam sh (keysN L t) m + ∑ c ∈ blk L t, Real.exp (qs c - m') := by
  unfold Lam
  rw [sum_keysN_succ h, Finset.mul_sum]
  congr 1
  · refine sum_congr rfl fun j hj => ?_
    have : j < 128 * t := by rw [keysN_eq_range h] at hj; exact mem_range.mp hj
    rw [shNext_old this, ← Real.exp_add]; congr 1; ring
  · refine sum_congr rfl fun c _ => ?_
    rw [shNext_new]

/-- The same for the accumulator. -/
theorem Asum_succ {L t : ℕ} (h : 128 * t ≤ L) (sh qs vv : ℕ → ℝ) (m m' : ℝ) :
    Asum (shNext t sh qs) vv (keysN L (t + 1)) m' =
      Real.exp (m - m') * Asum sh vv (keysN L t) m +
        ∑ c ∈ blk L t, Real.exp (qs c - m') * vv (128 * t + c) := by
  unfold Asum
  rw [sum_keysN_succ h, Finset.mul_sum]
  congr 1
  · refine sum_congr rfl fun j hj => ?_
    have : j < 128 * t := by rw [keysN_eq_range h] at hj; exact mem_range.mp hj
    rw [shNext_old this, ← mul_assoc, ← Real.exp_add]; congr 2; ring
  · refine sum_congr rfl fun c _ => ?_
    rw [shNext_new]

/-- The same for the magnitude. -/
theorem Aabs_succ {L t : ℕ} (h : 128 * t ≤ L) (sh qs vv : ℕ → ℝ) (m m' : ℝ) :
    Aabs (shNext t sh qs) vv (keysN L (t + 1)) m' =
      Real.exp (m - m') * Aabs sh vv (keysN L t) m +
        ∑ c ∈ blk L t, Real.exp (qs c - m') * |vv (128 * t + c)| := by
  unfold Aabs
  rw [sum_keysN_succ h, Finset.mul_sum]
  congr 1
  · refine sum_congr rfl fun j hj => ?_
    have : j < 128 * t := by rw [keysN_eq_range h] at hj; exact mem_range.mp hj
    rw [shNext_old this, ← mul_assoc, ← Real.exp_add]; congr 2; ring
  · refine sum_congr rfl fun c _ => ?_
    rw [shNext_new]

end Num

end Claims.PrefillAttnVarlen

end

section
/-!
# One trip of the sum `l`, from the float operations' errors

The facts one trip's float operations give (`TVA.Device`), as inequalities over `ℝ`:
`tl.exp`, the row sum, `alpha * l_i + beta * l_ij`, `beta / l_i_new`, `p * p_scale`,
`p.to(bfloat16)`, the rescaling `acc * acc_scale`, and `acc += tl.dot(p, v)`.
`trip_l` bounds the sum `l`'s trip against `α·l + Σ q`; `trip_a`, the accumulator's (times the new
`l`) against `α·(l·a) + Σ q·v`, where `q c = e c · β` is the exact weight of lane `c`.
-/

namespace Claims.PrefillAttnVarlen

namespace Num

open Finset

/-- The lanes off `C` hold zero: the sum of `p` over `range 128` is its sum over `C`. -/
theorem tl_sum_lanes {C : Finset ℕ} {p : ℕ → ℝ} (hC : C ⊆ range 128)
    (hp0 : ∀ c ∈ range 128, c ∉ C → p c = 0) : ∑ c ∈ range 128, p c = ∑ c ∈ C, p c :=
  (sum_subset hC hp0).symm

/-- The lanes off `C` hold zero: the sum of `|p|` over `range 128` is its sum over `C`. -/
theorem tl_sum_abs_lanes {C : Finset ℕ} {p : ℕ → ℝ} (hC : C ⊆ range 128)
    (hp0 : ∀ c ∈ range 128, c ∉ C → p c = 0) :
    ∑ c ∈ range 128, |p c| = ∑ c ∈ C, |p c| :=
  (sum_subset hC fun c hc hcC => by rw [hp0 c hc hcC, abs_zero]).symm

/-- **The row sum** `lij` against the exact `Σ_C e`: rate `ρ + rS·(1 + ρ)`. -/
theorem tl_row {C : Finset ℕ} {e p : ℕ → ℝ} {lij ρ : ℝ} (hC : C ⊆ range 128)
    (he0 : ∀ c ∈ C, 0 ≤ e c) (hp0 : ∀ c ∈ range 128, c ∉ C → p c = 0)
    (hp : ∀ c ∈ C, |p c - e c| ≤ ρ * e c + ηE)
    (hlij : |lij - ∑ c ∈ range 128, p c| ≤ rS * ∑ c ∈ range 128, |p c| + ηS) :
    |lij - ∑ c ∈ C, e c| ≤
      (ρ + rS * (1 + ρ)) * ∑ c ∈ C, e c + ((1 + rS) * (ηE * C.card) + ηS) := by
  rw [tl_sum_lanes hC hp0, tl_sum_abs_lanes hC hp0] at hlij
  exact TVA.Softmax.approx_trans (by norm_num [rS]) (TVA.Softmax.block_err_one hp)
    (TVA.Softmax.block_abs_le_one he0 hp) hlij

/-- **The block term** `β'·lij` against `P·β`, from `β'`'s and `lij`'s errors. -/
theorem tl_block {β β' lij P ρ r η a : ℝ} (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1) (hP0 : 0 ≤ P)
    (hP : P ≤ 128) (hρ : 0 ≤ ρ) (hr : 0 ≤ r) (hη : 0 ≤ η) (ha : 0 ≤ a)
    (hβ' : |β' - β| ≤ ρ * β + η) (hl : |lij - P| ≤ r * P + a) :
    |β' * lij - P * β| ≤
      (ρ * (1 + r) + r) * (P * β) + (ρ * a + η * ((1 + r) * 128 + a) + a) := by
  have hlabs : |lij| ≤ (1 + r) * P + a := by
    have := abs_sub_abs_le_abs_sub lij P
    rw [abs_of_nonneg hP0] at this
    linarith
  have e : β' * lij - P * β = (β' - β) * lij + β * (lij - P) := by ring
  rw [e]
  have h1 : |(β' - β) * lij| ≤ (ρ * β + η) * ((1 + r) * P + a) := by
    rw [abs_mul]
    exact mul_le_mul hβ' hlabs (abs_nonneg _) (by positivity)
  have h2 : |β * (lij - P)| ≤ β * (r * P + a) := by
    rw [abs_mul, abs_of_nonneg hβ0]
    exact mul_le_mul_of_nonneg_left hl hβ0
  have h3 := abs_add_le ((β' - β) * lij) (β * (lij - P))
  have h4 : ρ * β * a ≤ ρ * a := by nlinarith [mul_nonneg hρ ha]
  have h5 : η * ((1 + r) * P) ≤ η * ((1 + r) * 128) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hP (by linarith)) hη
  have h6 : β * a ≤ a := by nlinarith
  nlinarith

/-- **One trip with split rates**: `step_err` with the rates of the scaled state and of the block
kept apart. -/
theorem tl_step {ρa ρb ρo α x a b Y Z Bm ηa ηb ηB ηo x' : ℝ} (hρo : 0 ≤ ρo) (hα : 0 ≤ α)
    (ha : |a - α * x| ≤ ρa * (α * |x|) + ηa) (hb : |b - Y| ≤ ρb * Z + ηb)
    (hBm : Bm ≤ (1 + ρb) * Z + ηB) (ho : |x' - (a + b)| ≤ ρo * (|a| + Bm) + ηo) :
    |x' - (α * x + Y)| ≤ (ρa + ρo * (1 + ρa)) * (α * |x|) + (ρb + ρo * (1 + ρb)) * Z +
      ((1 + ρo) * ηa + ηb + ρo * ηB + ηo) := by
  have hax : |a| ≤ (1 + ρa) * (α * |x|) + ηa := by
    have h1 := abs_sub_abs_le_abs_sub a (α * x)
    rw [abs_mul, abs_of_nonneg hα] at h1
    linarith
  have e : x' - (α * x + Y) = (x' - (a + b)) + (a - α * x) + (b - Y) := by ring
  rw [e]
  have h2 := mul_le_mul_of_nonneg_left (add_le_add hax hBm) hρo
  have h3 := (abs_add_le ((x' - (a + b)) + (a - α * x)) (b - Y)).trans
    (add_le_add (abs_add_le (x' - (a + b)) (a - α * x)) le_rfl)
  nlinarith

/-- The scaled state's rate is within `cL0 + cL1·Δ`: `rA Δ ≤ 1/200`, so `rO·(1 + rA Δ)` is below
`2^-21`. -/
theorem tl_ra {Δ : ℝ} (hΔ0 : 0 ≤ Δ) (hΔ : Δ ≤ 10000) :
    rA Δ + rO * (1 + rA Δ) ≤ cL0 + cL1 * Δ := by
  have hw : rA Δ ≤ 1 / 200 := by norm_num [rA]; linarith
  have ho : rO * (1 + rA Δ) ≤ 2 ^ (-21 : ℤ) :=
    (mul_le_mul_of_nonneg_left (by linarith) (by norm_num [rO])).trans
      (show rO * (1 + 1 / 200) ≤ 2 ^ (-21 : ℤ) by norm_num [rO])
  have h1 : rA Δ + 2 ^ (-21 : ℤ) ≤ cL0 + cL1 * Δ := by norm_num [rA, cL0, cL1]; linarith
  linarith

/-- The block's rate is within `bL Y`. -/
theorem tl_rb {Y : ℝ} (hY0 : 0 ≤ Y) (hY : Y ≤ 5000) :
    (rE Y * (1 + (rE Y + rS * (1 + rE Y))) + (rE Y + rS * (1 + rE Y))) +
        rO * (1 + (rE Y * (1 + (rE Y + rS * (1 + rE Y))) + (rE Y + rS * (1 + rE Y)))) ≤
      bL Y := by
  have hYY : Y * Y ≤ 5000 * Y := mul_le_mul_of_nonneg_right hY hY0
  norm_num [rE, rS, rO, bL]
  nlinarith

/-- The absolute terms sum to at most `ηL`. -/
theorem tl_abs {ρ r a l k : ℝ} (hρ0 : 0 ≤ ρ) (hρ : ρ ≤ 1) (hr0 : 0 ≤ r) (hr : r ≤ 1)
    (hk0 : 0 ≤ k) (hk : k ≤ 128) (ha : a = (1 + rS) * (ηE * k) + ηS) (hl0 : 0 ≤ l)
    (hl : l ≤ 2 ^ 18) :
    (1 + rO) * (ηE * l) + (ρ * a + ηE * ((1 + r) * 128 + a) + a) +
        rO * (ρ * a + ηE * ((1 + r) * 128 + a) + a) + ηO ≤ ηL := by
  have ha0 : 0 ≤ a := by rw [ha]; norm_num [rS, ηE, ηS]; positivity
  have ha1 : a ≤ 2 ^ (-130 : ℤ) := by rw [ha]; norm_num [rS, ηE, ηS]; linarith
  have hb : ρ * a + ηE * ((1 + r) * 128 + a) + a ≤ 2 ^ (-125 : ℤ) := by
    have h1 : ρ * a ≤ a := by nlinarith
    have h2 : ηE * ((1 + r) * 128 + a) ≤ ηE * 300 := by
      apply mul_le_mul_of_nonneg_left _ (by norm_num [ηE])
      have : a ≤ 1 := by linarith [show (2 : ℝ) ^ (-130 : ℤ) ≤ 1 by norm_num]
      nlinarith
    norm_num [ηE] at h2 ha1 ⊢
    linarith
  have hb0 : 0 ≤ ρ * a + ηE * ((1 + r) * 128 + a) + a := by
    have : 0 ≤ ηE := by norm_num [ηE]
    positivity
  have hla : ηE * l ≤ 2 ^ (-122 : ℤ) := by norm_num [ηE]; linarith
  have hrO : rO ≤ 1 := by norm_num [rO]
  have hrO0 : 0 ≤ rO := by norm_num [rO]
  have h1 : (1 + rO) * (ηE * l) ≤ 2 * 2 ^ (-122 : ℤ) := by
    have : 0 ≤ ηE * l := mul_nonneg (by norm_num [ηE]) hl0
    nlinarith
  have h2 : rO * (ρ * a + ηE * ((1 + r) * 128 + a) + a) ≤ 2 ^ (-125 : ℤ) := by nlinarith
  norm_num [ηO, ηL] at h1 h2 hb ⊢
  linarith

/-- **One trip of `l`.** The lanes `C ⊆ [0, 128)` hold attended keys; `e c = exp (s c − m_ij) ∈ [0, 1]`
and `β = exp (m_ij − m_new) ∈ [0, 1]`, so the exact weight of lane `c` is `q c = e c · β`. The
computed `p c` (zero off `C`), `β'`, `α'`, the row sum `lij` and the new sum `l'` are within the
device's errors; then `l'` is within `ra·α·|l| + rb·Σ q + η` of `α·l + Σ q`. -/
theorem trip_l {C : Finset ℕ} {e p : ℕ → ℝ} {β β' α α' l l' lij Δ Y : ℝ}
    (hC : C ⊆ range 128) (hY0 : 0 ≤ Y) (hY : Y ≤ 5000) (hΔ0 : 0 ≤ Δ) (hΔ : Δ ≤ 2 * Y)
    (hα0 : 0 ≤ α) (_hα1 : α ≤ 1) (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1)
    (he0 : ∀ c ∈ C, 0 ≤ e c) (he1 : ∀ c ∈ C, e c ≤ 1)
    (hp0 : ∀ c ∈ range 128, c ∉ C → p c = 0)
    (hp : ∀ c ∈ C, |p c - e c| ≤ rE Y * e c + ηE)
    (hβ' : |β' - β| ≤ rE Y * β + ηE) (hα' : |α' - α| ≤ rA Δ * α + ηE)
    (hl0 : 0 ≤ l) (hl : l ≤ 2 ^ 18)
    (hlij : |lij - ∑ c ∈ range 128, p c| ≤ rS * ∑ c ∈ range 128, |p c| + ηS)
    (hl' : |l' - (α' * l + β' * lij)| ≤ rO * (|α' * l| + |β' * lij|) + ηO) :
    |l' - (α * l + ∑ c ∈ C, e c * β)| ≤
      (cL0 + cL1 * Δ) * (α * |l|) + bL Y * (∑ c ∈ C, e c * β) + ηL := by
  have hk : (C.card : ℝ) ≤ 128 := by
    have := card_le_card hC
    rw [card_range] at this
    exact_mod_cast this
  have hP0 : 0 ≤ ∑ c ∈ C, e c := sum_nonneg he0
  have hP : ∑ c ∈ C, e c ≤ 128 :=
    (sum_le_card_nsmul C e 1 he1).trans (by simpa using hk)
  have hρ0 : 0 ≤ rE Y := by unfold rE; positivity
  have hρ1 : rE Y ≤ 1 / 400 := by norm_num [rE]; linarith
  have hrS0 : 0 ≤ rS := by norm_num [rS]
  have hr0 : 0 ≤ rE Y + rS * (1 + rE Y) := by positivity
  have hr1 : rE Y + rS * (1 + rE Y) ≤ 1 := by norm_num [rS] at hρ1 ⊢; nlinarith
  have hηE : 0 ≤ ηE := by norm_num [ηE]
  have ha0 : 0 ≤ (1 + rS) * (ηE * C.card) + ηS := by norm_num [ηS]; positivity
  have hb := tl_block hβ0 hβ1 hP0 hP hρ0 hr0 hηE ha0 hβ' (tl_row hC he0 hp0 hp hlij)
  have hPβ : 0 ≤ (∑ c ∈ C, e c) * β := mul_nonneg hP0 hβ0
  have hst := tl_step (by norm_num [rO]) hα0 (TVA.Softmax.scale_err hα') hb
    (TVA.Softmax.abs_le_of_approx hb (abs_of_nonneg hPβ).le) hl'
  rw [abs_of_nonneg hl0] at hst
  rw [← sum_mul]
  have h1 := mul_le_mul_of_nonneg_right (tl_ra hΔ0 (by linarith)) (mul_nonneg hα0 (abs_nonneg l))
  have h2 := mul_le_mul_of_nonneg_right (tl_rb hY0 hY) hPβ
  have h3 := tl_abs hρ0 (by linarith) hr0 hr1 (Nat.cast_nonneg _) hk rfl hl0 hl
  rw [abs_of_nonneg hl0] at h1 ⊢
  linarith

end Num

end Claims.PrefillAttnVarlen

end

section
/-!
# One trip of the accumulator, from the float operations' errors

The facts one trip's float operations give (`TVA.Device`), as inequalities over `ℝ`:
`tl.exp`, the row sum, `alpha * l_i + beta * l_ij`, `beta / l_i_new`, `p * p_scale`,
`p.to(bfloat16)`, the rescaling `acc * acc_scale`, and `acc += tl.dot(p, v)`.
`trip_l` bounds the sum `l`'s trip against `α·l + Σ q`; `trip_a`, the accumulator's (times the new
`l`) against `α·(l·a) + Σ q·v`, where `q c = e c · β` is the exact weight of lane `c`.
-/

namespace Claims.PrefillAttnVarlen

namespace Num

open Finset

/-- Multiplying an approximation of a quotient `b / l'` by its positive denominator `l'`. -/
theorem ta_mul_den {l' ps b d η : ℝ} (hl' : 0 < l') (h : |ps - b / l'| ≤ d * (b / l') + η) :
    |l' * ps - b| ≤ d * b + η * l' := by
  have e1 : l' * ps - b = l' * (ps - b / l') := by field_simp
  have e2 : l' * (d * (b / l') + η) = d * b + η * l' := by field_simp
  rw [e1, abs_mul, abs_of_pos hl']
  have := mul_le_mul_of_nonneg_left h hl'.le
  linarith

/-- **One lane's weight.** The computed `p2` (the `bfloat16` of `p · ps`), times the new sum `l'`, is
within `ρq·q + 2^-110` of the exact weight `q = e·β`, where
`ρq = (1 + ρ2)·(1 + 2^-22)·(1 + ρ)^2 − 1` and `ρ2 = 2^-8 + 2^-23`. -/
theorem ta_weight {e p p2 β β' l' ps ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1 / 400)
    (he0 : 0 ≤ e) (he1 : e ≤ 1) (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1)
    (hp : |p - e| ≤ ρ * e + ηE) (hβ' : |β' - β| ≤ ρ * β + ηE)
    (hl'1 : 1 / 2 ≤ l') (hl'2 : l' ≤ 2 ^ 18)
    (hps : |ps - β' / l'| ≤ 2 ^ (-22 : ℤ) * (β' / l') + 2 ^ (-148 : ℤ))
    (hp2 : |p2 - p * ps| ≤ (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ)) * |p * ps| + 2 ^ (-130 : ℤ)) :
    |l' * p2 - e * β| ≤
      ((1 + (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) * (1 + ρ) ^ 2 - 1) * (e * β) +
        2 ^ (-110 : ℤ) := by
  have hl' : 0 < l' := by linarith
  have c8 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 2 ^ 8 := by norm_num
  have c22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
  have c23 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
  rw [c22, show (2 : ℝ) ^ (-148 : ℤ) = 1 / 2 ^ 148 by norm_num] at hps
  rw [c8, c23, show (2 : ℝ) ^ (-130 : ℤ) = 1 / 2 ^ 130 by norm_num] at hp2
  rw [c8, c22, c23, show (2 : ℝ) ^ (-110 : ℤ) = 1 / 2 ^ 110 by norm_num]
  rw [show ηE = 1 / 2 ^ 140 by norm_num [ηE]] at hp hβ'
  have hD := ta_mul_den hl' hps
  have hPle : |p| ≤ (1 + ρ) * e + 1 / 2 ^ 140 := by
    have := abs_sub_abs_le_abs_sub p e
    rw [abs_of_nonneg he0] at this
    linarith
  have hBle : |β'| ≤ (1 + ρ) * β + 1 / 2 ^ 140 := by
    have := abs_sub_abs_le_abs_sub β' β
    rw [abs_of_nonneg hβ0] at this
    linarith
  have hD' : |l' * ps - β'| ≤ 1 / 2 ^ 22 * |β'| + 1 / 2 ^ 148 * l' := by
    have := le_abs_self β'
    linarith
  have h2 : |l' * p2 - p * (l' * ps)| ≤
      (1 / 2 ^ 8 + 1 / 2 ^ 23) * (|p| * |l' * ps|) + 1 / 2 ^ 130 * l' := by
    have e1 : l' * p2 - p * (l' * ps) = l' * (p2 - p * ps) := by ring
    have e2 : |p| * |l' * ps| = l' * |p * ps| := by
      rw [abs_mul, abs_mul, abs_of_pos hl']; ring
    rw [e1, abs_mul, abs_of_pos hl', e2]
    have := mul_le_mul_of_nonneg_left hp2 hl'.le
    linarith
  have h3 : |p * (l' * ps) - p * β'| = |p| * |l' * ps - β'| := by rw [← mul_sub, abs_mul]
  have h4 : |p * β' - e * β| ≤ |p - e| * |β'| + e * |β' - β| := by
    have e1 : p * β' - e * β = (p - e) * β' + e * (β' - β) := by ring
    rw [e1]
    calc _ ≤ |(p - e) * β'| + |e * (β' - β)| := abs_add_le _ _
      _ = _ := by rw [abs_mul, abs_mul, abs_of_nonneg he0]
  have hu : |l' * ps| ≤ |β'| + |l' * ps - β'| := by
    have := abs_sub_abs_le_abs_sub (l' * ps) β'
    linarith
  have htri : |l' * p2 - e * β| ≤
      |l' * p2 - p * (l' * ps)| + |p * (l' * ps) - p * β'| + |p * β' - e * β| := by
    have e1 : l' * p2 - e * β =
        (l' * p2 - p * (l' * ps)) + (p * (l' * ps) - p * β') + (p * β' - e * β) := by ring
    rw [e1]
    exact (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
  have hP0 : 0 ≤ |p| := abs_nonneg _
  have hB0 : 0 ≤ |β'| := abs_nonneg _
  have hPe : 0 ≤ (1 + ρ) * e + 1 / 2 ^ 140 := by positivity
  have k1 := mul_le_mul_of_nonneg_left hu hP0
  have k2 := mul_le_mul_of_nonneg_left hD' hP0
  have k3 := mul_le_mul hPle hBle hB0 hPe
  have k4 := mul_le_mul hp hBle hB0 (by positivity)
  have k5 := mul_le_mul_of_nonneg_left hβ' he0
  have k6 := mul_le_mul hPle hl'2 hl'.le hPe
  have k7 : ρ * e ≤ 1 / 400 := by nlinarith
  have k8 : ρ * β ≤ 1 / 400 := by nlinarith
  nlinarith

/-- **The rescaled old state.** `acc1` (the rounded `a · S`, with `S` the `acc_scale` from `α'`), times
the new sum `l'`, is within `ρsc·(α·|l·a|) + 2^-115·(1 + V)` of `α·(l·a)`, where
`ρsc = (1 + x)·(1 + 2^-22 + 2^-23)·(1 + 2^-24) − 1` and `x` is the relative error of `α'`. -/
theorem ta_scale {α α' l l' S a acc1 x V : ℝ} (hα0 : 0 ≤ α)
    (hα' : |α' - α| ≤ x * α + ηE) (hl0 : 0 ≤ l) (hl : l ≤ 2 ^ 18) (hl'1 : 1 / 2 ≤ l')
    (hl'2 : l' ≤ 2 ^ 18)
    (hS : |S - (l / l') * α'| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ)) * ((l / l') * α') + 2 ^ (-140 : ℤ))
    (ha1 : |acc1 - a * S| ≤ 2 ^ (-24 : ℤ) * |a * S| + 2 ^ (-149 : ℤ))
    (hV0 : 0 ≤ V) (ha : |a| ≤ 8 * (1 + V)) :
    |l' * acc1 - α * (l * a)| ≤
      ((1 + x) * (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-24 : ℤ)) - 1) * (α * |l * a|) +
        2 ^ (-115 : ℤ) * (1 + V) := by
  have hl' : 0 < l' := by linarith
  have c22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
  have c23 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
  have c24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  rw [c22, c23, show (2 : ℝ) ^ (-140 : ℤ) = 1 / 2 ^ 140 by norm_num, div_mul_eq_mul_div] at hS
  rw [c24, show (2 : ℝ) ^ (-149 : ℤ) = 1 / 2 ^ 149 by norm_num] at ha1
  rw [c22, c23, c24, show (2 : ℝ) ^ (-115 : ℤ) = 1 / 2 ^ 115 by norm_num, abs_mul l a,
    abs_of_nonneg hl0]
  rw [show ηE = 1 / 2 ^ 140 by norm_num [ηE]] at hα'
  have hS' := ta_mul_den hl' hS
  have hα'le : α' ≤ (1 + x) * α + 1 / 2 ^ 140 := by
    have := le_abs_self (α' - α)
    linarith
  have hlα : |l * α' - l * α| ≤ l * (x * α + 1 / 2 ^ 140) := by
    rw [← mul_sub, abs_mul, abs_of_nonneg hl0]
    exact mul_le_mul_of_nonneg_left hα' hl0
  have k1 := mul_le_mul_of_nonneg_left hα'le hl0
  have hT : |l' * S - l * α| ≤ ((1 + x) * (1 + (1 / 2 ^ 22 + 1 / 2 ^ 23)) - 1) * (l * α) +
      1 / 2 ^ 120 := by
    have := abs_sub_le (l' * S) (l * α') (l * α)
    nlinarith
  have hU : |l' * (a * S) - α * (l * a)| ≤
      |a| * (((1 + x) * (1 + (1 / 2 ^ 22 + 1 / 2 ^ 23)) - 1) * (l * α) + 1 / 2 ^ 120) := by
    have e1 : l' * (a * S) - α * (l * a) = a * (l' * S - l * α) := by ring
    rw [e1, abs_mul]
    exact mul_le_mul_of_nonneg_left hT (abs_nonneg a)
  have hW : |l' * acc1 - l' * (a * S)| ≤ 1 / 2 ^ 24 * |l' * (a * S)| + 1 / 2 ^ 149 * l' := by
    rw [← mul_sub, abs_mul, abs_mul l', abs_of_pos hl']
    have := mul_le_mul_of_nonneg_left ha1 hl'.le
    linarith
  have hlaS : |l' * (a * S)| ≤ α * (l * |a|) + |l' * (a * S) - α * (l * a)| := by
    have h1 := abs_sub_abs_le_abs_sub (l' * (a * S)) (α * (l * a))
    rw [abs_mul α, abs_mul l, abs_of_nonneg hα0, abs_of_nonneg hl0] at h1
    linarith
  have htri := abs_sub_le (l' * acc1) (l' * (a * S)) (α * (l * a))
  have ka : |a| * (1 / 2 ^ 120) ≤ 8 * (1 + V) * (1 / 2 ^ 120) :=
    mul_le_mul_of_nonneg_right ha (by norm_num)
  nlinarith

/-- **One trip, split rates.** The new state `X2` is a dot accumulation (rate `rd`, absolute `A0`)
of the scaled state `X1` (within `ρs·M1 + η1` of `T1`, `|T1| ≤ M1`) and the block term `W` (within
`ρq·Z + η2` of `T2`, magnitude `Wm`); then `X2` is within
`(ρs + rd·(1 + ρs))·M1 + (ρq + rd·(1 + ρq))·Z + ((1 + rd)·(η1 + η2) + A0)` of `T1 + T2`. -/
theorem ta_combine {X2 X1 W Wm T1 T2 M1 Z rd ρs ρq A0 η1 η2 : ℝ} (hrd : 0 ≤ rd)
    (hM1 : |T1| ≤ M1) (hd : |X2 - (X1 + W)| ≤ rd * (|X1| + Wm) + A0)
    (hs : |X1 - T1| ≤ ρs * M1 + η1) (hw : |W - T2| ≤ ρq * Z + η2)
    (hwm : Wm ≤ (1 + ρq) * Z + η2) :
    |X2 - (T1 + T2)| ≤
      (ρs + rd * (1 + ρs)) * M1 + (ρq + rd * (1 + ρq)) * Z + ((1 + rd) * (η1 + η2) + A0) := by
  have hX1 : |X1| ≤ (1 + ρs) * M1 + η1 := by
    have := abs_sub_abs_le_abs_sub X1 T1
    linarith
  have k1 := mul_le_mul_of_nonneg_left (add_le_add hX1 hwm) hrd
  have e1 : X2 - (T1 + T2) = (X2 - (X1 + W)) + (X1 - T1) + (W - T2) := by ring
  rw [e1]
  have := (abs_add_le ((X2 - (X1 + W)) + (X1 - T1)) (W - T2)).trans
    (add_le_add (abs_add_le (X2 - (X1 + W)) (X1 - T1)) le_rfl)
  nlinarith

/-- The rescaled state's rate, through the dot: at most `cA0 + cA1·Δ`. -/
theorem ta_rate_a {Δ : ℝ} (hΔ0 : 0 ≤ Δ) :
    ((1 + rA Δ) * (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-24 : ℤ)) - 1) +
        rD * (1 + ((1 + rA Δ) * (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-24 : ℤ)) - 1)) ≤
      cA0 + cA1 * Δ := by
  norm_num [rA, rD, cA0, cA1]
  linarith

/-- The block's rate, through the dot: at most `bA Y`. -/
theorem ta_rate_b {Y : ℝ} (hY0 : 0 ≤ Y) (hY : Y ≤ 5000) :
    ((1 + (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) * (1 + rE Y) ^ 2 - 1) +
        rD * (1 + ((1 + (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) * (1 + rE Y) ^ 2 - 1)) ≤
      bA Y := by
  norm_num [rE, rD, bA]
  nlinarith [mul_nonneg hY0 (sub_nonneg.mpr hY)]

/-- **One trip of the accumulator** (one column, times the new `l'`). Besides `trip_l`'s data:
`ps` is `beta / l_i_new`, `p2 c` the `bfloat16` of `p c · ps`, `S` the `acc_scale`, `a` the old
accumulator (so `l·a` is the old unnormalised state), `acc1 = a·S` rounded, `acc2` the new
accumulator after the `tl.dot` of `p2` with the column `v` of values; `V` bounds `Σ |v|`. Then `l'·acc2`
is within `ra·α·|l·a| + rb·Σ q·|v| + η·(1 + V)` of `α·(l·a) + Σ q·v`. -/
theorem trip_a {C : Finset ℕ} {e p p2 v : ℕ → ℝ}
    {β β' α α' l l' ps S a acc1 acc2 Δ Y V : ℝ}
    (hC : C ⊆ range 128) (hY0 : 0 ≤ Y) (hY : Y ≤ 5000) (hΔ0 : 0 ≤ Δ) (hΔ : Δ ≤ 2 * Y)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (hβ0 : 0 ≤ β) (hβ1 : β ≤ 1)
    (he0 : ∀ c ∈ C, 0 ≤ e c) (he1 : ∀ c ∈ C, e c ≤ 1)
    (hp : ∀ c ∈ C, |p c - e c| ≤ rE Y * e c + ηE)
    (hβ' : |β' - β| ≤ rE Y * β + ηE) (hα' : |α' - α| ≤ rA Δ * α + ηE)
    (hl0 : 0 ≤ l) (hl : l ≤ 2 ^ 18) (hl'1 : 1 / 2 ≤ l') (hl'2 : l' ≤ 2 ^ 18)
    (hps : |ps - β' / l'| ≤ 2 ^ (-22 : ℤ) * (β' / l') + 2 ^ (-148 : ℤ))
    (hp2 : ∀ c ∈ C, |p2 c - p c * ps| ≤ (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ)) * |p c * ps| + 2 ^ (-130 : ℤ))
    (hS : |S - (l / l') * α'| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ)) * ((l / l') * α') + 2 ^ (-140 : ℤ))
    (ha1 : |acc1 - a * S| ≤ 2 ^ (-24 : ℤ) * |a * S| + 2 ^ (-149 : ℤ))
    (ha2 : |acc2 - (acc1 + ∑ c ∈ C, p2 c * v c)| ≤
      rD * (|acc1| + ∑ c ∈ C, |p2 c * v c|) + (adc0 + adc1 * ∑ c ∈ C, |v c|))
    (hv : ∀ c ∈ C, |v c| ≤ 2 ^ 120) (hV0 : 0 ≤ V) (hV : ∑ c ∈ C, |v c| ≤ V)
    (hla : |l * a| ≤ 4 * (1 + V)) (ha : |a| ≤ 8 * (1 + V)) :
    |l' * acc2 - (α * (l * a) + ∑ c ∈ C, (e c * β) * v c)| ≤
      (cA0 + cA1 * Δ) * (α * |l * a|) + bA Y * (∑ c ∈ C, (e c * β) * |v c|) + ηA * (1 + V) := by
  -- the lane set, `α ≤ 1`, `Δ ≤ 2·Y` and the size bounds of `v` and `l·a` are not needed here
  have _ := And.intro hC (And.intro hΔ (And.intro hα1 (And.intro hv hla)))
  have hl' : 0 < l' := by linarith
  have hρ0 : 0 ≤ rE Y := by unfold rE; positivity
  have hρ1 : rE Y ≤ 1 / 400 := by norm_num [rE]; linarith
  have hw : ∀ c ∈ C, |l' * p2 c - e c * β| ≤
      ((1 + (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) * (1 + rE Y) ^ 2 - 1) * (e c * β) +
        2 ^ (-110 : ℤ) := fun c hc =>
    ta_weight hρ0 hρ1 (he0 c hc) (he1 c hc) hβ0 hβ1 (hp c hc) hβ' hl'1 hl'2 hps (hp2 c hc)
  have hq0 : ∀ c ∈ C, 0 ≤ e c * β := fun c hc => mul_nonneg (he0 c hc) hβ0
  have hblk : |∑ c ∈ C, l' * p2 c * v c - ∑ c ∈ C, e c * β * v c| ≤
      ((1 + (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) * (1 + rE Y) ^ 2 - 1) *
          ∑ c ∈ C, e c * β * |v c| + 2 ^ (-110 : ℤ) * ∑ c ∈ C, |v c| :=
    TVA.Softmax.block_err (p := fun c => e c * β) (p' := fun c => l' * p2 c) (v := v) hw
  have hblkm : ∑ c ∈ C, |l' * p2 c * v c| ≤
      (1 + ((1 + (2 ^ (-8 : ℤ) + 2 ^ (-23 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) * (1 + rE Y) ^ 2 - 1)) *
          ∑ c ∈ C, e c * β * |v c| + 2 ^ (-110 : ℤ) * ∑ c ∈ C, |v c| :=
    TVA.Softmax.block_abs_le (p := fun c => e c * β) (p' := fun c => l' * p2 c) (v := v) hq0 hw
  have hsc := ta_scale hα0 hα' hl0 hl hl'1 hl'2 hS ha1 hV0 ha
  have e1 : ∑ c ∈ C, l' * p2 c * v c = l' * ∑ c ∈ C, p2 c * v c := by
    rw [mul_sum]; exact sum_congr rfl fun c _ => by ring
  have e2 : ∑ c ∈ C, |l' * p2 c * v c| = l' * ∑ c ∈ C, |p2 c * v c| := by
    rw [mul_sum]
    exact sum_congr rfl fun c _ => by rw [mul_assoc, abs_mul, abs_of_pos hl']
  have hd : |l' * acc2 - (l' * acc1 + ∑ c ∈ C, l' * p2 c * v c)| ≤
      rD * (|l' * acc1| + ∑ c ∈ C, |l' * p2 c * v c|) +
        l' * (adc0 + adc1 * ∑ c ∈ C, |v c|) := by
    have e3 : l' * acc2 - (l' * acc1 + ∑ c ∈ C, l' * p2 c * v c) =
        l' * (acc2 - (acc1 + ∑ c ∈ C, p2 c * v c)) := by rw [e1]; ring
    rw [e3, e2, abs_mul, abs_mul, abs_of_pos hl']
    have := mul_le_mul_of_nonneg_left ha2 hl'.le
    linarith
  have hM1 : |α * (l * a)| ≤ α * |l * a| := by rw [abs_mul, abs_of_nonneg hα0]
  have hrD : 0 ≤ rD := by unfold rD; positivity
  have hc := ta_combine hrD hM1 hd hsc hblk hblkm
  have hZ : 0 ≤ ∑ c ∈ C, e c * β * |v c| :=
    sum_nonneg fun c hc => mul_nonneg (hq0 c hc) (abs_nonneg _)
  have hra := mul_le_mul_of_nonneg_right (ta_rate_a hΔ0) (mul_nonneg hα0 (abs_nonneg (l * a)))
  have hrb := mul_le_mul_of_nonneg_right (ta_rate_b hY0 hY) hZ
  have hvs0 : 0 ≤ ∑ c ∈ C, |v c| := sum_nonneg fun c _ => abs_nonneg _
  have hlv := mul_le_mul hl'2 hV hvs0 (by norm_num)
  have habs : (1 + rD) * (2 ^ (-115 : ℤ) * (1 + V) + 2 ^ (-110 : ℤ) * ∑ c ∈ C, |v c|) +
      l' * (adc0 + adc1 * ∑ c ∈ C, |v c|) ≤ ηA * (1 + V) := by
    rw [show rD = 136 / 2 ^ 22 by norm_num [rD], show adc0 = 1 / 2 ^ 118 by norm_num [adc0],
      show adc1 = 1 / 2 ^ 140 by norm_num [adc1], show ηA = 1 / 2 ^ 96 by norm_num [ηA],
      show (2 : ℝ) ^ (-115 : ℤ) = 1 / 2 ^ 115 by norm_num,
      show (2 : ℝ) ^ (-110 : ℤ) = 1 / 2 ^ 110 by norm_num]
    linarith
  exact hc.trans (add_le_add (add_le_add hra hrb) habs)

end Num

end Claims.PrefillAttnVarlen

end
