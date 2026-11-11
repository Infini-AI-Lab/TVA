import Mathlib
import TVA.Proof

/-!
# The real-number core of the accuracy proof 

Everything here is about real numbers. `u = 2^-24`. A column (the sum `l`, or one component of
`acc`) is tracked against the exact sums at the running reference `μ`.
-/
namespace Claims.PagedAttention

namespace Num

open Finset

/-- `2^-24`, the unit roundoff of `float32`. -/
noncomputable def u : ℝ := 1 / 16777216

/-- `exp (t·e1 + γ·(μ + K))`: how the per-trip rate `1 + e1 + γ·(μ' − μ)` compounds over `t`
trips whose references rose from `≥ -K` to `μ`. -/
noncomputable def Ex (e1 γ K : ℝ) (t : ℕ) (μ : ℝ) : ℝ := Real.exp (t * e1 + γ * (μ + K))

/-- The relative error coefficient after `t` trips at reference `μ`: `(1 + ε2)·Ex − 1`. -/
noncomputable def Fc (e1 γ ε2 K : ℝ) (t : ℕ) (μ : ℝ) : ℝ := (1 + ε2) * Ex e1 γ K t μ - 1

/-- `Σ_{k<N} exp (S k − μ)·V k`, the exact accumulator over the first `N` tokens at reference `μ`
(scores `S`, values `V`). -/
noncomputable def aS (S V : ℕ → ℝ) (μ : ℝ) (N : ℕ) : ℝ :=
  ∑ k ∈ range N, Real.exp (S k - μ) * V k

/-- `Σ_{k<N} exp (S k − μ)·|V k|`, its magnitude. -/
noncomputable def mS (S V : ℕ → ℝ) (μ : ℝ) (N : ℕ) : ℝ :=
  ∑ k ∈ range N, Real.exp (S k - μ) * |V k|

/-- `Σ_{k<N} exp (S k − μ)`, the exact sum `l`. -/
noncomputable def lS (S : ℕ → ℝ) (μ : ℝ) (N : ℕ) : ℝ := ∑ k ∈ range N, Real.exp (S k - μ)

/-- **The column invariant**: after `t` trips (`N` tokens seen) at reference `μ`, the computed
value `x` is within `Fc·mS + G·t·Ex` of the exact accumulator. -/
def ColInv (e1 γ ε2 K G : ℝ) (S V : ℕ → ℝ) (N t : ℕ) (μ x : ℝ) : Prop :=
  |x - aS S V μ N| ≤ Fc e1 γ ε2 K t μ * mS S V μ N + G * t * Ex e1 γ K t μ

/-! ## The algebra of one trip -/

/-- One more trip multiplies `Ex` by `exp (e1 + γ·(μ' − μ))`. -/
theorem Ex_succ (e1 γ K : ℝ) (t : ℕ) (μ μ' : ℝ) :
    Ex e1 γ K (t + 1) μ' = Ex e1 γ K t μ * Real.exp (e1 + γ * (μ' - μ)) := by
  unfold Ex
  rw [← Real.exp_add]
  congr 1
  push_cast
  ring

/-- `Ex ≥ 1` while the reference is at least `−K`. -/
theorem Ex_ge_one {e1 γ K : ℝ} (he1 : 0 ≤ e1) (hγ : 0 ≤ γ) (t : ℕ) {μ : ℝ} (hμ : -K ≤ μ) :
    1 ≤ Ex e1 γ K t μ := by
  unfold Ex
  rw [← Real.exp_zero]
  apply Real.exp_le_exp.mpr
  have : 0 ≤ (t : ℝ) * e1 := mul_nonneg (Nat.cast_nonneg t) he1
  have : 0 ≤ γ * (μ + K) := mul_nonneg hγ (by linarith)
  linarith

/-- **One trip of a rescaled accumulation** (the second and later trips). The exact state follows
`X' = α·X + Y`, `M' = α·M + Z`; the computed `α'` is within `(a0 + a1·(μ' − μ))·α + ζ` of `α`
(the references never decrease: `μ ≤ μ'`), the block value `b` within `ρb·Z + ηb` of `Y`, and
the new state `x'` within `ρo·(|α'·x| + |b|) + ηo` of `α'·x + b`. If `x` is within the coefficient
`Fc (t, μ)` of `X`, then `x'` is within `Fc (t+1, μ')` of `α·X + Y`. -/
theorem gen_step {e1 γ ε2 K G a0 a1 ζ ρo ρb ηb ηo : ℝ} {t : ℕ}
    {μ μ' α α' X M Y Z x x' b Bx : ℝ}
    (he1 : 0 ≤ e1) (hγ : 0 ≤ γ) (hε2 : 0 ≤ ε2) (hG : 0 ≤ G) (ha0 : 0 ≤ a0) (ha1 : 0 ≤ a1)
    (hζ : 0 ≤ ζ) (hρo : 0 ≤ ρo)
    (hμ : -K ≤ μ) (hμμ : μ ≤ μ') (hα0 : 0 ≤ α) (hα1 : α ≤ 1)
    (hα' : |α' - α| ≤ (a0 + a1 * (μ' - μ)) * α + ζ)
    (hXM : |X| ≤ M) (hYZ : |Y| ≤ Z)
    (hx : |x - X| ≤ Fc e1 γ ε2 K t μ * M + G * t * Ex e1 γ K t μ) (hBx : |x| ≤ Bx)
    (hb : |b - Y| ≤ ρb * Z + ηb)
    (ho : |x' - (α' * x + b)| ≤ ρo * (|α' * x| + |b|) + ηo)
    (hr1 : a0 * (1 + ρo) + ρo ≤ e1) (hr2 : a1 * (1 + ρo) ≤ γ) (hr3 : ρb + ρo * (1 + ρb) ≤ ε2)
    (hη : (1 + ρo) * (ζ * Bx) + (1 + ρo) * ηb + ηo ≤ G) :
    |x' - (α * X + Y)| ≤
      Fc e1 γ ε2 K (t + 1) μ' * (α * M + Z) + G * (t + 1 : ℕ) * Ex e1 γ K (t + 1) μ' := by
  set D := μ' - μ with hD
  have hD0 : 0 ≤ D := by linarith
  set ρα := a0 + a1 * D with hρα
  have hρα0 : 0 ≤ ρα := by positivity
  set ε1 := ρα * (1 + ρo) + ρo with hε1
  have hε1le : ε1 ≤ e1 + γ * D := by
    have : a1 * D * (1 + ρo) ≤ γ * D := by
      have := mul_le_mul_of_nonneg_right hr2 hD0
      nlinarith
    rw [hε1, hρα]; nlinarith
  have hε10 : 0 ≤ ε1 := by positivity
  set E0 := Ex e1 γ K t μ with hE0
  set E1 := Ex e1 γ K (t + 1) μ' with hE1
  set W := Real.exp (e1 + γ * D) with hW
  have hE1eq : E1 = E0 * W := Ex_succ e1 γ K t μ μ'
  have hE0ge : 1 ≤ E0 := Ex_ge_one he1 hγ t hμ
  have hWge : 1 + ε1 ≤ W := by
    have := Real.add_one_le_exp (e1 + γ * D)
    linarith
  have hE1ge : 1 ≤ E1 := by
    have : 1 ≤ W := by linarith
    rw [hE1eq]; exact one_le_mul_of_one_le_of_one_le hE0ge this
  set F := Fc e1 γ ε2 K t μ with hF
  set F' := Fc e1 γ ε2 K (t + 1) μ' with hF'
  have hFeq : F + 1 = (1 + ε2) * E0 := by rw [hF]; unfold Fc; ring
  have hF'eq : F' + 1 = (1 + ε2) * E1 := by rw [hF']; unfold Fc; ring
  have hF'ge : ε2 ≤ F' := by
    have := mul_le_mul_of_nonneg_left hE1ge (by linarith : 0 ≤ 1 + ε2)
    linarith
  have hFge : 0 ≤ F := by
    have := mul_le_mul_of_nonneg_left hE0ge (by linarith : 0 ≤ 1 + ε2)
    linarith
  have hM0 : 0 ≤ M := (abs_nonneg _).trans hXM
  have hZ0 : 0 ≤ Z := (abs_nonneg _).trans hYZ
  have hg0 : 0 ≤ G * t * E0 := by positivity
  set e := |x - X| with he
  -- decomposition
  have hdec : x' - (α * X + Y) = (x' - (α' * x + b)) + (α' - α) * x + α * (x - X) + (b - Y) := by ring
  have h1 : |x' - (α * X + Y)| ≤ |x' - (α' * x + b)| + |α' - α| * |x| + α * e + |b - Y| := by
    rw [hdec]
    refine (abs_add_le _ _).trans ?_
    refine add_le_add ((abs_add_le _ _).trans (add_le_add ((abs_add_le _ _).trans (add_le_add le_rfl ?_)) ?_)) le_rfl
    · rw [abs_mul]
    · rw [abs_mul, abs_of_nonneg hα0]
  have hα'abs : |α'| ≤ α + (ρα * α + ζ) := by
    have := abs_sub_abs_le_abs_sub α' α
    rw [abs_of_nonneg hα0] at this
    linarith
  have hα'x : |α' * x| ≤ (α + (ρα * α + ζ)) * |x| := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_right hα'abs (abs_nonneg _)
  have hbabs : |b| ≤ (1 + ρb) * Z + ηb := by
    have := abs_sub_abs_le_abs_sub b Y
    linarith
  have hxabs : |x| ≤ M + e := by
    have := abs_sub_abs_le_abs_sub x X
    linarith
  have hxX : |α' - α| * |x| ≤ (ρα * α + ζ) * |x| := mul_le_mul_of_nonneg_right hα' (abs_nonneg _)
  -- the total
  have h3 : ρo * (|α' * x| + |b|) ≤ ρo * ((α + (ρα * α + ζ)) * |x| + ((1 + ρb) * Z + ηb)) :=
    mul_le_mul_of_nonneg_left (add_le_add hα'x hbabs) hρo
  have h2 : |x' - (α * X + Y)| ≤ ε1 * (α * |x|) + (ρb + ρo * (1 + ρb)) * Z +
      ((1 + ρo) * (ζ * |x|) + (1 + ρo) * ηb + ηo) + α * e := by
    calc |x' - (α * X + Y)| ≤ |x' - (α' * x + b)| + |α' - α| * |x| + α * e + |b - Y| := h1
      _ ≤ (ρo * ((α + (ρα * α + ζ)) * |x| + ((1 + ρb) * Z + ηb)) + ηo) + (ρα * α + ζ) * |x| + α * e +
          (ρb * Z + ηb) := by linarith
      _ = _ := by rw [hε1]; ring
  have h4 : ζ * |x| ≤ ζ * Bx := mul_le_mul_of_nonneg_left hBx hζ
  have h5 : ε1 * (α * |x|) ≤ ε1 * (α * (M + e)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hxabs hα0) hε10
  have h6 : (ρb + ρo * (1 + ρb)) * Z ≤ F' * Z :=
    mul_le_mul_of_nonneg_right (hr3.trans hF'ge) hZ0
  -- total with e ≤ F M + g
  set g := G * t * E0 with hg
  have heFg : e ≤ F * M + g := hx
  have h7 : α * ((1 + ε1) * e) ≤ α * ((1 + ε1) * (F * M + g)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left heFg (by linarith)) hα0
  have h8 : (1 + ε1) * (F + 1) ≤ F' + 1 := by
    rw [hFeq, hF'eq, hE1eq]
    have := mul_le_mul_of_nonneg_right hWge (by positivity : 0 ≤ (1 + ε2) * E0)
    calc (1 + ε1) * ((1 + ε2) * E0) ≤ W * ((1 + ε2) * E0) := this
      _ = (1 + ε2) * (E0 * W) := by ring
  have h9 : α * M * ((1 + ε1) * (F + 1) - 1) ≤ α * M * F' := by
    have : (1 + ε1) * (F + 1) - 1 ≤ F' := by linarith
    exact mul_le_mul_of_nonneg_left this (mul_nonneg hα0 hM0)
  have h10 : α * ((1 + ε1) * g) ≤ G * t * E1 := by
    have : (1 + ε1) * g ≤ W * g := mul_le_mul_of_nonneg_right hWge hg0
    have h11 : α * ((1 + ε1) * g) ≤ (1 + ε1) * g := by
      have := mul_le_mul_of_nonneg_right hα1 (mul_nonneg (by linarith : 0 ≤ 1 + ε1) hg0)
      linarith
    calc α * ((1 + ε1) * g) ≤ (1 + ε1) * g := h11
      _ ≤ W * g := this
      _ = G * t * E1 := by rw [hE1eq, hg]; ring
  have h12 : G ≤ G * E1 := le_mul_of_one_le_right hG hE1ge
  have hcast : G * ((t + 1 : ℕ) : ℝ) * E1 = G * t * E1 + G * E1 := by push_cast; ring
  rw [hcast]
  have key : ε1 * (α * (M + e)) + α * e ≤ α * M * F' + G * t * E1 := by
    have e1 : ε1 * (α * (M + e)) + α * e = α * M * ε1 + α * ((1 + ε1) * e) := by ring
    have e2 : α * ((1 + ε1) * (F * M + g)) =
        α * M * ((1 + ε1) * (F + 1) - 1) - α * M * ε1 + α * ((1 + ε1) * g) := by ring
    linarith
  have h4' : (1 + ρo) * (ζ * |x|) ≤ (1 + ρo) * (ζ * Bx) :=
    mul_le_mul_of_nonneg_left h4 (by linarith)
  have e3 : F' * (α * M + Z) = α * M * F' + F' * Z := by ring
  rw [e3]
  linarith


/-- **The first trip** (state `0`, `α' · 0 = 0`): `x'` is within `ρo·|b| + ηo` of `b`, which is
within `ρb·Z + ηb` of `Y`, `|Y| ≤ Z`. -/
theorem gen_first {e1 γ ε2 K G ρo ρb ηb ηo : ℝ} {μ' Y Z x' b : ℝ}
    (he1 : 0 ≤ e1) (hγ : 0 ≤ γ) (hε2 : 0 ≤ ε2) (hG : 0 ≤ G) (hρo : 0 ≤ ρo)
    (hμ' : -K ≤ μ')
    (hYZ : |Y| ≤ Z) (hb : |b - Y| ≤ ρb * Z + ηb) (ho : |x' - b| ≤ ρo * |b| + ηo)
    (hr3 : ρb + ρo * (1 + ρb) ≤ ε2) (hη : (1 + ρo) * ηb + ηo ≤ G) :
    |x' - Y| ≤ Fc e1 γ ε2 K 1 μ' * Z + G * (1 : ℕ) * Ex e1 γ K 1 μ' := by
  have hE := Ex_ge_one he1 hγ 1 hμ'
  have hZ0 : 0 ≤ Z := (abs_nonneg _).trans hYZ
  have hbabs : |b| ≤ (1 + ρb) * Z + ηb := by
    have := abs_sub_abs_le_abs_sub b Y
    linarith
  have h1 : |x' - Y| ≤ |x' - b| + |b - Y| := by
    have := abs_add_le (x' - b) (b - Y)
    rwa [show x' - b + (b - Y) = x' - Y by ring] at this
  have h2 : ρo * |b| ≤ ρo * ((1 + ρb) * Z + ηb) := mul_le_mul_of_nonneg_left hbabs hρo
  have h3 : (ρb + ρo * (1 + ρb)) * Z ≤ ε2 * Z := mul_le_mul_of_nonneg_right hr3 hZ0
  have hF' : ε2 ≤ Fc e1 γ ε2 K 1 μ' := by
    unfold Fc
    have := mul_le_mul_of_nonneg_left hE (by linarith : 0 ≤ 1 + ε2)
    linarith
  have h4 : ε2 * Z ≤ Fc e1 γ ε2 K 1 μ' * Z := mul_le_mul_of_nonneg_right hF' hZ0
  have h5 : G ≤ G * (1 : ℕ) * Ex e1 γ K 1 μ' := by
    have : G * (1 : ℕ) = G := by simp
    rw [this]; exact le_mul_of_one_le_right hG hE
  nlinarith


/-! ## The concrete rates -/

/-- `e1` of the column `l`: the rate per trip of `l`'s rescale-and-add. -/
noncomputable def eL : ℝ := 71 / 10 * u
/-- `e1` of the columns of `acc`. -/
noncomputable def eA : ℝ := 81 / 10 * u
/-- `γ`: the rate per unit of rise of the reference. -/
noncomputable def gam : ℝ := 31 / 10 * u
/-- The one-time rate of `l`'s block (the weights' error, the sum, the addition). -/
noncomputable def eps2L (X : ℝ) : ℝ := 48 * u * X + 20 * u
/-- The one-time rate of `acc`'s block. -/
noncomputable def eps2A (X : ℝ) : ℝ := 48 * u * X + 40 * u
/-- `K`: the references lie in `[-K, K]` when the scaled scores lie in `[-X, X]`. -/
noncomputable def Kx (X : ℝ) : ℝ := X + 1
/-- The absolute error allowed per trip. -/
noncomputable def Gabs : ℝ := (2 : ℝ) ^ (-80 : ℤ)
/-- The absolute error of a computed scaled score from the exact one is at most
`dS X = 28·u·X + 2^-60` when the scores are at most `X` in magnitude. -/
noncomputable def dS (X : ℝ) : ℝ := 28 * u * X + (2 : ℝ) ^ (-60 : ℤ)
/-- The relative error of a computed weight `exp (s − m)` against the exact one. -/
noncomputable def rhoP (X : ℝ) : ℝ := 48 * u * X + 8 * u
/-- Its absolute error. -/
noncomputable def etaP : ℝ := (2 : ℝ) ^ (-130 : ℤ)
/-- The relative error of the computed `alpha` against `exp (m_old − m_new)` is at most
`aA0 + aA1·(m_new − m_old)`. -/
noncomputable def aA0 : ℝ := 81 / 20 * u
/-- See `aA0`. -/
noncomputable def aA1 : ℝ := 61 / 20 * u
/-- Its absolute error. -/
noncomputable def zeta0 : ℝ := (2 : ℝ) ^ (-130 : ℤ)
/-- The relative error of the block sum of the weights (IEEE additions along a tree of depth
at most `8`). -/
noncomputable def rhoSL : ℝ := 81 / 10 * u
/-- The relative error of the block sum of the products `p·v` (rounded products, then fusable
additions along a tree of depth at most `8`). -/
noncomputable def rhoSA : ℝ := 255 / 10 * u
/-- The absolute error of such a block sum. -/
noncomputable def etaS : ℝ := (2 : ℝ) ^ (-120 : ℤ)
/-- The relative error of `l`'s update `l·α' + r` (`tl.mul_add`). -/
noncomputable def rhoOL : ℝ := 3 * u
/-- The relative error of `acc`'s update (`tl.mul_add`, or `tl.addc` of a rounded product when
`PAGE_SIZE = 1`). -/
noncomputable def rhoOA : ℝ := 401 / 100 * u
/-- The absolute error of an update. -/
noncomputable def etaO : ℝ := (2 : ℝ) ^ (-120 : ℤ)

/-- The invariant of the sum `l` (the values are all `1`). -/
def InvL (X : ℝ) (S : ℕ → ℝ) (N t : ℕ) (μ x : ℝ) : Prop :=
  ColInv eL gam (eps2L X) (Kx X) Gabs S (fun _ => 1) N t μ x

/-- The invariant of one component of `acc` (values `V`). -/
def InvA (X : ℝ) (S V : ℕ → ℝ) (N t : ℕ) (μ x : ℝ) : Prop :=
  ColInv eA gam (eps2A X) (Kx X) Gabs S V N t μ x


/-- `u > 0`. -/
theorem u_pos : 0 < u := by unfold u; norm_num

/-- `exp z ≤ 1 / (1 - z)` for `z < 1`. -/
theorem exp_le_inv {z : ℝ} (hz : z < 1) : Real.exp z ≤ 1 / (1 - z) := by
  have h := Real.add_one_le_exp (-z)
  calc Real.exp z = 1 / Real.exp (-z) := by rw [Real.exp_neg]; field_simp
    _ ≤ 1 / (1 - z) := one_div_le_one_div_of_le (by linarith) (by linarith)

/-- The accumulator over `N + c` tokens at `μ'` from the one over `N` at `μ`. -/
theorem aS_add (S V : ℕ → ℝ) (μ μ' : ℝ) (N c : ℕ) :
    aS S V μ' (N + c) = Real.exp (μ - μ') * aS S V μ N +
      ∑ j ∈ range c, Real.exp (S (N + j) - μ') * V (N + j) := by
  unfold aS
  have h1 : ∑ k ∈ range N, Real.exp (S k - μ') * V k =
      Real.exp (μ - μ') * ∑ k ∈ range N, Real.exp (S k - μ) * V k := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [← mul_assoc, ← Real.exp_add]; congr 2; ring
  rw [Finset.sum_range_add, h1]

/-- The magnitude sum is the accumulator of `|V|`. -/
theorem mS_eq (S V : ℕ → ℝ) (μ : ℝ) (N : ℕ) : mS S V μ N = aS S (fun k => |V k|) μ N := rfl

/-- `|aS| ≤ mS`. -/
theorem abs_aS_le (S V : ℕ → ℝ) (μ : ℝ) (N : ℕ) : |aS S V μ N| ≤ mS S V μ N := by
  unfold aS mS
  refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq (Finset.sum_congr rfl fun k _ => ?_))
  rw [abs_mul, abs_of_pos (Real.exp_pos _)]

/-- The magnitude sum is at most `N·exp Δ·Vm` when every score is at most `μ + Δ`. -/
theorem mS_le (S V : ℕ → ℝ) {μ Δ Vm : ℝ} {N : ℕ} (_hΔ : 0 ≤ Δ) (_hVm : 0 ≤ Vm)
    (hS : ∀ k < N, S k ≤ μ + Δ) (hV : ∀ k < N, |V k| ≤ Vm) :
    mS S V μ N ≤ N * Real.exp Δ * Vm := by
  unfold mS
  calc ∑ k ∈ range N, Real.exp (S k - μ) * |V k| ≤ ∑ k ∈ range N, Real.exp Δ * Vm := by
        refine Finset.sum_le_sum fun k hk => ?_
        have hk' := Finset.mem_range.mp hk
        exact mul_le_mul (Real.exp_le_exp.mpr (by linarith [hS k hk'])) (hV k hk')
          (abs_nonneg _) (Real.exp_pos _).le
    _ = N * Real.exp Δ * Vm := by simp [mul_assoc]

/-- **The invariant bounds the state**: `|x| ≤ (1 + Fc)·mS + G·t·Ex`. -/
theorem colInv_abs_le {e1 γ ε2 K G : ℝ} {S V : ℕ → ℝ} {N t : ℕ} {μ x : ℝ}
    (h : ColInv e1 γ ε2 K G S V N t μ x) :
    |x| ≤ (1 + Fc e1 γ ε2 K t μ) * mS S V μ N + G * t * Ex e1 γ K t μ := by
  have h1 := abs_aS_le S V μ N
  have h2 : |x| ≤ |aS S V μ N| + |x - aS S V μ N| := by
    have := abs_add_le (aS S V μ N) (x - aS S V μ N)
    rwa [add_sub_cancel] at this
  unfold ColInv at h
  nlinarith

/-- The exponent of `Ex` is small on the covered range. -/
theorem Ex_le {X e1 μ : ℝ} {t : ℕ} (_hX0 : 0 ≤ X) (hX1 : X ≤ 1024) (he1 : 0 ≤ e1) (he1' : e1 ≤ eA)
    (ht : t ≤ 2 ^ 16) (hμ : μ ≤ Kx X) (_hμ' : -(Kx X) ≤ μ) :
    Ex e1 gam (Kx X) t μ ≤ 21 / 20 := by
  unfold Ex
  have hu := u_pos
  have h1 : (t : ℝ) * e1 ≤ 2 ^ 16 * eA := by
    have : (t : ℝ) ≤ 2 ^ 16 := by exact_mod_cast ht
    exact mul_le_mul this he1' he1 (by positivity)
  have h2 : gam * (μ + Kx X) ≤ gam * 2050 := by
    refine mul_le_mul_of_nonneg_left ?_ (by unfold gam; positivity)
    unfold Kx at *; linarith
  have h3 : (t : ℝ) * e1 + gam * (μ + Kx X) ≤ 4 / 100 := by
    unfold eA gam u at *; norm_num at *; linarith
  calc Real.exp ((t : ℝ) * e1 + gam * (μ + Kx X)) ≤ 1 / (1 - ((t : ℝ) * e1 + gam * (μ + Kx X))) :=
        exp_le_inv (by linarith)
    _ ≤ 1 / (1 - 4 / 100) := one_div_le_one_div_of_le (by norm_num) (by linarith)
    _ ≤ 21 / 20 := by norm_num

/-- The error coefficient is small on the covered range. -/
theorem Fc_le {X e1 ε2 μ : ℝ} {t : ℕ} (hX0 : 0 ≤ X) (hX1 : X ≤ 1024) (he1 : 0 ≤ e1)
    (he1' : e1 ≤ eA) (hε2 : 0 ≤ ε2) (hε2' : ε2 ≤ 48 * u * X + 40 * u)
    (ht : t ≤ 2 ^ 16) (hμ : μ ≤ Kx X) (hμ' : -(Kx X) ≤ μ) :
    Fc e1 gam ε2 (Kx X) t μ ≤ 1 / 16 := by
  have hE := Ex_le hX0 hX1 he1 he1' ht hμ hμ'
  have hε : ε2 ≤ 1 / 100 := by
    have : 48 * u * X ≤ 48 * u * 1024 := by have := u_pos; nlinarith
    unfold u at *; norm_num at *; linarith
  unfold Fc
  have := Real.exp_pos ((t : ℝ) * e1 + gam * (μ + Kx X))
  unfold Ex at hE ⊢
  nlinarith

/-! ## One trip of a column: the exact sums -/

/-- The block value `b` is within `ρb·Z + ηb` of the exact block `Y`, where `Y` and `Z` use the
exact weights (the kit's `block_err`, `block_abs_le`, `approx_trans`). -/
theorem block_approx {X : ℝ} (S V : ℕ → ℝ) {c N : ℕ} {μ' b ρs Vm : ℝ} (ph : ℕ → ℝ)
    (hρs : 0 ≤ ρs) (hc : c ≤ 256) (hVm : 0 ≤ Vm) (hVm' : Vm ≤ 2 ^ 15)
    (hV : ∀ j < c, |V (N + j)| ≤ Vm)
    (hp : ∀ j < c, |ph j - Real.exp (S (N + j) - μ')| ≤
      rhoP X * Real.exp (S (N + j) - μ') + etaP)
    (hb : |b - ∑ j ∈ range c, ph j * V (N + j)| ≤ ρs * ∑ j ∈ range c, |ph j * V (N + j)| + etaS) :
    |b - ∑ j ∈ range c, Real.exp (S (N + j) - μ') * V (N + j)| ≤
      (rhoP X + ρs * (1 + rhoP X)) * ∑ j ∈ range c, Real.exp (S (N + j) - μ') * |V (N + j)| +
        ((1 + ρs) * (etaP * 2 ^ 23) + etaS) := by
  have hp' : ∀ j ∈ range c, |ph j - Real.exp (S (N + j) - μ')| ≤
      rhoP X * Real.exp (S (N + j) - μ') + etaP := fun j hj => hp j (Finset.mem_range.mp hj)
  have hp0 : ∀ j ∈ range c, 0 ≤ Real.exp (S (N + j) - μ') := fun j _ => (Real.exp_pos _).le
  have h1 := TVA.Softmax.block_err (v := fun j => V (N + j)) hp'
  have h2 := TVA.Softmax.block_abs_le (v := fun j => V (N + j)) hp0 hp'
  have hsV : ∑ j ∈ range c, |V (N + j)| ≤ 2 ^ 23 := by
    calc ∑ j ∈ range c, |V (N + j)| ≤ ∑ j ∈ range c, Vm :=
          Finset.sum_le_sum fun j hj => hV j (Finset.mem_range.mp hj)
      _ = c * Vm := by simp
      _ ≤ 256 * 2 ^ 15 := by
          have : (c : ℝ) ≤ 256 := by exact_mod_cast hc
          exact mul_le_mul this hVm' hVm (by norm_num)
      _ = 2 ^ 23 := by norm_num
  have hη0 : 0 ≤ etaP := by unfold etaP; positivity
  have h3 := TVA.Softmax.approx_trans (ρ' := ρs) hρs h1 h2 hb
  have h4 : etaP * ∑ j ∈ range c, |V (N + j)| ≤ etaP * 2 ^ 23 :=
    mul_le_mul_of_nonneg_left hsV hη0
  have h5 : (1 + ρs) * (etaP * ∑ j ∈ range c, |V (N + j)|) ≤ (1 + ρs) * (etaP * 2 ^ 23) :=
    mul_le_mul_of_nonneg_left h4 (by linarith)
  linarith

set_option maxHeartbeats 1000000 in
/-- **One trip of a column** (second and later trips). The tokens `N, …, N + c − 1` are the trip's
valid lanes (`c ≤ 256`); lane `j` has the computed weight `ph j` within
`rhoP X·exp (S (N+j) − μ') + etaP` of `exp (S (N+j) − μ')`, `α'` is the computed rescaling,
`b` the computed block value (within `ρs·Σ|ph·V| + etaS` of `Σ ph·V`), `x'` the computed new
state. Rates `ρo`, `ρs`, `e1`, `ε2` satisfy the three compatibility conditions of `gen_step`. -/
theorem col_step {X : ℝ} (hX0 : 0 ≤ X) (hX1 : X ≤ 1024) (S V : ℕ → ℝ) {t N c : ℕ}
    {μ μ' α α' x x' b ρo ρs e1 ε2 Vm : ℝ} (ph : ℕ → ℝ)
    (_ht : 1 ≤ t) (ht2 : t ≤ 2 ^ 16) (hc : c ≤ 256) (hNc : N + c ≤ 2 ^ 22)
    (hρo : 0 ≤ ρo) (hρs : 0 ≤ ρs) (he1 : 0 ≤ e1) (hVm : 0 ≤ Vm) (hVm' : Vm ≤ 2 ^ 15)
    (hr1 : aA0 * (1 + ρo) + ρo ≤ e1) (hr2 : aA1 * (1 + ρo) ≤ gam)
    (hr3 : rhoP X + ρs * (1 + rhoP X) + ρo * (1 + (rhoP X + ρs * (1 + rhoP X))) ≤ ε2)
    (hε2 : ε2 ≤ 48 * u * X + 40 * u)
    (he1' : e1 ≤ eA)
    (hμ : -(Kx X) ≤ μ) (hμμ : μ ≤ μ') (hμ'K : μ' ≤ Kx X)
    (hSold : ∀ k < N, S k ≤ μ + dS X) (hV : ∀ k < N + c, |V k| ≤ Vm)
    (hα : α = Real.exp (μ - μ')) (hα' : |α' - α| ≤ (aA0 + aA1 * (μ' - μ)) * α + zeta0)
    (hp : ∀ j < c, |ph j - Real.exp (S (N + j) - μ')| ≤
      rhoP X * Real.exp (S (N + j) - μ') + etaP)
    (hb : |b - ∑ j ∈ range c, ph j * V (N + j)| ≤ ρs * ∑ j ∈ range c, |ph j * V (N + j)| + etaS)
    (hx : ColInv e1 gam ε2 (Kx X) Gabs S V N t μ x)
    (ho : |x' - (α' * x + b)| ≤ ρo * (|α' * x| + |b|) + etaO) :
    ColInv e1 gam ε2 (Kx X) Gabs S V (N + c) (t + 1) μ' x' := by
  have hu := u_pos
  have hrP : 0 ≤ rhoP X := by unfold rhoP; positivity
  have hρb : 0 ≤ rhoP X + ρs * (1 + rhoP X) := by positivity
  have hε2' : 0 ≤ ε2 := by
    have : 0 ≤ ρo * (1 + (rhoP X + ρs * (1 + rhoP X))) := by positivity
    linarith
  have hα0 : 0 ≤ α := by rw [hα]; exact (Real.exp_pos _).le
  have hα1 : α ≤ 1 := by rw [hα]; exact Real.exp_le_one_iff.mpr (by linarith)
  have hρoe : ρo ≤ eA := by
    have : 0 ≤ aA0 * (1 + ρo) := mul_nonneg (by unfold aA0; positivity) (by linarith)
    linarith
  have hKX : -(Kx X) ≤ μ' := by linarith
  -- magnitude of the old state
  have hdS : dS X ≤ 1 / 2 := by unfold dS u; norm_num; nlinarith
  have hdS0 : 0 ≤ dS X := by unfold dS; positivity
  have hexpdS : Real.exp (dS X) ≤ 2 := by
    refine (exp_le_inv (by linarith)).trans ?_
    rw [div_le_iff₀ (by linarith)]; linarith
  have hM : mS S V μ N ≤ 2 ^ 38 := by
    have := mS_le S V (Vm := Vm) (N := N) hdS0 hVm hSold (fun k hk => hV k (by omega))
    have hN : (N : ℝ) ≤ 2 ^ 22 := by exact_mod_cast (show N ≤ 2 ^ 22 by omega)
    calc mS S V μ N ≤ N * Real.exp (dS X) * Vm := this
      _ ≤ 2 ^ 22 * 2 * 2 ^ 15 := by
          apply mul_le_mul (mul_le_mul hN hexpdS (Real.exp_pos _).le (by positivity)) hVm' hVm
            (by positivity)
      _ = 2 ^ 38 := by norm_num
  have hEx := Ex_le hX0 hX1 he1 he1' ht2 (by linarith : μ ≤ Kx X) hμ
  have hFc := Fc_le hX0 hX1 he1 he1' hε2' hε2 ht2 (by linarith : μ ≤ Kx X) hμ
  have hEx1 := Ex_ge_one he1 (show 0 ≤ gam by unfold gam; positivity) t hμ
  have hGt : Gabs * t * Ex e1 gam (Kx X) t μ ≤ 1 := by
    have : (t : ℝ) ≤ 2 ^ 16 := by exact_mod_cast ht2
    have hG : Gabs * t ≤ Gabs * 2 ^ 16 := mul_le_mul_of_nonneg_left this (by unfold Gabs; positivity)
    have : Gabs * 2 ^ 16 ≤ 1 / 2 ^ 60 := by unfold Gabs; norm_num
    calc Gabs * t * Ex e1 gam (Kx X) t μ ≤ (1 / 2 ^ 60) * (21 / 20) :=
          mul_le_mul (hG.trans this) hEx (by linarith) (by positivity)
      _ ≤ 1 := by norm_num
  have hBx : |x| ≤ 2 ^ 39 := by
    have h1 := colInv_abs_le hx
    have h2 : (1 + Fc e1 gam ε2 (Kx X) t μ) * mS S V μ N ≤ (17 / 16) * 2 ^ 38 :=
      mul_le_mul (by linarith) hM (by unfold mS; positivity) (by norm_num)
    linarith
  have hY := block_approx (X := X) S V (N := N) ph hρs hc hVm hVm' (fun j hj => hV _ (by omega)) hp hb
  have hYZ : |∑ j ∈ range c, Real.exp (S (N + j) - μ') * V (N + j)| ≤
      ∑ j ∈ range c, Real.exp (S (N + j) - μ') * |V (N + j)| := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq (Finset.sum_congr rfl fun k _ => ?_))
    rw [abs_mul, abs_of_pos (Real.exp_pos _)]
  have hηb : 0 ≤ (1 + ρs) * (etaP * 2 ^ 23) + etaS := by unfold etaP etaS; positivity
  have hρoe' : ρo ≤ 1 / 1000000 := by unfold eA u at hρoe; norm_num at hρoe; linarith
  have hρs' : ρs ≤ 1 / 100 := by
    have : 0 ≤ ρo * (1 + (rhoP X + ρs * (1 + rhoP X))) := by positivity
    have h48 : 48 * u * X ≤ 48 * u * 1024 := by nlinarith
    have : rhoP X + ρs * (1 + rhoP X) ≤ ε2 := by linarith
    have h0 : ρs ≤ rhoP X + ρs * (1 + rhoP X) := by nlinarith
    unfold u at *; norm_num at *; linarith
  have hη : (1 + ρo) * (zeta0 * 2 ^ 39) + (1 + ρo) * ((1 + ρs) * (etaP * 2 ^ 23) + etaS) + etaO ≤ Gabs := by
    have h1 : (1 + ρo) ≤ 2 := by linarith
    have h2 : (1 + ρs) ≤ 2 := by linarith
    have p85 : 0 < (1 / 2 ^ 85 : ℝ) := by positivity
    have hA : zeta0 * 2 ^ 39 ≤ (1 / 2 ^ 85 : ℝ) := by unfold zeta0; norm_num
    have hB : (1 + ρs) * (etaP * 2 ^ 23) + etaS ≤ (1 / 2 ^ 85 : ℝ) * 3 := by
      have e : etaP * 2 ^ 23 ≤ (1 / 2 ^ 86 : ℝ) := by unfold etaP; norm_num
      have e' : etaS ≤ (1 / 2 ^ 86 : ℝ) := by unfold etaS; norm_num
      have : (1 + ρs) * (etaP * 2 ^ 23) ≤ 2 * (1 / 2 ^ 86 : ℝ) :=
        mul_le_mul h2 e (by unfold etaP; positivity) (by norm_num)
      have h86 : (1 / 2 ^ 86 : ℝ) * 2 = 1 / 2 ^ 85 := by norm_num
      linarith
    have hO : etaO ≤ (1 / 2 ^ 85 : ℝ) := by unfold etaO; norm_num
    have hG : (1 / 2 ^ 85 : ℝ) * 10 ≤ Gabs := by unfold Gabs; norm_num
    have c1 : (1 + ρo) * (zeta0 * 2 ^ 39) ≤ 2 * (1 / 2 ^ 85 : ℝ) :=
      mul_le_mul h1 hA (by unfold zeta0; positivity) (by norm_num)
    have c2 : (1 + ρo) * ((1 + ρs) * (etaP * 2 ^ 23) + etaS) ≤ 2 * ((1 / 2 ^ 85 : ℝ) * 3) :=
      mul_le_mul h1 hB (by unfold etaP etaS; positivity) (by norm_num)
    linarith
  have hgs := gen_step (e1 := e1) (γ := gam) (ε2 := ε2) (K := Kx X) (G := Gabs) (a0 := aA0)
    (a1 := aA1) (ζ := zeta0) (ρo := ρo) (ρb := rhoP X + ρs * (1 + rhoP X))
    (ηb := (1 + ρs) * (etaP * 2 ^ 23) + etaS) (ηo := etaO) (t := t) (μ := μ) (μ' := μ') (α := α)
    (α' := α') (X := aS S V μ N) (M := mS S V μ N)
    (Y := ∑ j ∈ range c, Real.exp (S (N + j) - μ') * V (N + j))
    (Z := ∑ j ∈ range c, Real.exp (S (N + j) - μ') * |V (N + j)|) (x := x) (x' := x') (b := b)
    (Bx := 2 ^ 39) he1 (by unfold gam; positivity) hε2' (by unfold Gabs; positivity)
    (by unfold aA0; positivity) (by unfold aA1; positivity) (by unfold zeta0; positivity) hρo
    hμ hμμ hα0 hα1 hα' (abs_aS_le S V μ N) hYZ hx hBx hY ho hr1 hr2
    (by linarith) hη
  have hA := aS_add S V μ μ' N c
  have hM' := aS_add S (fun k => |V k|) μ μ' N c
  unfold ColInv
  rw [hA, show mS S V μ' (N + c) = Real.exp (μ - μ') * mS S V μ N +
      ∑ j ∈ range c, Real.exp (S (N + j) - μ') * |V (N + j)| from hM', ← hα]
  exact hgs

/-- **The first trip of a column**: no earlier tokens; `x'` is within `ρo·|b| + etaO` of the
block value `b`. -/
theorem col_first {X : ℝ} (hX0 : 0 ≤ X) (_hX1 : X ≤ 1024) (S V : ℕ → ℝ) {c : ℕ}
    {μ' x' b ρo ρs e1 ε2 Vm : ℝ} (ph : ℕ → ℝ)
    (hc : c ≤ 256)
    (hρo : 0 ≤ ρo) (hρs : 0 ≤ ρs) (he1 : 0 ≤ e1) (hVm : 0 ≤ Vm) (hVm' : Vm ≤ 2 ^ 15)
    (hρo' : ρo ≤ 1 / 1000000) (hρs' : ρs ≤ 1 / 100)
    (hr3 : rhoP X + ρs * (1 + rhoP X) + ρo * (1 + (rhoP X + ρs * (1 + rhoP X))) ≤ ε2)
    (hμ' : -(Kx X) ≤ μ') (hV : ∀ k < c, |V k| ≤ Vm)
    (hp : ∀ j < c, |ph j - Real.exp (S j - μ')| ≤ rhoP X * Real.exp (S j - μ') + etaP)
    (hb : |b - ∑ j ∈ range c, ph j * V j| ≤ ρs * ∑ j ∈ range c, |ph j * V j| + etaS)
    (ho : |x' - b| ≤ ρo * |b| + etaO) :
    ColInv e1 gam ε2 (Kx X) Gabs S V c 1 μ' x' := by
  have hu := u_pos
  have hrP : 0 ≤ rhoP X := by unfold rhoP; positivity
  have hρb : 0 ≤ rhoP X + ρs * (1 + rhoP X) := by positivity
  have hε2' : 0 ≤ ε2 := by
    have : 0 ≤ ρo * (1 + (rhoP X + ρs * (1 + rhoP X))) := by positivity
    linarith
  have hY := block_approx (X := X) S V (N := 0) ph hρs hc hVm hVm' (fun j hj => by simpa using hV j hj) 
    (fun j hj => by simpa using hp j hj) (by simpa using hb)
  simp only [Nat.zero_add] at hY
  have hYZ : |∑ j ∈ range c, Real.exp (S j - μ') * V j| ≤
      ∑ j ∈ range c, Real.exp (S j - μ') * |V j| := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq (Finset.sum_congr rfl fun k _ => ?_))
    rw [abs_mul, abs_of_pos (Real.exp_pos _)]
  have hηb : 0 ≤ (1 + ρs) * (etaP * 2 ^ 23) + etaS := by unfold etaP etaS; positivity
  have hη : (1 + ρo) * ((1 + ρs) * (etaP * 2 ^ 23) + etaS) + etaO ≤ Gabs := by
    have h1 : (1 + ρo) ≤ 2 := by linarith
    have h2 : (1 + ρs) ≤ 2 := by linarith
    have p85 : 0 < (1 / 2 ^ 85 : ℝ) := by positivity
    have hB : (1 + ρs) * (etaP * 2 ^ 23) + etaS ≤ (1 / 2 ^ 85 : ℝ) * 3 := by
      have e : etaP * 2 ^ 23 ≤ (1 / 2 ^ 86 : ℝ) := by unfold etaP; norm_num
      have e' : etaS ≤ (1 / 2 ^ 86 : ℝ) := by unfold etaS; norm_num
      have : (1 + ρs) * (etaP * 2 ^ 23) ≤ 2 * (1 / 2 ^ 86 : ℝ) :=
        mul_le_mul h2 e (by unfold etaP; positivity) (by norm_num)
      have h86 : (1 / 2 ^ 86 : ℝ) * 2 = 1 / 2 ^ 85 := by norm_num
      linarith
    have hO : etaO ≤ (1 / 2 ^ 85 : ℝ) := by unfold etaO; norm_num
    have hG : (1 / 2 ^ 85 : ℝ) * 10 ≤ Gabs := by unfold Gabs; norm_num
    have c2 : (1 + ρo) * ((1 + ρs) * (etaP * 2 ^ 23) + etaS) ≤ 2 * ((1 / 2 ^ 85 : ℝ) * 3) :=
      mul_le_mul h1 hB (by unfold etaP etaS; positivity) (by norm_num)
    linarith
  have hgf := gen_first (e1 := e1) (γ := gam) (ε2 := ε2) (K := Kx X) (G := Gabs) (ρo := ρo)
    (ρb := rhoP X + ρs * (1 + rhoP X)) (ηb := (1 + ρs) * (etaP * 2 ^ 23) + etaS) (ηo := etaO)
    (μ' := μ') (Y := ∑ j ∈ range c, Real.exp (S j - μ') * V j)
    (Z := ∑ j ∈ range c, Real.exp (S j - μ') * |V j|) (x' := x') (b := b) he1
    (by unfold gam; positivity) hε2' (by unfold Gabs; positivity) hρo
    hμ' hYZ hY ho hr3 hη
  unfold ColInv aS mS
  exact hgf

/-- The invariant at the start: zero tokens, state `0`. -/
theorem colInv_zero (e1 γ ε2 K G : ℝ) (S V : ℕ → ℝ) (μ : ℝ) :
    ColInv e1 γ ε2 K G S V 0 0 μ 0 := by
  simp [ColInv, aS, mS]

end Num

end Claims.PagedAttention
