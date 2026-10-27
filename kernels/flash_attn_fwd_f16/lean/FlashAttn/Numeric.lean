import TVA.Rules.Num
import TVA.Lemmas.Fused
import TVA.Lemmas.Sum
import TVA.Lemmas.Softmax

/-!
# Flash attention forward: the numeric part

The error constants of the specification (`Err`) and, per row of a tile, the online-softmax
invariant `RowInv` with its step `row_step` and the epilogue `row_out`. Everything here is about
float values and reals; the program proof (`TVAExamples.FlashAttn`) feeds it the relations each
operation's `Safe` rule gives.

Notation, for one query row: `s j` is the exact score `q · k_j` (fp16 values, exact products),
`c` the device's `float32` scale `sm_scale ⊗ fl(1.44269504)`, `z j = c · s j`, and after `n`
blocks with running maximum `μ`:

* `Λ n μ = Σ_{j < n·BN} 2^(z j − μ)`,
* `A n μ e = Σ_{j < n·BN} 2^(z j − μ) · v j e`.
-/
namespace TVAExamples.FlashAttn

open TVA TVA.Math.Float TVA.Softmax

/-! ## Constants -/

/-- The sizes and bounds the error constants depend on: `N = N_CTX`, `HD = HEAD_DIM`,
`BN = BLOCK_N`; `|q|, |k| ≤ R`, `|v| ≤ V`, the scale `c ≤ C`. -/
structure Bnd where
  N : Nat
  HD : Nat
  BN : Nat
  R : ℝ
  V : ℝ
  C : ℝ

namespace Err

/-- `float32`: unit roundoff and the smallest subnormal step's half. -/
noncomputable def u : ℝ := 2 ^ (-24 : ℤ)
noncomputable def η : ℝ := 2 ^ (-150 : ℤ)
/-- `float16`. -/
noncomputable def u16 : ℝ := 2 ^ (-11 : ℤ)
noncomputable def η16 : ℝ := 2 ^ (-25 : ℤ)
/-- The `exp2` constants assumed: `2^-22` relative, `2^-149` absolute (library issue L4). -/
noncomputable def relE : ℝ := 2 ^ (-22 : ℤ)
noncomputable def absE : ℝ := 2 ^ (-149 : ℤ)
/-- The `log2` constants assumed: `max(1, |log₂ x|)·2^-22` (library issue L4). -/
noncomputable def relL : ℝ := 2 ^ (-22 : ℤ)
noncomputable def absL : ℝ := 2 ^ (-22 : ℤ)
/-- `div.full.f32` (`Rules.Num.DivFull`). -/
noncomputable def relDiv : ℝ := 2 ^ (-22 : ℤ)
noncomputable def absDiv : ℝ := 2 ^ (-148 : ℤ)

/-- The `tl.dot` constants assumed for `float16` operands into `float32` (`wgmma`): a `K`-term
dot is within `relDot K · Σ|a·b| + absDot K` of the exact sum. -/
def relDot (K : Nat) : ℚ := (K + 2) * 2 ^ (-22 : ℤ)
def absDot (K : Nat) : ℚ := (K + 2) * 2 ^ (-126 : ℤ)

variable (b : Bnd)

/-- `|s| ≤ S = HD·R²`. -/
noncomputable def S : ℝ := b.HD * b.R ^ 2
/-- The error of a score: `|qk − s| ≤ ε₁`. -/
noncomputable def ε₁ : ℝ := (relDot b.HD : ℝ) * S b + (absDot b.HD : ℝ)
/-- `|qk| ≤ Q`. -/
noncomputable def Q : ℝ := S b + ε₁ b
/-- `|μ| ≤ Mμ`: the scaled scores and the running maximum. -/
noncomputable def Mμ : ℝ := b.C * Q b * (1 + u) + η
/-- The offset between the running maximum and the scaled scores. -/
noncomputable def δ : ℝ := b.C * ε₁ b + u * (b.C * Q b) + η
/-- The exponent error of `p = exp2(qk·c − μ)`. -/
noncomputable def δx : ℝ := δ b + 6 * u * Mμ b + 4 * η
/-- The relative error of `p` (with absolute error `absE`). -/
noncomputable def κp : ℝ := δx b + relE * (1 + δx b)
/-- The depth term of a row sum of `BN` elements. -/
noncomputable def γ : ℝ := 2 * b.BN * u
/-- The block sum `l_ij`: relative `κs`, absolute `νs`. -/
noncomputable def κs : ℝ := κp b + γ b * (1 + κp b)
noncomputable def νs : ℝ := b.BN * absE * (1 + γ b) + 2 * b.BN * η
/-- The exponent error of `α = exp2(m_i − m_ij)` and its relative error. -/
noncomputable def δα : ℝ := 2 * u * Mμ b + η
noncomputable def κα : ℝ := δα b + relE * (1 + δα b)
/-- The largest running sum, `2·N`. -/
noncomputable def Λmax : ℝ := 2 * b.N
/-- The number of blocks. -/
def T : Nat := b.N / b.BN

/-- The running sum `l_i`: after `n` blocks it is within `(ρl0 + n·dl)·Λ` of `Λ`. -/
noncomputable def νl : ℝ := 3 * absE * Λmax b + 2 * νs b + 4 * η
noncomputable def ρl0 : ℝ := κs b + 6 * u + 2 * νl b
noncomputable def dl : ℝ := 2 * κα b + 12 * u + 2 * νl b
noncomputable def ρL : ℝ := ρl0 b + T b * dl b

/-- The accumulator: after `n` blocks within `(ρv0 + n·dv)·Λ` of `A`. -/
noncomputable def κ16 : ℝ := κp b + u16 * (1 + κp b)
noncomputable def ν16 : ℝ := absE * (1 + u16) + η16
noncomputable def θ : ℝ := (relDot b.BN : ℝ)
noncomputable def νa : ℝ := 3 * absE * b.V * Λmax b + 2 * η + 2 * b.V * b.BN * ν16 + (absDot b.BN : ℝ)
noncomputable def ρv0 : ℝ := b.V * (κ16 b + 2 * θ b) + 2 * νa b
noncomputable def dv : ℝ := b.V * (3 * κα b + 2 * u + 3 * θ b) + 2 * νa b
noncomputable def ρV : ℝ := ρv0 b + T b * dv b

/-- **The output bound.** `eO` is the error of `acc / l` against the exact softmax average; the
rest is the division (`div.full.f32`) and the conversion to `float16`. -/
noncomputable def eO : ℝ := (ρV b + b.V * ρL b) / (1 - ρL b)
noncomputable def errO : ℝ :=
  eO b + relDiv * (b.V + eO b) + absDiv + u16 * ((b.V + eO b) * (1 + relDiv) + absDiv) + η16

/-- **The log-sum-exp bound.** -/
noncomputable def eL : ℝ := 2 * ρL b / (1 - ρL b)
noncomputable def Lg : ℝ := Real.logb 2 (2 * b.N) + eL b
noncomputable def errM : ℝ := u * (Mμ b + (1 + relL) * Lg b + absL) + η + relL * Lg b + absL + eL b

end Err


/-! ## Format constants and float operations over the reals -/

namespace Err

theorem u_pos : 0 < u := by unfold u; positivity
theorem η_pos : 0 < η := by unfold η; positivity
theorem u16_pos : 0 < u16 := by unfold u16; positivity
theorem η16_pos : 0 < η16 := by unfold η16; positivity
theorem relE_pos : 0 < relE := by unfold relE; positivity
theorem absE_pos : 0 < absE := by unfold absE; positivity
theorem relL_pos : 0 < relL := by unfold relL; positivity
theorem absL_pos : 0 < absL := by unfold absL; positivity
theorem u_le : u ≤ 1 / 2 ^ 24 := by unfold u; norm_num
theorem η_le : η ≤ 1 / 2 ^ 150 := by unfold η; norm_num
theorem relE_le : relE ≤ 1 / 2 ^ 22 := by unfold relE; norm_num
theorem absE_le : absE ≤ 1 / 2 ^ 149 := by unfold absE; norm_num
theorem u16_le : u16 ≤ 1 / 2 ^ 11 := by unfold u16; norm_num

theorem relDot_nonneg (K : Nat) : 0 ≤ relDot K := by unfold relDot; positivity
theorem absDot_nonneg (K : Nat) : 0 ≤ absDot K := by unfold absDot; positivity

end Err

theorem u32_eq : ((Format.binary32.u : ℚ) : ℝ) = Err.u := by
  rw [show Format.binary32.u = 2 ^ (-24 : ℤ) by unfold Format.u; rfl]; unfold Err.u; push_cast; rfl
theorem eta32_eq : ((Format.binary32.eta : ℚ) : ℝ) = Err.η := by
  rw [show Format.binary32.eta = 2 ^ (-150 : ℤ) by
    unfold Format.eta; rw [show Format.binary32.emin - (Format.binary32.m : ℤ) - 1 = -150 by decide]]
  unfold Err.η; push_cast; rfl
theorem u16_eq : ((Format.binary16.u : ℚ) : ℝ) = Err.u16 := by
  rw [show Format.binary16.u = 2 ^ (-11 : ℤ) by unfold Format.u; rfl]; unfold Err.u16; push_cast; rfl
theorem eta16_eq : ((Format.binary16.eta : ℚ) : ℝ) = Err.η16 := by
  rw [show Format.binary16.eta = 2 ^ (-25 : ℤ) by
    unfold Format.eta; rw [show Format.binary16.emin - (Format.binary16.m : ℤ) - 1 = -25 by decide]]
  unfold Err.η16; push_cast; rfl
theorem maxVal32_ge : (2 : ℚ) ^ 127 ≤ Format.binary32.maxVal := by
  norm_num [Format.maxVal, Format.binary32, Format.maxSig, Format.emax, Format.m, Format.expMax, Format.bias]
theorem maxVal32_ge' : (2 : ℝ) ^ 127 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
  exact_mod_cast maxVal32_ge
theorem u32_le : Format.binary32.u ≤ 1 / 16 := by
  rw [show Format.binary32.u = 2 ^ (-24 : ℤ) by unfold Format.u; rfl]; norm_num

/-- A rational bound read over the reals. -/
theorem toReal_def {f : Format} (x : FloatV f) : x.toReal = (x.toRat : ℝ) := rfl

/-- `a ⊗ b` (IEEE, `tl.mul`) of finite `float32` values in range. -/
theorem mul_val {a b r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (h : FloatV.NaNEq (FloatV.mul a b) r) (hroom : |a.toReal * b.toReal| ≤ 2 ^ 126) :
    r.isFinite = true ∧ |r.toReal - a.toReal * b.toReal| ≤ |a.toReal * b.toReal| * Err.u + Err.η := by
  have hmax : |a.toRat * b.toRat| ≤ Format.binary32.maxVal := by
    have h1 : ((|a.toRat * b.toRat| : ℚ) : ℝ) ≤ 2 ^ 126 := by push_cast; exact hroom
    have h2 : (|a.toRat * b.toRat| : ℚ) ≤ 2 ^ 126 := by exact_mod_cast h1
    linarith [maxVal32_ge]
  obtain ⟨h1, h2⟩ := FloatV.mul_err Format.binary32_regular ha hb hmax
  rw [h.eq ((FloatV.isFinite_iff.mp h1).1)]
  refine ⟨h1, ?_⟩
  simp only [toReal_def, ← u32_eq, ← eta32_eq]
  exact_mod_cast h2

/-- `a ⊕ b` (IEEE, `tl.add`) of finite `float32` values in range. -/
theorem add_val {a b r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (h : FloatV.NaNEq (FloatV.add a b) r) (hroom : |a.toReal + b.toReal| ≤ 2 ^ 126) :
    r.isFinite = true ∧ |r.toReal - (a.toReal + b.toReal)| ≤ |a.toReal + b.toReal| * Err.u + Err.η := by
  have hmax : |a.toRat + b.toRat| ≤ Format.binary32.maxVal := by
    have h1 : ((|a.toRat + b.toRat| : ℚ) : ℝ) ≤ 2 ^ 126 := by push_cast; exact hroom
    have h2 : (|a.toRat + b.toRat| : ℚ) ≤ 2 ^ 126 := by exact_mod_cast h1
    linarith [maxVal32_ge]
  obtain ⟨h1, h2⟩ := FloatV.add_err Format.binary32_regular ha hb hmax
  rw [h.eq ((FloatV.isFinite_iff.mp h1).1)]
  refine ⟨h1, ?_⟩
  simp only [toReal_def, ← u32_eq, ← eta32_eq]
  exact_mod_cast h2

/-- `a ⊖ b` (IEEE, `tl.sub`) of finite `float32` values in range. -/
theorem sub_val {a b r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (h : FloatV.NaNEq (FloatV.sub a b) r) (hroom : |a.toReal - b.toReal| ≤ 2 ^ 126) :
    r.isFinite = true ∧ |r.toReal - (a.toReal - b.toReal)| ≤ |a.toReal - b.toReal| * Err.u + Err.η := by
  have hmax : |a.toRat - b.toRat| ≤ Format.binary32.maxVal := by
    have h1 : ((|a.toRat - b.toRat| : ℚ) : ℝ) ≤ 2 ^ 126 := by push_cast; exact hroom
    have h2 : (|a.toRat - b.toRat| : ℚ) ≤ 2 ^ 126 := by exact_mod_cast h1
    linarith [maxVal32_ge]
  obtain ⟨h1, h2⟩ := FloatV.sub_err Format.binary32_regular ha hb hmax
  rw [h.eq ((FloatV.isFinite_iff.mp h1).1)]
  refine ⟨h1, ?_⟩
  simp only [toReal_def, ← u32_eq, ← eta32_eq]
  exact_mod_cast h2

/-- A contractible `x + y` (`tl.subc`, `tl.addc`) of finite `float32` values in range. -/
theorem addRel_val {fusion : Bool} {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (h : AddRel fusion x y r) (hroom : |x.toReal| + |y.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal + y.toReal)| ≤ 3 * Err.u * (|x.toReal| + |y.toReal|) + 4 * Err.η := by
  have hroom' : 2 * (|x.toRat| + |y.toRat| + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
    have h1 : ((|x.toRat| + |y.toRat| : ℚ) : ℝ) ≤ 2 ^ 120 := by push_cast; exact hroom
    have h2 : (|x.toRat| + |y.toRat| : ℚ) ≤ 2 ^ 120 := by exact_mod_cast h1
    have h3 : Format.binary32.eta ≤ 1 := by
      rw [show Format.binary32.eta = 2 ^ (-150 : ℤ) by
        unfold Format.eta; rw [show Format.binary32.emin - (Format.binary32.m : ℤ) - 1 = -150 by decide]]
      norm_num
    linarith [maxVal32_ge]
  obtain ⟨h1, h2⟩ := FloatV.AddRel.err Format.binary32_regular rfl u32_le h hx hy hroom'
  refine ⟨h1, ?_⟩
  simp only [toReal_def, ← u32_eq, ← eta32_eq]
  exact_mod_cast h2

/-- `a·b + c` (`tl.mul_add`: two IEEE operations or one `fma`) of finite `float32` values. -/
theorem mulAdd_val {fusion : Bool} {a b c r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hc : c.isFinite = true)
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul a b) c) r ∨ (fusion = true ∧ FloatV.NaNEq (FloatV.fma a b c) r))
    (hroom : |a.toReal * b.toReal| + |c.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧ |r.toReal - (a.toReal * b.toReal + c.toReal)| ≤
      3 * Err.u * (|a.toReal * b.toReal| + |c.toReal|) + 4 * Err.η := by
  have hroom' : 2 * (|a.toRat * b.toRat| + |c.toRat| + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
    have h1 : ((|a.toRat * b.toRat| + |c.toRat| : ℚ) : ℝ) ≤ 2 ^ 120 := by push_cast; exact hroom
    have h2 : (|a.toRat * b.toRat| + |c.toRat| : ℚ) ≤ 2 ^ 120 := by exact_mod_cast h1
    have h3 : Format.binary32.eta ≤ 1 := by
      rw [show Format.binary32.eta = 2 ^ (-150 : ℤ) by
        unfold Format.eta; rw [show Format.binary32.emin - (Format.binary32.m : ℤ) - 1 = -150 by decide]]
      norm_num
    linarith [maxVal32_ge]
  obtain ⟨h1, h2⟩ := FloatV.mulAdd_err Format.binary32_regular u32_le h ha hb hc hroom'
  refine ⟨h1, ?_⟩
  simp only [toReal_def, ← u32_eq, ← eta32_eq]
  exact_mod_cast h2

/-- `x.to(tl.float16)` of a finite `float32` value with `|x| ≤ 65504`. -/
theorem cvt16_val {x : F32} {r : F16} (hx : x.isFinite = true)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r) (hroom : |x.toReal| ≤ 65504) :
    r.isFinite = true ∧ |r.toReal - x.toReal| ≤ |x.toReal| * Err.u16 + Err.η16 := by
  have hmax : |x.toRat| ≤ Format.binary16.maxVal := by
    rw [Format.binary16_maxVal]
    have h1 : ((|x.toRat| : ℚ) : ℝ) ≤ 65504 := by push_cast; exact hroom
    exact_mod_cast h1
  obtain ⟨h1, h2⟩ := FloatV.cvt_err (g := Format.binary16) Format.binary16_regular hx hmax
  rw [FloatV.cvtWith_rn] at h
  rw [h.eq ((FloatV.isFinite_iff.mp h1).1)]
  refine ⟨h1, ?_⟩
  simp only [toReal_def, ← u16_eq, ← eta16_eq]
  exact_mod_cast h2

/-- `List.range` sums are `Finset.range` sums. -/
theorem list_sum_range {β : Type} [AddCommMonoid β] (g : ℕ → β) :
    ∀ n, ((List.range n).map g).sum = ∑ i ∈ Finset.range n, g i
  | 0 => by simp
  | n + 1 => by rw [List.range_succ, List.map_append, List.sum_append, list_sum_range g n,
      Finset.sum_range_succ]; simp

/-- **The conditions on the sizes and bounds.** The configuration sizes of the tutorial, the
ranges that keep every `float32` intermediate far from overflow, and the smallness that makes
the bounds meaningful (`ρL ≤ 1/2`: the running sum within a factor `1 ± 1/2`). -/
structure Bnd.Ok (b : Bnd) : Prop where
  hN : 0 < b.N
  hN31 : b.N < 2 ^ 31
  hBN : 0 < b.BN
  hBN256 : b.BN ≤ 256
  hBNN : b.BN ∣ b.N
  hHD : 0 < b.HD
  hHD256 : b.HD ≤ 256
  hR : 0 ≤ b.R
  hR16 : b.R ≤ 65504
  hV : 0 < b.V
  hV13 : b.V ≤ 2 ^ 13
  hC : 0 < b.C
  hCQ : b.C * Err.Q b ≤ 2 ^ 60
  hδx : Err.δx b ≤ 1 / 4
  hκα : Err.κα b ≤ 1 / 4
  hρL : Err.ρL b ≤ 1 / 2
  hρV : Err.ρV b ≤ b.V

namespace Bnd.Ok

variable {b : Bnd} (hb : b.Ok)
include hb

theorem S_nonneg : 0 ≤ Err.S b := by
  unfold Err.S; have := hb.hR; positivity
theorem ε₁_nonneg : 0 ≤ Err.ε₁ b := by
  unfold Err.ε₁
  have := hb.S_nonneg; have := Err.relDot_nonneg b.HD; have := Err.absDot_nonneg b.HD
  have : (0 : ℝ) ≤ (Err.relDot b.HD : ℝ) := by exact_mod_cast Err.relDot_nonneg b.HD
  have : (0 : ℝ) ≤ (Err.absDot b.HD : ℝ) := by exact_mod_cast Err.absDot_nonneg b.HD
  positivity
theorem Q_nonneg : 0 ≤ Err.Q b := by unfold Err.Q; linarith [hb.S_nonneg, hb.ε₁_nonneg]
theorem CQ_nonneg : 0 ≤ b.C * Err.Q b := mul_nonneg hb.hC.le hb.Q_nonneg
theorem Mμ_nonneg : 0 ≤ Err.Mμ b := by
  unfold Err.Mμ; have := hb.CQ_nonneg; have := Err.u_pos; have := Err.η_pos; positivity
theorem δ_nonneg : 0 ≤ Err.δ b := by
  unfold Err.δ
  have := hb.ε₁_nonneg; have := hb.CQ_nonneg; have := Err.u_pos; have := Err.η_pos; have := hb.hC
  positivity
theorem δ_le : Err.δ b ≤ Err.δx b := by
  unfold Err.δx; have := hb.Mμ_nonneg; have := Err.u_pos; have := Err.η_pos; nlinarith
theorem δx_nonneg : 0 ≤ Err.δx b := le_trans hb.δ_nonneg hb.δ_le
theorem δ_le4 : Err.δ b ≤ 1 / 4 := le_trans hb.δ_le hb.hδx
theorem κp_nonneg : 0 ≤ Err.κp b := by
  unfold Err.κp; have := hb.δx_nonneg; have := Err.relE_pos; positivity
theorem κp_le : Err.κp b ≤ 1 / 2 := by
  unfold Err.κp; have := hb.hδx; have := hb.δx_nonneg; have := Err.relE_le; nlinarith
omit hb in
theorem γ_nonneg : 0 ≤ Err.γ b := by unfold Err.γ; have := Err.u_pos; positivity
theorem γ_le : Err.γ b ≤ 1 / 2 ^ 15 := by
  unfold Err.γ
  have h1 : (b.BN : ℝ) ≤ 256 := by exact_mod_cast hb.hBN256
  have := Err.u_le; have := Err.u_pos
  nlinarith
theorem κs_nonneg : 0 ≤ Err.κs b := by
  unfold Err.κs; have := hb.κp_nonneg; have := (Bnd.Ok.γ_nonneg (b := b)); positivity
theorem κs_le : Err.κs b ≤ 1 := by
  unfold Err.κs; have := hb.κp_le; have := hb.κp_nonneg; have := hb.γ_le; have := (Bnd.Ok.γ_nonneg (b := b)); nlinarith
omit hb in
theorem νs_nonneg : 0 ≤ Err.νs b := by
  unfold Err.νs; have := (Bnd.Ok.γ_nonneg (b := b)); have := Err.absE_pos; have := Err.η_pos; positivity
theorem δα_nonneg : 0 ≤ Err.δα b := by
  unfold Err.δα; have := hb.Mμ_nonneg; have := Err.u_pos; have := Err.η_pos; positivity
theorem δα_le : Err.δα b ≤ Err.δx b := by
  unfold Err.δα Err.δx; have := hb.Mμ_nonneg; have := Err.u_pos; have := Err.η_pos
  have := hb.δ_nonneg; nlinarith
theorem κα_nonneg : 0 ≤ Err.κα b := by
  unfold Err.κα; have := hb.δα_nonneg; have := Err.relE_pos; positivity
omit hb in
theorem θ_nonneg : 0 ≤ Err.θ b := by
  unfold Err.θ; exact_mod_cast Err.relDot_nonneg b.BN
theorem θ_le : Err.θ b ≤ 1 / 16 := by
  unfold Err.θ Err.relDot
  have h1 : (b.BN : ℝ) ≤ 256 := by exact_mod_cast hb.hBN256
  push_cast
  have : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
  rw [this]; nlinarith
theorem κ16_nonneg : 0 ≤ Err.κ16 b := by
  unfold Err.κ16; have := hb.κp_nonneg; have := Err.u16_pos; positivity
theorem κ16_le : Err.κ16 b ≤ 1 := by
  unfold Err.κ16; have := hb.κp_le; have := hb.κp_nonneg; have := Err.u16_le; nlinarith
theorem Λmax_pos : 0 < Err.Λmax b := by
  unfold Err.Λmax; have : (0 : ℝ) < b.N := by exact_mod_cast hb.hN
  linarith
theorem νl_nonneg : 0 ≤ Err.νl b := by
  unfold Err.νl; have := (Bnd.Ok.νs_nonneg (b := b)); have := hb.Λmax_pos; have := Err.absE_pos; have := Err.η_pos
  positivity
theorem νa_nonneg : 0 ≤ Err.νa b := by
  unfold Err.νa Err.ν16
  have := hb.Λmax_pos; have := Err.absE_pos; have := Err.η_pos; have := hb.hV; have := Err.u16_pos
  have := Err.η16_pos
  have : (0 : ℝ) ≤ (Err.absDot b.BN : ℝ) := by exact_mod_cast Err.absDot_nonneg b.BN
  positivity
theorem dl_nonneg : 0 ≤ Err.dl b := by
  unfold Err.dl; have := hb.κα_nonneg; have := Err.u_pos; have := hb.νl_nonneg; positivity
theorem ρl0_nonneg : 0 ≤ Err.ρl0 b := by
  unfold Err.ρl0; have := hb.κs_nonneg; have := Err.u_pos; have := hb.νl_nonneg; positivity
theorem dv_nonneg : 0 ≤ Err.dv b := by
  unfold Err.dv; have := hb.κα_nonneg; have := Err.u_pos; have := hb.νa_nonneg; have := (Bnd.Ok.θ_nonneg (b := b))
  have := hb.hV; positivity
theorem ρv0_nonneg : 0 ≤ Err.ρv0 b := by
  unfold Err.ρv0; have := hb.κ16_nonneg; have := (Bnd.Ok.θ_nonneg (b := b)); have := hb.νa_nonneg; have := hb.hV
  positivity

theorem Mμ_le : Err.Mμ b ≤ 2 ^ 62 := by
  unfold Err.Mμ
  have h1 : b.C * Err.Q b * (1 + Err.u) ≤ 2 ^ 60 * 2 :=
    mul_le_mul hb.hCQ (by have := Err.u_le; norm_num at this ⊢; linarith)
      (by have := Err.u_pos; linarith) (by norm_num)
  have := Err.η_le
  norm_num at h1 this ⊢; linarith

/-- After `n ≤ T` blocks the running sum's relative error is at most `ρL`. -/
theorem ρl_le {n : ℕ} (hn : n ≤ Err.T b) : Err.ρl0 b + n * Err.dl b ≤ 1 / 2 := by
  have : (n : ℝ) ≤ Err.T b := by exact_mod_cast hn
  have := hb.hρL; unfold Err.ρL at this
  nlinarith [hb.dl_nonneg]

theorem ρv_le {n : ℕ} (hn : n ≤ Err.T b) : Err.ρv0 b + n * Err.dv b ≤ b.V := by
  have : (n : ℝ) ≤ Err.T b := by exact_mod_cast hn
  have := hb.hρV; unfold Err.ρV at this
  nlinarith [hb.dv_nonneg]

end Bnd.Ok

/-! ## Row maxima and sums -/

/-- **`tl.max(x, 1)` on one row** of finite `float32` values, by value. -/
theorem maxRow_val {N : ℕ} {x : Fin N → F32} {r : F32}
    (h : ∃ t : Tree, t.Covers N ∧
      t.Eval (FloatV.MaxRel true) (fun j => if hj : j < N then x ⟨j, hj⟩ else default) r)
    (hx : ∀ k, (x k).isFinite = true) :
    r.isFinite = true ∧ (∃ k, r.toReal = (x k).toReal) ∧ ∀ k, (x k).toReal ≤ r.toReal := by
  obtain ⟨t, hcov, hev⟩ := h
  have hleaf : ∀ i ∈ t.leaves, i < N := fun i hi => List.mem_range.mp (hcov.subset hi)
  have hmem : ∀ k : Fin N, k.1 ∈ t.leaves := fun k => hcov.symm.subset (List.mem_range.mpr k.2)
  obtain ⟨h1, ⟨i, hi, hv⟩, h3⟩ := maxTree_val hev (fun i hi => by
    simp only [dite_eq_left (hleaf i hi)]; exact hx _)
  refine ⟨h1, ⟨⟨i, hleaf i hi⟩, ?_⟩, fun k => ?_⟩
  · rw [toReal_def, toReal_def, hv]; simp only [dite_eq_left (hleaf i hi)]
  · have := h3 k.1 (hmem k)
    simp only [dite_eq_left k.2] at this
    rw [toReal_def, toReal_def]; exact_mod_cast this

/-- **`tl.sum(x, 1)` on one row** of `N ≤ 256` finite `float32` values, each within `κ·w + ν`
of a weight `0 ≤ w ≤ 2`: within `(κ + γ(1 + κ))·Σw + N·ν·(1 + γ) + 2·N·η` of `Σ w`, with
`γ = 2·N·u`. -/
theorem sumRow_val {N : ℕ} (hN : N ≤ 256) {x : Fin N → F32} {r : F32} {w : Fin N → ℝ} {κ ν : ℝ}
    (h : ∃ t : Tree, t.Covers N ∧
      t.Eval (AddRel false) (fun j => if hj : j < N then x ⟨j, hj⟩ else default) r)
    (hx : ∀ k, (x k).isFinite = true) (hxw : ∀ k, |(x k).toReal - w k| ≤ κ * w k + ν)
    (hw : ∀ k, 0 ≤ w k) (hw2 : ∀ k, w k ≤ 2) (hκ0 : 0 ≤ κ) (hκ : κ ≤ 1 / 2) (hν : ν ≤ 1) :
    r.isFinite = true ∧ |r.toReal - ∑ k, w k| ≤
      (κ + 2 * N * Err.u * (1 + κ)) * ∑ k, w k + N * ν * (1 + 2 * N * Err.u) + 2 * N * Err.η := by
  obtain ⟨t, hcov, hev⟩ := h
  set xf : ℕ → F32 := fun j => if hj : j < N then x ⟨j, hj⟩ else default with hxf
  have hfin : ∀ i < N, (xf i).isFinite = true := fun i hi => by
    simp only [hxf, dite_eq_left hi]; exact hx _
  have eA : ((List.range N).map fun i => |(xf i).toRat|).sum = ∑ k : Fin N, |(x k).toRat| := by
    rw [list_sum_range, ← Fin.sum_univ_eq_sum_range (fun i => |(xf i).toRat|)]
    apply Finset.sum_congr rfl; intro k _; simp only [hxf, dite_eq_left k.2]
  have eS : ((List.range N).map fun i => (xf i).toRat).sum = ∑ k : Fin N, (x k).toRat := by
    rw [list_sum_range, ← Fin.sum_univ_eq_sum_range (fun i => (xf i).toRat)]
    apply Finset.sum_congr rfl; intro k _; simp only [hxf, dite_eq_left k.2]
  -- each element is at most `3`, so the absolute sum is at most `3N`
  have hxa : ∀ k, |(x k).toReal| ≤ (1 + κ) * w k + ν := fun k => by
    have := abs_sub_abs_le_abs_sub (x k).toReal (w k)
    rw [abs_of_nonneg (hw k)] at this; linarith [hxw k]
  have hAR : ∑ k, |(x k).toReal| ≤ (1 + κ) * ∑ k, w k + N * ν := by
    calc ∑ k, |(x k).toReal| ≤ ∑ k, ((1 + κ) * w k + ν) := Finset.sum_le_sum fun k _ => hxa k
      _ = (1 + κ) * ∑ k, w k + N * ν := by
          rw [Finset.sum_add_distrib, Finset.mul_sum]; simp
  have hW : ∑ k, w k ≤ 2 * N := by
    calc ∑ k, w k ≤ ∑ _k : Fin N, (2 : ℝ) := Finset.sum_le_sum fun k _ => hw2 k
      _ = 2 * N := by simp; ring
  have hNr : (N : ℝ) ≤ 256 := by exact_mod_cast hN
  have hA4 : ∑ k, |(x k).toReal| ≤ 4 * N := by nlinarith [Finset.sum_nonneg fun k (_ : k ∈ Finset.univ) => hw k]
  have hAq : (∑ k : Fin N, |(x k).toRat| : ℚ) ≤ 4 * N := by
    have : ((∑ k : Fin N, |(x k).toRat| : ℚ) : ℝ) ≤ 4 * N := by push_cast; exact hA4
    exact_mod_cast this
  -- the depth term
  have hu0 := Format.u_nonneg (f := Format.binary32)
  have he0 := Format.eta_nonneg (f := Format.binary32)
  have hu24 : Format.binary32.u = 2 ^ (-24 : ℤ) := by unfold Format.u; rfl
  have hNq : ((N - 1 : ℕ) : ℚ) ≤ N := by exact_mod_cast Nat.sub_le N 1
  have hpow := pow_le hu0 (N - 1) (by
    rw [hu24]; have : ((N - 1 : ℕ) : ℚ) ≤ 256 := by exact_mod_cast (Nat.sub_le N 1).trans hN
    have h2 : (2 : ℚ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
    rw [h2]; nlinarith)
  have hEB : Tree.errBound Format.binary32.u Format.binary32.eta (N - 1) (N - 1)
      (∑ k : Fin N, |(x k).toRat|) ≤
      2 * N * Format.binary32.u * (∑ k : Fin N, |(x k).toRat|) + 2 * N * Format.binary32.eta := by
    unfold Tree.errBound
    have hA0 : (0 : ℚ) ≤ ∑ k : Fin N, |(x k).toRat| := Finset.sum_nonneg fun _ _ => abs_nonneg _
    have hp1 : (1 + Format.binary32.u) ^ (N - 1) ≤ 2 := by
      have : 2 * ((N - 1 : ℕ) : ℚ) * Format.binary32.u ≤ 1 := by
        rw [hu24]; have : ((N - 1 : ℕ) : ℚ) ≤ 256 := by exact_mod_cast (Nat.sub_le N 1).trans hN
        have h2 : (2 : ℚ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
        rw [h2]; nlinarith
      linarith
    have e1 : ((1 + Format.binary32.u) ^ (N - 1) - 1) * (∑ k : Fin N, |(x k).toRat|) ≤
        (2 * N * Format.binary32.u) * (∑ k : Fin N, |(x k).toRat|) := by
      apply mul_le_mul_of_nonneg_right _ hA0
      nlinarith
    have e2 : Format.binary32.eta * ((N - 1 : ℕ) : ℚ) * (1 + Format.binary32.u) ^ (N - 1) ≤
        2 * N * Format.binary32.eta := by
      have : 0 ≤ Format.binary32.eta * ((N - 1 : ℕ) : ℚ) := by positivity
      nlinarith
    linarith
  have hroom : ((List.range N).map fun i => |(xf i).toRat|).sum + Tree.errBound Format.binary32.u
      Format.binary32.eta (N - 1) (N - 1) ((List.range N).map fun i => |(xf i).toRat|).sum ≤
      Format.binary32.maxVal := by
    rw [eA]
    have hNq' : (N : ℚ) ≤ 256 := by exact_mod_cast hN
    have hu1 : Format.binary32.u ≤ 1 := by rw [hu24]; norm_num
    have he1 : Format.binary32.eta ≤ 1 := by
      rw [show Format.binary32.eta = 2 ^ (-150 : ℤ) by
        unfold Format.eta; rw [show Format.binary32.emin - (Format.binary32.m : ℤ) - 1 = -150 by decide]]
      norm_num
    have hA0 : (0 : ℚ) ≤ ∑ k : Fin N, |(x k).toRat| := Finset.sum_nonneg fun _ _ => abs_nonneg _
    have : 2 * N * Format.binary32.u * (∑ k : Fin N, |(x k).toRat|) ≤ 2 * 256 * 1 * (4 * 256) := by
      have h1 : 2 * (N : ℚ) * Format.binary32.u ≤ 2 * 256 * 1 := by nlinarith
      have h2 : (∑ k : Fin N, |(x k).toRat|) ≤ 4 * 256 := by linarith
      nlinarith
    have := maxVal32_ge
    nlinarith
  obtain ⟨hrf, hre⟩ := sum_err_covers Format.binary32_regular xf hcov hev hfin hroom
  rw [eS, eA] at hre
  refine ⟨hrf, ?_⟩
  have hre' : |r.toReal - ∑ k, (x k).toReal| ≤
      2 * N * Err.u * ∑ k, |(x k).toReal| + 2 * N * Err.η := by
    have := hre.trans hEB
    have h' : ((|r.toRat - ∑ k : Fin N, (x k).toRat| : ℚ) : ℝ) ≤
        ((2 * N * Format.binary32.u * (∑ k : Fin N, |(x k).toRat|) + 2 * N * Format.binary32.eta : ℚ) : ℝ) := by
      exact_mod_cast this
    push_cast at h'
    rw [u32_eq, eta32_eq] at h'
    simpa only [toReal_def] using h'
  have hdiff : |∑ k, (x k).toReal - ∑ k, w k| ≤ κ * ∑ k, w k + N * ν := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ k, ((x k).toReal - w k)| ≤ ∑ k, |(x k).toReal - w k| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ k, (κ * w k + ν) := Finset.sum_le_sum fun k _ => hxw k
      _ = κ * ∑ k, w k + N * ν := by rw [Finset.sum_add_distrib, Finset.mul_sum]; simp
  have hu := Err.u_pos
  have e3 : 2 * N * Err.u * ∑ k, |(x k).toReal| ≤ 2 * N * Err.u * ((1 + κ) * ∑ k, w k + N * ν) :=
    mul_le_mul_of_nonneg_left hAR (by positivity)
  calc |r.toReal - ∑ k, w k| ≤ |r.toReal - ∑ k, (x k).toReal| + |∑ k, (x k).toReal - ∑ k, w k| := by
        have := abs_sub_le r.toReal (∑ k, (x k).toReal) (∑ k, w k); linarith
    _ ≤ _ := by nlinarith

/-! ## The running state of one row -/

section Row
variable (b : Bnd) (c : ℝ) (s : ℕ → ℝ) (vv : ℕ → Fin b.HD → ℝ)

/-- `Λ n μ = Σ_{j < n·BN} 2^(c·s j − μ)`. -/
noncomputable def Lam (n : ℕ) (μ : ℝ) : ℝ := ∑ j ∈ Finset.range (n * b.BN), (2 : ℝ) ^ (c * s j - μ)

/-- `A n μ e = Σ_{j < n·BN} 2^(c·s j − μ) · v j e`. -/
noncomputable def Acc (n : ℕ) (μ : ℝ) (e : Fin b.HD) : ℝ :=
  ∑ j ∈ Finset.range (n * b.BN), (2 : ℝ) ^ (c * s j - μ) * vv j e

/-- **The invariant of one row after `n ≥ 1` blocks**: the running maximum `μ` bounds every
scaled score seen up to `δ` and is within `δ` of one of them; the running sum `l` and the
accumulator `acc` are within `(ρl0 + n·dl)·Λ` and `(ρv0 + n·dv)·Λ` of `Λ n μ` and `A n μ`. -/
structure RowInv (n : ℕ) (μ l : F32) (acc : Fin b.HD → F32) : Prop where
  μf : μ.isFinite = true
  lf : l.isFinite = true
  accf : ∀ e, (acc e).isFinite = true
  μb : |μ.toReal| ≤ Err.Mμ b
  up : ∀ j < n * b.BN, c * s j ≤ μ.toReal + Err.δ b
  lo : ∃ j < n * b.BN, μ.toReal ≤ c * s j + Err.δ b
  l : |l.toReal - Lam b c s n μ.toReal| ≤ (Err.ρl0 b + n * Err.dl b) * Lam b c s n μ.toReal
  acc : ∀ e, |(acc e).toReal - Acc b c s vv n μ.toReal e| ≤
    (Err.ρv0 b + n * Err.dv b) * Lam b c s n μ.toReal

end Row

/-- `j` in block `n` is `n·BN + k`. -/
theorem block_idx {BN n j : ℕ} (h1 : n * BN ≤ j) (h2 : j < (n + 1) * BN) :
    ∃ k : Fin BN, j = n * BN + k := ⟨⟨j - n * BN, by rw [Nat.succ_mul] at h2; omega⟩, by simp; omega⟩

theorem block_lt {BN n : ℕ} (k : Fin BN) : n * BN + k < (n + 1) * BN := by
  rw [Nat.succ_mul]; have := k.2; omega

/-- **The running maximum after one more block.** -/
theorem mu_step {b : Bnd} (hb : b.Ok) {c : ℝ} {s : ℕ → ℝ} (hc0 : 0 < c) (hcC : c ≤ b.C)
    (hs : ∀ j < b.N, |s j| ≤ Err.S b) {n : ℕ} (hn : (n + 1) * b.BN ≤ b.N)
    {qk : Fin b.BN → F32}
    (hqk : ∀ k, (qk k).isFinite = true ∧ |(qk k).toReal - s (n * b.BN + k)| ≤ Err.ε₁ b)
    {rm : F32}
    (hrm : rm.isFinite = true ∧ (∃ k, rm.toReal = (qk k).toReal) ∧ ∀ k, (qk k).toReal ≤ rm.toReal)
    {cF mx : F32} (hcF : cF.isFinite = true) (hcv : cF.toReal = c)
    (hmx : FloatV.NaNEq (FloatV.mul rm cF) mx) {μ μ' : F32}
    (hold : (n = 0 ∧ μ = negInf32) ∨ (0 < n ∧ μ.isFinite = true ∧ |μ.toReal| ≤ Err.Mμ b ∧
      (∀ j < n * b.BN, c * s j ≤ μ.toReal + Err.δ b) ∧ ∃ j < n * b.BN, μ.toReal ≤ c * s j + Err.δ b))
    (hμ' : FloatV.MaxRel true μ mx μ') :
    μ'.isFinite = true ∧ |μ'.toReal| ≤ Err.Mμ b ∧
      (∀ j < (n + 1) * b.BN, c * s j ≤ μ'.toReal + Err.δ b) ∧
      (∃ j < (n + 1) * b.BN, μ'.toReal ≤ c * s j + Err.δ b) ∧ (0 < n → μ.toReal ≤ μ'.toReal) := by
  have hCQ := hb.CQ_nonneg
  have hu := Err.u_pos
  have hη := Err.η_pos
  have hsk : ∀ k : Fin b.BN, |s (n * b.BN + k)| ≤ Err.S b := fun k =>
    hs _ (lt_of_lt_of_le (block_lt k) hn)
  have hqkQ : ∀ k, |(qk k).toReal| ≤ Err.Q b := fun k => by
    have := abs_sub_abs_le_abs_sub (qk k).toReal (s (n * b.BN + k))
    unfold Err.Q; linarith [(hqk k).2, hsk k]
  obtain ⟨hrmf, ⟨kstar, hkstar⟩, hrmge⟩ := hrm
  have hrmQ : |rm.toReal| ≤ Err.Q b := by rw [hkstar]; exact hqkQ kstar
  have hrc : |rm.toReal * c| ≤ b.C * Err.Q b := by
    rw [abs_mul, abs_of_pos hc0, mul_comm]
    exact mul_le_mul hcC hrmQ (abs_nonneg _) hb.hC.le
  have hroom : |rm.toReal * cF.toReal| ≤ 2 ^ 126 := by
    rw [hcv]; have := hb.hCQ; linarith [show (2 : ℝ) ^ 60 ≤ 2 ^ 126 by norm_num]
  obtain ⟨hmxf, hmxe⟩ := mul_val hrmf hcF hmx hroom
  rw [hcv] at hmxe
  have hmxe' : |mx.toReal - rm.toReal * c| ≤ Err.u * (b.C * Err.Q b) + Err.η := by
    have := mul_le_mul_of_nonneg_right hrc hu.le; nlinarith
  have hmxM : |mx.toReal| ≤ Err.Mμ b := by
    have := abs_sub_abs_le_abs_sub mx.toReal (rm.toReal * c)
    unfold Err.Mμ; nlinarith
  -- every new scaled score is at most `mx + δ`, and the maximal one at least `mx − δ`
  have hcs : ∀ k : Fin b.BN, |c * s (n * b.BN + k) - c * (qk k).toReal| ≤ b.C * Err.ε₁ b := fun k => by
    rw [← mul_sub, abs_mul, abs_of_pos hc0, abs_sub_comm]
    exact mul_le_mul hcC (hqk k).2 (abs_nonneg _) hb.hC.le
  have hnew : ∀ k : Fin b.BN, c * s (n * b.BN + k) ≤ mx.toReal + Err.δ b := fun k => by
    have h1 : c * (qk k).toReal ≤ rm.toReal * c := by rw [mul_comm]; exact mul_le_mul_of_nonneg_right (hrmge k) hc0.le
    have h2 := (abs_le.mp (hcs k)).2
    have h3 := (abs_le.mp hmxe').1
    unfold Err.δ; linarith
  have hstar : mx.toReal ≤ c * s (n * b.BN + kstar) + Err.δ b := by
    have h2 := (abs_le.mp (hcs kstar)).1
    have h3 := (abs_le.mp hmxe').2
    rw [hkstar] at h3
    unfold Err.δ; nlinarith
  rcases hold with ⟨rfl, rfl⟩ | ⟨hn0, hμf, hμb, hup, j0, hj0, hlo⟩
  · -- first block: `μ' = mx`
    have := maxRel_negInf hmxf hμ'
    subst this
    refine ⟨hmxf, hmxM, fun j hj => ?_, ⟨0 * b.BN + kstar, block_lt kstar, hstar⟩, fun h => absurd h (lt_irrefl 0)⟩
    obtain ⟨k, rfl⟩ := block_idx (by rw [Nat.zero_mul]; exact Nat.zero_le _) hj
    exact hnew k
  · obtain ⟨hμ'f, hv, hge1, hge2⟩ := maxRel_val hμf hmxf hμ'
    have hge1' : μ.toReal ≤ μ'.toReal := by rw [toReal_def, toReal_def]; exact_mod_cast hge1
    have hge2' : mx.toReal ≤ μ'.toReal := by rw [toReal_def, toReal_def]; exact_mod_cast hge2
    refine ⟨hμ'f, ?_, fun j hj => ?_, ?_, fun _ => hge1'⟩
    · rcases hv with h | h
      · rw [toReal_def, h]; exact hμb
      · rw [toReal_def, h]; exact hmxM
    · by_cases hjn : j < n * b.BN
      · linarith [hup j hjn]
      · obtain ⟨k, rfl⟩ := block_idx (not_lt.mp hjn) hj
        linarith [hnew k]
    · rcases hv with h | h
      · refine ⟨j0, lt_of_lt_of_le hj0 (Nat.mul_le_mul_right _ (Nat.le_succ n)), ?_⟩
        rw [toReal_def, h]; exact hlo
      · refine ⟨n * b.BN + kstar, block_lt kstar, ?_⟩
        rw [toReal_def, h]; exact hstar

/-! ## The exact sums -/

section Sums
variable {b : Bnd} {c : ℝ} {s : ℕ → ℝ} {vv : ℕ → Fin b.HD → ℝ}

theorem lam_nonneg (n : ℕ) (μ : ℝ) : 0 ≤ Lam b c s n μ :=
  Finset.sum_nonneg fun _ _ => (two_rpow_pos _).le

/-- One more block. -/
theorem lam_succ (n : ℕ) (μ : ℝ) :
    Lam b c s (n + 1) μ = Lam b c s n μ + ∑ k : Fin b.BN, (2 : ℝ) ^ (c * s (n * b.BN + k) - μ) := by
  unfold Lam
  rw [Nat.succ_mul, Finset.sum_range_add, Fin.sum_univ_eq_sum_range (fun k => (2 : ℝ) ^ (c * s (n * b.BN + k) - μ))]

theorem acc_succ (n : ℕ) (μ : ℝ) (e : Fin b.HD) :
    Acc b c s vv (n + 1) μ e = Acc b c s vv n μ e +
      ∑ k : Fin b.BN, (2 : ℝ) ^ (c * s (n * b.BN + k) - μ) * vv (n * b.BN + k) e := by
  unfold Acc
  rw [Nat.succ_mul, Finset.sum_range_add,
    Fin.sum_univ_eq_sum_range (fun k => (2 : ℝ) ^ (c * s (n * b.BN + k) - μ) * vv (n * b.BN + k) e)]

/-- A new maximum rescales the sums by `2^(μ − μ')`. -/
theorem rpow_shift (x μ μ' : ℝ) : (2 : ℝ) ^ (x - μ') = (2 : ℝ) ^ (μ - μ') * (2 : ℝ) ^ (x - μ) := by
  rw [← Real.rpow_add (by norm_num)]; ring_nf

theorem lam_scale (n : ℕ) (μ μ' : ℝ) : Lam b c s n μ' = (2 : ℝ) ^ (μ - μ') * Lam b c s n μ := by
  unfold Lam; rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => rpow_shift _ _ _

theorem acc_scale (n : ℕ) (μ μ' : ℝ) (e : Fin b.HD) :
    Acc b c s vv n μ' e = (2 : ℝ) ^ (μ - μ') * Acc b c s vv n μ e := by
  unfold Acc; rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by rw [rpow_shift _ μ μ']; ring

theorem acc_le (hv : ∀ j < b.N, ∀ e, |vv j e| ≤ b.V) {n : ℕ} (hn : n * b.BN ≤ b.N) (μ : ℝ)
    (e : Fin b.HD) : |Acc b c s vv n μ e| ≤ b.V * Lam b c s n μ := by
  unfold Acc Lam
  rw [Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j hj => ?_)
  have hj' : j < b.N := lt_of_lt_of_le (Finset.mem_range.mp hj) hn
  rw [abs_mul, abs_of_pos (two_rpow_pos _), mul_comm]
  exact mul_le_mul_of_nonneg_right (hv j hj' e) (two_rpow_pos _).le

/-- The sum is at most `Λmax = 2N` while every scaled score is at most `μ + δ`. -/
theorem lam_le (hb : b.Ok) {n : ℕ} (hn : n * b.BN ≤ b.N) {μ : ℝ}
    (hup : ∀ j < n * b.BN, c * s j ≤ μ + Err.δ b) : Lam b c s n μ ≤ Err.Λmax b := by
  unfold Lam Err.Λmax
  calc ∑ j ∈ Finset.range (n * b.BN), (2 : ℝ) ^ (c * s j - μ) ≤ ∑ _j ∈ Finset.range (n * b.BN), (2 : ℝ) :=
        Finset.sum_le_sum fun j hj => two_rpow_le_two (by
          have := hup j (Finset.mem_range.mp hj); have := hb.δ_le4; linarith)
    _ = 2 * (n * b.BN : ℕ) := by simp; ring
    _ ≤ 2 * b.N := by have : ((n * b.BN : ℕ) : ℝ) ≤ b.N := by exact_mod_cast hn
                      linarith

/-- The sum is at least `1/2` once one scaled score is at least `μ − δ`. -/
theorem lam_ge (hb : b.Ok) {n : ℕ} {μ : ℝ} (hlo : ∃ j < n * b.BN, μ ≤ c * s j + Err.δ b) :
    1 / 2 ≤ Lam b c s n μ := by
  obtain ⟨j, hj, hle⟩ := hlo
  unfold Lam
  have h1 : (2 : ℝ) ^ (-Err.δ b) ≤ (2 : ℝ) ^ (c * s j - μ) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
  have h2 := two_rpow_neg_ge hb.δ_nonneg (le_trans hb.δ_le4 (by norm_num))
  have h3 : (2 : ℝ) ^ (c * s j - μ) ≤ ∑ j ∈ Finset.range (n * b.BN), (2 : ℝ) ^ (c * s j - μ) :=
    Finset.single_le_sum (f := fun j => (2 : ℝ) ^ (c * s j - μ)) (fun _ _ => (two_rpow_pos _).le)
      (Finset.mem_range.mpr hj)
  linarith

end Sums

/-! ## The exponentials -/

/-- The reference function of `exp2` is `2^x`. -/
theorem exp2_acc {P : Platform} (hexp : Rules.Num.MathAcc P .exp2 .binary32 Set.univ Err.relE Err.absE)
    {x r : F32} (hr : P.math .exp2 .binary32 x r) (hx : x.isFinite = true) (hx1 : x.toReal ≤ 1) :
    r.isFinite = true ∧ |r.toReal - (2 : ℝ) ^ x.toReal| ≤ Err.relE * (2 : ℝ) ^ x.toReal + Err.absE := by
  obtain ⟨ref, href, hacc⟩ := hexp
  have hg : ref = fun y => (2 : ℝ) ^ y := (Option.some.inj href).symm
  subst hg
  have h2 := two_rpow_le_two hx1
  have hp := two_rpow_pos x.toReal
  have hroom : (1 + Err.relE) * |(2 : ℝ) ^ x.toReal| + Err.absE ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [abs_of_pos hp]
    have := maxVal32_ge'; have := Err.relE_le; have := Err.absE_le; have := Err.relE_pos
    nlinarith [show (2 : ℝ) ^ 127 ≥ 8 by norm_num]
  obtain ⟨h1, h3⟩ := hacc x r hr hx (Set.mem_univ _) (by simp [MathFn.natDom]) hroom
  rw [abs_of_pos hp] at h3
  exact ⟨h1, h3⟩

/-- **The weights of a block**: `p = exp2(qk ⊗ c ⊖' μ')` against `w = 2^(c·s − μ')`. -/
theorem p_step {b : Bnd} (hb : b.Ok) {c : ℝ} {s : ℕ → ℝ} (hc0 : 0 < c) (hcC : c ≤ b.C)
    {n : ℕ} {qk : Fin b.BN → F32} (hqk : ∀ k, (qk k).isFinite = true ∧
      |(qk k).toReal - s (n * b.BN + k)| ≤ Err.ε₁ b) (hqkQ : ∀ k, |(qk k).toReal| ≤ Err.Q b)
    {cF : F32} (hcF : cF.isFinite = true) (hcv : cF.toReal = c)
    {μ' : F32} (hμ'f : μ'.isFinite = true) (hμ'b : |μ'.toReal| ≤ Err.Mμ b)
    (hup : ∀ k : Fin b.BN, c * s (n * b.BN + k) ≤ μ'.toReal + Err.δ b)
    {fusion : Bool} {qks x p : Fin b.BN → F32}
    (hqks : ∀ k, FloatV.NaNEq (FloatV.mul (qk k) cF) (qks k))
    (hx : ∀ k, AddRel fusion (qks k) (FloatV.neg μ') (x k))
    {P : Platform} (hexp : Rules.Num.MathAcc P .exp2 .binary32 Set.univ Err.relE Err.absE)
    (hp : ∀ k, P.math .exp2 .binary32 (x k) (p k)) (k : Fin b.BN) :
    (p k).isFinite = true ∧ |(p k).toReal - (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal)| ≤
      Err.κp b * (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) + Err.absE := by
  have hCQ := hb.CQ_nonneg
  have hu := Err.u_pos
  have hη := Err.η_pos
  have hqc : |(qk k).toReal * c| ≤ b.C * Err.Q b := by
    rw [abs_mul, abs_of_pos hc0, mul_comm]; exact mul_le_mul hcC (hqkQ k) (abs_nonneg _) hb.hC.le
  have hroom : |(qk k).toReal * cF.toReal| ≤ 2 ^ 126 := by
    rw [hcv]; have := hb.hCQ; linarith [show (2 : ℝ) ^ 60 ≤ 2 ^ 126 by norm_num]
  obtain ⟨hqsf, hqse⟩ := mul_val (hqk k).1 hcF (hqks k) hroom
  rw [hcv] at hqse
  have hqse' : |(qks k).toReal - (qk k).toReal * c| ≤ Err.u * (b.C * Err.Q b) + Err.η := by
    have := mul_le_mul_of_nonneg_right hqc hu.le; nlinarith
  have hqsM : |(qks k).toReal| ≤ Err.Mμ b := by
    have := abs_sub_abs_le_abs_sub (qks k).toReal ((qk k).toReal * c)
    unfold Err.Mμ; nlinarith
  have hnegf : (FloatV.neg μ').isFinite = true := by rw [FloatV.isFinite_neg]; exact hμ'f
  have hnegv : (FloatV.neg μ').toReal = -μ'.toReal := by rw [toReal_def, FloatV.toRat_neg]; push_cast; rfl
  have hMμ : Err.Mμ b ≤ 2 ^ 62 := hb.Mμ_le
  obtain ⟨hxf, hxe⟩ := addRel_val hqsf hnegf (hx k) (by
    rw [hnegv, abs_neg]; nlinarith [show (2 : ℝ) ^ 62 + 2 ^ 62 ≤ 2 ^ 120 by norm_num])
  rw [hnegv, abs_neg] at hxe
  -- the exponent error
  have hcs : |c * s (n * b.BN + k) - c * (qk k).toReal| ≤ b.C * Err.ε₁ b := by
    rw [← mul_sub, abs_mul, abs_of_pos hc0, abs_sub_comm]
    exact mul_le_mul hcC (hqk k).2 (abs_nonneg _) hb.hC.le
  have hxz : |(x k).toReal - (c * s (n * b.BN + k) - μ'.toReal)| ≤ Err.δx b := by
    have e1 : 3 * Err.u * (|(qks k).toReal| + |μ'.toReal|) ≤ 6 * Err.u * Err.Mμ b := by nlinarith
    have e : (x k).toReal - (c * s (n * b.BN + k) - μ'.toReal) =
        ((x k).toReal - ((qks k).toReal + -μ'.toReal)) + ((qks k).toReal - (qk k).toReal * c) +
          (c * (qk k).toReal - c * s (n * b.BN + k)) := by ring
    rw [e]
    have h3 := abs_add_le (((x k).toReal - ((qks k).toReal + -μ'.toReal)) + ((qks k).toReal - (qk k).toReal * c))
      (c * (qk k).toReal - c * s (n * b.BN + k))
    have h4 := abs_add_le ((x k).toReal - ((qks k).toReal + -μ'.toReal)) ((qks k).toReal - (qk k).toReal * c)
    rw [abs_sub_comm (c * (qk k).toReal)] at h3
    unfold Err.δx Err.δ; linarith
  have hx1 : (x k).toReal ≤ 1 := by
    have := (abs_le.mp hxz).2; have := hup k; have := hb.hδx; have := hb.δ_le4; linarith
  obtain ⟨hpf, hpe⟩ := exp2_acc hexp (hp k) hxf hx1
  have hw := two_rpow_pos (c * s (n * b.BN + k) - μ'.toReal)
  have h2 := two_rpow_sub_le hxz (le_trans hb.hδx (by norm_num))
  have h2' : (2 : ℝ) ^ (x k).toReal ≤ (1 + Err.δx b) * (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) := by
    have := (abs_le.mp h2).2; linarith
  refine ⟨hpf, ?_⟩
  have := mul_le_mul_of_nonneg_left h2' Err.relE_pos.le
  have e : (p k).toReal - (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) =
      ((p k).toReal - (2 : ℝ) ^ (x k).toReal) + ((2 : ℝ) ^ (x k).toReal - (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal)) := by ring
  rw [e]
  have := abs_add_le ((p k).toReal - (2 : ℝ) ^ (x k).toReal)
    ((2 : ℝ) ^ (x k).toReal - (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal))
  unfold Err.κp; nlinarith

/-- **The rescaling factor** `α = exp2(μ ⊖ μ')` against `2^(μ − μ')`, for finite `μ ≤ μ'`. -/
theorem alpha_step {b : Bnd} (hb : b.Ok) {μ μ' dd α : F32} (hμf : μ.isFinite = true)
    (hμ'f : μ'.isFinite = true) (hμb : |μ.toReal| ≤ Err.Mμ b) (hμ'b : |μ'.toReal| ≤ Err.Mμ b)
    (hle : μ.toReal ≤ μ'.toReal) (hdd : FloatV.NaNEq (FloatV.sub μ μ') dd)
    {P : Platform} (hexp : Rules.Num.MathAcc P .exp2 .binary32 Set.univ Err.relE Err.absE)
    (hα : P.math .exp2 .binary32 dd α) :
    α.isFinite = true ∧ |α.toReal - (2 : ℝ) ^ (μ.toReal - μ'.toReal)| ≤
      Err.κα b * (2 : ℝ) ^ (μ.toReal - μ'.toReal) + Err.absE := by
  have hu := Err.u_pos
  have hMμ : Err.Mμ b ≤ 2 ^ 62 := hb.Mμ_le
  have hd : |μ.toReal - μ'.toReal| ≤ 2 * Err.Mμ b := by
    have := abs_sub μ.toReal μ'.toReal; linarith
  obtain ⟨hddf, hdde⟩ := sub_val hμf hμ'f hdd (by nlinarith [show (2 : ℝ) ^ 63 ≤ 2 ^ 126 by norm_num])
  have hdz : |dd.toReal - (μ.toReal - μ'.toReal)| ≤ Err.δα b := by
    unfold Err.δα; nlinarith [mul_le_mul_of_nonneg_right hd hu.le]
  have hd1 : dd.toReal ≤ 1 := by
    have := (abs_le.mp hdz).2; have := hb.δα_le; have := hb.hδx; linarith
  obtain ⟨hαf, hαe⟩ := exp2_acc hexp hα hddf hd1
  have hg := two_rpow_pos (μ.toReal - μ'.toReal)
  have h2 := two_rpow_sub_le hdz (le_trans (hb.δα_le.trans hb.hδx) (by norm_num))
  have h2' : (2 : ℝ) ^ dd.toReal ≤ (1 + Err.δα b) * (2 : ℝ) ^ (μ.toReal - μ'.toReal) := by
    have := (abs_le.mp h2).2; linarith
  refine ⟨hαf, ?_⟩
  have := mul_le_mul_of_nonneg_left h2' Err.relE_pos.le
  have e : α.toReal - (2 : ℝ) ^ (μ.toReal - μ'.toReal) =
      (α.toReal - (2 : ℝ) ^ dd.toReal) + ((2 : ℝ) ^ dd.toReal - (2 : ℝ) ^ (μ.toReal - μ'.toReal)) := by ring
  rw [e]
  have := abs_add_le (α.toReal - (2 : ℝ) ^ dd.toReal) ((2 : ℝ) ^ dd.toReal - (2 : ℝ) ^ (μ.toReal - μ'.toReal))
  unfold Err.κα; nlinarith

/-! ## One block of one row -/

/-- **The running sum after one more block**, from the bound `hla` on the rescaled old sum
`l·α` and `hls` on the new block's sum. -/
theorem l_step {b : Bnd} (hb : b.Ok) {c : ℝ} {s : ℕ → ℝ} {n : ℕ} (hn : (n + 1) * b.BN ≤ b.N)
    {μ' : F32} (hup' : ∀ j < (n + 1) * b.BN, c * s j ≤ μ'.toReal + Err.δ b)
    (hlo' : ∃ j < (n + 1) * b.BN, μ'.toReal ≤ c * s j + Err.δ b)
    {l α lsum l' : F32} (hlf : l.isFinite = true) (hαf : α.isFinite = true)
    (hlsf : lsum.isFinite = true) {ρa ν1 : ℝ} (hρa0 : 0 ≤ ρa)
    (hρa : ρa ≤ Err.ρl0 b + n * Err.dl b + 3 / 2 * Err.κα b) (hρa1 : ρa ≤ 1) (hν10 : 0 ≤ ν1)
    (hν1 : ν1 ≤ 3 / 2 * Err.absE * Err.Λmax b)
    (hla : |l.toReal * α.toReal - Lam b c s n μ'.toReal| ≤ ρa * Lam b c s n μ'.toReal + ν1)
    (hls : |lsum.toReal - ∑ k : Fin b.BN, (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal)| ≤
      Err.κs b * ∑ k : Fin b.BN, (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) + Err.νs b)
    {fusion : Bool}
    (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l α) lsum) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l α lsum) l')) :
    l'.isFinite = true ∧ |l'.toReal - Lam b c s (n + 1) μ'.toReal| ≤
      (Err.ρl0 b + (n + 1 : ℕ) * Err.dl b) * Lam b c s (n + 1) μ'.toReal := by
  set G := Lam b c s n μ'.toReal with hGdef
  set S := ∑ k : Fin b.BN, (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) with hSdef
  have hG0 : 0 ≤ G := lam_nonneg _ _
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun _ _ => (two_rpow_pos _).le
  have hnle : n * b.BN ≤ b.N := le_trans (Nat.mul_le_mul_right _ (Nat.le_succ n)) hn
  have hGm : G ≤ Err.Λmax b := lam_le hb hnle fun j hj =>
    hup' j (lt_of_lt_of_le hj (Nat.mul_le_mul_right _ (Nat.le_succ n)))
  have hLsucc : Lam b c s (n + 1) μ'.toReal = G + S := lam_succ n _
  have hGS := lam_ge hb hlo'
  rw [hLsucc] at hGS
  have hSm : S + G ≤ Err.Λmax b := by have := lam_le hb hn hup'; rw [hLsucc] at this; linarith
  have hu := Err.u_pos
  have hη := Err.η_pos
  have hΛm : Err.Λmax b ≤ 2 ^ 32 := by
    unfold Err.Λmax; have : (b.N : ℝ) < 2 ^ 31 := by exact_mod_cast hb.hN31
    linarith
  have hκs := hb.κs_le
  have hνs : Err.νs b ≤ 1 := by
    unfold Err.νs
    have h1 : (b.BN : ℝ) ≤ 256 := by exact_mod_cast hb.hBN256
    have h0 : (0 : ℝ) ≤ b.BN := Nat.cast_nonneg _
    have hγ1 : 1 + Err.γ b ≤ 2 := by have := hb.γ_le; norm_num at this ⊢; linarith
    have e1 : (b.BN : ℝ) * Err.absE ≤ 256 * (1 / 2 ^ 149) :=
      mul_le_mul h1 Err.absE_le Err.absE_pos.le (by norm_num)
    have e2 : (b.BN : ℝ) * Err.absE * (1 + Err.γ b) ≤ 256 * (1 / 2 ^ 149) * 2 :=
      mul_le_mul e1 hγ1 (by have := (Bnd.Ok.γ_nonneg (b := b)); linarith) (by norm_num)
    have e3 : 2 * (b.BN : ℝ) * Err.η ≤ 2 * 256 * (1 / 2 ^ 150) :=
      mul_le_mul (by linarith) Err.η_le Err.η_pos.le (by norm_num)
    norm_num at e2 e3 ⊢; linarith
  have hlsa : |lsum.toReal| ≤ (1 + Err.κs b) * S + Err.νs b := by
    have := abs_sub_abs_le_abs_sub lsum.toReal S; rw [abs_of_nonneg hS0] at this; linarith
  have hlaa : |l.toReal * α.toReal| ≤ (1 + ρa) * G + ν1 := by
    have := abs_sub_abs_le_abs_sub (l.toReal * α.toReal) G; rw [abs_of_nonneg hG0] at this; linarith
  have hAbsE : Err.absE * Err.Λmax b ≤ 1 := by
    have := Err.absE_le; have := Err.absE_pos; nlinarith
  obtain ⟨hl'f, hl'e⟩ := mulAdd_val hlf hαf hlsf hl' (by nlinarith)
  refine ⟨hl'f, ?_⟩
  have hacc := accum_err (V := 1) (G := G) (S := S) (YG := G) (YS := S) (B := |lsum.toReal|)
    (κB := Err.κs b) (νB := Err.νs b) (θ := 3 * Err.u) (by positivity)
    (by rw [abs_of_nonneg hG0, one_mul]) hla hls (by linarith) hl'e
  rw [hLsucc]
  have hc0 : 0 ≤ (1 + 3 * Err.u) * ν1 + Err.νs b + 3 * Err.u * Err.νs b + 4 * Err.η := by
    have := (Bnd.Ok.νs_nonneg (b := b)); positivity
  have hcle : (1 + 3 * Err.u) * ν1 + Err.νs b + 3 * Err.u * Err.νs b + 4 * Err.η ≤ Err.νl b := by
    unfold Err.νl
    have := Err.u_le; have := (Bnd.Ok.νs_nonneg (b := b)); have := Err.absE_pos; have := hb.Λmax_pos
    nlinarith
  have hmexp : Err.ρl0 b + ((n + 1 : ℕ) : ℝ) * Err.dl b - 2 * Err.νl b =
      Err.ρl0 b + n * Err.dl b + 2 * Err.κα b + 12 * Err.u := by unfold Err.dl; push_cast; ring
  have ha' : ρa + 3 * Err.u * (1 + ρa) ≤ Err.ρl0 b + ((n + 1 : ℕ) : ℝ) * Err.dl b - 2 * Err.νl b := by
    rw [hmexp]
    have : 3 * Err.u * (1 + ρa) ≤ 6 * Err.u := by nlinarith
    have := hb.κα_nonneg
    linarith
  have hb' : Err.κs b + 3 * Err.u * (1 + Err.κs b) ≤
      Err.ρl0 b + ((n + 1 : ℕ) : ℝ) * Err.dl b - 2 * Err.νl b := by
    rw [hmexp]
    have : 3 * Err.u * (1 + Err.κs b) ≤ 6 * Err.u := by nlinarith
    have := hb.κα_nonneg; have := hb.dl_nonneg
    have : (0 : ℝ) ≤ n * Err.dl b := mul_nonneg (Nat.cast_nonneg n) hb.dl_nonneg
    unfold Err.ρl0; linarith
  have hm := relify (lo := 1 / 2) (m := Err.ρl0 b + ((n + 1 : ℕ) : ℝ) * Err.dl b - 2 * Err.νl b)
    (a := ρa + 3 * Err.u * (1 + ρa)) (b := Err.κs b + 3 * Err.u * (1 + Err.κs b))
    (c := (1 + 3 * Err.u) * ν1 + Err.νs b + 3 * Err.u * Err.νs b + 4 * Err.η) hG0 hS0 hc0
    (by norm_num) hGS ha' hb' hacc
  have : ((1 + 3 * Err.u) * ν1 + Err.νs b + 3 * Err.u * Err.νs b + 4 * Err.η) / (1 / 2) ≤ 2 * Err.νl b := by
    rw [div_le_iff₀ (by norm_num)]; linarith
  have hGS0 : 0 ≤ G + S := by linarith
  calc |l'.toReal - (G + S)| ≤ _ := hm
    _ ≤ (Err.ρl0 b + (n + 1 : ℕ) * Err.dl b - 2 * Err.νl b + 2 * Err.νl b) * (G + S) :=
        mul_le_mul_of_nonneg_right (by linarith) hGS0
    _ = _ := by ring

/-- `DType.val? .f32` of a finite value. -/
theorem val?_f32 {x : F32} (hx : x.isFinite = true) : DType.val? .f32 x = some x.toRat := by
  simp [DType.val?, hx]

theorem val?_f16 {x : F16} (hx : x.isFinite = true) : DType.val? .f16 x = some x.toRat := by
  simp [DType.val?, hx]

theorem of_val?_f32 {x : F32} {v : ℚ} (h : DType.val? .f32 x = some v) :
    x.isFinite = true ∧ x.toRat = v := by
  by_cases hx : x.isFinite = true
  · rw [val?_f32 hx] at h; exact ⟨hx, Option.some.inj h⟩
  · simp [DType.val?, hx] at h

set_option maxHeartbeats 1000000 in
/-- **The accumulator after one more block** (column `e`): from the bound `hold` on the rescaled
old accumulator `accs = acc ⊗ α`, the `float16` weights `p16` and the values `v16` of the block,
and the `tl.dot` rule (`hdot`). -/
theorem acc_step {b : Bnd} (hb : b.Ok) {c : ℝ} {s : ℕ → ℝ} {vv : ℕ → Fin b.HD → ℝ}
    (hv : ∀ j < b.N, ∀ e, |vv j e| ≤ b.V) {n : ℕ} (hn : (n + 1) * b.BN ≤ b.N) {μ' : F32}
    (hup' : ∀ j < (n + 1) * b.BN, c * s j ≤ μ'.toReal + Err.δ b)
    (hlo' : ∃ j < (n + 1) * b.BN, μ'.toReal ≤ c * s j + Err.δ b) (e : Fin b.HD)
    {accs acc' : F32} (haccsf : accs.isFinite = true) {ρa ν1 : ℝ} (hρa0 : 0 ≤ ρa)
    (hρa : ρa ≤ Err.ρv0 b + n * Err.dv b + b.V * (3 * Err.κα b + 2 * Err.u)) (hρa2 : ρa ≤ 2 * b.V)
    (hν10 : 0 ≤ ν1) (hν1 : ν1 ≤ 17 / 8 * Err.absE * b.V * Err.Λmax b + Err.η)
    (hold : |accs.toReal - Acc b c s vv n μ'.toReal e| ≤ ρa * Lam b c s n μ'.toReal + ν1)
    {p16 v16 : Fin b.BN → F16}
    (hp16 : ∀ k, (p16 k).isFinite = true ∧
      |(p16 k).toReal - (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal)| ≤
        Err.κ16 b * (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) + Err.ν16)
    (hv16 : ∀ k, (v16 k).isFinite = true ∧ (v16 k).toReal = vv (n * b.BN + k) e)
    (hdot : ∀ vc : ℚ, DType.val? .f32 accs = some vc →
      |vc| + ∑ k, |(p16 k).toRat * (v16 k).toRat| +
          (Err.relDot b.BN * (|vc| + ∑ k, |(p16 k).toRat * (v16 k).toRat|) + Err.absDot b.BN) ≤
        Format.binary32.maxVal →
      ∃ vr, DType.val? .f32 acc' = some vr ∧
        |vr - (vc + ∑ k, (p16 k).toRat * (v16 k).toRat)| ≤
          Err.relDot b.BN * (|vc| + ∑ k, |(p16 k).toRat * (v16 k).toRat|) + Err.absDot b.BN) :
    acc'.isFinite = true ∧ |acc'.toReal - Acc b c s vv (n + 1) μ'.toReal e| ≤
      (Err.ρv0 b + (n + 1 : ℕ) * Err.dv b) * Lam b c s (n + 1) μ'.toReal := by
  set G := Lam b c s n μ'.toReal with hGdef
  set w : Fin b.BN → ℝ := fun k => (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) with hwdef
  set S := ∑ k : Fin b.BN, w k with hSdef
  have hG0 : 0 ≤ G := lam_nonneg _ _
  have hw0 : ∀ k, 0 ≤ w k := fun k => (two_rpow_pos _).le
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun k _ => hw0 k
  have hnle : n * b.BN ≤ b.N := le_trans (Nat.mul_le_mul_right _ (Nat.le_succ n)) hn
  have hLsucc : Lam b c s (n + 1) μ'.toReal = G + S := lam_succ n _
  have hAsucc : Acc b c s vv (n + 1) μ'.toReal e =
      Acc b c s vv n μ'.toReal e + ∑ k : Fin b.BN, w k * vv (n * b.BN + k) e := acc_succ n _ e
  have hGS := lam_ge hb hlo'
  rw [hLsucc] at hGS
  have hSm : G + S ≤ Err.Λmax b := by have := lam_le hb hn hup'; rw [hLsucc] at this; exact this
  have hV := hb.hV
  have hvk : ∀ k : Fin b.BN, |(v16 k).toReal| ≤ b.V := fun k => by
    rw [(hv16 k).2]; exact hv _ (lt_of_lt_of_le (block_lt k) hn) e
  have hw2 : ∀ k, w k ≤ 2 := fun k => two_rpow_le_two (by
    have := hup' _ (block_lt k); have := hb.δ_le4; linarith)
  have hκ16 := hb.κ16_le
  have hp16a : ∀ k, |(p16 k).toReal| ≤ (1 + Err.κ16 b) * w k + Err.ν16 := fun k => by
    have := abs_sub_abs_le_abs_sub (p16 k).toReal (w k); rw [abs_of_nonneg (hw0 k)] at this
    linarith [(hp16 k).2]
  -- the new block, over the reals
  have hν16 : 0 ≤ Err.ν16 := by
    unfold Err.ν16; have := Err.absE_pos; have := Err.u16_pos; have := Err.η16_pos; positivity
  have hBr : ∑ k, |(p16 k).toReal * (v16 k).toReal| ≤
      (b.V + b.V * Err.κ16 b) * S + b.V * b.BN * Err.ν16 := by
    calc ∑ k, |(p16 k).toReal * (v16 k).toReal| ≤ ∑ k, ((1 + Err.κ16 b) * w k + Err.ν16) * b.V :=
          Finset.sum_le_sum fun k _ => by
            rw [abs_mul]; exact mul_le_mul (hp16a k) (hvk k) (abs_nonneg _) (by
              have := hb.κ16_nonneg; nlinarith [hw0 k])
      _ = ∑ k, ((b.V + b.V * Err.κ16 b) * w k + b.V * Err.ν16) :=
          Finset.sum_congr rfl fun k _ => by ring
      _ = (b.V + b.V * Err.κ16 b) * S + ∑ _k : Fin b.BN, b.V * Err.ν16 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum]
      _ = (b.V + b.V * Err.κ16 b) * S + b.V * b.BN * Err.ν16 := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have hθ0 := (Bnd.Ok.θ_nonneg (b := b))
  have hθ1 := hb.θ_le
  have hΛm : Err.Λmax b ≤ 2 ^ 32 := by
    unfold Err.Λmax; have : (b.N : ℝ) < 2 ^ 31 := by exact_mod_cast hb.hN31
    linarith
  have hV13 := hb.hV13
  have hYG : |Acc b c s vv n μ'.toReal e| ≤ b.V * G := acc_le hv hnle _ e
  have hGm : G ≤ Err.Λmax b := le_trans (by linarith) hSm
  have hSm' : S ≤ Err.Λmax b := le_trans (by linarith) hSm
  have hν1' : ν1 ≤ 1 := by
    have := Err.η_le
    have h2 : Err.absE * b.V * Err.Λmax b ≤ 1 / 2 ^ 149 * 2 ^ 13 * 2 ^ 32 := by
      have h1 : Err.absE * b.V ≤ 1 / 2 ^ 149 * 2 ^ 13 := mul_le_mul Err.absE_le hV13 hV.le (by norm_num)
      exact mul_le_mul h1 hΛm hb.Λmax_pos.le (by norm_num)
    have h3 : 17 / 8 * Err.absE * b.V * Err.Λmax b = 17 / 8 * (Err.absE * b.V * Err.Λmax b) := by ring
    norm_num at h2 this ⊢; linarith
  have hν16' : Err.ν16 ≤ 1 := by
    unfold Err.ν16
    have h1 : Err.absE * (1 + Err.u16) ≤ 1 / 2 ^ 149 * 2 :=
      mul_le_mul Err.absE_le (by have := Err.u16_le; norm_num at this ⊢; linarith)
        (by have := Err.u16_pos; linarith) (by norm_num)
    have : Err.η16 ≤ 1 / 2 ^ 25 := by unfold Err.η16; norm_num
    norm_num at h1 this ⊢; linarith
  have hBN : (b.BN : ℝ) ≤ 256 := by exact_mod_cast hb.hBN256
  have haccsa : |accs.toReal| ≤ 3 * b.V * Err.Λmax b + 1 := by
    have := abs_sub_abs_le_abs_sub accs.toReal (Acc b c s vv n μ'.toReal e)
    have : ρa * G ≤ 2 * b.V * Err.Λmax b := mul_le_mul hρa2 hGm hG0 (by linarith)
    have : b.V * G ≤ b.V * Err.Λmax b := mul_le_mul_of_nonneg_left hGm hV.le
    linarith
  have hBa : ∑ k, |(p16 k).toReal * (v16 k).toReal| ≤ 2 * b.V * Err.Λmax b + b.V * b.BN := by
    have h1 : (b.V + b.V * Err.κ16 b) * S ≤ 2 * b.V * Err.Λmax b := by
      have : b.V + b.V * Err.κ16 b ≤ 2 * b.V := by nlinarith [hb.κ16_nonneg]
      exact mul_le_mul this hSm' hS0 (by linarith)
    have h2 : b.V * b.BN * Err.ν16 ≤ b.V * b.BN := by
      have : 0 ≤ b.V * b.BN := mul_nonneg hV.le (Nat.cast_nonneg _)
      nlinarith
    linarith
  -- the `tl.dot` rule at this column
  have hsumR : ((∑ k, |(p16 k).toRat * (v16 k).toRat| : ℚ) : ℝ) = ∑ k, |(p16 k).toReal * (v16 k).toReal| := by
    push_cast; rfl
  have hroomR : |accs.toReal| + ∑ k, |(p16 k).toReal * (v16 k).toReal| +
      (Err.θ b * (|accs.toReal| + ∑ k, |(p16 k).toReal * (v16 k).toReal|) + (Err.absDot b.BN : ℝ)) ≤ 2 ^ 120 := by
    have hX : |accs.toReal| + ∑ k, |(p16 k).toReal * (v16 k).toReal| ≤ 2 ^ 50 := by
      have : b.V * Err.Λmax b ≤ 2 ^ 13 * 2 ^ 32 := mul_le_mul hV13 hΛm hb.Λmax_pos.le (by norm_num)
      have : b.V * b.BN ≤ 2 ^ 13 * 256 := mul_le_mul hV13 hBN (Nat.cast_nonneg _) (by norm_num)
      have e1 : 3 * b.V * Err.Λmax b = 3 * (b.V * Err.Λmax b) := by ring
      have e2 : 2 * b.V * Err.Λmax b = 2 * (b.V * Err.Λmax b) := by ring
      linarith
    have hX0 : 0 ≤ |accs.toReal| + ∑ k, |(p16 k).toReal * (v16 k).toReal| := by positivity
    have : Err.θ b * (|accs.toReal| + ∑ k, |(p16 k).toReal * (v16 k).toReal|) ≤ 1 / 16 * 2 ^ 50 :=
      mul_le_mul hθ1 hX hX0 (by norm_num)
    have : (Err.absDot b.BN : ℝ) ≤ 1 := by
      unfold Err.absDot; push_cast
      have : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
      rw [this]
      have : ((b.BN : ℝ) + 2) * (1 / 2 ^ 126) ≤ 258 * (1 / 2 ^ 126) :=
        mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
      norm_num at this ⊢; linarith
    norm_num at *; linarith
  have hroomq : |accs.toRat| + ∑ k, |(p16 k).toRat * (v16 k).toRat| +
      (Err.relDot b.BN * (|accs.toRat| + ∑ k, |(p16 k).toRat * (v16 k).toRat|) + Err.absDot b.BN) ≤
      Format.binary32.maxVal := by
    have h1 : ((|accs.toRat| + ∑ k, |(p16 k).toRat * (v16 k).toRat| +
        (Err.relDot b.BN * (|accs.toRat| + ∑ k, |(p16 k).toRat * (v16 k).toRat|) + Err.absDot b.BN) : ℚ) : ℝ) ≤
        2 ^ 120 := by
      push_cast; simpa only [toReal_def, Err.θ] using hroomR
    have h2 : (|accs.toRat| + ∑ k, |(p16 k).toRat * (v16 k).toRat| +
        (Err.relDot b.BN * (|accs.toRat| + ∑ k, |(p16 k).toRat * (v16 k).toRat|) + Err.absDot b.BN) : ℚ) ≤
        2 ^ 120 := by exact_mod_cast h1
    linarith [maxVal32_ge, show (2 : ℚ) ^ 120 ≤ 2 ^ 127 by norm_num]
  obtain ⟨vr, hvr, hdote⟩ := hdot _ (val?_f32 haccsf) hroomq
  obtain ⟨hf', hval'⟩ := of_val?_f32 hvr
  refine ⟨hf', ?_⟩
  have hdotR : |acc'.toReal - (accs.toReal + ∑ k, (p16 k).toReal * (v16 k).toReal)| ≤
      Err.θ b * (|accs.toReal| + ∑ k, |(p16 k).toReal * (v16 k).toReal|) + (Err.absDot b.BN : ℝ) := by
    rw [← hval'] at hdote
    have h' : ((|acc'.toRat - (accs.toRat + ∑ k, (p16 k).toRat * (v16 k).toRat)| : ℚ) : ℝ) ≤
        ((Err.relDot b.BN * (|accs.toRat| + ∑ k, |(p16 k).toRat * (v16 k).toRat|) + Err.absDot b.BN : ℚ) : ℝ) := by
      exact_mod_cast hdote
    push_cast at h'
    simpa only [toReal_def, Err.θ] using h'
  -- the new block against the exact one
  have h2 : |∑ k, (p16 k).toReal * (v16 k).toReal - ∑ k, w k * vv (n * b.BN + k) e| ≤
      b.V * Err.κ16 b * S + b.V * b.BN * Err.ν16 := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ k, ((p16 k).toReal * (v16 k).toReal - w k * vv (n * b.BN + k) e)|
        ≤ ∑ k, |(p16 k).toReal * (v16 k).toReal - w k * vv (n * b.BN + k) e| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ k, (Err.κ16 b * w k + Err.ν16) * b.V := Finset.sum_le_sum fun k _ => by
          rw [← (hv16 k).2, ← sub_mul, abs_mul]
          exact mul_le_mul (hp16 k).2 (hvk k) (abs_nonneg _) (by have := hb.κ16_nonneg; nlinarith [hw0 k])
      _ = ∑ k, (b.V * Err.κ16 b * w k + b.V * Err.ν16) := Finset.sum_congr rfl fun k _ => by ring
      _ = b.V * Err.κ16 b * S + ∑ _k : Fin b.BN, b.V * Err.ν16 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum]
      _ = b.V * Err.κ16 b * S + b.V * b.BN * Err.ν16 := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have hacc := accum_err (V := b.V) (G := G) (S := S) (YG := Acc b c s vv n μ'.toReal e)
    (YS := ∑ k, w k * vv (n * b.BN + k) e) (la := accs.toReal)
    (ls := ∑ k, (p16 k).toReal * (v16 k).toReal) (B := ∑ k, |(p16 k).toReal * (v16 k).toReal|)
    (ρa := ρa) (κ := b.V * Err.κ16 b) (κB := b.V * Err.κ16 b) (θ := Err.θ b) (ν1 := ν1)
    (ν2 := b.V * b.BN * Err.ν16) (νB := b.V * b.BN * Err.ν16) (ν3 := (Err.absDot b.BN : ℝ))
    hθ0 hYG hold h2 hBr hdotR
  rw [hAsucc, hLsucc]
  have hc0 : 0 ≤ (1 + Err.θ b) * ν1 + b.V * b.BN * Err.ν16 + Err.θ b * (b.V * b.BN * Err.ν16) +
      (Err.absDot b.BN : ℝ) := by
    have : (0 : ℝ) ≤ (Err.absDot b.BN : ℝ) := by exact_mod_cast Err.absDot_nonneg b.BN
    have : 0 ≤ b.V * b.BN * Err.ν16 := by have := hV.le; positivity
    positivity
  have hcle : (1 + Err.θ b) * ν1 + b.V * b.BN * Err.ν16 + Err.θ b * (b.V * b.BN * Err.ν16) +
      (Err.absDot b.BN : ℝ) ≤ Err.νa b := by
    unfold Err.νa
    have hX : 0 ≤ b.V * b.BN * Err.ν16 := by have := hV.le; positivity
    have := Err.absE_pos; have := hb.Λmax_pos; have := Err.η_pos
    have hY : 0 ≤ Err.absE * b.V * Err.Λmax b := by positivity
    have e1 : (1 + Err.θ b) * ν1 ≤ 17 / 16 * (17 / 8 * Err.absE * b.V * Err.Λmax b + Err.η) :=
      mul_le_mul (by linarith) hν1 hν10 (by norm_num)
    have e2 : Err.θ b * (b.V * b.BN * Err.ν16) ≤ 1 / 16 * (b.V * b.BN * Err.ν16) :=
      mul_le_mul_of_nonneg_right hθ1 hX
    have e3 : 17 / 16 * (17 / 8 * Err.absE * b.V * Err.Λmax b + Err.η) =
      289 / 128 * (Err.absE * b.V * Err.Λmax b) + 17 / 16 * Err.η := by ring
    have e4 : 3 * Err.absE * b.V * Err.Λmax b = 3 * (Err.absE * b.V * Err.Λmax b) := by ring
    linarith
  have hmexp : Err.ρv0 b + ((n + 1 : ℕ) : ℝ) * Err.dv b - 2 * Err.νa b =
      Err.ρv0 b + n * Err.dv b + b.V * (3 * Err.κα b + 2 * Err.u + 3 * Err.θ b) := by
    unfold Err.dv; push_cast; ring
  have ha' : ρa + Err.θ b * (b.V + ρa) ≤ Err.ρv0 b + ((n + 1 : ℕ) : ℝ) * Err.dv b - 2 * Err.νa b := by
    rw [hmexp]
    have h1 : Err.θ b * (b.V + ρa) ≤ Err.θ b * (3 * b.V) := mul_le_mul_of_nonneg_left (by linarith) hθ0
    have e : b.V * (3 * Err.κα b + 2 * Err.u + 3 * Err.θ b) =
      b.V * (3 * Err.κα b + 2 * Err.u) + Err.θ b * (3 * b.V) := by ring
    linarith
  have hb' : b.V * Err.κ16 b + Err.θ b * (b.V + b.V * Err.κ16 b) ≤
      Err.ρv0 b + ((n + 1 : ℕ) : ℝ) * Err.dv b - 2 * Err.νa b := by
    rw [hmexp]
    have hk : b.V * Err.κ16 b ≤ b.V := by
      have := mul_le_mul_of_nonneg_left hκ16 hV.le; linarith
    have h1 : Err.θ b * (b.V + b.V * Err.κ16 b) ≤ Err.θ b * (2 * b.V) :=
      mul_le_mul_of_nonneg_left (by linarith) hθ0
    have : (0 : ℝ) ≤ n * Err.dv b := mul_nonneg (Nat.cast_nonneg n) hb.dv_nonneg
    have := hb.κα_nonneg; have := Err.u_pos
    have : 0 ≤ b.V * (3 * Err.κα b + 2 * Err.u + 3 * Err.θ b) := by have := hV.le; positivity
    have e : b.V * (Err.κ16 b + 2 * Err.θ b) = b.V * Err.κ16 b + Err.θ b * (2 * b.V) := by ring
    unfold Err.ρv0; linarith
  have hm := relify (lo := 1 / 2) (m := Err.ρv0 b + ((n + 1 : ℕ) : ℝ) * Err.dv b - 2 * Err.νa b)
    (a := ρa + Err.θ b * (b.V + ρa)) (b := b.V * Err.κ16 b + Err.θ b * (b.V + b.V * Err.κ16 b))
    (c := (1 + Err.θ b) * ν1 + b.V * b.BN * Err.ν16 + Err.θ b * (b.V * b.BN * Err.ν16) +
      (Err.absDot b.BN : ℝ)) hG0 hS0 hc0 (by norm_num) hGS ha' hb' hacc
  have : ((1 + Err.θ b) * ν1 + b.V * b.BN * Err.ν16 + Err.θ b * (b.V * b.BN * Err.ν16) +
      (Err.absDot b.BN : ℝ)) / (1 / 2) ≤ 2 * Err.νa b := by
    rw [div_le_iff₀ (by norm_num)]; linarith
  have hGS0 : 0 ≤ G + S := by linarith
  calc |acc'.toReal - (Acc b c s vv n μ'.toReal e + ∑ k, w k * vv (n * b.BN + k) e)| ≤ _ := hm
    _ ≤ (Err.ρv0 b + ((n + 1 : ℕ) : ℝ) * Err.dv b - 2 * Err.νa b + 2 * Err.νa b) * (G + S) :=
        mul_le_mul_of_nonneg_right (by linarith) hGS0
    _ = _ := by ring

/-- `+0` in `float32`. -/
def zero32 : F32 := FloatV.ofPat 0

theorem zero32_isFinite : zero32.isFinite = true := by decide
theorem zero32_toReal : zero32.toReal = 0 := by
  rw [toReal_def, show zero32.toRat = 0 by decide]; simp

/-- The rescaled old running sum, `l ⊗ α ≈ 2^(μ − μ')·Λ n μ = Λ n μ'`. -/
theorem la_old {b : Bnd} (hb : b.Ok) {c : ℝ} {s : ℕ → ℝ} {n : ℕ} (hn : n * b.BN ≤ b.N)
    {μ μ' l α : F32} (hup : ∀ j < n * b.BN, c * s j ≤ μ.toReal + Err.δ b)
    (hl : |l.toReal - Lam b c s n μ.toReal| ≤ (Err.ρl0 b + n * Err.dl b) * Lam b c s n μ.toReal)
    (hρ0 : 0 ≤ Err.ρl0 b + n * Err.dl b)
    (hα : |α.toReal - (2 : ℝ) ^ (μ.toReal - μ'.toReal)| ≤
      Err.κα b * (2 : ℝ) ^ (μ.toReal - μ'.toReal) + Err.absE) :
    |l.toReal * α.toReal - Lam b c s n μ'.toReal| ≤
      (Err.ρl0 b + n * Err.dl b + Err.κα b * (1 + (Err.ρl0 b + n * Err.dl b))) * Lam b c s n μ'.toReal +
        Err.absE * (1 + (Err.ρl0 b + n * Err.dl b)) * Err.Λmax b := by
  have h := rescale_err (Y := Lam b c s n μ.toReal) (V := 1) (lam_nonneg _ _) (lam_le hb hn hup) hρ0
    zero_le_one (two_rpow_pos _).le Err.absE_pos.le (by rw [abs_of_nonneg (lam_nonneg _ _), one_mul]) hl hα
  rwa [← lam_scale] at h

/-- The rescaled old accumulator, `accs = acc ⊗ α ≈ A n μ'` (with the rounding of `⊗`). -/
theorem accs_old {b : Bnd} (hb : b.Ok) {c : ℝ} {s : ℕ → ℝ} {vv : ℕ → Fin b.HD → ℝ}
    (hv : ∀ j < b.N, ∀ e, |vv j e| ≤ b.V) {n : ℕ} (hn : n * b.BN ≤ b.N) (hnT : n ≤ Err.T b)
    {μ μ' α : F32} (hup : ∀ j < n * b.BN, c * s j ≤ μ.toReal + Err.δ b) (e : Fin b.HD)
    {acc accs : F32} (haccf : acc.isFinite = true) (hαf : α.isFinite = true)
    (hacc : |acc.toReal - Acc b c s vv n μ.toReal e| ≤ (Err.ρv0 b + n * Err.dv b) * Lam b c s n μ.toReal)
    (hα : |α.toReal - (2 : ℝ) ^ (μ.toReal - μ'.toReal)| ≤
      Err.κα b * (2 : ℝ) ^ (μ.toReal - μ'.toReal) + Err.absE)
    (hle : μ.toReal ≤ μ'.toReal) (haccs : FloatV.NaNEq (FloatV.mul acc α) accs) :
    accs.isFinite = true ∧
      |accs.toReal - Acc b c s vv n μ'.toReal e| ≤
        ((1 + Err.u) * (Err.ρv0 b + n * Err.dv b + Err.κα b * (b.V + (Err.ρv0 b + n * Err.dv b))) +
          Err.u * b.V) * Lam b c s n μ'.toReal +
        ((1 + Err.u) * (Err.absE * (b.V + (Err.ρv0 b + n * Err.dv b)) * Err.Λmax b) + Err.η) := by
  set ρv := Err.ρv0 b + n * Err.dv b with hρvdef
  have hρv0 : 0 ≤ ρv := by have := hb.ρv0_nonneg; have := hb.dv_nonneg; positivity
  have hρvV := hb.ρv_le hnT
  have hV := hb.hV
  set g := (2 : ℝ) ^ (μ.toReal - μ'.toReal) with hgdef
  have hg0 : 0 < g := two_rpow_pos _
  have hg1 : g ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos (by norm_num) (by linarith)
  have hΛ0 := lam_nonneg (b := b) (c := c) (s := s) n μ.toReal
  have hΛm := lam_le hb hn hup
  have h := rescale_err (l := acc.toReal) (Y := Acc b c s vv n μ.toReal e) (V := b.V) hΛ0 hΛm hρv0 hV.le
    hg0.le Err.absE_pos.le (acc_le hv hn _ e) hacc hα
  rw [← lam_scale, ← acc_scale] at h
  set E := (ρv + Err.κα b * (b.V + ρv)) * Lam b c s n μ'.toReal + Err.absE * (b.V + ρv) * Err.Λmax b with hE
  have hYG : |Acc b c s vv n μ'.toReal e| ≤ b.V * Lam b c s n μ'.toReal := acc_le hv hn _ e
  have hprod : |acc.toReal * α.toReal| ≤ b.V * Lam b c s n μ'.toReal + E := by
    have := abs_sub_abs_le_abs_sub (acc.toReal * α.toReal) (Acc b c s vv n μ'.toReal e); linarith
  -- room for the product
  have hΛ'm : Lam b c s n μ'.toReal ≤ Err.Λmax b := by rw [lam_scale (μ := μ.toReal)]; nlinarith
  have hΛmax : Err.Λmax b ≤ 2 ^ 32 := by
    unfold Err.Λmax; have : (b.N : ℝ) < 2 ^ 31 := by exact_mod_cast hb.hN31
    linarith
  have hEb : E ≤ 2 ^ 50 := by
    have hκα := hb.hκα
    have h1 : ρv + Err.κα b * (b.V + ρv) ≤ 2 * b.V := by nlinarith [hb.κα_nonneg]
    have h2 : (ρv + Err.κα b * (b.V + ρv)) * Lam b c s n μ'.toReal ≤ 2 * b.V * Err.Λmax b :=
      mul_le_mul h1 hΛ'm (lam_nonneg _ _) (by linarith)
    have h3 : Err.absE * (b.V + ρv) * Err.Λmax b ≤ 1 := by
      have : Err.absE * (b.V + ρv) ≤ 1 / 2 ^ 149 * 2 ^ 14 :=
        mul_le_mul Err.absE_le (by have := hb.hV13; linarith) (by linarith) (by norm_num)
      have := mul_le_mul this hΛmax hb.Λmax_pos.le (by norm_num)
      norm_num at this ⊢; linarith
    have h4 : b.V * Err.Λmax b ≤ 2 ^ 13 * 2 ^ 32 := mul_le_mul hb.hV13 hΛmax hb.Λmax_pos.le (by norm_num)
    have heq : 2 * b.V * Err.Λmax b = 2 * (b.V * Err.Λmax b) := by ring
    linarith
  have hroom : |acc.toReal * α.toReal| ≤ 2 ^ 126 := by
    have : b.V * Lam b c s n μ'.toReal ≤ 2 ^ 13 * 2 ^ 32 :=
      mul_le_mul hb.hV13 (hΛ'm.trans hΛmax) (lam_nonneg _ _) (by norm_num)
    linarith
  obtain ⟨hf, herr⟩ := mul_val haccf hαf haccs hroom
  refine ⟨hf, ?_⟩
  have heq : accs.toReal - Acc b c s vv n μ'.toReal e =
      (accs.toReal - acc.toReal * α.toReal) + (acc.toReal * α.toReal - Acc b c s vv n μ'.toReal e) := by ring
  rw [heq]
  have := abs_add_le (accs.toReal - acc.toReal * α.toReal) (acc.toReal * α.toReal - Acc b c s vv n μ'.toReal e)
  have h5 : |acc.toReal * α.toReal| * Err.u ≤ (b.V * Lam b c s n μ'.toReal + E) * Err.u :=
    mul_le_mul_of_nonneg_right hprod Err.u_pos.le
  have e2 : ((1 + Err.u) * (ρv + Err.κα b * (b.V + ρv)) + Err.u * b.V) * Lam b c s n μ'.toReal +
      ((1 + Err.u) * (Err.absE * (b.V + ρv) * Err.Λmax b) + Err.η) =
      (b.V * Lam b c s n μ'.toReal + E) * Err.u + Err.η + E := by rw [hE]; ring
  rw [e2]; linarith

/-- **The `float16` weights**: `p16 = fl16(p)` against `w`. -/
theorem p16_val {b : Bnd} (hb : b.Ok) {p : F32} {p16 : F16} {w : ℝ} (hpf : p.isFinite = true)
    (hpw : |p.toReal - w| ≤ Err.κp b * w + Err.absE) (hw0 : 0 ≤ w) (hw2 : w ≤ 2)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn p) p16) :
    p16.isFinite = true ∧ |p16.toReal - w| ≤ Err.κ16 b * w + Err.ν16 := by
  have hκp := hb.κp_le
  have hκp0 := hb.κp_nonneg
  have hpa : |p.toReal| ≤ (1 + Err.κp b) * w + Err.absE := by
    have := abs_sub_abs_le_abs_sub p.toReal w; rw [abs_of_nonneg hw0] at this; linarith
  have habsE := Err.absE_le
  obtain ⟨hf, herr⟩ := cvt16_val hpf h (by nlinarith)
  refine ⟨hf, ?_⟩
  have e : p16.toReal - w = (p16.toReal - p.toReal) + (p.toReal - w) := by ring
  rw [e]
  have := abs_add_le (p16.toReal - p.toReal) (p.toReal - w)
  have h2 : |p.toReal| * Err.u16 ≤ ((1 + Err.κp b) * w + Err.absE) * Err.u16 :=
    mul_le_mul_of_nonneg_right hpa Err.u16_pos.le
  unfold Err.κ16 Err.ν16; nlinarith

/-! ## One block, one row: the step of the invariant -/

set_option maxHeartbeats 1000000 in
/-- **One block of one row.** From the state after `n` blocks (`hold`: the initial state
`m = -inf`, `l` finite, `acc = 0` for `n = 0`; `RowInv` otherwise) and the relations each
operation of the loop body gives, the state after `n + 1` blocks satisfies `RowInv`. -/
theorem row_step {b : Bnd} (hb : b.Ok) {P : Platform} {fusion : Bool}
    (hexp : Rules.Num.MathAcc P .exp2 .binary32 Set.univ Err.relE Err.absE)
    (hinf : Rules.Num.MathAt P .exp2 .binary32 negInf32 zero32)
    {c : ℝ} {s : ℕ → ℝ} {vv : ℕ → Fin b.HD → ℝ} (hc0 : 0 < c) (hcC : c ≤ b.C)
    (hs : ∀ j < b.N, |s j| ≤ Err.S b) (hv : ∀ j < b.N, ∀ e, |vv j e| ≤ b.V)
    {n : ℕ} (hnT : n < Err.T b) {μ l : F32} {acc : Fin b.HD → F32}
    (hold : (n = 0 ∧ μ = negInf32 ∧ l.isFinite = true ∧ ∀ e, acc e = zero32) ∨
      (0 < n ∧ RowInv b c s vv n μ l acc))
    {qk : Fin b.BN → F32}
    (hqk : ∀ k, (qk k).isFinite = true ∧ |(qk k).toReal - s (n * b.BN + k)| ≤ Err.ε₁ b)
    {rm : F32} (hrm : ∃ t : Tree, t.Covers b.BN ∧
      t.Eval (FloatV.MaxRel true) (fun j => if hj : j < b.BN then qk ⟨j, hj⟩ else default) rm)
    {cF mx μ' : F32} (hcF : cF.isFinite = true) (hcv : cF.toReal = c)
    (hmx : FloatV.NaNEq (FloatV.mul rm cF) mx) (hμ' : FloatV.MaxRel true μ mx μ')
    {qks x p : Fin b.BN → F32} (hqks : ∀ k, FloatV.NaNEq (FloatV.mul (qk k) cF) (qks k))
    (hx : ∀ k, AddRel fusion (qks k) (FloatV.neg μ') (x k))
    (hp : ∀ k, P.math .exp2 .binary32 (x k) (p k))
    {dd α : F32} (hdd : FloatV.NaNEq (FloatV.sub μ μ') dd) (hα : P.math .exp2 .binary32 dd α)
    {lsum : F32} (hls : ∃ t : Tree, t.Covers b.BN ∧
      t.Eval (AddRel false) (fun j => if hj : j < b.BN then p ⟨j, hj⟩ else default) lsum)
    {accs : Fin b.HD → F32} (haccs : ∀ e, FloatV.NaNEq (FloatV.mul (acc e) α) (accs e))
    {p16 : Fin b.BN → F16} (hp16 : ∀ k, FloatV.NaNEq (FloatV.cvtWith .rn (p k)) (p16 k))
    {v16 : Fin b.BN → Fin b.HD → F16}
    (hv16 : ∀ k e, (v16 k e).isFinite = true ∧ (v16 k e).toReal = vv (n * b.BN + k) e)
    {acc' : Fin b.HD → F32}
    (hdot : (∀ k, (p16 k).isFinite = true) → ∀ e (vc : ℚ), DType.val? .f32 (accs e) = some vc →
      |vc| + ∑ k, |(p16 k).toRat * (v16 k e).toRat| +
          (Err.relDot b.BN * (|vc| + ∑ k, |(p16 k).toRat * (v16 k e).toRat|) + Err.absDot b.BN) ≤
        Format.binary32.maxVal →
      ∃ vr, DType.val? .f32 (acc' e) = some vr ∧
        |vr - (vc + ∑ k, (p16 k).toRat * (v16 k e).toRat)| ≤
          Err.relDot b.BN * (|vc| + ∑ k, |(p16 k).toRat * (v16 k e).toRat|) + Err.absDot b.BN)
    {l' : F32} (hl' : FloatV.NaNEq (FloatV.add (FloatV.mul l α) lsum) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l α lsum) l')) :
    RowInv b c s vv (n + 1) μ' l' acc' := by
  have hn : (n + 1) * b.BN ≤ b.N := by
    have h1 := (Nat.le_div_iff_mul_le hb.hBN).mp (Nat.succ_le_of_lt hnT)
    exact h1
  have hnle : n * b.BN ≤ b.N := le_trans (Nat.mul_le_mul_right _ (Nat.le_succ n)) hn
  have hsk : ∀ k : Fin b.BN, |s (n * b.BN + k)| ≤ Err.S b := fun k =>
    hs _ (lt_of_lt_of_le (block_lt k) hn)
  have hqkQ : ∀ k, |(qk k).toReal| ≤ Err.Q b := fun k => by
    have := abs_sub_abs_le_abs_sub (qk k).toReal (s (n * b.BN + k))
    unfold Err.Q; linarith [(hqk k).2, hsk k]
  have hrm' := maxRow_val hrm (fun k => (hqk k).1)
  have hold' : (n = 0 ∧ μ = negInf32) ∨ (0 < n ∧ μ.isFinite = true ∧ |μ.toReal| ≤ Err.Mμ b ∧
      (∀ j < n * b.BN, c * s j ≤ μ.toReal + Err.δ b) ∧ ∃ j < n * b.BN, μ.toReal ≤ c * s j + Err.δ b) := by
    rcases hold with ⟨h0, hμ, -, -⟩ | ⟨hpos, hI⟩
    · exact Or.inl ⟨h0, hμ⟩
    · exact Or.inr ⟨hpos, hI.μf, hI.μb, hI.up, hI.lo⟩
  obtain ⟨hμ'f, hμ'b, hup', hlo', hle⟩ := mu_step hb hc0 hcC hs hn hqk hrm' hcF hcv hmx hold' hμ'
  have hpk := p_step hb hc0 hcC hqk hqkQ hcF hcv hμ'f hμ'b (fun k => hup' _ (block_lt k)) hqks hx hexp hp
  have hw0 : ∀ k : Fin b.BN, 0 ≤ (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) := fun k => (two_rpow_pos _).le
  have hw2 : ∀ k : Fin b.BN, (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) ≤ 2 := fun k =>
    two_rpow_le_two (by have := hup' _ (block_lt k); have := hb.δ_le4; linarith)
  obtain ⟨hlsf, hlse⟩ := sumRow_val hb.hBN256 hls (fun k => (hpk k).1) (fun k => (hpk k).2) hw0 hw2
    hb.κp_nonneg hb.κp_le (by have := Err.absE_le; norm_num at this ⊢; linarith)
  have hlse' : |lsum.toReal - ∑ k : Fin b.BN, (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal)| ≤
      Err.κs b * ∑ k : Fin b.BN, (2 : ℝ) ^ (c * s (n * b.BN + k) - μ'.toReal) + Err.νs b := by
    refine le_of_le_of_eq hlse ?_; unfold Err.κs Err.νs Err.γ; ring
  have hp16k := fun k => p16_val hb (hpk k).1 (hpk k).2 (hw0 k) (hw2 k) (hp16 k)
  have hp16f : ∀ k, (p16 k).isFinite = true := fun k => (hp16k k).1
  have hΛm := hb.Λmax_pos
  have hκα := hb.κα_nonneg
  rcases hold with ⟨rfl, rfl, hlf, hacc0⟩ | ⟨hpos, hI⟩
  · -- the first block: `α = exp2(-inf) = +0`
    rw [sub_negInf hμ'f] at hdd
    have hdd' : dd = negInf32 := hdd.eq (by decide)
    rw [hdd'] at hα
    have hα0 : α = zero32 := hinf α hα
    subst hα0
    have hlam0 : Lam b c s 0 μ'.toReal = 0 := by simp [Lam]
    have hacc00 : ∀ e, Acc b c s vv 0 μ'.toReal e = 0 := fun e => by simp [Acc]
    have hla : |l.toReal * zero32.toReal - Lam b c s 0 μ'.toReal| ≤ 0 * Lam b c s 0 μ'.toReal + 0 := by
      rw [zero32_toReal, hlam0]; simp
    obtain ⟨hl'f, hl'e⟩ := l_step (ρa := 0) (ν1 := 0) hb hn hup' hlo' hlf zero32_isFinite hlsf le_rfl
      (by push_cast; have := hb.ρl0_nonneg; linarith) (by norm_num) le_rfl
      (by have := Err.absE_pos; positivity) hla hlse' hl'
    have hacc' : ∀ e, (acc' e).isFinite = true ∧
        |(acc' e).toReal - Acc b c s vv (0 + 1) μ'.toReal e| ≤
          (Err.ρv0 b + ((0 + 1 : ℕ) : ℝ) * Err.dv b) * Lam b c s (0 + 1) μ'.toReal := fun e => by
      have haccse := haccs e
      rw [hacc0 e] at haccse
      obtain ⟨hasf, hase⟩ := mul_val zero32_isFinite zero32_isFinite haccse (by
        rw [zero32_toReal]; norm_num)
      rw [zero32_toReal, mul_zero, abs_zero, zero_mul, zero_add, sub_zero] at hase
      refine acc_step (ρa := 0) (ν1 := Err.η) hb hv hn hup' hlo' e hasf le_rfl
        (by
          push_cast
          have h1 := hb.ρv0_nonneg
          have h2 : 0 ≤ b.V * (3 * Err.κα b + 2 * Err.u) :=
            mul_nonneg hb.hV.le (by linarith [hb.κα_nonneg, Err.u_pos])
          linarith)
        (by have := hb.hV; linarith) Err.η_pos.le
        (by have := Err.absE_pos; have := hb.hV; have : 0 ≤ 17 / 8 * Err.absE * b.V * Err.Λmax b := by positivity
            linarith) (by rw [hacc00 e, hlam0]; simpa using hase) hp16k (fun k => hv16 k e)
        (hdot hp16f e)
    exact ⟨hμ'f, hl'f, fun e => (hacc' e).1, hμ'b, hup', hlo', hl'e, fun e => (hacc' e).2⟩
  · -- later blocks: `α ≈ 2^(μ − μ')`
    have hle' := hle hpos
    have hnT' : n ≤ Err.T b := le_of_lt hnT
    obtain ⟨hαf, hαe⟩ := alpha_step hb hI.μf hμ'f hI.μb hμ'b hle' hdd hexp hα
    have hρ := hb.ρl_le hnT'
    have hρ0 : 0 ≤ Err.ρl0 b + n * Err.dl b := by
      have := hb.ρl0_nonneg; have := mul_nonneg (Nat.cast_nonneg (α := ℝ) n) hb.dl_nonneg; linarith
    have hla := la_old hb hnle hI.up hI.l hρ0 hαe
    have hκα4 := hb.hκα
    obtain ⟨hl'f, hl'e⟩ := l_step hb hn hup' hlo' hI.lf hαf hlsf (by positivity)
      (by nlinarith) (by nlinarith)
      (by have := Err.absE_pos; positivity)
      (by
        have h1 : Err.absE * (1 + (Err.ρl0 b + n * Err.dl b)) ≤ Err.absE * (3 / 2) :=
          mul_le_mul_of_nonneg_left (by linarith) Err.absE_pos.le
        have h2 := mul_le_mul_of_nonneg_right h1 hΛm.le
        linarith) hla hlse' hl'
    have hacc' : ∀ e, (acc' e).isFinite = true ∧
        |(acc' e).toReal - Acc b c s vv (n + 1) μ'.toReal e| ≤
          (Err.ρv0 b + ((n + 1 : ℕ) : ℝ) * Err.dv b) * Lam b c s (n + 1) μ'.toReal := fun e => by
      obtain ⟨hasf, hase⟩ := accs_old hb hv hnle hnT' hI.up e (hI.accf e) hαf (hI.acc e) hαe hle' (haccs e)
      have hρv := hb.ρv_le hnT'
      have hρv0 : 0 ≤ Err.ρv0 b + n * Err.dv b := by
        have := hb.ρv0_nonneg; have := mul_nonneg (Nat.cast_nonneg (α := ℝ) n) hb.dv_nonneg; linarith
      have hV := hb.hV
      have hu := Err.u_pos
      have hu' := Err.u_le
      refine acc_step hb hv hn hup' hlo' e hasf (by positivity) ?_ ?_ (by
          have := Err.absE_pos; have := Err.η_pos; positivity) ?_ hase hp16k (fun k => hv16 k e)
        (hdot hp16f e)
      · -- `(1 + u)(ρv + κα(V + ρv)) + uV ≤ ρv + V(3κα + 2u)`
        have h1 : Err.κα b * (b.V + (Err.ρv0 b + n * Err.dv b)) ≤ 2 * Err.κα b * b.V := by nlinarith
        have h2 : Err.u * (Err.ρv0 b + n * Err.dv b + Err.κα b * (b.V + (Err.ρv0 b + n * Err.dv b))) ≤
            Err.u * (b.V + 2 * Err.κα b * b.V) := mul_le_mul_of_nonneg_left (by linarith) hu.le
        have h3 : Err.u * (2 * Err.κα b * b.V) ≤ Err.κα b * b.V := by
          have : Err.u * 2 ≤ 1 := by norm_num at hu' ⊢; linarith
          have : 0 ≤ Err.κα b * b.V := by positivity
          nlinarith
        nlinarith
      · have h1 : Err.κα b * (b.V + (Err.ρv0 b + n * Err.dv b)) ≤ 1 / 4 * (2 * b.V) :=
          mul_le_mul hκα4 (by linarith) (by linarith) (by norm_num)
        have hu16 : Err.u ≤ 1 / 16 := by norm_num at hu' ⊢; linarith
        have h2 : (1 + Err.u) * (Err.ρv0 b + n * Err.dv b + Err.κα b * (b.V + (Err.ρv0 b + n * Err.dv b))) ≤
            17 / 16 * (3 / 2 * b.V) := mul_le_mul (by linarith) (by linarith) (by positivity) (by norm_num)
        have h3 : Err.u * b.V ≤ 1 / 16 * b.V := mul_le_mul_of_nonneg_right hu16 hV.le
        linarith
      · have h1 : Err.absE * (b.V + (Err.ρv0 b + n * Err.dv b)) * Err.Λmax b ≤ Err.absE * (2 * b.V) * Err.Λmax b := by
          have := Err.absE_pos
          apply mul_le_mul_of_nonneg_right _ hΛm.le
          exact mul_le_mul_of_nonneg_left (by linarith) this.le
        have : Err.u ≤ 1 / 16 := by norm_num at hu' ⊢; linarith
        have h2 : (1 + Err.u) * (Err.absE * (b.V + (Err.ρv0 b + n * Err.dv b)) * Err.Λmax b) ≤
            (17 / 16) * (Err.absE * (2 * b.V) * Err.Λmax b) := by
          have := Err.absE_pos
          have : 0 ≤ Err.absE * (b.V + (Err.ρv0 b + n * Err.dv b)) * Err.Λmax b := by positivity
          exact mul_le_mul (by linarith) h1 this (by positivity)
        have e : (17 / 16) * (Err.absE * (2 * b.V) * Err.Λmax b) = 17 / 8 * Err.absE * b.V * Err.Λmax b := by ring
        linarith
    exact ⟨hμ'f, hl'f, fun e => (hacc' e).1, hμ'b, hup', hlo', hl'e, fun e => (hacc' e).2⟩

/-! ## The epilogue of one row -/

/-- The reference function of `log2` is `log₂`. -/
theorem log2_acc {P : Platform} (hlog : Rules.Num.MathAcc P .log2 .binary32 Set.univ Err.relL Err.absL)
    {x r : F32} (hr : P.math .log2 .binary32 x r) (hx : x.isFinite = true) (hx0 : 0 < x.toReal)
    (hroom : |Real.logb 2 x.toReal| ≤ 2 ^ 100) :
    r.isFinite = true ∧ |r.toReal - Real.logb 2 x.toReal| ≤ Err.relL * |Real.logb 2 x.toReal| + Err.absL := by
  obtain ⟨ref, href, hacc⟩ := hlog
  have hg : ref = Real.logb 2 := (Option.some.inj href).symm
  subst hg
  have hroom' : (1 + Err.relL) * |Real.logb 2 x.toReal| + Err.absL ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    have := maxVal32_ge'; have := Err.relL_pos
    have h1 : Err.relL ≤ 1 := by unfold Err.relL; norm_num
    have h2 : Err.absL ≤ 1 := by unfold Err.absL; norm_num
    have : (1 + Err.relL) * |Real.logb 2 x.toReal| ≤ 2 * 2 ^ 100 :=
      mul_le_mul (by linarith) hroom (abs_nonneg _) (by norm_num)
    nlinarith [show (2 : ℝ) * 2 ^ 100 + 1 ≤ 2 ^ 127 by norm_num]
  exact hacc x r hr hx (Set.mem_univ _) hx0 hroom'

set_option maxHeartbeats 1000000 in
/-- **The outputs of one row.** After `T = N/BN` blocks, `M = m_i + log₂ l_i` is within `errM`
of `log₂ Σ_j 2^(c·s j)`, and `O = fl16(acc / l_i)` within `errO` of the softmax average
`Σ_j 2^(c·s j)·v j e / Σ_j 2^(c·s j)`. -/
theorem row_out {b : Bnd} (hb : b.Ok) {P : Platform}
    (hlog : Rules.Num.MathAcc P .log2 .binary32 Set.univ Err.relL Err.absL)
    (hdivA : Rules.Num.DivFull P)
    {c : ℝ} {s : ℕ → ℝ} {vv : ℕ → Fin b.HD → ℝ} (hv : ∀ j < b.N, ∀ e, |vv j e| ≤ b.V)
    {μ l : F32} {acc : Fin b.HD → F32} (hinv : RowInv b c s vv (Err.T b) μ l acc)
    {lg mm : F32} (hlg : P.math .log2 .binary32 l lg) (hmm : FloatV.NaNEq (FloatV.add μ lg) mm)
    {o : Fin b.HD → F32} (ho : ∀ e, P.div .binary32 (acc e) l (o e))
    {o16 : Fin b.HD → F16} (ho16 : ∀ e, FloatV.NaNEq (FloatV.cvtWith .rn (o e)) (o16 e)) :
    mm.isFinite = true ∧
      |mm.toReal - Real.logb 2 (∑ j ∈ Finset.range b.N, (2 : ℝ) ^ (c * s j))| ≤ Err.errM b ∧
    ∀ e, (o16 e).isFinite = true ∧
      |(o16 e).toReal - (∑ j ∈ Finset.range b.N, (2 : ℝ) ^ (c * s j) * vv j e) /
        (∑ j ∈ Finset.range b.N, (2 : ℝ) ^ (c * s j))| ≤ Err.errO b := by
  have hTN : Err.T b * b.BN = b.N := Nat.div_mul_cancel hb.hBNN
  set Λ := Lam b c s (Err.T b) μ.toReal with hΛdef
  have hsum : ∑ j ∈ Finset.range b.N, (2 : ℝ) ^ (c * s j) = (2 : ℝ) ^ μ.toReal * Λ := by
    rw [hΛdef, Lam, hTN, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by rw [← Real.rpow_add (by norm_num)]; ring_nf
  have hsumv : ∀ e, ∑ j ∈ Finset.range b.N, (2 : ℝ) ^ (c * s j) * vv j e =
      (2 : ℝ) ^ μ.toReal * Acc b c s vv (Err.T b) μ.toReal e := fun e => by
    rw [Acc, hTN, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by rw [← mul_assoc, ← Real.rpow_add (by norm_num)]; ring_nf
  have hΛ1 : 1 / 2 ≤ Λ := lam_ge hb hinv.lo
  have hΛ0 : 0 < Λ := by linarith
  have hΛm : Λ ≤ Err.Λmax b := lam_le hb (by rw [hTN]) hinv.up
  have hρ : Err.ρL b ≤ 1 / 2 := hb.hρL
  have hρ0 : 0 ≤ Err.ρL b := by
    unfold Err.ρL; have := hb.ρl0_nonneg; have := mul_nonneg (Nat.cast_nonneg (α := ℝ) (Err.T b)) hb.dl_nonneg
    linarith
  have hl : |l.toReal - Λ| ≤ Err.ρL b * Λ := hinv.l
  have hacc : ∀ e, |(acc e).toReal - Acc b c s vv (Err.T b) μ.toReal e| ≤ Err.ρV b * Λ := hinv.acc
  have hl0 : (1 - Err.ρL b) * Λ ≤ l.toReal := by have := (abs_le.mp hl).1; nlinarith
  have hlp : 0 < l.toReal := lt_of_lt_of_le (mul_pos (by linarith) hΛ0) hl0
  have hN1 : (1 : ℝ) ≤ b.N := by exact_mod_cast hb.hN
  have hN31 : (b.N : ℝ) < 2 ^ 31 := by exact_mod_cast hb.hN31
  -- `|log₂ Λ| ≤ log₂ (2N)`
  have hL2N1 : 1 ≤ Real.logb 2 (2 * b.N) := by
    rw [← Real.logb_self_eq_one (b := 2) (by norm_num)]
    exact (Real.logb_le_logb (by norm_num) (by norm_num) (by linarith)).mpr (by linarith)
  have hL2N32 : Real.logb 2 (2 * b.N) ≤ 32 := by
    have : Real.logb 2 (2 * b.N) ≤ Real.logb 2 ((2 : ℝ) ^ (32 : ℝ)) :=
      (Real.logb_le_logb (by norm_num) (by linarith) (by positivity)).mpr (by
        rw [show (2 : ℝ) ^ (32 : ℝ) = 2 ^ (32 : ℕ) by norm_cast]; linarith)
    rwa [Real.logb_rpow (by norm_num) (by norm_num)] at this
  have hlogΛ : |Real.logb 2 Λ| ≤ Real.logb 2 (2 * b.N) := by
    rw [abs_le]; constructor
    · have : Real.logb 2 (1 / 2) ≤ Real.logb 2 Λ := (Real.logb_le_logb (by norm_num) (by norm_num) hΛ0).mpr hΛ1
      rw [show (1 : ℝ) / 2 = 2 ^ (-1 : ℝ) by norm_num, Real.logb_rpow (by norm_num) (by norm_num)] at this
      linarith
    · exact (Real.logb_le_logb (by norm_num) hΛ0 (by linarith)).mpr (by unfold Err.Λmax at hΛm; linarith)
  have heL := logb_err hΛ0 hρ0 hρ hl
  have hLg : |Real.logb 2 l.toReal| ≤ Err.Lg b := by
    unfold Err.Lg Err.eL
    have := abs_sub_abs_le_abs_sub (Real.logb 2 l.toReal) (Real.logb 2 Λ); linarith
  have hLg33 : Err.Lg b ≤ 40 := by
    unfold Err.Lg Err.eL
    have : 2 * Err.ρL b / (1 - Err.ρL b) ≤ 2 := by
      rw [div_le_iff₀ (by linarith)]; linarith
    linarith
  -- `M`
  obtain ⟨hlgf, hlge⟩ := log2_acc hlog hlg hinv.lf hlp (by
    have := abs_nonneg (Real.logb 2 l.toReal); linarith [show (40 : ℝ) ≤ 2 ^ 100 by norm_num])
  have hlga : |lg.toReal| ≤ (1 + Err.relL) * Err.Lg b + Err.absL := by
    have := abs_sub_abs_le_abs_sub lg.toReal (Real.logb 2 l.toReal)
    have := mul_le_mul_of_nonneg_left hLg Err.relL_pos.le
    linarith
  have hMμ : Err.Mμ b ≤ 2 ^ 62 := hb.Mμ_le
  have hsuma : |μ.toReal + lg.toReal| ≤ Err.Mμ b + ((1 + Err.relL) * Err.Lg b + Err.absL) := by
    have := abs_add_le μ.toReal lg.toReal; linarith [hinv.μb]
  obtain ⟨hmmf, hmme⟩ := add_val hinv.μf hlgf hmm (by
    have : (1 + Err.relL) * Err.Lg b + Err.absL ≤ 2 * 40 + 1 := by
      have h1 : Err.relL ≤ 1 := by unfold Err.relL; norm_num
      have h2 : Err.absL ≤ 1 := by unfold Err.absL; norm_num
      have : 0 ≤ Err.Lg b := le_trans (abs_nonneg _) hLg
      nlinarith
    nlinarith [show (2 : ℝ) ^ 62 + 81 ≤ 2 ^ 126 by norm_num])
  refine ⟨hmmf, ?_, fun e => ?_⟩
  · rw [hsum, Real.logb_mul (two_rpow_pos _).ne' hΛ0.ne', Real.logb_rpow (by norm_num) (by norm_num)]
    have e1 : mm.toReal - (μ.toReal + Real.logb 2 Λ) = (mm.toReal - (μ.toReal + lg.toReal)) +
        (lg.toReal - Real.logb 2 l.toReal) + (Real.logb 2 l.toReal - Real.logb 2 Λ) := by ring
    rw [e1]
    have h1 := abs_add_le ((mm.toReal - (μ.toReal + lg.toReal)) + (lg.toReal - Real.logb 2 l.toReal))
      (Real.logb 2 l.toReal - Real.logb 2 Λ)
    have h2 := abs_add_le (mm.toReal - (μ.toReal + lg.toReal)) (lg.toReal - Real.logb 2 l.toReal)
    have h3 : |μ.toReal + lg.toReal| * Err.u ≤ (Err.Mμ b + ((1 + Err.relL) * Err.Lg b + Err.absL)) * Err.u :=
      mul_le_mul_of_nonneg_right hsuma Err.u_pos.le
    have h4 := mul_le_mul_of_nonneg_left hLg Err.relL_pos.le
    unfold Err.errM Err.eL
    have : Err.u * (Err.Mμ b + (1 + Err.relL) * Err.Lg b + Err.absL) =
      (Err.Mμ b + ((1 + Err.relL) * Err.Lg b + Err.absL)) * Err.u := by ring
    linarith
  · -- `O`
    have hA := acc_le (c := c) (s := s) hv (by rw [hTN]) μ.toReal e
    have hρL1 : Err.ρL b < 1 := by linarith
    have hrat := ratio_err hΛ0 hρL1 (hacc e) hl hA
    have hV := hb.hV
    have hρV := hb.hρV
    have heO : Err.eO b ≤ 3 * b.V := by
      unfold Err.eO
      rw [div_le_iff₀ (by linarith)]
      have : b.V * Err.ρL b ≤ b.V * (1 / 2) := mul_le_mul_of_nonneg_left hρ hV.le
      nlinarith
    have hρV0 : 0 ≤ Err.ρV b := by
      unfold Err.ρV; have := hb.ρv0_nonneg
      have := mul_nonneg (Nat.cast_nonneg (α := ℝ) (Err.T b)) hb.dv_nonneg; linarith
    have heO0 : 0 ≤ Err.eO b := by
      unfold Err.eO; apply div_nonneg _ (by linarith)
      have := mul_nonneg hV.le hρ0; linarith
    have hAΛ : |Acc b c s vv (Err.T b) μ.toReal e / Λ| ≤ b.V := by
      rw [abs_div, abs_of_pos hΛ0, div_le_iff₀ hΛ0]; exact hA
    have hal : |(acc e).toReal / l.toReal| ≤ b.V + Err.eO b := by
      have := abs_sub_abs_le_abs_sub ((acc e).toReal / l.toReal) (Acc b c s vv (Err.T b) μ.toReal e / Λ)
      unfold Err.eO; linarith
    -- the division
    have hl0q : l.toRat ≠ 0 := by
      intro h; rw [toReal_def, h] at hlp; simp at hlp
    have halq : |(acc e).toRat / l.toRat| ≤ 2 ^ 126 := by
      have : (((acc e).toRat / l.toRat : ℚ) : ℝ) = (acc e).toReal / l.toReal := by push_cast; rfl
      have h2 : ((|(acc e).toRat / l.toRat| : ℚ) : ℝ) ≤ 2 ^ 126 := by
        push_cast; rw [← toReal_def, ← toReal_def]
        have := hb.hV13; linarith [show (4 : ℝ) * 2 ^ 13 ≤ 2 ^ 126 by norm_num]
      exact_mod_cast h2
    obtain ⟨hof, hoe⟩ := hdivA (acc e) l (o e) (ho e) (hinv.accf e) hinv.lf hl0q halq
    have hoeR : |(o e).toReal - (acc e).toReal / l.toReal| ≤
        Err.relDiv * |(acc e).toReal / l.toReal| + Err.absDiv := by
      have h' : ((|(o e).toRat - (acc e).toRat / l.toRat| : ℚ) : ℝ) ≤
          (((2 : ℚ) ^ (-22 : ℤ) * |(acc e).toRat / l.toRat| + 2 ^ (-148 : ℤ) : ℚ) : ℝ) := by exact_mod_cast hoe
      push_cast at h'
      simpa only [toReal_def, Err.relDiv, Err.absDiv] using h'
    have hrd : Err.relDiv ≤ 1 := by unfold Err.relDiv; norm_num
    have had : Err.absDiv ≤ 1 := by unfold Err.absDiv; norm_num
    have hrd0 : 0 ≤ Err.relDiv := by unfold Err.relDiv; positivity
    have hoa : |(o e).toReal| ≤ (b.V + Err.eO b) * (1 + Err.relDiv) + Err.absDiv := by
      have := abs_sub_abs_le_abs_sub (o e).toReal ((acc e).toReal / l.toReal)
      have := mul_le_mul_of_nonneg_left hal hrd0
      nlinarith
    obtain ⟨ho16f, ho16e⟩ := cvt16_val hof (ho16 e) (by
      have hrd' : Err.relDiv ≤ 1 / 2 := by unfold Err.relDiv; norm_num
      have : (b.V + Err.eO b) * (1 + Err.relDiv) ≤ 4 * 2 ^ 13 * (3 / 2) :=
        mul_le_mul (by have := hb.hV13; linarith) (by linarith) (by linarith) (by norm_num)
      linarith)
    refine ⟨ho16f, ?_⟩
    rw [hsum, hsumv e, mul_div_mul_left _ _ (two_rpow_pos μ.toReal).ne']
    have e1 : (o16 e).toReal - Acc b c s vv (Err.T b) μ.toReal e / Λ = ((o16 e).toReal - (o e).toReal) +
        ((o e).toReal - (acc e).toReal / l.toReal) + ((acc e).toReal / l.toReal - Acc b c s vv (Err.T b) μ.toReal e / Λ) := by ring
    rw [e1]
    have h1 := abs_add_le (((o16 e).toReal - (o e).toReal) + ((o e).toReal - (acc e).toReal / l.toReal))
      ((acc e).toReal / l.toReal - Acc b c s vv (Err.T b) μ.toReal e / Λ)
    have h2 := abs_add_le ((o16 e).toReal - (o e).toReal) ((o e).toReal - (acc e).toReal / l.toReal)
    have h3 := mul_le_mul_of_nonneg_right hoa Err.u16_pos.le
    have h4 := mul_le_mul_of_nonneg_left hal hrd0
    unfold Err.errO
    have : Err.u16 * ((b.V + Err.eO b) * (1 + Err.relDiv) + Err.absDiv) =
      ((b.V + Err.eO b) * (1 + Err.relDiv) + Err.absDiv) * Err.u16 := by ring
    unfold Err.eO at this h3 h4 ⊢
    linarith

end TVAExamples.FlashAttn
