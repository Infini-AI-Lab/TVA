import Proof.Step
import Proof.Relax
import TVA.Proof

/-!
# The invariant of one query row

A query row's exact data (`RowCtx`): the natural-log scores `s k = sm_scale · q · k_k`, the values,
and which keys are attended. After `t` trips (blocks of `64` keys) the kernel holds, for the row, a
reference `m` (base 2, never decreasing), the sum `l` and the accumulator `a`. The exact state at
the reference `m` is `Ls`, `As` (`Softmax.lsum`, `Softmax.asum`), and the computed state is within
the closed form of `online_step` of it (`InvL`, `InvA`), with facts about the reference
(`InvM`): it is within `Bm X` of `0`, at least every attended score up to `ξ`, and near one of
them.
-/

namespace Claims.AttnFwd

open TVA Finset

noncomputable section

/-- One query row's exact data: `D` columns, the natural-log score `s k` of key `k`, the value
`v k e`, and whether key `k` is attended (`ok`); values of attended keys are at most `2^15`. -/
structure RowCtx where
  D : ℕ
  s : ℕ → ℝ
  v : ℕ → ℕ → ℝ
  ok : ℕ → Bool
  hv : ∀ k e, ok k = true → e < D → |v k e| ≤ 2 ^ 15

namespace RowCtx

variable (R : RowCtx)

/-- The attended lanes of block `u`. -/
def blk (u : ℕ) : Finset (Fin 64) := Finset.univ.filter (fun j => R.ok (u * 64 + j) = true)

/-- The score of lane `j` of block `u`. -/
def sf : ℕ → Fin 64 → ℝ := fun u j => R.s (u * 64 + j)

/-- The value of lane `j` of block `u`, column `e`. -/
def vf (e : ℕ) : ℕ → Fin 64 → ℝ := fun u j => R.v (u * 64 + j) e

/-- The exact row sum after `t` blocks at the reference `m` (base 2). -/
def Ls (m : ℝ) (t : ℕ) : ℝ := Softmax.lsum R.sf R.blk (m * Real.log 2) t

/-- The exact accumulator (column `e`) after `t` blocks at the reference `m`. -/
def As (e : ℕ) (m : ℝ) (t : ℕ) : ℝ := Softmax.asum R.sf (R.vf e) R.blk (m * Real.log 2) t

/-- The accumulator's magnitude (column `e`): the same with `|v|`. -/
def Ms (e : ℕ) (m : ℝ) (t : ℕ) : ℝ :=
  Softmax.asum R.sf (fun u j => |R.vf e u j|) R.blk (m * Real.log 2) t

/-- `Σ|v|` over the attended keys of the first `t` blocks (column `e`). -/
def Vs (e : ℕ) (t : ℕ) : ℝ := ∑ u ∈ range t, ∑ j ∈ R.blk u, |R.vf e u j|

/-- The exact weights of block `t` at the reference `m1`: `exp(s − m1·ln 2)` on attended keys,
`0` elsewhere. -/
def pw (t : ℕ) (m1 : ℝ) (j : Fin 64) : ℝ :=
  if R.ok (t * 64 + j) = true then Real.exp (R.s (t * 64 + j) - m1 * Real.log 2) else 0


/-- A sum over the attended lanes of a block is a sum over all lanes, with `0` on the others. -/
theorem sum_blk (t : ℕ) (f : Fin 64 → ℝ) :
    ∑ j ∈ R.blk t, f j = ∑ j : Fin 64, if R.ok (t * 64 + j) = true then f j else 0 := by
  unfold RowCtx.blk; rw [Finset.sum_filter]

/-- **One block of the row sum**: moving from the reference `m0` to `m1` rescales the sum seen so
far by `exp((m0 − m1)·ln 2)` and adds the block's weights. -/
theorem Ls_succ (m0 m1 : ℝ) (t : ℕ) :
    R.Ls m1 (t + 1) = Real.exp ((m0 - m1) * Real.log 2) * R.Ls m0 t + ∑ j, R.pw t m1 j := by
  unfold RowCtx.Ls
  rw [Softmax.lsum_succ (m0 * Real.log 2) (m1 * Real.log 2) t]
  have e1 : m0 * Real.log 2 - m1 * Real.log 2 = (m0 - m1) * Real.log 2 := by ring
  rw [e1, R.sum_blk]
  congr 1

/-- **One block of the accumulator** (column `e`). -/
theorem As_succ (e : ℕ) (m0 m1 : ℝ) (t : ℕ) :
    R.As e m1 (t + 1) = Real.exp ((m0 - m1) * Real.log 2) * R.As e m0 t +
      ∑ j, R.pw t m1 j * R.v (t * 64 + j) e := by
  unfold RowCtx.As
  rw [Softmax.asum_succ (m0 * Real.log 2) (m1 * Real.log 2) t]
  have e1 : m0 * Real.log 2 - m1 * Real.log 2 = (m0 - m1) * Real.log 2 := by ring
  rw [e1, R.sum_blk]
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  unfold RowCtx.pw RowCtx.sf RowCtx.vf; split_ifs <;> simp

/-- **One block of the accumulator's magnitude** (column `e`). -/
theorem Ms_succ (e : ℕ) (m0 m1 : ℝ) (t : ℕ) :
    R.Ms e m1 (t + 1) = Real.exp ((m0 - m1) * Real.log 2) * R.Ms e m0 t +
      ∑ j, R.pw t m1 j * |R.v (t * 64 + j) e| := by
  unfold RowCtx.Ms
  rw [Softmax.asum_succ (m0 * Real.log 2) (m1 * Real.log 2) t]
  have e1 : m0 * Real.log 2 - m1 * Real.log 2 = (m0 - m1) * Real.log 2 := by ring
  rw [e1, R.sum_blk]
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  unfold RowCtx.pw RowCtx.sf RowCtx.vf; split_ifs <;> simp

/-- **One block of `Σ|v|`**. -/
theorem Vs_succ (e : ℕ) (t : ℕ) :
    R.Vs e (t + 1) = R.Vs e t + ∑ j : Fin 64, if R.ok (t * 64 + j) = true then |R.v (t * 64 + j) e| else 0 := by
  unfold RowCtx.Vs
  rw [Finset.sum_range_succ, R.sum_blk]
  rfl

/-- The weights of a block are non-negative. -/
theorem pw_nonneg (t : ℕ) (m1 : ℝ) (j : Fin 64) : 0 ≤ R.pw t m1 j := by
  unfold RowCtx.pw; split_ifs
  · exact (Real.exp_pos _).le
  · exact le_rfl

/-- The exact state at no trips is `0`. -/
theorem Ls_zero (m : ℝ) : R.Ls m 0 = 0 := Softmax.lsum_zero _
/-- The zero blocks has `As` sum zero. -/
theorem As_zero (e : ℕ) (m : ℝ) : R.As e m 0 = 0 := Softmax.asum_zero _
/-- The zero blocks has `Ms` sum zero. -/
theorem Ms_zero (e : ℕ) (m : ℝ) : R.Ms e m 0 = 0 := Softmax.asum_zero _
/-- The zero blocks has `Vs` sum zero. -/
theorem Vs_zero (e : ℕ) : R.Vs e 0 = 0 := by simp [RowCtx.Vs]

/-- The exact row sum is non-negative. -/
theorem Ls_nonneg (m : ℝ) (t : ℕ) : 0 ≤ R.Ls m t := Softmax.lsum_nonneg _ _

/-- The accumulator's magnitude is non-negative. -/
theorem Ms_nonneg (e : ℕ) (m : ℝ) (t : ℕ) : 0 ≤ R.Ms e m t := by
  unfold RowCtx.Ms Softmax.asum
  exact Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => mul_nonneg (Real.exp_pos _).le (abs_nonneg _)

/-- `Σ|v|` is non-negative. -/
theorem Vs_nonneg (e : ℕ) (t : ℕ) : 0 ≤ R.Vs e t := by
  unfold RowCtx.Vs
  exact Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- The accumulator's magnitude is at most `2^15` times the row sum. -/
theorem Ms_le {e : ℕ} (he : e < R.D) (m : ℝ) (t : ℕ) : R.Ms e m t ≤ 2 ^ 15 * R.Ls m t := by
  refine Softmax.asum_abs_le (s := R.sf) (v := R.vf e) (B := R.blk) ?_
  intro u _ j hj
  have : R.ok (u * 64 + j) = true := by simpa [RowCtx.blk] using hj
  exact R.hv _ _ this he

/-- The exact accumulator is at most its magnitude. -/
theorem abs_As_le (e : ℕ) (m : ℝ) (t : ℕ) : |R.As e m t| ≤ R.Ms e m t :=
  Softmax.abs_asum_le _ _

/-- `Σ|v|` over the attended keys of `t` blocks is at most `64·t·2^15`. -/
theorem Vs_le {e : ℕ} (he : e < R.D) (t : ℕ) : R.Vs e t ≤ 64 * t * 2 ^ 15 := by
  unfold RowCtx.Vs
  calc ∑ u ∈ range t, ∑ j ∈ R.blk u, |R.vf e u j| ≤ ∑ u ∈ range t, (64 * 2 ^ 15 : ℝ) := by
        refine Finset.sum_le_sum fun u _ => ?_
        calc ∑ j ∈ R.blk u, |R.vf e u j| ≤ ∑ _j ∈ R.blk u, (2 ^ 15 : ℝ) := by
              refine Finset.sum_le_sum fun j hj => ?_
              have : R.ok (u * 64 + j) = true := by simpa [RowCtx.blk] using hj
              exact R.hv _ _ this he
          _ = (R.blk u).card * 2 ^ 15 := by simp
          _ ≤ 64 * 2 ^ 15 := by
              have : (R.blk u).card ≤ 64 := by
                have := Finset.card_le_univ (R.blk u); simpa using this
              have : ((R.blk u).card : ℝ) ≤ 64 := by exact_mod_cast this
              nlinarith
    _ = 64 * t * 2 ^ 15 := by simp; ring

end RowCtx

/-! ## The invariant after `t` trips -/

/-- The relative error of `exp2` (and the tiny absolute part counted as relative). -/
def Qα : ℝ := q22 + 1 / 2 ^ 100

/-- Facts about the reference `m` after `t` trips, and the first reference `mf` (`mf ≤ m`): both are
at most `Bm X` in magnitude, `m` is at least every attended score seen up to `ξ`, the row sum at
`m` is at least `wlow` (some attended score is within `ξ` of `m`) and at most `128·t`. -/
structure InvM (R : RowCtx) (X : ℝ) (t : ℕ) (m mf : ℝ) : Prop where
  ht : t ≤ 1024
  hm : |m| ≤ Bm X
  hmf : |mf| ≤ Bm X
  hmm : mf ≤ m
  hup : ∀ k, k < 64 * t → R.ok k = true → R.s k ≤ (m + xi R.D X) * Real.log 2
  hlow : 1 ≤ t → wlow R.D X ≤ R.Ls m t
  hLup : R.Ls m t ≤ 128 * t

/-- The computed row sum `l` is within the closed form of `online_step` of the exact one. -/
def InvL (R : RowCtx) (X : ℝ) (t : ℕ) (m mf l : ℝ) : Prop :=
  ∃ Θ : ℝ, 0 ≤ Θ ∧ Θ ≤ t * EPSL + CD * (m - mf) ∧
    |l - R.Ls m t| ≤ Real.exp Θ * ((Θ + rbarL R.D X) * R.Ls m t + etaAL * t)

/-- The computed accumulator `a` (every column) is within the closed form of the exact one. -/
def InvA (R : RowCtx) (X : ℝ) (t : ℕ) (m mf : ℝ) (a : ℕ → ℝ) : Prop :=
  ∃ Θ : ℝ, 0 ≤ Θ ∧ Θ ≤ t * EPSA + CD * (m - mf) ∧ ∀ e, e < R.D →
    |a e - R.As e m t| ≤ Real.exp Θ * ((Θ + rbarA R.D X) * R.Ms e m t +
      kapA R.D X * R.Ls m t + etavA * R.Vs e t + etaAA * t)

/-- Before any trip, with the reference `m` (virtual: the first one the kernel will compute). -/
theorem InvM.zero (R : RowCtx) {X m : ℝ} (hm : |m| ≤ Bm X) : InvM R X 0 m m :=
  ⟨by norm_num, hm, hm, le_rfl, fun k hk => absurd hk (by simp), fun h => absurd h (by norm_num),
    by simp [RowCtx.Ls_zero]⟩

/-- The `InvL` invariant holds for the zero blocks. -/
theorem InvL.zero (R : RowCtx) (X m : ℝ) : InvL R X 0 m m 0 :=
  ⟨0, le_rfl, by simp, by simp [RowCtx.Ls_zero]⟩

/-- The `InvA` invariant holds for the zero blocks. -/
theorem InvA.zero (R : RowCtx) (X m : ℝ) : InvA R X 0 m m (fun _ => 0) :=
  ⟨0, le_rfl, by simp, fun e _ => by simp [RowCtx.As_zero, RowCtx.Ms_zero, RowCtx.Ls_zero, RowCtx.Vs_zero]⟩

/-- The sum of the rates stays small. -/
theorem theta_small {X : ℝ} {t : ℕ} {m mf : ℝ} (hX : 0 ≤ X ∧ X ≤ 1024) (ht : t ≤ 1024)
    (hm : |m| ≤ Bm X) (hmf : |mf| ≤ Bm X) (_hmm : mf ≤ m) {c : ℝ} (hc : c = EPSL ∨ c = EPSA) :
    t * c + CD * (m - mf) ≤ 1 / 16 := by
  have hBm : Bm X ≤ 1443 / 1000 * 1024 + 1 / 2 ^ 40 := by unfold Bm; nlinarith [hX.2]
  have h1 : m - mf ≤ 2 * (1443 / 1000 * 1024 + 1 / 2 ^ 40) := by
    have := (abs_le.mp hm).2; have := (abs_le.mp hmf).1; linarith
  have hc' : c ≤ EPSA := by rcases hc with rfl | rfl <;> norm_num [EPSL, EPSA]
  have ht' : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have hcd : 0 ≤ CD := by unfold CD u32; norm_num
  have hc0 : 0 ≤ c := by rcases hc with rfl | rfl <;> norm_num [EPSL, EPSA]
  have e1 : (t : ℝ) * c ≤ 1024 * EPSA := mul_le_mul ht' hc' hc0 (by norm_num)
  have e2 : CD * (m - mf) ≤ CD * (2 * (1443 / 1000 * 1024 + 1 / 2 ^ 40)) :=
    mul_le_mul_of_nonneg_left h1 hcd
  have : 1024 * EPSA + CD * (2 * (1443 / 1000 * 1024 + 1 / 2 ^ 40)) ≤ 1 / 16 := by
    unfold EPSA CD u32; norm_num
  linarith

/-- `exp Θ ≤ 5/4` for the small rate sums. -/
theorem exp_le_five_fourths {Θ : ℝ} (h0 : 0 ≤ Θ) (h1 : Θ ≤ 1 / 16) : Real.exp Θ ≤ 5 / 4 := by
  have := Softmax.exp_le_one_add_two_mul h0 (by linarith)
  linarith

/-- A computed row sum is small: `|l| ≤ 2^18`. -/
theorem InvL.abs_le {R : RowCtx} {X : ℝ} {t : ℕ} {m mf l : ℝ} (hP : Par R.D X)
    (hM : InvM R X t m mf) (hL : InvL R X t m mf l) : |l| ≤ 2 ^ 18 := by
  obtain ⟨Θ, hΘ0, hΘ, hl⟩ := hL
  have hΘ1 : Θ ≤ 1 / 16 := hΘ.trans (theta_small hP.hX hM.ht hM.hm hM.hmf hM.hmm (Or.inl rfl))
  have hex := exp_le_five_fourths hΘ0 hΘ1
  have hr := hP.rbarL_le
  have hr0 := hP.rbarL_nonneg
  have hLs0 := R.Ls_nonneg m t
  have hLs := hM.hLup
  have ht' : (t : ℝ) ≤ 1024 := by exact_mod_cast hM.ht
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg t
  have hE : Real.exp Θ * ((Θ + rbarL R.D X) * R.Ls m t + etaAL * t) ≤
      5 / 4 * ((1 / 16 + 1 / 5) * (128 * 1024) + etaAL * 1024) := by
    have hpos : 0 ≤ (Θ + rbarL R.D X) * R.Ls m t + etaAL * t := by
      have : 0 ≤ etaAL := by unfold etaAL; norm_num
      positivity
    refine mul_le_mul hex ?_ hpos (by norm_num)
    have e1 : (Θ + rbarL R.D X) * R.Ls m t ≤ (1 / 16 + 1 / 5) * (128 * 1024) :=
      mul_le_mul (by linarith) (by nlinarith) hLs0 (by norm_num)
    have e2 : etaAL * (t : ℝ) ≤ etaAL * 1024 :=
      mul_le_mul_of_nonneg_left ht' (by unfold etaAL; norm_num)
    linarith
  have hLs1 : R.Ls m t ≤ 128 * 1024 := by nlinarith
  have : |l| ≤ |R.Ls m t| + |l - R.Ls m t| := by
    have := abs_add_le (R.Ls m t) (l - R.Ls m t)
    rwa [add_sub_cancel] at this
  rw [abs_of_nonneg hLs0] at this
  have h3 : (5 : ℝ) / 4 * ((1 / 16 + 1 / 5) * (128 * 1024) + etaAL * 1024) ≤ 2 ^ 17 := by
    unfold etaAL; norm_num
  nlinarith

/-- The rate of the row sum, from the rescaling factor's relative error. -/
theorem epsL_rate {ρα δ : ℝ} (hδ : 0 ≤ δ) (h : ρα ≤ Qα + CA * δ) : epsL ρα ≤ EPSL + CD * δ := by
  unfold epsL
  have h1 : ρα * (1 + 3 * u32) ≤ (Qα + CA * δ) * (1 + 3 * u32) :=
    mul_le_mul_of_nonneg_right h (by unfold u32; norm_num)
  have h2 : Qα * (1 + 3 * u32) + 3 * u32 ≤ EPSL := by unfold Qα q22 u32 EPSL; norm_num
  have h3 : CA * (1 + 3 * u32) ≤ CD := by unfold CA CD u32; norm_num
  have h4 := mul_le_mul_of_nonneg_right h3 hδ
  nlinarith

/-- The rate of the accumulator, from the rescaling factor's relative error. -/
theorem epsA_rate {ρα δ : ℝ} (hδ : 0 ≤ δ) (h : ρα ≤ Qα + CA * δ) : epsA ρα ≤ EPSA + CD * δ := by
  unfold epsA
  have h0 : 0 ≤ (1 + u32) * (1 + r64) := by unfold u32 r64; norm_num
  have h1 : ρα * ((1 + u32) * (1 + r64)) ≤ (Qα + CA * δ) * ((1 + u32) * (1 + r64)) :=
    mul_le_mul_of_nonneg_right h h0
  have h2 : Qα * ((1 + u32) * (1 + r64)) + (u32 + r64 * (1 + u32)) ≤ EPSA := by
    unfold Qα q22 u32 r64 EPSA; norm_num
  have h3 : CA * ((1 + u32) * (1 + r64)) ≤ CD := by unfold CA CD u32 r64; norm_num
  have h4 := mul_le_mul_of_nonneg_right h3 hδ
  nlinarith

/-- **One trip of the row sum.** From the invariant after `t` trips (reference `m0`), the new
reference `m1 ≥ m0`, the rescaling factor `α̂` (within `ρα·α + 2^-125` of `α`, `ρα ≤ Qα + CA·(m1−m0)`),
the weights `p̂` of the block (within `rhoP·p + 2^-125` of the exact ones), their sum `lij` along a
tree, and `l' = α̂·l + lij` rounded (an `fma`, or a product then a sum), the new row sum is within
the closed form after `t + 1` trips. `α` is the exact factor: `exp((m0 − m1)·ln 2)` (or `0` when
`t = 0`, where the sum so far is `0`). -/
theorem trip_l {R : RowCtx} {X : ℝ} (hP : Par R.D X) {t : ℕ} (_ht : t < 1024)
    {m0 m1 mf l0 l' α α' ρα lij : ℝ} {p' : Fin 64 → ℝ}
    (hM : InvM R X t m0 mf) (hL : InvL R X t m0 mf l0)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1)
    (hLs : R.Ls m1 (t + 1) = α * R.Ls m0 t + ∑ j, R.pw t m1 j) (hmm : m0 ≤ m1)
    (hρα : 0 ≤ ρα) (hραq : ρα ≤ Qα + CA * (m1 - m0)) (hα' : |α' - α| ≤ ρα * α + 1 / 2 ^ 125)
    (hp' : ∀ j, |p' j - R.pw t m1 j| ≤ rhoP R.D X * R.pw t m1 j + 1 / 2 ^ 125)
    (hs : |lij - ∑ j, p' j| ≤ rhoS * ∑ j, |p' j| + 1 / 2 ^ 143)
    (hl : |l' - (α' * l0 + lij)| ≤ 3 * u32 * (|α' * l0| + |lij|) + 4 / 2 ^ 150) :
    InvL R X (t + 1) m1 mf l' := by
  obtain ⟨Θ, hΘ0, hΘ, hl0⟩ := hL
  have hl0abs := InvL.abs_le hP hM ⟨Θ, hΘ0, hΘ, hl0⟩
  have hstep := stepL (D := R.D) (X := X) hα0 hρα (by norm_num) (by norm_num) hα' hl0abs
    hP.rhoP_nonneg (hP.rhoP_le.trans (by norm_num)) (R.pw_nonneg t m1) hp' hs hl
  have hP0 : 0 ≤ ∑ j, R.pw t m1 j := Finset.sum_nonneg fun j _ => R.pw_nonneg t m1 j
  have hε := epsL_rate (sub_nonneg.mpr hmm) hραq
  have hε0 : 0 ≤ epsL ρα := by unfold epsL u32; positivity
  have hLs0 := R.Ls_nonneg m0 t
  have hr0 := hP.rbarL_nonneg
  have key := online_step (α := α) (ε := epsL ρα) (Θ := Θ) (ρb := rbarL R.D X) (κ := 0) (ηv := 0)
    (ηA := etaAL) (t := (t : ℝ)) (x := l0) (x' := l') (X := R.Ls m0 t) (Y := ∑ j, R.pw t m1 j)
    (M := R.Ls m0 t) (Z := ∑ j, R.pw t m1 j) (L := 0) (R := 0) (V := 0) (W := 0)
    hα0 hα1 hε0 hΘ0 hr0 le_rfl le_rfl (by unfold etaAL; norm_num) (Nat.cast_nonneg t) hLs0 hP0
    le_rfl le_rfl le_rfl le_rfl (by rw [abs_of_nonneg hLs0]) (by rw [abs_of_nonneg hP0])
    (by simpa using hl0) (by simpa using hstep)
  refine ⟨Θ + epsL ρα, by positivity, ?_, ?_⟩
  · push_cast
    linarith
  · rw [hLs]
    push_cast
    simpa using key.1

/-- A computed accumulator is small: `|a e| ≤ 2^34`. -/
theorem InvA.abs_le {R : RowCtx} {X : ℝ} {t : ℕ} {m mf : ℝ} {a : ℕ → ℝ} (hP : Par R.D X)
    (hM : InvM R X t m mf) (hA : InvA R X t m mf a) {e : ℕ} (he : e < R.D) : |a e| ≤ 2 ^ 34 := by
  obtain ⟨Θ, hΘ0, hΘ, ha⟩ := hA
  have hΘ1 : Θ ≤ 1 / 16 := hΘ.trans (theta_small hP.hX hM.ht hM.hm hM.hmf hM.hmm (Or.inr rfl))
  have hex := exp_le_five_fourths hΘ0 hΘ1
  have hr := hP.rbarA_le
  have hr0 := hP.rbarA_nonneg
  have hk := hP.kapA_le
  have hk0 := hP.kapA_nonneg
  have hLs0 := R.Ls_nonneg m t
  have hLs := hM.hLup
  have hMs := R.Ms_le he m t
  have hMs0 := R.Ms_nonneg e m t
  have hVs := R.Vs_le he t
  have hVs0 := R.Vs_nonneg e t
  have ht' : (t : ℝ) ≤ 1024 := by exact_mod_cast hM.ht
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg t
  have hLs1 : R.Ls m t ≤ 128 * 1024 := by nlinarith
  have hMs1 : R.Ms e m t ≤ 2 ^ 15 * (128 * 1024) := hMs.trans (by nlinarith)
  have hVs1 : R.Vs e t ≤ 64 * 1024 * 2 ^ 15 := hVs.trans (by nlinarith)
  have h1 : (Θ + rbarA R.D X) * R.Ms e m t ≤ (1 / 16 + 1 / 5) * (2 ^ 15 * (128 * 1024)) :=
    mul_le_mul (by linarith) hMs1 hMs0 (by norm_num)
  have h2 : kapA R.D X * R.Ls m t ≤ 1 / 2 ^ 10 * (128 * 1024) := mul_le_mul hk hLs1 hLs0 (by norm_num)
  have h3 : etavA * R.Vs e t ≤ etavA * (64 * 1024 * 2 ^ 15) :=
    mul_le_mul_of_nonneg_left hVs1 (by unfold etavA eta16 r64; norm_num)
  have h4 : etaAA * (t : ℝ) ≤ etaAA * 1024 := mul_le_mul_of_nonneg_left ht' (by unfold etaAA; norm_num)
  have hpos : 0 ≤ (Θ + rbarA R.D X) * R.Ms e m t + kapA R.D X * R.Ls m t + etavA * R.Vs e t +
      etaAA * t := by
    have : 0 ≤ etavA := by unfold etavA eta16 r64; norm_num
    have : 0 ≤ etaAA := by unfold etaAA; norm_num
    positivity
  have hE : Real.exp Θ * ((Θ + rbarA R.D X) * R.Ms e m t + kapA R.D X * R.Ls m t +
      etavA * R.Vs e t + etaAA * t) ≤ 5 / 4 * ((1 / 16 + 1 / 5) * (2 ^ 15 * (128 * 1024)) +
      1 / 2 ^ 10 * (128 * 1024) + etavA * (64 * 1024 * 2 ^ 15) + etaAA * 1024) :=
    mul_le_mul hex (by linarith) hpos (by norm_num)
  have h5 : (5 : ℝ) / 4 * ((1 / 16 + 1 / 5) * (2 ^ 15 * (128 * 1024)) + 1 / 2 ^ 10 * (128 * 1024) +
      etavA * (64 * 1024 * 2 ^ 15) + etaAA * 1024) ≤ 2 ^ 33 := by
    unfold etavA eta16 r64 etaAA; norm_num
  have habs := R.abs_As_le e m t
  have hh := ha e he
  have : |a e| ≤ |R.As e m t| + |a e - R.As e m t| := by
    have := abs_add_le (R.As e m t) (a e - R.As e m t)
    rwa [add_sub_cancel] at this
  have h6 : (2 : ℝ) ^ 15 * (128 * 1024) = 2 ^ 32 := by norm_num
  have h7 : (2 : ℝ) ^ 33 + 2 ^ 32 ≤ 2 ^ 34 := by norm_num
  linarith

/-- **One trip of the accumulator.** As `trip_l`, for every column `e`: `a1 e` is the rescaled
accumulator (`α̂·a0 e`, rounded), the weights `p16` are the `float16` roundings of the block's
weights (within `rho16·p + eta16`, and `0` with lifted magnitude `0` on the masked lanes), `lp`,
`lv e` the lifted magnitudes of `p16` and of the values, and `a' e` the `tl.dot`. -/
theorem trip_a {R : RowCtx} {X : ℝ} (hP : Par R.D X) {t : ℕ} (_ht : t < 1024)
    {m0 m1 mf α α' ρα : ℝ} {a0 a1 a' : ℕ → ℝ} {p16 lp : Fin 64 → ℝ} {lv : ℕ → Fin 64 → ℝ}
    (hM : InvM R X t m0 mf) (hA : InvA R X t m0 mf a0)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (hmm : m0 ≤ m1)
    (hLs : R.Ls m1 (t + 1) = α * R.Ls m0 t + ∑ j, R.pw t m1 j)
    (hAs : ∀ e, R.As e m1 (t + 1) = α * R.As e m0 t + ∑ j, R.pw t m1 j * R.v (t * 64 + j) e)
    (hMs : ∀ e, R.Ms e m1 (t + 1) = α * R.Ms e m0 t + ∑ j, R.pw t m1 j * |R.v (t * 64 + j) e|)
    (hρα : 0 ≤ ρα) (hραq : ρα ≤ Qα + CA * (m1 - m0)) (hα' : |α' - α| ≤ ρα * α + 1 / 2 ^ 125)
    (ha1 : ∀ e, e < R.D → |a1 e - α' * a0 e| ≤ u32 * |α' * a0 e| + 1 / 2 ^ 150)
    (hp16 : ∀ j, |p16 j - R.pw t m1 j| ≤ rho16 R.D X * R.pw t m1 j + eta16)
    (hmask : ∀ j : Fin 64, R.ok (t * 64 + j) = false → p16 j = 0 ∧ lp j = 0)
    (hlp : ∀ j, |p16 j| ≤ lp j ∧ lp j ≤ |p16 j| + 1 / 2 ^ 14)
    (hlv : ∀ e, e < R.D → ∀ j : Fin 64, |R.v (t * 64 + j) e| ≤ lv e j ∧ lv e j ≤ |R.v (t * 64 + j) e| + 1 / 2 ^ 14)
    (hdot : ∀ e, e < R.D → |a' e - (a1 e + ∑ j, p16 j * R.v (t * 64 + j) e)| ≤
      r64 * (|a1 e| + ∑ j, lp j * lv e j) + 66 / 2 ^ 126) :
    InvA R X (t + 1) m1 mf a' := by
  obtain ⟨Θ, hΘ0, hΘ, ha⟩ := hA
  have hε := epsA_rate (sub_nonneg.mpr hmm) hραq
  have hε0 : 0 ≤ epsA ρα := epsA_nonneg hρα
  have hP0 : 0 ≤ ∑ j, R.pw t m1 j := Finset.sum_nonneg fun j _ => R.pw_nonneg t m1 j
  have hLs0 := R.Ls_nonneg m0 t
  have hr0 := hP.rbarA_nonneg
  refine ⟨Θ + epsA ρα, by positivity, by push_cast; linarith, fun e he => ?_⟩
  have hx0 := InvA.abs_le hP hM ⟨Θ, hΘ0, hΘ, ha⟩ he
  have hstep := stepA (D := R.D) (X := X) (ok := fun j => R.ok (t * 64 + j)) (v := fun j => R.v (t * 64 + j) e)
    (p := R.pw t m1) (lp := lp) (lv := lv e) (p16 := p16) hα0 hρα (by norm_num) (by norm_num) hα' hx0
    (ha1 e he) hP.rho16_nonneg (R.pw_nonneg t m1) hp16
    (fun j hj => ⟨by simp [RowCtx.pw, hj], hmask j hj⟩) hlp (hlv e he) (hdot e he)
  have hZ0 : 0 ≤ ∑ j, R.pw t m1 j * |R.v (t * 64 + j) e| :=
    Finset.sum_nonneg fun j _ => mul_nonneg (R.pw_nonneg t m1 j) (abs_nonneg _)
  have hW0 : 0 ≤ ∑ j : Fin 64, (if R.ok (t * 64 + j) = true then |R.v (t * 64 + j) e| else 0) :=
    Finset.sum_nonneg fun j _ => by split_ifs <;> positivity
  have hYZ : |∑ j, R.pw t m1 j * R.v (t * 64 + j) e| ≤ ∑ j, R.pw t m1 j * |R.v (t * 64 + j) e| := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq (Finset.sum_congr rfl fun j _ => ?_))
    rw [abs_mul, abs_of_nonneg (R.pw_nonneg t m1 j)]
  have key := online_step (α := α) (ε := epsA ρα) (Θ := Θ) (ρb := rbarA R.D X) (κ := kapA R.D X)
    (ηv := etavA) (ηA := etaAA) (t := (t : ℝ)) (x := a0 e) (x' := a' e) (X := R.As e m0 t)
    (Y := ∑ j, R.pw t m1 j * R.v (t * 64 + j) e) (M := R.Ms e m0 t)
    (Z := ∑ j, R.pw t m1 j * |R.v (t * 64 + j) e|) (L := R.Ls m0 t) (R := ∑ j, R.pw t m1 j)
    (V := R.Vs e t) (W := ∑ j : Fin 64, (if R.ok (t * 64 + j) = true then |R.v (t * 64 + j) e| else 0))
    hα0 hα1 hε0 hΘ0 hr0 hP.kapA_nonneg (by unfold etavA eta16 r64; norm_num)
    (by unfold etaAA; norm_num) (Nat.cast_nonneg t) (R.Ms_nonneg e m0 t) hZ0 hLs0 hP0
    (R.Vs_nonneg e t) hW0 (R.abs_As_le e m0 t) hYZ (ha e he) hstep
  have := key.1
  rw [hAs e, hMs e, hLs, R.Vs_succ e t]
  push_cast
  simpa [add_comm, add_left_comm, add_assoc] using this

/-- `ln 2 < 1`. -/
theorem log_two_lt_one : Real.log 2 < 1 := by
  have := Real.log_two_lt_d9; linarith

/-- **One trip of the reference.** The new reference `m1 ≥ m0` is within `Bm X`, at least every
attended score of the block up to `ξ`, and equal to the old one or within `ξ` of an attended score
of the block. -/
theorem trip_m {R : RowCtx} {X : ℝ} (hP : Par R.D X) {t : ℕ} (ht : t < 1024) {m0 m1 mf : ℝ}
    (hM : InvM R X t m0 mf) (hmm : m0 ≤ m1) (hm1 : |m1| ≤ Bm X)
    (hupT : ∀ j : Fin 64, R.ok (t * 64 + j) = true → R.s (t * 64 + j) ≤ (m1 + xi R.D X) * Real.log 2)
    (hwit : (1 ≤ t ∧ m1 = m0) ∨ ∃ j : Fin 64, R.ok (t * 64 + j) = true ∧
      (m1 - xi R.D X) * Real.log 2 ≤ R.s (t * 64 + j)) :
    InvM R X (t + 1) m1 mf := by
  have hlog0 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hxi0 := hP.xi_nonneg
  have hxi1 := hP.xi_le
  have hsucc := R.Ls_succ m0 m1 t
  have hE : Real.exp ((m0 - m1) * Real.log 2) ≤ 1 :=
    Real.exp_le_one_iff.mpr (by nlinarith)
  have hLs0 := R.Ls_nonneg m0 t
  have hpw : ∀ j, R.pw t m1 j ≤ 2 := by
    intro j
    unfold RowCtx.pw
    split_ifs with h
    · have := hupT j h
      have h2 : Real.exp (R.s (t * 64 + j) - m1 * Real.log 2) ≤ Real.exp (Real.log 2) :=
        Real.exp_le_exp.mpr (by nlinarith [log_two_lt_one])
      rwa [Real.exp_log (by norm_num)] at h2
    · norm_num
  refine ⟨by omega, hm1, hM.hmf, hM.hmm.trans hmm, ?_, ?_, ?_⟩
  · intro k hk hok
    by_cases hk' : k < 64 * t
    · have := hM.hup k hk' hok
      nlinarith
    · obtain ⟨j, hj⟩ : ∃ j : Fin 64, k = t * 64 + j :=
        ⟨⟨k - 64 * t, by omega⟩, by simp; omega⟩
      subst hj
      exact hupT j hok
  · intro _
    rw [hsucc]
    have hP0 : 0 ≤ ∑ j, R.pw t m1 j := Finset.sum_nonneg fun j _ => R.pw_nonneg t m1 j
    rcases hwit with ⟨ht1, hEq⟩ | ⟨j, hj, hs⟩
    · have h1 := hM.hlow ht1
      subst hEq
      simp only [sub_self, zero_mul, Real.exp_zero, one_mul]
      linarith
    · have hexp : 1 - xi R.D X ≤ R.pw t m1 j := by
        unfold RowCtx.pw
        rw [ite_eq_left hj]
        have h1 : -(xi R.D X * Real.log 2) + 1 ≤ Real.exp (-(xi R.D X * Real.log 2)) :=
          Real.add_one_le_exp _
        have h2 : Real.exp (-(xi R.D X * Real.log 2)) ≤ Real.exp (R.s (t * 64 + j) - m1 * Real.log 2) :=
          Real.exp_le_exp.mpr (by nlinarith)
        nlinarith [log_two_lt_one, mul_nonneg hxi0 (le_of_lt (sub_pos.mpr log_two_lt_one))]
      have h3 : R.pw t m1 j ≤ ∑ j, R.pw t m1 j :=
        Finset.single_le_sum (f := fun j => R.pw t m1 j) (fun j _ => R.pw_nonneg t m1 j) (Finset.mem_univ j)
      unfold wlow
      have := mul_nonneg (Real.exp_pos ((m0 - m1) * Real.log 2)).le hLs0
      linarith
  · rw [hsucc]
    have h1 : Real.exp ((m0 - m1) * Real.log 2) * R.Ls m0 t ≤ R.Ls m0 t := by
      nlinarith
    have h2 : ∑ j, R.pw t m1 j ≤ 64 * 2 := by
      calc ∑ j, R.pw t m1 j ≤ ∑ _j : Fin 64, (2 : ℝ) := Finset.sum_le_sum fun j _ => hpw j
        _ = 64 * 2 := by simp
    have := hM.hLup
    push_cast
    linarith

end

end Claims.AttnFwd
