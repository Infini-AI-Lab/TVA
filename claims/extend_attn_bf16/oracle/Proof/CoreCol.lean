import Proof.CoreRow
import Proof.CoreColArith

/-!
# The column invariant: one trip, and a trip that changes nothing

`colInv_step` is the induction step of the accumulator of one output component.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- A sum over `S ∪ V` of a quantity that is `q ·` (old term) on `S` and a new term on `V`. -/
theorem sum_split {S V : Finset ℕ} (hd : Disjoint S V) {X A B : ℕ → ℝ} {q : ℝ}
    (hS : ∀ j ∈ S, X j = q * A j) (hV : ∀ j ∈ V, X j = B j) :
    ∑ j ∈ S ∪ V, X j = q * ∑ j ∈ S, A j + ∑ j ∈ V, B j := by
  rw [Finset.sum_union hd, Finset.mul_sum]
  exact congrArg₂ _ (Finset.sum_congr rfl hS) (Finset.sum_congr rfl hV)

/-- The rescaling of the old weights: a number `q` with `0 ≤ q ≤ 1 + 2^-20`, equal to
`exp (m − m')·g` when something was seen, such that the new ghost's old weights are `q ·` the old
ones and `re_scale` is within `eps0` of it. -/
theorem old_scale {sc : ℕ → ℝ} {dl Xp : ℝ} {t : ℕ} {S V : Finset ℕ} {m l : ℝ} {G : Ghost}
    {s' δ' f p : ℕ → ℝ} {m' g r Ssum l' : ℝ} (hXp2 : Xp ≤ 2 ^ 13) (hdisj : Disjoint S V)
    (hrow : RowInv sc dl Xp t S m l G) (hm' : (S ∪ V).Nonempty → |m'| ≤ Xp)
    (hT : TripStat sc dl Xp S V m l s' δ' f p m' g r Ssum l') :
    ∃ q : ℝ, 0 ≤ q ∧ q ≤ 1 + 1 / 2 ^ 20 ∧
      (∀ j ∈ S, (G.step V s' δ' f m m' g).u j = q * G.u j) ∧ |r| ≤ 1 + 1 / 2 ^ 19 ∧
      (S.Nonempty → |r - q| ≤ eps0 ∧ |r| ≤ q + eps0) := by
  by_cases hS : S.Nonempty
  · obtain ⟨hg, hrq⟩ := hT.scale hS
    have hmS : |m| ≤ Xp := by
      obtain ⟨-, ⟨j0, hj0, hjm⟩, -⟩ := hrow.2.1 hS
      rw [← hjm]; exact (hrow.1 j0 hj0).2.1
    obtain ⟨hq0, hq1⟩ := scale_q hXp2 (hT.ref_ge.2 hS) hmS
      (hm' ⟨hS.choose, Finset.mem_union_left _ hS.choose_spec⟩) ⟨hg, hrq⟩
    refine ⟨Real.exp (m - m') * g, hq0, hq1, ?_, ?_, fun _ => ⟨hrq, ?_⟩⟩
    · intro j hj
      have : j ∉ V := Finset.disjoint_left.mp hdisj hj
      simp [Ghost.step, this, mul_assoc]
    · have := abs_sub_abs_le_abs_sub r (Real.exp (m - m') * g)
      rw [abs_of_nonneg hq0] at this
      have h19 : (1 : ℝ) / 2 ^ 20 + eps0 ≤ 1 / 2 ^ 19 := by unfold eps0; norm_num
      linarith
    · have := abs_sub_abs_le_abs_sub r (Real.exp (m - m') * g)
      rw [abs_of_nonneg hq0] at this
      linarith
  · refine ⟨0, le_rfl, by norm_num, ?_, ?_, fun h => absurd h hS⟩
    · intro j hj; exact absurd ⟨j, hj⟩ hS
    · have := hT.scale_bd
      have : (1 : ℝ) / 2 ^ 20 ≤ 1 / 2 ^ 19 := by norm_num
      linarith

set_option maxHeartbeats 1600000 in
/-- **One trip of the column invariant** (one output component, values `v`). `p16` is the
computed `p.to(bfloat16)` (`hround`: `p (1 + δ')` up to the absolute part of the rounding), and
`a'` the new accumulator (`hacc`: `r · a + Σ_V p16 · v`, with the errors of the accumulation).
The exponent `γ` grows to `γ + g1`: `g1` covers the accumulation's rate on the old accumulator
(`ε1`), `g2 ≤ γ` the once-only rate on the new weights (`ε2`). -/
theorem colInv_step {sc : ℕ → ℝ} {dl Xp : ℝ} {t : ℕ} {S V : Finset ℕ} {m l : ℝ} {G : Ghost}
    {s' δ' f p p16 v : ℕ → ℝ} {m' g r Ssum l' γ g1 ε1 ε2 a a' : ℝ}
    (hXp2 : Xp ≤ 2 ^ 13) (ht : t + 1 ≤ tmax) (hcard : (S ∪ V).card ≤ nmax)
    (hdisj : Disjoint S V) (hrow : RowInv sc dl Xp t S m l G)
    (hT : TripStat sc dl Xp S V m l s' δ' f p m' g r Ssum l')
    (hcol : ColInv v γ t S G a)
    (hround : ∀ j ∈ V, |p16 j - p j * (1 + δ' j)| ≤ etab)
    (hacc : |a' - (r * a + ∑ j ∈ V, p16 j * v j)| ≤
      ε1 * |r * a| + ε2 * ∑ j ∈ V, |p16 j| * |v j| + eps1 * (1 + ∑ j ∈ V, |v j|))
    (hε : 0 ≤ ε1 ∧ ε1 ≤ 1 / 2 ^ 15 ∧ 0 ≤ ε2 ∧ ε2 ≤ 1 / 2 ^ 15)
    (hg1 : 1 + ε1 ≤ Real.exp g1 ∧ ε1 * (1 + ub) ≤ expm1 g1 ∧ 0 ≤ g1)
    (hg2 : ε2 * (1 + ub) ≤ expm1 g2) (hγ : g2 ≤ γ) (hγ1 : γ ≤ 1 / 2 ^ 3) :
    ColInv v (γ + g1) (t + 1) (S ∪ V) (G.step V s' δ' f m m' g) a' := by
  obtain ⟨hε1, hε1', hε2, hε2'⟩ := hε
  obtain ⟨hg1a, hg1b, hg10⟩ := hg1
  have ht1 : t ≤ tmax := by omega
  have htr : (t : ℝ) ≤ 2 ^ 13 := by
    have : (t : ℝ) + 1 ≤ ((tmax : ℕ) : ℝ) := by exact_mod_cast ht
    simp only [tmax] at this
    push_cast at this
    calc (t : ℝ) ≤ 8192 := by linarith
      _ = 2 ^ 13 := by norm_num
  have hm' := hT.m'_bound hrow
  obtain ⟨q, hq0, hq1, hqu, hr, hrq⟩ := old_scale hXp2 hdisj hrow hm' hT
  -- the new weights
  have hw : ∀ j ∈ V, 0 ≤ Real.exp (s' j - m') * f j ∧ Real.exp (s' j - m') * f j ≤ 1 + 1 / 2 ^ 20 := by
    intro j hjV
    exact new_weight hXp2 (hm' ⟨j, Finset.mem_union_right _ hjV⟩) (hT.lanes j hjV).2.1
      (hT.ref_ge.1 j hjV) (hT.weight j hjV).1
  -- the old weights
  have hu : ∀ j ∈ S, 0 ≤ G.u j ∧ G.u j ≤ 1 + 1 / 2 ^ 7 := fun j hj => hrow.u_bound hXp2 (by omega) hj
  have hδ : ∀ j ∈ S, |G.δ j| ≤ ub := fun j hj => (hrow.1 j hj).2.2
  -- the sums of the new ghost
  have hsN : ∑ j ∈ S ∪ V, (G.step V s' δ' f m m' g).u j * (1 + (G.step V s' δ' f m m' g).δ j) * v j =
      q * ∑ j ∈ S, G.u j * (1 + G.δ j) * v j +
        ∑ j ∈ V, Real.exp (s' j - m') * f j * (1 + δ' j) * v j := by
    refine sum_split hdisj ?_ ?_
    · intro j hj
      have hnV : j ∉ V := Finset.disjoint_left.mp hdisj hj
      have h := hqu j hj
      simp only [Ghost.step, hnV, ite_false] at h ⊢
      rw [h]
      ring
    · intro j hj; simp [Ghost.step, hj]
  have hsR : ∑ j ∈ S ∪ V, (G.step V s' δ' f m m' g).u j * |v j| =
      q * ∑ j ∈ S, G.u j * |v j| + ∑ j ∈ V, Real.exp (s' j - m') * f j * |v j| := by
    refine sum_split hdisj ?_ ?_
    · intro j hj
      have hnV : j ∉ V := Finset.disjoint_left.mp hdisj hj
      have h := hqu j hj
      simp only [Ghost.step, hnV, ite_false] at h ⊢
      rw [h]
      ring
    · intro j hj; simp [Ghost.step, hj]
  have hsV : ∑ j ∈ S ∪ V, |v j| = 1 * ∑ j ∈ S, |v j| + ∑ j ∈ V, |v j| :=
    sum_split hdisj (fun j _ => by ring) (fun j _ => rfl)
  unfold ColInv at hcol ⊢
  rw [hsN, hsR, hsV]
  -- the wording
  set N := ∑ j ∈ S, G.u j * (1 + G.δ j) * v j with hNdef
  set R := ∑ j ∈ S, G.u j * |v j| with hRdef
  set VS := ∑ j ∈ S, |v j| with hVSdef
  set NV := ∑ j ∈ V, Real.exp (s' j - m') * f j * (1 + δ' j) * v j with hNVdef
  set RV := ∑ j ∈ V, Real.exp (s' j - m') * f j * |v j| with hRVdef
  set VV := ∑ j ∈ V, |v j| with hVVdef
  have hVS0 : 0 ≤ VS := Finset.sum_nonneg fun j _ => abs_nonneg _
  have hVV0 : 0 ≤ VV := Finset.sum_nonneg fun j _ => abs_nonneg _
  have hR0 : 0 ≤ R := Finset.sum_nonneg fun j hj => mul_nonneg (hu j hj).1 (abs_nonneg _)
  have hR1 : R ≤ 2 * VS := by
    rw [hVSdef, Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by
      have := (hu j hj).2
      nlinarith [abs_nonneg (v j)]
  have hRV0 : 0 ≤ RV := Finset.sum_nonneg fun j hj => mul_nonneg (hw j hj).1 (abs_nonneg _)
  have hRV1 : RV ≤ 2 * VV := by
    rw [hVVdef, Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by
      have := (hw j hj).2
      nlinarith [abs_nonneg (v j)]
  have hN : |N| ≤ (1 + ub) * R := by
    rw [hNdef, hRdef, Finset.mul_sum]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j hj => ?_)
    have h1 := (hu j hj).1
    have h2 : |1 + G.δ j| ≤ 1 + ub := by
      have := abs_le.mp (hδ j hj)
      rw [abs_le]; constructor <;> linarith
    rw [abs_mul, abs_mul, abs_of_nonneg h1]
    nlinarith [abs_nonneg (v j), mul_le_mul_of_nonneg_left h2 h1]
  have hS0 : ¬ S.Nonempty → R = 0 ∧ N = 0 := by
    intro h
    have : S = ∅ := Finset.not_nonempty_iff_eq_empty.mp h
    subst this
    simp [hRdef, hNdef]
  have hrR : |r| * R ≤ (q + eps0) * R := by
    by_cases hS : S.Nonempty
    · exact mul_le_mul_of_nonneg_right (hrq hS).2 hR0
    · rw [(hS0 hS).1]; simp
  have hrq' : |r - q| * |N| ≤ eps0 * |N| := by
    by_cases hS : S.Nonempty
    · exact mul_le_mul_of_nonneg_right (hrq hS).1 (abs_nonneg _)
    · rw [(hS0 hS).2]; simp
  -- the new lanes
  have hc3 : ∀ j ∈ V, |p16 j - Real.exp (s' j - m') * f j * (1 + δ' j)| ≤ etab + eps0 * (1 + ub) := by
    intro j hj
    have h1 := hround j hj
    have h2 := (hT.weight j hj).2
    have hδ' : |1 + δ' j| ≤ 1 + ub := by
      have := abs_le.mp (hT.lanes j hj).2.2
      rw [abs_le]; constructor <;> linarith
    have e : p16 j - Real.exp (s' j - m') * f j * (1 + δ' j) =
        (p16 j - p j * (1 + δ' j)) + (p j - Real.exp (s' j - m') * f j) * (1 + δ' j) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul]
    have := mul_le_mul h2 hδ' (abs_nonneg _) (by unfold eps0; norm_num)
    linarith
  have hPA : ∑ j ∈ V, |p16 j| * |v j| ≤ (1 + ub) * RV + (etab + eps0 * (1 + ub)) * VV := by
    rw [hRVdef, hVVdef, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun j hj => ?_
    have h1 := hc3 j hj
    have h2 : |p16 j| ≤ (1 + ub) * (Real.exp (s' j - m') * f j) + (etab + eps0 * (1 + ub)) := by
      have h3 := abs_sub_abs_le_abs_sub (p16 j) (Real.exp (s' j - m') * f j * (1 + δ' j))
      have h4 : |Real.exp (s' j - m') * f j * (1 + δ' j)| ≤ (1 + ub) * (Real.exp (s' j - m') * f j) := by
        rw [abs_mul, abs_of_nonneg (hw j hj).1]
        have hδ' : |1 + δ' j| ≤ 1 + ub := by
          have := abs_le.mp (hT.lanes j hj).2.2
          rw [abs_le]; constructor <;> linarith
        nlinarith [(hw j hj).1]
      linarith
    nlinarith [abs_nonneg (v j), mul_le_mul_of_nonneg_right h2 (abs_nonneg (v j))]
  have hP : |∑ j ∈ V, p16 j * v j - NV| ≤ (etab + eps0 * (1 + ub)) * VV := by
    rw [hNVdef, hVVdef, ← Finset.sum_sub_distrib, Finset.mul_sum]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j hj => ?_)
    have e : p16 j * v j - Real.exp (s' j - m') * f j * (1 + δ' j) * v j =
        (p16 j - Real.exp (s' j - m') * f j * (1 + δ' j)) * v j := by ring
    rw [e, abs_mul]
    exact mul_le_mul_of_nonneg_right (hc3 j hj) (abs_nonneg _)
  -- the exponents
  have hg20 : 0 ≤ g2 := by
    have : 0 ≤ expm1 g2 := (mul_nonneg hε2 (by unfold ub; norm_num)).trans hg2
    unfold expm1 at this
    exact Real.one_le_exp_iff.mp (by linarith)
  have hγ0 : 0 ≤ γ := hg20.trans hγ
  have hΓ : 0 ≤ expm1 γ := expm1_nonneg hγ0
  have hΓ1 : expm1 γ ≤ 1 / 6 := by
    have := exp_le_quad hγ0 (by linarith)
    unfold expm1
    nlinarith
  have hΓ' : (1 + ε1) * expm1 γ + ε1 * (1 + ub) ≤ expm1 (γ + g1) := by
    have e : expm1 (γ + g1) = Real.exp g1 * expm1 γ + expm1 g1 := by
      unfold expm1; rw [Real.exp_add]; ring
    rw [e]
    nlinarith [mul_le_mul_of_nonneg_right hg1a hΓ]
  have hΓ'2 : ε2 * (1 + ub) ≤ expm1 (γ + g1) :=
    hg2.trans ((expm1_mono hγ).trans (expm1_mono (by linarith)))
  have hτ0 : 0 ≤ (t : ℝ) * t0 := mul_nonneg (Nat.cast_nonneg t) (by unfold t0; norm_num)
  have hτ : (t : ℝ) * t0 ≤ 2 ^ 13 * t0 := mul_le_mul_of_nonneg_right htr (by unfold t0; norm_num)
  have hcore := col_core (a := a) (a' := a') (r := r) (q := q) (N := N) (R := R) (VS := VS)
    (NV := NV) (RV := RV) (VV := VV) (P := ∑ j ∈ V, p16 j * v j)
    (PA := ∑ j ∈ V, |p16 j| * |v j|) (Γ := expm1 γ) (Γ' := expm1 (γ + g1)) (ε1 := ε1) (ε2 := ε2)
    (τ := (t : ℝ) * t0) hΓ hΓ1 hε1 hε1' hε2 hε2' hq0 hq1 hr hR0 hR1 hVS0 hRV0 hRV1 hVV0 hrR hrq'
    hN hcol hacc hPA hP hΓ' hΓ'2 hτ0 hτ
  push_cast
  linarith [hcore]

/-- A trip count or an exponent can be raised. -/
theorem colInv_mono {v : ℕ → ℝ} {γ γ' : ℝ} {t t' : ℕ} {S : Finset ℕ} {G : Ghost} {a : ℝ}
    (hu : ∀ j ∈ S, 0 ≤ G.u j) (htt : t ≤ t') (hγγ : γ ≤ γ') (hcol : ColInv v γ t S G a) :
    ColInv v γ' t' S G a := by
  unfold ColInv at hcol ⊢
  refine hcol.trans ?_
  have h1 : expm1 γ * ∑ j ∈ S, G.u j * |v j| ≤ expm1 γ' * ∑ j ∈ S, G.u j * |v j| :=
    mul_le_mul_of_nonneg_right (expm1_mono hγγ)
      (Finset.sum_nonneg fun j hj => mul_nonneg (hu j hj) (abs_nonneg _))
  have htt' : (t : ℝ) ≤ t' := by exact_mod_cast htt
  have h2 : (t : ℝ) * t0 * (1 + ∑ j ∈ S, |v j|) ≤ t' * t0 * (1 + ∑ j ∈ S, |v j|) := by
    have : 0 ≤ 1 + ∑ j ∈ S, |v j| := by
      have := Finset.sum_nonneg (fun j (_ : j ∈ S) => abs_nonneg (v j)); linarith
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right htt' (by unfold t0; norm_num)) this
  linarith

end

end Claims.ExtendAttn
