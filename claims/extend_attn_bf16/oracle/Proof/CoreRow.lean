import Proof.CoreLemmas

/-!
# The row invariant: one trip, and a trip that changes nothing

`rowInv_step` is the induction step of the online softmax's `deno` and weights: one trip, from the
float glue's facts (`TripStat`), keeps `RowInv`. `rowInv_mono` raises the trip count (a skipped
tile).
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-! ## Weights are at most `1.01`, and nonnegative -/

/-- Each seen weight is positive and at most `1 + 2^-7`, when the scores are at most `Xp ≤ 2^13`
in magnitude. -/
theorem RowInv.u_bound {sc : ℕ → ℝ} {dl Xp : ℝ} {t : ℕ} {S : Finset ℕ} {m l : ℝ} {G : Ghost}
    (hXp : Xp ≤ 2 ^ 13) (ht : t ≤ tmax) (hinv : RowInv sc dl Xp t S m l G) {j : ℕ} (hj : j ∈ S) :
    0 ≤ G.u j ∧ G.u j ≤ 1 + 1 / 2 ^ 7 := by
  obtain ⟨hL, hM, -, -⟩ := hinv
  obtain ⟨hle, ⟨j0, hj0, hjm⟩, hF⟩ := hM ⟨j, hj⟩
  obtain ⟨F, hF1, hu⟩ := hF j hj
  have htr : (t : ℝ) ≤ 2 ^ 13 := by exact_mod_cast ht
  have hgap : m - G.s j ≤ 2 ^ 15 := by
    have h1 := (hL j0 hj0).2.1
    have h2 := (hL j hj).2.1
    have := abs_le.mp h1
    have := abs_le.mp h2
    linarith
  have hgap0 : 0 ≤ m - G.s j := by linarith [hle j hj]
  have hs := expo_small htr (Nat.cast_nonneg t) hgap hgap0
  have hFg := fac_ge (by linarith) hF1
  rw [hu]
  refine ⟨mul_nonneg (Real.exp_pos _).le (by linarith), ?_⟩
  exact (weight_le (hle j hj) hF1).trans (exp_te1_le htr (Nat.cast_nonneg t))

/-- The sum of the seen weights is nonnegative and at most `2^18 · (1 + 2^-7)`. -/
theorem RowInv.sum_u_bound {sc : ℕ → ℝ} {dl Xp : ℝ} {t : ℕ} {S : Finset ℕ} {m l : ℝ} {G : Ghost}
    (hXp : Xp ≤ 2 ^ 13) (ht : t ≤ tmax) (hcard : S.card ≤ nmax)
    (hinv : RowInv sc dl Xp t S m l G) :
    0 ≤ ∑ j ∈ S, G.u j ∧ ∑ j ∈ S, G.u j ≤ 2 ^ 18 * (1 + 1 / 2 ^ 7) := by
  refine ⟨Finset.sum_nonneg fun j hj => (hinv.u_bound hXp ht hj).1, ?_⟩
  calc ∑ j ∈ S, G.u j ≤ ∑ _j ∈ S, (1 + 1 / 2 ^ 7 : ℝ) :=
        Finset.sum_le_sum fun j hj => (hinv.u_bound hXp ht hj).2
    _ = S.card * (1 + 1 / 2 ^ 7) := by simp; ring
    _ ≤ 2 ^ 18 * (1 + 1 / 2 ^ 7) := by
        have : (S.card : ℝ) ≤ 2 ^ 18 := by exact_mod_cast hcard
        nlinarith

/-! ## The sum of the new weights -/

theorem step_sum_u {S V : Finset ℕ} {G : Ghost} {s' δ' f : ℕ → ℝ} {m m' g : ℝ}
    (hd : Disjoint S V) :
    ∑ j ∈ S ∪ V, (G.step V s' δ' f m m' g).u j =
      Real.exp (m - m') * g * ∑ j ∈ S, G.u j + ∑ j ∈ V, Real.exp (s' j - m') * f j := by
  rw [Finset.sum_union hd]
  congr 1
  · rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j hj => ?_
    have : j ∉ V := Finset.disjoint_left.mp hd hj
    simp [Ghost.step, this]
  · exact Finset.sum_congr rfl fun j hj => by simp [Ghost.step, hj]

/-! ## The `deno` update in the reals -/

/-- **The `deno` arithmetic.** `R` stands for the rescaled old `deno`, `X1` for the rescaled old
weights' sum, `Ssum` for the computed sum of the new lanes and `W` for the new weights' sum. -/
theorem deno_core {X1 W Λ L0 lam R Ssum l' cR cS e0 : ℝ} (hX1 : 0 ≤ X1) (hW : 0 ≤ W)
    (hL0 : 0 ≤ L0) (hLL : L0 ≤ Λ) (hlam : 3 * uf ≤ lam)
    (hR : |R - X1| ≤ Λ * X1 + cR) (hS : |Ssum - W| ≤ L0 * W + cS)
    (hl : |l' - (R + Ssum)| ≤ 3 * uf * (|R| + |Ssum|) + e0) :
    |l' - (X1 + W)| ≤ (Λ + lam * (1 + Λ)) * (X1 + W) + ((1 + 3 * uf) * (cR + cS) + e0) := by
  have hu := uf_pos
  have h1 : |R| ≤ X1 + |R - X1| := by
    have := abs_add_le (R - X1) X1; rw [sub_add_cancel, abs_of_nonneg hX1] at this; linarith
  have h2 : |Ssum| ≤ W + |Ssum - W| := by
    have := abs_add_le (Ssum - W) W; rw [sub_add_cancel, abs_of_nonneg hW] at this; linarith
  have h3 : |l' - (X1 + W)| ≤ |l' - (R + Ssum)| + |R - X1| + |Ssum - W| := by
    have e : l' - (X1 + W) = (l' - (R + Ssum)) + (R - X1) + (Ssum - W) := by ring
    rw [e]
    have a1 := abs_add_le ((l' - (R + Ssum)) + (R - X1)) (Ssum - W)
    have a2 := abs_add_le (l' - (R + Ssum)) (R - X1)
    linarith
  have hΛ : 0 ≤ Λ := hL0.trans hLL
  nlinarith [mul_nonneg hX1 hΛ, mul_nonneg hW hL0, mul_nonneg hX1 hu.le, mul_nonneg hW hu.le,
    mul_nonneg hX1 (sub_nonneg.mpr hlam), mul_nonneg hW (sub_nonneg.mpr hlam),
    mul_nonneg (mul_nonneg hX1 hΛ) hu.le, mul_nonneg (mul_nonneg hW hL0) hu.le,
    mul_nonneg hW (sub_nonneg.mpr hLL), mul_nonneg (mul_nonneg hW hu.le) (sub_nonneg.mpr hLL),
    mul_nonneg (mul_nonneg hW (sub_nonneg.mpr hlam)) hΛ,
    mul_nonneg (mul_nonneg hX1 (sub_nonneg.mpr hlam)) hΛ,
    mul_nonneg (mul_nonneg hW (sub_nonneg.mpr hlam)) hL0,
    abs_nonneg (R - X1), abs_nonneg (Ssum - W)]

/-! ## One trip -/

/-- **The `deno` of a row that has seen nothing**: it stays within `t · a0` of `0`. -/
theorem deno_empty {t l r Ssum l' : ℝ} (ht0 : 0 ≤ t) (ht : t ≤ 2 ^ 13) (hl : |l| ≤ t * a0)
    (hr : |r| ≤ 1 + 1 / 2 ^ 20) (hS : |Ssum| ≤ eps0)
    (hd : |l' - (r * l + Ssum)| ≤ 3 * uf * (|r * l| + |Ssum|) + eps0) :
    |l'| ≤ (t + 1) * a0 := by
  have hta : t * a0 ≤ 2 ^ 13 * a0 := mul_le_mul_of_nonneg_right ht (by unfold a0; norm_num)
  have hta0 : 0 ≤ t * a0 := mul_nonneg ht0 (by unfold a0; norm_num)
  have hrl : |r * l| ≤ (1 + 1 / 2 ^ 20) * (t * a0) := by
    rw [abs_mul]
    exact mul_le_mul hr hl (abs_nonneg _) (by norm_num)
  have h1 : |l'| ≤ |r * l + Ssum| + (3 * uf * (|r * l| + |Ssum|) + eps0) := by
    have := abs_add_le (l' - (r * l + Ssum)) (r * l + Ssum)
    rw [sub_add_cancel] at this
    linarith
  have h2 := abs_add_le (r * l) Ssum
  have hu0 : 0 ≤ 3 * uf := by unfold uf; norm_num
  have h3 : |l'| ≤ (1 + 3 * uf) * ((1 + 1 / 2 ^ 20) * (t * a0) + eps0) + eps0 := by
    nlinarith [abs_nonneg (r * l), abs_nonneg Ssum, mul_nonneg hu0 (abs_nonneg (r * l)),
      mul_nonneg hu0 (abs_nonneg Ssum)]
  refine h3.trans ?_
  have hk : (1 + 3 * uf) * (1 + 1 / 2 ^ 20) ≤ 1 + 1 / 2 ^ 19 := by unfold uf; norm_num
  have h4 : (1 + 3 * uf) * ((1 + 1 / 2 ^ 20) * (t * a0)) ≤ (1 + 1 / 2 ^ 19) * (t * a0) := by
    calc (1 + 3 * uf) * ((1 + 1 / 2 ^ 20) * (t * a0))
        = ((1 + 3 * uf) * (1 + 1 / 2 ^ 20)) * (t * a0) := by ring
      _ ≤ (1 + 1 / 2 ^ 19) * (t * a0) := mul_le_mul_of_nonneg_right hk hta0
  have h5 : (1 + 1 / 2 ^ 19) * (t * a0) ≤ t * a0 + 1 / 2 ^ 6 * a0 := by
    unfold a0 at *; nlinarith
  have h6 : (1 + 3 * uf) * eps0 + eps0 ≤ 1 / 2 * a0 := by unfold uf eps0 a0; norm_num
  nlinarith

/-- The rescaling `q = exp (m − m')·g` of the old weights is nonnegative, at most `exp e1`, and
within `eps0` of the computed `re_scale` (for a nonempty old state). -/
theorem scale_q {m m' g r Xp : ℝ} (hXp2 : Xp ≤ 2 ^ 13) (hmm : m ≤ m') (hm : |m| ≤ Xp)
    (hm' : |m'| ≤ Xp) (hsc : |g - 1| ≤ expm1 (e1 + e2 * (m' - m)) ∧ |r - Real.exp (m - m') * g| ≤ eps0) :
    0 ≤ Real.exp (m - m') * g ∧ Real.exp (m - m') * g ≤ 1 + 1 / 2 ^ 20 := by
  obtain ⟨hg, -⟩ := hsc
  have hgap : m' - m ≤ 2 ^ 15 := by
    have := abs_le.mp hm; have := abs_le.mp hm'; linarith
  have hs := expo_small (t := 1) (by norm_num) (by norm_num) hgap (by linarith)
  have hg1 : e1 + e2 * (m' - m) ≤ 1 / 4 := by
    have : (1 : ℝ) * e1 = e1 := one_mul _
    linarith
  have h0 : 2 / 3 ≤ g := fac_ge hg1 hg
  have h1 := fac_le hg
  refine ⟨mul_nonneg (Real.exp_pos _).le (by linarith), ?_⟩
  calc Real.exp (m - m') * g ≤ Real.exp (m - m') * Real.exp (e1 + e2 * (m' - m)) :=
        mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le
    _ = Real.exp (e1 - (1 - e2) * (m' - m)) := by rw [← Real.exp_add]; congr 1; ring
    _ ≤ Real.exp e1 := by
        apply Real.exp_le_exp.mpr
        have : 0 ≤ (1 - e2) * (m' - m) := mul_nonneg (by unfold e2; norm_num) (by linarith)
        linarith
    _ ≤ 1 + 1 / 2 ^ 20 := exp_e1_le

/-- A seen state's reference is at most `Xp` in magnitude. -/
theorem RowInv.m_bound {sc : ℕ → ℝ} {dl Xp : ℝ} {t : ℕ} {S : Finset ℕ} {m l : ℝ} {G : Ghost}
    (hinv : RowInv sc dl Xp t S m l G) (hS : S.Nonempty) : |m| ≤ Xp := by
  obtain ⟨-, ⟨j0, hj0, hjm⟩, -⟩ := hinv.2.1 hS
  rw [← hjm]; exact (hinv.1 j0 hj0).2.1

/-- The new reference is at most `Xp` in magnitude. -/
theorem TripStat.m'_bound {sc : ℕ → ℝ} {dl Xp : ℝ} {t : ℕ} {S V : Finset ℕ} {m l : ℝ} {G : Ghost}
    {s' δ' f p : ℕ → ℝ} {m' g r Ssum l' : ℝ} (hinv : RowInv sc dl Xp t S m l G)
    (hT : TripStat sc dl Xp S V m l s' δ' f p m' g r Ssum l') (hne : (S ∪ V).Nonempty) :
    |m'| ≤ Xp := by
  rcases hT.ref_att hne with ⟨hS, hmm⟩ | ⟨j, hjV, hj⟩
  · rw [hmm]; exact hinv.m_bound hS
  · rw [← hj]; exact (hT.lanes j hjV).2.1

/-- A new weight `exp (ŝ − m')·f` is nonnegative and at most `1 + 2^-20`. -/
theorem new_weight {Xp sj m' f : ℝ} (hXp2 : Xp ≤ 2 ^ 13) (hm' : |m'| ≤ Xp) (hs : |sj| ≤ Xp)
    (hsm : sj ≤ m') (hf : |f - 1| ≤ expm1 (e1 + e2 * (m' - sj))) :
    0 ≤ Real.exp (sj - m') * f ∧ Real.exp (sj - m') * f ≤ 1 + 1 / 2 ^ 20 := by
  have hgap : m' - sj ≤ 2 ^ 15 := by
    have := abs_le.mp hm'; have := abs_le.mp hs; linarith
  have hs1 := expo_small (t := 1) (by norm_num) (by norm_num) hgap (by linarith)
  have hf0 := fac_ge (by have : (1 : ℝ) * e1 = e1 := one_mul _; linarith) hf
  refine ⟨mul_nonneg (Real.exp_pos _).le (by linarith), ?_⟩
  calc Real.exp (sj - m') * f ≤ Real.exp (sj - m') * Real.exp (e1 + e2 * (m' - sj)) :=
        mul_le_mul_of_nonneg_left (fac_le hf) (Real.exp_pos _).le
    _ = Real.exp (e1 - (1 - e2) * (m' - sj)) := by rw [← Real.exp_add]; congr 1; ring
    _ ≤ Real.exp e1 := by
        apply Real.exp_le_exp.mpr
        have : 0 ≤ (1 - e2) * (m' - sj) := mul_nonneg (by unfold e2; norm_num) (by linarith)
        linarith
    _ ≤ 1 + 1 / 2 ^ 20 := exp_e1_le

set_option maxHeartbeats 1600000 in
/-- **One trip of the row invariant.** The trip `TripStat` (the float glue's facts) takes the state
that has seen `S` after `t` trips to the one that has seen `S ∪ V` after `t + 1`, with the ghost
`G.step`. -/
theorem rowInv_step {sc : ℕ → ℝ} {dl Xp : ℝ} {t : ℕ} {S V : Finset ℕ} {m l : ℝ} {G : Ghost}
    {s' δ' f p : ℕ → ℝ} {m' g r Ssum l' : ℝ}
    (_hdl : 0 ≤ dl) (_hXp : 0 ≤ Xp) (hXp2 : Xp ≤ 2 ^ 13) (ht : t + 1 ≤ tmax)
    (hcard : (S ∪ V).card ≤ nmax)
    (hV64 : V.card ≤ 64) (hdisj : Disjoint S V) (hinv : RowInv sc dl Xp t S m l G)
    (hT : TripStat sc dl Xp S V m l s' δ' f p m' g r Ssum l') :
    RowInv sc dl Xp (t + 1) (S ∪ V) m' l' (G.step V s' δ' f m m' g) := by
  have hinv0 := hinv
  obtain ⟨hL, hM, hE, hl⟩ := hinv
  have ht1 : t ≤ tmax := by omega
  have htr : (t : ℝ) + 1 ≤ 2 ^ 13 := by
    have : ((t + 1 : ℕ) : ℝ) ≤ ((tmax : ℕ) : ℝ) := by exact_mod_cast ht
    simp only [tmax] at this
    push_cast at this
    norm_num at this ⊢
    exact this
  have htr' : (t : ℝ) ≤ 2 ^ 13 := t_le_real ht1
  have hmS : S.Nonempty → |m| ≤ Xp := by
    intro hS
    obtain ⟨-, ⟨j0, hj0, hjm⟩, -⟩ := hM hS
    rw [← hjm]; exact (hL j0 hj0).2.1
  have hm' : (S ∪ V).Nonempty → |m'| ≤ Xp := by
    intro hne
    rcases hT.ref_att hne with ⟨hS, hmm⟩ | ⟨j, hjV, hj⟩
    · rw [hmm]; exact hmS hS
    · rw [← hj]; exact (hT.lanes j hjV).2.1
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- the lanes
    intro j hj
    rcases Finset.mem_union.mp hj with hjS | hjV
    · have hnV : j ∉ V := Finset.disjoint_left.mp hdisj hjS
      simpa [Ghost.step, hnV] using hL j hjS
    · simpa [Ghost.step, hjV] using hT.lanes j hjV
  · -- the reference and the factors
    intro hne
    refine ⟨?_, ?_, ?_⟩
    · intro j hj
      rcases Finset.mem_union.mp hj with hjS | hjV
      · have hnV : j ∉ V := Finset.disjoint_left.mp hdisj hjS
        have h1 := (hM ⟨j, hjS⟩).1 j hjS
        have h2 := hT.ref_ge.2 ⟨j, hjS⟩
        simp only [Ghost.step, hnV, ite_false]
        linarith
      · simp only [Ghost.step, hjV, ite_true]
        exact hT.ref_ge.1 j hjV
    · rcases hT.ref_att hne with ⟨hS, hmm⟩ | ⟨j, hjV, hj⟩
      · obtain ⟨j, hj, hjm⟩ := (hM hS).2.1
        have hnV : j ∉ V := Finset.disjoint_left.mp hdisj hj
        exact ⟨j, Finset.mem_union_left _ hj, by simp [Ghost.step, hnV, hjm, hmm]⟩
      · exact ⟨j, Finset.mem_union_right _ hjV, by simp [Ghost.step, hjV, hj]⟩
    · intro j hj
      rcases Finset.mem_union.mp hj with hjS | hjV
      · have hnV : j ∉ V := Finset.disjoint_left.mp hdisj hjS
        have hSne : S.Nonempty := ⟨j, hjS⟩
        obtain ⟨F, hF, hu⟩ := (hM hSne).2.2 j hjS
        obtain ⟨hg, -⟩ := hT.scale hSne
        have hmm := hT.ref_ge.2 hSne
        have hsj := (hM hSne).1 j hjS
        refine ⟨F * g, ?_, ?_⟩
        · have := fac_mul (A := t * e1 + e2 * (m - G.s j)) (B := e1 + e2 * (m' - m))
            (by nlinarith [e1_pos, Nat.cast_nonneg (α := ℝ) t, mul_nonneg e2_pos.le (sub_nonneg.mpr hsj),
              mul_nonneg (Nat.cast_nonneg (α := ℝ) t) e1_pos.le])
            (by nlinarith [e1_pos, mul_nonneg e2_pos.le (sub_nonneg.mpr hmm)]) hF hg
          refine this.trans (le_of_eq ?_)
          simp only [Ghost.step, hnV, ite_false]
          unfold expm1; congr 2; push_cast; ring
        · simp only [Ghost.step, hnV, ite_false]
          rw [hu]
          have : Real.exp (m - m') * g * (Real.exp (G.s j - m) * F) =
              Real.exp (G.s j - m') * (F * g) := by
            have : Real.exp (G.s j - m') = Real.exp (m - m') * Real.exp (G.s j - m) := by
              rw [← Real.exp_add]; congr 1; ring
            rw [this]; ring
          exact this
      · refine ⟨f j, ?_, ?_⟩
        · refine (hT.weight j hjV).1.trans (expm1_mono ?_)
          simp only [Ghost.step, hjV, ite_true]
          push_cast
          nlinarith [Nat.cast_nonneg (α := ℝ) t, e1_pos]
        · simp [Ghost.step, hjV]
  · -- an empty state
    intro h
    have h' := Finset.union_eq_empty.mp h
    obtain ⟨hS0, hV0⟩ := h'
    have hsum := hT.sum
    rw [hV0] at hsum
    simp only [Finset.sum_empty, sub_zero, mul_zero, zero_add] at hsum
    have := deno_empty (Nat.cast_nonneg t) htr' (hE hS0) hT.scale_bd (by simpa using hsum) hT.deno
    push_cast
    exact this

  · -- the deno
    have hSc : S.card ≤ nmax := (Finset.card_le_card Finset.subset_union_left).trans hcard
    obtain ⟨hU0, hUb⟩ := hinv0.sum_u_bound hXp2 ht1 hSc
    have hsum := step_sum_u (G := G) (s' := s') (δ' := δ') (f := f) (m := m) (m' := m') (g := g) hdisj
    rw [hsum]
    -- the new weights
    have hwj : ∀ j ∈ V, 0 ≤ Real.exp (s' j - m') * f j := by
      intro j hjV
      have hne : (S ∪ V).Nonempty := ⟨j, Finset.mem_union_right _ hjV⟩
      exact (new_weight hXp2 (hm' hne) (hT.lanes j hjV).2.1 (hT.ref_ge.1 j hjV) (hT.weight j hjV).1).1
    set W := ∑ j ∈ V, Real.exp (s' j - m') * f j with hW
    have hW0 : 0 ≤ W := Finset.sum_nonneg hwj
    have hcard' : (V.card : ℝ) ≤ 64 := by exact_mod_cast hV64
    have hPsum : |∑ j ∈ V, p j - W| ≤ 64 * eps0 := by
      rw [hW, ← Finset.sum_sub_distrib]
      refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
      calc ∑ j ∈ V, |p j - Real.exp (s' j - m') * f j| ≤ ∑ _j ∈ V, eps0 :=
            Finset.sum_le_sum fun j hj => (hT.weight j hj).2
        _ = V.card * eps0 := by simp
        _ ≤ 64 * eps0 := by unfold eps0; nlinarith
    have hPabs : ∑ j ∈ V, |p j| ≤ W + 64 * eps0 := by
      calc ∑ j ∈ V, |p j| ≤ ∑ j ∈ V, (Real.exp (s' j - m') * f j + eps0) := by
            refine Finset.sum_le_sum fun j hj => ?_
            have h1 := (hT.weight j hj).2
            have h2 := hwj j hj
            have := abs_sub_abs_le_abs_sub (p j) (Real.exp (s' j - m') * f j)
            rw [abs_of_nonneg h2] at this
            linarith
        _ = W + V.card * eps0 := by rw [Finset.sum_add_distrib, ← hW]; simp
        _ ≤ W + 64 * eps0 := by unfold eps0; nlinarith
    -- constants
    have hL0 : expm1 lam0 ≤ 1 / 2 ^ 20 := by
      have := exp_le_quad (x := lam0) (by unfold lam0; norm_num) (by unfold lam0; norm_num)
      unfold expm1; unfold lam0 at this ⊢; norm_num at this ⊢; linarith
    have hL00 : 0 ≤ expm1 lam0 := expm1_nonneg (by unfold lam0; norm_num)
    have hΛ0 : 0 ≤ expm1 (lam0 + lam1 * t) :=
      expm1_nonneg (by have := Nat.cast_nonneg (α := ℝ) t; unfold lam0 lam1; positivity)
    have hΛ1 : expm1 (lam0 + lam1 * t) ≤ 1 / 100 := by
      have hx : lam0 + lam1 * t ≤ 1 / 2 ^ 9 := by
        have : (t : ℝ) ≤ 2 ^ 13 := by linarith
        have h2 : lam1 * t ≤ lam1 * 2 ^ 13 := mul_le_mul_of_nonneg_left this (by unfold lam1; norm_num)
        unfold lam0 lam1 at *; norm_num at h2 ⊢; linarith
      have := exp_le_quad (x := lam0 + lam1 * t) (by unfold lam0 lam1; positivity) (by linarith)
      unfold expm1
      have h3 : (lam0 + lam1 * t) ^ 2 ≤ (1 / 2 ^ 9) ^ 2 :=
        pow_le_pow_left₀ (by unfold lam0 lam1; positivity) hx 2
      norm_num at h3 hx this ⊢
      nlinarith
    have hLL : expm1 lam0 ≤ expm1 (lam0 + lam1 * t) :=
      expm1_mono (by have := Nat.cast_nonneg (α := ℝ) t; unfold lam1; nlinarith)
    have hlam : 3 * uf ≤ expm1 lam1 := by
      have := le_expm1 lam1; unfold lam1 uf at *; norm_num at *; linarith
    have hΛ' : expm1 (lam0 + lam1 * ((t + 1 : ℕ) : ℝ)) =
        expm1 (lam0 + lam1 * t) + expm1 lam1 * (1 + expm1 (lam0 + lam1 * t)) := by
      unfold expm1
      have : lam0 + lam1 * ((t + 1 : ℕ) : ℝ) = (lam0 + lam1 * t) + lam1 := by push_cast; ring
      rw [this, Real.exp_add]; ring
    -- the rescaled old deno
    have hR : ∃ q cR : ℝ, 0 ≤ q * ∑ j ∈ S, G.u j ∧
        Real.exp (m - m') * g * ∑ j ∈ S, G.u j = q * ∑ j ∈ S, G.u j ∧
        |r * l - q * ∑ j ∈ S, G.u j| ≤
          expm1 (lam0 + lam1 * t) * (q * ∑ j ∈ S, G.u j) + cR ∧
        cR ≤ (1 + 1 / 2 ^ 19) * (t * a0) + 1 / 2 ^ 112 := by
      by_cases hS : S.Nonempty
      · obtain ⟨hg, hrq⟩ := hT.scale hS
        have hmm := hT.ref_ge.2 hS
        obtain ⟨hq0, hq1⟩ := scale_q hXp2 hmm (hmS hS) (hm' ⟨hS.choose, Finset.mem_union_left _ hS.choose_spec⟩)
          ⟨hg, hrq⟩
        refine ⟨Real.exp (m - m') * g, eps0 * expm1 (lam0 + lam1 * t) * ∑ j ∈ S, G.u j +
          (Real.exp (m - m') * g + eps0) * (t * a0) + eps0 * ∑ j ∈ S, G.u j,
          mul_nonneg hq0 hU0, rfl, ?_, ?_⟩
        · set q := Real.exp (m - m') * g with hqdef
          set U := ∑ j ∈ S, G.u j with hUdef
          have e : r * l - q * U = r * (l - U) + (r - q) * U := by ring
          rw [e]
          have hr : |r| ≤ q + eps0 := by
            have := abs_sub_abs_le_abs_sub r q
            rw [abs_of_nonneg hq0] at this
            linarith
          have h1 : |r * (l - U)| ≤ (q + eps0) * (expm1 (lam0 + lam1 * t) * U + t * a0) := by
            rw [abs_mul]
            exact mul_le_mul hr hl (abs_nonneg _) (by linarith [abs_nonneg r])
          have h2 : |(r - q) * U| ≤ eps0 * U := by
            rw [abs_mul, abs_of_nonneg hU0]
            exact mul_le_mul_of_nonneg_right hrq hU0
          refine (abs_add_le _ _).trans ?_
          nlinarith
        · set q := Real.exp (m - m') * g with hqdef
          set U := ∑ j ∈ S, G.u j with hUdef
          have hε : (0 : ℝ) ≤ eps0 := by unfold eps0; norm_num
          have hta : (0 : ℝ) ≤ t * a0 := mul_nonneg (Nat.cast_nonneg t) (by unfold a0; norm_num)
          have hq2 : q + eps0 ≤ 1 + 1 / 2 ^ 19 := by
            unfold eps0; norm_num at hq1 ⊢; linarith
          have hA : eps0 * expm1 (lam0 + lam1 * t) * U + eps0 * U ≤ 1 / 2 ^ 112 := by
            have h1 : eps0 * expm1 (lam0 + lam1 * t) * U ≤ eps0 * (1 / 100) * (2 ^ 18 * (1 + 1 / 2 ^ 7)) :=
              mul_le_mul (mul_le_mul_of_nonneg_left hΛ1 hε) hUb hU0 (by unfold eps0; norm_num)
            have h2 : eps0 * U ≤ eps0 * (2 ^ 18 * (1 + 1 / 2 ^ 7)) := mul_le_mul_of_nonneg_left hUb hε
            unfold eps0 at h1 h2 ⊢
            norm_num at h1 h2 ⊢
            linarith
          nlinarith [mul_le_mul_of_nonneg_right hq2 hta]
      · have hS0 : S = ∅ := Finset.not_nonempty_iff_eq_empty.mp hS
        subst hS0
        have hta : (0 : ℝ) ≤ t * a0 := mul_nonneg (Nat.cast_nonneg t) (by unfold a0; norm_num)
        refine ⟨0, (1 + 1 / 2 ^ 20) * (t * a0), by simp, by simp, ?_, ?_⟩
        · simp only [Finset.sum_empty, mul_zero, sub_zero, zero_add]
          rw [abs_mul]
          exact mul_le_mul hT.scale_bd (hE rfl) (abs_nonneg _) (by norm_num)
        · nlinarith [hta]
    obtain ⟨q, cR, hX1, hq, hRb, hcR⟩ := hR
    rw [hq]
    have hcore := deno_core (Λ := expm1 (lam0 + lam1 * t)) (L0 := expm1 lam0) (lam := expm1 lam1)
      (R := r * l) (Ssum := Ssum) (l' := l') (cR := cR) (cS := 66 * eps0) (e0 := eps0) hX1 hW0
      hL00 hLL hlam hRb ?_ hT.deno
    · rw [hΛ']
      refine hcore.trans ?_
      have ht0 : (t : ℝ) ≤ 2 ^ 13 := by linarith
      have hta : (t : ℝ) * a0 ≤ 2 ^ 13 * a0 := mul_le_mul_of_nonneg_right ht0 (by unfold a0; norm_num)
      push_cast
      have h3 : (1 + 3 * uf) * ((1 + 1 / 2 ^ 19) * (t * a0) + 1 / 2 ^ 112 + 66 * eps0) + eps0 ≤
          (t + 1) * a0 := by
        unfold uf eps0 a0 at *
        norm_num at *
        nlinarith [Nat.cast_nonneg (α := ℝ) t]
      have h4 : (1 + 3 * uf) * (cR + 66 * eps0) + eps0 ≤
          (1 + 3 * uf) * ((1 + 1 / 2 ^ 19) * (t * a0) + 1 / 2 ^ 112 + 66 * eps0) + eps0 := by
        have : 0 ≤ 1 + 3 * uf := by unfold uf; norm_num
        nlinarith
      linarith
    · -- the sum of the new lanes
      have h1 := hT.sum
      have e : Ssum - W = (Ssum - ∑ j ∈ V, p j) + (∑ j ∈ V, p j - W) := by ring
      rw [e]
      refine (abs_add_le _ _).trans ?_
      have h2 : expm1 lam0 * ∑ j ∈ V, |p j| ≤ expm1 lam0 * (W + 64 * eps0) :=
        mul_le_mul_of_nonneg_left hPabs hL00
      rw [mul_add] at h2
      have h3 : expm1 lam0 * (64 * eps0) ≤ 1 / 2 ^ 20 * (64 * eps0) :=
        mul_le_mul_of_nonneg_right hL0 (by unfold eps0; norm_num)
      have h4 : (1 / 2 ^ 20 : ℝ) * (64 * eps0) ≤ eps0 := by unfold eps0; norm_num
      linarith

/-- A trip count can be raised (a skipped tile, or a bound on the number of trips). -/
theorem rowInv_mono {sc : ℕ → ℝ} {dl Xp : ℝ} {t t' : ℕ} {S : Finset ℕ} {m l : ℝ} {G : Ghost}
    (hXp : Xp ≤ 2 ^ 13) (hcard : S.card ≤ nmax) (htt : t ≤ t') (ht' : t' ≤ tmax)
    (hinv : RowInv sc dl Xp t S m l G) : RowInv sc dl Xp t' S m l G := by
  obtain ⟨hU0, -⟩ := hinv.sum_u_bound hXp (htt.trans ht') hcard
  obtain ⟨hL, hM, hE, hl⟩ := hinv
  have htt' : (t : ℝ) ≤ t' := by exact_mod_cast htt
  refine ⟨hL, fun hne => ?_, fun h => (hE h).trans (mul_le_mul_of_nonneg_right htt' (by unfold a0; norm_num)), ?_⟩
  · obtain ⟨h1, h2, h3⟩ := hM hne
    refine ⟨h1, h2, fun j hj => ?_⟩
    obtain ⟨F, hF, hu⟩ := h3 j hj
    refine ⟨F, hF.trans (expm1_mono ?_), hu⟩
    nlinarith [e1_pos]
  · refine hl.trans ?_
    have h1 : expm1 (lam0 + lam1 * t) ≤ expm1 (lam0 + lam1 * t') :=
      expm1_mono (by unfold lam1; nlinarith)
    have h2 : (t : ℝ) * a0 ≤ t' * a0 := mul_le_mul_of_nonneg_right htt' (by unfold a0; norm_num)
    nlinarith [mul_le_mul_of_nonneg_right h1 hU0]

end

end Claims.ExtendAttn
