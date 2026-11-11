import Proof.RowStat
import Proof.ColTrip
import Proof.CoreCol

/-!
# A row's loop state, and one trip of it

`RowFl`: the state of one row after `t` trips of the key loops, as floats related to the core's
invariants. `RowFl.trip`: one trip, from the trip's float facts (`row_stat`, the column lemmas).
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- The accumulator is at most `2^119` in magnitude, when the values are at most `2^100`. -/
theorem colInv_abs {v : ℕ → ℝ} {γ : ℝ} {t : ℕ} {S : Finset ℕ} {G : Ghost} {a : ℝ}
    (hu : ∀ j ∈ S, 0 ≤ G.u j ∧ G.u j ≤ 1 + 1 / 2 ^ 7) (hδ : ∀ j ∈ S, |G.δ j| ≤ ub)
    (hv : ∀ j ∈ S, |v j| ≤ 2 ^ 100) (hcard : S.card ≤ nmax) (hγ0 : 0 ≤ γ) (hγ : γ ≤ 1 / 8)
    (ht : t ≤ tmax) (hcol : ColInv v γ t S G a) : |a| ≤ 2 ^ 119 := by
  unfold ColInv at hcol
  have htr : (t : ℝ) ≤ 2 ^ 13 := t_le_real ht
  have hVS : ∑ j ∈ S, |v j| ≤ 2 ^ 18 * 2 ^ 100 := by
    calc ∑ j ∈ S, |v j| ≤ ∑ _j ∈ S, (2 : ℝ) ^ 100 := Finset.sum_le_sum hv
      _ = S.card * 2 ^ 100 := by simp
      _ ≤ 2 ^ 18 * 2 ^ 100 := by
        have : (S.card : ℝ) ≤ 2 ^ 18 := by exact_mod_cast hcard
        nlinarith [pow_nonneg (by norm_num : (0 : ℝ) ≤ 2) 100]
  have hR : ∑ j ∈ S, G.u j * |v j| ≤ (1 + 1 / 2 ^ 7) * ∑ j ∈ S, |v j| := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by
      have := (hu j hj).2
      nlinarith [abs_nonneg (v j)]
  have hR0 : 0 ≤ ∑ j ∈ S, G.u j * |v j| :=
    Finset.sum_nonneg fun j hj => mul_nonneg (hu j hj).1 (abs_nonneg _)
  have hN : |∑ j ∈ S, G.u j * (1 + G.δ j) * v j| ≤ (1 + 1 / 2 ^ 8) * ∑ j ∈ S, G.u j * |v j| := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun j hj => ?_
    have h1 := (hu j hj).1
    have h2 : |1 + G.δ j| ≤ 1 + 1 / 2 ^ 8 := by
      have := abs_le.mp (hδ j hj); unfold ub at this
      rw [abs_le]; constructor <;> linarith [this.1, this.2]
    rw [abs_mul, abs_mul, abs_of_nonneg h1]
    nlinarith [abs_nonneg (v j), mul_le_mul_of_nonneg_left h2 h1]
  have hΓ : expm1 γ ≤ 1 / 4 := by
    have := exp_le_quad hγ0 (by linarith)
    unfold expm1
    nlinarith
  have hΓ0 : 0 ≤ expm1 γ := expm1_nonneg hγ0
  have hta : (t : ℝ) * t0 ≤ 1 / 2 ^ 80 := by
    have := mul_le_mul_of_nonneg_right htr (by unfold t0; norm_num : (0 : ℝ) ≤ t0)
    unfold t0 at this ⊢; norm_num at this ⊢; linarith
  have hab := abs_sub_abs_le_abs_sub a (∑ j ∈ S, G.u j * (1 + G.δ j) * v j)
  have hV0 : 0 ≤ ∑ j ∈ S, |v j| := Finset.sum_nonneg fun j _ => abs_nonneg _
  have h3 : expm1 γ * ∑ j ∈ S, G.u j * |v j| ≤ 1 / 4 * ((1 + 1 / 2 ^ 7) * ∑ j ∈ S, |v j|) :=
    mul_le_mul hΓ hR hR0 (by norm_num)
  have h4 : (t : ℝ) * t0 * (1 + ∑ j ∈ S, |v j|) ≤ 1 / 2 ^ 80 * (1 + 2 ^ 18 * 2 ^ 100) :=
    mul_le_mul hta (by linarith) (by positivity) (by norm_num)
  have h5 : (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 7) * ∑ j ∈ S, |v j|) ≤ (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 7) * (2 ^ 18 * 2 ^ 100)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hVS (by norm_num)) (by norm_num)
  have h6 : (1 + 1 / 2 ^ 8) * ∑ j ∈ S, G.u j * |v j| ≤ (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 7) * ∑ j ∈ S, |v j|) :=
    mul_le_mul_of_nonneg_left hR (by norm_num)
  have h7 : 1 / 4 * ((1 + 1 / 2 ^ 7) * ∑ j ∈ S, |v j|) ≤ 1 / 4 * ((1 + 1 / 2 ^ 7) * (2 ^ 18 * 2 ^ 100)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hVS (by norm_num)) (by norm_num)
  have h8 : (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 7) * (2 ^ 18 * 2 ^ 100)) + 1 / 4 * ((1 + 1 / 2 ^ 7) * (2 ^ 18 * 2 ^ 100)) +
      1 / 2 ^ 80 * (1 + 2 ^ 18 * 2 ^ 100) ≤ 2 ^ 119 := by norm_num
  have hN' : |∑ j ∈ S, G.u j * (1 + G.δ j) * v j| ≤ (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 7) * ∑ j ∈ S, |v j|) :=
    hN.trans h6
  linarith

/-- **A row's state in the key loops**, after `t` trips that saw the positions `S`: the row's
reference `emax`, `deno` and accumulator `acc` (floats) against the core's invariants. -/
def RowFl (sc : ℕ → ℝ) (vcol : ℕ → ℕ → ℝ) (D BD : ℕ) (dl Xp : ℝ) (t : ℕ) (γ : ℝ)
    (S : Finset ℕ) (emax deno : F32) (acc : Fin BD → F32) : Prop :=
  ∃ (m l : ℝ) (G : Ghost), RowInv sc dl Xp t S m l G ∧ deno.isFinite = true ∧ deno.toReal = l ∧
    (S.Nonempty → emax.isFinite = true ∧ emax.toReal = m) ∧
    (¬ S.Nonempty → emax = FloatV.negInf ∨ emax = neg1e20) ∧
    ∀ e : Fin BD, (e : ℕ) < D → (acc e).isFinite = true ∧ |(acc e).toReal| ≤ 2 ^ 119 ∧
      ColInv (vcol e) γ t S G (acc e).toReal

/-- A sum over the valid lanes' positions of a product `ext64 f · g`, as the lane sum. -/
theorem sum_laneSet_ext {base : ℕ} {valid : Fin 64 → Bool} {f : Fin 64 → ℝ} {g : ℕ → ℝ}
    {gl : Fin 64 → ℝ} (hg : ∀ n, valid n = true → g (base + (n : ℕ)) = gl n)
    (F : ℝ → ℝ → ℝ) :
    ∑ j ∈ laneSet base valid, F (ext64 base f j) (g j) =
      ∑ n : Fin 64, if valid n = true then F (f n) (gl n) else 0 := by
  rw [sum_laneSet]
  refine Finset.sum_congr rfl fun n _ => ?_
  by_cases hv : valid n = true
  · simp only [hv, ite_true, ext64_lane, hg n hv]
  · simp [hv]

/-- The accumulator's lane-form error bound, as the position-form one `colInv_step` takes. -/
theorem acc_pos {base : ℕ} {valid : Fin 64 → Bool} {p16 : Fin 64 → BF16} {vT : Fin 64 → BF16}
    {vc : ℕ → ℝ} (hv : ∀ n, valid n = true → (vT n).toReal = vc (base + (n : ℕ)))
    {a a' r ε1 ε2 e1' : ℝ}
    (h : |a' - (r * a + sgSum valid p16 vT)| ≤ ε1 * |r * a| + ε2 * lpSum valid p16 vT +
      e1' * (1 + v1Sum valid vT)) :
    |a' - (r * a + ∑ j ∈ laneSet base valid, ext64 base (fun n => (p16 n).toReal) j * vc j)| ≤
      ε1 * |r * a| + ε2 * ∑ j ∈ laneSet base valid, |ext64 base (fun n => (p16 n).toReal) j| * |vc j| +
        e1' * (1 + ∑ j ∈ laneSet base valid, |vc j|) := by
  have h1 : ∑ j ∈ laneSet base valid, ext64 base (fun n => (p16 n).toReal) j * vc j = sgSum valid p16 vT := by
    rw [sum_laneSet_ext (f := fun n => (p16 n).toReal) (g := vc) (gl := fun n => (vT n).toReal)
      (fun n hn => (hv n hn).symm) (fun x y => x * y)]
    rfl
  have h2 : ∑ j ∈ laneSet base valid, |ext64 base (fun n => (p16 n).toReal) j| * |vc j| = lpSum valid p16 vT := by
    rw [sum_laneSet_ext (f := fun n => (p16 n).toReal) (g := vc) (gl := fun n => (vT n).toReal)
      (fun n hn => (hv n hn).symm) (fun x y => |x| * |y|)]
    rfl
  have h3 : ∑ j ∈ laneSet base valid, |vc j| = v1Sum valid vT := by
    rw [sum_laneSet_ext (f := fun n => (p16 n).toReal) (g := vc) (gl := fun n => (vT n).toReal)
      (fun n hn => (hv n hn).symm) (fun _ y => |y|)]
    rfl
  rw [h1, h2, h3]; exact h

set_option maxHeartbeats 1600000 in
/-- **One trip of one row's state.** The trip's facts (`row_stat`) and the accumulator elements'
facts (`col_pref`, `col_ext`) carry the state at `t` to the state at `t + 1`. -/
theorem RowFl.trip {sc : ℕ → ℝ} {vcol : ℕ → ℕ → ℝ} {D BD : ℕ} {dl Xp : ℝ} {t : ℕ} {γ g1 ε1 : ℝ}
    {S : Finset ℕ} {base : ℕ} {valid : Fin 64 → Bool} {deno l' re nmx : F32}
    {acc acc' : Fin BD → F32} {p16 : Fin 64 → BF16} {vT : Fin BD → Fin 64 → BF16}
    {δ' f : ℕ → ℝ} {m l m' g Ssum : ℝ} {G : Ghost} {qkr pr : Fin 64 → ℝ}
    (hXp : 0 ≤ Xp) (hXp2 : Xp ≤ 2 ^ 13) (hdl : 0 ≤ dl) (ht : t + 1 ≤ tmax)
    (hcard : (S ∪ laneSet base valid).card ≤ nmax) (hdisj : Disjoint S (laneSet base valid))
    (hrow : RowInv sc dl Xp t S m l G) (hcols : ∀ e : Fin BD, (e : ℕ) < D → (acc e).isFinite = true ∧
      |(acc e).toReal| ≤ 2 ^ 119 ∧ ColInv (vcol e) γ t S G (acc e).toReal)
    (hT : TripStat sc dl Xp S (laneSet base valid) m l (ext64 base qkr) δ' f (ext64 base pr) m' g
      re.toReal Ssum l'.toReal)
    (hround : ∀ n : Fin 64, valid n = true → |(p16 n).toReal - pr n * (1 + δ' (base + (n : ℕ)))| ≤ etab)
    (hl'F : l'.isFinite = true)
    (hnn : (S ∪ laneSet base valid).Nonempty → nmx.isFinite = true ∧ nmx.toReal = m')
    (hne : ¬ (S ∪ laneSet base valid).Nonempty → nmx = neg1e20)
    (hv : ∀ e : Fin BD, (e : ℕ) < D → ∀ n : Fin 64, valid n = true → (vT e n).toReal = vcol e (base + (n : ℕ)))
    (hvb : ∀ e : Fin BD, (e : ℕ) < D → ∀ j ∈ S ∪ laneSet base valid, |vcol e j| ≤ 2 ^ 100)
    (hacc : ∀ e : Fin BD, (e : ℕ) < D → (acc' e).isFinite = true ∧
      |(acc' e).toReal - (re.toReal * (acc e).toReal + sgSum valid p16 (vT e))| ≤
        ε1 * |re.toReal * (acc e).toReal| + eps2 * lpSum valid p16 (vT e) + eps1 * (1 + v1Sum valid (vT e)))
    (hε : 0 ≤ ε1 ∧ ε1 ≤ 1 / 2 ^ 15)
    (hg1 : 1 + ε1 ≤ Real.exp g1 ∧ ε1 * (1 + ub) ≤ expm1 g1 ∧ 0 ≤ g1) (hγ : g2 ≤ γ)
    (hγ0 : γ ≤ 1 / 2 ^ 3) (hγ1 : γ + g1 ≤ 1 / 2 ^ 3) (_hdF : deno.isFinite = true)
    (_hdV : deno.toReal = l) :
    RowFl sc vcol D BD dl Xp (t + 1) (γ + g1) (S ∪ laneSet base valid) nmx l' acc' := by
  obtain ⟨he2a, he2b, he2c⟩ := eps2_ok
  have hrow' := rowInv_step hdl hXp hXp2 ht hcard (laneSet_card base valid) hdisj hrow hT
  refine ⟨m', l'.toReal, _, hrow', hl'F, rfl, hnn, fun h => Or.inr (hne h), ?_⟩
  intro e he
  obtain ⟨haF, -, hcol⟩ := hcols e he
  have hp16 : ∀ j ∈ laneSet base valid, |ext64 base (fun n => (p16 n).toReal) j -
      ext64 base pr j * (1 + δ' j)| ≤ etab := by
    intro j hj
    obtain ⟨n, hn, rfl⟩ := mem_laneSet.mp hj
    rw [ext64_lane, ext64_lane]
    exact hround n hn
  have hcol' := colInv_step (p16 := ext64 base fun n => (p16 n).toReal) (v := vcol e) (γ := γ)
    (g1 := g1) (ε1 := ε1) (ε2 := eps2) (a := (acc e).toReal) (a' := (acc' e).toReal) (r := re.toReal)
    hXp2 ht hcard hdisj hrow hT hcol hp16
    (acc_pos (hv e he) (hacc e he).2)
    ⟨hε.1, hε.2, he2a, he2b⟩ hg1 he2c hγ hγ0
  refine ⟨(hacc e he).1, ?_, hcol'⟩
  have hg20 : 0 ≤ g2 := by unfold g2; norm_num
  exact colInv_abs (fun j hj => hrow'.u_bound hXp2 (by omega) hj) (fun j hj => (hrow'.1 j hj).2.2)
    (hvb e he) hcard (by linarith [hg1.2.2]) (by norm_num at hγ1 ⊢; linarith) (by omega) hcol'

end

end Claims.ExtendAttn
