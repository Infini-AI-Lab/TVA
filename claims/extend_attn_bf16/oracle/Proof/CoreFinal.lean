import Proof.CoreRow
import Proof.CoreCol
import Proof.CoreRatio
import Proof.CoreNum
import Proof.CoreStore

/-!
# The row's output against the claim
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- **The row's output.** A row of `Tp` cached-part trips and `Te ≥ 1` extend trips with the claim's
`BD`, `X` (the row's score bound) and the ghost invariants of the finished trips: the stored value
`c` is within the claim's tolerance of `wref`. -/
theorem final_bound {BD Tp Te : ℕ} {X : ℝ} {K : Finset ℕ} {sc v : ℕ → ℝ} {m l a γ q c : ℝ}
    {G : Ghost} (hBD : BD ≤ 256) (hX0 : 0 ≤ X) (hX : X ≤ 2 ^ 12) (hTe : 1 ≤ Te)
    (hT : Tp + Te ≤ 2 ^ 12 + 2) (hK : K.Nonempty) (hcard : K.card ≤ nmax)
    (hrow : RowInv sc (dlt BD X) (X + dlt BD X) (Tp + Te) K m l G)
    (hcol : ColInv v γ (Tp + Te) K G a) (hγ0 : 0 ≤ γ)
    (hγ : γ ≤ g2 + Tp * g1p + Te * g1e)
    (hq : |q - a / l| ≤ 1 / 2 ^ 22 * |a / l| + 1 / 2 ^ 148)
    (hc : |c - q| ≤ ub * |q| + 1 / 2 ^ 134) :
    1 / 4 ≤ l ∧ |c - wref K sc v| ≤ tolRel BD Tp Te X * wabs K sc v + tolAbs (∑ j ∈ K, |v j|) := by
  have hBD' : (BD : ℝ) ≤ 256 := by exact_mod_cast hBD
  have hd0 : 0 ≤ dlt BD X := by unfold dlt; positivity
  have hd1 : dlt BD X ≤ 1 / 2 := by
    unfold dlt
    have h1 : ((BD : ℝ) + 8) * (1 / 2 ^ 22) * X ≤ (256 + 8) * (1 / 2 ^ 22) * 2 ^ 12 := by
      apply mul_le_mul (mul_le_mul_of_nonneg_right (by linarith) (by norm_num)) hX hX0 (by norm_num)
    norm_num at h1 ⊢
    linarith
  have hTp : (Tp + Te : ℕ) ≤ tmax := by unfold tmax; omega
  have hTr : ((Tp + Te : ℕ) : ℝ) ≤ 4098 := by exact_mod_cast (by omega : Tp + Te ≤ 4098)
  have hγ1 : γ ≤ 1 / 2 ^ 3 := by
    have hTs : (Tp : ℝ) + Te ≤ 4098 := by exact_mod_cast hTr
    refine hγ.trans ?_
    unfold g2 g1p g1e
    nlinarith [Nat.cast_nonneg (α := ℝ) Tp, Nat.cast_nonneg (α := ℝ) Te]
  obtain ⟨hl0, hrat⟩ := ratio_bound (dl := dlt BD X) (Xp := X + dlt BD X) hd0 hd1
    (by norm_num at hX ⊢; linarith [hd0]) (by norm_num at hX ⊢; linarith) hγ0 hγ1 hTp hK hcard hrow hcol
  have hwa0 : 0 ≤ wabs K sc v := by
    unfold wabs
    exact div_nonneg (Finset.sum_nonneg fun j _ => mul_nonneg (Real.exp_pos _).le (abs_nonneg _))
      (Finset.sum_pos (fun j _ => Real.exp_pos _) hK).le
  have hwr : |wref K sc v| ≤ wabs K sc v := by
    unfold wref wabs
    have hS : 0 < ∑ j ∈ K, Real.exp (sc j) := Finset.sum_pos (fun j _ => Real.exp_pos _) hK
    rw [abs_div, abs_of_pos hS]
    refine div_le_div_of_nonneg_right ?_ hS.le
    refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq (Finset.sum_congr rfl fun j _ => ?_))
    rw [abs_mul, abs_of_pos (Real.exp_pos _)]
  have hvs : 0 ≤ ∑ j ∈ K, |v j| := Finset.sum_nonneg fun j _ => abs_nonneg _
  have hE1 := coreErr_le BD Tp Te X γ hBD hX0 hX hTe hT hγ0 hγ
  have := store_bound (E := coreErr (dlt BD X) (X + dlt BD X) (Tp + Te) γ) (E1 := errE BD Tp Te X)
    (A := 1 / 2 ^ 85 * (1 + ∑ j ∈ K, |v j|)) hwa0 hwr hvs le_rfl hrat hq hc hE1
  unfold tolRel tolAbs
  exact ⟨hl0, this⟩

end

end Claims.ExtendAttn
