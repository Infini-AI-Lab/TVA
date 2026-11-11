import Proof.RowFl

/-!
# A row's state: the start and a skipped trip

`RowFl.zero`: before any trip (`-∞`, `+0`, `+0`). `RowFl.mono`: a trip that sees no new position
(a skipped tile, or a bound on the trips) keeps the state, with the trip count and the accumulator's
exponent raised.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- **The state before any trip**: `e_max = -∞`, `deno = +0`, `acc = +0`. -/
theorem RowFl.zero (sc : ℕ → ℝ) (vcol : ℕ → ℕ → ℝ) (D BD : ℕ) (dl Xp γ : ℝ) {emax deno : F32}
    {acc : Fin BD → F32} (he : emax = FloatV.negInf) (hd : deno = FloatV.ofPat 0)
    (ha : ∀ e, acc e = FloatV.ofPat 0) : RowFl sc vcol D BD dl Xp 0 γ ∅ emax deno acc := by
  have hz : (FloatV.ofPat 0 : F32).toReal = 0 := by
    unfold FloatV.toReal; rw [FloatV.toRat_ofPat_zero]; simp
  have hf : (FloatV.ofPat 0 : F32).isFinite = true := FloatV.isFinite_ofPat_zero Format.binary32_regular
  refine ⟨0, 0, Ghost.zero, rowInv_zero sc dl Xp 0, by rw [hd]; exact hf, by rw [hd, hz],
    fun h => absurd h (by simp), fun _ => Or.inl he, fun e _ => ?_⟩
  rw [ha e, hz]
  exact ⟨hf, by norm_num, colInv_zero _ _ _⟩

/-- **A trip that sees nothing new** keeps the state: more trips, a larger exponent bound. -/
theorem RowFl.mono {sc : ℕ → ℝ} {vcol : ℕ → ℕ → ℝ} {D BD : ℕ} {dl Xp : ℝ} {t t' : ℕ} {γ γ' : ℝ}
    {S : Finset ℕ} {emax deno : F32} {acc : Fin BD → F32} (hXp : Xp ≤ 2 ^ 13)
    (hcard : S.card ≤ nmax) (htt : t ≤ t') (ht' : t' ≤ tmax) (hγ : γ ≤ γ')
    (h : RowFl sc vcol D BD dl Xp t γ S emax deno acc) :
    RowFl sc vcol D BD dl Xp t' γ' S emax deno acc := by
  obtain ⟨m, l, G, hrow, hdF, hdV, hne, hnn, hcols⟩ := h
  refine ⟨m, l, G, rowInv_mono hXp hcard htt ht' hrow, hdF, hdV, hne, hnn, fun e he => ?_⟩
  obtain ⟨haF, hab, hcol⟩ := hcols e he
  refine ⟨haF, hab, colInv_mono (fun j hj => ?_) htt hγ hcol⟩
  exact (hrow.u_bound hXp (by omega) hj).1

end

end Claims.ExtendAttn
