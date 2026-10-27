import TVA.Math.Float.F32.ChainErr

/-!
# The GEMM twin: `K` products in `s` chains, then an addition chain (`gemmTwin_err`)

Part of the `f32` error analysis (`TVA.Math.Float.F32.Error`); the definitions it reasons about are in
`TVA.Math.Float.F32.Bounds`. Split out of `F32/Bounds.lean`.
-/
set_option autoImplicit false

namespace TVA.Math.Float.F32

open TVA.Math.Float

open Rat

/-! ### The GEMM twin: `K` products in `s` chains, then an addition chain -/

theorem list_sum_map_sub {α : Type} (l : List α) (f g : α → ℚ) :
    (l.map f).sum - (l.map g).sum = (l.map fun i => f i - g i).sum := by
  induction l with
  | nil => simp
  | cons x l ih => simp only [List.map_cons, List.sum_cons]; rw [← ih]; ring

theorem list_sum_map_mul_left {α : Type} (l : List α) (a : ℚ) (f : α → ℚ) :
    (l.map fun i => a * f i).sum = a * (l.map f).sum := by
  induction l with
  | nil => simp
  | cons x l ih => simp only [List.map_cons, List.sum_cons]; rw [ih]; ring

theorem list_abs_sum_le (l : List ℚ) : |l.sum| ≤ (l.map abs).sum := by
  induction l with
  | nil => simp
  | cons x l ih =>
    simp only [List.sum_cons, List.map_cons]
    exact (abs_add_le _ _).trans (add_le_add le_rfl ih)

theorem length_le_gemmK {chains : List (List (Bits × Bits))} {ch : List (Bits × Bits)} (h : ch ∈ chains) :
    ch.length ≤ gemmK chains := by
  unfold gemmK
  exact List.single_le_sum (fun _ _ => Nat.zero_le _) _ (List.mem_map_of_mem h)

/-- **The bf16 GEMM bound, γ form**: with `K` products in `s` chains,
`|C − Σ aᵢbᵢ| ≤ γ_{K+s}·Σ|aᵢ||bᵢ|` (no size restriction). -/
theorem gemmTwin_err_gamma (chains : List (List (Bits × Bits))) (h : GemmOK chains) :
    isFinite (gemmTwin chains) = true ∧
    |toRat (gemmTwin chains) - gemmExact chains| ≤
      γ (gemmK chains + chains.length) * gemmAbs chains := by
  obtain ⟨hch, hadd⟩ := h
  set K := gemmK chains with hKdef
  set s := chains.length with hsdef
  -- per-chain bounds
  have hper : ∀ ch ∈ chains,
      isFinite (fmaChain posZero ch) = true ∧
      |toRat (fmaChain posZero ch) - (ch.map fun p => toRat p.1 * toRat p.2).sum| ≤
        γ K * (ch.map fun p => |toRat p.1| * |toRat p.2|).sum := by
    intro ch hc
    obtain ⟨hf, he⟩ := fmaChain_err posZero ch isFinite_posZero (hch ch hc)
    refine ⟨hf, ?_⟩
    simp only [exactSum, absSum, toRat_posZero, zero_add, abs_zero] at he
    exact he.trans (mul_le_mul_of_nonneg_right (γ_mono (length_le_gemmK hc)) (sum_abs_prod_nonneg ch))
  -- the addition chain
  obtain ⟨hfin, hsum⟩ := addChain_err posZero (chains.map (fmaChain posZero)) isFinite_posZero hadd
  simp only [toRat_posZero, zero_add, abs_zero, List.map_map, List.length_map, Function.comp_def] at hsum
  refine ⟨hfin, ?_⟩
  set T := gemmAbs chains with hTdef
  have hT0 : 0 ≤ T := by
    unfold gemmAbs at hTdef; rw [hTdef]
    refine List.sum_nonneg ?_
    intro x hx
    rw [List.mem_map] at hx
    obtain ⟨ch, _, rfl⟩ := hx
    exact sum_abs_prod_nonneg ch
  set P : ℚ := (chains.map fun ch => toRat (fmaChain posZero ch)).sum with hPdef
  set A : ℚ := (chains.map fun ch => |toRat (fmaChain posZero ch)|).sum with hAdef
  -- |Σ Pᵢ − Σ Sᵢ| ≤ γ_K · T
  have hPS : |P - gemmExact chains| ≤ γ K * T := by
    rw [hPdef]; unfold gemmExact
    rw [list_sum_map_sub]
    refine (list_abs_sum_le _).trans ?_
    rw [List.map_map, hTdef]; unfold gemmAbs
    rw [← list_sum_map_mul_left]
    refine List.sum_le_sum ?_
    intro ch hc
    exact (hper ch hc).2
  -- Σ |Pᵢ| ≤ (1 + γ_K) · T
  have hA : A ≤ (1 + γ K) * T := by
    rw [hAdef, hTdef]; unfold gemmAbs
    rw [← list_sum_map_mul_left]
    refine List.sum_le_sum ?_
    intro ch hc
    have h1 := (hper ch hc).2
    have h2 : |(ch.map fun p => toRat p.1 * toRat p.2).sum| ≤ (ch.map fun p => |toRat p.1| * |toRat p.2|).sum := by
      refine (list_abs_sum_le _).trans ?_
      rw [List.map_map]
      refine List.sum_le_sum ?_
      intro p _
      simp [abs_mul]
    have h3 := abs_sub_abs_le_abs_sub (toRat (fmaChain posZero ch)) ((ch.map fun p => toRat p.1 * toRat p.2).sum)
    nlinarith [h1, h2, h3]
  -- combine
  have hγs := γ_nonneg s
  have hC : |toRat (gemmTwin chains) - gemmExact chains| ≤ γ s * ((1 + γ K) * T) + γ K * T := by
    have h1 : toRat (gemmTwin chains) - gemmExact chains =
        (toRat (gemmTwin chains) - P) + (P - gemmExact chains) := by ring
    rw [h1]
    refine (abs_add_le _ _).trans (add_le_add ?_ hPS)
    have : |toRat (gemmTwin chains) - P| ≤ γ s * A := hsum
    exact this.trans (mul_le_mul_of_nonneg_left hA hγs)
  calc |toRat (gemmTwin chains) - gemmExact chains|
      ≤ γ s * ((1 + γ K) * T) + γ K * T := hC
    _ = γ (K + s) * T := by rw [Nat.add_comm K s, γ_add]; ring

/-- **The bf16 GEMM bound**: with `K` products in `s` chains (`K + s ≤ 2^23`),
`|C − Σ aᵢbᵢ| ≤ 2^-23·(K + s + 1)·Σ|aᵢ||bᵢ|`. The `+1` is slack: the proof gives `K + s`. -/
theorem gemmTwin_err (chains : List (List (Bits × Bits))) (h : GemmOK chains)
    (hK : gemmK chains + chains.length ≤ 2 ^ 23) :
    isFinite (gemmTwin chains) = true ∧
    |toRat (gemmTwin chains) - gemmExact chains| ≤
      (2 : ℚ) ^ (-23 : ℤ) * ((gemmK chains : ℚ) + chains.length + 1) * gemmAbs chains := by
  obtain ⟨hfin, herr⟩ := gemmTwin_err_gamma chains h
  refine ⟨hfin, herr.trans ?_⟩
  set K := gemmK chains with hKdef
  set s := chains.length with hsdef
  set T := gemmAbs chains with hTdef
  have hT0 : 0 ≤ T := by
    unfold gemmAbs at hTdef; rw [hTdef]
    refine List.sum_nonneg ?_
    intro x hx
    rw [List.mem_map] at hx
    obtain ⟨ch, _, rfl⟩ := hx
    exact sum_abs_prod_nonneg ch
  have hlin : γ (K + s) ≤ ((K + s : ℕ) : ℚ) * (2 * u) := γ_le_linear (K + s) hK
  have h2u : (2 : ℚ) * u = 2 ^ (-23 : ℤ) := by unfold u; norm_num
  calc γ (K + s) * T
      ≤ ((K + s : ℕ) : ℚ) * (2 * u) * T := mul_le_mul_of_nonneg_right hlin hT0
    _ = 2 ^ (-23 : ℤ) * ((K : ℚ) + s) * T := by rw [h2u]; push_cast; ring
    _ ≤ 2 ^ (-23 : ℤ) * ((K : ℚ) + s + 1) * T := by
        have hp := two_zpow_pos (-23 : ℤ)
        have : (K : ℚ) + s ≤ (K : ℚ) + s + 1 := by linarith
        exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left this hp.le) hT0

end TVA.Math.Float.F32
