import Proof.Sum

/-!
# The row sum and `lse`

From `SumInv` after the last trip and the device facts: the computed `se` is within
`(T + 143)·u·ZR` of the exact `ZR = Σ_k e^(x_k − mv)` (`sum_bound`, `T` trips), and
`lse = tl.log(se)` within `(T + 196)·u` of `log ZR` (`lse_bound`), which is at most
`lseTol vocab_size`.
-/
namespace Claims.TopkLogSoftmax

open TVA

/-! ## Sums -/

/-- A list sum over `range n` is a `Finset` sum. -/
lemma list_range_sum_eq {M : Type*} [AddCommMonoid M] (f : Nat → M) (n : Nat) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ m ih =>
      rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ]
      simp [ih]

/-- The cast of a mapped list sum. -/
lemma list_cast_sum_map {α : Type} (l : List α) (f : α → ℚ) :
    (((l.map f).sum : ℚ) : ℝ) = (l.map (fun i => ((f i : ℚ) : ℝ))).sum := by
  induction l with
  | nil => simp
  | cons a t ih => simp [ih]

/-- The cast of a mapped list sum of absolute values. -/
lemma list_cast_abs_sum_map {α : Type} (l : List α) (f : α → ℚ) :
    (((l.map (fun i => |f i|)).sum : ℚ) : ℝ) = (l.map (fun i => |((f i : ℚ) : ℝ)|)).sum := by
  induction l with
  | nil => simp
  | cons a t ih => simp [ih]

/-- The cast of the tree error bound to the reals. -/
lemma errBound_cast (d N : Nat) (Aq : ℚ) :
    ((Tree.errBound Format.binary32.u Format.binary32.eta d N Aq : ℚ) : ℝ)
      = ((1 + uR) ^ d - 1) * (Aq : ℝ) + etaR * (N : ℝ) * (1 + uR) ^ d := by
  unfold Tree.errBound
  push_cast
  rw [b32_u_cast, b32_eta_cast]

/-- A sum over leaf `0` and leaves `k + 1`, `k < N`, whose leaf `0` and leaves past `V` are `0`,
is the sum over the leaves `k + 1`, `k < V`. -/
lemma sum_leaves {f : Nat → ℝ} {V N : Nat} (hVN : V ≤ N) (h0 : f 0 = 0)
    (hz : ∀ k, V ≤ k → k < N → f (k + 1) = 0) :
    ∑ i ∈ Finset.range (1 + N), f i = ∑ k ∈ Finset.range V, f (k + 1) := by
  rw [add_comm, Finset.sum_range_succ', h0, add_zero]
  have h3 : N = V + (N - V) := (Nat.add_sub_of_le hVN).symm
  rw [h3, Finset.sum_range_add]
  have h2 : (∑ k ∈ Finset.range (N - V), f (V + k + 1)) = 0 := by
    apply Finset.sum_eq_zero
    intro k hk
    have := Finset.mem_range.mp hk
    exact hz (V + k) (by omega) (by omega)
  rw [h2, add_zero]

/-! ## One lane of the row sum -/

/-- `|tR| · ER` is bounded for a lane at or below the max. -/
lemma abs_t_mul_ER_le {mv x : F32} (hxle : x.toRat ≤ mv.toRat) :
    |tR mv x| * ER mv x ≤ 14 * ER mv x + (2 : ℝ) ^ (-15 : ℤ) := by
  by_cases h : x = FloatV.negInf
  · rw [h, ER_eq_zero rfl, mul_zero]
    have : (0 : ℝ) < (2 : ℝ) ^ (-15 : ℤ) := by norm_num
    linarith
  · rw [ER_eq_exp h]
    have ht : tR mv x ≤ 0 := by
      have h1 : (x.toRat : ℝ) ≤ (mv.toRat : ℝ) := by exact_mod_cast hxle
      simp only [tR, mR]
      linarith
    have habs : |tR mv x| = -tR mv x := abs_of_nonpos ht
    rw [habs]
    have h := mul_exp_neg_le (w := -tR mv x) (by linarith)
    rwa [neg_neg] at h

/-- A lane's error, relative to `ER`: `67u·ER + 2^-37 + 2^-137`. -/
lemma lane_err_le {mv x e : F32} (hxle : x.toRat ≤ mv.toRat)
    (he : |e.toReal - ER mv x| ≤
      (4 * uR * |tR mv x| + 11 * uR) * ER mv x + (2 : ℝ) ^ (-137 : ℤ)) :
    |e.toReal - ER mv x| ≤ 67 * uR * ER mv x + (2 : ℝ) ^ (-37 : ℤ) + (2 : ℝ) ^ (-137 : ℤ) := by
  have htE := abs_t_mul_ER_le hxle
  have hu := uR_pos
  have hE0 := ER_nonneg mv x
  have h4 : 4 * uR * (|tR mv x| * ER mv x) ≤ 4 * uR * (14 * ER mv x + (2 : ℝ) ^ (-15 : ℤ)) :=
    mul_le_mul_of_nonneg_left htE (by positivity)
  have h5 : 4 * uR * (2 : ℝ) ^ (-15 : ℤ) = (2 : ℝ) ^ (-37 : ℤ) := by unfold uR; norm_num
  nlinarith

/-- **The row sum.** After `T` trips covering the row (`V ≤ T·1024 < V + 1024`), on a row whose
values are finite or `-∞`, at most `mv`, a finite value of the row: `se` is finite and within
`(T + 143)·u·ZR` of `ZR`. -/
theorem sum_bound {P : Platform} (hexp : Device.ExpF32 P) (hninf : Device.ExpF32NegInf P)
    {V T : Nat} (hV19 : V ≤ 2 ^ 19) (hVT : V ≤ T * 1024) (hTV : T * 1024 < V + 1024)
    {xs : Nat → F32} {mv : F32} {i0 : Nat} (hi0 : i0 < V) (hmv : mv = xs i0)
    (hmvfin : mv.isFinite = true)
    (hok : ∀ k, k < V → (xs k).isFinite = true ∨ xs k = FloatV.negInf)
    (hle : ∀ k, k < V → (xs k).toRat ≤ mv.toRat)
    {se : F32} (hs : SumInv P V xs mv T se) :
    se.isFinite = true ∧ |se.toReal - ZR mv xs V| ≤ ((T : ℝ) + 143) * uR * ZR mv xs V := by
  obtain ⟨X, tr, hcov, hdep, hev, hX0, hlane⟩ := hs
  have hu := uR_pos
  have hmvne : mv ≠ FloatV.negInf := finite_ne_negInf hmvfin
  have hT513 : T ≤ 513 := by omega
  -- the lanes
  have hlb : ∀ k, k < V → (X (k + 1)).isFinite = true ∧
      |(X (k + 1)).toReal - ER mv (xs k)| ≤
        67 * uR * ER mv (xs k) + (2 : ℝ) ^ (-37 : ℤ) + (2 : ℝ) ^ (-137 : ℤ) := by
    intro k hk
    have h := hlane k (by omega)
    unfold LaneE at h
    rw [if_pos hk] at h
    obtain ⟨s, hs1, hs2⟩ := h
    obtain ⟨h1, h2⟩ := lane_bound hexp hninf hmvfin (hle k hk) (hok k hk) hs1 hs2
    exact ⟨h1, lane_err_le (hle k hk) h2⟩
  have hlz : ∀ k, V ≤ k → k < T * 1024 → X (k + 1) = zeroC := by
    intro k hk1 hk2
    have h := hlane k hk2
    unfold LaneE at h
    rwa [if_neg (by omega)] at h
  have hfin : ∀ i, i < 1 + T * 1024 → (X i).isFinite = true := by
    intro i hi
    rcases Nat.eq_zero_or_pos i with h0 | hpos
    · rw [h0, hX0]; exact zeroC_finite
    · obtain ⟨k, rfl⟩ : ∃ k, i = k + 1 := ⟨i - 1, by omega⟩
      by_cases hk : k < V
      · exact (hlb k hk).1
      · rw [hlz k (by omega) (by omega)]; exact zeroC_finite
  -- the exact quantities
  set Z := ZR mv xs V with hZ
  have hE1 : ∀ k, k < V → ER mv (xs k) ≤ 1 := by
    intro k hk
    by_cases hneg : xs k = FloatV.negInf
    · rw [hneg, ER_eq_zero rfl]; norm_num
    · refine ER_le_one hneg ?_
      rw [mR]; exact_mod_cast hle k hk
  have hZ1 : 1 ≤ Z := one_le_ZR hi0 (by rw [← hmv]; exact ER_self hmvne)
  have hZV : Z ≤ (V : ℝ) := ZR_le_n hE1
  -- `E`, the exact sum of the lanes, and `Eabs`, of their magnitudes
  set E : ℝ := ∑ k ∈ Finset.range V, (X (k + 1)).toReal with hE
  set Ea : ℝ := ∑ k ∈ Finset.range V, |(X (k + 1)).toReal| with hEa
  have hV19r : (V : ℝ) ≤ 2 ^ 19 := by exact_mod_cast hV19
  have hsmall : (V : ℝ) * ((2 : ℝ) ^ (-37 : ℤ) + (2 : ℝ) ^ (-137 : ℤ)) ≤ 65 * uR := by
    have h1 : (2 : ℝ) ^ 19 * ((2 : ℝ) ^ (-37 : ℤ) + (2 : ℝ) ^ (-137 : ℤ)) ≤ 65 * uR := by
      unfold uR; norm_num
    have h2 : (0 : ℝ) ≤ (2 : ℝ) ^ (-37 : ℤ) + (2 : ℝ) ^ (-137 : ℤ) := by positivity
    nlinarith
  have hEZ : |E - Z| ≤ 132 * uR * Z := by
    have h1 : E - Z = ∑ k ∈ Finset.range V, ((X (k + 1)).toReal - ER mv (xs k)) := by
      rw [hE, hZ, ZR, Finset.sum_sub_distrib]
    rw [h1]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    refine (Finset.sum_le_sum fun k hk => (hlb k (Finset.mem_range.mp hk)).2).trans ?_
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_const, Finset.sum_const,
      Finset.card_range, ← Finset.mul_sum]
    simp only [nsmul_eq_mul]
    have : (67 * uR) * (∑ k ∈ Finset.range V, ER mv (xs k)) = 67 * uR * Z := by rw [hZ, ZR]
    rw [this]
    nlinarith
  have hEaZ : Ea ≤ (1 + 132 * uR) * Z := by
    have h1 : Ea ≤ ∑ k ∈ Finset.range V, (ER mv (xs k) +
        (67 * uR * ER mv (xs k) + (2 : ℝ) ^ (-37 : ℤ) + (2 : ℝ) ^ (-137 : ℤ))) := by
      refine Finset.sum_le_sum fun k hk => ?_
      have hk' := Finset.mem_range.mp hk
      have h2 := (hlb k hk').2
      have h3 := abs_add_le ((X (k + 1)).toReal - ER mv (xs k)) (ER mv (xs k))
      rw [show (X (k + 1)).toReal - ER mv (xs k) + ER mv (xs k) = (X (k + 1)).toReal by ring,
        abs_of_nonneg (ER_nonneg mv (xs k))] at h3
      linarith
    refine h1.trans ?_
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_const,
      Finset.sum_const, Finset.card_range, ← Finset.mul_sum]
    simp only [nsmul_eq_mul]
    have : (∑ k ∈ Finset.range V, ER mv (xs k)) = Z := by rw [hZ, ZR]
    rw [this]
    nlinarith
  -- the tree
  have hcov' : tr.Covers (1 + T * 1024) := hcov
  have hlist : ∀ (g : Nat → ℝ), g 0 = 0 → (∀ k, V ≤ k → k < T * 1024 → g (k + 1) = 0) →
      ((List.range (1 + T * 1024)).map g).sum = ∑ k ∈ Finset.range V, g (k + 1) := by
    intro g h0 hz
    rw [list_range_sum_eq]
    exact sum_leaves hVT h0 hz
  have hS : (((List.range (1 + T * 1024)).map fun i => (X i).toRat).sum : ℝ) = E := by
    rw [list_cast_sum_map]
    refine hlist (fun i => ((X i).toRat : ℝ)) (by rw [hX0, zeroC_toRat]; simp) ?_
    intro k hk1 hk2
    show ((X (k + 1)).toRat : ℝ) = 0
    rw [hlz k hk1 hk2, zeroC_toRat]; simp
  have hSa : (((List.range (1 + T * 1024)).map fun i => |(X i).toRat|).sum : ℝ) = Ea := by
    rw [list_cast_abs_sum_map]
    refine hlist (fun i => |((X i).toRat : ℝ)|) (by rw [hX0, zeroC_toRat]; simp) ?_
    intro k hk1 hk2
    show |((X (k + 1)).toRat : ℝ)| = 0
    rw [hlz k hk1 hk2, zeroC_toRat]; simp
  have hD : T + 10 ≤ 600 := by omega
  have hpow := pow_le hD
  have hpow2 : ((1 : ℝ) + uR) ^ (T + 10) ≤ 2 := by
    have : ((T + 10 : Nat) : ℝ) ≤ 600 := by exact_mod_cast hD
    have h6 : (600 + 1 / 32 : ℝ) * uR ≤ 1 := by unfold uR; norm_num
    nlinarith
  have hpow1 : (1 : ℝ) ≤ (1 + uR) ^ (T + 10) := one_le_pow₀ (by linarith)
  have hTr : (T : ℝ) ≤ 513 := by exact_mod_cast hT513
  have hNr : ((1 + T * 1024 - 1 : Nat) : ℝ) ≤ 2 ^ 20 := by
    have : 1 + T * 1024 - 1 ≤ 2 ^ 20 := by omega
    exact_mod_cast this
  have hEa0 : 0 ≤ Ea := Finset.sum_nonneg fun k _ => abs_nonneg _
  have hZV' : Z ≤ 2 ^ 19 := hZV.trans hV19r
  have hroomR : Ea + (((1 + uR) ^ (T + 10) - 1) * Ea +
      etaR * ((1 + T * 1024 - 1 : Nat) : ℝ) * (1 + uR) ^ (T + 10)) ≤ 2 ^ 23 := by
    have h1 : Ea ≤ 2 ^ 20 := by
      have : (1 + 132 * uR) * Z ≤ 2 * 2 ^ 19 := by
        have : 132 * uR ≤ 1 := by unfold uR; norm_num
        nlinarith
      linarith
    have h2 : etaR * ((1 + T * 1024 - 1 : Nat) : ℝ) * (1 + uR) ^ (T + 10) ≤ 1 := by
      have : etaR * 2 ^ 20 * 2 ≤ 1 := by unfold etaR; norm_num
      have he := etaR_pos
      have h3 : etaR * ((1 + T * 1024 - 1 : Nat) : ℝ) ≤ etaR * 2 ^ 20 :=
        mul_le_mul_of_nonneg_left hNr he.le
      have h4 : 0 ≤ etaR * ((1 + T * 1024 - 1 : Nat) : ℝ) := by positivity
      nlinarith
    nlinarith
  have hroom : ((List.range (1 + T * 1024)).map fun i => |(X i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta (T + 10) (1 + T * 1024 - 1)
        ((List.range (1 + T * 1024)).map fun i => |(X i).toRat|).sum ≤
      Format.binary32.maxVal := by
    have h1 : ((((List.range (1 + T * 1024)).map fun i => |(X i).toRat|).sum +
        Tree.errBound Format.binary32.u Format.binary32.eta (T + 10) (1 + T * 1024 - 1)
          ((List.range (1 + T * 1024)).map fun i => |(X i).toRat|).sum : ℚ) : ℝ) ≤
        ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [Rat.cast_add, errBound_cast, hSa, b32_maxVal_cast]
      have : (2 : ℝ) ^ 23 ≤ ((2 : ℝ) ^ 24 - 1) * 2 ^ 104 := by norm_num
      linarith
    exact_mod_cast h1
  obtain ⟨hsefin, hseerr⟩ := sum_err_depth b32_reg X hcov' hdep hev hfin hroom
  refine ⟨hsefin, ?_⟩
  have hseR : |se.toReal - E| ≤ ((1 + uR) ^ (T + 10) - 1) * Ea +
      etaR * ((1 + T * 1024 - 1 : Nat) : ℝ) * (1 + uR) ^ (T + 10) := by
    have h1 : ((|se.toRat - ((List.range (1 + T * 1024)).map fun i => (X i).toRat).sum| : ℚ) : ℝ)
        ≤ ((Tree.errBound Format.binary32.u Format.binary32.eta (T + 10) (1 + T * 1024 - 1)
          ((List.range (1 + T * 1024)).map fun i => |(X i).toRat|).sum : ℚ) : ℝ) := by
      exact_mod_cast hseerr
    rw [errBound_cast, hSa, Rat.cast_abs, Rat.cast_sub, hS] at h1
    exact h1
  -- the growth factor
  have hg1 : ((1 + uR) ^ (T + 10) - 1) * Ea ≤ ((T : ℝ) + 10 + 1 / 32) * uR * ((1 + 132 * uR) * Z) := by
    have h1 : (1 + uR) ^ (T + 10) - 1 ≤ ((T : ℝ) + 10 + 1 / 32) * uR := by
      have : ((T + 10 : Nat) : ℝ) = (T : ℝ) + 10 := by push_cast; ring
      rw [this] at hpow; linarith
    have h2 : 0 ≤ (1 + uR) ^ (T + 10) - 1 := by linarith
    calc ((1 + uR) ^ (T + 10) - 1) * Ea ≤ ((1 + uR) ^ (T + 10) - 1) * ((1 + 132 * uR) * Z) :=
          mul_le_mul_of_nonneg_left hEaZ h2
      _ ≤ ((T : ℝ) + 10 + 1 / 32) * uR * ((1 + 132 * uR) * Z) :=
          mul_le_mul_of_nonneg_right h1 (by nlinarith)
  have hg2 : etaR * ((1 + T * 1024 - 1 : Nat) : ℝ) * (1 + uR) ^ (T + 10) ≤ uR / 64 * Z := by
    have he := etaR_pos
    have h3 : etaR * ((1 + T * 1024 - 1 : Nat) : ℝ) ≤ etaR * 2 ^ 20 :=
      mul_le_mul_of_nonneg_left hNr he.le
    have h4 : 0 ≤ etaR * ((1 + T * 1024 - 1 : Nat) : ℝ) := by positivity
    have h5 : etaR * 2 ^ 20 * 2 ≤ uR / 64 := by unfold etaR uR; norm_num
    nlinarith
  have hg3 : ((T : ℝ) + 10 + 1 / 32) * uR * ((1 + 132 * uR) * Z) ≤
      ((T : ℝ) + 10 + 1 / 32 + 1 / 128) * uR * Z := by
    have h1 : ((T : ℝ) + 10 + 1 / 32) * (132 * uR) ≤ 1 / 128 := by
      have : (513 + 10 + 1 / 32 : ℝ) * (132 * uR) ≤ 1 / 128 := by unfold uR; norm_num
      have h0 : (0 : ℝ) ≤ T := Nat.cast_nonneg T
      nlinarith
    have hZ0 : 0 ≤ Z := by linarith
    have h0 : (0 : ℝ) ≤ T := Nat.cast_nonneg T
    have : ((T : ℝ) + 10 + 1 / 32) * uR * ((1 + 132 * uR) * Z) =
        ((T : ℝ) + 10 + 1 / 32 + ((T : ℝ) + 10 + 1 / 32) * (132 * uR)) * uR * Z := by ring
    rw [this]
    have h6 : 0 ≤ uR * Z := by positivity
    nlinarith
  have h1 := abs_sub_le se.toReal E Z
  have h0 : (0 : ℝ) ≤ T := Nat.cast_nonneg T
  have hZ0 : 0 ≤ Z := by linarith
  nlinarith

end Claims.TopkLogSoftmax
