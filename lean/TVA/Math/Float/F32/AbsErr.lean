import TVA.Math.Float.F32.Bf16Round

/-!
# Absolute-term versions (no subnormal hypothesis): `δₙ`, chains and the GEMM twin with an absolute term, well-conditioning from a magnitude bound

Part of the `f32` error analysis (`TVA.Math.Float.F32.Error`); the definitions it reasons about are in
`TVA.Math.Float.F32.Bounds`. Split out of `F32/Bounds.lean`.
-/
set_option autoImplicit false

namespace TVA.Math.Float.F32

open TVA.Math.Float

theorem subErr_pos : 0 < subErr := by unfold subErr; positivity

/-- One rounding below overflow: relative `u` plus absolute `subErr`. -/
theorem roundRat_err_abs (q : ℚ) (hmax : |q| ≤ maxNormal) :
    isFinite (roundRat q) = true ∧ |toRat (roundRat q) - q| ≤ u * |q| + subErr := by
  by_cases hlt : |q| < minNormal
  · obtain ⟨hf, he⟩ := roundRat_err_sub q hlt
    exact ⟨hf, he.trans (by have := mul_nonneg u_pos.le (abs_nonneg q); linarith)⟩
  · rw [not_lt] at hlt
    obtain ⟨hf, he⟩ := roundRat_err_normal q hlt hmax
    exact ⟨hf, he.trans (by linarith [subErr_pos])⟩

/-- [meaning] For finite operands whose exact sum is below overflow, the f32 sum `add` is finite
and within `u·|x + y| + 2^-150` of the exact sum (one rounding to nearest; the absolute term
covers subnormal results).

**`add.f32`**, absolute form. -/
theorem add_err_abs (x y : Bits) (hx : isFinite x = true) (hy : isFinite y = true)
    (hmax : |toRat x + toRat y| ≤ maxNormal) :
    isFinite (add x y) = true ∧ |toRat (add x y) - (toRat x + toRat y)| ≤ u * |toRat x + toRat y| + subErr := by
  obtain ⟨hxn, hxi⟩ := finite_not_nan_inf hx
  obtain ⟨hyn, hyi⟩ := finite_not_nan_inf hy
  unfold add
  simp only [hxn, hxi, hyn, hyi, Bool.false_or, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  have hnn : 0 ≤ u * |toRat x + toRat y| + subErr := by
    have := mul_nonneg u_pos.le (abs_nonneg (toRat x + toRat y)); linarith [subErr_pos]
  split_ifs with hz hs
  · have hzz : isZero x = true ∧ isZero y = true := by simpa using hz
    rw [toRat_zero_of_isZero hzz.1, toRat_zero_of_isZero hzz.2, toRat_negZero]
    simpa [isFinite_negZero] using subErr_pos.le
  · have hzz : isZero x = true ∧ isZero y = true := by simpa using hz
    rw [toRat_zero_of_isZero hzz.1, toRat_zero_of_isZero hzz.2, toRat_posZero]
    simpa [isFinite_posZero] using subErr_pos.le
  · exact roundRat_err_abs _ hmax

/-- **`fma.rn.f32`**, absolute form. -/
theorem fma_err_abs (x y z : Bits) (hx : isFinite x = true) (hy : isFinite y = true) (hz : isFinite z = true)
    (hmax : |toRat x * toRat y + toRat z| ≤ maxNormal) :
    isFinite (fma x y z) = true ∧
    |toRat (fma x y z) - (toRat x * toRat y + toRat z)| ≤ u * |toRat x * toRat y + toRat z| + subErr := by
  obtain ⟨hxn, hxi⟩ := finite_not_nan_inf hx
  obtain ⟨hyn, hyi⟩ := finite_not_nan_inf hy
  obtain ⟨hzn, hzi⟩ := finite_not_nan_inf hz
  unfold fma
  simp only [hxn, hxi, hyn, hyi, hzn, hzi, Bool.false_or, Bool.false_eq_true, ↓reduceIte]
  have hnn : 0 ≤ u * |toRat x * toRat y + toRat z| + subErr := by
    have := mul_nonneg u_pos.le (abs_nonneg (toRat x * toRat y + toRat z)); linarith [subErr_pos]
  split_ifs with h0 hz0 hs
  · have h0' : isZero x = true ∨ isZero y = true := by simpa using h0
    have hprod : toRat x * toRat y = 0 := by
      rcases h0' with h | h
      · rw [toRat_zero_of_isZero h, zero_mul]
      · rw [toRat_zero_of_isZero h, mul_zero]
    rw [hprod, toRat_zero_of_isZero hz0, toRat_negZero]
    simpa [isFinite_negZero] using subErr_pos.le
  · have h0' : isZero x = true ∨ isZero y = true := by simpa using h0
    have hprod : toRat x * toRat y = 0 := by
      rcases h0' with h | h
      · rw [toRat_zero_of_isZero h, zero_mul]
      · rw [toRat_zero_of_isZero h, mul_zero]
    rw [hprod, toRat_zero_of_isZero hz0, toRat_posZero]
    simpa [isFinite_posZero] using subErr_pos.le
  · have h0' : isZero x = true ∨ isZero y = true := by simpa using h0
    have hprod : toRat x * toRat y = 0 := by
      rcases h0' with h | h
      · rw [toRat_zero_of_isZero h, zero_mul]
      · rw [toRat_zero_of_isZero h, mul_zero]
    rw [hprod, zero_add] at hnn ⊢
    simp only [hz, sub_self, abs_zero, true_and]
    exact hnn
  · exact roundRat_err_abs _ hmax

/-! ### `δₙ`: the accumulated absolute term -/

theorem δ_zero : δ 0 = 0 := rfl
theorem δ_succ (n : ℕ) : δ (n + 1) = (1 + u) * δ n + subErr := rfl

theorem δ_nonneg (n : ℕ) : 0 ≤ δ n := by
  induction n with
  | zero => simp [δ_zero]
  | succ k ih => rw [δ_succ]; have := one_add_u_pos; nlinarith [subErr_pos]

theorem δ_le_succ (n : ℕ) : δ n ≤ δ (n + 1) := by
  rw [δ_succ]
  have := δ_nonneg n
  nlinarith [u_pos, subErr_pos]

theorem δ_mono {m n : ℕ} (h : m ≤ n) : δ m ≤ δ n := by
  induction n with
  | zero =>
    have : m = 0 := by omega
    subst this; exact le_rfl
  | succ k ih =>
    rcases Nat.lt_or_ge m (k + 1) with hlt | hge
    · exact (ih (by omega)).trans (δ_le_succ k)
    · have : m = k + 1 := by omega
      subst this; exact le_rfl

/-- `u·δₙ = subErr·γₙ`. -/
theorem u_mul_δ (n : ℕ) : u * δ n = subErr * γ n := by
  induction n with
  | zero => simp [δ_zero, γ_zero]
  | succ k ih => rw [δ_succ, γ_succ]; linear_combination (1 + u) * ih

/-- `δₙ ≤ 2·n·subErr = n·2^-149` for `n ≤ 2^23`. -/
theorem δ_le_linear (n : ℕ) (hn : n ≤ 2 ^ 23) : δ n ≤ (n : ℚ) * (2 * subErr) := by
  induction n with
  | zero => simp [δ_zero]
  | succ k ih =>
    have hk : k ≤ 2 ^ 23 := by omega
    have ihk := ih hk
    have hk' : (k : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hk
    have hu : u = 1 / 2 ^ 24 := rfl
    have key : 2 * (k : ℚ) * u * subErr ≤ subErr := by
      have : 2 * (k : ℚ) * u ≤ 1 := by rw [hu]; norm_num; linarith
      nlinarith [subErr_pos]
    have h1 : (1 + u) * δ k ≤ (1 + u) * ((k : ℚ) * (2 * subErr)) :=
      mul_le_mul_of_nonneg_left ihk one_add_u_pos.le
    rw [δ_succ]; push_cast
    nlinarith [h1, key, u_pos, subErr_pos]

/-! ### FMA chains, absolute form -/

/-- **FMA chain error, absolute form**: `|chain − exact| ≤ γₙ·absSum + δₙ`. -/
theorem fmaChain_err_abs (c : Bits) (l : List (Bits × Bits)) (hc : isFinite c = true) (h : ChainFin c l) :
    isFinite (fmaChain c l) = true ∧
    |toRat (fmaChain c l) - exactSum c l| ≤ γ l.length * absSum c l + δ l.length := by
  induction l generalizing c with
  | nil => simp [fmaChain, exactSum, absSum, γ_zero, δ_zero, hc]
  | cons q l ih =>
    obtain ⟨a, b⟩ := q
    obtain ⟨ha, hb, hmax, hrest⟩ := h
    obtain ⟨hf1, he1⟩ := fma_err_abs a b c ha hb hc hmax
    obtain ⟨hf, he⟩ := ih (fma a b c) hf1 hrest
    refine ⟨by simpa [fmaChain] using hf, ?_⟩
    simp only [fmaChain, exactSum, absSum, List.map_cons, List.sum_cons, List.length_cons] at he ⊢
    set c' := fma a b c with hc'
    set p := toRat a * toRat b with hp
    set S := (l.map fun p => toRat p.1 * toRat p.2).sum with hS
    set T := (l.map fun p => |toRat p.1| * |toRat p.2|).sum with hT
    set n := l.length with hn
    have hT0 : 0 ≤ T := sum_abs_prod_nonneg l
    have hγ := γ_nonneg n
    have hδ := δ_nonneg n
    have hpabs : |p| = |toRat a| * |toRat b| := abs_mul _ _
    have hsum : |p + toRat c| ≤ |toRat c| + |p| := by rw [add_comm]; exact abs_add_le _ _
    have hc'le : |toRat c'| ≤ (1 + u) * (|toRat c| + |p|) + subErr := by
      have h1 : |toRat c'| ≤ |p + toRat c| + |toRat c' - (p + toRat c)| := by
        have := abs_sub_abs_le_abs_sub (toRat c') (p + toRat c)
        linarith
      calc |toRat c'| ≤ |p + toRat c| + (u * |p + toRat c| + subErr) := by linarith [he1]
        _ = (1 + u) * |p + toRat c| + subErr := by ring
        _ ≤ (1 + u) * (|toRat c| + |p|) + subErr := by
            have := mul_le_mul_of_nonneg_left hsum one_add_u_pos.le; linarith
    have hsplit : toRat (fmaChain c' l) - (toRat c + (p + S)) =
        (toRat (fmaChain c' l) - (toRat c' + S)) + (toRat c' - (p + toRat c)) := by ring
    have h2 : |toRat (fmaChain c' l) - (toRat c + (p + S))| ≤
        (γ n * (|toRat c'| + T) + δ n) + (u * |p + toRat c| + subErr) := by
      rw [hsplit]
      exact (abs_add_le _ _).trans (add_le_add he he1)
    have h3 : γ n * (|toRat c'| + T) ≤ γ n * ((1 + u) * (|toRat c| + |p|) + subErr + T) :=
      mul_le_mul_of_nonneg_left (by linarith) hγ
    have h4 : u * |p + toRat c| ≤ u * (|toRat c| + |p|) := mul_le_mul_of_nonneg_left hsum u_pos.le
    have h5 : γ n * T ≤ γ (n + 1) * T := mul_le_mul_of_nonneg_right (γ_mono (by omega)) hT0
    have hud := u_mul_δ n
    rw [hpabs.symm]
    calc |toRat (fmaChain c' l) - (toRat c + (p + S))|
        ≤ (γ n * (|toRat c'| + T) + δ n) + (u * |p + toRat c| + subErr) := h2
      _ ≤ (γ n * ((1 + u) * (|toRat c| + |p|) + subErr + T) + δ n) + (u * (|toRat c| + |p|) + subErr) :=
          by linarith [h3, h4]
      _ = ((1 + u) * γ n + u) * (|toRat c| + |p|) + γ n * T + (γ n * subErr + δ n + subErr) := by ring
      _ = γ (n + 1) * (|toRat c| + |p|) + γ n * T + δ (n + 1) := by
          rw [γ_succ, δ_succ]
          have : γ n * subErr = u * δ n := by rw [hud]; ring
          rw [this]; ring
      _ ≤ γ (n + 1) * (|toRat c| + |p|) + γ (n + 1) * T + δ (n + 1) := by linarith [h5]
      _ = γ (n + 1) * (|toRat c| + (|p| + T)) + δ (n + 1) := by ring

/-! ### Addition chains, absolute form -/

/-- **Addition chain error, absolute form**. -/
theorem addChain_err_abs (acc : Bits) (l : List Bits) (hacc : isFinite acc = true) (h : AddFin acc l) :
    isFinite (addChain acc l) = true ∧
    |toRat (addChain acc l) - (toRat acc + (l.map toRat).sum)| ≤
      γ l.length * (|toRat acc| + (l.map fun p => |toRat p|).sum) + δ l.length := by
  induction l generalizing acc with
  | nil => simp [addChain, γ_zero, δ_zero, hacc]
  | cons q l ih =>
    obtain ⟨hq, hmax, hrest⟩ := h
    obtain ⟨hf1, he1⟩ := add_err_abs acc q hacc hq hmax
    obtain ⟨hf, he⟩ := ih (add acc q) hf1 hrest
    refine ⟨by simpa [addChain] using hf, ?_⟩
    simp only [addChain, List.map_cons, List.sum_cons, List.length_cons] at he ⊢
    set c' := add acc q with hc'
    set S := (l.map toRat).sum with hS
    set T := (l.map fun p => |toRat p|).sum with hT
    set n := l.length with hn
    have hT0 : 0 ≤ T := sum_abs_nonneg l
    have hγ := γ_nonneg n
    have hδ := δ_nonneg n
    have hsum : |toRat acc + toRat q| ≤ |toRat acc| + |toRat q| := abs_add_le _ _
    have hc'le : |toRat c'| ≤ (1 + u) * (|toRat acc| + |toRat q|) + subErr := by
      have h1 : |toRat c'| ≤ |toRat acc + toRat q| + |toRat c' - (toRat acc + toRat q)| := by
        have := abs_sub_abs_le_abs_sub (toRat c') (toRat acc + toRat q)
        linarith
      calc |toRat c'| ≤ |toRat acc + toRat q| + (u * |toRat acc + toRat q| + subErr) := by linarith [he1]
        _ = (1 + u) * |toRat acc + toRat q| + subErr := by ring
        _ ≤ (1 + u) * (|toRat acc| + |toRat q|) + subErr := by
            have := mul_le_mul_of_nonneg_left hsum one_add_u_pos.le; linarith
    have hsplit : toRat (addChain c' l) - (toRat acc + (toRat q + S)) =
        (toRat (addChain c' l) - (toRat c' + S)) + (toRat c' - (toRat acc + toRat q)) := by ring
    have h2 : |toRat (addChain c' l) - (toRat acc + (toRat q + S))| ≤
        (γ n * (|toRat c'| + T) + δ n) + (u * |toRat acc + toRat q| + subErr) := by
      rw [hsplit]
      exact (abs_add_le _ _).trans (add_le_add he he1)
    have h3 : γ n * (|toRat c'| + T) ≤ γ n * ((1 + u) * (|toRat acc| + |toRat q|) + subErr + T) :=
      mul_le_mul_of_nonneg_left (by linarith) hγ
    have h4 : u * |toRat acc + toRat q| ≤ u * (|toRat acc| + |toRat q|) :=
      mul_le_mul_of_nonneg_left hsum u_pos.le
    have h5 : γ n * T ≤ γ (n + 1) * T := mul_le_mul_of_nonneg_right (γ_mono (by omega)) hT0
    have hud := u_mul_δ n
    calc |toRat (addChain c' l) - (toRat acc + (toRat q + S))|
        ≤ (γ n * (|toRat c'| + T) + δ n) + (u * |toRat acc + toRat q| + subErr) := h2
      _ ≤ (γ n * ((1 + u) * (|toRat acc| + |toRat q|) + subErr + T) + δ n) + (u * (|toRat acc| + |toRat q|) + subErr) :=
          by linarith [h3, h4]
      _ = ((1 + u) * γ n + u) * (|toRat acc| + |toRat q|) + γ n * T + (γ n * subErr + δ n + subErr) := by ring
      _ = γ (n + 1) * (|toRat acc| + |toRat q|) + γ n * T + δ (n + 1) := by
          rw [γ_succ, δ_succ]
          have : γ n * subErr = u * δ n := by rw [hud]; ring
          rw [this]; ring
      _ ≤ γ (n + 1) * (|toRat acc| + |toRat q|) + γ (n + 1) * T + δ (n + 1) := by linarith [h5]
      _ = γ (n + 1) * (|toRat acc| + (|toRat q| + T)) + δ (n + 1) := by ring

/-! ### Well-conditioning from a magnitude bound -/

theorem one_add_u_pow_le_two (j : ℕ) (hj : j ≤ 2 ^ 23) : (1 + u) ^ j ≤ 2 := by
  have h := γ_le_linear j hj
  unfold γ at h
  have hj' : (j : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hj
  have hu : u = 1 / 2 ^ 24 := rfl
  have : (j : ℚ) * (2 * u) ≤ 1 := by rw [hu]; norm_num; linarith
  linarith

theorem δ_le_one (j : ℕ) (hj : j ≤ 2 ^ 23) : δ j ≤ 1 := by
  have h := δ_le_linear j hj
  have hj' : (j : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hj
  have hs : subErr = 1 / 2 ^ 150 := rfl
  have : (j : ℚ) * (2 * subErr) ≤ 1 := by rw [hs]; norm_num; linarith
  linarith

/-- A chain whose condition-number numerator is bounded (with room for the `(1+u)^j` growth of
the rounded accumulator) is well conditioned: every exact step result is below overflow. -/
theorem chainFin_of_bound (B : ℚ) (hB : 0 ≤ B) (hBmax : 2 * B + 1 ≤ maxNormal) :
    ∀ (l : List (Bits × Bits)) (c : Bits) (j : ℕ), isFinite c = true →
      (∀ p ∈ l, isFinite p.1 = true ∧ isFinite p.2 = true) →
      absSum c l ≤ B * (1 + u) ^ j + δ j → j + l.length ≤ 2 ^ 23 → ChainFin c l := by
  intro l
  induction l with
  | nil => intro c j _ _ _ _; trivial
  | cons q l ih =>
    intro c j hc hfin hbound hlen
    simp only [List.length_cons] at hlen
    obtain ⟨a, b⟩ := q
    have hab := hfin (a, b) List.mem_cons_self
    simp only at hab
    have hfin' : ∀ p ∈ l, isFinite p.1 = true ∧ isFinite p.2 = true :=
      fun p hp => hfin p (List.mem_cons_of_mem _ hp)
    simp only [absSum, List.map_cons, List.sum_cons] at hbound
    set p := toRat a * toRat b with hp
    set T := (l.map fun p => |toRat p.1| * |toRat p.2|).sum with hT
    have hT0 : 0 ≤ T := sum_abs_prod_nonneg l
    have hpabs : |p| = |toRat a| * |toRat b| := abs_mul _ _
    have hpow : (1 + u) ^ j ≤ 2 := one_add_u_pow_le_two j (by omega)
    have hδ1 : δ j ≤ 1 := δ_le_one j (by omega)
    have hr : |p + toRat c| ≤ |toRat c| + |p| := by rw [add_comm]; exact abs_add_le _ _
    have hmax : |p + toRat c| ≤ maxNormal := by
      have : |toRat c| + |p| ≤ B * (1 + u) ^ j + δ j := by rw [hpabs]; linarith
      have hB2 : B * (1 + u) ^ j ≤ B * 2 := mul_le_mul_of_nonneg_left hpow hB
      linarith
    refine ⟨hab.1, hab.2, hmax, ?_⟩
    obtain ⟨hf', he'⟩ := fma_err_abs a b c hab.1 hab.2 hc hmax
    have hc'le : |toRat (fma a b c)| ≤ (1 + u) * (|toRat c| + |p|) + subErr := by
      have h1 : |toRat (fma a b c)| ≤ |p + toRat c| + |toRat (fma a b c) - (p + toRat c)| := by
        have := abs_sub_abs_le_abs_sub (toRat (fma a b c)) (p + toRat c)
        linarith
      calc |toRat (fma a b c)| ≤ |p + toRat c| + (u * |p + toRat c| + subErr) := by linarith [he']
        _ = (1 + u) * |p + toRat c| + subErr := by ring
        _ ≤ (1 + u) * (|toRat c| + |p|) + subErr := by
            have := mul_le_mul_of_nonneg_left hr one_add_u_pos.le; linarith
    refine ih (fma a b c) (j + 1) hf' hfin' ?_ (by omega)
    simp only [absSum]
    rw [← hT]
    have hu1 : (0 : ℚ) ≤ 1 + u := one_add_u_pos.le
    have hstep : (|toRat c| + |p|) + T ≤ B * (1 + u) ^ j + δ j := by rw [hpabs]; linarith
    calc |toRat (fma a b c)| + T ≤ (1 + u) * (|toRat c| + |p|) + subErr + T := by linarith
      _ ≤ (1 + u) * ((|toRat c| + |p|) + T) + subErr := by nlinarith [hT0, u_pos]
      _ ≤ (1 + u) * (B * (1 + u) ^ j + δ j) + subErr := by
          have := mul_le_mul_of_nonneg_left hstep hu1; linarith
      _ = B * (1 + u) ^ (j + 1) + δ (j + 1) := by rw [δ_succ, pow_succ]; ring

/-- Same for addition chains. -/
theorem addFin_of_bound (B : ℚ) (hB : 0 ≤ B) (hBmax : 2 * B + 1 ≤ maxNormal) :
    ∀ (l : List Bits) (acc : Bits) (j : ℕ), isFinite acc = true →
      (∀ p ∈ l, isFinite p = true) →
      |toRat acc| + (l.map fun p => |toRat p|).sum ≤ B * (1 + u) ^ j + δ j →
      j + l.length ≤ 2 ^ 23 → AddFin acc l := by
  intro l
  induction l with
  | nil => intro acc j _ _ _ _; trivial
  | cons q l ih =>
    intro acc j hacc hfin hbound hlen
    simp only [List.length_cons] at hlen
    have hq := hfin q List.mem_cons_self
    have hfin' : ∀ p ∈ l, isFinite p = true := fun p hp => hfin p (List.mem_cons_of_mem _ hp)
    simp only [List.map_cons, List.sum_cons] at hbound
    set T := (l.map fun p => |toRat p|).sum with hT
    have hT0 : 0 ≤ T := sum_abs_nonneg l
    have hpow : (1 + u) ^ j ≤ 2 := one_add_u_pow_le_two j (by omega)
    have hδ1 : δ j ≤ 1 := δ_le_one j (by omega)
    have hr : |toRat acc + toRat q| ≤ |toRat acc| + |toRat q| := abs_add_le _ _
    have hmax : |toRat acc + toRat q| ≤ maxNormal := by
      have hB2 : B * (1 + u) ^ j ≤ B * 2 := mul_le_mul_of_nonneg_left hpow hB
      linarith
    refine ⟨hq, hmax, ?_⟩
    obtain ⟨hf', he'⟩ := add_err_abs acc q hacc hq hmax
    have hc'le : |toRat (add acc q)| ≤ (1 + u) * (|toRat acc| + |toRat q|) + subErr := by
      have h1 : |toRat (add acc q)| ≤ |toRat acc + toRat q| + |toRat (add acc q) - (toRat acc + toRat q)| := by
        have := abs_sub_abs_le_abs_sub (toRat (add acc q)) (toRat acc + toRat q)
        linarith
      calc |toRat (add acc q)| ≤ |toRat acc + toRat q| + (u * |toRat acc + toRat q| + subErr) := by linarith [he']
        _ = (1 + u) * |toRat acc + toRat q| + subErr := by ring
        _ ≤ (1 + u) * (|toRat acc| + |toRat q|) + subErr := by
            have := mul_le_mul_of_nonneg_left hr one_add_u_pos.le; linarith
    refine ih (add acc q) (j + 1) hf' hfin' ?_ (by omega)
    have hu1 : (0 : ℚ) ≤ 1 + u := one_add_u_pos.le
    have hstep : (|toRat acc| + |toRat q|) + T ≤ B * (1 + u) ^ j + δ j := by linarith
    calc |toRat (add acc q)| + T ≤ (1 + u) * (|toRat acc| + |toRat q|) + subErr + T := by linarith
      _ ≤ (1 + u) * ((|toRat acc| + |toRat q|) + T) + subErr := by nlinarith [hT0, u_pos]
      _ ≤ (1 + u) * (B * (1 + u) ^ j + δ j) + subErr := by
          have := mul_le_mul_of_nonneg_left hstep hu1; linarith
      _ = B * (1 + u) ^ (j + 1) + δ (j + 1) := by rw [δ_succ, pow_succ]; ring

/-! ### The GEMM twin, absolute form -/

theorem chainAbs_le_gemmAbs {chains : List (List (Bits × Bits))} {ch : List (Bits × Bits)} (h : ch ∈ chains) :
    (ch.map fun p => |toRat p.1| * |toRat p.2|).sum ≤ gemmAbs chains := by
  unfold gemmAbs
  refine List.single_le_sum ?_ _ (List.mem_map_of_mem h)
  intro x hx
  obtain ⟨c, _, rfl⟩ := List.mem_map.1 hx
  exact sum_abs_prod_nonneg c

theorem gemmAbs_nonneg' (chains : List (List (Bits × Bits))) : 0 ≤ gemmAbs chains := by
  unfold gemmAbs
  refine List.sum_nonneg ?_
  intro x hx
  obtain ⟨c, _, rfl⟩ := List.mem_map.1 hx
  exact sum_abs_prod_nonneg c

theorem cast_gemmK (chains : List (List (Bits × Bits))) :
    ((gemmK chains : ℕ) : ℚ) = (chains.map fun ch => (ch.length : ℚ)).sum := by
  unfold gemmK
  rw [Nat.cast_list_sum, List.map_map]
  rfl

/-- **The bf16 GEMM bound, absolute form**: finite fragments whose condition-number numerator
is at most `2^100` (no overflow), `K + s ≤ 2^23`:
`|C − Σ aᵢbᵢ| ≤ 2^-23·(K + s + 1)·Σ|aᵢ||bᵢ| + (K + s + 1)·2^-148`. -/
theorem gemmTwin_err_abs (chains : List (List (Bits × Bits)))
    (hfin : ∀ ch ∈ chains, ∀ p ∈ ch, isFinite p.1 = true ∧ isFinite p.2 = true)
    (habs : gemmAbs chains ≤ 2 ^ 100)
    (hK : gemmK chains + chains.length ≤ 2 ^ 23) :
    isFinite (gemmTwin chains) = true ∧
    |toRat (gemmTwin chains) - gemmExact chains| ≤
      (2 : ℚ) ^ (-23 : ℤ) * ((gemmK chains : ℚ) + chains.length + 1) * gemmAbs chains +
        ((gemmK chains : ℚ) + chains.length + 1) * (2 : ℚ) ^ (-148 : ℤ) := by
  set K := gemmK chains with hKdef
  set s := chains.length with hsdef
  set T := gemmAbs chains with hTdef
  have hT0 : 0 ≤ T := gemmAbs_nonneg' chains
  have hmaxB : 2 * (2 : ℚ) ^ 100 + 1 ≤ maxNormal := by unfold maxNormal; norm_num
  have hKle : K ≤ 2 ^ 23 := by omega
  have hsle : s ≤ 2 ^ 23 := by omega
  -- per chain: finite, and the absolute-form error
  have hper : ∀ ch ∈ chains,
      isFinite (fmaChain posZero ch) = true ∧
      |toRat (fmaChain posZero ch) - (ch.map fun p => toRat p.1 * toRat p.2).sum| ≤
        γ K * (ch.map fun p => |toRat p.1| * |toRat p.2|).sum + δ ch.length := by
    intro ch hc
    have hfinc : ChainFin posZero ch := by
      refine chainFin_of_bound (2 ^ 100) (by positivity) hmaxB ch posZero 0 isFinite_posZero (hfin ch hc) ?_ ?_
      · simp only [absSum, toRat_posZero, abs_zero, zero_add, pow_zero, mul_one, δ_zero, add_zero]
        exact (chainAbs_le_gemmAbs hc).trans habs
      · have := length_le_gemmK hc; omega
    obtain ⟨hf, he⟩ := fmaChain_err_abs posZero ch isFinite_posZero hfinc
    refine ⟨hf, ?_⟩
    simp only [exactSum, absSum, toRat_posZero, zero_add, abs_zero] at he
    exact he.trans (by
      linarith [mul_le_mul_of_nonneg_right (γ_mono (length_le_gemmK hc)) (sum_abs_prod_nonneg ch)])
  -- the sum of the per-chain absolute terms
  have hδsum : (chains.map fun ch => δ ch.length).sum ≤ (K : ℚ) * (2 * subErr) := by
    have h1 : ∀ ch ∈ chains, δ ch.length ≤ (ch.length : ℚ) * (2 * subErr) :=
      fun ch hc => δ_le_linear _ (by have := length_le_gemmK hc; omega)
    calc (chains.map fun ch => δ ch.length).sum
        ≤ (chains.map fun ch => (ch.length : ℚ) * (2 * subErr)).sum := List.sum_le_sum h1
      _ = (chains.map fun ch => (2 * subErr) * (ch.length : ℚ)).sum := by simp_rw [mul_comm]
      _ = (2 * subErr) * (chains.map fun ch => (ch.length : ℚ)).sum := list_sum_map_mul_left _ _ _
      _ = (K : ℚ) * (2 * subErr) := by rw [hKdef, cast_gemmK]; ring
  -- the addition chain
  set P : ℚ := (chains.map fun ch => toRat (fmaChain posZero ch)).sum with hPdef
  set A : ℚ := (chains.map fun ch => |toRat (fmaChain posZero ch)|).sum with hAdef
  -- |Σ Pᵢ − Σ Sᵢ| ≤ γ_K · T + Σ δ
  have hPS : |P - gemmExact chains| ≤ γ K * T + (K : ℚ) * (2 * subErr) := by
    rw [hPdef]; unfold gemmExact
    rw [list_sum_map_sub]
    refine (list_abs_sum_le _).trans ?_
    rw [List.map_map]
    calc (chains.map (abs ∘ fun ch => toRat (fmaChain posZero ch) - (ch.map fun p => toRat p.1 * toRat p.2).sum)).sum
        ≤ (chains.map fun ch => γ K * (ch.map fun p => |toRat p.1| * |toRat p.2|).sum + δ ch.length).sum := by
          refine List.sum_le_sum ?_
          intro ch hc
          exact (hper ch hc).2
      _ = γ K * T + (chains.map fun ch => δ ch.length).sum := by
          rw [List.sum_map_add, list_sum_map_mul_left, hTdef]; rfl
      _ ≤ γ K * T + (K : ℚ) * (2 * subErr) := by linarith [hδsum]
  -- Σ |Pᵢ| ≤ (1 + γ_K) · T + Σ δ
  have hA : A ≤ (1 + γ K) * T + (K : ℚ) * (2 * subErr) := by
    rw [hAdef]
    calc (chains.map fun ch => |toRat (fmaChain posZero ch)|).sum
        ≤ (chains.map fun ch => (1 + γ K) * (ch.map fun p => |toRat p.1| * |toRat p.2|).sum + δ ch.length).sum := by
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
      _ = (1 + γ K) * T + (chains.map fun ch => δ ch.length).sum := by
          rw [List.sum_map_add, list_sum_map_mul_left, hTdef]; rfl
      _ ≤ (1 + γ K) * T + (K : ℚ) * (2 * subErr) := by linarith [hδsum]
  have hγK1 : γ K ≤ 1 := by
    have := γ_le_linear K hKle
    have hu : u = 1 / 2 ^ 24 := rfl
    have hK' : (K : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hKle
    have : (K : ℚ) * (2 * u) ≤ 1 := by rw [hu]; norm_num; linarith
    linarith
  have hγs1 : γ s ≤ 1 := by
    have := γ_le_linear s hsle
    have hu : u = 1 / 2 ^ 24 := rfl
    have hs' : (s : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hsle
    have : (s : ℚ) * (2 * u) ≤ 1 := by rw [hu]; norm_num; linarith
    linarith
  have hKsub : (K : ℚ) * (2 * subErr) ≤ 1 := by
    have hsE : subErr = 1 / 2 ^ 150 := rfl
    have hK' : (K : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hKle
    rw [hsE]; norm_num; linarith
  have hA' : A ≤ 2 * 2 ^ 100 + 1 := by
    have : (1 + γ K) * T ≤ 2 * T := by nlinarith [hγK1, hT0, γ_nonneg K]
    linarith
  have hfinP : ∀ p ∈ chains.map (fmaChain posZero), isFinite p = true := by
    intro p hp
    obtain ⟨ch, hc, rfl⟩ := List.mem_map.1 hp
    exact (hper ch hc).1
  have haddfin : AddFin posZero (chains.map (fmaChain posZero)) := by
    refine addFin_of_bound (2 * 2 ^ 100 + 1) (by positivity) (by unfold maxNormal; norm_num) _ posZero 0
      isFinite_posZero hfinP ?_ (by simp only [List.length_map]; omega)
    simp only [toRat_posZero, abs_zero, zero_add, pow_zero, mul_one, δ_zero, add_zero, List.map_map]
    exact hA'
  obtain ⟨hfin', hsum⟩ := addChain_err_abs posZero (chains.map (fmaChain posZero)) isFinite_posZero haddfin
  simp only [toRat_posZero, zero_add, abs_zero, List.map_map, List.length_map, Function.comp_def] at hsum
  refine ⟨hfin', ?_⟩
  have hγs := γ_nonneg s
  have hδs : δ s ≤ (s : ℚ) * (2 * subErr) := δ_le_linear s hsle
  -- combine
  have hC : |toRat (gemmTwin chains) - gemmExact chains| ≤
      γ s * ((1 + γ K) * T + (K : ℚ) * (2 * subErr)) + δ s + (γ K * T + (K : ℚ) * (2 * subErr)) := by
    have h1 : toRat (gemmTwin chains) - gemmExact chains =
        (toRat (gemmTwin chains) - P) + (P - gemmExact chains) := by ring
    rw [h1]
    refine (abs_add_le _ _).trans ?_
    have : |toRat (gemmTwin chains) - P| ≤ γ s * A + δ s := hsum
    have h2 : γ s * A ≤ γ s * ((1 + γ K) * T + (K : ℚ) * (2 * subErr)) := mul_le_mul_of_nonneg_left hA hγs
    linarith [hPS]
  have hlin : γ (s + K) ≤ ((s + K : ℕ) : ℚ) * (2 * u) := γ_le_linear (s + K) (by omega)
  have h2u : (2 : ℚ) * u = 2 ^ (-23 : ℤ) := by unfold u; norm_num
  have h4s : (4 : ℚ) * subErr = 2 ^ (-148 : ℤ) := by unfold subErr; norm_num
  have hKnn : (0 : ℚ) ≤ K := by positivity
  have hsnn : (0 : ℚ) ≤ s := by positivity
  have hsub0 := subErr_pos.le
  -- the relative part
  have hrel : γ s * ((1 + γ K) * T) + γ K * T ≤ 2 ^ (-23 : ℤ) * ((K : ℚ) + s + 1) * T := by
    calc γ s * ((1 + γ K) * T) + γ K * T = γ (s + K) * T := by rw [γ_add]; ring
      _ ≤ ((s + K : ℕ) : ℚ) * (2 * u) * T := mul_le_mul_of_nonneg_right hlin hT0
      _ = 2 ^ (-23 : ℤ) * ((s : ℚ) + K) * T := by rw [h2u]; push_cast; ring
      _ ≤ 2 ^ (-23 : ℤ) * ((K : ℚ) + s + 1) * T := by
          have hp := two_zpow_pos (-23 : ℤ)
          have : (s : ℚ) + K ≤ (K : ℚ) + s + 1 := by linarith
          exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left this hp.le) hT0
  -- the absolute part
  have habsp : γ s * ((K : ℚ) * (2 * subErr)) + δ s + (K : ℚ) * (2 * subErr) ≤
      ((K : ℚ) + s + 1) * 2 ^ (-148 : ℤ) := by
    rw [← h4s]
    have : γ s * ((K : ℚ) * (2 * subErr)) ≤ (K : ℚ) * (2 * subErr) := by
      have hx : 0 ≤ (K : ℚ) * (2 * subErr) := by positivity
      nlinarith [hγs1, hx]
    nlinarith [hδs, hsnn, hKnn, hsub0]
  calc |toRat (gemmTwin chains) - gemmExact chains|
      ≤ γ s * ((1 + γ K) * T + (K : ℚ) * (2 * subErr)) + δ s + (γ K * T + (K : ℚ) * (2 * subErr)) := hC
    _ = (γ s * ((1 + γ K) * T) + γ K * T) + (γ s * ((K : ℚ) * (2 * subErr)) + δ s + (K : ℚ) * (2 * subErr)) := by ring
    _ ≤ 2 ^ (-23 : ℤ) * ((K : ℚ) + s + 1) * T + ((K : ℚ) + s + 1) * 2 ^ (-148 : ℤ) := add_le_add hrel habsp

end TVA.Math.Float.F32
