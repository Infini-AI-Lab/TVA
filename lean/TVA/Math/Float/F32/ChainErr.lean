import TVA.Math.Float.F32.OpErr

/-!
# FMA chains and addition chains: `fmaChain_err`, `addChain_err`, `γ` and its linear form

Part of the `f32` error analysis (`TVA.Math.Float.F32.Error`); the definitions it reasons about are in
`TVA.Math.Float.F32.Bounds`. Split out of `F32/Bounds.lean`.
-/
set_option autoImplicit false

namespace TVA.Math.Float.F32

open TVA.Math.Float

open Rat

/-! ### FMA chains -/

theorem γ_zero : γ 0 = 0 := by simp [γ]
theorem γ_succ (n : ℕ) : γ (n + 1) = (1 + u) * γ n + u := by unfold γ; ring
theorem one_add_u_pos : (0 : ℚ) < 1 + u := by linarith [u_pos]
theorem γ_nonneg (n : ℕ) : 0 ≤ γ n := by
  unfold γ
  have : (1 : ℚ) ≤ (1 + u) ^ n := one_le_pow₀ (by linarith [u_pos])
  linarith
theorem γ_mono {m n : ℕ} (h : m ≤ n) : γ m ≤ γ n := by
  unfold γ
  have := pow_le_pow_right₀ (by linarith [u_pos] : (1 : ℚ) ≤ 1 + u) h
  linarith
theorem γ_add (m n : ℕ) : γ (m + n) = γ m + γ n + γ m * γ n := by unfold γ; rw [pow_add]; ring

theorem absSum_nonneg (c : Bits) (l : List (Bits × Bits)) : 0 ≤ absSum c l := by
  unfold absSum
  refine add_nonneg (abs_nonneg _) (List.sum_nonneg ?_)
  intro x hx
  rw [List.mem_map] at hx
  obtain ⟨p, _, rfl⟩ := hx
  positivity

theorem sum_abs_prod_nonneg (l : List (Bits × Bits)) :
    0 ≤ (l.map fun p => |toRat p.1| * |toRat p.2|).sum := by
  refine List.sum_nonneg ?_
  intro x hx
  rw [List.mem_map] at hx
  obtain ⟨p, _, rfl⟩ := hx
  positivity

/-- **FMA chain error**: `|chain − exact| ≤ γₙ·(|c| + Σ|aᵢ||bᵢ|)`. -/
theorem fmaChain_err (c : Bits) (l : List (Bits × Bits)) (hc : isFinite c = true) (h : ChainOK c l) :
    isFinite (fmaChain c l) = true ∧
    |toRat (fmaChain c l) - exactSum c l| ≤ γ l.length * absSum c l := by
  induction l generalizing c with
  | nil => simp [fmaChain, exactSum, absSum, γ_zero, hc]
  | cons q l ih =>
    obtain ⟨a, b⟩ := q
    obtain ⟨ha, hb, hr, hmax, hrest⟩ := h
    obtain ⟨hf1, he1⟩ := fma_err a b c ha hb hc hr hmax
    obtain ⟨hf, he⟩ := ih (fma a b c) hf1 hrest
    refine ⟨by simpa [fmaChain] using hf, ?_⟩
    simp only [fmaChain, exactSum, absSum, List.map_cons, List.sum_cons, List.length_cons] at he ⊢
    set c' := fma a b c with hc'
    set p := toRat a * toRat b with hp
    set S := (l.map fun p => toRat p.1 * toRat p.2).sum with hS
    set T := (l.map fun p => |toRat p.1| * |toRat p.2|).sum with hT
    have hT0 : 0 ≤ T := sum_abs_prod_nonneg l
    have hγ := γ_nonneg l.length
    have hpabs : |p| = |toRat a| * |toRat b| := abs_mul _ _
    have hsum : |p + toRat c| ≤ |toRat c| + |p| := by rw [add_comm]; exact abs_add_le _ _
    have hc'le : |toRat c'| ≤ (1 + u) * (|toRat c| + |p|) := by
      have h1 : |toRat c'| ≤ |p + toRat c| + |toRat c' - (p + toRat c)| := by
        have := abs_sub_abs_le_abs_sub (toRat c') (p + toRat c)
        linarith
      calc |toRat c'| ≤ |p + toRat c| + u * |p + toRat c| := by linarith [he1]
        _ = (1 + u) * |p + toRat c| := by ring
        _ ≤ (1 + u) * (|toRat c| + |p|) := mul_le_mul_of_nonneg_left hsum one_add_u_pos.le
    have hsplit : toRat (fmaChain c' l) - (toRat c + (p + S)) =
        (toRat (fmaChain c' l) - (toRat c' + S)) + (toRat c' - (p + toRat c)) := by ring
    have h2 : |toRat (fmaChain c' l) - (toRat c + (p + S))| ≤
        γ l.length * (|toRat c'| + T) + u * |p + toRat c| := by
      rw [hsplit]
      exact (abs_add_le _ _).trans (add_le_add he he1)
    have h3 : γ l.length * (|toRat c'| + T) ≤ γ l.length * ((1 + u) * (|toRat c| + |p|) + T) :=
      mul_le_mul_of_nonneg_left (by linarith) hγ
    have h4 : u * |p + toRat c| ≤ u * (|toRat c| + |p|) := mul_le_mul_of_nonneg_left hsum u_pos.le
    have h5 : γ l.length * T ≤ γ (l.length + 1) * T := mul_le_mul_of_nonneg_right (γ_mono (by omega)) hT0
    rw [hpabs.symm]
    calc |toRat (fmaChain c' l) - (toRat c + (p + S))|
        ≤ γ l.length * (|toRat c'| + T) + u * |p + toRat c| := h2
      _ ≤ γ l.length * ((1 + u) * (|toRat c| + |p|) + T) + u * (|toRat c| + |p|) := add_le_add h3 h4
      _ = ((1 + u) * γ l.length + u) * (|toRat c| + |p|) + γ l.length * T := by ring
      _ ≤ γ (l.length + 1) * (|toRat c| + |p|) + γ (l.length + 1) * T := by
          rw [← γ_succ]; linarith [h5]
      _ = γ (l.length + 1) * (|toRat c| + (|p| + T)) := by ring

/-! ### Addition chains (the atomic fix-up) -/

theorem sum_abs_nonneg (l : List Bits) : 0 ≤ (l.map fun p => |toRat p|).sum := by
  refine List.sum_nonneg ?_
  intro x hx
  rw [List.mem_map] at hx
  obtain ⟨p, _, rfl⟩ := hx
  positivity

/-- **Addition chain error**: `|chain − (acc + Σpᵢ)| ≤ γₘ·(|acc| + Σ|pᵢ|)`. -/
theorem addChain_err (acc : Bits) (l : List Bits) (hacc : isFinite acc = true) (h : AddOK acc l) :
    isFinite (addChain acc l) = true ∧
    |toRat (addChain acc l) - (toRat acc + (l.map toRat).sum)| ≤
      γ l.length * (|toRat acc| + (l.map fun p => |toRat p|).sum) := by
  induction l generalizing acc with
  | nil => simp [addChain, γ_zero, hacc]
  | cons q l ih =>
    obtain ⟨hq, hr, hmax, hrest⟩ := h
    obtain ⟨hf1, he1⟩ := add_err acc q hacc hq hr hmax
    obtain ⟨hf, he⟩ := ih (add acc q) hf1 hrest
    refine ⟨by simpa [addChain] using hf, ?_⟩
    simp only [addChain, List.map_cons, List.sum_cons, List.length_cons] at he ⊢
    set c' := add acc q with hc'
    set S := (l.map toRat).sum with hS
    set T := (l.map fun p => |toRat p|).sum with hT
    have hT0 : 0 ≤ T := sum_abs_nonneg l
    have hγ := γ_nonneg l.length
    have hsum : |toRat acc + toRat q| ≤ |toRat acc| + |toRat q| := abs_add_le _ _
    have hc'le : |toRat c'| ≤ (1 + u) * (|toRat acc| + |toRat q|) := by
      have h1 : |toRat c'| ≤ |toRat acc + toRat q| + |toRat c' - (toRat acc + toRat q)| := by
        have := abs_sub_abs_le_abs_sub (toRat c') (toRat acc + toRat q)
        linarith
      calc |toRat c'| ≤ |toRat acc + toRat q| + u * |toRat acc + toRat q| := by linarith [he1]
        _ = (1 + u) * |toRat acc + toRat q| := by ring
        _ ≤ (1 + u) * (|toRat acc| + |toRat q|) := mul_le_mul_of_nonneg_left hsum one_add_u_pos.le
    have hsplit : toRat (addChain c' l) - (toRat acc + (toRat q + S)) =
        (toRat (addChain c' l) - (toRat c' + S)) + (toRat c' - (toRat acc + toRat q)) := by ring
    have h2 : |toRat (addChain c' l) - (toRat acc + (toRat q + S))| ≤
        γ l.length * (|toRat c'| + T) + u * |toRat acc + toRat q| := by
      rw [hsplit]
      exact (abs_add_le _ _).trans (add_le_add he he1)
    have h3 : γ l.length * (|toRat c'| + T) ≤ γ l.length * ((1 + u) * (|toRat acc| + |toRat q|) + T) :=
      mul_le_mul_of_nonneg_left (by linarith) hγ
    have h4 : u * |toRat acc + toRat q| ≤ u * (|toRat acc| + |toRat q|) :=
      mul_le_mul_of_nonneg_left hsum u_pos.le
    have h5 : γ l.length * T ≤ γ (l.length + 1) * T := mul_le_mul_of_nonneg_right (γ_mono (by omega)) hT0
    calc |toRat (addChain c' l) - (toRat acc + (toRat q + S))|
        ≤ γ l.length * (|toRat c'| + T) + u * |toRat acc + toRat q| := h2
      _ ≤ γ l.length * ((1 + u) * (|toRat acc| + |toRat q|) + T) + u * (|toRat acc| + |toRat q|) := add_le_add h3 h4
      _ = ((1 + u) * γ l.length + u) * (|toRat acc| + |toRat q|) + γ l.length * T := by ring
      _ ≤ γ (l.length + 1) * (|toRat acc| + |toRat q|) + γ (l.length + 1) * T := by
          rw [← γ_succ]; linarith [h5]
      _ = γ (l.length + 1) * (|toRat acc| + (|toRat q| + T)) := by ring

/-! ### The linear form of `γ` -/

/-- `γₙ ≤ 2·n·u = n·2^-23` for `n ≤ 2^23`. -/
theorem γ_le_linear (n : ℕ) (hn : n ≤ 2 ^ 23) : γ n ≤ (n : ℚ) * (2 * u) := by
  induction n with
  | zero => simp [γ_zero]
  | succ k ih =>
    have hk : k ≤ 2 ^ 23 := by omega
    have ihk := ih hk
    have hk' : (k : ℚ) ≤ 2 ^ 23 := by exact_mod_cast hk
    have hu : u = 1 / 2 ^ 24 := rfl
    have key : 2 * (k : ℚ) * u * u ≤ u := by
      rw [hu]; norm_num; linarith
    have h1 : (1 + u) * γ k ≤ (1 + u) * ((k : ℚ) * (2 * u)) :=
      mul_le_mul_of_nonneg_left ihk one_add_u_pos.le
    rw [γ_succ]; push_cast
    nlinarith [h1, key, u_pos]

end TVA.Math.Float.F32
