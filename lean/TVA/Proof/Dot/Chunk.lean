import TVA.Proof.Dot.Lift

/-!
# Composing the chunks of a `tl.dot`

The fp8 and 16-bit `tl.dot` facts (`Device.DotChunkAcc`, `Device.DotChunkLoss`,
`Device.DotChunk16Loss`) describe one dot's accumulation along `K`: partial values
`s 0 = acc, s 1, …, s n = result`, one per chunk of consecutive `k` (32 for fp8, 64 for 16-bit
operands), each step adding the chunk's exact sum `c j` up to an error bounded by a bound `Mj` on
the terms the chunk aligns to. This file composes such steps into a bound from the exact partial
sums alone; it is pure arithmetic on sequences, used by `TVA.Proof.Dot.Chunk8` (chunks of 32) and
`TVA.Proof.Dot.Chunk16` (chunks of 64).

The bounds are those of `TVA.Device.Tolerance`, which claims name: `Dot8Chunk.psum`,
`Dot8Chunk.errB` (each step within `ρ·Mj + σ·|s j + c j| + α`) and `Dot8Chunk.errM` (each step
within the smaller of `ρ·Mj` and `κ·a j + ρ'·Mj`, plus the same terms).

* `Dot8Chunk.chain_err`: a chain of steps bounded as `DotChunkAcc` bounds them stays within
  `errB` of the exact partial sums; `errB_nonneg`, `errB_le` (closed form).
* `Dot8Chunk.chain_errM`: the same for `DotChunkLoss`'s minimum, within `errM`; `errM_nonneg`,
  `errM_le_errB` (never worse than `errB`), `errL` (the second arm alone), `errM_le_errL`,
  `errL_le` (closed form).
* `Dot8Chunk.psum_zero`, `psum_succ`.
-/

namespace TVA

namespace Dot8Chunk

/-- The partial sum before any chunk is the start. -/
@[simp] theorem psum_zero (v : ℚ) (c : ℕ → ℚ) : psum v c 0 = v := by simp [psum]

/-- One more chunk adds its sum. -/
theorem psum_succ (v : ℚ) (c : ℕ → ℚ) (j : ℕ) : psum v c (j + 1) = psum v c j + c j := by
  simp [psum, Finset.sum_range_succ, add_assoc]

/-! ## Each step within `ρ·Mj + σ·|s j + c j| + α` -/

/-- `errB` is nonnegative for nonnegative constants. -/
theorem errB_nonneg {ρ σ α : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) (v : ℚ) (c m : ℕ → ℚ) :
    ∀ j, 0 ≤ errB ρ σ α v c m j
  | 0 => le_rfl
  | j + 1 => by
    have ih := errB_nonneg hρ hσ hα v c m j
    have h1 : 0 ≤ max |psum v c j| (m j) := (abs_nonneg _).trans (le_max_left _ _)
    simp only [errB]
    have : 0 ≤ (1 + ρ + σ) * errB ρ σ α v c m j := mul_nonneg (by linarith) ih
    have : 0 ≤ ρ * max |psum v c j| (m j) := mul_nonneg hρ h1
    have : 0 ≤ σ * |psum v c (j + 1)| := mul_nonneg hσ (abs_nonneg _)
    linarith

/-- **A chain of chunk steps.** If each step `j < n` adds `c j` up to `ρ·Mj + σ·|s j + c j| + α`
for every `Mj` at least `|s j|` and `m j`, then after `j ≤ n` steps `s j` is within `errB` of the
exact partial sum `psum (s 0) c j`. Use it on the partial values of a `DotChunkAcc` fact. -/
theorem chain_err {ρ σ α : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) {s c m : ℕ → ℚ} {n : ℕ}
    (hstep : ∀ j < n, ∀ Mj : ℚ, |s j| ≤ Mj → m j ≤ Mj →
      |s (j + 1) - (s j + c j)| ≤ ρ * Mj + σ * |s j + c j| + α) :
    ∀ j ≤ n, |s j - psum (s 0) c j| ≤ errB ρ σ α (s 0) c m j := by
  intro j
  induction j with
  | zero => intro _; simp [errB]
  | succ j ih =>
    intro hj
    have e := ih (by omega)
    set S := psum (s 0) c j with hS
    set E := errB ρ σ α (s 0) c m j with hE
    have hE0 : 0 ≤ E := (abs_nonneg _).trans e
    have hsj : |s j| ≤ |S| + E := by
      have := abs_sub_abs_le_abs_sub (s j) S
      linarith
    have hstep' := hstep j (by omega) (max (|S| + E) (m j)) (hsj.trans (le_max_left _ _))
      (le_max_right _ _)
    have hsc : |s j + c j| ≤ |S + c j| + E := by
      have h1 : s j + c j = (S + c j) + (s j - S) := by ring
      rw [h1]
      exact (abs_add_le _ _).trans (by linarith)
    have hmax : max (|S| + E) (m j) ≤ max |S| (m j) + E := by
      apply max_le
      · linarith [le_max_left |S| (m j)]
      · linarith [le_max_right |S| (m j)]
    have hS1 : psum (s 0) c (j + 1) = S + c j := psum_succ _ _ _
    have htri : |s (j + 1) - (S + c j)| ≤ |s (j + 1) - (s j + c j)| + |s j - S| := by
      have h1 : s (j + 1) - (S + c j) = (s (j + 1) - (s j + c j)) + (s j - S) := by ring
      rw [h1]; exact abs_add_le _ _
    show |s (j + 1) - psum (s 0) c (j + 1)| ≤
      (1 + ρ + σ) * E + ρ * max |S| (m j) + σ * |psum (s 0) c (j + 1)| + α
    rw [hS1]
    have h2 := mul_le_mul_of_nonneg_left hmax hρ
    have h3 := mul_le_mul_of_nonneg_left hsc hσ
    nlinarith

/-- **The closed form of `errB`**:
`errB j ≤ (1+ρ+σ)^j · Σ_{i<j} (ρ·max |psum i| (m i) + σ·|psum (i+1)| + α)`. Use it to state a
tolerance without the recursion. -/
theorem errB_le {ρ σ α : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) (v : ℚ) (c m : ℕ → ℚ) :
    ∀ j, errB ρ σ α v c m j ≤ (1 + ρ + σ) ^ j *
      ∑ i ∈ Finset.range j, (ρ * max |psum v c i| (m i) + σ * |psum v c (i + 1)| + α)
  | 0 => by simp [errB]
  | j + 1 => by
    have ih := errB_le hρ hσ hα v c m j
    set T := ∑ i ∈ Finset.range j, (ρ * max |psum v c i| (m i) + σ * |psum v c (i + 1)| + α)
    set t := ρ * max |psum v c j| (m j) + σ * |psum v c (j + 1)| + α
    have ht : 0 ≤ t := by
      have h1 : 0 ≤ max |psum v c j| (m j) := (abs_nonneg _).trans (le_max_left _ _)
      have := mul_nonneg hρ h1
      have := mul_nonneg hσ (abs_nonneg (psum v c (j + 1)))
      linarith
    have hq : 1 ≤ 1 + ρ + σ := by linarith
    have hqj : 1 ≤ (1 + ρ + σ) ^ (j + 1) := one_le_pow₀ hq
    have hT : 0 ≤ T := Finset.sum_nonneg fun i _ => by
      have h1 : 0 ≤ max |psum v c i| (m i) := (abs_nonneg _).trans (le_max_left _ _)
      have := mul_nonneg hρ h1
      have := mul_nonneg hσ (abs_nonneg (psum v c (i + 1)))
      linarith
    rw [Finset.sum_range_succ]
    rw [show errB ρ σ α v c m (j + 1) = (1 + ρ + σ) * errB ρ σ α v c m j + t by
      simp only [errB, t]; ring]
    have e1 : (1 + ρ + σ) * errB ρ σ α v c m j ≤ (1 + ρ + σ) * ((1 + ρ + σ) ^ j * T) :=
      mul_le_mul_of_nonneg_left ih (by linarith)
    have e2 : (1 + ρ + σ) * ((1 + ρ + σ) ^ j * T) = (1 + ρ + σ) ^ (j + 1) * T := by ring
    have e3 : t ≤ (1 + ρ + σ) ^ (j + 1) * t := le_mul_of_one_le_left ht hqj
    nlinarith

/-! ## Each step within the smaller of two bounds -/

/-- `errM` is nonnegative for nonnegative constants and chunk magnitudes. -/
theorem errM_nonneg {ρ σ α κ ρ' : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) (hκ : 0 ≤ κ)
    (hρ' : 0 ≤ ρ') (v : ℚ) (c : ℕ → ℚ) {a : ℕ → ℚ} (ha : ∀ j, 0 ≤ a j) (m : ℕ → ℚ) :
    ∀ j, 0 ≤ errM ρ σ α κ ρ' v c a m j
  | 0 => le_rfl
  | j + 1 => by
    have ih := errM_nonneg hρ hσ hα hκ hρ' v c ha m j
    have hX : 0 ≤ max |psum v c j| (m j) + errM ρ σ α κ ρ' v c a m j :=
      add_nonneg ((abs_nonneg _).trans (le_max_left _ _)) ih
    have h1 := le_min (mul_nonneg hρ hX) (add_nonneg (mul_nonneg hκ (ha j)) (mul_nonneg hρ' hX))
    have h2 : 0 ≤ (1 + σ) * errM ρ σ α κ ρ' v c a m j := mul_nonneg (by linarith) ih
    have h3 : 0 ≤ σ * |psum v c (j + 1)| := mul_nonneg hσ (abs_nonneg _)
    simp only [errM]
    linarith

/-- **A chain of chunk steps, each within the smaller of two bounds.** If each step `j < n` adds
`c j` up to `min (ρ·Mj) (κ·a j + ρ'·Mj) + σ·|s j + c j| + α` for every `Mj` at least `|s j|` and
`m j`, then after `j ≤ n` steps `s j` is within `errM` of `psum (s 0) c j`. Use it on the partial
values of a `DotChunkLoss` or `DotChunk16Loss` fact. -/
theorem chain_errM {ρ σ α κ ρ' : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hρ' : 0 ≤ ρ') {s c a m : ℕ → ℚ}
    {n : ℕ}
    (hstep : ∀ j < n, ∀ Mj : ℚ, |s j| ≤ Mj → m j ≤ Mj →
      |s (j + 1) - (s j + c j)| ≤ min (ρ * Mj) (κ * a j + ρ' * Mj) + σ * |s j + c j| + α) :
    ∀ j ≤ n, |s j - psum (s 0) c j| ≤ errM ρ σ α κ ρ' (s 0) c a m j := by
  intro j
  induction j with
  | zero => intro _; simp [errM]
  | succ j ih =>
    intro hj
    have e := ih (by omega)
    set S := psum (s 0) c j with hS
    set E := errM ρ σ α κ ρ' (s 0) c a m j with hE
    set X := max |S| (m j) with hX
    have hE0 : 0 ≤ E := (abs_nonneg _).trans e
    have hsj : |s j| ≤ |S| + E := by
      have := abs_sub_abs_le_abs_sub (s j) S
      linarith
    have hstep' := hstep j (by omega) (max (|S| + E) (m j)) (hsj.trans (le_max_left _ _))
      (le_max_right _ _)
    have hmax : max (|S| + E) (m j) ≤ X + E := by
      apply max_le
      · linarith [le_max_left |S| (m j)]
      · linarith [le_max_right |S| (m j)]
    have hmin : min (ρ * max (|S| + E) (m j)) (κ * a j + ρ' * max (|S| + E) (m j)) ≤
        min (ρ * (X + E)) (κ * a j + ρ' * (X + E)) :=
      min_le_min (mul_le_mul_of_nonneg_left hmax hρ)
        (add_le_add le_rfl (mul_le_mul_of_nonneg_left hmax hρ'))
    have hsc : |s j + c j| ≤ |S + c j| + E := by
      have h1 : s j + c j = (S + c j) + (s j - S) := by ring
      rw [h1]
      exact (abs_add_le _ _).trans (by linarith)
    have hS1 : psum (s 0) c (j + 1) = S + c j := psum_succ _ _ _
    have htri : |s (j + 1) - (S + c j)| ≤ |s (j + 1) - (s j + c j)| + |s j - S| := by
      have h1 : s (j + 1) - (S + c j) = (s (j + 1) - (s j + c j)) + (s j - S) := by ring
      rw [h1]; exact abs_add_le _ _
    have h3 := mul_le_mul_of_nonneg_left hsc hσ
    show |s (j + 1) - psum (s 0) c (j + 1)| ≤
      (1 + σ) * E + min (ρ * (X + E)) (κ * a j + ρ' * (X + E)) + σ * |psum (s 0) c (j + 1)| + α
    rw [hS1]
    linarith

/-- The minimum is never worse than its first arm: `errM ≤ errB`. Use it to reuse an `errB`
tolerance under a `DotChunkLoss` fact. -/
theorem errM_le_errB {ρ σ α κ ρ' : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (v : ℚ) (c a m : ℕ → ℚ) :
    ∀ j, errM ρ σ α κ ρ' v c a m j ≤ errB ρ σ α v c m j
  | 0 => le_rfl
  | j + 1 => by
    have ih := errM_le_errB (α := α) (κ := κ) (ρ' := ρ') hρ hσ v c a m j
    have h1 := min_le_left (ρ * (max |psum v c j| (m j) + errM ρ σ α κ ρ' v c a m j))
      (κ * a j + ρ' * (max |psum v c j| (m j) + errM ρ σ α κ ρ' v c a m j))
    have h2 := mul_le_mul_of_nonneg_left ih (by linarith : (0 : ℚ) ≤ 1 + σ)
    have h3 := mul_le_mul_of_nonneg_left ih hρ
    simp only [errM, errB]
    linarith

/-- The error after `j` chunks with the second arm of the minimum alone, each chunk charged
`κ·a j + ρ'·max |psum v c j| (m j)` (`a j` its exact magnitude) and the error so far grown by
`1 + ρ' + σ`: `errL 0 = 0` and
`errL (j+1) = (1+ρ'+σ)·errL j + κ·a j + ρ'·max |psum v c j| (m j) + σ·|psum v c (j+1)| + α`.
It bounds `errM` (`errM_le_errL`) where the chunks are small against the running sum. -/
def errL (σ α κ ρ' v : ℚ) (c a m : ℕ → ℚ) : ℕ → ℚ
  | 0 => 0
  | j + 1 => (1 + ρ' + σ) * errL σ α κ ρ' v c a m j + κ * a j + ρ' * max |psum v c j| (m j) +
      σ * |psum v c (j + 1)| + α

/-- The minimum is never worse than its second arm: `errM ≤ errL`. -/
theorem errM_le_errL {ρ σ α κ ρ' : ℚ} (hσ : 0 ≤ σ) (hρ' : 0 ≤ ρ') (v : ℚ) (c a m : ℕ → ℚ) :
    ∀ j, errM ρ σ α κ ρ' v c a m j ≤ errL σ α κ ρ' v c a m j
  | 0 => le_rfl
  | j + 1 => by
    have ih := errM_le_errL (ρ := ρ) (α := α) (κ := κ) hσ hρ' v c a m j
    have h1 := min_le_right (ρ * (max |psum v c j| (m j) + errM ρ σ α κ ρ' v c a m j))
      (κ * a j + ρ' * (max |psum v c j| (m j) + errM ρ σ α κ ρ' v c a m j))
    have h2 := mul_le_mul_of_nonneg_left ih (by linarith : (0 : ℚ) ≤ 1 + σ)
    have h3 := mul_le_mul_of_nonneg_left ih hρ'
    simp only [errM, errL]
    linarith

/-- **The closed form of `errL`**:
`errL j ≤ (1+ρ'+σ)^j · Σ_{i<j} (κ·a i + ρ'·max |psum i| (m i) + σ·|psum (i+1)| + α)`. -/
theorem errL_le {σ α κ ρ' : ℚ} (hσ : 0 ≤ σ) (hα : 0 ≤ α) (hκ : 0 ≤ κ) (hρ' : 0 ≤ ρ') (v : ℚ)
    (c : ℕ → ℚ) {a : ℕ → ℚ} (ha : ∀ j, 0 ≤ a j) (m : ℕ → ℚ) :
    ∀ j, errL σ α κ ρ' v c a m j ≤ (1 + ρ' + σ) ^ j *
      ∑ i ∈ Finset.range j, (κ * a i + ρ' * max |psum v c i| (m i) + σ * |psum v c (i + 1)| + α)
  | 0 => by simp [errL]
  | j + 1 => by
    have ih := errL_le hσ hα hκ hρ' v c ha m j
    have hterm : ∀ i, 0 ≤ κ * a i + ρ' * max |psum v c i| (m i) + σ * |psum v c (i + 1)| + α :=
      fun i => by
        have h1 : 0 ≤ max |psum v c i| (m i) := (abs_nonneg _).trans (le_max_left _ _)
        have := mul_nonneg hκ (ha i)
        have := mul_nonneg hρ' h1
        have := mul_nonneg hσ (abs_nonneg (psum v c (i + 1)))
        linarith
    set T := ∑ i ∈ Finset.range j,
      (κ * a i + ρ' * max |psum v c i| (m i) + σ * |psum v c (i + 1)| + α)
    set t := κ * a j + ρ' * max |psum v c j| (m j) + σ * |psum v c (j + 1)| + α
    have ht : 0 ≤ t := hterm j
    have hq : 1 ≤ 1 + ρ' + σ := by linarith
    have hqj : 1 ≤ (1 + ρ' + σ) ^ (j + 1) := one_le_pow₀ hq
    rw [Finset.sum_range_succ]
    rw [show errL σ α κ ρ' v c a m (j + 1) = (1 + ρ' + σ) * errL σ α κ ρ' v c a m j + t by
      simp only [errL, t]; ring]
    have e1 : (1 + ρ' + σ) * errL σ α κ ρ' v c a m j ≤ (1 + ρ' + σ) * ((1 + ρ' + σ) ^ j * T) :=
      mul_le_mul_of_nonneg_left ih (by linarith)
    have e2 : (1 + ρ' + σ) * ((1 + ρ' + σ) ^ j * T) = (1 + ρ' + σ) ^ (j + 1) * T := by ring
    have e3 : t ≤ (1 + ρ' + σ) ^ (j + 1) * t := le_mul_of_one_le_left ht hqj
    nlinarith

end Dot8Chunk

end TVA
