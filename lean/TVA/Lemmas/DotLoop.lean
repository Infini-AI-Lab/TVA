import TVA.Lemmas.Fused
import TVA.Rules.Num

/-!
# Accumulating `tl.dot` over `K`

The GEMM loop `for kk in range(...): acc += tl.dot(a, b)` is `tl.dot_add` at each step. The
combine pass may turn it into `tl.dot(a, b, acc)`, or not (`tl.dot_add`). Given
`Rules.Num.DotAcc` for the configuration, each output element changes by at most
`ρ'·(|acc| + Σ|a·b|) + α'` from `acc + Σ a·b`, whichever form the device runs:

* `DotLoop.dotAdd_step`: one step, for a `float32` accumulator, with `ρ' = stepRel rel K` and
  `α' = stepAbs abs K`. The extra terms cover the separate addition (`FloatV.AddRel.err`).
* `DotLoop.accum`, `DotLoop.accum_step`: `T` such steps from `0` are within
  `bound ρ' α' T (Σ|a·b|)` of the exact sum.
* `DotLoop.bound_le`: when `T·ρ' ≤ 1/2`, that bound is at most `2T·(ρ'·Σ|a·b| + α')`.
* `DotLoop.pow_le`: `(1 + x)^n ≤ 1 + 2nx` for `nx ≤ 1/2`.
-/

namespace TVA

namespace DotLoop

open TVA.Math.Float

/-! ## The scalar recursion -/

/-- `(1 + x)^n · (1 − n·x) ≤ 1` for `0 ≤ x` and `n·x ≤ 1`. -/
theorem pow_mul_le {x : ℚ} (hx : 0 ≤ x) : ∀ n : ℕ, n * x ≤ 1 → (1 + x) ^ n * (1 - n * x) ≤ 1
  | 0, _ => by simp
  | n + 1, h => by
    have hn : (n : ℚ) * x ≤ 1 := by push_cast at h; nlinarith
    have ih := pow_mul_le hx n hn
    have hp : 0 ≤ (1 + x) ^ n := by positivity
    have e : (1 + x) * (1 - ((n + 1 : ℕ) : ℚ) * x) ≤ 1 - n * x := by
      have h2 : (1 + x) * (1 - ((n + 1 : ℕ) : ℚ) * x) = 1 - n * x - ((n : ℚ) + 1) * (x * x) := by
        push_cast; ring
      rw [h2]
      have : 0 ≤ ((n : ℚ) + 1) * (x * x) := mul_nonneg (by positivity) (mul_nonneg hx hx)
      linarith
    calc (1 + x) ^ (n + 1) * (1 - ((n + 1 : ℕ) : ℚ) * x)
        = (1 + x) ^ n * ((1 + x) * (1 - ((n + 1 : ℕ) : ℚ) * x)) := by ring
      _ ≤ (1 + x) ^ n * (1 - n * x) := mul_le_mul_of_nonneg_left e hp
      _ ≤ 1 := ih

/-- `(1 + x)^n ≤ 1 + 2·n·x` for `0 ≤ x` and `n·x ≤ 1/2`. -/
theorem pow_le {x : ℚ} (hx : 0 ≤ x) (n : ℕ) (h : n * x ≤ 1 / 2) : (1 + x) ^ n ≤ 1 + 2 * n * x := by
  have h1 := pow_mul_le hx n (by linarith)
  have hpos : 0 < 1 - n * x := by linarith
  have h2 : 1 ≤ (1 + 2 * n * x) * (1 - n * x) := by nlinarith [mul_nonneg (Nat.cast_nonneg n) hx]
  exact le_of_mul_le_mul_right (h1.trans h2) hpos

/-- The error after `t` steps, over steps of absolute sum `B`. -/
def bound (ρ α : ℚ) (t : ℕ) (B : ℚ) : ℚ := ((1 + ρ) ^ t - 1) * B + t * (1 + ρ) ^ t * α

theorem bound_nonneg {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) (t : ℕ) {B : ℚ} (hB : 0 ≤ B) :
    0 ≤ bound ρ α t B := by
  unfold bound
  have : 1 ≤ (1 + ρ) ^ t := one_le_pow₀ (by linarith)
  have : 0 ≤ ((1 + ρ) ^ t - 1) * B := mul_nonneg (by linarith) hB
  positivity

/-- **The closed form.** For `t·ρ ≤ 1/2`, the bound is at most `2t·(ρ·B + α)`. -/
theorem bound_le {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {t : ℕ} (ht : t * ρ ≤ 1 / 2) {B : ℚ}
    (hB : 0 ≤ B) : bound ρ α t B ≤ 2 * t * (ρ * B + α) := by
  unfold bound
  have hp := pow_le hρ t ht
  have ht0 : (0 : ℚ) ≤ t := Nat.cast_nonneg t
  have e1 : ((1 + ρ) ^ t - 1) * B ≤ 2 * t * ρ * B := mul_le_mul_of_nonneg_right (by linarith) hB
  have e2 : (t : ℚ) * (1 + ρ) ^ t * α ≤ t * 2 * α := by
    have : (1 + ρ) ^ t ≤ 2 := by nlinarith
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left this ht0) hα
  nlinarith

/-- **One step of the accumulation.** If the accumulator `a` is within `bound ρ α t Bt` of the
exact partial sum `St` (with `|St| ≤ Bt`), and a step adds `S` (with `|S| ≤ A`) up to
`ρ·(|a| + A) + α`, then the new value is within `bound ρ α (t + 1) (Bt + A)` of `St + S`. -/
theorem accum_step {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {t : ℕ} {a a' St S Bt A : ℚ}
    (hSt : |St| ≤ Bt) (hSA : |S| ≤ A) (he : |a - St| ≤ bound ρ α t Bt)
    (hs : |a' - (a + S)| ≤ ρ * (|a| + A) + α) :
    |a' - (St + S)| ≤ bound ρ α (t + 1) (Bt + A) := by
  have hA0 : 0 ≤ A := (abs_nonneg _).trans hSA
  have hacc : |a| ≤ Bt + bound ρ α t Bt := by
    have := abs_sub_abs_le_abs_sub a St
    linarith
  have htri : |a' - (St + S)| ≤ |a' - (a + S)| + |a - St| := by
    have e2 : a' - (St + S) = (a' - (a + S)) + (a - St) := by ring
    rw [e2]; exact abs_add_le _ _
  set P := (1 + ρ) ^ t with hP
  have hP1 : 1 ≤ P := one_le_pow₀ (by linarith)
  unfold bound at he hacc ⊢
  rw [← hP] at he hacc
  rw [pow_succ, ← hP]
  push_cast
  have hBt : 0 ≤ Bt := (abs_nonneg _).trans hSt
  have hm1 : ρ * |a| ≤ ρ * (Bt + ((P - 1) * Bt + t * P * α)) := mul_le_mul_of_nonneg_left hacc hρ
  nlinarith [mul_nonneg (mul_nonneg hρ (by linarith : (0 : ℚ) ≤ P - 1)) hA0,
    mul_nonneg (by linarith : (0 : ℚ) ≤ P - 1) hA0, mul_nonneg hρ hA0, mul_nonneg hρ hα,
    mul_nonneg (mul_nonneg hρ (by linarith : (0 : ℚ) ≤ P)) hα, mul_nonneg (Nat.cast_nonneg t) hα,
    mul_nonneg (mul_nonneg (Nat.cast_nonneg t) (by linarith : (0 : ℚ) ≤ P)) hα]

/-- **Accumulation.** If, while the error so far is within the bound, each step adds `S t` up to
`ρ·(|acc t| + A t) + α` (with `|S t| ≤ A t`), then after `t ≤ T` steps from `0` the accumulator
is within `bound ρ α t (Σ A)` of `Σ S`. -/
theorem accum {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) (acc S A : ℕ → ℚ) (T : ℕ) (h0 : acc 0 = 0)
    (hSA : ∀ t, |S t| ≤ A t)
    (hstep : ∀ t, t < T →
      |acc t - ∑ t' ∈ Finset.range t, S t'| ≤ bound ρ α t (∑ t' ∈ Finset.range t, A t') →
      |acc (t + 1) - (acc t + S t)| ≤ ρ * (|acc t| + A t) + α) :
    ∀ t, t ≤ T → |acc t - ∑ t' ∈ Finset.range t, S t'| ≤ bound ρ α t (∑ t' ∈ Finset.range t, A t') := by
  intro t
  induction t with
  | zero => intro _; simp [h0, bound]
  | succ t ih =>
    intro ht
    have e := ih (by omega)
    rw [Finset.sum_range_succ, Finset.sum_range_succ]
    exact accum_step hρ hα ((Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => hSA i))
      (hSA t) e (hstep t (by omega) e)

/-- Two nested sums over consecutive chunks of length `B` are one sum. -/
theorem sum_chunks (f : ℕ → ℚ) (B : ℕ) : ∀ T : ℕ,
    ∑ t ∈ Finset.range T, ∑ k ∈ Finset.range B, f (t * B + k) = ∑ j ∈ Finset.range (T * B), f j
  | 0 => by simp
  | T + 1 => by
    rw [Finset.sum_range_succ, sum_chunks f B T, show (T + 1) * B = T * B + B by ring,
      Finset.sum_range_add]

/-! ## One step of `acc += tl.dot(a, b)` -/

/-- The per-step relative error: the dot's, plus one addition. -/
def stepRel (rel : ℕ → ℚ) (K : ℕ) : ℚ := rel K + 3 * Format.binary32.u * (1 + rel K)

/-- The per-step absolute error. -/
def stepAbs (abs : ℕ → ℚ) (K : ℕ) : ℚ := abs K * (1 + 3 * Format.binary32.u) + 4 * Format.binary32.eta

theorem binary32_u : Format.binary32.u = 2 ^ (-24 : ℤ) := by unfold Format.u; rfl

theorem binary32_ieee : Format.binary32.special = .ieee := rfl

theorem val?_f32 {x : F32} {v : ℚ} : DType.val? .f32 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  show (if x.isFinite = true then some x.toRat else none) = some v ↔ _
  split_ifs with h
  · simp [h]
  · simp [h]

theorem zero_val : DType.val? .f32 (FloatV.ofPat 0 : F32) = some 0 := by
  have hlt : (0 : ℕ) < 2 ^ Format.binary32.width := Nat.two_pow_pos _
  rw [val?_f32, FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨Format.isFinite_zero Format.binary32_regular,
    Format.toRat_of_isZero (by decide)⟩

/-- **One step**, for one output element `(i, n)`. Whether the device ran `tl.dot(a, b, c)` or
`c + tl.dot(a, b)` (the two forms of `tl.dot_add`), the result is finite and within
`stepRel·(|c| + Σ|a·b|) + stepAbs` of `c + Σ a·b`, given `DotAcc` and room in `float32`. -/
theorem dotAdd_step {P : Platform} {fusion : Bool} {da db : DType} {pr : InputPrecision} {mi : Nat}
    {dom : ℚ → Prop} {rel abs : ℕ → ℚ}
    (hacc : Rules.Num.DotAcc P ⟨da, db, .f32, pr, mi⟩ dom rel abs)
    {M N K : Nat} (hrel : 0 ≤ rel K) (habs : 0 ≤ abs K)
    {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val} {c r : Tensor [M, N] F32}
    (hr : P.dot ⟨da, db, .f32, pr, mi⟩ a b c r ∨
      ∃ d, P.dot ⟨da, db, .f32, pr, mi⟩ a b (Tensor.full (FloatV.ofPat 0)) d ∧
        ∀ ix, AccAdd fusion .f32 (d.get ix) (c.get ix) (r.get ix))
    (i : Fin M) (n : Fin N) {va vb : Fin K → ℚ}
    (hva : ∀ k, da.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, db.val? (b.get2 k n) = some (vb k))
    (hdom : ∀ k, dom (va k) ∧ dom (vb k)) (hc : (c.get2 i n).isFinite = true)
    (hroom : 2 * ((1 + rel K) * (|(c.get2 i n).toRat| + ∑ k, |va k * vb k|) + abs K +
      2 * Format.binary32.eta) ≤ Format.binary32.maxVal) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - ((c.get2 i n).toRat + ∑ k, va k * vb k)| ≤
        stepRel rel K * (|(c.get2 i n).toRat| + ∑ k, |va k * vb k|) + stepAbs abs K := by
  have hu0 := Format.u_nonneg (f := Format.binary32)
  have he0 := Format.eta_nonneg (f := Format.binary32)
  have hA0 : 0 ≤ ∑ k, |va k * vb k| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hSA : |∑ k, va k * vb k| ≤ ∑ k, |va k * vb k| := Finset.abs_sum_le_sum_abs _ _
  set cv := (c.get2 i n).toRat with hcv
  set A := ∑ k, |va k * vb k|
  have hcv0 := abs_nonneg cv
  have hmA : 0 ≤ rel K * (|cv| + A) := mul_nonneg hrel (by linarith)
  unfold stepRel stepAbs
  rcases hr with h1 | ⟨d, hd, hadd⟩
  · obtain ⟨vr, hvr, herr⟩ := hacc a b c r h1 i n va vb cv hva hvb hdom
      (val?_f32.mpr ⟨hc, rfl⟩) (by simp only [DType.bound]; nlinarith)
    obtain ⟨hfin, hval⟩ := val?_f32.mp hvr
    refine ⟨hfin, ?_⟩
    rw [hval]
    refine herr.trans ?_
    nlinarith [mul_nonneg (mul_nonneg hu0 (by linarith : (0 : ℚ) ≤ 1 + rel K)) (by linarith : (0 : ℚ) ≤ |cv| + A),
      mul_nonneg hu0 habs]
  · have hz : DType.val? .f32 ((Tensor.full (FloatV.ofPat 0) : Tensor [M, N] F32).get2 i n) = some 0 :=
      zero_val
    obtain ⟨vd, hvd, herrd⟩ := hacc a b _ d hd i n va vb 0 hva hvb hdom hz
      (by simp only [DType.bound, abs_zero, zero_add]; nlinarith)
    obtain ⟨hdfin, hdval⟩ := val?_f32.mp hvd
    simp only [abs_zero, zero_add] at herrd
    have hdS : |(d.get2 i n).toRat - ∑ k, va k * vb k| ≤ rel K * A + abs K := by rw [hdval]; exact herrd
    have hdabs : |(d.get2 i n).toRat| ≤ (1 + rel K) * A + abs K := by
      have := abs_sub_abs_le_abs_sub (d.get2 i n).toRat (∑ k, va k * vb k)
      linarith
    have hrel' : AddRel fusion (d.get2 i n) (c.get2 i n) (r.get2 i n) := hadd (i, n, ())
    obtain ⟨hrfin, hrerr⟩ := FloatV.AddRel.err Format.binary32_regular binary32_ieee
      (by rw [binary32_u]; norm_num) hrel' hdfin hc (by nlinarith)
    refine ⟨hrfin, ?_⟩
    have e : (r.get2 i n).toRat - (cv + ∑ k, va k * vb k) =
        ((r.get2 i n).toRat - ((d.get2 i n).toRat + cv)) +
          ((d.get2 i n).toRat - ∑ k, va k * vb k) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    have hu3 : 3 * Format.binary32.u * (|(d.get2 i n).toRat| + |cv|) ≤
        3 * Format.binary32.u * ((1 + rel K) * A + abs K + |cv|) :=
      mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    nlinarith [mul_nonneg (mul_nonneg hu0 hrel) hcv0, mul_nonneg hrel hcv0]

end DotLoop

end TVA
