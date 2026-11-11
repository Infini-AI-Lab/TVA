import TVA.Device
import TVA.Device.Tolerance
import TVA.Proof.Float

/-!
# `acc += tl.dot(a, b)` over `K` blocks

The GEMM loop

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    acc += tl.dot(a, b)
```

is one `tl.dot_add` per trip (`TVA.Triton.Dot`): Triton's combine pass may fold the addition into
the dot (`tl.dot(a, b, acc)`), or not (the dot from a zero accumulator, then a fusable `float32`
addition). This file bounds the accumulator after `T` trips by the exact sum, whichever form the
device runs at each trip.

It rests on a whole-dot accuracy fact of the device: `Device.DotAcc` (relative to `|acc| + Σ|a·b|`,
e.g. `Device.DotF32Ieee`) or `Device.DotLiftAcc` (relative to lifted magnitudes, `Device.DotF16`,
`Device.DotBF16`; `TVA.Proof.Dot.Lift`). The tolerance formulas are those of
`TVA.Device.Tolerance`, which claims name: `DotLoop.bound`, `DotLoop.stepRel`, `DotLoop.stepAbs`.

* One trip: `DotLoop.dotAdd_step` (from `DotAcc`) and `DotLoop.dotAdd_step_of` (for any
  magnitude `A` of the products; `DotLift.dotAdd_step` is its instance for `DotLiftAcc`). The
  result is finite and within `stepRel rel K·(|acc| + A) + stepAbs abs K` of `acc + Σ a·b`.
  Both rest on `DotLoop.add_step`, which takes either form's bound on one element (also used for
  `tl.dot_scaled`, `TVA.Proof.Dot.Scaled`).
* `T` trips: `DotLoop.accum_step` (one trip of the recursion) and `DotLoop.accum` (by induction):
  from `0`, the accumulator stays within `bound ρ α t (Σ A)` of the exact `Σ S`.
* Closed form: `DotLoop.bound_le` (`bound ≤ 2t·(ρ·B + α)` when `t·ρ ≤ 1/2`), from `DotLoop.pow_le`
  (`(1 + x)^n ≤ 1 + 2nx`); `DotLoop.bound_nonneg`.
* `DotLoop.sum_chunks`: the trips' sums over consecutive blocks of `B` are one sum over `T·B`.
-/

namespace TVA

namespace DotLoop

/-! ## The scalar recursion -/

/-- `(1 + x)^n · (1 − n·x) ≤ 1` for `0 ≤ x` and `n·x ≤ 1`. Use it through `pow_le`. -/
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

/-- `(1 + x)^n ≤ 1 + 2·n·x` for `0 ≤ x` and `n·x ≤ 1/2`. Use it to turn a compounded relative
error into a linear one. -/
theorem pow_le {x : ℚ} (hx : 0 ≤ x) (n : ℕ) (h : n * x ≤ 1 / 2) : (1 + x) ^ n ≤ 1 + 2 * n * x := by
  have h1 := pow_mul_le hx n (by linarith)
  have hpos : 0 < 1 - n * x := by linarith
  have h2 : 1 ≤ (1 + 2 * n * x) * (1 - n * x) := by nlinarith [mul_nonneg (Nat.cast_nonneg n) hx]
  exact le_of_mul_le_mul_right (h1.trans h2) hpos

/-- `DotLoop.bound` is nonnegative for nonnegative constants and magnitude. -/
theorem bound_nonneg {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) (t : ℕ) {B : ℚ} (hB : 0 ≤ B) :
    0 ≤ bound ρ α t B := by
  unfold bound
  have : 1 ≤ (1 + ρ) ^ t := one_le_pow₀ (by linarith)
  have : 0 ≤ ((1 + ρ) ^ t - 1) * B := mul_nonneg (by linarith) hB
  positivity

/-- **The closed form.** For `t·ρ ≤ 1/2`, `bound ρ α t B ≤ 2t·(ρ·B + α)`. Use it to state a loop's
tolerance linearly in the number of trips. -/
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

/-- **One trip of the recursion.** If the accumulator `a` is within `bound ρ α t Bt` of the exact
partial sum `St` (`|St| ≤ Bt`), and the trip adds `S` (`|S| ≤ A`) up to `ρ·(|a| + A) + α`, the new
value `a'` is within `bound ρ α (t + 1) (Bt + A)` of `St + S`. Use it in a loop invariant. -/
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

/-- **The whole loop.** If, while the error so far is within the bound, trip `t` adds `S t` up to
`ρ·(|acc t| + A t) + α` (`|S t| ≤ A t`), then after `t ≤ T` trips from `acc 0 = 0` the accumulator
is within `bound ρ α t (Σ A)` of `Σ S`. Use it with `dotAdd_step` (or `DotLift.dotAdd_step`) as the
trip's hypothesis. -/
theorem accum {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) (acc S A : ℕ → ℚ) (T : ℕ) (h0 : acc 0 = 0)
    (hSA : ∀ t, |S t| ≤ A t)
    (hstep : ∀ t, t < T →
      |acc t - ∑ t' ∈ Finset.range t, S t'| ≤ bound ρ α t (∑ t' ∈ Finset.range t, A t') →
      |acc (t + 1) - (acc t + S t)| ≤ ρ * (|acc t| + A t) + α) :
    ∀ t, t ≤ T →
      |acc t - ∑ t' ∈ Finset.range t, S t'| ≤ bound ρ α t (∑ t' ∈ Finset.range t, A t') := by
  intro t
  induction t with
  | zero => intro _; simp [h0, bound]
  | succ t ih =>
    intro ht
    have e := ih (by omega)
    rw [Finset.sum_range_succ, Finset.sum_range_succ]
    exact accum_step hρ hα
      ((Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => hSA i))
      (hSA t) e (hstep t (by omega) e)

/-- Sums over `T` consecutive blocks of length `B` are one sum over `T·B`. Use it to turn the
loop's per-trip sums into the claim's `Σ_{k < K}`. -/
theorem sum_chunks (f : ℕ → ℚ) (B : ℕ) : ∀ T : ℕ,
    ∑ t ∈ Finset.range T, ∑ k ∈ Finset.range B, f (t * B + k) = ∑ j ∈ Finset.range (T * B), f j
  | 0 => by simp
  | T + 1 => by
    rw [Finset.sum_range_succ, sum_chunks f B T, show (T + 1) * B = T * B + B by ring,
      Finset.sum_range_add]

/-! ## One trip of `acc += tl.dot(a, b)` -/

/-- `float32` has infinities and NaN. -/
private theorem binary32_specials : Format.binary32.specials ≠ .finite := by decide

/-- **One trip, from either form of the addition**, for one `float32` accumulator element `c`:
if the result `r` is the folded form, within `ρ·(|c| + A) + α` of `c + S`, or the separate form,
an addition (`AddRel`) of `c` and a product `d` within `ρ·A + α` of `S` (`|S| ≤ A`), then `r` is
finite and within `(ρ + 3u(1 + ρ))·(|c| + A) + α(1 + 3u) + 4η` of `c + S`, given room in
`float32`. The common core of `dotAdd_step_of` and `Device.DotScaledAcc.addStep`. -/
theorem add_step {fusion : Bool} {ρ α A S : ℚ} (hrel : 0 ≤ ρ) (habs : 0 ≤ α) (hSA : |S| ≤ A)
    {c r : F32} (hc : c.isFinite = true)
    (hr : (r.isFinite = true ∧ |r.toRat - (c.toRat + S)| ≤ ρ * (|c.toRat| + A) + α) ∨
      ∃ d : F32, d.isFinite = true ∧ |d.toRat - S| ≤ ρ * A + α ∧ AddRel fusion d c r)
    (hroom : 2 * ((1 + ρ) * (|c.toRat| + A) + α + 2 * Format.binary32.eta) ≤
      Format.binary32.maxVal) :
    r.isFinite = true ∧
      |r.toRat - (c.toRat + S)| ≤
        (ρ + 3 * Format.binary32.u * (1 + ρ)) * (|c.toRat| + A) +
          (α * (1 + 3 * Format.binary32.u) + 4 * Format.binary32.eta) := by
  have hu0 := Format.u_nonneg (f := Format.binary32)
  have he0 := Format.eta_nonneg (f := Format.binary32)
  have hA0 : 0 ≤ A := (abs_nonneg _).trans hSA
  set cv := c.toRat with hcv
  have hcv0 := abs_nonneg cv
  have hmA : 0 ≤ ρ * (|cv| + A) := mul_nonneg hrel (by linarith)
  rcases hr with ⟨hfin, herr⟩ | ⟨d, hdfin, hdS, hadd⟩
  · refine ⟨hfin, herr.trans ?_⟩
    nlinarith [mul_nonneg (mul_nonneg hu0 (by linarith : (0 : ℚ) ≤ 1 + ρ))
        (by linarith : (0 : ℚ) ≤ |cv| + A),
      mul_nonneg hu0 habs]
  · have hdabs : |d.toRat| ≤ (1 + ρ) * A + α := by
      have := abs_sub_abs_le_abs_sub d.toRat S
      linarith
    obtain ⟨hrfin, hrerr⟩ := FloatV.AddRel.err Format.binary32_regular binary32_specials
      (by rw [Format.binary32_u]; norm_num) hadd hdfin hc (by nlinarith)
    refine ⟨hrfin, ?_⟩
    have e : r.toRat - (cv + S) = (r.toRat - (d.toRat + cv)) + (d.toRat - S) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    have hu3 : 3 * Format.binary32.u * (|d.toRat| + |cv|) ≤
        3 * Format.binary32.u * ((1 + ρ) * A + α + |cv|) :=
      mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    nlinarith [mul_nonneg (mul_nonneg hu0 hrel) hcv0, mul_nonneg hrel hcv0]

/-- **One trip, for any magnitude `A` of the products**, at output element `(i, n)`: if every
`tl.dot` of `a` and `b` from an accumulator of value `vc` (with room on `|vc| + A`) is within
`ρ·(|vc| + A) + α` of `vc + S`, `|S| ≤ A`, then whichever form of `tl.dot_add` the device runs
(`hr`, as `safe_dot_add` gives it; `z` the zero accumulator) the result is finite and within
`(ρ + 3u(1 + ρ))·(|c| + A) + α(1 + 3u) + 4η` of `c + S`, given room in `float32`. Use
`dotAdd_step` or `DotLift.dotAdd_step` instead unless the dot fact has another shape. -/
theorem dotAdd_step_of {P : Platform} {fusion : Bool} {da db : DType} {pr : InputPrecision}
    {mi : Nat} {M N K : Nat} {ρ α A S : ℚ} (hrel : 0 ≤ ρ) (habs : 0 ≤ α)
    {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val} {c z r : Tensor [M, N] F32}
    (i : Fin M) (n : Fin N) (hSA : |S| ≤ A)
    (hacc : ∀ (c' r' : Tensor [M, N] F32) (vc : ℚ), P.dot ⟨da, db, .f32, pr, mi⟩ a b c' r' →
      DType.val? .f32 (c'.get2 i n) = some vc →
      |vc| + A + (ρ * (|vc| + A) + α) ≤ DType.bound .f32 →
      ∃ vr, DType.val? .f32 (r'.get2 i n) = some vr ∧ |vr - (vc + S)| ≤ ρ * (|vc| + A) + α)
    (hr : P.dot ⟨da, db, .f32, pr, mi⟩ a b c r ∨
      ∃ d, P.dot ⟨da, db, .f32, pr, mi⟩ a b z d ∧
        ∀ ix, AccAdd fusion .f32 (d.get ix) (c.get ix) (r.get ix))
    (hz : DType.val? .f32 (z.get2 i n) = some 0)
    (hc : (c.get2 i n).isFinite = true)
    (hroom : 2 * ((1 + ρ) * (|(c.get2 i n).toRat| + A) + α + 2 * Format.binary32.eta) ≤
      Format.binary32.maxVal) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - ((c.get2 i n).toRat + S)| ≤
        (ρ + 3 * Format.binary32.u * (1 + ρ)) * (|(c.get2 i n).toRat| + A) +
          (α * (1 + 3 * Format.binary32.u) + 4 * Format.binary32.eta) := by
  have hA0 : 0 ≤ A := (abs_nonneg _).trans hSA
  have hc0 := abs_nonneg (c.get2 i n).toRat
  have hmA : 0 ≤ ρ * (|(c.get2 i n).toRat| + A) := mul_nonneg hrel (by linarith)
  have he0 := Format.eta_nonneg (f := Format.binary32)
  refine add_step (fusion := fusion) hrel habs hSA hc ?_ hroom
  rcases hr with h1 | ⟨d, hd, hadd⟩
  · obtain ⟨vr, hvr, herr⟩ := hacc c r _ h1 (DType.val?_f32.mpr ⟨hc, rfl⟩)
      (by simp only [DType.bound]; nlinarith)
    obtain ⟨hfin, hval⟩ := DType.val?_f32.mp hvr
    exact Or.inl ⟨hfin, hval ▸ herr⟩
  · obtain ⟨vd, hvd, herrd⟩ := hacc z d 0 hd hz
      (by simp only [DType.bound, abs_zero, zero_add]; nlinarith)
    obtain ⟨hdfin, hdval⟩ := DType.val?_f32.mp hvd
    simp only [abs_zero, zero_add] at herrd
    exact Or.inr ⟨d.get2 i n, hdfin, hdval ▸ herrd, hadd (i, n, ())⟩

/-- **One trip of `acc += tl.dot(a, b)` from a `DotAcc` fact** (e.g. `Device.DotF32Ieee`), at
output element `(i, n)`: whether the device ran `tl.dot(a, b, c)` or `c + tl.dot(a, b)` (`hr`, as
`safe_dot_add` gives it), the result is finite and within
`stepRel rel K·(|c| + Σ|a·b|) + stepAbs abs K` of `c + Σ a·b`, given room in `float32`. Feed it
to `accum`. For the 16-bit facts (`DotLiftAcc`) use `DotLift.dotAdd_step`. -/
theorem dotAdd_step {P : Platform} {fusion : Bool} {da db : DType} {pr : InputPrecision} {mi : Nat}
    {dom : ℚ → Prop} {rel abs : ℕ → ℚ}
    (hacc : Device.DotAcc P ⟨da, db, .f32, pr, mi⟩ dom rel abs)
    {M N K : Nat} (hrel : 0 ≤ rel K) (habs : 0 ≤ abs K)
    {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val} {c z r : Tensor [M, N] F32}
    (hr : P.dot ⟨da, db, .f32, pr, mi⟩ a b c r ∨
      ∃ d, P.dot ⟨da, db, .f32, pr, mi⟩ a b z d ∧
        ∀ ix, AccAdd fusion .f32 (d.get ix) (c.get ix) (r.get ix))
    (i : Fin M) (n : Fin N) (hz : DType.val? .f32 (z.get2 i n) = some 0) {va vb : Fin K → ℚ}
    (hva : ∀ k, da.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, db.val? (b.get2 k n) = some (vb k))
    (hdom : ∀ k, dom (va k) ∧ dom (vb k)) (hc : (c.get2 i n).isFinite = true)
    (hroom : 2 * ((1 + rel K) * (|(c.get2 i n).toRat| + ∑ k, |va k * vb k|) + abs K +
      2 * Format.binary32.eta) ≤ Format.binary32.maxVal) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - ((c.get2 i n).toRat + ∑ k, va k * vb k)| ≤
        stepRel rel K * (|(c.get2 i n).toRat| + ∑ k, |va k * vb k|) + stepAbs abs K := by
  unfold stepRel stepAbs
  exact dotAdd_step_of (fusion := fusion) hrel habs i n (Finset.abs_sum_le_sum_abs _ _)
    (fun c' r' vc hd' hvc' hroom' => hacc a b c' r' hd' i n va vb vc hva hvb hdom hvc' hroom')
    hr hz hc hroom

end DotLoop

end TVA
