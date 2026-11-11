import TVA.Proof.Dot.Loop

/-!
# `tl.dot` relative to lifted magnitudes (16-bit operands)

The 16-bit `tl.dot` facts `Device.DotF16` and `Device.DotBF16` have the shape `Device.DotLiftAcc`:
per output element, the result is within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with
`L = |acc| + Σ liftMag a_k · liftMag b_k`. A tensor core aligns a subnormal operand as if it were
the smallest normal value (`Device.liftFloor`: `2^-14` for `float16`, `2^-126` for `bfloat16`),
so the bound charges it that magnitude. This file puts the lifted magnitudes to use in the
accumulation pattern `acc += tl.dot(a, b)` over `K` blocks (`TVA.Proof.Dot.Loop`).

* `DotLift.dotAdd_step`: one trip of `acc += tl.dot(a, b)` (either form of `tl.dot_add`) from a
  `DotLiftAcc` fact, as `DotLoop.dotAdd_step` with `Σ liftMag a·liftMag b` in place of `Σ|a·b|`.
* `DotLift.liftMag_mul_le`, `DotLift.sum_liftMag_mul_le`: lifted against plain magnitudes,
  `liftMag da x·liftMag db y ≤ |x·y| + fb·|x| + fa·|y| + fa·fb` (`fa`, `fb` the floors), summed.
* `Device.DotLiftAcc.dotAcc`: on zero and normal values (where `liftMag` is `|q|`) a `DotLiftAcc`
  fact is a `DotAcc` fact with the same constants.
* For every dtype: `Device.liftMag_zero`, `Device.liftMag_of_le` (a normal value counts as `|q|`),
  `Device.liftMag_le` (`liftMag d q ≤ |q| + liftFloor d`), `Device.liftFloor_nonneg`,
  `Device.liftFloor_f16`, `Device.liftFloor_bf16` (with `Device.liftMag_nonneg`,
  `Device.abs_le_liftMag`, `Device.abs_mul_le_liftMag` of `TVA.Device.Weaken`).
-/

namespace TVA

namespace Device

/-- The floor of `float16` operands: `2^-14`, its smallest normal magnitude. -/
theorem liftFloor_f16 : liftFloor .f16 = 2 ^ (-14 : ℤ) := rfl

/-- The floor of `bfloat16` operands: `2^-126`, its smallest normal magnitude. -/
theorem liftFloor_bf16 : liftFloor .bf16 = 2 ^ (-126 : ℤ) := rfl

/-- A floor is nonnegative. -/
theorem liftFloor_nonneg (d : DType) : 0 ≤ liftFloor d := by
  cases d <;> simp only [liftFloor] <;> positivity

/-- The lifted magnitude of `0` is `0`. -/
@[simp] theorem liftMag_zero (d : DType) : liftMag d 0 = 0 := by simp [liftMag]

/-- A value at least the floor (a normal value of the dtype) counts as `|q|`. Use it to drop the
lift on normal operands. -/
theorem liftMag_of_le {d : DType} {q : ℚ} (h : liftFloor d ≤ |q|) : liftMag d q = |q| := by
  unfold liftMag
  split_ifs with h0
  · rw [h0, abs_zero]
  · exact max_eq_left h

/-- A lifted magnitude exceeds the magnitude by at most the floor: `liftMag d q ≤ |q| + liftFloor d`.
Use it to bound a lifted tolerance by a plain one. -/
theorem liftMag_le (d : DType) (q : ℚ) : liftMag d q ≤ |q| + liftFloor d := by
  have h0 := liftFloor_nonneg d
  unfold liftMag
  split_ifs
  · exact add_nonneg (abs_nonneg q) h0
  · exact max_le (by linarith) (by linarith [abs_nonneg q])

/-- **On zero and normal values a `DotLiftAcc` fact is a `DotAcc` fact** with the same constants:
where `liftMag` is `|q|` for both operand dtypes (`liftMag_of_le`, `liftMag_zero`), `L` is
`|acc| + Σ|a·b|`. Use it when a claim restricts the operands to normal values. -/
theorem DotLiftAcc.dotAcc {P : Platform} {cfg : DotCfg} {rel abs : ℕ → ℚ} {dom : ℚ → Prop}
    (h : DotLiftAcc P cfg rel abs) (hda : ∀ q, dom q → liftMag cfg.a q = |q|)
    (hdb : ∀ q, dom q → liftMag cfg.b q = |q|) : DotAcc P cfg dom rel abs := by
  intro M N K a b acc r hr i n va vb vc hva hvb hd hvc hroom
  have hL : ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k) = ∑ k, |va k * vb k| :=
    Finset.sum_congr rfl fun k _ => by rw [hda _ (hd k).1, hdb _ (hd k).2, abs_mul]
  obtain ⟨vr, hvr, herr⟩ := h a b acc r hr i n va vb vc hva hvb hvc (by rw [hL]; exact hroom)
  exact ⟨vr, hvr, by rw [hL] at herr; exact herr⟩

end Device

namespace DotLift

open Device

/-- A lifted product against the plain one:
`liftMag da x·liftMag db y ≤ |x·y| + fb·|x| + fa·|y| + fa·fb`, `fa = liftFloor da`,
`fb = liftFloor db`. Use it to compare a `DotLiftAcc` bound with a bound on `|a·b|`. -/
theorem liftMag_mul_le (da db : DType) (x y : ℚ) :
    liftMag da x * liftMag db y ≤
      |x * y| + liftFloor db * |x| + liftFloor da * |y| + liftFloor da * liftFloor db := by
  have hx := liftMag_le da x
  have hy := liftMag_le db y
  have h := mul_le_mul hx hy (liftMag_nonneg db y)
    (add_nonneg (abs_nonneg x) (liftFloor_nonneg da))
  calc liftMag da x * liftMag db y ≤ (|x| + liftFloor da) * (|y| + liftFloor db) := h
    _ = |x * y| + liftFloor db * |x| + liftFloor da * |y| + liftFloor da * liftFloor db := by
      rw [abs_mul]; ring

/-- `liftMag_mul_le` summed over `k`:
`Σ liftMag·liftMag ≤ Σ|va·vb| + fb·Σ|va| + fa·Σ|vb| + K·fa·fb`. Use it to state a 16-bit dot's
tolerance with plain magnitudes. -/
theorem sum_liftMag_mul_le (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ k, liftMag da (va k) * liftMag db (vb k) ≤
      ∑ k, |va k * vb k| + liftFloor db * ∑ k, |va k| + liftFloor da * ∑ k, |vb k| +
        K * (liftFloor da * liftFloor db) := by
  calc ∑ k, liftMag da (va k) * liftMag db (vb k)
      ≤ ∑ k, (|va k * vb k| + liftFloor db * |va k| + liftFloor da * |vb k| +
          liftFloor da * liftFloor db) := Finset.sum_le_sum fun k _ => liftMag_mul_le da db _ _
    _ = _ := by
      simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul]
      ring

/-- **One trip of `acc += tl.dot(a, b)` from a 16-bit fact** (`Device.DotF16`, `Device.DotBF16`:
`DotLiftAcc`), at output element `(i, n)`: whether the device ran `tl.dot(a, b, c)` or
`c + tl.dot(a, b)` (`hr`, as `safe_dot_add` gives it), the result is finite and within
`stepRel rel K·(|c| + Σ liftMag a·liftMag b) + stepAbs abs K` of `c + Σ a·b`, given room in
`float32`. Feed it to `DotLoop.accum` with `A t = Σ liftMag a·liftMag b`. -/
theorem dotAdd_step {P : Platform} {fusion : Bool} {da db : DType} {pr : InputPrecision} {mi : Nat}
    {rel abs : ℕ → ℚ} (hacc : DotLiftAcc P ⟨da, db, .f32, pr, mi⟩ rel abs)
    {M N K : Nat} (hrel : 0 ≤ rel K) (habs : 0 ≤ abs K)
    {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val} {c z r : Tensor [M, N] F32}
    (hr : P.dot ⟨da, db, .f32, pr, mi⟩ a b c r ∨
      ∃ d, P.dot ⟨da, db, .f32, pr, mi⟩ a b z d ∧
        ∀ ix, AccAdd fusion .f32 (d.get ix) (c.get ix) (r.get ix))
    (i : Fin M) (n : Fin N) (hz : DType.val? .f32 (z.get2 i n) = some 0) {va vb : Fin K → ℚ}
    (hva : ∀ k, da.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, db.val? (b.get2 k n) = some (vb k))
    (hc : (c.get2 i n).isFinite = true)
    (hroom : 2 * ((1 + rel K) * (|(c.get2 i n).toRat| + ∑ k, liftMag da (va k) * liftMag db (vb k)) +
      abs K + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - ((c.get2 i n).toRat + ∑ k, va k * vb k)| ≤
        DotLoop.stepRel rel K * (|(c.get2 i n).toRat| +
          ∑ k, liftMag da (va k) * liftMag db (vb k)) + DotLoop.stepAbs abs K := by
  unfold DotLoop.stepRel DotLoop.stepAbs
  have hSA : |∑ k, va k * vb k| ≤ ∑ k, liftMag da (va k) * liftMag db (vb k) :=
    (Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum fun k _ => abs_mul_le_liftMag _ _ _ _)
  exact DotLoop.dotAdd_step_of (fusion := fusion) hrel habs i n hSA
    (fun c' r' vc hd' hvc' hroom' => hacc a b c' r' hd' i n va vb vc hva hvb hvc' hroom')
    hr hz hc hroom

end DotLift

end TVA
