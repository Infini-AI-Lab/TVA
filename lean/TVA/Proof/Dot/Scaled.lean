import TVA.Proof.Dot.Loop

/-!
# `tl.dot_scaled` accumulated over `K` blocks

A block-scaled GEMM loop accumulates microscaled products into a `float32` accumulator:

```python
for k in range(0, tl.cdiv(K, BLOCK_K)):
    acc = tl.dot_scaled(a, scale_a, "e4m3", b, scale_b, "e4m3", acc)   # or acc += tl.dot_scaled(…)
```

Each operand element stands for its exact scaled value (`scaledValue`: the element's value times
its `e8m0` scale `2^(b − 127)`). The device facts `Device.DotScaledE4m3`,
`Device.DotScaledE4m3E2m1` (shape `Device.DotScaledAcc`) bound one `tl.dot_scaled` with the
accumulator folded in, relative to `|acc| + Σ|a·b|`; that is already the step hypothesis of
`DotLoop.accum`. This file adds what the loop needs beyond it.

* `Device.DotScaledAcc.addStep`: one trip of `acc += tl.dot_scaled(…)` (`tl.dot_scaled_add`,
  either form, as `safe_dot_scaled_add` gives it), within
  `stepRel rel K·(|c| + Σ|a·b|) + stepAbs abs K` of `c + Σ a·b` (`DotLoop.add_step`).
* The exact values: `e8m0Value_of_ne` (a scale byte other than `0xFF` is `2^(b − 127)`),
  `scaledValue_none`, `scaledValue_some` (an element with its scale), `e8m0Dom_ne`,
  `scaledDomBF16_zero`.
-/

namespace TVA

open Device

/-- A scale byte other than `0xFF` has the value `2^(b − 127)`. -/
theorem e8m0Value_of_ne {b : Nat} (h : b ≠ 255) : e8m0Value b = some ((2 : ℚ) ^ ((b : ℤ) - 127)) := by
  simp [e8m0Value, h]

/-- A scale byte in the device's domain is not the NaN `0xFF`. -/
theorem Device.e8m0Dom_ne {b : Nat} (h : e8m0Dom b) : b ≠ 255 := by
  unfold e8m0Dom at h; omega

/-- `0` is in the domain of the `tl.dot_scaled` facts. -/
theorem Device.scaledDomBF16_zero : scaledDomBF16 0 := Or.inl rfl

/-- An element without a scale stands for its own value. -/
theorem scaledValue_none {fmt : MXFmt} {sf : ScaleFmt} {bits : Nat} {v : ℚ}
    (h : fmt.value bits = some v) : scaledValue fmt sf bits none = some v := by
  simp [scaledValue, h]

/-- An element with an `e8m0` scale byte `b ≠ 0xFF` stands for its value times `2^(b − 127)`.
Use it to compute the `va`, `vb` of a `DotScaledAcc` fact. -/
theorem scaledValue_some {fmt : MXFmt} {bits b : Nat} {v : ℚ} (h : fmt.value bits = some v)
    (hb : b ≠ 255) :
    scaledValue fmt .e8m0 bits (some b) = some (v * (2 : ℚ) ^ ((b : ℤ) - 127)) := by
  simp [scaledValue, h, ScaleFmt.value, e8m0Value_of_ne hb]

/-- **One trip of `acc += tl.dot_scaled(…)` from a `DotScaledAcc` fact**, at output element
`(i, n)`: whether the device folded `c` into the accumulator or added it to the product from the
zero accumulator `z` (`hr`, as `safe_dot_scaled_add` gives it), the result is finite and within
`stepRel rel K·(|c| + Σ|a·b|) + stepAbs abs K` of `c + Σ a·b`, given room in `float32`. Feed it
to `DotLoop.accum`. -/
theorem Device.DotScaledAcc.addStep {P : Platform} {fusion : Bool} {cfg : ScaledCfg}
    {dom : ℚ → Prop} {sdom : Nat → Prop} {rel abs : ℕ → ℚ}
    (hacc : DotScaledAcc P cfg dom sdom rel abs) {M N K : Nat} (hrel : 0 ≤ rel K)
    (habs : 0 ≤ abs K) {a : Tensor [M, K] Nat} {sa : Option (Tensor [M, K] Nat)}
    {b : Tensor [K, N] Nat} {sb : Option (Tensor [K, N] Nat)} {c z r : Tensor [M, N] F32}
    (hr : P.dotScaled cfg a sa b sb c r ∨
      ∃ d, P.dotScaled cfg a sa b sb z d ∧ ∀ ix, AddRel fusion (d.get ix) (c.get ix) (r.get ix))
    (i : Fin M) (n : Fin N) (hz : DType.val? .f32 (z.get2 i n) = some 0) {va vb : Fin K → ℚ}
    (hva : ∀ k, scaledValue cfg.lhs cfg.lhsScale (a.get2 i k) (sa.map fun t => t.get2 i k) =
      some (va k))
    (hvb : ∀ k, scaledValue cfg.rhs cfg.rhsScale (b.get2 k n) (sb.map fun t => t.get2 k n) =
      some (vb k))
    (hdom : ∀ k, dom (va k) ∧ dom (vb k))
    (hsa : ∀ t, sa = some t → ∀ k, sdom (t.get2 i k))
    (hsb : ∀ t, sb = some t → ∀ k, sdom (t.get2 k n))
    (hc : (c.get2 i n).isFinite = true)
    (hroom : 2 * ((1 + rel K) * (|(c.get2 i n).toRat| + ∑ k, |va k * vb k|) + abs K +
      2 * Format.binary32.eta) ≤ Format.binary32.maxVal) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - ((c.get2 i n).toRat + ∑ k, va k * vb k)| ≤
        DotLoop.stepRel rel K * (|(c.get2 i n).toRat| + ∑ k, |va k * vb k|) +
          DotLoop.stepAbs abs K := by
  unfold DotLoop.stepRel DotLoop.stepAbs
  have hA0 : 0 ≤ ∑ k, |va k * vb k| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hc0 := abs_nonneg (c.get2 i n).toRat
  have hmA : 0 ≤ rel K * (|(c.get2 i n).toRat| + ∑ k, |va k * vb k|) :=
    mul_nonneg hrel (by linarith)
  have he0 := Format.eta_nonneg (f := Format.binary32)
  refine DotLoop.add_step (fusion := fusion) hrel habs (Finset.abs_sum_le_sum_abs _ _) hc ?_ hroom
  rcases hr with h1 | ⟨d, hd, hadd⟩
  · exact Or.inl (hacc a sa b sb c r h1 i n va vb hva hvb hdom hsa hsb hc (by nlinarith))
  · obtain ⟨hzfin, hzval⟩ := DType.val?_f32.mp hz
    obtain ⟨hdfin, hderr⟩ := hacc a sa b sb z d hd i n va vb hva hvb hdom hsa hsb hzfin
      (by rw [hzval]; simp only [abs_zero, zero_add]; nlinarith)
    rw [hzval] at hderr
    simp only [abs_zero, zero_add] at hderr
    exact Or.inr ⟨d.get2 i n, hdfin, hderr, hadd (i, n, ())⟩

end TVA
