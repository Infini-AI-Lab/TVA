import TVA.Lemmas.Dot8Chunk

/-!
# Using the 16-bit `tl.dot` facts (`DotF16`, `DotBF16`)

Page: `docs/rules/num.md` ("`tl.dot` (the chunk model)"). Since round 25 (library issue L126) the
facts `DotF16` and `DotBF16` have the shape `Rules.Num.DotLiftAcc`: per output element, the result is
within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with
`L = |acc| + Σ liftMag a_k · liftMag b_k`, a subnormal operand counted at the smallest normal
(`minNormal8`: `2^-14` for `float16`, `2^-126` for `bfloat16`). These lemmas put them to use.

* `DotLift.dotAdd_step`: one step of `acc += tl.dot(a, b)` (either form of `tl.dot_add`), as
  `DotLoop.dotAdd_step`, with `Σ liftMag a·liftMag b` in place of `Σ|a·b|` (from
  `DotLoop.dotAdd_step_of`).
* `DotLift.liftMag_mul_le`, `DotLift.sum_liftMag_mul_le`: the lifted magnitudes against the plain
  ones, `liftMag da x·liftMag db y ≤ |x·y| + mb·|x| + ma·|y| + ma·mb` (`ma = minNormal8 da`,
  `mb = minNormal8 db`), and summed over `k`. With the lemmas of `TVA.Lemmas.Dot8Chunk` that hold for
  every dtype: `Dot8Chunk.liftMag_nonneg`, `abs_le_liftMag`, `liftMag_zero`, `liftMag_of_le` (a
  normal value counts as `|q|`), `liftMag_le` (`liftMag d q ≤ |q| + minNormal8 d`),
  `abs_mul_le_liftMag`.
* `DotLift.minNormal8_f16`, `DotLift.minNormal8_bf16`: the two thresholds.
* `DotLift.DotLiftAcc.dotAcc`: on a domain of zero and normal values (where `liftMag` is `|q|`),
  a `DotLiftAcc` fact is `DotAcc` with the same constants.
* `Platform.ref.dotLiftAcc`: `DotLiftAcc` holds of `Platform.ref` with a float accumulator for
  `rel K ≥ u`, `abs K ≥ η` (`K > 0`), as `DotAcc` does; `Platform.sm90ref.dotF16`, `.dotBF16`
  (`TVA.Lemmas.Sm90`) are its instances for the facts.
-/
namespace TVA

open TVA.Math.Float Rules.Num

namespace DotLift

theorem minNormal8_f16 : minNormal8 .f16 = 2 ^ (-14 : ℤ) := rfl

theorem minNormal8_bf16 : minNormal8 .bf16 = 2 ^ (-126 : ℤ) := rfl

/-- A lifted product against the plain one: `liftMag da x·liftMag db y ≤ |x·y| + mb·|x| + ma·|y| +
ma·mb`, with `ma = minNormal8 da`, `mb = minNormal8 db` (`Dot8Chunk.liftMag_le` on each side). -/
theorem liftMag_mul_le (da db : DType) (x y : ℚ) :
    liftMag da x * liftMag db y ≤
      |x * y| + minNormal8 db * |x| + minNormal8 da * |y| + minNormal8 da * minNormal8 db := by
  have hx := Dot8Chunk.liftMag_le da x
  have hy := Dot8Chunk.liftMag_le db y
  have hx0 := Dot8Chunk.liftMag_nonneg da x
  have hb0 := Dot8Chunk.minNormal8_nonneg db
  have h := mul_le_mul hx hy (Dot8Chunk.liftMag_nonneg db y) (add_nonneg (abs_nonneg x) (Dot8Chunk.minNormal8_nonneg da))
  calc liftMag da x * liftMag db y ≤ (|x| + minNormal8 da) * (|y| + minNormal8 db) := h
    _ = |x * y| + minNormal8 db * |x| + minNormal8 da * |y| + minNormal8 da * minNormal8 db := by
      rw [abs_mul]; ring

/-- `liftMag_mul_le` summed over `k`: `Σ liftMag·liftMag ≤ Σ|va·vb| + mb·Σ|va| + ma·Σ|vb| + K·ma·mb`. -/
theorem sum_liftMag_mul_le (da db : DType) {K : ℕ} (va vb : Fin K → ℚ) :
    ∑ k, liftMag da (va k) * liftMag db (vb k) ≤
      ∑ k, |va k * vb k| + minNormal8 db * ∑ k, |va k| + minNormal8 da * ∑ k, |vb k| +
        K * (minNormal8 da * minNormal8 db) := by
  calc ∑ k, liftMag da (va k) * liftMag db (vb k)
      ≤ ∑ k, (|va k * vb k| + minNormal8 db * |va k| + minNormal8 da * |vb k| +
          minNormal8 da * minNormal8 db) := Finset.sum_le_sum fun k _ => liftMag_mul_le da db _ _
    _ = _ := by
      simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul]
      ring

/-- **One step of `acc += tl.dot(a, b)` from a 16-bit fact** (`DotF16`, `DotBF16`: `DotLiftAcc`),
for one output element `(i, n)`: whether the device ran `tl.dot(a, b, c)` or `c + tl.dot(a, b)`, the
result is finite and within `stepRel rel K·(|c| + Σ liftMag a·liftMag b) + stepAbs abs K` of
`c + Σ a·b`, given room in `float32`. `DotLoop.dotAdd_step` with lifted magnitudes and no domain. -/
theorem dotAdd_step {P : Platform} {fusion : Bool} {da db : DType} {pr : InputPrecision} {mi : Nat}
    {rel abs : ℕ → ℚ} (hacc : DotLiftAcc P ⟨da, db, .f32, pr, mi⟩ rel abs)
    {M N K : Nat} (hrel : 0 ≤ rel K) (habs : 0 ≤ abs K)
    {a : Tensor [M, K] da.Val} {b : Tensor [K, N] db.Val} {c r : Tensor [M, N] F32}
    (hr : P.dot ⟨da, db, .f32, pr, mi⟩ a b c r ∨
      ∃ d, P.dot ⟨da, db, .f32, pr, mi⟩ a b (Tensor.full (FloatV.ofPat 0)) d ∧
        ∀ ix, AccAdd fusion .f32 (d.get ix) (c.get ix) (r.get ix))
    (i : Fin M) (n : Fin N) {va vb : Fin K → ℚ}
    (hva : ∀ k, da.val? (a.get2 i k) = some (va k)) (hvb : ∀ k, db.val? (b.get2 k n) = some (vb k))
    (hc : (c.get2 i n).isFinite = true)
    (hroom : 2 * ((1 + rel K) * (|(c.get2 i n).toRat| + ∑ k, liftMag da (va k) * liftMag db (vb k)) +
      abs K + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - ((c.get2 i n).toRat + ∑ k, va k * vb k)| ≤
        DotLoop.stepRel rel K * (|(c.get2 i n).toRat| + ∑ k, liftMag da (va k) * liftMag db (vb k)) +
          DotLoop.stepAbs abs K := by
  unfold DotLoop.stepRel DotLoop.stepAbs
  have hSA : |∑ k, va k * vb k| ≤ ∑ k, liftMag da (va k) * liftMag db (vb k) :=
    (Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum fun k _ => Dot8Chunk.abs_mul_le_liftMag _ _ _ _)
  exact DotLoop.dotAdd_step_of (fusion := fusion) hrel habs i n hSA
    (fun c' r' vc hd' hvc' hroom' => hacc a b c' r' hd' i n va vb vc hva hvb hvc' hroom') hr hc hroom

/-- **On zero and normal values a `DotLiftAcc` fact is `DotAcc`** with the same constants: where
`liftMag` is `|q|` for both operand dtypes (`Dot8Chunk.liftMag_of_le`, `liftMag_zero`), `L` is
`|acc| + Σ|a·b|`. -/
theorem DotLiftAcc.dotAcc {P : Platform} {cfg : DotCfg} {rel abs : ℕ → ℚ} {dom : ℚ → Prop}
    (h : DotLiftAcc P cfg rel abs) (hda : ∀ q, dom q → liftMag cfg.a q = |q|)
    (hdb : ∀ q, dom q → liftMag cfg.b q = |q|) : DotAcc P cfg dom rel abs := by
  intro M N K a b acc r hr i n va vb vc hva hvb hd hvc hroom
  have hL : ∑ k, liftMag cfg.a (va k) * liftMag cfg.b (vb k) = ∑ k, |va k * vb k| :=
    Finset.sum_congr rfl fun k _ => by rw [hda _ (hd k).1, hdb _ (hd k).2, abs_mul]
  obtain ⟨vr, hvr, herr⟩ := h a b acc r hr i n va vb vc hva hvb hvc (by rw [hL]; exact hroom)
  exact ⟨vr, hvr, by rw [hL] at herr; exact herr⟩

end DotLift

/-! ## Satisfiability -/

namespace Platform.ref

variable {rel abs : MathFn → TVA.Math.Float.Format → ℝ}
  {sp : (fn : MathFn) → (f : TVA.Math.Float.Format) → FloatV f → Option (FloatV f)}

/-- **`DotLiftAcc` holds of `Platform.ref`** for a float accumulator, for `rel K ≥ u` and
`abs K ≥ η` when `K > 0` (`Platform.ref.dotAcc` and `DotAcc.liftAcc`). -/
theorem dotLiftAcc {cfg : DotCfg} {g : TVA.Math.Float.Format} (hg : cfg.acc.format? = some g)
    {relK absK : ℕ → ℚ} (hrel : ∀ K, 0 < K → g.u ≤ relK K) (habs : ∀ K, 0 < K → g.eta ≤ absK K)
    (hrel0 : ∀ K, 0 ≤ relK K) (habs0 : 0 ≤ absK 0) :
    DotLiftAcc (Platform.ref rel abs sp) cfg relK absK :=
  (Platform.ref.dotAcc hg (fun _ => True) hrel habs (hrel0 0) habs0).liftAcc hrel0

end Platform.ref

end TVA
