import AttnBwdPre.Model

/-!
# `attn_bwd_pre_f16`: the error of one output

`row_bound`: `float16` values widened exactly to `float32`, multiplied (IEEE `float32`) and summed
along any tree of `D ≤ 256` leaves whose additions may be contracted with the products, give a
finite `float32` within `tol D (Σ |o·do|)` of the exact `Σ o·do`. It is the library's
`sum_bcast_mul_tree_err` (`TVA.Lemmas.SumFused`, about exactly that computation: IEEE products
summed along any tree with `AddRel` nodes), with the room it needs (`room`) and the tolerance written
out (`tol_eq`). `tol_le` bounds the tolerance at the largest covered `HEAD_DIM`.
-/

namespace Kernels.AttnBwdPre

open TVA TVA.Math.Float

/-! ## The tolerance -/

/-- `tol` is the library's bound for rounded products summed along any tree with fusable
additions. -/
theorem tol_eq (D : Nat) (B : ℚ) : tol D B = bcastTreeBound D B := by
  unfold tol bcastTreeBound
  rw [Format.binary32_u, Format.binary32_eta]

/-- `(1 + 3·2^-24)^255`, the growth factor of `D − 1 ≤ 255` additions, is below `1 + 511·2^-23`. -/
theorem growth_le : ((1 : ℚ) + 3 * 2 ^ (-24 : ℤ)) ^ 255 ≤ 1 + 511 * 2 ^ (-23 : ℤ) := by
  norm_num

/-- The tolerance grows with `D`. -/
theorem tol_mono {D D' : Nat} (hD : D ≤ D') {B : ℚ} (hB : 0 ≤ B) : tol D B ≤ tol D' B := by
  unfold tol
  have hδ : (0 : ℚ) ≤ 3 * 2 ^ (-24 : ℤ) := by positivity
  have hη : (0 : ℚ) ≤ 4 * 2 ^ (-150 : ℤ) := by positivity
  have hDD : (D : ℚ) ≤ D' := by exact_mod_cast hD
  have hA : 0 ≤ (1 + 2 ^ (-24 : ℤ)) * B + D * 2 ^ (-150 : ℤ) := by positivity
  have hAA : (1 + 2 ^ (-24 : ℤ)) * B + D * 2 ^ (-150 : ℤ) ≤ (1 + 2 ^ (-24 : ℤ)) * B + D' * 2 ^ (-150 : ℤ) := by
    have : (D : ℚ) * 2 ^ (-150 : ℤ) ≤ D' * 2 ^ (-150 : ℤ) := mul_le_mul_of_nonneg_right hDD (by positivity)
    linarith
  have h1 := Tree.errBound_mono hδ hη (d := D - 1) (d' := D' - 1) (N := D - 1) (N' := D' - 1)
    (by omega) (by omega) hA hAA
  have h2 : (D : ℚ) * 2 ^ (-150 : ℤ) ≤ D' * 2 ^ (-150 : ℤ) := mul_le_mul_of_nonneg_right hDD (by positivity)
  linarith

/-- **The tolerance at the covered sizes**: for `HEAD_DIM ≤ 256`, `tol` is below
`2^-14·B + 2^-138`, where `B = Σ_k |o[k]·do[k]|`. -/
theorem tol_le {D : Nat} (hD : D ≤ 256) {B : ℚ} (hB : 0 ≤ B) :
    tol D B ≤ 2 ^ (-14 : ℤ) * B + 2 ^ (-138 : ℤ) := by
  refine (tol_mono hD hB).trans ?_
  unfold tol Tree.errBound
  have hc := growth_le
  have hc1 : (1 : ℚ) ≤ (1 + 3 * 2 ^ (-24 : ℤ)) ^ 255 := one_le_pow₀ (by norm_num)
  norm_num only [Nat.cast_ofNat] at hc hc1 ⊢
  generalize ((1 : ℚ) + 3 * 2 ^ (-24 : ℤ)) ^ 255 = c at hc hc1 ⊢
  nlinarith

/-! ## Room in `float32` -/

/-- With `D ≤ 256` products of absolute sum at most `2^40`, the tree has room in `float32`: the
hypothesis `sum_bcast_mul_tree_err` needs. -/
theorem room {D : Nat} (hD : D ≤ 256) {B : ℚ} (hB : 0 ≤ B) (hB' : B ≤ 2 ^ 40) :
    ((1 + Format.binary32.u) * B + D * Format.binary32.eta) +
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (D - 1) (D - 1)
        ((1 + Format.binary32.u) * B + D * Format.binary32.eta) ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
  have ht := tol_le hD hB
  have hnn : 0 ≤ 2 ^ (-24 : ℤ) * B + D * (2 : ℚ) ^ (-150 : ℤ) := by positivity
  unfold tol at ht
  rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
  have hD' : (D : ℚ) ≤ 256 := by exact_mod_cast hD
  have h1 : (D : ℚ) * 2 ^ (-150 : ℤ) ≤ 256 * 2 ^ (-150 : ℤ) := mul_le_mul_of_nonneg_right hD' (by positivity)
  have h2 : (1 + (2 : ℚ) ^ (-24 : ℤ)) * B ≤ 2 * 2 ^ 40 := by
    have : (1 + (2 : ℚ) ^ (-24 : ℤ)) ≤ 2 := by norm_num
    nlinarith
  have h3 : (2 : ℚ) ^ (-14 : ℤ) * B ≤ 2 ^ 40 := by
    have : (2 : ℚ) ^ (-14 : ℤ) ≤ 1 := by norm_num
    nlinarith
  norm_num at ht h1 h2 h3 ⊢
  linarith

/-! ## One output -/

/-- A finite `float16` value is at most `65504` in magnitude. -/
theorem f16_abs_le {x : F16} (hx : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have := FloatV.abs_toRat_le_maxVal Format.binary16_regular hx
  rwa [Format.binary16_maxVal] at this

/-- **One output.** Let `av k`, `bv k` be finite `float32` values equal to the finite `float16`
values `vo k`, `vdo k` (`k < D`, `D ≤ 256`), `pv k` their IEEE products, and `c` the value of any
tree over `0, …, D−1` of `AddRel fusion` additions of the `pv k`. Then `c` is finite and within
`tol D (dotAbs vo vdo D)` of `dotSum vo vdo D`. -/
theorem row_bound {fusion : Bool} {D : Nat} (hD : D ≤ 256) (vo vdo : Nat → F16) (av bv pv : Nat → F32)
    (hvo : ∀ k, k < D → (vo k).isFinite = true) (hvdo : ∀ k, k < D → (vdo k).isFinite = true)
    (hav : ∀ k, k < D → (av k).isFinite = true ∧ (av k).toRat = (vo k).toRat)
    (hbv : ∀ k, k < D → (bv k).isFinite = true ∧ (bv k).toRat = (vdo k).toRat)
    (hp : ∀ k, k < D → tl.MulRel .f32 (av k) (bv k) (pv k))
    {t : Tree} {c : F32} (hcov : t.Covers D) (ht : t.Eval (AddRel fusion) pv c) :
    c.isFinite = true ∧ |c.toRat - dotSum vo vdo D| ≤ tol D (dotAbs vo vdo D) := by
  have hsum : ((List.range D).map fun k => (av k).toRat * (bv k).toRat).sum = dotSum vo vdo D := by
    rw [sum_range_fin, dotSum, Finset.sum_range]
    exact Finset.sum_congr rfl fun k _ => by rw [(hav k k.isLt).2, (hbv k k.isLt).2]
  have habs : ((List.range D).map fun k => |(av k).toRat * (bv k).toRat|).sum = dotAbs vo vdo D := by
    rw [sum_range_fin, dotAbs, Finset.sum_range]
    exact Finset.sum_congr rfl fun k _ => by rw [(hav k k.isLt).2, (hbv k k.isLt).2]
  -- each product is at most `65504^2 < 2^32`
  have hprod : ∀ k, k < D → |(vo k).toRat * (vdo k).toRat| ≤ 2 ^ 32 := by
    intro k hk
    rw [abs_mul]
    have h1 := f16_abs_le (hvo k hk)
    have h2 := f16_abs_le (hvdo k hk)
    have h3 : |(vo k).toRat| * |(vdo k).toRat| ≤ 65504 * 65504 :=
      mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
    linarith
  have hB0 : 0 ≤ dotAbs vo vdo D := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hB1 : dotAbs vo vdo D ≤ 2 ^ 40 := by
    have : dotAbs vo vdo D ≤ ∑ _k ∈ Finset.range D, (2 : ℚ) ^ 32 :=
      Finset.sum_le_sum fun k hk => hprod k (Finset.mem_range.mp hk)
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at this
    have hD' : (D : ℚ) ≤ 256 := by exact_mod_cast hD
    nlinarith
  have hab : ∀ k, k < D → |(av k).toRat * (bv k).toRat| ≤ Format.binary32.maxVal := by
    intro k hk
    rw [(hav k hk).2, (hbv k hk).2, Format.binary32_maxVal]
    have := hprod k hk
    have : (2 : ℚ) ^ 32 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hfin, herr⟩ := sum_bcast_mul_tree_err (fusion := fusion) (K := D) av bv pv hp hcov ht
    (fun k hk => (hav k hk).1) (fun k hk => (hbv k hk).1) hab (by rw [habs]; exact room hD hB0 hB1)
  rw [hsum, habs, ← tol_eq] at herr
  exact ⟨hfin, herr⟩

end Kernels.AttnBwdPre
