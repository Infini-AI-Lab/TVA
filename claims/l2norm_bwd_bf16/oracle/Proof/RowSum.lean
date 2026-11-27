import Claim
import TVA.Proof

/-!
# `l2norm_bwd_bf16`: the row sum `tl.sum(b_dy * b_y, 1)`

One row: the `BD ≤ 512` lanes `x k` (`dy`) and `y k` (`y`), their IEEE products `p k`, summed along
a tree of fusable additions of depth at most `9` (`tl.sumBound` with `Device.SumDepth`). The lanes
from `D` on are zeros. With `g`, `a` the row's real values, `S = Σ_{k<D} g_k·a_k` and
`A = Σ_{k<D} |g_k·a_k| ≤ 2^124`, the sum is finite and within `29·2^-24·A + 2600·2^-150` of `S`
(`rowsum_err`, from `sum_bcast_mul_depth_err`).
-/

namespace Claims.L2NormBwd

open TVA

/-- The `List.sum` over `List.range n` as a `Finset.sum`. -/
theorem list_sum_range (f : Nat → ℚ) (n : Nat) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]
      simp

/-- `(1 + 3u)^9 ≤ 1 + 27.01·u` for `u = 2^-24`. -/
theorem pow9_le : (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ 9 ≤ 1 + 2701 / 100 * (2 : ℚ) ^ (-24 : ℤ) := by
  norm_num [zpow_neg]

/-- **The bound of the row sum**, at depth `9` and at most `512` lanes:
`bcastDepthBound 9 K B ≤ 29u·B + 2600η`. -/
theorem bcast_le {K : Nat} (hK : K ≤ 512) {B : ℚ} (hB : 0 ≤ B) :
    bcastDepthBound 9 K B ≤ 29 * (2 : ℚ) ^ (-24 : ℤ) * B + 2600 * (2 : ℚ) ^ (-150 : ℤ) := by
  unfold bcastDepthBound Tree.errBound
  rw [Format.binary32_u, Format.binary32_eta]
  have hKq : (K : ℚ) ≤ 512 := by exact_mod_cast hK
  have hK1 : ((K - 1 : Nat) : ℚ) ≤ 511 := by
    have : K - 1 ≤ 511 := by omega
    exact_mod_cast this
  have hK0 : (0 : ℚ) ≤ ((K - 1 : Nat) : ℚ) := by positivity
  have hc := pow9_le
  have hc1 : 1 ≤ (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ 9 := one_le_pow₀ (by norm_num)
  set c := (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ 9 with hcdef
  set u : ℚ := (2 : ℚ) ^ (-24 : ℤ) with hu
  set η : ℚ := (2 : ℚ) ^ (-150 : ℤ) with hη
  have hu0 : 0 < u := by rw [hu]; positivity
  have hη0 : 0 < η := by rw [hη]; positivity
  have huv : u = 1 / 2 ^ 24 := by rw [hu]; norm_num [zpow_neg]
  have hηv : η = 1 / 2 ^ 150 := by rw [hη]; norm_num [zpow_neg]
  -- `(c − 1)·((1 + u)·B + K·η)`
  have h1 : (c - 1) * ((1 + u) * B + K * η) ≤ 2701 / 100 * u * ((1 + u) * B + 512 * η) := by
    have hpos : 0 ≤ (1 + u) * B + K * η := by positivity
    have hKe : (1 + u) * B + K * η ≤ (1 + u) * B + 512 * η := by nlinarith
    calc (c - 1) * ((1 + u) * B + K * η) ≤ 2701 / 100 * u * ((1 + u) * B + K * η) :=
          mul_le_mul_of_nonneg_right (by linarith) hpos
      _ ≤ 2701 / 100 * u * ((1 + u) * B + 512 * η) :=
          mul_le_mul_of_nonneg_left hKe (by positivity)
  -- `4η·(K − 1)·c`
  have h2 : 4 * η * ((K - 1 : Nat) : ℚ) * c ≤ 4 * η * 511 * (1 + 2701 / 100 * u) := by
    have := mul_le_mul hK1 hc (by linarith) (by norm_num)
    nlinarith
  have h3 : K * η ≤ 512 * η := by nlinarith
  have hfin : 2701 / 100 * u * ((1 + u) * B + 512 * η) + 4 * η * 511 * (1 + 2701 / 100 * u) +
      (u * B + 512 * η) ≤ 29 * u * B + 2600 * η := by
    rw [huv, hηv]
    have hB' : 0 ≤ B := hB
    nlinarith
  clear_value c u η
  linarith

/-- **The row sum.** The IEEE products `p k` of the lanes `x k`, `y k` (`k < BD ≤ 512`), summed
along a tree of fusable additions of depth at most `Dp ≤ 9`, with the lanes from `D` on zero and
`A = Σ_{k<D} |g_k·a_k| ≤ 2^124`, give a finite `s` within `29·2^-24·A + 2600·2^-150` of
`S = Σ_{k<D} g_k·a_k`. -/
theorem rowsum_err {fusion : Bool} {D BD Dp : Nat} (hDB : D ≤ BD) (hBD : BD ≤ 512) (hDp : Dp ≤ 9)
    (x y p : Nat → F32) {t : Tree} {s : F32} (a g : Nat → ℝ)
    (hp : ∀ k, k < BD → tl.MulRel .f32 (x k) (y k) (p k))
    (hcov : t.Covers BD) (hd : t.depth ≤ Dp) (ht : t.Eval (AddRel fusion) p s)
    (hx : ∀ k, k < BD → (x k).isFinite = true) (hy : ∀ k, k < BD → (y k).isFinite = true)
    (hxg : ∀ k, k < D → (x k).toReal = g k) (hya : ∀ k, k < D → (y k).toReal = a k)
    (hz : ∀ k, D ≤ k → k < BD → (x k).toReal = 0)
    (hA : rowAbs a g D ≤ 2 ^ 124) :
    s.isFinite = true ∧
      |s.toReal - rowDot a g D| ≤ 29 / 2 ^ 24 * rowAbs a g D + 2600 / 2 ^ 150 := by
  set Bq : ℚ := ((List.range BD).map fun k => |(x k).toRat * (y k).toRat|).sum with hBq
  set Sq : ℚ := ((List.range BD).map fun k => (x k).toRat * (y k).toRat).sum with hSq
  -- the lanes as reals
  have hlane : ∀ k, k < BD → ((x k).toRat * (y k).toRat : ℝ) =
      if k < D then g k * a k else 0 := by
    intro k hk
    split_ifs with hkD
    · rw [← hxg k hkD, ← hya k hkD]; rfl
    · have := hz k (by omega) hk
      simp only [FloatV.toReal] at this
      rw [this, zero_mul]
  have hBR : (Bq : ℝ) = rowAbs a g D := by
    rw [hBq, list_sum_range]
    push_cast
    rw [rowAbs, ← Finset.sum_subset (Finset.range_subset_range.mpr hDB)]
    · refine Finset.sum_congr rfl (fun k hk => ?_)
      have hk' := Finset.mem_range.mp hk
      have := hlane k (by omega)
      rw [if_pos hk'] at this
      rw [this]
    · intro k hk hkD
      have hk' := Finset.mem_range.mp hk
      have hkD' : ¬ k < D := fun h => hkD (Finset.mem_range.mpr h)
      have := hlane k hk'
      rw [if_neg hkD'] at this
      rw [this, abs_zero]
  have hSR : (Sq : ℝ) = rowDot a g D := by
    rw [hSq, list_sum_range]
    push_cast
    rw [rowDot, ← Finset.sum_subset (Finset.range_subset_range.mpr hDB)]
    · refine Finset.sum_congr rfl (fun k hk => ?_)
      have hk' := Finset.mem_range.mp hk
      have := hlane k (by omega)
      rw [if_pos hk'] at this
      rw [this]
    · intro k hk hkD
      have hk' := Finset.mem_range.mp hk
      have hkD' : ¬ k < D := fun h => hkD (Finset.mem_range.mpr h)
      have := hlane k hk'
      rw [if_neg hkD'] at this
      rw [this]
  have hB0 : 0 ≤ Bq := by
    rw [hBq, list_sum_range]; exact Finset.sum_nonneg (fun _ _ => abs_nonneg _)
  have hBle : Bq ≤ 2 ^ 124 := by
    have : (Bq : ℝ) ≤ 2 ^ 124 := by rw [hBR]; exact hA
    exact_mod_cast this
  -- each product is in range
  have hab : ∀ k, k < BD → |(x k).toRat * (y k).toRat| ≤ Format.binary32.maxVal := by
    intro k hk
    have h1 : |(x k).toRat * (y k).toRat| ≤ Bq := by
      rw [hBq, list_sum_range]
      exact Finset.single_le_sum (f := fun k => |(x k).toRat * (y k).toRat|)
        (fun _ _ => abs_nonneg _) (Finset.mem_range.mpr hk)
    refine h1.trans (hBle.trans ?_)
    rw [Format.binary32_maxVal]; norm_num
  -- room for the tree
  have hM : ((1 + Format.binary32.u) * Bq + BD * Format.binary32.eta) +
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 9 (BD - 1)
        ((1 + Format.binary32.u) * Bq + BD * Format.binary32.eta) ≤
        Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have hb := bcast_le hBD hB0
    unfold bcastDepthBound at hb
    rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
    rw [Format.binary32_u, Format.binary32_eta] at hb
    have hBDq : (BD : ℚ) ≤ 512 := by exact_mod_cast hBD
    have key : ∀ E u η : ℚ, u = 2 ^ (-24 : ℤ) → η = 2 ^ (-150 : ℤ) →
        E + (u * Bq + BD * η) ≤ 29 * u * Bq + 2600 * η →
        (1 + u) * Bq + BD * η + E ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * η := by
      intro E u η hu hη h
      subst hu hη
      norm_num [zpow_neg] at h ⊢
      nlinarith [hBle, hBDq, hB0]
    exact key _ _ _ rfl rfl hb
  obtain ⟨hfin, herr⟩ := sum_bcast_mul_depth_err (fusion := fusion) (K := BD) (D := 9) x y p hp
    hcov (hd.trans hDp) ht hx hy hab hM
  refine ⟨hfin, ?_⟩
  have hb := bcast_le hBD hB0
  have hq : |s.toRat - Sq| ≤ 29 * (2 : ℚ) ^ (-24 : ℤ) * Bq + 2600 * (2 : ℚ) ^ (-150 : ℤ) :=
    herr.trans hb
  have hR := (Rat.cast_le (K := ℝ)).mpr hq
  push_cast at hR
  rw [hSR, hBR] at hR
  simp only [FloatV.toReal]
  norm_num [zpow_neg] at hR ⊢
  linarith

end Claims.L2NormBwd
