import Claim
import TVA.Proof

/-!
# `attn_bwd_pre_f16`: the value of one output element
-/

namespace Claims.AttnBwdPre

open TVA


/-- A `float16` widened to `float32` by a conversion is the exact value. -/
theorem cvt_f16_f32 {P : Platform} {x : F16} {y : F32}
    (h : ConvRel P .f16 .f32 x y) (hx : x.isFinite = true) :
    y = FloatV.cvt x ∧ y.isFinite = true ∧ y.toRat = x.toRat := by
  have hne : (Format.binary16 : Format) ≠ Format.binary32 := by
    decide
  have hcv := FloatV.toRat_cvt_of_wider (f := Format.binary16)
    (g := Format.binary32) Format.binary16_regular
    Format.binary32_regular Format.wider_binary16_binary32
    RoundMode.rn hx
  rw [FloatV.cvtWith_rn] at hcv
  simp only [ConvRel, ofFloatRel, FloatRel, hne] at h
  rcases h with ⟨-, -, hnan⟩ | ⟨-, hf8, -⟩ | ⟨hf8, -⟩
  · rcases hnan with hxy | ⟨hn1, -⟩
    · exact ⟨hxy.symm, by rw [← hxy]; exact hcv.1, by rw [← hxy]; exact hcv.2⟩
    · have hf := (FloatV.isFinite_iff (x := FloatV.cvt x)).mp hcv.1
      rw [hf.1] at hn1
      exact absurd hn1 (by decide)
  · rw [Format.F8] at hf8
    rcases hf8 with h | h
    · exact absurd h (by decide)
    · exact absurd h (by decide)
  · rw [Format.F8] at hf8
    rcases hf8 with h | h
    · exact absurd h (by decide)
    · exact absurd h (by decide)

/-- The sum of a list over `List.range n` is the sum over `Finset.range n`. -/
theorem sum_range_list (g : Nat → ℚ) (n : Nat) :
    ((List.range n).map g).sum = ∑ k ∈ Finset.range n, g k := by
  rw [sum_range_fin]
  exact (Finset.sum_range (fun k => g k) (n := n)).symm

/-- **Room for the products**: `HEAD_DIM ≤ 256` finite `float16` products, widened and rounded
once each in `float32` and summed along a tree of depth `⌈log₂ HEAD_DIM⌉`, do not overflow. -/
theorem room_bound (K : Nat) {B : ℚ} (hK : K ≤ 256) (hB0 : 0 ≤ B) (hB : B ≤ 256 * 65504 ^ 2) :
    (1 + Format.binary32.u) * B + (K : ℚ) * Format.binary32.eta +
        Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta)
          (Nat.clog 2 K) (K - 1)
          ((1 + Format.binary32.u) * B + (K : ℚ) * Format.binary32.eta)
      ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
  have hu : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u
  have he : Format.binary32.eta = 2 ^ (-150 : ℤ) := Format.binary32_eta
  have hu0 : (0 : ℚ) ≤ 3 * Format.binary32.u := by rw [hu]; norm_num
  have he0 : (0 : ℚ) ≤ 4 * Format.binary32.eta := by rw [he]; norm_num
  have hu1 : Format.binary32.u ≤ 1 := by rw [hu]; norm_num
  have he1 : Format.binary32.eta ≤ 1 := by rw [he]; norm_num
  have he0' : (0 : ℚ) ≤ Format.binary32.eta := by rw [he]; norm_num
  have hKq : (K : ℚ) ≤ 256 := by exact_mod_cast hK
  have hA0 : (0 : ℚ) ≤ (1 + Format.binary32.u) * B +
      (K : ℚ) * Format.binary32.eta := by
    have h1 : (0 : ℚ) ≤ 1 + Format.binary32.u := by linarith
    have h3 : (0 : ℚ) ≤ (K : ℚ) := by positivity
    nlinarith [mul_nonneg h1 hB0, mul_nonneg h3 he0']
  have hAle : (1 + Format.binary32.u) * B + (K : ℚ) * Format.binary32.eta ≤
      2 * (256 * 65504 ^ 2) + 256 := by
    have h1 : (1 + Format.binary32.u) * B ≤ 2 * (256 * 65504 ^ 2) := by
      have h2 : (1 + Format.binary32.u) ≤ 2 := by linarith
      calc (1 + Format.binary32.u) * B ≤ 2 * B := mul_le_mul_of_nonneg_right h2 hB0
        _ ≤ 2 * (256 * 65504 ^ 2) := by
            exact mul_le_mul_of_nonneg_left hB (by norm_num)
    have h2 : (K : ℚ) * Format.binary32.eta ≤ 256 := by
      calc (K : ℚ) * Format.binary32.eta ≤ 256 * 1 := by
              exact mul_le_mul hKq he1 he0' (by norm_num)
        _ = 256 := by norm_num
    linarith
  have hClog : Nat.clog 2 K ≤ 8 := by
    have h := Nat.clog_mono_right 2 hK
    rw [show (256 : Nat) = 2 ^ 8 by norm_num, Nat.clog_pow 2 8 (by norm_num)] at h
    exact h
  have hN : K - 1 ≤ 255 := by omega
  have hmono : Tree.errBound (3 * Format.binary32.u)
        (4 * Format.binary32.eta) (Nat.clog 2 K) (K - 1)
        ((1 + Format.binary32.u) * B + (K : ℚ) * Format.binary32.eta) ≤
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 8 255
        (2 * (256 * 65504 ^ 2) + 256) :=
    Tree.errBound_mono hu0 he0 hClog hN hA0 hAle
  have hpow : (1 + 3 * Format.binary32.u) ^ 8 ≤ (2 : ℚ) := by
    rw [hu]; norm_num
  have hstep : Tree.errBound (3 * Format.binary32.u)
        (4 * Format.binary32.eta) 8 255 (2 * (256 * 65504 ^ 2) + 256) ≤
      2 * (256 * 65504 ^ 2) + 256 + 2040 := by
    unfold Tree.errBound
    have hAub : (0 : ℚ) ≤ 2 * (256 * 65504 ^ 2) + 256 := by norm_num
    have hp1 : (1 + 3 * Format.binary32.u) ^ 8 - 1 ≤ 1 := by linarith
    have h1 : ((1 + 3 * Format.binary32.u) ^ 8 - 1) *
        (2 * (256 * 65504 ^ 2) + 256) ≤ 2 * (256 * 65504 ^ 2) + 256 := by
      calc ((1 + 3 * Format.binary32.u) ^ 8 - 1) * (2 * (256 * 65504 ^ 2) + 256)
          ≤ 1 * (2 * (256 * 65504 ^ 2) + 256) := mul_le_mul_of_nonneg_right hp1 hAub
        _ = 2 * (256 * 65504 ^ 2) + 256 := by ring
    have h2 : (4 * Format.binary32.eta) * ((255 : ℕ) : ℚ) *
        (1 + 3 * Format.binary32.u) ^ 8 ≤ 2040 := by
      have hq : ((255 : ℕ) : ℚ) = 255 := by norm_num
      rw [hq]
      calc (4 * Format.binary32.eta) * 255 *
            (1 + 3 * Format.binary32.u) ^ 8
          ≤ (4 * Format.binary32.eta) * 255 * 2 := by
            apply mul_le_mul_of_nonneg_left hpow
            linarith
        _ = 2040 * Format.binary32.eta := by ring
        _ ≤ 2040 * 1 := by
            apply mul_le_mul_of_nonneg_left he1
            norm_num
        _ = 2040 := by norm_num
    linarith
  have hroom : (4 * (256 * 65504 ^ 2) + 3000 : ℚ) ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [Format.binary32_maxVal, Format.binary32_eta]
    norm_num
  linarith

/-- **One output element**: `HEAD_DIM` products of widened `float16`s, rounded once each in
`float32` and summed along a tree of depth `⌈log₂ HEAD_DIM⌉` with fusable additions, are within
`tol` of the exact dot product. -/
theorem row_value {P : Platform} {fusion : Bool} {K : Nat} (hK : K ≤ 256)
    {vo vdo : Nat → F16} {a b p : Nat → F32} {t : Tree} {r : F32}
    (hvo : ∀ k, k < K → (vo k).isFinite = true)
    (hvdo : ∀ k, k < K → (vdo k).isFinite = true)
    (ha : ∀ k, k < K → ConvRel P .f16 .f32 (vo k) (a k))
    (hb : ∀ k, k < K → ConvRel P .f16 .f32 (vdo k) (b k))
    (hp : ∀ k, k < K → tl.MulRel .f32 (a k) (b k) (p k))
    (hcov : t.Covers K) (hdep : t.depth ≤ Nat.clog 2 K)
    (ht : t.Eval (AccAdd fusion .f32) p r) :
    r.isFinite = true ∧ |r.toRat - dotSum vo vdo K| ≤ tol (dotAbs vo vdo K) := by
  have haf : ∀ k, k < K → (a k).isFinite = true :=
    fun k hk => (cvt_f16_f32 (ha k hk) (hvo k hk)).2.1
  have har : ∀ k, k < K → (a k).toRat = (vo k).toRat :=
    fun k hk => (cvt_f16_f32 (ha k hk) (hvo k hk)).2.2
  have hbf : ∀ k, k < K → (b k).isFinite = true :=
    fun k hk => (cvt_f16_f32 (hb k hk) (hvdo k hk)).2.1
  have hbr : ∀ k, k < K → (b k).toRat = (vdo k).toRat :=
    fun k hk => (cvt_f16_f32 (hb k hk) (hvdo k hk)).2.2
  have hAB : ∀ k, k < K → (a k).toRat * (b k).toRat = (vo k).toRat * (vdo k).toRat :=
    fun k hk => by rw [har k hk, hbr k hk]
  have hdot : dotSum vo vdo K =
      ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum := by
    unfold dotSum
    rw [← sum_range_list]
    congr 1
    exact List.map_congr_left (fun k hk => (hAB k (List.mem_range.mp hk)).symm)
  have hdotabs : dotAbs vo vdo K =
      ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum := by
    unfold dotAbs
    rw [← sum_range_list]
    congr 1
    exact List.map_congr_left (fun k hk => by rw [hAB k (List.mem_range.mp hk)])
  have habs : ∀ k, k < K → |(a k).toRat * (b k).toRat| ≤ (65504 : ℚ) ^ 2 := by
    intro k hk
    have h1 : |(a k).toRat| ≤ (65504 : ℚ) := by
      rw [har k hk]
      have h := FloatV.abs_toRat_le_maxVal (f := Format.binary16)
        Format.binary16_regular (hvo k hk)
      rwa [Format.binary16_maxVal] at h
    have h2 : |(b k).toRat| ≤ (65504 : ℚ) := by
      rw [hbr k hk]
      have h := FloatV.abs_toRat_le_maxVal (f := Format.binary16)
        Format.binary16_regular (hvdo k hk)
      rwa [Format.binary16_maxVal] at h
    rw [abs_mul]
    have hm := mul_le_mul h1 h2 (abs_nonneg _) (by norm_num : (0 : ℚ) ≤ 65504)
    calc |(a k).toRat| * |(b k).toRat| ≤ 65504 * 65504 := hm
      _ = (65504 : ℚ) ^ 2 := by norm_num
  have habsm : ∀ k, k < K → |(a k).toRat * (b k).toRat| ≤
      Format.binary32.maxVal := by
    intro k hk
    refine (habs k hk).trans ?_
    rw [Format.binary32_maxVal]
    norm_num
  have hB0 : (0 : ℚ) ≤
      ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum := by
    apply List.sum_nonneg
    intro x hx
    obtain ⟨k, -, rfl⟩ := List.mem_map.mp hx
    exact abs_nonneg _
  have hB : ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum ≤
      256 * (65504 : ℚ) ^ 2 := by
    rw [sum_range_list]
    have h1 : ∑ k ∈ Finset.range K, |(a k).toRat * (b k).toRat| ≤
        ∑ k ∈ Finset.range K, (65504 : ℚ) ^ 2 :=
      Finset.sum_le_sum (fun k hk => habs k (Finset.mem_range.mp hk))
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at h1
    have h2 : (K : ℚ) * (65504 : ℚ) ^ 2 ≤ 256 * (65504 : ℚ) ^ 2 :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast hK) (by norm_num)
    linarith
  obtain ⟨hfin, herr⟩ := sum_bcast_mul_depth_err (fusion := fusion) (K := K) (D := Nat.clog 2 K)
    a b p hp hcov hdep ht haf hbf habsm (room_bound K hK hB0 hB)
  refine ⟨hfin, ?_⟩
  rw [hdot, hdotabs]
  refine herr.trans ?_
  unfold bcastDepthBound
  rw [Format.binary32_u, Format.binary32_eta]
  exact fine_le hK hB0

end Claims.AttnBwdPre
