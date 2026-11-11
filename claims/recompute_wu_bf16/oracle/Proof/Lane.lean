import Claim
import TVA.Proof
import Proof.Num

/-!
# One lane of the operands

The `bfloat16` operands of the two dots, computed lane by lane from the loaded values: line 115's
`vb = v · beta` and line 132's `kb = k · beta · 2^g`, for either dtype of `beta`. Each is finite
and within a stated bound of the exact product.
-/

namespace Claims.RecomputeWU

open TVA

/-- The `bfloat16` operand `vb` of line 115 for a `float32` `beta`. -/
def vb32 (v : BF16) (β : F32) : BF16 := FloatV.cvt (FloatV.mul (FloatV.cvt v : F32) β)

/-- The `bfloat16` operand `vb` of line 115 for a `bfloat16` `beta`. -/
def vb16 (v β : BF16) : BF16 := FloatV.mul v β

/-- A rounding of `q` is at most `B (1 + u) + η` in magnitude when `|q| ≤ B`. -/
theorem rnd_le {p q B u η : ℚ} (hq : |q| ≤ B) (hu : 0 ≤ u) (h : |p - q| ≤ |q| * u + η) :
    |p| ≤ B * (1 + u) + η := by
  have h1 : |p| ≤ |q| + |p - q| := by
    have := abs_sub_abs_le_abs_sub p q
    linarith
  nlinarith [abs_nonneg q]

/-- `|v · beta| ≤ 2^40` for `|v| ≤ 2^32`, `|beta| ≤ 2^8`. -/
theorem abs_mul_le_2_40 {v β : ℚ} (hv : |v| ≤ 2 ^ 32) (hβ : |β| ≤ 2 ^ 8) : |v * β| ≤ 2 ^ 40 := by
  rw [abs_mul]
  calc |v| * |β| ≤ 2 ^ 32 * 2 ^ 8 := mul_le_mul hv hβ (abs_nonneg _) (by positivity)
    _ = _ := by norm_num

/-- **Line 115 for a `float32` `beta`**: `vb` is finite and at most `2^41`, and so is the `float32` product before it. -/
theorem vb32_ok {v : BF16} {β : F32} (hv : KVOk v) (hβ : BetaOk β) :
    (FloatV.mul (FloatV.cvt v : F32) β).isFinite = true ∧
    (vb32 v β).isFinite = true ∧ |(vb32 v β).toRat| ≤ 2 ^ 41 := by
  obtain ⟨hvf, hvb⟩ := hv
  obtain ⟨hβf, hββ⟩ := hβ
  obtain ⟨h1, h2, -⟩ := FloatV.cvt_bf16_f32_exact hvf
  have h5 := abs_mul_le_2_40 hvb hββ
  have hmax : |(FloatV.cvt v : F32).toRat * β.toRat| ≤ Format.binary32.maxVal := by
    rw [h2, Format.binary32_maxVal]; exact h5.trans (by norm_num)
  obtain ⟨h3, h4⟩ := FloatV.mul_err Format.binary32_regular h1 hβf hmax
  rw [h2] at h4
  have hp := rnd_le h5 (u := Format.binary32.u) (η := Format.binary32.eta) (by norm_num [Format.u_eq, Format.binary32])
    h4
  rw [Format.binary32_u, Format.binary32_eta] at hp
  obtain ⟨h6, h7⟩ := FloatV.cvt_err (g := Format.bfloat16) Format.bfloat16_regular h3
    (by rw [Format.bfloat16_maxVal]; refine hp.trans ?_; norm_num)
  refine ⟨h3, h6, ?_⟩
  have hq := rnd_le hp (u := Format.bfloat16.u) (η := Format.bfloat16.eta) (by rw [Format.bfloat16_u]; norm_num) h7
  rw [Format.bfloat16_u, Format.bfloat16_eta] at hq
  refine hq.trans ?_
  norm_num

/-- **Line 115 for a `bfloat16` `beta`**: `vb` is finite and at most `2^41`. -/
theorem vb16_ok {v β : BF16} (hv : KVOk v) (hβ : BetaOkB β) :
    (vb16 v β).isFinite = true ∧ |(vb16 v β).toRat| ≤ 2 ^ 41 := by
  obtain ⟨hvf, hvb⟩ := hv
  obtain ⟨hβf, hββ⟩ := hβ
  have h5 := abs_mul_le_2_40 hvb hββ
  obtain ⟨h3, h4⟩ := FloatV.mul_err Format.bfloat16_regular hvf hβf
    (by rw [Format.bfloat16_maxVal]; exact h5.trans (by norm_num))
  refine ⟨h3, ?_⟩
  have hq := rnd_le h5 (u := Format.bfloat16.u) (η := Format.bfloat16.eta) (by rw [Format.bfloat16_u]; norm_num) h4
  rw [Format.bfloat16_u, Format.bfloat16_eta] at hq
  refine hq.trans ?_
  norm_num


/-- `kb`'s tail: a finite `float32` `p1` within `|x|·ε + δ` of `x` (`|x| ≤ 2^40`), multiplied by the
`exp2` result `e` of `g ≤ 8`, rounded to `bfloat16`: finite, within `α·|x|·2^g + 2^-125 (|x| + 1)`
of `x · 2^g`. -/
theorem kb_tail {x ε δ g : ℝ} {p1 e : F32} (hg : g ≤ 8) (hx : |x| ≤ 2 ^ 40)
    (hε : ε ≤ 2 ^ (-8 : ℤ)) (hδ : δ ≤ 2 ^ (-134 : ℤ)) (hp1 : p1.isFinite = true)
    (h1 : |p1.toReal - x| ≤ |x| * ε + δ) (hef : e.isFinite = true)
    (he : |e.toReal - 2 ^ g| ≤ 2 ^ (-22 : ℤ) * 2 ^ g + 2 ^ (-126 : ℤ)) :
    (FloatV.mul p1 e).isFinite = true ∧ (FloatV.cvt (FloatV.mul p1 e) : BF16).isFinite = true ∧
      |(FloatV.cvt (FloatV.mul p1 e) : BF16).toReal - x * 2 ^ g| ≤
        (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ)) * (|x| * 2 ^ g) + 2 ^ (-125 : ℤ) * (|x| + 1) := by
  have hG0 := two_rpow_pos g
  have hG8 := two_rpow_le hg
  have hxε : |x| * ε ≤ |x| * 2 ^ (-8 : ℤ) := mul_le_mul_of_nonneg_left hε (abs_nonneg x)
  have hx8 : |x| * 2 ^ (-8 : ℤ) ≤ 2 ^ 32 := by
    calc |x| * 2 ^ (-8 : ℤ) ≤ 2 ^ 40 * 2 ^ (-8 : ℤ) :=
          mul_le_mul_of_nonneg_right hx (by positivity)
      _ = 2 ^ 32 := by norm_num
  have hp1b : |p1.toReal| ≤ 2 ^ 41 := by
    have h := abs_sub_abs_le_abs_sub p1.toReal x
    have : (2 : ℝ) ^ (-134 : ℤ) ≤ 1 := by norm_num
    have : (2 : ℝ) ^ 40 + 2 ^ 32 + 1 ≤ 2 ^ 41 := by norm_num
    linarith
  have heb : |e.toReal| ≤ 2 ^ 9 := by
    have h := abs_sub_abs_le_abs_sub e.toReal (2 ^ g)
    rw [abs_of_pos hG0] at h
    have : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ g ≤ 2 ^ (-22 : ℤ) * 2 ^ 8 :=
      mul_le_mul_of_nonneg_left hG8 (by positivity)
    have : (2 : ℝ) ^ (-22 : ℤ) * 2 ^ 8 + 2 ^ (-126 : ℤ) ≤ 1 := by norm_num
    have : (2 : ℝ) ^ 8 + 1 ≤ 2 ^ 9 := by norm_num
    linarith
  have hpe : |p1.toReal * e.toReal| ≤ 2 ^ 50 := by
    rw [abs_mul]
    calc |p1.toReal| * |e.toReal| ≤ 2 ^ 41 * 2 ^ 9 :=
          mul_le_mul hp1b heb (abs_nonneg _) (by positivity)
      _ = 2 ^ 50 := by norm_num
  obtain ⟨hm1, hm2⟩ := FloatV.mul_err_real Format.binary32_regular hp1 hef
    (by rw [Format.binary32_maxVal_real]; exact hpe.trans (by norm_num))
  rw [Format.binary32_u_real, Format.binary32_eta_real] at hm2
  have hp2b : |(FloatV.mul p1 e).toReal| ≤ 2 ^ 51 := by
    have h := abs_sub_abs_le_abs_sub (FloatV.mul p1 e).toReal (p1.toReal * e.toReal)
    have : |p1.toReal * e.toReal| * 2 ^ (-24 : ℤ) ≤ 2 ^ 50 * 2 ^ (-24 : ℤ) :=
      mul_le_mul_of_nonneg_right hpe (by positivity)
    have : (2 : ℝ) ^ 50 * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) ≤ 2 ^ 50 := by norm_num
    have : (2 : ℝ) ^ 50 + 2 ^ 50 ≤ 2 ^ 51 := by norm_num
    linarith
  obtain ⟨hc1, hc2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hm1
    (by rw [Format.bfloat16_maxVal_real]; exact hp2b.trans (by norm_num))
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at hc2
  exact ⟨hm1, hc1, chain_core hG0 hG8 hε hδ he h1 hm2 hc2⟩

/-- The `bfloat16` operand `kb` of line 132 for a `float32` `beta`: `k` widened, times `beta`, times
`exp2(g)`'s result `e` (both `float32` products), rounded to `bfloat16`. -/
def kb32 (k : BF16) (β e : F32) : BF16 :=
  FloatV.cvt (FloatV.mul (FloatV.mul (FloatV.cvt k : F32) β) e)

/-- As `kb32` for a `bfloat16` `beta`: the product `k · beta` is rounded to `bfloat16` first, then
widened. -/
def kb16 (k β : BF16) (e : F32) : BF16 :=
  FloatV.cvt (FloatV.mul (FloatV.cvt (FloatV.mul k β : BF16) : F32) e)

/-- **Line 132's operand for a `float32` `beta`**: finite, within `α·|k beta|·2^g +
2^-125 (|k beta| + 1)` of `k · beta · 2^g`, `α = 2^-7 + 2^-10`. -/
theorem kb32_ok {k : BF16} {β e : F32} {g : ℝ} (hk : KVOk k) (hβ : BetaOk β) (hg : g ≤ 8)
    (hef : e.isFinite = true) (he : |e.toReal - 2 ^ g| ≤ 2 ^ (-22 : ℤ) * 2 ^ g + 2 ^ (-126 : ℤ)) :
    (FloatV.mul (FloatV.cvt k : F32) β).isFinite = true ∧
    (FloatV.mul (FloatV.mul (FloatV.cvt k : F32) β) e).isFinite = true ∧ (kb32 k β e).isFinite = true ∧
      |(kb32 k β e).toReal - k.toReal * β.toReal * 2 ^ g| ≤
        (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ)) * (|k.toReal * β.toReal| * 2 ^ g) +
          2 ^ (-125 : ℤ) * (|k.toReal * β.toReal| + 1) := by
  obtain ⟨hkf, hkb⟩ := hk
  obtain ⟨hβf, hββ⟩ := hβ
  obtain ⟨h1, h2, h3⟩ := FloatV.cvt_bf16_f32_exact hkf
  have hkb' : |k.toReal| ≤ 2 ^ 32 := by
    unfold FloatV.toReal; exact_mod_cast hkb
  have hββ' : |β.toReal| ≤ 2 ^ 8 := by
    unfold FloatV.toReal; exact_mod_cast hββ
  have hx := abs_kb_le hkb' hββ'
  have hmax : |(FloatV.cvt k : F32).toReal * β.toReal| ≤ (Format.binary32.maxVal : ℝ) := by
    rw [h3, Format.binary32_maxVal_real]; exact hx.trans (by norm_num)
  obtain ⟨hm1, hm2⟩ := FloatV.mul_err_real Format.binary32_regular h1 hβf hmax
  rw [Format.binary32_u_real, Format.binary32_eta_real, h3] at hm2
  obtain ⟨t1, t2, t3⟩ := kb_tail (ε := 2 ^ (-24 : ℤ)) (δ := 2 ^ (-150 : ℤ)) hg hx (by norm_num)
    (by norm_num) hm1 hm2 hef he
  exact ⟨hm1, t1, t2, t3⟩


/-- **Line 132's operand for a `bfloat16` `beta`**: as `kb32_ok`, with the extra rounding of
`k · beta` to `bfloat16`. -/
theorem kb16_ok {k β : BF16} {e : F32} {g : ℝ} (hk : KVOk k) (hβ : BetaOkB β) (hg : g ≤ 8)
    (hef : e.isFinite = true) (he : |e.toReal - 2 ^ g| ≤ 2 ^ (-22 : ℤ) * 2 ^ g + 2 ^ (-126 : ℤ)) :
    (FloatV.mul k β : BF16).isFinite = true ∧
    (FloatV.mul (FloatV.cvt (FloatV.mul k β : BF16) : F32) e).isFinite = true ∧
    (kb16 k β e).isFinite = true ∧
      |(kb16 k β e).toReal - k.toReal * β.toReal * 2 ^ g| ≤
        (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ)) * (|k.toReal * β.toReal| * 2 ^ g) +
          2 ^ (-125 : ℤ) * (|k.toReal * β.toReal| + 1) := by
  obtain ⟨hkf, hkb⟩ := hk
  obtain ⟨hβf, hββ⟩ := hβ
  have hkb' : |k.toReal| ≤ 2 ^ 32 := by
    unfold FloatV.toReal; exact_mod_cast hkb
  have hββ' : |β.toReal| ≤ 2 ^ 8 := by
    unfold FloatV.toReal; exact_mod_cast hββ
  have hx := abs_kb_le hkb' hββ'
  obtain ⟨hm1, hm2⟩ := FloatV.mul_err_real Format.bfloat16_regular hkf hβf
    (by rw [Format.bfloat16_maxVal_real]; exact hx.trans (by norm_num))
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at hm2
  obtain ⟨h1, h2, h3⟩ := FloatV.cvt_bf16_f32_exact hm1
  obtain ⟨t1, t2, t3⟩ := kb_tail (ε := 2 ^ (-8 : ℤ)) (δ := 2 ^ (-134 : ℤ)) hg hx le_rfl le_rfl h1
    (by rw [h3]; exact hm2) hef he
  exact ⟨hm1, t1, t2, t3⟩


/-- **A product with an exact zero is a zero**: finite operands, one of value `0`. -/
theorem mul_zero_fin {f : Format} (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (h0 : x.toRat = 0 ∨ y.toRat = 0) :
    (FloatV.mul x y).isFinite = true ∧ (FloatV.mul x y).toRat = 0 := by
  have hp : x.toRat * y.toRat = 0 := by
    rcases h0 with h | h
    · rw [h, zero_mul]
    · rw [h, mul_zero]
  unfold FloatV.mul
  rw [Format.mul_of_finite .rn hx hy]
  unfold FloatV.isFinite FloatV.toRat
  unfold FloatV.toRat at hp
  rw [hp]
  obtain ⟨h1, h2, h3⟩ := Format.roundRat_zero_spec (f := f) hf (rnd := .rn) (sat := false)
    (zs := (f.sign x.pat != f.sign y.pat))
  rw [FloatV.pat_ofPat h1]
  exact ⟨h2, h3⟩


/-- **A conversion of an exact zero is a zero.** -/
theorem cvt_zero_fin {f g : Format} (hg : g.Regular) {x : FloatV f} (hx : x.isFinite = true)
    (h0 : x.toRat = 0) :
    (FloatV.cvt x : FloatV g).isFinite = true ∧ (FloatV.cvt x : FloatV g).toRat = 0 := by
  unfold FloatV.cvt
  rw [Format.cvt_of_finite g .rn false hx]
  unfold FloatV.toRat at h0
  rw [h0]
  obtain ⟨h1, h2, h3⟩ := Format.roundRat_zero_spec (f := g) hg (rnd := .rn) (sat := false)
    (zs := (f.sign x.pat))
  unfold FloatV.isFinite FloatV.toRat
  rw [FloatV.pat_ofPat h1]
  exact ⟨h2, h3⟩

end Claims.RecomputeWU
