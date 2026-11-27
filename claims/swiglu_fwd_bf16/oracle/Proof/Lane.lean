import Claim
import TVA.Proof
import Proof.Real

/-!
# `swiglu_fwd_bf16`: one lane

`lane`: for finite `bfloat16` inputs `a`, `b` with `|zRef a b| ≤ 2^127`, every result the device
facts allow for the lane's chain (the widenings, `tl.sigmoid`'s negation, `tl.exp`, `1 + ·`, `/`,
the two products, the rounding to `bfloat16`) satisfies `ZOk a b`:

* `a ≤ -89` (`lane_under`): `tl.exp(-a)` is `+∞` (`Device.ExpF32Overflow`), `1 + ∞ = ∞`, `1/∞ = +0`
  (`Device.DivF32InfDivisor`), and every later value is a zero;
* `a > -89`, so `a ≥ -88.5` (`bf16_gap`; `lane_main`): `Device.ExpF32`, the IEEE sum,
  `Device.DivFull` and `sig_err` put the sigmoid within `2^-16` of `σ(a)`; then `a = 0` gives zeros,
  and otherwise `xs_err`, `out_err` give `zTol`.
-/

namespace Claims.SwigluFwd

open TVA

/-- The `float32` constant `1` of `tl.sigmoid`. -/
theorem one32_toRat : (FloatV.ofRat 1 : F32).toRat = 1 := by decide +kernel

theorem one32_toReal : (FloatV.ofRat 1 : F32).toReal = 1 := by
  simp [FloatV.toReal, one32_toRat]

theorem one32_fin : (FloatV.ofRat 1 : F32).isFinite = true := by decide +kernel

theorem add_one_posInf : FloatV.add (FloatV.ofRat 1 : F32) FloatV.posInf = FloatV.posInf := by
  decide +kernel

theorem div_one_posInf :
    (FloatV.ofPat (Format.binary32.zeroPat ((FloatV.ofRat 1 : F32).sign != (FloatV.posInf : F32).sign))
      : F32) = FloatV.ofPat 0 := by decide +kernel

/-- An exact `float32` product of finite operands one of which is zero is a zero. -/
theorem mul_zero_of {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (h0 : x.toReal * y.toReal = 0) (h : FloatV.NaNEq (FloatV.mul x y) r) :
    r.isFinite = true ∧ r.toReal = 0 := by
  obtain ⟨h1, h2⟩ := FloatV.mul_err_real Format.binary32_regular hx hy
    (by rw [h0, abs_zero, Format.binary32_maxVal_real]; norm_num)
  rw [h0, abs_zero, zero_mul, zero_add, sub_zero, Format.binary32_eta_real] at h2
  obtain ⟨hr1, -, hr3⟩ := h.exact_of_isFinite h1
  exact ⟨hr1, by rw [hr3]; exact f32_zero_of_small h2⟩

/-- The rounding to `bfloat16` of a `float32` zero is a zero. -/
theorem cvt_zero_of {z : F32} {o : BF16} (hz : z.isFinite = true) (h0 : z.toReal = 0)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn z) o) : o.isFinite = true ∧ o.toReal = 0 := by
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hz
    (by rw [h0, abs_zero, Format.bfloat16_maxVal_real]; norm_num)
  rw [h0, abs_zero, zero_mul, zero_add, sub_zero, Format.bfloat16_eta_real] at h2
  obtain ⟨hr1, -, hr3⟩ := h.exact_of_isFinite h1
  exact ⟨hr1, by rw [hr3]; exact bf16_zero_of_small h2⟩

/-- The `bfloat16 → float32` widening is exact. -/
theorem widen {a : BF16} {x : F32} (ha : a.isFinite = true)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn a) x) : x.isFinite = true ∧ x.toReal = a.toReal := by
  obtain ⟨h1, h2⟩ := FloatV.toRat_cvt_of_wider (g := Format.binary32) Format.bfloat16_regular
    Format.binary32_regular Format.wider_bfloat16_binary32 .rn ha
  obtain ⟨hr1, hr2, -⟩ := h.exact_of_isFinite h1
  exact ⟨hr1, by simp only [FloatV.toReal, hr2, h2]⟩

/-- **The lane when `a ≤ -89`**: the output is a zero. -/
theorem lane_under {P : Platform} (hov : Device.ExpF32Overflow P)
    (hinf : Device.DivF32InfDivisor P)
    {a b : BF16} (hle : a.toReal ≤ -89)
    {xf yf nx ex den s xs bz : F32} {o : BF16}
    (hxf : xf.isFinite = true ∧ xf.toReal = a.toReal) (hyf : yf.isFinite = true)
    (hnx : nx.isFinite = true ∧ nx.toReal = -a.toReal)
    (hex : P.math .exp .binary32 nx ex)
    (hden : FloatV.NaNEq (FloatV.add (FloatV.ofRat 1) ex) den)
    (hs : P.div .binary32 (FloatV.ofRat 1) den s)
    (hxs : FloatV.NaNEq (FloatV.mul xf s) xs)
    (hbz : FloatV.NaNEq (FloatV.mul xs yf) bz)
    (ho : FloatV.NaNEq (FloatV.cvtWith .rn bz) o) : ZOk a b o := by
  have hge : (1453635 / 16384 : ℚ) ≤ nx.toRat := by
    have h1 : ((1453635 / 16384 : ℚ) : ℝ) ≤ (nx.toRat : ℝ) := by
      show _ ≤ nx.toReal
      rw [hnx.2]; push_cast; linarith
    exact_mod_cast h1
  have hex' : ex = FloatV.posInf := hov nx (Or.inl ⟨hnx.1, hge⟩) ex hex
  subst hex'
  rw [add_one_posInf] at hden
  have hden' : den = FloatV.posInf := hden.eq (by decide)
  subst hden'
  have hs' := hinf _ _ one32_fin (by decide) s hs
  rw [div_one_posInf] at hs'
  subst hs'
  have hsfin : (FloatV.ofPat 0 : F32).isFinite = true := by decide
  obtain ⟨hxs1, hxs2⟩ := mul_zero_of hxf.1 hsfin (by simp) hxs
  obtain ⟨hbz1, hbz2⟩ := mul_zero_of hxs1 hyf (by rw [hxs2, zero_mul]) hbz
  obtain ⟨ho1, ho2⟩ := cvt_zero_of hbz1 hbz2 ho
  exact ⟨ho1, fun _ => ho2, fun h => absurd hle (not_le.mpr h)⟩

/-- **The lane when `a > -89`**: within `zTol` of `zRef`. -/
theorem lane_main {P : Platform} (hexp : Device.ExpF32 P) (hdiv : Device.DivFull P)
    {a b : BF16} (ha : a.isFinite = true)
    (hz : |zRef a.toReal b.toReal| ≤ 2 ^ 127) (hgt : -89 < a.toReal)
    {xf yf nx ex den s xs bz : F32} {o : BF16}
    (hxf : xf.isFinite = true ∧ xf.toReal = a.toReal)
    (hyf : yf.isFinite = true ∧ yf.toReal = b.toReal)
    (hnx : nx.isFinite = true ∧ nx.toReal = -a.toReal)
    (hex : P.math .exp .binary32 nx ex)
    (hden : FloatV.NaNEq (FloatV.add (FloatV.ofRat 1) ex) den)
    (hs : P.div .binary32 (FloatV.ofRat 1) den s)
    (hxs : FloatV.NaNEq (FloatV.mul xf s) xs)
    (hbz : FloatV.NaNEq (FloatV.mul xs yf) bz)
    (ho : FloatV.NaNEq (FloatV.cvtWith .rn bz) o) :
    o.isFinite = true ∧ |o.toReal - zRef a.toReal b.toReal| ≤ zTol (zRef a.toReal b.toReal) := by
  have hga := bf16_gap hgt
  set av := a.toReal with hav
  set e := Real.exp (-av) with he
  have he0 : 0 < e := Real.exp_pos _
  set u := 1 + e with hu
  have hu1 : 1 ≤ u := by linarith
  have hu2 : u ≤ 2 ^ 128 := one_add_exp_le hga
  -- `tl.exp(-a)`
  obtain ⟨ref, href, hacc⟩ := hexp |nx.toReal| (abs_nonneg _)
  have href' : ref = Real.exp := by
    simp only [MathFn.ref, Option.some.injEq] at href; exact href.symm
  subst href'
  have hroom := exp_room hga (bf16_abs_le ha)
  rw [← hnx.2] at hroom
  obtain ⟨hexfin, hE⟩ := hacc nx ex hex hnx.1 ⟨neg_abs_le _, le_abs_self _⟩ (Set.mem_univ _) hroom
  rw [hnx.2, abs_of_pos (Real.exp_pos _)] at hE
  -- `1 + E`
  have hrel := exp_rel hga
  have hEe : |ex.toReal - e| ≤ 3 * 2 ^ (-18 : ℤ) * u + 2 ^ (-140 : ℤ) := by
    rw [← he] at hE hrel; linarith
  have h1E : 0 ≤ 1 + ex.toReal ∧ 1 + ex.toReal ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]
    have h41 : e ≤ 41 / 50 * 2 ^ 128 := le_trans (Real.exp_le_exp.mpr (by linarith)) exp_885
    have hEe' := hEe
    rw [k18, k140] at hEe'
    have := abs_le.mp hEe'
    constructor <;> linarith
  obtain ⟨hd1, hd2⟩ := FloatV.add_err_real Format.binary32_regular one32_fin hexfin
    (by rw [one32_toReal, abs_of_nonneg h1E.1]; exact h1E.2)
  obtain ⟨hdenfin, -, hdenr⟩ := hden.exact_of_isFinite hd1
  rw [one32_toReal, Format.binary32_u_real, Format.binary32_eta_real, ← hdenr] at hd2
  have hD := den_err hga he hE (by rw [mul_comm] at hd2 ⊢; linarith)
  -- the division
  have hc16 : (4 / 5 : ℝ) * 2 ^ (-16 : ℤ) ≤ 1 / 50000 := by norm_num
  have hDlo : (1 - 1 / 50000) * u ≤ den.toReal := by
    have := (abs_le.mp hD).1; nlinarith
  have hDpos : 0 < den.toReal := by nlinarith
  have hDq : den.toRat ≠ 0 := by
    intro h0; have : den.toReal = 0 := by simp [FloatV.toReal, h0]
    linarith
  have hq126 : |(FloatV.ofRat 1 : F32).toRat / den.toRat| ≤ 2 ^ 126 := by
    rw [one32_toRat]
    have h1 : ((|1 / den.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast
      rw [show ((den.toRat : ℚ) : ℝ) = den.toReal from rfl, abs_of_pos (by positivity)]
      rw [div_le_iff₀ hDpos]; nlinarith
    exact_mod_cast h1
  obtain ⟨hsfin, hS⟩ := hdiv _ _ s hs one32_fin hdenfin hDq hq126
  have hS' : |s.toReal - 1 / den.toReal| ≤ 2 ^ (-22 : ℤ) * |1 / den.toReal| + 2 ^ (-148 : ℤ) := by
    rw [one32_toRat] at hS
    have h1 : ((|s.toRat - 1 / den.toRat| : ℚ) : ℝ) ≤
        (((2 : ℚ) ^ (-22 : ℤ) * |1 / den.toRat| + 2 ^ (-148 : ℤ) : ℚ) : ℝ) := by exact_mod_cast hS
    push_cast at h1
    exact h1
  have hsig := sig_err hu1 hu2 hD hS'
  -- `σ`
  set σ := 1 / u with hσ
  have hσ0 : 0 < σ := by positivity
  have hσ1 : σ ≤ 1 := by rw [hσ, div_le_one (by linarith)]; exact hu1
  have hzR : zRef av b.toReal = av * σ * b.toReal := by
    rw [zRef, hσ, hu, he]; ring
  -- the first product
  have hSle : |s.toReal| ≤ (1 + 2 ^ (-16 : ℤ)) * σ := by
    have : |s.toReal| ≤ |σ| + |s.toReal - σ| := by
      have : s.toReal = σ + (s.toReal - σ) := by ring
      conv_lhs => rw [this]
      exact abs_add_le _ _
    rw [abs_of_pos hσ0] at this; linarith
  have hxsroom : |xf.toReal * s.toReal| ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real, abs_mul, hxf.2]
    have h1 := bf16_abs_le ha
    have h2 : |s.toReal| ≤ 1 + 2 ^ (-16 : ℤ) := by
      calc |s.toReal| ≤ (1 + 2 ^ (-16 : ℤ)) * σ := hSle
        _ ≤ (1 + 2 ^ (-16 : ℤ)) * 1 := mul_le_mul_of_nonneg_left hσ1 (by positivity)
        _ = 1 + 2 ^ (-16 : ℤ) := mul_one _
    have h3 : (2 : ℝ) ^ (-16 : ℤ) = 1 / 65536 := by norm_num
    rw [h3] at h2
    nlinarith [abs_nonneg av, abs_nonneg s.toReal]
  obtain ⟨hp1, hp2⟩ := FloatV.mul_err_real Format.binary32_regular hxf.1 hsfin hxsroom
  obtain ⟨hxsfin, -, hxsr⟩ := hxs.exact_of_isFinite hp1
  rw [Format.binary32_u_real, Format.binary32_eta_real, ← hxsr, hxf.2] at hp2
  by_cases ha0 : av = 0
  · -- `a = 0`: every later value is a zero
    have hxs0 : xs.toReal = 0 := by
      have : |xs.toReal - av * s.toReal| ≤ |av * s.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := hp2
      rw [ha0, zero_mul, sub_zero, abs_zero, zero_mul, zero_add] at this
      exact f32_zero_of_small this
    obtain ⟨hbz1, hbz2⟩ := mul_zero_of hxsfin hyf.1 (by rw [hxs0, zero_mul]) hbz
    obtain ⟨ho1, ho2⟩ := cvt_zero_of hbz1 hbz2 ho
    refine ⟨ho1, ?_⟩
    rw [ho2, hzR, ha0, zero_mul, zero_mul, sub_zero, abs_zero]
    unfold zTol; positivity
  · have heta := eta_rel hga (bf16_abs_ge ha0)
    rw [← he, ← hu, ← hσ] at heta
    have hP := xs_err hσ0 hsig heta hp2
    -- the second product
    have hpb := pb_le (b := b.toReal) hP
    have hbzroom : |xs.toReal * yf.toReal| ≤ (Format.binary32.maxVal : ℝ) := by
      rw [hyf.2, Format.binary32_maxVal_real]
      have h1 : |xs.toReal * b.toReal| ≤ (1 + 2 ^ (-14 : ℤ)) * 2 ^ 127 :=
        le_trans hpb (mul_le_mul_of_nonneg_left (by rw [← hzR]; exact hz) (by positivity))
      have : (1 + (2 : ℝ) ^ (-14 : ℤ)) * 2 ^ 127 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
      linarith
    obtain ⟨hz1, hz2⟩ := FloatV.mul_err_real Format.binary32_regular hxsfin hyf.1 hbzroom
    obtain ⟨hbzfin, -, hbzr⟩ := hbz.exact_of_isFinite hz1
    rw [Format.binary32_u_real, Format.binary32_eta_real, ← hbzr, hyf.2] at hz2
    -- the rounding
    have hzl := z_le hP hz2
    have horoom : |bz.toReal| ≤ (Format.bfloat16.maxVal : ℝ) := by
      rw [Format.bfloat16_maxVal_real]
      have h1 : |bz.toReal| ≤ (1 + 2 ^ (-13 : ℤ)) * 2 ^ 127 + 2 ^ (-150 : ℤ) := by
        have := mul_le_mul_of_nonneg_left (show |av * σ * b.toReal| ≤ 2 ^ 127 by rw [← hzR]; exact hz)
          (show (0 : ℝ) ≤ 1 + 2 ^ (-13 : ℤ) by positivity)
        linarith
      have : (1 + (2 : ℝ) ^ (-13 : ℤ)) * 2 ^ 127 + 2 ^ (-150 : ℤ) ≤ (2 ^ 8 - 1) * 2 ^ 120 := by
        norm_num
      linarith
    obtain ⟨ho1, ho2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hbzfin
      horoom
    obtain ⟨hofin, -, hor⟩ := ho.exact_of_isFinite ho1
    rw [FloatV.cvtWith_rn] at hor
    rw [Format.bfloat16_u_real, Format.bfloat16_eta_real, ← hor] at ho2
    refine ⟨hofin, ?_⟩
    rw [hzR]
    exact out_err hP hz2 ho2

/-- **One lane**: every result the device facts allow for the chain satisfies `ZOk`. -/
theorem lane {P : Platform} (hexp : Device.ExpF32 P) (hov : Device.ExpF32Overflow P)
    (hdiv : Device.DivFull P) (hinf : Device.DivF32InfDivisor P)
    {a b : BF16} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hz : |zRef a.toReal b.toReal| ≤ 2 ^ 127)
    {xf yf nx ex den s xs bz : F32} {o : BF16}
    (hxf : FloatV.NaNEq (FloatV.cvtWith .rn a) xf)
    (hyf : FloatV.NaNEq (FloatV.cvtWith .rn b) yf)
    (hnx : FloatV.NaNEq (FloatV.neg xf) nx)
    (hex : P.math .exp .binary32 nx ex)
    (hden : FloatV.NaNEq (FloatV.add (FloatV.ofRat 1) ex) den)
    (hs : P.div .binary32 (FloatV.ofRat 1) den s)
    (hxs : FloatV.NaNEq (FloatV.mul xf s) xs)
    (hbz : FloatV.NaNEq (FloatV.mul xs yf) bz)
    (ho : FloatV.NaNEq (FloatV.cvtWith .rn bz) o) : ZOk a b o := by
  have hx := widen ha hxf
  have hy := widen hb hyf
  have hn : nx.isFinite = true ∧ nx.toReal = -a.toReal := by
    have h1 : (FloatV.neg xf).isFinite = true := by rw [FloatV.isFinite_neg]; exact hx.1
    obtain ⟨hr1, -, hr3⟩ := hnx.exact_of_isFinite h1
    exact ⟨hr1, by rw [hr3, FloatV.toReal_neg, hx.2]⟩
  by_cases hle : a.toReal ≤ -89
  · exact lane_under hov hinf hle hx hy.1 hn hex hden hs hxs hbz ho
  · push Not at hle
    obtain ⟨h1, h2⟩ := lane_main hexp hdiv ha hz hle hx hy hn hex hden hs hxs hbz ho
    exact ⟨h1, fun h => absurd h (not_le.mpr hle), fun _ => h2⟩

end Claims.SwigluFwd
