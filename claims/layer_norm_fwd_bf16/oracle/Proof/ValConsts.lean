import Proof.TolBase

/-! The values of the three `float32` constants of the row: the divisor `D`, the `1` of the
reciprocal, and `eps` widened from `float64` (`d.eps` is a `float64` value of the claim). -/

namespace Claims.LayerNormFwd

open TVA
open Format

noncomputable section

/-- A rational within `maxVal` rounds to a pattern below `2^width`. -/
theorem vcConsts_roundRat_lt_of_le {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    {q : ℚ} (hq : |q| ≤ f.maxVal) : f.roundRat rnd sat zs q < 2 ^ f.width := by
  by_cases hq0 : q = 0
  · rw [hq0]; exact (Format.roundRat_zero_spec hf).1
  · exact (Format.roundRat_spec hf rnd sat zs hq0 hq).1

/-- A rational within `maxVal` rounds to a finite pattern. -/
theorem vcConsts_isFinite_roundRat_of_le {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    {q : ℚ} (hq : |q| ≤ f.maxVal) : f.isFinite (f.roundRat rnd sat zs q) = true := by
  by_cases hq0 : q = 0
  · rw [hq0]; exact (Format.roundRat_zero_spec hf).2.1
  · exact (Format.roundRat_spec hf rnd sat zs hq0 hq).2.1

/-- The real value of an `ofPat` pattern. -/
theorem vcConsts_toReal_ofPat {f : Format} {n : ℕ} (h : n < 2 ^ f.width) :
    (FloatV.ofPat n : FloatV f).toReal = ((f.toRat n : ℚ) : ℝ) := by
  unfold FloatV.toReal
  rw [FloatV.toRat_ofPat h]

/-- A nonzero value is not a zero pattern. -/
theorem vcConsts_isZero_eq_false {f : Format} {x : FloatV f} (h : x.toRat ≠ 0) :
    x.isZero = false := by
  cases hz : x.isZero
  · rfl
  · exact absurd (Format.toRat_eq_zero_iff.mpr hz) h

/-- A finite `float64` converts to the rounded pattern. -/
theorem vcConsts_hostF32_eq_ofPat {c : F64} (h : c.isFinite = true) :
    FloatV.hostF32 c = FloatV.ofPat (Format.binary32.roundRat .rn false c.sign c.toRat) := by
  have hnan : c.isNaN = false := not_isNaN_of_isFinite h
  unfold FloatV.hostF32 FloatV.cvt
  simp only [hnan, Bool.false_eq_true, ite_false]
  rw [Format.cvt_of_finite Format.binary32 .rn false h]
  rfl

/-- The real value of the `float32` host conversion of a finite value. -/
theorem vcConsts_hostF32_toReal {c : F64} (h : c.isFinite = true)
    (hq : |c.toRat| ≤ Format.binary32.maxVal) {n : ℕ}
    (hval : Format.binary32.toRat (Format.binary32.roundRat .rn false c.sign c.toRat)
      = (n : ℚ)) :
    (FloatV.hostF32 c).toReal = (n : ℝ) := by
  rw [vcConsts_hostF32_eq_ofPat h,
    vcConsts_toReal_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ hq), hval]
  push_cast
  ring

/-- An integer at most `2^m` below the format's range rounds to itself. -/
theorem vcConsts_toRat_roundRat_natCast {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    (hem : f.emin ≤ 0) {n : ℕ} (hn : (n : ℚ) ≤ (2 : ℚ) ^ (f.m : ℤ))
    (hmax : (n : ℚ) ≤ f.maxVal) :
    f.toRat (f.roundRat rnd sat zs (n : ℚ)) = (n : ℚ) := by
  rcases Nat.eq_zero_or_pos n with h0 | hpos
  · subst h0
    rw [Nat.cast_zero, Format.roundRat_zero, Format.toRat_setSign]
    have hmag : f.mag 0 = 0 := by unfold Format.mag; simp
    cases zs <;> simp [hmag, Format.magVal_zero]
  · have hqpos : (0 : ℚ) < (n : ℚ) := by exact_mod_cast hpos
    have hq0 : ((n : ℚ)) ≠ 0 := ne_of_gt hqpos
    have habs : |((n : ℚ))| = (n : ℚ) := abs_of_pos hqpos
    have hq : |((n : ℚ))| ≤ f.maxVal := by rw [habs]; exact hmax
    obtain ⟨-, -, hval⟩ := Format.roundRat_spec hf rnd sat zs hq0 hq
    rw [hval, ite_eq_right (not_lt.mpr (le_of_lt hqpos)), one_mul, habs]
    set k : ℤ := Int.log 2 (n : ℚ) with hkdef
    have hk0 : (0 : ℤ) ≤ k := by
      rw [hkdef]
      refine Format.le_log_of_le hqpos ?_
      rw [zpow_zero]
      exact_mod_cast (show (1 : ℕ) ≤ n from hpos)
    have hk_le : k ≤ (f.m : ℤ) := by
      rw [hkdef]
      have hlt : Int.log 2 (n : ℚ) < (f.m : ℤ) + 1 := by
        refine Format.log_lt_of_lt hqpos ?_
        calc (n : ℚ) ≤ (2 : ℚ) ^ (f.m : ℤ) := hn
          _ < (2 : ℚ) ^ ((f.m : ℤ) + 1) := by
              rw [zpow_add_one₀ (by norm_num)]
              have hp : (0 : ℚ) < (2 : ℚ) ^ (f.m : ℤ) := by positivity
              linarith
      omega
    have hQ : f.qexp (n : ℚ) = k - (f.m : ℤ) := by
      rw [Format.qexp_eq hqpos, hkdef]
      rw [max_eq_left (by omega)]
    have hsub : (0 : ℤ) ≤ (f.m : ℤ) - k := by omega
    have hK : (n : ℚ) * 2 ^ (-(f.qexp (n : ℚ))) = ((n * 2 ^ ((f.m : ℤ) - k).toNat : ℕ) : ℚ) := by
      have hpow : (2 : ℚ) ^ ((f.m : ℤ) - k) = ((2 ^ ((f.m : ℤ) - k).toNat : ℕ) : ℚ) := by
        rw [show ((f.m : ℤ) - k) = (((f.m : ℤ) - k).toNat : ℤ) from
          (Int.toNat_of_nonneg hsub).symm]
        rw [zpow_natCast]
        push_cast
        rw [Int.toNat_natCast]
      rw [hQ, neg_sub, hpow]
      push_cast
      ring
    exact Format.roundSig_exact rnd hK

/-- Rounding a positive rational at least `2^E` keeps the value at least `2^E`. -/
theorem vcConsts_pow2_le_toRat_roundRat {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    {q : ℚ} {E : ℤ} (hE : f.emin ≤ E) (hq : (2 : ℚ) ^ E ≤ q) (hmax : |q| ≤ f.maxVal) :
    (2 : ℚ) ^ E ≤ f.toRat (f.roundRat rnd sat zs q) := by
  have hqpos : (0 : ℚ) < q := lt_of_lt_of_le (by positivity) hq
  have hq0 : q ≠ 0 := ne_of_gt hqpos
  have habs : |q| = q := abs_of_pos hqpos
  obtain ⟨-, -, hval⟩ := Format.roundRat_spec hf rnd sat zs hq0 hmax
  rw [hval, ite_eq_right (not_lt.mpr (le_of_lt hqpos)), one_mul, habs]
  have hlog : E ≤ Int.log 2 q := Format.le_log_of_le hqpos hq
  have hN : (2 : ℚ) ^ (f.m : ℤ) ≤ (roundSig rnd q (f.qexp q) : ℚ) := by
    have h := Format.roundSig_ge (f := f) rnd hqpos (le_trans hE hlog)
    rw [zpow_natCast]
    exact_mod_cast h
  have hQge : E - (f.m : ℤ) ≤ f.qexp q := by
    rw [Format.qexp_eq hqpos]
    omega
  have hfac : (2 : ℚ) ^ (E - (f.m : ℤ)) ≤ (2 : ℚ) ^ (f.qexp q) :=
    zpow_le_zpow_right₀ (by norm_num) hQge
  calc (2 : ℚ) ^ E = (2 : ℚ) ^ (f.m : ℤ) * (2 : ℚ) ^ (E - (f.m : ℤ)) := by
        rw [← zpow_add₀ (by norm_num)]
        congr 1
        ring
    _ ≤ (roundSig rnd q (f.qexp q) : ℚ) * (2 : ℚ) ^ (f.qexp q) :=
        mul_le_mul hN hfac (by positivity) (by positivity)

/-- The rounded magnitude of a rational within `2^E` stays within `2^E`. -/
theorem vcConsts_abs_roundRat_le_pow2 {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    {q : ℚ} {E : ℤ} (hE : f.emin ≤ E) (hEmax : (2 : ℚ) ^ E ≤ f.maxVal)
    (hq : |q| ≤ (2 : ℚ) ^ E) :
    |f.toRat (f.roundRat rnd sat zs q)| ≤ (2 : ℚ) ^ E := by
  by_cases hq0 : q = 0
  · rw [hq0, Format.roundRat_zero, Format.toRat_setSign]
    have hmag : f.mag 0 = 0 := by unfold Format.mag; simp
    have h0 : f.magVal (f.mag 0) = 0 := by rw [hmag, Format.magVal_zero]
    have hz : |(if zs then -f.magVal (f.mag 0) else f.magVal (f.mag 0))| = 0 := by
      rw [h0]
      cases zs <;> simp
    rw [hz]
    positivity
  · have hqpos : (0 : ℚ) < |q| := abs_pos.mpr hq0
    have hmax : |q| ≤ f.maxVal := le_trans hq hEmax
    obtain ⟨-, -, hval⟩ := Format.roundRat_spec hf rnd sat zs hq0 hmax
    rw [hval]
    have hsign : |(if q < 0 then (-1 : ℚ) else 1)| = 1 := by
      split_ifs <;> norm_num
    rw [abs_mul, hsign, one_mul, abs_of_nonneg (by positivity)]
    have hQle : f.qexp |q| ≤ E - (f.m : ℤ) := by
      rw [Format.qexp_eq hqpos]
      have hlog : Int.log 2 |q| ≤ E := by
        have hlt : Int.log 2 |q| < E + 1 := by
          refine Format.log_lt_of_lt hqpos ?_
          calc |q| ≤ (2 : ℚ) ^ E := hq
            _ < (2 : ℚ) ^ (E + 1) := by
                rw [zpow_add_one₀ (by norm_num)]
                have hp : (0 : ℚ) < (2 : ℚ) ^ E := by positivity
                linarith
        omega
      omega
    by_cases hQlt : f.qexp |q| < E - (f.m : ℤ)
    · have hN : (roundSig rnd |q| (f.qexp |q|) : ℚ) ≤ (2 : ℚ) ^ ((f.m : ℤ) + 1) := by
        have hb : roundSig rnd |q| (f.qexp |q|) ≤ 2 ^ (f.m + 1) :=
          (Format.roundSig_bounds (f := f) rnd hqpos).1
        calc (roundSig rnd |q| (f.qexp |q|) : ℚ) ≤ ((2 ^ (f.m + 1) : ℕ) : ℚ) := by
              exact_mod_cast hb
          _ = (2 : ℚ) ^ ((f.m : ℤ) + 1) := by
              rw [show ((f.m : ℤ) + 1) = ((f.m + 1 : ℕ) : ℤ) from by push_cast; ring,
                zpow_natCast]
              push_cast
              ring
      have hQl : f.qexp |q| ≤ E - (f.m : ℤ) - 1 := by omega
      have hfac : (2 : ℚ) ^ (f.qexp |q|) ≤ (2 : ℚ) ^ (E - (f.m : ℤ) - 1) :=
        zpow_le_zpow_right₀ (by norm_num) hQl
      calc (roundSig rnd |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|)
          ≤ (2 : ℚ) ^ ((f.m : ℤ) + 1) * (2 : ℚ) ^ (E - (f.m : ℤ) - 1) :=
            mul_le_mul hN hfac (by positivity) (by positivity)
        _ = (2 : ℚ) ^ E := by
            rw [← zpow_add₀ (by norm_num)]
            congr 1
            ring
    · have hQeq : f.qexp |q| = E - (f.m : ℤ) := le_antisymm hQle (not_lt.mp hQlt)
      have hxle : |q| * (2 : ℚ) ^ (-(E - (f.m : ℤ))) ≤ ((2 ^ f.m : ℕ) : ℚ) := by
        have hp : (0 : ℚ) < (2 : ℚ) ^ (-(E - (f.m : ℤ))) := by positivity
        calc |q| * (2 : ℚ) ^ (-(E - (f.m : ℤ)))
            ≤ (2 : ℚ) ^ E * (2 : ℚ) ^ (-(E - (f.m : ℤ))) :=
              mul_le_mul_of_nonneg_right hq (le_of_lt hp)
          _ = ((2 ^ f.m : ℕ) : ℚ) := by
              rw [← zpow_add₀ (by norm_num)]
              rw [show E + -(E - (f.m : ℤ)) = (f.m : ℤ) from by ring, zpow_natCast]
              push_cast
              ring
      have hrnat : rnat rnd (|q| * (2 : ℚ) ^ (-(E - (f.m : ℤ)))) ≤ 2 ^ f.m :=
        rnat_le rnd (by positivity) hxle
      have hN : (roundSig rnd |q| (f.qexp |q|) : ℚ) ≤ (2 : ℚ) ^ (f.m : ℤ) := by
        rw [hQeq, Format.roundSig_eq]
        calc (rnat rnd (|q| * (2 : ℚ) ^ (-(E - (f.m : ℤ)))) : ℚ) ≤ ((2 ^ f.m : ℕ) : ℚ) := by
              exact_mod_cast hrnat
          _ = (2 : ℚ) ^ (f.m : ℤ) := by
              rw [zpow_natCast]
              push_cast
              ring
      calc (roundSig rnd |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|)
          = (roundSig rnd |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (E - (f.m : ℤ)) := by rw [hQeq]
        _ ≤ (2 : ℚ) ^ (f.m : ℤ) * (2 : ℚ) ^ (E - (f.m : ℤ)) :=
            mul_le_mul_of_nonneg_right hN (by positivity)
        _ = (2 : ℚ) ^ E := by
            rw [← zpow_add₀ (by norm_num)]
            congr 1
            ring

/-- The `float32` conversion of `eps` is finite. -/
theorem epsV_isFinite {d : Inst} (hd : d.ok) :
    (FloatV.hostF32 d.eps).isFinite = true := by
  have hqpos : (0 : ℚ) < d.eps.toRat := lt_of_lt_of_le (by norm_num) hd.eps_ge
  have hmax : |d.eps.toRat| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos hqpos]
    exact le_trans hd.eps_le (by rw [Format.binary32_maxVal]; norm_num)
  rw [vcConsts_hostF32_eq_ofPat hd.eps_fin]
  unfold FloatV.isFinite
  rw [FloatV.pat_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ hmax)]
  exact vcConsts_isFinite_roundRat_of_le Format.binary32_regular .rn false d.eps.sign hmax

/-- Widening `eps` keeps it at least `2^-125`. -/
theorem epsV_ge {d : Inst} (hd : d.ok) : (2 : ℝ) ^ (-125 : ℤ) ≤ (FloatV.hostF32 d.eps).toReal := by
  have hqpos : (0 : ℚ) < d.eps.toRat := lt_of_lt_of_le (by norm_num) hd.eps_ge
  have hmax : |d.eps.toRat| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos hqpos]
    exact le_trans hd.eps_le (by rw [Format.binary32_maxVal]; norm_num)
  have hmain := vcConsts_pow2_le_toRat_roundRat (f := Format.binary32) Format.binary32_regular
    RoundMode.rn false d.eps.sign (E := -125)
    (by rw [Format.binary32_emin]; norm_num) hd.eps_ge hmax
  have hcast : ((2 : ℚ) ^ (-125 : ℤ) : ℝ) = (2 : ℝ) ^ (-125 : ℤ) := by norm_num
  calc (2 : ℝ) ^ (-125 : ℤ) = ((2 : ℚ) ^ (-125 : ℤ) : ℝ) := hcast.symm
    _ ≤ _ := by
        rw [vcConsts_hostF32_eq_ofPat hd.eps_fin,
          vcConsts_toReal_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ hmax)]
        exact_mod_cast hmain


/-- Widening `eps` keeps it at most `2^126`. -/
theorem epsV_le {d : Inst} (hd : d.ok) : (FloatV.hostF32 d.eps).toReal ≤ (2 : ℝ) ^ (126 : ℤ) := by
  have hqpos : (0 : ℚ) < d.eps.toRat := lt_of_lt_of_le (by norm_num) hd.eps_ge
  have hq : |d.eps.toRat| ≤ (2 : ℚ) ^ (126 : ℤ) := by
    rw [abs_of_pos hqpos]
    exact hd.eps_le
  have hmax : |d.eps.toRat| ≤ Format.binary32.maxVal :=
    le_trans hq (by rw [Format.binary32_maxVal]; norm_num)
  have hmain := vcConsts_abs_roundRat_le_pow2 (f := Format.binary32) Format.binary32_regular
    RoundMode.rn false d.eps.sign (E := 126)
    (by rw [Format.binary32_emin]; norm_num)
    (by rw [Format.binary32_maxVal]; norm_num) hq
  have hle : Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false d.eps.sign d.eps.toRat)
      ≤ (2 : ℚ) ^ (126 : ℤ) := le_trans (le_abs_self _) hmain
  have hcast : ((2 : ℚ) ^ (126 : ℤ) : ℝ) = (2 : ℝ) ^ (126 : ℤ) := by norm_num
  calc (FloatV.hostF32 d.eps).toReal
      = ((Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false d.eps.sign
          d.eps.toRat) : ℚ) : ℝ) := by
        rw [vcConsts_hostF32_eq_ofPat hd.eps_fin,
          vcConsts_toReal_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ hmax)]
    _ ≤ ((2 : ℚ) ^ (126 : ℤ) : ℝ) := by exact_mod_cast hle
    _ = (2 : ℝ) ^ (126 : ℤ) := hcast

/-- The `float64 → float32` conversion of `eps` is within one rounding. -/
theorem epsV_err {d : Inst} (hd : d.ok) :
    |(FloatV.hostF32 d.eps).toReal - epsR d| ≤ u32 * epsR d + eta32 := by
  have hqpos : (0 : ℚ) < d.eps.toRat := lt_of_lt_of_le (by norm_num) hd.eps_ge
  have hmax : |d.eps.toRat| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos hqpos]
    exact le_trans hd.eps_le (by rw [Format.binary32_maxVal]; norm_num)
  have hmain := (Format.roundRat_rn_err (f := Format.binary32) Format.binary32_regular
    false d.eps.sign hmax).2.2
  rw [Format.binary32_u, Format.binary32_eta, abs_of_pos hqpos] at hmain
  have hround : (FloatV.hostF32 d.eps).toReal
      = ((Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false d.eps.sign
          d.eps.toRat) : ℚ) : ℝ) := by
    rw [vcConsts_hostF32_eq_ofPat hd.eps_fin,
      vcConsts_toReal_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ hmax)]
  have heps : epsR d = ((d.eps.toRat : ℚ) : ℝ) := rfl
  have hreal : |((Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false
        d.eps.sign d.eps.toRat) : ℚ) : ℝ) - ((d.eps.toRat : ℚ) : ℝ)|
      ≤ (2 : ℝ) ^ (-24 : ℤ) * ((d.eps.toRat : ℚ) : ℝ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have habs : ((|Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false
          d.eps.sign d.eps.toRat) - d.eps.toRat| : ℚ) : ℝ)
        = |((Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false
              d.eps.sign d.eps.toRat) : ℚ) : ℝ) - ((d.eps.toRat : ℚ) : ℝ)| := by
      rw [Rat.cast_abs]
      push_cast
      ring_nf
    rw [← habs]
    have hrhs : (((d.eps.toRat * (2 : ℚ) ^ (-24 : ℤ) + (2 : ℚ) ^ (-150 : ℤ) : ℚ)) : ℝ)
        = (2 : ℝ) ^ (-24 : ℤ) * ((d.eps.toRat : ℚ) : ℝ) + (2 : ℝ) ^ (-150 : ℤ) := by
      push_cast
      ring
    rw [← hrhs]
    exact_mod_cast hmain
  rw [hround, heps]
  unfold u32 eta32
  exact hreal

/-- Facts about the `float64` value of the divisor: finite, its value, its magnitude, nonzero. -/
theorem vcConsts_ofRat_D_facts {d : Inst} (hd : d.ok) :
    (FloatV.ofRat (d.D : ℚ) : F64).isFinite = true ∧
    (FloatV.ofRat (d.D : ℚ) : F64).toRat = (d.D : ℚ) ∧
    |(FloatV.ofRat (d.D : ℚ) : F64).toRat| ≤ Format.binary32.maxVal ∧
    (FloatV.ofRat (d.D : ℚ) : F64).isZero = false := by
  have hDpos : (0 : ℚ) < (d.D : ℚ) := by
    have h : 0 < d.D := lt_of_lt_of_le Nat.zero_lt_one hd.d_pos
    exact_mod_cast h
  have hD512 : (d.D : ℚ) ≤ 512 := by exact_mod_cast hd.d_le
  have h64max : (d.D : ℚ) ≤ Format.binary64.maxVal :=
    le_trans hD512 (le_trans (by rw [Format.binary32_maxVal]; norm_num)
      Format.wider_binary32_binary64.2.2)
  have h32max : (d.D : ℚ) ≤ Format.binary32.maxVal :=
    le_trans hD512 (by rw [Format.binary32_maxVal]; norm_num)
  have h64abs : |(d.D : ℚ)| ≤ Format.binary64.maxVal := by
    rw [abs_of_pos hDpos]; exact h64max
  have h32abs : |(d.D : ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos hDpos]; exact h32max
  have hlt64 := vcConsts_roundRat_lt_of_le (f := Format.binary64) Format.binary64_regular
    RoundMode.rn false false h64abs
  have hem64 : Format.binary64.emin ≤ 0 := by
    unfold Format.emin Format.bias Format.binary64; norm_num
  have htoRat : (FloatV.ofRat (d.D : ℚ) : F64).toRat = (d.D : ℚ) := by
    unfold FloatV.ofRat FloatV.toRat
    rw [FloatV.pat_ofPat hlt64]
    exact vcConsts_toRat_roundRat_natCast (f := Format.binary64) Format.binary64_regular
      RoundMode.rn false false hem64 (le_trans hD512 (by norm_num [Format.binary64])) h64max
  refine ⟨?_, htoRat, ?_, ?_⟩
  · unfold FloatV.ofRat FloatV.isFinite
    rw [FloatV.pat_ofPat hlt64]
    exact vcConsts_isFinite_roundRat_of_le Format.binary64_regular .rn false false h64abs
  · rw [htoRat]; exact h32abs
  · exact vcConsts_isZero_eq_false (by rw [htoRat]; exact ne_of_gt hDpos)

/-- The divisor's value is `D`. -/
theorem dv_toReal {d : Inst} (hd : d.ok) :
    (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))).toReal = (d.D : ℝ) := by
  obtain ⟨hc_fin, hc_toRat, hc_abs, hc_zero⟩ := vcConsts_ofRat_D_facts hd
  have hDpos : (0 : ℚ) < (d.D : ℚ) := by
    have h : 0 < d.D := lt_of_lt_of_le Nat.zero_lt_one hd.d_pos
    exact_mod_cast h
  have hD512 : (d.D : ℚ) ≤ 512 := by exact_mod_cast hd.d_le
  have h32max : (d.D : ℚ) ≤ Format.binary32.maxVal :=
    le_trans hD512 (by rw [Format.binary32_maxVal]; norm_num)
  have hval : Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false
      (FloatV.ofRat (d.D : ℚ) : F64).sign (FloatV.ofRat (d.D : ℚ) : F64).toRat)
      = (d.D : ℚ) := by
    rw [hc_toRat]
    exact vcConsts_toRat_roundRat_natCast (f := Format.binary32) Format.binary32_regular
      RoundMode.rn false _ (by rw [Format.binary32_emin]; norm_num)
      (le_trans hD512 (by norm_num [Format.binary32])) h32max
  unfold FloatV.litF32
  simp only [hc_zero, Bool.false_eq_true, ite_false]
  exact vcConsts_hostF32_toReal hc_fin hc_abs hval

/-- The divisor's value is finite. -/
theorem dv_isFinite {d : Inst} (hd : d.ok) :
    (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))).isFinite = true := by
  obtain ⟨hc_fin, hc_toRat, hc_abs, hc_zero⟩ := vcConsts_ofRat_D_facts hd
  unfold FloatV.litF32
  simp only [hc_zero, Bool.false_eq_true, ite_false]
  rw [vcConsts_hostF32_eq_ofPat hc_fin]
  unfold FloatV.isFinite
  rw [FloatV.pat_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ hc_abs)]
  exact vcConsts_isFinite_roundRat_of_le Format.binary32_regular .rn false _ hc_abs

/-- The `1` of the reciprocal is `1`. -/
theorem one_toReal :
    (FloatV.litF32 (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64)).toReal = 1 := by
  have htoRat : (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).toRat = 1 := by decide +kernel
  have hfin : (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).isFinite = true := by decide +kernel
  have hzero : (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).isZero = false := by decide +kernel
  have habs : |(FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).toRat| ≤ Format.binary32.maxVal := by
    rw [htoRat, abs_one]
    rw [Format.binary32_maxVal]; norm_num
  have hval : Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false
      (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).sign
      (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).toRat) = (1 : ℚ) := by
    rw [htoRat]
    exact vcConsts_toRat_roundRat_natCast (f := Format.binary32) Format.binary32_regular
      RoundMode.rn false _ (by rw [Format.binary32_emin]; norm_num)
      (by norm_num [Format.binary32]) (by rw [Format.binary32_maxVal]; norm_num)
  unfold FloatV.litF32
  simp only [hzero, Bool.false_eq_true, ite_false]
  simpa using vcConsts_hostF32_toReal hfin habs hval

/-- The `1` of the reciprocal is finite. -/
theorem one_isFinite :
    (FloatV.litF32 (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64)).isFinite = true := by
  have htoRat : (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).toRat = 1 := by decide +kernel
  have hfin : (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).isFinite = true := by decide +kernel
  have hzero : (FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).isZero = false := by decide +kernel
  have habs : |(FloatV.ofPat (0x3FF0000000000000 : ℕ) : F64).toRat| ≤ Format.binary32.maxVal := by
    rw [htoRat, abs_one]
    rw [Format.binary32_maxVal]; norm_num
  unfold FloatV.litF32
  simp only [hzero, Bool.false_eq_true, ite_false]
  rw [vcConsts_hostF32_eq_ofPat hfin]
  unfold FloatV.isFinite
  rw [FloatV.pat_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ habs)]
  exact vcConsts_isFinite_roundRat_of_le Format.binary32_regular .rn false _ habs

end

end Claims.LayerNormFwd
