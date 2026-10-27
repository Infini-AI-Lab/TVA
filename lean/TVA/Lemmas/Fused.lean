import TVA.Lemmas.Num
import TVA.Lang.Ops

/-!
# Error bounds for contractible additions

`AddRel fusion x y r` (`tl.addc`, `tl.subc`, the addition after a `tl.dot` in `tl.dot_add`)
allows, with `enable_fp_fusion`, the rounding of `p + q` for any exact `p`, `q` that round to the
operands (`RoundsTo`). Bounding it needs the error of a rounding that comes out finite for *any*
rational, also one just above the largest finite value (which rounds down to it):

* `Format.roundMag_rn_err_of_finite`, `Format.roundRat_rn_err_of_finite`: in an IEEE format, a
  rounding to nearest (no saturation) that is finite is within half an ulp, `|q|·u + η`;
* `RoundsTo.err`, `RoundsTo.abs_le`: an exact `p` that rounds to a finite `x` is
  within `|p|·u + η` of it, so `|p| ≤ (|x| + η)/(1 − u)`;
* `FloatV.AddRel.err`: for finite operands with room in the format, every result `AddRel`
  allows is finite and within `3u·(|x| + |y|) + 4η` of `x + y` (for `u ≤ 1/16`);
* `FloatV.mulAdd_err`: the same for `tl.mul_add` (two IEEE operations or one `fma`).
-/

namespace TVA.Math.Float.Format

open TVA.Math.Float

variable {f : Format}

theorem infMag_lt_signBit : f.infMag < f.signBit := by
  unfold infMag signBit expMax
  rw [Nat.shiftLeft_eq, pow_add]
  exact Nat.mul_lt_mul_of_pos_right (by have := Nat.one_le_two_pow (n := f.e); omega)
    (Nat.two_pow_pos _)

theorem isFinite_infMag (hs : f.special = .ieee) : f.isFinite f.infMag = false := by
  have hE : f.expField f.infMag = f.expMax := by
    unfold expField infMag
    rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, Nat.mul_div_cancel _ (Nat.two_pow_pos _)]
    exact Nat.mod_eq_of_lt (by unfold expMax; have := Nat.one_le_two_pow (n := f.e); omega)
  have hF : f.fracField f.infMag = 0 := by
    unfold fracField infMag
    rw [Nat.shiftLeft_eq, Nat.mul_mod_left]
  unfold isFinite isInf isNaN
  rw [hs]
  simp [hE, hF]

/-- `2^emax ≤ maxVal` in an IEEE format. -/
theorem two_pow_emax_le (hs : f.special = .ieee) : (2 : ℚ) ^ f.emax ≤ f.maxVal := by
  have hms : f.maxSig = 2 ^ (f.m + 1) - 1 := by unfold maxSig; rw [hs]
  have hsig : (2 : ℚ) ^ f.m ≤ f.maxSig := by
    rw [hms]
    have h1 : 2 ^ f.m ≤ 2 ^ (f.m + 1) - 1 := by
      have : 1 ≤ 2 ^ f.m := Nat.one_le_two_pow
      rw [pow_succ]; omega
    exact_mod_cast h1
  unfold maxVal
  calc (2 : ℚ) ^ f.emax = 2 ^ f.m * 2 ^ (f.emax - f.m) := by
        rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]; congr 1; ring
    _ ≤ (f.maxSig : ℚ) * 2 ^ (f.emax - f.m) :=
        mul_le_mul_of_nonneg_right hsig (zpow_pos (by norm_num) _).le

/-- **A finite value is at most `maxVal`** in magnitude, in a regular IEEE format. -/
theorem magRat_le_maxVal (hf : f.Regular) (hs : f.special = .ieee) {x : ℕ}
    (hx : f.isFinite x = true) : f.magRat x ≤ f.maxVal := by
  have hpos : ∀ z : ℤ, (0 : ℚ) < 2 ^ z := fun z => zpow_pos (by norm_num) z
  have hEmax : f.expField x ≠ f.expMax := by
    intro h
    unfold isFinite isNaN isInf at hx
    rw [hs] at hx
    by_cases hF : f.fracField x = 0 <;> simp [h, hF] at hx
  have hElt : f.expField x < 2 ^ f.e := Nat.mod_lt _ (Nat.two_pow_pos _)
  have hMlt : f.fracField x < 2 ^ f.m := Nat.mod_lt _ (Nat.two_pow_pos _)
  have hemax : f.emax = (f.expMax : ℤ) - 1 - f.bias := by unfold emax; rw [hs]
  have h2 := two_pow_emax_le hs
  by_cases h0 : f.expField x = 0
  · -- subnormal: below `2^emin ≤ 2^emax`
    rw [magRat_sub f x h0]
    have hb : (f.bias + f.m - 1 : ℕ) = ((f.bias : ℤ) + f.m - 1) := by
      have := hf.m_pos; omega
    have hlt : (f.fracField x : ℚ) ≤ 2 ^ f.m := by exact_mod_cast hMlt.le
    have hden : (0 : ℚ) < 2 ^ (f.bias + f.m - 1) := by positivity
    have e1 : (2 : ℚ) ^ f.m / 2 ^ (f.bias + f.m - 1) = 2 ^ f.emin := by
      rw [← zpow_natCast, ← zpow_natCast, ← zpow_sub₀ (by norm_num), hb]
      unfold emin; congr 1; ring
    have e2 : (2 : ℚ) ^ f.emin ≤ 2 ^ f.emax := zpow_le_zpow_right₀ (by norm_num) hf.range
    calc (f.fracField x : ℚ) / 2 ^ (f.bias + f.m - 1) ≤ 2 ^ f.m / 2 ^ (f.bias + f.m - 1) :=
          div_le_div_of_nonneg_right hlt hden.le
      _ = 2 ^ f.emin := e1
      _ ≤ f.maxVal := e2.trans h2
  · -- normal: significand at most `maxSig`, exponent at most `emax`
    rw [magRat_normal f x h0]
    have hms : f.maxSig = 2 ^ (f.m + 1) - 1 := by unfold maxSig; rw [hs]
    have hsig : ((2 ^ f.m + f.fracField x : ℕ) : ℚ) ≤ f.maxSig := by
      rw [hms]
      have : 2 ^ f.m + f.fracField x ≤ 2 ^ (f.m + 1) - 1 := by rw [pow_succ]; omega
      exact_mod_cast this
    have hE : ((f.expField x : ℤ) - f.bias - f.m) ≤ f.emax - f.m := by
      rw [hemax]; unfold expMax at hEmax ⊢
      have : 1 ≤ 2 ^ f.e := Nat.one_le_two_pow
      omega
    unfold maxVal
    exact mul_le_mul hsig (zpow_le_zpow_right₀ (by norm_num) hE) (hpos _).le (by positivity)

/-- The half-ulp at the clamped exponent, in relative and absolute parts. -/
theorem two_pow_expOf_le {q : ℚ} (hq : 0 < q) :
    (2 : ℚ) ^ (f.expOf q - f.m - 1) ≤ q * f.u + f.eta := by
  obtain ⟨hlo, -⟩ := log2Floor_rat q hq
  have hpos : ∀ z : ℤ, (0 : ℚ) < 2 ^ z := fun z => zpow_pos (by norm_num) z
  have hEL : f.expOf q = max (log2Floor q.num.natAbs q.den) f.emin := rfl
  unfold u eta
  rcases le_total (log2Floor q.num.natAbs q.den) f.emin with hle | hle
  · rw [hEL, max_eq_right hle]
    have : 0 ≤ q * 2 ^ (-(f.m : ℤ) - 1) := mul_nonneg hq.le (hpos _).le
    linarith
  · rw [hEL, max_eq_left hle]
    have : (2 : ℚ) ^ (log2Floor q.num.natAbs q.den - f.m - 1) ≤ q * 2 ^ (-(f.m : ℤ) - 1) := by
      rw [show log2Floor q.num.natAbs q.den - (f.m : ℤ) - 1 =
          log2Floor q.num.natAbs q.den + (-(f.m : ℤ) - 1) by ring, zpow_add₀ (by norm_num)]
      exact mul_le_mul_of_nonneg_right hlo (hpos _).le
    linarith [hpos (f.emin - f.m - 1)]

/-- **A finite rounding is within half an ulp**, in an IEEE format, for every `q > 0`. Above
the largest finite value `maxVal`, rounding to nearest gives either `maxVal` (when `q` is
within half an ulp of it) or infinity; so when the signed result is finite, the bound of
`roundMag_rn_err` still holds. -/
theorem roundMag_rn_err_of_finite (hs : f.special = .ieee) (neg : Bool) {q : ℚ} (hq : 0 < q)
    (hm : 1 ≤ f.m) (hrange : f.emin ≤ f.emax)
    (hfin : f.isFinite (f.withSign neg (f.roundMag .rn false neg q)) = true) :
    f.roundMag .rn false neg q < f.signBit ∧ f.isFinite (f.roundMag .rn false neg q) = true ∧
      |f.magRat (f.roundMag .rn false neg q) - q| ≤ 2 ^ (f.expOf q - f.m - 1) := by
  by_cases hmax : q ≤ (f.maxSig : ℚ) * 2 ^ (f.emax - f.m)
  · exact f.roundMag_rn_err false neg hq hm hrange hmax
  push Not at hmax
  obtain ⟨hd, hv⟩ := f.scaled_val q hq
  have hround := f.roundMag_of_scale .rn false neg hq rfl hd hv
  have herr := roundDiv_rn_err neg (n := (f.scaled q).1) hd
  have hMle := roundDiv_scaled_le f neg q hq
  have hge := roundDiv_rn_ge neg (f.scaled q).1 (f.scaled q).2
  obtain ⟨hlo, hhi⟩ := log2Floor_rat q hq
  set n := (f.scaled q).1 with hn
  set d := (f.scaled q).2 with hd'
  set M := roundDiv .rn neg n d with hM
  set E := f.expOf q with hE
  set L := log2Floor q.num.natAbs q.den with hL
  have hEL : E = max L f.emin := rfl
  have hdq : (0 : ℚ) < d := by exact_mod_cast hd
  have hpos : ∀ z : ℤ, (0 : ℚ) < 2 ^ z := fun z => zpow_pos (by norm_num) z
  have hq' : q = (n : ℚ) / d * 2 ^ (E - f.m) := by
    rw [hv, mul_assoc, ← zpow_add₀ (by norm_num)]; simp
  have hms : f.maxSig = 2 ^ (f.m + 1) - 1 := by unfold maxSig; rw [hs]
  -- `2^emax ≤ maxVal < q`, so `⌊log₂ q⌋ ≥ emax`
  have h2emax : (2 : ℚ) ^ f.emax ≤ (f.maxSig : ℚ) * 2 ^ (f.emax - f.m) := by
    have hsig : (2 : ℚ) ^ f.m ≤ f.maxSig := by
      rw [hms]
      have h1 : 2 ^ f.m ≤ 2 ^ (f.m + 1) - 1 := by
        have : 1 ≤ 2 ^ f.m := Nat.one_le_two_pow
        rw [pow_succ]; omega
      exact_mod_cast h1
    calc (2 : ℚ) ^ f.emax = 2 ^ f.m * 2 ^ (f.emax - f.m) := by
          rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]; congr 1; ring
      _ ≤ (f.maxSig : ℚ) * 2 ^ (f.emax - f.m) := mul_le_mul_of_nonneg_right hsig (hpos _).le
  have hLge : f.emax ≤ L := by
    by_contra h
    push Not at h
    have : (2 : ℚ) ^ (L + 1) ≤ 2 ^ f.emax := zpow_le_zpow_right₀ (by norm_num) (by omega)
    linarith
  have hEeqL : E = L := by rw [hEL]; exact max_eq_left (hrange.trans hLge)
  -- overflow gives infinity, which contradicts `hfin`
  have hinf : f.roundMag .rn false neg q = f.infMag → False := by
    intro h
    rw [h, f.isFinite_withSign neg infMag_lt_signBit, isFinite_infMag hs] at hfin
    exact absurd hfin (by simp)
  have hov : f.overflow .rn false neg = f.infMag := by unfold overflow; simp [hs]
  have hover : ∀ (M' : ℕ) (E' : ℤ), f.emax < E' → f.over M' E' = true := by
    intro M' E' h; unfold Format.over; rw [hs]; simp; omega
  rcases lt_or_eq_of_le hLge with hgt | heq
  · exfalso
    apply hinf
    rw [hround]
    by_cases hc : M = 2 ^ (f.m + 1)
    · rw [hc, f.encode_carry_over .rn false neg (hover _ _ (by omega)), hov]
    · rw [f.encode_over .rn false neg hc (hover _ _ (by omega)), hov]
  · have hEe : E = f.emax := by rw [hEeqL]; exact heq.symm
    by_cases hc : M = 2 ^ (f.m + 1)
    · exfalso
      apply hinf
      rw [hround, hc, f.encode_carry_over .rn false neg (hover _ _ (by omega)), hov]
    · have hsub : M < 2 ^ f.m → E = f.emin := by
        intro hlt
        exfalso
        have h2E : (2 : ℚ) ^ E ≤ q := by rw [hEeqL]; exact hlo
        have hnd : (2 : ℚ) ^ f.m ≤ (n : ℚ) / d := by
          rw [hv]
          calc (2 : ℚ) ^ f.m = 2 ^ E * 2 ^ ((f.m : ℤ) - E) := by
                rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]; congr 1; ring
            _ ≤ q * 2 ^ ((f.m : ℤ) - E) := mul_le_mul_of_nonneg_right h2E (hpos _).le
        have hnat : 2 ^ f.m ≤ n / d := by
          rw [Nat.le_div_iff_mul_le hd]
          have := (le_div_iff₀ hdq).mp hnd
          exact_mod_cast this
        omega
      have htop : E = f.emax → M ≤ f.maxSig := fun _ => by rw [hms]; omega
      have hcarry : M = 2 ^ (f.m + 1) → E < f.emax := fun h => absurd h hc
      obtain ⟨h1, h2, h3⟩ := f.encode_val false neg hm hMle (by rw [hEe]; exact hrange) hEe.le
        hsub hcarry htop
      rw [hround]
      refine ⟨h1, h2, ?_⟩
      rw [h3]
      calc |(M : ℚ) * 2 ^ (E - f.m) - q| = |(M : ℚ) - n / d| * 2 ^ (E - f.m) := by
            rw [hq', ← sub_mul, abs_mul, abs_of_pos (hpos _)]
        _ ≤ 1 / 2 * 2 ^ (E - f.m) := mul_le_mul_of_nonneg_right herr (hpos _).le
        _ = 2 ^ (E - f.m - 1) := by rw [zpow_sub_one₀ (by norm_num)]; ring

/-- **A finite rounding of any rational is within `|q|·u + η`** (IEEE formats, no
saturation). -/
theorem roundRat_rn_err_of_finite (hf : f.Regular) (hs : f.special = .ieee) (z : Bool) {q : ℚ}
    (hfin : f.isFinite (f.roundRat .rn false z q) = true) :
    |f.toRat (f.roundRat .rn false z q) - q| ≤ |q| * f.u + f.eta := by
  have hu := u_nonneg (f := f)
  have he := eta_nonneg (f := f)
  by_cases hq : q = 0
  · subst hq
    rw [roundRat_zero, f.toRat_withSign z signBit_pos, magRat_zero]
    cases z <;> simp <;> linarith
  · rw [roundRat_ne_zero _ _ _ hq] at hfin ⊢
    have hpos : 0 < |q| := abs_pos.mpr hq
    obtain ⟨hlt, -, herr⟩ := roundMag_rn_err_of_finite hs (decide (q < 0)) hpos hf.m_pos hf.range hfin
    have hb : |(f.magRat (f.roundMag .rn false (decide (q < 0)) |q|) - |q|)| ≤ |q| * f.u + f.eta :=
      herr.trans (two_pow_expOf_le hpos)
    rw [f.toRat_withSign _ hlt]
    by_cases hneg : q < 0
    · rw [decide_eq_true hneg] at hb ⊢
      simp only [ite_true]
      rw [show ∀ a : ℚ, -a - q = -(a - |q|) from fun a => by rw [abs_of_neg hneg]; ring, abs_neg]
      exact hb
    · rw [decide_eq_false hneg] at hb ⊢
      simp only [Bool.false_eq_true, ite_false]
      rw [show ∀ a : ℚ, a - q = a - |q| from fun a => by rw [abs_of_nonneg (not_lt.mp hneg)]]
      exact hb

end TVA.Math.Float.Format

namespace TVA.FloatV

open TVA.Math.Float TVA.Math.Float.Format

variable {f : Format}

theorem isInf_false_of_isFinite {x : FloatV f} (hx : x.isFinite = true) : x.isInf = false := by
  unfold isFinite Format.isFinite at hx
  unfold isInf
  revert hx
  cases f.isInf x.pat <;> simp

/-- A finite float is at most `maxVal` in magnitude (regular IEEE formats). -/
theorem abs_toRat_le_maxVal (hf : f.Regular) (hs : f.special = .ieee) {x : FloatV f}
    (hx : x.isFinite = true) : |x.toRat| ≤ f.maxVal := by
  show |f.toRat x.pat| ≤ f.maxVal
  rw [Format.abs_toRat]
  exact Format.magRat_le_maxVal hf hs hx

/-- **An exact value that rounds to a finite `x` is within `|p|·u + η` of it** (IEEE
formats). -/
theorem _root_.TVA.RoundsTo.err (hf : f.Regular) (hs : f.special = .ieee) {p : ℚ} {x : FloatV f}
    (h : RoundsTo p x) (hx : x.isFinite = true) : |x.toRat - p| ≤ |p| * f.u + f.eta := by
  rcases h with ⟨hfin, -, hval⟩ | ⟨hinf, hpat⟩
  · rw [← hval]
    exact Format.roundRat_rn_err_of_finite hf hs false hfin
  · have : x.isInf = true := by unfold isInf; rw [← hpat]; exact hinf
    rw [isInf_false_of_isFinite hx] at this
    cases this

/-- **Its magnitude**: `|p|·(1 − u) ≤ |x| + η`. -/
theorem _root_.TVA.RoundsTo.abs_le (hf : f.Regular) (hs : f.special = .ieee) {p : ℚ} {x : FloatV f}
    (h : RoundsTo p x) (hx : x.isFinite = true) : |p| * (1 - f.u) ≤ |x.toRat| + f.eta := by
  have := RoundsTo.err hf hs h hx
  have h2 := abs_sub_abs_le_abs_sub p x.toRat
  rw [abs_sub_comm] at h2
  nlinarith

/-- `|p| ≤ 2·(|x| + η)` when `u ≤ 1/2`. -/
theorem _root_.TVA.RoundsTo.abs_le_two (hf : f.Regular) (hs : f.special = .ieee) (hu : f.u ≤ 1 / 2) {p : ℚ}
    {x : FloatV f} (h : RoundsTo p x) (hx : x.isFinite = true) : |p| ≤ 2 * (|x.toRat| + f.eta) := by
  have := RoundsTo.abs_le hf hs h hx
  nlinarith [abs_nonneg p]

/-- **Error of a contractible addition.** For finite `x`, `y` with room in the format, every
result that `AddRel` allows, with or without fusion, is finite and within
`3u·(|x| + |y|) + 4η` of `x + y` (IEEE formats with `u ≤ 1/16`: `float16`, `bfloat16`,
`float32`, `float64`). -/
theorem AddRel.err (hf : f.Regular) (hs : f.special = .ieee) (hu : f.u ≤ 1 / 16) {fusion : Bool}
    {x y r : FloatV f} (h : AddRel fusion x y r) (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hroom : 2 * (|x.toRat| + |y.toRat| + 2 * f.eta) ≤ f.maxVal) :
    r.isFinite = true ∧
      |r.toRat - (x.toRat + y.toRat)| ≤ 3 * f.u * (|x.toRat| + |y.toRat|) + 4 * f.eta := by
  have hu0 := u_nonneg (f := f)
  have he0 := eta_nonneg (f := f)
  have hax := abs_nonneg x.toRat
  have hay := abs_nonneg y.toRat
  rcases h with hieee | ⟨-, hfus⟩
  · -- the IEEE sum
    have hmax : |x.toRat + y.toRat| ≤ f.maxVal := by
      have := abs_add_le x.toRat y.toRat
      linarith
    obtain ⟨hfin, herr⟩ := add_err hf hx hy hmax
    rw [hieee.eq (isFinite_iff.mp hfin).1]
    refine ⟨hfin, herr.trans ?_⟩
    have := abs_add_le x.toRat y.toRat
    nlinarith
  · rcases hfus with ⟨p, q, hp, hq, z, hr⟩ | ⟨p, -, hinf, -⟩ | ⟨q, -, hinf, -⟩
    · have hpx := hp.err hf hs hx
      have hqy := hq.err hf hs hy
      have hp2 := hp.abs_le_two hf hs (by linarith) hx
      have hq2 := hq.abs_le_two hf hs (by linarith) hy
      have hp1 := hp.abs_le hf hs hx
      have hq1 := hq.abs_le hf hs hy
      have hpq : |p + q| ≤ |p| + |q| := abs_add_le p q
      have hmax : |p + q| ≤ f.maxVal := by linarith
      obtain ⟨hfin, herr⟩ := ofPat_roundRat_err hf false z hmax
      rw [hr.eq (isFinite_iff.mp hfin).1]
      refine ⟨hfin, ?_⟩
      -- `|p| ≤ (|x| + η)(1 + 2u)` and the same for `q`
      have hp3 : |p| ≤ (|x.toRat| + f.eta) * (1 + 2 * f.u) := by nlinarith [abs_nonneg p]
      have hq3 : |q| ≤ (|y.toRat| + f.eta) * (1 + 2 * f.u) := by nlinarith [abs_nonneg q]
      have hsplit : |(ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (x.toRat + y.toRat)| ≤
          |(ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (p + q)| +
            |x.toRat - p| + |y.toRat - q| := by
        have e : (ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (x.toRat + y.toRat) =
            ((ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (p + q)) -
              (x.toRat - p) - (y.toRat - q) := by ring
        rw [e]
        calc _ ≤ |(ofPat (f.roundRat .rn false z (p + q)) : FloatV f).toRat - (p + q) - (x.toRat - p)| +
              |y.toRat - q| := abs_sub _ _
          _ ≤ _ := by gcongr; exact abs_sub _ _
      refine hsplit.trans ?_
      have hpqu : |p + q| * f.u ≤ (|p| + |q|) * f.u := mul_le_mul_of_nonneg_right hpq hu0
      have hpu : |p| * f.u ≤ (|x.toRat| + f.eta) * (1 + 2 * f.u) * f.u :=
        mul_le_mul_of_nonneg_right hp3 hu0
      have hqu : |q| * f.u ≤ (|y.toRat| + f.eta) * (1 + 2 * f.u) * f.u :=
        mul_le_mul_of_nonneg_right hq3 hu0
      nlinarith [mul_nonneg hu0 he0, mul_nonneg (mul_nonneg hu0 hu0) he0,
        mul_nonneg (mul_nonneg hu0 hu0) (add_nonneg hax hay)]
    · rw [isInf_false_of_isFinite hy] at hinf; cases hinf
    · rw [isInf_false_of_isFinite hx] at hinf; cases hinf

/-- **Error of `tl.mul_add`** (`a * b + c`, the two IEEE operations or one `fma`) for finite
operands with room: finite and within `3u·(|a·b| + |c|) + 4η` of `a·b + c`. -/
theorem mulAdd_err (hf : f.Regular) (hu : f.u ≤ 1 / 16) {fusion : Bool} {a b c r : FloatV f}
    (h : NaNEq (add (mul a b) c) r ∨ (fusion = true ∧ NaNEq (fma a b c) r))
    (ha : a.isFinite = true) (hb : b.isFinite = true) (hc : c.isFinite = true)
    (hroom : 2 * (|a.toRat * b.toRat| + |c.toRat| + 2 * f.eta) ≤ f.maxVal) :
    r.isFinite = true ∧
      |r.toRat - (a.toRat * b.toRat + c.toRat)| ≤
        3 * f.u * (|a.toRat * b.toRat| + |c.toRat|) + 4 * f.eta := by
  have hu0 := u_nonneg (f := f)
  have he0 := eta_nonneg (f := f)
  have hab := abs_nonneg (a.toRat * b.toRat)
  have hcc := abs_nonneg c.toRat
  rcases h with h1 | ⟨-, h2⟩
  · have hmmax : |a.toRat * b.toRat| ≤ f.maxVal := by linarith
    obtain ⟨hmf, hme⟩ := mul_err hf ha hb hmmax
    have hm1 : |(mul a b).toRat| ≤ |a.toRat * b.toRat| * (1 + f.u) + f.eta := by
      have := abs_sub_abs_le_abs_sub (mul a b).toRat (a.toRat * b.toRat)
      nlinarith
    have hsmax : |(mul a b).toRat + c.toRat| ≤ f.maxVal := by
      have := abs_add_le (mul a b).toRat c.toRat
      nlinarith
    obtain ⟨hsf, hse⟩ := add_err hf hmf hc hsmax
    rw [h1.eq (isFinite_iff.mp hsf).1]
    refine ⟨hsf, ?_⟩
    have e : (add (mul a b) c).toRat - (a.toRat * b.toRat + c.toRat) =
        ((add (mul a b) c).toRat - ((mul a b).toRat + c.toRat)) +
          ((mul a b).toRat - a.toRat * b.toRat) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    have := abs_add_le (mul a b).toRat c.toRat
    have hsu : |(mul a b).toRat + c.toRat| * f.u ≤
        (|a.toRat * b.toRat| * (1 + f.u) + f.eta + |c.toRat|) * f.u :=
      mul_le_mul_of_nonneg_right (by linarith) hu0
    nlinarith [mul_nonneg hu0 he0, mul_nonneg (mul_nonneg hu0 hu0) hab]
  · have hfmax : |a.toRat * b.toRat + c.toRat| ≤ f.maxVal := by
      have := abs_add_le (a.toRat * b.toRat) c.toRat
      linarith
    obtain ⟨hff, hfe⟩ := fma_err hf ha hb hc hfmax
    rw [h2.eq (isFinite_iff.mp hff).1]
    refine ⟨hff, hfe.trans ?_⟩
    have := abs_add_le (a.toRat * b.toRat) c.toRat
    nlinarith

end TVA.FloatV
