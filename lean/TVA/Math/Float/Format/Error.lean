import TVA.Math.Float.Format.Exact
import TVA.Math.Float.Format.Sat
import Mathlib

/-!
# Rounding to nearest: half an ulp, for every format

* `roundDiv_rn_err`: the integer rounding is within `1/2` of the quotient;
* `encode_val`: an encoded significand that does not overflow is a finite pattern of value
  `M · 2^(E − m)`; helpers `emax_ieee`, `emax_nonieee`, `over_false`, `isFinite_of`;
* **`roundMag_rn_err`**: for `0 < q ≤ maxSig · 2^(emax − m)` (the largest finite value, `maxSig`),
  rounding to nearest gives a finite magnitude within `2^(expOf q − m − 1)` of `q`;
  `roundMag_rn_err_rel`: within `2^-(m+1)·q + 2^(emin − m − 1)`;
* **`cvt_rn_err`**: a conversion to nearest of a finite value in range is finite and within
  `2^-(m+1)·|v| + 2^(emin − m − 1)`; instances `cvt_f16_rn_err` (`|v| ≤ 65504`: `2^-11`, `2^-25`),
  `cvt_bf16_rn_err` (`2^-8`, `2^-134`), `cvt_e4m3_rn_err` (`|v| ≤ 448`: `2^-4`, `2^-10`),
  `cvt_e5m2_rn_err` (`|v| ≤ 57344`: `2^-3`, `2^-17`), with the largest values `binary16_max`,
  `bfloat16_max`, `e4m3_max`, `e5m2_max`; `magRat_pos`, `abs_toRat`.

These hold with or without saturation (in range nothing saturates). Proof module (Mathlib).
-/

set_option autoImplicit false

namespace TVA.Math.Float.Format

open TVA.Math.Float

/-- Round to nearest is within half a unit of the exact quotient. -/
theorem roundDiv_rn_err (neg : Bool) {n d : ℕ} (hd : 0 < d) :
    |((roundDiv .rn neg n d : ℕ) : ℚ) - (n : ℚ) / d| ≤ 1 / 2 := by
  have hdq : (0 : ℚ) < d := by exact_mod_cast hd
  have hn : (n : ℚ) / d = ((n / d : ℕ) : ℚ) + ((n % d : ℕ) : ℚ) / d := by rw [mod_div_rat hd]; ring
  have hr : n % d < d := Nat.mod_lt _ hd
  set t : ℚ := ((n % d : ℕ) : ℚ) / d with ht
  have ht0 : 0 ≤ t := by positivity
  have ht1 : t < 1 := by rw [ht, div_lt_one hdq]; exact_mod_cast hr
  have key : ∀ c : ℕ, (2 * (n % d) > d ↔ 1 / 2 < t) ∧ (2 * (n % d) < d ↔ t < 1 / 2) := by
    intro _
    constructor
    · rw [ht, lt_div_iff₀ hdq]
      constructor
      · intro h; have : (d : ℚ) < 2 * ((n % d : ℕ) : ℚ) := by exact_mod_cast h
        linarith
      · intro h; have : (d : ℚ) < 2 * ((n % d : ℕ) : ℚ) := by linarith
        exact_mod_cast this
    · rw [ht, div_lt_iff₀ hdq]
      constructor
      · intro h; have : 2 * ((n % d : ℕ) : ℚ) < d := by exact_mod_cast h
        linarith
      · intro h; have : 2 * ((n % d : ℕ) : ℚ) < d := by linarith
        exact_mod_cast this
  obtain ⟨hgt, hlt⟩ := key 0
  unfold roundDiv
  simp only
  rw [hn]
  split_ifs with h1 h2 h3
  · have := hgt.mp h1; push_cast
    rw [abs_le]; constructor <;> linarith
  · have := hlt.mp h2
    rw [abs_le]; constructor <;> linarith
  · have e : t = 1 / 2 := by
      have a := not_lt.mp (mt hgt.mpr h1); have b := not_lt.mp (mt hlt.mpr h2); linarith
    rw [e]; norm_num
  · have e : t = 1 / 2 := by
      have a := not_lt.mp (mt hgt.mpr h1); have b := not_lt.mp (mt hlt.mpr h2); linarith
    push_cast; rw [e]; norm_num


variable (f : Format)

/-- The largest significand of a finite value: `2^(m+1) − 1`, or `2^(m+1) − 2` for an `fn` format
(whose all-ones pattern at the top exponent is the NaN). The largest finite value is
`maxSig · 2^(emax − m)`. -/
def maxSig : ℕ :=
  match f.special with
  | .fn => 2 ^ (f.m + 1) - 2
  | _ => 2 ^ (f.m + 1) - 1

/-- Round to nearest is at least the floor. -/
theorem roundDiv_rn_ge (neg : Bool) (n d : ℕ) : n / d ≤ roundDiv .rn neg n d := by
  unfold roundDiv; simp only; split_ifs <;> omega

theorem emax_ieee (hs : f.special = .ieee) : f.emax = (f.expMax : ℤ) - 1 - f.bias := by
  unfold emax; rw [hs]

theorem emax_nonieee (hs : f.special ≠ .ieee) : f.emax = (f.expMax : ℤ) - f.bias := by
  unfold emax; cases h : f.special <;> simp_all

theorem over_false {M : ℕ} {E : ℤ} (hE : E ≤ f.emax) (hfn : f.special = .fn → E = f.emax → M ≠ 2 ^ (f.m + 1) - 1) :
    f.over M E = false := by
  unfold Format.over
  cases hs : f.special
  · simp; omega
  · have := hfn hs
    simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not, Bool.and_eq_false_iff, beq_eq_false_iff_ne]
    refine ⟨by omega, ?_⟩
    by_cases h : E = f.emax
    · right; exact this h
    · left; exact h
  · simp; omega

/-- A pattern is finite when its exponent field is below the all-ones field, or, in an `fn`
format, when its fraction is not all ones, or in a `finite` format. -/
theorem isFinite_of (x : ℕ) (h : f.expField x < f.expMax ∨ (f.special = .fn ∧ f.fracField x ≠ f.fracMax) ∨
    f.special = .finite) : f.isFinite x = true := by
  unfold isFinite isNaN isInf
  rcases h with hx | ⟨hs, hx⟩ | hs
  · cases f.special <;> simp [Nat.ne_of_lt hx]
  · rw [hs]; simp [hx]
  · rw [hs]; simp

/-- The value of an encoded significand, when nothing overflows: `encode` of `M ≤ 2^(m+1)` at an
exponent `E` in range is a finite pattern below the sign bit of value `M · 2^(E − m)`. -/
theorem encode_val (sat neg : Bool) {M : ℕ} {E : ℤ} (hm : 1 ≤ f.m) (hM : M ≤ 2 ^ (f.m + 1))
    (hE1 : f.emin ≤ E) (hE2 : E ≤ f.emax) (hsub : M < 2 ^ f.m → E = f.emin)
    (hcarry : M = 2 ^ (f.m + 1) → E < f.emax) (htop : E = f.emax → M ≤ f.maxSig) :
    f.encode .rn sat neg M E < f.signBit ∧ f.isFinite (f.encode .rn sat neg M E) = true ∧
      f.magRat (f.encode .rn sat neg M E) = (M : ℚ) * 2 ^ (E - f.m) := by
  have hexpMax : f.expMax < 2 ^ f.e := by unfold expMax; have := Nat.two_pow_pos f.e; omega
  have hemaxle : f.emax ≤ (f.expMax : ℤ) - f.bias := by
    by_cases hs : f.special = .ieee
    · rw [f.emax_ieee hs]; omega
    · rw [f.emax_nonieee hs]
  have hpm : 2 ≤ 2 ^ f.m := by
    calc 2 = 2 ^ 1 := rfl
      _ ≤ 2 ^ f.m := Nat.pow_le_pow_right (by norm_num) hm
  have hpm1 : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  have hsb : 2 ^ f.m ≤ f.signBit := by
    unfold signBit; exact Nat.pow_le_pow_right (by norm_num) (by omega)
  have hfracMax : 1 ≤ f.fracMax := by unfold fracMax; omega
  have hmaxSig : f.special = .fn → f.maxSig = 2 ^ (f.m + 1) - 2 := by intro hs; unfold maxSig; rw [hs]
  -- the exponent field of a normal pattern at `E'` is below the all-ones field unless `E' = emax`
  -- in a non-IEEE format
  have hemin : f.emin = 1 - (f.bias : ℤ) := rfl
  have hexp_lt : ∀ E' : ℤ, f.emin ≤ E' → E' ≤ f.emax → (f.special = .ieee ∨ E' < f.emax) →
      (E' + f.bias).toNat < f.expMax := by
    intro E' h0 h1 h2
    rcases h2 with hs | hlt
    · rw [f.emax_ieee hs] at h1; omega
    · omega
  by_cases hc : M = 2 ^ (f.m + 1)
  · -- the carry: `2^m` at `E + 1`
    have hlt := hcarry hc
    have hov : f.over (2 ^ f.m) (E + 1) = false :=
      f.over_false (by omega) (fun _ _ => by omega)
    have hT : (E + 1 + f.bias).toNat < 2 ^ f.e := by omega
    rw [hc, f.encode_carry .rn sat neg hov]
    have hpat : (E + 1 + f.bias).toNat <<< f.m = f.normalPat (2 ^ f.m) (E + 1) := by
      unfold normalPat; simp
    rw [hpat]
    obtain ⟨he, hf, hlt'⟩ := f.normalPat_fields (le_refl _) (by omega) hT
    refine ⟨hlt', f.isFinite_of _ ?_, ?_⟩
    · by_cases hs : f.special = .finite
      · exact Or.inr (Or.inr hs)
      · by_cases hE : E + 1 < f.emax
        · left; rw [he]; exact hexp_lt (E + 1) (by omega) (by omega) (Or.inr hE)
        · cases hs' : f.special
          · left; rw [he]; exact hexp_lt (E + 1) (by omega) (by omega) (Or.inl hs')
          · right; left; exact ⟨rfl, by rw [hf]; omega⟩
          · exact absurd hs' hs
    · rw [f.magRat_normalPat (le_refl _) (by omega) (by omega) hT]
      push_cast
      rw [← zpow_natCast, ← zpow_natCast, ← zpow_add₀ (by norm_num), ← zpow_add₀ (by norm_num)]
      congr 1; push_cast; ring
  by_cases hn : 2 ^ f.m ≤ M
  · -- a normal significand
    have hM2 : M < 2 ^ (f.m + 1) := lt_of_le_of_ne hM hc
    have hov : f.over M E = false := by
      refine f.over_false hE2 (fun hs hE => ?_)
      have := htop hE; rw [hmaxSig hs] at this; omega
    have hT : (E + f.bias).toNat < 2 ^ f.e := by omega
    rw [f.encode_mid .rn sat neg hn hM2 hov]
    obtain ⟨he, hf, hlt'⟩ := f.normalPat_fields hn hM2 hT
    unfold normalPat at he hf hlt'
    refine ⟨hlt', f.isFinite_of _ ?_, ?_⟩
    · by_cases hs : f.special = .finite
      · exact Or.inr (Or.inr hs)
      · by_cases hE : E < f.emax
        · left; rw [he]; exact hexp_lt E hE1 hE2 (Or.inr hE)
        · cases hs' : f.special
          · left; rw [he]; exact hexp_lt E hE1 hE2 (Or.inl hs')
          · right; left; refine ⟨rfl, ?_⟩; rw [hf]
            have := htop (by omega); rw [hmaxSig hs'] at this; unfold fracMax; omega
          · exact absurd hs' hs
    · exact f.magRat_normalPat hn hM2 hE1 hT
  · -- a subnormal significand (or zero) at `emin`
    have hM0 : M < 2 ^ f.m := by omega
    have hEe := hsub hM0
    have hov : f.over M E = false := by
      refine f.over_false hE2 (fun hs hE => ?_)
      omega
    rw [f.encode_lt .rn sat neg hM0 hov]
    refine ⟨lt_of_lt_of_le hM0 hsb, f.isFinite_of _ ?_, ?_⟩
    · have he0 : f.expField M = 0 := by rw [expField_eq, Nat.div_eq_of_lt hM0, Nat.zero_mod]
      by_cases hs : f.special = .finite
      · exact Or.inr (Or.inr hs)
      · left; rw [he0]
        have : 1 ≤ f.expMax := by
          by_cases hi : f.special = .ieee
          · rw [f.emax_ieee hi] at hE2; unfold emin at hE1; omega
          · rw [f.emax_nonieee hi] at hE2; unfold emin at hE1; omega
        omega
    · rcases Nat.eq_zero_or_pos M with h0 | hpos
      · subst h0; unfold magRat expField fracField; simp
      · rw [(f.magRat_subPat hpos hM0).2, hEe]

/-- **Rounding to nearest is within half a unit in the last place.** For `0 < q` no larger than the
largest finite value `maxSig · 2^(emax − m)`, rounding to nearest (with or without saturation) gives
a finite magnitude pattern whose value is within `2^(E − m − 1)` of `q`, where `E = expOf q` is
the clamped exponent (`⌊log₂ q⌋`, or `emin` for a subnormal). -/
theorem roundMag_rn_err (sat neg : Bool) {q : ℚ} (hq : 0 < q) (hm : 1 ≤ f.m) (hrange : f.emin ≤ f.emax)
    (hmax : q ≤ (f.maxSig : ℚ) * 2 ^ (f.emax - f.m)) :
    f.roundMag .rn sat neg q < f.signBit ∧ f.isFinite (f.roundMag .rn sat neg q) = true ∧
      |f.magRat (f.roundMag .rn sat neg q) - q| ≤ 2 ^ (f.expOf q - f.m - 1) := by
  obtain ⟨hd, hv⟩ := f.scaled_val q hq
  have hround := f.roundMag_of_scale .rn sat neg hq rfl hd hv
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
  -- the value and the scale
  have hq' : q = (n : ℚ) / d * 2 ^ (E - f.m) := by
    rw [hv, mul_assoc, ← zpow_add₀ (by norm_num)]; simp
  -- `maxSig < 2^(m+1)`
  have hms : f.maxSig < 2 ^ (f.m + 1) := by
    have : 1 ≤ 2 ^ (f.m + 1) := Nat.one_le_two_pow
    unfold maxSig; split <;> omega
  -- `E ≤ emax`
  have hqlt : q < 2 ^ (f.emax + 1) := by
    calc q ≤ (f.maxSig : ℚ) * 2 ^ (f.emax - f.m) := hmax
      _ < ((2 ^ (f.m + 1) : ℕ) : ℚ) * 2 ^ (f.emax - f.m) :=
          mul_lt_mul_of_pos_right (by exact_mod_cast hms) (hpos _)
      _ = 2 ^ (f.emax + 1) := by
          push_cast; rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]; congr 1; push_cast; ring
  have hLle : L ≤ f.emax := by
    have : (2 : ℚ) ^ L < 2 ^ (f.emax + 1) := lt_of_le_of_lt hlo hqlt
    have := (zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℚ) < 2)).mp this
    omega
  have hE2 : E ≤ f.emax := by rw [hEL]; exact max_le hLle hrange
  have hE1 : f.emin ≤ E := le_max_right _ _
  -- below `2^m` only at `emin`
  have hsub : M < 2 ^ f.m → E = f.emin := by
    intro hlt
    by_contra hne
    have hEL' : E = L := by rw [hEL] at hne ⊢; exact max_eq_left (by omega)
    have h2E : (2 : ℚ) ^ E ≤ q := by rw [hEL']; exact hlo
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
  -- at the top exponent, at most `maxSig`
  have htop : E = f.emax → M ≤ f.maxSig := by
    intro hEe
    have hnd : (n : ℚ) / d ≤ f.maxSig := by
      rw [hv, hEe]
      calc q * 2 ^ ((f.m : ℤ) - f.emax) ≤ (f.maxSig : ℚ) * 2 ^ (f.emax - f.m) * 2 ^ ((f.m : ℤ) - f.emax) :=
            mul_le_mul_of_nonneg_right hmax (hpos _).le
        _ = f.maxSig := by rw [mul_assoc, ← zpow_add₀ (by norm_num)]; simp
    by_contra hgt
    have : ((f.maxSig + 1 : ℕ) : ℚ) ≤ M := by exact_mod_cast (by omega : f.maxSig + 1 ≤ M)
    rw [abs_le] at herr
    push_cast at this
    linarith [herr.2]
  have hcarry : M = 2 ^ (f.m + 1) → E < f.emax := by
    intro hc
    rcases lt_or_eq_of_le hE2 with h | h
    · exact h
    · have := htop h; omega
  obtain ⟨h1, h2, h3⟩ := f.encode_val sat neg hm hMle hE1 hE2 hsub hcarry htop
  rw [hround]
  refine ⟨h1, h2, ?_⟩
  rw [h3]
  calc |(M : ℚ) * 2 ^ (E - f.m) - q| = |(M : ℚ) - n / d| * 2 ^ (E - f.m) := by
        rw [hq', ← sub_mul, abs_mul, abs_of_pos (hpos _)]
    _ ≤ 1 / 2 * 2 ^ (E - f.m) := mul_le_mul_of_nonneg_right herr (hpos _).le
    _ = 2 ^ (E - f.m - 1) := by rw [zpow_sub_one₀ (by norm_num)]; ring

/-- The half-ulp bound in relative and absolute parts: `|round(q) − q| ≤ 2^-(m+1)·q + 2^(emin−m−1)`
(the second term only for subnormal results). -/
theorem roundMag_rn_err_rel (sat neg : Bool) {q : ℚ} (hq : 0 < q) (hm : 1 ≤ f.m) (hrange : f.emin ≤ f.emax)
    (hmax : q ≤ (f.maxSig : ℚ) * 2 ^ (f.emax - f.m)) :
    |f.magRat (f.roundMag .rn sat neg q) - q| ≤ q * 2 ^ (-(f.m : ℤ) - 1) + 2 ^ (f.emin - f.m - 1) := by
  have h := (f.roundMag_rn_err sat neg hq hm hrange hmax).2.2
  obtain ⟨hlo, -⟩ := log2Floor_rat q hq
  have hpos : ∀ z : ℤ, (0 : ℚ) < 2 ^ z := fun z => zpow_pos (by norm_num) z
  have hEL : f.expOf q = max (log2Floor q.num.natAbs q.den) f.emin := rfl
  refine le_trans h ?_
  rcases le_total (log2Floor q.num.natAbs q.den) f.emin with hle | hle
  · rw [hEL, max_eq_right hle]
    have : 0 ≤ q * 2 ^ (-(f.m : ℤ) - 1) := mul_nonneg hq.le (hpos _).le
    linarith
  · rw [hEL, max_eq_left hle]
    have : (2 : ℚ) ^ (log2Floor q.num.natAbs q.den - f.m - 1) ≤ q * 2 ^ (-(f.m : ℤ) - 1) := by
      rw [show log2Floor q.num.natAbs q.den - (f.m : ℤ) - 1 = log2Floor q.num.natAbs q.den + (-(f.m : ℤ) - 1) by ring,
        zpow_add₀ (by norm_num)]
      exact mul_le_mul_of_nonneg_right hlo (hpos _).le
    linarith [hpos (f.emin - f.m - 1)]

theorem magRat_nonneg (x : ℕ) : 0 ≤ f.magRat x := by
  unfold magRat; simp only
  split_ifs <;> first | positivity | exact Rat.mkRat_nonneg (by positivity) _

/-- A finite pattern that is not a zero has a positive magnitude. -/
theorem magRat_pos (x : ℕ) (hz : f.isZero x = false) : 0 < f.magRat x := by
  by_cases hE : f.expField x = 0
  · have hF : f.fracField x ≠ 0 := by intro hF; unfold isZero at hz; simp [hE, hF] at hz
    rw [f.magRat_sub x hE]
    have : (0 : ℚ) < f.fracField x := by exact_mod_cast Nat.pos_of_ne_zero hF
    positivity
  · rw [f.magRat_normal x hE]; positivity

theorem abs_toRat (x : ℕ) : |f.toRat x| = f.magRat x := by
  unfold toRat; split_ifs <;> simp [abs_of_nonneg (f.magRat_nonneg x)]

/-- **Conversion to nearest is within half an ulp.** A finite `src` value whose magnitude is at most
`dst`'s largest finite value converts (round to nearest, saturating or not) to a finite `dst`
value within `2^-(m+1)·|v| + 2^(emin−m−1)` of it (`m`, `emin` of `dst`): e.g. `f32 → f16`
(`__float2half_rn`), `f32 → bf16`, `f32 → e4m3` with `|v| ≤ 448`, `f16 → e4m3`. -/
theorem cvt_rn_err (src dst : Format) (sat : Bool) (nan : NaNRule) (x : ℕ) (hx : src.isFinite x = true)
    (hm : 1 ≤ dst.m) (hrange : dst.emin ≤ dst.emax)
    (hmax : |src.toRat x| ≤ (dst.maxSig : ℚ) * 2 ^ (dst.emax - dst.m)) :
    dst.isFinite (cvt src dst .rn sat nan x) = true ∧
      |dst.toRat (cvt src dst .rn sat nan x) - src.toRat x| ≤
        |src.toRat x| * 2 ^ (-(dst.m : ℤ) - 1) + 2 ^ (dst.emin - dst.m - 1) := by
  have hnan : src.isNaN x = false := by
    unfold isFinite at hx; simp only [Bool.and_eq_true, Bool.not_eq_true'] at hx; exact hx.1
  have hinf : src.isInf x = false := by
    unfold isFinite at hx; simp only [Bool.and_eq_true, Bool.not_eq_true'] at hx; exact hx.2
  have hpos : ∀ z : ℤ, (0 : ℚ) < 2 ^ z := fun z => zpow_pos (by norm_num) z
  have hsb : 0 < dst.signBit := by unfold signBit; positivity
  by_cases hz : src.isZero x = true
  · rw [cvt_zero src dst .rn sat nan hnan hinf hz]
    have h0 : dst.magRat 0 = 0 := by unfold magRat expField fracField; simp
    have hsz : src.magRat x = 0 := by
      unfold isZero at hz; simp only [Bool.and_eq_true, beq_iff_eq] at hz
      unfold magRat; simp [hz.1, hz.2]
    have hexpMax : 1 ≤ dst.expMax := by
      by_cases hi : dst.special = .ieee
      · have := dst.emax_ieee hi; unfold emin at hrange; omega
      · have := dst.emax_nonieee hi; unfold emin at hrange; omega
    refine ⟨?_, ?_⟩
    · rw [dst.isFinite_withSign _ hsb]
      refine dst.isFinite_of 0 (Or.inl ?_)
      unfold expField; simp; omega
    · rw [dst.toRat_withSign _ hsb, h0]
      unfold toRat; rw [hsz]
      have h1 : (0 : ℚ) ≤ |src.toRat x| * 2 ^ (-(dst.m : ℤ) - 1) := mul_nonneg (abs_nonneg _) (hpos _).le
      have h2 := hpos (dst.emin - dst.m - 1)
      split_ifs <;> simp <;> linarith
  · have hz' : src.isZero x = false := by simpa using hz
    rw [cvt_finite src dst .rn sat nan hnan hinf hz']
    have hq := src.magRat_pos x hz'
    have habs := src.abs_toRat x
    rw [habs] at hmax
    obtain ⟨hlt, hfin, -⟩ := dst.roundMag_rn_err sat (src.sign x) hq hm hrange hmax
    have herr := dst.roundMag_rn_err_rel sat (src.sign x) hq hm hrange hmax
    refine ⟨by rw [dst.isFinite_withSign _ hlt]; exact hfin, ?_⟩
    rw [dst.toRat_withSign _ hlt, habs]
    unfold toRat
    split_ifs
    · rw [show ∀ a b : ℚ, -a - -b = -(a - b) from fun a b => by ring, abs_neg]
      exact herr
    · exact herr

/-! ## Instances: the largest values and the error bounds of the common conversions -/

theorem binary16_max : (binary16.maxSig : ℚ) * 2 ^ (binary16.emax - binary16.m) = 65504 := by
  norm_num [maxSig, emax, expMax, bias, binary16]

theorem bfloat16_max : (bfloat16.maxSig : ℚ) * 2 ^ (bfloat16.emax - bfloat16.m) = 255 * 2 ^ 120 := by
  norm_num [maxSig, emax, expMax, bias, bfloat16]

theorem e4m3_max : (e4m3.maxSig : ℚ) * 2 ^ (e4m3.emax - e4m3.m) = 448 := by
  norm_num [maxSig, emax, expMax, bias, e4m3]

theorem e5m2_max : (e5m2.maxSig : ℚ) * 2 ^ (e5m2.emax - e5m2.m) = 57344 := by
  norm_num [maxSig, emax, expMax, bias, e5m2]

/-- Rounding into f16 to nearest (`__float2half_rn`, `cvt.rn.f16.f32`; from any finite source of
magnitude at most 65504): finite, within `2^-11·|v| + 2^-25`. -/
theorem cvt_f16_rn_err (src : Format) (sat : Bool) (nan : NaNRule) (x : ℕ) (hx : src.isFinite x = true)
    (hmax : |src.toRat x| ≤ 65504) :
    binary16.isFinite (cvt src binary16 .rn sat nan x) = true ∧
      |binary16.toRat (cvt src binary16 .rn sat nan x) - src.toRat x| ≤ |src.toRat x| * 2 ^ (-11 : ℤ) + 2 ^ (-25 : ℤ) := by
  have h := cvt_rn_err src binary16 sat nan x hx (by decide) (by decide) (by rw [binary16_max]; exact hmax)
  have e1 : (-(binary16.m : ℤ) - 1) = -11 := by decide
  have e2 : binary16.emin - binary16.m - 1 = -25 := by decide
  rwa [e1, e2] at h

/-- Rounding into bf16 to nearest (`__float2bfloat16_rn`): finite, within `2^-8·|v| + 2^-134`, for
`|v| ≤ 255·2^120`. -/
theorem cvt_bf16_rn_err (src : Format) (sat : Bool) (nan : NaNRule) (x : ℕ) (hx : src.isFinite x = true)
    (hmax : |src.toRat x| ≤ 255 * 2 ^ 120) :
    bfloat16.isFinite (cvt src bfloat16 .rn sat nan x) = true ∧
      |bfloat16.toRat (cvt src bfloat16 .rn sat nan x) - src.toRat x| ≤ |src.toRat x| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ) := by
  have h := cvt_rn_err src bfloat16 sat nan x hx (by decide) (by decide) (by rw [bfloat16_max]; exact hmax)
  have e1 : (-(bfloat16.m : ℤ) - 1) = -8 := by decide
  have e2 : bfloat16.emin - bfloat16.m - 1 = -134 := by decide
  rwa [e1, e2] at h

/-- **fp8 e4m3 quantisation error** (`cvt.rn.satfinite.e4m3x2.*`, `__nv_cvt_float_to_fp8(…,
__NV_SATFINITE, __NV_E4M3)`, and without `.satfinite` in range): a finite value with `|v| ≤ 448`
becomes a finite e4m3 within `2^-4·|v| + 2^-10`. -/
theorem cvt_e4m3_rn_err (src : Format) (sat : Bool) (nan : NaNRule) (x : ℕ) (hx : src.isFinite x = true)
    (hmax : |src.toRat x| ≤ 448) :
    e4m3.isFinite (cvt src e4m3 .rn sat nan x) = true ∧
      |e4m3.toRat (cvt src e4m3 .rn sat nan x) - src.toRat x| ≤ |src.toRat x| * 2 ^ (-4 : ℤ) + 2 ^ (-10 : ℤ) := by
  have h := cvt_rn_err src e4m3 sat nan x hx (by decide) (by decide) (by rw [e4m3_max]; exact hmax)
  have e1 : (-(e4m3.m : ℤ) - 1) = -4 := by decide
  have e2 : e4m3.emin - e4m3.m - 1 = -10 := by decide
  rwa [e1, e2] at h

/-- **fp8 e5m2 quantisation error**: a finite value with `|v| ≤ 57344` becomes a finite e5m2 within
`2^-3·|v| + 2^-17`. -/
theorem cvt_e5m2_rn_err (src : Format) (sat : Bool) (nan : NaNRule) (x : ℕ) (hx : src.isFinite x = true)
    (hmax : |src.toRat x| ≤ 57344) :
    e5m2.isFinite (cvt src e5m2 .rn sat nan x) = true ∧
      |e5m2.toRat (cvt src e5m2 .rn sat nan x) - src.toRat x| ≤ |src.toRat x| * 2 ^ (-3 : ℤ) + 2 ^ (-17 : ℤ) := by
  have h := cvt_rn_err src e5m2 sat nan x hx (by decide) (by decide) (by rw [e5m2_max]; exact hmax)
  have e1 : (-(e5m2.m : ℤ) - 1) = -3 := by decide
  have e2 : e5m2.emin - e5m2.m - 1 = -17 := by decide
  rwa [e1, e2] at h

end TVA.Math.Float.Format
