import TVA.Math.Float.Format
import TVA.Math.Float.F32.Error

/-!
# Rounding into a generic format: the reduction to an integer rounding

`Format.roundMag` computes the clamped exponent `expOf q`, scales `q` to a fraction
`num / den = q · 2^(m − expOf q)`, rounds it to an integer with `roundDiv`, and encodes the
result (`encode`). This module states that decomposition (`roundMag_eq`, by definition) and
the facts the bridges use:

* `roundDiv_congr`: `roundDiv` depends only on the value `num / den`;
* `roundDiv_exact`: an integer fraction rounds to itself under every mode;
* `roundDiv_rn_two_pow`: rounding to nearest-even by `2^k` is the bit trick
  `(N + 2^(k-1) − 1 + lsb) >>> k`;
* `roundMag_of_scale`: `roundMag` of `q` is `encode (roundDiv num den) E` for any fraction
  `num / den` equal to `q · 2^(m − E)`, where `E` is the clamped exponent of `q`;
* `expOf_eq_of_bracket`, `expOf_eq_emin`: the clamped exponent of a bracketed `q`.

Proof module (Mathlib); the definitions are in `TVA.Math.Float.Format`.
-/
set_option autoImplicit false

namespace TVA.Math.Float.Format

open TVA.Math.Float

variable (f : Format)

/-- The clamped exponent `max ⌊log₂ q⌋ emin` of a positive rational. -/
def expOf (q : ℚ) : ℤ := max (log2Floor q.num.natAbs q.den) f.emin

/-- The fraction `num / den = q · 2^(m − expOf q)` that `roundMag` rounds to an integer. -/
def scaled (q : ℚ) : ℕ × ℕ :=
  let n := q.num.natAbs
  let d := q.den
  let k : ℤ := f.expOf q - f.m
  if k ≥ 0 then (n, d * 2 ^ k.toNat) else (n * 2 ^ (-k).toNat, d)

/-- Encode a rounded significand `m ≤ 2^(m+1)` at the unbiased exponent `e'`. -/
def encode (mode : RoundMode) (sat neg : Bool) (m : ℕ) (e' : ℤ) : ℕ :=
  let (m, e') := if m == 2 ^ (f.m + 1) then (2 ^ f.m, e' + 1) else (m, e')
  if f.over m e' then f.overflow mode sat neg
  else if m < 2 ^ f.m then m
  else ((e' + f.bias).toNat <<< f.m) ||| (m - 2 ^ f.m)

theorem roundMag_eq (mode : RoundMode) (sat neg : Bool) (q : ℚ) :
    f.roundMag mode sat neg q = if q ≤ 0 then 0 else
      match f.scaled q with
      | (num, den) => f.encode mode sat neg (roundDiv mode neg num den) (f.expOf q) := rfl

/-! ## `roundDiv` -/

theorem log2Floor_eq_f32 : log2Floor = F32.log2Floor := rfl

/-- Floor and remainder of `num / den` through the rational value. -/
theorem mod_div_rat {n d : ℕ} (hd : 0 < d) :
    ((n % d : ℕ) : ℚ) / d = (n : ℚ) / d - ((n / d : ℕ) : ℚ) := by
  have hdq : (d : ℚ) ≠ 0 := by exact_mod_cast hd.ne'
  have h : ((n % d : ℕ) : ℚ) + d * ((n / d : ℕ) : ℚ) = n := by exact_mod_cast Nat.mod_add_div n d
  rw [eq_sub_iff_add_eq, div_add' _ _ _ hdq]
  congr 1
  linarith

theorem div_eq_floor {n d : ℕ} : n / d = ⌊(n : ℚ) / d⌋₊ := (Nat.floor_div_eq_div n d).symm

/-- `roundDiv` depends only on the value of the fraction. -/
theorem roundDiv_congr (mode : RoundMode) (neg : Bool) {n d n' d' : ℕ} (hd : 0 < d) (hd' : 0 < d')
    (h : (n : ℚ) / d = (n' : ℚ) / d') : roundDiv mode neg n d = roundDiv mode neg n' d' := by
  have hfl : n / d = n' / d' := by rw [div_eq_floor, div_eq_floor, h]
  have hdq : (0 : ℚ) < d := by exact_mod_cast hd
  have hdq' : (0 : ℚ) < d' := by exact_mod_cast hd'
  -- the fractional part `t`, the same for both
  have ht : ((n % d : ℕ) : ℚ) / d = ((n' % d' : ℕ) : ℚ) / d' := by
    rw [mod_div_rat hd, mod_div_rat hd', h, hfl]
  have hgt : (2 * (n % d) > d) ↔ (2 * (n' % d') > d') := by
    have e1 : (2 * (n % d) > d) ↔ (1 : ℚ) / 2 < ((n % d : ℕ) : ℚ) / d := by
      rw [lt_div_iff₀ hdq]; constructor <;> intro hh
      · have : ((d : ℕ) : ℚ) < 2 * ((n % d : ℕ) : ℚ) := by exact_mod_cast hh
        linarith
      · have : ((d : ℕ) : ℚ) < 2 * ((n % d : ℕ) : ℚ) := by linarith
        exact_mod_cast this
    have e2 : (2 * (n' % d') > d') ↔ (1 : ℚ) / 2 < ((n' % d' : ℕ) : ℚ) / d' := by
      rw [lt_div_iff₀ hdq']; constructor <;> intro hh
      · have : ((d' : ℕ) : ℚ) < 2 * ((n' % d' : ℕ) : ℚ) := by exact_mod_cast hh
        linarith
      · have : ((d' : ℕ) : ℚ) < 2 * ((n' % d' : ℕ) : ℚ) := by linarith
        exact_mod_cast this
    rw [e1, e2, ht]
  have hlt : (2 * (n % d) < d) ↔ (2 * (n' % d') < d') := by
    have e1 : (2 * (n % d) < d) ↔ ((n % d : ℕ) : ℚ) / d < 1 / 2 := by
      rw [div_lt_iff₀ hdq]; constructor <;> intro hh
      · have : 2 * ((n % d : ℕ) : ℚ) < ((d : ℕ) : ℚ) := by exact_mod_cast hh
        linarith
      · have : 2 * ((n % d : ℕ) : ℚ) < ((d : ℕ) : ℚ) := by linarith
        exact_mod_cast this
    have e2 : (2 * (n' % d') < d') ↔ ((n' % d' : ℕ) : ℚ) / d' < 1 / 2 := by
      rw [div_lt_iff₀ hdq']; constructor <;> intro hh
      · have : 2 * ((n' % d' : ℕ) : ℚ) < ((d' : ℕ) : ℚ) := by exact_mod_cast hh
        linarith
      · have : 2 * ((n' % d' : ℕ) : ℚ) < ((d' : ℕ) : ℚ) := by linarith
        exact_mod_cast this
    rw [e1, e2, ht]
  have hge : (2 * (n % d) ≥ d) ↔ (2 * (n' % d') ≥ d') := by
    rw [ge_iff_le, ge_iff_le, ← not_lt, ← not_lt, hlt]
  have hz : (n % d = 0) ↔ (n' % d' = 0) := by
    have e1 : (n % d = 0) ↔ ((n % d : ℕ) : ℚ) / d = 0 := by
      rw [div_eq_zero_iff]; constructor
      · intro hh; left; exact_mod_cast hh
      · rintro (hh | hh)
        · exact_mod_cast hh
        · exact absurd hh hdq.ne'
    have e2 : (n' % d' = 0) ↔ ((n' % d' : ℕ) : ℚ) / d' = 0 := by
      rw [div_eq_zero_iff]; constructor
      · intro hh; left; exact_mod_cast hh
      · rintro (hh | hh)
        · exact_mod_cast hh
        · exact absurd hh hdq'.ne'
    rw [e1, e2, ht]
  have hz' : (n % d != 0) = (n' % d' != 0) := by
    rw [Bool.eq_iff_iff]; simp only [bne_iff_ne, ne_eq, hz]
  unfold roundDiv
  cases mode <;> simp only [hfl, hgt, hlt, hge, hz']

/-- An integer fraction rounds to itself under every mode. -/
theorem roundDiv_exact (mode : RoundMode) (neg : Bool) {d : ℕ} (hd : 0 < d) (M : ℕ) :
    roundDiv mode neg (M * d) d = M := by
  unfold roundDiv
  rw [Nat.mul_div_cancel M hd, Nat.mul_mod_left M d]
  cases mode <;> simp [hd, hd.ne']

/-- Rounding to nearest-even by `2^k` (`k ≥ 1`) is the bit trick. -/
theorem roundDiv_rn_two_pow (neg : Bool) (N k : ℕ) (hk : 1 ≤ k) :
    roundDiv .rn neg N (2 ^ k) = (N + 2 ^ (k - 1) - 1 + (N >>> k) % 2) >>> k := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  simp only [Nat.add_sub_cancel, Nat.shiftRight_eq_div_pow]
  set h := 2 ^ j with hh
  have hpos : 0 < h := by positivity
  have hD : 2 ^ (j + 1) = 2 * h := by rw [pow_succ]; ring
  rw [hD]
  set fl := N / (2 * h)
  set r := N % (2 * h)
  have hN : N = 2 * h * fl + r := (Nat.div_add_mod N (2 * h)).symm
  have hr : r < 2 * h := Nat.mod_lt _ (by omega)
  have hlsb : fl % 2 < 2 := Nat.mod_lt _ (by norm_num)
  -- `(N + h − 1 + lsb) / 2h = fl + (r + h − 1 + lsb) / 2h`
  have hsplit : (N + h - 1 + fl % 2) / (2 * h) = fl + (r + h - 1 + fl % 2) / (2 * h) := by
    rw [show N + h - 1 + fl % 2 = (r + h - 1 + fl % 2) + 2 * h * fl by omega,
      Nat.add_mul_div_left _ _ (by omega)]
    ring
  rw [hsplit]
  unfold roundDiv
  simp only
  split_ifs with h1 h2 h3
  · -- r > h: one more
    have : (r + h - 1 + fl % 2) / (2 * h) = 1 := by
      rw [Nat.div_eq_iff (by omega)]; constructor <;> omega
    omega
  · -- r < h
    have : (r + h - 1 + fl % 2) / (2 * h) = 0 := by
      rw [Nat.div_eq_iff (by omega)]; omega
    omega
  · -- tie, even
    have h3' : fl % 2 = 0 := by simpa using h3
    have : (r + h - 1 + fl % 2) / (2 * h) = 0 := by
      rw [Nat.div_eq_iff (by omega)]; omega
    omega
  · -- tie, odd
    have h3' : fl % 2 = 1 := by simp at h3; omega
    have : (r + h - 1 + fl % 2) / (2 * h) = 1 := by
      rw [Nat.div_eq_iff (by omega)]; constructor <;> omega
    omega

/-- Under `rn` the sign is irrelevant to `roundDiv`. -/
theorem roundDiv_rn_neg (neg : Bool) (n d : ℕ) : roundDiv .rn neg n d = roundDiv .rn false n d := rfl

/-! ## The scaled fraction and the clamped exponent -/

/-- The scaled fraction is `q · 2^(m − expOf q)`. -/
theorem scaled_val (q : ℚ) (hq : 0 < q) :
    0 < (f.scaled q).2 ∧ ((f.scaled q).1 : ℚ) / (f.scaled q).2 = q * (2 : ℚ) ^ ((f.m : ℤ) - f.expOf q) := by
  have hnum : ((q.num.natAbs : ℕ) : ℚ) = q.num := by
    have h1 : ((q.num.natAbs : ℕ) : ℤ) = q.num := Int.natAbs_of_nonneg (Rat.num_nonneg.2 hq.le)
    have h2 : ((q.num.natAbs : ℕ) : ℚ) = ((q.num.natAbs : ℕ) : ℤ) := (Int.cast_natCast _).symm
    rw [h2, h1]
  have hqv : ((q.num.natAbs : ℕ) : ℚ) / q.den = q := by rw [hnum, Rat.num_div_den]
  unfold scaled
  simp only
  split_ifs with hk
  · refine ⟨by positivity, ?_⟩
    have hk' : ((f.expOf q - f.m).toNat : ℤ) = f.expOf q - f.m := Int.toNat_of_nonneg hk
    push_cast
    rw [div_mul_eq_div_div, hqv, show ((f.m : ℤ) - f.expOf q) = -(f.expOf q - f.m) by ring, zpow_neg, ← hk',
      zpow_natCast, div_eq_mul_inv, Int.toNat_natCast]
  · refine ⟨q.den_pos, ?_⟩
    have hk' : ((-(f.expOf q - f.m)).toNat : ℤ) = -(f.expOf q - f.m) := Int.toNat_of_nonneg (by omega)
    push_cast
    rw [mul_div_right_comm, hqv, show ((f.m : ℤ) - f.expOf q) = -(f.expOf q - f.m) by ring, ← hk', zpow_natCast,
      Int.toNat_natCast]

/-- `roundMag` of a positive `q` through any fraction equal to `q · 2^(m − E)`, where `E` is the
clamped exponent of `q`. -/
theorem roundMag_of_scale (mode : RoundMode) (sat neg : Bool) {q : ℚ} (hq : 0 < q) {E : ℤ}
    (hE : f.expOf q = E) {num den : ℕ} (hden : 0 < den)
    (hval : (num : ℚ) / den = q * (2 : ℚ) ^ ((f.m : ℤ) - E)) :
    f.roundMag mode sat neg q = f.encode mode sat neg (roundDiv mode neg num den) E := by
  rw [roundMag_eq, ite_eq_right (not_le.mpr hq)]
  obtain ⟨h1, h2⟩ := f.scaled_val q hq
  rcases hs : f.scaled q with ⟨n, d⟩
  rw [hs] at h1 h2
  simp only at h1 h2 ⊢
  rw [hE, roundDiv_congr mode neg h1 hden (by rw [h2, hval, hE])]

/-- `log2Floor` brackets a positive rational. -/
theorem log2Floor_rat (q : ℚ) (hq : 0 < q) :
    (2 : ℚ) ^ (log2Floor q.num.natAbs q.den) ≤ q ∧ q < 2 ^ (log2Floor q.num.natAbs q.den + 1) :=
  F32.log2Floor_rat q hq

theorem log2Floor_eq_of_bracket {q : ℚ} {E : ℤ} (hq : 0 < q) (hlo : (2 : ℚ) ^ E ≤ q)
    (hhi : q < 2 ^ (E + 1)) : log2Floor q.num.natAbs q.den = E := by
  obtain ⟨hL1, hL2⟩ := log2Floor_rat q hq
  rcases lt_trichotomy (log2Floor q.num.natAbs q.den) E with h | h | h
  · exfalso
    have : (2 : ℚ) ^ (log2Floor q.num.natAbs q.den + 1) ≤ 2 ^ E :=
      zpow_le_zpow_right₀ (by norm_num) (by omega)
    linarith
  · exact h
  · exfalso
    have : (2 : ℚ) ^ (E + 1) ≤ 2 ^ (log2Floor q.num.natAbs q.den) :=
      zpow_le_zpow_right₀ (by norm_num) (by omega)
    linarith

/-- The clamped exponent of `q ∈ [2^E, 2^(E+1))` with `E ≥ emin` is `E`. -/
theorem expOf_eq_of_bracket {q : ℚ} {E : ℤ} (hq : 0 < q) (hlo : (2 : ℚ) ^ E ≤ q)
    (hhi : q < 2 ^ (E + 1)) (hE : f.emin ≤ E) : f.expOf q = E := by
  unfold expOf; rw [log2Floor_eq_of_bracket hq hlo hhi, max_eq_left hE]

/-- The clamped exponent of `0 < q < 2^emin` is `emin`. -/
theorem expOf_eq_emin {q : ℚ} (hq : 0 < q) (hlt : q < 2 ^ f.emin) : f.expOf q = f.emin := by
  unfold expOf
  apply max_eq_right
  obtain ⟨hL1, -⟩ := log2Floor_rat q hq
  by_contra hc
  rw [not_le] at hc
  have : (2 : ℚ) ^ f.emin ≤ 2 ^ (log2Floor q.num.natAbs q.den) :=
    zpow_le_zpow_right₀ (by norm_num) hc.le
  linarith

/-! ## Bits -/

theorem shl_or_eq (a b k : ℕ) (hb : b < 2 ^ k) : (a <<< k) ||| b = a * 2 ^ k + b := by
  rw [Nat.shiftLeft_eq, mul_comm, ← Nat.two_pow_add_eq_or_of_lt hb]

theorem or_two_pow_eq (b k : ℕ) (hb : b < 2 ^ k) : b ||| 2 ^ k = b + 2 ^ k := by
  rw [Nat.lor_comm, show 2 ^ k = 2 ^ k * 1 by ring, ← Nat.two_pow_add_eq_or_of_lt hb]; ring

/-- `withSign` on a magnitude below the sign bit. -/
theorem withSign_of_lt (s : Bool) {mag : ℕ} (h : mag < f.signBit) :
    f.withSign s mag = if s then mag + f.signBit else mag := by
  unfold withSign signBit at *
  cases s
  · simp [Nat.mod_eq_of_lt h]
  · simp [or_two_pow_eq _ _ h]

end TVA.Math.Float.Format
