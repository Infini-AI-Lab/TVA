import TVA.Math.Float.F32.Bounds

/-!
# Rounding to the binary32 grid: `log2Floor`, round-to-nearest-even, the characterization of `roundMag`, signs and `roundRat_err_*`

Part of the `f32` error analysis (`TVA.Math.Float.F32.Error`); the definitions it reasons about are in
`TVA.Math.Float.F32.Bounds`. Split out of `F32/Bounds.lean`.
-/
set_option autoImplicit false

namespace TVA.Math.Float.F32

open TVA.Math.Float

open Rat

/-! ### Constants: positivity -/

theorem u_pos : 0 < u := by unfold u; positivity
theorem minNormal_pos : 0 < minNormal := by unfold minNormal; positivity

/-! ### `log2Floor` -/

theorem log2Floor_spec {n d : ℕ} (hn : 0 < n) (hd : 0 < d) :
    (2 : ℚ) ^ (log2Floor n d) * d ≤ n ∧ (n : ℚ) < 2 ^ (log2Floor n d + 1) * d := by
  have hn' : n ≠ 0 := by omega
  have hd' : d ≠ 0 := by omega
  -- the four Nat facts
  have h1 : 2 ^ n.log2 ≤ n := (Nat.le_log2 hn').1 le_rfl
  have h2 : n < 2 ^ (n.log2 + 1) := (Nat.log2_lt hn').1 (Nat.lt_succ_self _)
  have h3 : 2 ^ d.log2 ≤ d := (Nat.le_log2 hd').1 le_rfl
  have h4 : d < 2 ^ (d.log2 + 1) := (Nat.log2_lt hd').1 (Nat.lt_succ_self _)
  have q1 : (2 : ℚ) ^ (n.log2 : ℤ) ≤ n := by rw [zpow_natCast]; exact_mod_cast h1
  have q2 : (n : ℚ) < 2 ^ ((n.log2 : ℤ) + 1) := by
    rw [show ((n.log2 : ℤ) + 1) = ((n.log2 + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]; exact_mod_cast h2
  have q3 : (2 : ℚ) ^ (d.log2 : ℤ) ≤ d := by rw [zpow_natCast]; exact_mod_cast h3
  have q4 : (d : ℚ) < 2 ^ ((d.log2 : ℤ) + 1) := by
    rw [show ((d.log2 : ℤ) + 1) = ((d.log2 + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]; exact_mod_cast h4
  have two_ne : (2 : ℚ) ≠ 0 := by norm_num
  have hdq : (0 : ℚ) < d := by exact_mod_cast hd
  unfold log2Floor
  set a := n.log2 with ha
  set b := d.log2 with hb
  by_cases hab : a ≥ b
  · -- quotient at least 2^(a-b) iff n ≥ d * 2^(a-b)
    have hpow : (2 : ℚ) ^ ((a : ℤ) - b) = (2 ^ (a - b) : ℕ) := by
      rw [show ((a : ℤ) - b) = ((a - b : ℕ) : ℤ) by omega, zpow_natCast]; push_cast; rfl
    by_cases hge : n ≥ d * 2 ^ (a - b)
    · simp only [hab, hge, ↓reduceIte, decide_true]
      constructor
      · rw [hpow]; exact_mod_cast (by nlinarith [hge] : (2 ^ (a - b) : ℕ) * d ≤ n)
      · -- n < 2^(a+1) = 2^(a-b+1) * 2^b ≤ 2^(a-b+1) * d
        have : (2 : ℚ) ^ ((a : ℤ) - b + 1) * 2 ^ (b : ℤ) = 2 ^ ((a : ℤ) + 1) := by
          rw [← zpow_add₀ two_ne]; congr 1; ring
        calc (n : ℚ) < 2 ^ ((a : ℤ) + 1) := q2
          _ = 2 ^ ((a : ℤ) - b + 1) * 2 ^ (b : ℤ) := this.symm
          _ ≤ 2 ^ ((a : ℤ) - b + 1) * d := by gcongr
    · simp only [hab, hge, ↓reduceIte, decide_false, Bool.false_eq_true]
      constructor
      · -- 2^(a-b-1) * d < 2^(a-b-1) * 2^(b+1) = 2^a ≤ n
        have : (2 : ℚ) ^ ((a : ℤ) - b - 1) * 2 ^ ((b : ℤ) + 1) = 2 ^ (a : ℤ) := by
          rw [← zpow_add₀ two_ne]; congr 1; ring
        calc (2 : ℚ) ^ ((a : ℤ) - b - 1) * d ≤ 2 ^ ((a : ℤ) - b - 1) * 2 ^ ((b : ℤ) + 1) := by gcongr
          _ = 2 ^ (a : ℤ) := this
          _ ≤ n := q1
      · rw [not_le] at hge
        rw [show ((a : ℤ) - b - 1 + 1) = (a : ℤ) - b by ring, hpow]
        exact_mod_cast (by nlinarith [hge] : n < (2 ^ (a - b) : ℕ) * d)
  · rw [not_le] at hab
    have hpow : (2 : ℚ) ^ ((b : ℤ) - a) = (2 ^ (b - a) : ℕ) := by
      rw [show ((b : ℤ) - a) = ((b - a : ℕ) : ℤ) by omega, zpow_natCast]; push_cast; rfl
    have hinv : (2 : ℚ) ^ ((a : ℤ) - b) = 1 / (2 ^ (b - a) : ℕ) := by
      rw [← hpow, show ((a : ℤ) - b) = -((b : ℤ) - a) by ring, zpow_neg, one_div]
    have hpos : (0 : ℚ) < (2 ^ (b - a) : ℕ) := by positivity
    have hab' : ¬ a ≥ b := by omega
    by_cases hge : n * 2 ^ (b - a) ≥ d
    · simp only [hab', hge, ↓reduceIte, decide_true]
      constructor
      · rw [hinv, div_mul_eq_mul_div, one_mul, div_le_iff₀ hpos]
        exact_mod_cast (by nlinarith [hge] : d ≤ n * (2 ^ (b - a) : ℕ))
      · have : (2 : ℚ) ^ ((a : ℤ) - b + 1) * 2 ^ (b : ℤ) = 2 ^ ((a : ℤ) + 1) := by
          rw [← zpow_add₀ two_ne]; congr 1; ring
        calc (n : ℚ) < 2 ^ ((a : ℤ) + 1) := q2
          _ = 2 ^ ((a : ℤ) - b + 1) * 2 ^ (b : ℤ) := this.symm
          _ ≤ 2 ^ ((a : ℤ) - b + 1) * d := by gcongr
    · simp only [hab', hge, ↓reduceIte, decide_false, Bool.false_eq_true]
      constructor
      · have : (2 : ℚ) ^ ((a : ℤ) - b - 1) * 2 ^ ((b : ℤ) + 1) = 2 ^ (a : ℤ) := by
          rw [← zpow_add₀ two_ne]; congr 1; ring
        calc (2 : ℚ) ^ ((a : ℤ) - b - 1) * d ≤ 2 ^ ((a : ℤ) - b - 1) * 2 ^ ((b : ℤ) + 1) := by gcongr
          _ = 2 ^ (a : ℤ) := this
          _ ≤ n := q1
      · rw [not_le] at hge
        rw [show ((a : ℤ) - b - 1 + 1) = (a : ℤ) - b by ring, hinv, div_mul_eq_mul_div, one_mul,
          lt_div_iff₀ hpos]
        exact_mod_cast (by nlinarith [hge] : n * (2 ^ (b - a) : ℕ) < d)

/-! ### Round-to-nearest-even of a non-negative fraction -/

theorem rne_err {num den : ℕ} (hd : 0 < den) : |(rne num den : ℚ) - num / den| ≤ 1 / 2 := by
  have hdq : (0 : ℚ) < den := by exact_mod_cast hd
  set t : ℚ := (num % den : ℕ) / den with ht
  have hdiv : (num : ℚ) / den = (num / den : ℕ) + t := by
    have hq : (num : ℚ) = (num / den : ℕ) * den + (num % den : ℕ) := by
      exact_mod_cast (Nat.div_add_mod' num den).symm
    rw [hq, add_div, mul_div_assoc, div_self hdq.ne', mul_one]
  have ht0 : 0 ≤ t := by positivity
  have ht1 : t < 1 := by rw [ht, div_lt_one hdq]; exact_mod_cast Nat.mod_lt _ hd
  rw [hdiv, abs_sub_le_iff]
  simp only [rne]
  split_ifs with h1 h2 h3
  · have hh : (den : ℚ) < 2 * (num % den : ℕ) := by exact_mod_cast (by omega : den < 2 * (num % den))
    have : 1 / 2 < t := by rw [ht, lt_div_iff₀ hdq]; linarith
    push_cast; constructor <;> linarith
  · have hh : (2 : ℚ) * (num % den : ℕ) < den := by exact_mod_cast (by omega : 2 * (num % den) < den)
    have : t < 1 / 2 := by rw [ht, div_lt_iff₀ hdq]; linarith
    constructor <;> linarith
  · have h2r : (2 : ℚ) * (num % den : ℕ) = den := by exact_mod_cast (by omega : 2 * (num % den) = den)
    have : t = 1 / 2 := by rw [ht, div_eq_iff hdq.ne']; linarith
    constructor <;> linarith
  · have h2r : (2 : ℚ) * (num % den : ℕ) = den := by exact_mod_cast (by omega : 2 * (num % den) = den)
    have : t = 1 / 2 := by rw [ht, div_eq_iff hdq.ne']; linarith
    push_cast; constructor <;> linarith

/-- An integer within `1/2` of `y` lies in `[A, B]` whenever `A ≤ y < B` (integers). -/
theorem rne_range {num den : ℕ} (hd : 0 < den) {A B : ℕ} (hA : (A : ℚ) ≤ num / den)
    (hB : (num : ℚ) / den < B) : A ≤ rne num den ∧ rne num den ≤ B := by
  have h := rne_err (num := num) hd
  rw [abs_sub_le_iff] at h
  constructor
  · by_contra hlt
    have h' : rne num den + 1 ≤ A := by omega
    have : ((rne num den : ℕ) : ℚ) + 1 ≤ A := by exact_mod_cast h'
    linarith
  · by_contra hlt
    have h' : B + 1 ≤ rne num den := by omega
    have : (B : ℚ) + 1 ≤ rne num den := by exact_mod_cast h'
    linarith

/-- The non-strict upper variant: `y ≤ B` integer gives `rne ≤ B`. -/
theorem rne_le_of_le {num den : ℕ} (hd : 0 < den) {B : ℕ} (hB : (num : ℚ) / den ≤ B) :
    rne num den ≤ B := by
  have h := rne_err (num := num) hd
  rw [abs_sub_le_iff] at h
  by_contra hlt
  have h' : B + 1 ≤ rne num den := by omega
  have : (B : ℚ) + 1 ≤ rne num den := by exact_mod_cast h'
  linarith

/-! ### Bit-level facts about the encodings -/

theorem toNat_ofNat_of_lt {v : ℕ} (h : v < 2 ^ 32) : (BitVec.ofNat 32 v).toNat = v := by
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt h]

theorem expField_eq (x : Bits) : expField x = x.toNat / 2 ^ 23 % 256 := by
  unfold expField; rw [Nat.shiftRight_eq_div_pow]

theorem mantField_eq (x : Bits) : mantField x = x.toNat % 2 ^ 23 := rfl

theorem sign_eq (x : Bits) : sign x = decide (x.toNat ≥ 2 ^ 31) := rfl

/-- `(a <<< 23) ||| m = a * 2^23 + m` for a 23-bit `m`. -/
theorem shl_or_eq (a m : ℕ) (hm : m < 2 ^ 23) : (a <<< 23) ||| m = a * 2 ^ 23 + m := by
  rw [Nat.shiftLeft_eq, mul_comm, ← Nat.two_pow_add_eq_or_of_lt hm]

theorem magRat_of_exp_ne {x : Bits} (he : expField x ≠ 0) :
    magRat x = ((2 ^ 23 + mantField x : ℕ) : ℚ) * (2 : ℚ) ^ ((expField x : ℤ) - 150) := by
  unfold magRat
  simp only [he, beq_iff_eq, ↓reduceIte]
  split_ifs with h
  · rw [show ((expField x : ℤ) - 150) = ((expField x - 150 : ℕ) : ℤ) by omega, zpow_natCast]
    push_cast; ring
  · rw [Rat.mkRat_eq_div, show ((expField x : ℤ) - 150) = -((150 - expField x : ℕ) : ℤ) by omega,
      zpow_neg, zpow_natCast, div_eq_mul_inv]
    push_cast; ring

theorem magRat_of_exp_zero {x : Bits} (he : expField x = 0) :
    magRat x = (mantField x : ℚ) / 2 ^ 149 := by
  unfold magRat
  simp only [he, beq_self_eq_true, ↓reduceIte, Rat.mkRat_eq_div]
  push_cast; ring

/-! ### `roundMag` as "round the scaled fraction to an integer, then encode" -/

theorem roundMag_eq (q : ℚ) :
    roundMag q = if q ≤ 0 then posZero else
      match scaled q with
      | (num, den) => encode (rne num den) (expOf q) := rfl

theorem expOf_ge (q : ℚ) : -126 ≤ expOf q := le_max_right _ _

/-- The scaled fraction is `q / 2^(expOf q − 23)`. -/
theorem scaled_val (q : ℚ) (hq : 0 < q) :
    0 < (scaled q).2 ∧ ((scaled q).1 : ℚ) / (scaled q).2 = q * (2 : ℚ) ^ (23 - expOf q) := by
  have hnum : ((q.num.natAbs : ℕ) : ℚ) = q.num := by
    have h1 : ((q.num.natAbs : ℕ) : ℤ) = q.num := Int.natAbs_of_nonneg (Rat.num_nonneg.2 hq.le)
    have h2 : ((q.num.natAbs : ℕ) : ℚ) = ((q.num.natAbs : ℕ) : ℤ) := (Int.cast_natCast _).symm
    rw [h2, h1]
  have hd : (0 : ℚ) < q.den := by exact_mod_cast q.den_pos
  have hqv : ((q.num.natAbs : ℕ) : ℚ) / q.den = q := by rw [hnum, Rat.num_div_den]
  unfold scaled
  simp only
  split_ifs with hk
  · refine ⟨by positivity, ?_⟩
    have hk' : ((expOf q - 23).toNat : ℤ) = expOf q - 23 := Int.toNat_of_nonneg hk
    push_cast
    rw [div_mul_eq_div_div, hqv, show (23 - expOf q) = -(expOf q - 23) by ring, zpow_neg, ← hk',
      zpow_natCast, div_eq_mul_inv, Int.toNat_natCast]
  · refine ⟨q.den_pos, ?_⟩
    have hk' : ((-(expOf q - 23)).toNat : ℤ) = -(expOf q - 23) := Int.toNat_of_nonneg (by omega)
    push_cast
    rw [mul_div_right_comm, hqv, show (23 - expOf q) = -(expOf q - 23) by ring, ← hk', zpow_natCast,
      Int.toNat_natCast]

/-- Value and well-formedness of an encoding. -/
theorem encode_spec (m : ℕ) (e' : ℤ) (he : -126 ≤ e') (hm : m ≤ 2 ^ 24)
    (hfin : e' ≤ 127) (hcarry : m = 2 ^ 24 → e' < 127) (hsub : m < 2 ^ 23 → e' = -126) :
    isFinite (encode m e') = true ∧ sign (encode m e') = false ∧
    magRat (encode m e') = (m : ℚ) * (2 : ℚ) ^ (e' - 23) := by
  by_cases hm' : m = 2 ^ 24
  · -- carry: m = 2^24 ↦ (2^23, e'+1)
    subst hm'
    have hlt : e' + 1 ≤ 127 := by have := hcarry rfl; omega
    simp only [encode, beq_self_eq_true, ↓reduceIte]
    rw [ite_eq_right (by omega), ite_eq_right (lt_irrefl _), Nat.sub_self]
    have hv : ((e' + 1 + 127).toNat <<< 23) ||| 0 = (e' + 1 + 127).toNat * 2 ^ 23 + 0 :=
      shl_or_eq _ _ (by norm_num)
    have hE : (e' + 1 + 127).toNat ≤ 254 := by omega
    have hlt32 : (e' + 1 + 127).toNat * 2 ^ 23 + 0 < 2 ^ 32 := by
      have : (e' + 1 + 127).toNat * 2 ^ 23 ≤ 254 * 2 ^ 23 := Nat.mul_le_mul_right _ hE
      omega
    have htn : (BitVec.ofNat 32 (((e' + 1 + 127).toNat <<< 23) ||| 0)).toNat = (e' + 1 + 127).toNat * 2 ^ 23 + 0 := by
      rw [hv]; exact toNat_ofNat_of_lt hlt32
    have hexp : expField (BitVec.ofNat 32 (((e' + 1 + 127).toNat <<< 23) ||| 0)) = (e' + 1 + 127).toNat := by
      rw [expField_eq, htn]; omega
    have hmant : mantField (BitVec.ofNat 32 (((e' + 1 + 127).toNat <<< 23) ||| 0)) = 0 := by
      rw [mantField_eq, htn]; omega
    refine ⟨?_, ?_, ?_⟩
    · unfold isFinite; rw [hexp]; all_goals exact bne_iff_ne.mpr (by omega)
    · rw [sign_eq, htn]; exact decide_eq_false (by omega)
    · rw [magRat_of_exp_ne (by rw [hexp]; omega), hexp, hmant]
      have : (((e' + 1 + 127).toNat : ℕ) : ℤ) = e' + 128 := by rw [Int.toNat_of_nonneg (by omega)]; ring
      rw [this, show (e' + 128 - 150) = (e' - 23) + 1 by ring, zpow_add₀ (by norm_num : (2:ℚ) ≠ 0)]
      push_cast; ring
  · have hb : (m == 2 ^ 24) = false := beq_eq_false_iff_ne.2 hm'
    have hm2 : m < 2 ^ 24 := lt_of_le_of_ne hm hm'
    simp only [encode, hb, Bool.false_eq_true, ↓reduceIte]
    rw [ite_eq_right (by omega)]
    split_ifs with hlt
    · -- subnormal / zero: pattern `m`, `e' = -126`
      have he' := hsub hlt
      have htn : (BitVec.ofNat 32 m).toNat = m := toNat_ofNat_of_lt (by omega)
      have hexp : expField (BitVec.ofNat 32 m) = 0 := by
        rw [expField_eq, htn, Nat.div_eq_of_lt hlt]
      refine ⟨?_, ?_, ?_⟩
      · unfold isFinite; rw [hexp]; all_goals exact bne_iff_ne.mpr (by omega)
      · rw [sign_eq, htn]; exact decide_eq_false (by omega)
      · rw [magRat_of_exp_zero hexp, mantField_eq, htn, Nat.mod_eq_of_lt hlt, he']
        rw [show ((-126 : ℤ) - 23) = -((149 : ℕ) : ℤ) by norm_num, zpow_neg, zpow_natCast, div_eq_mul_inv]
    · rw [not_lt] at hlt
      have hv : ((e' + 127).toNat <<< 23) ||| (m - 2 ^ 23) = (e' + 127).toNat * 2 ^ 23 + (m - 2 ^ 23) :=
        shl_or_eq _ _ (by omega)
      have hE : (e' + 127).toNat ≤ 254 := by omega
      have hlt32 : (e' + 127).toNat * 2 ^ 23 + (m - 2 ^ 23) < 2 ^ 32 := by
        have : (e' + 127).toNat * 2 ^ 23 ≤ 254 * 2 ^ 23 := Nat.mul_le_mul_right _ hE
        omega
      have htn : (BitVec.ofNat 32 (((e' + 127).toNat <<< 23) ||| (m - 2 ^ 23))).toNat =
          (e' + 127).toNat * 2 ^ 23 + (m - 2 ^ 23) := by
        rw [hv]; exact toNat_ofNat_of_lt hlt32
      have hexp : expField (BitVec.ofNat 32 (((e' + 127).toNat <<< 23) ||| (m - 2 ^ 23))) = (e' + 127).toNat := by
        rw [expField_eq, htn]; omega
      have hmant : mantField (BitVec.ofNat 32 (((e' + 127).toNat <<< 23) ||| (m - 2 ^ 23))) = m - 2 ^ 23 := by
        rw [mantField_eq, htn]; omega
      refine ⟨?_, ?_, ?_⟩
      · unfold isFinite; rw [hexp]; all_goals exact bne_iff_ne.mpr (by omega)
      · rw [sign_eq, htn]; exact decide_eq_false (by omega)
      · rw [magRat_of_exp_ne (by rw [hexp]; omega), hexp, hmant]
        have : (((e' + 127).toNat : ℕ) : ℤ) = e' + 127 := Int.toNat_of_nonneg (by omega)
        rw [this, show (e' + 127 - 150) = e' - 23 by ring, show 2 ^ 23 + (m - 2 ^ 23) = m by omega]

/-! ### The characterization of `roundMag` -/

theorem two_zpow_pos (n : ℤ) : (0 : ℚ) < 2 ^ n := zpow_pos (by norm_num) n

/-- `roundMag` of a positive rational below overflow: finite, sign bit clear, and within
half a quantum `2^(expOf q − 24)` of `q`. -/
theorem roundMag_spec (q : ℚ) (hq : 0 < q) (hmax : q ≤ maxNormal) :
    isFinite (roundMag q) = true ∧ sign (roundMag q) = false ∧
    |magRat (roundMag q) - q| ≤ (2 : ℚ) ^ (expOf q - 24) := by
  obtain ⟨hden, hsv⟩ := scaled_val q hq
  rcases hs : scaled q with ⟨num, den⟩
  rw [hs] at hden hsv
  simp only at hden hsv
  rw [roundMag_eq, ite_eq_right (not_le.mpr hq), hs]
  simp only
  -- the exponent
  have hn : 0 < q.num.natAbs := by have := Rat.num_pos.2 hq; omega
  obtain ⟨hlo, hhi⟩ := log2Floor_spec hn q.den_pos
  have hdq : (0 : ℚ) < q.den := by exact_mod_cast q.den_pos
  have hqeq : ((q.num.natAbs : ℕ) : ℚ) / q.den = q := by
    have h1 : ((q.num.natAbs : ℕ) : ℤ) = q.num := Int.natAbs_of_nonneg (Rat.num_nonneg.2 hq.le)
    have h2 : ((q.num.natAbs : ℕ) : ℚ) = ((q.num.natAbs : ℕ) : ℤ) := (Int.cast_natCast _).symm
    rw [h2, h1, Rat.num_div_den]
  set L := log2Floor q.num.natAbs q.den with hL
  have hL1 : (2 : ℚ) ^ L ≤ q := by rw [← hqeq, le_div_iff₀ hdq]; exact hlo
  have hL2 : q < (2 : ℚ) ^ (L + 1) := by rw [← hqeq, div_lt_iff₀ hdq]; exact hhi
  have hE : expOf q = max L (-126) := rfl
  have two_ne : (2 : ℚ) ≠ 0 := by norm_num
  -- the scaled fraction and its rounding
  set y : ℚ := (num : ℚ) / den with hy
  have hm := rne_err (num := num) hden
  rw [← hy] at hm
  set m := rne num den with hmdef
  have hy' : y = q * 2 ^ (23 - expOf q) := hsv
  -- useful powers
  have h23 : ((2 ^ 23 : ℕ) : ℚ) = (2 : ℚ) ^ (23 : ℤ) := by norm_num
  have h24 : ((2 ^ 24 : ℕ) : ℚ) = (2 : ℚ) ^ (24 : ℤ) := by norm_num
  have h24' : ((2 ^ 24 - 1 : ℕ) : ℚ) = (2 : ℚ) ^ (24 : ℤ) - 1 := by norm_num
  -- bounds on m, by the two cases of the clamp
  have key : (2 ^ 23 ≤ m ∧ m ≤ 2 ^ 24 ∧ -126 ≤ L) ∨ (m ≤ 2 ^ 23 ∧ L < -126) := by
    by_cases hLE : -126 ≤ L
    · left
      have hEL : expOf q = L := by rw [hE, max_eq_left hLE]
      have hA : ((2 ^ 23 : ℕ) : ℚ) ≤ y := by
        rw [hy', hEL, h23]
        calc (2 : ℚ) ^ (23 : ℤ) = 2 ^ L * 2 ^ (23 - L) := by rw [← zpow_add₀ two_ne]; congr 1; ring
          _ ≤ q * 2 ^ (23 - L) := by gcongr
      have hB : y < ((2 ^ 24 : ℕ) : ℚ) := by
        rw [hy', hEL, h24]
        calc q * (2 : ℚ) ^ (23 - L) < 2 ^ (L + 1) * 2 ^ (23 - L) := by gcongr
          _ = 2 ^ (24 : ℤ) := by rw [← zpow_add₀ two_ne]; congr 1; ring
      have hr := rne_range hden (A := 2 ^ 23) (B := 2 ^ 24) (num := num) (den := den) (hy ▸ hA) (hy ▸ hB)
      exact ⟨hr.1, hr.2, hLE⟩
    · right
      rw [not_le] at hLE
      have hEL : expOf q = -126 := by rw [hE, max_eq_right hLE.le]
      have hA : ((0 : ℕ) : ℚ) ≤ y := by rw [hy]; positivity
      have hB : y < ((2 ^ 23 : ℕ) : ℚ) := by
        rw [hy', hEL, h23]
        have hpow : (2 : ℚ) ^ (L + 1) ≤ 2 ^ (-126 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
        have hc : (0 : ℚ) < 2 ^ (23 - (-126 : ℤ)) := two_zpow_pos _
        calc q * (2 : ℚ) ^ (23 - (-126 : ℤ)) < 2 ^ (L + 1) * 2 ^ (23 - (-126 : ℤ)) := by gcongr
          _ ≤ 2 ^ (-126 : ℤ) * 2 ^ (23 - (-126 : ℤ)) := by gcongr
          _ = 2 ^ (23 : ℤ) := by rw [← zpow_add₀ two_ne]; norm_num
      have hr := rne_range hden (A := 0) (B := 2 ^ 23) (num := num) (den := den) (hy ▸ hA) (hy ▸ hB)
      exact ⟨hr.2, hLE⟩
  -- no overflow
  have hEle : expOf q ≤ 127 := by
    rcases key with ⟨_, _, hLE⟩ | ⟨_, hLE⟩
    · rw [hE, max_eq_left hLE]
      by_contra hgt
      rw [not_le] at hgt
      have h1 : (2 : ℚ) ^ (128 : ℤ) ≤ 2 ^ L := zpow_le_zpow_right₀ (by norm_num) (by omega)
      have h2 : (2 : ℚ) ^ (128 : ℤ) ≤ maxNormal := h1.trans (hL1.trans hmax)
      unfold maxNormal at h2; norm_num at h2
    · rw [hE, max_eq_right hLE.le]; norm_num
  have hcarry : m = 2 ^ 24 → expOf q < 127 := by
    intro hm24
    by_contra hge
    have hE127 : expOf q = 127 := by omega
    have hle : m ≤ 2 ^ 24 - 1 := rne_le_of_le hden (B := 2 ^ 24 - 1) (num := num) (den := den) (by
      rw [← hy, hy', hE127, h24']
      have hmn : (2 : ℚ) ^ (24 : ℤ) - 1 = maxNormal * 2 ^ (23 - (127 : ℤ)) := by
        unfold maxNormal; norm_num
      rw [hmn]; gcongr)
    omega
  have hsub : m < 2 ^ 23 → expOf q = -126 := by
    intro hlt
    rcases key with ⟨h1, _, _⟩ | ⟨_, hLE⟩
    · omega
    · rw [hE, max_eq_right hLE.le]
  have hmle : m ≤ 2 ^ 24 := by rcases key with ⟨_, h, _⟩ | ⟨h, _⟩ <;> omega
  obtain ⟨hf, hsg, hval⟩ := encode_spec m (expOf q) (expOf_ge q) hmle hEle hcarry hsub
  refine ⟨hf, hsg, ?_⟩
  rw [hval]
  -- |m·2^(E−23) − q| = |m − y|·2^(E−23) ≤ 2^(E−24)
  have hqy : q = y * 2 ^ (expOf q - 23) := by
    rw [hy', mul_assoc, ← zpow_add₀ two_ne, show (23 - expOf q + (expOf q - 23)) = 0 by ring, zpow_zero, mul_one]
  have hdiff : (m : ℚ) * 2 ^ (expOf q - 23) - q = ((m : ℚ) - y) * 2 ^ (expOf q - 23) := by
    rw [sub_mul, ← hqy]
  calc |(m : ℚ) * 2 ^ (expOf q - 23) - q| = |(m : ℚ) - y| * 2 ^ (expOf q - 23) := by
        rw [hdiff, abs_mul, abs_of_pos (two_zpow_pos _)]
    _ ≤ (1 / 2) * 2 ^ (expOf q - 23) := by gcongr
    _ = 2 ^ (expOf q - 24) := by
        rw [show (expOf q - 24) = (-1) + (expOf q - 23) by ring, zpow_add₀ two_ne]; norm_num

/-! ### Signs, `toRat` and `roundRat` -/

theorem toNat_withSign_false (mag : Bits) (h : mag.toNat < 2 ^ 31) :
    (withSign false mag).toNat = mag.toNat := by
  unfold withSign
  simp only [Bool.false_eq_true, ↓reduceIte, BitVec.toNat_and]
  rw [show (0x7fffffff#32).toNat = 2 ^ 31 - 1 by rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt h]

theorem toNat_withSign_true (mag : Bits) (h : mag.toNat < 2 ^ 31) :
    (withSign true mag).toNat = mag.toNat + 2 ^ 31 := by
  unfold withSign
  simp only [↓reduceIte, BitVec.toNat_or]
  rw [show (0x80000000#32).toNat = 2 ^ 31 * 1 by rfl, Nat.lor_comm, ← Nat.two_pow_add_eq_or_of_lt h]
  ring

theorem fields_of_toNat {x y : Bits} (h : x.toNat % 2 ^ 31 = y.toNat % 2 ^ 31) :
    expField x = expField y ∧ mantField x = mantField y := by
  have hx := x.isLt
  have hy := y.isLt
  rw [expField_eq, expField_eq, mantField_eq, mantField_eq]
  constructor <;> omega

theorem fields_withSign (s : Bool) (mag : Bits) (h : mag.toNat < 2 ^ 31) :
    expField (withSign s mag) = expField mag ∧ mantField (withSign s mag) = mantField mag := by
  apply fields_of_toNat
  cases s
  · rw [toNat_withSign_false mag h]
  · rw [toNat_withSign_true mag h]; omega

theorem magRat_withSign (s : Bool) (mag : Bits) (h : mag.toNat < 2 ^ 31) :
    magRat (withSign s mag) = magRat mag := by
  unfold magRat; rw [(fields_withSign s mag h).1, (fields_withSign s mag h).2]

theorem isFinite_withSign (s : Bool) (mag : Bits) (h : mag.toNat < 2 ^ 31) :
    isFinite (withSign s mag) = isFinite mag := by
  unfold isFinite; rw [(fields_withSign s mag h).1]

theorem sign_withSign (s : Bool) (mag : Bits) (h : mag.toNat < 2 ^ 31) : sign (withSign s mag) = s := by
  cases s
  · rw [sign_eq, toNat_withSign_false mag h]; exact decide_eq_false (by omega)
  · rw [sign_eq, toNat_withSign_true mag h]; exact decide_eq_true (by omega)

theorem toRat_withSign (s : Bool) (mag : Bits) (h : mag.toNat < 2 ^ 31) :
    toRat (withSign s mag) = if s then -magRat mag else magRat mag := by
  unfold toRat; rw [sign_withSign s mag h, magRat_withSign s mag h]

theorem toNat_lt_of_sign_false {x : Bits} (h : sign x = false) : x.toNat < 2 ^ 31 := by
  rw [sign_eq] at h; simpa using h

theorem magRat_nonneg (x : Bits) : 0 ≤ magRat x := by
  by_cases he : expField x = 0
  · rw [magRat_of_exp_zero he]; positivity
  · rw [magRat_of_exp_ne he]; positivity

theorem abs_toRat (x : Bits) : |toRat x| = magRat x := by
  unfold toRat; split_ifs <;> simp [abs_of_nonneg (magRat_nonneg x)]

theorem magRat_posZero : magRat posZero = 0 := by
  have h2 : mantField posZero = 0 := rfl
  rw [magRat_of_exp_zero rfl, h2]; simp

theorem magRat_negZero : magRat negZero = 0 := by
  have h1 : expField negZero = 0 := rfl
  have h2 : mantField negZero = 0 := rfl
  rw [magRat_of_exp_zero h1, h2]; simp

theorem toRat_posZero : toRat posZero = 0 := by
  have hs : sign posZero = false := rfl
  unfold toRat; rw [hs, magRat_posZero]; simp

theorem toRat_negZero : toRat negZero = 0 := by
  have hs : sign negZero = true := rfl
  unfold toRat; rw [hs, magRat_negZero]; simp

theorem isFinite_posZero : isFinite posZero = true := by rfl
theorem isFinite_negZero : isFinite negZero = true := by rfl

theorem isZero_iff {x : Bits} : isZero x = true ↔ expField x = 0 ∧ mantField x = 0 := by
  unfold isZero; simp

theorem toRat_zero_of_isZero {x : Bits} (h : isZero x = true) : toRat x = 0 := by
  obtain ⟨h1, h2⟩ := isZero_iff.1 h
  unfold toRat; rw [magRat_of_exp_zero h1, h2]; simp

theorem magRat_pos_of_not_zero {x : Bits} (h : isZero x = false) : 0 < magRat x := by
  have h' : ¬ (expField x = 0 ∧ mantField x = 0) := by
    intro hc; have := isZero_iff.2 hc; rw [h] at this; cases this
  by_cases he : expField x = 0
  · have hm : mantField x ≠ 0 := fun hm => h' ⟨he, hm⟩
    rw [magRat_of_exp_zero he]
    have : (0 : ℚ) < mantField x := by exact_mod_cast Nat.pos_of_ne_zero hm
    positivity
  · rw [magRat_of_exp_ne he]
    have : (0 : ℚ) < ((2 ^ 23 + mantField x : ℕ) : ℚ) := by positivity
    positivity

theorem finite_not_nan_inf {x : Bits} (h : isFinite x = true) : isNaN x = false ∧ isInf x = false := by
  have h' : expField x ≠ 255 := by simpa [isFinite] using h
  constructor <;> simp [isNaN, isInf, h']

theorem roundRat_zero : roundRat 0 = posZero := by unfold roundRat; simp

/-- `log2Floor` on the numerator and denominator of a positive rational brackets it. -/
theorem log2Floor_rat (q : ℚ) (hq : 0 < q) :
    (2 : ℚ) ^ (log2Floor q.num.natAbs q.den) ≤ q ∧ q < 2 ^ (log2Floor q.num.natAbs q.den + 1) := by
  have hn : 0 < q.num.natAbs := by have := Rat.num_pos.2 hq; omega
  obtain ⟨hlo, hhi⟩ := log2Floor_spec hn q.den_pos
  have hdq : (0 : ℚ) < q.den := by exact_mod_cast q.den_pos
  have hqeq : ((q.num.natAbs : ℕ) : ℚ) / q.den = q := by
    have h1 : ((q.num.natAbs : ℕ) : ℤ) = q.num := Int.natAbs_of_nonneg (Rat.num_nonneg.2 hq.le)
    have h2 : ((q.num.natAbs : ℕ) : ℚ) = ((q.num.natAbs : ℕ) : ℤ) := (Int.cast_natCast _).symm
    rw [h2, h1, Rat.num_div_den]
  have h1 : (2 : ℚ) ^ (log2Floor q.num.natAbs q.den) ≤ (q.num.natAbs : ℚ) / q.den := by
    rw [le_div_iff₀ hdq]; exact hlo
  have h2 : (q.num.natAbs : ℚ) / q.den < 2 ^ (log2Floor q.num.natAbs q.den + 1) := by
    rw [div_lt_iff₀ hdq]; exact hhi
  rw [hqeq] at h1 h2
  exact ⟨h1, h2⟩

theorem expOf_pow_le {q : ℚ} (hq : minNormal ≤ q) : (2 : ℚ) ^ expOf q ≤ q := by
  have hq0 : 0 < q := lt_of_lt_of_le minNormal_pos hq
  obtain ⟨hL1, hL2⟩ := log2Floor_rat q hq0
  have hL : -126 ≤ log2Floor q.num.natAbs q.den := by
    by_contra h
    rw [not_le] at h
    have h1 : (2 : ℚ) ^ (log2Floor q.num.natAbs q.den + 1) ≤ 2 ^ (-126 : ℤ) :=
      zpow_le_zpow_right₀ (by norm_num) (by omega)
    have : q < 2 ^ (-126 : ℤ) := hL2.trans_le h1
    unfold minNormal at hq; norm_num at hq this; linarith
  unfold expOf; rw [max_eq_left hL]; exact hL1

theorem expOf_eq_of_sub {q : ℚ} (hq : 0 < q) (hlt : q < minNormal) : expOf q = -126 := by
  obtain ⟨hL1, hL2⟩ := log2Floor_rat q hq
  have hL : log2Floor q.num.natAbs q.den < -126 := by
    by_contra h
    rw [not_lt] at h
    have h1 : (2 : ℚ) ^ (-126 : ℤ) ≤ 2 ^ (log2Floor q.num.natAbs q.den) :=
      zpow_le_zpow_right₀ (by norm_num) h
    have : (2 : ℚ) ^ (-126 : ℤ) ≤ q := h1.trans hL1
    unfold minNormal at hlt; norm_num at hlt this; linarith
  unfold expOf; exact max_eq_right hL.le

/-- `roundRat` of a nonzero rational below overflow: finite and within half a quantum. -/
theorem roundRat_spec (q : ℚ) (hq : q ≠ 0) (hmax : |q| ≤ maxNormal) :
    isFinite (roundRat q) = true ∧ |toRat (roundRat q) - q| ≤ (2 : ℚ) ^ (expOf |q| - 24) := by
  unfold roundRat
  rw [ite_eq_right (by simpa using hq)]
  rcases lt_or_gt_of_ne hq with hneg | hpos
  · have habs : |q| = -q := abs_of_neg hneg
    rw [ite_eq_left hneg]
    obtain ⟨hf, hs, herr⟩ := roundMag_spec (-q) (by linarith) (by rwa [← habs])
    have hlt := toNat_lt_of_sign_false hs
    rw [isFinite_withSign _ _ hlt, toRat_withSign _ _ hlt]
    refine ⟨hf, ?_⟩
    simp only [hneg, decide_true, ↓reduceIte]
    rw [habs, show -magRat (roundMag (-q)) - q = -(magRat (roundMag (-q)) - (-q)) by ring, abs_neg]
    exact herr
  · have habs : |q| = q := abs_of_pos hpos
    rw [ite_eq_right (not_lt.mpr hpos.le)]
    obtain ⟨hf, hs, herr⟩ := roundMag_spec q hpos (by rwa [← habs])
    have hlt := toNat_lt_of_sign_false hs
    rw [isFinite_withSign _ _ hlt, toRat_withSign _ _ hlt]
    refine ⟨hf, ?_⟩
    simp only [not_lt.mpr hpos.le, decide_false, Bool.false_eq_true, ↓reduceIte]
    rwa [habs]

/-- **Rounding on the normal range**: `|round q − q| ≤ u·|q|`. -/
theorem roundRat_err_normal (q : ℚ) (hlo : minNormal ≤ |q|) (hmax : |q| ≤ maxNormal) :
    isFinite (roundRat q) = true ∧ |toRat (roundRat q) - q| ≤ u * |q| := by
  have hq : q ≠ 0 := by
    intro h; rw [h, abs_zero] at hlo; linarith [minNormal_pos]
  obtain ⟨hf, herr⟩ := roundRat_spec q hq hmax
  refine ⟨hf, herr.trans ?_⟩
  have hle := expOf_pow_le hlo
  have two_ne : (2 : ℚ) ≠ 0 := by norm_num
  calc (2 : ℚ) ^ (expOf |q| - 24) = 2 ^ (expOf |q|) * 2 ^ (-24 : ℤ) := by
        rw [show expOf |q| - 24 = expOf |q| + (-24 : ℤ) by ring, zpow_add₀ two_ne]
    _ ≤ |q| * 2 ^ (-24 : ℤ) := by gcongr
    _ = u * |q| := by unfold u; rw [mul_comm]; all_goals norm_num

/-- **Rounding below the normal range**: an absolute error of at most `2^-150`. -/
theorem roundRat_err_sub (q : ℚ) (hlt : |q| < minNormal) :
    isFinite (roundRat q) = true ∧ |toRat (roundRat q) - q| ≤ subErr := by
  by_cases hq : q = 0
  · subst hq; rw [roundRat_zero, toRat_posZero]
    exact ⟨isFinite_posZero, by unfold subErr; norm_num⟩
  · have hmax : |q| ≤ maxNormal := by
      unfold minNormal at hlt; unfold maxNormal; norm_num at hlt ⊢; linarith
    obtain ⟨hf, herr⟩ := roundRat_spec q hq hmax
    refine ⟨hf, herr.trans ?_⟩
    rw [expOf_eq_of_sub (abs_pos.2 hq) hlt]
    unfold subErr; norm_num

/-- One rounding, in the form the chain lemmas use: exact zero, or the normal range. -/
theorem roundRat_err_u (r : ℚ) (hr : r = 0 ∨ minNormal ≤ |r|) (hmax : |r| ≤ maxNormal) :
    isFinite (roundRat r) = true ∧ |toRat (roundRat r) - r| ≤ u * |r| := by
  rcases hr with h0 | hlo
  · subst h0; rw [roundRat_zero, toRat_posZero]; simp [isFinite_posZero]
  · exact roundRat_err_normal r hlo hmax

end TVA.Math.Float.F32
