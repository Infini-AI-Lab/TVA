import Proof.TolRstd

/-! The output element's tolerance: the fused deviation (line 108), its two products, the fused
addition of the bias and the conversion to `bfloat16` (README steps 5, 6 and 8). -/

namespace Claims.LayerNormFwd

open TVA
open Format
open scoped BigOperators

noncomputable section


/-! ### Bit patterns and `ofPat` -/

/-- Bit arithmetic: `(n mod 2s)/s = (n/s) mod 2`. -/
lemma y_tol_aux (n s : Nat) (hs : 0 < s) : (n % (2*s)) / s = (n / s) % 2 := by
  have hq : s * (n / s) + n % s = n := Nat.div_add_mod n s
  have hr : n % s < s := Nat.mod_lt n hs
  have hb : (n / s) % 2 ≤ 1 := by have := Nat.mod_lt (n / s) (by norm_num : 0 < 2); omega
  set q := n / s
  set r := n % s
  set b := q % 2
  set c := q / 2
  have hqc : q = 2 * c + b := by
    have hq2 : 2 * (n / s / 2) + (n / s) % 2 = n / s := Nat.div_add_mod (n / s) 2
    omega
  have hn : n = (b*s + r) + (2*s) * c := by
    calc n = s * q + r := hq.symm
      _ = s * (2*c + b) + r := by rw [hqc]
      _ = (b*s + r) + (2*s) * c := by ring
  have hlt : b*s + r < 2*s := by nlinarith
  have hdiv : (b*s + r) / s = b := by
    rw [show b*s + r = s*b + r by ring, Nat.mul_add_div hs, Nat.div_eq_of_lt hr, Nat.add_zero]
  rw [hn, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt, hdiv]

/-- The sign of a bit pattern depends only on the pattern modulo `2^width`. -/
lemma y_tol_sign_mod (f : Format) (n : Nat) : f.sign (n % 2 ^ f.width) = f.sign n := by
  have hp : (0:Nat) < 2 ^ f.e * 2 ^ f.m := by positivity
  have hw : 2 ^ f.width = 2 * (2 ^ f.e * 2 ^ f.m) := by
    rw [Format.width_eq, pow_add]; ring
  have hs : f.signBit = 2 ^ f.e * 2 ^ f.m := Format.signBit_eq
  unfold Format.sign
  rw [hw, hs, y_tol_aux n (2 ^ f.e * 2 ^ f.m) hp, Nat.mod_mod_of_dvd _ (dvd_refl 2)]

/-- The magnitude of a bit pattern depends only on the pattern modulo `2^width`. -/
lemma y_tol_mag_mod (f : Format) (n : Nat) : f.mag (n % 2 ^ f.width) = f.mag n := by
  unfold Format.mag
  rw [Format.signBit_eq]
  have hdvd : 2 ^ f.e * 2 ^ f.m ∣ 2 ^ f.width := by
    rw [Format.width_eq, pow_add]; exact ⟨2, by ring⟩
  exact Nat.mod_mod_of_dvd n hdvd

/-- The rational value of a bit pattern depends only on the pattern modulo `2^width`. -/
lemma y_tol_toRat_mod (f : Format) (n : Nat) : f.toRat (n % 2 ^ f.width) = f.toRat n := by
  unfold Format.toRat
  rw [y_tol_sign_mod, y_tol_mag_mod]

/-- Finiteness of a bit pattern depends only on the pattern modulo `2^width`. -/
lemma y_tol_isFinite_mod (f : Format) (n : Nat) : f.isFinite (n % 2 ^ f.width) = f.isFinite n := by
  unfold Format.isFinite Format.isNaN Format.isInf Format.expField Format.fracField
  rw [y_tol_mag_mod]

/-- The rational value of the float built from pattern `n` is the format's rational value of `n`. -/
lemma y_tol_FtoRat_ofPat (f : Format) (n : Nat) :
    (FloatV.ofPat n : FloatV f).toRat = f.toRat n := by
  unfold FloatV.toRat FloatV.pat FloatV.ofPat
  rw [BitVec.toNat_ofNat, y_tol_toRat_mod]

/-- The float built from pattern `n` is finite exactly when the format calls `n` finite. -/
lemma y_tol_FisFinite_ofPat (f : Format) (n : Nat) :
    (FloatV.ofPat n : FloatV f).isFinite = f.isFinite n := by
  unfold FloatV.isFinite FloatV.pat FloatV.ofPat
  rw [BitVec.toNat_ofNat, y_tol_isFinite_mod]

/-- A `NaNEq` whose right operand is finite is an equality. -/
lemma y_tol_eq_of_naneq {f : Format} {x r : FloatV f} (h : FloatV.NaNEq x r)
    (hr : r.isFinite = true) : x = r := by
  rcases h with h | ⟨-, h2⟩
  · exact h
  · rw [FloatV.isFinite_iff] at hr
    rw [h2] at hr
    cases hr.1

/-- The binary32 format does not treat every pattern as finite-only, so it has special values. -/
lemma y_tol_b32_specials : Format.binary32.specials ≠ .finite := by decide

/-! ### Rounding error of float32 operations -/

/-- Result-relative rounding error: within `2u|q| + 2η`. -/
lemma y_tol_round_err_real {q : ℚ} {zs : Bool}
    (hfin : Format.binary32.isFinite (Format.binary32.roundRat .rn false zs q) = true) :
    |((Format.binary32.toRat (Format.binary32.roundRat .rn false zs q) : ℚ) : ℝ) - (q : ℝ)| ≤
      2 * (2:ℝ) ^ (-24 : ℤ) * |(q : ℝ)| + 2 * (2:ℝ) ^ (-150 : ℤ) := by
  have h := Format.roundRat_rn_err_of_finite (f := Format.binary32) Format.binary32_regular
    y_tol_b32_specials zs hfin
  have h1 : ((|Format.binary32.toRat (Format.binary32.roundRat .rn false zs q) - q| : ℚ) : ℝ) ≤
      ((|Format.binary32.toRat (Format.binary32.roundRat .rn false zs q)| * Format.binary32.u +
        Format.binary32.eta : ℚ) : ℝ) := by
    exact_mod_cast h
  push_cast at h1
  rw [Format.binary32_u, Format.binary32_eta] at h1
  push_cast at h1
  have hUval : (2:ℝ) ^ (-24 : ℤ) = 1/16777216 := by norm_num
  rw [hUval] at h1 ⊢
  set r : ℝ := ((Format.binary32.toRat (Format.binary32.roundRat .rn false zs q) : ℚ) : ℝ)
  have htri : |r| ≤ |(q:ℝ)| + |r - (q:ℝ)| := by
    calc |r| = |(r - (q:ℝ)) + (q:ℝ)| := by ring_nf
      _ ≤ |r - (q:ℝ)| + |(q:ℝ)| := abs_add_le _ _
      _ = |(q:ℝ)| + |r - (q:ℝ)| := by ring
  have h2 : |r - (q:ℝ)| ≤ (|(q:ℝ)| + |r - (q:ℝ)|) * (1/16777216) + 2^(-150:ℤ) := by
    calc |r - (q:ℝ)| ≤ |r| * (1/16777216) + 2^(-150:ℤ) := h1
      _ ≤ (|(q:ℝ)| + |r - (q:ℝ)|) * (1/16777216) + 2^(-150:ℤ) := by
          have := mul_le_mul_of_nonneg_right htri (by norm_num : (0:ℝ) ≤ 1/16777216)
          linarith
  have h4 : |r - (q:ℝ)| * (1 - 1/16777216) ≤ (1/16777216) * |(q:ℝ)| + 2^(-150:ℤ) := by
    nlinarith [h2]
  have hhalf : (1:ℝ)/2 ≤ 1 - 1/16777216 := by norm_num
  have hm := mul_le_mul_of_nonneg_left hhalf (abs_nonneg (r - (q:ℝ)))
  have h5 : |r - (q:ℝ)| / 2 ≤ (1/16777216) * |(q:ℝ)| + 2^(-150:ℤ) := by linarith [h4, hm]
  show |r - (q:ℝ)| ≤ 2 * (1/16777216) * |(q:ℝ)| + 2 * 2^(-150:ℤ)
  linarith [h5]

/-- The rounding error of a `float32` product, result-relative. -/
lemma y_tol_mul_rel {a c r : F32} (ha : a.isFinite = true) (hc : c.isFinite = true)
    (h : FloatV.NaNEq (FloatV.mul a c) r) (hr : r.isFinite = true) :
    |r.toReal - a.toReal * c.toReal| ≤
      2 * (2:ℝ) ^ (-24 : ℤ) * |a.toReal * c.toReal| + 2 * (2:ℝ) ^ (-150 : ℤ) := by
  have he : FloatV.mul a c = r := y_tol_eq_of_naneq h hr
  subst he
  unfold FloatV.mul at hr ⊢
  rw [Format.mul_of_finite .rn ha hc] at hr ⊢
  have hfin : Format.binary32.isFinite (Format.binary32.roundRat .rn false
      (Format.binary32.sign a.pat != Format.binary32.sign c.pat)
      (Format.binary32.toRat a.pat * Format.binary32.toRat c.pat)) = true := by
    rw [← y_tol_FisFinite_ofPat]; exact hr
  have herr := y_tol_round_err_real (q := Format.binary32.toRat a.pat * Format.binary32.toRat c.pat)
    (zs := Format.binary32.sign a.pat != Format.binary32.sign c.pat) hfin
  have htR : (FloatV.ofPat (Format.binary32.roundRat .rn false
      (Format.binary32.sign a.pat != Format.binary32.sign c.pat)
      (Format.binary32.toRat a.pat * Format.binary32.toRat c.pat)) : F32).toReal
      = ((Format.binary32.toRat (Format.binary32.roundRat .rn false
      (Format.binary32.sign a.pat != Format.binary32.sign c.pat)
      (Format.binary32.toRat a.pat * Format.binary32.toRat c.pat)) : ℚ) : ℝ) := by
    simp only [FloatV.toReal, y_tol_FtoRat_ofPat]
  have hqR : ((Format.binary32.toRat a.pat * Format.binary32.toRat c.pat : ℚ) : ℝ) =
      a.toReal * c.toReal := by
    show ((Format.binary32.toRat a.pat * Format.binary32.toRat c.pat : ℚ) : ℝ)
      = ((Format.binary32.toRat a.pat : ℚ) : ℝ) * ((Format.binary32.toRat c.pat : ℚ) : ℝ)
    push_cast; ring
  rw [hqR] at herr
  rw [htR] at ⊢
  exact herr

/-- The error of one contractible addition (`AddRel`, any fusion) of finite operands with finite
result: within `4u(|a| + |c|) + 8η`. -/
lemma y_tol_addrel_rel {a c res : F32} (ha : a.isFinite = true) (hc : c.isFinite = true)
    (h : AddRel true a c res) (hr : res.isFinite = true) :
    |res.toReal - (a.toReal + c.toReal)| ≤
      4 * (2:ℝ) ^ (-24 : ℤ) * (|a.toReal| + |c.toReal|) + 8 * (2:ℝ) ^ (-150 : ℤ) := by
  have hu0 : (0:ℝ) < (2:ℝ)^(-24:ℤ) := by norm_num
  have he0 : (0:ℝ) < (2:ℝ)^(-150:ℤ) := by norm_num
  have ha0 := abs_nonneg a.toReal
  have hc0 := abs_nonneg c.toReal
  rcases h with hne | ⟨-, hfu⟩
  · have he : FloatV.add a c = res := y_tol_eq_of_naneq hne hr
    subst he
    unfold FloatV.add at hr ⊢
    rw [Format.add_of_finite .rn ha hc] at hr ⊢
    have hfin : Format.binary32.isFinite (Format.binary32.roundRat .rn false
        (Format.binary32.sign a.pat && Format.binary32.sign c.pat)
        (Format.binary32.toRat a.pat + Format.binary32.toRat c.pat)) = true := by
      rw [← y_tol_FisFinite_ofPat]; exact hr
    have herr := y_tol_round_err_real (q := Format.binary32.toRat a.pat + Format.binary32.toRat c.pat)
      (zs := Format.binary32.sign a.pat && Format.binary32.sign c.pat) hfin
    have htR : (FloatV.ofPat (Format.binary32.roundRat .rn false
        (Format.binary32.sign a.pat && Format.binary32.sign c.pat)
        (Format.binary32.toRat a.pat + Format.binary32.toRat c.pat)) : F32).toReal
        = ((Format.binary32.toRat (Format.binary32.roundRat .rn false
        (Format.binary32.sign a.pat && Format.binary32.sign c.pat)
        (Format.binary32.toRat a.pat + Format.binary32.toRat c.pat)) : ℚ) : ℝ) := by
      simp only [FloatV.toReal, y_tol_FtoRat_ofPat]
    have hqR : ((Format.binary32.toRat a.pat + Format.binary32.toRat c.pat : ℚ) : ℝ) =
        a.toReal + c.toReal := by
      show ((Format.binary32.toRat a.pat + Format.binary32.toRat c.pat : ℚ) : ℝ)
        = ((Format.binary32.toRat a.pat : ℚ) : ℝ) + ((Format.binary32.toRat c.pat : ℚ) : ℝ)
      push_cast; ring
    rw [hqR] at herr
    rw [htR] at ⊢
    have htri : |a.toReal + c.toReal| ≤ |a.toReal| + |c.toReal| := abs_add_le _ _
    have h6 : 2 * (2:ℝ)^(-24:ℤ) * |a.toReal + c.toReal| ≤
        4 * (2:ℝ)^(-24:ℤ) * (|a.toReal| + |c.toReal|) := by
      have hS : (0:ℝ) ≤ |a.toReal| + |c.toReal| := by linarith
      nlinarith [mul_le_mul_of_nonneg_left htri (le_of_lt (by positivity : (0:ℝ) < 2 * (2:ℝ)^(-24:ℤ)))]
    linarith [herr, h6]
  · rcases hfu with ⟨p, q, hp, hq, hone, z, hne⟩ | ⟨p, hp, hyi, -⟩ | ⟨q, hq, hxi, -⟩
    · have hpe := RoundsTo.err Format.binary32_regular y_tol_b32_specials hp ha
      have hqe := RoundsTo.err Format.binary32_regular y_tol_b32_specials hq hc
      have hpq : |p + q| ≤ (1 + Format.binary32.u) * (|a.toRat| + |c.toRat|) +
          2 * Format.binary32.eta := by
        have h1 := abs_sub_abs_le_abs_sub p a.toRat
        have h2 := abs_sub_abs_le_abs_sub q c.toRat
        have h3 := abs_add_le p q
        nlinarith
      have hpqR : |((p + q : ℚ) : ℝ)| ≤ (1 + (2:ℝ)^(-24:ℤ)) * (|a.toReal| + |c.toReal|) +
          2 * (2:ℝ)^(-150:ℤ) := by
        have hcast : ((|p + q| : ℚ) : ℝ) ≤
            (((1 + Format.binary32.u) * (|a.toRat| + |c.toRat|) + 2 * Format.binary32.eta : ℚ) : ℝ) := by
          exact_mod_cast hpq
        push_cast at hcast
        rw [Format.binary32_u, Format.binary32_eta] at hcast
        push_cast at hcast
        have hq1 : ((FloatV.toRat a : ℚ) : ℝ) = a.toReal := rfl
        have hq2 : ((FloatV.toRat c : ℚ) : ℝ) = c.toReal := rfl
        rw [hq1, hq2] at hcast
        simpa only [Rat.cast_add] using hcast
      have he : (FloatV.ofPat (Format.binary32.roundRat .rn false z (p + q)) : F32) = res :=
        y_tol_eq_of_naneq hne hr
      subst he
      have hfin : Format.binary32.isFinite (Format.binary32.roundRat .rn false z (p + q)) = true := by
        rw [← y_tol_FisFinite_ofPat]; exact hr
      have herr := y_tol_round_err_real (q := p + q) (zs := z) hfin
      have htR : (FloatV.ofPat (Format.binary32.roundRat .rn false z (p + q)) : F32).toReal
          = ((Format.binary32.toRat (Format.binary32.roundRat .rn false z (p + q)) : ℚ) : ℝ) := by
        simp only [FloatV.toReal, y_tol_FtoRat_ofPat]
      rw [htR] at ⊢
      have hexact : |(p + q) - (a.toRat + c.toRat)| ≤
          Format.binary32.u * (|a.toRat| + |c.toRat|) + Format.binary32.eta := by
        rcases hone with hpa | hqc
        · have h0 : p - a.toRat = 0 := by rw [hpa]; ring
          calc |(p + q) - (a.toRat + c.toRat)| = |q - c.toRat| := by
                rw [show (p + q) - (a.toRat + c.toRat) = (p - a.toRat) + (q - c.toRat) by ring,
                  h0, zero_add]
            _ ≤ |c.toRat| * Format.binary32.u + Format.binary32.eta := hqe
            _ ≤ Format.binary32.u * (|a.toRat| + |c.toRat|) + Format.binary32.eta := by
                have hc0q := abs_nonneg c.toRat
                have ha0q := abs_nonneg a.toRat
                have hu0q : (0:ℚ) ≤ Format.binary32.u := by rw [Format.binary32_u]; norm_num
                nlinarith
        · have h0 : q - c.toRat = 0 := by rw [hqc]; ring
          calc |(p + q) - (a.toRat + c.toRat)| = |p - a.toRat| := by
                rw [show (p + q) - (a.toRat + c.toRat) = (p - a.toRat) + (q - c.toRat) by ring,
                  h0, add_zero]
            _ ≤ |a.toRat| * Format.binary32.u + Format.binary32.eta := hpe
            _ ≤ Format.binary32.u * (|a.toRat| + |c.toRat|) + Format.binary32.eta := by
                have hc0q := abs_nonneg c.toRat
                have ha0q := abs_nonneg a.toRat
                have hu0q : (0:ℚ) ≤ Format.binary32.u := by rw [Format.binary32_u]; norm_num
                nlinarith
      have hexactR : |((p + q : ℚ) : ℝ) - (a.toReal + c.toReal)| ≤
          (2:ℝ)^(-24:ℤ) * (|a.toReal| + |c.toReal|) + (2:ℝ)^(-150:ℤ) := by
        have hcast : ((|(p + q) - (a.toRat + c.toRat)| : ℚ) : ℝ) ≤
            ((Format.binary32.u * (|a.toRat| + |c.toRat|) + Format.binary32.eta : ℚ) : ℝ) := by
          exact_mod_cast hexact
        push_cast at hcast
        rw [Format.binary32_u, Format.binary32_eta] at hcast
        push_cast at hcast
        have hq1 : ((FloatV.toRat a : ℚ) : ℝ) = a.toReal := rfl
        have hq2 : ((FloatV.toRat c : ℚ) : ℝ) = c.toReal := rfl
        rw [hq1, hq2] at hcast
        have hsum : ((p + q : ℚ) : ℝ) - (a.toReal + c.toReal) =
            ((p : ℚ) : ℝ) + ((q : ℚ) : ℝ) - (a.toReal + c.toReal) := by push_cast; ring
        rw [hsum]
        exact hcast
      have hgoal : |((Format.binary32.toRat (Format.binary32.roundRat .rn false z (p + q)) : ℚ) : ℝ)
          - (a.toReal + c.toReal)| ≤
          2 * (2:ℝ)^(-24:ℤ) * ((1 + (2:ℝ)^(-24:ℤ)) * (|a.toReal| + |c.toReal|) +
            2 * (2:ℝ)^(-150:ℤ)) + 2 * (2:ℝ)^(-150:ℤ) +
            ((2:ℝ)^(-24:ℤ) * (|a.toReal| + |c.toReal|) + (2:ℝ)^(-150:ℤ)) := by
        have e1 : ((Format.binary32.toRat (Format.binary32.roundRat .rn false z (p + q)) : ℚ) : ℝ)
            - (a.toReal + c.toReal) =
            (((Format.binary32.toRat (Format.binary32.roundRat .rn false z (p + q)) : ℚ) : ℝ) - ((p+q:ℚ):ℝ))
              + (((p+q:ℚ):ℝ) - (a.toReal + c.toReal)) := by ring
        rw [e1]
        have h1 := abs_add_le (((Format.binary32.toRat (Format.binary32.roundRat .rn false z (p + q)) : ℚ) : ℝ) - ((p+q:ℚ):ℝ))
          (((p+q:ℚ):ℝ) - (a.toReal + c.toReal))
        have h2 : 2 * (2:ℝ)^(-24:ℤ) * |((p+q:ℚ):ℝ)| ≤
            2 * (2:ℝ)^(-24:ℤ) * ((1 + (2:ℝ)^(-24:ℤ)) * (|a.toReal| + |c.toReal|) +
              2 * (2:ℝ)^(-150:ℤ)) :=
          mul_le_mul_of_nonneg_left hpqR (by positivity)
        linarith [herr, hexactR, h1, h2]
      have hS : (0:ℝ) ≤ |a.toReal| + |c.toReal| := by linarith
      have h4 : 2 * ((2:ℝ)^(-24:ℤ) * ((2:ℝ)^(-24:ℤ) * (|a.toReal| + |c.toReal|))) ≤
          (2:ℝ)^(-24:ℤ) * (|a.toReal| + |c.toReal|) := by
        have hu2 : 2 * (2:ℝ)^(-24:ℤ) ≤ 1 := by
          have : (2:ℝ)^(-24:ℤ) = 1/16777216 := by norm_num
          rw [this]; norm_num
        calc 2 * ((2:ℝ)^(-24:ℤ) * ((2:ℝ)^(-24:ℤ) * (|a.toReal| + |c.toReal|)))
            = (2 * (2:ℝ)^(-24:ℤ)) * ((2:ℝ)^(-24:ℤ) * (|a.toReal| + |c.toReal|)) := by ring
          _ ≤ 1 * ((2:ℝ)^(-24:ℤ) * (|a.toReal| + |c.toReal|)) := by
              apply mul_le_mul_of_nonneg_right hu2
              exact mul_nonneg (le_of_lt hu0) hS
          _ = (2:ℝ)^(-24:ℤ) * (|a.toReal| + |c.toReal|) := one_mul _
      have h5 : 4 * ((2:ℝ)^(-24:ℤ) * (2:ℝ)^(-150:ℤ)) ≤ 5 * (2:ℝ)^(-150:ℤ) := by
        have hu4 : 4 * (2:ℝ)^(-24:ℤ) ≤ 5 := by
          have : (2:ℝ)^(-24:ℤ) = 1/16777216 := by norm_num
          rw [this]; norm_num
        calc 4 * ((2:ℝ)^(-24:ℤ) * (2:ℝ)^(-150:ℤ))
            = (4 * (2:ℝ)^(-24:ℤ)) * (2:ℝ)^(-150:ℤ) := by ring
          _ ≤ 5 * (2:ℝ)^(-150:ℤ) := mul_le_mul_of_nonneg_right hu4 (le_of_lt he0)
      nlinarith [hgoal, h4, h5, hS, hu0, he0]
    · exfalso; rw [(FloatV.isFinite_iff.mp hc).2] at hyi; cases hyi
    · exfalso; rw [(FloatV.isFinite_iff.mp ha).2] at hxi; cases hxi

/-- The sum of squared deviations is `Σ s² - D·mean²`. -/
theorem y_tol_sum_dev_sq {d : Inst} {init : Mem} {t : Nat} (hD : 0 < d.D) :
    (∑ j ∈ Finset.range d.D, (sR d init t j - meanRef d init t) ^ 2)
      = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2) - (d.D : ℝ) * meanRef d init t ^ 2 := by
  have hD' : (d.D : ℝ) ≠ 0 := by
    have : (0 : ℝ) < d.D := by exact_mod_cast hD
    exact ne_of_gt this
  have hsq : ∀ j : Nat, (sR d init t j - meanRef d init t) ^ 2
      = sR d init t j ^ 2 - 2 * meanRef d init t * sR d init t j + meanRef d init t ^ 2 := by
    intro j; ring
  calc (∑ j ∈ Finset.range d.D, (sR d init t j - meanRef d init t) ^ 2)
      = ∑ j ∈ Finset.range d.D, (sR d init t j ^ 2 - 2 * meanRef d init t * sR d init t j
          + meanRef d init t ^ 2) := Finset.sum_congr rfl (fun j _ => hsq j)
    _ = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2)
          - 2 * meanRef d init t * (∑ j ∈ Finset.range d.D, sR d init t j)
          + (d.D : ℝ) * meanRef d init t ^ 2 := by
        rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum,
          Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    _ = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2) - (d.D : ℝ) * meanRef d init t ^ 2 := by
        have hm : meanRef d init t = (∑ j ∈ Finset.range d.D, sR d init t j) / (d.D : ℝ) := rfl
        rw [hm]
        field_simp
        ring

/-- The squared root-mean-square equals the reference variance plus the squared reference mean. -/
theorem y_tol_rmsS_sq_add {d : Inst} {init : Mem} {t : Nat} (hD : 0 < d.D) :
    rmsS d init t ^ 2 = varRef d init t + meanRef d init t ^ 2 := by
  have hD' : (d.D : ℝ) ≠ 0 := by
    have : (0 : ℝ) < d.D := by exact_mod_cast hD
    exact ne_of_gt this
  rw [rstd_tol_rmsS_sq hD]
  unfold varRef
  rw [y_tol_sum_dev_sq hD]
  field_simp
  ring

/-- One squared deviation is at most `D·varRef`. -/
theorem y_tol_dj_sq_le {d : Inst} {init : Mem} {t j : Nat} (hD : 0 < d.D) (hj : j < d.D) :
    (sR d init t j - meanRef d init t) ^ 2 ≤ (d.D : ℝ) * varRef d init t := by
  have hD' : (d.D : ℝ) ≠ 0 := by
    have : (0 : ℝ) < d.D := by exact_mod_cast hD
    exact ne_of_gt this
  have hmem : j ∈ Finset.range d.D := Finset.mem_range.mpr hj
  have h1 : (sR d init t j - meanRef d init t) ^ 2
      ≤ ∑ k ∈ Finset.range d.D, (sR d init t k - meanRef d init t) ^ 2 :=
    Finset.single_le_sum (f := fun k => (sR d init t k - meanRef d init t) ^ 2)
      (fun k _ => sq_nonneg _) hmem
  have h2 : (d.D : ℝ) * varRef d init t
      = ∑ k ∈ Finset.range d.D, (sR d init t k - meanRef d init t) ^ 2 := by
    unfold varRef
    field_simp
  rw [h2]
  exact h1

/-- The normalised value `zRef` is at most `23` in magnitude. -/
theorem y_tol_zRef_abs_le {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D) :
    |zRef d init t j| ≤ 23 := by
  have hD := rstd_tol_D_pos hd
  have hDn : (0 : ℕ) < d.D := by exact_mod_cast hD
  have h1 : (sR d init t j - meanRef d init t) ^ 2 ≤ (d.D : ℝ) * varRef d init t :=
    y_tol_dj_sq_le hDn hj
  have h2 : rstdRef d init t ^ 2 * (varRef d init t + epsR d) = 1 := rstd_tol_R_sq_mul hd
  have hV0 : 0 ≤ varRef d init t := rstd_tol_varRef_nonneg
  have heps0 : 0 ≤ epsR d := le_of_lt (epsR_pos hd)
  have h3 : zRef d init t j ^ 2 ≤ (d.D : ℝ) := by
    have h4 : zRef d init t j ^ 2
        = (sR d init t j - meanRef d init t) ^ 2 * rstdRef d init t ^ 2 := by
      unfold zRef; ring
    rw [h4]
    have h5 : (sR d init t j - meanRef d init t) ^ 2 * rstdRef d init t ^ 2
        ≤ ((d.D : ℝ) * varRef d init t) * rstdRef d init t ^ 2 :=
      mul_le_mul_of_nonneg_right h1 (sq_nonneg _)
    have h7 : varRef d init t * rstdRef d init t ^ 2 ≤ 1 := by
      nlinarith [h2, hV0, heps0, sq_nonneg (rstdRef d init t)]
    have hD0 : (0 : ℝ) ≤ d.D := le_of_lt hD
    nlinarith [h5, h7, hD0]
  have h8 : |zRef d init t j| = Real.sqrt (zRef d init t j ^ 2) :=
    (Real.sqrt_sq_eq_abs _).symm
  rw [h8]
  have h9 : Real.sqrt (zRef d init t j ^ 2) ≤ Real.sqrt (d.D : ℝ) := Real.sqrt_le_sqrt h3
  have h10 : Real.sqrt (d.D : ℝ) ≤ 23 := by
    have h512 : (d.D : ℝ) ≤ 512 := by exact_mod_cast hd.d_le
    calc Real.sqrt (d.D : ℝ) ≤ Real.sqrt 512 := Real.sqrt_le_sqrt h512
      _ ≤ 23 := by
          nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 512 by norm_num),
            Real.sqrt_nonneg (512 : ℝ)]
  linarith

/-- `gam ≤ 2^-10`. -/
theorem y_tol_gam_le {d : Inst} (hd : d.ok) : gam d ≤ 2 ^ (-10 : ℤ) := by
  have hL := rstd_tol_depth_le hd
  have hpos := epsR_pos hd
  have hge := epsR_ge hd
  have hd10 : (depth d : ℝ) + 10 ≤ 20 := by
    have h : (depth d : ℝ) ≤ 10 := by exact_mod_cast hL
    linarith
  have h23 : (0 : ℝ) < 2 ^ (-23 : ℤ) := by norm_num
  have h1 : ((depth d : ℝ) + 10) * 2 ^ (-23 : ℤ) ≤ 20 * 2 ^ (-23 : ℤ) :=
    mul_le_mul_of_nonneg_right hd10 (le_of_lt h23)
  have h2 : (2 : ℝ) ^ (-136 : ℤ) / epsR d ≤ (2 : ℝ) ^ (-136 : ℤ) / (2 : ℝ) ^ (-125 : ℤ) :=
    div_le_div_of_nonneg_left (by norm_num) (by norm_num) hge
  have h3 : (2 : ℝ) ^ (-136 : ℤ) / (2 : ℝ) ^ (-125 : ℤ) = (2 : ℝ) ^ (-11 : ℤ) := by
    rw [← zpow_sub₀ (by norm_num : (2 : ℝ) ≠ 0)]
    norm_num
  unfold gam
  linarith [h1, h2, h3]

/-- For an accepted instance the constant `gam` is nonnegative. -/
theorem y_tol_gam_nonneg {d : Inst} (hd : d.ok) : 0 ≤ gam d := by
  have hpos := epsR_pos hd
  have h1 : (0 : ℝ) ≤ ((depth d : ℝ) + 10) * 2 ^ (-23 : ℤ) := by positivity
  have h2 : (0 : ℝ) ≤ (2 : ℝ) ^ (-136 : ℤ) / epsR d :=
    div_nonneg (by norm_num) (le_of_lt hpos)
  unfold gam
  linarith

/-- The condition number is nonnegative. -/
theorem y_tol_cond_nonneg {d : Inst} {init : Mem} {t : Nat} : 0 ≤ cond d init t := by
  have h1 : (0 : ℝ) ≤ meanTol d init t := rstd_tol_meanTol_nonneg
  have h2 : (0 : ℝ) ≤ absMean d init t := rstd_tol_absMean_nonneg
  have h3 : (0 : ℝ) ≤ rmsS d init t := rstd_tol_rmsS_nonneg
  unfold cond
  have hR : 0 ≤ rstdRef d init t := by
    unfold rstdRef; positivity
  have h4 : (0 : ℝ) ≤ meanTol d init t + 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t) := by
    have : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) := by norm_num
    nlinarith
  exact mul_nonneg h4 hR

/-- `rstdTol ≤ 2^64`. -/
theorem y_tol_rstdTol_le {d : Inst} (hd : d.ok) {init : Mem} {t : Nat} :
    rstdTol d init t ≤ 2 ^ (64 : ℤ) := by
  have hR := rstd_tol_R_le (init := init) (t := t) hd
  have hR0 := rstd_tol_R_pos (init := init) (t := t) hd
  have hspos := epsR_pos hd
  by_cases h : 4 * cond d init t ≤ 1
  · rw [rstdTol, ite_eq_left h]
    have hg := y_tol_gam_le hd
    have hg0 := y_tol_gam_nonneg hd
    have hgv : (2 : ℝ) ^ (-10 : ℤ) = 1 / 1024 := by norm_num
    have hg' : gam d ≤ 1 / 1024 := le_trans hg (le_of_eq hgv)
    have h1 : 2 * (cond d init t + gam d) ≤ 1 := by linarith
    have h2 : 2 * (cond d init t + gam d) * rstdRef d init t ≤ 1 * rstdRef d init t :=
      mul_le_mul_of_nonneg_right h1 (le_of_lt hR0)
    have h3 : (1 : ℝ) * rstdRef d init t ≤ 2 ^ (63 : ℤ) := by rw [one_mul]; exact hR
    have h4 : (2 : ℝ) ^ (63 : ℤ) ≤ 2 ^ (64 : ℤ) := by norm_num
    linarith
  · rw [rstdTol, ite_eq_right h]
    have hge := epsR_ge hd
    have hspos' : 0 < Real.sqrt (epsR d) := Real.sqrt_pos.mpr hspos
    rw [div_le_iff₀ hspos']
    have h63 : (2 : ℝ) ^ (-63 : ℤ) ≤ Real.sqrt (epsR d) := by
      refine Real.le_sqrt_of_sq_le ?_
      have hsq : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 = (2 : ℝ) ^ (-126 : ℤ) := by
        rw [pow_two, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
        norm_num
      rw [hsq]
      have hle : (2 : ℝ) ^ (-126 : ℤ) ≤ (2 : ℝ) ^ (-125 : ℤ) := by norm_num
      linarith
    have hmul : (2 : ℝ) ^ (64 : ℤ) * 2 ^ (-63 : ℤ) = 2 := by norm_num
    have hmono : (2 : ℝ) ^ (64 : ℤ) * 2 ^ (-63 : ℤ) ≤ (2 : ℝ) ^ (64 : ℤ) * Real.sqrt (epsR d) :=
      mul_le_mul_of_nonneg_left h63 (by positivity)
    linarith

/-- When `4·cond ≤ 1`, `rstdTol ≤ rstdRef`. -/
theorem y_tol_RT_le_R {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    (h : 4 * cond d init t ≤ 1) : rstdTol d init t ≤ rstdRef d init t := by
  rw [rstdTol, ite_eq_left h]
  have hg := y_tol_gam_le hd
  have hg0 := y_tol_gam_nonneg hd
  have hR0 := rstd_tol_R_pos (init := init) (t := t) hd
  have hgv : (2 : ℝ) ^ (-10 : ℤ) = 1 / 1024 := by norm_num
  have hg' : gam d ≤ 1 / 1024 := le_trans hg (le_of_eq hgv)
  have h1 : 2 * (cond d init t + gam d) ≤ 1 := by linarith
  calc 2 * (cond d init t + gam d) * rstdRef d init t
      = (2 * (cond d init t + gam d)) * rstdRef d init t := by ring
    _ ≤ 1 * rstdRef d init t := mul_le_mul_of_nonneg_right h1 (le_of_lt hR0)
    _ = rstdRef d init t := one_mul _

/-- `rstdTol` is at least `20u·rstdRef`. -/
theorem y_tol_rstdTol_low {d : Inst} (hd : d.ok) {init : Mem} {t : Nat} :
    20 * u32 * rstdRef d init t ≤ rstdTol d init t := by
  have hR0 := rstd_tol_R_pos (init := init) (t := t) hd
  have hpos := epsR_pos hd
  have hspos : 0 < Real.sqrt (epsR d) := Real.sqrt_pos.mpr hpos
  by_cases h : 4 * cond d init t ≤ 1
  · rw [rstdTol, ite_eq_left h]
    have hg : 10 * (2 : ℝ) ^ (-23 : ℤ) ≤ gam d := by
      unfold gam
      have h1 : (10 : ℝ) * 2 ^ (-23 : ℤ) ≤ ((depth d : ℝ) + 10) * 2 ^ (-23 : ℤ) := by
        have h0 : (0 : ℝ) ≤ (depth d : ℝ) := by positivity
        nlinarith [show (0 : ℝ) < 2 ^ (-23 : ℤ) by norm_num]
      have h2 : (0 : ℝ) ≤ (2 : ℝ) ^ (-136 : ℤ) / epsR d :=
        div_nonneg (by norm_num) (le_of_lt hpos)
      linarith
    have h2 : 20 * u32 * rstdRef d init t ≤ 2 * gam d * rstdRef d init t := by
      have h20 : 20 * u32 = 10 * (2 : ℝ) ^ (-23 : ℤ) := by unfold u32; norm_num
      have hgR : 10 * (2 : ℝ) ^ (-23 : ℤ) * rstdRef d init t ≤ gam d * rstdRef d init t :=
        mul_le_mul_of_nonneg_right hg (le_of_lt hR0)
      have hg0 := y_tol_gam_nonneg hd
      have hgg : gam d * rstdRef d init t ≤ 2 * gam d * rstdRef d init t := by
        nlinarith [hg0, hR0]
      rw [h20]
      linarith [hgR, hgg]
    have h3 : 2 * gam d * rstdRef d init t
        ≤ 2 * (cond d init t + gam d) * rstdRef d init t := by
      have hc := y_tol_cond_nonneg (d := d) (init := init) (t := t)
      nlinarith [hR0]
    linarith
  · rw [rstdTol, ite_eq_right h]
    have hV := rstd_tol_varRef_nonneg (d := d) (init := init) (t := t)
    have hRle : rstdRef d init t ≤ 1 / Real.sqrt (epsR d) := by
      unfold rstdRef
      apply one_div_le_one_div_of_le (a := Real.sqrt (epsR d))
        (b := Real.sqrt (varRef d init t + epsR d)) hspos
      exact Real.sqrt_le_sqrt (by linarith)
    have h20 : 20 * u32 ≤ 2 := by unfold u32; norm_num
    have h2 : 20 * u32 * rstdRef d init t ≤ 20 * u32 * (1 / Real.sqrt (epsR d)) := by
      have h0 : (0 : ℝ) ≤ 20 * u32 := by unfold u32; norm_num
      exact mul_le_mul_of_nonneg_left hRle h0
    have h3 : 20 * u32 * (1 / Real.sqrt (epsR d)) ≤ 2 * (1 / Real.sqrt (epsR d)) := by
      have h0 : (0 : ℝ) ≤ 1 / Real.sqrt (epsR d) := by positivity
      nlinarith [h20, h0]
    have h4 : 2 * (1 / Real.sqrt (epsR d)) = 2 / Real.sqrt (epsR d) := by ring
    linarith [h2, h3, h4]



/-! ### A conversion of a finite `float32` value in range -/

/-! ### A `float32 -> bfloat16` conversion in range -/

/-- The `bfloat16` conversion of a finite `float32` value of magnitude at most `255·2^120`,
matched by `NaNEq`: finite, and within `u16·|x| + η16`. -/
lemma y_tol_cvt_res {x : F32} {r : BF16} (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r)
    (hx : x.isFinite = true) (hmax : |x.toReal| ≤ (255 : ℝ) * 2 ^ (120 : ℤ)) :
    r.isFinite = true ∧ |r.toReal - x.toReal| ≤ u16 * |x.toReal| + eta16 := by
  have hmaxq : |x.toRat| ≤ Format.bfloat16.maxVal := by
    rw [Format.bfloat16_maxVal]
    have hm : ((|x.toRat| : ℚ) : ℝ) ≤ (255 : ℝ) * 2 ^ (120 : ℤ) := by
      rw [Rat.cast_abs]
      exact hmax
    have h2 : ((255 : ℝ) * 2 ^ (120 : ℤ)) = (((2 ^ 8 - 1) * 2 ^ 120 : ℚ) : ℝ) := by
      push_cast
      norm_num
    have h3 : ((|x.toRat| : ℚ) : ℝ) ≤ (((2 ^ 8 - 1) * 2 ^ 120 : ℚ) : ℝ) := by
      rw [← h2]
      exact hm
    exact_mod_cast h3
  obtain ⟨hfin, herr⟩ :=
    FloatV.cvt_err (f := Format.binary32) (g := Format.bfloat16) Format.bfloat16_regular hx hmaxq
  have heq : r = (FloatV.cvt x : BF16) :=
    FloatV.NaNEq.eq (by simpa only [FloatV.cvtWith_rn] using h)
      (FloatV.isFinite_iff.mp hfin).1
  subst heq
  refine ⟨hfin, ?_⟩
  have hcast : ((|(FloatV.cvt x : BF16).toRat - x.toRat| : ℚ) : ℝ) ≤
      ((|x.toRat| * Format.bfloat16.u + Format.bfloat16.eta : ℚ) : ℝ) := by
    exact_mod_cast herr
  push_cast at hcast
  rw [Format.bfloat16_u, Format.bfloat16_eta] at hcast
  push_cast at hcast
  unfold u16 eta16 FloatV.toReal
  linarith [hcast]


/-! ### Nonnegativity of the tolerances -/

lemma y_tol_eTol_nonneg {d : Inst} {init : Mem} {t j : Nat} : 0 ≤ eTol d init t j := by
  have h1 : (0 : ℝ) ≤ meanTol d init t := rstd_tol_meanTol_nonneg
  have h2 : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) := by norm_num
  have h3 : (0 : ℝ) ≤ |sR d init t j| + absMean d init t := by
    have := rstd_tol_absMean_nonneg (d := d) (init := init) (t := t)
    have := abs_nonneg (sR d init t j)
    linarith
  unfold eTol
  positivity

/-- For an accepted instance the rstd tolerance of a row is positive. -/
lemma y_tol_rstdTol_pos {d : Inst} (hd : d.ok) {init : Mem} {t : Nat} :
    0 < rstdTol d init t := by
  have hR := rstd_tol_R_pos (init := init) (t := t) hd
  have h := y_tol_rstdTol_low (d := d) (init := init) (t := t) hd
  have h0 : (0 : ℝ) < 20 * u32 := by unfold u32; norm_num
  nlinarith

/-- `zTol ≥ 0`. -/
lemma y_tol_zTol_nonneg {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} :
    0 ≤ zTol d init t j := by
  have h1 : (0 : ℝ) ≤ |sR d init t j - meanRef d init t| := abs_nonneg _
  have h2 : (0 : ℝ) ≤ rstdTol d init t := le_of_lt (y_tol_rstdTol_pos hd)
  have h3 : (0 : ℝ) ≤ eTol d init t j := y_tol_eTol_nonneg
  have h4 : (0 : ℝ) ≤ rstdRef d init t := le_of_lt (rstd_tol_R_pos (init := init) (t := t) hd)
  unfold zTol
  have h5 : (0 : ℝ) ≤ |sR d init t j - meanRef d init t| * rstdTol d init t :=
    mul_nonneg h1 h2
  have h6 : (0 : ℝ) ≤ eTol d init t j * (rstdRef d init t + rstdTol d init t) :=
    mul_nonneg h3 (by linarith)
  linarith

/-! ### Elementary bounds on the row's constants -/

/-- `meanTol` in terms of `absMean`. -/
lemma y_tol_meanTol_le {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    (_hbdd : ∀ j, j < d.D → |sR d init t j| ≤ 2 ^ 57) :
    meanTol d init t ≤ 28 * u32 * absMean d init t + eta32 * 2 ^ 10 := by
  have hL := rstd_tol_depth_le hd
  have hA0 : (0 : ℝ) ≤ absMean d init t := rstd_tol_absMean_nonneg
  have h14 : ((depth d : ℝ) + 4) ≤ 14 := by
    have : (depth d : ℝ) ≤ 10 := by exact_mod_cast hL
    linarith
  have h23 : (0 : ℝ) < 2 ^ (-23 : ℤ) := by norm_num
  have h28 : (14 : ℝ) * 2 ^ (-23 : ℤ) = 28 * u32 := by unfold u32; norm_num
  have h140 : (2 : ℝ) ^ (-140 : ℤ) = eta32 * 2 ^ 10 := by unfold eta32; norm_num
  unfold meanTol
  rw [← h140]
  have h1 : ((depth d : ℝ) + 4) * 2 ^ (-23 : ℤ) * absMean d init t
      ≤ 14 * 2 ^ (-23 : ℤ) * absMean d init t := by
    have := mul_le_mul_of_nonneg_right h14 (le_of_lt h23)
    exact mul_le_mul_of_nonneg_right this hA0
  have h28' : (14 : ℝ) * 2 ^ (-23 : ℤ) * absMean d init t = 28 * u32 * absMean d init t := by
    rw [h28]
  linarith [h1, h28']

set_option maxHeartbeats 1600000

/-! ### `FloatV.neg` -/

theorem y_tol_neg_toReal (x : F32) : (FloatV.neg x).toReal = -x.toReal := by
  show (((FloatV.neg x).toRat : ℚ) : ℝ) = -(((x.toRat : ℚ)) : ℝ)
  rw [FloatV.toRat_neg]
  push_cast
  ring

/-- The negation of a finite float32 is finite. -/
theorem y_tol_neg_fin {x : F32} (h : x.isFinite = true) : (FloatV.neg x).isFinite = true := by
  rw [FloatV.isFinite_neg]
  exact h

/-! ### The deviation of `bx` from `sR j - meanRef` -/

/-- The `float32` deviation `x - mean` is within `2·eTol` of the exact deviation. -/
theorem y_tol_ebx {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat}
    {bx0 bmean bx : F32}
    (hval : bx0.toReal = sR d init t j) (hbx0fin : bx0.isFinite = true)
    (hmeanfin : bmean.isFinite = true)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hdev : AddRel true bx0 (FloatV.neg bmean) bx) (hbxfin : bx.isFinite = true) :
    |bx.toReal - (sR d init t j - meanRef d init t)| ≤ 2 * eTol d init t j := by
  have hD := rstd_tol_D_pos hd
  have hDn : 0 < d.D := by exact_mod_cast hD
  have hneg : (FloatV.neg bmean).isFinite = true := y_tol_neg_fin hmeanfin
  have hnegR : (FloatV.neg bmean).toReal = -bmean.toReal := y_tol_neg_toReal bmean
  have hadd_err := y_tol_addrel_rel hbx0fin hneg hdev hbxfin
  have h1 : |bx.toReal - (sR d init t j - bmean.toReal)| ≤
      4 * u32 * (|sR d init t j| + |bmean.toReal|) + 8 * eta32 := by
    have hsub : bx0.toReal + (FloatV.neg bmean).toReal
        = sR d init t j - bmean.toReal := by
      rw [hnegR, hval]; ring
    simpa only [u32, eta32, hsub, hval, hnegR, abs_neg, sub_eq_add_neg] using hadd_err
  have hmeanbd : |bmean.toReal| ≤ absMean d init t + meanTol d init t := by
    have hA := abs_meanRef_le_absMean (d := d) (init := init) (t := t) hDn
    have h2 : |bmean.toReal| ≤ |meanRef d init t| + meanTol d init t := by
      have h3 := abs_sub_abs_le_abs_sub bmean.toReal (meanRef d init t)
      linarith [hmean]
    linarith [hA]
  have hMT0 : (0 : ℝ) ≤ meanTol d init t := rstd_tol_meanTol_nonneg
  have hA0 : (0 : ℝ) ≤ absMean d init t := rstd_tol_absMean_nonneg
  have hs0 : (0 : ℝ) ≤ |sR d init t j| := abs_nonneg _
  have hMTge : (2 : ℝ) ^ (-140 : ℤ) ≤ meanTol d init t := by
    have h0 : (0 : ℝ) ≤ ((depth d : ℝ) + 4) * 2 ^ (-23 : ℤ) * absMean d init t := by
      have hd0 : (0 : ℝ) ≤ (depth d : ℝ) := by positivity
      have h23 : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) := by norm_num
      exact mul_nonneg (mul_nonneg (by linarith) h23) hA0
    unfold meanTol
    linarith
  have h4u : (4 : ℝ) * u32 = 2 * 2 ^ (-23 : ℤ) := by unfold u32; norm_num
  have hstep : |bx.toReal - (sR d init t j - meanRef d init t)| ≤
      4 * u32 * (|sR d init t j| + absMean d init t) + 4 * u32 * meanTol d init t
        + 8 * eta32 + meanTol d init t := by
    have h2 : |sR d init t j| + |bmean.toReal|
        ≤ |sR d init t j| + (absMean d init t + meanTol d init t) := by linarith
    have h3 : (0 : ℝ) ≤ 4 * u32 := by unfold u32; norm_num
    have h4 := mul_le_mul_of_nonneg_left h2 h3
    have hm' : |meanRef d init t - bmean.toReal| ≤ meanTol d init t := by
      rw [abs_sub_comm]; exact hmean
    have h5 : |bx.toReal - (sR d init t j - meanRef d init t)| ≤
        |bx.toReal - (sR d init t j - bmean.toReal)| + |meanRef d init t - bmean.toReal| := by
      have he : bx.toReal - (sR d init t j - meanRef d init t)
          = (bx.toReal - (sR d init t j - bmean.toReal)) + (meanRef d init t - bmean.toReal) := by
        ring
      rw [he]
      exact abs_add_le _ _
    nlinarith [h1, h4, h5, hm', hMT0]
  have hstep2 : 4 * u32 * (|sR d init t j| + absMean d init t)
      + 4 * u32 * meanTol d init t + 8 * eta32 + meanTol d init t
      ≤ 2 * eTol d init t j := by
    have h4 : 4 * u32 * (|sR d init t j| + absMean d init t)
        = 2 * (2 ^ (-23 : ℤ) * (|sR d init t j| + absMean d init t)) := by
      rw [show (4 : ℝ) * u32 = 2 * 2 ^ (-23 : ℤ) by unfold u32; norm_num]
      ring
    have h5 : 4 * u32 * meanTol d init t ≤ 2 ^ (-21 : ℤ) * meanTol d init t := by
      have h : (4 : ℝ) * u32 ≤ 2 ^ (-21 : ℤ) := by unfold u32; norm_num
      exact mul_le_mul_of_nonneg_right h hMT0
    have h6 : 8 * eta32 ≤ 2 ^ (-7 : ℤ) * meanTol d init t := by
      have h : (8 : ℝ) * eta32 = 2 ^ (-7 : ℤ) * 2 ^ (-140 : ℤ) := by unfold eta32; norm_num
      rw [h]
      exact mul_le_mul_of_nonneg_left hMTge (by norm_num)
    have h7 : (2 : ℝ) ^ (-21 : ℤ) + 2 ^ (-7 : ℤ) + 1 ≤ 2 := by norm_num
    have h8 := mul_le_mul_of_nonneg_right h7 hMT0
    have hev : eTol d init t j
        = meanTol d init t + 2 ^ (-23 : ℤ) * (|sR d init t j| + absMean d init t) := rfl
    rw [hev]
    nlinarith [h4, h5, h6, h8]
  linarith [hstep, hstep2]

/-! ### The deviation of `bhat` from `zRef` -/

/-- The `float32` normalised value `(x - mean)·rstd` is within `7/2·zTol` (plus a tiny term) of `zRef`. -/
theorem y_tol_bhat_delta {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (_hj : j < d.D)
    {bx brstd bhat : F32}
    (hbxfin : bx.isFinite = true)
    (hebx : |bx.toReal - (sR d init t j - meanRef d init t)| ≤ 2 * eTol d init t j)
    (hrstdfin : brstd.isFinite = true)
    (hrstd : |brstd.toReal - rstdRef d init t| ≤ rstdTol d init t)
    (hmul1 : FloatV.NaNEq (FloatV.mul bx brstd) bhat) (hbhatfin : bhat.isFinite = true) :
    |bhat.toReal - zRef d init t j| ≤ 7 / 2 * zTol d init t j + 1024 * eta32 := by
  have hmul := y_tol_mul_rel hbxfin hrstdfin hmul1 hbhatfin
  have hmul' : |bhat.toReal - bx.toReal * brstd.toReal| ≤
      2 * u32 * |bx.toReal * brstd.toReal| + 2 * eta32 := by
    simpa only [u32, eta32] using hmul
  have hR0 : 0 < rstdRef d init t := rstd_tol_R_pos (init := init) (t := t) hd
  have hRT0 : 0 ≤ rstdTol d init t := le_of_lt (y_tol_rstdTol_pos hd)
  have hE0 : 0 ≤ eTol d init t j := y_tol_eTol_nonneg
  have heta : 2 * eta32 ≤ 1024 * eta32 := by unfold eta32; norm_num
  have hzR : zRef d init t j = (sR d init t j - meanRef d init t) * rstdRef d init t := rfl
  -- absolute-value bounds on the operands
  have hbx_abs : |bx.toReal|
      ≤ |sR d init t j - meanRef d init t| + 2 * eTol d init t j := by
    have he : bx.toReal = (bx.toReal - (sR d init t j - meanRef d init t))
        + (sR d init t j - meanRef d init t) := by ring
    have h := abs_add_le (bx.toReal - (sR d init t j - meanRef d init t))
      (sR d init t j - meanRef d init t)
    rw [he]
    linarith [h, hebx]
  have hRabs : |rstdRef d init t| = rstdRef d init t := abs_of_nonneg (le_of_lt hR0)
  have hrstd_abs : |brstd.toReal| ≤ rstdRef d init t + rstdTol d init t := by
    have he : brstd.toReal = (brstd.toReal - rstdRef d init t) + rstdRef d init t := by ring
    have h := abs_add_le (brstd.toReal - rstdRef d init t) (rstdRef d init t)
    rw [hRabs] at h
    rw [he]
    linarith [h, hrstd]
  -- decomposition
  have hdecomp : bhat.toReal - zRef d init t j
      = (bhat.toReal - bx.toReal * brstd.toReal) + bx.toReal * (brstd.toReal - rstdRef d init t)
        + (bx.toReal - (sR d init t j - meanRef d init t)) * rstdRef d init t := by
    rw [hzR]; ring
  have htri : |bhat.toReal - zRef d init t j| ≤
      (2 * u32 * |bx.toReal * brstd.toReal| + 2 * eta32)
        + |bx.toReal| * |brstd.toReal - rstdRef d init t|
        + |bx.toReal - (sR d init t j - meanRef d init t)| * rstdRef d init t := by
    rw [hdecomp]
    have h1 : |(bhat.toReal - bx.toReal * brstd.toReal)
        + bx.toReal * (brstd.toReal - rstdRef d init t)|
        ≤ |bhat.toReal - bx.toReal * brstd.toReal|
          + |bx.toReal| * |brstd.toReal - rstdRef d init t| := by
      have h := abs_add_le (bhat.toReal - bx.toReal * brstd.toReal)
        (bx.toReal * (brstd.toReal - rstdRef d init t))
      rw [abs_mul] at h
      linarith [h]
    have h2 : |((bhat.toReal - bx.toReal * brstd.toReal)
        + bx.toReal * (brstd.toReal - rstdRef d init t))
        + (bx.toReal - (sR d init t j - meanRef d init t)) * rstdRef d init t|
        ≤ (|bhat.toReal - bx.toReal * brstd.toReal|
          + |bx.toReal| * |brstd.toReal - rstdRef d init t|)
          + |bx.toReal - (sR d init t j - meanRef d init t)| * rstdRef d init t := by
      have h := abs_add_le ((bhat.toReal - bx.toReal * brstd.toReal)
        + bx.toReal * (brstd.toReal - rstdRef d init t))
        ((bx.toReal - (sR d init t j - meanRef d init t)) * rstdRef d init t)
      have h3 : |(bx.toReal - (sR d init t j - meanRef d init t)) * rstdRef d init t|
          = |bx.toReal - (sR d init t j - meanRef d init t)| * rstdRef d init t := by
        rw [abs_mul, abs_of_nonneg (le_of_lt hR0)]
      rw [h3] at h
      linarith [h]
    linarith [h1, h2, hmul']
  -- the four products
  have hprod : |bx.toReal * brstd.toReal|
      ≤ (|sR d init t j - meanRef d init t| + 2 * eTol d init t j)
        * (rstdRef d init t + rstdTol d init t) := by
    rw [abs_mul]
    have h0 : (0 : ℝ) ≤ |sR d init t j - meanRef d init t| + 2 * eTol d init t j := by
      linarith [abs_nonneg (sR d init t j - meanRef d init t), hE0]
    exact mul_le_mul hbx_abs hrstd_abs (abs_nonneg _) h0
  have hDv0 : (0 : ℝ) ≤ |sR d init t j - meanRef d init t| := abs_nonneg _
  have hDvle : |sR d init t j - meanRef d init t| ≤ 2 ^ (23 : ℤ) * eTol d init t j := by
    have hA0 : (0 : ℝ) ≤ absMean d init t := rstd_tol_absMean_nonneg
    have h1 : |sR d init t j - meanRef d init t| ≤ |sR d init t j| + absMean d init t := by
      have h2 : |sR d init t j - meanRef d init t|
          ≤ |sR d init t j| + |meanRef d init t| := by
        have := abs_add_le (sR d init t j) (-meanRef d init t)
        simpa only [sub_eq_add_neg, abs_neg] using this
      have hDn : 0 < d.D := by
        have h := rstd_tol_D_pos hd
        exact_mod_cast h
      have h3 := abs_meanRef_le_absMean (d := d) (init := init) (t := t) hDn
      linarith
    have h4 : 2 ^ (-23 : ℤ) * (|sR d init t j| + absMean d init t)
        ≤ eTol d init t j := by
      have hev : eTol d init t j
          = meanTol d init t + 2 ^ (-23 : ℤ) * (|sR d init t j| + absMean d init t) := rfl
      rw [hev]
      have := rstd_tol_meanTol_nonneg (d := d) (init := init) (t := t)
      linarith
    have h4' : |sR d init t j| + absMean d init t
        ≤ 2 ^ (23 : ℤ) * eTol d init t j := by
      have hpos : (0 : ℝ) ≤ 2 ^ (23 : ℤ) := by norm_num
      have hm := mul_le_mul_of_nonneg_left h4 hpos
      have he : (2 : ℝ) ^ (23 : ℤ) * (2 ^ (-23 : ℤ) * (|sR d init t j| + absMean d init t))
          = |sR d init t j| + absMean d init t := by
        rw [← mul_assoc, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
        norm_num
      linarith [hm, he]
    linarith [h1, h4']
  have hz20 : 20 * u32 * |zRef d init t j| ≤ zTol d init t j := by
    have hlow := y_tol_rstdTol_low (d := d) (init := init) (t := t) hd
    have h1 : 20 * u32 * rstdRef d init t * |sR d init t j - meanRef d init t|
        ≤ rstdTol d init t * |sR d init t j - meanRef d init t| :=
      mul_le_mul_of_nonneg_right hlow hDv0
    have h2 : zTol d init t j = |sR d init t j - meanRef d init t| * rstdTol d init t
        + eTol d init t j * (rstdRef d init t + rstdTol d init t) := rfl
    have h3 : 20 * u32 * |zRef d init t j|
        = 20 * u32 * rstdRef d init t * |sR d init t j - meanRef d init t| := by
      rw [hzR, abs_mul, abs_of_nonneg (le_of_lt hR0)]
      ring
    have h4 : |sR d init t j - meanRef d init t| * rstdTol d init t
        ≤ zTol d init t j := by
      rw [h2]
      have h5 : (0 : ℝ) ≤ eTol d init t j * (rstdRef d init t + rstdTol d init t) :=
        mul_nonneg hE0 (by linarith)
      linarith
    linarith [h1, h3, h4]
  have hztol_def : zTol d init t j = |sR d init t j - meanRef d init t| * rstdTol d init t
      + eTol d init t j * (rstdRef d init t + rstdTol d init t) := rfl
  have hDvRT : |sR d init t j - meanRef d init t| * rstdTol d init t ≤ zTol d init t j := by
    rw [hztol_def]
    have h5 : (0 : ℝ) ≤ eTol d init t j * (rstdRef d init t + rstdTol d init t) :=
      mul_nonneg hE0 (by linarith)
    linarith
  -- now assemble
  have hb1 : 2 * u32 * |bx.toReal * brstd.toReal|
      ≤ 2 * u32 * ((|sR d init t j - meanRef d init t| + 2 * eTol d init t j)
        * (rstdRef d init t + rstdTol d init t)) :=
    mul_le_mul_of_nonneg_left hprod (by unfold u32; norm_num)
  have hb2 : |bx.toReal| * |brstd.toReal - rstdRef d init t|
      ≤ (|sR d init t j - meanRef d init t| + 2 * eTol d init t j) * rstdTol d init t := by
    have h0 : (0 : ℝ) ≤ |sR d init t j - meanRef d init t| + 2 * eTol d init t j := by
      linarith [abs_nonneg (sR d init t j - meanRef d init t), hE0]
    exact mul_le_mul hbx_abs hrstd (abs_nonneg _) h0
  have hb3 : |bx.toReal - (sR d init t j - meanRef d init t)| * rstdRef d init t
      ≤ 2 * eTol d init t j * rstdRef d init t :=
    mul_le_mul_of_nonneg_right hebx (le_of_lt hR0)
  -- coefficient arithmetic
  have hcoef : 2 * u32 * ((|sR d init t j - meanRef d init t| + 2 * eTol d init t j)
        * (rstdRef d init t + rstdTol d init t))
      + (|sR d init t j - meanRef d init t| + 2 * eTol d init t j) * rstdTol d init t
      + 2 * eTol d init t j * rstdRef d init t + 2 * eta32
      ≤ 7 / 2 * zTol d init t j + 1024 * eta32 := by
    -- split into the terms of zTol
    have hu0 : (0 : ℝ) ≤ u32 := le_of_lt u32_pos
    have hz0 : (0 : ℝ) ≤ zTol d init t j := y_tol_zTol_nonneg hd
    -- u * |Dv| * R = u * |zRef|
    have hzRabs : |zRef d init t j| = |sR d init t j - meanRef d init t| * rstdRef d init t := by
      rw [hzR, abs_mul, abs_of_nonneg (le_of_lt hR0)]
    -- components
    have hc1 : 2 * u32 * (|sR d init t j - meanRef d init t| * rstdRef d init t)
        ≤ (1 / 10) * zTol d init t j := by
      have h := hz20
      rw [hzRabs] at h
      nlinarith [h]
    have hc2 : 2 * u32 * (|sR d init t j - meanRef d init t| * rstdTol d init t)
        ≤ 2 * u32 * zTol d init t j :=
      mul_le_mul_of_nonneg_left hDvRT (by unfold u32; norm_num)
    have hc3 : 2 * u32 * (2 * eTol d init t j * (rstdRef d init t + rstdTol d init t))
        ≤ 4 * u32 * zTol d init t j := by
      have h1 : eTol d init t j * (rstdRef d init t + rstdTol d init t) ≤ zTol d init t j := by
        rw [hztol_def]
        nlinarith [hDvRT]
      have h2 : 2 * u32 * (2 * (eTol d init t j * (rstdRef d init t + rstdTol d init t)))
          ≤ 2 * u32 * (2 * zTol d init t j) :=
        mul_le_mul_of_nonneg_left (by linarith) (by unfold u32; norm_num)
      nlinarith [h2]
    have hc4 : (|sR d init t j - meanRef d init t|) * rstdTol d init t
        ≤ zTol d init t j := hDvRT
    have hc5 : 2 * eTol d init t j * rstdTol d init t ≤ 2 * eTol d init t j * (rstdRef d init t + rstdTol d init t) := by
      have h : rstdTol d init t ≤ rstdRef d init t + rstdTol d init t := by linarith
      nlinarith [hE0, h]
    have hc6 : 2 * eTol d init t j * (rstdRef d init t + rstdTol d init t) ≤ 2 * zTol d init t j := by
      have h1 : eTol d init t j * (rstdRef d init t + rstdTol d init t) ≤ zTol d init t j := by
        rw [hztol_def]
        nlinarith [hDvRT]
      nlinarith [h1]
    have hexp : 2 * u32 * ((|sR d init t j - meanRef d init t| + 2 * eTol d init t j)
          * (rstdRef d init t + rstdTol d init t))
        = 2 * u32 * (|sR d init t j - meanRef d init t| * rstdRef d init t)
          + 2 * u32 * (|sR d init t j - meanRef d init t| * rstdTol d init t)
          + 2 * u32 * (2 * eTol d init t j * (rstdRef d init t + rstdTol d init t)) := by
      ring
    have hexp2 : (|sR d init t j - meanRef d init t| + 2 * eTol d init t j) * rstdTol d init t
        = |sR d init t j - meanRef d init t| * rstdTol d init t
          + 2 * eTol d init t j * rstdTol d init t := by
      ring
    rw [hexp, hexp2]
    have hu_small : 6 * u32 ≤ 4 / 10 := by unfold u32; norm_num
    have hcu : 6 * u32 * zTol d init t j ≤ 4 / 10 * zTol d init t j :=
      mul_le_mul_of_nonneg_right hu_small hz0
    have hc23 : 2 * u32 * zTol d init t j + 4 * u32 * zTol d init t j
        ≤ 4 / 10 * zTol d init t j := by linarith [hcu]
    linarith [hc1, hc2, hc3, hc4, hc5, hc6, heta, hc23]
  linarith [htri, hb1, hb2, hb3, hcoef]


/-! ### The output's tolerances -/

/-- `zTol` is at least `20u·|zRef|`. -/
lemma y_tol_zTol_low {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} :
    20 * u32 * |zRef d init t j| ≤ zTol d init t j := by
  have hR0 : 0 < rstdRef d init t := rstd_tol_R_pos (init := init) (t := t) hd
  have hE0 : 0 ≤ eTol d init t j := y_tol_eTol_nonneg
  have hDv0 : (0:ℝ) ≤ |sR d init t j - meanRef d init t| := abs_nonneg _
  have hlow := y_tol_rstdTol_low (d := d) (init := init) (t := t) hd
  have h1 : 20 * u32 * rstdRef d init t * |sR d init t j - meanRef d init t|
      ≤ rstdTol d init t * |sR d init t j - meanRef d init t| :=
    mul_le_mul_of_nonneg_right hlow hDv0
  have h3 : 20 * u32 * |zRef d init t j|
      = 20 * u32 * rstdRef d init t * |sR d init t j - meanRef d init t| := by
    rw [show zRef d init t j = (sR d init t j - meanRef d init t) * rstdRef d init t from rfl,
        abs_mul, abs_of_nonneg (le_of_lt hR0)]
    ring
  have h4 : |sR d init t j - meanRef d init t| * rstdTol d init t ≤ zTol d init t j := by
    have h2 : zTol d init t j = |sR d init t j - meanRef d init t| * rstdTol d init t
        + eTol d init t j * (rstdRef d init t + rstdTol d init t) := rfl
    have hRT0 : (0:ℝ) ≤ rstdTol d init t := le_of_lt (y_tol_rstdTol_pos hd)
    have h5 : (0:ℝ) ≤ eTol d init t j * (rstdRef d init t + rstdTol d init t) :=
      mul_nonneg hE0 (by linarith)
    rw [h2]; linarith
  linarith [h1, h3, h4]

/-- The sum of the scaled value and the bias in `float32` is finite and within a rounding error. -/
lemma y_tol_byy {by0 bb byy : F32}
    (hby0fin : by0.isFinite = true) (hbbfin : bb.isFinite = true)
    (hby0bd : |by0.toReal| ≤ (2:ℝ) ^ 127 - 2 ^ 110) (hbbbd : |bb.toReal| ≤ (2:ℝ) ^ 16)
    (hadd : AddRel true by0 bb byy) :
    byy.isFinite = true ∧
      |byy.toReal - (by0.toReal + bb.toReal)| ≤
        3 * u32 * (|by0.toReal| + |bb.toReal|) + 4 * eta32 := by
  have hu : Format.binary32.u ≤ 1 / 16 := by
    rw [Format.binary32_u]; norm_num
  have hq0 : |by0.toRat| ≤ ((2:ℚ) ^ 127 - 2 ^ 110) := by
    have h2 : ((((2:ℚ) ^ 127 - 2 ^ 110) : ℚ) : ℝ) = (2:ℝ) ^ 127 - 2 ^ 110 := by
      push_cast; norm_num
    have h1 : ((|by0.toRat| : ℚ) : ℝ) ≤ (((2:ℚ) ^ 127 - 2 ^ 110 : ℚ) : ℝ) := by
      rw [h2, Rat.cast_abs]
      exact hby0bd
    exact_mod_cast h1
  have hq1 : |bb.toRat| ≤ ((2:ℚ) ^ 16) := by
    have h2 : ((((2:ℚ) ^ 16) : ℚ) : ℝ) = (2:ℝ) ^ 16 := by push_cast; norm_num
    have h1 : ((|bb.toRat| : ℚ) : ℝ) ≤ (((2:ℚ) ^ 16 : ℚ) : ℝ) := by
      rw [h2, Rat.cast_abs]
      exact hbbbd
    exact_mod_cast h1
  have hroom : |by0.toRat| + |bb.toRat| ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have hsum : |by0.toRat| + |bb.toRat| ≤ ((2:ℚ) ^ 127 - 2 ^ 110) + 2 ^ 16 :=
      add_le_add hq0 hq1
    have hnum : ((2:ℚ) ^ 127 - 2 ^ 110) + 2 ^ 16 ≤
        ((2:ℚ) ^ 127 - 2 ^ 103) - 2 * (2:ℚ) ^ (-150 : ℤ) := by norm_num
    have hrhs : Format.binary32.maxVal / 2 - 2 * Format.binary32.eta
        = ((2:ℚ) ^ 127 - 2 ^ 103) - 2 * (2:ℚ) ^ (-150 : ℤ) := by
      rw [Format.binary32_maxVal, Format.binary32_eta]; ring
    calc |by0.toRat| + |bb.toRat| ≤ ((2:ℚ) ^ 127 - 2 ^ 110) + 2 ^ 16 := hsum
      _ ≤ ((2:ℚ) ^ 127 - 2 ^ 103) - 2 * (2:ℚ) ^ (-150 : ℤ) := hnum
      _ = Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := hrhs.symm
  obtain ⟨hfin, herr⟩ := FloatV.AddRel.err (f := Format.binary32) Format.binary32_regular
    y_tol_b32_specials hu hadd hby0fin hbbfin hroom
  refine ⟨hfin, ?_⟩
  have hcast : ((|byy.toRat - (by0.toRat + bb.toRat)| : ℚ) : ℝ) ≤
      ((3 * Format.binary32.u * (|by0.toRat| + |bb.toRat|)
        + 4 * Format.binary32.eta : ℚ) : ℝ) := by
    exact_mod_cast herr
  rw [Format.binary32_u, Format.binary32_eta] at hcast
  push_cast at hcast
  unfold u32 eta32
  simpa only [FloatV.toReal] using hcast


/-- The final chain: `byy`'s addition, its conversion, and the comparison with `yRef`. -/
lemma y_tol {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    {bx0 bmean bx brstd bw bb bhat by0 byy : F32} {by16 : BF16}
    (hval : bx0.toReal = sR d init t j) (hbx0fin : bx0.isFinite = true)
    (_hbdd : |sR d init t j| ≤ 2 ^ 57)
    (hmeanfin : bmean.isFinite = true)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hrstdfin : brstd.isFinite = true)
    (hrstd : |brstd.toReal - rstdRef d init t| ≤ rstdTol d init t)
    (hdev : AddRel true bx0 (FloatV.neg bmean) bx) (hbxfin : bx.isFinite = true)
    (hmul1 : FloatV.NaNEq (FloatV.mul bx brstd) bhat) (hbhatfin : bhat.isFinite = true)
    (_hbhatbd : |bhat.toReal| ≤ 2 ^ 111)
    (hmul2 : FloatV.NaNEq (FloatV.mul bhat bw) by0) (hby0fin : by0.isFinite = true)
    (hby0bd : |by0.toReal| ≤ 2 ^ 127 - 2 ^ 110)
    (hbwfin : bw.isFinite = true) (hbw : bw.toReal = (wv d init t j).toReal)
    (hbwbd : |bw.toReal| ≤ 2 ^ 16)
    (hbbfin : bb.isFinite = true) (hbb : bb.toReal = (bv d init t j).toReal)
    (hbbbd : |bb.toReal| ≤ 2 ^ 16)
    (hadd : AddRel true by0 bb byy)
    (hcvt : FloatV.NaNEq (FloatV.cvtWith .rn byy) by16) :
    by16.isFinite = true ∧ |by16.toReal - yRef d init t j| ≤ yTol d init t j := by
  have hu : (0:ℝ) < u32 := u32_pos
  have hu16 : (0:ℝ) < u16 := u16_pos
  have heta : (0:ℝ) < eta32 := eta32_pos
  -- 1. `bx` against the deviation
  have heBx := y_tol_ebx (d := d) hd (init := init) (t := t) (j := j)
    hval hbx0fin hmeanfin hmean hdev hbxfin
  -- 2. `bhat` against `zRef`
  have hDel := y_tol_bhat_delta (d := d) hd (init := init) (t := t) (j := j) hj
    hbxfin heBx hrstdfin hrstd hmul1 hbhatfin
  -- 3. the fused addition
  obtain ⟨hbyyfin, hbyyerr⟩ := y_tol_byy hby0fin hbbfin hby0bd hbbbd hadd
  -- 4. its size, for the conversion
  have hbyy_le : |byy.toReal| ≤ (255:ℝ) * (2:ℝ) ^ 120 := by
    have h1 : |byy.toReal| ≤ |by0.toReal| + |bb.toReal|
        + (3 * u32 * (|by0.toReal| + |bb.toReal|) + 4 * eta32) := by
      have he : byy.toReal
          = (byy.toReal - (by0.toReal + bb.toReal)) + (by0.toReal + bb.toReal) := by ring
      have h2 : |by0.toReal + bb.toReal| ≤ |by0.toReal| + |bb.toReal| := abs_add_le _ _
      rw [he]
      linarith [abs_add_le (byy.toReal - (by0.toReal + bb.toReal)) (by0.toReal + bb.toReal),
        h2, hbyyerr]
    have hS0 : (0:ℝ) ≤ |by0.toReal| + |bb.toReal| := by
      linarith [abs_nonneg (by0.toReal), abs_nonneg (bb.toReal)]
    have hb : |by0.toReal| + |bb.toReal| ≤ ((2:ℝ) ^ 127 - 2 ^ 110) + 2 ^ 16 := by
      linarith [hby0bd, hbbbd]
    have h3u : 3 * u32 ≤ 1 / 2 := by
      have h := u32_le_small
      linarith
    have h3u0 : (0:ℝ) ≤ 3 * u32 := by linarith
    have hm : 3 * u32 * (|by0.toReal| + |bb.toReal|)
        ≤ (1/2) * (((2:ℝ) ^ 127 - 2 ^ 110) + 2 ^ 16) := by
      calc 3 * u32 * (|by0.toReal| + |bb.toReal|)
          ≤ 3 * u32 * (((2:ℝ) ^ 127 - 2 ^ 110) + 2 ^ 16) :=
            mul_le_mul_of_nonneg_left hb h3u0
        _ ≤ (1/2) * (((2:ℝ) ^ 127 - 2 ^ 110) + 2 ^ 16) :=
            mul_le_mul_of_nonneg_right h3u (by linarith)
    have hX : ((2:ℝ) ^ 127 - 2 ^ 110) + 2 ^ 16 ≤ (2:ℝ) ^ 127 := by norm_num
    have hconst : (2:ℝ) ^ 127 + (1/2) * (2:ℝ) ^ 127 + 4 * eta32
        ≤ 255 * (2:ℝ) ^ 120 := by
      unfold eta32
      norm_num
    linarith [h1, hb, hm, hX, hconst]
  -- 5. the conversion
  obtain ⟨hby16fin, hby16err⟩ := y_tol_cvt_res hcvt hbyyfin hbyy_le
  refine ⟨hby16fin, ?_⟩
  -- 6. `by0` against `bhat * bw`
  have hby0err : |by0.toReal - bhat.toReal * bw.toReal|
      ≤ 2 * u32 * (|bhat.toReal| * |bw.toReal|) + 2 * eta32 := by
    have h := y_tol_mul_rel hbhatfin hbwfin hmul2 hby0fin
    rw [abs_mul] at h
    unfold u32 eta32
    exact h
  -- 7. `yTol` in the shapes the arithmetic uses
  have hyTol : yTol d init t j = 131072 * u32 * |yRef d init t j|
      + 4 * |bw.toReal| * zTol d init t j + 8 * u32 * |bb.toReal|
      + 1073741824 * eta32 := by
    have h1 : (2:ℝ) ^ (-7:ℤ) = 131072 * u32 := by unfold u32; norm_num
    have h2 : (2:ℝ) ^ (-21:ℤ) = 8 * u32 := by unfold u32; norm_num
    have h3 : (2:ℝ) ^ (-120:ℤ) = 1073741824 * eta32 := by unfold eta32; norm_num
    unfold yTol
    rw [h1, h2, h3, ← hbw, ← hbb]
  rw [hyTol]
  -- naming the pieces
  have hQ0 : (0:ℝ) ≤ |yRef d init t j| := abs_nonneg _
  have hT0 : (0:ℝ) ≤ zTol d init t j * |bw.toReal| :=
    mul_nonneg (y_tol_zTol_nonneg hd) (abs_nonneg _)
  have hBt0 : (0:ℝ) ≤ |bb.toReal| := abs_nonneg _
  -- the deviation of `byy` from `yRef`
  have hC : |bhat.toReal * bw.toReal + bb.toReal - yRef d init t j|
      ≤ |bhat.toReal - zRef d init t j| * |bw.toReal| := by
    have he : bhat.toReal * bw.toReal + bb.toReal - yRef d init t j
        = (bhat.toReal - zRef d init t j) * bw.toReal := by
      unfold yRef
      rw [← hbw, ← hbb]
      ring
    rw [he, abs_mul]
  have hE : |byy.toReal - yRef d init t j|
      ≤ (3 * u32 * (|by0.toReal| + |bb.toReal|) + 4 * eta32)
        + (2 * u32 * (|bhat.toReal| * |bw.toReal|) + 2 * eta32)
        + |bhat.toReal - zRef d init t j| * |bw.toReal| := by
    have he : byy.toReal - yRef d init t j
        = (byy.toReal - (by0.toReal + bb.toReal))
          + (by0.toReal - bhat.toReal * bw.toReal)
          + (bhat.toReal * bw.toReal + bb.toReal - yRef d init t j) := by ring
    have h1 := abs_add_le (byy.toReal - (by0.toReal + bb.toReal))
      (by0.toReal - bhat.toReal * bw.toReal)
    have h2 := abs_add_le ((byy.toReal - (by0.toReal + bb.toReal))
      + (by0.toReal - bhat.toReal * bw.toReal))
      (bhat.toReal * bw.toReal + bb.toReal - yRef d init t j)
    rw [he]
    linarith [h1, h2, hbyyerr, hby0err, hC]
  -- `by0`'s size
  have hby0_le : |by0.toReal|
      ≤ (1 + 2 * u32) * (|bhat.toReal| * |bw.toReal|) + 2 * eta32 := by
    have he : by0.toReal
        = (by0.toReal - bhat.toReal * bw.toReal) + bhat.toReal * bw.toReal := by ring
    have h1 := abs_add_le (by0.toReal - bhat.toReal * bw.toReal)
      (bhat.toReal * bw.toReal)
    have h2 : |bhat.toReal * bw.toReal| = |bhat.toReal| * |bw.toReal| := abs_mul _ _
    rw [he]
    linarith [h1, hby0err, h2]
  -- `bhat * bw` in terms of the tolerance
  have hPW : |bhat.toReal| * |bw.toReal| ≤ |bhat.toReal - zRef d init t j| * |bw.toReal|
      + |yRef d init t j| + |bb.toReal| := by
    have h1 : |bhat.toReal| ≤ |bhat.toReal - zRef d init t j| + |zRef d init t j| := by
      have he : bhat.toReal = (bhat.toReal - zRef d init t j) + zRef d init t j := by ring
      calc |bhat.toReal| = |(bhat.toReal - zRef d init t j) + zRef d init t j|
            := congrArg abs he
        _ ≤ _ := abs_add_le _ _
    have h2 := mul_le_mul_of_nonneg_right h1 (abs_nonneg (bw.toReal))
    have h3 : |zRef d init t j| * |bw.toReal| ≤ |yRef d init t j| + |bb.toReal| := by
      have he : zRef d init t j * (wv d init t j).toReal
          = yRef d init t j - (bv d init t j).toReal := by unfold yRef; ring
      have h4 : |zRef d init t j * (wv d init t j).toReal|
          ≤ |yRef d init t j| + |(bv d init t j).toReal| := by
        rw [he]
        have h := abs_add_le (yRef d init t j) (-(bv d init t j).toReal)
        simpa only [sub_eq_add_neg, abs_neg] using h
      calc |zRef d init t j| * |bw.toReal|
          = |zRef d init t j * (wv d init t j).toReal| := by rw [hbw, abs_mul]
        _ ≤ |yRef d init t j| + |(bv d init t j).toReal| := h4
        _ = |yRef d init t j| + |bb.toReal| := by rw [← hbb]
    linarith [h2, h3]
  -- the trade on `|bb|`
  have hZWT : |zRef d init t j| * |bw.toReal|
      ≤ (zTol d init t j * |bw.toReal|) / (20 * u32) := by
    have hZ := y_tol_zTol_low (d := d) (init := init) (t := t) (j := j) hd
    have h20 : (0:ℝ) < 20 * u32 := by linarith
    rw [le_div_iff₀ h20]
    have h := mul_le_mul_of_nonneg_right hZ (abs_nonneg (bw.toReal))
    linarith [h]
  have hBQ : |bb.toReal| ≤ |yRef d init t j| + |zRef d init t j| * |bw.toReal| := by
    have he : (bv d init t j).toReal
        = yRef d init t j - zRef d init t j * (wv d init t j).toReal := by
      unfold yRef; ring
    have h1 : |(bv d init t j).toReal|
        ≤ |yRef d init t j| + |zRef d init t j * (wv d init t j).toReal| := by
      rw [he]
      have h := abs_add_le (yRef d init t j) (-(zRef d init t j * (wv d init t j).toReal))
      simpa only [sub_eq_add_neg, abs_neg] using h
    have h2 : |zRef d init t j * (wv d init t j).toReal|
        = |zRef d init t j| * |(wv d init t j).toReal| := abs_mul _ _
    rw [h2, ← hbw, ← hbb] at h1
    exact h1
  have hB : |bb.toReal| ≤ |yRef d init t j| + (zTol d init t j * |bw.toReal|) / (20 * u32) := by
    linarith [hBQ, hZWT]
  -- combining the deviation with the elementary bounds
  have hc5 : (0:ℝ) ≤ 1 + 5 * u32 + 6 * u32 ^ 2 := by nlinarith [hu]
  have hc6 : (0:ℝ) ≤ 5 * u32 + 6 * u32 ^ 2 := by nlinarith [hu]
  have hc8 : (0:ℝ) ≤ 8 * u32 + 6 * u32 ^ 2 := by nlinarith [hu]
  have h1 : 3 * u32 * (|by0.toReal| + |bb.toReal|)
      ≤ 3 * u32 * ((1 + 2 * u32) * (|bhat.toReal| * |bw.toReal|) + 2 * eta32
        + |bb.toReal|) := by
    have h := mul_le_mul_of_nonneg_left (by linarith [hby0_le] : |by0.toReal| + |bb.toReal|
      ≤ ((1 + 2 * u32) * (|bhat.toReal| * |bw.toReal|) + 2 * eta32) + |bb.toReal|)
      (by linarith : (0:ℝ) ≤ 3 * u32)
    linarith [h]
  have h2 := mul_le_mul_of_nonneg_left hPW hc6
  have hEf : |byy.toReal - yRef d init t j|
      ≤ (1 + 5 * u32 + 6 * u32 ^ 2) * (|bhat.toReal - zRef d init t j| * |bw.toReal|)
        + (5 * u32 + 6 * u32 ^ 2) * |yRef d init t j|
        + (8 * u32 + 6 * u32 ^ 2) * |bb.toReal| + 6 * u32 * eta32 + 6 * eta32 := by
    linarith [hE, h1, h2]
  have hDelW : |bhat.toReal - zRef d init t j| * |bw.toReal|
      ≤ 7 / 2 * (zTol d init t j * |bw.toReal|) + 2 ^ 26 * eta32 := by
    have h1 := mul_le_mul_of_nonneg_right hDel (abs_nonneg (bw.toReal))
    have h2 : eta32 * |bw.toReal| ≤ 2 ^ 16 * eta32 := by
      have h := mul_le_mul_of_nonneg_left hbwbd (le_of_lt heta)
      linarith [h]
    nlinarith [h1, h2, hbwbd]
  have hDelW' := mul_le_mul_of_nonneg_left hDelW hc5
  have hBtT : (8 * u32 + 6 * u32 ^ 2) * |bb.toReal|
      ≤ (8 * u32 + 6 * u32 ^ 2) * |yRef d init t j|
        + (2/5 + 3 * u32 / 10) * (zTol d init t j * |bw.toReal|) := by
    have h1 := mul_le_mul_of_nonneg_left hB hc8
    have h2 : (8 * u32 + 6 * u32 ^ 2)
        * (|yRef d init t j| + (zTol d init t j * |bw.toReal|) / (20 * u32))
        = (8 * u32 + 6 * u32 ^ 2) * |yRef d init t j|
          + (2/5 + 3 * u32 / 10) * (zTol d init t j * |bw.toReal|) := by
      have hne : u32 ≠ 0 := ne_of_gt hu
      field_simp
      ring
    linarith [h1, h2]
  have hEf2 : |byy.toReal - yRef d init t j|
      ≤ (1 + 5 * u32 + 6 * u32 ^ 2) * (7 / 2 * (zTol d init t j * |bw.toReal|)
          + 2 ^ 26 * eta32)
        + (5 * u32 + 6 * u32 ^ 2) * |yRef d init t j|
        + ((8 * u32 + 6 * u32 ^ 2) * |yRef d init t j|
          + (2/5 + 3 * u32 / 10) * (zTol d init t j * |bw.toReal|))
        + 6 * u32 * eta32 + 6 * eta32 := by
    linarith [hEf, hDelW', hBtT]
  -- the split of `by16` against `byy`
  have hA1 : |by16.toReal - yRef d init t j|
      ≤ u16 * |byy.toReal| + eta16 + |byy.toReal - yRef d init t j| := by
    have he : by16.toReal - yRef d init t j
        = (by16.toReal - byy.toReal) + (byy.toReal - yRef d init t j) := by ring
    rw [he]
    linarith [abs_add_le (by16.toReal - byy.toReal) (byy.toReal - yRef d init t j),
      hby16err]
  have hsplit : |byy.toReal| ≤ |yRef d init t j| + |byy.toReal - yRef d init t j| := by
    have he : byy.toReal = yRef d init t j + (byy.toReal - yRef d init t j) := by ring
    calc |byy.toReal| = |yRef d init t j + (byy.toReal - yRef d init t j)| := congrArg abs he
      _ ≤ _ := abs_add_le _ _
  have hS2 : |by16.toReal - yRef d init t j|
      ≤ u16 * |yRef d init t j| + (1 + u16) * |byy.toReal - yRef d init t j| + eta16 := by
    have h1 := mul_le_mul_of_nonneg_left hsplit (le_of_lt hu16)
    nlinarith [hA1, h1]
  have hEf3 := mul_le_mul_of_nonneg_left hEf2 (by linarith : (0:ℝ) ≤ 1 + u16)
  -- numeric coefficient facts
  have hN1 : (1 + u16) * (7 / 2 * (1 + 5 * u32 + 6 * u32 ^ 2) + (2/5 + 3 * u32 / 10)) ≤ 4 := by
    unfold u16 u32
    norm_num
  have hN2 : u16 + (1 + u16) * (13 * u32 + 12 * u32 ^ 2) ≤ 131072 * u32 := by
    unfold u16 u32
    norm_num
  have hN3 : (1 + u16) * ((1 + 5 * u32 + 6 * u32 ^ 2) * 2 ^ 26 + 6 * u32 + 6) * eta32
      + eta16 ≤ 1073741824 * eta32 := by
    unfold u16 eta16 u32 eta32
    norm_num
  have hA1' := mul_le_mul_of_nonneg_right hN1 hT0
  have hA2' := mul_le_mul_of_nonneg_right hN2 hQ0
  have hBt0' : (0:ℝ) ≤ 8 * u32 * |bb.toReal| := by
    have := abs_nonneg (bb.toReal); nlinarith [hu]
  linarith [hS2, hEf3, hA1', hA2', hN3, hQ0, hT0, hBt0']


end

end Claims.LayerNormFwd
