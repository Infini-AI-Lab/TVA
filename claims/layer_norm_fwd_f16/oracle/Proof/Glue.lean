import Claim
import TVA.Proof
import Proof.Row

/-!
# Float facts for the kernel's operations

Each lemma turns what the model allows for one operation (`AddRel`, `NaNEq`, a device fact) into a
bound on real values, in the form the stages of `Proof.Row` take: `LaneErr` for the subtractions,
`3u d² + 4η` for the squares, and so on.
-/

namespace Claims.LayerNormFwdF16

open TVA

set_option maxHeartbeats 800000

/-- The arithmetic of a fused subtraction: one operand exact, the other within one rounding, and the
sum rounded once. -/
theorem lane_sub_core (a b p q res u η : ℚ) (hu : 0 ≤ u) (hu1 : u ≤ 1) (hη : 0 ≤ η)
    (hres : |res - (p + q)| ≤ |p + q| * u + η) (hp : |p - a| ≤ |a| * u + η)
    (hq : |q + b| ≤ |b| * u + η) (hpq : p = a ∨ q = -b) :
    |res - (a - b)| ≤ u * |a - b| + u * (1 + u) * (|a| + |b|) + 3 * η := by
  have ha := abs_nonneg a
  have hb := abs_nonneg b
  have hab := abs_nonneg (a - b)
  have hua : 0 ≤ u * (1 + u) * |a| := by positivity
  have hub : 0 ≤ u * (1 + u) * |b| := by positivity
  have huη : u * η ≤ η := by nlinarith
  rcases hpq with rfl | rfl
  · have h1 : |p + q| ≤ |p - b| + (|b| * u + η) := by
      have : p + q = (p - b) + (q + b) := by ring
      rw [this]; exact (abs_add_le _ _).trans (by linarith)
    have h2 : |res - (p - b)| ≤ |res - (p + q)| + |q + b| := by
      have : res - (p - b) = (res - (p + q)) + (q + b) := by ring
      rw [this]; exact abs_add_le _ _
    nlinarith [mul_le_mul_of_nonneg_right h1 hu]
  · have h1 : |p + -b| ≤ |a - b| + (|a| * u + η) := by
      have : p + (-b) = (a - b) + (p - a) := by ring
      rw [this]; exact (abs_add_le _ _).trans (by linarith)
    have h2 : |res - (a - b)| ≤ |res - (p + -b)| + |p - a| := by
      have : res - (a - b) = (res - (p + -b)) + (p - a) := by ring
      rw [this]; exact abs_add_le _ _
    nlinarith [mul_le_mul_of_nonneg_right h1 hu]

/-- **A fusable subtraction** `x - m` of finite `float32` values of moderate size: finite, with the
lane error `LaneErr`. -/
theorem lane_sub {fusion : Bool} {x m r : F32} (hx : x.isFinite = true) (hm : m.isFinite = true)
    (hxb' : |x.toReal| ≤ 2 ^ 20) (hmb' : |m.toReal| ≤ 2 ^ 20)
    (h : AddRel fusion x (FloatV.neg m) r) :
    r.isFinite = true ∧ LaneErr x.toReal m.toReal r.toReal := by
  have hxb : |x.toRat| ≤ 2 ^ 20 := by unfold FloatV.toReal at hxb'; exact_mod_cast hxb'
  have hmb : |m.toRat| ≤ 2 ^ 20 := by unfold FloatV.toReal at hmb'; exact_mod_cast hmb'
  have hf := Format.binary32_regular
  have hs : Format.binary32.specials ≠ Format.Specials.finite := by
    rw [(Format.binary32_std.ieee)]; decide
  have hu : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u
  have he : Format.binary32.eta = 2 ^ (-150 : ℤ) := Format.binary32_eta
  have hM : (Format.binary32.maxVal : ℚ) = (2 ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal
  have hmn : (FloatV.neg m).isFinite = true := (FloatV.isFinite_neg m).trans hm
  have hmnr : (FloatV.neg m).toRat = -m.toRat := FloatV.toRat_neg m
  have hu0 : (0 : ℚ) ≤ 2 ^ (-24 : ℤ) := by positivity
  have hu1 : (2 : ℚ) ^ (-24 : ℤ) ≤ 1 := by norm_num
  have hη0 : (0 : ℚ) ≤ 2 ^ (-150 : ℤ) := by positivity
  have hmaxv22 : (2 : ℚ) ^ 22 ≤ Format.binary32.maxVal := by rw [hM]; norm_num
  have habs : |x.toRat + (FloatV.neg m).toRat| ≤ 2 ^ 21 := by
    rw [hmnr]
    calc _ ≤ |x.toRat| + |-m.toRat| := abs_add_le _ _
      _ ≤ 2 ^ 20 + 2 ^ 20 := by rw [abs_neg]; linarith
      _ = 2 ^ 21 := by norm_num
  have hmaxv : (2 : ℚ) ^ 21 ≤ Format.binary32.maxVal := by rw [hM]; norm_num
  have key : r.isFinite = true ∧ |r.toRat - (x.toRat - m.toRat)| ≤
      2 ^ (-24 : ℤ) * |x.toRat - m.toRat| + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) *
        (|x.toRat| + |m.toRat|) + 3 * 2 ^ (-150 : ℤ) := by
    have h1 : 0 ≤ 2 ^ (-24 : ℤ) * (1 + 2 ^ (-24 : ℤ)) * (|x.toRat| + |m.toRat|) := by positivity
    rcases h with hne | ⟨-, hfu⟩
    · obtain ⟨hfin, herr⟩ := FloatV.add_err hf hx hmn (habs.trans hmaxv)
      rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
      refine ⟨hfin, ?_⟩
      rw [hmnr, ← sub_eq_add_neg, hu, he] at herr
      have h2 : |x.toRat - m.toRat| * 2 ^ (-24 : ℤ) = 2 ^ (-24 : ℤ) * |x.toRat - m.toRat| :=
        mul_comm _ _
      linarith
    · rcases hfu with ⟨p, q, hp, hq, hpq, z, hne⟩ | ⟨p, -, hyi, -⟩ | ⟨q, -, hxi, -⟩
      · have ep := RoundsTo.err hf hs hp hx
        have eq := RoundsTo.err hf hs hq hmn
        rw [hmnr, hu, he] at eq
        rw [hu, he] at ep
        rw [abs_neg] at eq
        have hpq' : |p + q| ≤ 2 ^ 22 := by
          have := abs_add_le p q
          have h3 : |p| ≤ |x.toRat| + (|x.toRat| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) := by
            have := abs_sub_abs_le_abs_sub p x.toRat; linarith [abs_sub_comm p x.toRat]
          have h4 : |q| ≤ |m.toRat| + (|m.toRat| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) := by
            have := abs_sub_abs_le_abs_sub q (-m.toRat); rw [abs_neg] at this
            linarith [abs_sub_comm q (-m.toRat)]
          norm_num at h3 h4 ⊢
          nlinarith [abs_nonneg p, abs_nonneg q]
        obtain ⟨hfin, herr⟩ := FloatV.ofPat_roundRat_err hf false z (hpq'.trans hmaxv22)
        rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
        refine ⟨hfin, ?_⟩
        rw [hu, he] at herr
        have := lane_sub_core x.toRat m.toRat p q _ (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ)) hu0 hu1 hη0
          herr ep (by rw [show q + m.toRat = q - -m.toRat by ring]; exact eq) (by rwa [hmnr] at hpq)
        exact this
      · exfalso; rw [(FloatV.isFinite_iff.mp hmn).2] at hyi; cases hyi
      · exfalso; rw [(FloatV.isFinite_iff.mp hx).2] at hxi; cases hxi
  obtain ⟨hfin, herr⟩ := key
  refine ⟨hfin, ?_⟩
  unfold LaneErr u32 eta32 FloatV.toReal
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  convert this using 2

/-- `+0 + x`: the first trip's `_mean += a` is within one rounding of `a`. -/
theorem add_zero_lane {x r : F32} (hx : x.isFinite = true) (hb : |x.toReal| ≤ 2 ^ 20)
    (h : FloatV.NaNEq (FloatV.add (FloatV.ofPat 0) x) r) :
    r.isFinite = true ∧ |r.toReal - x.toReal| ≤ u32 * |x.toReal| + eta32 := by
  have hf := Format.binary32_regular
  have hz : (FloatV.ofPat 0 : F32).isFinite = true := FloatV.isFinite_ofPat_zero hf
  have hM : (2 : ℝ) ^ 20 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]; norm_num
  obtain ⟨hfin, herr⟩ := FloatV.add_err_real hf hz hx (by
    rw [FloatV.toReal_ofPat_zero, zero_add]; exact hb.trans hM)
  rw [h.eq (FloatV.isFinite_iff.mp hfin).1]
  refine ⟨hfin, ?_⟩
  rw [FloatV.toReal_ofPat_zero, zero_add, Format.binary32_u_real, Format.binary32_eta_real] at herr
  unfold u32 eta32
  linarith

/-- `d·d + 0` (a fused square): within `3u d² + 4η` of `d²`. -/
theorem muladd_sq {fusion : Bool} {d r : F32} (hd : d.isFinite = true) (hb : |d.toReal| ≤ 2 ^ 20)
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul d d) (FloatV.ofPat 0)) r ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma d d (FloatV.ofPat 0)) r)) :
    r.isFinite = true ∧ |r.toReal - d.toReal ^ 2| ≤ 3 * u32 * d.toReal ^ 2 + 4 * eta32 := by
  have hf := Format.binary32_regular
  have hz : (FloatV.ofPat 0 : F32).isFinite = true := FloatV.isFinite_ofPat_zero hf
  have hu : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hM : (Format.binary32.maxVal : ℚ) = (2 ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal
  have hz0 : (FloatV.ofPat 0 : F32).toRat = 0 := by
    have := FloatV.toReal_ofPat_zero (f := Format.binary32)
    unfold FloatV.toReal at this; exact_mod_cast this
  have hbq : |d.toRat| ≤ 2 ^ 20 := by
    unfold FloatV.toReal at hb; exact_mod_cast hb
  have h1 : |d.toRat * d.toRat| ≤ 2 ^ 40 := by
    rw [abs_mul]; nlinarith [abs_nonneg d.toRat]
  have h2 : (2 : ℚ) ^ 40 ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [hM, Format.binary32_eta]; norm_num
  obtain ⟨hfin, herr⟩ := FloatV.mulAdd_err hf hu h hd hd hz (by rw [hz0, abs_zero]; linarith)
  refine ⟨hfin, ?_⟩
  rw [hz0, abs_zero, add_zero, add_zero, Format.binary32_u, Format.binary32_eta] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  unfold u32 eta32 FloatV.toReal
  rw [abs_mul, ← abs_mul_abs_self] at this
  convert this using 2 <;> simp [sq, abs_mul_abs_self]


/-- The sum of `f` over `List.range n` is the `Finset` sum. -/
theorem list_sum_range (f : ℕ → ℚ) (n : ℕ) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]; simp

/-- **A `tl.sum` of `n ≤ 2^15` finite `float32` lanes with tree depth at most `D ≤ 15`** (IEEE
additions, any tree): finite, within `((1+u)^D − 1) Σ|l| + η (n−1) (1+u)^D` of the exact sum. -/
theorem sum_real {n D : ℕ} (hD : D ≤ 15) (hn : n ≤ 2 ^ 15) (l : ℕ → F32) {t : Tree} {S : F32}
    (hcov : t.Covers n) (hd : t.depth ≤ D) (heval : t.Eval (AddRel false) l S)
    (hfin : ∀ i, i < n → (l i).isFinite = true)
    (hb : ∀ i, i < n → |(l i).toReal| ≤ 2 ^ 40) :
    S.isFinite = true ∧
      |S.toReal - ∑ i ∈ Finset.range n, (l i).toReal| ≤
        ((1 + u32) ^ D - 1) * ∑ i ∈ Finset.range n, |(l i).toReal| +
          eta32 * ((n - 1 : ℕ) : ℝ) * (1 + u32) ^ D := by
  have hf := Format.binary32_regular
  have hM : (Format.binary32.maxVal : ℚ) = (2 ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal
  have hbq : ∀ i, i < n → |(l i).toRat| ≤ 2 ^ 40 := fun i hi => by
    have := hb i hi; unfold FloatV.toReal at this; exact_mod_cast this
  have hA : ((List.range n).map fun i => |(l i).toRat|).sum ≤ 2 ^ 55 := by
    rw [list_sum_range]
    calc _ ≤ ∑ i ∈ Finset.range n, (2 : ℚ) ^ 40 :=
          Finset.sum_le_sum fun i hi => hbq i (Finset.mem_range.mp hi)
      _ = n * 2 ^ 40 := by simp
      _ ≤ 2 ^ 15 * 2 ^ 40 := by
          have : (n : ℚ) ≤ 2 ^ 15 := by exact_mod_cast hn
          nlinarith
      _ = 2 ^ 55 := by norm_num
  have hu1 : (1 + Format.binary32.u) ^ D ≤ 2 := by
    rw [Format.binary32_u]
    calc (1 + (2 : ℚ) ^ (-24 : ℤ)) ^ D ≤ (1 + 2 ^ (-24 : ℤ)) ^ 15 :=
          pow_le_pow_right₀ (by norm_num) hD
      _ ≤ 2 := by norm_num
  have hA0 : 0 ≤ ((List.range n).map fun i => |(l i).toRat|).sum := by
    rw [list_sum_range]; exact Finset.sum_nonneg fun i _ => abs_nonneg _
  obtain ⟨hfinS, herr⟩ := sum_err_depth hf l hcov hd heval hfin (by
    unfold Tree.errBound
    rw [Format.binary32_eta, hM]
    have hn' : ((n - 1 : ℕ) : ℚ) ≤ 2 ^ 15 := by
      have : n - 1 ≤ 2 ^ 15 := by omega
      exact_mod_cast this
    have hp : 0 ≤ (1 + Format.binary32.u) ^ D :=
      pow_nonneg (by linarith [Format.u_nonneg (f := Format.binary32)]) D
    have h1 : ((1 + Format.binary32.u) ^ D - 1) * ((List.range n).map fun i => |(l i).toRat|).sum ≤
        2 ^ 55 := by nlinarith
    have h2 : (2 : ℚ) ^ (-150 : ℤ) * ((n - 1 : ℕ) : ℚ) * (1 + Format.binary32.u) ^ D ≤ 1 := by
      have : (2 : ℚ) ^ (-150 : ℤ) * ((n - 1 : ℕ) : ℚ) * (1 + Format.binary32.u) ^ D ≤
          2 ^ (-150 : ℤ) * 2 ^ 15 * 2 := by
        gcongr
      refine this.trans ?_
      norm_num
    norm_num at h1 h2 ⊢
    linarith)
  refine ⟨hfinS, ?_⟩
  rw [list_sum_range, list_sum_range, Format.binary32_u, Format.binary32_eta] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  unfold Tree.errBound at this
  push_cast at this
  unfold u32 eta32 FloatV.toReal
  simpa using this


/-- A natural number below `2^24` converts to `float32` exactly. -/
theorem ofRat_nat_f32 {z : ℤ} (hz : |z| < 2 ^ 24) :
    (FloatV.ofRat (z : ℚ) : F32).isFinite = true ∧ (FloatV.ofRat (z : ℚ) : F32).toRat = z := by
  have hf := Format.binary32_regular
  have hq : |(z : ℚ)| ≤ Format.binary32.maxVal := by
    have : |(z : ℚ)| < 2 ^ 24 := by exact_mod_cast hz
    rw [Format.binary32_maxVal]; norm_num; linarith
  unfold FloatV.ofRat
  by_cases h0 : (z : ℚ) = 0
  · obtain ⟨h1, h2, h3⟩ := Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn)
      (sat := false) (zs := false) hf
    rw [h0]
    refine ⟨?_, ?_⟩
    · unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
    · rw [FloatV.toRat_ofPat h1, h3]
  obtain ⟨h1, h2, h3⟩ := Format.roundRat_spec hf .rn false false h0 hq
  refine ⟨?_, ?_⟩
  · unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
  rw [FloatV.toRat_ofPat h1, h3]
  have ha : 0 < |(z : ℚ)| := abs_pos.mpr h0
  have hlog : Int.log 2 |(z : ℚ)| ≤ 23 := by
    have : |(z : ℚ)| < (2 : ℚ) ^ (24 : ℤ) := by exact_mod_cast hz
    have := Format.log_lt_of_lt ha this
    omega
  have hQ : Format.binary32.qexp |(z : ℚ)| ≤ 0 := by
    rw [Format.qexp_eq ha]
    have : Format.binary32.emin = -126 := by decide
    have hm : (Format.binary32.m : ℤ) = 23 := rfl
    rw [this, hm]; omega
  set Q := Format.binary32.qexp |(z : ℚ)| with hQdef
  have hK : |(z : ℚ)| * (2 : ℚ) ^ (-Q) = ((z.natAbs * 2 ^ (-Q).toNat : ℕ) : ℚ) := by
    push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega)]
    congr 1
    rw [Nat.cast_natAbs, Int.cast_abs]
  rw [Format.roundSig_exact .rn hK]
  by_cases hneg : (z : ℚ) < 0
  · rw [ite_eq_left hneg, abs_of_neg hneg]; ring
  · rw [ite_eq_right hneg, abs_of_nonneg (not_lt.mp hneg), one_mul]

/-- The `float64` `1.0` is `1`. -/
theorem one64 : (FloatV.ofPat 4607182418800017408 : F64).toRat = 1 := by
  unfold FloatV.toRat
  rw [FloatV.pat_ofPat (by decide)]
  unfold Format.toRat Format.magVal
  norm_num [Format.sign, Format.mag, Format.pow2_eq, Format.signBit, Format.binary64, Format.bias]

/-- The pattern of `1.0` in `float64` is finite. -/
theorem fin64 : Format.binary64.isFinite 4607182418800017408 = true := by decide
/-- The `float64` `1.0` is not a zero. -/
theorem notzero64 : (FloatV.ofPat 4607182418800017408 : F64).isZero = false := by
  unfold FloatV.isZero Format.isZero Format.mag
  rw [FloatV.pat_ofPat (by decide)]
  decide
/-- The `float64` `1.0` is not a NaN. -/
theorem notnan64 : (FloatV.ofPat 4607182418800017408 : F64).isNaN = false := by
  unfold FloatV.isNaN
  rw [FloatV.pat_ofPat (by decide)]
  decide

/-- **The constant `1` of line 81**, as a `float32`: exactly `1`. -/
theorem litF32_one : (FloatV.litF32 (FloatV.ofPat 4607182418800017408 : F64)).isFinite = true ∧
    (FloatV.litF32 (FloatV.ofPat 4607182418800017408 : F64)).toReal = 1 := by
  unfold FloatV.litF32 FloatV.hostF32
  rw [notzero64]
  simp only [notnan64]
  simp only [Bool.false_eq_true, ite_false]
  have hc : (FloatV.ofPat 4607182418800017408 : F64).cvt = (FloatV.ofRat (1 : ℚ) : F32) := by
    unfold FloatV.cvt FloatV.ofRat
    rw [FloatV.pat_ofPat (by decide), Format.cvt_of_finite _ _ _ fin64]
    have hs : Format.binary64.sign 4607182418800017408 = false := by decide
    have ht : Format.binary64.toRat 4607182418800017408 = 1 := by
      have := one64; unfold FloatV.toRat at this; rwa [FloatV.pat_ofPat (by decide)] at this
    rw [hs, ht]
  rw [hc]
  obtain ⟨h1, h2⟩ := ofRat_nat_f32 (z := 1) (by norm_num)
  have h2' : (FloatV.ofRat (1 : ℚ) : F32).toRat = 1 := by simpa using h2
  refine ⟨by simpa using h1, ?_⟩
  unfold FloatV.toReal; rw [h2']; simp

/-- **`N` as a `float32` scalar** (`tl.itof` of the `int32` `N`, line 72): exact for `N < 2^24`. -/
theorem itof_exact {N : ℕ} (hN : N < 2 ^ 24) {r : F32}
    (h : tl.ItofRel .binary32 ((IntV.ofNat N : I32).toInt) r) :
    r.isFinite = true ∧ r.toReal = N := by
  have h1 : (IntV.ofNat N : I32).toInt = N := IntV.I32.toInt_ofNat (by omega)
  rw [h1] at h
  have hz : |(N : ℤ)| < 2 ^ 53 := by
    rw [abs_of_nonneg (by positivity)]; have : (N : ℤ) < 2 ^ 24 := by exact_mod_cast hN
    linarith
  rw [tl.itofRel_binary32 hz] at h
  subst h
  obtain ⟨hfin, hval⟩ := ofRat_nat_f32 (z := N) (by
    rw [abs_of_nonneg (by positivity)]; exact_mod_cast hN)
  refine ⟨hfin, ?_⟩
  unfold FloatV.toReal; rw [hval]; simp


/-- **`/` on `float32`** (`Device.DivFull`): finite, within `2^-22 |x/y| + 2^-148` of `x / y`. -/
theorem div_real {P : Platform} (hP : Device P) {x y r : F32} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hy0 : y.toReal ≠ 0) (hq : |x.toReal / y.toReal| ≤ 2 ^ 126)
    (h : P.div .binary32 x y r) :
    r.isFinite = true ∧
      |r.toReal - x.toReal / y.toReal| ≤ 2 ^ (-22 : ℤ) * |x.toReal / y.toReal| + 2 ^ (-148 : ℤ) := by
  have hy0' : y.toRat ≠ 0 := fun h0 => hy0 (by unfold FloatV.toReal; rw [h0]; simp)
  have hq' : |x.toRat / y.toRat| ≤ 2 ^ 126 := by
    unfold FloatV.toReal at hq
    rw [← Rat.cast_div, ← Rat.cast_abs] at hq
    exact_mod_cast hq
  obtain ⟨hfin, herr⟩ := hP.divFull x y r h hx hy hy0' hq'
  refine ⟨hfin, ?_⟩
  have := (Rat.cast_le (K := ℝ)).mpr herr
  unfold FloatV.toReal
  push_cast at this ⊢
  simpa [Rat.cast_div] using this

/-- **`tl.sqrt` on `float32`** (`Device.SqrtF32`): on a normal input of moderate size, finite and
within `2^-22 √s + 2^-149` of `√s`. -/
theorem sqrt_real {P : Platform} (hP : Device P) {s r : F32} (hs : s.isFinite = true)
    (hlo : (2 : ℝ) ^ (-126 : ℤ) ≤ s.toReal) (hhi : s.toReal ≤ 2 ^ 127)
    (h : P.math .sqrt .binary32 s r) :
    r.isFinite = true ∧
      |r.toReal - Real.sqrt s.toReal| ≤ 2 ^ (-22 : ℤ) * Real.sqrt s.toReal + 2 ^ (-149 : ℤ) := by
  obtain ⟨ref, href, hacc⟩ := hP.sqrtF32
  have hr : ref = Real.sqrt := by
    unfold MathFn.ref at href; exact (Option.some.inj href).symm
  subst hr
  have hsq : Real.sqrt s.toReal ≤ 2 ^ 64 := by
    rw [Real.sqrt_le_left (by norm_num)]; norm_num; linarith
  have hsq0 := Real.sqrt_nonneg s.toReal
  have hpos : (0 : ℝ) < 2 ^ (-126 : ℤ) := by positivity
  obtain ⟨h1, h2⟩ := hacc s r h hs (Or.inr hlo)
    (by simp only [MathFn.natDom]; exact Set.mem_Ici.mpr (by linarith)) (by
      rw [abs_of_nonneg hsq0, Format.binary32_maxVal_real]
      norm_num
      nlinarith)
  rw [abs_of_nonneg hsq0] at h2
  exact ⟨h1, h2⟩

/-- **The host's `eps` as a `float32`**: within `u |eps| + η`. -/
theorem hostF32_err {c : F64} (hc : c.isFinite = true) (hb : |c.toReal| ≤ 2 ^ 126) :
    (FloatV.hostF32 c).isFinite = true ∧
      |(FloatV.hostF32 c).toReal - c.toReal| ≤ u32 * |c.toReal| + eta32 := by
  have hnan : c.isNaN = false := (FloatV.isFinite_iff.mp hc).1
  unfold FloatV.hostF32
  rw [ite_eq_right (by simp [hnan])]
  have hM : (2 : ℝ) ^ 126 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]; norm_num
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real Format.binary32_regular hc (hb.trans hM)
  refine ⟨h1, ?_⟩
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  unfold u32 eta32
  linarith


/-- **A fusable addition** (`tl.addc`) of finite `float32` values with room: finite, within
`3u(|x| + |y|) + 4η` of `x + y`. -/
theorem addc_real {fusion : Bool} {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hb : |x.toReal| + |y.toReal| ≤ 2 ^ 126 + 2 ^ 111)
    (h : AddRel fusion x y r) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal + y.toReal)| ≤ 3 * u32 * (|x.toReal| + |y.toReal|) + 4 * eta32 := by
  have hf := Format.binary32_regular
  have hs : Format.binary32.specials ≠ Format.Specials.finite := by
    rw [(Format.binary32_std.ieee)]; decide
  have hu : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hbq : |x.toRat| + |y.toRat| ≤ 2 ^ 126 + 2 ^ 111 := by
    unfold FloatV.toReal at hb
    rw [← Rat.cast_abs, ← Rat.cast_abs] at hb
    exact_mod_cast hb
  obtain ⟨hfin, herr⟩ := FloatV.AddRel.err hf hs hu h hx hy (by
    rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num at hbq ⊢; linarith)
  refine ⟨hfin, ?_⟩
  rw [Format.binary32_u, Format.binary32_eta] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  unfold u32 eta32 FloatV.toReal
  simpa using this


/-- **An IEEE product** of finite `float32` values with room: finite, within `u |a b| + η`. -/
theorem mul_lane {a b r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hab : |a.toReal * b.toReal| ≤ 2 ^ 120) (h : FloatV.NaNEq (FloatV.mul a b) r) :
    r.isFinite = true ∧ |r.toReal - a.toReal * b.toReal| ≤ u32 * |a.toReal * b.toReal| + eta32 := by
  have hf := Format.binary32_regular
  have hM : (2 : ℝ) ^ 120 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]; norm_num
  obtain ⟨hfin, herr⟩ := FloatV.mul_err_real hf ha hb (hab.trans hM)
  rw [h.eq (FloatV.isFinite_iff.mp hfin).1]
  refine ⟨hfin, ?_⟩
  rw [Format.binary32_u_real, Format.binary32_eta_real] at herr
  unfold u32 eta32
  linarith

/-- **`tl.mul_add a b c`** of finite `float32` values with room: finite, within
`3u(|a b| + |c|) + 4η` of `a b + c`. -/
theorem muladd_lane {fusion : Bool} {a b c r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hc : c.isFinite = true) (hroom : |a.toReal * b.toReal| + |c.toReal| ≤ 2 ^ 40)
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul a b) c) r ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma a b c) r)) :
    r.isFinite = true ∧ |r.toReal - (a.toReal * b.toReal + c.toReal)| ≤
      3 * u32 * (|a.toReal * b.toReal| + |c.toReal|) + 4 * eta32 := by
  have hf := Format.binary32_regular
  have hu : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hM : (Format.binary32.maxVal : ℚ) = (2 ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal
  have hroomq : |a.toRat * b.toRat| + |c.toRat| ≤ 2 ^ 40 := by
    unfold FloatV.toReal at hroom
    rw [← Rat.cast_mul, ← Rat.cast_abs, ← Rat.cast_abs] at hroom
    exact_mod_cast hroom
  obtain ⟨hfin, herr⟩ := FloatV.mulAdd_err hf hu h ha hb hc (by
    rw [hM, Format.binary32_eta]; norm_num at hroomq ⊢; linarith)
  refine ⟨hfin, ?_⟩
  rw [Format.binary32_u, Format.binary32_eta] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  unfold u32 eta32 FloatV.toReal
  simpa using this

/-- **The final conversion** `float32 → float16` (line 95): finite, within `|y| 2^-11 + 2^-25`. -/
theorem cvt_f32_f16_lane {y : F32} {r16 : F16} (hy : y.isFinite = true) (hb : |y.toReal| ≤ 65504)
    (h : FloatV.NaNEq (FloatV.cvt y) r16) :
    r16.isFinite = true ∧ |r16.toReal - y.toReal| ≤ |y.toReal| * 2 ^ (-11 : ℤ) + 2 ^ (-25 : ℤ) := by
  have hM : |y.toReal| ≤ (Format.binary16.maxVal : ℝ) := by
    rw [Format.binary16_maxVal_real]; exact hb
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err_real (g := Format.binary16) Format.binary16_regular hy hM
  rw [h.eq (FloatV.isFinite_iff.mp hfin).1]
  refine ⟨hfin, ?_⟩
  rw [Format.binary16_u_real, Format.binary16_eta_real] at herr
  exact herr

end Claims.LayerNormFwdF16
