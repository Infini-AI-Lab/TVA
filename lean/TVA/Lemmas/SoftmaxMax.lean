import TVA.Lemmas.Num
import TVA.Lang.Ops
import TVA.Math.Float.Format.Bridge

/-!
# `float32` maxima by value

A softmax subtracts the row maximum. `tl.max` combines a row with `FloatV.MaxRel` along a tree
Triton chooses, and `tl.maximum` is one `MaxRel`. On finite `float32` values the result is one
of the operands by value and bounds them all, whatever the tree:

* `Softmax.toRat_le_of_key_le`: the value of a pattern is monotone in its IEEE order key;
* `Softmax.maxRel_val`: `MaxRel true a b r` on finite `a`, `b` gives a finite `r` equal by value
  to `a` or `b` and at least both;
* `Softmax.maxTree_val`: the same for a tree over finite leaves (`tl.max(x, axis)`);
* `Softmax.maxRel_negInf`: `tl.maximum(-inf, b) = b` for finite `b`, and `Softmax.sub_negInf`:
  `-inf - y = -inf` for finite `y` (the first step of an online softmax, from `m = -inf`).
-/
namespace TVA.Softmax

open TVA.Math.Float TVA.Math.Float.F32

/-! ## The value is monotone in the order key -/

/-- The IEEE order key of a `float32` pattern. -/
def okey (x : Bits) : Int := ieeeOrderKey 8 23 x.toNat

theorem okey_eq (x : Bits) :
    okey x = if x.toNat / 2 ^ 31 % 2 = 1 then -((x.toNat % 2 ^ 31 : Nat) : Int) - 1
      else ((x.toNat % 2 ^ 31 : Nat) : Int) := by
  have hx := x.isLt
  simp only [okey, ieeeOrderKey, floatFields, Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]
  have e : x.toNat / 2 ^ 23 % 2 ^ 8 * 2 ^ 23 + x.toNat % 2 ^ 23 = x.toNat % 2 ^ 31 := by omega
  by_cases hs : x.toNat / 2 ^ (8 + 23) % 2 = 1
  · simp only [hs, decide_true, ite_true, show (8 + 23 : Nat) = 31 from rfl] at *
    rw [e]
  · simp only [show (8 + 23 : Nat) = 31 from rfl] at hs
    simp only [show (8 + 23 : Nat) = 31 from rfl, hs, decide_false, Bool.false_eq_true, ite_false]
    rw [e]

/-- The magnitude of a pattern with low 31 bits `n`, in units of `2^-149`. -/
def magN (n : Nat) : Nat :=
  if n / 2 ^ 23 = 0 then n % 2 ^ 23 else (2 ^ 23 + n % 2 ^ 23) * 2 ^ (n / 2 ^ 23 - 1)

theorem magRat_eq_magN (x : Bits) : magRat x = (magN (x.toNat % 2 ^ 31) : ℚ) / 2 ^ 149 := by
  have hxl := x.isLt
  have he : expField x = x.toNat % 2 ^ 31 / 2 ^ 23 := by
    simp only [expField, Nat.shiftRight_eq_div_pow]; omega
  have hm : mantField x = x.toNat % 2 ^ 31 % 2 ^ 23 := by simp only [mantField]; omega
  unfold magN
  by_cases h0 : x.toNat % 2 ^ 31 / 2 ^ 23 = 0
  · rw [magRat_of_exp_zero (by rw [he, h0]), ite_eq_left h0, hm]
  · rw [magRat_of_exp_ne (by rw [he]; exact h0), ite_eq_right h0, hm, he]
    rw [show ((x.toNat % 2 ^ 31 / 2 ^ 23 : ℕ) : ℤ) - 150 = ((x.toNat % 2 ^ 31 / 2 ^ 23 - 1 : ℕ) : ℤ) - 149 by
        omega, zpow_sub₀ (by norm_num), zpow_natCast]
    push_cast; ring

theorem magN_mono {n n' : Nat} (h : n ≤ n') : magN n ≤ magN n' := by
  unfold magN
  by_cases h0 : n / 2 ^ 23 = 0
  · rw [ite_eq_left h0]
    by_cases h0' : n' / 2 ^ 23 = 0
    · rw [ite_eq_left h0']; omega
    · rw [ite_eq_right h0']
      have : 1 ≤ 2 ^ (n' / 2 ^ 23 - 1) := Nat.one_le_two_pow
      calc n % 2 ^ 23 ≤ 2 ^ 23 := by omega
        _ ≤ (2 ^ 23 + n' % 2 ^ 23) * 1 := by omega
        _ ≤ _ := Nat.mul_le_mul_left _ this
  · have h0' : ¬ n' / 2 ^ 23 = 0 := by
      intro h'; apply h0; have := Nat.div_le_div_right (c := 2 ^ 23) h; omega
    rw [ite_eq_right h0, ite_eq_right h0']
    have hee : n / 2 ^ 23 ≤ n' / 2 ^ 23 := Nat.div_le_div_right h
    rcases Nat.eq_or_lt_of_le hee with heq | hlt
    · rw [heq]
      apply Nat.mul_le_mul_right
      have : n % 2 ^ 23 ≤ n' % 2 ^ 23 := by
        have h1 := Nat.div_add_mod n (2 ^ 23); have h2 := Nat.div_add_mod n' (2 ^ 23); rw [heq] at h1; omega
      omega
    · have hp : 2 ^ (n / 2 ^ 23 - 1) * 2 ≤ 2 ^ (n' / 2 ^ 23 - 1) := by
        rw [← pow_succ]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
      calc (2 ^ 23 + n % 2 ^ 23) * 2 ^ (n / 2 ^ 23 - 1) ≤ 2 ^ 24 * 2 ^ (n / 2 ^ 23 - 1) :=
            Nat.mul_le_mul_right _ (by omega)
        _ = 2 ^ 23 * (2 ^ (n / 2 ^ 23 - 1) * 2) := by ring
        _ ≤ 2 ^ 23 * 2 ^ (n' / 2 ^ 23 - 1) := Nat.mul_le_mul_left _ hp
        _ ≤ (2 ^ 23 + n' % 2 ^ 23) * 2 ^ (n' / 2 ^ 23 - 1) := Nat.mul_le_mul_right _ (by omega)

theorem magRat_mono {x y : Bits} (h : x.toNat % 2 ^ 31 ≤ y.toNat % 2 ^ 31) : magRat x ≤ magRat y := by
  rw [magRat_eq_magN, magRat_eq_magN]
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact_mod_cast magN_mono h

/-- **The value is monotone in the key** (`float32` patterns). -/
theorem toRat_le_of_okey_le {x y : Bits} (h : okey x ≤ okey y) : toRat x ≤ toRat y := by
  have hxl := x.isLt; have hyl := y.isLt
  rw [okey_eq, okey_eq] at h
  have hmx := magRat_nonneg x; have hmy := magRat_nonneg y
  unfold toRat
  simp only [sign, ge_iff_le, decide_eq_true_eq]
  by_cases hsx : 2 ^ 31 ≤ x.toNat <;> by_cases hsy : 2 ^ 31 ≤ y.toNat
  · rw [ite_eq_left hsx, ite_eq_left hsy]
    rw [ite_eq_left (by omega), ite_eq_left (by omega)] at h
    have := magRat_mono (x := y) (y := x) (by omega); linarith
  · rw [ite_eq_left hsx, ite_eq_right hsy]; linarith
  · rw [ite_eq_right hsx, ite_eq_left hsy]
    rw [ite_eq_right (by omega), ite_eq_left (by omega)] at h; omega
  · rw [ite_eq_right hsx, ite_eq_right hsy]
    rw [ite_eq_right (by omega), ite_eq_right (by omega)] at h
    exact magRat_mono (by omega)

/-- The same for `FloatV .binary32`. -/
theorem toRat_le_of_key_le {x y : F32}
    (h : ieeeOrderKey 8 23 x.pat ≤ ieeeOrderKey 8 23 y.pat) : x.toRat ≤ y.toRat := by
  have e1 : FloatV.toRat x = F32.toRat x.bits := (toRat_binary32 x.bits).symm
  have e2 : FloatV.toRat y = F32.toRat y.bits := (toRat_binary32 y.bits).symm
  rw [e1, e2]
  exact toRat_le_of_okey_le (x := x.bits) (y := y.bits) h

/-! ## `max` by value -/

section Max
open TVA.Math.Float.Format

theorem isNaN_false {a : F32} (ha : a.isFinite = true) : binary32.isNaN a.pat = false :=
  (FloatV.isFinite_iff.mp ha).1

/-- `binary32.lt` on non-NaN patterns, not both zero, is the order of the keys. -/
theorem lt_eq_key {a b : Nat} (ha : binary32.isNaN a = false) (hb : binary32.isNaN b = false)
    (hz : ¬ (binary32.isZero a = true ∧ binary32.isZero b = true)) :
    binary32.lt a b = decide (ieeeOrderKey 8 23 a < ieeeOrderKey 8 23 b) := by
  have hz' : (binary32.isZero a && binary32.isZero b) = false := by
    cases h1 : binary32.isZero a <;> cases h2 : binary32.isZero b <;> simp_all
  unfold Format.lt Format.compare
  rw [ha, hb, hz']
  simp only [Bool.or_self, Bool.false_eq_true, ite_false, binary32_e, binary32_m]
  by_cases hlt : ieeeOrderKey 8 23 a < ieeeOrderKey 8 23 b
  · simp [compare_lt_iff_lt.2 hlt, hlt]
  · rcases lt_or_eq_of_le (not_lt.1 hlt) with hgt | heq
    · simp [compare_gt_iff_gt.2 hgt, hlt]
    · simp [heq]

/-- A zero pattern of `float32`: finite, value `0`. -/
theorem zeroPat_spec {n : Nat} (hn : n = 0 ∨ n = binary32.negZero) :
    (FloatV.ofPat n : F32).isFinite = true ∧ (FloatV.ofPat n : F32).toRat = 0 := by
  rcases hn with rfl | rfl <;> exact ⟨by decide, by decide⟩

/-- **`FloatV.max` of finite `float32` values, by value.** -/
theorem max_val {a b : F32} (ha : a.isFinite = true) (hb : b.isFinite = true) :
    (FloatV.max a b).isFinite = true ∧
      ((FloatV.max a b).toRat = a.toRat ∨ (FloatV.max a b).toRat = b.toRat) ∧
      a.toRat ≤ (FloatV.max a b).toRat ∧ b.toRat ≤ (FloatV.max a b).toRat := by
  have hna := isNaN_false ha
  have hnb := isNaN_false hb
  unfold FloatV.max Format.fmax
  rw [hna, hnb]
  simp only [Bool.and_self, Bool.false_eq_true, ite_false]
  by_cases hz : binary32.isZero a.pat = true ∧ binary32.isZero b.pat = true
  · have hzab : (binary32.isZero a.pat && binary32.isZero b.pat) = true := by simp [hz.1, hz.2]
    rw [ite_eq_left hzab]
    have hva : a.toRat = 0 := FloatV.toRat_of_isZero hz.1
    have hvb : b.toRat = 0 := FloatV.toRat_of_isZero hz.2
    obtain ⟨h1, h2⟩ := zeroPat_spec (n := if (binary32.sign a.pat && binary32.sign b.pat) = true then
      binary32.negZero else 0) (by split_ifs <;> simp)
    refine ⟨h1, Or.inl (by rw [h2, hva]), by rw [h2, hva], by rw [h2, hvb]⟩
  · have hzab : (binary32.isZero a.pat && binary32.isZero b.pat) = false := by
      cases h1 : binary32.isZero a.pat <;> cases h2 : binary32.isZero b.pat <;> simp_all
    rw [ite_eq_right (by rw [hzab]; simp), lt_eq_key hna hnb hz]
    by_cases hk : ieeeOrderKey 8 23 a.pat < ieeeOrderKey 8 23 b.pat
    · rw [ite_eq_left (by simpa using hk), FloatV.ofPat_pat]
      exact ⟨hb, Or.inr rfl, toRat_le_of_key_le hk.le, le_rfl⟩
    · rw [ite_eq_right (by simpa using hk), FloatV.ofPat_pat]
      exact ⟨ha, Or.inl rfl, le_rfl, toRat_le_of_key_le (not_lt.1 hk)⟩

/-- **`tl.maximum` of finite `float32` values, by value**: finite, one of the operands, and at
least both. -/
theorem maxRel_val {a b r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (h : FloatV.MaxRel true a b r) :
    r.isFinite = true ∧ (r.toRat = a.toRat ∨ r.toRat = b.toRat) ∧ a.toRat ≤ r.toRat ∧
      b.toRat ≤ r.toRat := by
  rcases h with h | ⟨hza, hzb, hr⟩
  · rw [ite_eq_left rfl] at h
    obtain ⟨h1, h2, h3, h4⟩ := max_val ha hb
    have hn : (FloatV.max a b).isNaN = false := isNaN_false h1
    rw [h.eq hn]
    exact ⟨h1, h2, h3, h4⟩
  · have hva : a.toRat = 0 := FloatV.toRat_of_isZero hza
    have hvb : b.toRat = 0 := FloatV.toRat_of_isZero hzb
    rcases hr with rfl | rfl
    · exact ⟨ha, Or.inl rfl, le_rfl, by rw [hva, hvb]⟩
    · exact ⟨hb, Or.inr rfl, by rw [hva, hvb], le_rfl⟩

/-- **A `tl.max` tree over finite `float32` leaves, by value**: finite, equal to a leaf, and at
least every leaf, whatever the tree. -/
theorem maxTree_val {x : Nat → F32} {t : Tree} {r : F32} (h : t.Eval (FloatV.MaxRel true) x r)
    (hx : ∀ i ∈ t.leaves, (x i).isFinite = true) :
    r.isFinite = true ∧ (∃ i ∈ t.leaves, r.toRat = (x i).toRat) ∧
      ∀ i ∈ t.leaves, (x i).toRat ≤ r.toRat := by
  induction h with
  | leaf i =>
    refine ⟨hx i (by simp [Tree.leaves]), ⟨i, by simp [Tree.leaves], rfl⟩, fun j hj => ?_⟩
    simp only [Tree.leaves, List.mem_singleton] at hj
    rw [hj]
  | @node l r' a b c _ _ hc ihl ihr =>
    have hxl : ∀ i ∈ l.leaves, (x i).isFinite = true :=
      fun i hi => hx i (by simp [Tree.leaves, hi])
    have hxr : ∀ i ∈ r'.leaves, (x i).isFinite = true :=
      fun i hi => hx i (by simp [Tree.leaves, hi])
    obtain ⟨hfa, ⟨ia, hia, hva⟩, hba⟩ := ihl hxl
    obtain ⟨hfb, ⟨ib, hib, hvb⟩, hbb⟩ := ihr hxr
    obtain ⟨hfc, hcv, hac, hbc⟩ := maxRel_val hfa hfb hc
    refine ⟨hfc, ?_, fun i hi => ?_⟩
    · rcases hcv with h' | h'
      · exact ⟨ia, by simp [Tree.leaves, hia], by rw [h', hva]⟩
      · exact ⟨ib, by simp [Tree.leaves, hib], by rw [h', hvb]⟩
    · simp only [Tree.leaves, List.mem_append] at hi
      rcases hi with hi | hi
      · exact (hba i hi).trans hac
      · exact (hbb i hi).trans hbc

/-! ## Starting from `-∞` -/

/-- `+∞` in `float32`. -/
def posInf32 : F32 := FloatV.ofPat 0x7F800000
/-- `-∞` in `float32`. -/
def negInf32 : F32 := FloatV.ofPat 0xFF800000

theorem negInf32_isFinite : negInf32.isFinite = false := by decide

/-- `tl.zeros(...) - float("inf")` is `-∞`. -/
theorem sub_zero_posInf : FloatV.sub (FloatV.ofPat 0 : F32) posInf32 = negInf32 := by decide

/-- `-∞ - y = -∞` for finite `y`. -/
theorem sub_negInf {y : F32} (hy : y.isFinite = true) : FloatV.sub negInf32 y = negInf32 := by
  have h1 : binary32.isNaN negInf32.pat = false := by decide
  have h2 : binary32.isInf negInf32.pat = true := by decide
  have hfin : binary32.isFinite (binary32.negate y.pat) = true := by
    rw [isFinite_negate y.pat_lt]; exact hy
  have h3 : binary32.isNaN (binary32.negate y.pat) = false := by
    unfold Format.isFinite at hfin; revert hfin; cases binary32.isNaN (binary32.negate y.pat) <;> simp
  have h4 : binary32.isInf (binary32.negate y.pat) = false := by
    unfold Format.isFinite at hfin; revert hfin; cases binary32.isInf (binary32.negate y.pat) <;> simp
  unfold FloatV.sub Format.sub Format.add
  rw [h1, h2, h3, h4]
  simp only [Bool.or_self, Bool.false_eq_true, ite_false, Bool.and_false, ite_true]
  exact FloatV.ofPat_pat negInf32

/-- The key of a `float32` pattern `n < 2^32`. -/
theorem key_eq {n : Nat} (hn : n < 2 ^ 32) :
    ieeeOrderKey 8 23 n = if n / 2 ^ 31 % 2 = 1 then -((n % 2 ^ 31 : Nat) : Int) - 1
      else ((n % 2 ^ 31 : Nat) : Int) := by
  have := okey_eq (BitVec.ofNat 32 n)
  rwa [okey, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hn] at this

/-- A finite `float32` pattern has an exponent field below `255`. -/
theorem lowBits_lt_of_finite {n : Nat} (h : binary32.isFinite n = true) : n % 2 ^ 31 < 0x7F800000 := by
  unfold Format.isFinite Format.isNaN Format.isInf Format.expField Format.fracField at h
  simp only [binary32_special, binary32_m, binary32_e, binary32_expMax, Nat.shiftRight_eq_div_pow] at h
  by_cases he : n / 2 ^ 23 % 2 ^ 8 = 255
  · rw [he] at h
    cases hf : (n % 2 ^ 23 == 0) <;> simp_all
  · omega

/-- `-∞` has the least key of the finite patterns. -/
theorem key_negInf_lt {b : F32} (hb : b.isFinite = true) :
    ieeeOrderKey 8 23 negInf32.pat < ieeeOrderKey 8 23 b.pat := by
  have hlt : b.pat < 2 ^ 32 := b.pat_lt
  have hlow := lowBits_lt_of_finite (n := b.pat) hb
  have hni : negInf32.pat = 0xFF800000 := by decide
  rw [hni, key_eq (by norm_num), key_eq hlt]
  split_ifs <;> omega

/-- **`tl.maximum(-∞, b) = b`** for finite `b`. -/
theorem maxRel_negInf {b r : F32} (hb : b.isFinite = true) (h : FloatV.MaxRel true negInf32 b r) :
    r = b := by
  rcases h with h | ⟨hz, -, -⟩
  · rw [ite_eq_left rfl] at h
    have hna : binary32.isNaN negInf32.pat = false := by decide
    have hza : binary32.isZero negInf32.pat = false := by decide
    have hmax : FloatV.max negInf32 b = b := by
      unfold FloatV.max Format.fmax
      rw [hna, isNaN_false hb, hza, lt_eq_key hna (isNaN_false hb) (by simp [hza])]
      simp only [Bool.and_self, Bool.false_eq_true, ite_false, Bool.false_and]
      rw [ite_eq_left (by simpa using key_negInf_lt hb)]
      exact FloatV.ofPat_pat b
    rw [hmax] at h
    exact h.eq (isNaN_false hb)
  · have : negInf32.isZero = false := by decide
    rw [this] at hz; cases hz

end Max

end TVA.Softmax
