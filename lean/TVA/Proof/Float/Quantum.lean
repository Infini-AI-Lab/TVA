import TVA.Proof.Float.Constants

/-!
# Rounding, step by step: the quantum and the significand

`Format.roundRat f rnd sat zs q` (`TVA.Base.Float.Round`) rounds an exact rational `q` into the
format `f` in three steps, for `a = |q| > 0`:
1. the *quantum* `2^Q`, `Q = qexp a = max(⌊log₂ a⌋, emin) − m`: the spacing of the representable
   values around `a`;
2. the *significand* `N = roundSig rnd a Q`: `a / 2^Q` rounded to an integer (`rnat`, by `rint`);
3. the magnitude `pack N Q` of the value `N · 2^Q`, or an overflow when `N · 2^Q > maxVal`
   (`encode`).

This file proves each step correct:

* `rnat_rn_err` (to nearest, within `1/2`), `rnat_rz_err` (toward zero, the floor);
* `qlog2_spec`, `qlog2_eq`, `qexp_eq`: the definitions' `qlog2` is `⌊log₂ a⌋` (Mathlib's
  `Int.log 2 a`), computed from bit lengths;
* `half_quantum_le`: half the quantum is at most `a·u + η`, the bound of one rounding, and at most
  `u·a` in the normal range (`half_quantum_normal`); it is exactly `η` below it
  (`half_quantum_sub`);
* `magVal_pack`: the packed significand has the value `N · 2^Q`; `pack_le_maxMag`: it is a finite
  magnitude when that value is at most `maxVal`;
* `roundSig_rn_err` (to nearest: within half a quantum), `roundSig_rz_err` (toward zero: below
  `a` by less than a quantum), `roundSig_bounds` (the significand satisfies `pack`'s conditions);
* `roundSig_le_maxVal`, `roundMag_of_le`: a value at most `maxVal` does not overflow;
* `roundSig_exact`: a multiple of the quantum rounds to itself.

Read `TVA.Proof.Float.Round` next: it assembles these into the error of one rounding.
-/
namespace TVA.Format

/-- Rounding the integer part `k` up or not gives at least `k`. -/
theorem rint_ge (rnd : RoundMode) (k : ℕ) (c : Ordering) : k ≤ rint rnd k c := by
  unfold rint; split <;> (try split_ifs) <;> omega

/-- Rounding the integer part `k` gives at most `k + 1`. -/
theorem rint_le (rnd : RoundMode) (k : ℕ) (c : Ordering) : rint rnd k c ≤ k + 1 := by
  unfold rint; split <;> (try split_ifs) <;> omega

/-- Toward zero, `rint` keeps the integer part. -/
theorem rint_rz (k : ℕ) (c : Ordering) : rint .rz k c = k := rfl

/-- Core's `Rat.floor`, as a natural number, is Mathlib's `⌊x⌋₊`. -/
theorem floor_toNat (x : ℚ) : (Rat.floor x).toNat = ⌊x⌋₊ := by
  rw [← Int.floor_toNat]; rfl

/-- `rnat` with Mathlib's floor `⌊x⌋₊`. -/
theorem rnat_eq (rnd : RoundMode) (x : ℚ) :
    rnat rnd x = rint rnd ⌊x⌋₊ (cmpq (x - ⌊x⌋₊) (1 / 2)) := by
  unfold rnat; rw [floor_toNat]

/-- Rounding `x ≥ 0` to the nearest integer moves it by at most `1/2`. -/
theorem rnat_rn_err {x : ℚ} (hx : 0 ≤ x) : |(rnat .rn x : ℚ) - x| ≤ 1 / 2 := by
  have h1 := Nat.floor_le hx
  have h2 := Nat.lt_floor_add_one x
  rw [rnat_eq]; unfold cmpq
  rw [abs_le]
  split_ifs with ha hb
  · simp only [rint]; constructor <;> linarith
  · simp only [rint]
    split_ifs <;> push_cast <;> constructor <;> linarith
  · simp only [rint]; push_cast
    have : 1 / 2 < x - ⌊x⌋₊ := lt_of_le_of_ne (not_lt.mp ha) (Ne.symm hb)
    constructor <;> linarith

/-- Rounding `x ≥ 0` toward zero gives its floor. -/
theorem rnat_rz_err {x : ℚ} (hx : 0 ≤ x) : (rnat .rz x : ℚ) ≤ x ∧ x < rnat .rz x + 1 := by
  rw [rnat_eq, rint_rz]
  exact ⟨Nat.floor_le hx, Nat.lt_floor_add_one x⟩

/-- Rounding `x ≥ K` gives at least `K`. -/
theorem le_rnat (rnd : RoundMode) {x : ℚ} {K : ℕ} (h : (K : ℚ) ≤ x) : K ≤ rnat rnd x :=
  rnat_eq rnd x ▸ (Nat.le_floor h).trans (rint_ge _ _ _)

/-- Rounding `0 ≤ x ≤ K` gives at most `K`. -/
theorem rnat_le (rnd : RoundMode) {x : ℚ} (hx : 0 ≤ x) {K : ℕ} (h : x ≤ K) : rnat rnd x ≤ K := by
  have hk : ⌊x⌋₊ ≤ K := Nat.floor_le_of_le h
  rw [rnat_eq]
  rcases hk.lt_or_eq with hlt | heq
  · exact (rint_le _ _ _).trans hlt
  · -- `⌊x⌋ = K`, so `x = K` and the fraction is `0`
    have hfl := Nat.floor_le hx
    rw [heq] at hfl ⊢
    have hxK : x = K := le_antisymm h hfl
    have : cmpq (x - (K : ℚ)) (1 / 2) = .lt := by
      unfold cmpq; rw [ite_eq_left (by rw [hxK]; norm_num)]
    rw [this]
    cases rnd <;> rfl

variable {f : Format}

/-! ## The quantum -/

/-- `2^⌊log₂ a⌋ ≤ a` for `a > 0` (Mathlib's `Int.log`). -/
theorem two_zpow_log_le {a : ℚ} (ha : 0 < a) : (2 : ℚ) ^ Int.log 2 a ≤ a := by
  have := Int.zpow_log_le_self (b := 2) (R := ℚ) (by norm_num) ha
  simpa using this

/-- `a < 2^(⌊log₂ a⌋ + 1)`. -/
theorem lt_two_zpow_log {a : ℚ} : a < (2 : ℚ) ^ (Int.log 2 a + 1) := by
  have := Int.lt_zpow_succ_log_self (b := 2) (R := ℚ) (by norm_num) a
  simpa using this

/-- `a < 2^k` gives `⌊log₂ a⌋ < k`. -/
theorem log_lt_of_lt {a : ℚ} (ha : 0 < a) {k : ℤ} (h : a < (2 : ℚ) ^ k) : Int.log 2 a < k := by
  have := (Int.lt_zpow_iff_log_lt (b := 2) (R := ℚ) (by norm_num) ha).mp (by simpa using h)
  exact this

/-- `2^k ≤ a` gives `k ≤ ⌊log₂ a⌋`. -/
theorem le_log_of_le {a : ℚ} (ha : 0 < a) {k : ℤ} (h : (2 : ℚ) ^ k ≤ a) : k ≤ Int.log 2 a := by
  exact (Int.zpow_le_iff_le_log (b := 2) (R := ℚ) (by norm_num) ha).mp (by simpa using h)

/-- The bit length of `n > 0` brackets it: `2^log₂ n ≤ n < 2^(log₂ n + 1)`. -/
private theorem log2_bounds {n : ℕ} (hn : n ≠ 0) :
    ((2 : ℚ) ^ (Nat.log2 n : ℤ) ≤ n) ∧ ((n : ℚ) < (2 : ℚ) ^ ((Nat.log2 n : ℤ) + 1)) := by
  have h1 := Nat.log2_self_le hn
  have h2 := Nat.lt_log2_self (n := n)
  constructor
  · rw [zpow_natCast]; exact_mod_cast h1
  · rw [show ((Nat.log2 n : ℤ) + 1) = ((Nat.log2 n + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]
    exact_mod_cast h2

/-- **`qlog2` is `⌊log₂ a⌋`**: `2^qlog2 a ≤ a < 2^(qlog2 a + 1)` for `a > 0`. -/
theorem qlog2_spec {a : ℚ} (ha : 0 < a) :
    (2 : ℚ) ^ qlog2 a ≤ a ∧ a < (2 : ℚ) ^ (qlog2 a + 1) := by
  have hn : 0 < a.num := Rat.num_pos.mpr ha
  have hnn : a.num.toNat ≠ 0 := by omega
  have hd : a.den ≠ 0 := a.den_nz
  have hnum : ((a.num.toNat : ℕ) : ℚ) = (a.num : ℚ) := by
    have : ((a.num.toNat : ℕ) : ℤ) = a.num := Int.toNat_of_nonneg hn.le
    exact_mod_cast this
  have hdpos : (0 : ℚ) < a.den := by exact_mod_cast Nat.pos_of_ne_zero hd
  have ha' : a = (a.num.toNat : ℚ) / a.den := by rw [hnum]; exact (Rat.num_div_den a).symm
  obtain ⟨n1, n2⟩ := log2_bounds hnn
  obtain ⟨d1, d2⟩ := log2_bounds hd
  set p : ℤ := (Nat.log2 a.num.toNat : ℤ)
  set q : ℤ := (Nat.log2 a.den : ℤ)
  set N : ℚ := (a.num.toNat : ℚ)
  set D : ℚ := (a.den : ℚ)
  have h2 : (0 : ℚ) < 2 := by norm_num
  -- `2^(p - q - 1) < a < 2^(p - q + 1)`
  have hlo : (2 : ℚ) ^ (p - q - 1) < a := by
    rw [ha', lt_div_iff₀ hdpos]
    calc (2 : ℚ) ^ (p - q - 1) * D < (2 : ℚ) ^ (p - q - 1) * (2 : ℚ) ^ (q + 1) :=
          mul_lt_mul_of_pos_left d2 (zpow_pos h2 _)
      _ = (2 : ℚ) ^ p := by rw [← zpow_add₀ h2.ne']; ring_nf
      _ ≤ N := n1
  have hhi : a < (2 : ℚ) ^ (p - q + 1) := by
    rw [ha', div_lt_iff₀ hdpos]
    calc N < (2 : ℚ) ^ (p + 1) := n2
      _ = (2 : ℚ) ^ (p - q + 1) * (2 : ℚ) ^ q := by rw [← zpow_add₀ h2.ne']; ring_nf
      _ ≤ (2 : ℚ) ^ (p - q + 1) * D := mul_le_mul_of_nonneg_left d1 (zpow_pos h2 _).le
  have hq : qlog2 a = if (2 : ℚ) ^ (p - q) ≤ a then p - q else p - q - 1 := by
    unfold qlog2; simp only [pow2_eq]; rfl
  rw [hq]
  split_ifs with h
  · exact ⟨h, hhi⟩
  · refine ⟨hlo.le, ?_⟩
    rw [show p - q - 1 + 1 = p - q by ring]; exact not_le.mp h

/-- The definitions' `qlog2 a` is Mathlib's `Int.log 2 a` for `a > 0`. -/
theorem qlog2_eq {a : ℚ} (ha : 0 < a) : qlog2 a = Int.log 2 a := by
  obtain ⟨h1, h2⟩ := qlog2_spec ha
  have := le_log_of_le ha h1
  have := log_lt_of_lt ha h2
  omega

/-- The quantum exponent with Mathlib's `Int.log`. -/
theorem qexp_eq {a : ℚ} (ha : 0 < a) : f.qexp a = max (Int.log 2 a) f.emin - f.m := by
  unfold qexp; rw [qlog2_eq ha]

/-- `a / 2^Q · 2^Q = a`. -/
theorem scaled_mul {a : ℚ} (Q : ℤ) : a * (2 : ℚ) ^ (-Q) * (2 : ℚ) ^ Q = a := by
  rw [zpow_neg, mul_assoc, inv_mul_cancel₀ (by positivity), mul_one]

/-- The scaled value is below `2^(m+1)`. -/
theorem scaled_lt {a : ℚ} (ha : 0 < a) :
    a * (2 : ℚ) ^ (-f.qexp a) < (2 : ℚ) ^ ((f.m : ℤ) + 1) := by
  have h1 := lt_two_zpow_log (a := a)
  have h2 : (2 : ℚ) ^ (Int.log 2 a + 1) ≤ (2 : ℚ) ^ (f.qexp a + f.m + 1) := by
    apply zpow_le_zpow_right₀ (by norm_num); rw [qexp_eq ha]; have := le_max_left (Int.log 2 a) f.emin; linarith
  have h3 : (2 : ℚ) ^ (f.qexp a + f.m + 1) = (2 : ℚ) ^ ((f.m : ℤ) + 1) * (2 : ℚ) ^ (f.qexp a) := by
    rw [← zpow_add₀ (by norm_num)]; ring_nf
  have hp : (0 : ℚ) < (2 : ℚ) ^ (-f.qexp a) := by positivity
  calc a * (2 : ℚ) ^ (-f.qexp a) < (2 : ℚ) ^ ((f.m : ℤ) + 1) * (2 : ℚ) ^ (f.qexp a) * (2 : ℚ) ^ (-f.qexp a) := by
        rw [← h3]; exact mul_lt_mul_of_pos_right (h1.trans_le h2) hp
    _ = (2 : ℚ) ^ ((f.m : ℤ) + 1) := by rw [mul_assoc, ← zpow_add₀ (by norm_num)]; simp

/-- In the normal range the scaled value is at least `2^m`. -/
theorem scaled_ge {a : ℚ} (ha : 0 < a) (hn : f.emin ≤ Int.log 2 a) :
    (2 : ℚ) ^ (f.m : ℤ) ≤ a * (2 : ℚ) ^ (-f.qexp a) := by
  have hq : f.qexp a = Int.log 2 a - f.m := by rw [qexp_eq ha, max_eq_left hn]
  have h1 := two_zpow_log_le ha
  have hp : (0 : ℚ) < (2 : ℚ) ^ (-f.qexp a) := by positivity
  calc (2 : ℚ) ^ (f.m : ℤ) = (2 : ℚ) ^ Int.log 2 a * (2 : ℚ) ^ (-f.qexp a) := by
        rw [← zpow_add₀ (by norm_num), hq]; ring_nf
    _ ≤ a * (2 : ℚ) ^ (-f.qexp a) := mul_le_mul_of_nonneg_right h1 hp.le

/-- Below the normal range the quantum is `2^(emin − m)` and the scaled value is below `2^m`. -/
theorem scaled_lt_sub {a : ℚ} (ha : 0 < a) (hs : Int.log 2 a < f.emin) :
    f.qexp a = f.emin - f.m ∧ a * (2 : ℚ) ^ (-f.qexp a) < (2 : ℚ) ^ (f.m : ℤ) := by
  have hq : f.qexp a = f.emin - f.m := by rw [qexp_eq ha, max_eq_right hs.le]
  refine ⟨hq, ?_⟩
  have h1 := lt_two_zpow_log (a := a)
  have h2 : (2 : ℚ) ^ (Int.log 2 a + 1) ≤ (2 : ℚ) ^ f.emin := zpow_le_zpow_right₀ (by norm_num) (by omega)
  have hp : (0 : ℚ) < (2 : ℚ) ^ (-f.qexp a) := by positivity
  calc a * (2 : ℚ) ^ (-f.qexp a) < (2 : ℚ) ^ f.emin * (2 : ℚ) ^ (-f.qexp a) :=
        mul_lt_mul_of_pos_right (h1.trans_le h2) hp
    _ = (2 : ℚ) ^ (f.m : ℤ) := by rw [← zpow_add₀ (by norm_num), hq]; ring_nf

/-- In the normal range `2^(m + Q) ≤ a`. -/
theorem quantum_mul_le {a : ℚ} (ha : 0 < a) (hn : f.emin ≤ Int.log 2 a) :
    (2 : ℚ) ^ ((f.m : ℤ) + f.qexp a) ≤ a := by
  have h := scaled_ge ha hn
  have hp : (0 : ℚ) < (2 : ℚ) ^ (f.qexp a) := by positivity
  have := mul_le_mul_of_nonneg_right h hp.le
  rwa [scaled_mul, ← zpow_add₀ (by norm_num)] at this

/-- Half the quantum is `u · 2^(m + Q)`. -/
theorem half_quantum_eq (Q : ℤ) : (2 : ℚ) ^ Q / 2 = f.u * (2 : ℚ) ^ ((f.m : ℤ) + Q) := by
  rw [u_eq, ← zpow_add₀ (by norm_num), div_eq_iff (by norm_num), ← zpow_add_one₀ (by norm_num)]
  ring_nf

/-- Half the quantum is at most `u·a` in the normal range. -/
theorem half_quantum_normal {a : ℚ} (ha : 0 < a) (hn : f.emin ≤ Int.log 2 a) :
    (2 : ℚ) ^ (f.qexp a) / 2 ≤ f.u * a := by
  rw [half_quantum_eq]
  exact mul_le_mul_of_nonneg_left (quantum_mul_le ha hn) u_nonneg

/-- Half the quantum is `η` below the normal range. -/
theorem half_quantum_sub {a : ℚ} (ha : 0 < a) (hs : Int.log 2 a < f.emin) :
    (2 : ℚ) ^ (f.qexp a) / 2 = f.eta := by
  rw [(scaled_lt_sub ha hs).1, eta_eq]
  rw [div_eq_iff (by norm_num), ← zpow_add_one₀ (by norm_num)]; ring_nf

/-- Half the quantum of `a > 0` is at most `a·u + η`: the bound of one rounding to nearest. -/
theorem half_quantum_le {a : ℚ} (ha : 0 < a) : (2 : ℚ) ^ (f.qexp a) / 2 ≤ a * f.u + f.eta := by
  by_cases hn : f.emin ≤ Int.log 2 a
  · have := half_quantum_normal ha hn; have := eta_nonneg (f := f); linarith [mul_comm a f.u]
  · rw [half_quantum_sub ha (not_le.mp hn)]
    have : 0 ≤ a * f.u := mul_nonneg ha.le u_nonneg
    linarith

/-! ## Packing a significand -/

/-- `emin + bias = 1`, by definition of `emin`. -/
theorem bias_add_emin : f.emin + f.bias = 1 := by rw [emin_eq]; ring

/-- **The value of a packed significand.** -/
theorem magVal_pack {N : ℕ} {Q : ℤ} (hN : N ≤ 2 ^ (f.m + 1)) (hQ : f.emin - f.m ≤ Q)
    (hsub : N < 2 ^ f.m → Q = f.emin - f.m) : f.magVal (f.pack N Q) = N * (2 : ℚ) ^ Q := by
  have hP : 0 < 2 ^ f.m := by positivity
  have h2 : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  have hb := bias_add_emin (f := f)
  unfold pack
  split_ifs with h1 h2'
  · rw [magVal_eq, ite_eq_left (Nat.div_eq_of_lt h1), Nat.mod_eq_of_lt h1, hsub h1]
  · -- a normal significand
    have hE : 1 ≤ Q + f.m + f.bias := by linarith
    have hEn : ((Q + f.m + f.bias).toNat : ℤ) = Q + f.m + f.bias := Int.toNat_of_nonneg (by linarith)
    have hd : ((Q + f.m + f.bias).toNat * 2 ^ f.m + (N - 2 ^ f.m)) / 2 ^ f.m = (Q + f.m + f.bias).toNat := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt (by omega)]; simp
    have hm : ((Q + f.m + f.bias).toNat * 2 ^ f.m + (N - 2 ^ f.m)) % 2 ^ f.m = N - 2 ^ f.m := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
    have hE1 : (Q + f.m + f.bias).toNat ≠ 0 := by omega
    rw [magVal_eq, hd, hm, ite_eq_right hE1, hEn]
    congr 1
    · push_cast [show 2 ^ f.m ≤ N from not_lt.mp h1]; ring
    · congr 1; ring
  · -- the carry: `N = 2^(m+1)`
    have hN2 : N = 2 ^ (f.m + 1) := le_antisymm hN (not_lt.mp h2')
    have hE : 2 ≤ Q + f.m + 1 + f.bias := by linarith
    have hEn : ((Q + f.m + 1 + f.bias).toNat : ℤ) = Q + f.m + 1 + f.bias :=
      Int.toNat_of_nonneg (by linarith)
    have hd : ((Q + f.m + 1 + f.bias).toNat * 2 ^ f.m) / 2 ^ f.m = (Q + f.m + 1 + f.bias).toNat :=
      Nat.mul_div_cancel _ hP
    have hm : ((Q + f.m + 1 + f.bias).toNat * 2 ^ f.m) % 2 ^ f.m = 0 := Nat.mul_mod_left _ _
    have hE1 : (Q + f.m + 1 + f.bias).toNat ≠ 0 := by omega
    rw [magVal_eq, hd, hm, ite_eq_right hE1, hEn, hN2,
      show Q + (f.m : ℤ) + 1 + f.bias - f.bias - f.m = Q + 1 by ring, zpow_add_one₀ (by norm_num)]
    push_cast; ring

/-- A packed significand whose value is at most `maxVal` is a finite magnitude. -/
theorem pack_le_maxMag (hf : f.Regular) {N : ℕ} {Q : ℤ} (hN : N ≤ 2 ^ (f.m + 1))
    (hQ : f.emin - f.m ≤ Q) (hsub : N < 2 ^ f.m → Q = f.emin - f.m)
    (hmax : (N : ℚ) * (2 : ℚ) ^ Q ≤ f.maxVal) : f.pack N Q ≤ f.maxMag := by
  rw [← magVal_pack hN hQ hsub, ← magVal_maxMag hf] at hmax
  by_contra h
  exact absurd hmax (not_le.mpr (magVal_lt_magVal (not_le.mp h)))

/-! ## The rounded significand -/

/-- The rounded significand with `pow2` unfolded: `a · 2^-Q` rounded to an integer. -/
theorem roundSig_eq (rnd : RoundMode) (a : ℚ) (Q : ℤ) :
    roundSig rnd a Q = rnat rnd (a * (2 : ℚ) ^ (-Q)) := by
  unfold roundSig; rw [pow2_eq]

/-- The rounded significand satisfies `pack`'s conditions. -/
theorem roundSig_bounds (rnd : RoundMode) {a : ℚ} (ha : 0 < a) :
    roundSig rnd a (f.qexp a) ≤ 2 ^ (f.m + 1) ∧ f.emin - f.m ≤ f.qexp a ∧
      (roundSig rnd a (f.qexp a) < 2 ^ f.m → f.qexp a = f.emin - f.m) := by
  rw [roundSig_eq]
  have hx0 : 0 ≤ a * (2 : ℚ) ^ (-f.qexp a) := by positivity
  refine ⟨?_, ?_, ?_⟩
  · apply rnat_le rnd hx0
    have h := scaled_lt (f := f) ha
    have e : (2 : ℚ) ^ ((f.m : ℤ) + 1) = ((2 ^ (f.m + 1) : ℕ) : ℚ) := by
      rw [show ((f.m : ℤ) + 1) = ((f.m + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]; push_cast; rfl
    rw [← e]; exact h.le
  · rw [qexp_eq ha]; have := le_max_right (Int.log 2 a) f.emin; linarith
  · intro hlt
    by_contra hne
    have hn : f.emin ≤ Int.log 2 a := by
      by_contra h
      exact hne (scaled_lt_sub ha (not_le.mp h)).1
    have h := scaled_ge ha hn
    rw [zpow_natCast] at h
    have := le_rnat rnd (x := a * (2 : ℚ) ^ (-f.qexp a)) (K := 2 ^ f.m) (by exact_mod_cast h)
    omega

/-- In the normal range the rounded significand is at least `2^m`. -/
theorem roundSig_ge (rnd : RoundMode) {a : ℚ} (ha : 0 < a) (hn : f.emin ≤ Int.log 2 a) :
    2 ^ f.m ≤ roundSig rnd a (f.qexp a) := by
  rw [roundSig_eq]
  have h := scaled_ge ha hn
  rw [zpow_natCast] at h
  exact le_rnat rnd (x := a * (2 : ℚ) ^ (-f.qexp a)) (K := 2 ^ f.m) (by exact_mod_cast h)

/-- **To nearest, the rounded value is within half a quantum.** -/
theorem roundSig_rn_err {a : ℚ} (ha : 0 < a) (Q : ℤ) :
    |(roundSig .rn a Q : ℚ) * (2 : ℚ) ^ Q - a| ≤ (2 : ℚ) ^ Q / 2 := by
  rw [roundSig_eq]
  have hx0 : 0 ≤ a * (2 : ℚ) ^ (-Q) := by positivity
  have h := rnat_rn_err hx0
  have hp : (0 : ℚ) < (2 : ℚ) ^ Q := by positivity
  have e : (rnat .rn (a * (2 : ℚ) ^ (-Q)) : ℚ) * (2 : ℚ) ^ Q - a =
      ((rnat .rn (a * (2 : ℚ) ^ (-Q)) : ℚ) - a * (2 : ℚ) ^ (-Q)) * (2 : ℚ) ^ Q := by
    rw [sub_mul, scaled_mul]
  rw [e, abs_mul, abs_of_pos hp]
  calc _ ≤ 1 / 2 * (2 : ℚ) ^ Q := mul_le_mul_of_nonneg_right h hp.le
    _ = (2 : ℚ) ^ Q / 2 := by ring

/-- **Toward zero, the rounded value is below `a` by less than a quantum.** -/
theorem roundSig_rz_err {a : ℚ} (ha : 0 < a) (Q : ℤ) :
    (roundSig .rz a Q : ℚ) * (2 : ℚ) ^ Q ≤ a ∧ a < (roundSig .rz a Q : ℚ) * (2 : ℚ) ^ Q + (2 : ℚ) ^ Q := by
  rw [roundSig_eq]
  have hx0 : 0 ≤ a * (2 : ℚ) ^ (-Q) := by positivity
  obtain ⟨h1, h2⟩ := rnat_rz_err hx0
  have hp : (0 : ℚ) < (2 : ℚ) ^ Q := by positivity
  constructor
  · have := mul_le_mul_of_nonneg_right h1 hp.le
    rwa [scaled_mul] at this
  · have := mul_lt_mul_of_pos_right h2 hp
    rwa [scaled_mul, add_mul, one_mul] at this

/-- **No overflow in range.** For `a ≤ maxVal` the rounded value is at most `maxVal` (both
modes). -/
theorem roundSig_le_maxVal (hf : f.Regular) (rnd : RoundMode) {a : ℚ} (ha : 0 < a)
    (hmax : a ≤ f.maxVal) : (roundSig rnd a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a) ≤ f.maxVal := by
  set T : ℤ := (f.topExp : ℤ) - f.bias - f.m with hT
  -- the quantum is at most the top binade's
  have hQT : f.qexp a ≤ T := by
    have hL : Int.log 2 a < (f.topExp : ℤ) + 1 - f.bias :=
      log_lt_of_lt ha (hmax.trans_lt (maxVal_lt hf))
    have hE : f.emin ≤ (f.topExp : ℤ) - f.bias := by
      rw [emin_eq]; have := topExp_pos hf; omega
    rw [qexp_eq ha, hT]; omega
  -- `maxVal / 2^Q` is the integer `K`
  set K : ℕ := f.topSig * 2 ^ (T - f.qexp a).toNat with hK
  have hKQ : (K : ℚ) * (2 : ℚ) ^ (f.qexp a) = f.maxVal := by
    rw [hK, maxVal_eq]; push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), mul_assoc, ← zpow_add₀ (by norm_num)]
    congr 2; ring
  have hp : (0 : ℚ) < (2 : ℚ) ^ (f.qexp a) := by positivity
  have hx : a * (2 : ℚ) ^ (-f.qexp a) ≤ K := by
    have : a * (2 : ℚ) ^ (-f.qexp a) * (2 : ℚ) ^ (f.qexp a) ≤ (K : ℚ) * (2 : ℚ) ^ (f.qexp a) := by
      rw [scaled_mul, hKQ]; exact hmax
    exact le_of_mul_le_mul_right this hp
  have hN : roundSig rnd a (f.qexp a) ≤ K := by
    rw [roundSig_eq]; exact rnat_le rnd (by positivity) hx
  rw [← hKQ]
  exact mul_le_mul_of_nonneg_right (by exact_mod_cast hN) hp.le

/-- In range, `roundMag` is the packed rounded significand. -/
theorem roundMag_of_le (hf : f.Regular) (rnd : RoundMode) (sat : Bool) {a : ℚ} (ha : 0 < a)
    (hmax : a ≤ f.maxVal) :
    f.roundMag rnd sat a = f.pack (roundSig rnd a (f.qexp a)) (f.qexp a) := by
  unfold roundMag encode
  rw [pow2_eq, ite_eq_left (roundSig_le_maxVal hf rnd ha hmax)]

/-! ## Exact values -/

/-- An integer rounds to itself (both modes). -/
theorem rnat_natCast (rnd : RoundMode) (K : ℕ) : rnat rnd (K : ℚ) = K := by
  rw [rnat_eq, Nat.floor_natCast]
  have : cmpq ((K : ℚ) - (K : ℚ)) (1 / 2) = .lt := by unfold cmpq; rw [ite_eq_left (by norm_num)]
  rw [this]; cases rnd <;> rfl

/-- A value that is an integer multiple of its quantum rounds to itself (both modes). -/
theorem roundSig_exact (rnd : RoundMode) {a : ℚ} {Q : ℤ} {K : ℕ} (h : a * (2 : ℚ) ^ (-Q) = K) :
    (roundSig rnd a Q : ℚ) * (2 : ℚ) ^ Q = a := by
  rw [roundSig_eq, h, rnat_natCast, ← h, scaled_mul]

end TVA.Format
