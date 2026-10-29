import TVA.Math.Encoding
import Mathlib.Analysis.Real.Sqrt

/-!
# Rounding a rational number into a format

`roundRat f rnd sat zs q` is the pattern of format `f` that the exact value `q` rounds to
(IEEE 754-2019 §4.3), with:
- `rnd`: `.rn` (to nearest, ties to the even significand) or `.rz` (toward zero);
- `sat`: an overflow gives `±maxVal` (saturation, PTX `.satfinite`) instead of what the format
  does (`±∞` for `.ieee`, NaN for `.nanOnly`); under `.rz` an overflow gives `±maxVal` anyway;
- `zs`: the sign of a zero result of an exact `q = 0` (the sign rules of IEEE 754-2019 §6.3 are
  the caller's: `x + y` gives `-0` only for two negative zeros). A nonzero `q` that rounds to zero
  keeps its own sign.

The algorithm: for `a = |q| > 0`, the quantum is `2^Q` with `Q = max(⌊log₂ a⌋, emin) − m`
(`qexp`), the significand `N` is `a / 2^Q` rounded to an integer (`roundSig`, by `rint`), and the
result is `N · 2^Q` (`pack`), unless that exceeds `maxVal` (`encode`: overflow). `N < 2^m` only in
the subnormal range, and `N = 2^(m+1)` when rounding carries into the next binade.

`sqrtMag` rounds `√a` the same way, from the integer square root: `sqrtMag_rn_spec` (to nearest,
for `0 < a ≤ maxVal`, a finite magnitude within `√a·u + η` of the real `√a`, from
`sqrt_rint_err`: the rounded integer square root is within `1/2` of `√t`; `one_le_maxVal`). The definitions are in
`TVA.Math.Defs` (core Lean, run natively by `evidence/float-layer`); `qlog2_spec` shows that
their `qlog2` is `⌊log₂ a⌋`, and `qexp_eq` restates the quantum with Mathlib's `Int.log`.

**Error bounds** (to nearest): `roundRat_rn_err` (`|r − q| ≤ |q|·u + η` for `|q| ≤ maxVal`, and
the pattern is finite), `roundRat_rn_err_normal` (`|q|·u` for `|q| ≥ 2^emin`),
`roundRat_rn_err_of_finite` (relative to the result, whenever the result is finite, also for
`|q|` above `maxVal`); toward zero: `roundRat_rz_err`. Saturation: `roundMag_sat_of_gt` (a
magnitude above `maxVal` gives `maxMag`, either mode).
-/
namespace TVA.Math.Float

theorem rint_ge (rnd : RoundMode) (k : ℕ) (c : Ordering) : k ≤ rint rnd k c := by
  unfold rint; split <;> (try split_ifs) <;> omega

theorem rint_le (rnd : RoundMode) (k : ℕ) (c : Ordering) : rint rnd k c ≤ k + 1 := by
  unfold rint; split <;> (try split_ifs) <;> omega

theorem rint_rz (k : ℕ) (c : Ordering) : rint .rz k c = k := rfl

/-- Core's `Rat.floor`, as a natural number, is Mathlib's `⌊x⌋₊`. -/
theorem floor_toNat (x : ℚ) : (Rat.floor x).toNat = ⌊x⌋₊ := by
  rw [← Int.floor_toNat]; rfl

theorem rnat_eq (rnd : RoundMode) (x : ℚ) :
    rnat rnd x = rint rnd ⌊x⌋₊ (cmpq (x - ⌊x⌋₊) (1 / 2)) := by
  unfold rnat; rw [floor_toNat]

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

theorem rnat_rz_err {x : ℚ} (hx : 0 ≤ x) : (rnat .rz x : ℚ) ≤ x ∧ x < rnat .rz x + 1 := by
  rw [rnat_eq, rint_rz]
  exact ⟨Nat.floor_le hx, Nat.lt_floor_add_one x⟩

theorem le_rnat (rnd : RoundMode) {x : ℚ} {K : ℕ} (h : (K : ℚ) ≤ x) : K ≤ rnat rnd x :=
  rnat_eq rnd x ▸ (Nat.le_floor h).trans (rint_ge _ _ _)

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

namespace Format

variable {f : Format}

/-! ## The quantum -/

theorem two_zpow_log_le {a : ℚ} (ha : 0 < a) : (2 : ℚ) ^ Int.log 2 a ≤ a := by
  have := Int.zpow_log_le_self (b := 2) (R := ℚ) (by norm_num) ha
  simpa using this

theorem lt_two_zpow_log {a : ℚ} : a < (2 : ℚ) ^ (Int.log 2 a + 1) := by
  have := Int.lt_zpow_succ_log_self (b := 2) (R := ℚ) (by norm_num) a
  simpa using this

theorem log_lt_of_lt {a : ℚ} (ha : 0 < a) {k : ℤ} (h : a < (2 : ℚ) ^ k) : Int.log 2 a < k := by
  have := (Int.lt_zpow_iff_log_lt (b := 2) (R := ℚ) (by norm_num) ha).mp (by simpa using h)
  exact this

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

theorem qlog2_eq {a : ℚ} (ha : 0 < a) : qlog2 a = Int.log 2 a := by
  obtain ⟨h1, h2⟩ := qlog2_spec ha
  have := le_log_of_le ha h1
  have := log_lt_of_lt ha h2
  omega

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

theorem half_quantum_le {a : ℚ} (ha : 0 < a) : (2 : ℚ) ^ (f.qexp a) / 2 ≤ a * f.u + f.eta := by
  by_cases hn : f.emin ≤ Int.log 2 a
  · have := half_quantum_normal ha hn; have := eta_nonneg (f := f); linarith [mul_comm a f.u]
  · rw [half_quantum_sub ha (not_le.mp hn)]
    have : 0 ≤ a * f.u := mul_nonneg ha.le u_nonneg
    linarith

/-! ## Packing a significand -/

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

theorem rnat_natCast (rnd : RoundMode) (K : ℕ) : rnat rnd (K : ℚ) = K := by
  rw [rnat_eq, Nat.floor_natCast]
  have : cmpq ((K : ℚ) - (K : ℚ)) (1 / 2) = .lt := by unfold cmpq; rw [ite_eq_left (by norm_num)]
  rw [this]; cases rnd <;> rfl

/-- A value that is an integer multiple of its quantum rounds to itself (both modes). -/
theorem roundSig_exact (rnd : RoundMode) {a : ℚ} {Q : ℤ} {K : ℕ} (h : a * (2 : ℚ) ^ (-Q) = K) :
    (roundSig rnd a Q : ℚ) * (2 : ℚ) ^ Q = a := by
  rw [roundSig_eq, h, rnat_natCast, ← h, scaled_mul]

/-! ## Rounded patterns -/

/-- The pattern of a nonzero `q` is its sign plus its rounded magnitude. -/
theorem roundRat_of_ne {rnd : RoundMode} {sat zs : Bool} {q : ℚ} (hq : q ≠ 0) :
    f.roundRat rnd sat zs q = (if q < 0 then f.signBit else 0) + f.roundMag rnd sat |q| := by
  unfold roundRat; rw [ite_eq_right hq]
  congr 2
  split_ifs with h
  · exact (abs_of_neg h).symm
  · exact (abs_of_nonneg (not_lt.mp h)).symm

theorem roundRat_zero {rnd : RoundMode} {sat zs : Bool} :
    f.roundRat rnd sat zs 0 = f.setSign zs 0 := by
  unfold roundRat setSign mag; rw [ite_eq_left rfl]; simp

theorem roundRat_zero_spec (hf : f.Regular) {rnd : RoundMode} {sat zs : Bool} :
    f.roundRat rnd sat zs 0 < 2 ^ f.width ∧ f.isFinite (f.roundRat rnd sat zs 0) = true ∧
      f.toRat (f.roundRat rnd sat zs 0) = 0 := by
  rw [roundRat_zero]
  refine ⟨setSign_lt _ _, by rw [isFinite_setSign]; exact isFinite_zero hf, ?_⟩
  rw [toRat_setSign]; unfold mag; simp [magVal_zero]

private theorem abs_signed_sub {q : ℚ} (v : ℚ) :
    |(if q < 0 then (-1 : ℚ) else 1) * v - q| = abs (v - abs q) := by
  by_cases h : q < 0
  · rw [ite_eq_left h, abs_of_neg h, ← abs_neg]; congr 1; ring
  · rw [ite_eq_right h, abs_of_nonneg (not_lt.mp h)]; ring_nf

private theorem abs_signed {q v : ℚ} (hv : 0 ≤ v) : |(if q < 0 then (-1 : ℚ) else 1) * v| = v := by
  by_cases h : q < 0
  · rw [ite_eq_left h, abs_mul]; simp [abs_of_nonneg hv]
  · rw [ite_eq_right h, one_mul, abs_of_nonneg hv]

/-- **A value in range rounds to a finite pattern with the value `N · 2^Q`** of its rounded
significand, signed. -/
theorem roundRat_spec (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool) {q : ℚ} (hq0 : q ≠ 0)
    (hq : |q| ≤ f.maxVal) :
    f.roundRat rnd sat zs q < 2 ^ f.width ∧ f.isFinite (f.roundRat rnd sat zs q) = true ∧
      f.toRat (f.roundRat rnd sat zs q) =
        (if q < 0 then -1 else 1) * ((roundSig rnd |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|)) := by
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨hN, hQ, hsub⟩ := roundSig_bounds (f := f) rnd ha
  have hle := pack_le_maxMag hf hN hQ hsub (roundSig_le_maxVal hf rnd ha hq)
  have hlt := (hle.trans_lt (maxMag_lt hf))
  have hpat : f.roundRat rnd sat zs q = f.setSign (decide (q < 0))
      (f.pack (roundSig rnd |q| (f.qexp |q|)) (f.qexp |q|)) := by
    rw [roundRat_of_ne hq0, roundMag_of_le hf rnd sat ha hq, setSign_of_lt hlt]
    by_cases h : q < 0 <;> simp [h]
  rw [hpat]
  refine ⟨setSign_lt _ _, ?_, ?_⟩
  · rw [isFinite_setSign]; exact isFinite_of_mag_le hf (by rw [mag_of_lt hlt]; exact hle)
  · rw [toRat_setSign, mag_of_lt hlt, magVal_pack hN hQ hsub]
    by_cases h : q < 0 <;> simp [h]

/-- **One rounding to nearest** of a value in range: a finite pattern within `|q|·u + η`. -/
theorem roundRat_rn_err (hf : f.Regular) (sat zs : Bool) {q : ℚ} (hq : |q| ≤ f.maxVal) :
    f.roundRat .rn sat zs q < 2 ^ f.width ∧ f.isFinite (f.roundRat .rn sat zs q) = true ∧
      |f.toRat (f.roundRat .rn sat zs q) - q| ≤ |q| * f.u + f.eta := by
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨h1, h2, h3⟩ := roundRat_zero_spec (f := f) (rnd := .rn) (sat := sat) (zs := zs) hf
    refine ⟨h1, h2, ?_⟩
    rw [h3]; simp [eta_nonneg]
  obtain ⟨hlt, hfin, hval⟩ := roundRat_spec hf .rn sat zs hq0 hq
  refine ⟨hlt, hfin, ?_⟩
  have ha : 0 < |q| := abs_pos.mpr hq0
  generalize hA : |q| = a at ha hval ⊢
  have herr := roundSig_rn_err ha (f.qexp a)
  have hh := half_quantum_le (f := f) ha
  rw [hval, abs_signed_sub, hA]
  linarith

/-- **One rounding to nearest in the normal range** (`|q| ≥ 2^emin`): within `|q|·u`. -/
theorem roundRat_rn_err_normal (hf : f.Regular) (sat zs : Bool) {q : ℚ}
    (hlo : (2 : ℚ) ^ f.emin ≤ |q|) (hq : |q| ≤ f.maxVal) :
    |f.toRat (f.roundRat .rn sat zs q) - q| ≤ |q| * f.u := by
  have hq0 : q ≠ 0 := by
    intro h; rw [h, abs_zero] at hlo; exact absurd hlo (not_le.mpr (by positivity))
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨-, -, hval⟩ := roundRat_spec hf .rn sat zs hq0 hq
  rw [hval, abs_signed_sub]
  generalize hA : |q| = a at ha hlo ⊢
  have herr := roundSig_rn_err ha (f.qexp a)
  have hh := half_quantum_normal (f := f) ha (le_log_of_le ha hlo)
  linarith [mul_comm f.u a]

/-- **One rounding toward zero** of a value in range: a finite pattern within `2·(|q|·u + η)`,
not above `|q|` in magnitude. -/
theorem roundRat_rz_err (hf : f.Regular) (sat zs : Bool) {q : ℚ} (hq : |q| ≤ f.maxVal) :
    f.roundRat .rz sat zs q < 2 ^ f.width ∧ f.isFinite (f.roundRat .rz sat zs q) = true ∧
      |f.toRat (f.roundRat .rz sat zs q) - q| ≤ 2 * (|q| * f.u + f.eta) ∧
      |f.toRat (f.roundRat .rz sat zs q)| ≤ |q| := by
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨h1, h2, h3⟩ := roundRat_zero_spec (f := f) (rnd := .rz) (sat := sat) (zs := zs) hf
    refine ⟨h1, h2, ?_⟩
    rw [h3]; simp [eta_nonneg]
  obtain ⟨hlt, hfin, hval⟩ := roundRat_spec hf .rz sat zs hq0 hq
  refine ⟨hlt, hfin, ?_⟩
  have ha : 0 < |q| := abs_pos.mpr hq0
  rw [hval, abs_signed_sub, abs_signed (by positivity)]
  generalize hA : |q| = a at ha ⊢
  obtain ⟨h1, h2⟩ := roundSig_rz_err ha (f.qexp a)
  have hh := half_quantum_le (f := f) ha
  refine ⟨?_, h1⟩
  rw [abs_of_nonpos (by linarith)]
  linarith

/-- The overflow of a rounding to nearest without saturation is not finite (for a format with
infinities or a NaN). -/
theorem roundMag_rn_of_over (hf : f.Regular) (hs : f.specials ≠ .finite) {a : ℚ}
    (hover : ¬ (roundSig .rn a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a) ≤ f.maxVal) :
    f.roundMag .rn false a < f.signBit ∧ f.isFinite (f.roundMag .rn false a) = false := by
  unfold roundMag encode
  rw [pow2_eq, ite_eq_right hover]
  simp only [Bool.false_or, beq_iff_eq, reduceCtorEq, ite_false]
  cases hsp : f.specials with
  | ieee =>
    simp only
    refine ⟨?_, not_isFinite_infMag hsp⟩
    rw [signBit_eq]; unfold infMag
    exact Nat.mul_lt_mul_of_pos_right (by have := Nat.one_le_two_pow (n := f.e); omega) (by positivity)
  | nanOnly =>
    simp only
    refine ⟨?_, not_isFinite_nanMag hf (by rw [hsp]; decide)⟩
    unfold nanMag signBit; have := Nat.one_le_two_pow (n := f.e + f.m); omega
  | finite => exact absurd hsp hs

/-- **A finite result of a rounding to nearest is within one rounding's bound of the exact value**,
also when `|q|` exceeds `maxVal` (then it rounds down to `±maxVal`), relative to the result:
`|r − q| ≤ u·|r| + η` (a bound of at least half an ulp of `r`, and up to about a whole one). A
format whose overflow is finite (`.finite`) is excluded. -/
theorem roundRat_rn_err_of_finite (hf : f.Regular) (hs : f.specials ≠ .finite) (zs : Bool) {q : ℚ}
    (h : f.isFinite (f.roundRat .rn false zs q) = true) :
    |f.toRat (f.roundRat .rn false zs q) - q| ≤ |f.toRat (f.roundRat .rn false zs q)| * f.u + f.eta := by
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨-, -, h3⟩ := roundRat_zero_spec (f := f) (rnd := .rn) (sat := false) (zs := zs) hf
    rw [h3]; simp [eta_nonneg]
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨hN, hQ, hsub⟩ := roundSig_bounds (f := f) .rn ha
  have hno : (roundSig .rn |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|) ≤ f.maxVal := by
    by_contra hover
    obtain ⟨hlt, hnf⟩ := roundMag_rn_of_over hf hs hover
    rw [roundRat_of_ne hq0, show (if q < 0 then f.signBit else 0) = if decide (q < 0) then f.signBit else 0 by simp,
      ← setSign_of_lt hlt, isFinite_setSign, hnf] at h
    cases h
  have hmag : f.roundMag .rn false |q| = f.pack (roundSig .rn |q| (f.qexp |q|)) (f.qexp |q|) := by
    unfold roundMag encode; rw [pow2_eq, ite_eq_left hno]
  have hle := pack_le_maxMag hf hN hQ hsub hno
  have hlt := hle.trans_lt (maxMag_lt hf)
  have hval : f.toRat (f.roundRat .rn false zs q) =
      (if q < 0 then -1 else 1) * ((roundSig .rn |q| (f.qexp |q|) : ℚ) * 2 ^ (f.qexp |q|)) := by
    rw [roundRat_of_ne hq0, hmag,
      show (if q < 0 then f.signBit else 0) = if decide (q < 0) then f.signBit else 0 by simp,
      ← setSign_of_lt hlt, toRat_setSign, mag_of_lt hlt, magVal_pack hN hQ hsub]
    by_cases h : q < 0 <;> simp [h]
  rw [hval, abs_signed_sub, abs_signed (by positivity)]
  generalize hA : |q| = a at ha hN hsub hno ⊢
  have herr := roundSig_rn_err ha (f.qexp a)
  -- half the quantum is at most `u·N·2^Q + η`
  have hhalf : (2 : ℚ) ^ (f.qexp a) / 2 ≤
      (roundSig .rn a (f.qexp a) : ℚ) * 2 ^ (f.qexp a) * f.u + f.eta := by
    by_cases hn : f.emin ≤ Int.log 2 a
    · have hNm := roundSig_ge (f := f) .rn ha hn
      have hNm' : (2 : ℚ) ^ (f.m : ℤ) ≤ (roundSig .rn a (f.qexp a) : ℚ) := by
        rw [zpow_natCast]; exact_mod_cast hNm
      rw [half_quantum_eq, zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
      have hp : (0 : ℚ) ≤ 2 ^ (f.qexp a) * f.u := by rw [u_eq]; positivity
      have := eta_nonneg (f := f)
      have := mul_le_mul_of_nonneg_right hNm' hp
      nlinarith
    · rw [half_quantum_sub ha (not_le.mp hn)]
      have : 0 ≤ (roundSig .rn a (f.qexp a) : ℚ) * 2 ^ (f.qexp a) * f.u :=
        mul_nonneg (by positivity) u_nonneg
      linarith
  linarith

/-- **Saturation beyond the range.** A magnitude above `maxVal`, rounded with saturation (either
mode), gives the largest finite magnitude: its rounded value is at least `maxVal`, so it is
`maxVal` itself or an overflow. -/
theorem roundMag_sat_of_gt (hf : f.Regular) (rnd : RoundMode) {a : ℚ} (ha : f.maxVal < a) :
    f.roundMag rnd true a = f.maxMag := by
  have ha0 : 0 < a := (maxVal_pos hf).trans ha
  obtain ⟨hN, hQ, hsub⟩ := roundSig_bounds (f := f) rnd ha0
  have hE : f.emin ≤ (f.topExp : ℤ) - f.bias := by rw [emin_eq]; have := topExp_pos hf; omega
  have hmlo : (2 : ℚ) ^ ((f.topExp : ℤ) - f.bias) ≤ f.maxVal := by
    rw [maxVal_eq]
    have hs : ((2 ^ f.m : ℕ) : ℚ) ≤ (f.topSig : ℚ) := by exact_mod_cast topSig_ge hf
    have e : (2 : ℚ) ^ ((f.topExp : ℤ) - f.bias) =
        ((2 ^ f.m : ℕ) : ℚ) * (2 : ℚ) ^ ((f.topExp : ℤ) - f.bias - f.m) := by
      push_cast; rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]; congr 1; ring
    rw [e]; exact mul_le_mul_of_nonneg_right hs (by positivity)
  have hlog : (f.topExp : ℤ) - f.bias ≤ Int.log 2 a := le_log_of_le ha0 (hmlo.trans ha.le)
  have hn : f.emin ≤ Int.log 2 a := hE.trans hlog
  have hQe : f.qexp a = Int.log 2 a - f.m := by rw [qexp_eq ha0, max_eq_left hn]
  have hp : (0 : ℚ) < (2 : ℚ) ^ (f.qexp a) := by positivity
  -- the rounded value is at least `maxVal`
  have hge : f.maxVal ≤ (roundSig rnd a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a) := by
    rcases lt_or_eq_of_le hlog with hlt | heq
    · -- a binade above `maxVal`'s: the significand is at least `2^m`
      have hNm : (2 : ℚ) ^ (f.m : ℤ) ≤ (roundSig rnd a (f.qexp a) : ℚ) := by
        rw [zpow_natCast]; exact_mod_cast roundSig_ge (f := f) rnd ha0 hn
      have h1 : (2 : ℚ) ^ (Int.log 2 a) ≤ (roundSig rnd a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a) := by
        have e : (2 : ℚ) ^ (Int.log 2 a) = (2 : ℚ) ^ (f.m : ℤ) * (2 : ℚ) ^ (f.qexp a) := by
          rw [← zpow_add₀ (by norm_num), hQe]; congr 1; ring
        rw [e]; exact mul_le_mul_of_nonneg_right hNm hp.le
      have h2 : (2 : ℚ) ^ ((f.topExp : ℤ) + 1 - f.bias) ≤ (2 : ℚ) ^ (Int.log 2 a) :=
        zpow_le_zpow_right₀ (by norm_num) (by omega)
      linarith [maxVal_lt hf]
    · -- `maxVal`'s binade: the quantum is `maxVal`'s, and `a / 2^Q > topSig`
      have hQT : f.qexp a = (f.topExp : ℤ) - f.bias - f.m := by rw [hQe, ← heq]
      have hK : (f.topSig : ℚ) ≤ a * (2 : ℚ) ^ (-f.qexp a) := by
        have h := mul_lt_mul_of_pos_right ha (show (0 : ℚ) < (2 : ℚ) ^ (-f.qexp a) by positivity)
        rw [maxVal_eq, ← hQT, mul_assoc, ← zpow_add₀ (by norm_num), add_neg_cancel, zpow_zero,
          mul_one] at h
        exact h.le
      have hN' : f.topSig ≤ roundSig rnd a (f.qexp a) := by rw [roundSig_eq]; exact le_rnat rnd hK
      rw [maxVal_eq, ← hQT]
      exact mul_le_mul_of_nonneg_right (by exact_mod_cast hN') hp.le
  unfold roundMag encode
  rw [pow2_eq]
  split_ifs with hle hsat
  · apply magVal_inj
    rw [magVal_pack hN hQ hsub, le_antisymm hle hge, magVal_maxMag hf]
  · rfl
  · simp at hsat


/-! ## Square roots -/

/-- The significand `sqrtMag` computes, `rint` of the integer square root, is within `1/2` of
`√t`. -/
theorem sqrt_rint_err {t : ℚ} (ht : 0 ≤ t) :
    |((rint .rn (Nat.sqrt (Rat.floor t).toNat)
        (cmpq t (((Nat.sqrt (Rat.floor t).toNat : ℕ) + 1 / 2) * ((Nat.sqrt (Rat.floor t).toNat : ℕ) + 1 / 2)))
        : ℕ) : ℝ) - Real.sqrt t| ≤ 1 / 2 := by
  rw [floor_toNat]
  set k := Nat.sqrt ⌊t⌋₊ with hk
  have hk1 : ((k : ℚ)) ^ 2 ≤ t := by
    have h1 : k ^ 2 ≤ ⌊t⌋₊ := Nat.sqrt_le' _
    have h2 : (⌊t⌋₊ : ℚ) ≤ t := Nat.floor_le ht
    calc ((k : ℚ)) ^ 2 = ((k ^ 2 : ℕ) : ℚ) := by push_cast; ring
      _ ≤ (⌊t⌋₊ : ℚ) := by exact_mod_cast h1
      _ ≤ t := h2
  have hk2 : t < ((k : ℚ) + 1) ^ 2 := by
    have h1 : ⌊t⌋₊ < (k + 1) ^ 2 := Nat.lt_succ_sqrt' _
    have h2 : t < (⌊t⌋₊ : ℚ) + 1 := Nat.lt_floor_add_one t
    have h3 : (⌊t⌋₊ : ℚ) + 1 ≤ ((k + 1) ^ 2 : ℕ) := by exact_mod_cast h1
    calc t < (⌊t⌋₊ : ℚ) + 1 := h2
      _ ≤ ((k + 1) ^ 2 : ℕ) := h3
      _ = ((k : ℚ) + 1) ^ 2 := by push_cast; ring
  have htR : (0 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have hk1R : (k : ℝ) ≤ Real.sqrt t := by
    apply Real.le_sqrt_of_sq_le
    have : ((k : ℚ) ^ 2 : ℚ) ≤ t := hk1
    exact_mod_cast this
  have hk2R : Real.sqrt t < (k : ℝ) + 1 := by
    rw [Real.sqrt_lt' (by positivity)]
    have : t < ((k : ℚ) + 1) ^ 2 := hk2
    exact_mod_cast this
  have hpos : (0 : ℝ) < (k : ℝ) + 1 / 2 := by positivity
  unfold cmpq
  rw [abs_le]
  split_ifs with h1 h2
  · -- below the midpoint: `k`
    have : Real.sqrt t < (k : ℝ) + 1 / 2 := by
      rw [Real.sqrt_lt' hpos]
      have h1' := (Rat.cast_lt (K := ℝ)).mpr h1
      push_cast at h1'
      nlinarith
    simp only [rint]; constructor <;> linarith
  · -- the midpoint: `k` or `k + 1`, half away
    have : Real.sqrt t = (k : ℝ) + 1 / 2 := by
      have e : (t : ℝ) = ((k : ℝ) + 1 / 2) ^ 2 := by
        have : (t : ℝ) = (((k : ℚ) + 1 / 2) * ((k : ℚ) + 1 / 2) : ℚ) := by exact_mod_cast h2
        rw [this]; push_cast; ring
      rw [e, Real.sqrt_sq hpos.le]
    simp only [rint]
    split_ifs <;> push_cast <;> constructor <;> linarith
  · -- above the midpoint: `k + 1`
    have hgt : ((k : ℚ) + 1 / 2) * ((k : ℚ) + 1 / 2) < t := lt_of_le_of_ne (not_lt.mp h1) (Ne.symm h2)
    have : (k : ℝ) + 1 / 2 < Real.sqrt t := by
      rw [Real.lt_sqrt hpos.le]
      have h1' := (Rat.cast_lt (K := ℝ)).mpr hgt
      push_cast at h1'
      nlinarith
    simp only [rint]; push_cast; constructor <;> linarith

/-- The largest value is at least `1` (at least `2^(topExp − bias)`). -/
theorem one_le_maxVal (hf : f.Regular) : 1 ≤ f.maxVal := by
  have hE := topExp_pos hf
  have h := magVal_ge_of_exp (f := f) (M := f.maxMag) (by rw [maxMag_div hf]; exact hE)
  rw [magVal_maxMag hf, maxMag_div hf] at h
  have hb : 0 ≤ ((f.topExp : ℕ) : ℤ) - f.bias := by
    have h4 : 4 ≤ 2 ^ f.e := two_pow_e_ge hf
    have h2 : (2 : ℤ) ^ f.e = 2 * 2 ^ (f.e - 1) := by
      rw [← pow_succ']; congr 1; have := hf.1; omega
    have h1 : (1 : ℤ) ≤ 2 ^ (f.e - 1) := one_le_pow₀ (by norm_num)
    have ht : 2 ^ f.e - 2 ≤ f.topExp := by unfold topExp; split <;> omega
    have ht' : ((2 ^ f.e - 2 : ℕ) : ℤ) = (2 : ℤ) ^ f.e - 2 := by
      push_cast [show 2 ≤ 2 ^ f.e by omega]; ring
    have : ((2 ^ f.e - 2 : ℕ) : ℤ) ≤ (f.topExp : ℤ) := by exact_mod_cast ht
    unfold bias
    linarith
  calc (1 : ℚ) = 2 ^ (0 : ℤ) := by simp
    _ ≤ 2 ^ (((f.topExp : ℕ) : ℤ) - f.bias) := zpow_le_zpow_right₀ (by norm_num) hb
    _ ≤ f.maxVal := h

/-- **`sqrtMag` rounds `√a` to nearest**: for `0 < a ≤ maxVal` the magnitude is finite (at most
`maxMag`) and its value is within `√a·u + η` of `√a`, the bound of one rounding to nearest (at least
half an ulp, and up to about a whole one: not the half-ulp bound itself). -/
theorem sqrtMag_rn_spec (hf : f.Regular) {a : ℚ} (ha : 0 < a) (hmax : a ≤ f.maxVal) :
    f.sqrtMag .rn a ≤ f.maxMag ∧
      |(f.magVal (f.sqrtMag .rn a) : ℝ) - Real.sqrt a| ≤ Real.sqrt a * f.u + f.eta := by
  obtain ⟨hL1, hL2⟩ := qlog2_spec ha
  set L := qlog2 a with hL
  set j : ℤ := L / 2 with hj
  have hj1 : 2 * j ≤ L := by omega
  have hj2 : L ≤ 2 * j + 1 := by omega
  set Q : ℤ := max j f.emin - f.m with hQ
  set t : ℚ := a * pow2 (-2 * Q) with ht
  have ht' : t = a * (2 : ℚ) ^ (-2 * Q) := by rw [ht, pow2_eq]
  have ht0 : 0 ≤ t := by rw [ht']; positivity
  set k := Nat.sqrt (Rat.floor t).toNat with hk
  set N := rint .rn k (cmpq t (((k : ℚ) + 1 / 2) * ((k : ℚ) + 1 / 2))) with hN
  have hsm : f.sqrtMag .rn a = f.encode .rn false N Q := rfl
  have herr : |(N : ℝ) - Real.sqrt t| ≤ 1 / 2 := sqrt_rint_err ht0
  -- the real square roots
  set σ := Real.sqrt a with hσ
  set τ := Real.sqrt t with hτ
  have h2 : (0 : ℝ) < 2 := by norm_num
  have haR : (0 : ℝ) < (a : ℝ) := by exact_mod_cast ha
  have hτσ : τ = σ * (2 : ℝ) ^ (-Q) := by
    have e1 : (t : ℝ) = (a : ℝ) * ((2 : ℝ) ^ (-Q)) ^ 2 := by
      rw [ht']; push_cast
      rw [← zpow_natCast, ← zpow_mul]; ring_nf
    rw [hτ, e1, Real.sqrt_mul haR.le, Real.sqrt_sq (by positivity)]
  have hστ : σ = τ * (2 : ℝ) ^ Q := by
    rw [hτσ, mul_assoc, ← zpow_add₀ h2.ne']; simp
  have hσlo : (2 : ℝ) ^ j ≤ σ := by
    apply Real.le_sqrt_of_sq_le
    have : (2 : ℚ) ^ (2 * j) ≤ a := (zpow_le_zpow_right₀ (by norm_num) hj1).trans hL1
    have : ((2 : ℚ) ^ (2 * j) : ℝ) ≤ (a : ℝ) := by exact_mod_cast this
    push_cast at this
    rw [← zpow_natCast, ← zpow_mul, show j * ((2 : ℕ) : ℤ) = 2 * j by push_cast; ring]; exact this
  have hσhi : σ < (2 : ℝ) ^ (j + 1) := by
    rw [hσ, Real.sqrt_lt' (by positivity)]
    have : a < (2 : ℚ) ^ (2 * j + 2) := hL2.trans_le (zpow_le_zpow_right₀ (by norm_num) (by omega))
    have : (a : ℝ) < ((2 : ℚ) ^ (2 * j + 2) : ℝ) := by exact_mod_cast this
    push_cast at this
    rw [← zpow_natCast, ← zpow_mul]
    convert this using 2; push_cast; ring
  have hmax1 := one_le_maxVal hf
  have hσmax : σ ≤ (f.maxVal : ℝ) := by
    have hm1 : (1 : ℝ) ≤ (f.maxVal : ℝ) := by exact_mod_cast hmax1
    have hle : (a : ℝ) ≤ (f.maxVal : ℝ) ^ 2 := by
      have : (a : ℝ) ≤ (f.maxVal : ℝ) := by exact_mod_cast hmax
      nlinarith
    calc σ ≤ Real.sqrt ((f.maxVal : ℝ) ^ 2) := Real.sqrt_le_sqrt hle
      _ = (f.maxVal : ℝ) := Real.sqrt_sq (by linarith)
  have hjQ : j ≤ Q + f.m := by rw [hQ]; omega
  have hτ0 : 0 ≤ τ := Real.sqrt_nonneg _
  -- `N ≤ 2^(m+1)`
  have hτhi : τ < (2 : ℝ) ^ ((f.m : ℤ) + 1) := by
    rw [hτσ]
    calc σ * (2 : ℝ) ^ (-Q) < (2 : ℝ) ^ (j + 1) * (2 : ℝ) ^ (-Q) :=
          mul_lt_mul_of_pos_right hσhi (by positivity)
      _ = (2 : ℝ) ^ (j + 1 - Q) := by rw [← zpow_add₀ h2.ne']; ring_nf
      _ ≤ (2 : ℝ) ^ ((f.m : ℤ) + 1) := zpow_le_zpow_right₀ (by norm_num) (by omega)
  have hN1 : N ≤ 2 ^ (f.m + 1) := by
    have h1 : (N : ℝ) < ((2 ^ (f.m + 1) : ℕ) : ℝ) + 1 := by
      have : (2 : ℝ) ^ ((f.m : ℤ) + 1) = ((2 ^ (f.m + 1) : ℕ) : ℝ) := by
        rw [show ((f.m : ℤ) + 1) = ((f.m + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]; push_cast; rfl
      rw [abs_le] at herr; linarith
    have : N < 2 ^ (f.m + 1) + 1 := by exact_mod_cast h1
    omega
  have hQ0 : f.emin - f.m ≤ Q := by rw [hQ]; have := le_max_right j f.emin; linarith
  have hsub : N < 2 ^ f.m → Q = f.emin - f.m := by
    intro hlt
    by_contra hne
    have hjm : f.emin ≤ j := by
      by_contra h'; exact hne (by rw [hQ, max_eq_right (not_le.mp h').le])
    have hQj : Q = j - f.m := by rw [hQ, max_eq_left hjm]
    have hτlo : (2 : ℝ) ^ (f.m : ℤ) ≤ τ := by
      rw [hτσ]
      calc (2 : ℝ) ^ (f.m : ℤ) = (2 : ℝ) ^ j * (2 : ℝ) ^ (-Q) := by
            rw [← zpow_add₀ h2.ne', hQj]; ring_nf
        _ ≤ σ * (2 : ℝ) ^ (-Q) := mul_le_mul_of_nonneg_right hσlo (by positivity)
    have h1 : ((2 ^ f.m : ℕ) : ℝ) - 1 < (N : ℝ) := by
      have : (2 : ℝ) ^ (f.m : ℤ) = ((2 ^ f.m : ℕ) : ℝ) := by rw [zpow_natCast]; push_cast; rfl
      rw [abs_le] at herr; linarith
    have : 2 ^ f.m < N + 1 := by
      have : ((2 ^ f.m : ℕ) : ℝ) < (N : ℝ) + 1 := by linarith
      exact_mod_cast this
    omega
  -- no overflow: `N · 2^Q ≤ maxVal`
  set T : ℤ := (f.topExp : ℤ) - f.bias - f.m with hT
  have hQT : Q ≤ T := by
    have hjT : j < (f.topExp : ℤ) + 1 - f.bias := by
      have hl : (2 : ℝ) ^ j < (2 : ℝ) ^ ((f.topExp : ℤ) + 1 - f.bias) := by
        have := maxVal_lt hf
        have : ((f.maxVal : ℚ) : ℝ) < (((2 : ℚ) ^ ((f.topExp : ℤ) + 1 - f.bias) : ℚ) : ℝ) := by
          exact_mod_cast this
        push_cast at this
        linarith
      exact (zpow_lt_zpow_iff_right₀ (by norm_num)).mp hl
    have hE : f.emin ≤ (f.topExp : ℤ) - f.bias := by
      rw [emin_eq]; have := topExp_pos hf; omega
    rw [hQ, hT]; omega
  set K : ℕ := f.topSig * 2 ^ (T - Q).toNat with hK
  have hKQ : (K : ℚ) * (2 : ℚ) ^ Q = f.maxVal := by
    rw [hK, maxVal_eq]; push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), mul_assoc, ← zpow_add₀ (by norm_num)]
    congr 2; ring
  have hτK : τ ≤ (K : ℝ) := by
    have hKQR : (K : ℝ) * (2 : ℝ) ^ Q = (f.maxVal : ℝ) := by
      have := congrArg (fun x : ℚ => (x : ℝ)) hKQ
      push_cast at this; exact this
    rw [hτσ]
    calc σ * (2 : ℝ) ^ (-Q) ≤ (f.maxVal : ℝ) * (2 : ℝ) ^ (-Q) := mul_le_mul_of_nonneg_right hσmax (by positivity)
      _ = (K : ℝ) := by rw [← hKQR, mul_assoc, ← zpow_add₀ h2.ne']; simp
  have hNK : N ≤ K := by
    have : (N : ℝ) < (K : ℝ) + 1 := by rw [abs_le] at herr; linarith
    have : N < K + 1 := by exact_mod_cast this
    omega
  have hNmax : (N : ℚ) * (2 : ℚ) ^ Q ≤ f.maxVal := by
    rw [← hKQ]; exact mul_le_mul_of_nonneg_right (by exact_mod_cast hNK) (by positivity)
  have henc : f.encode .rn false N Q = f.pack N Q := by
    unfold encode; rw [pow2_eq, ite_eq_left hNmax]
  refine ⟨by rw [hsm, henc]; exact pack_le_maxMag hf hN1 hQ0 hsub hNmax, ?_⟩
  rw [hsm, henc, magVal_pack hN1 hQ0 hsub]
  -- the error: `2^Q · |N − τ| ≤ 2^Q / 2 ≤ σ·u + η`
  have hval : (((N : ℚ) * (2 : ℚ) ^ Q : ℚ) : ℝ) - σ = ((N : ℝ) - τ) * (2 : ℝ) ^ Q := by
    rw [hστ]; push_cast; ring
  have hpQ : (0 : ℝ) < (2 : ℝ) ^ Q := by positivity
  rw [hval, abs_mul, abs_of_pos hpQ]
  have hhalf : (2 : ℝ) ^ Q / 2 ≤ σ * (f.u : ℝ) + (f.eta : ℝ) := by
    have hu0 : (0 : ℝ) ≤ (f.u : ℝ) := by exact_mod_cast u_nonneg
    have he0 : (0 : ℝ) ≤ (f.eta : ℝ) := by exact_mod_cast eta_nonneg
    by_cases hjm : f.emin ≤ j
    · have hQj : Q = j - f.m := by rw [hQ, max_eq_left hjm]
      have hq := half_quantum_eq (f := f) Q
      have hq' : (2 : ℝ) ^ Q / 2 = (f.u : ℝ) * (2 : ℝ) ^ ((f.m : ℤ) + Q) := by
        have := congrArg (fun x : ℚ => (x : ℝ)) hq; push_cast at this; exact this
      rw [hq', show (f.m : ℤ) + Q = j by rw [hQj]; ring]
      nlinarith
    · have hQe : Q = f.emin - f.m := by rw [hQ, max_eq_right (not_le.mp hjm).le]
      have he : (f.eta : ℝ) = (2 : ℝ) ^ Q / 2 := by
        rw [eta_eq]; push_cast
        rw [hQe, eq_div_iff (by norm_num), ← zpow_add_one₀ (by norm_num)]; ring_nf
      have : 0 ≤ σ * (f.u : ℝ) := mul_nonneg (Real.sqrt_nonneg _) hu0
      linarith
  calc |(N : ℝ) - τ| * (2 : ℝ) ^ Q ≤ 1 / 2 * (2 : ℝ) ^ Q := mul_le_mul_of_nonneg_right herr hpQ.le
    _ = (2 : ℝ) ^ Q / 2 := by ring
    _ ≤ σ * (f.u : ℝ) + (f.eta : ℝ) := hhalf

end Format

end TVA.Math.Float
