import MatmulE5m2.Basic

/-!
# `matmul_e5m2`: what the tolerance rejects

`Post` holds for every input, so it rejects a wrong kernel as soon as one input exposes it. On
inputs with one or two nonzero products the error bound `tol` stays small at every covered `K`:

* `tol_spike`: one product `X = x·y`, in the loop's last step: `tol ≤ ρ·|X| + 200·α`
  (`ρ = relDot8 64 = 257·2^-14`, about `1.57%`; `200·α < 10^-35`).
* `zero_rejected`: at every covered `K ≥ 1`, some finite input leaves room in `float16` and puts
  `0` outside the bound of `OutSpec`'s `float16` clause: a kernel that stores `0` breaks `Post`.
* `drop_rejected`: for every step `s` of the loop (`s < cdiv(K, 64)`), some finite input puts
  `S − X_s` outside that bound, `X_s` the exact sum of step `s`'s products: a kernel that skips
  step `s` (and is otherwise exact) breaks `Post`.

The inputs are `float8e5` rows and columns of `1`, `-1` and `0`: one product `1` at index
`K − 1` (the last step), or `1` at `64·s` and `-1` at `64·s + 64` (steps `s` and `s + 1`).
-/

namespace Kernels.MatmulE5m2

open TVA

/-! ## Sparse product sequences -/

/-- The product sequence with `x` at `e` and `0` elsewhere. -/
def spikeP (e : Nat) (x : ℚ) : Nat → ℚ := fun k => if k = e then x else 0

/-- The product sequence with `1` at `e1` and `-1` at `e2`. -/
def pairP (e1 e2 : Nat) : Nat → ℚ := fun k => spikeP e1 1 k + spikeP e2 (-1) k

/-- The exact sum of step `t`'s products, `p e` for `64·t ≤ e < 64·t + 64`: what a kernel that
skips step `t` leaves out. -/
def stepSum (p : Nat → ℚ) (t : Nat) : ℚ := ∑ k : Fin 64, p (64 * t + k)

theorem partSum_spikeP (e : Nat) (x : ℚ) (t : Nat) :
    partSum (spikeP e x) t = if e < 64 * t then x else 0 := by
  unfold partSum spikeP
  rw [Finset.sum_ite_eq' (Finset.range (64 * t)) e (fun _ => x)]
  simp only [Finset.mem_range]

theorem stepSum_spikeP (e : Nat) (x : ℚ) (t : Nat) :
    stepSum (spikeP e x) t = if 64 * t ≤ e ∧ e < 64 * t + 64 then x else 0 := by
  unfold stepSum
  rw [Fin.sum_univ_eq_sum_range (fun k => spikeP e x (64 * t + k)) 64]
  unfold spikeP
  by_cases h : 64 * t ≤ e ∧ e < 64 * t + 64
  · rw [ite_eq_left h, Finset.sum_eq_single (e - 64 * t)]
    · rw [ite_eq_left (by omega)]
    · intro k _ hk
      rw [ite_eq_right (by omega)]
    · intro hk
      exfalso
      apply hk
      rw [Finset.mem_range]
      omega
  · rw [ite_eq_right h]
    apply Finset.sum_eq_zero
    intro k hk
    rw [Finset.mem_range] at hk
    rw [ite_eq_right (by omega)]

theorem abs_spikeP (e : Nat) (x : ℚ) (k : Nat) : |spikeP e x k| = spikeP e |x| k := by
  unfold spikeP
  split_ifs <;> simp

theorem stepMag_spikeP (e : Nat) (x : ℚ) (t : Nat) :
    stepMag (spikeP e x) t = if 64 * t ≤ e ∧ e < 64 * t + 64 then |x| else 0 := by
  unfold stepMag
  simp only [abs_spikeP]
  exact stepSum_spikeP e |x| t

theorem partSum_pairP (e1 e2 t : Nat) :
    partSum (pairP e1 e2) t = partSum (spikeP e1 1) t + partSum (spikeP e2 (-1)) t := by
  unfold partSum pairP
  exact Finset.sum_add_distrib

theorem stepSum_pairP (e1 e2 t : Nat) :
    stepSum (pairP e1 e2) t = stepSum (spikeP e1 1) t + stepSum (spikeP e2 (-1)) t := by
  unfold stepSum pairP
  exact Finset.sum_add_distrib

theorem stepMag_pairP {e1 e2 : Nat} (hne : e1 ≠ e2) (t : Nat) :
    stepMag (pairP e1 e2) t = stepMag (spikeP e1 1) t + stepMag (spikeP e2 (-1)) t := by
  have h : ∀ k, |pairP e1 e2 k| = |spikeP e1 1 k| + |spikeP e2 (-1) k| := fun k => by
    unfold pairP spikeP
    by_cases h1 : k = e1
    · have h2 : ¬ k = e2 := fun h2 => hne (h1.symm.trans h2)
      simp [h1, hne]
    · simp [h1]
  unfold stepMag
  simp only [h]
  exact Finset.sum_add_distrib

/-! ## The error bound on steps without products -/

/-- Steps whose partial sum and products are all `0` only grow the bound by `(1 + ρ)` and `α`. -/
theorem errAt_quiet (p : Nat → ℚ) (t0 : Nat) : ∀ n : Nat,
    (∀ t, t0 ≤ t → t < t0 + n → |partSum p t| + stepMag p t = 0) →
    errAt p (t0 + n) ≤ (1 + Rules.Num.relDot8 64) ^ n * (errAt p t0 + n * Rules.Num.absDot8 64)
  | 0, _ => by simp
  | n + 1, h => by
    have ih := errAt_quiet p t0 n (fun t h1 h2 => h t h1 (by omega))
    have hz := h (t0 + n) (by omega) (by omega)
    have hρ : (0 : ℚ) ≤ Rules.Num.relDot8 64 := by rw [relDot8_64]; norm_num
    have hα : (0 : ℚ) ≤ Rules.Num.absDot8 64 := by rw [absDot8_64]; norm_num
    have hE := errAt_nonneg p t0
    have hP1 : (1 : ℚ) ≤ (1 + Rules.Num.relDot8 64) ^ (n + 1) := one_le_pow₀ (by linarith)
    have e : errAt p (t0 + (n + 1)) = (1 + Rules.Num.relDot8 64) * errAt p (t0 + n) +
        Rules.Num.relDot8 64 * (|partSum p (t0 + n)| + stepMag p (t0 + n)) + Rules.Num.absDot8 64 := rfl
    rw [e, hz, mul_zero, add_zero]
    have h1 := mul_le_mul_of_nonneg_left ih (show (0 : ℚ) ≤ 1 + Rules.Num.relDot8 64 by linarith)
    have h2 : Rules.Num.absDot8 64 ≤ (1 + Rules.Num.relDot8 64) ^ (n + 1) * Rules.Num.absDot8 64 := by
      nlinarith
    have h3 : (1 + Rules.Num.relDot8 64) * ((1 + Rules.Num.relDot8 64) ^ n *
        (errAt p t0 + n * Rules.Num.absDot8 64)) = (1 + Rules.Num.relDot8 64) ^ (n + 1) *
        (errAt p t0 + n * Rules.Num.absDot8 64) := by ring
    have h4 : (1 + Rules.Num.relDot8 64) ^ (n + 1) * (errAt p t0 + ((n + 1 : Nat) : ℚ) * Rules.Num.absDot8 64) =
        (1 + Rules.Num.relDot8 64) ^ (n + 1) * (errAt p t0 + n * Rules.Num.absDot8 64) +
          (1 + Rules.Num.relDot8 64) ^ (n + 1) * Rules.Num.absDot8 64 := by push_cast; ring
    rw [h4]
    linarith

/-! ## The inputs -/

/-- `float8e5` `1.0` (`0x3c`). -/
def one8 : F8E5 := FloatV.ofPat 60

/-- `float8e5` `-1.0` (`0xbc`). -/
def mone8 : F8E5 := FloatV.ofPat 188

/-- `float8e5` `+0` (`0x00`). -/
def zero8 : F8E5 := FloatV.ofPat 0

theorem one8_val : one8.toRat = 1 := by
  unfold one8
  rw [FloatV.toRat_ofPat (by decide)]
  norm_num [Math.Float.Format.toRat, Math.Float.Format.magVal, Math.Float.Format.mag, Math.Float.Format.sign,
    Math.Float.Format.signBit, Math.Float.Format.e5m2, Math.Float.Format.bias, Math.Float.Format.emin,
    Math.Float.pow2_eq]

theorem mone8_val : mone8.toRat = -1 := by
  unfold mone8
  rw [FloatV.toRat_ofPat (by decide)]
  norm_num [Math.Float.Format.toRat, Math.Float.Format.magVal, Math.Float.Format.mag, Math.Float.Format.sign,
    Math.Float.Format.signBit, Math.Float.Format.e5m2, Math.Float.Format.bias, Math.Float.Format.emin,
    Math.Float.pow2_eq]

theorem zero8_val : zero8.toRat = 0 := by
  unfold zero8
  rw [FloatV.toRat_ofPat (by decide)]
  exact Math.Float.Format.toRat_of_isZero (by decide)

theorem one8_fin : one8.isFinite = true := by decide

theorem mone8_fin : mone8.isFinite = true := by decide

theorem zero8_fin : zero8.isFinite = true := by decide

/-- A row (or column) with `x` at index `e` and `+0` elsewhere. -/
def spike8 (e : Nat) (x : F8E5) : Nat → F8E5 := fun k => if k = e then x else zero8

/-- A row with `1` at `e1`, `-1` at `e2` and `+0` elsewhere. -/
def pair8 (e1 e2 : Nat) : Nat → F8E5 := fun k => if k = e1 then one8 else if k = e2 then mone8 else zero8

/-- A column of `1`s. -/
def ones8 : Nat → F8E5 := fun _ => one8

theorem spike8_fin {e : Nat} {x : F8E5} (hx : x.isFinite = true) (k : Nat) : (spike8 e x k).isFinite = true := by
  unfold spike8
  split_ifs
  · exact hx
  · exact zero8_fin

theorem pair8_fin (e1 e2 k : Nat) : (pair8 e1 e2 k).isFinite = true := by
  unfold pair8
  split_ifs
  · exact one8_fin
  · exact mone8_fin
  · exact zero8_fin

theorem prodV_spike8 {e K : Nat} (he : e < K) (x y : F8E5) :
    prodV (spike8 e x) (spike8 e y) K = spikeP e (x.toRat * y.toRat) := by
  funext k
  unfold prodV spike8 spikeP
  by_cases hk : k = e
  · subst hk
    simp [he]
  · simp [hk, zero8_val]

theorem prodV_pair8 {e1 e2 K : Nat} (h1 : e1 < K) (h2 : e2 < K) (hne : e1 ≠ e2) :
    prodV (pair8 e1 e2) ones8 K = pairP e1 e2 := by
  funext k
  unfold prodV pair8 ones8 pairP spikeP
  by_cases hk1 : k = e1
  · subst hk1
    simp [h1, hne, one8_val]
  · by_cases hk2 : k = e2
    · subst hk2
      simp [h2, hk1, one8_val, mone8_val]
    · simp [hk1, hk2, zero8_val]

/-! ## One product in the last step -/

theorem steps_last {K : Nat} (hK1 : 1 ≤ K) {n : Nat} (hn : steps K = n + 1) :
    64 * n ≤ K - 1 ∧ K - 1 < 64 * n + 64 := by
  unfold steps at hn; omega

/-- **One product `X = x·y` at index `K − 1`** (the loop's last step): the bound is at most
`ρ·|X| + 200·α`, about `1.57%` of `|X|`, at every covered `K`. -/
theorem tol_spike {K : Nat} (hK1 : 1 ≤ K) (hK : K ≤ 2 ^ 12) (x y : F8E5) :
    tol (spike8 (K - 1) x) (spike8 (K - 1) y) K ≤
      Rules.Num.relDot8 64 * |x.toRat * y.toRat| + 200 * Rules.Num.absDot8 64 := by
  obtain ⟨n, hn⟩ : ∃ n, steps K = n + 1 := ⟨steps K - 1, by unfold steps; omega⟩
  have hl := steps_last hK1 hn
  have hn64 : n ≤ 64 := by unfold steps at hn; omega
  unfold tol
  rw [prodV_spike8 (by omega), hn]
  set p := spikeP (K - 1) (x.toRat * y.toRat)
  have e : errAt p (n + 1) = (1 + Rules.Num.relDot8 64) * errAt p n +
      Rules.Num.relDot8 64 * (|partSum p n| + stepMag p n) + Rules.Num.absDot8 64 := rfl
  rw [e, partSum_spikeP, stepMag_spikeP, ite_eq_right (by omega), ite_eq_left (by omega), abs_zero, zero_add]
  have hq := errAt_quiet p 0 n (fun t _ ht => by
    rw [partSum_spikeP, stepMag_spikeP, ite_eq_right (by omega), ite_eq_right (by omega)]
    simp)
  have h0 : errAt p 0 = 0 := rfl
  rw [zero_add, h0, zero_add] at hq
  have hρ : (0 : ℚ) ≤ Rules.Num.relDot8 64 := by rw [relDot8_64]; norm_num
  have hα : (0 : ℚ) ≤ Rules.Num.absDot8 64 := by rw [absDot8_64]; norm_num
  have hP := pow_rho_le hn64
  have hn' : (n : ℚ) ≤ 64 := by exact_mod_cast hn64
  have hE : errAt p n ≤ 3 * (64 * Rules.Num.absDot8 64) :=
    hq.trans (mul_le_mul hP (mul_le_mul_of_nonneg_right hn' hα) (by positivity) (by norm_num))
  rw [absDot8_64] at hE
  rw [relDot8_64, absDot8_64]
  linarith

theorem dotSum_spike {K : Nat} (hK1 : 1 ≤ K) (x y : F8E5) :
    dotSum (spike8 (K - 1) x) (spike8 (K - 1) y) K = x.toRat * y.toRat := by
  have h := (partS_end (spike8 (K - 1) x) (spike8 (K - 1) y) (K := K) (T := steps K)
    (by unfold steps; omega)).1
  rw [← h]
  unfold partS
  rw [prodV_spike8 (by omega), partSum_spikeP, ite_eq_left (by unfold steps; omega)]

/-! ## What `Post` rejects at every covered size -/

/-- **A zero result breaks `Post` at every covered `K ≥ 1`**: on the input with one product `1`
at index `K − 1`, there is room in `float16`, and `0` is farther from `S = 1` than `OutSpec`'s
`float16` bound `tol + (|S| + tol)·2^-11 + 2^-25` allows. So a kernel that stores `0` (or skips the
loop) meets `Post` on no instance with `K ≥ 1` whose memory holds this row and column. -/
theorem zero_rejected {K : Nat} (hK1 : 1 ≤ K) (hK : K ≤ 2 ^ 12) :
    ∃ va vb : Nat → F8E5, (∀ k, (va k).isFinite = true) ∧ (∀ k, (vb k).isFinite = true) ∧
      |dotSum va vb K| + tol va vb K ≤ Math.Float.Format.binary16.maxVal ∧
      tol va vb K + (|dotSum va vb K| + tol va vb K) * Math.Float.Format.binary16.u +
        Math.Float.Format.binary16.eta < |0 - dotSum va vb K| := by
  refine ⟨spike8 (K - 1) one8, spike8 (K - 1) one8, spike8_fin one8_fin, spike8_fin one8_fin, ?_⟩
  have hS := dotSum_spike hK1 one8 one8
  have ht := tol_spike hK1 hK one8 one8
  have ht0 : 0 ≤ tol (spike8 (K - 1) one8) (spike8 (K - 1) one8) K := errAt_nonneg _ _
  rw [one8_val, mul_one, abs_one, mul_one, relDot8_64, absDot8_64] at ht
  rw [one8_val, mul_one] at hS
  rw [hS, Math.Float.Format.binary16_maxVal, Math.Float.Format.binary16_u, Math.Float.Format.binary16_eta]
  have hu : (2 : ℚ) ^ (-11 : ℤ) = 1 / 2048 := by norm_num
  have he : (2 : ℚ) ^ (-25 : ℤ) = 1 / 33554432 := by norm_num
  rw [hu, he]
  constructor
  · rw [abs_one]; linarith
  · rw [abs_one, zero_sub, abs_neg, abs_one]
    nlinarith

/-- **A skipped `K`-step breaks `Post` at every covered `K`**: for every step `s` of the loop
there is a finite input with room in `float16` on which `S − X_s`, the result of a kernel that
skips step `s` (and is otherwise exact; `X_s` the exact sum of step `s`'s products), is farther
from `S` than `OutSpec`'s `float16` bound allows. For the last step the input is one product `1`
at index `K − 1`; for an earlier step `s`, the row `1` at `64·s`, `-1` at `64·s + 64` against a
column of `1`s (`S = 0`, `X_s = 1`, and the bound below `1/5`). -/
theorem drop_rejected {K s : Nat} (hK : K ≤ 2 ^ 12) (hs : s < steps K) :
    ∃ va vb : Nat → F8E5, (∀ k, (va k).isFinite = true) ∧ (∀ k, (vb k).isFinite = true) ∧
      |dotSum va vb K| + tol va vb K ≤ Math.Float.Format.binary16.maxVal ∧
      tol va vb K + (|dotSum va vb K| + tol va vb K) * Math.Float.Format.binary16.u +
        Math.Float.Format.binary16.eta <
          |(dotSum va vb K - stepSum (prodV va vb K) s) - dotSum va vb K| := by
  have hK1 : 1 ≤ K := by unfold steps at hs; omega
  have hu : Math.Float.Format.binary16.u = 1 / 2048 := by
    rw [Math.Float.Format.binary16_u]; norm_num
  have he : Math.Float.Format.binary16.eta = 1 / 33554432 := by
    rw [Math.Float.Format.binary16_eta]; norm_num
  have hρ : (0 : ℚ) ≤ Rules.Num.relDot8 64 := by rw [relDot8_64]; norm_num
  have hα : (0 : ℚ) ≤ Rules.Num.absDot8 64 := by rw [absDot8_64]; norm_num
  rcases Nat.lt_or_ge (s + 1) (steps K) with hlt | hge
  · -- an earlier step: `1` at `64·s`, `-1` at `64·s + 64`
    have hT : 64 * s + 64 < K := by unfold steps at hlt; omega
    refine ⟨pair8 (64 * s) (64 * s + 64), ones8, pair8_fin _ _, fun _ => one8_fin, ?_⟩
    have hne : 64 * s ≠ 64 * s + 64 := by omega
    have hp := prodV_pair8 (K := K) (by omega) hT hne
    set p := pairP (64 * s) (64 * s + 64)
    have hS : dotSum (pair8 (64 * s) (64 * s + 64)) ones8 K = 0 := by
      have h := (partS_end (pair8 (64 * s) (64 * s + 64)) ones8 (K := K) (T := steps K)
        (by unfold steps; omega)).1
      rw [← h]
      unfold partS
      rw [hp, partSum_pairP, partSum_spikeP, partSum_spikeP, ite_eq_left (by omega),
        ite_eq_left (by omega)]
      norm_num
    have hX : stepSum (prodV (pair8 (64 * s) (64 * s + 64)) ones8 K) s = 1 := by
      rw [hp, stepSum_pairP, stepSum_spikeP, stepSum_spikeP, ite_eq_left (by omega),
        ite_eq_right (by omega)]
      norm_num
    -- the bound, step by step
    have hlo : ∀ t, 0 ≤ t → t < 0 + s → |partSum p t| + stepMag p t = 0 := fun t _ ht => by
      rw [partSum_pairP, partSum_spikeP, partSum_spikeP, stepMag_pairP hne, stepMag_spikeP,
        stepMag_spikeP, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
        ite_eq_right (by omega)]
      norm_num
    have hat : |partSum p s| + stepMag p s = 1 := by
      rw [partSum_pairP, partSum_spikeP, partSum_spikeP, stepMag_pairP hne, stepMag_spikeP,
        stepMag_spikeP, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
        ite_eq_right (by omega)]
      norm_num
    have hat1 : |partSum p (s + 1)| + stepMag p (s + 1) = 2 := by
      rw [partSum_pairP, partSum_spikeP, partSum_spikeP, stepMag_pairP hne, stepMag_spikeP,
        stepMag_spikeP, ite_eq_left (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
        ite_eq_left (by omega)]
      norm_num
    obtain ⟨n, hn⟩ : ∃ n, steps K = s + 2 + n := ⟨steps K - (s + 2), by omega⟩
    have hhi : ∀ t, s + 2 ≤ t → t < s + 2 + n → |partSum p t| + stepMag p t = 0 := fun t ht _ => by
      rw [partSum_pairP, partSum_spikeP, partSum_spikeP, stepMag_pairP hne, stepMag_spikeP,
        stepMag_spikeP, ite_eq_left (by omega), ite_eq_left (by omega), ite_eq_right (by omega),
        ite_eq_right (by omega)]
      norm_num
    have hTle : steps K ≤ 64 := by unfold steps; omega
    have hq0 := errAt_quiet p 0 s hlo
    have h0 : errAt p 0 = 0 := rfl
    rw [zero_add, h0, zero_add] at hq0
    have hq2 := errAt_quiet p (s + 2) n hhi
    have e1 : errAt p (s + 1) = (1 + Rules.Num.relDot8 64) * errAt p s +
        Rules.Num.relDot8 64 * (|partSum p s| + stepMag p s) + Rules.Num.absDot8 64 := rfl
    have e2 : errAt p (s + 2) = (1 + Rules.Num.relDot8 64) * errAt p (s + 1) +
        Rules.Num.relDot8 64 * (|partSum p (s + 1)| + stepMag p (s + 1)) + Rules.Num.absDot8 64 := rfl
    rw [hat] at e1
    rw [hat1] at e2
    have hPs := pow_rho_le (t := s) (by omega)
    have hPn := pow_rho_le (t := n) (by omega)
    have hs' : (s : ℚ) ≤ 64 := by exact_mod_cast (show s ≤ 64 by omega)
    have hn' : (n : ℚ) ≤ 64 := by exact_mod_cast (show n ≤ 64 by omega)
    have hEs : errAt p s ≤ 3 * (64 * Rules.Num.absDot8 64) :=
      hq0.trans (mul_le_mul hPs (mul_le_mul_of_nonneg_right hs' hα) (by positivity) (by norm_num))
    have hE0 := errAt_nonneg p (s + 2)
    have hEn : errAt p (s + 2 + n) ≤ 3 * (errAt p (s + 2) + 64 * Rules.Num.absDot8 64) :=
      hq2.trans (mul_le_mul hPn (by linarith [mul_le_mul_of_nonneg_right hn' hα]) (by positivity)
        (by norm_num))
    have htol : tol (pair8 (64 * s) (64 * s + 64)) ones8 K = errAt p (s + 2 + n) := by
      unfold tol; rw [hp, hn]
    have ht0 := errAt_nonneg p (s + 2 + n)
    rw [hS, hX, htol, Math.Float.Format.binary16_maxVal, hu, he]
    rw [relDot8_64, absDot8_64] at e1 e2
    rw [absDot8_64] at hEs hEn
    have hb : errAt p (s + 2 + n) < 1 / 5 := by
      have hE1 := errAt_nonneg p s
      linarith
    constructor
    · rw [abs_zero]; linarith
    · rw [abs_zero, zero_add]
      norm_num
      linarith
  · -- the last step: one product `1` at index `K − 1`
    have hsT : s + 1 = steps K := by omega
    refine ⟨spike8 (K - 1) one8, spike8 (K - 1) one8, spike8_fin one8_fin, spike8_fin one8_fin, ?_⟩
    have hS := dotSum_spike hK1 one8 one8
    have ht := tol_spike hK1 hK one8 one8
    have ht0 : 0 ≤ tol (spike8 (K - 1) one8) (spike8 (K - 1) one8) K := errAt_nonneg _ _
    have hl := steps_last hK1 hsT.symm
    have hX : stepSum (prodV (spike8 (K - 1) one8) (spike8 (K - 1) one8) K) s = 1 := by
      rw [prodV_spike8 (by omega), stepSum_spikeP, ite_eq_left (by omega), one8_val, mul_one]
    rw [one8_val, mul_one, abs_one, mul_one, relDot8_64, absDot8_64] at ht
    rw [one8_val, mul_one] at hS
    rw [hS, hX, Math.Float.Format.binary16_maxVal, hu, he]
    constructor
    · rw [abs_one]; linarith
    · rw [abs_one]
      norm_num
      linarith

end Kernels.MatmulE5m2
