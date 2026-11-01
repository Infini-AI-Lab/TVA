import MatmulE5m2.Basic

/-!
# `matmul_e5m2`: what the tolerance rejects

`Post` holds for every input, so it rejects a wrong kernel as soon as one input exposes it. On
inputs with one or two nonzero products the error bound `tol` stays small at every covered `K`:

* `tol_spike`: one product `X = x·y`, at index `K − 1`: `tol ≤ 271425·2^-24·L + 400·2^-126`, with
  `L = liftMag x · liftMag y` (`|X|` when `x` and `y` are normal) and
  `271425·2^-24 = (1 + 2^-7 + 2^-12)^2 − 1`, about `1.62%`.
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

/-- Its lifted magnitudes: `1` at `e1` and at `e2`. -/
def pairL (e1 e2 : Nat) : Nat → ℚ := fun k => spikeP e1 1 k + spikeP e2 1 k

/-- The exact sum of step `t`'s products, `p e` for `64·t ≤ e < 64·t + 64`: what a kernel that
skips step `t` leaves out. -/
def stepSum (p : Nat → ℚ) (t : Nat) : ℚ := ∑ k : Fin 64, p (64 * t + k)

theorem partSum_spikeP (e : Nat) (x : ℚ) (j : Nat) :
    partSum (spikeP e x) j = if e < 32 * j then x else 0 := by
  unfold partSum spikeP
  rw [Finset.sum_ite_eq' (Finset.range (32 * j)) e (fun _ => x)]
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

theorem partSum_pairP (e1 e2 j : Nat) :
    partSum (pairP e1 e2) j = partSum (spikeP e1 1) j + partSum (spikeP e2 (-1)) j := by
  unfold partSum pairP
  exact Finset.sum_add_distrib

theorem stepSum_pairP (e1 e2 t : Nat) :
    stepSum (pairP e1 e2) t = stepSum (spikeP e1 1) t + stepSum (spikeP e2 (-1)) t := by
  unfold stepSum pairP
  exact Finset.sum_add_distrib

/-- A spike's largest value on a chunk is at most `x` (for `x ≥ 0`). -/
theorem chunkMax_spikeP_le {e : Nat} {x : ℚ} (hx : 0 ≤ x) (j : Nat) : chunkMax (spikeP e x) j ≤ x :=
  chunkMax_le fun k => by
    unfold spikeP
    split_ifs
    · exact le_rfl
    · exact hx

/-- On a chunk without the spike's index it is at most `0`. -/
theorem chunkMax_spikeP_out {e j : Nat} (x : ℚ) (h : e < 32 * j ∨ 32 * j + 32 ≤ e) :
    chunkMax (spikeP e x) j ≤ 0 :=
  chunkMax_le fun k => by
    have := k.isLt
    unfold spikeP
    split_ifs with h'
    · exact absurd h' (by omega)
    · exact le_rfl

theorem chunkMax_pairL_le {e1 e2 : Nat} (hne : e1 ≠ e2) (j : Nat) : chunkMax (pairL e1 e2) j ≤ 1 :=
  chunkMax_le fun k => by
    unfold pairL spikeP
    split_ifs with h1 h2 h2
    · exact absurd (h1.symm.trans h2) hne
    · norm_num
    · norm_num
    · norm_num

theorem chunkMax_pairL_out {e1 e2 j : Nat} (h1 : e1 < 32 * j ∨ 32 * j + 32 ≤ e1)
    (h2 : e2 < 32 * j ∨ 32 * j + 32 ≤ e2) : chunkMax (pairL e1 e2) j ≤ 0 :=
  chunkMax_le fun k => by
    have := k.isLt
    unfold pairL spikeP
    rw [ite_eq_right (by omega), ite_eq_right (by omega)]
    norm_num

/-! ## The error bound, chunk by chunk -/

/-- One chunk of the error bound, with bounds `a` on `max(|S j|, m j)` and `b` on `|S (j + 1)|`. -/
theorem errAt_succ_le (p q : Nat → ℚ) {j j' : Nat} (hj : j' = j + 1) {a b : ℚ}
    (ha : max |partSum p j| (chunkMax q j) ≤ a) (hb : |partSum p j'| ≤ b) :
    errAt p q j' ≤ 4129 / 4096 * errAt p q j + a / 128 + b / 4096 + 1 / 2 ^ 126 := by
  subst hj
  rw [errAt_succ, rho_eq, sigma_eq, alpha_eq]
  linarith

/-- Chunks whose partial sums and lifted products are all `0` only grow the bound by
`(1 + ρ + σ)` and `α`. -/
theorem errAt_quiet (p q : Nat → ℚ) (j0 : Nat) : ∀ n : Nat,
    (∀ j, j0 ≤ j → j < j0 + n → partSum p j = 0 ∧ partSum p (j + 1) = 0 ∧ chunkMax q j ≤ 0) →
    errAt p q (j0 + n) ≤ (4129 / 4096) ^ n * (errAt p q j0 + n / 2 ^ 126)
  | 0, _ => by simp
  | n + 1, h => by
    have ih := errAt_quiet p q j0 n (fun j h1 h2 => h j h1 (by omega))
    obtain ⟨h1, h2, h3⟩ := h (j0 + n) (by omega) (by omega)
    have hs := errAt_succ_le p q (j := j0 + n) (j' := j0 + (n + 1)) (by ring) (a := 0) (b := 0)
      (by rw [h1, abs_zero]; exact max_le le_rfl h3) (by rw [show j0 + (n + 1) = j0 + n + 1 by ring,
        h2, abs_zero])
    have hE := errAt_nonneg p q j0
    have hP1 : (1 : ℚ) ≤ (4129 / 4096) ^ (n + 1) := one_le_pow₀ (by norm_num)
    have hm := mul_le_mul_of_nonneg_left ih (show (0 : ℚ) ≤ 4129 / 4096 by norm_num)
    have e : (4129 / 4096 : ℚ) * ((4129 / 4096) ^ n * (errAt p q j0 + n / 2 ^ 126)) =
        (4129 / 4096) ^ (n + 1) * (errAt p q j0 + n / 2 ^ 126) := by ring
    have e2 : (1 / 2 ^ 126 : ℚ) ≤ (4129 / 4096) ^ (n + 1) * (1 / 2 ^ 126) :=
      le_mul_of_one_le_left (by norm_num) hP1
    have e3 : (4129 / 4096 : ℚ) ^ (n + 1) * (errAt p q j0 + ((n + 1 : Nat) : ℚ) / 2 ^ 126) =
        (4129 / 4096) ^ (n + 1) * (errAt p q j0 + n / 2 ^ 126) +
          (4129 / 4096) ^ (n + 1) * (1 / 2 ^ 126) := by push_cast; ring
    rw [e3]
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

/-- `1` and `-1` are normal: their lifted magnitude is `1`. -/
theorem liftMag_one : Rules.Num.liftMag .f8e5 1 = 1 := by
  norm_num [Rules.Num.liftMag, Rules.Num.minNormal8]

theorem liftMag_mone : Rules.Num.liftMag .f8e5 (-1) = 1 := by
  norm_num [Rules.Num.liftMag, Rules.Num.minNormal8]

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

theorem liftV_spike8 {e K : Nat} (he : e < K) (x y : F8E5) :
    liftV (spike8 e x) (spike8 e y) K =
      spikeP e (Rules.Num.liftMag .f8e5 x.toRat * Rules.Num.liftMag .f8e5 y.toRat) := by
  funext k
  unfold liftV spike8 spikeP
  by_cases hk : k = e
  · subst hk
    simp [he]
  · simp [hk, zero8_val, Dot8Chunk.liftMag_zero]

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

theorem liftV_pair8 {e1 e2 K : Nat} (h1 : e1 < K) (h2 : e2 < K) (hne : e1 ≠ e2) :
    liftV (pair8 e1 e2) ones8 K = pairL e1 e2 := by
  funext k
  unfold liftV pair8 ones8 pairL spikeP
  by_cases hk1 : k = e1
  · subst hk1
    simp [h1, hne, one8_val, liftMag_one]
  · by_cases hk2 : k = e2
    · subst hk2
      simp [h2, hk1, one8_val, mone8_val, liftMag_one, liftMag_mone]
    · simp [hk1, hk2, zero8_val, Dot8Chunk.liftMag_zero]

/-! ## One product at index `K − 1` -/

theorem steps_last {K : Nat} (hK1 : 1 ≤ K) {n : Nat} (hn : steps K = n + 1) :
    64 * n ≤ K - 1 ∧ K - 1 < 64 * n + 64 := by
  unfold steps at hn; omega

/-- **One product `X = x·y` at index `K − 1`** (the loop's last step): the bound is at most
`((1 + ρ + σ)^2 − 1)·L + 400·α = 271425·2^-24·L + 400·2^-126`, `L = liftMag x · liftMag y`, about
`1.62%` of `L` (of `|X|` for normal `x`, `y`), at every covered `K`. Before the chunk `J` of index
`K − 1` nothing is added; chunk `J` adds `ρ·L + σ·|X|`; the loop's last chunk, when it is not `J`
(it is all disabled lanes), adds `(ρ + σ)·|X|` more. -/
theorem tol_spike {K : Nat} (hK1 : 1 ≤ K) (hK : K ≤ 2 ^ 12) (x y : F8E5) :
    tol (spike8 (K - 1) x) (spike8 (K - 1) y) K ≤
      271425 / 2 ^ 24 * (Rules.Num.liftMag .f8e5 x.toRat * Rules.Num.liftMag .f8e5 y.toRat) +
        400 / 2 ^ 126 := by
  have hL0 : 0 ≤ Rules.Num.liftMag .f8e5 x.toRat * Rules.Num.liftMag .f8e5 y.toRat :=
    mul_nonneg (Dot8Chunk.liftMag_nonneg _ _) (Dot8Chunk.liftMag_nonneg _ _)
  have hXL : |x.toRat * y.toRat| ≤ Rules.Num.liftMag .f8e5 x.toRat * Rules.Num.liftMag .f8e5 y.toRat :=
    Dot8Chunk.abs_mul_le_liftMag _ _ _ _
  unfold tol
  rw [prodV_spike8 (by omega), liftV_spike8 (by omega)]
  set L := Rules.Num.liftMag .f8e5 x.toRat * Rules.Num.liftMag .f8e5 y.toRat with hL
  set p := spikeP (K - 1) (x.toRat * y.toRat) with hp
  set q := spikeP (K - 1) L with hq
  obtain ⟨J, hJ⟩ : ∃ J, J = (K - 1) / 32 := ⟨_, rfl⟩
  have hJ1 : 32 * J ≤ K - 1 ∧ K - 1 < 32 * J + 32 := by omega
  have hJn : J + 1 = 2 * steps K ∨ J + 2 = 2 * steps K := by unfold steps; omega
  have hJ128 : J ≤ 128 := by omega
  -- chunks before `J`: nothing
  have hq0 := errAt_quiet p q 0 J (fun j _ hj =>
    ⟨by rw [hp, partSum_spikeP, ite_eq_right (by omega)],
      by rw [hp, partSum_spikeP, ite_eq_right (by omega)], chunkMax_spikeP_out _ (by omega)⟩)
  rw [zero_add] at hq0
  have h00 : errAt p q 0 = 0 := rfl
  rw [h00, zero_add] at hq0
  have hP := pow_q_le hJ128
  have hJ' : (J : ℚ) / 2 ^ 126 ≤ 128 / 2 ^ 126 := by
    have : (J : ℚ) ≤ 128 := by exact_mod_cast hJ128
    linarith
  have hEJ : errAt p q J ≤ 3 * (128 / 2 ^ 126) :=
    hq0.trans (mul_le_mul hP hJ' (by positivity) (by norm_num))
  -- chunk `J`: the product
  have hS1 : |partSum p (J + 1)| ≤ L := by
    rw [hp, partSum_spikeP, ite_eq_left (by omega)]; exact hXL
  have hm : max |partSum p J| (chunkMax q J) ≤ L := by
    rw [hp, partSum_spikeP, ite_eq_right (by omega), abs_zero]
    exact max_le hL0 (chunkMax_spikeP_le hL0 J)
  have hJs := errAt_succ_le p q rfl hm hS1
  rcases hJn with h | h
  · rw [← h]
    linarith
  · rw [← h]
    have hS2 : |partSum p (J + 2)| ≤ L := by
      rw [hp, partSum_spikeP, ite_eq_left (by omega)]; exact hXL
    have hm2 : max |partSum p (J + 1)| (chunkMax q (J + 1)) ≤ L :=
      max_le hS1 ((chunkMax_spikeP_out _ (by omega)).trans hL0)
    have hJs2 := errAt_succ_le p q (j := J + 1) (j' := J + 2) rfl hm2 hS2
    linarith

theorem dotSum_spike {K : Nat} (hK1 : 1 ≤ K) (x y : F8E5) :
    dotSum (spike8 (K - 1) x) (spike8 (K - 1) y) K = x.toRat * y.toRat := by
  rw [← partS_end (spike8 (K - 1) x) (spike8 (K - 1) y) (K := K) (T := steps K)
    (by unfold steps; omega)]
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
  have ht0 : 0 ≤ tol (spike8 (K - 1) one8) (spike8 (K - 1) one8) K := errAt_nonneg _ _ _
  rw [one8_val, liftMag_one, mul_one] at ht
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
  rcases Nat.lt_or_ge (s + 1) (steps K) with hlt | hge
  · -- an earlier step: `1` at `64·s`, `-1` at `64·s + 64`
    have hT : 64 * s + 64 < K := by unfold steps at hlt; omega
    refine ⟨pair8 (64 * s) (64 * s + 64), ones8, pair8_fin _ _, fun _ => one8_fin, ?_⟩
    have hne : 64 * s ≠ 64 * s + 64 := by omega
    have hp := prodV_pair8 (K := K) (by omega) hT hne
    have hl := liftV_pair8 (K := K) (by omega) hT hne
    set p := pairP (64 * s) (64 * s + 64) with hpdef
    set q := pairL (64 * s) (64 * s + 64) with hqdef
    have hS : dotSum (pair8 (64 * s) (64 * s + 64)) ones8 K = 0 := by
      rw [← partS_end (pair8 (64 * s) (64 * s + 64)) ones8 (K := K) (T := steps K)
        (by unfold steps; omega)]
      unfold partS
      rw [hp, partSum_pairP, partSum_spikeP, partSum_spikeP, ite_eq_left (by unfold steps; omega),
        ite_eq_left (by unfold steps; omega)]
      norm_num
    have hX : stepSum (prodV (pair8 (64 * s) (64 * s + 64)) ones8 K) s = 1 := by
      rw [hp, stepSum_pairP, stepSum_spikeP, stepSum_spikeP, ite_eq_left (by omega),
        ite_eq_right (by omega)]
      norm_num
    -- the partial sums and largest lifted products at each chunk
    have hps : ∀ j, partSum p j =
        (if 64 * s < 32 * j then 1 else 0) + (if 64 * s + 64 < 32 * j then -1 else 0) := fun j => by
      rw [hpdef, partSum_pairP, partSum_spikeP, partSum_spikeP]
    have hp0 : ∀ j, j ≤ 2 * s → partSum p j = 0 := fun j hj => by
      rw [hps, ite_eq_right (by omega), ite_eq_right (by omega)]; norm_num
    have hp1 : partSum p (2 * s + 1) = 1 := by
      rw [hps, ite_eq_left (by omega), ite_eq_right (by omega)]; norm_num
    have hp2 : partSum p (2 * s + 2) = 1 := by
      rw [hps, ite_eq_left (by omega), ite_eq_right (by omega)]; norm_num
    have hp3 : ∀ j, 2 * s + 3 ≤ j → partSum p j = 0 := fun j hj => by
      rw [hps, ite_eq_left (by omega), ite_eq_left (by omega)]; norm_num
    have hq1 : ∀ j, chunkMax q j ≤ 1 := chunkMax_pairL_le hne
    have hqo : ∀ j, j ≠ 2 * s → j ≠ 2 * s + 2 → chunkMax q j ≤ 0 := fun j h1 h2 =>
      chunkMax_pairL_out (by omega) (by omega)
    obtain ⟨n, hn⟩ : ∃ n, 2 * steps K = 2 * s + 3 + n := ⟨2 * steps K - (2 * s + 3), by omega⟩
    have hTle : steps K ≤ 64 := by unfold steps; omega
    -- the chunks before `2·s`: nothing
    have hq0 := errAt_quiet p q 0 (2 * s) (fun j _ hj =>
      ⟨hp0 j (by omega), hp0 (j + 1) (by omega), hqo j (by omega) (by omega)⟩)
    rw [zero_add] at hq0
    have h00 : errAt p q 0 = 0 := rfl
    rw [h00, zero_add] at hq0
    have hPs := pow_q_le (j := 2 * s) (by omega)
    have hs' : ((2 * s : Nat) : ℚ) / 2 ^ 126 ≤ 128 / 2 ^ 126 := by
      have : ((2 * s : Nat) : ℚ) ≤ 128 := by exact_mod_cast (show 2 * s ≤ 128 by omega)
      linarith
    have hE0 : errAt p q (2 * s) ≤ 3 * (128 / 2 ^ 126) :=
      hq0.trans (mul_le_mul hPs hs' (by positivity) (by norm_num))
    -- chunks `2·s` (the `1`), `2·s + 1`, `2·s + 2` (the `-1`)
    have e1 := errAt_succ_le p q (j := 2 * s) (j' := 2 * s + 1) rfl (a := 1) (b := 1)
      (by rw [hp0 _ le_rfl, abs_zero]; exact max_le zero_le_one (hq1 _)) (by rw [hp1, abs_one])
    have e2 := errAt_succ_le p q (j := 2 * s + 1) (j' := 2 * s + 2) rfl (a := 1) (b := 1)
      (by rw [hp1, abs_one]; exact max_le le_rfl ((hqo _ (by omega) (by omega)).trans zero_le_one))
      (by rw [hp2, abs_one])
    have e3 := errAt_succ_le p q (j := 2 * s + 2) (j' := 2 * s + 3) rfl (a := 1) (b := 0)
      (by rw [hp2, abs_one]; exact max_le le_rfl (hq1 _)) (by rw [hp3 _ le_rfl, abs_zero])
    -- the chunks after: nothing
    have hq2 := errAt_quiet p q (2 * s + 3) n (fun j hj _ =>
      ⟨hp3 j hj, hp3 (j + 1) (by omega), hqo j (by omega) (by omega)⟩)
    have hPn := pow_q_le (j := n) (by omega)
    have hn' : (n : ℚ) / 2 ^ 126 ≤ 128 / 2 ^ 126 := by
      have : (n : ℚ) ≤ 128 := by exact_mod_cast (show n ≤ 128 by omega)
      linarith
    have hE3 := errAt_nonneg p q (2 * s + 3)
    have hEn : errAt p q (2 * s + 3 + n) ≤ 3 * (errAt p q (2 * s + 3) + 128 / 2 ^ 126) :=
      hq2.trans (mul_le_mul hPn (by linarith) (by positivity) (by norm_num))
    have htol : tol (pair8 (64 * s) (64 * s + 64)) ones8 K = errAt p q (2 * s + 3 + n) := by
      unfold tol; rw [hp, hl, hn]
    have ht0 := errAt_nonneg p q (2 * s + 3 + n)
    rw [hS, hX, htol, Math.Float.Format.binary16_maxVal, hu, he]
    have hb : errAt p q (2 * s + 3 + n) < 1 / 5 := by
      have hE1 := errAt_nonneg p q (2 * s)
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
    have ht0 : 0 ≤ tol (spike8 (K - 1) one8) (spike8 (K - 1) one8) K := errAt_nonneg _ _ _
    have hl := steps_last hK1 hsT.symm
    have hX : stepSum (prodV (spike8 (K - 1) one8) (spike8 (K - 1) one8) K) s = 1 := by
      rw [prodV_spike8 (by omega), stepSum_spikeP, ite_eq_left (by omega), one8_val, mul_one]
    rw [one8_val, liftMag_one, mul_one] at ht
    rw [one8_val, mul_one] at hS
    rw [hS, hX, Math.Float.Format.binary16_maxVal, hu, he]
    constructor
    · rw [abs_one]; linarith
    · rw [abs_one]
      norm_num
      linarith

end Kernels.MatmulE5m2
