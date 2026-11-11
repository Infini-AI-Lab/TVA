import Proof.Setup

/-!
# `tma_persistent_matmul_f16`: The error recursion of the inner reduction and the magnitudes it uses.
-/

namespace Claims.TmaPersistentMatmul

open TVA

/-!
## The constants of the reduction

`rho` and `alpha` are the per-step constants of `Device.DotF16` with `64` products. This file
bounds them, gives the closed form of the error recursion `E` and relaxes its value after
`steps K ≤ 128` steps to the claim's tolerance `2^-15·pathMag + 2^-112`.
-/


/-! ## The per-step constants -/

/-- The per-step relative constant is positive. -/
theorem rho_pos : 0 < rho := by
  unfold rho DotLoop.stepRel Device.relDot16
  rw [Format.binary32_u]
  positivity

/-- The per-step absolute constant is nonnegative. -/
theorem alpha_nonneg : 0 ≤ alpha := by
  unfold alpha DotLoop.stepAbs Device.absDot16
  rw [Format.binary32_u, Format.binary32_eta]
  positivity

/-- The per-step relative constant is at most `67·2^-22`. -/
theorem rho_le : rho ≤ 67 * 2 ^ (-22 : ℤ) := by
  unfold rho DotLoop.stepRel Device.relDot16
  rw [Format.binary32_u]
  norm_num

/-- The per-step absolute constant is at most `67·2^-126`. -/
theorem alpha_le : alpha ≤ 67 * 2 ^ (-126 : ℤ) := by
  unfold alpha DotLoop.stepAbs Device.absDot16
  rw [Format.binary32_u, Format.binary32_eta]
  norm_num

/-- `(1 + rho)^n` stays below `1 + 2·n·rho` up to `n = 128`. -/
theorem one_add_rho_pow_le {n : Nat} (hn : n ≤ 128) : (1 + rho) ^ n ≤ 1 + 2 * n * rho := by
  have h : (n : ℚ) * rho ≤ 1 / 2 := by
    have hn' : (n : ℚ) ≤ 128 := by exact_mod_cast hn
    have h1 : (n : ℚ) * rho ≤ 128 * rho := mul_le_mul_of_nonneg_right hn' rho_pos.le
    have h2 : (128 : ℚ) * rho ≤ 128 * (67 * 2 ^ (-22 : ℤ)) :=
      mul_le_mul_of_nonneg_left rho_le (by norm_num)
    have h3 : (128 : ℚ) * (67 * 2 ^ (-22 : ℤ)) ≤ 1 / 2 := by norm_num
    exact h1.trans (h2.trans h3)
  exact DotLoop.pow_le rho_pos.le n h

/-! ## Nonnegativity of the magnitudes -/

/-- The lifted magnitude of the first `t` steps is nonnegative. -/
theorem lSum_nonneg (d : Inst) (init : Mem) (i j t : Nat) : 0 ≤ lSum d init i j t := by
  unfold lSum stepMag liftV
  apply Finset.sum_nonneg
  intro k _
  split_ifs with h
  · exact mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  · norm_num

/-- `|pSum| + lSum` is nonnegative. -/
theorem abs_add_nonneg (d : Inst) (init : Mem) (i j t : Nat) :
    0 ≤ |pSum d init i j t| + lSum d init i j t := by
  exact add_nonneg (abs_nonneg _) (lSum_nonneg d init i j t)

/-- The path magnitude is a sum of nonnegative terms. -/
theorem dPath_nonneg (d : Inst) (init : Mem) (i j : Nat) : 0 ≤ dPath d init i j := by
  unfold dPath pathMag
  apply Finset.sum_nonneg
  intro t _
  apply add_nonneg (abs_nonneg _)
  unfold stepMag liftV
  apply Finset.sum_nonneg
  intro k _
  split_ifs with h
  · exact mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  · norm_num

/-! ## The closed form of `E` -/

/-- The error recursion `E` in closed form: a weighted sum of the per-step terms. -/
theorem E_closed (d : Inst) (init : Mem) (i j T : Nat) :
    E d init i j T =
      ∑ t ∈ Finset.range T,
        (1 + rho) ^ (T - 1 - t) * (rho * (|pSum d init i j t| + lSum d init i j t) + alpha) := by
  induction T with
  | zero => simp [E]
  | succ T ih =>
    rw [E.eq_2, ih, Finset.sum_range_succ, Finset.mul_sum]
    have hlast : (1 + rho) ^ (T + 1 - 1 - T) = 1 := by
      rw [show T + 1 - 1 - T = 0 by omega, pow_zero]
    rw [hlast, one_mul]
    have hpt : ∀ t ∈ Finset.range T,
        (1 + rho) *
            ((1 + rho) ^ (T - 1 - t) *
              (rho * (|pSum d init i j t| + lSum d init i j t) + alpha))
          = (1 + rho) ^ (T + 1 - 1 - t) *
              (rho * (|pSum d init i j t| + lSum d init i j t) + alpha) := by
      intro t ht
      have ht' : t < T := Finset.mem_range.mp ht
      have hidx : T + 1 - 1 - t = (T - 1 - t) + 1 := by omega
      rw [hidx, pow_succ]
      ring
    rw [Finset.sum_congr rfl hpt]
    ring

/-- The error after `T ≤ steps K` steps is at most `(1+rho)^(T-1)` times the sum of the step
terms, and that sum is bounded by `rho·pathMag + T·alpha`. -/
theorem E_le_bound {d : Inst} (_hd : d.ok) (init : Mem) (i j T : Nat) (hT0 : 0 < T)
    (hT : T ≤ steps d.K) :
    E d init i j T ≤ (1 + rho) ^ (T - 1) * (rho * dPath d init i j + T * alpha) := by
  rw [E_closed]
  have hc0 : ∀ t, 0 ≤ rho * (|pSum d init i j t| + lSum d init i j t) + alpha := by
    intro t
    have h := abs_add_nonneg d init i j t
    nlinarith [rho_pos, alpha_nonneg]
  have hP0 : 0 ≤ (1 + rho) ^ (T - 1) := by
    have h1 : (0 : ℚ) ≤ 1 + rho := by linarith [rho_pos]
    exact pow_nonneg h1 _
  have hterm : ∀ t ∈ Finset.range T,
      (1 + rho) ^ (T - 1 - t) * (rho * (|pSum d init i j t| + lSum d init i j t) + alpha)
        ≤ (1 + rho) ^ (T - 1) *
            (rho * (|pSum d init i j t| + lSum d init i j t) + alpha) := by
    intro t _
    apply mul_le_mul_of_nonneg_right _ (hc0 t)
    apply pow_le_pow_right₀ _ (by omega)
    linarith [rho_pos]
  have hsum_eq : ∑ t ∈ Finset.range T,
      (rho * (|pSum d init i j t| + lSum d init i j t) + alpha)
      = rho * (∑ t ∈ Finset.range T, (|pSum d init i j t| + lSum d init i j t)) + T * alpha := by
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hS : ∑ t ∈ Finset.range T, (|pSum d init i j t| + lSum d init i j t) ≤ dPath d init i j := by
    have hsub : Finset.range T ⊆ Finset.range (steps d.K) := Finset.range_subset_range.mpr hT
    have h := Finset.sum_le_sum_of_subset_of_nonneg
      (f := fun t => |pSum d init i j t| + lSum d init i j t) hsub
      (fun x _ _ => abs_add_nonneg d init i j x)
    simpa [dPath, pathMag, pSum, lSum] using h
  calc ∑ t ∈ Finset.range T,
        (1 + rho) ^ (T - 1 - t) * (rho * (|pSum d init i j t| + lSum d init i j t) + alpha)
      ≤ ∑ t ∈ Finset.range T,
        (1 + rho) ^ (T - 1) *
          (rho * (|pSum d init i j t| + lSum d init i j t) + alpha) :=
        Finset.sum_le_sum hterm
    _ = (1 + rho) ^ (T - 1) * (∑ t ∈ Finset.range T,
        (rho * (|pSum d init i j t| + lSum d init i j t) + alpha)) := by
        rw [Finset.mul_sum]
    _ = (1 + rho) ^ (T - 1) * (rho * (∑ t ∈ Finset.range T,
        (|pSum d init i j t| + lSum d init i j t)) + T * alpha) := by rw [hsum_eq]
    _ ≤ (1 + rho) ^ (T - 1) * (rho * dPath d init i j + T * alpha) := by
        apply mul_le_mul_of_nonneg_left _ hP0
        have h1 : rho * (∑ t ∈ Finset.range T, (|pSum d init i j t| + lSum d init i j t))
            ≤ rho * dPath d init i j := mul_le_mul_of_nonneg_left hS rho_pos.le
        linarith

/-- **The tolerance.** After all `steps K ≤ 128` steps the accumulated error is within the
claim's tolerance. -/
theorem E_le_dTol {d : Inst} (hd : d.ok) (init : Mem) (i j : Nat) :
    E d init i j (steps d.K) ≤ dTol d init i j := by
  have hK0 : 0 < d.K := hd.pos.2.2
  have hT0 : 0 < steps d.K := steps_pos hK0
  have hTle : steps d.K ≤ 128 := steps_le hd.K_le
  have hP0 : 0 ≤ (1 + rho) ^ (steps d.K - 1) := by
    have h1 : (0 : ℚ) ≤ 1 + rho := by linarith [rho_pos]
    exact pow_nonneg h1 _
  have hPle : (1 + rho) ^ (steps d.K - 1) ≤ 1 + 254 * rho := by
    have h1 := one_add_rho_pow_le (n := steps d.K - 1) (by omega)
    have hTq : ((steps d.K - 1 : Nat) : ℚ) ≤ 127 := by
      have h2 : steps d.K - 1 ≤ 127 := by omega
      exact_mod_cast h2
    have h2 : (2 : ℚ) * ((steps d.K - 1 : Nat) : ℚ) * rho ≤ 254 * rho := by
      have h2' : (2 : ℚ) * ((steps d.K - 1 : Nat) : ℚ) ≤ 254 := by linarith
      calc (2 : ℚ) * ((steps d.K - 1 : Nat) : ℚ) * rho
          = ((2 : ℚ) * ((steps d.K - 1 : Nat) : ℚ)) * rho := by ring
        _ ≤ 254 * rho := mul_le_mul_of_nonneg_right h2' rho_pos.le
    linarith
  have hA : (1 + rho) ^ (steps d.K - 1) * rho ≤ 2 ^ (-15 : ℤ) := by
    have h1 : (1 + rho) ^ (steps d.K - 1) * rho ≤ (1 + 254 * rho) * rho :=
      mul_le_mul_of_nonneg_right hPle rho_pos.le
    have h2 : (1 + 254 * rho) * rho ≤ 2 ^ (-15 : ℤ) := by
      have h22 : (2 : ℚ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
      have h15 : (2 : ℚ) ^ (-15 : ℤ) = 1 / 32768 := by norm_num
      have hrle : rho ≤ 67 / 4194304 := by
        have h := rho_le
        rw [h22] at h
        norm_num at h
        exact h
      rw [h15]
      have hsq : rho * rho ≤ (67 / 4194304) * (67 / 4194304) :=
        mul_le_mul hrle hrle rho_pos.le (by norm_num)
      nlinarith
    linarith
  have hB : (1 + rho) ^ (steps d.K - 1) * ((steps d.K : ℚ) * alpha) ≤ 2 ^ (-112 : ℤ) := by
    have h1 : (1 + rho) ^ (steps d.K - 1) * ((steps d.K : ℚ) * alpha)
        ≤ (1 + 254 * rho) * (128 * (67 * 2 ^ (-126 : ℤ))) := by
      apply mul_le_mul hPle _ _ _
      · apply mul_le_mul _ alpha_le alpha_nonneg (by norm_num)
        exact_mod_cast hTle
      · exact mul_nonneg (Nat.cast_nonneg _) alpha_nonneg
      · nlinarith [rho_pos]
    have h2 : (1 + 254 * rho) * (128 * (67 * 2 ^ (-126 : ℤ))) ≤ 2 ^ (-112 : ℤ) := by
      have h22 : (2 : ℚ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
      have h126 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 85070591730234615865843651857942052864 := by
        norm_num
      have h112 : (2 : ℚ) ^ (-112 : ℤ) = 1 / 5192296858534827628530496329220096 := by
        norm_num
      have hrle : rho ≤ 67 / 4194304 := by
        have h := rho_le
        rw [h22] at h
        norm_num at h
        exact h
      rw [h126, h112]
      nlinarith
    linarith
  have hdP : 0 ≤ dPath d init i j := dPath_nonneg d init i j
  have hmain := E_le_bound hd init i j (steps d.K) hT0 le_rfl
  calc E d init i j (steps d.K)
      ≤ (1 + rho) ^ (steps d.K - 1) *
          (rho * dPath d init i j + steps d.K * alpha) := hmain
    _ = ((1 + rho) ^ (steps d.K - 1) * rho) * dPath d init i j
          + (1 + rho) ^ (steps d.K - 1) * ((steps d.K : ℚ) * alpha) := by ring
    _ ≤ 2 ^ (-15 : ℤ) * dPath d init i j + 2 ^ (-112 : ℤ) := by
        have e1 : ((1 + rho) ^ (steps d.K - 1) * rho) * dPath d init i j
            ≤ 2 ^ (-15 : ℤ) * dPath d init i j := mul_le_mul_of_nonneg_right hA hdP
        linarith
    _ = dTol d init i j := by simp only [dTol, tol, dPath]

/-!
## One step of the reduction

`DotLift.dotAdd_step` bounds one `tl.dot` against the accumulator and the step's products. This file
turns that bound into a step of the error recursion `E`: the accumulator after the step is within
`E (q + 1)` of the exact partial sum after the step, and the room condition the step needs follows
from the magnitude bounds.
-/


/-- The step's products sum to the difference of the two partial sums. -/
theorem pSum_succ {d : Inst} {init : Mem} {I J q : Nat} (hK : 64 * (q + 1) ≤ d.K) :
    pSum d init I J (q + 1) = pSum d init I J q
      + ∑ k : Fin 64, (aGet d init I (64 * q + k.val)).toRat
          * (bGet d init J (64 * q + k.val)).toRat := by
  unfold pSum partSum prodV
  rw [show 64 * (q + 1) = 64 * q + 64 by ring, Finset.sum_range_add]
  have hB : (∑ x ∈ Finset.range 64,
        (if 64 * q + x < d.K then
          (aGet d init I (64 * q + x)).toRat * (bGet d init J (64 * q + x)).toRat else 0))
      = ∑ x ∈ Finset.range 64,
        ((aGet d init I (64 * q + x)).toRat * (bGet d init J (64 * q + x)).toRat) := by
    apply Finset.sum_congr rfl
    intro x hx
    have hx' : x < 64 := Finset.mem_range.mp hx
    rw [ite_eq_left (by omega : 64 * q + x < d.K)]
  rw [hB, Fin.sum_univ_eq_sum_range
    (fun n => (aGet d init I (64 * q + n)).toRat * (bGet d init J (64 * q + n)).toRat)]

/-- The step's lifted magnitude, as the sum over its `64` products. -/
theorem lSum_eq {d : Inst} {init : Mem} {I J q : Nat} (hK : 64 * (q + 1) ≤ d.K) :
    lSum d init I J q = ∑ k : Fin 64,
      Device.liftMag .f16 (aGet d init I (64 * q + k.val)).toRat
        * Device.liftMag .f16 (bGet d init J (64 * q + k.val)).toRat := by
  unfold lSum stepMag liftV
  exact Finset.sum_congr rfl fun k _ => by
    have hk : 64 * q + k.val < d.K := by
      have := k.isLt
      omega
    rw [ite_eq_left hk]

/-- The error recursion is nonnegative. -/
theorem E_nonneg (d : Inst) (init : Mem) (i j t : Nat) : 0 ≤ E d init i j t := by
  induction t with
  | zero => rw [E.eq_1]
  | succ t ih =>
    rw [E.eq_2]
    have h1 : 0 ≤ |pSum d init i j t| + lSum d init i j t := abs_add_nonneg d init i j t
    have h2 : 0 ≤ (1 + rho) * E d init i j t := by
      apply mul_nonneg _ ih
      linarith [rho_pos]
    have h3 : 0 ≤ rho * (|pSum d init i j t| + lSum d init i j t) :=
      mul_nonneg rho_pos.le h1
    linarith [alpha_nonneg]

/-- The error recursion is monotone. -/
theorem E_mono {d : Inst} {init : Mem} {i j t u : Nat} (h : t ≤ u) :
    E d init i j t ≤ E d init i j u := by
  have hstep : ∀ n, E d init i j n ≤ E d init i j (n + 1) := by
    intro n
    rw [E.eq_2]
    have h1 : 0 ≤ |pSum d init i j n| + lSum d init i j n := abs_add_nonneg d init i j n
    have h2 : 0 ≤ E d init i j n := E_nonneg d init i j n
    have h3 : 0 ≤ rho * (|pSum d init i j n| + lSum d init i j n) :=
      mul_nonneg rho_pos.le h1
    nlinarith [rho_pos, alpha_nonneg]
  refine Nat.le_induction (P := fun n _ => E d init i j t ≤ E d init i j n)
    le_rfl ?_ u h
  intro n _ ih
  exact ih.trans (hstep n)

/-- Its value at any step of the inner loop is within the tolerance. -/
theorem E_le_dTol_q {d : Inst} (hd : d.ok) (init : Mem) (i j q : Nat) (hq : q ≤ steps d.K) :
    E d init i j q ≤ dTol d init i j := by
  exact le_trans (E_mono hq) (E_le_dTol hd init i j)

/-- One step of the error recursion: a step within `rho·(|c| + l) + alpha` of `c + (p1 − p0)`,
starting from an accumulator within `e` of `p0`, ends within `(1 + rho)·e + rho·(|p0| + l) + alpha`
of `p1`. -/
theorem red_arith {p0 p1 l c r e : ℚ}
    (hstep : |r - (c + (p1 - p0))| ≤ rho * (|c| + l) + alpha) (hE : |c - p0| ≤ e) :
    |r - p1| ≤ (1 + rho) * e + rho * (|p0| + l) + alpha := by
  have hc : |c| ≤ e + |p0| := by
    have hce : c = (c - p0) + p0 := by ring
    rw [hce]
    linarith [abs_add_le (c - p0) p0, hE]
  have hrw : r - p1 = (r - (c + (p1 - p0))) + (c - p0) := by ring
  have htri : |r - p1| ≤ |r - (c + (p1 - p0))| + |c - p0| := by
    rw [hrw]
    exact abs_add_le _ _
  have hmul : rho * (|c| + l) ≤ rho * (e + |p0| + l) := by
    apply mul_le_mul_of_nonneg_left _ rho_pos.le
    linarith
  nlinarith [htri, hstep, hmul, hE]

/-- The room condition `DotLift.dotAdd_step` asks for holds for the accumulators of the inner
loop. -/
theorem room_of_le {x l : ℚ} (hx : |x| ≤ 2 ^ (46 : ℤ)) (_hl0 : 0 ≤ l)
    (hl : l ≤ 2 ^ (38 : ℤ)) :
    2 * ((1 + Device.relDot16 64) * (|x| + l) + Device.absDot16 64
      + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
  unfold Device.relDot16 Device.absDot16
  rw [Format.binary32_eta]
  have h46 : (2 : ℚ) ^ (46 : ℤ) = 70368744177664 := by norm_num
  have h38 : (2 : ℚ) ^ (38 : ℤ) = 274877906944 := by norm_num
  have h22 : (2 : ℚ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  have h126 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 85070591730234615865843651857942052864 := by
    norm_num
  have h150 : (2 : ℚ) ^ (-150 : ℤ) =
      1 / 1427247692705959881058285969449495136382746624 := by norm_num
  have hx' : |x| ≤ 70368744177664 := by rw [← h46]; exact hx
  have hl' : l ≤ 274877906944 := by rw [← h38]; exact hl
  have hmain : 2 * ((1 + (64 + 2) * (1 / 4194304)) * (|x| + l)
      + (64 + 2) * (1 / 85070591730234615865843651857942052864)
      + 2 * (1 / 1427247692705959881058285969449495136382746624)) ≤ (2 : ℚ) ^ 50 := by
    nlinarith [hx', hl', abs_nonneg x]
  have hmax : (2 : ℚ) ^ 50 ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    norm_num
  rw [h22, h126, h150]
  linarith

/-!
## The magnitudes of the products

Every finite `float16` is at most `65504` in magnitude. A reduction of at most `2^13` products of
such values therefore has partial sums of magnitude below `2^45`, one step's lifted magnitude below
`2^38`, and a path magnitude below `2^52`.
-/


/-- A finite `float16` is at most `65504` in magnitude. -/
theorem f16_abs_le {x : F16} (h : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have h1 := FloatV.abs_toRat_le_maxVal Format.binary16_regular h
  rwa [Format.binary16_maxVal] at h1

/-- A finite `float16` element of `a` has magnitude at most `65504`. -/
theorem aGet_abs_le {d : Inst} {init : Mem} {I k : Nat} (h : (aGet d init I k).isFinite = true) :
    |(aGet d init I k).toRat| ≤ 65504 :=
  f16_abs_le h

/-- A finite `float16` element of `b` has magnitude at most `65504`. -/
theorem bGet_abs_le {d : Inst} {init : Mem} {J k : Nat} (h : (bGet d init J k).isFinite = true) :
    |(bGet d init J k).toRat| ≤ 65504 :=
  f16_abs_le h

/-- A lifted magnitude of a finite value is at most `65504`. -/
theorem liftMag_le {x : ℚ} (h : |x| ≤ 65504) : Device.liftMag .f16 x ≤ 65504 := by
  unfold Device.liftMag Device.liftFloor
  split_ifs with hx
  · norm_num
  · exact max_le h (by norm_num)

/-- The partial sum before step `q` is at most `64·q` products of magnitude `65504^2`. -/
theorem partSum_abs_le {d : Inst} (init : Mem) (I J q : Nat)
    (hI : RowFin d init I) (hJ : ColFin d init J) (hq : 64 * q ≤ d.K) :
    |pSum d init I J q| ≤ 64 * (q : ℚ) * (65504 * 65504) := by
  have hterm : ∀ e ∈ Finset.range (64 * q),
      |prodV (aGet d init I) (bGet d init J) d.K e| ≤ 65504 * 65504 := by
    intro e he
    have heK : e < d.K := lt_of_lt_of_le (Finset.mem_range.mp he) hq
    rw [prodV]; simp only [heK, ↓reduceIte, abs_mul]
    exact mul_le_mul (aGet_abs_le (hI e heK)) (bGet_abs_le (hJ e heK))
      (abs_nonneg _) (by norm_num)
  calc |pSum d init I J q|
      = |∑ e ∈ Finset.range (64 * q), prodV (aGet d init I) (bGet d init J) d.K e| := rfl
    _ ≤ ∑ e ∈ Finset.range (64 * q), |prodV (aGet d init I) (bGet d init J) d.K e| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ e ∈ Finset.range (64 * q), (65504 * 65504 : ℚ) := Finset.sum_le_sum hterm
    _ = ((64 * q : ℕ) : ℚ) * (65504 * 65504) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    _ = 64 * (q : ℚ) * (65504 * 65504) := by push_cast; ring

/-- The partial sums are small. -/
theorem pSum_abs_le {d : Inst} (hd : d.ok) (init : Mem) (I J q : Nat)
    (hI : RowFin d init I) (hJ : ColFin d init J) (hq : q ≤ steps d.K) :
    |pSum d init I J q| ≤ 2 ^ (45 : ℤ) := by
  have hKdvd : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.1
  have hKm : 64 * steps d.K = d.K := steps_mul hKdvd
  have hq64 : 64 * q ≤ d.K := by
    have h := Nat.mul_le_mul_left 64 hq
    omega
  have h1 := partSum_abs_le init I J q hI hJ hq64
  have hq' : 64 * (q : ℚ) ≤ 2 ^ 13 := by
    have h2 : ((64 * q : ℕ) : ℚ) ≤ (d.K : ℚ) := by exact_mod_cast hq64
    have h3 : (d.K : ℚ) ≤ 2 ^ 13 := by exact_mod_cast hd.K_le
    push_cast at h2 ⊢
    linarith
  calc |pSum d init I J q| ≤ 64 * (q : ℚ) * (65504 * 65504) := h1
    _ ≤ 2 ^ 13 * (65504 * 65504) := by
        apply mul_le_mul_of_nonneg_right hq' (by norm_num)
    _ ≤ 2 ^ (45 : ℤ) := by norm_num

/-- One step's lifted magnitude is at most `64` products of magnitude `65504^2`. -/
theorem stepMag_le {d : Inst} (init : Mem) (I J q : Nat)
    (hI : RowFin d init I) (hJ : ColFin d init J) (hq : 64 * (q + 1) ≤ d.K) :
    lSum d init I J q ≤ 64 * (65504 * 65504) := by
  have hterm : ∀ k : Fin 64,
      liftV (aGet d init I) (bGet d init J) d.K (64 * q + k.val) ≤ 65504 * 65504 := by
    intro k
    have heK : 64 * q + k.val < d.K := by
      have hk : (k : Nat) < 64 := k.isLt
      have h1 : 64 * (q + 1) = 64 * q + 64 := by ring
      omega
    rw [liftV]; simp only [heK, ↓reduceIte]
    exact mul_le_mul (liftMag_le (aGet_abs_le (hI _ heK)))
      (liftMag_le (bGet_abs_le (hJ _ heK))) (Device.liftMag_nonneg _ _) (by norm_num)
  calc lSum d init I J q
      = ∑ k : Fin 64, liftV (aGet d init I) (bGet d init J) d.K (64 * q + k.val) := rfl
    _ ≤ ∑ _k : Fin 64, (65504 * 65504 : ℚ) := Finset.sum_le_sum (fun k _ => hterm k)
    _ = 64 * (65504 * 65504) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        norm_num

/-- The step's lifted magnitude is small. -/
theorem lSum_le {d : Inst} (hd : d.ok) (init : Mem) (I J q : Nat)
    (hI : RowFin d init I) (hJ : ColFin d init J) (hq : q < steps d.K) :
    lSum d init I J q ≤ 2 ^ (38 : ℤ) := by
  have hKdvd : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.1
  have hKm : 64 * steps d.K = d.K := steps_mul hKdvd
  have hq64 : 64 * (q + 1) ≤ d.K := by
    have h : q + 1 ≤ steps d.K := hq
    have h' := Nat.mul_le_mul_left 64 h
    omega
  have h1 := stepMag_le init I J q hI hJ hq64
  calc lSum d init I J q ≤ 64 * (65504 * 65504) := h1
    _ ≤ 2 ^ (38 : ℤ) := by norm_num

/-- The path magnitude is bounded. -/
theorem dPath_le {d : Inst} (hd : d.ok) (init : Mem) (I J : Nat)
    (hI : RowFin d init I) (hJ : ColFin d init J) :
    dPath d init I J ≤ 2 ^ (52 : ℤ) := by
  have hKdvd : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.1
  have hKm : 64 * steps d.K = d.K := steps_mul hKdvd
  have hNle : steps d.K ≤ 128 := steps_le hd.K_le
  have hN0 : 0 < steps d.K := steps_pos hd.pos.2.2
  have hC0 : (0 : ℚ) ≤ 65504 * 65504 := by norm_num
  have hterm : ∀ t ∈ Finset.range (steps d.K),
      |partSum (prodV (aGet d init I) (bGet d init J) d.K) t|
        + stepMag (liftV (aGet d init I) (bGet d init J) d.K) t
      ≤ 64 * (steps d.K : ℚ) * (65504 * 65504) := by
    intro t ht
    have htN : t < steps d.K := Finset.mem_range.mp ht
    have htK : 64 * t ≤ d.K := by omega
    have h1 : |partSum (prodV (aGet d init I) (bGet d init J) d.K) t|
        ≤ 64 * (t : ℚ) * (65504 * 65504) := partSum_abs_le init I J t hI hJ htK
    have h2 : stepMag (liftV (aGet d init I) (bGet d init J) d.K) t ≤ 64 * (65504 * 65504) := by
      apply stepMag_le init I J t hI hJ
      omega
    have ht1 : (t : ℚ) + 1 ≤ (steps d.K : ℚ) := by
      have h : ((t + 1 : ℕ) : ℚ) ≤ (steps d.K : ℚ) := by exact_mod_cast htN
      push_cast at h
      exact h
    have hCa : 64 * (t : ℚ) * (65504 * 65504) + 64 * (65504 * 65504)
        ≤ 64 * (steps d.K : ℚ) * (65504 * 65504) := by
      have h : 64 * ((t : ℚ) + 1) ≤ 64 * (steps d.K : ℚ) := by linarith
      have h' := mul_le_mul_of_nonneg_right h hC0
      calc 64 * (t : ℚ) * (65504 * 65504) + 64 * (65504 * 65504)
          = 64 * ((t : ℚ) + 1) * (65504 * 65504) := by ring
        _ ≤ 64 * (steps d.K : ℚ) * (65504 * 65504) := h'
    linarith
  calc dPath d init I J
      = ∑ t ∈ Finset.range (steps d.K),
          (|partSum (prodV (aGet d init I) (bGet d init J) d.K) t|
            + stepMag (liftV (aGet d init I) (bGet d init J) d.K) t) := rfl
    _ ≤ ∑ _t ∈ Finset.range (steps d.K), (64 * (steps d.K : ℚ) * (65504 * 65504)) :=
        Finset.sum_le_sum hterm
    _ = ((steps d.K : ℕ) : ℚ) * (64 * (steps d.K : ℚ) * (65504 * 65504)) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    _ ≤ 2 ^ (52 : ℤ) := by
        have hNq : (steps d.K : ℚ) ≤ 128 := by exact_mod_cast hNle
        have hNq0 : (0 : ℚ) ≤ (steps d.K : ℚ) := by positivity
        have hsq : (steps d.K : ℚ) * (steps d.K : ℚ) ≤ 128 * 128 := by nlinarith
        nlinarith [hC0]

/-- The tolerance is below `2^38`. -/
theorem dTol_le {d : Inst} (hd : d.ok) (init : Mem) (I J : Nat)
    (hI : RowFin d init I) (hJ : ColFin d init J) :
    dTol d init I J ≤ 2 ^ (38 : ℤ) := by
  have hP : dPath d init I J ≤ 2 ^ (52 : ℤ) := dPath_le hd init I J hI hJ
  have hP0 : 0 ≤ dPath d init I J := dPath_nonneg d init I J
  simp only [dTol, tol, dPath] at hP hP0 ⊢
  have h1 : (2 : ℚ) ^ (-15 : ℤ) * pathMag (aGet d init I) (bGet d init J) d.K
      ≤ (2 : ℚ) ^ (-15 : ℤ) * 2 ^ (52 : ℤ) :=
    mul_le_mul_of_nonneg_left hP (by norm_num)
  have h2 : (2 : ℚ) ^ (-15 : ℤ) * 2 ^ (52 : ℤ) = 2 ^ (37 : ℤ) := by norm_num
  have h3 : (2 : ℚ) ^ (-112 : ℤ) ≤ 2 ^ (37 : ℤ) := by norm_num
  have h4 : (2 : ℚ) ^ (37 : ℤ) ≤ 2 ^ (38 : ℤ) := by norm_num
  linarith

end Claims.TmaPersistentMatmul
