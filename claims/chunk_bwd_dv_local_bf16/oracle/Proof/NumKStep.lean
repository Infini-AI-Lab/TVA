import Claim
import TVA.Proof
import Proof.Util

/-!
# One trip of the `K` loop in the reals

Trip `t` of `b_A += scale * tl.dot(b_k, b_q)` adds the product `σ̂ · X` of the `float32` scale and
the dot's result `X` to the accumulator `a` with one fusable multiply-add. `kstep_real` is the
arithmetic: if `X` is within `ρ·m + α` of the slab's sum `s` (magnitude `m`) and `a` is within
`σ((c0 + 4u t)·Mp + t·2^-114)` of `σ` times the sum `Sp` of the first `t` slabs (magnitude `Mp`),
then the new accumulator is within the same bound for `t + 1` slabs. The relative constant `c0 =
ρ + 5u` does not grow with the number of trips: each trip's error is relative to its own slab.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

set_option maxHeartbeats 1600000 in
/-- **The arithmetic of one trip.** All quantities are real numbers: `g` is the `float32`
scale (within `v` relative of `σ`), `x` the dot result, `a` the accumulator, `r` the new
accumulator, with the fused multiply-add's error bound `hr`. -/
theorem kstep_real {σ g x a r Sp Mp s m ρ α : ℝ} {t : Nat}
    (hσ1 : 1 / 2 ^ 32 ≤ σ) (hg : |g - σ| ≤ σ * (1 / 2 ^ 24 + 1 / 2 ^ 117))
    (hm0 : 0 ≤ m) (hs : |s| ≤ m) (hMp0 : 0 ≤ Mp) (hSp : |Sp| ≤ Mp)
    (ht : t ≤ 15) (hρ0 : 0 ≤ ρ) (hρ : ρ ≤ 1 / 2 ^ 14) (hα0 : 0 ≤ α) (hα : α ≤ 1 / 2 ^ 118)
    (hx : |x - s| ≤ ρ * m + α)
    (ha : |a - σ * Sp| ≤ σ * ((ρ + 5 / 2 ^ 24 + 4 / 2 ^ 24 * t) * Mp + t / 2 ^ 114))
    (hr : |r - (x * g + a)| ≤ 3 * (1 / 2 ^ 24) * (|x * g| + |a|) + 4 * (1 / 2 ^ 150)) :
    |r - σ * (Sp + s)| ≤ σ * ((ρ + 5 / 2 ^ 24 + 4 / 2 ^ 24 * (t + 1)) * (Mp + m) + (t + 1) / 2 ^ 114) := by
  have hσ0 : 0 < σ := lt_of_lt_of_le (by positivity) hσ1
  set v : ℝ := 1 / 2 ^ 24 + 1 / 2 ^ 117 with hv
  have hv0 : 0 ≤ v := by rw [hv]; positivity
  have hv1 : v ≤ 1 / 2 ^ 23 := by rw [hv]; norm_num
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg t
  have ht15 : (t : ℝ) ≤ 15 := by exact_mod_cast ht
  set c0 : ℝ := ρ + 5 / 2 ^ 24 with hc0
  set d0 : ℝ := (c0 + 4 / 2 ^ 24 * t) with hd0
  set Ea : ℝ := σ * (d0 * Mp + t / 2 ^ 114) with hEa
  have hd00 : 0 ≤ d0 := by rw [hd0, hc0]; positivity
  have hg' : |g| ≤ σ * (1 + v) := by
    have := abs_sub_abs_le_abs_sub g σ
    rw [abs_of_pos hσ0] at this
    nlinarith
  have hx' : |x| ≤ (1 + ρ) * m + α := by
    have := abs_sub_abs_le_abs_sub x s
    nlinarith
  have hxg : |x * g| ≤ ((1 + ρ) * m + α) * (σ * (1 + v)) := by
    rw [abs_mul]
    exact mul_le_mul hx' hg' (abs_nonneg _) (by positivity)
  have hxs : |x * g - σ * s| ≤ (ρ * m + α) * (σ * (1 + v)) + m * (σ * v) := by
    have e : x * g - σ * s = (x - s) * g + s * (g - σ) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul]
    have h1 : |x - s| * |g| ≤ (ρ * m + α) * (σ * (1 + v)) :=
      mul_le_mul hx hg' (abs_nonneg _) (by positivity)
    have h2 : |s| * |g - σ| ≤ m * (σ * v) := by
      refine mul_le_mul hs ?_ (abs_nonneg _) hm0
      calc |g - σ| ≤ σ * (1 / 2 ^ 24 + 1 / 2 ^ 117) := hg
        _ = σ * v := by rw [hv]
    linarith
  have haa : |a| ≤ σ * Mp + Ea := by
    have h1 := abs_sub_abs_le_abs_sub a (σ * Sp)
    have h2 : |σ * Sp| ≤ σ * Mp := by
      rw [abs_mul, abs_of_pos hσ0]; exact mul_le_mul_of_nonneg_left hSp hσ0.le
    linarith
  -- the coefficient inequalities
  have hcm : 3 * (1 / 2 ^ 24) * ((1 + ρ) * (1 + v)) + ρ * (1 + v) + v ≤ c0 + 4 / 2 ^ 24 * (t + 1) := by
    have h1 : (1 + ρ) * (1 + v) ≤ (1 + 1 / 2 ^ 14) * (1 + 1 / 2 ^ 23) :=
      mul_le_mul (by linarith) (by linarith) (by positivity) (by positivity)
    have h2 : ρ * v ≤ 1 / 2 ^ 14 * (1 / 2 ^ 23) := mul_le_mul hρ hv1 hv0 (by positivity)
    have h3 : 3 * (1 / 2 ^ 24) * ((1 + ρ) * (1 + v)) ≤ 3 * (1 / 2 ^ 24) * ((1 + 1 / 2 ^ 14) * (1 + 1 / 2 ^ 23)) :=
      mul_le_mul_of_nonneg_left h1 (by norm_num)
    have h4 : 4 / 2 ^ 24 * ((t : ℝ) + 1) ≥ 4 / 2 ^ 24 := by nlinarith
    have h5 : v ≤ 1 / 2 ^ 24 + 1 / 2 ^ 117 := hv.le
    rw [hc0]
    have e : ρ * (1 + v) = ρ + ρ * v := by ring
    rw [e]
    norm_num at h2 h3 h5 h4 ⊢
    linarith
  have hcM : 3 * (1 / 2 ^ 24) + (1 + 3 * (1 / 2 ^ 24)) * d0 ≤ c0 + 4 / 2 ^ 24 * (t + 1) := by
    have h1 : d0 ≤ 1 / 2 ^ 13 := by
      rw [hd0, hc0]
      have : (4 : ℝ) / 2 ^ 24 * t ≤ 4 / 2 ^ 24 * 15 := by gcongr
      norm_num at this ⊢
      linarith
    have h2 : 3 * (1 / 2 ^ 24) * d0 ≤ 3 * (1 / 2 ^ 24) * (1 / 2 ^ 13) :=
      mul_le_mul_of_nonneg_left h1 (by norm_num)
    have e : (1 + 3 * (1 / 2 ^ 24)) * d0 = d0 + 3 * (1 / 2 ^ 24) * d0 := by ring
    rw [e]
    have h3 : d0 + 4 / 2 ^ 24 ≤ c0 + 4 / 2 ^ 24 * (t + 1) := by
      rw [hd0]; nlinarith
    norm_num at h2 ⊢
    linarith
  have hca : (1 + 3 * (1 / 2 ^ 24)) * (t / 2 ^ 114) + 3 * (1 / 2 ^ 24) * α * (1 + v) +
      α * (1 + v) + 4 * (1 / 2 ^ 150) * 2 ^ 32 ≤ (t + 1) / 2 ^ 114 := by
    have h1 : α * (1 + v) ≤ 1 / 2 ^ 118 * (1 + 1 / 2 ^ 23) := by
      calc α * (1 + v) ≤ α * (1 + 1 / 2 ^ 23) := by gcongr
        _ ≤ 1 / 2 ^ 118 * (1 + 1 / 2 ^ 23) := by gcongr
    have h2 : 3 * (1 / 2 ^ 24) * α * (1 + v) ≤ 3 * (1 / 2 ^ 24) * (1 / 2 ^ 118 * (1 + 1 / 2 ^ 23)) := by
      have := mul_le_mul_of_nonneg_left h1 (by norm_num : (0 : ℝ) ≤ 3 * (1 / 2 ^ 24))
      linarith
    have h3 : (3 : ℝ) * (1 / 2 ^ 24) * (t / 2 ^ 114) ≤ 3 * (1 / 2 ^ 24) * (15 / 2 ^ 114) := by gcongr
    have hnum : 3 * (1 / 2 ^ 24) * (15 / 2 ^ 114) + 3 * (1 / 2 ^ 24) * (1 / 2 ^ 118 * (1 + 1 / 2 ^ 23)) +
        1 / 2 ^ 118 * (1 + 1 / 2 ^ 23) + 4 * (1 / 2 ^ 150) * 2 ^ 32 ≤ (1 : ℝ) / 2 ^ 114 := by norm_num
    have e : (1 + 3 * (1 / 2 ^ 24)) * ((t : ℝ) / 2 ^ 114) = t / 2 ^ 114 + 3 * (1 / 2 ^ 24) * (t / 2 ^ 114) := by ring
    have e2 : ((t : ℝ) + 1) / 2 ^ 114 = t / 2 ^ 114 + 1 / 2 ^ 114 := by ring
    rw [e, e2]
    linarith
  -- assemble
  have hrr : |r - (x * g + a)| ≤ 3 * (1 / 2 ^ 24) * (((1 + ρ) * m + α) * (σ * (1 + v)) + (σ * Mp + Ea)) +
      4 * (1 / 2 ^ 150) := by
    refine hr.trans ?_
    have := add_le_add hxg haa
    have h3 : (0 : ℝ) ≤ 3 * (1 / 2 ^ 24) := by norm_num
    have h4 := mul_le_mul_of_nonneg_left this h3
    linarith
  have hsplit : |r - σ * (Sp + s)| ≤ |r - (x * g + a)| + |x * g - σ * s| + |a - σ * Sp| := by
    have e : r - σ * (Sp + s) = (r - (x * g + a)) + (x * g - σ * s) + (a - σ * Sp) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    have := abs_add_le (r - (x * g + a)) (x * g - σ * s)
    linarith
  have hEa' : |a - σ * Sp| ≤ Ea := ha
  have hσ32 : 1 ≤ σ * 2 ^ 32 := by
    have := mul_le_mul_of_nonneg_right hσ1 (by positivity : (0 : ℝ) ≤ 2 ^ 32)
    norm_num at this ⊢; linarith
  -- expand
  have hE : Ea = σ * d0 * Mp + σ * (t / 2 ^ 114) := by rw [hEa]; ring
  have hη : 4 * (1 / 2 ^ 150) ≤ σ * (4 * (1 / 2 ^ 150) * 2 ^ 32) := by
    have h := mul_le_mul_of_nonneg_left hσ32 (by norm_num : (0 : ℝ) ≤ 4 * (1 / 2 ^ 150))
    have e : σ * (4 * (1 / 2 ^ 150) * 2 ^ 32) = 4 * (1 / 2 ^ 150) * (σ * 2 ^ 32) := by ring
    rw [e]; linarith
  have hMm : σ * m * (3 * (1 / 2 ^ 24) * ((1 + ρ) * (1 + v)) + ρ * (1 + v) + v) ≤
      σ * m * (c0 + 4 / 2 ^ 24 * (t + 1)) :=
    mul_le_mul_of_nonneg_left hcm (by positivity)
  have hMM : σ * Mp * (3 * (1 / 2 ^ 24) + (1 + 3 * (1 / 2 ^ 24)) * d0) ≤
      σ * Mp * (c0 + 4 / 2 ^ 24 * (t + 1)) :=
    mul_le_mul_of_nonneg_left hcM (by positivity)
  have hAA : σ * ((1 + 3 * (1 / 2 ^ 24)) * (t / 2 ^ 114) + 3 * (1 / 2 ^ 24) * α * (1 + v) +
      α * (1 + v) + 4 * (1 / 2 ^ 150) * 2 ^ 32) ≤ σ * ((t + 1) / 2 ^ 114) :=
    mul_le_mul_of_nonneg_left hca hσ0.le
  have hcalc : σ * ((c0 + 4 / 2 ^ 24 * (t + 1)) * (Mp + m) + (t + 1) / 2 ^ 114) =
      σ * m * (c0 + 4 / 2 ^ 24 * (t + 1)) + σ * Mp * (c0 + 4 / 2 ^ 24 * (t + 1)) +
        σ * ((t + 1) / 2 ^ 114) := by ring
  have hlhs : 3 * (1 / 2 ^ 24) * (((1 + ρ) * m + α) * (σ * (1 + v)) + (σ * Mp + Ea)) +
      4 * (1 / 2 ^ 150) + ((ρ * m + α) * (σ * (1 + v)) + m * (σ * v)) + Ea =
      σ * m * (3 * (1 / 2 ^ 24) * ((1 + ρ) * (1 + v)) + ρ * (1 + v) + v) +
      σ * Mp * (3 * (1 / 2 ^ 24) + (1 + 3 * (1 / 2 ^ 24)) * d0) +
      σ * ((1 + 3 * (1 / 2 ^ 24)) * (t / 2 ^ 114) + 3 * (1 / 2 ^ 24) * α * (1 + v) + α * (1 + v)) +
      4 * (1 / 2 ^ 150) := by
    rw [hE]; ring
  have hfinal : σ * ((1 + 3 * (1 / 2 ^ 24)) * (t / 2 ^ 114) + 3 * (1 / 2 ^ 24) * α * (1 + v) +
      α * (1 + v)) + 4 * (1 / 2 ^ 150) ≤ σ * ((t + 1) / 2 ^ 114) := by
    have : σ * ((1 + 3 * (1 / 2 ^ 24)) * (t / 2 ^ 114) + 3 * (1 / 2 ^ 24) * α * (1 + v) +
      α * (1 + v) + 4 * (1 / 2 ^ 150) * 2 ^ 32) = σ * ((1 + 3 * (1 / 2 ^ 24)) * (t / 2 ^ 114) +
        3 * (1 / 2 ^ 24) * α * (1 + v) + α * (1 + v)) + σ * (4 * (1 / 2 ^ 150) * 2 ^ 32) := by ring
    linarith
  have h1 : |r - σ * (Sp + s)| ≤
      3 * (1 / 2 ^ 24) * (((1 + ρ) * m + α) * (σ * (1 + v)) + (σ * Mp + Ea)) +
      4 * (1 / 2 ^ 150) + ((ρ * m + α) * (σ * (1 + v)) + m * (σ * v)) + Ea := by
    exact hsplit.trans (add_le_add (add_le_add hrr hxs) hEa')
  rw [hlhs] at h1
  rw [hcalc]
  linarith [h1, hMm, hMM, hfinal]

/-- **One trip of the `K` loop on `float32` values**: the fused multiply-add `X · σ̂ + a` of the
dot's result `X`, the scale `σ̂` and the accumulator `a` is finite and keeps the invariant. -/
theorem kstep {fusion : Bool} {X sg a r : F32} {σ Sp Mp s m ρ α : ℝ} {t : Nat}
    (hr : FloatV.NaNEq (FloatV.add (FloatV.mul X sg) a) r ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma X sg a) r))
    (hXf : X.isFinite = true) (hsf : sg.isFinite = true) (haf : a.isFinite = true)
    (hσ1 : 1 / 2 ^ 32 ≤ σ) (hσ2 : σ ≤ 2 ^ 32)
    (hsg : |sg.toReal - σ| ≤ σ * (1 / 2 ^ 24 + 1 / 2 ^ 117))
    (hm0 : 0 ≤ m) (hs : |s| ≤ m) (hMp0 : 0 ≤ Mp) (hSp : |Sp| ≤ Mp) (hM : σ * (Mp + m) ≤ 2 ^ 100)
    (ht : t ≤ 15) (hρ0 : 0 ≤ ρ) (hρ : ρ ≤ 1 / 2 ^ 14) (hα0 : 0 ≤ α) (hα : α ≤ 1 / 2 ^ 118)
    (hX : |X.toReal - s| ≤ ρ * m + α)
    (ha : |a.toReal - σ * Sp| ≤ σ * ((ρ + 5 / 2 ^ 24 + 4 / 2 ^ 24 * t) * Mp + t / 2 ^ 114)) :
    r.isFinite = true ∧
      |r.toReal - σ * (Sp + s)| ≤
        σ * ((ρ + 5 / 2 ^ 24 + 4 / 2 ^ 24 * (t + 1)) * (Mp + m) + (t + 1) / 2 ^ 114) := by
  have hf := Format.binary32_regular
  have hσ0 : 0 < σ := lt_of_lt_of_le (by positivity) hσ1
  -- magnitudes
  have hx' : |X.toReal| ≤ (1 + ρ) * m + α := by
    have := abs_sub_abs_le_abs_sub X.toReal s
    nlinarith
  have hv : (1 / 2 ^ 24 + 1 / 2 ^ 117 : ℝ) ≤ 1 / 2 ^ 23 := by norm_num
  have hg' : |sg.toReal| ≤ σ * (1 + 1 / 2 ^ 23) := by
    have := abs_sub_abs_le_abs_sub sg.toReal σ
    rw [abs_of_pos hσ0] at this
    nlinarith
  have hroomA : |X.toReal * sg.toReal| ≤ 2 ^ 101 := by
    rw [abs_mul]
    have h1 : |X.toReal| * |sg.toReal| ≤ ((1 + ρ) * m + α) * (σ * (1 + 1 / 2 ^ 23)) :=
      mul_le_mul hx' hg' (abs_nonneg _) (by positivity)
    have h2 : σ * m ≤ 2 ^ 100 := by nlinarith
    have h3 : σ * α ≤ 1 := by nlinarith
    have : ((1 + ρ) * m + α) * (σ * (1 + 1 / 2 ^ 23)) =
        (1 + ρ) * (1 + 1 / 2 ^ 23) * (σ * m) + (1 + 1 / 2 ^ 23) * (σ * α) := by ring
    rw [this] at h1
    have h4 : (1 + ρ) * (1 + 1 / 2 ^ 23) ≤ 2 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_right h4 (by positivity : 0 ≤ σ * m)]
  have haa : |a.toReal| ≤ 2 ^ 102 := by
    have h1 := abs_sub_abs_le_abs_sub a.toReal (σ * Sp)
    have h2 : |σ * Sp| ≤ σ * Mp := by
      rw [abs_mul, abs_of_pos hσ0]; exact mul_le_mul_of_nonneg_left hSp hσ0.le
    have h3 : σ * Mp ≤ 2 ^ 100 := by nlinarith
    have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg t
    have ht15 : (t : ℝ) ≤ 15 := by exact_mod_cast ht
    have h4 : σ * ((ρ + 5 / 2 ^ 24 + 4 / 2 ^ 24 * t) * Mp + t / 2 ^ 114) ≤ 2 ^ 100 + 1 := by
      have e : σ * ((ρ + 5 / 2 ^ 24 + 4 / 2 ^ 24 * t) * Mp + t / 2 ^ 114) =
          (ρ + 5 / 2 ^ 24 + 4 / 2 ^ 24 * t) * (σ * Mp) + σ * (t / 2 ^ 114) := by ring
      have h5 : (ρ + 5 / 2 ^ 24 + 4 / 2 ^ 24 * t) ≤ 1 := by nlinarith
      have h6 : σ * (t / 2 ^ 114) ≤ 1 := by
        have : (t : ℝ) / 2 ^ 114 ≤ 1 / 2 ^ 32 := by
          rw [div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith
        nlinarith
      rw [e]
      nlinarith [mul_le_mul_of_nonneg_right h5 (by positivity : 0 ≤ σ * Mp)]
    linarith
  have hmax : (2 ^ 101 + 2 ^ 102 : ℝ) ≤ ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
    push_cast
    rw [Format.binary32_maxVal_real, Format.binary32_eta_real]
    norm_num
  have hroom : |X.toRat * sg.toRat| + |a.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have h : ((|X.toRat * sg.toRat| + |a.toRat| : ℚ) : ℝ) ≤
        ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      push_cast
      have h1 : |X.toReal * sg.toReal| ≤ 2 ^ 101 := hroomA
      have h2 : |a.toReal| ≤ 2 ^ 102 := haa
      unfold FloatV.toReal at h1 h2
      push_cast at hmax
      linarith
    exact_mod_cast h
  obtain ⟨hrf, herr⟩ := FloatV.mulAdd_err hf (by rw [Format.binary32_u]; norm_num) hr hXf hsf haf hroom
  refine ⟨hrf, ?_⟩
  have herrR : |r.toReal - (X.toReal * sg.toReal + a.toReal)| ≤
      3 * (1 / 2 ^ 24) * (|X.toReal * sg.toReal| + |a.toReal|) + 4 * (1 / 2 ^ 150) := by
    have h := (by exact_mod_cast herr :
      |(r.toRat : ℝ) - ((X.toRat : ℝ) * (sg.toRat : ℝ) + (a.toRat : ℝ))| ≤
        3 * ((Format.binary32.u : ℚ) : ℝ) * (|(X.toRat : ℝ) * (sg.toRat : ℝ)| + |(a.toRat : ℝ)|) +
          4 * ((Format.binary32.eta : ℚ) : ℝ))
    rw [Format.binary32_u_real, Format.binary32_eta_real] at h
    unfold FloatV.toReal
    have e1 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
    have e2 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
    rw [e1, e2] at h
    exact h
  exact kstep_real hσ1 hsg hm0 hs hMp0 hSp ht hρ0 hρ hα0 hα hX ha herrR

/-- **The kernel's `float32` scale** is finite and within `2^-24 + 2^-117` relative of the host's
double, which is in `[2^-32, 2^32]`. -/
theorem scale_f32 {c : F64} (h : c.isFinite = true) (h1 : (1 / 2 ^ 32 : ℚ) ≤ c.toRat)
    (h2 : c.toRat ≤ 2 ^ 32) :
    (FloatV.hostF32 c).isFinite = true ∧
      |(FloatV.hostF32 c).toReal - (c.toRat : ℝ)| ≤
        (c.toRat : ℝ) * (1 / 2 ^ 24 + 1 / 2 ^ 117) := by
  have hnan : c.isNaN = false := ((FloatV.isFinite_iff).mp h).1
  have hh : FloatV.hostF32 c = (FloatV.cvt c : F32) := by
    unfold FloatV.hostF32; rw [hnan]; simp
  rw [hh]
  have hmax : |c.toRat| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos (lt_of_lt_of_le (by norm_num) h1), Format.binary32_maxVal]
    exact h2.trans (by norm_num)
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err Format.binary32_regular h hmax
  refine ⟨hfin, ?_⟩
  have h' := (by exact_mod_cast herr :
    |((FloatV.cvt c : F32).toRat : ℝ) - (c.toRat : ℝ)| ≤ |(c.toRat : ℝ)| * ((Format.binary32.u : ℚ) : ℝ) +
      ((Format.binary32.eta : ℚ) : ℝ))
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h'
  have hc1 : (1 / 2 ^ 32 : ℝ) ≤ (c.toRat : ℝ) := by
    have := (Rat.cast_le (K := ℝ)).mpr h1
    push_cast at this
    exact this
  rw [abs_of_pos (lt_of_lt_of_le (by norm_num) hc1)] at h'
  have e1 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have e2 : (2 : ℝ) ^ (-150 : ℤ) ≤ (1 / 2 ^ 32) * (1 / 2 ^ 117) := by norm_num
  rw [e1] at h'
  unfold FloatV.toReal
  have := mul_le_mul_of_nonneg_right hc1 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 117)
  nlinarith

end Claims.ChunkBwdDvLocal
