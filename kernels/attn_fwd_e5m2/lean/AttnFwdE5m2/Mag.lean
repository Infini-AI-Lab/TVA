import AttnFwdE5m2.Final

/-!
# `attn_fwd_e5m2`: the second bound on the output, over the computed weights

`acc` and `l_i` are rescaled by the same `alpha` (lines 95 and 106), so `acc − r·l_i` for a fixed
`r` (the reference `attnRef`) is a sum over the computed `p` of `p·(v − r)`, up to the roundings
and the fp8 rounding of `p`. These lemmas bound it without comparing `p` with the exact weights.

* `lminB`, `lmin_real`: `l_i ≥ 1 − 2^-9 − t/2^12` after `t ≥ 1` blocks (the lane that sets the
  running maximum has `p ≈ 1` (`p_big`), and a kept maximum has `alpha ≈ 1`);
* `lane_pq`, `lane_E`: one lane's `p`, its fp8 rounding `q`, and `q·v − r·p`;
* `estep_real`: one block of `|acc − r·l_i| ≤ A·l_i + D` (`Model.magNext`, `Model.magD`), with the
  `p·v` `tl.dot`'s error in the form `Row.mag_tail` gives it from the chunk facts;
* `final_v`: the end of a row (lines 242, 245) from that bound; `ref_le`: `|attnRef| ≤ 57344`.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## `l_i` from below -/

/-- The lower bound on `l_i` after `t ≥ 1` blocks. -/
noncomputable def lminB (t : ℕ) : ℝ := 1 - 1 / 2 ^ 9 - t / 2 ^ 12

theorem lminB_half {t : ℕ} (ht : t ≤ 1024) : 1 / 2 ≤ lminB t := by
  unfold lminB
  have : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  have : (t : ℝ) / 2 ^ 12 ≤ 1 / 4 := by rw [div_le_iff₀ (by norm_num)]; linarith
  norm_num at this ⊢; linarith

theorem lminB_le (t : ℕ) : lminB t ≤ 1 - 1 / 2 ^ 9 := by
  unfold lminB
  have : (0 : ℝ) ≤ t / 2 ^ 12 := by positivity
  linarith

theorem lminB_succ (t : ℕ) : lminB (t + 1) = lminB t - 1 / 2 ^ 12 := by
  unfold lminB; push_cast; ring

theorem two_rpow_ge_one_sub {δ : ℝ} (hδ : 0 ≤ δ) : 1 - δ ≤ (2 : ℝ) ^ (-δ) := by
  rw [Real.rpow_def_of_pos (by norm_num)]
  have h1 := Real.add_one_le_exp (Real.log 2 * -δ)
  have h2 : Real.log 2 < 1 := by have := Real.log_two_lt_d9; linarith
  have h3 : δ * Real.log 2 ≤ δ * 1 := mul_le_mul_of_nonneg_left h2.le hδ
  linarith

/-- A lane's `p` from below, summed: `Σ p ≥ Σ|p| − 2^-118` when every `p ≥ −2^-125`. -/
theorem sum_p_ge {p : Fin 64 → ℝ} (hp0 : ∀ jj, -(1 / 2 ^ 125) ≤ p jj) :
    ∑ jj, |p jj| - 1 / 2 ^ 118 ≤ ∑ jj, p jj := by
  have h := lane_sum_ge (a := p) (c := fun jj => |p jj|) (k := 1) (e := 1 / 2 ^ 124) fun jj => by
    rcases le_or_gt 0 (p jj) with h | h
    · rw [abs_of_nonneg h]; norm_num
    · rw [abs_of_neg h]; have := hp0 jj; norm_num at this ⊢; linarith
  norm_num at h ⊢; linarith

/-- The old `l_i`'s rounding: `l·α − 3·|l·α|/2^24 ≥ −2^-81` for `α ≥ −2^-126`, `0 ≤ l ≤ 2^44`. -/
theorem la_low {l α : ℝ} (hl0 : 0 ≤ l) (hl1 : l ≤ 2 ^ 44) (hα0 : -(1 / 2 ^ 126) ≤ α) :
    -(1 / 2 ^ 81) ≤ l * α - 3 * |l * α| / 2 ^ 24 := by
  rcases le_or_gt 0 (l * α) with h | h
  · rw [abs_of_nonneg h]
    have : 0 ≤ l * α * (1 - 3 / 2 ^ 24) := mul_nonneg h (by norm_num)
    have e : l * α - 3 * (l * α) / 2 ^ 24 = l * α * (1 - 3 / 2 ^ 24) := by ring
    rw [e]; linarith [show (0 : ℝ) < 1 / 2 ^ 81 by positivity]
  · rw [abs_of_neg h]
    have h1 : -(l * (1 / 2 ^ 126)) ≤ l * α := by nlinarith
    have h2 : l * (1 / 2 ^ 126) ≤ 2 ^ 44 * (1 / 2 ^ 126) := mul_le_mul_of_nonneg_right hl1 (by norm_num)
    have e : l * α - 3 * -(l * α) / 2 ^ 24 = l * α * (1 + 3 / 2 ^ 24) := by ring
    rw [e]
    have h3 : -(2 ^ 44 * (1 / 2 ^ 126)) * (1 + 3 / 2 ^ 24) ≤ l * α * (1 + 3 / 2 ^ 24) :=
      mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
    norm_num at h3 ⊢; linarith

/-- **One block for `l_i` from below** (lines 84–85, 106). Either the maximum is kept, so `alpha`
is at least `1 − 2^-12` and the old `l_i` at least `lminB t`, or a lane's `p` is at least
`1 − 2^-10` (the lane that sets the new maximum). -/
theorem lmin_real {l l' α lij : ℝ} {p : Fin 64 → ℝ} {t : ℕ} (ht : t ≤ 1024) (hl0 : 0 ≤ l)
    (hl1 : l ≤ 2 ^ 44) (hα0 : -(1 / 2 ^ 126) ≤ α) (hp0 : ∀ jj, -(1 / 2 ^ 125) ≤ p jj)
    (hpb : ∀ jj, |p jj| ≤ 2)
    (hlij : |lij - ∑ jj, p jj| ≤ gamr * ∑ jj, |p jj| + 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24) ^ 63)
    (hl' : |l' - (l * α + lij)| ≤ 3 * (|l * α| + |lij|) / 2 ^ 24 + 4 / 2 ^ 150)
    (hcase : (lminB t ≤ l ∧ 1 - 1 / 2 ^ 12 ≤ α) ∨ ∃ jj, 1 - 1 / 2 ^ 10 ≤ p jj) :
    lminB (t + 1) ≤ l' := by
  rw [lminB_succ]
  have hγ0 := gamr_nonneg
  have hγ1 := gamr_le
  have hη : 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24 : ℝ) ^ 63 ≤ 1 / 2 ^ 143 := by norm_num
  have hη0 : 0 ≤ 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24 : ℝ) ^ 63 := by positivity
  set S := ∑ jj, |p jj| with hSd
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hSb : S ≤ 128 := by
    have := lane_sum_le (a := fun jj => |p jj|) (c := fun _ => (0 : ℝ)) (k := 0) (e := 2) fun jj => by
      simp only [zero_mul, zero_add]; exact hpb jj
    norm_num at this; linarith
  have hsp := sum_p_ge hp0
  rw [← hSd] at hsp
  have hγS : gamr * S ≤ S / 2 ^ 17 := by
    have := mul_le_mul_of_nonneg_right hγ1 hS0; linarith
  have hlij1 := abs_le.mp hlij
  have hsabs : |∑ jj, p jj| ≤ S := Finset.abs_sum_le_sum_abs _ _
  have hlijlow : (1 - 1 / 2 ^ 17) * S - 1 / 2 ^ 117 ≤ lij := by
    have := hlij1.1; norm_num at hη ⊢; nlinarith
  have hlijabs : |lij| ≤ (1 + 1 / 2 ^ 17) * S + 1 / 2 ^ 143 := by
    have := abs_sub_abs_le_abs_sub lij (∑ jj, p jj)
    have := hsabs; nlinarith
  have hla := la_low hl0 hl1 hα0
  have hl'1 := (abs_le.mp hl').1
  rcases hcase with ⟨hlm, hα⟩ | ⟨jj, hpj⟩
  · -- the maximum kept
    have hα0' : 0 ≤ α := by linarith
    have hlα : 0 ≤ l * α := mul_nonneg hl0 hα0'
    rw [abs_of_nonneg hlα] at hl'1
    have hlm0 : 1 / 2 ≤ lminB t := lminB_half ht
    have hlm1 := lminB_le t
    have h1 : lminB t * (1 - 1 / 2 ^ 12) ≤ l * α :=
      mul_le_mul hlm hα (by norm_num) hl0
    have h2 : 0 ≤ (1 - 1 / 2 ^ 17) * S - 3 * ((1 + 1 / 2 ^ 17) * S) / 2 ^ 24 := by
      have : 0 ≤ S * (1 - 1 / 2 ^ 17 - 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24) := mul_nonneg hS0 (by norm_num)
      linarith
    have h3 : lminB t * (1 - 1 / 2 ^ 12) * (1 - 3 / 2 ^ 24) ≤ l * α * (1 - 3 / 2 ^ 24) :=
      mul_le_mul_of_nonneg_right h1 (by norm_num)
    have h4 : lminB t - 1 / 2 ^ 12 + 1 / 2 ^ 22 ≤ lminB t * (1 - 1 / 2 ^ 12) * (1 - 3 / 2 ^ 24) := by
      nlinarith
    nlinarith
  · -- a lane at the new maximum
    have hsp2 : p jj - 63 / 2 ^ 125 ≤ ∑ i, p i := by
      have h := Finset.single_le_sum (f := fun i => p i + 1 / 2 ^ 125)
        (fun i _ => by have := hp0 i; linarith) (Finset.mem_univ jj)
      rw [Finset.sum_add_distrib] at h
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h
      norm_num at h ⊢; linarith
    have hlij2 : 1 - 1 / 2 ^ 9 - 1 / 2 ^ 120 ≤ lij := by
      have := hlij1.1
      have : gamr * S ≤ 128 / 2 ^ 17 := by linarith
      norm_num at hη hpj ⊢; nlinarith
    have hlij0 : 0 ≤ lij := by
      have : (0 : ℝ) ≤ 1 - 1 / 2 ^ 9 - 1 / 2 ^ 120 := by norm_num
      linarith
    rw [abs_of_nonneg hlij0] at hl'1
    have hlm1 := lminB_le t
    linarith

/-! ## One lane -/

/-- The lane at the new maximum: an exponent within `2^-11` of `0` gives `p ≥ 1 − 2^-10`. -/
theorem p_big {P : Platform} (hexp : Rules.Num.Exp2F32 P) {x p : F32} (hxf : x.isFinite = true)
    (hx : |x.toReal| ≤ 1 / 2 ^ 11) (hp : P.math .exp2 .binary32 x p) : 1 - 1 / 2 ^ 10 ≤ p.toReal := by
  have hx1 : x.toReal ≤ 100 := by have := le_abs_self x.toReal; linarith
  obtain ⟨-, hpe⟩ := exp2_acc hexp hp hxf hx1
  have h1 : (2 : ℝ) ^ (-(1 / 2 ^ 11 : ℝ)) ≤ (2 : ℝ) ^ x.toReal :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by have := neg_abs_le x.toReal; linarith)
  have h2 := two_rpow_ge_one_sub (show (0 : ℝ) ≤ 1 / 2 ^ 11 by norm_num)
  have h3 := (abs_le.mp hpe).1
  linarith

/-- One lane's `p` (line 82) and its fp8 rounding `q` (line 101): `p` finite, at most `1 + 2^-9`,
at least `−2^-125`; `q` finite; an attended lane's `q` within one rounding of `p`, a masked
lane's `q` zero and its `p` at most `2^-125`. -/
theorem lane_pq {P : Platform} (hexp : Rules.Num.Exp2F32 P) (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {t : ℕ} {m' : F32} {x p : Fin 64 → F32} {q : Fin 64 → F8E5}
    (hX : LaneX R t m' x) (hp : ∀ jj, P.math .exp2 .binary32 (x jj) (p jj))
    (hq : ∀ jj, P.cvt Format.binary32 Format.e5m2 .rn (p jj) (q jj)) (jj : Fin 64) :
    (p jj).isFinite = true ∧ |(p jj).toReal| ≤ 1 + 1 / 2 ^ 9 ∧ -(1 / 2 ^ 125) ≤ (p jj).toReal ∧
      (q jj).isFinite = true ∧
      (R.al (64 * t + jj) = true → |(q jj).toReal - (p jj).toReal| ≤ |(p jj).toReal| / 8 + 1 / 2 ^ 17) ∧
      (R.al (64 * t + jj) = false → (q jj).toReal = 0 ∧ |(p jj).toReal| ≤ 1 / 2 ^ 125) := by
  obtain ⟨hxf, hxle, hxa, hxm⟩ := hX jj
  obtain ⟨hpf, hpb, hpa, hpm⟩ := lane_p hexp hR hxf hxle hxa hxm (hp jj)
  obtain ⟨hqf, hqe⟩ := f8_cvt hcvt (hq jj) hpf (by linarith)
  obtain ⟨-, -, -, hqm⟩ := lane_p8 hcvt hpf hpb (fun h => (hpa h).1) hpm (hq jj)
  refine ⟨hpf, hpb, ?_, hqf, fun _ => hqe, fun h => ⟨by rw [toReal_def, hqm h]; simp, hpm h⟩⟩
  cases hal : R.al (64 * t + jj)
  · have := hpm hal; have := neg_abs_le (p jj).toReal; linarith
  · have h1 := (hpa hal).2
    have h2 : 0 ≤ wt R m'.toReal (64 * t + jj) * (1 - 1 / 2 ^ 22) / pER R := by
      have hw := wt_pos R m'.toReal (64 * t + jj)
      have hpe := pER_ge hR
      exact div_nonneg (mul_nonneg hw.le (by norm_num)) (by linarith)
    have : (0 : ℝ) < 1 / 2 ^ 126 := by positivity
    norm_num at h1 ⊢; linarith

/-- One lane's `q·v − r·p`, for a value `v` within `K` of `r` when the lane is attended. -/
theorem lane_E {q p v r K : ℝ} {al : Bool} (hK : 0 ≤ K)
    (hatt : al = true → |q - p| ≤ |p| / 8 + 1 / 2 ^ 17 ∧ |v - r| ≤ K)
    (hmask : al = false → q = 0 ∧ |p| ≤ 1 / 2 ^ 125) :
    |q * v - r * p| ≤ (9 / 8 * K + |r| / 8) * |p| + (K + |r|) / 2 ^ 17 ∧
      |q * v| ≤ (K + |r|) * (9 / 8 * |p| + 1 / 2 ^ 17) := by
  have hr := abs_nonneg r
  have hpa := abs_nonneg p
  cases al
  · obtain ⟨rfl, hp⟩ := hmask rfl
    simp only [zero_mul, zero_sub, abs_neg, abs_zero, abs_mul]
    constructor
    · have h1 : |r| * |p| ≤ |r| * (1 / 2 ^ 125) := mul_le_mul_of_nonneg_left hp hr
      have h2 : 0 ≤ (9 / 8 * K + |r| / 8) * |p| := by positivity
      have h3 : |r| * (1 / 2 ^ 125) ≤ (K + |r|) / 2 ^ 17 := by
        have : |r| * (1 / 2 ^ 125) ≤ |r| / 2 ^ 17 := by
          rw [mul_one_div]; exact div_le_div_of_nonneg_left hr (by norm_num) (by norm_num)
        have : |r| / 2 ^ 17 ≤ (K + |r|) / 2 ^ 17 := div_le_div_of_nonneg_right (by linarith) (by norm_num)
        linarith
      linarith
    · positivity
  · obtain ⟨hq, hv⟩ := hatt rfl
    have hqb : |q| ≤ 9 / 8 * |p| + 1 / 2 ^ 17 := by
      have := abs_sub_abs_le_abs_sub q p; linarith
    constructor
    · have e : q * v - r * p = q * (v - r) + r * (q - p) := by ring
      rw [e]
      have h1 : |q * (v - r)| ≤ (9 / 8 * |p| + 1 / 2 ^ 17) * K := by
        rw [abs_mul]; exact mul_le_mul hqb hv (abs_nonneg _) (by positivity)
      have h2 : |r * (q - p)| ≤ |r| * (|p| / 8 + 1 / 2 ^ 17) := by
        rw [abs_mul]; exact mul_le_mul_of_nonneg_left hq hr
      have h3 := abs_add_le (q * (v - r)) (r * (q - p))
      have e2 : (9 / 8 * |p| + 1 / 2 ^ 17) * K + |r| * (|p| / 8 + 1 / 2 ^ 17) =
          (9 / 8 * K + |r| / 8) * |p| + (K + |r|) / 2 ^ 17 := by ring
      linarith
    · rw [abs_mul]
      have hv' : |v| ≤ K + |r| := by have := abs_sub_abs_le_abs_sub v r; linarith
      calc |q| * |v| ≤ (9 / 8 * |p| + 1 / 2 ^ 17) * (K + |r|) :=
            mul_le_mul hqb hv' (abs_nonneg _) (by positivity)
        _ = (K + |r|) * (9 / 8 * |p| + 1 / 2 ^ 17) := by ring

/-! ## `magA`, `magD` -/

theorem iotaV_nonneg {K R : ℝ} (hK : 0 ≤ K) (hR : 0 ≤ R) : 0 ≤ iotaV K R :=
  le_min zero_le_one (by positivity)

theorem iotaV_le_one (K R : ℝ) : iotaV K R ≤ 1 := min_le_left _ _

theorem magA_nonneg {K R : ℝ} (hK : 0 ≤ K) (hR : 0 ≤ R) : ∀ t, 0 ≤ magA K R t
  | 0 => le_rfl
  | t + 1 => by
    show 0 ≤ magNext K R (magA K R t)
    have := iotaV_nonneg hK hR
    unfold magNext magCp
    rw [cS8_r, cP8_r]
    exact le_max_of_le_right (div_nonneg (by positivity) (by norm_num))

theorem magD_nonneg {K R : ℝ} (hK : 0 ≤ K) (hR : 0 ≤ R) : ∀ t, 0 ≤ magD K R t
  | 0 => le_rfl
  | t + 1 => by
    have h1 := magD_nonneg hK hR t
    have h2 := magA_nonneg hK hR (t + 1)
    have := iotaV_nonneg hK hR
    show 0 ≤ (1 + (cS8 : ℝ)) * (1 + 1 / 2 ^ 24) * (1 + 1 / 2 ^ 12) * magD K R t +
      (K + R) * (1 + (cS8 : ℝ)) / 2 ^ 11 + (cP8 : ℝ) * iotaV K R * (K + R + 1 / 2 ^ 13) / 2 ^ 8 +
      (magA K R (t + 1) + R + 1) / 2 ^ 80
    rw [cS8_r, cP8_r]
    positivity

/-! ## One block of `acc − r·l_i` -/

set_option maxHeartbeats 1000000 in
/-- **One block of `|acc − r·l_i| ≤ A·l_i + D`** (lines 95, 101–103, 106), in real numbers. `a`, `l`
are the old `acc` element and `l_i`, `α` is `alpha`, `a1` the element of `acc * alpha`, `a'` that
of `tl.dot(p, v, acc)`, `lij` the block's `tl.sum(p)`, `l'` the new `l_i`; `q`, `p`, `v` the lanes'
fp8 `p`, `p` and values. -/
theorem estep_real {a a1 a' l l' α lij r K A D : ℝ} {q p v : Fin 64 → ℝ}
    (hK : 0 ≤ K) (hA : 0 ≤ A) (hl0 : 0 ≤ l) (hl1 : l ≤ 2 ^ 44)
    (hα0 : -(1 / 2 ^ 126) ≤ α) (hp0 : ∀ jj, -(1 / 2 ^ 125) ≤ p jj)
    (hlane : ∀ jj, |q jj * v jj - r * p jj| ≤ (9 / 8 * K + |r| / 8) * |p jj| + (K + |r|) / 2 ^ 17)
    (hqv : ∀ jj, |q jj * v jj| ≤ (K + |r|) * (9 / 8 * |p jj| + 1 / 2 ^ 17))
    (hE : |α| * |a - r * l| ≤ |α| * A * l + (1 + 1 / 2 ^ 12) * D)
    (ha1 : |a1 - a * α| ≤ |a * α| / 2 ^ 24 + 1 / 2 ^ 150)
    (hdot : |a' - (a1 + ∑ jj, q jj * v jj)| ≤ (cS8 : ℝ) * (|a1| + ∑ jj, |q jj * v jj|) +
      (cP8 : ℝ) * iotaV K |r| * (9 / 8 * ∑ jj, |p jj| + 64 * (K + |r| + 1 / 2 ^ 13)) / 2 ^ 14 + (cA8 : ℝ))
    (hlij : |lij - ∑ jj, p jj| ≤ gamr * ∑ jj, |p jj| + 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24) ^ 63)
    (hl' : |l' - (l * α + lij)| ≤ 3 * (|l * α| + |lij|) / 2 ^ 24 + 4 / 2 ^ 150) :
    |a' - r * l'| ≤ magNext K |r| A * l' +
      ((1 + (cS8 : ℝ)) * (1 + 1 / 2 ^ 24) * (1 + 1 / 2 ^ 12) * D + (K + |r|) * (1 + (cS8 : ℝ)) / 2 ^ 11 +
        (cP8 : ℝ) * iotaV K |r| * (K + |r| + 1 / 2 ^ 13) / 2 ^ 8 + (magNext K |r| A + |r| + 1) / 2 ^ 80) := by
  have hM1 := le_max_left
    (((1 + (cS8 : ℝ)) * (1 + 1 / 2 ^ 24) * A + |r| * (4 / 2 ^ 24 + (cS8 : ℝ) * (1 + 1 / 2 ^ 24))) /
      (1 - 3 / 2 ^ 24))
    (magCp K |r| / (1 - 1 / 2 ^ 17 - 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24))
  have hM2 := le_max_right
    (((1 + (cS8 : ℝ)) * (1 + 1 / 2 ^ 24) * A + |r| * (4 / 2 ^ 24 + (cS8 : ℝ) * (1 + 1 / 2 ^ 24))) /
      (1 - 3 / 2 ^ 24))
    (magCp K |r| / (1 - 1 / 2 ^ 17 - 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24))
  rw [← magNext] at hM1 hM2
  unfold magCp at hM2
  simp only [cS8_r, cP8_r] at hM1 hM2 ⊢
  rw [cS8_r, cP8_r, cA8_r] at hdot
  have hι0 : 0 ≤ iotaV K |r| := iotaV_nonneg hK (abs_nonneg r)
  have hι1 := iotaV_le_one K |r|
  generalize iotaV K |r| = ι at hι0 hι1 hM2 hdot ⊢
  generalize magNext K |r| A = M at hM1 hM2 ⊢
  set R := |r| with hRd
  have hR0 : 0 ≤ R := abs_nonneg r
  -- `M` against its two parts
  have hM1' : (1 + 271425 / 2 ^ 24) * (1 + 1 / 2 ^ 24) * A + R * (4 / 2 ^ 24 + 271425 / 2 ^ 24 * (1 + 1 / 2 ^ 24)) ≤
      M * (1 - 3 / 2 ^ 24) := by
    rw [div_le_iff₀ (by norm_num)] at hM1; exact hM1
  have hM2' : 9 / 8 * K * (1 + 271425 / 2 ^ 24) +
      R * (9 / 8 * (271425 / 2 ^ 24) + 1 / 8 + 1 / 2 ^ 17 + 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24) +
      9 / 8 * (132128 / 2 ^ 24) * ι / 2 ^ 14 ≤
      M * (1 - 1 / 2 ^ 17 - 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24) := by
    rw [div_le_iff₀ (by norm_num)] at hM2; exact hM2
  have hM0 : 0 ≤ M := by
    have : 0 ≤ 9 / 8 * K * (1 + 271425 / 2 ^ 24) +
        R * (9 / 8 * (271425 / 2 ^ 24) + 1 / 8 + 1 / 2 ^ 17 + 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24) +
        9 / 8 * (132128 / 2 ^ 24) * ι / 2 ^ 14 := by positivity
    nlinarith
  -- the atoms
  set S := ∑ jj, |p jj| with hSd
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun _ _ => abs_nonneg _
  set X := l * |α| with hXd
  have hX0 : 0 ≤ X := mul_nonneg hl0 (abs_nonneg _)
  have hγ0 := gamr_nonneg
  have hγ1 := gamr_le
  have hη : 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24 : ℝ) ^ 63 ≤ 1 / 2 ^ 143 := by norm_num
  have hη0 : 0 ≤ 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24 : ℝ) ^ 63 := by positivity
  generalize 63 / 2 ^ 150 * (1 + 1 / 2 ^ 24 : ℝ) ^ 63 = η at hη hη0 hlij
  -- the lanes, summed
  have hSqv : ∑ jj, |q jj * v jj| ≤ (K + R) * (9 / 8) * S + (K + R) / 2 ^ 11 := by
    have := lane_sum_le (a := fun jj => |q jj * v jj|) (c := fun jj => |p jj|) (k := (K + R) * (9 / 8))
      (e := (K + R) / 2 ^ 17) fun jj => by have := hqv jj; nlinarith
    rw [← hSd] at this; norm_num at this ⊢; linarith
  have hT4 : |∑ jj, q jj * v jj - r * ∑ jj, p jj| ≤ (9 / 8 * K + R / 8) * S + (K + R) / 2 ^ 11 := by
    rw [Finset.mul_sum]
    have := lane_sum_err (a := fun jj => q jj * v jj) (b := fun jj => r * p jj) (c := fun jj => |p jj|)
      (k := 9 / 8 * K + R / 8) (e := (K + R) / 2 ^ 17) hlane
    rw [← hSd] at this; norm_num at this ⊢; linarith
  have hsp := sum_p_ge hp0
  rw [← hSd] at hsp
  have hsabs : |∑ jj, p jj| ≤ S := Finset.abs_sum_le_sum_abs _ _
  have hγS : gamr * S ≤ S / 2 ^ 17 := by
    have := mul_le_mul_of_nonneg_right hγ1 hS0; linarith
  -- `l_ij`
  have hlij1 := abs_le.mp hlij
  have hlijabs : |lij| ≤ (1 + 1 / 2 ^ 17) * S + 1 / 2 ^ 143 := by
    have := abs_sub_abs_le_abs_sub lij (∑ jj, p jj); linarith
  have hlijlow : S - 1 / 2 ^ 118 - S / 2 ^ 17 - 1 / 2 ^ 143 ≤ lij := by linarith
  -- `alpha`
  have hαabs : |α| ≤ α + 1 / 2 ^ 125 := by
    rcases le_or_gt 0 α with h | h
    · rw [abs_of_nonneg h]; linarith [show (0 : ℝ) < 1 / 2 ^ 125 by positivity]
    · rw [abs_of_neg h]; norm_num at hα0 ⊢; linarith
  have hlα : X ≤ l * α + 1 / 2 ^ 81 := by
    have h1 : l * |α| ≤ l * (α + 1 / 2 ^ 125) := mul_le_mul_of_nonneg_left hαabs hl0
    have h2 : l * (1 / 2 ^ 125) ≤ 2 ^ 44 * (1 / 2 ^ 125) := mul_le_mul_of_nonneg_right hl1 (by norm_num)
    rw [hXd]; norm_num at h2 ⊢; nlinarith
  have hlαabs : |l * α| = X := by rw [abs_mul, abs_of_nonneg hl0]
  -- `l'` from below
  have hl'low : X * (1 - 3 / 2 ^ 24) + S * (1 - 1 / 2 ^ 17 - 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24) ≤ l' + 1 / 2 ^ 80 := by
    have h1 := (abs_le.mp hl').1
    rw [hlαabs] at h1
    have h2 : 3 * (X + |lij|) / 2 ^ 24 ≤ 3 * (X + ((1 + 1 / 2 ^ 17) * S + 1 / 2 ^ 143)) / 2 ^ 24 :=
      div_le_div_of_nonneg_right (by linarith) (by norm_num)
    norm_num at h1 h2 hlα hlijlow ⊢
    linarith
  -- the old element
  have hEX : |α| * |a - r * l| ≤ A * X + (1 + 1 / 2 ^ 12) * D := by
    have : |α| * A * l = A * X := by rw [hXd]; ring
    linarith
  have haα : |a * α| ≤ R * X + A * X + (1 + 1 / 2 ^ 12) * D := by
    have h1 : |a| ≤ R * l + |a - r * l| := by
      have := abs_sub_abs_le_abs_sub a (r * l)
      rw [abs_mul r l, abs_of_nonneg hl0] at this; linarith
    have h2 : |a| * |α| ≤ (R * l + |a - r * l|) * |α| := mul_le_mul_of_nonneg_right h1 (abs_nonneg _)
    have e : (R * l + |a - r * l|) * |α| = R * X + |α| * |a - r * l| := by rw [hXd]; ring
    rw [abs_mul]; linarith
  have ha1b : |a1| ≤ (1 + 1 / 2 ^ 24) * |a * α| + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub a1 (a * α); linarith
  -- the six parts of `a' − r·l'`
  have hT3 : |α * (a - r * l)| ≤ A * X + (1 + 1 / 2 ^ 12) * D := by rw [abs_mul]; exact hEX
  have hT5 : |r * (lij - ∑ jj, p jj)| ≤ R * S / 2 ^ 17 + R / 2 ^ 143 := by
    rw [abs_mul]
    have h1 : |lij - ∑ jj, p jj| ≤ S / 2 ^ 17 + 1 / 2 ^ 143 := by linarith
    have := mul_le_mul_of_nonneg_left h1 hR0
    rw [← hRd]; linarith [show R * (S / 2 ^ 17 + 1 / 2 ^ 143) = R * S / 2 ^ 17 + R / 2 ^ 143 by ring]
  have hT6 : |r * (l' - (l * α + lij))| ≤ 3 * (R * X) / 2 ^ 24 + 3 * (1 + 1 / 2 ^ 17) * (R * S) / 2 ^ 24 +
      R / 2 ^ 140 := by
    rw [abs_mul]
    have hl'2 := hl'
    rw [hlαabs] at hl'2
    have h1 : |l' - (l * α + lij)| ≤ 3 * (X + ((1 + 1 / 2 ^ 17) * S + 1 / 2 ^ 143)) / 2 ^ 24 + 4 / 2 ^ 150 := by
      have : 3 * (X + |lij|) / 2 ^ 24 ≤ 3 * (X + ((1 + 1 / 2 ^ 17) * S + 1 / 2 ^ 143)) / 2 ^ 24 :=
        div_le_div_of_nonneg_right (by linarith) (by norm_num)
      linarith
    have h2 := mul_le_mul_of_nonneg_left h1 hR0
    have e2 : R * (3 * (X + ((1 + 1 / 2 ^ 17) * S + 1 / 2 ^ 143)) / 2 ^ 24 + 4 / 2 ^ 150) =
        3 * (R * X) / 2 ^ 24 + 3 * (1 + 1 / 2 ^ 17) * (R * S) / 2 ^ 24 + R * (3 / 2 ^ 167 + 4 / 2 ^ 150) := by ring
    have h3 : R * (3 / 2 ^ 167 + 4 / 2 ^ 150) ≤ R / 2 ^ 140 := by
      have : R * (3 / 2 ^ 167 + 4 / 2 ^ 150) ≤ R * (1 / 2 ^ 140) := mul_le_mul_of_nonneg_left (by norm_num) hR0
      linarith [show R * (1 / 2 ^ 140) = R / 2 ^ 140 by ring]
    rw [← hRd]; linarith
  have hT1 := hdot
  have hT2 := ha1
  -- combine
  have hsum : |a' - r * l'| ≤ ((1 + 271425 / 2 ^ 24) * (1 + 1 / 2 ^ 24) * A +
      R * (4 / 2 ^ 24 + 271425 / 2 ^ 24 * (1 + 1 / 2 ^ 24))) * X +
      (9 / 8 * K * (1 + 271425 / 2 ^ 24) +
        R * (9 / 8 * (271425 / 2 ^ 24) + 1 / 8 + 1 / 2 ^ 17 + 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24) +
        9 / 8 * (132128 / 2 ^ 24) * ι / 2 ^ 14) * S +
      ((1 + 271425 / 2 ^ 24) * (1 + 1 / 2 ^ 24) * (1 + 1 / 2 ^ 12) * D + (K + R) * (1 + 271425 / 2 ^ 24) / 2 ^ 11 +
        132128 / 2 ^ 24 * ι * (K + R + 1 / 2 ^ 13) / 2 ^ 8 + (R + 1) / 2 ^ 120) := by
    obtain ⟨u1, v1⟩ := abs_le.mp hT1
    obtain ⟨u2, v2⟩ := abs_le.mp hT2
    obtain ⟨u3, v3⟩ := abs_le.mp hT3
    obtain ⟨u4, v4⟩ := abs_le.mp hT4
    obtain ⟨u5, v5⟩ := abs_le.mp hT5
    obtain ⟨u6, v6⟩ := abs_le.mp hT6
    rw [abs_le]
    constructor <;> linarith
  -- `l'` bounds the parts
  have hfin : ((1 + 271425 / 2 ^ 24) * (1 + 1 / 2 ^ 24) * A +
      R * (4 / 2 ^ 24 + 271425 / 2 ^ 24 * (1 + 1 / 2 ^ 24))) * X +
      (9 / 8 * K * (1 + 271425 / 2 ^ 24) +
        R * (9 / 8 * (271425 / 2 ^ 24) + 1 / 8 + 1 / 2 ^ 17 + 3 * (1 + 1 / 2 ^ 17) / 2 ^ 24) +
        9 / 8 * (132128 / 2 ^ 24) * ι / 2 ^ 14) * S ≤
      M * l' + M / 2 ^ 80 := by
    have h1 := mul_le_mul_of_nonneg_right hM1' hX0
    have h2 := mul_le_mul_of_nonneg_right hM2' hS0
    have h3 := mul_le_mul_of_nonneg_left hl'low hM0
    nlinarith
  have : M / 2 ^ 80 + (R + 1) / 2 ^ 120 ≤ (M + R + 1) / 2 ^ 80 := by
    have : (R + 1) / 2 ^ 120 ≤ (R + 1) / 2 ^ 80 :=
      div_le_div_of_nonneg_left (by linarith) (by norm_num) (by norm_num)
    linarith [show (M + R + 1) / 2 ^ 80 = M / 2 ^ 80 + (R + 1) / 2 ^ 80 by ring]
  linarith

/-! ## The end of a row -/

/-- The exact attention of values at most `57344` is at most `57344`. -/
theorem ref_le {R : RowCtx} {t : ℕ} {m : F32} (hm : MInv R t m) {ve : ℕ → ℝ} (hve : ∀ j, |ve j| ≤ 57344) :
    |Sx R ve t / psum R.al t (ew R)| ≤ 57344 := by
  have hW0 : 0 < psum R.al t (ew R) := psum_ew_pos hm
  rw [abs_div, abs_of_pos hW0, div_le_iff₀ hW0]
  refine (abs_psum_le _ _ _).trans ?_
  rw [← psum_mul]
  exact psum_le fun j _ _ => by
    rw [abs_mul, abs_of_pos (ew_pos R j), mul_comm (57344 : ℝ)]
    exact mul_le_mul_of_nonneg_left (hve j) (ew_pos R j).le

/-- **The output element from the second bound** (lines 242 and 245): with `|acc − r·l_i| ≤
A·l_i + D` and `l_i ≥ 1/2`, `(acc / l_i).to(float8e5)` is within `9/8·((A + 2D)(1 + 2^-22) + 2^-22·|r| +
2^-148) + |r|/8 + 2^-17` of `r`. -/
theorem final_v {P : Platform} (hdivF : Rules.Num.DivFull P) (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {a l o' : F32} {o8 : F8E5} {r A D : ℝ} (haf : a.isFinite = true) (hlf : l.isFinite = true)
    (hl : 1 / 2 ≤ l.toReal) (hab : |a.toReal| ≤ 2 ^ 58) (hr : |r| ≤ 57344) (hD : 0 ≤ D)
    (hE : |a.toReal - r * l.toReal| ≤ A * l.toReal + D)
    (hdv : P.div .binary32 a l o') (hc : P.cvt Format.binary32 Format.e5m2 .rn o' o8) :
    o8.isFinite = true ∧ |o8.toReal - r| ≤
      9 / 8 * ((A + 2 * D) * (1 + 1 / 2 ^ 22) + |r| / 2 ^ 22 + 1 / 2 ^ 148) + |r| / 8 + 1 / 2 ^ 17 := by
  have hl0 : 0 < l.toReal := by linarith
  have hal : |a.toReal / l.toReal| ≤ 2 ^ 59 := by
    rw [abs_div, abs_of_pos hl0, div_le_iff₀ hl0]; nlinarith
  have hal' : |a.toRat / l.toRat| ≤ 2 ^ 126 := by
    have : ((|a.toRat / l.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast; rw [← toReal_def, ← toReal_def]; linarith
    exact_mod_cast this
  have hl0' : l.toRat ≠ 0 := by
    intro h; rw [toReal_def, h] at hl0; norm_num at hl0
  obtain ⟨ho'f, ho'e⟩ := hdivF a l o' hdv haf hlf hl0' hal'
  have ho'r : |o'.toReal - a.toReal / l.toReal| ≤ 1 / 2 ^ 22 * |a.toReal / l.toReal| + 1 / 2 ^ 148 := by
    have := (Rat.cast_le (K := ℝ)).mpr ho'e
    push_cast at this
    rw [show (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 by norm_num, show (2 : ℝ) ^ (-148 : ℤ) = 1 / 2 ^ 148 by norm_num] at this
    simp only [← toReal_def] at this
    exact this
  -- `acc / l_i` against `r`
  have hq : |a.toReal / l.toReal - r| ≤ A + 2 * D := by
    have e : a.toReal / l.toReal - r = (a.toReal - r * l.toReal) / l.toReal := by field_simp
    rw [e, abs_div, abs_of_pos hl0, div_le_iff₀ hl0]
    have : D ≤ 2 * D * l.toReal := by nlinarith
    nlinarith
  have hd1 : |a.toReal / l.toReal| ≤ |r| + (A + 2 * D) := by
    have := abs_sub_abs_le_abs_sub (a.toReal / l.toReal) r; linarith
  have hd2 : |o'.toReal - r| ≤ (A + 2 * D) * (1 + 1 / 2 ^ 22) + |r| / 2 ^ 22 + 1 / 2 ^ 148 := by
    have := abs_sub_le o'.toReal (a.toReal / l.toReal) r
    have := mul_le_mul_of_nonneg_left hd1 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 22)
    nlinarith
  -- line 245
  obtain ⟨ho8f, ho8e⟩ := f8_cvt_sat hcvt hc ho'f
  obtain ⟨hc1, -⟩ := clamp_err (x := o'.toReal) hr
  refine ⟨ho8f, ?_⟩
  have hc3 : |max (-57344) (min o'.toReal 57344)| ≤ |r| + |o'.toReal - r| := by
    have := abs_sub_abs_le_abs_sub (max (-57344) (min o'.toReal 57344)) r; linarith
  have := abs_sub_le o8.toReal (max (-57344) (min o'.toReal 57344)) r
  linarith

end Kernels.AttnFwdE5m2
