import Claim
import TVA.Proof
import Proof.Setup
import Proof.Cast
import Proof.ExpNeg
import Proof.Fused

/-!
# The softmax normaliser, and `1 / den` against it

For one token `t`, write `c k` for the computed soft-capped logit of expert `k`
(`capped.get1 k`). The block forms `diffs = capped - c i0` (a fused subtraction), sums
`exps = exp diffs` (`tl.sum`, no fusion), and then inverts the total:
`qinv = 1 / den`.

The reference denominator is

  `D = sum over k < 8 of exp (c k - c i0)`   (`wD capped i0`)

and the claim proved here is the scale-free form

  `|qinv * D - 1| <= 3/200000`

together with finiteness of `den`, `den != 0`, and finiteness of `qinv`. The scale-free form
is what the second weight needs; the first weight itself is compared with `1 / D` in the
spine (with `epsq = 1.5e-5 < wRel0 = 2^-14` this is immediate from the same inequality).

## Numbers

Throughout `B = 30 * (1 + 2^-20)`, `u = 2^-24 = 5.9604645e-8`, `eta = 2^-150`.
`hsmall` gives `|c k| <= B` for every `k`, and `hle0` gives `c k <= c i0`.

1. `d k = c k - c i0 <= 0`, `|d k| <= 2 B = 60.000057`.
2. Fused subtraction (`hdiffs`): `|d' k - d k| <= u * 3B + 2u^2 B + 4 eta <= 5.37e-6`
   (`δd`), for the sharpened fused bound `u(|x|+|y|+max |x| |y|) + 2u^2 max + 4 eta`,
   which is what `AddRel` with `fusion = true` and one exact operand allows.
   Use `Proof.Fused`'s sharpened lemma `addrel_err_fusion` (root namespace, proved in this
   project) rather than `FloatV.AddRel.err`, whose constant `3u(|x|+|y|)` would be tight.
3. `exps` (`hexps`, `ExpF32` with `X = |d' k| <= 61`): with `rho = 2^-22 + 2^-23 |d' k|`
   (`<= 7.4e-6`),
   `|e' k - exp (d k)| <= (rho * exp δd + (exp δd - 1)) * exp (d k) + 2^-140`,
   and `exp δd - 1 <= δd / (1 - δd) <= 5.38e-6`;
   so summing, with `D = sum exp (d k)` and `D <= 8`,
   `sum |e' k - exp (d k)| <= 1.279e-5 * D + 8 * 2^-140`.
4. The fused sum (`hden`, `sumc_err_depth` with `fusion = false`, tree depth at most
   `clog 2 8 = 3` from `SumDepth`): `|den - sum e' k| <= ((1+3u)^3 - 1) A + 28 eta (1+3u)^3`
   with `A = sum |e' k| <= D (1 + 1.279e-5)` and `(1+3u)^3 - 1 <= 5.3643e-7`.
   Hence `|den - D| <= 1.335e-5 * D`; take `epsD = 7/500000 = 1.4e-5` (a 5 per cent
   margin: do not loosen any of the intermediate constants).
5. `qinv` (`hinv`, `DivFull` with `x = 1`, `y = den`; `den >= (1 - epsD) D >= 1 - epsD`
   so `|1/den| <= 2^126` holds): `|qinv - 1/den| <= 2^-22 |1/den| + 2^-148`.
   Then
   `|qinv * D - 1| <= 2^-22 / (1 - epsD) + epsD / (1 - epsD) + 2^-148 * D`
   `<= 2.384e-7 + 1.40003e-5 + 2^-145 = 1.4239e-5 <= 1.5e-5`.
-/

namespace Claims.MoeRouter

open TVA

open Block

variable {d : Inst}

/-- The computed soft-capped logit of expert `e`, as a total function. -/
noncomputable def wC (capped : Tensor [8 - 0] F32) (e : Nat) : ℝ :=
  if h : e < 8 - 0 then (capped.get1 ⟨e, h⟩).toReal else 0

/-- The reference softmax denominator `sum over k < 8 of exp (C k - C i0)`. -/
noncomputable def wD (capped : Tensor [8 - 0] F32) (i0 : Nat) : ℝ :=
  ∑ k ∈ Finset.range 8, Real.exp (wC capped k - wC capped i0)

/-! ## Helpers -/

/-- The magnitude bound on the soft-capped logits, in `ℚ`. -/
private def Bq : ℚ := 30 * (1 + 2 ^ (-20 : ℤ))

private theorem Bq_cast : ((Bq : ℚ) : ℝ) = 30 * (1 + (2 : ℝ) ^ (-20 : ℤ)) := by
  unfold Bq
  push_cast
  norm_num

private theorem c32_one_isFinite : (c32 0x3FF0000000000000 : F32).isFinite = true := by
  have hz : (FloatV.ofPat 0x3FF0000000000000 : F64).isZero = false := by decide
  have hn : (FloatV.ofPat 0x3FF0000000000000 : F64).isNaN = false := by decide
  have hfin : (FloatV.ofPat 0x3FF0000000000000 : F64).isFinite = true := by decide
  have hval : (FloatV.ofPat 0x3FF0000000000000 : F64).toRat = 1 := by
    rw [FloatV.toRat_ofPat (by norm_num [Format.binary64, Format.width])]
    simp only [Format.toRat, Format.mag, Format.magVal,
      Format.pow2_eq, Format.binary64, Format.sign,
      Format.signBit, Format.bias, Format.emin]
    norm_num
  have hc : (c32 0x3FF0000000000000 : F32) = FloatV.cvt (FloatV.ofPat 0x3FF0000000000000 : F64) := by
    unfold c32 FloatV.litF32 FloatV.hostF32
    simp only [hz, hn, Bool.false_eq_true, ite_false]
  have hmax : |(FloatV.ofPat 0x3FF0000000000000 : F64).toRat| ≤ Format.binary32.maxVal := by
    rw [hval, Format.binary32_maxVal]; norm_num
  obtain ⟨hcf, herr⟩ := FloatV.cvt_err (f := Format.binary64)
    (g := Format.binary32) (by decide) hfin hmax
  rw [hc]
  exact hcf

private theorem c32_one_err : |(c32 0x3FF0000000000000 : F32).toReal - 1| ≤ u32 + eta32 := by
  have hz : (FloatV.ofPat 0x3FF0000000000000 : F64).isZero = false := by decide
  have hn : (FloatV.ofPat 0x3FF0000000000000 : F64).isNaN = false := by decide
  have hfin : (FloatV.ofPat 0x3FF0000000000000 : F64).isFinite = true := by decide
  have hval : (FloatV.ofPat 0x3FF0000000000000 : F64).toRat = 1 := by
    rw [FloatV.toRat_ofPat (by norm_num [Format.binary64, Format.width])]
    simp only [Format.toRat, Format.mag, Format.magVal,
      Format.pow2_eq, Format.binary64, Format.sign,
      Format.signBit, Format.bias, Format.emin]
    norm_num
  have hc : (c32 0x3FF0000000000000 : F32) = FloatV.cvt (FloatV.ofPat 0x3FF0000000000000 : F64) := by
    unfold c32 FloatV.litF32 FloatV.hostF32
    simp only [hz, hn, Bool.false_eq_true, ite_false]
  have hmax : |(FloatV.ofPat 0x3FF0000000000000 : F64).toRat| ≤ Format.binary32.maxVal := by
    rw [hval, Format.binary32_maxVal]; norm_num
  obtain ⟨hcf, herr⟩ := FloatV.cvt_err (f := Format.binary64)
    (g := Format.binary32) (by decide) hfin hmax
  rw [hc, FloatV.toReal]
  have h3 : |((FloatV.cvt (FloatV.ofPat 0x3FF0000000000000 : F64) : F32).toRat : ℝ) - (1 : ℝ)|
      ≤ u32 + eta32 := by
    have hcast : ((|(FloatV.cvt (FloatV.ofPat 0x3FF0000000000000 : F64) : F32).toRat -
        (FloatV.ofPat 0x3FF0000000000000 : F64).toRat| : ℚ) : ℝ) ≤
        ((|(FloatV.ofPat 0x3FF0000000000000 : F64).toRat| * Format.binary32.u +
          Format.binary32.eta : ℚ) : ℝ) := by
      exact_mod_cast herr
    have h4 : ((|(1 : ℚ)| * Format.binary32.u +
        Format.binary32.eta : ℚ) : ℝ) = u32 + eta32 := by
      rw [Format.binary32_u, Format.binary32_eta]
      norm_num [u32, eta32]
    rw [hval] at hcast
    rw [h4] at hcast
    simpa using hcast
  exact h3

/-- `Tree.errBound`, cast to `ℝ`. -/
private theorem errBound_cast (δ η : ℚ) (d N : Nat) (A : ℚ) :
    ((Tree.errBound δ η d N A : ℚ) : ℝ) =
      ((1 + (δ : ℝ)) ^ d - 1) * (A : ℝ) + (η : ℝ) * (N : ℝ) * (1 + (δ : ℝ)) ^ d := by
  show ((((1 + δ) ^ d - 1) * A + η * (N : ℚ) * (1 + δ) ^ d : ℚ) : ℝ) = _
  push_cast
  ring

private theorem list_sum_range_congr_real (g : Nat → ℝ) (n : Nat) :
    ((List.range n).map g).sum = ∑ k ∈ Finset.range n, g k := by
  induction n with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]
    simp

private theorem list_sum_range_congr_rat (g : Nat → ℚ) (n : Nat) :
    ((List.range n).map g).sum = ∑ k ∈ Finset.range n, g k := by
  induction n with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]
    simp

private theorem cast_list_sum_rat (g : Nat → ℚ) (n : Nat) :
    ((((List.range n).map g).sum : ℚ) : ℝ) = ((List.range n).map fun i => (g i : ℝ)).sum := by
  rw [Rat.cast_list_sum, List.map_map]
  rfl

private theorem exp_le_inv_one_sub {z : ℝ} (h0 : 0 ≤ z) (h1 : z ≤ 1 / 2) :
    Real.exp z ≤ 1 / (1 - z) := by
  have hz : |z| ≤ 1 := by rw [abs_of_nonneg h0]; linarith
  have h2 := Real.abs_exp_sub_one_sub_id_le (x := z) hz
  have h3 : Real.exp z - 1 - z ≤ z ^ 2 := (abs_le.mp h2).2
  have h4 : z + z ^ 2 ≤ z / (1 - z) := by
    rw [le_div_iff₀ (by linarith)]
    nlinarith
  have h5 : Real.exp z ≤ 1 + z + z ^ 2 := by linarith
  have h6 : 1 + z / (1 - z) = 1 / (1 - z) := by
    have hz1 : (1 : ℝ) - z ≠ 0 := by linarith
    calc 1 + z / (1 - z) = (1 - z) / (1 - z) + z / (1 - z) := by rw [div_self hz1]
      _ = ((1 - z) + z) / (1 - z) := by rw [add_div]
      _ = 1 / (1 - z) := by norm_num
  linarith

private theorem exp_abs_le_inv {z : ℝ} (hz : |z| ≤ 1 / 2) :
    Real.exp z ≤ 1 / (1 - |z|) := by
  have h1 : Real.exp z ≤ Real.exp |z| := Real.exp_le_exp.mpr (le_abs_self z)
  have h2 : Real.exp |z| ≤ 1 / (1 - |z|) := exp_le_inv_one_sub (abs_nonneg z) hz
  linarith

private theorem abs_exp_sub_one_le_div {z : ℝ} (hz : |z| ≤ 1 / 2) :
    |Real.exp z - 1| ≤ |z| / (1 - |z|) := by
  have h1 : |z| ≤ 1 := by linarith
  have h2 := Real.abs_exp_sub_one_sub_id_le (x := z) h1
  have h3 : |Real.exp z - 1| ≤ z ^ 2 + |z| := by
    calc |Real.exp z - 1| = |(Real.exp z - 1 - z) + z| := by ring_nf
      _ ≤ |Real.exp z - 1 - z| + |z| := abs_add_le _ _
      _ ≤ z ^ 2 + |z| := by linarith [abs_nonneg z]
  have h4 : z ^ 2 = |z| ^ 2 := (sq_abs z).symm
  have h5 : |z| + |z| ^ 2 ≤ |z| / (1 - |z|) := by
    rw [le_div_iff₀ (by linarith)]
    nlinarith [abs_nonneg z, sq_nonneg |z|, mul_nonneg (sq_nonneg |z|) (abs_nonneg z)]
  linarith

private theorem div_one_sub_mono {a b : ℝ} (hab : a ≤ b) (hb : b < 1) :
    a / (1 - a) ≤ b / (1 - b) := by
  have hb1 : (0 : ℝ) < 1 - b := by linarith
  have ha1 : (0 : ℝ) < 1 - a := by linarith
  rw [div_le_div_iff₀ ha1 hb1]
  nlinarith

/-- Transfer an error bound for `exp` from a computed argument `r` to the exact `dk`. -/
private theorem exp_err_transfer {e r dk ρ ω c : ℝ}
    (hρ : 0 ≤ ρ) (_hω0 : 0 ≤ ω) (hω : ω ≤ 1 / 2)
    (hr : |r - dk| ≤ ω)
    (h : |e - Real.exp r| ≤ ρ * Real.exp r + c) :
    |e - Real.exp dk| ≤ (ρ / (1 - ω) + ω / (1 - ω)) * Real.exp dk + c := by
  have hz : |r - dk| ≤ 1 / 2 := hr.trans hω
  have hexpz : Real.exp (r - dk) ≤ 1 / (1 - |r - dk|) := exp_abs_le_inv hz
  have hzω : 1 / (1 - |r - dk|) ≤ 1 / (1 - ω) := by
    have h1 : (0 : ℝ) < 1 - ω := by linarith
    have h2 : (1 : ℝ) - ω ≤ 1 - |r - dk| := by linarith [abs_nonneg (r - dk)]
    exact one_div_le_one_div_of_le h1 h2
  have hexpz1 : |Real.exp (r - dk) - 1| ≤ ω / (1 - ω) := by
    have h1 := abs_exp_sub_one_le_div hz
    have h2 : |r - dk| / (1 - |r - dk|) ≤ ω / (1 - ω) :=
      div_one_sub_mono hr (by linarith)
    linarith
  have hexpr : Real.exp r = Real.exp dk * Real.exp (r - dk) := by
    rw [← Real.exp_add]; ring_nf
  have hEdk : 0 < Real.exp dk := Real.exp_pos dk
  have h2 : |Real.exp r - Real.exp dk| = Real.exp dk * |Real.exp (r - dk) - 1| := by
    rw [hexpr]
    have hEq : Real.exp dk * Real.exp (r - dk) - Real.exp dk
        = Real.exp dk * (Real.exp (r - dk) - 1) := by ring
    rw [hEq, abs_mul, abs_of_pos hEdk]
  have hle1 : Real.exp r ≤ Real.exp dk * (1 / (1 - ω)) := by
    rw [hexpr]
    exact mul_le_mul_of_nonneg_left (hexpz.trans hzω) hEdk.le
  have hle2 : ρ * Real.exp r ≤ ρ * (Real.exp dk * (1 / (1 - ω))) :=
    mul_le_mul_of_nonneg_left hle1 hρ
  have hle3 : Real.exp dk * |Real.exp (r - dk) - 1| ≤ Real.exp dk * (ω / (1 - ω)) :=
    mul_le_mul_of_nonneg_left hexpz1 hEdk.le
  have htri : |e - Real.exp dk| ≤ |e - Real.exp r| + |Real.exp r - Real.exp dk| := by
    calc |e - Real.exp dk| = |(e - Real.exp r) + (Real.exp r - Real.exp dk)| := by ring_nf
      _ ≤ _ := abs_add_le _ _
  have hfin : (ρ / (1 - ω) + ω / (1 - ω)) * Real.exp dk
      = ρ * (Real.exp dk * (1 / (1 - ω))) + Real.exp dk * (ω / (1 - ω)) := by
    have h1 : (1 : ℝ) - ω ≠ 0 := by linarith
    field_simp
  rw [hfin]
  linarith [htri, h, h2, hle2, hle3]

private theorem final_assembly {P X Dn D : ℝ} (hdenpos : 0 < Dn)
    (hdenD : |Dn - D| ≤ (7 / 500000) * D)
    (hDge1 : 1 ≤ D) (hDle8 : D ≤ 8)
    (hqerr : |P - X / Dn| ≤ (2 : ℝ) ^ (-22 : ℤ) * |X / Dn| + (2 : ℝ) ^ (-148 : ℤ))
    (hXabs : |X| ≤ 1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ))
    (hX1 : |X - 1| ≤ (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) :
    |P * D - 1| ≤ 3 / 200000 := by
  have hDpos : (0 : ℝ) < D := by linarith
  have heps1 : (0 : ℝ) < 1 - 7 / 500000 := by norm_num
  have hDnonneg : (0 : ℝ) ≤ D := le_of_lt hDpos
  have hdenlower : (1 - 7 / 500000) * D ≤ Dn := by
    have h := (abs_le.mp hdenD).1
    linarith
  have hDn_abs : (1 - 7 / 500000) * D ≤ |Dn| := by
    rw [abs_of_pos hdenpos]
    exact hdenlower
  have hdenpos' : (0 : ℝ) < (1 - 7 / 500000) * D := mul_pos heps1 hDpos
  have hfracD : |X / Dn| * D ≤ (100002 / 100000 : ℝ) := by
    have hA : |X| / |Dn| ≤ |X| / ((1 - 7 / 500000) * D) :=
      div_le_div₀ (abs_nonneg _) (le_refl _) hdenpos' hDn_abs
    have hB : |X| / ((1 - 7 / 500000) * D)
        ≤ (1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) / ((1 - 7 / 500000) * D) :=
      div_le_div_of_nonneg_right hXabs (le_of_lt hdenpos')
    have hC : (1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) / ((1 - 7 / 500000) * D) * D
        = (1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) / (1 - 7 / 500000) := by
      field_simp
    rw [abs_div]
    calc |X| / |Dn| * D
        ≤ |X| / ((1 - 7 / 500000) * D) * D := mul_le_mul_of_nonneg_right hA hDnonneg
      _ ≤ (1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) / ((1 - 7 / 500000) * D) * D :=
          mul_le_mul_of_nonneg_right hB hDnonneg
      _ = (1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) / (1 - 7 / 500000) := hC
      _ ≤ (100002 / 100000 : ℝ) := by
          rw [div_le_iff₀ heps1]
          norm_num
  have hdecomp : |P * D - 1| ≤ |P - X / Dn| * D + |X / Dn * D - 1| := by
    have hEq : P * D - 1 = (P - X / Dn) * D + (X / Dn * D - 1) := by ring
    rw [hEq]
    calc |(P - X / Dn) * D + (X / Dn * D - 1)|
        ≤ |(P - X / Dn) * D| + |X / Dn * D - 1| := abs_add_le _ _
      _ = |P - X / Dn| * D + |X / Dn * D - 1| := by rw [abs_mul, abs_of_nonneg hDnonneg]
  have hT1 : |P - X / Dn| * D
      ≤ (2 : ℝ) ^ (-22 : ℤ) * (100002 / 100000) + (2 : ℝ) ^ (-148 : ℤ) * 8 := by
    calc |P - X / Dn| * D
        ≤ ((2 : ℝ) ^ (-22 : ℤ) * |X / Dn| + (2 : ℝ) ^ (-148 : ℤ)) * D :=
          mul_le_mul_of_nonneg_right hqerr hDnonneg
      _ = (2 : ℝ) ^ (-22 : ℤ) * (|X / Dn| * D) + (2 : ℝ) ^ (-148 : ℤ) * D := by ring
      _ ≤ (2 : ℝ) ^ (-22 : ℤ) * (100002 / 100000) + (2 : ℝ) ^ (-148 : ℤ) * 8 :=
          add_le_add (mul_le_mul_of_nonneg_left hfracD (by positivity))
            (mul_le_mul_of_nonneg_left hDle8 (by positivity))
  have hT2 : |X / Dn * D - 1|
      ≤ (1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) * ((7 / 500000) / (1 - 7 / 500000))
        + ((2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) := by
    have hEq : X / Dn * D - 1 = X * (D / Dn - 1) + (X - 1) := by
      field_simp
      ring
    rw [hEq]
    have hq : |D / Dn - 1| ≤ (7 / 500000) / (1 - 7 / 500000) := by
      have hEq2 : D / Dn - 1 = (D - Dn) / Dn := by field_simp
      rw [hEq2, abs_div, abs_of_pos hdenpos]
      rw [div_le_div_iff₀ hdenpos heps1]
      have h1 : |D - Dn| ≤ (7 / 500000) * D := by
        rw [abs_sub_comm]
        exact hdenD
      have heps0 : (0 : ℝ) ≤ 7 / 500000 := by norm_num
      nlinarith [h1, hdenlower, heps0, heps1]
    calc |X * (D / Dn - 1) + (X - 1)|
        ≤ |X * (D / Dn - 1)| + |X - 1| := abs_add_le _ _
      _ = |X| * |D / Dn - 1| + |X - 1| := by rw [abs_mul]
      _ ≤ (1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) * ((7 / 500000) / (1 - 7 / 500000))
            + ((2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) := by
          have hmul := mul_le_mul hXabs hq (abs_nonneg _)
            (by positivity : (0 : ℝ) ≤ 1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ))
          exact add_le_add hmul hX1
  have hnum : ((2 : ℝ) ^ (-22 : ℤ) * (100002 / 100000) + (2 : ℝ) ^ (-148 : ℤ) * 8)
      + ((1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) * ((7 / 500000) / (1 - 7 / 500000))
          + ((2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ))) ≤ 3 / 200000 := by norm_num
  linarith [hdecomp, hT1, hT2, hnum]

/-- `den = tl.sum (exp (capped - c i0))` is within `1.4e-5` (relatively) of `wD`, and
`qinv = 1 / den` satisfies `|qinv * wD - 1| <= 3/200000`. -/
theorem den_qinv_bound {d : Inst} {P : Platform} {t : Nat} (_ht : t < d.bs)
    {capped : Tensor [8 - 0] F32}
    (hfin : ∀ e : Fin (8 - 0), (capped.get1 e).isFinite = true)
    (hsmall : ∀ e : Fin (8 - 0), |(capped.get1 e).toReal| ≤ 30 * (1 + 2 ^ (-20 : ℤ)))
    (hsum : Device.SumDepth P)
    (hexpacc : Device.ExpF32 P)
    (hdivfull : Device.DivFull P)
    {i0 : Nat} (hi0 : i0 < 8 - 0)
    (hle0 : ∀ e : Fin (8 - 0), (capped.get1 e).toReal ≤ (capped.get1 ⟨i0, hi0⟩).toReal)
    {top1v : F32} (htopfin : top1v.isFinite = true)
    (htopval : top1v.toReal = (capped.get1 ⟨i0, hi0⟩).toReal)
    {diffs exps : Tensor [8 - 0] F32}
    (hdiffs : ∀ j : Idx [8 - 0],
      AddRel ((launch d).env P).fusion (capped.get j) ((bc top1v).get j).neg (diffs.get j))
    (hexps : ∀ j : Idx [8 - 0],
      P.math MathFn.exp Format.binary32 (diffs.get j) (exps.get j))
    {den : F32}
    (hden : ∃ tr, Tree.Covers (8 - 0) tr ∧ tr.depth ≤ tl.sumBound P (8 - 0) ∧
      Tree.Eval (AddRel false)
        (fun i => if h : i < 8 - 0 then exps.get1 ⟨i, h⟩ else default) tr den)
    {qinv : Tensor [] F32}
    (hinv : ∀ i : Idx [], P.div Format.binary32
      ((Tensor.scalar (c32 0x3FF0000000000000)).get i) ((Tensor.scalar den).get i)
      (qinv.get i)) :
    den.isFinite = true ∧ den.toReal ≠ 0 ∧ qinv.item.isFinite = true ∧
      |qinv.item.toReal * wD capped i0 - 1| ≤ (3 : ℝ) / 200000 := by
  have hfus : ((launch d).env P).fusion = true := rfl
  have hclog : Nat.clog 2 (8 - 0) = 3 := by
    have h8 : (8 - 0 : Nat) = 2 ^ 3 := by norm_num
    rw [h8]
    exact Nat.clog_pow 2 3 (by norm_num)
  set c0 : ℝ := (capped.get1 ⟨i0, hi0⟩).toReal with hc0
  have htopval' : top1v.toReal = c0 := htopval.trans hc0.symm
  have hcle : ∀ e : Fin (8 - 0), (capped.get1 e).toReal ≤ c0 := by
    intro e
    simpa [hc0] using hle0 e
  have hsmallq : ∀ e : Fin (8 - 0), |(capped.get1 e).toRat| ≤ Bq := by
    intro e
    apply Rat.cast_le (K := ℝ).mp
    rw [Rat.cast_abs]
    show |((capped.get1 e).toRat : ℝ)| ≤ ((Bq : ℚ) : ℝ)
    rw [Bq_cast]
    exact hsmall e
  have htopq : |top1v.toRat| ≤ Bq := by
    apply Rat.cast_le (K := ℝ).mp
    rw [Rat.cast_abs]
    show |((top1v.toRat : ℚ) : ℝ)| ≤ ((Bq : ℚ) : ℝ)
    rw [Bq_cast]
    have h : |top1v.toReal| ≤ 30 * (1 + (2 : ℝ) ^ (-20 : ℤ)) := by
      rw [htopval']
      exact hsmall ⟨i0, hi0⟩
    simpa [FloatV.toReal] using h
  have htopneg : (FloatV.neg top1v).isFinite = true := by
    rw [FloatV.isFinite_neg]
    exact htopfin
  -- the `exp 61` room estimate
  have hexp61 : Real.exp 61 ≤ (2 : ℝ) ^ 97 := by
    have h3 : Real.exp 61 = Real.exp 1 ^ 61 := by
      rw [show (61 : ℝ) = ((61 : ℕ) : ℝ) * 1 by norm_num, Real.exp_nat_mul]
    have h4 : Real.exp 1 ≤ 3 := le_of_lt Real.exp_one_lt_three
    have h5 : Real.exp 1 ^ 61 ≤ (3 : ℝ) ^ 61 := pow_le_pow_left₀ (Real.exp_pos 1).le h4 61
    have h6 : (3 : ℝ) ^ 61 ≤ (2 : ℝ) ^ 97 := by norm_num
    linarith
  -- per-element bounds
  have hperk : ∀ k : Fin (8 - 0),
      (diffs.get1 k).isFinite = true ∧
      |(diffs.get1 k).toReal - (wC capped k.val - c0)| ≤ (27 / 5000000 : ℝ) ∧
      (exps.get1 k).isFinite = true ∧
      |(exps.get1 k).toReal - Real.exp (wC capped k.val - c0)| ≤
        (1292 / 100000000 : ℝ) * Real.exp (wC capped k.val - c0) + (2 : ℝ) ^ (-140 : ℤ) := by
    intro k
    have hk := k.isLt
    have hwC : wC capped k.val = (capped.get1 k).toReal := by
      simp only [wC]
      rw [dite_eq_left hk]
    have hAr : AddRel ((launch d).env P).fusion (capped.get1 k) (FloatV.neg top1v)
        (diffs.get1 k) := hdiffs (k, ())
    have hMq : |(capped.get1 k).toRat| + |(FloatV.neg top1v).toRat| ≤
        Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
      have hxq := hsmallq k
      have hyq : |(FloatV.neg top1v).toRat| ≤ Bq := by
        rw [FloatV.toRat_neg, abs_neg]
        exact htopq
      have hbig : (62 : ℚ) ≤
          Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
        rw [Format.binary32_maxVal, Format.binary32_eta]
        norm_num
      have hB31 : Bq ≤ 31 := by unfold Bq; norm_num
      linarith
    obtain ⟨hdifff, hdiffq⟩ := addrel_err_fusion bin32_regular bin32_specials bin32_u_le
      hAr hfus (hfin k) htopneg hMq
    have hdeltaq : Format.binary32.u *
          (|(capped.get1 k).toRat| + |(FloatV.neg top1v).toRat| +
            max |(capped.get1 k).toRat| |(FloatV.neg top1v).toRat|)
        + 2 * Format.binary32.u * Format.binary32.u *
            max |(capped.get1 k).toRat| |(FloatV.neg top1v).toRat|
        + 4 * Format.binary32.eta ≤ (27 / 5000000 : ℚ) := by
      have hxq := hsmallq k
      have hyq : |(FloatV.neg top1v).toRat| ≤ Bq := by
        rw [FloatV.toRat_neg, abs_neg]
        exact htopq
      have h1 : |(capped.get1 k).toRat| + |(FloatV.neg top1v).toRat| +
          max |(capped.get1 k).toRat| |(FloatV.neg top1v).toRat| ≤ 3 * Bq := by
        have hb : max |(capped.get1 k).toRat| |(FloatV.neg top1v).toRat| ≤ Bq := max_le hxq hyq
        linarith
      have h2 : max |(capped.get1 k).toRat| |(FloatV.neg top1v).toRat| ≤ Bq := max_le hxq hyq
      have hu0 : (0 : ℚ) ≤ Format.binary32.u := le_of_lt bin32_u_pos
      have hu2 : (0 : ℚ) ≤ 2 * Format.binary32.u * Format.binary32.u :=
        mul_nonneg (mul_nonneg (by norm_num) hu0) hu0
      have h3 : Format.binary32.u *
            (|(capped.get1 k).toRat| + |(FloatV.neg top1v).toRat| +
              max |(capped.get1 k).toRat| |(FloatV.neg top1v).toRat|)
          ≤ Format.binary32.u * (3 * Bq) := mul_le_mul_of_nonneg_left h1 hu0
      have h4 : 2 * Format.binary32.u * Format.binary32.u *
            max |(capped.get1 k).toRat| |(FloatV.neg top1v).toRat|
          ≤ 2 * Format.binary32.u * Format.binary32.u * Bq :=
        mul_le_mul_of_nonneg_left h2 hu2
      have h5 : Format.binary32.u * (3 * Bq)
          + 2 * Format.binary32.u * Format.binary32.u * Bq
          + 4 * Format.binary32.eta ≤ (27 / 5000000 : ℚ) := by
        rw [Format.binary32_u, Format.binary32_eta]
        unfold Bq
        norm_num
      linarith
    have hdiffq' : |(diffs.get1 k).toRat - ((capped.get1 k).toRat + -top1v.toRat)| ≤
        (27 / 5000000 : ℚ) := by
      have h := hdiffq.trans hdeltaq
      rwa [FloatV.toRat_neg] at h
    have hdiffR : |(diffs.get1 k).toReal - ((capped.get1 k).toReal + -top1v.toReal)| ≤
        (27 / 5000000 : ℝ) := by
      have h := Rat.cast_le (K := ℝ).mpr hdiffq'
      rw [Rat.cast_abs] at h
      push_cast at h
      simpa [FloatV.toReal] using h
    have hdiffW : |(diffs.get1 k).toReal - (wC capped k.val - c0)| ≤ (27 / 5000000 : ℝ) := by
      calc |(diffs.get1 k).toReal - (wC capped k.val - c0)|
          = |(diffs.get1 k).toReal - ((capped.get1 k).toReal + -top1v.toReal)| := by
            rw [hwC, htopval']
            ring_nf
        _ ≤ _ := hdiffR
    obtain ⟨eref, heref, hacc⟩ := hexpacc 61 (by norm_num)
    have heref' : eref = Real.exp := by
      have h := heref
      rw [show MathFn.exp.ref = some Real.exp from rfl] at h
      exact (Option.some.inj h).symm
    have hdk0 : wC capped k.val - c0 ≤ 0 := by
      have h := hcle k
      rw [← hwC] at h
      linarith
    have hdk61 : |wC capped k.val - c0| ≤ 601 / 10 := by
      have h1 : |wC capped k.val| ≤ 30 * (1 + (2 : ℝ) ^ (-20 : ℤ)) := by
        rw [hwC]
        exact hsmall k
      have h2 : |c0| ≤ 30 * (1 + (2 : ℝ) ^ (-20 : ℤ)) := by
        rw [hc0]
        exact hsmall ⟨i0, hi0⟩
      have h3 : |wC capped k.val - c0| ≤ |wC capped k.val| + |c0| := by
        calc |wC capped k.val - c0| = |wC capped k.val + -c0| := by ring_nf
          _ ≤ |wC capped k.val| + |-c0| := abs_add_le _ _
          _ = |wC capped k.val| + |c0| := by rw [abs_neg]
      have h4 : (30 : ℝ) * (1 + (2 : ℝ) ^ (-20 : ℤ)) +
          (30 : ℝ) * (1 + (2 : ℝ) ^ (-20 : ℤ)) ≤ 601 / 10 := by norm_num
      calc |wC capped k.val - c0| ≤ |wC capped k.val| + |c0| := h3
        _ ≤ (30 : ℝ) * (1 + (2 : ℝ) ^ (-20 : ℤ)) +
              (30 : ℝ) * (1 + (2 : ℝ) ^ (-20 : ℤ)) := add_le_add h1 h2
        _ ≤ 601 / 10 := h4
    have hr61 : |(diffs.get1 k).toReal| ≤ 61 := by
      have h := abs_le.mp hdiffW
      have h2 := abs_le.mp hdk61
      rw [abs_le]
      constructor <;> linarith [h.1, h.2, h2.1, h2.2]
    have hdom : (diffs.get1 k).toReal ∈ Set.Icc (-61) 61 := by
      rw [Set.mem_Icc]
      exact abs_le.mp hr61
    have hnat : (diffs.get1 k).toReal ∈ MathFn.exp.natDom := by
      simp [MathFn.natDom]
    have hroom : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 61)) *
          |eref ((diffs.get1 k).toReal)| + (2 : ℝ) ^ (-140 : ℤ) ≤
        ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [heref']
      have hle : (diffs.get1 k).toReal ≤ 61 := (abs_le.mp hr61).2
      have hexp := Real.exp_le_exp.mpr hle
      have hrel : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 61)) ≤ 2 := by norm_num
      have hpos : (0 : ℝ) < Real.exp ((diffs.get1 k).toReal) := Real.exp_pos _
      rw [abs_of_pos (Real.exp_pos _), maxVal32_eq]
      have hmul : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 61)) *
            Real.exp ((diffs.get1 k).toReal) ≤ 2 * (2 : ℝ) ^ 97 := by
        nlinarith [hexp, hexp61, hrel, hpos, Real.exp_pos ((diffs.get1 k).toReal)]
      have hnum : 2 * (2 : ℝ) ^ 97 + (2 : ℝ) ^ (-140 : ℤ) ≤
          (2 : ℝ) ^ 128 - (2 : ℝ) ^ 104 := by norm_num
      linarith
    have hexps_j : P.math MathFn.exp Format.binary32 (diffs.get1 k) (exps.get1 k) :=
      hexps (k, ())
    have hexps_k := hacc (diffs.get1 k) (exps.get1 k) hexps_j hdifff hdom hnat hroom
    obtain ⟨hexpf, hexpe⟩ := hexps_k
    rw [heref'] at hexpe
    have hexpe' : |(exps.get1 k).toReal - Real.exp ((diffs.get1 k).toReal)| ≤
        ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 61) *
          Real.exp ((diffs.get1 k).toReal) + (2 : ℝ) ^ (-140 : ℤ) := by
      rw [abs_of_pos (Real.exp_pos _)] at hexpe
      exact hexpe
    have hρ : (0 : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 61 := by norm_num
    have hγ : (0 : ℝ) ≤ (27 : ℝ) / 5000000 := by norm_num
    have hγhalf : (27 : ℝ) / 5000000 ≤ 1 / 2 := by norm_num
    have hratio : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 61) / (1 - 27 / 5000000)
        + (27 / 5000000) / (1 - 27 / 5000000) ≤ (1292 / 100000000 : ℝ) := by norm_num
    have hexpfin : |(exps.get1 k).toReal - Real.exp (wC capped k.val - c0)| ≤
        (1292 / 100000000 : ℝ) * Real.exp (wC capped k.val - c0) + (2 : ℝ) ^ (-140 : ℤ) := by
      have htr := exp_err_transfer (e := (exps.get1 k).toReal) (r := (diffs.get1 k).toReal)
        (dk := wC capped k.val - c0)
        (ρ := (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * 61)
        (ω := 27 / 5000000) (c := (2 : ℝ) ^ (-140 : ℤ)) hρ hγ hγhalf hdiffW hexpe'
      have hE : (0 : ℝ) ≤ Real.exp (wC capped k.val - c0) := (Real.exp_pos _).le
      nlinarith [htr, mul_le_mul_of_nonneg_right hratio hE]
    exact ⟨hdifff, hdiffW, hexpf, hexpfin⟩
  -- ===== the sum `den` =====
  obtain ⟨tr, hcov, hdep, hev⟩ := hden
  set x : Nat → F32 := (fun i => if h : i < 8 - 0 then exps.get1 ⟨i, h⟩ else default) with hx_def
  have hev' : Tree.Eval (AddRel false) x tr den := by
    rw [hx_def]
    exact hev
  have hxi : ∀ (i : Nat) (hi : i < 8 - 0), x i = exps.get1 ⟨i, hi⟩ := by
    intro i hi
    rw [hx_def]
    simp only [dite_eq_left hi]
  have hd3 : tr.depth ≤ 3 := by
    have h := depth_le_clog hsum hdep
    rwa [hclog] at h
  have hwCk : ∀ (k : Nat) (hk : k < 8 - 0), wC capped k = (capped.get1 ⟨k, hk⟩).toReal := by
    intro k hk
    unfold wC
    rw [dite_eq_left hk]
  have hwC0 : wC capped i0 = c0 := by
    rw [hwCk i0 hi0, ← hc0]
  have hwD : wD capped i0 = ∑ k ∈ Finset.range 8, Real.exp (wC capped k - c0) := by
    show (∑ k ∈ Finset.range 8, Real.exp (wC capped k - wC capped i0)) = _
    exact Finset.sum_congr rfl (fun k _ => by rw [hwC0])
  have huq : Format.binary32.u = (2 : ℚ) ^ (-24 : ℤ) :=
    Format.binary32_u
  have hetq : Format.binary32.eta = (2 : ℚ) ^ (-150 : ℤ) :=
    Format.binary32_eta
  have huq_pos : (0 : ℚ) < Format.binary32.u := by rw [huq]; norm_num
  have hetq_pos : (0 : ℚ) < Format.binary32.eta := by rw [hetq]; norm_num
  have hxfin : ∀ i, i < 8 - 0 → (x i).isFinite = true := by
    intro i hi
    rw [hxi i hi]
    exact (hperk ⟨i, hi⟩).2.2.1
  have hperr : ∀ k : Fin (8 - 0),
      |(x k.val).toReal - Real.exp (wC capped k.val - c0)| ≤
        (1292 / 100000000 : ℝ) * Real.exp (wC capped k.val - c0) + (2 : ℝ) ^ (-140 : ℤ) := by
    intro k
    have h := (hperk k).2.2.2
    rw [hxi k.val k.isLt]
    exact h
  set Sx : ℝ := ∑ k ∈ Finset.range (8 - 0), (x k).toReal with hSx
  have hSdev : |Sx - wD capped i0| ≤
      (1292 / 100000000 : ℝ) * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ) := by
    have hdiff : Sx - wD capped i0
        = ∑ k ∈ Finset.range (8 - 0), ((x k).toReal - Real.exp (wC capped k - c0)) := by
      rw [hSx, hwD, Finset.sum_sub_distrib]
    rw [hdiff]
    have h1 : |∑ k ∈ Finset.range (8 - 0), ((x k).toReal - Real.exp (wC capped k - c0))|
        ≤ ∑ k ∈ Finset.range (8 - 0),
            |(x k).toReal - Real.exp (wC capped k - c0)| :=
      Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ k ∈ Finset.range (8 - 0),
            |(x k).toReal - Real.exp (wC capped k - c0)|
        ≤ ∑ k ∈ Finset.range (8 - 0),
            ((1292 / 100000000 : ℝ) * Real.exp (wC capped k - c0) + (2 : ℝ) ^ (-140 : ℤ)) :=
      Finset.sum_le_sum (fun k hk => hperr ⟨k, Finset.mem_range.mp hk⟩)
    have h3 : ∑ k ∈ Finset.range (8 - 0),
            ((1292 / 100000000 : ℝ) * Real.exp (wC capped k - c0) + (2 : ℝ) ^ (-140 : ℤ))
        = (1292 / 100000000 : ℝ) * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ) := by
      rw [hwD, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
        nsmul_eq_mul]
      push_cast
      ring
    linarith [h1, h2, h3]
  have hSge1 : (1 : ℝ) ≤ wD capped i0 := by
    rw [hwD]
    have h := Finset.single_le_sum (f := fun k => Real.exp (wC capped k - c0))
      (fun k _ => (Real.exp_pos _).le) (Finset.mem_range.mpr hi0)
    simpa [hwC0] using h
  have hSle8 : wD capped i0 ≤ 8 := by
    rw [hwD]
    have h1 : ∑ k ∈ Finset.range 8, Real.exp (wC capped k - c0)
        ≤ ∑ k ∈ Finset.range 8, (1 : ℝ) := by
      refine Finset.sum_le_sum (fun k hk => ?_)
      have hk' : k < 8 - 0 := Finset.mem_range.mp hk
      have hw : wC capped k ≤ c0 := by
        rw [hwCk k hk', hc0]
        exact hcle ⟨k, hk'⟩
      exact Real.exp_le_one_iff.mpr (by linarith)
    calc ∑ k ∈ Finset.range 8, Real.exp (wC capped k - c0) ≤ ∑ k ∈ Finset.range 8, (1 : ℝ) := h1
      _ = 8 := by rw [Finset.sum_const, Finset.card_range]; norm_num
  set A : ℚ := ((List.range (8 - 0)).map fun i => |(x i).toRat|).sum with hA
  have hAnonneg : (0 : ℚ) ≤ A := by
    rw [hA, list_sum_range_congr_rat]
    exact Finset.sum_nonneg (fun i _ => abs_nonneg _)
  have hAc : ((A : ℚ) : ℝ) = ∑ k ∈ Finset.range (8 - 0), |(x k).toReal| := by
    rw [hA, cast_list_sum_rat]
    simp only [Rat.cast_abs, FloatV.toReal]
    exact (list_sum_range_congr_real (fun k => |(x k).toReal|) (8 - 0)).symm
  have hArel : ((A : ℚ) : ℝ) ≤
      (1 + 1292 / 100000000 : ℝ) * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ) := by
    rw [hAc]
    have h1 : ∑ k ∈ Finset.range (8 - 0), |(x k).toReal|
        ≤ ∑ k ∈ Finset.range (8 - 0), (Real.exp (wC capped k - c0)
            + ((1292 / 100000000 : ℝ) * Real.exp (wC capped k - c0)
              + (2 : ℝ) ^ (-140 : ℤ))) := by
      refine Finset.sum_le_sum (fun k hk => ?_)
      have hp := hperr ⟨k, Finset.mem_range.mp hk⟩
      have htri : |(x k).toReal| ≤ Real.exp (wC capped k - c0)
          + |(x k).toReal - Real.exp (wC capped k - c0)| := by
        calc |(x k).toReal|
            = |Real.exp (wC capped k - c0)
                + ((x k).toReal - Real.exp (wC capped k - c0))| := by ring_nf
          _ ≤ |Real.exp (wC capped k - c0)|
                + |(x k).toReal - Real.exp (wC capped k - c0)| := abs_add_le _ _
          _ = Real.exp (wC capped k - c0)
                + |(x k).toReal - Real.exp (wC capped k - c0)| := by
                rw [abs_of_pos (Real.exp_pos _)]
      linarith [htri, hp]
    have h2 : ∑ k ∈ Finset.range (8 - 0), (Real.exp (wC capped k - c0)
            + ((1292 / 100000000 : ℝ) * Real.exp (wC capped k - c0)
              + (2 : ℝ) ^ (-140 : ℤ)))
        = (1 + 1292 / 100000000 : ℝ) * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ) := by
      rw [hwD, Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
        Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      push_cast
      ring
    linarith [h1, h2]
  have hA9 : A ≤ 9 := by
    apply Rat.cast_le (K := ℝ).mp
    push_cast
    have hc8 : 8 * (2 : ℝ) ^ (-140 : ℤ) ≤ 1 / 1000 := by norm_num
    have h1 : (1 + 1292 / 100000000 : ℝ) * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ) ≤ 9 := by
      nlinarith [hSge1, hSle8, hc8]
    linarith [hArel, h1]
  have hδq : (0 : ℚ) ≤ 3 * Format.binary32.u := by linarith [huq_pos]
  have hηq : (0 : ℚ) ≤ 4 * Format.binary32.eta := by linarith [hetq_pos]
  have hmonoA : Tree.errBound (3 * Format.binary32.u)
        (4 * Format.binary32.eta) 3 7 A
      ≤ Tree.errBound (3 * Format.binary32.u)
        (4 * Format.binary32.eta) 3 7 9 :=
    Tree.errBound_mono hδq hηq (le_refl 3) (le_refl 7) hAnonneg hA9
  have hnumE : Tree.errBound (3 * Format.binary32.u)
      (4 * Format.binary32.eta) 3 7 9 ≤ 1 := by
    show ((1 + 3 * Format.binary32.u) ^ 3 - 1) * 9
        + (4 * Format.binary32.eta) * ((7 : ℕ) : ℚ)
          * (1 + 3 * Format.binary32.u) ^ 3 ≤ 1
    rw [huq, hetq]
    norm_num
  have hmax2 : (17 : ℚ) ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [Format.binary32_maxVal, hetq]
    norm_num
  have hM : ((List.range (8 - 0)).map fun i => |(x i).toRat|).sum
      + Tree.errBound (3 * Format.binary32.u)
          (4 * Format.binary32.eta) 3 7
          ((List.range (8 - 0)).map fun i => |(x i).toRat|).sum
      ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have h := add_le_add hA9 (le_trans hmonoA hnumE)
    linarith [h, hmax2]
  obtain ⟨hresfin, hreserr⟩ := TVA.sumc_err_depth
    (f := Format.binary32) bin32_regular bin32_specials bin32_u_le false
    x hcov hd3 hev' hxfin hM
  have hlist : ((List.map (fun i => FloatV.toRat (x i)) (List.range (8 - 0))).map Rat.cast).sum
      = Sx := by
    rw [List.map_map, hSx]
    exact (list_sum_range_congr_real (fun i => (x i).toReal) (8 - 0)).symm
  have htree_q : |den.toReal - Sx|
      ≤ ((Tree.errBound (3 * Format.binary32.u)
          (4 * Format.binary32.eta) 3 7 A : ℚ) : ℝ) := by
    have h := Rat.cast_le (K := ℝ).mpr hreserr
    rw [Rat.cast_abs, Rat.cast_sub, Rat.cast_list_sum] at h
    rw [hlist] at h
    exact h
  have hδeq : ((3 * Format.binary32.u : ℚ) : ℝ) = 3 * u32 := by
    push_cast
    rw [u32_eq]
  have hηeq : ((4 * Format.binary32.eta : ℚ) : ℝ) = 4 * eta32 := by
    push_cast
    rw [eta32_eq]
  have htree : |den.toReal - Sx|
      ≤ ((1 + 3 * u32) ^ 3 - 1) * (A : ℝ) + (4 * eta32) * 7 * (1 + 3 * u32) ^ 3 := by
    have h := htree_q
    rw [errBound_cast] at h
    rw [hδeq, hηeq] at h
    simpa using h
  have hcoef : ((1 + 3 * u32) ^ 3 - 1) * (1 + 1292 / 100000000 : ℝ) + 1292 / 100000000
      ≤ 138 / 10000000 := by
    unfold u32
    norm_num
  have hconst : ((1 + 3 * u32) ^ 3 - 1) * (8 * (2 : ℝ) ^ (-140 : ℤ))
      + (4 * eta32) * 7 * (1 + 3 * u32) ^ 3 + 8 * (2 : ℝ) ^ (-140 : ℤ) ≤ 2 / 10000000 := by
    unfold u32 eta32
    norm_num
  have htri : |den.toReal - wD capped i0| ≤ |den.toReal - Sx| + |Sx - wD capped i0| := by
    calc |den.toReal - wD capped i0| = |(den.toReal - Sx) + (Sx - wD capped i0)| := by ring_nf
      _ ≤ _ := abs_add_le _ _
  have hdenD : |den.toReal - wD capped i0| ≤ (7 / 500000 : ℝ) * wD capped i0 := by
    have hpow : (0 : ℝ) ≤ (1 + 3 * u32) ^ 3 - 1 := by
      have h0 : (0 : ℝ) ≤ u32 := by unfold u32; positivity
      have h1 : (1 : ℝ) ≤ 1 + 3 * u32 := by linarith
      have h2 : (1 : ℝ) ≤ (1 + 3 * u32) ^ 3 := one_le_pow₀ h1
      linarith
    have hbound1 : ((1 + 3 * u32) ^ 3 - 1) * (A : ℝ)
        ≤ ((1 + 3 * u32) ^ 3 - 1)
          * ((1 + 1292 / 100000000 : ℝ) * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ)) :=
      mul_le_mul_of_nonneg_left hArel hpow
    have hmain : |den.toReal - wD capped i0|
        ≤ ((1 + 3 * u32) ^ 3 - 1)
            * ((1 + 1292 / 100000000 : ℝ) * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ))
          + (4 * eta32) * 7 * (1 + 3 * u32) ^ 3
          + (1292 / 100000000 * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ)) := by
      linarith [htri, htree, hSdev, hbound1]
    have hmain2 : ((1 + 3 * u32) ^ 3 - 1)
            * ((1 + 1292 / 100000000 : ℝ) * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ))
          + (4 * eta32) * 7 * (1 + 3 * u32) ^ 3
          + (1292 / 100000000 * wD capped i0 + 8 * (2 : ℝ) ^ (-140 : ℤ))
        = (((1 + 3 * u32) ^ 3 - 1) * (1 + 1292 / 100000000 : ℝ) + 1292 / 100000000)
            * wD capped i0
          + (((1 + 3 * u32) ^ 3 - 1) * (8 * (2 : ℝ) ^ (-140 : ℤ))
            + (4 * eta32) * 7 * (1 + 3 * u32) ^ 3 + 8 * (2 : ℝ) ^ (-140 : ℤ)) := by
      ring
    rw [hmain2] at hmain
    have h2 : (((1 + 3 * u32) ^ 3 - 1) * (1 + 1292 / 100000000 : ℝ) + 1292 / 100000000)
          * wD capped i0
        ≤ (138 / 10000000 : ℝ) * wD capped i0 :=
      mul_le_mul_of_nonneg_right hcoef (by linarith [hSge1])
    have h3 : (2 / 10000000 : ℝ) ≤ (2 / 10000000 : ℝ) * wD capped i0 := by
      nlinarith [hSge1]
    linarith [hmain, h2, h3, hconst]
  -- ===== `den` is bounded away from zero =====
  have hdenhalf : (1 / 2 : ℝ) ≤ den.toReal := by
    have h := abs_le.mp hdenD
    nlinarith [h.1, h.2, hSge1]
  have hdenpos : (0 : ℝ) < den.toReal := by linarith
  have hdenne : den.toReal ≠ 0 := ne_of_gt hdenpos
  have hdenrat_pos : (0 : ℚ) < den.toRat := by
    apply Rat.cast_pos (K := ℝ).mp
    show (0 : ℝ) < den.toReal
    exact hdenpos
  have hyq : (1 / 2 : ℚ) ≤ |den.toRat| := by
    rw [abs_of_pos hdenrat_pos]
    apply Rat.cast_le (K := ℝ).mp
    push_cast
    simpa [FloatV.toReal] using hdenhalf
  -- ===== the numerator is `c32 1` =====
  have hX1 : |(c32 0x3FF0000000000000 : F32).toReal - 1|
      ≤ (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have h := c32_one_err
    unfold u32 eta32 at h
    exact h
  have hXabs : |(c32 0x3FF0000000000000 : F32).toReal|
      ≤ 1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have h2 : |(c32 0x3FF0000000000000 : F32).toReal|
        ≤ |(c32 0x3FF0000000000000 : F32).toReal - 1| + 1 := by
      calc |(c32 0x3FF0000000000000 : F32).toReal|
          = |((c32 0x3FF0000000000000 : F32).toReal - 1) + 1| := by ring_nf
        _ ≤ |(c32 0x3FF0000000000000 : F32).toReal - 1| + |(1 : ℝ)| := abs_add_le _ _
        _ = |(c32 0x3FF0000000000000 : F32).toReal - 1| + 1 := by norm_num
    linarith [h2, hX1]
  have hXq : |(c32 0x3FF0000000000000 : F32).toRat| ≤ 2 := by
    apply Rat.cast_le (K := ℝ).mp
    rw [Rat.cast_abs]
    push_cast
    have h2 : (1 + (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) ≤ 2 := by norm_num
    simpa [FloatV.toReal] using le_trans hXabs h2
  have hqmax : |((Tensor.scalar (c32 0x3FF0000000000000)).get ()).toRat
      / ((Tensor.scalar den).get ()).toRat| ≤ (2 : ℚ) ^ (126 : ℤ) := by
    show |(c32 0x3FF0000000000000 : F32).toRat / den.toRat| ≤ (2 : ℚ) ^ (126 : ℤ)
    rw [abs_div, div_le_iff₀ (by linarith [hyq] : (0 : ℚ) < |den.toRat|)]
    have hpos : (0 : ℚ) ≤ (2 : ℚ) ^ (126 : ℤ) := by positivity
    have hmul : (2 : ℚ) ^ (126 : ℤ) * (1 / 2) ≤ (2 : ℚ) ^ (126 : ℤ) * |den.toRat| :=
      mul_le_mul_of_nonneg_left hyq hpos
    have h2 : (2 : ℚ) ≤ (2 : ℚ) ^ (126 : ℤ) * (1 / 2) := by norm_num
    linarith [hXq, hmul, h2]
  obtain ⟨hqfin, hqerrq⟩ := hdivfull
    ((Tensor.scalar (c32 0x3FF0000000000000)).get ())
    ((Tensor.scalar den).get ()) (qinv.get ()) (hinv ())
    (show FloatV.isFinite ((Tensor.scalar (c32 0x3FF0000000000000)).get ()) = true from
      c32_one_isFinite)
    hresfin (show ((Tensor.scalar den).get ()).toRat ≠ 0 from ne_of_gt hdenrat_pos) hqmax
  have hqerr : |qinv.item.toReal - (c32 0x3FF0000000000000 : F32).toReal / den.toReal|
      ≤ (2 : ℝ) ^ (-22 : ℤ) * |(c32 0x3FF0000000000000 : F32).toReal / den.toReal|
        + (2 : ℝ) ^ (-148 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr hqerrq
    push_cast at h
    exact h
  -- ===== final estimate =====
  have hgoal : |qinv.item.toReal * wD capped i0 - 1| ≤ (3 : ℝ) / 200000 :=
    final_assembly hdenpos hdenD hSge1 hSle8 hqerr hXabs hX1
  exact ⟨hresfin, hdenne, hqfin, hgoal⟩

end Claims.MoeRouter
