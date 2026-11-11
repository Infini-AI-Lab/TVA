import Proof.NumSum
import Proof.Defs

/-!
# `l2norm_fwd_bf16`: the sum of squares of one row

`tl.sum(x * x, 1)` on one row: `s1` is the computed sum of the squared `float32` values `bxx`, `a`
the row's `bfloat16` inputs. Given that each `bxx j` is within `u·a_j² + η` of the exact `a_j²`,
`s1` is finite and within `2^-13·A + (3/4)·2^-137` of `A = Σ a_j²` (`NumSum.sum_sq_compose`).
-/

set_option maxHeartbeats 1000000

namespace Claims.L2NormFwd

open TVA
open TVA TVA.Format

/-! The `List.sum` form of `sumc_err_covers` as a `Finset.sum` over `Finset.range`. -/
theorem list_sum_range (f : Nat → ℚ) (n : Nat) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]
      simp

/-- **The row sum**: `s1`, a sum of the squares `bxx` along any tree of fusable additions, is finite and within `2^-13·A + (3/4)·2^-137` of the exact `A = Σ a_j²`. -/
theorem sum_err {d : Inst} (hd : d.ok) (fusion : Bool) {a : Nat → BF16} {bxx : Nat → F32}
    {s1 : F32}
    (hfin : ∀ j, j < d.D → (bxx j).isFinite = true)
    (hb : ∀ j, j < d.D →
      |(bxx j).toReal - (a j).toReal ^ 2|
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ))
    (hA : rowSum a d.D + d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ))
    (he : (2 : ℝ) ^ (-125 : ℤ) ≤ d.eps.toReal)
    (htree : ∃ tr : Tree, tr.Covers d.D ∧ tr.Eval (AddRel fusion) bxx s1) :
    s1.isFinite = true ∧
      |s1.toReal - rowSum a d.D|
        ≤ (2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := by
  obtain ⟨tr, hcov, hev⟩ := htree
  have hu : Format.binary32.u ≤ 1 / 16 := by
    rw [Format.binary32_u]; norm_num
  -- the row sum, the two `List.sum`s of `sumc_err_covers`, in `ℚ`
  set Aq : ℚ := ∑ j ∈ Finset.range d.D, (a j).toRat ^ 2 with hAqdef
  set Bq : ℚ := ((List.range d.D).map fun i => |(bxx i).toRat|).sum with hBqdef
  set Sq : ℚ := ((List.range d.D).map fun i => (bxx i).toRat).sum with hSqdef
  -- ## `Aq ≤ 2^126 - 2^-125` and `eps ≥ 2^-125`, in `ℚ`
  have hrowSum_cast : rowSum a d.D = (Aq : ℝ) := by
    rw [hAqdef]
    simp only [rowSum, FloatV.toReal]
    push_cast
    rfl
  have hAq : Aq + d.eps.toRat ≤ (2 : ℚ) ^ (126 : ℤ) := by
    have h := hA
    rw [hrowSum_cast] at h
    simp only [FloatV.toReal] at h
    exact_mod_cast h
  have heps : (2 : ℚ) ^ (-125 : ℤ) ≤ d.eps.toRat := by
    have h := he
    simp only [FloatV.toReal] at h
    refine Rat.cast_le (K := ℝ).mp ?_
    push_cast
    exact h
  have hAqle : Aq ≤ (2 : ℚ) ^ (126 : ℤ) - (2 : ℚ) ^ (-125 : ℤ) := by linarith
  -- ## each `|bxx j|` and the sum `Bq`
  have hper : ∀ j, j < d.D → |(bxx j).toRat| ≤
      (1 + (2 : ℚ) ^ (-24 : ℤ)) * (a j).toRat ^ 2 + (2 : ℚ) ^ (-150 : ℤ) := by
    intro j hj
    have h1 : |(bxx j).toRat - (a j).toRat ^ 2| ≤
        (2 : ℚ) ^ (-24 : ℤ) * (a j).toRat ^ 2 + (2 : ℚ) ^ (-150 : ℤ) := by
      have h := hb j hj
      simp only [FloatV.toReal] at h
      refine Rat.cast_le (K := ℝ).mp ?_
      push_cast
      exact h
    have h2 : |(bxx j).toRat| ≤ |(a j).toRat ^ 2| + |(bxx j).toRat - (a j).toRat ^ 2| := by
      have h3 := abs_add_le ((a j).toRat ^ 2) ((bxx j).toRat - (a j).toRat ^ 2)
      rwa [add_sub_cancel] at h3
    rw [abs_of_nonneg (sq_nonneg ((a j).toRat))] at h2
    nlinarith
  have hBq : Bq ≤ (1 + (2 : ℚ) ^ (-24 : ℤ)) * Aq + (d.D : ℚ) * (2 : ℚ) ^ (-150 : ℤ) := by
    rw [hBqdef, hAqdef, list_sum_range]
    have h := Finset.sum_le_sum (fun j hj => hper j (Finset.mem_range.mp hj))
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
      nsmul_eq_mul] at h
    exact h
  have hBq0 : 0 ≤ Bq := by
    rw [hBqdef, list_sum_range]
    exact Finset.sum_nonneg (fun i _ => abs_nonneg _)
  -- ## room for the tree: `Bq + errBound ≤ maxVal/2 - 2η`
  have hDle : (d.D : ℚ) ≤ 512 := by exact_mod_cast hd.d_le
  set X : ℚ := (2 : ℚ) ^ (126 : ℤ) + (2 : ℚ) ^ (102 : ℤ) + (2 : ℚ) ^ (-141 : ℤ) with hX
  have hBqle : Bq ≤ X := by
    have h1 : (1 + (2 : ℚ) ^ (-24 : ℤ)) * Aq ≤
        (1 + (2 : ℚ) ^ (-24 : ℤ)) * ((2 : ℚ) ^ (126 : ℤ) - (2 : ℚ) ^ (-125 : ℤ)) :=
      mul_le_mul_of_nonneg_left hAqle (by norm_num)
    have h2 : (d.D : ℚ) * (2 : ℚ) ^ (-150 : ℤ) ≤ 512 * (2 : ℚ) ^ (-150 : ℤ) :=
      mul_le_mul_of_nonneg_right hDle (by norm_num)
    have h3 : (1 + (2 : ℚ) ^ (-24 : ℤ)) * ((2 : ℚ) ^ (126 : ℤ) - (2 : ℚ) ^ (-125 : ℤ)) +
        512 * (2 : ℚ) ^ (-150 : ℤ) ≤ X := by
      rw [hX]; norm_num
    exact hBq.trans ((add_le_add h1 h2).trans h3)
  have herrB := errBound_le hd.d_le (B := Bq) hBq0
  have herrnum : Tree.errBound (3 * (2 : ℚ) ^ (-24 : ℤ)) (4 * (2 : ℚ) ^ (-150 : ℤ))
      (d.D - 1) (d.D - 1) Bq ≤ (3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * X + (2 : ℚ) ^ (-138 : ℤ) := by
    refine herrB.trans ?_
    have h := mul_le_mul_of_nonneg_left hBqle (by norm_num : (0 : ℚ) ≤ (3 / 4) * (2 : ℚ) ^ (-13 : ℤ))
    simpa only [add_comm] using add_le_add_right h ((2 : ℚ) ^ (-138 : ℤ))
  have hMnumeric : X + ((3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * X + (2 : ℚ) ^ (-138 : ℤ)) ≤
      ((2 : ℚ) ^ 24 - 1) * (2 : ℚ) ^ 104 / 2 - 2 * (2 : ℚ) ^ (-150 : ℤ) := by
    rw [hX]; norm_num
  have hstep : Bq + Tree.errBound (3 * (2 : ℚ) ^ (-24 : ℤ)) (4 * (2 : ℚ) ^ (-150 : ℤ))
      (d.D - 1) (d.D - 1) Bq
      ≤ X + ((3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * X + (2 : ℚ) ^ (-138 : ℤ)) :=
    add_le_add hBqle herrnum
  have hM : Bq + Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta)
      (d.D - 1) (d.D - 1) Bq ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
    exact hstep.trans hMnumeric
  -- ## the tree's error, from `sumc_err_covers`
  obtain ⟨hfinS, herr⟩ :=
    sumc_err_covers (f := Format.binary32) Format.binary32_regular (by decide) hu fusion bxx
      hcov hev hfin hM
  -- ## the same bound in `ℚ`, then in `ℝ` with `Finset.sum`s
  have hq : |s1.toRat - Sq| ≤ (3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * Bq + (2 : ℚ) ^ (-138 : ℤ) := by
    have h := herr
    rw [← hSqdef, ← hBqdef] at h
    refine h.trans ?_
    rw [Format.binary32_u, Format.binary32_eta]
    exact herrB
  have hqR : |s1.toReal - (Sq : ℝ)| ≤
      (3 / 4) * (2 : ℝ) ^ (-13 : ℤ) * (Bq : ℝ) + (2 : ℝ) ^ (-138 : ℤ) := by
    have hcast : ((|s1.toRat - Sq| : ℚ) : ℝ) ≤
        ((((3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * Bq + (2 : ℚ) ^ (-138 : ℤ)) : ℚ) : ℝ) :=
      Rat.cast_le (K := ℝ).mpr hq
    have hL : ((|s1.toRat - Sq| : ℚ) : ℝ) = |s1.toReal - (Sq : ℝ)| := by
      rw [Rat.cast_abs]
      simp only [FloatV.toReal]
      push_cast
      ring_nf
    have hR : ((((3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * Bq + (2 : ℚ) ^ (-138 : ℤ)) : ℚ) : ℝ) =
        (3 / 4) * (2 : ℝ) ^ (-13 : ℤ) * (Bq : ℝ) + (2 : ℝ) ^ (-138 : ℤ) := by
      push_cast
      ring
    rwa [hL, hR] at hcast
  have hcastS : (Sq : ℝ) = ∑ j ∈ Finset.range d.D, (bxx j).toReal := by
    rw [hSqdef, list_sum_range]
    simp only [FloatV.toReal]
    push_cast
    rfl
  have hcastB : (Bq : ℝ) = ∑ j ∈ Finset.range d.D, |(bxx j).toReal| := by
    rw [hBqdef, list_sum_range]
    simp only [FloatV.toReal]
    push_cast
    rfl
  have hrt : |s1.toReal - ∑ j ∈ Finset.range d.D, (bxx j).toReal| ≤
      (3 / 4) * (2 : ℝ) ^ (-13 : ℤ) * (∑ j ∈ Finset.range d.D, |(bxx j).toReal|) +
        (2 : ℝ) ^ (-138 : ℤ) := by
    rw [← hcastS, ← hcastB]
    exact hqR
  -- ## compose with the per-element rounding of the squares
  have hres := sum_sq_compose (n := d.D) hd.d_le
    (a := fun j => (a j).toReal ^ 2) (x := fun j => (bxx j).toReal) (r := s1.toReal)
    (fun j _ => sq_nonneg _) (fun j hj => hb j hj) hrt
  refine ⟨hfinS, ?_⟩
  simpa only [rowSum] using hres


/-! ## The row sum over `BD` lanes -/

/-- `RowSum.sum_err` with `d.D` replaced by the padded block's width `d.BD`. -/
theorem sum_err_bd {d : Inst} (hd : d.ok) (fusion : Bool) {a : Nat → BF16} {bxx : Nat → F32}
    {s1 : F32}
    (hfin : ∀ j, j < d.BD → (bxx j).isFinite = true)
    (hb : ∀ j, j < d.BD →
      |(bxx j).toReal - (a j).toReal ^ 2|
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ))
    (hA : rowSum a d.BD + d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ))
    (he : (2 : ℝ) ^ (-125 : ℤ) ≤ d.eps.toReal)
    (htree : ∃ tr : Tree, tr.Covers d.BD ∧ tr.Eval (AddRel fusion) bxx s1) :
    s1.isFinite = true ∧
      |s1.toReal - rowSum a d.BD|
        ≤ (2 : ℝ) ^ (-13 : ℤ) * rowSum a d.BD + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := by
  obtain ⟨tr, hcov, hev⟩ := htree
  have hu : Format.binary32.u ≤ 1 / 16 := by
    rw [Format.binary32_u]; norm_num
  -- the row sum, the two `List.sum`s of `sumc_err_covers`, in `ℚ`
  set Aq : ℚ := ∑ j ∈ Finset.range d.BD, (a j).toRat ^ 2 with hAqdef
  set Bq : ℚ := ((List.range d.BD).map fun i => |(bxx i).toRat|).sum with hBqdef
  set Sq : ℚ := ((List.range d.BD).map fun i => (bxx i).toRat).sum with hSqdef
  -- ## `Aq ≤ 2^126 - 2^-125` and `eps ≥ 2^-125`, in `ℚ`
  have hrowSum_cast : rowSum a d.BD = (Aq : ℝ) := by
    rw [hAqdef]
    simp only [rowSum, FloatV.toReal]
    push_cast
    rfl
  have hAq : Aq + d.eps.toRat ≤ (2 : ℚ) ^ (126 : ℤ) := by
    have h := hA
    rw [hrowSum_cast] at h
    simp only [FloatV.toReal] at h
    exact_mod_cast h
  have heps : (2 : ℚ) ^ (-125 : ℤ) ≤ d.eps.toRat := by
    have h := he
    simp only [FloatV.toReal] at h
    refine Rat.cast_le (K := ℝ).mp ?_
    push_cast
    exact h
  have hAqle : Aq ≤ (2 : ℚ) ^ (126 : ℤ) - (2 : ℚ) ^ (-125 : ℤ) := by linarith
  -- ## each `|bxx j|` and the sum `Bq`
  have hper : ∀ j, j < d.BD → |(bxx j).toRat| ≤
      (1 + (2 : ℚ) ^ (-24 : ℤ)) * (a j).toRat ^ 2 + (2 : ℚ) ^ (-150 : ℤ) := by
    intro j hj
    have h1 : |(bxx j).toRat - (a j).toRat ^ 2| ≤
        (2 : ℚ) ^ (-24 : ℤ) * (a j).toRat ^ 2 + (2 : ℚ) ^ (-150 : ℤ) := by
      have h := hb j hj
      simp only [FloatV.toReal] at h
      refine Rat.cast_le (K := ℝ).mp ?_
      push_cast
      exact h
    have h2 : |(bxx j).toRat| ≤ |(a j).toRat ^ 2| + |(bxx j).toRat - (a j).toRat ^ 2| := by
      have h3 := abs_add_le ((a j).toRat ^ 2) ((bxx j).toRat - (a j).toRat ^ 2)
      rwa [add_sub_cancel] at h3
    rw [abs_of_nonneg (sq_nonneg ((a j).toRat))] at h2
    nlinarith
  have hBq : Bq ≤ (1 + (2 : ℚ) ^ (-24 : ℤ)) * Aq + (d.BD : ℚ) * (2 : ℚ) ^ (-150 : ℤ) := by
    rw [hBqdef, hAqdef, list_sum_range]
    have h := Finset.sum_le_sum (fun j hj => hper j (Finset.mem_range.mp hj))
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
      nsmul_eq_mul] at h
    exact h
  have hBq0 : 0 ≤ Bq := by
    rw [hBqdef, list_sum_range]
    exact Finset.sum_nonneg (fun i _ => abs_nonneg _)
  -- ## room for the tree: `Bq + errBound ≤ maxVal/2 - 2η`
  have hDle : (d.BD : ℚ) ≤ 512 := by exact_mod_cast hd.bd_le
  set X : ℚ := (2 : ℚ) ^ (126 : ℤ) + (2 : ℚ) ^ (102 : ℤ) + (2 : ℚ) ^ (-141 : ℤ) with hX
  have hBqle : Bq ≤ X := by
    have h1 : (1 + (2 : ℚ) ^ (-24 : ℤ)) * Aq ≤
        (1 + (2 : ℚ) ^ (-24 : ℤ)) * ((2 : ℚ) ^ (126 : ℤ) - (2 : ℚ) ^ (-125 : ℤ)) :=
      mul_le_mul_of_nonneg_left hAqle (by norm_num)
    have h2 : (d.BD : ℚ) * (2 : ℚ) ^ (-150 : ℤ) ≤ 512 * (2 : ℚ) ^ (-150 : ℤ) :=
      mul_le_mul_of_nonneg_right hDle (by norm_num)
    have h3 : (1 + (2 : ℚ) ^ (-24 : ℤ)) * ((2 : ℚ) ^ (126 : ℤ) - (2 : ℚ) ^ (-125 : ℤ)) +
        512 * (2 : ℚ) ^ (-150 : ℤ) ≤ X := by
      rw [hX]; norm_num
    exact hBq.trans ((add_le_add h1 h2).trans h3)
  have herrB := errBound_le hd.bd_le (B := Bq) hBq0
  have herrnum : Tree.errBound (3 * (2 : ℚ) ^ (-24 : ℤ)) (4 * (2 : ℚ) ^ (-150 : ℤ))
      (d.BD - 1) (d.BD - 1) Bq ≤ (3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * X + (2 : ℚ) ^ (-138 : ℤ) := by
    refine herrB.trans ?_
    have h := mul_le_mul_of_nonneg_left hBqle (by norm_num : (0 : ℚ) ≤ (3 / 4) * (2 : ℚ) ^ (-13 : ℤ))
    simpa only [add_comm] using add_le_add_right h ((2 : ℚ) ^ (-138 : ℤ))
  have hMnumeric : X + ((3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * X + (2 : ℚ) ^ (-138 : ℤ)) ≤
      ((2 : ℚ) ^ 24 - 1) * (2 : ℚ) ^ 104 / 2 - 2 * (2 : ℚ) ^ (-150 : ℤ) := by
    rw [hX]; norm_num
  have hstep : Bq + Tree.errBound (3 * (2 : ℚ) ^ (-24 : ℤ)) (4 * (2 : ℚ) ^ (-150 : ℤ))
      (d.BD - 1) (d.BD - 1) Bq
      ≤ X + ((3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * X + (2 : ℚ) ^ (-138 : ℤ)) :=
    add_le_add hBqle herrnum
  have hM : Bq + Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta)
      (d.BD - 1) (d.BD - 1) Bq ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
    exact hstep.trans hMnumeric
  -- ## the tree's error, from `sumc_err_covers`
  obtain ⟨hfinS, herr⟩ :=
    sumc_err_covers (f := Format.binary32) Format.binary32_regular (by decide) hu fusion bxx
      hcov hev hfin hM
  -- ## the same bound in `ℚ`, then in `ℝ` with `Finset.sum`s
  have hq : |s1.toRat - Sq| ≤ (3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * Bq + (2 : ℚ) ^ (-138 : ℤ) := by
    have h := herr
    rw [← hSqdef, ← hBqdef] at h
    refine h.trans ?_
    rw [Format.binary32_u, Format.binary32_eta]
    exact herrB
  have hqR : |s1.toReal - (Sq : ℝ)| ≤
      (3 / 4) * (2 : ℝ) ^ (-13 : ℤ) * (Bq : ℝ) + (2 : ℝ) ^ (-138 : ℤ) := by
    have hcast : ((|s1.toRat - Sq| : ℚ) : ℝ) ≤
        ((((3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * Bq + (2 : ℚ) ^ (-138 : ℤ)) : ℚ) : ℝ) :=
      Rat.cast_le (K := ℝ).mpr hq
    have hL : ((|s1.toRat - Sq| : ℚ) : ℝ) = |s1.toReal - (Sq : ℝ)| := by
      rw [Rat.cast_abs]
      simp only [FloatV.toReal]
      push_cast
      ring_nf
    have hR : ((((3 / 4) * (2 : ℚ) ^ (-13 : ℤ) * Bq + (2 : ℚ) ^ (-138 : ℤ)) : ℚ) : ℝ) =
        (3 / 4) * (2 : ℝ) ^ (-13 : ℤ) * (Bq : ℝ) + (2 : ℝ) ^ (-138 : ℤ) := by
      push_cast
      ring
    rwa [hL, hR] at hcast
  have hcastS : (Sq : ℝ) = ∑ j ∈ Finset.range d.BD, (bxx j).toReal := by
    rw [hSqdef, list_sum_range]
    simp only [FloatV.toReal]
    push_cast
    rfl
  have hcastB : (Bq : ℝ) = ∑ j ∈ Finset.range d.BD, |(bxx j).toReal| := by
    rw [hBqdef, list_sum_range]
    simp only [FloatV.toReal]
    push_cast
    rfl
  have hrt : |s1.toReal - ∑ j ∈ Finset.range d.BD, (bxx j).toReal| ≤
      (3 / 4) * (2 : ℝ) ^ (-13 : ℤ) * (∑ j ∈ Finset.range d.BD, |(bxx j).toReal|) +
        (2 : ℝ) ^ (-138 : ℤ) := by
    rw [← hcastS, ← hcastB]
    exact hqR
  -- ## compose with the per-element rounding of the squares
  have hres := sum_sq_compose (n := d.BD) hd.bd_le
    (a := fun j => (a j).toReal ^ 2) (x := fun j => (bxx j).toReal) (r := s1.toReal)
    (fun j _ => sq_nonneg _) (fun j hj => hb j hj) hrt
  refine ⟨hfinS, ?_⟩
  simpa only [rowSum] using hres

end Claims.L2NormFwd
