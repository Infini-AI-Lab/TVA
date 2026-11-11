import Proof.NumDefs

/-!
# `chunk_delta_h_bf16`: norms and sums

The Euclidean norm `colNorm` and the spectral norm `specNorm` of the claim: the triangle
inequality, Cauchy–Schwarz and the defining inequality of `specNorm`.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- A norm is nonnegative. -/
theorem colNorm_nonneg (n : ℕ) (x : ℕ → ℝ) : 0 ≤ colNorm n x := Real.sqrt_nonneg _

/-- A pointwise bound on magnitudes bounds the norm. -/
theorem colNorm_mono_abs {n : ℕ} {x y : ℕ → ℝ} (h : ∀ i, i < n → |x i| ≤ |y i|) :
    colNorm n x ≤ colNorm n y := by
  unfold colNorm
  exact Real.sqrt_le_sqrt (Finset.sum_le_sum fun i hi =>
    sq_le_sq.mpr (h i (Finset.mem_range.mp hi)))

/-- Cauchy–Schwarz. -/
theorem abs_sum_mul_le_colNorm (n : ℕ) (x y : ℕ → ℝ) :
    |∑ i ∈ Finset.range n, x i * y i| ≤ colNorm n x * colNorm n y := by
  have h := Finset.sum_mul_sq_le_sq_mul_sq (Finset.range n) x y
  unfold colNorm
  rw [← Real.sqrt_mul (Finset.sum_nonneg fun i _ => sq_nonneg _)]
  exact Real.abs_le_sqrt h

/-- The triangle inequality. -/
theorem colNorm_add_le (n : ℕ) (x y : ℕ → ℝ) :
    colNorm n (fun i => x i + y i) ≤ colNorm n x + colNorm n y := by
  have hx : 0 ≤ ∑ i ∈ Finset.range n, x i ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hy : 0 ≤ ∑ i ∈ Finset.range n, y i ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hcs := abs_sum_mul_le_colNorm n x y
  have hxs := Real.sq_sqrt hx
  have hys := Real.sq_sqrt hy
  have hsum : ∑ i ∈ Finset.range n, (x i + y i) ^ 2 =
      ∑ i ∈ Finset.range n, x i ^ 2 + 2 * ∑ i ∈ Finset.range n, x i * y i +
        ∑ i ∈ Finset.range n, y i ^ 2 := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  unfold colNorm
  rw [Real.sqrt_le_iff]
  refine ⟨by positivity, ?_⟩
  rw [hsum]
  have := le_abs_self (∑ i ∈ Finset.range n, x i * y i)
  unfold colNorm at hcs
  nlinarith

/-- The triangle inequality for a difference. -/
theorem colNorm_sub_le (n : ℕ) (x y : ℕ → ℝ) :
    colNorm n (fun i => x i - y i) ≤ colNorm n x + colNorm n y := by
  have h := colNorm_add_le n x (fun i => - y i)
  have hneg : colNorm n (fun i => - y i) = colNorm n y := by
    unfold colNorm; simp only [neg_sq]
  rw [hneg] at h
  simpa [sub_eq_add_neg] using h

/-- A scalar multiple. -/
theorem colNorm_smul (n : ℕ) (a : ℝ) (x : ℕ → ℝ) :
    colNorm n (fun i => a * x i) = |a| * colNorm n x := by
  unfold colNorm
  have : ∀ i, (a * x i) ^ 2 = a ^ 2 * x i ^ 2 := fun i => by ring
  simp only [this]
  rw [← Finset.mul_sum, Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq_eq_abs]

/-- A constant vector. -/
theorem colNorm_const (n : ℕ) (a : ℝ) : colNorm n (fun _ => a) = Real.sqrt n * |a| := by
  unfold colNorm
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  rw [Real.sqrt_mul (Nat.cast_nonneg _), Real.sqrt_sq_eq_abs]

/-- Cauchy–Schwarz for magnitudes. -/
theorem sum_abs_mul_le_colNorm (n : ℕ) (x y : ℕ → ℝ) :
    ∑ i ∈ Finset.range n, |x i| * |y i| ≤ colNorm n x * colNorm n y := by
  have h := abs_sum_mul_le_colNorm n (fun i => |x i|) (fun i => |y i|)
  have hx : colNorm n (fun i => |x i|) = colNorm n x := by unfold colNorm; simp only [sq_abs]
  have hy : colNorm n (fun i => |y i|) = colNorm n y := by unfold colNorm; simp only [sq_abs]
  rw [hx, hy] at h
  exact (le_abs_self _).trans (by simpa [abs_mul] using h)

/-- The Frobenius norm is in the set that `specNorm` is the infimum of. -/
theorem frob_mem_specSet (m n : ℕ) (A : ℕ → ℕ → ℝ) :
    Real.sqrt (∑ a ∈ Finset.range m, ∑ b ∈ Finset.range n, A a b ^ 2) ∈
      {c : ℝ | 0 ≤ c ∧ ∀ x : ℕ → ℝ,
        colNorm m (fun a => ∑ b ∈ Finset.range n, A a b * x b) ≤ c * colNorm n x} := by
  refine ⟨Real.sqrt_nonneg _, fun x => ?_⟩
  unfold colNorm
  rw [← Real.sqrt_mul (Finset.sum_nonneg fun a _ => Finset.sum_nonneg fun b _ => sq_nonneg _)]
  refine Real.sqrt_le_sqrt ?_
  rw [Finset.sum_mul]
  exact Finset.sum_le_sum fun a _ => Finset.sum_mul_sq_le_sq_mul_sq (Finset.range n) (A a) x

/-- The defining inequality of the spectral norm. -/
theorem specNorm_spec (m n : ℕ) (A : ℕ → ℕ → ℝ) (x : ℕ → ℝ) :
    colNorm m (fun a => ∑ b ∈ Finset.range n, A a b * x b) ≤ specNorm m n A * colNorm n x := by
  have hne := frob_mem_specSet m n A
  have hbdd : BddBelow {c : ℝ | 0 ≤ c ∧ ∀ x : ℕ → ℝ,
      colNorm m (fun a => ∑ b ∈ Finset.range n, A a b * x b) ≤ c * colNorm n x} :=
    ⟨0, fun c hc => hc.1⟩
  have hcn := colNorm_nonneg n x
  refine le_of_forall_pos_le_add fun ε hε => ?_
  have hpos : 0 < ε / (colNorm n x + 1) := div_pos hε (by linarith)
  obtain ⟨c, hc, hlt⟩ := exists_lt_of_csInf_lt ⟨_, hne⟩ (lt_add_of_pos_right (specNorm m n A) hpos)
  have h1 := hc.2 x
  have h2 : c * colNorm n x ≤ (specNorm m n A + ε / (colNorm n x + 1)) * colNorm n x :=
    mul_le_mul_of_nonneg_right hlt.le hcn
  have h3 : ε / (colNorm n x + 1) * colNorm n x ≤ ε := by
    rw [div_mul_eq_mul_div, div_le_iff₀ (by linarith)]
    nlinarith
  nlinarith

/-- The spectral norm is at most the Frobenius norm. -/
theorem specNorm_le_frob (m n : ℕ) (A : ℕ → ℕ → ℝ) :
    specNorm m n A ≤ Real.sqrt (∑ a ∈ Finset.range m, ∑ b ∈ Finset.range n, A a b ^ 2) := by
  exact csInf_le ⟨0, fun c hc => hc.1⟩ (frob_mem_specSet m n A)

/-- The spectral norm is nonnegative. -/
theorem specNorm_nonneg (m n : ℕ) (A : ℕ → ℕ → ℝ) : 0 ≤ specNorm m n A := by
  exact le_csInf ⟨_, frob_mem_specSet m n A⟩ fun c hc => hc.1

/-- The norm depends only on the first `n` entries. -/
theorem colNorm_congr {n : ℕ} {x y : ℕ → ℝ} (h : ∀ i, i < n → x i = y i) :
    colNorm n x = colNorm n y := by
  unfold colNorm
  exact congrArg Real.sqrt (Finset.sum_congr rfl fun i hi => by rw [h i (Finset.mem_range.mp hi)])

/-- Each entry is at most the norm. -/
theorem abs_le_colNorm {n : ℕ} (x : ℕ → ℝ) {i : ℕ} (hi : i < n) : |x i| ≤ colNorm n x := by
  unfold colNorm
  exact Real.abs_le_sqrt (Finset.single_le_sum (f := fun i => x i ^ 2) (fun j _ => sq_nonneg _)
    (Finset.mem_range.mpr hi))

/-- The norm of a vector bounded entrywise by `a·|y| + a'·|z| + b`. -/
theorem colNorm_lin2_le {n : ℕ} {x y z : ℕ → ℝ} {a a' b : ℝ} (ha : 0 ≤ a) (ha' : 0 ≤ a')
    (hb : 0 ≤ b) (h : ∀ i, i < n → |x i| ≤ a * |y i| + a' * |z i| + b) :
    colNorm n x ≤ a * colNorm n y + a' * colNorm n z + Real.sqrt n * b := by
  have hmono : colNorm n x ≤ colNorm n (fun i => (a * |y i| + a' * |z i|) + b) := by
    refine colNorm_mono_abs fun i hi => ?_
    have hpos : 0 ≤ a * |y i| + a' * |z i| + b := by positivity
    exact (h i hi).trans (abs_of_nonneg hpos).symm.le
  refine hmono.trans ?_
  refine (colNorm_add_le _ _ _).trans ?_
  have h1 := colNorm_add_le n (fun i => a * |y i|) (fun i => a' * |z i|)
  have e1 : colNorm n (fun i => a * |y i|) = a * colNorm n y := by
    rw [colNorm_smul, abs_of_nonneg ha]; congr 1; unfold colNorm; simp only [sq_abs]
  have e2 : colNorm n (fun i => a' * |z i|) = a' * colNorm n z := by
    rw [colNorm_smul, abs_of_nonneg ha']; congr 1; unfold colNorm; simp only [sq_abs]
  have e3 := colNorm_const n b
  rw [abs_of_nonneg hb] at e3
  rw [e1, e2] at h1
  linarith

/-- A rounding to nearest fits in the width. -/
theorem roundRat_lt {f : Format} (hf : f.Regular) (hs : f.specials ≠ .finite) (zs : Bool)
    {q : ℚ} : f.roundRat .rn false zs q < 2 ^ f.width := by
  by_cases hq0 : q = 0
  · subst hq0
    exact (Format.roundRat_zero_spec (f := f) (rnd := .rn) (sat := false) (zs := zs) hf).1
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨hN, hQ, hsub⟩ := Format.roundSig_bounds (f := f) .rn ha
  have hsetS : (if q < 0 then f.signBit else 0) = if decide (q < 0) then f.signBit else 0 := by simp
  by_cases hno : (Format.roundSig .rn |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|) ≤ f.maxVal
  · have hmag : f.roundMag .rn false |q| = f.pack (Format.roundSig .rn |q| (f.qexp |q|)) (f.qexp |q|) := by
      unfold Format.roundMag Format.encode; rw [Format.pow2_eq, ite_eq_left hno]
    have hle := Format.pack_le_maxMag hf hN hQ hsub hno
    have hlt := hle.trans_lt (Format.maxMag_lt hf)
    rw [Format.roundRat_of_ne hq0, hmag, hsetS, ← Format.setSign_of_lt hlt]
    exact Format.setSign_lt _ _
  · obtain ⟨hlt, -⟩ := Format.roundMag_rn_of_over hf hs hno
    rw [Format.roundRat_of_ne hq0, hsetS, ← Format.setSign_of_lt hlt]
    exact Format.setSign_lt _ _

/-- A finite `float32` value that rounds to a finite `bfloat16` is within the format's relative
rounding error plus its smallest increment. -/
theorem cvt_bf16_err_of_finite {y : F32} {h : BF16} (hy : y.isFinite = true)
    (hh : FloatV.NaNEq (FloatV.cvt y) h) (hhf : h.isFinite = true) :
    |h.toReal - y.toReal| ≤ 2 ^ (-8 : ℤ) * |y.toReal| + 2 ^ (-134 : ℤ) := by
  have hcf : (FloatV.cvt y : BF16).isFinite = true := by rw [hh.isFinite_eq]; exact hhf
  have heq : h = FloatV.cvt y := hh.eq_of_isFinite hcf
  have hu := Format.bfloat16_u_real
  have he := Format.bfloat16_eta_real
  have hmr := Format.bfloat16_maxVal_real
  subst heq
  by_cases hm : |y.toReal| ≤ (Format.bfloat16.maxVal : ℝ)
  · obtain ⟨-, h2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hy hm
    rw [hu, he] at h2
    linarith [mul_comm |y.toReal| ((2:ℝ) ^ (-8 : ℤ))]
  · rw [not_le] at hm
    have hpat : (FloatV.cvt y : BF16) = FloatV.ofPat
        (Format.bfloat16.roundRat .rn false (Format.binary32.sign y.pat) y.toRat) :=
      congrArg FloatV.ofPat (Format.cvt_of_finite _ _ _ hy)
    have hlt := roundRat_lt Format.bfloat16_regular (by decide) (Format.binary32.sign y.pat) (q := y.toRat)
    have hfin : Format.bfloat16.isFinite (Format.bfloat16.roundRat .rn false (Format.binary32.sign y.pat) y.toRat) = true := by
      rw [hpat, FloatV.isFinite_ofPat hlt] at hcf; exact hcf
    have herr := Format.roundRat_rn_err_of_finite Format.bfloat16_regular (by decide) (Format.binary32.sign y.pat) (q := y.toRat) hfin
    have habs := Format.abs_toRat_le_maxVal Format.bfloat16_regular hfin
    rw [hpat, FloatV.toReal_eq, FloatV.toRat_ofPat hlt]
    generalize Format.bfloat16.toRat _ = r at herr habs ⊢
    have h1 : |((r : ℚ) : ℝ) - ((y.toRat : ℚ) : ℝ)| ≤ |(r : ℝ)| * (Format.bfloat16.u : ℝ) + (Format.bfloat16.eta : ℝ) := by
      have := herr
      exact_mod_cast this
    have h2 : |(r : ℝ)| ≤ (Format.bfloat16.maxVal : ℝ) := by exact_mod_cast habs
    rw [hu, he] at h1
    have h3 : |(r : ℝ)| ≤ |y.toReal| := by linarith
    have hq : ((y.toRat : ℚ) : ℝ) = y.toReal := rfl
    rw [hq] at h1
    nlinarith [zpow_pos (by norm_num : (0:ℝ) < 2) (-8 : ℤ)]

end Claims.ChunkDeltaH
