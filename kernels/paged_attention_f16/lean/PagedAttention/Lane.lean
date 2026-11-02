import PagedAttention.Real

/-!
# `paged_attention_f16`: facts about single operations

* the IEEE operations on `float32` with one rounding's bound, in `ℝ` (`f32_mul`, `f32_sub`,
  `f32_mulAdd`), and the rounding to `float16` (`f16_cvt`);
* `exp_acc`: `Rules.Num.ExpF32` at the bound `|x|` of the input itself;
* `laneSum_err`, `laneSumc_err`: a reduction of a block's lanes along any tree (IEEE or fusable
  additions), against the exact sum.
-/

namespace Kernels.PagedAttention

open TVA TVA.Math.Float

theorem toReal_def {f : Format} (x : FloatV f) : x.toReal = (x.toRat : ℝ) := rfl

theorem maxVal32 : (Format.binary32.maxVal : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 := by
  rw [Format.binary32_maxVal]; push_cast; ring

theorem maxVal16 : (Format.binary16.maxVal : ℝ) = 65504 := by
  rw [Format.binary16_maxVal]; norm_num

theorem u32_eq : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u

theorem eta32_eq : Format.binary32.eta = 2 ^ (-150 : ℤ) := Format.binary32_eta

theorem z24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
theorem z150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num

theorem isNaN_of_fin {f : Format} {x : FloatV f} (h : x.isFinite = true) : x.isNaN = false :=
  (FloatV.isFinite_iff.mp h).1

/-- **`x * y`** in `float32`, the result up to NaN payload. -/
theorem f32_mul {x y r : F32} (h : FloatV.NaNEq (FloatV.mul x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hroom : |x.toReal * y.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧ |r.toReal - x.toReal * y.toReal| ≤ |x.toReal * y.toReal| / 2 ^ 24 + 1 / 2 ^ 150 := by
  have hroom' : |x.toRat * y.toRat| ≤ Format.binary32.maxVal := by
    have : ((|x.toRat * y.toRat| : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      rw [maxVal32]; push_cast; rw [← toReal_def, ← toReal_def]; linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.mul_err Format.binary32_regular hx hy hroom'
  rw [h.eq (FloatV.isFinite_iff.mp hf).1]
  refine ⟨hf, ?_⟩
  rw [u32_eq, eta32_eq] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  linarith [abs_nonneg ((x.toRat : ℝ) * y.toRat)]

/-- **`x − y`** in `float32`. -/
theorem f32_sub {x y r : F32} (h : FloatV.NaNEq (FloatV.sub x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hroom : |x.toReal - y.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧ |r.toReal - (x.toReal - y.toReal)| ≤ |x.toReal - y.toReal| / 2 ^ 24 + 1 / 2 ^ 150 := by
  have hroom' : |x.toRat - y.toRat| ≤ Format.binary32.maxVal := by
    have : ((|x.toRat - y.toRat| : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      rw [maxVal32]; push_cast; rw [← toReal_def, ← toReal_def]; linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.sub_err Format.binary32_regular hx hy hroom'
  rw [h.eq (FloatV.isFinite_iff.mp hf).1]
  refine ⟨hf, ?_⟩
  rw [u32_eq, eta32_eq] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  linarith [abs_nonneg ((x.toRat : ℝ) - y.toRat)]

/-- **`a * b + c`** (`tl.mul_add`) in `float32`. -/
theorem f32_mulAdd {fusion : Bool} {a b c r : F32}
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul a b) c) r ∨ (fusion = true ∧ FloatV.NaNEq (FloatV.fma a b c) r))
    (ha : a.isFinite = true) (hb : b.isFinite = true) (hc : c.isFinite = true)
    (hroom : |a.toReal * b.toReal| + |c.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧ |r.toReal - (a.toReal * b.toReal + c.toReal)| ≤
      3 * (|a.toReal * b.toReal| + |c.toReal|) / 2 ^ 24 + 4 / 2 ^ 150 := by
  have hroom' : |a.toRat * b.toRat| + |c.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have : ((|a.toRat * b.toRat| + |c.toRat| : ℚ) : ℝ) ≤
        ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      rw [eta32_eq]; push_cast; rw [maxVal32, ← toReal_def, ← toReal_def, ← toReal_def]; norm_num
      have := abs_mul a.toReal b.toReal
      linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.mulAdd_err Format.binary32_regular (by rw [u32_eq]; norm_num) h ha hb hc hroom'
  refine ⟨hf, ?_⟩
  rw [u32_eq, eta32_eq] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  have := abs_mul (a.toRat : ℝ) (b.toRat : ℝ)
  linarith

/-- **`x.to(float16)`** of a `float32` within `float16`'s range. -/
theorem f16_cvt {x : F32} {r : F16} (h : FloatV.NaNEq (FloatV.cvt x) r) (hx : x.isFinite = true)
    (hroom : |x.toReal| ≤ 65504) :
    r.isFinite = true ∧ |r.toReal - x.toReal| ≤ |x.toReal| / 2 ^ 11 + 1 / 2 ^ 25 := by
  have hroom' : |x.toRat| ≤ Format.binary16.maxVal := by
    have : ((|x.toRat| : ℚ) : ℝ) ≤ (Format.binary16.maxVal : ℝ) := by
      rw [maxVal16]; push_cast; rw [← toReal_def]; exact hroom
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.cvt_err (g := Format.binary16) Format.binary16_regular hx hroom'
  rw [h.eq (FloatV.isFinite_iff.mp hf).1]
  refine ⟨hf, ?_⟩
  rw [Format.binary16_u, Format.binary16_eta] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  linarith [abs_nonneg (x.toRat : ℝ)]

/-- **`tl.exp`** on `float32` (`Rules.Num.ExpF32` at the bound `|x|`), at a finite input
`x ≤ 1` with `|x| ≤ 2^12`. -/
theorem exp_acc {P : Platform} (h : Rules.Num.ExpF32 P) {x r : F32}
    (hr : P.math .exp .binary32 x r) (hx : x.isFinite = true) (hx1 : x.toReal ≤ 1)
    (hx2 : |x.toReal| ≤ 2 ^ 12) :
    r.isFinite = true ∧ |r.toReal - Real.exp x.toReal| ≤
      (1 / 2 ^ 22 + |x.toReal| / 2 ^ 23) * Real.exp x.toReal + 1 / 2 ^ 140 := by
  obtain ⟨g, hg, hacc⟩ := h |x.toReal| (abs_nonneg _)
  have hg' : g = Real.exp := by
    have e : MathFn.ref .exp = some Real.exp := rfl
    rw [e] at hg; exact (Option.some.inj hg).symm
  subst hg'
  have hp := Real.exp_pos x.toReal
  have he1 : Real.exp x.toReal ≤ Real.exp 1 := Real.exp_le_exp.mpr hx1
  have he3 : Real.exp 1 < 3 := lt_trans Real.exp_one_lt_d9 (by norm_num)
  have hroom : (1 + ((2 : ℝ) ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |x.toReal|)) * |Real.exp x.toReal| +
      2 ^ (-140 : ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
    rw [maxVal32, abs_of_pos hp]
    have h1 : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 := by norm_num
    have h2 : (2 : ℝ) ^ (-23 : ℤ) * |x.toReal| ≤ 1 := by
      have : (2 : ℝ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
      rw [this]; nlinarith [abs_nonneg x.toReal]
    have h3 : (2 : ℝ) ^ (-140 : ℤ) ≤ 1 := by norm_num
    have h4 : (0 : ℝ) ≤ 2 ^ (-22 : ℤ) := by positivity
    have h5 : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) * |x.toReal| := by positivity
    nlinarith
  obtain ⟨hf, he⟩ := hacc x r hr hx ⟨neg_abs_le _, le_abs_self _⟩ (Set.mem_univ _) hroom
  refine ⟨hf, ?_⟩
  rw [abs_of_pos hp] at he
  have e1 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
  have e2 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 2 ^ 23 := by norm_num
  have e3 : (2 : ℝ) ^ (-140 : ℤ) = 1 / 2 ^ 140 := by norm_num
  rw [e1, e2, e3] at he
  have : (1 / 2 ^ 22 + 1 / 2 ^ 23 * |x.toReal|) * Real.exp x.toReal =
      (1 / 2 ^ 22 + |x.toReal| / 2 ^ 23) * Real.exp x.toReal := by ring
  linarith

/-! ## Reductions of a block's lanes -/

/-- A block's lanes as a sequence (a reduction's leaves). -/
def laneSeq {n : ℕ} (p : Fin n → F32) (j : ℕ) : F32 := if h : j < n then p ⟨j, h⟩ else default

theorem list_sum_range {n : ℕ} (f : ℕ → ℚ) : ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

theorem sum_laneSeq {n : ℕ} (p : Fin n → F32) (g : F32 → ℚ) :
    ((List.range n).map fun i => g (laneSeq p i)).sum = ∑ jj : Fin n, g (p jj) := by
  rw [list_sum_range, Finset.sum_range (fun i => g (laneSeq p i))]
  refine Finset.sum_congr rfl fun jj _ => ?_
  simp [laneSeq, jj.2]

theorem pow255_le {n : ℕ} (hn : n ≤ 256) : ((1 : ℝ) + 3 / 2 ^ 24) ^ (n - 1) ≤ 1 + 1 / 2 ^ 12 := by
  have h := pow_le_of_step (s := 1 + 3 / 2 ^ 24) (n := n - 1) (N := 255) (c := 3 / 2 ^ 24)
    (by norm_num) le_rfl (by omega) (by norm_num)
  refine h.trans ?_
  norm_num

theorem pow255_le' {n : ℕ} (hn : n ≤ 256) : ((1 : ℝ) + 1 / 2 ^ 24) ^ (n - 1) ≤ 1 + 1 / 2 ^ 12 := by
  have h := pow_le_of_step (s := 1 + 1 / 2 ^ 24) (n := n - 1) (N := 255) (c := 1 / 2 ^ 24)
    (by norm_num) le_rfl (by omega) (by norm_num)
  refine h.trans ?_
  norm_num

/-- **`tl.sum` of a block's lanes** (`n ≤ 256`), IEEE additions along any tree. -/
theorem laneSum_err {n : ℕ} (hn : n ≤ 256) {p : Fin n → F32} {r : F32}
    (h : ∃ tr : Tree, tr.Covers n ∧ tr.Eval (AddRel false) (laneSeq p) r)
    (hf : ∀ jj, (p jj).isFinite = true) (hA : ∑ jj, |(p jj).toReal| ≤ 2 ^ 100) :
    r.isFinite = true ∧ |r.toReal - ∑ jj, (p jj).toReal| ≤
      ((1 + 1 / 2 ^ 24) ^ (n - 1) - 1) * ∑ jj, |(p jj).toReal| +
        ((n - 1 : ℕ) : ℝ) / 2 ^ 150 * (1 + 1 / 2 ^ 24) ^ (n - 1) := by
  obtain ⟨tr, hcov, hev⟩ := h
  have hx : ∀ i, i < n → (laneSeq p i).isFinite = true := fun i hi => by simp [laneSeq, hi, hf]
  have hS1 := sum_laneSeq p (fun x => |x.toRat|)
  have hS2 := sum_laneSeq p (fun x => x.toRat)
  have hK : ((1 : ℝ) + 1 / 2 ^ 24) ^ (n - 1) ≤ 2 := (pow255_le' hn).trans (by norm_num)
  have hn1 : ((n - 1 : ℕ) : ℝ) ≤ 256 := by exact_mod_cast (show n - 1 ≤ 256 by omega)
  have hM : ((List.range n).map fun i => |(laneSeq p i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta (n - 1) (n - 1)
        ((List.range n).map fun i => |(laneSeq p i).toRat|).sum ≤ Format.binary32.maxVal := by
    rw [hS1]
    have : (((∑ jj, |(p jj).toRat|) + Tree.errBound Format.binary32.u Format.binary32.eta (n - 1) (n - 1)
        (∑ jj, |(p jj).toRat|) : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      rw [maxVal32]; unfold Tree.errBound; rw [u32_eq, eta32_eq]
      push_cast
      rw [z24, z150]
      simp only [← toReal_def]
      have hA0 : 0 ≤ ∑ jj, |(p jj).toReal| := Finset.sum_nonneg fun _ _ => abs_nonneg _
      have hK1 : (1 : ℝ) ≤ ((1 : ℝ) + 1 / 2 ^ 24) ^ (n - 1) := one_le_pow₀ (by norm_num)
      generalize ((1 : ℝ) + 1 / 2 ^ 24) ^ (n - 1) = K at hK hK1 ⊢
      have : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
      nlinarith
    exact_mod_cast this
  obtain ⟨hrf, herr⟩ := sum_err_covers Format.binary32_regular (laneSeq p) hcov hev hx hM
  refine ⟨hrf, ?_⟩
  rw [hS1, hS2] at herr
  unfold Tree.errBound at herr
  rw [u32_eq, eta32_eq] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  rw [z24, z150] at this
  simp only [← toReal_def] at this
  have e : (1 / 2 ^ 150 : ℝ) * ((n - 1 : ℕ) : ℝ) * (1 + 1 / 2 ^ 24) ^ (n - 1) =
      ((n - 1 : ℕ) : ℝ) / 2 ^ 150 * (1 + 1 / 2 ^ 24) ^ (n - 1) := by ring
  linarith

/-- **A reduction of products** (`tl.sum1c`, `tl.sum0c`) over `n ≤ 256` lanes, fusable additions
along any tree. -/
theorem laneSumc_err {n : ℕ} (hn : n ≤ 256) {fusion : Bool} {p : Fin n → F32} {r : F32}
    (h : ∃ tr : Tree, tr.Covers n ∧ tr.Eval (AddRel fusion) (laneSeq p) r)
    (hf : ∀ jj, (p jj).isFinite = true) (hA : ∑ jj, |(p jj).toReal| ≤ 2 ^ 100) :
    r.isFinite = true ∧ |r.toReal - ∑ jj, (p jj).toReal| ≤
      ((1 + 3 / 2 ^ 24) ^ (n - 1) - 1) * ∑ jj, |(p jj).toReal| +
        4 * ((n - 1 : ℕ) : ℝ) / 2 ^ 150 * (1 + 3 / 2 ^ 24) ^ (n - 1) := by
  obtain ⟨tr, hcov, hev⟩ := h
  have hx : ∀ i, i < n → (laneSeq p i).isFinite = true := fun i hi => by simp [laneSeq, hi, hf]
  have hS1 := sum_laneSeq p (fun x => |x.toRat|)
  have hS2 := sum_laneSeq p (fun x => x.toRat)
  have hK : ((1 : ℝ) + 3 / 2 ^ 24) ^ (n - 1) ≤ 2 := (pow255_le hn).trans (by norm_num)
  have hn1 : ((n - 1 : ℕ) : ℝ) ≤ 256 := by exact_mod_cast (show n - 1 ≤ 256 by omega)
  have hM : ((List.range n).map fun i => |(laneSeq p i).toRat|).sum +
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (n - 1) (n - 1)
        ((List.range n).map fun i => |(laneSeq p i).toRat|).sum ≤
        Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [hS1]
    have : (((∑ jj, |(p jj).toRat|) + Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta)
        (n - 1) (n - 1) (∑ jj, |(p jj).toRat|) : ℚ) : ℝ) ≤
        ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      unfold Tree.errBound; rw [u32_eq, eta32_eq]
      push_cast
      rw [maxVal32, z24, z150]
      simp only [← toReal_def]
      have hA0 : 0 ≤ ∑ jj, |(p jj).toReal| := Finset.sum_nonneg fun _ _ => abs_nonneg _
      have e3 : (1 : ℝ) + 3 * (1 / 2 ^ 24) = 1 + 3 / 2 ^ 24 := by ring
      rw [e3]
      have hK1 : (1 : ℝ) ≤ ((1 : ℝ) + 3 / 2 ^ 24) ^ (n - 1) := one_le_pow₀ (by norm_num)
      generalize ((1 : ℝ) + 3 / 2 ^ 24) ^ (n - 1) = K at hK hK1 ⊢
      nlinarith
    exact_mod_cast this
  obtain ⟨hrf, herr⟩ := sumc_err_covers Format.binary32_regular (by decide) (by rw [u32_eq]; norm_num)
    fusion (laneSeq p) hcov hev hx hM
  refine ⟨hrf, ?_⟩
  rw [hS1, hS2] at herr
  unfold Tree.errBound at herr
  rw [u32_eq, eta32_eq] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  rw [z24, z150] at this
  simp only [← toReal_def] at this
  have e3 : (1 : ℝ) + 3 * (1 / 2 ^ 24) = 1 + 3 / 2 ^ 24 := by ring
  rw [e3] at this
  have e : 4 * (1 / 2 ^ 150 : ℝ) * ((n - 1 : ℕ) : ℝ) * (1 + 3 / 2 ^ 24) ^ (n - 1) =
      4 * ((n - 1 : ℕ) : ℝ) / 2 ^ 150 * (1 + 3 / 2 ^ 24) ^ (n - 1) := by ring
  linarith

end Kernels.PagedAttention
