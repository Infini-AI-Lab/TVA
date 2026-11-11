import Proof.Num

/-!
# Float facts, part 2: the block sums, the state updates, the division, the output

Each lemma turns what the rules `safe_*` hand a walk (a tree of additions, a platform quotient,
a rounding) into a bound between real numbers.
-/
namespace Claims.PagedAttention

namespace Num

open TVA Finset

/-- Lane `j` of a `float32` vector as a real (`0` beyond the vector). -/
noncomputable def phr {Pg : ℕ} (pp : Tensor [Pg] F32) (j : ℕ) : ℝ :=
  if h : j < Pg then (pp.get1 ⟨j, h⟩).toReal else 0

/-- Entry `(j, e)` of a `float32` matrix as a real (`0` beyond it). -/
noncomputable def mr {Pg Dm : ℕ} (vv : Tensor [Pg, Dm] F32) (j e : ℕ) : ℝ :=
  if h : j < Pg then if h' : e < Dm then (vv.get2 ⟨j, h⟩ ⟨e, h'⟩).toReal else 0 else 0

/-- `float32`'s `u` as a real is `2^-24`. -/
theorem u32_real : ((Format.binary32.u : ℚ) : ℝ) = u := by
  rw [Format.binary32_u_real]; unfold u; norm_num
/-- `float32`'s `η` as a real is `2^-150`. -/
theorem eta32_real : ((Format.binary32.eta : ℚ) : ℝ) = 1 / 2 ^ 150 := by
  rw [Format.binary32_eta_real]; norm_num
/-- `float32`'s largest finite value as a real. -/
theorem max32_real : ((Format.binary32.maxVal : ℚ) : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 :=
  Format.binary32_maxVal_real

/-- A list sum over `range n` is the finset sum. -/
theorem list_range_sum (f : ℕ → ℚ) (n : ℕ) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]; simp

/-- The depth of every admissible tree is at most `8` for `n ≤ 256`. -/
theorem depth_le_eight {E : Env} (hP : Device E.P) {n : ℕ} (hn : n ≤ 256) {t : Tree}
    (hd : t.depth ≤ tl.sumBound E.P n) : t.depth ≤ 8 := by
  have h1 := hP.sumDepth n
  have h2 : Nat.clog 2 n ≤ 8 := by
    calc Nat.clog 2 n ≤ Nat.clog 2 256 := Nat.clog_mono_right 2 hn
      _ = 8 := by decide
  unfold tl.sumBound at hd
  omega

/-- `(1 + u)^8 ≤ 1 + 8.1u`: eight IEEE additions. -/
theorem pow8_le : (1 + u) ^ 8 ≤ 1 + 81 / 10 * u := by unfold u; norm_num

/-- The `float32` `+0` has value `0`. -/
theorem zero32_rat : (FloatV.ofPat 0 : F32).toRat = 0 := by decide +kernel
/-- The `float32` `+0` is finite. -/
theorem zero32_fin : (FloatV.ofPat 0 : F32).isFinite = true := by decide +kernel

/-- **The block sum of the weights** (`tl.sum`, IEEE additions along a tree of depth at most
`tl.sumBound`, `Pg ≤ 256` lanes): lanes `i < c` hold finite weights of magnitude at most `2`, the
others `+0`. The sum is finite, at most `1024` in magnitude, and within
`rhoSL·Σ|p̂| + etaS` of `Σ p̂`. -/
theorem lsum_block {E : Env} (hP : Device E.P) {Pg c : ℕ} (hPg : 0 < Pg) (hPg' : Pg ≤ 256)
    (hc : c ≤ Pg) (pp : Tensor [Pg] F32) (r : F32)
    (hfin : ∀ i : Fin Pg, i.val < c → (pp.get1 i).isFinite = true ∧ |(pp.get1 i).toReal| ≤ 2)
    (hz : ∀ i : Fin Pg, c ≤ i.val → pp.get1 i = FloatV.ofPat 0)
    (hr : ∃ t : Tree, t.Covers Pg ∧ t.depth ≤ tl.sumBound E.P Pg ∧
      t.Eval (AddRel false) (fun i => if h : i < Pg then pp.get1 ⟨i, h⟩ else default) r) :
    r.isFinite = true ∧ |r.toReal| ≤ 1024 ∧
      |r.toReal - ∑ j ∈ range c, phr pp j| ≤ rhoSL * ∑ j ∈ range c, |phr pp j| + etaS := by
  obtain ⟨t, hcov, hdep, heval⟩ := hr
  have hd8 := depth_le_eight hP hPg' hdep
  set x : ℕ → F32 := fun i => if h : i < Pg then pp.get1 ⟨i, h⟩ else default with hx
  have hxfin : ∀ i, i < Pg → (x i).isFinite = true := by
    intro i hi
    simp only [hx, dite_eq_left hi]
    by_cases hic : i < c
    · exact (hfin ⟨i, hi⟩ hic).1
    · rw [hz ⟨i, hi⟩ (by simpa using hic)]; exact zero32_fin
  have hxq : ∀ i, i < Pg → ((x i).toRat : ℝ) = if i < c then phr pp i else 0 := by
    intro i hi
    by_cases hic : i < c
    · rw [ite_eq_left hic]; simp only [hx, phr, dite_eq_left hi]; rfl
    · rw [ite_eq_right hic]
      simp only [hx, dite_eq_left hi]
      rw [hz ⟨i, hi⟩ (by simpa using hic), zero32_rat]; simp
  have hxb : ∀ i, i < Pg → |((x i).toRat : ℝ)| ≤ if i < c then 2 else 0 := by
    intro i hi
    rw [hxq i hi]
    by_cases hic : i < c
    · rw [ite_eq_left hic, ite_eq_left hic]
      simp only [phr, dite_eq_left hi]
      exact (hfin ⟨i, hi⟩ hic).2
    · simp [hic]
  -- the sums
  have hsum : ((∑ i ∈ range Pg, (x i).toRat : ℚ) : ℝ) = ∑ j ∈ range c, phr pp j := by
    push_cast
    rw [← Finset.sum_subset (Finset.range_subset_range.mpr hc)]
    · refine Finset.sum_congr rfl fun j hj => ?_
      have := hxq j (by have := Finset.mem_range.mp hj; omega)
      rw [ite_eq_left (Finset.mem_range.mp hj)] at this
      exact this
    · intro j hj hjc
      have := hxq j (Finset.mem_range.mp hj)
      rw [ite_eq_right (by simpa using hjc)] at this
      exact this
  have habs : ((∑ i ∈ range Pg, |(x i).toRat| : ℚ) : ℝ) = ∑ j ∈ range c, |phr pp j| := by
    push_cast
    rw [← Finset.sum_subset (Finset.range_subset_range.mpr hc)]
    · refine Finset.sum_congr rfl fun j hj => ?_
      have := hxq j (by have := Finset.mem_range.mp hj; omega)
      rw [ite_eq_left (Finset.mem_range.mp hj)] at this
      rw [this]
    · intro j hj hjc
      have := hxq j (Finset.mem_range.mp hj)
      rw [ite_eq_right (by simpa using hjc)] at this
      rw [this]; simp
  have habs2 : (∑ j ∈ range c, |phr pp j|) ≤ 512 := by
    rw [← habs]
    push_cast
    calc ∑ i ∈ range Pg, |((x i).toRat : ℝ)| ≤ ∑ i ∈ range Pg, (2 : ℝ) := by
          refine Finset.sum_le_sum fun i hi => ?_
          have := hxb i (Finset.mem_range.mp hi)
          split_ifs at this <;> linarith
      _ = 2 * Pg := by simp [mul_comm]
      _ ≤ 512 := by
          have : (Pg : ℝ) ≤ 256 := by exact_mod_cast hPg'
          linarith
  have hPgR : ((Pg - 1 : ℕ) : ℝ) ≤ 255 := by
    have : Pg - 1 ≤ 255 := by omega
    exact_mod_cast this
  have hu0 := u_pos
  have hp8 := pow8_le
  have hp8' : 1 ≤ (1 + u) ^ 8 := one_le_pow₀ (by linarith)
  have hEB : ∀ A : ℚ, 0 ≤ A → (A : ℝ) ≤ 512 →
      ((Tree.errBound Format.binary32.u Format.binary32.eta 8 (Pg - 1) A : ℚ) : ℝ) ≤
        81 / 10 * u * 512 + 1 / 2 ^ 140 := by
    intro A hA0 hA
    unfold Tree.errBound
    push_cast
    rw [u32_real, eta32_real]
    have h1 : ((1 + u) ^ 8 - 1) * (A : ℝ) ≤ 81 / 10 * u * 512 := by
      have : (1 + u) ^ 8 - 1 ≤ 81 / 10 * u := by linarith
      calc ((1 + u) ^ 8 - 1) * (A : ℝ) ≤ (81 / 10 * u) * 512 :=
            mul_le_mul this hA (by exact_mod_cast hA0) (by positivity)
        _ = 81 / 10 * u * 512 := rfl
    have h2 : (1 / 2 ^ 150 : ℝ) * ((Pg - 1 : ℕ) : ℝ) * (1 + u) ^ 8 ≤ 1 / 2 ^ 140 := by
      have h8 : (1 + u) ^ 8 ≤ 2 := by unfold u; norm_num
      have : (1 / 2 ^ 150 : ℝ) * ((Pg - 1 : ℕ) : ℝ) ≤ (1 / 2 ^ 150) * 255 :=
        mul_le_mul_of_nonneg_left hPgR (by norm_num)
      calc (1 / 2 ^ 150 : ℝ) * ((Pg - 1 : ℕ) : ℝ) * (1 + u) ^ 8 ≤ (1 / 2 ^ 150 * 255) * 2 :=
            mul_le_mul this h8 (by positivity) (by positivity)
        _ ≤ 1 / 2 ^ 140 := by norm_num
    linarith
  have hAq : (0 : ℚ) ≤ ∑ i ∈ range Pg, |(x i).toRat| := Finset.sum_nonneg fun i _ => abs_nonneg _
  have hAr : ((∑ i ∈ range Pg, |(x i).toRat| : ℚ) : ℝ) ≤ 512 := by rw [habs]; exact habs2
  have hM : ((List.range Pg).map fun i => |(x i).toRat|).sum +
      Tree.errBound Format.binary32.u Format.binary32.eta 8 (Pg - 1)
        ((List.range Pg).map fun i => |(x i).toRat|).sum ≤ Format.binary32.maxVal := by
    rw [list_range_sum]
    rw [← Rat.cast_le (K := ℝ)]
    push_cast at hAr ⊢
    have := hEB _ hAq (by exact_mod_cast hAr)
    rw [max32_real]
    have hu8 : (81 / 10 * u * 512 : ℝ) ≤ 1 := by unfold u; norm_num
    have h140 : (1 / 2 ^ 140 : ℝ) ≤ 1 := by norm_num
    have : (512 : ℝ) + 1 + 1 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hrfin, herr⟩ := sum_err_depth Format.binary32_regular x hcov hd8 heval hxfin hM
  rw [list_range_sum, list_range_sum] at herr
  have herr' : |r.toReal - ∑ j ∈ range c, phr pp j| ≤
      ((1 + u) ^ 8 - 1) * ∑ j ∈ range c, |phr pp j| + 1 / 2 ^ 150 * ((Pg - 1 : ℕ) : ℝ) * (1 + u) ^ 8 := by
    have h1 : ((|r.toRat - ∑ i ∈ range Pg, (x i).toRat| : ℚ) : ℝ) ≤
        ((Tree.errBound Format.binary32.u Format.binary32.eta 8 (Pg - 1)
          (∑ i ∈ range Pg, |(x i).toRat|) : ℚ) : ℝ) := by exact_mod_cast herr
    unfold Tree.errBound at h1
    push_cast at h1
    have h2 : ((∑ i ∈ range Pg, (x i).toRat : ℚ) : ℝ) = ∑ j ∈ range c, phr pp j := hsum
    have h3 : ((∑ i ∈ range Pg, |(x i).toRat| : ℚ) : ℝ) = ∑ j ∈ range c, |phr pp j| := habs
    push_cast at h2 h3
    rw [u32_real, eta32_real, h2, h3] at h1
    unfold FloatV.toReal
    exact h1
  have hA0 : 0 ≤ ∑ j ∈ range c, |phr pp j| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hdiff : (1 + u) ^ 8 - 1 ≤ 81 / 10 * u := by linarith
  have herrB : |r.toReal - ∑ j ∈ range c, phr pp j| ≤
      81 / 10 * u * ∑ j ∈ range c, |phr pp j| + 1 / 2 ^ 140 := by
    refine herr'.trans ?_
    have h1 : ((1 + u) ^ 8 - 1) * ∑ j ∈ range c, |phr pp j| ≤ 81 / 10 * u * ∑ j ∈ range c, |phr pp j| :=
      mul_le_mul_of_nonneg_right hdiff hA0
    have h8 : (1 + u) ^ 8 ≤ 2 := by unfold u; norm_num
    have h2 : (1 / 2 ^ 150 : ℝ) * ((Pg - 1 : ℕ) : ℝ) * (1 + u) ^ 8 ≤ 1 / 2 ^ 140 := by
      have : (1 / 2 ^ 150 : ℝ) * ((Pg - 1 : ℕ) : ℝ) ≤ (1 / 2 ^ 150) * 255 :=
        mul_le_mul_of_nonneg_left hPgR (by norm_num)
      calc (1 / 2 ^ 150 : ℝ) * ((Pg - 1 : ℕ) : ℝ) * (1 + u) ^ 8 ≤ (1 / 2 ^ 150 * 255) * 2 :=
            mul_le_mul this h8 (by positivity) (by positivity)
        _ ≤ 1 / 2 ^ 140 := by norm_num
    linarith
  refine ⟨hrfin, ?_, ?_⟩
  · have hS : |∑ j ∈ range c, phr pp j| ≤ 512 :=
      (Finset.abs_sum_le_sum_abs _ _).trans habs2
    have hu8 : (81 / 10 * u * ∑ j ∈ range c, |phr pp j|) ≤ 1 := by
      have : 81 / 10 * u * ∑ j ∈ range c, |phr pp j| ≤ 81 / 10 * u * 512 :=
        mul_le_mul_of_nonneg_left habs2 (by positivity)
      have : (81 / 10 * u * 512 : ℝ) ≤ 1 := by unfold u; norm_num
      linarith
    have h140 : (1 / 2 ^ 140 : ℝ) ≤ 1 := by norm_num
    have := abs_sub_abs_le_abs_sub r.toReal (∑ j ∈ range c, phr pp j)
    linarith
  · unfold rhoSL etaS
    have h140 : (1 / 2 ^ 140 : ℝ) ≤ 2 ^ (-120 : ℤ) := by norm_num
    linarith

/-- `(1 + 3u)^8 ≤ 1 + 24.1u`: eight fusable additions. -/
theorem pow8_3_le : (1 + 3 * u) ^ 8 ≤ 1 + 241 / 10 * u := by unfold u; norm_num

/-- **The block sum of the products** `p̂·v` (`tl.mul`, then `tl.sum0c`: fusable additions along
a tree of depth at most `tl.sumBound` per column, `Pg ≤ 256` rows): for the rows `i < c`, `p̂`
is finite of magnitude at most `2` and `v` is finite of magnitude at most `2^15`; the other rows
have `p̂ = +0` and `v` finite. Every column's sum is finite, at most `2^25` in magnitude, and
within `rhoSA·Σ|p̂·v| + etaS` of `Σ p̂·v`. -/
theorem acc_block {E : Env} (hP : Device E.P) {Pg Dm c : ℕ} (hPg : 0 < Pg) (hPg' : Pg ≤ 256)
    (hc : c ≤ Pg) (pp : Tensor [Pg] F32) (vv prod : Tensor [Pg, Dm] F32) (r : Tensor [Dm] F32)
    (hfin : ∀ i : Fin Pg, i.val < c → (pp.get1 i).isFinite = true ∧ |(pp.get1 i).toReal| ≤ 2)
    (hz : ∀ i : Fin Pg, c ≤ i.val → pp.get1 i = FloatV.ofPat 0)
    (hv : ∀ (i : Fin Pg) (e : Fin Dm), (vv.get2 i e).isFinite = true ∧
      (i.val < c → |(vv.get2 i e).toReal| ≤ 2 ^ 15))
    (hprod : ∀ (i : Fin Pg) (e : Fin Dm),
      FloatV.NaNEq (FloatV.mul (pp.get1 i) (vv.get2 i e)) (prod.get2 i e))
    (hr : ∀ e : Fin Dm, ∃ t : Tree, t.Covers Pg ∧ t.depth ≤ tl.sumBound E.P Pg ∧
      t.Eval (AddRel E.fusion) (fun i => if h : i < Pg then prod.get2 ⟨i, h⟩ e else default)
        (r.get1 e)) :
    ∀ e : Fin Dm, (r.get1 e).isFinite = true ∧ |(r.get1 e).toReal| ≤ 2 ^ 25 ∧
      |(r.get1 e).toReal - ∑ j ∈ range c, phr pp j * mr vv j e| ≤
        rhoSA * ∑ j ∈ range c, |phr pp j * mr vv j e| + etaS := by
  intro e
  obtain ⟨t, hcov, hdep, heval⟩ := hr e
  have hd8 := depth_le_eight hP hPg' hdep
  have hu0 := u_pos
  set x : ℕ → F32 := fun i => if h : i < Pg then prod.get2 ⟨i, h⟩ e else default with hx
  -- the exact products
  set w : ℕ → ℝ := fun i => phr pp i * mr vv i e with hw
  have hwz : ∀ i, c ≤ i → w i = 0 := by
    intro i hci
    simp only [hw, phr]
    by_cases hi : i < Pg
    · rw [dite_eq_left hi, hz ⟨i, hi⟩ hci]; unfold FloatV.toReal; rw [zero32_rat]; simp
    · simp [hi]
  have hwb : ∀ i, i < c → |w i| ≤ 2 ^ 16 := by
    intro i hic
    have hi : i < Pg := by omega
    simp only [hw, phr, mr, dite_eq_left hi, dite_eq_left (show e.val < Dm from e.2)]
    rw [abs_mul]
    have h1 := (hfin ⟨i, hi⟩ hic).2
    have h2 := (hv ⟨i, hi⟩ e).2 hic
    calc |(pp.get1 ⟨i, hi⟩).toReal| * |(vv.get2 ⟨i, hi⟩ ⟨e.val, e.2⟩).toReal| ≤ 2 * 2 ^ 15 :=
          mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
      _ = 2 ^ 16 := by norm_num
  have hwb' : ∀ i, |w i| ≤ 2 ^ 16 := by
    intro i
    by_cases hic : i < c
    · exact hwb i hic
    · rw [hwz i (by omega)]; norm_num
  -- the products as rounded
  have hxe : ∀ i, i < Pg → (x i).isFinite = true ∧ |((x i).toRat : ℝ) - w i| ≤ u * |w i| + 1 / 2 ^ 150 := by
    intro i hi
    have hxi : x i = prod.get2 ⟨i, hi⟩ e := by simp only [hx, dite_eq_left hi]
    have hpf : (pp.get1 ⟨i, hi⟩).isFinite = true := by
      by_cases hic : i < c
      · exact (hfin ⟨i, hi⟩ hic).1
      · rw [hz ⟨i, hi⟩ (by show c ≤ i; omega)]; exact zero32_fin
    have hvf := (hv ⟨i, hi⟩ e).1
    have hwi : w i = (pp.get1 ⟨i, hi⟩).toReal * (vv.get2 ⟨i, hi⟩ e).toReal := by
      simp only [hw, phr, mr, dite_eq_left hi, dite_eq_left (show e.val < Dm from e.2)]
    have hmax : |(pp.get1 ⟨i, hi⟩).toReal * (vv.get2 ⟨i, hi⟩ e).toReal| ≤ (Format.binary32.maxVal : ℝ) := by
      rw [← hwi, max32_real]; exact (hwb' i).trans (by norm_num)
    obtain ⟨hmf, hme⟩ := FloatV.mul_err_real Format.binary32_regular hpf hvf hmax
    have hxm : prod.get2 ⟨i, hi⟩ e = FloatV.mul (pp.get1 ⟨i, hi⟩) (vv.get2 ⟨i, hi⟩ e) :=
      (hprod ⟨i, hi⟩ e).eq_of_isFinite hmf
    rw [hxi, hxm]
    refine ⟨hmf, ?_⟩
    rw [u32_real, eta32_real, ← hwi] at hme
    unfold FloatV.toReal at hme
    rw [mul_comm]; exact hme
  have hxfin : ∀ i, i < Pg → (x i).isFinite = true := fun i hi => (hxe i hi).1
  have hs : Format.binary32.specials ≠ .finite := by decide
  have hu32 : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hPgR : (Pg : ℝ) ≤ 256 := by exact_mod_cast hPg'
  have hPg1 : ((Pg - 1 : ℕ) : ℝ) ≤ 255 := by
    have : Pg - 1 ≤ 255 := by omega
    exact_mod_cast this
  -- sums of the exact products
  have hsub : ∀ f : ℕ → ℝ, (∀ i, c ≤ i → f i = 0) → ∑ i ∈ range Pg, f i = ∑ j ∈ range c, f j := by
    intro f hf
    symm
    exact Finset.sum_subset (Finset.range_subset_range.mpr hc)
      (fun j _ hjc => hf j (by simpa using hjc))
  have hW : ∑ j ∈ range c, |w j| ≤ 2 ^ 24 := by
    calc ∑ j ∈ range c, |w j| ≤ ∑ j ∈ range c, (2 : ℝ) ^ 16 :=
          Finset.sum_le_sum fun j hj => hwb j (Finset.mem_range.mp hj)
      _ = c * 2 ^ 16 := by simp
      _ ≤ 256 * 2 ^ 16 := by
          have : (c : ℝ) ≤ 256 := by
            have : c ≤ 256 := by omega
            exact_mod_cast this
          exact mul_le_mul_of_nonneg_right this (by norm_num)
      _ = 2 ^ 24 := by norm_num
  have hsW : ∑ i ∈ range Pg, |w i| = ∑ j ∈ range c, |w j| := hsub (fun i => |w i|) (fun i hi => by rw [hwz i hi]; simp)
  have hsw : ∑ i ∈ range Pg, w i = ∑ j ∈ range c, w j := hsub w hwz
  have hW0 : 0 ≤ ∑ j ∈ range c, |w j| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  -- rounding of the products: sums
  have hX1 : ∑ i ∈ range Pg, |((x i).toRat : ℝ)| ≤ (1 + u) * ∑ j ∈ range c, |w j| + Pg * (1 / 2 ^ 150) := by
    rw [← hsW, Finset.mul_sum]
    calc ∑ i ∈ range Pg, |((x i).toRat : ℝ)| ≤ ∑ i ∈ range Pg, ((1 + u) * |w i| + 1 / 2 ^ 150) := by
          refine Finset.sum_le_sum fun i hi => ?_
          have h := (hxe i (Finset.mem_range.mp hi)).2
          have := abs_sub_abs_le_abs_sub ((x i).toRat : ℝ) (w i)
          linarith
      _ = ∑ i ∈ range Pg, (1 + u) * |w i| + Pg * (1 / 2 ^ 150) := by
          rw [Finset.sum_add_distrib]; simp
  have hX2 : |∑ i ∈ range Pg, ((x i).toRat : ℝ) - ∑ i ∈ range Pg, w i| ≤
      u * ∑ j ∈ range c, |w j| + Pg * (1 / 2 ^ 150) := by
    rw [← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ i ∈ range Pg, |((x i).toRat : ℝ) - w i| ≤ ∑ i ∈ range Pg, (u * |w i| + 1 / 2 ^ 150) :=
          Finset.sum_le_sum fun i hi => (hxe i (Finset.mem_range.mp hi)).2
      _ = u * ∑ i ∈ range Pg, |w i| + Pg * (1 / 2 ^ 150) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum]; simp
      _ = _ := by rw [hsW]
  set Aq : ℚ := ∑ i ∈ range Pg, |(x i).toRat| with hAq
  have hAr : (Aq : ℝ) ≤ 2 ^ 25 := by
    rw [hAq]; push_cast
    have hu24 : u * 2 ^ 24 = 1 := by unfold u; norm_num
    have h2 : u * ∑ j ∈ range c, |w j| ≤ u * 2 ^ 24 := mul_le_mul_of_nonneg_left hW hu0.le
    have h3 : (Pg : ℝ) * (1 / 2 ^ 150) ≤ 256 * (1 / 2 ^ 150) :=
      mul_le_mul_of_nonneg_right hPgR (by norm_num)
    have h4 : (256 : ℝ) * (1 / 2 ^ 150) ≤ 1 := by norm_num
    have h5 : (2 : ℝ) ^ 24 + 2 ≤ 2 ^ 25 := by norm_num
    nlinarith
  have hAq0 : (0 : ℚ) ≤ Aq := Finset.sum_nonneg fun i _ => abs_nonneg _
  have hEB : ((Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 8 (Pg - 1) Aq : ℚ) : ℝ) ≤
      241 / 10 * u * (Aq : ℝ) + 1 / 2 ^ 138 := by
    unfold Tree.errBound
    push_cast
    rw [u32_real, eta32_real]
    have hp8 := pow8_3_le
    have h1 : ((1 + 3 * u) ^ 8 - 1) * (Aq : ℝ) ≤ 241 / 10 * u * (Aq : ℝ) :=
      mul_le_mul_of_nonneg_right (by linarith) (by exact_mod_cast hAq0)
    have h8 : (1 + 3 * u) ^ 8 ≤ 2 := by unfold u; norm_num
    have h2 : 4 * (1 / 2 ^ 150 : ℝ) * ((Pg - 1 : ℕ) : ℝ) * (1 + 3 * u) ^ 8 ≤ 1 / 2 ^ 138 := by
      have : 4 * (1 / 2 ^ 150 : ℝ) * ((Pg - 1 : ℕ) : ℝ) ≤ 4 * (1 / 2 ^ 150) * 255 :=
        mul_le_mul_of_nonneg_left hPg1 (by norm_num)
      calc 4 * (1 / 2 ^ 150 : ℝ) * ((Pg - 1 : ℕ) : ℝ) * (1 + 3 * u) ^ 8 ≤ (4 * (1 / 2 ^ 150) * 255) * 2 :=
            mul_le_mul this h8 (by positivity) (by positivity)
        _ ≤ 1 / 2 ^ 138 := by norm_num
    linarith
  have hM : ((List.range Pg).map fun i => |(x i).toRat|).sum +
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 8 (Pg - 1)
        ((List.range Pg).map fun i => |(x i).toRat|).sum ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    rw [list_range_sum, ← hAq]
    rw [← Rat.cast_le (K := ℝ)]
    push_cast
    have := hEB
    rw [max32_real, eta32_real]
    have hu8 : (241 / 10 * u * 2 ^ 25 : ℝ) ≤ 49 := by unfold u; norm_num
    have hA2 : 241 / 10 * u * (Aq : ℝ) ≤ 241 / 10 * u * 2 ^ 25 :=
      mul_le_mul_of_nonneg_left hAr (by positivity)
    have h140 : (1 / 2 ^ 138 : ℝ) ≤ 1 := by norm_num
    have : (2 : ℝ) ^ 25 + 49 + 1 + 2 * (1 / 2 ^ 150) ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 := by norm_num
    linarith
  obtain ⟨hrfin, herr⟩ := sumc_err_depth Format.binary32_regular hs hu32 E.fusion x hcov hd8 heval hxfin hM
  rw [list_range_sum, list_range_sum] at herr
  have herr' : |(r.get1 e).toReal - ∑ i ∈ range Pg, ((x i).toRat : ℝ)| ≤
      ((Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 8 (Pg - 1) Aq : ℚ) : ℝ) := by
    have h1 : ((|(r.get1 e).toRat - ∑ i ∈ range Pg, (x i).toRat| : ℚ) : ℝ) ≤
        ((Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) 8 (Pg - 1) Aq : ℚ) : ℝ) := by
      exact_mod_cast herr
    push_cast at h1
    unfold FloatV.toReal
    exact h1
  have herrB : |(r.get1 e).toReal - ∑ i ∈ range Pg, ((x i).toRat : ℝ)| ≤
      241 / 10 * u * (Aq : ℝ) + 1 / 2 ^ 138 := herr'.trans hEB
  have hAr2 : 241 / 10 * u * (Aq : ℝ) ≤ 241 / 10 * u * ((1 + u) * ∑ j ∈ range c, |w j| + Pg * (1 / 2 ^ 150)) := by
    have : (Aq : ℝ) ≤ (1 + u) * ∑ j ∈ range c, |w j| + Pg * (1 / 2 ^ 150) := by
      rw [hAq]; push_cast; exact hX1
    exact mul_le_mul_of_nonneg_left this (by positivity)
  have hfinal : |(r.get1 e).toReal - ∑ j ∈ range c, w j| ≤ rhoSA * ∑ j ∈ range c, |w j| + etaS := by
    have h1 : |(r.get1 e).toReal - ∑ j ∈ range c, w j| ≤
        |(r.get1 e).toReal - ∑ i ∈ range Pg, ((x i).toRat : ℝ)| +
          |∑ i ∈ range Pg, ((x i).toRat : ℝ) - ∑ i ∈ range Pg, w i| := by
      rw [← hsw]
      have := abs_add_le ((r.get1 e).toReal - ∑ i ∈ range Pg, ((x i).toRat : ℝ))
        (∑ i ∈ range Pg, ((x i).toRat : ℝ) - ∑ i ∈ range Pg, w i)
      rwa [show (r.get1 e).toReal - ∑ i ∈ range Pg, ((x i).toRat : ℝ) +
        (∑ i ∈ range Pg, ((x i).toRat : ℝ) - ∑ i ∈ range Pg, w i) =
          (r.get1 e).toReal - ∑ i ∈ range Pg, w i by ring] at this
    have hPgη : (Pg : ℝ) * (1 / 2 ^ 150) ≤ 1 / 2 ^ 142 := by
      have := mul_le_mul_of_nonneg_right hPgR (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 150)
      norm_num at this ⊢; linarith
    have hu1 : u ≤ 1 / 1000 := by unfold u; norm_num
    have hW0' := hW0
    have hc1 : 241 / 10 * u * ((1 + u) * ∑ j ∈ range c, |w j| + Pg * (1 / 2 ^ 150)) ≤
        (241 / 10 * u + 1 / 10 * u) * ∑ j ∈ range c, |w j| + 241 / 10 * u * (1 / 2 ^ 142) := by
      have e : 241 / 10 * u * ((1 + u) * ∑ j ∈ range c, |w j| + Pg * (1 / 2 ^ 150)) =
          241 / 10 * u * ∑ j ∈ range c, |w j| + 241 / 10 * u * u * ∑ j ∈ range c, |w j| +
            241 / 10 * u * (Pg * (1 / 2 ^ 150)) := by ring
      have h2 : 241 / 10 * u * u * ∑ j ∈ range c, |w j| ≤ 1 / 10 * u * ∑ j ∈ range c, |w j| := by
        have : 241 / 10 * u * u ≤ 1 / 10 * u := by nlinarith
        exact mul_le_mul_of_nonneg_right this hW0
      have h3 : 241 / 10 * u * (Pg * (1 / 2 ^ 150)) ≤ 241 / 10 * u * (1 / 2 ^ 142) :=
        mul_le_mul_of_nonneg_left hPgη (by positivity)
      linarith
    have hsmall : 241 / 10 * u * (1 / 2 ^ 142) + 1 / 2 ^ 138 + (Pg : ℝ) * (1 / 2 ^ 150) ≤ etaS := by
      unfold etaS; norm_num; unfold u; norm_num; linarith
    unfold rhoSA
    nlinarith
  refine ⟨hrfin, ?_, ?_⟩
  · have hsw2 : |∑ j ∈ range c, w j| ≤ 2 ^ 24 :=
      (Finset.abs_sum_le_sum_abs _ _).trans hW
    have hrr : 255 / 10 * u * ∑ j ∈ range c, |w j| ≤ 255 / 10 * u * 2 ^ 24 :=
      mul_le_mul_of_nonneg_left hW (by positivity)
    have hu24 : (255 / 10 * u * 2 ^ 24 : ℝ) = 255 / 10 := by unfold u; norm_num
    have hes : etaS ≤ 1 := by unfold etaS; norm_num
    have := abs_sub_abs_le_abs_sub (r.get1 e).toReal (∑ j ∈ range c, w j)
    have h5 : (2 : ℝ) ^ 24 + 26 + 1 ≤ 2 ^ 25 := by norm_num
    unfold rhoSA at hfinal
    linarith
  · exact hfinal

/-- **`tl.mul_add(a, b, c)`** on `float32`: the product rounded then the sum rounded, or one
`fma` (with fusion). -/
theorem mulAdd32 {E : Env} {a b c r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hc : c.isFinite = true) (hM : |a.toReal * b.toReal| + |c.toReal| ≤ 2 ^ 100)
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul a b) c) r ∨
      (E.fusion = true ∧ FloatV.NaNEq (FloatV.fma a b c) r)) :
    r.isFinite = true ∧ |r.toReal - (a.toReal * b.toReal + c.toReal)| ≤
      rhoOL * (|a.toReal * b.toReal| + |c.toReal|) + etaO := by
  have hu32 : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hM' : |a.toRat * b.toRat| + |c.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have : ((|a.toRat * b.toRat| + |c.toRat| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      push_cast
      rw [max32_real, eta32_real]
      unfold FloatV.toReal at hM
      norm_num at hM ⊢
      linarith
    exact_mod_cast this
  obtain ⟨hf, hr⟩ := FloatV.mulAdd_err Format.binary32_regular hu32 h ha hb hc hM'
  refine ⟨hf, ?_⟩
  have : ((|r.toRat - (a.toRat * b.toRat + c.toRat)| : ℚ) : ℝ) ≤
      ((3 * Format.binary32.u * (|a.toRat * b.toRat| + |c.toRat|) + 4 * Format.binary32.eta : ℚ) : ℝ) := by
    exact_mod_cast hr
  push_cast at this
  rw [u32_real, eta32_real] at this
  unfold FloatV.toReal
  unfold rhoOL etaO
  norm_num at this ⊢
  linarith

/-- **`tl.addc(tl.mul(x, al), b)`** on `float32` (the `PAGE_SIZE = 1` update of `acc`): a rounded
product, then a fusable addition. -/
theorem addc32 {E : Env} {x al pr b r : F32} (hx : x.isFinite = true) (hal : al.isFinite = true)
    (hb : b.isFinite = true) (hM : |x.toReal * al.toReal| + |b.toReal| ≤ 2 ^ 100)
    (hmul : FloatV.NaNEq (FloatV.mul x al) pr) (hadd : AddRel E.fusion pr b r) :
    r.isFinite = true ∧ |r.toReal - (al.toReal * x.toReal + b.toReal)| ≤
      rhoOA * (|al.toReal * x.toReal| + |b.toReal|) + etaO := by
  have hu := u_pos
  have hu32 : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hxa : |x.toReal * al.toReal| ≤ 2 ^ 100 := by
    have := abs_nonneg b.toReal; linarith
  obtain ⟨hpf, hpe⟩ := FloatV.mul_err_real Format.binary32_regular hx hal (by
    rw [max32_real]; refine hxa.trans ?_; norm_num)
  have hpr : pr = FloatV.mul x al := hmul.eq_of_isFinite hpf
  rw [← hpr] at hpf hpe
  have hs : Format.binary32.specials ≠ .finite := by decide
  have hpabs : |pr.toReal| ≤ (1 + u) * |x.toReal * al.toReal| + 1 / 2 ^ 150 := by
    have h1 := abs_sub_abs_le_abs_sub pr.toReal (x.toReal * al.toReal)
    rw [u32_real, eta32_real] at hpe
    nlinarith [abs_nonneg (x.toReal * al.toReal)]
  have hM' : |pr.toRat| + |b.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have hp2 : |pr.toReal| ≤ 2 ^ 102 := by
      have : (1 + u) * |x.toReal * al.toReal| ≤ 2 * 2 ^ 100 := by
        have hu1 : u ≤ 1 := by unfold u; norm_num
        nlinarith [abs_nonneg (x.toReal * al.toReal)]
      have : (1:ℝ) / 2 ^ 150 ≤ 1 := by norm_num
      linarith
    have hb2 : |b.toReal| ≤ 2 ^ 100 := by
      have := abs_nonneg (x.toReal * al.toReal); linarith
    have key : |pr.toReal| + |b.toReal| ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * (1 / 2 ^ 150 : ℝ) := by
      have h1 : (2 : ℝ) ^ 102 + 2 ^ 100 ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * (1 / 2 ^ 150 : ℝ) := by norm_num
      linarith
    rw [← Rat.cast_le (K := ℝ)]
    push_cast
    rw [max32_real, eta32_real]
    unfold FloatV.toReal at key
    exact key
  obtain ⟨hrf, hre⟩ := FloatV.AddRel.err Format.binary32_regular hs hu32 hadd hpf hb hM'
  refine ⟨hrf, ?_⟩
  have hre' : |r.toReal - (pr.toReal + b.toReal)| ≤
      3 * u * (|pr.toReal| + |b.toReal|) + 4 * (1 / 2 ^ 150) := by
    have : ((|r.toRat - (pr.toRat + b.toRat)| : ℚ) : ℝ) ≤
        ((3 * Format.binary32.u * (|pr.toRat| + |b.toRat|) + 4 * Format.binary32.eta : ℚ) : ℝ) := by
      exact_mod_cast hre
    push_cast at this
    rw [u32_real, eta32_real] at this
    unfold FloatV.toReal
    exact this
  rw [u32_real, eta32_real] at hpe
  have e : r.toReal - (al.toReal * x.toReal + b.toReal) =
      (r.toReal - (pr.toReal + b.toReal)) + (pr.toReal - x.toReal * al.toReal) := by ring
  have habs : |al.toReal * x.toReal| = |x.toReal * al.toReal| := by rw [mul_comm]
  rw [habs, e]
  have h2 := abs_add_le (r.toReal - (pr.toReal + b.toReal)) (pr.toReal - x.toReal * al.toReal)
  unfold rhoOA etaO
  have h3 : 3 * u * |pr.toReal| ≤ 3 * u * ((1 + u) * |x.toReal * al.toReal| + 1 / 2 ^ 150) :=
    mul_le_mul_of_nonneg_left hpabs (by positivity)
  have hu1 : u ≤ 1 / 1000 := by unfold u; norm_num
  have h4 : 3 * u * u * |x.toReal * al.toReal| ≤ 3 * u * (1 / 1000) * |x.toReal * al.toReal| := by
    have := mul_le_mul_of_nonneg_left hu1 (by positivity : (0 : ℝ) ≤ 3 * u)
    nlinarith [abs_nonneg (x.toReal * al.toReal)]
  have hc : (1 / 2 ^ 150 : ℝ) * 20 ≤ 1 / 2 ^ 120 := by norm_num
  nlinarith [abs_nonneg (x.toReal * al.toReal), abs_nonneg b.toReal]

/-- **The division** `acc / l` (`/` on `float32`: the platform's `div.full`): finite operands,
`l > 0`, `|a/l| ≤ 40000`: the quotient is finite, at most `40001` in magnitude, within
`2^-22·|a/l| + 2^-148`. -/
theorem div_ok {E : Env} (hP : Device E.P) {a l q : F32} (ha : a.isFinite = true)
    (hl : l.isFinite = true) (hl0 : 0 < l.toReal) (hq : |a.toReal / l.toReal| ≤ 40000)
    (hdiv : E.P.div .binary32 a l q) :
    q.isFinite = true ∧ |q.toReal| ≤ 40001 ∧
      |q.toReal - a.toReal / l.toReal| ≤ 1 / 2 ^ 22 * |a.toReal / l.toReal| + 1 / 2 ^ 148 := by
  have hl0' : l.toRat ≠ 0 := by
    intro h; unfold FloatV.toReal at hl0; rw [h] at hl0; simp at hl0
  have hq' : |a.toRat / l.toRat| ≤ 2 ^ 126 := by
    have : ((|a.toRat / l.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast
      unfold FloatV.toReal at hq
      have : (40000 : ℝ) ≤ 2 ^ 126 := by norm_num
      linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := hP.divFull a l q hdiv ha hl hl0' hq'
  refine ⟨hf, ?_, ?_⟩ <;>
  · have he' : |q.toReal - a.toReal / l.toReal| ≤ 2 ^ (-22 : ℤ) * |a.toReal / l.toReal| + 2 ^ (-148 : ℤ) := by
      have : ((|q.toRat - a.toRat / l.toRat| : ℚ) : ℝ) ≤
          (((2 : ℚ) ^ (-22 : ℤ) * |a.toRat / l.toRat| + (2 : ℚ) ^ (-148 : ℤ) : ℚ) : ℝ) := by
        exact_mod_cast he
      push_cast at this
      unfold FloatV.toReal
      exact this
    norm_num at he' ⊢
    first
    | exact he'
    | (have h1 := abs_sub_abs_le_abs_sub q.toReal (a.toReal / l.toReal)
       linarith)


/-- **The `float16` rounding** `out.to(tl.float16)` of a finite `float32` of magnitude at most
`40001`: finite, within `2^-11·|q| + 2^-25`. -/
theorem cvt16_ok {q : F32} {c : F16} (hq : q.isFinite = true) (hqb : |q.toReal| ≤ 40001)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn q) c) :
    c.isFinite = true ∧ |c.toReal - q.toReal| ≤ 1 / 2 ^ 11 * |q.toReal| + 1 / 2 ^ 25 := by
  have hmax : |q.toRat| ≤ Format.binary16.maxVal := by
    have : ((|q.toRat| : ℚ) : ℝ) ≤ ((Format.binary16.maxVal : ℚ) : ℝ) := by
      push_cast
      rw [Format.binary16_maxVal_real]
      unfold FloatV.toReal at hqb
      linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.cvt_err Format.binary16_regular hq hmax
  have hc : c = FloatV.cvt q := h.eq_of_isFinite hf
  subst hc
  refine ⟨hf, ?_⟩
  have : ((|(FloatV.cvt q : F16).toRat - q.toRat| : ℚ) : ℝ) ≤
      ((|q.toRat| * Format.binary16.u + Format.binary16.eta : ℚ) : ℝ) := by exact_mod_cast he
  push_cast at this
  rw [Format.binary16_u_real, Format.binary16_eta_real] at this
  unfold FloatV.toReal
  norm_num at this ⊢
  linarith

end Num

end Claims.PagedAttention
