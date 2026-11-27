import Proof.Consts

/-!
# The model's float operations as bounds on reals

Each lemma turns what the model allows for one operation (`NaNEq` to the IEEE result, `AddRel`, a
device fact) into finiteness and a bound on real values.
-/

namespace Claims.GeluAndMulSparse

open TVA

set_option maxHeartbeats 800000

/-- The sum of `f` over `List.range n` is the `Finset` sum. -/
theorem list_sum_range (f : ℕ → ℚ) (n : ℕ) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ, ih]; simp

/-- **A `tl.sum` of `n ≤ 2^15` finite `float32` lanes with tree depth at most `D ≤ 15`** (IEEE
additions, any tree): finite, within `((1+u)^D − 1) Σ|l| + η (n−1) (1+u)^D` of the exact sum. -/
theorem sum_real {n D : ℕ} (hD : D ≤ 15) (hn : n ≤ 2 ^ 15) (l : ℕ → F32) {t : Tree} {S : F32}
    (hcov : t.Covers n) (hd : t.depth ≤ D) (heval : t.Eval (AddRel false) l S)
    (hfin : ∀ i, i < n → (l i).isFinite = true)
    (hb : ∀ i, i < n → |(l i).toReal| ≤ 2 ^ 40) :
    S.isFinite = true ∧
      |S.toReal - ∑ i ∈ Finset.range n, (l i).toReal| ≤
        ((1 + u32) ^ D - 1) * ∑ i ∈ Finset.range n, |(l i).toReal| +
          eta32 * ((n - 1 : ℕ) : ℝ) * (1 + u32) ^ D := by
  have hf := Format.binary32_regular
  have hM : (Format.binary32.maxVal : ℚ) = (2 ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal
  have hbq : ∀ i, i < n → |(l i).toRat| ≤ 2 ^ 40 := fun i hi => by
    have := hb i hi; unfold FloatV.toReal at this; exact_mod_cast this
  have hA : ((List.range n).map fun i => |(l i).toRat|).sum ≤ 2 ^ 55 := by
    rw [list_sum_range]
    calc _ ≤ ∑ i ∈ Finset.range n, (2 : ℚ) ^ 40 :=
          Finset.sum_le_sum fun i hi => hbq i (Finset.mem_range.mp hi)
      _ = n * 2 ^ 40 := by simp
      _ ≤ 2 ^ 15 * 2 ^ 40 := by
          have : (n : ℚ) ≤ 2 ^ 15 := by exact_mod_cast hn
          nlinarith
      _ = 2 ^ 55 := by norm_num
  have hu1 : (1 + Format.binary32.u) ^ D ≤ 2 := by
    rw [Format.binary32_u]
    calc (1 + (2 : ℚ) ^ (-24 : ℤ)) ^ D ≤ (1 + 2 ^ (-24 : ℤ)) ^ 15 :=
          pow_le_pow_right₀ (by norm_num) hD
      _ ≤ 2 := by norm_num
  have hA0 : 0 ≤ ((List.range n).map fun i => |(l i).toRat|).sum := by
    rw [list_sum_range]; exact Finset.sum_nonneg fun i _ => abs_nonneg _
  obtain ⟨hfinS, herr⟩ := sum_err_depth hf l hcov hd heval hfin (by
    unfold Tree.errBound
    rw [Format.binary32_eta, hM]
    have hn' : ((n - 1 : ℕ) : ℚ) ≤ 2 ^ 15 := by
      have : n - 1 ≤ 2 ^ 15 := by omega
      exact_mod_cast this
    have hp : 0 ≤ (1 + Format.binary32.u) ^ D :=
      pow_nonneg (by linarith [Format.u_nonneg (f := Format.binary32)]) D
    have h1 : ((1 + Format.binary32.u) ^ D - 1) * ((List.range n).map fun i => |(l i).toRat|).sum ≤
        2 ^ 55 := by nlinarith
    have h2 : (2 : ℚ) ^ (-150 : ℤ) * ((n - 1 : ℕ) : ℚ) * (1 + Format.binary32.u) ^ D ≤ 1 := by
      have : (2 : ℚ) ^ (-150 : ℤ) * ((n - 1 : ℕ) : ℚ) * (1 + Format.binary32.u) ^ D ≤
          2 ^ (-150 : ℤ) * 2 ^ 15 * 2 := by
        gcongr
      refine this.trans ?_
      norm_num
    norm_num at h1 h2 ⊢
    linarith)
  refine ⟨hfinS, ?_⟩
  rw [list_sum_range, list_sum_range, Format.binary32_u, Format.binary32_eta] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  unfold Tree.errBound at this
  push_cast at this
  unfold u32 eta32 FloatV.toReal
  simpa using this

/-- **`/` on `float32`** (`Device.DivFull`): finite, within `2^-22 |x/y| + 2^-148` of `x / y`. -/
theorem div_real {P : Platform} (hP : Device P) {x y r : F32} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hy0 : y.toReal ≠ 0) (hq : |x.toReal / y.toReal| ≤ 2 ^ 126)
    (h : P.div .binary32 x y r) :
    r.isFinite = true ∧
      |r.toReal - x.toReal / y.toReal| ≤ 2 ^ (-22 : ℤ) * |x.toReal / y.toReal| + 2 ^ (-148 : ℤ) := by
  have hy0' : y.toRat ≠ 0 := fun h0 => hy0 (by unfold FloatV.toReal; rw [h0]; simp)
  have hq' : |x.toRat / y.toRat| ≤ 2 ^ 126 := by
    unfold FloatV.toReal at hq
    rw [← Rat.cast_div, ← Rat.cast_abs] at hq
    exact_mod_cast hq
  obtain ⟨hfin, herr⟩ := hP.divFull x y r h hx hy hy0' hq'
  refine ⟨hfin, ?_⟩
  have := (Rat.cast_le (K := ℝ)).mpr herr
  unfold FloatV.toReal
  push_cast at this ⊢
  simpa [Rat.cast_div] using this

/-- **`tl.sqrt` on `float32`** (`Device.SqrtF32`): on a normal input of moderate size, finite and
within `2^-22 √s + 2^-149` of `√s`. -/
theorem sqrt_real {P : Platform} (hP : Device P) {s r : F32} (hs : s.isFinite = true)
    (hlo : (2 : ℝ) ^ (-126 : ℤ) ≤ s.toReal) (hhi : s.toReal ≤ 2 ^ 127)
    (h : P.math .sqrt .binary32 s r) :
    r.isFinite = true ∧
      |r.toReal - Real.sqrt s.toReal| ≤ 2 ^ (-22 : ℤ) * Real.sqrt s.toReal + 2 ^ (-149 : ℤ) := by
  obtain ⟨ref, href, hacc⟩ := hP.sqrtF32
  have hr : ref = Real.sqrt := by
    unfold MathFn.ref at href; exact (Option.some.inj href).symm
  subst hr
  have hsq : Real.sqrt s.toReal ≤ 2 ^ 64 := by
    rw [Real.sqrt_le_left (by norm_num)]; norm_num; linarith
  have hsq0 := Real.sqrt_nonneg s.toReal
  have hpos : (0 : ℝ) < 2 ^ (-126 : ℤ) := by positivity
  obtain ⟨h1, h2⟩ := hacc s r h hs (Or.inr hlo)
    (by simp only [MathFn.natDom]; exact Set.mem_Ici.mpr (by linarith)) (by
      rw [abs_of_nonneg hsq0, Format.binary32_maxVal_real]
      norm_num
      nlinarith)
  rw [abs_of_nonneg hsq0] at h2
  exact ⟨h1, h2⟩

/-- **The host's `eps` as a `float32`**: within `u |eps| + η`. -/
theorem hostF32_err {c : F64} (hc : c.isFinite = true) (hb : |c.toReal| ≤ 2 ^ 126) :
    (FloatV.hostF32 c).isFinite = true ∧
      |(FloatV.hostF32 c).toReal - c.toReal| ≤ u32 * |c.toReal| + eta32 := by
  have hnan : c.isNaN = false := (FloatV.isFinite_iff.mp hc).1
  unfold FloatV.hostF32
  rw [ite_eq_right (by simp [hnan])]
  have hM : (2 : ℝ) ^ 126 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]; norm_num
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real Format.binary32_regular hc (hb.trans hM)
  refine ⟨h1, ?_⟩
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  unfold u32 eta32
  linarith

/-- **A fusable addition** (`tl.addc`) of finite `float32` values with room: finite, within
`3u(|x| + |y|) + 4η` of `x + y`. -/
theorem addc_real {fusion : Bool} {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hb : |x.toReal| + |y.toReal| ≤ 2 ^ 126 + 2 ^ 111)
    (h : AddRel fusion x y r) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal + y.toReal)| ≤ 3 * u32 * (|x.toReal| + |y.toReal|) + 4 * eta32 := by
  have hf := Format.binary32_regular
  have hs : Format.binary32.specials ≠ Format.Specials.finite := by
    rw [(Format.binary32_std.ieee)]; decide
  have hu : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hbq : |x.toRat| + |y.toRat| ≤ 2 ^ 126 + 2 ^ 111 := by
    unfold FloatV.toReal at hb
    rw [← Rat.cast_abs, ← Rat.cast_abs] at hb
    exact_mod_cast hb
  obtain ⟨hfin, herr⟩ := FloatV.AddRel.err hf hs hu h hx hy (by
    rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num at hbq ⊢; linarith)
  refine ⟨hfin, ?_⟩
  rw [Format.binary32_u, Format.binary32_eta] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  unfold u32 eta32 FloatV.toReal
  simpa using this

/-- **An IEEE product** of finite `float32` values with room: finite, within `u |a b| + η`. -/
theorem mul_lane {a b r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hab : |a.toReal * b.toReal| ≤ 2 ^ 120) (h : FloatV.NaNEq (FloatV.mul a b) r) :
    r.isFinite = true ∧ |r.toReal - a.toReal * b.toReal| ≤ u32 * |a.toReal * b.toReal| + eta32 := by
  have hf := Format.binary32_regular
  have hM : (2 : ℝ) ^ 120 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]; norm_num
  obtain ⟨hfin, herr⟩ := FloatV.mul_err_real hf ha hb (hab.trans hM)
  rw [h.eq (FloatV.isFinite_iff.mp hfin).1]
  refine ⟨hfin, ?_⟩
  rw [Format.binary32_u_real, Format.binary32_eta_real] at herr
  unfold u32 eta32
  linarith

/-- **`tl.mul_add a b c`** of finite `float32` values with room: finite, within
`3u(|a b| + |c|) + 4η` of `a b + c`. -/
theorem muladd_lane {fusion : Bool} {a b c r : F32} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hc : c.isFinite = true) (hroom : |a.toReal * b.toReal| + |c.toReal| ≤ 2 ^ 120)
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul a b) c) r ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma a b c) r)) :
    r.isFinite = true ∧ |r.toReal - (a.toReal * b.toReal + c.toReal)| ≤
      3 * u32 * (|a.toReal * b.toReal| + |c.toReal|) + 4 * eta32 := by
  have hf := Format.binary32_regular
  have hu : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hM : (Format.binary32.maxVal : ℚ) = (2 ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal
  have hroomq : |a.toRat * b.toRat| + |c.toRat| ≤ 2 ^ 120 := by
    unfold FloatV.toReal at hroom
    rw [← Rat.cast_mul, ← Rat.cast_abs, ← Rat.cast_abs] at hroom
    exact_mod_cast hroom
  obtain ⟨hfin, herr⟩ := FloatV.mulAdd_err hf hu h ha hb hc (by
    rw [hM, Format.binary32_eta]; norm_num at hroomq ⊢; linarith)
  refine ⟨hfin, ?_⟩
  rw [Format.binary32_u, Format.binary32_eta] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  push_cast at this
  unfold u32 eta32 FloatV.toReal
  simpa using this

/-- **A `tl.sum` of products along a tree of fusable additions** (`tl.sumc`), `n ≤ 2^15` finite
lanes of magnitude at most `2^70`, depth at most `D ≤ 15`: finite, within
`((1+3u)^D − 1) Σ|l| + 4η (n−1) (1+3u)^D` of the exact sum. -/
theorem sumc_real {fusion : Bool} {n D : ℕ} (hD : D ≤ 15) (hn : n ≤ 2 ^ 15) (l : ℕ → F32)
    {t : Tree} {S : F32}
    (hcov : t.Covers n) (hd : t.depth ≤ D) (heval : t.Eval (AddRel fusion) l S)
    (hfin : ∀ i, i < n → (l i).isFinite = true)
    (hb : ∀ i, i < n → |(l i).toReal| ≤ 2 ^ 70) :
    S.isFinite = true ∧
      |S.toReal - ∑ i ∈ Finset.range n, (l i).toReal| ≤
        ((1 + 3 * u32) ^ D - 1) * ∑ i ∈ Finset.range n, |(l i).toReal| +
          4 * eta32 * ((n - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ D := by
  have hf := Format.binary32_regular
  have hs : Format.binary32.specials ≠ Format.Specials.finite := by
    rw [(Format.binary32_std.ieee)]; decide
  have hu : Format.binary32.u ≤ 1 / 16 := by rw [Format.binary32_u]; norm_num
  have hM : (Format.binary32.maxVal : ℚ) = (2 ^ 24 - 1) * 2 ^ 104 := Format.binary32_maxVal
  have hbq : ∀ i, i < n → |(l i).toRat| ≤ 2 ^ 70 := fun i hi => by
    have := hb i hi; unfold FloatV.toReal at this; exact_mod_cast this
  have hA : ((List.range n).map fun i => |(l i).toRat|).sum ≤ 2 ^ 85 := by
    rw [list_sum_range]
    calc _ ≤ ∑ i ∈ Finset.range n, (2 : ℚ) ^ 70 :=
          Finset.sum_le_sum fun i hi => hbq i (Finset.mem_range.mp hi)
      _ = n * 2 ^ 70 := by simp
      _ ≤ 2 ^ 15 * 2 ^ 70 := by
          have : (n : ℚ) ≤ 2 ^ 15 := by exact_mod_cast hn
          nlinarith
      _ = 2 ^ 85 := by norm_num
  have hu1 : (1 + 3 * Format.binary32.u) ^ D ≤ 2 := by
    rw [Format.binary32_u]
    calc (1 + 3 * (2 : ℚ) ^ (-24 : ℤ)) ^ D ≤ (1 + 3 * 2 ^ (-24 : ℤ)) ^ 15 :=
          pow_le_pow_right₀ (by norm_num) hD
      _ ≤ 2 := by norm_num
  have hA0 : 0 ≤ ((List.range n).map fun i => |(l i).toRat|).sum := by
    rw [list_sum_range]; exact Finset.sum_nonneg fun i _ => abs_nonneg _
  obtain ⟨hfinS, herr⟩ := sumc_err_depth hf hs hu fusion l hcov hd heval hfin (by
    unfold Tree.errBound
    rw [Format.binary32_eta, hM]
    have hn' : ((n - 1 : ℕ) : ℚ) ≤ 2 ^ 15 := by
      have : n - 1 ≤ 2 ^ 15 := by omega
      exact_mod_cast this
    have hp : 0 ≤ (1 + 3 * Format.binary32.u) ^ D :=
      pow_nonneg (by linarith [Format.u_nonneg (f := Format.binary32)]) D
    have h1 : ((1 + 3 * Format.binary32.u) ^ D - 1) *
        ((List.range n).map fun i => |(l i).toRat|).sum ≤ 2 ^ 85 := by nlinarith
    have h2 : 4 * (2 : ℚ) ^ (-150 : ℤ) * ((n - 1 : ℕ) : ℚ) * (1 + 3 * Format.binary32.u) ^ D ≤ 1 := by
      have : 4 * (2 : ℚ) ^ (-150 : ℤ) * ((n - 1 : ℕ) : ℚ) * (1 + 3 * Format.binary32.u) ^ D ≤
          4 * 2 ^ (-150 : ℤ) * 2 ^ 15 * 2 := by
        gcongr
      refine this.trans ?_
      norm_num
    norm_num at h1 h2 ⊢
    linarith)
  refine ⟨hfinS, ?_⟩
  rw [list_sum_range, list_sum_range, Format.binary32_u, Format.binary32_eta] at herr
  have := (Rat.cast_le (K := ℝ)).mpr herr
  unfold Tree.errBound at this
  push_cast at this
  unfold u32 eta32 FloatV.toReal
  simpa using this

/-- **An IEEE sum** of finite `float32` values with room: finite, within `u |x + y| + η`. -/
theorem add_lane {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hxy : |x.toReal + y.toReal| ≤ 2 ^ 120) (h : FloatV.NaNEq (FloatV.add x y) r) :
    r.isFinite = true ∧ |r.toReal - (x.toReal + y.toReal)| ≤ u32 * |x.toReal + y.toReal| + eta32 := by
  have hf := Format.binary32_regular
  have hM : (2 : ℝ) ^ 120 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]; norm_num
  obtain ⟨hfin, herr⟩ := FloatV.add_err_real hf hx hy (hxy.trans hM)
  rw [h.eq (FloatV.isFinite_iff.mp hfin).1]
  refine ⟨hfin, ?_⟩
  rw [Format.binary32_u_real, Format.binary32_eta_real] at herr
  unfold u32 eta32
  linarith

/-- **An IEEE difference** of finite `float32` values with room: finite, within `u |x − y| + η`. -/
theorem sub_lane {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hxy : |x.toReal - y.toReal| ≤ 2 ^ 120) (h : FloatV.NaNEq (FloatV.sub x y) r) :
    r.isFinite = true ∧ |r.toReal - (x.toReal - y.toReal)| ≤ u32 * |x.toReal - y.toReal| + eta32 := by
  have hf := Format.binary32_regular
  have hM : (2 : ℝ) ^ 120 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]; norm_num
  obtain ⟨hfin, herr⟩ := FloatV.sub_err_real hf hx hy (hxy.trans hM)
  rw [h.eq (FloatV.isFinite_iff.mp hfin).1]
  refine ⟨hfin, ?_⟩
  rw [Format.binary32_u_real, Format.binary32_eta_real] at herr
  unfold u32 eta32
  linarith

/-- **`float32 → bfloat16`** (`.to(bfloat16)`, the store's conversion): finite, within
`2^-8 |y| + 2^-134`. -/
theorem cvt16_lane {y : F32} {r : BF16} (hy : y.isFinite = true) (hb : |y.toReal| ≤ 2 ^ 120)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn y) r) :
    r.isFinite = true ∧ |r.toReal - y.toReal| ≤ ub * |y.toReal| + etab := by
  rw [FloatV.cvtWith_rn] at h
  have hM : |y.toReal| ≤ (Format.bfloat16.maxVal : ℝ) := by
    rw [Format.bfloat16_maxVal_real]; exact hb.trans (by norm_num)
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hy hM
  rw [h.eq (FloatV.isFinite_iff.mp hfin).1]
  refine ⟨hfin, ?_⟩
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at herr
  unfold ub etab
  linarith

/-- **`bfloat16 → float32`** (`.to(tl.float32)`): exact on a finite value. -/
theorem cvt32_lane {x : BF16} {r : F32} (hx : x.isFinite = true)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r) :
    r.isFinite = true ∧ r.toReal = x.toReal := by
  rw [FloatV.cvtWith_rn] at h
  obtain ⟨h1, -, h3⟩ := FloatV.cvt_bf16_f32_exact hx
  rw [h.eq (FloatV.isFinite_iff.mp h1).1]
  exact ⟨h1, h3⟩

/-- **`-x`**: exact on a finite value. -/
theorem neg_lane {x r : F32} (hx : x.isFinite = true) (h : FloatV.NaNEq (FloatV.neg x) r) :
    r.isFinite = true ∧ r.toReal = -x.toReal := by
  have h1 : (FloatV.neg x).isFinite = true := (FloatV.isFinite_neg x).trans hx
  rw [h.eq (FloatV.isFinite_iff.mp h1).1]
  exact ⟨h1, FloatV.toReal_neg x⟩

/-- **`tl.exp` on `float32`** (`Device.ExpF32` at `X = |x|`): for a finite `x` with the room
`(1 + 2^-22 + 2^-23 |x|) e^x + 2^-140 ≤ 2^10`, finite and within `(2^-22 + 2^-23 |x|) e^x + 2^-140`. -/
theorem exp_lane {P : Platform} (hP : Device P) {x r : F32} (hx : x.isFinite = true)
    (hroom : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |x.toReal|)) * |Real.exp x.toReal| +
      2 ^ (-140 : ℤ) ≤ 2 ^ 10)
    (h : P.math .exp .binary32 x r) :
    r.isFinite = true ∧ |r.toReal - Real.exp x.toReal| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |x.toReal|) * Real.exp x.toReal + 2 ^ (-140 : ℤ) := by
  obtain ⟨ref, href, hacc⟩ := hP.expF32 |x.toReal| (abs_nonneg _)
  have hr : ref = Real.exp := by
    unfold MathFn.ref at href; exact (Option.some.inj href).symm
  subst hr
  have hdom : x.toReal ∈ Set.Icc (-|x.toReal|) |x.toReal| := ⟨neg_abs_le _, le_abs_self _⟩
  have hM : (2 : ℝ) ^ 10 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal_real]; norm_num
  obtain ⟨h1, h2⟩ := hacc x r h hx hdom (by simp [MathFn.natDom]) (hroom.trans hM)
  rw [abs_of_pos (Real.exp_pos _)] at h2
  exact ⟨h1, h2⟩

end Claims.GeluAndMulSparse
