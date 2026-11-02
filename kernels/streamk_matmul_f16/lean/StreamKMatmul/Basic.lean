import StreamKMatmul.Defs
import StreamKMatmul.Facts

/-!
# `streamk_matmul_f16`: the float facts the proof uses

* the zero `other=0.0` converted to `float16` (`zero_conv`);
* the products of a reduction in chunks of `64` (`prodV`, `liftV`, `chunkS`, `chunkA`), the lifted
  magnitudes `mag16` (`Rules.Num.liftMag .f16`), and one `tl.dot` step of a tile's part
  (`acc_step`), from `Rules.Num.DotF16`;
* the bound `DotLoop.bound` is monotone, and the parts' bounds add up to at most the whole tile's
  (`sum_bound_le`);
* the size of the tolerance (`tol_le`).
-/

namespace Kernels.StreamKMatmul

open TVA

/-! ## `other=0.0` -/

theorem f32_zero_lit : (FloatV.litF32 (FloatV.ofPat 0) : F32) = FloatV.ofPat 0 := by
  unfold FloatV.litF32
  rw [ite_eq_left]
  rw [FloatV.isZero_ofPat (by norm_num)]
  decide

theorem f16_cvt_zero : (FloatV.cvt (FloatV.ofPat 0 : F32) : F16).isFinite = true ∧
    (FloatV.cvt (FloatV.ofPat 0 : F32) : F16).toRat = 0 := by
  have h0 : (FloatV.ofPat 0 : F32).pat = 0 := FloatV.pat_ofPat (by norm_num)
  have hfin : Math.Float.Format.binary32.isFinite 0 = true := by decide
  have hsign : Math.Float.Format.binary32.sign 0 = false := by decide
  have htr : Math.Float.Format.binary32.toRat 0 = 0 :=
    Math.Float.Format.toRat_of_isZero (by decide)
  unfold FloatV.cvt
  rw [h0, Math.Float.Format.cvt_of_finite _ _ _ hfin, hsign, htr]
  obtain ⟨hlt, hf, hr⟩ := Math.Float.Format.roundRat_zero_spec (f := Math.Float.Format.binary16)
    (rnd := .rn) (sat := false) (zs := false) Math.Float.Format.binary16_regular
  rw [FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨hf, hr⟩

/-- The source's `other=0.0` converted to `float16`: some result exists. -/
theorem zero_conv_ex (P : Platform) :
    ∃ r, ConvRel P .f32 .f16 (FloatV.litF32 (FloatV.ofPat 0) : F32) r :=
  ⟨FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0) : F32), by
    show FloatRel P Math.Float.Format.binary32 Math.Float.Format.binary16 _ _
    unfold FloatRel
    rw [ite_eq_right (by decide)]
    exact Or.inl ⟨by unfold Math.Float.Format.Std; simp, by unfold Math.Float.Format.Std; simp,
      FloatV.NaNEq.refl _⟩⟩

/-- Every result of the conversion is a finite zero. -/
theorem zero_conv {P : Platform} {r : F16}
    (h : ConvRel P .f32 .f16 (FloatV.litF32 (FloatV.ofPat 0) : F32) r) :
    r.isFinite = true ∧ r.toRat = 0 := by
  change FloatRel P Math.Float.Format.binary32 Math.Float.Format.binary16 _ r at h
  unfold FloatRel at h
  rw [ite_eq_right (by decide), f32_zero_lit] at h
  obtain ⟨hf, hr⟩ := f16_cvt_zero
  rcases h with ⟨-, -, h⟩ | ⟨-, hF8, -⟩ | ⟨hF8, -, -⟩
  · have hn : (FloatV.cvt (FloatV.ofPat 0 : F32) : F16).isNaN = false :=
      (FloatV.isFinite_iff.mp hf).1
    rw [h.eq hn]
    exact ⟨hf, hr⟩
  · exact absurd hF8 (by unfold Math.Float.Format.F8; decide)
  · exact absurd hF8 (by unfold Math.Float.Format.F8; decide)

/-! ## Products in chunks of `64` -/

/-- Product `e` of the reduction, `0` from `K` on (the lanes the masks disable). -/
def prodV (va vb : Nat → F16) (K e : Nat) : ℚ := if e < K then (va e).toRat * (vb e).toRat else 0

/-- The lifted magnitude of product `e` (`mag16` of each operand), `0` from `K` on. -/
def liftV (va vb : Nat → F16) (K e : Nat) : ℚ :=
  if e < K then mag16 (va e).toRat * mag16 (vb e).toRat else 0

/-- The exact sum of chunk `u`: the products `64·u ≤ e < 64·u + 64` (one `tl.dot`, line 57). -/
def chunkS (va vb : Nat → F16) (K u : Nat) : ℚ := ∑ k : Fin 64, prodV va vb K (64 * u + k)

/-- The sum of the lifted magnitudes of chunk `u`. -/
def chunkA (va vb : Nat → F16) (K u : Nat) : ℚ := ∑ k : Fin 64, liftV va vb K (64 * u + k)

/-- `mag16` is the library's `liftMag .f16` (by definition). -/
theorem liftMag_f16 (q : ℚ) : Rules.Num.liftMag .f16 q = mag16 q := rfl

theorem mag16_nonneg (q : ℚ) : 0 ≤ mag16 q := Dot8Chunk.liftMag_nonneg .f16 q

theorem mag16_zero : mag16 0 = 0 := Dot8Chunk.liftMag_zero .f16

theorem abs_mul_le_mag16 (x y : ℚ) : |x * y| ≤ mag16 x * mag16 y :=
  Dot8Chunk.abs_mul_le_liftMag .f16 .f16 x y

theorem relDot16_64 : Rules.Num.relDot16 64 = 66 / 2 ^ 22 := by
  unfold Rules.Num.relDot16; norm_num

theorem absDot16_64 : Rules.Num.absDot16 64 = 66 / 2 ^ 126 := by
  unfold Rules.Num.absDot16; norm_num

theorem relDot16_nonneg : (0 : ℚ) ≤ Rules.Num.relDot16 64 := by rw [relDot16_64]; norm_num

theorem absDot16_nonneg : (0 : ℚ) ≤ Rules.Num.absDot16 64 := by rw [absDot16_64]; norm_num

theorem f16_abs_le {x : F16} (hx : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have := FloatV.abs_toRat_le_maxVal Math.Float.Format.binary16_regular hx
  rwa [Math.Float.Format.binary16_maxVal] at this

theorem mag16_le {x : F16} (hx : x.isFinite = true) : mag16 x.toRat ≤ 65504 := by
  unfold mag16
  split_ifs
  · norm_num
  · exact max_le (f16_abs_le hx) (by norm_num)

theorem liftV_nonneg (va vb : Nat → F16) (K e : Nat) : 0 ≤ liftV va vb K e := by
  unfold liftV
  split_ifs
  · exact mul_nonneg (mag16_nonneg _) (mag16_nonneg _)
  · exact le_rfl

theorem abs_prodV_le {va vb : Nat → F16} {K : Nat} (e : Nat) : |prodV va vb K e| ≤ liftV va vb K e := by
  unfold prodV liftV
  split_ifs
  · exact abs_mul_le_mag16 _ _
  · simp

theorem liftV_le {va vb : Nat → F16} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (e : Nat) : liftV va vb K e ≤ 2 ^ 32 := by
  unfold liftV
  split_ifs with h
  · have h1 := mag16_le (hfa e h)
    have h2 := mag16_le (hfb e h)
    have h3 : mag16 (va e).toRat * mag16 (vb e).toRat ≤ 65504 * 65504 :=
      mul_le_mul h1 h2 (mag16_nonneg _) (by norm_num)
    linarith
  · norm_num

theorem chunkA_nonneg (va vb : Nat → F16) (K u : Nat) : 0 ≤ chunkA va vb K u :=
  Finset.sum_nonneg fun _ _ => liftV_nonneg _ _ _ _

theorem abs_chunkS_le (va vb : Nat → F16) (K u : Nat) : |chunkS va vb K u| ≤ chunkA va vb K u :=
  (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun _ _ => abs_prodV_le _)

theorem chunkA_le {va vb : Nat → F16} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (u : Nat) : chunkA va vb K u ≤ 2 ^ 38 := by
  unfold chunkA
  calc ∑ k : Fin 64, liftV va vb K (64 * u + k) ≤ ∑ _k : Fin 64, (2 : ℚ) ^ 32 :=
        Finset.sum_le_sum fun k _ => liftV_le hfa hfb _
    _ = 2 ^ 38 := by simp; norm_num

theorem val_f16 {x : F16} {v : ℚ} (hx : x.isFinite = true) (hv : x.toRat = v) :
    DType.val? .f16 x = some v := by
  show (if x.isFinite = true then some x.toRat else none) = some v
  rw [ite_eq_left hx, hv]

/-- The chunks of a whole tile add up to the exact sums. -/
theorem chunks_end (va vb : Nat → F16) (K : Nat) :
    ∑ u ∈ Finset.range (steps K), chunkS va vb K u = dotSum va vb K ∧
      ∑ u ∈ Finset.range (steps K), chunkA va vb K u = dotMag va vb K := by
  have hT : K ≤ steps K * 64 := by unfold steps; omega
  have h1 : ∀ f : Nat → ℚ, ∑ u ∈ Finset.range (steps K), ∑ k : Fin 64, f (64 * u + k) =
      ∑ e ∈ Finset.range (steps K * 64), f e := by
    intro f
    rw [← DotLoop.sum_chunks f 64 (steps K)]
    refine Finset.sum_congr rfl fun u _ => ?_
    rw [Fin.sum_univ_eq_sum_range (fun k => f (64 * u + k)) 64]
    exact Finset.sum_congr rfl fun k _ => by rw [Nat.mul_comm]
  obtain ⟨x, hx⟩ : ∃ x, steps K * 64 = K + x := ⟨steps K * 64 - K, by omega⟩
  unfold chunkS chunkA dotSum dotMag
  rw [h1, h1, hx, Finset.sum_range_add, Finset.sum_range_add]
  have e1 : ∀ e ∈ Finset.range K, prodV va vb K e = (va e).toRat * (vb e).toRat := fun e he => by
    unfold prodV; rw [ite_eq_left (Finset.mem_range.mp he)]
  have e2 : ∀ y, prodV va vb K (K + y) = 0 := fun y => by
    unfold prodV; rw [ite_eq_right (by omega)]
  have e3 : ∀ e ∈ Finset.range K, liftV va vb K e = mag16 (va e).toRat * mag16 (vb e).toRat :=
    fun e he => by unfold liftV; rw [ite_eq_left (Finset.mem_range.mp he)]
  have e4 : ∀ y, liftV va vb K (K + y) = 0 := fun y => by
    unfold liftV; rw [ite_eq_right (by omega)]
  constructor
  · rw [Finset.sum_congr rfl e1]; simp [e2]
  · rw [Finset.sum_congr rfl e3]; simp [e4]

/-! ## One step of a tile's part -/

/-- **One step of the inner loop** (line 57), for one output element `(i, j)`: from the
accumulator after `cnt` chunks of the part starting at chunk `lo`, finite and within
`DotLoop.bound` of their exact sum (relative to their lifted magnitudes), the next chunk keeps it
so, by `Rules.Num.DotF16` (`Rules.Num.DotLiftAcc`) and `DotLoop.accum_step`. `A` and `B` are the
loaded tiles: their enabled lanes hold the matrix elements, their disabled lanes finite zeros. -/
theorem acc_step {P : Platform} (hdot : Rules.Num.DotF16 P)
    {A : Tensor [128, 64] F16} {B : Tensor [64, 128] F16} {acc r : Tensor [128, 128] F32}
    (hr : P.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 } A B acc r)
    (i : Fin 128) (j : Fin 128) {va vb : Nat → F16} {K lo cnt : Nat} (hK : K ≤ 2 ^ 16)
    (hc : 64 * (lo + cnt) < K)
    (hfa : ∀ e, e < K → (va e).isFinite = true) (hfb : ∀ e, e < K → (vb e).isFinite = true)
    (hA : ∀ k : Fin 64, (64 * (lo + cnt) + k < K → A.get2 i k = va (64 * (lo + cnt) + k)) ∧
      (K ≤ 64 * (lo + cnt) + k → (A.get2 i k).isFinite = true ∧ (A.get2 i k).toRat = 0))
    (hB : ∀ k : Fin 64, (64 * (lo + cnt) + k < K → B.get2 k j = vb (64 * (lo + cnt) + k)) ∧
      (K ≤ 64 * (lo + cnt) + k → (B.get2 k j).isFinite = true ∧ (B.get2 k j).toRat = 0))
    (hinv : (acc.get2 i j).isFinite = true ∧
      |(acc.get2 i j).toRat - ∑ u ∈ Finset.Ico lo (lo + cnt), chunkS va vb K u| ≤
        DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) cnt
          (∑ u ∈ Finset.Ico lo (lo + cnt), chunkA va vb K u)) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toRat - ∑ u ∈ Finset.Ico lo (lo + (cnt + 1)), chunkS va vb K u| ≤
        DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) (cnt + 1)
          (∑ u ∈ Finset.Ico lo (lo + (cnt + 1)), chunkA va vb K u) := by
  set c := lo + cnt with hcdef
  -- the lanes' values
  have hfA : ∀ k : Fin 64, (A.get2 i k).isFinite = true := fun k => by
    by_cases h : 64 * c + k < K
    · rw [(hA k).1 h]; exact hfa _ h
    · exact ((hA k).2 (by omega)).1
  have hfB : ∀ k : Fin 64, (B.get2 k j).isFinite = true := fun k => by
    by_cases h : 64 * c + k < K
    · rw [(hB k).1 h]; exact hfb _ h
    · exact ((hB k).2 (by omega)).1
  have hprod : ∀ k : Fin 64, (A.get2 i k).toRat * (B.get2 k j).toRat = prodV va vb K (64 * c + k) := by
    intro k
    unfold prodV
    by_cases h : 64 * c + k < K
    · rw [ite_eq_left h, (hA k).1 h, (hB k).1 h]
    · rw [ite_eq_right h, ((hA k).2 (by omega)).2, zero_mul]
  have hchunk : ∑ k : Fin 64, (A.get2 i k).toRat * (B.get2 k j).toRat = chunkS va vb K c :=
    Finset.sum_congr rfl fun k _ => hprod k
  have hlift : ∀ k : Fin 64, mag16 (A.get2 i k).toRat * mag16 (B.get2 k j).toRat = liftV va vb K (64 * c + k) := by
    intro k
    unfold liftV
    by_cases h : 64 * c + k < K
    · rw [ite_eq_left h, (hA k).1 h, (hB k).1 h]
    · rw [ite_eq_right h, ((hA k).2 (by omega)).2, mag16_zero, zero_mul]
  have hchunkA : ∑ k : Fin 64, Rules.Num.liftMag .f16 (A.get2 i k).toRat * Rules.Num.liftMag .f16 (B.get2 k j).toRat =
      chunkA va vb K c := Finset.sum_congr rfl fun k _ => hlift k
  -- sizes
  set St := ∑ u ∈ Finset.Ico lo c, chunkS va vb K u with hSt
  set At := ∑ u ∈ Finset.Ico lo c, chunkA va vb K u with hAt
  have hcnt : cnt ≤ 1024 := by omega
  have ht1 : (cnt : ℚ) ≤ 1024 := by exact_mod_cast hcnt
  have hB0 : At ≤ 2 ^ 48 := by
    have h1 : At ≤ ∑ _u ∈ Finset.Ico lo c, (2 : ℚ) ^ 38 :=
      Finset.sum_le_sum fun u _ => chunkA_le hfa hfb u
    rw [Finset.sum_const, Nat.card_Ico, nsmul_eq_mul] at h1
    have h2 : ((c - lo : Nat) : ℚ) ≤ 1024 := by
      have : c - lo ≤ 1024 := by omega
      exact_mod_cast this
    nlinarith
  have hBn : 0 ≤ At := Finset.sum_nonneg fun _ _ => chunkA_nonneg _ _ _ _
  have hcA : chunkA va vb K c ≤ 2 ^ 38 := chunkA_le hfa hfb c
  have hcA0 : 0 ≤ chunkA va vb K c := chunkA_nonneg _ _ _ _
  have hρ0 := relDot16_nonneg
  have hα0 := absDot16_nonneg
  have hbound : DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) cnt At ≤ 2 ^ 44 := by
    have htρ : (cnt : ℚ) * Rules.Num.relDot16 64 ≤ 1 / 2 := by
      rw [relDot16_64]; nlinarith
    refine (DotLoop.bound_le hρ0 hα0 htρ hBn).trans ?_
    rw [relDot16_64, absDot16_64]
    have ht0 : (0 : ℚ) ≤ cnt := Nat.cast_nonneg cnt
    nlinarith
  have hStA : |St| ≤ At :=
    (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun u _ => abs_chunkS_le _ _ _ u)
  have hSc : |chunkS va vb K c| ≤ chunkA va vb K c := abs_chunkS_le _ _ _ c
  have hvc : |(acc.get2 i j).toRat| ≤ 2 ^ 49 := by
    have := abs_sub_abs_le_abs_sub (acc.get2 i j).toRat St
    linarith [hinv.2]
  -- the step, by `DotF16`
  obtain ⟨vr, hvr, herr⟩ := hdot .tf32 A B acc r hr i j (fun k => (A.get2 i k).toRat)
    (fun k => (B.get2 k j).toRat) (acc.get2 i j).toRat (fun k => val_f16 (hfA k) rfl)
    (fun k => val_f16 (hfB k) rfl)
    (DotLoop.val?_f32.mpr ⟨hinv.1, rfl⟩) (by
      simp only [DType.bound, Math.Float.Format.binary32_maxVal]
      rw [hchunkA, relDot16_64, absDot16_64]
      have h0 := abs_nonneg (acc.get2 i j).toRat
      nlinarith)
  obtain ⟨hrfin, hrval⟩ := DotLoop.val?_f32.mp hvr
  refine ⟨hrfin, ?_⟩
  have e1 : lo + (cnt + 1) = c + 1 := by omega
  rw [e1, Finset.sum_Ico_succ_top (by omega), Finset.sum_Ico_succ_top (by omega), hrval]
  rw [hchunk, hchunkA] at herr
  exact DotLoop.accum_step hρ0 hα0 hStA hSc hinv.2 herr

/-! ## The bound -/

theorem bound_mono {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {t t' : ℕ} {B B' : ℚ} (ht : t ≤ t')
    (hB : 0 ≤ B) (hBB : B ≤ B') : DotLoop.bound ρ α t B ≤ DotLoop.bound ρ α t' B' := by
  unfold DotLoop.bound
  have h1 : 1 ≤ (1 + ρ) ^ t := one_le_pow₀ (by linarith)
  have h2 : (1 + ρ) ^ t ≤ (1 + ρ) ^ t' := pow_le_pow_right₀ (by linarith) ht
  have ht' : (t : ℚ) ≤ t' := by exact_mod_cast ht
  have e1 : ((1 + ρ) ^ t - 1) * B ≤ ((1 + ρ) ^ t' - 1) * B' := mul_le_mul (by linarith) hBB hB (by linarith)
  have e2 : (t : ℚ) * (1 + ρ) ^ t * α ≤ t' * (1 + ρ) ^ t' * α :=
    mul_le_mul_of_nonneg_right (mul_le_mul ht' h2 (by positivity) (by positivity)) hα
  linarith

/-- **The parts' bounds add up to at most the whole's**: steps `n i` and magnitudes `A i ≥ 0` with
`Σ n i = T` give `Σ bound ρ α (n i) (A i) ≤ bound ρ α T (Σ A i)`. -/
theorem sum_bound_le {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {ι : Type} (s : Finset ι) (n : ι → ℕ)
    (A : ι → ℚ) (hA : ∀ i ∈ s, 0 ≤ A i) {T : ℕ} (hT : ∑ i ∈ s, n i = T) :
    ∑ i ∈ s, DotLoop.bound ρ α (n i) (A i) ≤ DotLoop.bound ρ α T (∑ i ∈ s, A i) := by
  have hle : ∀ i ∈ s, n i ≤ T := fun i hi => hT ▸ Finset.single_le_sum (fun _ _ => Nat.zero_le _) hi
  have h1 : ∀ i ∈ s, DotLoop.bound ρ α (n i) (A i) ≤
      ((1 + ρ) ^ T - 1) * A i + (n i : ℚ) * ((1 + ρ) ^ T * α) := by
    intro i hi
    unfold DotLoop.bound
    have h2 : (1 + ρ) ^ n i ≤ (1 + ρ) ^ T := pow_le_pow_right₀ (by linarith) (hle i hi)
    have e1 : ((1 + ρ) ^ n i - 1) * A i ≤ ((1 + ρ) ^ T - 1) * A i :=
      mul_le_mul_of_nonneg_right (by linarith) (hA i hi)
    have e2 : (n i : ℚ) * (1 + ρ) ^ n i * α ≤ (n i : ℚ) * ((1 + ρ) ^ T * α) := by
      rw [mul_assoc]
      exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right h2 hα) (Nat.cast_nonneg _)
    linarith
  refine (Finset.sum_le_sum h1).trans (le_of_eq ?_)
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_mul, ← Nat.cast_sum, hT]
  unfold DotLoop.bound
  ring

/-! ## The tolerance -/

theorem tolDot_nonneg (K : Nat) {A : ℚ} (hA : 0 ≤ A) : 0 ≤ tolDot K A :=
  DotLoop.bound_nonneg relDot16_nonneg absDot16_nonneg _ hA

theorem tol_nonneg (K : Nat) {A : ℚ} (hA : 0 ≤ A) : 0 ≤ tol K A := by
  unfold tol
  have := tolDot_nonneg K hA
  have := DotLoop.bound_nonneg (ρ := 2 ^ (-23 : ℤ)) (α := 2 ^ (-124 : ℤ)) (by positivity) (by positivity)
    (steps K) (B := A + tolDot K A) (by linarith)
  linarith

theorem tolDot_le_tol (K : Nat) {A : ℚ} (hA : 0 ≤ A) : tolDot K A ≤ tol K A := by
  unfold tol
  have := tolDot_nonneg K hA
  have := DotLoop.bound_nonneg (ρ := 2 ^ (-23 : ℤ)) (α := 2 ^ (-124 : ℤ)) (by positivity) (by positivity)
    (steps K) (B := A + tolDot K A) (by linarith)
  linarith

/-- **The tolerance on covered instances** (`K ≤ 2^13`, `Inst.ok.K_le`): at most `0.21%` of `A`,
plus `10^-33`. `0.21%` is about half the largest relative error of a rounding to `bfloat16`
(`2^-8/(1 + 2^-8)`, about `0.39%`). -/
theorem tol_arith {A D Y c : ℚ} (hA : 0 ≤ A) (hD : D ≤ 20163 / 10 ^ 7 * A + 2 / 10 ^ 34) (hD0 : 0 ≤ D)
    (hY1 : Y ≤ 153 / 10 ^ 7) (hc : c ≤ 2 / 10 ^ 34) :
    D + (Y * (A + D) + c) ≤ 21 / 10000 * A + 1 / 10 ^ 33 := by
  have h1 : Y * (A + D) ≤ 153 / 10 ^ 7 * (A + D) := mul_le_mul_of_nonneg_right hY1 (by linarith)
  nlinarith

theorem tol_le {K : Nat} (hK : K ≤ 2 ^ 13) {A : ℚ} (hA : 0 ≤ A) :
    tol K A ≤ 21 / 10000 * A + 1 / 10 ^ 33 := by
  have hT : steps K ≤ 128 := by unfold steps; omega
  have hρ := relDot16_nonneg
  have hα := absDot16_nonneg
  have hρa : (0 : ℚ) ≤ 2 ^ (-23 : ℤ) := by positivity
  have hαa : (0 : ℚ) ≤ 2 ^ (-124 : ℤ) := by positivity
  set D := DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) 128 A with hDdef
  have h1 : tolDot K A ≤ D := bound_mono hρ hα hT hA le_rfl
  have h0 := tolDot_nonneg K hA
  have h2 : DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (steps K) (A + tolDot K A) ≤
      DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) 128 (A + D) :=
    bound_mono hρa hαa hT (by linarith) (by linarith)
  have hD0 : 0 ≤ D := DotLoop.bound_nonneg hρ hα _ hA
  have hD : D ≤ 20163 / 10 ^ 7 * A + 2 / 10 ^ 34 := by
    rw [hDdef]
    unfold DotLoop.bound
    rw [relDot16_64, absDot16_64]
    have hx1 : (1 + 66 / 2 ^ 22 : ℚ) ^ 128 - 1 ≤ 20163 / 10 ^ 7 := by norm_num
    have hx2 : (1 + 66 / 2 ^ 22 : ℚ) ^ 128 ≤ 2 := by norm_num
    have e1 : ((1 + 66 / 2 ^ 22 : ℚ) ^ 128 - 1) * A ≤ 20163 / 10 ^ 7 * A := mul_le_mul_of_nonneg_right hx1 hA
    have e2 : ((128 : ℕ) : ℚ) * (1 + 66 / 2 ^ 22 : ℚ) ^ 128 * (66 / 2 ^ 126) ≤ 2 / 10 ^ 34 := by
      push_cast
      have : (128 : ℚ) * (1 + 66 / 2 ^ 22 : ℚ) ^ 128 * (66 / 2 ^ 126) ≤ 128 * 2 * (66 / 2 ^ 126) := by
        gcongr
      refine this.trans ?_
      norm_num
    linarith
  have hatom : DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) 128 (A + D) =
      ((1 + 2 ^ (-23 : ℤ)) ^ 128 - 1) * (A + D) + ((128 : ℕ) : ℚ) * (1 + 2 ^ (-23 : ℤ)) ^ 128 * 2 ^ (-124 : ℤ) := by
    unfold DotLoop.bound; ring
  have hy1 : (1 + 2 ^ (-23 : ℤ) : ℚ) ^ 128 - 1 ≤ 153 / 10 ^ 7 := by norm_num
  have hc : ((128 : ℕ) : ℚ) * (1 + 2 ^ (-23 : ℤ) : ℚ) ^ 128 * 2 ^ (-124 : ℤ) ≤ 2 / 10 ^ 34 := by
    have hy2 : (1 + 2 ^ (-23 : ℤ) : ℚ) ^ 128 ≤ 2 := by norm_num
    have : ((128 : ℕ) : ℚ) * (1 + 2 ^ (-23 : ℤ) : ℚ) ^ 128 * 2 ^ (-124 : ℤ) ≤ 128 * 2 * 2 ^ (-124 : ℤ) := by
      push_cast; gcongr
    refine this.trans ?_
    norm_num
  unfold tol
  refine (add_le_add h1 h2).trans ?_
  rw [hatom]
  exact tol_arith hA hD hD0 hy1 hc

end Kernels.StreamKMatmul
