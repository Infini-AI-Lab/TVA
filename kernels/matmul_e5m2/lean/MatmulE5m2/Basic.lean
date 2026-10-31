import MatmulE5m2.Defs
import MatmulE5m2.Arith

/-!
# `matmul_e5m2`: facts the per-program proof uses

* `int32` arithmetic on small values (`i32_add`, `i32_sub`, `i32_mul`, `i32_lt`, `i32_le`);
* the zero `other=0.0` converted to `float8e5` (`zero_conv`), from `Rules.Num.CvtF32E5m2`;
* the accumulation of one `tl.dot` step (`acc_step`), from `Rules.Num.DotE5m2`;
* the last rounding to `float16` (`out_f16`).
-/

namespace Kernels.MatmulE5m2

open TVA

/-! ## `int32` arithmetic -/

theorem i32_add {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x + y) (h1 : x + y < 2 ^ 31) : (a + b).toInt = x + y := by
  rw [IntV.toInt_add (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_sub {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x - y) (h1 : x - y < 2 ^ 31) : (a - b).toInt = x - y := by
  rw [IntV.toInt_sub (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_mul {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x * y) (h1 : x * y < 2 ^ 31) : (a * b).toInt = x * y := by
  rw [IntV.toInt_mul (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_nat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n := IntV.I32.toInt_ofNat h

theorem i32_lt {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    IntV.lt a b = decide (x < y) := by
  rw [IntV.lt_eq, ha, hb]

theorem i32_le {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    IntV.le a b = decide (x ≤ y) := by
  rw [IntV.le_eq, ha, hb]

theorem i32_zero : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)

theorem i32_one : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)

/-- `int32` `%` of a nonnegative value by a positive one. -/
theorem i32_rem {x y q : I32} (hx : 0 ≤ x.toInt) (hy : 0 < y.toInt) (h : x.rem? y = some q) :
    q.toInt = x.toInt % y.toInt := by
  obtain ⟨q', hq', hv⟩ := IntV.rem?_of_nonneg hx hy
  rw [hq'] at h
  cases h
  exact hv

/-- `int32` `min`. -/
theorem i32_min {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    (IntV.min a b).toInt = min x y := by
  unfold IntV.min
  rw [i32_lt ha hb]
  by_cases h : x < y
  · simp [h, ha]; omega
  · simp [h, hb]; omega

/-! ## `other=0.0` -/

theorem f32_zero_lit : (FloatV.litF32 (FloatV.ofPat 0) : F32) = FloatV.ofPat 0 := by
  unfold FloatV.litF32
  rw [ite_eq_left]
  rw [FloatV.isZero_ofPat (by norm_num)]
  decide

/-- The rounding to nearest even of the `float32` `+0` into `float8e5` is a finite zero. -/
theorem e5m2_cvt_zero : (FloatV.cvtWith .rn (FloatV.ofPat 0 : F32) : F8E5).isFinite = true ∧
    (FloatV.cvtWith .rn (FloatV.ofPat 0 : F32) : F8E5).toRat = 0 := by
  have h0 : (FloatV.ofPat 0 : F32).pat = 0 := FloatV.pat_ofPat (by norm_num)
  have hfin : Math.Float.Format.binary32.isFinite 0 = true := by decide
  have hsign : Math.Float.Format.binary32.sign 0 = false := by decide
  have htr : Math.Float.Format.binary32.toRat 0 = 0 :=
    Math.Float.Format.toRat_of_isZero (by decide)
  unfold FloatV.cvtWith
  rw [h0, Math.Float.Format.cvt_of_finite _ _ _ hfin, hsign, htr]
  obtain ⟨hlt, hf, hr⟩ := Math.Float.Format.roundRat_zero_spec (f := Math.Float.Format.e5m2)
    (rnd := .rn) (sat := false) (zs := false) Math.Float.Format.e5m2_regular
  rw [FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨hf, hr⟩

/-- The source's `other=0.0` converted to `float8e5` has some result (`Rules.Num.CvtTotal`). -/
theorem zero_conv_ex {P : Platform}
    (htot : Rules.Num.CvtTotal P Math.Float.Format.binary32 Math.Float.Format.e5m2 .rn) :
    ∃ r, ConvRel P .f32 .f8e5 (FloatV.litF32 (FloatV.ofPat 0) : F32) r := by
  obtain ⟨r, hr⟩ := htot (FloatV.litF32 (FloatV.ofPat 0) : F32)
  refine ⟨r, ?_⟩
  show FloatRel P Math.Float.Format.binary32 Math.Float.Format.e5m2 _ _
  unfold FloatRel
  rw [ite_eq_right (by decide)]
  exact Or.inr (Or.inl ⟨Or.inr (Or.inr rfl), Or.inl rfl, hr⟩)

/-- Every result of the source's `other=0.0` converted to `float8e5` is a finite zero
(`Rules.Num.CvtF32E5m2`: the rounding to nearest even of the `float32` `+0`). -/
theorem zero_conv {P : Platform} (hcvt : Rules.Num.CvtF32E5m2 P) {r : F8E5}
    (h : ConvRel P .f32 .f8e5 (FloatV.litF32 (FloatV.ofPat 0) : F32) r) :
    r.isFinite = true ∧ r.toRat = 0 := by
  change FloatRel P Math.Float.Format.binary32 Math.Float.Format.e5m2 _ r at h
  unfold FloatRel at h
  rw [ite_eq_right (by decide), f32_zero_lit] at h
  obtain ⟨hf, hr⟩ := e5m2_cvt_zero
  have h0 : (FloatV.ofPat 0 : F32).pat = 0 := FloatV.pat_ofPat (by norm_num)
  have hx : (FloatV.ofPat 0 : F32).isFinite = true := by
    unfold FloatV.isFinite; rw [h0]; decide
  have hxv : (FloatV.ofPat 0 : F32).toRat = 0 := by
    unfold FloatV.toRat; rw [h0]; exact Math.Float.Format.toRat_of_isZero (by decide)
  rcases h with ⟨-, hstd, -⟩ | ⟨-, -, hc⟩ | ⟨hF8, -, -⟩
  · exact absurd hstd (by unfold Math.Float.Format.Std; decide)
  · have he := hcvt _ _ hc hx (by rw [hxv, Math.Float.Format.e5m2_maxVal]; norm_num)
    have hn : (FloatV.cvtWith .rn (FloatV.ofPat 0 : F32) : F8E5).isNaN = false :=
      (FloatV.isFinite_iff.mp hf).1
    rw [he.eq hn]
    exact ⟨hf, hr⟩
  · exact absurd hF8 (by unfold Math.Float.Format.F8; decide)

/-! ## The accumulation -/

/-- The exact partial sum after `t` steps: the products `e < 64·t`. -/
def partS (va vb : Nat → F8E5) (K t : Nat) : ℚ := partSum (prodV va vb K) t

/-- The partial sum of the magnitudes. -/
def partA (va vb : Nat → F8E5) (K t : Nat) : ℚ := partMag (prodV va vb K) t

theorem e5m2_abs_le {x : F8E5} (hx : x.isFinite = true) : |x.toRat| ≤ 57344 := by
  have := FloatV.abs_toRat_le_maxVal Math.Float.Format.e5m2_regular hx
  rwa [Math.Float.Format.e5m2_maxVal] at this

theorem prodV_le {va vb : Nat → F8E5} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (e : Nat) : |prodV va vb K e| ≤ 2 ^ 32 := by
  unfold prodV
  split_ifs with h
  · rw [abs_mul]
    have h1 := e5m2_abs_le (hfa e h)
    have h2 := e5m2_abs_le (hfb e h)
    have h3 : |(va e).toRat| * |(vb e).toRat| ≤ 57344 * 57344 :=
      mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
    linarith
  · simp

theorem partA_le {va vb : Nat → F8E5} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (t : Nat) : partA va vb K t ≤ 64 * t * 2 ^ 32 := by
  unfold partA partMag
  calc ∑ e ∈ Finset.range (64 * t), |prodV va vb K e| ≤ ∑ _e ∈ Finset.range (64 * t), (2 : ℚ) ^ 32 :=
        Finset.sum_le_sum fun e _ => prodV_le hfa hfb e
    _ = 64 * t * 2 ^ 32 := by simp

theorem partS_succ (va vb : Nat → F8E5) (K t : Nat) :
    partS va vb K (t + 1) = partS va vb K t + ∑ k : Fin 64, prodV va vb K (64 * t + k) :=
  partSum_succ _ t

theorem partA_succ (va vb : Nat → F8E5) (K t : Nat) :
    partA va vb K (t + 1) = partA va vb K t + ∑ k : Fin 64, |prodV va vb K (64 * t + k)| :=
  partMag_succ _ t

theorem val_f8e5 {x : F8E5} {v : ℚ} (hx : x.isFinite = true) (hv : x.toRat = v) :
    DType.val? .f8e5 x = some v := by
  show (if x.isFinite = true then some x.toRat else none) = some v
  rw [ite_eq_left hx, hv]

/-- `(1 + ρ)^t ≤ 3` for the `t ≤ 64` steps of a covered instance. -/
theorem pow_rho_le {t : Nat} (ht : t ≤ 64) : (1 + Rules.Num.relDot8 64) ^ t ≤ 3 := by
  have h1 : (1 : ℚ) ≤ 1 + Rules.Num.relDot8 64 := by rw [relDot8_64]; norm_num
  have h2 := pow_le_pow_right₀ h1 ht
  have h3 : (1 + Rules.Num.relDot8 64) ^ 64 < 3 := by rw [relDot8_64]; norm_num
  linarith

/-- One step of the error bound: an accumulator `a` within `E` of the exact partial sum `S`,
to which a step adds `X` up to `ρ·(|a| + B) + α`, is then within `(1 + ρ)·E + ρ·(|S| + B) + α`
of `S + X`, since `|a| ≤ |S| + E`. -/
theorem err_step {ρ α a a' S X B E : ℚ} (hρ : 0 ≤ ρ) (he : |a - S| ≤ E)
    (hs : |a' - (a + X)| ≤ ρ * (|a| + B) + α) :
    |a' - (S + X)| ≤ (1 + ρ) * E + ρ * (|S| + B) + α := by
  have h1 : |a| ≤ |S| + E := by
    have := abs_sub_abs_le_abs_sub a S
    linarith
  have h2 : |a' - (S + X)| ≤ |a' - (a + X)| + |a - S| := by
    have e : a' - (S + X) = (a' - (a + X)) + (a - S) := by ring
    rw [e]; exact abs_add_le _ _
  have h3 : ρ * |a| ≤ ρ * (|S| + E) := mul_le_mul_of_nonneg_left h1 hρ
  nlinarith

/-- **One step of the loop** (line 300), for one output element `(i, j)` of the tile: the
accumulator stays finite and within the error bound `errAt` of the exact partial sum, by
`Rules.Num.DotE5m2` (`err_step`). `A` and `B` are the loaded tiles: their enabled lanes
hold the matrix elements, their disabled lanes finite zeros. -/
theorem acc_step {P : Platform} (hdot : Rules.Num.DotE5m2 P)
    {A : Tensor [128, 64] F8E5} {B : Tensor [64, 256] F8E5} {acc r : Tensor [128, 256] F32}
    (hr : P.dot { a := .f8e5, b := .f8e5, acc := .f32, precision := .tf32, maxImpreciseAcc := 2 ^ 30 } A B acc r)
    (i : Fin 128) (j : Fin 256) {va vb : Nat → F8E5} {K t : Nat} (hK : K ≤ 2 ^ 12) (ht : 64 * t < K)
    (hfa : ∀ e, e < K → (va e).isFinite = true) (hfb : ∀ e, e < K → (vb e).isFinite = true)
    (hA : ∀ k : Fin 64, (64 * t + k < K → A.get2 i k = va (64 * t + k)) ∧
      (K ≤ 64 * t + k → (A.get2 i k).isFinite = true ∧ (A.get2 i k).toRat = 0))
    (hB : ∀ k : Fin 64, (64 * t + k < K → B.get2 k j = vb (64 * t + k)) ∧
      (K ≤ 64 * t + k → (B.get2 k j).isFinite = true ∧ (B.get2 k j).toRat = 0))
    (hinv : (acc.get2 i j).isFinite = true ∧
      |(acc.get2 i j).toRat - partS va vb K t| ≤ errAt (prodV va vb K) t) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toRat - partS va vb K (t + 1)| ≤ errAt (prodV va vb K) (t + 1) := by
  -- the lanes' values
  have hfA : ∀ k : Fin 64, (A.get2 i k).isFinite = true := fun k => by
    by_cases h : 64 * t + k < K
    · rw [(hA k).1 h]; exact hfa _ h
    · exact ((hA k).2 (by omega)).1
  have hfB : ∀ k : Fin 64, (B.get2 k j).isFinite = true := fun k => by
    by_cases h : 64 * t + k < K
    · rw [(hB k).1 h]; exact hfb _ h
    · exact ((hB k).2 (by omega)).1
  have hprod : ∀ k : Fin 64, (A.get2 i k).toRat * (B.get2 k j).toRat = prodV va vb K (64 * t + k) := by
    intro k
    unfold prodV
    by_cases h : 64 * t + k < K
    · rw [ite_eq_left h, (hA k).1 h, (hB k).1 h]
    · rw [ite_eq_right h, ((hA k).2 (by omega)).2, zero_mul]
  have hchunk : ∑ k : Fin 64, (A.get2 i k).toRat * (B.get2 k j).toRat =
      ∑ k : Fin 64, prodV va vb K (64 * t + k) := Finset.sum_congr rfl fun k _ => hprod k
  have hchunkA : ∑ k : Fin 64, |(A.get2 i k).toRat * (B.get2 k j).toRat| =
      ∑ k : Fin 64, |prodV va vb K (64 * t + k)| := Finset.sum_congr rfl fun k _ => by rw [hprod k]
  -- sizes
  have ht64 : t ≤ 64 := by omega
  have ht1 : (t : ℚ) ≤ 64 := by exact_mod_cast ht64
  have ht0 : (0 : ℚ) ≤ t := Nat.cast_nonneg t
  have hB0 : partA va vb K t ≤ 2 ^ 44 := by
    have h1 := partA_le hfa hfb t
    have h2 : (64 * t : ℚ) ≤ 2 ^ 12 := by
      have : 64 * t ≤ 2 ^ 12 := by omega
      exact_mod_cast this
    nlinarith
  have hBn : 0 ≤ partA va vb K t := by
    unfold partA partMag; exact Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hcA : ∑ k : Fin 64, |prodV va vb K (64 * t + k)| ≤ 2 ^ 38 := by
    calc ∑ k : Fin 64, |prodV va vb K (64 * t + k)| ≤ ∑ _k : Fin 64, (2 : ℚ) ^ 32 :=
          Finset.sum_le_sum fun k _ => prodV_le hfa hfb _
      _ = 2 ^ 38 := by simp; norm_num
  have hcA0 : 0 ≤ ∑ k : Fin 64, |prodV va vb K (64 * t + k)| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hρ0 : (0 : ℚ) ≤ Rules.Num.relDot8 64 := by rw [relDot8_64]; norm_num
  have hα0 : (0 : ℚ) ≤ Rules.Num.absDot8 64 := by rw [absDot8_64]; norm_num
  have hP3 := pow_rho_le ht64
  have hP1 : (1 : ℚ) ≤ (1 + Rules.Num.relDot8 64) ^ t := one_le_pow₀ (by linarith)
  have hbound : DotLoop.bound (Rules.Num.relDot8 64) (Rules.Num.absDot8 64) t (partA va vb K t) ≤ 2 ^ 46 := by
    unfold DotLoop.bound
    rw [absDot8_64]
    have e1 : ((1 + Rules.Num.relDot8 64) ^ t - 1) * partA va vb K t ≤ 2 * 2 ^ 44 :=
      mul_le_mul (by linarith) hB0 hBn (by norm_num)
    have e2 : (t : ℚ) * (1 + Rules.Num.relDot8 64) ^ t ≤ 64 * 3 :=
      mul_le_mul ht1 hP3 (by linarith) (by norm_num)
    have e3 : (t : ℚ) * (1 + Rules.Num.relDot8 64) ^ t * (16641 / 2 ^ 139) ≤ 64 * 3 * (16641 / 2 ^ 139) :=
      mul_le_mul_of_nonneg_right e2 (by norm_num)
    have e4 : (64 * 3 * (16641 / 2 ^ 139) : ℚ) ≤ 1 := by norm_num
    linarith
  have hSt : |partS va vb K t| ≤ partA va vb K t := by
    unfold partS partA partSum partMag; exact Finset.abs_sum_le_sum_abs _ _
  have herrB : errAt (prodV va vb K) t ≤ 2 ^ 46 := (errAt_le_bound _ t).trans hbound
  have hvc : |(acc.get2 i j).toRat| ≤ 2 ^ 47 := by
    have := abs_sub_abs_le_abs_sub (acc.get2 i j).toRat (partS va vb K t)
    linarith [hinv.2]
  -- the step, by `DotE5m2`
  obtain ⟨vr, hvr, herr⟩ := hdot .tf32 A B acc r hr i j (fun k => (A.get2 i k).toRat)
    (fun k => (B.get2 k j).toRat) (acc.get2 i j).toRat (fun k => val_f8e5 (hfA k) rfl)
    (fun k => val_f8e5 (hfB k) rfl) (fun _ => ⟨trivial, trivial⟩)
    (DotLoop.val?_f32.mpr ⟨hinv.1, rfl⟩) (by
      simp only [DType.bound, Math.Float.Format.binary32_maxVal]
      rw [hchunkA, relDot8_64, absDot8_64]
      have h0 := abs_nonneg (acc.get2 i j).toRat
      nlinarith)
  obtain ⟨hrfin, hrval⟩ := DotLoop.val?_f32.mp hvr
  refine ⟨hrfin, ?_⟩
  rw [partS_succ, hrval]
  rw [hchunk, hchunkA] at herr
  exact err_step hρ0 hinv.2 herr

/-- After the last step (`64·T ≥ K`) the partial sums are the whole sums. -/
theorem partS_end (va vb : Nat → F8E5) {K T : Nat} (hT : K ≤ 64 * T) :
    partS va vb K T = dotSum va vb K ∧ partA va vb K T = dotAbs va vb K := by
  unfold partS partA partSum partMag dotSum dotAbs
  obtain ⟨x, hx⟩ : ∃ x, 64 * T = K + x := ⟨64 * T - K, by omega⟩
  rw [hx, Finset.sum_range_add, Finset.sum_range_add]
  have h1 : ∀ e ∈ Finset.range K, prodV va vb K e = (va e).toRat * (vb e).toRat := fun e he => by
    unfold prodV; rw [ite_eq_left (Finset.mem_range.mp he)]
  have h2 : ∀ y, prodV va vb K (K + y) = 0 := fun y => by
    unfold prodV; rw [ite_eq_right (by omega)]
  constructor
  · rw [Finset.sum_congr rfl h1]; simp [h2]
  · rw [Finset.sum_congr rfl fun e he => by rw [h1 e he]]; simp [h2]

/-- **The tolerance is never above the bound in terms of `Σ|a·b|` alone**: `tol va vb K ≤
tolA K (dotAbs va vb K)` (`errAt_le_bound` after the loop's `steps K` steps). -/
theorem tol_le_tolA (va vb : Nat → F8E5) (K : Nat) : tol va vb K ≤ tolA K (dotAbs va vb K) := by
  have h := errAt_le_bound (prodV va vb K) (steps K)
  have hA := (partS_end va vb (K := K) (T := steps K) (by unfold steps; omega)).2
  unfold partA at hA
  unfold tol tolA
  rw [← hA]
  exact h

/-- **The last rounding** (line 308): a finite `v` within `E` of `S`, rounded to `float16`
(`c` equal to it up to NaN payload), is finite and within `E` plus one rounding when
`|S| + E ≤ 65504`. -/
theorem out_f16 {v : F32} {c : F16} {S E : ℚ} (hv : v.isFinite = true) (hE : |v.toRat - S| ≤ E)
    (hc : FloatV.NaNEq (FloatV.cvt v) c) (hroom : |S| + E ≤ Math.Float.Format.binary16.maxVal) :
    c.isFinite = true ∧
      |c.toRat - S| ≤ E + (|S| + E) * Math.Float.Format.binary16.u + Math.Float.Format.binary16.eta := by
  have hvS : |v.toRat| ≤ |S| + E := by
    have := abs_sub_abs_le_abs_sub v.toRat S
    linarith
  obtain ⟨hcf, hce⟩ := FloatV.cvt_err (g := Math.Float.Format.binary16)
    Math.Float.Format.binary16_regular hv (hvS.trans hroom)
  have hn : (FloatV.cvt v : F16).isNaN = false := (FloatV.isFinite_iff.mp hcf).1
  rw [hc.eq hn]
  refine ⟨hcf, ?_⟩
  have hu : 0 ≤ Math.Float.Format.binary16.u := Math.Float.Format.u_nonneg
  have e : (FloatV.cvt v : F16).toRat - S = ((FloatV.cvt v : F16).toRat - v.toRat) + (v.toRat - S) := by ring
  rw [e]
  refine (abs_add_le _ _).trans ?_
  have : |v.toRat| * Math.Float.Format.binary16.u ≤ (|S| + E) * Math.Float.Format.binary16.u :=
    mul_le_mul_of_nonneg_right hvS hu
  linarith

end Kernels.MatmulE5m2
