import MatmulE5m2.Defs
import MatmulE5m2.Arith

/-!
# `matmul_e5m2`: facts the per-program proof uses

* `int32` arithmetic on small values (`i32_add`, `i32_sub`, `i32_mul`, `i32_lt`, `i32_le`);
* the zero `other=0.0` converted to `float8e5` (`zero_conv`), from `Rules.Num.CvtF32E5m2`;
* the accumulation of one `tl.dot` step, two chunks of `32` (`acc_step`, `chunk_step`), from
  `Rules.Num.DotE5m2Chunk`; the error bound's size (`errAt_le_room`) and its bound by the lifted
  magnitudes alone (`tol_le_tolA`);
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

/-- `relDot8 64 = (1 + 2^-7)^2 − 1 = 257/2^14`: the relative constant of the room that
`Rules.Num.DotE5m2Chunk` asks of one `tl.dot` of `K = 64`. -/
theorem relDot8_64 : Rules.Num.relDot8 64 = 257 / 2 ^ 14 := by
  unfold Rules.Num.relDot8 Rules.Num.chunks8; norm_num

/-- `absDot8 64 = 2·(1 + 2^-7)^2·2^-126 = 16641/2^139`, its absolute constant. -/
theorem absDot8_64 : Rules.Num.absDot8 64 = 16641 / 2 ^ 139 := by
  unfold Rules.Num.absDot8 Rules.Num.chunks8; norm_num

/-- The exact partial sum after `j` chunks: the products `e < 32·j`. -/
def partS (va vb : Nat → F8E5) (K j : Nat) : ℚ := partSum (prodV va vb K) j

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

theorem liftV_nonneg (va vb : Nat → F8E5) (K e : Nat) : 0 ≤ liftV va vb K e := by
  unfold liftV
  split_ifs
  · exact mul_nonneg (Dot8Chunk.liftMag_nonneg _ _) (Dot8Chunk.liftMag_nonneg _ _)
  · exact le_rfl

theorem abs_prodV_le_liftV (va vb : Nat → F8E5) (K e : Nat) :
    |prodV va vb K e| ≤ liftV va vb K e := by
  unfold prodV liftV
  split_ifs
  · exact Dot8Chunk.abs_mul_le_liftMag _ _ _ _
  · simp

/-- A lifted product of finite `float8e5` values is below `2^32` (`liftMag x ≤ |x| + 2^-14`). -/
theorem liftV_le {va vb : Nat → F8E5} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (e : Nat) : liftV va vb K e ≤ 2 ^ 32 := by
  unfold liftV
  split_ifs with h
  · have h1 := Dot8Chunk.liftMag_le .f8e5 (va e).toRat
    have h2 := Dot8Chunk.liftMag_le .f8e5 (vb e).toRat
    have hm : Rules.Num.minNormal8 .f8e5 = 1 / 2 ^ 14 := by norm_num [Rules.Num.minNormal8]
    rw [hm] at h1 h2
    have ha := e5m2_abs_le (hfa e h)
    have hb := e5m2_abs_le (hfb e h)
    have h0b := Dot8Chunk.liftMag_nonneg .f8e5 (vb e).toRat
    have h3 : Rules.Num.liftMag .f8e5 (va e).toRat * Rules.Num.liftMag .f8e5 (vb e).toRat ≤
        (57344 + 1 / 2 ^ 14) * (57344 + 1 / 2 ^ 14) :=
      mul_le_mul (by linarith) (by linarith) h0b (by norm_num)
    have h4 : ((57344 + 1 / 2 ^ 14) * (57344 + 1 / 2 ^ 14) : ℚ) ≤ 2 ^ 32 := by norm_num
    linarith
  · norm_num

theorem partS_abs_le {va vb : Nat → F8E5} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (j : Nat) :
    |partS va vb K j| ≤ 32 * j * 2 ^ 32 := by
  unfold partS partSum
  calc |∑ e ∈ Finset.range (32 * j), prodV va vb K e|
        ≤ ∑ e ∈ Finset.range (32 * j), |prodV va vb K e| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _e ∈ Finset.range (32 * j), (2 : ℚ) ^ 32 := Finset.sum_le_sum fun e _ => prodV_le hfa hfb e
    _ = 32 * j * 2 ^ 32 := by simp

theorem partS_succ (va vb : Nat → F8E5) (K j : Nat) :
    partS va vb K (j + 1) = partS va vb K j + ∑ k : Fin 32, prodV va vb K (32 * j + k) :=
  partSum_succ _ j

theorem val_f8e5 {x : F8E5} {v : ℚ} (hx : x.isFinite = true) (hv : x.toRat = v) :
    DType.val? .f8e5 x = some v := by
  show (if x.isFinite = true then some x.toRat else none) = some v
  rw [ite_eq_left hx, hv]

/-- `(1 + ρ + σ)^j ≤ 3` for the `j ≤ 128` chunks of a covered instance. -/
theorem pow_q_le {j : Nat} (hj : j ≤ 128) : (4129 / 4096 : ℚ) ^ j ≤ 3 := by
  have h2 := pow_le_pow_right₀ (show (1 : ℚ) ≤ 4129 / 4096 by norm_num) hj
  have h3 : (4129 / 4096 : ℚ) ^ 128 < 3 := by norm_num
  linarith

/-- On a covered instance the error bound stays below `2^46` for the `j ≤ 128` chunks: each
product and lifted product is below `2^32`, so each partial sum is below `2^44`. -/
theorem errAt_le_room {va vb : Nat → F8E5} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) {j : Nat} (hj : j ≤ 128) :
    errAt (prodV va vb K) (liftV va vb K) j ≤ 2 ^ 46 := by
  have hS : ∀ i ≤ j, |partSum (prodV va vb K) i| ≤ 2 ^ 44 := fun i hi => by
    have h1 := partS_abs_le hfa hfb (K := K) i
    have h2 : (i : ℚ) ≤ 128 := by exact_mod_cast (show i ≤ 128 by omega)
    unfold partS at h1
    nlinarith
  have hq : ∀ i < j, chunkMax (liftV va vb K) i ≤ 2 ^ 44 := fun i _ =>
    chunkMax_le fun k => (liftV_le hfa hfb _).trans (by norm_num)
  refine (errAt_le_bound _ _ hS hq j le_rfl).trans ?_
  unfold DotLoop.bound
  rw [q_eq, alpha_eq]
  have hP := pow_q_le hj
  have hP1 : (1 : ℚ) ≤ (4129 / 4096) ^ j := one_le_pow₀ (by norm_num)
  have hj' : (j : ℚ) ≤ 128 := by exact_mod_cast hj
  have e1 : ((4129 / 4096 : ℚ) ^ j - 1) * 2 ^ 44 ≤ 2 * 2 ^ 44 :=
    mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
  have e2 : (j : ℚ) * (4129 / 4096) ^ j ≤ 128 * 3 := mul_le_mul hj' hP (by linarith) (by norm_num)
  have e3 : (j : ℚ) * (4129 / 4096) ^ j * (1 / 2 ^ 126) ≤ 128 * 3 * (1 / 2 ^ 126) :=
    mul_le_mul_of_nonneg_right e2 (by norm_num)
  have e4 : (128 * 3 * (1 / 2 ^ 126) : ℚ) ≤ 1 := by norm_num
  linarith

/-- **One chunk of the error bound**: an accumulator `s` within `E` of the exact partial sum `S`,
to which a chunk adds `c` up to `ρ·M + σ·|s + c| + α` for every `M` at least `|s|` and `m`, is then
within `(1 + ρ + σ)·E + ρ·max(|S|, m) + σ·|S'| + α` of `S' = S + c`: take `M = max(|S| + E, m)`;
`|s| ≤ |S| + E` and `|s + c| ≤ |S'| + E`. -/
theorem chunk_step {ρ σ α s s' S S' c m E : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (he : |s - S| ≤ E)
    (hS' : S' = S + c)
    (hstep : ∀ M, |s| ≤ M → m ≤ M → |s' - (s + c)| ≤ ρ * M + σ * |s + c| + α) :
    |s' - S'| ≤ (1 + ρ + σ) * E + ρ * max |S| m + σ * |S'| + α := by
  subst hS'
  have hE0 : 0 ≤ E := (abs_nonneg _).trans he
  have hs : |s| ≤ |S| + E := by
    have := abs_sub_abs_le_abs_sub s S
    linarith
  have h1 := hstep (max (|S| + E) m) (hs.trans (le_max_left _ _)) (le_max_right _ _)
  have hsc : |s + c| ≤ |S + c| + E := by
    have e : s + c = (S + c) + (s - S) := by ring
    rw [e]
    exact (abs_add_le _ _).trans (by linarith)
  have hmax : max (|S| + E) m ≤ max |S| m + E := by
    apply max_le
    · linarith [le_max_left |S| m]
    · linarith [le_max_right |S| m]
  have htri : |s' - (S + c)| ≤ |s' - (s + c)| + |s - S| := by
    have e : s' - (S + c) = (s' - (s + c)) + (s - S) := by ring
    rw [e]
    exact abs_add_le _ _
  have h2 := mul_le_mul_of_nonneg_left hmax hρ
  have h3 := mul_le_mul_of_nonneg_left hsc hσ
  linarith

/-- Chunk `j < 2` of a `tl.dot` of `K = 64` (`Rules.Num.chunk8 64 j`) is the `32` consecutive `k`
from `32·j`. -/
theorem sum_chunk8 {j : Nat} (hj : j < 2) (f : Nat → ℚ) :
    ∑ k ∈ Rules.Num.chunk8 64 j, f k.val = ∑ k : Fin 32, f (32 * j + k) := by
  symm
  refine Finset.sum_nbij' (fun k : Fin 32 => (⟨32 * j + k.val, by have := k.isLt; omega⟩ : Fin 64))
    (fun k : Fin 64 => (⟨k.val % 32, Nat.mod_lt _ (by norm_num)⟩ : Fin 32)) ?_ ?_ ?_ ?_ ?_
  · intro k _
    have := k.isLt
    simp only [Rules.Num.chunk8, Finset.mem_filter, Finset.mem_univ, true_and]
    show (32 * j + k.val) / 32 = j
    omega
  · intro k _
    exact Finset.mem_univ _
  · intro k _
    have := k.isLt
    apply Fin.ext
    show (32 * j + k.val) % 32 = k.val
    omega
  · intro k hk
    simp only [Rules.Num.chunk8, Finset.mem_filter, Finset.mem_univ, true_and] at hk
    apply Fin.ext
    show 32 * j + k.val % 32 = k.val
    omega
  · intro k _
    rfl

/-- **One step of the loop** (line 300), for one output element `(i, j)` of the tile: the
accumulator stays finite and within the error bound `errAt` of the exact partial sum, two chunks
further, by `Rules.Num.DotE5m2Chunk` (`chunk_step` for each chunk of the step's `tl.dot`; chunk `c`
of step `t` is chunk `2·t + c` of the loop). `A` and `B` are the loaded tiles: their enabled lanes
hold the matrix elements, their disabled lanes finite zeros. -/
theorem acc_step {P : Platform} (hdot : Rules.Num.DotE5m2Chunk P)
    {A : Tensor [128, 64] F8E5} {B : Tensor [64, 256] F8E5} {acc r : Tensor [128, 256] F32}
    (hr : P.dot { a := .f8e5, b := .f8e5, acc := .f32, precision := .tf32, maxImpreciseAcc := 2 ^ 30 } A B acc r)
    (i : Fin 128) (j : Fin 256) {va vb : Nat → F8E5} {K t : Nat} (hK : K ≤ 2 ^ 12) (ht : 64 * t < K)
    (hfa : ∀ e, e < K → (va e).isFinite = true) (hfb : ∀ e, e < K → (vb e).isFinite = true)
    (hA : ∀ k : Fin 64, (64 * t + k < K → A.get2 i k = va (64 * t + k)) ∧
      (K ≤ 64 * t + k → (A.get2 i k).isFinite = true ∧ (A.get2 i k).toRat = 0))
    (hB : ∀ k : Fin 64, (64 * t + k < K → B.get2 k j = vb (64 * t + k)) ∧
      (K ≤ 64 * t + k → (B.get2 k j).isFinite = true ∧ (B.get2 k j).toRat = 0))
    (hinv : (acc.get2 i j).isFinite = true ∧
      |(acc.get2 i j).toRat - partS va vb K (2 * t)| ≤ errAt (prodV va vb K) (liftV va vb K) (2 * t)) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toRat - partS va vb K (2 * (t + 1))| ≤
        errAt (prodV va vb K) (liftV va vb K) (2 * (t + 1)) := by
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
  have hlift : ∀ k : Fin 64, Rules.Num.liftMag .f8e5 (A.get2 i k).toRat *
      Rules.Num.liftMag .f8e5 (B.get2 k j).toRat = liftV va vb K (64 * t + k) := by
    intro k
    unfold liftV
    by_cases h : 64 * t + k < K
    · rw [ite_eq_left h, (hA k).1 h, (hB k).1 h]
    · rw [ite_eq_right h, ((hA k).2 (by omega)).2, Dot8Chunk.liftMag_zero, zero_mul]
  have hchunkA : ∑ k : Fin 64, |(A.get2 i k).toRat * (B.get2 k j).toRat| =
      ∑ k : Fin 64, |prodV va vb K (64 * t + k)| := Finset.sum_congr rfl fun k _ => by rw [hprod k]
  -- chunk `c` of the step is chunk `2·t + c` of the loop
  have hcs : ∀ c < 2, Rules.Num.chunkSum8 (fun k => (A.get2 i k).toRat) (fun k => (B.get2 k j).toRat) c =
      ∑ k : Fin 32, prodV va vb K (32 * (2 * t + c) + k) := by
    intro c hc
    unfold Rules.Num.chunkSum8
    rw [Finset.sum_congr rfl fun k _ => hprod k]
    have e : ∑ k ∈ Rules.Num.chunk8 64 c, prodV va vb K (64 * t + k) =
        ∑ k : Fin 32, prodV va vb K (64 * t + (32 * c + k)) :=
      sum_chunk8 hc (fun e => prodV va vb K (64 * t + e))
    rw [e]
    refine Finset.sum_congr rfl fun k _ => ?_
    congr 1
    ring
  have hcm : ∀ c < 2, ∀ k ∈ Rules.Num.chunk8 64 c,
      Rules.Num.liftMag .f8e5 (A.get2 i k).toRat * Rules.Num.liftMag .f8e5 (B.get2 k j).toRat ≤
        chunkMax (liftV va vb K) (2 * t + c) := by
    intro c hc k hk
    simp only [Rules.Num.chunk8, Finset.mem_filter, Finset.mem_univ, true_and] at hk
    rw [hlift k]
    have h : liftV va vb K (32 * (2 * t + c) + (k.val - 32 * c)) ≤ chunkMax (liftV va vb K) (2 * t + c) :=
      le_chunkMax _ _ ⟨k.val - 32 * c, by omega⟩
    have e : 32 * (2 * t + c) + (k.val - 32 * c) = 64 * t + k.val := by omega
    rw [e] at h
    exact h
  -- sizes
  have hEt : errAt (prodV va vb K) (liftV va vb K) (2 * t) ≤ 2 ^ 46 := errAt_le_room hfa hfb (by omega)
  have hSt : |partS va vb K (2 * t)| ≤ 2 ^ 44 := by
    have h1 := partS_abs_le hfa hfb (K := K) (2 * t)
    have h2 : ((2 * t : Nat) : ℚ) ≤ 128 := by exact_mod_cast (show 2 * t ≤ 128 by omega)
    nlinarith
  have hvc : |(acc.get2 i j).toRat| ≤ 2 ^ 47 := by
    have := abs_sub_abs_le_abs_sub (acc.get2 i j).toRat (partS va vb K (2 * t))
    linarith [hinv.2]
  have hcA : ∑ k : Fin 64, |prodV va vb K (64 * t + k)| ≤ 2 ^ 38 := by
    calc ∑ k : Fin 64, |prodV va vb K (64 * t + k)| ≤ ∑ _k : Fin 64, (2 : ℚ) ^ 32 :=
          Finset.sum_le_sum fun k _ => prodV_le hfa hfb _
      _ = 2 ^ 38 := by simp; norm_num
  have hcA0 : 0 ≤ ∑ k : Fin 64, |prodV va vb K (64 * t + k)| :=
    Finset.sum_nonneg fun _ _ => abs_nonneg _
  -- the step, by `DotE5m2Chunk`: partial values `s 0 = acc`, `s 1`, `s 2 = result`
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := hdot .tf32 A B acc r hr i j (fun k => (A.get2 i k).toRat)
    (fun k => (B.get2 k j).toRat) (acc.get2 i j).toRat (fun k => val_f8e5 (hfA k) rfl)
    (fun k => val_f8e5 (hfB k) rfl) (DotLoop.val?_f32.mpr ⟨hinv.1, rfl⟩) (by
      simp only [DType.bound, Math.Float.Format.binary32_maxVal]
      rw [hchunkA, relDot8_64, absDot8_64]
      have h0 := abs_nonneg (acc.get2 i j).toRat
      nlinarith)
  have h2 : Rules.Num.chunks8 64 = 2 := by unfold Rules.Num.chunks8; norm_num
  rw [h2] at hn hstep
  obtain ⟨hrfin, hrval⟩ := DotLoop.val?_f32.mp hvr
  refine ⟨hrfin, ?_⟩
  rw [hrval, ← hn]
  have hρ0 : (0 : ℚ) ≤ Rules.Num.rhoChunk8 := Dot8Chunk.rhoChunk8_nonneg
  have hσ0 : (0 : ℚ) ≤ Rules.Num.sigmaChunk8 := Dot8Chunk.sigmaChunk8_nonneg
  -- chunk `0` of the step: chunk `2·t` of the loop
  have hs0 : |s 0 - partS va vb K (2 * t)| ≤ errAt (prodV va vb K) (liftV va vb K) (2 * t) := by
    rw [h0]; exact hinv.2
  have hc0 := hcs 0 (by norm_num)
  simp only [add_zero] at hc0
  have hS1 : partS va vb K (2 * t + 1) = partS va vb K (2 * t) +
      Rules.Num.chunkSum8 (fun k => (A.get2 i k).toRat) (fun k => (B.get2 k j).toRat) 0 := by
    rw [hc0]; exact partS_succ va vb K (2 * t)
  have e1 : |s 1 - partS va vb K (2 * t + 1)| ≤ errAt (prodV va vb K) (liftV va vb K) (2 * t + 1) := by
    rw [errAt_succ]
    exact chunk_step hρ0 hσ0 hs0 hS1
      (fun M hM hm => hstep 0 (by norm_num) M hM fun k hk => (hcm 0 (by norm_num) k hk).trans hm)
  -- chunk `1` of the step: chunk `2·t + 1` of the loop
  have hc1 := hcs 1 (by norm_num)
  have hS2 : partS va vb K (2 * t + 1 + 1) = partS va vb K (2 * t + 1) +
      Rules.Num.chunkSum8 (fun k => (A.get2 i k).toRat) (fun k => (B.get2 k j).toRat) 1 := by
    rw [hc1]; exact partS_succ va vb K (2 * t + 1)
  have e2 : |s 2 - partS va vb K (2 * t + 1 + 1)| ≤
      errAt (prodV va vb K) (liftV va vb K) (2 * t + 1 + 1) := by
    rw [errAt_succ]
    exact chunk_step hρ0 hσ0 e1 hS2
      (fun M hM hm => hstep 1 (by norm_num) M hM fun k hk => (hcm 1 (by norm_num) k hk).trans hm)
  rw [show 2 * (t + 1) = 2 * t + 1 + 1 by ring]
  exact e2

/-- After the last step (`64·T ≥ K`) the partial sum is the whole sum. -/
theorem partS_end (va vb : Nat → F8E5) {K T : Nat} (hT : K ≤ 64 * T) :
    partS va vb K (2 * T) = dotSum va vb K := by
  unfold partS partSum dotSum
  obtain ⟨x, hx⟩ : ∃ x, 32 * (2 * T) = K + x := ⟨32 * (2 * T) - K, by omega⟩
  rw [hx, Finset.sum_range_add]
  have h1 : ∀ e ∈ Finset.range K, prodV va vb K e = (va e).toRat * (vb e).toRat := fun e he => by
    unfold prodV; rw [ite_eq_left (Finset.mem_range.mp he)]
  have h2 : ∀ y, prodV va vb K (K + y) = 0 := fun y => by
    unfold prodV; rw [ite_eq_right (by omega)]
  rw [Finset.sum_congr rfl h1]
  simp [h2]

theorem sum_liftV (va vb : Nat → F8E5) (K : Nat) :
    ∑ e ∈ Finset.range K, liftV va vb K e = dotLift va vb K := by
  unfold dotLift
  exact Finset.sum_congr rfl fun e he => by unfold liftV; rw [ite_eq_left (Finset.mem_range.mp he)]

theorem dotLift_nonneg (va vb : Nat → F8E5) (K : Nat) : 0 ≤ dotLift va vb K := by
  rw [← sum_liftV]
  exact Finset.sum_nonneg fun e _ => liftV_nonneg va vb K e

/-- The lifted products before chunk `j` sum to at most `dotLift`. -/
theorem liftSum_le (va vb : Nat → F8E5) (K j : Nat) :
    ∑ e ∈ Finset.range (32 * j), liftV va vb K e ≤ dotLift va vb K := by
  calc ∑ e ∈ Finset.range (32 * j), liftV va vb K e
        ≤ ∑ e ∈ Finset.range (K + 32 * j), liftV va vb K e :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.mpr (by omega))
            fun e _ _ => liftV_nonneg va vb K e
    _ = dotLift va vb K := by
        rw [Finset.sum_range_add, sum_liftV]
        have h2 : ∀ y, liftV va vb K (K + y) = 0 := fun y => by
          unfold liftV; rw [ite_eq_right (by omega)]
        simp only [h2, Finset.sum_const_zero, add_zero]

theorem liftV_le_dotLift (va vb : Nat → F8E5) (K e : Nat) : liftV va vb K e ≤ dotLift va vb K := by
  by_cases h : e < K
  · rw [← sum_liftV]
    exact Finset.single_le_sum (f := liftV va vb K) (fun e _ => liftV_nonneg va vb K e)
      (Finset.mem_range.mpr h)
  · have : liftV va vb K e = 0 := by unfold liftV; rw [ite_eq_right h]
    rw [this]
    exact dotLift_nonneg va vb K

/-- **The tolerance is never above the bound in terms of `dotLift` alone**:
`tol va vb K ≤ tolA K (dotLift va vb K)` (`errAt_le_bound`: each partial sum is at most
`Σ|a·b| ≤ dotLift`, and each lifted product at most `dotLift`). -/
theorem tol_le_tolA (va vb : Nat → F8E5) (K : Nat) : tol va vb K ≤ tolA K (dotLift va vb K) := by
  have hS : ∀ j ≤ 2 * steps K, |partSum (prodV va vb K) j| ≤ dotLift va vb K := fun j _ =>
    (Finset.abs_sum_le_sum_abs _ _).trans
      ((Finset.sum_le_sum fun e _ => abs_prodV_le_liftV va vb K e).trans (liftSum_le va vb K j))
  have hq : ∀ j < 2 * steps K, chunkMax (liftV va vb K) j ≤ dotLift va vb K := fun j _ =>
    chunkMax_le fun k => liftV_le_dotLift va vb K _
  exact errAt_le_bound _ _ hS hq _ le_rfl

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
