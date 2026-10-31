import Matmul.Defs
import Matmul.Arith

/-!
# `matmul_f16`: facts the per-program proof uses

* `int32` arithmetic on small values (`i32_add`, `i32_sub`, `i32_mul`, `i32_lt`, `i32_le`);
* the zero `other=0.0` converted to `float16` (`zero_conv`);
* the accumulation of one `tl.dot` step (`acc_step`), from `Rules.Num.DotF16`;
* the last rounding to `float16` (`out_f16`).
-/

namespace Kernels.Matmul

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

/-- The source's `other=0.0` converted to `float16`: some result exists, and every result is a
finite zero. -/
theorem zero_conv_ex (P : Platform) :
    ∃ r, ConvRel P .f32 .f16 (FloatV.litF32 (FloatV.ofPat 0) : F32) r :=
  ⟨FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0) : F32), by
    show FloatRel P Math.Float.Format.binary32 Math.Float.Format.binary16 _ _
    unfold FloatRel
    rw [ite_eq_right (by decide)]
    exact Or.inl ⟨by unfold Math.Float.Format.Std; simp, by unfold Math.Float.Format.Std; simp,
      FloatV.NaNEq.refl _⟩⟩

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

/-! ## The accumulation -/

/-- Product `e` of the reduction, `0` from `K` on (the lanes the masks disable). -/
def prodV (va vb : Nat → F16) (K e : Nat) : ℚ := if e < K then (va e).toRat * (vb e).toRat else 0

/-- The exact partial sum after `t` steps: the products `e < 64·t`. -/
def partS (va vb : Nat → F16) (K t : Nat) : ℚ := ∑ e ∈ Finset.range (64 * t), prodV va vb K e

/-- The partial sum of the magnitudes. -/
def partA (va vb : Nat → F16) (K t : Nat) : ℚ := ∑ e ∈ Finset.range (64 * t), |prodV va vb K e|

theorem relDot16_64 : Rules.Num.relDot16 64 = 66 / 2 ^ 22 := by
  unfold Rules.Num.relDot16; norm_num

theorem absDot16_64 : Rules.Num.absDot16 64 = 66 / 2 ^ 126 := by
  unfold Rules.Num.absDot16; norm_num

theorem f16_abs_le {x : F16} (hx : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have := FloatV.abs_toRat_le_maxVal Math.Float.Format.binary16_regular hx
  rwa [Math.Float.Format.binary16_maxVal] at this

theorem prodV_le {va vb : Nat → F16} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (e : Nat) : |prodV va vb K e| ≤ 2 ^ 32 := by
  unfold prodV
  split_ifs with h
  · rw [abs_mul]
    have h1 := f16_abs_le (hfa e h)
    have h2 := f16_abs_le (hfb e h)
    have h3 : |(va e).toRat| * |(vb e).toRat| ≤ 65504 * 65504 :=
      mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
    linarith
  · simp

theorem partA_le {va vb : Nat → F16} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (t : Nat) : partA va vb K t ≤ 64 * t * 2 ^ 32 := by
  unfold partA
  calc ∑ e ∈ Finset.range (64 * t), |prodV va vb K e| ≤ ∑ _e ∈ Finset.range (64 * t), (2 : ℚ) ^ 32 :=
        Finset.sum_le_sum fun e _ => prodV_le hfa hfb e
    _ = 64 * t * 2 ^ 32 := by simp

theorem partS_succ (va vb : Nat → F16) (K t : Nat) :
    partS va vb K (t + 1) = partS va vb K t + ∑ k : Fin 64, prodV va vb K (64 * t + k) := by
  unfold partS
  rw [show 64 * (t + 1) = 64 * t + 64 by ring, Finset.sum_range_add,
    Fin.sum_univ_eq_sum_range (fun k => prodV va vb K (64 * t + k)) 64]

theorem partA_succ (va vb : Nat → F16) (K t : Nat) :
    partA va vb K (t + 1) = partA va vb K t + ∑ k : Fin 64, |prodV va vb K (64 * t + k)| := by
  unfold partA
  rw [show 64 * (t + 1) = 64 * t + 64 by ring, Finset.sum_range_add,
    Fin.sum_univ_eq_sum_range (fun k => |prodV va vb K (64 * t + k)|) 64]

theorem val_f16 {x : F16} {v : ℚ} (hx : x.isFinite = true) (hv : x.toRat = v) :
    DType.val? .f16 x = some v := by
  show (if x.isFinite = true then some x.toRat else none) = some v
  rw [ite_eq_left hx, hv]

/-- **One step of the loop** (line 300), for one output element `(i, j)` of the tile: the
accumulator stays finite and within `DotLoop.bound` of the exact partial sum, by
`Rules.Num.DotF16` (`DotLoop.accum_step`). `A` and `B` are the loaded tiles: their enabled lanes
hold the matrix elements, their disabled lanes finite zeros. -/
theorem acc_step {P : Platform} (hdot : Rules.Num.DotF16 P)
    {A : Tensor [128, 64] F16} {B : Tensor [64, 256] F16} {acc r : Tensor [128, 256] F32}
    (hr : P.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 } A B acc r)
    (i : Fin 128) (j : Fin 256) {va vb : Nat → F16} {K t : Nat} (hK : K ≤ 2 ^ 16) (ht : 64 * t < K)
    (hfa : ∀ e, e < K → (va e).isFinite = true) (hfb : ∀ e, e < K → (vb e).isFinite = true)
    (hA : ∀ k : Fin 64, (64 * t + k < K → A.get2 i k = va (64 * t + k)) ∧
      (K ≤ 64 * t + k → (A.get2 i k).isFinite = true ∧ (A.get2 i k).toRat = 0))
    (hB : ∀ k : Fin 64, (64 * t + k < K → B.get2 k j = vb (64 * t + k)) ∧
      (K ≤ 64 * t + k → (B.get2 k j).isFinite = true ∧ (B.get2 k j).toRat = 0))
    (hinv : (acc.get2 i j).isFinite = true ∧
      |(acc.get2 i j).toRat - partS va vb K t| ≤
        DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) t (partA va vb K t)) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toRat - partS va vb K (t + 1)| ≤
        DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) (t + 1) (partA va vb K (t + 1)) := by
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
  have ht1 : (t : ℚ) ≤ 1024 := by
    have : t ≤ 1024 := by omega
    exact_mod_cast this
  have hB0 : partA va vb K t ≤ 2 ^ 48 := by
    have h1 := partA_le hfa hfb t
    have h2 : (64 * t : ℚ) ≤ 2 ^ 16 := by
      have : 64 * t ≤ 2 ^ 16 := by omega
      exact_mod_cast this
    nlinarith
  have hBn : 0 ≤ partA va vb K t := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hcA : ∑ k : Fin 64, |prodV va vb K (64 * t + k)| ≤ 2 ^ 38 := by
    calc ∑ k : Fin 64, |prodV va vb K (64 * t + k)| ≤ ∑ _k : Fin 64, (2 : ℚ) ^ 32 :=
          Finset.sum_le_sum fun k _ => prodV_le hfa hfb _
      _ = 2 ^ 38 := by simp; norm_num
  have hcA0 : 0 ≤ ∑ k : Fin 64, |prodV va vb K (64 * t + k)| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hρ0 : (0 : ℚ) ≤ Rules.Num.relDot16 64 := by rw [relDot16_64]; norm_num
  have hα0 : (0 : ℚ) ≤ Rules.Num.absDot16 64 := by rw [absDot16_64]; norm_num
  have hbound : DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) t (partA va vb K t) ≤ 2 ^ 44 := by
    have htρ : (t : ℚ) * Rules.Num.relDot16 64 ≤ 1 / 2 := by
      rw [relDot16_64]; nlinarith
    refine (DotLoop.bound_le hρ0 hα0 htρ hBn).trans ?_
    rw [relDot16_64, absDot16_64]
    have ht0 : (0 : ℚ) ≤ t := Nat.cast_nonneg t
    nlinarith
  have hSt : |partS va vb K t| ≤ partA va vb K t := Finset.abs_sum_le_sum_abs _ _
  have hvc : |(acc.get2 i j).toRat| ≤ 2 ^ 49 := by
    have := abs_sub_abs_le_abs_sub (acc.get2 i j).toRat (partS va vb K t)
    linarith [hinv.2]
  -- the step, by `DotF16`
  obtain ⟨vr, hvr, herr⟩ := hdot .tf32 A B acc r hr i j (fun k => (A.get2 i k).toRat)
    (fun k => (B.get2 k j).toRat) (acc.get2 i j).toRat (fun k => val_f16 (hfA k) rfl)
    (fun k => val_f16 (hfB k) rfl) (fun _ => ⟨trivial, trivial⟩)
    (DotLoop.val?_f32.mpr ⟨hinv.1, rfl⟩) (by
      simp only [DType.bound, Math.Float.Format.binary32_maxVal]
      rw [hchunkA, relDot16_64, absDot16_64]
      have h0 := abs_nonneg (acc.get2 i j).toRat
      nlinarith)
  obtain ⟨hrfin, hrval⟩ := DotLoop.val?_f32.mp hvr
  refine ⟨hrfin, ?_⟩
  rw [partS_succ, partA_succ, hrval]
  rw [hchunk, hchunkA] at herr
  exact DotLoop.accum_step hρ0 hα0 hSt (Finset.abs_sum_le_sum_abs _ _) hinv.2 herr

/-- After the last step (`64·T ≥ K`) the partial sums are the whole sums. -/
theorem partS_end (va vb : Nat → F16) {K T : Nat} (hT : K ≤ 64 * T) :
    partS va vb K T = dotSum va vb K ∧ partA va vb K T = dotAbs va vb K := by
  unfold partS partA dotSum dotAbs
  obtain ⟨x, hx⟩ : ∃ x, 64 * T = K + x := ⟨64 * T - K, by omega⟩
  rw [hx, Finset.sum_range_add, Finset.sum_range_add]
  have h1 : ∀ e ∈ Finset.range K, prodV va vb K e = (va e).toRat * (vb e).toRat := fun e he => by
    unfold prodV; rw [ite_eq_left (Finset.mem_range.mp he)]
  have h2 : ∀ y, prodV va vb K (K + y) = 0 := fun y => by
    unfold prodV; rw [ite_eq_right (by omega)]
  constructor
  · rw [Finset.sum_congr rfl h1]; simp [h2]
  · rw [Finset.sum_congr rfl fun e he => by rw [h1 e he]]; simp [h2]

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

end Kernels.Matmul
