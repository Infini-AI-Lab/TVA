import Mathlib.Analysis.Complex.ExponentialBounds
import TVA.Device.Ref.Chunk

/-!
# The device facts hold of the reference platform

One theorem per fact of `TVA.Device.Sm90`, `TVA.Device.Dot8Chunk` and `TVA.Device.Dot16Chunk`
that `Device P` assumes, each proved of `Platform.ref`. Every fact's constants are at least one
rounding's (`u = 2^-24`, `η = 2^-150` in `float32`), and the reference platform is that accurate
(`TVA.Device.Ref.Ops`); the special values are its table `Platform.refSpecial`. The one fact that
needs more is `expF32Overflow`: from `1453635/16384` up, `e^x` is beyond every finite `float32`
by more than one rounding (`Real.log_two_lt_d9`), so the table's `+∞` applies.

`TVA.Device` assembles them into `Device.ref_ok`.
-/
namespace TVA.Platform.ref

open Device

private theorem u32 : (Format.binary32.u : ℝ) = 2 ^ (-24 : ℤ) := by
  rw [Format.binary32_u]; push_cast; rfl

private theorem eta32 : (Format.binary32.eta : ℝ) = 2 ^ (-150 : ℤ) := by
  rw [Format.binary32_eta]; push_cast; rfl

/-- A `float32` `MathAcc` fact with `rel ≥ 2^-24` and `abs ≥ 2^-150` follows from one rounding's
bound. -/
private theorem of_const {fn : MathFn} {g : ℝ → ℝ} (hg : fn.ref = some g) (dom : Set ℝ)
    {rel abs : ℝ} (hrel : (2 : ℝ) ^ (-24 : ℤ) ≤ rel) (habs : (2 : ℝ) ^ (-150 : ℤ) ≤ abs) :
    MathAcc Platform.ref fn .binary32 dom rel abs := by
  refine (mathAcc hg Format.binary32_regular Set.univ).mono (Set.subset_univ _)
    fun g' hg' x _ _ => ?_
  rw [hg] at hg'
  cases hg'
  rw [u32, eta32]
  have := mul_le_mul_of_nonneg_right hrel (abs_nonneg (g x))
  linarith

private theorem zle {a b : ℤ} (h : a ≤ b) : (2 : ℝ) ^ a ≤ 2 ^ b :=
  zpow_le_zpow_right₀ (by norm_num) h

/-- `ExpF32` holds of the reference platform. -/
theorem expF32 : ExpF32 Platform.ref := by
  intro X hX
  refine of_const (g := Real.exp) rfl _ ?_ (zle (by norm_num))
  have : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) * X := mul_nonneg (by positivity) hX
  have h : (2 : ℝ) ^ (-24 : ℤ) ≤ 2 ^ (-22 : ℤ) := zle (by norm_num)
  linarith

/-- `Exp2F32` holds of the reference platform. -/
theorem exp2F32 : Exp2F32 Platform.ref :=
  of_const (g := fun x => (2 : ℝ) ^ x) rfl _ (zle (by norm_num)) (zle (by norm_num))

/-- `LogF32` holds of the reference platform. -/
theorem logF32 : LogF32 Platform.ref :=
  of_const (g := Real.log) rfl _ (zle (by norm_num)) (zle (by norm_num))

/-- `Log2F32` holds of the reference platform. -/
theorem log2F32 : Log2F32 Platform.ref :=
  of_const (g := Real.logb 2) rfl _ (zle (by norm_num)) (zle (by norm_num))

/-- `SqrtF32` holds of the reference platform. -/
theorem sqrtF32 : SqrtF32 Platform.ref :=
  of_const (g := Real.sqrt) rfl _ (zle (by norm_num)) (zle (by norm_num))

/-- `RsqrtF32` holds of the reference platform. -/
theorem rsqrtF32 : RsqrtF32 Platform.ref :=
  of_const (g := fun x => 1 / Real.sqrt x) rfl _ (zle (by norm_num)) (zle (by norm_num))

private theorem negInf_not_finite : (FloatV.negInf : F32).isFinite = false := by decide

/-- `ExpF32NegInf` holds of the reference platform (its table). -/
theorem expF32NegInf : ExpF32NegInf Platform.ref :=
  mathAt negInf_not_finite (by decide)

/-- `Exp2F32NegInf` holds of the reference platform (its table). -/
theorem exp2F32NegInf : Exp2F32NegInf Platform.ref :=
  mathAt negInf_not_finite (by decide)

/-- From `1453635/16384` up, `e^x ≥ 2^128·(1 + 2^-23)`: `128·log 2 + 2^-23 < 1453635/16384`. -/
private theorem exp_ge_of_ge {x : ℝ} (hx : ((1453635 / 16384 : ℚ) : ℝ) ≤ x) :
    (2 : ℝ) ^ 128 * (1 + 2 ^ (-23 : ℤ)) ≤ Real.exp x := by
  have hlog := Real.log_two_lt_d9
  have h1 : 128 * Real.log 2 + 2 ^ (-23 : ℤ) ≤ x := by
    have : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
    rw [this]; push_cast at hx; norm_num at hlog; linarith
  have h2 : Real.exp (128 * Real.log 2) = 2 ^ 128 := by
    have : (128 : ℝ) * Real.log 2 = ((128 : ℕ) : ℝ) * Real.log 2 := by norm_num
    rw [this, Real.exp_nat_mul, Real.exp_log (by norm_num)]
  have h3 : 1 + (2 : ℝ) ^ (-23 : ℤ) ≤ Real.exp (2 ^ (-23 : ℤ)) := by
    have := Real.add_one_le_exp ((2 : ℝ) ^ (-23 : ℤ)); linarith
  calc (2 : ℝ) ^ 128 * (1 + 2 ^ (-23 : ℤ))
      ≤ Real.exp (128 * Real.log 2) * Real.exp (2 ^ (-23 : ℤ)) := by
        rw [h2]; exact mul_le_mul_of_nonneg_left h3 (by positivity)
    _ = Real.exp (128 * Real.log 2 + 2 ^ (-23 : ℤ)) := (Real.exp_add _ _).symm
    _ ≤ Real.exp x := Real.exp_le_exp.mpr h1

/-- No finite `float32` is within one rounding of `e^x` from `1453635/16384` up. -/
private theorem farExp {x : F32} (hx : (1453635 / 16384 : ℚ) ≤ x.toRat) :
    ¬ ∃ ref, MathFn.exp.ref = some ref ∧ ∃ r', Platform.RefNear ref x r' := by
  rintro ⟨g, hg, r', hr', hclose⟩
  have hge : g = Real.exp := by
    have h : MathFn.exp.ref = some Real.exp := rfl
    rw [h] at hg; exact (Option.some.inj hg).symm
  subst hge
  have he := exp_ge_of_ge (x := x.toReal) (by unfold FloatV.toReal; exact_mod_cast hx)
  have hmax : |r'.toReal| ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
    have h := FloatV.abs_toRat_le_maxVal Format.binary32_regular hr'
    rw [Format.binary32_maxVal] at h
    unfold FloatV.toReal
    exact_mod_cast h
  rw [u32, eta32, abs_of_pos (Real.exp_pos _)] at hclose
  have h1 : Real.exp x.toReal - r'.toReal ≤ |r'.toReal - Real.exp x.toReal| := by
    rw [abs_sub_comm]; exact le_abs_self _
  have h2 := le_abs_self r'.toReal
  have hz1 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
  have hz2 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
  have hz3 : (0 : ℝ) ≤ 2 ^ (-150 : ℤ) ∧ (2 : ℝ) ^ (-150 : ℤ) ≤ 1 :=
    ⟨by positivity, zpow_le_one_of_nonpos₀ (by norm_num) (by norm_num)⟩
  rw [hz1] at hclose; rw [hz2] at he
  norm_num at he hmax
  linarith [h1, h2, hz3.1, hz3.2]

/-- `ExpF32Overflow` holds of the reference platform: its table gives `+∞` there, and no float is
near enough to be returned instead. -/
theorem expF32Overflow : ExpF32Overflow Platform.ref := by
  intro x hx r hr
  rcases hx with ⟨hfin, hT⟩ | rfl
  · have hne : ¬ ((MathFn.exp = .exp ∨ MathFn.exp = .exp2) ∧
        x.pat = Format.binary32.setSign true Format.binary32.infMag) := by
      rintro ⟨_, hp⟩
      have : x.isFinite = false := by unfold FloatV.isFinite; rw [hp]; decide
      rw [this] at hfin; exact Bool.false_ne_true hfin
    have hsp : Platform.refSpecial .exp .binary32 x = some FloatV.posInf := by
      unfold Platform.refSpecial
      rw [ite_eq_right_of_eq_false _ _ (eq_false hne),
        ite_eq_left_of_eq_true _ _ (eq_true ⟨rfl, rfl, Or.inl ⟨hfin, hT⟩⟩)]; rfl
    exact mathAt_far hfin (farExp hT) hsp r hr
  · exact mathAt (by decide) (by decide) r hr

/-- `DivFull` holds of the reference platform (IEEE division meets it). -/
theorem divFull : DivFull Platform.ref := DivFull.of_divRn (divRn _)

/-- `DivF64` holds of the reference platform. -/
theorem divF64 : DivF64 Platform.ref := divRn _

/-- `DivF32InfDivisor` holds of the reference platform: IEEE division of a finite value by an
infinite one is the signed zero. -/
theorem divF32InfDivisor : DivF32InfDivisor Platform.ref := by
  intro x y hx hy r hr
  have hxn : x.isNaN = false := (FloatV.isFinite_iff.mp hx).1
  have hyn : y.isNaN = false := by
    unfold FloatV.isInf at hy; unfold FloatV.isNaN
    have hs : Format.binary32.specials = .ieee := rfl
    simp only [Format.isInf, Format.isNaN, hs, Bool.and_eq_true, beq_iff_eq] at hy ⊢
    simp [hy.2]
  have hxi : x.isInf = false := (FloatV.isFinite_iff.mp hx).2
  have hd : FloatV.divRn x y = FloatV.ofPat (Format.binary32.zeroPat (x.sign != y.sign)) := by
    unfold FloatV.divRn FloatV.sign
    unfold FloatV.isNaN at hxn hyn; unfold FloatV.isInf at hy hxi
    simp only [Format.div, hxn, hyn, hxi, hy, Bool.or_self, Bool.false_eq_true, ite_false,
      ite_true]
  change FloatV.NaNEq (FloatV.divRn x y) r at hr
  rw [← hd]
  refine FloatV.NaNEq.eq hr ?_
  rw [hd]; unfold FloatV.isNaN FloatV.ofPat FloatV.pat
  cases x.sign != y.sign <;> decide

/-- Every `CvtSat` holds of the reference platform. -/
theorem cvtSatF32E4m3 : CvtSatF32E4m3 Platform.ref := cvtSat _ _
/-- `CvtSatF32E5m2` holds of the reference platform. -/
theorem cvtSatF32E5m2 : CvtSatF32E5m2 Platform.ref := cvtSat _ _
/-- `CvtSatF16E4m3` holds of the reference platform. -/
theorem cvtSatF16E4m3 : CvtSatF16E4m3 Platform.ref := cvtSat _ _
/-- `CvtSatF16E5m2` holds of the reference platform. -/
theorem cvtSatF16E5m2 : CvtSatF16E5m2 Platform.ref := cvtSat _ _
/-- `CvtSatBF16E4m3` holds of the reference platform. -/
theorem cvtSatBF16E4m3 : CvtSatBF16E4m3 Platform.ref := cvtSat _ _
/-- `CvtSatBF16E5m2` holds of the reference platform. -/
theorem cvtSatBF16E5m2 : CvtSatBF16E5m2 Platform.ref := cvtSat _ _

private theorem u32q : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u

private theorem eta32q : Format.binary32.eta = 2 ^ (-150 : ℤ) := Format.binary32_eta

private theorem qle {a b : ℤ} (h : a ≤ b) : (2 : ℚ) ^ a ≤ 2 ^ b :=
  zpow_le_zpow_right₀ (by norm_num) h

/-- `(K + c)·2^b` is at least `2^a` for `a ≤ b`, `c ≥ 1`. -/
private theorem le_mul_two_zpow (K : ℕ) {c : ℚ} (hc : 1 ≤ c) {a b : ℤ} (h : a ≤ b) :
    (2 : ℚ) ^ a ≤ (K + c) * 2 ^ b := by
  have : (1 : ℚ) ≤ K + c := by have := Nat.cast_nonneg (α := ℚ) K; linarith
  nlinarith [qle h, zpow_pos (show (0 : ℚ) < 2 by norm_num) b]

/-- `DotAcc` with the 16-bit constants holds of the reference platform for a `float32`
accumulator. -/
private theorem dot16 (cfg : DotCfg) (hacc : cfg.acc = .f32) :
    DotAcc Platform.ref cfg (fun _ => True) relDot16 absDot16 := by
  refine dotAcc (g := Format.binary32) (by rw [hacc]; rfl) _ (fun K _ => ?_) (fun K _ => ?_)
    (by unfold relDot16; positivity) (by unfold absDot16; positivity)
  · rw [u32q]; exact le_mul_two_zpow K (by norm_num) (by norm_num)
  · rw [eta32q]; exact le_mul_two_zpow K (by norm_num) (by norm_num)

private theorem relDot16_nonneg (K : ℕ) : 0 ≤ relDot16 K := by unfold relDot16; positivity

/-- `DotF16` holds of the reference platform. -/
theorem dotF16 : DotF16 Platform.ref := fun _ => (dot16 _ rfl).liftAcc relDot16_nonneg

/-- `DotBF16` holds of the reference platform. -/
theorem dotBF16 : DotBF16 Platform.ref := fun _ => (dot16 _ rfl).liftAcc relDot16_nonneg

/-- `DotF32Ieee` holds of the reference platform. -/
theorem dotF32Ieee : DotF32Ieee Platform.ref := by
  refine dotAcc (g := Format.binary32) rfl _ (fun K _ => ?_) (fun K _ => ?_)
    (by positivity) (by positivity)
  · rw [u32q]; exact le_mul_two_zpow K le_rfl (by norm_num)
  · rw [eta32q]; exact le_mul_two_zpow K le_rfl (by norm_num)

/-- `DotI8` holds of the reference platform. -/
theorem dotI8 : DotI8 Platform.ref := fun p =>
  dotAcc_i8 p 0 _ (fun _ => le_refl 0) (fun _ => le_refl 0)

/-- `DotScaledMx` holds of the reference platform for every pair of formats. -/
private theorem dotScaledMx (lhs rhs : MXFmt) : DotScaledMx Platform.ref lhs rhs := by
  intro fm
  refine dotScaledAcc _ _ _ (fun K _ => ?_) (fun K _ => ?_)
    (by unfold relDot16; positivity) (by unfold absDot16; positivity)
  · rw [u32q]; exact le_mul_two_zpow K (by norm_num) (by norm_num)
  · rw [eta32q]; exact le_mul_two_zpow K (by norm_num) (by norm_num)

/-- `DotScaledE4m3` holds of the reference platform. -/
theorem dotScaledE4m3 : DotScaledE4m3 Platform.ref := dotScaledMx _ _

/-- `DotScaledE4m3E2m1` holds of the reference platform. -/
theorem dotScaledE4m3E2m1 : DotScaledE4m3E2m1 Platform.ref := dotScaledMx _ _

/-- `AtomicAddF32` holds of the reference platform. -/
theorem atomicAddF32 : AtomicAddF32 Platform.ref :=
  atomicAddAcc Format.binary32_regular (by rw [u32q]; exact qle (by norm_num))
    (by rw [eta32q]; exact qle (by norm_num))

/-- `SumDepth` holds of the reference platform: its trees are balanced. -/
theorem sumDepth : SumDepth Platform.ref := fun _ => le_rfl

/-- The fp8 chunk facts hold of the reference platform. -/
private theorem dotF8ChunkLoss (a b : DType) : DotF8ChunkLoss Platform.ref a b := fun _ =>
  dotChunkLoss (g := Format.binary32) rfl (by unfold rhoChunk8; positivity)
    (by unfold kappaChunk8; norm_num) (by unfold rhoLossChunk8; positivity)
    (by rw [u32q]; exact qle (by norm_num)) (by rw [eta32q]; exact qle (by norm_num))

/-- `DotE4m3ChunkLoss` holds of the reference platform. -/
theorem dotE4m3ChunkLoss : DotE4m3ChunkLoss Platform.ref := dotF8ChunkLoss _ _
/-- `DotE5m2ChunkLoss` holds of the reference platform. -/
theorem dotE5m2ChunkLoss : DotE5m2ChunkLoss Platform.ref := dotF8ChunkLoss _ _
/-- `DotE4m3E5m2ChunkLoss` holds of the reference platform. -/
theorem dotE4m3E5m2ChunkLoss : DotE4m3E5m2ChunkLoss Platform.ref := dotF8ChunkLoss _ _
/-- `DotE5m2E4m3ChunkLoss` holds of the reference platform. -/
theorem dotE5m2E4m3ChunkLoss : DotE5m2E4m3ChunkLoss Platform.ref := dotF8ChunkLoss _ _

/-- The 16-bit chunk facts hold of the reference platform. -/
private theorem dot16BitChunk (a b : DType) : Dot16BitChunk Platform.ref a b := fun _ =>
  dotChunk16Loss (g := Format.binary32) rfl (by unfold rhoChunk16; positivity)
    (by unfold kappaChunk16; norm_num) (by unfold rhoLossChunk16; positivity)
    (by rw [u32q]; exact qle (by norm_num)) (by rw [eta32q]; exact qle (by norm_num))

/-- `DotF16Chunk` holds of the reference platform. -/
theorem dotF16Chunk : DotF16Chunk Platform.ref := dot16BitChunk _ _
/-- `DotBF16Chunk` holds of the reference platform. -/
theorem dotBF16Chunk : DotBF16Chunk Platform.ref := dot16BitChunk _ _

end TVA.Platform.ref
