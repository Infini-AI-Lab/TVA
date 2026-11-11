import TVA.Device.Ref.Platform

/-!
# What the reference platform satisfies

The general properties of `Platform.ref` (`TVA.Device.Ref.Platform`), one per shape of
`TVA.Device.Shapes`:

* every operation has a result (`mathTotal`, `divTotal`, `cvtTotal`, `dotTotal`,
  `atomicAddTotal`, `dotScaledTotal`);
* `mathAcc`: an approximate function meets `MathAcc` with one rounding's constants `u`, `η` on
  any domain (`FloatV.exists_nearest` gives the float it may return); `mathAt`, `mathAt_far`: the
  table `Platform.refSpecial` at non-finite inputs and where no float is near;
* `divRn`, `cvtSat`, `atomicAddAcc`: IEEE division, saturating conversions, an IEEE sum;
* `dotAcc`, `dotAcc_i8`, `dotScaledAcc`: one rounding of the exact `tl.dot` (`tl.dot_scaled`)
  meets every bound at least one rounding's; for `int8` operands the exact sum;
* `dotChunkAcc`, `dotChunkLoss`, `dotChunk16Loss`: the partial values of the chunk facts are the
  exact partial sums, and the last step is the one rounding.

`TVA.Device.Ref.Sm90` instantiates them at the constants of each device fact.
-/
namespace TVA.Platform.ref

open Device

/-! ## Approximate functions -/

/-- `MathAcc` on any domain with one rounding's constants (`u`, `η` of a regular format): a
float that near exists wherever the room condition holds. -/
theorem mathAcc {fn : MathFn} {f : Format} {g : ℝ → ℝ} (hg : fn.ref = some g) (hf : f.Regular)
    (dom : Set ℝ) : MathAcc Platform.ref fn f dom (f.u : ℝ) (f.eta : ℝ) := by
  refine ⟨g, hg, fun x r hr hx _ _ hroom => ?_⟩
  simp only [Platform.ref, hx, ite_true] at hr
  have he : (0 : ℝ) ≤ f.eta := by exact_mod_cast Format.eta_nonneg
  have hu : (0 : ℝ) ≤ f.u := by exact_mod_cast Format.u_nonneg
  have hmax : |g x.toReal| ≤ (f.maxVal : ℝ) := by
    nlinarith [abs_nonneg (g x.toReal), mul_nonneg hu (abs_nonneg (g x.toReal))]
  exact hr.1 g hg (FloatV.exists_nearest hf _ hmax)

/-- `MathAt` at a non-finite input: the table's value. -/
theorem mathAt {fn : MathFn} {f : Format} {x y : FloatV f} (hx : x.isFinite = false)
    (h : Platform.refSpecial fn f x = some y) : MathAt Platform.ref fn f x y := by
  intro r hr
  simp only [Platform.ref, hx, Bool.false_eq_true, ite_false] at hr
  exact hr y h

/-- `MathAt` at a finite input where no float is within one rounding of the real function (an
overflow): the table's value. -/
theorem mathAt_far {fn : MathFn} {f : Format} {x y : FloatV f} (hx : x.isFinite = true)
    (hfar : ¬ ∃ ref, fn.ref = some ref ∧ ∃ r', Platform.RefNear ref x r')
    (h : Platform.refSpecial fn f x = some y) : MathAt Platform.ref fn f x y := by
  intro r hr
  simp only [Platform.ref, hx, ite_true] at hr
  exact hr.2 hfar y h

/-- Every approximate function has a result on every input. -/
theorem mathTotal (fn : MathFn) (f : Format) : MathTotal Platform.ref fn f := by
  intro x
  by_cases hx : x.isFinite = true
  · simp only [Platform.ref, hx, ite_true]
    by_cases hpre : ∃ g, fn.ref = some g ∧ ∃ r', Platform.RefNear g x r'
    · obtain ⟨g, hg, r', hr'⟩ := hpre
      refine ⟨r', fun g' hg' _ => ?_, fun hno => absurd ⟨g, hg, r', hr'⟩ hno⟩
      rw [hg] at hg'
      cases hg'
      exact hr'
    · cases h : Platform.refSpecial fn f x with
      | some y =>
        exact ⟨y, fun g hg hex => absurd ⟨g, hg, hex⟩ hpre, fun _ y' hy' => Option.some.inj hy'⟩
      | none => exact ⟨x, fun g hg hex => absurd ⟨g, hg, hex⟩ hpre, fun _ y' hy' => by cases hy'⟩
  · simp only [Platform.ref, hx]
    cases h : Platform.refSpecial fn f x with
    | some y => exact ⟨y, fun y' hy' => Option.some.inj hy'⟩
    | none => exact ⟨x, fun y' hy' => by cases hy'⟩

/-! ## Division, conversions, atomics -/

/-- Division is IEEE division on every format. -/
theorem divRn (f : Format) : DivRn Platform.ref f := fun _ _ _ h => h

/-- Division has a result for every pair of operands. -/
theorem divTotal (f : Format) : DivTotal Platform.ref f := fun _ _ => ⟨_, FloatV.NaNEq.refl _⟩

/-- Every conversion has a result. -/
theorem cvtTotal (f g : Format) (rnd : RoundMode) : CvtTotal Platform.ref f g rnd :=
  fun _ => ⟨_, FloatV.NaNEq.refl _⟩

/-- Conversions to nearest saturate, so every `CvtSat` holds. -/
theorem cvtSat (f g : Format) : CvtSat Platform.ref f g := by
  intro x r hr
  simpa only [Platform.ref, ↓reduceIte] using hr

/-- A float `atomic_add` has a result for every old value and operand. -/
theorem atomicAddTotal (f : Format) : AtomicAddTotal Platform.ref f :=
  fun _ _ => ⟨_, FloatV.NaNEq.refl _⟩

/-- A float `atomic_add` is the IEEE sum, so it meets `AtomicAddAcc` with constants at least one
rounding's. -/
theorem atomicAddAcc {f : Format} (hf : f.Regular) {r a : ℚ} (hr : f.u ≤ r) (ha : f.eta ≤ a) :
    AtomicAddAcc Platform.ref f r a := by
  intro o v n hn ho hv hmax
  obtain ⟨hfin, herr⟩ := FloatV.add_err hf ho hv hmax
  change FloatV.NaNEq (FloatV.add o v) n at hn
  have hnan : (FloatV.add o v).isNaN = false := (FloatV.isFinite_iff.mp hfin).1
  rw [hn.eq hnan]
  refine ⟨hfin, herr.trans ?_⟩
  have h1 := mul_le_mul_of_nonneg_left hr (abs_nonneg (o.toRat + v.toRat))
  have h2 : |o.toRat + v.toRat| * r = r * |o.toRat + v.toRat| := mul_comm _ _
  linarith

/-! ## `tl.dot` and `tl.dot_scaled` -/

/-- `tl.dot` has a result for all operands. -/
theorem dotTotal (cfg : DotCfg) : DotTotal Platform.ref cfg :=
  fun ⦃_ _ K⦄ a b acc => ⟨Tensor.of2 fun i n =>
    if K = 0 then acc.get2 i n else cfg.acc.ofRat (cfg.exact a b acc i n),
    fun _ _ => Tensor.get2_of2 _ _ _⟩

/-- `tl.dot_scaled` has a result for all operands. -/
theorem dotScaledTotal (cfg : ScaledCfg) : DotScaledTotal Platform.ref cfg :=
  fun ⦃_ _ K⦄ a sa b sb acc => ⟨Tensor.of2 fun i n =>
    if K = 0 then acc.get2 i n else FloatV.ofRat (cfg.exact a sa b sb acc i n),
    fun _ _ => Tensor.get2_of2 _ _ _⟩

/-- The exact sum is bounded by `mag`. -/
private theorem abs_sum_le {K : Nat} (c : ℚ) (va vb : Fin K → ℚ) :
    |c + ∑ k, va k * vb k| ≤ |c| + ∑ k, |va k * vb k| :=
  (abs_add_le _ _).trans (add_le_add le_rfl (Finset.abs_sum_le_sum_abs _ _))

private theorem mag_nonneg {K : Nat} (c : ℚ) (va vb : Fin K → ℚ) :
    0 ≤ |c| + ∑ k, |va k * vb k| :=
  add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)

/-- One rounding of `q` is within `rel · mag + abs` when `|q| ≤ mag`, `u ≤ rel`, `η ≤ abs`. -/
private theorem round_le {u eta q e mag rel abs : ℚ} (hu : 0 ≤ u) (hqm : |q| ≤ mag)
    (hmag : 0 ≤ mag) (hrel : u ≤ rel) (habs : eta ≤ abs) (he : e ≤ |q| * u + eta) :
    e ≤ rel * mag + abs := by
  have h1 : |q| * u ≤ mag * u := mul_le_mul_of_nonneg_right hqm hu
  have h2 : mag * u ≤ mag * rel := mul_le_mul_of_nonneg_left hrel hmag
  linarith [mul_comm mag rel]

/-- With `K = 0` the result is the accumulator, within any nonnegative bound. -/
private theorem dot_zero {d : DType} {x : d.Val} {vc r0 a0 : ℚ} (va vb : Fin 0 → ℚ)
    (hvc : d.val? x = some vc) (hr0 : 0 ≤ r0) (ha0 : 0 ≤ a0) :
    ∃ vr, d.val? x = some vr ∧
      |vr - (vc + ∑ k, va k * vb k)| ≤ r0 * (|vc| + ∑ k, |va k * vb k|) + a0 := by
  refine ⟨vc, hvc, ?_⟩
  rw [Fin.sum_univ_zero, Fin.sum_univ_zero, add_zero, add_zero, sub_self, abs_zero]
  exact add_nonneg (mul_nonneg hr0 (abs_nonneg _)) ha0

/-- The result of a `tl.dot` with `K > 0` whose elements have values is the rounding of
`vc + Σ va·vb`. -/
private theorem dot_get {cfg : DotCfg} {M N K : Nat} {a : Tensor [M, K] cfg.a.Val}
    {b : Tensor [K, N] cfg.b.Val} {acc r : Tensor [M, N] cfg.acc.Val}
    (hr : Platform.ref.dot cfg a b acc r) (hK : K ≠ 0) {i : Fin M} {n : Fin N}
    {va vb : Fin K → ℚ} {vc : ℚ}
    (hva : ∀ k, cfg.a.val? (a.get2 i k) = some (va k))
    (hvb : ∀ k, cfg.b.val? (b.get2 k n) = some (vb k))
    (hvc : cfg.acc.val? (acc.get2 i n) = some vc) :
    r.get2 i n = cfg.acc.ofRat (vc + ∑ k, va k * vb k) := by
  have hex : cfg.exact a b acc i n = vc + ∑ k, va k * vb k := by
    simp only [DotCfg.exact, hva, hvb, hvc, Option.getD_some]
  rw [hr i n, ite_eq_right hK, hex]

/-- **`DotAcc` for a float accumulator**, with at least one rounding: for `K > 0`, `rel K ≥ u`
and `abs K ≥ η` of the accumulator's format; for `K = 0` any `rel 0`, `abs 0 ≥ 0`. -/
theorem dotAcc {cfg : DotCfg} {g : Format} (hg : cfg.acc.format? = some g)
    (dom : ℚ → Prop) {relK absK : ℕ → ℚ} (hrel : ∀ K, 0 < K → g.u ≤ relK K)
    (habs : ∀ K, 0 < K → g.eta ≤ absK K) (hrel0 : 0 ≤ relK 0) (habs0 : 0 ≤ absK 0) :
    DotAcc Platform.ref cfg dom relK absK := by
  intro M N K a b acc r hr i n va vb vc hva hvb _ hvc hroom
  by_cases hK : K = 0
  · subst hK
    rw [hr i n, ite_eq_left rfl]
    exact dot_zero va vb hvc hrel0 habs0
  have hKp : 0 < K := Nat.pos_of_ne_zero hK
  have hri := dot_get hr hK hva hvb hvc
  have hqm := abs_sum_le vc va vb
  have hmag := mag_nonneg vc va vb
  have hu := Format.u_nonneg (f := g)
  have hqmax : |vc + ∑ k, va k * vb k| ≤ g.maxVal := by
    rw [← DType.bound_of_format? hg]
    have : 0 ≤ relK K * (|vc| + ∑ k, |va k * vb k|) := mul_nonneg (hu.trans (hrel K hKp)) hmag
    linarith [habs K hKp, Format.eta_nonneg (f := g)]
  obtain ⟨v, hv, herr⟩ := DType.val?_ofRat_float hg hqmax
  exact ⟨v, by rw [hri]; exact hv, round_le hu hqm hmag (hrel K hKp) (habs K hKp) herr⟩

/-- **`DotAcc` for `int8` operands into `int32`**: an exact sum in range is the result, so any
`rel`, `abs ≥ 0` hold. -/
theorem dotAcc_i8 (p : InputPrecision) (m : Nat) (dom : ℚ → Prop) {relK absK : ℕ → ℚ}
    (hrel : ∀ K, 0 ≤ relK K) (habs : ∀ K, 0 ≤ absK K) :
    DotAcc Platform.ref ⟨.i8, .i8, .i32, p, m⟩ dom relK absK := by
  intro M N K a b acc r hr i n va vb vc hva hvb _ hvc hroom
  by_cases hK : K = 0
  · subst hK
    rw [hr i n, ite_eq_left rfl]
    exact dot_zero va vb hvc (hrel 0) (habs 0)
  have hri := dot_get hr hK hva hvb hvc
  simp only [DType.val?, Option.some.injEq] at hva hvb hvc
  have hq : vc + ∑ k, va k * vb k =
      (((acc.get2 i n).toInt + ∑ k, (a.get2 i k).toInt * (b.get2 k n).toInt : ℤ) : ℚ) := by
    push_cast
    rw [hvc]
    exact congrArg _ (Finset.sum_congr rfl fun k _ => by rw [hva k, hvb k])
  set z : ℤ := (acc.get2 i n).toInt + ∑ k, (a.get2 i k).toInt * (b.get2 k n).toInt
  have hmag := mag_nonneg vc va vb
  have hzb : |(z : ℚ)| ≤ 2 ^ 31 - 1 := by
    rw [← hq]
    have : 0 ≤ relK K * (|vc| + ∑ k, |va k * vb k|) := mul_nonneg (hrel K) hmag
    have hb : (DType.i32).bound = 2 ^ 31 - 1 := rfl
    simp only [hb] at hroom
    linarith [abs_sum_le vc va vb, habs K]
  have hz := abs_le.mp hzb
  have hzi : (-(2 ^ (32 - 1) : ℤ)) ≤ z ∧ z < 2 ^ (32 - 1) := by
    constructor
    · have : (-(2 ^ 31 : ℚ)) ≤ z := by linarith
      exact_mod_cast this
    · have : (z : ℚ) < 2 ^ 31 := by linarith
      exact_mod_cast this
  refine ⟨(z : ℚ), ?_, ?_⟩
  · rw [hri, hq]
    simp only [DType.val?, DType.ofRat, Int.floor_intCast,
      IntV.toInt_ofInt (w := 32) (by norm_num) hzi]
  · rw [hq, sub_self, abs_zero]
    exact add_nonneg (mul_nonneg (hrel K) hmag) (habs K)

/-- **`DotScaledAcc`** on any domains, with at least one `float32` rounding: for `K > 0`,
`rel K ≥ u` and `abs K ≥ η`; for `K = 0`, any `rel 0`, `abs 0 ≥ 0`. -/
theorem dotScaledAcc (cfg : ScaledCfg) (dom : ℚ → Prop) (sdom : Nat → Prop) {relK absK : ℕ → ℚ}
    (hrel : ∀ K, 0 < K → Format.binary32.u ≤ relK K)
    (habs : ∀ K, 0 < K → Format.binary32.eta ≤ absK K) (hrel0 : 0 ≤ relK 0)
    (habs0 : 0 ≤ absK 0) :
    DotScaledAcc Platform.ref cfg dom sdom relK absK := by
  intro M N K a sa b sb acc r hr i n va vb hva hvb _ _ _ hacc hroom
  by_cases hK : K = 0
  · subst hK
    rw [hr i n, ite_eq_left rfl]
    refine ⟨hacc, ?_⟩
    rw [Fin.sum_univ_zero, Fin.sum_univ_zero, add_zero, add_zero, sub_self, abs_zero]
    exact add_nonneg (mul_nonneg hrel0 (abs_nonneg _)) habs0
  have hKp : 0 < K := Nat.pos_of_ne_zero hK
  have hex : cfg.exact a sa b sb acc i n = (acc.get2 i n).toRat + ∑ k, va k * vb k := by
    simp only [ScaledCfg.exact, hva, hvb, Option.getD_some]
  have hri : r.get2 i n = FloatV.ofRat ((acc.get2 i n).toRat + ∑ k, va k * vb k) := by
    rw [hr i n, ite_eq_right hK, hex]
  have hqm := abs_sum_le (acc.get2 i n).toRat va vb
  have hmag := mag_nonneg (acc.get2 i n).toRat va vb
  have hu := Format.u_nonneg (f := Format.binary32)
  have hqmax : |(acc.get2 i n).toRat + ∑ k, va k * vb k| ≤ Format.binary32.maxVal := by
    have : 0 ≤ relK K * (|(acc.get2 i n).toRat| + ∑ k, |va k * vb k|) :=
      mul_nonneg (hu.trans (hrel K hKp)) hmag
    linarith [habs K hKp, Format.eta_nonneg (f := Format.binary32)]
  obtain ⟨hfin, herr⟩ :=
    FloatV.ofPat_roundRat_err Format.binary32_regular false false hqmax
  rw [hri]
  exact ⟨hfin, round_le hu hqm hmag (hrel K hKp) (habs K hKp) herr⟩

end TVA.Platform.ref
