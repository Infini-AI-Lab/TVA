import TVA.Device.Weaken

/-!
# The reference platform

`Platform.ref` is a platform built from TVA's own float arithmetic, as accurate as a float
result can be:

* an approximate function (`P.math`) on a finite input gives a finite float within one
  rounding's bound (`u·|y| + η`) of the real function, wherever such a float exists. At the
  other inputs (`±∞`, NaN, and a finite input whose exact value overflows) it gives the value of
  the table `Platform.refSpecial`, if any: `exp` and `exp2` of `-∞` are `+0`, and `exp` in
  `float32` is `+∞` at `+∞` and from `1453635/16384` up;
* division and float `atomic_add` are IEEE (`FloatV.divRn`, `FloatV.add`), up to NaN payload;
* a conversion to nearest saturates (`FloatV.cvtSat .rn`, as sm_90's into fp8 do), toward zero
  it is `FloatV.cvtWith .rz`;
* `tl.dot` and `tl.dot_scaled` round the exact `acc + Σ a·b` once into the accumulator (with
  `K = 0`, the result is the accumulator);
* a float reduction combines along trees of depth at most `⌈log₂ n⌉`.

This file proves the general properties of `Platform.ref`: every operation has a result, and
each accuracy shape holds with constants at least one rounding's `u` and `η`. The facts of
`Device P` follow in `TVA.Device.Ref.Sm90`. `FloatV.exists_nearest` (a finite float within
`u·|y| + η` of every real `y` in range) is the one fact about floats the approximate functions
need.
-/
namespace TVA

/-- A rational of `y`'s sign within `ε` of `y`, with magnitude in `[lo, hi]` when `|y|` is. -/
private theorem exists_rat_near {y ε : ℝ} (hε : 0 < ε) {lo hi : ℚ} (hlo0 : 0 ≤ lo)
    (hlo : (lo : ℝ) ≤ |y|) (hhi : |y| ≤ (hi : ℝ)) :
    ∃ q : ℚ, |(q : ℝ) - y| ≤ ε ∧ lo ≤ |q| ∧ |q| ≤ hi := by
  obtain ⟨t, ht1, ht2, ht3⟩ : ∃ t : ℚ, (lo : ℝ) ≤ t ∧ (t : ℝ) ≤ hi ∧ abs ((t : ℝ) - |y|) ≤ ε := by
    by_cases h : max (lo : ℝ) (|y| - ε) < min (hi : ℝ) (|y| + ε)
    · obtain ⟨t, h1, h2⟩ := exists_rat_btwn h
      refine ⟨t, (le_max_left _ _).trans h1.le, h2.le.trans (min_le_left _ _), ?_⟩
      rw [abs_le]
      constructor
      · linarith [le_max_right (lo : ℝ) (|y| - ε)]
      · linarith [min_le_right (hi : ℝ) (|y| + ε)]
    · push Not at h
      have hm1 : max (lo : ℝ) (|y| - ε) ≤ |y| := max_le hlo (by linarith)
      have hm2 : |y| ≤ min (hi : ℝ) (|y| + ε) := le_min hhi (by linarith)
      have heq : max (lo : ℝ) (|y| - ε) = |y| := le_antisymm hm1 (hm2.trans h)
      have hly : (lo : ℝ) = |y| := by
        rcases max_choice (lo : ℝ) (|y| - ε) with hc | hc
        · rw [hc] at heq; exact heq
        · rw [hc] at heq; linarith
      exact ⟨lo, le_rfl, by rw [hly]; exact hhi, by rw [hly, sub_self, abs_zero]; exact hε.le⟩
  have ht0 : (0 : ℚ) ≤ t := by
    have : (0 : ℝ) ≤ t := le_trans (by exact_mod_cast hlo0) ht1
    exact_mod_cast this
  rcases le_or_gt 0 y with hy | hy
  · refine ⟨t, ?_, ?_, ?_⟩
    · rw [abs_of_nonneg hy] at ht3; exact ht3
    · rw [abs_of_nonneg ht0]; exact_mod_cast ht1
    · rw [abs_of_nonneg ht0]; exact_mod_cast ht2
  · refine ⟨-t, ?_, ?_, ?_⟩
    · rw [abs_of_neg hy] at ht3
      push_cast
      rw [show -(t : ℝ) - y = -((t : ℝ) - -y) by ring, abs_neg]
      exact ht3
    · rw [abs_neg, abs_of_nonneg ht0]; exact_mod_cast ht1
    · rw [abs_neg, abs_of_nonneg ht0]; exact_mod_cast ht2

/-- **The nearest float.** For a real `y` with `|y| ≤ maxVal`, some finite float is within
`u·|y| + η` of it (the nearest one: it is at least as near as the rounding of every rational near
`y`). Use it to show that an accuracy bound at least one rounding's can be met. -/
theorem FloatV.exists_nearest {f : Format} (hf : f.Regular) (y : ℝ)
    (hmax : |y| ≤ (f.maxVal : ℝ)) :
    ∃ r : FloatV f, r.isFinite = true ∧ |r.toReal - y| ≤ f.u * |y| + f.eta := by
  classical
  let S : Finset (FloatV f) := ((Finset.range (2 ^ f.width)).image
    (fun n => (FloatV.ofPat n : FloatV f))).filter (fun r => r.isFinite = true)
  have hround : ∀ q : ℚ, |q| ≤ f.maxVal →
      (FloatV.ofPat (f.roundRat .rn false false q) : FloatV f) ∈ S := by
    intro q hq
    obtain ⟨hlt, hfin, -⟩ := Format.roundRat_rn_err hf false false hq
    refine Finset.mem_filter.mpr ⟨Finset.mem_image_of_mem _ (Finset.mem_range.mpr hlt), ?_⟩
    rw [FloatV.isFinite_ofPat hlt]
    exact hfin
  have hmax0 : (0 : ℚ) ≤ f.maxVal := by
    have : (0 : ℝ) ≤ (f.maxVal : ℝ) := (abs_nonneg y).trans hmax
    exact_mod_cast this
  obtain ⟨r, hrS, hrmin⟩ :=
    S.exists_min_image (fun r => |r.toReal - y|) ⟨_, hround 0 (by simpa using hmax0)⟩
  have key : ∀ q : ℚ, |q| ≤ f.maxVal → |r.toReal - y| ≤
      |(((FloatV.ofPat (f.roundRat .rn false false q) : FloatV f).toRat : ℚ) : ℝ) - q| +
        |(q : ℝ) - y| := fun q hq =>
    (hrmin _ (hround q hq)).trans (by unfold FloatV.toReal; exact abs_sub_le _ _ _)
  have hu : (0 : ℝ) ≤ f.u := by exact_mod_cast Format.u_nonneg
  have hd : ∀ ε : ℝ, 0 < ε → 0 < ε / (1 + f.u) := fun ε hε => by positivity
  have hsum : ∀ (a ε : ℝ), 0 < ε → (a + ε / (1 + f.u)) * f.u + ε / (1 + f.u) = a * f.u + ε :=
    fun a ε _ => by field_simp; ring
  refine ⟨r, (Finset.mem_filter.mp hrS).2, ?_⟩
  refine le_of_forall_pos_le_add fun ε hε => ?_
  obtain ⟨q, hqy, -, hqhi⟩ :=
    exists_rat_near (hd ε hε) (lo := 0) (hi := f.maxVal) le_rfl (by simp) hmax
  have herr : |(((FloatV.ofPat (f.roundRat .rn false false q) : FloatV f).toRat : ℚ) : ℝ) - q| ≤
      |(q : ℝ)| * f.u + f.eta := by
    exact_mod_cast (FloatV.ofPat_roundRat_err hf false false hqhi).2
  have hq : |(q : ℝ)| ≤ |y| + ε / (1 + f.u) := by
    have := abs_sub_abs_le_abs_sub (q : ℝ) y
    linarith
  calc |r.toReal - y| ≤ _ := key q hqhi
    _ ≤ |(q : ℝ)| * f.u + f.eta + ε / (1 + f.u) := add_le_add herr hqy
    _ ≤ (|y| + ε / (1 + f.u)) * f.u + ε / (1 + f.u) + f.eta := by
        nlinarith [mul_le_mul_of_nonneg_right hq hu]
    _ = f.u * |y| + f.eta + ε := by rw [hsum _ _ hε]; ring

/-- The exact value of output element `(i, n)` of `tl.dot`: `acc + Σₖ a·b` on the elements'
values (`DType.val?`; `0` for an element without one). -/
def DotCfg.exact (cfg : DotCfg) {M N K : Nat} (a : Tensor [M, K] cfg.a.Val)
    (b : Tensor [K, N] cfg.b.Val) (acc : Tensor [M, N] cfg.acc.Val) (i : Fin M) (n : Fin N) : ℚ :=
  (cfg.acc.val? (acc.get2 i n)).getD 0 +
    ∑ k, (cfg.a.val? (a.get2 i k)).getD 0 * (cfg.b.val? (b.get2 k n)).getD 0

/-- The exact value of output element `(i, n)` of `tl.dot_scaled` (`scaledValue`; `0` for an
element without one). -/
def ScaledCfg.exact (cfg : ScaledCfg) {M N K : Nat} (a : Tensor [M, K] Nat)
    (sa : Option (Tensor [M, K] Nat)) (b : Tensor [K, N] Nat) (sb : Option (Tensor [K, N] Nat))
    (acc : Tensor [M, N] F32) (i : Fin M) (n : Fin N) : ℚ :=
  (acc.get2 i n).toRat +
    ∑ k, (scaledValue cfg.lhs cfg.lhsScale (a.get2 i k) (sa.map fun t => t.get2 i k)).getD 0 *
      (scaledValue cfg.rhs cfg.rhsScale (b.get2 k n) (sb.map fun t => t.get2 k n)).getD 0

/-- The element of a dtype for a rational: rounded to nearest for the floats, the floor (wrapped)
for the integers, `q ≠ 0` for `tl.int1`. -/
def DType.ofRat : (d : DType) → ℚ → d.Val
  | .f16, q => FloatV.ofRat q | .bf16, q => FloatV.ofRat q | .f32, q => FloatV.ofRat q
  | .f64, q => FloatV.ofRat q | .f8e5, q => FloatV.ofRat q | .f8e4nv, q => FloatV.ofRat q
  | .i8, q => IntV.ofInt ⌊q⌋ | .i16, q => IntV.ofInt ⌊q⌋ | .i32, q => IntV.ofInt ⌊q⌋
  | .i64, q => IntV.ofInt ⌊q⌋
  | .u8, q => IntV.ofInt ⌊q⌋ | .u16, q => IntV.ofInt ⌊q⌋ | .u32, q => IntV.ofInt ⌊q⌋
  | .u64, q => IntV.ofInt ⌊q⌋
  | .i1, q => decide (q ≠ 0)

/-- The format of a float dtype is regular. -/
theorem DType.regular_of_format? {d : DType} {g : Format} (hd : d.format? = some g) :
    g.Regular := by
  cases d <;> simp only [DType.format?, reduceCtorEq, Option.some.injEq] at hd <;> subst hd
  · exact Format.binary16_regular
  · exact Format.bfloat16_regular
  · exact Format.binary32_regular
  · exact Format.binary64_regular
  · exact Format.e5m2_regular
  · exact Format.e4m3_regular

/-- The bound of a float dtype is its format's `maxVal`. -/
theorem DType.bound_of_format? {d : DType} {g : Format} (hd : d.format? = some g) :
    d.bound = g.maxVal := by
  cases d <;> simp only [DType.format?, reduceCtorEq, Option.some.injEq] at hd <;> subst hd <;> rfl

/-- Rounding a rational in range into a float dtype gives a finite value within one rounding's
bound, `|q|·u + η`. -/
theorem DType.val?_ofRat_float {d : DType} {g : Format} (hd : d.format? = some g)
    {q : ℚ} (hq : |q| ≤ g.maxVal) :
    ∃ v, d.val? (d.ofRat q) = some v ∧ |v - q| ≤ |q| * g.u + g.eta := by
  have hg := DType.regular_of_format? hd
  cases d <;> simp only [DType.format?, reduceCtorEq, Option.some.injEq] at hd <;> subst hd
  all_goals
    obtain ⟨hfin, herr⟩ := FloatV.ofPat_roundRat_err hg false false hq
    exact ⟨_, by simp only [DType.val?, DType.ofRat, FloatV.ofRat, hfin, ite_true], herr⟩

/-- The special values of `Platform.ref`: `exp` and `exp2` of `-∞` are `+0`; `exp` in `float32`
is `+∞` at `+∞` and at the finite inputs from `1453635/16384` (`88.72283935546875`), where no
float is within one rounding of `e^x`; nothing else is fixed. -/
def Platform.refSpecial : (fn : MathFn) → (f : Format) → FloatV f → Option (FloatV f) :=
  fun fn f x =>
    if (fn = .exp ∨ fn = .exp2) ∧ x.pat = f.setSign true f.infMag then some (FloatV.ofPat 0)
    else if fn = .exp ∧ f = Format.binary32 ∧
        ((x.isFinite = true ∧ (1453635 / 16384 : ℚ) ≤ x.toRat) ∨ x.pat = 0x7f800000) then
      some (FloatV.ofPat 0x7f800000)
    else none

/-- `r` is a finite float within one rounding's bound (`u·|ref x| + η`) of the real function at
`x`, for the reference `ref` of an approximate function. -/
def Platform.RefNear {f : Format} (ref : ℝ → ℝ) (x r : FloatV f) : Prop :=
  r.isFinite = true ∧ |r.toReal - ref x.toReal| ≤ (f.u : ℝ) * |ref x.toReal| + (f.eta : ℝ)

/-- The reference platform (see the module docstring). -/
noncomputable def Platform.ref : Platform where
  math fn f x r :=
    if x.isFinite = true then
      (∀ ref, fn.ref = some ref → (∃ r', Platform.RefNear ref x r') → Platform.RefNear ref x r) ∧
      ((¬ ∃ ref, fn.ref = some ref ∧ ∃ r', Platform.RefNear ref x r') →
        ∀ y, Platform.refSpecial fn f x = some y → r = y)
    else ∀ y, Platform.refSpecial fn f x = some y → r = y
  div _ x y r := FloatV.NaNEq (FloatV.divRn x y) r
  dot := fun cfg {_ _ K} a b acc r => ∀ i n,
    r.get2 i n = if K = 0 then acc.get2 i n else cfg.acc.ofRat (cfg.exact a b acc i n)
  cvt _ _ rnd x r :=
    FloatV.NaNEq (if rnd = .rn then FloatV.cvtSat .rn x else FloatV.cvtWith rnd x) r
  atomicAdd _ o v n := FloatV.NaNEq (FloatV.add o v) n
  dotScaled := fun cfg {_ _ K} a sa b sb acc r => ∀ i n,
    r.get2 i n = if K = 0 then acc.get2 i n else FloatV.ofRat (cfg.exact a sa b sb acc i n)
  sumDepth n := Nat.clog 2 n

end TVA
