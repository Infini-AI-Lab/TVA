import TVA.Lemmas.Claim
import TVA.Lemmas.Int
import TVA.Rules.Num

/-!
# Consistency of `Compiler.Faithful`

A claim's assumptions could be contradictory, and then every specification follows. A bundle
rules this out by proving `Claim.Consistent Assumptions`. For the rule `Compiler.Faithful`, this
file gives the witness:

* `Step.progress`: a program that has not returned can take a step (normally or into undefined
  behaviour); no state is stuck.
* `WellBehaved.outcome`: a well-behaved launch ends normally or runs forever.
* `exists_total_faithful`: the behaviour that follows the model (and faults where the launch is
  not well behaved) is total and faithful on every instance.
* `Platform.any`: a platform whose operations may return anything, for bundles that assume
  nothing about `P`.
* `Platform.ref rel abs sp`: the most accurate platform.
  - An approximate function on a finite input returns a finite float within
    `rel · |ref x| + abs` of the real function whenever one exists. On other inputs, and on a
    finite input where no float is that close (an overflow), it returns the value the table `sp`
    gives, if any (library issue L154).
  - Division and float `atomic_add` are IEEE (up to NaN payload). A conversion to nearest
    saturates (`FloatV.cvtSat .rn`: `±maxVal` beyond the range, as sm_90's conversions into fp8
    do); toward zero it is `FloatV.cvtWith .rz`.
  - `tl.dot` and `tl.dot_scaled` round the exact `acc + Σ a·b` once into the accumulator.

  So these hold together, for every rule a bundle may assume about `P`:
  - `MathTotal` (`mathTotal`);
  - `MathAcc`: `mathAcc_of_eta` for `rel ≥ u`, `abs ≥ η`; `mathAcc_of_normal` for `rel ≥ u`,
    `abs ≥ 0` where the reference values are normal; `mathAcc` from any proof that near floats
    exist;
  - `MathAt` at non-finite inputs (`mathAt`), and at finite inputs where no float is near
    (`mathAt_far`);
  - `DivRn`, `DivFull`, `DivTotal`;
  - `CvtAcc`, `CvtSat` (`cvtSat`), `CvtTotal`;
  - `AtomicAddAcc` (`rel ≥ u`, `abs ≥ η`) and `AtomicAddTotal`;
  - `DotTotal`, and `DotAcc`: `dotAcc` for float accumulators with `rel K ≥ u`, `abs K ≥ η`
    for `K > 0`, and `dotAcc_int` for integer operands with a signed accumulator, with `rel`,
    `abs ≥ 0`;
  - `DotScaledTotal`, and `DotScaledAcc` for `rel K ≥ u`, `abs K ≥ η` for `K > 0`
    (`dotScaledAcc`).

  At `K = 0` the bounds need only be nonnegative, since the result is the accumulator. The
  rounding argument is `FloatV.exists_nearest`: the nearest float to a real `y` with
  `|y| ≤ maxVal` is within one rounding's bound (`u·|y| + η`, and `u·|y|` in the normal range).
-/
namespace TVA

open TVA.Claim

/-- **Progress.** A program that has not returned can step, normally or into undefined
behaviour. -/
theorem Step.progress {α : Type} (E : Env) (m : Mem) (p : Prog α) (hp : ∀ a, p ≠ .ret a) :
    ∃ o, Step E m p o := by
  classical
  cases p with
  | ret a => exact absurd rfl (hp a)
  | load d s addrs k =>
    by_cases hb : LoadBad m d addrs
    · exact ⟨_, .loadUB hb⟩
    · refine ⟨_, .load ⟨fun i => match addrs i with
          | some a => (m.read d a).getD default
          | none => default⟩ ?_⟩
      intro i a ha
      have : m.read d a ≠ none := fun h => hb ⟨i, a, ha, h⟩
      obtain ⟨v, hv⟩ := Option.ne_none_iff_exists'.mp this
      simp [ha, hv]
  | store d s addrs vals k =>
    by_cases hs : StoreOk m d addrs
    · exact ⟨_, .store _ hs (List.Perm.refl _)⟩
    · exact ⟨_, .storeUB hs⟩
  | rmw d a upd sem sc k =>
    cases hr : m.read d a with
    | none => exact ⟨_, .rmwUB (Or.inl hr)⟩
    | some old =>
      by_cases hn : ∃ new, upd old new
      · obtain ⟨new, hu⟩ := hn; exact ⟨_, .rmw old new hr hu⟩
      · push Not at hn; exact ⟨_, .rmwUB (Or.inr ⟨old, hr, hn⟩)⟩
  | pick P k =>
    by_cases h : ∃ b, P b
    · obtain ⟨b, hb⟩ := h; exact ⟨_, .pick b hb⟩
    · push Not at h; exact ⟨_, .pickUB h⟩
  | ub => exact ⟨_, .ub⟩
  | loop init body k => exact ⟨_, .loop⟩
  | env k => exact ⟨_, .env⟩

variable {E : Env} {k : Kernel} {L : Launch} {init : Mem}

/-- A reachable configuration of a well-behaved launch that has not finished can step. -/
theorem WellBehaved.step (hwb : WellBehaved E k L init) {tr : List (Fin L.size × Event)}
    {c : Config L.size} (hr : Reach E (Config.start k L init) tr c) (hnd : ¬ c.Done) :
    ∃ ev c', LStep E c ev c' := by
  obtain ⟨i, hi⟩ : ∃ i, c.progs i ≠ .ret () := by
    by_contra h
    push Not at h
    exact hnd h
  obtain ⟨o, ho⟩ := Step.progress E c.mem (c.progs i) (fun a => by cases a; exact hi)
  cases o with
  | ub => exact absurd ⟨i, ho⟩ (hwb tr c hr).1
  | next m' p' e => exact ⟨(i, e), _, .step i ho⟩

/-- **A well-behaved launch ends normally or runs forever.** -/
theorem WellBehaved.outcome (hwb : WellBehaved E k L init) :
    (∃ fin, Outcomes E k L init (.ok fin)) ∨ Outcomes E k L init .hang := by
  classical
  by_cases hok : ∃ tr c, Reach E (Config.start k L init) tr c ∧ c.Done
  · obtain ⟨tr, c, hr, hd⟩ := hok
    exact Or.inl ⟨c.mem, tr, c, hr, hd, rfl⟩
  · right
    push Not at hok
    have hnext : ∀ s : {c : Config L.size // ∃ tr, Reach E (Config.start k L init) tr c},
        ∃ s' : {c : Config L.size // ∃ tr, Reach E (Config.start k L init) tr c},
          ∃ ev, LStep E s.1 ev s'.1 := by
      rintro ⟨c, tr, hr⟩
      obtain ⟨ev, c', hs⟩ := hwb.step hr (hok tr c hr)
      exact ⟨⟨c', tr ++ [ev], hr.tail hs⟩, ev, hs⟩
    choose nxt hnxt using hnext
    let g : Nat → {c : Config L.size // ∃ tr, Reach E (Config.start k L init) tr c} :=
      fun n => Nat.rec ⟨Config.start k L init, [], .refl _⟩ (fun _ s => nxt s) n
    exact ⟨fun n => (g n).1, rfl, fun n => hnxt (g n)⟩

open Classical in
/-- The behaviour of a device that follows the model exactly: a well-behaved launch ends as the
semantics says, and any other launch faults. -/
noncomputable def modelBehaviour (P : Platform) (k : Kernel) (L : Launch) (init : Mem) (o : Outcome) : Prop :=
  if WellBehaved (L.env P) k L init then
    match o with
    | .ok fin => Outcomes (L.env P) k L init (.ok fin)
    | .fault => False
    | .hang => Outcomes (L.env P) k L init .hang
  else o = .fault

theorem faithful_modelBehaviour (P : Platform) (k : Kernel) (L : Launch) :
    Rules.Compiler.Faithful P k L (modelBehaviour P k L) := by
  intro _ _ init hwb o ho
  simp only [modelBehaviour, if_pos hwb] at ho
  cases o <;> exact ho

theorem modelBehaviour_total (P : Platform) (k : Kernel) (L : Launch) (init : Mem) :
    ∃ o, modelBehaviour P k L init o := by
  by_cases hwb : WellBehaved (L.env P) k L init
  · rcases hwb.outcome with ⟨fin, h⟩ | h
    · exact ⟨.ok fin, by simp only [modelBehaviour, if_pos hwb]; exact h⟩
    · exact ⟨.hang, by simp only [modelBehaviour, if_pos hwb]; exact h⟩
  · exact ⟨.fault, by simp only [modelBehaviour, if_neg hwb]⟩

/-- **`Compiler.Faithful` is consistent.** For any models and launches, some total behaviour is
faithful on every instance. -/
theorem exists_total_faithful {Inst : Type} (P : Platform) (model : Inst → Kernel)
    (launch : Inst → Launch) :
    ∃ K : Behaviour Inst, Total K ∧ ∀ d, Rules.Compiler.Faithful P (model d) (launch d) (K d) :=
  ⟨fun d => modelBehaviour P (model d) (launch d), fun d init => modelBehaviour_total _ _ _ init,
    fun d => faithful_modelBehaviour _ _ _⟩

/-- A platform whose operations may return anything. -/
def Platform.any : Platform where
  math _ _ _ _ := True
  div _ _ _ _ := True
  dot := fun _ {_ _ _} _ _ _ _ => True
  cvt _ _ _ _ _ := True
  atomicAdd _ _ _ _ := True
  dotScaled := fun _ {_ _ _} _ _ _ _ _ _ => True

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

/-- **The nearest float.** For a real `y` with `|y| ≤ maxVal`, the finite float nearest to `y`
is within `u·|y| + η` of it, and within `u·|y|` when `|y| ≥ 2^emin`: the bound of one rounding to
nearest (at least half an ulp, and up to about a whole one). It is at least
as near as the rounding of every rational near `y`, and these come arbitrarily close to that
bound. -/
theorem FloatV.exists_nearest {f : TVA.Math.Float.Format} (hf : f.Regular) (y : ℝ)
    (hmax : |y| ≤ (f.maxVal : ℝ)) :
    ∃ r : FloatV f, r.isFinite = true ∧ |r.toReal - y| ≤ f.u * |y| + f.eta ∧
      ((((2 : ℚ) ^ f.emin : ℚ) : ℝ) ≤ |y| → |r.toReal - y| ≤ f.u * |y|) := by
  classical
  let S : Finset (FloatV f) := ((Finset.range (2 ^ f.width)).image
    (fun n => (FloatV.ofPat n : FloatV f))).filter (fun r => r.isFinite = true)
  have hround : ∀ q : ℚ, |q| ≤ f.maxVal →
      (FloatV.ofPat (f.roundRat .rn false false q) : FloatV f) ∈ S := by
    intro q hq
    obtain ⟨hlt, hfin, -⟩ := TVA.Math.Float.Format.roundRat_rn_err hf false false hq
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
  have hu : (0 : ℝ) ≤ f.u := by exact_mod_cast TVA.Math.Float.Format.u_nonneg
  have hd : ∀ ε : ℝ, 0 < ε → 0 < ε / (1 + f.u) := fun ε hε => by positivity
  have hsum : ∀ (a ε : ℝ), 0 < ε → (a + ε / (1 + f.u)) * f.u + ε / (1 + f.u) = a * f.u + ε :=
    fun a ε _ => by field_simp; ring
  refine ⟨r, (Finset.mem_filter.mp hrS).2, ?_, ?_⟩
  · refine le_of_forall_pos_le_add fun ε hε => ?_
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
  · intro hmin
    refine le_of_forall_pos_le_add fun ε hε => ?_
    obtain ⟨q, hqy, hqlo, hqhi⟩ := exists_rat_near (hd ε hε) (lo := 2 ^ f.emin) (hi := f.maxVal)
      (zpow_pos (by norm_num) _).le hmin hmax
    have herr : |(((FloatV.ofPat (f.roundRat .rn false false q) : FloatV f).toRat : ℚ) : ℝ) - q| ≤
        |(q : ℝ)| * f.u := by
      exact_mod_cast (FloatV.ofPat_roundRat_err_normal hf false false hqlo hqhi).2
    have hq : |(q : ℝ)| ≤ |y| + ε / (1 + f.u) := by
      have := abs_sub_abs_le_abs_sub (q : ℝ) y
      linarith
    calc |r.toReal - y| ≤ _ := key q hqhi
      _ ≤ |(q : ℝ)| * f.u + ε / (1 + f.u) := add_le_add herr hqy
      _ ≤ (|y| + ε / (1 + f.u)) * f.u + ε / (1 + f.u) := by
          nlinarith [mul_le_mul_of_nonneg_right hq hu]
      _ = f.u * |y| + ε := by rw [hsum _ _ hε]; ring

/-- **Some float is near any real with room in the format**: within `rel·|y| + abs` when
`rel ≥ u` and `abs ≥ η` (the nearest float). -/
theorem FloatV.exists_near {f : TVA.Math.Float.Format} (hf : f.Regular) (y rel abs : ℝ)
    (hrel : (f.u : ℝ) ≤ rel) (habs : (f.eta : ℝ) ≤ abs)
    (hroom : (1 + rel) * |y| + abs ≤ (f.maxVal : ℝ)) :
    ∃ r : FloatV f, r.isFinite = true ∧ |r.toReal - y| ≤ rel * |y| + abs := by
  have hu : (0 : ℝ) ≤ f.u := by exact_mod_cast TVA.Math.Float.Format.u_nonneg
  have he : (0 : ℝ) ≤ f.eta := by exact_mod_cast TVA.Math.Float.Format.eta_nonneg
  have h1 := mul_le_mul_of_nonneg_right hrel (abs_nonneg y)
  have h2 : 0 ≤ rel * |y| := mul_nonneg (hu.trans hrel) (abs_nonneg y)
  obtain ⟨r, hr, hnear, -⟩ := FloatV.exists_nearest hf y (by linarith)
  exact ⟨r, hr, by linarith⟩

/-- **Some float is near any real in the normal range**, without an absolute term: within
`rel·|y| + abs` when `rel ≥ u`, `abs ≥ 0` and `|y| ≥ 2^emin` (the nearest float). -/
theorem FloatV.exists_near_normal {f : TVA.Math.Float.Format} (hf : f.Regular) (y rel abs : ℝ)
    (hrel : (f.u : ℝ) ≤ rel) (habs : 0 ≤ abs)
    (hmin : (((2 : ℚ) ^ f.emin : ℚ) : ℝ) ≤ |y|) (hroom : (1 + rel) * |y| + abs ≤ (f.maxVal : ℝ)) :
    ∃ r : FloatV f, r.isFinite = true ∧ |r.toReal - y| ≤ rel * |y| + abs := by
  have hu : (0 : ℝ) ≤ f.u := by exact_mod_cast TVA.Math.Float.Format.u_nonneg
  have h1 := mul_le_mul_of_nonneg_right hrel (abs_nonneg y)
  have h2 : 0 ≤ rel * |y| := mul_nonneg (hu.trans hrel) (abs_nonneg y)
  obtain ⟨r, hr, -, hnear⟩ := FloatV.exists_nearest hf y (by linarith)
  exact ⟨r, hr, by linarith [hnear hmin]⟩

/-- The exact value of output element `(i, n)` of `tl.dot`: `acc + Σₖ a·b` on the elements'
values (`DType.val?`; `0` for an element without one). -/
def DotCfg.exact (cfg : DotCfg) {M N K : Nat} (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc : Tensor [M, N] cfg.acc.Val) (i : Fin M) (n : Fin N) : ℚ :=
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

/-- The dtype element for a rational: rounded to nearest for floats, the floor (wrapped) for
integers. -/
def DType.ofRat : (d : DType) → ℚ → d.Val
  | .f16, q => FloatV.ofRat q | .bf16, q => FloatV.ofRat q | .f32, q => FloatV.ofRat q
  | .f64, q => FloatV.ofRat q | .f8e5, q => FloatV.ofRat q | .f8e4nv, q => FloatV.ofRat q
  | .i8, q => IntV.ofInt ⌊q⌋ | .i16, q => IntV.ofInt ⌊q⌋ | .i32, q => IntV.ofInt ⌊q⌋
  | .i64, q => IntV.ofInt ⌊q⌋
  | .u8, q => IntV.ofInt ⌊q⌋ | .u16, q => IntV.ofInt ⌊q⌋ | .u32, q => IntV.ofInt ⌊q⌋
  | .u64, q => IntV.ofInt ⌊q⌋
  | .i1, q => decide (q ≠ 0)

/-- The integer dtypes. -/
def DType.isInt : DType → Bool
  | .i8 | .i16 | .i32 | .i64 | .u8 | .u16 | .u32 | .u64 => true
  | _ => false

/-- The signed integer dtypes. -/
def DType.isSInt : DType → Bool
  | .i8 | .i16 | .i32 | .i64 => true
  | _ => false

theorem DType.isInt_of_isSInt {d : DType} (h : d.isSInt = true) : d.isInt = true := by
  cases d <;> first | rfl | cases h

theorem DType.regular_of_format? {d : DType} {g : TVA.Math.Float.Format} (hd : d.format? = some g) :
    g.Regular := by
  cases d <;> simp only [DType.format?, reduceCtorEq, Option.some.injEq] at hd <;> subst hd
  · exact TVA.Math.Float.Format.binary16_regular
  · exact TVA.Math.Float.Format.bfloat16_regular
  · exact TVA.Math.Float.Format.binary32_regular
  · exact TVA.Math.Float.Format.binary64_regular
  · exact TVA.Math.Float.Format.e5m2_regular
  · exact TVA.Math.Float.Format.e4m3_regular

theorem DType.bound_of_format? {d : DType} {g : TVA.Math.Float.Format} (hd : d.format? = some g) :
    d.bound = g.maxVal := by
  cases d <;> simp only [DType.format?, reduceCtorEq, Option.some.injEq] at hd <;> subst hd <;> rfl

/-- Rounding a rational in range into a float dtype gives a finite value within one rounding's
bound, `|q|·u + η`. -/
theorem DType.val?_ofRat_float {d : DType} {g : TVA.Math.Float.Format} (hd : d.format? = some g)
    {q : ℚ} (hq : |q| ≤ g.maxVal) : ∃ v, d.val? (d.ofRat q) = some v ∧ |v - q| ≤ |q| * g.u + g.eta := by
  have hg := DType.regular_of_format? hd
  cases d <;> simp only [DType.format?, reduceCtorEq, Option.some.injEq] at hd <;> subst hd
  all_goals
    obtain ⟨hfin, herr⟩ := FloatV.ofPat_roundRat_err hg false false hq
    exact ⟨_, by simp only [DType.val?, DType.ofRat, FloatV.ofRat, hfin,
      if_true], herr⟩

theorem DType.int_of_val? {d : DType} (hd : d.isInt = true) {x : d.Val} {v : ℚ}
    (h : d.val? x = some v) : ∃ z : ℤ, v = z := by
  cases d <;> simp only [DType.isInt, Bool.false_eq_true] at hd <;>
    simp only [DType.val?, Option.some.injEq] at h <;> exact ⟨_, h.symm⟩

private theorem toInt_ofInt_rat {w : Nat} (hw : 0 < w) {z : ℤ} {B : ℚ} (hB : (2 : ℚ) ^ (w - 1) - 1 = B)
    (hz : |(z : ℚ)| ≤ B) : (((IntV.ofInt z : IntV true w).toInt : ℤ) : ℚ) = z := by
  subst hB
  have h2 := abs_le.mp hz
  have h1 : (z : ℚ) < 2 ^ (w - 1) := by linarith [h2.2]
  have h0 : -(2 ^ (w - 1) : ℚ) ≤ z := by linarith [h2.1]
  rw [IntV.toInt_ofInt hw ⟨by exact_mod_cast h0, by exact_mod_cast h1⟩]

/-- An integer in range, put into a signed integer dtype, reads back as itself. -/
theorem DType.val?_ofRat_sint {d : DType} (hd : d.isSInt = true) (z : ℤ) (hz : |(z : ℚ)| ≤ d.bound) :
    d.val? (d.ofRat z) = some (z : ℚ) := by
  cases d <;> simp only [DType.isSInt, Bool.false_eq_true] at hd <;>
    simp only [DType.bound] at hz <;>
    simp only [DType.val?, DType.ofRat, Int.floor_intCast, Option.some.injEq]
  · exact toInt_ofInt_rat (w := 8) (by norm_num) (by norm_num) hz
  · exact toInt_ofInt_rat (w := 16) (by norm_num) (by norm_num) hz
  · exact toInt_ofInt_rat (w := 32) (by norm_num) (by norm_num) hz
  · exact toInt_ofInt_rat (w := 64) (by norm_num) (by norm_num) hz

/-- The most accurate platform (see the module docstring): approximate functions as accurate as
`rel fn f`, `abs fn f` ask wherever a float that accurate exists, special values from the table
`sp` at the other inputs (non-finite ones, and finite ones where no float is that accurate); IEEE division and float `atomic_add`; conversions to nearest that saturate (`FloatV.cvtSat`),
toward zero as `FloatV.cvtWith .rz`; `tl.dot` and `tl.dot_scaled` rounded once from the exact
result (with `K = 0`, the accumulator itself); float reductions along trees of depth at most
`⌈log₂ n⌉` (`Platform.sumDepth`, the depth of a balanced tree). -/
noncomputable def Platform.ref (rel abs : MathFn → TVA.Math.Float.Format → ℝ)
    (sp : (fn : MathFn) → (f : TVA.Math.Float.Format) → FloatV f → Option (FloatV f) :=
      fun _ _ _ => none) : Platform where
  math fn f x r :=
    if x.isFinite = true then
      (∀ ref, fn.ref = some ref →
        (∃ r' : FloatV f, r'.isFinite = true ∧
          |r'.toReal - ref x.toReal| ≤ rel fn f * |ref x.toReal| + abs fn f) →
        r.isFinite = true ∧ |r.toReal - ref x.toReal| ≤ rel fn f * |ref x.toReal| + abs fn f) ∧
      ((¬ ∃ ref, fn.ref = some ref ∧ ∃ r' : FloatV f, r'.isFinite = true ∧
          |r'.toReal - ref x.toReal| ≤ rel fn f * |ref x.toReal| + abs fn f) →
        ∀ y, sp fn f x = some y → r = y)
    else ∀ y, sp fn f x = some y → r = y
  div _ x y r := FloatV.NaNEq (FloatV.divRn x y) r
  dot := fun cfg {_ _ K} a b acc r => ∀ i n,
    r.get2 i n = if K = 0 then acc.get2 i n else cfg.acc.ofRat (cfg.exact a b acc i n)
  cvt _ _ rnd x r :=
    FloatV.NaNEq (if rnd = .rn then FloatV.cvtSat .rn x else FloatV.cvtWith rnd x) r
  atomicAdd _ o v n := FloatV.NaNEq (FloatV.add o v) n
  dotScaled := fun cfg {_ _ K} a sa b sb acc r => ∀ i n,
    r.get2 i n = if K = 0 then acc.get2 i n else FloatV.ofRat (cfg.exact a sa b sb acc i n)
  sumDepth n := Nat.clog 2 n

namespace Platform.ref

variable {rel abs : MathFn → TVA.Math.Float.Format → ℝ}
  {sp : (fn : MathFn) → (f : TVA.Math.Float.Format) → FloatV f → Option (FloatV f)}

/-- `MathAcc` on `dom`, given that floats that accurate exist there (`hnear`). -/
theorem mathAcc {fn : MathFn} {f : TVA.Math.Float.Format} {g : ℝ → ℝ} (hg : fn.ref = some g)
    {dom : Set ℝ}
    (hnear : ∀ x : FloatV f, x.isFinite = true → x.toReal ∈ dom → x.toReal ∈ fn.natDom →
      (1 + rel fn f) * |g x.toReal| + abs fn f ≤ (f.maxVal : ℝ) →
      ∃ r' : FloatV f, r'.isFinite = true ∧
        |r'.toReal - g x.toReal| ≤ rel fn f * |g x.toReal| + abs fn f) :
    Rules.Num.MathAcc (Platform.ref rel abs sp) fn f dom (rel fn f) (abs fn f) := by
  refine ⟨g, hg, fun x r hr hx hd hn hroom => ?_⟩
  simp only [Platform.ref, hx, if_true] at hr
  exact hr.1 g hg (hnear x hx hd hn hroom)

/-- `MathAcc` on any domain for `rel ≥ u` and `abs ≥ η`. -/
theorem mathAcc_of_eta {fn : MathFn} {f : TVA.Math.Float.Format} {g : ℝ → ℝ} (hg : fn.ref = some g)
    (hf : f.Regular) (hrel : (f.u : ℝ) ≤ rel fn f) (habs : (f.eta : ℝ) ≤ abs fn f)
    (dom : Set ℝ) : Rules.Num.MathAcc (Platform.ref rel abs sp) fn f dom (rel fn f) (abs fn f) :=
  mathAcc hg fun _ _ _ _ hroom => FloatV.exists_near hf _ _ _ hrel habs hroom

/-- `MathAcc` with any `abs ≥ 0` (also below η) where the reference values are normal
(`|ref x| ≥ 2^emin` on `dom`), for `rel ≥ u`. For example `exp2` on `float32` over
`[-126, 126]` with `abs = 0`. -/
theorem mathAcc_of_normal {fn : MathFn} {f : TVA.Math.Float.Format} {g : ℝ → ℝ}
    (hg : fn.ref = some g) (hf : f.Regular) (hrel : (f.u : ℝ) ≤ rel fn f)
    (habs : 0 ≤ abs fn f) {dom : Set ℝ}
    (hnorm : ∀ x ∈ dom, x ∈ fn.natDom → (((2 : ℚ) ^ f.emin : ℚ) : ℝ) ≤ |g x|) :
    Rules.Num.MathAcc (Platform.ref rel abs sp) fn f dom (rel fn f) (abs fn f) :=
  mathAcc hg fun _ _ hd hn hroom =>
    FloatV.exists_near_normal hf _ _ _ hrel habs (hnorm _ hd hn) hroom

theorem mathAt {fn : MathFn} {f : TVA.Math.Float.Format} {x y : FloatV f} (hx : x.isFinite = false)
    (h : sp fn f x = some y) : Rules.Num.MathAt (Platform.ref rel abs sp) fn f x y := by
  intro r hr
  simp only [Platform.ref, hx, Bool.false_eq_true] at hr
  exact hr y h

/-- `MathAt` at a finite input where no float is within `rel fn f · |ref x| + abs fn f` of the
real function (an overflow): the table's value. -/
theorem mathAt_far {fn : MathFn} {f : TVA.Math.Float.Format} {x y : FloatV f} (hx : x.isFinite = true)
    (hfar : ¬ ∃ ref, fn.ref = some ref ∧ ∃ r' : FloatV f, r'.isFinite = true ∧
      |r'.toReal - ref x.toReal| ≤ rel fn f * |ref x.toReal| + abs fn f)
    (h : sp fn f x = some y) : Rules.Num.MathAt (Platform.ref rel abs sp) fn f x y := by
  intro r hr
  simp only [Platform.ref, hx, if_true] at hr
  exact hr.2 hfar y h

theorem mathTotal {fn : MathFn} {f : TVA.Math.Float.Format} :
    Rules.Num.MathTotal (Platform.ref rel abs sp) fn f := by
  intro x
  by_cases hx : x.isFinite = true
  · simp only [Platform.ref, hx, if_true]
    by_cases hpre : ∃ g, fn.ref = some g ∧ ∃ r' : FloatV f, r'.isFinite = true ∧
        |r'.toReal - g x.toReal| ≤ rel fn f * |g x.toReal| + abs fn f
    · obtain ⟨g, hg, r', hr'⟩ := hpre
      refine ⟨r', fun g' hg' _ => ?_, fun hno => absurd ⟨g, hg, r', hr'⟩ hno⟩
      rw [hg] at hg'
      cases hg'
      exact hr'
    · cases h : sp fn f x with
      | some y => exact ⟨y, fun g hg hex => absurd ⟨g, hg, hex⟩ hpre, fun _ y' hy' => Option.some.inj hy'⟩
      | none => exact ⟨x, fun g hg hex => absurd ⟨g, hg, hex⟩ hpre, fun _ y' hy' => by cases hy'⟩
  · simp only [Platform.ref, hx]
    cases h : sp fn f x with
    | some y => exact ⟨y, fun y' hy' => Option.some.inj hy'⟩
    | none => exact ⟨x, fun y' hy' => by cases hy'⟩

theorem divRn (f : TVA.Math.Float.Format) : Rules.Num.DivRn (Platform.ref rel abs sp) f :=
  fun _ _ _ h => h

theorem divTotal (f : TVA.Math.Float.Format) : Rules.Num.DivTotal (Platform.ref rel abs sp) f :=
  fun _ _ => ⟨_, FloatV.NaNEq.refl _⟩

theorem divFull : Rules.Num.DivFull (Platform.ref rel abs sp) :=
  Rules.Num.DivFull.of_divRn (divRn _)

theorem cvtTotal (f g : TVA.Math.Float.Format) (rnd : TVA.Math.Float.RoundMode) :
    Rules.Num.CvtTotal (Platform.ref rel abs sp) f g rnd :=
  fun _ => ⟨_, FloatV.NaNEq.refl _⟩

theorem cvtAcc {f g : TVA.Math.Float.Format} (hg : g.Regular) :
    Rules.Num.CvtAcc (Platform.ref rel abs sp) f g := by
  intro x r hr hx hmax
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err hg hx hmax
  simp only [Platform.ref, ↓reduceIte] at hr
  rw [FloatV.cvtSat_of_le hg .rn hx hmax, FloatV.cvtWith_rn] at hr
  rw [hr.eq (FloatV.isFinite_iff.mp hfin).1]
  exact ⟨hfin, herr⟩

/-- Conversions to nearest saturate, so every `CvtSat` holds. -/
theorem cvtSat (f g : TVA.Math.Float.Format) : Rules.Num.CvtSat (Platform.ref rel abs sp) f g := by
  intro x r hr
  simpa only [Platform.ref, ↓reduceIte] using hr

theorem atomicAddTotal (f : TVA.Math.Float.Format) :
    Rules.Num.AtomicAddTotal (Platform.ref rel abs sp) f :=
  fun _ _ => ⟨_, FloatV.NaNEq.refl _⟩

theorem atomicAddAcc {f : TVA.Math.Float.Format} (hf : f.Regular) {r a : ℚ} (hr : f.u ≤ r)
    (ha : f.eta ≤ a) : Rules.Num.AtomicAddAcc (Platform.ref rel abs sp) f r a := by
  intro o v n hn ho hv hmax
  obtain ⟨hfin, herr⟩ := FloatV.add_err hf ho hv hmax
  change FloatV.NaNEq (FloatV.add o v) n at hn
  have hnan : (FloatV.add o v).isNaN = false := (FloatV.isFinite_iff.mp hfin).1
  rw [hn.eq hnan]
  refine ⟨hfin, herr.trans ?_⟩
  have h1 := mul_le_mul_of_nonneg_left hr (abs_nonneg (o.toRat + v.toRat))
  have h2 : |o.toRat + v.toRat| * r = r * |o.toRat + v.toRat| := mul_comm _ _
  linarith

theorem dotTotal (cfg : DotCfg) : Rules.Num.DotTotal (Platform.ref rel abs sp) cfg :=
  fun ⦃_ _ K⦄ a b acc => ⟨Tensor.of2 fun i n =>
    if K = 0 then acc.get2 i n else cfg.acc.ofRat (cfg.exact a b acc i n),
    fun _ _ => Tensor.get2_of2 _ _ _⟩

/-- The exact sum is bounded by `mag`. -/
private theorem abs_sum_le {K : Nat} (c : ℚ) (va vb : Fin K → ℚ) :
    |c + ∑ k, va k * vb k| ≤ |c| + ∑ k, |va k * vb k| :=
  (abs_add_le _ _).trans (add_le_add le_rfl (Finset.abs_sum_le_sum_abs _ _))

private theorem mag_nonneg {K : Nat} (c : ℚ) (va vb : Fin K → ℚ) :
    0 ≤ |c| + ∑ k, |va k * vb k| :=
  add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)

/-- One rounding of `q` is within `rel · mag + abs` when `|q| ≤ mag`, `u ≤ rel`, `η ≤ abs`. -/
private theorem round_le {u eta q e mag rel abs : ℚ} (hu : 0 ≤ u) (hqm : |q| ≤ mag) (hmag : 0 ≤ mag)
    (hrel : u ≤ rel) (habs : eta ≤ abs) (he : e ≤ |q| * u + eta) : e ≤ rel * mag + abs := by
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

/-- **`DotAcc` for a float accumulator**, with at least one rounding: for `K > 0`,
`rel K ≥ u` and `abs K ≥ η` of the accumulator's format; for `K = 0` (the accumulator is the
result) any `rel 0`, `abs 0 ≥ 0`. So `γ_K = K·u / (1 − K·u)` is fine. -/
theorem dotAcc {cfg : DotCfg} {g : TVA.Math.Float.Format} (hg : cfg.acc.format? = some g)
    (dom : ℚ → Prop) {relK absK : ℕ → ℚ} (hrel : ∀ K, 0 < K → g.u ≤ relK K)
    (habs : ∀ K, 0 < K → g.eta ≤ absK K) (hrel0 : 0 ≤ relK 0) (habs0 : 0 ≤ absK 0) :
    Rules.Num.DotAcc (Platform.ref rel abs sp) cfg dom relK absK := by
  intro M N K a b acc r hr i n va vb vc hva hvb _ hvc hroom
  by_cases hK : K = 0
  · subst hK
    rw [hr i n, if_pos rfl]
    exact dot_zero va vb hvc hrel0 habs0
  have hKp : 0 < K := Nat.pos_of_ne_zero hK
  have hex : cfg.exact a b acc i n = vc + ∑ k, va k * vb k := by
    simp only [DotCfg.exact, hva, hvb, hvc, Option.getD_some]
  have hri : r.get2 i n = cfg.acc.ofRat (vc + ∑ k, va k * vb k) := by
    rw [hr i n, if_neg hK, hex]
  have hqm := abs_sum_le vc va vb
  have hmag := mag_nonneg vc va vb
  have hu := TVA.Math.Float.Format.u_nonneg (f := g)
  have he := TVA.Math.Float.Format.eta_nonneg (f := g)
  have hqmax : |vc + ∑ k, va k * vb k| ≤ g.maxVal := by
    rw [← DType.bound_of_format? hg]
    have : 0 ≤ relK K * (|vc| + ∑ k, |va k * vb k|) := mul_nonneg (hu.trans (hrel K hKp)) hmag
    linarith [habs K hKp]
  obtain ⟨v, hv, herr⟩ := DType.val?_ofRat_float hg hqmax
  exact ⟨v, by rw [hri]; exact hv, round_le hu hqm hmag (hrel K hKp) (habs K hKp) herr⟩

/-- **`DotAcc` for integer operands and a signed accumulator**: an exact sum in range is the
result, so any `rel`, `abs ≥ 0` hold. -/
theorem dotAcc_int {cfg : DotCfg} (ha : cfg.a.isInt = true) (hb : cfg.b.isInt = true)
    (hacc : cfg.acc.isSInt = true) (dom : ℚ → Prop) {relK absK : ℕ → ℚ} (hrel : ∀ K, 0 ≤ relK K)
    (habs : ∀ K, 0 ≤ absK K) : Rules.Num.DotAcc (Platform.ref rel abs sp) cfg dom relK absK := by
  intro M N K a b acc r hr i n va vb vc hva hvb _ hvc hroom
  by_cases hK : K = 0
  · subst hK
    rw [hr i n, if_pos rfl]
    exact dot_zero va vb hvc (hrel 0) (habs 0)
  have hex : cfg.exact a b acc i n = vc + ∑ k, va k * vb k := by
    simp only [DotCfg.exact, hva, hvb, hvc, Option.getD_some]
  have hri : r.get2 i n = cfg.acc.ofRat (vc + ∑ k, va k * vb k) := by
    rw [hr i n, if_neg hK, hex]
  choose za hza using fun k => DType.int_of_val? ha (hva k)
  choose zb hzb using fun k => DType.int_of_val? hb (hvb k)
  obtain ⟨zc, hzc⟩ := DType.int_of_val? (DType.isInt_of_isSInt hacc) hvc
  have hq : vc + ∑ k, va k * vb k = ((zc + ∑ k, za k * zb k : ℤ) : ℚ) := by
    push_cast
    simp only [hza, hzb, hzc]
  have hmag := mag_nonneg vc va vb
  have hqmax : |(((zc + ∑ k, za k * zb k : ℤ)) : ℚ)| ≤ cfg.acc.bound := by
    rw [← hq]
    have : 0 ≤ relK K * (|vc| + ∑ k, |va k * vb k|) := mul_nonneg (hrel K) hmag
    linarith [abs_sum_le vc va vb, habs K]
  refine ⟨((zc + ∑ k, za k * zb k : ℤ) : ℚ), ?_, ?_⟩
  · rw [hri, hq]; exact DType.val?_ofRat_sint hacc _ hqmax
  · rw [hq, sub_self, abs_zero]
    exact add_nonneg (mul_nonneg (hrel K) hmag) (habs K)

theorem dotScaledTotal (cfg : ScaledCfg) : Rules.Num.DotScaledTotal (Platform.ref rel abs sp) cfg :=
  fun ⦃_ _ K⦄ a sa b sb acc => ⟨Tensor.of2 fun i n =>
    if K = 0 then acc.get2 i n else FloatV.ofRat (cfg.exact a sa b sb acc i n),
    fun _ _ => Tensor.get2_of2 _ _ _⟩

/-- **`DotScaledAcc`** on any domains, with at least one `float32` rounding: for `K > 0`,
`rel K ≥ u` and `abs K ≥ η`; for `K = 0`, any `rel 0`, `abs 0 ≥ 0`. -/
theorem dotScaledAcc (cfg : ScaledCfg) (dom : ℚ → Prop) (sdom : Nat → Prop) {relK absK : ℕ → ℚ}
    (hrel : ∀ K, 0 < K → TVA.Math.Float.Format.binary32.u ≤ relK K)
    (habs : ∀ K, 0 < K → TVA.Math.Float.Format.binary32.eta ≤ absK K) (hrel0 : 0 ≤ relK 0)
    (habs0 : 0 ≤ absK 0) :
    Rules.Num.DotScaledAcc (Platform.ref rel abs sp) cfg dom sdom relK absK := by
  intro M N K a sa b sb acc r hr i n va vb hva hvb _ _ _ hacc hroom
  by_cases hK : K = 0
  · subst hK
    rw [hr i n, if_pos rfl]
    refine ⟨hacc, ?_⟩
    rw [Fin.sum_univ_zero, Fin.sum_univ_zero, add_zero, add_zero, sub_self, abs_zero]
    exact add_nonneg (mul_nonneg hrel0 (abs_nonneg _)) habs0
  have hKp : 0 < K := Nat.pos_of_ne_zero hK
  have hex : cfg.exact a sa b sb acc i n = (acc.get2 i n).toRat + ∑ k, va k * vb k := by
    simp only [ScaledCfg.exact, hva, hvb, Option.getD_some]
  have hri : r.get2 i n = FloatV.ofRat ((acc.get2 i n).toRat + ∑ k, va k * vb k) := by
    rw [hr i n, if_neg hK, hex]
  have hqm := abs_sum_le (acc.get2 i n).toRat va vb
  have hmag := mag_nonneg (acc.get2 i n).toRat va vb
  have hu := TVA.Math.Float.Format.u_nonneg (f := TVA.Math.Float.Format.binary32)
  have hqmax : |(acc.get2 i n).toRat + ∑ k, va k * vb k| ≤ TVA.Math.Float.Format.binary32.maxVal := by
    have : 0 ≤ relK K * (|(acc.get2 i n).toRat| + ∑ k, |va k * vb k|) :=
      mul_nonneg (hu.trans (hrel K hKp)) hmag
    linarith [habs K hKp, TVA.Math.Float.Format.eta_nonneg (f := TVA.Math.Float.Format.binary32)]
  obtain ⟨hfin, herr⟩ :=
    FloatV.ofPat_roundRat_err TVA.Math.Float.Format.binary32_regular false false hqmax
  rw [hri]
  exact ⟨hfin, round_le hu hqm hmag (hrel K hKp) (habs K hKp) herr⟩

end Platform.ref

end TVA
