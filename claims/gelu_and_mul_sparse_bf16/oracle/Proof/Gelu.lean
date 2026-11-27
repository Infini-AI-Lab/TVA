import Proof.Defs

/-!
# The GELU on the reals

`geluRef x = x · sig (Xof x)` (`1 + tanh z = 2·sig (2z)`); on `[0, ∞)` it lies between `x/2` and `x`
and is `2`-Lipschitz (its derivative is `sig X + x·X'·sig X·(1 − sig X) ≤ 1 + 3·X·sig X·(1 − sig X)
≤ 2`, as `3X ≤ e^X + 2`). `sig_chain` bounds the kernel's `1 / (1 + exp(−x2))` against `sig X` when
`x2` is close to `X` and `exp` is within `Device.ExpF32`'s bound.
-/

namespace Claims.GeluAndMulSparse

open TVA

/-- The logistic function is positive. -/
theorem sig_pos (z : ℝ) : 0 < sig z := by
  unfold sig; have := Real.exp_pos (-z); positivity

/-- The logistic function is at most `1`. -/
theorem sig_le_one (z : ℝ) : sig z ≤ 1 := by
  unfold sig
  have := Real.exp_pos (-z)
  rw [div_le_one (by linarith)]; linarith

/-- The logistic function is at least `1/2` on `[0, ∞)`. -/
theorem sig_ge_half {z : ℝ} (hz : 0 ≤ z) : 1 / 2 ≤ sig z := by
  unfold sig
  have h1 : Real.exp (-z) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have := Real.exp_pos (-z)
  rw [div_le_div_iff₀ (by norm_num) (by linarith)]; linarith

/-- The logistic's argument is nonnegative on `[0, ∞)`. -/
theorem Xof_nonneg {x : ℝ} (hx : 0 ≤ x) : 0 ≤ Xof x := by
  unfold Xof kR aR; positivity

/-- `geluRef x = x · sig (Xof x)`: `1 + tanh z = 2 / (1 + e^(-2z))`. -/
theorem geluRef_eq (x : ℝ) : geluRef x = x * sig (Xof x) := by
  unfold geluRef sig Xof
  set z := kR * (x + aR * x ^ 3) with hz
  have hk : (0.7978845608028654 : ℝ) * (x + 0.044715 * x ^ 3) = z := by rw [hz]; unfold kR aR; rfl
  rw [hk, Real.tanh_eq]
  have h2 : -(2 * kR * (x + aR * x ^ 3)) = -z + -z := by rw [hz]; ring
  rw [h2, Real.exp_add]
  have e1 : Real.exp z * Real.exp (-z) = 1 := by rw [← Real.exp_add]; simp
  have p1 := Real.exp_pos z
  have p2 := Real.exp_pos (-z)
  have hne : Real.exp z + Real.exp (-z) ≠ 0 := by positivity
  have hne2 : 1 + Real.exp (-z) * Real.exp (-z) ≠ 0 := by positivity
  field_simp
  linear_combination (2 * x * Real.exp (-z)) * e1

/-- `geluRef 0 = 0`. -/
theorem geluRef_zero : geluRef 0 = 0 := by rw [geluRef_eq]; ring

/-- `geluRef` is nonnegative on `[0, ∞)`. -/
theorem geluRef_nonneg {x : ℝ} (hx : 0 ≤ x) : 0 ≤ geluRef x := by
  rw [geluRef_eq]; exact mul_nonneg hx (sig_pos _).le

/-- `x / 2 ≤ geluRef x` on `[0, ∞)`. -/
theorem geluRef_ge {x : ℝ} (hx : 0 ≤ x) : x / 2 ≤ geluRef x := by
  rw [geluRef_eq]; have := sig_ge_half (Xof_nonneg hx); nlinarith

/-- `geluRef x ≤ x` on `[0, ∞)`. -/
theorem geluRef_le {x : ℝ} (hx : 0 ≤ x) : geluRef x ≤ x := by
  rw [geluRef_eq]; have := sig_le_one (Xof x); nlinarith

/-- The derivative of the logistic function: `e^(-z) / (1 + e^(-z))²`. -/
theorem hasDerivAt_sig (z : ℝ) :
    HasDerivAt sig (Real.exp (-z) / (1 + Real.exp (-z)) ^ 2) z := by
  have hE : HasDerivAt (fun w => 1 + Real.exp (-w)) (Real.exp (-z) * (-1)) z :=
    ((Real.hasDerivAt_exp (-z)).comp z (hasDerivAt_neg z)).const_add 1
  have hne : (1 + Real.exp (-z)) ≠ 0 := by have := Real.exp_pos (-z); positivity
  have h := hE.inv hne
  have hs : sig = (fun w => 1 + Real.exp (-w))⁻¹ := by
    funext w; simp [sig, one_div]
  rw [hs]
  convert h using 1
  ring

/-- The derivative of `Xof`: `2k·(1 + 3a·x²)`. -/
theorem hasDerivAt_Xof (x : ℝ) : HasDerivAt Xof (2 * kR * (1 + aR * (3 * x ^ 2))) x := by
  have h1 : HasDerivAt (fun y : ℝ => y ^ 3) (3 * x ^ 2) x := by
    simpa using hasDerivAt_pow 3 x
  have h2 := ((hasDerivAt_id x).add (h1.const_mul aR)).const_mul (2 * kR)
  unfold Xof
  convert h2 using 1
  simp

/-- The derivative of `x · sig (Xof x)`. -/
theorem hasDerivAt_gelu (x : ℝ) :
    HasDerivAt (fun y => y * sig (Xof y))
      (sig (Xof x) + x * ((Real.exp (-Xof x) / (1 + Real.exp (-Xof x)) ^ 2) *
        (2 * kR * (1 + aR * (3 * x ^ 2))))) x := by
  have h := (hasDerivAt_id x).mul ((hasDerivAt_sig (Xof x)).comp x (hasDerivAt_Xof x))
  convert h using 1
  · funext y; simp [Function.comp_def]
  · simp

/-- `3X·e^(-X)/(1 + e^(-X))² ≤ 1` for `X ≥ 0` (from `3X ≤ e^X + 2`). -/
theorem three_X_dsig_le {X : ℝ} (hX : 0 ≤ X) :
    3 * X * (Real.exp (-X) / (1 + Real.exp (-X)) ^ 2) ≤ 1 := by
  have hq := Real.quadratic_le_exp_of_nonneg hX
  set e := Real.exp X with he
  have he0 : 0 < e := Real.exp_pos X
  have hinv : Real.exp (-X) = e⁻¹ := by rw [Real.exp_neg]
  rw [hinv]
  have h3 : 3 * X ≤ e + 2 := by nlinarith [sq_nonneg (X - 2)]
  have hd : 0 < (1 + e⁻¹) ^ 2 := by positivity
  rw [mul_div_assoc', div_le_one hd]
  have : (1 + e⁻¹) ^ 2 = (e + 1) ^ 2 * e⁻¹ * e⁻¹ := by field_simp
  rw [this]
  have hi : 0 < e⁻¹ := inv_pos.mpr he0
  have : 3 * X * e⁻¹ ≤ (e + 1) ^ 2 * e⁻¹ * e⁻¹ := by
    have : (e + 1) ^ 2 * e⁻¹ = e + 2 + e⁻¹ := by field_simp; ring
    rw [this]
    have : 3 * X ≤ e + 2 + e⁻¹ := by linarith
    nlinarith
  exact this

/-- The derivative of `x · sig (Xof x)` lies in `[0, 2]` on `[0, ∞)`. -/
theorem gelu_deriv_bounds {x : ℝ} (hx : 0 ≤ x) :
    0 ≤ sig (Xof x) + x * ((Real.exp (-Xof x) / (1 + Real.exp (-Xof x)) ^ 2) *
        (2 * kR * (1 + aR * (3 * x ^ 2)))) ∧
      sig (Xof x) + x * ((Real.exp (-Xof x) / (1 + Real.exp (-Xof x)) ^ 2) *
        (2 * kR * (1 + aR * (3 * x ^ 2)))) ≤ 2 := by
  have hX := Xof_nonneg hx
  have hs0 := sig_pos (Xof x)
  have hs1 := sig_le_one (Xof x)
  set q := Real.exp (-Xof x) / (1 + Real.exp (-Xof x)) ^ 2 with hq
  have hq0 : 0 ≤ q := by rw [hq]; have := Real.exp_pos (-Xof x); positivity
  have hk : (0 : ℝ) ≤ kR := by unfold kR; norm_num
  have ha : (0 : ℝ) ≤ aR := by unfold aR; norm_num
  have hxX : x * (2 * kR * (1 + aR * (3 * x ^ 2))) ≤ 3 * Xof x := by
    unfold Xof
    have : 0 ≤ 2 * kR * x := by positivity
    nlinarith [mul_nonneg this (by positivity : (0:ℝ) ≤ aR * x ^ 2)]
  have h3 := three_X_dsig_le hX
  rw [← hq] at h3
  have hpos : 0 ≤ x * (2 * kR * (1 + aR * (3 * x ^ 2))) := by positivity
  constructor
  · have : 0 ≤ x * (q * (2 * kR * (1 + aR * (3 * x ^ 2)))) := by positivity
    linarith
  · have : x * (q * (2 * kR * (1 + aR * (3 * x ^ 2)))) ≤ 1 := by
      have e : x * (q * (2 * kR * (1 + aR * (3 * x ^ 2)))) =
          q * (x * (2 * kR * (1 + aR * (3 * x ^ 2)))) := by ring
      rw [e]
      calc q * (x * (2 * kR * (1 + aR * (3 * x ^ 2)))) ≤ q * (3 * Xof x) :=
            mul_le_mul_of_nonneg_left hxX hq0
        _ = 3 * Xof x * q := by ring
        _ ≤ 1 := h3
    linarith

/-- `geluRef` is monotone on `[0, ∞)`, and so is `x ↦ 2x − geluRef x`. -/
theorem gelu_mono :
    MonotoneOn (fun y => y * sig (Xof y)) (Set.Ici 0) ∧
      MonotoneOn (fun y => 2 * y - y * sig (Xof y)) (Set.Ici 0) := by
  have hd : ∀ x, DifferentiableAt ℝ (fun y => y * sig (Xof y)) x :=
    fun x => (hasDerivAt_gelu x).differentiableAt
  have hc : Continuous (fun y => y * sig (Xof y)) := continuous_iff_continuousAt.mpr
    fun x => (hd x).continuousAt
  constructor
  · refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 0) hc.continuousOn
      (fun x _ => (hasDerivAt_gelu x).hasDerivWithinAt) fun x hx => ?_
    rw [interior_Ici] at hx
    exact (gelu_deriv_bounds (le_of_lt hx)).1
  · refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 0)
      ((continuous_const.mul continuous_id).sub hc).continuousOn
      (fun x _ => (((hasDerivAt_id x).const_mul 2).sub (hasDerivAt_gelu x)).hasDerivWithinAt)
      fun x hx => ?_
    rw [interior_Ici] at hx
    have := (gelu_deriv_bounds (le_of_lt hx)).2
    simp only [mul_one]
    linarith

/-- `geluRef` is `2`-Lipschitz on `[0, ∞)`. -/
theorem geluRef_lip {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    |geluRef x - geluRef y| ≤ 2 * |x - y| := by
  rw [geluRef_eq, geluRef_eq]
  obtain ⟨h1, h2⟩ := gelu_mono
  have key : ∀ a b : ℝ, 0 ≤ a → a ≤ b →
      |a * sig (Xof a) - b * sig (Xof b)| ≤ 2 * |a - b| := by
    intro a b ha hab
    have m1 := h1 (Set.mem_Ici.mpr ha) (Set.mem_Ici.mpr (ha.trans hab)) hab
    have m2 := h2 (Set.mem_Ici.mpr ha) (Set.mem_Ici.mpr (ha.trans hab)) hab
    simp only at m1 m2
    rw [abs_sub_comm, abs_of_nonneg (by linarith), abs_sub_comm, abs_of_nonneg (by linarith)]
    linarith
  rcases le_total x y with h | h
  · exact key x y hx h
  · rw [abs_sub_comm, abs_sub_comm x]; exact key y x hy h

/-- `e^b − e^a ≤ (b − a)·e^b`. -/
theorem exp_sub_exp_le (a b : ℝ) : Real.exp b - Real.exp a ≤ (b - a) * Real.exp b := by
  have h1 : Real.exp a = Real.exp b * Real.exp (a - b) := by rw [← Real.exp_add]; ring_nf
  have h2 := Real.add_one_le_exp (a - b)
  have h3 := Real.exp_pos b
  rw [h1]; nlinarith

/-- `y·e^(-y) ≤ 1`. -/
theorem mul_exp_neg_le (y : ℝ) : y * Real.exp (-y) ≤ 1 := by
  have h1 := Real.add_one_le_exp y
  have h2 : Real.exp y * Real.exp (-y) = 1 := by rw [← Real.exp_add]; simp
  have h3 := Real.exp_pos (-y)
  nlinarith

/-- `|x|·e^x ≤ 3` and `e^x ≤ 3` for `x ≤ 1`. -/
theorem abs_mul_exp_le {x : ℝ} (hx : x ≤ 1) : |x| * Real.exp x ≤ 3 ∧ Real.exp x ≤ 3 := by
  have he1 : Real.exp x ≤ Real.exp 1 := Real.exp_le_exp.mpr hx
  have he3 : Real.exp 1 < 3 := by have := Real.exp_one_lt_d9; norm_num at this; linarith
  have hp := Real.exp_pos x
  refine ⟨?_, by linarith⟩
  rcases le_total 0 x with h | h
  · rw [abs_of_nonneg h]; nlinarith
  · rw [abs_of_nonpos h]
    have := mul_exp_neg_le (-x)
    rw [neg_neg] at this; linarith

/-- The room `Device.ExpF32` asks for, at `X = |x|`, for every `x ≤ 1`. -/
theorem exp_room {x : ℝ} (hx : x ≤ 1) :
    (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |x|)) * |Real.exp x| + 2 ^ (-140 : ℤ) ≤ 2 ^ 10 := by
  obtain ⟨h1, h2⟩ := abs_mul_exp_le hx
  rw [abs_of_pos (Real.exp_pos x)]
  have e : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |x|)) * Real.exp x =
      (1 + 2 ^ (-22 : ℤ)) * Real.exp x + 2 ^ (-23 : ℤ) * (|x| * Real.exp x) := by ring
  rw [e]
  have h3 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * Real.exp x ≤ (1 + 2 ^ (-22 : ℤ)) * 3 :=
    mul_le_mul_of_nonneg_left h2 (by positivity)
  have h4 : (2 : ℝ) ^ (-23 : ℤ) * (|x| * Real.exp x) ≤ 2 ^ (-23 : ℤ) * 3 :=
    mul_le_mul_of_nonneg_left h1 (by positivity)
  have h5 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * 3 + 2 ^ (-23 : ℤ) * 3 + 2 ^ (-140 : ℤ) ≤ 2 ^ 10 := by
    norm_num
  linarith

/-- `|e^(-X) − e^(-Xk)|` for `Xk` within `δX + τ` of `X ≥ 0` (`δ ≤ 1/2`, `e^τ ≤ 1 + 2τ`). -/
theorem exp_neg_diff {δ τ X Xk : ℝ} (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1 / 2) (hτ0 : 0 ≤ τ)
    (hexpτ : Real.exp τ ≤ 1 + 2 * τ) (hX : 0 ≤ X) (hXk : |Xk - X| ≤ δ * X + τ) :
    |Real.exp (-X) - Real.exp (-Xk)| ≤ δ * (1 + 2 * δ) * (1 + 2 * τ) + τ * (1 + 2 * τ) := by
  have hab := abs_le.mp hXk
  have hE0 := Real.exp_pos (-X)
  have hEk0 := Real.exp_pos (-Xk)
  have hpos1 : 0 ≤ δ * (2 * δ + 2 * τ + 4 * δ * τ) + 2 * τ * τ := by positivity
  have hge : δ + τ ≤ δ * (1 + 2 * δ) * (1 + 2 * τ) + τ * (1 + 2 * τ) := by
    have e : δ * (1 + 2 * δ) * (1 + 2 * τ) + τ * (1 + 2 * τ) - (δ + τ) =
        δ * (2 * δ + 2 * τ + 4 * δ * τ) + 2 * τ * τ := by ring
    linarith
  rcases le_total X Xk with h | h
  · have h1 := exp_sub_exp_le (-Xk) (-X)
    have hle : Real.exp (-Xk) ≤ Real.exp (-X) := Real.exp_le_exp.mpr (by linarith)
    rw [abs_of_nonneg (by linarith)]
    have hXE := mul_exp_neg_le X
    have h2 : (-X - -Xk) * Real.exp (-X) ≤ (δ * X + τ) * Real.exp (-X) :=
      mul_le_mul_of_nonneg_right (by linarith) hE0.le
    have hE1 : Real.exp (-X) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have h3 : (δ * X + τ) * Real.exp (-X) = δ * (X * Real.exp (-X)) + τ * Real.exp (-X) := by ring
    have h4 : δ * (X * Real.exp (-X)) ≤ δ := by
      calc δ * (X * Real.exp (-X)) ≤ δ * 1 := mul_le_mul_of_nonneg_left hXE hδ0
        _ = δ := by ring
    have h5 : τ * Real.exp (-X) ≤ τ := by
      calc τ * Real.exp (-X) ≤ τ * 1 := mul_le_mul_of_nonneg_left hE1 hτ0
        _ = τ := by ring
    linarith
  · have h1 := exp_sub_exp_le (-X) (-Xk)
    have hle : Real.exp (-X) ≤ Real.exp (-Xk) := Real.exp_le_exp.mpr (by linarith)
    rw [abs_sub_comm, abs_of_nonneg (by linarith)]
    have hc : 0 ≤ (1 - δ) * X := mul_nonneg (by linarith) hX
    have hEk' : Real.exp (-Xk) ≤ Real.exp (-((1 - δ) * X)) * Real.exp τ := by
      rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith)
    have hm := mul_exp_neg_le ((1 - δ) * X)
    have hf0 : 0 < Real.exp (-((1 - δ) * X)) := Real.exp_pos _
    have hτe : 0 < Real.exp τ := Real.exp_pos _
    have hEk1 : Real.exp (-Xk) ≤ 1 + 2 * τ :=
      (Real.exp_le_exp.mpr (by linarith)).trans hexpτ
    have hxf : X * Real.exp (-((1 - δ) * X)) ≤ 1 + 2 * δ := by
      have h1d : 1 ≤ (1 - δ) * (1 + 2 * δ) := by nlinarith
      have hpos : 0 ≤ X * Real.exp (-((1 - δ) * X)) := by positivity
      have e : (1 - δ) * X * Real.exp (-((1 - δ) * X)) =
          (X * Real.exp (-((1 - δ) * X))) * (1 - δ) := by ring
      rw [e] at hm
      nlinarith
    have hk : δ * X * Real.exp (-((1 - δ) * X)) ≤ δ * (1 + 2 * δ) := by
      calc δ * X * Real.exp (-((1 - δ) * X)) = δ * (X * Real.exp (-((1 - δ) * X))) := by ring
        _ ≤ δ * (1 + 2 * δ) := mul_le_mul_of_nonneg_left hxf hδ0
    have hδX : 0 ≤ δ * X := mul_nonneg hδ0 hX
    calc Real.exp (-Xk) - Real.exp (-X) ≤ (-Xk - -X) * Real.exp (-Xk) := h1
      _ ≤ (δ * X + τ) * Real.exp (-Xk) := mul_le_mul_of_nonneg_right (by linarith) hEk0.le
      _ = δ * X * Real.exp (-Xk) + τ * Real.exp (-Xk) := by ring
      _ ≤ δ * X * (Real.exp (-((1 - δ) * X)) * Real.exp τ) + τ * (1 + 2 * τ) := by
          gcongr
      _ = δ * X * Real.exp (-((1 - δ) * X)) * Real.exp τ + τ * (1 + 2 * τ) := by ring
      _ ≤ δ * (1 + 2 * δ) * (1 + 2 * τ) + τ * (1 + 2 * τ) := by
          gcongr


/-- **The kernel's logistic** `1 / (1 + ex)` against `sig X`, when `Xk` is within `2^-20·X + 2^-120`
of `X ≥ 0` and `ex` within `Device.ExpF32`'s bound of `e^(-Xk)`. -/
theorem sig_chain {X Xk ex : ℝ} (hX : 0 ≤ X) (hXk : |Xk - X| ≤ 2 ^ (-20 : ℤ) * X + 2 ^ (-120 : ℤ))
    (hex : |ex - Real.exp (-Xk)| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |Xk|) * Real.exp (-Xk) + 2 ^ (-140 : ℤ)) :
    1 / 2 ≤ 1 + ex ∧ 1 + ex ≤ 3 ∧ |1 / (1 + ex) - sig X| ≤ 2 ^ (-19 : ℤ) := by
  set δ : ℝ := 2 ^ (-20 : ℤ) with hδ
  set τ : ℝ := 2 ^ (-120 : ℤ) with hτ
  have hδv : δ = 1 / 1048576 := by rw [hδ]; norm_num
  have hτ0 : 0 < τ := by positivity
  have hτ1 : τ ≤ 1 / 1099511627776 := by rw [hτ]; norm_num
  set E := Real.exp (-X) with hE
  set Ek := Real.exp (-Xk) with hEk
  have hE0 : 0 < E := Real.exp_pos _
  have hEk0 : 0 < Ek := Real.exp_pos _
  have hE1 : E ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hsig : sig X = 1 / (1 + E) := by rw [hE]; rfl
  clear_value δ τ
  have hab := abs_le.mp hXk
  have hδX : δ * X ≤ X := by rw [hδv]; linarith
  have hXklo : -τ ≤ Xk := by linarith [hab.1]
  -- `e^τ ≤ 1 + 2τ`
  have hexpτ : Real.exp τ ≤ 1 + 2 * τ := by
    have := Real.abs_exp_sub_one_le (x := τ) (by rw [abs_of_pos hτ0]; linarith)
    rw [abs_of_pos hτ0] at this
    linarith [(abs_le.mp this).2]
  have hEk1 : Ek ≤ 1 + 2 * τ := (Real.exp_le_exp.mpr (by linarith)).trans hexpτ
  -- `|Xk|·Ek ≤ 1`
  have hXkE : |Xk| * Ek ≤ 1 := by
    rcases le_total 0 Xk with h | h
    · rw [abs_of_nonneg h]; exact mul_exp_neg_le _
    · rw [abs_of_nonpos h]
      have : -Xk ≤ τ := by linarith
      nlinarith
  have hAk0 : 0 ≤ |Xk| * Ek := by positivity
  have hex' := abs_le.mp hex
  have hsplit : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |Xk|) * Ek =
      2 ^ (-22 : ℤ) * Ek + 2 ^ (-23 : ℤ) * (|Xk| * Ek) := by ring
  have hc22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  have hc23 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
  have hc140 : (2 : ℝ) ^ (-140 : ℤ) ≤ 1 / 1099511627776 := by norm_num
  rw [hsplit, hc22, hc23] at hex'
  -- `|Ek − ex|`
  have hd1 : |ex - Ek| ≤ Ek / 4194304 + 1 / 8388608 + 1 / 1099511627776 := by
    refine abs_le.mpr ⟨?_, ?_⟩ <;> nlinarith
  have hex_lo : -(1 / 8388608) - 1 / 1099511627776 ≤ ex := by
    have := (abs_le.mp hd1).1; nlinarith
  have hex_hi : ex ≤ 2 := by
    have := (abs_le.mp hd1).2; nlinarith
  -- `|E − Ek|`
  have hd2 : |E - Ek| ≤ δ * (1 + 2 * δ) * (1 + 2 * τ) + τ * (1 + 2 * τ) :=
    exp_neg_diff (by rw [hδv]; norm_num) (by rw [hδv]; norm_num) hτ0.le hexpτ hX hXk
  have hden : 0 < 1 + ex := by nlinarith
  refine ⟨by nlinarith, by linarith, ?_⟩
  rw [hsig]
  have e : 1 / (1 + ex) - 1 / (1 + E) = (E - ex) / ((1 + ex) * (1 + E)) := by
    field_simp; ring
  have hE0' : 0 < 1 + E := by linarith
  have hpp : 0 < (1 + ex) * (1 + E) := mul_pos hden hE0'
  rw [e, abs_div, abs_of_pos hpp, div_le_iff₀ hpp]
  have htri : |E - ex| ≤ |E - Ek| + |ex - Ek| := by
    have : E - ex = (E - Ek) - (ex - Ek) := by ring
    rw [this]; exact abs_sub _ _
  have hprod : 1 + ex ≤ (1 + ex) * (1 + E) := by
    have : 0 ≤ (1 + ex) * E := mul_nonneg hden.le hE0.le
    linarith [show (1 + ex) * (1 + E) = (1 + ex) + (1 + ex) * E by ring]
  have h19 : (2 : ℝ) ^ (-19 : ℤ) = 1 / 524288 := by norm_num
  have hτ2 : τ * τ ≤ τ * (1 / 1099511627776) := mul_le_mul_of_nonneg_left hτ1 hτ0.le
  have hB : δ * (1 + 2 * δ) * (1 + 2 * τ) + τ * (1 + 2 * τ) +
      ((1 + 2 * τ) / 4194304 + 1 / 8388608 + 1 / 1099511627776) ≤
      1 / 524288 * (1 - 1 / 8388608 - 1 / 1099511627776) := by
    rw [hδv]; linarith
  have hd1' : |ex - Ek| ≤ (1 + 2 * τ) / 4194304 + 1 / 8388608 + 1 / 1099511627776 := by
    have : Ek / 4194304 ≤ (1 + 2 * τ) / 4194304 := by gcongr
    linarith
  rw [h19]
  calc |E - ex| ≤ |E - Ek| + |ex - Ek| := htri
    _ ≤ 1 / 524288 * (1 - 1 / 8388608 - 1 / 1099511627776) := by linarith
    _ ≤ 1 / 524288 * (1 + ex) := by linarith
    _ ≤ 1 / 524288 * ((1 + ex) * (1 + E)) := by linarith

end Claims.GeluAndMulSparse
