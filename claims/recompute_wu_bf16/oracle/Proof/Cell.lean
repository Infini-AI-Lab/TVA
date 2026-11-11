import Claim
import TVA.Proof
import Proof.Lane

/-!
# One output element of a dot, and the cell bounds

`dot_cell`: an element of `tl.dot(a, b, zeros)` on `bfloat16` operands into `float32` is finite and
within the device's bound of the exact sum of products (`Device.DotBF16`). `u_cell`, `w_cell`:
with the lane facts of the operands, the stored `bfloat16` value meets `CellU`, `CellW`'s bound.
-/

namespace Claims.RecomputeWU

open TVA

/-- **An element of a `bfloat16` dot with the zero accumulator** (`Device.DotBF16`): finite, and
within `relDot16 64 · L + absDot16 64` of the exact sum, `L` the sum of the lifted magnitudes of
the products. `ea`, `eb` give the row `i` of the left operand and the column `n` of the right one. -/
theorem dot_cell {P : Platform} (hP : Device P) (p : InputPrecision) {N : Nat}
    {a : Tensor [64, 64] BF16} {b : Tensor [64, N] BF16} {r : Tensor [64, N] F32}
    (hr : P.dot ⟨.bf16, .bf16, .f32, p, 0⟩ a b (tl.zeros [64, N] .f32) r) (i : Fin 64) (n : Fin N)
    (ea eb : Nat → BF16) (ha : ∀ k : Fin 64, a.get2 i k = ea k) (hb : ∀ k : Fin 64, b.get2 k n = eb k)
    (hfa : ∀ k < 64, (ea k).isFinite = true) (hfb : ∀ k < 64, (eb k).isFinite = true)
    (hroom : ∑ k ∈ Finset.range 64, Device.liftMag .bf16 (ea k).toRat *
      Device.liftMag .bf16 (eb k).toRat ≤ 2 ^ 120) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toRat - ∑ k ∈ Finset.range 64, (ea k).toRat * (eb k).toRat| ≤
        Device.relDot16 64 * ∑ k ∈ Finset.range 64, Device.liftMag .bf16 (ea k).toRat *
          Device.liftMag .bf16 (eb k).toRat + Device.absDot16 64 := by
  have hv : ∀ (x : BF16), x.isFinite = true → DType.val? .bf16 x = some x.toRat := by
    intro x hx; show (if x.isFinite = true then some x.toRat else none) = _; simp [hx]
  have hacc := hP.dotBF16 p (M := 64) (N := N) (K := 64) a b (tl.zeros [64, N] .f32) r hr i n
    (fun k => (ea k).toRat) (fun k => (eb k).toRat) 0
    (fun k => by rw [ha k]; exact hv _ (hfa k k.isLt))
    (fun k => by rw [hb k]; exact hv _ (hfb k k.isLt))
    (by
      have h0 : (tl.zeros [64, N] .f32).get2 i n = (FloatV.ofPat 0 : F32) := tl.zeros_get_f32 _
      show DType.val? .f32 ((tl.zeros [64, N] .f32).get2 i n) = some 0
      rw [h0]
      show (if (FloatV.ofPat 0 : F32).isFinite = true then some (FloatV.ofPat 0 : F32).toRat else none) = _
      simp)
    (by
      have hL : ∑ k : Fin 64, Device.liftMag .bf16 (ea k).toRat * Device.liftMag .bf16 (eb k).toRat
          ≤ 2 ^ 120 := by
        rw [Fin.sum_univ_eq_sum_range (fun k => Device.liftMag .bf16 (ea k).toRat *
          Device.liftMag .bf16 (eb k).toRat) 64]
        exact hroom
      have h0 : 0 ≤ ∑ k : Fin 64, Device.liftMag .bf16 (ea k).toRat *
          Device.liftMag .bf16 (eb k).toRat :=
        Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
      simp only [abs_zero, zero_add, DType.bound]
      rw [Format.binary32_maxVal]
      unfold Device.relDot16 Device.absDot16
      generalize ∑ k : Fin 64, Device.liftMag .bf16 (ea k).toRat *
        Device.liftMag .bf16 (eb k).toRat = S at hL h0 ⊢
      norm_num
      linarith)
  obtain ⟨vr, hvr, herr⟩ := hacc
  have hrf : (r.get2 i n).isFinite = true ∧ (r.get2 i n).toRat = vr := by
    have h : (if (r.get2 i n).isFinite = true then some (r.get2 i n).toRat else none) = some vr := hvr
    split_ifs at h with hf
    · exact ⟨hf, Option.some.inj h⟩
  refine ⟨hrf.1, ?_⟩
  rw [hrf.2]
  simp only [abs_zero, zero_add] at herr ⊢
  rw [Fin.sum_univ_eq_sum_range (fun k => (ea k).toRat * (eb k).toRat) 64,
    Fin.sum_univ_eq_sum_range (fun k => Device.liftMag .bf16 (ea k).toRat *
      Device.liftMag .bf16 (eb k).toRat) 64] at herr
  exact herr


/-- The sum of the dot's products, restricted to the enabled columns, equals the sum over all
columns when the disabled ones of the right operand are `0`. -/
theorem sum_en {en : Nat → Bool} {f : Nat → ℚ} (h : ∀ k < 64, en k = false → f k = 0) :
    ∑ k ∈ Finset.range 64, (if en k = true then f k else 0) = ∑ k ∈ Finset.range 64, f k := by
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk' := Finset.mem_range.mp hk
  by_cases he : en k = true
  · simp [he]
  · have : en k = false := by simpa using he
    simp [this, h k hk' this]

/-- **The `u` cell**: for row `i` of `b_A` (entries `ea`) and column `n` of `b_vb` (entries `eb`,
zero on the columns `en` disables), the `bfloat16` rounding `o` of the dot's element is finite and
within `tolU` of `uRef`'s sum, with `uMag`'s magnitude. -/
theorem u_cell {P : Platform} (hP : Device P) (p : InputPrecision) {N : Nat}
    {a : Tensor [64, 64] BF16} {b : Tensor [64, N] BF16} {r : Tensor [64, N] F32}
    (hr : P.dot ⟨.bf16, .bf16, .f32, p, 0⟩ a b (tl.zeros [64, N] .f32) r) (i : Fin 64) (n : Fin N)
    (ea eb : Nat → BF16) (en : Nat → Bool) (ha : ∀ k : Fin 64, a.get2 i k = ea k)
    (hb : ∀ k : Fin 64, b.get2 k n = eb k)
    (hfa : ∀ k < 64, (ea k).isFinite = true) (hAa : ∀ k < 64, |(ea k).toRat| ≤ 2 ^ 64)
    (hfb : ∀ k < 64, (eb k).isFinite = true) (hAb : ∀ k < 64, |(eb k).toRat| ≤ 2 ^ 41)
    (hz : ∀ k < 64, en k = false → (eb k).toRat = 0) {o : BF16}
    (ho : o = FloatV.cvt (r.get2 i n)) :
    o.isFinite = true ∧
      |o.toRat - ∑ k ∈ Finset.range 64, (if en k = true then (ea k).toRat * (eb k).toRat else 0)| ≤
        tolU (∑ k ∈ Finset.range 64, (if en k = true then (ea k).toRat * (eb k).toRat else 0))
          (∑ k ∈ Finset.range 64, (if en k = true then
            Device.liftMag .bf16 (ea k).toRat * Device.liftMag .bf16 (eb k).toRat else 0)) := by
  have hterm : ∀ k < 64, Device.liftMag .bf16 (ea k).toRat * Device.liftMag .bf16 (eb k).toRat
      ≤ 2 ^ 107 := by
    intro k hk
    have h1 := Device.liftMag_le .bf16 (ea k).toRat
    have h2 := Device.liftMag_le .bf16 (eb k).toRat
    have h3 := Device.liftMag_nonneg .bf16 (ea k).toRat
    have h4 := Device.liftMag_nonneg .bf16 (eb k).toRat
    have := hAa k hk
    have := hAb k hk
    rw [Device.liftFloor_bf16] at h1 h2
    calc _ ≤ (2 ^ 65 : ℚ) * 2 ^ 42 :=
          mul_le_mul (h1.trans (by norm_num; linarith)) (h2.trans (by norm_num; linarith)) h4
            (by positivity)
      _ = 2 ^ 107 := by norm_num
  have hroom : ∑ k ∈ Finset.range 64, Device.liftMag .bf16 (ea k).toRat *
      Device.liftMag .bf16 (eb k).toRat ≤ 2 ^ 120 := by
    calc _ ≤ ∑ k ∈ Finset.range 64, (2 ^ 107 : ℚ) :=
          Finset.sum_le_sum fun k hk => hterm k (Finset.mem_range.mp hk)
      _ = 64 * 2 ^ 107 := by simp
      _ ≤ 2 ^ 120 := by norm_num
  obtain ⟨hrf, herr⟩ := dot_cell hP p hr i n ea eb ha hb hfa hfb hroom
  have hS := sum_en (en := en) (f := fun k => (ea k).toRat * (eb k).toRat)
    (fun k hk he => by simp [hz k hk he])
  have hL := sum_en (en := en)
    (f := fun k => Device.liftMag .bf16 (ea k).toRat * Device.liftMag .bf16 (eb k).toRat)
    (fun k hk he => by simp [hz k hk he])
  rw [hS, hL]
  have hL0 : 0 ≤ ∑ k ∈ Finset.range 64, Device.liftMag .bf16 (ea k).toRat *
      Device.liftMag .bf16 (eb k).toRat :=
    Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hSL : |∑ k ∈ Finset.range 64, (ea k).toRat * (eb k).toRat| ≤
      ∑ k ∈ Finset.range 64, Device.liftMag .bf16 (ea k).toRat * Device.liftMag .bf16 (eb k).toRat :=
    (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => by
      rw [abs_mul]
      exact mul_le_mul (Device.abs_le_liftMag _ _) (Device.abs_le_liftMag _ _) (abs_nonneg _)
        (Device.liftMag_nonneg _ _))
  unfold Device.relDot16 Device.absDot16 at herr
  norm_num at herr
  have hrmax : |(r.get2 i n).toRat| ≤ Format.bfloat16.maxVal := by
    have := abs_sub_abs_le_abs_sub (r.get2 i n).toRat
      (∑ k ∈ Finset.range 64, (ea k).toRat * (eb k).toRat)
    rw [Format.bfloat16_maxVal]
    generalize ∑ k ∈ Finset.range 64, Device.liftMag .bf16 (ea k).toRat *
      Device.liftMag .bf16 (eb k).toRat = L at *
    norm_num
    linarith
  obtain ⟨hof, hoe⟩ := FloatV.cvt_err (g := Format.bfloat16) Format.bfloat16_regular hrf hrmax
  rw [← ho] at hof hoe
  rw [Format.bfloat16_u, Format.bfloat16_eta] at hoe
  refine ⟨hof, ?_⟩
  have := u_arith (ref := ∑ k ∈ Finset.range 64, (ea k).toRat * (eb k).toRat) (r := (r.get2 i n).toRat)
    (o := o.toRat) hL0 (by
      generalize ∑ k ∈ Finset.range 64, Device.liftMag .bf16 (ea k).toRat *
        Device.liftMag .bf16 (eb k).toRat = L at *
      norm_num
      linarith) (by norm_num at hoe ⊢; linarith)
  unfold tolU
  norm_num at this ⊢
  linarith


/-- The operand bound of an enabled column of the `w` dot: `eb` is within `α·s + γ·(|x| + 1)` of
`x · G`, `s = |x| G`, for `|x| ≤ 2^40`, `0 < G ≤ 2^8`. -/
def KbNear (eb x G : ℝ) : Prop :=
  |eb - x * G| ≤ (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ)) * (|x| * G) + 2 ^ (-125 : ℤ) * (|x| + 1)

/-- An operand near `x · G` is at most `(1 + α)·|x| G + 2^-125 (|x| + 1)` and `2^49` in magnitude. -/
theorem kbNear_abs {eb x G : ℝ} (h : KbNear eb x G) (hx : |x| ≤ 2 ^ 40) (hG0 : 0 < G)
    (hG : G ≤ 2 ^ 8) :
    |eb| ≤ (1 + (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ))) * (|x| * G) + 2 ^ (-125 : ℤ) * (|x| + 1) ∧
      |eb| ≤ 2 ^ 49 := by
  unfold KbNear at h
  have h1 : |eb| ≤ |x * G| + |eb - x * G| := by
    have := abs_sub_abs_le_abs_sub eb (x * G)
    linarith
  rw [abs_mul, abs_of_pos hG0] at h1
  have hs : |x| * G ≤ 2 ^ 48 := by
    calc |x| * G ≤ 2 ^ 40 * 2 ^ 8 := mul_le_mul hx hG (hG0.le) (by positivity)
      _ = 2 ^ 48 := by norm_num
  have hs0 : 0 ≤ |x| * G := by positivity
  constructor
  · linarith
  · norm_num at h h1 ⊢
    nlinarith [abs_nonneg x]

/-- **The `w` cell**: for row `i` of `b_A` (entries `ea`) and column `n` of `b_kbh` (entries `eb`,
each within `KbNear` of `xs k · Gs k` on the enabled columns, `0` elsewhere), the `bfloat16`
rounding `o` of the dot's element is finite and within `tolW` of `wRef`'s sum. -/
theorem w_cell {P : Platform} (hP : Device P) (p : InputPrecision) {N : Nat}
    {a : Tensor [64, 64] BF16} {b : Tensor [64, N] BF16} {r : Tensor [64, N] F32}
    (hr : P.dot ⟨.bf16, .bf16, .f32, p, 0⟩ a b (tl.zeros [64, N] .f32) r) (i : Fin 64) (n : Fin N)
    (ea eb : Nat → BF16) (en : Nat → Bool) (xs Gs : Nat → ℝ) (ha : ∀ k : Fin 64, a.get2 i k = ea k)
    (hb : ∀ k : Fin 64, b.get2 k n = eb k)
    (hfa : ∀ k < 64, (ea k).isFinite = true) (hAa : ∀ k < 64, |(ea k).toRat| ≤ 2 ^ 64)
    (hfb : ∀ k < 64, (eb k).isFinite = true)
    (hnear : ∀ k < 64, en k = true → KbNear (eb k).toReal (xs k) (Gs k))
    (hx : ∀ k < 64, en k = true → |xs k| ≤ 2 ^ 40)
    (hG : ∀ k < 64, en k = true → 0 < Gs k ∧ Gs k ≤ 2 ^ 8)
    (hz : ∀ k < 64, en k = false → (eb k).toRat = 0) {o : BF16}
    (ho : o = FloatV.cvt (r.get2 i n)) :
    o.isFinite = true ∧
      |o.toReal - ∑ k ∈ Finset.range 64, (if en k = true then (ea k).toReal * xs k * Gs k else 0)| ≤
        tolW (∑ k ∈ Finset.range 64, (if en k = true then
            (Device.liftMag .bf16 (ea k).toRat : ℝ) * |xs k| * Gs k else 0))
          (∑ k ∈ Finset.range 64, (if en k = true then
            (Device.liftMag .bf16 (ea k).toRat : ℝ) * (|xs k| + 1) else 0)) := by
  -- the operands' magnitudes
  have hbr : ∀ k < 64, |(eb k).toReal| ≤ 2 ^ 49 := by
    intro k hk
    by_cases he : en k = true
    · exact (kbNear_abs (hnear k hk he) (hx k hk he) (hG k hk he).1 (hG k hk he).2).2
    · have : en k = false := by simpa using he
      have h0 := hz k hk this
      unfold FloatV.toReal; rw [h0]; norm_num
  have hbq : ∀ k < 64, |(eb k).toRat| ≤ 2 ^ 49 := by
    intro k hk
    have := hbr k hk
    unfold FloatV.toReal at this
    exact_mod_cast this
  have hterm : ∀ k < 64, Device.liftMag .bf16 (ea k).toRat * Device.liftMag .bf16 (eb k).toRat
      ≤ 2 ^ 114 := by
    intro k hk
    have h1 := Device.liftMag_le .bf16 (ea k).toRat
    have h2 := Device.liftMag_le .bf16 (eb k).toRat
    have h4 := Device.liftMag_nonneg .bf16 (eb k).toRat
    have := hAa k hk
    have := hbq k hk
    rw [Device.liftFloor_bf16] at h1 h2
    calc _ ≤ (2 ^ 64 + 1 : ℚ) * (2 ^ 49 + 1) :=
          mul_le_mul (h1.trans (by norm_num; linarith)) (h2.trans (by norm_num; linarith)) h4
            (by positivity)
      _ ≤ 2 ^ 114 := by norm_num
  have hroom : ∑ k ∈ Finset.range 64, Device.liftMag .bf16 (ea k).toRat *
      Device.liftMag .bf16 (eb k).toRat ≤ 2 ^ 120 := by
    calc _ ≤ ∑ k ∈ Finset.range 64, (2 ^ 114 : ℚ) :=
          Finset.sum_le_sum fun k hk => hterm k (Finset.mem_range.mp hk)
      _ = 64 * 2 ^ 114 := by simp
      _ ≤ 2 ^ 120 := by norm_num
  obtain ⟨hrf, herr⟩ := dot_cell hP p hr i n ea eb ha hb hfa hfb hroom
  -- the dot's error in the reals
  have herr' : |(r.get2 i n).toReal - ∑ k ∈ Finset.range 64, (ea k).toReal * (eb k).toReal| ≤
      66 * 2 ^ (-22 : ℤ) * (∑ k ∈ Finset.range 64, (Device.liftMag .bf16 (ea k).toRat : ℝ) *
        (Device.liftMag .bf16 (eb k).toRat : ℝ)) + 66 * 2 ^ (-126 : ℤ) := by
    have := (Rat.cast_le (K := ℝ)).mpr herr
    unfold Device.relDot16 Device.absDot16 at this
    push_cast at this
    unfold FloatV.toReal
    norm_num at this ⊢
    linarith
  -- the sums
  have hla : ∀ k, 0 ≤ (Device.liftMag .bf16 (ea k).toRat : ℝ) := fun k => by
    exact_mod_cast Device.liftMag_nonneg _ _
  have hdom : ∀ k < 64, en k = true → 0 ≤ |xs k| * Gs k := fun k hk he =>
    mul_nonneg (abs_nonneg _) (hG k hk he).1.le
  have hM0 : 0 ≤ ∑ k ∈ Finset.range 64, (if en k = true then
      (Device.liftMag .bf16 (ea k).toRat : ℝ) * |xs k| * Gs k else 0) := by
    refine Finset.sum_nonneg fun k hk => ?_
    split_ifs with he
    · have := hdom k (Finset.mem_range.mp hk) he
      rw [mul_assoc]; exact mul_nonneg (hla k) this
    · exact le_rfl
  have hF0 : 0 ≤ ∑ k ∈ Finset.range 64, (if en k = true then
      (Device.liftMag .bf16 (ea k).toRat : ℝ) * (|xs k| + 1) else 0) := by
    refine Finset.sum_nonneg fun k hk => ?_
    split_ifs with he
    · exact mul_nonneg (hla k) (by positivity)
    · exact le_rfl
  -- the exact sum is bounded by the magnitude
  have hw : |∑ k ∈ Finset.range 64, (if en k = true then (ea k).toReal * xs k * Gs k else 0)| ≤
      ∑ k ∈ Finset.range 64, (if en k = true then
        (Device.liftMag .bf16 (ea k).toRat : ℝ) * |xs k| * Gs k else 0) := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k hk => ?_)
    split_ifs with he
    · have hG' := (hG k (Finset.mem_range.mp hk) he).1
      rw [abs_mul, abs_mul, abs_of_pos hG']
      have : |(ea k).toReal| ≤ (Device.liftMag .bf16 (ea k).toRat : ℝ) := by
        unfold FloatV.toReal
        exact_mod_cast Device.abs_le_liftMag _ _
      exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right this (abs_nonneg _)) hG'.le
    · simp
  -- the dot's sum against the exact sum
  have hS : |∑ k ∈ Finset.range 64, (ea k).toReal * (eb k).toReal -
      ∑ k ∈ Finset.range 64, (if en k = true then (ea k).toReal * xs k * Gs k else 0)| ≤
      (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ)) * (∑ k ∈ Finset.range 64, (if en k = true then
        (Device.liftMag .bf16 (ea k).toRat : ℝ) * |xs k| * Gs k else 0)) +
      2 ^ (-125 : ℤ) * (∑ k ∈ Finset.range 64, (if en k = true then
        (Device.liftMag .bf16 (ea k).toRat : ℝ) * (|xs k| + 1) else 0)) := by
    rw [← Finset.sum_sub_distrib, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k hk => ?_)
    have hk' := Finset.mem_range.mp hk
    split_ifs with he
    · have hn := hnear k hk' he
      unfold KbNear at hn
      have : |(ea k).toReal| ≤ (Device.liftMag .bf16 (ea k).toRat : ℝ) := by
        unfold FloatV.toReal
        exact_mod_cast Device.abs_le_liftMag _ _
      calc |(ea k).toReal * (eb k).toReal - (ea k).toReal * xs k * Gs k|
          = |(ea k).toReal| * |(eb k).toReal - xs k * Gs k| := by
            rw [← abs_mul]; congr 1; ring
        _ ≤ (Device.liftMag .bf16 (ea k).toRat : ℝ) *
            ((2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ)) * (|xs k| * Gs k) + 2 ^ (-125 : ℤ) * (|xs k| + 1)) :=
          mul_le_mul this hn (abs_nonneg _) (hla k)
        _ = _ := by ring
    · have : en k = false := by simpa using he
      have h0 := hz k hk' this
      have : (eb k).toReal = 0 := by unfold FloatV.toReal; rw [h0]; norm_num
      simp [this]
  have hL : ∑ k ∈ Finset.range 64, (Device.liftMag .bf16 (ea k).toRat : ℝ) *
        (Device.liftMag .bf16 (eb k).toRat : ℝ) ≤
      (1 + (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ))) * (∑ k ∈ Finset.range 64, (if en k = true then
        (Device.liftMag .bf16 (ea k).toRat : ℝ) * |xs k| * Gs k else 0)) +
      3 / 2 * 2 ^ (-125 : ℤ) * (∑ k ∈ Finset.range 64, (if en k = true then
        (Device.liftMag .bf16 (ea k).toRat : ℝ) * (|xs k| + 1) else 0)) := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun k hk => ?_
    have hk' := Finset.mem_range.mp hk
    split_ifs with he
    · obtain ⟨hb1, -⟩ := kbNear_abs (hnear k hk' he) (hx k hk' he) (hG k hk' he).1 (hG k hk' he).2
      have hl : (Device.liftMag .bf16 (eb k).toRat : ℝ) ≤
          (1 + (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ))) * (|xs k| * Gs k) +
            3 / 2 * 2 ^ (-125 : ℤ) * (|xs k| + 1) := by
        have h := Device.liftMag_le .bf16 (eb k).toRat
        have h' : (Device.liftMag .bf16 (eb k).toRat : ℝ) ≤ |(eb k).toReal| + 2 ^ (-126 : ℤ) := by
          rw [Device.liftFloor_bf16] at h
          have h2 := (Rat.cast_le (K := ℝ)).mpr h
          push_cast at h2
          unfold FloatV.toReal
          refine h2.trans (le_of_eq ?_)
          norm_num
        have hf : (1 : ℝ) ≤ |xs k| + 1 := by linarith [abs_nonneg (xs k)]
        have e126 : (2 : ℝ) ^ (-126 : ℤ) = 2 ^ (-125 : ℤ) / 2 := by norm_num
        have hc0 : (0 : ℝ) ≤ 2 ^ (-125 : ℤ) := by positivity
        rw [e126] at h'
        generalize (2 : ℝ) ^ (-125 : ℤ) = c at *
        generalize (2 : ℝ) ^ (-7 : ℤ) + 2 ^ (-10 : ℤ) = α at *
        nlinarith [mul_nonneg hc0 (abs_nonneg (xs k))]
      calc (Device.liftMag .bf16 (ea k).toRat : ℝ) * (Device.liftMag .bf16 (eb k).toRat : ℝ)
          ≤ (Device.liftMag .bf16 (ea k).toRat : ℝ) * ((1 + (2 ^ (-7 : ℤ) + 2 ^ (-10 : ℤ))) *
              (|xs k| * Gs k) + 3 / 2 * 2 ^ (-125 : ℤ) * (|xs k| + 1)) :=
            mul_le_mul_of_nonneg_left hl (hla k)
        _ = _ := by ring
    · have : en k = false := by simpa using he
      have h0 := hz k hk' this
      rw [h0]; simp
  have hrmax : |(r.get2 i n).toReal| ≤ (Format.bfloat16.maxVal : ℝ) := by
    have hLq : ∑ k ∈ Finset.range 64, (Device.liftMag .bf16 (ea k).toRat : ℝ) *
        (Device.liftMag .bf16 (eb k).toRat : ℝ) ≤ 2 ^ 120 := by exact_mod_cast hroom
    have hSL : |∑ k ∈ Finset.range 64, (ea k).toReal * (eb k).toReal| ≤
        ∑ k ∈ Finset.range 64, (Device.liftMag .bf16 (ea k).toRat : ℝ) *
          (Device.liftMag .bf16 (eb k).toRat : ℝ) := by
      refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => ?_)
      rw [abs_mul]
      have h1 : |(ea k).toReal| ≤ (Device.liftMag .bf16 (ea k).toRat : ℝ) := by
        unfold FloatV.toReal; exact_mod_cast Device.abs_le_liftMag _ _
      have h2 : |(eb k).toReal| ≤ (Device.liftMag .bf16 (eb k).toRat : ℝ) := by
        unfold FloatV.toReal; exact_mod_cast Device.abs_le_liftMag _ _
      exact mul_le_mul h1 h2 (abs_nonneg _) (hla k)
    have := abs_sub_abs_le_abs_sub (r.get2 i n).toReal
      (∑ k ∈ Finset.range 64, (ea k).toReal * (eb k).toReal)
    rw [Format.bfloat16_maxVal_real]
    generalize ∑ k ∈ Finset.range 64, (Device.liftMag .bf16 (ea k).toRat : ℝ) *
      (Device.liftMag .bf16 (eb k).toRat : ℝ) = L at *
    norm_num at *
    linarith
  obtain ⟨hof, hoe⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hrf hrmax
  rw [← ho] at hof hoe
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at hoe
  refine ⟨hof, ?_⟩
  unfold tolW
  exact w_arith hM0 hF0 hw hS hL herr' hoe

end Claims.RecomputeWU
