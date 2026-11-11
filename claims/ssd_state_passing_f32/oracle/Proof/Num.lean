import Claim
import TVA.Proof

/-!
# The numerics of one chunk

One trip of the chunk loop computes `scale · state + new_states` with `scale = tl.exp(dA_cs)`.
This file bounds it from the device fact for `tl.exp` (`Device.ExpF32`) and the IEEE
operations, with the constants of the claim's tolerance `tol`.
-/
namespace Claims.SsdStatePassing

open TVA

/-- `u` of `float32` in `ℝ`. -/
theorem u32 : ((Format.binary32.u : ℚ) : ℝ) = 2 ^ (-24 : ℤ) := Format.binary32_u_real
/-- `η` of `float32` in `ℝ`. -/
theorem eta32 : ((Format.binary32.eta : ℚ) : ℝ) = 2 ^ (-150 : ℤ) := Format.binary32_eta_real

/-- `2^127` is below the largest `float32`. -/
theorem two127_le_max : (2 : ℝ) ^ 127 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
  rw [Format.binary32_maxVal_real]
  have h : (2 : ℝ) ^ 127 = 2 ^ 23 * 2 ^ 104 := by rw [← pow_add]
  rw [h]
  have : (2 : ℝ) ^ 23 ≤ 2 ^ 24 - 1 := by norm_num
  have h0 : (0 : ℝ) < 2 ^ 104 := by norm_num
  nlinarith

/-- **`scale * state + new_states`**: with finite operands whose product is at most `M` and
`M + |n| ≤ 2^126`, the result (two roundings, or one `fma`) is finite and within
`rho·(M + |n|) + 3·2^-150` of the exact `s·v + n`. -/
theorem muladd_step {fusion : Bool} {s v n r : F32} (hs : s.isFinite = true)
    (hv : v.isFinite = true) (hn : n.isFinite = true)
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul s v) n) r ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma s v n) r))
    {M : ℝ} (hM : |s.toReal * v.toReal| ≤ M) (hroom : M + |n.toReal| ≤ 2 ^ 126) :
    r.isFinite = true ∧
      |r.toReal - (s.toReal * v.toReal + n.toReal)| ≤ rho * (M + |n.toReal|) + 3 * 2 ^ (-150 : ℤ) := by
  have hreg := Format.binary32_regular
  have hmax := two127_le_max
  have hu : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
  have he : (2 : ℝ) ^ (-150 : ℤ) ≥ 0 := by positivity
  set a := s.toReal * v.toReal with ha
  set N := n.toReal with hN
  have haM : |a| ≤ 2 ^ 126 := by
    have := abs_nonneg N; linarith
  have hrho : rho = 2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ) := rfl
  have hMn : |a| + |N| ≤ M + |N| := by linarith
  have h0a := abs_nonneg a
  have h0N := abs_nonneg N
  have h126 : (2 : ℝ) ^ 126 ≤ 2 ^ 127 / 2 := by norm_num
  rcases h with hne | ⟨-, hne⟩
  · -- two roundings
    obtain ⟨hpf, hpe⟩ := FloatV.mul_err_real hreg hs hv (by linarith)
    rw [u32, eta32] at hpe
    set p := (FloatV.mul s v).toReal with hp
    have hpabs : |p| ≤ |a| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) + |a| := by
      have := abs_sub_abs_le_abs_sub p a
      have h2 := abs_sub_comm p a
      linarith
    have hpN : |p + N| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      have := abs_add_le p N
      have hh : |a| * 2 ^ (-24 : ℤ) ≤ 2 ^ 102 := by
        have := mul_le_mul_of_nonneg_right haM (show (0:ℝ) ≤ 2 ^ (-24 : ℤ) by positivity)
        have e : (2:ℝ) ^ 126 * 2 ^ (-24 : ℤ) = 2 ^ 102 := by norm_num
        linarith
      have h1 : (2:ℝ) ^ (-150 : ℤ) ≤ 1 := by norm_num
      have h2 : (2:ℝ) ^ 126 + 2 ^ 102 + 1 ≤ 2 ^ 127 := by norm_num
      have h3 : |N| + |a| ≤ 2 ^ 126 := by linarith
      linarith
    obtain ⟨hrf, hre⟩ := FloatV.add_err_real hreg hpf hn hpN
    rw [u32, eta32] at hre
    have hr := hne.exact_of_isFinite hrf
    refine ⟨hr.1, ?_⟩
    rw [hr.2.2]
    have e1 : (FloatV.add (FloatV.mul s v) n).toReal - (a + N) =
        ((FloatV.add (FloatV.mul s v) n).toReal - (p + N)) + (p - a) := by ring
    rw [e1]
    have t := abs_add_le ((FloatV.add (FloatV.mul s v) n).toReal - (p + N)) (p - a)
    have hpN' : |p + N| ≤ (1 + 2 ^ (-24 : ℤ)) * |a| + 2 ^ (-150 : ℤ) + |N| := by
      have := abs_add_le p N
      nlinarith
    have hb : |p + N| * 2 ^ (-24 : ℤ) ≤
        ((1 + 2 ^ (-24 : ℤ)) * |a| + 2 ^ (-150 : ℤ) + |N|) * 2 ^ (-24 : ℤ) :=
      mul_le_mul_of_nonneg_right hpN' (by positivity)
    rw [hrho]
    have k1 : (2 : ℝ) ^ (-24 : ℤ) * 2 ^ (-24 : ℤ) ≤ 2 ^ (-47 : ℤ) := by norm_num
    have k2 : (2 : ℝ) ^ (-24 : ℤ) * 2 ^ (-150 : ℤ) ≤ 2 ^ (-150 : ℤ) := by norm_num
    have k3 : (2 : ℝ) ^ (-24 : ℤ) ≤ 2 ^ (-23 : ℤ) := by norm_num
    have k4 : (2 : ℝ) ^ (-24 : ℤ) * 2 = 2 ^ (-23 : ℤ) := by norm_num
    nlinarith [mul_nonneg h0a h0N, mul_le_mul_of_nonneg_left k1 h0a,
      mul_le_mul_of_nonneg_left k3 h0N, mul_le_mul_of_nonneg_left hMn (show (0:ℝ) ≤ 2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ) by positivity)]
  · -- one fma
    obtain ⟨hrf, hre⟩ := FloatV.fma_err_real hreg hs hv hn (by
      have := abs_add_le a N; linarith)
    rw [u32, eta32] at hre
    have hr := hne.exact_of_isFinite hrf
    refine ⟨hr.1, ?_⟩
    rw [hr.2.2]
    refine hre.trans ?_
    have hsum := abs_add_le a N
    rw [hrho]
    have k3 : (2 : ℝ) ^ (-24 : ℤ) ≤ 2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ) := by norm_num
    have k5 : (2 : ℝ) ^ (-150 : ℤ) ≤ 3 * 2 ^ (-150 : ℤ) := by linarith
    nlinarith [mul_le_mul_of_nonneg_left (show |a + N| ≤ M + |N| by linarith)
      (show (0:ℝ) ≤ 2 ^ (-24 : ℤ) by positivity),
      mul_le_mul_of_nonneg_right k3 (show (0:ℝ) ≤ M + |N| by linarith)]


/-! ## The decay -/

theorem z22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
/-- `2^-23` as a fraction. -/
theorem z23 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
/-- `2^-24` as a fraction. -/
theorem z24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
/-- `2^-150` as a fraction. -/
theorem z150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
/-- `2^-140` as a fraction. -/
theorem z140 : (2 : ℝ) ^ (-140 : ℤ) = 1 / 2 ^ 140 := by norm_num

/-- `exp 65 ≤ 2^105`. -/
theorem exp65_le : Real.exp 65 ≤ 2 ^ 105 := by
  have h1 : Real.exp 1 < 3 := lt_trans Real.exp_one_lt_d9 (by norm_num)
  have h2 : Real.exp 65 = Real.exp 1 ^ 65 := by
    rw [← Real.exp_nat_mul]; norm_num
  rw [h2]
  calc Real.exp 1 ^ 65 ≤ 3 ^ 65 := pow_le_pow_left₀ (Real.exp_pos 1).le h1.le 65
    _ ≤ 2 ^ 105 := by norm_num

/-- **Room for `tl.exp`**: for an exponent at most `64 + δ`, `(1 + rel)·e^x + 2^-140` stays below
`2^127`, where `rel` is the device fact's relative constant at `|D| + δ`. -/
theorem exp_room {D δ x : ℝ} (hD : D ≤ 64)
    (hδ : δ ≤ |D| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) (hx : x ≤ D + δ) :
    (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (|D| + δ))) * Real.exp x + 2 ^ (-140 : ℤ) ≤ 2 ^ 127 := by
  simp only [z22, z23, z24, z150, z140] at hδ ⊢
  have hep := Real.exp_pos x
  by_cases hD100 : -100 ≤ D
  · have hDa : |D| ≤ 100 := by rw [abs_le]; constructor <;> linarith
    have hδ1 : δ ≤ 1 := by
      have : |D| * (1/16777216 : ℝ) ≤ 100 * (1/16777216 : ℝ) :=
        mul_le_mul_of_nonneg_right hDa (by positivity)
      have e1 : (100 : ℝ) * (1/16777216 : ℝ) ≤ 1 / 2 := by norm_num
      have e2 : (1/2^150 : ℝ) ≤ 1 / 2 := by norm_num
      linarith
    have hrel : (1/4194304 : ℝ) + (1/8388608 : ℝ) * (|D| + δ) ≤ (1 : ℝ) := by
      have : (1/8388608 : ℝ) * (|D| + δ) ≤ (1/8388608 : ℝ) * 101 :=
        mul_le_mul_of_nonneg_left (by linarith) (by positivity)
      have e1 : (1/8388608 : ℝ) * 101 ≤ 1 / 2 := by norm_num
      have e2 : (1/4194304 : ℝ) ≤ 1 / 2 := by norm_num
      linarith
    have hex : Real.exp x ≤ 2 ^ 105 :=
      (Real.exp_le_exp.mpr (by linarith : x ≤ 65)).trans exp65_le
    have : (1 + ((1/4194304 : ℝ) + (1/8388608 : ℝ) * (|D| + δ))) * Real.exp x ≤ 2 * 2 ^ 105 :=
      mul_le_mul (by linarith) hex hep.le (by norm_num)
    have e3 : (1/2^140 : ℝ) ≤ 1 := by norm_num
    have e4 : (2 : ℝ) * 2 ^ 105 + 1 ≤ 2 ^ 127 := by norm_num
    linarith
  · rw [not_le] at hD100
    have hDa : |D| = -D := abs_of_neg (by linarith)
    have hxD : x ≤ D / 2 := by
      have : |D| * (1/16777216 : ℝ) ≤ (-D) * (1 / 4) := by
        rw [hDa]
        exact mul_le_mul_of_nonneg_left (by norm_num) (by linarith)
      have e2 : (1/2^150 : ℝ) ≤ 1 / 4 := by norm_num
      linarith
    have hex : Real.exp x ≤ Real.exp (D / 2) := Real.exp_le_exp.mpr hxD
    have hpos := Real.exp_pos (D / 2)
    have hprod : (-D / 2) * Real.exp (D / 2) ≤ 1 := by
      have h1 : -D / 2 ≤ Real.exp (-D / 2) := by linarith [Real.add_one_le_exp (-D / 2)]
      have h2 : Real.exp (-D / 2) * Real.exp (D / 2) = 1 := by
        rw [← Real.exp_add, show -D / 2 + D / 2 = 0 by ring, Real.exp_zero]
      calc (-D / 2) * Real.exp (D / 2) ≤ Real.exp (-D / 2) * Real.exp (D / 2) :=
            mul_le_mul_of_nonneg_right h1 hpos.le
        _ = 1 := h2
    have hone : Real.exp (D / 2) ≤ 1 := by
      rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr (by linarith)
    have hX : (1/8388608 : ℝ) * (|D| + δ) ≤ (1/8388608 : ℝ) * (2 * (-D)) :=
      mul_le_mul_of_nonneg_left (by
        have : δ ≤ -D := by
          have : |D| * (1/16777216 : ℝ) ≤ (-D) * (1 / 4) := by
            rw [hDa]; exact mul_le_mul_of_nonneg_left (by norm_num) (by linarith)
          have e2 : (1/2^150 : ℝ) ≤ 1 / 4 := by norm_num
          linarith
        rw [hDa]; linarith) (by positivity)
    have h1 : (1 + ((1/4194304 : ℝ) + (1/8388608 : ℝ) * (|D| + δ))) * Real.exp x ≤
        (1 + ((1/4194304 : ℝ) + (1/8388608 : ℝ) * (2 * (-D)))) * Real.exp (D / 2) :=
      mul_le_mul (by linarith) hex hep.le (by linarith)
    have h2 : (1 + ((1/4194304 : ℝ) + (1/8388608 : ℝ) * (2 * (-D)))) * Real.exp (D / 2) ≤ 3 := by
      have e : (1 + ((1/4194304 : ℝ) + (1/8388608 : ℝ) * (2 * (-D)))) * Real.exp (D / 2) =
          (1 + (1/4194304 : ℝ)) * Real.exp (D / 2) + (1/4194304 : ℝ) * ((-D / 2) * Real.exp (D / 2)) * 2 := by
        ring
      rw [e]
      have a1 : (1 + (1/4194304 : ℝ)) * Real.exp (D / 2) ≤ 2 := by
        have : (1 + (1/4194304 : ℝ)) ≤ 2 := by norm_num
        nlinarith
      have a2 : (1/4194304 : ℝ) * ((-D / 2) * Real.exp (D / 2)) * 2 ≤ 1 := by
        have : (1/4194304 : ℝ) * ((-D / 2) * Real.exp (D / 2)) ≤ (1/4194304 : ℝ) * 1 :=
          mul_le_mul_of_nonneg_left hprod (by positivity)
        have e5 : (1/4194304 : ℝ) * 2 ≤ 1 := by norm_num
        nlinarith
      linarith
    have e3 : (1/2^140 : ℝ) ≤ 1 := by norm_num
    have e4 : (3 : ℝ) + 1 ≤ 2 ^ 127 := by norm_num
    linarith

/-- **The decay's error** (`Device.ExpF32`): if the computed exponent is within `δ` of the exact
`D`, `D ≤ 64`, then every result of `tl.exp` is finite and within the claim's `errE` of `e^D`. -/
theorem exp_step {P : Platform} (hexp : Device.ExpF32 P) {D δ : ℝ} (hD : D ≤ 64) (hδ0 : 0 ≤ δ)
    (hδ : δ ≤ |D| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) {a e : F32} (ha : a.isFinite = true)
    (hae : |a.toReal - D| ≤ δ) (he : P.math .exp .binary32 a e) :
    e.isFinite = true ∧ |e.toReal - Real.exp D| ≤
      (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (|D| + δ))) * Real.exp (D + δ) - Real.exp D +
        2 ^ (-140 : ℤ) := by
  obtain ⟨g, hg, hacc⟩ := hexp (|D| + δ) (by positivity)
  have hgr : g = Real.exp := by simpa only [MathFn.ref, Option.some.injEq] using hg.symm
  rw [hgr] at hacc
  set x := a.toReal with hx
  have hxl : D - δ ≤ x := by linarith [(abs_le.mp hae).1]
  have hxu : x ≤ D + δ := by linarith [(abs_le.mp hae).2]
  have hdom : x ∈ Set.Icc (-(|D| + δ)) (|D| + δ) := by
    constructor
    · linarith [neg_abs_le D]
    · linarith [le_abs_self D]
  have hnat : x ∈ MathFn.natDom .exp := Set.mem_univ _
  have hepos := Real.exp_pos x
  have hroom := exp_room hD hδ hxu
  have hroom' : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (|D| + δ))) * |Real.exp x| + 2 ^ (-140 : ℤ) ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [abs_of_pos hepos]
    exact hroom.trans (by
      have := two127_le_max
      have h : (2 : ℝ) ^ 127 = 2 ^ 127 := rfl
      exact this)
  obtain ⟨hfin, herr⟩ := hacc a e he ha hdom hnat hroom'
  refine ⟨hfin, ?_⟩
  rw [abs_of_pos hepos] at herr
  set rel : ℝ := 2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (|D| + δ) with hrel
  have hrel0 : 0 ≤ rel := by positivity
  have hEx : Real.exp x ≤ Real.exp (D + δ) := Real.exp_le_exp.mpr hxu
  have hE := Real.exp_pos D
  have hxl' : Real.exp (D - δ) ≤ Real.exp x := Real.exp_le_exp.mpr hxl
  have ht : 1 ≤ Real.exp δ := Real.one_le_exp hδ0
  have hDδ : Real.exp (D + δ) = Real.exp D * Real.exp δ := Real.exp_add D δ
  have hDmδ : Real.exp (D - δ) * Real.exp δ = Real.exp D := by
    rw [← Real.exp_add]; congr 1; ring
  set E := Real.exp D with hEdef
  set t := Real.exp δ with htdef
  set y := Real.exp x with hydef
  have step1 : |e.toReal - E| ≤ |e.toReal - y| + |y - E| := by
    have := abs_add_le (e.toReal - y) (y - E)
    rwa [show e.toReal - y + (y - E) = e.toReal - E by ring] at this
  have step2 : rel * y + |y - E| ≤ (1 + rel) * (E * t) - E := by
    by_cases hxD : D ≤ x
    · have hyE : E ≤ y := Real.exp_le_exp.mpr hxD
      rw [abs_of_nonneg (by linarith)]
      rw [hDδ] at hEx
      nlinarith
    · rw [not_le] at hxD
      have hyE : y ≤ E := Real.exp_le_exp.mpr hxD.le
      rw [abs_of_nonpos (by linarith)]
      by_cases hr1 : rel ≤ 1
      · -- (1 - rel)·y ≥ (1 - rel)·E / t
        have hEt : E ≤ Real.exp (D - δ) * t := by linarith
        have h1 : (1 - rel) * E ≤ (1 - rel) * (y * t) := by
          have : E ≤ y * t := by nlinarith
          exact mul_le_mul_of_nonneg_left this (by linarith)
        have h2 : 0 ≤ E * ((t - 1) ^ 2 + rel * (t ^ 2 - 1)) := by
          apply mul_nonneg hE.le
          have : 0 ≤ t ^ 2 - 1 := by nlinarith
          positivity
        have : 0 < t := by linarith
        nlinarith
      · rw [not_le] at hr1
        nlinarith
  unfold FloatV.toReal at *
  have hrw : (1 + rel) * Real.exp (D + δ) = (1 + rel) * (E * t) := by rw [hDδ]
  rw [hrw]
  linarith [step1, step2]


/-- **The decay's exponent**: the `float32` difference of two non-positive finite `dA_cumsum`
values is finite and within `|d|·2^-24 + 2^-150` of the exact difference `d`. -/
theorem sub_step {q b a : F32} (hq : q.isFinite = true) (hq0 : q.toRat ≤ 0)
    (hb : b.isFinite = true) (hb0 : b.toRat ≤ 0) (h : FloatV.NaNEq (FloatV.sub q b) a) :
    a.isFinite = true ∧
      |a.toReal - (q.toReal - b.toReal)| ≤ |q.toReal - b.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
  have hreg := Format.binary32_regular
  have hqm := FloatV.abs_toRat_le_maxVal hreg hq
  have hbm := FloatV.abs_toRat_le_maxVal hreg hb
  have hqr : q.toReal ≤ 0 := by unfold FloatV.toReal; exact_mod_cast hq0
  have hbr : b.toReal ≤ 0 := by unfold FloatV.toReal; exact_mod_cast hb0
  have hqmR : |q.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    unfold FloatV.toReal; rw [← Rat.cast_abs]; exact_mod_cast hqm
  have hbmR : |b.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    unfold FloatV.toReal; rw [← Rat.cast_abs]; exact_mod_cast hbm
  have hd : |q.toReal - b.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [abs_of_nonpos] at hqmR hbmR <;> try assumption
    rw [abs_le]; constructor <;> linarith
  obtain ⟨hfin, herr⟩ := FloatV.sub_err_real hreg hq hb hd
  rw [u32, eta32] at herr
  have hr := h.exact_of_isFinite hfin
  exact ⟨hr.1, by rw [hr.2.2]; exact herr⟩

/-- **The new state** `scale·state + new_states` of one chunk: from the product's error against
the exact `e^{dRef}·x` and its size, the result is finite and within the claim's `tol` of
`sRef`. -/
theorem chunk_final {m₀ : Mem} {d : Inst} {h m c : Nat} (hV : Vals m₀ d) (hh : h < d.nheads)
    (hm : m < d.dim) (hc : c < d.nchunks) {fusion : Bool} {s v nw r : F32}
    (hs : s.isFinite = true) (hvf : v.isFinite = true) (hnf : nw.isFinite = true)
    (hn : nw.toReal = newV m₀ d c h m)
    (hprod : |s.toReal * v.toReal - Real.exp (dRef m₀ d h c) * xRef m₀ d h m c| ≤
      eHat m₀ d h c * tolIn m₀ d h m c + errE m₀ d h c * |xRef m₀ d h m c|)
    (hmag : |s.toReal * v.toReal| ≤ eHat m₀ d h c * (|xRef m₀ d h m c| + tolIn m₀ d h m c))
    (hmul : FloatV.NaNEq (FloatV.add (FloatV.mul s v) nw) r ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma s v nw) r)) :
    r.isFinite = true ∧ |r.toReal - sRef m₀ d h m (c + 1)| ≤ tol m₀ d h m (c + 1) := by
  obtain ⟨-, hroom⟩ := hV.2.2.2 h m c hh hm hc
  rw [← hn] at hroom
  obtain ⟨hrf, hre⟩ := muladd_step hs hvf hnf hmul hmag hroom
  refine ⟨hrf, ?_⟩
  rw [z150] at hre
  have e1 : r.toReal - sRef m₀ d h m (c + 1) =
      (r.toReal - (s.toReal * v.toReal + nw.toReal)) +
        (s.toReal * v.toReal - Real.exp (dRef m₀ d h c) * xRef m₀ d h m c) := by
    simp only [sRef, xRef, hn]
    ring
  rw [e1, hn]
  refine (abs_add_le _ _).trans ?_
  have : tol m₀ d h m (c + 1) =
      eHat m₀ d h c * tolIn m₀ d h m c + errE m₀ d h c * |xRef m₀ d h m c| +
        rho * (eHat m₀ d h c * (|xRef m₀ d h m c| + tolIn m₀ d h m c) + |newV m₀ d c h m|) +
        3 * 2 ^ (-150 : ℤ) := by
    simp only [tol, tolIn]
  rw [this, z150]
  rw [hn] at hre
  linarith


/-- The decay's error bound is non-negative. -/
theorem dErr_nonneg (m₀ : Mem) (d : Inst) (h c : Nat) : 0 ≤ dErr m₀ d h c := by
  unfold dErr; split_ifs <;> positivity

/-- The decay's error bound is at most one `float32` rounding of the exponent. -/
theorem dErr_le (m₀ : Mem) (d : Inst) (h c : Nat) :
    dErr m₀ d h c ≤ |dRef m₀ d h c| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
  unfold dErr; split_ifs
  · exact le_rfl
  · positivity

/-- **The computed exponent** of chunk `c`: the loaded `dA_cumsum` value, or (when the chunk
starts a request after its first token) the `float32` difference with the value before it, is
finite and within `dErr` of `dRef`. -/
theorem dA_ok {m₀ : Mem} {d : Inst} {h c : Nat} (hV : Vals m₀ d) (hh : h < d.nheads)
    (hc : c < d.nchunks) {q b dAcs : F32}
    (hq : m₀.read .f32 (d.dA.ptr (dAIdx d h c (d.chunk - 1) : Int)).addr = some q)
    (hch : 0 < d.chunk)
    (hcase : (hasSub d c = false ∧ dAcs = q) ∨
      (hasSub d c = true ∧ m₀.read .f32 (d.dA.ptr (dAIdx d h c (offR d c - 1) : Int)).addr = some b ∧
        FloatV.NaNEq (FloatV.sub q b) dAcs))
    (hoff : hasSub d c = true → offR d c - 1 < d.chunk) :
    dAcs.isFinite = true ∧ |dAcs.toReal - dRef m₀ d h c| ≤ dErr m₀ d h c := by
  obtain ⟨q', hq', hqf, hq0⟩ := hV.2.2.1 h c (d.chunk - 1) hh hc (by omega)
  rw [hq] at hq'
  cases hq'
  have hvq : valRat m₀ d.dA (dAIdx d h c (d.chunk - 1)) = q.toRat := by
    simp only [valRat, hq]
  rcases hcase with ⟨hs, rfl⟩ | ⟨hs, hb, hsub⟩
  · refine ⟨hqf, ?_⟩
    have : dRef m₀ d h c = dAcs.toReal := by
      simp only [dRef, hs, hvq]
      simp [FloatV.toReal]
    rw [this, sub_self, abs_zero]
    unfold dErr; rw [hs]; simp
  · obtain ⟨b', hb', hbf, hb0⟩ := hV.2.2.1 h c (offR d c - 1) hh hc (hoff hs)
    rw [hb] at hb'
    cases hb'
    have hvb : valRat m₀ d.dA (dAIdx d h c (offR d c - 1)) = b.toRat := by
      simp only [valRat, hb]
    obtain ⟨hf, he⟩ := sub_step hqf hq0 hbf hb0 hsub
    have hD : dRef m₀ d h c = q.toReal - b.toReal := by
      simp only [dRef, hs, hvq, hvb, ite_true]
      simp [FloatV.toReal]
    refine ⟨hf, ?_⟩
    rw [hD]
    unfold dErr; simp only [hs, ite_true]; rw [hD]
    exact he

/-- **The decay, `exp`, and the state**: with the exponent within `dErr` of `dRef` and the state
within `tolIn` of `xRef`, the product `tl.exp(dA_cs) · state` is within `eHat·tolIn +
errE·|xRef|` of `e^{dRef}·xRef` and at most `eHat·(|xRef| + tolIn)`. -/
theorem prod_exp {P : Platform} (hexp : Device.ExpF32 P) {m₀ : Mem} {d : Inst} {h m c : Nat}
    (hV : Vals m₀ d) (hh : h < d.nheads) (hm : m < d.dim) (hc : c < d.nchunks)
    {dAcs e v : F32} (hdAf : dAcs.isFinite = true)
    (hdAe : |dAcs.toReal - dRef m₀ d h c| ≤ dErr m₀ d h c) (he : P.math .exp .binary32 dAcs e)
    (hve : |v.toReal - xRef m₀ d h m c| ≤ tolIn m₀ d h m c) :
    e.isFinite = true ∧
      |e.toReal * v.toReal - Real.exp (dRef m₀ d h c) * xRef m₀ d h m c| ≤
        eHat m₀ d h c * tolIn m₀ d h m c + errE m₀ d h c * |xRef m₀ d h m c| ∧
      |e.toReal * v.toReal| ≤ eHat m₀ d h c * (|xRef m₀ d h m c| + tolIn m₀ d h m c) := by
  obtain ⟨hD64, -⟩ := hV.2.2.2 h m c hh hm hc
  obtain ⟨hef, hee⟩ := exp_step hexp hD64 (dErr_nonneg m₀ d h c) (dErr_le m₀ d h c) hdAf hdAe he
  have hee' : |e.toReal - Real.exp (dRef m₀ d h c)| ≤ errE m₀ d h c := hee
  have hE := Real.exp_pos (dRef m₀ d h c)
  have he_abs : |e.toReal| ≤ eHat m₀ d h c := by
    unfold eHat
    have := abs_sub_abs_le_abs_sub e.toReal (Real.exp (dRef m₀ d h c))
    rw [abs_of_pos hE] at this
    linarith
  have hT : 0 ≤ tolIn m₀ d h m c := (abs_nonneg _).trans hve
  have hX := abs_nonneg (xRef m₀ d h m c)
  refine ⟨hef, ?_, ?_⟩
  · have e1 : e.toReal * v.toReal - Real.exp (dRef m₀ d h c) * xRef m₀ d h m c =
        e.toReal * (v.toReal - xRef m₀ d h m c) +
          (e.toReal - Real.exp (dRef m₀ d h c)) * xRef m₀ d h m c := by ring
    rw [e1]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul]
    have a1 : |e.toReal| * |v.toReal - xRef m₀ d h m c| ≤ eHat m₀ d h c * tolIn m₀ d h m c :=
      mul_le_mul he_abs hve (abs_nonneg _) ((abs_nonneg _).trans he_abs)
    have a2 : |e.toReal - Real.exp (dRef m₀ d h c)| * |xRef m₀ d h m c| ≤
        errE m₀ d h c * |xRef m₀ d h m c| := mul_le_mul_of_nonneg_right hee' hX
    linarith
  · rw [abs_mul]
    have hv : |v.toReal| ≤ |xRef m₀ d h m c| + tolIn m₀ d h m c := by
      have := abs_sub_abs_le_abs_sub v.toReal (xRef m₀ d h m c)
      linarith
    exact mul_le_mul he_abs hv (abs_nonneg _) ((abs_nonneg _).trans he_abs)

end Claims.SsdStatePassing
