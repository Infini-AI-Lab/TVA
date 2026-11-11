import TVA.Proof
import Claim
import Proof.Lanes

/-!
# One loop step

One loop step: the room `tl.dot` needs, the tolerance of the accumulator, `lane_step` from the device's `float16` dot fact, the pointer advance, and the masks of the tile.
-/


/-!
# The `float32` to `float16` conversion

`tl.cvt`'s result for a `float32` element is the format's `FloatV.cvt`.
-/

namespace Claims.Matmul

open TVA

/-- One rounding down to `float16` is what `ConvRel` allows from `float32`. -/
theorem convRel_cvt16 (P : Platform) (x : F32) :
    ConvRel P .f32 .f16 x (FloatV.cvt x) := by
  show FloatRel P .binary32 .binary16 x (FloatV.cvt x)
  unfold FloatRel
  have hne : ¬(Format.binary32 = Format.binary16) := by decide
  simp only [hne, ite_false]
  exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, FloatV.NaNEq.refl _⟩

end Claims.Matmul

/- ===== Room ===== -/

/-!
# `tl.dot`'s room

The accumulator and the step's lifted magnitudes stay far below `float32`'s range, so `tl.dot` has
room: both the `DotLiftAcc` premise and `dotAdd_step`'s hypothesis hold.
-/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- With `|vc| + S ≤ 2^47` the dot has room: the lifted bound fits `float32`'s and so does the
step's allowance. -/
theorem dot_room {vc S : ℚ} (h : |vc| + S ≤ 2 ^ 47) :
    |vc| + S + (Device.relDot16 64 * (|vc| + S) + Device.absDot16 64) ≤ DType.bound .f32 ∧
      2 * ((1 + Device.relDot16 64) * (|vc| + S) + Device.absDot16 64 +
        2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
  have hrel : (0 : ℚ) ≤ Device.relDot16 64 := by
    unfold Device.relDot16; positivity
  constructor
  · have h1 : Device.relDot16 64 * (|vc| + S) ≤ Device.relDot16 64 * 2 ^ 47 :=
      mul_le_mul_of_nonneg_left h hrel
    have hA : (2 : ℚ) ^ 47 + (Device.relDot16 64 * 2 ^ 47 + Device.absDot16 64) ≤
        DType.bound .f32 := by
      simp only [DType.bound, Format.binary32_maxVal]
      unfold Device.relDot16 Device.absDot16
      norm_num
    linarith
  · have h1 : (1 + Device.relDot16 64) * (|vc| + S) ≤
        (1 + Device.relDot16 64) * 2 ^ 47 :=
      mul_le_mul_of_nonneg_left h (by linarith)
    have hB : 2 * ((1 + Device.relDot16 64) * 2 ^ 47 + Device.absDot16 64 +
        2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
      rw [Format.binary32_maxVal, Format.binary32_eta]
      unfold Device.relDot16 Device.absDot16
      norm_num
    linarith

/-- The magnitude the loop's lane keeps: `2^45 + 2^37` of accumulated value plus `2^38` of step is
below `2^47`. -/
theorem dot_room' {vc S : ℚ} (hvc : |vc| ≤ 2 ^ 45 + 2 ^ 37 + 2 ^ (-100 : ℤ))
    (hS : S ≤ 2 ^ 38) : |vc| + S ≤ 2 ^ 47 := by
  have h1 : |vc| + S ≤ ((2 : ℚ) ^ 45 + 2 ^ 37 + 2 ^ (-100 : ℤ)) + 2 ^ 38 :=
    add_le_add hvc hS
  have h2 : ((2 : ℚ) ^ 45 + 2 ^ 37 + 2 ^ (-100 : ℤ)) + 2 ^ 38 ≤ 2 ^ 47 := by norm_num
  linarith

end Claims.Matmul

/- ===== Tol ===== -/

/-!
# The accumulation's bounds

The accumulator's error after `t` steps, `AccB`, against the claim's `tol`: the `2^-15 · pathMag`
tolerance, the relaxed fine bound used by the final `float16` rounding, and the crude `2^37` that
keeps the accumulator far from `float32`'s range.
-/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- The total lifted magnitude of the reduction is at most `2^52`. -/
theorem pathMag_le {d : Inst} (hd : d.ok) {va vb : Nat → F16}
    (ha : ∀ e, e < d.K → (va e).isFinite = true)
    (hb : ∀ e, e < d.K → (vb e).isFinite = true) :
    pathMag va vb d.K ≤ 2 ^ 52 := by
  unfold pathMag
  set n := steps d.K with hn
  have hn128 : n ≤ 128 := steps_le d hd
  have h238 : (0 : ℚ) ≤ 2 ^ 38 := by norm_num
  have hterm : ∀ s ∈ Finset.range n,
      |partSum (prodV va vb d.K) s| + stepMag (liftV va vb d.K) s ≤ (n : ℚ) * 2 ^ 38 := by
    intro s hs
    have hsn : s + 1 ≤ n := by
      have := Finset.mem_range.mp hs
      omega
    have h1 := partSum_abs_le ha hb s
    have h2 := stepMag_lt (K := d.K) (t := s) ha hb
    have h3 : ((s : ℚ) + 1) ≤ (n : ℚ) := by exact_mod_cast hsn
    nlinarith
  have hsum : (∑ s ∈ Finset.range n,
        (|partSum (prodV va vb d.K) s| + stepMag (liftV va vb d.K) s)) ≤
      (n : ℚ) * ((n : ℚ) * 2 ^ 38) := by
    calc (∑ s ∈ Finset.range n,
          (|partSum (prodV va vb d.K) s| + stepMag (liftV va vb d.K) s))
        ≤ ∑ _s ∈ Finset.range n, (n : ℚ) * 2 ^ 38 := Finset.sum_le_sum hterm
      _ = (n : ℚ) * ((n : ℚ) * 2 ^ 38) := by
            rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hnq : (n : ℚ) ≤ 128 := by exact_mod_cast hn128
  have hn0 : (0 : ℚ) ≤ (n : ℚ) := by positivity
  have hsq : (n : ℚ) * (n : ℚ) ≤ 128 * 128 := by nlinarith
  have hpow : (128 : ℚ) * 128 * 2 ^ 38 = 2 ^ 52 := by norm_num
  nlinarith

/-- The accumulator's error after `t ≤ steps K` steps is within the fine bound. -/
theorem AccB_le_fine {d : Inst} (hd : d.ok) {va vb : Nat → F16}
    {T : Nat} (hT : T ≤ steps d.K) :
    AccB d.K va vb T ≤ (3 / 2) * rho * pathMag va vb d.K + (3 / 2) * (T : ℚ) * alp := by
  unfold AccB
  have hT128 : T ≤ 128 := le_trans hT (steps_le d hd)
  have hρ : 0 ≤ rho := by unfold rho; norm_num
  have hα : 0 ≤ alp := by unfold alp; norm_num
  have hP : 0 ≤ pathMag va vb d.K := pathMag_nonneg va vb d.K
  have hpow : (1 + rho) ^ T ≤ 3 / 2 := pow_le hT128
  have hSle : (∑ s ∈ Finset.range T,
        (|partSum (prodV va vb d.K) s| + stepMag (liftV va vb d.K) s)) ≤
      pathMag va vb d.K := by
    unfold pathMag
    apply Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hT)
    intro x _ _
    exact add_nonneg (abs_nonneg _) (stepMag_nonneg (liftV_nonneg va vb d.K) x)
  have hbase : (0 : ℚ) ≤ (1 + rho) ^ T := pow_nonneg (by linarith) T
  have hmono : (1 + rho) ^ T *
        (rho * (∑ s ∈ Finset.range T,
          (|partSum (prodV va vb d.K) s| + stepMag (liftV va vb d.K) s)) + (T : ℚ) * alp) ≤
      (1 + rho) ^ T * (rho * pathMag va vb d.K + (T : ℚ) * alp) := by
    apply mul_le_mul_of_nonneg_left _ hbase
    have hm := mul_le_mul_of_nonneg_left hSle hρ
    linarith
  have hfb := fine_bound hρ hα hP hpow
  linarith

/-- The accumulator's error after `t ≤ steps K` steps is within the claim's `tol`. -/
theorem AccB_le_tol {d : Inst} (hd : d.ok) {va vb : Nat → F16}
    (_ha : ∀ e, e < d.K → (va e).isFinite = true)
    (_hb : ∀ e, e < d.K → (vb e).isFinite = true)
    {T : Nat} (hT : T ≤ steps d.K) :
    AccB d.K va vb T ≤ tol va vb d.K := by
  have h1 := AccB_le_fine (va := va) (vb := vb) hd hT
  have hT128 : T ≤ 128 := le_trans hT (steps_le d hd)
  have h2 := relax_tol (pathMag_nonneg va vb d.K) T hT128
  unfold tol
  exact h1.trans h2

/-- The accumulator's error after `t ≤ steps K` steps is at most `2^37 + 2^-100`. -/
theorem AccB_le {d : Inst} (hd : d.ok) {va vb : Nat → F16}
    (ha : ∀ e, e < d.K → (va e).isFinite = true)
    (hb : ∀ e, e < d.K → (vb e).isFinite = true)
    {T : Nat} (hT : T ≤ steps d.K) :
    AccB d.K va vb T ≤ 2 ^ 37 + 2 ^ (-100 : ℤ) := by
  have h1 := AccB_le_tol hd ha hb hT
  have hM := pathMag_le hd ha hb
  have hc : (0 : ℚ) ≤ 2 ^ (-15 : ℤ) := by norm_num
  have h3 : 2 ^ (-15 : ℤ) * pathMag va vb d.K ≤ 2 ^ (-15 : ℤ) * 2 ^ 52 :=
    mul_le_mul_of_nonneg_left hM hc
  have h4 : (2 : ℚ) ^ (-15 : ℤ) * 2 ^ 52 = 2 ^ 37 := by norm_num
  unfold tol at h1
  refine h1.trans ?_
  have h5 := add_le_add_right h3 (2 ^ (-100 : ℤ))
  rw [h4] at h5
  simpa only [add_comm] using h5

end Claims.Matmul

/- ===== LaneStep ===== -/

/-!
# One step of the loop, at one lane

The numeric core of the loop body: given the two loaded tiles' values at a lane, the previous
accumulator value's bound and the `tl.dot`'s result, the new accumulator value is within `AccB` of
the exact partial sum after `t + 1` steps.
-/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- The lifted magnitude of step `t`, elementwise. -/
theorem stepMag_liftV_eq {d : Inst} (hd : d.ok) {va vb : Nat → F16} {t : Nat}
    (ht : t < steps d.K) :
    stepMag (liftV va vb d.K) t
      = ∑ kk : Fin 64, Device.liftMag .f16 (va (64 * t + kk)).toRat *
          Device.liftMag .f16 (vb (64 * t + kk)).toRat := by
  unfold stepMag
  refine Finset.sum_congr rfl ?_
  intro kk _
  simp only [liftV, step_lt hd ht kk.isLt, ite_true]

/-- The exact products of step `t`, elementwise. -/
theorem stepMag_prodV_eq {d : Inst} (hd : d.ok) {va vb : Nat → F16} {t : Nat}
    (ht : t < steps d.K) :
    stepMag (prodV va vb d.K) t
      = ∑ kk : Fin 64, (va (64 * t + kk)).toRat * (vb (64 * t + kk)).toRat := by
  unfold stepMag
  refine Finset.sum_congr rfl ?_
  intro kk _
  simp only [prodV, step_lt hd ht kk.isLt, ite_true]

/-- **One step at one lane.** -/
theorem lane_step {P : Platform} {d : Inst} (hd : d.ok) (hdot : Device.DotF16 P)
    {t : Nat} (ht : t < steps d.K) {rr : Fin 128} {cc : Fin 256}
    {va vb : Nat → F16}
    (ha : ∀ e, e < d.K → (va e).isFinite = true)
    (hb : ∀ e, e < d.K → (vb e).isFinite = true)
    {a : Tensor [128, 64] F16} {b : Tensor [64, 256] F16}
    {old new : Tensor [128, 256] F32} {v0 : ℚ}
    (hva : ∀ kk : Fin 64, DType.val? .f16 (a.get2 rr kk) = some ((va (64 * t + kk)).toRat))
    (hvb : ∀ kk : Fin 64, DType.val? .f16 (b.get2 kk cc) = some ((vb (64 * t + kk)).toRat))
    (hval0 : DType.val? .f32 (old.get2 rr cc) = some v0)
    (hprev : |v0 - partSum (prodV va vb d.K) t| ≤ AccB d.K va vb t)
    (hdotres : P.dot ⟨.f16, .f16, .f32, .tf32, 0⟩ a b old new) :
    ∃ v, DType.val? .f32 (new.get2 rr cc) = some v ∧
      |v - partSum (prodV va vb d.K) (t + 1)| ≤ AccB d.K va vb (t + 1) := by
  have ht128 : t ≤ 128 := le_trans (le_of_lt ht) (steps_le d hd)
  have hrel : (0 : ℚ) ≤ Device.relDot16 64 := by
    unfold Device.relDot16; positivity
  have habs : (0 : ℚ) ≤ Device.absDot16 64 := by
    unfold Device.absDot16; positivity
  have hrho : (0 : ℚ) ≤ rho := rho_nonneg
  have halp : (0 : ℚ) ≤ alp := alp_nonneg
  -- the step's exact values and lifted magnitudes
  have hSsucc : stepMag (liftV va vb d.K) t
      = ∑ kk : Fin 64, Device.liftMag .f16 (va (64 * t + kk)).toRat *
          Device.liftMag .f16 (vb (64 * t + kk)).toRat := stepMag_liftV_eq hd ht
  have hPsucc : stepMag (prodV va vb d.K) t
      = ∑ kk : Fin 64, (va (64 * t + kk)).toRat * (vb (64 * t + kk)).toRat :=
    stepMag_prodV_eq hd ht
  have hLle : stepMag (liftV va vb d.K) t ≤ 2 ^ 38 := stepMag_lt ha hb
  -- the previous value's magnitude
  have hacc_le : AccB d.K va vb t ≤ 2 ^ 37 + 2 ^ (-100 : ℤ) := AccB_le hd ha hb (le_of_lt ht)
  have hS_le : |partSum (prodV va vb d.K) t| ≤ (t : ℚ) * 2 ^ 38 := partSum_abs_le ha hb t
  have htq : (t : ℚ) ≤ 128 := by exact_mod_cast ht128
  have ht38 : (t : ℚ) * 2 ^ 38 ≤ 2 ^ 45 := by nlinarith
  have hvc : |v0| ≤ 2 ^ 45 + 2 ^ 37 + 2 ^ (-100 : ℤ) := by
    have h1 : |v0| ≤ |partSum (prodV va vb d.K) t| + |v0 - partSum (prodV va vb d.K) t| := by
      have e : v0 = (v0 - partSum (prodV va vb d.K) t) + partSum (prodV va vb d.K) t := by ring
      calc |v0| = |(v0 - partSum (prodV va vb d.K) t) + partSum (prodV va vb d.K) t| :=
            congrArg abs e
        _ ≤ |v0 - partSum (prodV va vb d.K) t| + |partSum (prodV va vb d.K) t| := abs_add_le _ _
        _ = |partSum (prodV va vb d.K) t| + |v0 - partSum (prodV va vb d.K) t| := by ring
    linarith
  -- the room `tl.dot` needs
  have hroom47 : |(old.get2 rr cc).toRat| + stepMag (liftV va vb d.K) t ≤ 2 ^ 47 := by
    have h0 : (old.get2 rr cc).toRat = v0 := (DType.val?_f32.mp hval0).2
    rw [h0]
    exact dot_room' hvc hLle
  have hroom : 2 * ((1 + Device.relDot16 64) *
        (|(old.get2 rr cc).toRat| + ∑ kk : Fin 64,
          Device.liftMag .f16 (va (64 * t + kk)).toRat *
            Device.liftMag .f16 (vb (64 * t + kk)).toRat) +
      Device.absDot16 64 + 2 * Format.binary32.eta) ≤
      Format.binary32.maxVal := by
    rw [← hSsucc]
    exact (dot_room hroom47).2
  have hfin0 : (old.get2 rr cc).isFinite = true := (DType.val?_f32.mp hval0).1
  obtain ⟨hfin, herr⟩ := DotLift.dotAdd_step (fusion := false) (hacc := hdot .tf32) (K := 64) hrel habs
    (Or.inl hdotres) rr cc (z := tl.zeros [128, 256] .f32) DType.val?_zero_f32 hva hvb hfin0 hroom
  -- rewrite the step bound with the loop's constants
  have hstep : |(new.get2 rr cc).toRat -
      (v0 + stepMag (prodV va vb d.K) t)| ≤ rho * (|v0| + stepMag (liftV va vb d.K) t) + alp := by
    have h0 : (old.get2 rr cc).toRat = v0 := (DType.val?_f32.mp hval0).2
    rw [← hPsucc] at herr
    rw [← hSsucc] at herr
    rw [h0] at herr
    refine herr.trans ?_
    have hnn : (0 : ℚ) ≤ |v0| + stepMag (liftV va vb d.K) t := by
      linarith [abs_nonneg v0, stepMag_nonneg (liftV_nonneg va vb d.K) t]
    have h1 : DotLoop.stepRel Device.relDot16 64 * (|v0| + stepMag (liftV va vb d.K) t) ≤
        rho * (|v0| + stepMag (liftV va vb d.K) t) :=
      mul_le_mul_of_nonneg_right stepRel_le_rho hnn
    have h2 : DotLoop.stepAbs Device.absDot16 64 ≤ alp := stepAbs_le_alp
    linarith
  -- the recurrence
  set acc : Nat → ℚ := fun s => if s = t then v0 else if s = t + 1 then (new.get2 rr cc).toRat else 0
  have hacc_t : acc t = v0 := by simp [acc]
  have hacc_s : acc (t + 1) = (new.get2 rr cc).toRat := by
    simp [acc]
  have hLt : (0 : ℚ) ≤ stepMag (liftV va vb d.K) t := stepMag_nonneg (liftV_nonneg va vb d.K) t
  have hprev' : |acc t - partSum (prodV va vb d.K) t| ≤
      (1 + rho) ^ t * (rho * (∑ s ∈ Finset.range t,
        (|partSum (prodV va vb d.K) s| + stepMag (liftV va vb d.K) s)) + (t : ℚ) * alp) := by
    rw [hacc_t]
    unfold AccB at hprev
    exact hprev
  have hstep' : |acc (t + 1) - (acc t + (partSum (prodV va vb d.K) (t + 1) -
      partSum (prodV va vb d.K) t))| ≤ rho * (|acc t| + stepMag (liftV va vb d.K) t) + alp := by
    have hs : v0 + (partSum (prodV va vb d.K) (t + 1) - partSum (prodV va vb d.K) t)
        = v0 + stepMag (prodV va vb d.K) t := by rw [partSum_succ]; ring
    rw [hacc_s, hacc_t, hs]
    exact hstep
  have hres := err_step hrho hLt halp hprev' hstep'
  refine ⟨(new.get2 rr cc).toRat, (DType.val?_f32).mpr ⟨hfin, rfl⟩, ?_⟩
  rw [hacc_s] at hres
  exact hres

end Claims.Matmul

/- ===== Step ===== -/

/-!
# Advancing the tile pointers one step

Adding the broadcast `int32` offset `64 * stride_ak` (`64 * stride_bk`) to `a`'s (`b`'s) tile
pointers advances every lane from step `t` to step `t + 1`.
-/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- Offsetting the pointer to element `x` of a buffer by `y` gives the pointer to element
`x + y`. -/
theorem Buf_ptr_offset {dd : DType} (b : Buf dd) (x y : Int) :
    (b.ptr x).offset y = b.ptr (x + y) := by
  simp only [Buf.ptr, Ptr.offset]
  congr 1
  rw [Int.add_mul]
  ring

/-- Lane `(i, j)` of a broadcast tensor. -/
theorem get2_full {m n : Nat} {α : Type} (a : α) (i : Fin m) (j : Fin n) :
    (Tensor.full a : Tensor [m, n] α).get2 i j = a := rfl

/-- The broadcast step offset `64 * stride_ak` of `a`'s pointers, at its integer value. -/
theorem step_a_toInt {d : Inst} (hd : d.ok) :
    ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_ak : I32)).toInt
      = 64 * (d.stride_ak : Int) := by
  have hak : ((IntV.ofNat d.stride_ak : I32)).toInt = (d.stride_ak : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.ak.2; omega)
  have h64' : ((IntV.ofNat 64 : I32)).toInt = (64 : Int) := i32_64
  rw [IntV.toInt_mul rfl rfl
    (by rw [h64', hak]; exact s32 (by omega) (by have := stride_step_a hd; omega)),
    h64', hak]

/-- The address advances by one step: `aAdr t + 64 * stride_ak = aAdr (t + 1)`. -/
theorem aAdr_succ {d : Inst} (i t r kk : Nat) :
    (aAdr d i t r kk : Int) + 64 * (d.stride_ak : Int)
      = (aAdr d i (t + 1) r kk : Int) := by
  unfold aAdr
  push_cast
  ring

/-- Advancing `a`'s tile pointers one step. -/
theorem aStep_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {st1 : Tensor [128, 64] (Ptr .f16)} {t : Nat} {r : Fin 128} {kk : Fin 64}
    (h : st1.get2 r kk = d.a.ptr ↑(aAdr d i t r kk)) :
    (st1 + (Tensor.full ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_ak : I32)) :
        Tensor [128, 64] I32)).get2 r kk
      = d.a.ptr ↑(aAdr d i (t + 1) r kk) := by
  rw [ptrAddT_get2, get2_full, step_a_toInt hd, h, Buf_ptr_offset, aAdr_succ]

/-- The broadcast step offset `64 * stride_bk` of `b`'s pointers, at its integer value. -/
theorem step_b_toInt {d : Inst} (hd : d.ok) :
    ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_bk : I32)).toInt
      = 64 * (d.stride_bk : Int) := by
  have hbk : ((IntV.ofNat d.stride_bk : I32)).toInt = (d.stride_bk : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.bk.2; omega)
  have h64 : ((IntV.ofNat 64 : I32)).toInt = 64 := i32_64
  have h1 : 64 * d.stride_bk < 2 ^ 31 := stride_step_b hd
  have hfit : IntV.SFits 32
      (((IntV.ofNat 64 : I32)).toInt * ((IntV.ofNat d.stride_bk : I32)).toInt) := by
    rw [h64, hbk]
    have h4 : (64 : Int) * (d.stride_bk : Int) = ((64 * d.stride_bk : Nat) : Int) := by
      push_cast; ring
    rw [h4]
    apply s32
    · have h0 : (0 : Int) ≤ ((64 * d.stride_bk : Nat) : Int) := by positivity
      omega
    · exact_mod_cast h1
  rw [IntV.toInt_mul rfl rfl hfit, h64, hbk]

/-- The address advances by one step: `bAdr t + 64 * stride_bk = bAdr (t + 1)`. -/
theorem bAdr_succ {d : Inst} (i t kk c : Nat) :
    (bAdr d i t kk c : Int) + 64 * (d.stride_bk : Int)
      = (bAdr d i (t + 1) kk c : Int) := by
  unfold bAdr
  push_cast
  ring

/-- Advancing `b`'s tile pointers one step. -/
theorem bStep_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {st2 : Tensor [64, 256] (Ptr .f16)} {t : Nat} {kk : Fin 64} {c : Fin 256}
    (h : st2.get2 kk c = d.b.ptr ↑(bAdr d i t kk c)) :
    (st2 + (Tensor.full ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_bk : I32)) :
        Tensor [64, 256] I32)).get2 kk c
      = d.b.ptr ↑(bAdr d i (t + 1) kk c) := by
  rw [ptrAddT_get2, get2_full, step_b_toInt hd, h, Buf_ptr_offset, bAdr_succ]

end Claims.Matmul

/- ===== MaskLane ===== -/

/-!
# The masks and the `c` pointers, lane by lane

The three masks of the model at a lane, and the pointer lane of the store.
-/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- Lane `(r, c)` of a scalar times a tensor: the scalar times the lane. -/
theorem get2_smul {m n : Nat} (a : I32) (x : Tensor [m, n] I32) (r : Fin m) (c : Fin n) :
    (a * x).get2 r c = a * x.get2 r c := rfl

/-- The scalar `K - 64·t` of the two load masks, as an integer. -/
theorem step_mask_val {d : Inst} (hd : d.ok) {t : Nat} (ht : t ≤ 128) :
    (IntV.ofNat d.K - IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) *
        (IntV.ofNat 64 : I32)).toInt = (d.K : Int) - 64 * (t : Int) := by
  have h0 : ((IntV.ofInt 0 : I32)).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)
  have h1 : ((IntV.ofInt 1 : I32)).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)
  have htI : (t : Int) ≤ 128 := by exact_mod_cast ht
  have ht0 : (0 : Int) ≤ (t : Int) := by exact_mod_cast Nat.zero_le t
  have hBv : ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) = (t : Int) := by
    rw [h0, h1]; ring
  have hB : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32).toInt
      = (t : Int) := by
    rw [IntV.I32.toInt_ofInt (by rw [hBv]; omega), hBv]
  have hK : (IntV.ofNat d.K : I32).toInt = (d.K : Int) := i32_K d hd
  have h64 : ((IntV.ofNat 64 : I32)).toInt = 64 := i32_64
  have hmul : ((IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32) *
      (IntV.ofNat 64 : I32)).toInt = (t : Int) * 64 := by
    rw [IntV.toInt_mul rfl rfl (by rw [hB, h64]; exact s32 (by omega) (by omega)), hB, h64]
  have hKle : (d.K : Int) ≤ 2 ^ 13 := by
    have := hd.K_le; exact_mod_cast this
  rw [IntV.toInt_sub rfl rfl (by rw [hK, hmul]; exact s32 (by omega) (by omega)), hK, hmul]
  ring

/-- The `a` mask at a lane is on. -/
theorem a_mask_get2 {d : Inst} (hd : d.ok) {t : Nat} (ht : t < steps d.K) (r : Fin 128) (kk : Fin 64) :
    (((tl.arange 0 64).row 128).ltS
      (IntV.ofNat d.K - IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) *
        (IntV.ofNat 64 : I32))).get2 r kk = true := by
  have ht128 : t ≤ 128 := by have := steps_le d hd; omega
  have hs := step_mask_val hd ht128
  show IntV.lt (((tl.arange 0 64).row 128).get (r, kk, ())) _ = true
  rw [Tensor.get_row]
  exact mask_k_true hd hs ht

/-- The `b` mask at a lane is on. -/
theorem b_mask_get2 {d : Inst} (hd : d.ok) {t : Nat} (ht : t < steps d.K) (kk : Fin 64) (c : Fin 256) :
    (((tl.arange 0 64).col 256).ltS
      (IntV.ofNat d.K - IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) *
        (IntV.ofNat 64 : I32))).get2 kk c = true := by
  have ht128 : t ≤ 128 := by have := steps_le d hd; omega
  have hs := step_mask_val hd ht128
  show IntV.lt (((tl.arange 0 64).col 256).get (kk, c, ())) _ = true
  rw [Tensor.get_col]
  exact mask_k_true hd hs ht

/-- The store's mask at a lane is the in-range test. -/
theorem c_mask_get2 {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r3 r5 r7 : Tensor [] I32}
    (hg : r3.item.toInt = (grp d i : Int))
    (ho : r5.item.toInt = ((pidM d i - fpm d i : Nat) : Int))
    (hpn : r7.item.toInt = ((pidN d i : Nat) : Int)) (r : Fin 128) (c : Fin 256) :
    (((((r3.item * IntV.ofNat 8 + r5.item) * IntV.ofNat 128 + tl.arange 0 128).col 256).ltS
        (IntV.ofNat d.M)).and
      (((r7.item * IntV.ofNat 256 + tl.arange 0 256).row 128).ltS (IntV.ofNat d.N))).get2 r c
      = (decide (cRow d i r < d.M) && decide (cCol d i c < d.N)) := by
  have hrowv : ((((r3.item * IntV.ofNat 8 + r5.item) * IntV.ofNat 128 + tl.arange 0 128).col 256).get (r, c, ())).toInt
      = (cRow d i r : Int) := by
    rw [Tensor.get_col, offs_row_val hd i hg ho r]
    unfold cRow
    push_cast
    ring
  have hcolv : ((((r7.item * IntV.ofNat 256 + tl.arange 0 256).row 128).get (r, c, ())).toInt)
      = (cCol d i c : Int) := by
    rw [Tensor.get_row, offs_col_val hd i hpn c]
    unfold cCol
    push_cast
    ring
  have hrow : IntV.lt ((((r3.item * IntV.ofNat 8 + r5.item) * IntV.ofNat 128 + tl.arange 0 128).col 256).get (r, c, ()))
      (IntV.ofNat d.M) = decide (cRow d i r < d.M) := by
    rw [IntV.lt_eq, hrowv, i32_M d hd]
    simp only [Nat.cast_lt]
  have hcol : IntV.lt ((((r7.item * IntV.ofNat 256 + tl.arange 0 256).row 128).get (r, c, ())))
      (IntV.ofNat d.N) = decide (cCol d i c < d.N) := by
    rw [IntV.lt_eq, hcolv, i32_N d hd]
    simp only [Nat.cast_lt]
  show (IntV.lt ((((r3.item * IntV.ofNat 8 + r5.item) * IntV.ofNat 128 + tl.arange 0 128).col 256).get (r, c, ()))
        (IntV.ofNat d.M) &&
      IntV.lt ((((r7.item * IntV.ofNat 256 + tl.arange 0 256).row 128).get (r, c, ())))
        (IntV.ofNat d.N)) = (decide (cRow d i r < d.M) && decide (cCol d i c < d.N))
  rw [hrow, hcol]

/-- The store's pointer at a lane points at the lane's element of `c`, for a lane that is in range
(without the hypotheses `d.ok` and the in-range tests this is false: `stride_cm` is an
unconstrained `Nat` there, and the `int32` products may wrap). -/
theorem cptr_get2_fit {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r3 r5 r7 : Tensor [] I32}
    (hg : r3.item.toInt = (grp d i : Int))
    (ho : r5.item.toInt = ((pidM d i - fpm d i : Nat) : Int))
    (hpn : r7.item.toInt = ((pidN d i : Nat) : Int)) (r : Fin 128) (c : Fin 256)
    (hM : cRow d i r < d.M) (hN : cCol d i c < d.N) :
    (d.c.ptr 0 + (IntV.ofNat d.stride_cm : I32) *
        (((r3.item * IntV.ofNat 8 + r5.item) * IntV.ofNat 128 + tl.arange 0 128).col 256) +
      (IntV.ofNat d.stride_cn : I32) *
        ((r7.item * IntV.ofNat 256 + tl.arange 0 256).row 128)).get2 r c
      = d.c.ptr ↑(cAdr d i r c) := by
  have hcm : ((IntV.ofNat d.stride_cm : I32)).toInt = (d.stride_cm : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.cm.2; omega)
  have hcn : ((IntV.ofNat d.stride_cn : I32)).toInt = (d.stride_cn : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.cn.2; omega)
  have hrowv : ((((r3.item * IntV.ofNat 8 + r5.item) * IntV.ofNat 128 + tl.arange 0 128).col 256).get2 r c).toInt
      = (cRow d i r : Int) := by
    change ((((r3.item * IntV.ofNat 8 + r5.item) * IntV.ofNat 128 + tl.arange 0 128).col 256).get (r, c, ())).toInt
      = (cRow d i r : Int)
    rw [Tensor.get_col, offs_row_val hd i hg ho r]
    unfold cRow
    push_cast
    ring
  have hcolv : ((((r7.item * IntV.ofNat 256 + tl.arange 0 256).row 128).get2 r c).toInt)
      = (cCol d i c : Int) := by
    change ((((r7.item * IntV.ofNat 256 + tl.arange 0 256).row 128).get (r, c, ())).toInt)
      = (cCol d i c : Int)
    rw [Tensor.get_row, offs_col_val hd i hpn c]
    unfold cCol
    push_cast
    ring
  have hfitrow : (d.stride_cm : Int) * (cRow d i r : Int) < 2 ^ 31 := by
    have h := c_fit_row hd hM
    have e : (d.stride_cm : Int) * (cRow d i r : Int) = ((cRow d i r * d.stride_cm : Nat) : Int) := by
      push_cast; ring
    rw [e]
    exact_mod_cast h
  have hfitcol : (d.stride_cn : Int) * (cCol d i c : Int) < 2 ^ 31 := by
    have h := c_fit_col hd hN
    have e : (d.stride_cn : Int) * (cCol d i c : Int) = ((cCol d i c * d.stride_cn : Nat) : Int) := by
      push_cast; ring
    rw [e]
    exact_mod_cast h
  have hlowrow : -(2 ^ 31 : Int) ≤ (d.stride_cm : Int) * (cRow d i r : Int) := by
    have h1 : (0 : Int) < (d.stride_cm : Int) := by exact_mod_cast hd.cm.1
    have h2 : (0 : Int) ≤ (cRow d i r : Int) := by exact_mod_cast Nat.zero_le _
    nlinarith
  have hlowcol : -(2 ^ 31 : Int) ≤ (d.stride_cn : Int) * (cCol d i c : Int) := by
    have h1 : (0 : Int) < (d.stride_cn : Int) := by exact_mod_cast hd.cn.1
    have h2 : (0 : Int) ≤ (cCol d i c : Int) := by exact_mod_cast Nat.zero_le _
    nlinarith
  have hA : (((IntV.ofNat d.stride_cm : I32) *
      (((r3.item * IntV.ofNat 8 + r5.item) * IntV.ofNat 128 + tl.arange 0 128).col 256)).get2 r c).toInt
      = (d.stride_cm : Int) * (cRow d i r : Int) := by
    rw [get2_smul,
      IntV.toInt_mul rfl rfl (by rw [hcm, hrowv]; exact s32 hlowrow hfitrow), hcm, hrowv]
  have hB : (((IntV.ofNat d.stride_cn : I32) *
      ((r7.item * IntV.ofNat 256 + tl.arange 0 256).row 128)).get2 r c).toInt
      = (d.stride_cn : Int) * (cCol d i c : Int) := by
    rw [get2_smul,
      IntV.toInt_mul rfl rfl (by rw [hcn, hcolv]; exact s32 hlowcol hfitcol), hcn, hcolv]
  have hsum : (((IntV.ofNat d.stride_cm : I32) *
        (((r3.item * IntV.ofNat 8 + r5.item) * IntV.ofNat 128 + tl.arange 0 128).col 256)).get2 r c).toInt +
      (((IntV.ofNat d.stride_cn : I32) *
        ((r7.item * IntV.ofNat 256 + tl.arange 0 256).row 128)).get2 r c).toInt
      = (cAdr d i r c : Int) := by
    rw [hA, hB]
    unfold cAdr
    push_cast
    ring
  rw [ptrAddT_get2, ptrAdd_get2, Ptr.offset_offset, hsum]
  exact Buf_ptr_zero_offset d.c _

end Claims.Matmul
