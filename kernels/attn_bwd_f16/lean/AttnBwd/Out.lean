import AttnBwd.Dq

/-!
# `attn_bwd_f16`: from the accumulators to `OutSpec`

* `ln2_val`: the `float32` constant `LN2` is `1453635/2097152` (`0x3f317218`).
* `vanish`: the per-pair quantities of `Spec` are `0` on pairs that are not `Allowed`, so a sum
  over the rows a program covered is the sum over all rows (`sum_ext_range`).
* `out_dv`, `out_scaled`: the stored `float16` element satisfies `OutSpec`'s condition.
-/

namespace Kernels.AttnBwd

open TVA

/-! ## `LN2` -/

theorem ln2_val : ln2F.toRat = 1453635 / 2097152 := by decide +kernel

theorem ln2_fin : ln2F.isFinite = true := by decide +kernel

theorem ln2_real : ln2F.toReal = 1453635 / 2097152 := by
  rw [toReal_def, ln2_val]; push_cast; ring

/-! ## Pairs that are not `Allowed` -/

section Vanish
variable {d : Inst} {hv : HeadV} {n m : Nat}

theorem vanish (h : ¬ Allowed d n m) :
    pE d hv n m = 0 ∧ p16Err d hv n m = 0 ∧ p16B d hv n m = 0 ∧ dsE d hv n m = 0 ∧
      ds16Err d hv n m = 0 ∧ ds16B d hv n m = 0 := by
  simp only [pE, p16Err, p16B, dsE, ds16Err, ds16B, if_neg h, zero_mul, and_self]

theorem sum_ext_range {C : Finset ℕ} {N : ℕ} (hC : C ⊆ Finset.range N) (f : ℕ → ℝ)
    (hf : ∀ x, x < N → x ∉ C → f x = 0) : ∑ x ∈ C, f x = ∑ x ∈ Finset.range N, f x :=
  Finset.sum_subset hC fun x hx hn => hf x (Finset.mem_range.mp hx) hn

theorem two_xE_le (hd : d.ok) (n m : Nat) : (2 : ℝ) ^ xE d hv n m ≤ pBr d hv n m := by
  unfold pBr
  have h0 : 0 ≤ eX d hv n m := eSub_nn hd (aQK_nn n m)
  have h1 := two_rpow_le (show xE d hv n m ≤ xE d hv n m + eX d hv n m by linarith)
  have h2 := two_rpow_pos (xE d hv n m + eX d hv n m)
  have : (0 : ℝ) ≤ 2 ^ (-22 : ℤ) * 2 ^ (xE d hv n m + eX d hv n m) := by positivity
  have : (0 : ℝ) < 2 ^ (-126 : ℤ) := by positivity
  nlinarith

theorem dsE_abs (hd : d.ok) (hok : hv.Ok d) (hn : n < d.N_CTX) (hm : m < d.N_CTX) :
    |dsE d hv n m| ≤ 2 ^ 15 := by
  unfold dsE pE
  split_ifs with hal
  · obtain ⟨-, h2⟩ := hok.fits n m hn hm hal
    have h1 := two_xE_le (hv := hv) hd n m
    have hp := two_rpow_pos (xE d hv n m)
    have hy : |yE d hv n m| ≤ yB d hv n m := by
      unfold yB; have := eSub_nn (s := sVD d hv n m) (x := (hv.dl m).toReal) hd (aVD_nn (d := d) (hv := hv) n m)
      unfold eY; linarith
    rw [abs_mul, abs_of_pos hp]
    have := mul_le_mul h1 hy (abs_nonneg _) (by linarith)
    linarith
  · simp

theorem ds16Err_bd (hd : d.ok) (hok : hv.Ok d) (hn : n < d.N_CTX) (hm : m < d.N_CTX) :
    0 ≤ ds16Err d hv n m ∧ ds16Err d hv n m ≤ 2 ^ 17 := by
  unfold ds16Err
  split_ifs with hal
  · obtain ⟨-, h2⟩ := hok.fits n m hn hm hal
    obtain ⟨hb0, hb1⟩ := dsB_bd hd hok hn hm hal
    have h1 := two_xE_le (hv := hv) hd n m
    have hp := two_rpow_pos (xE d hv n m)
    have hy0 := yB_nn (hv := hv) hd n m
    have heY0 : 0 ≤ eY d hv n m := eSub_nn hd (aVD_nn n m)
    have heY : eY d hv n m ≤ yB d hv n m := by unfold yB; have := abs_nonneg (yE d hv n m); linarith
    have hpB0 : 0 ≤ pBr d hv n m := (pBr_le hok hn hm hal).1
    have hu32 : u32 = 2 ^ (-24 : ℤ) := u32_eq
    have heta32 : eta32 = 2 ^ (-150 : ℤ) := eta32_eq
    have hu16 : u16 = 2 ^ (-11 : ℤ) := u16_eq
    have heta16 : eta16 = 2 ^ (-25 : ℤ) := eta16_eq
    unfold dsErr
    have e1 : (pBr d hv n m - 2 ^ xE d hv n m) * yB d hv n m ≤ pBr d hv n m * yB d hv n m :=
      mul_le_mul_of_nonneg_right (by linarith) hy0
    have e1' : 0 ≤ (pBr d hv n m - 2 ^ xE d hv n m) * yB d hv n m := mul_nonneg (by linarith) hy0
    have e2 : 2 ^ xE d hv n m * eY d hv n m ≤ pBr d hv n m * yB d hv n m := mul_le_mul h1 heY heY0 hpB0
    have e2' : 0 ≤ 2 ^ xE d hv n m * eY d hv n m := mul_nonneg hp.le heY0
    have hpy0 : 0 ≤ pBr d hv n m * yB d hv n m := mul_nonneg hpB0 hy0
    have e3 : u32 * (pBr d hv n m * yB d hv n m) ≤ 2 ^ (-24 : ℤ) * 2 ^ 15 := by
      rw [hu32]; exact mul_le_mul_of_nonneg_left h2 (by positivity)
    have e4 : u16 * dsB d hv n m ≤ 2 ^ (-11 : ℤ) * 2 ^ 16 := by
      rw [hu16]; exact mul_le_mul_of_nonneg_left hb1 (by positivity)
    have : (0 : ℝ) ≤ u32 * (pBr d hv n m * yB d hv n m) := by rw [hu32]; positivity
    have : (0 : ℝ) ≤ u16 * dsB d hv n m := by rw [hu16]; positivity
    have c1 : (2 : ℝ) ^ 15 + 2 ^ 15 + 2 ^ (-24 : ℤ) * 2 ^ 15 + 2 ^ (-150 : ℤ) + 2 ^ (-11 : ℤ) * 2 ^ 16 +
        2 ^ (-25 : ℤ) ≤ 2 ^ 17 := by norm_num
    have : (0 : ℝ) ≤ 2 ^ (-150 : ℤ) := by positivity
    have : (0 : ℝ) ≤ 2 ^ (-25 : ℤ) := by positivity
    rw [heta32, heta16] at *
    constructor <;> linarith
  · exact ⟨le_rfl, by norm_num⟩

end Vanish

/-! ## `OutSpec` from the accumulators -/

/-- **`dv`** (lines 445–446): the accumulator after the last step, rounded to `float16`. -/
theorem out_dv {d : Inst} {hv : HeadV} {n j : Nat} {C : Finset ℕ} {a : F32} {c16 : F16}
    (hacc : AccI a (∑ m ∈ C, pE d hv n m * (hv.dO m j).toReal)
      (∑ m ∈ C, p16Err d hv n m * |(hv.dO m j).toReal|) (∑ m ∈ C, p16B d hv n m * |(hv.dO m j).toReal|)
      (stepsKV d n))
    (hC : C ⊆ Finset.range d.N_CTX) (hvan : ∀ m, m < d.N_CTX → m ∉ C → ¬ Allowed d n m)
    (hc : FloatV.NaNEq (FloatV.cvt a) c16) :
    ∃ v : F32, v.isFinite = true ∧ |v.toReal - dVE d hv n j| ≤ dVTol d hv n j ∧
      FloatV.NaNEq (FloatV.cvt v) c16 ∧
      (|dVE d hv n j| + dVTol d hv n j ≤ 65504 →
        c16.isFinite = true ∧ |c16.toReal - dVE d hv n j| ≤ dVTol d hv n j + (|dVE d hv n j| + dVTol d hv n j) * u16 + eta16) := by
  rw [sum_ext_range hC _ (fun m hm hn => by rw [(vanish (hv := hv) (hvan m hm hn)).1, zero_mul]),
    sum_ext_range hC _ (fun m hm hn => by rw [(vanish (hv := hv) (hvan m hm hn)).2.1, zero_mul]),
    sum_ext_range hC _ (fun m hm hn => by rw [(vanish (hv := hv) (hvan m hm hn)).2.2.1, zero_mul])] at hacc
  obtain ⟨hf, he⟩ := accI_final hacc
  exact ⟨a, hf, he, hc, fun hroom => out16 hf he hc hroom⟩

/-- **`dk`** and **`dq`** (lines 449–451, 498–500): the accumulator after the last step, times a
finite scale `s` with `|s|·(|S| + Acc) ≤ 2^126`, rounded to `float16`. -/
theorem out_scaled {a s v : F32} {c16 : F16} {S Er Bm : ℝ} {T : ℕ}
    (hacc : AccI a S Er Bm T) (hs : s.isFinite = true)
    (hsize : |s.toReal| * (|S| + (Er + accB T Bm)) ≤ 2 ^ 126)
    (hv : FloatV.NaNEq (FloatV.mul a s) v) (hc : FloatV.NaNEq (FloatV.cvt v) c16) {E Tl : ℝ}
    (hE : E = s.toReal * S)
    (hTl : Tl = |s.toReal| * (Er + accB T Bm) + u32 * (|s.toReal| * (|S| + (Er + accB T Bm))) + eta32) :
    ∃ v : F32, v.isFinite = true ∧ |v.toReal - E| ≤ Tl ∧ FloatV.NaNEq (FloatV.cvt v) c16 ∧
      (|E| + Tl ≤ 65504 → c16.isFinite = true ∧ |c16.toReal - E| ≤ Tl + (|E| + Tl) * u16 + eta16) := by
  obtain ⟨hf, he⟩ := accI_final hacc
  obtain ⟨hvf, hve⟩ := scale_out hf hs he hsize hv
  rw [← hE, ← hTl] at hve
  exact ⟨v, hvf, hve, hc, fun hroom => out16 hvf hve hc hroom⟩

end Kernels.AttnBwd
