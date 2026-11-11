import Claim
import TVA.Proof
import Proof.Peel

/-!
# The `g_gamma` gate's index vector

`tl.convert(tl.arange(0, BT) + 1, float32)` (line 168) is the exact vector `1, …, 64`: an integer
below `2^24` converts to `float32` exactly.
-/

open TVA
namespace Claims.ChunkFwdO

section Itof
open Format

/-- `2^24` is below `float32`'s largest value. -/
private theorem maxVal32_ge : (2 : ℚ) ^ 24 ≤ Format.binary32.maxVal := by
  have h : Format.binary32.maxVal = (2 ^ 24 - 1) * 2 ^ 104 := by
    rw [Format.maxVal_eq]; unfold Format.topSig Format.topExp Format.bias Format.binary32; norm_num
  rw [h]
  calc (2 : ℚ) ^ 24 ≤ (2 ^ 24 - 1) * 2 := by norm_num
    _ ≤ (2 ^ 24 - 1) * 2 ^ 104 :=
        mul_le_mul_of_nonneg_left (le_self_pow₀ (by norm_num) (by norm_num)) (by norm_num)

/-- An integer of magnitude below `2^53` converts to `float64` exactly. Use it for an integer
that a kernel converts through `float64`. -/
theorem ofRat_int_f32 {z : ℤ} (hz : |z| < 2 ^ 24) :
    (FloatV.ofRat (z : ℚ) : F32).isFinite = true ∧ (FloatV.ofRat (z : ℚ) : F32).toRat = z := by
  have hf := Format.binary32_regular
  have hq : |(z : ℚ)| ≤ Format.binary32.maxVal := by
    have : |(z : ℚ)| < 2 ^ 24 := by exact_mod_cast hz
    exact this.le.trans maxVal32_ge
  unfold FloatV.ofRat
  by_cases h0 : (z : ℚ) = 0
  · obtain ⟨h1, h2, h3⟩ := roundRat_zero_spec (f := Format.binary32) (rnd := .rn) (sat := false) (zs := false) hf
    rw [h0]
    refine ⟨?_, ?_⟩
    · unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
    · rw [FloatV.toRat_ofPat h1, h3]
  obtain ⟨h1, h2, h3⟩ := roundRat_spec hf .rn false false h0 hq
  refine ⟨?_, ?_⟩
  · unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
  rw [FloatV.toRat_ofPat h1, h3]
  have ha : 0 < |(z : ℚ)| := abs_pos.mpr h0
  have hlog : Int.log 2 |(z : ℚ)| ≤ 23 := by
    have : |(z : ℚ)| < (2 : ℚ) ^ (24 : ℤ) := by exact_mod_cast hz
    have := log_lt_of_lt ha this
    omega
  have hQ : Format.binary32.qexp |(z : ℚ)| ≤ 0 := by
    rw [qexp_eq ha]
    have : Format.binary32.emin = -126 := by decide
    have hm : (Format.binary32.m : ℤ) = 23 := rfl
    rw [this, hm]; omega
  set Q := Format.binary32.qexp |(z : ℚ)| with hQdef
  have hK : |(z : ℚ)| * (2 : ℚ) ^ (-Q) = ((z.natAbs * 2 ^ (-Q).toNat : ℕ) : ℚ) := by
    push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega)]
    congr 1
    rw [Nat.cast_natAbs, Int.cast_abs]
  rw [roundSig_exact .rn hK]
  by_cases hneg : (z : ℚ) < 0
  · rw [ite_eq_left hneg, abs_of_neg hneg]; ring
  · rw [ite_eq_right hneg, abs_of_nonneg (not_lt.mp hneg), one_mul]


end Itof

/-- An integer below `2^24` converts to `float32` exactly, by either alternative of `ItofRel`. -/
theorem itof32_exact {z : ℤ} (hz : |z| < 2 ^ 24) {r : F32} (h : tl.ItofRel .binary32 z r) :
    r.isFinite = true ∧ r.toRat = z := by
  have hz' : |z| < 2 ^ 53 := lt_trans hz (by norm_num)
  rw [tl.itofRel_binary32 hz'] at h
  rw [h]; exact ofRat_int_f32 hz

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **The index vector** `tl.convert(tl.arange(0, BT) + 1, float32)`: lane `i` is finite with value
`i + 1`. -/
theorem safe_gamma_idx {k : Tensor [BT] F32 → Prog α}
    (hk : ∀ nf : Tensor [BT] F32, (∀ i : Fin BT, (nf.get1 i).isFinite = true ∧
      (nf.get1 i).toRat = ((i : Nat) + 1 : ℚ)) → Safe E R W Q m (k nf)) :
    Safe E R W Q m (tl.convert .i32 .f32 ((tl.arange 0 BT : Tensor [BT] I32) + (IntV.ofNat 1 : I32))
      >>= k) := by
  have hval : ∀ i : Fin BT, (((tl.arange 0 BT : Tensor [BT] I32) + (IntV.ofNat 1 : I32)).get1 i).toInt =
      (((i : Nat) + 1 : Nat) : Int) := by
    intro i
    rw [Tensor.get1_adds]
    exact IntV.add_nat (IntV.I32.toInt_tl_arange (by norm_num [BT]) i)
      (IntV.I32.toInt_ofNat (n := 1) (by norm_num)) (by have := i.isLt; simp [BT] at *; omega)
  refine safe_convert (fun i => ⟨FloatV.ofRat (((i.1.val + 1 : Nat) : Int) : ℚ), ?_⟩) fun r hr =>
    hk r fun i => ?_
  · show tl.ItofRel .binary32 _ _
    rw [show (((tl.arange 0 BT : Tensor [BT] I32) + (IntV.ofNat 1 : I32)).get i).toInt = _ from
      hval i.1]
    exact Or.inl rfl
  · have h := hr (i, ())
    change tl.ItofRel .binary32 _ _ at h
    rw [show (((tl.arange 0 BT : Tensor [BT] I32) + (IntV.ofNat 1 : I32)).get (i, ())).toInt = _ from
      hval i] at h
    obtain ⟨hf, hv⟩ := itof32_exact (by have := i.isLt; simp only [BT] at this; rw [abs_of_nonneg (by positivity)]; push_cast; omega) h
    refine ⟨hf, ?_⟩
    rw [show r.get1 i = r.get (i, ()) from rfl, hv]
    push_cast; ring

end Claims.ChunkFwdO
