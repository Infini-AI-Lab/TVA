import TVA.Lang.Random
import TVA.Lemmas.Ops
import TVA.Lemmas.FloatErr

/-!
# Proof rules for random numbers

`tl.randint` is an exact function (`Philox.randint4x`), so it needs no rule. `tl.rand` rounds
twice: the conversion of the non-negative integer to `float32` (`tl.ItofRel`) and the IEEE
product with `Philox.scale32`. Below `2^53` in magnitude the alternatives of `tl.ItofRel .binary32`
coincide (`tl.itofRel_binary32`: the integer is exact in `float64`, `ofRat_int_f64`), so that
conversion is the one rounding to nearest even `FloatV.ofRat z`.
-/
namespace TVA

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}

/-- **`uint_to_uniform_float(x)`**: each element is the product of a conversion of
`Philox.toNonneg x` with `Philox.scale32`. -/
theorem safe_uint_to_uniform_float {x : Tensor s U32} {k : Tensor s F32 → Prog α}
    (hk : ∀ r : Tensor s F32, (∀ i, ∃ v, tl.ItofRel .binary32 (Philox.toNonneg (x.get i)).toInt v ∧
      FloatV.NaNEq (FloatV.mul v Philox.scale32) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.uint_to_uniform_float x >>= k) := by
  simp only [tl.uint_to_uniform_float, Prog.bind_eq, Prog.bind_assoc]
  refine safe_itof (Or.inr (Or.inr (Or.inl rfl))) fun xf hxf => safe_mul fun r hr =>
    hk r fun i => ⟨xf.get i, ?_, ?_⟩
  · simpa [Tensor.get_map] using hxf i
  · simpa [Tensor.get_full] using hr i

/-- **`tl.rand(seed, offset)`**: as `uint_to_uniform_float` of `tl.randint`. -/
theorem safe_rand {sg : Bool} {w : Nat} {sg' : Bool} {w' : Nat} {seed : IntV sg w}
    {offset : Tensor s (IntV sg' w')} {nRounds : Nat} {k : Tensor s F32 → Prog α}
    (hk : ∀ r : Tensor s F32, (∀ i, ∃ v,
      tl.ItofRel .binary32 (Philox.toNonneg (Philox.randint4x seed (offset.get i) nRounds).1).toInt v ∧
      FloatV.NaNEq (FloatV.mul v Philox.scale32) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.rand seed offset nRounds >>= k) :=
  safe_uint_to_uniform_float fun r hr => hk r fun i => by simpa [tl.randint, Tensor.get_map] using hr i

section Itof
open TVA.Math.Float TVA.Math.Float.Format

/-- `2^53` is below `float64`'s largest value. -/
theorem maxVal64_ge : (2 : ℚ) ^ 53 ≤ Format.binary64.maxVal := by
  have h : Format.binary64.maxVal = (2 ^ 53 - 1) * 2 ^ 971 := by
    rw [Format.maxVal_eq]; unfold Format.topSig Format.topExp Format.bias Format.binary64; norm_num
  rw [h]
  calc (2 : ℚ) ^ 53 ≤ (2 ^ 53 - 1) * 2 := by norm_num
    _ ≤ (2 ^ 53 - 1) * 2 ^ 971 :=
        mul_le_mul_of_nonneg_left (le_self_pow₀ (by norm_num) (by norm_num)) (by norm_num)

/-- An integer of magnitude below `2^53` converts to `float64` exactly. -/
theorem ofRat_int_f64 {z : ℤ} (hz : |z| < 2 ^ 53) :
    (FloatV.ofRat (z : ℚ) : F64).isFinite = true ∧ (FloatV.ofRat (z : ℚ) : F64).toRat = z := by
  have hf := Format.binary64_regular
  have hq : |(z : ℚ)| ≤ Format.binary64.maxVal := by
    have : |(z : ℚ)| < 2 ^ 53 := by exact_mod_cast hz
    exact this.le.trans maxVal64_ge
  unfold FloatV.ofRat
  by_cases h0 : (z : ℚ) = 0
  · obtain ⟨h1, h2, h3⟩ := roundRat_zero_spec (f := Format.binary64) (rnd := .rn) (sat := false) (zs := false) hf
    rw [h0]
    refine ⟨?_, ?_⟩
    · unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
    · rw [FloatV.toRat_ofPat h1, h3]
  obtain ⟨h1, h2, h3⟩ := roundRat_spec hf .rn false false h0 hq
  refine ⟨?_, ?_⟩
  · unfold FloatV.isFinite; rw [FloatV.pat_ofPat h1]; exact h2
  rw [FloatV.toRat_ofPat h1, h3]
  have ha : 0 < |(z : ℚ)| := abs_pos.mpr h0
  have hlog : Int.log 2 |(z : ℚ)| ≤ 52 := by
    have : |(z : ℚ)| < (2 : ℚ) ^ (53 : ℤ) := by exact_mod_cast hz
    have := log_lt_of_lt ha this
    omega
  have hQ : Format.binary64.qexp |(z : ℚ)| ≤ 0 := by
    rw [qexp_eq ha]
    have : Format.binary64.emin = -1022 := by decide
    have hm : (Format.binary64.m : ℤ) = 52 := rfl
    rw [this, hm]; omega
  set Q := Format.binary64.qexp |(z : ℚ)| with hQdef
  have hK : |(z : ℚ)| * (2 : ℚ) ^ (-Q) = ((z.natAbs * 2 ^ (-Q).toNat : ℕ) : ℚ) := by
    push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega)]
    congr 1
    rw [Nat.cast_natAbs, Int.cast_abs]
  rw [roundSig_exact .rn hK]
  by_cases hneg : (z : ℚ) < 0
  · rw [ite_eq_left hneg, abs_of_neg hneg]; ring
  · rw [ite_eq_right hneg, abs_of_nonneg (not_lt.mp hneg), one_mul]

/-- **The alternatives of `tl.ItofRel .binary32 z` coincide below `2^53`**: an integer of magnitude
below `2^53` is exact in `float64`, so its conversion through `float64` rounds once, as the direct
conversion does (library issue L154). -/
theorem tl.itofRel_binary32 {z : ℤ} (hz : |z| < 2 ^ 53) {r : F32} :
    tl.ItofRel .binary32 z r ↔ r = FloatV.ofRat (z : ℚ) := by
  constructor
  · intro h
    rcases h with h | ⟨-, h⟩ | ⟨hg, -⟩
    · exact h
    · rw [h]
      obtain ⟨hfin, hval⟩ := ofRat_int_f64 hz
      set y : F64 := FloatV.ofRat (z : ℚ) with hy
      unfold FloatV.cvt FloatV.ofRat
      rw [Format.cvt_of_finite _ _ _ hfin]
      congr 1
      have hval' : Format.binary64.toRat y.pat = z := hval
      rw [hval']
      by_cases h0 : (z : ℚ) = 0
      · rw [h0, roundRat_zero, roundRat_zero]
        have hs : Format.binary64.sign y.pat = false := by
          rw [hy]; unfold FloatV.ofRat; rw [h0, roundRat_zero]
          rw [FloatV.pat_ofPat (by
            obtain ⟨h1, -, -⟩ := roundRat_zero_spec (f := Format.binary64) (rnd := .rn) (sat := false)
              (zs := false) Format.binary64_regular
            rwa [roundRat_zero] at h1)]
          decide
        rw [hs]
      · rw [roundRat_of_ne h0, roundRat_of_ne h0]
    · rcases hg with hg | hg <;> exact absurd hg (by decide)
  · intro h; exact Or.inl h

end Itof

end TVA
