import Proof.LeafAux
import Proof.RoundBound
import Proof.Lanes

/-! The facts a row's leaf needs about individual lanes: reading its own lane of a block, the
mask's value at a lane, the masked constant's value and finiteness, and the finiteness and
magnitude of the `float32` sum of two covered `bfloat16` values. -/

namespace Claims.LayerNormFwd

open TVA
open Format

noncomputable section


/-! ### Reading a program's own lane -/

/-- **A block read at row `t`'s own lane**: the read hypothesis of a `[32, BD]` load/store holds
at the lane `(t % 32, j)` of program `t / 32`. -/
theorem mat_read {d : Inst} {p t j : Nat} (hp : t / 32 = p) (ht : t < d.T) (hj : j < d.D)
    (hjBD : j < d.BD) {m : Mem} {dty : DType} {B : Buf dty}
    {val : Idx [32, d.BD] → dty.Val}
    (h : ∀ ix : Idx [32, d.BD],
      decide (32 * p + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D) = true →
      m.read dty (B.ptr (↑(idx d (32 * p + (ix.1 : Nat)) (ix.2.1 : Nat)) : Int)).addr
        = some (val ix)) :
    m.read dty (B.ptr (↑(idx d t j) : Int)).addr
      = some (val ⟨⟨t % 32, Nat.mod_lt _ (by norm_num)⟩, ⟨j, hjBD⟩, ()⟩) := by
  have h32 : 32 * p + t % 32 = t := by
    rw [← hp]
    exact Nat.div_add_mod t 32
  have hen : decide (32 * p + t % 32 < d.T ∧ j < d.D) = true := by
    rw [h32]
    exact decide_eq_true ⟨ht, hj⟩
  have := h ⟨⟨t % 32, Nat.mod_lt _ (by norm_num)⟩, ⟨j, hjBD⟩, ()⟩ hen
  simpa [h32] using this

/-- **A row read at row `t`'s own lane**: same for a `[32]` load/store at the row's address. -/
theorem row_read {d : Inst} {p t : Nat} (hp : t / 32 = p) (ht : t < d.T) {m : Mem}
    {dty : DType} {B : Buf dty} {val : Idx [32] → dty.Val}
    (h : ∀ ix : Idx [32], decide (32 * p + (ix.1 : Nat) < d.T) = true →
      m.read dty (B.ptr ((32 * p + (ix.1 : Nat) : Nat) : Int)).addr = some (val ix)) :
    m.read dty (B.ptr (t : Int)).addr
      = some (val ⟨⟨t % 32, Nat.mod_lt _ (by norm_num)⟩, ()⟩) := by
  have h32 : 32 * p + t % 32 = t := by
    rw [← hp]
    exact Nat.div_add_mod t 32
  have hen : decide (32 * p + t % 32 < d.T) = true := by
    rw [h32]
    exact decide_eq_true ht
  have := h ⟨⟨t % 32, Nat.mod_lt _ (by norm_num)⟩, ()⟩ hen
  simpa [h32] using this

/-! ### The mask and the masked constant -/

/-- **The mask's value at a lane**: lane `j` is enabled exactly when `j < D`. -/
theorem mask_lane {d : Inst} (hD : d.D < 2 ^ 31) (hBD : d.BD < 2 ^ 31) (r : Fin 32)
    (j : Fin d.BD) :
    (((tl.arange 0 d.BD).ltS (IntV.ofNat d.D)).row 32).get2 r j
      = decide ((j : Nat) < d.D) := by
  rw [Tensor.get2_row, get1_ltS, IntV.lt_eq]
  simp only [tl.arange, Tensor.get1_arange, Nat.sub_zero]
  refine decide_eq_decide.mpr ?_
  rw [toInt_ofInt32' (by omega) (by have := j.isLt; omega), toInt_ofNat32' hD]
  omega

/-- **`tl.where` at a lane.** -/
theorem where_lane {α : Type} {m n : Nat} (c : Tensor [m, n] Bool) (x y : Tensor [m, n] α)
    (i : Fin m) (j : Fin n) :
    (tl.where c x y).get2 i j = if c.get2 i j then x.get2 i j else y.get2 i j := rfl

/-- The element of a broadcast constant. -/
theorem bc_get {α : Type} {s : Shape} (a : α) (i : Idx s) : (bc a : Tensor s α).get i = a := rfl

/-- **The masked lane's constant is the `float32` zero.** -/
theorem bc_zero_get2 {m n : Nat} (i : Fin m) (j : Fin n) :
    (bc (FloatV.litF32 (FloatV.ofPat (0 : ℕ) : F64)) : Tensor [m, n] F32).get2 i j
      = (FloatV.ofPat (0 : ℕ) : F32) := by
  have h : (bc (FloatV.litF32 (FloatV.ofPat (0 : ℕ) : F64)) : Tensor [m, n] F32).get2 i j
      = FloatV.litF32 (FloatV.ofPat (0 : ℕ) : F64) := rfl
  rw [h, FloatV.litF32_ofPat_zero]

/-- Every lane of a broadcast float32 zero literal has real value 0. -/
theorem bc_zero_toReal {m n : Nat} (i : Fin m) (j : Fin n) :
    ((bc (FloatV.litF32 (FloatV.ofPat (0 : ℕ) : F64)) : Tensor [m, n] F32).get2 i j).toReal
      = 0 := by
  rw [bc_zero_get2, FloatV.toReal_ofPat_zero]

/-- Every lane of a broadcast float32 zero literal is finite. -/
theorem bc_zero_isFinite {m n : Nat} (i : Fin m) (j : Fin n) :
    ((bc (FloatV.litF32 (FloatV.ofPat (0 : ℕ) : F64)) : Tensor [m, n] F32).get2 i j).isFinite
      = true := by
  rw [bc_zero_get2]
  exact FloatV.isFinite_ofPat_zero Format.binary32_regular

/-- A finite value is not NaN. -/
theorem isNaN_eq_false_of_isFinite {f : Format} {x : FloatV f} (h : x.isFinite = true) :
    x.isNaN = false := by
  have h' : f.isFinite x.pat = true := h
  unfold Format.isFinite at h'
  simp only [Bool.and_eq_true, Bool.not_eq_true'] at h'
  exact h'.1

/-- **A `NaNEq` whose left side is finite is an equality.** -/
theorem naneq_eq_of_isFinite {f : Format} {x y : FloatV f} (h : FloatV.NaNEq x y)
    (hx : x.isFinite = true) : y = x :=
  FloatV.NaNEq.eq h (isNaN_eq_false_of_isFinite hx)

/-- A `float32` whose real value is zero has `toRat` zero. -/
theorem FloatV.toRat_eq_zero_of_toReal {f : Format} {x : FloatV f} (h : x.toReal = 0) :
    x.toRat = 0 := by
  unfold FloatV.toReal at h
  exact_mod_cast h

/-! ### Conversions and the `float32` sum of two covered inputs -/

/-- `2^57` is a finite `bfloat16`. -/
theorem two_pow_57_le_bfloat16_maxVal : (2 : ℚ) ^ 57 ≤ Format.bfloat16.maxVal := by
  rw [Format.bfloat16_maxVal]; norm_num

/-- **`bfloat16 → float32` is exact** on finite values (the conversion is a widening). -/
theorem cvt_bf16_f32_spec {x : BF16} (hx : x.isFinite = true) :
    (FloatV.cvt x : F32).isFinite = true ∧ (FloatV.cvt x : F32).toRat = x.toRat := by
  simpa only [FloatV.cvtWith_rn] using
    FloatV.toRat_cvt_of_wider Format.bfloat16_regular Format.binary32_regular
      Format.wider_bfloat16_binary32 .rn hx

/-- **A `float32 → bfloat16` conversion of a value below `2^57` is finite.** -/
theorem cvt_f32_bf16_fin {x : F32} (hx : x.isFinite = true)
    (h : |x.toRat| ≤ (2 : ℚ) ^ 57) : (FloatV.cvt x : BF16).isFinite = true :=
  FloatV.isFinite_cvt Format.bfloat16_regular hx (h.trans two_pow_57_le_bfloat16_maxVal)

/-- **The `float32` sum of two widened covered inputs is finite.** -/
theorem add_cvt_isFinite {a b : BF16} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hab : |a.toRat| + |b.toRat| ≤ (2 : ℚ) ^ 57) :
    (FloatV.add (FloatV.cvt a) (FloatV.cvt b) : F32).isFinite = true := by
  obtain ⟨haf, har⟩ := cvt_bf16_f32_spec ha
  obtain ⟨hbf, hbr⟩ := cvt_bf16_f32_spec hb
  refine FloatV.isFinite_add Format.binary32_regular haf hbf ?_
  have h1 : |(FloatV.cvt a : F32).toRat + (FloatV.cvt b : F32).toRat|
      ≤ |a.toRat| + |b.toRat| := by
    rw [har, hbr]; exact abs_add_le _ _
  exact h1.trans (hab.trans two_pow_57_le_binary32_maxVal)

/-- **The `float32` sum of two widened covered inputs is below `2^57`.** -/
theorem add_cvt_abs_le {a b : BF16} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (hab : |a.toRat| + |b.toRat| ≤ (2 : ℚ) ^ 57) :
    |(FloatV.add (FloatV.cvt a) (FloatV.cvt b) : F32).toRat| ≤ (2 : ℚ) ^ 57 := by
  obtain ⟨haf, har⟩ := cvt_bf16_f32_spec ha
  obtain ⟨hbf, hbr⟩ := cvt_bf16_f32_spec hb
  have h1 : |(FloatV.cvt a : F32).toRat + (FloatV.cvt b : F32).toRat| ≤ (2 : ℚ) ^ 57 := by
    refine (abs_add_le _ _).trans ?_
    rw [har, hbr]; exact hab
  have hs : |(FloatV.cvt a : F32).toRat + (FloatV.cvt b : F32).toRat| ≤ (2 : ℚ) ^ 57 := h1
  have hmax : |(FloatV.cvt a : F32).toRat + (FloatV.cvt b : F32).toRat|
      ≤ Format.binary32.maxVal := hs.trans two_pow_57_le_binary32_maxVal
  unfold FloatV.add FloatV.toRat
  rw [Format.add_of_finite .rn haf hbf]
  rw [FloatV.pat_ofPat
    (Format.roundRat_lt (f := Format.binary32)
      (q := Format.binary32.toRat (FloatV.cvt a).pat + Format.binary32.toRat (FloatV.cvt b).pat)
      Format.binary32_regular .rn false _ hmax)]
  exact Format.abs_roundRat_le_pow2 Format.binary32_regular .rn false _
    two_pow_57_le_binary32_maxVal binary32_emin_le_57 hs

/-- **The `float32` sum of two widened zero inputs is zero.** -/
theorem add_cvt_zero {a b : BF16} (ha : a.isFinite = true) (hb : b.isFinite = true)
    (ha0 : a.toRat = 0) (hb0 : b.toRat = 0) :
    (FloatV.add (FloatV.cvt a) (FloatV.cvt b) : F32).toRat = 0 := by
  obtain ⟨haf, har⟩ := cvt_bf16_f32_spec ha
  obtain ⟨hbf, hbr⟩ := cvt_bf16_f32_spec hb
  exact FloatV.toRat_add_of_zero Format.binary32_regular haf hbf (by rw [har, ha0])
    (by rw [hbr, hb0])

end

end Claims.LayerNormFwd
