import Claim
import TVA.Proof

/-!
# Helpers for the scalar part of the walk

Peeling a `tl.convert`, `tl.idiv`, `tl.irem` of scalars of known value, and facts about the
`+0` patterns (`bfloat16`, `float32`) that masked loads and `tl.where` use.
-/

open TVA
namespace Claims.ChunkFwdO

/-- Two signed integer values with the same `toInt` are equal. -/
theorem IntV.eq_of_toInt_signed {w : Nat} {a b : IntV true w} (h : a.toInt = b.toInt) : a = b := by
  have hb : a.bits = b.bits := BitVec.toInt_inj.mp h
  cases a; cases b; simp_all

/-- A rank-zero tensor equals the scalar tensor of its item. -/
theorem tensor_scalar_item {α : Type} (r : Tensor [] α) : r = Tensor.scalar r.item := by
  ext j; rfl

/-- `<natural> : SFits w` for a natural below `2^(w-1)`. -/
theorem sfits_nat {w : Nat} (_hw : 0 < w) {n : Nat} (hn : (n : ℤ) < 2 ^ (w - 1)) :
    IntV.SFits w (n : ℤ) := by
  constructor
  · have : (0 : ℤ) ≤ (n : ℤ) := Int.natCast_nonneg n
    have : (0 : ℤ) ≤ 2 ^ (w - 1) := by positivity
    linarith
  · exact hn

/-- The one result of `tl.convert .i32 .i64` of a scalar is the sign extension of its value. -/
theorem conv_scalar_iff {E : Env} (n : I32) (i : Idx []) (r : I64) :
    ConvRel E.P .i32 .i64 ((Tensor.scalar n).get i) r ↔
      r = (Tensor.scalar (IntV.ofInt n.toInt) : Tensor [] I64).get i := by
  rw [ConvRel.iff_cast_i32_i64]
  exact Iff.of_eq (congrArg (r = ·) (IntV.cast_eq_ofInt _))

/-- Peel `tl.convert .i32 .i64` of a scalar. -/
theorem safe_conv_i32_i64_scalar {E : Env} {R W : Set ℤ} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    (n : I32) {k : Tensor [] I64 → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt n.toInt)))) :
    Safe E R W Q m (tl.convert .i32 .i64 (Tensor.scalar n) >>= k) :=
  (safe_convert_iff (conv_scalar_iff n)).2 hk

/-- `tl.idiv` of two scalars with known values. -/
theorem safe_idiv_val {w : Nat} {E : Env} {R W : Set ℤ} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {x y : IntV true w} (a b : Nat) (hw : 0 < w) (hx : x.toInt = (a : ℤ)) (hy : y.toInt = (b : ℤ))
    (hb : 0 < b) (hq : ((a / b : Nat) : ℤ) < 2 ^ (w - 1))
    {k : Tensor [] (IntV true w) → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt ((a / b : Nat) : ℤ))))) :
    Safe E R W Q m (tl.idiv (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  have ha0 : 0 ≤ x.toInt := by rw [hx]; exact Int.natCast_nonneg a
  have hb0 : 0 < y.toInt := by rw [hy]; exact_mod_cast hb
  refine safe_idiv1 ha0 hb0 ?_
  intro r hr
  have h1 : r.item = IntV.ofInt ((a / b : Nat) : ℤ) := by
    apply IntV.eq_of_toInt_signed
    rw [hr, hx, hy, IntV.toInt_ofInt hw (sfits_nat hw hq), Int.natCast_ediv]
  rw [tensor_scalar_item r, h1]; exact hk

/-- `tl.irem` of two scalars with known values. -/
theorem safe_irem_val {w : Nat} {E : Env} {R W : Set ℤ} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {x y : IntV true w} (a b : Nat) (hw : 0 < w) (hx : x.toInt = (a : ℤ)) (hy : y.toInt = (b : ℤ))
    (hb : 0 < b) (hq : ((a % b : Nat) : ℤ) < 2 ^ (w - 1))
    {k : Tensor [] (IntV true w) → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt ((a % b : Nat) : ℤ))))) :
    Safe E R W Q m (tl.irem (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  have ha0 : 0 ≤ x.toInt := by rw [hx]; exact Int.natCast_nonneg a
  have hb0 : 0 < y.toInt := by rw [hy]; exact_mod_cast hb
  refine safe_irem1 ha0 hb0 ?_
  intro r hr
  have h1 : r.item = IntV.ofInt ((a % b : Nat) : ℤ) := by
    apply IntV.eq_of_toInt_signed
    rw [hr, hx, hy, IntV.toInt_ofInt hw (sfits_nat hw hq), Int.natCast_emod]
  rw [tensor_scalar_item r, h1]; exact hk

/-! ## The `+0` patterns -/

theorem f32_zero_isFinite : (FloatV.ofPat 0 : F32).isFinite = true := by
  rw [FloatV.isFinite_ofPat (Nat.two_pow_pos 32)]
  exact Format.isFinite_zero Format.binary32_regular

/-- The `bfloat16` `+0` pattern is finite. -/
theorem bf16_zero_isFinite : (FloatV.ofPat 0 : BF16).isFinite = true := by
  rw [FloatV.isFinite_ofPat (Nat.two_pow_pos 16)]
  exact Format.isFinite_zero Format.bfloat16_regular

/-- The `float32` `+0` pattern has rational value `0`. -/
theorem f32_zero_toRat : (FloatV.ofPat 0 : F32).toRat = 0 :=
  FloatV.toRat_of_isZero (x := (FloatV.ofPat 0 : F32)) Format.isZero_zero

/-- The `bfloat16` `+0` pattern has rational value `0`. -/
theorem bf16_zero_toRat : (FloatV.ofPat 0 : BF16).toRat = 0 :=
  FloatV.toRat_of_isZero (x := (FloatV.ofPat 0 : BF16)) Format.isZero_zero

/-- `zf32` is the `float32` `+0` pattern. -/
theorem zf32_eq : zf32 = (FloatV.ofPat 0 : F32) := by
  unfold zf32 FloatV.litF32
  have h : (FloatV.ofPat 0 : F64).isZero = true := by
    rw [FloatV.isZero_ofPat (Nat.two_pow_pos _)]
    exact Format.isZero_zero
  simp [h]

/-- Converting `binary32` pattern `0` to `bfloat16` (round to nearest) gives `0`. -/
theorem bf16_cvt_zero_fmt : Format.binary32.cvt Format.bfloat16 .rn false 0 = 0 := by
  rw [Format.cvt_of_finite (g := .bfloat16) (rnd := .rn) (sat := false) (hx := by rfl)]
  have h0 : Format.binary32.toRat 0 = 0 := Format.toRat_of_isZero Format.isZero_zero
  rw [h0, Format.roundRat_zero]
  show (if Format.binary32.sign 0 then Format.bfloat16.signBit else 0) +
    Format.bfloat16.mag 0 = 0
  rfl

/-- Converting the `float32` `+0` to `bfloat16` gives the `bfloat16` `+0`. -/
theorem bf16_cvt_zero : FloatV.cvt (g := .bfloat16) (FloatV.ofPat 0 : F32) = (FloatV.ofPat 0 : BF16) := by
  show (FloatV.ofPat (Format.binary32.cvt Format.bfloat16 .rn false 0) : BF16) = FloatV.ofPat 0
  rw [bf16_cvt_zero_fmt]

/-- Converting the `float32` `+0` to `bfloat16` is not a NaN. -/
theorem bf16_cvt_zero_notNaN :
    (FloatV.cvt (g := .bfloat16) (FloatV.ofPat 0 : F32)).isNaN = false := by
  show (FloatV.ofPat (Format.binary32.cvt Format.bfloat16 .rn false 0) : BF16).isNaN = false
  rw [bf16_cvt_zero_fmt]
  rfl

/-- `tl.convert .f32 .bf16` sends `+0` to `+0`. -/
theorem conv_bf16_zero {P : Platform} {r : BF16}
    (h : ConvRel P .f32 .bf16 (FloatV.ofPat 0) r) : r = FloatV.ofPat 0 := by
  have hF : FloatRel P .binary32 .bfloat16 (FloatV.ofPat 0) r := h
  unfold FloatRel at hF
  simp only [show ¬ (Format.binary32 = Format.bfloat16) from by decide, ↓reduceIte] at hF
  rcases hF with ⟨-, -, h1⟩ | ⟨-, h2, -⟩ | ⟨h3, -⟩
  · exact (h1.eq bf16_cvt_zero_notNaN).trans bf16_cvt_zero
  · exact absurd h2 (by
      intro hh; rcases hh with hh | hh <;> exact absurd hh (by decide))
  · exact absurd h3 (by
      intro hh; rcases hh with hh | hh <;> exact absurd hh (by decide))

end Claims.ChunkFwdO
