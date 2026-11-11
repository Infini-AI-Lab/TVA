import Proof.Pieces
import TVA.Proof

/-!
# `qo_indptr`'s integer type

The values built from `qo_indptr` are `int32` or `int64` (`QI`). `castQ` and `toQv` are the
promotion `toQ` models; `safe_toQ` reads the monadic `toQ` as the pure value `toQv`; the `toInt`
lemmas hold for either width.
-/

namespace Claims.ExtendAttn

open TVA

/-- The promotion of an `int32` to `qo_indptr`'s type, as a value. -/
def castQ : (qo32 : Bool) → I32 → QI qo32
  | true, x => x
  | false, x => IntV.cast x

/-- The promotion of an `int32` tensor to `qo_indptr`'s type, as a value. -/
def toQv {s : Shape} : (qo32 : Bool) → Tensor s I32 → Tensor s (QI qo32)
  | true, x => x
  | false, x => x.castInt

/-- Entry `i` of the query-width cast of a tensor is the cast of entry `i`. -/
theorem toQv_get {s : Shape} (qo32 : Bool) (x : Tensor s I32) (i : Idx s) :
    (toQv qo32 x).get i = castQ qo32 (x.get i) := by
  cases qo32 <;> rfl

/-- The query-width cast preserves the integer value. -/
theorem castQ_toInt (qo32 : Bool) (a : I32) : (castQ qo32 a).toInt = a.toInt := by
  cases qo32
  · exact IntV.toInt_cast a (by decide)
  · rfl

/-- The width of `qo_indptr`'s type is at least 32. -/
theorem qoW_ge (qo32 : Bool) : 32 ≤ qoW qo32 := by cases qo32 <;> simp [qoW]

/-- A natural number below `2^31` fits `qo_indptr`'s type. -/
theorem lt_pow_qoW {n : Nat} (qo32 : Bool) (h : n < 2 ^ 31) : n < 2 ^ (qoW qo32 - 1) :=
  lt_of_lt_of_le h (Nat.pow_le_pow_right (by norm_num) (by have := qoW_ge qo32; omega))

section
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}

/-- `toQ` is a pure promotion: the rest runs with `toQv`. -/
theorem safe_toQ (qo32 : Bool) {x : Tensor s I32} {k : Tensor s (QI qo32) → Prog α} :
    Safe E R W Q m (toQ qo32 x >>= k) ↔ Safe E R W Q m (k (toQv qo32 x)) := by
  cases qo32
  · exact safe_convert_iff fun i r => ConvRel.iff_cast_i32_i64
  · exact Iff.rfl

/-- `x.to(tl.int64)` of an `int32` tensor is the sign extension. -/
theorem safe_conv64 {x : Tensor s I32} {k : Tensor s I64 → Prog α} :
    Safe E R W Q m (tl.convert .i32 .i64 x >>= k) ↔ Safe E R W Q m (k x.castInt) :=
  safe_convert_iff fun _i _r => ConvRel.iff_cast_i32_i64

/-- `x.to(tl.int32)` of an `int1` tensor is `0` or `1`. -/
def b2i (b : Bool) : I32 := IntV.ofInt (if b then 1 else 0)

/-- Safety of converting a boolean tensor to i32 followed by a continuation equals safety of the continuation applied to `x.map b2i`. -/
theorem safe_convB {x : Tensor s Bool} {k : Tensor s I32 → Prog α} :
    Safe E R W Q m (tl.convert .i1 .i32 x >>= k) ↔ Safe E R W Q m (k (x.map b2i)) :=
  safe_convert_iff fun _i _r => Iff.rfl

/-- `tl.convert(0.0, bfloat16)` (the `other=0.0` of a `bfloat16` load) is a tile of `+0`. -/
theorem convert_zero_safe {k : Tensor s BF16 → Prog α}
    (hk : ∀ o : Tensor s BF16, (∀ ix, o.get ix = FloatV.ofPat 0) → Safe E R W Q m (k o)) :
    Safe E R W Q m (tl.convert .f32 .bf16 (tl.full s zf32) >>= k) := by
  refine safe_convert (d := .f32) (d' := .bf16) (fun i => ⟨_, ConvRel.cvt_f32_bf16 _⟩) ?_
  intro z hz
  refine hk z fun i => ?_
  have h := ConvRel.f32_bf16_iff.mp (hz i)
  have h' : (FloatV.cvt ((tl.full s zf32 : Tensor s F32).get i) : BF16) = FloatV.ofPat 0 := by
    show (FloatV.cvt (zf32 : F32) : BF16) = _
    unfold zf32
    rw [FloatV.litF32_ofPat_zero]
    exact FloatV.cvt_ofPat_zero Format.binary32_regular
  rw [h'] at h
  exact h.eq (by decide +kernel)

end

end Claims.ExtendAttn
