import Claim
import TVA.Proof
import Proof.Peel

/-!
# Table entries converted to `int32` / `int64`

Lines 114–116 convert the `int32` or `int64` entries of `chunk_indices` and `cu_seqlens` with
`.to(tl.int32)` / `.to(tl.int64)`: a value in range keeps its value.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- `x.to(tl.int32)` of an `int32` or `int64` table entry of value `n` that fits. -/
theorem safe_conv_idx32 {D : DType} (hD : D = .i32 ∨ D = .i64) {x : D.Val} {n : Int}
    (hx : entryInt D x = n) (_hfit : IntV.SFits 32 n) {k : Tensor [] I32 → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt n)))) :
    Safe E R W Q m (tl.convert D .i32 (Tensor.scalar x) >>= k) := by
  rcases hD with rfl | rfl
  · refine (safe_convert_iff (y := Tensor.scalar (IntV.ofInt n)) (fun i r => ?_)).2 hk
    rw [ConvRel.iff_cast_i32_i32, IntV.cast_eq_ofInt]
    have : (x : I32).toInt = n := hx
    show r = IntV.ofInt (IntV.toInt x) ↔ r = IntV.ofInt n
    rw [this]
  · refine (safe_convert_iff (y := Tensor.scalar (IntV.ofInt n)) (fun i r => ?_)).2 hk
    rw [ConvRel.iff_cast_i64_i32, IntV.cast_eq_ofInt]
    have : (x : I64).toInt = n := hx
    show r = IntV.ofInt (IntV.toInt x) ↔ r = IntV.ofInt n
    rw [this]

/-- `x.to(tl.int64)` of an `int32` or `int64` table entry of value `n`. -/
theorem safe_conv_idx64 {D : DType} (hD : D = .i32 ∨ D = .i64) {x : D.Val} {n : Int}
    (hx : entryInt D x = n) {k : Tensor [] I64 → Prog α}
    (hk : Safe E R W Q m (k (Tensor.scalar (IntV.ofInt n)))) :
    Safe E R W Q m (tl.convert D .i64 (Tensor.scalar x) >>= k) := by
  rcases hD with rfl | rfl
  · refine (safe_convert_iff (y := Tensor.scalar (IntV.ofInt n)) (fun i r => ?_)).2 hk
    rw [ConvRel.iff_cast_i32_i64, IntV.cast_eq_ofInt]
    have : (x : I32).toInt = n := hx
    show r = IntV.ofInt (IntV.toInt x) ↔ r = IntV.ofInt n
    rw [this]
  · refine (safe_convert_iff (y := Tensor.scalar (IntV.ofInt n)) (fun i r => ?_)).2 hk
    rw [ConvRel.iff_cast_i64_i64, IntV.cast_eq_ofInt]
    have : (x : I64).toInt = n := hx
    show r = IntV.ofInt (IntV.toInt x) ↔ r = IntV.ofInt n
    rw [this]

end Claims.ChunkFwdO
