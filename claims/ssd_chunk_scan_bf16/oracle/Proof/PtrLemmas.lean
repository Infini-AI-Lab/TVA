import Proof.Decode

/-!
# Pointer arithmetic of the set-ups

The `int64` offsets the kernel adds to each base pointer (lines 186–216) are sums of products of
nonnegative values that fit: this file evaluates them.
-/

namespace Claims.SsdChunkScan

open TVA

/-- The `int64` constant `n` (below `2^63`). -/
theorem I64_ofNat {n : Nat} (h : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = (n : Int) :=
  IntV.toInt_ofNat (by simpa using h)

/-- An `int32` constant widened to `int64` keeps its value. -/
theorem cast_ofNat {n : Nat} (h : n < 2 ^ 31) :
    ((IntV.cast (IntV.ofNat n : I32) : I64)).toInt = (n : Int) := by
  rw [IntV.toInt_cast _ (by norm_num)]; exact IntV.I32.toInt_ofNat h

/-- A widened `int32` product `g · C` of small naturals. -/
theorem cast_mul {g : I32} {gv C : Nat} (hg : g.toInt = (gv : Int)) (hC : C < 2 ^ 31)
    (h : gv * C < 2 ^ 31) : ((IntV.cast (g * (IntV.ofNat C : I32)) : I64)).toInt = ((gv * C : Nat) : Int) := by
  rw [IntV.toInt_cast _ (by norm_num)]
  exact IntV.mul_nat hg (IntV.I32.toInt_ofNat hC) (by simpa using h)

/-- A widened `int32` value keeps its value. -/
theorem cast_val {g : I32} {gv : Int} (hg : g.toInt = gv) : ((IntV.cast g : I64)).toInt = gv := by
  rw [IntV.toInt_cast _ (by norm_num)]; exact hg

/-- `int64` sum of three nonnegative values. -/
theorem sum3 {a b c : I64} {x y z : Nat} (ha : a.toInt = (x : Int)) (hb : b.toInt = (y : Int))
    (hc : c.toInt = (z : Int)) (h : x + y + z < 2 ^ 62) :
    (a + b + c).toInt = ((x + y + z : Nat) : Int) := by
  have h1 : (a + b).toInt = ((x + y : Nat) : Int) :=
    IntV.add_nat ha hb (show x + y < 2 ^ (64 - 1) by omega)
  exact IntV.add_nat h1 hc (show x + y + z < 2 ^ (64 - 1) by omega)

/-- `base + z` for a buffer's element-`0` pointer and an offset of value `v`. -/
theorem ptr_zero_add {δ : DType} (b : Buf δ) {z : I64} {v : Nat} (hz : z.toInt = (v : Int)) :
    b.ptr 0 + z = b.ptr (v : Int) := by
  show (b.ptr 0).offset z.toInt = _
  rw [Buf.ptr_offset, hz]; simp

/-- `int64` product of two nonnegative values. -/
theorem mul64 {a b : I64} {x y : Nat} (ha : a.toInt = (x : Int)) (hb : b.toInt = (y : Int))
    (h : x * y < 2 ^ 62) : (a * b).toInt = ((x * y : Nat) : Int) :=
  IntV.mul_nat ha hb (show x * y < 2 ^ (64 - 1) by omega)

/-- `b.ptr 0 + (pid_b·A + c·B + (g·C widened))` (`cb`, `dt`, `dA_cumsum`): element `c·B + g·C`. -/
theorem ptr_a {δ : DType} (b : Buf δ) {pb ci : I64} {g : I32} {A B C kv gv : Nat}
    (hpb : pb.toInt = 0) (hci : ci.toInt = (kv : Int)) (hg : g.toInt = (gv : Int))
    (hA : A < 2 ^ 31) (hB : B < 2 ^ 31) (hC : C < 2 ^ 31) (hgC : gv * C < 2 ^ 31)
    (hkB : kv * B < 2 ^ 31) :
    b.ptr 0 + (pb * (IntV.cast (IntV.ofNat A : I32) : I64) + ci * (IntV.cast (IntV.ofNat B : I32) : I64) +
      (IntV.cast (g * (IntV.ofNat C : I32)) : I64)) = b.ptr ((kv * B + gv * C : Nat) : Int) := by
  have h1 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64)).toInt = ((0 * A : Nat) : Int) :=
    mul64 (x := 0) (y := A) (by simpa using hpb) (cast_ofNat hA) (by omega)
  have h2 := mul64 hci (cast_ofNat hB) (by omega : kv * B < 2 ^ 62)
  have h3 := cast_mul hg hC hgC
  refine ptr_zero_add b (sum3 h1 h2 h3 (by omega))
    |>.trans ?_
  congr 1; push_cast; ring

/-- The same with `c·B₁·B₂` (`x`, `C`, `out`). -/
theorem ptr_b {δ : DType} (b : Buf δ) {pb ci : I64} {g : I32} {A B₁ B₂ C kv gv : Nat}
    (hpb : pb.toInt = 0) (hci : ci.toInt = (kv : Int)) (hg : g.toInt = (gv : Int))
    (hA : A < 2 ^ 31) (hB₁ : B₁ < 2 ^ 31) (hB₂ : B₂ < 2 ^ 31) (hC : C < 2 ^ 31)
    (hgC : gv * C < 2 ^ 31) (hkB₁ : kv * B₁ < 2 ^ 31) (hkB : kv * B₁ * B₂ < 2 ^ 31) :
    b.ptr 0 + (pb * (IntV.cast (IntV.ofNat A : I32) : I64) +
      ci * (IntV.cast (IntV.ofNat B₁ : I32) : I64) * (IntV.cast (IntV.ofNat B₂ : I32) : I64) +
      (IntV.cast (g * (IntV.ofNat C : I32)) : I64)) = b.ptr ((kv * B₁ * B₂ + gv * C : Nat) : Int) := by
  have h1 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64)).toInt = ((0 * A : Nat) : Int) :=
    mul64 (x := 0) (y := A) (by simpa using hpb) (cast_ofNat hA) (by omega)
  have h2a := mul64 hci (cast_ofNat hB₁) (show kv * B₁ < 2 ^ 62 by omega)
  have h2 := mul64 h2a (cast_ofNat hB₂) (show kv * B₁ * B₂ < 2 ^ 62 by omega)
  have h3 := cast_mul hg hC hgC
  refine ptr_zero_add b (sum3 h1 h2 h3 (by omega)) |>.trans ?_
  congr 1; push_cast; ring

/-- The `states` pointer: `b.ptr 0 + pid_b·A + c·B + g·C` (the last term an `int32` product). -/
theorem ptr_s {δ : DType} (b : Buf δ) {pb ci : I64} {g : I32} {A B C kv gv : Nat}
    (hpb : pb.toInt = 0) (hci : ci.toInt = (kv : Int)) (hg : g.toInt = (gv : Int))
    (hA : A < 2 ^ 31) (hB : B < 2 ^ 31) (hC : C < 2 ^ 31) (hgC : gv * C < 2 ^ 31)
    (hkB : kv * B < 2 ^ 31) :
    b.ptr 0 + pb * (IntV.cast (IntV.ofNat A : I32) : I64) +
      ci * (IntV.cast (IntV.ofNat B : I32) : I64) + g * (IntV.ofNat C : I32) =
      b.ptr ((kv * B + gv * C : Nat) : Int) := by
  have h1 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64)).toInt = ((0 * A : Nat) : Int) :=
    mul64 (x := 0) (y := A) (by simpa using hpb) (cast_ofNat hA) (by omega)
  have h2 := mul64 hci (cast_ofNat hB) (by omega : kv * B < 2 ^ 62)
  have h3 : (g * (IntV.ofNat C : I32)).toInt = ((gv * C : Nat) : Int) :=
    IntV.mul_nat hg (IntV.I32.toInt_ofNat hC) (by simpa using hgC)
  show (((b.ptr 0).offset _).offset _).offset _ = _
  rw [Buf.ptr_offset, Buf.ptr_offset, Buf.ptr_offset, h1, h2, h3]
  congr 1; push_cast; ring

/-- The `seq_idx` pointer: `b.ptr 0 + (pid_b·A + c·B₁·B₂)`. -/
theorem ptr_c {δ : DType} (b : Buf δ) {pb ci : I64} {A B₁ B₂ kv : Nat}
    (hpb : pb.toInt = 0) (hci : ci.toInt = (kv : Int)) (hA : A < 2 ^ 31) (hB₁ : B₁ < 2 ^ 31)
    (hB₂ : B₂ < 2 ^ 31) (hkB₁ : kv * B₁ < 2 ^ 31) (hkB : kv * B₁ * B₂ < 2 ^ 31) :
    b.ptr 0 + (pb * (IntV.cast (IntV.ofNat A : I32) : I64) +
      ci * (IntV.cast (IntV.ofNat B₁ : I32) : I64) * (IntV.cast (IntV.ofNat B₂ : I32) : I64)) =
      b.ptr ((kv * B₁ * B₂ : Nat) : Int) := by
  have h1 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64)).toInt = ((0 * A : Nat) : Int) :=
    mul64 (x := 0) (y := A) (by simpa using hpb) (cast_ofNat hA) (by omega)
  have h2a := mul64 hci (cast_ofNat hB₁) (show kv * B₁ < 2 ^ 62 by omega)
  have h2 := mul64 h2a (cast_ofNat hB₂) (show kv * B₁ * B₂ < 2 ^ 62 by omega)
  have h3 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64) +
      ci * (IntV.cast (IntV.ofNat B₁ : I32) : I64) * (IntV.cast (IntV.ofNat B₂ : I32) : I64)).toInt =
      ((0 * A + kv * B₁ * B₂ : Nat) : Int) :=
    IntV.add_nat h1 h2 (show 0 * A + kv * B₁ * B₂ < 2 ^ (64 - 1) by omega)
  refine ptr_zero_add b h3 |>.trans ?_
  congr 1; simp

/-- With an `int32` chunk index (initial states): `b.ptr 0 + (pid_b·A + widen(c·B) + widen(g·C))`. -/
theorem ptr_a' {δ : DType} (b : Buf δ) {pb : I64} {ci g : I32} {A B C kv gv : Nat}
    (hpb : pb.toInt = 0) (hci : ci.toInt = (kv : Int)) (hg : g.toInt = (gv : Int))
    (hA : A < 2 ^ 31) (hB : B < 2 ^ 31) (hC : C < 2 ^ 31) (hgC : gv * C < 2 ^ 31)
    (hkB : kv * B < 2 ^ 31) :
    b.ptr 0 + (pb * (IntV.cast (IntV.ofNat A : I32) : I64) +
      (IntV.cast (ci * (IntV.ofNat B : I32)) : I64) +
      (IntV.cast (g * (IntV.ofNat C : I32)) : I64)) = b.ptr ((kv * B + gv * C : Nat) : Int) := by
  have h1 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64)).toInt = ((0 * A : Nat) : Int) :=
    mul64 (x := 0) (y := A) (by simpa using hpb) (cast_ofNat hA) (by omega)
  have h2 := cast_mul hci hB hkB
  have h3 := cast_mul hg hC hgC
  refine ptr_zero_add b (sum3 h1 h2 h3 (by omega)) |>.trans ?_
  congr 1; push_cast; ring

/-- The same with `widen(c·B₁·B₂)` (`x`, `C`, `out`). -/
theorem ptr_b' {δ : DType} (b : Buf δ) {pb : I64} {ci g : I32} {A B₁ B₂ C kv gv : Nat}
    (hpb : pb.toInt = 0) (hci : ci.toInt = (kv : Int)) (hg : g.toInt = (gv : Int))
    (hA : A < 2 ^ 31) (hB₁ : B₁ < 2 ^ 31) (hB₂ : B₂ < 2 ^ 31) (hC : C < 2 ^ 31)
    (hgC : gv * C < 2 ^ 31) (hkB₁ : kv * B₁ < 2 ^ 31) (hkB : kv * B₁ * B₂ < 2 ^ 31) :
    b.ptr 0 + (pb * (IntV.cast (IntV.ofNat A : I32) : I64) +
      (IntV.cast (ci * (IntV.ofNat B₁ : I32) * (IntV.ofNat B₂ : I32)) : I64) +
      (IntV.cast (g * (IntV.ofNat C : I32)) : I64)) = b.ptr ((kv * B₁ * B₂ + gv * C : Nat) : Int) := by
  have h1 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64)).toInt = ((0 * A : Nat) : Int) :=
    mul64 (x := 0) (y := A) (by simpa using hpb) (cast_ofNat hA) (by omega)
  have h2a : (ci * (IntV.ofNat B₁ : I32)).toInt = ((kv * B₁ : Nat) : Int) :=
    IntV.mul_nat hci (IntV.I32.toInt_ofNat hB₁) (by simpa using hkB₁)
  have h2 : (IntV.cast (ci * (IntV.ofNat B₁ : I32) * (IntV.ofNat B₂ : I32)) : I64).toInt =
      ((kv * B₁ * B₂ : Nat) : Int) := by
    rw [IntV.toInt_cast _ (by norm_num)]
    exact IntV.mul_nat h2a (IntV.I32.toInt_ofNat hB₂) (by simpa using hkB)
  have h3 := cast_mul hg hC hgC
  refine ptr_zero_add b (sum3 h1 h2 h3 (by omega)) |>.trans ?_
  congr 1; push_cast; ring

/-- The `states` pointer with an `int32` chunk index: `b.ptr 0 + pid_b·A + c·B + g·C`. -/
theorem ptr_s' {δ : DType} (b : Buf δ) {pb : I64} {ci g : I32} {A B C kv gv : Nat}
    (hpb : pb.toInt = 0) (hci : ci.toInt = (kv : Int)) (hg : g.toInt = (gv : Int))
    (hA : A < 2 ^ 31) (hB : B < 2 ^ 31) (hC : C < 2 ^ 31) (hgC : gv * C < 2 ^ 31)
    (hkB : kv * B < 2 ^ 31) :
    b.ptr 0 + pb * (IntV.cast (IntV.ofNat A : I32) : I64) +
      ci * (IntV.ofNat B : I32) + g * (IntV.ofNat C : I32) =
      b.ptr ((kv * B + gv * C : Nat) : Int) := by
  have h1 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64)).toInt = ((0 * A : Nat) : Int) :=
    mul64 (x := 0) (y := A) (by simpa using hpb) (cast_ofNat hA) (by omega)
  have h2 : (ci * (IntV.ofNat B : I32)).toInt = ((kv * B : Nat) : Int) :=
    IntV.mul_nat hci (IntV.I32.toInt_ofNat hB) (by simpa using hkB)
  have h3 : (g * (IntV.ofNat C : I32)).toInt = ((gv * C : Nat) : Int) :=
    IntV.mul_nat hg (IntV.I32.toInt_ofNat hC) (by simpa using hgC)
  show (((b.ptr 0).offset _).offset _).offset _ = _
  rw [Buf.ptr_offset, Buf.ptr_offset, Buf.ptr_offset, h1, h2, h3]
  congr 1; push_cast; ring

/-- The `seq_idx` pointer with an `int32` chunk index. -/
theorem ptr_c' {δ : DType} (b : Buf δ) {pb : I64} {ci : I32} {A B₁ B₂ kv : Nat}
    (hpb : pb.toInt = 0) (hci : ci.toInt = (kv : Int)) (hA : A < 2 ^ 31) (hB₁ : B₁ < 2 ^ 31)
    (hB₂ : B₂ < 2 ^ 31) (hkB₁ : kv * B₁ < 2 ^ 31) (hkB : kv * B₁ * B₂ < 2 ^ 31) :
    b.ptr 0 + (pb * (IntV.cast (IntV.ofNat A : I32) : I64) +
      (IntV.cast (ci * (IntV.ofNat B₁ : I32) * (IntV.ofNat B₂ : I32)) : I64)) =
      b.ptr ((kv * B₁ * B₂ : Nat) : Int) := by
  have h1 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64)).toInt = ((0 * A : Nat) : Int) :=
    mul64 (x := 0) (y := A) (by simpa using hpb) (cast_ofNat hA) (by omega)
  have h2a : (ci * (IntV.ofNat B₁ : I32)).toInt = ((kv * B₁ : Nat) : Int) :=
    IntV.mul_nat hci (IntV.I32.toInt_ofNat hB₁) (by simpa using hkB₁)
  have h2 : (IntV.cast (ci * (IntV.ofNat B₁ : I32) * (IntV.ofNat B₂ : I32)) : I64).toInt =
      ((kv * B₁ * B₂ : Nat) : Int) := by
    rw [IntV.toInt_cast _ (by norm_num)]
    exact IntV.mul_nat h2a (IntV.I32.toInt_ofNat hB₂) (by simpa using hkB)
  have h3 : (pb * (IntV.cast (IntV.ofNat A : I32) : I64) +
      (IntV.cast (ci * (IntV.ofNat B₁ : I32) * (IntV.ofNat B₂ : I32)) : I64)).toInt =
      ((0 * A + kv * B₁ * B₂ : Nat) : Int) :=
    IntV.add_nat h1 h2 (show 0 * A + kv * B₁ * B₂ < 2 ^ (64 - 1) by omega)
  refine ptr_zero_add b h3 |>.trans ?_
  congr 1; simp

/-- The initial-state pointer: `b.ptr 0 + r·B + g·C` (two `int32` products). -/
theorem ptr_i {δ : DType} (b : Buf δ) {x g : I32} {B C xv gv : Nat}
    (hx : x.toInt = (xv : Int)) (hg : g.toInt = (gv : Int)) (hB : B < 2 ^ 31) (hC : C < 2 ^ 31)
    (hxB : xv * B < 2 ^ 31) (hgC : gv * C < 2 ^ 31) :
    b.ptr 0 + x * (IntV.ofNat B : I32) + g * (IntV.ofNat C : I32) =
      b.ptr ((xv * B + gv * C : Nat) : Int) := by
  have h2 : (x * (IntV.ofNat B : I32)).toInt = ((xv * B : Nat) : Int) :=
    IntV.mul_nat hx (IntV.I32.toInt_ofNat hB) (by simpa using hxB)
  have h3 : (g * (IntV.ofNat C : I32)).toInt = ((gv * C : Nat) : Int) :=
    IntV.mul_nat hg (IntV.I32.toInt_ofNat hC) (by simpa using hgC)
  show ((b.ptr 0).offset _).offset _ = _
  rw [Buf.ptr_offset, Buf.ptr_offset, h2, h3]
  congr 1; push_cast; ring

end Claims.SsdChunkScan
