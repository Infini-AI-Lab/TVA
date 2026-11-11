import TVA.Proof
import Claim
import Proof.Ints

/-!
# The kernel's lane values

The kernel's lane values: tile offsets, the loop invariant `InInv`, the pointers of `a` and `b`, and the `k` masks.
-/


/-!
# The lane values of the tile offsets

The two offset vectors `offs_am` and `offs_bn`, and the masks of the loop: which element of the
matrices a lane of a program touches.
-/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- The `int32` tile coordinates `pid_m` (and `fpm`) are small. -/
theorem pid_bounds {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ((pidM d i : Nat) : Int) < 2 ^ 15 ∧ ((fpm d i : Nat) : Int) < 2 ^ 15 ∧
      8 * (grp d i : Int) = ((fpm d i : Nat) : Int) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hpm : pidM d i < npm d := pidM_lt hi
  have hfpm : fpm d i < npm d := pid_fpm_lt_npm hi
  have hnpm : npm d ≤ 2 ^ 15 := npm_le d hd
  have h1 : ((pidM d i : Nat) : Int) < 2 ^ 15 := by
    have h2 : ((pidM d i : Nat) : Int) < ((npm d : Nat) : Int) := by exact_mod_cast hpm
    have h3 : ((npm d : Nat) : Int) ≤ 2 ^ 15 := by exact_mod_cast hnpm
    omega
  have h2 : ((fpm d i : Nat) : Int) < 2 ^ 15 := by
    have h4 : ((fpm d i : Nat) : Int) < ((npm d : Nat) : Int) := by exact_mod_cast hfpm
    have h3 : ((npm d : Nat) : Int) ≤ 2 ^ 15 := by exact_mod_cast hnpm
    omega
  have h3 : 8 * (grp d i : Int) = ((fpm d i : Nat) : Int) := by
    rw [pid_fpm_eq]; push_cast; ring
  exact ⟨h1, h2, h3⟩

/-- Lane `r` of `offs_am` is `pid_m · 128 + r`. -/
theorem offs_row_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r3 r5 : Tensor [] I32}
    (hg : r3.item.toInt = (grp d i : Int))
    (ho : r5.item.toInt = ((pidM d i - fpm d i : Nat) : Int)) (r : Fin 128) :
    ((((r3.item * IntV.ofNat 8 + r5.item) : I32) * (IntV.ofNat 128 : I32)
        + tl.arange 0 128).get (r, ())).toInt
      = (pidM d i : Int) * 128 + (r : Int) := by
  obtain ⟨hpmb, hfpmb, hgrp⟩ := pid_bounds hd i
  have hlt : fpm d i ≤ pidM d i := by rw [pidM]; exact Nat.le_add_right _ _
  have hpmS : IntV.SFits 32 (pidM d i : Int) := s32 (by omega) (by omega)
  have h8S : IntV.SFits 32 ((grp d i : Int) * 8) := by
    have e : (grp d i : Int) * 8 = ((fpm d i : Nat) : Int) := by rw [← hgrp]; ring
    rw [e]; exact s32 (by omega) (by omega)
  have hmul8 : (r3.item * (IntV.ofNat 8 : I32)).toInt = 8 * (grp d i : Int) := by
    rw [IntV.toInt_mul rfl rfl (by rw [hg, i32_eight]; exact h8S), hg, i32_eight]
    ring
  have hsub : ((pidM d i - fpm d i : Nat) : Int) = (pidM d i : Int) - ((fpm d i : Nat) : Int) := by
    push_cast [hlt]; ring
  have hadd : (r3.item * IntV.ofNat 8 + r5.item).toInt = (pidM d i : Int) := by
    rw [IntV.toInt_add rfl rfl (by rw [hmul8, ho, hsub, hgrp]; convert hpmS using 1; ring),
      hmul8, ho, hsub, hgrp]
    ring
  have h128 : ((IntV.ofNat 128 : I32)).toInt = 128 := i32_128
  have hrmul : ((r3.item * IntV.ofNat 8 + r5.item) * (IntV.ofNat 128 : I32)).toInt
      = (pidM d i : Int) * 128 := by
    rw [IntV.toInt_mul rfl rfl
      (by rw [hadd, h128]; exact s32 (by omega) (by nlinarith)), hadd, h128]
  have hr : ((IntV.ofInt (0 + (r : Nat)) : I32)).toInt = ((r : Nat) : Int) := by
    rw [IntV.I32.toInt_ofInt (by omega)]; ring
  have hget : ((((r3.item * IntV.ofNat 8 + r5.item) : I32) * (IntV.ofNat 128 : I32)
        + tl.arange 0 128).get (r, ()))
      = ((r3.item * IntV.ofNat 8 + r5.item) * (IntV.ofNat 128 : I32))
          + (Tensor.arange (0 : Int) 128).get (r, ()) := rfl
  have har : (Tensor.arange (0 : Int) 128).get (r, ()) = IntV.ofInt (0 + (r : Nat)) :=
    Tensor.get_arange 0 128 r
  rw [hget, har, IntV.toInt_add rfl rfl (by
    rw [hrmul, hr]; exact s32 (by omega) (by omega)), hrmul, hr]

/-- Lane `c` of `offs_bn` is `pid_n · 256 + c`. -/
theorem offs_col_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r7 : Tensor [] I32}
    (hpn : r7.item.toInt = ((pidN d i : Nat) : Int)) (c : Fin 256) :
    (((r7.item * (IntV.ofNat 256 : I32) + tl.arange 0 256).get (c, ()))).toInt
      = (pidN d i : Int) * 256 + (c : Int) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hlt : pidN d i < npn d := pidN_lt hi
  have hnpn : npn d ≤ 2 ^ 9 := npn_le d hd
  have hb : ((pidN d i : Nat) : Int) < 2 ^ 15 := by
    have h2 : ((pidN d i : Nat) : Int) < ((npn d : Nat) : Int) := by exact_mod_cast hlt
    have h3 : ((npn d : Nat) : Int) ≤ 2 ^ 9 := by exact_mod_cast hnpn
    omega
  have h256 : ((IntV.ofNat 256 : I32)).toInt = 256 := i32_256
  have hmul : (r7.item * (IntV.ofNat 256 : I32)).toInt = (pidN d i : Int) * 256 := by
    rw [IntV.toInt_mul rfl rfl
      (by rw [hpn, h256]; exact s32 (by omega) (by nlinarith)), hpn, h256]
  have hc : ((IntV.ofInt (0 + (c : Nat)) : I32)).toInt = ((c : Nat) : Int) := by
    rw [IntV.I32.toInt_ofInt (by omega)]; ring
  have hget : (((r7.item * (IntV.ofNat 256 : I32) + tl.arange 0 256).get (c, ())))
      = (r7.item * (IntV.ofNat 256 : I32)) + (Tensor.arange (0 : Int) 256).get (c, ()) := rfl
  have har : (Tensor.arange (0 : Int) 256).get (c, ()) = IntV.ofInt (0 + (c : Nat)) :=
    Tensor.get_arange 0 256 c
  rw [hget, har, IntV.toInt_add rfl rfl (by
    rw [hmul, hc]; exact s32 (by omega) (by omega)), hmul, hc]

/-- The value of a lane of `offs_am`: the row of `a` (and of `c`) it names. -/
theorem lane_am {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r3 r5 : Tensor [] I32}
    (hg : r3.item.toInt = (grp d i : Int))
    (ho : r5.item.toInt = ((pidM d i - fpm d i : Nat) : Int)) {res : Tensor [128] I32}
    (hres : ∀ idx : Idx [128],
      (((r3.item * IntV.ofNat 8 + r5.item) * (IntV.ofNat 128 : I32) + tl.arange 0 128).get idx).rem?
          ((Tensor.full (IntV.ofNat d.M) : Tensor [128] I32).get idx) = some (res.get idx))
    (r : Fin 128) : (res.get (r, ())).toInt = (aRow d i r : Int) := by
  have hx := offs_row_val hd i hg ho r
  have hy : (IntV.ofNat d.M : I32).toInt = (d.M : Int) := i32_M d hd
  have hx0 : 0 ≤ (((r3.item * IntV.ofNat 8 + r5.item) * (IntV.ofNat 128 : I32)
      + tl.arange 0 128).get (r, ())).toInt := by rw [hx]; omega
  have hy0 : 0 < (IntV.ofNat d.M : I32).toInt := by rw [hy]; exact_mod_cast M_pos i
  obtain ⟨q, hq, hqv⟩ := IntV.rem?_of_nonneg (a := ((r3.item * IntV.ofNat 8 + r5.item)
    * (IntV.ofNat 128 : I32) + tl.arange 0 128).get (r, ())) (b := (IntV.ofNat d.M)) hx0 hy0
  have h1 := hres (r, ())
  have h2 : res.get (r, ()) = q := Option.some.inj (h1.symm.trans hq)
  rw [h2, hqv, hx, hy]
  show ((pidM d i : Int) * 128 + (r : Int)) % (d.M : Int) = ((aRow d i r : Nat) : Int)
  unfold aRow
  rw [Int.natCast_mod]
  push_cast
  ring

/-- The value of a lane of `offs_bn`: the column of `b` (and of `c`) it names. -/
theorem lane_bn {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {r7 : Tensor [] I32}
    (hpn : r7.item.toInt = ((pidN d i : Nat) : Int)) {res : Tensor [256] I32}
    (hres : ∀ idx : Idx [256],
      ((r7.item * (IntV.ofNat 256 : I32) + tl.arange 0 256).get idx).rem?
          ((Tensor.full (IntV.ofNat d.N) : Tensor [256] I32).get idx) = some (res.get idx))
    (c : Fin 256) : (res.get (c, ())).toInt = (bCol d i c : Int) := by
  have hx := offs_col_val hd i hpn c
  have hy : (IntV.ofNat d.N : I32).toInt = (d.N : Int) := i32_N d hd
  have hx0 : 0 ≤ ((r7.item * (IntV.ofNat 256 : I32) + tl.arange 0 256).get (c, ())).toInt := by
    rw [hx]; omega
  have hy0 : 0 < (IntV.ofNat d.N : I32).toInt := by rw [hy]; exact_mod_cast N_pos i
  obtain ⟨q, hq, hqv⟩ := IntV.rem?_of_nonneg (a := (r7.item * (IntV.ofNat 256 : I32)
    + tl.arange 0 256).get (c, ())) (b := (IntV.ofNat d.N)) hx0 hy0
  have h1 := hres (c, ())
  have h2 : res.get (c, ()) = q := Option.some.inj (h1.symm.trans hq)
  rw [h2, hqv, hx, hy]
  show ((pidN d i : Int) * 256 + (c : Int)) % (d.N : Int) = ((bCol d i c : Nat) : Int)
  unfold bCol
  rw [Int.natCast_mod]
  push_cast
  ring

/-- The trip count `cdiv(K, 64)` is `steps K`. -/
theorem steps_val {d : Inst} (hd : d.ok) {r : Tensor [] I32}
    (h : r.item.toInt
      = ((IntV.ofNat d.K : I32) + (IntV.ofNat (64 - 1) : I32)).toInt
          / (IntV.ofNat 64 : I32).toInt) :
    r.item.toInt = (steps d.K : Int) := by
  have hsum : (((IntV.ofNat d.K : I32)) + (IntV.ofNat (64 - 1) : I32)).toInt
      = (d.K : Int) + 63 := by
    rw [toInt_add_ofNat (by have := hd.K_le; omega)]; push_cast; ring
  rw [h, hsum, i32_64]
  have hcast : ((d.K + 63 : Nat) : Int) = (d.K : Int) + 63 := by push_cast; ring
  rw [← hcast]
  show ((d.K + 63 : Nat) : Int) / 64 = ((steps d.K : Nat) : Int)
  unfold steps
  rw [Int.natCast_div]
  norm_num

/-- The trip count is at most `128`. -/
theorem steps_le (d : Inst) (hd : d.ok) : steps d.K ≤ 128 := by
  have h1 : d.K + 63 ≤ 2 ^ 13 + 63 := by have := hd.K_le; omega
  have h2 : (d.K + 63) / 64 ≤ (2 ^ 13 + 63) / 64 := Nat.div_le_div_right h1
  have h3 : (2 ^ 13 + 63) / 64 = 128 := by norm_num
  unfold steps
  omega

end Claims.Matmul

/- ===== Err ===== -/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- One step of the error recurrence: the bound after a step follows from the bound before it and the step's error. -/
theorem err_step {ρ α : ℚ} (hρ : 0 ≤ ρ) {acc S L : Nat → ℚ} {t : Nat}
    (hL : 0 ≤ L t) (hα : 0 ≤ α)
    (hprev : |acc t - S t| ≤ (1 + ρ) ^ t * (ρ * (∑ s ∈ Finset.range t, (|S s| + L s)) + (t : ℚ) * α))
    (hstep : |acc (t + 1) - (acc t + (S (t + 1) - S t))| ≤ ρ * (|acc t| + L t) + α) :
    |acc (t + 1) - S (t + 1)| ≤
      (1 + ρ) ^ (t + 1) * (ρ * (∑ s ∈ Finset.range (t + 1), (|S s| + L s)) + ((t + 1 : Nat) : ℚ) * α) := by
  have hB : 0 ≤ ρ * (|S t| + L t) + α := by
    have h1 : 0 ≤ ρ * (|S t| + L t) := mul_nonneg hρ (by linarith [abs_nonneg (S t), hL])
    linarith
  have htri : |acc (t + 1) - S (t + 1)| ≤
      |acc (t + 1) - (acc t + (S (t + 1) - S t))| + |acc t - S t| := by
    have e : acc (t + 1) - S (t + 1) =
        (acc (t + 1) - (acc t + (S (t + 1) - S t))) + (acc t - S t) := by ring
    rw [e]
    exact abs_add_le _ _
  have habs : |acc t| ≤ |S t| + |acc t - S t| := by
    have e : acc t = S t + (acc t - S t) := by ring
    calc |acc t| = |S t + (acc t - S t)| := by conv_lhs => rw [e]
      _ ≤ |S t| + |acc t - S t| := abs_add_le _ _
  have hmul : ρ * (|acc t| + L t) ≤ ρ * (|S t| + |acc t - S t| + L t) := by
    apply mul_le_mul_of_nonneg_left _ hρ
    linarith
  have hE : |acc (t + 1) - S (t + 1)| ≤
      (ρ * (|S t| + L t) + α) + (1 + ρ) * |acc t - S t| := by
    linarith
  have hA : (ρ * (|S t| + L t) + α) + (1 + ρ) * |acc t - S t| ≤
      (ρ * (|S t| + L t) + α) + (1 + ρ) *
        ((1 + ρ) ^ t * (ρ * (∑ s ∈ Finset.range t, (|S s| + L s)) + (t : ℚ) * α)) := by
    have h1ρ : (0 : ℚ) ≤ 1 + ρ := by linarith
    linarith [mul_le_mul_of_nonneg_left hprev h1ρ]
  have hp1 : (1 : ℚ) ≤ (1 + ρ) ^ (t + 1) := one_le_pow₀ (by linarith)
  have hBble : ρ * (|S t| + L t) + α ≤ (1 + ρ) ^ (t + 1) * (ρ * (|S t| + L t) + α) := by
    calc ρ * (|S t| + L t) + α = 1 * (ρ * (|S t| + L t) + α) := by ring
      _ ≤ (1 + ρ) ^ (t + 1) * (ρ * (|S t| + L t) + α) :=
          mul_le_mul_of_nonneg_right hp1 hB
  have hsum : (∑ s ∈ Finset.range (t + 1), (|S s| + L s)) =
      (∑ s ∈ Finset.range t, (|S s| + L s)) + (|S t| + L t) :=
    Finset.sum_range_succ (fun s => |S s| + L s) t
  have hcast : ((t + 1 : Nat) : ℚ) * α = (t : ℚ) * α + α := by push_cast; ring
  rw [hsum, hcast]
  calc |acc (t + 1) - S (t + 1)|
      ≤ (ρ * (|S t| + L t) + α) + (1 + ρ) * |acc t - S t| := hE
    _ ≤ (ρ * (|S t| + L t) + α) + (1 + ρ) *
          ((1 + ρ) ^ t * (ρ * (∑ s ∈ Finset.range t, (|S s| + L s)) + (t : ℚ) * α)) := hA
    _ = (1 + ρ) ^ (t + 1) * (ρ * (∑ s ∈ Finset.range t, (|S s| + L s)) + (t : ℚ) * α) +
          (ρ * (|S t| + L t) + α) := by rw [pow_succ]; ring
    _ ≤ (1 + ρ) ^ (t + 1) * (ρ * (∑ s ∈ Finset.range t, (|S s| + L s)) + (t : ℚ) * α) +
          (1 + ρ) ^ (t + 1) * (ρ * (|S t| + L t) + α) := by linarith
    _ = (1 + ρ) ^ (t + 1) *
          (ρ * ((∑ s ∈ Finset.range t, (|S s| + L s)) + (|S t| + L t)) +
            ((t : ℚ) * α + α)) := by ring

end Claims.Matmul

/- ===== Inv ===== -/

/-!
# The loop's invariant

The reduction loop carries three tensors: the running pointers of `a` and `b`'s tiles and the
`float32` accumulator. The invariant of the induction: the memory does not change, the pointers are
the tile's at step `t`, and every lane of the tile's accumulator is within `AccB` of the exact
partial sum of its row and column.
-/

namespace Claims.Matmul

open TVA

/-- The state the reduction loop carries: `a`'s tile pointers, `b`'s tile pointers, and the
accumulator. -/
abbrev St :=
  Tensor [128, 64] (Ptr .f16) × Tensor [64, 256] (Ptr .f16) × Tensor [128, 256] F32

/-- The error bound of the accumulator after `t` steps: the one of `err_chain` with the step's
relative and absolute constants `rho` and `alp`, the exact partial sums `partSum` and the lifted
step magnitudes `stepMag`. -/
def AccB (K : Nat) (va vb : Nat → F16) (t : Nat) : ℚ :=
  (1 + rho) ^ t *
    (rho * (∑ s ∈ Finset.range t, (|partSum (prodV va vb K) s| + stepMag (liftV va vb K) s)) +
      (t : ℚ) * alp)

/-- The loop's invariant at step `t`: the memory is the initial one, the pointers are the tile's
at step `t`, and every lane of the tile's accumulator is within `AccB` of the exact partial sum of
its row and column. -/
def InInv (d : Inst) (i : Fin (launch d).size) (init : Mem) (t : Nat) (st : St)
    (m : Mem) : Prop :=
  m = init ∧
  (∀ (r : Fin 128) (kk : Fin 64), st.1.get2 r kk = d.a.ptr ↑(aAdr d i t r kk)) ∧
  (∀ (kk : Fin 64) (c : Fin 256), st.2.1.get2 kk c = d.b.ptr ↑(bAdr d i t kk c)) ∧
  (∀ (r : Fin 128) (c : Fin 256), cRow d i r < d.M → cCol d i c < d.N →
    ∀ va vb : Nat → F16,
      (∀ kk, kk < d.K →
        init.read .f16 (d.a.ptr ↑(aRow d i r * d.stride_am + kk * d.stride_ak)).addr =
            some (va kk) ∧ (va kk).isFinite = true) →
      (∀ kk, kk < d.K →
        init.read .f16 (d.b.ptr ↑(kk * d.stride_bk + bCol d i c * d.stride_bn)).addr =
            some (vb kk) ∧ (vb kk).isFinite = true) →
      ∃ v : ℚ, DType.val? .f32 (st.2.2.get2 r c) = some v ∧
        |v - partSum (prodV va vb d.K) t| ≤ AccB d.K va vb t)

end Claims.Matmul

/- ===== Ptr ===== -/

/-!
# Pointer-tensor lane values

The two pointer operations of the kernel: `p + offs` (a scalar pointer and an `int32` offset
tensor) and `ptrs + offs` (a pointer tensor), lane by lane.
-/

namespace Claims.Matmul

open TVA

/-- The pointer to element `i` of a buffer, as an offset of its first pointer. -/
theorem Buf_ptr_zero_offset {d : DType} (b : Buf d) (i : Int) :
    (b.ptr 0).offset i = b.ptr i := by
  simp [Buf.ptr, Ptr.offset]

/-- Lane `(r, kk)` of a scalar pointer advanced by an offset tensor. -/
theorem ptrAdd_get {s : Shape} {d : DType} {sg : Bool} {w : Nat} (p : Ptr d)
    (offs : Tensor s (IntV sg w)) (i : Idx s) :
    (Tensor.ptrAdd p offs).get i = p.offset (offs.get i).toInt := by
  simp [Tensor.ptrAdd]

/-- Lane `(r, kk)` of a pointer tensor advanced by an offset tensor. -/
theorem ptrAddT_get {s : Shape} {d : DType} {sg : Bool} {w : Nat}
    (ps : Tensor s (Ptr d)) (offs : Tensor s (IntV sg w)) (i : Idx s) :
    (Tensor.ptrAddT ps offs).get i = (ps.get i).offset (offs.get i).toInt := by
  simp [Tensor.ptrAddT]

end Claims.Matmul

/- ===== Get2 ===== -/

/-!
# `get2` forms of the pointer-arithmetic lanes

`Idx [m, n]` does not reduce at implicit transparency, so the `Idx`-typed `Tensor.get_ptrAdd`
lemmas do not rewrite a `get2`. These are their `get2` forms, definitional by `rfl`.
-/

namespace Claims.Matmul

open TVA

/-- Lane `(i, j)` of a scalar pointer advanced by an offset tensor. -/
theorem ptrAdd_get2 {m n : Nat} {d : DType} {sg : Bool} {w : Nat} (p : Ptr d)
    (offs : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    (p + offs).get2 i j = p.offset (offs.get2 i j).toInt := rfl

/-- Lane `(i, j)` of a pointer tensor advanced by an offset tensor. -/
theorem ptrAddT_get2 {m n : Nat} {d : DType} {sg : Bool} {w : Nat}
    (ps : Tensor [m, n] (Ptr d)) (offs : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    (ps + offs).get2 i j = (ps.get2 i j).offset (offs.get2 i j).toInt := rfl

end Claims.Matmul

/- ===== AOff ===== -/

/-!
# The initial `a` offset of a lane

Lane `(r, kk)` of `offs_am.col 64 * stride_am + offs_k.row 128 * stride_ak` is the element of `a`'s
storage lane `(r, kk)` reads at step `0`.
-/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- The `int32` product of two naturals that fit is the product of their values. -/
private theorem s32_mul_nat {a b : Nat} (h : a * b < 2 ^ 31) :
    IntV.SFits 32 ((a : Int) * (b : Int)) := by
  refine s32 ?_ ?_
  · have e1 : (0 : Int) ≤ (a : Int) := Nat.cast_nonneg _
    have e2 : (0 : Int) ≤ (b : Int) := Nat.cast_nonneg _
    have := mul_nonneg e1 e2
    omega
  · have h' : ((a * b : Nat) : Int) < (2 ^ 31 : Int) := by exact_mod_cast h
    rwa [Nat.cast_mul] at h'

/-- The `int32` offsets of the `a` tile have the values of `aAdr` at step `0`. -/
theorem aOff_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {res : Tensor [128] I32}
    (hres : ∀ r : Fin 128, (res.get (r, ())).toInt = (aRow d i r : Int))
    (r : Fin 128) (kk : Fin 64) :
    ((res.col 64 * (IntV.ofNat d.stride_am : I32) +
        (tl.arange 0 64).row 128 * (IntV.ofNat d.stride_ak : I32)).get2 r kk).toInt
      = ((aAdr d i 0 r kk : Nat) : Int) := by
  have ham : ((IntV.ofNat d.stride_am : I32)).toInt = (d.stride_am : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.am.2; omega)
  have hak : ((IntV.ofNat d.stride_ak : I32)).toInt = (d.stride_ak : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.ak.2; omega)
  have hkk : ((IntV.ofInt (0 + (kk : Nat)) : I32)).toInt = ((kk : Nat) : Int) := by
    rw [IntV.I32.toInt_ofInt (by omega)]
    ring
  have h1m : ((res.get (r, ())) * (IntV.ofNat d.stride_am : I32)).toInt
      = (aRow d i r : Int) * (d.stride_am : Int) := by
    rw [IntV.toInt_mul rfl rfl (by rw [hres r, ham]; exact s32_mul_nat (a_fit_row hd i r)),
      hres r, ham]
  have h2m : ((Tensor.arange (0 : Int) 64).get (kk, ()) *
        (IntV.ofNat d.stride_ak : I32)).toInt
      = ((kk : Nat) : Int) * (d.stride_ak : Int) := by
    rw [Tensor.get_arange,
      IntV.toInt_mul rfl rfl (by rw [hkk, hak]; exact s32_mul_nat (a_fit_k hd (kk : Nat) kk.isLt)),
      hkk, hak]
  have hsumfit : (aRow d i r : Int) * (d.stride_am : Int)
      + ((kk : Nat) : Int) * (d.stride_ak : Int) < 2 ^ 31 := by
    have h := a_fit_lane hd i (r : Nat) (kk : Nat) kk.isLt
    have h' : ((aRow d i r * d.stride_am + (kk : Nat) * d.stride_ak : Nat) : Int)
        < (2 ^ 31 : Int) := by exact_mod_cast h
    push_cast at h'
    exact h'
  have hsf : IntV.SFits 32 ((aRow d i r : Int) * (d.stride_am : Int)
      + ((kk : Nat) : Int) * (d.stride_ak : Int)) := by
    refine s32 ?_ hsumfit
    have e1 : (0 : Int) ≤ (aRow d i r : Int) * (d.stride_am : Int) :=
      mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    have e2 : (0 : Int) ≤ ((kk : Nat) : Int) * (d.stride_ak : Int) :=
      mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    omega
  have hget : ((res.col 64 * (IntV.ofNat d.stride_am : I32) +
        (tl.arange 0 64).row 128 * (IntV.ofNat d.stride_ak : I32)).get (r, kk, ()))
      = (res.get (r, ()) * (IntV.ofNat d.stride_am : I32))
        + ((Tensor.arange (0 : Int) 64).get (kk, ()) * (IntV.ofNat d.stride_ak : I32)) := rfl
  show ((res.col 64 * (IntV.ofNat d.stride_am : I32) +
        (tl.arange 0 64).row 128 * (IntV.ofNat d.stride_ak : I32)).get (r, kk, ())).toInt
      = ((aAdr d i 0 r kk : Nat) : Int)
  rw [hget, IntV.toInt_add rfl rfl (by rw [h1m, h2m]; exact hsf), h1m, h2m]
  unfold aAdr
  push_cast
  ring

end Claims.Matmul

/- ===== BOff ===== -/

/-!
# The initial `b` offset of a lane

Lane `(kk, c)` of `offs_k.col 256 * stride_bk + offs_bn.row 64 * stride_bn` is the element of `b`'s
storage lane `(kk, c)` reads at step `0`.
-/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- The `int32` product of two naturals that fit is the product of their values. -/
private theorem s32_mul_nat_off {a b : Nat} (h : a * b < 2 ^ 31) :
    IntV.SFits 32 ((a : Int) * (b : Int)) := by
  refine s32 ?_ ?_
  · have e1 : (0 : Int) ≤ (a : Int) := Nat.cast_nonneg _
    have e2 : (0 : Int) ≤ (b : Int) := Nat.cast_nonneg _
    have := mul_nonneg e1 e2
    omega
  · have h' : ((a * b : Nat) : Int) < (2 ^ 31 : Int) := by exact_mod_cast h
    rwa [Nat.cast_mul] at h'

/-- The `int32` offsets of the `b` tile have the values of `bAdr` at step `0`. -/
theorem bOff_val {d : Inst} (hd : d.ok) (i : Fin (launch d).size) {res2 : Tensor [256] I32}
    (hres2 : ∀ c : Fin 256, (res2.get (c, ())).toInt = (bCol d i c : Int))
    (kk : Fin 64) (c : Fin 256) :
    (((tl.arange 0 64).col 256 * (IntV.ofNat d.stride_bk : I32) +
        res2.row 64 * (IntV.ofNat d.stride_bn : I32)).get2 kk c).toInt
      = ((bAdr d i 0 kk c : Nat) : Int) := by
  have hbk : ((IntV.ofNat d.stride_bk : I32)).toInt = (d.stride_bk : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.bk.2; omega)
  have hbn : ((IntV.ofNat d.stride_bn : I32)).toInt = (d.stride_bn : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.bn.2; omega)
  have hkk : ((IntV.ofInt (0 + (kk : Nat)) : I32)).toInt = ((kk : Nat) : Int) := by
    rw [IntV.I32.toInt_ofInt (by omega)]
    ring
  have hfitk : (kk : Nat) * d.stride_bk < 2 ^ 31 := by
    have h := b_fit_lane hd i (c : Nat) (kk : Nat) kk.isLt
    have h0 : 0 ≤ bCol d i (c : Nat) * d.stride_bn := Nat.zero_le _
    omega
  have h1m : (((Tensor.arange (0 : Int) 64).get (kk, ())) *
        (IntV.ofNat d.stride_bk : I32)).toInt
      = ((kk : Nat) : Int) * (d.stride_bk : Int) := by
    rw [Tensor.get_arange,
      IntV.toInt_mul rfl rfl (by
        rw [hkk, hbk]; exact s32_mul_nat_off (a := (kk : Nat)) (b := d.stride_bk) hfitk),
      hkk, hbk]
  have hfitr : bCol d i (c : Nat) * d.stride_bn < 2 ^ 31 := by
    have h := b_fit_lane hd i (c : Nat) 0 (by norm_num)
    simpa using h
  have h2m : ((res2.get (c, ())) * (IntV.ofNat d.stride_bn : I32)).toInt
      = (bCol d i (c : Nat) : Int) * (d.stride_bn : Int) := by
    rw [IntV.toInt_mul rfl rfl (by
        rw [hres2 c, hbn]
        exact s32_mul_nat_off (a := bCol d i (c : Nat)) (b := d.stride_bn) hfitr),
      hres2 c, hbn]
  have hsumfit : ((kk : Nat) : Int) * (d.stride_bk : Int)
      + (bCol d i (c : Nat) : Int) * (d.stride_bn : Int) < 2 ^ 31 := by
    have h := b_fit_lane hd i (c : Nat) (kk : Nat) kk.isLt
    have h' : ((kk * d.stride_bk + bCol d i (c : Nat) * d.stride_bn : Nat) : Int)
        < (2 ^ 31 : Int) := by exact_mod_cast h
    push_cast at h'
    exact h'
  have hsf : IntV.SFits 32 (((kk : Nat) : Int) * (d.stride_bk : Int)
      + (bCol d i (c : Nat) : Int) * (d.stride_bn : Int)) := by
    refine s32 ?_ hsumfit
    have e1 : (0 : Int) ≤ ((kk : Nat) : Int) * (d.stride_bk : Int) :=
      mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    have e2 : (0 : Int) ≤ (bCol d i (c : Nat) : Int) * (d.stride_bn : Int) :=
      mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    omega
  have hget : (((tl.arange 0 64).col 256 * (IntV.ofNat d.stride_bk : I32) +
        res2.row 64 * (IntV.ofNat d.stride_bn : I32)).get (kk, c, ()))
      = ((Tensor.arange (0 : Int) 64).get (kk, ()) * (IntV.ofNat d.stride_bk : I32))
        + (res2.get (c, ()) * (IntV.ofNat d.stride_bn : I32)) := rfl
  show (((tl.arange 0 64).col 256 * (IntV.ofNat d.stride_bk : I32) +
        res2.row 64 * (IntV.ofNat d.stride_bn : I32)).get (kk, c, ())).toInt
      = ((bAdr d i 0 kk c : Nat) : Int)
  rw [hget, IntV.toInt_add rfl rfl (by rw [h1m, h2m]; exact hsf), h1m, h2m]
  unfold bAdr
  push_cast
  ring

end Claims.Matmul

/- ===== MaskK ===== -/

/-!
# The loop's mask holds on every lane

The mask `offs_k < K - k · 64` of the loads holds on every lane: `K` is a multiple of `64`, so at
step `t < K / 64` the lane's `k` is below `K`.
-/

namespace Claims.Matmul

open TVA

set_option maxHeartbeats 1000000

/-- The `k` mask of a step below `steps K` is true on every lane. -/
theorem mask_k_true {d : Inst} (hd : d.ok) {t : Nat} {kk : Fin 64} {s : I32}
    (hs : s.toInt = (d.K : Int) - 64 * (t : Int)) (ht : t < steps d.K) :
    (IntV.lt ((tl.arange 0 64).get (kk, ())) s) = true := by
  have hval : ((tl.arange 0 64).get (kk, ())).toInt = (kk.val : Int) := by
    rw [tl.arange, Tensor.get_arange]
    have h := IntV.I32.toInt_ofInt (x := (0 : Int) + (kk.val : Int)) (by
      have h3 : (kk.val : Int) < (2 ^ 31 : Int) := by
        have h4 : kk.val < 2 ^ 31 := by have := kk.isLt; omega
        exact_mod_cast h4
      constructor <;> omega)
    simpa using h
  rw [IntV.lt_eq, decide_eq_true_iff, hval, hs]
  have hstep := step_lt hd ht kk.isLt
  have hstep' : ((64 * t + kk.val : Nat) : Int) < (d.K : Int) := by exact_mod_cast hstep
  push_cast at hstep'
  omega

end Claims.Matmul
