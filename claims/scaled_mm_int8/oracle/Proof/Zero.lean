import Proof.Setup

/-!
# `scaled_mm_int8`: the loop invariant before the first trip

The state the kernel enters the reduction loop with: the offset row is `arange`, the two pointer
tiles are the base pointers plus the row and column offsets times the strides, and the accumulator
is zero.
-/

namespace Claims.ScaledMmInt8
open TVA
set_option maxHeartbeats 800000

/-- `s·a + t·b < 2^63` for `s, t < 2^31`, `a < 2^31 + 2^8`, `b < 2^9`. -/
theorem sum_mul_lt63 {s a t b : Nat} (hs : s < 2^31) (ha : a < 2^31 + 2^8) (ht : t < 2^31)
    (hb : b < 2^9) : s * a + t * b < 2 ^ 63 := by
  have h1 : s * a ≤ (2^31 - 1) * (2^31 + 2^8 - 1) := Nat.mul_le_mul (by omega) (by omega)
  have h2 : t * b ≤ (2^31 - 1) * (2^9 - 1) := Nat.mul_le_mul (by omega) (by omega)
  have h3 : (2^31 - 1) * (2^31 + 2^8 - 1) + (2^31 - 1) * (2^9 - 1) < 2 ^ 63 := by norm_num
  omega

/-- The same bound with the roles of the two products swapped. -/
theorem sum_mul_lt63' {s a t b : Nat} (hs : s < 2^31) (ha : a < 2^9) (ht : t < 2^31)
    (hb : b < 2^31 + 2^8) : s * a + t * b < 2 ^ 63 := by
  have h1 : s * a ≤ (2^31 - 1) * (2^9 - 1) := Nat.mul_le_mul (by omega) (by omega)
  have h2 : t * b ≤ (2^31 - 1) * (2^31 + 2^8 - 1) := Nat.mul_le_mul (by omega) (by omega)
  have h3 : (2^31 - 1) * (2^9 - 1) + (2^31 - 1) * (2^31 + 2^8 - 1) < 2 ^ 63 := by norm_num
  omega

/-- The `A` pointer offsets of the first trip are `(pm·BM + i)·stride_am + kk·stride_ak`. -/
theorem a_get2_toInt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d)
    (r4 r11 r12 : Tensor [] I64) (r5 : Tensor [d.BM] I64) (r10 : Tensor [d.BK] I64)
    (h4 : r4.item.toInt = ((pm d p * d.BM : Nat) : ℤ))
    (h5 : ∀ i : Fin d.BM, (r5.get1 i).toInt = (i.1 : ℤ))
    (h10 : ∀ kk : Fin d.BK, (r10.get1 kk).toInt = (kk.1 : ℤ))
    (h11 : r11.item.toInt = (d.stride_am : ℤ))
    (h12 : r12.item.toInt = (d.stride_ak : ℤ))
    (i : Fin d.BM) (kk : Fin d.BK) :
    (((r11.item * (r4.item + r5)).col d.BK + (r12.item * r10).row d.BM).get2 i kk).toInt
      = (((pm d p * d.BM + i.1) * d.stride_am + (0 * d.BK + kk.1) * d.stride_ak : Nat) : ℤ) := by
  have hpm := pmBM_lt31 hd hp
  have hBM := BM_le256 hd
  have hi : pm d p * d.BM + i.1 < 2 ^ 31 + 2 ^ 8 := by have := i.2; omega
  have hsum_toInt : ((r4.item + r5).get1 i).toInt = ((pm d p * d.BM + i.1 : Nat) : ℤ) := by
    have e : (r4.item + r5).get1 i = r4.item + r5.get1 i := rfl
    rw [e, IntV.toInt_add rfl rfl (by rw [h4, h5 i]; exact sfits_add_nat (by omega)),
      h4, h5 i]
    push_cast; ring
  have hmul_toInt : ((r11.item * (r4.item + r5)).get1 i).toInt
      = (d.stride_am : ℤ) * ((pm d p * d.BM + i.1 : Nat) : ℤ) := by
    have e : (r11.item * (r4.item + r5)).get1 i = r11.item * (r4.item + r5).get1 i := rfl
    rw [e, IntV.toInt_mul rfl rfl (by rw [h11, hsum_toInt]; exact sfits_mul (by
      have := hd.strides.1; have := kk.2; nlinarith)), h11, hsum_toInt]
  have hY : ((r12.item * r10).get1 kk).toInt = (d.stride_ak : ℤ) * (kk.1 : ℤ) := by
    have e : (r12.item * r10).get1 kk = r12.item * r10.get1 kk := rfl
    rw [e, IntV.toInt_mul rfl rfl (by rw [h12, h10 kk]; exact sfits_mul (by
      have := hd.strides.2.1; have := BK_le hd; have := kk.2; nlinarith)), h12, h10 kk]
  have e : ((r11.item * (r4.item + r5)).col d.BK + (r12.item * r10).row d.BM).get2 i kk
      = (r11.item * (r4.item + r5)).get1 i + (r12.item * r10).get1 kk := by
    show ((r11.item * (r4.item + r5)).col d.BK + (r12.item * r10).row d.BM).get (i, kk, ()) = _
    rfl
  rw [e, IntV.toInt_add rfl rfl (by
    rw [hmul_toInt, hY]
    exact sfits_sum_mul (sum_mul_lt63 hd.strides.1 hi hd.strides.2.1 (by
      have := BK_le hd; have := kk.2; omega))), hmul_toInt, hY]
  push_cast
  ring

/-- The `B` pointer offsets of the first trip are `kk·stride_bk + (pn·BN + j)·stride_bn`. -/
theorem b_get2_toInt {d : Inst} (hd : d.ok) (p : Nat)
    (r13 r14 : Tensor [] I64) (r7 : Tensor [] I64) (r8 : Tensor [d.BN] I64) (r10 : Tensor [d.BK] I64)
    (h7 : r7.item.toInt = ((pn d p * d.BN : Nat) : ℤ))
    (h8 : ∀ j : Fin d.BN, (r8.get1 j).toInt = (j.1 : ℤ))
    (h10 : ∀ kk : Fin d.BK, (r10.get1 kk).toInt = (kk.1 : ℤ))
    (h13 : r13.item.toInt = (d.stride_bk : ℤ))
    (h14 : r14.item.toInt = (d.stride_bn : ℤ))
    (kk : Fin d.BK) (j : Fin d.BN) :
    (((r13.item * r10).col d.BN + (r14.item * (r7.item + r8)).row d.BK).get2 kk j).toInt
      = (((0 * d.BK + kk.1) * d.stride_bk + (pn d p * d.BN + j.1) * d.stride_bn : Nat) : ℤ) := by
  have hpn := pnBN_lt31 hd p
  have hBN := BN_le256 hd
  have hi : pn d p * d.BN + j.1 < 2 ^ 31 + 2 ^ 8 := by have := j.2; omega
  have hsum_toInt : ((r7.item + r8).get1 j).toInt = ((pn d p * d.BN + j.1 : Nat) : ℤ) := by
    have e : (r7.item + r8).get1 j = r7.item + r8.get1 j := rfl
    rw [e, IntV.toInt_add rfl rfl (by rw [h7, h8 j]; exact sfits_add_nat (by omega)),
      h7, h8 j]
    push_cast; ring
  have hmul_toInt : ((r14.item * (r7.item + r8)).get1 j).toInt
      = (d.stride_bn : ℤ) * ((pn d p * d.BN + j.1 : Nat) : ℤ) := by
    have e : (r14.item * (r7.item + r8)).get1 j = r14.item * (r7.item + r8).get1 j := rfl
    rw [e, IntV.toInt_mul rfl rfl (by rw [h14, hsum_toInt]; exact sfits_mul (by
      have := hd.strides.2.2.2.1; have := kk.2; nlinarith)), h14, hsum_toInt]
  have hY : ((r13.item * r10).get1 kk).toInt = (d.stride_bk : ℤ) * (kk.1 : ℤ) := by
    have e : (r13.item * r10).get1 kk = r13.item * r10.get1 kk := rfl
    rw [e, IntV.toInt_mul rfl rfl (by rw [h13, h10 kk]; exact sfits_mul (by
      have := hd.strides.2.2.1; have := BK_le hd; have := kk.2; nlinarith)), h13, h10 kk]
  have e : ((r13.item * r10).col d.BN + (r14.item * (r7.item + r8)).row d.BK).get2 kk j
      = (r13.item * r10).get1 kk + (r14.item * (r7.item + r8)).get1 j := by
    show ((r13.item * r10).col d.BN + (r14.item * (r7.item + r8)).row d.BK).get (kk, j, ()) = _
    rfl
  rw [e, IntV.toInt_add rfl rfl (by
    rw [hY, hmul_toInt]
    exact sfits_sum_mul (by
      have h := sum_mul_lt63' (s := d.stride_bk) (a := kk.1) (t := d.stride_bn)
        (b := pn d p * d.BN + j.1) hd.strides.2.2.1 (by
          have := BK_le hd; have := kk.2; omega) hd.strides.2.2.2.1 hi
      omega)), hY, hmul_toInt]
  push_cast
  ring

/-- The loop invariant holds at trip `0`. -/
theorem inv_zero (d : Inst) (hd : d.ok) (init : Mem) (p : Nat) (hp : p < grid d)
    (r4 r11 r12 r13 r14 : Tensor [] I64) (r5 : Tensor [d.BM] I64)
    (r7 : Tensor [] I64) (r8 : Tensor [d.BN] I64) (r10 : Tensor [d.BK] I64)
    (h4 : r4.item.toInt = ((pm d p * d.BM : Nat) : ℤ))
    (h5 : ∀ i : Fin d.BM, (r5.get1 i).toInt = (i.1 : ℤ))
    (h7 : r7.item.toInt = ((pn d p * d.BN : Nat) : ℤ))
    (h8 : ∀ j : Fin d.BN, (r8.get1 j).toInt = (j.1 : ℤ))
    (h10 : ∀ kk : Fin d.BK, (r10.get1 kk).toInt = (kk.1 : ℤ))
    (h11 : r11.item.toInt = (d.stride_am : ℤ))
    (h12 : r12.item.toInt = (d.stride_ak : ℤ))
    (h13 : r13.item.toInt = (d.stride_bk : ℤ))
    (h14 : r14.item.toInt = (d.stride_bn : ℤ)) :
    Inv d init p 0
      (r10,
       d.a.ptr 0 + ((r11.item * (r4.item + r5)).col d.BK + (r12.item * r10).row d.BM),
       d.b.ptr 0 + ((r13.item * r10).col d.BN + (r14.item * (r7.item + r8)).row d.BK),
       tl.zeros [d.BM, d.BN] .i32) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro kk
    rw [h10 kk]
    simp
  · intro i kk
    have hZ : (d.a.ptr 0 + ((r11.item * (r4.item + r5)).col d.BK + (r12.item * r10).row d.BM)).get2 i kk
        = (d.a.ptr 0).offset (((r11.item * (r4.item + r5)).col d.BK + (r12.item * r10).row d.BM).get2 i kk).toInt := by
      show (Tensor.ptrAdd (s := [d.BM, d.BK]) (d.a.ptr 0) _).get (i, kk, ()) = _
      rfl
    rw [hZ, a_get2_toInt hd hp r4 r11 r12 r5 r10 h4 h5 h10 h11 h12 i kk]
    apply Ptr.eq_of_addr
    simp [Buf.ptr, Ptr.offset]
  · intro kk j
    have hZ : (d.b.ptr 0 + ((r13.item * r10).col d.BN + (r14.item * (r7.item + r8)).row d.BK)).get2 kk j
        = (d.b.ptr 0).offset (((r13.item * r10).col d.BN + (r14.item * (r7.item + r8)).row d.BK).get2 kk j).toInt := by
      show (Tensor.ptrAdd (s := [d.BK, d.BN]) (d.b.ptr 0) _).get (kk, j, ()) = _
      rfl
    rw [hZ, b_get2_toInt hd p r13 r14 r7 r8 r10 h7 h8 h10 h13 h14 kk j]
    apply Ptr.eq_of_addr
    simp [Buf.ptr, Ptr.offset]
  · intro i j _ _
    have e : (tl.zeros [d.BM, d.BN] .i32).get2 i j = (DType.ofBits .i32 0 : I32) := rfl
    rw [e]
    have h0 : ((DType.ofBits .i32 0 : I32)).toInt = 0 := by
      show (IntV.ofNat 0 : I32).toInt = (0 : ℤ)
      rw [IntV.toInt_ofNat (by norm_num)]
      norm_num
    rw [h0]
    simp

end Claims.ScaledMmInt8
