import Claim
import TVA.Proof

/-!
# Lanes of the kernel's blocks

The kernel's three loops run once (`N ≤ BLOCK_SIZE`), with `cols = tl.arange(0, BLOCK_SIZE)`; lane
`i` addresses element `t * N + i` of a row's buffer and is enabled when `i < N`.
-/

namespace Claims.LayerNormFwdF16

open TVA

/-- A loop `range(0, N, BLOCK)` that runs once (`0 < N ≤ BLOCK`) is its body at `0`. -/
theorem forRange_one {σ : Type} {N B : Nat} (hN : 0 < N) (hNB : N ≤ B) (hB : B < 2 ^ 30) (init : σ)
    (body : I32 → σ → Prog σ) :
    tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat N) (IntV.ofNat B) init body =
      body (IntV.ofInt 0) init := by
  have hN' : (IntV.ofNat N : I32).toInt = N := IntV.I32.toInt_ofNat (by omega)
  have hB' : (IntV.ofNat B : I32).toInt = B := IntV.I32.toInt_ofNat (by omega)
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 := IntV.toInt_ofInt_zero
  have hr : tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat N) (IntV.ofNat B) := by
    unfold tl.RangeOk
    simp only [hN', hB', h0, IntV.maxInt, ite_true]
    have : (2 : Int) ^ (32 - 1) = 2147483648 := by norm_num
    rw [this]
    have : (B : Int) < 1073741824 := by omega
    omega
  have ht : tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofNat N : I32).toInt
      (IntV.ofNat B : I32).toInt = 1 := by
    simp only [hN', hB', h0]
    unfold tl.tripCount
    have h : ((N : Int) - 0 + B - 1) / B = 1 := by
      have e : ((N : Int) - 0 + B - 1) = ((N : Int) - 1) + B * 1 := by ring
      rw [e, Int.add_mul_ediv_left _ _ (by omega : (B : Int) ≠ 0),
        Int.ediv_eq_zero_of_lt (by omega) (by omega)]
      rfl
    rw [h]; rfl
  unfold tl.forRange
  rw [ite_eq_left hr, ht]
  simp [tl.forN, h0]

/-- The padding `other=0.` of the loads of `X`: every lane is `+0` in `float16`. -/
theorem safe_other16 {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {B : Nat}
    {k : Tensor [B] F16 → Prog α}
    (hk : ∀ o : Tensor [B] F16, (∀ i, o.get i = FloatV.ofPat 0) → Safe E R W Q m (k o)) :
    Safe E R W Q m (other16 B >>= k) := by
  have h0 : FloatV.NaNEq (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0) : F32) : F16) (FloatV.ofPat 0) := by
    rw [FloatV.litF32_ofPat_zero, FloatV.cvt_ofPat_zero Format.binary32_regular]
    exact Or.inl rfl
  unfold other16
  refine safe_convert (fun i => ⟨FloatV.ofPat 0, ?_⟩) fun o ho => hk o fun i => ?_
  · exact ConvRel.f32_f16_iff.mpr h0
  · have : FloatV.NaNEq (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0) : F32) : F16) (o.get i) :=
      ConvRel.f32_f16_iff.mp (ho i)
    have hn : (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0) : F32) : F16).isNaN = false := by
      rw [FloatV.litF32_ofPat_zero, FloatV.cvt_ofPat_zero Format.binary32_regular]
      exact FloatV.isNaN_ofPat_zero Format.binary16_regular
    rw [this.eq hn, FloatV.litF32_ofPat_zero, FloatV.cvt_ofPat_zero Format.binary32_regular]


/-- `BLOCK_SIZE ≤ 32768` on a covered instance. -/
theorem Inst.ok.block_le {d : Inst} (hd : d.ok) : d.BLOCK ≤ 32768 := by
  obtain ⟨e, he⟩ := hd.block_pow2
  have := hd.n_le; have := hd.block_lt
  by_contra h
  have h16 : 16 ≤ e := by
    by_contra h'
    have : 2 ^ e ≤ 2 ^ 15 := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  have : 2 ^ 16 ≤ 2 ^ e := Nat.pow_le_pow_right (by norm_num) h16
  omega

/-- `off + tl.arange 0 B` with `off = 0`: lane `i` has the value `i`. -/
theorem cols_toInt {B : Nat} (hB : B ≤ 2 ^ 31) (i : Idx [B - 0]) :
    (((IntV.ofInt 0 : I32) + tl.arange 0 B).get i).toInt = (i.1 : Nat) := by
  obtain ⟨i, ⟨⟩⟩ := i
  have := IntV.I32.toInt_sadd_arange (c := (IntV.ofInt 0 : I32)) (r0 := 0) (n := B)
    (by simpa using IntV.toInt_ofInt_zero) (by omega) i
  rw [Nat.zero_add] at this
  exact this

/-- The mask `cols < N` of lane `i`. -/
theorem cols_mask {B N : Nat} (hB : B ≤ 2 ^ 31) (hN : N < 2 ^ 31) (i : Idx [B - 0]) :
    (((IntV.ofInt 0 : I32) + tl.arange 0 B).ltS (IntV.ofNat N : I32)).get i = decide ((i.1 : Nat) < N) := by
  have h2 : (IntV.ofNat N : I32).toInt = N := IntV.I32.toInt_ofNat hN
  simp only [Tensor.get_ltS, IntV.lt_eq, cols_toInt hB i, h2]
  norm_cast


/-- Program `t`'s row start `row * stride` is `t * N`, without wrapping. -/
theorem Inst.ok.row_start {d : Inst} (hd : d.ok) {c : Ctx} {t : Nat} (ht : t < d.M) (hc : c.pid 0 = t) :
    (tl.program_id c 0 * (IntV.ofNat d.N : I32)).toInt = ((t * d.N : Nat) : Int) := by
  have h1 := hd.rows_lt
  have h2 : t * d.N ≤ (d.M - 1) * d.N := Nat.mul_le_mul_right _ (by omega)
  have hpid : (tl.program_id c 0).toInt = (t : Int) := by
    rw [tl.program_id, hc]
    exact IntV.I32.toInt_ofNat (by nlinarith [hd.n_ge])
  exact IntV.I32.mul_nat hpid (IntV.I32.toInt_ofNat (by have := hd.n_le; omega)) (by simpa using (by omega : t * d.N < 2 ^ 31))

/-- Lane `i` of the pointer `B + row * stride + cols` is element `t * N + i`. -/
theorem row_ptr {d : Inst} (hd : d.ok) {c : Ctx} {t : Nat} (ht : t < d.M) (hc : c.pid 0 = t)
    {dy : DType} (B : Buf dy) (i : Idx [d.BLOCK - 0]) :
    (B.ptr 0 + tl.program_id c 0 * (IntV.ofNat d.N : I32) +
      ((IntV.ofInt 0 : I32) + tl.arange 0 d.BLOCK)).get i = B.ptr ((idx d t i.1 : Nat) : Int) := by
  have h := hd.row_start ht hc
  have := cols_toInt (B := d.BLOCK) (by have := hd.block_le; omega) i
  show (B.ptr 0 + tl.program_id c 0 * (IntV.ofNat d.N : I32)).offset _ = _
  rw [Ptr.add_int, Buf.ptr_offset, h]
  rw [this, Buf.ptr_offset]
  simp only [idx]
  push_cast
  ring_nf


/-- Lane `i` of the pointer `B + cols` (the loads of `weight` and `bias`) is element `i`. -/
theorem vec_ptr {n : Nat} (hn : n ≤ 2 ^ 31) {dy : DType} (B : Buf dy) (i : Idx [n - 0]) :
    (B.ptr 0 + ((IntV.ofInt 0 : I32) + tl.arange 0 n)).get i = B.ptr ((i.1 : Nat) : Int) := by
  have := cols_toInt hn i
  show (B.ptr 0).offset _ = _
  rw [this, Buf.ptr_offset]
  simp

end Claims.LayerNormFwdF16
