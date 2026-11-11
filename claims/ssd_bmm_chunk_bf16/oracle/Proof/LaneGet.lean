import Claim
import TVA.Proof
import Proof.Inv

/-!
# The lane values of the pointer tiles

Each pointer tile defined in `Proof/Lane.lean` is a base pointer plus a scalar `int64` offset
plus an `int32` tile offset. This file proves that no offset wraps and records the lane value of
each tile: lane `(mll, kk)` of `a_ptrs` is the pointer to element
`aIdx d (rowA d i mll) (hOf d i) kk` of `a`, and so on for `b_ptrs`, the two `seq_idx` pointer
vectors and `out_ptrs`.
-/

namespace Claims.SsdBmm

open TVA

/-! ## The instance's bounds -/

/-- `chunk_size² < 2^31` (`out.stride(1) = ngroups · C²` is an `int32`). -/
theorem C_sq_lt {d : Inst} (hd : d.ok) : d.C * d.C < 2 ^ 31 := by
  have h := hd.oC_lt
  unfold oC at h
  have := Nat.le_mul_of_pos_left (d.C * d.C) hd.G_pos
  omega

/-- `chunk_size < 2^31`. -/
theorem C_lt_31 {d : Inst} (hd : d.ok) : d.C < 2 ^ 31 := by
  have := C_sq_lt hd
  have := Nat.le_mul_of_pos_left d.C (show 0 < d.C by have := hd.C_ge; omega)
  omega

/-- `chunk_size ≤ 46340`, so a row offset `m·64 + lane` is below `2^31`. -/
theorem C_le_46340 {d : Inst} (hd : d.ok) : d.C ≤ 46340 := by
  have h := C_sq_lt hd
  by_contra hc
  have h3 : 46341 * 46341 ≤ d.C * d.C := Nat.mul_le_mul (by omega) (by omega)
  omega

/-- The group of a program is below `2^31`. -/
theorem hOf_lt_31 {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : hOf d i < 2 ^ 31 := by
  have := hOf_lt hd i
  have := G_lt_31 hd
  omega

/-- The chunk of a program is below `2^31`. -/
theorem cOf_lt_31 {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : cOf d i < 2 ^ 31 := by
  have := cOf_mul_C_lt hd i
  have := hd.T_lt
  have := Nat.le_mul_of_pos_right (cOf d i) (show 0 < d.C by have := hd.C_ge; omega)
  omega

/-- The chunk's first token `pid_c · chunk_size` is below `2^31`. -/
theorem cOf_mul_C_lt_31 {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    cOf d i * d.C < 2 ^ 31 :=
  lt_trans (cOf_mul_C_lt hd i) hd.T_lt

/-! ## The scalar offsets -/

/-- The `int64` offset `0 + x·y·z + u·v` of naturals does not wrap when the sum is below `2^63`
(`pid_c · chunk_size · stride + pid_h · stride`, lines 125–129). -/
theorem toInt_ofs3 {x y z u v : Nat} (h : x * y * z + u * v < 2 ^ 63) :
    ((IntV.ofInt 0 : I64) + IntV.ofInt (x : ℤ) * IntV.ofInt (y : ℤ) * IntV.ofInt (z : ℤ) +
        IntV.ofInt (u : ℤ) * IntV.ofInt (v : ℤ)).toInt = (x : ℤ) * y * z + u * v := by
  simp only [IntV.ofInt_mul_ofInt, IntV.ofInt_add_ofInt, zero_add]
  have h0 : (0 : ℤ) ≤ (x : ℤ) * y * z + u * v := by positivity
  exact IntV.I64.toInt_ofInt ⟨by linarith [(by positivity : (0 : ℤ) ≤ 2 ^ 63)], by exact_mod_cast h⟩

/-- The `int64` offset `0 + x·y·z` of naturals does not wrap when the product is below `2^63`
(line 138). -/
theorem toInt_ofs1 {x y z : Nat} (h : x * y * z < 2 ^ 63) :
    ((IntV.ofInt 0 : I64) + IntV.ofInt (x : ℤ) * IntV.ofInt (y : ℤ) * IntV.ofInt (z : ℤ)).toInt =
      (x : ℤ) * y * z := by
  simp only [IntV.ofInt_mul_ofInt, IntV.ofInt_add_ofInt, zero_add]
  have h0 : (0 : ℤ) ≤ (x : ℤ) * y * z := by positivity
  exact IntV.I64.toInt_ofInt ⟨by linarith [(by positivity : (0 : ℤ) ≤ 2 ^ 63)], by exact_mod_cast h⟩

/-- The `int64` offset `0 + x·y + u·v` of naturals does not wrap when the sum is below `2^63`
(`pid_c · stride_out_chunk + pid_h · stride_out_head`, lines 182–184). -/
theorem toInt_ofs2 {x y u v : Nat} (h : x * y + u * v < 2 ^ 63) :
    ((IntV.ofInt 0 : I64) + IntV.ofInt (x : ℤ) * IntV.ofInt (y : ℤ) +
        IntV.ofInt (u : ℤ) * IntV.ofInt (v : ℤ)).toInt = (x : ℤ) * y + u * v := by
  simp only [IntV.ofInt_mul_ofInt, IntV.ofInt_add_ofInt, zero_add]
  have h0 : (0 : ℤ) ≤ (x : ℤ) * y + u * v := by positivity
  exact IntV.I64.toInt_ofInt ⟨by linarith [(by positivity : (0 : ℤ) ≤ 2 ^ 63)], by exact_mod_cast h⟩

/-- A product `x·y < 2^31` times `z < 2^31`, and a product of two naturals below `2^31`, add up
below `2^63`. -/
theorem ofs_lt {x y z u v : Nat} (hxy : x * y < 2 ^ 31) (hz : z < 2 ^ 31) (hu : u < 2 ^ 31)
    (hv : v < 2 ^ 31) : x * y * z + u * v < 2 ^ 63 := by
  have h1 := Nat.mul_lt_mul'' hxy hz
  have h2 := Nat.mul_lt_mul'' hu hv
  omega

/-- `pid_c · chunk_size · stride_a_seqlen + pid_h · stride_a_head` (line 129). -/
theorem toInt_aOfs1 {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    (aOfs1 d i).toInt = (cOf d i : ℤ) * d.C * d.aS + (hOf d i : ℤ) * d.aH :=
  toInt_ofs3 (x := cOf d i) (ofs_lt (cOf_mul_C_lt_31 hd i) hd.aS_lt (hOf_lt_31 hd i) hd.aH_lt)

/-- `pid_c · chunk_size · stride_b_seqlen + pid_h · stride_b_head` (line 134). -/
theorem toInt_bOfs1 {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    (bOfs1 d i).toInt = (cOf d i : ℤ) * d.C * d.bS + (hOf d i : ℤ) * d.bH :=
  toInt_ofs3 (x := cOf d i) (ofs_lt (cOf_mul_C_lt_31 hd i) hd.bS_lt (hOf_lt_31 hd i) hd.bH_lt)

/-- `pid_c · chunk_size · stride_seq_idx_seqlen` (line 138; the stride is `1`). -/
theorem toInt_sOfs1 {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    (sOfs1 d i).toInt = (cOf d i : ℤ) * d.C := by
  have h : cOf d i * d.C * 1 < 2 ^ 63 := by
    have := cOf_mul_C_lt_31 hd i
    omega
  unfold sOfs1
  exact toInt_ofs1 (x := cOf d i) (y := d.C) (z := 1) h |>.trans (by ring)

/-- `pid_c · stride_out_chunk + pid_h · stride_out_head` (line 184). -/
theorem toInt_outOfs1 {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    (outOfs1 d i).toInt = (cOf d i : ℤ) * oC d + (hOf d i : ℤ) * oH d := by
  have h1 := Nat.mul_lt_mul'' (cOf_lt_31 hd i) hd.oC_lt
  have hoH : oH d < 2 ^ 31 := C_sq_lt hd
  have h2 := Nat.mul_lt_mul'' (hOf_lt_31 hd i) hoH
  unfold outOfs1
  exact toInt_ofs2 (x := cOf d i) (by omega)

/-! ## Pointer arithmetic -/

/-- A base pointer moved by a scalar `int64` offset and by a lane's `int32` offset is the buffer's
pointer to the sum. -/
theorem ptr_lane {δ : DType} (b : Buf δ) (o1 : I64) (x : I32) {j : ℤ}
    (h : o1.toInt + x.toInt = j) : ((b.ptr 0 + o1).offset x.toInt) = b.ptr j := by
  subst h
  show ((b.ptr 0).offset o1.toInt).offset x.toInt = _
  simp only [Buf.ptr, Ptr.offset, Ptr.mk.injEq]
  ring

/-! ## The tile offsets -/

/-- `num_pid_n` as `numPidN`. -/
theorem numPidN_eq {d : Inst} : (d.C + (BN - 1)) / BN = numPidN d := by
  unfold numPidN
  have h : d.C + (BN - 1) = d.C + BN - 1 := by unfold BN; omega
  rw [h]

/-- `pid_m` as `mOf`. -/
theorem mOf_cast {d : Inst} (i : Fin (launch d).size) :
    ((↑i % gx d / ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) = (mOf d i : ℤ) := by
  rw [numPidN_eq]
  rfl

/-- `pid_n` as `nOf`. -/
theorem nOf_cast {d : Inst} (i : Fin (launch d).size) :
    ((↑i % gx d % ((d.C + (BN - 1)) / BN) : ℕ) : ℤ) = (nOf d i : ℤ) := by
  rw [numPidN_eq]
  rfl

/-- A row offset `pid_m · 64 + lane` is below `2^31`. -/
theorem mOf_offs_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) :
    mOf d i * BM + mll.val < 2 ^ 31 := by
  have := mOf_mul_BM_lt hd i
  have := C_le_46340 hd
  have := mll.isLt
  unfold BM at *
  omega

/-- A column offset `pid_n · 128 + lane` is below `2^31`. -/
theorem nOf_offs_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (nn : Fin BN) :
    nOf d i * BN + nn.val < 2 ^ 31 := by
  have := nOf_mul_BN_lt hd i
  have := C_le_46340 hd
  have := nn.isLt
  unfold BN at *
  omega

/-- The first row `pid_m · 64` of a program's tile, as an `int32` product (line 140). -/
theorem pidM_mul_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ((IntV.ofInt (↑(↑i % gx d / ((d.C + (BN - 1)) / BN)) : ℤ) : I32) *
      (IntV.ofNat BM : I32)).toInt = ((mOf d i * BM : ℕ) : ℤ) := by
  have h1 := mOf_mul_BM_lt hd i
  have h2 := C_le_46340 hd
  have h3 : mOf d i ≤ mOf d i * BM := Nat.le_mul_of_pos_right _ BM_pos
  refine IntV.I32.mul_nat ?_ (IntV.I32.toInt_ofNat (by norm_num [BM])) (by omega)
  rw [mOf_cast]; exact IntV.I32.toInt_ofInt ⟨by omega, by omega⟩

/-- The first column `pid_n · 128` of a program's tile, as an `int32` product (line 141). -/
theorem pidN_mul_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ((IntV.ofInt (↑(↑i % gx d % ((d.C + (BN - 1)) / BN)) : ℤ) : I32) *
      (IntV.ofNat BN : I32)).toInt = ((nOf d i * BN : ℕ) : ℤ) := by
  have h1 := nOf_mul_BN_lt hd i
  have h2 := C_le_46340 hd
  have h3 : nOf d i ≤ nOf d i * BN := Nat.le_mul_of_pos_right _ BN_pos
  refine IntV.I32.mul_nat ?_ (IntV.I32.toInt_ofNat (by norm_num [BN])) (by omega)
  rw [nOf_cast]; exact IntV.I32.toInt_ofInt ⟨by omega, by omega⟩

/-- Lane `mll` of `offs_m` (line 140) has the value `pid_m · 64 + mll`. -/
theorem offsM_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) :
    ((offsM d i).get (mll, ())).toInt = ((mOf d i * BM + mll.val : ℕ) : ℤ) := by
  have := mOf_mul_BM_lt hd i
  have := C_le_46340 hd
  have : BM = 64 := rfl
  exact IntV.I32.toInt_sadd_arange (pidM_mul_toInt hd i) (by simp only [BM] at *; omega) mll

/-- Lane `nn` of `offs_n` (line 141) has the value `pid_n · 128 + nn`. -/
theorem offsN_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (nn : Fin BN) :
    ((offsN d i).get (nn, ())).toInt = ((nOf d i * BN + nn.val : ℕ) : ℤ) := by
  have := nOf_mul_BN_lt hd i
  have := C_le_46340 hd
  have : BN = 128 := rfl
  exact IntV.I32.toInt_sadd_arange (pidN_mul_toInt hd i) (by simp only [BN] at *; omega) nn

/-- Lane `kk` of `offs_k` (line 142) has the value `kk`. -/
theorem offsK_toInt (kk : Fin BK) : ((tl.arange 0 BK).get (kk, ())).toInt = kk.val :=
  IntV.I32.toInt_tl_arange (by norm_num [BK]) kk

/-- Lane `(mll, kk)` of the `int32` offset of `a_ptrs` (line 143), on a row below `chunk_size`. -/
theorem aOfs2_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) (kk : Fin BK)
    (hm : mOf d i * BM + mll.val < d.C) :
    ((aOfs2 d i).get (mll, kk, ())).toInt =
      (((mOf d i * BM + mll.val) * d.aS + kk.val * d.aK : ℕ) : ℤ) := by
  have h1 := Nat.mul_le_mul_right d.aS (show mOf d i * BM + mll.val ≤ d.C - 1 by omega)
  have h2 := Nat.mul_le_mul_right d.aK (show kk.val ≤ BK - 1 by have := kk.isLt; omega)
  have h3 := hd.a_fit
  exact IntV.add_nat (IntV.I32.mul_nat (offsM_toInt hd i mll) (IntV.I32.toInt_ofNat hd.aS_lt)
    (by omega)) (IntV.I32.mul_nat (offsK_toInt kk) (IntV.I32.toInt_ofNat hd.aK_lt) (by omega))
    (by omega)

/-- Lane `(kk, nn)` of the `int32` offset of `b_ptrs` (line 144), on a column below `chunk_size`. -/
theorem bOfs2_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (kk : Fin BK) (nn : Fin BN)
    (hn : nOf d i * BN + nn.val < d.C) :
    ((bOfs2 d i).get (kk, nn, ())).toInt =
      (((kk.val * d.bK) + (nOf d i * BN + nn.val) * d.bS : ℕ) : ℤ) := by
  have h1 := Nat.mul_le_mul_right d.bS (show nOf d i * BN + nn.val ≤ d.C - 1 by omega)
  have h2 := Nat.mul_le_mul_right d.bK (show kk.val ≤ BK - 1 by have := kk.isLt; omega)
  have h3 := hd.b_fit
  exact IntV.add_nat (IntV.I32.mul_nat (offsK_toInt kk) (IntV.I32.toInt_ofNat hd.bK_lt)
    (by omega)) (IntV.I32.mul_nat (offsN_toInt hd i nn) (IntV.I32.toInt_ofNat hd.bS_lt)
    (by omega)) (by omega)

/-- Lane `(mll, nn)` of the `int32` offset of `out_ptrs` (line 185), on a tile element inside the
chunk. -/
theorem outOfs2_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) (nn : Fin BN)
    (hm : mOf d i * BM + mll.val < d.C) (hn : nOf d i * BN + nn.val < d.C) :
    ((outOfs2 d i).get (mll, nn, ())).toInt =
      (((d.C * (mOf d i * BM + mll.val)) + (nOf d i * BN + nn.val) : ℕ) : ℤ) := by
  have hC : 0 < d.C := by have := hd.C_ge; omega
  have h1 : d.C * (mOf d i * BM + mll.val) < d.C * d.C := (Nat.mul_lt_mul_left hC).mpr hm
  have h2 : d.C * (mOf d i * BM + mll.val) + (nOf d i * BN + nn.val) < d.C * d.C := by
    have : d.C * (mOf d i * BM + mll.val) + d.C ≤ d.C * d.C := by
      rw [← Nat.mul_succ]; exact Nat.mul_le_mul_left _ hm
    omega
  have h3 := C_sq_lt hd
  have hone : (IntV.ofNat 1 : I32).toInt = ((1 : ℕ) : ℤ) := IntV.I32.toInt_ofNat (by norm_num)
  have hN := IntV.I32.mul_nat (offsN_toInt hd i nn) hone (by omega)
  exact IntV.add_nat (IntV.I32.mul_nat (IntV.I32.toInt_ofNat (C_lt_31 hd)) (offsM_toInt hd i mll)
    (by omega)) (by simp only [Nat.mul_one] at hN; exact hN) (by omega)

/-! ## The pointer lanes -/

/-- Lane `(mll, kk)` of `a_ptrs` points to `a[aIdx (row) (group) kk]`. -/
theorem aPtrs_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) (kk : Fin BK)
    (hm : mOf d i * BM + mll.val < d.C) :
    (aPtrs d i).get (mll, kk, ()) =
      d.a.ptr (aIdx d (rowA d i mll.val) (hOf d i) kk.val : ℤ) :=
  ptr_lane d.a _ _ (by
    rw [toInt_aOfs1 hd i, aOfs2_toInt hd i mll kk hm]
    simp only [aIdx, rowA]; push_cast; ring)

/-- Lane `(kk, nn)` of `b_ptrs` points to `b[bIdx (column) (group) kk]`. -/
theorem bPtrs_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (kk : Fin BK) (nn : Fin BN)
    (hn : nOf d i * BN + nn.val < d.C) :
    (bPtrs d i).get (kk, nn, ()) =
      d.b.ptr (bIdx d (colB d i nn.val) (hOf d i) kk.val : ℤ) :=
  ptr_lane d.b _ _ (by
    rw [toInt_bOfs1 hd i, bOfs2_toInt hd i kk nn hn]
    simp only [bIdx, colB]; push_cast; ring)

/-- Lane `mll` of the `seq_idx` pointers of the rows (line 171) points to `seq_idx[row]`. -/
theorem seqPtrsM_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) :
    (seqPtrsM d i).get (mll, ()) = d.seqIdx.ptr (rowA d i mll.val : ℤ) :=
  ptr_lane d.seqIdx _ _ (by
    have := IntV.I32.mul_nat (offsM_toInt hd i mll) (IntV.I32.toInt_ofNat (by norm_num : 1 < 2 ^ 31))
      (by have := mOf_offs_lt hd i mll; omega)
    rw [toInt_sOfs1 hd i, show ((offsM d i * (IntV.ofNat 1 : I32)).get (mll, ())).toInt = _ from
      by simp only [Nat.mul_one] at this; exact this]
    simp only [rowA]; push_cast; ring)

/-- Lane `nn` of the `seq_idx` pointers of the columns (line 176) points to `seq_idx[column]`. -/
theorem seqPtrsN_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (nn : Fin BN) :
    (seqPtrsN d i).get (nn, ()) = d.seqIdx.ptr (colB d i nn.val : ℤ) :=
  ptr_lane d.seqIdx _ _ (by
    have := IntV.I32.mul_nat (offsN_toInt hd i nn) (IntV.I32.toInt_ofNat (by norm_num : 1 < 2 ^ 31))
      (by have := nOf_offs_lt hd i nn; omega)
    rw [toInt_sOfs1 hd i, show ((offsN d i * (IntV.ofNat 1 : I32)).get (nn, ())).toInt = _ from
      by simp only [Nat.mul_one] at this; exact this]
    simp only [colB]; push_cast; ring)

/-- Lane `(mll, nn)` of `out_ptrs` points to `out[oIdx chunk group row column]`. -/
theorem outPtrs_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (mll : Fin BM) (nn : Fin BN)
    (hm : mOf d i * BM + mll.val < d.C) (hn : nOf d i * BN + nn.val < d.C) :
    (outPtrs d i).get (mll, nn, ()) =
      d.out.ptr (oIdx d (cOf d i) (hOf d i) (mOf d i * BM + mll.val) (nOf d i * BN + nn.val) : ℤ) :=
  ptr_lane d.out _ _ (by
    rw [toInt_outOfs1 hd i, outOfs2_toInt hd i mll nn hm hn]
    simp only [oIdx]; push_cast; ring)

end Claims.SsdBmm
