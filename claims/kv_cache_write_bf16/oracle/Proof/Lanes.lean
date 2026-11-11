import Proof.Basic

/-!
# `kv_cache_write_bf16`: lane arithmetic

The tile positions `tpos`, the mask, the slot pointer, and the `Int`-`Nat` bridges used to
compute the index of every enabled lane.
-/

namespace Claims.KvCacheWrite

open TVA

/-! ## Pointers and buffers -/

theorem Ptr.ext_addr {d : DType} {p q : Ptr d} (h : p.addr = q.addr) : p = q := by
  obtain ⟨a⟩ := p
  obtain ⟨b⟩ := q
  exact congrArg Ptr.mk h

/-- The pointer at index i equals the base pointer offset by i. -/
theorem Buf.ptr_eq_offset {d : DType} (b : Buf d) (i : Int) :
    b.ptr i = (b.ptr 0).offset i := by
  refine Ptr.ext_addr ?_
  simp [Buf.ptr, Ptr.offset]

/-! ## Signed ranges and the values of integer constants -/

theorem sfits32 {x : Int} (h0 : 0 ≤ x) (h : x < 2 ^ 31) : IntV.SFits 32 x := by
  have h' : (2 : Int) ^ (32 - 1) = 2 ^ 31 := by norm_num
  simp only [IntV.SFits, h']
  omega

/-- A nonnegative integer below 2^63 fits in a signed 64-bit integer. -/
theorem sfits64 {x : Int} (h0 : 0 ≤ x) (h : x < 2 ^ 63) : IntV.SFits 64 x := by
  have h' : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  simp only [IntV.SFits, h']
  omega

/-- Converting a natural number below 2^31 to a 32-bit integer and back is the identity. -/
theorem I32_toInt_ofNat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = (n : Int) :=
  IntV.I32.toInt_ofNat h

/-- Converting a natural number below 2^63 to a 64-bit integer and back is the identity. -/
theorem I64_toInt_ofNat {n : Nat} (h : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = (n : Int) :=
  IntV.I64.toInt_ofNat h

/-- Converting an integer that fits in 32 signed bits to a 32-bit integer and back is the identity. -/
theorem I32_toInt_ofInt {x : Int} (h : IntV.SFits 32 x) : (IntV.ofInt x : I32).toInt = x :=
  IntV.toInt_ofInt (w := 32) (by norm_num) h

/-- Converting an integer that fits in 64 signed bits to a 64-bit integer and back is the identity. -/
theorem I64_toInt_ofInt {x : Int} (h : IntV.SFits 64 x) : (IntV.ofInt x : I64).toInt = x :=
  IntV.toInt_ofInt (w := 64) (by norm_num) h

/-- The value of an arange at an index. -/
theorem get_arange' (start : Int) (n : Nat) (l : Idx [n]) :
    (Tensor.arange start n).get l = IntV.ofInt (start + (l.1.val : Int)) := by
  obtain ⟨i, u⟩ := l
  cases u
  simp

/-! ## The tile positions and the mask -/

/-- The tile positions of program `p` (`tile_i * TILE_SIZE + tile_offs`). -/
abbrev tpos (d : Inst) (p : Nat) : Tensor [d.TILE_SIZE] I32 :=
  IntV.ofNat (ti d p) * (IntV.ofNat d.TILE_SIZE : I32) + tl.arange 0 d.TILE_SIZE

/-- The position tensor at a lane is tile index times tile size plus the lane number, in 32-bit arithmetic. -/
theorem tpos_get (d : Inst) (p : Nat) (l : Idx [d.TILE_SIZE]) :
    (tpos d p).get l =
      IntV.ofNat (ti d p) * (IntV.ofNat d.TILE_SIZE : I32) + IntV.ofInt (l.1.val : Int) := by
  simp only [tpos, tl.arange]
  rw [Tensor.get_sadd]
  rw [get_arange']
  simp

/-- The tile index of a program is below 2^31. -/
theorem ti_lt_2_31 {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    ti d p < 2 ^ 31 := by
  have h1 : ti d p < numTiles d := ti_lt hp
  have h2 : numTiles d * d.TILE_SIZE ≤ 2 ^ 31 := tiles_mul_le hd
  have h3 : numTiles d ≤ 2 ^ 31 := by
    have hT : 0 < d.TILE_SIZE := tile_pos' hd
    nlinarith
  omega

/-- The tile size is below 2^31. -/
theorem tile_lt_2_31 {d : Inst} (hd : d.ok) : d.TILE_SIZE < 2 ^ 31 := by
  have h := hd.tile_le
  omega

/-- The value of a tile position. -/
theorem tpos_toInt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d)
    (l : Idx [d.TILE_SIZE]) : ((tpos d p).get l).toInt = (pos d p l : Int) := by
  have hti : ti d p < 2 ^ 31 := ti_lt_2_31 hd hp
  have hT : d.TILE_SIZE < 2 ^ 31 := tile_lt_2_31 hd
  have hl : (l.1.val : Int) < 2 ^ 31 := by
    have := l.1.isLt
    have hle := hd.tile_le
    omega
  have h1 : ((IntV.ofNat (ti d p) : I32)).toInt = (ti d p : Int) := I32_toInt_ofNat hti
  have h2 : ((IntV.ofNat d.TILE_SIZE : I32)).toInt = (d.TILE_SIZE : Int) := I32_toInt_ofNat hT
  have h3 : ((IntV.ofInt (l.1.val : Int) : I32)).toInt = (l.1.val : Int) :=
    I32_toInt_ofInt (sfits32 (by positivity) hl)
  have hp' : pos d p l < 2 ^ 31 := pos_lt hd hp l
  have hmulT : ti d p * d.TILE_SIZE < 2 ^ 31 :=
    lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right (ti_lt hp) (tile_pos' hd)) (tiles_mul_le hd)
  have hmul : ((IntV.ofNat (ti d p) : I32) * (IntV.ofNat d.TILE_SIZE : I32)).toInt =
      ((ti d p * d.TILE_SIZE : Nat) : Int) :=
    IntV.I32.mul_nat h1 h2 hmulT
  have hadd : ((IntV.ofNat (ti d p) : I32) * (IntV.ofNat d.TILE_SIZE : I32) +
      IntV.ofInt (l.1.val : Int)).toInt = ((ti d p * d.TILE_SIZE + l.1.val : Nat) : Int) :=
    IntV.I32.add_nat hmul h3 (by simp only [pos] at hp'; omega)
  rw [tpos_get, hadd]
  simp only [pos]

/-- The mask `tile_pos < num_heads * head_size`. -/
abbrev maskT (d : Inst) (p : Nat) : Tensor [d.TILE_SIZE] Bool :=
  (tpos d p).ltS (IntV.ofNat (numVals d) : I32)

/-- The mask at a lane holds exactly when the lane's position is below the number of values. -/
theorem maskT_get {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d)
    (l : Idx [d.TILE_SIZE]) : (maskT d p).get l = decide (pos d p l < numVals d) := by
  rw [maskT, Tensor.get_ltS, IntV.lt_eq, tpos_toInt hd hp, I32_toInt_ofNat hd.vals_lt]
  simp

/-- For a nonnegative slot, a lane's enabled flag equals the kernel's mask at that lane. -/
theorem en_eq_mask {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d) (hsl : 0 ≤ slotAt d init (tk d p))
    (l : Idx [d.TILE_SIZE]) : en d init p l = (maskT d p).get l := by
  rw [maskT_get hd hp]
  have h : decide (0 ≤ slotAt d init (tk d p)) = true := decide_eq_true hsl
  simp only [en, h, Bool.and_true]

/-- The program's slot pointer is the pointer of its token. -/
theorem slot_ptr_eq (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    d.slot_mapping.ptr 0 + tl.program_id ((launch d).ctx i) 0 =
      d.slot_mapping.ptr (tk d (i : Nat)) := by
  have hp : (i : Nat) < d.num_tokens * numTiles d := by
    simpa [launch_size] using i.isLt
  have hpid : ((launch d).ctx i).pid 0 < 2 ^ 31 := by
    rw [launch_pid0]
    exact Nat.lt_of_lt_of_le (tk_lt hp) hd.tokens_lt.le
  refine Ptr.ext_addr ?_
  rw [Ptr.add_int]
  simp only [Ptr.offset_addr, Buf.ptr_addr, tl.program_id]
  rw [I32_toInt_ofNat hpid, launch_pid0]
  simp only [tk]
  ring

end Claims.KvCacheWrite
