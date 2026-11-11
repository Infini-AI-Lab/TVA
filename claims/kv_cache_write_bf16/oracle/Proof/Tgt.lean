import Proof.Idx

/-!
# `kv_cache_write_bf16`: addresses of the source and destination elements

The pointers the model computes name the elements `giK`, `giV` and `gi`, and the `Buf.Addrs`
facts the safe rules need.
-/

namespace Claims.KvCacheWrite

open TVA

/-! ## Widening casts keep the value -/

/-- A 32-bit signed value fits in 64 bits. -/
theorem sfits32_64 {x : Int} (h : IntV.SFits 32 x) : IntV.SFits 64 x := by
  have h1 : (2 : Int) ^ (32 - 1) = 2 ^ 31 := by norm_num
  have h2 : (2 : Int) ^ (64 - 1) = 2 ^ 63 := by norm_num
  simp only [IntV.SFits, h1, h2] at h ⊢
  omega

/-- A 32-bit to 64-bit integer conversion preserves the integer value. -/
theorem conv_i32_i64_toInt {P : Platform} {x : I32} {r : I64}
    (h : ConvRel P .i32 .i64 x r) : r.toInt = x.toInt := by
  simp only [ConvRel, ofIntRel] at h
  rw [h, I64_toInt_ofInt (sfits32_64 (IntV.sfits_toInt x))]

/-- A 64-bit to 64-bit integer conversion preserves the integer value. -/
theorem conv_i64_i64_toInt {P : Platform} {x r : I64}
    (h : ConvRel P .i64 .i64 x r) : r.toInt = x.toInt := by
  simp only [ConvRel, ofIntRel] at h
  rw [h, I64_toInt_ofInt (IntV.sfits_toInt x)]

/-! ## Bridge from `Int` division to `Nat` division -/

theorem int_div_eq_nat_div {a : Int} (ha : 0 ≤ a) {b : Nat} :
    a / (b : Int) = ((a.toNat / b : Nat) : Int) := by
  conv_lhs => rw [← Int.toNat_of_nonneg ha]
  exact Int.natCast_ediv _ _

/-- For a nonnegative integer, the integer remainder by a natural number equals the natural remainder of its natural form. -/
theorem int_mod_eq_nat_mod {a : Int} (ha : 0 ≤ a) {b : Nat} :
    a % (b : Int) = ((a.toNat % b : Nat) : Int) := by
  conv_lhs => rw [← Int.toNat_of_nonneg ha]
  exact Int.natCast_emod _ _

/-! ## `toInt` of sums and products of `int64`s in range -/

theorem toInt_mul64 {a b : I64} (ha : 0 ≤ a.toInt) (hb : 0 ≤ b.toInt)
    (h : a.toInt * b.toInt < 2 ^ 63) : (a * b).toInt = a.toInt * b.toInt :=
  IntV.toInt_mul rfl rfl (sfits64 (mul_nonneg ha hb) h)

/-- Adding four nonnegative 64-bit integers whose true sum is below 2^63 gives the true sum. -/
theorem toInt_add4 {a b c e : I64} (ha : 0 ≤ a.toInt) (hb : 0 ≤ b.toInt) (hc : 0 ≤ c.toInt)
    (he : 0 ≤ e.toInt) (h : a.toInt + b.toInt + c.toInt + e.toInt < 2 ^ 63) :
    (((a + b) + c) + e).toInt = a.toInt + b.toInt + c.toInt + e.toInt := by
  have hab : (a + b).toInt = a.toInt + b.toInt :=
    IntV.toInt_add rfl rfl (sfits64 (by omega) (by omega))
  have habc : ((a + b) + c).toInt = (a + b).toInt + c.toInt := by
    refine IntV.toInt_add rfl rfl (sfits64 ?_ ?_)
    · rw [hab]; omega
    · rw [hab]; omega
  have hall : (((a + b) + c) + e).toInt = ((a + b) + c).toInt + e.toInt := by
    refine IntV.toInt_add rfl rfl (sfits64 ?_ ?_)
    · rw [habc, hab]; omega
    · rw [habc, hab]; omega
  rw [hall, habc, hab]

/-! ## The source elements -/

/-- The value of the token part of the key offset. -/
theorem key_idx_toInt {d : Inst} (hd : d.ok) {p : Nat}
    (hp : p < d.num_tokens * numTiles d) {tok : I64} (hst : tok.toInt = (tk d p : Int)) :
    (tok * (IntV.ofNat d.key_stride : I64)).toInt = (tk d p * d.key_stride : Nat) := by
  have hk : (IntV.ofNat d.key_stride : I64).toInt = (d.key_stride : Int) :=
    I64_toInt_ofNat hd.strides_lt.1
  have hb : tk d p * d.key_stride < 2 ^ 63 := by
    have h := (hd.key_len (tk d p) 0 (tk_lt hp) (numVals_pos hd hp)).le
    have h2 := hd.lens_lt.1
    simp only [keyIdx, Nat.add_zero] at h
    omega
  rw [toInt_mul64 (by rw [hst]; positivity) (by rw [hk]; positivity) (by
    rw [hst, hk]; exact_mod_cast hb), hst, hk]
  push_cast
  ring

/-- The value of the token part of the value offset. -/
theorem value_idx_toInt {d : Inst} (hd : d.ok) {p : Nat}
    (hp : p < d.num_tokens * numTiles d) {tok : I64} (hst : tok.toInt = (tk d p : Int)) :
    (tok * (IntV.ofNat d.value_stride : I64)).toInt = (tk d p * d.value_stride : Nat) := by
  have hk : (IntV.ofNat d.value_stride : I64).toInt = (d.value_stride : Int) :=
    I64_toInt_ofNat hd.strides_lt.2.1
  have hb : tk d p * d.value_stride < 2 ^ 63 := by
    have h := (hd.value_len (tk d p) 0 (tk_lt hp) (numVals_pos hd hp)).le
    have h2 := hd.lens_lt.2.1
    simp only [valueIdx, Nat.add_zero] at h
    omega
  rw [toInt_mul64 (by rw [hst]; positivity) (by rw [hk]; positivity) (by
    rw [hst, hk]; exact_mod_cast hb), hst, hk]
  push_cast
  ring

/-- The offsets of a source pointer add up to the element index. -/
theorem key_offsum {d : Inst} (hd : d.ok) {p : Nat}
    (hp : p < d.num_tokens * numTiles d) {tok : I64} (hst : tok.toInt = (tk d p : Int))
    (l : Idx [d.TILE_SIZE]) :
    (tok * (IntV.ofNat d.key_stride : I64)).toInt + ((tpos d p).get l).toInt =
      (giK d p l : Int) := by
  rw [key_idx_toInt hd hp hst, tpos_toInt hd hp l]
  simp only [giK, keyIdx]
  push_cast
  ring

/-- The token offset plus the lane position, as integers, equals the flat value index of the lane. -/
theorem value_offsum {d : Inst} (hd : d.ok) {p : Nat}
    (hp : p < d.num_tokens * numTiles d) {tok : I64} (hst : tok.toInt = (tk d p : Int))
    (l : Idx [d.TILE_SIZE]) :
    (tok * (IntV.ofNat d.value_stride : I64)).toInt + ((tpos d p).get l).toInt =
      (giV d p l : Int) := by
  rw [value_idx_toInt hd hp hst, tpos_toInt hd hp l]
  simp only [giV, valueIdx]
  push_cast
  ring

/-! ## The destination element -/

/-- The tensor the model computes the destination element's offset in. -/
abbrev tgt (d : Inst) (blk off : I64) (head64 dim64 : Tensor [d.TILE_SIZE] I64) :
    Tensor [d.TILE_SIZE] I64 :=
  blk * (IntV.ofNat d.block_stride : I64) + off * (IntV.ofNat d.page_stride : I64)
    + head64 * (IntV.ofNat d.head_stride : I64) + dim64

/-- The destination offset of an enabled lane is its cache element. -/
theorem tgt_get_toInt {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d)
    (hsl : slotAt d init (tk d p) < (numSlots d : Int))
    {blk off : I64}
    (hblk : blk.toInt = ((slotAt d init (tk d p)).toNat / d.block_size : Nat))
    (hoff : off.toInt = ((slotAt d init (tk d p)).toNat % d.block_size : Nat))
    {head64 dim64 : Tensor [d.TILE_SIZE] I64}
    (hhead : ∀ l, (head64.get l).toInt = (pos d p l / d.head_size : Nat))
    (hdim : ∀ l, (dim64.get l).toInt = (pos d p l % d.head_size : Nat))
    {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) :
    ((tgt d blk off head64 dim64).get l).toInt = (gi d init p l : Int) := by
  have hB : (IntV.ofNat d.block_stride : I64).toInt = (d.block_stride : Int) :=
    I64_toInt_ofNat hd.strides_lt.2.2.1
  have hP : (IntV.ofNat d.page_stride : I64).toInt = (d.page_stride : Int) :=
    I64_toInt_ofNat hd.strides_lt.2.2.2.1
  have hH : (IntV.ofNat d.head_stride : I64).toInt = (d.head_stride : Int) :=
    I64_toInt_ofNat hd.strides_lt.2.2.2.2
  have hkv : ((slotAt d init (tk d p)).toNat / d.block_size * d.block_stride +
      (slotAt d init (tk d p)).toNat % d.block_size * d.page_stride +
      pos d p l / d.head_size * d.head_stride + pos d p l % d.head_size) < 2 ^ 63 := by
    have h := gi_lt_kc hd hp hsl hen
    have h2 := hd.lens_lt.2.2.1
    simp only [gi, dstIdx, kvIdx] at h ⊢
    exact Nat.lt_trans h h2
  have hle1 : (slotAt d init (tk d p)).toNat / d.block_size * d.block_stride ≤
      (slotAt d init (tk d p)).toNat / d.block_size * d.block_stride +
      (slotAt d init (tk d p)).toNat % d.block_size * d.page_stride +
      pos d p l / d.head_size * d.head_stride + pos d p l % d.head_size := by omega
  have hle2 : (slotAt d init (tk d p)).toNat % d.block_size * d.page_stride ≤
      (slotAt d init (tk d p)).toNat / d.block_size * d.block_stride +
      (slotAt d init (tk d p)).toNat % d.block_size * d.page_stride +
      pos d p l / d.head_size * d.head_stride + pos d p l % d.head_size := by omega
  have hle3 : pos d p l / d.head_size * d.head_stride ≤
      (slotAt d init (tk d p)).toNat / d.block_size * d.block_stride +
      (slotAt d init (tk d p)).toNat % d.block_size * d.page_stride +
      pos d p l / d.head_size * d.head_stride + pos d p l % d.head_size := by omega
  have hP1 : (blk * (IntV.ofNat d.block_stride : I64)).toInt =
      ((slotAt d init (tk d p)).toNat / d.block_size * d.block_stride : Nat) := by
    rw [toInt_mul64 (a := blk) (b := (IntV.ofNat d.block_stride : I64))
      (by rw [hblk]; positivity) (by rw [hB]; positivity) (by
        rw [hblk, hB]; exact_mod_cast lt_of_le_of_lt hle1 hkv), hblk, hB]
    push_cast
    ring
  have hP2 : (off * (IntV.ofNat d.page_stride : I64)).toInt =
      ((slotAt d init (tk d p)).toNat % d.block_size * d.page_stride : Nat) := by
    rw [toInt_mul64 (a := off) (b := (IntV.ofNat d.page_stride : I64))
      (by rw [hoff]; positivity) (by rw [hP]; positivity) (by
        rw [hoff, hP]; exact_mod_cast lt_of_le_of_lt hle2 hkv), hoff, hP]
    push_cast
    ring
  have hP3 : (head64.get l * (IntV.ofNat d.head_stride : I64)).toInt =
      (pos d p l / d.head_size * d.head_stride : Nat) := by
    rw [toInt_mul64 (a := head64.get l) (b := (IntV.ofNat d.head_stride : I64))
      (by rw [hhead l]; positivity) (by rw [hH]; positivity) (by
        rw [hhead l, hH]; exact_mod_cast lt_of_le_of_lt hle3 hkv), hhead l, hH]
    push_cast
    ring
  have hP4 : (dim64.get l).toInt = (pos d p l % d.head_size : Nat) := hdim l
  have hsum := toInt_add4 (a := blk * (IntV.ofNat d.block_stride : I64))
    (b := off * (IntV.ofNat d.page_stride : I64))
    (c := head64.get l * (IntV.ofNat d.head_stride : I64)) (e := dim64.get l)
    (by rw [hP1]; positivity) (by rw [hP2]; positivity) (by rw [hP3]; positivity)
    (by rw [hP4]; positivity) (by
      rw [hP1, hP2, hP3, hP4]
      push_cast
      exact_mod_cast hkv)
  simp only [tgt, Tensor.get_add, Tensor.get_sadd, Tensor.get_muls]
  rw [hsum, hP1, hP2, hP3, hP4]
  simp only [gi, dstIdx, kvIdx]
  push_cast
  ring

/-! ## The pointer identities -/

theorem key_addr_get {d : Inst} (hd : d.ok) {p : Nat}
    (hp : p < d.num_tokens * numTiles d) {tok : I64} (hst : tok.toInt = (tk d p : Int))
    (l : Idx [d.TILE_SIZE]) :
    (((d.key.ptr 0) + tok * (IntV.ofNat d.key_stride : I64) + tpos d p).get l).addr =
      (d.key.ptr (giK d p l : Int)).addr := by
  have hR : d.key.ptr (giK d p l : Int) = (d.key.ptr 0).offset (giK d p l : Int) :=
    Buf.ptr_eq_offset _ _
  rw [hR, Tensor.get_ptrAdd, Ptr.add_int, Ptr.offset_offset, Ptr.offset_addr,
    key_offsum hd hp hst l]
  simp only [Ptr.offset_addr]

/-- The value input address computed by the kernel at a lane equals the address of the lane's flat value index. -/
theorem value_addr_get {d : Inst} (hd : d.ok) {p : Nat}
    (hp : p < d.num_tokens * numTiles d) {tok : I64} (hst : tok.toInt = (tk d p : Int))
    (l : Idx [d.TILE_SIZE]) :
    (((d.value.ptr 0) + tok * (IntV.ofNat d.value_stride : I64) + tpos d p).get l).addr =
      (d.value.ptr (giV d p l : Int)).addr := by
  have hR : d.value.ptr (giV d p l : Int) = (d.value.ptr 0).offset (giV d p l : Int) :=
    Buf.ptr_eq_offset _ _
  rw [hR, Tensor.get_ptrAdd, Ptr.add_int, Ptr.offset_offset, Ptr.offset_addr,
    value_offsum hd hp hst l]
  simp only [Ptr.offset_addr]

/-- For an enabled lane, the key cache address computed by the kernel equals the address of the lane's destination index. -/
theorem kc_addr_get {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d)
    (hsl : slotAt d init (tk d p) < (numSlots d : Int))
    {blk off : I64}
    (hblk : blk.toInt = ((slotAt d init (tk d p)).toNat / d.block_size : Nat))
    (hoff : off.toInt = ((slotAt d init (tk d p)).toNat % d.block_size : Nat))
    {head64 dim64 : Tensor [d.TILE_SIZE] I64}
    (hhead : ∀ l, (head64.get l).toInt = (pos d p l / d.head_size : Nat))
    (hdim : ∀ l, (dim64.get l).toInt = (pos d p l % d.head_size : Nat))
    {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) :
    (((d.key_cache.ptr 0) + tgt d blk off head64 dim64).get l).addr =
      (d.key_cache.ptr (gi d init p l : Int)).addr := by
  have hR : d.key_cache.ptr (gi d init p l : Int) =
      (d.key_cache.ptr 0).offset (gi d init p l : Int) :=
    Buf.ptr_eq_offset _ _
  rw [hR, Tensor.get_ptrAdd, Ptr.offset_addr,
    tgt_get_toInt hd hp hsl hblk hoff hhead hdim hen]
  simp only [Ptr.offset_addr]

/-- For an enabled lane, the value cache address computed by the kernel equals the address of the lane's destination index. -/
theorem vc_addr_get {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d)
    (hsl : slotAt d init (tk d p) < (numSlots d : Int))
    {blk off : I64}
    (hblk : blk.toInt = ((slotAt d init (tk d p)).toNat / d.block_size : Nat))
    (hoff : off.toInt = ((slotAt d init (tk d p)).toNat % d.block_size : Nat))
    {head64 dim64 : Tensor [d.TILE_SIZE] I64}
    (hhead : ∀ l, (head64.get l).toInt = (pos d p l / d.head_size : Nat))
    (hdim : ∀ l, (dim64.get l).toInt = (pos d p l % d.head_size : Nat))
    {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) :
    (((d.value_cache.ptr 0) + tgt d blk off head64 dim64).get l).addr =
      (d.value_cache.ptr (gi d init p l : Int)).addr := by
  have hR : d.value_cache.ptr (gi d init p l : Int) =
      (d.value_cache.ptr 0).offset (gi d init p l : Int) :=
    Buf.ptr_eq_offset _ _
  rw [hR, Tensor.get_ptrAdd, Ptr.offset_addr,
    tgt_get_toInt hd hp hsl hblk hoff hhead hdim hen]
  simp only [Ptr.offset_addr]

/-! ## The `Buf.Addrs` facts -/

theorem key_addrs {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d) (hsl0 : 0 ≤ slotAt d init (tk d p))
    {tok : I64} (hst : tok.toInt = (tk d p : Int)) :
    d.key.Addrs (tl.addrs ((d.key.ptr 0) + tok * (IntV.ofNat d.key_stride : I64) + tpos d p)
      (some (maskT d p))) (giK d p) (en d init p) := by
  intro i
  simp only [tl.addrs, en_eq_mask hd hp hsl0 i]
  split_ifs with h
  · exact congrArg some (key_addr_get hd hp hst i)
  · rfl

/-- The kernel's value input addresses match the value buffer's indices on enabled lanes. -/
theorem value_addrs {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d) (hsl0 : 0 ≤ slotAt d init (tk d p))
    {tok : I64} (hst : tok.toInt = (tk d p : Int)) :
    d.value.Addrs (tl.addrs ((d.value.ptr 0) + tok * (IntV.ofNat d.value_stride : I64)
      + tpos d p) (some (maskT d p))) (giV d p) (en d init p) := by
  intro i
  simp only [tl.addrs, en_eq_mask hd hp hsl0 i]
  split_ifs with h
  · exact congrArg some (value_addr_get hd hp hst i)
  · rfl

/-- The kernel's key cache addresses match the key cache's destination indices on enabled lanes. -/
theorem kc_addrs {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d) (hsl0 : 0 ≤ slotAt d init (tk d p))
    (hsl : slotAt d init (tk d p) < (numSlots d : Int))
    {blk off : I64}
    (hblk : blk.toInt = ((slotAt d init (tk d p)).toNat / d.block_size : Nat))
    (hoff : off.toInt = ((slotAt d init (tk d p)).toNat % d.block_size : Nat))
    {head64 dim64 : Tensor [d.TILE_SIZE] I64}
    (hhead : ∀ l, (head64.get l).toInt = (pos d p l / d.head_size : Nat))
    (hdim : ∀ l, (dim64.get l).toInt = (pos d p l % d.head_size : Nat)) :
    d.key_cache.Addrs (tl.addrs ((d.key_cache.ptr 0) + tgt d blk off head64 dim64)
      (some (maskT d p))) (gi d init p) (en d init p) := by
  intro i
  simp only [tl.addrs, en_eq_mask hd hp hsl0 i]
  split_ifs with h
  · exact congrArg some (kc_addr_get hd hp hsl hblk hoff hhead hdim (by
      rw [en_eq_mask hd hp hsl0 i]; exact h))
  · rfl

/-- The kernel's value cache addresses match the value cache's destination indices on enabled lanes. -/
theorem vc_addrs {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d) (hsl0 : 0 ≤ slotAt d init (tk d p))
    (hsl : slotAt d init (tk d p) < (numSlots d : Int))
    {blk off : I64}
    (hblk : blk.toInt = ((slotAt d init (tk d p)).toNat / d.block_size : Nat))
    (hoff : off.toInt = ((slotAt d init (tk d p)).toNat % d.block_size : Nat))
    {head64 dim64 : Tensor [d.TILE_SIZE] I64}
    (hhead : ∀ l, (head64.get l).toInt = (pos d p l / d.head_size : Nat))
    (hdim : ∀ l, (dim64.get l).toInt = (pos d p l % d.head_size : Nat)) :
    d.value_cache.Addrs (tl.addrs ((d.value_cache.ptr 0) + tgt d blk off head64 dim64)
      (some (maskT d p))) (gi d init p) (en d init p) := by
  intro i
  simp only [tl.addrs, en_eq_mask hd hp hsl0 i]
  split_ifs with h
  · exact congrArg some (vc_addr_get hd hp hsl hblk hoff hhead hdim (by
      rw [en_eq_mask hd hp hsl0 i]; exact h))
  · rfl

end Claims.KvCacheWrite
