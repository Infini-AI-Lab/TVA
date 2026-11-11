import Claim
import TVA.Proof
import Proof.Basic

/-!
# `gdn_gating_bf16`: the lanes of one program

Program `i` of the grid `(batch, 1, cdiv(H, 8))` handles token `i % batch` and head block
`i / batch`.  Lane `l : Fin 8` of it is head `hidx = 8 * (i / batch) + l`, enabled when
`hidx < H` (the `mask`).  This file gives the index maps of the six accesses, the bounds that
keep the `int32` index arithmetic from wrapping, the pointer equalities and the `Buf.Addrs`
facts the access rules need.
-/

namespace Claims.GdnGating

open TVA

/-- `int32` `a * b` has value `a.toInt * b.toInt` when that product does not wrap. -/
theorem mul_toInt_of_fits {a b : I32} (h : IntV.SFits 32 (a.toInt * b.toInt)) :
    (a * b).toInt = a.toInt * b.toInt := IntV.toInt_mul rfl rfl h

/-- `int32` `a + b` has value `a.toInt + b.toInt` when that sum does not wrap. -/
theorem add_toInt_of_fits {a b : I32} (h : IntV.SFits 32 (a.toInt + b.toInt)) :
    (a + b).toInt = a.toInt + b.toInt := IntV.toInt_add rfl rfl h

/-! ## Lanes -/

/-- The token (row of `a`, `b`, `g`) of program `i`. -/
abbrev tok (d : Inst) (i : Nat) : Nat := i % d.batch

/-- The head block of program `i`. -/
abbrev blk (d : Inst) (i : Nat) : Nat := i / d.batch

/-- The head of lane `l` of program `i`. -/
abbrev hidx (d : Inst) (i : Nat) (l : Nat) : Nat := blk d i * 8 + l

/-- Lane `l` of program `i` is enabled: its head is below `num_heads`. -/
abbrev en (d : Inst) (i : Nat) (l : Nat) : Bool := decide (hidx d i l < d.H)

/-- The element of `A_log`/`dt_bias` that lane `p` reads. -/
abbrev gH (d : Inst) (i : Nat) : Idx [8] → Nat := fun p => hidx d i p.1.val

/-- The element of `a`/`b` that lane `p` of program `i` reads (row stride `stride`). -/
abbrev gR (d : Inst) (i : Nat) (stride : Nat) : Idx [8] → Nat :=
  fun p => tok d i * stride + hidx d i p.1.val

/-- The element of `g`/`beta_output` that lane `p` of program `i` writes. -/
abbrev gO (d : Inst) (i : Nat) : Idx [8] → Nat := fun p => tok d i * d.H + hidx d i p.1.val

/-- The mask of the block, as a predicate on `Idx [8]`. -/
abbrev en8 (d : Inst) (i : Nat) : Idx [8] → Bool := fun p => en d i p.1.val

/-- The index of lane `l` of a one-dimensional block of 8. -/
abbrev lane (l : Fin 8) : Idx [8] := (l, ())

/-! ## The index tensors of the body -/

/-- `i_b = tl.program_id ctx 0`. -/
abbrev iB (d : Inst) (i : Fin (launch d).size) : I32 := tl.program_id ((launch d).ctx i) 0

/-- `i_s = tl.program_id ctx 1`. -/
abbrev iS (d : Inst) (i : Fin (launch d).size) : I32 := tl.program_id ((launch d).ctx i) 1

/-- `i_d = tl.program_id ctx 2`. -/
abbrev iD (d : Inst) (i : Fin (launch d).size) : I32 := tl.program_id ((launch d).ctx i) 2

/-- `head_off = 8 * i_d + tl.arange 0 8`. -/
abbrev headOff (d : Inst) (i : Fin (launch d).size) : Tensor [8] I32 :=
  iD d i * (IntV.ofNat 8 : I32) + tl.arange 0 8

/-- `i_b * seq_len * num_heads` (`seq_len = 1`). -/
abbrev rowOff (d : Inst) (i : Fin (launch d).size) : I32 :=
  iB d i * (IntV.ofNat 1 : I32) * (IntV.ofNat d.H : I32)

/-- `i_s * num_heads` (`i_s = 0` on the grid). -/
abbrev tokOff (d : Inst) (i : Fin (launch d).size) : I32 := iS d i * (IntV.ofNat d.H : I32)

/-- The scalar part of `off`. -/
abbrev offS (d : Inst) (i : Fin (launch d).size) : I32 := rowOff d i + tokOff d i

/-- `off = i_b * seq_len * num_heads + i_s * num_heads + head_off`. -/
abbrev offD (d : Inst) (i : Fin (launch d).size) : Tensor [8] I32 := offS d i + headOff d i

/-- `mask = head_off < num_heads` (signed). -/
abbrev maskD (d : Inst) (i : Fin (launch d).size) : Tensor [8] Bool :=
  (headOff d i).ltS (IntV.ofNat d.H : I32)

/-! ## Small facts -/

/-- An enabled lane's head is a head. -/
theorem hidx_of_en {d : Inst} {i : Nat} {l : Nat} (h : en d i l = true) : hidx d i l < d.H :=
  of_decide_eq_true h

/-- A natural below `2^31` is a signed 32-bit integer. -/
theorem sfits_nat {n : Nat} (h : n < 2 ^ 31) : IntV.SFits 32 (n : Int) := by
  have h2 : ((2 : Int) ^ (32 - 1)) = 2147483648 := by norm_num
  have h0 : (0 : Int) ≤ (n : Int) := by positivity
  have h1 : (n : Int) < 2147483648 := by
    have hp : ((2 : Nat) ^ 31) = 2147483648 := by norm_num
    omega
  simp only [IntV.SFits, h2]
  omega

/-- The program id `i_b`. -/
theorem iB_eq (d : Inst) (i : Fin (launch d).size) : iB d i = IntV.ofNat (tok d i.val) := by
  rw [iB, tl.program_id, launch_pid0]

/-- The program id `i_s` (the grid's second axis is `1`). -/
theorem iS_eq (d : Inst) (i : Fin (launch d).size) : iS d i = IntV.ofNat 0 := by
  rw [iS, tl.program_id, launch_pid1]

/-- The program id `i_d`. -/
theorem iD_eq (d : Inst) (i : Fin (launch d).size) : iD d i = IntV.ofNat (blk d i.val) := by
  rw [iD, tl.program_id, launch_pid2]

/-- The grid's first axis is positive. -/
theorem batch_pos {d : Inst} (i : Fin (launch d).size) : 0 < d.batch := by
  rcases Nat.eq_zero_or_pos d.batch with h | h
  · exfalso
    have hi := lt_of_lt_of_eq i.isLt (launch_size d)
    rw [h, Nat.zero_mul] at hi
    exact Nat.not_lt_zero _ hi
  · exact h

/-- The token of a program is a token of the batch. -/
theorem tok_lt {d : Inst} (i : Fin (launch d).size) : tok d i.val < d.batch :=
  Nat.mod_lt _ (batch_pos i)

/-- `i_b` is below `2^31`. -/
theorem tok_lt_int {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : tok d i.val < 2 ^ 31 :=
  lt_trans (tok_lt i) hd.batch_lt

/-- The head block of a program is a block of the grid. -/
theorem blk_lt {d : Inst} (i : Fin (launch d).size) : blk d i.val < Bn d := by
  have hb := batch_pos i
  have hi := lt_of_lt_of_eq i.isLt (launch_size d)
  rw [blk, Nat.div_lt_iff_lt_mul hb]
  rwa [Nat.mul_comm]

/-- The head block is below `2^16` (the grid's third axis). -/
theorem blk_lt_big {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : blk d i.val < 2 ^ 16 :=
  lt_trans (blk_lt i) hd.grid_z

/-- `num_heads` is below `2^31` (there are at most `2^16` blocks of 8). -/
theorem H_lt {d : Inst} (hd : d.ok) : d.H < 2 ^ 31 := by
  have h := Nat.div_add_mod (d.H + 7) 8
  have h2 := Nat.mod_lt (d.H + 7) (by norm_num : 0 < 8)
  have hz := hd.grid_z
  unfold Bn at hz
  omega

/-- Every head of the block, enabled or not, is within the `int32` range. -/
theorem hidx_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8) :
    hidx d i.val l.val < 2 ^ 31 := by
  have hb := blk_lt_big hd i
  have hl := l.isLt
  simp only [hidx]
  omega

/-- The `int32` offset `i_b * num_heads + h` of a written element. -/
theorem gO_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) :
    tok d i.val * d.H + hidx d i.val l.val < 2 ^ 31 := by
  have ht : tok d i.val + 1 ≤ d.batch := by have := tok_lt i; omega
  have hflat := hd.flat_le
  rcases Nat.eq_zero_or_pos d.H with hH | hH
  · rw [hH] at h
    exact absurd h (Nat.not_lt_zero _)
  · have heq : (tok d i.val + 1) * d.H = tok d i.val * d.H + d.H := by ring
    have h3 : (tok d i.val + 1) * d.H ≤ d.batch * d.H := Nat.mul_le_mul_right _ ht
    omega

/-- The `int32` offset `i_b * stride + h` of a read element of `a`/`b`. -/
theorem gR_lt {d : Inst} (i : Fin (launch d).size) (stride : Nat)
    (hstr : (d.batch - 1) * stride + d.H ≤ 2 ^ 31) (l : Fin 8) (h : hidx d i.val l.val < d.H) :
    tok d i.val * stride + hidx d i.val l.val < 2 ^ 31 := by
  have ht : tok d i.val ≤ d.batch - 1 := by have := tok_lt i; omega
  have h1 : tok d i.val * stride ≤ (d.batch - 1) * stride := Nat.mul_le_mul_right stride ht
  omega

/-- The `int32` value of `i_b * stride` (an enabled lane puts the row in range). -/
theorem iB_stride_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (stride : Nat)
    (hs31 : stride < 2 ^ 31) (hstr : (d.batch - 1) * stride + d.H ≤ 2 ^ 31) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) :
    (iB d i * IntV.ofNat stride).toInt = ((tok d i.val * stride : Nat) : Int) := by
  have ht31 : tok d i.val < 2 ^ 31 := tok_lt_int hd i
  rw [iB_eq, mul_toInt_of_fits]
  · rw [IntV.I32.toInt_ofNat ht31, IntV.I32.toInt_ofNat hs31]
    push_cast; ring
  · rw [IntV.I32.toInt_ofNat ht31, IntV.I32.toInt_ofNat hs31]
    refine sfits_nat ?_
    have hl := gR_lt i stride hstr l h
    omega

/-! ## Values of the index tensors -/

/-- `o.toInt` for `o = a * b` built from `ofNat`s, below `2^31`. -/
theorem ofNat_mul_toInt {a b : Nat} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) (hab : a * b < 2 ^ 31) :
    ((IntV.ofNat a : I32) * IntV.ofNat b).toInt = ((a * b : Nat) : Int) := by
  rw [mul_toInt_of_fits]
  · rw [IntV.I32.toInt_ofNat ha, IntV.I32.toInt_ofNat hb]
    push_cast; ring
  · rw [IntV.I32.toInt_ofNat ha, IntV.I32.toInt_ofNat hb, ← Nat.cast_mul]
    exact sfits_nat hab

/-- `o.toInt` for `o = a + b` built from `ofNat`s, below `2^31`. -/
theorem ofNat_add_toInt {a b : Nat} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) (hab : a + b < 2 ^ 31) :
    ((IntV.ofNat a : I32) + IntV.ofNat b).toInt = ((a + b : Nat) : Int) := by
  rw [add_toInt_of_fits]
  · rw [IntV.I32.toInt_ofNat ha, IntV.I32.toInt_ofNat hb]
    push_cast; ring
  · rw [IntV.I32.toInt_ofNat ha, IntV.I32.toInt_ofNat hb, ← Nat.cast_add]
    exact sfits_nat hab

/-- **`head_off`**: the `int32` value of lane `p`. -/
theorem headOff_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (p : Idx [8]) :
    ((headOff d i).get p).toInt = ((blk d i.val * 8 + p.1.val : Nat) : Int) := by
  obtain ⟨l, u⟩ := p
  cases u
  have hb := blk_lt_big hd i
  have hl := l.isLt
  have hb31 : blk d i.val < 2 ^ 31 := by omega
  have hli : (IntV.ofInt ((0 : Nat) + (l : Nat)) : I32).toInt = (l : Int) := by
    rw [IntV.I32.toInt_ofInt] <;> push_cast
    · ring
    · constructor <;> omega
  have hmul : ((IntV.ofNat (blk d i.val) : I32) * IntV.ofNat 8).toInt =
      ((blk d i.val * 8 : Nat) : Int) :=
    ofNat_mul_toInt hb31 (by norm_num) (by omega)
  rw [headOff, iD_eq]
  show ((IntV.ofNat (blk d i.val) : I32) * IntV.ofNat 8 +
      IntV.ofInt ((0 : Nat) + (l : Nat))).toInt = ((blk d i.val * 8 + l.val : Nat) : Int)
  rw [add_toInt_of_fits, hmul, hli]
  · push_cast; ring
  · rw [hmul, hli, ← Nat.cast_add]
    exact sfits_nat (by omega)

/-- **`off`**: the `int32` value of an enabled lane. -/
theorem offD_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) :
    ((offD d i).get (lane l)).toInt =
      ((tok d i.val * d.H + blk d i.val * 8 + l.val : Nat) : Int) := by
  have hH := H_lt hd
  have hb := blk_lt_big hd i
  have hl := l.isLt
  have ht31 := tok_lt_int hd i
  have htH : tok d i.val * d.H < 2 ^ 31 := by
    have ht : tok d i.val + 1 ≤ d.batch := by have := tok_lt i; omega
    have hflat := hd.flat_le
    rcases Nat.eq_zero_or_pos d.H with hH0 | hH0
    · rw [hH0, Nat.mul_zero]; norm_num
    · have heq : (tok d i.val + 1) * d.H = tok d i.val * d.H + d.H := by ring
      have h3 : (tok d i.val + 1) * d.H ≤ d.batch * d.H := Nat.mul_le_mul_right _ ht
      omega
  have hrow : ((IntV.ofNat (tok d i.val) : I32) * IntV.ofNat 1).toInt =
      ((tok d i.val : Nat) : Int) := by
    rw [mul_toInt_of_fits]
    · rw [IntV.I32.toInt_ofNat ht31, IntV.I32.toInt_ofNat (by norm_num)]
      push_cast; ring
    · rw [IntV.I32.toInt_ofNat ht31, IntV.I32.toInt_ofNat (by norm_num), ← Nat.cast_mul]
      exact sfits_nat (by simpa using ht31)
  have hrw : (((IntV.ofNat (tok d i.val) : I32) * IntV.ofNat 1) * IntV.ofNat d.H).toInt =
      ((tok d i.val * d.H : Nat) : Int) := by
    rw [mul_toInt_of_fits]
    · rw [hrow, IntV.I32.toInt_ofNat hH]
      push_cast; ring
    · rw [hrow, IntV.I32.toInt_ofNat hH, ← Nat.cast_mul]
      exact sfits_nat htH
  have htw : ((IntV.ofNat 0 : I32) * IntV.ofNat d.H).toInt = ((0 : Nat) : Int) := by
    rw [mul_toInt_of_fits]
    · rw [IntV.I32.toInt_ofNat (by norm_num), IntV.I32.toInt_ofNat hH]
      push_cast; ring
    · rw [IntV.I32.toInt_ofNat (by norm_num), IntV.I32.toInt_ofNat hH, ← Nat.cast_mul]
      exact sfits_nat (by simp)
  have hS : (offS d i).toInt = ((tok d i.val * d.H : Nat) : Int) := by
    rw [offS, rowOff, tokOff, iB_eq, iS_eq, add_toInt_of_fits]
    · rw [hrw, htw]; push_cast; ring
    · rw [hrw, htw, ← Nat.cast_add]
      exact sfits_nat (by omega)
  have hhead := headOff_toInt hd i (lane l)
  have hsum : tok d i.val * d.H + (blk d i.val * 8 + l.val) < 2 ^ 31 := by
    have hl' : hidx d i.val l.val < d.H := h
    have := gO_lt hd i l h
    simp only [hidx] at this hl' ⊢
    omega
  rw [offD]
  show (offS d i + (headOff d i).get (lane l)).toInt =
    ((tok d i.val * d.H + blk d i.val * 8 + l.val : Nat) : Int)
  rw [add_toInt_of_fits, hS, hhead]
  · push_cast; ring
  · rw [hS, hhead, ← Nat.cast_add]
    exact sfits_nat (by omega)

/-- **`mask`**: the mask of the block is `en`. -/
theorem maskD_get {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (p : Idx [8]) :
    (maskD d i).get p = en8 d i.val p := by
  rw [maskD, Tensor.get_ltS, IntV.lt_eq, headOff_toInt hd i p,
    IntV.I32.toInt_ofNat (H_lt hd)]
  simp only [en8, en, hidx]
  rw [decide_eq_decide]
  omega

/-! ## Pointers of the accesses -/

/-- `A_log + head_off` at lane `p`. -/
theorem ptr_A_log {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (p : Idx [8]) :
    (d.A_log.ptr 0 + headOff d i).get p = d.A_log.ptr (gH d i.val p) := by
  rw [Tensor.get_ptrAdd, Buf.ptr_offset, Int.zero_add, headOff_toInt hd i p]

/-- `dt_bias + head_off` at lane `p`. -/
theorem ptr_bias {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (p : Idx [8]) :
    (d.dt_bias.ptr 0 + headOff d i).get p = d.dt_bias.ptr (gH d i.val p) := by
  rw [Tensor.get_ptrAdd, Buf.ptr_offset, Int.zero_add, headOff_toInt hd i p]

/-- `a + i_b * stride_a + head_off` at an enabled lane. -/
theorem ptr_a {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) :
    ((d.a.ptr 0 + (iB d i * IntV.ofNat d.stride_a)) + headOff d i).get (lane l) =
      d.a.ptr (gR d i.val d.stride_a (lane l)) := by
  rw [Tensor.get_ptrAdd, Ptr.add_int, Ptr.offset_offset, Buf.ptr_offset, Int.zero_add,
    iB_stride_toInt hd i d.stride_a hd.stride_a_lt hd.arow_le l h,
    headOff_toInt hd i (lane l)]
  congr 1

/-- `b + i_b * stride_b + head_off` at an enabled lane. -/
theorem ptr_b {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) :
    ((d.b.ptr 0 + (iB d i * IntV.ofNat d.stride_b)) + headOff d i).get (lane l) =
      d.b.ptr (gR d i.val d.stride_b (lane l)) := by
  rw [Tensor.get_ptrAdd, Ptr.add_int, Ptr.offset_offset, Buf.ptr_offset, Int.zero_add,
    iB_stride_toInt hd i d.stride_b hd.stride_b_lt hd.brow_le l h,
    headOff_toInt hd i (lane l)]
  congr 1

/-- `g + off` at an enabled lane. -/
theorem ptr_g {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) :
    (d.g.ptr 0 + offD d i).get (lane l) = d.g.ptr (gO d i.val (lane l)) := by
  rw [Tensor.get_ptrAdd, Buf.ptr_offset, Int.zero_add, offD_toInt hd i l h]
  congr 1
  simp only [gO, hidx]
  push_cast; ring

/-- `beta_output + off` at an enabled lane. -/
theorem ptr_bout {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) :
    (d.bout.ptr 0 + offD d i).get (lane l) = d.bout.ptr (gO d i.val (lane l)) := by
  rw [Tensor.get_ptrAdd, Buf.ptr_offset, Int.zero_add, offD_toInt hd i l h]
  congr 1
  simp only [gO, hidx]
  push_cast; ring

/-! ## Lengths of the accesses -/

/-- The head of an enabled lane is an element of `A_log`. -/
theorem gH_len_A {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) : gH d i.val (lane l) < d.A_log.len :=
  lt_of_lt_of_le h hd.A_len

/-- The head of an enabled lane is an element of `dt_bias`. -/
theorem gH_len_bias {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) : gH d i.val (lane l) < d.dt_bias.len :=
  lt_of_lt_of_le h hd.bias_len

/-- The element of `a` read by an enabled lane is an element of `a`. -/
theorem gR_len_a {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) : gR d i.val d.stride_a (lane l) < d.a.len := by
  have hrow := hd.a_len (tok d i.val) (tok_lt i)
  simp only [gR, hidx] at hrow h ⊢
  omega

/-- The element of `b` read by an enabled lane is an element of `b`. -/
theorem gR_len_b {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) : gR d i.val d.stride_b (lane l) < d.b.len := by
  have hrow := hd.b_len (tok d i.val) (tok_lt i)
  simp only [gR, hidx] at hrow h ⊢
  omega

/-- The element of `g` written by an enabled lane is an element of `g`. -/
theorem gO_len_g {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) : gO d i.val (lane l) < d.g.len := by
  have hlen := hd.g_len
  have ht : tok d i.val + 1 ≤ d.batch := by have := tok_lt i; omega
  have heq : (tok d i.val + 1) * d.H = tok d i.val * d.H + d.H := by ring
  have h3 : (tok d i.val + 1) * d.H ≤ d.batch * d.H := Nat.mul_le_mul_right _ ht
  simp only [gO, hidx] at h ⊢
  omega

/-- The element of `beta_output` written by an enabled lane is an element of it. -/
theorem gO_len_bout {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (l : Fin 8)
    (h : hidx d i.val l.val < d.H) : gO d i.val (lane l) < d.bout.len := by
  have hlen := hd.bout_len
  have ht : tok d i.val + 1 ≤ d.batch := by have := tok_lt i; omega
  have heq : (tok d i.val + 1) * d.H = tok d i.val * d.H + d.H := by ring
  have h3 : (tok d i.val + 1) * d.H ≤ d.batch * d.H := Nat.mul_le_mul_right _ ht
  simp only [gO, hidx] at h ⊢
  omega

/-! ## Addresses -/

/-- The addresses of the `A_log` load. -/
theorem addrs_A_log {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    d.A_log.Addrs (tl.addrs (d.A_log.ptr 0 + headOff d i) (some (maskD d i)))
      (gH d i.val) (en8 d i.val) :=
  Buf.addrs_of_get d.A_log (fun p => maskD_get hd i p)
    (fun p _ => ptr_A_log hd i p)

/-- The addresses of the `dt_bias` load. -/
theorem addrs_bias {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    d.dt_bias.Addrs (tl.addrs (d.dt_bias.ptr 0 + headOff d i) (some (maskD d i)))
      (gH d i.val) (en8 d i.val) :=
  Buf.addrs_of_get d.dt_bias (fun p => maskD_get hd i p)
    (fun p _ => ptr_bias hd i p)

/-- The addresses of the `a` load. -/
theorem addrs_a {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    d.a.Addrs (tl.addrs (d.a.ptr 0 + (iB d i * IntV.ofNat d.stride_a) + headOff d i)
      (some (maskD d i))) (gR d i.val d.stride_a) (en8 d i.val) :=
  Buf.addrs_of_get d.a (fun p => maskD_get hd i p)
    (fun p hp => ptr_a hd i p.1 (hidx_of_en hp))

/-- The addresses of the `b` load. -/
theorem addrs_b {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    d.b.Addrs (tl.addrs (d.b.ptr 0 + (iB d i * IntV.ofNat d.stride_b) + headOff d i)
      (some (maskD d i))) (gR d i.val d.stride_b) (en8 d i.val) :=
  Buf.addrs_of_get d.b (fun p => maskD_get hd i p)
    (fun p hp => ptr_b hd i p.1 (hidx_of_en hp))

/-- The addresses of the `g` store. -/
theorem addrs_g {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    d.g.Addrs (tl.addrs (d.g.ptr 0 + offD d i) (some (maskD d i))) (gO d i.val) (en8 d i.val) :=
  Buf.addrs_of_get d.g (fun p => maskD_get hd i p)
    (fun p hp => ptr_g hd i p.1 (hidx_of_en hp))

/-- The addresses of the `beta_output` store. -/
theorem addrs_bout {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    d.bout.Addrs (tl.addrs (d.bout.ptr 0 + offD d i) (some (maskD d i)))
      (gO d i.val) (en8 d i.val) :=
  Buf.addrs_of_get d.bout (fun p => maskD_get hd i p)
    (fun p hp => ptr_bout hd i p.1 (hidx_of_en hp))

/-- Distinct lanes of one program write distinct elements. -/
theorem gO_inj {d : Inst} (i : Fin (launch d).size) (l l' : Fin 8)
    (h : gO d i.val (lane l) = gO d i.val (lane l')) : l = l' := by
  have hv : l.val = l'.val := by
    simp only [gO, hidx] at h
    omega
  exact Fin.ext hv

end Claims.GdnGating
