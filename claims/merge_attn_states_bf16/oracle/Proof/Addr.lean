import Claim
import TVA.Proof
import Proof.Defs

/-!
# Lane addresses

Each load and store of the kernel addresses, lane by lane, one element of its buffer: the lse
loads and stores one element, the row loads and stores `head_size` elements of one row. These
lemmas compute the pointers the model builds, and the mask's lanes.
-/

namespace Claims.MergeAttnStates

open TVA

/-! ## Small facts -/

/-- The lane of a pointer tensor. -/
theorem ptrAdd_get1 {d : DType} {n : Nat} (p : Ptr d) (offs : Tensor [n] I32) (l : Fin n) :
    (p + offs).get1 l = p.offset (offs.get1 l).toInt := rfl

/-- A full index of a rank-1 tensor is its `get1` index. -/
theorem get_eq_get1 {n : Nat} {α : Type} (x : Tensor [n] α) (i : Idx [n]) :
    x.get i = x.get1 i.1 := rfl

/-- `<` of a lane of `x` against a scalar, in terms of `get1`. -/
theorem get1_ltS {n : Nat} (x : Tensor [n] I32) (a : I32) (i : Fin n) :
    (x.ltS a).get1 i = IntV.lt (x.get1 i) a := rfl

/-- A lane of `tl.arange 0 n` is its index. -/
theorem arange_get1_toInt (d : Inst) (hd : d.ok) (l : Fin d.padded) :
    (((tl.arange 0 d.padded).get1 l) : I32).toInt = (l.val : Int) := by
  have hpad : d.padded < 2 ^ 31 := by
    have := hd.padded_le
    have h2 : (2 : Nat) ^ 20 < 2 ^ 31 := by norm_num
    omega
  simp only [tl.arange, Tensor.get1_arange, Nat.cast_zero, Int.zero_add]
  exact IntV.I32.toInt_ofInt (by omega)

/-- The mask's lanes: `arange 0 padded < head_size`. -/
theorem head_mask_get1 (d : Inst) (hd : d.ok) (l : Fin d.padded) :
    ((tl.arange 0 d.padded).ltS (IntV.ofNat d.head_size : I32)).get1 l =
      decide (l.val < d.head_size) := by
  have hhs : d.head_size < 2 ^ 31 := by
    have h1 := hd.head_le
    have h2 := hd.padded_le
    have h3 : (2 : Nat) ^ 20 < 2 ^ 31 := by norm_num
    omega
  rw [get1_ltS, IntV.lt_eq, arange_get1_toInt d hd l]
  have h2 : (IntV.ofNat d.head_size : I32).toInt = (d.head_size : Int) :=
    IntV.I32.toInt_ofNat hhs
  rw [h2]
  simp

/-! ## Program ids -/

/-- Program `i`'s `program_id(0)` is its token, as an integer. -/
theorem pidT_toInt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    ((tl.program_id ((launch d).ctx i) 0 : I32)).toInt = (pidT d i : Int) := by
  have h : pidT d i < 2 ^ 31 := by
    have := hd.T_lt
    have := pidT_lt d i
    omega
  simp only [tl.program_id, launch_pidT]
  exact IntV.I32.toInt_ofNat h

/-- Program `i`'s `program_id(1)` is its head, as an integer. -/
theorem pidH_toInt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    ((tl.program_id ((launch d).ctx i) 1 : I32)).toInt = (pidH d i : Int) := by
  have h : pidH d i < 2 ^ 31 := by
    have := hd.H_lt
    have := pidH_lt d i
    omega
  simp only [tl.program_id, launch_pidH]
  exact IntV.I32.toInt_ofNat h

/-- `num_programs(1)` is the number of heads `H`, as an integer. -/
theorem numH_toInt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    ((tl.num_programs ((launch d).ctx i) 1 : I32)).toInt = (d.H : Int) := by
  have h : d.H < 2 ^ 31 := by
    have := hd.H_lt
    omega
  simp only [tl.num_programs, launch_grid1]
  exact IntV.I32.toInt_ofNat h

/-! ## Bounds on the row and lse offsets -/

/-- The token times `H` does not wrap in `int32`. -/
theorem pidT_H (d : Inst) (hd : d.ok) (i : Fin (launch d).size) : pidT d i * d.H < 2 ^ 31 := by
  have h1 := pidT_lt d i
  have h2 := pidH_lt d i
  have hT := T_pos i
  have hH := H_pos i
  have hg := hd.grid_lt
  have : pidT d i * d.H ≤ (d.T - 1) * d.H := Nat.mul_le_mul_right _ (by omega)
  have : (d.T - 1) * d.H < d.T * d.H := by
    apply Nat.mul_lt_mul_of_pos_right (by omega) hH
  omega

/-- The token's offset `t · H · phs` does not wrap in `int32`. -/
theorem pidT_H_phs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    pidT d i * d.H * d.phs < 2 ^ 31 := by
  have h1 := hd.in_i32.1
  have h2 := hd.in_i32.2
  have hp : pidT d i < d.T := pidT_lt d i
  have hT := T_pos i
  have hH := H_pos i
  have hle : pidT d i * d.H * d.phs ≤ (d.T - 1) * d.H * d.phs := by
    apply Nat.mul_le_mul_right
    apply Nat.mul_le_mul_right
    omega
  omega

/-- The head's offset `h · phs` does not wrap in `int32`. -/
theorem pidH_phs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    pidH d i * d.phs < 2 ^ 31 := by
  have h1 := hd.in_i32.1
  have h2 := hd.in_i32.2
  have hp : pidH d i < d.H := pidH_lt d i
  have hH := H_pos i
  have hle : pidH d i * d.phs ≤ (d.H - 1) * d.phs := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The token's offset `t · H · ohs` does not wrap in `int32`. -/
theorem pidT_H_ohs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    pidT d i * d.H * d.ohs < 2 ^ 31 := by
  have h1 := hd.out_i32.1
  have h2 := hd.out_i32.2
  have hp : pidT d i < d.T := pidT_lt d i
  have hT := T_pos i
  have hH := H_pos i
  have hle : pidT d i * d.H * d.ohs ≤ (d.T - 1) * d.H * d.ohs := by
    apply Nat.mul_le_mul_right
    apply Nat.mul_le_mul_right
    omega
  omega

/-- The head's offset `h · ohs` does not wrap in `int32`. -/
theorem pidH_ohs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    pidH d i * d.ohs < 2 ^ 31 := by
  have h1 := hd.out_i32.1
  have h2 := hd.out_i32.2
  have hp : pidH d i < d.H := pidH_lt d i
  have hH := H_pos i
  have hle : pidH d i * d.ohs ≤ (d.H - 1) * d.ohs := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The head's offset in `suffix_lse` does not wrap in `int32`. -/
theorem pidH_slhs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    pidH d i * d.slhs < 2 ^ 31 := by
  have h1 : (d.H - 1) * d.slhs < 2 ^ 31 := hd.lse_i32.2.1
  have h2 : (d.T - 1) * d.slts < 2 ^ 31 := hd.lse_i32.2.2.1
  have h3 : (d.H - 1) * d.olhs < 2 ^ 31 := hd.lse_i32.2.2.2.1
  have h4 : (d.T - 1) * d.olts < 2 ^ 31 := hd.lse_i32.2.2.2.2
  have hp : pidH d i < d.H := pidH_lt d i
  have hH := H_pos i
  have hle : pidH d i * d.slhs ≤ (d.H - 1) * d.slhs := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The token's offset in `suffix_lse` does not wrap in `int32`. -/
theorem pidT_slts (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    pidT d i * d.slts < 2 ^ 31 := by
  have h1 : (d.H - 1) * d.slhs < 2 ^ 31 := hd.lse_i32.2.1
  have h2 : (d.T - 1) * d.slts < 2 ^ 31 := hd.lse_i32.2.2.1
  have h3 : (d.H - 1) * d.olhs < 2 ^ 31 := hd.lse_i32.2.2.2.1
  have h4 : (d.T - 1) * d.olts < 2 ^ 31 := hd.lse_i32.2.2.2.2
  have hp : pidT d i < d.T := pidT_lt d i
  have hT := T_pos i
  have hle : pidT d i * d.slts ≤ (d.T - 1) * d.slts := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The head's offset in `output_lse` does not wrap in `int32`. -/
theorem pidH_olhs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    pidH d i * d.olhs < 2 ^ 31 := by
  have h1 : (d.H - 1) * d.slhs < 2 ^ 31 := hd.lse_i32.2.1
  have h2 : (d.T - 1) * d.slts < 2 ^ 31 := hd.lse_i32.2.2.1
  have h3 : (d.H - 1) * d.olhs < 2 ^ 31 := hd.lse_i32.2.2.2.1
  have h4 : (d.T - 1) * d.olts < 2 ^ 31 := hd.lse_i32.2.2.2.2
  have hp : pidH d i < d.H := pidH_lt d i
  have hH := H_pos i
  have hle : pidH d i * d.olhs ≤ (d.H - 1) * d.olhs := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The token's offset in `output_lse` does not wrap in `int32`. -/
theorem pidT_olts (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    pidT d i * d.olts < 2 ^ 31 := by
  have h1 : (d.H - 1) * d.slhs < 2 ^ 31 := hd.lse_i32.2.1
  have h2 : (d.T - 1) * d.slts < 2 ^ 31 := hd.lse_i32.2.2.1
  have h3 : (d.H - 1) * d.olhs < 2 ^ 31 := hd.lse_i32.2.2.2.1
  have h4 : (d.T - 1) * d.olts < 2 ^ 31 := hd.lse_i32.2.2.2.2
  have hp : pidT d i < d.T := pidT_lt d i
  have hT := T_pos i
  have hle : pidT d i * d.olts ≤ (d.T - 1) * d.olts := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The head's offset in `prefix_lse` does not wrap in `int32` (merged tokens). -/
theorem pidH_plhs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (hpt : pidT d i < d.ptwc) :
    pidH d i * d.plhs < 2 ^ 31 := by
  have h0 := hd.lse_i32.1
  have hptwc : 0 < d.ptwc := lt_of_le_of_lt (Nat.zero_le _) hpt
  have hp := h0 hptwc
  have hpH : pidH d i < d.H := pidH_lt d i
  have hH := H_pos i
  have hle : pidH d i * d.plhs ≤ (d.H - 1) * d.plhs := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The token's offset in `prefix_lse` does not wrap in `int32` (merged tokens). -/
theorem pidT_plts (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (hpt : pidT d i < d.ptwc) :
    pidT d i * d.plts < 2 ^ 31 := by
  have h0 := hd.lse_i32.1
  have hptwc : 0 < d.ptwc := lt_of_le_of_lt (Nat.zero_le _) hpt
  have hp := h0 hptwc
  have hle : pidT d i * d.plts ≤ (d.ptwc - 1) * d.plts := Nat.mul_le_mul_right _ (by omega)
  omega

/-! ## The lse pointers -/

/-- The kernel's pointer to `suffix_lse` is the buffer's element `slIdx`. -/
theorem suffix_lse_ptr (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    d.suffix_lse.ptr 0 + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.slhs
      + tl.program_id ((launch d).ctx i) 0 * IntV.ofNat d.slts
      = d.suffix_lse.ptr (slIdx d (pidT d i) (pidH d i)) := by
  have hsl : d.slhs < 2 ^ 31 := hd.strides_lt.2.2.2.2.1
  have hst : d.slts < 2 ^ 31 := hd.strides_lt.2.2.2.2.2.1
  have h1 : (tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.slhs).toInt
      = ((pidH d i * d.slhs : Nat) : Int) :=
    IntV.I32.mul_nat (pidH_toInt d hd i) (IntV.I32.toInt_ofNat hsl) (pidH_slhs d hd i)
  have h2 : (tl.program_id ((launch d).ctx i) 0 * IntV.ofNat d.slts).toInt
      = ((pidT d i * d.slts : Nat) : Int) :=
    IntV.I32.mul_nat (pidT_toInt d hd i) (IntV.I32.toInt_ofNat hst) (pidT_slts d hd i)
  simp only [Ptr.add_int, Buf.ptr_offset, h1, h2]
  congr 1
  simp only [slIdx]
  push_cast
  ring

/-- The kernel's pointer to `prefix_lse` is the buffer's element `plIdx`. -/
theorem prefix_lse_ptr (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (hpt : pidT d i < d.ptwc) :
    d.prefix_lse.ptr 0 + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.plhs
      + tl.program_id ((launch d).ctx i) 0 * IntV.ofNat d.plts
      = d.prefix_lse.ptr (plIdx d (pidT d i) (pidH d i)) := by
  have hpl : d.plhs < 2 ^ 31 := hd.strides_lt.2.2.1
  have hpt' : d.plts < 2 ^ 31 := hd.strides_lt.2.2.2.1
  have h1 : (tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.plhs).toInt
      = ((pidH d i * d.plhs : Nat) : Int) :=
    IntV.I32.mul_nat (pidH_toInt d hd i) (IntV.I32.toInt_ofNat hpl) (pidH_plhs d hd i hpt)
  have h2 : (tl.program_id ((launch d).ctx i) 0 * IntV.ofNat d.plts).toInt
      = ((pidT d i * d.plts : Nat) : Int) :=
    IntV.I32.mul_nat (pidT_toInt d hd i) (IntV.I32.toInt_ofNat hpt') (pidT_plts d hd i hpt)
  simp only [Ptr.add_int, Buf.ptr_offset, h1, h2]
  congr 1
  simp only [plIdx]
  push_cast
  ring

/-- The kernel's pointer to `output_lse` is the buffer's element `olIdx`. -/
theorem output_lse_ptr (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    d.output_lse.ptr 0 + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.olhs
      + tl.program_id ((launch d).ctx i) 0 * IntV.ofNat d.olts
      = d.output_lse.ptr (olIdx d (pidT d i) (pidH d i)) := by
  have hol : d.olhs < 2 ^ 31 := hd.strides_lt.2.2.2.2.2.2.1
  have hot : d.olts < 2 ^ 31 := hd.strides_lt.2.2.2.2.2.2.2
  have h1 : (tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.olhs).toInt
      = ((pidH d i * d.olhs : Nat) : Int) :=
    IntV.I32.mul_nat (pidH_toInt d hd i) (IntV.I32.toInt_ofNat hol) (pidH_olhs d hd i)
  have h2 : (tl.program_id ((launch d).ctx i) 0 * IntV.ofNat d.olts).toInt
      = ((pidT d i * d.olts : Nat) : Int) :=
    IntV.I32.mul_nat (pidT_toInt d hd i) (IntV.I32.toInt_ofNat hot) (pidT_olts d hd i)
  simp only [Ptr.add_int, Buf.ptr_offset, h1, h2]
  congr 1
  simp only [olIdx]
  push_cast
  ring

/-! ## The row pointers -/

/-- The lane of a row access, for a buffer with stride `stride`. -/
theorem row_get1_gen (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.padded)
    (b : Buf .bf16) (stride : Nat) (hstride : stride < 2 ^ 31)
    (h1 : pidT d i * d.H * stride < 2 ^ 31) (h2 : pidH d i * stride < 2 ^ 31)
    (idx : Nat) (hidx : idx = (pidT d i * d.H * stride + pidH d i * stride) + l.val) :
    (b.ptr 0 + tl.program_id ((launch d).ctx i) 0 * tl.num_programs ((launch d).ctx i) 1
        * IntV.ofNat stride
      + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat stride
      + tl.arange 0 d.padded).get1 l = b.ptr idx := by
  have hK1 : (tl.program_id ((launch d).ctx i) 0 * tl.num_programs ((launch d).ctx i) 1
      * IntV.ofNat stride).toInt = ((pidT d i * d.H * stride : Nat) : Int) :=
    IntV.I32.mul_nat
      (IntV.I32.mul_nat (pidT_toInt d hd i) (numH_toInt d hd i) (pidT_H d hd i))
      (IntV.I32.toInt_ofNat hstride) h1
  have hK2 : (tl.program_id ((launch d).ctx i) 1 * IntV.ofNat stride).toInt
      = ((pidH d i * stride : Nat) : Int) :=
    IntV.I32.mul_nat (pidH_toInt d hd i) (IntV.I32.toInt_ofNat hstride) h2
  rw [ptrAdd_get1, arange_get1_toInt d hd l]
  simp only [Ptr.add_int, Buf.ptr_offset, hK1, hK2, hidx]
  congr 1
  push_cast
  ring

/-- Lane `l` of the kernel's `suffix_output` row pointers is the buffer's element `soIdx l`. -/
theorem suffix_row_get1 (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.padded) :
    (d.suffix_output.ptr 0 + tl.program_id ((launch d).ctx i) 0
        * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.phs
      + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.phs
      + tl.arange 0 d.padded).get1 l
      = d.suffix_output.ptr (soIdx d (pidT d i) (pidH d i) l.val) :=
  row_get1_gen d hd i l d.suffix_output d.phs hd.strides_lt.1
    (pidT_H_phs d hd i) (pidH_phs d hd i) _
    (by simp only [soIdx, hd.sots_eq, hd.shs_eq, Nat.mul_assoc])

/-- Lane `l` of the kernel's `prefix_output` row pointers is the buffer's element `poIdx l`. -/
theorem prefix_row_get1 (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.padded) :
    (d.prefix_output.ptr 0 + tl.program_id ((launch d).ctx i) 0
        * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.phs
      + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.phs
      + tl.arange 0 d.padded).get1 l
      = d.prefix_output.ptr (poIdx d (pidT d i) (pidH d i) l.val) :=
  row_get1_gen d hd i l d.prefix_output d.phs hd.strides_lt.1
    (pidT_H_phs d hd i) (pidH_phs d hd i) _
    (by simp only [poIdx, hd.pots_eq, Nat.mul_assoc])

/-- Lane `l` of the kernel's `output` row pointers is the buffer's element `oIdx l`. -/
theorem output_row_get1 (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Fin d.padded) :
    (d.output.ptr 0 + tl.program_id ((launch d).ctx i) 0
        * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.ohs
      + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.ohs
      + tl.arange 0 d.padded).get1 l
      = d.output.ptr (oIdx d (pidT d i) (pidH d i) l.val) :=
  row_get1_gen d hd i l d.output d.ohs hd.strides_lt.2.1
    (pidT_H_ohs d hd i) (pidH_ohs d hd i) _
    (by simp only [oIdx, hd.ots_eq, Nat.mul_assoc])

/-! ## The row pointers, as full indices -/

/-- The `suffix_output` row pointers, lane by lane, as buffer elements. -/
theorem suffix_row_get (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (j : Idx [d.padded]) :
    (d.suffix_output.ptr 0 + tl.program_id ((launch d).ctx i) 0
        * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.phs
      + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.phs
      + tl.arange 0 d.padded : Tensor [d.padded] (Ptr .bf16)).get j
      = d.suffix_output.ptr (soIdx d (pidT d i) (pidH d i) j.1.val) := by
  rw [get_eq_get1]
  exact suffix_row_get1 d hd i j.1

/-- The `prefix_output` row pointers, lane by lane, as buffer elements. -/
theorem prefix_row_get (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (j : Idx [d.padded]) :
    (d.prefix_output.ptr 0 + tl.program_id ((launch d).ctx i) 0
        * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.phs
      + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.phs
      + tl.arange 0 d.padded : Tensor [d.padded] (Ptr .bf16)).get j
      = d.prefix_output.ptr (poIdx d (pidT d i) (pidH d i) j.1.val) := by
  rw [get_eq_get1]
  exact prefix_row_get1 d hd i j.1

/-- The `output` row pointers, lane by lane, as buffer elements. -/
theorem output_row_get (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (j : Idx [d.padded]) :
    (d.output.ptr 0 + tl.program_id ((launch d).ctx i) 0
        * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.ohs
      + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.ohs
      + tl.arange 0 d.padded : Tensor [d.padded] (Ptr .bf16)).get j
      = d.output.ptr (oIdx d (pidT d i) (pidH d i) j.1.val) := by
  rw [get_eq_get1]
  exact output_row_get1 d hd i j.1

/-- A lane of the head mask. -/
theorem head_mask_get (d : Inst) (hd : d.ok) (j : Idx [d.padded]) :
    ((tl.arange 0 d.padded).ltS (IntV.ofNat d.head_size : I32) : Tensor [d.padded] Bool).get j
      = decide (j.1.val < d.head_size) := by
  rw [get_eq_get1]
  exact head_mask_get1 d hd j.1

/-! ## The lanes of a row access -/

/-- The element a lane of a row access addresses. -/
def rowG (d : Inst) (base : Nat) : Idx [d.padded] → Nat := fun j => base + j.1.val

/-- The enabled lanes of a row access: those below `head_size`. -/
def rowEn (d : Inst) : Idx [d.padded] → Bool := fun j => decide (j.1.val < d.head_size)

/-- The lanes enabled by the head mask cover exactly the row. -/
theorem img_rowG (d : Inst) (hd : d.ok) (base : Nat) :
    Buf.img (rowG d base) (rowEn d) = row d base := by
  ext j
  constructor
  · rintro ⟨i, hi, hg⟩
    have hi' : i.1.val < d.head_size := by simpa [rowEn] using hi
    have hg' : base + i.1.val = j := hg
    exact row_iff.mpr ⟨by omega, by omega⟩
  · intro hj
    rw [row_iff] at hj
    obtain ⟨h1, h2⟩ := hj
    have hlt : j - base < d.padded := by
      have := hd.head_le
      omega
    refine ⟨(⟨j - base, hlt⟩, ()), ?_, ?_⟩
    · simp [rowEn]
      omega
    · simp [rowG]
      omega

/-- The `suffix_output` row load addresses are the buffer's elements, under the head mask. -/
theorem suffix_row_addrs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    d.suffix_output.Addrs
      (tl.addrs (d.suffix_output.ptr 0 + tl.program_id ((launch d).ctx i) 0
          * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.phs
        + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.phs
        + tl.arange 0 d.padded : Tensor [d.padded] (Ptr .bf16))
        (some ((tl.arange 0 d.padded).ltS (IntV.ofNat d.head_size : I32) :
          Tensor [d.padded] Bool)))
      (rowG d (soIdx d (pidT d i) (pidH d i) 0)) (rowEn d) := by
  apply Buf.addrs_of_get
  · intro j
    exact head_mask_get d hd j
  · intro j _
    exact suffix_row_get d hd i j

/-- The `prefix_output` row load addresses are the buffer's elements, under the head mask. -/
theorem prefix_row_addrs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    d.prefix_output.Addrs
      (tl.addrs (d.prefix_output.ptr 0 + tl.program_id ((launch d).ctx i) 0
          * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.phs
        + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.phs
        + tl.arange 0 d.padded : Tensor [d.padded] (Ptr .bf16))
        (some ((tl.arange 0 d.padded).ltS (IntV.ofNat d.head_size : I32) :
          Tensor [d.padded] Bool)))
      (rowG d (poIdx d (pidT d i) (pidH d i) 0)) (rowEn d) := by
  apply Buf.addrs_of_get
  · intro j
    exact head_mask_get d hd j
  · intro j _
    exact prefix_row_get d hd i j

/-- The `output` row store addresses are the buffer's elements, under the head mask. -/
theorem output_row_addrs (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    d.output.Addrs
      (tl.addrs (d.output.ptr 0 + tl.program_id ((launch d).ctx i) 0
          * tl.num_programs ((launch d).ctx i) 1 * IntV.ofNat d.ohs
        + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.ohs
        + tl.arange 0 d.padded : Tensor [d.padded] (Ptr .bf16))
        (some ((tl.arange 0 d.padded).ltS (IntV.ofNat d.head_size : I32) :
          Tensor [d.padded] Bool)))
      (rowG d (oIdx d (pidT d i) (pidH d i) 0)) (rowEn d) := by
  apply Buf.addrs_of_get
  · intro j
    exact head_mask_get d hd j
  · intro j _
    exact output_row_get d hd i j

end Claims.MergeAttnStates
