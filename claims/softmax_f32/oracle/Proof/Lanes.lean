import Claim
import TVA.Proof

/-!
# One row of the softmax kernel's block

A trip of `softmax_kernel` touches row `r` of `x` (or `y`) as one block of `BLOCK_SIZE` lanes:
`(ptr + row_idx * stride) + tl.arange 0 BLOCK_SIZE`, masked by `l < n_cols` (`row_idx` is the
`int32` loop variable, and `row_idx * stride` is an `int32` product, so it must not wrap).
`lane_addr` computes the address of lane `l`, `mask_lane` its mask, `addrs_row` gives the access
as a `Buf.Addrs` relation with `g i = r * stride + i.1` and `en i = i.1 < n_cols`, and `img_row`
identifies the elements it touches as `rowIn d r` / `rowOut d r`. `rowOut_disjoint` says that two
different rows' output elements are disjoint.

These are the block lemmas of `TVA.Lemmas.Block` (`lane_toInt`, `mask_lane`, `addrs_block`,
`img_block`) for the row pattern of this kernel, which `safe_load_block` and `safe_store_block`
do not cover: the row's start pointer is loop-carried, so the access is not `b.ptr 0 + offs p B`
of a program index `p`.
-/
namespace Claims.Softmax

open TVA 

/-- The input elements of row `r`. -/
def rowIn (d : Inst) (r : Nat) : Set Nat := {j | ∃ l, l < d.n_cols ∧ j = inIdx d r l}

/-- The output elements of row `r`. -/
def rowOut (d : Inst) (r : Nat) : Set Nat := {j | ∃ l, l < d.n_cols ∧ j = outIdx d r l}

/-- The elements of an input row are input elements of the launch. -/
theorem rowIn_subset {d : Inst} {r : Nat} (hr : r < d.n_rows) : rowIn d r ⊆ inSet d := by
  rintro j ⟨l, hl, rfl⟩
  exact ⟨r, l, hr, hl, rfl⟩

/-- The elements of an output row are output elements of the launch. -/
theorem rowOut_subset {d : Inst} {r : Nat} (hr : r < d.n_rows) : rowOut d r ⊆ outSet d := by
  rintro j ⟨l, hl, rfl⟩
  exact ⟨r, l, hr, hl, rfl⟩

/-- The lane index of `i : Idx [B - 0]`. Kept as a definition so that the lane projections can be
rewritten without unfolding `Idx` at the `implicit` transparency level. -/
private def laneOf {B : Nat} (i : Idx [B - 0]) : Fin (B - 0) := i.1

private theorem laneOf_mk {B : Nat} (l : Fin (B - 0)) (u : Idx []) : laneOf (l, u) = l := rfl

private theorem arange_get_i {B : Nat} (i : Idx [B - 0]) :
    (tl.arange 0 B).get i = IntV.ofInt ((0 : Int) + ((laneOf i) : Nat)) := rfl

private theorem mask_lane_i {B n : Nat} (hB : B ≤ 2 ^ 31) (hn : n < 2 ^ 31) (i : Idx [B - 0]) :
    ((tl.arange 0 B).ltS (IntV.ofNat n)).get i = decide (((laneOf i) : Nat) < n) := by
  have hi : ((laneOf i) : Nat) < B := by
    have := (laneOf i).isLt; simp only [Nat.sub_zero] at this; exact this
  have hli : (IntV.ofInt ((0 : Int) + ((laneOf i) : Nat)) : I32).toInt = ((laneOf i) : Nat) := by
    rw [IntV.I32.toInt_ofInt] <;> push_cast
    · ring
    · constructor <;> omega
  rw [Tensor.get_ltS, arange_get_i, IntV.lt_eq, IntV.I32.toInt_ofNat hn, hli]
  simp only [decide_eq_decide]
  norm_cast

/-- Lane `l` of the mask `tl.arange 0 B < n`. -/
theorem mask_lane {B n : Nat} (hB : B ≤ 2 ^ 31) (hn : n < 2 ^ 31) (l : Fin (B - 0)) :
    ((tl.arange 0 B).ltS (IntV.ofNat n)).get (l, ()) = decide ((l : Nat) < n) :=
  mask_lane_i hB hn (l, ())

private theorem lane_addr_i {dt : DType} (b : Buf dt) {row : I32} {stride B r : Nat}
    (hprod : (row * IntV.ofNat stride : I32).toInt = (r : Int) * stride)
    (hfit : (r : Int) * stride + (B : Int) ≤ 2 ^ 31) (i : Idx [B - 0]) :
    (((b.ptr 0 + row * IntV.ofNat stride) + tl.arange 0 B).get i).addr =
      (b.ptr ((r * stride + ((laneOf i) : Nat) : Nat) : Int)).addr := by
  have hB31 : B ≤ 2 ^ 31 := by
    have h0 : (0 : Int) ≤ (r : Int) * (stride : Int) := by positivity
    have h : (B : Int) ≤ 2 ^ 31 := by linarith
    omega
  have hi : ((laneOf i) : Nat) < B := by
    have := (laneOf i).isLt; simp only [Nat.sub_zero] at this; exact this
  have hli : (IntV.ofInt ((0 : Int) + ((laneOf i) : Nat)) : I32).toInt = ((laneOf i) : Int) := by
    rw [IntV.I32.toInt_ofInt] <;> push_cast
    · ring
    · constructor <;> omega
  rw [Tensor.get_ptrAdd, arange_get_i, hli]
  simp only [Ptr.add_int, Ptr.offset_addr, Buf.ptr_addr, hprod, Int.zero_mul, Int.add_zero]
  push_cast
  ring

/-- The address of lane `l` of `(b.ptr 0 + row * stride) + tl.arange 0 B`, when the `int32`
product `row * stride` is `r * stride` (`hprod`) and the largest offset fits `int32`. -/
theorem lane_addr {dt : DType} (b : Buf dt) {row : I32} {stride B r : Nat}
    (hprod : (row * IntV.ofNat stride : I32).toInt = (r : Int) * stride)
    (hfit : (r : Int) * stride + (B : Int) ≤ 2 ^ 31) (l : Fin (B - 0)) :
    (((b.ptr 0 + row * IntV.ofNat stride) + tl.arange 0 B).get (l, ())).addr =
      (b.ptr ((r * stride + (l : Nat) : Nat) : Int)).addr :=
  lane_addr_i b hprod hfit (l, ())

private theorem addrs_row_i {dt : DType} (b : Buf dt) {row : I32} {stride B n r : Nat}
    (hprod : (row * IntV.ofNat stride : I32).toInt = (r : Int) * stride)
    (hfit : (r : Int) * stride + (B : Int) ≤ 2 ^ 31) (hn : n < 2 ^ 31) (i : Idx [B - 0]) :
    tl.addrs ((b.ptr 0 + row * IntV.ofNat stride) + tl.arange 0 B)
        (some ((tl.arange 0 B).ltS (IntV.ofNat n))) i =
      if decide (((laneOf i) : Nat) < n) then
        some (b.ptr ((r * stride + ((laneOf i) : Nat) : Nat) : Int)).addr else none := by
  have hB31 : B ≤ 2 ^ 31 := by
    have h0 : (0 : Int) ≤ (r : Int) * (stride : Int) := by positivity
    have h : (B : Int) ≤ 2 ^ 31 := by linarith
    omega
  simp only [tl.addrs]
  rw [mask_lane_i hB31 hn i, lane_addr_i b hprod hfit i]

/-- The row access as a buffer relation: lane `l` of row `r` is element `r * stride + l`, enabled
when `l < n`. -/
theorem addrs_row {dt : DType} (b : Buf dt) {row : I32} {stride B n r : Nat}
    (hprod : (row * IntV.ofNat stride : I32).toInt = (r : Int) * stride)
    (hfit : (r : Int) * stride + (B : Int) ≤ 2 ^ 31) (hn : n < 2 ^ 31) :
    b.Addrs (tl.addrs ((b.ptr 0 + row * IntV.ofNat stride) + tl.arange 0 B)
        (some ((tl.arange 0 B).ltS (IntV.ofNat n))))
      (fun i => r * stride + (i.1 : Nat)) (fun i => decide ((i.1 : Nat) < n)) := by
  intro i
  exact addrs_row_i b hprod hfit hn i

/-- `img_row` with the missing side condition `n ≤ B` (the lane count bounds the columns). -/
theorem img_row_of_le {B n stride r : Nat} (hnB : n ≤ B) :
    Buf.img (s := [B - 0]) (fun i => r * stride + (i.1 : Nat)) (fun i => decide ((i.1 : Nat) < n)) =
      {j | ∃ l, l < n ∧ j = r * stride + l} := by
  ext j; constructor
  · rintro ⟨⟨l, u⟩, h, rfl⟩
    have h' : (l : Nat) < n := by
      have hh : decide ((l : Nat) < n) = true := h
      exact of_decide_eq_true hh
    exact ⟨l, h', rfl⟩
  · rintro ⟨l, hl, rfl⟩
    refine ⟨(⟨l, by simp only [Nat.sub_zero]; omega⟩, ()), ?_, rfl⟩
    have hh : decide ((l : Nat) < n) = true := decide_eq_true hl
    exact hh

/-- The elements the row access touches. The side condition `n ≤ B` is needed (`img_row_of_le`);
in the kernel, `n = d.n_cols`, `B = d.BLOCK_SIZE` and it is `d.ok.cols_le`. -/
theorem img_row {B n stride r : Nat} (hnB : n ≤ B) :
    Buf.img (s := [B - 0]) (fun i => r * stride + (i.1 : Nat)) (fun i => decide ((i.1 : Nat) < n)) =
      {j | ∃ l, l < n ∧ j = r * stride + l} :=
  img_row_of_le hnB

/-- `lane_addr` without the (unprovable) `int32` side condition: pointer arithmetic is exact in
`Int`, so only the `int32` product `row * stride` (`hprod`) matters. -/
theorem lane_addr_i' {dt : DType} (b : Buf dt) {row : I32} {stride B r : Nat}
    (hprod : (row * IntV.ofNat stride : I32).toInt = (r : Int) * stride) (hB : B ≤ 2 ^ 31)
    (i : Idx [B - 0]) :
    (((b.ptr 0 + row * IntV.ofNat stride) + tl.arange 0 B).get i).addr =
      (b.ptr ((r * stride + ((laneOf i) : Nat) : Nat) : Int)).addr := by
  have hi : ((laneOf i) : Nat) < B := by
    have := (laneOf i).isLt; simp only [Nat.sub_zero] at this; exact this
  have hli : (IntV.ofInt ((0 : Int) + ((laneOf i) : Nat)) : I32).toInt = ((laneOf i) : Int) := by
    rw [IntV.I32.toInt_ofInt] <;> push_cast
    · ring
    · constructor <;> omega
  rw [Tensor.get_ptrAdd, arange_get_i, hli]
  simp only [Ptr.add_int, Ptr.offset_addr, Buf.ptr_addr, hprod, Int.zero_mul, Int.add_zero]
  push_cast
  ring

/-- `lane_addr` without the (unprovable) `int32` side condition. -/
theorem lane_addr' {dt : DType} (b : Buf dt) {row : I32} {stride B r : Nat}
    (hprod : (row * IntV.ofNat stride : I32).toInt = (r : Int) * stride) (hB : B ≤ 2 ^ 31)
    (l : Fin (B - 0)) :
    (((b.ptr 0 + row * IntV.ofNat stride) + tl.arange 0 B).get (l, ())).addr =
      (b.ptr ((r * stride + (l : Nat) : Nat) : Int)).addr :=
  lane_addr_i' b hprod hB (l, ())

/-- `addrs_row` without the (unprovable) `int32` side condition. -/
theorem addrs_row_i' {dt : DType} (b : Buf dt) {row : I32} {stride B n r : Nat}
    (hprod : (row * IntV.ofNat stride : I32).toInt = (r : Int) * stride) (hB : B ≤ 2 ^ 31)
    (hn : n < 2 ^ 31) (i : Idx [B - 0]) :
    tl.addrs ((b.ptr 0 + row * IntV.ofNat stride) + tl.arange 0 B)
        (some ((tl.arange 0 B).ltS (IntV.ofNat n))) i =
      if decide (((laneOf i) : Nat) < n) then
        some (b.ptr ((r * stride + ((laneOf i) : Nat) : Nat) : Int)).addr else none := by
  simp only [tl.addrs, mask_lane_i hB hn i, lane_addr_i' b hprod hB i]

/-- `addrs_row` without the (unprovable) `int32` side condition. -/
theorem addrs_row' {dt : DType} (b : Buf dt) {row : I32} {stride B n r : Nat}
    (hprod : (row * IntV.ofNat stride : I32).toInt = (r : Int) * stride) (hB : B ≤ 2 ^ 31)
    (hn : n < 2 ^ 31) :
    b.Addrs (tl.addrs ((b.ptr 0 + row * IntV.ofNat stride) + tl.arange 0 B)
        (some ((tl.arange 0 B).ltS (IntV.ofNat n))))
      (fun i => r * stride + (i.1 : Nat)) (fun i => decide ((i.1 : Nat) < n)) := by
  intro i
  exact addrs_row_i' b hprod hB hn i

/-- Two different rows' output elements share no byte (`out_rows`). -/
theorem rowOut_disjoint {d : Inst} (hd : d.ok) {r s : Nat} (hr : r < d.n_rows) (hs : s < d.n_rows)
    (hrs : r ≠ s) : Disjoint (d.y.elems (rowOut d r)) (d.y.elems (rowOut d s)) := by
  refine Buf.elems_disjoint _ ?_
  rw [Set.disjoint_left]
  rintro j ⟨l, hl, rfl⟩ ⟨l', hl', h⟩
  unfold outIdx at h
  have hrs' : r = s := by
    rcases hd.out_rows with h1 | h1
    · omega
    · have hl1 : l < d.output_row_stride := lt_of_lt_of_le hl h1
      have hl2 : l' < d.output_row_stride := lt_of_lt_of_le hl' h1
      rcases lt_trichotomy r s with hlt | heq | hgt
      · exfalso
        have hs1 : (r + 1) * d.output_row_stride ≤ s * d.output_row_stride :=
          Nat.mul_le_mul_right _ hlt
        rw [Nat.succ_mul] at hs1
        omega
      · exact heq
      · exfalso
        have hr1 : (s + 1) * d.output_row_stride ≤ r * d.output_row_stride :=
          Nat.mul_le_mul_right _ hgt
        rw [Nat.succ_mul] at hr1
        omega
  exact hrs hrs'

end Claims.Softmax
