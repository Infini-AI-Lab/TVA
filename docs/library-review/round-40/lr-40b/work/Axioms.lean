import TVA
open TVA

/-! Round 40 (lr-40b): the axioms of every declaration the batch changed or added. -/
#print axioms Tile2.colMul_toInt_on
#print axioms Tile2.mulCol_toInt_on
#print axioms Tile2.rowMul_toInt_on
#print axioms Tile2.mulRow_toInt_on
#print axioms Tile2.add_toInt_on
#print axioms Tile2.off_toInt_on
#print axioms Tile2.ptr_lane_on
#print axioms Tile2.ptr_lane_seq_on
#print axioms Tile2.ptr_lane_sum_of_on
#print axioms Tile2.ptr_lane_seq_of_on
#print axioms Tile2.ptr_lane_sum_swap_of_on
#print axioms Tile2.ptr_lane_seq_swap_of_on
#print axioms Tile2.colMul_toInt
#print axioms Tile2.mulCol_toInt
#print axioms Tile2.rowMul_toInt
#print axioms Tile2.mulRow_toInt
#print axioms Tile2.add_toInt
#print axioms Tile2.off_toInt
#print axioms Tile2.ptr_lane
#print axioms Tile2.ptr_lane_seq
#print axioms Tile2.ptr_lane_sum_of
#print axioms Tile2.ptr_lane_sum_swap_of
#print axioms Tile2.elem_lt
#print axioms Buf.tile2_sub_bytes
#print axioms Buf.tile2_disjoint_of
#print axioms addrs_tile2
#print axioms accBytes_tile2
#print axioms safe_load_tile2
#print axioms safe_store_tile2
#print axioms Tile2.start_add_toInt
#print axioms Tile2.arange64_toInt
#print axioms Tile2.start_arange64_toInt
#print axioms Block.fits_of_dvd
#print axioms Block.fits_of_pow2
#print axioms Block.fits_of_lt_pow2
#print axioms Recurrence.exp_sub_mem_Ioc

/-! Usability (lr-40b): a column-major tile, `base + y + x` with the column term first, on the last
partial tile of lr-39b's shape with both terms in `int32`, through `ptr_lane_seq_swap_of_on`, the mask
rule and `safe_store_tile2`, with an `N`-masked factor (`N = 3`, `BN = 4`) as well. -/
example {d : DType} (b : Buf d) {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {rows : Tensor [64] I32} {cols : Tensor [4] I32} {a0 a1 mv nv : I32} {vals : Tensor [64, 4] d.Val}
    (hr : ∀ i : Fin 64, (rows.get1 i).toInt = ((960 + i : Nat) : Int))
    (hc : ∀ j : Fin 4, (cols.get1 j).toInt = ((0 + j : Nat) : Int))
    (ha0 : a0.toInt = (2149000 : Nat)) (ha1 : a1.toInt = (1 : Nat))
    (hmv : mv.toInt = (1000 : Nat)) (hnv : nv.toInt = (3 : Nat))
    (hlen : 5 + (1000 - 1) * 2149000 + (3 - 1) * 1 < b.len) (hv : b.Valid m)
    (hinj : Tile2.Inj 2149000 1 1000 3)
    (hW : b.tile2 5 2149000 1 960 0 64 4 1000 3 ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m', Safe E R (W \ b.tile2 5 2149000 1 960 0 64 4 1000 3) Q m' (k ())) :
    Safe E R W Q m (tl.store (b.ptr 5 + cols.row 64 * a1 + rows.col 4 * a0) vals
      (some (((rows.col 4).ltS mv).and ((cols.row 64).ltS nv))) >>= k) :=
  safe_store_tile2 b
    (Tile2.ptr_lane_seq_swap_of_on (e := 5) b (by simp)
      (fun i j hi => Tile2.colMul_toInt_on (M := 1000) hr ha0 (by norm_num) i j hi)
      (fun i j hj => Tile2.rowMul_toInt_on (N := 3) hc ha1 (by norm_num) i j hj))
    (Tile2.maskOk_and hr hc hmv hnv) (.inr (.inr hlen)) hinj hv hW fun m' _ _ _ => hk m'
