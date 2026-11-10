import TVA

/-! Round 41 probes (library author): L184 (tiles of rows and columns of any form, tutorial 03's
`% M` rows) and L185 (the integer cast and `tl.convert`). Run with `lake env lean` from `lean/`. -/

namespace TVA
open Tile2

section L184
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- Tutorial 03's `a` tile on the first trip of the `K` loop:
`offs_am = (pid_m * BM + tl.arange(0, BM)) % M` (rows `(r0 + i) % M`, no row mask),
`offs_k = tl.arange(0, BK)`, `a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)`,
`tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)`. From the `int32` lanes of
`pid_m * BM + tl.arange(0, BM)` (`hx`), the broadcast `M` (`hy`), the matrix bound only. -/
example {BM BK M K r0 sa sk : Nat} (b : Buf .f16) {base : Ptr .f16} (hb : base = b.ptr (0 : Int))
    {x y : Tensor [BM] I32} {cols : Tensor [BK] I32} {a0 a1 kv : I32}
    (hx : ∀ i : Fin BM, (x.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hy : ∀ i : Fin BM, (y.get1 i).toInt = (M : Int)) (hM : 0 < M)
    (hc : ∀ j : Fin BK, (cols.get1 j).toInt = ((j : Nat) : Int))
    (ha0 : a0.toInt = sa) (ha1 : a1.toInt = sk) (hkv : kv.toInt = K)
    (hfit : (M - 1) * sa + (K - 1) * sk < 2 ^ (32 - 1))
    (hlen : (M - 1) * sa + (K - 1) * sk < b.len) (hv : b.Valid m)
    (hR : b.tile2 0 sa sk 0 0 M K M K ⊆ R)
    {k : Tensor [BM, BK] (DType.f16).Val → Prog α}
    (hk : ∀ v : Tensor [BM, BK] (DType.f16).Val,
      (∀ (i : Fin BM) (j : Fin BK), (r0 + i) % M < M → (j : Nat) < K →
        m.read .f16 (b.ptr ((0 + (r0 + i) % M * sa + (j : Nat) * sk : Nat) : Int)).addr =
          some (v.get2 i j)) → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.irem x y >>= fun rows =>
      tl.load (base + ((rows.col BK) * a0 + (cols.row BM) * a1))
        (some ((cols.row BM).ltS kv)) none >>= k) := by
  refine safe_irem_lanes hx hy hM fun rows hr => ?_
  have hp := ptr_lane_on (e := 0) b hb hr hc ha0 ha1 hfit
  have hm : MaskAt (some ((cols.row BM).ltS kv)) (fun i => (r0 + i) % M) (fun j => (j : Nat)) M K :=
    maskAt_cols hc hkv fun i => Nat.mod_lt _ hM
  exact safe_load_tileAt b hp hm (.inr (.inr (by simpa using hlen))) hv
    ((Buf.tileAt_sub_matrix b).trans hR) fun v h1 _ => hk v h1

/-- The `r0 + i` instance still goes through the general rules: a `MaskOk` proof is a `MaskAt`
proof, the old lane rules feed `hp`, and `add_inj` gives distinct rows and columns for the store. -/
example {BM BN M N r0 c0 s0 : Nat} (b : Buf .f32) {base : Ptr .f32} (hb : base = b.ptr (0 : Int))
    {rows : Tensor [BM] I32} {cols : Tensor [BN] I32} {a0 a1 mv nv : I32} {vals : Tensor [BM, BN] F32}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = 1) (hmv : mv.toInt = M) (hnv : nv.toInt = N)
    (hN : N ≤ s0) (hfit : (M - 1) * s0 + (N - 1) * 1 < 2 ^ (32 - 1))
    (hlen : (M - 1) * s0 + (N - 1) * 1 < b.len) (hv : b.Valid m)
    (hW : b.tile2 0 s0 1 r0 c0 BM BN M N ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m', Safe E R (W \ b.tileAt 0 s0 1 (fun i : Fin BM => r0 + i) (fun j : Fin BN => c0 + j) M N) Q m' (k ())) :
    Safe E R W Q m (tl.store (base + ((rows.col BN) * a0 + (cols.row BM) * a1)) vals
      (some (((rows.col BN).ltS mv).and ((cols.row BM).ltS nv))) >>= k) := by
  have hm : MaskOk _ r0 c0 M N := maskOk_and hr hc hmv hnv
  exact safe_store_tileAt b (ptr_lane_on (e := 0) b hb hr hc ha0 ha1 hfit) hm
    (.inr (.inr (by simpa using hlen))) (inj_rowMajor1 hN) add_inj add_inj hv
    (by rw [← Buf.tile2_eq_tileAt]; exact hW) fun m' _ _ _ => hk m'

/-- With `BM > M`, the `% M` rows repeat (`BM = 4`, `M = 3`: rows `0, 1, 2, 0`), so the store's
`hρ` is false for them, as it must be: two lanes would write one element. -/
example : ¬ ∀ i i' : Fin 4, (0 + (i : Nat)) % 3 < 3 → (0 + (i' : Nat)) % 3 < 3 →
    (0 + (i : Nat)) % 3 = (0 + (i' : Nat)) % 3 → i = i' := by
  intro h
  exact absurd (h 0 3 (by decide) (by decide) (by decide)) (by decide)

end L184

section L185
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- `x.to(tl.int64)` as `tl.convert` and as `castInt`: the same program, either way. -/
example {n : Nat} (x : Tensor [n] I32) (k : Tensor [n] I64 → Prog α) :
    Safe E R W Q m (tl.convert .i32 .i64 x >>= k) ↔ Safe E R W Q m (k x.castInt) :=
  safe_convert_iff fun _ _ => ConvRel.iff_cast_i32_i64

/-- A narrowing and an unsigned pair. -/
example {P : Platform} (x : I64) (r : I32) : ConvRel P .i64 .i32 x r ↔ r = IntV.cast x :=
  ConvRel.iff_cast_i64_i32
example {P : Platform} (x : U32) (r : I64) : ConvRel P .u32 .i64 x r ↔ r = IntV.cast x :=
  ConvRel.iff_cast_u32_i64

/-- The cast's value: `-1 : int32` cast to `uint32` is `2^32 - 1`, to `int64` is `-1`. -/
example : ((IntV.cast (IntV.ofInt (-1) : I32) : U32)).toInt = 2 ^ 32 - 1 := by decide
example : ((IntV.cast (IntV.ofInt (-1) : I32) : I64)).toInt = -1 := by decide

end L185

end TVA

#print axioms TVA.Tile2.colMul_toInt_on
#print axioms TVA.Tile2.mulCol_toInt_on
#print axioms TVA.Tile2.rowMul_toInt_on
#print axioms TVA.Tile2.mulRow_toInt_on
#print axioms TVA.Tile2.add_toInt_on
#print axioms TVA.Tile2.off_toInt_on
#print axioms TVA.Tile2.ptr_lane_sum_of_on
#print axioms TVA.Tile2.ptr_lane_sum_swap_of_on
#print axioms TVA.Tile2.ptr_lane_seq_of_on
#print axioms TVA.Tile2.ptr_lane_seq_swap_of_on
#print axioms TVA.Tile2.ptr_lane_on
#print axioms TVA.Tile2.ptr_lane_seq_on
#print axioms TVA.Tile2.irem_ok
#print axioms TVA.Tile2.irem_toInt
#print axioms TVA.Tile2.safe_irem_lanes
#print axioms TVA.Tile2.maskOk_iff_maskAt
#print axioms TVA.Tile2.maskAt_of_get
#print axioms TVA.Tile2.maskAt_none
#print axioms TVA.Tile2.maskAt_and
#print axioms TVA.Tile2.maskAt_rows
#print axioms TVA.Tile2.maskAt_cols
#print axioms TVA.Tile2.elem_iff_elemAt
#print axioms TVA.Tile2.elemAt_matrix
#print axioms TVA.Tile2.elemAt_lt
#print axioms TVA.Buf.tile2_eq_tileAt
#print axioms TVA.Buf.tileAt_sub_matrix
#print axioms TVA.Buf.tileAt_sub_bytes
#print axioms TVA.Buf.span_sub_tileAt
#print axioms TVA.addrs_tileAt
#print axioms TVA.tileAt_lane_elem
#print axioms TVA.img_tileAt
#print axioms TVA.accBytes_tileAt
#print axioms TVA.tileAt_inj
#print axioms TVA.add_inj
#print axioms TVA.safe_load_tileAt
#print axioms TVA.safe_store_tileAt
#print axioms TVA.IntV.cast_eq_ofInt
#print axioms TVA.safe_convert_iff
#print axioms TVA.ConvRel.iff_cast_i8_i8
#print axioms TVA.ConvRel.iff_cast_i8_i16
#print axioms TVA.ConvRel.iff_cast_i8_i32
#print axioms TVA.ConvRel.iff_cast_i8_i64
#print axioms TVA.ConvRel.iff_cast_i8_u8
#print axioms TVA.ConvRel.iff_cast_i8_u16
#print axioms TVA.ConvRel.iff_cast_i8_u32
#print axioms TVA.ConvRel.iff_cast_i8_u64
#print axioms TVA.ConvRel.iff_cast_i16_i8
#print axioms TVA.ConvRel.iff_cast_i16_i16
#print axioms TVA.ConvRel.iff_cast_i16_i32
#print axioms TVA.ConvRel.iff_cast_i16_i64
#print axioms TVA.ConvRel.iff_cast_i16_u8
#print axioms TVA.ConvRel.iff_cast_i16_u16
#print axioms TVA.ConvRel.iff_cast_i16_u32
#print axioms TVA.ConvRel.iff_cast_i16_u64
#print axioms TVA.ConvRel.iff_cast_i32_i8
#print axioms TVA.ConvRel.iff_cast_i32_i16
#print axioms TVA.ConvRel.iff_cast_i32_i32
#print axioms TVA.ConvRel.iff_cast_i32_i64
#print axioms TVA.ConvRel.iff_cast_i32_u8
#print axioms TVA.ConvRel.iff_cast_i32_u16
#print axioms TVA.ConvRel.iff_cast_i32_u32
#print axioms TVA.ConvRel.iff_cast_i32_u64
#print axioms TVA.ConvRel.iff_cast_i64_i8
#print axioms TVA.ConvRel.iff_cast_i64_i16
#print axioms TVA.ConvRel.iff_cast_i64_i32
#print axioms TVA.ConvRel.iff_cast_i64_i64
#print axioms TVA.ConvRel.iff_cast_i64_u8
#print axioms TVA.ConvRel.iff_cast_i64_u16
#print axioms TVA.ConvRel.iff_cast_i64_u32
#print axioms TVA.ConvRel.iff_cast_i64_u64
#print axioms TVA.ConvRel.iff_cast_u8_i8
#print axioms TVA.ConvRel.iff_cast_u8_i16
#print axioms TVA.ConvRel.iff_cast_u8_i32
#print axioms TVA.ConvRel.iff_cast_u8_i64
#print axioms TVA.ConvRel.iff_cast_u8_u8
#print axioms TVA.ConvRel.iff_cast_u8_u16
#print axioms TVA.ConvRel.iff_cast_u8_u32
#print axioms TVA.ConvRel.iff_cast_u8_u64
#print axioms TVA.ConvRel.iff_cast_u16_i8
#print axioms TVA.ConvRel.iff_cast_u16_i16
#print axioms TVA.ConvRel.iff_cast_u16_i32
#print axioms TVA.ConvRel.iff_cast_u16_i64
#print axioms TVA.ConvRel.iff_cast_u16_u8
#print axioms TVA.ConvRel.iff_cast_u16_u16
#print axioms TVA.ConvRel.iff_cast_u16_u32
#print axioms TVA.ConvRel.iff_cast_u16_u64
#print axioms TVA.ConvRel.iff_cast_u32_i8
#print axioms TVA.ConvRel.iff_cast_u32_i16
#print axioms TVA.ConvRel.iff_cast_u32_i32
#print axioms TVA.ConvRel.iff_cast_u32_i64
#print axioms TVA.ConvRel.iff_cast_u32_u8
#print axioms TVA.ConvRel.iff_cast_u32_u16
#print axioms TVA.ConvRel.iff_cast_u32_u32
#print axioms TVA.ConvRel.iff_cast_u32_u64
#print axioms TVA.ConvRel.iff_cast_u64_i8
#print axioms TVA.ConvRel.iff_cast_u64_i16
#print axioms TVA.ConvRel.iff_cast_u64_i32
#print axioms TVA.ConvRel.iff_cast_u64_i64
#print axioms TVA.ConvRel.iff_cast_u64_u8
#print axioms TVA.ConvRel.iff_cast_u64_u16
#print axioms TVA.ConvRel.iff_cast_u64_u32
#print axioms TVA.ConvRel.iff_cast_u64_u64
