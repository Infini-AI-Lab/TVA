import TVA
open TVA Tile2

/-! End to end: the load of a row-major `A[r0:r0+BM, c0:c0+BN]` tile, as a kernel writes it, from
the round-39 lemmas alone (pointers, mask, load). -/
example {d : DType} (b : Buf d) {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {BM BN r0 c0 M N s0 : Nat} {rows : Tensor [BM] I32} {cols : Tensor [BN] I32} {a0 mv nv : I32}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (hmv : mv.toInt = M) (hnv : nv.toInt = N)
    (hfit : (r0 + BM - 1) * s0 + (c0 + BN - 1) * 1 < 2 ^ (32 - 1))
    (hlen : 0 + (M - 1) * s0 + (N - 1) * 1 < b.len) (hv : b.Valid m)
    (hR : b.tile2 0 s0 1 r0 c0 BM BN M N ⊆ R) {k : Tensor [BM, BN] d.Val → Prog α}
    (hk : ∀ v : Tensor [BM, BN] d.Val,
      (∀ (i : Fin BM) (j : Fin BN), r0 + i < M → c0 + j < N →
        m.read d (b.ptr ((0 + (r0 + i) * s0 + (c0 + j) * 1 : Nat) : Int)).addr = some (v.get2 i j)) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load (b.ptr ((0 : Nat) : Int) + (rows.col BN * a0 + cols.row BM * (1 : I32)))
      (some (((rows.col BN).ltS mv).and ((cols.row BM).ltS nv))) none >>= k) :=
  safe_load_tile2 b (fun i j => ptr_lane b rfl (by norm_num) hr hc ha0 (by simp) hfit i j)
    (maskOk_and hr hc hmv hnv) hlen hv hR (fun v h1 _ => hk v h1)
