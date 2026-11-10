import TVA
open TVA Tile2

/-! A tile of an empty matrix (`M = 0`: every lane masked off) that starts at the end of its
buffer (`e = b.len`: the last sequence of a varlen batch, of length 0). -/

-- 1. The load is safe: `safe_load_buf` needs only the enabled lanes in `b`, and there are none.
example {d : DType} (b : Buf d) {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop}
    {m : Mem} {BM BN c0 N s0 s1 : Nat} {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)}
    (hp : ∀ i j, ptrs.get2 i j = b.ptr ((b.len + (0 + i) * s0 + (c0 + j) * s1 : Nat) : Int))
    (hm : MaskOk mk 0 c0 0 N) (hv : b.Valid m) (hR : b.tile2 b.len s0 s1 0 c0 BM BN 0 N ⊆ R) {k : Tensor [BM, BN] d.Val → Prog α}
    (hk : ∀ v, Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs mk none >>= k) :=
  safe_load_buf b (addrs_tile2 b hp hm) (fun i h => by simp at h) hv
    (by rw [img_tile2]; exact hR)
    (fun v _ _ => hk v)

-- 2. `safe_load_tile2` (and `safe_store_tile2`, `Buf.tile2_sub_bytes`) cannot be applied to it:
-- their premise `e + (M - 1) * s0 + (N - 1) * s1 < b.len` is false for `e = b.len`.
example {d : DType} (b : Buf d) {s0 s1 N : Nat} :
    ¬ (b.len + (0 - 1) * s0 + (N - 1) * s1 < b.len) := by omega
