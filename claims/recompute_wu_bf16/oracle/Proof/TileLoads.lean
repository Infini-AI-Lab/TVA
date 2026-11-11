import Claim
import TVA.Proof
import Proof.VecLoads

/-!
# The `[64, 64]` tile loads

The pointer tile of lines 106, 112 and 126 is `b.ptr 0 + S + o_t[:, None] · c + cols[None, :]`
for a scalar `S` (the sequence's and head's base) and a row stride `c`. `safe_load_tile_ptr`
loads it, masked by `m_t[:, None] & (cols < N)`, `other=0.0`.
-/

namespace Claims.RecomputeWU

open TVA

variable {E : Env} {W : Set Int} {Q : Unit → Mem → Prop}

/-- **A masked tile load from an input `bfloat16` buffer** with the pointers and mask the source
writes: the base `S` is the buffer's element `e`, the row stride `c` is `s`, the columns are
`c0 + j` below `N`. Lane `(i, j)` reads element `e + (it·64 + i)·s + (c0 + j)`, and a masked lane
holds `0`. -/
theorem safe_load_tile_ptr {d : Inst} {init m : Mem} (hag : Ag d init m) (b : Buf .bf16)
    (hbR : b.bytes ⊆ Rall d) (hv : b.Valid init) {S c : I64} {nv : I32}
    {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool} {cols : Tensor [64] I32} {it len c0 N e s : Nat}
    (hS : S.toInt = e) (hc : c.toInt = s)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r)
    (hcols : ∀ j : Fin 64, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (hnv : nv.toInt = N) (hfit : (len - 1) * s < 2 ^ 63)
    (hlen : len = 0 ∨ N = 0 ∨ e + (len - 1) * s + (N - 1) * 1 < b.len)
    {k : Tensor [64, 64] BF16 → Prog Unit}
    (hk : ∀ v : Tensor [64, 64] BF16,
      (∀ i j : Fin 64, it * 64 + i < len → c0 + j < N →
        v.get2 i j = bfV b init (e + (it * 64 + i) * s + (c0 + j) * 1)) →
      (∀ i j : Fin 64, ¬ (it * 64 + i < len ∧ c0 + j < N) → v.get2 i j = FloatV.ofPat 0) →
      Safe E (Rall d) W Q m (k v)) :
    Safe E (Rall d) W Q m
      (tl.convert .f32 .bf16 (tl.full [64, 64] (FloatV.litF32 (FloatV.ofPat 0))) >>= fun z =>
        tl.load (b.ptr 0 + S + o_t.col 64 * c + cols.row 64 : Tensor [64, 64] (Ptr .bf16))
          (some ((m_t.col 64).and ((cols.ltS nv).row 64))) (some z) >>= k) := by
  refine safe_conv_zero fun z hz => ?_
  refine safe_tile_load hag b hbR hv (ptrs := b.ptr 0 + S + o_t.col 64 * c + cols.row 64) ?_ ?_ hlen
    (fun i j => hz (i, j, ())) hk
  · intro i j hi hj
    exact Tile2.ptr_lane_seq_of_on b (ρ := fun i : Fin 64 => it * 64 + i.val)
      (κ := fun j : Fin 64 => c0 + j.val) (M := len) (N := N) (ptr_zero_add b hS)
      (fun i j hi => Tile2.colMul_toInt_on ho hc hfit i j hi)
      (fun i j _ => Tile2.row_toInt hcols i j) i j hi hj
  · intro i j
    exact mask_tile hmt hcols hnv i j


/-- **A masked tile store to an `bfloat16` output** with the pointers and mask the source writes
(as `safe_load_tile_ptr`): afterwards lane `(i, j)`'s element holds the value, nothing outside the
tile changed. The layout is row-major (`N ≤ s`). -/
theorem safe_store_tile_ptr {m : Mem} (b : Buf .bf16) (hv : b.Valid m) {S c : I64}
    {nv : I32} {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool} {cols : Tensor [64] I32}
    {it len c0 N e s : Nat} (hS : S.toInt = e) (hc : c.toInt = s)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r)
    (hcols : ∀ j : Fin 64, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (hnv : nv.toInt = N) (hfit : (len - 1) * s < 2 ^ 63) (hNs : N ≤ s)
    (hlen : len = 0 ∨ N = 0 ∨ e + (len - 1) * s + (N - 1) * 1 < b.len)
    (vals : Tensor [64, 64] BF16) {R : Set Int}
    (hW : b.tileAt e s 1 (fun i : Fin 64 => it * 64 + i.val) (fun j : Fin 64 => c0 + j.val) len N ⊆ W)
    {k : Unit → Prog Unit}
    (hk : ∀ m' : Mem,
      (∀ i j : Fin 64, it * 64 + i < len → c0 + j < N →
        m'.read .bf16 (b.ptr ((e + (it * 64 + i) * s + (c0 + j) * 1 : Nat) : Int)).addr =
          some (vals.get2 i j)) →
      (∀ x, x ∉ b.tileAt e s 1 (fun i : Fin 64 => it * 64 + i.val)
        (fun j : Fin 64 => c0 + j.val) len N → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.tileAt e s 1 (fun i : Fin 64 => it * 64 + i.val)
        (fun j : Fin 64 => c0 + j.val) len N) Q m' (k ())) :
    Safe E R W Q m
      (tl.store (b.ptr 0 + S + o_t.col 64 * c + cols.row 64 : Tensor [64, 64] (Ptr .bf16)) vals
        (some ((m_t.col 64).and ((cols.ltS nv).row 64))) >>= k) := by
  refine safe_store_tileAt (ρ := fun i : Fin 64 => it * 64 + i.val)
    (κ := fun j : Fin 64 => c0 + j.val) (M := len) (N := N) (e := e) (s0 := s) (s1 := 1) b ?_
    (Tile2.maskAt_of_get fun i j => mask_tile hmt hcols hnv i j) hlen (Tile2.inj_rowMajor1 hNs)
    (fun i i' hi hi' h => Fin.ext (by omega))
    (fun j j' hj hj' h => Fin.ext (by omega)) hv hW hk
  intro i j hi hj
  exact Tile2.ptr_lane_seq_of_on b (ρ := fun i : Fin 64 => it * 64 + i.val)
    (κ := fun j : Fin 64 => c0 + j.val) (M := len) (N := N) (ptr_zero_add b hS)
    (fun i j hi => Tile2.colMul_toInt_on ho hc hfit i j hi)
    (fun i j _ => Tile2.row_toInt hcols i j) i j hi hj

end Claims.RecomputeWU
