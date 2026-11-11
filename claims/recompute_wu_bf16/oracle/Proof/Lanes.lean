import Claim
import TVA.Proof
import Proof.Prog
import Proof.Defs

/-!
# The row offsets, masks and pointers of the tiles

`o_t` (line 99) and `m_t` (line 101) lane by lane; a tile mask `m_t[:, None] & (cols < N)`; the
base pointers `b.ptr 0 + k`.
-/

namespace Claims.RecomputeWU

open TVA

/-- Lane `r` of `o_t = i_t · 64 + tl.arange(0, 64)` is `it·64 + r`. -/
theorem oT_lane {i_t : I64} {it : Nat} (hit : i_t.toInt = it) (hfit : it * 64 + 64 < 2 ^ 62)
    (r : Fin 64) : ((oT i_t).get1 r).toInt = ((it * 64 + r : Nat) : Int) := by
  unfold oT
  rw [Tensor.get1_sadd]
  have h1 : (i_t * (IntV.ofNat 64 : I64)).toInt = ((it * 64 : Nat) : Int) :=
    IntV.mul_ofNat hit (by omega)
  have h2 : (((tl.arange 0 64).castInt : Tensor [64] I64).get1 r).toInt = (r : Nat) := by
    rw [Tensor.get1_castInt]
    have h := IntV.I32.toInt_tl_arange (n := 64) (by norm_num) r
    rw [IntV.toInt_cast_of_sfits _ (by norm_num) (by rw [h]; exact ⟨by omega, by have := r.isLt; omega⟩), h]
  exact IntV.add_nat h1 h2 (by have := r.isLt; omega)

/-- Lane `r` of `m_t = o_t < T` is whether the row is inside the sequence. -/
theorem mT_lane {o_t : Tensor [64] I64} {T_n : I64} {it len : Nat}
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int)) (hT : T_n.toInt = len)
    (r : Fin 64) : (mT o_t T_n).get1 r = rowEn it len r := by
  unfold mT rowEn
  rw [Tensor.get1_ltS, IntV.lt_eq, ho r, hT]
  simp only [Nat.cast_lt]

/-- A tile mask `m_t[:, None] & (cols < N)`: lane `(i, j)` is `rowEn ∧ c0 + j < N`. -/
theorem mask_tile {m_t : Tensor [64] Bool} {cols : Tensor [64] I32} {nv : I32} {it len c0 N : Nat}
    (hm : ∀ r : Fin 64, m_t.get1 r = rowEn it len r)
    (hc : ∀ j : Fin 64, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (hn : nv.toInt = N)
    (i j : Fin 64) :
    ((m_t.col 64).and ((cols.ltS nv).row 64)).get2 i j =
      (decide (it * 64 + i < len) && decide (c0 + j < N)) := by
  show (m_t.get1 i && (cols.ltS nv).get1 j) = _
  rw [hm i, Tensor.get1_ltS, IntV.lt_eq, hc j, hn]
  simp only [Nat.cast_lt, rowEn]

/-- `b.ptr 0 + k` for a scalar `k` of value `x`. -/
theorem ptr_zero_add {d : DType} (b : Buf d) {sg : Bool} {w : Nat} {k : IntV sg w} {x : Int}
    (h : k.toInt = x) : b.ptr 0 + k = b.ptr x := by
  rw [Ptr.add_int, Buf.ptr_offset, h, zero_add]

/-- `b.ptr 0 + k + k'` for scalars of values `x`, `y`. -/
theorem ptr_zero_add2 {d : DType} (b : Buf d) {sg : Bool} {w : Nat} {k : IntV sg w}
    {sg' : Bool} {w' : Nat} {k' : IntV sg' w'} {x y : Int} (h : k.toInt = x) (h' : k'.toInt = y) :
    b.ptr 0 + k + k' = b.ptr (x + y) := by
  rw [ptr_zero_add b h, Ptr.add_int, Buf.ptr_offset, h']

end Claims.RecomputeWU
