import Proof.Defs

/-!
# The index arithmetic of one program

Program `(n, i)` computes `i_n = n`, `i_d = i` (`int64`), the lanes `o_d[l] = i·BD + l`, the mask
`o_d < D`, the pointers `x + i_n·D + o_d` (element `n·D + i·BD + l`) and the store's pointer
`z + i_n·ND + i_d` (element `n·ND + i`). No `int64` value wraps: `N·D` and `N·ND` are below
`2^62`, since the buffers hold them below `2^64` bytes.
-/

namespace Claims.LogsumexpFwd

open TVA

/-- `N·D ≤ 2^62`: `x` holds `N·D` four-byte elements below `2^64`. -/
theorem ND_bound {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) : d.N * d.D ≤ 2 ^ 62 := by
  obtain ⟨h0, h1, -, -⟩ := hpre.1
  have h2 := hd.x_len
  have hb : (DType.f32.bytes : Int) = 4 := rfl
  rw [hb] at h1
  have h3 : ((d.N * d.D : Nat) : Int) ≤ (d.x.len : Int) := by exact_mod_cast h2
  have : ((d.N * d.D : Nat) : Int) ≤ 2 ^ 62 := by linarith
  exact_mod_cast this

/-- The `int64` value of a converted `int32` scalar of value `k`. -/
theorem conv_item {P : Platform} {x : I32} {r : Tensor [] I64} {k : Int} (hx : x.toInt = k)
    (hk : 0 ≤ k ∧ k < 2 ^ 31)
    (hr : ∀ i, ConvRel P .i32 .i64 ((Tensor.scalar x).get i) (r.get i)) :
    r.item.toInt = k := by
  have h1 : r.item = IntV.ofInt (IntV.toInt x) := hr ()
  rw [h1, hx]
  exact IntV.I64.toInt_ofInt ⟨by omega, by omega⟩

/-- Program `p`'s `tl.program_id(0)`. -/
theorem pid0_toInt {d : Inst} {p : Nat} (hp : p < d.N * ND d) (hN : d.N < 2 ^ 31) :
    (tl.program_id ((launch d).ctx p) 0).toInt = (pn d p : Int) := by
  simp only [tl.program_id, launch_pid0]
  exact IntV.I32.toInt_ofNat (lt_trans (pn_lt hp) hN)

/-- Program `p`'s `tl.program_id(1)`. -/
theorem pid1_toInt {d : Inst} {p : Nat} (hp : p < d.N * ND d) (hd : d.ok) :
    (tl.program_id ((launch d).ctx p) 1).toInt = (pi d p : Int) := by
  simp only [tl.program_id, launch_pid1]
  exact IntV.I32.toInt_ofNat (lt_trans (pi_lt hp) (lt_trans hd.nd_lt (by norm_num)))

/-- The lanes of the converted `tl.arange(0, BD)`. -/
theorem oa_toInt {P : Platform} {B : Nat} (hB : B ≤ 2 ^ 16) {o_a : Tensor [B] I64}
    (hoa : ∀ l, ConvRel P .i32 .i64 ((tl.arange 0 B).get l) (o_a.get l)) (l : Idx [B]) :
    (o_a.get l).toInt = (l.1.val : Int) := by
  have h1 : o_a.get l = IntV.ofInt (IntV.toInt ((tl.arange 0 B).get l)) := hoa l
  have hl := l.1.isLt
  have h2 : IntV.toInt ((tl.arange 0 B).get l) = (l.1.val : Int) := by
    have h16 : (l.1.val : Int) < 2 ^ 31 := by
      have : l.1.val < 2 ^ 31 := by omega
      exact_mod_cast this
    have hge : -(2 ^ 31 : Int) ≤ (l.1.val : Int) := by
      have : (0 : Int) ≤ (l.1.val : Int) := by exact_mod_cast Nat.zero_le _
      linarith
    show IntV.toInt ((tl.arange 0 B).get1 l.1) = _
    simp only [tl.arange, Nat.sub_zero, Tensor.get1_arange, Nat.cast_zero, Int.zero_add,
      IntV.I32.toInt_ofInt ⟨hge, h16⟩]
  rw [h1, h2]
  exact IntV.I64.toInt_ofInt ⟨by omega, by omega⟩

/-- The lanes of `o_d = i_d·BD + o_a`. -/
theorem od_toInt {B i : Nat} (hB : B ≤ 2 ^ 16) (hi : i < 2 ^ 16) {i_d : I64} {o_a : Tensor [B] I64}
    (hid : i_d.toInt = (i : Int)) (hoa : ∀ l : Idx [B], (o_a.get l).toInt = (l.1.val : Int))
    (l : Idx [B]) :
    ((i_d * (IntV.ofNat B : I64) + o_a).get l).toInt = ((i * B + l.1.val : Nat) : Int) := by
  have hl := l.1.isLt
  have hBv : (IntV.ofNat B : I64).toInt = (B : Int) := IntV.I64.toInt_ofNat (by omega)
  have hiB : i * B ≤ 2 ^ 16 * 2 ^ 16 := Nat.mul_le_mul hi.le hB
  have hm := IntV.I64.mul_nat hid hBv (by omega)
  rw [Tensor.get_sadd]
  exact IntV.I64.add_nat hm (hoa l) (by omega)

/-- The mask lanes: `o_d[l] < D`. -/
theorem md_get {B D i : Nat} (hD : D < 2 ^ 31) {o_d : Tensor [B] I64}
    (hod : ∀ l : Idx [B], (o_d.get l).toInt = ((i * B + l.1.val : Nat) : Int)) (l : Idx [B]) :
    (o_d.ltS (IntV.ofNat D : I64)).get l = decide (i * B + l.1.val < D) := by
  rw [Tensor.get_ltS, IntV.lt_eq, hod l, IntV.I64.toInt_ofNat (by omega)]
  exact decide_eq_decide.mpr Nat.cast_lt

/-- The load's pointer lanes. -/
theorem px_get {d : Inst} {n : Nat} {B : Nat} {i_n : I64} {o_d : Tensor [B] I64} {i : Nat}
    (hin : i_n.toInt = (n : Int)) (hnD : n * d.D ≤ 2 ^ 62) (hD : d.D < 2 ^ 31)
    (hod : ∀ l : Idx [B], (o_d.get l).toInt = ((i * B + l.1.val : Nat) : Int)) (l : Idx [B]) :
    (d.x.ptr 0 + i_n * (IntV.ofNat d.D : I64) + o_d).get l =
      d.x.ptr ((xIdx d n (i * B + l.1.val) : Nat) : Int) := by
  have hDv : (IntV.ofNat d.D : I64).toInt = (d.D : Int) := IntV.I64.toInt_ofNat (by omega)
  have hm := IntV.I64.mul_nat hin hDv (by omega)
  rw [Tensor.get_ptrAdd, Ptr.add_int, hm, hod l, Buf.ptr_offset, Buf.ptr_offset]
  congr 1
  simp only [xIdx]
  push_cast
  ring

/-- `tl.cdiv(D, BD)` of the two constexprs is `ND`. -/
theorem cdiv_eq {d : Inst} (hd : d.ok) : tl.py.cdiv (d.D : Int) (d.BD : Int) = (ND d : Int) := by
  have hb := bd_pos hd
  unfold tl.py.cdiv ND
  rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
  have : (d.D : Int) + ((d.BD : Int) - 1) = ((d.D + d.BD - 1 : Nat) : Int) := by
    push_cast [show 1 ≤ d.D + d.BD by omega]; ring
  rw [this]
  norm_cast

/-- The store's pointer. -/
theorem pz_eq {d : Inst} (hd : d.ok) {n i : Nat} {i_n i_d : I64}
    (hin : i_n.toInt = (n : Int)) (hid : i_d.toInt = (i : Int)) (hn : n < d.N) (hi : i < ND d) :
    d.z.ptr 0 + i_n * (IntV.ofInt (tl.py.cdiv d.D d.BD) : I64) + i_d =
      d.z.ptr ((zIdx d n i : Nat) : Int) := by
  have hg := hd.grid_lt
  have hnd := hd.nd_lt
  have hcv : (IntV.ofInt (tl.py.cdiv d.D d.BD) : I64).toInt = (ND d : Int) := by
    rw [cdiv_eq hd]
    exact IntV.I64.toInt_ofInt ⟨by omega, by omega⟩
  have hnND : n * ND d < 2 ^ 31 := by
    have : n * ND d ≤ d.N * ND d := Nat.mul_le_mul_right _ hn.le
    omega
  have hm := IntV.I64.mul_nat hin hcv (by omega)
  rw [Ptr.add_int, Ptr.add_int, hm, hid, Buf.ptr_offset, Buf.ptr_offset]
  congr 1
  simp only [zIdx]
  push_cast
  ring

end Claims.LogsumexpFwd
