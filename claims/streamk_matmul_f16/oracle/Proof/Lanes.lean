import Claim
import TVA.Proof
import Proof.Index

/-!
# `Proof.Lanes`

Lane pointers, masks and offsets of the `a`, `b` and `c` tiles, lane by lane.
-/

/-!
# The two-dimensional lanes of the `a`, `b` and `c` accesses

The kernel touches `a` with the `[128, 64]` pointer tensor
`a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak`
(masked by `offs_m < M` and `offs_k < K`), `b` with the `[64, 128]` tensor
`b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn`
(masked by `offs_k < K` and `offs_n < N`), and `c` with the `[128, 128]` tensor
`c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn`
(masked by `offs_m < M` and `offs_n < N`).

This file computes the address and the mask of every lane: lane `(i, j)` of the `a` access is
element `(pid_m * 128 + i) * stride_am + (kk * 64 + j) * stride_ak` of `aIdx`, and so on.

The shapes are concrete lists, so `simp` cannot rewrite `(x + y).get (i, j, ())` (the `Idx` of the
shape is not reducible at the transparency level a rewrite uses).  Every lemma below therefore
states the unfolded form (`offs1_val`, `ptrs*_get`, `mask*_get`) or begins with `show` of the
unfolded form, as `TVA/Lemmas/Block.lean` does.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-! ## One-dimensional offsets -/

/-- `offs = s * BLOCK + tl.arange 0 BLOCK`, the kernel's `offs_m`, `offs_n`, `offs_k`. -/
def offs1 (s : I32) (n : Nat) : Tensor [n] I32 := s * (IntV.ofNat n : I32) + tl.arange 0 n

/-- The value of an `offs` tensor at a lane. -/
theorem offs1_val (s : I32) (n : Nat) (i : Fin n) :
    (offs1 s n).get (i, ()) = s * (IntV.ofNat n : I32) + IntV.ofInt (0 + (i.val : Int)) := rfl

/-- The value of an `offs` tensor at a lane, as an integer: `u * n + i`. -/
theorem offs1_toInt {s : I32} {u n : Nat} (hs : s.toInt = (u : Int)) (hn : n < 2 ^ 31)
    (i : Fin n) (ha : u * n + i.val < 2 ^ 31) :
    ((offs1 s n).get (i, ())).toInt = ((u * n + i.val : Nat) : Int) := by
  have hi : i.val < n := i.isLt
  have hm : u * n < 2 ^ 31 := by omega
  have hb : (IntV.ofInt (0 + (i.val : Int)) : I32).toInt = (i.val : Nat) := by
    rw [IntV.I32.toInt_ofInt] <;> push_cast
    · ring
    · constructor <;> omega
  show ((s * (IntV.ofNat n : I32) + IntV.ofInt (0 + (i.val : Int))).toInt) = _
  rw [IntV.I32.add_nat (IntV.I32.mul_nat hs (IntV.I32.toInt_ofNat (by omega)) hm) hb (by omega)]

/-- Two pointers with the same address are equal. -/
theorem ptr_eq_of_addr {d : DType} {p q : Ptr d} (h : p.addr = q.addr) : p = q := by
  obtain ⟨a⟩ := p
  obtain ⟨b⟩ := q
  simpa using h

/-! ## The `a` access -/

/-- The `[128, 64]` pointer tensor of the `a` load. -/
def ptrsA (ap : Ptr .f16) (om : Tensor [128] I32) (ok : Tensor [64] I32) (am ak : I32) :
    Tensor [128, 64] (Ptr .f16) := ap + om.col 64 * am + ok.row 128 * ak

/-- The mask of the `a` load. -/
def maskA (om : Tensor [128] I32) (ok : Tensor [64] I32) (M K : I32) : Tensor [128, 64] Bool :=
  (om.col 64).ltS M |>.and ((ok.row 128).ltS K)

/-- The pointer of lane `(i, k)` of an `a` tile. -/
theorem ptrsA_get (ap : Ptr .f16) (om : Tensor [128] I32) (ok : Tensor [64] I32) (am ak : I32)
    (i : Fin 128) (j : Fin 64) :
    (ptrsA ap om ok am ak).get (i, j, ()) =
      ap.offset (((om.get (i, ())) * am).toInt + ((ok.get (j, ())) * ak).toInt) := by
  show (ap.offset (((om.get (i, ())) * am).toInt)).offset (((ok.get (j, ())) * ak).toInt) = _
  rw [Ptr.offset_offset]

/-- **The pointer of a lane of the `a` load.** -/
theorem ptrsA_lane (hd : d.ok) {pm kk : I32} {u v : Nat} (hpm : pm.toInt = (u : Int))
    (hkk : kk.toInt = (v : Int)) (i : Fin 128) (j : Fin 64)
    (hr : u * 128 + i.val < d.M) (hk : v * 64 + j.val < d.K) :
    (ptrsA (d.a.ptr 0) (offs1 pm 128) (offs1 kk 64) (IntV.ofNat d.stride_am)
        (IntV.ofNat d.stride_ak)).get (i, j, ()) =
      d.a.ptr (((u * 128 + i.val) * d.stride_am + (v * 64 + j.val) * d.stride_ak : Nat) : Int) := by
  have hr31 : u * 128 + i.val < 2 ^ 31 := by have := M_lt hd; omega
  have hk31 : v * 64 + j.val < 2 ^ 31 := by have := K_lt hd; omega
  have hrow := offs1_toInt (s := pm) (u := u) (n := 128) hpm (by norm_num) i hr31
  have hcol := offs1_toInt (s := kk) (u := v) (n := 64) hkk (by norm_num) j hk31
  have ham : (IntV.ofNat d.stride_am : I32).toInt = (d.stride_am : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.am; omega)
  have hak : (IntV.ofNat d.stride_ak : I32).toInt = (d.stride_ak : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.ak; omega)
  have hp1 : (u * 128 + i.val) * d.stride_am < 2 ^ 31 := by
    have h1 : u * 128 + i.val ≤ d.M - 1 := by omega
    have h2 : (u * 128 + i.val) * d.stride_am ≤ (d.M - 1) * d.stride_am :=
      Nat.mul_le_mul_right _ h1
    have h3 := hd.am_fit
    omega
  have hp2 : (v * 64 + j.val) * d.stride_ak < 2 ^ 31 := by
    have h1 : v * 64 + j.val ≤ d.K - 1 := by omega
    have h2 : (v * 64 + j.val) * d.stride_ak ≤ (d.K - 1) * d.stride_ak :=
      Nat.mul_le_mul_right _ h1
    have h3 := hd.ak_fit
    omega
  have hmul1 : ((((offs1 pm 128).get (i, ())) * (IntV.ofNat d.stride_am : I32)).toInt)
      = (((u * 128 + i.val) * d.stride_am : Nat) : Int) :=
    IntV.I32.mul_nat hrow ham hp1
  have hmul2 : ((((offs1 kk 64).get (j, ())) * (IntV.ofNat d.stride_ak : I32)).toInt)
      = (((v * 64 + j.val) * d.stride_ak : Nat) : Int) :=
    IntV.I32.mul_nat hcol hak hp2
  rw [ptrsA_get, hmul1, hmul2]
  apply ptr_eq_of_addr
  simp only [Ptr.offset_addr, Buf.ptr]
  push_cast
  ring

/-- **The mask of a lane of the `a` load.** -/
theorem maskA_get (hd : d.ok) {pm kk : I32} {u v : Nat} (hpm : pm.toInt = (u : Int))
    (hkk : kk.toInt = (v : Int)) (i : Fin 128) (j : Fin 64)
    (hu : u * 128 + i.val < 2 ^ 31) (hv : v * 64 + j.val < 2 ^ 31) :
    (maskA (offs1 pm 128) (offs1 kk 64) (IntV.ofNat d.M) (IntV.ofNat d.K)).get (i, j, ()) =
      (decide (u * 128 + i.val < d.M) && decide (v * 64 + j.val < d.K)) := by
  have hrow := offs1_toInt (s := pm) (u := u) (n := 128) hpm (by norm_num) i hu
  have hcol := offs1_toInt (s := kk) (u := v) (n := 64) hkk (by norm_num) j hv
  have hM : (IntV.ofNat d.M : I32).toInt = (d.M : Int) := M_toInt hd
  have hK : (IntV.ofNat d.K : I32).toInt = (d.K : Int) := K_toInt hd
  show (IntV.lt ((offs1 pm 128).get (i, ())) (IntV.ofNat d.M) &&
      IntV.lt ((offs1 kk 64).get (j, ())) (IntV.ofNat d.K)) = _
  rw [IntV.lt_eq, IntV.lt_eq, hrow, hcol, hM, hK]
  have e1 : decide (((u * 128 + i.val : Nat) : Int) < ((d.M : Nat) : Int)) =
      decide (u * 128 + i.val < d.M) := by
    simp only [decide_eq_decide]
    norm_cast
  have e2 : decide (((v * 64 + j.val : Nat) : Int) < ((d.K : Nat) : Int)) =
      decide (v * 64 + j.val < d.K) := by
    simp only [decide_eq_decide]
    norm_cast
  rw [e1, e2]

/-! ## The `b` access -/

/-- The `[64, 128]` pointer tensor of the `b` load. -/
def ptrsB (bp : Ptr .f16) (ok : Tensor [64] I32) (on : Tensor [128] I32) (bk bn : I32) :
    Tensor [64, 128] (Ptr .f16) := bp + ok.col 128 * bk + on.row 64 * bn

/-- The mask of the `b` load. -/
def maskB (ok : Tensor [64] I32) (on : Tensor [128] I32) (K N : I32) : Tensor [64, 128] Bool :=
  (ok.col 128).ltS K |>.and ((on.row 64).ltS N)

/-- The pointer of lane `(k, j)` of a `b` tile. -/
theorem ptrsB_get (bp : Ptr .f16) (ok : Tensor [64] I32) (on : Tensor [128] I32) (bk bn : I32)
    (i : Fin 64) (j : Fin 128) :
    (ptrsB bp ok on bk bn).get (i, j, ()) =
      bp.offset (((ok.get (i, ())) * bk).toInt + ((on.get (j, ())) * bn).toInt) := by
  show (bp.offset (((ok.get (i, ())) * bk).toInt)).offset (((on.get (j, ())) * bn).toInt) = _
  rw [Ptr.offset_offset]

/-- **The pointer of a lane of the `b` load.** -/
theorem ptrsB_lane (hd : d.ok) {kk pn : I32} {v w : Nat} (hkk : kk.toInt = (v : Int))
    (hpn : pn.toInt = (w : Int)) (i : Fin 64) (j : Fin 128)
    (hk : v * 64 + i.val < d.K) (hn : w * 128 + j.val < d.N) :
    (ptrsB (d.b.ptr 0) (offs1 kk 64) (offs1 pn 128) (IntV.ofNat d.stride_bk)
        (IntV.ofNat d.stride_bn)).get (i, j, ()) =
      d.b.ptr (((v * 64 + i.val) * d.stride_bk + (w * 128 + j.val) * d.stride_bn : Nat) : Int) := by
  have hk31 : v * 64 + i.val < 2 ^ 31 := by have := K_lt hd; omega
  have hn31 : w * 128 + j.val < 2 ^ 31 := by have := N_lt hd; omega
  have hrow := offs1_toInt (s := kk) (u := v) (n := 64) hkk (by norm_num) i hk31
  have hcol := offs1_toInt (s := pn) (u := w) (n := 128) hpn (by norm_num) j hn31
  have hbk : (IntV.ofNat d.stride_bk : I32).toInt = (d.stride_bk : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.bk; omega)
  have hbn : (IntV.ofNat d.stride_bn : I32).toInt = (d.stride_bn : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.bn; omega)
  have hp1 : (v * 64 + i.val) * d.stride_bk < 2 ^ 31 := by
    have h1 : v * 64 + i.val ≤ d.K - 1 := by omega
    have h2 : (v * 64 + i.val) * d.stride_bk ≤ (d.K - 1) * d.stride_bk :=
      Nat.mul_le_mul_right _ h1
    have h3 := hd.bk_fit
    omega
  have hp2 : (w * 128 + j.val) * d.stride_bn < 2 ^ 31 := by
    have h1 : w * 128 + j.val ≤ d.N - 1 := by omega
    have h2 : (w * 128 + j.val) * d.stride_bn ≤ (d.N - 1) * d.stride_bn :=
      Nat.mul_le_mul_right _ h1
    have h3 := hd.bn_fit
    omega
  have hmul1 : ((((offs1 kk 64).get (i, ())) * (IntV.ofNat d.stride_bk : I32)).toInt)
      = (((v * 64 + i.val) * d.stride_bk : Nat) : Int) :=
    IntV.I32.mul_nat hrow hbk hp1
  have hmul2 : ((((offs1 pn 128).get (j, ())) * (IntV.ofNat d.stride_bn : I32)).toInt)
      = (((w * 128 + j.val) * d.stride_bn : Nat) : Int) :=
    IntV.I32.mul_nat hcol hbn hp2
  rw [ptrsB_get, hmul1, hmul2]
  apply ptr_eq_of_addr
  simp only [Ptr.offset_addr, Buf.ptr]
  push_cast
  ring

/-- **The mask of a lane of the `b` load.** -/
theorem maskB_get (hd : d.ok) {kk pn : I32} {v w : Nat} (hkk : kk.toInt = (v : Int))
    (hpn : pn.toInt = (w : Int)) (i : Fin 64) (j : Fin 128)
    (hv : v * 64 + i.val < 2 ^ 31) (hw : w * 128 + j.val < 2 ^ 31) :
    (maskB (offs1 kk 64) (offs1 pn 128) (IntV.ofNat d.K) (IntV.ofNat d.N)).get (i, j, ()) =
      (decide (v * 64 + i.val < d.K) && decide (w * 128 + j.val < d.N)) := by
  have hrow := offs1_toInt (s := kk) (u := v) (n := 64) hkk (by norm_num) i hv
  have hcol := offs1_toInt (s := pn) (u := w) (n := 128) hpn (by norm_num) j hw
  have hK : (IntV.ofNat d.K : I32).toInt = (d.K : Int) := K_toInt hd
  have hN : (IntV.ofNat d.N : I32).toInt = (d.N : Int) := N_toInt hd
  show (IntV.lt ((offs1 kk 64).get (i, ())) (IntV.ofNat d.K) &&
      IntV.lt ((offs1 pn 128).get (j, ())) (IntV.ofNat d.N)) = _
  rw [IntV.lt_eq, IntV.lt_eq, hrow, hcol, hK, hN]
  have e1 : decide (((v * 64 + i.val : Nat) : Int) < ((d.K : Nat) : Int)) =
      decide (v * 64 + i.val < d.K) := by
    simp only [decide_eq_decide]
    norm_cast
  have e2 : decide (((w * 128 + j.val : Nat) : Int) < ((d.N : Nat) : Int)) =
      decide (w * 128 + j.val < d.N) := by
    simp only [decide_eq_decide]
    norm_cast
  rw [e1, e2]

/-! ## The `c` access -/

/-- The `[128, 128]` pointer tensor of the `c` store / atomic add. -/
def ptrsC (cp : Ptr .f32) (om : Tensor [128] I32) (on : Tensor [128] I32) (cm cn : I32) :
    Tensor [128, 128] (Ptr .f32) := cp + om.col 128 * cm + on.row 128 * cn

/-- The mask of the `c` store / atomic add. -/
def maskC (om : Tensor [128] I32) (on : Tensor [128] I32) (M N : I32) : Tensor [128, 128] Bool :=
  (om.col 128).ltS M |>.and ((on.row 128).ltS N)

/-- The pointer of lane `(i, j)` of a `c` tile. -/
theorem ptrsC_get (cp : Ptr .f32) (om : Tensor [128] I32) (on : Tensor [128] I32) (cm cn : I32)
    (i : Fin 128) (j : Fin 128) :
    (ptrsC cp om on cm cn).get (i, j, ()) =
      cp.offset (((om.get (i, ())) * cm).toInt + ((on.get (j, ())) * cn).toInt) := by
  show (cp.offset (((om.get (i, ())) * cm).toInt)).offset (((on.get (j, ())) * cn).toInt) = _
  rw [Ptr.offset_offset]

/-- **The pointer of a lane of the `c` store.** -/
theorem ptrsC_lane (hd : d.ok) {pm pn : I32} {u w : Nat} (hpm : pm.toInt = (u : Int))
    (hpn : pn.toInt = (w : Int)) (i : Fin 128) (j : Fin 128)
    (hr : u * 128 + i.val < d.M) (hn : w * 128 + j.val < d.N) :
    (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
        (IntV.ofNat d.stride_cn)).get (i, j, ()) =
      d.c.ptr (((u * 128 + i.val) * d.stride_cm + (w * 128 + j.val) * d.stride_cn : Nat) : Int) := by
  have hr31 : u * 128 + i.val < 2 ^ 31 := by have := M_lt hd; omega
  have hn31 : w * 128 + j.val < 2 ^ 31 := by have := N_lt hd; omega
  have hrow := offs1_toInt (s := pm) (u := u) (n := 128) hpm (by norm_num) i hr31
  have hcol := offs1_toInt (s := pn) (u := w) (n := 128) hpn (by norm_num) j hn31
  have hcm : (IntV.ofNat d.stride_cm : I32).toInt = (d.stride_cm : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.cm; omega)
  have hcn : (IntV.ofNat d.stride_cn : I32).toInt = (d.stride_cn : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.cn.2; omega)
  have hp1 : (u * 128 + i.val) * d.stride_cm < 2 ^ 31 := by
    have h1 : u * 128 + i.val ≤ d.M - 1 := by omega
    have h2 : (u * 128 + i.val) * d.stride_cm ≤ (d.M - 1) * d.stride_cm :=
      Nat.mul_le_mul_right _ h1
    have h3 := hd.cm_fit
    omega
  have hp2 : (w * 128 + j.val) * d.stride_cn < 2 ^ 31 := by
    have h1 : w * 128 + j.val ≤ d.N - 1 := by omega
    have h2 : (w * 128 + j.val) * d.stride_cn ≤ (d.N - 1) * d.stride_cn :=
      Nat.mul_le_mul_right _ h1
    have h3 : (d.N - 1) * d.stride_cn ≤ d.N * d.stride_cn :=
      Nat.mul_le_mul_right _ (by omega)
    have h4 : d.N * d.stride_cn ≤ d.stride_cm := hd.c_rows
    have h5 := hd.cm
    omega
  have hmul1 : ((((offs1 pm 128).get (i, ())) * (IntV.ofNat d.stride_cm : I32)).toInt)
      = (((u * 128 + i.val) * d.stride_cm : Nat) : Int) :=
    IntV.I32.mul_nat hrow hcm hp1
  have hmul2 : ((((offs1 pn 128).get (j, ())) * (IntV.ofNat d.stride_cn : I32)).toInt)
      = (((w * 128 + j.val) * d.stride_cn : Nat) : Int) :=
    IntV.I32.mul_nat hcol hcn hp2
  rw [ptrsC_get, hmul1, hmul2]
  apply ptr_eq_of_addr
  simp only [Ptr.offset_addr, Buf.ptr]
  push_cast
  ring

/-- **The mask of a lane of the `c` store.** -/
theorem maskC_get (hd : d.ok) {pm pn : I32} {u w : Nat} (hpm : pm.toInt = (u : Int))
    (hpn : pn.toInt = (w : Int)) (i : Fin 128) (j : Fin 128)
    (hu : u * 128 + i.val < 2 ^ 31) (hw : w * 128 + j.val < 2 ^ 31) :
    (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)).get (i, j, ()) =
      (decide (u * 128 + i.val < d.M) && decide (w * 128 + j.val < d.N)) := by
  have hrow := offs1_toInt (s := pm) (u := u) (n := 128) hpm (by norm_num) i hu
  have hcol := offs1_toInt (s := pn) (u := w) (n := 128) hpn (by norm_num) j hw
  have hM : (IntV.ofNat d.M : I32).toInt = (d.M : Int) := M_toInt hd
  have hN : (IntV.ofNat d.N : I32).toInt = (d.N : Int) := N_toInt hd
  show (IntV.lt ((offs1 pm 128).get (i, ())) (IntV.ofNat d.M) &&
      IntV.lt ((offs1 pn 128).get (j, ())) (IntV.ofNat d.N)) = _
  rw [IntV.lt_eq, IntV.lt_eq, hrow, hcol, hM, hN]
  have e1 : decide (((u * 128 + i.val : Nat) : Int) < ((d.M : Nat) : Int)) =
      decide (u * 128 + i.val < d.M) := by
    simp only [decide_eq_decide]
    norm_cast
  have e2 : decide (((w * 128 + j.val : Nat) : Int) < ((d.N : Nat) : Int)) =
      decide (w * 128 + j.val < d.N) := by
    simp only [decide_eq_decide]
    norm_cast
  rw [e1, e2]

/-! ## The lane bounds of a tile -/

/-- The tile row is below `tilesM` (for a tile index). -/
theorem pidMOf_lt' {u : Nat} (hd : d.ok) (hu : u < tilesM d * tilesN d) :
    pidMOf d u < tilesM d := by
  have hN := tilesN_pos hd
  unfold pidMOf
  rw [Nat.div_lt_iff_lt_mul hN]
  simpa [Nat.mul_comm] using hu

/-- The tile column is below `tilesN` (for a tile index). -/
theorem pidNOf_lt' {u : Nat} (hd : d.ok) (_hu : u < tilesM d * tilesN d) :
    pidNOf d u < tilesN d := by
  have hN := tilesN_pos hd
  unfold pidNOf
  exact Nat.mod_lt _ hN

/-- The `m` lane of a tile is below `2 ^ 31`. -/
theorem pidM_bound {u : Nat} (hd : d.ok) (hu : u < tilesM d * tilesN d) (i : Fin 128) :
    pidMOf d u * 128 + i.val < 2 ^ 31 := by
  have hlt := pidMOf_lt' hd hu
  have h1 : pidMOf d u + 1 ≤ tilesM d := hlt
  have h2 : (pidMOf d u + 1) * 128 ≤ tilesM d * 128 := Nat.mul_le_mul_right _ h1
  have h3 : tilesM d * 128 ≤ d.M + 127 := by
    unfold tilesM
    exact Nat.div_mul_le_self _ _
  have h4 := hd.M_le
  have h5 : i.val < 128 := i.isLt
  have h6 : (pidMOf d u + 1) * 128 = pidMOf d u * 128 + 128 := by ring
  omega

/-- The `n` lane of a tile is below `2 ^ 31`. -/
theorem pidN_bound {u : Nat} (hd : d.ok) (hu : u < tilesM d * tilesN d) (j : Fin 128) :
    pidNOf d u * 128 + j.val < 2 ^ 31 := by
  have hlt := pidNOf_lt' hd hu
  have h1 : pidNOf d u + 1 ≤ tilesN d := hlt
  have h2 : (pidNOf d u + 1) * 128 ≤ tilesN d * 128 := Nat.mul_le_mul_right _ h1
  have h3 : tilesN d * 128 ≤ d.N + 127 := by
    unfold tilesN
    exact Nat.div_mul_le_self _ _
  have h4 := hd.N_le
  have h5 : j.val < 128 := j.isLt
  have h6 : (pidNOf d u + 1) * 128 = pidNOf d u * 128 + 128 := by ring
  omega

/-- The `k` lane of a chunk is below `2 ^ 31`. -/
theorem kk_bound {v : Nat} (hd : d.ok) (hv : v < steps d.K) (j : Fin 64) :
    v * 64 + j.val < 2 ^ 31 := by
  have h1 : v + 1 ≤ steps d.K := hv
  have h2 : (v + 1) * 64 ≤ steps d.K * 64 := Nat.mul_le_mul_right _ h1
  have h3 : steps d.K * 64 ≤ d.K + 63 := by
    unfold steps
    exact Nat.div_mul_le_self _ _
  have h4 := K_lt hd
  have h5 : j.val < 64 := j.isLt
  have h6 : (v + 1) * 64 = v * 64 + 64 := by ring
  omega

end Claims.StreamKMatmul


/-!
# The shared cells and write footprints of the Stream-K launch

The launch splits the `total_iters = tilesM · tilesN · steps K` iterations (chunk `it % steps K`
of tile `it / steps K`) into contiguous runs, one per program.  A tile whose chunk run is the
whole `[0, steps K)` for one program is *stored* by that program; a tile no program owns entirely
is accumulated with `tl.atomic_add`, so its cells are shared between the programs that handle it.

* `owns d p u` — program `p`'s run is the whole of tile `u`.
* `storedOuts d p` — the output elements of the tiles `p` owns (private writes).
* `splitOuts d` — the output elements of the tiles nobody owns (atomic cells).

The byte sets `Wset d p` (private writes of `p`) and `Cbytes d` (atomic cells) are what
`TVA.shared` needs; `Ccells d` is the same set as cells.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-! ## Output elements -/

/-- The byte address of output element `(i, j)`. -/
def cAddr (d : Inst) (i j : Nat) : Int :=
  (d.c.ptr ((i * d.stride_cm + j * d.stride_cn : Nat) : Int)).addr

/-- The output elements, in bounds. -/
def allOuts (d : Inst) : Set (Nat × Nat) := {q | q.1 < d.M ∧ q.2 < d.N}

/-- The tiles of the launch. -/
def tiles (d : Inst) : Set Nat := {u | u < tilesM d * tilesN d}

/-- The in-bounds output elements of tile `u`. -/
def tileOut (d : Inst) (u : Nat) : Set (Nat × Nat) :=
  {q | q ∈ allOuts d ∧ q.1 / 128 = pidMOf d u ∧ q.2 / 128 = pidNOf d u}

/-- Program `p` handles the whole chunk range of tile `u`: it stores the tile. -/
def owns (d : Inst) (p u : Nat) : Prop := c0 d p u = 0 ∧ c1 d p u = steps d.K

instance (d : Inst) (p u : Nat) : Decidable (owns d p u) := by
  unfold owns
  infer_instance

/-- No program owns tile `u` entirely: the tile is combined with atomics. -/
def splitTile (d : Inst) (u : Nat) : Prop := ∀ p, p < d.num_programs → ¬ owns d p u

/-- The output elements program `p` stores. -/
def storedOuts (d : Inst) (p : Nat) : Set (Nat × Nat) :=
  {q | ∃ u ∈ tiles d, owns d p u ∧ q ∈ tileOut d u}

/-- The output elements combined with atomics. -/
def splitOuts (d : Inst) : Set (Nat × Nat) :=
  {q | ∃ u ∈ tiles d, splitTile d u ∧ q ∈ tileOut d u}

/-! ## Bytes and cells -/

/-- The bytes of a set of output elements. -/
def outBytes (d : Inst) (S : Set (Nat × Nat)) : Set Int :=
  {x | ∃ q ∈ S, x ∈ Mem.span .f32 (cAddr d q.1 q.2)}

/-- The bytes program `p` writes. -/
def Wset (d : Inst) (p : Nat) : Set Int := outBytes d (storedOuts d p)

/-- The bytes of the atomic cells. -/
def Cbytes (d : Inst) : Set Int := outBytes d (splitOuts d)

/-- The atomic cells. -/
def Ccells (d : Inst) : Set (DType × Int) :=
  {c | ∃ q ∈ splitOuts d, c = (.f32, cAddr d q.1 q.2)}

/-- The bytes of the `a` and `b` operands, read by every program. -/
def inElems (d : Inst) : Set Int := d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)

/-- The bytes of the output buffer. -/
def outElems (d : Inst) : Set Int := d.c.elems (cIdx d)

end Claims.StreamKMatmul

