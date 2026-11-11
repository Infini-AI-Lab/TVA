import Claim
import Proof.Defs
import TVA.Proof

/-!
# Ptr

The pointer and offset arithmetic of the program: that the `int32` and `int64` offsets of `x`, `B`, `dt`, `dA_cumsum`, `seq_idx` and `states` equal the indices `Claim.lean` names.
-/

namespace Claims.SsdChunkState

open TVA

set_option maxHeartbeats 4000000

/-!
# `ssd_chunk_state_bf16`: the prelude's pointer arithmetic

Each buffer pointer is `d.b.ptr 0 + (…)` with `int64` components the prelude converted from
`int32`; with the `toInt` values of those components (as recorded by the `Safe` walk), the sum is
the buffer's pointer to the element the tile starts at.
-/

set_option maxHeartbeats 800000


open TVA

/-! ## Small arithmetic helpers -/

namespace PtrAux

/-- A pointer plus an `int64` offset: the offset's value decides the element. -/
theorem ptr_add_eq_ptr {dd : DType} (b : Buf dd) (X : I64) (j : Nat)
    (h : X.toInt = (j : Int)) : (b.ptr 0 + X) = b.ptr (j : Int) := by
  show (b.ptr 0).offset X.toInt = b.ptr (j : Int)
  simp only [Ptr.offset, Buf.ptr]
  rw [h]
  simp

/-- A product whose left factor is zero has value zero. -/
theorem toInt_mul_zero_left {a b : I64} (ha : a.toInt = 0) : (a * b).toInt = 0 := by
  rw [IntV.toInt_mul ha rfl (x := 0) (y := b.toInt) (by simp [IntV.sfits_iff])]
  ring

/-- A product of two values below `2^31` is below `2^62`. -/
theorem mul_lt_62 {a b : Nat} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) : a * b < 2 ^ 62 := by
  nlinarith

/-- A product of a value below `2^16` and one below `2^31` is below `2^47`. -/
theorem mul_lt_47 {a b : Nat} (ha : a < 2 ^ 16) (hb : b < 2 ^ 31) : a * b < 2 ^ 47 := by
  nlinarith

/-- A value below `2^62` is below `2^63`. -/
theorem lt_63_of_lt_62 {a : Nat} (h : a < 2 ^ 62) : a < 2 ^ 63 := lt_trans h (by norm_num)

/-- A value below `2^31` is below `2^63`. -/
theorem lt_63_of_lt_31 {a : Nat} (h : a < 2 ^ 31) : a < 2 ^ 63 := lt_trans h (by norm_num)

/-- The group offset `(head // ratio)·stride_b_head` does not wrap. -/
theorem grp_mul_lt_aux {d : Inst} (hd : d.ok) {h : Nat} (hh : h < d.nheads) :
    grp d h * d.sBg < 2 ^ 31 := by
  obtain ⟨k, hk⟩ := hd.groups_dvd
  have hk1 : 0 < k := by
    have hk0 : k ≠ 0 := by
      rintro rfl
      have h0 : d.nheads = 0 := by simpa using hk
      exact (Nat.pos_iff_ne_zero.mp hd.heads_pos) h0
    exact Nat.pos_of_ne_zero hk0
  have hgl : grp d h < d.ngroups := by
    have hrk : ratio d = k := by
      unfold ratio
      rw [hk, Nat.mul_comm d.ngroups k]
      exact Nat.mul_div_left k hd.groups_pos
    unfold grp
    rw [hrk, Nat.div_lt_iff_lt_mul hk1]
    rw [hk] at hh
    exact hh
  have h1 : grp d h * d.sBg ≤ (d.ngroups - 1) * d.sBg :=
    Nat.mul_le_mul_right _ (Nat.le_pred_of_lt hgl)
  exact lt_of_le_of_lt h1 hd.bhead_lt

/-- `chunkOf · chunk_size < 2^31`. -/
theorem chunkOf_mul_lt {d : Inst} (hd : d.ok) (i : Nat) :
    chunkOf d i * d.chunk < 2 ^ 31 :=
  lt_trans (Nat.mul_lt_mul_of_pos_right (chunkOf_lt hd i) (chunk_pos hd)) hd.tok_lt

/-- `headOf · … < 2^16`. -/
theorem headOf_lt_16 {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    headOf d i < 2 ^ 16 :=
  lt_trans (headOf_lt hd i.isLt) hd.grid_z

end PtrAux

/-- `b_ptr`: `d.b.ptr (bIdx d (pid_c·chunk) (pid_h // ratio) 0)`. -/
theorem b_base {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {pb pc c1 s1 s2 : I64} (hb : pb.toInt = 0) (hc : pc.toInt = (chunkOf d i : Int))
    (hc1 : c1.toInt = (d.chunk : Int)) (hs1 : s1.toInt = (d.sBs : Int))
    (hs2 : s2.toInt = (grp d (headOf d i) * d.sBg : Int)) :
    (d.b.ptr 0 + (pb * (IntV.ofNat d.sBb : I64) + pc * c1 * s1 + s2))
      = d.b.ptr (bIdx d (chunkOf d i * d.chunk) (grp d (headOf d i)) 0) := by
  have hh : headOf d i < d.nheads := headOf_lt hd i.isLt
  have hgrp : grp d (headOf d i) * d.sBg < 2 ^ 31 := PtrAux.grp_mul_lt_aux hd hh
  have hsBs : d.sBs < 2 ^ 31 := hd.sB_lt.1
  have hck : chunkOf d i * d.chunk < 2 ^ 31 := PtrAux.chunkOf_mul_lt hd i
  have hprod : (chunkOf d i * d.chunk) * d.sBs < 2 ^ 62 := PtrAux.mul_lt_62 hck hsBs
  have t1 : (pb * (IntV.ofNat d.sBb : I64)).toInt = 0 := PtrAux.toInt_mul_zero_left hb
  have t2 : (pc * c1).toInt = ((chunkOf d i * d.chunk : Nat) : Int) :=
    IntV.I64.mul_nat hc hc1 (PtrAux.lt_63_of_lt_31 hck)
  have t3 : ((pc * c1) * s1).toInt = (((chunkOf d i * d.chunk) * d.sBs : Nat) : Int) :=
    IntV.I64.mul_nat t2 hs1 (PtrAux.lt_63_of_lt_62 hprod)
  have t12 : (pb * (IntV.ofNat d.sBb : I64) + (pc * c1) * s1).toInt
      = ((0 + (chunkOf d i * d.chunk) * d.sBs : Nat) : Int) :=
    IntV.I64.add_nat t1 t3 (by omega)
  have hs2n : s2.toInt = ((grp d (headOf d i) * d.sBg : Nat) : Int) := by
    rw [hs2]
    push_cast
    ring
  have t4 : (pb * (IntV.ofNat d.sBb : I64) + (pc * c1) * s1 + s2).toInt
      = ((0 + (chunkOf d i * d.chunk) * d.sBs + grp d (headOf d i) * d.sBg : Nat) : Int) :=
    IntV.I64.add_nat t12 hs2n (by omega)
  apply PtrAux.ptr_add_eq_ptr
  rw [show (bIdx d (chunkOf d i * d.chunk) (grp d (headOf d i)) 0 : Nat)
      = (0 + (chunkOf d i * d.chunk) * d.sBs + grp d (headOf d i) * d.sBg) by
    simp [bIdx]]
  exact t4

/-- `x_ptr`: `d.x.ptr (xIdx d (pid_c·chunk) pid_h 0)`. -/
theorem x_base {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {pb pc c1 s1 s2 : I64} (hb : pb.toInt = 0) (hc : pc.toInt = (chunkOf d i : Int))
    (hc1 : c1.toInt = (d.chunk : Int)) (hs1 : s1.toInt = (d.sXs : Int))
    (hs2 : s2.toInt = (headOf d i * d.sXh : Nat)) :
    (d.x.ptr 0 + (pb * (IntV.ofNat d.sXb : I64) + pc * c1 * s1 + s2))
      = d.x.ptr (xIdx d (chunkOf d i * d.chunk) (headOf d i) 0) := by
  have hsXs : d.sXs < 2 ^ 31 := hd.sX_lt.1
  have hsXh : d.sXh < 2 ^ 31 := hd.sX_lt.2.1
  have hck : chunkOf d i * d.chunk < 2 ^ 31 := PtrAux.chunkOf_mul_lt hd i
  have hprod : (chunkOf d i * d.chunk) * d.sXs < 2 ^ 62 := PtrAux.mul_lt_62 hck hsXs
  have hs2b : headOf d i * d.sXh < 2 ^ 47 :=
    PtrAux.mul_lt_47 (PtrAux.headOf_lt_16 hd i) hsXh
  have t1 : (pb * (IntV.ofNat d.sXb : I64)).toInt = 0 := PtrAux.toInt_mul_zero_left hb
  have t2 : (pc * c1).toInt = ((chunkOf d i * d.chunk : Nat) : Int) :=
    IntV.I64.mul_nat hc hc1 (PtrAux.lt_63_of_lt_31 hck)
  have t3 : ((pc * c1) * s1).toInt = (((chunkOf d i * d.chunk) * d.sXs : Nat) : Int) :=
    IntV.I64.mul_nat t2 hs1 (PtrAux.lt_63_of_lt_62 hprod)
  have t12 : (pb * (IntV.ofNat d.sXb : I64) + (pc * c1) * s1).toInt
      = ((0 + (chunkOf d i * d.chunk) * d.sXs : Nat) : Int) :=
    IntV.I64.add_nat t1 t3 (by omega)
  have t4 : (pb * (IntV.ofNat d.sXb : I64) + (pc * c1) * s1 + s2).toInt
      = ((0 + (chunkOf d i * d.chunk) * d.sXs + headOf d i * d.sXh : Nat) : Int) :=
    IntV.I64.add_nat t12 hs2 (by omega)
  apply PtrAux.ptr_add_eq_ptr
  rw [show (xIdx d (chunkOf d i * d.chunk) (headOf d i) 0 : Nat)
      = (0 + (chunkOf d i * d.chunk) * d.sXs + headOf d i * d.sXh) by
    simp [xIdx]]
  exact t4

/-- `dt_ptr`: `d.dt.ptr (dIdx d pid_h pid_c 0)`. -/
theorem dt_base {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {pb pc c1 s2 : I64} (hb : pb.toInt = 0) (hc : pc.toInt = (chunkOf d i : Int))
    (hc1 : c1.toInt = (d.chunk : Int)) (hs2 : s2.toInt = (headOf d i * sDh d : Nat)) :
    (d.dt.ptr 0 + (pb * (IntV.ofNat (sDb d) : I64) + pc * c1 + s2))
      = d.dt.ptr (dIdx d (headOf d i) (chunkOf d i) 0) := by
  have hsDh : sDh d = nc d * d.chunk := rfl
  have hck : chunkOf d i * d.chunk < 2 ^ 31 := PtrAux.chunkOf_mul_lt hd i
  have hs2b : headOf d i * sDh d < 2 ^ 47 := by
    rw [hsDh]
    exact PtrAux.mul_lt_47 (PtrAux.headOf_lt_16 hd i) hd.tok_lt
  have t1 : (pb * (IntV.ofNat (sDb d) : I64)).toInt = 0 := PtrAux.toInt_mul_zero_left hb
  have t2 : (pc * c1).toInt = ((chunkOf d i * d.chunk : Nat) : Int) :=
    IntV.I64.mul_nat hc hc1 (PtrAux.lt_63_of_lt_31 hck)
  have t12 : (pb * (IntV.ofNat (sDb d) : I64) + pc * c1).toInt
      = ((0 + chunkOf d i * d.chunk : Nat) : Int) :=
    IntV.I64.add_nat t1 t2 (by omega)
  have t4 : (pb * (IntV.ofNat (sDb d) : I64) + pc * c1 + s2).toInt
      = ((0 + chunkOf d i * d.chunk + headOf d i * sDh d : Nat) : Int) :=
    IntV.I64.add_nat t12 hs2 (by omega)
  apply PtrAux.ptr_add_eq_ptr
  rw [show (dIdx d (headOf d i) (chunkOf d i) 0 : Nat)
      = (0 + chunkOf d i * d.chunk + headOf d i * sDh d) by
    simp only [dIdx, sDh, Nat.add_zero]
    omega]
  exact t4

/-- `dA_cumsum_ptr`: `d.dA.ptr (dIdx d pid_h pid_c 0)`. -/
theorem dA_base {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {pb pc c1 s2 : I64} (hb : pb.toInt = 0) (hc : pc.toInt = (chunkOf d i : Int))
    (hc1 : c1.toInt = (d.chunk : Int)) (hs2 : s2.toInt = (headOf d i * sDh d : Nat)) :
    (d.dA.ptr 0 + (pb * (IntV.ofNat (sDb d) : I64) + pc * c1 + s2))
      = d.dA.ptr (dIdx d (headOf d i) (chunkOf d i) 0) := by
  have hsDh : sDh d = nc d * d.chunk := rfl
  have hck : chunkOf d i * d.chunk < 2 ^ 31 := PtrAux.chunkOf_mul_lt hd i
  have hs2b : headOf d i * sDh d < 2 ^ 47 := by
    rw [hsDh]
    exact PtrAux.mul_lt_47 (PtrAux.headOf_lt_16 hd i) hd.tok_lt
  have t1 : (pb * (IntV.ofNat (sDb d) : I64)).toInt = 0 := PtrAux.toInt_mul_zero_left hb
  have t2 : (pc * c1).toInt = ((chunkOf d i * d.chunk : Nat) : Int) :=
    IntV.I64.mul_nat hc hc1 (PtrAux.lt_63_of_lt_31 hck)
  have t12 : (pb * (IntV.ofNat (sDb d) : I64) + pc * c1).toInt
      = ((0 + chunkOf d i * d.chunk : Nat) : Int) :=
    IntV.I64.add_nat t1 t2 (by omega)
  have t4 : (pb * (IntV.ofNat (sDb d) : I64) + pc * c1 + s2).toInt
      = ((0 + chunkOf d i * d.chunk + headOf d i * sDh d : Nat) : Int) :=
    IntV.I64.add_nat t12 hs2 (by omega)
  apply PtrAux.ptr_add_eq_ptr
  rw [show (dIdx d (headOf d i) (chunkOf d i) 0 : Nat)
      = (0 + chunkOf d i * d.chunk + headOf d i * sDh d) by
    simp only [dIdx, sDh, Nat.add_zero]
    omega]
  exact t4

/-- `states_ptr`: `d.states.ptr (stIdx d pid_c pid_h 0 0)`. -/
theorem st_base {d : Inst} (hd : d.ok) (i : Fin (launch d).size)
    {pb pc s1 s2 : I64} (hb : pb.toInt = 0) (hc : pc.toInt = (chunkOf d i : Int))
    (hs1 : s1.toInt = (sSc d : Int)) (hs2 : s2.toInt = (headOf d i * sSh d : Nat)) :
    (d.states.ptr 0 + (pb * (IntV.ofNat (sSb d) : I64) + pc * s1 + s2))
      = d.states.ptr (stIdx d (chunkOf d i) (headOf d i) 0 0) := by
  have hnc : chunkOf d i < nc d := chunkOf_lt hd i
  have hh : headOf d i < d.nheads := headOf_lt hd i.isLt
  have hSb : sSb d < 2 ^ 63 := hd.batch_lt.2.2.1
  have hck : chunkOf d i * sSc d < 2 ^ 63 := by
    have h1 : chunkOf d i * sSc d < nc d * sSc d :=
      Nat.mul_lt_mul_of_pos_right hnc (sSc_pos hd)
    have h2 : nc d * sSc d = sSb d := (sSb_eq d).symm
    rw [h2] at h1
    exact lt_trans h1 hSb
  have hbIdx : (stIdx d (chunkOf d i) (headOf d i) 0 0 : Nat)
      = (0 + chunkOf d i * sSc d + headOf d i * sSh d) := by
    simp [stIdx, sSc, sSh]
  have hsum : 0 + chunkOf d i * sSc d + headOf d i * sSh d < 2 ^ 63 := by
    have h1 := stIdx_lt (d := d) (c := chunkOf d i) (h := headOf d i) (p := 0) (n := 0)
      hnc hh hd.headdim_pos hd.dstate_pos
    rw [hbIdx] at h1
    exact lt_trans h1 hSb
  have t1 : (pb * (IntV.ofNat (sSb d) : I64)).toInt = 0 := PtrAux.toInt_mul_zero_left hb
  have t2 : (pc * s1).toInt = ((chunkOf d i * sSc d : Nat) : Int) :=
    IntV.I64.mul_nat hc hs1 hck
  have t12 : (pb * (IntV.ofNat (sSb d) : I64) + pc * s1).toInt
      = ((0 + chunkOf d i * sSc d : Nat) : Int) :=
    IntV.I64.add_nat t1 t2 (by omega)
  have t4 : (pb * (IntV.ofNat (sSb d) : I64) + pc * s1 + s2).toInt
      = ((0 + chunkOf d i * sSc d + headOf d i * sSh d : Nat) : Int) :=
    IntV.I64.add_nat t12 hs2 hsum
  apply PtrAux.ptr_add_eq_ptr
  rw [hbIdx]
  exact t4


/-! ## St -/

/-!
# Address arithmetic for the `states` tile

The store at the end of the kernel writes the tile of `states` of program `i`.  Its pointer is
the tile's base pointer plus the flat lane offset `p · dstate + n`; these two lemmas turn that
into the element index `stIdx`.
-/

open TVA

set_option maxHeartbeats 800000

/-- `stIdx` splits as the tile's origin plus the flat `p · dstate + n` offset. -/
theorem stIdx_eq_base (d : Inst) (c h p n : Nat) :
    stIdx d c h p n = stIdx d c h 0 0 + p * d.dstate + n := by
  unfold stIdx
  ring

/-! ## Store -/

/-!
# The final store

The last line of the kernel writes the accumulator tile to `states`:

* the base pointer is `states_ptr + pid_b·stride_batch + pid_c·(chunk stride) + pid_h·(head
  stride)`, which (`PtrAux.st_base`) is the tile's first element;
* the lane `(a, n)` of the tile is `(pid_m·64 + a)·dstate + (pid_n·128 + n)` elements further, so
  it is the `states` element `stIdx d c h (pid_m·64+a) (pid_n·128+n)` (`St.stIdx_eq_base`);
* the mask enables exactly the lanes with `pid_m·64+a < headdim` and `pid_n·128+n < dstate`.

So the store is `safe_store_buf` of the tile of `states`, and what it leaves is `Q`.
-/

open TVA

set_option maxHeartbeats 800000

namespace StoreAux

variable {d : Inst}

/-- The element of `states` that lane `idx` of program `i`'s tile names. -/
def stG (d : Inst) (i : Nat) (idx : Idx [BM, BN]) : Nat :=
  stIdx d (chunkOf d i) (headOf d i) (pmOf d i * BM + (idx.1 : Nat))
    (pnOf d i * BN + (idx.2.1 : Nat))

/-- Whether lane `idx` of program `i`'s tile is inside `headdim × dstate`. -/
def stEn (d : Inst) (i : Nat) (idx : Idx [BM, BN]) : Bool :=
  decide (pmOf d i * BM + (idx.1 : Nat) < d.headdim) &&
    decide (pnOf d i * BN + (idx.2.1 : Nat) < d.dstate)

@[simp] theorem stG_apply (d : Inst) (i : Nat) (a : Fin BM) (n : Fin BN) (u : Unit) :
    stG d i (a, n, u) = stIdx d (chunkOf d i) (headOf d i) (pmOf d i * BM + (a : Nat))
      (pnOf d i * BN + (n : Nat)) := rfl

@[simp] theorem stEn_apply (d : Inst) (i : Nat) (a : Fin BM) (n : Fin BN) (u : Unit) :
    stEn d i (a, n, u) = (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
      decide (pnOf d i * BN + (n : Nat) < d.dstate)) := rfl

/-- A tile's offsets stay below `2^31` for `headdim` and `dstate`. -/
theorem stTileBounds (hd : d.ok) : d.headdim + 128 < 2 ^ 31 ∧ d.dstate + 256 < 2 ^ 31 := by
  have h1 := hd.sttile_lt
  have hp := hd.headdim_pos
  have hs := hd.dstate_pos
  have h2 : d.headdim + 63 ≤ (d.headdim + 63) * d.dstate := Nat.le_mul_of_pos_right (d.headdim + 63) hs
  have h3 : 64 * d.dstate ≤ (d.headdim + 63) * d.dstate := Nat.mul_le_mul_right _ (by omega)
  constructor <;> omega

/-- A row tile starts below `headdim + 64`. -/
theorem pmOf_mul_lt (hd : d.ok) (i : Nat) : pmOf d i * BM < d.headdim + BM := by
  have hgx := gx_pos hd
  have hq : qOf d i < gx d := by unfold qOf; exact Nat.mod_lt _ hgx
  have hn := npmOf_pos hd
  have h1 : pmOf d i * npmOf d ≤ qOf d i := by
    unfold pmOf; exact Nat.div_mul_le_self _ _
  rw [gx_eq] at hq
  have h2 : pmOf d i * npmOf d < (d.headdim + BM - 1) / BM * npmOf d := lt_of_le_of_lt h1 hq
  have h3 : pmOf d i < (d.headdim + BM - 1) / BM := Nat.lt_of_mul_lt_mul_right h2
  have h4 : (d.headdim + BM - 1) / BM * BM ≤ d.headdim + BM - 1 := Nat.div_mul_le_self _ _
  have h5 : pmOf d i * BM < (d.headdim + BM - 1) :=
    lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right h3 (by simp only [BM]; norm_num)) h4
  have hBM : BM = 64 := rfl
  omega

/-- A column tile starts below `dstate + 128`. -/
theorem pnOf_mul_lt (hd : d.ok) (i : Nat) : pnOf d i * BN < d.dstate + BN := by
  have hn := npmOf_pos hd
  have h1 : pnOf d i < npmOf d := by
    unfold pnOf; exact Nat.mod_lt _ hn
  have h2 : pnOf d i * BN < npmOf d * BN := Nat.mul_lt_mul_of_pos_right h1 (by simp only [BN]; norm_num)
  have h3 : npmOf d * BN ≤ d.dstate + BN - 1 := by
    unfold npmOf; exact Nat.div_mul_le_self _ _
  have hBN : BN = 128 := rfl
  omega

/-- A row of the tile is below `2^31`. -/
theorem pmOf_mul_add_lt (hd : d.ok) (i a : Nat) (ha : a < BM) : pmOf d i * BM + a < 2 ^ 31 := by
  have h1 := pmOf_mul_lt hd i
  have h2 := (stTileBounds hd).1
  have hBM : BM = 64 := rfl
  omega

/-- A column of the tile is below `2^31`. -/
theorem pnOf_mul_add_lt (hd : d.ok) (i n : Nat) (hn : n < BN) : pnOf d i * BN + n < 2 ^ 31 := by
  have h1 := pnOf_mul_lt hd i
  have h2 := (stTileBounds hd).2
  have hBN : BN = 128 := rfl
  omega

/-- The tile's row and column offsets sum below `2^31`. -/
theorem offs_add_lt (hd : d.ok) {p n : Nat} (hp : p < d.headdim) (hn : n < d.dstate) :
    p * d.dstate + n < 2 ^ 31 := by
  have h := hd.sttile_lt
  have hd1 := hd.dstate_pos
  have h2 : p * d.dstate < d.headdim * d.dstate := Nat.mul_lt_mul_of_pos_right hp hd1
  have h3 : d.headdim * d.dstate ≤ (d.headdim + 63) * d.dstate := Nat.mul_le_mul_right _ (by omega)
  omega

/-- Comparing casts to `Int` agrees with comparing the naturals. -/
theorem decide_cast_lt {X Y : Nat} : decide ((X : Int) < (Y : Int)) = decide (X < Y) := by
  have h : ((X : Int) < (Y : Int)) ↔ X < Y := Nat.cast_lt
  by_cases hh : X < Y
  · rw [decide_eq_true (h.mpr hh), decide_eq_true hh]
  · rw [decide_eq_false (fun hc => hh (h.mp hc)), decide_eq_false hh]

/-- The value of the row-offset vector at a lane. -/
theorem colVal (hd : d.ok) {i : Nat} {r5 : I32} (h5 : r5.toInt = (pmOf d i : Int)) (a : Fin BM) :
    (((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).get (a, ())).toInt)
      = ((pmOf d i * BM + (a : Nat) : Nat) : Int) := by
  have hP : pmOf d i * BM + (a : Nat) < 2 ^ 31 := pmOf_mul_add_lt hd i a a.isLt
  have h1 : (r5 * (IntV.ofNat BM : I32)).toInt = ((pmOf d i * BM : Nat) : Int) := by
    refine IntV.I32.mul_nat h5 (IntV.I32.toInt_ofNat ?_) ?_
    · simp only [BM]; norm_num
    · omega
  have h2 : ((tl.arange 0 BM).get (a, ())).toInt = ((a : Nat) : Int) := by
    have har : (tl.arange 0 BM).get (a, ()) = IntV.ofInt ((0 : Int) + (a : Nat)) := rfl
    rw [har, IntV.toInt_ofInt (by norm_num) (by
      simp only [IntV.SFits]
      have ha : (a : Nat) < 64 := by simpa [BM] using a.isLt
      exact ⟨by norm_num, by norm_num; omega⟩)]
    ring
  have hsplit : ((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).get (a, ()))
      = r5 * (IntV.ofNat BM : I32) + (tl.arange 0 BM).get (a, ()) := rfl
  rw [hsplit]
  exact IntV.I32.add_nat h1 h2 (by omega)

/-- The value of the column-offset vector at a lane. -/
theorem rowVal (hd : d.ok) {i : Nat} {r6 : I32} (h6 : r6.toInt = (pnOf d i : Int)) (n : Fin BN) :
    (((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).get (n, ())).toInt)
      = ((pnOf d i * BN + (n : Nat) : Nat) : Int) := by
  have hP : pnOf d i * BN + (n : Nat) < 2 ^ 31 := pnOf_mul_add_lt hd i n n.isLt
  have h1 : (r6 * (IntV.ofNat BN : I32)).toInt = ((pnOf d i * BN : Nat) : Int) := by
    refine IntV.I32.mul_nat h6 (IntV.I32.toInt_ofNat ?_) ?_
    · simp only [BN]; norm_num
    · omega
  have h2 : ((tl.arange 0 BN).get (n, ())).toInt = ((n : Nat) : Int) := by
    have har : (tl.arange 0 BN).get (n, ()) = IntV.ofInt ((0 : Int) + (n : Nat)) := rfl
    rw [har, IntV.toInt_ofInt (by norm_num) (by
      simp only [IntV.SFits]
      have hn : (n : Nat) < 128 := by simpa [BN] using n.isLt
      exact ⟨by norm_num, by norm_num; omega⟩)]
    ring
  have hsplit : ((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).get (n, ()))
      = r6 * (IntV.ofNat BN : I32) + (tl.arange 0 BN).get (n, ()) := rfl
  rw [hsplit]
  exact IntV.I32.add_nat h1 h2 (by omega)

/-- The value of the tile offset at an enabled lane. -/
theorem offsVal (hd : d.ok) {i : Nat} {r5 r6 : I32}
    (h5 : r5.toInt = (pmOf d i : Int)) (h6 : r6.toInt = (pnOf d i : Int))
    (a : Fin BM) (n : Fin BN)
    (hp : pmOf d i * BM + (a : Nat) < d.headdim)
    (hn : pnOf d i * BN + (n : Nat) < d.dstate) :
    ((((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN * (IntV.ofNat d.dstate : I32) +
       (r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM * (IntV.ofNat 1 : I32)).get2 a n).toInt)
      = (((pmOf d i * BM + (a : Nat)) * d.dstate + (pnOf d i * BN + (n : Nat)) : Nat) : Int) := by
  have hD : d.dstate < 2 ^ 31 := by have := (stTileBounds hd).2; omega
  have hX : ((((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN) *
      (IntV.ofNat d.dstate : I32)).get2 a n).toInt
      = (((pmOf d i * BM + (a : Nat)) * d.dstate : Nat) : Int) := by
    refine IntV.I32.mul_nat ?_ (IntV.I32.toInt_ofNat hD) ?_
    · simpa only [Tensor.get2, Tensor.get_col] using colVal hd h5 a
    · exact lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right hp hd.dstate_pos) (by
        have := hd.sttile_lt
        have := Nat.mul_le_mul_right d.dstate (show d.headdim ≤ d.headdim + 63 by omega)
        omega)
  have hY : ((((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM) *
      (IntV.ofNat 1 : I32)).get2 a n).toInt
      = ((pnOf d i * BN + (n : Nat) : Nat) : Int) := by
    show ((((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM).get2 a n) *
        (IntV.ofNat 1 : I32)).toInt = ((pnOf d i * BN + (n : Nat) : Nat) : Int)
    have hraw := IntV.I32.mul_nat
      (a := (((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM).get2 a n))
      (b := (IntV.ofNat 1 : I32))
      (x := (pnOf d i * BN + (n : Nat))) (y := 1)
      (by simpa only [Tensor.get2, Tensor.get_row] using rowVal hd h6 n)
      (IntV.I32.toInt_ofNat (by norm_num)) (by omega)
    simpa only [Nat.mul_one] using hraw
  have hsplit : ((((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN * (IntV.ofNat d.dstate : I32) +
       (r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM * (IntV.ofNat 1 : I32)).get2 a n))
      = ((((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN) *
          (IntV.ofNat d.dstate : I32)).get2 a n)
        + ((((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM) *
          (IntV.ofNat 1 : I32)).get2 a n) := rfl
  rw [hsplit]
  exact IntV.I32.add_nat hX hY (by have := offs_add_lt hd hp hn; omega)

/-- The store mask at a lane. -/
theorem maskVal (hd : d.ok) {i : Nat} {r5 r6 : I32}
    (h5 : r5.toInt = (pmOf d i : Int)) (h6 : r6.toInt = (pnOf d i : Int))
    (a : Fin BM) (n : Fin BN) :
    (((((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN).ltS
        ((IntV.ofNat d.headdim : I32))).and
      (((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM).ltS
        ((IntV.ofNat d.dstate : I32)))).get2 a n)
    = (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
       decide (pnOf d i * BN + (n : Nat) < d.dstate)) := by
  have hH : d.headdim < 2 ^ 31 := by have := (stTileBounds hd).1; omega
  have hD : d.dstate < 2 ^ 31 := by have := (stTileBounds hd).2; omega
  show (IntV.lt ((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).get (a, ()))
          (IntV.ofNat d.headdim : I32) &&
        IntV.lt ((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).get (n, ()))
          (IntV.ofNat d.dstate : I32))
      = (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
         decide (pnOf d i * BN + (n : Nat) < d.dstate))
  rw [IntV.lt_eq, IntV.lt_eq, colVal hd h5 a, IntV.I32.toInt_ofNat hH, rowVal hd h6 n,
    IntV.I32.toInt_ofNat hD, decide_cast_lt, decide_cast_lt]

/-- An enabled lane of the tile is the `states` pointer of its element. -/
theorem ptrEq {d : Inst} {i : Fin (launch d).size} {s1 s2 pb pc : I64} {r5 r6 : I32}
    (hd : d.ok) (hs1 : s1.toInt = (sSc d : Int)) (hs2 : s2.toInt = (headOf d i * sSh d : Nat))
    (hpb : pb.toInt = 0) (hc : pc.toInt = (chunkOf d i : Int))
    (h5 : r5.toInt = (pmOf d i : Int)) (h6 : r6.toInt = (pnOf d i : Int))
    (idx : Idx [BM, BN]) (hen : stEn d i idx = true) :
    ((d.states.ptr 0 + (pb * (IntV.ofNat (sSb d) : I64) + pc * s1 + s2) +
        ((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN * (IntV.ofNat d.dstate : I32) +
          (r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM * (IntV.ofNat 1 : I32))).get idx)
      = d.states.ptr (stG d i idx : Int) := by
  rcases idx with ⟨a, n, u⟩
  cases u
  have ha : (a : Nat) < BM := a.isLt
  have hn0 : (n : Nat) < BN := n.isLt
  rw [stEn_apply] at hen
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
  obtain ⟨hp, hn⟩ := hen
  have hbase : (d.states.ptr 0 + (pb * (IntV.ofNat (sSb d) : I64) + pc * s1 + s2))
      = d.states.ptr (stIdx d (chunkOf d i) (headOf d i) 0 0) :=
    st_base hd i hpb hc hs1 hs2
  rw [hbase]
  show ((d.states.ptr (stIdx d (chunkOf d i) (headOf d i) 0 0)).offset
      (((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN * (IntV.ofNat d.dstate : I32) +
        (r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM * (IntV.ofNat 1 : I32)).get2 a n).toInt)
      = d.states.ptr (stG d i (a, n, ()) : Int)
  rw [Buf.ptr_offset, stG_apply]
  have hK := offsVal hd h5 h6 a n hp hn
  rw [hK]
  have hS : (stIdx d (chunkOf d i) (headOf d i) (pmOf d i * BM + (a : Nat))
        (pnOf d i * BN + (n : Nat)) : Int)
      = (stIdx d (chunkOf d i) (headOf d i) 0 0 : Int)
        + (((pmOf d i * BM + (a : Nat)) * d.dstate + (pnOf d i * BN + (n : Nat)) : Nat) : Int) := by
    rw [stIdx_eq_base]
    push_cast
    ring
  rw [hS]

end StoreAux

/-- **The store of the accumulator tile.** -/
theorem store_states {E : Env} {d : Inst} {init : Mem} {i : Fin (launch d).size}
    (hd : d.ok) (hvs : d.states.Valid init)
    {s1 s2 pb pc : I64} {r5 r6 : I32}
    (hs1 : s1.toInt = (sSc d : Int)) (hs2 : s2.toInt = (headOf d i * sSh d : Nat))
    (hpb : pb.toInt = 0) (hc : pc.toInt = (chunkOf d i : Int))
    (h5 : r5.toInt = (pmOf d i : Int)) (h6 : r6.toInt = (pnOf d i : Int))
    (acc : Tensor [BM, BN] (DType.f32.Val))
    (hfin : Vals init d → ∀ (a : Fin BM) (n : Fin BN),
      pmOf d i * BM + (a : Nat) < d.headdim → pnOf d i * BN + (n : Nat) < d.dstate →
      (acc.get2 a n).isFinite = true)
    (hval : Vals init d → ∀ (a : Fin BM) (n : Fin BN),
      pmOf d i * BM + (a : Nat) < d.headdim → pnOf d i * BN + (n : Nat) < d.dstate →
      |(acc.get2 a n).toReal -
          sRef init d (chunkOf d i) (headOf d i) (pmOf d i * BM + (a : Nat))
            (pnOf d i * BN + (n : Nat))| ≤
        tol init d (chunkOf d i) (headOf d i) (pmOf d i * BM + (a : Nat))
          (pnOf d i * BN + (n : Nat))) :
    Safe E (R d init i) (W d init i) (fun _ m => Q d init i m) init
      (tl.store
        (d.states.ptr 0 +
            (pb * (IntV.ofNat (sSb d) : I64) + pc * s1 + s2) +
          ((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN * (IntV.ofNat d.dstate : I32) +
            (r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM * (IntV.ofNat 1 : I32)))
        acc
        (some
          ((((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BN).ltS ((IntV.ofNat d.headdim : I32))).and
            (((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BM).ltS ((IntV.ofNat d.dstate : I32)))))) := by
  refine safe_last ?_
  refine safe_store_buf (b := d.states) (g := StoreAux.stG d i) (en := StoreAux.stEn d i)
    (k := pure) ?_ ?_ ?_ hvs ?_ ?_
  · refine Buf.addrs_of_get d.states ?_ ?_
    · intro idx
      rcases idx with ⟨a, n, u⟩
      cases u
      exact StoreAux.maskVal hd h5 h6 a n
    · intro idx hen
      exact StoreAux.ptrEq hd hs1 hs2 hpb hc h5 h6 idx hen
  · intro idx hen
    rcases idx with ⟨a, n, u⟩
    cases u
    rw [StoreAux.stEn_apply] at hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    obtain ⟨hp, hn⟩ := hen
    have h1 : StoreAux.stG d i (a, n, ()) < sSb d := by
      rw [StoreAux.stG_apply]
      exact stIdx_lt (chunkOf_lt hd i) (headOf_lt hd i.isLt) hp hn
    exact lt_of_lt_of_le h1 hd.states_len
  · intro idx idx' hen hen' heq
    rcases idx with ⟨a, n, u⟩
    cases u
    rcases idx' with ⟨a', n', u'⟩
    cases u'
    rw [StoreAux.stEn_apply] at hen hen'
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen hen'
    obtain ⟨hp, hn⟩ := hen
    obtain ⟨hp', hn'⟩ := hen'
    rw [StoreAux.stG_apply] at heq
    obtain ⟨_, _, hpe, hne⟩ :=
      stIdx_inj hd (chunkOf_lt hd i) (chunkOf_lt hd i) (headOf_lt hd i.isLt)
        (headOf_lt hd i.isLt) hp hp' hn hn' heq
    have hae : a = a' := Fin.ext (by omega)
    have hne' : n = n' := Fin.ext (by omega)
    rw [hae, hne']
  · refine Buf.elems_mono d.states ?_
    rintro j ⟨idx, hen, rfl⟩
    rcases idx with ⟨a, n, u⟩
    cases u
    have ha : (a : Nat) < BM := a.isLt
    have hn0 : (n : Nat) < BN := n.isLt
    rw [StoreAux.stEn_apply] at hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    obtain ⟨hp, hn⟩ := hen
    have hs1' : (pmOf d i + 1) * BM = pmOf d i * BM + BM := by
      rw [Nat.add_mul, Nat.one_mul]
    have hs2' : (pnOf d i + 1) * BN = pnOf d i * BN + BN := by
      rw [Nat.add_mul, Nat.one_mul]
    simp only [stSet, pSet, nSet, Set.mem_ofPred_eq]
    rw [hs1', hs2']
    exact ⟨pmOf d i * BM + (a : Nat), ⟨by omega, by omega, hp⟩,
      pnOf d i * BN + (n : Nat), ⟨by omega, by omega, hn⟩, by rw [StoreAux.stG_apply]⟩
  · intro m' hread _ _
    refine safe_ret ?_
    intro p n hp hn
    obtain ⟨hp1, hp2, hp3⟩ := hp
    obtain ⟨hn1, hn2, hn3⟩ := hn
    have hp2' : p < pmOf d i * BM + BM := by
      rw [Nat.add_mul, Nat.one_mul] at hp2
      exact hp2
    have hn2' : n < pnOf d i * BN + BN := by
      rw [Nat.add_mul, Nat.one_mul] at hn2
      exact hn2
    set a : Fin BM := ⟨p - pmOf d i * BM, by omega⟩ with ha
    set nn : Fin BN := ⟨n - pnOf d i * BN, by omega⟩ with hna
    have hpa : pmOf d i * BM + (a : Nat) = p := by
      rw [ha]; show pmOf d i * BM + (p - pmOf d i * BM) = p; omega
    have hnab : pnOf d i * BN + (nn : Nat) = n := by
      rw [hna]; show pnOf d i * BN + (n - pnOf d i * BN) = n; omega
    have hen : StoreAux.stEn d i (a, nn, ()) = true := by
      rw [StoreAux.stEn_apply, hpa, hnab]
      simp only [Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨hp3, hn3⟩
    have hg : StoreAux.stG d i (a, nn, ()) = stIdx d (chunkOf d i) (headOf d i) p n := by
      rw [StoreAux.stG_apply, hpa, hnab]
    have hr := hread (a, nn, ()) hen
    rw [hg] at hr
    refine ⟨acc.get2 a nn, ?_, ?_⟩
    · show m'.read .f32 (d.states.ptr (stIdx d (chunkOf d i) (headOf d i) p n)).addr
        = some (acc.get2 a nn)
      exact hr
    · intro hV
      have hb1 : pmOf d i * BM + (a : Nat) < d.headdim := by rw [hpa]; exact hp3
      have hb2 : pnOf d i * BN + (nn : Nat) < d.dstate := by rw [hnab]; exact hn3
      refine ⟨hfin hV a nn hb1 hb2, ?_⟩
      have hv := hval hV a nn hb1 hb2
      rwa [hpa, hnab] at hv


/-! ## LMask -/

/-!
# The loop's load masks, lane by lane

The three loads of the accumulation loop are masked by comparisons of the tile offsets with the
loop bound; this file evaluates those masks at a lane.
-/

open TVA

set_option maxHeartbeats 800000

/-- The `x` load's mask at lane `(a, j)`. -/
theorem xMaskEq {d : Inst} (hd : d.ok) {i : Nat} {r5 : I32} (h5 : r5.toInt = (pmOf d i : Int))
    {t : Nat} {HI K0 : I64} {offs : Tensor [BK] I64} {Nch : Nat}
    (hNK : (HI - K0).toInt = ((Nch - t * BK : Nat) : Int))
    (hOff : ∀ j : Fin BK, (offs.get (j, ())).toInt = ((j : Nat) : Int))
    (a : Fin BM) (j : Fin BK) :
    ((((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).col BK).ltS (IntV.ofNat d.headdim)).and
        ((offs.ltS (HI - K0)).row BM)).get2 a j
      = (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
          decide ((j : Nat) < Nch - t * BK)) := by
  have hH : d.headdim < 2 ^ 31 := by have := (StoreAux.stTileBounds hd).1; omega
  show (IntV.lt ((r5 * (IntV.ofNat BM : I32) + tl.arange 0 BM).get (a, ()))
          (IntV.ofNat d.headdim : I32) &&
        IntV.lt ((offs).get (j, ())) (HI - K0))
      = (decide (pmOf d i * BM + (a : Nat) < d.headdim) &&
          decide ((j : Nat) < Nch - t * BK))
  rw [IntV.lt_eq, IntV.lt_eq, StoreAux.colVal hd h5 a, IntV.I32.toInt_ofNat hH, hNK, hOff j,
    StoreAux.decide_cast_lt, StoreAux.decide_cast_lt]

/-- The `b` load's mask at lane `(j, n)`. -/
theorem bMaskEq {d : Inst} (hd : d.ok) {i : Nat} {r6 : I32} (h6 : r6.toInt = (pnOf d i : Int))
    {t : Nat} {HI K0 : I64} {offs : Tensor [BK] I64} {Nch : Nat}
    (hNK : (HI - K0).toInt = ((Nch - t * BK : Nat) : Int))
    (hOff : ∀ j : Fin BK, (offs.get (j, ())).toInt = ((j : Nat) : Int))
    (j : Fin BK) (n : Fin BN) :
    (((offs.ltS (HI - K0)).col BN).and
        (((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).row BK).ltS
          (IntV.ofNat d.dstate))).get2 j n
      = (decide ((j : Nat) < Nch - t * BK) &&
          decide (pnOf d i * BN + (n : Nat) < d.dstate)) := by
  have hD : d.dstate < 2 ^ 31 := by have := (StoreAux.stTileBounds hd).2; omega
  show (IntV.lt ((offs).get (j, ())) (HI - K0) &&
        IntV.lt ((r6 * (IntV.ofNat BN : I32) + tl.arange 0 BN).get (n, ()))
          (IntV.ofNat d.dstate : I32))
      = (decide ((j : Nat) < Nch - t * BK) &&
          decide (pnOf d i * BN + (n : Nat) < d.dstate))
  rw [IntV.lt_eq, IntV.lt_eq, hNK, hOff j, StoreAux.rowVal hd h6 n, IntV.I32.toInt_ofNat hD,
    StoreAux.decide_cast_lt, StoreAux.decide_cast_lt]

/-- The scalar loads' mask (over the token axis) at lane `j`. -/
theorem offsMaskEq {t : Nat} {HI K0 : I64} {offs : Tensor [BK] I64} {Nch : Nat}
    (hNK : (HI - K0).toInt = ((Nch - t * BK : Nat) : Int))
    (hOff : ∀ j : Fin BK, (offs.get (j, ())).toInt = ((j : Nat) : Int))
    (j : Fin BK) :
    (offs.ltS (HI - K0)).get1 j = decide ((j : Nat) < Nch - t * BK) := by
  show IntV.lt ((offs).get (j, ())) (HI - K0) = decide ((j : Nat) < Nch - t * BK)
  rw [IntV.lt_eq, hNK, hOff j, StoreAux.decide_cast_lt]


end Claims.SsdChunkState
