import Proof.Seg
import Proof.Foot

/-!
# `chunk_delta_h_bf16`: facts used by the walk of a loop trip

The program's integer lanes, the validity of the buffers in the program's memory, and the
footprints of the stores of chunk `t`.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- A buffer valid initially is valid in a memory that keeps every initially present byte. -/
theorem valid_of_memOk {dt : DType} {b : Buf dt} {d : Inst} {init m : Mem} (hv : b.Valid init)
    (hm : MemOk d init m) : b.Valid m :=
  Buf.valid_of_agree hv fun x hx => hm.2 x (hv.2.2.2 x hx)

/-- Lane `i` of `tl.convert(tl.arange(0, BT), int64)` is `i`. -/
theorem ot_lane {P : Platform} {ot : Tensor [BT - 0] I64}
    (h : ∀ i, ConvRel P .i32 .i64 ((tl.arange 0 BT).get i) (ot.get i)) (i : Fin BT) :
    (ot.get1 i).toInt = (i : Nat) := by
  have h1 := ConvRel.toInt_i32_i64 (h (i, ()))
  have := IntV.I32.toInt_tl_arange (n := BT) (by unfold BT; norm_num) i
  exact h1.trans this

/-- The `int64` row offsets of chunk `t`: lane `i` is `64t + i`. -/
theorem rows_lane {d : Inst} (hd : d.ok) {n t : Nat} (hn : n < d.N) (ht : t < nch d n)
    {iT : I64} (hiT : iT.toInt = (t : Int)) {ot : Tensor [BT - 0] I64}
    (hot : ∀ i : Fin BT, (ot.get1 i).toInt = (i : Nat)) (i : Fin BT) :
    ((iT * IntV.ofNat BT + ot).get1 i).toInt = ((BT * t + i : Nat) : Int) := by
  have hT := hd.T_ctx
  have hlen : lenOf d n ≤ d.T := by have := start_add_len_le hd hn; omega
  have htb : BT * t < 2 ^ 22 := by simp only [nch, BT] at ht ⊢; omega
  have h1 : (iT * IntV.ofNat BT : I64).toInt = ((t * BT : Nat) : Int) :=
    IntV.mul_nat (x := t) (y := BT) hiT (IntV.I64.toInt_ofNat (by unfold BT; norm_num))
      (by unfold BT at *; omega)
  have h2 := IntV.add_nat (a := iT * IntV.ofNat BT) (b := ot.get1 i) h1 (hot i)
    (by have := i.isLt; unfold BT at *; omega)
  have hg : (iT * IntV.ofNat BT + ot).get1 i = iT * IntV.ofNat BT + ot.get1 i := by simp
  rw [hg, h2]
  congr 1
  ring

/-- The `int64` scalar `t · HV · K · V` of the `h` pointer. -/
theorem sc_lane {d : Inst} (hd : d.ok) {n t : Nat} (hn : n < d.N) (ht : t < nch d n)
    {iT : I64} (hiT : iT.toInt = (t : Int)) :
    (iT * IntV.ofNat d.HV * IntV.ofNat d.K * IntV.ofNat d.V : I64).toInt =
      ((t * d.HV * d.K * d.V : Nat) : Int) := by
  have hT := hd.T_ctx
  have hlen : lenOf d n ≤ d.T := by have := start_add_len_le hd hn; omega
  have htb : t < 2 ^ 16 := by simp only [nch, BT] at ht; omega
  have hHV := hd.HV_lt; have hK := hd.K_le; have hV := hd.V_lt
  have hHVi : (IntV.ofNat d.HV : I64).toInt = (d.HV : Int) := IntV.I64.toInt_ofNat (by omega)
  have hKi : (IntV.ofNat d.K : I64).toInt = (d.K : Int) := IntV.I64.toInt_ofNat (by omega)
  have hVi : (IntV.ofNat d.V : I64).toInt = (d.V : Int) := IntV.I64.toInt_ofNat (by omega)
  have h1 : (iT * IntV.ofNat d.HV : I64).toInt = ((t * d.HV : Nat) : Int) :=
    IntV.mul_nat hiT hHVi (by
      have := Nat.mul_le_mul htb.le hHV.le
      exact lt_of_le_of_lt this (by norm_num))
  have h2 : (iT * IntV.ofNat d.HV * IntV.ofNat d.K : I64).toInt = ((t * d.HV * d.K : Nat) : Int) :=
    IntV.mul_nat h1 hKi (by
      have := Nat.mul_le_mul (Nat.mul_le_mul htb.le hHV.le) hK
      exact lt_of_le_of_lt this (by norm_num))
  exact IntV.mul_nat h2 hVi (by
    have := Nat.mul_le_mul (Nat.mul_le_mul (Nat.mul_le_mul htb.le hHV.le) hK) hV.le
    exact lt_of_le_of_lt this (by norm_num))

/-- The chunk `t` of sequence `n` is inside `h`. -/
theorem hlen_h {d : Inst} (hd : d.ok) {n t : Nat} (hn : n < d.N) (ht : t < nch d n) :
    (cbase d n + t + 1) * d.HV * d.K * d.V ≤ d.h.len := by
  have h1 := chunk_lt hd hn ht
  have h2 := hd.h_len
  have h3 : (cbase d n + t + 1) * d.HV * d.K * d.V ≤ d.B * NTt d * d.HV * d.K * d.V := by
    apply Nat.mul_le_mul_right; apply Nat.mul_le_mul_right; apply Nat.mul_le_mul_right; omega
  omega

/-- The stores of the blocks of one chunk are inside the chunk's cells. -/
theorem hBlk_sub_Wt {d : Inst} {n hv jv t base0 : Nat} :
    d.h.elems (hBlk d n hv jv t base0) ⊆ Wt d n hv jv t :=
  fun _ hx => Or.inl (d.h.elems_mono hBlk_sub_hCells hx)

/-- The blocks of states of one chunk (different blocks of `64` rows) are disjoint. -/
theorem hBlk_disj {d : Inst} {n hv jv t b0 b0' : Nat} (hhv : hv < d.HV) (hb : b0 + 64 ≤ b0') :
    Disjoint (d.h.elems (hBlk d n hv jv t b0)) (d.h.elems (hBlk d n hv jv t b0')) := by
  refine d.h.elems_disjoint ?_
  rw [Set.disjoint_left]
  rintro x ⟨a, jj, ha, haK, hjj, hin, rfl⟩ ⟨a', jj', ha', haK', hjj', hin', h⟩
  have := (hIdx_inj hhv hhv haK haK' hin.2.2 hin'.2.2 h).2.2.1
  omega

/-- The `v_new` tile of chunk `t` is inside the chunk's cells. -/
theorem vCells_sub_Wt {d : Inst} {n hv jv t : Nat} :
    d.vn.elems (vCells d n hv jv t) ⊆ Wt d n hv jv t :=
  fun _ hx => Or.inr hx

/-- A state block's tile and the `v_new` tile of one chunk are disjoint. -/
theorem hBlk_vn_disj {d : Inst} (hd : d.ok) {n hv jv t b0 : Nat} (hn : n < d.N) (hhv : hv < d.HV)
    (ht : t < nch d n) :
    Disjoint (d.h.elems (hBlk d n hv jv t b0)) (d.vn.elems (vCells d n hv jv t)) :=
  hvn_disjoint hd (fun y hy => hCells_lt hd hn hhv ht y (hBlk_sub_hCells hy))
    (vCells_lt hd hn hhv)

/-- The inputs `k`, `w`, `u`, `g` lie in the read footprint. -/
theorem inputs_Rset {d : Inst} (hd : d.ok) :
    d.k.bytes ⊆ Rset d ∧ d.w.bytes ⊆ Rset d ∧ d.u.bytes ⊆ Rset d ∧ d.g.bytes ⊆ Rset d :=
  ⟨bytes_sub_Rset d.k hd.disj_h.1.symm hd.disj_vn.1.symm hd.disj_ht.1.symm,
   bytes_sub_Rset d.w hd.disj_h.2.1.symm hd.disj_vn.2.1.symm hd.disj_ht.2.1.symm,
   bytes_sub_Rset d.u hd.disj_h.2.2.1.symm hd.disj_vn.2.2.1.symm hd.disj_ht.2.2.1.symm,
   bytes_sub_Rset d.g hd.disj_h.2.2.2.1.symm hd.disj_vn.2.2.2.1.symm hd.disj_ht.2.2.2.1.symm⟩

/-- A sequence's tokens end inside the `B · T` tokens. -/
theorem bos_le {d : Inst} (hd : d.ok) {n : Nat} (hn : n < d.N) : bosOf d n + lenOf d n ≤ d.B * d.T := by
  have h1 := row_lt hd hn
  have h2 := start_add_len_le hd hn
  have h3 : (rowOf d n + 1) * d.T ≤ d.B * d.T := Nat.mul_le_mul_right _ h1
  have h4 : (rowOf d n + 1) * d.T = rowOf d n * d.T + d.T := by ring
  unfold bosOf
  omega

/-- Sequence `n`'s `w` elements are inside the tensor. -/
theorem len_w {d : Inst} (hd : d.ok) {n : Nat} (hn : n < d.N) :
    (bosOf d n + lenOf d n) * d.HV * d.K ≤ d.w.len :=
  le_trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (bos_le hd hn))) hd.w_len

/-- Sequence `n`'s `k` elements are inside the tensor. -/
theorem len_k {d : Inst} (hd : d.ok) {n : Nat} (hn : n < d.N) :
    (bosOf d n + lenOf d n) * d.H * d.K ≤ d.k.len :=
  le_trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (bos_le hd hn))) hd.k_len

/-- Sequence `n`'s `u` elements are inside the tensor. -/
theorem len_u {d : Inst} (hd : d.ok) {n : Nat} (hn : n < d.N) :
    (bosOf d n + lenOf d n) * d.HV * d.V ≤ d.u.len :=
  le_trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (bos_le hd hn))) hd.u_len

/-- Sequence `n`'s `v_new` elements are inside the tensor. -/
theorem len_vn {d : Inst} (hd : d.ok) {n : Nat} (hn : n < d.N) :
    (bosOf d n + lenOf d n) * d.HV * d.V ≤ d.vn.len :=
  le_trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (bos_le hd hn))) hd.vn_len

/-- Sequence `n`'s `g` elements are inside the tensor. -/
theorem len_g {d : Inst} (hd : d.ok) (hg : d.useG = true) {n : Nat} (hn : n < d.N) :
    (bosOf d n + lenOf d n) * d.HV ≤ d.g.len :=
  le_trans (Nat.mul_le_mul_right _ (bos_le hd hn)) (hd.g_len hg)

/-- A write inside the outputs, with the frame and presence facts, keeps `MemOk`. -/
theorem memOk_step {d : Inst} {init m m' : Mem} {S : Set Int} (hm : MemOk d init m)
    (hS : ∀ y ∈ S, y ∉ Rset d) (hf : ∀ y, y ∉ S → m' y = m y)
    (hp : ∀ y, m y ≠ none → m' y ≠ none) : MemOk d init m' :=
  ⟨fun x hx => (hf x fun hxS => hS x hxS hx).trans (hm.1 x hx), fun x hx => hp x (hm.2 x hx)⟩

/-- The tiles of `h` (of chunk `t`) are not in the read footprint. -/
theorem hBlk_not_Rset {d : Inst} (hd : d.ok) {n hv jv t b0 : Nat} (hn : n < d.N) (hhv : hv < d.HV)
    (ht : t < nch d n) : ∀ y ∈ d.h.elems (hBlk d n hv jv t b0), y ∉ Rset d := by
  intro y hy hR
  have hsub := d.h.elems_sub_bytes (S := hBlk d n hv jv t b0)
    (fun j hj => lt_of_lt_of_le (hCells_lt hd hn hhv ht j (hBlk_sub_hCells hj)) hd.h_len) hy
  exact hR.1 hsub

/-- The `v_new` tile of chunk `t` is not in the read footprint. -/
theorem vCells_not_Rset {d : Inst} (hd : d.ok) {n hv jv t : Nat} (hn : n < d.N) (hhv : hv < d.HV) :
    ∀ y ∈ d.vn.elems (vCells d n hv jv t), y ∉ Rset d := by
  intro y hy hR
  have hsub := d.vn.elems_sub_bytes (S := vCells d n hv jv t)
    (fun j hj => lt_of_lt_of_le (vCells_lt hd hn hhv j hj) hd.vn_len) hy
  exact hR.2.1 hsub

end Claims.ChunkDeltaH
