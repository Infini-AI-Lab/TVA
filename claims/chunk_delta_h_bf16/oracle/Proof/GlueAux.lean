import Proof.Seg
import Proof.TripAux

/-!
# `chunk_delta_h_bf16`: the inputs' values at the cells a program reads

`Vals` says every input element is finite and bounded; here it is read at the element of a token
of a sequence, and the values `kq`, `wq`, `uq`, `gq` are those of the elements read.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The cells of `w` are inside its tensor. -/
theorem wIdx_lt {d : Inst} {b t hv c : Nat} (hb : b < d.B) (ht : t < d.T) (hhv : hv < d.HV)
    (hc : c < d.K) : wIdx d b t hv c < d.B * d.T * d.HV * d.K :=
  idx4_lt hb ht hhv hc

/-- The cells of `k` are inside its tensor. -/
theorem kIdx_lt {d : Inst} {b t hk c : Nat} (hb : b < d.B) (ht : t < d.T) (hhk : hk < d.H)
    (hc : c < d.K) : kIdx d b t hk c < d.B * d.T * d.H * d.K :=
  idx4_lt hb ht hhk hc

/-- The cells of `g` are inside its tensor. -/
theorem gIdx_lt {d : Inst} {b t hv : Nat} (hb : b < d.B) (ht : t < d.T) (hhv : hv < d.HV) :
    gIdx d b t hv < d.B * d.T * d.HV := by
  unfold gIdx
  exact pair_lt (by
    have : (b + 1) * d.T ≤ d.B * d.T := Nat.mul_le_mul_right _ hb
    have : (b + 1) * d.T = b * d.T + d.T := by ring
    omega) hhv

/-- The key head of a value head is a head. -/
theorem kHead_lt {d : Inst} (hd : d.ok) {hv : Nat} (hhv : hv < d.HV) : kHead d hv < d.H := by
  have hHle : d.H ≤ d.HV := Nat.le_of_dvd hd.HV_pos hd.H_dvd
  unfold kHead
  have hdv : 0 < d.HV / d.H := Nat.div_pos hHle hd.H_pos
  rw [Nat.div_lt_iff_lt_mul hdv]
  obtain ⟨c, hc⟩ := hd.H_dvd
  have : d.HV / d.H = c := by rw [hc]; exact Nat.mul_div_cancel_left _ hd.H_pos
  rw [this]; rw [hc] at hhv; linarith

/-- The value of an element that was read. -/
theorem valB_of_rd {m : Mem} {b : Buf .bf16} {i : Nat} {x : BF16} (h : rdB m b i = some x) :
    valB m b i = x.toRat := by
  simp [valB, h]

/-- The value of a `float32` element that was read. -/
theorem valF_of_rd {m : Mem} {b : Buf .f32} {i : Nat} {x : F32} (h : rdF m b i = some x) :
    valF m b i = x.toRat := by
  simp [valF, h]

/-- `Vals` at a token of a sequence: the `w` element is finite and at most `2^16`. -/
theorem vals_w {d : Inst} (hd : d.ok) {init : Mem} (hV : Vals init d) {n hv s c : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) (hs : s < lenOf d n) (hc : c < d.K) :
    ∃ x, rdB init d.w (wIdx d (rowOf d n) (startOf d n + s) hv c) = some x ∧ x.isFinite = true ∧
      |x.toRat| ≤ 2 ^ 16 :=
  hV.2.1 _ (wIdx_lt (row_lt hd hn) (tok_lt hd hn hs) hhv hc)

/-- `Vals` at a token of a sequence: the `u` element is finite and at most `2^16`. -/
theorem vals_u {d : Inst} (hd : d.ok) {init : Mem} (hV : Vals init d) {n hv s j : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) (hs : s < lenOf d n) (hj : j < d.V) :
    ∃ x, rdB init d.u (uIdx d (rowOf d n) (startOf d n + s) hv j) = some x ∧ x.isFinite = true ∧
      |x.toRat| ≤ 2 ^ 16 :=
  hV.2.2.1 _ (uIdx_lt (row_lt hd hn) (tok_lt hd hn hs) hhv hj)

/-- `Vals` at a token of a sequence: the `k` element is finite and at most `2^16`. -/
theorem vals_k {d : Inst} (hd : d.ok) {init : Mem} (hV : Vals init d) {n hv s c : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) (hs : s < lenOf d n) (hc : c < d.K) :
    ∃ x, rdB init d.k (kIdx d (rowOf d n) (startOf d n + s) (kHead d hv) c) = some x ∧
      x.isFinite = true ∧ |x.toRat| ≤ 2 ^ 16 :=
  hV.1 _ (kIdx_lt (row_lt hd hn) (tok_lt hd hn hs) (kHead_lt hd hhv) hc)

/-- When the gate is used, its entries read from the initial memory are finite with absolute value at most `2 ^ 24`. -/
theorem vals_g {d : Inst} (hd : d.ok) {init : Mem} (hV : Vals init d) (hG : d.useG = true)
    {n hv s : Nat} (hn : n < d.N) (hhv : hv < d.HV) (hs : s < lenOf d n) :
    ∃ x, rdF init d.g (gTok d n hv s) = some x ∧ x.isFinite = true ∧ |x.toRat| ≤ 2 ^ 24 :=
  hV.2.2.2.1 hG _ (gIdx_lt (row_lt hd hn) (tok_lt hd hn hs) hhv)

end Claims.ChunkDeltaH
