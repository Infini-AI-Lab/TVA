import Proof.Lanes

/-!
# `attn_bwd_f16`: the scalars of the kernel

The `int32` and `int64` values of the head offsets (lines 383–397) do not wrap.
-/

namespace Claims.AttnBwd

open TVA

/-- A program of the grid has a head below `BATCH · N_HEAD`. -/
theorem head_lt {d : Inst} {i : Nat} (hi : i < gx d * 1 * G d) : head d i < G d := by
  unfold head
  by_cases hgx : gx d = 0
  · rw [hgx] at hi; simp at hi
  · exact Nat.div_lt_of_lt_mul (by simpa [Nat.mul_comm] using hi)

/-- A program of the grid has a block of rows inside the grid's axis 0. -/
theorem pidX_lt {d : Inst} {i : Nat} (hi : i < gx d * 1 * G d) : pidX d i < gx d := by
  unfold pidX
  apply Nat.mod_lt
  by_cases hgx : gx d = 0
  · rw [hgx] at hi; simp at hi
  · omega

/-- `128` rows per block, `gx` blocks: `N_CTX`. -/
theorem mul_gx {d : Inst} (hd : d.ok) : 128 * gx d = d.N_CTX := by
  have := hd.n_mod
  unfold gx
  omega

/-- The offset `adj` of line 385: `stride_h·(b % H) + stride_z·(b // H) = b·N_CTX·HEAD_DIM`. -/
theorem adj_val {d : Inst} (hd : d.ok) {b : Nat} (hb : b < G d) (hH : 0 < d.H)
    {l0 l1 : Tensor [] I32} (hl0 : l0.item.toInt = (b : Int) % (d.H : Int))
    (hl1 : l1.item.toInt = (b : Int) / (d.H : Int)) :
    (IntV.cast (IntV.ofNat (d.N_CTX * d.HEAD_DIM) * l0.item +
      IntV.ofNat (d.H * d.N_CTX * d.HEAD_DIM) * l1.item) : I64).toInt =
      ((b * d.N_CTX * d.HEAD_DIM : Nat) : Int) := by
  obtain ⟨hs1, hs2⟩ := hd.size_lt
  have hl0' : l0.item.toInt = ((b % d.H : Nat) : Int) := by rw [hl0]; push_cast; rfl
  have hl1' : l1.item.toInt = ((b / d.H : Nat) : Int) := by rw [hl1]; push_cast; rfl
  have hbd : d.H * (b / d.H) + b % d.H = b := Nat.div_add_mod b d.H
  have hmod : b % d.H < d.H := Nat.mod_lt _ hH
  have hNH : 0 < d.N_CTX * d.HEAD_DIM ∨ d.N_CTX * d.HEAD_DIM = 0 := by omega
  have hP1 : d.N_CTX * d.HEAD_DIM * (b % d.H) < 2 ^ 31 := by
    have : d.N_CTX * d.HEAD_DIM * (b % d.H) ≤ d.N_CTX * d.HEAD_DIM * d.H :=
      Nat.mul_le_mul_left _ hmod.le
    nlinarith
  have hbG : b * d.N_CTX * d.HEAD_DIM < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (by omega : b ≤ G d - 1))) hs1
  have hsum : d.N_CTX * d.HEAD_DIM * (b % d.H) + d.H * d.N_CTX * d.HEAD_DIM * (b / d.H) =
      b * d.N_CTX * d.HEAD_DIM := by
    calc _ = d.N_CTX * d.HEAD_DIM * (d.H * (b / d.H) + b % d.H) := by ring
      _ = d.N_CTX * d.HEAD_DIM * b := by rw [hbd]
      _ = _ := by ring
  have hP2 : d.H * d.N_CTX * d.HEAD_DIM * (b / d.H) < 2 ^ 31 := by omega
  have hm1 := IntV.I32.mul_nat (IntV.I32.toInt_ofNat (n := d.N_CTX * d.HEAD_DIM) (by nlinarith)) hl0' hP1
  have hm2 := IntV.I32.mul_nat (IntV.I32.toInt_ofNat (n := d.H * d.N_CTX * d.HEAD_DIM) hs2) hl1' hP2
  have hs := IntV.I32.add_nat hm1 hm2 (by rw [hsum]; exact hbG)
  rw [IntV.toInt_cast _ (by norm_num), hs, hsum]

/-- The offset `off_chz` of line 384: `b·N_CTX`. -/
theorem off_val {d : Inst} (hd : d.ok) {b : Nat} (hb : b < G d) {pid2 : I32}
    (hp : pid2.toInt = (b : Int)) :
    (IntV.cast (pid2 * IntV.ofNat d.N_CTX) : I64).toInt = ((b * d.N_CTX : Nat) : Int) := by
  obtain ⟨hs1, hs2⟩ := hd.size_lt
  have hHD : 0 < d.HEAD_DIM := by have := hd.hd; omega
  have hbG : b * d.N_CTX < 2 ^ 31 := by
    have h1 : b * d.N_CTX ≤ b * d.N_CTX * d.HEAD_DIM := Nat.le_mul_of_pos_right _ hHD
    have h2 : b * d.N_CTX * d.HEAD_DIM ≤ (G d - 1) * d.N_CTX * d.HEAD_DIM :=
      Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (by omega : b ≤ G d - 1))
    omega
  have hN : d.N_CTX < 2 ^ 31 := by
    have : d.N_CTX ≤ 2 ^ 14 := hd.n_le
    omega
  have hm := IntV.I32.mul_nat hp (IntV.I32.toInt_ofNat hN) hbG
  rw [IntV.toInt_cast _ (by norm_num), hm]

/-- A buffer's first pointer plus an `int64` offset of value `e` is its pointer to element `e`. -/
theorem ptr_add_zero {dt : DType} (B : Buf dt) (a : I64) {e : Nat} (h : a.toInt = (e : Int)) :
    B.ptr 0 + a = B.ptr (e : Int) := by
  show (B.ptr 0).offset a.toInt = _
  rw [h, Buf.ptr_offset]; simp

end Claims.AttnBwd
