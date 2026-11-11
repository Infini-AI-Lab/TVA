import Claim
import TVA.Proof
import Proof.Foot

/-!
# The lane addresses and masks of the kernel's pointer tiles

`a_ptrs` and `b_ptrs` are a base pointer plus an `int64` offset (`aOfs1`) plus an `int32` tile
offset (`aOfs2`); the post-loop pointers of `seq_idx` and `out` have the same shape. The load
masks compare the sign-extended row and column offsets with `chunk_size_limit` and with
`K - k·BLOCK_SIZE_K`. This file defines those terms and records their lane values.
-/

namespace Claims.SsdBmm

open TVA

set_option maxHeartbeats 800000

/-! ## The row and column offsets -/

/-- `offs_m = pid_m * BLOCK_SIZE_M + arange(BLOCK_SIZE_M)`. -/
def offsM (d : Inst) (i : Fin (launch d).size) : Tensor [BM] I32 :=
  (Tensor.scalar (IntV.ofInt (↑(↑i % gx d / ((d.C + (BN - 1)) / BN)) : ℤ) : I32)).item *
      (IntV.ofNat BM : I32) +
    tl.arange 0 BM

/-- `offs_n = pid_n * BLOCK_SIZE_N + arange(BLOCK_SIZE_N)`. -/
def offsN (d : Inst) (i : Fin (launch d).size) : Tensor [BN] I32 :=
  (Tensor.scalar (IntV.ofInt (↑(↑i % gx d % ((d.C + (BN - 1)) / BN)) : ℤ) : I32)).item *
      (IntV.ofNat BN : I32) +
    tl.arange 0 BN

/-! ## The `a` pointer tile -/

/-- The `int64` part of `a_ptrs`' offset. -/
def aOfs1 (d : Inst) (i : Fin (launch d).size) : I64 :=
  (IntV.ofInt 0 : I64) +
      (Tensor.scalar (IntV.ofInt (↑(↑i / gx d / d.G) : ℤ))).item * IntV.ofInt (d.C : ℤ) *
        IntV.ofInt (d.aS : ℤ) +
    IntV.ofInt (hOf d i : ℤ) * IntV.ofInt (d.aH : ℤ)

/-- The `int32` tile offsets of `a_ptrs`. -/
def aOfs2 (d : Inst) (i : Fin (launch d).size) : Tensor [BM, BK] I32 :=
  (offsM d i).col BK * (IntV.ofNat d.aS : I32) +
    (tl.arange 0 BK).row BM * (IntV.ofNat d.aK : I32)

/-- `a_ptrs` before the loop. -/
def aPtrs (d : Inst) (i : Fin (launch d).size) : Tensor [BM, BK] (Ptr .bf16) :=
  d.a.ptr 0 + aOfs1 d i + aOfs2 d i

/-! ## The `b` pointer tile -/

/-- The `int64` part of `b_ptrs`' offset. -/
def bOfs1 (d : Inst) (i : Fin (launch d).size) : I64 :=
  (IntV.ofInt 0 : I64) +
      (Tensor.scalar (IntV.ofInt (↑(↑i / gx d / d.G) : ℤ))).item * IntV.ofInt (d.C : ℤ) *
        IntV.ofInt (d.bS : ℤ) +
    IntV.ofInt (hOf d i : ℤ) * IntV.ofInt (d.bH : ℤ)

/-- The `int32` tile offsets of `b_ptrs`. -/
def bOfs2 (d : Inst) (i : Fin (launch d).size) : Tensor [BK, BN] I32 :=
  (tl.arange 0 BK).col BN * (IntV.ofNat d.bK : I32) +
    (offsN d i).row BK * (IntV.ofNat d.bS : I32)

/-- `b_ptrs` before the loop. -/
def bPtrs (d : Inst) (i : Fin (launch d).size) : Tensor [BK, BN] (Ptr .bf16) :=
  d.b.ptr 0 + bOfs1 d i + bOfs2 d i

/-- `chunk_size_limit = min chunk_size (seqlen - pid_c * chunk_size)`, in `int64`. -/
def chunkLimit (d : Inst) (i : Fin (launch d).size) : I64 :=
  (tl.iminimum (Tensor.scalar (IntV.ofInt (d.C : ℤ) : I64))
      (Tensor.scalar
        ((IntV.ofInt (d.T : ℤ) : I64) - IntV.ofInt (↑(↑i / gx d / d.G) : ℤ) * IntV.ofInt (d.C : ℤ)))).item

/-! ## The `seq_idx` and `out` pointers -/

/-- The `int64` part of the `seq_idx` pointers' offset. -/
def sOfs1 (d : Inst) (i : Fin (launch d).size) : I64 :=
  (IntV.ofInt 0 : I64) +
    (Tensor.scalar (IntV.ofInt (↑(↑i / gx d / d.G) : ℤ))).item * IntV.ofInt (d.C : ℤ) *
      (IntV.ofInt 1 : I64)

/-- The `seq_idx` pointers of the tile's rows (line 171). -/
def seqPtrsM (d : Inst) (i : Fin (launch d).size) : Tensor [BM] (Ptr .i32) :=
  d.seqIdx.ptr 0 + sOfs1 d i + offsM d i * (IntV.ofNat 1 : I32)

/-- The `seq_idx` pointers of the tile's columns (line 176). -/
def seqPtrsN (d : Inst) (i : Fin (launch d).size) : Tensor [BN] (Ptr .i32) :=
  d.seqIdx.ptr 0 + sOfs1 d i + offsN d i * (IntV.ofNat 1 : I32)

/-- The `int64` part of `out_ptrs`' offset. -/
def outOfs1 (d : Inst) (i : Fin (launch d).size) : I64 :=
  (IntV.ofInt 0 : I64) +
      (Tensor.scalar (IntV.ofInt (↑(↑i / gx d / d.G) : ℤ))).item *
        (IntV.ofInt (oC d) : I64) +
    IntV.ofInt (hOf d i : ℤ) * (IntV.ofInt (oH d) : I64)

/-- The `int32` tile offsets of `out_ptrs`. -/
def outOfs2 (d : Inst) (i : Fin (launch d).size) : Tensor [BM, BN] I32 :=
  (IntV.ofNat d.C : I32) * (offsM d i).col BN + (offsN d i).row BM * (IntV.ofNat 1 : I32)

/-- `out_ptrs`: the pointers to the program's `64 × 128` tile of `out`. -/
def outPtrs (d : Inst) (i : Fin (launch d).size) : Tensor [BM, BN] (Ptr .f32) :=
  d.out.ptr 0 + outOfs1 d i + outOfs2 d i

/-! ## The loop-step displacement -/

/-! ## The tile rows and columns -/

/-- Row of `a` (a token index) of lane `mll` of program `i`'s tile. -/
def rowA (d : Inst) (i : Fin (launch d).size) (mll : Nat) : Nat :=
  cOf d i * d.C + mOf d i * BM + mll

/-- Column of `b` (a token index) of lane `nn` of program `i`'s tile. -/
def colB (d : Inst) (i : Fin (launch d).size) (nn : Nat) : Nat :=
  cOf d i * d.C + nOf d i * BN + nn

/-! ## Index bounds -/

/-- `n < ⌈T/B⌉` gives `n·B < T`. -/
theorem mul_lt_of_lt_ceil {n T B : Nat} (hB : 0 < B) (h : n < (T + B - 1) / B) : n * B < T := by
  set m := n * B with hm
  have h1 : n + 1 ≤ (T + B - 1) / B := h
  have h2 : (n + 1) * B ≤ T + B - 1 := by
    rw [Nat.le_div_iff_mul_le hB] at h1
    exact h1
  have h3 : (n + 1) * B = m + B := by rw [Nat.add_mul, Nat.one_mul]
  rw [h3] at h2
  omega

/-- The chunk's first token is a token of the sequence. -/
theorem cOf_mul_C_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : cOf d i * d.C < d.T := by
  have h := cOf_lt i
  unfold nch at h
  set m := cOf d i * d.C with hm
  have hC : 0 < d.C := by have := hd.C_ge; omega
  have h1 : cOf d i + 1 ≤ (d.T + d.C - 1) / d.C := h
  have h2 : (cOf d i + 1) * d.C ≤ d.T + d.C - 1 := by
    rw [Nat.le_div_iff_mul_le hC] at h1
    exact h1
  have h3 : (cOf d i + 1) * d.C = m + d.C := by rw [Nat.add_mul, Nat.one_mul]
  rw [h3] at h2
  omega

/-- `pid_m` is below the number of row blocks. -/
theorem mOf_lt_num {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    mOf d i < (d.C + BM - 1) / BM := by
  have h : (i : Nat) % gx d < gx d := Nat.mod_lt _ (gx_pos hd)
  have hgx : gx d = (d.C + BM - 1) / BM * numPidN d := rfl
  unfold mOf
  rw [Nat.div_lt_iff_lt_mul (numPidN_pos hd)]
  exact h.trans_eq hgx

/-- `pid_n` is below the number of column blocks. -/
theorem nOf_lt_num {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    nOf d i < numPidN d := by
  unfold nOf numPidN
  exact Nat.mod_lt _ (numPidN_pos hd)

/-- The tile's first row is a row of the chunk. -/
theorem mOf_mul_BM_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : mOf d i * BM < d.C := by
  have h := mOf_lt_num hd i
  have h1 := mul_lt_of_lt_ceil (n := mOf d i) (T := d.C) (B := BM) BM_pos h
  rw [BM] at h1 ⊢
  exact h1

/-- The tile's first column is a column of the chunk. -/
theorem nOf_mul_BN_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : nOf d i * BN < d.C := by
  have h := nOf_lt_num hd i
  unfold numPidN at h
  exact mul_lt_of_lt_ceil (n := nOf d i) (T := d.C) (B := BN) BN_pos h

/-- Step `t` of the loop starts below `K`. -/
theorem t_mul_BK_lt {d : Inst} {t : Nat} (ht : t < steps d.K) : t * BK < d.K := by
  unfold steps at ht
  exact mul_lt_of_lt_ceil (n := t) (T := d.K) (B := BK) (by simp [BK]) ht

end Claims.SsdBmm



