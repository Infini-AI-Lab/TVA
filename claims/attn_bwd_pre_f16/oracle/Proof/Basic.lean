import Claim
import TVA.Proof

/-!
# `attn_bwd_pre_f16`: the grid, the launch and the index arithmetic
-/

namespace Claims.AttnBwdPre

open TVA


/-- Axis 0 of the grid: `N_CTX / BLOCK_M`. -/
def gx (d : Inst) : Nat := d.N_CTX / d.BLOCK_M

/-- Axis 1 of the grid: `Z · H`. -/
def gy (d : Inst) : Nat := d.Z * d.H

/-- Program `i`'s coordinate on axis 0. -/
def pm (d : Inst) (i : Nat) : Nat := i % gx d

/-- Program `i`'s coordinate on axis 1. -/
def bh (d : Inst) (i : Nat) : Nat := i / gx d % gy d

/-- The first row of `O`, `DO` and `Delta` that program `i` handles:
`bh · N_CTX + pm · BLOCK_M`. -/
def rowOf (d : Inst) (i : Nat) : Nat := bh d i * d.N_CTX + pm d i * d.BLOCK_M

/-- The launch has `gx · gy` programs. -/
theorem launch_size (d : Inst) : (launch d).size = gx d * gy d := by
  simp [Launch.size, launch, gx, gy]

/-- Program `i` has `program_id(0) = pm i`. -/
theorem launch_pid0 (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = pm d i := by
  simp [Launch.ctx, launch, pm, gx]

/-- Program `i` has `program_id(1) = bh i`. -/
theorem launch_pid1 (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 1 = bh d i := by
  simp [Launch.ctx, launch, bh, gx, gy]

/-- Program `i` has `program_id(2) = i / (gx · gy)`. -/
theorem launch_pid2 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 2 = i / (gx d * gy d) := by
  simp [Launch.ctx, launch, gx, gy]

/-- A launch with a program has a nonempty grid. -/
theorem grid_pos (d : Inst) (i : Fin (launch d).size) : 0 < gx d * gy d := by
  have h := i.isLt
  have h2 := launch_size d
  omega

/-- `BLOCK_M` is positive. -/
theorem block_pos {d : Inst} (hd : d.ok) : 0 < d.BLOCK_M := by
  obtain ⟨e, he⟩ := hd.block_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- `HEAD_DIM` is positive. -/
theorem head_pos {d : Inst} (hd : d.ok) : 0 < d.HEAD_DIM := by
  obtain ⟨e, he⟩ := hd.head_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- Axis 0 of a nonempty grid is positive. -/
theorem gx_pos (d : Inst) (i : Fin (launch d).size) : 0 < gx d := by
  have h := grid_pos d i
  rcases Nat.eq_zero_or_pos (gx d) with h0 | h0
  · rw [h0, Nat.zero_mul] at h; exact absurd h (Nat.lt_irrefl 0)
  · exact h0

/-- Axis 1 of a nonempty grid is positive. -/
theorem gy_pos (d : Inst) (i : Fin (launch d).size) : 0 < gy d := by
  have h := grid_pos d i
  rcases Nat.eq_zero_or_pos (gy d) with h0 | h0
  · rw [h0, Nat.mul_zero] at h; exact absurd h (Nat.lt_irrefl 0)
  · exact h0

/-- `BLOCK_M · (N_CTX / BLOCK_M) = N_CTX`. -/
theorem block_mul_gx (d : Inst) (hd : d.ok) : d.BLOCK_M * gx d = d.N_CTX := by
  unfold gx
  rw [Nat.mul_comm, Nat.div_mul_cancel hd.ctx_dvd]

/-- Program `i`'s axis-0 coordinate is inside the grid. -/
theorem pm_lt (d : Inst) (i : Fin (launch d).size) : pm d i < gx d :=
  Nat.mod_lt _ (gx_pos d i)

/-- Program `i`'s axis-1 coordinate is inside the grid. -/
theorem bh_lt (d : Inst) (i : Fin (launch d).size) : bh d i < gy d :=
  Nat.mod_lt _ (gy_pos d i)

/-- Program `i`'s own block lies below `N_CTX`. -/
theorem pm_block_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Nat)
    (hl : l < d.BLOCK_M) : pm d i * d.BLOCK_M + l < d.N_CTX := by
  have hp := pm_lt d i
  have hg := block_mul_gx d hd
  have hle : (pm d i + 1) * d.BLOCK_M ≤ gx d * d.BLOCK_M :=
    Nat.mul_le_mul_right _ (by omega)
  rw [Nat.succ_mul] at hle
  rw [Nat.mul_comm (gx d) d.BLOCK_M, hg] at hle
  omega

/-- Every row program `i` handles is below `Z · H · N_CTX`. -/
theorem row_lt (d : Inst) (hd : d.ok) (i : Fin (launch d).size) (l : Nat)
    (hl : l < d.BLOCK_M) : rowOf d i + l < d.Z * d.H * d.N_CTX := by
  have h1 := pm_block_lt d hd i l hl
  have hb : bh d i < d.Z * d.H := by
    have := bh_lt d i
    simpa [gy] using this
  unfold rowOf
  have h2 : (bh d i + 1) * d.N_CTX ≤ d.Z * d.H * d.N_CTX := by
    exact Nat.mul_le_mul_right d.N_CTX (by omega : bh d i + 1 ≤ d.Z * d.H)
  have h4 : bh d i * d.N_CTX + d.N_CTX = (bh d i + 1) * d.N_CTX := by ring
  omega

end Claims.AttnBwdPre
