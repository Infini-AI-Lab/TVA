import Claim
import TVA.Proof

/-!
# `moe_sum_reduce_bf16`: the launch

The grid of `launch d` is the 2-D `cdiv(T, BLOCK_M) × cdiv(D, BLOCK_DIM)`. This file proves it
valid, and computes the program that owns an output element `(t, j)`.
-/

namespace Claims.MoeSumReduce

open TVA

/-- The grid's extent along tokens. -/
abbrev gT (d : Inst) : Nat := (d.T + d.BLOCK_M - 1) / d.BLOCK_M

/-- The grid's extent along the hidden dimension. -/
abbrev gD (d : Inst) : Nat := (d.D + d.BLOCK_DIM - 1) / d.BLOCK_DIM

/-- The launch grid has `gT` programs along axis 0. -/
theorem grid0_eq (d : Inst) : (launch d).grid 0 = gT d := rfl

/-- The launch grid has `gD` programs along axis 1. -/
theorem grid1_eq (d : Inst) : (launch d).grid 1 = gD d := rfl

/-- The launch has `gT * gD` programs. -/
theorem launch_size (d : Inst) : (launch d).size = gT d * gD d := by
  simp [Launch.size, launch]

/-- The launch enables floating-point fusion. -/
theorem launch_fusion (d : Inst) : (launch d).fusion = true := rfl

/-- The launch is not a debug launch. -/
theorem launch_debug (d : Inst) : (launch d).debug = false := rfl

/-- `BLOCK_M` is positive. -/
theorem block_m_pos {d : Inst} (hd : d.ok) : 0 < d.BLOCK_M := by
  obtain ⟨e, he⟩ := hd.bm_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- `BLOCK_DIM` is positive. -/
theorem block_d_pos {d : Inst} (hd : d.ok) : 0 < d.BLOCK_DIM := by
  obtain ⟨e, he⟩ := hd.bd_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- The token grid is non-empty. -/
theorem gT_pos {d : Inst} (hd : d.ok) : 0 < gT d := by
  have h1 := block_m_pos hd
  have h2 := hd.T_pos
  refine Nat.div_pos ?_ h1
  omega

/-- The column grid is non-empty. -/
theorem gD_pos {d : Inst} (hd : d.ok) : 0 < gD d := by
  have h1 := block_d_pos hd
  have h2 := hd.D_pos
  refine Nat.div_pos ?_ h1
  omega

/-- The token grid is below `2^31`. -/
theorem gT_lt {d : Inst} (hd : d.ok) : gT d < 2 ^ 31 := by
  have hD := gD_pos hd
  have h : gT d * gD d < 2 ^ 31 := hd.grid_size_lt
  have : gT d ≤ gT d * gD d := Nat.le_mul_of_pos_right _ hD
  omega

/-- `BLOCK_M` is at most `2^20`. -/
theorem block_m_le {d : Inst} (hd : d.ok) : d.BLOCK_M ≤ 2 ^ 20 := by
  have h := hd.block_le
  have hD := block_d_pos hd
  have : d.BLOCK_M ≤ d.BLOCK_M * d.BLOCK_DIM := Nat.le_mul_of_pos_right _ hD
  omega

/-- `BLOCK_DIM` is at most `2^20`. -/
theorem block_d_le {d : Inst} (hd : d.ok) : d.BLOCK_DIM ≤ 2 ^ 20 := by
  have h := hd.block_le
  have hM := block_m_pos hd
  have : d.BLOCK_DIM ≤ d.BLOCK_M * d.BLOCK_DIM := Nat.le_mul_of_pos_left _ hM
  omega

/-- One power of two, at most `2^20`. -/
theorem tritonOk_one {a : Nat} (ha : ∃ e, a = 2 ^ e) (h : a ≤ 2 ^ 20) :
    Shape.TritonOk [a] := by
  obtain ⟨e, rfl⟩ := ha
  refine ⟨?_, by simpa using h⟩
  intro k hk
  simp only [List.mem_singleton] at hk
  exact ⟨e, hk.symm ▸ rfl⟩

/-- Two powers of two whose product is at most `2^20`. -/
theorem tritonOk_two {a b : Nat} (ha : ∃ e, a = 2 ^ e) (hb : ∃ e, b = 2 ^ e)
    (h : a * b ≤ 2 ^ 20) : Shape.TritonOk [a, b] := by
  obtain ⟨e, rfl⟩ := ha
  obtain ⟨e', rfl⟩ := hb
  refine ⟨?_, by simpa using h⟩
  intro k hk
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl
  · exact ⟨e, rfl⟩
  · exact ⟨e', rfl⟩

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have h1 : (launch d).grid 0 < 2 ^ 31 := by rw [grid0_eq]; exact gT_lt hd
  have h2 : (launch d).grid 1 < 2 ^ 16 := by rw [grid1_eq]; exact hd.grid1_lt
  have h3 : (launch d).grid 2 < 2 ^ 16 := by simp [launch]
  have h4 : (launch d).size < 2 ^ 31 := by rw [launch_size]; exact hd.grid_size_lt
  have h5 : (launch d).numCtas = 1 := rfl
  have h6 : (launch d).pdl = false := rfl
  refine ⟨h1, h2, h3, h4, h5, h6, rfl, ?_⟩
  intro s hs
  have hbM := hd.bm_pow2
  have hbD := hd.bd_pow2
  have hle := hd.block_le
  simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl
  · exact tritonOk_one hbM (block_m_le hd)
  · exact tritonOk_one hbD (block_d_le hd)
  · refine tritonOk_two hbM ⟨0, rfl⟩ ?_
    simpa using block_m_le hd
  · refine tritonOk_two ⟨0, rfl⟩ hbD ?_
    simpa using block_d_le hd
  · exact tritonOk_two hbM hbD hle

/-- Program `i` has token block `i % gT`. -/
theorem launch_pid0 (d : Inst) (i : Nat) :
    ((launch d).ctx i).pid 0 = i % gT d := by
  simp [Launch.ctx, launch]

/-- Program `i` has column block `i / gT`. -/
theorem launch_pid1 (d : Inst) (i : Nat) :
    ((launch d).ctx i).pid 1 = i / gT d % gD d := by
  simp [Launch.ctx, launch]

/-- The token block of any program is inside the grid. -/
theorem launch_pid0_lt {d : Inst} (hd : d.ok) (i : Nat) :
    ((launch d).ctx i).pid 0 < gT d := by
  rw [launch_pid0]; exact Nat.mod_lt _ (gT_pos hd)

/-- The column block of any program is inside the grid. -/
theorem launch_pid1_lt {d : Inst} (hd : d.ok) (i : Nat) :
    ((launch d).ctx i).pid 1 < gD d := by
  rw [launch_pid1]
  exact Nat.mod_lt _ (gD_pos hd)

/-- Every token tile is a program's. -/
theorem div_lt_gT {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) : t / d.BLOCK_M < gT d := by
  have hB := block_m_pos hd
  show t / d.BLOCK_M < (d.T + d.BLOCK_M - 1) / d.BLOCK_M
  rw [Nat.div_lt_iff_lt_mul hB]
  have h1 := Nat.div_add_mod (d.T + d.BLOCK_M - 1) d.BLOCK_M
  have h2 := Nat.mod_lt (d.T + d.BLOCK_M - 1) hB
  rw [Nat.mul_comm] at h1
  omega

/-- Every column tile is a program's. -/
theorem div_lt_gD {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.D) : j / d.BLOCK_DIM < gD d := by
  have hB := block_d_pos hd
  show j / d.BLOCK_DIM < (d.D + d.BLOCK_DIM - 1) / d.BLOCK_DIM
  rw [Nat.div_lt_iff_lt_mul hB]
  have h1 := Nat.div_add_mod (d.D + d.BLOCK_DIM - 1) d.BLOCK_DIM
  have h2 := Nat.mod_lt (d.D + d.BLOCK_DIM - 1) hB
  rw [Nat.mul_comm] at h1
  omega

/-- The program that owns output element `(t, j)`, as a number below the grid's size. -/
theorem prog_lt {d : Inst} (hd : d.ok) {t j : Nat} (ht : t < d.T) (hj : j < d.D) :
    t / d.BLOCK_M + gT d * (j / d.BLOCK_DIM) < (launch d).size := by
  rw [launch_size]
  have h1 : t / d.BLOCK_M + 1 ≤ gT d := div_lt_gT hd ht
  have h2 : j / d.BLOCK_DIM + 1 ≤ gD d := div_lt_gD hd hj
  have hmul : gT d * (j / d.BLOCK_DIM) + gT d ≤ gT d * gD d := by
    have h := Nat.mul_le_mul_left (gT d) h2
    rwa [Nat.mul_add, Nat.mul_one] at h
  omega

/-- The program owning element `(t, j)` has token block `t / BLOCK_M`. -/
theorem prog_pid0 {d : Inst} (hd : d.ok) {t j : Nat} (ht : t < d.T) (_hj : j < d.D) :
    ((launch d).ctx (t / d.BLOCK_M + gT d * (j / d.BLOCK_DIM))).pid 0
      = t / d.BLOCK_M := by
  rw [launch_pid0, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (div_lt_gT hd ht)]

/-- The program owning element `(t, j)` has column block `j / BLOCK_DIM`. -/
theorem prog_pid1 {d : Inst} (hd : d.ok) {t j : Nat} (ht : t < d.T) (hj : j < d.D) :
    ((launch d).ctx (t / d.BLOCK_M + gT d * (j / d.BLOCK_DIM))).pid 1
      = j / d.BLOCK_DIM := by
  rw [launch_pid1]
  have h1 := div_lt_gT hd ht
  have h2 := div_lt_gD hd hj
  have : (t / d.BLOCK_M + gT d * (j / d.BLOCK_DIM)) / gT d = j / d.BLOCK_DIM := by
    rw [Nat.add_mul_div_left _ _ (gT_pos hd), Nat.div_eq_of_lt h1, Nat.zero_add]
  rw [this, Nat.mod_eq_of_lt h2]

end Claims.MoeSumReduce
