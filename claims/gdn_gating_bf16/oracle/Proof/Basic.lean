import Claim
import TVA.Proof

/-!
# `gdn_gating_bf16`: the launch and the program's context

The grid `(batch, 1, cdiv(H, 8))` is valid on every covered instance, `debug = false`, and
program `i` of it has `pid 0 = i % batch`, `pid 1 = 0`, `pid 2 = i / batch`.
-/

namespace Claims.GdnGating

open TVA

/-- The launch has `batch * Bn d` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.batch * Bn d := by
  simp [Launch.size, launch]

/-- Program `i` of the grid: `pid 0 = i % batch`. -/
theorem launch_pid0 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = (i : Nat) % d.batch := rfl

/-- Program `i` of the grid: `pid 1 = 0` (the grid's second axis is `1`). -/
theorem launch_pid1 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 1 = 0 := by
  simp [Launch.ctx, launch, Nat.mod_one]

/-- Program `i` of the grid: `pid 2 = i / batch`. -/
theorem launch_pid2 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 2 = (i : Nat) / d.batch := by
  simp [Launch.ctx, launch]

/-- `Bn d ≤ d.H`: the number of blocks of 8 heads does not exceed the heads. -/
theorem bn_le (d : Inst) : Bn d ≤ d.H := by
  unfold Bn
  omega

/-- For more than one head, the number of blocks is strictly below the head count. -/
theorem bn_lt {d : Inst} (h : 1 < d.H) : Bn d < d.H := by
  unfold Bn
  omega

/-- For at most one head, there is at most one block. -/
theorem bn_le_one {d : Inst} (h : d.H ≤ 1) : Bn d ≤ 1 := by
  unfold Bn
  omega

/-- The grid has fewer than `2^31` programs. -/
theorem launch_size_lt {d : Inst} (hd : d.ok) : (launch d).size < 2 ^ 31 := by
  rw [launch_size]
  rcases Nat.eq_zero_or_pos d.batch with hb | hb
  · rw [hb]; simp
  rcases Nat.lt_or_ge 1 d.H with hH | hH
  · exact lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_left (bn_lt hH) hb) hd.flat_le
  · have h1 : Bn d ≤ 1 := bn_le_one hH
    calc d.batch * Bn d ≤ d.batch * 1 := Nat.mul_le_mul_left _ h1
      _ = d.batch := Nat.mul_one _
      _ < 2 ^ 31 := hd.batch_lt

/-- The block shape `[8]` is one Triton accepts. -/
theorem block_ok : TVA.Shape.TritonOk ([8] : Shape) := by
  refine ⟨?_, ?_⟩
  · intro k hk
    simp only [List.mem_singleton] at hk
    subst hk
    exact ⟨3, by norm_num⟩
  · norm_num [List.prod]

/-- Every covered launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine ⟨hd.batch_lt, by simp [launch], hd.grid_z, launch_size_lt hd, rfl, rfl, rfl, ?_⟩
  intro s hs
  simp only [launch, List.mem_singleton] at hs
  subst hs
  exact block_ok

/-- Every covered launch has `debug = false`. -/
theorem launch_debug (d : Inst) : (launch d).debug = false := rfl

end Claims.GdnGating
