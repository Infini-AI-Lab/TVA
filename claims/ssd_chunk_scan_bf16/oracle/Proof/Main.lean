import Proof.Post

/-!
# The launch

The launch is valid, each program is safe (`Noinit`, `Init`), the tiles are disjoint, and the
programs' results give `Post`.
-/

namespace Claims.SsdChunkScan

open TVA

/-- The launch is valid: the grid is within CUDA's limits and the block shapes are powers of two. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hg := gx_pos hd
  have hgrid := hd.grid_lt
  obtain ⟨a, ha⟩ := hd.bsd_pow2
  have hH := hd.H_pos
  have hm := nmeta_pos hd
  have hb := hd.bsd_le
  refine ⟨?_, hd.nmeta_lt, hd.H_lt, hgrid, rfl, rfl, rfl, ?_⟩
  · show gx d < 2 ^ 31
    calc gx d ≤ gx d * d.nmeta := Nat.le_mul_of_pos_right _ hm
      _ ≤ gx d * d.nmeta * d.H := Nat.le_mul_of_pos_right _ hH
      _ < 2 ^ 31 := hgrid
  · intro s hs
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    have h64 : ∃ e, (64 : Nat) = 2 ^ e := ⟨6, by norm_num⟩
    have h32 : ∃ e, (32 : Nat) = 2 ^ e := ⟨5, by norm_num⟩
    have h1 : ∃ e, (1 : Nat) = 2 ^ e := ⟨0, by norm_num⟩
    have hB : ∃ e, d.BSD = 2 ^ e := ⟨a, ha⟩
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      refine ⟨fun k hk => ?_, ?_⟩
    all_goals first
      | (simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
         rcases hk with rfl | rfl <;> assumption)
      | (simp only [List.prod_cons, List.prod_nil, BM, BN, BK]; omega)

/-- A tile's bytes are bytes of `out`. -/
theorem W_sub_bytes {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * d.nmeta * d.H) :
    W d i ⊆ d.out.bytes :=
  (W_sub hd hi).trans (d.out.elems_sub_bytes (fun _ hx => lt_of_lt_of_le hx hd.out_len))

end Claims.SsdChunkScan
