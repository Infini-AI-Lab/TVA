import Claim
import TVA.Proof
import Proof.Foot

/-!
# The launch is valid
-/

namespace Claims.RecomputeWU

open TVA

/-- The source's block shapes are shapes Triton accepts. -/
theorem blocks_ok : ∀ s ∈ blockShapes, Shape.TritonOk s := by
  have h64 : ∀ k : Nat, (k = 64 ∨ k = 1) → ∃ e, k = 2 ^ e := by
    rintro k (rfl | rfl)
    · exact ⟨6, by norm_num⟩
    · exact ⟨0, by norm_num⟩
  intro s hs
  simp only [blockShapes, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl <;> refine ⟨fun k hk => h64 k ?_, by simp⟩ <;>
    simp at hk <;> tauto

/-- **The launch is valid**: the grid is within CUDA's limits. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid :=
  ⟨hd.grid_t, hd.grid_bh, by simp [launch], by
    have := hd.grid_size; show g0 d * (d.B * d.HV) * 1 < 2 ^ 31; omega,
    rfl, rfl, rfl, blocks_ok⟩

end Claims.RecomputeWU
