import TVA.Claim

/-! The claim's shape (rules survey): a `[kernel]` field mentions neither `K` nor `P` (the gate's `kinds`
check), so it is a closed proposition, and `Claim.Consistent Assumptions`, which every bundle proves,
proves it. A sample `Assumptions` with one field of each kind: -/

namespace Probe

structure Inst where
  n : Nat

def bound : Nat := 7

structure Assumptions (K : TVA.Claim.Behaviour Inst) (P : TVA.Platform) : Prop where
  /-- [rule] stands in for `Compiler.Faithful` (any statement about `K`). -/
  rule : ∀ d init o, K d init o → True
  /-- [kernel] a fact about the kernel's own values. -/
  own : bound ≤ 8

/-- Whatever the `[kernel]` field says, `consistent` proves it. -/
theorem own_of_consistent (h : TVA.Claim.Consistent Assumptions) : bound ≤ 8 := by
  obtain ⟨K, P, _, hA⟩ := h
  exact hA.own

end Probe
