import TVA

/-! TVA library evidence (`evidence/gate-replay`, library issue L13): the control. A small module in
the shape of a kernel bundle, with an ordinary, checked theorem. `leanchecker` must accept it and the
gate's hygiene list must not flag it. -/

namespace GateReplay.Control

open TVA

/-- A trivial claim, proved the ordinary way. -/
theorem main : ∀ n : Nat, n + 0 = n := fun n => Nat.add_zero n

/-- A definition the theorem does not need. -/
def body : Kernel := ⟨fun _ => pure ()⟩

end GateReplay.Control
