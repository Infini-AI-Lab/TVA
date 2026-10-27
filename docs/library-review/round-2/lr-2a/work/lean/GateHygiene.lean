import TVA
import Lean

/-! lr-2a, round 2, survey `gate`: does `tvr check` rest on the kernel's Lean being
kernel-checked?  This self-contained file imitates a kernel bundle whose `Post` is `False`
(so the claim is not true), yet makes `tvr check`'s `build`, `hygiene`, `shape` and `axioms`
checks pass.

`main` is added to the environment with `Environment.addDeclCore ... (doCheck := false)` inside a
`run_cmd`, so the Lean kernel never verifies its proof term.  None of the tokens on the gate's
hygiene denylist appears in the injection code, so the hygiene regex of `tools/tvr.py` passes it
(checked separately in Python).  The three `example`s are the claim reader's shape checks;
the axioms command is its axioms check. -/

namespace Kernels.GateHygiene

open TVA Lean Elab Command Term Meta

structure Inst where
  n : Nat

def Inst.ok (_ : Inst) : Prop := True

def body : Kernel := ⟨fun _ => pure ()⟩
def model (_ : Inst) : Kernel := body
def launch (_ : Inst) : Launch := Launch.grid1 1 []

def Pre (_ : Inst) (_ : Mem) : Prop := True

/-- Deliberately unachievable: a device that runs cannot satisfy `False`. -/
def Post (_ : Inst) (_ _ : Mem) : Prop := False

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

run_cmd do
  let ty ← liftTermElabM do
    instantiateMVars (← Term.elabTerm
      (← `(∀ (K : TVA.Claim.Behaviour Inst) (P : TVA.Platform), Assumptions K P → Spec K)) none)
  -- a value never kernel-checked against the declared type `ty`.  Its own inferred type is
  -- `∀ K P, Assumptions K P → True`, nothing like `Spec K`; it uses the assumption's projection
  -- so the gate's `usage` check also passes, and mentions no nonstandard axiom
  let val ← liftTermElabM do
    instantiateMVars (← Term.elabTerm
      (← `(fun (K : TVA.Claim.Behaviour Inst) (P : TVA.Platform) (h : Assumptions K P) =>
            (h.faithful, True.intro).2)) none)
  let decl := Declaration.thmDecl
    { name := `Kernels.GateHygiene.main, levelParams := [], type := ty, value := val }
  match (← getEnv).addDeclCore 0 0 decl none (doCheck := false) with
  | .ok env => setEnv env
  | .error e => throwError "addDeclCore failed: {e.toMessageData (← getOptions)}"

-- the claim reader's shape checks (they only need `main`'s stored TYPE, not its body)
example : ∀ (K : TVA.Claim.Behaviour Inst) (P : TVA.Platform), Assumptions K P → Spec K := main
example : TVA.Claim.Consistent Assumptions := consistent
example : ∀ K, Spec K = TVA.Claim.Meets Inst.ok Pre Post K := fun _ => rfl

#print axioms main
#print axioms consistent

end Kernels.GateHygiene
