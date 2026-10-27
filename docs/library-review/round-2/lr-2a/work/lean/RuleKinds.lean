import TVA

/-! lr-2a, round 2, survey `gate`: what the gate's `kinds` check sees of a `[rule]` field.

The claim reader lists, for each field, the constants of its statement filtered by `TVR.inScope`
(root `TVA` or the kernel's root), and `tvr check` asks of a `[rule]` only that one of them is a
`TVA.Rules.*` constant and none is a kernel constant other than `Inst`, `Inst.ok`, `model`,
`launch`. This file applies the same filter to two `[rule]` statements that are not library
facts: one with constants the kernel chose, one with an extra conjunct about the device. -/

namespace Kernels.RuleKinds

open TVA Lean Meta

structure Inst where
  n : Nat

def model (_ : Inst) : Kernel := ⟨fun _ => pure ()⟩
def launch (_ : Inst) : Launch := Launch.grid1 1 []

/-- `[rule]`, with accuracy constants the kernel chose (not a fact of `rules/num.md`). -/
def chosenConstants (P : Platform) : Prop :=
  Rules.Num.DivAcc P Math.Float.Format.binary32 ((2 : ℚ) ^ 100) ((2 : ℚ) ^ (-30 : ℤ)) 0

/-- `[rule]`, with an extra conjunct of the kernel's own: the device never faults. -/
def extraConjunct (K : Claim.Behaviour Inst) (P : Platform) : Prop :=
  ∀ d, Rules.Compiler.Faithful P (model d) (launch d) (K d) ∧ ∀ init, ¬ K d init .fault

def inScope (n : Name) : Bool := n.getRoot == `TVA || n.getRoot == `Kernels

def seen (n : Name) : MetaM (List Name) := do
  let some ci := (← getEnv).find? n | throwError "missing {n}"
  let some v := ci.value? | throwError "no value {n}"
  return (v.getUsedConstants.filter inScope).toList

#eval show MetaM Unit from do
  for n in [``chosenConstants, ``extraConjunct] do
    let cs ← seen n
    let own := cs.filter fun c => (`Kernels.RuleKinds).isPrefixOf c &&
      !([`Kernels.RuleKinds.Inst, `Kernels.RuleKinds.Inst.ok, `Kernels.RuleKinds.model,
         `Kernels.RuleKinds.launch].contains c)
    let rules := cs.filter fun c => (`TVA.Rules).isPrefixOf c
    IO.println s!"{n}: constants the gate sees {cs}; kernel-own {own}; TVA.Rules {rules}; \
      kinds check passes: {own.isEmpty && !rules.isEmpty}"

end Kernels.RuleKinds
