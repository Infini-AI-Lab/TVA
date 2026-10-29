import TVA
open Lean Meta Elab Command

/-- lr-4b, round 4: every theorem declared in a `TVA` module, and the axioms it depends on beyond
`propext`, `Classical.choice`, `Quot.sound`. -/
elab "#tva_axioms" : command => do
  let env ← getEnv
  let mut n := 0
  let mut bad : Array (Name × Array Name) := #[]
  for (name, ci) in env.constants.toList do
    let some idx := env.getModuleIdxFor? name | continue
    let mod := env.header.moduleNames[idx.toNat]!
    unless (`TVA).isPrefixOf mod do continue
    unless ci matches .thmInfo _ do continue
    n := n + 1
    let axs ← liftCoreM (Lean.collectAxioms name)
    let extra := axs.filter fun a => !(a == ``propext || a == ``Classical.choice || a == ``Quot.sound)
    unless extra.isEmpty do bad := bad.push (name, extra)
  logInfo m!"theorems in TVA modules: {n}; with other axioms: {bad.size}"
  for (nm, ax) in bad do logInfo m!"{nm}: {ax}"

#tva_axioms
