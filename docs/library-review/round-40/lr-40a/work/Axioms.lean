import TVA
open Lean Elab Command

/-- lr-40a, round 40: for every declaration of every `TVA.Lemmas.*` module, the axioms it depends
on (only non-standard ones are listed), and the count of declarations and theorems. -/
elab "#lemma_axioms" : command => do
  let env ← getEnv
  let mods := env.header.moduleNames.filter fun m => (`TVA.Lemmas).isPrefixOf m
  let mut total := 0
  for m in mods do
    let some idx := env.getModuleIdx? m | logInfo m!"missing {m}"
    let mut n := 0
    let mut thms := 0
    let mut bad : Array Name := #[]
    for (nm, ci) in env.constants.map₁.toList do
      if env.getModuleIdxFor? nm == some idx && !nm.isInternal then
        n := n + 1
        if ci matches .thmInfo _ then thms := thms + 1
        let axs ← liftCoreM <| Lean.collectAxioms nm
        for a in axs do
          unless a == ``propext || a == ``Classical.choice || a == ``Quot.sound do
            bad := bad.push (nm ++ a)
    total := total + n
    logInfo m!"{m}: {n} declarations, {thms} theorems, non-standard axioms: {bad}"
  logInfo m!"total {total} declarations in {mods.size} modules"

#lemma_axioms
