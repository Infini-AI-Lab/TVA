import TVA
open Lean Elab Command

/-- For every declaration of the five round-39 modules, the axioms it depends on. -/
elab "#round39_axioms" : command => do
  let env ← getEnv
  let mods := #[`TVA.Lemmas.IntW, `TVA.Lemmas.Tile2, `TVA.Lemmas.FloatFacts,
    `TVA.Lemmas.OnlineSoftmax, `TVA.Lemmas.Recurrence]
  for m in mods do
    let some idx := env.getModuleIdx? m | logInfo m!"missing {m}"
    let mut n := 0
    let mut bad : Array Name := #[]
    let mut thms := 0
    for (nm, ci) in env.constants.map₁.toList do
      if env.getModuleIdxFor? nm == some idx && !nm.isInternal then
        n := n + 1
        if ci matches .thmInfo _ then thms := thms + 1
        let axs ← liftCoreM <| Lean.collectAxioms nm
        for a in axs do
          unless a == ``propext || a == ``Classical.choice || a == ``Quot.sound do
            bad := bad.push (nm ++ a)
    logInfo m!"{m}: {n} declarations, {thms} theorems, non-standard axioms: {bad}"

#round39_axioms
