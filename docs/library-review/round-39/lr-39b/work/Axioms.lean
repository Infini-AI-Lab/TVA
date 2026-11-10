import TVA
open Lean Elab Command

/-- For every declaration of the five round-39 modules: its count, and any axiom beyond the three
standard ones (or `sorryAx`). -/
elab "#check_round39" : command => do
  let env ← getEnv
  let mods := #[`TVA.Lemmas.IntW, `TVA.Lemmas.Tile2, `TVA.Lemmas.FloatFacts,
    `TVA.Lemmas.Recurrence, `TVA.Lemmas.OnlineSoftmax]
  for m in mods do
    let some idx := env.getModuleIdx? m | logError s!"no module {m}"
    let mut n := 0
    let mut thms := 0
    let mut bad : Array (Name × Array Name) := #[]
    for (c, info) in env.constants.toList do
      if env.getModuleIdxFor? c == some idx && !c.isInternal then
        n := n + 1
        if info matches .thmInfo _ then thms := thms + 1
        let axs ← liftCoreM <| collectAxioms c
        let extra := axs.filter fun a => a ∉ [``propext, ``Classical.choice, ``Quot.sound]
        if !extra.isEmpty then bad := bad.push (c, extra)
    logInfo m!"{m}: {n} public declarations, {thms} theorems, nonstandard axioms: {bad.toList}"

#check_round39
