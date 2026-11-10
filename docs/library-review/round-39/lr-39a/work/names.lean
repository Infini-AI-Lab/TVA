import TVA
open Lean Elab Command
elab "#dump" : command => do
  let env ← getEnv
  let mods := #[`TVA.Lemmas.IntW, `TVA.Lemmas.Tile2, `TVA.Lemmas.FloatFacts,
    `TVA.Lemmas.OnlineSoftmax, `TVA.Lemmas.Recurrence]
  let mut out := ""
  for m in mods do
    let some idx := env.getModuleIdx? m | pure ()
    for (nm, _) in env.constants.map₁.toList do
      if env.getModuleIdxFor? nm == some idx && !nm.isInternal then
        out := out ++ s!"{m} {nm}\n"
  IO.FS.writeFile "names.txt" out
#dump
