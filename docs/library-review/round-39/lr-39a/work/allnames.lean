import TVA
open Lean Elab Command
elab "#dumpall" : command => do
  let env ← getEnv
  let mut out := ""
  for (nm, _) in env.constants.map₁.toList do
    if (`TVA).isPrefixOf nm && !nm.isInternal then
      let m := match env.getModuleIdxFor? nm with
        | some i => env.header.moduleNames[i.toNat]!
        | none => `none
      out := out ++ s!"{m} {nm}\n"
  IO.FS.writeFile "allnames.txt" out
#dumpall
