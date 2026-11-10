import TVA
open Lean Elab Command
/-- lr-40a, round 40: every public `TVA` declaration with its module, to work/allnames.txt (run from `lean/`). -/
elab "#dumpall" : command => do
  let env ← getEnv
  let mut out := ""
  for (nm, _) in env.constants.map₁.toList do
    if (`TVA).isPrefixOf nm && !nm.isInternal then
      let m := match env.getModuleIdxFor? nm with
        | some i => env.header.moduleNames[i.toNat]!
        | none => `none
      out := out ++ s!"{m} {nm}\n"
  IO.FS.writeFile "../docs/library-review/round-40/lr-40a/work/allnames.txt" out
#dumpall
