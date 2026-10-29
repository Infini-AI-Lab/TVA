-- lr-10a, round 10: the constants of the library's modules whose name is not under `TVA` (which the claim
-- reader's cone leaves out, `inScope`), by root.
import TVA
open Lean Meta
#eval show MetaM Unit from do
  let env ← getEnv
  let mut n := 0
  let mut byRoot : Std.HashMap String Nat := {}
  let mut examples : Array String := #[]
  for (c, _) in env.constants.map₁.toList do
    match env.getModuleIdxFor? c with
    | some i =>
      let m := env.header.moduleNames[i.toNat]!
      if (`TVA).isPrefixOf m && c.getRoot != `TVA then
        n := n + 1
        let r := c.getRoot.toString
        byRoot := byRoot.insert r ((byRoot.getD r 0) + 1)
        if !c.isInternal && examples.size < 40 then examples := examples.push s!"{c} ({m})"
    | none => pure ()
  IO.println s!"library constants not rooted at TVA: {n}"
  IO.println s!"by root: {byRoot.toList}"
  for e in examples do IO.println e
