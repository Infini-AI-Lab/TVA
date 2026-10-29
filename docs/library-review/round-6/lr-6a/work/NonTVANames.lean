import Lean
import TVA

/-! lr-6a: constants that the library's modules declare outside the `TVA` namespace. The claim
reader's cone follows only constants rooted at `TVA` (or the kernel's), so such a constant's
definition body is not followed, and its module file is hashed only if a `TVA` constant of the
same module is in the cone. -/

open Lean Meta

#eval show MetaM Unit from do
  let env ← getEnv
  let mut out : Array String := #[]
  for i in [0:env.header.moduleNames.size] do
    let m := env.header.moduleNames[i]!
    if (`TVA).isPrefixOf m then
      for n in env.header.moduleData[i]!.constNames do
        if n.getRoot != `TVA && !n.isInternal then
          let some ci := env.find? n | continue
          let k := match ci with
            | .defnInfo _ => "def" | .thmInfo _ => "theorem" | .axiomInfo _ => "axiom"
            | .opaqueInfo _ => "opaque" | .inductInfo _ => "inductive" | .ctorInfo _ => "ctor"
            | .recInfo _ => "rec" | .quotInfo _ => "quot"
          out := out.push s!"{m} {k} {n}"
  IO.println s!"{out.size} constants outside TVA"
  for s in out do IO.println s
