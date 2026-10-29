import TVA
open Lean in
#eval show CoreM Unit from do
  let env ← getEnv
  let mut out : Array String := #[]
  for (n, _) in env.constants.toList do
    if (`TVA).isPrefixOf n && !n.isInternal then
      out := out.push n.toString
  IO.FS.writeFile "names.txt" ("\n".intercalate (out.qsort (· < ·)).toList)
