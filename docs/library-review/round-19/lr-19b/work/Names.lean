import TVA

/-! lr-19b: print every declaration under `TVA` (one per line), for names_check.py. -/
open Lean in
#eval show CoreM Unit from do
  let env ← getEnv
  for (n, _) in env.constants.map₁.toList do
    if (`TVA).isPrefixOf n && !n.isInternal then IO.println n
