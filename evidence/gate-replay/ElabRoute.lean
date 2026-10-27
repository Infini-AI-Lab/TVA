import TVA
import Lean

/-! TVA library evidence (`evidence/gate-replay`, library issue L13): the same unchecked theorem as
`AddUnchecked.lean`, added by a command the file defines with `elab` and then runs, so that no
`run_cmd` appears. -/

namespace GateReplay.ElabRoute

open Lean Elab Command

elab "add_unchecked_main" : command => do
  let decl := Declaration.thmDecl
    { name := `GateReplay.ElabRoute.main, levelParams := [], type := mkConst ``False,
      value := mkConst ``True.intro }
  match (← getEnv).addDeclCore 0 0 decl none (doCheck := false) with
  | .ok env => setEnv env
  | .error _ => throwError "addDeclCore failed"

add_unchecked_main

example : False := main

end GateReplay.ElabRoute
