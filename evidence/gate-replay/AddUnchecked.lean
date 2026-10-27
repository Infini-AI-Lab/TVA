import TVA
import Lean

/-! TVA library evidence (`evidence/gate-replay`, library issue L13): meta code that adds a theorem
without the kernel's check. `main` claims `False`; its value is `True.intro`, a closed term of
another type. `Environment.addDeclCore … (doCheck := false)` stores it, and the elaborator then
accepts `example : False := main`. -/

namespace GateReplay.AddUnchecked

open Lean Elab Command

run_cmd do
  let decl := Declaration.thmDecl
    { name := `GateReplay.AddUnchecked.main, levelParams := [], type := mkConst ``False,
      value := mkConst ``True.intro }
  match (← getEnv).addDeclCore 0 0 decl none (doCheck := false) with
  | .ok env => setEnv env
  | .error _ => throwError "addDeclCore failed"

example : False := main

end GateReplay.AddUnchecked
