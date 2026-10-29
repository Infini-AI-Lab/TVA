-- lr-7a: a plain namespace `_root_` (accepted by manifest_problems) in the claim reader's form: the reader fails.
import Lean
open Lean Meta
structure Assumptions (n : Nat) : Prop where
  h : True
def Spec (n : Nat) : Prop := True
theorem main : ∀ n, Assumptions n → Spec n := fun _ _ => trivial
example : ∀ n, _root_.Assumptions n → _root_.Spec n := _root_.main
#eval show MetaM Unit from do
  let env ← getEnv
  let ns : Name := `_root_
  IO.println s!"ns ++ Assumptions = {ns ++ `Assumptions}; found: {(env.find? (ns ++ `Assumptions)).isSome}"
  let _ ← match env.find? (ns ++ `Assumptions) with
    | some (.inductInfo v) => pure v.numParams
    | _ => throwError "Assumptions is not a structure"
  IO.println "TVR_JSON {}"
