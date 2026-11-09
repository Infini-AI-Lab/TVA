import TVA.Rules.Compiler
import TVA.Lang.Ops

open Lean in
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← getEnv
  for n in [`TVA.Rules.Compiler.Faithful, `TVA.tl.div_rn] do
    match ← findDocString? env n with
    | some d => IO.println s!"{n}: {d.length} chars; scope 12 L165: {(d.splitOn "L165").length - 1}; L166: {(d.splitOn "L166").length - 1}"
    | none => IO.println s!"{n}: no docstring"
