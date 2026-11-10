import TVA
open Lean Elab Command

/-- Each name a round-39 module's header cites (read from `libnames.json`, written by
`docnames.py`) must be some constant of the environment, written in full or as a suffix after a
dot. Prints the ones that are not. -/
elab "#check_docnames " path:str : command => do
  let env ← getEnv
  let txt ← IO.FS.readFile path.getString
  let json ← IO.ofExcept (Json.parse txt)
  let some obj := (json.getObj?.toOption) | throwError "bad json"
  let names := env.constants.toList.map (·.1.toString)
  for ⟨m, v⟩ in obj.toArray do
    let some arr := v.getArr?.toOption | continue
    let mut missing := #[]
    for t in arr do
      let some s := t.getStr?.toOption | continue
      unless names.any (fun n => n == s || n.endsWith ("." ++ s)) do missing := missing.push s
    logInfo m!"{m}: {arr.size} names cited, not found: {missing.toList}"

#check_docnames "../docs/library-review/round-39/lr-39b/work/libnames.json"
