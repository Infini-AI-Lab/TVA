/-
The last step of `tva-check`: Lean's kernel re-checks every declaration of the named modules from their compiled
files, one module at a time. Installed at /usr/local/lib/tva/replay.lean and run as
`lake env lean --run /usr/local/lib/tva/replay.lean MODULE...`.

`replayFromImports` is leanchecker's own (LeanChecker.lean in the Lean 4 toolchain, Apache 2.0), unchanged: it
imports the module's imports, then replays the module's own constants into that environment with the kernel, and
frees the environment. `leanchecker M` instead takes every compiled module whose name starts with `M` (stale ones
left in the build folder too) and replays them all at once, each with its own copy of its imports, so its memory
grows with the number of modules. Here only the modules named are replayed, one after another.
-/
import Lean.CoreM
import Lean.Replay

open Lean

unsafe def replayFromImports (module : Name) : IO Unit := do
  let mFile ← findOLean module
  unless (← mFile.pathExists) do
    throw <| IO.userError s!"object file '{mFile}' of module {module} does not exist"
  -- Load all module data parts (exported, server, private)
  let mut fnames := #[mFile]
  let sFile := OLeanLevel.server.adjustFileName mFile
  if (← sFile.pathExists) then
    fnames := fnames.push sFile
    let pFile := OLeanLevel.private.adjustFileName mFile
    if (← pFile.pathExists) then
      fnames := fnames.push pFile
  let parts ← readModuleDataParts fnames
  if h : parts.size = 0 then throw <| IO.userError "failed to read module data" else
  let (mod, _) := parts[0]
  let (_, s) ← importModulesCore mod.imports |>.run
  let env ← finalizeImport s mod.imports {} 0 false false (isModule := true)
  let mut newConstants := {}
  -- Collect constants from last ("most private") part, which subsumes all prior ones
  for name in parts[parts.size-1].1.constNames, ci in parts[parts.size-1].1.constants do
    newConstants := newConstants.insert name ci
  discard <| env.toKernelEnv.replay newConstants
  env.freeRegions

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  if args.isEmpty then
    throw <| IO.userError "usage: replay.lean MODULE..."
  for a in args do
    let m := a.toName
    if m.isAnonymous then
      throw <| IO.userError s!"not a module name: {a}"
    replayFromImports m
    IO.println s!"replayed {m}"
  return 0
