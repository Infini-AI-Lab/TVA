# Theorems added without the kernel's check

Library evidence (not a kernel's), for library issue L13. It tests TVA's gate (`tools/tvr.py`)
and Lean 4.35.0's own checker, not the device: Lean 4.35.0 with `leanchecker` from the same
toolchain, Python 3.13, 2026-10-27. Script: [run.py](run.py); output: [out.txt](out.txt).

**Question.** Lean code that runs while a file elaborates (a `run_cmd`, a command defined with
`elab`, …) can call `Environment.addDeclCore … (doCheck := false)` and store a theorem whose value
does not prove its statement. The library reviewer of round 2 built such a kernel bundle, with
`Post = False`, that passed the gate's `build`, `hygiene`, `shape`, `axioms` and `usage` checks.
Does the gate now catch it?

**What the gate does** (`tools/tvr.py`, `check`; protocol §8):
- `hygiene` reads each of the kernel's Lean files whole, comments and strings included, against
  the list `HYGIENE`: proof escapes, kernel and compiler bypasses (`addDecl…`, `doCheck`,
  `setEnv`, `Environment`, `set_option debug.*`, …), meta code (`#eval`, `run_cmd`, `elab`,
  `macro`, `syntax`, `initialize`, …), meta types (`IO`, `MetaM`, `CommandElabM`, …) and
  attributes that install handlers. It runs *before* the build, since building runs the file's
  meta code. The build runs only when it passes.
- `replay` runs `leanchecker <root>` on the compiled kernel: Lean's own tool replays every
  declaration of each module whose name starts with the kernel's module root into the
  environment of the module's imports, with Lean's kernel checking each one. It reads the compiled
  `.olean` files, so whatever the elaborator was made to store is checked again.

**Method.** Each case is a Lean file compiled with the library's Lean (`lake env lean -o`) into a
fresh temporary directory. The script records whether it elaborates, what `leanchecker` says of
the compiled module, and which kinds of the `HYGIENE` list its text hits.

**Result.**

| Case | Elaborates | `replay` (`leanchecker`) | `hygiene` |
|---|---|---|---|
| C1 control: an ordinary module with a checked theorem ([Control.lean](Control.lean)) | yes | accepts | passes |
| A1 `run_cmd` + `addDeclCore (doCheck := false)`: `main : False := True.intro` ([AddUnchecked.lean](AddUnchecked.lean)) | yes, and `example : False := main` too | rejects: declaration type mismatch | flags (bypass, meta code) |
| A2 the same through a command the file defines with `elab`, no `run_cmd` ([ElabRoute.lean](ElabRoute.lean)) | yes | rejects: declaration type mismatch | flags (bypass, meta code) |
| R1 the library reviewer's round-2 bundle (`Post = False`) | yes | rejects: declaration has metavariables | flags (bypass, meta code) |

**Conclusions.**
1. Elaboration alone does not prove a theorem: A1, A2 and R1 elaborate, and Lean then accepts
   `example : False := main`.
2. `leanchecker` rejects each of them and accepts the control. Since it re-checks every
   declaration of the kernel's compiled modules, the gate's `replay` check closes the hole
   whatever route the meta code takes.
3. The `hygiene` list flags each of them before anything is built, so their meta code does not
   even run during the gate. It is a list of tokens, so it is a first line of defence; `replay`
   is what the soundness of a kernel's theorem rests on.
