# The tool after round 13

Library evidence (not a kernel's), for library issues L7, L50, L52, L62, L63, L68, L69 and L70 (round 13).
It tests TVA's tool (`tools/tvr.py`, `tools/claim_reader.lean.in`), not the device: Lean 4.35.0, git 2.52
(`/usr/bin/git`), Python 3.13, 2026-10-28; no GPU and no Triton run. Part N needs the network (it reads
Mathlib's upstream refs and history). Script: [run.py](run.py); output: [out.txt](out.txt). The release
check's own run is in the round-13 author's notes.

**Questions.**
1. (L62, L63) Does the gate's `kinds` check accept in a `[rule]` field only a fact the library states,
   applied to the platform (`Compiler.Faithful`, a `Platform → Prop` fact such as `DivFull` or
   `Exp2F32`, a `…Total` rule, `DivRn` on `float64`), and refuse a shape with constants (lr-9a's
   `MathAcc` for `tl.exp`, `DivAcc` with the library's own constants written out, the family `DotF8`,
   `DivRn` on `float32`), a constant of the kernel's own however it is declared (`private`,
   `_root_.Kernels._tol`, `_root_._tol`), `K` outside `Faithful`, and two facts in one field? Does the
   namespace check refuse the `_root_` names and an instance named `_x`, which `Name.isInternal` used to
   exempt?
2. (L68) Do lr-10a's three edits of an assumption's statement (a tighter literal, a `∀ _ : Nat` binder, a
   changed `[kernel]` statement) change that item's digest, and only that one, while a proof edit changes
   none?
3. (L50) Does the map check fail on a function of the source that lies neither in the kernel lines, nor
   in the host lines, nor in `source.unused`, however `jit` is spelled (a call `triton.jit(_h)`, an alias, a
   decorator bound to a name, a `triton.constexpr_function`, a lambda), and on a `@triton.jit` function in
   the host lines?
4. (L52) Does a kernel's ledger fail on an event that no recorded filing or response gives (lr-10a's
   hand-written accept, an edited accept, a release with no filing, a decision on a finding that is not
   escalated), and on a file recorded with two sha256s? Do the events `tvr file` and `tvr respond` write
   pass? Does the library's ledger fail on a folder recorded twice and on an unlinted line after round 7,
   and does `tvr record` refuse `--no-lint` there? What does an edited sha256 in a ledger line show?
5. (L69) Is a gate stale after a change to `lean/lean-toolchain`, `lean/lakefile.toml`, or an imported
   library file that holds no constant of any cone?
6. (L70) Does the audit read a folder of `kernels/` with no `kernel.json`, and refuse a citation of a
   file that does not exist, but not a placeholder?
7. (L7) Does the tool run the system's git, bwrap and find when wrappers come first on `PATH`, and refuse
   a Mathlib branch other than `master` as `inputRev`?

**Method.** Part K compiles one small kernel with `lean -o` (not `lake build`) into the scratch folder, with one
`[rule]` or `[kernel]` field per case and the declarations above, runs the tree's claim reader on it (without its
`consistent` example: the kernel tests the kinds and the namespace check, not consistency; it covers every
instance, proves `satisfiable`, which the reader asks for since library issue L71, and proves `main` from its
`faithful` field), and calls the gate's `kinds_problems` on the output; it also lists, with Lean, the `TVA.Rules`
constants of type `Platform → Prop`, and compiles a control kernel of ordinary declarations for the namespace
check. Part D compiles five such kernels (each with the same `faithful` field) and computes their item digests
with `item_digests`. Part U calls the tool's functions on small inputs and on scratch copies of the tree (tools,
docs, lean without its build, `kernels/vec_add_f32`), with the module's root pointed at the copy; U3 makes a
passing gate by hand in a copy (the kernel does not build against this library), files a second round-2 filing
with `tvr file` and answers it with `tvr respond`. Part N runs the tool in a subprocess whose `PATH` starts with
a folder of `git`, `bwrap` and `find` wrappers that write a marker file, and asks Mathlib's upstream repository
for its refs.

**Result** (`out.txt`).

| Case | Result |
|---|---|
| K (L62) | the `TVA.Rules` constants of type `Platform → Prop` are 25: the 24 facts of `TVA.Rules.Num.Sm90` and `DivFull`; `Faithful`, `DivFull P`, `∀ d, d.ok → Exp2F32 P`, `MathTotal P .exp .binary32` and `DivRn P .binary64` pass; `DivRn P .binary32`, `DivAcc` with `DivFull`'s constants, lr-9a's `MathAcc P .exp .binary32 Set.univ 2^-24 2^-150` and `DotF8 P .f8e4nv .f8e5` are refused as shapes; `(K = K) → DivFull P` is refused (`K` outside `Faithful`); `DivFull P ∧ Exp2F32 P` is refused (no single fact); the `[kernel]` field passes |
| K (L63) | lr-10a's bounds `tol` (`private`), `Kernels._tol` and `_root_._tol` are each named as a constant of the kernel's own; the namespace check names `Kernels._tol` and `_tol` as declared outside the namespace; the instance `_hdiv` is the kernel's (`internal` false), Lean's `Inst._sizeOf_inst` is not; the control kernel (structures with defaults, recursion, a private definition, `simp`, `omega`, `decide`, `norm_num`) has no namespace problem, and only Lean's `_sizeOf_inst` instances |
| D (L68) | R2 (`2^-40`), R3 (`∀ _ : Nat`) and K2 (`… ∧ True`) each change exactly the edited assumption's digest; C (a proof edit) changes none (lr-10a's run: all three kept every digest) |
| U1 (L50) | vec-add's manifest passes (its seven functions are in the kernel or host lines); with the benchmark's host lines removed it fails on `benchmark` and its lambdas; lr-10a's G6 (`h = triton.jit(_h)`), an aliased decorator, `deco = triton.jit`, lr-10a's G5 (`@triton.constexpr_function`) and a lambda made a constexpr function each fail; the lambda in the host lines and a helper nested in the kernel pass; a `@triton.jit` helper in the host lines fails; `source.unused = [_h]` passes, an unused name of no function fails |
| U2 (L52) | vec-add's shipped ledger is consistent; lr-10a's G1 accept, round 2's accept edited to other digests, a release with no filing and a decision on a closed finding are each named, and the certificate's ledger condition fails; an edited recorded report re-recorded by an appended line is named (two sha256s) |
| U3 (L52) | `tvr file` (a second filing in round 2) and `tvr respond` write events that the derivation accepts, and every recorded file is as recorded |
| U4 (L52) | the tree's library ledger passes; lr-10b's P2 report edit is named; route 1 (an appended re-record) is named twice (two sha256s, a folder recorded again); route 2 (the original line's sha256 edited) passes the records step, and the earlier ledger digest no longer matches the ledger's first lines; lr-10a's G2 `--no-lint` for round 10 is refused, and the same line written by hand is named |
| U5 (L69) | a gate recorded now is not stale; lr-10a's G4 toolchain change, a `leanOptions` key, a comment in `lean/TVA.lean` (no constant) and one in `lean/TVA/Lemmas/Sm90.lean` each make it stale, naming the file; a gate without `build` is stale; `gate_mismatch` names a differing `build` |
| U6 (L70) | lr-10a's G8 `kernels/notes/readme.md` is named by the audit; of eight citations in a page, the two missing files are named, and a line number, an anchor, `round-N`, `<topic>`, `*.md` and a code span are not; lr-9a's `docs/evidence/conformance/cvt.md` in a Lean docstring is named; the tree cites no missing file |
| N (L7) | with wrappers first on `PATH`, `_git`, `system_tool` and `system_problems` use `/usr/bin/git` (2.52.0), `/usr/bin/bwrap` and `/usr/bin/find`, and no wrapper ran; git gets `PATH=/usr/bin:/bin`; the allowed `inputRev`s are `v4.35.0` (a tag) and `master` (a branch); lr-10a's G7 feature branch is refused by the audit and by the packages step before any fetch; Mathlib's pin is the tag `v4.35.0`'s commit and is reachable from `master`; `master` asked for as a tag is refused |

**Conclusions.**
1. A `[rule]` field states a fact the library states, by its name, applied to the platform, with no
   constant of the kernel's own however it is declared (L62, L63).
2. An assumption item's digest changes with its statement, and only with it (L68).
3. Every function of the source is classed by the prover; none escapes the map check by how `jit` is
   spelled (L50). Which functions the launch runs stays the reviewer's judgment.
4. Every event of a kernel's ledger follows from a recorded file, no file is recorded with two sha256s,
   and no library folder twice or unlinted after round 7 (L52). A ledger's own lines are held by the git
   history, or by the digest `release.json` keeps, not by the ledger.
5. A gate goes stale on every input of the build (L69).
6. The audit reads every file under `kernels/` and every citation of a page or Lean file (L70).
7. The release check runs only the system's programs, and only Mathlib's release tag or `master` (L7).
