# Library batch, round 1: author's notes

Issues: L3, L6, L7 ([ISSUES.md](../../ISSUES.md)). 2026-10-27.

## L3: a launch rejected before it runs is outside the claim

The user's decision of 2026-10-27, written into the library.

- **What changed.**
  - `TVA.Claim` (module docstring): a launch that Triton or the driver rejects before any
    program runs has no outcome. The examples are resources, `tl.dot` minimum sizes and missing
    dtypes. So a claim says nothing about whether a kernel compiles.
  - `TVA.Claim.Outcome.fault`: "a fault of a launch that runs" (it no longer lists "launch
    failure").
  - `TVA.Claim.Behaviour`: "the outcomes of the launches that start".
  - `TVA.Rules.Compiler.Faithful` (docstring): the sentence asking the reviewer to accept only
    instances that compile is replaced by the statement above.
  - [rules/compiler-faithful.md](../../../rules/compiler-faithful.md): "What it says" states it.
    The item "compile and launch limits" under "What it does not cover" becomes "whether the
    kernel compiles". The reviewer's check list no longer asks that each instance compiles.
- **Why it is sound.** `Meets` constrains only the outcomes `K` has. A rejected launch runs no
  program and changes no memory, and the host gets the error, so leaving it out loses no silent
  failure. A claim's coverage may include instances that never compile: the claim promises
  nothing there, and the write-up should say which instances the host actually uses.
- **Lean.** Only docstrings changed; no definition did. `exists_total_faithful` and the
  consistency lemmas are untouched.
- **Constants whose meaning changed.** `TVA.Claim.Behaviour`, `TVA.Claim.Outcome`,
  `TVA.Claim.Outcome.fault` and `TVA.Rules.Compiler.Faithful`. The claim reader now hashes the
  docstrings of `TVA.Claim.*`, `TVA.Rules.*` and `TVA.Platform.*` with their terms
  ([REVIEW_PROTOCOL.md](../../../REVIEW_PROTOCOL.md) §4). So every kernel item that reaches
  them re-opens.

## L6: a drop reason for `tl.static_assert`

- **What changed.** The drop reason `compile-check`, whose note names the condition of `Inst.ok`
  that makes the assertion hold. It is in the protocol (§7), the tool's drop list
  (`tools/tvr.py`, `DROPS`, with a note required), and
  [TRANSCRIPTION.md](../../../TRANSCRIPTION.md) §10.
- **Why.** A `tl.static_assert` that fails stops compilation, so it is a rejection before the
  launch (L3). One that holds has no effect. Neither changes a launch that runs.
- **Constants.** None (protocol and tool only).

## L7: TVA cites no other project

- **What changed.**
  - Every mention of another project was removed from the library's Lean docstrings, the docs,
    the kernels' files and the review records. The review records were redacted with a ledger
    note.
  - `DivFull` had cited another project's measurement. It now says that TVA's own measurement
    is library issue L4.
  - The tool enforces independence:
    - `tvr audit` (the whole project) and the gate's `independence` check (one kernel, plus the
      library);
    - Lean files import only `TVA`, the kernel's own modules and `Mathlib`;
    - `lean/lakefile.toml` requires only Mathlib, and the manifest has no other direct package;
    - no text names another project (their names are kept as sha256 hashes, so TVA does not
      name them), or a home path outside TVA. The Lean toolchain (`~/.elan`) and Triton's own
      compile cache (`~/.triton`) are allowed;
    - `tvr file` and `tvr respond` refuse a filing, a report, a work file, a response or notes
      that do.
  - The prompts tell every role that TVA and Mathlib are the only sources of results and lemmas,
    and that evidence is TVA's own.
- **Why.** Kernel reviewers trust the library, so whatever the library rests on is trusted
  unchecked. A result, lemma or measurement taken from another project would enter every claim
  without TVA's review. With imports limited to TVA and Mathlib and the text checked, everything
  a claim rests on is TVA's own, or Mathlib's mathematics.
- **Evidence.** `python3 -I tools/tvr.py audit` prints "independent". A test text naming another
  project and a path into it is caught (both kinds).
- **Open.** `DivFull` is stated without TVA's own evidence until L4 is done.
- **Constants whose meaning changed.** `TVA.Rules.Num.DivFull` (docstring: its evidence).
