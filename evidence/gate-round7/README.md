# The gate and the audit after round 7

Library evidence (not a kernel's), for library issues L38, L39, L40, L41, L42 and L43 (round 7). It
tests TVA's tool (`tools/tvr.py`), not the device: Lean 4.35.0 (with `leanchecker` from the same
toolchain), Python 3.13, git 2.52, 2026-10-28; no GPU and no Triton run. Script: [run.py](run.py);
output: [out.txt](out.txt). The release check's own evidence is in
[evidence/release-check](../release-check/README.md).

**Since round 35** (library issues L160, L170) the release check recomputes no kernel's gate:
Conclusion 5's last sentence records a step that no longer runs.

**Questions.**
1. (L38) Does the gate refuse a `kernel.json` whose module or namespace is not a plain name, before
   anything is spliced into the claim reader?
2. (L39) Does the map check count as code every line that holds code, a triple-quoted argument and
   lines between triple quotes in comments included, while a docstring stays exempt?
3. (L40) Does the library hygiene flag a meta command that a naive comment stripper would hide (after
   a string, a character literal, a raw string or a `«…»` name holding `--`), and not the library's
   prose? Does the lake check refuse package-wide options beyond the allowlist?
4. (L41) Does the audit fail on a shipped file inside a folder the text scan skips?
5. (L42) After a library change, is the gate stale, the certificate lapsed and a filing refused, and
   do the items that reach the changed file re-open after `tvr check`?
6. (L43) Does a `TVA.Rules` module in a subfolder get its ancestor's rule page in the digests, and
   does the audit name a module that has no page above it?

**Method.** Part U imports the tree's `tools/tvr.py` and calls `code_lines` (against the round-6
coverage loop, copied into the script), `manifest_problems`, `_uncommented` with `LIBRARY_HYGIENE`, and
`library_hygiene_problems` on the library. Part C copies the tree with its git metadata to a fresh
folder (the Lean build copied, Mathlib's packages linked, as in
[evidence/gate-digests](../gate-digests/README.md)), changes one thing at a time there, and runs the
copy's `tvr`; for C5 it first brings `kernels/vec_add_f32` up to date in the copy (the same update as
gate-digests) and clears its review record there, so that one reviewer's acceptance grants a
certificate.

**Result** (`out.txt`).

| Case | Result |
|---|---|
| U1 K1–K5 | round 6 missed lr-6a's and lr-6b's lines (a triple-quoted argument; code between `"""` in two comments; `; s = """"""`; a two-line string argument); round 7 counts all of them as code, and exempts the docstring (K3) and a bare string statement (K5 line 3) |
| U1 | a source that does not parse gives no code lines and a reason (the map check then fails) |
| U2 | lr-6a's injected namespace, a module with a line break, a namespace with a space and a source file outside the kernel folder are each refused; the real manifest passes |
| U3 | a meta command after `"--"`, `'"'`, `r#"a"--"#` or `«a--b»` is flagged; a nested comment, a docstring naming `syntax` and `macro`, and a line comment naming `run_cmd` are not; the library as it is has nothing flagged |
| C0 | the copy passes the audit; the shipped gate of `vec_add_f32` (ok, with a granted certificate, from 2026-10-27) is stale: its snapshot records no library files |
| C1 | `TVA.Rules.Num.Exp2` passes the audit, and an item that reaches it holds `docs/rules/num.md`; `TVA.Rules.Extra.Y` fails the audit, named |
| C2 | `leanOptions` `debug.skipKernelTC`, `trace.*` and a string value fail the audit; `maxHeartbeats`, `linter.unusedVariables` (a nested TOML table) and `pp.all` pass |
| C3 | an untracked note in `docs/.lake/` is ignored by git (not shipped) and the audit passes; force-added (`git add -f`), it ships and the audit fails; the same note in `docs/` fails the scan (control) |
| C4 | `tvr check` with lr-6a's namespace fails at `manifest`, and no claim reader is written |
| C5 | gate passes, one reviewer accepts all, the certificate is granted; after a comment is added to `lean/TVA/Lang/Ops.lean` and `tvr index` runs, the certificate is gone, `REVIEW.md` and `kernels/README.md` mark the gate stale and name the file, and a further `tvr file` is refused; after `tvr check`, the four items that reach `Ops.lean` are STALE (each naming the file) and `conditions`, which does not reach it, stays accepted |

**Conclusions.**
1. Nothing the prover writes in `kernel.json` reaches the claim reader unless it is a plain name (L38).
2. The map check's code lines are Python's own (L39).
3. The library's code is checked for meta code the way a kernel's is, with Lean's literals read as
   Lean reads them, and `lean/lakefile.toml` can set no option that changes what Lean checks (L40).
   The replay of the library (`leanchecker TVA`) is a step of the release check
   ([evidence/release-check](../release-check/README.md)).
4. What ships is what the scan reads (L41).
5. A library change makes every gate it reaches stale until `tvr check` runs again; the certificate
   lapses meanwhile, and the tool refuses filings against the stale gate (L42). The release check
   also compared each recomputed gate with the shipped one (until round 35; no longer run).
6. Rule pages follow modules into subfolders (L43).
