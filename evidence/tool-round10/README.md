# The tool after round 10

Library evidence (not a kernel's), for library issues L7, L40, L48, L50, L51, L52, L53, L54 and L55
(round 10). It tests TVA's tool (`tools/tvr.py`, `tools/claim_reader.lean.in`), not the device: Lean
4.35.0, git 2.52, Python 3.13, 2026-10-28; no GPU and no Triton run. Part N needs the network (it
fetches from the packages' upstream repositories). Script: [run.py](run.py); output: [out.txt](out.txt).
The release check's own run is in the round-10 author's notes.

**Questions.**
1. (L7) Does the `packages` step refuse a Mathlib commit that is not the upstream ref the manifest's
   `inputRev` names, an inherited package that Mathlib's own manifest does not pin, and a checkout whose
   files differ from the commit's blobs, also when the user's git configuration installs a filter?
2. (L40) Does the hygiene refuse the string literals whose reading depends on the syntax before them
   (interpolated or plain), and the audit a library file that `lean/TVA.lean` does not reach?
3. (L48) Do the two models whose round-7 hashes collide get different item digests, while a proof edit
   and a renamed bound variable change none?
4. (L50) Does the map check fail when the ranges leave out a line of a `@triton.jit` function?
5. (L51) Does the gate refuse a source or a kernel Lean file with a line break other than a line feed?
6. (L52, L53) Do the ledgers, the certificate and the generated pages fail the release check's
   comparisons when they do not match, and is a gate stale after an edit of the source or the
   write-up, an edit whose old modification time is put back, or a change to the gate's code?
7. (L54) Does the audit refuse a toolchain that is not a release of Lean, and the release check a
   toolchain folder that is a link?
8. (L55) Does the gate refuse a `TVA.Rules.*` constant declared outside `lean/TVA/Rules`, and does its
   rule page reach the digest anyway?

**Method.** Part U imports the tree's `tools/tvr.py` and calls its functions on small inputs and on
scratch copies of the parts of the tree they read (with the module's root pointed at the copy).
Part N calls `upstream_problem` on the real manifest's packages and on lr-7a's local commit, and runs
`make_packages` for the tree's manifest into a scratch folder with a home directory whose
`.gitconfig` gives `*.lean` a smudge filter (lr-7b's probe), then changes the checkout. Part L
compiles four small kernels with `lean -o` (not `lake build`) into the scratch folder: A and B are
lr-7a's collision pair (`fun x => x + a`, `fun x => x + b`), C is A with a proof edit and a renamed
bound variable, D is A with `a + 1`; it runs the tree's claim reader on each and computes the item
digests with the tree's `item_digests`.

**Result** (`out.txt`).

| Case | Result |
|---|---|
| U1 (L40) | lr-7a's H1 and lr-7b's three strings (`s!"{"--"}"`, `s!"a{"/-"}b"`, and one before `open Nat in`) are refused, line 1 each; a plain reading would blank the `#eval` or `open` after each (so refusing is what makes the reader exact); a plain string with `--`, and braces in a docstring and a comment, are not refused; the library's code has nothing flagged |
| U2 (L40) | every library file of the tree is reached from `lean/TVA.lean`; an orphan `lean/TVA/Orphan.lean` in a copy is named |
| U3 (L50) | vec-add's manifest passes; lr-7a's M1 ranges `[[29, 51], [53, 54]]` fail on line 52 and `[]` on lines 29–54 (both passed in round 7); a helper `@triton.jit` function outside the ranges fails, and passes once `source.unused` names it; an unused name that is no jit function fails; `source.unused = [1]` fails `manifest` |
| U4 (L51) | lr-7a's M2: `splitlines` gives line 6 as `x = tl.load(x_ptr)`, line feeds give `tl.store(x_ptr, x)`, the statement Python's parser puts there; the map check refuses U+2028, a form feed, a lone carriage return and U+2029, and accepts CRLF; the kernel hygiene refuses a form feed in `VecAdd.lean` |
| U5 (L53) | a gate recorded now is not stale; after a comment in the source, a line in the write-up, a change to `VecAdd.lean` with its old modification time put back, an added Lean file, or a change to `tools/claim_reader.lean.in`, it is stale and names what changed; the tree's shipped vec-add gate is stale (no kernel-file digests, no library files) |
| U6 (L52, L53) | the shipped kernel ledger is consistent; a recorded file changed after recording, and a work file added to a recorded folder, are named; lr-7a's G1 certificate (not granted by the ledger) is refused, one that differs only in its time is not; a shipped gate whose kernel files differ is stale; in a copy, round 7 recorded with `tvr record`, a second record refused, a changed and an added work file named, rounds 1–6 named as unrecorded, and only the last round's unrecorded folders linted; generated pages that differ only in the gate's time are the same, a shipped REVIEW.md that says otherwise is named |
| U7 (L54) | `someone/lean4:v4.35.0-patched`, `leanprover/lean4:nightly` and a two-line file are refused, `leanprover/lean4:v4.35.0` accepted; a toolchain folder that is a symbolic link is refused, a real one and the user's accepted |
| U8 (L55) | `TVA.Rules.Num.borrowed` declared in `TVA.Lemmas.Num` fails `kinds`; `Faithful` and its equation lemma in the kernel's module do not; a cone holding it has `docs/rules/num.md` in its digest |
| U9 (L7) | TVA's inherited packages equal Mathlib's manifest at the pinned commit; lr-7a's Batteries at a local commit and from another URL are refused; `packagesDir` elsewhere is refused |
| N1 (L7) | Mathlib's pin is the tag `v4.35.0`'s commit; lr-7a's local commit and a hash as `inputRev` are refused; Batteries' pin is reachable from `main` (not its head); a commit of another repository is not |
| N2 (L7) | with lr-7b's smudge-filter home directory, `make_packages` on the tree's manifest (9 packages, 16 s) reports nothing and `Cli.lean` starts with `module`, as its blob: the user's configuration is not read; a changed and an added file are then named |
| N3 (L7) | lr-7a's route end to end: Mathlib pinned at a local commit is refused before any package is made |
| L (L48) | lr-7a's pair gets different `model` and `assumption:own` digests (the `conditions`, `spec` and `map` items, which do not reach the model, are equal); C (a proof edit and a renamed bound variable) has every digest equal to A's; D (`a + 1`) differs in `model` and `assumption:own` |

**Conclusions.**
1. The packages are Mathlib's: its commit is the upstream tag's, the others are what its own manifest
   pins, they come from their upstream URLs only, git reads no configuration of the user's, and every
   file is its commit's blob (L7).
2. The comment reader is exact wherever the hygiene passes, and every library module is built and
   replayed (L40).
3. A kernel constant's digest changes with its meaning and only with it: lr-7a's collision is gone,
   and proof edits and bound-variable names re-open nothing (L48).
4. The map's coverage covers every `@triton.jit` function the launch may run (L50), with every
   reader numbering lines alike (L51).
5. The ledgers, the certificate and the generated pages are held to the records, and a gate goes
   stale by contents and by the gate's code (L52, L53).
6. The toolchain is a release of Lean, not a link (L54), and a rule's page follows its name (L55).
