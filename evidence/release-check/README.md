# The release check: TVA built and checked with nothing but what it ships

Library evidence (not a kernel's), for library issues L36 and L7 (rounds 6 and 7), L40, L41 and L42
(round 7). It tests TVA's tool (`tools/tvr.py release`), not the device: Lean 4.35.0, git 2.52, `bwrap`
(bubblewrap 0.6.3) and Python 3.12 from the system, 2026-10-28; no GPU and no Triton run. Script:
[run.py](run.py); output: [out.txt](out.txt).

**Question.** TVA will be released on its own (user, 2026-10-28), so its checks must pass with
nothing outside it. Does `tvr release` pass on what TVA ships, and fail on a dependency on anything
outside what TVA ships, Mathlib's pinned source, the Lean toolchain and the system, also one that the
imports check and the text scan cannot see? In round 6, lr-6a found two such routes: the sandbox hid
only `/home` and `/tmp`, and it borrowed Mathlib's package trees, whose ignored files its `packages`
step did not see.

**The sandbox since round 7** (protocol §2). It is built from an allowlist: its root is empty, and it
shows only the files TVA ships (copied into an empty home directory at TVA's path), Mathlib and its
dependencies made afresh in the copy from their pinned commits (a new git repository fetches exactly
the commit and checks it out; no working-tree file or build output is used) and built there, `/usr` and
`/etc` (and the links `/bin`, `/sbin`, `/lib`, `/lib64`) read-only, checked to hold nothing writable by
the user or owned by a regular user, the pinned Lean toolchain read-only, the system's Python (the tool
runs on `/usr/bin/python3.12 -I -S`), `/dev`, `/proc`, an empty `/tmp`, and no network. There it runs
`lake build TVA`, `tvr library-check` (the library's hygiene, and `leanchecker TVA`), the gate on every
kernel (compared with the kernel's shipped `gate.json` and certificate), `tvr lint` on the current
records and `tvr audit`. A shipped file inside a folder the text scan skips fails its first step.

**Method.** The script runs the check four times, one at a time, each in a fresh folder of a scratch
directory outside the home directory (each run builds Mathlib: about 13 GB, removed after the run, and
17–21 minutes here). The copies
are made from the tree with its git metadata, and the tree is never written.

| Run | Tree | What it adds |
|---|---|---|
| R0 | TVA as it is | nothing |
| R1 | a copy | (a) `tvr audit` first reads a file in a fresh folder of the system's second temporary directory, at a path computed at run time (lr-6a's route); (b) `tvr audit` first puts the site-packages folder of the Python that runs the script on `sys.path` and imports a package that only that Python has (lr-6a: "`-I -S` holds only until the code extends `sys.path`"); (c) a note naming a sibling of TVA, force-added at `docs/.lake/note.md` (L41); (d) `kernels/vec_add_f32` brought up to date and gated in the copy (as in evidence/gate-digests), then a comment added to `lean/TVA/Lang/Ops.lean`, so that the shipped `gate.json` is stale (L42) |
| R2 | a copy whose Lean packages are clones of the tree's | Mathlib's clone holds `Mathlib/__pycache__/Lemmas.lean`, a file its `.gitignore` ignores (lr-6a's route), and `lean/TVA.lean` imports `Mathlib.__pycache__.Lemmas` |
| R3 | a copy | `kernels/vec_add_f32` brought up to date and gated in the copy, so its shipped `gate.json` is current (the control for R1 (d)) |

**Result** (`out.txt`).

| Step | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| outside the sandbox | | the copy's audit passes with probes (a) and (b) in place ((b) imports `IPython`); the kernel's gate passes in the copy | the file is ignored by Mathlib's `.gitignore`; the round-6 `packages` test sees nothing; TVA's import check accepts the import; the audit passes | the kernel's gate passes in the copy |
| shipped | pass | **fails**: `docs/.lake/note.md` ships inside a folder the scan skips | pass | pass |
| packages | pass | pass | pass (made from the commit: the ignored file is not there) | pass |
| sandbox | pass | pass | pass | pass |
| build (`lake build TVA`) | pass | pass | **fails**: "bad import 'Mathlib.__pycache__.Lemmas'" | pass |
| library (hygiene, `leanchecker TVA`) | pass | pass | not run (the build failed) | pass |
| gate `vec_add_f32` | fails: its own build, as in the tree; and **the shipped `gate.json` and certificate are stale** | **fails**: the recomputed gate passes, but the shipped `gate.json` is stale (four items) | fails: the build | **pass**: the recomputed gate equals the shipped one |
| lint (current records) | regular | regular | regular | regular |
| audit | independent | **fails**: "probe failed: (a) a file outside TVA; (b) a module of the user's Python" | independent | independent |
| time; the packages' `.olean` files | 21 min; 54186, digest `3ae1aa797d96970a` | 19 min; 54186, digest `3f335c460304e5f1` | 17 min; 54186, digest `3f335c460304e5f1` | 19 min; 54186, digest `3f335c460304e5f1` |

Each run shipped 508 files (509 in R1, with the force-added note); the sandbox showed only the copy,
`/usr` and `/etc` (with nothing in them writable by the user or owned by a regular user), the links
`/bin`, `/sbin`, `/lib`, `/lib64`, the Lean toolchain `leanprover/lean4:v4.35.0`, `/dev`, `/proc` and an
empty `/tmp`, and the tool ran on `/usr/bin/python3.12`. R1, R2 and R3 ran in one invocation of the
script, one after the other, and R0 last, in a second one. The `.olean` files of R1, R2 and R3 (copies
named `copy1`, `copy2`, `copy3`, so at paths of the same length in the sandbox) were identical bit for
bit; those of R0 (TVA at its own path, `fpv/tva`) had another digest. All build files together had a
different digest in every run.

**Conclusions.**

1. What TVA ships builds, re-checks, lints and passes the audit with nothing but itself, Mathlib's
   pinned source, the Lean toolchain and the system (R0, R3). The one failing step of R0 is
   `vec_add_f32`'s gate, which fails at the kernel's own build and on its stale shipped gate; brought up
   to date, the kernel passes and so does every step (R3).
2. A dependency on a file outside TVA at a path computed at run time, and on the user's Python
   packages, fails the release check (R1 (a), (b)). The probe's file was in the system's second
   temporary directory; any other place outside the allowlist is absent from the sandbox in the same
   way, since its root is empty but for the allowlist. This closes lr-6a's first route (round 6 showed
   every top-level folder but `/home` and `/tmp`, and the Python prefix).
3. A module in a file Mathlib's checkout holds but its commit does not fails the build, though the
   import check, the round-6 `packages` test and the audit pass (R2): the packages are made from their
   pinned commits and built in the sandbox, so neither an ignored source file nor a borrowed build output
   reaches TVA. This closes lr-6a's second route.
4. A shipped file the text scan would skip fails the check (R1 (c), L41), and so does a shipped gate or
   certificate that no longer matches the gate recomputed from what ships (R0, R1 (d); R3 is the
   control, L42).
5. Builds of Mathlib from the same pinned source gave the same `.olean` files bit for bit in three
   runs (R1, R2, R3) but not in a fourth (R0), and Lake's build folders as a whole differed every
   time. **Corrected in round 10** (library issue L36): the round-7 notes read R0's other digest as an
   effect of its path (TVA at its own path, `fpv/tva`). Library review round 7 (lr-7a,
   `docs/library-review/round-7/lr-7a/work/build_digest_watch.out`) ran a fifth build with TVA at that
   same path and got R1–R3's digest, `3f335c460304e5f1`; so the `.olean` files do not depend on the
   path, 4 of 5 builds agree, and R0 is an outlier of unknown cause. The release check still builds
   Mathlib from source in each run (both round-7 reviewers judged that this holds); a pinned build
   would need that outlier explained first. Since round 10 the packages come from their upstream
   repositories, checked against Mathlib's own refs and manifest (L7,
   [evidence/tool-round10](../tool-round10/README.md)), not from the working tree's checkout as in R0–R3.
