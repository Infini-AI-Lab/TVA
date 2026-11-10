# Library batch, round 42: author's notes

Issues L190 to L194 ([ISSUES.md](../../ISSUES.md)). L190 and L191 are in `tools/tvr.py` and the pages that describe the
release check and the gate (`docs/REVIEW_PROTOCOL.md` §2, `docs/LIBRARY_REVIEW.md`, `docs/LIBRARY.md`, `README.md`).
L192 and L193 are in the proof kit (`lean/TVA/Lemmas/Tile2.lean`, `lean/TVA/Lemmas/IntW.lean`). L194 is in
`docs/PROVER_GUIDE.md` and `docs/DESIGN.md`. No definition or statement of `TVA.Lang`, `TVA.Sem`, `TVA.Claim` or
`TVA.Rules` changed, no semantics, rule or device fact changed, and no `simp` lemma or instance was added.

Checks, all from `lean/` after `lake build TVA` (green), with `tvr library-check` passing (`leanchecker TVA`):
- [work/probes.lean](work/probes.lean), output [work/probes.out](work/probes.out): the probes of L192 and L193
  below, and `#print axioms` of the two changed lemmas (none).
- The coordinator's constraint for this round: the 41 fixed claims stay unaffected.
  [work/claims.sh](work/claims.sh), output [work/claims.out](work/claims.out): every `claims/<k>/Claim.lean`
  elaborated with `lake env lean` against the new library; all 41 exit 0 with no error line. `git diff --stat --
  claims/` is empty (the last line of the output): nothing under `claims/` was edited.
- The round-41 probes (author's `work/probes.lean`, lr-41b's `work/Probes.lean` and `work/UnitStride.lean`), which
  use `Tile2` and the integer cast, still elaborate with no error, unchanged.
- No local kernel uses `Tile2.col_toInt` or `Tile2.row_toInt` (grep over `kernels/*/lean`), so none stops
  building. `tools/tvr.py` changed, so every local kernel's gate is stale until its next `tvr check`. I ran no tvr
  command that writes into `kernels/` (`index`, `new`, `check`, `render`): L191's test ran in a scratch copy
  outside TVA, and `kernels/vec_add_f32/gate.json` is still dated 2026-11-05.

## L190: The fixed claims elaborate

- **What changed.** `tvr release` has a new step, `claims`, right after `build` (`cmd_release`, `tools/tvr.py`).
  If the build passed, it runs `lake env lean ../claims/<k>/Claim.lean` from the copy's `lean/` in the sandbox,
  once for each `claims/<k>/Claim.lean` of the copy (eight at a time). The step fails if any run exits non-zero or
  prints an error line, or if there is no claim. If the build failed, the step is recorded as failed (`not run`).
  The step is listed in `docs/REVIEW_PROTOCOL.md` §2 ("There it runs, in order"), in `docs/LIBRARY_REVIEW.md`'s
  `gate` row, and as a new item 7, "The fixed claims", in its reviewer checklist ("What a library reviewer
  checks"). It is also mentioned in `docs/LIBRARY.md` and `README.md`.
- **Why.** Round 39's `simp` lemmas broke `claims/layer_norm_fwd_f16/Claim.lean` (L189), and no step noticed:
  the release check built `TVA` but elaborated no claim. A claim holds no proof, so the step checks only that each
  still elaborates, not that any holds. The check still checks no claim in L160's sense. The checklist item also
  asks the reviewer whether the batch keeps each claim's meaning, which elaboration alone cannot show.
- **Evidence.** [work/claims.out](work/claims.out): the same check run by hand outside the sandbox. All 41 claims
  elaborate against this round's library with exit 0 and no error. The release check's own run (below, in the
  final report) runs the step in the sandbox.
- **Constants whose meaning changed.** None (`tools/tvr.py` and documentation only).

## L191: Two local kernels with the same module root

- **What changed.** Of the issue's two options, I took the refusal: each kernel's module root must be its own.
  The new `root_problems(k, mod)` (`tools/tvr.py`) is called by `manifest_problems`, so by the gate's first check,
  `manifest`, and by `tvr new` before it creates a folder. It refuses a module whose root (its first component),
  compared ignoring case, is in either of these:
  - the root of another local kernel's module;
  - a root in `reserved_roots()`:
    - `TVA` and `Mathlib`;
    - the fixed list `RESERVED_ROOTS`: `Init`, `Std`, `Lean`, `Lake`, the toolchain's other top-level modules,
      `Batteries`, `Aesop`, `Qq`, `ProofWidgets`, `Plausible`, `ImportGraph`, `LeanSearchClient`, `Cli`,
      `Archive`, `Counterexamples`;
    - each top-level `.olean` of the pinned toolchain's `lib/lean`;
    - each top-level module of a package in `lean/.lake/packages` (`<root>.lean`, or a folder named as a module
      is).

  A refused manifest fails the gate before anything is built. Two kernels that share a root both fail, and so do
  the copy and its original. `docs/REVIEW_PROTOCOL.md` §2 ("Lake") says which option was taken and why. The
  comment on `kernel_lakefile` points to it.
- **Why.** `lean/.lake/tvr/lakefile.toml` declares one library per kernel, rooted at its module's first
  component. Two libraries with one root make `lake build <module>` compile one file for both, into one `.olean`.
  The claim reader and `leanchecker` then read another kernel's build. lr-41a showed this with a broken copy of
  `vec_add_f32` that passed all 13 checks. A root shared with Lean or a pinned package has the same effect on the
  build. Ignoring case also covers a file system that ignores case.
- **Evidence.** [work/same_root.py](work/same_root.py), output [work/same_root.out](work/same_root.out). Like
  lr-41a's test, it copies what TVA ships and the local `vec_add_f32` into an empty folder outside TVA and runs
  `tvr check` there only. Results:
  - A: `vec_add_aa`, a copy of `vec_add_f32` with a theorem that does not elaborate appended. It keeps the module
    `VecAdd`. Both `vec_add_aa` and `vec_add_f32` now fail at `manifest`, and each names the other.
  - B: the same copy with its module renamed `VecAddAa`. It passes `manifest`, and then `build` fails on its own
    appended theorem: its own file is compiled.
  - C: the modules `Aesop.VecAdd`, `std.VecAdd` and `TVA.VecAdd` are refused at `manifest`.
  - D: with the copy removed, `vec_add_f32` passes again with its digest `7be6709b6e4f426e`, as before.
- **Constants whose meaning changed.** None (`tools/tvr.py` and documentation only).

## L192: `Tile2`'s unit-stride terms

- **What changed.** `Tile2.col_toInt` and `Tile2.row_toInt` (`rows[:, None]` and `cols[None, :]` with no stride)
  now take rows and columns of any form. The hypothesis is `hr : ∀ i, (rows.get1 i).toInt = ((ρ i : Nat) : Int)`
  (or `hc` with `κ j`), and the result is `ρ i * 1` (or `κ j * 1`), on every lane. They need no bound, since
  nothing is multiplied. The docstrings of the two lemmas, of `add_toInt_on` (which now offers
  `fun i j _ => row_toInt hc i j` as the column term too), of the module and `docs/LIBRARY.md`'s `Tile2` row say
  so.
- **Why.** L184 generalised the strided terms but not these two. `% N` columns used with no stride
  (`rows[:, None] * s + (offs % N)[None, :]`) needed a hand proof (lr-41b's U1).
- **Evidence.** [work/probes.lean](work/probes.lean):
  - P1 is lr-41b's U1: `% N` columns through `row_toInt` itself.
  - P2: `% M` rows in `int64` through `col_toInt`.
  - P3: an old `r0 + i` caller gets back the `(r0 + i) * 1` form unchanged.
  - P4: the whole offset `rows[:, None] * a + (offs % N)[None, :]` on an enabled lane, by `add_toInt_on` with
    `colMul_toInt_on` and `row_toInt`.

  All elaborate. No caller in the library, the claims or the local kernels names `r0` or `c0` of these lemmas.
- **Constants whose meaning changed.** `TVA.Tile2.col_toInt`, `TVA.Tile2.row_toInt` (generalised statements); the
  docstrings of `TVA.Tile2.add_toInt_on` and of the module `TVA.Lemmas.Tile2`.

## L193: The `safe_convert_iff` example

- **What changed.** The example in `lean/TVA/Lemmas/IntW.lean`'s section "The integer cast (library issue L185)"
  is now `safe_convert_iff fun _ _ => ConvRel.iff_cast_i32_i64`. The section says what it proves:
  `Safe E R W Q m (tl.convert .i32 .i64 x >>= k) ↔ Safe E R W Q m (k x.castInt)` at any shape, with `y` taken
  from the goal.
- **Why.** The old form, `(y := x.castInt)`, is elaborated before `d'` is known and does not elaborate (lr-41b's
  V1 and V3).
- **Evidence.** [work/probes.lean](work/probes.lean): P5 is the docstring's form at any shape `s`, and P6 is the
  same at shape `[n]`, lr-41b's V1 shape. Both elaborate.
- **Constants whose meaning changed.** None (a module docstring of `TVA.Lemmas.IntW`).

## L194: Guidance that points into `kernels/`

- **What changed.**
  - `docs/PROVER_GUIDE.md` (formerly line 34) now gives `claims/vec_add_f32/Claim.lean`, which ships, as the
    example of the model, the specification and the assumptions (with `consistent`; it holds no proof). It says
    that a bundle with all four parts, `theorem main` included, is `kernels/vec_add_f32/lean/VecAdd.lean`, but only
    in a working tree that holds the local `kernels/`, which TVA does not ship.
  - `docs/DESIGN.md` (formerly line 124) says the same of that bundle.
  - The layout in `docs/DESIGN.md` marks `../kernels/` as local only (git-ignored, not shipped), and lists
    `../claims/` (the fixed claims, with no proof).
- **Why.** `kernels/` is local since L187 and L188. The two pages offered a file that does not ship as the worked
  example.
- **Evidence.** A grep of `README.md`, `docs/*.md` and `docs/rules/*.md` for a path into a named kernel folder
  (`kernels/<name>/…`) now finds only these two sentences, and each says the file is local.
- **Constants whose meaning changed.** None (documentation only).
