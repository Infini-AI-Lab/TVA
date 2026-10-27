# Library review, round 1, by lr-1

## Verdicts

- **L3** REOPEN — The decision is written into `TVA.Claim`, `Compiler.Faithful` and the rule page, and
  the rejections I tried are what the fix says. Shared memory, threads, a `tl.dot` with `K = 8`, a dtype
  CUDA lacks (`float8e4b8`), a failing `tl.static_assert` and grid `y = 65536` each raise on the host. In
  each case memory is unchanged and the device stays usable (`work/l3_rejections.out`, R1–R6). Three things
  are wrong, and each needs fixing.
  (1) The fix argues that "nothing goes wrong silently: the host gets the error". That is false on Triton
  3.8.0. The launcher skips a valid launch with no error when its program count is in `[2^31, 2^32]`
  (NEW below; W1, W2). The new docstring of `Behaviour`, "the outcomes of the launches that start", reads
  such a skipped launch as having no outcome, so a claim would be vacuous there without anyone knowing.
  Fix: define a rejection as a launch that is *reported to the host with an error* before any program
  runs. Say that a launch which returns without an error has an outcome even if no program ran (grid 0
  ends `ok` with memory unchanged, Z1). Then fix the NEW issue.
  (2) Some text still contradicts the decision, and should change or go:
  `TVA.Claim.Total` ("Every launch has some outcome, as on a real device");
  `TVA.Launch.Valid` ("Other limits (shared memory, registers) are part of the reviewed
  `Compiler.Faithful` instance");
  `docs/DESIGN.md` "No vacuous claims" ("The reviewer checks … that the other compile limits hold …, so
  that a claim covers only instances that compile and launch").
  (3) Reach is fine as far as I can check. The claim reader now hashes the docstrings of
  `TVA.Claim/Rules/Platform` (`claim_reader.lean.in`, `constJson`), so the next gate re-opens every item
  that reaches them, including any docstring edit the notes do not list.
- **L6** REOPEN — The drop is right for ordinary assertions. A holding `tl.static_assert` on constexpr,
  `.dtype` or `.shape` conditions leaves TTIR, TTGIR, LLVM IR and PTX identical (S1). A failing assertion,
  or one whose condition is not a compile-time bool, is rejected on the host (R5, S2). But the notes'
  "one that holds has no effect" is not true in general. `execute_static_assert`
  (`triton-3.8.0-code-generator.py:1617-1637`) evaluates the condition as an ordinary expression, so any
  operation inside it is emitted, and `x is not None` / `x is None` are compile-time bools (`:1083-1086`).
  On the H200, `tl.static_assert(tl.store(p, v) is not None)` and
  `tl.static_assert(helper(p) is None)`, where the `@triton.jit` helper stores, both compile, and their
  stores run (S3, S4). Dropping either line loses a store, and the kernel reviewer, who trusts §10,
  accepts the drop. Fix (TRANSCRIPTION.md §10 and protocol §7): `compile-check` applies only when the
  condition and the message use nothing but constexpr values, `.dtype`/`.shape`/`.numel`, arithmetic,
  comparisons and `is`. They may contain no call of a `@triton.jit` function and no `tl.*` operation.
  Otherwise the operations in the arguments are transcribed. It would also help to say that the
  assertion must hold wherever the front end visits it: inside runtime branches, and in each unrolled
  `tl.static_range` iteration.
- **L7** REOPEN — Text and Lean in the current tree pass. `tvr audit` prints "independent". No file outside
  `.venv`/`.lake`/`.git` contains the hashed name as a word or a home path outside TVA. The author's two
  probes (the name, and a home path) are caught. But the claimed enforcement has holes in exactly the
  places the user cares about, results and lemmas (`work/l7_audit_probe.out`, 14 of 21 probes missed):
  (a) `import_violations` matches only lines that start with `import`. `public import Aesop` in a `module`
  file elaborates in Lean 4.35 (`work/ImportSyntax.lean`) and passes the audit (I2), as does
  `meta import` (I3).
  (b) The lake check reads `[[require]]` only when `name` is its first key (K2). It ignores
  `[[lean_lib]]` entries whose `srcDir` lies outside TVA (K3), although that is how kernel libraries are
  added (`../kernels/<k>/lean`). With (a), that makes another project's Lean modules importable while the
  audit still prints "independent". The manifest check can catch K2 only once lake rewrites the manifest,
  and it never catches K3.
  (c) The text scan misses several things. It reads only `.lean/.md/.json/.jsonl/.py/.toml/.txt/.in` files
  (a `.log`, `.csv` or `.sh` evidence file is skipped, F1–F3), and never reads `lean/lakefile.toml` (K4) or
  `kernels/README.md` (F4). It misses the name as an identifier prefix or joined by `_` (N3, N4). Its
  `HOME_OK` prefix test accepts any `fpv/tva*` directory (P3), and it does not flag relative paths that
  climb out of TVA (P4).
  Fix: take the imports from Lean's own header parser (or match every `public`/`meta`/`all` form, and
  fail on a header it cannot read). Parse `lakefile.toml` with `tomllib`: the requires must be exactly
  `mathlib`, and every `srcDir` must resolve inside TVA. Scan every UTF-8 file outside `.venv`/`.lake`/`.git`,
  whatever its extension. Match the hashed names on identifier pieces too (split on `_` and on case). Test
  `HOME_OK` by whole path component, and flag `../` paths that resolve outside the root.
  The open consequence for `DivFull` (no TVA measurement until L4) is stated in its docstring and on
  `rules/num.md`. I did not judge `DivFull` itself.

## New issues

- **NEW** — `Launch.Valid` admits grids that Triton 3.8.0 never runs, so `Compiler.Faithful` is false for
  them. The launcher's guard `if (gridX * gridY * gridZ > 0)` is computed in C `int`
  (`third_party/nvidia/backend/driver.c:968-972`, identical to the installed `triton/backends/nvidia/driver.c`).
  When the program count is in `[2^31, 2^32]` it wraps to a value `≤ 0`, and the launch returns
  normally with no program run. On the H200, grid `(2^30, 2, 1)` and grid `(2^16, 2^15, 2)` return at once
  with no error and memory unchanged, while `(2^30, 1, 1)` runs and writes (`work/l3_rejections.out` W1, W2, C2).
  Both grids satisfy `Launch.Valid` (`work/GridValid.lean`, checked with Lean). So for any kernel that
  writes, the device's `ok init` is not an outcome of the model. Fix: add `L.size < 2^31` to `Launch.Valid`
  (docstring, SEMANTICS.md "Launches", the rule page), pin `driver.c` in `primitives/`, and put this test in
  `evidence/`.
- **NEW** — The drop `no-op:print` has the same hole as L6. `tl.static_print`, `tl.device_assert` and
  `tl.device_print` evaluate their arguments as ordinary expressions, and with `debug` off a
  `tl.atomic_add` inside any of them still runs (grid 4 gives a counter of 4 in all three cases;
  `work/l6_static_assert.out` S5). The drop should require pure arguments, or the argument operations
  should be transcribed. A Python `assert` statement lowers to `device_assert` (`code-generator.py:1353-1356`)
  and is not in the §10 table.
- **NEW** — `docs/rules/num.md` "What the reviewer judges" still asks the reviewer to check "Truth on the
  device" ("Ask for the PTX the operation lowers to … or for a test"). Kernel reviewers read that page, and
  protocol §1–§2 forbid them that check. It should address library review, or go (with L4).

## Re-ran

- `python3 -I tools/tvr.py audit` on the tree, which printed "independent", and on 21 mutated copies in a
  scratch directory (`work/l7_audit_probe.py`, output `.out`). This includes the author's probes: the name
  (N1, N2, N5) and home paths (P1, P2), all caught.
- `work/l3_rejections.py` on GPU 0 (Triton 3.8.0, torch 2.11.0+cu130, H200 NVL, sm_90, 2026-10-27). R1–R6 are
  host rejections; Z1 is a zero grid; C1 and C2 are controls; W1 and W2 are silent skips.
- `work/l6_static_assert.py` on GPU 0, same setup: S1 compares IR and PTX with and without the
  assertions; S2–S4 cover the condition kinds; S5 covers the `no-op:print` drops.
- Lean (`lake env lean`, my files only): `work/GridValid.lean` (`Launch.Valid` and `size` of W1, W2);
  `work/ImportSyntax.lean` (`public import` elaborates).
- Read the L3 diff against commit `089d4c5`: `Claim.lean`, `Rules/Compiler.lean`, `rules/compiler-faithful.md`
  and `claim_reader.lean.in`. Read `DROPS`/`NOTE_REQUIRED` and the independence code in `tools/tvr.py`.
  Read the pinned `code-generator.py` (`visit_Compare`, `visit_Call`, `execute_static_assert`,
  `static_executor`, `visit_Assert`). From the Triton 3.8.0 tree, read `compiler.py`
  (`_init_handles`: shared memory, threads) and `driver.c` (`gpuAssert`, `_launch`). Read
  `nvidia/backend/compiler.py` (`min_dot_size`, fp8 dtypes).
- The files I reviewed, by sha256 prefix: `Claim.lean` 4f393ca16257, `Rules/Compiler.lean` fe3e43e008ee,
  `Sem/Launch.lean` 35d6762299fd, `rules/compiler-faithful.md` d4cef8e005fd, `TRANSCRIPTION.md`
  755c471d690c, `REVIEW_PROTOCOL.md` ff76944ce7af, `tvr.py` bd61911eabdb.

## Not checked

- Reach on real kernel items: the coordinator cleared `kernels/` during this review. I checked the
  claim reader's hashing instead, and ran no `tvr check` (it runs `lake build`).
- Whether lake, once it re-resolves, records K2 in `lake-manifest.json` (that needs a lake run). Whether
  `meta import` (I3) elaborates in Lean 4.35; I verified only `public import`.
- The other rejection examples on the rule page: a descriptor block under 16 bytes, a gather or scatter of
  fewer than 8 rows, and register exhaustion beyond the threads limit. Also host-side argument
  rejections (CPU tensors, missing allocator).
- Whether the review records were fully redacted: they were deleted with the kernels.
- `DivFull`'s constants (L4).
