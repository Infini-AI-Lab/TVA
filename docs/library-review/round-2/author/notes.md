# Library batch, round 2: author's notes

Issues: L3, L6, L7, L8, L9, L10, L11 ([ISSUES.md](../../ISSUES.md)). 2026-10-27. The library
builds (`lake build TVA`), and every `TVA` constant depends only on `propext`,
`Classical.choice` and `Quot.sound`. `python3 -I tools/tvr.py audit` prints "independent".

**Kernel that no longer builds:** `kernels/vec_add_f32`. Its `launch_valid` destructures
`Launch.Valid`, which has a new conjunct (L9). The prover can prove it with the new
`Launch.valid_grid1`. I did not edit the kernel folder or run `tvr check`.

## L3: a launch rejected before it runs is outside the claim

- **What changed.**
  - A *rejection* is now defined as a launch that Triton or the driver reports to the host with
    an error before any program runs. A launch that returns without an error has an outcome,
    even if no program ran: a grid with a zero dimension ends `ok` with memory unchanged.
  - Where it is stated:
    - `TVA.Claim` (module docstring), `TVA.Claim.Behaviour` and `TVA.Claim.Outcome.ok`;
    - `TVA.Rules.Compiler.Faithful` (docstring);
    - [rules/compiler-faithful.md](../../../rules/compiler-faithful.md) ("What it says", "What it
      does not cover");
    - [SEMANTICS.md](../../../SEMANTICS.md) "Launches", [REVIEW_PROTOCOL.md](../../../REVIEW_PROTOCOL.md)
      §2 and §6, [REVIEWER_GUIDE.md](../../../REVIEWER_GUIDE.md), the kernel reviewer's prompt and
      [LIBRARY_REVIEW.md](../../../LIBRARY_REVIEW.md).
  - Stale text removed or rewritten:
    - `TVA.Claim.Total` no longer says "every launch has some outcome, as on a real device". It
      says why a consistency witness must be total, and that the device itself may reject
      launches;
    - `TVA.Launch.Valid` no longer calls the other limits "part of the reviewed
      `Compiler.Faithful` instance". They are rejections with an error;
    - [DESIGN.md](../../../DESIGN.md) "No vacuous claims" no longer asks the reviewer to check
      compile limits, and the decision table no longer calls a rejected launch a fault.
  - The rule page's claim that "nothing goes wrong silently" now holds: it is said of *valid*
    launches, and the one silent skip found is outside `Launch.Valid` (L9).
- **Why.** The library reviewer showed that a launch can return without an error and run no
  program (L9). Reading "no outcome" as "the launch did not start" would make a claim vacuous
  there without anyone noticing. With rejections defined by the error the host gets, every
  launch the host sees succeed has an outcome, and `Meets` constrains it.
- **Evidence.** [evidence/launch-outcomes](../../../../evidence/launch-outcomes/README.md):
  - A1–A8: shared memory, threads, `tl.dot` with `K = 8`, `float8e4b8`, a failing
    `tl.static_assert`, and grid dimensions over CUDA's limits each raise on the host. No
    program runs, memory is unchanged, and the device stays usable;
  - B1, B2: grids with a zero dimension return without an error and run no program.
- **Constants whose meaning changed.** Docstrings only: `TVA.Claim.Behaviour`,
  `TVA.Claim.Outcome.ok`, `TVA.Claim.Total`, `TVA.Rules.Compiler.Faithful`, and
  `TVA.Launch.Valid` (whose definition also changes, L9). In `vec_add_f32` they re-open
  `item:spec` (it reaches `Behaviour`, `Outcome.ok`) and `item:assumption:faithful` (it reaches
  `Faithful`, `Outcome.ok`, `Launch.Valid`). `Total` is in no item's cone.

## L6: a drop reason for `tl.static_assert`

- **What changed.**
  - [TRANSCRIPTION.md](../../../TRANSCRIPTION.md) §10 defines a **pure expression**: names,
    literals, `tl.constexpr` values and dtypes; `.dtype`, `.shape`, `.numel` and a dtype's
    attributes; subscripts of shapes and tuples, and subscripts that only add axes; strings;
    `+ - * ~ & | ^`, comparisons, `not`, `and`, `or`, `is`, `is not`; and `/ // % ** << >>` only
    between constexpr values. It has **no call of any kind** (`tl.*`, `@triton.jit`, a method).
  - `compile-check` applies only to a `tl.static_assert` whose condition is pure. It must hold
    wherever the front end evaluates it: both branches of a runtime `if`, each unrolled
    `tl.static_range` iteration, each inlined call, and loop bodies (also when they run zero
    times), but not an untaken constexpr branch. The note names the condition of `Inst.ok`.
  - A `tl.static_assert` whose condition is not pure is transcribed: the operations in its
    condition become model lines.
  - The message need not be pure: Triton evaluates it only when the assertion fails, and then
    compilation stops.
  - The protocol's §7 says the same. The tool's drop list (`DROPS`) is unchanged.
- **Why.** The front end evaluates the condition as an ordinary expression, so an operation
  inside it is emitted and runs. `x is not None` is a compile-time `bool` for any `x`. So
  `tl.static_assert(tl.atomic_add(c, 1) is not None)` compiles and its atomic runs, and
  dropping the line would lose the atomic. A pure condition emits nothing that has an effect or
  can be undefined.
- **Evidence.** [evidence/debug-calls](../../../../evidence/debug-calls/README.md):
  - P1: four pure `tl.static_assert`s leave TTIR and PTX identical;
  - E1, E2: an atomic in the condition runs, directly or through a `@triton.jit` helper;
  - E8: an atomic in the message of a holding assertion does not run;
  - V1–V4: where the assertion is evaluated.
- **Constants whose meaning changed.** None (transcription rules and protocol only). Kernel maps
  that drop a `tl.static_assert` with `compile-check` must be re-checked by hand, since no
  digest covers the transcription rules. `vec_add_f32` has none.

## L7: TVA cites no other project's results, lemmas or measurements, and the tool enforces it

- **What changed** in `tools/tvr.py`:
  - **Imports** (`lean_header`): each header is read as Lean 4.35's
    `Lean.Parser.Module.header` parses it: `[module] [prelude] ([public] [meta] import [all]
    ident)*`, with nested comments as whitespace and `«»` names. Several imports on a line are
    read. A header the tool cannot read, or an `import` line after the header (outside
    comments), is a problem. The library's files are every `.lean` file under `lean/` (outside
    `.lake`), not only `lean/TVA`.
  - **Lake** (`lake_violations`, with `tomllib`):
    - the only requirement is `mathlib` with scope `leanprover-community` (no `path`, no
      `git`), whatever the key order;
    - the libraries are `TVA` (its name only) and, per kernel folder, exactly
      `srcDir = "../kernels/<kernel>/lean"` and `roots = [<the kernel's module>]`, with the
      sources resolving into that folder;
    - no other top-level key, and no `lakefile.lean`;
    - the manifest's only direct package is Mathlib from its repository, and every package is a
      git package.
  - **Text** (`scan_files`, `text_files`, `foreign_refs`):
    - every file outside `.lake`, `.venv`, `.git` and `__pycache__`, whatever its extension, in
      the whole project; `lean/lakefile.toml` and `kernels/README.md` included;
    - a file that is not UTF-8 is a problem, and so is a symbolic link out of TVA;
    - names are matched on words and word pieces (split at `_`, case changes and digits), and
      at the start or end of a one-case piece;
    - home paths are tested by whole component after normalising `..`, so `fpv/tva-old` and
      `fpv/tva/../x` fail;
    - another absolute path of two components or more outside the system directories fails;
    - a relative path whose `..` leaves TVA from its file's folder fails;
    - Triton's own pinned files (primitives, tutorials, a kernel's source, each checked by
      sha256) get the name check only, because a tutorial's placeholder `save_path` is Triton's
      text.
  - Links in the pages `tvr` generates are computed from the page's folder (`link`), so the
    tool's own text holds no `../../`. The generated pages are byte-identical apart from their
    timestamps, as I checked on a copy of the tree.
  - [REVIEW_PROTOCOL.md](../../../REVIEW_PROTOCOL.md) §2 "Enforcement" lists what the checks
    read.
- **Why.** The round-1 audit matched only lines starting with `import`, a `[[require]]` whose
  first key is `name`, nine extensions, whole words and path prefixes. So 14 of the library
  reviewer's 21 probes passed it, among them ways to import another project's modules.
- **Evidence.** [evidence/independence-audit](../../../../evidence/independence-audit/README.md):
  - 53 probes on copies of the tree, each flagged or passed as the rule asks, with 4 controls
    that must pass;
  - the library reviewer's round-1 probe script also reports "0 of 21 probes missed".
- **Constants whose meaning changed.** None (the tool only). The `DivFull` consequence noted in
  round 1 stays open under L4.

## L8: `Block.Fits` is stricter than `int32` indexing needs

- **What changed.**
  - `TVA.Block.Fits p B` is now `p * B + B ≤ 2^31`: the largest offset `p * B + B − 1` is below
    `2^31`.
  - The proof of `lane_toInt` changed (its statement did not). It treats program `0` apart,
    where `B` may be `2^31` and wrap as an `int32`, since `0 * B` is `0` anyway.
  - New: `Block.fits_of_grid_dvd` (every program of the grid `cdiv(n, B)` fits when
    `n < 2^31` and `B ∣ 2^31`) and `Block.fits_of_grid_pow2` (the same for `B = 2^e`,
    `e ≤ 31`).
  - `fits_of_grid` (`n + B ≤ 2^31`) is unchanged.
  - Updated: [LIBRARY.md](../../../LIBRARY.md) and [PROVER_GUIDE.md](../../../PROVER_GUIDE.md).
- **Why.** The lane lemmas need only the largest offset to fit `int32`, so a kernel on `n`
  elements can now cover `2^31 − B < n < 2^31`. The program's first offset `p * B` is a
  multiple of `B` below `n < 2^31`, so when `B` divides `2^31` it is at most `2^31 − B`.
- **Evidence.** Lean proofs only.
- **Constants whose meaning changed.** `TVA.Block.Fits` (definition). The statements of
  `lane_toInt`, `mask_lane`, `addrs_lane`, `addrs_block`, `accBytes_block`, `safe_load_block`
  and `safe_store_block` mention `Fits`, so they are weaker in their hypothesis and so
  stronger. New: `TVA.Block.fits_of_grid_dvd`, `TVA.Block.fits_of_grid_pow2`. No item of
  `vec_add_f32` reaches `Fits`, and it still uses `fits_of_grid`, whose statement is unchanged.

## L9: `Launch.Valid` admits grids the launcher silently skips

- **What changed.**
  - `TVA.Launch.Valid` adds `L.size < 2 ^ 31`, after the three grid bounds. The docstring
    cites the launcher's guard.
  - `driver.c` is pinned as `primitives/triton-3.8.0-nvidia-driver.c`, listed in
    `primitives/semantics-manifest.json` (sha256 `c32586d1…`, identical to the installed
    `triton/backends/nvidia/driver.c`).
  - New lemmas: `TVA.Launch.size_grid1` and `TVA.Launch.valid_grid1`. A 1-D grid of `n < 2^31`
    programs whose block shapes Triton accepts is valid.
  - Updated: [SEMANTICS.md](../../../SEMANTICS.md) "Launches", the rule page,
    [TRANSCRIPTION.md](../../../TRANSCRIPTION.md) §9, [PROVER_GUIDE.md](../../../PROVER_GUIDE.md),
    [DESIGN.md](../../../DESIGN.md) and protocol §6.
- **Why.** The launcher reads each grid dimension as a C `int` and launches only
  `if (gridX * gridY * gridZ > 0)`, with that product computed in C `int`
  (`triton-3.8.0-nvidia-driver.c:968-972`, `:1437`). A program count in `[2^31, 2^32]` makes it
  `≤ 0`, and the launch returns normally with no program run. For any kernel that writes,
  `Compiler.Faithful` was false there. Above `2^32` the guard depends on the product modulo
  `2^32`. Below `2^31` the whole grid runs. So `size < 2^31` is the bound that leaves no silent
  case.
- **Evidence.** [evidence/launch-outcomes](../../../../evidence/launch-outcomes/README.md), part C
  (H200 NVL, Triton 3.8.0). The script counts the programs that ran:
  - all `2^31 − 1` programs of `(2^31 − 1, 1, 1)` run, and all those of `(65535, 2^15, 1)`;
  - no program of `(2^30, 2, 1)`, `(2^15, 2^15, 2)`, `(2^16 + 1, 2^15, 1)` or `(2^16, 2^15, 2)`
    runs, and the host gets no error;
  - all `2^32 + 1` programs of `(6700417, 641, 1)` run.

  A Lean check confirms that `(2^30, 2, 1)` is no longer valid.
- **Constants whose meaning changed.** `TVA.Launch.Valid` (definition and docstring). It is in
  the cone of `vec_add_f32`'s `item:assumption:faithful`, through `Rules.Compiler.Faithful`.
  New: `TVA.Launch.size_grid1`, `TVA.Launch.valid_grid1`.

## L10: `no-op:print` drops arguments that may have effects

- **What changed.**
  - `no-op:print` now names `tl.static_print`, `tl.device_print`, Python's `print`,
    `tl.device_assert` and the Python `assert` statement. It applies only when every argument
    is pure (L6's definition). That means the printed values, `cond`, `msg` and `mask` of
    `tl.device_assert`, and `test` and `msg` of `assert`.
  - Otherwise the argument expressions are transcribed as model lines, in Python's order, and
    the call adds nothing.
  - Stated in [TRANSCRIPTION.md](../../../TRANSCRIPTION.md) §10, protocol §7,
    [exclusions.md](../../../exclusions.md) and [PROVER_GUIDE.md](../../../PROVER_GUIDE.md).
  - The same definition now gives `dead` its meaning ("a value no covered path uses, computed
    by a pure expression"). Before, it said "a pure value" without a definition.
- **Why.** Each of these calls evaluates its arguments as ordinary expressions. Without `debug`,
  `tl.device_assert` then emits nothing and `tl.device_print` emits a `printf`, so an operation
  inside an argument still runs. `assert` is `device_assert` and `print` is `device_print`
  (`code-generator.py:1353-1356`, `:357-365`). For `dead`, the definition rules out dropping an
  operation that has an effect or can be undefined (integer `//` by zero).
- **Evidence.** [evidence/debug-calls](../../../../evidence/debug-calls/README.md):
  - P1: pure `tl.static_print`, `tl.device_assert` and `assert` leave TTIR and PTX identical;
  - P2: pure `tl.device_print` and `print` add only the `printf`;
  - E3–E7, E9: an atomic inside `tl.static_print`, `tl.device_print`, `tl.device_assert` (in
    `cond` or `mask`), `assert` and `print` runs in all 4 programs.
- **Constants whose meaning changed.** None (transcription rules and protocol only). Kernel maps
  that use `no-op:print` or `dead` must be re-checked by hand. `vec_add_f32` uses neither.

## L11: `rules/num.md` asks kernel reviewers to check device truth

- **What changed.** [rules/num.md](../../../rules/num.md):
  - "What the reviewer judges" is replaced by three sections:
    - **What a kernel reviewer checks.** A `[rule]` field cites the definition for the model's
      operation, with the model's format and configuration, and with the library's arguments.
      A field whose constants the kernel chose is `RULE_MISCITED`, and the kernel asks the
      library for the fact. Device truth, lowering and constants are not the kernel
      reviewer's to check.
    - **Facts the library states (sm_90, Triton 3.8.0):** the `…Total` facts, `DivFull P` and
      `DivRn P .binary64`, each with its evidence. The accuracy facts with constants wait for L4.
    - **Device questions (library review):** the former "truth on the device" points and the
      `tl.dot` points, addressed to library review.
  - "Satisfiability" is kept as its own section.
  - The page's example no longer shows a kernel-chosen `MathAcc` instance. It cites `DivTotal`
    and `DivFull`.
- **Why.** Kernel reviewers read the page, and protocol §1–§2 forbid them the device check. They
  check that the cited fact is the right one for the operation.
- **Evidence.** None needed (the page's audience and wording).
- **Constants whose meaning changed.** None.
