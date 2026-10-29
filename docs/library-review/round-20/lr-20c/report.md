# Library review, round 20, by lr-20c

## Verdicts

- **L64** ACCEPT — `FloatV.litF8` (the exact conversion of the `float32` constant, `none` where Triton
  rejects it on the host; `±∞` only in `float8e5`; a NaN through `nanNarrow` only when the `float32`'s
  lower fraction bits are zero) and `Operand.constIn?` on both fp8 dtypes give what the device builds:
  I re-ran evidence/fp8-constants on GPU 0 and its output is identical to `out.txt` (F1 every value of
  both formats in `tl.full` and `tl.where`, F2 inexact values rejected on the host where `litF8` is
  `none`, F3 the specials as `tl.constexpr` arguments, F4 the run-time paths against `CvtSat`;
  `differences in all: 0`). num.md now confines the `CvtSat…` facts to conversions at run time, and the
  notes list the changed constants (`litF8`, `constIn?`).
- **L91** ACCEPT — the exclusion is unchanged and its reason is now true in every place the notes
  list (scope 9 in `Rules/Compiler.lean` and the rule page, `Launch.Valid`, `Claim.lean`, SEMANTICS,
  exclusions.md, protocol §6, both guides). I re-ran evidence/host-round20 (identical): G0, the call on
  the capturing stream ran no program and the replay wrote `x + 1`; G1–G3, a call on another stream ran
  at once in all three capture modes and the replay left the sentinel.
- **L95** REOPEN — the section is now read through an allowlist, and the ten pages of tool-round20
  Part L fail closed (re-ran, same result). But what comes before the heading is still judged by a
  fence tracker that does not follow CommonMark and by a check for HTML comments only, and cells are
  split unlike GFM. In `work/facts_probe.py` the page is otherwise the tree's, and on each of these
  pages the section renders as code or not at all, yet the gate still reads 38 facts and the audit is
  silent: (1) a fence of four backticks with a three-backtick line inside it (too short to close it);
  (2) a ```` ``` ```` fence with a ```` ```python ```` line inside it (a closing fence has no info
  string); (3) a line ```` ```a`b ```` (no fence: a backtick fence's info string holds no backtick)
  before a real fence; (4) `<script>` and (5) `<pre>` before the heading, closed after the next `##`
  heading (CommonMark's HTML block type 1 runs on across blank lines and headings). A row starting
  `||` makes `DivRn P .binary32` citable while GFM shows it in the second column, not under "Fact".
  Inline HTML (`<details>` in a text line, `<sub>` in a cell) passes, though protocol §2 says HTML is
  refused. What would fix it (cheap, in `facts_section` and `stated_facts`):
  - track fences as CommonMark does: a closing fence uses the opener's character, is at least as long,
    and has nothing after it but spaces; a backtick opener whose info string holds a backtick opens
    nothing;
  - make it a problem when a line before the heading, outside a fence, starts with `<` (after up to
    three spaces), and when the section holds a `<` followed by a letter, `/`, `!` or `?`;
  - split a row as GFM does (remove one leading and one trailing `|`, then split at each unescaped
    `|`), or refuse a row that starts with `||`;
  - and make §2's sentence say exactly this.
- **L98** ACCEPT — a NaN passed to a helper is now covered only when no earlier call in the same
  compilation names the same copy (SEMANTICS "Compile-time values Triton keys apart", from `mangle_fn`,
  which I checked against `code-generator.py:32-41`, `core.py:196-201` and `:1358-1396`), and a
  collision between NaNs at the kernel level is scope 11's matter. evidence/compile-keys K2 and K5
  re-ran identical. The reviewer's check for scope 11 has a defect, filed under L103.
- **L101** ACCEPT — all five points are fixed:
  - num-cvt8's README says which inputs `run.py` converts, and `specials.py` covers every non-finite
    pattern;
  - the num-dot sentence holds against `ieee_ratio.out`: `1023/2050 − 0.498994 = 3.04·10^-5` at
    `K = 1024`, and `1.6·10^-5` at `K = 512`;
  - `sem_scope.py` runs all 12 combinations, and the `AtomicAddF32` row and docstring say every `sem`
    and `scope`;
  - fp8's `u = 2^-(m+1)` and `eta = 2^(emin−m−1)` are right (`2^-4`, `2^-10`; `2^-3`, `2^-17`) and
    proved in `Encoding.lean`;
  - `replay_err`'s docstring names `L.filterMap Prod.snd`.
- **L102** ACCEPT — SEMANTICS states the conditional expression on a runtime condition (each side
  goes through `to_tensor`; both sides one type, or the kernel is rejected on the host) and what a
  helper's call is worth. The subset admits only helpers whose value is that of the `return` the
  program takes, and exclusions.md has the row. I re-ran evidence/front-end-round20 (identical): E1a
  `0`/`-1`; E1b 13,076 and 13,102 values differ from the `bfloat16` product; E2a `-3`/`-4`; E2b and E2c
  as the notes say.
- **L103** REOPEN — the scope itself is accurate (in the process, a dictionary lookup and then the
  `str` key; on disk, the `str` of each value joined with `-`), and compile-keys re-ran identical: 23
  launches ran code compiled for other values. But the kernel reviewer's check states a sufficient
  condition that the library's own evidence refutes:
  - REVIEWER_GUIDE item 11 says "A fresh cache directory, or values Triton keys apart, meets it";
  - compiler-faithful.md's checklist 11 says "a fresh cache directory, or …, meets it";
  - scope 11, in the Lean docstring and on the page, says "on a fresh cache".

  K1 runs each pair in one process with an empty cache folder of its own, and `C = 1.0` after `C = 1`
  still ran the integer kernel (`16777217`): the in-process cache belongs to the `JITFunction`, and a
  fresh folder does not clear it. So a reviewer who follows the guide accepts a host that sets a fresh
  `TRITON_CACHE_DIR` and launches `C = 1` before the claimed `C = 1.0`. What would fix it: in those four
  places, say that a fresh cache folder meets only the part that goes through the cache directory, and
  that in the process no earlier launch or warm-up of the kernel (an autotuner's benchmark runs
  included) may have values that share the claimed launch's key. For example: "values Triton keys
  apart for every launch of the kernel in the process, and a fresh cache folder (or values keyed apart
  on disk too), meet it".
- **L104** ACCEPT — scope 10 is stated in the Lean docstring, the rule page and its checklist, both
  guides, SEMANTICS "Names" and "Launches", exclusions.md, protocol §7 and `Claim.lean`. A reviewer who
  follows item 10 catches every case of the evidence, which re-ran identical: M1
  `kmod.double = kmod.triple`, M2 `fn.__globals__`, M3 `tl.exp = tl.log`, M4 `_unsafe_update_src`.
  The same changes, made in another process, still reach the launch through the cache folder. Since
  that also concerns scopes 2 and 3, it is filed below as a new issue.
- **L105** ACCEPT — `Launch.debug`'s docstring matches `kwargs.get("debug", self.debug) or
  knobs.runtime.debug` (`jit.py:727`), and scope 4 now excludes launches after a `preload`. D0/D1 and
  P0/P1 re-ran identical. `preload` stores its code in the process's cache under the key its data
  names, and compiles with its own options, so the disk entry it writes carries those options in its
  key. Its effect is therefore per process, as stated (`jit.py:823-875`).
- **L106** ACCEPT — the reader refuses an `Assumptions` that is not a structure. `reader_result` fails
  on a non-zero exit, on any number of result lines but one, and on a PANIC line. I re-ran
  tool-round20 Part S: the `inductive` and the `def` are refused and the structure passes. In
  `work/reader_probe.py`, a structure valued in `Prop` cannot hold a data field (Lean refuses it), so
  every field is a proof and none can carry a value that a later field depends on. A `private` field
  fails `kinds` and `usage` (fail closed; the gap below).
- **L107** ACCEPT — the host (x86-64, CPython 3.13) is named at the top of SEMANTICS, in scope 7,
  exclusions.md, both guides and `Claim.lean`, and `pyproject.toml` pins `>=3.13,<3.14`. Every evidence
  output I re-ran prints Python 3.13.9, from the x86-64 venv. The release check runs the tool itself
  on the system's `python3.12`. That is not the host, since the tool never runs Triton, so it agrees
  with protocol §2.
- **L108** ACCEPT — each point checked: the following all hold.
  - `tl.cdiv`, and the exceptions of the two general sentences.
  - `rerun-round20.txt` is present.
  - `normal_path` is used in `is_work_file` and `ledger_problems`. tool-round20 Part F re-ran: three
    paths that are not normalised are refused in a freeze, and the tree's ledgers pass.
  - num.md's rows write `\|`, and the gate still reads 38 facts.
  - The half-ulp wording, `FloatV.nan`, `IntV.le_toInt` (`SharedAcc.lean:367`), LIBRARY.md's rows, and
    float-layer's conclusion 4.

## New issues

- **NEW** soundness — the code a launch runs can have been compiled outside the scope, and no scope
  point says so. Scopes 2 (knobs) and 3 (hooks) are stated of the launch, scope 10 of "the process",
  and scope 11 only of compile-time values. But Triton reuses compiled code from the process's cache
  and from the cache folder, which any process may have written, under a key that holds neither
  `TRITON_KERNEL_OVERRIDE` / `knobs.compilation.override` (absent from `CACHE_INVALIDATING_ENV_VARS`,
  `GetEnv.h`) nor the state of `triton.language` (`DependenciesFinder.visit_Attribute` skips Triton's
  modules). In `work/cache_poison.py` (H200, Triton 3.8.0), the claimed launch's own process meets all
  eleven points as written in every case, and the reviewer's checks read only the claimed program:
  - C1: process A ran `tl.exp = tl.log` and launched `y = tl.exp(x)` on cache folder C. A clean process
    B then launched it on C and stored `log x`.
  - C2: process A, with `TRITON_KERNEL_OVERRIDE=1` and its `.ttir` edited to `x + 2`, launched on C. A
    clean B then stored `x + 2`.
  - C3: in one process with a fresh folder, `knobs.compilation.override = True` for a first launch,
    then the knob back to its default. The second launch still stored `x + 2`.

  What would fix it, a stated limit plus the reviewer's check (the user's rule): add to scope 11 that
  the code the launch runs was compiled within the scope. That is, points 2–4 and 10 held at every
  earlier launch or warm-up of the kernel in the process, from its start. And the cache folder the
  launch reads (`TRITON_CACHE_DIR`, else the one under `TRITON_HOME`) holds only what Triton 3.8.0
  wrote in processes that were themselves within those points, or is fresh, or the process sets
  `TRITON_ALWAYS_COMPILE=1` (an exception of scope 2, which skips the folder's entries). Put it in the
  places that state scope 11: the Lean docstring, the rule page and its checklist, REVIEWER_GUIDE item
  11, PROVER_GUIDE, SEMANTICS "Launches", exclusions.md and `Claim.lean`, citing this test or a new
  evidence folder.
- **NEW** gap — the claim reader and the gate misreport two claims, both failing closed:
  - `reader_result` takes any output line holding `PANIC` as a panic, the `TVR_JSON` line included. So
    a field docstring or a name holding the word fails `shape` as "the claim reader panicked"
    (`work/reader_probe.py` R3). What would fix it: skip the `TVR_JSON ` line, or match Lean's
    `PANIC at ` at a line's start.
  - A `private` field of `Assumptions` is listed in `fields`, but its statement is not read: the reader
    looks it up as `<ns>.Assumptions.<f>`, and its projection is private. So `kinds` says "not tagged
    [rule] or [kernel]" of a tagged field (R1). What would fix it: read each field through its
    `StructureFieldInfo.projFn`, or refuse a private field in the reader with a named error.

## Re-ran

- The release check, once, from an empty folder; the output is kept as `work/release.out` and
  `work/release.json`. 11 of 17 steps pass. The six that fail are the lint steps of the six round-20
  reports still in progress, mine included. Every other step passes: `shipped` (1094 files),
  `packages`, `sandbox`, `build`, `library` (`leanchecker`), `gate vec_add_f32` (equal to the shipped
  gate), `pages`, `records`, the lint of vec-add's two reviewers, and `audit`. The folder was deleted
  afterwards.
- evidence/tool-round20/run.py (parts L, S and F): the same as `out.txt`. Two problems are listed in
  another order, and truncated details differ in length with the scratch path.
- On GPU 0: evidence/compile-keys, fp8-constants, host-round20 and front-end-round20. Each output is
  identical to its `out.txt` past the version line, and is kept in `work/`.
- My own tests: `work/facts_probe.py` (L95), `work/reader_probe.py` (L106 and the claim reader) and
  `work/cache_poison.py` (the new soundness issue). `tvr audit` on the tree raises nothing about my
  folder.

## Not checked

- These I judged from the notes, the code and the outputs, without re-running them:
  `evidence/num-cvt8/specials.py`, `evidence/num-atomic-add/sem_scope.py`, the additions to
  `evidence/literals`, and `evidence/tool-round18/rerun-round20.txt`.
- L102 and L107: judged from SEMANTICS, the notes and the re-run above. I did not read
  TRANSCRIPTION.md's new pitfalls in full.
- Whether each fact that num.md's "Facts the library states" lists is true on the device, beyond how
  the gate parses the list (the device area).
