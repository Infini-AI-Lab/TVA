# Library review, round 20, by lr-20b

## Verdicts

- **L64** ACCEPT — An fp8 constant is now `FloatV.litF8`, the exact conversion of `litF32 c` or
  `none` where the host rejects, and `Operand.constIn?` uses it for both fp8 dtypes; this is what
  `scalar_constant` does (`primitives/triton-3.8.0-semantic.py:601-613`: `get_fp32`, then
  `create_fp_trunc`, folded only when exact). `evidence/fp8-constants` re-ran bit for bit identical
  (F1–F4, 0 differences, including the Lean side). My own corner cases agree with `litF8` too
  (work/probes2.out): `-1e-50`, whose `float32` is `-0.0`, stores `0x80` in `tl.full` and
  `tl.where`; `1e-50` stores `0x00`; `-1e-45` (`float32` subnormal) is rejected; `1 + 2^-40` stores
  `0x3c`. `constIn?` evaluated in Lean (work/Check.out) gives `0xfc` for `-inf` in `float8e5`, `none`
  for `+inf` in `float8e4nv`, `0x7e` for `nan`, `0xc2` for `-3`, `0x38` for `True`. `docs/rules/num.md`
  now confines the `CvtSat…` facts to conversions at run time, and TRANSCRIPTION has the pitfall.
  SEMANTICS' prose about which `c` are rejected says more than `litF8` does: filed below as wording.
- **L91** ACCEPT — The exclusion is unchanged, and its reason is now true everywhere it was given:
  `Claim.lean`, `Launch.Valid`'s docstring, scope 9, the rule page, SEMANTICS ("One launch", the
  scope list, the outcome note), exclusions.md, REVIEW_PROTOCOL §6 and both guides. A call on the
  capturing stream is captured, and a call on another stream runs at once. `evidence/host-round20`
  G0–G3 re-ran identical: the capturing stream left `-7` and the replay wrote `2`; another stream
  wrote `2` at once in all three modes, and the replay left `-7`.
- **L95** ACCEPT — `facts_section` admits only blank lines, `|` rows, `###`–`######` headings and
  text lines, with a blank line around each table. Anything else, a second heading of the name, or a
  section heading inside a fence or comment makes `stated_facts` list no fact and the audit name the
  page. I read the code against the forms a renderer could read differently and found none that
  passes. Each one fails closed: a heading indented by one to three spaces, `#######`, a GFM table
  without leading pipes (its delimiter row is no allowed form), a table right under a `###`
  heading, and a text line against a table. The tree's page gives 38 facts with no problem
  (evidence/tool-round20 Part L, read).
- **L98** ACCEPT — Settled as one case of L103. SEMANTICS "Constants" and the section docstring of
  `TVA.Lang.Scalar` say that a NaN reaching the kernel as a compile-time value has its own pattern
  only when the code the launch runs, and the helper copy the call uses, are its own. Two helper
  calls whose NaNs share a `mangle_fn` name are outside the subset (exclusions.md row 59).
  `evidence/compile-keys` K2 and K5 re-ran identical. The kernel reviewer's check for the
  `tl.constexpr` NaN of K2 is the scope-11 check, whose defects are under L103.
- **L101** ACCEPT — `Format.e4m3_u = 2^-4`, `e4m3_eta = 2^-10`, `e5m2_u = 2^-3` and
  `e5m2_eta = 2^-17` match the formats (precisions 4 and 3; smallest subnormals `2^-9` and `2^-16`).
  They and `IntV.le_toInt` use only the standard axioms (work/Check.out). The num-dot README now
  gives the true gap (`3.1·10^-5`). num-cvt8's `specials.out` reports `rows not as CvtSat says: 0`.
  num-atomic-add's `sem_scope.out` covers all 12 `sem`/`scope` pairs with 0 outside the bound.
  `replay_err`'s docstring names `L.filterMap Prod.snd`.
- **L102** REOPEN — The conditional expression is stated truly. E1 re-ran identical. In
  work/probes.out, a one-element or two-element block condition is rejected (B), and with `c = 1`
  the condition is compile-time, giving Python's `-1` as TRANSCRIPTION warns (O). The helper rule is
  also true of the front end (E2 re-ran identical; `decide_return_type`'s `common_type` uses
  `constexpr_type.__eq__`, which is Python's `==` on the values, `core.py:187-188`). **But one
  admitted form breaks the rule's own promise.** The subset admits "`return`s that all give one
  compile-time value, equal values of one Python type". `(1, 2)` and `(1.0, 2)` are equal and both
  of type `tuple`, so the form admits them; `common_type` compares tuples field by field with `==`,
  and the call is the first visited `return` on every path. In work/probes.out T1–T3, `c` is a
  runtime `int`. With `if c > 0: return (1, 2)` then `return (1.0, 2)`, `x * th(c)[0]` and
  `a, b = th(c); x * a` stored `16777217` for both `c = 5` and `c = -5`. The value of the `return`
  taken, which SEMANTICS says the model gives the call, is `16777216` for `c = -5` (a `float32`
  product). In the reverse order the call stored `16777216` for both. **Fix:** make the third form
  "the same `None`, `int`, `float` (not a NaN), `bool`, string or dtype, of one Python type; for a
  tuple, element by element", or exclude tuples from it. Make the change in SEMANTICS "The supported
  subset" (`return`) and "A helper's value", exclusions.md row 60, TRANSCRIPTION's pitfall "A
  helper's value", and REVIEWER_GUIDE §1's list.
- **L103** REOPEN — The front-end side and the scope's rule text are true. `mangle_fn` and
  `constexpr_type.mangle` are as SEMANTICS describes. `compute_cache_key` and `ASTSource.hash` are
  as scope 11 describes. `evidence/compile-keys` re-ran identical (23 launches). **But the check the
  kernel reviewer applies, which the user's rule relies on, is wrong in two ways.** These are
  REVIEWER_GUIDE item 11 and the checklist item 11 of `docs/rules/compiler-faithful.md`:
  (1) Both say "a fresh cache directory … meets it". K1 and K2 ran in one process on a fresh cache
  directory and collided (`C = 1.0` after `C = 1` ran the integer kernel), so a fresh directory does
  not meet it. Scope 11's last sentence ("on a fresh cache", in `lean/TVA/Rules/Compiler.lean` and
  on the rule page) reads the same way. A config the autotuner benchmarks first collides even in a
  fresh process.
  (2) Their lists name only `1`/`1.0`/`True` and `0`/`0.0`/`-0.0`/`False`, not the rule "values
  Python's `==` equates". In work/probes.out K, in one process on a fresh cache, `C = 2` then
  `C = 2.0` stored `33554434` both times (alone, `2.0` stores `33554432`), and the reverse order
  stored `33554432` both times. A reviewer who reads the list as the check misses every other
  integer and its float.
  **Fix:** in both checklists, state the two rules with examples. The first is values `==` equates:
  `2` and `2.0`, any integer and the float equal to it, `1` and `True`, `0`, `0.0`, `-0.0` and
  `False`. The second is values whose `str`s, joined with `-` across the parameters, give one text:
  a value and its `str`, and a `-` at another place. Two NaNs follow from both. Replace the
  sufficient condition, in both lists and in scope 11's last sentence, with: "every earlier launch
  and warm-up of the kernel, in the process (an autotuner's runs included) and through its cache
  directory, gave each compile-time parameter the claimed launch's value or one Triton keys apart
  from it; a fresh cache directory alone does not meet it".
- **L104** ACCEPT — Scope 10 states the limit (Triton as installed, no change to its modules or
  objects, no binding of the kernel module's names from outside). REVIEWER_GUIDE item 10 and the
  checklist name each route the evidence shows (`kmod.double = kmod.triple`, `fn.__globals__`,
  `tl.exp = tl.log`, `_unsafe_update_src`), so a reviewer who reads the host code as told catches
  M1–M4. `evidence/host-round20` M0–M4 re-ran identical.
- **L105** ACCEPT — `Launch.debug`'s docstring now counts the `JITFunction` attribute
  (`jit.py:727`, `kwargs.get("debug", self.debug)`). Scope 4, REVIEWER_GUIDE item 4 and
  exclusions.md exclude `preload`. D0, D1, P0 and P1 re-ran with the same results (D1's assert
  reports another thread id, `[96,0,0]` against `[64,0,0]`: a nondeterministic detail).
- **L106** ACCEPT — The reader template requires `isStructure` before reading fields, and
  `reader_result` fails `shape` on a non-zero exit, on anything but one result line, and on any
  `PANIC` line. Protocol §5 and §8 and REVIEWER_GUIDE §4 say so. I read the code and
  evidence/tool-round20 Part S (lr-18a's `inductive` refused; the round-start reader's panic now
  fails).
- **L107** ACCEPT — The host (x86-64, CPython 3.13) is named in SEMANTICS' introduction and
  "Launches", scope 7, the rule page, exclusions.md's targets row, both guides and TRANSCRIPTION.
  `pyproject.toml` pins `>=3.13,<3.14`, and every round-20 output prints Python 3.13.9. One more
  host property belongs in the same place: filed below (`-O`).
- **L108** ACCEPT — The points in my area hold. `tl.cdiv`'s `div - 1` is Python's for a compile-time
  `div` (the body is `(x + (div - 1)) // div`, `standard.py:43`, inlined with `div` a constexpr
  parameter). "Every argument is evaluated" names its one exception, and `execute_static_assert`
  (`code_generator.py:1617-1637`) visits `msg` only on failure. I found no other: `static_executor`
  visits keywords and then arguments. "What runs is what it evaluates" names the loop dry run and the
  code after an `if` that returns. From the files: float-layer's conclusion 4 is corrected; no
  num.md or LIBRARY.md table row keeps an unescaped `|` in a code span (my scan); "half an ulp"
  remains only where it is true.

## New issues

- **NEW** soundness — **A host running CPython with `-O` turns some of SEMANTICS' "rejected on the
  host" into launches that run.** Triton's front end makes some of its host rejections with Python
  `assert` statements (`primitives/triton-3.8.0-code-generator.py:497, 872, 886, 1020, 1117-1121`).
  `-O` (or `PYTHONOPTIMIZE`) removes them, and a kernel that nothing later catches then compiles. Scope 7, SEMANTICS' introduction and exclusions.md's
  targets row allow such a host: "CPython 3.13". The probe is work/probes3.out D. A loop body
  assigns `tl.float32` to a name holding `tl.float16`, which SEMANTICS "Names after a loop" says is
  rejected on the host. It was rejected under `python -I`. Under `python -I -O` it compiled and ran,
  and the name kept `tl.float16` after the loop even when the loop ran 3 times: Python's value would
  give `1.00048828125`, the launch stored `1.0`. (E, a runtime conditional expression with two
  types, stays rejected under `-O`: the IR verifier catches it.) Launches rejected on the host are
  outside every claim (L3). So a claim whose `Inst.ok` admits such an instance, with any model
  there, is wrong on an `-O` host. **Fix:** in scope 7 (`lean/TVA/Rules/Compiler.lean`, the rule
  page and its checklist item 7), SEMANTICS' target paragraph and "Launches", exclusions.md's
  targets row, REVIEWER_GUIDE and PROVER_GUIDE item 7, and TRANSCRIPTION's host pitfall, name the
  host as "CPython 3.13 run without `-O`/`-OO`, with `PYTHONOPTIMIZE` unset". Give the reason:
  Triton rejects some kernels through Python `assert`s.
- **NEW** wording — **fp8 constants: SEMANTICS' prose rejects more `c` than `litF8` and the
  device do, and PROVER_GUIDE omits `litF8`.** In SEMANTICS "Values" (the fp8 row: "a Python number
  built in them is its exact value") and "Constants" (the fp8 bullet: "is never rounded"; "A value
  the dtype holds is that value … every other `c` … is rejected on the host"), and in `litF8`'s
  docstring ("a number: its value in `g` when `g` holds it exactly"), the text speaks of `c` itself.
  Triton first rounds `c` to `float32` on the host, and `litF8` (correctly) tests that `float32`.
  work/probes2.out: `1 + 2^-40`, which no fp8 format holds, stores `0x3c` (`1.0`); `-1e-50` stores
  `0x80`; both are as `litF8` gives. A reader of the prose would think these launches are rejected,
  and so of no outcome. Separately, PROVER_GUIDE's "Write each Python number as the constant Triton
  builds" lists only `litF64`, `litF32`, `litF16` and `litBF16` for `tl.full`/`tl.where`. That leaves
  "Use `tl.cvt8` into fp8" as a prover's nearest guide for an fp8 constant, which is L64's wrong
  model. **Fix:** say "a `c` whose `float32` constant (`litF32 c`) the dtype holds is that value,
  also when `c` itself is not (`1 + 2^-40` is `1.0`; `-1e-50` is `-0`)", and drop "is never rounded"
  (it is rounded once, to `float32`). Add "`FloatV.litF8` in an fp8 dtype (not `tl.cvt8`)", or
  "`Operand.constIn?`", to PROVER_GUIDE's list.

## Re-ran

- evidence/front-end-round20 (L102), evidence/compile-keys (L98, L103) and evidence/fp8-constants
  (L64, with its Lean evaluation of `litF8`), all in full: outputs identical to the folders'
  `out.txt` (work/rerun-front-end-round20.txt, rerun-compile-keys.txt, rerun-fp8-constants.txt).
- evidence/host-round20 (L91, L104, L105) in full: identical except D1's reporting thread id
  (work/rerun-host-round20.txt).
- Added work/probes.py (probes.out): T1–T3, a helper returning equal tuples `(1, 2)`/`(1.0, 2)` on two
  runtime paths; K, `C = 2` then `C = 2.0` in one process on a fresh cache, both orders, against each
  alone; B, a runtime conditional expression with a `[1]` and a `[2]` block condition; O, the
  scalar `c = 1` in a conditional expression.
- Added work/probes2.py (probes2.out): fp8 constants whose `float32` differs from `c` (`±1e-50`,
  `-1e-45`, `1 + 2^-40`) in `tl.full` and `tl.where`, against `litF8`.
- Added work/probes3.py (probes3.out): the same kernels under `python -I` and `python -I -O` (a dtype
  reassigned in a loop body; a two-type runtime conditional expression).
- Added work/Check.lean (Check.out): `Operand.constIn?` on fp8 corner values, and `#print axioms`
  of the round's new theorems.
- Read against the pinned sources: `code_generator.py:32-41, 465-517, 544-626, 966-1036,
  1355-1396, 1617-1660`; `core.py:182-201, 645-797`; `jit.py:592-614`; `compiler.py:71-76`;
  `semantic.py:118-158, 601-623`; `standard.py:32-43`; `Ops.cpp:999-1026`.

## Not checked

- L95, L106 and the tool points of L108 (`normal_path`, the ledger checks, evidence/tool-round18's
  re-run): judged from the notes, the code of `tools/tvr.py` and `tools/claim_reader.lean.in`, and
  evidence/tool-round20's `out.txt`. I did not re-run `evidence/tool-round20` or
  `evidence/tool-round18`.
- L101's new scripts (`evidence/num-cvt8/specials.py`, `evidence/num-atomic-add/sem_scope.py`): I
  read their outputs; I did not re-run them.
- The release check (lr-20c's) and `kernels/vec_add_f32` (not to be touched).
