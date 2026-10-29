# Library review, round 6, by lr-6b

## Verdicts

- **L7** ACCEPT — The release check does what protocol §2 says. I ran
  `tvr release --dir <empty scratch folder>` on the tree as it stands, with this report's work files
  present. It copied 459 shipped files, hid `/home` and `/tmp` with `bwrap`, and gave: packages PASS,
  `lake build TVA` PASS, `tvr audit` PASS ("independent"), `lint vec_add_f32 reviewer r-vecadd` and
  `lint library 6 author` regular. The failures were the expected ones: `vec_add_f32`'s gate (its own
  build, `VecAdd.lean:155`), and the lint of this round's records that are still being written
  (lr-6a, lr-6b, coverage). From the code (`shipped_files`, `package_problems`, `cmd_release`): what
  git ignores is not shipped, each Mathlib package must be a clean checkout at the pinned revision, the
  tool runs with `-I -S`, and a shipped symbolic link that leads out of TVA is still caught by the
  audit, which runs inside the sandbox. Round 5's text-scan changes stand (not extended, as L36
  decided). One overstatement is filed below as wording ("nothing outside exists"). The L36 item "add
  the names of the other projects that share TVA's parent directory" is not done: the author left it
  to the coordinator (notes, L7), and I could not check it without looking outside TVA.
- **L20** ACCEPT — The user's "keep both" is met. In the kernel-reviewer documents, the reviewer's
  full read of the kernel's Lean is the main check. Round 5's backstops are all still in
  `tools/claim_reader.lean.in` and `tools/tvr.py`: generated names judged by kind; `open` read over
  every line; root names and aliases; no instance; plain-ASCII names; binders named like a library
  namespace or a structure/inductive (`libNames`, `binder_problems`), after NFKC and the look-alike
  fold. As L36 decided, the routes these backstops miss are left to the reviewer, and they are
  visible to a reader. One example: a local `let bc := …` shadows the bare abbreviation `TVA.bc`,
  which is not a namespace, so the binder check passes it. I did not re-run `evidence/gate-digests`
  (see Not checked); this verdict rests on reading the code and on the author's re-run of cases L–X.
- **L22** ACCEPT — The text is no longer written for a sequential program. The module docstring of
  `TVA.Lang.Desc` says descriptor accesses are "not ordered (library issue L15)" and that a pair with
  a write conflicts unless both are atomics. The exclusions.md row and SEMANTICS.md "Tensor
  descriptors" say the same, and both name the two cases the exclusion adds beyond `IntraRace`. The
  Lean agrees: `desc_atomic` is `atomicWith … .relaxed .gpu`, so it races across programs (not
  `Event.Strong`) and is `Event.SameAtomic` within a program. A different stale sentence in the same
  docstring, about in-kernel descriptors (L5), is filed below.
- **L25** ACCEPT — SEMANTICS.md "How the front end evaluates the source" matches Triton 3.8.0's front
  end. I checked it against `primitives/triton-3.8.0-code-generator.py`: `visit_BoolOp` (1481–1530,
  stops only at a compile-time operand that decides), `visit_If` (967–996), `visit_IfExp`
  (998–1049), `visit_Compare` (1076–1095, `is`/`is not` give a constexpr), `execute_static_assert`
  (1617–1637, `msg` only on failure), `static_executor` (1639–1650), `visit_Assert` (1353–1356) and
  `_find_carries` (465–503, the dry-run block is erased). `evidence/config-cond` and
  `evidence/debug-calls` re-ran with identical results. My `work/sem_probes.py` Q8 shows the dry run
  adds no atomic on the device: a `while` whose condition holds an atomic runs it 4 times (tests 0,
  1, 2 and 3, the last false), and a `for` body's atomic runs 3 times. TRANSCRIPTION §4's worked
  examples agree with these facts. Two wording slips in the section are filed below: keyword
  arguments are evaluated first, and "both sides run".
- **L31** REOPEN — The stale hint text is gone from SEMANTICS.md, but it is still in a Lean file that
  kernel reviewers read and that is hashed as a fact file. `lean/TVA/Lang/Ops.lean`, section
  "Compiler hints" (lines 685–701), still says: "A hint on a block argument (a loop's induction
  variable, or a variable the loop carries) has no effect … The model drops such a hint." That is the
  reading L1 withdrew, and it is false: in `evidence/hints-dry-run`, the hint marks the pre-loop
  value and the kernel faults. The docstrings of `tl.multiple_of` (715–721) and `tl.multiple_ofN`
  (743) still define `contig` as "the contiguity Triton assigns to `x`", "and the reviewer checks
  it". SEMANTICS.md calls that question open (L2), because a kernel reviewer cannot check Triton's
  contiguity. So the library states two contradicting readings, and the Lean one can lead a prover to
  drop a hint the device relies on. Fix: replace that paragraph and those two docstrings with
  SEMANTICS.md's text (in-loop hints on carried names or on the induction variable are open, L1; a
  `contig` above 1 and per-dimension values are open, L2; the withdrawn reading named as withdrawn).
  The other two items, `debug_barrier`'s reason and `dot_add`'s "±0 after folding", are fixed.
- **L32** ACCEPT — Subsumed by the whole-file digests. `library_files` (`tools/tvr.py:674-685`)
  hashes, in full, the Lean file of every `TVA` module that holds a constant of the cone, so module
  docstrings, field docstrings and comments are all included. It adds `GENERAL_PAGES` (SEMANTICS.md,
  exclusions.md) whenever a library module is reached, the `RULE_PAGES` page of each `TVA.Rules`
  module reached (so num.md comes with `TVA.Rules.Num`), and the packages. The `map` digest is the
  map, the source's sha and the `model` item's files (1059–1061). TRANSCRIPTION.md is in no digest.
  Read in the code; not re-run (see Not checked).
- **L35** ACCEPT — All seven items are fixed. `DivFull` is described as the relative bound in
  `Num.lean:146-151`, `rules/num.md:171-173`, SEMANTICS.md and the `tl.div` docstring. `TVA.Lang.Scalar`'s
  module docstring no longer lists `tl.maximum`, and points to the two cases. The `tl.full` wrap is in
  SEMANTICS.md "Constants" and the Scalar docstring (`evidence/literals/int_full.py` re-ran
  identical). `tl.ItofRel` restricts the fold to the default rounding and gives the `rtz` model.
  `tl.sum_bcast_mul` says an `a + a` combine has no meaning yet. `Num.lean` says "Library review
  judges". Protocol §2 names globs, `cd` then a relative path, and variables.
- **L36** ACCEPT — Transcription is now guidance, and every Triton fact it stated is in SEMANTICS.md.
  I compared, rule by rule, the rule page in git (`HEAD:docs/TRANSCRIPTION.md`, 44 rules) and the
  round-3 to round-5 additions in the authors' notes against today's SEMANTICS.md. Each fact is
  there, and each pattern and pitfall of the new page points to its section: `[call]`, `[implicit]`,
  `[index]`, `[pointer]` and `[float-prog]` are in "Values"; `[literal]` (per dtype, the `+0` for a zero
  included) is in "Constants"; `[itof-narrow]` and `rtz` are under `x.to(float)`; `[pure]`, `[config]`,
  `[compile-check]`, `[no-op:print]` and the aggregate operators are in the new front-end section;
  `[dot-add]`, `[sum-bcast-mul]` and `[reshape-reorder]` are under the combine pass; `[bool-ops]` is in
  "Order"; `[no-op:gdc]` and §9 are in "Launches". My `work/sem_probes.py` Q2 confirms that `-0.0` as a
  tensor of its own (`other=`, a stored scalar, an argument of `tl.fma`) is `+0`, as `FloatV.litF32`
  says. The map check (`tools/tvr.py:1003-1028`) checks only coverage, model text and regular
  entries; `rules` is free. The release check works (L7). Two loose ends are filed below: one fact
  page is missing for a common drop (`no-op:cache` on stores), and the map's coverage check is less
  complete than the page says. Not done on the coordinator's instruction: the sibling project names
  (L7).

## New issues

- **NEW** gap — SEMANTICS.md does not say what `/` means when both operands are integers. Triton
  converts both to `float32`, int64 included, and then divides (`P.div`): `semantic.truediv`,
  `primitives/triton-3.8.0-semantic.py:292-315`. Python divides in float64, so a prover who follows
  Python gets a different result. `work/sem_probes.py` Q1: int32 `16777217 / 1` gives 16777216, int64
  `(2**40 + 1) / 1` gives 2**40, and the TTIR divides in `f32` in all three cases (Python: 16777217,
  1099511627777). Where: SEMANTICS.md "`/` on floats" and the promotion list, and the `tl.div`
  docstring. Fix: state that integer `/` integer is `tl.itof` of each operand to `float32`, then
  `tl.div` (`DivFull`), and that integer `/` float converts the integer to the float's dtype.
- **NEW** gap — SEMANTICS.md says which operands of `and`, `or` and `not` the front end evaluates,
  but not what they compute on tensors. On `int1` tensors they are `&`, `|` and `~`
  (`semantic.logical_and/or/not_`, `primitives/triton-3.8.0-semantic.py:440-457`), and the tensor
  operands are combined from the right (`visit_BoolOp`, 1524–1528). A non-`int1` tensor operand is
  rejected on the host, because `bitcast` to `int1` fails. When every other operand is a
  compile-time value that decides nothing, the one tensor operand is returned unchanged, whatever its
  dtype: `True and x` is `x`. `work/sem_probes.py` Q5: int32 `x and y` raises "Cannot bitcast
  data-type of size 32 to data-type of size 1", and on `int1` the results are the logical ones. A
  `while` condition such as `i < n and x > 0` needs this fact. Fix: state it in "Order", or beside
  "Operators on tensors".
- **NEW** wording — SEMANTICS.md "Order" says an expression is evaluated "as Python evaluates it, left
  to right, operand by operand". But the front end evaluates a call's keyword arguments before its
  positional ones: `visit_Call`, `primitives/triton-3.8.0-code-generator.py:1455-1476`, and the same
  in `static_executor`, 1642–1647. `work/sem_probes.py` Q3: in
  `tl.store(out_ptr, tl.load(a_ptr), mask=tl.load(m_ptr) > 0)`, the TTIR loads `m` before `a`. No
  well-behaved model can observe this: the semantics already performs independent atomics of a
  program in any order (`Step.defer`), and any other pair of accesses to the same bytes with a write
  is a conflict. But the sentence is false as written. Fix: say that keywords come first, and why
  the order of evaluated operations does not matter, only the set the front end evaluates.
- **NEW** wording — SEMANTICS.md "Order", the case `x if c else y` with a tensor `c`, says "both
  sides run inside a runtime branch". In fact each side is emitted in its own branch of an `scf.if`,
  and only the taken side runs: `work/sem_probes.py` Q4, 4 programs, counter 4, not 44. The
  construct stays open (L28), but the sentence could lead a prover to put both sides' operations in
  the model. Fix: "each side is emitted inside its branch of a runtime `if`, and only the taken side
  runs (open: L28)".
- **NEW** wording — The module docstring of `TVA.Lang.Desc` (`lean/TVA/Lang/Desc.lean:40-42`) says "A
  device-side descriptor occupies scratch memory from `triton.set_allocator`; the model assumes it
  does not overlap the kernel's buffers (the reviewer checks the allocator)". That gives in-kernel
  descriptors a covered reading. SEMANTICS.md "Tensor descriptors" calls them open (L5), because the
  scratch writes make "every other byte is unchanged" false. Fix: say in the docstring that
  `tl.make_tensor_descriptor` is open (L5), as SEMANTICS.md does.
- **NEW** wording — TRANSCRIPTION.md §4 bases the drop `no-op:cache` ("cache, eviction and
  `volatile` modifiers") on SEMANTICS.md `tl.load`. That bullet (SEMANTICS.md:185-188) covers only
  loads. Nothing in SEMANTICS.md says that `tl.store`'s `cache_modifier` and `eviction_policy` leave
  the result unchanged, so the guidance rests on a fact the page does not state. Fix: state it in
  SEMANTICS.md's `tl.store` bullet.
- **NEW** wording — Protocol §2 (line 139) and the comment above the release check
  (`tools/tvr.py:2131-2134`) say a dependency on anything outside TVA "fails there, because nothing
  outside exists". But the sandbox binds `/` read-only and replaces only `/home` and `/tmp`
  (`tools/tvr.py:2247-2249`). `/scratch`, `/data`, `/afs`, `/opt` and `/mnt` stay visible, so a path
  computed at run time into one of them would pass. This is within the user's decision (a HOME holding
  only the toolchain, the venv and Triton's cache), and the projects TVA must not depend on live
  under the home directory. Fix: say "nothing under the home directory or `/tmp` exists", or bind only
  the system directories.
- **NEW** gap — The gate's `map` check exempts from coverage every source line that contains `"""`
  (`tools/tvr.py:1042-1052`: `if in_doc or q: … continue`). A code line with a triple-quoted string
  is therefore exempt, and so is every line after an odd count, until the next `"""`. Protocol §7
  and TRANSCRIPTION.md §5 say the gate fails when a code line is not covered.
  `work/map_coverage.py` runs the tool's own loop (checked against the tool's text) on a small source
  with an empty map. `tl.device_print("""v""", tl.atomic_add(c_ptr, 1))` and a line ending in
  `; s = """"""` are reported exempt, both lines holding atomics. A reader of the source can see such
  a line, so this is not a soundness reopen, but the mechanical check is less complete than the
  protocol says. Fix: find docstrings and comments with Python's `tokenize`/`ast` (a string
  statement), and require every other line holding a token to be covered.

## Re-ran

- On GPU 0 (NVIDIA H200 NVL, Triton 3.8.0 from `.venv`, torch 2.11.0+cu130, 2026-10-28), with
  `work/rerun/run_all.sh` (outputs in `work/rerun/`): `evidence/config-cond`, `evidence/literals` and
  `evidence/literals/int_full.py` gave output identical to the recorded `out.txt` files.
  `evidence/debug-calls` and `evidence/pure-operands` were identical except for the date in their
  first line.
- `tvr release --dir <empty scratch folder>` on the tree (results under L7).
- Tests added in `work/`, each with its `.out.txt`:
  - `sem_probes.py`: Q1 integer `/` (new gap); Q2 `-0.0` as a tensor of its own is `+0`; Q3 keyword
    arguments first; Q4 only the taken side of a tensor-condition conditional expression runs; Q5
    `and` on int32 rejected, logical on `int1`; Q6 `x_i32 * 1e-40` computes in float64, `x_f32 *
    1e-40` in float32 with the subnormal constant, `x_f16 * 1e-40` with `+0` (as SEMANTICS.md
    "Constants" says); Q7 integer `//` and `%` truncate; Q8 the loop dry run leaves no atomic.
  - `ptx_probes.py`: `tl.fma` is `fma.rn.bf16`/`fma.rn.f16` (one rounding, as `tl.fma` says);
    bfloat16 `+ - *` are native; `(f16 a * f16 b).to(f32) + c` is not contracted across the widening
    (`mul.f16`, `cvt.f32.f16`, `add.f32`), so "a product feeds it" over-approximates safely there.
  - `atomic_shift_probes.py`: a tensor atomic is one rmw per enabled lane (16 lanes on one address
    add 16; a `[16]` block under 4 warps adds 1 per element; a `[16, 16]` block repeating 16 addresses
    adds 16 each); `atomic_cas` compares bit patterns (`-0.0` against `+0.0` does not swap, equal NaN
    bits do); `int8 >> uint8` is an arithmetic shift of the promoted `uint8` values (`-128 >> 1` gives
    `0xc0`), and `0xF0 >> s` with `s : uint8` is logical (`0x78`).
  - `div_fusion.py`: does a float64 `x / 4.0 + y` become a product contracted into an `fma` (LLVM can
    turn `llvm.nvvm.div.rn.d` into `fdiv`, `primitives/llvm-NVPTXTargetTransformInfo.cpp:298-299`)?
    No: the PTX keeps `div.rn.f64` then `add.f64`, with fusion on or off. On the subnormal input
    where an fma would differ (x = 11·2^-1074, y = 2^-1021), the result is the IEEE one. The same
    holds for float32 `/` and `tl.div_rn`.
  - `map_coverage.py` (new gap above).
- Survey (`semantics`): I read `lean/TVA/Lang` (Scalar, Tensor, Mem, Platform, Prog, Ops, Desc,
  BlockPtr, Scan, Random) and `lean/TVA/Sem` (Step, Launch) against SEMANTICS.md, and both against
  the pinned sources: `semantic.py` (promotion, `to_tensor`, `scalar_constant`, `cast`, `truediv`,
  logical operators, `device_assert`, block-pointer conversion), `core.py` (`_block_ptr`, `where`,
  `device_assert`, constexpr operators), `math.py` (`_check_dtype`: the math functions,
  `sqrt_rn` and `div_rn` accept only float32 or float32/float64, so other formats are rejected launches),
  `code_generator.py` (above), `random.py` (`uint_to_uniform_float`),
  `nvidia-ElementwiseOpToLLVM.cpp` (division lowering) and `llvm-NVPTXTargetTransformInfo.cpp`.
  Block pointers (offsets cast to `int64` by `_canonicalize_block_ptr_dynamic_tuple`, the mask and
  padding of `_materialize`/`load`), `int1` loads (`!= 0`), integer-to-float and float-to-float casts
  (the fp16/bf16 routes through float32 round once), the Philox wrapper, scans, sort, histogram, and
  the race and conflict definitions all agree with the library. Apart from the issues above, I found
  nothing false.

## Not checked

- `evidence/gate-digests` (L20, L32, L36) was not re-run. It runs `tvr check`, and so `lake build`,
  on a copy of the tree whose `lean/.lake/packages` is a link to the tree's, and I was told never to
  run `lake build`. I judged the digest, map and namespace changes from the code. `evidence/release-check`
  R1 and R2 were not re-run either: they build a copy outside the sandbox. My own run was the tree
  as shipped (R0).
- `evidence/independence-audit` (its script takes the other project's name on the command line), and
  whether the names of TVA's sibling projects should be added (that needs a look outside TVA).
- The rewritten prompts and guides (REVIEWER_GUIDE.md, PROVER_GUIDE.md, docs/prompts/), beyond what the
  protocol says of them.
- Tutorial coverage, which is the coverage auditor's. I did not ask for each tutorial construct
  whether the library defines it (for example `tl.clamp`, `tl.flip` and float `%` have no `tl.*`
  definition).
- The device facts (`TVA.Rules.Num`) and the lemmas, which are other survey areas.
