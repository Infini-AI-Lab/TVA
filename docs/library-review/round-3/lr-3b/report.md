# Library review, round 3, by lr-3b

## Verdicts

- **L7** REOPEN — The round-2 spellings are now caught, and on a scratch copy the author's 63
  probes give their recorded result ("63 probes, 0 wrong"). But a literal home path still passes
  when it follows `>`, `)`, `]`, `}` or `-`, because `_ABS` in `tools/tvr.py` does not start a path
  after those characters. That skips the placeholder pieces (`<N>/`, `{kernel}/`) as intended, but
  it also skips real paths. `work/l7_probes.py`, run against the tool's own `foreign_refs`, gets 10
  of 17 probes wrong, for example a shell redirect with no space (`python3 run.py
  >/home/<user>/fpv/<dir>/out.txt`, `echo x >~/fpv/<dir>/x`, `2>…`, `>>…`), `(notes)/home/…`,
  `x]~/…`, `a-/home/…`, `$(dirname $0)/../…` and Python's `f"{os.environ['HOME']}/fpv/<dir>/…"`.
  End to end (`work/l7_e2e.py`), a copy whose `docs/SEMANTICS.md` holds the redirect line still
  passes `tvr audit` as "independent"; the same line with a space after `>` fails it. Protocol §2
  says a path counts after "a bracket". Fix: start a *home* path (`/home/<user>/`, `~/`,
  `~<user>/`) after any character other than a letter, digit, `_`, `.` or `/`, and keep the
  placeholder exception only for other absolute paths. Read `)`/`}` followed by `/fpv/…` as being
  under the home directory, or state in §2 that paths computed at run time are not checked. Add
  these probes to `evidence/independence-audit`.
- **L12** REOPEN — Identities, the `rules` field of the map, the closure over `[id]` references,
  the per-rule hashes and the `map` check all work: the re-run of `evidence/gate-digests` on a copy
  matches `out.txt` (cases A–H). But the parser does not agree with the page about where the
  general text is. `transcription_rules` ends a rule only at the next rule or heading. So the last
  paragraph of §3 ("No construct of exclusions.md is used … no aggregate class") is folded into
  `[atomic]`, while the page's introduction (and protocol §4) say it is general text that applies
  to every model (`work/rule_parse.out`). On a copy, deleting that paragraph's aggregate clause
  changes no digest of `vec_add_f32`, whose map does not name `[atomic]`. With the map naming
  `[atomic]`, the same edit re-opens `map` and `model` (`work/l12_digests.py`, X1/X2). Fix: end a
  rule at the first unindented paragraph after a blank line, or move that paragraph above `[mask]`.
  Then have `tvr audit` check that the parts the introduction calls general really are general.
- **L13** ACCEPT — The `hygiene` list now reads whole files and runs before the build. `replay`
  runs the toolchain's `leanchecker <root>`. I read `LeanChecker.lean` (Lean 4.35.0): it replays
  the constants of every module whose name has the root as a prefix into the environment of that
  module's imports, so whatever meta code stored is checked again by the kernel. The re-run of
  `evidence/gate-replay` is identical (C1 accepted; A1, A2 and R1 rejected and flagged). The re-run
  of `evidence/gate-digests` is identical: F stops at `hygiene`, and G (hygiene off) fails
  `replay`. The things a token list cannot stop, such as pure code run by `#guard`, cannot add an
  unchecked declaration, and `replay` is what the soundness rests on.
- **L14** ACCEPT — `[pure]` now limits operand types to tensors, constexpr and plain Python
  values, dtypes, and tuples of these. It excludes aggregates (and tuples that hold one), and
  `[no-op:print]`, `[compile-check]` and `[dead]` name it. In `code-generator.py`
  (`_apply_binary_method`, `visit_Compare`, `visit_UnaryOp`, `visit_Subscript_Load`,
  `get_Attribute`), only aggregates dispatch to jitted methods; on tensors the operators are
  builtins with no effect without `debug`, since `sanitize_overflow` asserts emit nothing then. The
  re-run of `evidence/pure-operands` is identical (A1–A5 run the method; B1 has identical PTX).
- **L15** ACCEPT — The rule matches the user's decision. `IntraRace` treats two events of one
  program that conflict as a conflict unless they are `Event.SameAtomic`. `WellBehaved` excludes
  it on every trace, across iterations. `Step.defer` lets an `rmw` be overtaken only by a step that
  is the same for every old value, which gives every order the program's dependences allow.
  Partly overlapping atomics conflict, which is stricter, hence sound. The re-run of
  `evidence/intra-program` gives the same picture (reversed read-back stale in 41.9% and 23.6% of
  the elements; 0 with a barrier or the same offsets; 96.9% stale when pipelined; atomic pairs
  reordered in 0.33–0.77% of the addresses, every counter 2). `[no-op:debug_barrier]` and
  `[no-op:range-kw]` stay sound because the semantics allows a superset of the orders a barrier
  or pipelining allows. One wording slip is filed below. The constants whose meaning changed are
  listed (`WellBehaved`, hence `Faithful`'s cone).
- **L16** ACCEPT — `[reduce]` and SEMANTICS.md agree with `standard.py:173-190, 232-249`
  (`.to(int32)` for any integer narrower than 32 bits; `IntV.cast` sign-extends a signed and
  zero-extends an unsigned source, as `semantic.cast` does) and with `_pick_sum_dtype`
  (`:269-293`). Index reductions convert only bf16 (`_promote_bfloat16_to_float32`). The re-run of
  `evidence/int-reductions` is identical (A: 0 of 35 differ; B: 8 sums wrap as stated; C: the
  value of `return_indices` is not widened).
- **L17** REOPEN — `[permute]` and `Tensor.permute` are right. `core.py:2070-2121` matches, and
  `tl.trans(x)` gives `[0, …, r-3, r-1, r-2]`. The re-run of `evidence/permute` is identical. In
  Lean, the forms the rule names elaborate with the permutation found by `decide`, and
  `get_permute` follows torch's convention (`work/lean/PermuteUse.lean`). The index reductions and
  `ArgRes` are right too, and the re-run of `evidence/argmax` is identical. But `[no-op:gdc]` drops
  `gdc_launch_dependents()` on a false ground ("no dependent kernel is waiting for the signal").
  The kernel's own `pdl = false` does not stop the *next* kernel on the stream from being launched
  with `launch_pdl=True`. TVA's own `evidence/pdl` part C shows it: the producer, launched with the
  default `launch_pdl` (`run.py:117`), calls `gdc_launch_dependents()`, and the PDL consumer then
  runs while the producer is still running (stale reads in 19 of 20 runs, in my re-run too). Such
  a successor can write memory the kernel reads, and no model of the kernel accounts for it.
  Fix: either keep `gdc_launch_dependents` out of scope, or state in `Compiler.Faithful` / the
  claim's scope (SEMANTICS "Launches", exclusions.md) that the launch runs with no other agent on
  its memory, in particular no PDL successor started by it. Then reword `[no-op:gdc]` to rest on
  that condition.
- **L18** ACCEPT — `Faithful`'s docstring, the rule page, `[dot]`, §9 and exclusions.md now give
  the environment-at-defaults scope. `core.py:2384-2388` resolves an omitted `input_precision`
  from `knobs.language.fp32_default` first, so with it unset the precision is `tf32` (`ieee` with
  `allow_tf32=False`). `TRITON_F32_DEFAULT` and `TRITON_DEFAULT_FP_FUSION` are in the
  cache-invalidating list (`GetEnv.h`), so a cached kernel cannot carry them over. The exempt
  variables only name directories, dump, time or touch debug information. The re-run of
  `evidence/env-knobs` is identical. The hashed `Faithful` docstring re-opens
  `assumption:faithful`, as the notes say.

## New issues

- **NEW** soundness — `[literal]` (TRANSCRIPTION §2) and SEMANTICS.md "Constants" make a Python
  float constant the nearest float32 (then converted, by `[implicit]`). That is false in two dtypes.
  A Python scalar operand becomes `scalar_constant(c, dtype of the operation)`
  (`semantic.py:205-206, 601-613`), and so do `tl.full` and `tl.where`. Then:
  - **fp64:** the constant is the exact double (`get_fp64(double)`, `ir.cc:968-972`). So `x * 0.1`
    on float64 uses `0.1`, not `float32(0.1)`.
  - **bf16:** `get_bf16` builds the constant from `std::to_string(float)`, six decimals
    (`ir.cc:952-957`). So `x * 1e-7` on bf16 multiplies by `+0`, `-1e-7` gives `-0.0`, and
    `1.23456e-5` becomes bf16(`1.2e-5`), 3% off.

  On the H200, `work/literals.py` and `literals2.py` show 128 of 128 elements differ from the
  rule's model for fp64 `*`/`+`, `tl.full` and `tl.where`, and for bf16 with small constants.
  `other=` and fp32/fp16 follow the rule. Fix: give the constant per the operation's dtype: fp32
  is the nearest fp32; fp16 is the nearest fp16 of that; fp64 is the Python double; bf16 is the
  bf16 nearest to the `%f` text of the nearest fp32 (or exclude bf16 literals that this changes).
  Keep nearest-fp32-then-convert for `other=` and `.to`.
- **NEW** soundness — `[itof-narrow]` (and SEMANTICS "x.to(float)") makes every narrowing of an
  int-to-float value one `tl.itof`. MLIR's fold `truncf(sitofp x) → sitofp x` needs the default
  rounding mode (`llvm-ArithCanonicalization.td:453-463`), and a narrowing with
  `fp_downcast_rounding="rtz"` is `tt.fp_to_fp`, not `truncf` (`semantic.py:822-845`). On the
  H200, `x.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtz")` differs from every value
  `tl.ItofRel` allows for 32128 of the 65536 ints in `[0, 2^16)` (259 gives 258, the rule allows
  only 260; `work/itof_rtz.py`). Fix: restrict the rule to default rounding. With `rtz` the model
  is `tl.itof` to the intermediate format, then `tl.cvt … .rtz`.
- **NEW** soundness — `[config]` (TRANSCRIPTION §10) drops "a branch not taken … (a branch on
  constexprs)" and asks nothing of the condition. But the front end evaluates the condition of
  every `if` and conditional expression before it picks a branch (`visit_If`, `visit_IfExp`,
  `code-generator.py:967-1049`). A condition that is a compile-time bool can still contain an
  operation, as `[compile-check]` already says of `x is not None`. On the H200,
  `if tl.atomic_add(c, 1) is not None:` and `v = 1 if tl.atomic_add(c, 1) is None else 2` compile
  as constexpr branches (no `scf.if`), and the atomic runs in all 4 programs (`work/config_cond.py`;
  a pure `if FLAG:` control runs none). Fix: let the `if` line and the untaken branch be dropped
  only when the condition is pure (`[config]` names `[pure]`); otherwise the condition's
  operations are model lines.
- **NEW** soundness — `[sum-bcast-mul]` names one spelling,
  `tl.sum(x[:, :, None] * y[None, :, :], axis=1)`. The combine pass matches the IR
  (`Combine.cpp:112-198`), so `tl.sum(tl.expand_dims(x, 2) * tl.expand_dims(y, 0), axis=1)` and a
  `tl.reduce` with a `+` combine are rewritten too, into an IEEE `tt.dot` (an FMA loop), even with
  `enable_fp_fusion=False`. A reviewer who reads the rule literally applies `[reduce]`/`[fusion]`
  there. With fusion off that is IEEE additions, and 58 of 256 outputs are values no IEEE tree
  gives (`work/bcast_mul.py`, K = 2). Fix: state the rule by the computation, not its spelling:
  operands expanded on axes 2 and 0 (`[:, :, None]` or `tl.expand_dims(…, 2)`, `[None, :, :]` or
  `tl.expand_dims(…, 0)`, with or without `tl.broadcast_to`), multiplied in that order, then
  reduced on axis 1 by `tl.sum` (or by `tl.reduce` with `+`, once that has a rule).
- **NEW** soundness — `[call]` makes a call "the library definition SEMANTICS.md gives for it",
  and SEMANTICS.md also holds conversions the model must write. Examples are the constants,
  `tl.maximum`'s bf16 → fp32, `/` and `%` on fp16/bf16 in fp32, and the store conversion. But
  SEMANTICS.md is outside every digest: `rule_parts` folds in only TRANSCRIPTION.md and
  exclusions.md, and tl.* docstrings are not hashed. On a copy, changing SEMANTICS.md's
  "Constants" bullet or its `tl.maximum` conversion re-opens nothing of `vec_add_f32`
  (`work/l12_digests.py`, X3/X4). This is L12's problem for the other page a map rests on. Fix:
  fold SEMANTICS.md into the general text of the `map` and `model` digests, or move each "the model
  writes …" obligation into a TRANSCRIPTION rule and let a rule's text name the SEMANTICS
  sections it relies on.
- **NEW** gap — No rule covers a runtime `if`/`else`, a conditional expression on a scalar
  tensor, or a `return` inside a runtime `if`. The front end lowers that `return` by placing the
  rest of the function in the other branch (`visit_if_top_level`, `code-generator.py:893-929`).
  Its condition is `cond.to(int1)`: nonzero, and true for a NaN. §7 has only `[range]`,
  `[static-range]`, `[while]` and `[bool-ops]`, and no drop reason fits an `else:` or `pass` line.
  The pinned grouped GEMM tutorial (`tutorials/08-grouped-gemm.py:264`) needs one. Fix: add an
  `[if]` rule (Lean `if` on the converted condition, the early-return form, both branches'
  conversions written).
- **NEW** gap — No rule gives the type of a scalar argument other than an int below `2^31`
  (`[index]`). Triton 3.8.0 passes a Python `float` as `fp32`, rounded from the host's double, a
  `bool` as `int1`, an int at or above `2^31` as `int64` (`uint64` from `2^63`), and the value `1`
  as a constexpr (Triton tree `python/src/specialize.cc:264-301, 383-387`). Kernels with
  `sm_scale` or `eps` arguments need the float case. Fix: add a `[scalar-arg]` rule.
- **NEW** gap — exclusions.md ("aggregate classes … made with Triton's `_aggregate`,
  `triton.language.core`") and the general paragraph of TRANSCRIPTION §3 exclude every aggregate.
  But in 3.8.0 `tl.make_block_ptr` returns `_block_ptr`, an `@_aggregate` class of
  `triton.language.core` (`core.py:1828`, `:2626`). So block-pointer kernels are formally out of
  scope, while `[block-ptr]` and SEMANTICS "Block pointers" give them meaning, and the row's reason
  ("no transcription rule covers them") is false for that class. Fix: exclude aggregates other
  than Triton's `_block_ptr`, and say that block-pointer values are not pure operands.
- **NEW** wording — SEMANTICS.md "Compiler hints" still states the withdrawn rule: a hint on a
  block argument "has no effect and the model drops it". That contradicts `[hint]`, which says it
  is withdrawn and open under L1. Fix: replace it with L1's open status now.
- **NEW** wording — `[no-op:debug_barrier]` says that in a well-behaved model "no two accesses of
  a program to the same bytes depend on their order". Two atomics of a program on the same bytes
  do depend on their order (old values, `xchg`, `cas`). The drop is still sound, because the
  semantics allows every order (`Step.defer`). Fix: say that instead.
- **NEW** wording — `[dot-add]` says "the dot's accumulator is a constant zero", but the pattern
  matches any accumulator that folds to `±0` (`m_Zero`/`m_AnyZeroFloat`, `Combine.cpp:19-21`),
  for example `tl.zeros(…) * 2.0`. It also skips fp8 dots whose `max_num_imprecise_acc ≠ 0`; this
  part is harmless, since `dot_add` allows both forms. Fix: say "an accumulator that is `±0` after
  constants are folded".
- **NEW** wording — `[decl]` drops a decorator line as "covered by the model's definition". But
  `@triton.autotune`'s `reset_to_zero`, `restore_value`, `pre_hook`/`post_hook` and
  `Config.pre_hook` run host code that writes the kernel's buffers, and benchmarking launches the
  kernel repeatedly on the real arguments (`runtime/autotuner.py:151-153, 245, 273-275`). Fix: say
  that a claim covers one launch, from the memory at that launch, and that these hooks and
  benchmark runs are host code the conditions (`Pre`) must account for.

## Re-ran

- On GPU 0 (H200 NVL, Triton 3.8.0, torch 2.11.0+cu130, 2026-10-27; `TMPDIR` in a scratch
  directory), I re-ran `evidence/int-reductions`, `permute`, `pdl`, `argmax`, `pure-operands`,
  `intra-program` and `env-knobs`. The outputs (`work/rerun_*.out`) are identical to `out.txt`,
  except the counts of racy cases: in `intra-program`, 41.9% and 23.6% stale, and atomic reorders
  of 0.33–0.77% of the addresses; in `pdl`, the stale element count. The conclusions are the same.
- `evidence/gate-replay` gave output identical to `out.txt`. I ran `evidence/gate-digests` on a
  temporary copy; the output is identical to `out.txt` (`work/rerun_gate-digests.out`).
- `evidence/independence-audit/probes.py` on a scratch copy gave "63 probes, 0 wrong", identical
  to `out.txt`. `tvr audit` on the tree prints "independent".
- Added, each with its `.out` beside it in `work/`:
  - `rule_parse.py` and `l12_digests.py` (L12; new issue on SEMANTICS.md);
  - `l7_probes.py` and `l7_e2e.py` (L7);
  - `literals.py`, `literals2.py`, `itof_rtz.py`, `config_cond.py` and `bcast_mul.py` (new
    issues);
  - `lean/PermuteUse.lean` (L17; standard axioms only).
- Read in `primitives/`: `semantic.py` (constants, promotion, casts, `mod`, `dot`), `core.py`
  (`_block_ptr`, `trans`, `permute`, `dot`, reductions), `standard.py`, `code-generator.py`
  (`visit_If`, `visit_IfExp`, `visit_BoolOp`, `get_Attribute`), `ir.cc`, `Combine.cpp`,
  `llvm-ArithCanonicalization.td`, `llvm-ArithOps.cpp`, `knobs.py`, `GetEnv.h` and `gdc.py`. Also
  read: the Triton 3.8.0 tree's `specialize.cc` and `autotuner.py`, and Lean 4.35.0's
  `LeanChecker.lean`.

## Not checked

- The Lean proofs of the L15 kit (`Safe`, `SafeA`, `independent`, `shared`, `Step.transfer`): I
  read the definitions (`IntraRace`, `Event.SameAtomic`, `Step.defer`, `WellBehaved`) but built no
  traces and re-proved nothing; that is the `rules`/`lemmas` areas.
- Whether `leanchecker` can be defeated by tampering with the imports' `.olean` files or the
  toolchain during a build. It trusts both, and I found no route to code with effects that the
  hygiene list leaves open.
- Paths that code computes at run time. The audit is lexical; probes Q8 and Q9 show the f-string
  forms.
- The `tl.range` keywords `warp_specialize`, `flatten`, `disallow_acc_multi_buffer` and
  `disable_licm` on the device.
- The issues outside this batch (L1, L2, L4, L5).
- `tvr check` on the real tree (the coordinator's instruction). `vec_add_f32` was gated only in
  copies.
