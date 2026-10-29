# Library review, round 3, by lr-3a

## Verdicts

- **L7** REOPEN — The import, lake and name checks hold, and the round-2 spellings are now caught:
  the author's 63 probes re-run with the recorded results (`work/rerun_independence-audit.out`). But
  four more spellings that a shell or a JSON reader resolves to a home path outside TVA still pass the
  tool's own `foreign_refs` (`work/l7_more_spellings.py`, `.out`): `${HOME:-}/…` and `${HOME%/}/…`
  (`_HOME_VARS` reads only `${HOME}`, and a `/` after `}` never starts a path), `$(echo ~)/…` (a `/`
  after `)` never starts a path), and JSON `/` escapes (only `\/` is decoded). Fix: read every
  `${HOME…}` parameter expansion and every `$(…)` substitution as a home prefix in `_path_text`,
  decode `\uXXXX` escapes before the test, and add the four as probes.
- **L12** REOPEN — Rule identities, map `rules`, the folding of rule texts into the `map`/`model`
  digests and the page-consistency check work: the re-run of `evidence/gate-digests` on a scratch
  copy gives A–H as recorded (`work/rerun_gate-digests.out`). But the tool's split of the page
  disagrees with the page. TRANSCRIPTION.md says "the last paragraph of §3" is general text whose
  change re-opens every kernel, while `transcription_rules` runs a rule to the next rule or heading,
  so that paragraph ("No construct of exclusions.md is used … no aggregate class") is part of
  `[atomic]`. Changing it changes no digest part of a map that names `[mask]` and not `[atomic]`
  (`work/l12_general_text.py`, `.out`); it is the only unindented paragraph a rule absorbs. Fix: move
  the paragraph above §3's first rule (or end a rule at an unindented paragraph, and say so on the
  page). The same sentence lists §1 as general, but §1 holds `[helper]`.
- **L13** ACCEPT — `replay` (Lean's `leanchecker` on the compiled modules) answers unchecked
  declarations by any route, and `hygiene` now runs before the build. Re-ran `evidence/gate-replay`
  (control accepted; A1, A2 and round 2's R1 rejected) and `evidence/gate-digests` F and G (hygiene
  stops the build; with hygiene off, replay rejects `extraFalse`): identical to `out.txt`. A different
  route, model text that means another operation with no unchecked declaration, is a new issue below.
- **L14** ACCEPT — `[pure]` now restricts the operand types (tensors, `tl.constexpr` and plain Python
  values, dtypes, tuples of these), aggregates are excluded in exclusions.md, and `[no-op:print]`,
  `[compile-check]` and `[dead]` name `[pure]`. The cited code-generator lines dispatch to the
  aggregate's methods as stated. Re-ran `evidence/pure-operands`: A1–A5 run the method in all 4
  programs, A6 is rejected, A7–A8 run nothing, B1's PTX is identical: as recorded.
- **L15** ACCEPT — Judged against the user's decision, it is implemented as decided. `IntraRace` (two
  events of one program that conflict, unless both are atomics on the same bytes) is excluded on every
  reachable trace of `WellBehaved`, so program order and loop iterations excuse nothing, and
  `Step.defer` lets an atomic stay pending past any step that does not depend on its old value. My
  Lean checks (`work/lean/L15Sem.lean`, standard axioms only): a one-program `xchg(p, 1); xchg(p, 2)`
  can end with `1` at `p`, so `defer` is not vacuous; `int32` and `int16` atomics at the same address
  are a conflict (the stricter "same bytes" choice, listed in exclusions.md, which PTX's
  morally-strong rule needs). `exists_total_faithful`, `independent`, `shared`, `meets_independent`
  and `valid_grid1` depend only on the standard axioms (`work/lean/RuleAxioms.out`). Re-ran
  `evidence/intra-program`: reversed read-back stale in 41.8% and 23.9%, none with a barrier or at the
  same offsets, 96.9% under pipelining, reversed atomic pairs on 0.3–0.7% of addresses with every
  counter at 2. The `Faithful` docstring, the rule page, SEMANTICS.md, exclusions.md and S1 agree,
  and the changed constants are listed. One stale docstring is filed as wording.
- **L16** ACCEPT — `[reduce]` matches `standard.py:185-190`, `:244-249` and `:270-281` (`tl.max` and
  `tl.min` widen integers below 32 bits to `int32` whatever the sign, `tl.sum` to `int32`/`uint32` by
  sign, the index reductions keep the input type). Re-ran `evidence/int-reductions`: 0 of 35 differ in
  A, 0 of 8 in B, C as recorded.
- **L17** REOPEN — `tl.permute`/`trans` (re-ran `evidence/permute`: all 6 and 24 permutations), the
  index reductions (re-ran `evidence/argmax`: the same outcomes) and `Launch.pdl` (my Lean check:
  `pdl = true` is not `Valid`, `valid_grid1` still proves the 1-D case) are right. But `[no-op:gdc]`
  says that for a valid launch "no dependent kernel is waiting for the signal", and that is false:
  whether a dependent kernel starts early is set by the *next* launch's `launch_pdl`, not by this
  kernel's. `work/pdl_successor.py` (H200, GPU 0): kernel A, at the default `launch_pdl` (so
  `Launch.Valid`), calls `gdc_launch_dependents()`, then reads `x` and writes `y = x + 1`; the next
  kernel B, launched with `launch_pdl=True`, writes `x`. A's `y` differed from `x_init + 1` in 20 of
  20 runs (42% of the elements), and in none without the call or without B's PDL. So A's model, with
  the call dropped, is not what runs, and `Compiler.Faithful` is false for a valid, well-behaved
  launch. Fix: state in `TVA.Claim` and `Compiler.Faithful` (docstrings, rule page) that a claim
  covers a launch only while nothing else writes the bytes it accesses until it ends (no kernel on
  another stream, the host, another device, or a later kernel that PDL starts early); and restrict
  `[no-op:gdc]` for `gdc_launch_dependents()` to launches whose next launch on the stream does not
  use `launch_pdl`, listing the other case in exclusions.md.
- **L18** REOPEN — The environment list is right as far as it goes: checked against
  `knobs.py:343-513` and `GetEnv.h:15-57`, the exceptions only name directories, dump, print, time or
  change line information, and `evidence/env-knobs` re-runs identically. But Triton reads its
  configuration from the `triton.knobs` objects, whose environment variables are only the default,
  and the rule's scope ("none of the environment variables … is set") does not cover the same knobs
  set from Python. `work/options_knobs.py`, each case in a fresh process with no `TRITON_*` variable
  set when the kernel compiles: D2, `triton.knobs.propagate_env = False` then
  `knobs.language.fp32_default = "ieee"`, makes a `tl.dot` without `input_precision` `ieee`
  (`16 + 2^-8`, not `16`), against `[dot]`; A2, `knobs.runtime.add_stages_inspection_hook` replacing
  the PTX stage, gives `y = x - 1` for the model's `x + 1`; S1, `knobs.runtime.debug = True` (a plain
  attribute: `TRITON_DEBUG` is read once, at import), makes a dropped `tl.device_assert` trap. And
  `knobs.runtime.jit_cache_hook` returning true makes `run` return without compiling or launching
  (`jit.py:753`, `:880-882`), a silent no-op launch like L9's. D3: assigning a knob with the default
  `propagate_env` does set the variable, which a kernel reviewer reading host code cannot know. Fix:
  scope `Faithful` (docstring, rule page, SEMANTICS.md, TRANSCRIPTION §9, exclusions.md) to Triton's
  knobs at their defaults however they are set (environment, or assignment to `triton.knobs`), with
  no hook installed (`add_stages_inspection_hook`, `jit_cache_hook`, `jit_post_compile_hook`,
  `knobs.compilation.listener`), except the listed directory, dump and print ones; and say in
  `Launch.debug` that `knobs.runtime.debug` turns `debug` on.

## New issues

- **NEW** soundness — Compile options given at the launch are neither in `Launch` nor excluded.
  `Faithful`'s scope (`debug = false`, `Launch.Valid`, the environment) admits launches whose
  `CUDAOptions` keywords change the compiled code: every field is accepted as a launch keyword
  (`jit.py:713-715`, fields at `nvidia-compiler.py:108-133`), and `ir_override` also as a field of an
  autotune `triton.Config` (`compiler.py:330-333`). `work/options_knobs.py`, no `TRITON_*` variable
  set: A1, `ir_override=<the kernel's own PTX with 1.0 replaced by 2.0>`, gives `y = x + 2` for the
  model's `x + 1`; D1, `allowed_dot_input_precisions=("ieee",)`, makes a `tl.dot` without
  `input_precision` `ieee`, against `[dot]`; E1, `enable_reflect_ftz=False`, compiles `tl.math.exp2`
  to `ex2.approx.f32` (`exp2(-130) = 2^-130`, not `0`). By the source, also `extern_libs` (links other
  bitcode for libdevice calls, `nvidia-compiler.py:458-460`), `ptx_options` (extra `ptxas` flags,
  `:539`; its environment default `PTXAS_OPTIONS` is already excluded), `arch`, `ptx_version`,
  `warp_size`, `supported_fp8_dtypes` and `deprecated_fp8_dot_operand_dtypes`. Where:
  `Rules.Compiler.Faithful`, `TVA.Launch`, `rules/compiler-faithful.md`, TRANSCRIPTION §9,
  exclusions.md. Fix: `Faithful` covers a launch only when every compile or launch option that
  `Launch` does not record is at its default, except a listed set with no effect on the meaning
  (`num_warps`, `num_stages`, `maxnreg`, `launch_cooperative_grid`); the kernel reviewer checks the
  host's launch keywords and every autotune `Config` against it.
- **NEW** soundness — The gate does not stop model text from meaning another operation.
  TRANSCRIPTION §1 forbids declarations "in the `TVA` or `tl` namespaces", but the `namespace` check
  flags only names whose root is `TVA`, and `hygiene` does not list `notation` or its relatives.
  `work/lean/ShadowRoute.lean` declares `Kernels.ShadowRoute.tl.add := TVA.tl.addc`, and
  `work/lean/NotationRoute.lean` writes `local notation "tl.add" => TVA.tl.addc`. In both, the model
  text `tl.add x y`, which the map quotes and `claim.md` shows from the source ranges, is by `rfl` the
  fusable `tl.addc`; `#print` even renders it as `tl.add`. Neither file trips `hygiene` or the
  namespace rule (`work/hygiene_routes.py`, `.out`). Fix: have the claim reader flag every kernel
  constant with a `tl` name component, or whose name ends like a `TVA` constant's; add `notation`,
  `infix`, `infixl`, `infixr`, `prefix`, `postfix` and `binder_predicate` to `HYGIENE`.
- **NEW** gap — A claim is about one launch from `init`, but the library does not say what that
  means for host code that runs the kernel several times, or touches memory just before it. On the
  first call for a key, `@triton.autotune` runs every config several times on the real arguments
  (`autotuner.py:240`; its docstring: "the kernel will run multiple times … whatever value the kernel
  updates will be updated multiple times"), with `reset_to_zero`, `restore_value` and `Config.pre_hook`
  between runs. `knobs.runtime.launch_enter_hook` and a `JITFunction`'s pre-run hooks (`jit.py:735`,
  `:780`) also run host code before the launch. TRANSCRIPTION §9 says only that `Inst` ranges over the
  configs. So for a kernel that accumulates into its output with atomics (`TVA.shared`), the call's
  memory is not the `init` of its last launch, and a kernel reviewer cannot tell from the library
  whether `Pre` covers that. Fix: state in `TVA.Claim` and §9 that a claim covers each launch from the
  memory that launch starts with, that an autotuned call is several launches (benchmark runs, then
  the chosen config), and that hooks run on the host before `init`. The reviewer then checks that
  `Pre` holds at each launch (for example, that `reset_to_zero` names the accumulated outputs).
- **NEW** wording — Stale after L15: the `TVA.Lang.Desc` module docstring still says descriptor
  accesses are "ordinary sequential accesses" that "need Triton to order pointer and TMA accesses to
  the same bytes", and exclusions.md still excludes pointer and descriptor accesses to the same bytes
  for proxy ordering. Under L15 any such pair with a write is a conflict anyway, and two reads need no
  order. Fix: say that the exclusion now adds nothing beyond `IntraRace` (or drop it), and reword the
  docstring.

## Re-ran

- On the H200 NVL (GPU 0, Triton 3.8.0, 2026-10-27), output in `work/rerun_<topic>.out`:
  `evidence/intra-program`, `pdl`, `env-knobs`, `int-reductions`, `permute`, `argmax` and
  `pure-operands`. Every output has the same lines as its `out.txt` once counts are masked, and the
  counts agree with the READMEs.
- Without a GPU: `evidence/gate-replay` and `evidence/gate-digests` (on a scratch copy of the tree, as
  the script does) are identical to `out.txt`; `evidence/independence-audit` gives 63 probes, 0
  wrong, baseline independent.
- Tests added in `work/`: `options_knobs.py` (launch keywords and `triton.knobs`, 11 cases),
  `pdl_successor.py` (a PDL successor overlapping a valid launch), `l12_general_text.py` (the page's
  rule/general split), `l7_more_spellings.py` (home-path spellings), `hygiene_routes.py`; and in Lean,
  `lean/L15Sem.lean` (`Step.defer`, a partly overlapping atomic conflict, `pdl` validity),
  `lean/RuleAxioms.lean`, `lean/NotationRoute.lean`, `lean/ShadowRoute.lean`, each with its `.out`.
- Checked that the pinned `knobs.py` and NVIDIA `compiler.py` equal the 3.8.0 tree, and that the
  tree's `jit.py` and `autotuner.py` equal the installed files (sha256).

## Not checked

- The L15 proof kit beyond the statements of `independent`, `shared`, `meets_independent` and
  `exists_total_faithful` and their axioms: Lean checks the proofs, and the lemmas area is another
  reviewer's.
- `extern_libs`, `ptx_options`, `arch`, `ptx_version`, `warp_size` and the fp8 dtype options were read
  in the compiler source, not run on the device.
- The device facts and their constants (`Rules.Num`; L4 is open and not in this round).
- `tvr check` on the real tree (coordinator's instruction); the gate ran only on the evidence
  script's scratch copy.
- PDL and the intra-program results on GPUs other than the H200.
