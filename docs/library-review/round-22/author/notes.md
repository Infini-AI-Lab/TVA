# Library batch, round 22: author's notes

Issue: L116 ([ISSUES.md](../../ISSUES.md)), 2026-10-29: one issue, from the user's decision on
autotuning and the user's correction of it the same day, with no review after this batch
(coordinator). Worked in TVA's tree itself.

**Build and checks.** `lake build TVA` is green (3192 jobs). No definition, statement or proof
changed, only docstrings, so no axiom changed (no theorem was added); no `sorry` or `axiom` in the
library. `tvr library-check` passes (hygiene, and `leanchecker TVA` re-checks every declaration).
`tvr check vec_add_f32`, run once after the last library change, passes: gate PASS, digest
`19119d88d3f01d9c`. Its items re-open, as the coordinator expected, because `docs/SEMANTICS.md`,
`docs/exclusions.md`, `docs/rules/compiler-faithful.md`, `lean/TVA/Claim.lean` and
`lean/TVA/Rules/Compiler.lean` are hashed whole; no kernel stops building.

**Release check.** Run at the end, from TVA's root (after `tvr check vec_add_f32` and these
notes): `release check PASS (12 of 12 steps pass)`, 1211 shipped files, kept as
[evidence/tool-round22/release.out](../../../../evidence/tool-round22/release.out) and
`release.json` beside it, with a README.

## L116: the autotuner is not verified; for an autotuned tutorial kernel we pick one config when we set up its folder

- **What changed.** TVA verifies the kernel itself, and the autotuner is no part of a kernel. In
  every place that spoke of autotuning (`grep -rn -i autotun docs lean/TVA`, the review records
  aside):
  - `lean/TVA/Rules/Compiler.lean`, the docstring of `Rules.Compiler.Faithful`:
    - the scope's list of issues gains L116, and the user's two statements on the autotuner are
      quoted with the earlier decisions;
    - part (A) lists `triton.heuristics` as before, and `triton.autotune` as outside the supported
      subset, where it listed "`triton.autotune` with its `triton.Config`s and their hooks"; the
      launch call is "a direct launch of the `@triton.jit` function";
    - condition 11 reads as the decision says: the code the launch runs was compiled for it, from
      its own source, compile-time values and settings, "or earlier for the same source,
      compile-time values and settings (a repeated launch of the same instance)". The mention of an
      autotuner's benchmark run is dropped, and so is "an autotune config's values" among the
      compile-time values (a config's meta-parameters are now `tl.constexpr` arguments of the
      launch);
    - "Which launch" is rewritten: the rule is about one direct launch of the `@triton.jit`
      function (`kernel[grid](…)`, or of its `CompiledKernel` within scope 6), its `init` the memory
      when it starts. `@triton.autotune` is outside the supported subset, and the rule covers no
      call of an autotuned kernel: its benchmark runs, key, pruning, `reset_to_zero` and
      `restore_value`, `do_bench` and choice are not verified and are outside every claim. For a
      kernel a tutorial autotunes, the user and the coordinator pick one config when they set up the
      kernel's folder, before any prover starts: the source is the tutorial's without the
      decorator, the launch passes the config's meta-parameters and launch options as the
      autotuner's own launch passes them (`Config.all_kwargs`), and where the config has a
      `pre_hook` the host code does what the hook does before the launch (tutorials 06 and 09 set a
      host `TensorDescriptor`'s `block_shape`). That launch is a direct launch, covered as any
      other.
  - `lean/TVA/Claim.lean`, the module docstring: a claim covers one direct launch of the kernel's
    `@triton.jit` function and no call of an autotuned kernel (it said: "for an autotuned call, the
    final launch, with the arguments and the memory the last hook leaves"); L116 joins its list of
    issues.
  - `docs/rules/compiler-faithful.md`: the same changes to "Scope" (the list of issues, the quotes,
    part (A), point 11) and to "Which launch", in three paragraphs (the direct launch; "The
    autotuner is not verified", with the evidence of what a call runs; "A kernel a tutorial
    autotunes", covered through the config chosen at setup, the prover proving the kernel as given
    and the kernel reviewer judging that kernel, not the choice of config). "What it does not
    cover" names the autotuner with the host code around the launch. The checklist asks that the
    launch be a direct launch of the `@triton.jit` function, a source with `@triton.autotune` being
    outside the supported subset, where it asked that the launch covered be the autotuner's last.
  - `docs/SEMANTICS.md`: "The supported subset" no longer admits one `@triton.autotune(…)` above
    the kernel ("not under `@triton.autotune(…)`, whose autotuner TVA does not verify"), and lists
    `@triton.autotune` (L116) among the forms outside it. "Launches", "One launch" says what "Which
    launch" says; the scope's list of issues gains L116; its point "the kernel is new" reads as
    condition 11.
  - `docs/exclusions.md`: a new row, the autotuner (`@triton.autotune` on a kernel, and a call of
    an autotuned kernel with its benchmark runs, key, pruning, `reset_to_zero` and
    `restore_value`, hooks, `do_bench` and choice), not verified by the user's decisions, quoted;
    the row on tutorial 09's pointer kernels speaks of the config chosen when the folder is set up,
    and of the `BLOCK_SIZE_M` and `BLOCK_SIZE_N` the launch passes, not of "every config the
    autotuner may pick".
  - `docs/REVIEW_PROTOCOL.md` §3 says where a kernel folder is set up: a new bullet "Setting up a
    kernel folder". The user and the coordinator (whoever creates `kernels/<kernel>/`) set up each
    folder before any prover starts. For a kernel a tutorial autotunes they pick one config of the
    tutorial's list at this step; the source they give `tvr new` is the tutorial's without the
    decorator, its launch passing the config's meta-parameters and launch options, the host code
    doing what a config's `pre_hook` does. They record in `SETUP.md`, in the kernel's folder, which
    config they chose and where it is in the tutorial (the file, the kernel, the `triton.Config` or
    the list and position it comes from), and what they changed from the tutorial's source. The
    prover does not write `SETUP.md` or choose the config, is not told about autotuning, and proves
    the kernel as given in `source/`; the kernel reviewer judges that kernel, not the choice, and
    files `UNSUPPORTED` on a source that still has `@triton.autotune`. The folder's table gains
    `SETUP.md` and gives `source/<file>.py` the author "setup" (it said "prover"). §2's
    "the conditions" no longer names "the autotune configs".
  - `docs/PROVER_GUIDE.md` says nothing about choosing a config: the sentence on the autotuner's
    last launch is gone, and the bullet "Stay inside the supported subset" adds only that a source
    with `@triton.autotune` is outside the subset, and to ask the coordinator to set the kernel up
    again.
  - `docs/REVIEWER_GUIDE.md`: §1's list of forms outside the subset names `@triton.autotune`;
    "The launch" no longer lists "the autotune configs"; the paragraph on the autotuner's last
    launch now says that the launch is a direct launch, that a source with `@triton.autotune` is
    `UNSUPPORTED`, and that the reviewer judges the kernel as given in `source/`, not the choice of
    config.
  - `docs/TRANSCRIPTION.md` (guidance): its examples of forms outside the subset name
    `@triton.autotune`, and the host pitfall says that `init` is the memory when the direct launch
    starts, where it said that `Inst` ranges over the configs the autotuner may pick.
  - `docs/prompts/kernel-reviewer.md` no longer asks whether the conditions cover "the autotune
    configs"; `docs/prompts/library-coverage.md` lists `@triton.autotune` as excluded (L116), a
    kernel a tutorial autotunes being set up with one config, where it listed "autotuning and its
    hooks" among the constructs to cover.
  - Left as they were, as the coordinator asked (nothing else changes): `triton.heuristics`
    everywhere; condition 4's "in an autotune `triton.Config`" and its checklists' "a
    `triton.Config`", now moot and still true; condition 10's part (c), which allows Triton's API
    as the in-scope tutorials use it, `triton.autotune` included (other code of the process may
    still use the autotuner); the docstrings of `TVA.Launch.fusion` and `TVA.Launch.debug` and
    SEMANTICS's `debug` point, which state as a fact of Triton that the options of an autotuner's
    chosen config reach the JIT; the evidence pages, and the record of point 11 (an autotuner's
    chosen config ran the code of its benchmark run of another, `evidence/host-round21` K4), which
    still shows why the caches are an assumption; `tools/tvr.py`, whose map check still classes a
    function under `@triton.autotune` as a JIT function.
- **Why.** The user's decision, 2026-10-29 (`ISSUES.md`, under L36): "actually we do not
  neccessarily cover autotune. we do verification basically for the kernel. that is to say, for
  autotune, we actually should prove the selected hyper-parameters. for the kernels in the tutorial
  with autotune, we can select a reasonable autotune config to provide proof." And the user's
  correction, the same day: "our principle is, we are verifying kernel itself. it is not correct
  that 'For an autotuned tutorial kernel, the prover picks a reasonable config from the tutorial's
  list and says which. The reviewer checks that the claim fixes it'. actually not the prover picks
  a config, it is we pick a config. prover is not aware of the autotune." The coordinator relayed
  the correction while the batch ran, with what to state: `@triton.autotune` outside the supported
  subset, with a row in exclusions.md; the config chosen by the user and the coordinator when they
  set up the kernel's folder, recorded in the folder; the prover not told about autotuning; the
  kernel reviewer judging the kernel as given in `source/`; REVIEW_PROTOCOL saying where kernel
  folders are set up; "Which launch" and part (A) covering a direct launch of the JIT function and
  no autotuned call; condition 11 read as the first decision says. Before this batch the library
  covered the autotuner's final launch, with one of the configs it may pick, from the memory its
  benchmark runs and hooks leave, so a claim spoke of the autotuner's behaviour; it now speaks only
  of the kernel launched with one config's values. The direct launch is the launch the autotuner
  itself makes after choosing a config: `Autotuner.run` runs the config's `pre_hook` and then
  `self.fn.run(*args, **kwargs, **config.all_kwargs())`, where `all_kwargs` is the config's
  `kwargs` with `num_warps`, `num_ctas`, `num_stages`, `maxnreg` and `ir_override` when not `None`
  (`primitives/triton-3.8.0-autotuner.py:273-280, 365-377`).
- **Evidence.** None new: the change is a decision of scope, and the facts it states are Triton's
  sources, pinned in `primitives/triton-3.8.0-autotuner.py` (`_bench` 133-173, `run` 217-282,
  `Config.all_kwargs` 365-377), and [evidence/autotune-launch](../../../../evidence/autotune-launch/README.md)
  (what a call runs around its launch). No GPU was used.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful`: its definition is unchanged,
  and its docstring, the rule's stated scope, now covers a direct launch of the `@triton.jit`
  function and no call of an autotuned kernel (it covered the autotuner's final launch, from the
  memory after the last hook), and states condition 11 as "compiled for it, or earlier for the same
  source, compile-time values and settings". No other constant: the `TVA.Claim` module docstring
  changed (no constant's), and the library files hashed whole that changed are
  `lean/TVA/Rules/Compiler.lean`, `lean/TVA/Claim.lean`, `docs/rules/compiler-faithful.md`,
  `docs/SEMANTICS.md` and `docs/exclusions.md`. Left open: no tool writes or checks `SETUP.md`
  (`tvr new` copies the source it is given), so the setup record is kept by hand.
