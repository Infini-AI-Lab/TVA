# Library review, round 20, by lr-20d

## Verdicts

- **L64** ACCEPT — `FloatV.litF8` (`lean/TVA/Lang/Scalar.lean:361-373`) is what SEMANTICS
  "Constants" now says: `+0` for a zero; the exact `cvt` of `litF32 c` when it is finite and loses
  nothing; `±∞` only in `float8e5`; a NaN by `nanNarrow` only when the `float32` fraction bits it
  drops are zero; `none` (the host's rejection) otherwise. `Operand.constIn?` uses it for both fp8
  dtypes (`lean/TVA/Lang/Promote.lean:175-176`). Read against `scalar_constant`
  (`primitives/triton-3.8.0-semantic.py:601-613`: `get_fp32`, then `create_fp_trunc`, which MLIR
  folds only when lossless). evidence/fp8-constants re-ran in full, identical to its `out.txt` (F1–F4,
  `differences in all: 0`): the constants are never the saturating conversion, and the run-time paths
  (a store, `other=`, `.to`) stay the `CvtSat` facts' (F4), which num.md no longer extends to
  constants. The constants whose meaning changed are listed.
- **L91** ACCEPT — Scope 9's account is now true everywhere it was false (the docstrings of
  `Rules.Compiler.Faithful`, `Launch.Valid` and `TVA.Claim`, the rule page, SEMANTICS, exclusions.md,
  protocol §6, both guides), and the exclusion itself is unchanged and covers both kinds of call.
  evidence/host-round20 re-ran in full, identical: G0 (the capturing stream) ran no program until
  the replay; G1–G3 (another stream, in the modes `global`, `thread_local`, `relaxed`) ran at once,
  and the replay did not run them.
- **L95** ACCEPT — `facts_section` (`tools/tvr.py:2078-2147`) admits only blank lines, `|` rows with
  a blank line around each table, `###`–`######` headings and text lines; anything else (a fence, an
  indented line, a quote, a list, HTML or a comment, a setext underline or rule, a second heading, the
  heading inside a fence or comment) is a problem, on which `stated_facts` returns `None` (the gate
  accepts no fact) and the audit names the line. So what the gate reads as the facts table is what
  the page shows. Judged from the code and evidence/tool-round20 Part L (not re-run: the gate's
  area).
- **L98** ACCEPT — Settled under L103, as the issue asked: SEMANTICS "Constants" gives a NaN that
  reaches the kernel as a compile-time value its pattern only when the launch's code and the helper
  copy are its own, and exclusions.md's NaN row is now a case of the family's row.
  evidence/compile-keys K2 (a launch) and K5 (a helper) re-ran identical.
- **L101** ACCEPT — `Format.e4m3_u = 2^-4`, `e4m3_eta = 2^-10`, `e5m2_u = 2^-3` and
  `e5m2_eta = 2^-17` agree with `u = 2^-(m+1)` and `eta = 2^(emin-m-1)` (biases 7 and 15, `emin`
  −6 and −14); `replay_err`'s docstring names `L.filterMap Prod.snd`; the three READMEs say what their
  scripts run. The new scripts (`num-cvt8/specials.py`, `num-atomic-add/sem_scope.py`) were not
  re-run (the device area).
- **L102** ACCEPT — SEMANTICS "Order" states the runtime conditional expression as `visit_IfExp`
  builds it (each side `to_tensor`ed, one type or a rejection on the host), and "`return`" admits
  only the helpers whose call is the taken `return`'s value (one `return`; tensors of one type; one
  compile-time value of one Python type; no path off the end when a value is returned), with a row
  in exclusions.md for the rest. evidence/front-end-round20 re-ran in full, identical (E1a–E2e).
- **L103** ACCEPT — Scope 11 states Triton's keys: `compute_cache_key` looks the specialisation and
  `str(options)` up by `==` and names the kernel by `str` of the specialisation, which writes each
  value's `repr` (every NaN `nan`; `primitives/triton-3.8.0-jit.py:592-614`), and `ASTSource.hash`
  joins each constant's `str` with `-` (`primitives/triton-3.8.0-compiler.py:71-76`). The helper-copy
  account matches `mangle_fn` and `constexpr_type.mangle`. evidence/compile-keys re-ran in full,
  identical (23 launches ran other values' code). The rule's text gives the general in-process rule
  ("values Python's `==` equates"), and my Y1 (work/keys_extra.py) confirms it beyond the examples
  (`C = 2.0` after `C = 2`, `3.0` after `3`, ran the integer kernel); the reviewer's checklist lists
  only the examples (new wording issue below).
- **L104** REOPEN — The limit and the reviewer's check are right for what the evidence shows
  (host-round20 M0–M4 re-ran identical), but scope 10's text is wider than the rest of the library
  allows. "No code of the process … changes a module, class or object of Triton …, other than the
  `JITFunction`'s `debug`" (`lean/TVA/Rules/Compiler.lean` scope 10; compiler-faithful.md scope 10
  and checklist item 10; REVIEWER_GUIDE item 10, "assigns to a Triton module, class or object";
  PROVER_GUIDE (10); SEMANTICS "Names" and "Launches"; the exclusions row) also excludes Triton's own
  API, which other scopes require or allow: `triton.set_allocator(alloc_fn)`, which sets the module's
  `_allocator` (`primitives/triton-3.8.0-allocation.py`) and which scope 5 requires of a kernel that
  builds descriptors (tutorials 08, line 364, and 09, line 589, call it); the assignment of a knob
  scope 2 allows (`triton.knobs.cache.dir`); and a config `pre_hook`'s
  `nargs["desc_q"].block_shape = …` on a host `TensorDescriptor` (a class of `triton.tools`; tutorials
  06, lines 119-121, and 09, lines 188-193), which "Which launch" covers. Read as written, the in-scope
  hosts of tutorials 06, 08 and 09 are outside the rule. Fix, in each of those places: keep the wide
  exclusion and name the exceptions: "other than through Triton's API: the `JITFunction`'s `debug`
  (scope 4), `triton.set_allocator` (scope 5), the knobs scope 2 allows, and the fields of a host
  `TensorDescriptor` the launch is given (a hook may set its `block_shape`, 'Which launch')".
- **L105** ACCEPT — `JITFunction.run` takes `kwargs.get("debug", self.debug) or
  knobs.runtime.debug` (`primitives/triton-3.8.0-jit.py:727`), so `Launch.debug`'s docstring is now
  true of a JIT launch and a warm-up (for `triton.compile`, which reads neither the attribute nor the
  knob, it over-states `debug`, which only narrows a claim). `preload` compiles with its data's
  options and stores the code under its data's key (`jit.py:823-875`), and scope 4 covers no launch of
  a kernel it was called on in the process; in another process it writes the disk cache only under
  the key of the options it really used (`compile`'s `get_cache_key`), so the in-process limit
  suffices. host-round20 D0/D1 and P0/P1 re-ran identical.
- **L106** ACCEPT — The reader stops unless `isStructure` holds (`tools/claim_reader.lean.in:255-259`),
  and `reader_result` (`tools/tvr.py:2003-2017`) fails a non-zero exit, any count of result lines but
  one, and any `PANIC` line. evidence/tool-round20 Part S re-ran, identical to its `out.txt` but for
  the truncated scratch path (lr-18a's `inductive` refused by the tree's reader, the round's-start
  reader's panic failed by `reader_result`, a `def` refused by both).
- **L107** ACCEPT — The host is named wherever the target is (SEMANTICS's introduction and
  "Launches", scope 7 in Lean and on the page, the checklist's and both guides' item 7, exclusions.md's
  targets row, `TVA.Claim`), and `pyproject.toml` pins `>=3.13,<3.14`. On TVA's host (x86-64, CPython
  3.13.9) `struct.pack('>e', …)` (`PyFloat_Pack2`) dropped a NaN's payload (`0x7e00`, `0xfe00`) and
  `inf - inf` was `0xfff8000000000000`, as stated. The host process's floating-point environment also
  changes stated values on this host; that is filed below as a new issue, not against this fix.
- **L108** ACCEPT — Checked: `tl.cdiv`'s two cases, the exceptions of the two general sentences
  (`tl.static_assert`'s `msg`, the first visit of a loop body, code after an `if` that returns),
  `IntV.le_toInt` (`lean/TVA/Lemmas/SharedAcc.lean:367`, for both signednesses), `FloatV.nan`'s
  docstring, float-layer's conclusion 4, and `normal_path`/`is_work_file` in the code. The
  tool-round18 re-run, Part F and every "one rounding's bound" rewording were not re-run or read in
  full.

## New issues

- **NEW** soundness — Scopes 2, 3 and 10 of `Compiler.Faithful` constrain the claimed launch's own
  process at the launch, but a launch may run code compiled earlier, under conditions those scopes
  exclude, that Triton's cache keys do not record. The in-process key is the specialisation and
  `str(options)` (`primitives/triton-3.8.0-jit.py:592-614`); the disk key is `triton_key()`, the
  source's hash, the backend, the options and the environment variables of
  `CACHE_INVALIDATING_ENV_VARS` (`primitives/triton-3.8.0-GetEnv.h:15-57`; Triton 3.8.0's
  `runtime/cache.py`, `get_cache_key`). Neither holds a knob the front end reads
  (`knobs.language.fp32_default`, `primitives/triton-3.8.0-core.py:2387`) set from Python with
  `propagate_env = False`, nor `TRITON_KERNEL_OVERRIDE` (not in that list), nor a change to Triton's
  objects in memory. On the H200 (work/cache_cross_process.py, `.out`), a process with all of Triton
  at its defaults, on a cache folder an earlier process had used, ran that process's code: `log 2`
  for `tl.exp(2.0)` after `tl.exp = tl.log` there (X1), the `ieee` `tl.dot` after a Python-set
  `fp32_default = "ieee"` there (X2), `5 + 2` for `x + 1.0` after a `TRITON_KERNEL_OVERRIDE` there
  (X3); a fresh folder gave the model's values, and `TRITON_F32_DEFAULT` in the environment, which
  is in the key, did not leak (X4). Within one process (work/knob_history.py), a launch with
  `fp32_default` back at its default ran the `ieee` code compiled while it was `"ieee"`, with
  `propagate_env` either way (Z1, Z2). Scope 11 states this channel for compile-time values only, and
  no reviewer reading the claimed program's host can see another process. Fix (the Lean docstring's
  scopes 2, 3, 10; compiler-faithful.md's scopes and checklist items 2, 3, 10; REVIEWER_GUIDE's and
  PROVER_GUIDE's items 2, 3, 10; SEMANTICS "Launches"; exclusions.md's knob, hook and Triton rows;
  `TVA.Claim`'s module docstring): these conditions hold at every compile whose code the launch may
  run, at any time before the launch in its process and in every process that has written the cache
  directory it uses; a process that keeps them from its start, with a cache directory no other
  process writes, meets it, and so does one that sets `TRITON_ALWAYS_COMPILE=1` (an exception of
  scope 2), which ignores what the directory holds (work/keys_extra.py Y2: it also kept `MODE = "1"`
  from `MODE = 1`'s code).
- **NEW** soundness — The host process's floating-point environment changes values SEMANTICS states,
  on the host scope 7 names. `get_fp32` takes pybind's C++ `float` of the Python float
  (`primitives/triton-3.8.0-ir.cc:963-967`), the launcher converts a `float` argument with C's
  `(float)` (`FloatV.hostF32`, "Scalar arguments"), `get_bf16` prints with C's `%f`, and the front end
  computes compile-time float arithmetic with Python's floats: each uses the process's rounding mode
  and its MXCSR flush-to-zero and denormals-are-zero bits, which host code changes with no Triton
  knob (`fesetround`; loading a shared library built with `-ffast-math`, whose start-up code sets
  FTZ/DAZ). On the H200, x86-64, CPython 3.13.9 (work/host_env.py, `.out`): after
  `fesetround(FE_TOWARDZERO)`, `tl.full([1], 0.1, tl.float32)` and the argument `y = 0.1` stored
  `0x3dcccccc` (`litF32` and `hostF32` give `0x3dcccccd`), the constexprs `A + B` (`0.1`, `0.2`)
  gave `0x3fd3333333333333` (Python's `0.30000000000000004` is `…334`), and
  `tl.full([1], 1.5e-6, tl.bfloat16)` stored `0x3586` (`litBF16`, from the text `0.000002`:
  `0x3606`); with FTZ and DAZ set,
  `tl.full([1], 1e-40, tl.float32)` stored `0` (`litF32`: `0x000116c2`), and `C * 1e300` with the
  constexpr `C = 1e-310` gave `0`. (A numeric locale whose decimal point is a comma, `en_DK`, makes
  `get_bf16`'s `std::to_string` text unparseable, and LLVM's assertion aborts the process before the
  launch: no outcome, so no defect.) Fix, as a stated limit with the reviewer's check (the user's rule
  on a malicious host): scope 7 in Lean and on the page, checklist item 7, both guides' item 7,
  SEMANTICS's introduction and "Launches", exclusions.md's targets row and `FloatV.hostF32`'s
  docstring add that the host process keeps the default floating-point environment, rounding to
  nearest even with flush-to-zero and denormals-are-zero off (no `fesetround` or `_mm_setcsr`, and no
  loaded library that changes them), as TVA's evidence ran.
- **NEW** wording — REVIEWER_GUIDE's host item 11, compiler-faithful.md's checklist item 11 and
  PROVER_GUIDE's (11) give the in-process collisions as a list (`1`, `1.0` and `True`; `0`, `0.0`,
  `-0.0` and `False`) that reads as the whole rule, while every pair Python's `==` equates shares the
  key: `C = 2.0` after `C = 2`, and `3.0` after `3`, ran the integer kernel (work/keys_extra.py Y1).
  Scope 11's own text says "values Python's `==` equates". A reviewer who follows the list misses
  `SCALE = 2` then `SCALE = 2.0`. Fix: lead each list with the rule: "values Python's `==` equates
  (an integer and the float of the same value, such as `2` and `2.0`; `True` and `1`, `False` and
  `0`; `0.0` and `-0.0`), values with the same `str` (every NaN; on disk a value and its `str`, `1`
  and `"1"`), and the `str`s joined with `-` across parameters".

## Re-ran

- In full, each identical to the folder's output but for the first line: evidence/compile-keys
  (work/rerun-compile-keys.txt), evidence/host-round20 (work/rerun-host-round20.txt),
  evidence/fp8-constants (work/rerun-fp8-constants.txt), evidence/front-end-round20
  (work/rerun-front-end-round20.txt); Triton 3.8.0, torch 2.11.0+cu130, CPython 3.13.9, H200 NVL,
  GPU 0, 2026-10-28.
- evidence/tool-round20 Part S only (work/rerun-tool-round20-S.txt): identical but for the truncated
  scratch path.
- New tests, same setup: work/cache_cross_process.py (X1–X4: code compiled by an earlier process
  outside scopes 2 and 10 reaches a later one through the cache folder), work/knob_history.py (Z1, Z2:
  the same within one process), work/host_env.py (rounding mode, FTZ/DAZ, a comma locale),
  work/keys_extra.py (Y1: `==` collisions beyond the listed ones; Y2: `TRITON_ALWAYS_COMPILE=1`).
- By hand: fp8's `u` and `eta`; on CPython 3.13.9, `PyFloat_Pack2` of NaNs and the sign of
  `inf - inf`.

## Not checked

- L95: tool-round20 Part L and the audit on scratch pages were not re-run (the gate's area); judged
  from the code.
- L101: num-cvt8's `specials.py`, num-atomic-add's `sem_scope.py` and num-dot's ratios were not
  re-run (the device area); judged from the READMEs and the Lean lemmas.
- L102: the helper forms beyond front-end-round20's cases (the semantics and transcription areas).
- L108: tool-round18's re-run and tool-round20 Part F were not re-run; the "one rounding's bound"
  rewording was not read in every Lean file.
- The release check (lr-20c only), and the versions of torch and the NVIDIA driver as part of the
  host.
