# Library review, round 20, by lr-20a

## Verdicts

- **L64** ACCEPT — `FloatV.litF8`, and `Operand.constIn?` on the two fp8 dtypes, are what `scalar_constant` builds (`semantic.py:601-613`: `+0` for a zero, otherwise `get_fp32` then `arith.truncf`) and what `TruncFOp::fold` folds (`llvm-ArithOps.cpp:1588-1611, 1788-1825`: `±∞` and NaN into `float8e4nv` fail, a NaN into `float8e5` folds only when no fraction bit is lost, a number only when it is exact). I re-ran evidence/fp8-constants (output identical, `differences in all: 0`) and added cases it lacks (`work/probes.out` P1, against `work/Fp8Probe.out`): `1 + 2^-30` stored `0x3c`/`0x38`, `-1e-50` `0x80` (in `tl.full` and in `tl.where`), `1e39` `0x7c`, the `int`s `3` and `448` `0x42` and `0x7e` (also in `tl.where`), and `True` `0x3c`, each equal to `litF8`. The account in num.md ("Conversions into fp8") and TRANSCRIPTION's pitfall are corrected. The summary in SEMANTICS' "Values" row is loose (filed below, wording).
- **L91** ACCEPT — Re-ran evidence/host-round20 (output identical). G0: the call on the capturing stream kept the sentinel, and the replay wrote `x + 1`. G1–G3: a call on another stream ran at once, and the replay left the sentinel. Round 16's `graph_capture.out` G3 shows the compile during a capture. The exclusion is unchanged, and its reason now reads the same in scope 9, `Launch.Valid`, `TVA.Claim`, the rule page, SEMANTICS, exclusions.md, protocol §6 and both guides.
- **L95** ACCEPT — Judged from `facts_section` and `stated_facts` (`tools/tvr.py`). The section is read through an allowlist of forms: blank lines, `|` rows, `###` headings, and text that is not a list, with a blank line around each table. A second heading, or a heading inside a fence or a comment, is a problem. Any problem makes `stated_facts` return `None`, so the gate fails closed. Not run (the gate area).
- **L98** ACCEPT — Re-ran evidence/compile-keys (output identical); K2 and K5 are the NaN cases. `mangle_fn` (`code_generator.py:32-41`) and `constexpr_type.mangle` (`core.py:196-201`) write a compile-time argument by its `repr`, so every NaN is `nan`. SEMANTICS "Constants" covers a NaN compile-time value only when the launch's code and the helper copy are its own; two helper calls whose copies collide are outside the subset (L103), and a kernel argument is scope 11.
- **L101** ACCEPT — The new theorems have the right values: `Format.e4m3_u = 2^-4` and `e5m2_u = 2^-3` (`2^-p`); `e4m3_eta = 2^-10` and `e5m2_eta = 2^-17` (half the smallest subnormals, `2^-9` and `2^-16`). `IntV.le_toInt` states `<=` for either signedness. `replay_err`'s docstring names `L.filterMap Prod.snd`. The outputs of the new num-cvt8 `specials.py` and num-atomic-add `sem_scope.py` were read, not re-run.
- **L102** ACCEPT — Re-ran evidence/front-end-round20 (output identical). Checked against SEMANTICS "Order" and "A helper's value":
  - `visit_IfExp` (`code_generator.py:998-1036`) `to_tensor`s each side in its own block, then asserts that both sides have one type;
  - `decide_return_type`, `cast_to` and `handle_returns` (`:556-626`): returns of `==`-equal compile-time values give the first one's value (`constexpr_type._unflatten_ir`); other returns become tensors of one type; a path that falls through returns `poison`.

  The helper forms the subset admits give the value of the `return` the program takes (E2d, E2e).
- **L103** REOPEN — The rule's own condition is right: no earlier launch with other compile-time values that share the key. evidence/compile-keys re-ran identically. But the sufficient conditions given to the kernel reviewer are false, so a reviewer who follows them accepts hosts that the evidence shows running the wrong code. They are:
  - REVIEWER_GUIDE item 11: "A fresh cache directory, or values Triton keys apart, meets it";
  - compiler-faithful.md checklist item 11: "a fresh cache directory, or each compile-time parameter given values that Triton keys apart, meets it";
  - the last sentence of scope 11 (`Rules.Compiler.Faithful`'s docstring and the rule page): "on a fresh cache, or with each compile-time parameter given values that Triton keys apart".

  Counterexamples, in `work/probes.out`:
  - P2a: one process with a fresh cache directory launches `C = 1` and then `C = 1.0`. The second stored `16777217`; `C = 1.0` alone stores `16777216` (K1 also ran on fresh caches).
  - P2b: one process with a fresh cache directory launches `(A, B) = ("a-b", "c")` and then `("a", "b-c")`. Each parameter's two values are keyed apart, yet the second stored `31`; alone it stores `13`.

  Fix, in those three places, by replacing the shortcut with what meets the condition:
  - in the process, no other launch or warm-up of the kernel had compile-time values that differ from this launch's but are `==` element by element, or have the same `repr` (every NaN);
  - through the cache directory since it was empty, no launch had values whose `str`s, over all compile-time parameters together, join with `-` to the same text;
  - the values are compared as one tuple over all parameters, not parameter by parameter;
  - a cache directory new to the process meets only the second condition.

  Also, in REVIEWER_GUIDE item 11's list of compile-time values, add an integer `1` and `None` passed to a parameter without annotation: `specialize.cc` makes both constexprs, and scope 11 names the first.
- **L104** REOPEN — M1–M4 re-ran identically, and item 10 catches them. But scope 10 as written also forbids what in-scope hosts must do, so REVIEWER_GUIDE does not tell the reviewer exactly what to check. Scope 10 of `Rules.Compiler.Faithful` (also the rule page, SEMANTICS "Names" and the last row of exclusions.md) says that no code of the process, "a library it imports" included, "changes a module, class or object of Triton". REVIEWER_GUIDE item 10 and checklist item 10 say that no code "assigns to a Triton module, class or object". The in-scope hosts do both (`work/scope10.out`):
  - tutorials 06 and 09: their config `pre_hook`s assign `block_shape` on host `TensorDescriptor`s, objects of Triton's class `triton.tools.tensor_descriptor.TensorDescriptor` (06:119-125, 09:188-193). "Which launch" names this case as covered;
  - tutorials 06, 08 and 09 call `triton.set_allocator`, which sets Triton's `_allocator`. Scope 5 requires an allocator;
  - scope 2 allows a knob assigned its default;
  - Triton's own code changes Triton's objects at every launch.

  Read literally, every in-scope host that uses descriptors is `CONDITION_WRONG`; read loosely, the reviewer must guess where the rule stops. In the other direction, "assigns to" leaves out changes made by a call (`setattr`, `triton.runtime.driver.set_active(…)`, `importlib.reload` of a Triton module). Nothing states that the `triton` the process imports is the installed one (not another `triton` earlier on `sys.path`, not a substitute in `sys.modules`), or that the cache directory holds only what Triton 3.8.0 wrote.

  Fix: state scope 10 the same way in all those places and in PROVER_GUIDE (10):
  - (a) the process imports Triton 3.8.0's installed package, unmodified (no other `triton` on `sys.path`, no substitute in `sys.modules`), and the cache directory holds only files Triton 3.8.0 wrote;
  - (b) no code but Triton's own changes an attribute of a Triton module, a class Triton defines, or the kernel's or a helper's `JITFunction`, whether by assignment, by `setattr`/`delattr`, or by a call made to change it (`_unsafe_update_src`, `driver.set_active`, `importlib.reload`). The `JITFunction`'s `debug` is scope 4, and the knobs are scope 2;
  - (c) explicitly allowed: Triton's API used as the in-scope tutorials use it (launches, `warmup`, `triton.autotune` with its `triton.Config`s and hooks, `triton.set_allocator`), and setting the fields of a host `TensorDescriptor` the launch is given (`block_shape` in a `pre_hook`; scope 8 still bars `round_f32_to_tf32=True`);
  - (d) the clause on the names of the kernel's module, unchanged.
- **L105** ACCEPT — Re-ran evidence/host-round20 (output identical). D1: with `kmod.add_i32.debug = True` the overflow trapped. P1: a plain launch after `preload` gave `2^-130`. `jit.py:727` is `kwargs.get("debug", self.debug) or knobs.runtime.debug`, as `Launch.debug` now says. A helper's own `debug` attribute is never read (the callee's `CodeGenerator` gets the caller's `options`, `code_generator.py:1372-1378`), so only the kernel's attribute counts. `preload` installs its code only in the process's kernel cache (the disk key includes the options), so "in the process" is enough.
- **L106** ACCEPT — Judged from the code. `claim_reader.lean.in:255-259` throws unless `isStructure env (ns ++ `Assumptions)`. `reader_result` passes only on exit `0` with one `TVR_JSON` line and no `PANIC` line, so it fails closed. Not run (the gate area).
- **L107** ACCEPT — The host (x86-64, CPython 3.13) is named in SEMANTICS (its introduction and "Launches"), scope 7, the rule page, exclusions.md, `TVA.Claim` and both guides. `pyproject.toml` requires `>=3.13,<3.14`. TVA's `.venv` is CPython 3.13.9 on x86_64, and every output I re-ran prints Python 3.13.9.
- **L108** ACCEPT — Checked in my area:
  - `tl.cdiv`: `div - 1` is Python's for a compile-time `div`;
  - the two general sentences now name their exceptions; `tl.static_assert`'s `msg` is evaluated only when the assertion fails (`code_generator.py:1617-1637`);
  - `FloatV.nan`'s docstring;
  - no text still calls `u·|y| + η` half an ulp;
  - `IntV.le_toInt`.

  The tool points (tool-round20 Part F, tool-round18's re-run) were not run.

## New issues

- **NEW** wording — The fp8 row of SEMANTICS "Values" says "a Python number built in them is its exact value, or the kernel is rejected". That is false for a double whose `float32` constant differs from it, and none of these is rejected (`work/probes.out` P1; `litF8` agrees, `work/Fp8Probe.out`):
  - `1 + 2^-30` builds `1.0` (`0x3c`, `0x38`);
  - `1e39` builds `+∞` (`0x7c`);
  - `-1e-50` builds `-0` (`0x80`).

  A prover who reads the row expects a host error for these. Fix: "the exact value of its `float32` constant (`FloatV.litF32`), or the kernel is rejected (`FloatV.litF8`, "Constants")".

## Re-ran

- In full on GPU 0 (H200 NVL, Triton 3.8.0, CPython 3.13.9, 2026-10-28), output in `work/rerun-*.out`. All four are identical to the folders' `out.txt`:
  - evidence/fp8-constants (`differences in all: 0`);
  - evidence/front-end-round20;
  - evidence/compile-keys (`23` launches ran other values' code);
  - evidence/host-round20.
- Added `work/probes.py` (output `work/probes.out`): P1, fp8 constants that evidence/fp8-constants lacks, against `litF8` evaluated by Lean (`work/Fp8Probe.lean`, `work/Fp8Probe.out`); P2a and P2b, the counterexamples to the scope-11 shortcuts (L103).
- Added `work/scope10.py` (output `work/scope10.out`, no GPU): the Triton objects that the in-scope hosts set (L104).

## Not checked

- L95 and L106 were judged from the code and the notes only; tool-round20 Parts L and S were not run (the gate area).
- L101: the new device outputs (num-cvt8 `specials.out`, num-atomic-add `sem_scope.out`) were read, not re-run.
- L108: the tool points (tool-round20 Part F, tool-round18 `rerun-round20.txt`) were not run.
- The truth of the facts listed in num.md's "Facts the library states" (the device area). The release check was not run (lr-20c's).
- Outside the round's changes, my survey of `lean/TVA/Lang`, `lean/TVA/Sem` and SEMANTICS read the texts and the cited sources. It re-ran no older evidence folder.
