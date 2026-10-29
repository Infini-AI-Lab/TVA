# Library review, round 17, by lr-17a

## Verdicts

- **L58** ACCEPT — The bound SEMANTICS "The contiguity the analysis infers" and the hint section of `TVA.Lang.Ops` now give a value carrying a `max_contiguous` (alone or in the pair) is the smaller of the hinted `C` and the list's bound without the hint. That is a lower bound in every case the subset admits:
  - When the analysis visits the marked operation, the hint replaces the computed contiguity and the result is `C` (`AxisInfo.cpp:1232-1259`, or `1309-1341` for an operation no visitor handles).
  - In every other case the mark is gone: a loop's or a runtime `if`'s result, an inlined call's, a lost several-result mark, a fold. On a function parameter `set_attr` writes nothing (`ir.cc:411-425`). The analysis then keeps its own contiguity, and the list bounds it from below: the list's items are the make-range, add, sub, addptr, expand-dims, broadcast and integer-cast visitors (`AxisInfo.cpp:138-183, 242-279, 553-668`), and each gives a result no lower than the list's bound when its operands' contiguities are at least theirs.
  - Neither the pinned `Combine.cpp` nor `llvm-ArithCanonicalization.td`/`llvm-ArithOps.cpp` copies an attribute, and a rewrite that drops a mark only leaves the analysis its own contiguity (see "Not checked" for the generated and unpinned rewrites).
  - Every several-result operation has the list's bound `1` anyway, so the `min` costs nothing there.
  - `multiple_of`'s and `multiple_ofN`'s `contig` docstrings say the same. No in-scope kernel needs more: tutorial 06 hints scalars, and tutorial 09 uses the pair, whose promise does not use the bound (tutorial 08's kernel is excluded).
  - evidence/hint-unread-round17 N0-N10 and lr-15a's B0-B5 re-ran identical.
- **L88** ACCEPT — "`tl.cdiv`" matches Triton's body `(x + (div - 1)) // div` (`standard.py:32-43`). It matches how `call_JitFunction` binds it: a Python number stays a constexpr (`normalize_value`), and a body that returns one compile-time value gives a compile-time result (`:556-585, 1396`). So it is Python's arithmetic on two compile-time values, and the wrapping `+` and truncating `tl.idiv` with a tensor. evidence/cdiv-round17 C1-C6 re-ran identical, every `int32` `x` for each of the eight divisors included. My `work/call_scan.py` lists every call shape (callee, positional count, keywords) of the 23 in-scope functions:
  - every `tl.*` call, method (`.to`, `.reshape`, `.permute`, `.split`, `.trans`, a descriptor's `.load`/`.store`), builtin (`float`, `isinstance`, `min`, `range`) and keyword used in a kernel is stated in SEMANTICS or `TVA.Lang`;
  - `list`, `filter` and the config builders appear only in `@triton.autotune`'s arguments, which run on the host;
  - evidence/subset-calls-round17 re-ran identical.
- **L89** ACCEPT — Excluded clearly, with one rule in each place: the subset's `return` item and its excluded forms, exclusions.md row 56, SEMANTICS "`return`" and TRANSCRIPTION §3. The recursive definition of "an `if` that returns on the instance" is exactly when the front end goes on visiting into a dead block:
  - `visit_compound_statement` stops only after a `return` of its own list (`:508-517`), and `visit_Return` moves the insertion point to a new block no branch enters (`:544-554`);
  - a compile-time `if` visits its taken branch in the same block (`:995-996`);
  - a runtime `if` that holds a `return` joins its two branches' last blocks in the block after it (`visit_if_top_level`, `:893-929`), which is unreachable only when both returned;
  - `tl.static_range` copies are visited in sequence (`:1208-1216`).

  The sentence saying why is true: lr-15b's D1-D9 and evidence/dead-code-round17 E1-E7 re-ran identical. No in-scope function has a statement after an `if` with a `return`, or a `return` in a `tl.static_range` body (`work/call_scan.out`).
- **L90** ACCEPT — `visit_For` evaluates the keywords (`:1207`), and for Python's `range` uses only `iter_args[0..2]` (`:1237-1243`). The subset's `for` item ("one, two or three positional arguments and no keyword"), the excluded forms, exclusions.md row 57, SEMANTICS "The iterator's arguments" and TRANSCRIPTION's loop pitfall say so, and `tl.range` keeps its keywords. lr-15b's K1-K4 re-ran identical (8, 8, 4, 4 trips).
- **L91** ACCEPT — Capture is excluded consistently in every place that needs it:
  - `Faithful`'s scope 9, in its docstring and on its page ("What it says", "Scope", "Which launch", "What it does not cover", the checklist);
  - `TVA.Claim`'s module docstring and `Behaviour`, and `Launch.Valid`'s docstring;
  - SEMANTICS "One launch", its scope list and its outcome bullet;
  - exclusions.md row 58, protocol §6, both guides and TRANSCRIPTION's host pitfall.

  Excluding any capture in progress, not only on the launch's stream, narrows the scope, so it is conservative. The sentence saying why is true: lr-16b's G1-G3 re-ran identical (sentinel kept, `x + 1`, then `11`; 16 elements written).
- **L92** ACCEPT — All seven points are true of the pinned source:
  - (a) `visit_AnnAssign` writes `lscope`, not `local_defs` (`:704-710`), and only `local_defs` is merged (`:840-891`): lr-15a's A2/A3 re-ran identical;
  - (b) `Autotuner.run` passes `config.all_kwargs()` beside the launch's keywords (`autotuner.py:276-281, 365-377`; a key given by both is a host `TypeError`), and `@triton.jit` takes no `enable_fp_fusion`;
  - (c) the value is visited at `:698` and again at `:757` (A1: 16 atomics);
  - (d) `tl.static_range` sets the variable to each `constexpr` in the same scope (`:1208-1216`; S1 `2`, S2 `20`);
  - (e) `visit_FunctionDef` visits the defaults (`:634-649`) before it binds the arguments (`:660-663`; H1 rejected, H2 ran), and no in-scope helper has a default;
  - (f) TRANSCRIPTION's row and pitfall say "for how `x` is computed";
  - (g) the checklist is numbered 1-9, with the autotune clause.
- **L93** ACCEPT — REVIEWER_GUIDE §1 and PROVER_GUIDE §2 list all nine scope points, numbered as the page's checklist, with the autotune clause (the last launch, from the memory and with the arguments the last hook leaves), and point to the page. PROVER_GUIDE's point 7 says only "an H200 as the device", without the checklist's "with code compiled for it". Its `arch` and `TRITON_OVERRIDE_ARCH` parts are points 2 and 4, and a `triton.compile` target other than the device's is rejected on the host (evidence/compiled-options O5), so nothing a claim covers is lost.

## New issues

- **NEW** soundness — **A helper's value when its `return`s differ, or when a path reaches its end, is not what its body gives as Python reads it.** The subset admits, in a helper, a `return e` inside a runtime `if` with statements after the `if`. SEMANTICS "Helpers are inlined" says the call "means its body with the call's arguments", and "`return`" says such a `return` ends the inlined helper for the programs that take it.
  - **The front end.** The function gets one return type, the common type of every `return` it visits (`decide_return_type`, `code_generator.py:556-585`), and each returned value is converted to it (`cast_to`, `:587-604`):
    - equal compile-time values stay a compile-time value;
    - otherwise each Python number becomes a scalar tensor of its own type (`to_tensor_type`), and tensors of different types are rejected on the host;
    - a path that falls off the end returns `poison` of that type, or nothing at all when the type is a compile-time value, and the call then is that value on every path (`handle_returns`, `:606-626`).
  - **The device** (`work/helper_returns.py`, H200):
    - R1: `h(c)`, `if c > 0: return -7` then `return -9`, stored `h(c) // 2` as `-3` and `-4`, the truncating tensor `//`. Python numbers give `-4` and `-5`.
    - R5: with `return 0.1` / `return 0.2`, `xb * h(c)` on a `bfloat16` `xb` computed in `float32`, not in `bfloat16`.
    - R3: `if c > 0: return 5` with nothing after gave the compile-time `5` also for `c = -1` (a `tl.static_assert(h(c) == 5)` passed), where Python gives `None`.
    - R4: a helper returning a block on one path only gives `poison` on the other (it stored `x + 1` here).
    - R2: equal returns stayed compile-time (`-4`).

    No in-scope helper has two visited returns.
  - **Fix.** State the rule in "Helpers are inlined" or "`return`" (the common type; a Python number converted to its own type; a fall-through path's value: the compile-time value, or unspecified), or exclude it under L88: a helper whose visited `return`s are not all one compile-time value or all one type, and a helper that can reach its end without a `return` when another of its `return`s gives a value.
- **NEW** soundness — **Two `tl.constexpr` arguments of different types with the same `str` share one compiled kernel.** The subset admits a `tl.constexpr` argument that is a Python `int`, `float`, `bool`, `str` or `None`.
  - **The cache keys.** The in-process cache keys a launch by each constexpr's `repr` (`jit.py:592-613`). The disk cache keys it by `ASTSource.hash`, which writes each constexpr with `str` (`compiler.py:71-76`; `get_cache_key`, `triton/runtime/cache.py:319-321` of the installed 3.8.0, which `primitives/` does not pin). So `"1"` and `1`, `"None"` and `None`, `"True"` and `True`, `"2.5"` and `2.5` (and `"nan"` and a NaN, beside L98's row) have one disk entry.
  - **The device** (`work/constexpr_cache.py`; the kernel stores `10` if `MODE == "1"` and `20` otherwise):
    - X1: after a launch with `MODE = 1` in another process, a launch with `MODE = "1"` stored `20`. Its model stores `10`.
    - X2: the same happened within one process.
    - X3: in the other order, `MODE = 1` stored `10`.
    - X4, X5: the same with `None`/`"None"` and `2.5`/`"2.5"`.
    - X0: alone, `MODE = "1"` stored `10`.

    So `Faithful` is false for such a launch: the device ran another instance's code. No in-scope tutorial passes such a string (tutorial 03's `ACTIVATION` is `"leaky_relu"` or `""`).
  - **Fix.** Exclude a `tl.constexpr` argument, or autotune `triton.Config` value, that is a `str` spelling the `str` of an `int`, `float`, `bool` or `None` (`"1"`, `"2.5"`, `"True"`, `"None"`, `"nan"`, `"inf"`), in SEMANTICS' "Parameters and arguments", exclusions.md and `Faithful`'s scope. Add to the scope that the Triton cache the launch reads (in the process, and the cache directory) holds no kernel that a launch with such a string compiled from the kernel's source, since a non-`str` launch then runs it (X3). Pin `triton/runtime/cache.py`.
- **NEW** gap — **`Faithful`'s scope does not say that Triton itself is used unmodified.** The scope names knobs, hooks, options and heuristics, but not host or module code that assigns to Triton's objects, which changes what is compiled with none of those. In `work/host_modifies_triton.py` (H200):
  - U1: the kernel's `JITFunction._unsafe_update_src` (`jit.py:561-569`, a method of every kernel) replaced `x + 1` by `x + 2`, and the launch stored `x + 2`;
  - U2: `tl.arange = fake` at module level made the kernel's `tl.arange(0, 64)` a helper's `tl.zeros`, and 63 of 64 elements kept their sentinel;
  - U0, the control, stored `x + 1`.

  The subset's "Names" item forbids only binding the module-level names, and `tl.arange = …` binds none. Under the user's rule (L36, 2026-10-28) such a hole needs a stated limit and the reviewer's check. Fix: a scope point, or a sentence in scope 3, saying that neither the host nor the module assigns to an attribute of Triton's modules, classes or the kernel's `JITFunction` (`_unsafe_update_src`, `kernel.fn`, `tl.<name> = …`), with the item in the checklist and REVIEWER_GUIDE §1's host list.
- **NEW** wording — SEMANTICS "`tl.cdiv`", second bullet: "with a tensor operand …: `div - 1` and the addition wrap". When only `x` is a tensor and `div` a compile-time value, `div - 1` is Python's, computed at compile time, and only the addition wraps. That is the case of every in-scope call (`tl.cdiv(M, BLOCK_SIZE_M)`), and a `div - 1` outside the dtype's range is rejected on the host ("Constants"). Fix: "`div - 1` is Python's when `div` is a compile-time value, and a wrapping tensor `-` otherwise".

## Re-ran

- **Re-ran in full**, on GPU 0 (NVIDIA H200 NVL, Triton 3.8.0, torch 2.11.0+cu130, 2026-10-28), through `work/rerun_all.sh`. Each output was identical to the recorded one apart from the header line (`work/rerun-*.out`):
  - evidence/hint-unread-round17 (N0-N10);
  - evidence/dead-code-round17 (E1-E7);
  - evidence/cdiv-round17 (C1-C6, with all `2^32` inputs for each of the eight divisors);
  - evidence/subset-calls-round17 (no GPU);
  - lr-15a's `docs/library-review/round-15/lr-15a/work/probes.py` (B0-B5, A1-A3, K1-K2);
  - lr-15b's `docs/library-review/round-15/lr-15b/work/fe_edges.py` (K1-K4, A1-A3, S1-S2, H1-H2, W1, R1) and `dead_after_return.py` (D1-D9);
  - lr-16b's `docs/library-review/round-16/lr-16b/work/graph_capture.py` (G1-G3).
- **New tests**, each case in its own process with its own Triton cache:
  - `work/call_scan.py` (`.out`): every call shape of the 23 in-scope functions, their `for` iterators and `return`s, and statements after an `if` that holds a `return`;
  - `work/helper_returns.py` (R1-R5);
  - `work/constexpr_cache.py` (X0-X5);
  - `work/host_modifies_triton.py` (U0-U2).
- **Read against `primitives/`:**
  - `code-generator.py`: `visit_compound_statement`, `visit_Return`, `decide_return_type`, `cast_to`, `handle_returns`, `visit_FunctionDef`, `visit_AnnAssign`, `visit_Assign`, `visit_then_else_blocks`, `visit_if_top_level`, `visit_If`, `visit_For`, `visit_Call`, `call_JitFunction` and the static executors;
  - `AxisInfo.cpp` (the visitors used by the list, and 1220-1345);
  - `ir.cc` 400-430, `Combine.cpp`, `standard.py` (every `@jit` function; none has code after an `if` that returns), `autotuner.py` 150-175, 260-285 and 355-380, `jit.py` (`compute_cache_key`, `_pack_args`, `_do_compile`, `_unsafe_update_src`), and `compiler.py` 60-80 and 225-300;
  - in the installed Triton 3.8.0, `triton/runtime/cache.py` `get_cache_key`.
- **The coordinator's release check of the merged tree:** I read its output (11 of 11 steps pass) and did not run it.

## Not checked

- The rewrites that `primitives/` does not pin: the generated `TritonCombine.inc`, the other dialects' canonicalization patterns and the TTIR-to-TTGIR conversion. I checked that the `min` bound of L58 holds for any rewrite that drops a mark, or keeps it on the operation that defines the same value. I did not check that none moves a hint attribute onto an operation defining another value.
- Graph capture with in-kernel descriptors, autotuning or PDL (excluded under L91).
- `kernels/vec_add_f32` and `tvr check` (the coordinator's note), and the release check (above).
- Outside this batch, rounds 18 and 19 beyond the text they changed in my area: I read round 18's sentence in the subset's "Names" item and round 19's NaN text in SEMANTICS "Constants" and "Scalar arguments", and did not re-run evidence/literals or evidence/tool-round18.
