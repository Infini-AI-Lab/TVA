# Library review, round 2, by lr-2b

## Verdicts

- **L3** ACCEPT — A rejection is now defined by the error the host gets before any program runs, in
  `TVA.Claim` (module, `Behaviour`, `Outcome.ok`), `Rules.Compiler.Faithful`, the rule page, SEMANTICS.md
  "Launches", DESIGN.md and protocol §2/§6. The three stale texts of round 1 are gone (`Claim.Total` no
  longer says every launch has an outcome; `Launch.Valid` and DESIGN.md no longer ask for compile limits).
  The re-run of `evidence/launch-outcomes` is identical: A1–A8 raise on the host, with memory unchanged and
  the device usable; B1, B2 return without an error and run nothing, which is the semantics' `ok init`. In
  the pinned `driver.c` (`launchKernel`, `_launch`) and in `jit.py` `run`, the only path that returns
  without an error and without launching is the `gridX * gridY * gridZ > 0` guard, now outside
  `Launch.Valid` (L9).
- **L6** ACCEPT — `compile-check` now needs a pure condition, defined in TRANSCRIPTION.md §10 (no call of
  any kind), and an impure condition is transcribed. Where the front end evaluates the assertion is
  stated. The re-run of `evidence/debug-calls` is identical (P1, E1, E2, E8, V1–V4). I read
  `execute_static_assert`, `visit_Compare` (`is`, `is not`), `visit_BoolOp` and `visit_JoinedStr`
  (constexpr only), and the `semantic.py` functions behind each pure operator: `add`, `sub` and `mul`
  wrap or are IEEE, and comparisons, `expand_dims` and the bitwise operators have no effect. The rule
  excludes `/ // % << >>` on runtime values. I found no pure expression that has an effect or can be
  undefined.
- **L7** REOPEN — The import, lake and name checks hold. Re-run on copies of the tree, the author's 53
  probes and round 1's 21 give their recorded results (`work/rerun_independence_audit.out`,
  `work/rerun_round1_l7_probe.out`). Lean 4.35 rejects `import A B`, the one header form the parser
  would misread. But the path check still misses home paths written in forms a shell or a browser
  resolves: `$HOME/fpv/<dir>/…`, `${HOME}/fpv/<dir>/…`, `file:///home/<user>/fpv/<dir>/…` (bare or in
  `<…>`) and `~<user>/fpv/<dir>/…` all pass `tvr audit`, while `/home/<user>/fpv/<dir>/…` is caught
  (`work/l7_more_probes.py`: 5 of 7 probes not as the rule asks). Protocol §2 says a home path other than
  TVA's fails. Fix: in `foreign_refs`, treat `$HOME`, `${HOME}` and `~<user>` as home prefixes, and match
  `/home/` and other absolute paths after any character (`file://` included). Add these probes to
  `evidence/independence-audit`.
- **L8** ACCEPT — `Block.Fits p B` is now `p * B + B ≤ 2^31`: exactly "the largest offset `p*B + B − 1`
  fits `int32`". `Block.offs` is `pid * B + tl.arange(0, B)` in wrapping `int32`, as Triton computes it.
  Checked in Lean (`work/BatchChecks.lean`):
  - `Fits (2^21 − 1) 1024` holds at the boundary, and `Fits 2^21 1024` fails;
  - `fits_of_grid_pow2` gives the last program of `cdiv(2^31 − 1, 1024)`, which `fits_of_grid` cannot;
  - `lane_toInt` gives that program's lane 1023 as `2^31 − 1`;
  - the new and changed lemmas use only the three standard axioms.

  `Fits` and the lemmas that mention it are listed as changed.
- **L9** ACCEPT — `Launch.Valid` adds `L.size < 2^31`. The pinned `driver.c` has the installed file's
  sha256. `launchKernel` parses each grid dimension as a C `int`, and `_launch` launches only if their
  `int` product is positive. With each dimension within CUDA's bounds and the product below `2^31`,
  nothing overflows: a valid launch with no zero dimension runs every program, and one with a zero
  dimension runs none, which is the semantics' `ok init`. The re-run of part C is identical. Lean
  (`work/BatchChecks.lean`):
  - `(2^30, 2, 1)`, `(2^16, 2^15, 2)` and `(2^16 + 1, 2^15, 1)` are not valid;
  - `(2^31 − 1, 1, 1)` and `(65535, 2^15, 1)` are valid;
  - `(0, 70, 1)` is valid with size `0`;
  - `valid_grid1` applies.

  SEMANTICS.md, TRANSCRIPTION §9, the rule page, PROVER_GUIDE and protocol §6 say the same.
- **L10** ACCEPT — `no-op:print` now names `tl.static_print`, `tl.device_print`, `print`,
  `tl.device_assert` and `assert`, and needs every argument pure; otherwise the argument operations are
  model lines. `dead` also needs a pure expression. The re-runs of P1, P2, E3–E7 and E9 are identical.
  The source agrees: `visit_Assert` evaluates `test` and `msg`, `builtin_namespace` maps `print` to
  `device_print`, and `semantic.device_assert` returns before emitting anything without `debug`.
- **L11** ACCEPT — `rules/num.md` now tells kernel reviewers what they check. That is the library's fact
  for the model's operation, format and configuration, with the library's arguments, and a field whose
  constants the kernel chose is `RULE_MISCITED`. The device questions are in a section addressed to
  library review, and no sentence asks a kernel reviewer for PTX or a test.

## New issues

- **NEW** soundness — One program is not sequential for global memory on Triton 3.8.0 and the H200. So
  `Compiler.Faithful` is false for well-behaved models whose program reads or writes bytes it accessed
  in an earlier step. The library says each step sees the memory left by the program's earlier steps:
  - SEMANTICS.md "Programs", "No barrier semantics" ("a program that writes and then reads back its own
    data sees the write") and "Granularity";
  - the `TVA.Lang.Prog` and `TVA.Sem.Launch` docstrings, and `InterRace`, which never relates two events
    of one program;
  - review-questions S1.

  TRANSCRIPTION §10 also drops `tl.debug_barrier()` and `tl.range`'s `num_stages` as no-ops. On the
  device:
  - (a) A program that stores a block and then loads it back reversed reads the old contents in 24–40%
    of elements (B = 1024–8192). An in-place reversed copy differs in 0.3–3.6%. With a
    `tl.debug_barrier()` between the store and the load, or reading back at the same offsets, nothing
    differs (`work/seq_readback.py`).
  - (b) In a loop where iteration `i` loads the row that iteration `i − 1` stored, at the same offsets,
    63 of 65 rows read the initial zeros when Triton pipelines the loop. That happens with
    `tl.range(…, num_stages=3)`, and with a plain `range` whose body has a `tl.dot`, under the launch's
    default `num_stages=3` (and 4). The result is sequential with `num_stages=1`, without the dot, or with
    a `tl.debug_barrier()` after the store (`work/loop_carried_mem.py`).

  Each model is one program with no `InterRace`, so it is well behaved, and the device's result is not
  one of its outcomes. Fix (it needs the user's decision, since S1 settled the barrier the other way):
  - make two conflicting events of one program, in different steps, a race (no meaning), unless the
    library states with evidence what orders them, for example a `tl.debug_barrier()` kept as a model
    step;
  - withdraw `no-op:range-kw` for `num_stages` (and check `warp_specialize`);
  - correct "Granularity".
- **NEW** gap — Without `return_indices`, `tl.max` and `tl.min` widen integers narrower than 32 bits to
  `int32` (`triton-3.8.0-standard.py:190, 249`). `tl.max` of an `int8` tensor has dtype `int32`, and
  `tl.max(x) + 100` is `200` where `int8` arithmetic gives `-56` (`work/misc_probe.py` M1). SEMANTICS.md
  lists the promotions as "`tl.max`/`tl.min` reductions widen floats narrower than 32 bits to fp32;
  `tl.sum` widens only small integers", and the exact integer reductions (`Tensor.foldAll`, `foldRows`,
  `foldCols`) say nothing about the result type. So a reviewer who trusts the list accepts a model that
  keeps `int8`. Fix: state the integer widening in SEMANTICS.md and next to the integer reductions. It is to `int32` for any signedness, and
  `uint8`/`uint16` are zero-extended.
- **NEW** gap — The pinned tutorials use these, and the library has no definition or rule for them:
  - `tl.permute`, and `trans` of rank 3 or more (tutorials 06:90, 09:437, 10:226; `Tensor.trans` is rank
    2 only);
  - `tl.extra.cuda.gdc_wait` and `gdc_launch_dependents` with `launch_pdl=True` (tutorial 11). `Launch`
    has no `launch_pdl`, and `Launch.Valid` does not exclude it. By the tutorial's own comment, a
    program that does not wait may read before the previous kernel's writes are visible, so the claim's
    `init` is not the memory it reads;
  - `tl.argmax`, which SEMANTICS.md "Shape changes" and TRANSCRIPTION §5 name, though nothing defines it.

  Fix: add `permute` for any rank. Define `launch_pdl` and `gdc_*`, or require `launch_pdl = false` in
  `Launch.Valid` and list them in exclusions.md. Define the index reductions, or drop the references to
  them.
- **NEW** gap — The environment changes defaults the model fixes. `TRITON_F32_DEFAULT` sets `tl.dot`'s
  `input_precision` when the source omits it (`triton-3.8.0-core.py:2387`), and `TRITON_DEFAULT_FP_FUSION`
  sets `enable_fp_fusion` (`triton-3.8.0-nvidia-compiler.py:212`). The library names only `TRITON_DEBUG`,
  and `DotCfg.precision` defaults to `tf32`. So for a source without `input_precision`, the model's
  `DotCfg`, and the `DotAcc` fact cited for it, may name a precision the device does not use. Fusion is
  harmless, because a model with fusion admits the IEEE result. Fix: say that an omitted
  `input_precision` is `TRITON_F32_DEFAULT` when that is set, and either record it in the launch or state
  that claims assume these variables unset.

## Re-ran

- `evidence/launch-outcomes/run.py` and `evidence/debug-calls/run.py` on GPU 0 (H200 NVL, Triton 3.8.0,
  torch 2.11.0+cu130, 2026-10-27). Their output equals the recorded `out.txt` apart from timings
  (`work/rerun_launch_outcomes.out`, `work/rerun_debug_calls.out`).
- `evidence/independence-audit/probes.py` (53 probes, 0 wrong, identical to `out.txt`) and round 1's
  `l7_audit_probe.py` (0 of 21 missed). `tvr audit` on the tree prints "independent".
- Added, each with its output beside it in `work/`:
  - `l7_more_probes.py`: home-path forms;
  - `BatchChecks.lean`: L8 and L9 boundary cases and axioms;
  - `seq_readback.py`: store and load back within one program;
  - `loop_carried_mem.py`: pipelined loops;
  - `misc_probe.py`: M1 integer `tl.max` dtype; M2 fusion across a widening, none found; M3 float
    `atomic_max`/`atomic_min` on signed zeros, infinities and NaNs, all 288 within `fBound`;
  - `fusion_math.py`: whether `f(x) + y` contracts into an `fma` for the `P.math` functions and
    `tl.sigmoid` in `float32`/`float64`. It does not: every result is the IEEE sum of `f(x)`.
- Read in `primitives/`:
  - `driver.c` (`_launch`, `launchKernel`);
  - `semantic.py` (promotions, arithmetic, shifts, `cast`, `load`, atomics, `where`, `histogram`,
    `device_assert`);
  - `standard.py` (`max`, `min`, `sum`, `sort`), `code-generator.py` (`visit_For`) and `core.py`
    (`_block_ptr`);
  - `Combine.cpp` (dot + add, broadcast-mul-reduce), `nvidia-ElementwiseOpToLLVM.cpp` (`exp`, `FpToFp`)
    and `llvm-ArithOps.cpp` (`truncf`/`extf` folds).

  These sources agree with `TVA.Lang` on everything except the new issues above.

## Not checked

- L7: Lean code that runs at elaboration time (`#eval`, `include_str`, `initialize`) and reads files
  outside TVA, and names spelled with other letters (a limit the evidence states).
- Which same-offset patterns stay sequential under every pipelining and layout choice; whether
  `tl.debug_barrier()` orders accesses in every pipelined loop; `warp_specialize`; descriptor (TMA)
  stores and loads of the same bytes within one program; two atomics of one program on one address.
- The issues not in this batch: hints (L1, L2), accuracy facts (L4), descriptor scratch (L5),
  transcription digests (L12). I ran nothing on the device for `tl.dot`, `tl.dot_scaled`, scans,
  `tl.sort`, `tl.histogram` or Philox.
- Reach on kernel items: the one kernel does not build against this batch (expected), and I did not run
  `tvr check`. I compared the notes' lists of changed constants with the definitions I read.
