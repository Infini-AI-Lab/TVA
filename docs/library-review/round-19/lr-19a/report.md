# Library review, round 19, by lr-19a

## Verdicts

- **L98** REOPEN — The new functions are the device's: `nan.py` re-ran with output identical to
  `nan.out.txt` (N1–N4: 14 source expressions, 306 payload NaNs, the cache collision, the runtime
  arguments), my own Lean evaluation (work/Names.lean, Names.out) gives `0x7fc00000`/`0xffc00000`,
  `0x7e00`/`0xfe00`/`0x7f00`, `0x7fc0`/`0xffc0` and `constIn?` the same, and SEMANTICS "Scalar
  arguments" matches the launcher's extractors (`triton-3.8.0-nvidia-driver.c:1130-1163`: `(float)` of
  the double, its top half for `bfloat16`, `PyFloat_Pack2` for `float16`) and N4. But the fix says
  that a NaN written in the source "is covered, with its exact pattern" (`docs/exclusions.md`, last
  row; SEMANTICS "Constants"), and a source NaN passed to a helper is not. The front end names each
  specialised helper by `mangle_fn` (`primitives/triton-3.8.0-code-generator.py:32-41, 1369-1371`),
  which writes a compile-time argument as its `repr` (`primitives/triton-3.8.0-core.py:196-201`),
  and every NaN's `repr` is `nan`: so `store_c(o, float('nan')); store_c(o + 1, -float('nan'))` makes
  one function, the first call's, and stores `7fc00000, 7fc00000` where `litF32` of the two doubles
  is `7fc00000, ffc00000`; the reverse order stores `ffc00000` twice, and so do the same calls through
  `A: tl.constexpr = float('nan')` and `B: tl.constexpr = float('inf') - float('inf')`, by keyword, in
  `bfloat16` (`7fc0` twice), and as a value of `tl.where` in the helper (work/helper_nan.py,
  helper_nan.out: H1–H6 differ; the controls, the same constants inline and `±∞` through the helper,
  agree). Helper calls with Python numbers are in the supported subset (SEMANTICS "Parameters and
  arguments", "Helpers are inlined"). Fix: exclude a float NaN passed to a helper's parameter (or
  state in "Helpers are inlined" that calls of one helper whose arguments have the same types and
  whose compile-time arguments have the same `repr` run the first call's body, so that a NaN
  parameter holds the first such call's NaN), and correct the exclusion row's "covered, with its
  exact pattern" and SEMANTICS "Constants" to say so.
- **L99** ACCEPT — Each point is fixed and true. `toRat` of `±∞`: the docstrings now say
  `±2^(emax+1)`, beyond `±maxVal`, and the new `Format.toRat_of_isInf` and `maxVal_lt_inf` state it
  (Names.out: `+∞` is `2^128` in `float32`, `2^16` in `float8e5`, above its `maxVal`; standard axioms
  only). `FloatV.nan`'s account of the device matches part F, which I re-ran: `nan_patterns.py`
  printed exactly `nan_patterns.out`. The `u ≤ 1/16` formats (`m ≥ 3`) and `ofRat`'s overflow by
  `Specials` are right, and every name of `FloatMax`'s docstring resolves to TVA's own declaration
  (Names.out, `#check`). `FloatV.sqrtRn_err` and `Format.sqrtMag_rn_spec` state what num.md says, with
  standard axioms, and `B.sqrt.f32` re-ran agree on every `float32` input, so the bound is about
  the device's `tl.sqrt_rn`. The README's "six `"rtne"` and one saturating `"rtz"`" and its
  argument hold: `Format.cvt_rz_sat` is proved, `A.cvt.f32.e5m2.rz.sat0` re-ran agree on every
  input, and `A.cvt.f32.e5m2.rz.sat1.sample` re-ran with the recorded 64 differing chunks, every
  listed input `7f800000` or `ff800000`. `build.sh` built the driver with `PATH=/usr/bin:/bin`.
  No meaning changed: `Format.nanMag` and `Format.toRat` keep their definitions (docstrings only;
  the driver built from the current `Defs.lean` agrees with the device on every test I re-ran).
- **L100** ACCEPT — Each corrected name resolves to the library's declaration (`Buf.Addrs.accBytes`,
  `Buf.addrs_of_get`, `addrs_of_get_all`, `SafeA.atomic1`, `SafeA.atomic`, `safe_cvt`, `Safe.env`,
  `SafeA.env`, `IntV.lt_toInt`; work/Names.lean). `tl.atomic_add` is `.env fun e => atomicWith …`
  (`lean/TVA/Lang/Ops.lean:256-259`), as the new docstrings say. `safe_upToNaN`'s list is exactly
  the operations defined as `upToNaN` (`Ops.lean:352-411`), with `tl.cvt` a `pick` (`:443`).
  `IntV.toInt_cast` and `toInt_cast_unsigned` state the widening casts, with standard axioms only.

## New issues

- **NEW** soundness — Triton's kernel caches conflate distinct `tl.constexpr` arguments other than
  NaNs, and the library states none of it (only the NaN case is excluded, L98). In the process,
  `compute_cache_key` looks the specialization up in a dict keyed by `("constexpr", value)` tuples
  (`primitives/triton-3.8.0-jit.py:409, 592-611`), and Python's `1 == 1.0 == True` and
  `0 == 0.0 == -0.0 == False` with equal hashes. So a launch with `C=1.0` after one with `C=1`
  ran the kernel compiled for `1`: `isinstance(C, float)` was false and `x * C` with
  `x = 2^24 + 1` (`int32`) stored `16777217`, where `x * 1.0` computes in `float32` and stores
  `16777216`; likewise `C=1` after `C=1.0`, `C=True` after `C=1.0`, and `0.0` after `0`, `0` after
  `-0.0` (work/cache_num.py, cache_num.out). On disk, `ASTSource.hash` writes each constant as
  `str(value)` (`primitives/triton-3.8.0-compiler.py:71-76`), so `1` and `"1"` (also `True`/`"True"`,
  `None`/`"None"`, `1.0`/`"1.0"`) share an entry: a process launching with `MODE="1"` on a cache
  folder where another launched `MODE=1` ran the `MODE=1` kernel (`if MODE == 1` taken), in both
  orders (work/cache_str.py, cache_str.out). The supported subset admits a `tl.constexpr` argument
  of any of these types (SEMANTICS "Parameters and arguments"), so a claim about the second launch
  is wrong. Fix: state in `Compiler.Faithful`'s scope (`docs/rules/compiler-faithful.md`, "Scope",
  and `docs/exclusions.md`) that the launch runs the kernel compiled for its own arguments only when
  no earlier launch of the kernel, in the process or on the same cache folder, had a
  `tl.constexpr` argument equal to this one under `==` or `str` but of another type (an `int`, a
  `float` or a `bool`; a string spelling a number, `True`, `False` or `None`), and exclude the rest;
  or exclude a `tl.constexpr` parameter given arguments of different Python types across launches,
  and a string argument that spells another admitted value.
- **NEW** soundness — Two calls of one helper whose string arguments differ only where one has `[`
  or `]` and the other `_` run one body, the first call's: `mangle_fn` replaces `[` and `]` by `_`
  in each compile-time argument's `repr` (`primitives/triton-3.8.0-code-generator.py:32-41`), so
  `pick(o, "x[y"); pick(o + 1, "x_y")`, with `pick` storing `1` when `MODE == "x_y"`, stored `2, 2`
  where Python's meaning is `2, 1`, and the reverse order `1, 1` (work/helper_str.py,
  helper_str.out). String arguments of helpers are in the supported subset, and SEMANTICS "Helpers
  are inlined" says a call means its body with the call's arguments. Fix: state the rule there
  (with L98's NaN case: calls whose compile-time arguments have the same `repr` after `mangle_fn`'s
  replacements share the first call's body) and exclude a helper's string argument holding `[` or
  `]`.
- **NEW** wording — `evidence/float-layer/README.md`, conclusion 4, ends "The library states neither
  yet (L64)", but since round 14 the `CvtSat…` facts state the saturating conversions into fp8
  (`docs/rules/num.md`, "Conversions into fp8"); only `float32` into `float8e5` with `"rtz"` is not
  stated (num.md, "What the library does not state"). Fix: "The library states the first (the
  `CvtSat…` facts), not the second." (The round-19 author named it for the coordinator; it is filed
  here so that it has an issue.)

## Re-ran

- evidence/literals `nan.py` in full (Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28): output identical
  to `nan.out.txt` (work/nan-rerun.out). evidence/literals `run.py` in full: output identical to
  `out.txt` (work/literals-rerun.out).
- evidence/float-layer: `build.sh` with `PATH=/usr/bin:/bin` (built), then `run.py` from my work
  folder (work/float-layer-rerun/results.out): in full every `2^16`/`2^8` test of part A
  (`float16`, `bfloat16` into every format, fp8 `sat0` and `sat1`; out of fp8), `C.neg`/`C.abs` on
  `float16` and `bfloat16`, and over all `2^32` inputs `B.sqrt.f32` and `A.cvt.f32.e5m2.rz.sat0`;
  the `2^26` samples `A.cvt.f32.e5m2.rz.sat1.sample` and `B.sqrt.f32.sample`. Every verdict and
  difference count is the recorded one (the sqrt sample, not recorded, agrees). Part F,
  `nan_patterns.py`: identical to `nan_patterns.out` (work/nan_patterns-rerun.out).
- Added: work/helper_nan.py (L98, NaN through a helper), work/helper_str.py (helper strings),
  work/cache_num.py and work/cache_str.py (the constexpr caches), work/Names.lean (the names L99 and
  L100 cite, the new theorems' axioms, the NaN constants and `constIn?` evaluated), and
  work/fp8_const.py, which confirms L64's open fp8-constant case on this tree (`tl.full([1],
  float('inf'), tl.float8e5)` and `tl.where` store `0x7c`, the run-time conversion and
  `FloatV.cvtSat .rn` `0x7b`; an inexact or out-of-range fp8 constant, and `±∞` or NaN in
  `float8e4nv`, are rejected on the host), not re-filed (L64, round 20).

## Not checked

- The other `2^32` tests of evidence/float-layer and their samples (the batch changed no definition
  they run; part A's `float32` `sat1` tests were re-run in round 12).
- evidence/num-math-f32, num-div-f32, num-div-f64, num-dot, num-dot-scaled, num-atomic-add and
  num-cvt8 not re-run: the batch changed none of their facts; I read rules/num.md, `TVA.Rules.Num`
  and `Rules.Num.Sm90` against them and the pinned sources.
- The two `tl.atomic_add` recipes of L100 in Lean (lr-19b's area): I checked only that the
  definitions and the constructors they name exist as the docstrings say.
- A Python other than the environment's 3.13.9: SEMANTICS' `tl.float16` NaN argument (`0x7e00`,
  payload dropped) is CPython's `PyFloat_Pack2` as that version implements it.
