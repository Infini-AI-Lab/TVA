# Library batch, round 19: author's notes

Issues: L98, L99, L100 ([ISSUES.md](../../ISSUES.md)), 2026-10-28: the round-12 reviewers'
follow-ups on TVA's float layer. Built in a separate tree, beside rounds 17 (semantics) and 18
(tools), on rounds 12 to 16 (round 14, in review, included). Every fix is final (coordinator,
2026-10-28: the cycle stops after round 20): no item is left "for later".

**Build.** `lake build TVA` is green. Every new theorem (`Format.toRat_of_isInf`,
`maxVal_lt_inf`, `cvt_rz_sat`, `sqrt_rint_err`, `one_le_maxVal`, `sqrtMag_rn_spec`,
`sqrt_rn_err`, `FloatV.sqrtRn_err`, `IntV.toInt_cast`, `toInt_cast_unsigned`) depends only on
`propext`, `Classical.choice` and `Quot.sound` (`#print axioms` on each). A scratch file (not
shipped) used them as a kernel proof would: `FloatV.sqrtRn_err` at `float32` with
`binary32_u`/`binary32_eta`, `toRat_of_isInf` for `+∞` in `float32` (`2^128`), the NaN constants
by `decide +kernel`, and the two `tl.atomic_add` recipes of L100 (below) with `safe_atomic_buf`
and `SafeA.atomic`: it checks without errors. `tvr library-check` passes (hygiene, and
`leanchecker TVA` re-checks every declaration); `tvr lint library 19 author` prints `regular` and
`tvr audit` `independent`. No release check this round (coordinator).

**Kernel.** `python3 -I tools/tvr.py check vec_add_f32` in this tree: PASS, digest
`ee46b4d8bdd325fc` (was `401e1fa92db421d7`). Every item's digest changed (all five items
re-open): its cones reach `lean/TVA/Lang/Scalar.lean` and `lean/TVA/Math/Defs.lean`, whose
docstrings L99 asks to correct, and `docs/SEMANTICS.md` and `docs/exclusions.md` (L98), which are
hashed whole; `REVIEW.md` names `TVA.Math.Float.Format.nanMag` and `Format.toRat` (docstrings
only) and those files. No fix as good leaves them alone: the texts L99 names are in those files.
The run rewrote `kernels/vec_add_f32/` (`gate.json`, `claim.md`, `REVIEW.md`, a snapshot) and
`kernels/README.md` in this tree; the coordinator does not merge them.

**Files of other rounds** (round 17's), the smallest changes, for L98: `docs/SEMANTICS.md`,
"Constants" (one sentence after "Each of them gives `+0` for a zero": the NaN patterns, and the
excluded `tl.constexpr` NaN) and "Scalar arguments" (the unannotated `float` and the `tl.float32`,
`tl.float16`, `tl.bfloat16` cases: `FloatV.hostF32` instead of `FloatV.cvt`, and a NaN's
pattern); `docs/exclusions.md` (one row at the end: a `tl.constexpr` argument holding a NaN).

**Not changed, as recorded:** round 12's notes, which L99 also names ("meaningless", "the 7 that
differ are the non-saturating variants"): they are a recorded record, and this round's texts
(the layer's docstrings, evidence/float-layer's README) replace them. evidence/float-layer's
conclusion 4 still says "The library states neither yet (L64)", which round 14's `CvtSat` facts
made stale for the saturation; it is outside this batch (coordinator: nothing else changes), so
I name it here for the coordinator.

## L98: a Python NaN constant had the canonical NaN, not the pattern Triton builds

- **What changed.**
  - `lean/TVA/Lang/Scalar.lean`: three new definitions and the NaN case of the constants.
    `FloatV.qNaN s p` (the quiet NaN with sign `s` and payload `p`); `FloatV.nanNarrow` (a NaN
    converted on the host into a narrower `.ieee` format: sign kept, quiet bit set, the top
    fraction bits kept); `FloatV.hostF32` (C's `(float)` of a double: `cvt` for a number,
    `nanNarrow` for a NaN). `litF32 c` is `hostF32 c` (was `cvt c`, which gives the canonical
    `0x7fffffff`); `litF16` converts a NaN `float32` constant by `nanNarrow` (was `cvt`); `litBF16`
    gives `qNaN sign 0`, `0x7fc0`/`0xffc0` (was `nan`, `0x7fff`); `litF64` unchanged (the double,
    payload and sign included; docstring says so). So `float('nan')` is `0x7fc00000`, `0x7e00`,
    `0x7fc0`, and `-float('nan')` `0xffc00000`, `0xfe00`, `0xffc0`. The section docstring
    "Python number constants" states the NaN case, the builders' sources, and the exclusion below.
  - `docs/SEMANTICS.md` "Constants": the NaN patterns, both signs (with Python's
    `float('inf') - float('inf')`, the negative `0xfff8000000000000`), and that a `tl.constexpr`
    argument holding a NaN is excluded. "Scalar arguments": a runtime `float` (and `tl.float32`) is
    `FloatV.hostF32` of the double (the text said `FloatV.cvt`, wrong for a NaN in the same way);
    `tl.bfloat16` is the top 16 bits of `FloatV.hostF32`; a `tl.float16` NaN is `0x7e00`/`0xfe00`.
  - `docs/exclusions.md`: a `tl.constexpr` argument (or autotune `triton.Config` value) holding a
    float NaN is out of scope. Triton's cache keys write every NaN as `nan`
    (`primitives/triton-3.8.0-jit.py:611`, `primitives/triton-3.8.0-compiler.py:73-75`), so the
    launch may run the kernel compiled first, in the process or in the disk cache, for another NaN:
    after `C = float('nan')`, a launch with `C = -float('nan')` stored `0x7fc00000`. No pinned
    tutorial passes one. A NaN written in the source (`float('nan')`, `-float('nan')`) keys the
    cache by its source text, so it is covered with its exact pattern (L88: a form outside the
    subset closes by excluding it).
  - `evidence/literals`: `nan.py`, `nan.out.txt` and the README section "NaN constants";
    `run.py` now compares `tl.full` and `x * c` bit for bit (`other=`, a conversion, still up to
    NaN payload), and its docstring and the README say that its list has no NaN or infinity (a
    `repr` literal cannot write them), which `nan.py` covers.
  - `docs/LIBRARY.md` (`Scalar` row).
- **Why.** lr-12a: the constants are exact functions, not up to NaN payload, so a model storing
  `tl.full(…, float('nan'), tl.float32)` claimed `0x7fffffff` where the device writes
  `0x7fc00000`, and `-float('nan')` lost its sign. Of the two fixes the issue offers, the
  device's patterns need no follow-up: SEMANTICS's "takes the value Triton builds for it" is then
  true as written, and `Operand.constIn?` keeps its functional form. The device evidence found one
  route where no pattern is fixed, a `tl.constexpr` NaN (the cache), which the exclusion closes.
  The same defect sat in SEMANTICS's scalar arguments (`FloatV.cvt` of a NaN double), fixed with
  `hostF32`, which the evidence also covers.
- **Evidence.** [evidence/literals](../../../../evidence/literals/README.md), "NaN constants"
  (`nan.py`, `nan.out.txt`; Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28), every comparison bit for
  bit against the library's functions evaluated by Lean: N1, 14 NaN and infinity expressions
  written in the source, in `tl.full` and as a value of `tl.where`, in the four dtypes: 0
  differences (and `other=` up to payload: 0); N2, 306 NaN doubles as `tl.constexpr` arguments
  (quiet and signalling, every single payload bit, all-ones and random payloads, both signs), one
  process and empty cache per batch of 16: 0 of 306 differ in each dtype; N3, the cache
  collision above, both orders; N4, runtime `float` arguments: unannotated and `tl.float32` are
  `hostF32` on all 8 NaNs, `tl.float64` the double, `tl.bfloat16` the top half of `hostF32`.
  `run.py`, with its bit-for-bit comparison, re-run on this round's library: output identical to
  the recorded `out.txt` (9364 literals, 0 differences in each dtype).
- **Constants whose meaning changed.** `TVA.FloatV.litF32`, `TVA.FloatV.litF16`,
  `TVA.FloatV.litBF16` (their value on NaN doubles; on every other double unchanged), and through
  them `TVA.Operand.constIn?` (`f32`, `f16`, `bf16` of a NaN) and every constant that evaluates
  them. New: `TVA.FloatV.qNaN`, `TVA.FloatV.nanNarrow`, `TVA.FloatV.hostF32`. Docstring only:
  `TVA.FloatV.litF64`. Statements outside Lean: SEMANTICS "Constants" and "Scalar arguments", and
  the new exclusion.

## L99: the float layer's wording, and its evidence's README and build script

- **What changed.** One change per point of the issue:
  - **`toRat` of `±∞`.** `TVA.Math.Encoding`'s module docstring, `Format.toRat`
    (`lean/TVA/Math/Defs.lean`) and `FloatV.toRat` (`lean/TVA/Lang/Scalar.lean`) now say: of a
    finite pattern, its number; of `±∞`, `±2^(emax+1)` (`emax` the bias), beyond `±maxVal`, which
    orders the infinities below and above every number (`maxNum`, `minNum`, `cmp`, `orderKey`,
    `tl.maximum`, the comparisons, the sort keys rely on it); only a NaN's value is meaningless;
    every error bound is about finite values (it concludes that its result is finite, or, as
    `RoundsTo.err`, assumes it). New theorems behind the text: `Format.toRat_of_isInf` (the value
    of `±∞` is `±2^(bias+1)`) and `Format.maxVal_lt_inf` (`maxVal < 2^(bias+1)`), in `Encoding`.
  - **`FloatV.nan`** is "the library's canonical NaN", with what the device does: every NaN
    result of `float32`, `float16` and `bfloat16` arithmetic and of the conversions into fp8 and
    between those three formats except `bfloat16 → float32`; not `-x`/`tl.abs` (payload kept),
    `float64` arithmetic (`0xfff8000000000000` for an invalid operation, a NaN operand's quieted
    payload), conversions into `float64` (payload) and out of it (sign), nor the constants (L98).
    "canonical NaN" became "the library's canonical NaN" in `FloatV.maxNaN`, `cvt`, `cvtSat`,
    `Format.nanMag` and `TVA.Math.Arith`'s module docstring.
  - **`u ≤ 1/16`** (`TVA.Lemmas.FusedErr`): "at least 3 fraction bits: `float16`, `bfloat16`,
    `float32`, `float64` and `float8e4nv` (`u = 1/16`), not `float8e5` (`u = 1/8`) or `e2m1`".
    **`FloatV.ofRat`**: the overflow by `Specials`: `±∞` (`.ieee`), the NaN (`.nanOnly`,
    `float8e4nv`), `±maxVal` (`.finite`, `e2m1`).
  - **`TVA.Lemmas.FloatMax`'s module docstring** names what the module provides: `le_iff`,
    `lt_iff`, `eq_iff`, `le_refl'`, `le_trans'`, `ext_pat`, `max_mem`, `min_mem`, `toRat_max`,
    `toRat_min`, `isNaN_of_mem`, `MaxRel.mem`, `MaxRel.le`, `MaxRel.ge`, `Tree.eval_maxRel`,
    `negInf_toRat_le`, `max_negInf_left`, `MaxRel.negInf_left`, `sub_negInf_left`,
    `sub_negInf_right`, `sub_negInf_negInf` (no `le_max_left`/`le_max_right`, which were Mathlib's).
  - **`tl.sqrt_rn`'s error bound, proved** (the issue's first option): `Format.sqrt_rint_err`
    (the rounded integer square root `sqrtMag` computes is within `1/2` of `√t`),
    `Format.one_le_maxVal`, **`Format.sqrtMag_rn_spec`** (`lean/TVA/Math/Rounding.lean`: for
    `0 < a ≤ maxVal`, `sqrtMag .rn a` is a finite magnitude whose value is within `√a·u + η` of the
    real `√a`; it does not overflow), `Format.sqrt_rn_err` (`lean/TVA/Math/Arith.lean`: the
    pattern operation, `−0` and `+0` included), and **`FloatV.sqrtRn_err`**
    (`lean/TVA/Lemmas/FloatErr.lean`: `tl.sqrt_rn` of a finite `x ≥ 0` is finite, nonnegative,
    within `√x·u + η` of `√x`, in `ℝ`). The characterisation is by the real `√a`, since `√a` is
    not rational in general; it is the statement "rounded to nearest" means. `docs/rules/num.md`
    ("Exact operations need no rule") names it; the module docstrings of `Rounding`, `Arith` and
    `FloatErr` list it.
  - **evidence/float-layer's README**: "The 7 that differ are variants … the device does not
    perform: six non-saturating conversions to nearest (`"rtne"`) and one saturating conversion
    toward zero (`"rtz"` with `sat := true`, `float32` into `float8e5`)", in the result and the
    table; and the argument for "exactly": toward zero saturation changes only `±∞`
    (`Format.cvt_rz_sat`, new, proved: on every other pattern `cvt … .rz true = cvt … .rz false`),
    `.rz false` agrees with the device on every `float32` input (`A.cvt.f32.e5m2.rz.sat0`,
    exhaustive), so `sat := true` differs exactly on the `±∞` inputs; `results.out` lists 24 of
    them (the first six of the four chunks it dumps), every one `7f800000` or `ff800000`.
  - **`build.sh`** finds the toolchain that `lean/lean-toolchain` names through `lake` run in
    `lean/` (elan's proxy runs that toolchain; `lake env` puts its `bin` first on `PATH`) and runs
    `lean` and `leanc` from that `bin`; it needs only `lake` (on `PATH`, elan's, or `LAKE`). The
    README's "Re-run" says so, and that `run.py` appends to `results.out` in the current folder.
  - **New part F of evidence/float-layer** (`nan_patterns.py`, `nan_patterns.out`, README
    section), the device evidence for `FloatV.nan`'s docstring.
  - `docs/LIBRARY.md`: the `Encoding`, `Rounding`, `Arith`, `FloatErr` and `FloatMax` rows.
- **Why.** lr-12a, lr-12b: texts that said less or more than the definitions and the device
  (the infinities' values are used; `float64` NaNs differ; `u = 1/8` exists; `ofRat`'s overflow
  depends on the format; FloatMax named Mathlib's lemmas; `tl.sqrt_rn` had no bound; one of the 7
  differing tests saturates; `build.sh` ignored the `BIN` it computed and failed without `lean` on
  `PATH`). Round 12's notes keep their wording, being recorded.
- **Evidence.** Proofs for the new theorems (standard axioms). evidence/float-layer part F
  (2026-10-28, GPU 0): the NaN patterns above, from runtime operands. `build.sh` re-run with
  `PATH=/usr/bin:/bin` (where the old script stopped at "lean: command not found"): it built the
  driver, and its `Defs.c` is identical (`cmp`) to the one compiled from `Defs.lean` before this
  round's docstring changes, so the native code of results.out is still the layer's; with it
  `A.cvt.f16.e4m3.sat1` (agree) and `A.cvt.f32.e5m2.rz.sat1.sample` (differ in 64 chunks, as
  recorded) re-ran with their recorded results.
- **Constants whose meaning changed.** None: docstrings only for `TVA.FloatV.toRat`,
  `TVA.FloatV.nan`, `TVA.FloatV.ofRat`, `TVA.FloatV.maxNaN`, `TVA.FloatV.cvt`,
  `TVA.FloatV.cvtSat`, `TVA.Math.Float.Format.toRat`, `TVA.Math.Float.Format.nanMag`, and the
  module docstrings of `TVA.Math.Encoding`, `Arith`, `Rounding`, `TVA.Lemmas.FusedErr`, `FloatMax`,
  `FloatErr`. New theorems: `TVA.Math.Float.Format.toRat_of_isInf`, `Format.maxVal_lt_inf`,
  `Format.cvt_rz_sat`, `Format.sqrt_rint_err`, `Format.one_le_maxVal`, `Format.sqrtMag_rn_spec`,
  `Format.sqrt_rn_err`, `TVA.FloatV.sqrtRn_err`. `lean/TVA/Math/Rounding.lean` now imports
  `Mathlib.Analysis.Real.Sqrt`. Through the files hashed whole (`Defs.lean`, `Scalar.lean`,
  `docs/rules/num.md`), every item whose cone reaches them has a new digest.

## L100: names and docstrings in the proof kit that did not match their statements

- **What changed.** Each text now says what the code states (each name checked with `#check`):
  - `lean/TVA/Lemmas/Access.lean`, module docstring: the footprint lemma is
    `Buf.Addrs.accBytes` (not `accBytes_of`); an access pattern proves `Buf.Addrs` from its lanes
    (`Buf.addrs_of_get`, `addrs_of_get_all`) or by its own lemma, the 1-D block's `addrs_block`
    (`TVA.Lemmas.Block`), `BlockPtr.addrs_buf` or `TensorDesc.addrs_buf` (any rank); a 2-D tile of
    pointer arithmetic has no lemma of its own and goes through `Buf.addrs_of_get` (no "row of a
    matrix, 2-D tile" promised in `Block`).
  - `lean/TVA/Lemmas/Shared.lean`, module docstring: the rule is `SafeA.atomic1` (not
    `safeA_atomic1`), with `SafeA.atomic` for tensors; the determined replays are those the
    library proves (integer `atomic_add`: `replay_add`, `replay_add_any`; integer `max`/`min`:
    `replay_max`, `replay_min`; float `atomic_add` bounded by `replay_err_shared`), and the bitwise
    atomics "commute too, but have no lemma yet".
  - `tl.atomic_add` is `.env fun e => tl.atomicWith …`: `SafeA.atomic1` (`Shared`), `SafeA.atomic`
    and the module docstring of `SharedAcc`, and `safe_atomic_buf` (`Access`) now say that they are
    about `tl.atomicWith` (every other `tl.*` atomic), and that for `tl.atomic_add` a proof unfolds
    it and applies the constructor `Safe.env`/`SafeA.env` first, then the rule with
    `upd := fun i => tl.addUpd E d (vals.get i)`. Both recipes were checked in Lean (scratch file).
  - `lean/TVA/Lemmas/Ops.lean`: `safe_upToNaN` lists the operations that are `tl.upToNaN`
    (`add`, `sub`, `mul`, `fma`, `div_rn`, `sqrt_rn`, `neg`, `abs`) and says `tl.cvt`'s rule is
    `safe_cvt`; `safe_dot_add` names its one hypothesis, `hex` (the `tl.dot` with accumulator `c`
    has a result), and says why neither the separate product nor the addition needs one;
    `safe_sum_bcast_mul` says the form is chosen once for the whole tensor (each element of the
    tree form with its own tree), as the statement and `tl.sum_bcast_mul` do.
  - `lean/TVA/Lemmas/Int.lean`: "a literal (`IntV.ofInt`, `IntV.ofNat`) in range keeps its value",
    "comparisons of signed integers … in any range" (unsigned ones through `IntV.lt_toInt`), and
    the casts the docstring promised are now proved: `IntV.toInt_cast` (a widening cast of a signed
    integer, sign extension) and `IntV.toInt_cast_unsigned` (zero extension).
  - `lean/TVA/Lemmas/Claim.lean`: both lemmas take `hdebug` and `hvalid` (what
    `Compiler.Faithful` needs of every covered launch), `meets_independent` also `hRW`.
  - `docs/LIBRARY.md`: the `Independent` row states `independent`'s conclusion (every outcome
    normal, the final memory agreeing on each program's footprint with a memory its `Q` holds of,
    the initial memory outside every write footprint: determined only as far as the `Q`s say); the
    `Int` row.
- **Why.** lr-12b: names that do not exist or resolve elsewhere, and promises the statements do
  not keep. The two cast lemmas make "casts keep their value" true rather than dropping it (index
  arithmetic widens `int32` to `int64`).
- **Evidence.** None needed (texts, and two lemmas proved with the standard axioms).
- **Constants whose meaning changed.** None: docstrings of `TVA.safe_atomic_buf`,
  `TVA.SafeA.atomic1`, `TVA.SafeA.atomic`, `TVA.safe_upToNaN`, `TVA.safe_dot_add`,
  `TVA.safe_sum_bcast_mul`, and the module docstrings of `TVA.Lemmas.Access`, `Shared`,
  `SharedAcc`, `Int`, `Claim`. New: `TVA.IntV.toInt_cast`, `TVA.IntV.toInt_cast_unsigned`.

