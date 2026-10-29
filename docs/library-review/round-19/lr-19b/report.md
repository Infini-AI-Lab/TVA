# Library review, round 19, by lr-19b

## Verdicts

- **L98** ACCEPT — `FloatV.litF32`, `litF16` and `litBF16` now give the patterns Triton 3.8.0
  builds, and SEMANTICS "Constants" says so. Read against the builders: `scalar_constant` sends a
  NaN to `get_<dtype>` (`value == 0` is false; `primitives/triton-3.8.0-semantic.py:601-613`),
  `get_fp32`/`get_fp16`/`get_bf16` take pybind's C `(float)` of the double, `getF16FloatAttr`
  converts that `float` with APFloat, and `get_bf16` parses `std::to_string` (`nan`, `-nan`)
  (`primitives/triton-3.8.0-ir.cc:952-972`): that is `hostF32` (`nanNarrow`, `qNaN`), `nanNarrow`
  into `float16`, and `qNaN sign 0`. evidence/literals `nan.py` re-ran with its output identical
  to `nan.out.txt` (N1 to N4, bit for bit). My device test (work/nan_where.py, nan_where.out) adds
  forms N1 does not run: a NaN on the true side of `tl.where` with a `float16`/`bfloat16` `x`, with
  an `int32`/`int64` `x` (the `float32` constant `whereTyping?` gives), and NaNs Python computes
  from a `tl.constexpr` that is not a NaN (`C - C`, `C * 0.0`, `-(C - C)` with `C = inf`, whose
  key is `inf`): 0 differences. work/Check19b.lean evaluates the constants and `Operand.constIn?`
  on quiet, signalling and payload NaNs of both signs by `decide +kernel`, as a proof would, and
  checks that the numbers are unchanged (`0.1`, `1.23456e-5` in `bfloat16`). The exclusion is
  needed and right: both caches key on `str(specialization)` and every NaN's repr is `nan`
  (`primitives/triton-3.8.0-jit.py:611`, `primitives/triton-3.8.0-compiler.py:73-75`; N3
  reproduced in the re-run); the other route to a cached NaN, a module-level `tl.constexpr`
  (keyed by `str(constexpr_globals)`, `jit.py:531-537`), is already outside the subset
  (exclusions.md, the top-level values row). The constants whose meaning changed are listed. The
  `tl.float16`-argument sentence the batch added to "Scalar arguments" holds on TVA's host only
  (first new issue below), which is a defect of the unstated host, not of these constants.
- **L99** ACCEPT — Each point is fixed, and each new text is true. `toRat` of `±∞`: the module
  docstring of `TVA.Math.Encoding`, `Format.toRat` and `FloatV.toRat` give `±2^(emax+1)`, beyond
  `maxVal`, ordering the infinities, and the new `toRat_of_isInf`, `maxVal_lt_inf` prove it
  (applied in work/Check19b.lean: `+∞` of `float32` is `2^128`, above `maxVal`); every error bound
  I read (FloatErr, Sum, SumFused, FusedErr, Tree, DotLoop, SharedAcc) concludes finiteness or
  assumes it. `FloatV.nan`: the account matches part F, which re-ran identical to
  `nan_patterns.out` (two small slips, new issue below). FusedErr's formats match
  `u = 2^-(m+1)`, and `FloatV.ofRat`'s overflow by `Specials` evaluates as stated (`e2m1` `0x7`,
  `e4m3` `0x7f`, `e5m2` `0x7c`). FloatMax's list: every name resolves to a TVA declaration (no
  `le_max_left`). `tl.sqrt_rn`: `Format.sqrtMag_rn_spec`, `sqrt_rn_err` and `FloatV.sqrtRn_err` are
  proved with the standard axioms (`#print axioms`), `sqrtRn_err` applies at `float32` with
  `binary32_u`/`binary32_eta`, and the device side it rests on, `tl.sqrt_rn` equal to `Format.sqrt`
  on every `float32` input, re-ran in full (`B.sqrt.f32`). The README's seven differences and its
  argument for "exactly" are right: `cvt_rz_sat` is the statement needed, `A.cvt.f32.e5m2.rz.sat0`
  agrees on every `float32` input (re-ran in full), the 24 listed inputs are all `±∞`, and my re-run
  of `A.cvt.f32.e5m2.rz.sat1.sample` differs in 64 chunks, as recorded. `build.sh` built the driver
  under `env -i` with `PATH=/usr/bin:/bin`. The wording that remains is filed below.
- **L100** ACCEPT — Every text now says what the code states. A scripted check matched every
  backticked name in LIBRARY.md and in the docstrings of `lean/TVA/Lemmas` and `lean/TVA/Math`
  against the 2,998 declarations under `TVA` (work/names_check.out): none is missing (`Buf.Addrs.accBytes`,
  `SafeA.atomic1`, `addrs_block`, `IntV.lt_toInt`, …). `safe_upToNaN`'s list is exactly the
  definitions that are `tl.upToNaN` (`tl.cvt` is a `pick` around it, `safe_cvt`); `safe_dot_add`
  takes only `hex`; `safe_sum_bcast_mul` chooses one form for the whole tensor, as
  `tl.sum_bcast_mul` does; both Claim lemmas take `hdebug` and `hvalid`; LIBRARY.md's `Independent`
  row states `independent`'s conclusion. Both `tl.atomic_add` recipes (unfold, then `Safe.env` and
  `safe_atomic_buf`; `SafeA.env` and `SafeA.atomic`) check in work/Check19b.lean, and
  `toInt_cast`/`toInt_cast_unsigned` apply to `I32 → I64` and `U32 → U64`; the only integer casts
  the pinned tutorials write are `int32 → int64` (06, 10), which `toInt_cast` covers.

## New issues

- **NEW** soundness — SEMANTICS states values that depend on the host as facts of Triton 3.8.0,
  and nothing names the host. "Scalar arguments" (added by L98): a `tl.float16` argument holding a
  NaN is `0x7e00` or `0xfe00`, its payload dropped. That is CPython 3.13's `PyFloat_Pack2`, which
  the launcher's `extractFP16` calls (`primitives/triton-3.8.0-nvidia-driver.c:1130-1142`).
  CPython 3.14, which TVA's own `pyproject.toml` admits (`requires-python = ">=3.10"`), keeps the
  sign and the top payload bits and does not quiet: `0x7ff4000000000000` packs to `0x7d00`, a
  signalling `float16` NaN, and `0x7fffffffffffffff` to `0x7fff` (work/host_pack2.py,
  host_pack2.out, `struct.pack("<e")`, which calls the same function). "Constants": Python's
  `float('inf') - float('inf')` is `0xfff8000000000000`, x86-64's default NaN; an aarch64 host
  computes the positive default NaN `0x7ff8000000000000`, so the constant is `0x7fc00000`, not
  `0xffc00000`. `FloatV.hostF32`'s docstring says x86-64, SEMANTICS does not, and
  compiler-faithful.md point 7 names only the GPU. A claim that relies on either pattern is false on
  such a host, and no condition excludes it. Fix: state the host the facts were measured on
  (x86-64, CPython 3.13, as evidence/literals records) as part of the target, in point 7 and in
  SEMANTICS's introduction, or give these values per host.
- **NEW** wording — "Half an ulp" names a bound that can be close to a whole ulp.
  `docs/rules/num.md` ("Exact operations need no rule") says `Format.sqrtMag_rn_spec` shows
  `sqrtMag` "is within half an ulp of `√x`", and the lemma's docstring
  (`lean/TVA/Math/Rounding.lean`) says "within half an ulp of `√a`, `√a·u + η`". The lemma proves
  `√a·u + η`: for `√a` in `[2^E, 2^(E+1))` half an ulp is `2^(E−m−1)`, while `√a·u` grows to
  `2^(E−m)` at the top of the binade. The same idiom calls `u·|y| + η` "half an ulp" in
  `RoundsTo.err` (`TVA.Lemmas.FusedErr`), `FloatV.exists_nearest` (`TVA.Lemmas.Consistency`) and
  `TVA.Lemmas.Sm90`'s module docstring. Fix: say "within `√x·u + η`" (the bound of one rounding to
  nearest) there, or prove the half-ulp statement.
- **NEW** wording — `FloatV.nan`'s docstring (`lean/TVA/Lang/Scalar.lean`) lists `tl.div_rn` and
  `tl.sqrt_rn` among the `float16` and `bfloat16` operations that give the canonical NaN, but
  Triton accepts both on `float32` only (`_check_dtype(dtypes=["fp32"])`,
  `primitives/triton-3.8.0-math.py:158, 204`; part F ran `/` and `tl.sqrt` there); and it says
  `float64` arithmetic "keeps a NaN operand's quieted payload", while `tl.maximum`/`tl.minimum` with
  `propagate_nan=ALL` give `0x7ff8000000000000` from the payload NaN `0x7ff0000000012345`
  (evidence/float-layer `nan_patterns.out`). The operations are up to NaN payload, so no meaning
  changes. Fix: say which operations exist in which format, and name the `float64`
  `propagate_nan=ALL` case.
- **NEW** wording — Two pointers of the proof kit say more than the code.
  `lean/TVA/Lemmas/Int.lean`'s module docstring says unsigned comparisons "compare `toInt` too, by
  `IntV.lt_toInt`": that lemma is `<` only, and no lemma gives unsigned `<=` (`IntV.le`). And
  `docs/LIBRARY.md`'s `Block` row lists `addrs_lane` but not `addrs_block` (nor `accBytes_block`,
  `img_block`), the lemma `TVA.Lemmas.Access`'s docstring sends a 1-D block to for `Buf.Addrs`.
  Fix: say `<` (or add `IntV.le_toInt`), and list `addrs_block` in the row.
- **NEW** wording — evidence/float-layer/README.md, conclusion 4, still says of the conversions
  into fp8 "The library states neither yet (L64)". Round 14's `CvtSat` facts
  (`CvtSatF32E4m3` … `CvtSatBF16E5m2`, `docs/rules/num.md`) state the saturation to nearest; only
  the `"rtz"` conversion `float32 → float8e5` is still unstated (num.md lists it as not stated).
  The round-19 notes name this line for the coordinator. Fix: say which facts state the first and
  that the second is not stated.

## Re-ran

- evidence/literals `nan.py`, in full (Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28): output
  identical to `nan.out.txt` (work/nan.rerun.out).
- evidence/float-layer part F, `nan_patterns.py`: output identical to `nan_patterns.out`
  (work/nan_patterns.rerun.out).
- evidence/float-layer `build.sh`, under `env -i` with `PATH=/usr/bin:/bin` (it found the
  v4.35.0 toolchain through elan's `lake` and built the driver), then `run.py` from my work folder
  with that driver (work/results.out): `A.cvt.f16.e4m3.sat1` (every input, agree),
  `A.cvt.f32.e5m2.rz.sat1.sample` (differs in 64 chunks, every listed input `±∞`, as recorded),
  and in full over all `2^32` inputs `A.cvt.f32.e5m2.rz.sat0` and `B.sqrt.f32` (both agree, as
  recorded).
- Added: work/nan_where.py, nan_where.out (NaN constants on the true side of `tl.where`, with
  integer tensors, and computed from a non-NaN `tl.constexpr`; 0 differences); work/Check19b.lean,
  Check19b.out (the constants by `decide +kernel`, `constIn?`, the new theorems applied as a kernel
  proof would, both `tl.atomic_add` recipes, `#print axioms` of the batch's theorems: standard
  axioms only); work/host_pack2.py, host_pack2.out (`PyFloat_Pack2` on NaN doubles in TVA's
  Python 3.13.9 and in CPython 3.14.7); work/Names.lean, names_check.py, names_check.out (every
  backticked name in LIBRARY.md, num.md, TRANSCRIPTION.md, the two guides, and the docstrings of
  `lean/TVA/Lang/Scalar.lean`, `lean/TVA/Lemmas` and `lean/TVA/Math`, against the 2,998 declarations under
  `TVA`: what is not a declaration is a variable, a module, or a Python, LLVM or PTX name).
- The coordinator's release check on the merged tree (read only): `release check PASS (11 of 11
  steps pass)`.

## Not checked

- The other 146 tests of evidence/float-layer, and evidence/literals `run.py` (the batch reports
  its re-run identical; L98 changed no non-NaN constant, which work/Check19b.lean spot-checks).
- An aarch64 host: the sign of Python's default NaN there is the architecture's documented
  behaviour, not measured by me.
- The proofs of the lemma modules beyond their statements (Lean checks them; `tvr library-check`
  passes on the merged tree).
