# Library batch, round 36: author's notes

Issues L160, L164, L165 and L166 ([ISSUES.md](../../ISSUES.md)). No kernel or claim folder was edited.
`lake build TVA` is green. The only Lean edits are two docstrings (`Rules.Compiler.Faithful`, `tl.div_rn`), so no
definition or statement changed; the digests re-open every item whose cone reaches the edited files
(`lean/TVA/Rules/Compiler.lean`, `lean/TVA/Lang/Ops.lean`, `docs/SEMANTICS.md`, `docs/exclusions.md`,
`docs/rules/num.md`, `docs/rules/compiler-faithful.md`). No claim changes: none uses a float `atomic_max`/`atomic_min`,
an atomic's result, `tl.debug_barrier`, `tl.fdiv(…, ieee_rounding=True)` or `tl.softmax` (every `softmax` in
`claims/*/source` is the host's `torch.softmax`).

## L160: The release check without the kernels' gates

- **What changed.** The descriptions still naming the dropped steps (lr-35a): `docs/REVIEW_PROTOCOL.md` §2 (a scratch
  path in a work file fails the audit, in every run; the clause "and so the `independence` check of every kernel's
  gate" is gone), the same clause in `docs/prompts/library-reviewer.md` and `docs/prompts/library-author.md`, and
  `docs/DESIGN.md`: the release-check row now lists the steps the sandbox runs as "the build, the library's replay,
  the records, the lint and the audit" ("built and checked", not "gated"), and the ledger row says a gate in the tree
  is as the last `tvr check` wrote it, since the release check no longer recomputes it.
- **Why.** lr-35a: no kernel's gate runs in the release check since round 35, so those sentences described steps that
  no longer exist. `cmd_release` itself was accepted by both reviewers and is unchanged.
- **Evidence.** None needed (wording). The release check of this round is reported in the final message.
- **Constants whose meaning changed.** None (documentation).

## L164: "One `atom`" per atomic

- **What changed.** `docs/SEMANTICS.md` "Atomics" and Part P of `evidence/relaxed-atomics/README.md` now say: a float
  `atomic_max` or `atomic_min` is two integer atomics predicated on the operand's sign bit, so exactly one runs per
  enabled lane, NaN operands included: `max.s32` (sign bit clear) or `min.u32` (set) for `atomic_max`, `min.s32` or
  `max.u32` for `atomic_min`, `.s64`/`.u64` for `float64`. "At most one", "neither with a NaN operand" and "or none"
  are gone. A new Part F of the evidence checks it.
- **Why.** lr-35a and lr-35b: Triton 3.8.0 predicates the two atomics on `mask & !signbit(val)` and
  `mask & signbit(val)` (`triton-3.8.0-semantic.py:1292-1373`), so the old sentence was false.
- **Evidence.** `evidence/relaxed-atomics/float_maxmin.py`, `float_maxmin.out.txt` (H200 NVL, Triton 3.8.0,
  2026-11-08), after lr-35b's `work/atom_maxmin.py` and lr-35a's `work/fmax_nan.py`: `float32` and `float64`, `max`
  and `min`, 2048 lanes, every pair of 40 bit patterns (both zeros, infinities, subnormals, quiet and signalling NaNs
  of both signs, random), every 17th lane masked. In each of the four cases 0 of 1927 enabled lanes differ from the
  one atomic the sign bit picks, 0 return other than the old value, 0 lie outside `fBound`, and 0 of 121 masked-off
  cells changed; 73 or 74 of 151 NaN operands were written. `fBound` is unchanged.
- **Constants whose meaning changed.** None (wording).

## L165: Wording around the atomic-result exclusion

- **What changed.** (1) `docs/SEMANTICS.md` "No barrier semantics": "a barrier could only remove behaviours the model
  has" now holds "outside the exclusion of Atomics"; the paragraph cites the barrier result, says the barrier still has
  no meaning (so such a launch stays excluded), and points to Q17, which now records that a `tl.debug_barrier()`
  between the use and the next atomic gave only the semantics' outcomes. (2) Launch A with `relaxed` is no longer
  called a counterexample: it races (its used atomics keep `relaxed`, and each conflicts with the other program's
  atomic without being `Event.Strong`), while launch B stays well behaved with `relaxed` (`Event.SameAtomic`). This is
  stated in `evidence/atomic-result-order/README.md` (method and conclusions), `docs/SEMANTICS.md` "Atomics",
  `docs/exclusions.md`, `docs/rules/compiler-faithful.md` and `Rules.Compiler.Faithful`'s docstring (scope 12), and
  `docs/review-questions.md` Q17. (3) Scope point 12 (both texts), SEMANTICS.md's exclusion item and the exclusions
  row now say: a reviewer who cannot tell whether another program or another atomic operation of the program reaches
  a cell the call reaches treats the cell as shared, and the launch as excluded.
- **Why.** lr-35a's new issue, round 35.
- **Evidence.** `evidence/atomic-result-order/variants.py`, `variants.out.txt` (lr-35a's `work/atomic_variants.py`,
  re-run; H200 NVL, Triton 3.8.0, 2026-11-08, `acq_rel`, 100 launches each): the barrier variant 0 of 100 outside the
  semantics; the loop, address-only and unused-second-result variants 100 of 100. The race of launch A with `relaxed`
  is read off `Event.Strong`, `InterRace` and `IntraRace` (`lean/TVA/Sem/Launch.lean`).
- **Constants whose meaning changed.** None (docstring of `Rules.Compiler.Faithful`, scope 12).

## L166: `ieee_rounding=True` divisions

- **What changed.** `docs/SEMANTICS.md` "`/` on floats": `tl.fdiv(x, y, ieee_rounding=True)` on `float32` operands is
  `tl.div_rn` (both `tt.precise_divf`, `div.rn.f32`), and on other formats it fails to compile; `tl.softmax` ends in
  `tl.div_rn` with `ieee_rounding=True` and in `/` with the default. "Float `+` and `-`": `tl.softmax` is a product only
  with `ieee_rounding=False`; `tl.fdiv(…, ieee_rounding=True)` and `tl.softmax(…, ieee_rounding=True)` join the
  non-products, and writing their following addition as the fusable one is sound but loose. `docs/rules/num.md`
  "Division" (front end; a following addition), the docstring of `tl.div_rn` in `lean/TVA/Lang/Ops.lean`, and a row of
  `docs/TRANSCRIPTION.md`'s table say the same.
- **Why.** lr-35b's new issue, round 35: the library gave `ieee_rounding=True` no meaning and called `tl.softmax` a
  product unconditionally.
- **Evidence.** `evidence/fdiv-ieee` (new; H200 NVL, Triton 3.8.0, 2026-11-08): the PTX (`div.rn.f32` for
  `ieee_rounding=True` and `tl.div_rn`, `div.full.f32` for the default; `float16`, `bfloat16` and, with
  `ieee_rounding=True`, `float64` operands fail to compile); quotients bit for bit equal to `tl.div_rn`'s and to the
  IEEE quotient on every pair of 256 patterns and on `2^24` random pairs; `tl.fdiv(…, ieee_rounding=True) ± c`: 0 of
  1048576 results off the IEEE sum, where an unrounded quotient differs on 1048575; `tl.softmax(…, ieee_rounding=True)`:
  0 of 65536 off `num / den` with the IEEE division. lr-35b's `work/fusion_sweep.py` agrees.
- **Constants whose meaning changed.** None (docstring of `tl.div_rn`). A model of `tl.fdiv(x, y, ieee_rounding=True)`
  is now `tl.div_rn`, and its following addition the IEEE one.
