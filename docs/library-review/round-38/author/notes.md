# Library batch, round 38: author's notes

Issues L168 (reopened), L169 to L174 ([ISSUES.md](../../ISSUES.md)). This is the final fix: by the user's decision
(2026-11-08, "let's run up to round37, and file a final fix without further review") no library review follows,
so every fix is wording and evidence only, and each issue is marked `FIXED`. No kernel or claim folder was edited.
`lake build TVA` is green. The only Lean edit is the docstring of `Step.defer` (`lean/TVA/Sem/Step.lean`), so no
definition or statement changed; the digests re-open every item whose cone reaches the edited files
(`lean/TVA/Sem/Step.lean`, `docs/SEMANTICS.md`). No claim changes: none uses `tl.softmax` (every `softmax` in
`claims/*/source` is the host's), and the four claims with atomics (`streamk_matmul_f16`, `lora_shrink_bf16`,
`route_histogram_i32`, `layer_norm_fwd_f16`) use none on a 16-bit cell, none with an operand of another type than the
cell, and none whose result they use inside their claim. Last sweep: `docs/` (prompts included), every
`evidence/*/README.md` and the Lean docstrings were grepped for "as a whole", "depends on every lane", "holds back",
`f16x2`, "pre-sm_90", "compare-and-swap", "signed max", "unsigned min", the old line references
(`1315-1335`, `1272-1275`), the old softmax format lists, "converted" near atomics, and the release check's
removed steps; each hit that contradicted a fix is fixed below, the others are consistent (Q17, `Prog.lean`,
`AtomicResult.lean`, `Compiler.lean`, `compiler-faithful.md` say a step that uses any lane's result holds back every
later atomic, which is true; `f16x2` elsewhere names conversions and `fma` forms, not atomics).

## L168: Softmax formats and atomic wording

- **What changed.** Part (1), the reopened point: SEMANTICS.md "`/` on floats" gives `tl.softmax` on `float8e5` and
  `float8e4nv` with the 16-bit formats (it computes and returns `float32`; `div.rn.f32` with `ieee_rounding=True`,
  `div.full.f32` by default) and adds integer inputs (no outcome: `tl.exp` takes only `float32` and `float64`,
  `math.py:94-98`). "Float `+` and `-`" lists the fp8 formats among the products (default) and the non-products
  (`ieee_rounding=True`). The same in the conclusion of `evidence/div-fusion` and in `evidence/fdiv-ieee` (new Part F).
  Part (2) was accepted in round 37 and is unchanged.
- **Why.** lr-37a (REOPEN) and lr-37b: the per-format statement left out the fp8 formats, which compile.
- **Evidence.** `evidence/fdiv-ieee/fmt_probe.py` (lr-37a's `work/fmt_probe.py`), `fmt_probe.out.txt` (GPU 0, H200 NVL,
  Triton 3.8.0, 2026-11-08): fp8 softmax 32 bits wide, `div.full.f32` / `div.rn.f32`; `int32`, `int16` rejected;
  identical to lr-37a's output. `softmax_fmt.py` re-run: identical to `softmax_fmt.out.txt`.
- **Constants whose meaning changed.** None (wording).

## L169: Lane granularity of an atomic's result

- **What changed.** SEMANTICS.md "Programs": a tensor step waits for the atomic of each lane whose result it uses
  (only those: a store masked to lane 0 waits only for lane 0's), and since it is one step, every later step of the
  program, every later atomic included, waits for it. "Atomics" and review-questions.md Q7: a tensor step that uses
  one lane's result holds back every later atomic of the program (the "Granularity"/Q17 wording). The `Step.defer`
  docstring says the same (it waits for this lane's atomic, and holds back every later atomic).
- **Why.** lr-37a: "depends on every lane" and "as a whole" were coarser than `Step.defer`, which lets a step pass a
  pending lane whose old value changes neither its memory nor its event.
- **Evidence.** None needed (wording, read off `Step.defer`, `writeAll` and `accBytes` in `lean/TVA/Sem/Step.lean`).
- **Constants whose meaning changed.** None (a docstring only).

## L170: Evidence describing removed release steps

- **What changed.** `evidence/gate-round7/README.md` and `evidence/tool-round18/README.md` gain a "Since round 35" note
  (as `evidence/release-check` has) saying that the release check no longer recomputes gates or compares pages, and
  their conclusions 5 are put in the past tense. The same note in `evidence/tool-round10/README.md`, whose Question 6
  named the certificate and page comparisons (the ledgers' comparison remains, as `records`).
- **Why.** lr-37a: the READMEs described steps removed by L160 in the present tense.
- **Evidence.** `cmd_release` in `tools/tvr.py` (steps shipped, packages, sandbox, build, library, records, lint, audit).
- **Constants whose meaning changed.** None (wording).

## L171: The release check's steps in LIBRARY_REVIEW.md

- **What changed.** The `gate` row of `docs/LIBRARY_REVIEW.md` asks whether `tvr release` can pass a library that
  depends on anything it does not ship (`shipped`, `packages`, `sandbox`), does not build (`build`), or fails
  `library`, `records`, `lint` (the records not yet recorded: the library's last round and each kernel's current
  round, `release_records`) or `audit`; and says that the audit reads the text and the kernels' Lean imports
  (`kernel_independence`).
- **Why.** lr-37a: the row omitted the check's own purpose and the imports.
- **Evidence.** `cmd_release`, `release_records`, `kernel_independence` in `tools/tvr.py`.
- **Constants whose meaning changed.** None (wording).

## L172: 16-bit atomics on sm_90

- **What changed.** review-questions.md Q7 now describes sm_90: a lane of a `float16`/`bfloat16` `atomic_add` is
  `atom.add.noftz.f16` (`.bf16`), and of a 16-bit `atomic_cas` `atom.cas.b16`, each compiled by `ptxas` to a 32-bit
  compare-and-swap loop on the aligned word (writing the neighbouring half back); contiguous unmasked lanes one
  thread holds become one vector atomic (`.v8.f16`, `.v8.bf16`); `f16x2` packing only where vector atomics are not
  supported (`nvidia-LoadStoreOpToLLVM.cpp:636-646, 729-740`). The trust it describes is unchanged (part of the
  compiler trust; a test, not a proof). Part P of `evidence/relaxed-atomics` says "one `atom`" is in the PTX.
- **Why.** lr-37b: Q7 described the pre-sm_90 forms.
- **Evidence.** New `evidence/half-atomics` (`run.py`, lr-37b's `work/half_atomics.py`; `sass_dump.py`; GPU 0,
  2026-11-08): 0 of 1024 lanes off one IEEE addition in each of 8 forms; 0 of 40960 cells wrong with plain stores
  into the neighbouring halves (20 launches per format); the SASS loop for both formats; outputs identical to
  lr-37b's. `evidence/atomic-dtypes/adv.py` part S: the 16-bit `atomic_cas` is the same loop.
- **Constants whose meaning changed.** None (wording).

## L173: Atomic operand conversion

- **What changed.** SEMANTICS.md "Atomics" gains "The operand": every `tl.atomic_*` but `atomic_cas` converts its
  operand to the cell's element type with `semantic.cast`'s default rounding, as `tl.store` (`semantic.py:1280`,
  `core.py:2757-2829`); `atomic_cas` converts nothing (`semantic.py:1257-1263`, `core.py:2746-2751`), so operands of
  another type than the cell are rejected on the host (no outcome). The transcription row of `tl.atomic_add` says the
  same.
- **Why.** lr-37b: the conversion was stated only in passing, and the `atomic_cas` case not at all.
- **Evidence.** `evidence/atomic-dtypes/adv.py` (lr-37b's `work/atomic_adv.py`), `adv.out.txt` (GPU 0, 2026-11-08,
  identical to lr-37b's): `float32` operands into `float16`, `bfloat16`, `int32` cells match torch's conversion on 64
  of 64 lanes each (`65520` to `+∞`, truncation into `int32`); four mismatched `atomic_cas` calls rejected ("'tt.atomic_cas'
  op failed to verify that ptr type matches cmp type").
- **Constants whose meaning changed.** None (wording).

## L174: S3's max/min pair

- **What changed.** review-questions.md S3: `atomic_max` is signed max or unsigned min, `atomic_min` signed min or
  unsigned max, chosen by the sign bit of `val` (`semantic.py:1299-1374`); the fp16/bf16 check is
  `semantic.py:1271-1275`.
- **Why.** lr-37b: the sentence held for `atomic_max` only, and its line references were off.
- **Evidence.** The pinned `primitives/triton-3.8.0-semantic.py`; `evidence/atomic-dtypes/adv.out.txt` part S/U
  (`max.s32`/`min.u32` for `atomic_max`, `max.u32`/`min.s32` for `atomic_min`).
- **Constants whose meaning changed.** None (wording).
