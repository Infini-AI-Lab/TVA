# Library batch, round 35: author's notes

Issues L160, L161, L162, L163 and L164 ([ISSUES.md](../../ISSUES.md)). No kernel or claim folder was edited.
`lake build TVA` is green; the new theorems use only the standard axioms. No Lean definition used by a kernel
changed: the edits to existing modules are docstrings, and the two new modules add constants.

## L160: The release check without the kernels' gates

- **What changed.** `tvr release` (`tools/tvr.py`, `cmd_release`) no longer runs the gate on any kernel, and so no
  longer compares a recomputed `gate.json` or a regenerated certificate with the shipped ones, and no longer has the
  `pages` step (every page it compared, each kernel's `claim.md` and `REVIEW.md`, `kernels/README.md` and
  `REPORTS.md`, was regenerated only by the gates). It keeps `shipped`, `packages`, `sandbox`, `build`
  (`lake build TVA`), `library` (`tvr library-check`), `records`, `lint` on the records not yet recorded, and
  `audit`. The helper functions (`gate_mismatch`, `certificate_mismatch`, `page_mismatch`, `shipped_records`)
  stay, since evidence scripts of earlier rounds call them. `docs/REVIEW_PROTOCOL.md` §2 (the steps, and a paragraph
  on what was dropped and what of `kernels/` is still read), §8 ("What a gate in the tree rests on") and §9 (the
  certificate), the tool's usage text and its comment, and `evidence/release-check/README.md` (a note) say so.
- **Why.** The user's decision, 2026-11-08: "we do not need 12 certified kernels' shipped gates now. since they
  are becoming tasks in our benchmark." and "I do not think we need recomputing the gates of all 38 kernels in
  kernels/". The gates were most of the check's time.
- **Evidence.** What the check still reads of `kernels/`: each kernel's review records (`records`: its
  `review/ledger.jsonl`, the files it records, their events, its `review/snapshots`, its role folders; `lint` on
  its current round, which reads its shipped `gate.json`) and the text and imports of every file (`tvr audit`). The
  `records` step read no recomputed gate before, so it is unchanged. The release check's run is reported in the
  final message.
- **Constants whose meaning changed.** None (tooling).

## L161: A tensor store waits for every lane's atomic

- **What changed.** Fix B, by the user's decision ("let's exclude with Fix B"): the model is kept, and a launch is
  excluded in which a program uses the result of a `tl.atomic_*` call (in a store, an address, a mask, a branch or
  any operation that runs) when a cell that call reaches is also accessed by another program of the launch, **or by
  another atomic operation of the same program** (another call, or the same call run again in a loop; the lanes of
  one call are not another operation). The second case is wider than the decision's wording; it is needed (below).
  Stated on the source: `docs/exclusions.md` (new row; the relaxed-atomics row points to it), `docs/SEMANTICS.md`
  "Atomics" (the exclusion item; the `MemSem.unused` reason; "Granularity"), `docs/rules/compiler-faithful.md` and
  `Rules.Compiler.Faithful`'s docstring (scope point 12, checked; "What it says"; "What it does not cover"),
  `docs/review-questions.md` (new Q17: a per-lane model for work queues, tickets and locks; Q7, Q10), and the
  guides (`PROVER_GUIDE.md`, `DESIGN.md`, `LIBRARY.md`). In Lean: `TVA.UsesSharedAtomicResult`
  (`TVA/Sem/AtomicResult.lean`), the first case read on a model (a reachable configuration where a program is at a
  read-modify-write whose continuation differs for two old values, and a reachable step of another program
  touching its bytes); `TVA.AtomicResult.crosswarp_excluded` (`TVA/Lemmas/AtomicResult.lean`) proves that lr-34a's
  launch (`prog0`, `prog1`, `kernel`, `launch`; the delay chains, loads of unwritten buffers, written as their run
  values) is under it, for every `SEM`, platform and initial memory holding program 1's two cells apart. L157's
  premise is corrected in `MemSem.unused`'s docstring, `Step.lean` (header, `Step.defer`), `Launch.lean`
  (`Event.Strong`, the header), `Prog.lean` and `Compiler.lean`: the semantics orders a program's steps, coarser
  than the device's threads, and the launches where that shows are excluded.
- **Why.** lr-34a's launch is well behaved, yet the H200 gave an outcome the semantics lacks. A one-program launch
  shows the same (two calls of 512 lanes on one cell, each result stored), so an exclusion of other programs' cells
  alone would leave `Compiler.Faithful` false there. None of the 41 claims uses an atomic's result (each
  `tl.atomic_add` in `claims/*/Claim.lean` is `let _ ←`).
- **Evidence.** [evidence/atomic-result-order](../../../../evidence/atomic-result-order/README.md) (H200 NVL,
  Triton 3.8.0, 2026-11-08): lr-34a's launch outside the semantics in 200 of 200 runs with `acq_rel` and with
  `relaxed`; the one-program launch in 200 of 200 (first-call old values up to 895, second-call down to 384; the
  semantics: all below 512, all at least 512).
- **Constants whose meaning changed.** New: `TVA.UsesSharedAtomicResult`, `TVA.AtomicResult.skipRmw`,
  `skipRmw_rmwSeq`, `foldl_update_other`, `cont0`, `prog0`, `cont1`, `prog1`, `kernel`, `launch`, `launch_size`,
  `p0`, `p1`, `p1_ne_p0`, `reach_snoc`, `reach_snoc_upd`, `crosswarp_excluded`. Docstrings only: `Rules.Compiler.Faithful`
  (scope 12), `MemSem.unused`, `Step.defer`, `Event.Strong`. Every kernel item that reaches `Rules.Compiler`,
  `Sem.Step`, `Sem.Launch` or `Lang.Prog` is re-opened by the digests.

## L162: "Unused" on the source and on the IR

- **What changed.** `docs/SEMANTICS.md` "Atomics" (the unused-result item) and the docstring of the unused atomics
  in `lean/TVA/Lang/Ops.lean` say where the readings differ (a value read only by an operation whose own result is
  unused, such as `s = r + 1` with `s` never read or `tl.device_assert(r > 0)` without `debug`; a value returned from
  a helper whose call is a statement), that either may be applied and both are sound, since the model drops the
  value only when nothing that runs on the device reads it, and that a reviewer who cannot tell treats the result as
  used. The L161 exclusion reads "uses" the same way.
- **Why.** lr-34a: the two definitions did not say which governs.
- **Evidence.** None needed (wording).
- **Constants whose meaning changed.** None (docstring of `tl.atomic_add_unused` and the others' section).

## L163: Float `/` fused into a following add

- **What changed.** A float `/` whose quotient is `float32` counts as a product for the fusion rule:
  `docs/SEMANTICS.md` "Float `+` and `-`" (the definition of a product, and a new item: `/` and `tl.fdiv` on
  `float32`, `/` of `float16`, `bfloat16` or integer operands, so `tl.sigmoid` and `tl.softmax`; the addition is
  `tl.addc`/`tl.subc`, `AddRel E.fusion`, on the quotient `tl.div` picks, with its own `DivFull` error; a reduction
  or scan of quotients `tl.sumc`/`tl.cumsumc`; not `tl.div_rn`, `float64` `/`, `//`, `%`, `tl.cdiv`), "`/` on
  floats" and `tl.sigmoid`; `docs/rules/num.md` "Division" (a following addition); the docstrings of `tl.add` and
  `tl.div`; `docs/TRANSCRIPTION.md` (table row and pitfall), `PROVER_GUIDE.md`, `DESIGN.md`. `AddRel` itself is
  unchanged: the narrow relation already allows an operand entering unrounded as an exact value that rounds to it.
- **Why.** lr-34b: ptxas fuses `div.full.f32`'s final multiply (reciprocal times the scaled dividend) with a
  following addition into one `FFMA`, so the IEEE form gave no outcome for `x / y - q`.
- **Evidence.** [evidence/div-fusion](../../../../evidence/div-fusion/README.md) (H200 NVL, Triton 3.8.0,
  2026-11-08): 18 spellings (`x / y + c`, `c + x / y`, `x / y - c`, `c - x / y`, `-(x / y) + c`, through
  `tl.where`, `.to(tl.float32)`, `.to(tl.float64)`, `.to(tl.float16)`, `tl.fdiv`, `tl.sigmoid`, a constant divisor,
  `1.0 / y`, two quotients, a `tl.sum`, a loop `+=`, `float16` and `bfloat16` operands), six regimes each (nearly
  cancelling addends, random patterns, divisors beyond `2^126` and subnormal, subnormal quotients, all special
  values): 0 of 430080 results outside `AddRel` at the quotient the division alone stores, 162200 differ from the
  IEEE sum; the `tl.where` and converting `.to` forms are not fused; with fusion off, all IEEE. Claims whose
  covered path has a `float32` quotient feeding an addition (to be re-checked; not edited):
  `claims/layer_norm_fwd_f16` (`source/05-layer-norm.py:72` `mean` → `:78` `x - mean`, `Claim.lean:82` `tl.sub`;
  `:72` → `:92` `x - mean`, `Claim.lean:111` `tl.sub`; `:80` `var` → `:81` `var + eps`, `Claim.lean:91` `tl.add`)
  and `claims/layer_norm_fwd_bf16` (`source/layernorm.py:91` `b_mean` → `:94` `b_x - b_mean`, `Claim.lean:104`
  `tl.sub`; `:91` → `:108`, `Claim.lean:132` `tl.sub`; `:95` `b_var` → `:99` `b_var + eps`, `Claim.lean:116`, already
  `tl.addc`). The other 39 have no such path: their quotients are only stored, converted, compared or multiplied
  first (`attn_fwd_f16`, `extend_attn_bf16`, `paged_attention_f16`, `softmax_f32`, `seeded_dropout_f32`,
  `kv_cache_write_bf16`, `l2norm_fwd_bf16`, `gdn_gating_bf16`, `merge_attn_states_bf16`, `moe_router_bf16`,
  `prefill_attn_varlen_bf16`, …), or have no float `/`.
- **Constants whose meaning changed.** None in Lean (docstrings of `tl.add`, `tl.div`). The transcription of a
  `float32` `/` followed by an addition changes: such an addition is the fusable form.

## L164: "One `atom`" per atomic

- **What changed.** `docs/SEMANTICS.md` "Atomics" and `evidence/relaxed-atomics/README.md` (Part P) say "one atomic
  read-modify-write per enabled lane": a float `atomic_max`/`atomic_min` is two predicated integer `atom`s
  (`max.s32`, `min.u32`), at most one of which runs per lane, neither with a NaN operand, which `fBound` allows.
- **Why.** lr-34b: "one `atom`" was wrong for float `max`/`min`.
- **Evidence.** lr-34b's `work/atom_ptx_bar.txt` (round 34); the measurement of `evidence/relaxed-atomics` is
  unchanged.
- **Constants whose meaning changed.** None (wording).
