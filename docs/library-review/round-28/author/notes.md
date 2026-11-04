# Library batch, round 28: author's notes

Issue: L136 ([ISSUES.md](../../ISSUES.md)). The user's request (2026-11-04), after a sandbox run in which
`paged_attention_f16` was certified while its README stated round 1's tolerance constants: "fix both and
push" (the other fix is the sandbox's: its reviewers could not run Lean). No library review follows unless
the user asks for one. Checked by `tvr check` and `tvr status` on every kernel and `tvr lint library 28
author`. No Lean file changed, so `lake build TVA` and `tvr library-check` were not re-run.

## L136: a write-up that states a claim other than the Lean's blocks the certificate

- **What changed.**
  - `tools/tvr.py`: a new finding kind `DOC_WRONG` in `KINDS`, blocking, with the targets `item:`,
    `decl:` and `doc:`. No other kind changed, so no recorded ledger derives a different event (library
    issue L82): no ledger holds the new kind.
  - `docs/REVIEW_PROTOCOL.md` (the findings table): `DOC_WRONG`, the write-up (`README.md`, or a docstring
    of a constant in an item's cone) states a claim other than the Lean's: a constant, tolerance,
    condition, narrowing or assumption that differs from its definition.
  - `docs/REVIEWER_GUIDE.md` §5 (the write-up): when to file `DOC_WRONG` rather than `DOC`.
- **Why.** In the sandbox run, the prover tightened `Xcap` from `2^6` to `2^4` and `tolR`'s last term from
  `2^-9` to `2^-10` in round 1, and four sentences of `README.md` kept the old values. Both reviewers
  found them in round 2 and filed them as `DOC`, the only kind there was, so the certificate was granted
  with a write-up that told its reader the accuracy bound covers scores up to 64 where the Lean covers
  16. A reader who checks a workload against the write-up would be wrong about it. `CONDITION_WRONG`
  covers a condition "not what the write-up says"; nothing covered the specification's constants, a
  narrowing or an assumption, and wording remarks must stay non-blocking.
- **Evidence.** `tvr`'s filing check on three findings of the new kind: a complete one passes; one
  without `detail` is refused (a blocking finding explains itself); one marked `"blocking": false` is
  refused (`DOC_WRONG is blocking`). `tvr check` re-ran on each of the twelve kernels (`attn_bwd_f16`,
  `attn_bwd_pre_f16`, `attn_fwd_e5m2`, `attn_fwd_f16`, `block_scaled_matmul_mxfp8`,
  `grouped_matmul_tma_f16`, `matmul_e5m2`, `matmul_f16`, `paged_attention_f16`, `softmax_f32`,
  `streamk_matmul_f16`, `vec_add_f32`): each passed with the digest it had, and `tvr status` grants
  each certificate again. The kernel files that changed are the gates' records (`gate.json`: its time
  and the sha256 of `tools/tvr.py`; `claim.md` and `REVIEW.md`: the gate's time;
  `review/certificate.json`: its time).
- **Constants whose meaning changed.** None: no Lean file changed. The gate's code changed, so every
  kernel's gate was stale until re-run; item digests do not cover `tools/tvr.py`, so no acceptance
  changed.
