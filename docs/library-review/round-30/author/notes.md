# Library batch, round 30: author's notes

Issue: L138 ([ISSUES.md](../../ISSUES.md)). Found by the coordinator in the sandbox test set's second wave
(2026-11-05): the prover of `l2norm_fwd_bf16` reported that every submission after round 1 was rejected
on the independence line of a page it may not edit. No library review follows unless the user asks for
one. No Lean file changed, so `lake build TVA` and `tvr library-check` were not re-run.

## L138: table cells escape `|` instead of writing `/`

- **What changed.** `tools/tvr.py`: a function `md_cell` escapes `|` as `\|` (and a line break as a
  space) in a Markdown table cell; the finding tables of `REVIEW.md` and `REPORTS.md` and the gate table
  of the review packet use it in place of `.replace('|', '/')`.
- **Why.** A summary such as `|x|=2^30` became `/x/` followed directly by `=2^30` on the generated page, which the text scan
  reads as an absolute path outside TVA, so the kernel's gate failed on its own review record, with no
  remedy open to the prover.
- **Evidence.** `foreign_refs` on the row with `/x/` followed directly by `=2^30` reports "a path outside TVA (that string)"; on the
  same row through `md_cell` (`\|x\|=2^30`) it reports nothing. Six kernels' `REVIEW.md` and `REPORTS.md`
  held summaries with `|`; they now show it escaped.
- **Constants whose meaning changed.** None: no Lean file changed.
- **Effect on kernels.** The gate records its own code's digest, so every gate was stale; `tvr check` was
  re-run on the twelve kernels with a gate (all PASS, the same digests). No item's digest changed: the
  certificates round 29 withdrew stay withdrawn until their reviewers re-accept, as before.
