---
name: tva-respond
description: Answer the reviewers' findings in a TVA review round (fix, re-gate, response.json and notes.md via tva-kit respond, resubmit). Use whenever tva-review status lists open findings.
---

# Answering a round (PROVER_GUIDE §5, REVIEW_PROTOCOL §9 and §11)

One batch answers **every** open finding. The certificate needs every reviewer to accept every item at its
current digest, with no blocking finding unresolved.

## Procedure

1. `tva-kit findings`: each open finding in full (kind, target, summary, detail, evidence). Read the
   reviewer's report too (the path is printed) and `REVIEW.md` in your kernel folder.
2. Decide each one:
   - the reviewer is right → **fix** the Lean/README/map, and answer `REVISED`, naming the items your fix
     changed (`item:model`, `item:map`, `item:spec`, `item:conditions`, `item:assumption:<field>`);
   - it is wording only → fix the words and answer `CLARIFIED`;
   - you keep a narrowing deliberately → state it in README and answer `DEFERRED` with the reason;
   - you think it is wrong → `DISPUTED` with a precise argument (cite SEMANTICS.md, the Lean, the source line);
   - it is about the library, not your kernel → `REFERRED --issue L<n>` (an issue of
     `docs/library-review/ISSUES.md`).
3. After all fixes: `tva-kit gate` until PASS (the notes carry its digest).
4. `tva-kit respond init` (once per round), then for each finding:
   `tva-kit respond set F3 REVISED --changed item:spec --text "Post now bounds … (Lean line …)."`
   `tva-kit respond set F4 DISPUTED --text "Line 52 adds two loaded values; no product feeds it (…)."`
   and `tva-kit respond change "…"` for each change you made.
5. `tva-kit respond lint` until it says regular, then `tva-review submit` and `tva-review wait`.

## What wins rounds

- Fix rather than argue when the finding has a point: a reviewer re-raises a finding it thinks is not
  settled, and the **second re-raise escalates it to a user who is not here: no certificate can follow**.
- `REVISED` must name items whose digest actually changed (`tva-kit respond set` warns when not).
- A dispute must be checkable: quote the source line, the SEMANTICS.md section, the Lean declaration.
- Never write instructions to the reviewers in your files; they are told to treat that as a defect.
- Keep the scope honest: narrowing to pass a finding is fine only when stated in README and justified
  by the kernel's real use.
