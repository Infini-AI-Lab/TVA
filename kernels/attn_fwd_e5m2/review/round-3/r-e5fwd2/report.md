# Review of attn_fwd_e5m2, round 3, by r-e5fwd2

## Verdict

accept
The README changes answer both of my round-2 DOC findings, so I release them. The test-data
figures are now a table per N_CTX and causal or not. Every cell covers my round-2 runs, and the
derived percentages are consistent with the table. The new paragraph "What tolV checks" is correct:
tolV is at least 9/8·K + |attnRef|/8 + 2^-17, so it accepts any average of the row's own values. No
item digest changed, so I accept all items again. There are no new findings; two wording remarks
are in the notes.

## Findings

- **release F7** — the per-N_CTX table covers my round-2 figures (87–89.5% at N_CTX = 128, D = 64 causal; floor 26–54% at 128), and Narrowings now say 87–100% and attribute the gap at 128 and on causal rows to Num.DotE5m2
- **release F8** — the paragraph "What tolV checks" is correct: tolV ≥ 9/8·K + |attnRef|/8 + 2^-17, so any average of the row's values meets it, and on its own tolV checks the key set and the range, not the weights

## Accepted

All items.

## Checked

- The per-N_CTX table and the text around it against my round-2 tolerance.out (seed 20 plus three seeds at N_CTX = 128 and 1024): every share of 0 meeting Post, every median |attnRef| and every fp8-floor share lies inside the stated ranges. 46–74%, 7–46%, 13% and 3.5% follow from the table, and pE is about 1.1 and 1.5 on the test's data.
- The main table's new N = 128 test-like rows, and the updated benchmark figures: tolV at N = 1024, the pE medians and maxima, and the N = 2^14 tolO, tolV and tolM.
- The derivation of tolV ≥ 9/8·K + |attnRef|/8 + 2^-17 from Model.lean's magA, magCp and tolV, and the float8e5 rounding bound that makes any average of the row's values meet the tolV clause.
- That the Narrowings match the Specification, and that none of the old figures (97.7%, 81%, 0.009 to 0.04) remain in the README.

## Not checked

- The one-key row's tolO ranges in the main table, which I did not compute.
