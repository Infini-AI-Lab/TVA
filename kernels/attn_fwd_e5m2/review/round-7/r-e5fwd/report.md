# Review of attn_fwd_e5m2, round 7, by r-e5fwd

## Verdict

accept

The README's benchmark ranges for N = 1024 now cover every figure from my recorded round-5 runs, so F15 is settled. The gate is unchanged and only the README changed, so all items stay accepted. I raise no new finding.

## Findings

- **release F15**: settled. All of my recorded bench figures (seeds 41, 42, 50-55) lie inside the stated ranges. tolO: D = 64 0.408-0.430 (README 0.40-0.46), causal 0.335-0.376 (0.33-0.39); D = 128 0.651-0.839 (0.65-0.90), causal 0.502-0.639 (0.50-0.69). Falls: 11.0-21.8% (11-27%) and 20.2-38.5% (20-51%). Rejected: D = 64 64.3-70.0% (60-70%), causal 70.2-77.4% (70-78%); D = 128 40.6-50.9% (38-51%), causal 49.9-61.8% (47-62%). 0 meets Post: 22.6-35.7% (22-40%) and 38.2-59.4% (38-62%). tolO/floor: 3.75-10.3 (3.7-12.2). tolV and tolM are also inside, at the three significant digits my runs print.

## Accepted

All items.

## Checked

- The README's benchmark table, the paragraph 'On the benchmark's data' and its coverage sentence, checked against the bench lines of my round-5 work/tol.out and work/tol_spread.out.
- The gate digest is unchanged at 3d7848ee51b5e6a6.

## Not checked

- New samples, as the coordinator asked.
