# Review of block_scaled_matmul_mxfp8, round 2, by r-mx8b

## Verdict

accept
The prover's write-up revision settles F2: the README now says the benchmark's launches, made under proton's hook, are not covered, and names what is covered (the test's launch and hook-free launches of the benchmark's sizes). The gate and every digest are unchanged since round 1, and I have no other blocking finding, so I accept every item.

## Findings

- **release F2** — the README and SETUP.md now state that the benchmark's hooked launches (after line 756, until line 764) are outside point 3, and what the claim covers

## Accepted

All items.

## Checked

- The prover's response and notes (round 1), and REVIEW.md: the gate is unchanged at `2fac55fa1cb0c3c5`, and so is every item's digest.
- The revised README in full where it changed: the introduction (lines 32–38), Conditions ("Which of the host's launches are covered", lines 156–175), the part (A) preamble and the point-4 and point-9 bullets, the part (B) paragraph (lines 209–219), and the Narrowings bullets on the benchmark, the environment and `K ≤ 2^14`. Each matches the source: `proton.start` at line 756, `proton.deactivate()` at lines 501 and 757, `proton.finalize()` at line 764, warm-up launches at line 484, timed launches at line 492, and the test at line 749, before line 756.
- SETUP.md (new), read whole: it states the same narrowing.

## Not checked

Nothing.
