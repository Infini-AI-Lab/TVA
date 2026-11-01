# Review of block_scaled_matmul_mxfp8, round 2, by r-mx8

## Verdict

accept
The prover's CLARIFIED response settles F1. The README now states that the benchmark's launches, which the script makes under proton's launch hooks, are outside `Compiler.Faithful`'s scope point 3. It also says which of the host's launches the claim does cover. No Lean, condition or digest changed since I accepted every item in round 1, so I keep that acceptance.

## Findings

- **release F1** — the README (introduction, Conditions, part (A) and (B) paragraphs, a Narrowings bullet) and SETUP.md now say the benchmark's hooked launches are not covered, and name the covered ones (the test's launch and hook-free launches of the benchmark's sizes)

## Accepted

All items.

## Checked

- The prover's response and notes (round 1), and the README's changed sections against the source: line 749 (the test, before line 756), line 756 (`proton.start(…, hook="triton")`), lines 484 and 490–501 (the warm-up and timed launches), and line 764 (`proton.finalize()`). The statement agrees with rules/compiler-faithful.md scope point 3 and with exclusions.md's "a hook" row.
- SETUP.md, read whole: it states the same coverage.
- The gate is unchanged at `2fac55fa1cb0c3c5`, its recorded sha256 of README.md is the file's current one, and every item's digest is the one I accepted in round 1.
- The `K ≤ 2^14` Narrowings bullet, revised to say that the proof itself needs only the `int32` bounds and the room condition, and that the bound is kept for the tolerance: it is consistent with the Lean.

## Not checked

- Proton's code. For which hooks it installs and how long they stay, I rely on the coordinator's check.
