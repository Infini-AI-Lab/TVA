# Review of vec_add_f32, round 2, by r-vecadd

## Verdict

accept
Only the write-up changed since round 1. The Lean, the map and the manifest are unchanged, and
every item's digest is still the one I accepted at gate `b7289baba9114347`. The revised README
now explains each excluded size range correctly: the `int32` wrap for `n > 2^31`, the `int64`
`n_elements` at `n = 2^31`, and `Block.Fits` (library issue L8) for
`2^31 − BLOCK_SIZE < n < 2^31`. So I release F1. The revised aliasing wording matches `Inst.ok`.
I have no new finding, and I accept all items again.

## Findings

- **release F1** — the README now gives the right reason for each of the three excluded ranges (wrap for n > 2^31, int64 n_elements at n = 2^31, L8 below 2^31), checked in work/ranges.py

## Accepted

All items.

## Checked

- What changed: `REVIEW.md` and `tvr status` show the gate `b7289baba9114347` and item digests unchanged since my round-1 acceptance. The prover's notes and response say only `README.md` changed.
- F1's resolution: the README Narrowings bullet on `n + BLOCK_SIZE ≤ 2^31` against the kernel's arithmetic. work/ranges.py checks the following for every `BLOCK_SIZE = 2^e ≤ 2^20`:
  - For `n > 2^31`, the last program has `pid ≥ 2^31 / BLOCK_SIZE` and `pid * BLOCK_SIZE ≥ 2^31`. Its `int32` offsets are negative up to `n = 2^32`, and at `n = 2^32 + 1` a product wraps to offset `0`, onto another program's elements.
  - At `n = 2^31`, the largest offset is `2^31 − 1` and nothing wraps.
  - The range `2^31 − BLOCK_SIZE < n < 2^31` has `BLOCK_SIZE − 1` sizes. Each has a last program with `p * BLOCK_SIZE + BLOCK_SIZE = 2^31`, so the strict `Block.Fits` of the recorded library fails there.
  - With `BLOCK_SIZE = 1024`, the bound is `n ≤ 2^31 − 1024`.
- The Conditions table row for `n_fits` gives the same three reasons, and "the last `BLOCK_SIZE − 1` sizes below `2^31`" is exact.
- The overlap row and the Aliasing bullet ("can make programs race"; conditions on whole buffers, `Buf.bytes`, so a harmless overlap beyond the first `n` elements is also excluded) match `Inst.ok.output_x`, `output_y` and `TVA.Buf.bytes`.
- The `Pre` description in the README (base non-negative and 4-byte aligned, inside `[0, 2^64)`, every byte present) matches `TVA.Buf.Valid`.
- work/Axioms.lean: `main`'s statement is `∀ K P, Assumptions K P → Spec K`, and `main` and `consistent` depend only on `propext`, `Classical.choice` and `Quot.sound`.
- The round-1 checks of the model, the map, the conditions, the spec and `assumption:faithful` stand: their digests are unchanged.

## Not checked

- The proofs beyond their statement and axioms (Lean checks them).
- The library (`tl.*` meanings, `Compiler.Faithful`, Triton's specialisation of `n_elements`), which kernel review trusts. The library is being edited in parallel, and I judged the claim against the recorded gate, not against the edited library.
