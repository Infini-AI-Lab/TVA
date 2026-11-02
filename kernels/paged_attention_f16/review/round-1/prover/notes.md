# Prover's notes on paged_attention_f16, round 1

## Changes

- `Inst.ok.page_size` narrowed to `PAGE_SIZE ≥ 2` (`∃ b, 1 ≤ b ∧ b ≤ 8 ∧ PAGE_SIZE = 2^b`): at `PAGE_SIZE = 1` line 68 adds two products (`tl.addc`), which the model's `tl.mul_add` does not cover; the test's pages are `16` and `32`. The reason is in the field's docstring, the README and the map entry for line 68; the proofs only read the new field.
- `seq_lens` contiguous stated as a narrowing (Inst docstrings, README Conditions and Narrowings, map entry for line 45).
- The stride-equal-to-1 reason reworded (`paged_attention_kernel`, `model`, map entry 31-41, README): the constant is built in the other operand's dtype, `int64` for `page * stride_kp`, with the same value.
- README: long-context rows in the tolerance table (`4096` and `2^16` pages at `X = 11`), and the kernel recorded against L123.

## Responses

- **F1** REVISED — narrowed to `PAGE_SIZE ≥ 2`, stated with its reason.
- **F2** REFERRED — the `tl.where`/`tl.maximum` reading of lines 64-65 is L123's question.
- **F3** CLARIFIED — the stride-equal-to-1 reason reworded for `page * stride_kp`.
- **F4** DEFERRED — `seq_lens` contiguous, kept and stated as a narrowing.
- **F5** REVISED — as F1.
- **F6** DEFERRED — as F4.
- **F7** CLARIFIED — as F3.
- **F8** REFERRED — L123; the one-element reduction no longer arises with `PAGE_SIZE ≥ 2`.

## Library requests

None.

## Gate

PASS, digest `22c8afda93dec27b`
