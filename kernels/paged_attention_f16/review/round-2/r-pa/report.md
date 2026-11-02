# Review of paged_attention_f16, round 2, by r-pa

## Verdict

accept
The prover settled each of my round-1 findings. F5: `Inst.ok.page_size` now requires `PAGE_SIZE >= 2`, so line 68's `tl.mul_add` is the library's form on every covered instance; the narrowing is stated with its reason. F6: `seq_lens`'s contiguity is stated as a narrowing. F7: the reason for a stride equal to `1` is reworded correctly. F8 is referred to L123, and its one-element part no longer arises. The model body is unchanged, and so are `Post` and the tolerances. Round 1's checks still hold, so I accept every item at gate `22c8afda93dec27b`.

## Findings

- **release F5** — `PAGE_SIZE >= 2` (`1 <= b <= 8`), stated in the field's docstring, the README and the map entry for line 68; every reduction of line 68 now ends in an addition.
- **release F6** — `seq_lens` contiguous is stated as a narrowing in `Inst`, the field's docstring, the README and the map entry for line 45.
- **release F7** — the docstrings and the map entry for lines 31–41 now say the constant `1` is built in the other operand's dtype (`int64` for `page * stride_kp`).
- **release F8** — referred to L123 (open); the one-element case no longer arises.

## Accepted

All items.

## Checked

- REVIEW.md: the prover's responses to F1–F8, and the constants changed since my acceptance (`Inst.ok.mk`, `Inst.ok.page_size`).
- `Model.lean` in full again. The model body (lines 33–105) is the same as in round 1. The changes are `paged_attention_kernel`'s, `model`'s and `Inst`'s docstrings, the `seq_lens` field's docstring, and `Inst.ok.page_size` (`∃ b, 1 ≤ b ∧ b ≤ 8 ∧ PAGE_SIZE = 2^b`) with its docstring. `SeqOk`, `Pre`, `Post`, `SeqVals`, the tolerances and `Assumptions` are as in round 1.
- The other Lean files: they read the new field (`PagedAttention.lean` line 44, `Facts.lean` lines 191 and 196), and have no `instance`, `notation`, `attribute`, `export`, `set_option` or renaming. `satisfiable` uses `PAGE_SIZE = 16`.
- The map entries for lines 31–41, 45 and 68, and the README's Conditions (lines 94–99, 108) and Narrowings and modelling decisions (lines 211–237). They match the Lean.
- Real use is still covered: the test's pages are 16 and 32. Round 1's work files (`round-1/r-pa/work/tol.out`, `test_data.out`) are all at those sizes, and `Post` and the tolerances are unchanged, so their figures still apply.
- L123 is open in `docs/library-review/ISSUES.md`, and the README records this kernel against it.

## Not checked

- No new sampling or Lean runs this round: the model, `Post` and the tolerances are unchanged, and I relied on round 1's work files.
- As in round 1: the proofs' inner bounds (Lean's), the GPU figures, and the library's facts.
