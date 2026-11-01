# Review of attn_fwd_e5m2, round 5, by r-e5fwd

## Verdict

accept
Library round 26 only added facts and lemmas (the ChunkLoss family, errM, errL) and fixed wording in
num.md. The facts the kernel cites, their listings and constants, and the lemmas the specification
uses are unchanged, so I re-accept every item. The filing notes one wording remark for the
coordinator: the README repeats a library sentence that L127 corrected.

## Findings

None.

## Accepted

All items.

## Checked

- L124 and L127 to L133 in ISSUES.md, the round-26 author notes on this kernel, and the per-item changes in REVIEW.md.
- The diff of docs/rules/num.md and lean/TVA. Every removed line. Every added declaration and command: new definitions and theorems only, with no redefinition, instance, attribute or option.
- num.md's rows for the cited facts and the reviewer checklist's fp8 tl.dot row.

## Not checked

- Whether the new ChunkLoss facts are true, which the kernel does not cite. Kernel review trusts the library.
