# Library batch, round 29: author's notes

Issue: L137 ([ISSUES.md](../../ISSUES.md)). The user's request (2026-11-05), after the sandbox test set's
first wave, in which a reviewer of `gdn_gating_bf16` filed the model of `tl.sigmoid` as `UNSUPPORTED`
and the prover referred it: "fix sigmoid". No library review follows unless the user asks for one.
No Lean file changed, so `lake build TVA` and `tvr library-check` were not re-run.

## L137: `tl.sigmoid` is Triton's `1 / (1 + math.exp(-x))`

- **What changed.** `docs/SEMANTICS.md`, after "`tl.cdiv`": a paragraph "`tl.sigmoid(x)`" stating the
  body, its operations by the page's rules and their accuracy rules, and the dtypes the front end accepts
  (`fp32`, `fp64`). The list of inlined `tl.*` functions under "Compiler hints" is unchanged.
- **Why.** The page named `sigmoid` as inlined but stated no body, so a kernel's model of it rested on a
  fact about Triton that only `primitives/` held, and a reviewer could not judge it against the page.
  Two kernels of the sandbox's test set call it on the path proved (`gdn_gating_bf16`'s `β`,
  `act_and_mul_bf16`'s SiLU), and more of the set's linear-attention kernels do.
- **Evidence.** [evidence/sigmoid-round29](../../../../evidence/sigmoid-round29/README.md), on an H200:
  `tl.sigmoid(x)` against `1 / (1 + tl.exp(-x))` written out, bit for bit on all `2^32` `fp32` inputs,
  on `4·10^8` inputs in the method form, on `2^24` `fp64` inputs with edge values, and on compile-time
  values; `bfloat16`, `float16` and `int32` arguments rejected at compile time
  (`Expected dtype ['fp32', 'fp64']`).
- **Constants whose meaning changed.** None: no Lean file changed. The page states what the subset
  already admitted; a model that wrote the body, as the two test-set runs did, is unchanged.
- **Effect on kernels.** SEMANTICS.md is a general fact page (protocol §4), in the digest of every item
  whose cone reaches the library, so the change re-opens those items of the twelve certified kernels;
  none of them calls `tl.sigmoid`, and their reviewers re-accept them in the next kernel round.
