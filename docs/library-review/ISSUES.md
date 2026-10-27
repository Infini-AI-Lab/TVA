# Library issues

The library's open questions and defects, from kernel review (`LIBRARY_REPORT`s and `REFERRED`
findings, collected in [REPORTS.md](REPORTS.md)), from library review, and from the author.
The process is in [../LIBRARY_REVIEW.md](../LIBRARY_REVIEW.md); kernel findings are cited as
`<kernel>#F<n>` (each kernel's `REVIEW.md`).

| Issue | State | Source | Summary |
|---|---|---|---|
| [L1](#l1) | OPEN | grouped_gemm_f16#F1, #F2 | A hint inside a loop body on a loop-carried name is not a no-op |
| [L2](#l2) | OPEN | grouped_gemm_f16#F2 | What a hint on a rank ≥ 2 tensor promises, in terms a kernel can check |
| [L3](#l3) | OPEN | vec_add_f32#F1, grouped_gemm_f16#F3, flash_attn_fwd_f16#F4 | A launch rejected before it runs is outside the claim (user, 2026-10-27) |
| [L4](#l4) | OPEN | flash_attn_fwd_f16#F1, #F2, #F7, grouped_gemm_f16#F6 | Device facts with constants, owned by the library, on TVA's own evidence |
| [L5](#l5) | OPEN | flash_attn_fwd_f16#F5 | A descriptor built inside the kernel writes launch scratch memory |
| [L6](#l6) | OPEN | flash_attn_fwd_f16#F8 | A drop reason for `tl.static_assert` |

## L1

**A hint inside a loop body on a loop-carried name is not a no-op.**

- **Problem.** The library said such a hint is a no-op, and tutorial 08 relied on that. But to
  find the loop-carried names, Triton's front end first visits a `for` or `while` body once with
  every name still bound to its pre-loop value. So the hint marks the pre-loop value, and the
  compiled loads then assume 16-byte alignment. With `lda = 36` the kernel faults.
- **Evidence.** [evidence/hints-dry-run](../../evidence/hints-dry-run/README.md).
- **To do.** State the meaning (a check on the pre-loop value at the loop's entry, plus no effect
  on the block argument) and the transcription rule. Cover the induction variable, and loops
  inside loops.
- **Affects.** The hint operations' documentation, [TRANSCRIPTION.md](../TRANSCRIPTION.md) §8,
  and the protocol's drop reasons (`no-op:block-arg-hint` is withdrawn).

## L2

**What a hint on a rank ≥ 2 tensor promises.**

- **Problem.** `tl.multiple_of(a_ptrs, [16, 16])` on a 2-D pointer tensor, read per element,
  would claim every element is 16-byte aligned, which is false for tutorial 08. Triton 3.8.0
  uses the value along the fastest-varying dimension only.
- **Problem.** The `contig` argument of `multiple_of` is defined by "Triton's contiguity", which
  a kernel reviewer cannot check.
- **To do.** Define both in terms of the tensor's values.

## L3

**A launch rejected before it runs is outside the claim** (user decision, 2026-10-27).

- **Problem.** Too much shared memory, too many registers, or a `tl.dot` below its minimum
  size: Triton reports these on the host before any program runs. The rule page asked kernel
  reviewers to check that every covered instance compiles.
- **To do.** State in `TVA.Claim` and `Rules.Compiler.Faithful` (docstrings and the rule page)
  that such a launch has no outcome. Remove the compile-log check from the rule page.

## L4

**Device facts with constants, owned by the library.**

- **Problem.** Kernels chose their own accuracy constants, and justified them with another
  project's measurements. Under the protocol, device facts are the library's, on TVA's own
  evidence (Triton 3.8.0, the H200), and kernels cite them by name.
- **To do,** for sm_90 with Triton 3.8.0:
  - `tl.math.exp2` on `float32`: the lowering (libdevice, with flush-to-zero), the constants,
    and `exp2(-∞)`;
  - `tl.math.log2` on `float32`: the lowering (libdevice), the constants;
  - `/` on `float32` (`DivFull`): TVA's own measurement;
  - `tl.dot` with `float16` and `bfloat16` operands into `float32`: the constants;
  - each as a named fact with a page and evidence.

## L5

**A descriptor built inside the kernel writes launch scratch memory.**

- **Problem.** With `tl.make_tensor_descriptor`, each program writes its descriptors into
  memory the launcher allocates. The semantics has no such memory, so "every other byte is
  unchanged" would be false.
- **To do.** Model launch scratch, or exclude in-kernel descriptors (with host-built
  descriptors still covered).

## L6

**A drop reason for `tl.static_assert`.**

- **Problem.** None of the drop reasons fit a compile-time assertion.
- **Done in the protocol.** The drop reason `compile-check`, whose note names the condition of
  `Inst.ok` that makes the assertion hold. The tool and [TRANSCRIPTION.md](../TRANSCRIPTION.md)
  §10 still need it.
