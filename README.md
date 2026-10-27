# TVA: verified Triton kernels

A Lean library for proving that Triton kernels (Triton 3.8.0) meet their specifications on the
device. It trusts the Triton compiler and its documentation (`Rules.Compiler.Faithful`).

A proof bundle states

```lean
theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K
```

and has four parts:
- a **model** of the kernel, which the prover writes in the `tl.*` vocabulary;
- a **specification**;
- explicit **assumptions**;
- a **proof**, which Lean checks.

Reviewers judge the model against the source, the specification and the assumptions.

- Design: [docs/DESIGN.md](docs/DESIGN.md)
- Semantics: [docs/SEMANTICS.md](docs/SEMANTICS.md)
- Library map: [docs/LIBRARY.md](docs/LIBRARY.md)
- Writing a proof: [docs/PROVER_GUIDE.md](docs/PROVER_GUIDE.md)
- The protocol (who writes and checks what, the gate, findings, acceptance):
  [docs/REVIEW_PROTOCOL.md](docs/REVIEW_PROTOCOL.md), tool `python3 -I tools/tvr.py`, role prompts
  in [docs/prompts/](docs/prompts/)
- Transcribing a kernel: [docs/TRANSCRIPTION.md](docs/TRANSCRIPTION.md)
- Reviewing a kernel (trusting the library): [docs/REVIEWER_GUIDE.md](docs/REVIEWER_GUIDE.md)
- Reviewing the library: [docs/LIBRARY_REVIEW.md](docs/LIBRARY_REVIEW.md), issues in
  [docs/library-review/ISSUES.md](docs/library-review/ISSUES.md), evidence in `evidence/`
- Plan: [docs/plan/PLAN.md](docs/plan/PLAN.md)
- Kernels: [kernels/](kernels/README.md), one folder per kernel with its Lean bundle, source,
  write-up and review. Restarted from scratch on 2026-10-27, beginning with vector addition

Build:

```
cd lean && ~/.elan/bin/lake build
```

`primitives/` holds the pinned Triton sources the semantics cites, and `tutorials/` the Triton
tutorials.
