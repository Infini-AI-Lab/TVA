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
- Transcribing a kernel: [docs/TRANSCRIPTION.md](docs/TRANSCRIPTION.md) (guidance; the facts are in
  [docs/SEMANTICS.md](docs/SEMANTICS.md))
- Reviewing a kernel (trusting the library): [docs/REVIEWER_GUIDE.md](docs/REVIEWER_GUIDE.md)
- Reviewing the library: [docs/LIBRARY_REVIEW.md](docs/LIBRARY_REVIEW.md), issues in
  [docs/library-review/ISSUES.md](docs/library-review/ISSUES.md), evidence in `evidence/`
- Plan: [docs/plan/PLAN.md](docs/plan/PLAN.md)
- Kernels: `kernels/` is local, and TVA does not ship it (user decision 2026-11-10): one folder per
  kernel with its Lean bundle, source, write-up and review, made by `tvr new`; the release check reads
  none of it. The fixed claims of the test kernels ship in [claims/](claims/README.md), and the
  release check elaborates each against the library (library issue L190)
- Agents in containers: [sandbox/](sandbox/README.md) runs the protocol on one kernel with a prover
  agent, reviewer agents and a review service, each in a container (a Harbor task), with a harness
  that helps weaker models ([sandbox/HARNESS.md](sandbox/HARNESS.md))

Build (the library, `TVA`; a local kernel builds with `lake -f .lake/tvr/lakefile.toml build <Module>`,
the configuration `tvr` writes):

```
cd lean && ~/.elan/bin/lake build
```

`primitives/` holds the pinned Triton sources the semantics cites, and `tutorials/` the Triton
tutorials.
