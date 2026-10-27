# TVA design (v2)

TVA proves properties of **Triton** kernels (Triton 3.8.0, sources pinned in `primitives/`) in
Lean. v2 was rebuilt on 2026-10-26. v1, with its static translator, is archived
outside this repository, and nothing here depends on it.

## Principles

1. **Trust Triton and its documentation.** User, 2026-10-26: "we trust triton compiler and
   documentations". One rule, `Rules.Compiler.Faithful`, says that the device runs the Triton
   program as its semantics says. We trust nothing below the language: no PTX, no LLVM, no
   memory-model litmus.
2. **No static translator.** Translating the Triton source into a Lean model is the
   **prover's** work, and the model is part of the claim. Reviewers check the model against the
   source and may challenge it (`MODEL_MISMATCH`). User: "'translate triton program' into formal
   model, is also part of work of provers. and this step can be challenged by reviewers."
3. **Prover and reviewers.** A bundle proves
   `∀ K P, Assumptions K P → Spec K`, with the assumptions explicit and named. Lean checks the
   implication. Reviewers judge the specification, the conditions, the model and every
   assumption.
4. **Simple programs.** User: "triton programs are much simpler than cuda programs." A Triton
   program is a sequential block program over tensors. It has no threads, warps, shared memory
   or barriers. User, on `tl.debug_barrier`: "we do not need debug semantics here." So TVA has
   one execution level (programs), one memory (global) and one composition theorem for the
   common case.

## Architecture

```
TVA/Math/        exact float formats, rounding, error bounds
TVA/Lang/        the language a prover writes models in
  Scalar           IntV (wrapping ints), FloatV (IEEE formats), DType
  Tensor           Shape, Idx, Tensor (pure functions), Tree (reduction orders)
  Mem              byte memory, Ptr, aligned typed read/write
  Platform         what the docs leave to the device: approximate math, dot, fp8, float atomics
  Prog             the effect monad: load, store, rmw, pick, ub, loop, env
  Ops              the tl.* API (names as in triton.language)
  BlockPtr, Desc   block pointers and tensor descriptors (TMA)
  Scan, Scaled     scans, sort, histogram; tl.dot_scaled
  Random           Philox (tl.randint, tl.rand), exact
TVA/Sem/         the meaning
  Step             one step of one program
  Launch           programs interleaved over one memory; Launch.Valid; races; WellBehaved;
                   Outcomes
TVA/Claim.lean   Outcome, Behaviour, Meets, Total, Consistent
TVA/Rules/       trusted rules (Compiler.Faithful) and facts about P (Num)
TVA/Lemmas/      proof kit: Safe, per-op rules, Buf, Access, Block, BlockPtr, Desc, Scan,
                 Scaled, Random, Layout, Int, Num, Tree, Sum, Independent, Shared, Claim,
                 Consistency
../kernels/      one folder per proved kernel: Lean bundle, source, write-up, evidence, review
```

**Shallow values, deep effects.** Tensors, integers and floats are ordinary Lean values, and
exact operations are Lean functions. So a proof computes with them using `simp` and arithmetic.
Anything that touches memory, rounds in a way the documentation leaves open, or is
nondeterministic is a `Prog` constructor. The semantics, and the proof rules, handle each
constructor once.

**The platform `P`.** `tl.exp`, `/` on floats, `tl.dot`, fp8 conversions and float
`atomic_add` are relations in a `Platform` structure, and the claim quantifies over `P`. A proof
that needs accuracy cites an explicit rule about `P` (`Rules.Num.*`). Because the library never
guesses these operations, it cannot guess them wrong.

**No vacuous claims.** A bundle proves `Claim.Consistent Assumptions`: some total behaviour and
some platform satisfy its assumptions (`exists_total_faithful`, `Platform.ref`). And
`Compiler.Faithful` holds only for valid launches (`Launch.Valid`: grid limits, fewer than `2^31`
programs, `num_ctas = 1`, and the block shapes the prover lists); the reviewer checks that the
list is complete. A launch that Triton or the driver reports to the host with an error before any
program runs (shared memory, registers, `tl.dot` and descriptor minimum sizes) has no outcome, so
a claim says nothing about whether the kernel compiles (user, 2026-10-27). That is not a silent
gap: the host sees the error. The one silent case found, a grid of `2^31` programs or more that
the launcher skips, is outside `Launch.Valid`.

**Fusion.** With `enable_fp_fusion` (the default), an addition that consumes a product may be
compiled to an `fma`. The model says where:
- `tl.add` is the IEEE sum, for additions no product feeds;
- `tl.mul_add a b c` is either the IEEE `a*b + c` or `fma`;
- `tl.addc` over-approximates any fusion (`AddRel`).

Choosing among them is part of the model, and the reviewer checks the choice. Contraction is the
only float rewrite the backend applies (no fast-math flags, subnormals kept). The
graph-level rewrites of Triton's combine pass are separate operations: `tl.dot_add`,
`tl.dot_scaled_add` and `tl.sum_bcast_mul` ([review-questions.md](review-questions.md) S5,
S7).

## Semantics in one paragraph

Each program runs its `Prog` sequentially, as the documentation presents it. A launch
interleaves the steps of all programs over one sequentially consistent byte memory. A tensor
load or store is one step, and lanes masked off are not accessed. A launch is **well behaved**
when no interleaving reaches undefined behaviour and no two programs race. A race is a pair of
conflicting accesses from different programs, unless both are `acq_rel` atomics at `gpu`/`sys`
scope on the same bytes. Programs do not synchronise with each other: a scalar atomic is
issued by one thread, so its release does not publish the program's other writes. `Faithful`
holds only for well-behaved, valid launches. Then every device outcome is a semantic outcome: a normal end has a
reachable final memory, a fault is impossible, and a hang needs an infinite interleaving. Details
are in [SEMANTICS.md](SEMANTICS.md).

## The proof route

1. **Per program.** `Safe E R W Q m p`: run alone from `m`, program `p` has no undefined
   behaviour, terminates, reads only `R`, writes only `W`, and ends with `Q`. It is proved line
   by line with one rule per `tl.*` operation (`TVA.Lemmas.Ops`, `TVA.Lemmas.Block`).
2. **Compose.** `independent`: if every program is `Safe` and no program writes into another's
   footprint, the launch is well behaved and cannot hang. Its final memory then agrees with each
   program's solo run on that program's footprint, and with the initial memory elsewhere.
3. **Claim.** `meets_independent` combines (2) with `Compiler.Faithful` into `Meets ok Pre Post K`.

`kernels/vec_add_f32/lean/VecAdd.lean` follows this route end to end. Its main theorem depends only
on `propext`, `Classical.choice` and `Quot.sound`.

Kernels whose programs share memory only through `acq_rel` atomics (split-K `atomic_add`,
histograms, global counters whose value is not used to choose what to touch) use the second
composition theorem, `shared` (`TVA.Lemmas.Shared`):
- each program is proved alone with `SafeA`, where an atomic on the shared region may find any
  old value and is recorded in a log;
- the theorem gives the shared region as the replay of all logs, in some interleaving;
- for integer accumulation, `replay_add` and `sum_proj` turn that into "initial value plus
  every program's increments".

Kernels that wait on other programs (locks, spin loops), that hand data from one program to
another behind a flag, or that take work from a counter are not covered
([exclusions.md](exclusions.md)).

## Decisions and why

| Decision | Why |
|---|---|
| A program is sequential; `tl.debug_barrier` has no meaning | The documentation presents programs sequentially; user: "we do not need debug semantics here" |
| Masked-off load lanes hold `other` or an unspecified value | Documentation: "If `other` is `None`, the masked-out value is undefined"; the 3.8.0 lowering gives `0`, which the library does not promise (S6) |
| Reductions combine along a nondeterministic `Tree` | Triton does not document the reduction order |
| `max`/`min` reductions also use a tree | `max(+0.0, -0.0)` may be either zero, so the order can matter |
| Approximate operations go in `Platform` | The documentation does not define them; claims stay honest |
| Fusion is explicit in the model (`add` / `mul_add` / `addc`) | `enable_fp_fusion` is the default; the reviewer can see where it may apply |
| Sequentially consistent memory with a DRF condition | With only `acq_rel` atomics (Triton's default) a race-free launch is SC; relaxed atomics on shared memory count as races |
| No synchronises-with between programs | A program is many threads and a scalar atomic is one thread's, with no barrier before it; release/acquire would otherwise publish writes the device does not (review 2026-10-26, round 2) |
| Claims cover only valid launches, and assumptions must be consistent | A launch outside `Launch.Valid` may be skipped without an error (`2^31` programs or more); a launch rejected with an error has no outcome (user, 2026-10-27); contradictory assumptions make any claim provable (round 2) |
| NaN results may be any NaN; `max(+0,-0)` may be either zero | The device and MLIR folds do not fix them (library review 2026-10-26) |
| Compiler hints are checks, not no-ops | A false `multiple_of`/`max_contiguous`/`assume` lets the compiler misvectorise |
| A non-positive `range` step is undefined behaviour | Not used in practice; listed in [exclusions.md](exclusions.md) |
