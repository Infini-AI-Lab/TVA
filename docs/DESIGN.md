# TVA: design

TVA is a Lean 4 library for proving that Triton kernels do what a fixed claim says. A claim is
written by hand once: it transcribes the kernel into TVA's `tl.*` vocabulary (the *model*), and
states which launches it covers and what the launch leaves in memory. A prover then proves the
claim. Lean checks the proof; nobody reviews it.

There is no review process and no transcription tool: the claims are fixed and checked by hand
once, the proofs by Lean. Every part should be understandable by reading it.

## Principles

1. **Small trusted base, stated in one place.** What a claim's reader must trust is listed in
   `TVA/Claim.lean` and nowhere else: Lean and Mathlib; that the model is the kernel (the claim's
   author transcribed it); that Triton compiles the kernel as the model says (the compiler is
   trusted, outside the exclusions of `docs/EXCLUSIONS.md`); and the device facts in `TVA/Device` (each with TVA's own evidence in `evidence/`).
2. **A claim is about the model.** Because the compiler is trusted, a claim does not mention the
   compiled kernel: it says that the model's launch is well behaved and that every way it can end
   is a normal end whose memory satisfies the specification.
3. **One device.** `Device P` is the conjunction of every fact the library states about the
   platform `P` (the H200, sm_90): accuracy of `exp`, division, `tl.dot`, conversions, float
   atomics, reduction depth. Every claim assumes all of them; `Device.ref_ok` proves that a
   reference platform satisfies them, so no claim is vacuous through its assumptions.
4. **Explainable.** Each file opens with a module docstring that says what it defines, how that
   maps to Triton, and what to read next. Each definition has a docstring that says what it means
   in Triton's terms (with a source reference, e.g. `semantic.py:1516`, when the meaning is
   subtle). Each theorem of the proof kit says when to use it. No history: no issue numbers, no
   dates, no quotes, no "round N".
5. **One way to do each thing.** No duplicate lemmas, no parallel APIs. Names follow Triton
   (`tl.load`, `tl.dot`, `tl.where`) for operations and Mathlib's conventions for lemmas.
6. **Simple statements over clever ones.** `@[simp]` only for rewrites that always help; small
   files (aim for under 600 lines); no `sorry`; only the standard axioms (`propext`,
   `Classical.choice`, `Quot.sound`).

## Layout

```
lean/TVA/
  Base/        values: Bytes (little-endian), Int (`IntV`: fixed-width integers), Float (formats,
               rounding, `FloatV`), DType (Triton's dtypes and their element values)
  Tensor       shapes, indices, `Tensor s α`, elementwise and layout operations
  Memory       `Mem` (a partial map from addresses to bytes), `Ptr`, `Buf` (a contiguous buffer)
  Platform     the relations Triton leaves to the device (`math`, `div`, `dot`, `cvt`,
               `atomicAdd`, `dotScaled`, `sumDepth`) and the environment a program reads
  Program      `Prog α`: a program instance as a Lean value (effects are constructors)
  Triton/      the `tl.*` vocabulary a model is written in, one file per family:
               Core (program ids, arange, full, where, shapes), Arith (integer and float
               arithmetic, comparisons, math functions), Convert (casts and dtype promotion),
               Memory (load, store, atomics, block pointers, descriptors), Reduce (reductions,
               scans, histogram, sort), Dot (dot, dot_scaled), Random (Philox `rand`)
  Launch       a kernel, a launch (grid, block shapes, options), how programs step and
               interleave, races, `WellBehaved`, outcomes
  Device/      the device facts: their shapes (`MathAcc`, `DotAcc`, …), their constants, `Device P`
  Claim        `TVA.Claim` (what a claim is), `Claim.Holds`, `Claim.Satisfiable`, the trusted base
  Proof/       the proof kit (not trusted: everything in it is proved): `Safe` (per-program total
               correctness with footprints) and its rules, composition of programs into
               `Claim.Holds`, memory access patterns (1-D blocks, 2-D tiles), loops, integer
               width, float error, reductions, `tl.dot` accumulation, online softmax, decayed
               recurrences, scans, shared atomics, the reference platform
claims/<kernel>/   one fixed claim per kernel: `Claim.lean`, the kernel's `source/`, `README.md`, and its
                   oracle proof `oracle/` (kept from provers)
sandbox/           the container a prover works in, and the tool that grades a proof
evidence/          TVA's own measurements behind the device facts
docs/              how to read a claim, how to prove one, the library map
```

## The claim

```lean
structure TVA.Claim where
  Inst   : Type                      -- what the host chooses: shapes, scalars, buffers
  ok     : Inst → Prop               -- the instances covered
  kernel : Inst → Kernel             -- the model, applied to the instance's arguments
  launch : Inst → Launch             -- the grid and block shapes the host launches with
  Pre    : Inst → Mem → Prop         -- the initial memories covered
  Post   : Inst → Mem → Mem → Prop   -- what the final memory satisfies

def Claim.Holds (c : Claim) : Prop :=
  ∀ P, Device P → ∀ d, c.ok d → ∀ init, c.Pre d init →
    Correct ((c.launch d).env P) (c.kernel d) (c.launch d) init (c.Post d init)
-- `Correct`: the launch is well behaved from `init` (no undefined behaviour, no race) and every
-- outcome is a normal end (no fault, no hang) whose final memory satisfies the predicate.

def Claim.Satisfiable (c : Claim) : Prop := ∃ d init, c.ok d ∧ c.Pre d init
```

A claim file `claims/<k>/Claim.lean` defines the model, `Inst`, `ok`, `launch`, `Pre`, `Post`,
packs them as `def claim : TVA.Claim`, and proves `claim_satisfiable : claim.Satisfiable` (it
covers some launch). The prover's file proves `theorem main : claim.Holds`. A claim is fixed:
its `Claim.lean` is pinned by its sha256.

## Semantics in one paragraph

A launch runs one program instance per grid point. Each program is a `Prog` value: pure Lean for
values, constructors for effects (`load`, `store`, `rmw` for one atomic element, `pick` for a
value the program cannot predict, `ub`, `loop`, `env`). Programs interleave one step at a time
over a shared memory (sequential consistency). The launch is *well behaved* when no program
reaches undefined behaviour, no two programs make conflicting non-atomic accesses to the same
bytes, a program makes no conflicting accesses to the same bytes itself (its threads do not order
them), and no program uses an atomic's result on a cell another access also reaches. Approximate
operations (`exp`, `tl.dot`, fp8 conversion, float atomics) take their result from the platform's
relation through `pick` (`Prog.plat`, an `env` then a `pick`), so a proof works for every result
the device facts allow. Each `pick` and `rmw` also carries a solver (`Solve`), used only to run a
model (`validation/Exec.lean`); the semantics ignores it.

## Building, checking

- Lean 4 `v4.35.0`, Mathlib `v4.35.0`. Build from `lean/`: `lake build` (wrap in
  `flock /var/tmp/tva-lake.lock` when agents share the machine).
- A claim's proof is checked by `sandbox/tva-check`: the claim file unchanged (sha256), the proof
  builds, `main : claim.Holds` exists with that exact type, and it uses only the standard axioms.
