import TVA.Launch
import TVA.Device

/-!
# Claims

A *claim* says that a Triton kernel does what its specification says, on every launch it
covers. It is written once, by hand, and then fixed: the claim's author transcribes the kernel's
source into TVA's `tl.*` vocabulary (the *model*, `TVA.Triton`), says which launches are covered
(`Inst`, `ok`, `launch`, `Pre`) and states what the final memory must satisfy (`Post`). A prover
then proves `Claim.Holds`; Lean checks the proof.

```
claims/<kernel>/Claim.lean    def claim : TVA.Claim := …        (fixed, pinned by its sha256)
                              theorem claim_satisfiable : claim.Satisfiable
the prover's proof            theorem main : claim.Holds
```

## What a claim means

`c.Holds`: for every platform `P` that satisfies the device facts (`Device P`), every covered
instance `d` (`c.ok d`) and every covered initial memory `init` (`c.Pre d init`), the model's
launch is `Correct`: it is a valid launch, it is well behaved (no undefined behaviour, no race,
no conflicting accesses within a program, no use of an atomic's result on a shared cell), and
every way it can end is a normal end (no fault, no hang) whose final memory `fin` satisfies
`c.Post d init fin`.

`c.Satisfiable`: some instance and some memory are covered. Without it, a claim with
`ok := fun _ => False` would hold of every kernel. `Device.ref_ok` plays the same role for the
device facts: some platform satisfies them all, so `Device P` is not a contradiction.

## What a reader of a proved claim trusts

1. **Lean and Mathlib**, and that the proof uses only the standard axioms (`propext`,
   `Classical.choice`, `Quot.sound`; the checker verifies this).
2. **The transcription.** The model is the kernel: each line of the model does what the line of
   source it transcribes does, with the meaning `TVA.Triton` gives each `tl.*` operation. The
   claim's author wrote it; each claim folder keeps the source next to it so a reader can compare.
3. **The compiler and the environment.** Triton 3.8.0 compiles the kernel so that every execution
   of a valid, well-behaved launch on the H200 ends in one of the outcomes of the model's launch
   (`TVA.Launch`), and touches no global memory beyond the bytes the launch is given. This is
   trusted for a direct launch of the kernel's own `@triton.jit` function, `kernel[grid](args…)`
   (not a compiled kernel launched by hand, from `warmup` or `triton.compile`), with any
   performance options (`num_warps`, `num_stages`, …: they change speed, not
   outcomes) and every option that changes meaning
   recorded in `Launch` (`enable_fp_fusion`, `num_ctas`, `launch_pdl`); Triton's knobs at their
   defaults, no autotuner or hook, the kernel compiled afresh from its source (a cache that returns
   that same code counts); the H200 driven from an x86-64 host running CPython 3.13 without `-O`,
   in the default floating-point environment; and **nothing else writes the bytes the launch is
   given until it ends** (no other kernel, stream or host write: a launch on a stream not ordered
   after the writes that prepare its inputs is outside every claim). It is trusted only outside
   the exclusions listed in `docs/EXCLUSIONS.md` (known miscompiles, features the library gives
   no meaning, host situations assumed not to happen): a claim's model uses no excluded feature,
   and its README names the exclusions that touch its kernel. A claim says nothing about
   launches Triton rejects before running them (too much shared memory, a block too small for
   `tl.dot`, a grid beyond CUDA's limits): those have no outcome.
4. **The device facts** (`TVA.Device`): the accuracy of the operations Triton leaves to the device
   (`exp`, division, `tl.dot`, fp8 conversion, float atomics, reduction order), each measured on
   the H200 by TVA's own tests in `evidence/`.

Everything else (the proof kit in `TVA.Proof`, the prover's lemmas) is proved, not trusted.
-/
namespace TVA

/-- A claim about one Triton kernel: the launches it covers and what they leave in memory. -/
structure Claim where
  /-- What the host chooses for one launch: shapes, scalar arguments, compile-time constants,
  buffer addresses. -/
  Inst : Type
  /-- The instances covered: the conditions under which the kernel is claimed correct (for
  example `n < 2^31`, a block size that is a power of two, output disjoint from the inputs). -/
  ok : Inst → Prop
  /-- The model: the kernel transcribed into `tl.*`, applied to the instance's arguments. -/
  kernel : Inst → Kernel
  /-- How the host launches it: the grid, the block shapes, the launch options. -/
  launch : Inst → Launch
  /-- The initial memories covered: typically that the buffers are present and aligned. -/
  Pre : Inst → Mem → Prop
  /-- The specification: what the final memory satisfies, given the initial one. -/
  Post : Inst → Mem → Mem → Prop

namespace Claim

/-- **The claim holds**: on every platform the device facts allow, every covered launch of the
model is correct and ends with memory satisfying `Post`. This is what a prover proves
(`theorem main : claim.Holds`). -/
def Holds (c : Claim) : Prop :=
  ∀ P : Platform, Device P → ∀ d, c.ok d → ∀ init, c.Pre d init →
    Correct ((c.launch d).env P) (c.kernel d) (c.launch d) init (c.Post d init)

/-- The claim covers at least one launch: some instance and some initial memory satisfy its
conditions. Every claim file proves this, so that no claim is true only because it covers
nothing. -/
def Satisfiable (c : Claim) : Prop := ∃ d init, c.ok d ∧ c.Pre d init

end Claim

end TVA
