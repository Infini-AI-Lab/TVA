import TVA.Lang.Mem
import TVA.Lang.Platform

/-!
# Claims

A bundle claims one theorem about the kernel *as it runs on the device*, an NVIDIA H200 (sm_90),
TVA's only target, driven from an x86-64 host running CPython 3.13 (`docs/exclusions.md`;
`Rules.Compiler.Faithful` covers launches on it, with code Triton compiles for it, its scope 7):

```
theorem main : ∀ (K : TVA.Claim.Behaviour Inst) (P : TVA.Platform), Assumptions K P → Spec K
theorem consistent : TVA.Claim.Consistent Assumptions
theorem satisfiable : TVA.Claim.Satisfiable Inst.ok Pre
```

* `Inst` is the bundle's type of instances (shapes, scalar arguments, buffer addresses).
* `K d init o`: launching the compiled kernel on instance `d` from global memory `init` can end
  with outcome `o`. Nothing about `K` is defined, and the theorem holds for every `K` that
  satisfies the assumptions.
* A launch that is *rejected*, that is, reported to the host with an error before any program
  runs, has no outcome. Examples are too much shared memory, too many threads or registers, a
  `tl.dot` below its minimum size, a dtype the architecture lacks, or a grid dimension above
  CUDA's limits (`evidence/launch-outcomes`). So a claim says nothing about whether a kernel
  compiles (user, 2026-10-27; library issue L3).
* A launch that returns without an error has an outcome, even if no program ran: a grid with a
  zero dimension ends `ok` with memory unchanged. Triton 3.8.0's launcher also returns without
  an error, running no program, when the grid has `2^31` to `2^32` programs. Such a launch is not
  `Launch.Valid`, so `Rules.Compiler.Faithful` says nothing about it, and a claim cannot cover it
  (library issue L9). A call made on a stream that is capturing into a CUDA graph also returns
  without an error and runs no program (its programs run at each replay of the graph), and a call
  made on another stream while a capture is in progress runs at once; the rule's scope excludes
  every call made while a capture is in progress, on any stream, and every replay (library issue
  L91).
* `init` is the global memory when the launch starts, as far as the launch is given it: the bytes
  present in `init` (`Mem` is a partial map), the buffers of `Pre`. The outcome `ok fin` gives the
  same bytes, with their contents when the launch ends, so `Post` speaks only of them. It covers
  one launch: for an autotuned call, the final launch, with the arguments and the memory the last
  hook leaves. Its scope has two parts (the user's decisions, 2026-10-29: "To make the verifier
  easy, we do not verify triton cache.", "we assume the kernel is new." and "So basically, we want
  to verify the kernel itself, rather than triton installation or other code programmed by
  users."). (A) The kernel and its launches, which the kernel reviewer checks: no compile option
  `Launch` does not record, given to the launch, to `warmup` or `triton.compile`, in an autotune
  config or by `@triton.heuristics`; a descriptor scratch, if any, that holds none of its bytes;
  host descriptors that do not round to tf32; through a precompiled kernel, only one compiled from
  the kernel's source, with arguments of its specialisation; and a launch made outside CUDA graph
  capture (no call while a stream capture is in progress, no replay of a captured graph). (B) The
  environment, which TVA assumes and nobody checks: nothing else writes the launch's bytes until
  it ends (not the host, another kernel on any stream or device, or a later kernel that
  programmatic dependent launch lets start early; writes to other bytes do not matter); Triton at
  its defaults (no knob changed, no hook installed, no `kernel.debug` set from outside, no
  `JITFunction.preload`); the H200, from an x86-64 host running CPython 3.13 without `-O` and in
  the default floating-point environment (round to nearest even, no flush to zero); Triton 3.8.0 as
  installed and the kernel's module as written (no code but Triton's own changes Triton's modules,
  classes or objects or the kernel's `JITFunction`s, other than through Triton's API as the
  in-scope tutorials use it, and none binds a name of the kernel's module but the module's own
  statements); and the kernel new: the code the launch runs was compiled for it, from its own
  source, compile-time values and these settings, not code compiled for other values or settings
  and taken from Triton's caches (the `JITFunction`'s cache in the process, or the cache
  directory). `Rules.Compiler.Faithful` states this scope
  (library issues L5, L17, L18, L19, L21, L47, L49, L72, L74, L76, L77, L78, L91, L103, L104,
  L105, L107, L109, L110).
* **Other memory** (library issue L73). `Rules.Compiler.Faithful` also states that a valid,
  well-behaved launch within its scope writes no byte of global memory other than those `init`
  gives, except the scratch of the descriptors the kernel builds (its scope 5). A claim's `main`
  can only be proved when the launch is valid and well behaved on every covered instance and memory
  (the rule says nothing of the others, and no other assumption constrains `K`, so a behaviour that
  faults there would satisfy the assumptions). So a claim that cites the rule also implies, through
  the rule's text, that on every covered instance and memory, within the rule's scope, the launch
  leaves every other byte as it was, the descriptor scratch aside. `Post` cannot
  state this frame: it rests on the rule's text, not on the outcome.
* `P` is the platform (`TVA.Platform`): the device's meaning of the operations Triton leaves
  open (approximate math, `tl.dot`, fp8 conversions, float atomics).
* `Assumptions K P` is a structure of named propositions the prover writes, each with a
  `[kind]` docstring: typically `Rules.Compiler.Faithful` for the model, plus accuracy rules
  about `P`. The reviewer judges each one.
* `Spec K := Meets ok Pre Post K`: on every instance with `ok` and memory with `Pre`, every
  outcome is a normal end whose final memory satisfies `Post`. Faults and hangs violate it.

A bundle also proves `Consistent Assumptions`, that some total behaviour and some platform satisfy
the assumptions, and `Satisfiable Inst.ok Pre`, that some instance and some memory satisfy the
conditions (library issue L71). Together they make the claim not vacuous: with the total witness,
the covered instance has an outcome, which `main` makes a normal end satisfying `Post`.
`TVA.exists_total_faithful` gives the behaviour for `Compiler.Faithful`; each rule about `P` comes
with a satisfiability lemma. Whether the conditions cover the kernel's real use is the reviewer's
judgment.
-/
namespace TVA.Claim

/-- How one launch on the device can end. -/
inductive Outcome where
  /-- Normal completion with final global memory `fin` (the bytes `init` gives the launch, with
  their contents at the end): the launch returned without an error, and every program ran to its
  end. -/
  | ok (fin : Mem)
  /-- A fault of a launch that runs (illegal address, trap, …). -/
  | fault
  /-- The launch never completes. -/
  | hang

/-- The compiled kernel as it runs on the device (the H200): `K d init o` says that a launch on
instance `d` from global memory `init` can end with `o`. A launch reported to the host with an error before
any program runs (resources, unsupported sizes or dtypes, grid limits, a missing allocator) has no
outcome. A launch that returns without an error has one, even if no program ran. `init` and the
final memory of `ok` give the bytes the launch is given. The rules relate `K` to the model only for
a launch whose bytes nothing else writes until it ends, through Triton at its defaults, as installed
and unmodified, outside CUDA graph capture, running the code compiled for it
(`Rules.Compiler.Faithful`, "Scope"). -/
abbrev Behaviour (Inst : Type) : Type := Inst → Mem → Outcome → Prop

/-- `K` meets the specification `(ok, Pre, Post)`. -/
def Meets {Inst : Type} (ok : Inst → Prop) (Pre : Inst → Mem → Prop)
    (Post : Inst → Mem → Mem → Prop) (K : Behaviour Inst) : Prop :=
  ∀ d, ok d → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin

/-- Every instance and initial memory has some outcome: no launch is rejected. A consistency
witness for assumptions must be total, because the empty behaviour (every launch rejected)
satisfies every `Meets`. A total witness shows that the assumptions allow launches that run. The
device itself may reject launches (`Behaviour`); `Total` is asked of the witness only. -/
def Total {Inst : Type} (K : Behaviour Inst) : Prop := ∀ d init, ∃ o, K d init o

/-- The assumptions `A` are consistent: a total behaviour and a platform satisfy them. -/
def Consistent {Inst : Type} (A : Behaviour Inst → Platform → Prop) : Prop :=
  ∃ K P, Total K ∧ A K P

/-- The conditions `(ok, Pre)` cover some launch: an instance with `ok` and a memory with `Pre`
(library issue L71). Without it, `Meets ok Pre Post K` holds of every `K`, and a claim with
`ok := fun _ => False` would pass every other check. -/
def Satisfiable {Inst : Type} (ok : Inst → Prop) (Pre : Inst → Mem → Prop) : Prop :=
  ∃ d init, ok d ∧ Pre d init

end TVA.Claim
