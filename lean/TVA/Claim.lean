import TVA.Lang.Mem
import TVA.Lang.Platform

/-!
# Claims

A bundle claims one theorem about the kernel *as it runs on the device*:

```
theorem main : ∀ (K : TVA.Claim.Behaviour Inst) (P : TVA.Platform), Assumptions K P → Spec K
```

* `Inst` is the bundle's type of instances (shapes, scalar arguments, buffer addresses).
* `K d init o`: launching the compiled kernel on instance `d` from global memory `init` can end
  with outcome `o`. Nothing about `K` is defined, and the theorem holds for every `K` that
  satisfies the assumptions.
* `P` is the platform (`TVA.Platform`): the device's meaning of the operations Triton leaves
  open (approximate math, `tl.dot`, fp8 conversions, float atomics).
* `Assumptions K P` is a structure of named propositions the prover writes, each with a
  `[kind]` docstring: typically `Rules.Compiler.Faithful` for the model, plus accuracy rules
  about `P`. The reviewer judges each one.
* `Spec K := Meets ok Pre Post K`: on every instance with `ok` and memory with `Pre`, every
  outcome is a normal end whose final memory satisfies `Post`. Faults and hangs violate it.

A bundle also proves `Consistent Assumptions`: some total behaviour and some platform satisfy
the assumptions, so the claim is not vacuous. `TVA.exists_total_faithful` gives the behaviour
for `Compiler.Faithful`; each rule about `P` comes with a satisfiability lemma.
-/
namespace TVA.Claim

/-- How one launch on the device can end. -/
inductive Outcome where
  /-- Normal completion with final global memory `fin`. -/
  | ok (fin : Mem)
  /-- A fault (illegal address, trap, launch failure, …). -/
  | fault
  /-- The launch never completes. -/
  | hang

/-- The compiled kernel as it runs on the device. -/
abbrev Behaviour (Inst : Type) : Type := Inst → Mem → Outcome → Prop

/-- `K` meets the specification `(ok, Pre, Post)`. -/
def Meets {Inst : Type} (ok : Inst → Prop) (Pre : Inst → Mem → Prop)
    (Post : Inst → Mem → Mem → Prop) (K : Behaviour Inst) : Prop :=
  ∀ d, ok d → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin

/-- Every launch has some outcome, as on a real device. A consistency witness for assumptions
must be total: the empty behaviour satisfies every `Meets`. -/
def Total {Inst : Type} (K : Behaviour Inst) : Prop := ∀ d init, ∃ o, K d init o

/-- The assumptions `A` are consistent: a total behaviour and a platform satisfy them. -/
def Consistent {Inst : Type} (A : Behaviour Inst → Platform → Prop) : Prop :=
  ∃ K P, Total K ∧ A K P

end TVA.Claim
