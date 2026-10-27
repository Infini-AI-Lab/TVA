import TVA.Claim
import TVA.Sem.Launch

/-!
# Rule `Compiler.Faithful`: the compiled kernel behaves as its Triton program

Page: `docs/rules/compiler-faithful.md`.

This is the trust the project places in the Triton compiler and its documentation (user,
2026-10-26: "we trust triton compiler and documentations"). If a launch of the kernel's model is
well behaved, so that no interleaving reaches undefined behaviour or a data race between
programs, then every device outcome is an outcome of the model's semantics. A normal end has the
same final memory, the launch never faults, and it hangs only if some interleaving runs forever.

The rule is about a *model*, and the model is the prover's transcription of the source. Whether
the model is the source, with every implicit broadcast, conversion and mask, is a separate
reviewed claim (`MODEL_MISMATCH`). The accuracy of the operations the platform `P` decides is
not part of this rule but of the `Rules.Num` rules about `P`.
-/
namespace TVA.Rules.Compiler

open TVA.Claim

/-- **Rule `Compiler.Faithful`.** The device's behaviour `K` on one instance (launch `L`) is
included in the semantics of the model `k` on platform `P`, for every initial memory from which
that launch is well behaved, when the kernel is compiled without `debug` and the launch is valid
(`Launch.Valid`). The reviewer accepts an instance only if every covered instance also fits the
device's other limits (shared memory, registers), since a launch that fails is a `fault`. -/
def Faithful (P : Platform) (k : Kernel) (L : Launch) (K : Mem → Outcome → Prop) : Prop :=
  L.debug = false → L.Valid → ∀ init, WellBehaved (L.env P) k L init → ∀ o, K init o →
    match o with
    | .ok fin => Outcomes (L.env P) k L init (.ok fin)
    | .fault => False
    | .hang => Outcomes (L.env P) k L init .hang

end TVA.Rules.Compiler
