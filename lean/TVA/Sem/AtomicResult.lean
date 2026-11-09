import TVA.Sem.Launch

/-!
# Excluded: an atomic's result used on a shared cell (library issue L161)

`Step.defer` lets a pending atomic be passed only by a later step that is the same for every old
value it may return. A tensor `tl.load` or `tl.store` is one step, so a step that uses *any* lane
of an atomic's result holds back every later atomic of the program. On the device each thread
orders only its own accesses: another thread of the program may perform a later atomic first
(library review lr-34a, round 34; `evidence/atomic-result-order`). So the semantics lacks outcomes
of such launches, and `Rules.Compiler.Faithful` would be false of them.

The user's decision (2026-11-08, "let's exclude with Fix B") keeps the model and excludes, as
stated on the source (`docs/exclusions.md`, `docs/SEMANTICS.md` "Atomics"), every launch in which a
program uses an atomic's result (in a store, an address, a branch, or any other operation that
runs) when the atomic's cell is also accessed by another program of the launch, or by another
atomic operation of the same program (another `tl.atomic_*` call, or the same call run again). Two
lanes of one call are not another operation: the semantics already performs them in any order.

`UsesSharedAtomicResult` is the first part of that exclusion read on a model: some interleaving
reaches a point where a program is about to perform a read-modify-write whose continuation is not
the same for every old value (its result is used), and some interleaving has a step of another
program that touches the atomic's bytes. The exclusion as stated on the source governs; the
reviewer checks the source, not this predicate (as for `MemSem.unused`). `TVA.Lemmas.AtomicResult`
proves that lr-34a's counterexample is under it.
-/
namespace TVA

/-- **A program uses an atomic's result on a cell another program also accesses** (library issue
L161; excluded). Some interleaving of the launch reaches a configuration in which program `i` is
about to perform a read-modify-write of the element of dtype `d` at `a` whose continuation `kk`
differs for two old values, and some interleaving has a step of another program `j` that reads or
writes a byte of that element. -/
def UsesSharedAtomicResult (E : Env) (k : Kernel) (L : Launch) (init : Mem) : Prop :=
  ∃ (tr : List (Fin L.size × Event)) (c : Config L.size) (i : Fin L.size) (d : DType) (a : Int)
    (upd : d.Val → d.Val → Prop) (sem : MemSem) (sc : Scope) (kk : d.Val → Prog Unit),
    Reach E (Config.start k L init) tr c ∧ c.progs i = .rmw d a upd sem sc kk ∧
    (∃ v w, kk v ≠ kk w) ∧
    ∃ (tr' : List (Fin L.size × Event)) (c' : Config L.size) (j : Fin L.size) (e : Event),
      Reach E (Config.start k L init) tr' c' ∧ (j, e) ∈ tr' ∧ j ≠ i ∧
        ((e.reads ∪ e.writes) ∩ Mem.span d a).Nonempty

end TVA
