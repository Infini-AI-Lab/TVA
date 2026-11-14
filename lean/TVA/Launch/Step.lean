import TVA.Core

/-!
# One step of one program

`Step E m p o`: the program `p`, running in environment `E` against global memory `m`, can take
one step with outcome `o`. The outcome is either `next m' p' e` (the memory after the step, the
rest of the program, and the `Event` that records which bytes the step read and wrote) or `ub`
(undefined behaviour). A program that is `ret a` has finished and takes no step.

Each step does the program's next effect (`TVA.Program`) as one indivisible action:

* `load`: every enabled lane reads its element; a disabled lane's value is whatever the model
  chose. An enabled lane whose access is invalid (`Mem.read` is `none`: a byte the launch was not
  given, or a misaligned address) is undefined behaviour: on the device it may fault or return
  garbage.
* `store`: every enabled lane writes its element, the lanes in some order (which matters only when
  two lanes have the same address: the later one wins). An invalid address is undefined behaviour.
* `rmw`: one element of an atomic, read and written in one step; it continues with the old value.
* `pick`: any value satisfying the predicate, or undefined behaviour when there is none. The
  `pick`'s solver plays no part.
* `plat`: any result the platform's relation allows (`PlatQ.Rel`), or undefined behaviour when
  there is none.
* `loop`: one unfolding (`Prog.loopStep`); `env`: read the environment.

The steps follow the program's text, with one exception: an atomic need not be performed before
the steps after it (`Step.defer`). The threads of a program issue its atomics with no barrier
between them, so the device may perform a program's atomics in an order other than the text's.
`Step.defer` lets a pending `rmw` be passed by any later step that does not depend on the old
value it returns, so the semantics performs a program's atomics in every order its own
dependences allow. Those dependences are coarser than the device's (a tensor step that uses one
lane's result waits for that lane's atomic and, being one step, holds back every later step);
`TVA.Launch.WellBehaved` excludes the launches in which that shows (`UsesSharedAtomicResult`).

The same relation drives one program alone (the proof kit's per-program reasoning) and all the
programs of a launch interleaved over one memory (`TVA.Launch.Run`).
-/
namespace TVA

/-- What one step did to global memory, as another access can observe it. -/
structure Event where
  /-- The bytes the step read. -/
  reads : Set Int := ∅
  /-- The bytes the step wrote. -/
  writes : Set Int := ∅
  /-- For one element of an atomic: its `sem` and `scope`. `none` for every other step. -/
  atomic : Option (MemSem × Scope) := none
  /-- For one element of an atomic: the element's dtype and byte address. -/
  cell : Option (DType × Int) := none

namespace Event

/-- A step that touches no memory (`pick`, `plat`, `loop`, `env`). -/
def silent : Event := {}

/-- Two steps conflict when one writes a byte the other reads or writes. Conflicting steps of
different programs race, and conflicting steps of one program are a conflict within the program
(`TVA.Launch.WellBehaved`), unless both are suitable atomics. -/
def Conflict (e f : Event) : Prop :=
  (e.writes ∩ (f.reads ∪ f.writes)).Nonempty ∨ (e.reads ∩ f.writes).Nonempty

/-- The step reads or writes some byte of `S`. -/
def Touches (e : Event) (S : Set Int) : Prop := ((e.reads ∪ e.writes) ∩ S).Nonempty

end Event

/-- The outcome of one step of a program whose final value has type `α`. -/
inductive Out (α : Type) where
  /-- The step happened: memory is now `m`, the program continues as `p`, and `e` records what
  the step read and wrote. -/
  | next (m : Mem) (p : Prog α) (e : Event)
  /-- The step is undefined behaviour. -/
  | ub

section Access
variable {d : DType} {s : Shape}

/-- The enabled lanes of a `tl.load` or `tl.store` (those whose address is `some a`; the others
are masked off), in index order. -/
def enabled (addrs : Idx s → Option Int) : List (Idx s) :=
  (Idx.all s).filter (fun i => (addrs i).isSome)

/-- The bytes a `tl.load` or `tl.store` of elements of dtype `d` touches: every byte of the
element at each enabled lane's address. -/
def accBytes (d : DType) (addrs : Idx s → Option Int) : Set Int :=
  {x | ∃ i a, addrs i = some a ∧ x ∈ Mem.span d a}

/-- `v` is a result a `tl.load` may give: at every enabled lane it holds the element memory holds
at the lane's address (disabled lanes are unconstrained here; the model chooses their value). -/
def LoadOk (m : Mem) (d : DType) (addrs : Idx s → Option Int) (v : Tensor s d.Val) : Prop :=
  ∀ i a, addrs i = some a → m.read d a = some (v.get i)

/-- Some enabled lane of a `tl.load` reads an invalid address (a byte the launch was not given, or
a misaligned address): the load is undefined behaviour. -/
def LoadBad (m : Mem) (d : DType) (addrs : Idx s → Option Int) : Prop :=
  ∃ i a, addrs i = some a ∧ m.read d a = none

/-- Every enabled lane of a `tl.store` can write its element. -/
def StoreOk (m : Mem) (d : DType) (addrs : Idx s → Option Int) : Prop :=
  ∀ i a, addrs i = some a → m.Writable d a

/-- Memory after a `tl.store` writes the lanes of `order`, in that order (on a duplicate address
the later lane wins). Disabled lanes in `order` write nothing. -/
def writeAll (m : Mem) (d : DType) (addrs : Idx s → Option Int) (vals : Tensor s d.Val) :
    List (Idx s) → Mem
  | [] => m
  | i :: rest => match addrs i with
    | some a => writeAll (m.write d a (vals.get i)) d addrs vals rest
    | none => writeAll m d addrs vals rest

end Access

/-- One step of a program: `Step E m p o` when `p`, in environment `E` against memory `m`, can take
a step with outcome `o`. -/
inductive Step (E : Env) (m : Mem) : {α : Type} → Prog α → Out α → Prop where
  /-- `tl.load`: every enabled lane reads its element at once; disabled lanes get any value. -/
  | load {α : Type} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {k : Tensor s d.Val → Prog α} (v : Tensor s d.Val) :
      LoadOk m d addrs v →
      Step E m (.load d s addrs k) (.next m (k v) { reads := accBytes d addrs })
  /-- A `tl.load` with an invalid enabled address is undefined behaviour. -/
  | loadUB {α : Type} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {k : Tensor s d.Val → Prog α} :
      LoadBad m d addrs → Step E m (.load d s addrs k) .ub
  /-- `tl.store`: the enabled lanes are written at once, in some order `order` (any permutation
  of the enabled lanes). -/
  | store {α : Type} {d : DType} {s : Shape} {addrs : Idx s → Option Int} {vals : Tensor s d.Val}
      {k : Prog α} (order : List (Idx s)) :
      StoreOk m d addrs → order.Perm (enabled addrs) →
      Step E m (.store d s addrs vals k)
        (.next (writeAll m d addrs vals order) k { writes := accBytes d addrs })
  /-- A `tl.store` with an invalid enabled address is undefined behaviour. -/
  | storeUB {α : Type} {d : DType} {s : Shape} {addrs : Idx s → Option Int} {vals : Tensor s d.Val}
      {k : Prog α} :
      ¬ StoreOk m d addrs → Step E m (.store d s addrs vals k) .ub
  /-- One atomic read-modify-write: read the old value, write any new value `upd` allows, and
  continue with the old value. -/
  | rmw {α : Type} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop} {sv : (old : d.Val) → Solve (upd old)}
      {sem : MemSem} {sc : Scope}
      {k : d.Val → Prog α} (old new : d.Val) :
      m.read d a = some old → upd old new →
      Step E m (.rmw d a upd sv sem sc k)
        (.next (m.write d a new) (k old)
          { reads := Mem.span d a, writes := Mem.span d a, atomic := some (sem, sc),
            cell := some (d, a) })
  /-- An atomic on an invalid address, or whose update allows no new value, is undefined
  behaviour. -/
  | rmwUB {α : Type} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop} {sv : (old : d.Val) → Solve (upd old)}
      {sem : MemSem} {sc : Scope}
      {k : d.Val → Prog α} :
      (m.read d a = none ∨ ∃ old, m.read d a = some old ∧ ∀ new, ¬ upd old new) →
      Step E m (.rmw d a upd sv sem sc k) .ub
  /-- **An atomic is performed later.** The atomic stays pending while the program takes a later
  step first. That step must not depend on the atomic's result: whatever old value the atomic
  returns, the continuation takes that step, to the same memory with the same event. The atomic
  is performed afterwards, and the old value it returns reflects that order. A later step that
  depends on the result (for any lane) cannot pass it, and holds back everything after it.

  A step taken before the atomic cannot know its result, so when the continuation does not depend
  on the result at all, neither does the continuation after the step (a `pick` makes the same
  choice for every old value). This changes no outcome, and it keeps an unused result unused in
  every reachable state, which `UsesSharedAtomicResult` reads. -/
  | defer {α : Type} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop} {sv : (old : d.Val) → Solve (upd old)}
      {sem : MemSem} {sc : Scope}
      {k k' : d.Val → Prog α} {m' : Mem} {e : Event} :
      (∀ v, Step E m (k v) (.next m' (k' v) e)) →
      ((∀ v w, k v = k w) → ∀ v w, k' v = k' w) →
      Step E m (.rmw d a upd sv sem sc k) (.next m' (.rmw d a upd sv sem sc k') e)
  /-- `pick`: continue with any value satisfying the predicate. -/
  | pick {α β : Type} {P : β → Prop} {sv : Solve P} {k : β → Prog α} (b : β) :
      P b → Step E m (.pick P sv k) (.next m (k b) .silent)
  /-- A `pick` that nothing satisfies is undefined behaviour. -/
  | pickUB {α β : Type} {P : β → Prop} {sv : Solve P} {k : β → Prog α} :
      (∀ b, ¬ P b) → Step E m (.pick P sv k) .ub
  /-- An operation of the platform: continue with any result its relation allows. -/
  | plat {α : Type} {q : PlatQ} {k : q.Out → Prog α} (r : q.Out) :
      q.Rel E.P r → Step E m (.plat q k) (.next m (k r) .silent)
  /-- An operation of the platform that allows no result is undefined behaviour. -/
  | platUB {α : Type} {q : PlatQ} {k : q.Out → Prog α} :
      (∀ r, ¬ q.Rel E.P r) → Step E m (.plat q k) .ub
  /-- `ub` is undefined behaviour. -/
  | ub {α : Type} : Step E m (.ub : Prog α) .ub
  /-- A loop unfolds one iteration (`Prog.loopStep`). -/
  | loop {α σ β : Type} {init : σ} {body : σ → Prog (σ ⊕ β)} {k : β → Prog α} :
      Step E m (.loop init body k) (.next m (Prog.loopStep init body k) .silent)
  /-- `env` reads the environment. -/
  | env {α : Type} {k : Env → Prog α} :
      Step E m (.env k) (.next m (k E) .silent)

end TVA
