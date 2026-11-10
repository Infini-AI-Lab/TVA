import TVA.Lang.Ops

/-!
# One step of one program

`Step E m p o`: program `p`, running with environment `E` against memory `m`, can take one step
with outcome `o`: either `next m' p' e` (memory `m'`, continuation `p'`, and the event `e`
recording which bytes it read and wrote) or `ub` (undefined behaviour). A program that is
`ret a` has finished and takes no step.

The steps follow the program's text, with one exception: an atomic need not be performed before
the steps after it (`Step.defer`). The threads of a program issue its atomics with no barrier
between them, so the device may perform them in other orders than the text's (library issue L15).
The semantics' order is the program's, not each thread's: a tensor step that uses any lane of an
atomic's result is one step, and it holds back every later atomic of the program, while on the
device another thread may perform a later atomic first. The launches where that shows, in which a
program uses an atomic's result on a cell another program, or another atomic operation of the same
program, also accesses, are excluded (library issue L161, `TVA.Sem.AtomicResult`).

The same relation drives a program alone (`TVA.Safe`) and all programs of a launch
interleaved over one memory (`TVA.Sem.Launch`). This file is the meaning of each `Prog`
constructor; it is reviewed once, with `docs/SEMANTICS.md`.
-/
namespace TVA

open TVA.Math.Bytes

/-- What a step did to memory, as other programs can observe it. -/
structure Event where
  /-- Bytes read. -/
  reads : Set Int := ∅
  /-- Bytes written. -/
  writes : Set Int := ∅
  /-- For an atomic read-modify-write: its `sem` and `scope`. -/
  atomic : Option (MemSem × Scope) := none
  /-- For an atomic read-modify-write: the dtype and address of its element. -/
  cell : Option (DType × Int) := none

namespace Event

/-- A step without memory effect. -/
def silent : Event := {}

/-- Two events conflict when one writes a byte the other reads or writes. -/
def Conflict (e f : Event) : Prop :=
  (e.writes ∩ (f.reads ∪ f.writes)).Nonempty ∨ (e.reads ∩ f.writes).Nonempty

end Event

/-- The outcome of one step. -/
inductive Out (α : Type) where
  | next (m : Mem) (p : Prog α) (e : Event)
  | ub

section Access
variable {d : DType} {s : Shape}

/-- The enabled lanes of an access. -/
def enabled (addrs : Idx s → Option Int) : List (Idx s) :=
  (Idx.all s).filter (fun i => (addrs i).isSome)

/-- The bytes an access of dtype `d` at the enabled addresses touches. -/
def accBytes (d : DType) (addrs : Idx s → Option Int) : Set Int :=
  {x | ∃ i a, addrs i = some a ∧ x ∈ Mem.span d a}

/-- `v` holds, at every enabled lane, the element memory holds at its address. -/
def LoadOk (m : Mem) (d : DType) (addrs : Idx s → Option Int) (v : Tensor s d.Val) : Prop :=
  ∀ i a, addrs i = some a → m.read d a = some (v.get i)

/-- Some enabled lane's access is invalid. -/
def LoadBad (m : Mem) (d : DType) (addrs : Idx s → Option Int) : Prop :=
  ∃ i a, addrs i = some a ∧ m.read d a = none

/-- Every enabled lane's address can be written. -/
def StoreOk (m : Mem) (d : DType) (addrs : Idx s → Option Int) : Prop :=
  ∀ i a, addrs i = some a → m.Writable d a

/-- Write the lanes `order`, in that order (a later lane wins on a duplicate address). -/
def writeAll (m : Mem) (d : DType) (addrs : Idx s → Option Int) (vals : Tensor s d.Val) :
    List (Idx s) → Mem
  | [] => m
  | i :: rest => match addrs i with
    | some a => writeAll (m.write d a (vals.get i)) d addrs vals rest
    | none => writeAll m d addrs vals rest

end Access

/-- One step of a program. -/
inductive Step (E : Env) (m : Mem) : {α : Type} → Prog α → Out α → Prop where
  /-- `tl.load`: each enabled lane reads its element; disabled lanes get any value. -/
  | load {α : Type} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {k : Tensor s d.Val → Prog α} (v : Tensor s d.Val) :
      LoadOk m d addrs v →
      Step E m (.load d s addrs k) (.next m (k v) { reads := accBytes d addrs })
  | loadUB {α : Type} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {k : Tensor s d.Val → Prog α} :
      LoadBad m d addrs → Step E m (.load d s addrs k) .ub
  /-- `tl.store`: the enabled lanes are written in some order. -/
  | store {α : Type} {d : DType} {s : Shape} {addrs : Idx s → Option Int} {vals : Tensor s d.Val}
      {k : Prog α} (order : List (Idx s)) :
      StoreOk m d addrs → order.Perm (enabled addrs) →
      Step E m (.store d s addrs vals k)
        (.next (writeAll m d addrs vals order) k { writes := accBytes d addrs })
  | storeUB {α : Type} {d : DType} {s : Shape} {addrs : Idx s → Option Int} {vals : Tensor s d.Val}
      {k : Prog α} :
      ¬ StoreOk m d addrs → Step E m (.store d s addrs vals k) .ub
  /-- One atomic read-modify-write. -/
  | rmw {α : Type} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope}
      {k : d.Val → Prog α} (old new : d.Val) :
      m.read d a = some old → upd old new →
      Step E m (.rmw d a upd sem sc k)
        (.next (m.write d a new) (k old)
          { reads := Mem.span d a, writes := Mem.span d a, atomic := some (sem, sc),
            cell := some (d, a) })
  | rmwUB {α : Type} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope}
      {k : d.Val → Prog α} :
      (m.read d a = none ∨ ∃ old, m.read d a = some old ∧ ∀ new, ¬ upd old new) →
      Step E m (.rmw d a upd sem sc k) .ub
  /-- An atomic stays pending while the program takes a later step first. The later step must
  not depend on the atomic's result: whatever old value the atomic returns, the continuation can
  take that step, with the same memory and the same event. The atomic is performed afterwards,
  and its old value reflects that order. A step that depends on this lane's result waits for
  this atomic; since a tensor step is one step, it then holds back every later atomic of the
  program (library issues L161, L169: the excluded launches are those in which that order could be
  seen). -/
  | defer {α : Type} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope}
      {k k' : d.Val → Prog α} {m' : Mem} {e : Event} :
      (∀ v, Step E m (k v) (.next m' (k' v) e)) →
      Step E m (.rmw d a upd sem sc k) (.next m' (.rmw d a upd sem sc k') e)
  /-- A `pick`: any value satisfying `P`. The solver `sv` plays no part. -/
  | pick {α β : Type} {P : β → Prop} {sv : Solve P} {k : β → Prog α} (b : β) :
      P b → Step E m (.pick P sv k) (.next m (k b) .silent)
  | pickUB {α β : Type} {P : β → Prop} {sv : Solve P} {k : β → Prog α} :
      (∀ b, ¬ P b) → Step E m (.pick P sv k) .ub
  /-- An operation of the platform: any result its relation allows. -/
  | plat {α : Type} {q : PlatQ} {k : q.Out → Prog α} (r : q.Out) :
      q.Rel E.P r → Step E m (.plat q k) (.next m (k r) .silent)
  | platUB {α : Type} {q : PlatQ} {k : q.Out → Prog α} :
      (∀ r, ¬ q.Rel E.P r) → Step E m (.plat q k) .ub
  | ub {α : Type} : Step E m (.ub : Prog α) .ub
  | loop {α σ β : Type} {init : σ} {body : σ → Prog (σ ⊕ β)} {k : β → Prog α} :
      Step E m (.loop init body k) (.next m (Prog.loopStep init body k) .silent)
  | env {α : Type} {k : Env → Prog α} :
      Step E m (.env k) (.next m (k E) .silent)

end TVA
