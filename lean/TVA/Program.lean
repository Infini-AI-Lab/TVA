import TVA.Platform
import TVA.Memory

/-!
# Programs

A Triton kernel launch runs one *program instance* per point of the grid (one `tl.program_id`).
TVA describes what one program does as a Lean value of type `Prog α`: a tree whose leaves return
a value of type `α` and whose inner nodes are the program's *effects*. Pure computation (index
arithmetic, `tl.where`, exact integer and float operations) is ordinary Lean on tensors; only the
effects are constructors, so that the semantics (`TVA.Launch`) and the proof kit can see them:

| constructor | Triton |
|---|---|
| `load d s addrs k` | `tl.load`: read the elements at the enabled addresses (`some a`); disabled lanes (`none`) are not accessed |
| `store d s addrs vals k` | `tl.store`: write each enabled lane's value; the lanes land in an unspecified order (it matters only for duplicate addresses) |
| `rmw d a upd solve sem scope k` | one element of a `tl.atomic_*`: read the old value, write a new one related to it by `upd`, continue with the old value. `solve` is for running the model, as for `pick` |
| `pick P solve k` | a value the program cannot predict, constrained by `P`: what a masked-out load returns, a reduction tree, a rounding choice. Undefined behaviour when nothing satisfies `P`. `solve` is for running the model (`Solve`); the semantics ignores it |
| `plat q k` | an operation the platform defines (`tl.exp`, `/`, `tl.dot`, a conversion into fp8): any result `PlatQ.Rel` allows on the environment's platform. Undefined behaviour when there is none |
| `ub` | undefined behaviour (an invalid access, `//` by zero, an over-wide shift) |
| `loop init body k` | a loop whose trip count is decided at run time: run `body` from state `init` until it returns `inr b`, then continue with `k b` |
| `env k` | read the environment `Env`: the platform's relations and `enable_fp_fusion` |

`Prog` is a monad, so a model is written in `do` notation and follows the source line by line.

## What the order of the constructors means

A program is many GPU threads, and they access global memory with no barrier between the
program's operations. So a program is **not sequential for global memory**: a load need not see
a store the same program made before it, and two of its stores to the same bytes may land in
either order. The semantics does not order a program's accesses to the same bytes; instead, a
program **must not access the same bytes twice in a conflicting way**: two accesses of one program
that touch a common byte, at least one of them writing it, make the launch not well behaved
(`TVA.Launch`), across loop iterations included. Atomics are the exception: two atomics of one
program on the same element do not conflict, and they are performed in some order.
`tl.debug_barrier()` orders nothing here.

**Atomics may be performed later than written.** The semantics may perform a pending `rmw` after
later steps of the program, as long as those steps do not depend on the old value it returns.
On the device each thread orders only its own accesses, while here a step that uses one lane's
result is one step for the whole tensor; so a launch in which a program uses an atomic's result
on a cell that another program, or another atomic operation of the same program, also accesses is
not well behaved (`TVA.Launch`), and no claim covers it.

Between programs there is no synchronisation: happens-before is program order only. An atomic's
`sem` and `scope` decide only whether two atomics of different programs on the same bytes race
(`TVA.Launch`).
-/
namespace TVA

/-! ## Running a model

`Step` (`TVA.Launch`) is a relation, so a model cannot simply be run. An executor
(`validation/Exec.lean`) runs one anyway, on a concrete memory, and proves that `Step` can do
whatever it reports. At a `pick P` it needs a value satisfying `P`, and in compiled code `P` and
the value's type are erased: the executor cannot tell a bounds guard from a reduction tree. So
each `pick` carries a `Solve P`, code that proposes a value, asking an oracle (`Hints`) for the
parts it cannot compute, and returns it with its proof (`Ans`). The semantics ignores the
solver: a `pick`'s meaning is `P` alone, so a reader of a model can skip every solver, and since
every answer carries its proof, a wrong solver can only make the executor give up. -/

/-- A certified answer to "some `b` with `P b`". -/
inductive Ans {β : Type} (P : β → Prop) : Type where
  /-- A value satisfying `P`. -/
  | ok (b : β) (h : P b)
  /-- No value satisfies `P`: the step is undefined behaviour. -/
  | empty (h : ∀ b, ¬ P b)

namespace Ans

variable {β : Type} {P : β → Prop}

/-- The value of an answer, if it has one. -/
def val? : Option (Ans P) → Option β
  | some (.ok b _) => some b
  | _ => none

theorem val?_spec : {o : Option (Ans P)} → {b : β} → val? o = some b → P b
  | some (.ok _ h), _, rfl => h

/-- The answer is "no value". -/
def isEmpty : Option (Ans P) → Bool
  | some (.empty _) => true
  | _ => false

theorem isEmpty_spec : {o : Option (Ans P)} → isEmpty o = true → ∀ b, ¬ P b
  | some (.empty h), _ => h
  | some (.ok _ _), h => absurd h (by simp [isEmpty])
  | none, h => absurd h (by simp [isEmpty])

end Ans

/-- What a solver may ask the oracle: the parts of a value it cannot compute. The family is
closed, so that an oracle can match on the question after compilation. -/
inductive Query : Type → Type where
  /-- A reduction or scan tree over `n` leaves. -/
  | tree (n : Nat) : Query Tree
  /-- An order of the lanes of a block of shape `s`. -/
  | order (s : Shape) : Query (List (Idx s))
  /-- An element of dtype `d` (padding, the value of a disabled lane). -/
  | val (d : DType) : Query d.Val
  /-- A choice among alternatives, by number. -/
  | nat : Query Nat

/-- An oracle for `Query`: `none` when it has no answer. -/
abbrev Hints := {τ : Type} → Query τ → Option τ

/-- How to run a `pick P`: `run` returns an answer, or `none` when it has none. -/
structure Solve {β : Type} (P : β → Prop) where
  /-- The answer, given the oracle. -/
  run : Hints → Option (Ans P)

namespace Solve

variable {β : Type} {P : β → Prop}

/-- No solver: an executor gives up at this `pick`. -/
def none : Solve P := ⟨fun _ => Option.none⟩

/-- The value `b`, with its proof. -/
def ok (b : β) (h : P b) : Solve P := ⟨fun _ => some (.ok b h)⟩

/-- Try the value `b`; no answer if it fails. -/
def check (b : β) [Decidable (P b)] : Solve P :=
  ⟨fun _ => if h : P b then some (.ok b h) else Option.none⟩

/-- `b` is the only candidate: if it fails, nothing satisfies `P`. -/
def only (b : β) [Decidable (P b)] (h : ∀ b', P b' → P b) : Solve P :=
  ⟨fun _ => if hb : P b then some (.ok b hb) else some (.empty fun b' hb' => hb (h b' hb'))⟩

/-- A guard `pick (fun _ : Unit => c)`: decide `c`. -/
def guard {c : Prop} [Decidable c] : Solve (fun _ : Unit => c) := only () fun _ h => h

/-- A compiler hint `pick (fun y => y = x ∧ c)`: the value is `x`, if `c` holds. -/
def hint (x : β) {c : Prop} [Decidable c] : Solve (fun y => y = x ∧ c) :=
  ⟨fun _ => if h : c then some (.ok x ⟨rfl, h⟩) else some (.empty fun _ hy => h hy.2)⟩

/-- Ask the oracle for `q`, then solve with its answer. -/
def ask {τ : Type} (q : Query τ) (f : τ → Solve P) : Solve P :=
  ⟨fun H => (H q).bind fun t => (f t).run H⟩

/-- Lane by lane: a tensor whose lane `i` satisfies `R i`, from an answer for each lane. When
every lane has a value, their tensor; when some lane has none, no tensor. -/
def each {s : Shape} {α : Type} {R : Idx s → α → Prop} (f : (i : Idx s) → Option (Ans (R i))) :
    Solve (fun r : Tensor s α => ∀ i, R i (r.get i)) :=
  ⟨fun _ =>
    if hall : ∀ i, (Ans.val? (f i)).isSome then
      some (.ok ⟨fun i => (Ans.val? (f i)).get (hall i)⟩ fun i =>
        Ans.val?_spec (Option.some_get (hall i)).symm)
    else if hex : ∃ i, Ans.isEmpty (f i) = true then
      some (.empty fun r hr => by
        obtain ⟨i, hi⟩ := hex
        exact Ans.isEmpty_spec hi _ (hr i))
    else Option.none⟩

/-- A lanewise partial function `g`: lane `i` of the result is `g i`, and there is no result when
some lane has no value (integer `//` by zero, an oversized shift). -/
def lanes {s : Shape} {α : Type} (g : Idx s → Option α) :
    Solve (fun r : Tensor s α => ∀ i, g i = some (r.get i)) :=
  each fun i => match h : g i with
    | some a => some (.ok a h)
    | Option.none => some (.empty fun _ h' => by rw [h] at h'; cases h')

/-- Lane by lane, with one candidate `c i` per lane: when some lane's candidate fails, no value
satisfies that lane. -/
def eachOnly {s : Shape} {α : Type} {R : Idx s → α → Prop} (c : Idx s → α)
    [∀ i, Decidable (R i (c i))] (h : ∀ i b, R i b → R i (c i)) :
    Solve (fun r : Tensor s α => ∀ i, R i (r.get i)) :=
  each fun i => if hb : R i (c i) then some (.ok (c i) hb)
    else some (.empty fun b hb' => hb (h i b hb'))

/-- Ask the oracle for a value of `β` and take it if it satisfies `P`; otherwise take `b`. -/
def askOr (q : Query β) [DecidablePred P] (b : β) (hb : P b) : Solve P :=
  ⟨fun H => match H q with
    | some t => if ht : P t then some (.ok t ht) else some (.ok b hb)
    | Option.none => some (.ok b hb)⟩

/-- Row by row of a rank-1 tensor, each row's answer computed with the oracle. -/
def each1 {m : Nat} {α : Type} {R : Fin m → α → Prop} (f : Hints → (i : Fin m) → Option (Ans (R i))) :
    Solve (fun r : Tensor [m] α => ∀ i, R i (r.get1 i)) :=
  ⟨fun H =>
    if hall : ∀ i, (Ans.val? (f H i)).isSome then
      some (.ok (Tensor.of1 fun i => (Ans.val? (f H i)).get (hall i)) fun i =>
        Ans.val?_spec (Option.some_get (hall i)).symm)
    else if hex : ∃ i, Ans.isEmpty (f H i) = true then
      some (.empty fun r hr => by
        obtain ⟨i, hi⟩ := hex
        exact Ans.isEmpty_spec hi _ (hr i))
    else Option.none⟩

/-- A solver for `Q` from one for `P`, through `f`; gives up where `P` has no value. -/
def conv {γ : Type} {Q : γ → Prop} (sv : Solve P) (f : β → γ) (hf : ∀ b, P b → Q (f b)) : Solve Q :=
  ⟨fun H => (sv.run H).bind fun
    | .ok b h => some (.ok (f b) (hf b h))
    | .empty _ => Option.none⟩

/-- A solver for a weaker predicate. -/
def mono {Q : β → Prop} (h : ∀ b, P b ↔ Q b) (sv : Solve P) : Solve Q :=
  ⟨fun H => (sv.run H).map fun
    | .ok b hb => .ok b ((h b).mp hb)
    | .empty hn => .empty fun b hb => hn b ((h b).mpr hb)⟩

end Solve

/-- The `sem` argument of a Triton atomic (`tl.atomic_add(…, sem=…)`; default `"acq_rel"`). -/
inductive MemSem where
  /-- `"relaxed"`. -/
  | relaxed
  /-- `"acquire"`. -/
  | acquire
  /-- `"release"`. -/
  | release
  /-- `"acq_rel"`, the default. -/
  | acqRel
  deriving DecidableEq, Repr

/-- The `scope` argument of a Triton atomic (default `"gpu"`). -/
inductive Scope where
  /-- `"cta"`: ordered only with respect to the program's own threads. -/
  | cta
  /-- `"gpu"`, the default. -/
  | gpu
  /-- `"sys"`. -/
  | sys
  deriving DecidableEq, Repr

/-- **The order of an atomic whose result is unused**: `acq_rel` at `gpu` or `sys` scope,
whatever the source's `sem`; the source's `sem` at `cta` scope. The model of a `tl.atomic_*` whose
returned value the source never uses gives its read-modify-writes this order (`TVA.Triton.Memory`),
so that two of them, or one and an `acq_rel` atomic, on the same bytes from different programs do
not race (`TVA.Launch`).

Why this is sound. In TVA's semantics an atomic's `sem` decides nothing but whether a pair of
atomics races; atomics create no synchronisation between programs. What a weaker order
(`relaxed`, `acquire`, `release`) adds on the device is that the atomic may be performed out of
order with its thread's other accesses. *Later* is already allowed: the semantics performs an
atomic after later steps that do not depend on its result. *Earlier* cannot be observed:

* within the program, since its result is unused, no step depends on when it was performed;
* through the atomic's own bytes, since in a well-behaved launch only atomics access them (a plain
  load or store of them would race with the atomic, or conflict within its program), and
  atomics on one cell are performed with single-copy atomicity (PTX ISA §8.10.3; a `.relaxed`
  `atom` or `red` is a strong operation, §8.4), so each update is one step in the cell's order
  of updates, as in an interleaving;
* through other bytes the program accessed, since another program accessing them would race.

At `cta` scope the atomic is not ordered with respect to other programs' accesses at all, so the
source's `sem` is kept and the race rule applies to it unchanged. -/
def MemSem.unused (sem : MemSem) : Scope → MemSem
  | .cta => sem
  | _ => .acqRel

/-- What one Triton program instance does, ending with a value of type `α`. Each constructor but
`ret` is an effect, followed by the rest of the program (`k`). -/
inductive Prog : Type → Type 1 where
  /-- The program ends with `a`. -/
  | ret {α : Type} (a : α) : Prog α
  /-- `tl.load` of elements of dtype `d` in a tensor of shape `s`: lane `i` reads the element
  at byte address `a` when `addrs i = some a`, and is disabled when it is `none`. A disabled
  lane's value is chosen by the model (`other`), not by memory. -/
  | load {α : Type} (d : DType) (s : Shape) (addrs : Idx s → Option Int)
      (k : Tensor s d.Val → Prog α) : Prog α
  /-- `tl.store`: each enabled lane `i` writes `vals.get i` at its address. -/
  | store {α : Type} (d : DType) (s : Shape) (addrs : Idx s → Option Int) (vals : Tensor s d.Val)
      (k : Prog α) : Prog α
  /-- One element of a `tl.atomic_*`: read the element of dtype `d` at `addr`, write a new value
  related to the old one by `upd`, and continue with the old value. `sem` and `scope` are the
  atomic's arguments (`MemSem.unused` when the result is never used). -/
  | rmw {α : Type} (d : DType) (addr : Int) (upd : d.Val → d.Val → Prop)
      (solve : (old : d.Val) → Solve (upd old)) (sem : MemSem) (scope : Scope)
      (k : d.Val → Prog α) : Prog α
  /-- A value of type `β` satisfying `P`, which the program cannot predict. Undefined behaviour
  when no value satisfies `P`. `solve` plays no part in the meaning (`Solve`). -/
  | pick {α β : Type} (P : β → Prop) (solve : Solve P) (k : β → Prog α) : Prog α
  /-- An operation the platform defines, on concrete operands (`PlatQ`): a result `q.Rel` allows
  for the environment's platform. Undefined behaviour when there is none. -/
  | plat {α : Type} (q : PlatQ) (k : q.Out → Prog α) : Prog α
  /-- Undefined behaviour. -/
  | ub {α : Type} : Prog α
  /-- A loop with state `σ`: run `body` from `init`; `inl s` goes round again from `s`, `inr b`
  leaves the loop and continues with `k b`. -/
  | loop {α σ β : Type} (init : σ) (body : σ → Prog (σ ⊕ β)) (k : β → Prog α) : Prog α
  /-- Read the environment (the platform and `enable_fp_fusion`). -/
  | env {α : Type} (k : Env → Prog α) : Prog α

namespace Prog

/-- Sequential composition: run `p`, then `f` on its result. -/
def bind {α β : Type} : Prog α → (α → Prog β) → Prog β
  | ret a, f => f a
  | load d s addrs k, f => load d s addrs (fun v => (k v).bind f)
  | store d s addrs vals k, f => store d s addrs vals (k.bind f)
  | rmw d a upd sv sem sc k, f => rmw d a upd sv sem sc (fun v => (k v).bind f)
  | pick P sv k, f => pick P sv (fun b => (k b).bind f)
  | plat q k, f => plat q (fun r => (k r).bind f)
  | ub, _ => ub
  | loop init body k, f => loop init body (fun b => (k b).bind f)
  | env k, f => env (fun e => (k e).bind f)

instance : Monad Prog where
  pure := ret
  bind := bind

/-- `pure a` in `do` notation is the program `ret a`. -/
@[simp] theorem pure_eq {α : Type} (a : α) : (pure a : Prog α) = ret a := rfl
/-- `p >>= f` in `do` notation is `p.bind f`. -/
@[simp] theorem bind_eq {α β : Type} (p : Prog α) (f : α → Prog β) : p >>= f = p.bind f := rfl

/-- Running `ret a` and then `f` is `f a`. -/
@[simp] theorem ret_bind {α β : Type} (a : α) (f : α → Prog β) : (ret a).bind f = f a := rfl

/-- Running `p` and then returning its result is `p`. -/
@[simp] theorem bind_ret {α : Type} (p : Prog α) : p.bind ret = p := by
  induction p with
  | ret a => rfl
  | load d s addrs k ih => simp only [bind, ih]
  | store d s addrs vals k ih => simp only [bind, ih]
  | rmw d a upd sv sem sc k ih => simp only [bind, ih]
  | pick P sv k ih => simp only [bind, ih]
  | plat q k ih => simp only [bind, ih]
  | ub => rfl
  | loop init body k _ ih => simp only [bind, ih]
  | env k ih => simp only [bind, ih]

/-- Sequential composition of programs is associative. -/
theorem bind_assoc {α β γ : Type} (p : Prog α) (f : α → Prog β) (g : β → Prog γ) :
    (p.bind f).bind g = p.bind (fun a => (f a).bind g) := by
  induction p with
  | ret a => rfl
  | load d s addrs k ih => simp only [bind, ih]
  | store d s addrs vals k ih => simp only [bind, ih]
  | rmw d a upd sv sem sc k ih => simp only [bind, ih]
  | pick P sv k ih => simp only [bind, ih]
  | plat q k ih => simp only [bind, ih]
  | ub => rfl
  | loop init body k _ ih => simp only [bind, ih]
  | env k ih => simp only [bind, ih]

instance : LawfulMonad Prog := LawfulMonad.mk'
  (id_map := fun p => bind_ret p)
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := fun p f g => bind_assoc p f g)

/-- One unfolding of a loop: run the body once from `init`, then go round again or leave. The
semantics steps `loop init body k` to this. -/
def loopStep {α σ β : Type} (init : σ) (body : σ → Prog (σ ⊕ β)) (k : β → Prog α) : Prog α :=
  (body init).bind (fun r => match r with
    | .inl s => loop s body k
    | .inr b => k b)

end Prog

/-- What a running program knows about where it is in the grid: its index along each of the
three grid axes (`tl.program_id(axis)`) and the grid's size along each (`tl.num_programs(axis)`).
A launch gives each program its own `Ctx` (`TVA.Launch`). -/
structure Ctx where
  /-- `tl.program_id(axis)` for `axis = 0, 1, 2`. -/
  pid : Fin 3 → Nat
  /-- `tl.num_programs(axis)` for `axis = 0, 1, 2`. -/
  grid : Fin 3 → Nat

end TVA
