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
| `rmw d a upd sem scope k` | one element of a `tl.atomic_*`: read the old value, write a new one related to it by `upd`, continue with the old value |
| `pick P k` | a value the program cannot predict, constrained by `P`: what a masked-out load returns, a platform result (`tl.exp`, `tl.dot`), a reduction tree. Undefined behaviour when nothing satisfies `P` |
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
  | rmw {α : Type} (d : DType) (addr : Int) (upd : d.Val → d.Val → Prop) (sem : MemSem)
      (scope : Scope) (k : d.Val → Prog α) : Prog α
  /-- A value of type `β` satisfying `P`, which the program cannot predict. Undefined behaviour
  when no value satisfies `P`. -/
  | pick {α β : Type} (P : β → Prop) (k : β → Prog α) : Prog α
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
  | rmw d a upd sem sc k, f => rmw d a upd sem sc (fun v => (k v).bind f)
  | pick P k, f => pick P (fun b => (k b).bind f)
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
  | rmw d a upd sem sc k ih => simp only [bind, ih]
  | pick P k ih => simp only [bind, ih]
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
  | rmw d a upd sem sc k ih => simp only [bind, ih]
  | pick P k ih => simp only [bind, ih]
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
