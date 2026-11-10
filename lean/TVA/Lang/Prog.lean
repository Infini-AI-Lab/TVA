import TVA.Lang.Platform

/-!
# Programs

What one Triton program instance (one `pid`) does, as a Lean value of type `Prog α`. Pure
computation is ordinary Lean (`TVA.Lang.Tensor`). The *effects* are constructors, so that the
semantics (`TVA.Sem`) and the lemmas can see them:

| constructor | Triton |
|---|---|
| `load d s addrs k` | `tl.load`: read the elements at the enabled addresses (`some a`); disabled lanes (`none`) are not accessed and get unspecified values |
| `store d s addrs vals k` | `tl.store`: write each enabled lane's value, in an unspecified order (it matters only for duplicate addresses) |
| `rmw d a upd sem scope k` | one element of `tl.atomic_*`: read the old value, write a new one related to it by `upd`, return the old value; the step may happen after later steps that do not depend on the old value (`Step.defer`). An atomic whose result the source never uses has `sem` `acq_rel` at `gpu`/`sys` scope (`MemSem.unused`, library issue L157) |
| `pick P solve k` | a value the program cannot predict, constrained by `P` (masked-out loads, rounding choices, reduction trees). It is undefined behaviour when nothing satisfies `P`. `solve` is for running the model (`Solve`); the semantics ignores it |
| `plat q k` | an operation the platform defines (`tl.exp`, `/`, `tl.dot`, conversions into fp8): a result `PlatQ.Rel` allows for the environment's platform. It is undefined behaviour when there is none |
| `ub` | undefined behaviour |
| `loop init body k` | a `while` loop: run `body` until it returns `inr b` |
| `env k` | read the environment (platform, `enable_fp_fusion`) |

`Prog` is a monad, so a model is written in `do` notation. It follows the source line by line,
and the reviewer compares the two.

The constructors follow the source in order, but a program is not sequential for global
memory (library issue L15, user decision 2026-10-27). The threads that implement a program
access memory with no barrier between its operations, so a load need not see a store the program
made before it, and two stores to the same bytes may land in either order. The semantics does
not order them: a model in which two accesses of one program touch the same bytes and at least
one writes them, unless both are atomics on the same bytes, is not well behaved (`IntraRace`,
`TVA.Sem.Launch`), and no claim covers it. A program's atomics on the same bytes do not
conflict, and `Step.defer` lets the semantics perform an atomic after later steps that do not
depend on its result. That is coarser than the device, whose threads each order only their own
accesses (a tensor step that uses one lane's result holds back every later atomic here), so a
launch in which a program uses an atomic's result on a cell another program, or another atomic
operation of the same program, also accesses is excluded (library issue L161,
`TVA.Sem.AtomicResult`).
`tl.debug_barrier()` keeps no meaning here (a debugging aid; user, 2026-10-26: "we do not need
debug semantics here"), and `num_stages` stays dropped: the reuse of a program's own data that
pipelining could reorder is excluded.
-/
namespace TVA

/-! ## Running a model

`Step` (`TVA.Sem.Step`) is a relation, so a model cannot simply be run. The executor of
`validation/Exec.lean` runs one anyway, against concrete memory, and its soundness lemma says
`Step` can do whatever it reports. What it needs at each `pick` is a witness of `P`, and in
compiled code `P` is erased. So each `pick` carries a `Solve P`: code that proposes a witness,
asking an oracle (`Hints`) for the parts it cannot compute, and returns it with a proof. Because
the answer carries its proof, a wrong solver can only make the executor give up. It never changes
the model's meaning, and the reviewer of a model can ignore it. -/

/-- A certified answer to "some `b` with `P b`". -/
inductive Ans {β : Type} (P : β → Prop) : Type where
  /-- A witness. -/
  | ok (b : β) (h : P b)
  /-- No witness exists: the step is undefined behaviour. -/
  | empty (h : ∀ b, ¬ P b)

/-- What a solver may ask the oracle for: the parts of a witness it cannot compute. The family is
closed, so an oracle can match on the question. -/
inductive Query : Type → Type where
  /-- A reduction or scan tree over `n` leaves. -/
  | tree (n : Nat) : Query Tree
  /-- An order of the lanes of a block. -/
  | perm (s : Shape) : Query (List (Idx s))
  /-- A bijection between the lanes of two blocks (`reshape(can_reorder=True)`). -/
  | bij (s s' : Shape) : Query (Idx s' ≃ Idx s)
  /-- A rational (the exact operands of a fused addition). -/
  | rat : Query Rat
  /-- A choice between alternatives. -/
  | nat : Query Nat
  /-- An element (padding, the junk of a disabled lane). -/
  | val (d : DType) : Query d.Val

/-- An oracle for `Query`. -/
abbrev Hints := {τ : Type} → Query τ → Option τ

/-- How to run a `pick P`. `run` returns `none` when it has no answer. -/
structure Solve {β : Type} (P : β → Prop) where
  run : Hints → Option (Ans P)

namespace Solve

variable {β : Type} {P : β → Prop}

/-- No solver: the executor gives up at this `pick`. -/
def none : Solve P := ⟨fun _ => Option.none⟩

/-- The witness `b`, with its proof. -/
def ok (b : β) (h : P b) : Solve P := ⟨fun _ => some (.ok b h)⟩

/-- Try the witness `b`. -/
def check (b : β) [Decidable (P b)] : Solve P :=
  ⟨fun _ => if h : P b then some (.ok b h) else Option.none⟩

/-- `b` is the only candidate: if it fails, nothing satisfies `P`. -/
def only (b : β) [Decidable (P b)] (h : ∀ b', P b' → P b) : Solve P :=
  ⟨fun _ => if hb : P b then some (.ok b hb) else some (.empty fun b' hb' => hb (h b' hb'))⟩

/-- A guard `pick (fun _ : Unit => c)`. -/
def guard {c : Prop} [Decidable c] : Solve (fun _ : Unit => c) := only () fun _ h => h

/-- A compiler hint `pick (fun y => y = x ∧ c y)`: the witness is `x`. -/
def hint {c : β → Prop} (x : β) [Decidable (c x)] : Solve (fun y => y = x ∧ c y) :=
  ⟨fun _ => if h : c x then some (.ok x ⟨rfl, h⟩)
    else some (.empty fun _ hy => h (hy.1 ▸ hy.2))⟩

/-- A lanewise partial function `g`: the result is `g` at every lane, and there is none when some
lane has no value (integer `//` by zero, an oversized shift). -/
def lanes {s : Shape} {α : Type} [Inhabited α] (g : Idx s → Option α) :
    Solve (fun r : Tensor s α => ∀ i, g i = some (r.get i)) :=
  ⟨fun _ => if h : ∀ i, (g i).isSome then
      some (.ok ⟨fun i => (g i).getD default⟩ fun i => by
        obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp (h i)
        show g i = some ((g i).getD default)
        rw [ha]; rfl)
    else some (.empty fun r hr => h fun i => by rw [hr i]; rfl)⟩

end Solve

/-- The `sem` argument of a Triton atomic (default `acq_rel`). -/
inductive MemSem where
  | relaxed | acquire | release | acqRel
  deriving DecidableEq, Repr

/-- The `scope` argument of a Triton atomic (default `gpu`). -/
inductive Scope where
  | cta | gpu | sys
  deriving DecidableEq, Repr

def MemSem.releases : MemSem → Bool
  | .release | .acqRel => true
  | _ => false

def MemSem.acquires : MemSem → Bool
  | .acquire | .acqRel => true
  | _ => false

/-- **The order of an atomic whose result is unused** (library issue L157, user decision
2026-11-08): `acq_rel` at `gpu` or `sys` scope, whatever the source's `sem` (`relaxed`, `acquire`,
`release` or `acq_rel`); the source's `sem` at `cta` scope. The models of `tl.atomic_*` whose
returned value the source never uses (`tl.atomic_add_unused` and the others, `TVA.Lang.Ops`) give
their read-modify-writes this order, so that two of them, or one and an `acq_rel` atomic, on the
same bytes from different programs are `Event.Strong` and do not race (`TVA.Sem.Launch`).

Why it is sound. In the semantics an atomic's `sem` decides nothing but whether a pair of atomics
races: there is no synchronises-with between programs (happens-before is program order). A
program's atomics may be performed after any later step that does not depend on their result
(`Step.defer`), but not before an earlier step: so the semantics does not perform a later atomic
of the program before an earlier step that uses another atomic's result, while on the device
another thread may (library issue L161, lr-34a). Launches in which that can show are excluded: no
program uses an atomic's result on a cell another program, or another atomic operation of the same
program, also accesses (`TVA.Sem.AtomicResult`). Within that scope, an atomic whose result is used
reaches a cell only its own call reaches, so when it is performed is seen by no other program and
by no other operation of its own. What a relaxed (or acquire-only, release-only) atomic adds on the
device is that it may also be performed *before* earlier accesses of its thread. Its result is
unused, so no step of the program depends on when it was performed. Another program could see that order only through
memory: through the atomic's own bytes, which, in a well-behaved launch, no plain load or store of
the launch touches (one would race with the atomic, or conflict within the program, since no
exemption covers a plain access), and every atomic on which is performed with single-copy atomicity
(PTX ISA 9.5, §8.7 "Morally strong operations": two strong operations whose scopes include each
other's threads, through the same proxy, overlapping completely; a `.relaxed` `atom` or `red` is
strong, §8.4; §8.10.3 "Atomicity"); or through other bytes the program accessed, which another
program accessing them would race with. So the atomic's final effect on each cell is one update in
that cell's order of updates, as in the interleaving. Evidence: `evidence/relaxed-atomics` (the
instructions Triton emits, and every update kept under contention, with each `sem`). -/
def MemSem.unused (sem : MemSem) : Scope → MemSem
  | .cta => sem
  | _ => .acqRel

/-- A program, returning a value of type `α`. -/
inductive Prog : Type → Type 1 where
  | ret {α : Type} (a : α) : Prog α
  | load {α : Type} (d : DType) (s : Shape) (addrs : Idx s → Option Int)
      (k : Tensor s d.Val → Prog α) : Prog α
  | store {α : Type} (d : DType) (s : Shape) (addrs : Idx s → Option Int) (vals : Tensor s d.Val)
      (k : Prog α) : Prog α
  | rmw {α : Type} (d : DType) (addr : Int) (upd : d.Val → d.Val → Prop) (sem : MemSem)
      (scope : Scope) (k : d.Val → Prog α) : Prog α
  | pick {α β : Type} (P : β → Prop) (solve : Solve P) (k : β → Prog α) : Prog α
  | plat {α : Type} (q : PlatQ) (k : q.Out → Prog α) : Prog α
  | ub {α : Type} : Prog α
  | loop {α σ β : Type} (init : σ) (body : σ → Prog (σ ⊕ β)) (k : β → Prog α) : Prog α
  | env {α : Type} (k : Env → Prog α) : Prog α

namespace Prog

/-- Sequential composition. -/
def bind {α β : Type} : Prog α → (α → Prog β) → Prog β
  | ret a, f => f a
  | load d s addrs k, f => load d s addrs (fun v => (k v).bind f)
  | store d s addrs vals k, f => store d s addrs vals (k.bind f)
  | rmw d a upd sem sc k, f => rmw d a upd sem sc (fun v => (k v).bind f)
  | pick P sv k, f => pick P sv (fun b => (k b).bind f)
  | plat q k, f => plat q (fun r => (k r).bind f)
  | ub, _ => ub
  | loop init body k, f => loop init body (fun b => (k b).bind f)
  | env k, f => env (fun e => (k e).bind f)

instance : Monad Prog where
  pure := ret
  bind := bind

@[simp] theorem pure_eq {α : Type} (a : α) : (pure a : Prog α) = ret a := rfl
@[simp] theorem bind_eq {α β : Type} (p : Prog α) (f : α → Prog β) : p >>= f = p.bind f := rfl

@[simp] theorem ret_bind {α β : Type} (a : α) (f : α → Prog β) : (ret a).bind f = f a := rfl

@[simp] theorem bind_ret {α : Type} (p : Prog α) : p.bind ret = p := by
  induction p with
  | ret a => rfl
  | load d s addrs k ih => simp only [bind, ih]
  | store d s addrs vals k ih => simp only [bind, ih]
  | rmw d a upd sem sc k ih => simp only [bind, ih]
  | pick P sv k ih => simp only [bind, ih]
  | plat q k ih => simp only [bind, ih]
  | ub => rfl
  | loop init body k _ ih => simp only [bind, ih]
  | env k ih => simp only [bind, ih]

theorem bind_assoc {α β γ : Type} (p : Prog α) (f : α → Prog β) (g : β → Prog γ) :
    (p.bind f).bind g = p.bind (fun a => (f a).bind g) := by
  induction p with
  | ret a => rfl
  | load d s addrs k ih => simp only [bind, ih]
  | store d s addrs vals k ih => simp only [bind, ih]
  | rmw d a upd sem sc k ih => simp only [bind, ih]
  | pick P sv k ih => simp only [bind, ih]
  | plat q k ih => simp only [bind, ih]
  | ub => rfl
  | loop init body k _ ih => simp only [bind, ih]
  | env k ih => simp only [bind, ih]

instance : LawfulMonad Prog := LawfulMonad.mk'
  (id_map := fun p => bind_ret p)
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := fun p f g => bind_assoc p f g)

/-- One unfolding of a `while` loop: run the body, then stop or go round again. -/
def loopStep {α σ β : Type} (init : σ) (body : σ → Prog (σ ⊕ β)) (k : β → Prog α) : Prog α :=
  (body init).bind (fun r => match r with
    | .inl s => loop s body k
    | .inr b => k b)

end Prog

end TVA
