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
| `rmw d a upd sem scope k` | one element of `tl.atomic_*`: read the old value, write a new one related to it by `upd`, return the old value |
| `pick P k` | a value the program cannot predict, constrained by `P` (masked-out loads, rounding choices, reduction trees, platform results). It is undefined behaviour when nothing satisfies `P` |
| `ub` | undefined behaviour |
| `loop init body k` | a `while` loop: run `body` until it returns `inr b` |
| `env k` | read the environment (platform, `enable_fp_fusion`) |

`Prog` is a monad, so a model is written in `do` notation. It follows the source line by line,
and the reviewer compares the two.

A program is sequential, as the Triton documentation presents it: each operation sees the
effects of the program's earlier operations. The threads that implement a program are not part
of the language, so `tl.debug_barrier()` has no meaning here (a debugging aid; user,
2026-10-26: "we do not need debug semantics here").
-/
namespace TVA

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

/-- A program, returning a value of type `α`. -/
inductive Prog : Type → Type 1 where
  | ret {α : Type} (a : α) : Prog α
  | load {α : Type} (d : DType) (s : Shape) (addrs : Idx s → Option Int)
      (k : Tensor s d.Val → Prog α) : Prog α
  | store {α : Type} (d : DType) (s : Shape) (addrs : Idx s → Option Int) (vals : Tensor s d.Val)
      (k : Prog α) : Prog α
  | rmw {α : Type} (d : DType) (addr : Int) (upd : d.Val → d.Val → Prop) (sem : MemSem)
      (scope : Scope) (k : d.Val → Prog α) : Prog α
  | pick {α β : Type} (P : β → Prop) (k : β → Prog α) : Prog α
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
  | pick P k, f => pick P (fun b => (k b).bind f)
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
  | pick P k ih => simp only [bind, ih]
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
  | pick P k ih => simp only [bind, ih]
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
