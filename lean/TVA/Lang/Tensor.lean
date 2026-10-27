import TVA.Lang.Scalar
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.Fintype.Basic

/-!
# Tensors

A Triton block value: a tensor of a static shape (`List Nat`; every shape in a compiled kernel
is a compile-time constant). `Tensor s α` holds an `α` at each index of `Idx s`; a scalar is the
rank-0 tensor `Tensor [] α`, or just an `α`.

The operations here are the *exact* ones: no rounding and no nondeterminism (index arithmetic,
comparisons, `tl.where`, broadcasting, `tl.trans`, integer reductions). Floating-point arithmetic
rounds and may be nondeterministic, so it lives in `TVA.Lang.Ops` as program actions.

Triton broadcasts implicitly (`x[:, None] + y[None, :]`, `offs < n`). A model broadcasts
explicitly: `bc a` is the scalar `a` at every index, `x.col n` is `x[:, None]` broadcast to `n`
columns, `x.row m` is `x[None, :]` broadcast to `m` rows. The reviewer checks that each
broadcast matches the source.
-/
namespace TVA

/-- A tensor shape. -/
abbrev Shape := List Nat

/-- The indices of a tensor of shape `s`: one `Fin` per dimension. -/
def Idx : Shape → Type
  | [] => Unit
  | n :: s => Fin n × Idx s

namespace Idx

instance instDecEq : (s : Shape) → DecidableEq (Idx s)
  | [] => inferInstanceAs (DecidableEq Unit)
  | _ :: s => @instDecidableEqProd _ _ _ (instDecEq s)

/-- Every index, in row-major order (the last dimension fastest). -/
def all : (s : Shape) → List (Idx s)
  | [] => [()]
  | n :: s => (List.finRange n).flatMap (fun i => (all s).map (fun j => ((i, j) : Fin n × Idx s)))

theorem mem_all : (s : Shape) → (i : Idx s) → i ∈ all s
  | [], () => List.mem_singleton.mpr rfl
  | _ :: s, (i, j) =>
      List.mem_flatMap.mpr ⟨i, List.mem_finRange i, List.mem_map.mpr ⟨j, mem_all s j, rfl⟩⟩

theorem nodup_all : (s : Shape) → (all s).Nodup
  | [] => List.nodup_singleton _
  | n :: s => by
      refine List.nodup_flatMap.mpr ⟨fun i _ => (nodup_all s).map ?_, ?_⟩
      · intro a b h; exact (Prod.mk.inj h).2
      · refine (List.nodup_finRange n).pairwise_of_forall_ne ?_
        intro a _ b _ hab
        refine List.disjoint_left.mpr ?_
        intro x hx hy
        obtain ⟨u, _, rfl⟩ := List.mem_map.mp hx
        obtain ⟨v, _, h⟩ := List.mem_map.mp hy
        exact hab (Prod.mk.inj h).1.symm

instance instFintype (s : Shape) : Fintype (Idx s) :=
  ⟨⟨(all s : Multiset (Idx s)), nodup_all s⟩, mem_all s⟩

/-- The row-major position of an index (`0 ≤ pos < s.prod`). -/
def pos : (s : Shape) → Idx s → Nat
  | [], () => 0
  | _ :: s, (i, j) => i.val * s.prod + pos s j

end Idx

/-- A tensor of shape `s` with elements in `α`. -/
@[ext] structure Tensor (s : Shape) (α : Type) where
  get : Idx s → α

namespace Tensor

variable {s : Shape} {α β γ δ : Type}

instance [Inhabited α] : Inhabited (Tensor s α) := ⟨⟨fun _ => default⟩⟩

/-- The tensor with every element `a` (`tl.full`, and implicit broadcasting of a scalar). -/
def full (a : α) : Tensor s α := ⟨fun _ => a⟩

/-- Elementwise application. -/
def map (f : α → β) (x : Tensor s α) : Tensor s β := ⟨fun i => f (x.get i)⟩
def map2 (f : α → β → γ) (x : Tensor s α) (y : Tensor s β) : Tensor s γ :=
  ⟨fun i => f (x.get i) (y.get i)⟩
def map3 (f : α → β → γ → δ) (x : Tensor s α) (y : Tensor s β) (z : Tensor s γ) : Tensor s δ :=
  ⟨fun i => f (x.get i) (y.get i) (z.get i)⟩

/-- The elements in row-major order. -/
def toList (x : Tensor s α) : List α := (Idx.all s).map x.get

/-! ### Rank 0, 1 and 2 -/

/-- The value of a rank-0 tensor, and the rank-0 tensor of a value. -/
def item (x : Tensor [] α) : α := x.get ()
def scalar (a : α) : Tensor [] α := ⟨fun _ => a⟩

/-- Rank-1 and rank-2 tensors from functions, and their elements. -/
def of1 {n : Nat} (f : Fin n → α) : Tensor [n] α := ⟨fun i => f i.1⟩
def of2 {m n : Nat} (f : Fin m → Fin n → α) : Tensor [m, n] α := ⟨fun i => f i.1 i.2.1⟩
def get1 {n : Nat} (x : Tensor [n] α) (i : Fin n) : α := x.get (i, ())
def get2 {m n : Nat} (x : Tensor [m, n] α) (i : Fin m) (j : Fin n) : α := x.get (i, j, ())

@[simp] theorem get1_of1 {n : Nat} (f : Fin n → α) (i : Fin n) : (of1 f).get1 i = f i := rfl
@[simp] theorem get2_of2 {m n : Nat} (f : Fin m → Fin n → α) (i : Fin m) (j : Fin n) :
    (of2 f).get2 i j = f i j := rfl

/-- `x[:, None]`, broadcast to `n` columns. -/
def col {m : Nat} (x : Tensor [m] α) (n : Nat) : Tensor [m, n] α := of2 (fun i _ => x.get1 i)
/-- `x[None, :]`, broadcast to `m` rows. -/
def row {n : Nat} (x : Tensor [n] α) (m : Nat) : Tensor [m, n] α := of2 (fun _ j => x.get1 j)
/-- `x[:, None]` (shape `[m, 1]`) and `x[None, :]` (shape `[1, n]`), without broadcasting. -/
def unsqueeze1 {m : Nat} (x : Tensor [m] α) : Tensor [m, 1] α := of2 (fun i _ => x.get1 i)
def unsqueeze0 {n : Nat} (x : Tensor [n] α) : Tensor [1, n] α := of2 (fun _ j => x.get1 j)
/-- Broadcast a `[m, 1]` tensor to `[m, n]`, and a `[1, n]` tensor to `[m, n]`. -/
def bcastCols {m : Nat} (x : Tensor [m, 1] α) (n : Nat) : Tensor [m, n] α := of2 (fun i _ => x.get2 i 0)
def bcastRows {n : Nat} (x : Tensor [1, n] α) (m : Nat) : Tensor [m, n] α := of2 (fun _ j => x.get2 0 j)

/-- `tl.trans` (`x.T`) of a rank-2 tensor. -/
def trans {m n : Nat} (x : Tensor [m, n] α) : Tensor [n, m] α := of2 (fun j i => x.get2 i j)

@[simp] theorem get2_col {m n : Nat} (x : Tensor [m] α) (i : Fin m) (j : Fin n) :
    (x.col n).get2 i j = x.get1 i := rfl
@[simp] theorem get2_row {m n : Nat} (x : Tensor [n] α) (i : Fin m) (j : Fin n) :
    (x.row m).get2 i j = x.get1 j := rfl
@[simp] theorem get2_trans {m n : Nat} (x : Tensor [m, n] α) (j : Fin n) (i : Fin m) :
    x.trans.get2 j i = x.get2 i j := rfl
@[simp] theorem get_full (a : α) (i : Idx s) : (full a : Tensor s α).get i = a := rfl
@[simp] theorem get_map (f : α → β) (x : Tensor s α) (i : Idx s) : (x.map f).get i = f (x.get i) := rfl
@[simp] theorem get_map2 (f : α → β → γ) (x : Tensor s α) (y : Tensor s β) (i : Idx s) :
    (map2 f x y).get i = f (x.get i) (y.get i) := rfl
@[simp] theorem get_map3 (f : α → β → γ → δ) (x : Tensor s α) (y : Tensor s β) (z : Tensor s γ)
    (i : Idx s) : (map3 f x y z).get i = f (x.get i) (y.get i) (z.get i) := rfl

/-! ### Construction -/

/-- `tl.arange(start, start + n)`: `start, start+1, …` as `tl.int32`. (Triton requires `n` to be
a power of two; a model of a compiled kernel satisfies that.) -/
def arange (start : Int) (n : Nat) : Tensor [n] I32 := of1 (fun i => IntV.ofInt (start + i.val))

@[simp] theorem get1_arange (start : Int) (n : Nat) (i : Fin n) :
    (arange start n).get1 i = IntV.ofInt (start + i.val) := rfl

/-- `tl.where(c, x, y)`: both operands are computed (eagerly); each element is selected. -/
def where_ (c : Tensor s Bool) (x y : Tensor s α) : Tensor s α :=
  ⟨fun i => if c.get i then x.get i else y.get i⟩

@[simp] theorem get_where (c : Tensor s Bool) (x y : Tensor s α) (i : Idx s) :
    (where_ c x y).get i = if c.get i then x.get i else y.get i := rfl

/-! ### Elementwise operations on integer, boolean and pointer tensors -/

section IntOps
variable {sg : Bool} {w : Nat}

instance : Add (Tensor s (IntV sg w)) := ⟨map2 (· + ·)⟩
instance : Sub (Tensor s (IntV sg w)) := ⟨map2 (· - ·)⟩
instance : Mul (Tensor s (IntV sg w)) := ⟨map2 (· * ·)⟩
instance : Neg (Tensor s (IntV sg w)) := ⟨map (- ·)⟩

/-- `a + x` and `x + a` for a scalar `a` (implicit broadcasting, as in `block_start + offs`). -/
instance : HAdd (IntV sg w) (Tensor s (IntV sg w)) (Tensor s (IntV sg w)) := ⟨fun a x => x.map (a + ·)⟩
instance : HAdd (Tensor s (IntV sg w)) (IntV sg w) (Tensor s (IntV sg w)) := ⟨fun x a => x.map (· + a)⟩
instance : HMul (IntV sg w) (Tensor s (IntV sg w)) (Tensor s (IntV sg w)) := ⟨fun a x => x.map (a * ·)⟩
instance : HMul (Tensor s (IntV sg w)) (IntV sg w) (Tensor s (IntV sg w)) := ⟨fun x a => x.map (· * a)⟩

@[simp] theorem get_sadd (a : IntV sg w) (x : Tensor s (IntV sg w)) (i : Idx s) :
    (a + x).get i = a + x.get i := rfl
@[simp] theorem get_adds (x : Tensor s (IntV sg w)) (a : IntV sg w) (i : Idx s) :
    (x + a).get i = x.get i + a := rfl
@[simp] theorem get_smul (a : IntV sg w) (x : Tensor s (IntV sg w)) (i : Idx s) :
    (a * x).get i = a * x.get i := rfl
@[simp] theorem get_muls (x : Tensor s (IntV sg w)) (a : IntV sg w) (i : Idx s) :
    (x * a).get i = x.get i * a := rfl

@[simp] theorem get_add (x y : Tensor s (IntV sg w)) (i : Idx s) : (x + y).get i = x.get i + y.get i := rfl
@[simp] theorem get_sub (x y : Tensor s (IntV sg w)) (i : Idx s) : (x - y).get i = x.get i - y.get i := rfl
@[simp] theorem get_mul (x y : Tensor s (IntV sg w)) (i : Idx s) : (x * y).get i = x.get i * y.get i := rfl
@[simp] theorem get_neg (x : Tensor s (IntV sg w)) (i : Idx s) : (-x).get i = -x.get i := rfl

end IntOps

/-- `<`, `<=`, `==`, `!=` on integer tensors, elementwise. -/
def lt {sg : Bool} {w : Nat} (x y : Tensor s (IntV sg w)) : Tensor s Bool := map2 IntV.lt x y
/-- `x < a` for a scalar `a`. -/
def ltS {sg : Bool} {w : Nat} (x : Tensor s (IntV sg w)) (a : IntV sg w) : Tensor s Bool := x.map (IntV.lt · a)
@[simp] theorem get_ltS {sg : Bool} {w : Nat} (x : Tensor s (IntV sg w)) (a : IntV sg w) (i : Idx s) :
    (x.ltS a).get i = IntV.lt (x.get i) a := rfl
@[simp] theorem get_lt {sg : Bool} {w : Nat} (x y : Tensor s (IntV sg w)) (i : Idx s) :
    (x.lt y).get i = IntV.lt (x.get i) (y.get i) := rfl
def le {sg : Bool} {w : Nat} (x y : Tensor s (IntV sg w)) : Tensor s Bool := map2 IntV.le x y
def eq {sg : Bool} {w : Nat} (x y : Tensor s (IntV sg w)) : Tensor s Bool := map2 IntV.eq x y
/-- `<`, `<=`, `==`, `!=` on float tensors (ordered, except `!=`). -/
def flt {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) : Tensor s Bool := map2 FloatV.lt x y
def fle {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) : Tensor s Bool := map2 FloatV.le x y
def feq {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) : Tensor s Bool := map2 FloatV.eq x y
def fne {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) : Tensor s Bool := map2 FloatV.ne x y

/-- `&`, `|`, `~` on boolean tensors. -/
def and (x y : Tensor s Bool) : Tensor s Bool := map2 (· && ·) x y
def or (x y : Tensor s Bool) : Tensor s Bool := map2 (· || ·) x y
def not (x : Tensor s Bool) : Tensor s Bool := map (! ·) x

/-- Integer cast of each element (`x.to(tl.int64)` etc.). Not for `x.to(tl.int1)`, which is
`x != 0` (`semantic.py:874-887`): write that comparison. -/
def castInt {sg sg' : Bool} {w w' : Nat} (x : Tensor s (IntV sg w)) : Tensor s (IntV sg' w') :=
  map IntV.cast x

section Get
variable {sg : Bool} {w : Nat}

/-! Elementwise access in the `x.get (i, j, ())` form, the form `tl.addrs` and the load/store
rules use. -/

@[simp] theorem get_col {m n : Nat} (x : Tensor [m] α) (i : Fin m) (j : Fin n) :
    (x.col n).get (i, j, ()) = x.get (i, ()) := rfl
@[simp] theorem get_row {m n : Nat} (x : Tensor [n] α) (i : Fin m) (j : Fin n) :
    (x.row m).get (i, j, ()) = x.get (j, ()) := rfl
@[simp] theorem get_of1 {n : Nat} (f : Fin n → α) (i : Fin n) : (of1 f).get (i, ()) = f i := rfl
@[simp] theorem get_of2 {m n : Nat} (f : Fin m → Fin n → α) (i : Fin m) (j : Fin n) :
    (of2 f).get (i, j, ()) = f i j := rfl
@[simp] theorem get_arange (start : Int) (n : Nat) (i : Fin n) :
    (arange start n).get (i, ()) = IntV.ofInt (start + i.val) := rfl
@[simp] theorem get_le (x y : Tensor s (IntV sg w)) (i : Idx s) :
    (x.le y).get i = IntV.le (x.get i) (y.get i) := rfl
@[simp] theorem get_eq (x y : Tensor s (IntV sg w)) (i : Idx s) :
    (x.eq y).get i = IntV.eq (x.get i) (y.get i) := rfl
@[simp] theorem get_and (x y : Tensor s Bool) (i : Idx s) : (x.and y).get i = (x.get i && y.get i) := rfl
@[simp] theorem get_or (x y : Tensor s Bool) (i : Idx s) : (x.or y).get i = (x.get i || y.get i) := rfl
@[simp] theorem get_not (x : Tensor s Bool) (i : Idx s) : x.not.get i = !x.get i := rfl
@[simp] theorem get_castInt {sg' : Bool} {w' : Nat} (x : Tensor s (IntV sg w)) (i : Idx s) :
    (x.castInt : Tensor s (IntV sg' w')).get i = IntV.cast (x.get i) := rfl

end Get

/-! ### Exact reductions

Reductions whose result does not depend on the order of combination: integer `+` (wrapping),
boolean `and`/`or`, and integer `max`/`min`. Floating-point reductions are in `TVA.Lang.Ops`. -/

/-- Combine all elements with `op` from `init`, in row-major order. -/
def foldAll (op : β → α → β) (init : β) (x : Tensor s α) : β := x.toList.foldl op init

/-- Rank-2 reductions along axis 1 (each row) and axis 0 (each column). -/
def foldRows {m n : Nat} (op : β → α → β) (init : β) (x : Tensor [m, n] α) : Tensor [m] β :=
  of1 (fun i => (List.finRange n).foldl (fun acc j => op acc (x.get2 i j)) init)
def foldCols {m n : Nat} (op : β → α → β) (init : β) (x : Tensor [m, n] α) : Tensor [n] β :=
  of1 (fun j => (List.finRange m).foldl (fun acc i => op acc (x.get2 i j)) init)

end Tensor

/-- `bc a`: the scalar `a` broadcast to the shape the context needs (`a` in `offs < n`). -/
abbrev bc {s : Shape} {α : Type} (a : α) : Tensor s α := Tensor.full a

/-! ## Reduction trees

Triton reduces in an order of its choosing (within a thread, then across lanes and warps), so a
floating-point reduction is described by a binary tree whose leaves are the reduced positions,
in some order. A scan (`tl.cumsum`) keeps the order of its leaves. -/

/-- A binary tree with leaves numbered by `Nat`. -/
inductive Tree where
  | leaf (i : Nat)
  | node (l r : Tree)
  deriving DecidableEq, Repr

namespace Tree

/-- The leaves, left to right. -/
def leaves : Tree → List Nat
  | leaf i => [i]
  | node l r => l.leaves ++ r.leaves

/-- `t` combines each of `0, …, n-1` exactly once, in some order. -/
def Covers (n : Nat) (t : Tree) : Prop := t.leaves.Perm (List.range n)

/-- `t` combines `0, …, n-1` in order (a scan's prefix). -/
def InOrder (n : Nat) (t : Tree) : Prop := t.leaves = List.range n

/-- Evaluate a tree with a *relation* for each combination (a rounding, possibly fused, add):
`Eval op x t r` says `t` can produce `r` when leaf `i` is `x i`. -/
inductive Eval {α : Type} (op : α → α → α → Prop) (x : Nat → α) : Tree → α → Prop where
  | leaf (i : Nat) : Eval op x (leaf i) (x i)
  | node {l r : Tree} {a b c : α} : Eval op x l a → Eval op x r b → op a b c → Eval op x (node l r) c

/-- The left-to-right sequential tree over `0, …, n-1` (`n ≥ 1`). -/
def seq : Nat → Tree
  | 0 => leaf 0
  | 1 => leaf 0
  | n + 2 => node (seq (n + 1)) (leaf (n + 1))

end Tree

/-! ## Coordinates

`Idx.coord i k` is the `k`-th coordinate of an index and `Idx.set i k t` replaces it, so that a
definition can work along any axis (scans, sorting, hints, block pointers). `Shape.dim s k` is
the size of dimension `k`. -/

/-- The size of dimension `k` (`0` beyond the rank). -/
def Shape.dim : Shape → Nat → Nat
  | [], _ => 0
  | n :: _, 0 => n
  | _ :: s, k + 1 => Shape.dim s k

namespace Idx

/-- The `k`-th coordinate of an index (`0` beyond the rank). -/
def coord : {s : Shape} → Idx s → Nat → Nat
  | [], (), _ => 0
  | _ :: _, (i, _), 0 => i.1
  | _ :: s, (_, j), k + 1 => coord (s := s) j k

/-- Replace the `k`-th coordinate by `t`; no change when `t` is out of range. -/
def set : {s : Shape} → Idx s → Nat → Nat → Idx s
  | [], (), _, _ => ()
  | n :: _, (i, j), 0, t => (if h : t < n then ⟨t, h⟩ else i, j)
  | _ :: s, (i, j), k + 1, t => (i, set (s := s) j k t)

theorem coord_lt : ∀ {s : Shape} (i : Idx s) {k : Nat}, k < s.length → i.coord k < Shape.dim s k
  | [], (), _, h => absurd h (Nat.not_lt_zero _)
  | _ :: _, (i, _), 0, _ => i.2
  | _ :: s, (_, j), _ + 1, h => coord_lt (s := s) j (Nat.lt_of_succ_lt_succ h)

theorem coord_set_self : ∀ {s : Shape} (i : Idx s) {k t : Nat}, t < Shape.dim s k →
    (i.set k t).coord k = t
  | [], (), _, _, h => absurd h (Nat.not_lt_zero _)
  | n :: _, (i, j), 0, t, h => by
      have h' : t < n := h
      simp [set, coord, h']
  | _ :: s, (_, j), k + 1, _, h => coord_set_self (s := s) j h

theorem coord_set_ne : ∀ {s : Shape} (i : Idx s) {k k' t : Nat}, k' ≠ k →
    (i.set k t).coord k' = i.coord k'
  | [], (), _, _, _, _ => rfl
  | _ :: _, (_, _), 0, 0, _, h => absurd rfl h
  | _ :: _, (_, _), 0, _ + 1, _, _ => rfl
  | _ :: _, (_, _), _ + 1, 0, _, _ => rfl
  | _ :: s, (_, j), _ + 1, _ + 1, _, h =>
      coord_set_ne (s := s) j (fun e => h (congrArg (· + 1) e))

theorem set_coord_self : ∀ {s : Shape} (i : Idx s) (k : Nat), i.set k (i.coord k) = i
  | [], (), _ => rfl
  | _ :: _, (i, j), 0 => by simp [set, coord, i.2]; rfl
  | _ :: s, (i, j), k + 1 => by simp only [set, coord, set_coord_self (s := s) j k]; rfl

theorem set_set : ∀ {s : Shape} (i : Idx s) {k t t' : Nat}, t' < Shape.dim s k →
    (i.set k t).set k t' = i.set k t'
  | [], (), _, _, _, _ => rfl
  | n :: _, (i, j), 0, t, t', h => by
      have h' : t' < n := h
      simp [set, h']; rfl
  | _ :: s, (i, j), k + 1, _, _, h => by simp only [set, set_set (s := s) j h]; rfl

theorem ext_coord : ∀ {s : Shape} {i j : Idx s}, (∀ k, i.coord k = j.coord k) → i = j
  | [], (), (), _ => rfl
  | _ :: _, (a, i), (b, j), h => by
      have h0 := h 0
      simp only [coord] at h0
      rw [Fin.ext h0, ext_coord (fun k => h (k + 1))]

/-- `i` and `j` lie in the same line along axis `k`: they agree off `k`. -/
theorem eq_set_of_coord : ∀ {s : Shape} {i j : Idx s} {k : Nat},
    (∀ k', k' ≠ k → i.coord k' = j.coord k') → j = i.set k (j.coord k)
  | [], (), (), _, _ => rfl
  | _ :: _, (a, i), (b, j), 0, h => by
      have hrest : i = j := ext_coord fun k' => h (k' + 1) (Nat.succ_ne_zero _)
      simp [set, coord, b.2, hrest]
      rfl
  | _ :: s, (a, i), (b, j), k + 1, h => by
      have ha : a = b := Fin.ext (h 0 (Nat.succ_ne_zero k).symm)
      have hr := eq_set_of_coord (s := s) (i := i) (j := j) (k := k)
        (fun k' hk => h (k' + 1) (fun e => hk (Nat.succ_injective e)))
      simp only [set, coord, ha] at hr ⊢
      rw [← hr]

/-! ### The last coordinate -/

/-- Append a last coordinate. -/
def snoc {n : Nat} : {s : Shape} → Idx s → Fin n → Idx (s ++ [n])
  | [], (), t => (t, ())
  | _ :: _, (i, j), t => (i, snoc j t)

/-- The index without its last coordinate, and that coordinate. -/
def unsnoc {n : Nat} : {s : Shape} → Idx (s ++ [n]) → Idx s × Fin n
  | [], (t, ()) => ((), t)
  | _ :: _, (i, j) => ((i, (unsnoc j).1), (unsnoc j).2)

@[simp] theorem unsnoc_snoc {n : Nat} : ∀ {s : Shape} (i : Idx s) (t : Fin n),
    unsnoc (snoc i t) = (i, t)
  | [], (), _ => rfl
  | _ :: _, (i, j), t => by simp only [snoc, unsnoc, unsnoc_snoc j t]

@[simp] theorem snoc_unsnoc {n : Nat} : ∀ {s : Shape} (i : Idx (s ++ [n])),
    snoc (unsnoc i).1 (unsnoc i).2 = i
  | [], (_, ()) => rfl
  | _ :: _, (i, j) => by simp only [snoc, unsnoc, snoc_unsnoc j]; rfl

/-! ### Row-major positions -/

theorem prod_pos_of_lt {n : Nat} {s : Shape} {p : Nat} (h : p < (n :: s).prod) : 0 < s.prod := by
  rw [List.prod_cons] at h
  rcases Nat.eq_zero_or_pos s.prod with h0 | h0
  · rw [h0, Nat.mul_zero] at h; exact absurd h (Nat.not_lt_zero _)
  · exact h0

theorem pos_lt : ∀ (s : Shape) (i : Idx s), pos s i < s.prod
  | [], () => Nat.one_pos
  | n :: s, (i, j) => by
      rw [List.prod_cons]
      simp only [pos]
      calc i.val * s.prod + pos s j < i.val * s.prod + s.prod := Nat.add_lt_add_left (pos_lt s j) _
        _ = (i.val + 1) * s.prod := by rw [Nat.succ_mul]
        _ ≤ n * s.prod := Nat.mul_le_mul_right _ i.2

/-- The index at row-major position `p`. -/
def ofPos : (s : Shape) → (p : Nat) → p < s.prod → Idx s
  | [], _, _ => ()
  | n :: s, p, h =>
      (⟨p / s.prod, (Nat.div_lt_iff_lt_mul (prod_pos_of_lt h)).mpr (by rw [List.prod_cons] at h; exact h)⟩,
        ofPos s (p % s.prod) (Nat.mod_lt _ (prod_pos_of_lt h)))

theorem ofPos_congr {s : Shape} {p q : Nat} (h : p = q) (hp : p < s.prod) (hq : q < s.prod) :
    ofPos s p hp = ofPos s q hq := by subst h; rfl

theorem pos_ofPos : ∀ (s : Shape) (p : Nat) (h : p < s.prod), pos s (ofPos s p h) = p
  | [], p, h => by
      have h' : p < 1 := h
      show 0 = p
      omega
  | n :: s, p, h => by
      simp only [ofPos, pos, pos_ofPos s]
      exact Nat.div_add_mod' p s.prod

theorem ofPos_pos : ∀ (s : Shape) (i : Idx s) (h : pos s i < s.prod), ofPos s (pos s i) h = i
  | [], (), _ => rfl
  | n :: s, (i, j), h => by
      have hP : 0 < s.prod := prod_pos_of_lt h
      have hj := pos_lt s j
      simp only [ofPos, pos]
      have h1 : (i.val * s.prod + pos s j) / s.prod = i.val := by
        rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt hj, Nat.zero_add]
      have h2 : (i.val * s.prod + pos s j) % s.prod = pos s j := by
        rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
      congr 1
      · exact Fin.ext h1
      · rw [ofPos_congr h2 _ hj, ofPos_pos s j]

end Idx

/-! ## Shape changes

`tl.join`, `tl.split`, `tl.interleave` and `tl.reshape` (with `can_reorder=False`) move
elements without changing them. `reshape` keeps row-major order (`semantic.py:639-646`);
`interleave` is `join` followed by such a reshape (`standard.py:540-560`). -/

namespace Tensor

variable {s : Shape} {α : Type}

/-- `tl.join(a, b)`: a new last dimension of size 2, `a` at `0` and `b` at `1`
(`core.py:2160`). Two scalars give a tensor of shape `[2]`. -/
def join (a b : Tensor s α) : Tensor (s ++ [2]) α :=
  ⟨fun i => if (Idx.unsnoc i).2.val = 0 then a.get (Idx.unsnoc i).1 else b.get (Idx.unsnoc i).1⟩

/-- `tl.split(x)`: the two halves along a last dimension of size 2 (`core.py:2198`). -/
def split (x : Tensor (s ++ [2]) α) : Tensor s α × Tensor s α :=
  (⟨fun i => x.get (Idx.snoc i 0)⟩, ⟨fun i => x.get (Idx.snoc i 1)⟩)

/-- `tl.interleave(a, b)`: along the last dimension, the elements of `a` and `b` alternate,
`a` first (`join(a, b).reshape(a.shape[:-1] + [2 * a.shape[-1]])`). -/
def interleave {n : Nat} (a b : Tensor (s ++ [n]) α) : Tensor (s ++ [2 * n]) α :=
  ⟨fun i =>
    let half : Fin n := ⟨(Idx.unsnoc i).2.val / 2, Nat.div_lt_of_lt_mul (Idx.unsnoc i).2.2⟩
    if (Idx.unsnoc i).2.val % 2 = 0 then a.get (Idx.snoc (Idx.unsnoc i).1 half)
    else b.get (Idx.snoc (Idx.unsnoc i).1 half)⟩

/-- `tl.reshape(x, s')` (`can_reorder=False`): the same elements in row-major order. -/
def reshape (x : Tensor s α) (s' : Shape) (h : s'.prod = s.prod) : Tensor s' α :=
  ⟨fun i => x.get (Idx.ofPos s (Idx.pos s' i) (h ▸ Idx.pos_lt s' i))⟩

@[simp] theorem get_join_snoc (a b : Tensor s α) (i : Idx s) (t : Fin 2) :
    (join a b).get (Idx.snoc i t) = if t.val = 0 then a.get i else b.get i := by
  simp [join]

@[simp] theorem split_join (a b : Tensor s α) : (join a b).split = (a, b) := by
  simp [split, join]

@[simp] theorem join_split (x : Tensor (s ++ [2]) α) : join x.split.1 x.split.2 = x := by
  ext i
  obtain ⟨j, t, rfl⟩ : ∃ j t, i = Idx.snoc j t := ⟨_, _, (Idx.snoc_unsnoc i).symm⟩
  rw [get_join_snoc]
  simp only [split]
  split_ifs with h0
  · have ht : t = 0 := Fin.ext h0
    rw [ht]
  · have ht : t = 1 := by
      apply Fin.ext
      have := t.2
      simp only [Fin.val_one]
      omega
    rw [ht]

@[simp] theorem get_interleave_snoc {n : Nat} (a b : Tensor (s ++ [n]) α) (i : Idx s) (t : Fin (2 * n)) :
    (interleave a b).get (Idx.snoc i t) =
      if t.val % 2 = 0 then a.get (Idx.snoc i ⟨t.val / 2, Nat.div_lt_of_lt_mul t.2⟩)
      else b.get (Idx.snoc i ⟨t.val / 2, Nat.div_lt_of_lt_mul t.2⟩) := by
  simp [interleave]

theorem get_reshape (x : Tensor s α) (s' : Shape) (h : s'.prod = s.prod) (i : Idx s') :
    (x.reshape s' h).get i = x.get (Idx.ofPos s (Idx.pos s' i) (h ▸ Idx.pos_lt s' i)) := rfl

/-- Reshaping back gives the original tensor. -/
@[simp] theorem reshape_reshape (x : Tensor s α) (s' : Shape) (h : s'.prod = s.prod)
    (h' : s.prod = s'.prod) : (x.reshape s' h).reshape s h' = x := by
  ext i
  simp only [get_reshape]
  rw [Idx.ofPos_congr (Idx.pos_ofPos s' _ _) _ (Idx.pos_lt s i), Idx.ofPos_pos]

end Tensor

end TVA
