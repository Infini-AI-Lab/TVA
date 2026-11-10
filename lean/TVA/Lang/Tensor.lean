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
columns, `x.row m` is `x[None, :]` broadcast to `m` rows. `tl.broadcast_to(x, t)` of a block is
`x.broadcastTo t`, any rank, with Triton's check of the shapes (`Shape.bcastOk`; library issue
L140). The reviewer checks that each broadcast matches the source.
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

/-- `tl.trans` (`x.T`) of a rank-2 tensor. Other ranks, and explicit `dims`, are
`Tensor.permute` (`permute_trans`: this is `permute [1, 0]`). -/
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

/-- Integer cast of each element (`IntV.cast`: the value modulo `2^w'`). The model writes
`x.to(d')` and an implicit promotion as `tl.convert d d' x` (`TVA.Lang.Promote`); between integer
dtypes other than `int1`, that conversion has exactly one result, `x.castInt`
(`ConvRel.iff_cast_i32_i64` and the other pairs, `safe_convert_iff` in `TVA.Lemmas.IntW`), so
`x.castInt` is the value of `x.to(tl.int64)` too, and either form transcribes it. The model also
uses it for its own widening steps (an integer reduction, `tl.make_block_ptr`). Not for
`x.to(tl.int1)`, which is `x != 0` (`semantic.py:874-887`): write that comparison. -/
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
boolean `and`/`or`, and integer `max`/`min`. Floating-point reductions are in `TVA.Lang.Ops`.

Triton widens small integers before these reductions, and the model writes the widening:
* `tl.max` and `tl.min` without `return_indices` convert an integer tensor narrower than 32 bits
  to `int32`, whatever its signedness (`triton-3.8.0-standard.py:184-190, 233-239`): `int8` and
  `int16` are sign-extended, `uint8` and `uint16` zero-extended, and `tl.int1` becomes `0`/`1`.
  The result is `int32`. The model converts first (`Tensor.castInt` to `I32`; a `Bool` tensor
  is mapped to `0`/`1`) and reduces with the `int32` `IntV.max`/`IntV.min`.
* `tl.sum` converts an integer tensor narrower than 32 bits to `int32` if it is signed and to
  `uint32` if it is unsigned, `tl.int1` included (`_pick_sum_dtype`,
  `triton-3.8.0-standard.py:254-283`). The sum wraps in that type.
* `tl.max`/`tl.min` with `return_indices`, and `tl.argmax`/`tl.argmin`, do not widen integers:
  they are `tl.argmaxAll`, `tl.argmax1`, `tl.argmax0` in `TVA.Lang.Ops`. -/

/-- Combine all elements with `op` from `init`, in row-major order. For a source `tl.max`,
`tl.min` or `tl.sum` of integers, `x` is the widened tensor (see the section's docstring): a
tensor narrower than 32 bits is first converted to `int32` (`max`/`min`, any signedness) or to
`int32`/`uint32` by signedness (`sum`). -/
def foldAll (op : β → α → β) (init : β) (x : Tensor s α) : β := x.toList.foldl op init

/-- Rank-2 reductions along axis 1 (each row) and axis 0 (each column), with the same widening of
small integers as `foldAll`. -/
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

/-- The depth: a leaf has depth `0`. -/
def depth : Tree → Nat
  | leaf _ => 0
  | node l r => max l.depth r.depth + 1

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

/-! ## Permutations of the axes

`tl.permute(x, dims)` and `tl.trans(x, dims)` (`triton-3.8.0-core.py:2070-2121`) reorder the
axes of a tensor without changing its elements: dimension `k` of the result is dimension
`dims[k]` of the input (`triton-3.8.0-semantic.py:701-709`), so the result's element at index
`o` is the input's element whose coordinate `dims[k]` is `o`'s coordinate `k`. -/

/-- `Shape.dim` is the list's entry, `0` beyond the rank. -/
theorem Shape.dim_eq_getD : ∀ (s : Shape) (k : Nat), Shape.dim s k = s.getD k 0
  | [], _ => rfl
  | _ :: _, 0 => rfl
  | _ :: s, k + 1 => Shape.dim_eq_getD s k

namespace Idx

/-- The index with coordinates `c 0, c 1, …` (each below its dimension). -/
def ofCoords : (s : Shape) → (c : Nat → Nat) → (∀ k, k < s.length → c k < Shape.dim s k) → Idx s
  | [], _, _ => ()
  | _ :: s, c, h => (⟨c 0, h 0 (Nat.succ_pos _)⟩,
      ofCoords s (fun k => c (k + 1)) (fun k hk => h (k + 1) (Nat.succ_lt_succ hk)))

theorem coord_ofCoords : ∀ (s : Shape) (c : Nat → Nat) (h : ∀ k, k < s.length → c k < Shape.dim s k)
    {k : Nat}, k < s.length → (ofCoords s c h).coord k = c k
  | [], _, _, _, hk => absurd hk (Nat.not_lt_zero _)
  | _ :: _, _, _, 0, _ => rfl
  | _ :: s, c, _, _ + 1, hk => coord_ofCoords s (fun k => c (k + 1)) _ (Nat.lt_of_succ_lt_succ hk)

/-- A coordinate beyond the rank is `0`. -/
theorem coord_of_ge : ∀ {s : Shape} (i : Idx s) {k : Nat}, s.length ≤ k → i.coord k = 0
  | [], (), _, _ => rfl
  | _ :: _, (_, _), 0, h => absurd h (by simp)
  | _ :: s, (_, j), k + 1, h => coord_of_ge (s := s) j (by simpa using h)

end Idx

/-- `getD` below the length is the entry. -/
theorem list_getD_of_lt {β : Type} {l : List β} {k : Nat} {d : β} (hk : k < l.length) :
    l.getD k d = l[k] := by
  simp [List.getD_eq_getElem?_getD, hk]

namespace Tensor

variable {s : Shape} {α : Type}

/-- The coordinates `permute` reads are in range. -/
theorem permute_coord_lt (dims : List Nat) (h : dims.Perm (List.range s.length))
    (o : Idx (dims.map (Shape.dim s))) :
    ∀ j, j < s.length → o.coord (dims.idxOf j) < Shape.dim s j := by
  intro j hj
  have hmem : j ∈ dims := h.mem_iff.mpr (List.mem_range.mpr hj)
  have hp : dims.idxOf j < dims.length := List.idxOf_lt_length_of_mem hmem
  have hlt := Idx.coord_lt o (k := dims.idxOf j) (by simpa using hp)
  rw [Shape.dim_eq_getD, list_getD_of_lt (by simpa using hp), List.getElem_map,
    List.getElem_idxOf] at hlt
  exact hlt

/-- `tl.permute(x, dims)` and `tl.trans(x, dims)` (`triton-3.8.0-core.py:2070-2121`,
`triton-3.8.0-semantic.py:701-709`): the result has shape `[s[dims[0]], s[dims[1]], …]`, and its
element at index `o` is the element of `x` whose coordinate `dims[k]` is `o.coord k`, for every
`k` (`get_permute`). `dims` must be a permutation of `0, …, rank − 1` (Triton rejects any other
list at compile time); for a shape of literal rank the default proof `h` is found by itself.

* A rank-2 `tl.trans(x)` or `x.T` is `Tensor.trans`, which equals `x.permute [1, 0]`
  (`permute_trans`).
* `tl.trans(x)` with no `dims` on a tensor of rank `r ≥ 3` swaps the last two axes
  (`triton-3.8.0-core.py:2091-2097`): `dims` is `List.range (r - 2) ++ [r - 1, r - 2]`, which the
  model writes as a literal list (`x.permute [0, 2, 1]` for rank 3, `[0, 1, 3, 2]` for rank 4),
  so that the result's shape reduces to the source's. -/
def permute (x : Tensor s α) (dims : List Nat)
    (h : dims.Perm (List.range s.length) := by
      (try simp only [List.length_cons, List.length_nil]); decide) :
    Tensor (dims.map (Shape.dim s)) α :=
  ⟨fun o => x.get (Idx.ofCoords s (fun j => o.coord (dims.idxOf j)) (permute_coord_lt dims h o))⟩

/-- The element of `x.permute dims` at `o` is `x` at the index `i` whose coordinate `dims[k]` is
`o.coord k` for every `k`. -/
theorem get_permute (x : Tensor s α) (dims : List Nat) (h : dims.Perm (List.range s.length))
    (o : Idx (dims.map (Shape.dim s))) (i : Idx s)
    (hi : ∀ k, k < dims.length → i.coord (dims.getD k 0) = o.coord k) :
    (x.permute dims h).get o = x.get i := by
  unfold permute
  refine congrArg x.get (Idx.ext_coord fun j => ?_)
  by_cases hj : j < s.length
  · rw [Idx.coord_ofCoords _ _ _ hj]
    have hmem : j ∈ dims := h.mem_iff.mpr (List.mem_range.mpr hj)
    have hp : dims.idxOf j < dims.length := List.idxOf_lt_length_of_mem hmem
    have := hi (dims.idxOf j) hp
    rw [list_getD_of_lt hp, List.getElem_idxOf] at this
    exact this.symm
  · rw [Idx.coord_of_ge _ (Nat.le_of_not_lt hj), Idx.coord_of_ge _ (Nat.le_of_not_lt hj)]

/-- On rank 2, `permute [1, 0]` is the transpose `Tensor.trans`. -/
theorem permute_trans {m n : Nat} (x : Tensor [m, n] α) : x.permute [1, 0] = x.trans := by
  ext ⟨j, i, ⟨⟩⟩
  exact (get_permute x [1, 0] _ (j, i, ()) (i, j, ()) fun k hk =>
    match k, hk with
    | 0, _ => rfl
    | 1, _ => rfl).trans rfl

end Tensor

/-! ## Broadcasting to a shape (library issue L140) -/

/-- Triton's check that a block of shape `s` broadcasts to `t` (`broadcast_impl_shape`,
`triton-3.8.0-semantic.py:711-725`): the same rank, and each dimension of `s` equal to `t`'s or
`1`. Triton rejects every other pair on the host. -/
def Shape.bcastOk : Shape → Shape → Bool
  | [], [] => true
  | a :: s, b :: t => (a == b || a == 1) && Shape.bcastOk s t
  | _, _ => false

theorem Shape.bcastOk_length : ∀ {s t : Shape}, Shape.bcastOk s t = true → s.length = t.length
  | [], [], _ => rfl
  | _ :: s, _ :: t, h => by
      simp only [Shape.bcastOk, Bool.and_eq_true] at h
      simp [Shape.bcastOk_length (s := s) (t := t) h.2]
  | [], _ :: _, h => by simp [Shape.bcastOk] at h
  | _ :: _, [], h => by simp [Shape.bcastOk] at h

theorem Shape.bcastOk_dim : ∀ {s t : Shape}, Shape.bcastOk s t = true → ∀ k,
    Shape.dim s k = Shape.dim t k ∨ Shape.dim s k = 1 ∨ s.length ≤ k
  | [], [], _, k => Or.inr (Or.inr (Nat.zero_le k))
  | a :: s, b :: t, h, k => by
      simp only [Shape.bcastOk, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
      cases k with
      | zero =>
        rcases h.1 with h1 | h1
        · exact Or.inl h1
        · exact Or.inr (Or.inl h1)
      | succ k =>
        rcases Shape.bcastOk_dim (s := s) (t := t) h.2 k with h2 | h2 | h2
        · exact Or.inl h2
        · exact Or.inr (Or.inl h2)
        · exact Or.inr (Or.inr (by simpa using h2))
  | [], _ :: _, h, _ => by simp [Shape.bcastOk] at h
  | _ :: _, [], h, _ => by simp [Shape.bcastOk] at h

namespace Tensor

variable {s : Shape} {α : Type}

/-- The coordinates `broadcastTo` reads are in range. -/
theorem bcast_coord_lt {t : Shape} (h : Shape.bcastOk s t = true) (o : Idx t) :
    ∀ k, k < s.length → (if Shape.dim s k = 1 then 0 else o.coord k) < Shape.dim s k := by
  intro k hk
  by_cases h1 : Shape.dim s k = 1
  · simp only [h1, ite_true]; exact Nat.zero_lt_one
  · simp only [h1, ite_false]
    rcases Shape.bcastOk_dim h k with h2 | h2 | h2
    · rw [h2]; exact Idx.coord_lt o (by rw [← Shape.bcastOk_length h]; exact hk)
    · exact absurd h2 h1
    · exact absurd hk (Nat.not_lt_of_le h2)

/-- `tl.broadcast_to(x, t)` and its method forms `x.broadcast_to(t)`, `x.broadcast_to(*t)`
(`triton-3.8.0-core.py:2049-2066`, `triton-3.8.0-semantic.py:711-725`) of a block `x` of shape `s`:
the block of shape `t` whose element at index `o` is `x`'s element at the index with `o`'s
coordinates, except that each dimension of size `1` in `s` is read at `0` (`get_broadcastTo`).
Triton accepts only a `t` of the same rank whose every dimension equals `s`'s or where `s` has `1`
(`Shape.bcastOk`, the proof `h`; for shapes of literal sizes the default proof is found by itself),
and rejects every other `t` on the host. A scalar's broadcast is `Tensor.full` (`splat`,
`semantic.py:629-634`). `col`, `row`, `bcastCols` and `bcastRows` are special cases. -/
def broadcastTo (x : Tensor s α) (t : Shape) (h : Shape.bcastOk s t = true := by decide) :
    Tensor t α :=
  ⟨fun o => x.get (Idx.ofCoords s (fun k => if Shape.dim s k = 1 then 0 else o.coord k)
    (bcast_coord_lt h o))⟩

/-- The element of `x.broadcastTo t` at `o` is `x` at the index `i` whose coordinate `k` is `0` where
`s` has size `1`, and `o`'s coordinate `k` otherwise. -/
theorem get_broadcastTo (x : Tensor s α) (t : Shape) (h : Shape.bcastOk s t = true) (o : Idx t)
    (i : Idx s) (hi : ∀ k, k < s.length → i.coord k = if Shape.dim s k = 1 then 0 else o.coord k) :
    (x.broadcastTo t h).get o = x.get i := by
  unfold broadcastTo
  refine congrArg x.get (Idx.ext_coord fun k => ?_)
  by_cases hk : k < s.length
  · rw [Idx.coord_ofCoords _ _ _ hk, hi k hk]
  · rw [Idx.coord_of_ge _ (Nat.le_of_not_lt hk), Idx.coord_of_ge _ (Nat.le_of_not_lt hk)]

/-- On rank 2, broadcasting a column `[m, 1]` to `[m, n]` is `bcastCols`. -/
theorem broadcastTo_cols {m n : Nat} (x : Tensor [m, 1] α) (h : Shape.bcastOk [m, 1] [m, n] = true) :
    x.broadcastTo [m, n] h = x.bcastCols n := by
  ext ⟨i, j, ⟨⟩⟩
  rw [get_broadcastTo x [m, n] h (i, j, ()) (i, ⟨0, Nat.zero_lt_one⟩, ()) fun k hk => ?_]
  · rfl
  · match k, hk with
    | 0, _ =>
      show i.1 = if m = 1 then 0 else i.1
      split
      · next hm => have := i.2; omega
      · rfl
    | 1, _ => rfl

/-- On rank 2, broadcasting a row `[1, n]` to `[m, n]` is `bcastRows`. -/
theorem broadcastTo_rows {m n : Nat} (x : Tensor [1, n] α) (h : Shape.bcastOk [1, n] [m, n] = true) :
    x.broadcastTo [m, n] h = x.bcastRows m := by
  ext ⟨i, j, ⟨⟩⟩
  rw [get_broadcastTo x [m, n] h (i, j, ()) (⟨0, Nat.zero_lt_one⟩, j, ()) fun k hk => ?_]
  · rfl
  · match k, hk with
    | 0, _ => rfl
    | 1, _ =>
      show j.1 = if n = 1 then 0 else j.1
      split
      · next hn => have := j.2; omega
      · rfl

end Tensor

end TVA
