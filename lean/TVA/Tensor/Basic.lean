import TVA.Base
import Mathlib.Data.List.Perm.Basic
import Mathlib.Data.Fintype.Basic

/-!
# Tensors

A Triton block value is a tensor of a static shape: every shape in a compiled kernel is a
compile-time constant, so a shape is a `List Nat`. `Tensor s α` holds an `α` at each index of
`Idx s`. A scalar is an `α`, or the rank-0 tensor `Tensor [] α` (`item`, `scalar`).

This file holds the *exact* operations: they neither round nor choose (construction, elementwise
maps, `tl.where`, integer arithmetic and comparisons, float comparisons, broadcasting of rank 1
and 2, `tl.trans`, and the integer reductions). Floating-point arithmetic rounds and may be left
to the device, so it is a program action in `TVA.Triton.Arith`.

Triton broadcasts implicitly (`x[:, None] + y[None, :]`, `offs < n`); a model broadcasts
explicitly. `bc a` is the scalar `a` at every index; `x.col n` is `x[:, None]` broadcast to `n`
columns; `x.row m` is `x[None, :]` broadcast to `m` rows. Broadcasting of any rank,
`tl.reshape`, `tl.permute` and `tl.join` are in `TVA.Tensor.Layout`; index coordinates in
`TVA.Tensor.Index`.

Each operation comes with a `@[simp]` lemma `get_*` (or `get1_*`, `get2_*`) that reads one
element of the result, so that a proof reasons element by element.
-/
namespace TVA

/-- A tensor shape: the sizes of the dimensions, outermost first (Triton's block shape). -/
abbrev Shape := List Nat

/-- The indices of a tensor of shape `s`: one `Fin` per dimension, ending in `()`. The element
`(i, j)` of a matrix is at `(i, j, ())`. -/
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

/-- Every index is listed by `all`. -/
theorem mem_all : (s : Shape) → (i : Idx s) → i ∈ all s
  | [], () => List.mem_singleton.mpr rfl
  | _ :: s, (i, j) =>
      List.mem_flatMap.mpr ⟨i, List.mem_finRange i, List.mem_map.mpr ⟨j, mem_all s j, rfl⟩⟩

/-- `all` lists each index once. -/
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

end Idx

/-- A tensor of shape `s` with elements in `α`: a Triton block (`tl.tensor` of a block type). -/
@[ext] structure Tensor (s : Shape) (α : Type) where
  /-- The element at an index. -/
  get : Idx s → α

/-- Two tensors of the same shape with the same element at every index are equal. -/
add_decl_doc Tensor.ext

namespace Tensor

variable {s : Shape} {α β γ δ : Type}

instance [Inhabited α] : Inhabited (Tensor s α) := ⟨⟨fun _ => default⟩⟩

/-- The tensor with every element `a`: `tl.full`, and the implicit broadcast of a scalar. -/
def full (a : α) : Tensor s α := ⟨fun _ => a⟩

/-- An elementwise unary operation. -/
def map (f : α → β) (x : Tensor s α) : Tensor s β := ⟨fun i => f (x.get i)⟩

/-- An elementwise binary operation on two tensors of the same shape. -/
def map2 (f : α → β → γ) (x : Tensor s α) (y : Tensor s β) : Tensor s γ :=
  ⟨fun i => f (x.get i) (y.get i)⟩

/-- An elementwise ternary operation (`tl.fma`). -/
def map3 (f : α → β → γ → δ) (x : Tensor s α) (y : Tensor s β) (z : Tensor s γ) : Tensor s δ :=
  ⟨fun i => f (x.get i) (y.get i) (z.get i)⟩

/-- The elements in row-major order. -/
def toList (x : Tensor s α) : List α := (Idx.all s).map x.get

/-- Every element of `tl.full` (or a broadcast scalar) is the fill value. -/
@[simp] theorem get_full (a : α) (i : Idx s) : (full a : Tensor s α).get i = a := rfl
/-- Reading a tensor built by `map` applies the function to the input at the same position. -/
@[simp] theorem get_map (f : α → β) (x : Tensor s α) (i : Idx s) :
    (x.map f).get i = f (x.get i) := rfl
/-- Reading a tensor built by `map2` reads the two tensors at the same position. -/
@[simp] theorem get_map2 (f : α → β → γ) (x : Tensor s α) (y : Tensor s β) (i : Idx s) :
    (map2 f x y).get i = f (x.get i) (y.get i) := rfl
/-- Reading a tensor built by `map3` reads the three tensors at the same position. -/
@[simp] theorem get_map3 (f : α → β → γ → δ) (x : Tensor s α) (y : Tensor s β) (z : Tensor s γ)
    (i : Idx s) : (map3 f x y z).get i = f (x.get i) (y.get i) (z.get i) := rfl

/-! ### Rank 0, 1 and 2 -/

/-- The value of a rank-0 tensor. -/
def item (x : Tensor [] α) : α := x.get ()

/-- The rank-0 tensor of a value. -/
def scalar (a : α) : Tensor [] α := ⟨fun _ => a⟩

/-- The rank-1 tensor with element `f i` at `i`. -/
def of1 {n : Nat} (f : Fin n → α) : Tensor [n] α := ⟨fun i => f i.1⟩

/-- The rank-2 tensor with element `f i j` at `(i, j)`. -/
def of2 {m n : Nat} (f : Fin m → Fin n → α) : Tensor [m, n] α := ⟨fun i => f i.1 i.2.1⟩

/-- The element `x[i]` of a rank-1 tensor. -/
def get1 {n : Nat} (x : Tensor [n] α) (i : Fin n) : α := x.get (i, ())

/-- The element `x[i, j]` of a rank-2 tensor. -/
def get2 {m n : Nat} (x : Tensor [m, n] α) (i : Fin m) (j : Fin n) : α := x.get (i, j, ())

/-- The value of the rank-0 tensor of `a` is `a`. -/
@[simp] theorem item_scalar (a : α) : (scalar a).item = a := rfl
/-- Element `i` of the rank-1 tensor `of1 f` is `f i`. -/
@[simp] theorem get1_of1 {n : Nat} (f : Fin n → α) (i : Fin n) : (of1 f).get1 i = f i := rfl
/-- Element `(i, j)` of the rank-2 tensor `of2 f` is `f i j`. -/
@[simp] theorem get2_of2 {m n : Nat} (f : Fin m → Fin n → α) (i : Fin m) (j : Fin n) :
    (of2 f).get2 i j = f i j := rfl

/-- `x[:, None]`, broadcast to `n` columns. -/
def col {m : Nat} (x : Tensor [m] α) (n : Nat) : Tensor [m, n] α := of2 (fun i _ => x.get1 i)

/-- `x[None, :]`, broadcast to `m` rows. -/
def row {n : Nat} (x : Tensor [n] α) (m : Nat) : Tensor [m, n] α := of2 (fun _ j => x.get1 j)

/-- `x[:, None]` without broadcasting: shape `[m, 1]`. -/
def unsqueeze1 {m : Nat} (x : Tensor [m] α) : Tensor [m, 1] α := of2 (fun i _ => x.get1 i)

/-- `x[None, :]` without broadcasting: shape `[1, n]`. -/
def unsqueeze0 {n : Nat} (x : Tensor [n] α) : Tensor [1, n] α := of2 (fun _ j => x.get1 j)

/-- The implicit broadcast of a `[m, 1]` tensor to `[m, n]`. -/
def bcastCols {m : Nat} (x : Tensor [m, 1] α) (n : Nat) : Tensor [m, n] α :=
  of2 (fun i _ => x.get2 i 0)

/-- The implicit broadcast of a `[1, n]` tensor to `[m, n]`. -/
def bcastRows {n : Nat} (x : Tensor [1, n] α) (m : Nat) : Tensor [m, n] α :=
  of2 (fun _ j => x.get2 0 j)

/-- `tl.trans(x)` (or `x.T`) of a rank-2 tensor. Other ranks, and explicit `dims`, are
`Tensor.permute` (`permute_trans`: this is `permute [1, 0]`). -/
def trans {m n : Nat} (x : Tensor [m, n] α) : Tensor [n, m] α := of2 (fun j i => x.get2 i j)

/-- Element `(i, j)` of `x[:, None]` broadcast to `n` columns is `x[i]`, whatever the column. -/
@[simp] theorem get2_col {m n : Nat} (x : Tensor [m] α) (i : Fin m) (j : Fin n) :
    (x.col n).get2 i j = x.get1 i := rfl
/-- Element `(i, j)` of `x[None, :]` broadcast to `m` rows is `x[j]`, whatever the row. -/
@[simp] theorem get2_row {m n : Nat} (x : Tensor [n] α) (i : Fin m) (j : Fin n) :
    (x.row m).get2 i j = x.get1 j := rfl
/-- Element `(i, 0)` of `x[:, None]` is `x[i]`. -/
@[simp] theorem get2_unsqueeze1 {m : Nat} (x : Tensor [m] α) (i : Fin m) (j : Fin 1) :
    x.unsqueeze1.get2 i j = x.get1 i := rfl
/-- Element `(0, j)` of `x[None, :]` is `x[j]`. -/
@[simp] theorem get2_unsqueeze0 {n : Nat} (x : Tensor [n] α) (i : Fin 1) (j : Fin n) :
    x.unsqueeze0.get2 i j = x.get1 j := rfl
/-- Every column of a `[m, 1]` tensor broadcast to `[m, n]` repeats its single column. -/
@[simp] theorem get2_bcastCols {m n : Nat} (x : Tensor [m, 1] α) (i : Fin m) (j : Fin n) :
    (x.bcastCols n).get2 i j = x.get2 i 0 := rfl
/-- Every row of a `[1, n]` tensor broadcast to `[m, n]` repeats its single row. -/
@[simp] theorem get2_bcastRows {m n : Nat} (x : Tensor [1, n] α) (i : Fin m) (j : Fin n) :
    (x.bcastRows m).get2 i j = x.get2 0 j := rfl
/-- Element `(j, i)` of `tl.trans(x)` is element `(i, j)` of `x`. -/
@[simp] theorem get2_trans {m n : Nat} (x : Tensor [m, n] α) (j : Fin n) (i : Fin m) :
    x.trans.get2 j i = x.get2 i j := rfl

/-! The same reads in the `x.get (i, j, ())` form, which pointer arithmetic and the memory
operations produce. -/

/-- `get1_of1` in the `x.get (i, ())` form: element `i` of `of1 f` is `f i`. -/
@[simp] theorem get_of1 {n : Nat} (f : Fin n → α) (i : Fin n) : (of1 f).get (i, ()) = f i := rfl
/-- `get2_of2` in the `x.get (i, j, ())` form: element `(i, j)` of `of2 f` is `f i j`. -/
@[simp] theorem get_of2 {m n : Nat} (f : Fin m → Fin n → α) (i : Fin m) (j : Fin n) :
    (of2 f).get (i, j, ()) = f i j := rfl
/-- `get2_col` in the `x.get (i, j, ())` form: `x[:, None]` broadcast to columns reads `x[i]`. -/
@[simp] theorem get_col {m n : Nat} (x : Tensor [m] α) (i : Fin m) (j : Fin n) :
    (x.col n).get (i, j, ()) = x.get (i, ()) := rfl
/-- `get2_row` in the `x.get (i, j, ())` form: `x[None, :]` broadcast to `m` rows reads `x[j]`. -/
@[simp] theorem get_row {m n : Nat} (x : Tensor [n] α) (i : Fin m) (j : Fin n) :
    (x.row m).get (i, j, ()) = x.get (j, ()) := rfl

/-! ### Construction and selection -/

/-- `tl.arange(start, start + n)`: `start, start + 1, …` as `tl.int32`. Triton requires `n` to
be a power of two (`semantic.py:594`); the model of a compiled kernel satisfies that. -/
def arange (start : Int) (n : Nat) : Tensor [n] I32 := of1 (fun i => IntV.ofInt (start + i.val))

/-- Lane `i` of `tl.arange(start, start + n)` holds `start + i` as a `tl.int32`. -/
@[simp] theorem get1_arange (start : Int) (n : Nat) (i : Fin n) :
    (arange start n).get1 i = IntV.ofInt (start + i.val) := rfl
/-- `get1_arange` in the `x.get (i, ())` form: lane `i` of `tl.arange` holds `start + i`. -/
@[simp] theorem get_arange (start : Int) (n : Nat) (i : Fin n) :
    (arange start n).get (i, ()) = IntV.ofInt (start + i.val) := rfl

/-- `tl.where(c, x, y)`: both operands are computed; each element is taken from `x` where `c`
holds and from `y` elsewhere. -/
def where_ (c : Tensor s Bool) (x y : Tensor s α) : Tensor s α :=
  ⟨fun i => if c.get i then x.get i else y.get i⟩

/-- Each element of `tl.where(c, x, y)` is `x`'s where `c` holds and `y`'s elsewhere. -/
@[simp] theorem get_where (c : Tensor s Bool) (x y : Tensor s α) (i : Idx s) :
    (where_ c x y).get i = if c.get i then x.get i else y.get i := rfl

/-! ### Integer arithmetic

`+ - *` and unary `-` on integer tensors wrap in the element type (`IntV`). A scalar operand is
broadcast (`block_start + offs`). -/

section IntOps
variable {sg : Bool} {w : Nat}

/-- `x + y` on integer tensors, elementwise and wrapping. -/
instance : Add (Tensor s (IntV sg w)) := ⟨map2 (· + ·)⟩
/-- `x - y` on integer tensors, elementwise and wrapping. -/
instance : Sub (Tensor s (IntV sg w)) := ⟨map2 (· - ·)⟩
/-- `x * y` on integer tensors, elementwise and wrapping. -/
instance : Mul (Tensor s (IntV sg w)) := ⟨map2 (· * ·)⟩
/-- `-x` on an integer tensor, elementwise and wrapping. -/
instance : Neg (Tensor s (IntV sg w)) := ⟨map (- ·)⟩

/-- `a + x` for a scalar `a`. -/
instance : HAdd (IntV sg w) (Tensor s (IntV sg w)) (Tensor s (IntV sg w)) :=
  ⟨fun a x => x.map (a + ·)⟩
/-- `x + a` for a scalar `a`. -/
instance : HAdd (Tensor s (IntV sg w)) (IntV sg w) (Tensor s (IntV sg w)) :=
  ⟨fun x a => x.map (· + a)⟩
/-- `a * x` for a scalar `a`. -/
instance : HMul (IntV sg w) (Tensor s (IntV sg w)) (Tensor s (IntV sg w)) :=
  ⟨fun a x => x.map (a * ·)⟩
/-- `x * a` for a scalar `a`. -/
instance : HMul (Tensor s (IntV sg w)) (IntV sg w) (Tensor s (IntV sg w)) :=
  ⟨fun x a => x.map (· * a)⟩

/-- An element of `x + y` on integer tensors is the wrapping sum of the two elements. -/
@[simp] theorem get_add (x y : Tensor s (IntV sg w)) (i : Idx s) :
    (x + y).get i = x.get i + y.get i := rfl
/-- An element of `x - y` on integer tensors is the wrapping difference of the two elements. -/
@[simp] theorem get_sub (x y : Tensor s (IntV sg w)) (i : Idx s) :
    (x - y).get i = x.get i - y.get i := rfl
/-- An element of `x * y` on integer tensors is the wrapping product of the two elements. -/
@[simp] theorem get_mul (x y : Tensor s (IntV sg w)) (i : Idx s) :
    (x * y).get i = x.get i * y.get i := rfl
/-- An element of `-x` on an integer tensor is the wrapping negation of the element. -/
@[simp] theorem get_neg (x : Tensor s (IntV sg w)) (i : Idx s) : (-x).get i = -x.get i := rfl
/-- An element of `a + x` (scalar plus tensor) is `a` plus the element, wrapping. -/
@[simp] theorem get_sadd (a : IntV sg w) (x : Tensor s (IntV sg w)) (i : Idx s) :
    (a + x).get i = a + x.get i := rfl
/-- An element of `x + a` (as in `offs + block_start`) is the element plus `a`, wrapping. -/
@[simp] theorem get_adds (x : Tensor s (IntV sg w)) (a : IntV sg w) (i : Idx s) :
    (x + a).get i = x.get i + a := rfl
/-- An element of `a * x` (scalar times tensor) is `a` times the element, wrapping. -/
@[simp] theorem get_smul (a : IntV sg w) (x : Tensor s (IntV sg w)) (i : Idx s) :
    (a * x).get i = a * x.get i := rfl
/-- An element of `x * a` (as in `offs * stride`) is the element times `a`, wrapping. -/
@[simp] theorem get_muls (x : Tensor s (IntV sg w)) (a : IntV sg w) (i : Idx s) :
    (x * a).get i = x.get i * a := rfl

/-- `x.to(d')` of an integer tensor into another integer dtype other than `tl.int1`: each
element sign-extended (signed source) or zero-extended, then truncated (`IntV.cast`,
`semantic.py:870-880`). The model also uses it for its own widening steps (before an integer
reduction). `x.to(tl.int1)` is `x != 0` (`semantic.py:874-887`), not this. -/
def castInt {sg' : Bool} {w' : Nat} (x : Tensor s (IntV sg w)) : Tensor s (IntV sg' w') :=
  map IntV.cast x

/-- An element of an integer tensor cast with `x.to(...)` is the element cast by `IntV.cast`. -/
@[simp] theorem get_castInt {sg' : Bool} {w' : Nat} (x : Tensor s (IntV sg w)) (i : Idx s) :
    (x.castInt : Tensor s (IntV sg' w')).get i = IntV.cast (x.get i) := rfl

/-! ### Comparisons -/

/-- `x < y` on integer tensors, signed or unsigned by the dtype. -/
def lt (x y : Tensor s (IntV sg w)) : Tensor s Bool := map2 IntV.lt x y
/-- `x < a` for a scalar `a` (the usual mask `offs < n`). -/
def ltS (x : Tensor s (IntV sg w)) (a : IntV sg w) : Tensor s Bool := x.map (IntV.lt · a)
/-- `x <= y` on integer tensors, signed or unsigned by the dtype. -/
def le (x y : Tensor s (IntV sg w)) : Tensor s Bool := map2 IntV.le x y
/-- `x == y` on integer tensors. -/
def eq (x y : Tensor s (IntV sg w)) : Tensor s Bool := map2 IntV.eq x y

/-- An element of `x < y` on integer tensors compares the two elements. -/
@[simp] theorem get_lt (x y : Tensor s (IntV sg w)) (i : Idx s) :
    (x.lt y).get i = IntV.lt (x.get i) (y.get i) := rfl
/-- An element of the mask `x < a` (e.g. `offs < n`) compares the element with the scalar `a`. -/
@[simp] theorem get_ltS (x : Tensor s (IntV sg w)) (a : IntV sg w) (i : Idx s) :
    (x.ltS a).get i = IntV.lt (x.get i) a := rfl
/-- An element of `x <= y` on integer tensors compares the two elements. -/
@[simp] theorem get_le (x y : Tensor s (IntV sg w)) (i : Idx s) :
    (x.le y).get i = IntV.le (x.get i) (y.get i) := rfl
/-- An element of `x == y` on integer tensors tests the two elements for equality. -/
@[simp] theorem get_eq (x y : Tensor s (IntV sg w)) (i : Idx s) :
    (x.eq y).get i = IntV.eq (x.get i) (y.get i) := rfl

end IntOps

section FloatCmp
variable {f : Format}

/-- `x < y` on float tensors: false when either is NaN. -/
def flt (x y : Tensor s (FloatV f)) : Tensor s Bool := map2 FloatV.lt x y
/-- `x <= y` on float tensors: false when either is NaN. -/
def fle (x y : Tensor s (FloatV f)) : Tensor s Bool := map2 FloatV.le x y
/-- `x == y` on float tensors: false when either is NaN; `-0 == +0`. -/
def feq (x y : Tensor s (FloatV f)) : Tensor s Bool := map2 FloatV.eq x y
/-- `x != y` on float tensors: the negation of `feq` (true when either is NaN). -/
def fne (x y : Tensor s (FloatV f)) : Tensor s Bool := map2 FloatV.ne x y

/-- An element of `x < y` on float tensors compares the two elements (false on NaN). -/
@[simp] theorem get_flt (x y : Tensor s (FloatV f)) (i : Idx s) :
    (x.flt y).get i = FloatV.lt (x.get i) (y.get i) := rfl
/-- An element of `x <= y` on float tensors compares the two elements (false on NaN). -/
@[simp] theorem get_fle (x y : Tensor s (FloatV f)) (i : Idx s) :
    (x.fle y).get i = FloatV.le (x.get i) (y.get i) := rfl
/-- An element of `x == y` on float tensors compares the two elements (false on NaN, `-0 == +0`). -/
@[simp] theorem get_feq (x y : Tensor s (FloatV f)) (i : Idx s) :
    (x.feq y).get i = FloatV.eq (x.get i) (y.get i) := rfl
/-- An element of `x != y` on float tensors compares the two elements (true on NaN). -/
@[simp] theorem get_fne (x y : Tensor s (FloatV f)) (i : Idx s) :
    (x.fne y).get i = FloatV.ne (x.get i) (y.get i) := rfl

end FloatCmp

/-- `x & y` on boolean tensors. -/
def and (x y : Tensor s Bool) : Tensor s Bool := map2 (· && ·) x y
/-- `x | y` on boolean tensors. -/
def or (x y : Tensor s Bool) : Tensor s Bool := map2 (· || ·) x y
/-- `~x` on a boolean tensor. -/
def not (x : Tensor s Bool) : Tensor s Bool := map (! ·) x

/-- An element of the mask `x & y` is the `and` of the two elements. -/
@[simp] theorem get_and (x y : Tensor s Bool) (i : Idx s) :
    (x.and y).get i = (x.get i && y.get i) := rfl
/-- An element of the mask `x | y` is the `or` of the two elements. -/
@[simp] theorem get_or (x y : Tensor s Bool) (i : Idx s) :
    (x.or y).get i = (x.get i || y.get i) := rfl
/-- An element of the mask `~x` is the negation of the element. -/
@[simp] theorem get_not (x : Tensor s Bool) (i : Idx s) : x.not.get i = !x.get i := rfl

/-! ### Exact reductions

Reductions whose result does not depend on the order of combination: wrapping integer `+`,
boolean `and`/`or`, integer `max`/`min`. Floating-point reductions combine along a tree the
device chooses (`TVA.Tensor.Tree`, `TVA.Triton.Reduce`).

Triton widens small integers before these reductions, and the model writes the widening:
* `tl.max` and `tl.min` without `return_indices` convert an integer tensor narrower than 32 bits
  to `int32`, whatever its signedness (`standard.py:184-190, 233-239`): `int8` and `int16` are
  sign-extended, `uint8` and `uint16` zero-extended, and `tl.int1` becomes `0`/`1`. The model
  converts first (`castInt` to `I32`; a `Bool` tensor mapped to `0`/`1`) and reduces with the
  `int32` `IntV.max`/`IntV.min`.
* `tl.sum` converts an integer tensor narrower than 32 bits to `int32` if it is signed and to
  `uint32` if it is unsigned, `tl.int1` included (`_pick_sum_dtype`, `standard.py:254-283`). The
  sum wraps in that type.
* `tl.max`/`tl.min` with `return_indices`, and `tl.argmax`/`tl.argmin`, do not widen; they are in
  `TVA.Triton.Reduce`. -/

/-- Combine all elements with `op` from `init`, in row-major order. For an integer `tl.max`,
`tl.min` or `tl.sum`, `x` is the widened tensor (see above). -/
def foldAll (op : β → α → β) (init : β) (x : Tensor s α) : β := x.toList.foldl op init

/-- A rank-2 reduction along axis 1: each row combined with `op` from `init`, left to right. -/
def foldRows {m n : Nat} (op : β → α → β) (init : β) (x : Tensor [m, n] α) : Tensor [m] β :=
  of1 (fun i => (List.finRange n).foldl (fun acc j => op acc (x.get2 i j)) init)

/-- A rank-2 reduction along axis 0: each column combined with `op` from `init`, top to
bottom. -/
def foldCols {m n : Nat} (op : β → α → β) (init : β) (x : Tensor [m, n] α) : Tensor [n] β :=
  of1 (fun j => (List.finRange m).foldl (fun acc i => op acc (x.get2 i j)) init)

end Tensor

/-- `bc a`: the scalar `a` broadcast to the shape the context needs (the `n` in `offs < n`). -/
abbrev bc {s : Shape} {α : Type} (a : α) : Tensor s α := Tensor.full a

end TVA
