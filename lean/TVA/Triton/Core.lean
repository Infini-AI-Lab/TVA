import TVA.Core

/-!
# The `tl.*` vocabulary: conventions, program ids, creation, shapes, hints

A claim's model transcribes a Triton kernel into Lean with the definitions of `TVA.Triton`,
named as in `triton.language` (`tl.load`, `tl.dot`, `tl.where`), so that the model reads like
the source, line by line, in `do` notation over `Prog` (`TVA.Program`). This file holds the
operations every model uses (program ids, `tl.arange`, `tl.full`, `tl.where`, shape changes) and
the compiler hints; the other families are in `TVA.Triton.Arith`, `Convert`, `Promote`, `Memory`,
`BlockPtr`, `Desc`, `Reduce`, `Scan`, `Dot`, `Random` and `Loop`.

## Conventions

* **Exact operations are pure Lean; the others are programs.** An operation whose result Triton
  defines exactly (integer arithmetic, comparisons, `tl.where`, shape changes, integer
  reductions) is a function on tensors (`TVA.Tensor`). An operation that rounds, that the device
  computes only up to an accuracy, that may be undefined, or that touches memory returns a
  `Prog`, and the model binds its result with `←`.
* **Implicit broadcasts and conversions are explicit.** Triton broadcasts operands
  (`x[:, None] + y[None, :]`, `offs < n`) and converts mixed dtypes (`TVA.Triton.Promote`)
  without saying so; a model writes each broadcast (`bc`, `.col`, `.row`, `tl.broadcast_to`) and
  each conversion (`tl.convert`, `tl.cvt`, `tl.itof`, …).
* **Integer arguments equal to `1` are compile-time constants.** Triton specialises an integer
  argument whose value is `1` (unless the kernel lists it in `do_not_specialize`): the kernel is
  compiled with the Python constant `1` in its place, not an `int32` tensor. A model takes such an
  argument as an `int32` value all the same when every use of it converts it to a tensor first (an
  arithmetic or a comparison with a tensor does), which gives the same result; where it does not,
  the claim's model says how it treats the value `1`.
* **Compile-time values are Lean values.** Block sizes and every other `tl.constexpr` are Lean
  numbers fixed by the claim's instance, and Python arithmetic on them is computed in Lean
  (`tl.py`). A shape is a `List Nat`.
* **What the device chooses is a `pick`.** When Triton leaves a result to the device (a masked-out
  lane without `other`, an approximate `tl.exp`, a `tl.dot`, the tree of a float reduction, the
  order of an atomic's lanes), the operation is `Prog.pick P`: the program continues with *some*
  value satisfying `P`, and a proof must work for each. When no value satisfies `P` the operation
  is undefined behaviour; that is how a model states a precondition (`tl.assume`, a division by
  zero, an out-of-range conversion).
* **NaN payloads are unspecified.** The device returns *some* NaN, not a particular one (PTX gives
  "an unspecified NaN" for `neg`, `abs` and arithmetic on NaN inputs, and MLIR folds such as
  `x * 1.0 → x` keep an input's payload). So every exact float operation returns its IEEE result
  up to NaN payload (`FloatV.NaNEq`, `tl.upToNaN`).
* **Fusion.** With `enable_fp_fusion` (Triton's default) the compiler may contract a product and
  a following addition into one `fma`. An addition, reduction or scan that a product may feed is
  written in its fusable form (`tl.addc`, `tl.mul_add`, `tl.sumc`, `tl.cumsumc`), any other in its
  IEEE form (`tl.add`, `tl.sum`, `tl.cumsum`); `TVA.Triton.Arith` says which values are products.
-/
namespace TVA

/-! ## Compiler hints: what a promise says

A hint is a promise to the compiler, which uses it to vectorise and align accesses and to compute
a value once per group of equal ones. A false promise is undefined behaviour, so the model keeps
each hint as a check (`core.py:3307-3382`, `semantic.py:1775-1791`): the hint is a `pick` that
has no result when the promise is false.

A hint promises something about the values of an integer or pointer tensor (`HintVal`: a
pointer's divisibility is counted in bytes, its contiguity in elements), along each dimension `k`:

* `ContigAlong x k C`: each group of `C` positions along `k` holds consecutive values;
* `ContigIs x k C`: `C` is exactly the contiguity (groups of `C` are consecutive, groups of `2·C`
  are not);
* `FirstsDiv x k C v`: `v` divides the first element of each group of `C` positions;
* `ConstAlong x k C`: each group of `C` positions holds equal values. -/

/-- The integer a compiler hint speaks about: an integer's value, or a pointer's byte address.
`unit` is the step between consecutive elements: `1` for an integer, the element size for a
pointer (Triton counts a pointer's contiguity in elements and its divisibility in bytes). -/
class HintVal (β : Type) where
  /-- The integer the hint constrains. -/
  val : β → Int
  /-- The difference between consecutive values. -/
  unit : Int

instance {sg : Bool} {w : Nat} : HintVal (IntV sg w) := ⟨IntV.toInt, 1⟩
instance {d : DType} : HintVal (Ptr d) := ⟨Ptr.addr, d.bytes⟩

/-- Positions `i` and `j` lie in the same group of `C` along dimension `k`: they agree on every
other coordinate, and their `k`-th coordinates have the same quotient by `C`. -/
def Idx.SameGroup {s : Shape} (k C : Nat) (i j : Idx s) : Prop :=
  (∀ k', k' ≠ k → i.coord k' = j.coord k') ∧ i.coord k / C = j.coord k / C

/-- Along dimension `k`, each group of `C` positions (`Idx.SameGroup`) holds consecutive values,
`HintVal.unit β` apart. -/
def ContigAlong {s : Shape} {β : Type} [HintVal β] (x : Tensor s β) (k C : Nat) : Prop :=
  ∀ i j : Idx s, Idx.SameGroup k C i j →
    HintVal.val (x.get j) = HintVal.val (x.get i) + ((j.coord k : Int) - i.coord k) * HintVal.unit β

/-- `C` is `x`'s contiguity along dimension `k`: each group of `C` positions holds consecutive
values and, unless `2·C` exceeds the dimension's size, the groups of `2·C` positions do not all
do so. -/
def ContigIs {s : Shape} {β : Type} [HintVal β] (x : Tensor s β) (k C : Nat) : Prop :=
  ContigAlong x k C ∧ (2 * C ≤ s.dim k → ¬ ContigAlong x k (2 * C))

/-- `v` divides the first element of each group of `C` positions along dimension `k` (the
elements whose coordinate `k` is a multiple of `C`); with `C = 1`, every element. -/
def FirstsDiv {s : Shape} {β : Type} [HintVal β] (x : Tensor s β) (k C v : Nat) : Prop :=
  ∀ i : Idx s, i.coord k % C = 0 → (v : Int) ∣ HintVal.val (x.get i)

/-- Along dimension `k`, each group of `C` positions holds equal values. -/
def ConstAlong {s : Shape} {β : Type} (x : Tensor s β) (k C : Nat) : Prop :=
  ∀ i j : Idx s, Idx.SameGroup k C i j → x.get j = x.get i

namespace tl

variable {s : Shape} {α : Type}

/-! ## Program identity -/

/-- `tl.program_id(axis)`: this program's index along a grid axis, an `int32`. -/
def program_id (c : Ctx) (axis : Fin 3) : I32 := IntV.ofNat (c.pid axis)

/-- `tl.num_programs(axis)`: the grid's size along an axis, an `int32`. -/
def num_programs (c : Ctx) (axis : Fin 3) : I32 := IntV.ofNat (c.grid axis)

/-! ## Creation and selection -/

/-- `tl.arange(start, end)`: the `int32` values `start, …, end - 1`. -/
def arange (start stop : Nat) : Tensor [stop - start] I32 := Tensor.arange start (stop - start)

/-- `tl.zeros(shape, dtype)`: the all-zero bit pattern (`+0.0`, `0`, `False`). -/
def zeros (s : Shape) (d : DType) : Tensor s d.Val := Tensor.full (d.ofBits 0)

/-- `tl.full(shape, value, dtype)`: `value` at every index. The model writes the constant Triton
builds for `value` in `dtype` (`Operand.constIn?`, `TVA.Triton.Promote`): in particular a zero,
`-0.0` included, becomes `+0.0` (`semantic.py:601-613`). -/
def full (s : Shape) {α : Type} (a : α) : Tensor s α := Tensor.full a

/-- `tl.where(cond, x, y)`: `x` where `cond` holds, `y` elsewhere. The model converts a
non-boolean condition with `tl.convert _ .i1` (`c != 0`) and the two values to their common dtype
(`whereTyping?`). -/
def «where» (c : Tensor s Bool) (x y : Tensor s α) : Tensor s α := Tensor.where_ c x y

/-! ## Shape changes

All exact; they are the operations of `TVA.Tensor.Layout` under Triton's names, so the `get_*`
lemmas of `TVA.Tensor` apply. -/

/-- `tl.broadcast_to(x, t)` of a block (`Tensor.broadcastTo`): each dimension of size `1` is
repeated; Triton accepts only targets that `Shape.bcastOk` admits. -/
abbrev broadcast_to (x : Tensor s α) (t : Shape) (h : Shape.bcastOk s t = true := by decide) :
    Tensor t α :=
  x.broadcastTo t h

/-- `tl.reshape(x, s')` with `can_reorder=False` (`Tensor.reshape`): row-major order kept. -/
abbrev reshape (x : Tensor s α) (s' : Shape) (h : s'.prod = s.prod) : Tensor s' α :=
  x.reshape s' h

/-- `tl.permute(x, dims)` and `tl.trans(x, dims)` (`Tensor.permute`): dimension `k` of the result
is dimension `dims[k]` of `x`. -/
abbrev permute (x : Tensor s α) (dims : List Nat)
    (h : dims.Perm (List.range s.length) := by
      (try simp only [List.length_cons, List.length_nil]); decide) :
    Tensor (dims.map (Shape.dim s)) α :=
  x.permute dims h

/-- `tl.trans(x)` and `x.T` of a rank-2 tensor (`Tensor.trans`). On rank `r ≥ 3`, `tl.trans(x)`
swaps the last two axes: `tl.permute x [0, …, r - 3, r - 1, r - 2]`. -/
abbrev trans {m n : Nat} (x : Tensor [m, n] α) : Tensor [n, m] α := x.trans

/-- `tl.join(a, b)` (`Tensor.join`): a new last dimension of size 2. -/
abbrev join (a b : Tensor s α) : Tensor (s ++ [2]) α := Tensor.join a b

/-- `tl.split(x)` (`Tensor.split`): the two halves along a last dimension of size 2. -/
abbrev split (x : Tensor (s ++ [2]) α) : Tensor s α × Tensor s α := x.split

/-- `tl.interleave(a, b)` (`Tensor.interleave`): `a` and `b` alternate along the last
dimension. -/
abbrev interleave {n : Nat} (a b : Tensor (s ++ [n]) α) : Tensor (s ++ [2 * n]) α :=
  Tensor.interleave a b

/-- `tl.reshape(x, s', can_reorder=True)` and `tl.view`: the same elements in an order the
compiler chooses. A reshape to rank 1 whose only users are reductions or histograms is also this,
since Triton's combine pass may then reorder it (and an index reduction of it pairs reordered
values with the positions' indices). -/
def reshape_any (x : Tensor s α) (s' : Shape) : Prog (Tensor s' α) :=
  .pick (fun r : Tensor s' α => ∃ e : Idx s' ≃ Idx s, ∀ i, r.get i = x.get (e i)) .ret

/-! ## Compile-time Python arithmetic

Python computes `tl.constexpr` arithmetic at compile time, with floor division; the model
computes it in Lean with these. On tensors (runtime values) `//` and `%` truncate instead
(`tl.idiv`, `tl.irem`). -/

namespace py

/-- Python's `x // y` on integers (floor division). -/
def floordiv (x y : Int) : Int := Int.fdiv x y

/-- Python's `x % y` on integers (the sign of `y`). -/
def mod (x y : Int) : Int := Int.fmod x y

/-- `tl.cdiv(x, div)` on two compile-time integers: Python's `(x + (div - 1)) // div`, the ceiling
of `x / div` for a positive `div` (`standard.py:32-43`). -/
def cdiv (x div : Int) : Int := Int.fdiv (x + (div - 1)) div

end py

/-! ## Assertions and hints

A hint marks the IR operation that defined its argument's value (`set_attr`, `ir.cc:411-425`),
and the compiler uses the mark at every use of that value, uses before the call and on paths that
never reach it included. So the model places the check where the hinted value is defined, before
any use, even when the source writes the hint later or inside a branch.

**Which value a hint marks.** The operation that defined its argument's value when the front end
visits the call. A value no operation defines takes no mark: a kernel's or a helper's parameter,
a loop's induction variable, and the value a loop carries inside its body. The front end visits a
loop's body once with every name bound to its value before the loop, so:
- a hint on a value the body defines checks each iteration's value;
- a hint on a name the loop carries checks its value before the (outermost such) loop, once, when
  an operation defined that value, and promises nothing about later iterations;
- a hint on a parameter, an induction variable or the result of an inlined `@triton.jit` call
  (every call but a `noinline=True` helper's, the `tl.*` functions written in Triton included:
  `tl.cdiv`, `tl.sum`, `tl.max`, `tl.zeros`, …) has no effect, and the model drops it;
- a hint on one result of an operation with several results (`tl.split`, a tuple `tl.reduce` or
  `tl.associative_scan`, a `noinline` helper returning a tuple, the names a loop or a runtime `if`
  carries) may apply to every result: the model checks each, where the operation defines them.

**What a hint promises**, along each dimension `k`, for the source's per-dimension values (one
value for rank 1); each hint replaces one part of what the compiler's analysis believes of the
value (`AxisInfo.cpp:1244-1259`):
- `tl.max_contiguous(x, C)` alone: `ContigIs x k C`. "Groups of `C` are consecutive" alone is not
  enough: with a `C` below `x`'s own contiguity the compiler keeps the divisibility it inferred for
  its own, longer groups.
- `tl.max_constancy(x, C)`: `ConstAlong x k C`.
- `tl.multiple_of(x, v)` alone: `FirstsDiv x k C v`, where `C` is the contiguity the compiler
  infers for `x`. The model may take for `C` any lower bound of it (below), and `1` otherwise,
  which makes it every element; a smaller `C` is a stronger promise.
- both on the same value, in either order: `ContigAlong x k C` and `FirstsDiv x k C v`
  (`tl.max_contiguous_multiple_of`).

**Lower bounds of the inferred contiguity** along `k`, for the value the hinted operation
defines: `b - a` for `tl.arange(a, b)`; for `x + y`, `y + x`, `x - y` (and pointer `p + x`) with
`y` constant along `k`, `x`'s bound; for `x[:, None]`, `x[None, :]`, `tl.expand_dims` and a
broadcast, the kept dimensions' bounds and `1` for the new or broadcast one; for an integer
conversion, `x`'s bound; for a value carrying a `max_contiguous` hint, the smaller of the hinted `C`
and the bound it would have without the hint; `1` for anything else. These readings are given for
values that are powers of two (at most the dimension's size for `max_contiguous` and
`max_constancy`). -/

/-- `tl.assume(cond)`: a promise to the compiler that `cond` holds. A false `assume` is undefined
behaviour (the compiler may have optimised on it). -/
def assume (c : Bool) : Prog Unit := .pick (fun _ : Unit => c = true) .ret

/-- `tl.static_assert(cond)`: Triton checks `cond` at compile time and rejects the kernel on the
host when it is false, so no program runs. A claim's `ok` excludes such instances; the model
makes a false one undefined behaviour, so that no claim covering it can be proved. -/
def static_assert (c : Bool) : Prog Unit := .pick (fun _ : Unit => c = true) .ret

/-- `tl.max_contiguous(x, C)` of a rank-1 tensor, with no `tl.multiple_of` on the same value:
`C` is `x`'s contiguity (`ContigIs`). -/
def max_contiguous {n : Nat} {β : Type} [HintVal β] (x : Tensor [n] β) (C : Nat) :
    Prog (Tensor [n] β) :=
  .pick (fun y => y = x ∧ ContigIs x 0 C) .ret

/-- `tl.max_constancy(x, C)` of a rank-1 tensor: `x` is runs of `C` equal values. -/
def max_constancy {n : Nat} {β : Type} (x : Tensor [n] β) (C : Nat) : Prog (Tensor [n] β) :=
  .pick (fun y => y = x ∧ ConstAlong x 0 C) .ret

/-- `tl.multiple_of(x, v)` of a rank-1 tensor, with no `tl.max_contiguous` on the same value: `v`
divides the first element of each group of `contig` positions (`FirstsDiv`). `contig` is a lower
bound of the contiguity the compiler infers for `x` (`B` for `s + tl.arange(0, B)`), and `1`, every
element, by default. -/
def multiple_of {n : Nat} {β : Type} [HintVal β] (x : Tensor [n] β) (v : Nat)
    (contig : Nat := 1) : Prog (Tensor [n] β) :=
  .pick (fun y => y = x ∧ FirstsDiv x 0 contig v) .ret

/-- `tl.max_contiguous(tl.multiple_of(x, v), C)` of a rank-1 tensor (both hints on one value, in
either order): groups of `C` positions hold consecutive values (`ContigAlong`) and `v` divides
the first element of each group (`FirstsDiv`). -/
def max_contiguous_multiple_of {n : Nat} {β : Type} [HintVal β] (x : Tensor [n] β) (C v : Nat) :
    Prog (Tensor [n] β) :=
  .pick (fun y => y = x ∧ ContigAlong x 0 C ∧ FirstsDiv x 0 C v) .ret

/-- `tl.multiple_of(x, v)` of a scalar: `v` divides it. -/
def multiple_of1 {β : Type} [HintVal β] (x : β) (v : Nat) : Prog β :=
  .pick (fun y => y = x ∧ (v : Int) ∣ HintVal.val x) .ret

/-- `tl.max_contiguous(x, Cs)` for any rank, with no `tl.multiple_of` on the same value: along
each dimension `k`, `Cs[k]` is `x`'s contiguity (`ContigIs`). -/
def max_contiguousN {β : Type} [HintVal β] (x : Tensor s β) (Cs : List Nat) : Prog (Tensor s β) :=
  .pick (fun y => y = x ∧ ∀ k, k < s.length → ContigIs x k (Cs.getD k 1)) .ret

/-- `tl.max_constancy(x, Cs)` for any rank: along each dimension `k`, groups of `Cs[k]` positions
hold equal values. -/
def max_constancyN {β : Type} (x : Tensor s β) (Cs : List Nat) : Prog (Tensor s β) :=
  .pick (fun y => y = x ∧ ∀ k, k < s.length → ConstAlong x k (Cs.getD k 1)) .ret

/-- `tl.multiple_of(x, vs)` for any rank, with no `tl.max_contiguous` on the same value: along each
dimension `k`, `vs[k]` divides the first element of each group of `contig[k]` positions, where
`contig[k]` is a lower bound of the inferred contiguity (`1` where none is given). So
`tl.multiple_of(ptrs, [16, 16])` on a 2-D block of row pointers (contiguity `1` along dimension `0`)
promises that every pointer is 16-byte aligned. -/
def multiple_ofN {β : Type} [HintVal β] (x : Tensor s β) (vs : List Nat) (contig : List Nat := []) :
    Prog (Tensor s β) :=
  .pick (fun y => y = x ∧ ∀ k, k < s.length → FirstsDiv x k (contig.getD k 1) (vs.getD k 1)) .ret

/-- `tl.max_contiguous(tl.multiple_of(x, vs), Cs)` for any rank (both hints on one value): along
each dimension `k`, groups of `Cs[k]` positions hold consecutive values and `vs[k]` divides the
first element of each group. -/
def max_contiguous_multiple_ofN {β : Type} [HintVal β] (x : Tensor s β) (Cs vs : List Nat) :
    Prog (Tensor s β) :=
  .pick (fun y => y = x ∧ ∀ k, k < s.length →
    ContigAlong x k (Cs.getD k 1) ∧ FirstsDiv x k (Cs.getD k 1) (vs.getD k 1)) .ret

end tl

end TVA
