import TVA.Lang.Ops

/-!
# Tensor descriptors (TMA)

`desc = tl.make_tensor_descriptor(base, shape, strides, block_shape, padding_option)`, or a
host-side `TensorDescriptor` argument, describes a tensor of rank 1 to 5 in global memory. On
NVIDIA GPUs with TMA its accesses are TMA copies. Their asynchrony and barriers are the
compiler's business, and we trust it.

At the language level (`core.py:1424-1484, 2647-2668`, `semantic.py:1065-1190, 1851-1890`):

* lane `i` of a block at `offsets` is element `offsets + i` of the tensor, at
  `base + Σₖ (offsetsₖ + iₖ) · stridesₖ`;
* `desc.load(offsets)`: lanes inside the tensor read memory; lanes outside hold the
  descriptor's padding (`"zero"`, the default, or `"nan"`, float dtypes only);
* `desc.store(offsets, value)`: lanes outside the tensor are dropped;
* `desc.atomic_add/min/max/and/or/xor(offsets, value)`: each lane inside the tensor is one
  atomic reduction of its element. The TMA reduction (`cp.reduce.async.bulk`) is *relaxed* at
  `gpu` scope, so two programs reducing into the same element race in the semantics
  (`Event.Strong`). Within a program, or for elements one program owns, they are covered. The
  allowed dtypes are Triton's (`RedOp.Allowed`). A float `add` rounds as the platform's
  `P.atomicAdd`;
* `desc.gather(x_offsets, y_offset)` (2-D, block `[1, BN]`): output row `i` is tensor row
  `x_offsets[i]`, columns `y_offset + j`. The documentation does not say what an out-of-bounds
  lane holds, so it is unspecified;
* `desc.scatter(value, x_offsets, y_offset)`: the inverse. The documentation does not say what
  happens to an out-of-bounds lane, so it is undefined behaviour here;
* the documented requirements: the base is 16-byte aligned, every stride but the last is a
  multiple of 16 bytes, the last stride is 1, and the offset is a multiple of 16 bytes (read
  here as the innermost offset, in bytes). A violation is undefined behaviour in the model.

Descriptor accesses are modelled as ordinary sequential accesses. That needs Triton to order
pointer and TMA accesses to the same bytes (`fence.proxy.async`), which is not yet confirmed
(`docs/review-questions.md` Q11). Until then, a model must not access the same bytes both
ways. A device-side descriptor occupies scratch memory from `triton.set_allocator`; the model
assumes it does not overlap the kernel's buffers (the reviewer checks the allocator).
-/
namespace TVA

/-- A tensor descriptor: element type `d`, base pointer, shape and strides (in elements), the
`block_shape` fixed when it is made (every access has this block, `semantic.py:1075, 1082`), and
its `padding_option`. -/
structure TensorDesc (d : DType) where
  base : Ptr d
  shape : List Nat
  strides : List Nat
  block : Shape
  padding : Padding := .zero

namespace TensorDesc

variable {d : DType}

/-- The rank. -/
def rank (D : TensorDesc d) : Nat := D.shape.length

/-- Triton's requirements on the descriptor. The shape entries are `int32`
(`make_tensor_descriptor` casts them, `semantic.py:1872`). -/
def Valid (D : TensorDesc d) : Prop :=
  1 ≤ D.rank ∧ D.rank ≤ 5 ∧ D.strides.length = D.rank ∧ D.block.length = D.rank ∧
    (∀ k, k < D.rank → D.shape.getD k 0 < 2 ^ 31) ∧ D.base.addr % 16 = 0 ∧
    D.strides.getLast? = some 1 ∧ (∀ k, k + 1 < D.rank → (D.strides.getD k 0 * d.bytes) % 16 = 0) ∧
    D.padding ≠ .undef ∧ (D.padding = .nan → d.format?.isSome = true)

/-- A position given by its coordinates `c k` lies inside the tensor. -/
def inbAt (D : TensorDesc d) (c : Nat → Int) : Bool :=
  decide (∀ k, k < D.rank → 0 ≤ c k ∧ c k < D.shape.getD k 0)

/-- The element at coordinates `c`, relative to the base. -/
def elemAt (D : TensorDesc d) (c : Nat → Int) : Int :=
  ((List.range D.rank).map fun k => c k * D.strides.getD k 0).sum

/-- The pointers of a block whose lane `i` has coordinates `c i`. -/
def ptrsAt {bs : Shape} (D : TensorDesc d) (c : Idx bs → Nat → Int) : Tensor bs (Ptr d) :=
  ⟨fun i => D.base.offset (D.elemAt (c i))⟩

/-- The mask of the lanes inside the tensor. -/
def maskAt {bs : Shape} (D : TensorDesc d) (c : Idx bs → Nat → Int) : Tensor bs Bool :=
  ⟨fun i => D.inbAt (c i)⟩

/-- The coordinates of lane `i` of the block at `offs`. -/
def coord (offs : List Int) {bs : Shape} (i : Idx bs) (k : Nat) : Int := offs.getD k 0 + i.coord k

/-- The requirements of one block access at `offs` with block shape `bs`: the descriptor's
block. The innermost offset is a multiple of 16 bytes (documented for `load`/`store`,
`core.py:1430, 1440`; required here of every access, a conservative reading). -/
def Access (D : TensorDesc d) (offs : List Int) (bs : Shape) : Prop :=
  D.Valid ∧ bs = D.block ∧ bs.length = D.rank ∧ offs.length = D.rank ∧
    (offs.getLast?.getD 0 * d.bytes) % 16 = 0

/-- The coordinates of lane `(i, j)` of a gather or scatter: row `xs[i]`, column `y + j`. -/
def rowCoord {M BN : Nat} {w : Nat} (xs : Tensor [M] (IntV true w)) (y : I32) (l : Idx [M, BN])
    (k : Nat) : Int :=
  if k = 0 then (xs.get1 l.1).toInt else y.toInt + l.2.1.val

/-- The requirements of a gather or scatter of `BN` columns at column `y`: a 2-D descriptor
with block `[1, BN]` (`semantic.py:1150-1151`), and `y` a multiple of 16 bytes (a conservative
reading, as in `Access`). -/
def RowAccess (D : TensorDesc d) (BN : Nat) (y : I32) : Prop :=
  D.Valid ∧ D.rank = 2 ∧ D.block = [1, BN] ∧ (y.toInt * d.bytes) % 16 = 0

end TensorDesc

/-- The reductions of `desc.atomic_*`. -/
inductive RedOp where
  | add | min | max | and | or | xor
  deriving DecidableEq, Repr

namespace RedOp

/-- The dtypes Triton allows (`semantic.py:1092-1141`). -/
def Allowed : RedOp → DType → Prop
  | .add, d => d ∈ [DType.u32, .i32, .u64, .f32, .f16, .bf16]
  | .min, d | .max, d => d ∈ [DType.u32, .i32, .u64, .i64, .f16, .bf16]
  | .and, d | .or, d | .xor, d => d ∈ [DType.u32, .i32, .u64, .i64]

/-- The new value of an element under a float `min`/`max` reduction: as `tl.maximum`, or a NaN
when an operand is NaN (the hardware may propagate it). -/
def fMinMax {f : TVA.Math.Float.Format} (isMax : Bool) (v o n : FloatV f) : Prop :=
  FloatV.MaxRel isMax o v n ∨ ((o.isNaN = true ∨ v.isNaN = true) ∧ n.isNaN = true)

/-- The new value `n` of an element holding `o` reduced with `v`: `Upd e op d v o n`. -/
def Upd (e : Env) : RedOp → (d : DType) → d.Val → d.Val → d.Val → Prop
  | .add, d => tl.addUpd e d
  | .min, .f16 => fMinMax false | .min, .bf16 => fMinMax false
  | .max, .f16 => fMinMax true | .max, .bf16 => fMinMax true
  | .min, d => tl.maxUpd false d
  | .max, d => tl.maxUpd true d
  | .and, d => tl.bitUpd IntV.and d
  | .or, d => tl.bitUpd IntV.or d
  | .xor, d => tl.bitUpd IntV.xor d

end RedOp

namespace tl

variable {d : DType}

/-- `desc.load(offsets)`: the block at `offsets`; lanes outside the tensor hold the padding. -/
def desc_load {bs : Shape} (D : TensorDesc d) (offsets : List I32) : Prog (Tensor bs d.Val) :=
  let o := offsets.map IntV.toInt
  .pick (fun _ : Unit => D.Access o bs) fun _ => do
    let v ← tl.load (D.ptrsAt (TensorDesc.coord o)) (some (D.maskAt (TensorDesc.coord o)))
    tl.pad D.padding (fun i => D.inbAt (TensorDesc.coord o i)) v

/-- `desc.store(offsets, value)`: the block at `offsets`; lanes outside the tensor are dropped.
The model converts `value` to the descriptor's dtype first, as Triton does implicitly. -/
def desc_store {bs : Shape} (D : TensorDesc d) (offsets : List I32) (vals : Tensor bs d.Val) :
    Prog Unit :=
  let o := offsets.map IntV.toInt
  .pick (fun _ : Unit => D.Access o bs) fun _ =>
    tl.store (D.ptrsAt (TensorDesc.coord o)) vals (some (D.maskAt (TensorDesc.coord o)))

/-- `desc.atomic_add(offsets, value)` etc.: one relaxed, `gpu`-scope reduction per lane inside
the tensor, in an unspecified order; no result. -/
def desc_atomic {bs : Shape} (op : RedOp) (D : TensorDesc d) (offsets : List I32)
    (vals : Tensor bs d.Val) : Prog Unit :=
  let o := offsets.map IntV.toInt
  .pick (fun _ : Unit => D.Access o bs ∧ op.Allowed d) fun _ =>
  .env fun e => do
    let _ ← tl.atomicWith (D.ptrsAt (TensorDesc.coord o)) (fun i => op.Upd e d (vals.get i))
      (some (D.maskAt (TensorDesc.coord o))) .relaxed .gpu
    return ()

/-- `desc.gather(x_offsets, y_offset)` with block `[1, BN]`: rows `x_offsets`, columns
`y_offset + [0, BN)`. Lanes outside the tensor hold unspecified values. -/
def desc_gather {M BN : Nat} {w : Nat} (D : TensorDesc d) (xs : Tensor [M] (IntV true w)) (y : I32) :
    Prog (Tensor [M, BN] d.Val) :=
  .pick (fun _ : Unit => D.RowAccess BN y) fun _ =>
    tl.load (D.ptrsAt (TensorDesc.rowCoord xs y)) (some (D.maskAt (TensorDesc.rowCoord xs y)))

/-- `desc.scatter(value, x_offsets, y_offset)`: the inverse of `desc_gather`. Every lane must
lie inside the tensor. Rows may repeat; the elements are then written in an unspecified order. -/
def desc_scatter {M BN : Nat} {w : Nat} (D : TensorDesc d) (vals : Tensor [M, BN] d.Val)
    (xs : Tensor [M] (IntV true w)) (y : I32) : Prog Unit :=
  .pick (fun _ : Unit => D.RowAccess BN y ∧ ∀ l : Idx [M, BN], D.inbAt (TensorDesc.rowCoord xs y l) = true)
    fun _ =>
    tl.store (D.ptrsAt (TensorDesc.rowCoord xs y)) vals

end tl

end TVA
