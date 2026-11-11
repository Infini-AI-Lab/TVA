import TVA.Triton.Memory
import TVA.Triton.Arith

/-!
# Tensor descriptors (TMA)

A host-built `TensorDescriptor` argument describes a tensor of rank 1 to 5 in global memory, with
the host's shape, strides, block shape and padding. Its accesses are TMA copies; their asynchrony
and barriers are the compiler's, which TVA trusts. At the language level (`core.py:1424-1484`,
`semantic.py:1065-1190`):

* lane `i` of a block at `offsets` is element `offsets + i` of the tensor, at
  `base + Σₖ (offsetsₖ + iₖ) · stridesₖ`;
* `desc.load(offsets)`: lanes inside the tensor read memory; lanes outside hold the padding
  (`"zero"`, the default, or `"nan"`);
* `desc.store(offsets, value)`: lanes outside the tensor are dropped;
* `desc.atomic_add/min/max/and/or/xor(offsets, value)`: one atomic reduction per lane inside the
  tensor, *relaxed* at `gpu` scope (`cp.reduce.async.bulk`), so two programs reducing into one
  element race;
* `desc.gather(x_offsets, y_offset)` / `desc.scatter` (2-D, block `[1, BN]`): rows `x_offsets[i]`,
  columns `y_offset + j`. They do not compile for the H200 (`.tile::gather4` needs sm_100), so a
  launch that uses one is rejected on the host; the definitions state the documentation's reading
  for later targets.

Undefined behaviour unless Triton's requirements hold (`TensorDesc.Access`): rank 1–5; positive
shape entries below `2^31` (an access through a zero shape entry faults on the H200); a 16-byte
aligned base; leading strides that are multiples of 16 bytes and a last stride of `1`; an
innermost offset that is a multiple of 16 bytes (documented for load and store, required of every
access here).

Descriptor accesses are steps like pointer accesses and are not ordered with the program's other
accesses (`TVA.Program`). A model does not access the same bytes both through a descriptor and
through pointers.

A descriptor built in the kernel (`tl.make_tensor_descriptor`) is outside every claim: on the
device each program writes it into a scratch buffer outside the claim's memory.
`tl.make_tensor_descriptor` here is the call's meaning in the language only.
-/
namespace TVA

/-- A tensor descriptor: element dtype `d`, base pointer, shape and strides (in elements), the
`block_shape` every access uses (`semantic.py:1075, 1082`), and its `padding_option`. -/
structure TensorDesc (d : DType) where
  /-- The base pointer. -/
  base : Ptr d
  /-- The tensor's shape. -/
  shape : List Nat
  /-- The tensor's strides, in elements. -/
  strides : List Nat
  /-- The block shape. -/
  block : Shape
  /-- What a load gives outside the tensor. -/
  padding : Padding := .zero

namespace TensorDesc

variable {d : DType}

/-- The rank. -/
def rank (D : TensorDesc d) : Nat := D.shape.length

/-- Triton's requirements on the descriptor an access goes through: rank 1–5, one stride and one
block dimension per dimension, positive `int32` shape entries, a 16-byte-aligned base, a last
stride of `1` and the others multiples of 16 bytes (`0` included), and a defined padding (`nan`
only for floats). -/
def Valid (D : TensorDesc d) : Prop :=
  1 ≤ D.rank ∧ D.rank ≤ 5 ∧ D.strides.length = D.rank ∧ D.block.length = D.rank ∧
    (∀ k, k < D.rank → 0 < D.shape.getD k 0 ∧ D.shape.getD k 0 < 2 ^ 31) ∧ D.base.addr % 16 = 0 ∧
    D.strides.getLast? = some 1 ∧ (∀ k, k + 1 < D.rank → (D.strides.getD k 0 * d.bytes) % 16 = 0) ∧
    D.padding ≠ .undef ∧ (D.padding = .nan → d.format?.isSome = true)

/-- The position with coordinates `c k` lies inside the tensor. -/
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

/-- The requirements of one block access at `offs` with block shape `bs`: a valid descriptor, its
own block, and an innermost offset that is a multiple of 16 bytes. -/
def Access (D : TensorDesc d) (offs : List Int) (bs : Shape) : Prop :=
  D.Valid ∧ bs = D.block ∧ bs.length = D.rank ∧ offs.length = D.rank ∧
    (offs.getLast?.getD 0 * d.bytes) % 16 = 0

/-- The coordinates of lane `(i, j)` of a gather or scatter: row `xs[i]`, column `y + j`. -/
def rowCoord {M BN : Nat} {w : Nat} (xs : Tensor [M] (IntV true w)) (y : I32) (l : Idx [M, BN])
    (k : Nat) : Int :=
  if k = 0 then (xs.get1 l.1).toInt else y.toInt + l.2.1.val

/-- The requirements of a gather or scatter of `BN` columns at column `y`: a valid 2-D descriptor
with block `[1, BN]` (`semantic.py:1150-1151`), and `y` a multiple of 16 bytes. -/
def RowAccess (D : TensorDesc d) (BN : Nat) (y : I32) : Prop :=
  D.Valid ∧ D.rank = 2 ∧ D.block = [1, BN] ∧ (y.toInt * d.bytes) % 16 = 0

end TensorDesc

/-- The reductions of `desc.atomic_*`. -/
inductive RedOp where
  | add | min | max | and | or | xor
  deriving DecidableEq, Repr

namespace RedOp

/-- The dtypes Triton allows for each reduction (`semantic.py:1092-1141`). -/
def Allowed : RedOp → DType → Prop
  | .add, d => d ∈ [DType.u32, .i32, .u64, .f32, .f16, .bf16]
  | .min, d | .max, d => d ∈ [DType.u32, .i32, .u64, .i64, .f16, .bf16]
  | .and, d | .or, d | .xor, d => d ∈ [DType.u32, .i32, .u64, .i64]

/-- The new value of an element under a float `min`/`max` reduction: as `tl.maximum`, or a NaN
when an operand is NaN (the hardware may propagate it). -/
def fMinMax {f : Format} (isMax : Bool) (v o n : FloatV f) : Prop :=
  FloatV.MaxRel isMax o v n ∨ ((o.isNaN = true ∨ v.isNaN = true) ∧ n.isNaN = true)

/-- `Upd e op d v o n`: the new value `n` of an element holding `o` reduced with `v`. -/
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

/-- `tl.make_tensor_descriptor(base, shape, strides, block_shape, padding_option)` in the kernel
(`semantic.py:1851-1890`), outside every claim (see the module docstring): the descriptor of the
arguments' values, shape entries `int32` and strides `int64`. Undefined behaviour for a negative
shape entry or stride; an access through a zero shape entry is undefined (`TensorDesc.Valid`). -/
def make_tensor_descriptor (base : Ptr d) (shape : List I32) (strides : List I64) (block : Shape)
    (padding : Padding := .zero) : Prog (TensorDesc d) :=
  .pick (fun D : TensorDesc d => (∀ x ∈ shape, 0 ≤ x.toInt) ∧ (∀ x ∈ strides, 0 ≤ x.toInt) ∧
    D = ⟨base, shape.map (fun x => x.toInt.toNat), strides.map (fun x => x.toInt.toNat), block,
      padding⟩) .ret

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

/-- `desc.atomic_add(offsets, value)` and the other reductions: one relaxed, `gpu`-scope reduction
per lane inside the tensor, in an unspecified order; no result. -/
def desc_atomic {bs : Shape} (op : RedOp) (D : TensorDesc d) (offsets : List I32)
    (vals : Tensor bs d.Val) : Prog Unit :=
  let o := offsets.map IntV.toInt
  .pick (fun _ : Unit => D.Access o bs ∧ op.Allowed d) fun _ =>
  .env fun e => do
    let _ ← tl.atomicWith (D.ptrsAt (TensorDesc.coord o)) (fun i => op.Upd e d (vals.get i))
      (some (D.maskAt (TensorDesc.coord o))) .relaxed .gpu
    return ()

/-- `desc.gather(x_offsets, y_offset)` with block `[1, BN]`: rows `x_offsets`, columns
`y_offset + [0, BN)`; lanes outside the tensor hold unspecified values. -/
def desc_gather {M BN : Nat} {w : Nat} (D : TensorDesc d) (xs : Tensor [M] (IntV true w)) (y : I32) :
    Prog (Tensor [M, BN] d.Val) :=
  .pick (fun _ : Unit => D.RowAccess BN y) fun _ =>
    tl.load (D.ptrsAt (TensorDesc.rowCoord xs y)) (some (D.maskAt (TensorDesc.rowCoord xs y)))

/-- `desc.scatter(value, x_offsets, y_offset)`: the inverse of `desc_gather`; every lane must lie
inside the tensor (the documentation does not say what happens otherwise). Repeated rows are
written in an unspecified order. -/
def desc_scatter {M BN : Nat} {w : Nat} (D : TensorDesc d) (vals : Tensor [M, BN] d.Val)
    (xs : Tensor [M] (IntV true w)) (y : I32) : Prog Unit :=
  .pick (fun _ : Unit => D.RowAccess BN y ∧ ∀ l : Idx [M, BN], D.inbAt (TensorDesc.rowCoord xs y l) = true)
    fun _ =>
    tl.store (D.ptrsAt (TensorDesc.rowCoord xs y)) vals

end tl

end TVA
