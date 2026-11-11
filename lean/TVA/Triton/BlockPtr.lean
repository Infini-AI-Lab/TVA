import TVA.Triton.Memory

/-!
# Block pointers (`tl.make_block_ptr`, `tl.advance`)

Triton deprecates block pointers in favour of tensor descriptors (`TVA.Triton.Desc`) but still
accepts them. Its front end lowers a block pointer to pointer arithmetic and a masked
`tl.load`/`tl.store` (`core.py:1829-1930`, `_block_ptr._materialize`), and these definitions do
the same, so block pointers add no semantics of their own:

* `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` casts each entry of
  `shape`, `strides` and `offsets` to `int64` (the model writes the casts); `order` describes the
  memory layout and does not change values;
* lane `i` of the block has coordinate `offsets[k] + i_k` along dimension `k` (`int64`, wrapping)
  and the pointer `base + Σₖ coordₖ · strides[k]` (each product `int64`, wrapping);
* `tl.load(p, boundary_check, padding_option)`: a lane outside `[0, shape[k])` along a checked
  dimension `k` is masked off and holds the padding (`""` unspecified, `"zero"`, `"nan"`).
  Unchecked dimensions are not checked: such a lane accesses its address;
* `tl.store(p, value, boundary_check)`: masked-off lanes are not written;
* `tl.advance(p, offsets)` adds to the offsets.

These are a block pointer's whole meaning: it is Triton's own aggregate, and `tl.load`,
`tl.store` and `tl.advance` call exactly these methods. An operator or subscript applied to a block
pointer has no meaning here.
-/
namespace TVA

/-- A block pointer: base, the tensor's shape, strides and the block's offsets (`int64`, one
entry per dimension of `block`, which Triton checks), and the block shape. -/
structure BlockPtr (d : DType) where
  /-- The base pointer. -/
  base : Ptr d
  /-- The tensor's shape. -/
  shape : List I64
  /-- The tensor's strides, in elements. -/
  strides : List I64
  /-- The block's offsets. -/
  offsets : List I64
  /-- The block shape. -/
  block : Shape
  /-- `order`: the memory layout, which changes no value. -/
  order : List Nat := []

namespace BlockPtr

variable {d : DType}

/-- The rank. -/
def rank (p : BlockPtr d) : Nat := p.block.length

/-- Triton's checks when it builds the block pointer (`_block_ptr.__init__`): rank at least 1 and
one entry per dimension. -/
def Valid (p : BlockPtr d) : Prop :=
  0 < p.rank ∧ p.shape.length = p.rank ∧ p.strides.length = p.rank ∧ p.offsets.length = p.rank

/-- A `boundary_check` tuple: distinct dimensions of the block. -/
def CheckOk (p : BlockPtr d) (bc : List Nat) : Prop := bc.Nodup ∧ ∀ k ∈ bc, k < p.rank

/-- Lane `i`'s coordinate along dimension `k`: `offsets[k] + i_k` in `int64`. -/
def coord (p : BlockPtr d) (i : Idx p.block) (k : Nat) : I64 :=
  p.offsets.getD k 0 + IntV.ofNat (i.coord k)

/-- The element lane `i` addresses, relative to `base`. -/
def elem (p : BlockPtr d) (i : Idx p.block) : Int :=
  ((List.range p.rank).map fun k => (p.coord i k * p.strides.getD k 0).toInt).sum

/-- The pointers of the block. -/
def ptrs (p : BlockPtr d) : Tensor p.block (Ptr d) := ⟨fun i => p.base.offset (p.elem i)⟩

/-- Lane `i` passes the checks of the dimensions `bc`. -/
def inb (p : BlockPtr d) (bc : List Nat) (i : Idx p.block) : Bool :=
  bc.all fun k => decide (0 ≤ (p.coord i k).toInt ∧ (p.coord i k).toInt < (p.shape.getD k 0).toInt)

/-- The load/store mask of `boundary_check = bc`. -/
def mask (p : BlockPtr d) (bc : List Nat) : Tensor p.block Bool := ⟨p.inb bc⟩

end BlockPtr

namespace tl

variable {d : DType}

/-- `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)`. -/
def make_block_ptr (base : Ptr d) (shape strides offsets : List I64) (block : Shape)
    (order : List Nat) : BlockPtr d :=
  ⟨base, shape, strides, offsets, block, order⟩

/-- `tl.advance(p, deltas)`: the offsets moved by `deltas` (`int64`). -/
def advance (p : BlockPtr d) (deltas : List I64) : BlockPtr d :=
  { p with offsets := List.zipWith (· + ·) p.offsets deltas }

/-- `tl.load(p, boundary_check=bc, padding_option=…)` of a block pointer: a masked load, then the
padding on the lanes outside the checked bounds. -/
def load_block (p : BlockPtr d) (bc : List Nat := []) (padding : Padding := .undef) :
    Prog (Tensor p.block d.Val) :=
  .pick (fun _ : Unit => p.Valid ∧ p.CheckOk bc) fun _ => do
    let v ← tl.load p.ptrs (some (p.mask bc))
    tl.pad padding (p.inb bc) v

/-- `tl.store(p, value, boundary_check=bc)` of a block pointer: a masked store. The model converts
`value` to the element dtype first, as Triton does implicitly. -/
def store_block (p : BlockPtr d) (vals : Tensor p.block d.Val) (bc : List Nat := []) : Prog Unit :=
  .pick (fun _ : Unit => p.Valid ∧ p.CheckOk bc) fun _ => tl.store p.ptrs vals (some (p.mask bc))

end tl

end TVA
