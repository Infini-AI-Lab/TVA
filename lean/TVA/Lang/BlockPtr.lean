import TVA.Lang.Ops

/-!
# Block pointers (`tl.make_block_ptr`, `tl.advance`)

Triton 3.8.0 deprecates block pointers in favour of tensor descriptors (`TVA.Lang.Desc`) but
still accepts them. Its frontend lowers a block pointer to ordinary pointer arithmetic and a
masked `tl.load`/`tl.store` (`core.py:1829-1930`, `_block_ptr._materialize`). This file defines
them the same way, so they add no semantics of their own:

* `make_block_ptr(base, shape, strides, offsets, block_shape, order)`: each entry of `shape`,
  `strides` and `offsets` is cast to `int64` (the model writes the casts, `castInt`). `order`
  describes the memory layout and does not change values;
* lane `i` of the block has coordinate `offsets[k] + i_k` along dimension `k` (`int64`,
  wrapping), and the pointer `base + Σₖ coordₖ · strides[k]` (each product `int64`, wrapping);
* `load(boundary_check, padding_option)`: a lane outside `[0, shape[k])` along a checked
  dimension `k` is masked off and holds what the padding says (`""` unspecified, `"zero"`,
  `"nan"`). Unchecked dimensions are not checked: such a lane accesses its address, which is
  undefined behaviour unless that memory is valid;
* `store(value, boundary_check)`: masked-off lanes are not written;
* `advance(offsets)`: adds to the offsets (`int64`).
-/
namespace TVA

/-- A block pointer. All lists have one entry per dimension of `block` (Triton checks it). -/
structure BlockPtr (d : DType) where
  base : Ptr d
  shape : List I64
  strides : List I64
  offsets : List I64
  block : Shape
  order : List Nat := []

namespace BlockPtr

variable {d : DType}

/-- The rank. -/
def rank (p : BlockPtr d) : Nat := p.block.length

/-- Triton's checks when it builds the block pointer (`_block_ptr.__init__`): rank at least
1 and one entry per dimension. -/
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

/-- `tl.advance(p, deltas)`. -/
def advance (p : BlockPtr d) (deltas : List I64) : BlockPtr d :=
  { p with offsets := List.zipWith (· + ·) p.offsets deltas }

/-- `tl.load(p, boundary_check=bc, padding_option=…)` of a block pointer. -/
def load_block (p : BlockPtr d) (bc : List Nat := []) (padding : Padding := .undef) :
    Prog (Tensor p.block d.Val) :=
  .pick (fun _ : Unit => p.Valid ∧ p.CheckOk bc) fun _ => do
    let v ← tl.load p.ptrs (some (p.mask bc))
    tl.pad padding (p.inb bc) v

/-- `tl.store(p, value, boundary_check=bc)` of a block pointer. The model converts `value` to the
element type first, as Triton does implicitly. -/
def store_block (p : BlockPtr d) (vals : Tensor p.block d.Val) (bc : List Nat := []) : Prog Unit :=
  .pick (fun _ : Unit => p.Valid ∧ p.CheckOk bc) fun _ => tl.store p.ptrs vals (some (p.mask bc))

end tl

end TVA
