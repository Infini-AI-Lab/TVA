import TVA.Tensor.Basic
import TVA.Tensor.Index
import TVA.Tensor.Layout
import TVA.Tensor.Tree

/-!
# Tensors

Triton's block values, as Lean values of type `Tensor s α` (`s` a static shape):

* `TVA.Tensor.Basic`: shapes, indices, `Tensor`, elementwise maps, rank-1 and rank-2 forms
  (`of1`, `get2`, `col`, `row`, `trans`), `tl.where`, integer arithmetic and comparisons, float
  comparisons, and the exact reductions `foldAll`, `foldRows`, `foldCols`;
* `TVA.Tensor.Index`: coordinates of an index along any axis, and row-major positions;
* `TVA.Tensor.Layout`: `tl.join`, `tl.split`, `tl.interleave`, `tl.reshape`, `tl.permute`,
  `tl.broadcast_to`;
* `TVA.Tensor.Tree`: the reduction trees of float reductions.

Next: `TVA.Memory` (pointers are tensors of `Ptr d`, and loads and stores move tensors).
-/
