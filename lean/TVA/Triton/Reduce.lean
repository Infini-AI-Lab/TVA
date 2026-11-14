import TVA.Triton.Arith
import TVA.Triton.Reduce.Rel

/-!
# Reductions: `tl.sum`, `tl.max`, `tl.argmax`, `tl.histogram`

**Float reductions** combine their elements along a tree Triton chooses (`Tree`, `Tree.Eval`): the
tree covers the axis's `n` positions in some order, and its depth is at most
`tl.sumBound P n = max (P.sumDepth n) ⌈log₂ n⌉` on the launch's platform (the device facts state
sm_90's depth). As for `tl.add`, the model says whether a product may feed the additions: `tl.sum`
(`sum1`, `sum0`) for inputs that are not products, every node an IEEE addition; `tl.sumc`
(`sum1c`, `sum0c`) for a sum of products (`tl.sum(a * b)`), every node `AddRel` with the launch's
fusion flag. `tl.max`/`tl.min` combine with `FloatV.MaxRel` along some tree.

**Integer reductions** do not depend on the order: they are the folds of `TVA.Tensor`
(`Tensor.foldAll`, `foldRows`, `foldCols`).

**Triton's promotions** are written in the model: `tl.max`/`tl.min` (without `return_indices`)
widen floats narrower than 32 bits to `float32` (`standard.py:178-190`); `tl.sum` widens no float;
the index reductions (`return_indices`, `tl.argmax`, `tl.argmin`) widen only `bfloat16`, to
`float32`. A reduction keeps the source's axis; with `keep_dims=True` the reduced axis stays, of
size 1 (the model reshapes).
-/
namespace TVA

namespace tl

variable {s : Shape}

/-! ## Float reductions -/

section Float
variable {f : Format}

/-- The largest depth a float reduction's tree over `n` positions may have on `P`. -/
def sumBound (P : Platform) (n : Nat) : Nat := max (P.sumDepth n) (Nat.clog 2 n)

/-- A whole rank-1 reduction along some tree, each node an `AddRel fusion` addition. -/
def sumWith (fusion : Bool) {n : Nat} (x : Tensor [n] (FloatV f)) : Prog (FloatV f) :=
  .env fun e => .pick (.fsum f fusion (sumBound e.P n) n x) .ret

/-- Row reductions of a rank-2 tensor (`axis=1`), each along its own tree. -/
def sum1With (fusion : Bool) {m n : Nat} (x : Tensor [m, n] (FloatV f)) :
    Prog (Tensor [m] (FloatV f)) :=
  .env fun e => .pick (.fsum1 f fusion (sumBound e.P n) m n x) .ret

/-- Column reductions of a rank-2 tensor (`axis=0`), each along its own tree. -/
def sum0With (fusion : Bool) {m n : Nat} (x : Tensor [m, n] (FloatV f)) :
    Prog (Tensor [n] (FloatV f)) :=
  .env fun e => .pick (.fsum0 f fusion (sumBound e.P m) m n x) .ret

/-- `tl.sum(x)` of a rank-1 float tensor that is not a product: IEEE additions along a tree, in the
input's own format. -/
def sum {n : Nat} (x : Tensor [n] (FloatV f)) : Prog (FloatV f) := sumWith false x
/-- `tl.sum(x, axis=1)` of a rank-2 float tensor, IEEE additions. -/
def sum1 {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) :=
  sum1With false x
/-- `tl.sum(x, axis=0)` of a rank-2 float tensor, IEEE additions. -/
def sum0 {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) :=
  sum0With false x

/-- `tl.sum(x)` of a product: the additions may be fused. -/
def sumc {n : Nat} (x : Tensor [n] (FloatV f)) : Prog (FloatV f) := .env fun e => sumWith e.fusion x
/-- `tl.sum(x, axis=1)` of a product. -/
def sum1c {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) :=
  .env fun e => sum1With e.fusion x
/-- `tl.sum(x, axis=0)` of a product. -/
def sum0c {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) :=
  .env fun e => sum0With e.fusion x

/-- `tl.max(x, axis=1)` (`isMax`) or `tl.min(x, axis=1)` of a rank-2 float tensor: each row
combined with `MaxRel` along a tree. -/
def max1 {m n : Nat} (isMax : Bool) (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) :=
  .pick (.fmax1 f isMax m n x) .ret

/-- `tl.max(x)` (`isMax`) or `tl.min(x)` of a rank-1 float tensor: the whole tensor combined with
`MaxRel` along a tree. -/
def maxAll {n : Nat} (isMax : Bool) (x : Tensor [n] (FloatV f)) : Prog (FloatV f) :=
  .pick (.fmax f isMax n x) .ret

/-- `tl.max(x, axis=0)` (`isMax`) or `tl.min(x, axis=0)` of a rank-2 float tensor: each column
combined with `MaxRel` along a tree. -/
def max0 {m n : Nat} (isMax : Bool) (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) :=
  .pick (.fmax0 f isMax m n x) .ret

end Float

end tl

/-! ## Index reductions

`tl.max(x, axis, return_indices=True)`, `tl.min(…, return_indices=True)`, `tl.argmax` and
`tl.argmin` reduce `(value, index)` pairs along an axis with Triton's `_argmax_combine` and
`_argmin_combine` (`standard.py:146-155, 199-208`), along a tree Triton chooses. The index is an
`int32` `tl.arange` along the axis (`core.py:3093-3107`). `CmpKey`, `tl.argCombine` and `tl.ArgRes`
are in `TVA.Triton.Reduce.Rel`, before the catalog of sites. -/

namespace tl

variable {s : Shape} {α : Type}

/-- `tl.max(x, 0, return_indices=True, return_indices_tie_break_left=tieLeft)` (`isMax`) or
`tl.min(…)` of a rank-1 tensor: a pair `ArgRes` allows. `tl.argmax(x, 0, tie_break_left=tieLeft)`
(`tl.argmin`) is its index. -/
def argmaxAll [CmpKey α] [Inhabited α] {n : Nat} (isMax tieLeft : Bool) (x : Tensor [n] α) :
    Prog (α × I32) :=
  .pick (.arg α isMax tieLeft n x) .ret

/-- `tl.max(x, 1, return_indices=True)` (`isMax`) or `tl.min(x, 1, …)` of a rank-2 tensor: for each
row, a pair `ArgRes` allows; `tl.argmax(x, 1)` is the index tensor. -/
def argmax1 [CmpKey α] [Inhabited α] {m n : Nat} (isMax tieLeft : Bool) (x : Tensor [m, n] α) :
    Prog (Tensor [m] α × Tensor [m] I32) :=
  .pick (.arg1 α isMax tieLeft m n x) .ret

/-- `tl.max(x, 0, return_indices=True)` (`isMax`) or `tl.min(x, 0, …)` of a rank-2 tensor: for each
column, a pair `ArgRes` allows; `tl.argmax(x, 0)` is the index tensor. -/
def argmax0 [CmpKey α] [Inhabited α] {m n : Nat} (isMax tieLeft : Bool) (x : Tensor [m, n] α) :
    Prog (Tensor [n] α × Tensor [n] I32) :=
  .pick (.arg0 α isMax tieLeft m n x) .ret

/-! ## Histograms

`tl.enabledBy` is in `TVA.Triton.Reduce.Rel`. -/

/-- `tl.histogram(x, bins, mask)` (`semantic.py:1764-1774`): bin `b` counts the enabled elements
equal to `b`, an `int32`. An enabled element outside `[0, bins)` is undefined behaviour (the
documentation does not say what it does). -/
def histogram {n : Nat} {sg : Bool} {w : Nat} (x : Tensor [n] (IntV sg w)) (bins : Nat)
    (mask : Option (Tensor [n] Bool) := none) : Prog (Tensor [bins] I32) :=
  .pick (.histogram n bins sg w x mask) .ret

end tl

end TVA
