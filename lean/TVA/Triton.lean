import TVA.Triton.Core
import TVA.Triton.Loop
import TVA.Triton.Arith
import TVA.Triton.Convert
import TVA.Triton.Promote
import TVA.Triton.Memory
import TVA.Triton.BlockPtr
import TVA.Triton.Desc
import TVA.Triton.Reduce
import TVA.Triton.Scan
import TVA.Triton.Dot
import TVA.Triton.Random

/-!
# The `tl.*` vocabulary

Everything a claim's model is written in, one file per family; start with `TVA.Triton.Core`, whose
module docstring explains the conventions (exact operations are Lean functions, the others
programs; explicit broadcasts and conversions; `pick` for what the device chooses; NaN payloads;
fusion).

| file | Triton |
|---|---|
| `Core` | `tl.program_id`, `tl.num_programs`, `tl.arange`, `tl.full`, `tl.zeros`, `tl.where`, shape changes, `tl.assume`, `tl.static_assert`, `tl.multiple_of`, `tl.max_contiguous`, compile-time arithmetic (`tl.py`) |
| `Loop` | `range`, `tl.range`, `while` (`tl.forRange`, `tl.whileLoop`, `tl.forN`) |
| `Arith` | float `+ - * /`, `tl.fma`, `tl.exp` and the other math functions, `tl.maximum`, `tl.sigmoid`, integer `// % << >>`, `tl.cdiv` |
| `Convert` | `x.to(d)`, `tl.cast`, bit casts, `tl.convert`, `tl.cvt`, `tl.itof` |
| `Promote` | the dtype an operator computes in; the constant a Python number becomes |
| `Memory` | `tl.load`, `tl.store`, the atomics |
| `BlockPtr` | `tl.make_block_ptr`, `tl.advance`, their loads and stores |
| `Desc` | tensor descriptors (`desc.load`, `desc.store`, `desc.atomic_add`, …) |
| `Reduce` | `tl.sum`, `tl.max`, `tl.argmax`, `tl.histogram` |
| `Scan` | `tl.associative_scan`, `tl.cumsum`, `tl.cumprod`, `tl.sort` |
| `Dot` | `tl.dot`, `tl.dot_scaled` |
| `Random` | `tl.randint`, `tl.rand` |

Exact operations on tensors (integer arithmetic, comparisons, layouts) are in `TVA.Tensor`; what a
program's effects mean, and when a launch is well behaved, is `TVA.Launch`'s.
-/
