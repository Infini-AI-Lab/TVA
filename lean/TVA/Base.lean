import TVA.Base.Bytes
import TVA.Base.Int
import TVA.Base.Float
import TVA.Base.DType

/-!
# Values

The values a Triton kernel computes with, and how they sit in memory:

* `TVA.Base.Bytes`: bytes and the little-endian encoding of an element.
* `TVA.Base.Int`: `IntV signed w`, Triton's fixed-width integers (`I8` … `I64`, `U8` … `U64`),
  with LLVM's wrapping arithmetic, truncating `//` and `%`, shifts, bitwise operations and
  comparisons.
* `TVA.Base.Float`: float formats, rounding, `FloatV f` with the IEEE operations (`F16`, `BF16`,
  `F32`, `F64`, `F8E5`, `F8E4`), and the constants Triton builds for Python numbers.
* `TVA.Base.DType`: Triton's dtypes, the Lean type of their values (`DType.Val`), their memory
  encoding (`bytes`, `toBits`, `ofBits`) and exact values (`val?`).

Next: `TVA.Tensor` (tensors of these values) and `TVA.Memory` (memory of bytes).
-/
