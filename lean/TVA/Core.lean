import TVA.Base
import TVA.Tensor
import TVA.Memory
import TVA.Platform
import TVA.Program

/-!
# The core of TVA

What a model of a Triton kernel is built from, below the `tl.*` vocabulary:

* `TVA.Base`: values (`IntV`, `FloatV`, `DType`) and their byte encoding;
* `TVA.Tensor`: block values `Tensor s α` and their exact operations;
* `TVA.Memory`: global memory `Mem`, pointers `Ptr d`, buffers `Buf d`;
* `TVA.Platform`: the relations Triton leaves to the device, and the environment `Env`;
* `TVA.Program`: one program instance as a value `Prog α`.

Next: `TVA.Triton` (the `tl.*` operations, written with these) and `TVA.Launch` (how the
programs of a launch run together).
-/
