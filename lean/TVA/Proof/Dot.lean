import TVA.Proof.Dot.Loop
import TVA.Proof.Dot.Lift
import TVA.Proof.Dot.Chunk
import TVA.Proof.Dot.Chunk8
import TVA.Proof.Dot.Chunk16
import TVA.Proof.Dot.Scaled
import TVA.Proof.Dot.Sm90

/-!
# The `tl.dot` accumulation kit

A model's `tl.dot` takes its result from the platform (`TVA.Triton.Dot`), and the rules
`safe_dot`, `safe_dot_add`, `safe_dot_scaled_add` hand a proof every result the platform allows.
How close that result is to the exact product is what the device facts say (`TVA.Device`), each
in the shape it was measured in. This kit turns them into bounds on the patterns kernels use.

* `TVA.Proof.Dot.Loop`: `acc += tl.dot(a, b)` over `K` blocks. One trip (`DotLoop.dotAdd_step`,
  either form of `tl.dot_add`) from a whole-dot fact (`Device.DotAcc`), and the loop
  (`DotLoop.accum`) against the claim's tolerance `DotLoop.bound` (`TVA.Device.Tolerance`), with
  its closed form `DotLoop.bound_le`.
* `TVA.Proof.Dot.Lift`: the same trip from a fact relative to lifted magnitudes
  (`Device.DotLiftAcc`: `Device.DotF16`, `Device.DotBF16`), `DotLift.dotAdd_step`, and lifted
  against plain magnitudes.
* `TVA.Proof.Dot.Chunk`: the arithmetic of a dot added chunk by chunk (`Dot8Chunk.errB`,
  `Dot8Chunk.errM`, `chain_err`, `chain_errM`).
* `TVA.Proof.Dot.Chunk8`: an fp8 dot, chunks of 32 (`Device.DotF8ChunkLoss.err`,
  `Device.DotF8Chunk.liftAcc`).
* `TVA.Proof.Dot.Chunk16`: a 16-bit dot, chunks of 64 (`Device.Dot16BitChunk.err`,
  `Device.Dot16BitChunk.liftAcc`).
* `TVA.Proof.Dot.Scaled`: `acc += tl.dot_scaled(…)` (`Device.DotScaledAcc.addStep`) and the
  exact values of scaled elements.
* `TVA.Proof.Dot.Sm90`: the device facts in the shapes a proof wants, from `h : Device P`
  (`h.dotE5m2Lift`, `h.dotF16Acc`, `h.dotI8_eq`), and `sum_bcast_mul_err`.

A GEMM proof: `safe_dot_add` in the loop body, `DotLoop.dotAdd_step` (or `DotLift.dotAdd_step`
with `h.dotF16 p`, `h.dotE5m2Lift p`, …) for each output element, and `DotLoop.accum` across the
loop.
-/
