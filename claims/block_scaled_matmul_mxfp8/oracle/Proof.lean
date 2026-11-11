import Claim
import Proof.Main

/-!
# `block_scaled_matmul_mxfp8`: the proof

Program `i` of the grid computes one `128 × 256` tile of `c` from every input element; the tiles are
disjoint, and `correct_of_safe` composes the programs.

1. `Proof.Basic`, `Proof.Foot`: the grid, each program's footprints `R` (the inputs), `W` (its tile of
   `c`) and its post-condition `Q`; the tiles are disjoint.
2. `Proof.Off`, `Proof.ScaleOff`, `Proof.ScaleIdx`, `Proof.Fill`, `Proof.Body`: the loads of a trip
   through the descriptors read the scaled values of the inputs, including the 5-D scale layout.
3. `Proof.Tol`, `Proof.Num`, `Proof.Step`, `Proof.Tail`, `Proof.Inv`: the loop's invariant, the
   accumulator within the tolerance `tol` of the partial sum after each trip, from the device facts
   of `tl.dot_scaled`.
4. `Proof.Loop`, `Proof.Walk`: the loop, the `float16` rounding and the store; `prog_safe` is one program.
5. `Proof.Main`, `main`: `correct_of_safe` with `prog_safe`, `prog_disj`, `prog_rw` and `prog_post`.
-/
