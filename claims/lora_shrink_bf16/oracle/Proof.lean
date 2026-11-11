import Claim
import TVA.Proof
import Proof.Main

/-!
# The proof of `lora_shrink_bf16`

A launch's programs meet only through the `float32` atomic additions into the output, so the
proof is `correct_of_safeA` (shared cells), not `correct_of_safe`.

1. Each program is `SafeA` alone (`Prog.safe_program`) with read footprint `Rall` (the input
   buffers), no private writes and the output elements as shared cells (`Cells`). The walk covers
   the grouped program ids, the adapter group and its early exits, the token gather, the weights'
   address, the pointer tiles and the loop over `K` (`Loop.LoopInv`: every accumulator lane is a
   `LaneInv` value, bounded two ways against the exact partial sums by `Trip.lane_trip` from the
   device's `tl.dot` facts), and the end (`Epilogue`): the product with `scaling` is a `PartOk`
   operand and the atomic adds the tile into its cells, leaving the log `Q`.
2. `Pid`: each output element of a token with an adapter is added to by exactly `SPLIT_K`
   programs, one per part of `K`; the program ids are valid (`Grp`: the grouped order).
3. `Foot`, `Post`: the shared cells are disjoint from everything read and from each other.
4. `Numeric`: the parts add up to the dot product and the `SPLIT_K` operands, summed by the
   atomics in any order, are within `tol` of `scaling · dotSum`.
5. `Spec`, `Main`: from the programs' logs and the replay of all atomic steps, `Post`.
-/
