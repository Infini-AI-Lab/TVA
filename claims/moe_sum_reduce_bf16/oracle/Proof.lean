import Proof.Main

/-!
# `moe_sum_reduce_bf16`: the proof

The launch runs one program per block of the token/column tiling.

1. `Basic`: the grid is valid, and the program that owns output element `(t, j)`.
2. `Math`: the error of the `float32` accumulation of `K` terms, the scale's conversion, and the
   arithmetic showing the total error is within `tol`.
3. `Value`, `Addr`, `Program`: the value an output lane ends with, the `int32`/`int64` addresses
   and masks of the lanes, and the program body with its loop body factored out.
4. `Walk`: program `p` reads the input elements of its tile and writes the output elements of its
   tile, leaving there a `bfloat16` within `tol` of `sigma * refSum` (`safe_prog`).
5. `Main`: tiles of distinct programs are disjoint and the output shares no byte with the input,
   so `correct_of_safe` gives `claim.Holds`.
-/
