import Proof.Main

/-!
# The proof of `count_expert_tokens_i32`

Program `e` of the grid counts, for the expert `e`, the positions of `topk_ids` routed to it, and
stores the count in element `e` of `expert_num_tokens`. It reads the ids (and the map) and writes
that one element, so the footprints of distinct programs are disjoint, and `correct_of_safe`
composes the programs. The theorem `main` is in `Proof/Main.lean`; the plan:

1. `Proof/Defs.lean`: the trip count `numTrips = cdiv(numel, B)`, the per-lane count `cnt d e t l`
   (positions `x * B + l`, `x < t`, routed to `e`), the footprints `Rin`, `Wout d e` and the
   post-condition `Qout`; the bounds that keep every `int32` step from wrapping;
2. `Proof/Lane.lean`: the values of the lanes of masks, comparisons, `tl.where` and pointer sums;
3. `Proof/Sum.lean`: the wrapping `int32` sum of the lane counts is `countOf d e` (`foldAll_toInt`);
4. `Proof/Trip.lean`: one trip of the loop, written out (`loadBody`), and the values of its mask,
   loaded ids and gathered values;
5. `Proof/Loop.lean`: one trip is safe in each branch of `HAS_EXPERT_MAP`
   (`safe_trip_false`, `safe_trip_true`), advancing the invariant "lane `l` has counted the first
   `t` trips, its pointer is at position `t * B + l`";
6. `Proof/Main.lean`: the loop by `safe_forRange`, the final store, `safe_program`, the disjoint
   footprints, and `main`.
-/
