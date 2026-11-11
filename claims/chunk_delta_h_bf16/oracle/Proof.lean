import Proof.Main

/-!
# The proof of `chunk_delta_h_bf16`

The launch's programs each own the columns of one block of `32` value columns of one value head of
one sequence (`Proof.Defs`): they read nothing the launch writes and write disjoint cells
(`Proof.Foot`). `correct_of_safe` (`Proof.Main`) composes the programs, once each is `Safe` with the
post-condition `Q`.

1. `Proof.Seg`, `Proof.Seg2`: rules for the segments of a trip and of the prologue and epilogue (the
   stores of the state, the `w` dots, the difference and the `v_new` store, the gate, the `k` dots,
   the `h0` loads and the `ht` stores), each for an arbitrary rest of the program;
2. `Proof.Trip`, `Proof.TripVf`: a trip of the main loop (`chunk_step`, `chunk_step_vf`) walked with
   those rules, ending in the loop invariant (`Proof.Inv`) after one more chunk;
3. the numerics of a trip, one value column (`Proof.NumV`, `NumG`, `NumS` and the `Glue*`, `Lane`
   modules): the bounds on the new values, on the gated values and on one step of the state;
4. `Proof.Loop`, `Proof.Init`, `Proof.Final`, `Proof.Body`, `Proof.BodyRest`: the loop's trips,
   the state entering it, the stores of the final state and the program after its sequence is chosen;
5. `Proof.Program`: one program from `program_id`; `Proof.Main`: the launch.
-/
