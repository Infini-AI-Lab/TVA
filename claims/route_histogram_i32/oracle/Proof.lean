import Proof.Main

/-!
# The proof of `route_histogram_i32`

Program `p` handles the pairs `p·BLOCK .. p·BLOCK+BLOCK-1` (below `num_pairs`). The programs meet
only through the `tl.atomic_add`s on the first `NUM_BUCKETS` elements of `counts`.

1. `Proof.Defs`, `Proof.Arith`: the shared cells, the per-program post-condition `Qset` (the
   program's log adds exactly its number of pairs to each bucket), `int32` count arithmetic;
2. `Proof.Lane`, `Proof.Private`: the lanes of the private part (token, slot, group, bucket of
   each pair), proved with the `Safe` rules and lifted into `SafeA`;
3. `Proof.Count`, `Proof.AddCounts`: both counting paths (`tl.histogram` and one atomic per pair)
   are `SafeA` and log the program's count of each bucket;
4. `Proof.Post`: the replay of all logs (`replay_add`, `sum_proj`) gives `Post`;
5. `Proof.Main`: `correct_of_safeA` composes the programs into `main`.
-/
