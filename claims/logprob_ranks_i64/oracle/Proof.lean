import Proof.Main

/-!
# The proof of `logprob_ranks_i64`

Program `r` of the grid loads row `r`'s token id and the token's logit `x`, walks the row in
blocks of `BLOCK_SIZE` lanes, counts the lanes at least `x` (the padding lanes hold `-inf`), and
stores the count at element `r` of `token_ranks`. It reads its token id and its row and writes
that one element, so the footprints of distinct programs are disjoint, and `correct_of_safe`
composes the programs. The theorem `main` is in `Proof/Main.lean`; the plan:

1. `Proof/Defs.lean`: `hit d r q` (whether position `q` of the padded walk counts), the count
   `cnt d r t` after `t` trips and `cnt_final` (after the last trip it is `rank d r`); the
   footprints `Rp`, `Wp` and the post-condition `Qp`; the lane of `geX`, the value of the converted
   `-inf` (`other_eq`), and the wrapping `int32` sum of a block of `0`/`1` lanes (`foldAll_toInt`);
2. `Proof/Trip.lean`: one trip is safe and adds its lanes to the count (`safe_trip`);
3. `Proof/Main.lean`: the body walked line by line (`safe_program`), with the loop by
   `safe_forRange`, the disjoint footprints, and `main`.
-/
