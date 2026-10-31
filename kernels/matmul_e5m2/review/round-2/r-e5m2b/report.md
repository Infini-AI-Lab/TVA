# Review of matmul_e5m2, round 2, by r-e5m2b

## Verdict

accept
The revision settles F4 and F5. `Post`'s tolerance is now the library's `DotE5m2` fact applied step by step against the exact partial sums (`errAt`), which is never above the round-1 bound and is the tightest that per-step use of the fact gives. Since `Post` holds for every input, a zero result, a skipped loop and any skipped `K`-step now break it at every covered `K`, `2817 ≤ K ≤ 4096` included (`zero_rejected`, `drop_rejected`, `tol_spike`), and so do `S/2`, `−S` and a float8e5 rounding of the result where `S` is no float8e5 value. What the bound still admits on dense inputs is the device's error as the library states it, which the README spells out with figures that I re-evaluated; a tighter fact is the library's matter (L119). So `K ≤ 4096` can stay. I have no blocking finding and accept every item.

## Findings

- **release F4** — the step-by-step tolerance `errAt` rejects a zero result and every skipped step at each covered `K` (`MatmulE5m2.Reject`, read in full and checked); what it still admits is the library's allowed device error, stated in the README.
- **release F5** — the README's account is rewritten: `−S` and a read of another row are no longer said to pass, `S/2` and `−S` are rejected on the spike input, and the figures of "What it does not reject" agree with my own evaluation.

## Accepted

All items.

## Checked

- Every changed Lean file in full: `Defs.lean` (`prodV`, `partSum`, `partMag`, `stepMag`, `errAt`, `tol`, `tolA`, `errAt_le_bound`, the new `OutSpec`), `Basic.lean` (`err_step`, `acc_step` with `errAt`, `tol_le_tolA`), `Program.lean` (`AccInv` with `errAt`, `acc0`, the end of `safe_program`), the new `Reject.lean` and the root module; no notation, instance, `open … renaming`, option or look-alike name; the new names are the kernel's own definitions.
- The new tolerance's meaning: `errAt`'s recurrence is what `DotE5m2` at `K = 64` gives per step with `|acc| ≤ |S_t| + E_t` (`err_step`); it depends on the products' positions through the loop's steps of `64`, which is the launch's `BLOCK_SIZE_K`; products from `K` on are `0`, as the masked lanes are.
- The statements of `tol_spike`, `zero_rejected` and `drop_rejected` with full names (`work/Round2.lean`): each exhibits finite `float8e5` rows and columns with room in `float16` on which the wrong result is outside `OutSpec`'s own `float16` clause; and a check that `S/2` is rejected on the spike input at `K = 4096`.
- `#print axioms` of `main`, `consistent`, `satisfiable`, `zero_rejected` and `drop_rejected`: only `propext`, `Classical.choice`, `Quot.sound`.
- The README's figures, by evaluating `errAt` with `python3 -I`: one product in step `0` (`13.3%` at `512`, `170.8%` at `4096`), all-ones products (`7.3%`, `72.8%`), the pair input at `s = 0`, `K = 4096` (`0.124`), the spike input (`ρ|X|`, `1.57%`; bound below `0.026` at `1.5625`), and `torch.randn` data cast to `float8e5` (medians of about `6.7` at `512` and `117` at `4096`, against the README's `7.0` and `114`).
- The rewritten README (Specification, the `K_le` row of Conditions, Proof, the `K ≤ 4096` narrowing) against the Lean; no reference to the old tolerance is left but `tolA` as the corollary.
- That the model, map, conditions and assumption items have the digests I checked in round 1.

## Not checked

- Whether a tighter fp8 `tl.dot` fact holds (library issue L119, the library's matter), and the library's facts in general (trusted).
- Part (B) of `Compiler.Faithful`'s scope (the environment), by the protocol.
- I did not re-run the gate or `lake build`; Lean checked the proofs, and I ran only my work files.
