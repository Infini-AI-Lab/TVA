# Review of matmul_e5m2, round 2, by r-e5m2

## Verdict

accept
The revision settles all three of my round-1 findings. `OutSpec`'s tolerance is now the library's fp8 `tl.dot` fact applied step by step against the exact partial sums (`errAt`). It is never above the first tolerance, and at every covered `K` it rejects a zero result, any skipped step, `S/2`, `−S` and a float8e5 rounding, each on some finite input with room in `float16` (`MatmulE5m2.Reject`). What it still admits on some inputs, up to `170.8%` of `|X|` for one product in step 0 at `K = 4096` and about `2.3·|S|` on the script's randn data, is the device's error as the library's fact states it (L119). The kernel cannot tighten that, and the write-up says so. So `K ≤ 4096` with this tolerance is right for the kernel's real use. The model, map, conditions and assumptions are unchanged, and I accept every item.

## Findings

- **release F1** — the order-aware tolerance is `Post`'s, carried in `AccInv`, never above `tolA`; zero, a skipped step, `S/2`, `−S` and a float8e5 rounding are rejected at every covered `K` (`zero_rejected`, `drop_rejected`, `tol_spike`).
- **release F2** — the README's account of the tolerance is rewritten and its figures check out.
- **release F3** — settled as library issue L119; the kernel no longer needs a tighter fact.

## Accepted

All items.

## Checked

- Every Lean file of the kernel again in full (`Defs`, `Basic`, `Program`, the new `Reject`, the root module; `Arith` unchanged): no instance, notation, renaming, local binder or kernel name that changes a library name. `tol`, `errAt`, `partSum`, `stepMag`, `prodV`, `OutSpec`, `Post` and the statements of `tol_le_tolA`, `zero_rejected`, `drop_rejected` and `tol_spike` printed with `pp.fullNames` (`work/names2.lean`). `main`, `consistent`, `satisfiable`, `zero_rejected` and `drop_rejected` use only `propext`, `Classical.choice` and `Quot.sound`.
- That `errAt` is the recurrence the per-step `DotE5m2` fact gives (`err_step` in `acc_step`). Also that the rejection lemmas say what the write-up claims for a kernel that stores `0` or skips a step: each gives a finite input with room in `float16` on which that result is outside `OutSpec`'s `float16` bound.
- The write-up's figures, recomputed exactly: `13.3%` and `170.8%` (one product in step 0), `7.3%` and `72.8%` (products of one sign), `0.124` (pair input, `K = 4096`), `0.02528 < 0.0625` (the float8e5 rounding of `1.5625`), `α < 2.4·10^-38`. The randn medians agree with my own simulation, with the fp8 cast.
- The model and launch are unchanged (same digest). The conditions digest is the same; the `K_le` docstring changed, and it is accurate.
- The `K ≤ 4096` narrowing and the tolerance against the host's use: the test (`K = 512`) and every benchmark size are covered, with a tolerance that rejects plausible wrong kernels at each.

## Not checked

- Part (B) of `Compiler.Faithful`'s scope (the environment), as the protocol says.
- Whether the library's facts and constants are true; L119 is the library's.
- The proofs step by step, beyond what Lean checks and what the names mean.
