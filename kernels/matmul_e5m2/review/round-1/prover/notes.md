# Prover's notes on matmul_e5m2, round 1

## Changes

- The tolerance of `OutSpec` (F1, F4) is now the library's fp8 `tl.dot` fact applied step by step against the exact partial sums. The new definitions are in `Defs.lean`: `prodV`, `partSum`, `partMag`, `stepMag` and `errAt`. `tol va vb K = errAt (prodV va vb K) (steps K)`, with `E (t+1) = (1 + ρ)·E t + ρ·(|S t| + B t) + α`. The first tolerance stays as the corollary `tolA K A`: `tol_le_tolA`, `tolA_rel_lt`, `tolA_le`, and `errAt_le_bound`.
- The proof (`Basic.lean`, `Program.lean`) carries `errAt` in `AccInv` in place of `DotLoop.bound`. `acc_step` closes each step with `err_step`. The `float32` room still follows from `errAt_le_bound`. `partS` and `partA` are now `partSum` and `partMag` of `prodV`.
- A new module, `MatmulE5m2.Reject`, imported by the root module, shows what the tolerance rejects at every covered size:
  - `tol_spike`: one product in the last step gives a bound of at most `ρ·|X| + 200·α`;
  - `zero_rejected`: a zero result fails at every `1 ≤ K ≤ 4096`;
  - `drop_rejected`: a skipped step `s` fails at every covered `K` and every `s < cdiv(K, 64)`;
  - helpers: `errAt_quiet`, the sparse product sequences `spikeP` and `pairP`, and the `float8e5` inputs `one8`, `mone8`, `zero8`, `spike8`, `pair8`, `ones8`.
- `K_le` stays `K ≤ 4096`, and its docstring now cites the rejection lemmas. The conditions' digest is unchanged.
- `README.md` (F2, F5): the Specification is rewritten for the new tolerance, saying what it rejects and what it does not at `K = 512` and `K = 4096`. The Proof section and the `K ≤ 4096` narrowing are rewritten too, and the module list names `MatmulE5m2.Reject`.

## Responses

- **F1** REVISED — the order-aware tolerance from the exact partial sums is now `Post`'s; a zero result and a skipped step are rejected at every covered `K` (`zero_rejected`, `drop_rejected`), so `K ≤ 4096` stays.
- **F2** REVISED — the README's account is rewritten: the step-by-step bound is tighter than the first, and what is and is not rejected is stated with lemmas and figures.
- **F3** REFERRED — library issue L119 (settled by the coordinator); the kernel no longer needs a tighter fact.
- **F4** REVISED — settled with F1, without narrowing `K_le`: from `K = 2817` to `4096` a zero `c`, a skipped loop and a dropped step now break `Post`.
- **F5** REVISED — the README lists what the new tolerance rejects at every covered size and what it still admits (the device's allowed error on the input at hand), at `K = 512` and `4096`.

## Library requests

None.

## Gate

PASS, digest `8e94d6c624254843`
