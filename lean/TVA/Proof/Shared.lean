import TVA.Proof.Shared.Log
import TVA.Proof.Shared.SafeA
import TVA.Proof.Shared.Acc
import TVA.Proof.Shared.Rules
import TVA.Proof.Shared.Launch
import TVA.Proof.Shared.Correct
import TVA.Proof.Shared.Replay
import TVA.Proof.Shared.FloatAcc

/-!
# The shared-atomics kit

For kernels whose programs update common memory cells with atomics: split-K GEMM and stream-K add
partial tiles into one output tile with `tl.atomic_add`, a histogram increments shared bins.
`correct_of_safe` does not cover them, since their footprints overlap; `correct_of_safeA` does.

**The pattern.** Many programs `tl.atomic_add` into one cell. Each addition is one indivisible
read-modify-write, so the cell ends at the result of performing all of them one after another, in
some order nobody controls. For integers every order gives the same value, the initial value plus
the sum of all increments; for floats every order gives the exact sum up to a bound. Such a launch
is well behaved: the atomics are `acq_rel` at `gpu` or `sys` scope on whole cells, so two of them
on one cell do not race; the cells are touched by nothing but these atomics; and no program uses
an atomic's result (`let _ ← tl.atomic_add …`), so the order in which they happen cannot leak into
anything else the program does.

**The route.**

1. Prove each program `SafeA` (`TVA.Proof.Shared.SafeA`): its private parts with the `Safe` rules
   (lifted by `SafeA.of_safe`), each atomic on shared cells with `SafeA.atomic_add`
   (`SafeA.atomic`, `SafeA.atomic1` for other atomics; `TVA.Proof.Shared.Rules`). The rule
   quantifies over every value the cell may hold and records the program's steps in a log; the
   post-condition says what the log holds (its increments, or `Paired` with its operands), in a
   way that ignores the log's order (`PermInv`).
2. `correct_of_safeA` (`TVA.Proof.Shared.Correct`): the launch is well behaved, every outcome is a
   normal end, and the cells hold the replay of all programs' steps in some interleaving.
3. Compute the cells from the replay (`TVA.Proof.Shared.Replay`, `TVA.Proof.Shared.FloatAcc`):
   `replay_add` with `sum_proj` for integer additions, `replay_max`/`replay_min` for integer
   maxima, `replay_err_shared` (with `Paired.merge`) for float additions.

| file | what it gives |
|---|---|
| `Shared/Log` | `AtomicStep`, `Replay`, `proj`, `cellBytes`, `PermInv` |
| `Shared/SafeA` | `SafeA` and `bind`, `mono`, `of_safe`, `perm`; `stepT`, `no_ub` |
| `Shared/Acc` | `SafeA.acc`: a `SafeA` program ends |
| `Shared/Rules` | `SafeA.atomic_add`, `atomic_add_unused`, `atomic`, `atomic1` |
| `Shared/Launch` | the launch invariant `InvA`, `no_interRaceA` |
| `Shared/Correct` | `shared`, `correct_of_safeA` |
| `Shared/Replay` | `MapsCell`, `replay_maps`, `replay_add`, `sum_proj`, `Paired`, `replay_fold`, `replay_max`, `replay_min` |
| `Shared/FloatAcc` | `AccStep`, `replay_err`, `replay_err_shared`, `accStep_atomicAdd` |
-/
