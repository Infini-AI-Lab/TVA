import TVA.Launch.Step
import TVA.Launch.Run
import TVA.Launch.WellBehaved
import TVA.Launch.Progress

/-!
# The semantics of a launch

A Triton launch `kernel[grid](args…)` runs one program instance per point of the grid, all
sharing global memory. This module says what such a launch does, when that meaning applies
(*well behaved*), and what a claim states about it (`Correct`).

## How a launch runs

* **The model.** A `Kernel` gives, for each program's `Ctx` (`tl.program_id`,
  `tl.num_programs`), the program it runs: a `Prog Unit` (`TVA.Program`), whose effects are
  loads, stores, atomic read-modify-writes of one element (`rmw`), unpredictable values (`pick`),
  operations the platform defines (`plat`), undefined behaviour, loops and reads of the
  environment.
* **The launch.** A `Launch` gives the grid, the options that change the meaning
  (`enable_fp_fusion`, and `debug`, `num_ctas`, `launch_pdl`, which `Launch.Valid` requires at
  their defaults) and the source's block shapes. Program number `i` gets
  `tl.program_id = (i % x, i / x % y, i / (x·y))`. The launch starts with every program at its
  beginning and global memory `init` (`Config.start`). `init` is a partial map: the bytes present
  in it are the ones the launch is given, and accessing any other byte is undefined behaviour, so
  the final memory has the same bytes present.
* **One step of one program** (`Step`, in `TVA.Launch.Step`) performs its next effect as one
  indivisible action: a `tl.load` reads all its enabled lanes at once, a `tl.store` writes all its
  enabled lanes at once (duplicate addresses in some order), an atomic is one step per element.
  Each step records an `Event`: the bytes it read and wrote, and for an atomic its `sem`, `scope`
  and element. An invalid access (a byte the launch was not given, a misaligned address) is
  undefined behaviour.
* **Interleaving.** A launch step (`LStep`) is a step of any one program on the shared memory;
  a trace (`Reach`) is any sequence of them. Every interleaving is allowed, and each step sees the
  memory all earlier steps left: sequential consistency between programs. Nothing orders the
  steps of different programs: happens-before (`HB`) is each program's own order.
* **Atomics may be performed later.** A pending atomic may be passed by a later step of its
  program that does not depend on its result (`Step.defer`), so the semantics performs a
  program's atomics, and the old values they return, in every order the program's own
  dependences allow. A step taken before an atomic cannot know its result.
* **Endings** (`Outcomes`): every program finishes (`ok fin`, with final memory `fin`), some
  interleaving reaches undefined behaviour (`ub`), or some interleaving runs forever (`hang`).

## What makes a launch well behaved

The interleaving is the launch's meaning only when it is `WellBehaved`
(`TVA.Launch.WellBehaved`):

1. **No undefined behaviour**: no reachable state has a program whose next step is undefined
   behaviour (an invalid access, a `pick` or `plat` nothing satisfies, an integer division by
   zero, …).
2. **No race between programs** (`InterRace`): no trace has two steps of different programs that
   touch a common byte, at least one writing it, unless both are `acq_rel` atomics at `gpu` or
   `sys` scope on exactly the same bytes (`Event.Strong`).
3. **No conflict within a program** (`IntraRace`): no trace has two steps of one program that
   touch a common byte, at least one writing it, unless both are atomics on exactly the same bytes
   (`Event.SameAtomic`). Loop iterations count.
4. **No atomic's result used on a shared cell** (`UsesSharedAtomicResult`): no program uses the
   old value an atomic returns when the atomic's element is also reached by another program, or
   by another access of the same program.

## Why each exclusion is there

1. **Undefined behaviour.** The device may then do anything (fault, return garbage, corrupt
   memory), so nothing about the launch can be said.
2. **Races between programs.** The device does not perform a tensor access as one indivisible
   step: its threads' accesses reach memory piece by piece and in orders an interleaving of whole
   steps does not show (a load may see half of another program's store). When no two programs'
   accesses conflict, those orders cannot be observed, and the interleaving is exact. Two `acq_rel`
   atomics on the same element are each indivisible and ordered as in an interleaving; a
   `relaxed`, `acquire` or `release` atomic may be reordered with its program's other accesses,
   which an interleaving does not show, so it races (an atomic whose result is never used is
   modelled as `acq_rel`: `MemSem.unused` says why that is sound). Programs do not synchronise:
   one thread issues a scalar atomic with no barrier before it, so a flag set with it does not
   publish the program's other writes; handing data between programs behind a flag is a race.
3. **Conflicts within a program.** A program is many threads, with no barrier between its
   operations, and the threads do not order the program's own accesses: a load may miss a store
   the program made before it, and two stores to the same bytes may land in either order. The
   semantics follows the program text, so it would give only one of those results; a program
   must therefore not access the same bytes twice in a conflicting way. Atomics on the same
   element are exempt: each is indivisible, and the semantics already performs them in every
   order (`Step.defer`). For the same reason `tl.debug_barrier()` has no meaning here.
4. **An atomic's result used on a shared cell.** A tensor step that uses one lane's result is one
   step here: it waits for that lane's atomic and holds back every later step of the program,
   later atomics included. On the device each thread orders only its own accesses, so another
   thread may perform a later atomic first. When the atomic's element is reached by anything else,
   the old value can show that order, which the semantics lacks.

## What a claim states

`Correct E k L init Q`: the launch is valid, well behaved, and every outcome is a normal end
whose final memory satisfies `Q`. A fault (undefined behaviour), a hang, or a launch outside the
semantics makes it false. A well-behaved launch always has an outcome
(`WellBehaved.exists_outcome`), so `Correct` is never satisfied for want of one. `TVA.Claim`
states `Correct` for every covered instance; the proof kit (`TVA.Proof`) proves it.

Files: `Launch/Step` (one program's step), `Launch/Run` (kernels, launches, interleaving),
`Launch/WellBehaved` (races, the atomic-result exclusion, `WellBehaved`, `Outcomes`, `Correct`),
`Launch/Progress` (a launch never gets stuck).
-/
