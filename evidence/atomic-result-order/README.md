# The order of a program's atomics when a result is used (library issue L161)

Library evidence (not a kernel's), for library issue L161. Triton 3.8.0 (TVA's `.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-11-08. Script: [run.py](run.py); output:
[out.txt](out.txt) (about a minute); the variants of launch B (library issue L165):
[variants.py](variants.py), [variants.out.txt](variants.out.txt).

**Question.** `Step.defer` (`lean/TVA/Sem/Step.lean`) lets a pending atomic be passed only by a
later step that is the same for every old value it may return. A tensor `tl.store` is one step, so a
store that uses any lane of an atomic's result holds back every later atomic of the program. On the
device each thread orders only its own accesses. Library review lr-34a (round 34,
`docs/library-review/round-34/lr-34a/work/crosswarp_nowait.py`) found a well-behaved two-program
launch whose device outcome the semantics lacks. Does the device give such outcomes, and does it
also within one program, between two atomic calls on one cell?

**Method** (`run.py`). Both launches are well behaved in the semantics: every cell two accesses
share is reached by `acq_rel` gpu atomics only (`Event.Strong`), no program has an `IntraRace`
(two atomics on the same bytes are exempt), and no program waits on another. Element 0 of program
0 is delayed by a chain of 400 dependent loads of a buffer no program writes (held by warp 0,
`num_warps=4`; on these data the chain leaves `x = offs`).
- **A** (lr-34a's launch). Program 0: `r = tl.atomic_add(Y + x, 0)`, `tl.store(OUT + x, r)`,
  `tl.atomic_add(C + x, 1)`. Program 1, after a chain of 60 loads that ends at `j = 0`:
  `q = tl.atomic_add(C + 511 + j, 0)`, `tl.atomic_add(Y, q)`, `tl.store(Q, q)`. In the semantics
  program 0's atomic on `Y[0]` precedes its store, which precedes its atomic on `C[511]`, and
  program 1 updates `Y[0]` after reading `C[511]`: no outcome has `Q = 1` and `OUT[0] = 1`.
- **B** (one program). `r = tl.atomic_add(Y0 + 0 * x, 1)` (512 lanes on the one cell `Y0[0]`),
  `tl.store(OUT + x, r)`, `s = tl.atomic_add(Y0 + 0 * x, 1)`, `tl.store(OUT2 + x, s)`. In the
  semantics every lane of the first call precedes the store, which precedes every lane of the
  second: from `0`, every `r` is below 512 and every `s` at least 512.

Each with `sem="acq_rel"` and `sem="relaxed"` on every atomic, 200 launches each; every run is
checked to have kept every update. With `relaxed`, launch A is not well behaved (library issue
L165, from library review lr-35a): program 1's atomic on `C[511]` and program 0's atomic on `Y[0]`
have their results used, so they are not `MemSem.unused` and keep `relaxed`, and each conflicts with
the other program's atomic on the same cell without being `Event.Strong` (`InterRace`). Launch B
with `relaxed` stays well behaved: its two calls are of one program, and two atomics on the same
bytes are no `IntraRace` (`Event.SameAtomic`).
- **Variants of B** (`variants.py`, from library review lr-35a's `work/atomic_variants.py`, round
  35; GPU 0, 2026-11-08; `acq_rel`, 100 launches each): `barrier`, B with `tl.debug_barrier()`
  between the store of `r` and the second call; `loop`, the same call run twice in a loop, each
  result stored; `addr`, the first result used only in an address (`tl.store(OUT + r, x)`); `unused2`,
  the first result stored and the second call's result unused (the semantics still has every
  `r < 512`, since the second call follows the store).

**Result** (`out.txt`).

| launch | `sem` | runs outside the semantics |
|---|---|---|
| A | `acq_rel` | 200 of 200 (`Q = 1` and `OUT[0] = 1`) |
| A | `relaxed` | 200 of 200 |
| B | `acq_rel` | 200 of 200 (the last: largest `r` 895, smallest `s` 384, `r[0]` 768) |
| B | `relaxed` | 200 of 200 |

The variants of B (`variants.out.txt`): `barrier` 0 of 100 outside the semantics; `loop`, `addr` and
`unused2` 100 of 100 each (the last: largest first-call value 895, smallest second-call value 384).

**Conclusions.** `Compiler.Faithful` is false of launch A with `acq_rel` and of launch B with
either order (launch A with `relaxed` races, so the rule says nothing of it, and it shows nothing
the rule gets wrong): the device performs a later
atomic of another thread before an earlier step of the program that uses an atomic's result, and
the semantics does not. By the user's decision (2026-11-08, "let's exclude with Fix B") the model is
kept, and every launch in which a program uses an atomic's result on a cell another program, or
another atomic operation of the same program, also accesses is excluded (`docs/exclusions.md`,
`docs/SEMANTICS.md` "Atomics", `docs/rules/compiler-faithful.md` scope 12). Launch A is under
`TVA.UsesSharedAtomicResult` (`TVA.AtomicResult.crosswarp_excluded`, `lean/TVA/Lemmas/AtomicResult.lean`).
Launch B shows why the exclusion also covers a second atomic operation of the same program on the
cell: the lanes of one call are performed in any order in the semantics, but two calls are not. The
variants show the same with the same call run twice in a loop, with a result used only in an
address, and with the second call's result unused, so the exclusion covers them; and that a
`tl.debug_barrier()` between the use and the second call gave only outcomes the semantics has, which
a per-lane model may use (`docs/review-questions.md` Q17; the semantics gives the barrier no meaning,
`docs/SEMANTICS.md` "No barrier semantics", so such a launch stays excluded).
