import TVA.Launch.Step

/-!
# Kernels, launches, and how the programs of a launch run

A Triton launch `kernel[grid](args…)` runs one program instance per point of the grid. Here:

* `Kernel`: the model, the program each grid point runs, given its `Ctx` (`tl.program_id`,
  `tl.num_programs`). A claim applies its model to the instance's arguments to get a `Kernel`.
* `Launch`: the grid, the compile and launch options that change the meaning, and the source's
  block shapes; `Launch.Valid` says when the semantics describes the launch at all.
* `Config`: a state of the launch (one memory, each program's rest); `Config.start`.
* `LStep`: one step of any one program (`Step`) acting on the shared memory, labelled with the
  program and its `Event`. `Reach`: a sequence of such steps (a *trace*). Every interleaving of the
  programs' steps is allowed and each step is indivisible: the programs see memory sequentially
  consistently.

What makes a launch well behaved, and how it ends, is in `TVA.Launch.WellBehaved`.
-/
namespace TVA

/-- A kernel model: the program `body ctx` that the program instance with context `ctx` runs. -/
structure Kernel where
  /-- The program of each grid point. -/
  body : Ctx → Prog Unit

/-- A block shape Triton accepts: every dimension a power of two, at most `2^20` elements
(`validate_block_shape`, `triton-3.8.0-utils.py`, applied to every block type at
`triton-3.8.0-core.py:721`; `tl.arange` checks the same). -/
def Shape.TritonOk (s : Shape) : Prop := (∀ k ∈ s, ∃ e, k = 2 ^ e) ∧ s.prod ≤ 2 ^ 20

/-- A launch of a kernel: the grid, the compile and launch options that change what the model
means, and the source's block shapes. Every other option of Triton's NVIDIA backend is assumed at
its default, except `num_warps`, `num_stages`, `maxnreg` and `launch_cooperative_grid`, which do
not change the meaning (`TVA.Claim` states this). -/
structure Launch where
  /-- The grid `(x, y, z)`, as the host's grid formula gives it. -/
  grid : Fin 3 → Nat
  /-- `enable_fp_fusion` (Triton's default `True`): when set, the compiler may fuse a
  multiplication into a following addition. Programs read it through `Env.fusion`. -/
  fusion : Bool := true
  /-- The `debug` option (default `False`). With it, `tl.device_assert` and integer overflow checks
  trap, which the semantics does not model; `Launch.Valid` requires `false`. -/
  debug : Bool := false
  /-- `num_ctas` (default `1`). With more, a program is a cluster of CTAs and a `cta`-scope atomic
  is atomic only within one of them; `Launch.Valid` requires `1`. -/
  numCtas : Nat := 1
  /-- `launch_pdl` (programmatic dependent launch, default `False`; the launcher turns it into a
  launch attribute, `triton-3.8.0-nvidia-driver.c:1004-1008`). With it, the kernel may start
  before the previous kernel on the stream has finished, so its programs may read memory before
  that kernel's writes; `Launch.Valid` requires `false`. -/
  pdl : Bool := false
  /-- The shapes of the source's block values on this launch (every `tl.arange`, loaded tile,
  accumulator, …), as the claim lists them. `Launch.Valid` requires Triton to accept each; it
  checks only the shapes listed. -/
  blocks : List Shape

namespace Launch

/-- The number of programs: the product of the grid's three dimensions. -/
def size (L : Launch) : Nat := L.grid 0 * L.grid 1 * L.grid 2

/-- The context of program number `i`: `tl.program_id` is `(i % x, i / x % y, i / (x·y))` (axis
`0` varies fastest), and `tl.num_programs` is the grid. -/
def ctx (L : Launch) (i : Nat) : Ctx :=
  ⟨![i % L.grid 0, i / L.grid 0 % L.grid 1, i / (L.grid 0 * L.grid 1)], L.grid⟩

/-- A launch on the one-dimensional grid `(n,)`, with the source's block shapes, and the other
options at Triton's defaults (except `fusion` if given). -/
def grid1 (n : Nat) (blocks : List Shape) (fusion : Bool := true) : Launch :=
  { grid := ![n, 1, 1], fusion, blocks }

/-- **The semantics describes the launch.** A launch that is not `Valid` either does not run its
grid as the semantics runs it, or runs code the semantics does not model:

* the grid is within CUDA's limits, `x < 2^31` and `y, z < 2^16` (beyond them the driver reports
  an error and no program runs);
* fewer than `2^31` programs: Triton's launcher computes `x · y · z` in a C `int` and skips the
  launch, with no error, when that product is not positive
  (`triton-3.8.0-nvidia-driver.c:972`). With `2^31` to `2^32` programs no program runs and memory
  is unchanged, while the semantics runs every program. (A grid with a zero dimension runs no
  program both on the device and here: it ends with memory unchanged.);
* `num_ctas = 1`, no programmatic dependent launch and no `debug` (see the fields);
* the listed block shapes are ones Triton accepts.

The other limits (shared memory, registers, threads, `tl.dot` sizes, dtypes the device lacks) are
not part of `Valid`: Triton or the driver rejects a launch that exceeds one with an error before
any program runs, so such a launch has no outcome on the device (`TVA.Claim`). -/
def Valid (L : Launch) : Prop :=
  L.grid 0 < 2 ^ 31 ∧ L.grid 1 < 2 ^ 16 ∧ L.grid 2 < 2 ^ 16 ∧ L.size < 2 ^ 31 ∧ L.numCtas = 1 ∧
    L.pdl = false ∧ L.debug = false ∧ ∀ s ∈ L.blocks, s.TritonOk

/-- The environment the launch's programs run in, on platform `P`. -/
def env (L : Launch) (P : Platform) : Env := ⟨P, L.fusion⟩

end Launch

/-- A state of a launch of `N` programs: the global memory, and what each program has left to
run (`ret ()` once it has finished). -/
structure Config (N : Nat) where
  /-- The global memory. -/
  mem : Mem
  /-- The rest of each program. -/
  progs : Fin N → Prog Unit

namespace Config

/-- The start of the launch `L` of kernel `k` from memory `init`: program `i` is the kernel's
body for context `L.ctx i`. -/
def start (k : Kernel) (L : Launch) (init : Mem) : Config L.size :=
  ⟨init, fun i => k.body (L.ctx i)⟩

/-- Every program has finished. -/
def Done {N : Nat} (c : Config N) : Prop := ∀ i, c.progs i = .ret ()

/-- Some program's next step can be undefined behaviour. -/
def StuckUB {N : Nat} (E : Env) (c : Config N) : Prop := ∃ i, Step E c.mem (c.progs i) .ub

end Config

/-- One step of the launch: program `i` takes a step from the shared memory, which becomes the
step's memory; the step is labelled `(i, e)` with its event. -/
inductive LStep {N : Nat} (E : Env) : Config N → Fin N × Event → Config N → Prop where
  /-- Program `i` steps; the other programs are unchanged. -/
  | step {c : Config N} (i : Fin N) {m' : Mem} {p' : Prog Unit} {e : Event} :
      Step E c.mem (c.progs i) (.next m' p' e) →
      LStep E c (i, e) ⟨m', Function.update c.progs i p'⟩

/-- `Reach E c tr c'`: from `c` the launch can take the steps labelled `tr`, in that order, and
reach `c'`. The list `tr` is the *trace*: races are read on it (`TVA.Launch.WellBehaved`). -/
inductive Reach {N : Nat} (E : Env) : Config N → List (Fin N × Event) → Config N → Prop where
  /-- No step. -/
  | refl (c : Config N) : Reach E c [] c
  /-- One more step at the end. -/
  | tail {c c' c'' : Config N} {tr : List (Fin N × Event)} {ev : Fin N × Event} :
      Reach E c tr c' → LStep E c' ev c'' → Reach E c (tr ++ [ev]) c''

end TVA
