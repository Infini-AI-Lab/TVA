import TVA.Sem.Step
import Mathlib.Logic.Relation
import Mathlib.Data.Fin.VecNotation

/-!
# A launch

A kernel launch runs one program per point of the grid. The programs share global memory and
run concurrently: the semantics takes every interleaving of their steps, each step acting
atomically on memory (sequential consistency).

That is the language's meaning only for *well-behaved* launches. The rule
`TVA.Rules.Compiler.Faithful` gives the device the outcomes of this semantics only when no
interleaving reaches undefined behaviour, a data race between programs (`InterRace`) or a
conflict within one program (`IntraRace`).

A data race is a pair of conflicting accesses from different programs, unless both are
`Event.Strong` atomics: `acq_rel` (Triton's default), at `gpu`/`sys` scope, on the same bytes.
Relaxed, acquire-only and release-only atomics of different programs race when the program uses
the value they return. When it does not (library issue L157, user decision 2026-11-08), their
model is the `acq_rel` atomic (`MemSem.unused`, `tl.atomic_add_unused` and the others,
`TVA.Lang.Ops`), so on the same bytes, at `gpu`/`sys` scope, they are `Event.Strong` and do not
race: `MemSem.unused` says why. A plain load or store of those bytes in the launch still races with
them (or conflicts within its program), so the cells such atomics share are reached by atomics
only. The exemption is not stated on events because an event does not show whether its program
uses the old value (a deferred atomic's continuation is re-chosen at each later step, `Step.defer`).

A program is not sequential for global memory either (library issue L15, user decision
2026-10-27). Its threads access memory with no barrier between its operations, so a load may
miss a store the program made before it, and two stores to the same bytes may land in either
order. Two accesses of one program conflict when they touch the same bytes and at least one
writes them (a store or an atomic), unless both are atomics on the same bytes (`IntraRace`); a
model with such a conflict anywhere, across loop iterations included, is not well behaved, like a
race between programs, and no claim covers it. Two atomics of one program on the same bytes do
not conflict, but their order is unspecified: `Step.defer` lets the semantics perform an atomic
after later steps that do not depend on its result. A tensor step that uses one lane's result is
one step here, while on the device each thread orders only its own accesses, so a launch in which a
program uses an atomic's result on a cell another program, or another atomic operation of the same
program, also accesses is excluded (library issue L161, `TVA.Sem.AtomicResult`). `tl.debug_barrier()` keeps no meaning
here, and no rule orders a program's accesses.

There is no synchronises-with edge between programs. A program is many threads. A scalar
atomic is issued by one of them, with no barrier before it
(`triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:756-787, 914-916`), so its release does not cover the
other threads' earlier writes. Handing data from one program to another through plain memory
behind a flag or counter is therefore a race here; such kernels are out of scope
(`docs/exclusions.md`). Happens-before is program order only.
-/
namespace TVA

/-- A kernel model: the program each `pid` runs. -/
structure Kernel where
  body : Ctx → Prog Unit

/-- A block shape Triton accepts: every dimension a power of two, at most `2^20` elements
(`validate_block_shape`, `triton-3.8.0-utils.py:63-74`, applied to every block type at
`triton-3.8.0-core.py:721`; `tl.arange` checks the same, `semantic.py:594`). -/
def Shape.TritonOk (s : Shape) : Prop := (∀ k ∈ s, ∃ e, k = 2 ^ e) ∧ s.prod ≤ 2 ^ 20

/-- A launch configuration: the grid, and the launch and compile options that change what the
model means (`enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl`), with the source's block
shapes. `Rules.Compiler.Faithful` covers a launch only when every other compile or launch option is
not given, except `num_warps`, `num_stages`, `maxnreg` and `launch_cooperative_grid`, which do not
change the meaning (library issue L19). -/
structure Launch where
  /-- The grid `(x, y, z)`. -/
  grid : Fin 3 → Nat
  /-- `enable_fp_fusion`: the value given to the call that compiles the kernel, that is the
  launch's keyword, `kernel.warmup(…)`'s or `triton.compile`'s option (for a `CompiledKernel`), or,
  under `@triton.autotune`, the chosen `triton.Config`'s, whose keywords reach the JIT with the
  launch's (`Config.all_kwargs`, `triton-3.8.0-autotuner.py:276-281, 365-375`; library issue L92);
  Triton's default `True` when none gives one. That default is `triton.knobs.language.default_fp_fusion` at its own default:
  `Rules.Compiler.Faithful` covers only launches with Triton's knobs at their defaults, whether
  they would be set by the environment variable `TRITON_DEFAULT_FP_FUSION` or from Python. Nor does
  it cover a kernel under `@triton.heuristics`, whose heuristics may set this option, or any other,
  over the launch's own (its scope 4; library issue L78). -/
  fusion : Bool := true
  /-- The `debug` compile option (library issue L105). On when the call that compiles the kernel
  gives `debug=True`: the launch's keyword, `kernel.warmup(…)`'s, `triton.compile`'s option, or,
  under `@triton.autotune`, the chosen `triton.Config`'s; when that call gives no `debug`, when the
  kernel's `JITFunction` attribute `debug` is true, which the decorator sets
  (`@triton.jit(debug=True)`) and Python code may assign (`kernel.debug = True`; `JITFunction.run`
  takes `kwargs.get("debug", self.debug)`, `triton-3.8.0-jit.py:727`); and in every case when the
  environment variable `TRITON_DEBUG=1` or the knob `triton.knobs.runtime.debug = True` (a plain
  attribute, set from Python with no environment variable) asks for it. `False` otherwise. With it,
  `tl.device_assert` and integer overflow checks trap (with the attribute set, an `int32` addition
  that overflows trapped, `evidence/host-round20` D1).
  `Compiler.Faithful` covers only `debug = false`. -/
  debug : Bool := false
  /-- `num_ctas` (default `1`). With more, a program is a cluster of CTAs and `cta`-scope atomics
  are atomic only within one of them; `Launch.Valid` requires `1`. -/
  numCtas : Nat := 1
  /-- The launch option `launch_pdl` (programmatic dependent launch): the NVIDIA backend's option
  (`triton-3.8.0-nvidia-compiler.py:124`, default `False`), which the launcher turns into a
  launch attribute (`triton-3.8.0-nvidia-driver.c:1004-1008`). With it, the kernel may start before the
  previous kernel on the stream has finished, so its programs may read memory before that
  kernel's writes, and the claim's initial memory would not be the memory they read.
  `Launch.Valid` requires `false`. The other side, a *later* launch with `launch_pdl=True` that
  starts while this kernel runs (after its programs call `gdc_launch_dependents()`), is outside
  `Rules.Compiler.Faithful` too when it writes a byte `init` gives this launch: nothing else may
  write those bytes until the launch ends (library issues L17, L49). -/
  pdl : Bool := false
  /-- The shapes of the source's block values on this instance (every `tl.arange`, loaded tile,
  accumulator, …). The prover lists them and the reviewer checks the list against the source;
  `Launch.Valid` requires Triton to accept each. -/
  blocks : List Shape

namespace Launch

/-- The number of programs. -/
def size (L : Launch) : Nat := L.grid 0 * L.grid 1 * L.grid 2

/-- The context of program number `i` (`x` fastest). -/
def ctx (L : Launch) (i : Nat) : Ctx :=
  ⟨![i % L.grid 0, i / L.grid 0 % L.grid 1, i / (L.grid 0 * L.grid 1)], L.grid⟩

/-- A one-dimensional grid, with the source's block shapes. -/
def grid1 (n : Nat) (blocks : List Shape) (fusion : Bool := true) : Launch :=
  { grid := ![n, 1, 1], fusion, blocks }

/-- The launch is one whose execution runs its whole grid whenever it returns without an error, as
far as the library can state it (an execution of the kernel, as `Rules.Compiler.Faithful` reads a
launch; library issue L144):
* the CUDA grid limits, `x < 2^31` and `y, z < 2^16`;
* fewer than `2^31` programs. Triton 3.8.0's launcher computes `gridX * gridY * gridZ` in C `int`
  and skips the launch, with no error, when that product is not positive
  (`triton-3.8.0-nvidia-driver.c:968-972`). With `2^31` to `2^32` programs it returns at once and
  no program runs (`evidence/launch-outcomes`; library issue L9): that call ends `ok` with memory
  unchanged, an outcome the model of the grid does not give, so this bound keeps it out;
* `num_ctas = 1`;
* no programmatic dependent launch (`pdl = false`, `launch_pdl=False`): the kernel starts after
  the previous kernel on the stream has finished, so its programs read the memory that kernel
  left (library issue L17);
* block shapes Triton accepts, for the shapes listed in `blocks`.

The other limits (shared memory, registers, threads, `tl.dot` sizes, dtypes) are not part of
`Valid`. Triton or the driver reports a launch that exceeds one to the host with an error, before
any program runs, and such a launch has no outcome (`TVA.Claim.Behaviour`). -/
def Valid (L : Launch) : Prop :=
  L.grid 0 < 2 ^ 31 ∧ L.grid 1 < 2 ^ 16 ∧ L.grid 2 < 2 ^ 16 ∧ L.size < 2 ^ 31 ∧ L.numCtas = 1 ∧
    L.pdl = false ∧ ∀ s ∈ L.blocks, s.TritonOk

/-- The environment programs run in. -/
def env (L : Launch) (P : Platform) : Env := ⟨P, L.fusion⟩

end Launch

/-- The state of a launch with `N` programs. -/
structure Config (N : Nat) where
  mem : Mem
  progs : Fin N → Prog Unit

/-- The start of a launch. -/
def Config.start (k : Kernel) (L : Launch) (init : Mem) : Config L.size :=
  ⟨init, fun i => k.body (L.ctx i)⟩

/-- All programs have finished. -/
def Config.Done {N : Nat} (c : Config N) : Prop := ∀ i, c.progs i = .ret ()

/-- Some program's next step is undefined behaviour. -/
def Config.StuckUB {N : Nat} (E : Env) (c : Config N) : Prop := ∃ i, Step E c.mem (c.progs i) .ub

/-- One step of one program of the launch, labelled by the program and its event. -/
inductive LStep {N : Nat} (E : Env) : Config N → Fin N × Event → Config N → Prop where
  | step {c : Config N} (i : Fin N) {m' : Mem} {p' : Prog Unit} {e : Event} :
      Step E c.mem (c.progs i) (.next m' p' e) →
      LStep E c (i, e) ⟨m', Function.update c.progs i p'⟩

/-- `Reach E c tr c'`: from `c`, the launch can take the steps labelled `tr` to `c'`. -/
inductive Reach {N : Nat} (E : Env) : Config N → List (Fin N × Event) → Config N → Prop where
  | refl (c : Config N) : Reach E c [] c
  | tail {c c' c'' : Config N} {tr : List (Fin N × Event)} {ev : Fin N × Event} :
      Reach E c tr c' → LStep E c' ev c'' → Reach E c (tr ++ [ev]) c''

/-! ## Races of a trace -/

section Trace
variable {N : Nat} (tr : List (Fin N × Event))

/-- Program order: `a` before `b`, same program. -/
def PO (a b : Nat) : Prop :=
  a < b ∧ ∃ x y, tr[a]? = some x ∧ tr[b]? = some y ∧ x.1 = y.1

/-- Happens-before: program order. Programs do not synchronise with each other (see the module
docstring). -/
def HB : Nat → Nat → Prop := Relation.TransGen (PO tr)

/-- Two atomics that cannot race with each other: both `acq_rel`, both at `gpu` or `sys`
scope, on the same bytes. Atomics whose order is relaxed, or that only acquire or only release,
and whose result the program uses, can be reordered with respect to other locations, which a
sequentially consistent interleaving does not show (store buffering). Atomics on partly
overlapping bytes are not morally strong in PTX. So only this case is exempt from races; for such
atomics the interleaving semantics is exact up to the order of a program's own steps, which is
coarser than each thread's: launches in which a program uses an atomic's result on a cell another
program, or another atomic operation of the same program, also accesses are excluded (library issue
L161, `TVA.Sem.AtomicResult`). An atomic of any order whose result the source never
uses is modelled as `acq_rel` at `gpu`/`sys` scope (`MemSem.unused`, library issue L157), so its
events are `acq_rel` here. The descriptor reductions (`tl.desc_atomic`, relaxed TMA reductions)
are not: their model keeps `relaxed`, and two programs reducing into one element race. -/
def Event.Strong (e f : Event) : Prop :=
  (∃ sc, e.atomic = some (.acqRel, sc) ∧ sc ≠ .cta) ∧ (∃ sc, f.atomic = some (.acqRel, sc) ∧ sc ≠ .cta) ∧
    e.reads = f.reads ∧ e.writes = f.writes

/-- A data race between two programs: conflicting events of different programs, unless they are
`Event.Strong` atomics. (`HB` relates only events of one program, so it never excuses one.) -/
def InterRace : Prop :=
  ∃ a b x y, a < b ∧ tr[a]? = some x ∧ tr[b]? = some y ∧ x.1 ≠ y.1 ∧ x.2.Conflict y.2 ∧
    ¬ Event.Strong x.2 y.2 ∧ ¬ HB tr a b

/-- Two atomics on the same bytes: both events are atomic read-modify-writes, and they read and
write the same bytes. Atomics on partly overlapping bytes are not such a pair. -/
def Event.SameAtomic (e f : Event) : Prop :=
  e.atomic.isSome ∧ f.atomic.isSome ∧ e.reads = f.reads ∧ e.writes = f.writes

/-- A conflict within one program (library issue L15): two events of the same program that
touch the same bytes, at least one writing them, unless both are atomics on the same bytes
(`Event.SameAtomic`). Program order does not excuse it: the program's threads do not order their
global accesses. -/
def IntraRace : Prop :=
  ∃ (a b : Nat) (x y : Fin N × Event), a < b ∧ tr[a]? = some x ∧ tr[b]? = some y ∧ x.1 = y.1 ∧
    x.2.Conflict y.2 ∧ ¬ Event.SameAtomic x.2 y.2

end Trace

/-! ## Outcomes -/

/-- The launch of `k` with `L` from `init` is well behaved: no interleaving reaches undefined
behaviour, a data race between programs, or a conflict within a program. -/
def WellBehaved (E : Env) (k : Kernel) (L : Launch) (init : Mem) : Prop :=
  ∀ tr c, Reach E (Config.start k L init) tr c → ¬ c.StuckUB E ∧ ¬ InterRace tr ∧ ¬ IntraRace tr

/-- What a launch can do, in the semantics. -/
inductive SemOutcome where
  /-- All programs finished, with final memory `fin`. -/
  | ok (fin : Mem)
  /-- Some interleaving reaches undefined behaviour. -/
  | ub
  /-- Some interleaving runs forever. -/
  | hang

/-- The outcomes of launching `k` with `L` from `init` in environment `E`. -/
def Outcomes (E : Env) (k : Kernel) (L : Launch) (init : Mem) : SemOutcome → Prop
  | .ok fin => ∃ tr c, Reach E (Config.start k L init) tr c ∧ c.Done ∧ c.mem = fin
  | .ub => ∃ tr c, Reach E (Config.start k L init) tr c ∧ c.StuckUB E
  | .hang => ∃ f : Nat → Config L.size, f 0 = Config.start k L init ∧
      ∀ n, ∃ ev, LStep E (f n) ev (f (n + 1))

end TVA
