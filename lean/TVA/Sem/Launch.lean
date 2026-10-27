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
interleaving reaches undefined behaviour or a data race between programs (`InterRace`).

A data race is a pair of conflicting accesses from different programs, unless both are
`Event.Strong` atomics: `acq_rel` (Triton's default), at `gpu`/`sys` scope, on the same bytes.
Relaxed, acquire-only and release-only atomics of different programs race: the interleaving
would hide their reorderings. Each program itself is sequential (`TVA.Lang.Prog`).

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

/-- A launch configuration. -/
structure Launch where
  /-- The grid `(x, y, z)`. -/
  grid : Fin 3 → Nat
  /-- `enable_fp_fusion` (default `True`). -/
  fusion : Bool := true
  /-- The `debug` compile option (default `False`; `TRITON_DEBUG=1`). With it, `tl.device_assert`
  and integer overflow checks trap. `Compiler.Faithful` covers only `debug = false`. -/
  debug : Bool := false
  /-- `num_ctas` (default `1`). With more, a program is a cluster of CTAs and `cta`-scope atomics
  are atomic only within one of them; `Launch.Valid` requires `1`. -/
  numCtas : Nat := 1
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

/-- The launch is one the compiler and the driver accept, as far as the library can state it:
the CUDA grid limits (`x < 2^31`, `y, z < 2^16`), `num_ctas = 1`, and block shapes Triton
accepts. Other limits (shared memory, registers) are part of the reviewed `Compiler.Faithful`
instance. -/
def Valid (L : Launch) : Prop :=
  L.grid 0 < 2 ^ 31 ∧ L.grid 1 < 2 ^ 16 ∧ L.grid 2 < 2 ^ 16 ∧ L.numCtas = 1 ∧
    ∀ s ∈ L.blocks, s.TritonOk

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
can be reordered with respect to other locations, which a sequentially consistent interleaving
does not show (store buffering). Atomics on partly overlapping bytes are not morally strong in
PTX. So only this case is exempt from races; for such atomics the interleaving semantics is
exact. -/
def Event.Strong (e f : Event) : Prop :=
  (∃ sc, e.atomic = some (.acqRel, sc) ∧ sc ≠ .cta) ∧ (∃ sc, f.atomic = some (.acqRel, sc) ∧ sc ≠ .cta) ∧
    e.reads = f.reads ∧ e.writes = f.writes

/-- A data race between two programs: conflicting events of different programs, unless they are
`Event.Strong` atomics. (`HB` relates only events of one program, so it never excuses one.) -/
def InterRace : Prop :=
  ∃ a b x y, a < b ∧ tr[a]? = some x ∧ tr[b]? = some y ∧ x.1 ≠ y.1 ∧ x.2.Conflict y.2 ∧
    ¬ Event.Strong x.2 y.2 ∧ ¬ HB tr a b

end Trace

/-! ## Outcomes -/

/-- The launch of `k` with `L` from `init` is well behaved: no interleaving reaches undefined
behaviour or a data race. -/
def WellBehaved (E : Env) (k : Kernel) (L : Launch) (init : Mem) : Prop :=
  ∀ tr c, Reach E (Config.start k L init) tr c → ¬ c.StuckUB E ∧ ¬ InterRace tr

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
