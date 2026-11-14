import TVA.Launch.Run

/-!
# Well-behaved launches, outcomes, and `Correct`

The interleaving semantics (`TVA.Launch.Run`) is the meaning of a launch only when the launch is
*well behaved* (`WellBehaved`):

1. no reachable state has a program whose next step is undefined behaviour;
2. no trace has a race between programs (`InterRace`);
3. no trace has a conflict within one program (`IntraRace`);
4. no program uses an atomic's result on a cell another access also reaches
   (`UsesSharedAtomicResult`).

`TVA.Launch` explains why each is there. `Outcomes` lists how a launch can end in the semantics
(`ok fin`, `ub`, `hang`), and `Correct` is what a claim states of each covered launch: valid, well
behaved, and every outcome a normal end whose memory satisfies the specification.
-/
namespace TVA

/-! ## Races of a trace -/

section Trace
variable {N : Nat} (tr : List (Fin N × Event))

/-- Program order on the positions of a trace: `a` comes before `b` and both are steps of the
same program. -/
def PO (a b : Nat) : Prop :=
  a < b ∧ ∃ x y, tr[a]? = some x ∧ tr[b]? = some y ∧ x.1 = y.1

/-- Happens-before on the positions of a trace: program order only. Programs never synchronise
with each other: a scalar atomic is issued by one thread of its program with no barrier before
it (`triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:756-787`), so its release does not cover the
other threads' earlier writes, and Triton has no other inter-program synchronisation. So `HB` never
orders steps of different programs, and handing data from one program to another behind a flag or
a counter is a race. -/
def HB : Nat → Nat → Prop := Relation.TransGen (PO tr)

/-- Two atomics of different programs that do not race: both `acq_rel` (Triton's default), both
at `gpu` or `sys` scope, on exactly the same bytes. Each is then one indivisible step in the
order of the cell's updates (PTX's morally strong atomics), as in an interleaving. A `relaxed`,
`acquire` or `release` atomic may be reordered on the device with its program's accesses to other
bytes, which an interleaving does not show (store buffering), so it races; an atomic whose result
the source never uses is modelled as `acq_rel` (`MemSem.unused` says why that is sound). A
`cta`-scope atomic is not atomic with respect to other programs, and atomics on partly
overlapping bytes are not morally strong, so both race. -/
def Event.Strong (e f : Event) : Prop :=
  (∃ sc, e.atomic = some (.acqRel, sc) ∧ sc ≠ .cta) ∧
    (∃ sc, f.atomic = some (.acqRel, sc) ∧ sc ≠ .cta) ∧
    e.reads = f.reads ∧ e.writes = f.writes

/-- **A race between programs**: two conflicting steps of different programs, unless they are
`Event.Strong` atomics, and not ordered by happens-before (which never orders steps of different
programs, so this conjunct never excuses a pair). -/
def InterRace : Prop :=
  ∃ a b x y, a < b ∧ tr[a]? = some x ∧ tr[b]? = some y ∧ x.1 ≠ y.1 ∧ x.2.Conflict y.2 ∧
    ¬ Event.Strong x.2 y.2 ∧ ¬ HB tr a b

/-- Two atomics on exactly the same bytes (of any `sem` and `scope`). -/
def Event.SameAtomic (e f : Event) : Prop :=
  e.atomic.isSome ∧ f.atomic.isSome ∧ e.reads = f.reads ∧ e.writes = f.writes

/-- **A conflict within a program**: two conflicting steps of the same program, unless both are
atomics on exactly the same bytes. Program order does not excuse the pair: the program's threads
do not order its accesses to global memory. Steps of different loop iterations count. -/
def IntraRace : Prop :=
  ∃ (a b : Nat) (x y : Fin N × Event), a < b ∧ tr[a]? = some x ∧ tr[b]? = some y ∧ x.1 = y.1 ∧
    x.2.Conflict y.2 ∧ ¬ Event.SameAtomic x.2 y.2

/-- The bytes `S` are reached, in the trace, by an access other than one given step of program
`i`: by a step of another program, or by two steps of program `i`. -/
def SharedIn (i : Fin N) (S : Set Int) : Prop :=
  (∃ x ∈ tr, x.1 ≠ i ∧ x.2.Touches S) ∨
    ∃ (a b : Nat) (x y : Fin N × Event), a < b ∧ tr[a]? = some x ∧ tr[b]? = some y ∧
      x.1 = i ∧ y.1 = i ∧
      x.2.Touches S ∧ y.2.Touches S

end Trace

/-- **A program uses an atomic's result on a cell another access also reaches.** Some
interleaving reaches a state in which program `i` is about to perform a read-modify-write of the
element of dtype `d` at `a` whose continuation `kk` is not the same for every old value (the
program uses the result; a continuation that does not depend on it stays so in every state,
`Step.defer`), and some interleaving has a step of another program that touches a
byte of that element, or two steps of program `i` that do.

`Step.defer` lets later steps pass a pending atomic only when they do not depend on its result,
and a tensor step that uses one lane's result holds back every later step of its program. On the
device each thread orders only its own accesses, so another thread of the program may perform a
later atomic before the atomic whose result this thread uses. When the cell is reached by
anything else, that order can show in the old value, and the semantics lacks it. The model does
not tell the lanes of one `tl.atomic_*` call apart from separate calls, so a second lane of the
same call on the same cell counts as another access too. -/
def UsesSharedAtomicResult (E : Env) (k : Kernel) (L : Launch) (init : Mem) : Prop :=
  ∃ (tr : List (Fin L.size × Event)) (c : Config L.size) (i : Fin L.size) (d : DType) (a : Int)
    (upd : d.Val → d.Val → Prop) (sv : (old : d.Val) → Solve (upd old)) (sem : MemSem)
    (sc : Scope) (kk : d.Val → Prog Unit),
    Reach E (Config.start k L init) tr c ∧ c.progs i = .rmw d a upd sv sem sc kk ∧
    (∃ v w, kk v ≠ kk w) ∧
    ∃ (tr' : List (Fin L.size × Event)) (c' : Config L.size),
      Reach E (Config.start k L init) tr' c' ∧ SharedIn tr' i (Mem.span d a)

/-! ## Well-behaved launches and their outcomes -/

/-- **The launch `L` of kernel `k` from memory `init`, in environment `E`, is well behaved**:
the semantics is its meaning. See `TVA.Launch` for why each part is there. -/
structure WellBehaved (E : Env) (k : Kernel) (L : Launch) (init : Mem) : Prop where
  /-- No interleaving reaches undefined behaviour (an invalid access, an impossible `pick`, …). -/
  noUB : ∀ tr c, Reach E (Config.start k L init) tr c → ¬ c.StuckUB E
  /-- No interleaving has a race between programs. -/
  noInterRace : ∀ tr c, Reach E (Config.start k L init) tr c → ¬ InterRace tr
  /-- No interleaving has a conflict within a program. -/
  noIntraRace : ∀ tr c, Reach E (Config.start k L init) tr c → ¬ IntraRace tr
  /-- No program uses an atomic's result on a cell another access also reaches. -/
  noSharedAtomicResult : ¬ UsesSharedAtomicResult E k L init

/-- How a launch can end, in the semantics. -/
inductive Outcome where
  /-- Every program finished; the final memory is `fin`. -/
  | ok (fin : Mem)
  /-- Some interleaving reaches undefined behaviour (on the device: a fault, garbage, anything). -/
  | ub
  /-- Some interleaving runs forever. -/
  | hang

/-- The outcomes of the launch `L` of kernel `k` from memory `init` in environment `E`. -/
def Outcomes (E : Env) (k : Kernel) (L : Launch) (init : Mem) : Outcome → Prop
  | .ok fin => ∃ tr c, Reach E (Config.start k L init) tr c ∧ c.Done ∧ c.mem = fin
  | .ub => ∃ tr c, Reach E (Config.start k L init) tr c ∧ c.StuckUB E
  | .hang => ∃ f : Nat → Config L.size, f 0 = Config.start k L init ∧
      ∀ n, ∃ ev, LStep E (f n) ev (f (n + 1))

/-- The launch `L` of kernel `k` from memory `init`, under environment `E`, is well behaved, and
every way it can end is a normal end whose final memory satisfies `Q`. Each part is needed:

* `L.Valid`: otherwise the device does not run the launch as the semantics does (a grid of
  `2^31` to `2^32` programs runs nothing, `debug` adds traps, …), so the semantics says nothing
  about it;
* `WellBehaved`: otherwise the device has behaviours the interleaving lacks (torn or reordered
  accesses in a race, stale reads within a program, atomics in an order the semantics does not
  give), so a property of the semantics' outcomes would not be one of the device's;
* every outcome is `ok fin` with `Q fin`: the launch does not fault (undefined behaviour is the
  outcome `ub`), does not hang (some interleaving running forever is `hang`), and every final
  memory it can leave satisfies `Q`.

A well-behaved launch always has an outcome (every program that has not finished can take a step
or reach undefined behaviour), so `Correct` is not satisfied vacuously by a launch with none. -/
def Correct (E : Env) (k : Kernel) (L : Launch) (init : Mem) (Q : Mem → Prop) : Prop :=
  L.Valid ∧ WellBehaved E k L init ∧ ∀ o, Outcomes E k L init o → ∃ fin, o = .ok fin ∧ Q fin

end TVA
