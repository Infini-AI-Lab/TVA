import Exec
import Lean.Data.Json

/-!
# Verdicts: what a differential test concludes about one case

Stage 2, step 5 of `validation/PICKS.md`. `exec_sound` is the executor's contract: whatever
`exec` reports is a run of the semantics, provided the values it assumed are allowed. A test
compares that report with the device's, and concludes one of five things (`Verdict`):

1. `certified`: the model's run agrees with the device and every answer of the run was proved:
   the device's behaviour is one the model allows (`ofRes_certified`).
2. `conditional n`: the run agrees, and `n` of its answers were assumed (platform values: the
   device's own outputs for `exp`, `/`, `tl.dot`, fp8, float atomics). The agreement holds if the
   assumed values are allowed (`ofRes_conditional`); whether they are is the device facts'
   question (`evidence/num-*`), checked per logged value.
3. `discrepancy n what`: the run finished (or predicted undefined behaviour) and disagrees with
   the device. A *candidate* finding, to investigate and bring to zhuominc: the model is a
   relation, and the run is one of its executions (`ofRes_discrepancy`), so a different choice at
   some site may still agree; `n` counts the assumed answers it rests on.
4. `inconclusive what`: a solver found neither a value nor a proof of none, the oracle gave no
   answer or one that does not fit (`Gap.searchFailed`, `noOracle`, `badAnswer`), or a test's
   own search (candidate trees) found no agreeing choice. Says nothing about the model.
5. `unsupported what`: a site with no solver (`Gap.unsupported`), or the fuel ran out. Says
   nothing about the model either.

Only 1–3 are claims about the model; 4 and 5 are about the harness. Nothing here is trusted: a
verdict is computed from `exec`'s report, and the theorems below say what each claims.
-/
namespace TVA.Validation

open TVA Lean

/-- What a differential test concludes about one case. -/
inductive Verdict where
  /-- Agreement; every answer proved. -/
  | certified
  /-- Agreement, given `assumed` platform answers. -/
  | conditional (assumed : Nat)
  /-- Disagreement (or a predicted undefined behaviour) on one execution of the model, resting on
  `assumed` platform answers: a candidate finding. -/
  | discrepancy (assumed : Nat) (what : String)
  /-- No answer from a solver, the oracle, or the test's own search. -/
  | inconclusive (what : String)
  /-- No solver for a site, or out of fuel. -/
  | unsupported (what : String)
  deriving Repr, BEq, Inhabited

namespace Verdict

/-- The verdict's name, as reports write it. -/
def name : Verdict → String
  | .certified => "certified"
  | .conditional _ => "conditional"
  | .discrepancy .. => "discrepancy"
  | .inconclusive _ => "inconclusive"
  | .unsupported _ => "unsupported"

/-- The verdict is an agreement (`certified` or `conditional`). -/
def agrees : Verdict → Bool
  | .certified | .conditional _ => true
  | _ => false

/-- The verdict is a candidate finding. -/
def isDiscrepancy : Verdict → Bool
  | .discrepancy .. => true
  | _ => false

/-- Agreement resting on `n` assumed answers. -/
def agreeWith (n : Nat) : Verdict := if n = 0 then .certified else .conditional n

/-- How much a verdict says against the model, for `join`: a discrepancy over everything, then
the harness's failures, then agreements. -/
def rank : Verdict → Nat
  | .certified => 0
  | .conditional _ => 1
  | .inconclusive _ => 2
  | .unsupported _ => 3
  | .discrepancy .. => 4

/-- The number of assumed answers a verdict rests on. -/
def assumed : Verdict → Nat
  | .conditional n | .discrepancy n _ => n
  | _ => 0

/-- The verdict of a case made of two parts (lanes run one at a time): the part that says most
against the model, with the assumed answers of both. -/
def join (v w : Verdict) : Verdict :=
  let n := v.assumed + w.assumed
  match if w.rank > v.rank then w else v with
  | .discrepancy _ what => .discrepancy n what
  | .certified | .conditional _ => agreeWith n
  | u => u

instance : ToJson Verdict where
  toJson v := Json.mkObj <|
    [("verdict", toJson v.name)] ++
    (match v with
      | .conditional n => [("assumed", toJson n)]
      | .discrepancy n w => [("assumed", toJson n), ("what", toJson w)]
      | .inconclusive w | .unsupported w => [("what", toJson w)]
      | .certified => [])

/-- A gap's verdict: `unsupported` for a site with no solver, `inconclusive` otherwise. -/
def ofGap (occ : Nat) : Gap → Verdict
  | .unsupported w => .unsupported s!"request {occ}: no solver: {w}"
  | .searchFailed w => .inconclusive s!"request {occ}: search failed: {w}"
  | .badAnswer w => .inconclusive s!"request {occ}: the oracle's answer does not fit: {w}"
  | .noOracle w => .inconclusive s!"request {occ}: no answer from the oracle: {w}"

end Verdict

theorem Verdict.agreeWith_certified {n : Nat} (h : Verdict.agreeWith n = .certified) : n = 0 := by
  unfold Verdict.agreeWith at h; split at h <;> simp_all

theorem Verdict.agreeWith_agrees (n : Nat) : (Verdict.agreeWith n).agrees = true := by
  unfold Verdict.agreeWith; split <;> rfl

variable [Sig] [Solvers]

/-- The number of assumed (unproved) answers in a log. -/
def Record.assumedCount (l : List Record) : Nat := l.countP (·.proved = false)

theorem Record.Assumptions.of_assumedCount {l : List Record} (h : Record.assumedCount l = 0) :
    Record.Assumptions l := by
  intro r hr hp
  have := List.countP_eq_zero.mp h r hr
  simp [hp] at this

/-- **The verdict of a run** of `exec`, given `agree a m`: `none` when the result `a` and final
memory `m` agree with the device, else what differs. A predicted undefined behaviour is a
discrepancy: call this only where the device's run is defined (a test that localises undefined
behaviour handles `Res.ub` itself). -/
def Verdict.ofRes {α : Type} (agree : α → CMem → Option String) : Res α × List Record → Verdict
  | (.done a m, l) => match agree a m with
    | none => .agreeWith (Record.assumedCount l)
    | some w => .discrepancy (Record.assumedCount l) w
  | (.ub w, l) => .discrepancy (Record.assumedCount l) s!"the model predicts undefined behaviour: {w}"
  | (.stuck occ g, _) => .ofGap occ g
  | (.outOfFuel, _) => .unsupported "out of fuel"

/-- **The verdict of a launch** (`execLaunch`), given `agree c` on the final configuration. -/
def Verdict.ofLaunch {N : Nat} (agree : CConfig N → Option String) : LRes N × List Record → Verdict
  | (.done c, l) => match agree c with
    | none => .agreeWith (Record.assumedCount l)
    | some w => .discrepancy (Record.assumedCount l) w
  | (.ub i w, l) => .discrepancy (Record.assumedCount l)
      s!"the model predicts undefined behaviour in program {i}: {w}"
  | (.stopped i r, _) => match r with
    | .stuck occ g => match Verdict.ofGap occ g with
      | .unsupported w => .unsupported s!"program {i}: {w}"
      | .inconclusive w => .inconclusive s!"program {i}: {w}"
      | _ => .inconclusive s!"program {i}: stuck"
    | .outOfFuel => .unsupported s!"program {i}: out of fuel"
    | _ => .inconclusive s!"program {i}: stopped"  -- `execProgs` stops only at `stuck`, `outOfFuel`

/-! ## What the verdicts claim -/

section
variable {E : Env} {O : Oracle} {fill : Fill} {fuel occ : Nat} {α : Type} {m : CMem}
  {p : Prog α} {agree : α → CMem → Option String}

/-- An agreeing verdict is a finished run that agrees. -/
theorem Verdict.ofRes_agrees {x : Res α × List Record} (h : (Verdict.ofRes agree x).agrees = true) :
    ∃ a m', x.1 = .done a m' ∧ agree a m' = none := by
  obtain ⟨r, l⟩ := x
  cases r with
  | done a m' =>
    simp only [Verdict.ofRes] at h
    split at h
    · exact ⟨a, m', rfl, by assumption⟩
    · simp [Verdict.agrees] at h
  | ub _ => simp [Verdict.ofRes, Verdict.agrees] at h
  | stuck _ g => cases g <;> simp [Verdict.ofRes, Verdict.ofGap, Verdict.agrees] at h
  | outOfFuel => simp [Verdict.ofRes, Verdict.agrees] at h

/-- **Certified agreement**: the program can reach, under `Step`, a final state that agrees with
the device, with no assumption. -/
theorem Verdict.ofRes_certified (h : Verdict.ofRes agree (exec E O fill fuel occ m p) = .certified) :
    ∃ a m', agree a m' = none ∧ Runs E m.toMem p m'.toMem (.ret a) := by
  generalize hx : exec E O fill fuel occ m p = x at h
  obtain ⟨r, l⟩ := x
  obtain ⟨a, m', hr, ha⟩ := Verdict.ofRes_agrees (agree := agree) (x := (r, l)) (by rw [h]; rfl)
  subst hr
  have hl : Record.assumedCount l = 0 := by
    simp only [Verdict.ofRes, ha] at h; exact Verdict.agreeWith_certified h
  exact ⟨a, m', ha, exec_sound E O fill fuel occ m p _ l hx (.of_assumedCount hl)⟩

/-- **Conditional agreement** (and certified): if the assumed values of the run are allowed, the
program can reach a final state that agrees with the device. -/
theorem Verdict.ofRes_conditional
    (h : (Verdict.ofRes agree (exec E O fill fuel occ m p)).agrees = true)
    (hl : Record.Assumptions (exec E O fill fuel occ m p).2) :
    ∃ a m', agree a m' = none ∧ Runs E m.toMem p m'.toMem (.ret a) := by
  obtain ⟨a, m', hr, ha⟩ := Verdict.ofRes_agrees h
  refine ⟨a, m', ha, ?_⟩
  have := exec_sound E O fill fuel occ m p _ _ rfl hl
  rw [hr] at this
  exact this

/-- **A candidate discrepancy**: if the assumed values of the run are allowed, the program can
reach a final state that disagrees with the device, or a step that is undefined behaviour. (Only
one execution: the model may allow others that agree.) -/
theorem Verdict.ofRes_discrepancy {n : Nat} {w : String}
    (h : Verdict.ofRes agree (exec E O fill fuel occ m p) = .discrepancy n w)
    (hl : Record.Assumptions (exec E O fill fuel occ m p).2) :
    (∃ a m', agree a m' ≠ none ∧ Runs E m.toMem p m'.toMem (.ret a)) ∨
      (∃ m' p', Runs E m.toMem p m' p' ∧ Step E m' p' .ub) := by
  have hs := exec_sound E O fill fuel occ m p _ _ rfl hl
  generalize hx : exec E O fill fuel occ m p = x at h hs
  obtain ⟨r, l⟩ := x
  cases r with
  | done a m' =>
    simp only [Verdict.ofRes] at h
    split at h
    · rename_i hn; unfold Verdict.agreeWith at h; split at h <;> simp at h
    · rename_i w' hw; exact .inl ⟨a, m', by simp [hw], hs⟩
  | ub _ => exact .inr hs
  | stuck _ g => cases g <;> simp [Verdict.ofRes, Verdict.ofGap] at h
  | outOfFuel => simp [Verdict.ofRes] at h

end

/-- **Certified agreement of a launch**: a trace of the launch from the start to a configuration
where every program has finished, with a memory that agrees with the device, with no
assumption. -/
theorem Verdict.ofLaunch_certified {E : Env} {O : Oracle} {fill : Fill} {fuel : Nat} {k : Kernel}
    {L : Launch} {init : CMem} {agree : CConfig L.size → Option String}
    (h : Verdict.ofLaunch agree (execLaunch E O fill fuel k L init) = .certified) :
    ∃ c tr c', agree c = none ∧ Reach E (Config.start k L (CMem.toMem init)) tr c' ∧ c'.Done ∧
      c'.mem = CMem.toMem c.mem := by
  generalize hx : execLaunch E O fill fuel k L init = x at h
  obtain ⟨r, l⟩ := x
  cases r with
  | done c =>
    simp only [Verdict.ofLaunch] at h
    split at h
    · rename_i ha
      have hl := Verdict.agreeWith_certified h
      obtain ⟨tr, c', h1, h2, h3⟩ :=
        (execLaunch_sound E O fill fuel k L init l (.of_assumedCount hl)).1 c hx
      exact ⟨c, tr, c', ha, h1, h2, h3⟩
    · simp at h
  | ub _ _ => simp [Verdict.ofLaunch] at h
  | stopped i r =>
    simp only [Verdict.ofLaunch] at h
    cases r with
    | stuck _ g => cases g <;> simp [Verdict.ofGap] at h
    | _ => simp at h

/-- One line of a test's report, as `run_test.py` reads it: `verdict <json>`. -/
def Verdict.line (test : String) (v : Verdict) : String :=
  "verdict " ++ (Json.mkObj [("test", toJson test), ("result", toJson v)]).compress

end TVA.Validation
