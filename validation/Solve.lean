import TVA

/-!
# Solver answers and replay records

Stage 2 of `validation/PICKS.md`. The semantics (`TVA.Launch`) never looks inside a site: a
`pick s` continues with any value `s` allows. To *run* a model, the executor (`Exec.lean`) needs a
value at each site, and asks a solver keyed on the site's constructor. This module gives the
types, for any signature `Sig`; the solvers for Triton's catalog (`TVA.Site`) come with each
family.

**Four outcomes** (`Ans`). A solver for site `s` returns
1. `ok b h`: a value with a proof that `s` allows it;
2. `empty h`: a proof that `s` allows no value (a predicted undefined behaviour);
3. `none g`: no answer (`Gap`: unsupported, search failed, or a supplied answer that does not fit
   the request), which is inconclusive and never undefined behaviour;
4. `assumed b hp`: a value with no proof, only at a site the signature marks as platform
   dependent (`hp : Solvers.Platform s`).
The same for an atomic's update `u` given the old value. What the executor reports is conditional
on outcome 4 only: each assumed value is recorded (`Record`), and `exec_sound` assumes them.

**Replay.** What a solver cannot compute it asks the oracle (`Hints`): a question from the
signature's closed family `Solvers.Query` (a tree over `n` leaves, a lane order, a device value
for given operands). The executor numbers the sites it meets (occurrences), so repeated identical
requests may receive different answers; the number is not part of the site. A supplied answer is
checked against the request, and one that does not fit is a `Gap.badAnswer`, not a reason to use
a default: a default is used only when the oracle supplied nothing.
-/
namespace TVA.Validation

open TVA

/-- Why a solver gave no answer. All three are inconclusive: the executor stops and claims
nothing. -/
inductive Gap where
  /-- The executor has no solver for this kind of site. -/
  | unsupported (what : String)
  /-- The solver searched and found neither a value nor a proof that there is none. -/
  | searchFailed (what : String)
  /-- The oracle supplied an answer that does not fit the request (a tree over the wrong number
  of leaves, an order that is not a permutation of the enabled lanes, a value the solver can
  refute). -/
  | badAnswer (what : String)
  /-- A question only the oracle can answer (a device value at a platform-dependent site) got no
  answer. -/
  | noOracle (what : String)
  deriving Repr, Inhabited

/-- An answer to "some `b` with `P b`", where an unproved value is allowed only when `plat`
holds (the request is platform dependent). -/
inductive Ans {β : Type} (P : β → Prop) (plat : Prop) : Type where
  /-- A value satisfying `P`. -/
  | ok (b : β) (h : P b)
  /-- No value satisfies `P`: undefined behaviour. -/
  | empty (h : ∀ b, ¬ P b)
  /-- No answer. -/
  | none (g : Gap)
  /-- A value assumed to satisfy `P`, with no proof, at a platform-dependent request. -/
  | assumed (b : β) (hp : plat)

namespace Ans

variable {β γ : Type} {P : β → Prop} {Q : γ → Prop} {plat plat' : Prop}

/-- The value, certified or assumed. -/
def val? : Ans P plat → Option β
  | .ok b _ | .assumed b _ => some b
  | _ => Option.none

/-- The answer is certified (`ok` or `empty`): it carries its proof. -/
def Certified : Ans P plat → Prop
  | .ok _ _ | .empty _ => True
  | _ => False

/-- A certified value, through `f`, for a relation `Q` that `f` maps `P` into; an empty answer
becomes `none` (the image need not be empty), an assumed one stays assumed. -/
def map (a : Ans P plat) (f : β → γ) (hf : ∀ b, P b → Q (f b)) (hp : plat → plat') :
    Ans Q plat' :=
  match a with
  | .ok b h => .ok (f b) (hf b h)
  | .empty _ => .none (.searchFailed "empty before a map")
  | .none g => .none g
  | .assumed b h => .assumed (f b) (hp h)

/-- The same answer for an equivalent predicate (an `Allows` equivalence theorem). -/
def cast (a : Ans P plat) {P' : β → Prop} (h : ∀ b, P b ↔ P' b) (hp : plat → plat') :
    Ans P' plat' :=
  match a with
  | .ok b hb => .ok b ((h b).mp hb)
  | .empty hn => .empty fun b hb => hn b ((h b).mpr hb)
  | .none g => .none g
  | .assumed b h => .assumed b (hp h)

/-- The same answer through a bijection-like pair: `f` maps `P` into `Q` and `g` maps `Q` back
into `P`, so an empty answer stays empty (unlike `map`). For a tensor read as a function. -/
def transport (a : Ans P plat) (f : β → γ) (g : γ → β) (hf : ∀ b, P b → Q (f b))
    (hg : ∀ c, Q c → P (g c)) (hp : plat → plat') : Ans Q plat' :=
  match a with
  | .ok b h => .ok (f b) (hf b h)
  | .empty hn => .empty fun c hc => hn (g c) (hg c hc)
  | .none g => .none g
  | .assumed b h => .assumed (f b) (hp h)

/-- Which outcome, for reports and tests: `ok`, `empty`, `assumed`, or the gap. -/
def tag : Ans P plat → String
  | .ok _ _ => "ok"
  | .empty _ => "empty"
  | .none g => s!"none: {repr g}"
  | .assumed _ _ => "assumed"

/-- The answer is `ok`. -/
def isOk : Ans P plat → Bool
  | .ok _ _ => true
  | _ => false

/-- The answer is `empty`. -/
def isEmpty : Ans P plat → Bool
  | .empty _ => true
  | _ => false

/-- The gap of a `none` answer. -/
def gap? : Ans P plat → Option Gap
  | .none g => some g
  | _ => Option.none

theorem isEmpty_spec : {a : Ans P plat} → a.isEmpty = true → ∀ b, ¬ P b
  | .empty h, _ => h

theorem ok_spec : {a : Ans P plat} → a.isOk = true → (h : a.val?.isSome = true) →
    P (a.val?.get h)
  | .ok _ hb, _, _ => hb

theorem plat_of : {a : Ans P plat} → a.val?.isSome = true → a.isOk = false → plat
  | .assumed _ hp, _, _ => hp

/-- **Lane by lane**: a function whose value at `i` satisfies `R i`, from an answer at each `i`.
Empty when some lane is empty (no lane value, so no function); `ok` when every lane is; assumed
when every lane has a value and some are assumed; otherwise the gap of the first lane of `elems`
(the indices in order, used only to report) without an answer. -/
def pi {ι α : Type} [Fintype ι] {R : ι → α → Prop} (elems : List ι)
    (f : (i : ι) → Ans (R i) plat) : Ans (fun g : ι → α => ∀ i, R i (g i)) plat :=
  if hex : ∃ i, (f i).isEmpty = true then
    .empty fun g hg => by
      obtain ⟨i, hi⟩ := hex
      exact isEmpty_spec hi _ (hg i)
  else if hall : ∀ i, (f i).val?.isSome = true then
    if hok : ∀ i, (f i).isOk = true then
      .ok (fun i => (f i).val?.get (hall i)) fun i => ok_spec (hok i) (hall i)
    else
      .assumed (fun i => (f i).val?.get (hall i)) (by
        obtain ⟨i, hi⟩ := Classical.not_forall.mp hok
        exact plat_of (hall i) (by simpa using hi))
  else .none ((elems.findSome? fun i => (f i).gap?).getD (.searchFailed "a lane has no answer"))

/-- `b` is the only candidate: decide it. -/
def only (b : β) [Decidable (P b)] (h : ∀ b', P b' → P b) : Ans P plat :=
  if hb : P b then .ok b hb else .empty fun b' hb' => hb (h b' hb')

/-- Try `b`; `none` (search failed) if it fails. -/
def check (b : β) [Decidable (P b)] (what : String) : Ans P plat :=
  if hb : P b then .ok b hb else .none (.searchFailed what)

end Ans

/-! ## The oracle -/

/-- An oracle for a family of questions `Q`: `none` when it has no answer. -/
abbrev Hints (Q : Type → Type) := {τ : Type} → Q τ → Option τ

/-- **What an executor needs from a signature**: which sites are platform dependent, what a
solver may ask, and a solver for every site and every update. -/
class Solvers [Sig] where
  /-- The questions a solver may ask the oracle (closed, so that the oracle can match on them
  after compilation). -/
  Query : Type → Type
  /-- The sites at which an unproved answer is allowed: those whose relation mentions the
  platform. -/
  Platform : Sig.Site → Prop
  /-- The updates at which an unproved answer is allowed. -/
  UpdPlatform : {d : DType} → Sig.Upd d → Prop
  /-- The solver for a site. -/
  solve : (s : Sig.Site) → Hints Query → Ans (Sig.Allows s) (Platform s)
  /-- The solver for an update, given the old value. -/
  solveUpd : {d : DType} → (u : Sig.Upd d) → (old : d.Val) → Hints Query →
    Ans (Sig.UpdAllows u old) (UpdPlatform u)

variable [Sig] [Solvers]

/-- An oracle for one run: the answers for the `n`-th site the executor meets (`n` counts the
`pick`s and updates of the run, in order). -/
abbrev Oracle := Nat → Hints Solvers.Query

/-- The oracle that answers nothing: every solver uses its defaults. -/
def Oracle.empty : Oracle := fun _ _ _ => Option.none

/-! ## Replay records -/

/-- What was asked: a site, or an update with its old value. -/
inductive Request where
  /-- A `pick` at site `s`. -/
  | site (s : Sig.Site)
  /-- An atomic's update `u` of an element of dtype `d` holding `old`. -/
  | upd {d : DType} (u : Sig.Upd d) (old : d.Val)

/-- The type of a request's answer. -/
def Request.Out : Request → Type
  | .site s => Sig.Out s
  | .upd (d := d) _ _ => d.Val

/-- What a request allows. -/
def Request.Allows : (r : Request) → r.Out → Prop
  | .site s => Sig.Allows s
  | .upd u old => Sig.UpdAllows u old

/-- The request is platform dependent: an unproved answer is allowed. -/
def Request.Platform : Request → Prop
  | .site s => Solvers.Platform s
  | .upd u _ => Solvers.UpdPlatform u

/-- **A replay record**: the `occ`-th request of the run (`req`: the operation, its operands and
its configuration, as the site carries them), and the value the executor continued with. `proved`
says whether the value came with a proof; an unproved value is an assumption of the run. -/
structure Record where
  /-- The occurrence number. -/
  occ : Nat
  /-- The request. -/
  req : Request
  /-- The value returned. -/
  val : req.Out
  /-- `true` when the value is certified (`Ans.ok`), `false` when assumed (`Ans.assumed`). -/
  proved : Bool
  /-- An assumed value is at a platform-dependent request. -/
  plat : proved = false → req.Platform

/-- The record's value is allowed. For a proved record this holds by construction (the executor
had the proof); for an assumed one, it is the assumption. -/
def Record.Holds (r : Record) : Prop := r.req.Allows r.val

/-- Every assumed value of a log is allowed: the condition under which an executor's report is
a run of the semantics. -/
def Record.Assumptions (l : List Record) : Prop := ∀ r ∈ l, r.proved = false → r.Holds

theorem Record.Assumptions.nil : Record.Assumptions [] := fun _ h => absurd h List.not_mem_nil

theorem Record.Assumptions.tail {r : Record} {l : List Record}
    (h : Assumptions (r :: l)) : Assumptions l :=
  fun x hx => h x (List.mem_cons_of_mem _ hx)

theorem Record.Assumptions.head {r : Record} {l : List Record}
    (h : Assumptions (r :: l)) (hp : r.proved = false) : r.Holds :=
  h r List.mem_cons_self hp

theorem Record.Assumptions.append {l l' : List Record} (h : Assumptions (l ++ l')) :
    Assumptions l ∧ Assumptions l' :=
  ⟨fun x hx => h x (List.mem_append_left _ hx), fun x hx => h x (List.mem_append_right _ hx)⟩

end TVA.Validation
