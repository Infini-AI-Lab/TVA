import Solve

/-!
# An executor for v3 models, and its soundness

Phase 3 of `validation/PLAN.md`. `exec` runs one program (`Prog`) on a concrete memory in a fixed
environment `E`; `execLaunch` runs the programs of a launch one after another, in grid order.
Their soundness theorems say that whatever they report, the semantics can do:

* `exec_sound`: a finished run is a sequence of `Step`s to `ret`; a predicted undefined behaviour
  is a sequence of `Step`s to a step whose outcome is `ub`.
* `execLaunch_sound`: a finished launch is a trace (`Reach`) from `Config.start` to a
  configuration where every program has finished, with the reported memory; a predicted
  undefined behaviour is a trace to a configuration that is `StuckUB`.

Nothing here is trusted: `Step` stays the meaning. `exec` is written for any signature with
solvers (`[Sig] [Solvers]`, `Solve.lean`); Triton's are in `Solvers.lean`. Every answer it uses
carries its proof, except the values a solver assumed (`Ans.assumed`, only at platform-dependent
sites). `exec` logs every request it answered (`Record`: occurrence number, request, value,
proved or not), and both theorems are conditional on the assumed ones being allowed
(`Record.Assumptions`). Completeness is not claimed: the executor makes one choice where the
semantics allows many (below), and stops without a claim (`Res.stuck` with the solver's `Gap`,
`Res.outOfFuel`) where it has no answer.

## Where the answers come from

* `pick s k`: the solver of site `s` (`Solvers.solve`), given the oracle's hints for this
  occurrence (`O occ`; requests are numbered in the order the run meets them, across a launch),
  for what it cannot compute (a reduction tree, an order of lanes, a device value).
* `rmw d a u …`: the solver of update `u` (`Solvers.solveUpd`), given the old value.
* `load`: the values of disabled lanes (`Fill`).

`exec` makes the remaining choices itself: a `store` writes its enabled lanes in index order (one
of the orders `Step.store` admits), an atomic is performed where it is written (never
`Step.defer`), and `env` reads `E`. `execLaunch` runs program `0` to the end, then program `1`,
and so on: one interleaving of the launch.
-/
namespace TVA.Validation

open TVA Bytes

/-! ## Concrete memory

A `Mem` is a function, so a memory after `k` writes is `k` nested closures, and every byte read
walks through them. `exec` keeps its memory in a hash table (`CMem`) instead, and reads it
through `CMem.toMem`; `toMem_writeAll` says a store on the table is `writeAll` on the memory. -/

/-- A memory as a table: the bytes present, by address. -/
abbrev CMem := Std.HashMap Int Byte

namespace CMem

/-- The memory of a table: a byte is present when the table has it. -/
def toMem (c : CMem) : Mem := fun x => c[x]?

/-- The bytes `bs` written at `a`, `a + 1`, …. -/
def writeBytes (c : CMem) (a : Int) : List Byte → CMem
  | [] => c
  | b :: bs => writeBytes (c.insert a b) (a + 1) bs

theorem toMem_insert (c : CMem) (a : Int) (b : Byte) (x : Int) :
    toMem (c.insert a b) x = if x = a then some b else toMem c x := by
  simp only [toMem]
  rw [Std.HashMap.getElem?_insert]
  by_cases h : x = a
  · subst h; simp
  · simp [h, Ne.symm h]

theorem toMem_writeBytes (c : CMem) (a : Int) (bs : List Byte) :
    toMem (c.writeBytes a bs) = (toMem c).writeBytes a bs := by
  funext x
  induction bs generalizing c a with
  | nil => simp [writeBytes, Mem.writeBytes]
  | cons b bs ih =>
    rw [writeBytes, ih]
    simp only [Mem.writeBytes, List.length_cons, toMem_insert]
    by_cases hx : x = a
    · subst hx; simp
    · by_cases hin : a + 1 ≤ x ∧ x < a + 1 + bs.length
      · have hk : (x - a).toNat = (x - (a + 1)).toNat + 1 := by omega
        rw [ite_eq_left hin, ite_eq_left (by push_cast; omega), hk, List.getElem?_cons_succ]
      · rw [ite_eq_right hin, ite_eq_right hx, ite_eq_right (by push_cast; omega)]

/-- An element written at `a`. -/
def write (c : CMem) (d : DType) (a : Int) (v : d.Val) : CMem :=
  c.writeBytes a (Bytes.leBytes d.bytes (d.toBits v))

theorem toMem_write (c : CMem) (d : DType) (a : Int) (v : d.Val) :
    toMem (c.write d a v) = (toMem c).write d a v := by
  simp only [write, toMem_writeBytes, Mem.write]

/-- The lanes of a store written in the order `order`, as `writeAll`. -/
def writeAll {s : Shape} (c : CMem) (d : DType) (addrs : Idx s → Option Int)
    (vals : Tensor s d.Val) : List (Idx s) → CMem
  | [] => c
  | i :: rest => match addrs i with
    | some a => writeAll (c.write d a (vals.get i)) d addrs vals rest
    | none => writeAll c d addrs vals rest

theorem toMem_writeAll {s : Shape} (c : CMem) (d : DType) (addrs : Idx s → Option Int)
    (vals : Tensor s d.Val) (order : List (Idx s)) :
    toMem (c.writeAll d addrs vals order) = TVA.writeAll (toMem c) d addrs vals order := by
  induction order generalizing c with
  | nil => rfl
  | cons i rest ih =>
    simp only [writeAll, TVA.writeAll]
    split <;> rename_i h <;> simp [h, ih, toMem_write]

/-- A table of the elements `vals` of dtype `d` at the byte addresses `addrs`, written in order
into `c` (an instance's buffers, for a test). -/
def ofElems (c : CMem) (d : DType) (elems : List (Int × d.Val)) : CMem :=
  elems.foldl (fun c (a, v) => c.write d a v) c

end CMem

/-! ## One program -/


/-- The values of the disabled lanes of a load (any values: `Step.load` constrains only the
enabled lanes). -/
abbrev Fill := (d : DType) → (s : Shape) → Idx s → d.Val

variable [Sig] in
/-- Zero or more steps of one program, none of them undefined behaviour. -/
inductive Runs (E : Env) {α : Type} : Mem → Prog α → Mem → Prog α → Prop where
  | refl (m : Mem) (p : Prog α) : Runs E m p m p
  | step {m m₁ m₂ : Mem} {p p₁ p₂ : Prog α} {e : Event} :
      Step E m p (.next m₁ p₁ e) → Runs E m₁ p₁ m₂ p₂ → Runs E m p m₂ p₂

/-- What `exec` reports. Only `done` and `ub` are claims; the others say why it stopped. -/
inductive Res (α : Type) where
  /-- The program returned `a`, leaving memory `m`. -/
  | done (a : α) (m : CMem)
  /-- The program reaches undefined behaviour, at `what` (an access, or the `occ`-th request,
  whose solver proved it allows nothing). -/
  | ub (what : String)
  /-- The solver of the `occ`-th request had no answer (`Gap`): inconclusive. -/
  | stuck (occ : Nat) (g : Gap)
  /-- Out of fuel: inconclusive. -/
  | outOfFuel

section Access
variable {d : DType} {s : Shape}

/-- Lane `i` is enabled and its access is invalid. -/
def BadLane (m : Mem) (d : DType) (addrs : Idx s → Option Int) (i : Idx s) : Prop :=
  match addrs i with
  | some a => m.read d a = none
  | none => False

instance (m : Mem) (d : DType) (addrs : Idx s → Option Int) (i : Idx s) :
    Decidable (BadLane m d addrs i) := by
  unfold BadLane; split <;> infer_instance

instance (m : Mem) (d : DType) (a : Int) : Decidable (m.Writable d a) := by
  unfold Mem.Writable; infer_instance

/-- Lane `i` is enabled and its address cannot be written. -/
def UnwritableLane (m : Mem) (d : DType) (addrs : Idx s → Option Int) (i : Idx s) : Prop :=
  match addrs i with
  | some a => ¬ m.Writable d a
  | none => False

instance (m : Mem) (d : DType) (addrs : Idx s → Option Int) (i : Idx s) :
    Decidable (UnwritableLane m d addrs i) := by
  unfold UnwritableLane; split <;> infer_instance

/-- The loaded tensor: memory at the enabled lanes, `fill` elsewhere. -/
def loadVals (fill : Fill) (m : Mem) (d : DType) (addrs : Idx s → Option Int) :
    Tensor s d.Val :=
  ⟨fun i => match addrs i with
    | some a => (m.read d a).getD (fill d s i)
    | none => fill d s i⟩

theorem loadOk_of_noBad (fill : Fill) (m : Mem) (d : DType)
    (addrs : Idx s → Option Int) (h : ¬ ∃ i, BadLane m d addrs i) :
    LoadOk m d addrs (loadVals fill m d addrs) := by
  intro i a ha
  have hi : ¬ BadLane m d addrs i := fun hb => h ⟨i, hb⟩
  simp only [BadLane, ha] at hi
  simp only [loadVals, ha]
  cases hr : m.read d a with
  | none => exact absurd hr hi
  | some v => rfl

theorem loadBad_of_bad (m : Mem) (d : DType) (addrs : Idx s → Option Int)
    (h : ∃ i, BadLane m d addrs i) : LoadBad m d addrs := by
  obtain ⟨i, hi⟩ := h
  unfold BadLane at hi
  split at hi
  · rename_i a ha; exact ⟨i, a, ha, hi⟩
  · exact hi.elim

theorem storeOk_of_noUnwritable (m : Mem) (d : DType) (addrs : Idx s → Option Int)
    (h : ¬ ∃ i, UnwritableLane m d addrs i) : StoreOk m d addrs := by
  intro i a ha
  have hi : ¬ UnwritableLane m d addrs i := fun hb => h ⟨i, hb⟩
  simp only [UnwritableLane, ha, not_not] at hi
  exact hi

theorem not_storeOk_of_unwritable (m : Mem) (d : DType) (addrs : Idx s → Option Int)
    (h : ∃ i, UnwritableLane m d addrs i) : ¬ StoreOk m d addrs := by
  obtain ⟨i, hi⟩ := h
  intro hok
  unfold UnwritableLane at hi
  split at hi
  · rename_i a ha; exact hi (hok i a ha)
  · exact hi

end Access

variable [Sig] [Solvers]

/-- Add a record to a run's log. -/
def Record.cons {γ : Type} (r : Record) (p : γ × List Record) : γ × List Record :=
  (p.1, r :: p.2)

theorem Record.cons_eq {γ : Type} {r : Record} {p : γ × List Record} {x : γ}
    {l : List Record} (h : Record.cons r p = (x, l)) : ∃ l', p = (x, l') ∧ l = r :: l' := by
  obtain ⟨x', l'⟩ := p
  simp only [Record.cons, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  exact ⟨l', rfl, rfl⟩


/-- Run `p` from memory `m` for at most `fuel` steps, the next request being the `occ`-th: the
solver of the `occ`-th `pick` or update is given the oracle's hints `O occ`. With the result, the
log of every request answered (`Record`), in order; the assumed ones are the run's assumptions. -/
def exec (E : Env) (O : Oracle) (fill : Fill) : Nat → Nat → {α : Type} → CMem → Prog α →
    Res α × List Record
  | _, _, _, m, .ret a => (.done a m, [])
  | 0, _, _, _, _ => (.outOfFuel, [])
  | n + 1, occ, _, m, .load d _ addrs k =>
    if ∃ i, BadLane m.toMem d addrs i then (.ub "a load of an invalid address", [])
    else exec E O fill n occ m (k (loadVals fill m.toMem d addrs))
  | n + 1, occ, _, m, .store d _ addrs vals k =>
    if ∃ i, UnwritableLane m.toMem d addrs i then (.ub "a store to an unwritable address", [])
    else exec E O fill n occ (m.writeAll d addrs vals (enabled addrs)) k
  | n + 1, occ, _, m, .rmw d a u _ _ k =>
    match m.toMem.read d a with
    | none => (.ub "an atomic on an invalid address", [])
    | some old => match Solvers.solveUpd u old (O occ) with
      | .ok new _ => Record.cons ⟨occ, .upd u old, new, true, fun h => Bool.noConfusion h⟩
          (exec E O fill n (occ + 1) (m.write d a new) (k old))
      | .empty _ => (.ub s!"request {occ}: an update with no new value", [])
      | .none g => (.stuck occ g, [])
      | .assumed new hp => Record.cons ⟨occ, .upd u old, new, false, fun _ => hp⟩
          (exec E O fill n (occ + 1) (m.write d a new) (k old))
  | n + 1, occ, _, m, .pick s k =>
    match Solvers.solve s (O occ) with
    | .ok b _ => Record.cons ⟨occ, .site s, b, true, fun h => Bool.noConfusion h⟩
        (exec E O fill n (occ + 1) m (k b))
    | .empty _ => (.ub s!"request {occ}: a site that allows no value", [])
    | .none g => (.stuck occ g, [])
    | .assumed b hp => Record.cons ⟨occ, .site s, b, false, fun _ => hp⟩
        (exec E O fill n (occ + 1) m (k b))
  | _ + 1, _, _, _, .ub => (.ub "Prog.ub", [])
  | n + 1, occ, _, m, .loop init body k => exec E O fill n occ m (Prog.loopStep init body k)
  | n + 1, occ, _, m, .env k => exec E O fill n occ m (k E)

/-- What `exec` reports, `Step` can do. `stuck` and `outOfFuel` claim nothing. -/
def Sound (E : Env) {α : Type} (m : Mem) (p : Prog α) : Res α → Prop
  | .done a m' => Runs E m p m'.toMem (.ret a)
  | .ub _ => ∃ m' p', Runs E m p m' p' ∧ Step E m' p' .ub
  | .stuck _ _ | .outOfFuel => True

omit [Solvers] in
theorem Sound.cons {E : Env} {α : Type} {m m₁ : Mem} {p p₁ : Prog α} {e : Event} {r : Res α}
    (hs : Step E m p (.next m₁ p₁ e)) (h : Sound E m₁ p₁ r) : Sound E m p r := by
  cases r with
  | done a m' => exact Runs.step hs h
  | ub _ => obtain ⟨m', p', hr, hu⟩ := h; exact ⟨m', p', Runs.step hs hr, hu⟩
  | stuck _ _ => trivial
  | outOfFuel => trivial

omit [Solvers] in
theorem Sound.now {E : Env} {α : Type} {m : Mem} {p : Prog α} {w : String}
    (h : Step E m p .ub) : Sound E m p (.ub w) := ⟨m, p, Runs.refl m p, h⟩

/-- **Soundness of `exec`.** If `exec` reports a result, and every value it assumed is allowed
(`Record.Assumptions`), the program can reach the result under `Step`. -/
theorem exec_sound (E : Env) (O : Oracle) (fill : Fill) :
    ∀ (fuel occ : Nat) {α : Type} (m : CMem) (p : Prog α) (r : Res α) (l : List Record),
      exec E O fill fuel occ m p = (r, l) → Record.Assumptions l → Sound E m.toMem p r := by
  intro fuel
  induction fuel with
  | zero =>
    intro occ α m p r l h _
    cases p <;> simp only [exec, Prod.mk.injEq] at h <;> obtain ⟨rfl, -⟩ := h <;>
      first | exact Runs.refl _ _ | trivial
  | succ n ih =>
    intro occ α m p r l h hl
    cases p with
    | ret a =>
      simp only [exec, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h; exact Runs.refl _ _
    | load d s addrs k =>
      simp only [exec] at h
      split at h
      · rename_i hb
        simp only [Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
        exact Sound.now (Step.loadUB (loadBad_of_bad _ d addrs hb))
      · rename_i hb
        exact Sound.cons (Step.load _ (loadOk_of_noBad fill _ d addrs hb)) (ih _ _ _ _ _ h hl)
    | store d s addrs vals k =>
      simp only [exec] at h
      split at h
      · rename_i hb
        simp only [Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
        exact Sound.now (Step.storeUB (not_storeOk_of_unwritable _ d addrs hb))
      · rename_i hb
        have hs := Step.store (k := k) (vals := vals) _ (storeOk_of_noUnwritable _ d addrs hb)
          (List.Perm.refl (enabled addrs)) (E := E)
        rw [← CMem.toMem_writeAll] at hs
        exact Sound.cons hs (ih _ _ _ _ _ h hl)
    | rmw d a u sem sc k =>
      simp only [exec] at h
      split at h
      · rename_i hr
        simp only [Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
        exact Sound.now (Step.rmwUB (Or.inl hr))
      · rename_i old hr
        split at h
        · rename_i new hu _
          obtain ⟨l', h', rfl⟩ := Record.cons_eq h
          have hs := Step.rmw (E := E) (sem := sem) (sc := sc) (k := k) old new hr hu
          rw [← CMem.toMem_write] at hs
          exact Sound.cons hs (ih _ _ _ _ _ h' hl.tail)
        · rename_i hu _
          simp only [Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
          exact Sound.now (Step.rmwUB (Or.inr ⟨old, hr, hu⟩))
        · simp only [Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h; trivial
        · rename_i new _ _
          obtain ⟨l', h', rfl⟩ := Record.cons_eq h
          have hs := Step.rmw (E := E) (sem := sem) (sc := sc) (k := k) old new hr
            (hl.head rfl)
          rw [← CMem.toMem_write] at hs
          exact Sound.cons hs (ih _ _ _ _ _ h' hl.tail)
    | pick s k =>
      simp only [exec] at h
      split at h
      · rename_i b hb _
        obtain ⟨l', h', rfl⟩ := Record.cons_eq h
        exact Sound.cons (Step.pick b hb) (ih _ _ _ _ _ h' hl.tail)
      · rename_i hb _
        simp only [Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
        exact Sound.now (Step.pickUB hb)
      · simp only [Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h; trivial
      · rename_i b _ _
        obtain ⟨l', h', rfl⟩ := Record.cons_eq h
        exact Sound.cons (Step.pick b (hl.head rfl)) (ih _ _ _ _ _ h' hl.tail)
    | ub =>
      simp only [exec, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h; exact Sound.now Step.ub
    | loop init body k =>
      simp only [exec] at h; exact Sound.cons Step.loop (ih _ _ _ _ _ h hl)
    | env k => simp only [exec] at h; exact Sound.cons Step.env (ih _ _ _ _ _ h hl)

/-! ## A launch, one program after another -/

omit [Solvers] in
/-- Traces compose. -/
theorem Reach.trans {N : Nat} {E : Env} {c₁ c₂ c₃ : Config N} {t₁ t₂ : List (Fin N × Event)}
    (h₁ : Reach E c₁ t₁ c₂) (h₂ : Reach E c₂ t₂ c₃) : Reach E c₁ (t₁ ++ t₂) c₃ := by
  induction h₂ with
  | refl => simpa using h₁
  | tail _ hs ih => rw [← List.append_assoc]; exact .tail ih hs

omit [Solvers] in
/-- The steps of program `i` alone, in a configuration: a trace that changes only the memory
and program `i`. -/
theorem Runs.reach {N : Nat} {E : Env} {m m' : Mem} {p p' : Prog Unit} (h : Runs E m p m' p') :
    ∀ (c : Config N) (i : Fin N), c.mem = m → c.progs i = p →
      ∃ tr, Reach E c tr ⟨m', Function.update c.progs i p'⟩ := by
  induction h with
  | refl m p =>
    intro c i hm hp
    refine ⟨[], ?_⟩
    have : (⟨m, Function.update c.progs i p⟩ : Config N) = c := by
      cases c; subst hm hp; simp
    rw [this]; exact .refl c
  | @step m m₁ m₂ p p₁ p₂ e hs _ ih =>
    intro c i hm hp
    have h1 : LStep E c (i, e) ⟨m₁, Function.update c.progs i p₁⟩ :=
      .step i (by rw [hm, hp]; exact hs)
    obtain ⟨tr, htr⟩ := ih ⟨m₁, Function.update c.progs i p₁⟩ i rfl (by simp)
    refine ⟨[(i, e)] ++ tr, Reach.trans (Reach.tail (Reach.refl c) h1) ?_⟩
    simpa using htr

/-- A configuration with a concrete memory. -/
structure CConfig (N : Nat) where
  /-- The memory. -/
  mem : CMem
  /-- The rest of each program. -/
  progs : Fin N → Prog Unit

/-- The configuration it stands for. -/
def CConfig.toConfig {N : Nat} (c : CConfig N) : Config N := ⟨CMem.toMem c.mem, c.progs⟩

/-- What `execProgs` reports. Only `done` and `ub` are claims. -/
inductive LRes (N : Nat) where
  /-- Every program finished; the configuration reached. -/
  | done (c : CConfig N)
  /-- Program `i` reaches undefined behaviour, at `what`. -/
  | ub (i : Fin N) (what : String)
  /-- Program `i` stopped without a claim (`Res.stuck`, `Res.outOfFuel`). -/
  | stopped (i : Fin N) (r : Res Unit)

/-- Run the programs `is`, one after another, each to its end, from configuration `c`, the next
request being the `occ`-th (numbers continue from one program to the next). With the result, the
log of the requests answered. -/
def execProgs (E : Env) (O : Oracle) (fill : Fill) (fuel : Nat) {N : Nat} :
    List (Fin N) → Nat → CConfig N → LRes N × List Record
  | [], _, c => (.done c, [])
  | i :: is, occ, c => match exec E O fill fuel occ c.mem (c.progs i) with
    | (.done () m', l) =>
      let rest := execProgs E O fill fuel is (occ + l.length) ⟨m', Function.update c.progs i (.ret ())⟩
      (rest.1, l ++ rest.2)
    | (.ub w, l) => (.ub i w, l)
    | (r, l) => (.stopped i r, l)

/-- What `execProgs` reports, the launch can do from `c`. -/
def LSound (E : Env) {N : Nat} (is : List (Fin N)) (c : CConfig N) : LRes N → Prop
  | .done c' => (∃ tr, Reach E c.toConfig tr c'.toConfig) ∧ (∀ i ∈ is, c'.progs i = .ret ()) ∧
      ∀ i, i ∉ is → c'.progs i = c.progs i
  | .ub _ _ => ∃ tr c', Reach E c.toConfig tr c' ∧ c'.StuckUB E
  | .stopped _ _ => True

theorem execProgs_sound (E : Env) (O : Oracle) (fill : Fill) (fuel : Nat) {N : Nat} :
    ∀ (is : List (Fin N)) (occ : Nat) (c : CConfig N) (r : LRes N) (l : List Record),
      execProgs E O fill fuel is occ c = (r, l) → Record.Assumptions l → LSound E is c r := by
  intro is
  induction is with
  | nil =>
    intro occ c r l h _
    simp only [execProgs, Prod.mk.injEq] at h
    obtain ⟨rfl, -⟩ := h
    exact ⟨⟨[], .refl _⟩, fun _ h => absurd h List.not_mem_nil, fun _ _ => rfl⟩
  | cons i is ih =>
    intro occ c r l h hl
    simp only [execProgs] at h
    split at h
    · rename_i m' l₁ hex
      simp only [Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨hl₁, hl₂⟩ := hl.append
      have hs := exec_sound E O fill fuel occ c.mem (c.progs i) _ _ hex hl₁
      obtain ⟨tr₁, htr₁⟩ := Runs.reach hs c.toConfig i rfl rfl
      have hr := ih _ _ _ _ rfl hl₂
      generalize (execProgs E O fill fuel is (occ + l₁.length)
        ⟨m', Function.update c.progs i (.ret ())⟩).1 = r' at hr ⊢
      cases r' with
      | done c' =>
        obtain ⟨⟨tr₂, htr₂⟩, hin, hout⟩ := hr
        refine ⟨⟨tr₁ ++ tr₂, Reach.trans htr₁ htr₂⟩, fun j hj => ?_, fun j hj => ?_⟩
        · rcases List.mem_cons.mp hj with rfl | hj
          · by_cases hjs : j ∈ is
            · exact hin j hjs
            · rw [hout j hjs]; simp
          · exact hin j hj
        · have hji : j ≠ i := fun e => hj (e ▸ List.mem_cons_self)
          have hjs : j ∉ is := fun e => hj (List.mem_cons_of_mem _ e)
          rw [hout j hjs]; simp [Function.update_of_ne hji]
      | ub _ _ =>
        obtain ⟨tr₂, c', htr₂, hu⟩ := hr
        exact ⟨tr₁ ++ tr₂, c', Reach.trans htr₁ htr₂, hu⟩
      | stopped _ _ => trivial
    · rename_i w l₁ hex
      simp only [Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨m', p', hruns, hu⟩ := exec_sound E O fill fuel occ c.mem (c.progs i) _ _ hex hl
      obtain ⟨tr, htr⟩ := Runs.reach hruns c.toConfig i rfl rfl
      exact ⟨tr, _, htr, i, by simpa using hu⟩
    · simp only [Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h
      trivial

/-- Run the launch `L` of kernel `k` from `init`: programs `0, 1, …` in grid order, each for at
most `fuel` steps, requests numbered from `0` across the launch. With the result, the log. -/
def execLaunch (E : Env) (O : Oracle) (fill : Fill) (fuel : Nat) (k : Kernel) (L : Launch)
    (init : CMem) : LRes L.size × List Record :=
  execProgs E O fill fuel (List.finRange L.size) 0 ⟨init, fun i => k.body (L.ctx i)⟩

/-- **Soundness of `execLaunch`.** If every value it assumed is allowed: a finished run is a
trace of the launch to a configuration where every program has finished, with the reported
memory; a reported undefined behaviour is a trace to a configuration where some program's next
step can be undefined behaviour. -/
theorem execLaunch_sound (E : Env) (O : Oracle) (fill : Fill) (fuel : Nat) (k : Kernel)
    (L : Launch) (init : CMem) (l : List Record) (hl : Record.Assumptions l) :
    (∀ c, execLaunch E O fill fuel k L init = (.done c, l) →
      ∃ tr c', Reach E (Config.start k L (CMem.toMem init)) tr c' ∧ c'.Done ∧
        c'.mem = CMem.toMem c.mem) ∧
    (∀ i w, execLaunch E O fill fuel k L init = (.ub i w, l) →
      ∃ tr c', Reach E (Config.start k L (CMem.toMem init)) tr c' ∧ c'.StuckUB E) := by
  have hs := execProgs_sound E O fill fuel (List.finRange L.size) 0
    ⟨init, fun i => k.body (L.ctx i)⟩
  refine ⟨fun c hc => ?_, fun i w hc => hs _ _ hc hl⟩
  obtain ⟨⟨tr, htr⟩, hin, -⟩ := hs _ _ hc hl
  exact ⟨tr, c.toConfig, htr, fun i => hin i (List.mem_finRange i), rfl⟩

end TVA.Validation
