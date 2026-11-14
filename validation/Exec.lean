import TVA

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

Nothing here is trusted: `Step` stays the meaning. Every answer the executor uses carries its
proof, except the answers taken without one (`Ans.assume`, at platform sites): `exec` returns them
(`Assumed`), and both theorems are conditional on each of them satisfying its site's predicate. Completeness is not claimed: the executor makes one choice where the semantics allows
many (below), and gives up (`none`) where it has no answer.

## Where the answers come from

* `pick P sv k`: the site's solver `sv` (`TVA.Solve`), given the oracle's `Hints` for what it
  cannot compute (a reduction tree, an order of lanes, a padding value).
  A platform operation (`Prog.plat q k`, an `env` then a `pick`) asks `Query.plat q` and assumes
  the answer: the device's value for those operands (`P_dev`, a table).
* `rmw d a upd sv …`: the atomic's solver `sv old`, given the old value.
* `load`: the values of disabled lanes (`Oracle.fill`).

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

/-- The choices `exec` delegates, for environment `E`. `none` means "no answer": `exec` gives up
and claims nothing. -/
structure Oracle (E : Env) where
  /-- What the solvers of `pick`s and atomics may ask for. -/
  hints : Hints
  /-- The value of a disabled lane of a load. -/
  fill : (d : DType) → (s : Shape) → Idx s → d.Val

/-- Zero or more steps of one program, none of them undefined behaviour. -/
inductive Runs (E : Env) {α : Type} : Mem → Prog α → Mem → Prog α → Prop where
  | refl (m : Mem) (p : Prog α) : Runs E m p m p
  | step {m m₁ m₂ : Mem} {p p₁ p₂ : Prog α} {e : Event} :
      Step E m p (.next m₁ p₁ e) → Runs E m₁ p₁ m₂ p₂ → Runs E m p m₂ p₂

/-- What `exec` reports. -/
inductive Res (α : Type) where
  /-- The program returned `a`, leaving memory `m`. -/
  | done (a : α) (m : CMem)
  /-- The program reaches undefined behaviour. -/
  | ub

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
def loadVals {E : Env} (O : Oracle E) (m : Mem) (d : DType) (addrs : Idx s → Option Int) :
    Tensor s d.Val :=
  ⟨fun i => match addrs i with
    | some a => (m.read d a).getD (O.fill d s i)
    | none => O.fill d s i⟩

theorem loadOk_of_noBad {E : Env} (O : Oracle E) (m : Mem) (d : DType)
    (addrs : Idx s → Option Int) (h : ¬ ∃ i, BadLane m d addrs i) :
    LoadOk m d addrs (loadVals O m d addrs) := by
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

/-- An answer `exec` took without a proof (`Ans.assume`): the value `b`, assumed to satisfy `P`. -/
structure Assumed where
  {β : Type}
  /-- The site's predicate. -/
  P : β → Prop
  /-- The value taken. -/
  b : β

/-- The assumption holds. -/
def Assumed.Holds (a : Assumed) : Prop := a.P a.b

/-- All the assumptions hold. -/
def Assumed.All (l : List Assumed) : Prop := ∀ a ∈ l, a.Holds

theorem Assumed.All.tail {a : Assumed} {l : List Assumed} (h : All (a :: l)) : All l :=
  fun x hx => h x (List.mem_cons_of_mem _ hx)

theorem Assumed.All.head {a : Assumed} {l : List Assumed} (h : All (a :: l)) : a.Holds :=
  h a List.mem_cons_self

/-- Add an assumption to a run's log. -/
def Assumed.cons {γ : Type} (a : Assumed) : Option (γ × List Assumed) → Option (γ × List Assumed)
  | some (r, l) => some (r, a :: l)
  | none => none

theorem Assumed.cons_eq {γ : Type} {a : Assumed} {o : Option (γ × List Assumed)} {r : γ}
    {l : List Assumed} (h : Assumed.cons a o = some (r, l)) :
    ∃ l', o = some (r, l') ∧ l = a :: l' := by
  cases o with
  | none => cases h
  | some p => obtain ⟨r', l'⟩ := p; simp only [Assumed.cons, Option.some.injEq, Prod.mk.injEq] at h
              obtain ⟨rfl, rfl⟩ := h; exact ⟨l', rfl, rfl⟩

/-- Run `p` from memory `m` for at most `fuel` steps. `none`: out of fuel, or no answer at a
choice. With the result, the answers taken without proof (`Ans.assume`), in order. -/
def exec (E : Env) (O : Oracle E) : Nat → {α : Type} → CMem → Prog α →
    Option (Res α × List Assumed)
  | _, _, m, .ret a => some (.done a m, [])
  | 0, _, _, _ => none
  | n + 1, _, m, .load d _ addrs k =>
    if ∃ i, BadLane m.toMem d addrs i then some (.ub, [])
    else exec E O n m (k (loadVals O m.toMem d addrs))
  | n + 1, _, m, .store d _ addrs vals k =>
    if ∃ i, UnwritableLane m.toMem d addrs i then some (.ub, [])
    else exec E O n (m.writeAll d addrs vals (enabled addrs)) k
  | n + 1, _, m, .rmw d a upd sv _ _ k =>
    match m.toMem.read d a with
    | none => some (.ub, [])
    | some old => match (sv old).run O.hints with
      | some (.ok new _) => exec E O n (m.write d a new) (k old)
      | some (.empty _) => some (.ub, [])
      | some (.assume new) => Assumed.cons ⟨upd old, new⟩ (exec E O n (m.write d a new) (k old))
      | none => none
  | n + 1, _, m, .pick P sv k =>
    match sv.run O.hints with
    | some (.ok b _) => exec E O n m (k b)
    | some (.empty _) => some (.ub, [])
    | some (.assume b) => Assumed.cons ⟨P, b⟩ (exec E O n m (k b))
    | none => none
  | _ + 1, _, _, .ub => some (.ub, [])
  | n + 1, _, m, .loop init body k => exec E O n m (Prog.loopStep init body k)
  | n + 1, _, m, .env k => exec E O n m (k E)

/-- What `exec` reports, `Step` can do. -/
def Sound (E : Env) {α : Type} (m : Mem) (p : Prog α) : Res α → Prop
  | .done a m' => Runs E m p m'.toMem (.ret a)
  | .ub => ∃ m' p', Runs E m p m' p' ∧ Step E m' p' .ub

theorem Sound.cons {E : Env} {α : Type} {m m₁ : Mem} {p p₁ : Prog α} {e : Event} {r : Res α}
    (hs : Step E m p (.next m₁ p₁ e)) (h : Sound E m₁ p₁ r) : Sound E m p r := by
  cases r with
  | done a m' => exact Runs.step hs h
  | ub => obtain ⟨m', p', hr, hu⟩ := h; exact ⟨m', p', Runs.step hs hr, hu⟩

theorem Sound.now {E : Env} {α : Type} {m : Mem} {p : Prog α} (h : Step E m p .ub) :
    Sound E m p .ub := ⟨m, p, Runs.refl m p, h⟩

/-- **Soundness of `exec`.** If `exec` reports a result, and every answer it took without proof
satisfies its site's predicate, the program can reach the result under `Step`. -/
theorem exec_sound (E : Env) (O : Oracle E) :
    ∀ (fuel : Nat) {α : Type} (m : CMem) (p : Prog α) (r : Res α) (l : List Assumed),
      exec E O fuel m p = some (r, l) → Assumed.All l → Sound E m.toMem p r := by
  intro fuel
  induction fuel with
  | zero =>
    intro α m p r l h _
    cases p <;> simp only [exec, reduceCtorEq] at h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, -⟩ := h; exact Runs.refl _ _
  | succ n ih =>
    intro α m p r l h hl
    cases p with
    | ret a =>
      simp only [exec, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h; exact Runs.refl _ _
    | load d s addrs k =>
      simp only [exec] at h
      split at h
      · rename_i hb
        simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
        exact Sound.now (Step.loadUB (loadBad_of_bad _ d addrs hb))
      · rename_i hb
        exact Sound.cons (Step.load _ (loadOk_of_noBad O _ d addrs hb)) (ih _ _ _ _ h hl)
    | store d s addrs vals k =>
      simp only [exec] at h
      split at h
      · rename_i hb
        simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
        exact Sound.now (Step.storeUB (not_storeOk_of_unwritable _ d addrs hb))
      · rename_i hb
        have hs := Step.store (k := k) (vals := vals) _ (storeOk_of_noUnwritable _ d addrs hb)
          (List.Perm.refl (enabled addrs)) (E := E)
        rw [← CMem.toMem_writeAll] at hs
        exact Sound.cons hs (ih _ _ _ _ h hl)
    | rmw d a upd sv sem sc k =>
      simp only [exec] at h
      split at h
      · rename_i hr
        simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
        exact Sound.now (Step.rmwUB (Or.inl hr))
      · rename_i old hr
        split at h
        · rename_i new hu _
          have hs := Step.rmw (E := E) (sv := sv) (sem := sem) (sc := sc) (k := k) old new hr hu
          rw [← CMem.toMem_write] at hs
          exact Sound.cons hs (ih _ _ _ _ h hl)
        · rename_i hu _
          simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
          exact Sound.now (Step.rmwUB (Or.inr ⟨old, hr, hu⟩))
        · rename_i new _
          obtain ⟨l', h', rfl⟩ := Assumed.cons_eq h
          have hs := Step.rmw (E := E) (sv := sv) (sem := sem) (sc := sc) (k := k) old new hr
            hl.head
          rw [← CMem.toMem_write] at hs
          exact Sound.cons hs (ih _ _ _ _ h' hl.tail)
        · cases h
    | pick P sv k =>
      simp only [exec] at h
      split at h
      · rename_i b hb _; exact Sound.cons (Step.pick b hb) (ih _ _ _ _ h hl)
      · rename_i hb _
        simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, -⟩ := h
        exact Sound.now (Step.pickUB hb)
      · rename_i b _
        obtain ⟨l', h', rfl⟩ := Assumed.cons_eq h
        exact Sound.cons (Step.pick b hl.head) (ih _ _ _ _ h' hl.tail)
      · cases h
    | ub =>
      simp only [exec, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h; exact Sound.now Step.ub
    | loop init body k => simp only [exec] at h; exact Sound.cons Step.loop (ih _ _ _ _ h hl)
    | env k => simp only [exec] at h; exact Sound.cons Step.env (ih _ _ _ _ h hl)

/-! ## A launch, one program after another -/

/-- Traces compose. -/
theorem Reach.trans {N : Nat} {E : Env} {c₁ c₂ c₃ : Config N} {t₁ t₂ : List (Fin N × Event)}
    (h₁ : Reach E c₁ t₁ c₂) (h₂ : Reach E c₂ t₂ c₃) : Reach E c₁ (t₁ ++ t₂) c₃ := by
  induction h₂ with
  | refl => simpa using h₁
  | tail _ hs ih => rw [← List.append_assoc]; exact .tail ih hs

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

/-- What `execLaunch` reports. -/
inductive LRes (N : Nat) where
  /-- Every program finished; the configuration reached. -/
  | done (c : CConfig N)
  /-- Some program reaches undefined behaviour. -/
  | ub

/-- Run the programs `is`, one after another, each to its end, from configuration `c`. With the
result, the answers taken without proof. -/
def execProgs (E : Env) (O : Oracle E) (fuel : Nat) {N : Nat} :
    List (Fin N) → CConfig N → Option (LRes N × List Assumed)
  | [], c => some (.done c, [])
  | i :: is, c => match exec E O fuel c.mem (c.progs i) with
    | some (.done () m', l) => (execProgs E O fuel is ⟨m', Function.update c.progs i (.ret ())⟩).map
        fun (r, l') => (r, l ++ l')
    | some (.ub, l) => some (.ub, l)
    | none => none

/-- What `execProgs` reports, the launch can do from `c`. -/
def LSound (E : Env) {N : Nat} (is : List (Fin N)) (c : CConfig N) : LRes N → Prop
  | .done c' => (∃ tr, Reach E c.toConfig tr c'.toConfig) ∧ (∀ i ∈ is, c'.progs i = .ret ()) ∧
      ∀ i, i ∉ is → c'.progs i = c.progs i
  | .ub => ∃ tr c', Reach E c.toConfig tr c' ∧ c'.StuckUB E

theorem Assumed.All.append {l l' : List Assumed} (h : All (l ++ l')) : All l ∧ All l' :=
  ⟨fun x hx => h x (List.mem_append_left _ hx), fun x hx => h x (List.mem_append_right _ hx)⟩

theorem execProgs_sound (E : Env) (O : Oracle E) (fuel : Nat) {N : Nat} :
    ∀ (is : List (Fin N)) (c : CConfig N) (r : LRes N) (l : List Assumed),
      execProgs E O fuel is c = some (r, l) → Assumed.All l → LSound E is c r := by
  intro is
  induction is with
  | nil =>
    intro c r l h _
    simp only [execProgs, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, -⟩ := h
    exact ⟨⟨[], .refl _⟩, fun _ h => absurd h List.not_mem_nil, fun _ _ => rfl⟩
  | cons i is ih =>
    intro c r l h hl
    simp only [execProgs] at h
    split at h
    · rename_i m' l₁ hex
      obtain ⟨⟨r', l₂⟩, hrest, hrl⟩ := Option.map_eq_some_iff.mp h
      simp only [Prod.mk.injEq] at hrl
      obtain ⟨rfl, rfl⟩ := hrl
      obtain ⟨hl₁, hl₂⟩ := hl.append
      have hs := exec_sound E O fuel c.mem (c.progs i) _ _ hex hl₁
      obtain ⟨tr₁, htr₁⟩ := Runs.reach hs c.toConfig i rfl rfl
      have hr := ih _ _ _ hrest hl₂
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
      | ub =>
        obtain ⟨tr₂, c', htr₂, hu⟩ := hr
        exact ⟨tr₁ ++ tr₂, c', Reach.trans htr₁ htr₂, hu⟩
    · rename_i l₁ hex
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨m', p', hruns, hu⟩ := exec_sound E O fuel c.mem (c.progs i) _ _ hex hl
      obtain ⟨tr, htr⟩ := Runs.reach hruns c.toConfig i rfl rfl
      exact ⟨tr, _, htr, i, by simpa using hu⟩
    · cases h

/-- Run the launch `L` of kernel `k` from `init`: programs `0, 1, …` in grid order, each for at
most `fuel` steps. With the result, the answers taken without proof. -/
def execLaunch (E : Env) (O : Oracle E) (fuel : Nat) (k : Kernel) (L : Launch) (init : CMem) :
    Option (Res Unit × List Assumed) :=
  match execProgs E O fuel (List.finRange L.size) ⟨init, fun i => k.body (L.ctx i)⟩ with
  | some (.done c, l) => some (.done () c.mem, l)
  | some (.ub, l) => some (.ub, l)
  | none => none

/-- **Soundness of `execLaunch`.** If every answer taken without proof satisfies its site's
predicate: a finished run is a trace of the launch to a configuration where every program has
finished, with the reported memory; a reported undefined behaviour is a trace to a configuration
where some program's next step can be undefined behaviour. -/
theorem execLaunch_sound (E : Env) (O : Oracle E) (fuel : Nat) (k : Kernel) (L : Launch)
    (init : CMem) (l : List Assumed) (hl : Assumed.All l) :
    (∀ m, execLaunch E O fuel k L init = some (.done () m, l) →
      ∃ tr c, Reach E (Config.start k L (CMem.toMem init)) tr c ∧ c.Done ∧
        c.mem = CMem.toMem m) ∧
    (execLaunch E O fuel k L init = some (.ub, l) →
      ∃ tr c, Reach E (Config.start k L (CMem.toMem init)) tr c ∧ c.StuckUB E) := by
  unfold execLaunch
  have hs := execProgs_sound E O fuel (List.finRange L.size) ⟨init, fun i => k.body (L.ctx i)⟩
  split
  · rename_i c l' hc
    refine ⟨fun m hm => ?_, fun h => by simp at h⟩
    simp only [Option.some.injEq, Prod.mk.injEq, Res.done.injEq, true_and] at hm
    obtain ⟨hm, rfl⟩ := hm
    obtain ⟨⟨tr, htr⟩, hin, -⟩ := hs _ _ hc hl
    exact ⟨tr, c.toConfig, htr, fun i => hin i (List.mem_finRange i), by rw [← hm]; rfl⟩
  · rename_i l' hc
    refine ⟨fun _ h => by simp at h, fun h => ?_⟩
    simp only [Option.some.injEq, Prod.mk.injEq, true_and] at h
    subst h
    exact hs _ _ hc hl
  · exact ⟨fun _ h => (nomatch h), fun h => (nomatch h)⟩

end TVA.Validation
