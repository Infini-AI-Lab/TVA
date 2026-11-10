import TVA.Sem.Step

/-!
# An executor for one program, and its soundness

Phase 1 of `validation/PLAN.md`. `exec` runs one `Prog` against a concrete memory in a fixed
environment `E`, and `exec_sound` says that whatever `exec` reports, `Step` can do too: a
finished run is a sequence of `Step`s to `ret`, and a predicted undefined behaviour is a
sequence of `Step`s to a step whose outcome is `ub`. Nothing here is trusted: `Step` stays the
meaning.

## Where the answers come from

* `pick P sv k`: the site's own solver `sv` (`TVA.Solve`), given the oracle's `Hints` for the
  parts it cannot compute (a tree, a permutation, …).
* `plat q k`: the oracle's `plat`, for the platform of `E`. For a platform calibrated on the
  device (`P_dev`, a table), this is a lookup and a decision.
* `load`: the values of disabled lanes (`Oracle.fill`).
* `rmw d a upd …`: `Oracle.rmw`. `upd` is an erased relation, so no computable oracle can
  answer yet (`PICKS.md`).

Every answer carries its proof, so `exec_sound` holds for every oracle and every solver.

`exec` makes the other choices itself: a `store` writes its enabled lanes in index order
(`enabled addrs`, one permutation `Step.store` admits), atomics are performed in program order
(never `Step.defer`), and `env` reads `E`.
-/
namespace TVA.Validation

open TVA

/-- The choices `exec` delegates, for environment `E`. `none` means "no answer": `exec` gives up
(and claims nothing). -/
structure Oracle (E : Env) where
  /-- What the solvers of `pick`s may ask for. -/
  hints : Hints
  /-- An operation of `E`'s platform. -/
  plat : (q : PlatQ) → Option (Ans (q.Rel E.P))
  /-- The new value of an atomic, given its relation and the old value. -/
  rmw : (d : DType) → (upd : d.Val → d.Val → Prop) → (old : d.Val) → Option (Ans (upd old))
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
  | done (a : α) (m : Mem)
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

/-- Lane `i` is enabled and its address cannot be written. -/
def UnwritableLane (m : Mem) (d : DType) (addrs : Idx s → Option Int) (i : Idx s) : Prop :=
  match addrs i with
  | some a => ¬ m.Writable d a
  | none => False

instance (m : Mem) (d : DType) (a : Int) : Decidable (m.Writable d a) := by
  unfold Mem.Writable; infer_instance

instance (m : Mem) (d : DType) (addrs : Idx s → Option Int) (i : Idx s) :
    Decidable (UnwritableLane m d addrs i) := by
  unfold UnwritableLane; split <;> infer_instance

/-- The loaded tensor: memory at the enabled lanes, `fill` elsewhere. -/
def loadVals {E : Env} (O : Oracle E) (m : Mem) (d : DType) (addrs : Idx s → Option Int) : Tensor s d.Val :=
  ⟨fun i => match addrs i with
    | some a => (m.read d a).getD (O.fill d s i)
    | none => O.fill d s i⟩

theorem loadOk_of_noBad {E : Env} (O : Oracle E) (m : Mem) (d : DType) (addrs : Idx s → Option Int)
    (h : ¬ ∃ i, BadLane m d addrs i) : LoadOk m d addrs (loadVals O m d addrs) := by
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

/-- Run `p` from memory `m` for at most `fuel` steps. `none`: out of fuel, or the oracle gave
no answer. -/
def exec (E : Env) (O : Oracle E) : Nat → {α : Type} → Mem → Prog α → Option (Res α)
  | _, _, m, .ret a => some (.done a m)
  | 0, _, _, _ => none
  | n + 1, _, m, .load d _ addrs k =>
    if ∃ i, BadLane m d addrs i then some .ub
    else exec E O n m (k (loadVals O m d addrs))
  | n + 1, _, m, .store d _ addrs vals k =>
    if ∃ i, UnwritableLane m d addrs i then some .ub
    else exec E O n (writeAll m d addrs vals (enabled addrs)) k
  | n + 1, _, m, .rmw d a upd _ _ k =>
    match m.read d a with
    | none => some .ub
    | some old => match O.rmw d upd old with
      | some (.ok new _) => exec E O n (m.write d a new) (k old)
      | some (.empty _) => some .ub
      | none => none
  | n + 1, _, m, .pick _ sv k =>
    match sv.run O.hints with
    | some (.ok b _) => exec E O n m (k b)
    | some (.empty _) => some .ub
    | none => none
  | n + 1, _, m, .plat q k =>
    match O.plat q with
    | some (.ok r _) => exec E O n m (k r)
    | some (.empty _) => some .ub
    | none => none
  | _ + 1, _, _, .ub => some .ub
  | n + 1, _, m, .loop init body k => exec E O n m (Prog.loopStep init body k)
  | n + 1, _, m, .env k => exec E O n m (k E)

/-- What `exec` reports, `Step` can do. -/
def Sound (E : Env) {α : Type} (m : Mem) (p : Prog α) : Res α → Prop
  | .done a m' => Runs E m p m' (.ret a)
  | .ub => ∃ m' p', Runs E m p m' p' ∧ Step E m' p' .ub

theorem Sound.cons {E : Env} {α : Type} {m m₁ : Mem} {p p₁ : Prog α} {e : Event} {r : Res α}
    (hs : Step E m p (.next m₁ p₁ e)) (h : Sound E m₁ p₁ r) : Sound E m p r := by
  cases r with
  | done a m' => exact Runs.step hs h
  | ub => obtain ⟨m', p', hr, hu⟩ := h; exact ⟨m', p', Runs.step hs hr, hu⟩

theorem Sound.now {E : Env} {α : Type} {m : Mem} {p : Prog α} (h : Step E m p .ub) :
    Sound E m p .ub := ⟨m, p, Runs.refl m p, h⟩

/-- **Soundness.** If `exec` reports a result, the program can reach it under `Step`. -/
theorem exec_sound (E : Env) (O : Oracle E) :
    ∀ (fuel : Nat) {α : Type} (m : Mem) (p : Prog α) (r : Res α),
      exec E O fuel m p = some r → Sound E m p r := by
  intro fuel
  induction fuel with
  | zero =>
    intro α m p r h
    cases p <;> simp only [exec, reduceCtorEq] at h
    cases h; exact Runs.refl _ _
  | succ n ih =>
    intro α m p r h
    cases p with
    | ret a => simp only [exec, Option.some.injEq] at h; cases h; exact Runs.refl _ _
    | load d s addrs k =>
      simp only [exec] at h
      split at h
      · rename_i hb; cases h; exact Sound.now (Step.loadUB (loadBad_of_bad m d addrs hb))
      · rename_i hb
        exact Sound.cons (Step.load _ (loadOk_of_noBad O m d addrs hb)) (ih _ _ _ h)
    | store d s addrs vals k =>
      simp only [exec] at h
      split at h
      · rename_i hb; cases h; exact Sound.now (Step.storeUB (not_storeOk_of_unwritable m d addrs hb))
      · rename_i hb
        exact Sound.cons (Step.store _ (storeOk_of_noUnwritable m d addrs hb) (List.Perm.refl _))
          (ih _ _ _ h)
    | rmw d a upd sem sc k =>
      simp only [exec] at h
      split at h
      · rename_i hr; cases h; exact Sound.now (Step.rmwUB (Or.inl hr))
      · rename_i old hr
        split at h
        · rename_i new hu _; exact Sound.cons (Step.rmw old new hr hu) (ih _ _ _ h)
        · rename_i hu _; cases h; exact Sound.now (Step.rmwUB (Or.inr ⟨old, hr, hu⟩))
        · cases h
    | pick P sv k =>
      simp only [exec] at h
      split at h
      · rename_i b hb _; exact Sound.cons (Step.pick b hb) (ih _ _ _ h)
      · rename_i hb _; cases h; exact Sound.now (Step.pickUB hb)
      · cases h
    | plat q k =>
      simp only [exec] at h
      split at h
      · rename_i r hr _; exact Sound.cons (Step.plat r hr) (ih _ _ _ h)
      · rename_i hr _; cases h; exact Sound.now (Step.platUB hr)
      · cases h
    | ub => simp only [exec, Option.some.injEq] at h; cases h; exact Sound.now Step.ub
    | loop init body k => simp only [exec] at h; exact Sound.cons Step.loop (ih _ _ _ h)
    | env k => simp only [exec] at h; exact Sound.cons Step.env (ih _ _ _ h)

end TVA.Validation
