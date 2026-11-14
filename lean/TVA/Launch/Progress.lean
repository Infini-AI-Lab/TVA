import TVA.Launch.WellBehaved

/-!
# Every launch can go on: progress

A program that has not finished can always take a step or reach undefined behaviour
(`Step.progress`): the semantics never gets stuck in any other way. So a well-behaved launch
always has an outcome (`WellBehaved.exists_outcome`), and `Correct` cannot hold because a launch
has no outcome at all.
-/
namespace TVA

/-- A program that has not finished can take a step, or its next step is undefined behaviour.
Use it to show a launch is not stuck. -/
theorem Step.progress (E : Env) (m : Mem) {α : Type} (p : Prog α) (hp : ∀ a, p ≠ .ret a) :
    (∃ m' p' e, Step E m p (.next m' p' e)) ∨ Step E m p .ub := by
  cases p with
  | ret a => exact absurd rfl (hp a)
  | load d s addrs k =>
    by_cases hb : LoadBad m d addrs
    · exact .inr (.loadUB hb)
    · refine .inl ⟨_, _, _, Step.load
        ⟨fun i => match addrs i with | some a => (m.read d a).getD default | none => default⟩ ?_⟩
      intro i a ha
      show m.read d a = some (match addrs i with
        | some a => (m.read d a).getD default | none => default)
      rw [ha]
      cases h : m.read d a with
      | none => exact absurd ⟨i, a, ha, h⟩ hb
      | some v => simp [h]
  | store d s addrs vals k =>
    by_cases hs : StoreOk m d addrs
    · exact .inl ⟨_, _, _, Step.store (enabled addrs) hs (List.Perm.refl _)⟩
    · exact .inr (.storeUB hs)
  | rmw d a upd sv sem sc k =>
    cases h : m.read d a with
    | none => exact .inr (.rmwUB (.inl h))
    | some old =>
      by_cases hn : ∃ new, upd old new
      · obtain ⟨new, hnew⟩ := hn
        exact .inl ⟨_, _, _, Step.rmw old new h hnew⟩
      · exact .inr (.rmwUB (.inr ⟨old, h, fun new hnew => hn ⟨new, hnew⟩⟩))
  | pick P sv k =>
    by_cases hb : ∃ b, P b
    · obtain ⟨b, hb⟩ := hb
      exact .inl ⟨_, _, _, Step.pick b hb⟩
    · exact .inr (.pickUB fun b h => hb ⟨b, h⟩)
  | ub => exact .inr .ub
  | loop init body k => exact .inl ⟨_, _, _, .loop⟩
  | env k => exact .inl ⟨_, _, _, .env⟩

/-- A well-behaved launch has an outcome: every program finishes, or some interleaving runs
forever. Use it to show that a `Correct` launch has a normal end. -/
theorem WellBehaved.exists_outcome {E : Env} {k : Kernel} {L : Launch} {init : Mem}
    (h : WellBehaved E k L init) : ∃ o, Outcomes E k L init o := by
  by_contra hno
  simp only [not_exists] at hno
  have hstep : ∀ c : {c // ∃ tr, Reach E (Config.start k L init) tr c},
      ∃ ev c', LStep E c.1 ev c' := by
    rintro ⟨c, tr, hr⟩
    have hnd : ¬ c.Done := fun hd => hno (.ok c.mem) ⟨tr, c, hr, hd, rfl⟩
    simp only [Config.Done, not_forall] at hnd
    obtain ⟨i, hi⟩ := hnd
    rcases Step.progress E c.mem (c.progs i) (fun a => by cases a; exact hi) with
      ⟨m', p', e, hs⟩ | hub
    · exact ⟨_, _, .step i hs⟩
    · exact absurd ⟨i, hub⟩ (h.noUB tr c hr)
  choose ev nxt hnxt using hstep
  let g : Nat → {c // ∃ tr, Reach E (Config.start k L init) tr c} := fun n =>
    Nat.rec ⟨Config.start k L init, [], .refl _⟩
      (fun _ c => ⟨nxt c, by obtain ⟨tr, hr⟩ := c.2; exact ⟨_, hr.tail (hnxt c)⟩⟩) n
  exact hno .hang ⟨fun n => (g n).1, rfl, fun n => ⟨ev (g n), hnxt (g n)⟩⟩

end TVA
