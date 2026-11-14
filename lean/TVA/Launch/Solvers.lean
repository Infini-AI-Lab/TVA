import TVA.Launch.WellBehaved

/-!
# Solvers play no part in the meaning

Each `pick` and `rmw` of a model carries a solver (`Solve`, `TVA.Program`), code that an executor
runs to propose a value. The semantics is meant to ignore them: a `pick`'s meaning is its
predicate alone. This file proves it. Two kernels whose programs are the same up to solvers
(`Kernel.SameUpToSolvers`) have the same outcomes, are well behaved together, and are `Correct`
together (`Correct.congr_solvers`), so a reader of a claim may skip every solver of its model.

The only places the semantics looks at a program's text beyond its next effect are
`Step.defer` and `UsesSharedAtomicResult`, which ask whether a continuation depends on an
atomic's old value. Both ask it up to solvers, so a solver built from the old value is not a
dependence, and the argument goes through. With `=` there, two such kernels could differ.

The proof is a simulation: a step of a program is matched by a step of any program the same up to
solvers, with the same memory and event, to programs again the same up to solvers
(`Step.sameUpToSolvers`); the same holds for the steps (`LStep`) and traces (`Reach`) of a launch.
-/
namespace TVA

namespace Prog

/-- Being the same up to solvers is symmetric. -/
theorem SameUpToSolvers.symm {α : Type} {p q : Prog α} (h : SameUpToSolvers p q) :
    SameUpToSolvers q p := by
  induction h with
  | ret a => exact .ret a
  | load _ ih => exact .load ih
  | store _ ih => exact .store ih
  | rmw _ ih => exact .rmw ih
  | pick _ ih => exact .pick ih
  | ub => exact .ub
  | loop _ _ ihb ih => exact .loop ihb ih
  | env _ ih => exact .env ih

/-- Being the same up to solvers is transitive. -/
theorem SameUpToSolvers.trans {α : Type} {p q r : Prog α} (h : SameUpToSolvers p q) :
    SameUpToSolvers q r → SameUpToSolvers p r := by
  induction h with
  | ret a => intro h; exact h
  | load _ ih => intro h; cases h with | load h => exact .load fun v => ih v (h v)
  | store _ ih => intro h; cases h with | store h => exact .store (ih h)
  | rmw _ ih => intro h; cases h with | rmw h => exact .rmw fun v => ih v (h v)
  | pick _ ih => intro h; cases h with | pick h => exact .pick fun b => ih b (h b)
  | ub => intro h; exact h
  | loop _ _ ihb ih => intro h; cases h with
    | loop hb hk => exact .loop (fun x => ihb x (hb x)) fun b => ih b (hk b)
  | env _ ih => intro h; cases h with | env h => exact .env fun e => ih e (h e)

/-- `bind` respects being the same up to solvers, on both sides. -/
theorem SameUpToSolvers.bind {α β : Type} {p q : Prog α} (h : SameUpToSolvers p q)
    {f g : α → Prog β} (hf : ∀ a, SameUpToSolvers (f a) (g a)) :
    SameUpToSolvers (p.bind f) (q.bind g) := by
  induction h with
  | ret a => exact hf a
  | load _ ih => exact .load fun v => ih v hf
  | store _ ih => exact .store (ih hf)
  | rmw _ ih => exact .rmw fun v => ih v hf
  | pick _ ih => exact .pick fun b => ih b hf
  | ub => exact .ub
  | loop hb _ _ ih => exact .loop hb fun b => ih b hf
  | env _ ih => exact .env fun e => ih e hf

/-- One unfolding of loops the same up to solvers. -/
theorem SameUpToSolvers.loopStep {α σ β : Type} {init : σ} {body body' : σ → Prog (σ ⊕ β)}
    {k k' : β → Prog α} (hb : ∀ x, SameUpToSolvers (body x) (body' x))
    (hk : ∀ b, SameUpToSolvers (k b) (k' b)) :
    SameUpToSolvers (loopStep init body k) (loopStep init body' k') :=
  (hb init).bind fun r => match r with
    | .inl _ => .loop hb hk
    | .inr b => hk b

/-- A finished program is the same up to solvers only as itself. -/
theorem SameUpToSolvers.ret_left {α : Type} {a : α} {q : Prog α}
    (h : SameUpToSolvers (Prog.ret a) q) : q = Prog.ret a := by
  cases h; rfl

/-- What is the same as an `rmw` up to solvers: the same `rmw`, with another solver, and a
continuation the same up to solvers for every old value. -/
theorem SameUpToSolvers.rmw_left {α : Type} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop}
    {sv : (old : d.Val) → Solve (upd old)} {sem : MemSem} {sc : Scope} {k : d.Val → Prog α}
    {q : Prog α} (h : SameUpToSolvers (Prog.rmw d a upd sv sem sc k) q) :
    ∃ sv' k', q = Prog.rmw d a upd sv' sem sc k' ∧ ∀ v, SameUpToSolvers (k v) (k' v) := by
  cases h with | rmw h => exact ⟨_, _, rfl, h⟩

end Prog

/-- Two outcomes of a step the same up to solvers: both undefined behaviour, or the same memory
and event with programs the same up to solvers. -/
inductive Out.SameUpToSolvers {α : Type} : Out α → Out α → Prop where
  | next {m : Mem} {p p' : Prog α} {e : Event} :
      p.SameUpToSolvers p' → Out.SameUpToSolvers (.next m p e) (.next m p' e)
  | ub : Out.SameUpToSolvers .ub .ub

/-- **A step is matched by any program the same up to solvers**: same memory, same event, and
programs again the same up to solvers. -/
theorem Step.sameUpToSolvers {E : Env} {m : Mem} {α : Type} {p : Prog α} {o : Out α}
    (h : Step E m p o) : ∀ {p' : Prog α}, p.SameUpToSolvers p' →
      ∃ o', Step E m p' o' ∧ o.SameUpToSolvers o' := by
  induction h with
  | load v hv =>
    intro p' hs; cases hs with | load hk => exact ⟨_, .load v hv, .next (hk v)⟩
  | loadUB hb => intro p' hs; cases hs; exact ⟨_, .loadUB hb, .ub⟩
  | store order hok hperm =>
    intro p' hs; cases hs with | store hk => exact ⟨_, .store order hok hperm, .next hk⟩
  | storeUB hb => intro p' hs; cases hs; exact ⟨_, .storeUB hb, .ub⟩
  | rmw old new hr hu =>
    intro p' hs; cases hs with | rmw hk => exact ⟨_, .rmw old new hr hu, .next (hk old)⟩
  | rmwUB hb => intro p' hs; cases hs; exact ⟨_, .rmwUB hb, .ub⟩
  | @defer _ _ _ _ _ _ k kk m' e _ hdep ih =>
    intro p' hs
    obtain ⟨sv', k', rfl, hk⟩ := Prog.SameUpToSolvers.rmw_left hs
    -- each old value's step, matched in `k'`
    have hq : ∀ v, ∃ q, Step E m (k' v) (.next m' q e) ∧ (kk v).SameUpToSolvers q := by
      intro v
      obtain ⟨o', hst, ho⟩ := ih v (hk v)
      cases ho with | next hp => exact ⟨_, hst, hp⟩
    choose q hqs hqe using hq
    refine ⟨_, .defer hqs fun hind v w => ?_, .next (.rmw hqe)⟩
    have hk' : ∀ v w, (k v).SameUpToSolvers (k w) := fun v w =>
      (hk v).trans ((hind v w).trans (hk w).symm)
    exact (hqe v).symm.trans ((hdep hk' v w).trans (hqe w))
  | pick b hb => intro p' hs; cases hs with | pick hk => exact ⟨_, .pick b hb, .next (hk b)⟩
  | pickUB hn => intro p' hs; cases hs; exact ⟨_, .pickUB hn, .ub⟩
  | ub => intro p' hs; cases hs; exact ⟨_, .ub, .ub⟩
  | loop => intro p' hs; cases hs with | loop hb hk => exact ⟨_, .loop, .next (.loopStep hb hk)⟩
  | env => intro p' hs; cases hs with | env hk => exact ⟨_, .env, .next (hk _)⟩

/-- Two kernels the same up to solvers: every program is. -/
def Kernel.SameUpToSolvers (k k' : Kernel) : Prop :=
  ∀ ctx, (k.body ctx).SameUpToSolvers (k'.body ctx)

theorem Kernel.SameUpToSolvers.symm {k k' : Kernel} (h : k.SameUpToSolvers k') :
    k'.SameUpToSolvers k := fun ctx => (h ctx).symm

/-- Two states of a launch the same up to solvers: the same memory, and each program's rest the
same up to solvers. -/
def Config.SameUpToSolvers {N : Nat} (c c' : Config N) : Prop :=
  c.mem = c'.mem ∧ ∀ i, (c.progs i).SameUpToSolvers (c'.progs i)

theorem Config.SameUpToSolvers.start {k k' : Kernel} (h : k.SameUpToSolvers k') (L : Launch)
    (init : Mem) : (Config.start k L init).SameUpToSolvers (Config.start k' L init) :=
  ⟨rfl, fun i => h (L.ctx i)⟩

/-- A step of a launch is matched from a state the same up to solvers, with the same label. -/
theorem LStep.sameUpToSolvers {N : Nat} {E : Env} {c d c' : Config N} {ev : Fin N × Event}
    (h : LStep E c ev d) (hc : c.SameUpToSolvers c') :
    ∃ d', LStep E c' ev d' ∧ d.SameUpToSolvers d' := by
  cases h with
  | step i hst =>
    obtain ⟨o', hst', ho⟩ := hst.sameUpToSolvers (hc.2 i)
    cases ho with
    | next hp =>
      rw [hc.1] at hst'
      refine ⟨_, .step i hst', rfl, fun j => ?_⟩
      by_cases hj : j = i
      · subst hj; simpa using hp
      · simpa [Function.update_of_ne hj] using hc.2 j

/-- A trace of a launch is matched, step for step, from a state the same up to solvers. -/
theorem Reach.sameUpToSolvers {N : Nat} {E : Env} {c d c' : Config N}
    {tr : List (Fin N × Event)} (h : Reach E c tr d) (hc : c.SameUpToSolvers c') :
    ∃ d', Reach E c' tr d' ∧ d.SameUpToSolvers d' := by
  induction h with
  | refl => exact ⟨c', .refl _, hc⟩
  | tail _ hl ih =>
    obtain ⟨d', hr, hd⟩ := ih
    obtain ⟨e', hl', he⟩ := hl.sameUpToSolvers hd
    exact ⟨e', .tail hr hl', he⟩

theorem Config.SameUpToSolvers.stuckUB {N : Nat} {E : Env} {c c' : Config N}
    (hc : c.SameUpToSolvers c') (h : c.StuckUB E) : c'.StuckUB E := by
  obtain ⟨i, hi⟩ := h
  obtain ⟨o', hst, ho⟩ := hi.sameUpToSolvers (hc.2 i)
  cases ho
  exact ⟨i, hc.1 ▸ hst⟩

theorem Config.SameUpToSolvers.done {N : Nat} {c c' : Config N} (hc : c.SameUpToSolvers c')
    (h : c.Done) : c'.Done := fun i => by
  have := hc.2 i
  rw [h i] at this
  exact this.ret_left

section Kernels

variable {E : Env} {k k' : Kernel} {L : Launch} {init : Mem}

/-- Every outcome of a kernel is one of any kernel the same up to solvers. -/
theorem Outcomes.of_sameUpToSolvers (h : k.SameUpToSolvers k') {o : Outcome}
    (ho : Outcomes E k L init o) : Outcomes E k' L init o := by
  have hs := Config.SameUpToSolvers.start h L init
  cases o with
  | ok fin =>
    obtain ⟨tr, c, hr, hd, hm⟩ := ho
    obtain ⟨c', hr', hc⟩ := hr.sameUpToSolvers hs
    exact ⟨tr, c', hr', hc.done hd, hc.1 ▸ hm⟩
  | ub =>
    obtain ⟨tr, c, hr, hu⟩ := ho
    obtain ⟨c', hr', hc⟩ := hr.sameUpToSolvers hs
    exact ⟨tr, c', hr', hc.stuckUB hu⟩
  | hang =>
    obtain ⟨f, h0, hf⟩ := ho
    -- follow `f` step by step from the other kernel's start
    have hnext : ∀ n (x : {c' // (f n).SameUpToSolvers c'}),
        ∃ d', (∃ ev, LStep E x.1 ev d') ∧ (f (n + 1)).SameUpToSolvers d' := by
      intro n x
      obtain ⟨ev, hl⟩ := hf n
      obtain ⟨d', hl', hd⟩ := hl.sameUpToSolvers x.2
      exact ⟨d', ⟨ev, hl'⟩, hd⟩
    let next : ∀ n, {c' // (f n).SameUpToSolvers c'} → {c' // (f (n + 1)).SameUpToSolvers c'} :=
      fun n x => ⟨Classical.choose (hnext n x), (Classical.choose_spec (hnext n x)).2⟩
    let g : ∀ n, {c' // (f n).SameUpToSolvers c'} :=
      fun n => Nat.rec ⟨Config.start k' L init, h0 ▸ hs⟩ (fun n x => next n x) n
    refine ⟨fun n => (g n).1, rfl, fun n => ?_⟩
    exact (Classical.choose_spec (hnext n (g n))).1

/-- A kernel that uses an atomic's result on a shared cell is matched by any kernel the same up to
solvers. -/
theorem UsesSharedAtomicResult.of_sameUpToSolvers (h : k.SameUpToSolvers k')
    (hu : UsesSharedAtomicResult E k L init) : UsesSharedAtomicResult E k' L init := by
  have hs := Config.SameUpToSolvers.start h L init
  obtain ⟨tr, c, i, d, a, upd, sv, sem, sc, kk, hr, hi, ⟨v, w, hvw⟩, tr', c₂, hr₂, hsh⟩ := hu
  obtain ⟨c', hr', hc⟩ := hr.sameUpToSolvers hs
  have hci := hc.2 i
  rw [hi] at hci
  obtain ⟨sv', kk', hi', hk⟩ := hci.rmw_left
  obtain ⟨c₂', hr₂', -⟩ := hr₂.sameUpToSolvers hs
  refine ⟨tr, c', i, d, a, upd, sv', sem, sc, kk', hr', hi', ⟨v, w, fun hs' => hvw ?_⟩,
    tr', c₂', hr₂', hsh⟩
  exact (hk v).trans (hs'.trans (hk w).symm)

/-- A kernel the same up to solvers as a well-behaved one is well behaved. -/
theorem WellBehaved.of_sameUpToSolvers (h : k.SameUpToSolvers k')
    (hw : WellBehaved E k L init) : WellBehaved E k' L init where
  noUB tr c hr hu := by
    obtain ⟨c', hr', hc⟩ := hr.sameUpToSolvers (Config.SameUpToSolvers.start h.symm L init)
    exact hw.noUB tr c' hr' (hc.stuckUB hu)
  noInterRace tr c hr := by
    obtain ⟨c', hr', -⟩ := hr.sameUpToSolvers (Config.SameUpToSolvers.start h.symm L init)
    exact hw.noInterRace tr c' hr'
  noIntraRace tr c hr := by
    obtain ⟨c', hr', -⟩ := hr.sameUpToSolvers (Config.SameUpToSolvers.start h.symm L init)
    exact hw.noIntraRace tr c' hr'
  noSharedAtomicResult hu := hw.noSharedAtomicResult (hu.of_sameUpToSolvers h.symm)

/-- **Solvers play no part in the meaning**: two kernels the same up to solvers have the same
outcomes. -/
theorem Outcomes.congr_solvers (h : k.SameUpToSolvers k') (o : Outcome) :
    Outcomes E k L init o ↔ Outcomes E k' L init o :=
  ⟨Outcomes.of_sameUpToSolvers h, Outcomes.of_sameUpToSolvers h.symm⟩

/-- **Solvers play no part in the meaning**: two kernels the same up to solvers are well behaved
together. -/
theorem WellBehaved.congr_solvers (h : k.SameUpToSolvers k') :
    WellBehaved E k L init ↔ WellBehaved E k' L init :=
  ⟨WellBehaved.of_sameUpToSolvers h, WellBehaved.of_sameUpToSolvers h.symm⟩

/-- **Solvers play no part in the meaning**: two kernels the same up to solvers are `Correct`
together, so a claim holds of a model whatever solvers its `pick`s and `rmw`s carry. -/
theorem Correct.congr_solvers (h : k.SameUpToSolvers k') (Q : Mem → Prop) :
    Correct E k L init Q ↔ Correct E k' L init Q := by
  have key : ∀ {k k' : Kernel}, k.SameUpToSolvers k' →
      Correct E k L init Q → Correct E k' L init Q := fun h ⟨hv, hw, ho⟩ =>
    ⟨hv, hw.of_sameUpToSolvers h, fun o hoo => ho o (hoo.of_sameUpToSolvers h.symm)⟩
  exact ⟨key h, key h.symm⟩

end Kernels

end TVA
