import TVA.Launch.WellBehaved

/-!
# The semantics sees a site only through its relation

`Prog.erase` replaces each site by its relation (`Sig.Allows`) and each update by its
(`Sig.UpdAllows`), giving a program over `Sig.pred`. Erasure is a bisimulation of `Step`: each
step of a program is a step of its erasure, with the same memory and event, and each step of the
erasure is the erasure of a step of the program (`Step.erase`, `Step.unerase`), `Step.defer`
included. So a kernel and its erasure have the same outcomes, are well behaved together and are
`Correct` together, and so are two kernels with the same erasure (`Outcomes.congr_erase`,
`WellBehaved.congr_erase`, `Correct.congr_erase`): what a site carries beyond its relation plays
no part in the meaning.
-/
namespace TVA

/-- Over `Sig.pred` erasure changes nothing: its sites already are their relations. -/
theorem Prog.erase_pred {α : Type} (p : @Prog Sig.pred α) : @Prog.erase Sig.pred α p = p := by
  induction p with
  | ret a => rfl
  | load d s addrs k ih => simp only [Prog.erase, ih]
  | store d s addrs vals k ih => simp only [Prog.erase, ih]
  | rmw d a u sem sc k ih => simp only [Prog.erase, ih]; rfl
  | pick s k ih => exact congrArg (@Prog.pick Sig.pred _ s) (funext ih)
  | ub => rfl
  | loop init body k ihb ih => simp only [Prog.erase, ihb, ih]
  | env k ih => simp only [Prog.erase, ih]

variable [Sig]

namespace Prog

theorem erase_loopStep {α σ β : Type} (init : σ) (body : σ → Prog (σ ⊕ β)) (k : β → Prog α) :
    (loopStep init body k).erase =
      @loopStep Sig.pred _ _ _ init (fun x => (body x).erase) (fun b => (k b).erase) := by
  unfold loopStep
  rw [erase_bind]
  congr 1
  funext r
  cases r <;> rfl

theorem erase_eq_ret {α : Type} {p : Prog α} {a : α} (h : p.erase = @ret Sig.pred _ a) :
    p = ret a := by
  cases p <;> simp only [erase] at h <;> cases h; rfl

theorem erase_eq_rmw {α : Type} {p : Prog α} {d : DType} {a : Int} {u : PredUpd d} {sem : MemSem}
    {sc : Scope} {k : d.Val → @Prog Sig.pred α}
    (h : p.erase = @rmw Sig.pred _ d a u sem sc k) :
    ∃ u' k', p = rmw d a u' sem sc k' ∧ ∀ v, (k' v).erase = k v := by
  cases p <;> simp only [erase] at h <;> cases h
  exact ⟨_, _, rfl, fun _ => rfl⟩

end Prog

/-- A step's outcome with its program erased. -/
def Out.erase {α : Type} : Out α → @Out Sig.pred α
  | .next m p e => @Out.next Sig.pred _ m p.erase e
  | .ub => @Out.ub Sig.pred _

theorem Out.erase_eq_ub {α : Type} {o : Out α} (h : o.erase = @Out.ub Sig.pred _) : o = .ub := by
  cases o
  · cases h
  · rfl

theorem Out.erase_eq_next {α : Type} {o : Out α} {m : Mem} {q : @Prog Sig.pred α} {e : Event}
    (h : o.erase = @Out.next Sig.pred _ m q e) : ∃ p, o = .next m p e ∧ p.erase = q := by
  cases o
  · cases h; exact ⟨_, rfl, rfl⟩
  · cases h

variable {E : Env} {m : Mem}

/-- **A step is a step of the erasure**: the same memory and event, the rest erased. -/
theorem Step.erase {α : Type} {p : Prog α} {o : Out α} (h : Step E m p o) :
    @Step Sig.pred E m α p.erase o.erase := by
  induction h with
  | load v hv => exact @Step.load Sig.pred E m _ _ _ _ _ v hv
  | loadUB hb => exact @Step.loadUB Sig.pred E m _ _ _ _ _ hb
  | store order hok hperm => exact @Step.store Sig.pred E m _ _ _ _ _ _ order hok hperm
  | storeUB hb => exact @Step.storeUB Sig.pred E m _ _ _ _ _ _ hb
  | rmw old new hr hu => exact @Step.rmw Sig.pred E m _ _ _ _ _ _ _ old new hr hu
  | rmwUB hb => exact @Step.rmwUB Sig.pred E m _ _ _ _ _ _ _ hb
  | defer _ hdep ih =>
    refine @Step.defer Sig.pred E m _ _ _ _ _ _ _ _ _ _ ih fun hc v w => ?_
    simp only [Prog.erase_pred] at hc ⊢
    exact hdep hc v w
  | pick b hb => refine @Step.pick Sig.pred E m _ ⟨_, _⟩ _ b ?_; exact hb
  | pickUB hn => exact @Step.pickUB Sig.pred E m _ _ _ hn
  | ub => exact @Step.ub Sig.pred E m _
  | loop =>
    simp only [Prog.erase, Out.erase, Prog.erase_loopStep]
    exact @Step.loop Sig.pred E m _ _ _ _ _ _
  | env => exact @Step.env Sig.pred E m _ _

/-- **A step of the erasure is the erasure of a step.** -/
theorem Step.unerase {α : Type} {p : Prog α} {o' : @Out Sig.pred α}
    (h : @Step Sig.pred E m α p.erase o') : ∃ o, Step E m p o ∧ o.erase = o' := by
  generalize hq : p.erase = q at h
  induction h generalizing p with
  | load v hv =>
    cases p <;> simp only [Prog.erase] at hq <;> cases hq; exact ⟨_, .load v hv, rfl⟩
  | loadUB hb => cases p <;> simp only [Prog.erase] at hq <;> cases hq; exact ⟨_, .loadUB hb, rfl⟩
  | store order hok hperm =>
    cases p <;> simp only [Prog.erase] at hq <;> cases hq; exact ⟨_, .store order hok hperm, rfl⟩
  | storeUB hb =>
    cases p <;> simp only [Prog.erase] at hq <;> cases hq; exact ⟨_, .storeUB hb, rfl⟩
  | rmw old new hr hu =>
    cases p <;> simp only [Prog.erase] at hq <;> cases hq; exact ⟨_, .rmw old new hr hu, rfl⟩
  | rmwUB hb => cases p <;> simp only [Prog.erase] at hq <;> cases hq; exact ⟨_, .rmwUB hb, rfl⟩
  | @defer _ _ _ _ _ _ kk' m' e _ hdep ih =>
    cases p with
    | rmw _ _ _ _ _ k => ?_
    | _ => cases hq
    simp only [Prog.erase] at hq
    cases hq
    -- each old value's step, taken by `k`
    have hs : ∀ v, ∃ q, Step E m (k v) (.next m' q e) ∧ q.erase = kk' v := fun v => by
      obtain ⟨o, hst, ho⟩ := ih v rfl
      obtain ⟨q, rfl, hqe⟩ := Out.erase_eq_next ho
      exact ⟨q, hst, hqe⟩
    choose q hqs hqe using hs
    refine ⟨_, .defer hqs fun hc v w => ?_, ?_⟩
    · have := hdep (by simpa only [Prog.erase_pred] using hc) v w
      rwa [Prog.erase_pred, Prog.erase_pred, ← hqe, ← hqe] at this
    · simp only [Out.erase, Prog.erase, hqe]
  | pick b hb =>
    cases p with
    | pick s k => simp only [Prog.erase] at hq; cases hq; exact ⟨_, .pick (s := s) b hb, rfl⟩
    | _ => cases hq
  | pickUB hn => cases p <;> simp only [Prog.erase] at hq <;> cases hq; exact ⟨_, .pickUB hn, rfl⟩
  | ub => cases p <;> simp only [Prog.erase] at hq <;> cases hq; exact ⟨_, .ub, rfl⟩
  | loop =>
    cases p <;> simp only [Prog.erase] at hq <;> cases hq
    exact ⟨_, .loop, by simp only [Out.erase, Prog.erase_loopStep]⟩
  | env => cases p <;> simp only [Prog.erase] at hq <;> cases hq; exact ⟨_, .env, rfl⟩

/-- A kernel with each program erased. -/
def Kernel.erase (k : Kernel) : @Kernel Sig.pred := @Kernel.mk Sig.pred fun ctx => (k.body ctx).erase

/-- A state of a launch with each program's rest erased. -/
def Config.erase {N : Nat} (c : Config N) : @Config Sig.pred N :=
  @Config.mk Sig.pred N c.mem fun i => (c.progs i).erase

theorem Config.erase_start (k : Kernel) (L : Launch) (init : Mem) :
    (Config.start k L init).erase = @Config.start Sig.pred k.erase L init := rfl

theorem Config.erase_update {N : Nat} (c : Config N) (i : Fin N) (m' : Mem) (p : Prog Unit) :
    (⟨m', Function.update c.progs i p⟩ : Config N).erase =
      @Config.mk Sig.pred N m' (Function.update (@Config.progs Sig.pred N c.erase) i p.erase) := by
  simp only [Config.erase, Config.mk.injEq, true_and]
  funext j
  by_cases hj : j = i
  · subst hj; simp
  · simp [Function.update_of_ne hj]

theorem LStep.erase {N : Nat} {c d : Config N} {ev : Fin N × Event} (h : LStep E c ev d) :
    @LStep Sig.pred N E c.erase ev d.erase := by
  cases h with
  | step i hst => rw [Config.erase_update]; exact @LStep.step Sig.pred N E c.erase i _ _ _ hst.erase

theorem LStep.unerase {N : Nat} {c : Config N} {ev : Fin N × Event} {d' : @Config Sig.pred N}
    (h : @LStep Sig.pred N E c.erase ev d') : ∃ d, LStep E c ev d ∧ d.erase = d' := by
  cases h with
  | step i hst =>
    obtain ⟨o, hst', ho⟩ := Step.unerase hst
    obtain ⟨q, rfl, rfl⟩ := Out.erase_eq_next ho
    exact ⟨_, .step i hst', Config.erase_update c i _ q⟩

theorem Reach.erase {N : Nat} {c d : Config N} {tr : List (Fin N × Event)}
    (h : Reach E c tr d) : @Reach Sig.pred N E c.erase tr d.erase := by
  induction h with
  | refl => exact @Reach.refl Sig.pred N E _
  | tail _ hl ih => exact @Reach.tail Sig.pred N E _ _ _ _ _ ih hl.erase

theorem Reach.unerase {N : Nat} {c : Config N} {tr : List (Fin N × Event)}
    {d' : @Config Sig.pred N} (h : @Reach Sig.pred N E c.erase tr d') :
    ∃ d, Reach E c tr d ∧ d.erase = d' := by
  generalize hc : c.erase = c' at h
  induction h with
  | refl => exact ⟨c, .refl c, hc⟩
  | tail _ hl ih =>
    obtain ⟨d, hr, rfl⟩ := ih
    obtain ⟨e, hl', rfl⟩ := LStep.unerase hl
    exact ⟨e, .tail hr hl', rfl⟩

theorem Config.erase_stuckUB {N : Nat} {c : Config N} :
    @Config.StuckUB Sig.pred N E c.erase ↔ c.StuckUB E := by
  constructor
  · rintro ⟨i, hi⟩
    obtain ⟨o, hst, ho⟩ := Step.unerase hi
    exact ⟨i, Out.erase_eq_ub ho ▸ hst⟩
  · rintro ⟨i, hi⟩
    exact ⟨i, hi.erase⟩

theorem Config.erase_done {N : Nat} {c : Config N} :
    @Config.Done Sig.pred N c.erase ↔ c.Done :=
  ⟨fun h i => Prog.erase_eq_ret (h i), fun h i => by simp only [Config.erase, h i]; rfl⟩

section Kernels

variable {k k' : Kernel} {L : Launch} {init : Mem}

/-- A run of the erasure, forever, is the erasure of a run of the kernel. -/
private theorem hang_unerase {f : ℕ → @Config Sig.pred L.size}
    (h0 : f 0 = (Config.start k L init).erase)
    (hf : ∀ n, ∃ ev, @LStep Sig.pred L.size E (f n) ev (f (n + 1))) :
    ∃ g : ℕ → Config L.size, g 0 = Config.start k L init ∧ ∀ n, ∃ ev, LStep E (g n) ev (g (n + 1)) := by
  have hnext : ∀ n (x : {c : Config L.size // c.erase = f n}),
      ∃ d : {c : Config L.size // c.erase = f (n + 1)}, ∃ ev, LStep E x.1 ev d.1 := by
    intro n x
    obtain ⟨ev, hl⟩ := hf n
    rw [← x.2] at hl
    obtain ⟨d, hl', hd⟩ := LStep.unerase hl
    exact ⟨⟨d, hd⟩, ev, hl'⟩
  let g : ∀ n, {c : Config L.size // c.erase = f n} :=
    fun n => Nat.rec ⟨Config.start k L init, h0.symm⟩ (fun n x => Classical.choose (hnext n x)) n
  exact ⟨fun n => (g n).1, rfl, fun n => Classical.choose_spec (hnext n (g n))⟩

/-- **A kernel has the outcomes of its erasure.** -/
theorem Outcomes.erase (o : Outcome) :
    @Outcomes Sig.pred E k.erase L init o ↔ Outcomes E k L init o := by
  cases o <;> simp only [Outcomes] <;> rw [← Config.erase_start]
  case ok =>
    constructor
    · rintro ⟨tr, c', hr, hd, hm⟩
      obtain ⟨c, hr', rfl⟩ := Reach.unerase hr
      exact ⟨tr, c, hr', Config.erase_done.1 hd, hm⟩
    · rintro ⟨tr, c, hr, hd, hm⟩
      exact ⟨tr, c.erase, hr.erase, Config.erase_done.2 hd, hm⟩
  case ub =>
    constructor
    · rintro ⟨tr, c', hr, hu⟩
      obtain ⟨c, hr', rfl⟩ := Reach.unerase hr
      exact ⟨tr, c, hr', Config.erase_stuckUB.1 hu⟩
    · rintro ⟨tr, c, hr, hu⟩
      exact ⟨tr, c.erase, hr.erase, Config.erase_stuckUB.2 hu⟩
  case hang =>
    constructor
    · rintro ⟨f, h0, hf⟩
      exact hang_unerase h0 hf
    · rintro ⟨f, h0, hf⟩
      refine ⟨fun n => (f n).erase, by simp only [h0], fun n => ?_⟩
      obtain ⟨ev, hl⟩ := hf n
      exact ⟨ev, hl.erase⟩

/-- **A kernel uses an atomic's result on a shared cell exactly when its erasure does.** -/
theorem UsesSharedAtomicResult.erase :
    @UsesSharedAtomicResult Sig.pred E k.erase L init ↔ UsesSharedAtomicResult E k L init := by
  unfold UsesSharedAtomicResult
  rw [← Config.erase_start]
  constructor
  · rintro ⟨tr, c', i, d, a, u, sem, sc, kk, hr, hi, ⟨v, w, hvw⟩, tr', c₂', hr₂, hsh⟩
    obtain ⟨c, hr', rfl⟩ := Reach.unerase hr
    obtain ⟨u', kk', hi', hk⟩ := Prog.erase_eq_rmw hi
    obtain ⟨c₂, hr₂', -⟩ := Reach.unerase hr₂
    refine ⟨tr, c, i, d, a, u', sem, sc, kk', hr', hi', ⟨v, w, fun he => hvw ?_⟩, tr', c₂, hr₂',
      hsh⟩
    rw [Prog.erase_pred, Prog.erase_pred, ← hk, ← hk, he]
  · rintro ⟨tr, c, i, d, a, u, sem, sc, kk, hr, hi, ⟨v, w, hvw⟩, tr', c₂, hr₂, hsh⟩
    refine ⟨tr, c.erase, i, d, a, ⟨Sig.UpdAllows u⟩, sem, sc, fun v => (kk v).erase, hr.erase,
      by simp only [Config.erase, hi, Prog.erase], ⟨v, w, ?_⟩, tr', c₂.erase, hr₂.erase, hsh⟩
    simpa only [Prog.erase_pred] using hvw

/-- **A kernel is well behaved exactly when its erasure is.** -/
theorem WellBehaved.erase :
    @WellBehaved Sig.pred E k.erase L init ↔ WellBehaved E k L init := by
  constructor
  · intro hw
    refine ⟨fun tr c hr hu => ?_, fun tr c hr => ?_, fun tr c hr => ?_, fun hu => ?_⟩
    · exact @WellBehaved.noUB Sig.pred E k.erase L init hw tr c.erase hr.erase
        (Config.erase_stuckUB.2 hu)
    · exact @WellBehaved.noInterRace Sig.pred E k.erase L init hw tr c.erase hr.erase
    · exact @WellBehaved.noIntraRace Sig.pred E k.erase L init hw tr c.erase hr.erase
    · exact @WellBehaved.noSharedAtomicResult Sig.pred E k.erase L init hw
        (UsesSharedAtomicResult.erase.2 hu)
  · intro hw
    refine @WellBehaved.mk Sig.pred E k.erase L init (fun tr c' hr hu => ?_) (fun tr c' hr => ?_)
      (fun tr c' hr => ?_) (fun hu => ?_)
    · obtain ⟨c, hr', rfl⟩ := Reach.unerase (c := Config.start k L init) hr
      exact hw.noUB tr c hr' (Config.erase_stuckUB.1 hu)
    · obtain ⟨c, hr', -⟩ := Reach.unerase (c := Config.start k L init) hr
      exact hw.noInterRace tr c hr'
    · obtain ⟨c, hr', -⟩ := Reach.unerase (c := Config.start k L init) hr
      exact hw.noIntraRace tr c hr'
    · exact hw.noSharedAtomicResult (UsesSharedAtomicResult.erase.1 hu)

/-- **A kernel is `Correct` exactly when its erasure is.** -/
theorem Correct.erase (Q : Mem → Prop) :
    @Correct Sig.pred E k.erase L init Q ↔ Correct E k L init Q := by
  unfold Correct
  simp only [WellBehaved.erase, Outcomes.erase]

/-- Two kernels with the same erasure, as kernels over `Sig.pred`. -/
theorem Kernel.erase_eq (h : ∀ ctx, (k.body ctx).erase = (k'.body ctx).erase) :
    k.erase = k'.erase := by
  unfold Kernel.erase; congr 1; funext ctx; exact h ctx

/-- **Two kernels with the same erasure have the same outcomes.** -/
theorem Outcomes.congr_erase (h : ∀ ctx, (k.body ctx).erase = (k'.body ctx).erase)
    (o : Outcome) : Outcomes E k L init o ↔ Outcomes E k' L init o := by
  rw [← Outcomes.erase, ← Outcomes.erase, Kernel.erase_eq h]

/-- **Two kernels with the same erasure are well behaved together.** -/
theorem WellBehaved.congr_erase (h : ∀ ctx, (k.body ctx).erase = (k'.body ctx).erase) :
    WellBehaved E k L init ↔ WellBehaved E k' L init := by
  rw [← WellBehaved.erase, ← WellBehaved.erase, Kernel.erase_eq h]

/-- **Two kernels with the same erasure are `Correct` together.** -/
theorem Correct.congr_erase (h : ∀ ctx, (k.body ctx).erase = (k'.body ctx).erase)
    (Q : Mem → Prop) : Correct E k L init Q ↔ Correct E k' L init Q := by
  rw [← Correct.erase, ← Correct.erase, Kernel.erase_eq h]

end Kernels

end TVA
