import TVA.Proof.Step

/-!
# `Safe`: what a prover proves about one program

`Safe E R W Q m p`: the program `p`, run *alone* from memory `m` in environment `E`, cannot reach
undefined behaviour and always ends, along every nondeterministic choice and every order of its
atomics (`Step.defer`). It reads only bytes of `R` and writes (stores or atomics) only bytes of
`W`, each byte at most once: after a store or an atomic, the rest of the program has the write
footprint `W` without those bytes. An atomic's bytes are outside `R`, and the program does not use
an atomic's result (its continuation is the same for every old value: `let _ ← tl.atomic_add …`,
or a `tl.atomic_*_unused`). It ends, returning `a`, in a memory `m'` with `Q a m'`.

When `R` and `W` are disjoint (which `correct_of_safe` requires), two accesses of the program
touch a common byte only when both are loads, so the program has no conflict within itself
(`IntraRace`); and since no atomic's result is used, no program uses the result of an atomic on a
cell another access reaches (`UsesSharedAtomicResult`). Atomics on cells that several programs
update are outside `Safe`'s reach: their programs' footprints overlap.

`Safe` is total correctness in the weakest-precondition style. It is an inductive predicate, one
constructor per effect of `Prog`, so every path is finite even when a choice has infinitely many
options.

How this fits: model → `Safe` per program → `correct_of_safe` (`TVA.Proof.Compose`). A proof of
`Safe` follows the model's text: each `tl.*` operation has a rule (`TVA.Proof.Rules`), and
`Safe.bind` (or `Safe.seq`, `Safe.bind_ro`) composes them; `Safe.mono` weakens the post-condition
or enlarges the write footprint. The rest of this file (`Safe.stepT`, `Safe.no_ub`, `Safe.acc`) is
what the composition theorem uses: a safe program's steps stay inside its footprint, never reach
undefined behaviour, and cannot go on forever.
-/
namespace TVA

/-! ## The predicate -/

/-- The program `p`, run alone from `m` in environment `E`, is safe with read footprint `R`, write
footprint `W` (each byte written at most once) and post-condition `Q` on its result and final
memory. -/
inductive Safe (E : Env) (R : Set Int) :
    Set Int → {α : Type} → (α → Mem → Prop) → Mem → Prog α → Prop where
  /-- The program has ended with `a` in `m`: the post-condition holds. -/
  | ret {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {a : α} :
      Q a m → Safe E R W Q m (.ret a)
  /-- A load: every enabled lane's element can be read, its bytes are in `R`, and the rest is safe
  for every value the load may return. -/
  | load {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {d : DType} {s : Shape}
      {addrs : Idx s → Option Int} {k : Tensor s d.Val → Prog α} :
      ¬ LoadBad m d addrs → accBytes d addrs ⊆ R →
      (∀ v, LoadOk m d addrs v → Safe E R W Q m (k v)) →
      Safe E R W Q m (.load d s addrs k)
  /-- A store writes bytes of `W`; the rest of the program may not write them again. -/
  | store {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {d : DType} {s : Shape}
      {addrs : Idx s → Option Int} {vals : Tensor s d.Val} {k : Prog α} :
      StoreOk m d addrs → accBytes d addrs ⊆ W →
      (∀ order, order.Perm (enabled addrs) →
        Safe E R (W \ accBytes d addrs) Q (writeAll m d addrs vals order) k) →
      Safe E R W Q m (.store d s addrs vals k)
  /-- An atomic on bytes of `W` outside `R`, whose result the program does not use (the rest of
  the program is the same for every old value); the rest may not touch the bytes again. The
  constructor covers performing it now; performing it later (`Step.defer`) follows
  (`Safe.stepT`). -/
  | rmw {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {d : DType} {a : Int}
      {upd : d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope} {k : d.Val → Prog α} :
      (∃ old, m.read d a = some old ∧ ∃ new, upd old new) →
      Mem.span d a ⊆ W → Disjoint (Mem.span d a) R → (∀ v w, k v = k w) →
      (∀ old new, m.read d a = some old → upd old new →
        Safe E R (W \ Mem.span d a) Q (m.write d a new) (k old)) →
      Safe E R W Q m (.rmw d a upd sem sc k)
  /-- A choice: some value is allowed, and the rest is safe for every allowed value. -/
  | pick {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {β : Type} {P : β → Prop}
      {k : β → Prog α} :
      (∃ b, P b) → (∀ b, P b → Safe E R W Q m (k b)) → Safe E R W Q m (.pick P k)
  /-- A loop: its first unfolding is safe. -/
  | loop {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {σ β : Type} {init : σ}
      {body : σ → Prog (σ ⊕ β)} {k : β → Prog α} :
      Safe E R W Q m (Prog.loopStep init body k) → Safe E R W Q m (.loop init body k)
  /-- Reading the environment: the rest is safe for `E`. -/
  | env {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {k : Env → Prog α} :
      Safe E R W Q m (k E) → Safe E R W Q m (.env k)

namespace Safe

variable {E : Env} {R : Set Int}

/-- Unfolding a loop and then binding is binding and then unfolding. -/
theorem loopStep_bind {α β γ σ : Type} (init : σ) (body : σ → Prog (σ ⊕ β)) (k : β → Prog α)
    (f : α → Prog γ) :
    (Prog.loopStep init body k).bind f = Prog.loopStep init body (fun b => (k b).bind f) := by
  unfold Prog.loopStep
  rw [Prog.bind_assoc]
  congr 1
  funext r
  cases r <;> rfl

/-- **Weakening**: a weaker post-condition and a larger write footprint. (The read footprint
cannot grow: an atomic's bytes must stay outside it.) -/
theorem mono {α : Type} {Q Q' : α → Mem → Prop} {W W' : Set Int} {m : Mem} {p : Prog α}
    (h : Safe E R W Q m p) (hQ : ∀ a m, Q a m → Q' a m) (hW : W ⊆ W') :
    Safe E R W' Q' m p := by
  induction h generalizing W' with
  | ret hq => exact .ret (hQ _ _ hq)
  | load hb hRs _ ih => exact .load hb hRs fun v hv => ih v hv hQ hW
  | store hok hWs _ ih =>
    exact .store hok (hWs.trans hW) fun o hp => ih o hp hQ (Set.sdiff_subset_sdiff_left hW)
  | rmw hex hS hSR hc _ ih =>
    exact .rmw hex (hS.trans hW) hSR hc fun o n ho hn =>
      ih o n ho hn hQ (Set.sdiff_subset_sdiff_left hW)
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hQ hW
  | loop _ ih => exact .loop (ih hQ hW)
  | env _ ih => exact .env (ih hQ hW)

/-- Removing bytes of the first of two disjoint footprints from their union. -/
theorem union_diff_of_sub {W₁ W₂ B : Set Int} (hd : Disjoint W₁ W₂) (hB : B ⊆ W₁) :
    (W₁ ∪ W₂) \ B = (W₁ \ B) ∪ W₂ := by
  ext x
  simp only [Set.mem_sdiff, Set.mem_union]
  constructor
  · rintro ⟨h1 | h2, hn⟩
    · exact Or.inl ⟨h1, hn⟩
    · exact Or.inr h2
  · rintro (⟨h1, hn⟩ | h2)
    · exact ⟨Or.inl h1, hn⟩
    · exact ⟨Or.inr h2, fun hb => Set.disjoint_left.mp hd (hB hb) h2⟩

/-- **Sequencing.** To show `p.bind f` safe, show `p` safe with the post-condition "the rest is
safe". The two parts write disjoint footprints. -/
theorem bind {α β : Type} {Q : β → Mem → Prop} {f : α → Prog β} {p : Prog α} {m : Mem}
    {W₁ W₂ : Set Int} (h : Safe E R W₁ (fun a m => Safe E R W₂ Q m (f a)) m p)
    (hd : Disjoint W₁ W₂) : Safe E R (W₁ ∪ W₂) Q m (p.bind f) := by
  generalize hQ' : (fun a m => Safe E R W₂ Q m (f a)) = Q' at h
  induction h with
  | ret hq => subst hQ'; exact hq.mono (fun _ _ h => h) Set.subset_union_right
  | load hb hR _ ih => exact .load hb hR fun v hv => ih v hv hd hQ'
  | store hok hW _ ih =>
    refine .store hok (hW.trans Set.subset_union_left) fun o hp => ?_
    rw [union_diff_of_sub hd hW]
    exact ih o hp (hd.mono_left Set.sdiff_subset) hQ'
  | rmw hex hS hSR hc _ ih =>
    refine .rmw hex (hS.trans Set.subset_union_left) hSR (fun v w => by rw [hc v w]) fun o n ho hn => ?_
    rw [union_diff_of_sub hd hS]
    exact ih o n ho hn (hd.mono_left Set.sdiff_subset) hQ'
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hd hQ'
  | loop _ ih => exact .loop (by rw [← loopStep_bind]; exact ih hd hQ')
  | env _ ih => exact .env (ih hd hQ')

/-- `Safe.bind` for `p >>= f` (a line of a `do` block followed by the rest). -/
theorem seq {α β : Type} {Q : β → Mem → Prop} {f : α → Prog β} {p : Prog α} {m : Mem}
    {W₁ W₂ : Set Int} (h : Safe E R W₁ (fun a m => Safe E R W₂ Q m (f a)) m p)
    (hd : Disjoint W₁ W₂) : Safe E R (W₁ ∪ W₂) Q m (p >>= f) := bind h hd

/-- `Safe.bind` when the first part writes nothing (it only loads and computes). -/
theorem bind_ro {α β : Type} {Q : β → Mem → Prop} {f : α → Prog β} {p : Prog α} {m : Mem}
    {W : Set Int} (h : Safe E R ∅ (fun a m => Safe E R W Q m (f a)) m p) :
    Safe E R W Q m (p.bind f) := by
  simpa using bind h (Set.empty_disjoint W)

/-! ## Safe programs step safely -/

/-- **Steps of a safe program.** Run a program that is safe from its own memory `m₀` against a
memory `m` that agrees with `m₀` on the footprint. Every step it takes, an atomic performed later
(`Step.defer`) included, is a normal step that reads inside `R` or inside what it writes, and
writes inside `W`. The same step is possible from `m₀`, giving `m₀` with the step's writes, and
the rest is safe there with the written bytes removed from the write footprint. -/
theorem stepT {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m₀ m : Mem} {p : Prog α}
    (h : Safe E R W Q m₀ p) (hag : ∀ x ∈ R ∪ W, m x = m₀ x)
    {m' : Mem} {p' : Prog α} {e : Event} (hs : Step E m p (.next m' p' e)) :
    e.reads ⊆ R ∪ e.writes ∧ e.writes ⊆ W ∧
      Step E m₀ p (.next (Mem.mix m₀ m' e.writes) p' e) ∧
      Safe E R (W \ e.writes) Q (Mem.mix m₀ m' e.writes) p' := by
  generalize ho : (Out.next m' p' e) = o at hs
  induction hs generalizing W m₀ m' p' e with
  | load v hok =>
    cases ho
    cases h with
    | load hb hR hk =>
      have hok₀ := (loadOk_congr (fun x hx => hag x (Or.inl (hR hx)))).mp hok
      refine ⟨fun x hx => Or.inl (hR hx), by simp, ?_, ?_⟩
      · simpa using Step.load v hok₀
      · simpa using hk v hok₀
  | store order hok hperm =>
    cases ho
    cases h with
    | store hok₀ hW hk =>
      have hag' : ∀ x ∈ accBytes _ _, m₀ x = _ := fun x hx => (hag x (Or.inr (hW hx))).symm
      refine ⟨by simp, hW, ?_, ?_⟩
      · simp only; rw [writeAll_mix _ _ hag']; exact .store order hok₀ hperm
      · simp only; rw [writeAll_mix _ _ hag']; exact hk order hperm
  | rmw old new hr hu =>
    cases ho
    cases h with
    | rmw hex hS hSR _ hk =>
      have hr₀ := (Mem.read_congr fun x hx => (hag x (Or.inr (hS hx))).symm).trans hr
      refine ⟨fun x hx => Or.inr hx, hS, ?_, ?_⟩
      · simp only; rw [Mem.mix_write_self]; exact .rmw old new hr₀ hu
      · simp only; rw [Mem.mix_write_self]; exact hk old new hr₀ hu
  | pick b hb =>
    cases ho
    cases h with
    | pick _ hk =>
      refine ⟨by simp [Event.silent], by simp [Event.silent], ?_, ?_⟩
      · simpa [Event.silent] using Step.pick b hb
      · simpa [Event.silent] using hk b hb
  | loop =>
    cases ho
    cases h with
    | loop hk =>
      refine ⟨by simp [Event.silent], by simp [Event.silent], ?_, ?_⟩
      · simpa [Event.silent] using Step.loop
      · simpa [Event.silent] using hk
  | env =>
    cases ho
    cases h with
    | env hk =>
      refine ⟨by simp [Event.silent], by simp [Event.silent], ?_, ?_⟩
      · simpa [Event.silent] using Step.env
      · simpa [Event.silent] using hk
  | defer hk_step hc ih =>
    cases ho
    cases h with
    | rmw hex hS hSR hconst hk =>
      rename_i d a upd sem sc k k' m'' e''
      obtain ⟨old, hr₀, n₀, hu₀⟩ := hex
      have hagw : ∀ n, ∀ x ∈ R ∪ (W \ Mem.span d a), m x = (m₀.write d a n) x := by
        intro n x hx
        have hxS : x ∉ Mem.span d a := by
          rcases hx with hx | hx
          · exact fun h => Set.disjoint_left.mp hSR h hx
          · exact hx.2
        rw [Mem.write_frame _ _ _ _ hxS]
        exact hag x (hx.elim Or.inl (fun h => Or.inr h.1))
      obtain ⟨hR', hW', -, -⟩ := ih old (hk old n₀ hr₀ hu₀) (hagw n₀) rfl
      have hWS : Disjoint (Mem.span d a) e''.writes :=
        Set.disjoint_left.mpr fun x hx hw => (hW' hw).2 hx
      refine ⟨hR', fun x hx => (hW' hx).1, ?_, ?_⟩
      · refine .defer (fun v => Step.transfer (hk_step v) fun x hx => (hag x ?_).symm) hc
        rcases hx with hx | hx
        · rcases hR' hx with h | h
          · exact Or.inl h
          · exact Or.inr (hW' h).1
        · exact Or.inr (hW' hx).1
      · have hread : (Mem.mix m₀ m'' e''.writes).read d a = some old := by
          rw [← hr₀]; exact Mem.read_congr fun x hx => Mem.mix_out (Set.disjoint_left.mp hWS hx)
        refine .rmw ⟨old, hread, n₀, hu₀⟩ (fun x hx => ⟨hS hx, Set.disjoint_left.mp hWS hx⟩) hSR
          (hc hconst) ?_
        intro o n ho hn
        rw [hread] at ho
        cases ho
        obtain ⟨-, -, -, hsafe⟩ := ih old (hk old n hr₀ hn) (hagw n) rfl
        rw [Mem.mix_write n hWS, Set.sdiff_sdiff_comm] at hsafe
        exact hsafe
  | loadUB => cases ho
  | storeUB => cases ho
  | rmwUB => cases ho
  | pickUB => cases ho
  | ub => cases ho

/-- A safe program does not step into undefined behaviour, in any memory that agrees with its own
on the footprint. -/
theorem no_ub {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m₀ m : Mem} {p : Prog α}
    (h : Safe E R W Q m₀ p) (hag : ∀ x ∈ R ∪ W, m x = m₀ x) : ¬ Step E m p .ub := by
  intro hs
  cases hs with
  | loadUB hbad =>
    cases h with
    | load hb hR _ => exact hb ((loadBad_congr fun x hx => hag x (Or.inl (hR hx))).mp hbad)
  | storeUB hbad =>
    cases h with
    | store hok hW _ => exact hbad ((storeOk_congr fun x hx => hag x (Or.inr (hW hx))).mpr hok)
  | rmwUB hbad =>
    cases h with
    | rmw hex hS _ _ _ =>
      have hrd := Mem.read_congr (fun x hx => hag x (Or.inr (hS hx)))
      obtain ⟨old, hr, new, hu⟩ := hex
      rcases hbad with h1 | ⟨old', hr', hno⟩
      · rw [hrd, hr] at h1; cases h1
      · rw [hrd, hr] at hr'; cases hr'; exact hno new hu
  | pickUB hno => cases h with | pick hex _ => obtain ⟨b, hb⟩ := hex; exact hno b hb
  | ub => cases h

/-- **Progress and preservation.** Every step of a safe program is a normal step, within the
footprint, to a program that is safe with the written bytes removed from its write footprint. -/
theorem step {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {p : Prog α}
    (h : Safe E R W Q m p) {o : Out α} (hs : Step E m p o) :
    ∃ m' p' e, o = .next m' p' e ∧ e.reads ⊆ R ∪ e.writes ∧ e.writes ⊆ W ∧
      Safe E R (W \ e.writes) Q m' p' := by
  cases o with
  | ub => exact absurd hs (h.no_ub fun _ _ => rfl)
  | next m' p' e =>
    obtain ⟨h1, h2, -, h4⟩ := h.stepT (fun _ _ => rfl) hs
    rw [Mem.mix_of_frame (Step.frame hs)] at h4
    exact ⟨m', p', e, rfl, h1, h2, h4⟩

/-! ## Safe programs end -/

/-- One step of a program alone, on its state (memory, program), as a relation from the state
after to the state before (so that `Acc` says every run is finite). -/
def Next {α : Type} (E : Env) : Mem × Prog α → Mem × Prog α → Prop
  | (m', p'), (m, p) => ∃ e, Step E m p (.next m' p' e)

/-- An atomic that stays pending only while its continuation takes steps: the run from the
atomic's state is as long as the continuation's. -/
private theorem acc_rmw {α : Type} {Q : α → Mem → Prop} {d : DType} {a : Int}
    {upd : d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope} :
    ∀ s : Mem × Prog α, Acc (Next E) s → ∀ (W : Set Int) (m : Mem) (k : d.Val → Prog α)
      (old n₀ : d.Val), s = (m.write d a n₀, k old) → m.read d a = some old → upd old n₀ →
      Safe E R W Q m (.rmw d a upd sem sc k) →
      (∀ new, upd old new → Acc (Next E) (m.write d a new, k old)) →
      Acc (Next E) (m, .rmw d a upd sem sc k) := by
  intro s hs
  induction hs with
  | intro s _ ih =>
    intro W m k old n₀ hs hr hu hsafe hall
    constructor
    rintro ⟨m', p'⟩ ⟨e, hstep⟩
    cases hstep with
    | rmw old' new' hr' hu' =>
      rw [hr] at hr'
      cases hr'
      exact hall new' hu'
    | defer hk' hc =>
      rename_i k'
      cases hsafe with
      | rmw hex hS hSR hconst hk =>
        obtain ⟨-, -, -, hsafe'⟩ :=
          (Safe.rmw hex hS hSR hconst hk).stepT (m := m) (fun _ _ => rfl) (.defer hk' hc)
        rw [Mem.mix_of_frame (Step.frame (hk' old))] at hsafe'
        have hagw : ∀ n, ∀ x ∈ R ∪ (W \ Mem.span d a), m x = (m.write d a n) x := by
          intro n x hx
          have hxS : x ∉ Mem.span d a := by
            rcases hx with hx | hx
            · exact fun h => Set.disjoint_left.mp hSR h hx
            · exact hx.2
          rw [Mem.write_frame _ _ _ _ hxS]
        have hWS : Disjoint (Mem.span d a) e.writes := by
          obtain ⟨-, hW', -, -⟩ := (hk old n₀ hr hu).stepT (hagw n₀) (hk' old)
          exact Set.disjoint_left.mpr fun x hx hw => (hW' hw).2 hx
        have hcomm : ∀ n, upd old n → Next E (m'.write d a n, k' old) (m.write d a n, k old) := by
          intro n hn
          obtain ⟨-, -, hst, -⟩ := (hk old n hr hn).stepT (hagw n) (hk' old)
          rw [Mem.mix_write n hWS, Mem.mix_of_frame (Step.frame (hk' old))] at hst
          exact ⟨e, hst⟩
        have hr' : m'.read d a = some old := by
          rw [← hr]
          exact Mem.read_congr fun x hx => Step.frame (hk' old) x (Set.disjoint_left.mp hWS hx)
        exact ih (m'.write d a n₀, k' old) (by rw [hs]; exact hcomm n₀ hu) (W \ e.writes) m' k'
          old n₀ rfl hr' hu hsafe' fun new hnew => (hall new hnew).inv (hcomm new hnew)

/-- **Termination.** From a safe state, every sequence of steps of the program alone is
finite. -/
theorem acc {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {p : Prog α}
    (h : Safe E R W Q m p) : Acc (Next E) (m, p) := by
  induction h with
  | ret => constructor; rintro ⟨_, _⟩ ⟨e, hs⟩; cases hs
  | load _ _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | load v hok => exact ih v hok
  | store _ _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩
    cases hs with | store order _ hperm => exact ih order hperm
  | rmw hex hS hSR hc hk ih =>
    obtain ⟨old, hr, n₀, hu⟩ := hex
    exact acc_rmw _ (ih old n₀ hr hu) _ _ _ old n₀ rfl hr hu
      (.rmw ⟨old, hr, n₀, hu⟩ hS hSR hc hk)
      fun new hnew => ih old new hr hnew
  | pick _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | pick b hb => exact ih b hb
  | loop _ ih => constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | loop => exact ih
  | env _ ih => constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | env => exact ih

end Safe

end TVA
