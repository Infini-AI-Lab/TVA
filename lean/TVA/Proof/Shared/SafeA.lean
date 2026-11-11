import TVA.Proof.Shared.Log

/-!
# `SafeA`: one program that shares cells through atomics

`SafeA E R W C Q log m p` is `Safe` (`TVA.Proof.Safe`) extended by one kind of step: an atomic on
a **shared cell**, a cell of the set `C` that other programs (or other atomics of this program)
also update with atomics. Run alone, a program cannot know what it finds in such a cell, so the
rule for it quantifies over *every* old value, and records the step in a **log**
(`AtomicStep`: the cell, the old value, the new value). The post-condition sees the log.

Everything else is as in `Safe`: private loads read `R`, private stores and private atomics write
`W` (each byte once), private atomics' bytes are outside `R`. The shared cells must lie outside
`R ∪ W` (the composition theorem asks for it), so a program never loads or stores a shared cell:
it only updates it with atomics.

**Why this is well behaved.** A shared atomic must be `acq_rel` (Triton's default, and what every
`tl.atomic_*_unused` is at `gpu` or `sys` scope) at `gpu` or `sys` scope, on the whole cell. Two
such atomics of different programs on the same cell are `Event.Strong`, so they do not race, and
the cells of `C` are equal or disjoint, so no two atomics overlap partially. Two atomics of one
program on the same cell are `Event.SameAtomic`, so they are no conflict within the program. And
no atomic's result is used: the continuation of every atomic, shared or private, is the same for
every old value (`let _ ← tl.atomic_add …`), so `UsesSharedAtomicResult` cannot hold.

How this fits: prove each program `SafeA` (private parts with the `Safe` rules, lifted by
`SafeA.of_safe`; shared atomics with `SafeA.atomic_add`, `SafeA.atomic`, `SafeA.atomic1` from
`TVA.Proof.Shared.Rules`), then `correct_of_safeA` (`TVA.Proof.Shared.Correct`) gives `Correct`.
The rest of this file (`SafeA.stepT`, `SafeA.no_ub`) is what the composition uses.
-/
namespace TVA

/-- One program, alone, with private read footprint `R`, private write footprint `W` (each byte
written at most once), and shared cells `C`, which it updates only with atomics. Its
post-condition sees the log of its shared atomics. -/
inductive SafeA (E : Env) (R : Set Int) :
    Set Int → Set (DType × Int) → {α : Type} → (α → List AtomicStep → Mem → Prop) →
      List AtomicStep → Mem → Prog α → Prop where
  /-- The program has ended: the post-condition holds of its result, its log and its memory. -/
  | ret {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {a : α} : Q a log m → SafeA E R W C Q log m (.ret a)
  /-- A private load, as in `Safe`. -/
  | load {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {k : Tensor s d.Val → Prog α} :
      ¬ LoadBad m d addrs → accBytes d addrs ⊆ R →
      (∀ v, LoadOk m d addrs v → SafeA E R W C Q log m (k v)) →
      SafeA E R W C Q log m (.load d s addrs k)
  /-- A private store, as in `Safe`. -/
  | store {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {vals : Tensor s d.Val} {k : Prog α} :
      StoreOk m d addrs → accBytes d addrs ⊆ W →
      (∀ order, order.Perm (enabled addrs) →
        SafeA E R (W \ accBytes d addrs) C Q log (writeAll m d addrs vals order) k) →
      SafeA E R W C Q log m (.store d s addrs vals k)
  /-- A private atomic, as in `Safe`: on bytes of `W` outside `R`, its result unused. -/
  | rmw {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop}
      {sem : MemSem} {sc : Scope} {k : d.Val → Prog α} :
      (∃ old, m.read d a = some old ∧ ∃ new, upd old new) →
      Mem.span d a ⊆ W → Disjoint (Mem.span d a) R → (∀ v w, k v = k w) →
      (∀ old new, m.read d a = some old → upd old new →
        SafeA E R (W \ Mem.span d a) C Q log (m.write d a new) (k old)) →
      SafeA E R W C Q log m (.rmw d a upd sem sc k)
  /-- **A shared atomic**: an aligned cell of `C`, `acq_rel` at `gpu` or `sys` scope, an update
  with a new value for every old value, its result unused. Any old value may be found; the step
  is appended to the log. -/
  | shared {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop}
      {sem : MemSem} {sc : Scope} {k : d.Val → Prog α} :
      Mem.Aligned d a → (d, a) ∈ C → sem = .acqRel → sc ≠ .cta → (∀ old, ∃ new, upd old new) →
      (∀ v w, k v = k w) →
      (∀ old new, upd old new → SafeA E R W C Q (log ++ [⟨d, a, old, new⟩]) m (k old)) →
      SafeA E R W C Q log m (.rmw d a upd sem sc k)
  /-- A choice, as in `Safe`. -/
  | pick {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {β : Type} {P : β → Prop} {k : β → Prog α} :
      (∃ b, P b) → (∀ b, P b → SafeA E R W C Q log m (k b)) → SafeA E R W C Q log m (.pick P k)
  /-- A loop: its first unfolding. -/
  | loop {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {σ β : Type} {init : σ} {body : σ → Prog (σ ⊕ β)}
      {k : β → Prog α} :
      SafeA E R W C Q log m (Prog.loopStep init body k) → SafeA E R W C Q log m (.loop init body k)
  /-- Reading the environment. -/
  | env {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {k : Env → Prog α} :
      SafeA E R W C Q log m (k E) → SafeA E R W C Q log m (.env k)

namespace SafeA

variable {E : Env} {R : Set Int} {C : Set (DType × Int)}

/-- **Weakening**: a weaker post-condition and a larger private write footprint. -/
theorem mono {α : Type} {Q Q' : α → List AtomicStep → Mem → Prop} {W W' : Set Int} {p : Prog α}
    {log : List AtomicStep} {m : Mem} (h : SafeA E R W C Q log m p)
    (hQ : ∀ a log m, Q a log m → Q' a log m) (hW : W ⊆ W') : SafeA E R W' C Q' log m p := by
  induction h generalizing W' with
  | ret hq => exact .ret (hQ _ _ _ hq)
  | load hb hR _ ih => exact .load hb hR fun v hv => ih v hv hQ hW
  | store hok hWs _ ih =>
    exact .store hok (hWs.trans hW) fun o hp => ih o hp hQ (Set.sdiff_subset_sdiff_left hW)
  | rmw hex hS hSR hc _ ih =>
    exact .rmw hex (hS.trans hW) hSR hc fun o n ho hn =>
      ih o n ho hn hQ (Set.sdiff_subset_sdiff_left hW)
  | shared hal hC hsem hsc htot hc _ ih =>
    exact .shared hal hC hsem hsc htot hc fun o n h => ih o n h hQ hW
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hQ hW
  | loop _ ih => exact .loop (ih hQ hW)
  | env _ ih => exact .env (ih hQ hW)

/-- **Sequencing.** To show `p.bind f`, show `p` with the post-condition "the rest is `SafeA`
from here, with this log". The two parts write disjoint private footprints. -/
theorem bind {α β : Type} {Q : β → List AtomicStep → Mem → Prop} {f : α → Prog β} {p : Prog α}
    {log : List AtomicStep} {m : Mem} {W₁ W₂ : Set Int}
    (h : SafeA E R W₁ C (fun a log m => SafeA E R W₂ C Q log m (f a)) log m p)
    (hd : Disjoint W₁ W₂) : SafeA E R (W₁ ∪ W₂) C Q log m (p.bind f) := by
  generalize hQ' : (fun a log m => SafeA E R W₂ C Q log m (f a)) = Q' at h
  induction h with
  | ret hq => subst hQ'; exact hq.mono (fun _ _ _ h => h) Set.subset_union_right
  | load hb hR _ ih => exact .load hb hR fun v hv => ih v hv hd hQ'
  | store hok hW _ ih =>
    refine .store hok (hW.trans Set.subset_union_left) fun o hp => ?_
    rw [Safe.union_diff_of_sub hd hW]
    exact ih o hp (hd.mono_left Set.sdiff_subset) hQ'
  | rmw hex hS hSR hc _ ih =>
    refine .rmw hex (hS.trans Set.subset_union_left) hSR (fun v w => by rw [hc v w])
      fun o n ho hn => ?_
    rw [Safe.union_diff_of_sub hd hS]
    exact ih o n ho hn (hd.mono_left Set.sdiff_subset) hQ'
  | shared hal hC hsem hsc htot hc _ ih =>
    exact .shared hal hC hsem hsc htot (fun v w => by rw [hc v w]) fun o n h => ih o n h hd hQ'
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hd hQ'
  | loop _ ih => exact .loop (by rw [← Safe.loopStep_bind]; exact ih hd hQ')
  | env _ ih => exact .env (ih hd hQ')

/-- **Private segments.** A segment with no shared atomic, proved with the ordinary `Safe` rules,
followed by the rest, which writes a disjoint private footprint. Use it for every part of a
program between its shared atomics. -/
theorem of_safe {α β : Type} {Q : β → List AtomicStep → Mem → Prop} {f : α → Prog β} {p : Prog α}
    {log : List AtomicStep} {m : Mem} {W₁ W₂ : Set Int}
    (h : Safe E R W₁ (fun a m => SafeA E R W₂ C Q log m (f a)) m p) (hd : Disjoint W₁ W₂) :
    SafeA E R (W₁ ∪ W₂) C Q log m (p.bind f) := by
  generalize hQ' : (fun a m => SafeA E R W₂ C Q log m (f a)) = Q' at h
  induction h with
  | ret hq => subst hQ'; exact hq.mono (fun _ _ _ h => h) Set.subset_union_right
  | load hb hR _ ih => exact .load hb hR fun v hv => ih v hv hd hQ'
  | store hok hW _ ih =>
    refine .store hok (hW.trans Set.subset_union_left) fun o hp => ?_
    rw [Safe.union_diff_of_sub hd hW]
    exact ih o hp (hd.mono_left Set.sdiff_subset) hQ'
  | rmw hex hS hSR hc _ ih =>
    refine .rmw hex (hS.trans Set.subset_union_left) hSR (fun v w => by rw [hc v w])
      fun o n ho hn => ?_
    rw [Safe.union_diff_of_sub hd hS]
    exact ih o n ho hn (hd.mono_left Set.sdiff_subset) hQ'
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hd hQ'
  | loop _ ih => exact .loop (by rw [← Safe.loopStep_bind]; exact ih hd hQ')
  | env _ ih => exact .env (ih hd hQ')

/-- With a post-condition that ignores the log's order (`PermInv`), the log may be permuted. -/
theorem perm {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {log log' : List AtomicStep} {m : Mem} {p : Prog α} (hQ : PermInv Q)
    (h : SafeA E R W C Q log m p) (hp : log.Perm log') : SafeA E R W C Q log' m p := by
  induction h generalizing log' with
  | ret hq => exact .ret (hQ _ _ _ _ hp hq)
  | load hb hR _ ih => exact .load hb hR fun v hv => ih v hv hQ hp
  | store hok hW _ ih => exact .store hok hW fun o ho => ih o ho hQ hp
  | rmw hex hS hSR hc _ ih => exact .rmw hex hS hSR hc fun o n ho hn => ih o n ho hn hQ hp
  | shared hal hC hsem hsc htot hc _ ih =>
    exact .shared hal hC hsem hsc htot hc fun o n h => ih o n h hQ (hp.append_right _)
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hQ hp
  | loop _ ih => exact .loop (ih hQ hp)
  | env _ ih => exact .env (ih hQ hp)

/-- A program about to perform an atomic does not use its result. -/
theorem rmw_const {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {log : List AtomicStep} {m : Mem} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop}
    {sem : MemSem} {sc : Scope} {k : d.Val → Prog α}
    (h : SafeA E R W C Q log m (.rmw d a upd sem sc k)) : ∀ v w, k v = k w := by
  cases h with
  | rmw _ _ _ hc _ => exact hc
  | shared _ _ _ _ _ hc _ => exact hc

/-! ## Steps of a program with shared atomics -/

/-- A shared step writes no byte of a private footprint disjoint from the cells. -/
private theorem shared_not_sub {W' : Set Int} {e : Event} {s : AtomicStep}
    (hdA : Disjoint (cellBytes C) W') (hC : (s.d, s.a) ∈ C) (hw : e.writes = Mem.span s.d s.a) :
    ¬ e.writes ⊆ W' := by
  intro hsub
  obtain ⟨x, hx⟩ := span_nonempty s.d s.a
  rw [← hw] at hx
  exact Set.disjoint_left.mp hdA (span_sub_cellBytes hC (hw ▸ hx)) (hsub hx)

/-- **Steps.** Run against the launch's memory `m`, a program that is `SafeA` in its own memory
`m₀` (which agrees with `m` on the private footprint) takes either a private step (as
`Safe.stepT`), or a shared atomic step, which leaves its own memory alone and appends to its log.
An atomic performed later (`Step.defer`) is covered, with the log reordered (`PermInv`). -/
theorem stepT {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {log : List AtomicStep} {m₀ m : Mem} {p : Prog α} (h : SafeA E R W C Q log m₀ p)
    (hQ : PermInv Q) (hdA : Disjoint (cellBytes C) (R ∪ W)) (hag : ∀ x ∈ R ∪ W, m x = m₀ x)
    {m' : Mem} {p' : Prog α} {e : Event} (hs : Step E m p (.next m' p' e)) :
    (e.reads ⊆ R ∪ e.writes ∧ e.writes ⊆ W ∧
      Step E m₀ p (.next (Mem.mix m₀ m' e.writes) p' e) ∧
      SafeA E R (W \ e.writes) C Q log (Mem.mix m₀ m' e.writes) p') ∨
    (∃ s : AtomicStep, (s.d, s.a) ∈ C ∧ e.cell = some (s.d, s.a) ∧ m.read s.d s.a = some s.old ∧
      m' = m.write s.d s.a s.new ∧ e.reads = Mem.span s.d s.a ∧ e.writes = Mem.span s.d s.a ∧
      (∃ sc, e.atomic = some (.acqRel, sc) ∧ sc ≠ .cta) ∧
      SafeA E R W C Q (log ++ [s]) m₀ p') := by
  generalize ho : (Out.next m' p' e) = o at hs
  induction hs generalizing W m₀ log m' p' e with
  | load v hok =>
    cases ho
    cases h with
    | load hb hR hk =>
      have hok₀ := (loadOk_congr (fun x hx => hag x (Or.inl (hR hx)))).mp hok
      refine Or.inl ⟨fun x hx => Or.inl (hR hx), by simp, ?_, ?_⟩
      · simpa using Step.load v hok₀
      · simpa using hk v hok₀
  | store order hok hperm =>
    cases ho
    cases h with
    | store hok₀ hW hk =>
      have hag' : ∀ x ∈ accBytes _ _, m₀ x = _ := fun x hx => (hag x (Or.inr (hW hx))).symm
      refine Or.inl ⟨by simp, hW, ?_, ?_⟩
      · simp only; rw [writeAll_mix _ _ hag']; exact .store order hok₀ hperm
      · simp only; rw [writeAll_mix _ _ hag']; exact hk order hperm
  | rmw old new hr hu =>
    cases ho
    cases h with
    | rmw hex hS hSR _ hk =>
      have hr₀ := (Mem.read_congr fun x hx => (hag x (Or.inr (hS hx))).symm).trans hr
      refine Or.inl ⟨fun x hx => Or.inr hx, hS, ?_, ?_⟩
      · simp only; rw [Mem.mix_write_self]; exact .rmw old new hr₀ hu
      · simp only; rw [Mem.mix_write_self]; exact hk old new hr₀ hu
    | shared hal hC hsem hsc htot _ hk =>
      subst hsem
      exact Or.inr ⟨⟨_, _, old, new⟩, hC, rfl, hr, rfl, rfl, rfl, ⟨_, rfl, hsc⟩, hk old new hu⟩
  | pick b hb =>
    cases ho
    cases h with
    | pick _ hk =>
      refine Or.inl ⟨by simp [Event.silent], by simp [Event.silent], ?_, ?_⟩
      · simpa [Event.silent] using Step.pick b hb
      · simpa [Event.silent] using hk b hb
  | loop =>
    cases ho
    cases h with
    | loop hk =>
      refine Or.inl ⟨by simp [Event.silent], by simp [Event.silent], ?_, ?_⟩
      · simpa [Event.silent] using Step.loop
      · simpa [Event.silent] using hk
  | env =>
    cases ho
    cases h with
    | env hk =>
      refine Or.inl ⟨by simp [Event.silent], by simp [Event.silent], ?_, ?_⟩
      · simpa [Event.silent] using Step.env
      · simpa [Event.silent] using hk
  | defer hk_step hcd ih =>
    cases ho
    rename_i d a upd sem sc k k' m'' e''
    have hsh : ∀ {W' : Set Int} {s : AtomicStep}, W' ⊆ W → (s.d, s.a) ∈ C →
        e''.writes = Mem.span s.d s.a → ¬ e''.writes ⊆ W' := fun hW' hC hw =>
      shared_not_sub ((hdA.mono_right Set.subset_union_right).mono_right hW') hC hw
    -- the deferred step, taken from the agreeing memory
    have hdefer : ∀ {W' : Set Int}, W' ⊆ W → e''.reads ⊆ R ∪ e''.writes → e''.writes ⊆ W' →
        Step E m₀ (.rmw d a upd sem sc k)
          (.next (Mem.mix m₀ m'' e''.writes) (.rmw d a upd sem sc k') e'') := by
      intro W' hW' hR' hW''
      refine .defer (fun v => Step.transfer (hk_step v) fun x hx => (hag x ?_).symm) hcd
      rcases hx with hx | hx
      · rcases hR' hx with h | h
        · exact Or.inl h
        · exact Or.inr (hW' (hW'' h))
      · exact Or.inr (hW' (hW'' hx))
    cases h with
    | rmw hex hS hSR hconst hk =>
      obtain ⟨old, hr₀, n₀, hu₀⟩ := hex
      have hagw : ∀ n, ∀ x ∈ R ∪ (W \ Mem.span d a), m x = (m₀.write d a n) x := by
        intro n x hx
        have hxS : x ∉ Mem.span d a := by
          rcases hx with hx | hx
          · exact fun h => Set.disjoint_left.mp hSR h hx
          · exact hx.2
        rw [Mem.write_frame _ _ _ _ hxS]
        exact hag x (hx.elim Or.inl (fun h => Or.inr h.1))
      have hdA' : Disjoint (cellBytes C) (R ∪ (W \ Mem.span d a)) :=
        hdA.mono_right (Set.union_subset_union_right _ Set.sdiff_subset)
      have hsh' : ∀ {s : AtomicStep}, (s.d, s.a) ∈ C → e''.writes = Mem.span s.d s.a →
          ¬ e''.writes ⊆ W \ Mem.span d a := fun hC hw => hsh Set.sdiff_subset hC hw
      rcases ih old (hk old n₀ hr₀ hu₀) hdA' (hagw n₀) rfl with
        ⟨hR', hW', -, -⟩ | ⟨s, hsC, hsc, hsr, hsw, hsre, hswe, hsg, -⟩
      · -- a private step goes first
        have hWS : Disjoint (Mem.span d a) e''.writes :=
          Set.disjoint_left.mpr fun x hx hw => (hW' hw).2 hx
        refine Or.inl ⟨hR', fun x hx => (hW' hx).1, hdefer le_rfl hR' (fun x hx => (hW' hx).1), ?_⟩
        have hread : (Mem.mix m₀ m'' e''.writes).read d a = some old := by
          rw [← hr₀]; exact Mem.read_congr fun x hx => Mem.mix_out (Set.disjoint_left.mp hWS hx)
        refine .rmw ⟨old, hread, n₀, hu₀⟩ (fun x hx => ⟨hS hx, Set.disjoint_left.mp hWS hx⟩) hSR
          (hcd hconst) fun o n ho hn => ?_
        rw [hread] at ho
        cases ho
        rcases ih old (hk old n hr₀ hn) hdA' (hagw n) rfl with
          ⟨-, -, -, hsafe⟩ | ⟨s, hsC, -, -, -, -, hswe, -, -⟩
        · rw [Mem.mix_write n hWS, Set.sdiff_sdiff_comm] at hsafe
          exact hsafe
        · exact absurd (fun x hx => (hW' hx)) (hsh' hsC hswe)
      · -- a shared atomic goes first
        refine Or.inr ⟨s, hsC, hsc, hsr, hsw, hsre, hswe, hsg, ?_⟩
        refine .rmw ⟨old, hr₀, n₀, hu₀⟩ hS hSR (hcd hconst) fun o n ho hn => ?_
        rw [hr₀] at ho
        cases ho
        rcases ih old (hk old n hr₀ hn) hdA' (hagw n) rfl with
          ⟨-, hW', -, -⟩ | ⟨s', -, hsc', hsr', hsw', -, -, -, hsafe⟩
        · exact absurd hW' (hsh' hsC hswe)
        · rwa [AtomicStep.unique hsc' hsc hsr' hsr hsw' hsw] at hsafe
    | shared hal hC hsem hsc htot hconst hk =>
      obtain ⟨n₀, hu₀⟩ := htot default
      rcases ih default (hk default n₀ hu₀) hdA hag rfl with
        ⟨hR', hW', -, -⟩ | ⟨s, hsC, hsc', hsr, hsw, hsre, hswe, hsg, -⟩
      · -- a private step goes first
        refine Or.inl ⟨hR', hW', hdefer le_rfl hR' hW', ?_⟩
        refine .shared hal hC hsem hsc htot (hcd hconst) fun o n hn => ?_
        rcases ih o (hk o n hn) hdA hag rfl with
          ⟨-, -, -, hsafe⟩ | ⟨s, hsC, -, -, -, -, hswe, -, -⟩
        · exact hsafe
        · exact absurd hW' (hsh le_rfl hsC hswe)
      · -- another shared atomic goes first
        refine Or.inr ⟨s, hsC, hsc', hsr, hsw, hsre, hswe, hsg, ?_⟩
        refine .shared hal hC hsem hsc htot (hcd hconst) fun o n hn => ?_
        rcases ih o (hk o n hn) hdA hag rfl with
          ⟨-, hW', -, -⟩ | ⟨s', -, hsc'', hsr', hsw', -, -, -, hsafe⟩
        · exact absurd hW' (hsh le_rfl hsC hswe)
        · rw [AtomicStep.unique hsc'' hsc' hsr' hsr hsw' hsw] at hsafe
          have hp : List.Perm ((log ++ [⟨d, a, o, n⟩]) ++ [s]) ((log ++ [s]) ++ [⟨d, a, o, n⟩]) := by
            simpa only [List.append_assoc] using
              (List.perm_append_comm (l₁ := [⟨d, a, o, n⟩]) (l₂ := [s])).append_left log
          exact hsafe.perm hQ hp
  | loadUB => cases ho
  | storeUB => cases ho
  | rmwUB => cases ho
  | pickUB => cases ho
  | ub => cases ho

/-- A `SafeA` program does not step into undefined behaviour against a memory that agrees with
its own on the private footprint and in which the shared cells' bytes are present. -/
theorem no_ub {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {log : List AtomicStep} {m₀ m : Mem} {p : Prog α} (h : SafeA E R W C Q log m₀ p)
    (hag : ∀ x ∈ R ∪ W, m x = m₀ x) (hpres : ∀ x ∈ cellBytes C, m x ≠ none) :
    ¬ Step E m p .ub := by
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
    | shared hal hC hsem hsc htot _ _ =>
      obtain ⟨v, hv⟩ := Mem.read_some_of_writable (m := m)
        ⟨hal, fun i hi => hpres _ (span_sub_cellBytes hC (Mem.mem_span hi))⟩
      rcases hbad with h1 | ⟨old', hr', hno⟩
      · rw [hv] at h1; cases h1
      · obtain ⟨new, hu⟩ := htot old'; exact hno new hu
  | pickUB hno => cases h with | pick hex _ => obtain ⟨b, hb⟩ := hex; exact hno b hb
  | ub => cases h

end SafeA

end TVA
