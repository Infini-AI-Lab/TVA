import TVA.Lemmas.Mem

/-!
# `Safe`: what a prover proves about one program

`Safe E R W Q m p`: program `p`, run *alone* from memory `m` in environment `E`, cannot reach
undefined behaviour and always terminates, along every nondeterministic choice, every order of
its atomics included (`Step.defer`). It loads only bytes in `R` and writes (stores or atomics)
only bytes in `W`, and it writes each byte at most once: after a store or an atomic, the rest of
the program has the write footprint `W` without those bytes. An atomic's bytes are outside `R`.
It ends, returning `a`, in a memory `m'` with `Q a m'`.

With `R` and `W` disjoint, which the launch theorems require, two accesses of the program touch
the same bytes only when both are loads, so the program has no conflict within itself
(`IntraRace`, library issue L15).

It is total correctness in the weakest-precondition style: an inductive predicate, so every path
is finite even when a choice has infinitely many options. A proof follows the program text. Each
`tl.*` operation has a rule (`TVA.Lemmas.Ops`), and `Safe.bind` composes them; a rule for a store
hands the rest of the program the smaller write footprint.

`TVA.Lemmas.Independent` turns `Safe` for every program of a launch, with footprints that do
not conflict across programs, into the launch's outcome. That route covers most Triton kernels,
whose programs write disjoint tiles.

The generic facts about one step (`Step.frame`, `Step.transfer`) are here too: a step changes
only the bytes it writes, and it can be taken the same way in any memory that agrees on the
bytes it reads and writes.
-/
namespace TVA

/-! ## Memory mixes and one step -/

open Classical in
/-- `m` on `S`, `m₀` elsewhere. -/
noncomputable def Mem.mix (m₀ m : Mem) (S : Set Int) : Mem := fun x => if x ∈ S then m x else m₀ x

@[simp] theorem Mem.mix_empty (m₀ m : Mem) : Mem.mix m₀ m ∅ = m₀ := by
  funext x; simp [Mem.mix]

theorem Mem.mix_of_frame {m₀ m : Mem} {S : Set Int} (h : ∀ x, x ∉ S → m x = m₀ x) :
    Mem.mix m₀ m S = m := by
  funext x
  by_cases hx : x ∈ S
  · simp [Mem.mix, hx]
  · simp [Mem.mix, hx, h x hx]

theorem Mem.mix_in {m₀ m : Mem} {S : Set Int} {x : Int} (hx : x ∈ S) : Mem.mix m₀ m S x = m x := by
  simp [Mem.mix, hx]

theorem Mem.mix_out {m₀ m : Mem} {S : Set Int} {x : Int} (hx : x ∉ S) : Mem.mix m₀ m S x = m₀ x := by
  simp [Mem.mix, hx]

/-- On its own bytes, a write does not depend on the memory written. -/
theorem Mem.write_span_eq (m m' : Mem) (d : DType) (a : Int) (v : d.Val) {x : Int}
    (hx : x ∈ Mem.span d a) : m.write d a v x = m'.write d a v x := by
  unfold Mem.write Mem.writeBytes
  rw [Mem.leBytes_len]
  simp only [Mem.span, Set.mem_ofPred_eq] at hx
  simp [hx]

theorem Mem.mix_write_self (m₀ m : Mem) (d : DType) (a : Int) (v : d.Val) :
    Mem.mix m₀ (m.write d a v) (Mem.span d a) = m₀.write d a v := by
  funext x
  by_cases hx : x ∈ Mem.span d a
  · rw [Mem.mix_in hx]; exact Mem.write_span_eq _ _ _ _ _ hx
  · rw [Mem.mix_out hx]; exact (Mem.write_frame _ _ _ _ hx).symm

/-- A write and a mix on disjoint bytes commute. -/
theorem Mem.mix_write {m₀ m' : Mem} {T : Set Int} {d : DType} {a : Int} (v : d.Val)
    (hT : Disjoint (Mem.span d a) T) :
    Mem.mix (m₀.write d a v) m' T = (Mem.mix m₀ m' T).write d a v := by
  funext x
  by_cases hx : x ∈ Mem.span d a
  · have hxT : x ∉ T := Set.disjoint_left.mp hT hx
    rw [Mem.mix_out hxT]; exact Mem.write_span_eq _ _ _ _ _ hx
  · rw [Mem.write_frame _ _ _ _ hx]
    by_cases hxT : x ∈ T
    · rw [Mem.mix_in hxT, Mem.mix_in hxT]
    · rw [Mem.mix_out hxT, Mem.mix_out hxT, Mem.write_frame _ _ _ _ hx]

theorem writeAll_mix {m₀ m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
    (vals : Tensor s d.Val) (order : List (Idx s)) (h : ∀ x ∈ accBytes d addrs, m₀ x = m x) :
    Mem.mix m₀ (writeAll m d addrs vals order) (accBytes d addrs) = writeAll m₀ d addrs vals order := by
  funext x
  by_cases hx : x ∈ accBytes d addrs
  · rw [Mem.mix_in hx]
    exact writeAll_agree (fun y hy => (h y hy).symm) addrs vals order x hx
  · rw [Mem.mix_out hx]; exact (writeAll_frame m₀ addrs vals order hx).symm

/-- A step changes only the bytes its event writes. -/
theorem Step.frame {α : Type} {E : Env} {m m' : Mem} {p p' : Prog α} {e : Event}
    (h : Step E m p (.next m' p' e)) : ∀ x, x ∉ e.writes → m' x = m x := by
  generalize ho : (Out.next m' p' e) = o at h
  induction h generalizing m' p' e with
  | store order _ _ => cases ho; exact fun x hx => writeAll_frame _ _ _ _ hx
  | rmw old new _ _ => cases ho; exact fun x hx => Mem.write_frame _ _ _ _ hx
  | load => cases ho; exact fun _ _ => rfl
  | pick => cases ho; exact fun _ _ => rfl
  | loop => cases ho; exact fun _ _ => rfl
  | env => cases ho; exact fun _ _ => rfl
  | defer _ ih => cases ho; exact ih default rfl
  | loadUB => cases ho
  | storeUB => cases ho
  | rmwUB => cases ho
  | pickUB => cases ho
  | ub => cases ho

/-- **Transfer of one step.** A step can be taken the same way in any memory that agrees on the
bytes its event reads and writes; the result is that memory with the step's writes. -/
theorem Step.transfer {α : Type} {E : Env} {m m₀ m' : Mem} {p p' : Prog α} {e : Event}
    (h : Step E m p (.next m' p' e)) (hag : ∀ x ∈ e.reads ∪ e.writes, m₀ x = m x) :
    Step E m₀ p (.next (Mem.mix m₀ m' e.writes) p' e) := by
  generalize ho : (Out.next m' p' e) = o at h
  induction h generalizing m' p' e with
  | load v hok =>
    cases ho
    simpa using Step.load v ((loadOk_congr (fun x hx => hag x (Or.inl hx))).mpr hok)
  | store order hok hperm =>
    cases ho
    have hag' : ∀ x ∈ accBytes _ _, m₀ x = _ := fun x hx => hag x (Or.inr hx)
    simp only
    rw [writeAll_mix _ _ hag']
    exact .store order ((storeOk_congr hag').mpr hok) hperm
  | rmw old new hr hu =>
    cases ho
    simp only
    rw [Mem.mix_write_self]
    exact .rmw old new ((Mem.read_congr fun x hx => hag x (Or.inl hx)).trans hr) hu
  | pick b hb => cases ho; simpa [Event.silent] using Step.pick b hb
  | loop => cases ho; simpa [Event.silent] using Step.loop
  | env => cases ho; simpa [Event.silent] using Step.env
  | defer _ ih => cases ho; exact .defer fun v => ih v hag rfl
  | loadUB => cases ho
  | storeUB => cases ho
  | rmwUB => cases ho
  | pickUB => cases ho
  | ub => cases ho

/-! ## The predicate -/

/-- One program, run alone, is safe with read footprint `R`, write footprint `W` (each byte
written at most once) and postcondition `Q`. -/
inductive Safe (E : Env) (R : Set Int) : Set Int → {α : Type} → (α → Mem → Prop) → Mem → Prog α → Prop where
  | ret {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {a : α} : Q a m → Safe E R W Q m (.ret a)
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
  /-- An atomic on bytes of `W` outside `R`; the rest of the program may not touch them again.
  The rule covers performing it now; `Step.defer`, performing it later, follows
  (`Safe.stepT`). -/
  | rmw {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {d : DType} {a : Int}
      {upd : d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope} {k : d.Val → Prog α} :
      (∃ old, m.read d a = some old ∧ ∃ new, upd old new) →
      Mem.span d a ⊆ W → Disjoint (Mem.span d a) R →
      (∀ old new, m.read d a = some old → upd old new →
        Safe E R (W \ Mem.span d a) Q (m.write d a new) (k old)) →
      Safe E R W Q m (.rmw d a upd sem sc k)
  | pick {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {β : Type} {P : β → Prop}
      {k : β → Prog α} :
      (∃ b, P b) → (∀ b, P b → Safe E R W Q m (k b)) → Safe E R W Q m (.pick P k)
  | loop {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {σ β : Type} {init : σ}
      {body : σ → Prog (σ ⊕ β)} {k : β → Prog α} :
      Safe E R W Q m (Prog.loopStep init body k) → Safe E R W Q m (.loop init body k)
  | env {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {k : Env → Prog α} :
      Safe E R W Q m (k E) → Safe E R W Q m (.env k)

namespace Safe

variable {E : Env} {R : Set Int}

theorem loopStep_bind {α β γ σ : Type} (init : σ) (body : σ → Prog (σ ⊕ β)) (k : β → Prog α)
    (f : α → Prog γ) :
    (Prog.loopStep init body k).bind f = Prog.loopStep init body (fun b => (k b).bind f) := by
  unfold Prog.loopStep
  rw [Prog.bind_assoc]
  congr 1
  funext r
  cases r <;> rfl

/-- A weaker postcondition and a larger write footprint. (The read footprint stays: an atomic's
bytes must be outside it.) -/
theorem mono {α : Type} {Q Q' : α → Mem → Prop} {W W' : Set Int} {m : Mem} {p : Prog α}
    (h : Safe E R W Q m p) (hQ : ∀ a m, Q a m → Q' a m) (hW : W ⊆ W') :
    Safe E R W' Q' m p := by
  induction h generalizing W' with
  | ret hq => exact .ret (hQ _ _ hq)
  | load hb hRs _ ih => exact .load hb hRs fun v hv => ih v hv hQ hW
  | store hok hWs _ ih =>
    exact .store hok (hWs.trans hW) fun o hp => ih o hp hQ (Set.sdiff_subset_sdiff_left hW)
  | rmw hex hS hSR _ ih =>
    exact .rmw hex (hS.trans hW) hSR fun o n ho hn => ih o n ho hn hQ (Set.sdiff_subset_sdiff_left hW)
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hQ hW
  | loop _ ih => exact .loop (ih hQ hW)
  | env _ ih => exact .env (ih hQ hW)

private theorem union_diff_of_sub {W₁ W₂ B : Set Int} (hd : Disjoint W₁ W₂) (hB : B ⊆ W₁) :
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

/-- **Sequencing.** To show `p >>= f` safe, show `p` safe with the postcondition "the rest is
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
  | rmw hex hS hSR _ ih =>
    refine .rmw hex (hS.trans Set.subset_union_left) hSR fun o n ho hn => ?_
    rw [union_diff_of_sub hd hS]
    exact ih o n ho hn (hd.mono_left Set.sdiff_subset) hQ'
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hd hQ'
  | loop _ ih => exact .loop (by rw [← loopStep_bind]; exact ih hd hQ')
  | env _ ih => exact .env (ih hd hQ')

/-- `bind` in `do` notation. -/
theorem seq {α β : Type} {Q : β → Mem → Prop} {f : α → Prog β} {p : Prog α} {m : Mem}
    {W₁ W₂ : Set Int} (h : Safe E R W₁ (fun a m => Safe E R W₂ Q m (f a)) m p)
    (hd : Disjoint W₁ W₂) : Safe E R (W₁ ∪ W₂) Q m (p >>= f) := bind h hd

/-- `bind` when the first part writes nothing. -/
theorem bind_ro {α β : Type} {Q : β → Mem → Prop} {f : α → Prog β} {p : Prog α} {m : Mem}
    {W : Set Int} (h : Safe E R ∅ (fun a m => Safe E R W Q m (f a)) m p) :
    Safe E R W Q m (p.bind f) := by
  simpa using bind h (Set.empty_disjoint W)

/-! ## Safe programs step safely -/

/-- **Steps of a safe program.** Run a program that is safe in its own memory `m₀` against a
memory `m` that agrees with `m₀` on the footprint. Every step it takes, an atomic performed
later (`Step.defer`) included, is a normal step that reads inside `R` or inside what it writes,
and writes inside `W`. The same step is possible from `m₀`, giving `m₀` with the step's writes,
and the rest is safe there with the written bytes removed from the write footprint. -/
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
    | rmw hex hS hSR hk =>
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
  | defer hk_step ih =>
    cases ho
    cases h with
    | rmw hex hS hSR hk =>
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
      · refine .defer fun v => Step.transfer (hk_step v) fun x hx => (hag x ?_).symm
        rcases hx with hx | hx
        · rcases hR' hx with h | h
          · exact Or.inl h
          · exact Or.inr (hW' h).1
        · exact Or.inr (hW' hx).1
      · have hread : (Mem.mix m₀ m'' e''.writes).read d a = some old := by
          rw [← hr₀]; exact Mem.read_congr fun x hx => Mem.mix_out (Set.disjoint_left.mp hWS hx)
        refine .rmw ⟨old, hread, n₀, hu₀⟩ (fun x hx => ⟨hS hx, Set.disjoint_left.mp hWS hx⟩) hSR ?_
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

/-- A safe program does not step into undefined behaviour, in any memory agreeing with its own
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
    | rmw hex hS _ _ =>
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

/-- A safe program never steps to undefined behaviour. -/
theorem not_ub {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {p : Prog α}
    (h : Safe E R W Q m p) : ¬ Step E m p .ub :=
  h.no_ub fun _ _ => rfl

/-! ## Safe programs terminate -/

/-- One step of a program alone, on its state (memory, program). -/
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
    | defer hk' =>
      rename_i k'
      cases hsafe with
      | rmw hex hS hSR hk =>
        obtain ⟨-, -, -, hsafe'⟩ := (Safe.rmw hex hS hSR hk).stepT (m := m) (fun _ _ => rfl) (.defer hk')
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

/-- **Termination.** From a safe state, every sequence of steps is finite. -/
theorem acc {W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {p : Prog α}
    (h : Safe E R W Q m p) : Acc (Next E) (m, p) := by
  induction h with
  | ret => constructor; rintro ⟨_, _⟩ ⟨e, hs⟩; cases hs
  | load _ _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | load v hok => exact ih v hok
  | store _ _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | store order _ hperm => exact ih order hperm
  | rmw hex hS hSR hk ih =>
    obtain ⟨old, hr, n₀, hu⟩ := hex
    exact acc_rmw _ (ih old n₀ hr hu) _ _ _ old n₀ rfl hr hu (.rmw ⟨old, hr, n₀, hu⟩ hS hSR hk)
      fun new hnew => ih old new hr hnew
  | pick _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | pick b hb => exact ih b hb
  | loop _ ih => constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | loop => exact ih
  | env _ ih => constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | env => exact ih

end Safe

end TVA
