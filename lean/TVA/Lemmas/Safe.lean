import TVA.Lemmas.Mem

/-!
# `Safe`: what a prover proves about one program

`Safe E R W Q m p`: program `p`, run *alone* from memory `m` in environment `E`, cannot reach
undefined behaviour and always terminates, along every nondeterministic choice. It reads only
bytes in `R`, writes only bytes in `W`, and ends, returning `a`, in a memory `m'` with `Q a m'`.

It is total correctness in the weakest-precondition style: an inductive predicate, so every path
is finite even when a choice has infinitely many options. A proof follows the program text. Each
`tl.*` operation has a rule (`TVA.Lemmas.Ops`), and `Safe.bind` composes them.

`TVA.Lemmas.Independent` turns `Safe` for every program of a launch, with footprints that do
not conflict across programs, into the launch's outcome. That route covers most Triton kernels,
whose programs write disjoint tiles.
-/
namespace TVA

/-- One program, run alone, is safe with footprint `R`/`W` and postcondition `Q`. -/
inductive Safe (E : Env) (R W : Set Int) {α : Type} (Q : α → Mem → Prop) : Mem → Prog α → Prop where
  | ret {m : Mem} {a : α} : Q a m → Safe E R W Q m (.ret a)
  | load {m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {k : Tensor s d.Val → Prog α} :
      ¬ LoadBad m d addrs → accBytes d addrs ⊆ R →
      (∀ v, LoadOk m d addrs v → Safe E R W Q m (k v)) →
      Safe E R W Q m (.load d s addrs k)
  | store {m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int} {vals : Tensor s d.Val}
      {k : Prog α} :
      StoreOk m d addrs → accBytes d addrs ⊆ W →
      (∀ order, order.Perm (enabled addrs) → Safe E R W Q (writeAll m d addrs vals order) k) →
      Safe E R W Q m (.store d s addrs vals k)
  | rmw {m : Mem} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope}
      {k : d.Val → Prog α} :
      (∃ old, m.read d a = some old ∧ ∃ new, upd old new) →
      Mem.span d a ⊆ R → Mem.span d a ⊆ W →
      (∀ old new, m.read d a = some old → upd old new → Safe E R W Q (m.write d a new) (k old)) →
      Safe E R W Q m (.rmw d a upd sem sc k)
  | pick {m : Mem} {β : Type} {P : β → Prop} {k : β → Prog α} :
      (∃ b, P b) → (∀ b, P b → Safe E R W Q m (k b)) → Safe E R W Q m (.pick P k)
  | loop {m : Mem} {σ β : Type} {init : σ} {body : σ → Prog (σ ⊕ β)} {k : β → Prog α} :
      Safe E R W Q m (Prog.loopStep init body k) → Safe E R W Q m (.loop init body k)
  | env {m : Mem} {k : Env → Prog α} :
      Safe E R W Q m (k E) → Safe E R W Q m (.env k)

namespace Safe

variable {E : Env} {R W : Set Int}

theorem loopStep_bind {α β γ σ : Type} (init : σ) (body : σ → Prog (σ ⊕ β)) (k : β → Prog α)
    (f : α → Prog γ) :
    (Prog.loopStep init body k).bind f = Prog.loopStep init body (fun b => (k b).bind f) := by
  unfold Prog.loopStep
  rw [Prog.bind_assoc]
  congr 1
  funext r
  cases r <;> rfl

/-- **Sequencing.** To show `p >>= f` safe, show `p` safe with the postcondition "the rest is
safe". -/
theorem bind {α β : Type} {Q : β → Mem → Prop} {f : α → Prog β} {p : Prog α} {m : Mem}
    (h : Safe E R W (fun a m => Safe E R W Q m (f a)) m p) : Safe E R W Q m (p.bind f) := by
  induction h with
  | ret hq => exact hq
  | load hb hR _ ih => exact .load hb hR ih
  | store hok hW _ ih => exact .store hok hW ih
  | rmw hex hR hW _ ih => exact .rmw hex hR hW ih
  | pick hex _ ih => exact .pick hex ih
  | loop _ ih => exact .loop (by rw [← loopStep_bind]; exact ih)
  | env _ ih => exact .env ih

/-- `bind` in `do` notation. -/
theorem seq {α β : Type} {Q : β → Mem → Prop} {f : α → Prog β} {p : Prog α} {m : Mem}
    (h : Safe E R W (fun a m => Safe E R W Q m (f a)) m p) : Safe E R W Q m (p >>= f) := bind h

/-- A weaker postcondition and larger footprints. -/
theorem mono {α : Type} {Q Q' : α → Mem → Prop} {R' W' : Set Int} {m : Mem} {p : Prog α}
    (h : Safe E R W Q m p) (hQ : ∀ a m, Q a m → Q' a m) (hR : R ⊆ R') (hW : W ⊆ W') :
    Safe E R' W' Q' m p := by
  induction h with
  | ret hq => exact .ret (hQ _ _ hq)
  | load hb hRs _ ih => exact .load hb (hRs.trans hR) ih
  | store hok hWs _ ih => exact .store hok (hWs.trans hW) ih
  | rmw hex hRs hWs _ ih => exact .rmw hex (hRs.trans hR) (hWs.trans hW) ih
  | pick hex _ ih => exact .pick hex ih
  | loop _ ih => exact .loop ih
  | env _ ih => exact .env ih

/-! ## Safe programs step safely -/

/-- **Progress and preservation.** Every step of a safe program is a normal step, within the
footprint, to a safe continuation. -/
theorem step {α : Type} {Q : α → Mem → Prop} {m : Mem} {p : Prog α} (h : Safe E R W Q m p)
    {o : Out α} (hs : Step E m p o) :
    ∃ m' p' e, o = .next m' p' e ∧ e.reads ⊆ R ∧ e.writes ⊆ W ∧ Safe E R W Q m' p' := by
  cases hs with
  | load v hok => cases h with | load _ hR hk => exact ⟨_, _, _, rfl, hR, by simp, hk v hok⟩
  | loadUB hbad => cases h with | load hb _ _ => exact absurd hbad hb
  | store order hok hperm =>
    cases h with | store _ hW hk => exact ⟨_, _, _, rfl, by simp, hW, hk order hperm⟩
  | storeUB hbad => cases h with | store hok _ _ => exact absurd hok hbad
  | rmw old new hr hu => cases h with | rmw _ hR hW hk => exact ⟨_, _, _, rfl, hR, hW, hk old new hr hu⟩
  | rmwUB hbad =>
    cases h with
    | rmw hex _ _ _ =>
      obtain ⟨old, hr, new, hu⟩ := hex
      rcases hbad with h1 | ⟨old', hr', hno⟩
      · rw [hr] at h1; cases h1
      · rw [hr] at hr'; cases hr'; exact absurd hu (hno new)
  | pick b hb =>
    cases h with | pick _ hk => exact ⟨_, _, _, rfl, by simp [Event.silent], by simp [Event.silent], hk b hb⟩
  | pickUB hno => cases h with | pick hex _ => obtain ⟨b, hb⟩ := hex; exact absurd hb (hno b)
  | ub => cases h
  | loop => cases h with | loop hk => exact ⟨_, _, _, rfl, by simp [Event.silent], by simp [Event.silent], hk⟩
  | env => cases h with | env hk => exact ⟨_, _, _, rfl, by simp [Event.silent], by simp [Event.silent], hk⟩

/-- A safe program never steps to undefined behaviour. -/
theorem not_ub {α : Type} {Q : α → Mem → Prop} {m : Mem} {p : Prog α} (h : Safe E R W Q m p) :
    ¬ Step E m p .ub := by
  intro hs
  obtain ⟨_, _, _, h', _⟩ := h.step hs
  cases h'

/-- **Transfer.** A program that is safe in memory `m₂` steps the same way in any memory `m₁`
agreeing with `m₂` on its footprint. Afterwards the two still agree there, and `m₁` changed
only at the event's writes. -/
theorem transfer {α : Type} {Q : α → Mem → Prop} {m₁ m₂ : Mem} {p : Prog α} (h : Safe E R W Q m₂ p)
    (hag : ∀ x ∈ R ∪ W, m₁ x = m₂ x) {o : Out α} (hs : Step E m₁ p o) :
    ∃ m₁' m₂' p' e, o = .next m₁' p' e ∧ Step E m₂ p (.next m₂' p' e) ∧
      (∀ x ∈ R ∪ W, m₁' x = m₂' x) ∧ (∀ x, x ∉ e.writes → m₁' x = m₁ x) := by
  have agR : ∀ x ∈ R, m₁ x = m₂ x := fun x hx => hag x (Or.inl hx)
  have agW : ∀ x ∈ W, m₁ x = m₂ x := fun x hx => hag x (Or.inr hx)
  cases hs with
  | load v hok =>
    cases h with
    | load hb hR hk =>
      exact ⟨_, _, _, _, rfl, .load v ((loadOk_congr (fun x hx => agR x (hR hx))).mp hok), hag,
        fun _ _ => rfl⟩
  | loadUB hbad =>
    cases h with
    | load hb hR _ => exact absurd ((loadBad_congr (fun x hx => agR x (hR hx))).mp hbad) hb
  | store order hok hperm =>
    cases h with
    | store _ hW hk =>
      exact ⟨_, _, _, _, rfl, .store order ((storeOk_congr (fun x hx => agW x (hW hx))).mp hok) hperm,
        writeAll_agree hag _ _ _, fun x hx => writeAll_frame _ _ _ _ hx⟩
  | storeUB hbad =>
    cases h with
    | store hok hW _ => exact absurd ((storeOk_congr (fun x hx => agW x (hW hx))).mpr hok) hbad
  | rmw old new hr hu =>
    cases h with
    | rmw _ hR hW hk =>
      have hr2 : _ = some old := (Mem.read_congr (fun x hx => agR x (hR hx))).symm.trans hr
      exact ⟨_, _, _, _, rfl, .rmw old new hr2 hu, Mem.write_agree hag _ _ _,
        fun x hx => Mem.write_frame _ _ _ _ hx⟩
  | rmwUB hbad =>
    cases h with
    | rmw hex hR _ _ =>
      have hrd := Mem.read_congr (m₁ := m₁) (m₂ := m₂) (fun x hx => agR x (hR hx))
      obtain ⟨old, hr, new, hu⟩ := hex
      rcases hbad with h1 | ⟨old', hr', hno⟩
      · rw [hrd, hr] at h1; cases h1
      · rw [hrd, hr] at hr'; cases hr'; exact absurd hu (hno new)
  | pick b hb => exact ⟨_, _, _, _, rfl, .pick b hb, hag, fun _ _ => rfl⟩
  | pickUB hno => cases h with | pick hex _ => obtain ⟨b, hb⟩ := hex; exact absurd hb (hno b)
  | ub => cases h
  | loop => exact ⟨_, _, _, _, rfl, .loop, hag, fun _ _ => rfl⟩
  | env => exact ⟨_, _, _, _, rfl, .env, hag, fun _ _ => rfl⟩

/-! ## Safe programs terminate -/

/-- One step of a program alone, on its state (memory, program). -/
def Next {α : Type} (E : Env) : Mem × Prog α → Mem × Prog α → Prop
  | (m', p'), (m, p) => ∃ e, Step E m p (.next m' p' e)

/-- **Termination.** From a safe state, every sequence of steps is finite. -/
theorem acc {α : Type} {Q : α → Mem → Prop} {m : Mem} {p : Prog α} (h : Safe E R W Q m p) :
    Acc (Next E) (m, p) := by
  induction h with
  | ret => constructor; rintro ⟨_, _⟩ ⟨e, hs⟩; cases hs
  | load _ _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | load v hok => exact ih v hok
  | store _ _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | store order _ hperm => exact ih order hperm
  | rmw _ _ _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | rmw old new hr hu => exact ih old new hr hu
  | pick _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | pick b hb => exact ih b hb
  | loop _ ih => constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | loop => exact ih
  | env _ ih => constructor; rintro ⟨m', p'⟩ ⟨e, hs⟩; cases hs with | env => exact ih

end Safe

end TVA
