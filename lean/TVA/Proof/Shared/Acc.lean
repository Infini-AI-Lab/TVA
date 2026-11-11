import TVA.Proof.Shared.SafeA

/-!
# `SafeA` programs end

A program that is `SafeA` cannot take infinitely many steps, whatever the other programs write
into the shared cells meanwhile (`SafeA.acc`). Its steps are seen in its own view of memory: the
launch's memory off the shared cells, its own memory on them (`SafeA.Next`). A shared atomic may
find any value, so the view does not record it; an atomic left pending (`Step.defer`) is
performed later and is counted then.

How this fits: `shared` (`TVA.Proof.Shared.Correct`) rules out a hang by this, as `independent`
does with `Safe.acc`.
-/
namespace TVA

namespace SafeA

variable {E : Env} {R : Set Int} {C : Set (DType × Int)}

/-- One step of a program in its own view: a step against a memory that agrees with its own off
the shared cells, whose cells may hold anything (other programs change them); afterwards its own
memory takes the step's writes off the cells. -/
def Next {α : Type} (E : Env) (C : Set (DType × Int)) : Mem × Prog α → Mem × Prog α → Prop
  | (m', p'), (m, p) => ∃ m₁ m₁' e, (∀ x, x ∉ cellBytes C → m₁ x = m x) ∧
      Step E m₁ p (.next m₁' p' e) ∧ m' = Mem.mix m₁' m (cellBytes C)

/-- A shared atomic pending while its continuation steps: the run is as long as the
continuation's, whatever old value the atomic finds. -/
private theorem acc_shared {α : Type} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop}
    {sem : MemSem} {sc : Scope} (hC : (d, a) ∈ C) :
    ∀ s : Mem × Prog α, Acc (Next E C) s → ∀ (m : Mem) (k : d.Val → Prog α), s = (m, k default) →
      (∀ old, Acc (Next E C) (m, k old)) → Acc (Next E C) (m, .rmw d a upd sem sc k) := by
  intro s hs
  induction hs with
  | intro s _ ih =>
    intro m k hs hall
    constructor
    rintro ⟨m', p'⟩ ⟨m₁, m₁', e, hag, hstep, rfl⟩
    cases hstep with
    | rmw old new hr hu =>
      have : Mem.mix (m₁.write d a new) m (cellBytes C) = m := by
        funext x
        by_cases hx : x ∈ cellBytes C
        · rw [Mem.mix_in hx]
        · rw [Mem.mix_out hx, Mem.write_frame _ _ _ _ (fun h => hx (span_sub_cellBytes hC h)),
            hag x hx]
      rw [this]; exact hall old
    | defer hk' _hcd =>
      rename_i k'
      have hnext : ∀ v, Next E C (Mem.mix m₁' m (cellBytes C), k' v) (m, k v) :=
        fun v => ⟨m₁, m₁', e, hag, hk' v, rfl⟩
      exact ih _ (hs ▸ hnext default) _ k' rfl fun v => (hall v).inv (hnext v)

/-- A private atomic pending while its continuation steps. -/
private theorem acc_private {α : Type} {Q : α → List AtomicStep → Mem → Prop} {d : DType}
    {a : Int} {upd : d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope} (hQ : PermInv Q) :
    ∀ s : Mem × Prog α, Acc (Next E C) s → ∀ (W : Set Int) (log : List AtomicStep) (m : Mem)
      (k : d.Val → Prog α) (old n₀ : d.Val), Disjoint (cellBytes C) (R ∪ W) →
      s = (m.write d a n₀, k old) → m.read d a = some old → upd old n₀ →
      Mem.span d a ⊆ W → Disjoint (Mem.span d a) R → (∀ v w, k v = k w) →
      (∀ old new, m.read d a = some old → upd old new →
        SafeA E R (W \ Mem.span d a) C Q log (m.write d a new) (k old)) →
      (∀ new, upd old new → Acc (Next E C) (m.write d a new, k old)) →
      Acc (Next E C) (m, .rmw d a upd sem sc k) := by
  intro s hs
  induction hs with
  | intro s _ ih =>
    intro W log m k old n₀ hdA hs hr hu hS hSR hconst hk hall
    have hSC : Disjoint (Mem.span d a) (cellBytes C) :=
      Set.disjoint_left.mpr fun x hx hc => Set.disjoint_left.mp hdA hc (Or.inr (hS hx))
    -- an atomic on these bytes is not shared
    have hnotC : (d, a) ∉ C := fun hC => by
      obtain ⟨x, hx⟩ := span_nonempty d a
      exact Set.disjoint_left.mp hSC hx (span_sub_cellBytes hC hx)
    have hdA' : Disjoint (cellBytes C) (R ∪ (W \ Mem.span d a)) :=
      hdA.mono_right (Set.union_subset_union_right _ Set.sdiff_subset)
    -- a shared step writes no private byte
    have hsh : ∀ {e : Event} {s : AtomicStep}, (s.d, s.a) ∈ C →
        e.writes = Mem.span s.d s.a → ¬ e.writes ⊆ W := by
      intro e s hC hw hsub
      obtain ⟨x, hx⟩ := span_nonempty s.d s.a
      rw [← hw] at hx
      exact Set.disjoint_left.mp hdA (span_sub_cellBytes hC (hw ▸ hx)) (Or.inr (hsub hx))
    constructor
    rintro ⟨m', p'⟩ ⟨m₁, m₁', e, hag, hstep, rfl⟩
    have hagS : ∀ x ∈ Mem.span d a, m₁ x = m x := fun x hx => hag x (Set.disjoint_left.mp hSC hx)
    have hagRW : ∀ x ∈ R ∪ W, m₁ x = m x := fun x hx =>
      hag x (fun hc => Set.disjoint_left.mp hdA hc hx)
    cases hstep with
    | rmw old' new' hr' hu' =>
      rw [Mem.read_congr hagS, hr] at hr'
      cases hr'
      have : Mem.mix (m₁.write d a new') m (cellBytes C) = m.write d a new' := by
        funext x
        by_cases hx : x ∈ cellBytes C
        · rw [Mem.mix_in hx, Mem.write_frame _ _ _ _ (fun h => Set.disjoint_left.mp hSC h hx)]
        · rw [Mem.mix_out hx]
          by_cases hxS : x ∈ Mem.span d a
          · exact Mem.write_span_eq _ _ _ _ _ hxS
          · rw [Mem.write_frame _ _ _ _ hxS, Mem.write_frame _ _ _ _ hxS, hag x hx]
      rw [this]; exact hall new' hu'
    | defer hk' hcd =>
      rename_i k'
      have hagw : ∀ n, ∀ x ∈ R ∪ (W \ Mem.span d a), m₁ x = (m.write d a n) x := by
        intro n x hx
        have hxS : x ∉ Mem.span d a := by
          rcases hx with hx | hx
          · exact fun h => Set.disjoint_left.mp hSR h hx
          · exact hx.2
        rw [Mem.write_frame _ _ _ _ hxS]
        exact hagRW x (hx.elim Or.inl (fun h => Or.inr h.1))
      have hrmw : SafeA E R W C Q log m (.rmw d a upd sem sc k) :=
        .rmw ⟨old, hr, n₀, hu⟩ hS hSR hconst hk
      rcases hrmw.stepT hQ hdA hagRW (.defer hk' hcd) with
        ⟨-, hW', -, hsafe'⟩ | ⟨s', hsC, -, -, hsw, hsre, hswe, -, hsafe'⟩
      · -- a private step went first
        have hWS : Disjoint (Mem.span d a) e.writes := by
          rcases (hk old n₀ hr hu).stepT hQ hdA' (hagw n₀) (hk' old) with
            ⟨-, hW'', -, -⟩ | ⟨s, hsC, -, -, -, -, hswe, -, -⟩
          · exact Set.disjoint_left.mpr fun x hx hw => (hW'' hw).2 hx
          · exact absurd hW' (hsh hsC hswe)
        have hm' : Mem.mix m₁' m (cellBytes C) = Mem.mix m m₁' e.writes := by
          funext x
          by_cases hx : x ∈ cellBytes C
          · rw [Mem.mix_in hx,
              Mem.mix_out (fun hw => Set.disjoint_left.mp hdA hx (Or.inr (hW' hw)))]
          · rw [Mem.mix_out hx]
            by_cases hxe : x ∈ e.writes
            · rw [Mem.mix_in hxe]
            · rw [Mem.mix_out hxe, Step.frame (hk' old) x hxe, hag x hx]
        have hcomm : ∀ n, upd old n →
            Next E C ((Mem.mix m m₁' e.writes).write d a n, k' old) (m.write d a n, k old) := by
          intro n hn
          rcases (hk old n hr hn).stepT hQ hdA' (hagw n) (hk' old) with
            ⟨-, -, hst, -⟩ | ⟨s, hsC, -, -, -, -, hswe, -, -⟩
          · refine ⟨_, _, e, fun _ _ => rfl, hst, ?_⟩
            rw [← Mem.mix_write n hWS]
            funext x
            by_cases hx : x ∈ cellBytes C
            · rw [Mem.mix_in hx,
                Mem.mix_out (fun hw => Set.disjoint_left.mp hdA hx (Or.inr (hW' hw)))]
            · rw [Mem.mix_out hx]
          · exact absurd hW' (hsh hsC hswe)
        have hr' : (Mem.mix m m₁' e.writes).read d a = some old := by
          rw [← hr]
          exact Mem.read_congr fun x hx => Mem.mix_out (Set.disjoint_left.mp hWS hx)
        rw [hm']
        cases hsafe' with
        | rmw _ hS' hSR' hc' hk'' =>
          exact ih _ (by rw [hs]; exact hcomm n₀ hu) (W \ e.writes) log _ k' old n₀
            (hdA.mono_right (Set.union_subset_union_right _ Set.sdiff_subset)) rfl hr' hu hS'
            hSR' hc' hk'' fun new hnew => (hall new hnew).inv (hcomm new hnew)
        | shared _ hC' => exact absurd hC' hnotC
      · -- a shared atomic went first: its own memory is unchanged
        have hm' : Mem.mix m₁' m (cellBytes C) = m := by
          funext x
          by_cases hx : x ∈ cellBytes C
          · rw [Mem.mix_in hx]
          · rw [Mem.mix_out hx, hsw,
              Mem.write_frame _ _ _ _ (fun h => hx (span_sub_cellBytes hsC h)), hag x hx]
        have hcomm : ∀ n, upd old n → Next E C (m.write d a n, k' old) (m.write d a n, k old) := by
          intro n hn
          refine ⟨Mem.mix (m.write d a n) m₁ (cellBytes C), _, e, fun x hx => Mem.mix_out hx,
            Step.transfer (hk' old) (fun x hx => ?_), ?_⟩
          · have hxc : x ∈ cellBytes C := by
              rcases hx with hx | hx
              · rw [hsre] at hx; exact span_sub_cellBytes hsC hx
              · rw [hswe] at hx; exact span_sub_cellBytes hsC hx
            rw [Mem.mix_in hxc]
          · funext x
            by_cases hx : x ∈ cellBytes C
            · rw [Mem.mix_in hx]
            · rw [Mem.mix_out hx,
                Mem.mix_out (by rw [hswe]; exact fun h => hx (span_sub_cellBytes hsC h)),
                Mem.mix_out hx]
        rw [hm']
        cases hsafe' with
        | rmw _ hS' hSR' hc' hk'' =>
          exact ih _ (by rw [hs]; exact hcomm n₀ hu) W (log ++ [s']) m k' old n₀ hdA rfl hr hu
            hS' hSR' hc' hk'' fun new hnew => (hall new hnew).inv (hcomm new hnew)
        | shared _ hC' => exact absurd hC' hnotC

/-- A step that writes nothing outside the cells leaves a program's own memory as it was. -/
private theorem mix_off {m₁ m : Mem} (hag : ∀ x, x ∉ cellBytes C → m₁ x = m x) :
    Mem.mix m₁ m (cellBytes C) = m := by
  funext x
  by_cases hx : x ∈ cellBytes C
  · rw [Mem.mix_in hx]
  · rw [Mem.mix_out hx, hag x hx]

/-- **Termination.** From a `SafeA` state, every sequence of the program's own steps is finite,
whatever the shared cells hold at each step. -/
theorem acc {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {log : List AtomicStep} {m : Mem} {p : Prog α} (h : SafeA E R W C Q log m p)
    (hQ : PermInv Q) (hdA : Disjoint (cellBytes C) (R ∪ W)) : Acc (Next E C) (m, p) := by
  induction h with
  | ret =>
    constructor; rintro ⟨_, _⟩ ⟨_, _, _, _, hs, -⟩; cases hs
  | load _ hR _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨m₁, m₁', e, hag, hs, rfl⟩
    cases hs with
    | load v hok =>
      have hagR : ∀ x ∈ accBytes _ _, m₁ x = _ := fun x hx =>
        hag x (fun hc => Set.disjoint_left.mp hdA hc (Or.inl (hR hx)))
      rw [mix_off hag]
      exact ih v ((loadOk_congr hagR).mp hok) hQ hdA
  | @store W' C' α' Q' log' m₀ d s addrs vals k _ hW _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨m₁, m₁', e, hag, hs, rfl⟩
    cases hs with
    | store order _ hperm =>
      have : Mem.mix (writeAll m₁ d addrs vals order) m₀ (cellBytes C') =
          writeAll m₀ d addrs vals order := by
        funext x
        by_cases hx : x ∈ cellBytes C'
        · rw [Mem.mix_in hx,
            writeAll_frame _ _ _ _ (fun h => Set.disjoint_left.mp hdA hx (Or.inr (hW h)))]
        · rw [Mem.mix_out hx]
          exact writeAll_agree (fun y hy => hag y hy) addrs vals order x hx
      rw [this]
      exact ih order hperm hQ (hdA.mono_right (Set.union_subset_union_right _ Set.sdiff_subset))
  | @rmw W' C' α' Q' log' m₀ d a upd sem sc k hex hS hSR hc hk ih =>
    obtain ⟨old, hr, n₀, hu⟩ := hex
    have hdA' : Disjoint (cellBytes C') (R ∪ (W' \ Mem.span d a)) :=
      hdA.mono_right (Set.union_subset_union_right R Set.sdiff_subset)
    exact acc_private hQ _ (ih old n₀ hr hu hQ hdA') _ _ _ _ old n₀ hdA rfl hr hu hS hSR hc hk
      fun new hnew => ih old new hr hnew hQ hdA'
  | shared hal hC hsem hsc htot _ _ ih =>
    obtain ⟨n₀, hu₀⟩ := htot default
    exact acc_shared hC _ (ih default n₀ hu₀ hQ hdA) _ _ rfl fun old => by
      obtain ⟨n, hu⟩ := htot old; exact ih old n hu hQ hdA
  | pick _ _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨m₁, m₁', e, hag, hs, rfl⟩
    cases hs with
    | pick b hb => rw [mix_off hag]; exact ih b hb hQ hdA
  | loop _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨m₁, m₁', e, hag, hs, rfl⟩
    cases hs with
    | loop => rw [mix_off hag]; exact ih hQ hdA
  | env _ ih =>
    constructor; rintro ⟨m', p'⟩ ⟨m₁, m₁', e, hag, hs, rfl⟩
    cases hs with
    | env => rw [mix_off hag]; exact ih hQ hdA

end SafeA

end TVA
