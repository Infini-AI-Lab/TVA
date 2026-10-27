import TVA.Lemmas.Independent
import TVA.Lemmas.Buf
import TVA.Lemmas.Int

/-!
# Programs that share memory through atomics

Many Triton kernels have programs that touch common memory only with atomics:
- split-K GEMM accumulates partial tiles with `tl.atomic_add`;
- a histogram increments bins;
- a counter hands out work.

`independent` does not cover them, since their footprints overlap. **Theorem `shared`** does.

Memory is split into each program's **private** footprint (`R i`, `W i`, disjoint across
programs as for `independent`) and a **shared** region `A`, the bytes of a set of cells `C`
(`cellBytes C`). Cells are equal or disjoint. Programs access `A` only with `acq_rel`
read-modify-writes (Triton's default) at `gpu` or `sys` scope, each on a whole cell. Relaxed
shared atomics race in the semantics, and kernels with them are not covered.

Each program is proved alone with `SafeA`. It is `Safe` for private accesses. An atomic on `A`
may find *any* old value there, because other programs change `A`. The program records each
such atomic in a **log** of `AtomicStep`s (cell, update, old value, new value), and its
postcondition may depend on the log.

The theorem then gives the launch's outcome:
- it is well behaved and cannot hang;
- each program's private footprint holds what its solo run left;
- `A` holds the **replay** of all programs' logs, interleaved in some order, from the initial
  memory.

For commuting updates (integer `atomic_add`, `max`, `min`, bitwise), every interleaving
replays to the same value, so the result is determined.

Private segments of a program are proved with the ordinary `Safe` rules and lifted with
`SafeA.of_safe`. Only the shared atomics need `SafeA` rules (`safeA_atomic1`).
-/
namespace TVA

open Classical

/-- A memory whose element bytes are present at an aligned address can read it. -/
theorem Mem.read_some_of_writable {m : Mem} {d : DType} {a : Int} (hw : m.Writable d a) :
    ∃ v, m.read d a = some v := by
  unfold Mem.read
  rw [if_pos hw.1]
  have key : ∀ (n : Nat) (a : Int), (∀ k : Nat, k < n → m (a + k) ≠ none) →
      ∃ bs, m.readBytes a n = some bs := by
    intro n
    induction n with
    | zero => intro a _; exact ⟨[], rfl⟩
    | succ n ih =>
      intro a ha
      obtain ⟨b0, hb0⟩ := Option.ne_none_iff_exists'.mp (by simpa using ha 0 (by omega))
      obtain ⟨bs, hbs⟩ := ih (a + 1) (fun k hk => by
        have := ha (k + 1) (by omega); push_cast at this; rwa [show a + 1 + k = a + (k + 1) by ring])
      exact ⟨b0 :: bs, by simp [Mem.readBytes, hb0, hbs]⟩
  obtain ⟨bs, hbs⟩ := key _ _ hw.2
  exact ⟨_, by rw [hbs]; rfl⟩

/-! ## Logs and their replay -/

/-- One atomic read-modify-write on a shared cell, as it happened. -/
structure AtomicStep where
  d : DType
  a : Int
  upd : d.Val → d.Val → Prop
  old : d.Val
  new : d.Val
  ok : upd old new

/-- Replaying atomic steps from a memory: each finds its `old` value and writes its `new`. -/
inductive Replay : Mem → List AtomicStep → Mem → Prop where
  | nil (m : Mem) : Replay m [] m
  | cons {m m' : Mem} {s : AtomicStep} {rest : List AtomicStep} :
      m.read s.d s.a = some s.old → Replay (m.write s.d s.a s.new) rest m' → Replay m (s :: rest) m'

theorem Replay.snoc {m m' : Mem} {l : List AtomicStep} {s : AtomicStep} (h : Replay m l m')
    (hr : m'.read s.d s.a = some s.old) : Replay m (l ++ [s]) (m'.write s.d s.a s.new) := by
  induction h with
  | nil m => exact .cons hr (.nil _)
  | cons h0 _ ih => exact .cons h0 (ih hr)

/-- Program `i`'s part of an interleaved log. -/
def proj {N : Nat} (G : List (Fin N × AtomicStep)) (i : Fin N) : List AtomicStep :=
  (G.filter (fun g => decide (g.1 = i))).map Prod.snd

theorem proj_snoc_self {N : Nat} (G : List (Fin N × AtomicStep)) (j : Fin N) (s : AtomicStep) :
    proj (G ++ [(j, s)]) j = proj G j ++ [s] := by
  simp [proj, List.filter_append]

theorem proj_snoc_other {N : Nat} (G : List (Fin N × AtomicStep)) {i j : Fin N} (h : i ≠ j)
    (s : AtomicStep) : proj (G ++ [(j, s)]) i = proj G i := by
  simp [proj, List.filter_append, Ne.symm h]

/-! ## Shared cells -/

/-- The bytes of a set of cells (a dtype and an address each). -/
def cellBytes (C : Set (DType × Int)) : Set Int := {x | ∃ c ∈ C, x ∈ Mem.span c.1 c.2}

theorem span_sub_cellBytes {C : Set (DType × Int)} {d : DType} {a : Int} (h : (d, a) ∈ C) :
    Mem.span d a ⊆ cellBytes C := fun _ hx => ⟨(d, a), h, hx⟩

/-! ## One program -/

/-- One program, alone, with private footprint `R`/`W` and shared region `A`. Its postcondition
sees the log of its shared atomics. -/
inductive SafeA (E : Env) (R W : Set Int) (C : Set (DType × Int)) {α : Type} (Q : α → List AtomicStep → Mem → Prop) :
    List AtomicStep → Mem → Prog α → Prop where
  | ret {log : List AtomicStep} {m : Mem} {a : α} : Q a log m → SafeA E R W C Q log m (.ret a)
  | load {log : List AtomicStep} {m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {k : Tensor s d.Val → Prog α} :
      ¬ LoadBad m d addrs → accBytes d addrs ⊆ R →
      (∀ v, LoadOk m d addrs v → SafeA E R W C Q log m (k v)) →
      SafeA E R W C Q log m (.load d s addrs k)
  | store {log : List AtomicStep} {m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {vals : Tensor s d.Val} {k : Prog α} :
      StoreOk m d addrs → accBytes d addrs ⊆ W →
      (∀ order, order.Perm (enabled addrs) →
        SafeA E R W C Q log (writeAll m d addrs vals order) k) →
      SafeA E R W C Q log m (.store d s addrs vals k)
  /-- A private atomic, as in `Safe`. -/
  | rmw {log : List AtomicStep} {m : Mem} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop}
      {sem : MemSem} {sc : Scope} {k : d.Val → Prog α} :
      (∃ old, m.read d a = some old ∧ ∃ new, upd old new) →
      Mem.span d a ⊆ R → Mem.span d a ⊆ W →
      (∀ old new, m.read d a = some old → upd old new →
        SafeA E R W C Q log (m.write d a new) (k old)) →
      SafeA E R W C Q log m (.rmw d a upd sem sc k)
  /-- A shared atomic: any old value, a total update, recorded in the log. -/
  | shared {log : List AtomicStep} {m : Mem} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop}
      {sem : MemSem} {sc : Scope} {k : d.Val → Prog α} :
      Mem.Aligned d a → (d, a) ∈ C → sem = .acqRel → sc ≠ .cta → (∀ old, ∃ new, upd old new) →
      (∀ old new (h : upd old new),
        SafeA E R W C Q (log ++ [⟨d, a, upd, old, new, h⟩]) m (k old)) →
      SafeA E R W C Q log m (.rmw d a upd sem sc k)
  | pick {log : List AtomicStep} {m : Mem} {β : Type} {P : β → Prop} {k : β → Prog α} :
      (∃ b, P b) → (∀ b, P b → SafeA E R W C Q log m (k b)) → SafeA E R W C Q log m (.pick P k)
  | loop {log : List AtomicStep} {m : Mem} {σ β : Type} {init : σ} {body : σ → Prog (σ ⊕ β)}
      {k : β → Prog α} :
      SafeA E R W C Q log m (Prog.loopStep init body k) → SafeA E R W C Q log m (.loop init body k)
  | env {log : List AtomicStep} {m : Mem} {k : Env → Prog α} :
      SafeA E R W C Q log m (k E) → SafeA E R W C Q log m (.env k)

namespace SafeA

variable {E : Env} {R W : Set Int} {C : Set (DType × Int)}

/-- **Sequencing.** -/
theorem bind {α β : Type} {Q : β → List AtomicStep → Mem → Prop} {f : α → Prog β} {p : Prog α}
    {log : List AtomicStep} {m : Mem}
    (h : SafeA E R W C (fun a log m => SafeA E R W C Q log m (f a)) log m p) :
    SafeA E R W C Q log m (p.bind f) := by
  induction h with
  | ret hq => exact hq
  | load hb hR _ ih => exact .load hb hR ih
  | store hok hW _ ih => exact .store hok hW ih
  | rmw hex hR hW _ ih => exact .rmw hex hR hW ih
  | shared hal hC hsem hsc htot _ ih => exact .shared hal hC hsem hsc htot ih
  | pick hex _ ih => exact .pick hex ih
  | loop _ ih => exact .loop (by rw [← Safe.loopStep_bind]; exact ih)
  | env _ ih => exact .env ih

/-- A weaker postcondition. -/
theorem mono {α : Type} {Q Q' : α → List AtomicStep → Mem → Prop} {p : Prog α}
    {log : List AtomicStep} {m : Mem} (h : SafeA E R W C Q log m p)
    (hQ : ∀ a log m, Q a log m → Q' a log m) : SafeA E R W C Q' log m p := by
  induction h with
  | ret hq => exact .ret (hQ _ _ _ hq)
  | load hb hR _ ih => exact .load hb hR ih
  | store hok hW _ ih => exact .store hok hW ih
  | rmw hex hR hW _ ih => exact .rmw hex hR hW ih
  | shared hal hC hsem hsc htot _ ih => exact .shared hal hC hsem hsc htot ih
  | pick hex _ ih => exact .pick hex ih
  | loop _ ih => exact .loop ih
  | env _ ih => exact .env ih

/-- **Private segments.** A segment proved with the ordinary `Safe` rules, followed by the rest. -/
theorem of_safe {α β : Type} {Q : β → List AtomicStep → Mem → Prop} {f : α → Prog β} {p : Prog α}
    {log : List AtomicStep} {m : Mem}
    (h : Safe E R W (fun a m => SafeA E R W C Q log m (f a)) m p) :
    SafeA E R W C Q log m (p.bind f) := by
  induction h with
  | ret hq => exact hq
  | load hb hR _ ih => exact .load hb hR ih
  | store hok hW _ ih => exact .store hok hW ih
  | rmw hex hR hW _ ih => exact .rmw hex hR hW ih
  | pick hex _ ih => exact .pick hex ih
  | loop _ ih => exact .loop (by rw [← Safe.loopStep_bind]; exact ih)
  | env _ ih => exact .env ih

/-- A program in its shared-atomic state: the next step is an atomic on `A`. -/
def IsShared (C : Set (DType × Int)) {α : Type} (p : Prog α) : Prop :=
  ∃ (d : DType) (a : Int) (upd : d.Val → d.Val → Prop) (sem : MemSem) (sc : Scope) (k : d.Val → Prog α),
    p = .rmw d a upd sem sc k ∧ (d, a) ∈ C

theorem span_nonempty (d : DType) (a : Int) : (Mem.span d a).Nonempty :=
  ⟨a, by simp only [Mem.span, Set.mem_setOf_eq]; have := d.bytes_pos; omega⟩

/-- **Transfer.** Run against the launch's memory `m₁`, a program that is `SafeA` in its own
memory `m₂` takes either a private step (as `Safe.transfer`) or a shared atomic step, which
leaves its own memory alone and extends its log. -/
theorem transfer {α : Type} {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep}
    {m₁ m₂ : Mem} {p : Prog α} (h : SafeA E R W C Q log m₂ p) (hdA : Disjoint (cellBytes C) (R ∪ W))
    (hag : ∀ x ∈ R ∪ W, m₁ x = m₂ x) (hpres : ∀ x ∈ cellBytes C, m₁ x ≠ none) {o : Out α}
    (hs : Step E m₁ p o) :
    (∃ m₁' m₂' p' e, o = .next m₁' p' e ∧ Step E m₂ p (.next m₂' p' e) ∧ ¬ IsShared C p ∧
      e.reads ⊆ R ∧ e.writes ⊆ W ∧ (∀ x ∈ R ∪ W, m₁' x = m₂' x) ∧
      (∀ x, x ∉ e.writes → m₁' x = m₁ x) ∧ SafeA E R W C Q log m₂' p') ∨
    (∃ (s : AtomicStep) (p' : Prog α) (e : Event), o = .next (m₁.write s.d s.a s.new) p' e ∧
      m₁.read s.d s.a = some s.old ∧ e.reads = Mem.span s.d s.a ∧ e.writes = Mem.span s.d s.a ∧
      (s.d, s.a) ∈ C ∧ (∃ sc, e.atomic = some (.acqRel, sc) ∧ sc ≠ .cta) ∧
      SafeA E R W C Q (log ++ [s]) m₂ p' ∧
      ∃ (sem : MemSem) (sc : Scope) (k : s.d.Val → Prog α), p = .rmw s.d s.a s.upd sem sc k ∧
        p' = k s.old) := by
  have agR : ∀ x ∈ R, m₁ x = m₂ x := fun x hx => hag x (Or.inl hx)
  have agW : ∀ x ∈ W, m₁ x = m₂ x := fun x hx => hag x (Or.inr hx)
  -- a private atomic is not shared
  have notShared : ∀ {d : DType} {a : Int}, Mem.span d a ⊆ R → ∀ {upd sem sc} {k : d.Val → Prog α},
      ¬ IsShared C (Prog.rmw d a upd sem sc k) := by
    rintro d a hR upd sem sc k ⟨d', a', upd', sem', sc', k', heq, hA'⟩
    cases heq
    obtain ⟨x, hx⟩ := span_nonempty d a
    exact Set.disjoint_left.mp hdA (span_sub_cellBytes hA' hx) (Or.inl (hR hx))
  have noShared : ∀ {β : Type} {q : Prog α}, (∀ (d : DType) (a : Int) upd sem sc (k : d.Val → Prog α),
      q ≠ .rmw d a upd sem sc k) → ¬ IsShared C q := by
    rintro β q hq ⟨d, a, upd, sem, sc, k, heq, -⟩; exact hq d a upd sem sc k heq
  cases hs with
  | load v hok =>
    cases h with
    | load hb hR hk =>
      exact Or.inl ⟨_, _, _, _, rfl, .load v ((loadOk_congr (fun x hx => agR x (hR hx))).mp hok),
        noShared (β := Unit) (fun _ _ _ _ _ _ h => by cases h), hR, by simp, hag, fun _ _ => rfl,
        hk v ((loadOk_congr (fun x hx => agR x (hR hx))).mp hok)⟩
  | loadUB hbad =>
    cases h with
    | load hb hR _ => exact absurd ((loadBad_congr (fun x hx => agR x (hR hx))).mp hbad) hb
  | store order hok hperm =>
    cases h with
    | store _ hW hk =>
      exact Or.inl ⟨_, _, _, _, rfl,
        .store order ((storeOk_congr (fun x hx => agW x (hW hx))).mp hok) hperm,
        noShared (β := Unit) (fun _ _ _ _ _ _ h => by cases h), by simp, hW,
        writeAll_agree hag _ _ _, fun x hx => writeAll_frame _ _ _ _ hx, hk order hperm⟩
  | storeUB hbad =>
    cases h with
    | store hok hW _ => exact absurd ((storeOk_congr (fun x hx => agW x (hW hx))).mpr hok) hbad
  | rmw old new hr hu =>
    cases h with
    | rmw _ hR hW hk =>
      have hr2 : _ = some old := (Mem.read_congr (fun x hx => agR x (hR hx))).symm.trans hr
      exact Or.inl ⟨_, _, _, _, rfl, .rmw old new hr2 hu, notShared hR, hR, hW,
        Mem.write_agree hag _ _ _, fun x hx => Mem.write_frame _ _ _ _ hx, hk old new hr2 hu⟩
    | shared hal hC hsem hsc htot hk =>
      rename_i d a upd sem sc k
      subst hsem
      exact Or.inr ⟨⟨d, a, upd, old, new, hu⟩, _, _, rfl, hr, rfl, rfl, hC,
        ⟨sc, rfl, hsc⟩, hk old new hu, _, sc, k, rfl, rfl⟩
  | rmwUB hbad =>
    cases h with
    | rmw hex hR _ _ =>
      have hrd := Mem.read_congr (m₁ := m₁) (m₂ := m₂) (fun x hx => agR x (hR hx))
      obtain ⟨old, hr, new, hu⟩ := hex
      rcases hbad with h1 | ⟨old', hr', hno⟩
      · rw [hrd, hr] at h1; cases h1
      · rw [hrd, hr] at hr'; cases hr'; exact absurd hu (hno new)
    | shared hal hC hsem hsc htot _ =>
      obtain ⟨v, hv⟩ := Mem.read_some_of_writable (m := m₁)
        ⟨hal, fun i hi => hpres _ (span_sub_cellBytes hC (Mem.mem_span hi))⟩
      rcases hbad with h1 | ⟨old', hr', hno⟩
      · rw [hv] at h1; cases h1
      · obtain ⟨new, hu⟩ := htot old'; exact absurd hu (hno new)
  | pick b hb =>
    cases h with
    | pick _ hk =>
      exact Or.inl ⟨_, _, _, _, rfl, .pick b hb, noShared (β := Unit) (fun _ _ _ _ _ _ h => by cases h),
        by simp [Event.silent], by simp [Event.silent], hag, fun _ _ => rfl, hk b hb⟩
  | pickUB hno => cases h with | pick hex _ => obtain ⟨b, hb⟩ := hex; exact absurd hb (hno b)
  | ub => cases h
  | loop =>
    cases h with
    | loop hk =>
      exact Or.inl ⟨_, _, _, _, rfl, .loop, noShared (β := Unit) (fun _ _ _ _ _ _ h => by cases h),
        by simp [Event.silent], by simp [Event.silent], hag, fun _ _ => rfl, hk⟩
  | env =>
    cases h with
    | env hk =>
      exact Or.inl ⟨_, _, _, _, rfl, .env, noShared (β := Unit) (fun _ _ _ _ _ _ h => by cases h),
        by simp [Event.silent], by simp [Event.silent], hag, fun _ _ => rfl, hk⟩

/-- One step of a program in its own view: a private step, or a shared atomic that leaves its
memory alone and returns any old value. -/
def Next {α : Type} (E : Env) (C : Set (DType × Int)) : Mem × Prog α → Mem × Prog α → Prop
  | (m', p'), (m, p) => ((∃ e, Step E m p (.next m' p' e)) ∧ ¬ IsShared C p) ∨
      (m' = m ∧ ∃ (d : DType) (a : Int) (upd : d.Val → d.Val → Prop) (sem : MemSem) (sc : Scope)
        (k : d.Val → Prog α) (old : d.Val),
        p = .rmw d a upd sem sc k ∧ (d, a) ∈ C ∧ p' = k old)

/-- **Termination.** -/
theorem acc {α : Type} {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep} {m : Mem}
    {p : Prog α} (h : SafeA E R W C Q log m p) (hdA : Disjoint (cellBytes C) (R ∪ W)) : Acc (Next E C) (m, p) := by
  induction h with
  | ret =>
    constructor; rintro ⟨_, _⟩ (⟨⟨e, hs⟩, -⟩ | ⟨-, _, _, _, _, _, _, _, h, -⟩)
    · cases hs
    · cases h
  | load _ _ _ ih =>
    constructor; rintro ⟨m', p'⟩ (⟨⟨e, hs⟩, -⟩ | ⟨-, _, _, _, _, _, _, _, h, -⟩)
    · cases hs with | load v hok => exact ih v hok
    · cases h
  | store _ _ _ ih =>
    constructor; rintro ⟨m', p'⟩ (⟨⟨e, hs⟩, -⟩ | ⟨-, _, _, _, _, _, _, _, h, -⟩)
    · cases hs with | store order _ hperm => exact ih order hperm
    · cases h
  | rmw _ hR _ _ ih =>
    constructor; rintro ⟨m', p'⟩ (⟨⟨e, hs⟩, -⟩ | ⟨-, d, a, upd, sem, sc, k, old, h, hA', -⟩)
    · cases hs with | rmw old new hr hu => exact ih old new hr hu
    · cases h
      exact absurd (Or.inl (hR (span_nonempty _ _).some_mem))
        (Set.disjoint_left.mp hdA (span_sub_cellBytes hA' (span_nonempty _ _).some_mem))
  | shared hal hC hsem hsc htot _ ih =>
    constructor; rintro ⟨m', p'⟩ (⟨-, hns⟩ | ⟨rfl, d', a', upd', sem', sc', k', old, h, -, rfl⟩)
    · exact absurd ⟨_, _, _, _, _, _, rfl, hC⟩ hns
    · cases h
      obtain ⟨new, hu⟩ := htot old
      exact ih old new hu
  | pick _ _ ih =>
    constructor; rintro ⟨m', p'⟩ (⟨⟨e, hs⟩, -⟩ | ⟨-, _, _, _, _, _, _, _, h, -⟩)
    · cases hs with | pick b hb => exact ih b hb
    · cases h
  | loop _ ih =>
    constructor; rintro ⟨m', p'⟩ (⟨⟨e, hs⟩, -⟩ | ⟨-, _, _, _, _, _, _, _, h, -⟩)
    · cases hs with | loop => exact ih
    · cases h
  | env _ ih =>
    constructor; rintro ⟨m', p'⟩ (⟨⟨e, hs⟩, -⟩ | ⟨-, _, _, _, _, _, _, _, h, -⟩)
    · cases hs with | env => exact ih
    · cases h

/-- **A scalar atomic on the shared region** (`tl.atomic_*(ptr, v)` with a scalar pointer and
no mask). Any old value may be found; the step is logged. -/
theorem atomic1 {α : Type} {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep}
    {m : Mem} {d : DType} {ptrs : Tensor [] (Ptr d)} {upd : Idx [] → d.Val → d.Val → Prop}
    {sem : MemSem} {sc : Scope} {k : Tensor [] d.Val → Prog α}
    (hal : Mem.Aligned d (ptrs.get ()).addr) (hC : (d, (ptrs.get ()).addr) ∈ C)
    (hsem : sem = .acqRel) (hsc : sc ≠ .cta) (htot : ∀ old, ∃ new, upd () old new)
    (hk : ∀ old new (h : upd () old new),
      SafeA E R W C Q (log ++ [⟨d, (ptrs.get ()).addr, upd (), old, new, h⟩]) m (k (Tensor.scalar old))) :
    SafeA E R W C Q log m (tl.atomicWith ptrs upd none sem sc >>= k) := by
  simp only [tl.atomicWith, Prog.bind_eq, Prog.bind]
  have hen : (Idx.all []).filter (fun i => (tl.addrs ptrs none i).isSome) = [()] := by
    simp [Idx.all, tl.addrs] <;> rfl
  refine .pick ⟨fun _ => default, trivial⟩ fun junk _ => ?_
  rw [hen]
  refine .pick ⟨[()], List.Perm.refl _⟩ fun order horder => ?_
  rw [List.perm_singleton.mp horder]
  simp only [tl.rmwSeq, Prog.bind]
  refine .shared hal hC hsem hsc htot fun old new h => ?_
  simp only [pure, Prog.bind]
  convert hk old new h using 2
  ext ⟨⟩; exact Function.update_self (f := junk) () old

end SafeA

/-! ## The launch -/

section
variable {N : Nat} (E : Env) (R W : Fin N → Set Int) (C : Set (DType × Int))
  (Q : Fin N → List AtomicStep → Mem → Prop) (init : Mem)

/-- The invariant of a launch whose programs share `A` through atomics. -/
structure InvA (tr : List (Fin N × Event)) (c : Config N) : Prop where
  safe : ∃ G : List (Fin N × AtomicStep),
    (∀ i, SafeA E (R i) (W i) C (fun _ log m => Q i log m) (proj G i) (localMem R W init c i)
      (c.progs i)) ∧
    ∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ cellBytes C, c.mem x = mA x
  frame : ∀ x, x ∉ cellBytes C → (∀ i, x ∉ W i) → c.mem x = init x
  present : ∀ x ∈ cellBytes C, c.mem x ≠ none
  good : ∀ (n : Nat) (j : Fin N) (e : Event), tr[n]? = some (j, e) →
    (e.reads ⊆ R j ∧ e.writes ⊆ W j) ∨
      ((∃ c ∈ C, e.reads = Mem.span c.1 c.2 ∧ e.writes = Mem.span c.1 c.2) ∧
        ∃ sc, e.atomic = some (.acqRel, sc) ∧ sc ≠ .cta)

variable {E R W C Q init}

theorem good_snoc {tr : List (Fin N × Event)} {P : Fin N → Event → Prop}
    (hold : ∀ (n : Nat) (j : Fin N) (e : Event), tr[n]? = some (j, e) → P j e) {j : Fin N} {e : Event}
    (hnew : P j e) : ∀ (n : Nat) (i : Fin N) (e₀ : Event), (tr ++ [(j, e)])[n]? = some (i, e₀) → P i e₀ := by
  intro n i e₀ hn
  by_cases hlt : n < tr.length
  · rw [List.getElem?_append_left hlt] at hn; exact hold n i e₀ hn
  · have hn' : n = tr.length := by
      rcases Nat.lt_or_ge n (tr.length + 1) with h | h
      · omega
      · rw [List.getElem?_eq_none (by simp; omega)] at hn; cases hn
    subst hn'
    rw [List.getElem?_append_right le_rfl] at hn
    simp at hn
    obtain ⟨rfl, rfl⟩ := hn
    exact hnew

theorem invA_step (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i)) {tr : List (Fin N × Event)} {c c' : Config N} {j : Fin N}
    {e : Event} (hinv : InvA E R W C Q init tr c) (hs : LStep E c (j, e) c') :
    InvA E R W C Q init (tr ++ [(j, e)]) c' ∧
    SafeA.Next E C (localMem R W init c' j, c'.progs j) (localMem R W init c j, c.progs j) ∧
    ∀ i, i ≠ j → localMem R W init c' i = localMem R W init c i ∧ c'.progs i = c.progs i := by
  cases hs with
  | step _ hstep =>
    rename_i m' p'
    obtain ⟨G, hsafeG, mA, hrep, hagA⟩ := hinv.safe
    have hloc_ag : ∀ x ∈ R j ∪ W j, c.mem x = localMem R W init c j x := fun x hx => by
      simp [localMem, hx]
    rcases (hsafeG j).transfer (hA j) hloc_ag hinv.present hstep with
      ⟨m₁', m₂', p'', e', ho, hs2, hns, hR, hW, hag, hfr, hsafe'⟩ |
      ⟨st, p'', e', ho, hr, hre, hwe, hspA, hglob, hsafe', sem, sc, k, hp, hp'⟩
    · -- a private step
      simp only [Out.next.injEq] at ho
      obtain ⟨rfl, rfl, rfl⟩ := ho
      have hframe2 := Step.frame hs2
      have hloc : localMem R W init ⟨m', Function.update c.progs j p'⟩ j = m₂' := by
        funext x
        by_cases hx : x ∈ R j ∪ W j
        · simp only [localMem, hx, ite_true]; exact hag x hx
        · simp only [localMem, hx, ite_false]
          have hxw : x ∉ e.writes := fun hw => hx (Or.inr (hW hw))
          rw [hframe2 x hxw]; simp [localMem, hx]
      have hother : ∀ i, i ≠ j → localMem R W init ⟨m', Function.update c.progs j p'⟩ i =
          localMem R W init c i := by
        intro i hij
        funext x
        by_cases hx : x ∈ R i ∪ W i
        · simp only [localMem, hx, ite_true]
          exact hfr x fun hw => Set.disjoint_left.mp (hdisj j i hij.symm) (hW hw) hx
        · simp [localMem, hx]
      have hnotA : ∀ x ∈ cellBytes C, x ∉ e.writes := fun x hx hw =>
        Set.disjoint_left.mp (hA j) hx (Or.inr (hW hw))
      refine ⟨⟨⟨G, fun i => ?_, mA, hrep, fun x hx => ?_⟩, fun x hx hw => ?_, fun x hx => ?_,
        good_snoc hinv.good (Or.inl ⟨hR, hW⟩)⟩, ?_, ?_⟩
      · by_cases hij : i = j
        · subst hij; rw [hloc]; simpa using hsafe'
        · rw [hother i hij]; simpa [Function.update_of_ne hij] using hsafeG i
      · simp only; rw [hfr x (hnotA x hx)]; exact hagA x hx
      · simp only; rw [hfr x fun h => hw j (hW h)]; exact hinv.frame x hx hw
      · simp only; rw [hfr x (hnotA x hx)]; exact hinv.present x hx
      · refine Or.inl ⟨⟨e, ?_⟩, hns⟩
        simp only [Function.update_self]; rw [hloc]; exact hs2
      · intro i hij; exact ⟨hother i hij, Function.update_of_ne hij _ _⟩
    · -- a shared atomic
      simp only [Out.next.injEq] at ho
      obtain ⟨rfl, rfl, rfl⟩ := ho
      have hloc : ∀ i, localMem R W init ⟨c.mem.write st.d st.a st.new, Function.update c.progs j p'⟩ i =
          localMem R W init c i := by
        intro i
        funext x
        by_cases hx : x ∈ R i ∪ W i
        · simp only [localMem, hx, ite_true]
          exact Mem.write_frame _ _ _ _ fun hsp =>
            Set.disjoint_left.mp (hA i) (span_sub_cellBytes hspA hsp) hx
        · simp [localMem, hx]
      have hrA : mA.read st.d st.a = some st.old := by
        rw [← hr]; exact Mem.read_congr fun x hx => (hagA x (span_sub_cellBytes hspA hx)).symm
      refine ⟨⟨⟨G ++ [(j, st)], fun i => ?_, mA.write st.d st.a st.new, ?_,
        Mem.write_agree hagA _ _ _⟩, fun x hx hw => ?_, fun x hx => Mem.write_present _ (hinv.present x hx),
        good_snoc hinv.good (Or.inr ⟨⟨(st.d, st.a), hspA, hre, hwe⟩, hglob⟩)⟩, ?_, ?_⟩
      · rw [hloc i]
        by_cases hij : i = j
        · subst hij; rw [proj_snoc_self]; simpa using hsafe'
        · rw [proj_snoc_other _ hij]; simpa [Function.update_of_ne hij] using hsafeG i
      · rw [List.map_append]; exact hrep.snoc hrA
      · simp only
        rw [Mem.write_frame _ _ _ _ fun hsp => hx (span_sub_cellBytes hspA hsp)]
        exact hinv.frame x hx hw
      · refine Or.inr ⟨hloc j, st.d, st.a, st.upd, sem, sc, k, st.old, hp, hspA, ?_⟩
        simp [hp']
      · intro i hij; exact ⟨hloc i, Function.update_of_ne hij _ _⟩

theorem invA_reach (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i)) {c₀ c : Config N} {tr : List (Fin N × Event)}
    (h0 : InvA E R W C Q init [] c₀) (hr : Reach E c₀ tr c) : InvA E R W C Q init tr c := by
  induction hr with
  | refl => exact h0
  | @tail c₂ c₃ tr₁ ev _ hs ih =>
    obtain ⟨j, e⟩ := ev
    exact (invA_step hdisj hA ih hs).1

/-- No two programs race. Private events do not conflict across programs or with the shared
cells. Two shared events that conflict touch the same cell, since cells are equal or disjoint,
and both are `acq_rel` at `gpu`/`sys` scope, so they are `Event.Strong`. -/
theorem no_interRaceA (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i))
    (hcells : ∀ c ∈ C, ∀ c' ∈ C, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    {tr : List (Fin N × Event)} {c : Config N}
    (hinv : InvA E R W C Q init tr c) : ¬ InterRace tr := by
  rintro ⟨a, b, ⟨i, x⟩, ⟨j, y⟩, -, ha, hb, hij, hcon, hstrong, -⟩
  have hAi := hA i
  have hAj := hA j
  rcases hinv.good a i x ha with ⟨hxr, hxw⟩ | ⟨⟨cx, hcx, hxr, hxw⟩, hxg⟩ <;>
    rcases hinv.good b j y hb with ⟨hyr, hyw⟩ | ⟨⟨cy, hcy, hyr, hyw⟩, hyg⟩
  · rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
    · rcases hzy with hzy | hzy
      · exact Set.disjoint_left.mp (hdisj i j hij) (hxw hzx) (Or.inl (hyr hzy))
      · exact Set.disjoint_left.mp (hdisj i j hij) (hxw hzx) (Or.inr (hyw hzy))
    · exact Set.disjoint_left.mp (hdisj j i (Ne.symm hij)) (hyw hzy) (Or.inl (hxr hzx))
  · have hyA : ∀ z, z ∈ y.reads ∪ y.writes → z ∈ cellBytes C := by
      rintro z (hz | hz)
      · rw [hyr] at hz; exact span_sub_cellBytes (d := cy.1) (a := cy.2) hcy hz
      · rw [hyw] at hz; exact span_sub_cellBytes (d := cy.1) (a := cy.2) hcy hz
    rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
    · exact Set.disjoint_left.mp hAi (hyA z hzy) (Or.inr (hxw hzx))
    · exact Set.disjoint_left.mp hAi (hyA z (Or.inr hzy)) (Or.inl (hxr hzx))
  · have hxA : ∀ z, z ∈ x.reads ∪ x.writes → z ∈ cellBytes C := by
      rintro z (hz | hz)
      · rw [hxr] at hz; exact span_sub_cellBytes (d := cx.1) (a := cx.2) hcx hz
      · rw [hxw] at hz; exact span_sub_cellBytes (d := cx.1) (a := cx.2) hcx hz
    rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
    · rcases hzy with hzy | hzy
      · exact Set.disjoint_left.mp hAj (hxA z (Or.inr hzx)) (Or.inl (hyr hzy))
      · exact Set.disjoint_left.mp hAj (hxA z (Or.inr hzx)) (Or.inr (hyw hzy))
    · exact Set.disjoint_left.mp hAj (hxA z (Or.inl hzx)) (Or.inr (hyw hzy))
  · -- both shared: the same cell
    have hsame : cx = cy := by
      refine hcells cx hcx cy hcy ?_
      rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
      · rw [hxw] at hzx
        rcases hzy with hzy | hzy
        · rw [hyr] at hzy; exact ⟨z, hzx, hzy⟩
        · rw [hyw] at hzy; exact ⟨z, hzx, hzy⟩
      · rw [hxr] at hzx; rw [hyw] at hzy; exact ⟨z, hzx, hzy⟩
    subst hsame
    exact hstrong ⟨hxg, hyg, hxr.trans hyr.symm, hxw.trans hyw.symm⟩

end

/-- **Programs sharing memory through atomics.**

Suppose each program is `SafeA` alone from the initial memory, with private footprints that no
other program writes into. Suppose also that the shared cells `C` are equal or disjoint, are
disjoint from every private footprint, and are present initially. Then:
- the launch is well behaved and cannot hang;
- every outcome is a normal end;
- each program's private footprint holds a memory satisfying its postcondition for its own log;
- `A` holds the replay of all logs, interleaved in some order `G`;
- every other byte keeps its initial value. -/
theorem shared {k : Kernel} {L : Launch} {E : Env} {init : Mem}
    {R W : Fin L.size → Set Int} {C : Set (DType × Int)} {Q : Fin L.size → List AtomicStep → Mem → Prop}
    (hsafe : ∀ i, SafeA E (R i) (W i) C (fun _ log m => Q i log m) [] init (k.body (L.ctx i)))
    (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i))
    (hcells : ∀ c ∈ C, ∀ c' ∈ C, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    (hpres : ∀ x ∈ cellBytes C, init x ≠ none) :
    WellBehaved E k L init ∧
    ∀ o, Outcomes E k L init o → ∃ fin, o = .ok fin ∧
      ∃ G : List (Fin L.size × AtomicStep),
        (∀ i, ∃ m, Q i (proj G i) m ∧ ∀ x ∈ R i ∪ W i, fin x = m x) ∧
        (∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ cellBytes C, fin x = mA x) ∧
        (∀ x, x ∉ cellBytes C → (∀ i, x ∉ W i) → fin x = init x) := by
  have hloc0 : ∀ i, localMem R W init (Config.start k L init) i = init := by
    intro i; funext x; simp [localMem, Config.start]
  have h0 : InvA E R W C Q init [] (Config.start k L init) := by
    refine ⟨⟨[], fun i => ?_, init, .nil _, fun _ _ => rfl⟩, fun _ _ _ => rfl, hpres,
      fun n j e h => by simp at h⟩
    rw [hloc0]
    simpa [Config.start, proj] using hsafe i
  have hinv : ∀ tr c, Reach E (Config.start k L init) tr c → InvA E R W C Q init tr c :=
    fun tr c hr => invA_reach hdisj hA h0 hr
  have hnoub : ∀ tr c, Reach E (Config.start k L init) tr c → ¬ c.StuckUB E := by
    rintro tr c hr ⟨i, hub⟩
    have hI := hinv tr c hr
    obtain ⟨G, hsafeG, -⟩ := hI.safe
    rcases (hsafeG i).transfer (hA i) (fun x hx => by simp [localMem, hx]) hI.present hub with
      ⟨_, _, _, _, ho, -⟩ | ⟨_, _, _, ho, -⟩ <;> cases ho
  refine ⟨fun tr c hr => ⟨hnoub tr c hr, no_interRaceA hdisj hA hcells (hinv tr c hr)⟩, ?_⟩
  intro o ho
  cases o with
  | ok fin =>
    obtain ⟨tr, c, hr, hdone, hmem⟩ := ho
    have hI := hinv tr c hr
    obtain ⟨G, hsafeG, mA, hrep, hagA⟩ := hI.safe
    refine ⟨fin, rfl, G, fun i => ?_, ⟨mA, hrep, fun x hx => hmem ▸ hagA x hx⟩,
      fun x hx hw => hmem ▸ hI.frame x hx hw⟩
    have hs := hsafeG i
    rw [hdone i] at hs
    cases hs with
    | ret hq => exact ⟨_, hq, fun x hx => by simp [localMem, hx, hmem]⟩
  | ub =>
    obtain ⟨tr, c, hr, hub⟩ := ho
    exact absurd hub (hnoub tr c hr)
  | hang =>
    exfalso
    obtain ⟨f, hf0, hf⟩ := ho
    let ev : Nat → Fin L.size × Event := fun n => Classical.choose (hf n)
    have hev : ∀ n, LStep E (f n) (ev n) (f (n + 1)) := fun n => Classical.choose_spec (hf n)
    let tr : Nat → List (Fin L.size × Event) := fun n => (List.range n).map ev
    have htr : ∀ n, tr (n + 1) = tr n ++ [ev n] := by
      intro n; simp [tr, List.range_succ]
    have hreach : ∀ n, Reach E (Config.start k L init) (tr n) (f n) := by
      intro n
      induction n with
      | zero => rw [← hf0]; exact .refl _
      | succ n ih => rw [htr]; exact .tail ih (hev n)
    let S : Fin L.size → Nat → Mem × Prog Unit := fun i n =>
      (localMem R W init (f n) i, (f n).progs i)
    have hstepS : ∀ n, SafeA.Next E C (S (ev n).1 (n + 1)) (S (ev n).1 n) ∧
        ∀ i, i ≠ (ev n).1 → S i (n + 1) = S i n := by
      intro n
      obtain ⟨-, hnext, hothers⟩ := invA_step (j := (ev n).1) (e := (ev n).2) hdisj hA
        (hinv _ _ (hreach n)) (by simpa using hev n)
      refine ⟨hnext, fun i hij => ?_⟩
      obtain ⟨h1, h2⟩ := hothers i hij
      simp only [S, h1, h2]
    have hT : ∀ i, ∃ T, ∀ m, T ≤ m → ¬ SafeA.Next E C (S i (m + 1)) (S i m) := by
      intro i
      have hacc : Acc (SafeA.Next E C) (S i 0) := by
        obtain ⟨G, hsafeG, -⟩ := (hinv _ _ (hreach 0)).safe
        simpa [S] using (hsafeG i).acc (hA i)
      have hmono : ∀ n, S i (n + 1) = S i n ∨ SafeA.Next E C (S i (n + 1)) (S i n) := by
        intro n
        by_cases hij : i = (ev n).1
        · right; rw [hij]; exact (hstepS n).1
        · left; exact (hstepS n).2 i hij
      obtain ⟨T, -, hT⟩ := eventually_no_descent (r := SafeA.Next E C) (S i) hmono (S i 0) hacc 0 rfl
      exact ⟨T, hT⟩
    choose T hTi using hT
    let Tmax := Finset.univ.sup T
    have hle : T (ev Tmax).1 ≤ Tmax := Finset.le_sup (Finset.mem_univ _)
    exact hTi (ev Tmax).1 Tmax hle (hstepS Tmax).1

/-! ## Integer accumulation

When every shared atomic on an `int32` cell adds to it, the cell ends at its initial value plus
the sum of all increments, whatever the interleaving (`replay_add`). That sum is the sum over
programs of their own increments (`sum_proj`). -/

/-- A successful read means the element's bytes are present at an aligned address. -/
theorem Mem.writable_of_read {m : Mem} {d : DType} {a : Int} {v : d.Val} (h : m.read d a = some v) :
    m.Writable d a := by
  unfold Mem.read at h
  split_ifs at h with hal
  refine ⟨hal, ?_⟩
  have key : ∀ (n : Nat) (a : Int) (bs : List TVA.Math.Bytes.Byte), m.readBytes a n = some bs →
      ∀ k : Nat, k < n → m (a + k) ≠ none := by
    intro n
    induction n with
    | zero => intro _ _ _ k hk; omega
    | succ n ih =>
      intro a bs h k hk
      simp only [Mem.readBytes] at h
      cases ha : m a with
      | none => rw [ha] at h; simp at h
      | some b =>
        cases hr : m.readBytes (a + 1) n with
        | none => rw [ha, hr] at h; simp at h
        | some bs' =>
          rcases k with _ | k
          · simp [ha]
          · have := ih (a + 1) bs' hr k (by omega)
            rwa [show a + 1 + (k : Int) = a + ((k + 1 : Nat) : Int) by push_cast; ring] at this
  cases hb : m.readBytes a d.bytes with
  | none => rw [hb] at h; simp at h
  | some bs => exact key _ _ _ hb

/-- Step `s` adds `inc` to the `int32` cell at `a` (`inc = 0` for a step on another cell). -/
def AddsTo (a : Int) (inc : I32) (s : AtomicStep) : Prop :=
  ∀ (m : Mem) (x : I32), m.read .i32 a = some x → m.read s.d s.a = some s.old →
    (m.write s.d s.a s.new).read .i32 a = some (x + inc)

/-- An `int32` add of `v` at `a` adds `v` to that cell. -/
theorem addsTo_add {a : Int} {v : I32} {upd : I32 → I32 → Prop} {old new : I32} {h : upd old new}
    (hnew : new = old + v) : AddsTo a v ⟨.i32, a, upd, old, new, h⟩ := by
  intro m x hx hold
  simp only at hold
  rw [hx] at hold
  cases hold
  rw [Mem.read_write_self (Mem.writable_of_read hx)]
  exact congrArg some hnew

/-- A step on bytes disjoint from the cell adds nothing to it. -/
theorem addsTo_other {a : Int} {s : AtomicStep} (hdis : Disjoint (Mem.span s.d s.a) (Mem.span .i32 a)) :
    AddsTo a 0 s := by
  intro m x hx _
  rw [Mem.read_congr (fun y hy => Mem.write_frame _ _ _ _ (fun h => Set.disjoint_left.mp hdis h hy)),
    hx, add_zero]

/-- **Accumulation.** Replaying steps that each add to the cell leaves it at the initial value
plus the sum of the increments. -/
theorem replay_add {a : Int} {inc : AtomicStep → I32} {m m' : Mem} {l : List AtomicStep} {x : I32}
    (hrep : Replay m l m') (hl : ∀ s ∈ l, AddsTo a (inc s) s) (h0 : m.read .i32 a = some x) :
    m'.read .i32 a = some (x + (l.map inc).sum) := by
  induction hrep generalizing x with
  | nil => simp [h0]
  | @cons m₀ m₁ s rest hr _ ih =>
    have h1 := hl s (List.mem_cons_self) m₀ x h0 hr
    rw [ih (fun s' hs' => hl s' (List.mem_cons_of_mem _ hs')) h1]
    simp [add_assoc]

/-- The sum over an interleaved log is the sum over programs of their own parts. -/
theorem sum_proj {N : Nat} (G : List (Fin N × AtomicStep)) (f : AtomicStep → I32) :
    ((G.map Prod.snd).map f).sum = ∑ i, ((proj G i).map f).sum := by
  induction G with
  | nil => simp [proj]
  | cons g G ih =>
    obtain ⟨j, s⟩ := g
    have hsplit : ∀ i, ((proj ((j, s) :: G) i).map f).sum =
        (if j = i then f s else 0) + ((proj G i).map f).sum := by
      intro i; by_cases h : j = i <;> simp [proj, List.filter_cons, h]
    simp only [List.map_cons, List.sum_cons, ih, hsplit, Finset.sum_add_distrib, Finset.sum_ite_eq,
      Finset.mem_univ, ite_true]

end TVA
