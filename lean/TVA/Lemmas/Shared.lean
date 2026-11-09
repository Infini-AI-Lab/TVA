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
read-modify-writes (Triton's default) at `gpu` or `sys` scope, each on a whole cell. An atomic
whose result the source never uses is one of these whatever its `sem` (`tl.atomic_add_unused` and
the others, whose order is `MemSem.unused sem sc`, `acq_rel` at `gpu`/`sys` by `MemSem.unused_gpu`;
library issue L157). Relaxed shared atomics whose result is used race in the semantics, and
kernels with them are not covered.

Each program is proved alone with `SafeA`. It is `Safe` for private accesses. An atomic on `A`
may find *any* old value there, because other programs change `A`. The program records each
such atomic in a **log** of `AtomicStep`s (the cell, its old value and its new value; not the
operand, which `TVA.Lemmas.SharedAcc` pairs with each step), and its postcondition may depend on
the log.

The theorem then gives the launch's outcome:
- it is well behaved and cannot hang;
- each program's private footprint holds what its solo run left;
- `A` holds the **replay** of all programs' logs, interleaved in some order, from the initial
  memory.

For commuting updates every interleaving replays to the same value, so the result is
determined. The library proves it for integer `atomic_add` (`replay_add` here, any width
`replay_add_any`) and integer `atomic_max`/`atomic_min` (`replay_max`, `replay_min`), and bounds a
float `atomic_add` (`replay_err_shared`), in `TVA.Lemmas.SharedAcc`; the bitwise `atomic_and`,
`atomic_or` and `atomic_xor` commute too, but have no lemma yet.

Private segments of a program are proved with the ordinary `Safe` rules and lifted with
`SafeA.of_safe`. Only the shared atomics need `SafeA` rules: `SafeA.atomic1` (a scalar pointer, no
mask) and `SafeA.atomic` (`TVA.Lemmas.SharedAcc`: a tensor, with a mask). Both are about
`tl.atomicWith`, which every `tl.atomic_*` is except `tl.atomic_add`: that is
`.env fun e => tl.atomicWith ptrs (fun i => tl.addUpd e d (vals.get i)) mask sem sc`, so unfold it
and apply the constructor `SafeA.env` first.
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

/-! ## Logs and their replay -/

/-- One atomic read-modify-write on a shared cell, as it happened: the cell (dtype and address),
the old value it found and the new value it wrote. -/
structure AtomicStep where
  d : DType
  a : Int
  old : d.Val
  new : d.Val

/-- The step a launch step performed is determined by the step: its event names the cell, the
memory before gives the old value and the memory after the new one. -/
theorem AtomicStep.unique {m m' : Mem} {e : Event} {s s' : AtomicStep}
    (hc : e.cell = some (s.d, s.a)) (hc' : e.cell = some (s'.d, s'.a))
    (hr : m.read s.d s.a = some s.old) (hr' : m.read s'.d s'.a = some s'.old)
    (hw : m' = m.write s.d s.a s.new) (hw' : m' = m.write s'.d s'.a s'.new) : s = s' := by
  obtain ⟨d, a, o, n⟩ := s
  obtain ⟨d', a', o', n'⟩ := s'
  rw [hc] at hc'
  simp only [Option.some.injEq, Prod.mk.injEq] at hc'
  obtain ⟨rfl, rfl⟩ := hc'
  simp only at hr hr' hw hw'
  rw [hr] at hr'
  cases hr'
  have h1 := Mem.read_write_self (Mem.writable_of_read hr) n
  have h2 := Mem.read_write_self (Mem.writable_of_read hr) n'
  rw [← hw] at h1
  rw [← hw', h1] at h2
  cases h2
  rfl

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

theorem span_nonempty (d : DType) (a : Int) : (Mem.span d a).Nonempty :=
  ⟨a, by simp only [Mem.span, Set.mem_setOf_eq]; have := d.bytes_pos; omega⟩

/-- A postcondition that does not depend on the order of the log. A program's shared atomics
may be performed in any order (`Step.defer`), so the theorem `shared` needs this. -/
def PermInv {α : Type} (Q : α → List AtomicStep → Mem → Prop) : Prop :=
  ∀ a l l' m, l.Perm l' → Q a l m → Q a l' m

/-! ## One program -/

/-- One program, alone, with private footprint `R`/`W` (each private byte written at most once)
and shared region `A`. Its postcondition sees the log of its shared atomics. -/
inductive SafeA (E : Env) (R : Set Int) :
    Set Int → Set (DType × Int) → {α : Type} → (α → List AtomicStep → Mem → Prop) →
      List AtomicStep → Mem → Prog α → Prop where
  | ret {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {a : α} : Q a log m → SafeA E R W C Q log m (.ret a)
  | load {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {k : Tensor s d.Val → Prog α} :
      ¬ LoadBad m d addrs → accBytes d addrs ⊆ R →
      (∀ v, LoadOk m d addrs v → SafeA E R W C Q log m (k v)) →
      SafeA E R W C Q log m (.load d s addrs k)
  | store {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {d : DType} {s : Shape} {addrs : Idx s → Option Int}
      {vals : Tensor s d.Val} {k : Prog α} :
      StoreOk m d addrs → accBytes d addrs ⊆ W →
      (∀ order, order.Perm (enabled addrs) →
        SafeA E R (W \ accBytes d addrs) C Q log (writeAll m d addrs vals order) k) →
      SafeA E R W C Q log m (.store d s addrs vals k)
  /-- A private atomic, as in `Safe`. -/
  | rmw {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop}
      {sem : MemSem} {sc : Scope} {k : d.Val → Prog α} :
      (∃ old, m.read d a = some old ∧ ∃ new, upd old new) →
      Mem.span d a ⊆ W → Disjoint (Mem.span d a) R →
      (∀ old new, m.read d a = some old → upd old new →
        SafeA E R (W \ Mem.span d a) C Q log (m.write d a new) (k old)) →
      SafeA E R W C Q log m (.rmw d a upd sem sc k)
  /-- A shared atomic: any old value, a total update, recorded in the log. -/
  | shared {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop}
      {sem : MemSem} {sc : Scope} {k : d.Val → Prog α} :
      Mem.Aligned d a → (d, a) ∈ C → sem = .acqRel → sc ≠ .cta → (∀ old, ∃ new, upd old new) →
      (∀ old new, upd old new → SafeA E R W C Q (log ++ [⟨d, a, old, new⟩]) m (k old)) →
      SafeA E R W C Q log m (.rmw d a upd sem sc k)
  | pick {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {β : Type} {P : β → Prop} {k : β → Prog α} :
      (∃ b, P b) → (∀ b, P b → SafeA E R W C Q log m (k b)) → SafeA E R W C Q log m (.pick P k)
  | loop {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {σ β : Type} {init : σ} {body : σ → Prog (σ ⊕ β)}
      {k : β → Prog α} :
      SafeA E R W C Q log m (Prog.loopStep init body k) → SafeA E R W C Q log m (.loop init body k)
  | env {W : Set Int} {C : Set (DType × Int)} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
      {log : List AtomicStep} {m : Mem} {k : Env → Prog α} :
      SafeA E R W C Q log m (k E) → SafeA E R W C Q log m (.env k)

namespace SafeA

variable {E : Env} {R : Set Int} {C : Set (DType × Int)}

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

/-- A weaker postcondition and a larger write footprint. -/
theorem mono {α : Type} {Q Q' : α → List AtomicStep → Mem → Prop} {W W' : Set Int} {p : Prog α}
    {log : List AtomicStep} {m : Mem} (h : SafeA E R W C Q log m p)
    (hQ : ∀ a log m, Q a log m → Q' a log m) (hW : W ⊆ W') : SafeA E R W' C Q' log m p := by
  induction h generalizing W' with
  | ret hq => exact .ret (hQ _ _ _ hq)
  | load hb hR _ ih => exact .load hb hR fun v hv => ih v hv hQ hW
  | store hok hWs _ ih =>
    exact .store hok (hWs.trans hW) fun o hp => ih o hp hQ (Set.sdiff_subset_sdiff_left hW)
  | rmw hex hS hSR _ ih =>
    exact .rmw hex (hS.trans hW) hSR fun o n ho hn => ih o n ho hn hQ (Set.sdiff_subset_sdiff_left hW)
  | shared hal hC hsem hsc htot _ ih => exact .shared hal hC hsem hsc htot fun o n h => ih o n h hQ hW
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hQ hW
  | loop _ ih => exact .loop (ih hQ hW)
  | env _ ih => exact .env (ih hQ hW)

/-- **Sequencing.** The two parts write disjoint private footprints. -/
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
    rw [union_diff_of_sub hd hW]
    exact ih o hp (hd.mono_left Set.sdiff_subset) hQ'
  | rmw hex hS hSR _ ih =>
    refine .rmw hex (hS.trans Set.subset_union_left) hSR fun o n ho hn => ?_
    rw [union_diff_of_sub hd hS]
    exact ih o n ho hn (hd.mono_left Set.sdiff_subset) hQ'
  | shared hal hC hsem hsc htot _ ih => exact .shared hal hC hsem hsc htot fun o n h => ih o n h hd hQ'
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hd hQ'
  | loop _ ih => exact .loop (by rw [← Safe.loopStep_bind]; exact ih hd hQ')
  | env _ ih => exact .env (ih hd hQ')

/-- **Private segments.** A segment proved with the ordinary `Safe` rules, followed by the rest,
which writes a disjoint private footprint. -/
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
    rw [union_diff_of_sub hd hW]
    exact ih o hp (hd.mono_left Set.sdiff_subset) hQ'
  | rmw hex hS hSR _ ih =>
    refine .rmw hex (hS.trans Set.subset_union_left) hSR fun o n ho hn => ?_
    rw [union_diff_of_sub hd hS]
    exact ih o n ho hn (hd.mono_left Set.sdiff_subset) hQ'
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hd hQ'
  | loop _ ih => exact .loop (by rw [← Safe.loopStep_bind]; exact ih hd hQ')
  | env _ ih => exact .env (ih hd hQ')

/-- With a postcondition that ignores the log's order, the log may be permuted. -/
theorem perm {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop} {log log' : List AtomicStep}
    {m : Mem} {p : Prog α} (hQ : PermInv Q) (h : SafeA E R W C Q log m p) (hp : log.Perm log') :
    SafeA E R W C Q log' m p := by
  induction h generalizing log' with
  | ret hq => exact .ret (hQ _ _ _ _ hp hq)
  | load hb hR _ ih => exact .load hb hR fun v hv => ih v hv hQ hp
  | store hok hW _ ih => exact .store hok hW fun o ho => ih o ho hQ hp
  | rmw hex hS hSR _ ih => exact .rmw hex hS hSR fun o n ho hn => ih o n ho hn hQ hp
  | shared hal hC hsem hsc htot _ ih =>
    exact .shared hal hC hsem hsc htot fun o n h => ih o n h hQ (hp.append_right _)
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hQ hp
  | loop _ ih => exact .loop (ih hQ hp)
  | env _ ih => exact .env (ih hQ hp)

/-! ## Steps of a program with shared atomics -/

/-- **Steps.** Run against the launch's memory `m`, a program that is `SafeA` in its own memory
`m₀` (agreeing with `m` on the private footprint) takes either a private step (as
`Safe.stepT`) or a shared atomic step, which leaves its own memory alone and extends its log.
Atomics performed later (`Step.defer`) are covered, with the log reordered (`PermInv`). -/
theorem stepT {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep}
    {m₀ m : Mem} {p : Prog α} (h : SafeA E R W C Q log m₀ p) (hQ : PermInv Q)
    (hdA : Disjoint (cellBytes C) (R ∪ W)) (hag : ∀ x ∈ R ∪ W, m x = m₀ x)
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
    | rmw hex hS hSR hk =>
      have hr₀ := (Mem.read_congr fun x hx => (hag x (Or.inr (hS hx))).symm).trans hr
      refine Or.inl ⟨fun x hx => Or.inr hx, hS, ?_, ?_⟩
      · simp only; rw [Mem.mix_write_self]; exact .rmw old new hr₀ hu
      · simp only; rw [Mem.mix_write_self]; exact hk old new hr₀ hu
    | shared hal hC hsem hsc htot hk =>
      rename_i d a upd sem sc k
      subst hsem
      exact Or.inr ⟨⟨d, a, old, new⟩, hC, rfl, hr, rfl, rfl, rfl, ⟨sc, rfl, hsc⟩, hk old new hu⟩
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
  | defer hk_step ih =>
    cases ho
    rename_i d a upd sem sc k k' m'' e''
    -- a shared step writes no private byte
    have hsh : ∀ {W' : Set Int} {s : AtomicStep}, W' ⊆ W → (s.d, s.a) ∈ C →
        e''.writes = Mem.span s.d s.a → ¬ e''.writes ⊆ W' := by
      intro W' s hW' hC hw hsub
      obtain ⟨x, hx⟩ := span_nonempty s.d s.a
      rw [← hw] at hx
      exact Set.disjoint_left.mp hdA (span_sub_cellBytes hC (hw ▸ hx)) (Or.inr (hW' (hsub hx)))
    -- the deferred step, taken from the agreeing memory
    have hdefer : ∀ {W' : Set Int}, W' ⊆ W → e''.reads ⊆ R ∪ e''.writes → e''.writes ⊆ W' →
        Step E m₀ (.rmw d a upd sem sc k) (.next (Mem.mix m₀ m'' e''.writes) (.rmw d a upd sem sc k') e'') := by
      intro W' hW' hR' hW''
      refine .defer fun v => Step.transfer (hk_step v) fun x hx => (hag x ?_).symm
      rcases hx with hx | hx
      · rcases hR' hx with h | h
        · exact Or.inl h
        · exact Or.inr (hW' (hW'' h))
      · exact Or.inr (hW' (hW'' hx))
    cases h with
    | rmw hex hS hSR hk =>
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
        refine .rmw ⟨old, hread, n₀, hu₀⟩ (fun x hx => ⟨hS hx, Set.disjoint_left.mp hWS hx⟩) hSR ?_
        intro o n ho hn
        rw [hread] at ho
        cases ho
        rcases ih old (hk old n hr₀ hn) hdA' (hagw n) rfl with
          ⟨-, -, -, hsafe⟩ | ⟨s, hsC, -, -, -, -, hswe, -, -⟩
        · rw [Mem.mix_write n hWS, Set.sdiff_sdiff_comm] at hsafe
          exact hsafe
        · exact absurd (fun x hx => (hW' hx)) (hsh' hsC hswe)
      · -- a shared atomic goes first
        refine Or.inr ⟨s, hsC, hsc, hsr, hsw, hsre, hswe, hsg, ?_⟩
        refine .rmw ⟨old, hr₀, n₀, hu₀⟩ hS hSR fun o n ho hn => ?_
        rw [hr₀] at ho
        cases ho
        rcases ih old (hk old n hr₀ hn) hdA' (hagw n) rfl with
          ⟨-, hW', -, -⟩ | ⟨s', -, hsc', hsr', hsw', -, -, -, hsafe⟩
        · exact absurd hW' (hsh' hsC hswe)
        · rwa [AtomicStep.unique hsc' hsc hsr' hsr hsw' hsw] at hsafe
    | shared hal hC hsem hsc htot hk =>
      obtain ⟨n₀, hu₀⟩ := htot default
      rcases ih default (hk default n₀ hu₀) hdA hag rfl with
        ⟨hR', hW', -, -⟩ | ⟨s, hsC, hsc', hsr, hsw, hsre, hswe, hsg, -⟩
      · -- a private step goes first
        refine Or.inl ⟨hR', hW', hdefer le_rfl hR' hW', ?_⟩
        refine .shared hal hC hsem hsc htot fun o n hn => ?_
        rcases ih o (hk o n hn) hdA hag rfl with
          ⟨-, -, -, hsafe⟩ | ⟨s, hsC, -, -, -, -, hswe, -, -⟩
        · exact hsafe
        · exact absurd hW' (hsh le_rfl hsC hswe)
      · -- another shared atomic goes first
        refine Or.inr ⟨s, hsC, hsc', hsr, hsw, hsre, hswe, hsg, ?_⟩
        refine .shared hal hC hsem hsc htot fun o n hn => ?_
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

/-- A `SafeA` program does not step into undefined behaviour against a memory that agrees on its
private footprint and holds its shared cells. -/
theorem no_ub {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep}
    {m₀ m : Mem} {p : Prog α} (h : SafeA E R W C Q log m₀ p) (hag : ∀ x ∈ R ∪ W, m x = m₀ x)
    (hpres : ∀ x ∈ cellBytes C, m x ≠ none) : ¬ Step E m p .ub := by
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
    | shared hal hC hsem hsc htot _ =>
      obtain ⟨v, hv⟩ := Mem.read_some_of_writable (m := m)
        ⟨hal, fun i hi => hpres _ (span_sub_cellBytes hC (Mem.mem_span hi))⟩
      rcases hbad with h1 | ⟨old', hr', hno⟩
      · rw [hv] at h1; cases h1
      · obtain ⟨new, hu⟩ := htot old'; exact hno new hu
  | pickUB hno => cases h with | pick hex _ => obtain ⟨b, hb⟩ := hex; exact hno b hb
  | ub => cases h

/-! ## Termination -/

/-- One step of a program in its own view: a step against a memory that agrees with its own off
the shared cells, whose cells may hold anything (other programs change them); afterwards its own
memory takes the step's writes off the cells. -/
def Next {α : Type} (E : Env) (C : Set (DType × Int)) : Mem × Prog α → Mem × Prog α → Prop
  | (m', p'), (m, p) => ∃ m₁ m₁' e, (∀ x, x ∉ cellBytes C → m₁ x = m x) ∧
      Step E m₁ p (.next m₁' p' e) ∧ m' = Mem.mix m₁' m (cellBytes C)

/-- A shared atomic pending while its continuation steps: the run is as long as the
continuation's, whatever old value the atomic finds. -/
private theorem acc_shared {α : Type} {d : DType} {a : Int} {upd : d.Val → d.Val → Prop} {sem : MemSem}
    {sc : Scope} (hC : (d, a) ∈ C) :
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
        · rw [Mem.mix_out hx, Mem.write_frame _ _ _ _ (fun h => hx (span_sub_cellBytes hC h)), hag x hx]
      rw [this]; exact hall old
    | defer hk' =>
      rename_i k'
      have hnext : ∀ v, Next E C (Mem.mix m₁' m (cellBytes C), k' v) (m, k v) :=
        fun v => ⟨m₁, m₁', e, hag, hk' v, rfl⟩
      exact ih _ (hs ▸ hnext default) _ k' rfl fun v => (hall v).inv (hnext v)

/-- A private atomic pending while its continuation steps. -/
private theorem acc_private {α : Type} {Q : α → List AtomicStep → Mem → Prop} {d : DType} {a : Int}
    {upd : d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope} (hQ : PermInv Q) :
    ∀ s : Mem × Prog α, Acc (Next E C) s → ∀ (W : Set Int) (log : List AtomicStep) (m : Mem)
      (k : d.Val → Prog α) (old n₀ : d.Val), Disjoint (cellBytes C) (R ∪ W) →
      s = (m.write d a n₀, k old) → m.read d a = some old → upd old n₀ →
      Mem.span d a ⊆ W → Disjoint (Mem.span d a) R →
      (∀ old new, m.read d a = some old → upd old new →
        SafeA E R (W \ Mem.span d a) C Q log (m.write d a new) (k old)) →
      (∀ new, upd old new → Acc (Next E C) (m.write d a new, k old)) →
      Acc (Next E C) (m, .rmw d a upd sem sc k) := by
  intro s hs
  induction hs with
  | intro s _ ih =>
    intro W log m k old n₀ hdA hs hr hu hS hSR hk hall
    have hSC : Disjoint (Mem.span d a) (cellBytes C) :=
      Set.disjoint_left.mpr fun x hx hc => Set.disjoint_left.mp hdA hc (Or.inr (hS hx))
    -- an atomic on these bytes is not shared
    have hnotC : (d, a) ∉ C := fun hC => by
      obtain ⟨x, hx⟩ := span_nonempty d a
      exact Set.disjoint_left.mp hSC hx (span_sub_cellBytes hC hx)
    have hdA' : Disjoint (cellBytes C) (R ∪ (W \ Mem.span d a)) :=
      hdA.mono_right (Set.union_subset_union_right _ Set.sdiff_subset)
    -- a shared step writes no private byte
    have hsh : ∀ {e : Event} {W' : Set Int} {s : AtomicStep}, W' ⊆ W → (s.d, s.a) ∈ C →
        e.writes = Mem.span s.d s.a → ¬ e.writes ⊆ W' := by
      intro e W' s hW' hC hw hsub
      obtain ⟨x, hx⟩ := span_nonempty s.d s.a
      rw [← hw] at hx
      exact Set.disjoint_left.mp hdA (span_sub_cellBytes hC (hw ▸ hx)) (Or.inr (hW' (hsub hx)))
    constructor
    rintro ⟨m', p'⟩ ⟨m₁, m₁', e, hag, hstep, rfl⟩
    have hagS : ∀ x ∈ Mem.span d a, m₁ x = m x := fun x hx => hag x (Set.disjoint_left.mp hSC hx)
    have hagRW : ∀ x ∈ R ∪ W, m₁ x = m x := fun x hx => hag x (fun hc => Set.disjoint_left.mp hdA hc hx)
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
    | defer hk' =>
      rename_i k'
      have hagw : ∀ n, ∀ x ∈ R ∪ (W \ Mem.span d a), m₁ x = (m.write d a n) x := by
        intro n x hx
        have hxS : x ∉ Mem.span d a := by
          rcases hx with hx | hx
          · exact fun h => Set.disjoint_left.mp hSR h hx
          · exact hx.2
        rw [Mem.write_frame _ _ _ _ hxS]
        exact hagRW x (hx.elim Or.inl (fun h => Or.inr h.1))
      have hrmw : SafeA E R W C Q log m (.rmw d a upd sem sc k) := .rmw ⟨old, hr, n₀, hu⟩ hS hSR hk
      rcases hrmw.stepT hQ hdA hagRW (.defer hk') with
        ⟨-, hW', -, hsafe'⟩ | ⟨s', hsC, -, -, hsw, hsre, hswe, -, hsafe'⟩
      · -- a private step went first
        have hWS : Disjoint (Mem.span d a) e.writes := by
          rcases (hk old n₀ hr hu).stepT hQ hdA' (hagw n₀) (hk' old) with
            ⟨-, hW'', -, -⟩ | ⟨s, hsC, -, -, -, -, hswe, -, -⟩
          · exact Set.disjoint_left.mpr fun x hx hw => (hW'' hw).2 hx
          · exact absurd hW' (hsh le_rfl hsC hswe)
        have hm' : Mem.mix m₁' m (cellBytes C) = Mem.mix m m₁' e.writes := by
          funext x
          by_cases hx : x ∈ cellBytes C
          · rw [Mem.mix_in hx, Mem.mix_out (fun hw => Set.disjoint_left.mp hdA hx (Or.inr (hW' hw)))]
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
            · rw [Mem.mix_in hx, Mem.mix_out (fun hw => Set.disjoint_left.mp hdA hx (Or.inr (hW' hw)))]
            · rw [Mem.mix_out hx]
          · exact absurd hW' (hsh le_rfl hsC hswe)
        have hr' : (Mem.mix m m₁' e.writes).read d a = some old := by
          rw [← hr]
          exact Mem.read_congr fun x hx => Mem.mix_out (Set.disjoint_left.mp hWS hx)
        rw [hm']
        cases hsafe' with
        | rmw _ hS' hSR' hk'' =>
          exact ih _ (by rw [hs]; exact hcomm n₀ hu) (W \ e.writes) log _ k' old n₀
            (hdA.mono_right (Set.union_subset_union_right _ Set.sdiff_subset)) rfl hr' hu hS' hSR' hk''
            fun new hnew => (hall new hnew).inv (hcomm new hnew)
        | shared _ hC' => exact absurd hC' hnotC
      · -- a shared atomic went first: its own memory is unchanged
        have hm' : Mem.mix m₁' m (cellBytes C) = m := by
          funext x
          by_cases hx : x ∈ cellBytes C
          · rw [Mem.mix_in hx]
          · rw [Mem.mix_out hx, hsw, Mem.write_frame _ _ _ _ (fun h => hx (span_sub_cellBytes hsC h)),
              hag x hx]
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
                Mem.mix_out (by rw [hswe]; exact fun h => hx (span_sub_cellBytes hsC h)), Mem.mix_out hx]
        rw [hm']
        cases hsafe' with
        | rmw _ hS' hSR' hk'' =>
          exact ih _ (by rw [hs]; exact hcomm n₀ hu) W (log ++ [s']) m k' old n₀ hdA rfl hr hu hS' hSR'
            hk'' fun new hnew => (hall new hnew).inv (hcomm new hnew)
        | shared _ hC' => exact absurd hC' hnotC

/-- A step that writes nothing outside the cells leaves a program's own memory as it was. -/
private theorem mix_off {m₁ m : Mem} (hag : ∀ x, x ∉ cellBytes C → m₁ x = m x) :
    Mem.mix m₁ m (cellBytes C) = m := by
  funext x
  by_cases hx : x ∈ cellBytes C
  · rw [Mem.mix_in hx]
  · rw [Mem.mix_out hx, hag x hx]

/-- **Termination.** -/
theorem acc {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep}
    {m : Mem} {p : Prog α} (h : SafeA E R W C Q log m p) (hQ : PermInv Q)
    (hdA : Disjoint (cellBytes C) (R ∪ W)) : Acc (Next E C) (m, p) := by
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
      have : Mem.mix (writeAll m₁ d addrs vals order) m₀ (cellBytes C') = writeAll m₀ d addrs vals order := by
        funext x
        by_cases hx : x ∈ cellBytes C'
        · rw [Mem.mix_in hx, writeAll_frame _ _ _ _ (fun h => Set.disjoint_left.mp hdA hx (Or.inr (hW h)))]
        · rw [Mem.mix_out hx]
          exact writeAll_agree (fun y hy => hag y hy) addrs vals order x hx
      rw [this]
      exact ih order hperm hQ (hdA.mono_right (Set.union_subset_union_right _ Set.sdiff_subset))
  | @rmw W' C' α' Q' log' m₀ d a upd sem sc k hex hS hSR hk ih =>
    obtain ⟨old, hr, n₀, hu⟩ := hex
    have hdA' : Disjoint (cellBytes C') (R ∪ (W' \ Mem.span d a)) :=
      hdA.mono_right (Set.union_subset_union_right R Set.sdiff_subset)
    exact acc_private hQ _ (ih old n₀ hr hu hQ hdA') _ _ _ _ old n₀ hdA rfl hr hu hS hSR hk
      fun new hnew => ih old new hr hnew hQ hdA'
  | shared hal hC hsem hsc htot _ ih =>
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

/-- **A scalar atomic on the shared region** (`tl.atomic_*(ptr, v)` with a scalar pointer and
no mask, as `tl.atomicWith`). Any old value may be found; the step is logged. For
`tl.atomic_add`, which is `.env fun e => tl.atomicWith …`, apply `SafeA.env` first (module
docstring). -/
theorem atomic1 {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep}
    {m : Mem} {d : DType} {ptrs : Tensor [] (Ptr d)} {upd : Idx [] → d.Val → d.Val → Prop}
    {sem : MemSem} {sc : Scope} {k : Tensor [] d.Val → Prog α}
    (hal : Mem.Aligned d (ptrs.get ()).addr) (hC : (d, (ptrs.get ()).addr) ∈ C)
    (hsem : sem = .acqRel) (hsc : sc ≠ .cta) (htot : ∀ old, ∃ new, upd () old new)
    (hk : ∀ old new, upd () old new →
      SafeA E R W C Q (log ++ [⟨d, (ptrs.get ()).addr, old, new⟩]) m (k (Tensor.scalar old))) :
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

/-- The invariant of a launch whose programs share `A` through atomics. Program `i` is `SafeA`
in its own view with the private bytes it has not yet written as write footprint. -/
structure InvA (tr : List (Fin N × Event)) (c : Config N) : Prop where
  safe : ∃ G : List (Fin N × AtomicStep),
    (∀ i, ∃ Wc, Wc ⊆ W i ∧ (∀ (n : Nat) (e : Event), tr[n]? = some (i, e) → Disjoint e.writes Wc) ∧
      SafeA E (R i) Wc C (fun _ log m => Q i log m) (proj G i) (localMem R W init c i) (c.progs i)) ∧
    ∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ cellBytes C, c.mem x = mA x
  frame : ∀ x, x ∉ cellBytes C → (∀ i, x ∉ W i) → c.mem x = init x
  present : ∀ x ∈ cellBytes C, c.mem x ≠ none
  good : ∀ (n : Nat) (j : Fin N) (e : Event), tr[n]? = some (j, e) →
    (e.reads ⊆ R j ∪ e.writes ∧ e.writes ⊆ W j) ∨
      ((∃ c ∈ C, e.reads = Mem.span c.1 c.2 ∧ e.writes = Mem.span c.1 c.2) ∧
        ∃ sc, e.atomic = some (.acqRel, sc) ∧ sc ≠ .cta)
  clean : ¬ IntraRace tr

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

/-- The history condition of program `i` after one more event of program `j`. -/
theorem hist_snoc {tr : List (Fin N × Event)} {i j : Fin N} {e : Event} {Wc Wc' : Set Int}
    (hold : ∀ (n : Nat) (f : Event), tr[n]? = some (i, f) → Disjoint f.writes Wc) (hsub : Wc' ⊆ Wc)
    (hnew : i = j → Disjoint e.writes Wc') :
    ∀ (n : Nat) (f : Event), (tr ++ [(j, e)])[n]? = some (i, f) → Disjoint f.writes Wc' := by
  intro n f hn
  by_cases hlt : n < tr.length
  · rw [List.getElem?_append_left hlt] at hn; exact (hold n f hn).mono_right hsub
  · have hn' : n = tr.length := by
      rcases Nat.lt_or_ge n (tr.length + 1) with h | h
      · omega
      · rw [List.getElem?_eq_none (by simp; omega)] at hn; cases hn
    subst hn'
    rw [List.getElem?_append_right le_rfl] at hn
    simp at hn
    obtain ⟨rfl, rfl⟩ := hn
    exact hnew rfl

theorem invA_step (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i)) (hRW : ∀ i, Disjoint (R i) (W i))
    (hcells : ∀ c ∈ C, ∀ c' ∈ C, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    (hQ : ∀ i, PermInv (fun (_ : Unit) log m => Q i log m))
    {tr : List (Fin N × Event)} {c c' : Config N} {j : Fin N}
    {e : Event} (hinv : InvA E R W C Q init tr c) (hs : LStep E c (j, e) c') :
    InvA E R W C Q init (tr ++ [(j, e)]) c' ∧
    SafeA.Next E C (localMem R W init c' j, c'.progs j) (localMem R W init c j, c.progs j) ∧
    ∀ i, i ≠ j → localMem R W init c' i = localMem R W init c i ∧ c'.progs i = c.progs i := by
  cases hs with
  | step _ hstep =>
    rename_i m' p'
    obtain ⟨G, hsafeG, mA, hrep, hagA⟩ := hinv.safe
    obtain ⟨Wc, hWc, hhist, hsj⟩ := hsafeG j
    have hag : ∀ x ∈ R j ∪ Wc, c.mem x = localMem R W init c j x := fun x hx => by
      have : x ∈ R j ∪ W j := hx.elim Or.inl (fun h => Or.inr (hWc h))
      simp [localMem, this]
    have hdA : Disjoint (cellBytes C) (R j ∪ Wc) :=
      (hA j).mono_right (Set.union_subset_union_right _ hWc)
    have hframe := Step.frame hstep
    -- earlier events of `j` are private, inside the footprint, or shared, on the cells
    have hold : ∀ (a : Nat) (f : Event), tr[a]? = some (j, f) →
        (f.reads ∪ f.writes ⊆ R j ∪ W j) ∨ (f.reads ∪ f.writes ⊆ cellBytes C) := by
      intro a f ha
      rcases hinv.good a j f ha with ⟨hfr, hfw⟩ | ⟨⟨cf, hcf, hfr, hfw⟩, -⟩
      · left
        rintro z (hz | hz)
        · exact (hfr hz).elim Or.inl (fun h => Or.inr (hfw h))
        · exact Or.inr (hfw hz)
      · right
        rintro z (hz | hz)
        · rw [hfr] at hz; exact span_sub_cellBytes hcf hz
        · rw [hfw] at hz; exact span_sub_cellBytes hcf hz
    rcases hsj.stepT (hQ j) hdA hag hstep with
      ⟨hR, hW, hs2, hsafe'⟩ | ⟨s, hsC, hsc, hsr, hsw, hsre, hswe, hsg, hsafe'⟩
    · -- a private step
      have hloc : localMem R W init ⟨m', Function.update c.progs j p'⟩ j =
          Mem.mix (localMem R W init c j) m' e.writes := by
        funext x
        by_cases hx : x ∈ R j ∪ W j
        · simp only [localMem, hx, ite_true]
          by_cases hxe : x ∈ e.writes
          · rw [Mem.mix_in hxe]
          · rw [Mem.mix_out hxe, hframe x hxe]; simp [localMem, hx]
        · have hxe : x ∉ e.writes := fun h => hx (Or.inr (hWc (hW h)))
          simp only [localMem, hx, ite_false]
          rw [Mem.mix_out hxe]; simp [localMem, hx]
      have hother : ∀ i, i ≠ j → localMem R W init ⟨m', Function.update c.progs j p'⟩ i =
          localMem R W init c i := by
        intro i hij
        funext x
        by_cases hx : x ∈ R i ∪ W i
        · simp only [localMem, hx, ite_true]
          exact hframe x fun hw => Set.disjoint_left.mp (hdisj j i hij.symm) (hWc (hW hw)) hx
        · simp [localMem, hx]
      have hnotA : ∀ x ∈ cellBytes C, x ∉ e.writes := fun x hx hw =>
        Set.disjoint_left.mp (hA j) hx (Or.inr (hWc (hW hw)))
      have heRW : e.reads ∪ e.writes ⊆ R j ∪ W j := by
        rintro z (hz | hz)
        · exact (hR hz).elim Or.inl (fun h => Or.inr (hWc (hW h)))
        · exact Or.inr (hWc (hW hz))
      refine ⟨⟨⟨G, fun i => ?_, mA, hrep, fun x hx => ?_⟩, fun x hx hw => ?_, fun x hx => ?_,
        good_snoc hinv.good (Or.inl ⟨hR, hW.trans hWc⟩), ?_⟩, ?_, ?_⟩
      · by_cases hij : i = j
        · subst hij
          refine ⟨Wc \ e.writes, Set.sdiff_subset.trans hWc,
            hist_snoc hhist Set.sdiff_subset (fun _ => Set.disjoint_sdiff_right), ?_⟩
          rw [hloc]; simpa using hsafe'
        · obtain ⟨Wi, hWi, hhi, hsi⟩ := hsafeG i
          refine ⟨Wi, hWi, hist_snoc hhi le_rfl (fun h => absurd h hij), ?_⟩
          rw [hother i hij]; simpa [Function.update_of_ne hij] using hsi
      · simp only; rw [hframe x (hnotA x hx)]; exact hagA x hx
      · simp only; rw [hframe x fun h => hw j (hWc (hW h))]; exact hinv.frame x hx hw
      · simp only; rw [hframe x (hnotA x hx)]; exact hinv.present x hx
      · intro hrace
        rcases intraRace_snoc hrace with h | ⟨a, f, ha, hcon, -⟩
        · exact hinv.clean h
        · rcases hold a f ha with hf | hf
          · rcases hinv.good a j f ha with ⟨hfr, hfw⟩ | ⟨⟨cf, hcf, hfr, hfw⟩, -⟩
            · have hfe : Disjoint f.writes e.writes := (hhist a f ha).mono_right hW
              have hfR : Disjoint f.writes (R j) := ((hRW j).symm).mono_left hfw
              have heR : Disjoint e.writes (R j) := ((hRW j).symm).mono_left (hW.trans hWc)
              rcases hcon with ⟨z, hzf, hze⟩ | ⟨z, hzf, hze⟩
              · rcases hze with hze | hze
                · rcases hR hze with hz | hz
                  · exact Set.disjoint_left.mp hfR hzf hz
                  · exact Set.disjoint_left.mp hfe hzf hz
                · exact Set.disjoint_left.mp hfe hzf hze
              · rcases hfr hzf with hz | hz
                · exact Set.disjoint_left.mp heR hze hz
                · exact Set.disjoint_left.mp hfe hz hze
            · obtain ⟨x, hx⟩ := span_nonempty cf.1 cf.2
              exact Set.disjoint_left.mp (hA j) (span_sub_cellBytes hcf hx) (hf (Or.inr (hfw ▸ hx)))
          · rcases hcon with ⟨z, hzf, hze⟩ | ⟨z, hzf, hze⟩
            · exact Set.disjoint_left.mp (hA j) (hf (Or.inr hzf)) (heRW hze)
            · exact Set.disjoint_left.mp (hA j) (hf (Or.inl hzf)) (heRW (Or.inr hze))
      · refine ⟨localMem R W init c j, Mem.mix (localMem R W init c j) m' e.writes, e,
          fun _ _ => rfl, ?_, ?_⟩
        · simpa using hs2
        · rw [hloc]
          funext x
          by_cases hx : x ∈ cellBytes C
          · rw [Mem.mix_in hx, Mem.mix_out (hnotA x hx)]
          · rw [Mem.mix_out hx]
      · intro i hij; exact ⟨hother i hij, Function.update_of_ne hij _ _⟩
    · -- a shared atomic
      subst hsw
      have hloc : ∀ i, localMem R W init ⟨c.mem.write s.d s.a s.new, Function.update c.progs j p'⟩ i =
          localMem R W init c i := by
        intro i
        funext x
        by_cases hx : x ∈ R i ∪ W i
        · simp only [localMem, hx, ite_true]
          exact Mem.write_frame _ _ _ _ fun hsp =>
            Set.disjoint_left.mp (hA i) (span_sub_cellBytes hsC hsp) hx
        · simp [localMem, hx]
      have hrA : mA.read s.d s.a = some s.old := by
        rw [← hsr]; exact Mem.read_congr fun x hx => (hagA x (span_sub_cellBytes hsC hx)).symm
      have heA : e.reads ∪ e.writes ⊆ cellBytes C := by
        rintro z (hz | hz)
        · rw [hsre] at hz; exact span_sub_cellBytes hsC hz
        · rw [hswe] at hz; exact span_sub_cellBytes hsC hz
      refine ⟨⟨⟨G ++ [(j, s)], fun i => ?_, mA.write s.d s.a s.new, ?_,
        Mem.write_agree hagA _ _ _⟩, fun x hx hw => ?_, fun x hx => Mem.write_present _ (hinv.present x hx),
        good_snoc hinv.good (Or.inr ⟨⟨(s.d, s.a), hsC, hsre, hswe⟩, hsg⟩), ?_⟩, ?_, ?_⟩
      · rw [hloc i]
        by_cases hij : i = j
        · subst hij
          refine ⟨Wc, hWc, hist_snoc hhist le_rfl (fun _ => ?_), ?_⟩
          · rw [hswe]
            exact Set.disjoint_left.mpr fun x hx hxW => Set.disjoint_left.mp (hA i)
              (span_sub_cellBytes hsC hx) (Or.inr (hWc hxW))
          · rw [proj_snoc_self]; simpa using hsafe'
        · obtain ⟨Wi, hWi, hhi, hsi⟩ := hsafeG i
          refine ⟨Wi, hWi, hist_snoc hhi le_rfl (fun h => absurd h hij), ?_⟩
          rw [proj_snoc_other _ hij]; simpa [Function.update_of_ne hij] using hsi
      · rw [List.map_append]; exact hrep.snoc hrA
      · simp only
        rw [Mem.write_frame _ _ _ _ fun hsp => hx (span_sub_cellBytes hsC hsp)]
        exact hinv.frame x hx hw
      · intro hrace
        rcases intraRace_snoc hrace with h | ⟨a, f, ha, hcon, hsame⟩
        · exact hinv.clean h
        · rcases hinv.good a j f ha with ⟨hfr, hfw⟩ | ⟨⟨cf, hcf, hfr, hfw⟩, hfg⟩
          · have hf : f.reads ∪ f.writes ⊆ R j ∪ W j := by
              rintro z (hz | hz)
              · exact (hfr hz).elim Or.inl (fun h => Or.inr (hfw h))
              · exact Or.inr (hfw hz)
            rcases hcon with ⟨z, hzf, hze⟩ | ⟨z, hzf, hze⟩
            · exact Set.disjoint_left.mp (hA j) (heA hze) (hf (Or.inr hzf))
            · exact Set.disjoint_left.mp (hA j) (heA (Or.inr hze)) (hf (Or.inl hzf))
          · -- two shared atomics on overlapping cells: the same cell
            have hsame' : cf = (s.d, s.a) := by
              refine hcells cf hcf _ hsC ?_
              rcases hcon with ⟨z, hzf, hze⟩ | ⟨z, hzf, hze⟩
              · rw [hfw] at hzf
                rcases hze with hze | hze
                · rw [hsre] at hze; exact ⟨z, hzf, hze⟩
                · rw [hswe] at hze; exact ⟨z, hzf, hze⟩
              · rw [hfr] at hzf; rw [hswe] at hze; exact ⟨z, hzf, hze⟩
            subst hsame'
            obtain ⟨sc₁, hf1, -⟩ := hfg
            obtain ⟨sc₂, he1, -⟩ := hsg
            exact hsame ⟨by rw [hf1]; rfl, by rw [he1]; rfl, hfr.trans hsre.symm, hfw.trans hswe.symm⟩
      · refine ⟨Mem.mix (localMem R W init c j) c.mem (cellBytes C),
          Mem.mix (Mem.mix (localMem R W init c j) c.mem (cellBytes C)) (c.mem.write s.d s.a s.new) e.writes,
          e, fun x hx => Mem.mix_out hx, ?_, ?_⟩
        · simpa using Step.transfer hstep fun x hx => Mem.mix_in (heA hx)
        · rw [hloc j]
          funext x
          by_cases hx : x ∈ cellBytes C
          · rw [Mem.mix_in hx]
          · rw [Mem.mix_out hx, Mem.mix_out (fun h => hx (heA (Or.inr h))), Mem.mix_out hx]
      · intro i hij; exact ⟨hloc i, Function.update_of_ne hij _ _⟩

theorem invA_reach (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j))
    (hA : ∀ i, Disjoint (cellBytes C) (R i ∪ W i)) (hRW : ∀ i, Disjoint (R i) (W i))
    (hcells : ∀ c ∈ C, ∀ c' ∈ C, (Mem.span c.1 c.2 ∩ Mem.span c'.1 c'.2).Nonempty → c = c')
    (hQ : ∀ i, PermInv (fun (_ : Unit) log m => Q i log m))
    {c₀ c : Config N} {tr : List (Fin N × Event)}
    (h0 : InvA E R W C Q init [] c₀) (hr : Reach E c₀ tr c) : InvA E R W C Q init tr c := by
  induction hr with
  | refl => exact h0
  | @tail c₂ c₃ tr₁ ev _ hs ih =>
    obtain ⟨j, e⟩ := ev
    exact (invA_step hdisj hA hRW hcells hQ ih hs).1

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
  · have hy : y.reads ∪ y.writes ⊆ R j ∪ W j := by
      rintro z (hz | hz)
      · exact (hyr hz).elim Or.inl (fun h => Or.inr (hyw h))
      · exact Or.inr (hyw hz)
    have hx : x.reads ⊆ R i ∪ W i := fun z hz => (hxr hz).elim Or.inl (fun h => Or.inr (hxw h))
    rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
    · exact Set.disjoint_left.mp (hdisj i j hij) (hxw hzx) (hy hzy)
    · exact Set.disjoint_left.mp (hdisj j i (Ne.symm hij)) (hyw hzy) (hx hzx)
  · have hyA : ∀ z, z ∈ y.reads ∪ y.writes → z ∈ cellBytes C := by
      rintro z (hz | hz)
      · rw [hyr] at hz; exact span_sub_cellBytes (d := cy.1) (a := cy.2) hcy hz
      · rw [hyw] at hz; exact span_sub_cellBytes (d := cy.1) (a := cy.2) hcy hz
    have hx : x.reads ∪ x.writes ⊆ R i ∪ W i := by
      rintro z (hz | hz)
      · exact (hxr hz).elim Or.inl (fun h => Or.inr (hxw h))
      · exact Or.inr (hxw hz)
    rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
    · exact Set.disjoint_left.mp hAi (hyA z hzy) (hx (Or.inr hzx))
    · exact Set.disjoint_left.mp hAi (hyA z (Or.inr hzy)) (hx (Or.inl hzx))
  · have hxA : ∀ z, z ∈ x.reads ∪ x.writes → z ∈ cellBytes C := by
      rintro z (hz | hz)
      · rw [hxr] at hz; exact span_sub_cellBytes (d := cx.1) (a := cx.2) hcx hz
      · rw [hxw] at hz; exact span_sub_cellBytes (d := cx.1) (a := cx.2) hcx hz
    have hy : y.reads ∪ y.writes ⊆ R j ∪ W j := by
      rintro z (hz | hz)
      · exact (hyr hz).elim Or.inl (fun h => Or.inr (hyw h))
      · exact Or.inr (hyw hz)
    rcases hcon with ⟨z, hzx, hzy⟩ | ⟨z, hzx, hzy⟩
    · exact Set.disjoint_left.mp hAj (hxA z (Or.inr hzx)) (hy hzy)
    · exact Set.disjoint_left.mp hAj (hxA z (Or.inl hzx)) (hy (Or.inr hzy))
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
other program writes into and with its read footprint disjoint from its own write footprint, and
with a postcondition that does not depend on the order of its log (its atomics may be performed
in any order, `Step.defer`). Suppose also that the shared cells `C` are equal or disjoint, are
disjoint from every private footprint, and are present initially. Then:
- the launch is well behaved and cannot hang;
- every outcome is a normal end;
- each program's private footprint holds a memory satisfying its postcondition for its own log;
- `A` holds the replay of all logs, interleaved in some order `G`;
- every other byte keeps its initial value. -/
theorem shared {k : Kernel} {L : Launch} {E : Env} {init : Mem}
    {R W : Fin L.size → Set Int} {C : Set (DType × Int)} {Q : Fin L.size → List AtomicStep → Mem → Prop}
    (hsafe : ∀ i, SafeA E (R i) (W i) C (fun _ log m => Q i log m) [] init (k.body (L.ctx i)))
    (hQ : ∀ i, PermInv (fun (_ : Unit) log m => Q i log m))
    (hdisj : ∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j)) (hRW : ∀ i, Disjoint (R i) (W i))
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
    refine ⟨⟨[], fun i => ⟨W i, le_rfl, fun n e h => by simp at h, ?_⟩, init, .nil _, fun _ _ => rfl⟩,
      fun _ _ _ => rfl, hpres, fun n j e h => by simp at h, fun ⟨a, _, _, _, _, ha, _⟩ => by simp at ha⟩
    rw [hloc0]
    simpa [Config.start, proj] using hsafe i
  have hinv : ∀ tr c, Reach E (Config.start k L init) tr c → InvA E R W C Q init tr c :=
    fun tr c hr => invA_reach hdisj hA hRW hcells hQ h0 hr
  have hnoub : ∀ tr c, Reach E (Config.start k L init) tr c → ¬ c.StuckUB E := by
    rintro tr c hr ⟨i, hub⟩
    have hI := hinv tr c hr
    obtain ⟨G, hsafeG, -⟩ := hI.safe
    obtain ⟨Wc, hWc, -, hs⟩ := hsafeG i
    exact hs.no_ub (m := c.mem) (fun x hx => by
      have : x ∈ R i ∪ W i := hx.elim Or.inl (fun h => Or.inr (hWc h))
      simp [localMem, this]) hI.present hub
  refine ⟨fun tr c hr => ⟨hnoub tr c hr, no_interRaceA hdisj hA hcells (hinv tr c hr),
    (hinv tr c hr).clean⟩, ?_⟩
  intro o ho
  cases o with
  | ok fin =>
    obtain ⟨tr, c, hr, hdone, hmem⟩ := ho
    have hI := hinv tr c hr
    obtain ⟨G, hsafeG, mA, hrep, hagA⟩ := hI.safe
    refine ⟨fin, rfl, G, fun i => ?_, ⟨mA, hrep, fun x hx => hmem ▸ hagA x hx⟩,
      fun x hx hw => hmem ▸ hI.frame x hx hw⟩
    obtain ⟨Wc, -, -, hs⟩ := hsafeG i
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
      obtain ⟨-, hnext, hothers⟩ := invA_step (j := (ev n).1) (e := (ev n).2) hdisj hA hRW hcells hQ
        (hinv _ _ (hreach n)) (by simpa using hev n)
      refine ⟨hnext, fun i hij => ?_⟩
      obtain ⟨h1, h2⟩ := hothers i hij
      simp only [S, h1, h2]
    have hT : ∀ i, ∃ T, ∀ m, T ≤ m → ¬ SafeA.Next E C (S i (m + 1)) (S i m) := by
      intro i
      have hacc : Acc (SafeA.Next E C) (S i 0) := by
        obtain ⟨G, hsafeG, -⟩ := (hinv _ _ (hreach 0)).safe
        obtain ⟨Wc, hWc, -, hs⟩ := hsafeG i
        simpa [S] using hs.acc (hQ i) ((hA i).mono_right (Set.union_subset_union_right _ hWc))
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

/-- Step `s` adds `inc` to the `int32` cell at `a` (`inc = 0` for a step on another cell). -/
def AddsTo (a : Int) (inc : I32) (s : AtomicStep) : Prop :=
  ∀ (m : Mem) (x : I32), m.read .i32 a = some x → m.read s.d s.a = some s.old →
    (m.write s.d s.a s.new).read .i32 a = some (x + inc)

/-- An `int32` add of `v` at `a` adds `v` to that cell. -/
theorem addsTo_add {a : Int} {v : I32} {old new : I32}
    (hnew : new = old + v) : AddsTo a v ⟨.i32, a, old, new⟩ := by
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
      intro i; by_cases h : j = i <;> simp [proj, h]
    simp only [List.map_cons, List.sum_cons, ih, hsplit, Finset.sum_add_distrib, Finset.sum_ite_eq,
      Finset.mem_univ, ite_true]

end TVA
