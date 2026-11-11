import TVA.Proof.Shared.Log
import TVA.Proof.Int.Basic

/-!
# What a replay leaves in a cell

`correct_of_safeA` (`TVA.Proof.Shared.Correct`) says that the shared cells end as the replay of
every program's atomic steps, in some interleaving `G`. This file computes the cell from that,
whatever the interleaving.

**Updates that are functions of the cell** (`MapsCell`, `replay_maps`). Many programs
`tl.atomic_add` integers into one cell: each step maps the cell `x ↦ x + inc`, and addition
commutes, so the cell ends at its initial value plus the sum of all increments in every order
(`replay_add`, for every integer width); `sum_proj` splits that sum over programs. An integer
step determines its increment (`new − old`), so a program can state its increments from its own
log.

**Steps paired with their operands** (`Paired`). A float addition's step does not determine its
operand (adding `1.0` or `0.5` to `2^24` both leave `2^24`), and neither does a `max`. So each
program pairs its own log with its operands (`Paired.lanes` for the lanes of one atomic,
`Paired.append` across atomics); the pairing survives a reordering of the log (`Paired.perm`, so a
post-condition stated with it is `PermInv`), and `Paired.merge` merges the programs' pairings along
the interleaving. Then:

* `replay_fold`: the cell is the fold of the operands in the log's order;
* `replay_max`, `replay_min`: integer `atomic_max`/`atomic_min` from several programs leave the
  largest (smallest) of the initial value and all operands, in any order;
* `replay_err_shared` (`TVA.Proof.Shared.FloatAcc`): a float `atomic_add` from several programs
  leaves the exact sum up to an error bound, in any order.

The bitwise `atomic_and`, `atomic_or`, `atomic_xor` commute too, but have no lemma here.
-/
namespace TVA

open Classical

/-! ## The cell after a replay -/

/-- Step `s` changes the cell `(d, a)` from `x` to `g x`: replayed from any memory in which the
cell holds `x`, it leaves `g x` there. -/
def MapsCell (d : DType) (a : Int) (g : d.Val → d.Val) (s : AtomicStep) : Prop :=
  ∀ (m : Mem) (x : d.Val), m.read d a = some x → m.read s.d s.a = some s.old →
    (m.write s.d s.a s.new).read d a = some (g x)

/-- A step on the cell itself whose new value is `g` of its old value maps the cell by `g`. Use
it for a program's own steps: an integer `atomic_add` of `v` has `new = old + v`. -/
theorem mapsCell_self {d : DType} {a : Int} {old new : d.Val} {g : d.Val → d.Val} (hg : g old = new) :
    MapsCell d a g ⟨d, a, old, new⟩ := by
  intro m x hx hold
  simp only at hold
  rw [hx] at hold
  cases hold
  rw [Mem.read_write_self (Mem.writable_of_read hx), hg]

/-- A step on bytes disjoint from the cell leaves it alone. -/
theorem mapsCell_other {d : DType} {a : Int} {s : AtomicStep}
    (hdis : Disjoint (Mem.span s.d s.a) (Mem.span d a)) : MapsCell d a id s := by
  intro m x hx _
  rw [Mem.read_congr (fun y hy => Mem.write_frame _ _ _ _ (fun h => Set.disjoint_left.mp hdis h hy)), hx]
  rfl

/-- **The cell after a replay**: the left fold of the steps' maps, in the log's order. Use it
for an update that is a function of the cell; `replay_add` is the common case. -/
theorem replay_maps {d : DType} {a : Int} {g : AtomicStep → d.Val → d.Val} {m m' : Mem}
    {l : List AtomicStep} {x : d.Val} (hrep : Replay m l m') (hl : ∀ s ∈ l, MapsCell d a (g s) s)
    (h0 : m.read d a = some x) : m'.read d a = some (l.foldl (fun x s => g s x) x) := by
  induction hrep generalizing x with
  | nil => simpa using h0
  | @cons m₀ m₁ s rest hr _ ih =>
    have h1 := hl s List.mem_cons_self m₀ x h0 hr
    exact ih (fun s' hs' => hl s' (List.mem_cons_of_mem _ hs')) h1

/-- A left fold of additions is the sum. -/
private theorem foldl_add {β : Type} [AddCommMonoid β] (inc : AtomicStep → β) :
    ∀ (l : List AtomicStep) (x : β), l.foldl (fun x s => x + inc s) x = x + (l.map inc).sum
  | [], x => by simp
  | s :: l, x => by rw [List.foldl_cons, foldl_add inc l, List.map_cons, List.sum_cons, add_assoc]

/-- **Integer accumulation.** When every step adds `inc s` to the cell (`inc s = 0` for a step
on other bytes, `mapsCell_other_add`), the cell ends at its initial value plus the sum of the
increments, whatever the interleaving. It holds for every dtype whose values form a commutative
monoid, as those of every integer dtype do (wrapping addition, `IntV`). With `sum_proj`, the sum
over the interleaved log is the sum over programs of their own increments. -/
theorem replay_add {d : DType} [AddCommMonoid d.Val] {a : Int} {inc : AtomicStep → d.Val}
    {m m' : Mem} {l : List AtomicStep} {x : d.Val} (hrep : Replay m l m')
    (hl : ∀ s ∈ l, MapsCell d a (· + inc s) s) (h0 : m.read d a = some x) :
    m'.read d a = some (x + (l.map inc).sum) := by
  rw [← foldl_add]
  exact replay_maps hrep hl h0

/-- A step on bytes disjoint from the cell adds `0` to it (for `replay_add`). -/
theorem mapsCell_other_add {d : DType} [AddCommMonoid d.Val] {a : Int} {s : AtomicStep}
    (hdis : Disjoint (Mem.span s.d s.a) (Mem.span d a)) : MapsCell d a (· + 0) s := by
  have h := mapsCell_other (d := d) (a := a) hdis
  intro m x hx hold
  show _ = some (x + 0)
  rw [add_zero]
  exact h m x hx hold

/-- The sum over an interleaved log is the sum over programs of their own parts, in any
commutative monoid. Use it to split `replay_add`'s total into each program's increments. -/
theorem sum_proj {β : Type} [AddCommMonoid β] {N : Nat} (G : List (Fin N × AtomicStep))
    (f : AtomicStep → β) : ((G.map Prod.snd).map f).sum = ∑ i, ((proj G i).map f).sum := by
  induction G with
  | nil => simp [proj]
  | cons g G ih =>
    obtain ⟨j, s⟩ := g
    have hsplit : ∀ i, ((proj ((j, s) :: G) i).map f).sum =
        (if j = i then f s else 0) + ((proj G i).map f).sum := by
      intro i; by_cases h : j = i <;> simp [proj, h]
    simp only [List.map_cons, List.sum_cons, ih, hsplit, Finset.sum_add_distrib, Finset.sum_ite_eq,
      Finset.mem_univ, ite_true]

/-! ## Steps paired with their operands

A logged step records the cell and its old and new values, not the operand of the atomic that
made it: two programs adding `1.0` and `0.5` to a `float32` cell holding `2^24` log the same step
`⟨.f32, a, 2^24, 2^24⟩`. So the operand is attached per position: a log is `Paired` with the
multiset of its operands when its steps, in order, can each be given an optional operand
(`none`: the step leaves the cell alone) satisfying a relation `R`. A program proves this of its
own log (`Paired.lanes`, `Paired.append`); it survives a reordering of the log
(`Paired.perm`, so a post-condition built from it is `PermInv`); and `Paired.merge` merges the
programs' pairings along the interleaving `G` that `correct_of_safeA` gives, the operands of the
whole log being the sum over programs of their own. -/

/-- A list of pairs whose first components are a permutation of `l'` can be permuted so that
they are exactly `l'`. -/
private theorem exists_perm_of_map_perm {α β : Type} (f : α → β) {L : List α} {l' : List β}
    (h : (L.map f).Perm l') : ∃ L' : List α, L'.Perm L ∧ L'.map f = l' := by
  generalize hl : L.map f = l at h
  induction h generalizing L with
  | nil => exact ⟨[], by simp_all⟩
  | cons x _ ih =>
    obtain ⟨a, L₁, rfl, rfl, rfl⟩ := List.map_eq_cons_iff.mp hl
    obtain ⟨L₁', hp, hm⟩ := ih rfl
    exact ⟨a :: L₁', hp.cons a, by simp [hm]⟩
  | swap x y l =>
    obtain ⟨a, L₁, rfl, rfl, h₁⟩ := List.map_eq_cons_iff.mp hl
    obtain ⟨b, L₂, rfl, rfl, rfl⟩ := List.map_eq_cons_iff.mp h₁
    exact ⟨b :: a :: L₂, List.Perm.swap _ _ _, rfl⟩
  | trans _ _ ih₁ ih₂ =>
    obtain ⟨L₁, hp₁, hm₁⟩ := ih₁ hl
    obtain ⟨L₂, hp₂, hm₂⟩ := ih₂ hm₁
    exact ⟨L₂, hp₂.trans hp₁, hm₂⟩

/-- **A log paired with operands.** The steps of `log`, in order, each with an optional operand
(`none`: a step that leaves the cell alone), every pair satisfying `R`; `ops` is the multiset of
the operands that are present. -/
def Paired {β : Type} (R : AtomicStep → Option β → Prop) (log : List AtomicStep) (ops : Multiset β) :
    Prop :=
  ∃ L : List (AtomicStep × Option β), L.map Prod.fst = log ∧
    ((L.filterMap Prod.snd : List β) : Multiset β) = ops ∧ ∀ p ∈ L, R p.1 p.2

namespace Paired

variable {β : Type} {R : AtomicStep → Option β → Prop}

/-- The empty log, with no operands. -/
theorem nil : Paired R [] 0 := ⟨[], rfl, rfl, by simp⟩

/-- Two parts of a log, each paired: a program's log built atomic by atomic. -/
theorem append {l₁ l₂ : List AtomicStep} {o₁ o₂ : Multiset β} (h₁ : Paired R l₁ o₁)
    (h₂ : Paired R l₂ o₂) : Paired R (l₁ ++ l₂) (o₁ + o₂) := by
  obtain ⟨L₁, hm₁, ho₁, hr₁⟩ := h₁
  obtain ⟨L₂, hm₂, ho₂, hr₂⟩ := h₂
  refine ⟨L₁ ++ L₂, by simp [hm₁, hm₂], ?_, fun p hp => ?_⟩
  · rw [List.filterMap_append, ← Multiset.coe_add, ho₁, ho₂]
  · rcases List.mem_append.mp hp with h | h
    · exact hr₁ p h
    · exact hr₂ p h

/-- **The lanes of a tensor atomic** (`SafeA.atomic`: the log grows by `order.map step`), each
step with its lane's operand `op i`. -/
theorem lanes {ι : Type} (order : List ι) (step : ι → AtomicStep) (op : ι → Option β)
    (h : ∀ i ∈ order, R (step i) (op i)) :
    Paired R (order.map step) ((order.filterMap op : List β) : Multiset β) := by
  refine ⟨order.map fun i => (step i, op i), by simp, by simp [List.filterMap_map], ?_⟩
  intro p hp
  obtain ⟨i, hi, rfl⟩ := List.mem_map.mp hp
  exact h i hi

/-- A pairing survives a reordering of the log. So a postcondition `Q a log m` that says
`Paired R log ops` (with `ops` fixed) is `PermInv`. -/
theorem perm {log log' : List AtomicStep} {ops : Multiset β} (h : Paired R log ops)
    (hp : log.Perm log') : Paired R log' ops := by
  obtain ⟨L, hm, ho, hr⟩ := h
  obtain ⟨L', hL', hm'⟩ := exists_perm_of_map_perm Prod.fst (hm ▸ hp : (L.map Prod.fst).Perm log')
  refine ⟨L', hm', ?_, fun p hp' => hr p (hL'.subset hp')⟩
  rw [← ho]
  exact Multiset.coe_eq_coe.mpr (hL'.filterMap _)

/-- A pairing for a weaker relation. -/
theorem mono {R' : AtomicStep → Option β → Prop} (hR : ∀ s o, R s o → R' s o) {log : List AtomicStep}
    {ops : Multiset β} (h : Paired R log ops) : Paired R' log ops := by
  obtain ⟨L, hm, ho, hr⟩ := h
  exact ⟨L, hm, ho, fun p hp => hR _ _ (hr p hp)⟩

private theorem coe_filterMap_cons (p : AtomicStep × Option β) (L : List (AtomicStep × Option β)) :
    (((p :: L).filterMap Prod.snd : List β) : Multiset β) =
      ((p.2.toList : List β) : Multiset β) + ((L.filterMap Prod.snd : List β) : Multiset β) := by
  obtain ⟨s, o⟩ := p
  cases o <;> simp

/-- **Merging along the interleaving.** If each program's part of the interleaved log `G`
(`proj G i`) is paired with its operands `ops i`, the whole log `G.map Prod.snd` is paired with
the sum over programs of their operands. -/
theorem merge {N : Nat} {G : List (Fin N × AtomicStep)} {ops : Fin N → Multiset β}
    (h : ∀ i, Paired R (proj G i) (ops i)) : Paired R (G.map Prod.snd) (∑ i, ops i) := by
  induction G generalizing ops with
  | nil =>
    have h0 : ∀ i, ops i = 0 := by
      intro i
      obtain ⟨L, hm, ho, -⟩ := h i
      have hm0 : L.map Prod.fst = [] := by simpa [proj] using hm
      have : L = [] := List.map_eq_nil_iff.mp hm0
      subst this; exact ho.symm
    simp only [List.map_nil, h0, Finset.sum_const_zero]
    exact nil
  | cons g G ih =>
    obtain ⟨j, s⟩ := g
    have hself : proj ((j, s) :: G) j = s :: proj G j := by simp [proj]
    have hother : ∀ i, i ≠ j → proj ((j, s) :: G) i = proj G i := by
      intro i hij; simp [proj, Ne.symm hij]
    obtain ⟨Lj, hmj, hoj, hrj⟩ := h j
    rw [hself] at hmj
    obtain ⟨p, Lj', hp, hp1, hmj'⟩ := List.map_eq_cons_iff.mp hmj
    subst hp
    let ops' : Fin N → Multiset β := Function.update ops j ((Lj'.filterMap Prod.snd : List β) : Multiset β)
    have h' : ∀ i, Paired R (proj G i) (ops' i) := by
      intro i
      by_cases hij : i = j
      · subst hij
        refine ⟨Lj', hmj', by simp [ops'], fun q hq => hrj q (List.mem_cons_of_mem _ hq)⟩
      · rw [show ops' i = ops i from Function.update_of_ne hij _ _, ← hother i hij]
        exact h i
    obtain ⟨LL, hmL, hoL, hrL⟩ := ih h'
    refine ⟨p :: LL, by simp [hmL, hp1], ?_, fun q hq => ?_⟩
    · rw [coe_filterMap_cons, hoL]
      have hj : ops j = ((p.2.toList : List β) : Multiset β) + ops' j := by
        rw [← hoj, coe_filterMap_cons]; simp [ops']
      rw [← Finset.add_sum_erase _ ops' (Finset.mem_univ j), ← Finset.add_sum_erase _ ops (Finset.mem_univ j),
        hj, add_assoc]
      congr 2
      refine Finset.sum_congr rfl fun i hi => ?_
      exact Function.update_of_ne (Finset.ne_of_mem_erase hi) _ _
    · rcases List.mem_cons.mp hq with rfl | hq
      · exact hrj _ List.mem_cons_self
      · exact hrL q hq

end Paired

/-- **The cell after a replay, per position.** If every step of the replay, paired with its
operand, maps the cell by `g v` (`some v`) or leaves it alone (`none`), the cell ends at the left
fold of `g` over the operands, in the log's order. -/
theorem replay_fold {β : Type} {d : DType} {a : Int} (g : β → d.Val → d.Val) :
    ∀ {m m' : Mem} {L : List (AtomicStep × Option β)} {x : d.Val},
      Replay m (L.map Prod.fst) m' → (∀ p ∈ L, MapsCell d a (fun y => p.2.elim y (g · y)) p.1) →
      m.read d a = some x → m'.read d a = some ((L.filterMap Prod.snd).foldl (fun y v => g v y) x)
  | m, m', [], x, hrep, _, h0 => by cases hrep; simpa using h0
  | m, m', (s, o) :: L, x, hrep, hl, h0 => by
    cases hrep with
    | cons hr hrest =>
      have h1 := hl (s, o) List.mem_cons_self m x h0 hr
      have ih := replay_fold g hrest (fun p hp => hl p (List.mem_cons_of_mem _ hp)) h1
      cases o <;> simpa using ih

/-! ## Integer `max` and `min` -/

/-- **The `max` fold, in any order.** For `op` that takes the larger by `key`
(`key (op x v) = max (key x) (key v)`: `IntV.max` with `IntV.toInt`, `IntV.toInt_max`), the fold
of `op` over the operands `l` from `x` has a key at least `x`'s and each operand's, and equal to
one of them. -/
theorem foldl_max_spec {α : Type} (op : α → α → α) (key : α → ℤ)
    (hop : ∀ x v, key (op x v) = Max.max (key x) (key v)) :
    ∀ (l : List α) (x : α),
      key x ≤ key (l.foldl op x) ∧ (∀ v ∈ l, key v ≤ key (l.foldl op x)) ∧
      (key (l.foldl op x) = key x ∨ ∃ v ∈ l, key (l.foldl op x) = key v)
  | [], x => by simp
  | v :: l, x => by
    obtain ⟨h1, h2, h3⟩ := foldl_max_spec op key hop l (op x v)
    simp only [List.foldl_cons] at h1 h2 h3 ⊢
    rw [hop] at h1
    refine ⟨le_trans (le_max_left _ _) h1, fun v' hv' => ?_, ?_⟩
    · rcases List.mem_cons.mp hv' with rfl | h
      · exact le_trans (le_max_right _ _) h1
      · exact h2 v' h
    · rcases h3 with h | ⟨v', hv', h⟩
      · rw [hop] at h
        rcases max_choice (key x) (key v) with hm | hm
        · exact Or.inl (h.trans hm)
        · exact Or.inr ⟨v, List.mem_cons_self, h.trans hm⟩
      · exact Or.inr ⟨v', List.mem_cons_of_mem _ hv', h⟩

/-- **The `min` fold, in any order** (as `foldl_max_spec`, with `IntV.min`, `IntV.toInt_min`). -/
theorem foldl_min_spec {α : Type} (op : α → α → α) (key : α → ℤ)
    (hop : ∀ x v, key (op x v) = Min.min (key x) (key v)) (l : List α) (x : α) :
    key (l.foldl op x) ≤ key x ∧ (∀ v ∈ l, key (l.foldl op x) ≤ key v) ∧
      (key (l.foldl op x) = key x ∨ ∃ v ∈ l, key (l.foldl op x) = key v) := by
  have hop' : ∀ x v, -key (op x v) = Max.max (-key x) (-key v) := by
    intro x v; rw [hop, max_neg_neg]
  obtain ⟨h1, h2, h3⟩ := foldl_max_spec op (fun y => -key y) hop' l x
  refine ⟨by simpa using h1, fun v hv => by simpa using h2 v hv, ?_⟩
  rcases h3 with h | ⟨v, hv, h⟩
  · exact Or.inl (neg_inj.mp h)
  · exact Or.inr ⟨v, hv, neg_inj.mp h⟩

/-- **Integer `atomic_max` from several programs, in any order.** Suppose each program's part of
the interleaved log is paired with its operands `ops i`, every step on the cell `(d, a)` taking
`op` with its operand (`some v`) and every other step leaving the cell alone (`none`), where `op`
takes the larger by `key` (`IntV.max`, `IntV.toInt`, `IntV.toInt_max`). Then the replay leaves the
cell at a value whose key is at least the initial value's and every operand's, and is one of
them: the largest. -/
theorem replay_max {d : DType} {a : Int} (op : d.Val → d.Val → d.Val) (key : d.Val → ℤ)
    (hop : ∀ x v, key (op x v) = Max.max (key x) (key v)) {N : Nat} {G : List (Fin N × AtomicStep)}
    {ops : Fin N → Multiset d.Val} {m m' : Mem} {x : d.Val} (hrep : Replay m (G.map Prod.snd) m')
    (hp : ∀ i, Paired (fun s o => MapsCell d a (fun y => o.elim y (op y)) s) (proj G i) (ops i))
    (h0 : m.read d a = some x) :
    ∃ y, m'.read d a = some y ∧ key x ≤ key y ∧ (∀ i, ∀ v ∈ ops i, key v ≤ key y) ∧
      (key y = key x ∨ ∃ i, ∃ v ∈ ops i, key y = key v) := by
  obtain ⟨L, hm, ho, hr⟩ := Paired.merge hp
  have hfold := replay_fold (d := d) (a := a) (fun v y => op y v) (hm ▸ hrep) hr h0
  have hmem : ∀ v, v ∈ L.filterMap Prod.snd ↔ ∃ i, v ∈ ops i := by
    intro v
    rw [← Multiset.mem_coe, ho, Multiset.mem_sum]
    simp
  obtain ⟨h1, h2, h3⟩ := foldl_max_spec op key hop (L.filterMap Prod.snd) x
  refine ⟨_, hfold, h1, fun i v hv => h2 v ((hmem v).mpr ⟨i, hv⟩), ?_⟩
  rcases h3 with h | ⟨v, hv, h⟩
  · exact Or.inl h
  · obtain ⟨i, hi⟩ := (hmem v).mp hv
    exact Or.inr ⟨i, v, hi, h⟩

/-- **Integer `atomic_min` from several programs, in any order** (as `replay_max`, with
`IntV.min`, `IntV.toInt_min`): the smallest of the initial value and the operands. -/
theorem replay_min {d : DType} {a : Int} (op : d.Val → d.Val → d.Val) (key : d.Val → ℤ)
    (hop : ∀ x v, key (op x v) = Min.min (key x) (key v)) {N : Nat} {G : List (Fin N × AtomicStep)}
    {ops : Fin N → Multiset d.Val} {m m' : Mem} {x : d.Val} (hrep : Replay m (G.map Prod.snd) m')
    (hp : ∀ i, Paired (fun s o => MapsCell d a (fun y => o.elim y (op y)) s) (proj G i) (ops i))
    (h0 : m.read d a = some x) :
    ∃ y, m'.read d a = some y ∧ key y ≤ key x ∧ (∀ i, ∀ v ∈ ops i, key y ≤ key v) ∧
      (key y = key x ∨ ∃ i, ∃ v ∈ ops i, key y = key v) := by
  have hop' : ∀ x v, -key (op x v) = Max.max (-key x) (-key v) := by
    intro x v; rw [hop, max_neg_neg]
  obtain ⟨y, hy, h1, h2, h3⟩ := replay_max op (fun y => -key y) hop' hrep hp h0
  refine ⟨y, hy, by simpa using h1, fun i v hv => by simpa using h2 i v hv, ?_⟩
  rcases h3 with h | ⟨i, v, hv, h⟩
  · exact Or.inl (neg_inj.mp h)
  · exact Or.inr ⟨i, v, hv, neg_inj.mp h⟩

end TVA
