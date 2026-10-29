import TVA.Lemmas.Shared
import TVA.Lemmas.DotLoop
import TVA.Rules.Num

/-!
# More of the shared-atomics kit (library issue L34)

`TVA.Lemmas.Shared` proves programs that share a region only through atomics (`SafeA`,
`shared`), with a rule for a scalar unmasked atomic (`SafeA.atomic1`) and the accumulation lemma
for `int32` addition (`replay_add`). This module adds:

* **`SafeA.atomic`**: a tensor atomic with an optional mask (`tl.atomicWith ptrs upd mask`,
  which `tl.atomic_max(ptrs, v, mask=m)` and every other `tl.atomic_*` is, except `tl.atomic_add`:
  that is `.env fun e => tl.atomicWith …` with the platform's float addition, so a proof unfolds it
  and applies `SafeA.env` first) on cells of the shared region. Each enabled lane
  adds one step to the log, in an order the program does not control (any permutation of the
  enabled lanes), with any old value; the continuation gets the old values of the enabled lanes
  and unspecified values for the others.
* **The cell after a replay**, for updates that are functions of the cell's value
  (`MapsCell`, `replay_maps`): integer addition of any width (`replay_add_any`, for every
  dtype whose values form a commutative monoid, as every integer dtype's do), with
  `sum_proj_any` (the sum over an interleaved log is the sum over programs). A step of an
  integer addition determines its increment (`new − old`), so weighting each step by a function
  of the step is exact there.
* **Steps paired with their operands** (library issue L34). A logged step records the cell and
  its old and new values, not the operand: two programs adding `1.0` and `0.5` to a `float32`
  cell holding `2^24` log the same step. So float accumulation and `max`/`min` attach the operand
  per position: `Paired R log ops` (each step of the log with an optional operand, `none` for a
  step that leaves the cell alone; `ops` the multiset of the operands), built per program
  (`Paired.lanes` for a tensor atomic's lanes, `Paired.append`), kept by a reordering of the log
  (`Paired.perm`, so a postcondition built from it is `PermInv`), and merged along the
  interleaving that `shared` returns (`Paired.merge`: the whole log's operands are the sum over
  programs of their own). `replay_fold`: the cell is the fold over the operands in the log's
  order.
* **Integer `max`/`min`** of any width (`replay_max`, `replay_min`, from `foldl_max_spec`,
  `foldl_min_spec` with `IntV.max`/`IntV.min` and `IntV.toInt_max`/`toInt_min`): the cell ends at
  the largest (smallest) of its initial value and every program's operands, whatever the order.
* **Float accumulation** (`AccStep`, `replay_err`, `replay_err_shared`): if each step on the
  cell adds its operand `w` up to `ρ·|x + w| + α` (as `Rules.Num.AtomicAddAcc` states for a float
  `atomic_add`, `accStep_atomicAdd`) and every other step leaves the cell alone
  (`accStep_other`), the cell ends within `DotLoop.bound ρ α n (|x₀| + Σ|w|)` of `x₀ + Σ w`, the
  sums over every program's operands and `n` their number, in every order.
-/
namespace TVA

open Classical

/-! ## A tensor atomic on the shared region -/

namespace SafeA

variable {E : Env} {R : Set Int} {C : Set (DType × Int)}

/-- The lanes `order` of a tensor atomic, each a logged shared step. -/
theorem rmwSeq_shared {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop} {m : Mem}
    {s : Shape} {d : DType} {ptrs : Tensor s (Ptr d)} {upd : Idx s → d.Val → d.Val → Prop}
    {sem : MemSem} {sc : Scope} {k : (Idx s → d.Val) → Prog α}
    (hsem : sem = .acqRel) (hsc : sc ≠ .cta) (htot : ∀ i old, ∃ new, upd i old new) :
    ∀ (order : List (Idx s)), order.Nodup →
      (∀ i ∈ order, Mem.Aligned d (ptrs.get i).addr ∧ (d, (ptrs.get i).addr) ∈ C) →
      ∀ (acc : Idx s → d.Val) (log : List AtomicStep),
      (∀ olds news : Idx s → d.Val, (∀ i ∈ order, upd i (olds i) (news i)) →
        SafeA E R W C Q (log ++ order.map fun i => ⟨d, (ptrs.get i).addr, olds i, news i⟩) m
          (k fun i => if i ∈ order then olds i else acc i)) →
      SafeA E R W C Q log m ((tl.rmwSeq ptrs upd sem sc order acc).bind k)
  | [], _, _, acc, log, hk => by
    have := hk acc acc (by simp)
    simpa [tl.rmwSeq, Prog.bind] using this
  | i :: rest, hnd, hcell, acc, log, hk => by
    have hi : i ∉ rest := (List.nodup_cons.mp hnd).1
    have hnd' : rest.Nodup := (List.nodup_cons.mp hnd).2
    obtain ⟨hal, hC⟩ := hcell i (List.mem_cons_self)
    simp only [tl.rmwSeq, Prog.bind]
    refine .shared hal hC hsem hsc (htot i) fun old new hon => ?_
    refine rmwSeq_shared hsem hsc htot rest hnd' (fun j hj => hcell j (List.mem_cons_of_mem _ hj))
      (Function.update acc i old) _ fun olds news hup => ?_
    have hk' := hk (Function.update olds i old) (Function.update news i new) (by
      intro j hj
      rcases List.mem_cons.mp hj with rfl | hj
      · simpa using hon
      · have hne : j ≠ i := fun h => hi (h ▸ hj)
        simpa [Function.update_of_ne hne] using hup j hj)
    have hmap : (rest.map fun j => (⟨d, (ptrs.get j).addr, Function.update olds i old j,
        Function.update news i new j⟩ : AtomicStep)) =
        rest.map fun j => ⟨d, (ptrs.get j).addr, olds j, news j⟩ := by
      refine List.map_congr_left fun j hj => ?_
      have hne : j ≠ i := fun h => hi (h ▸ hj)
      simp [Function.update_of_ne hne]
    have hfun : (fun j => if j ∈ i :: rest then Function.update olds i old j else acc j) =
        fun j => if j ∈ rest then olds j else Function.update acc i old j := by
      funext j
      by_cases hji : j = i
      · subst hji; simp [hi]
      · have : (j ∈ i :: rest) ↔ j ∈ rest := by simp [hji]
        by_cases hj : j ∈ rest <;> simp [hji, hj]
    rw [List.map_cons, hmap, hfun] at hk'
    simpa [List.append_assoc] using hk'

/-- **A tensor atomic on the shared region**, with an optional mask: every enabled lane's
element is a cell of `C`, aligned; the order is `acq_rel` at `gpu` or `sys` scope, and the update
is total. For every order of the enabled lanes, every old value and every new value the update
allows, the log grows by one step per enabled lane, in that order, and the continuation receives
the old values (arbitrary values `junk` on the disabled lanes). For `tl.atomic_add`
(`.env fun e => tl.atomicWith ptrs (fun i => tl.addUpd e d (vals.get i)) mask sem sc`), apply
`SafeA.env` first, then this rule with `upd := fun i => tl.addUpd E d (vals.get i)`. -/
theorem atomic {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep}
    {m : Mem} {s : Shape} {d : DType} {ptrs : Tensor s (Ptr d)} {upd : Idx s → d.Val → d.Val → Prop}
    {mask : Option (Tensor s Bool)} {sem : MemSem} {sc : Scope} {k : Tensor s d.Val → Prog α}
    (hcell : ∀ i, (tl.addrs ptrs mask i).isSome = true →
      Mem.Aligned d (ptrs.get i).addr ∧ (d, (ptrs.get i).addr) ∈ C)
    (hsem : sem = .acqRel) (hsc : sc ≠ .cta) (htot : ∀ i old, ∃ new, upd i old new)
    (hk : ∀ order : List (Idx s), order.Perm ((Idx.all s).filter fun i => (tl.addrs ptrs mask i).isSome) →
      ∀ junk olds news : Idx s → d.Val, (∀ i ∈ order, upd i (olds i) (news i)) →
      SafeA E R W C Q (log ++ order.map fun i => ⟨d, (ptrs.get i).addr, olds i, news i⟩) m
        (k ⟨fun i => if i ∈ order then olds i else junk i⟩)) :
    SafeA E R W C Q log m (tl.atomicWith ptrs upd mask sem sc >>= k) := by
  simp only [tl.atomicWith, Prog.bind_eq, Prog.bind]
  refine .pick ⟨fun _ => default, trivial⟩ fun junk _ => ?_
  refine .pick ⟨_, List.Perm.refl _⟩ fun order horder => ?_
  have hnd : order.Nodup := horder.nodup_iff.mpr ((Idx.nodup_all s).filter _)
  rw [Prog.bind_assoc]
  refine rmwSeq_shared hsem hsc htot order hnd (fun i hi => hcell i
      (by simpa using (List.mem_filter.mp (horder.subset hi)).2)) junk log fun olds news hup => ?_
  simpa [pure, Prog.bind] using hk order horder junk olds news hup

end SafeA

/-! ## The cell after a replay -/

/-- Step `s` changes the cell `(d, a)` from `x` to `g x`. -/
def MapsCell (d : DType) (a : Int) (g : d.Val → d.Val) (s : AtomicStep) : Prop :=
  ∀ (m : Mem) (x : d.Val), m.read d a = some x → m.read s.d s.a = some s.old →
    (m.write s.d s.a s.new).read d a = some (g x)

/-- A step on the cell itself whose new value is `g` of its old value. -/
theorem mapsCell_self {d : DType} {a : Int} {old new : d.Val} {g : d.Val → d.Val} (hg : g old = new) :
    MapsCell d a g ⟨d, a, old, new⟩ := by
  intro m x hx hold
  simp only at hold
  rw [hx] at hold
  cases hold
  rw [Mem.read_write_self (Mem.writable_of_read hx), hg]

/-- A step on other bytes leaves the cell alone. -/
theorem mapsCell_other {d : DType} {a : Int} {s : AtomicStep}
    (hdis : Disjoint (Mem.span s.d s.a) (Mem.span d a)) : MapsCell d a id s := by
  intro m x hx _
  rw [Mem.read_congr (fun y hy => Mem.write_frame _ _ _ _ (fun h => Set.disjoint_left.mp hdis h hy)), hx]
  rfl

/-- **The cell after a replay**: the left fold of the steps' maps. -/
theorem replay_maps {d : DType} {a : Int} {g : AtomicStep → d.Val → d.Val} {m m' : Mem}
    {l : List AtomicStep} {x : d.Val} (hrep : Replay m l m') (hl : ∀ s ∈ l, MapsCell d a (g s) s)
    (h0 : m.read d a = some x) : m'.read d a = some (l.foldl (fun x s => g s x) x) := by
  induction hrep generalizing x with
  | nil => simpa using h0
  | @cons m₀ m₁ s rest hr _ ih =>
    have h1 := hl s List.mem_cons_self m₀ x h0 hr
    exact ih (fun s' hs' => hl s' (List.mem_cons_of_mem _ hs')) h1

/-- A left fold of additions is the sum. -/
theorem foldl_add {β : Type} [AddCommMonoid β] (inc : AtomicStep → β) :
    ∀ (l : List AtomicStep) (x : β), l.foldl (fun x s => x + inc s) x = x + (l.map inc).sum
  | [], x => by simp
  | s :: l, x => by rw [List.foldl_cons, foldl_add inc l, List.map_cons, List.sum_cons, add_assoc]

/-- **Integer accumulation of any width.** When every step adds `inc s` to the cell (`inc s = 0`
for a step on other bytes), the cell ends at its initial value plus the sum of the increments,
whatever the interleaving. It holds for every dtype whose values form a commutative monoid, as
those of every integer dtype do (wrapping addition, `IntV`). -/
theorem replay_add_any {d : DType} [AddCommMonoid d.Val] {a : Int} {inc : AtomicStep → d.Val}
    {m m' : Mem} {l : List AtomicStep} {x : d.Val} (hrep : Replay m l m')
    (hl : ∀ s ∈ l, MapsCell d a (· + inc s) s) (h0 : m.read d a = some x) :
    m'.read d a = some (x + (l.map inc).sum) := by
  rw [← foldl_add]
  exact replay_maps hrep hl h0

theorem mapsCell_other_add {d : DType} [AddCommMonoid d.Val] {a : Int} {s : AtomicStep}
    (hdis : Disjoint (Mem.span s.d s.a) (Mem.span d a)) : MapsCell d a (· + 0) s := by
  have h := mapsCell_other (d := d) (a := a) hdis
  intro m x hx hold
  show _ = some (x + 0)
  rw [add_zero]
  exact h m x hx hold

/-- The sum over an interleaved log is the sum over programs of their own parts, in any
commutative monoid. -/
theorem sum_proj_any {β : Type} [AddCommMonoid β] {N : Nat} (G : List (Fin N × AtomicStep))
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
(`Paired.perm`, so a postcondition built from it is `PermInv`); and `Paired.merge` merges the
programs' pairings along the interleaving `G` that `shared` returns, the operands of the whole
log being the sum over programs of their own. -/

/-- A list of pairs whose first components are a permutation of `l'` can be permuted so that
they are exactly `l'`. -/
theorem exists_perm_of_map_perm {α β : Type} (f : α → β) {L : List α} {l' : List β}
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

theorem nil : Paired R [] 0 := ⟨[], rfl, rfl, by simp⟩

/-- Two parts of a log, each paired. -/
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

namespace IntV

variable {sg : Bool} {w : Nat}

theorem lt_toInt (x y : IntV sg w) : x.lt y = decide (x.toInt < y.toInt) := by
  cases sg
  · simp only [IntV.lt, IntV.toInt, Bool.false_eq_true, ite_false]
    rw [BitVec.ult]
    simp
  · simp only [IntV.lt, IntV.toInt, ite_true]
    rw [BitVec.slt]

/-- `<=` of integers of either signedness is the comparison of their values. -/
theorem le_toInt (x y : IntV sg w) : x.le y = decide (x.toInt ≤ y.toInt) := by
  cases sg
  · simp only [IntV.le, IntV.toInt, Bool.false_eq_true, ite_false]
    rw [BitVec.ule]
    simp
  · simp only [IntV.le, IntV.toInt, ite_true]
    rw [BitVec.sle]

/-- `IntV.max` is the larger value. -/
theorem toInt_max (x y : IntV sg w) : (x.max y).toInt = Max.max x.toInt y.toInt := by
  unfold IntV.max
  rw [lt_toInt]
  by_cases h : x.toInt < y.toInt
  · simp [h, max_eq_right h.le]
  · simp [h, max_eq_left (not_lt.mp h)]

/-- `IntV.min` is the smaller value. -/
theorem toInt_min (x y : IntV sg w) : (x.min y).toInt = Min.min x.toInt y.toInt := by
  unfold IntV.min
  rw [lt_toInt]
  by_cases h : x.toInt < y.toInt
  · simp [h, min_eq_left h.le]
  · simp [h, min_eq_right (not_lt.mp h)]

end IntV

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

/-! ## Float accumulation -/

/-- Step `s`, applied to the cell `(d, a)` holding `x`, leaves a value `y` with `r x y`. -/
def StepRel (d : DType) (a : Int) (r : d.Val → d.Val → Prop) (s : AtomicStep) : Prop :=
  ∀ (m : Mem) (x : d.Val), m.read d a = some x → m.read s.d s.a = some s.old →
    ∃ y, (m.write s.d s.a s.new).read d a = some y ∧ r x y

/-- A step on the cell itself relates its old and new values. -/
theorem stepRel_self {d : DType} {a : Int} {old new : d.Val} {r : d.Val → d.Val → Prop}
    (h : r old new) : StepRel d a r ⟨d, a, old, new⟩ := by
  intro m x hx hold
  simp only at hold
  rw [hx] at hold
  cases hold
  exact ⟨new, Mem.read_write_self (Mem.writable_of_read hx) new, h⟩

/-- A step on other bytes leaves the cell's value, which `r` must allow. -/
theorem stepRel_other {d : DType} {a : Int} {s : AtomicStep} {r : d.Val → d.Val → Prop}
    (hr : ∀ x, r x x) (hdis : Disjoint (Mem.span s.d s.a) (Mem.span d a)) : StepRel d a r s := by
  intro m x hx _
  refine ⟨x, ?_, hr x⟩
  rw [Mem.read_congr (fun y hy => Mem.write_frame _ _ _ _ (fun h => Set.disjoint_left.mp hdis h hy)), hx]

/-- What one accumulation step promises: from a `good` value `x`, when `|x + w| ≤ B`, a `good`
value within `ρ·|x + w| + α` of `x + w`. -/
def AddsApprox {β : Type} (val : β → ℚ) (good : β → Prop) (ρ α B w : ℚ) (x y : β) : Prop :=
  good x → |val x + w| ≤ B → good y ∧ |val y - (val x + w)| ≤ ρ * |val x + w| + α

/-- A step that leaves the value adds `0` exactly. -/
theorem addsApprox_refl {β : Type} (val : β → ℚ) (good : β → Prop) {ρ α B : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α)
    (x : β) : AddsApprox val good ρ α B 0 x x := fun hx _ =>
  ⟨hx, by rw [add_zero, sub_self, abs_zero]; positivity⟩

/-- One logged step with its operand, for float accumulation into the cell `(d, a)`: with
`some w`, the step adds `w` to the cell up to `ρ·|x + w| + α` (`AddsApprox`; `accStep_atomicAdd`);
with `none`, it leaves the cell alone (`accStep_other`). -/
def AccStep (d : DType) (a : Int) (val : d.Val → ℚ) (good : d.Val → Prop) (ρ α B : ℚ) :
    AtomicStep → Option ℚ → Prop
  | s, none => MapsCell d a id s
  | s, some w => StepRel d a (AddsApprox val good ρ α B w) s

/-- A step on other bytes is an `AccStep` without operand. -/
theorem accStep_other {d : DType} {a : Int} {val : d.Val → ℚ} {good : d.Val → Prop} {ρ α B : ℚ}
    {s : AtomicStep} (hdis : Disjoint (Mem.span s.d s.a) (Mem.span d a)) :
    AccStep d a val good ρ α B s none :=
  mapsCell_other hdis

private theorem bound_mono {ρ α : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {t t' : ℕ} {B B' : ℚ} (ht : t ≤ t')
    (hB : 0 ≤ B) (hBB : B ≤ B') : DotLoop.bound ρ α t B ≤ DotLoop.bound ρ α t' B' := by
  unfold DotLoop.bound
  have h1 : 1 ≤ (1 + ρ) ^ t := one_le_pow₀ (by linarith)
  have h2 : (1 + ρ) ^ t ≤ (1 + ρ) ^ t' := pow_le_pow_right₀ (by linarith) ht
  have ht' : (t : ℚ) ≤ t' := by exact_mod_cast ht
  have e1 : ((1 + ρ) ^ t - 1) * B ≤ ((1 + ρ) ^ t' - 1) * B' := mul_le_mul (by linarith) hBB hB (by linarith)
  have e2 : (t : ℚ) * (1 + ρ) ^ t * α ≤ t' * (1 + ρ) ^ t' * α :=
    mul_le_mul_of_nonneg_right (mul_le_mul ht' h2 (by positivity) (by positivity)) hα
  linarith

private theorem replay_err_aux {d : DType} {a : Int} (val : d.Val → ℚ) (good : d.Val → Prop)
    {ρ α B : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) (Btot : ℚ) (n : ℕ) :
    ∀ (L : List (AtomicStep × Option ℚ)) (m m' : Mem) (x : d.Val) (t : ℕ) (St Bt : ℚ),
      Replay m (L.map Prod.fst) m' → (∀ p ∈ L, AccStep d a val good ρ α B p.1 p.2) →
      m.read d a = some x → good x → |St| ≤ Bt →
      |val x - St| ≤ DotLoop.bound ρ α t Bt → t + (L.filterMap Prod.snd).length ≤ n →
      Bt + ((L.filterMap Prod.snd).map fun w => |w|).sum ≤ Btot → Btot + DotLoop.bound ρ α n Btot ≤ B →
      ∃ y, m'.read d a = some y ∧ good y ∧
        |val y - (St + (L.filterMap Prod.snd).sum)| ≤
          DotLoop.bound ρ α (t + (L.filterMap Prod.snd).length)
            (Bt + ((L.filterMap Prod.snd).map fun w => |w|).sum)
  | [], m, m', x, t, St, Bt, hrep, _, h0, hx, _, he, _, _, _ => by
    cases hrep
    exact ⟨x, h0, hx, by simpa using he⟩
  | (s, none) :: L, m, m', x, t, St, Bt, hrep, hl, h0, hx, hSt, he, hn, hB, hroom => by
    cases hrep with
    | cons hr hrest =>
      have h1 : (m.write s.d s.a s.new).read d a = some x := hl (s, none) List.mem_cons_self m x h0 hr
      have hc : ((s, none) :: L).filterMap Prod.snd = L.filterMap Prod.snd := rfl
      rw [hc] at hn hB ⊢
      exact replay_err_aux val good hρ hα Btot n L _ m' x t St Bt hrest
        (fun p hp => hl p (List.mem_cons_of_mem _ hp)) h1 hx hSt he hn hB hroom
  | (s, some w) :: L, m, m', x, t, St, Bt, hrep, hl, h0, hx, hSt, he, hn, hB, hroom => by
    cases hrep with
    | cons hr hrest =>
      obtain ⟨y, hy, hxy⟩ := (hl (s, some w) List.mem_cons_self : StepRel d a _ s) m x h0 hr
      have hw0 : 0 ≤ ((L.filterMap Prod.snd).map fun w => |w|).sum :=
        List.sum_nonneg (by simp only [List.mem_map]; rintro _ ⟨w', -, rfl⟩; exact abs_nonneg _)
      have hBt0 : 0 ≤ Bt := (abs_nonneg _).trans hSt
      have hc : ((s, some w) :: L).filterMap Prod.snd = w :: L.filterMap Prod.snd := rfl
      rw [hc] at hB hn ⊢
      simp only [List.map_cons, List.sum_cons, List.length_cons] at hB hn ⊢
      have hxabs : |val x| ≤ Bt + DotLoop.bound ρ α t Bt := by
        have := abs_sub_abs_le_abs_sub (val x) St
        linarith
      have hbt : DotLoop.bound ρ α t Bt ≤ DotLoop.bound ρ α n Btot :=
        bound_mono hρ hα (by omega) hBt0 (by linarith [abs_nonneg w])
      have hroom1 : |val x + w| ≤ B := by
        have := abs_add_le (val x) w
        linarith [abs_nonneg w]
      obtain ⟨hgy, herr⟩ := hxy hx hroom1
      have hstep : |val y - (St + w)| ≤ DotLoop.bound ρ α (t + 1) (Bt + |w|) :=
        DotLoop.accum_step hρ hα hSt le_rfl he (by
          have := abs_add_le (val x) w
          have : ρ * |val x + w| ≤ ρ * (|val x| + |w|) := mul_le_mul_of_nonneg_left this hρ
          linarith)
      have hSt' : |St + w| ≤ Bt + |w| := (abs_add_le _ _).trans (by linarith)
      obtain ⟨z, hz, hgz, hze⟩ := replay_err_aux val good hρ hα Btot n L _ m' y (t + 1) (St + w)
        (Bt + |w|) hrest (fun p hp => hl p (List.mem_cons_of_mem _ hp)) hy hgy hSt' hstep
        (by omega) (by linarith) hroom
      refine ⟨z, hz, hgz, ?_⟩
      have e1 : St + (w + (L.filterMap Prod.snd).sum) = St + w + (L.filterMap Prod.snd).sum := by ring
      have e2 : Bt + (|w| + ((L.filterMap Prod.snd).map fun w => |w|).sum) =
          Bt + |w| + ((L.filterMap Prod.snd).map fun w => |w|).sum := by ring
      rw [e1, e2, show t + ((L.filterMap Prod.snd).length + 1) = t + 1 + (L.filterMap Prod.snd).length by omega]
      exact hze

/-- **Float accumulation through atomics, per position.** If every step of a replay, paired with
its operand, either adds `w` to the cell up to `ρ·|x + w| + α` (`some w`; `accStep_atomicAdd`
from `Rules.Num.AtomicAddAcc`) or leaves it alone (`none`; `accStep_other`), the cell ends within
`DotLoop.bound ρ α n (|x₀| + Σ|w|)` of `x₀ + Σ w`, the sums over the operands of the steps on the
cell, `L.filterMap Prod.snd` (the `w` of each step paired with `some w`), and `n` their number, in
whatever order the programs' atomics interleaved, given room for that bound (`… ≤ B`). Across
programs, `replay_err_shared`. -/
theorem replay_err {d : DType} {a : Int} (val : d.Val → ℚ) (good : d.Val → Prop)
    {ρ α B : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {m m' : Mem} {L : List (AtomicStep × Option ℚ)}
    {x : d.Val} (hrep : Replay m (L.map Prod.fst) m')
    (hl : ∀ p ∈ L, AccStep d a val good ρ α B p.1 p.2) (h0 : m.read d a = some x) (hx : good x)
    (hroom : |val x| + ((L.filterMap Prod.snd).map fun w => |w|).sum +
      DotLoop.bound ρ α (L.filterMap Prod.snd).length
        (|val x| + ((L.filterMap Prod.snd).map fun w => |w|).sum) ≤ B) :
    ∃ y, m'.read d a = some y ∧ good y ∧
      |val y - (val x + (L.filterMap Prod.snd).sum)| ≤
        DotLoop.bound ρ α (L.filterMap Prod.snd).length
          (|val x| + ((L.filterMap Prod.snd).map fun w => |w|).sum) := by
  obtain ⟨y, hy, hgy, he⟩ := replay_err_aux val good hρ hα _ (L.filterMap Prod.snd).length L m m' x 0
    (val x) |val x| hrep hl h0 hx le_rfl (by simp [DotLoop.bound]) (by omega) le_rfl hroom
  exact ⟨y, hy, hgy, by simpa using he⟩

/-- **Float accumulation from several programs, in any order.** Suppose each program's part of
the interleaved log `G` (from `shared`) is paired with its operands `ops i` by `AccStep`: its
steps on the cell `(d, a)` each add their operand up to `ρ·|x + w| + α`, its other steps leave the
cell alone. Then the replay leaves the cell within `DotLoop.bound ρ α n (|x₀| + A)` of
`x₀ + S`, where `S = Σᵢ Σ (ops i)` is the sum of every program's operands, `A` the sum of their
magnitudes and `n` their number, given room for that bound. -/
theorem replay_err_shared {d : DType} {a : Int} (val : d.Val → ℚ) (good : d.Val → Prop)
    {ρ α B : ℚ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {N : Nat} {G : List (Fin N × AtomicStep)}
    {ops : Fin N → Multiset ℚ} {m m' : Mem} {x : d.Val} (hrep : Replay m (G.map Prod.snd) m')
    (hp : ∀ i, Paired (AccStep d a val good ρ α B) (proj G i) (ops i))
    (h0 : m.read d a = some x) (hx : good x)
    (hroom : |val x| + (∑ i, ((ops i).map fun w => |w|).sum) +
      DotLoop.bound ρ α (∑ i, Multiset.card (ops i))
        (|val x| + ∑ i, ((ops i).map fun w => |w|).sum) ≤ B) :
    ∃ y, m'.read d a = some y ∧ good y ∧
      |val y - (val x + ∑ i, (ops i).sum)| ≤
        DotLoop.bound ρ α (∑ i, Multiset.card (ops i)) (|val x| + ∑ i, ((ops i).map fun w => |w|).sum) := by
  obtain ⟨L, hm, ho, hr⟩ := Paired.merge hp
  have hS : (L.filterMap Prod.snd).sum = ∑ i, (ops i).sum := by
    rw [← Multiset.sum_coe, ho]
    exact map_sum Multiset.sumAddMonoidHom ops Finset.univ
  have hA : ((L.filterMap Prod.snd).map fun w => |w|).sum = ∑ i, ((ops i).map fun w => |w|).sum := by
    rw [← Multiset.sum_coe, ← Multiset.map_coe, ho,
      show Multiset.map (fun w => |w|) (∑ i, ops i) = ∑ i, (ops i).map fun w => |w| from
        map_sum (Multiset.mapAddMonoidHom fun w : ℚ => |w|) ops Finset.univ]
    exact map_sum Multiset.sumAddMonoidHom _ Finset.univ
  have hn : (L.filterMap Prod.snd).length = ∑ i, Multiset.card (ops i) := by
    rw [← Multiset.coe_card, ho, Multiset.card_sum]
  have h := replay_err (a := a) val good hρ hα (hm ▸ hrep) hr h0 hx (by rw [hA, hn]; exact hroom)
  rwa [hS, hA, hn] at h

/-- **A float `atomic_add` step on the cell**, from `Rules.Num.AtomicAddAcc`: a `float32` step
`⟨.f32, a, old, new⟩` whose new value the platform's `atomicAdd` gave for the finite operand `v`
adds `v` up to `rel·|x + v| + abs`, with room `maxVal`. -/
theorem stepRel_atomicAdd {P : Platform} {rel abs : ℚ}
    (hacc : Rules.Num.AtomicAddAcc P .binary32 rel abs) {a : Int} {old v new : F32}
    (hnew : P.atomicAdd .binary32 old v new) (hv : v.isFinite = true) :
    StepRel .f32 a (AddsApprox FloatV.toRat (fun x : F32 => x.isFinite = true) rel abs
      TVA.Math.Float.Format.binary32.maxVal v.toRat) ⟨.f32, a, old, new⟩ :=
  stepRel_self fun ho hroom => hacc old v new hnew ho hv hroom

/-- `stepRel_atomicAdd` as an `AccStep` with the operand `v`. -/
theorem accStep_atomicAdd {P : Platform} {rel abs : ℚ}
    (hacc : Rules.Num.AtomicAddAcc P .binary32 rel abs) {a : Int} {old v new : F32}
    (hnew : P.atomicAdd .binary32 old v new) (hv : v.isFinite = true) :
    AccStep .f32 a FloatV.toRat (fun x : F32 => x.isFinite = true) rel abs
      TVA.Math.Float.Format.binary32.maxVal ⟨.f32, a, old, new⟩ (some v.toRat) :=
  stepRel_atomicAdd hacc hnew hv

end TVA
