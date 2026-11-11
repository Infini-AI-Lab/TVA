import TVA.Proof.Shared.SafeA

/-!
# Rules for atomics on shared cells

The `SafeA` rules for a `tl.atomic_*` whose elements are shared cells, its result unused
(`let _ ← tl.atomic_add …`, or `tl.atomic_add_unused …`):

* `SafeA.atomic_add`: `tl.atomic_add(ptrs, vals, mask)`, the common case (split-K, stream-K,
  histograms);
* `SafeA.atomic_add_unused`: the same written `tl.atomic_add_unused` (a source `sem` other than
  `acq_rel`, whose result is unused);
* `SafeA.atomic`: any `tl.atomicWith ptrs upd mask`, which every other `tl.atomic_*` is
  (`atomic_max`, `atomic_min`, `atomic_xchg`, …);
* `SafeA.atomic1`: a scalar pointer without a mask.

Each enabled lane performs one atomic step on its cell, in an order the program does not control
(any permutation of the enabled lanes), finding any old value, writing any new value the update
allows. The rule appends one `AtomicStep` per enabled lane to the log, in that order, and the rest
of the program must be `SafeA` for every such order and values. A lane's step records its cell and
old and new values; for a float addition, `Paired.lanes` (`TVA.Proof.Shared.Replay`) attaches each
lane's operand.

Private atomics (each element owned by one program) are `Safe` rules (`safe_atomic_buf`), lifted
by `SafeA.of_safe`.
-/
namespace TVA

namespace SafeA

variable {E : Env} {R : Set Int} {C : Set (DType × Int)}

/-- The lanes `order` of a tensor atomic on shared cells, each a logged shared step, followed by a
continuation that ignores their old values. -/
private theorem rmwSeq_shared {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {m : Mem} {s : Shape} {d : DType} {ptrs : Tensor s (Ptr d)}
    {upd : Idx s → d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope} {K : Prog α}
    (hsem : sem = .acqRel) (hsc : sc ≠ .cta) (htot : ∀ i old, ∃ new, upd i old new) :
    ∀ (order : List (Idx s)), order.Nodup →
      (∀ i ∈ order, Mem.Aligned d (ptrs.get i).addr ∧ (d, (ptrs.get i).addr) ∈ C) →
      ∀ (acc : Idx s → d.Val) (log : List AtomicStep),
      (∀ olds news : Idx s → d.Val, (∀ i ∈ order, upd i (olds i) (news i)) →
        SafeA E R W C Q (log ++ order.map fun i => ⟨d, (ptrs.get i).addr, olds i, news i⟩) m K) →
      SafeA E R W C Q log m ((tl.rmwSeq ptrs upd sem sc order acc).bind fun _ => K)
  | [], _, _, acc, log, hk => by
    have := hk acc acc (by simp)
    simpa [tl.rmwSeq] using this
  | i :: rest, hnd, hcell, acc, log, hk => by
    have hi : i ∉ rest := (List.nodup_cons.mp hnd).1
    obtain ⟨hal, hC⟩ := hcell i List.mem_cons_self
    simp only [tl.rmwSeq, Prog.bind]
    refine .shared hal hC hsem hsc (htot i)
      (fun v w => tl.rmwSeq_bind_const ptrs upd sem sc K rest _ _) fun old new hon => ?_
    refine rmwSeq_shared hsem hsc htot rest (List.nodup_cons.mp hnd).2
      (fun j hj => hcell j (List.mem_cons_of_mem _ hj)) (Function.update acc i old) _
      fun olds news hup => ?_
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
    rw [List.map_cons, hmap] at hk'
    simpa [List.append_assoc] using hk'

/-- **A tensor atomic on shared cells, result unused** (`let _ ← tl.atomic_…(ptrs, v, mask)`, as
`tl.atomicWith ptrs upd mask`): every enabled lane's element is an aligned cell of `C`, the order
is `acq_rel` at `gpu` or `sys` scope, and the update has a new value for every old value. For
every order of the enabled lanes and all old and new values the update allows, the log grows by
one step per enabled lane, in that order. For `tl.atomic_add`, use `SafeA.atomic_add`. -/
theorem atomic {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {log : List AtomicStep} {m : Mem} {s : Shape} {d : DType} {ptrs : Tensor s (Ptr d)}
    {upd : Idx s → d.Val → d.Val → Prop} {mask : Option (Tensor s Bool)} {sem : MemSem}
    {sc : Scope} {k : Prog α}
    (hcell : ∀ i, (tl.addrs ptrs mask i).isSome = true →
      Mem.Aligned d (ptrs.get i).addr ∧ (d, (ptrs.get i).addr) ∈ C)
    (hsem : sem = .acqRel) (hsc : sc ≠ .cta) (htot : ∀ i old, ∃ new, upd i old new)
    (hk : ∀ order : List (Idx s), order.Perm (enabled (tl.addrs ptrs mask)) →
      ∀ olds news : Idx s → d.Val, (∀ i ∈ order, upd i (olds i) (news i)) →
      SafeA E R W C Q (log ++ order.map fun i => ⟨d, (ptrs.get i).addr, olds i, news i⟩) m k) :
    SafeA E R W C Q log m (tl.atomicWith ptrs upd mask sem sc >>= fun _ => k) := by
  simp only [tl.atomicWith, Prog.bind_eq, Prog.bind, Prog.bind_assoc, Prog.pure_eq]
  refine .pick ⟨fun _ => default, trivial⟩ fun junk _ => ?_
  refine .pick ⟨_, List.Perm.refl _⟩ fun order horder => ?_
  have hnd : order.Nodup := horder.nodup_iff.mpr (enabled_nodup _)
  exact rmwSeq_shared hsem hsc htot order hnd (fun i hi => hcell i
    (by simpa using (List.mem_filter.mp (horder.subset hi)).2)) junk log (hk order horder)

/-- **A scalar atomic on a shared cell, result unused** (`tl.atomic_…(ptr, v)` with a scalar
pointer and no mask, as `tl.atomicWith`): one logged step. -/
theorem atomic1 {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {log : List AtomicStep} {m : Mem} {d : DType} {ptrs : Tensor [] (Ptr d)}
    {upd : Idx [] → d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope} {k : Prog α}
    (hal : Mem.Aligned d (ptrs.get ()).addr) (hC : (d, (ptrs.get ()).addr) ∈ C)
    (hsem : sem = .acqRel) (hsc : sc ≠ .cta) (htot : ∀ old, ∃ new, upd () old new)
    (hk : ∀ old new, upd () old new →
      SafeA E R W C Q (log ++ [⟨d, (ptrs.get ()).addr, old, new⟩]) m k) :
    SafeA E R W C Q log m (tl.atomicWith ptrs upd none sem sc >>= fun _ => k) := by
  have hen : enabled (tl.addrs ptrs none) = [(() : Idx [])] := by
    simp [enabled, Idx.all, tl.addrs]; rfl
  refine atomic (fun i _ => ⟨hal, hC⟩) hsem hsc (fun i => htot) fun order horder olds news hup => ?_
  rw [hen] at horder
  obtain rfl := List.perm_singleton.mp horder
  exact hk (olds ()) (news ()) (hup () List.mem_cons_self)

/-- **`tl.atomic_add` on shared cells, result unused** (`let _ ← tl.atomic_add(ptrs, vals,
mask)`): `SafeA.atomic` with the update `tl.addUpd E d (vals.get i)` (integers wrap; a float
addition rounds as the platform's `atomicAdd`, total by `Device.AtomicAddTotal`). -/
theorem atomic_add {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {log : List AtomicStep} {m : Mem} {s : Shape} {d : DType} {ptrs : Tensor s (Ptr d)}
    {vals : Tensor s d.Val} {mask : Option (Tensor s Bool)} {sem : MemSem} {sc : Scope}
    {k : Prog α}
    (hcell : ∀ i, (tl.addrs ptrs mask i).isSome = true →
      Mem.Aligned d (ptrs.get i).addr ∧ (d, (ptrs.get i).addr) ∈ C)
    (hsem : sem = .acqRel) (hsc : sc ≠ .cta)
    (htot : ∀ i old, ∃ new, tl.addUpd E d (vals.get i) old new)
    (hk : ∀ order : List (Idx s), order.Perm (enabled (tl.addrs ptrs mask)) →
      ∀ olds news : Idx s → d.Val, (∀ i ∈ order, tl.addUpd E d (vals.get i) (olds i) (news i)) →
      SafeA E R W C Q (log ++ order.map fun i => ⟨d, (ptrs.get i).addr, olds i, news i⟩) m k) :
    SafeA E R W C Q log m (tl.atomic_add ptrs vals mask sem sc >>= fun _ => k) := by
  simp only [tl.atomic_add, Prog.bind_eq, Prog.bind]
  exact .env (atomic hcell hsem hsc htot hk)

/-- **`tl.atomic_add_unused` on shared cells**: `SafeA.atomic_add`, whatever the source's `sem`,
at `gpu` or `sys` scope (where an unused atomic's order is `acq_rel`, `MemSem.unused`). -/
theorem atomic_add_unused {W : Set Int} {α : Type} {Q : α → List AtomicStep → Mem → Prop}
    {log : List AtomicStep} {m : Mem} {s : Shape} {d : DType} {ptrs : Tensor s (Ptr d)}
    {vals : Tensor s d.Val} {mask : Option (Tensor s Bool)} {sem : MemSem} {sc : Scope}
    {k : Prog α}
    (hcell : ∀ i, (tl.addrs ptrs mask i).isSome = true →
      Mem.Aligned d (ptrs.get i).addr ∧ (d, (ptrs.get i).addr) ∈ C)
    (hsc : sc ≠ .cta) (htot : ∀ i old, ∃ new, tl.addUpd E d (vals.get i) old new)
    (hk : ∀ order : List (Idx s), order.Perm (enabled (tl.addrs ptrs mask)) →
      ∀ olds news : Idx s → d.Val, (∀ i ∈ order, tl.addUpd E d (vals.get i) (olds i) (news i)) →
      SafeA E R W C Q (log ++ order.map fun i => ⟨d, (ptrs.get i).addr, olds i, news i⟩) m k) :
    SafeA E R W C Q log m (tl.atomic_add_unused ptrs vals mask sem sc >>= fun _ => k) := by
  have hsem : sem.unused sc = .acqRel := by cases sc <;> first | rfl | exact absurd rfl hsc
  have h := atomic_add (k := k) hcell hsem hsc htot hk
  simpa only [tl.atomic_add_unused, Prog.bind_eq, Prog.bind_assoc, Prog.pure_eq, Prog.ret_bind]
    using h

end SafeA

end TVA
