import TVA.Lemmas.Buf
import TVA.Lemmas.Ops

/-!
# Accesses to buffer elements

Most Triton loads and stores touch elements of one buffer: lane `i` addresses element `g i` of
`b` when the mask enables it (`en i`). `Buf.Addrs b addrs g en` says exactly that about the
lanes' addresses `addrs` (`tl.addrs ptrs mask`). From it:

* the footprint is `b.elems (Buf.img g en)`: the bytes of the elements touched
  (`accBytes_of`);
* `safe_load_buf`: a load gives each enabled lane its element and each disabled lane `other`;
* `safe_store_buf`: a store to distinct elements leaves each one holding its value, nothing else
  changed, and every present byte still present.

An access pattern (a 1-D block, a row of a matrix, a 2-D tile: `TVA.Lemmas.Block`) then only
proves `Buf.Addrs` once, by computing `g` and `en`.
-/
namespace TVA

namespace Buf

variable {d : DType} {s : Shape}

/-- The bytes of the elements `S` of `b`. -/
def elems (b : Buf d) (S : Set Nat) : Set Int :=
  {x | ∃ j ∈ S, x ∈ Mem.span d (b.ptr (j : Int)).addr}

/-- The elements an access touches: `g i` for the enabled lanes. -/
def img (g : Idx s → Nat) (en : Idx s → Bool) : Set Nat := {j | ∃ i, en i = true ∧ g i = j}

/-- Lane `i` addresses element `g i` of `b` when `en i`, and no address otherwise. -/
def Addrs (b : Buf d) (addrs : Idx s → Option Int) (g : Idx s → Nat) (en : Idx s → Bool) : Prop :=
  ∀ i, addrs i = if en i then some (b.ptr (g i : Int)).addr else none

@[simp] theorem ptr_offset (b : Buf d) (i k : Int) : (b.ptr i).offset k = b.ptr (i + k) := by
  simp only [ptr, Ptr.offset]; congr 1; ring

/-- **From lanes to `Addrs`.** Show, lane by lane, that an enabled lane points at element `g i`
and that the mask is `en`. -/
theorem addrs_of_get (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Tensor s Bool} {g : Idx s → Nat}
    {en : Idx s → Bool} (hm : ∀ i, mask.get i = en i) (hp : ∀ i, en i = true → ptrs.get i = b.ptr (g i : Int)) :
    b.Addrs (tl.addrs ptrs (some mask)) g en := by
  intro i
  simp only [tl.addrs, hm i]
  split_ifs with he
  · rw [hp i he]
  · rfl

/-- The same without a mask: every lane is enabled. -/
theorem addrs_of_get_all (b : Buf d) {ptrs : Tensor s (Ptr d)} {g : Idx s → Nat}
    (hp : ∀ i, ptrs.get i = b.ptr (g i : Int)) : b.Addrs (tl.addrs ptrs none) g (fun _ => true) := by
  intro i
  simp only [tl.addrs, hp i, if_true]

theorem span_sub_elems (b : Buf d) {S : Set Nat} {j : Nat} (hj : j ∈ S) :
    Mem.span d (b.ptr (j : Int)).addr ⊆ b.elems S := fun _ hx => ⟨j, hj, hx⟩

theorem elems_mono (b : Buf d) {S T : Set Nat} (h : S ⊆ T) : b.elems S ⊆ b.elems T := by
  rintro x ⟨j, hj, hx⟩; exact ⟨j, h hj, hx⟩

theorem elems_union (b : Buf d) (S T : Set Nat) : b.elems (S ∪ T) = b.elems S ∪ b.elems T := by
  ext x; constructor
  · rintro ⟨j, hj | hj, hx⟩
    · exact Or.inl ⟨j, hj, hx⟩
    · exact Or.inr ⟨j, hj, hx⟩
  · rintro (⟨j, hj, hx⟩ | ⟨j, hj, hx⟩)
    · exact ⟨j, Or.inl hj, hx⟩
    · exact ⟨j, Or.inr hj, hx⟩

theorem elems_sub_bytes (b : Buf d) {S : Set Nat} (hS : ∀ j ∈ S, j < b.len) : b.elems S ⊆ b.bytes := by
  rintro x ⟨j, hj, hx⟩; exact b.span_sub (hS j hj) hx

/-- Different elements of one buffer do not overlap. -/
theorem elems_disjoint (b : Buf d) {S T : Set Nat} (hST : Disjoint S T) :
    Disjoint (b.elems S) (b.elems T) := by
  rw [Set.disjoint_left]
  rintro x ⟨j, hj, hx⟩ ⟨j', hj', hx'⟩
  by_cases h : j = j'
  · subst h; exact Set.disjoint_left.mp hST hj hj'
  · exact Set.disjoint_left.mp (b.span_disjoint (by exact_mod_cast h)) hx hx'

/-- Elements of disjoint buffers do not overlap. -/
theorem elems_disjoint_of {d' : DType} (b : Buf d) (c : Buf d') (hbc : Disjoint b.bytes c.bytes)
    {S T : Set Nat} (hS : ∀ j ∈ S, j < b.len) (hT : ∀ j ∈ T, j < c.len) :
    Disjoint (b.elems S) (c.elems T) :=
  hbc.mono (b.elems_sub_bytes hS) (c.elems_sub_bytes hT)

theorem Addrs.accBytes {b : Buf d} {addrs : Idx s → Option Int} {g : Idx s → Nat}
    {en : Idx s → Bool} (h : b.Addrs addrs g en) : accBytes d addrs = b.elems (img g en) := by
  ext x
  constructor
  · rintro ⟨i, a, ha, hx⟩
    rw [h i] at ha
    split_ifs at ha with he
    · cases ha; exact ⟨_, ⟨i, he, rfl⟩, hx⟩
  · rintro ⟨_, ⟨i, he, rfl⟩, hx⟩
    exact ⟨i, _, by rw [h i, if_pos he], hx⟩

theorem Addrs.distinct {b : Buf d} {addrs : Idx s → Option Int} {g : Idx s → Nat}
    {en : Idx s → Bool} (h : b.Addrs addrs g en)
    (hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j) : DistinctLanes d addrs := by
  intro i j a a' hne ha ha'
  rw [h i] at ha; rw [h j] at ha'
  split_ifs at ha ha' with hi hj <;> cases ha <;> cases ha'
  exact b.span_disjoint (fun e => hne (hinj i j hi hj (by exact_mod_cast e)))

end Buf

/-! ## Load and store rules -/

section Rules
variable {d : DType} {s : Shape} {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop}
  {m : Mem}

/-- **Load of buffer elements.** Each enabled lane gets its element; a disabled lane gets
`other` when given. -/
theorem safe_load_buf (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Option (Tensor s Bool)}
    {other : Option (Tensor s d.Val)} {g : Idx s → Nat} {en : Idx s → Bool}
    (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len)
    (hv : b.Valid m) (hR : b.elems (Buf.img g en) ⊆ R) {k : Tensor s d.Val → Prog α}
    (hk : ∀ v : Tensor s d.Val,
      (∀ i, en i = true → m.read d (b.ptr (g i : Int)).addr = some (v.get i)) →
      (∀ i, en i = false → ∀ o, other = some o → v.get i = o.get i) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs mask other >>= k) := by
  refine safe_load ?_ (by rw [hA.accBytes]; exact hR) (fun v hv' => hk v ?_ ?_)
  · intro i a ha
    rw [hA i] at ha
    split_ifs at ha with he
    · cases ha; exact b.read_some hv (hin i he)
  · intro i he
    exact (hv' i).1 _ (by rw [hA i, if_pos he])
  · intro i he o ho
    exact (hv' i).2 (by rw [hA i, he]; rfl) o ho

/-- **Store to distinct buffer elements.** Afterwards each enabled lane's element holds its
value, nothing outside the elements changed, and every present byte is still present. -/
theorem safe_store_buf (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Option (Tensor s Bool)}
    {vals : Tensor s d.Val} {g : Idx s → Nat} {en : Idx s → Bool}
    (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len)
    (hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j)
    (hv : b.Valid m) (hW : b.elems (Buf.img g en) ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m', (∀ i, en i = true → m'.read d (b.ptr (g i : Int)).addr = some (vals.get i)) →
      (∀ x, x ∉ b.elems (Buf.img g en) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R W Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals mask >>= k) := by
  have hok : StoreOk m d (tl.addrs ptrs mask) := by
    intro i a ha
    rw [hA i] at ha
    split_ifs at ha with he
    · cases ha; exact b.writable hv (hin i he)
  refine safe_store_any hok (by rw [hA.accBytes]; exact hW) (fun order hperm => hk _ ?_ ?_ ?_)
  · intro i he
    have ha : tl.addrs ptrs mask i = some (b.ptr (g i : Int)).addr := by rw [hA i, if_pos he]
    exact writeAll_read (hA.distinct hinj) hok order (hperm.nodup_iff.mpr (enabled_nodup _)) i _
      (hperm.mem_iff.mpr (mem_enabled ha)) ha
  · intro x hx
    rw [← hA.accBytes] at hx
    exact writeAll_frame _ _ _ _ hx
  · intro x hx
    exact writeAll_present _ _ _ hx

/-! ## Padding -/

/-- A padding value exists: always for `undef` and `zero`, and for `nan` on float dtypes. -/
theorem Padding.exists_holds {d : DType} (p : Padding) (h : p = .nan → d.format?.isSome = true) :
    ∃ z : d.Val, p.Holds z := by
  cases p with
  | undef => exact ⟨default, trivial⟩
  | zero => exact ⟨d.ofBits 0, rfl⟩
  | nan => exact d.exists_nan (h rfl)

/-- **Padding.** In-bounds lanes keep the loaded value; the others hold the padding. -/
theorem safe_pad {p : Padding} {inb : Idx s → Bool} {v : Tensor s d.Val} {k : Tensor s d.Val → Prog α}
    (hz : ∃ z : d.Val, p.Holds z)
    (hk : ∀ r : Tensor s d.Val, (∀ i, inb i = true → r.get i = v.get i) →
      (∀ i, inb i = false → p.Holds (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.pad p inb v >>= k) := by
  obtain ⟨z, hz⟩ := hz
  simp only [tl.pad, Prog.bind_eq, Prog.bind]
  refine .pick ⟨⟨fun i => if inb i then v.get i else z⟩, fun i => ?_⟩
    fun r hr => hk r (fun i hi => ?_) (fun i hi => ?_)
  · by_cases h : inb i = true <;> simp [h, hz]
  · have := hr i; rw [if_pos hi] at this; exact this
  · have := hr i; rw [if_neg (by simp [hi])] at this; exact this

/-! ## Atomics on owned elements -/

theorem addrs_some {ptrs : Tensor s (Ptr d)} {mask : Option (Tensor s Bool)} {i : Idx s} {a : Int}
    (h : tl.addrs ptrs mask i = some a) : a = (ptrs.get i).addr := by
  unfold tl.addrs at h
  cases mask with
  | none => simp only [Option.some.injEq] at h; exact h.symm
  | some mk =>
    by_cases hm : mk.get i = true
    · simp only [hm, ite_true, Option.some.injEq] at h; exact h.symm
    · simp [hm] at h

/-- The read-modify-writes of the lanes `L`, each on its own element of `b`. -/
private theorem safe_rmwSeq_buf (b : Buf d) {ptrs : Tensor s (Ptr d)}
    {upd : Idx s → d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope} {g : Idx s → Nat}
    {β : Type} {Q' : β → Mem → Prop} (htot : ∀ i o, ∃ n, upd i o n) :
    ∀ (L : List (Idx s)), L.Nodup →
    (∀ i ∈ L, (ptrs.get i).addr = (b.ptr (g i : Int)).addr ∧ g i < b.len) →
    (∀ i ∈ L, ∀ j ∈ L, g i = g j → i = j) →
    (∀ i ∈ L, Mem.span d (b.ptr (g i : Int)).addr ⊆ R ∧ Mem.span d (b.ptr (g i : Int)).addr ⊆ W) →
    ∀ (acc : Idx s → d.Val) (m₀ : Mem) {f : (Idx s → d.Val) → Prog β}, b.Valid m₀ →
    (∀ acc' m', (∀ i ∈ L, ∃ n, m₀.read d (b.ptr (g i : Int)).addr = some (acc' i) ∧ upd i (acc' i) n ∧
        m'.read d (b.ptr (g i : Int)).addr = some n) →
      (∀ i, i ∉ L → acc' i = acc i) →
      (∀ x, (∀ i ∈ L, x ∉ Mem.span d (b.ptr (g i : Int)).addr) → m' x = m₀ x) →
      b.Valid m' → (∀ x, m₀ x ≠ none → m' x ≠ none) → Safe E R W Q' m' (f acc')) →
    Safe E R W Q' m₀ ((tl.rmwSeq ptrs upd sem sc L acc).bind f)
  | [], _, _, _, _, acc, m₀, f, hv, hf => by
    simp only [tl.rmwSeq, Prog.ret_bind]
    exact hf acc m₀ (fun _ h => absurd h List.not_mem_nil) (fun _ _ => rfl) (fun _ _ => rfl) hv
      (fun _ h => h)
  | i :: rest, hnd, hL, hinj, hRW, acc, m₀, f, hv, hf => by
    have hi := hL i List.mem_cons_self
    have hrestL : ∀ j ∈ rest, (ptrs.get j).addr = (b.ptr (g j : Int)).addr ∧ g j < b.len :=
      fun j hj => hL j (List.mem_cons_of_mem _ hj)
    have hni : i ∉ rest := (List.nodup_cons.mp hnd).1
    -- the lanes of `rest` touch elements other than `i`'s
    have hdis : ∀ j ∈ rest, Disjoint (Mem.span d (b.ptr (g i : Int)).addr) (Mem.span d (b.ptr (g j : Int)).addr) :=
      fun j hj => b.span_disjoint fun e => hni (by
        have := hinj i List.mem_cons_self j (List.mem_cons_of_mem _ hj) (by exact_mod_cast e)
        exact this ▸ hj)
    simp only [tl.rmwSeq, Prog.bind]
    rw [hi.1]
    obtain ⟨o, ho⟩ := b.read_some hv hi.2
    obtain ⟨n, hn⟩ := htot i o
    refine .rmw ⟨o, ho, n, hn⟩ (hRW i List.mem_cons_self).1 (hRW i List.mem_cons_self).2
      fun old new hold hnew => ?_
    have hw := b.writable hv hi.2
    have hv₁ : b.Valid (m₀.write d (b.ptr (g i : Int)).addr new) :=
      Buf.valid_of_agree hv fun x hx => Mem.write_present _ (hv.2.2.2 x hx)
    refine safe_rmwSeq_buf b htot rest (List.nodup_cons.mp hnd).2 hrestL
      (fun j hj j' hj' => hinj j (List.mem_cons_of_mem _ hj) j' (List.mem_cons_of_mem _ hj'))
      (fun j hj => hRW j (List.mem_cons_of_mem _ hj)) _ _ hv₁
      fun acc' m' hdone hother hframe hv' hpres => hf acc' m' ?_ ?_ ?_ hv' ?_
    · intro j hj
      rcases List.mem_cons.mp hj with rfl | hj
      · refine ⟨new, ?_, ?_, ?_⟩
        · rw [hother j hni, Function.update_self]; exact hold
        · rw [hother j hni, Function.update_self]; exact hnew
        · rw [Mem.read_congr (m₂ := m₀.write d (b.ptr (g j : Int)).addr new)
            fun x hx => hframe x fun j' hj' hx' => Set.disjoint_left.mp (hdis j' hj') hx hx']
          exact Mem.read_write_self hw new
      · obtain ⟨n', h1, h2, h3⟩ := hdone j hj
        exact ⟨n', by rw [← h1, Mem.read_write_other _ (hdis j hj)], h2, h3⟩
    · intro j hj
      rw [hother j fun h => hj (List.mem_cons_of_mem _ h)]
      exact Function.update_of_ne (fun e => hj (by rw [e]; exact List.mem_cons_self)) _ _
    · intro x hx
      rw [hframe x fun j hj => hx j (List.mem_cons_of_mem _ hj)]
      exact Mem.write_frame _ _ _ _ (hx i List.mem_cons_self)
    · intro x hx
      exact hpres x (Mem.write_present _ hx)

/-- **Tensor atomic on owned buffer elements.** Each enabled lane updates its own element of
`b` once, from the element's value before the atomic. The rule gives the old values (the
result) and the new contents; nothing outside the elements changes. This covers atomics on
memory no other program touches (`TVA.Lemmas.Independent`); shared atomics need
`TVA.Lemmas.Shared`. -/
theorem safe_atomic_buf (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Option (Tensor s Bool)}
    {upd : Idx s → d.Val → d.Val → Prop} {sem : MemSem} {sc : Scope} {g : Idx s → Nat}
    {en : Idx s → Bool}
    (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len)
    (hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j)
    (htot : ∀ i o, ∃ n, upd i o n) (hv : b.Valid m)
    (hR : b.elems (Buf.img g en) ⊆ R) (hW : b.elems (Buf.img g en) ⊆ W)
    {k : Tensor s d.Val → Prog α}
    (hk : ∀ (old : Tensor s d.Val) (m' : Mem),
      (∀ i, en i = true → ∃ n, m.read d (b.ptr (g i : Int)).addr = some (old.get i) ∧
        upd i (old.get i) n ∧ m'.read d (b.ptr (g i : Int)).addr = some n) →
      (∀ x, x ∉ b.elems (Buf.img g en) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R W Q m' (k old)) :
    Safe E R W Q m (tl.atomicWith ptrs upd mask sem sc >>= k) := by
  have hen : ∀ i, i ∈ enabled (tl.addrs ptrs mask) ↔ en i = true := by
    intro i
    simp only [enabled, List.mem_filter, Idx.mem_all, true_and]
    rw [hA i]; cases en i <;> simp
  have hptr : ∀ i, en i = true → (ptrs.get i).addr = (b.ptr (g i : Int)).addr := fun i he =>
    (addrs_some (by rw [hA i, if_pos he])).symm
  simp only [tl.atomicWith, Prog.bind_eq, Prog.bind, Prog.bind_assoc, Prog.pure_eq]
  refine .pick ⟨fun _ => default, trivial⟩ fun junk _ => ?_
  refine .pick ⟨_, List.Perm.refl _⟩ fun order hperm => ?_
  have hmem : ∀ i, i ∈ order ↔ en i = true := fun i => (hperm.mem_iff).trans (hen i)
  refine safe_rmwSeq_buf b htot order (hperm.nodup_iff.mpr (enabled_nodup _))
    (fun i hi => ⟨hptr i ((hmem i).mp hi), hin i ((hmem i).mp hi)⟩)
    (fun i hi j hj => hinj i j ((hmem i).mp hi) ((hmem j).mp hj))
    (fun i hi => ⟨fun x hx => hR ⟨_, ⟨i, (hmem i).mp hi, rfl⟩, hx⟩,
      fun x hx => hW ⟨_, ⟨i, (hmem i).mp hi, rfl⟩, hx⟩⟩) junk m hv
    fun acc' m' hdone _ hframe _ hpres => hk ⟨acc'⟩ m' (fun i he => hdone i ((hmem i).mpr he))
      (fun x hx => hframe x fun i hi hxi => hx ⟨_, ⟨i, (hmem i).mp hi, rfl⟩, hxi⟩) hpres

end Rules

end TVA
