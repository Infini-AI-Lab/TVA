import TVA.Proof.Buf
import TVA.Proof.Safe

/-!
# Rules: loads, stores, padding, atomics on owned elements

The proof rules for `TVA.Triton.Memory`.

* **Any addresses.** `safe_load` (the result is any tensor `LoadRes` allows), `safe_store`
  (lanes on distinct bytes: `StoreRes`), `safe_store_any` (lanes that may collide: every order of
  the writes).
* **Buffer elements.** Most accesses address elements of one buffer (`Buf.Addrs`,
  `TVA.Proof.Buf`): `safe_load_buf` gives each enabled lane its element, `safe_store_buf` leaves
  each stored element holding its value. Scalars: `safe_load1`, `safe_load1_buf`, `safe_store1`.
* **Padding** of block-pointer and descriptor loads: `safe_pad`, `Padding.exists_holds`.
* **Atomics on elements the program owns**: `safe_atomic_buf`, for an atomic whose result is
  unused (`Safe` requires it). Atomics on cells several programs update need the shared-atomics
  kit.

Footprints: a load's bytes must be in the read footprint `R`; a store's or an atomic's in the write
footprint `W`, and the rest of the program gets `W` without them (each byte is written once).

How this fits: model → `Safe` per program (these rules and the others of `TVA.Proof.Rules`) →
`correct_of_safe`.
-/
namespace TVA

open Prog

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
variable {s : Shape} {d : DType}

/-! ## Any addresses -/

/-- What `tl.load ptrs mask other` may return from memory `m`: each enabled lane holds its
element; a disabled lane holds `other` when it is given. -/
def LoadRes (m : Mem) (ptrs : Tensor s (Ptr d)) (mask : Option (Tensor s Bool))
    (other : Option (Tensor s d.Val)) (v : Tensor s d.Val) : Prop :=
  ∀ i, (∀ a, tl.addrs ptrs mask i = some a → m.read d a = some (v.get i)) ∧
    (tl.addrs ptrs mask i = none → ∀ o, other = some o → v.get i = o.get i)

/-- **`tl.load`**: every enabled lane's element can be read and lies in `R`; the rest must be safe
for every result `LoadRes` allows. -/
theorem safe_load {ptrs : Tensor s (Ptr d)} {mask : Option (Tensor s Bool)}
    {other : Option (Tensor s d.Val)} {k : Tensor s d.Val → Prog α}
    (hval : ∀ i a, tl.addrs ptrs mask i = some a → ∃ v, m.read d a = some v)
    (hR : accBytes d (tl.addrs ptrs mask) ⊆ R)
    (hk : ∀ v, LoadRes m ptrs mask other v → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs mask other >>= k) := by
  simp only [tl.load, bind_eq, Prog.bind]
  refine .load ?_ hR (fun v hok => hk _ ?_)
  · rintro ⟨i, a, ha, hnone⟩
    obtain ⟨v, hv⟩ := hval i a ha
    rw [hv] at hnone; cases hnone
  · intro i
    constructor
    · intro a ha
      have := hok i a ha
      cases mask with
      | none => cases other <;> exact this
      | some mm =>
        cases other with
        | none => exact this
        | some o =>
          have hm : mm.get i = true := by
            simp only [tl.addrs] at ha; split at ha
            · assumption
            · cases ha
          simpa [Tensor.where_, hm] using this
    · intro hnone o ho
      subst ho
      cases mask with
      | none => simp [tl.addrs] at hnone
      | some mm =>
        have hm : mm.get i = false := by
          simp only [tl.addrs] at hnone; split at hnone
          · cases hnone
          · simpa using ‹¬ mm.get i = true›
        simp [Tensor.where_, hm]

/-- What memory is after `tl.store ptrs vals mask` from `m`: each enabled lane's element holds its
value; nothing else changed. -/
def StoreRes (m m' : Mem) (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool)) : Prop :=
  (∀ i a, tl.addrs ptrs mask i = some a → m'.read d a = some (vals.get i)) ∧
    (∀ x, x ∉ accBytes d (tl.addrs ptrs mask) → m' x = m x)

/-- **`tl.store` to pairwise distinct elements**: every enabled lane can write, the bytes lie in
`W`, and the rest (with `W` minus those bytes) must be safe in the memory `StoreRes` describes. -/
theorem safe_store {ptrs : Tensor s (Ptr d)} {vals : Tensor s d.Val}
    {mask : Option (Tensor s Bool)} {k : Unit → Prog α}
    (hok : StoreOk m d (tl.addrs ptrs mask))
    (hdist : DistinctLanes d (tl.addrs ptrs mask))
    (hW : accBytes d (tl.addrs ptrs mask) ⊆ W)
    (hk : ∀ m', StoreRes m m' ptrs vals mask →
      Safe E R (W \ accBytes d (tl.addrs ptrs mask)) Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals mask >>= k) := by
  simp only [tl.store, bind_eq, Prog.bind]
  refine .store hok hW (fun order hperm => hk _ ⟨?_, ?_⟩)
  · intro i a ha
    exact writeAll_read hdist hok order (hperm.nodup_iff.mpr (enabled_nodup _)) i a
      (hperm.mem_iff.mpr (mem_enabled ha)) ha
  · intro x hx
    exact writeAll_frame _ _ _ _ hx

/-- **`tl.store` to any lanes** (two lanes may address the same element): the rest must be safe
after every order of the lanes' writes. -/
theorem safe_store_any {ptrs : Tensor s (Ptr d)} {vals : Tensor s d.Val}
    {mask : Option (Tensor s Bool)} {k : Unit → Prog α}
    (hok : StoreOk m d (tl.addrs ptrs mask))
    (hW : accBytes d (tl.addrs ptrs mask) ⊆ W)
    (hk : ∀ order, order.Perm (enabled (tl.addrs ptrs mask)) →
      Safe E R (W \ accBytes d (tl.addrs ptrs mask)) Q
        (writeAll m d (tl.addrs ptrs mask) vals order) (k ())) :
    Safe E R W Q m (tl.store ptrs vals mask >>= k) := by
  simp only [tl.store, bind_eq, Prog.bind]
  exact .store hok hW hk

/-! ## Scalars -/

/-- A scalar access touches the element at its address. -/
theorem accBytes_scalar (d : DType) (a : Int) :
    accBytes d (fun _ : Idx [] => some a) = Mem.span d a := by
  ext x
  constructor
  · rintro ⟨_, a', h, hx⟩; cases h; exact hx
  · intro hx; exact ⟨(), a, rfl, hx⟩

/-- **`tl.load(p)` of one element** that memory can read: the rest runs with that element. -/
theorem safe_load1 {p : Ptr d} {v0 : d.Val} (hv : m.read d p.addr = some v0)
    (hR : Mem.span d p.addr ⊆ R) {k : d.Val → Prog α} (hk : Safe E R W Q m (k v0)) :
    Safe E R W Q m (tl.load1 p >>= k) := by
  simp only [tl.load1, bind_eq, Prog.bind]
  refine .load ?_ (by rw [accBytes_scalar]; exact hR) fun v hok => ?_
  · rintro ⟨_, a, ha, hnone⟩
    cases ha
    rw [hv] at hnone; cases hnone
  · have : v.item = v0 := by
      have := hok () p.addr rfl
      rw [hv] at this
      exact (Option.some.inj this).symm
    show Safe E R W Q m (k v.item)
    rw [this]
    exact hk

/-- **`tl.load` of element `i` of a valid buffer.** -/
theorem safe_load1_buf (b : Buf d) {i : Nat} (hi : i < b.len) (hv : b.Valid m)
    (hR : Mem.span d (b.ptr i).addr ⊆ R) {k : d.Val → Prog α}
    (hk : ∀ v, m.read d (b.ptr i).addr = some v → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load1 (b.ptr i) >>= k) := by
  obtain ⟨v, hv'⟩ := b.read_some hv hi
  exact safe_load1 hv' hR (hk v hv')

/-- **`tl.store(p, v)` of one element**: the element can be written and lies in `W`; afterwards it
holds `v` and nothing else changed. -/
theorem safe_store1 {p : Ptr d} {v : d.Val} {k : Unit → Prog α}
    (hok : m.Writable d p.addr) (hW : Mem.span d p.addr ⊆ W)
    (hk : ∀ m', m'.read d p.addr = some v → (∀ x, x ∉ Mem.span d p.addr → m' x = m x) →
      Safe E R (W \ Mem.span d p.addr) Q m' (k ())) :
    Safe E R W Q m (tl.store1 p v >>= k) := by
  simp only [tl.store1, bind_eq, Prog.bind]
  have hok' : StoreOk m d (fun _ : Idx [] => some p.addr) := fun _ a h => by cases h; exact hok
  refine .store hok' (by rw [accBytes_scalar]; exact hW) fun order hperm => ?_
  rw [accBytes_scalar]
  refine hk _ ?_ ?_
  · exact writeAll_read (vals := Tensor.scalar v) (fun i j _ _ hij => absurd rfl hij) hok' order
      (hperm.nodup_iff.mpr (enabled_nodup _)) () p.addr
      (hperm.mem_iff.mpr (mem_enabled (a := p.addr) rfl)) rfl
  · intro x hx
    exact writeAll_frame _ _ _ _ (by rw [accBytes_scalar]; exact hx)

/-! ## Buffer elements -/

/-- **Load of buffer elements.** Each enabled lane gets its element; a disabled lane gets `other`
when it is given. -/
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
    cases ha; exact b.read_some hv (hin i he)
  · intro i he
    exact (hv' i).1 _ (hA.some he)
  · intro i he o ho
    exact (hv' i).2 (by rw [hA i, he]; rfl) o ho

/-- **Store to distinct buffer elements.** Afterwards each enabled lane's element holds its value,
nothing outside the elements changed, and every present byte is still present. The rest of the
program may not write the elements again. -/
theorem safe_store_buf (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Option (Tensor s Bool)}
    {vals : Tensor s d.Val} {g : Idx s → Nat} {en : Idx s → Bool}
    (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len)
    (hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j)
    (hv : b.Valid m) (hW : b.elems (Buf.img g en) ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m', (∀ i, en i = true → m'.read d (b.ptr (g i : Int)).addr = some (vals.get i)) →
      (∀ x, x ∉ b.elems (Buf.img g en) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems (Buf.img g en)) Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals mask >>= k) := by
  have hok : StoreOk m d (tl.addrs ptrs mask) := by
    intro i a ha
    rw [hA i] at ha
    split_ifs at ha with he
    cases ha; exact b.writable hv (hin i he)
  refine safe_store_any hok (by rw [hA.accBytes]; exact hW) (fun order hperm => ?_)
  rw [hA.accBytes]
  refine hk _ ?_ ?_ ?_
  · intro i he
    exact writeAll_read (hA.distinct hinj) hok order (hperm.nodup_iff.mpr (enabled_nodup _)) i _
      (hperm.mem_iff.mpr (mem_enabled (hA.some he))) (hA.some he)
  · intro x hx
    rw [← hA.accBytes] at hx
    exact writeAll_frame _ _ _ _ hx
  · intro x hx
    exact writeAll_present _ _ _ hx

/-! ## Padding -/

/-- A padding value exists: always for `undef` and `zero`, and for `nan` on float dtypes. Use it
for the first premise of `safe_pad`. -/
theorem Padding.exists_holds (p : Padding) (h : p = .nan → d.format?.isSome = true) :
    ∃ z : d.Val, p.Holds z := by
  cases p with
  | undef => exact ⟨default, trivial⟩
  | zero => exact ⟨d.ofBits 0, rfl⟩
  | nan =>
    have hd := h rfl
    cases d <;> first
      | exact ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
      | simp [DType.format?] at hd

/-- **Padding** (`tl.pad`): in-bounds lanes keep the loaded value, the others hold the padding. -/
theorem safe_pad {p : Padding} {inb : Idx s → Bool} {v : Tensor s d.Val}
    {k : Tensor s d.Val → Prog α} (hz : ∃ z : d.Val, p.Holds z)
    (hk : ∀ r : Tensor s d.Val, (∀ i, inb i = true → r.get i = v.get i) →
      (∀ i, inb i = false → p.Holds (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.pad p inb v >>= k) := by
  obtain ⟨z, hz⟩ := hz
  simp only [tl.pad, Prog.bind_eq, Prog.bind]
  refine .pick ⟨⟨fun i => if inb i then v.get i else z⟩, fun i => ?_⟩
    fun r hr => hk r (fun i hi => ?_) (fun i hi => ?_)
  · by_cases h : inb i = true <;> simp [h, hz]
  · have := hr i; rw [ite_eq_left hi] at this; exact this
  · have := hr i; rw [ite_eq_right (by simp [hi])] at this; exact this

/-! ## Atomics on owned elements -/

/-- After the read-modify-writes of some lanes, a continuation that ignores their old values does
not depend on the values collected before. -/
theorem tl.rmwSeq_bind_const {β : Type} (ptrs : Tensor s (Ptr d))
    (upd : Idx s → d.Val → d.Val → Prop)
    (sv : (i : Idx s) → (old : d.Val) → Solve (upd i old)) (sem : MemSem) (sc : Scope) (K : Prog β) :
    ∀ (L : List (Idx s)) (acc acc' : Idx s → d.Val),
      (tl.rmwSeq ptrs upd sv sem sc L acc).bind (fun _ => K) =
        (tl.rmwSeq ptrs upd sv sem sc L acc').bind (fun _ => K)
  | [], _, _ => rfl
  | i :: rest, acc, acc' => by
    simp only [tl.rmwSeq, Prog.bind]
    congr 1
    funext old
    exact tl.rmwSeq_bind_const ptrs upd sv sem sc K rest _ _

/-- The read-modify-writes of the lanes `L`, each on its own element of `b`, followed by a
continuation that ignores their old values. -/
private theorem safe_rmwSeq_buf (b : Buf d) {ptrs : Tensor s (Ptr d)}
    {upd : Idx s → d.Val → d.Val → Prop}
    {sv : (i : Idx s) → (old : d.Val) → Solve (upd i old)} {sem : MemSem} {sc : Scope} {g : Idx s → Nat}
    {β : Type} {Q' : β → Mem → Prop} {K : Prog β} (htot : ∀ i o, ∃ n, upd i o n) :
    ∀ (L : List (Idx s)), L.Nodup →
    (∀ i ∈ L, (ptrs.get i).addr = (b.ptr (g i : Int)).addr ∧ g i < b.len) →
    (∀ i ∈ L, ∀ j ∈ L, g i = g j → i = j) →
    ∀ (W : Set Int), (∀ i ∈ L, Mem.span d (b.ptr (g i : Int)).addr ⊆ W ∧
      Disjoint (Mem.span d (b.ptr (g i : Int)).addr) R) →
    ∀ (acc : Idx s → d.Val) (m₀ : Mem), b.Valid m₀ →
    (∀ m', (∀ i ∈ L, ∃ old n, m₀.read d (b.ptr (g i : Int)).addr = some old ∧ upd i old n ∧
        m'.read d (b.ptr (g i : Int)).addr = some n) →
      (∀ x, (∀ i ∈ L, x ∉ Mem.span d (b.ptr (g i : Int)).addr) → m' x = m₀ x) →
      b.Valid m' → (∀ x, m₀ x ≠ none → m' x ≠ none) →
      Safe E R (W \ {x | ∃ i ∈ L, x ∈ Mem.span d (b.ptr (g i : Int)).addr}) Q' m' K) →
    Safe E R W Q' m₀ ((tl.rmwSeq ptrs upd sv sem sc L acc).bind (fun _ => K))
  | [], _, _, _, W, _, acc, m₀, hv, hf => by
    simp only [tl.rmwSeq, Prog.ret_bind]
    have h := hf m₀ (fun _ h => absurd h List.not_mem_nil) (fun _ _ => rfl) hv (fun _ h => h)
    simpa using h
  | i :: rest, hnd, hL, hinj, W, hRW, acc, m₀, hv, hf => by
    have hi := hL i List.mem_cons_self
    have hrestL : ∀ j ∈ rest, (ptrs.get j).addr = (b.ptr (g j : Int)).addr ∧ g j < b.len :=
      fun j hj => hL j (List.mem_cons_of_mem _ hj)
    have hni : i ∉ rest := (List.nodup_cons.mp hnd).1
    -- the lanes of `rest` touch elements other than `i`'s
    have hdis : ∀ j ∈ rest, Disjoint (Mem.span d (b.ptr (g i : Int)).addr)
        (Mem.span d (b.ptr (g j : Int)).addr) :=
      fun j hj => b.span_disjoint fun e => hni (by
        have := hinj i List.mem_cons_self j (List.mem_cons_of_mem _ hj) (by exact_mod_cast e)
        exact this ▸ hj)
    simp only [tl.rmwSeq, Prog.bind]
    rw [hi.1]
    obtain ⟨o, ho⟩ := b.read_some hv hi.2
    obtain ⟨n, hn⟩ := htot i o
    refine .rmw ⟨o, ho, n, hn⟩ (hRW i List.mem_cons_self).1 (hRW i List.mem_cons_self).2
      (fun v w => tl.rmwSeq_bind_const ptrs upd sv sem sc K rest _ _) fun old new hold hnew => ?_
    have hw := b.writable hv hi.2
    have hv₁ : b.Valid (m₀.write d (b.ptr (g i : Int)).addr new) :=
      Buf.valid_of_agree hv fun x hx => Mem.write_present _ (hv.2.2.2 x hx)
    have hset : (W \ Mem.span d (b.ptr (g i : Int)).addr) \
        {x | ∃ j ∈ rest, x ∈ Mem.span d (b.ptr (g j : Int)).addr} =
        W \ {x | ∃ j ∈ i :: rest, x ∈ Mem.span d (b.ptr (g j : Int)).addr} := by
      ext x
      simp only [Set.mem_sdiff, Set.mem_ofPred_eq, List.mem_cons]
      constructor
      · rintro ⟨⟨hxW, hxi⟩, hxr⟩
        refine ⟨hxW, ?_⟩
        rintro ⟨j, rfl | hj, hx⟩
        · exact hxi hx
        · exact hxr ⟨j, hj, hx⟩
      · rintro ⟨hxW, hx⟩
        exact ⟨⟨hxW, fun h => hx ⟨i, Or.inl rfl, h⟩⟩, fun ⟨j, hj, h⟩ => hx ⟨j, Or.inr hj, h⟩⟩
    refine safe_rmwSeq_buf b htot rest (List.nodup_cons.mp hnd).2 hrestL
      (fun j hj j' hj' => hinj j (List.mem_cons_of_mem _ hj) j' (List.mem_cons_of_mem _ hj'))
      (W \ Mem.span d (b.ptr (g i : Int)).addr)
      (fun j hj => ⟨fun x hx => ⟨(hRW j (List.mem_cons_of_mem _ hj)).1 hx,
          fun hx' => Set.disjoint_left.mp (hdis j hj) hx' hx⟩, (hRW j (List.mem_cons_of_mem _ hj)).2⟩)
      _ _ hv₁
      fun m' hdone hframe hv' hpres => hset ▸ hf m' ?_ ?_ hv' ?_
    · intro j hj
      rcases List.mem_cons.mp hj with rfl | hj
      · refine ⟨old, new, hold, hnew, ?_⟩
        rw [Mem.read_congr (m₂ := m₀.write d (b.ptr (g j : Int)).addr new)
          fun x hx => hframe x fun j' hj' hx' => Set.disjoint_left.mp (hdis j' hj') hx hx']
        exact Mem.read_write_self hw new
      · obtain ⟨o', n', h1, h2, h3⟩ := hdone j hj
        exact ⟨o', n', by rw [← h1, Mem.read_write_other _ (hdis j hj)], h2, h3⟩
    · intro x hx
      rw [hframe x fun j hj => hx j (List.mem_cons_of_mem _ hj)]
      exact Mem.write_frame _ _ _ _ (hx i List.mem_cons_self)
    · intro x hx
      exact hpres x (Mem.write_present _ hx)

/-- **A tensor atomic on owned buffer elements, result unused** (`let _ ← tl.atomic_…`). Each
enabled lane updates its own element of `b` once, from the element's value before the atomic; the
rule gives each element's old and new value, and nothing outside the elements changes. The
elements are outside the read footprint, and the rest of the program may not touch them again.
The rule is about `tl.atomicWith`, which every `tl.atomic_*` is except `tl.atomic_add`: that is
`.env fun e => tl.atomicWith ptrs (fun i => tl.addUpd e d (vals.get i)) mask sem sc`, so unfold
it, apply `Safe.env`, then this rule with `upd := fun i => tl.addUpd E d (vals.get i)`. -/
theorem safe_atomic_buf (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Option (Tensor s Bool)}
    {upd : Idx s → d.Val → d.Val → Prop}
    {sv : (i : Idx s) → (old : d.Val) → Solve (upd i old)} {sem : MemSem} {sc : Scope} {g : Idx s → Nat}
    {en : Idx s → Bool}
    (hA : b.Addrs (tl.addrs ptrs mask) g en) (hin : ∀ i, en i = true → g i < b.len)
    (hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j)
    (htot : ∀ i o, ∃ n, upd i o n) (hv : b.Valid m)
    (hW : b.elems (Buf.img g en) ⊆ W) (hRd : Disjoint (b.elems (Buf.img g en)) R)
    {k : Prog α}
    (hk : ∀ m' : Mem,
      (∀ i, en i = true → ∃ old n, m.read d (b.ptr (g i : Int)).addr = some old ∧
        upd i old n ∧ m'.read d (b.ptr (g i : Int)).addr = some n) →
      (∀ x, x ∉ b.elems (Buf.img g en) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems (Buf.img g en)) Q m' k) :
    Safe E R W Q m (tl.atomicWith ptrs upd sv mask sem sc >>= fun _ => k) := by
  have hen : ∀ i, i ∈ enabled (tl.addrs ptrs mask) ↔ en i = true := by
    intro i
    simp only [enabled, List.mem_filter, Idx.mem_all, true_and]
    rw [hA i]; cases en i <;> simp
  have hptr : ∀ i, en i = true → (ptrs.get i).addr = (b.ptr (g i : Int)).addr := fun i he =>
    (tl.addrs_some (hA.some he)).symm
  simp only [tl.atomicWith, Prog.bind_eq, Prog.bind, Prog.bind_assoc, Prog.pure_eq]
  refine .pick ⟨fun _ => default, trivial⟩ fun junk _ => ?_
  refine .pick ⟨_, List.Perm.refl _⟩ fun order hperm => ?_
  have hmem : ∀ i, i ∈ order ↔ en i = true := fun i => (hperm.mem_iff).trans (hen i)
  refine safe_rmwSeq_buf b htot order (hperm.nodup_iff.mpr (enabled_nodup _))
    (fun i hi => ⟨hptr i ((hmem i).mp hi), hin i ((hmem i).mp hi)⟩)
    (fun i hi j hj => hinj i j ((hmem i).mp hi) ((hmem j).mp hj))
    W (fun i hi => ⟨fun x hx => hW ⟨_, ⟨i, (hmem i).mp hi, rfl⟩, hx⟩,
      Set.disjoint_left.mpr fun x hx => Set.disjoint_left.mp hRd ⟨_, ⟨i, (hmem i).mp hi, rfl⟩, hx⟩⟩)
    junk m hv fun m' hdone hframe _ hpres => ?_
  have hset : {x | ∃ i ∈ order, x ∈ Mem.span d (b.ptr (g i : Int)).addr} =
      b.elems (Buf.img g en) := by
    ext x
    constructor
    · rintro ⟨i, hi, hx⟩; exact ⟨_, ⟨i, (hmem i).mp hi, rfl⟩, hx⟩
    · rintro ⟨_, ⟨i, he, rfl⟩, hx⟩; exact ⟨i, (hmem i).mpr he, hx⟩
  rw [hset]
  exact hk m' (fun i he => hdone i ((hmem i).mpr he))
    (fun x hx => hframe x fun i hi hxi => hx ⟨_, ⟨i, (hmem i).mp hi, rfl⟩, hxi⟩) hpres

end TVA
